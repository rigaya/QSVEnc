// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#pragma once

#include "rgy_filter_cl.h"
#if ENCODER_VCEENC
#include "vce_util.h"
#include "Factory.h"
#include "Trace.h"
#endif
#if ENCODER_QSV
#include "qsv_util.h"
#include "rgy_queue.h"
class QSVMfxDec;
class PipelineTaskMFXDecode;
struct RGYBitstream;
#endif
#if ENCODER_MPP
#include "mpp_util.h"
#endif
#include <array>
#include <thread>
#include <deque>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#if ENABLE_VMAF
#include "rgy_libvmaf.h"
#endif
#if ENABLE_LIBVSHIP
#include "rgy_libvship.h"
#endif

class RGYFilterParamSsim : public RGYFilterParam {
public:
    RGYVideoQualityMetric metric;
    int deviceId;
    int bitDepth;
    VideoInfo input;
    rgy_rational<int> streamtimebase;
    RGYParamThread threadParam;
#if ENCODER_VCEENC
    amf::AMFFactory *factory;
    amf::AMFTrace *trace;
    amf::AMFContextPtr context;
#endif
#if ENCODER_QSV
    std::unique_ptr<QSVMfxDec> mfxDEC;
#endif

    RGYFilterParamSsim();
    virtual ~RGYFilterParamSsim();

    tstring print() const;
};

class RGYFilterSsim : public RGYFilter {
public:
    RGYFilterSsim(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterSsim();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
    virtual RGY_ERR initDecode(const RGYBitstream *bitstream);
    bool decodeStarted() { return m_decodeStarted; }
    RGY_ERR finish();
    RGY_ERR metricStatus() const;
    virtual void showResult();
    RGY_ERR thread_func(RGYParamThread threadParam);
    RGY_ERR thread_func_compare_frames();
    RGY_ERR compare_frames();

    virtual RGY_ERR addBitstream(const RGYBitstream *bitstream);
protected:
    RGY_ERR init_cl_resources();
    void close_cl_resources();
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;
    RGY_ERR build_kernel(const RGY_CSP csp);
    RGY_ERR calc_ssim_plane(const RGYFrameInfo *p0, const RGYFrameInfo *p1, std::unique_ptr<RGYCLBuf> &tmp, RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR calc_ssim_frame(const RGYFrameInfo *p0, const RGYFrameInfo *p1);
    RGY_ERR calc_psnr_plane(const RGYFrameInfo *p0, const RGYFrameInfo *p1, std::unique_ptr<RGYCLBuf> &tmp, RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR calc_psnr_frame(const RGYFrameInfo *p0, const RGYFrameInfo *p1);
    RGY_ERR calc_ssim_psnr(const RGYFrameInfo *p0, const RGYFrameInfo *p1);
    RGY_ERR init_metric_worker();
    RGY_ERR submit_metric_frame(RGYCLFrame *reference, RGYCLFrame *distorted, const std::vector<RGYOpenCLEvent> &referenceWaitEvents, const std::vector<RGYOpenCLEvent> &distortedWaitEvents);
    RGY_ERR metric_worker();
    RGY_ERR finish_metric_worker();
    void close_metric_resources();

    bool m_decodeStarted; //デコードが開始したか
    int m_deviceId;       //SSIM計算で使用するCUDA device ID

    //スレッド関連
    std::thread m_thread; //スレッド本体
    std::mutex m_mtx;     //m_input, m_unused操作用のロック
    bool m_abort;         //スレッド中断用
    bool m_dec_flush;

    int m_inputOriginal;
    int m_inputEnc;
    std::deque<std::unique_ptr<RGYCLFrame>> m_input;  //使用中のフレームバッファ(オリジナルフレーム格納用)
    std::deque<std::unique_ptr<RGYCLFrame>> m_unused; //使っていないフレームバッファ(オリジナルフレーム格納用)
    std::deque<RGYOpenCLEvent> m_inputReady;           //原画像のコピー・変換完了イベント
#if ENCODER_VCEENC
    amf::AMFTrace *m_trace;
    amf::AMFFactory *m_factory;
    amf::AMFContextPtr m_context;
    amf::AMFComponentPtr m_decoder;
#endif
#if ENCODER_QSV
    RGYQueueMPMP<RGYBitstream> m_encBitstream;
    RGYQueueMPMP<RGYBitstream> m_encBitstreamUnused;
    std::unique_ptr<QSVMfxDec> m_mfxDEC;
    std::unique_ptr<PipelineTaskMFXDecode> m_taskDec;
    std::unordered_map<mfxFrameSurface1 *, std::unique_ptr<RGYCLFrameInterop>> m_surfVppInInterop;
#endif

    std::unique_ptr<RGYFilterCspCrop> m_cropOrg;      // NV12->YV12変換用
    std::unique_ptr<RGYFilterCspCrop> m_cropDec;      // NV12->YV12変換用
    std::unique_ptr<RGYCLFrame> m_decFrameCopy; //デコード後にcrop(NV12->YV12変換)したフレームの格納場所
    std::array<std::unique_ptr<RGYCLBuf>, 3> m_tmpSsim; //評価結果を返すための一時バッファ
    std::array<std::unique_ptr<RGYCLBuf>, 3> m_tmpPsnr; //評価結果を返すための一時バッファ
    RGYOpenCLEvent m_cropEvent; //デコードしたフレームがcrop(NV12->YV12変換)し終わったかを示すイベント
    RGYOpenCLQueue m_queueCrop; //デコードしたフレームをcrop(NV12->YV12変換)するstream
    std::array<RGYOpenCLQueue, 3> m_queueCalcSsim; //評価計算を行うstream
    std::array<RGYOpenCLQueue, 3> m_queueCalcPsnr; //評価計算を行うstream
    std::array<double, 3> m_planeCoef;      // 評価結果に関する YUVの重み
    std::array<double, 3> m_ssimTotalPlane; // 評価結果の累積値 YUV
    double m_ssimTotal;                     // 評価結果の累積値 All
    std::array<double, 3> m_psnrTotalPlane; // 評価結果の累積値 YUV
    double m_psnrTotal;                     // 評価結果の累積値 All
    int m_frames;                           // 評価したフレーム数
    bool m_finishCalled;
    RGY_ERR m_finishResult;

#if ENABLE_VMAF || ENABLE_LIBVSHIP
    struct MetricHostFrame {
        RGY_CSP csp;
        int width;
        int height;
        std::array<int, 3> planeWidth;
        std::array<int, 3> planeHeight;
        std::array<int64_t, 3> pitch;
        std::array<std::vector<uint8_t>, 3> plane;
        MetricHostFrame() : csp(RGY_CSP_NA), width(0), height(0), planeWidth(), planeHeight(), pitch(), plane() {}
    };
    struct MetricHostPair {
        MetricHostFrame reference;
        MetricHostFrame distorted;
        int index;
        MetricHostPair() : reference(), distorted(), index(-1) {}
    };
    RGY_ERR copy_metric_frame(RGYCLFrame *source, const std::vector<RGYOpenCLEvent> &waitEvents, MetricHostFrame& destination);
    RGY_ERR process_metric_pair(const MetricHostPair& pair);
#if ENABLE_VMAF
    RGY_ERR init_vmaf_metric();
    RGY_ERR finish_vmaf_metric();
    RGY_ERR process_vmaf_metric(const MetricHostPair& pair);
#endif
#if ENABLE_LIBVSHIP
    RGY_ERR init_vship_metric();
    RGY_ERR process_vship_metric(const MetricHostPair& pair);
    RGY_ERR finish_vship_metric();
#endif
    std::array<MetricHostPair, 2> m_metricSlots;
    std::deque<int> m_metricReady;
    std::deque<int> m_metricFree;
    std::thread m_metricThread;
    mutable std::mutex m_metricMutex;
    std::condition_variable m_metricReadyCv;
    std::condition_variable m_metricFreeCv;
    std::condition_variable m_metricInitCv;
    bool m_metricInputFin;
    bool m_metricStop;
    bool m_metricWorkerReady;
    bool m_metricWorkerDone;
    bool m_metricFinishCalled;
    int m_metricNextIndex;
    RGY_ERR m_metricError;
    RGY_ERR m_metricFinishResult;
#if ENABLE_VMAF
    RGYLibVMAFLoader m_libvmaf;
    VmafContext *m_vmafContext;
    VmafModel *m_vmafModel;
    VmafModelCollection *m_vmafModelCollection;
    double m_vmafScore;
    int m_vmafFrames;
#endif
#if ENABLE_LIBVSHIP
    RGYLibVshipLoader m_libvship;
    Vship_SSIMU2Handler m_vshipSsimu2;
    Vship_ButteraugliHandler m_vshipButteraugli;
    Vship_CVVDPHandler m_vshipCvvdp;
    bool m_vshipSsimu2Initialized;
    bool m_vshipButteraugliInitialized;
    bool m_vshipCvvdpInitialized;
    double m_vshipSsimu2Total;
    int m_vshipSsimu2Frames;
    std::vector<double> m_vshipSsimu2Scores;
    double m_vshipButteraugliNormQ;
    double m_vshipButteraugliNorm3;
    double m_vshipButteraugliNormInf;
    int m_vshipButteraugliFrames;
    double m_vshipCvvdpScore;
    int m_vshipCvvdpFrames;
#endif
#endif

    RGYOpenCLProgramAsync m_kernel;
};
