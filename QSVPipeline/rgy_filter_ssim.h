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

#include "rgy_filter.h"
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
#include <array>
#include <thread>
#include <deque>
#include <mutex>

class RGYFilterParamSsim : public RGYFilterParam {
public:
    RGYVideoQualityMetric metric;
    int deviceId;
    int bitDepth;
    VideoInfo input;
    rgy_rational<int> streamtimebase;
    RGYThreadAffinity threadAffinityCompare;
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
    virtual void showResult();
    RGY_ERR thread_func(RGYThreadAffinity threadAffinity);
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

    bool m_decodeStarted; //デコードが開始したか
    int m_deviceId;       //SSIM計算で使用するCUDA device ID

    //スレッド関連
    std::thread m_thread; //スレッド本体
    std::mutex m_mtx;     //m_input, m_unused操作用のロック
    bool m_abort;         //スレッド中断用

    int m_inputOriginal;
    int m_inputEnc;
    std::deque<std::unique_ptr<RGYCLFrame>> m_input;  //使用中のフレームバッファ(オリジナルフレーム格納用)
    std::deque<std::unique_ptr<RGYCLFrame>> m_unused; //使っていないフレームバッファ(オリジナルフレーム格納用)
#if ENCODER_VCEENC
    amf::AMFTrace *m_trace;
    amf::AMFFactory *m_factory;
    amf::AMFContextPtr m_context;
    amf::AMFComponentPtr m_decoder;
#endif
#if ENCODER_QSV
    RGYQueueSPSP<RGYBitstream> m_encBitstream;
    RGYQueueSPSP<RGYBitstream> m_encBitstreamUnused;
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

    RGYOpenCLProgramAsync m_kernel;
};
