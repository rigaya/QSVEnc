// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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

#ifndef __PIPELINE_ENCODE_H__
#define __PIPELINE_ENCODE_H__

#include "rgy_version.h"
#include "rgy_osdep.h"
#include "qsv_util.h"
#include "qsv_prm.h"
#include "rgy_thread.h"
#include "rgy_event.h"
#include "rgy_log.h"
#include "rgy_filter.h"

#include "qsv_hw_device.h"

#include "qsv_allocator.h"

#ifndef BUILD_AUO
#ifdef NDEBUG
#pragma comment(lib, "vpl.lib")
#else
#pragma comment(lib, "vpld.lib")
#endif
#endif

#include "rgy_perf_monitor.h"
#include "rgy_bitstream.h"
#include "rgy_input.h"
#include "rgy_output.h"
#include "rgy_opencl.h"
#include "rgy_dummy_load.h"
#include "rgy_device_usage.h"
#include "rgy_parallel_enc.h"
#include "qsv_vpp_mfx.h"
#include "qsv_mfx_dec.h"
#include "qsv_pipeline_ctrl.h"
#include "qsv_session.h"
#include "qsv_device.h"

#include <vector>
#include <memory>
#include <string>
#include <iostream>

struct AVChapter;
class RGYTimecode;

enum {
    MFX_PRM_EX_SCENE_CHANGE = 0x01,
    MFX_PRM_EX_VQP          = 0x02,
    MFX_PRM_EX_DEINT_NORMAL = 0x04,
    MFX_PRM_EX_DEINT_BOB    = 0x08
};
enum {
    SC_FIELDFLAG_INVALID_ALL  = 0xffffffff,
    SC_FIELDFLAG_INVALID_LOW  = 0x0000ffff,
    SC_FIELDFLAG_INVALID_HIGH = 0xffff0000,
};

const uint32_t QSV_PTS_SORT_SIZE = 16u;

class CQSVPipeline
{
public:
    CQSVPipeline();
    virtual ~CQSVPipeline();

    virtual RGY_ERR CheckParam(sInputParams *pParams);
    virtual RGY_ERR Init(sInputParams *pParams);
    virtual RGY_ERR Run();
    virtual void Close();
    virtual RGY_ERR ResetDevice();
    virtual RGY_ERR ResetMFXComponents(sInputParams* pParams);
    virtual RGY_ERR CheckCurrentVideoParam(TCHAR *buf = NULL, mfxU32 bufSize = 0);

    virtual void SetAbortFlagPointer(bool *abort);

    virtual RGY_ERR GetEncodeStatusData(EncodeStatusData *data);
    virtual void GetEncodeLibInfo(mfxVersion *ver, bool *hardware);
    virtual const TCHAR *GetInputMessage();
    virtual MemType GetMemType();

    virtual void PrintMes(RGYLogLevel log_level, const TCHAR *format, ...);
    shared_ptr<RGYLog> m_pQSVLog;

    virtual RGY_ERR RunEncode2();
    bool CompareParam(const QSVVideoParam& prmA, const QSVVideoParam& prmB);
protected:
    mfxVersion m_mfxVer;
    std::unique_ptr<QSVDevice> m_device;
    std::vector<tstring> m_devNames;
    shared_ptr<EncodeStatus> m_pStatus;
    shared_ptr<CPerfMonitor> m_pPerfMonitor;
    std::unique_ptr<RGYDeviceUsage> m_deviceUsage;
    std::unique_ptr<RGYParallelEnc> m_parallelEnc;

    int m_encWidth;
    int m_encHeight;
    RGY_PICSTRUCT m_encPicstruct;
    rgy_rational<int> m_inputFps;
    rgy_rational<int> m_encFps;
    rgy_rational<int> m_outputTimebase;
    VideoVUIInfo m_encVUI;

    bool m_bTimerPeriodTuning; //timeBeginPeriodを使ったかどうか記憶する

    vector<shared_ptr<RGYOutput>> m_pFileWriterListAudio;
    shared_ptr<RGYOutput> m_pFileWriter;
    vector<shared_ptr<RGYInput>> m_AudioReaders;
    shared_ptr<RGYInput> m_pFileReader;
    std::unique_ptr<RGYPoolAVPacket> m_poolPkt;
    std::unique_ptr<RGYPoolAVFrame> m_poolFrame;

    int m_nAsyncDepth;
    RGYAVSync m_nAVSyncMode;
    bool m_timestampPassThrough;

    QSVVideoParam m_encParams;
    std::unique_ptr<QSVMfxDec> m_mfxDEC;
    std::unique_ptr<MFXVideoENCODE> m_pmfxENC;
    std::vector<std::unique_ptr<QSVVppMfx>> m_mfxVPP;
    std::vector<QSVRCParam> m_dynamicRC;
    QSVEncFeatures m_encFeatures;

    sTrimParam m_trimParam;

    QSVVideoParam m_prmSetIn;

#if ENABLE_AVSW_READER
    vector<unique_ptr<AVChapter>> m_Chapters;
#endif
    std::unique_ptr<RGYTimecode> m_timecode;
    std::unique_ptr<RGYHDRMetadata> m_hdrseiIn;
    std::unique_ptr<RGYHDRMetadata> m_hdrseiOut;
    std::unique_ptr<RGYHDR10Plus> m_hdr10plus;
    bool                          m_hdr10plusMetadataCopy;
    std::unique_ptr<DOVIRpu>      m_dovirpu;
    bool                          m_dovirpuMetadataCopy;
    RGYDOVIProfile                m_doviProfile;
    std::unique_ptr<RGYTimestamp> m_encTimestamp;

    MFXVideoSession2Params m_sessionParams;
    uint32_t m_nProcSpeedLimit;
    bool m_taskPerfMonitor;
    std::unique_ptr<RGYDummyLoadCL> m_dummyLoad;

    bool *m_pAbortByUser;
    unique_ptr<std::remove_pointer<HANDLE>::type, handle_deleter> m_heAbort;

    RGYBitstream m_DecInputBitstream;

    std::shared_ptr<RGYOpenCLContext> m_cl;
    std::vector<VppType> m_vppFilterList;
    std::vector<VppVilterBlock> m_vpFilters;
    unique_ptr<RGYFilterSsim> m_videoQualityMetric;

    std::vector<std::unique_ptr<PipelineTask>> m_pipelineTasks;

    virtual RGY_ERR InitLog(sInputParams *pParams);
    virtual RGY_ERR InitPerfMonitor(const sInputParams *pParams);
    virtual RGY_ERR InitParallelEncode(sInputParams *inputParam, const int maxEncoders);
    virtual RGY_ERR InitInput(sInputParams *pParams, DeviceCodecCsp& HWDecCodecCsp);
    virtual RGY_ERR InitChapters(const sInputParams *inputParam);
    virtual RGY_ERR InitFilters(sInputParams *inputParam);
    virtual std::vector<VppType> InitFiltersCreateVppList(const sInputParams *inputParam, const bool cspConvRequired, const bool cropRequired, const RGY_VPP_RESIZE_TYPE resizeRequired);
    virtual std::pair<RGY_ERR, std::unique_ptr<QSVVppMfx>> AddFilterMFX(
        RGYFrameInfo& frameInfo, rgy_rational<int>& fps,
        const VppType vppType, const sVppParams *params, const RGY_CSP outCsp, const int outBitdepth, const sInputCrop *crop, const std::pair<int, int> resize, const int blockSize);
    virtual RGY_ERR AddFilterOpenCL(std::vector<std::unique_ptr<RGYFilter>>& clfilters,
        RGYFrameInfo& inputFrame, const VppType vppType, const sInputParams *params, const sInputCrop *crop, const std::pair<int, int> resize, VideoVUIInfo& vuiInfo);
    virtual RGY_ERR createOpenCLCopyFilterForPreVideoMetric();
    virtual RGY_ERR InitOutput(sInputParams *pParams);
    virtual RGY_ERR InitMfxDecParams();
    std::pair<RGY_ERR, QSVEncFeatures> CheckMFXRCMode(QSVRCParam& rcParam, sInputParams *pParams, const int codecMaxQP);
    virtual RGY_ERR InitMfxEncodeParams(sInputParams *pParams, std::vector<std::unique_ptr<QSVDevice>>& devList);
    virtual RGY_ERR InitPowerThrottoling(sInputParams *pParams);
    virtual RGY_ERR InitAvoidIdleClock(const sInputParams *pParams);
    virtual RGY_ERR InitMfxDec();
    virtual RGY_ERR InitMfxVpp();
    virtual RGY_ERR InitMfxEncode();
    RGY_ERR checkGPUListByEncoder(const sInputParams *inputParam, std::vector<std::unique_ptr<QSVDevice>>& deviceList);
    RGY_ERR deviceAutoSelect(const sInputParams *inputParam, std::vector<std::unique_ptr<QSVDevice>>& deviceList, const RGYDeviceUsageLockManager *lock);
    virtual RGY_ERR InitSession(const sInputParams *inputParam, std::vector<std::unique_ptr<QSVDevice>>& deviceList);
    virtual RGY_ERR InitVideoQualityMetric(sInputParams *pParams);
    void applyInputVUIToColorspaceParams(sInputParams *inputParam);
    bool preferD3D11Mode(const sInputParams *pParams);
    RGY_CSP getEncoderCsp(const sInputParams *pParams, int *pShift = nullptr) const;
    bool VppAfsRffAware() const;
    DeviceCodecCsp getHWDecCodecCsp(const bool skipHWDecodeCheck, std::vector<std::unique_ptr<QSVDevice>>& devList);

    virtual RGY_ERR readChapterFile(tstring chapfile);

    virtual bool CPUGenOpenCLSupported(const QSV_CPU_GEN cpu_gen);
    virtual RGY_ERR InitOpenCL(const bool enableOpenCL, const int openCLBuildThreads, const bool checkVppPerformance);

    virtual RGY_ERR AllocFrames();

    virtual RGY_ERR AllocateSufficientBuffer(mfxBitstream* pBS);

    RGY_ERR SetPerfMonitorThreadHandles();
    RGY_ERR CreatePipeline(const sInputParams* prm);
    std::pair<RGY_ERR, std::unique_ptr<QSVVideoParam>> GetOutputVideoInfo();

    RGY_ERR CheckParamList(int value, const CX_DESC *list, const char *param_name);
    int clamp_param_int(int value, int low, int high, const TCHAR *param_name);
    RGYParamLogLevel logTemporarilyIgnoreErrorMes();
};

#endif // __PIPELINE_ENCODE_H__
