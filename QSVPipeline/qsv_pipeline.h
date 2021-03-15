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

#include "mfxmvc.h"
#include "mfxvideo.h"
#include "mfxvideo++.h"
#include "mfxplugin.h"
#include "mfxplugin++.h"

#ifndef BUILD_AUO
#pragma comment(lib, "libmfx_vs2015.lib")
#endif

#include "vpp_plugins.h"
#include "rgy_perf_monitor.h"
#include "rgy_bitstream.h"
#include "qsv_plugin.h"
#include "rgy_input.h"
#include "rgy_output.h"
#include "rgy_opencl.h"
#include "qsv_task.h"
#include "qsv_vpp_mfx.h"
#include "qsv_pipeline_ctrl.h"
#include "qsv_control.h"

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

struct mfxParamSet {
    mfxVideoParam vidprm;
    mfxExtCodingOption cop;
    mfxExtCodingOption2 cop2;
    mfxExtCodingOption3 cop3;
    mfxExtHEVCParam hevc;
};

struct VppVilterBlock {
    VppFilterType type;
    std::unique_ptr<QSVVppMfx> vppmfx;
    std::vector<std::unique_ptr<RGYFilter>> vppcl;

    VppVilterBlock(std::unique_ptr<QSVVppMfx>& filter) : type(VppFilterType::FILTER_MFX), vppmfx(std::move(filter)), vppcl() {};
    VppVilterBlock(std::vector<std::unique_ptr<RGYFilter>>& filter) : type(), vppmfx(), vppcl(std::move(filter)) {};
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
    virtual RGY_ERR CheckCurrentVideoParam(TCHAR *buf = NULL, mfxU32 bufSize = 0);

    virtual void SetAbortFlagPointer(bool *abort);

    virtual RGY_ERR GetEncodeStatusData(EncodeStatusData *data);
    virtual void GetEncodeLibInfo(mfxVersion *ver, bool *hardware);
    virtual const TCHAR *GetInputMessage();
    virtual MemType GetMemType();

    virtual void PrintMes(int log_level, const TCHAR *format, ...);
    shared_ptr<RGYLog> m_pQSVLog;

    virtual RGY_ERR RunEncode2();
    static void RunEncThreadLauncher(void *pParam);
    bool CompareParam(const mfxParamSet& prmA, const mfxParamSet& prmB);
protected:
    mfxVersion m_mfxVer;
    shared_ptr<EncodeStatus> m_pStatus;
    shared_ptr<CPerfMonitor> m_pPerfMonitor;
    CEncodingThread m_EncThread;

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

    CQSVTaskControl m_TaskPool; // 廃止予定
    int m_nAsyncDepth;
    RGYAVSync m_nAVSyncMode;
    RGYTimestamp m_outputTimestamp;

    mfxInitParam m_InitParam;
    mfxExtBuffer *m_pInitParamExtBuf[1];
    mfxExtThreadsParam m_ThreadsParam;

    mfxExtVideoSignalInfo m_VideoSignalInfo;
    mfxExtChromaLocInfo m_chromalocInfo;
    mfxExtCodingOption m_CodingOption;
    mfxExtCodingOption2 m_CodingOption2;
    mfxExtCodingOption3 m_CodingOption3;
    mfxExtVP8CodingOption m_ExtVP8CodingOption;
    mfxExtHEVCParam m_ExtHEVCParam;
    MFXVideoSession m_mfxSession;
    unique_ptr<MFXVideoDECODE> m_pmfxDEC;
    unique_ptr<MFXVideoENCODE> m_pmfxENC;
    std::vector<std::unique_ptr<QSVVppMfx>> m_mfxVPP;

    unique_ptr<CSessionPlugins> m_SessionPlugins;

    sTrimParam m_trimParam;

    mfxVideoParam m_mfxDecParams;
    mfxVideoParam m_mfxEncParams;

    mfxParamSet m_prmSetIn;

    vector<mfxExtBuffer*> m_DecExtParams;
    vector<mfxExtBuffer*> m_EncExtParams;

    mfxExtDecVideoProcessing m_DecVidProc;
#if ENABLE_AVSW_READER
    vector<unique_ptr<AVChapter>> m_Chapters;
#endif
    std::unique_ptr<RGYTimecode> m_timecode;
    std::unique_ptr<HEVCHDRSei> m_HDRSei;

    unique_ptr<QSVAllocator> m_pMFXAllocator;
    unique_ptr<mfxAllocatorParams> m_pmfxAllocatorParams;
    int m_nMFXThreads;
    MemType m_memType;
    bool m_bExternalAlloc;
    uint32_t m_nProcSpeedLimit;

    bool *m_pAbortByUser;
    unique_ptr<std::remove_pointer<HANDLE>::type, handle_deleter> m_heAbort;

    RGYBitstream m_DecInputBitstream;

    mfxStatus GetNextFrame(mfxFrameSurface1 **pSurface);
    mfxStatus SetNextSurface(mfxFrameSurface1 *pSurface);

    // for disabling VPP algorithms
    //mfxExtVPPDoNotUse m_VppDoNotUse;

    std::shared_ptr<RGYOpenCLContext> m_cl;
    std::vector<VppVilterBlock> m_vpFilters;

    std::shared_ptr<CQSVHWDevice> m_hwdev;
    std::vector<std::unique_ptr<PipelineTask>> m_pipelineTasks;

    virtual RGY_ERR InitSessionInitParam(int threads, int priority);
    virtual RGY_ERR InitLog(sInputParams *pParams);
    virtual RGY_ERR InitPerfMonitor(const sInputParams *pParams);
    virtual RGY_ERR InitInput(sInputParams *pParams);
    virtual RGY_ERR InitChapters(const sInputParams *inputParam);
    virtual RGY_ERR InitFilters(sInputParams *inputParam);
    virtual std::vector<VppType> InitFiltersCreateVppList(sInputParams *inputParam, const bool cropRequired, const bool resizeRequired);
    virtual std::pair<RGY_ERR, std::unique_ptr<QSVVppMfx>> AddFilterMFX(
        FrameInfo& frameInfo, VideoVUIInfo& vuiIn, rgy_rational<int>& fps,
        const VppType vppType, const sVppParams *params, sInputCrop *crop, const int blockSize);
    virtual std::pair<RGY_ERR, std::unique_ptr<RGYFilter>> AddFilterOpenCL(
        FrameInfo& frameInfo, rgy_rational<int>& fps,  VppType vppType, RGYParamVpp *params);
    virtual RGY_ERR InitOutput(sInputParams *pParams);
    virtual RGY_ERR InitMfxDecParams(sInputParams *pInParams);
    virtual RGY_ERR InitMfxEncodeParams(sInputParams *pParams);
    virtual RGY_ERR InitMfxDec();
    virtual RGY_ERR InitMfxEncode();
    virtual RGY_ERR InitSession(bool useHWLib, uint32_t memType);
    int getEncoderBitdepth(const sInputParams *pParams);
    RGY_CSP getEncoderCsp(const sInputParams *pParams, int *pShift = nullptr);

    virtual RGY_ERR readChapterFile(tstring chapfile);

    virtual RGY_ERR initOpenCL();
    virtual RGY_ERR CreateHWDevice();
    virtual RGY_ERR CreateAllocator();
    virtual void DeleteAllocator();

    virtual void DeleteHWDevice();

    virtual RGY_ERR AllocFrames();

    virtual RGY_ERR AllocateSufficientBuffer(mfxBitstream* pBS);

    virtual mfxStatus GetFreeTask(QSVTask **ppTask);
    virtual mfxStatus SynchronizeFirstTask();

    RGY_ERR CreatePipeline();
    std::pair<RGY_ERR, std::unique_ptr<QSVVideoParam>> GetOutputVideoInfo();

    RGY_ERR CheckParamList(int value, const CX_DESC *list, const char *param_name);
    int clamp_param_int(int value, int low, int high, const TCHAR *param_name);
    int logTemporarilyIgnoreErrorMes();
};

#endif // __PIPELINE_ENCODE_H__
