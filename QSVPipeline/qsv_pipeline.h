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

#include "qsv_hw_device.h"

#include "qsv_allocator.h"

#include "mfxmvc.h"
#include "mfxvideo.h"
#include "mfxvideo++.h"
#include "mfxplugin.h"
#include "mfxplugin++.h"

#ifndef BUILD_AUO
#pragma comment(lib, "libmfx.lib")
#endif

#include "vpp_plugins.h"
#include "rgy_perf_monitor.h"
#include "qsv_plugin.h"
#include "rgy_input.h"
#include "rgy_output.h"
#include "qsv_task.h"
#include "qsv_control.h"

#include <vector>
#include <memory>
#include <string>
#include <iostream>

struct AVChapter;

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

const uint32_t QSV_PTS_SORT_SIZE = 16u;

class CQSVPipeline
{
public:
    CQSVPipeline();
    virtual ~CQSVPipeline();

    virtual mfxStatus CheckParam(sInputParams *pParams);
    virtual mfxStatus Init(sInputParams *pParams);
    virtual mfxStatus Run();
    virtual mfxStatus Run(size_t SubThreadAffinityMask);
    virtual void Close();
    virtual mfxStatus ResetMFXComponents(sInputParams* pParams);
    virtual mfxStatus ResetDevice();
    virtual mfxStatus CheckCurrentVideoParam(TCHAR *buf = NULL, mfxU32 bufSize = 0);

    virtual void SetAbortFlagPointer(bool *abort);

    virtual mfxStatus GetEncodeStatusData(EncodeStatusData *data);
    virtual void GetEncodeLibInfo(mfxVersion *ver, bool *hardware);
    virtual const TCHAR *GetInputMessage();
    virtual MemType GetMemType();

    virtual void PrintMes(int log_level, const TCHAR *format, ...);
    shared_ptr<RGYLog> m_pQSVLog;

    virtual mfxStatus RunEncode();
    static void RunEncThreadLauncher(void *pParam);
    bool CompareParam(const mfxParamSet& prmA, const mfxParamSet& prmB);
protected:
    mfxVersion m_mfxVer;
    shared_ptr<EncodeStatus> m_pEncSatusInfo;
    shared_ptr<CPerfMonitor> m_pPerfMonitor;
    CEncodingThread m_EncThread;

    bool m_bTimerPeriodTuning; //timeBeginPeriodを使ったかどうか記憶する

    vector<shared_ptr<RGYOutput>> m_pFileWriterListAudio;
    shared_ptr<RGYOutput> m_pFileWriter;
    vector<shared_ptr<RGYInput>> m_AudioReaders;
    shared_ptr<RGYInput> m_pFileReader;

    CQSVTaskControl m_TaskPool;
    mfxU16 m_nAsyncDepth;
    RGYAVSync m_nAVSyncMode;

    mfxInitParam m_InitParam;
    mfxExtBuffer *m_pInitParamExtBuf[1];
    mfxExtThreadsParam m_ThreadsParam;

    mfxExtVideoSignalInfo m_VideoSignalInfo;
    mfxExtCodingOption m_CodingOption;
    mfxExtCodingOption2 m_CodingOption2;
    mfxExtCodingOption3 m_CodingOption3;
    mfxExtVP8CodingOption m_ExtVP8CodingOption;
    mfxExtHEVCParam m_ExtHEVCParam;
    MFXVideoSession m_mfxSession;
    unique_ptr<MFXVideoDECODE> m_pmfxDEC;
    unique_ptr<MFXVideoENCODE> m_pmfxENC;
    unique_ptr<MFXVideoVPP>    m_pmfxVPP;

    unique_ptr<CSessionPlugins> m_SessionPlugins;
    vector<unique_ptr<CVPPPlugin>> m_VppPrePlugins;
    vector<unique_ptr<CVPPPlugin>> m_VppPostPlugins;

    sTrimParam *m_pTrimParam;

    mfxVideoParam m_mfxDecParams;
    mfxVideoParam m_mfxEncParams;
    mfxVideoParam m_mfxVppParams;

    mfxParamSet m_prmSetIn;
    
    unique_ptr<MFXVideoUSER>  m_pUserModule;

    vector<mfxExtBuffer*> m_DecExtParams;
    vector<mfxExtBuffer*> m_EncExtParams;
    vector<mfxExtBuffer*> m_VppExtParams;
    tstring VppExtMes;

    mfxExtDecVideoProcessing m_DecVidProc;
    mfxExtVPPDoNotUse m_VppDoNotUse;
    mfxExtVPPDoNotUse m_VppDoUse;
    mfxExtVPPDenoise m_ExtDenoise;
    mfxExtVPPDetail m_ExtDetail;
    mfxExtVPPDeinterlacing m_ExtDeinterlacing;
    mfxExtVPPFrameRateConversion m_ExtFrameRateConv;
    mfxExtVPPRotation m_ExtRotate;
    mfxExtVPPVideoSignalInfo m_ExtVppVSI;
    mfxExtVPPImageStab m_ExtImageStab;
    mfxExtVPPMirroring m_ExtMirror;
    mfxExtVPPScaling m_ExtScaling;
    vector<mfxU32> m_VppDoNotUseList;
    vector<mfxU32> m_VppDoUseList;
#if ENABLE_AVSW_READER
    vector<unique_ptr<AVChapter>> m_AVChapterFromFile;
#endif

    unique_ptr<QSVAllocator> m_pMFXAllocator;
    unique_ptr<mfxAllocatorParams> m_pmfxAllocatorParams;
    int m_nMFXThreads;
    MemType m_memType;
    bool m_bd3dAlloc;
    bool m_bExternalAlloc;
    uint32_t m_nProcSpeedLimit;

    bool *m_pAbortByUser;
    unique_ptr<std::remove_pointer<HANDLE>::type, handle_deleter> m_heAbort;

    RGYBitstream m_DecInputBitstream;

    vector<mfxFrameSurface1> m_pEncSurfaces; //enc input用のフレーム (vpp output, decoder output)
    vector<mfxFrameSurface1> m_pVppSurfaces; //vpp input用のフレーム (decoder output)
    vector<mfxFrameSurface1> m_pDecSurfaces; //dec input用のフレーム
    mfxFrameAllocResponse m_EncResponse;  //enc用 memory allocation response
    mfxFrameAllocResponse m_VppResponse;  //vpp用 memory allocation response
    mfxFrameAllocResponse m_DecResponse;  //dec用 memory allocation response

    mfxStatus GetNextFrame(mfxFrameSurface1 **pSurface);
    mfxStatus SetNextSurface(mfxFrameSurface1 *pSurface);

    // for disabling VPP algorithms
    //mfxExtVPPDoNotUse m_VppDoNotUse;

    shared_ptr<CQSVHWDevice> m_hwdev;

    virtual mfxStatus InitSessionInitParam(mfxU16 threads, mfxU16 priority);
    virtual mfxStatus InitLog(sInputParams *pParams);
    virtual mfxStatus InitInput(sInputParams *pParams);
    virtual mfxStatus InitOutput(sInputParams *pParams);
    virtual mfxStatus InitMfxDecParams(sInputParams *pInParams);
    virtual mfxStatus InitMfxEncParams(sInputParams *pParams);
    virtual mfxStatus InitMfxVppParams(sInputParams *pParams);
    virtual mfxStatus InitVppPrePlugins(sInputParams *pParams);
    virtual mfxStatus InitVppPostPlugins(sInputParams *pParams);
    virtual mfxStatus InitSession(bool useHWLib, mfxU16 memType);
    virtual RGY_CSP EncoderCsp(const sInputParams *pParams, int *pShift);
    //virtual void InitVppExtParam();
    virtual mfxStatus CreateVppExtBuffers(sInputParams *pParams);

    virtual mfxStatus readChapterFile(tstring chapfile);

    virtual mfxStatus AllocAndInitVppDoNotUse();
    virtual void FreeVppDoNotUse();

    virtual mfxStatus CreateAllocator();
    virtual void DeleteAllocator();

    virtual mfxStatus CreateHWDevice();
    virtual void DeleteHWDevice();

    virtual mfxStatus AllocFrames();
    virtual void DeleteFrames();

    virtual mfxStatus AllocateSufficientBuffer(mfxBitstream* pBS);

    virtual mfxStatus GetFreeTask(QSVTask **ppTask);
    virtual mfxStatus SynchronizeFirstTask();

    mfxStatus CheckParamList(int value, const CX_DESC *list, const char *param_name);
    int clamp_param_int(int value, int low, int high, const TCHAR *param_name);
};

#endif // __PIPELINE_ENCODE_H__
