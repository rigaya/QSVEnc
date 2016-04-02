//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#include "qsv_tchar.h"
#include <math.h>
#include <iostream>
#include <mfxvideo++.h>
#include <emmintrin.h>
#include "qsv_osdep.h"
#include "qsv_event.h"
#include "qsv_log.h"
#include "qsv_output.h"

#include "mfxcommon.h"
#include "qsv_control.h"
#include "qsv_allocator.h"

#pragma warning( disable : 4748 )
CEncodeStatusInfo::CEncodeStatusInfo() {
    m_sData.nProcessedFramesNum = 0;
    m_sData.nWrittenBytes = 0;
    m_sData.nIDRCount = 0;
    m_sData.nICount = 0;
    m_sData.nPCount = 0;
    m_sData.nBCount = 0;
    m_sData.nIFrameSize = 0;
    m_sData.nPFrameSize = 0;
    m_sData.nBFrameSize = 0;
    m_sData.fEncodeFps = 0.0;
    m_sData.fBitrateKbps = 0.0;
    m_sData.fGPUClockTotal = 0.0;
    m_sData.fGPULoadPercentTotal = 0.0;
    m_sData.fMFXLoadPercentTotal = 0.0;
    m_sData.nGPUInfoCountFail = 0;
    m_sData.nGPUInfoCountSuccess = 0;
    m_nInputFrames = 0;
    m_nTotalOutFrames = 0;
    m_nOutputFPSRate = 0;
    m_nOutputFPSScale = 0;
    m_bStdErrWriteToConsole = true;
    m_bEncStarted = false;
    m_tmStart = std::chrono::system_clock::now();
    m_tmLastUpdate = m_tmStart;
    m_pause = FALSE;
}

CEncodeStatusInfo::~CEncodeStatusInfo() {
    m_pPerfMonitor.reset();
    m_pQSVLog.reset();
}

void CEncodeStatusInfo::Init(mfxU32 outputFPSRate, mfxU32 outputFPSScale, mfxU32 totalOutputFrames, shared_ptr<CQSVLog> pQSVLog, shared_ptr<CPerfMonitor> pPerfMonitor) {
    m_pause = FALSE;
    m_nOutputFPSRate = outputFPSRate;
    m_nOutputFPSScale = outputFPSScale;
    m_nTotalOutFrames = totalOutputFrames;
    m_pQSVLog = pQSVLog;
    m_pPerfMonitor = pPerfMonitor;
#if defined(_WIN32) || defined(_WIN64)
    DWORD mode = 0;
    m_bStdErrWriteToConsole = 0 != GetConsoleMode(GetStdHandle(STD_ERROR_HANDLE), &mode); //stderrの出力先がコンソールかどうか
#endif //#if defined(_WIN32) || defined(_WIN64)
}

void CEncodeStatusInfo::SetStart() {
    m_tmStart = std::chrono::system_clock::now();
    m_tmLastUpdate = m_tmStart;
    GetProcessTime(&m_sStartTime);
    m_bEncStarted = true;
}

CEncodingThread::CEncodingThread() {
    m_nFrameBuffer = 0;
    m_bthForceAbort = FALSE;
    m_bthSubAbort = FALSE;
    m_nFrameSet = 0;
    m_nFrameGet = 0;
    m_InputBuf = NULL;
    m_bInit = false;
}

CEncodingThread::~CEncodingThread() {
    Close();
}

mfxStatus CEncodingThread::Init(mfxU16 bufferSize) {
    Close();

    m_nFrameBuffer = bufferSize;
    if (nullptr == (m_InputBuf = (sInputBufSys *)_mm_malloc(m_nFrameBuffer * sizeof(sInputBufSys), 64)))
        return MFX_ERR_NULL_PTR;

    memset(m_InputBuf, 0, m_nFrameBuffer * sizeof(sInputBufSys));

    for (mfxU32 i = 0; i < m_nFrameBuffer; i++) {
        if (   NULL == (m_InputBuf[i].heInputDone  = CreateEvent(NULL, FALSE, FALSE, NULL))
            || NULL == (m_InputBuf[i].heSubStart   = CreateEvent(NULL, FALSE, FALSE, NULL))
            || NULL == (m_InputBuf[i].heInputStart = CreateEvent(NULL, FALSE, FALSE, NULL))) {
            return MFX_ERR_INVALID_HANDLE;
        }
    }
    m_bInit = true;
    m_bthForceAbort = FALSE;
    m_bthSubAbort = FALSE;
    return MFX_ERR_NONE;
}

mfxStatus CEncodingThread::RunEncFuncbyThread(void(*func)(void *prm), CQSVPipeline *pipeline, size_t threadAffinityMask) {
    if (!m_bInit) return MFX_ERR_NOT_INITIALIZED;

    m_thEncode = std::thread(func, pipeline);

    if (threadAffinityMask)
        SetThreadAffinityMask(m_thEncode.native_handle(), threadAffinityMask);

    return MFX_ERR_NONE;
}

mfxStatus CEncodingThread::RunSubFuncbyThread(void(*func)(void *prm), CQSVPipeline *pipeline, size_t threadAffinityMask) {
    if (!m_bInit) return MFX_ERR_NOT_INITIALIZED;

    m_thSub = std::thread(func, pipeline);

    if (threadAffinityMask)
        SetThreadAffinityMask(m_thSub.native_handle(), threadAffinityMask);

    return MFX_ERR_NONE;
}

//終了を待機する
mfxStatus CEncodingThread::WaitToFinish(mfxStatus sts, shared_ptr<CQSVLog> pQSVLog) {
    if (!m_bInit) return MFX_ERR_NOT_INITIALIZED;
    //最後のLoadNextFrameの結果をm_stsThreadにセットし、RunEncodeに知らせる
    m_stsThread = sts;
    //読み込み終了(MFX_ERR_MORE_DATA)ではなく、エラーや中断だった場合、
    //直ちに終了する
    if (sts != MFX_ERR_MORE_DATA) {
        pQSVLog->write(QSV_LOG_DEBUG, _T("WaitToFinish: Encode Aborted, putting abort flag on.\n"));
        m_bthForceAbort++; //m_bthForceAbort = TRUE;
        m_bthSubAbort++;   //m_bthSubAbort = TRUE;
        if (m_InputBuf) {
            pQSVLog->write(QSV_LOG_DEBUG, _T("WaitToFinish: Settings event on.\n"));
            for (mfxU32 i = 0; i < m_nFrameBuffer; i++) {
                SetEvent(m_InputBuf[i].heInputDone);
                SetEvent(m_InputBuf[i].heSubStart);
            }
        }
    }
    //RunEncodeの終了を待つ
    m_thEncode.join();
    pQSVLog->write(QSV_LOG_DEBUG, _T("WaitToFinish: Encode thread shut down.\n"));
    return MFX_ERR_NONE;
}

void CEncodingThread::Close() {
    if (m_thEncode.joinable()) {
        m_thEncode.join();
    }
    if (m_thSub.joinable()) {
        m_bthForceAbort++;
        for (mfxU32 i = 0; i < m_nFrameBuffer; i++)
            SetEvent(m_InputBuf[i].heSubStart);
        m_thSub.join();
    }
    if (m_InputBuf) {
        for (mfxU32 i = 0; i < m_nFrameBuffer; i++) {
            if (m_InputBuf[i].heInputDone)
                CloseEvent(m_InputBuf[i].heInputDone);
            if (m_InputBuf[i].heSubStart)
                CloseEvent(m_InputBuf[i].heSubStart);
            if (m_InputBuf[i].heInputStart)
                CloseEvent(m_InputBuf[i].heInputStart);
        }
        _mm_free(m_InputBuf);
        m_InputBuf = NULL;
    }
    m_nFrameBuffer = 0;
    m_nFrameSet = 0;
    m_nFrameGet = 0;
    m_bthSubAbort = FALSE;
    m_bthForceAbort = FALSE;
    m_stsThread = MFX_ERR_NONE;
    m_bInit = false;
}
