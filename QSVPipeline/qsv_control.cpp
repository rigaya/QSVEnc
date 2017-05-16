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

#include "rgy_tchar.h"
#include <math.h>
#include <iostream>
#include <mfxvideo++.h>
#include <emmintrin.h>
#include "rgy_osdep.h"
#include "qsv_event.h"
#include "rgy_log.h"
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

void CEncodeStatusInfo::Init(mfxU32 outputFPSRate, mfxU32 outputFPSScale, mfxU32 totalOutputFrames, shared_ptr<RGYLog> pQSVLog, shared_ptr<CPerfMonitor> pPerfMonitor) {
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

RGY_ERR CEncodingThread::Init(mfxU16 bufferSize) {
    Close();

    m_nFrameBuffer = bufferSize;
    if (nullptr == (m_InputBuf = (sInputBufSys *)_mm_malloc(m_nFrameBuffer * sizeof(sInputBufSys), 64)))
        return RGY_ERR_NULL_PTR;

    memset(m_InputBuf, 0, m_nFrameBuffer * sizeof(sInputBufSys));

    for (mfxU32 i = 0; i < m_nFrameBuffer; i++) {
        if (   NULL == (m_InputBuf[i].heInputDone  = CreateEvent(NULL, FALSE, FALSE, NULL))
            || NULL == (m_InputBuf[i].heSubStart   = CreateEvent(NULL, FALSE, FALSE, NULL))
            || NULL == (m_InputBuf[i].heInputStart = CreateEvent(NULL, FALSE, FALSE, NULL))) {
            return RGY_ERR_INVALID_HANDLE;
        }
    }
    m_bInit = true;
    m_bthForceAbort = FALSE;
    m_bthSubAbort = FALSE;
    return RGY_ERR_NONE;
}

RGY_ERR CEncodingThread::RunEncFuncbyThread(void(*func)(void *prm), CQSVPipeline *pipeline, size_t threadAffinityMask) {
    if (!m_bInit) return RGY_ERR_NOT_INITIALIZED;

    m_thEncode = std::thread(func, pipeline);

    if (threadAffinityMask)
        SetThreadAffinityMask(m_thEncode.native_handle(), threadAffinityMask);

    return RGY_ERR_NONE;
}

RGY_ERR CEncodingThread::RunSubFuncbyThread(void(*func)(void *prm), CQSVPipeline *pipeline, size_t threadAffinityMask) {
    if (!m_bInit) return RGY_ERR_NOT_INITIALIZED;

    m_thSub = std::thread(func, pipeline);

    if (threadAffinityMask)
        SetThreadAffinityMask(m_thSub.native_handle(), threadAffinityMask);

    return RGY_ERR_NONE;
}

//終了を待機する
RGY_ERR CEncodingThread::WaitToFinish(RGY_ERR sts, shared_ptr<RGYLog> pQSVLog) {
    if (!m_bInit) return RGY_ERR_NOT_INITIALIZED;
    //最後のLoadNextFrameの結果をm_stsThreadにセットし、RunEncodeに知らせる
    m_stsThread = sts;
    //読み込み終了(RGY_ERR_MORE_DATA)ではなく、エラーや中断だった場合、
    //直ちに終了する
    if (sts != RGY_ERR_MORE_DATA) {
        pQSVLog->write(RGY_LOG_DEBUG, _T("WaitToFinish: Encode Aborted, putting abort flag on.\n"));
        m_bthForceAbort++; //m_bthForceAbort = TRUE;
        m_bthSubAbort++;   //m_bthSubAbort = TRUE;
        if (m_InputBuf) {
            pQSVLog->write(RGY_LOG_DEBUG, _T("WaitToFinish: Settings event on.\n"));
            for (mfxU32 i = 0; i < m_nFrameBuffer; i++) {
                SetEvent(m_InputBuf[i].heInputDone);
                SetEvent(m_InputBuf[i].heSubStart);
            }
        }
    }
    //RunEncodeの終了を待つ
    m_thEncode.join();
    pQSVLog->write(RGY_LOG_DEBUG, _T("WaitToFinish: Encode thread shut down.\n"));
    return RGY_ERR_NONE;
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
    m_stsThread = RGY_ERR_NONE;
    m_bInit = false;
}
