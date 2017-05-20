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
#include "rgy_event.h"
#include "rgy_log.h"
#include "rgy_output.h"

#include "mfxcommon.h"
#include "qsv_control.h"
#include "qsv_allocator.h"

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
mfxStatus CEncodingThread::WaitToFinish(mfxStatus sts, shared_ptr<RGYLog> pQSVLog) {
    if (!m_bInit) return MFX_ERR_NOT_INITIALIZED;
    //最後のLoadNextFrameの結果をm_stsThreadにセットし、RunEncodeに知らせる
    m_stsThread = sts;
    //読み込み終了(MFX_ERR_MORE_DATA)ではなく、エラーや中断だった場合、
    //直ちに終了する
    if (sts != MFX_ERR_MORE_DATA) {
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
