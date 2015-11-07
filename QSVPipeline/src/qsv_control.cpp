/* ////////////////////////////////////////////////////////////////////////////// */
/*
//
//              INTEL CORPORATION PROPRIETARY INFORMATION
//  This software is supplied under the terms of a license  agreement or
//  nondisclosure agreement with Intel Corporation and may not be copied
//  or disclosed except in  accordance  with the terms of that agreement.
//        Copyright (c) 2005-2011 Intel Corporation. All Rights Reserved.
//
//
*/
#include "qsv_tchar.h"
#include <math.h>
#include <iostream>
#include <emmintrin.h>
#include "qsv_osdep.h"

#include "sample_defs.h"
#include "sample_utils.h"
#include "mfxcommon.h"
#include "qsv_control.h"

#pragma warning( disable : 4748 )
CEncodeStatusInfo::CEncodeStatusInfo()
{
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
    m_nInputFrames = 0;
    m_nTotalOutFrames = 0;
    m_nOutputFPSRate = 0;
    m_nOutputFPSScale = 0;
    m_bStdErrWriteToConsole = true;
    m_bEncStarted = false;
    m_tmStart = std::chrono::system_clock::now();
}

CEncodeStatusInfo::~CEncodeStatusInfo() {
    m_pQSVLog.reset();
}

void CEncodeStatusInfo::Init(mfxU32 outputFPSRate, mfxU32 outputFPSScale, mfxU32 totalOutputFrames, shared_ptr<CQSVLog> pQSVLog) {
    m_nOutputFPSRate = outputFPSRate;
    m_nOutputFPSScale = outputFPSScale;
    m_nTotalOutFrames = totalOutputFrames;
    m_pQSVLog = pQSVLog;
#if defined(_WIN32) || defined(_WIN64)
    DWORD mode = 0;
    m_bStdErrWriteToConsole = 0 != GetConsoleMode(GetStdHandle(STD_ERROR_HANDLE), &mode); //stderrの出力先がコンソールかどうか
#endif //#if defined(_WIN32) || defined(_WIN64)
}

void CEncodeStatusInfo::SetStart() {
    m_tmStart = std::chrono::system_clock::now();
    GetProcessTime(&m_sStartTime);
    m_bEncStarted = true;
}

CEncodingThread::CEncodingThread()
{
    m_nFrameBuffer = 0;
    m_bthForceAbort = FALSE;
    m_bthSubAbort = FALSE;
    m_nFrameSet = 0;
    m_nFrameGet = 0;
    m_InputBuf = NULL;
    m_bInit = false;
}

CEncodingThread::~CEncodingThread()
{
    Close();
}

mfxStatus CEncodingThread::Init(mfxU16 bufferSize) {
    Close();

    m_nFrameBuffer = bufferSize;
    m_InputBuf = (sInputBufSys *)_mm_malloc(m_nFrameBuffer * sizeof(sInputBufSys), 64);
    MSDK_CHECK_POINTER(m_InputBuf, MFX_ERR_NULL_PTR);
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

mfxStatus CEncodingThread::RunEncFuncbyThread(void(*func)(void *prm), CEncodingPipeline *pipeline, size_t threadAffinityMask) {
    MSDK_CHECK_ERROR(m_bInit, false, MFX_ERR_NOT_INITIALIZED);

    m_thEncode = std::thread(func, pipeline);

    if (threadAffinityMask)
        SetThreadAffinityMask(m_thEncode.native_handle(), threadAffinityMask);

    return MFX_ERR_NONE;
}

mfxStatus CEncodingThread::RunSubFuncbyThread(void(*func)(void *prm), CEncodingPipeline *pipeline, size_t threadAffinityMask) {
    MSDK_CHECK_ERROR(m_bInit, false, MFX_ERR_NOT_INITIALIZED);

    m_thSub = std::thread(func, pipeline);

    if (threadAffinityMask)
        SetThreadAffinityMask(m_thSub.native_handle(), threadAffinityMask);

    return MFX_ERR_NONE;
}

//終了を待機する
mfxStatus CEncodingThread::WaitToFinish(mfxStatus sts, shared_ptr<CQSVLog> pQSVLog) {
    MSDK_CHECK_ERROR(m_bInit, false, MFX_ERR_NOT_INITIALIZED);
    //最後のLoadNextFrameの結果をm_stsThreadにセットし、RunEncodeに知らせる
    m_stsThread = sts;
    //読み込み終了(MFX_ERR_MORE_DATA)ではなく、エラーや中断だった場合、
    //直ちに終了する
    if (sts != MFX_ERR_MORE_DATA) {
        (*pQSVLog)(QSV_LOG_DEBUG, _T("WaitToFinish: Encode Aborted, putting abort flag on.\n"));
        m_bthForceAbort++; //m_bthForceAbort = TRUE;
        m_bthSubAbort++;   //m_bthSubAbort = TRUE;
        if (m_InputBuf) {
            (*pQSVLog)(QSV_LOG_DEBUG, _T("WaitToFinish: Settings event on.\n"));
            for (mfxU32 i = 0; i < m_nFrameBuffer; i++) {
                SetEvent(m_InputBuf[i].heInputDone);
                SetEvent(m_InputBuf[i].heSubStart);
            }
        }
    }
    //RunEncodeの終了を待つ
    m_thEncode.join();
    (*pQSVLog)(QSV_LOG_DEBUG, _T("WaitToFinish: Encode thread shut down.\n"));
    return MFX_ERR_NONE;
}

void CEncodingThread::Close()
{
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
