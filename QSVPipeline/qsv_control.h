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

#ifndef __QSV_CONTROL_H__
#define __QSV_CONTROL_H__

#include "rgy_tchar.h"
#include <stdio.h>
#include <math.h>
#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <atomic>
#include <thread>
#include <mfxvideo++.h>
#include "mfxstructures.h"
#include "mfxvideo.h"
#include "mfxjpeg.h"
#include "qsv_prm.h"
#include "qsv_util.h"
#include "rgy_log.h"
#include "cpu_info.h"
#include "gpuz_info.h"
#include "rgy_perf_monitor.h"
#include "rgy_err.h"

using std::chrono::duration_cast;
using std::shared_ptr;
class CQSVPipeline;

const uint32_t MSDK_DEC_WAIT_INTERVAL = 60000;
const uint32_t MSDK_ENC_WAIT_INTERVAL = 10000;
const uint32_t MSDK_VPP_WAIT_INTERVAL = 60000;
const uint32_t MSDK_WAIT_INTERVAL = MSDK_DEC_WAIT_INTERVAL+3*MSDK_VPP_WAIT_INTERVAL+MSDK_ENC_WAIT_INTERVAL; // an estimate for the longest pipeline we have in samples

const uint32_t MSDK_INVALID_SURF_IDX = 0xFFFF;

typedef struct {
    RGYFrame* pFrameSurface;
    HANDLE heInputStart;
    HANDLE heInputDone;
    uint8_t reserved[64-(sizeof(RGYFrame*)+sizeof(HANDLE)*2)];
} sInputBufSys;

class CEncodingThread {
public:
    CEncodingThread();
    ~CEncodingThread();

    mfxStatus Init(int bufferSize);
    void Close();
    //終了を待機する
    mfxStatus WaitToFinish(mfxStatus sts, shared_ptr<RGYLog> pQSVLog);
    mfxStatus RunEncFuncbyThread(void(*func)(void *prm), CQSVPipeline *pipeline, size_t threadAffinityMask);

    std::thread& GetHandleEncThread() {
        return m_thEncode;
    }

    std::atomic_int m_bthForceAbort;
    sInputBufSys *m_InputBuf;
    uint32_t m_nFrameSet;
    uint32_t m_nFrameGet;
    mfxStatus m_stsThread;
    int  m_nFrameBuffer;
protected:
    std::thread m_thEncode;
    bool m_bInit;
};

#endif //__QSV_CONTROL_H__
