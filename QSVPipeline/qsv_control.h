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
    HANDLE heSubStart;
    HANDLE heInputDone;
    std::atomic<uint32_t> frameFlag;
    std::atomic_int AQP[2];
    mfxU8 reserved[64-(sizeof(RGYFrame*)+sizeof(HANDLE)*3+sizeof(std::atomic<uint32_t>)+sizeof(std::atomic_int)*2)];
} sInputBufSys;

typedef struct {
    int frameCountI;
    int frameCountP;
    int frameCountB;
    int sumQPI;
    int sumQPP;
    int sumQPB;
} sFrameTypeInfo;

class CQSVFrameTypeSimulation
{
public:
    CQSVFrameTypeSimulation() {
        i_frame = 0;
        BFrames = 0;
        GOPSize = 1;
        openGOP = false;
        maxIdrInterval = 0;
    }
    void Init(int _GOPSize, int _BFrames, int _QPI, int _QPP, int _QPB, bool _openGOP, double frameRate) {
        GOPSize = (std::max)(_GOPSize, 1);
        BFrames = (std::max)(_BFrames, 0);
        QPI = _QPI;
        QPP = _QPP;
        QPB = _QPB;
        i_frame = 0;
        i_from_last_idr = 0;
        openGOP = _openGOP;
        maxIdrInterval = (int)(frameRate + 0.5) * 20;
        RGY_MEMSET_ZERO(m_info);
    }
    ~CQSVFrameTypeSimulation() {
    }
    mfxU32 GetFrameType(bool I_Insert) {
        mfxU32 ret;
        if (I_Insert || (GOPSize && i_frame % GOPSize == 0)) {
            i_frame = 0;
        }
        if (i_frame == 0) {
            ret = MFX_FRAMETYPE_I | MFX_FRAMETYPE_REF;
            if (!openGOP || i_from_last_idr >= maxIdrInterval || 0 == i_from_last_idr) {
                i_from_last_idr = 0;
                ret |= MFX_FRAMETYPE_IDR;
            }
        } else if ((i_frame - 1) % (BFrames + 1) == BFrames) {
            ret = MFX_FRAMETYPE_P | MFX_FRAMETYPE_REF;
        } else {
            ret = MFX_FRAMETYPE_B;
        }
        return ret;
    }
    void ToNextFrame() {
        i_frame++;
        i_from_last_idr++;
    }
    int CurrentQP(bool I_Insert, int qp_offset) {
        mfxU32 frameType = GetFrameType(I_Insert);
        int qp;
        if (frameType & MFX_FRAMETYPE_I) {
            qp = QPI;
            m_info.sumQPI += qp;
            m_info.frameCountI++;
        } else if (frameType & MFX_FRAMETYPE_P) {
            qp = clamp(QPP + qp_offset, 0, 51);
            m_info.sumQPP += qp;
            m_info.frameCountP++;
        } else {
            qp = clamp(QPB + qp_offset, 0, 51);
            m_info.sumQPB += qp;
            m_info.frameCountB++;
        }
        return qp;
    }
    void getFrameInfo(sFrameTypeInfo *info) {
        memcpy(info, &m_info, sizeof(info[0]));
    }
private:
    int i_frame;
    int i_from_last_idr;

    int GOPSize;
    int BFrames;

    int QPI;
    int QPP;
    int QPB;

    bool openGOP;
    int maxIdrInterval;

    sFrameTypeInfo m_info;
};

class CEncodingThread {
public:
    CEncodingThread();
    ~CEncodingThread();

    mfxStatus Init(mfxU16 bufferSize);
    void Close();
    //終了を待機する
    mfxStatus WaitToFinish(mfxStatus sts, shared_ptr<RGYLog> pQSVLog);
    mfxStatus RunEncFuncbyThread(void(*func)(void *prm), CQSVPipeline *pipeline, size_t threadAffinityMask);
    mfxStatus RunSubFuncbyThread(void(*func)(void *prm), CQSVPipeline *pipeline, size_t threadAffinityMask);

    std::thread& GetHandleEncThread() {
        return m_thEncode;
    }
    std::thread& GetHandleSubThread() {
        return m_thSub;
    }

    std::atomic_int m_bthForceAbort;
    std::atomic_int m_bthSubAbort;
    sInputBufSys *m_InputBuf;
    mfxU32 m_nFrameSet;
    mfxU32 m_nFrameGet;
    mfxStatus m_stsThread;
    mfxU16  m_nFrameBuffer;
protected:
    std::thread m_thEncode;
    std::thread m_thSub;
    bool m_bInit;
};

#endif //__QSV_CONTROL_H__
