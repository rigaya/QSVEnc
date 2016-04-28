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

#ifndef __QSV_TASK_H__
#define __QSV_TASK_H__

#include "qsv_tchar.h"
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
#include "qsv_log.h"
#include "qsv_output.h"
#include "cpu_info.h"
#include "gpuz_info.h"
#include "qsv_allocator.h"
#include "qsv_thread.h"

static inline int GetFreeSurface(mfxFrameSurface1 *pSurfacesPool, int nPoolSize) {
    for (mfxU32 j = 0; j < MSDK_WAIT_INTERVAL; j++) {
        for (mfxU16 i = 0; i < nPoolSize; i++) {
            if (0 == pSurfacesPool[i].Data.Locked)
                return i;
        }
        sleep_hybrid(j);
    }
    return MSDK_INVALID_SURF_IDX;
}

static inline mfxU16 GetFreeSurfaceIndex(mfxFrameSurface1 *pSurfacesPool, mfxU16 nPoolSize, mfxU16 step) {
    if (pSurfacesPool) {
        for (mfxU16 i = 0; i < nPoolSize; i = (mfxU16)(i + step), pSurfacesPool += step) {
            if (0 == pSurfacesPool[0].Data.Locked) {
                return i;
            }
        }
    }
    return MSDK_INVALID_SURF_IDX;
}

struct QSVTask {
    mfxBitstream mfxBS;
    mfxFrameSurface1 *mfxSurf;
    mfxSyncPoint encSyncPoint;
    vector<mfxSyncPoint> vppSyncPoint;
    shared_ptr<CQSVOut> pWriter;
    QSVAllocator *pmfxAllocator;

    QSVTask();

    mfxStatus WriteBitstream() {
        if (!pWriter || pWriter->getOutType() == OUT_TYPE_NONE) {
            return MFX_ERR_NOT_INITIALIZED;
        }

        mfxStatus sts = MFX_ERR_NONE;
        if (pWriter->getOutType() == OUT_TYPE_BITSTREAM) {
            sts = pWriter->WriteNextFrame(&mfxBS);
        } else {
            if (mfxSurf->Data.MemId) {
                sts = pmfxAllocator->Lock(pmfxAllocator->pthis, mfxSurf->Data.MemId, &(mfxSurf->Data));
                if (sts < MFX_ERR_NONE) {
                    return sts;
                }
            }

            sts = pWriter->WriteNextFrame(mfxSurf);

            if (mfxSurf->Data.MemId) {
                pmfxAllocator->Unlock(pmfxAllocator->pthis, mfxSurf->Data.MemId, &(mfxSurf->Data));
            }

            //最終で加算したLockをここで減算する
            mfxSurf->Data.Locked--;
        }
        return sts;
    }

    mfxStatus Clear() {
        encSyncPoint = 0;
        mfxSurf = nullptr;

        mfxBS.DataOffset = 0;
        mfxBS.DataLength = 0;

        vppSyncPoint.clear();

        return MFX_ERR_NONE;
    }
    mfxStatus Init(shared_ptr<CQSVOut> pTaskWriter, uint32_t nBufferSize, QSVAllocator *pAllocator = nullptr);
    mfxStatus Close();
};

class CQSVTaskControl {
public:
    CQSVTaskControl();
    virtual ~CQSVTaskControl();

    virtual mfxStatus Init(MFXVideoSession *pmfxSession, QSVAllocator *pAllocator, shared_ptr<CQSVOut> pTaskWriter, uint32_t nPoolSize, uint32_t nBufferSize);

    mfxStatus GetFreeTask(QSVTask **ppTask) {
        if (ppTask == nullptr) {
            return MFX_ERR_NULL_PTR;
        }

        if (m_pTasks.size()) {
            for (uint32_t i = 0; i < m_nPoolSize; i++) {
                auto pTask = &m_pTasks[(m_nTaskBufferStart + i) % m_nPoolSize];
                if (NULL == pTask->encSyncPoint) {
                    *ppTask = pTask;
                    return MFX_ERR_NONE;
                }
            }
        }
        return MFX_ERR_NOT_FOUND;
    }

    virtual mfxStatus SynchronizeFirstTask();
    virtual void Close();

protected:
    vector<QSVTask> m_pTasks;
    uint32_t m_nPoolSize;
    uint32_t m_nTaskBufferStart;

    MFXVideoSession *m_pmfxSession;
};

#endif //__QSV_TASK_H__
