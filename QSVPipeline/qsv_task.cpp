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
#include "qsv_allocator.h"
#include "qsv_task.h"

QSVTask::QSVTask() :
    mfxSurf(nullptr),
    encSyncPoint(0),
    vppSyncPoint(),
    pWriter(),
    pmfxAllocator(nullptr) {
    QSV_MEMSET_ZERO(mfxBS);
}

mfxStatus QSVTask::Init(shared_ptr<CQSVOut> pTaskWriter, uint32_t nBufferSize, QSVAllocator *pAllocator) {
    Close();

    pWriter = pTaskWriter;

    mfxStatus sts = Clear();
    if (sts < MFX_ERR_NONE) return sts;

    if (pWriter->getOutType() == OUT_TYPE_BITSTREAM) {
        if (nBufferSize == 0) {
            return MFX_ERR_INCOMPATIBLE_VIDEO_PARAM;
        }
        if (MFX_ERR_NONE != (sts = mfxBitstreamInit(&mfxBS, nBufferSize))) {
            mfxBitstreamClear(&mfxBS);
            return sts;
        }
    } else {
        //フレーム出力時には、Allocatorも必要
        if (pAllocator == nullptr) {
            return MFX_ERR_NULL_PTR;
        }
        pmfxAllocator = pAllocator;
    }

    return sts;
}

mfxStatus QSVTask::Close() {
    mfxBitstreamClear(&mfxBS);
    encSyncPoint = 0;
    pWriter.reset();
    pmfxAllocator = nullptr;
    mfxSurf = nullptr;
    vppSyncPoint.clear();

    return MFX_ERR_NONE;
}

CQSVTaskControl::CQSVTaskControl() :
    m_pTasks(),
    m_nPoolSize(0),
    m_nTaskBufferStart(0),
    m_pmfxSession(nullptr) {
}

CQSVTaskControl::~CQSVTaskControl() {
    Close();
}

mfxStatus CQSVTaskControl::Init(MFXVideoSession *pmfxSession, QSVAllocator *pAllocator, shared_ptr<CQSVOut> pTaskWriter, uint32_t nPoolSize, uint32_t nBufferSize) {
    if (nPoolSize == 0) {
        return MFX_ERR_INCOMPATIBLE_VIDEO_PARAM;
    }
    auto outputType = pTaskWriter->getOutType();
    if (outputType == OUT_TYPE_SURFACE) {
        //フレーム出力時には、Allocatorも必要
        if (pAllocator == nullptr) {
            return MFX_ERR_NULL_PTR;
        }
    } else if (outputType == OUT_TYPE_BITSTREAM) {
        if (nBufferSize == 0) {
            return MFX_ERR_UNDEFINED_BEHAVIOR;
        }
    } else {
        return MFX_ERR_UNSUPPORTED;
    }

    m_pmfxSession = pmfxSession;
    m_nPoolSize = nPoolSize;
    m_pTasks.resize(m_nPoolSize);

    for (uint32_t i = 0; i < m_nPoolSize; i++) {
        mfxStatus sts = m_pTasks[i].Init(pTaskWriter, nBufferSize, pAllocator);
        if (sts < MFX_ERR_NONE) {
            return sts;
        }
    }

    return MFX_ERR_NONE;
}

mfxStatus CQSVTaskControl::SynchronizeFirstTask() {
    if (m_pTasks[m_nTaskBufferStart].encSyncPoint == NULL) {
        return MFX_ERR_NOT_FOUND; //タスクバッファにもうタスクはない
    }

    mfxStatus sts = m_pmfxSession->SyncOperation(m_pTasks[m_nTaskBufferStart].encSyncPoint, MSDK_WAIT_INTERVAL);

    if (sts == MFX_ERR_NONE) {
        if (MFX_ERR_NONE > (sts = m_pTasks[m_nTaskBufferStart].WriteBitstream())) {
            return sts;
        }

        if (MFX_ERR_NONE > (sts = m_pTasks[m_nTaskBufferStart].Clear())) {
            return sts;
        }

        for (uint32_t i = 0; i < m_nPoolSize; i++) {
            m_nTaskBufferStart = (m_nTaskBufferStart + 1) % m_nPoolSize;
            if (m_pTasks[m_nTaskBufferStart].encSyncPoint != NULL) {
                break;
            }
        }
    } else if (sts == MFX_ERR_ABORTED) {
        for (auto syncp : m_pTasks[m_nTaskBufferStart].vppSyncPoint) {
            auto vppsts = m_pmfxSession->SyncOperation(syncp, 0);
            if (MFX_ERR_NONE != vppsts) {
                return vppsts;
            }
        }
        m_pTasks[m_nTaskBufferStart].Clear();
    }
    return sts;
}

void CQSVTaskControl::Close() {
    if (m_pTasks.size()) {
        for (mfxU32 i = 0; i < m_nPoolSize; i++) {
            m_pTasks[i].Close();
        }
    }
    m_pTasks.clear();

    m_pmfxSession = NULL;
    m_nTaskBufferStart = 0;
    m_nPoolSize = 0;
}
