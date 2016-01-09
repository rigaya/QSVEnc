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
#include "base_allocator.h"
#include "qsv_task.h"

QSVTask::QSVTask() :
    mfxSurf(nullptr),
    encSyncPoint(0),
    vppSyncPoint(),
    pWriter(),
    pmfxAllocator(nullptr) {
    QSV_MEMSET_ZERO(mfxBS);
}

mfxStatus QSVTask::Init(shared_ptr<CQSVOut> pTaskWriter, uint32_t nBufferSize, MFXFrameAllocator *pAllocator) {
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

mfxStatus CQSVTaskControl::Init(MFXVideoSession *pmfxSession, MFXFrameAllocator *pmfxAllocator, shared_ptr<CQSVOut> pTaskWriter, uint32_t nPoolSize, uint32_t nBufferSize) {
    if (nPoolSize == 0) {
        return MFX_ERR_INCOMPATIBLE_VIDEO_PARAM;
    }
    auto outputType = pTaskWriter->getOutType();
    if (outputType == OUT_TYPE_SURFACE) {
        //フレーム出力時には、Allocatorも必要
        if (pmfxAllocator == nullptr) {
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
        mfxStatus sts = m_pTasks[i].Init(pTaskWriter, nBufferSize, pmfxAllocator);
        if (sts < MFX_ERR_NONE) {
            return sts;
        }
    }

    return MFX_ERR_NONE;
}

mfxStatus CQSVTaskControl::SynchronizeFirstTask() {
    if (m_pTasks[m_nTaskBufferStart].encSyncPoint != NULL) {
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
