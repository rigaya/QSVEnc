//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <assert.h>
#include <algorithm>
#include "qsv_allocator.h"

QSVBufferAllocator::QSVBufferAllocator() {
    pthis = this;
    Alloc = Alloc_;
    Lock  = Lock_;
    Free  = Free_;
    Unlock = Unlock_;
}

QSVBufferAllocator::~QSVBufferAllocator() {
}

mfxStatus QSVBufferAllocator::Alloc_(mfxHDL pthis, mfxU32 nbytes, mfxU16 type, mfxMemId *mid) {
    return reinterpret_cast<QSVBufferAllocator *>(pthis)->BufAlloc(nbytes, type, mid);
}

mfxStatus QSVBufferAllocator::Lock_(mfxHDL pthis, mfxMemId mid, mfxU8 **ptr) {
    return reinterpret_cast<QSVBufferAllocator *>(pthis)->BufLock(mid, ptr);
}

mfxStatus QSVBufferAllocator::Unlock_(mfxHDL pthis, mfxMemId mid) {
    return reinterpret_cast<QSVBufferAllocator *>(pthis)->BufUnlock(mid);
}

mfxStatus QSVBufferAllocator::Free_(mfxHDL pthis, mfxMemId mid) {
    return reinterpret_cast<QSVBufferAllocator *>(pthis)->BufFree(mid);
}

QSVAllocator::QSVAllocator() {
    pthis = this;
    Alloc = Alloc_;
    Lock  = Lock_;
    Free  = Free_;
    Unlock = Unlock_;
    GetHDL = GetHDL_;
}

QSVAllocator::~QSVAllocator() {
}

mfxStatus QSVAllocator::Alloc_(mfxHDL pthis, mfxFrameAllocRequest *request, mfxFrameAllocResponse *response) {
    return reinterpret_cast<QSVAllocator *>(pthis)->FrameAlloc(request, response);
}

mfxStatus QSVAllocator::Lock_(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr) {
    return reinterpret_cast<QSVAllocator *>(pthis)->FrameLock(mid, ptr);
}

mfxStatus QSVAllocator::Unlock_(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr) {
    return reinterpret_cast<QSVAllocator *>(pthis)->FrameUnlock(mid, ptr);
}

mfxStatus QSVAllocator::Free_(mfxHDL pthis, mfxFrameAllocResponse *response) {
    return reinterpret_cast<QSVAllocator *>(pthis)->FrameFree(response);
}

mfxStatus QSVAllocator::GetHDL_(mfxHDL pthis, mfxMemId mid, mfxHDL *handle) {
    return reinterpret_cast<QSVAllocator *>(pthis)->GetFrameHDL(mid, handle);
}

mfxStatus QSVAllocator::CheckRequestType(mfxFrameAllocRequest *request) {
    if (0 == request)
        return MFX_ERR_NULL_PTR;

    if ((request->Type & MEMTYPE_FROM_MASK) != 0)
        return MFX_ERR_NONE;
    else
        return MFX_ERR_UNSUPPORTED;
}

mfxStatus QSVAllocator::FrameAlloc(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response) {
    if (0 == request || 0 == response || 0 == request->NumFrameSuggested) {
        return MFX_ERR_MEMORY_ALLOC;
    }

    if (MFX_ERR_NONE != CheckRequestType(request)) {
        return MFX_ERR_UNSUPPORTED;
    }

    mfxStatus sts = MFX_ERR_NONE;

    if ((request->Type & (MFX_MEMTYPE_EXTERNAL_FRAME | MFX_MEMTYPE_FROM_DECODE)) == (MFX_MEMTYPE_EXTERNAL_FRAME | MFX_MEMTYPE_FROM_DECODE)) {
        //external
        auto it = std::find_if( m_ExtResponses.begin(), m_ExtResponses.end(), UniqueResponse(*response, request->Info.CropW, request->Info.CropH, 0));

        if (it != m_ExtResponses.end()) {
            if (request->NumFrameSuggested > it->NumFrameActual) {
                return MFX_ERR_MEMORY_ALLOC;
            }

            it->m_refCount++;
            *response = (mfxFrameAllocResponse&)*it;
        } else if (AllocImpl(request, response) == MFX_ERR_NONE) {
            m_ExtResponses.push_back(UniqueResponse(*response, request->Info.CropW, request->Info.CropH, request->Type & MEMTYPE_FROM_MASK));
        }
    } else {
        //internal
        m_responses.push_back(mfxFrameAllocResponse());
        if (AllocImpl(request, response) == MFX_ERR_NONE) {
            m_responses.back() = *response;
        } else {
            m_responses.pop_back();
        }
    }

    return sts;
}

mfxStatus QSVAllocator::FrameFree(mfxFrameAllocResponse *response) {
    if (response == 0)
        return MFX_ERR_INVALID_HANDLE;

    mfxStatus sts = MFX_ERR_NONE;

    auto compare_response = [response](const mfxFrameAllocResponse & r) {
        return r.mids != 0 && response->mids != 0 && r.mids[0] == response->mids[0] && r.NumFrameActual == response->NumFrameActual;
    };

    //external decoder responseを検索
    auto it = std::find_if(m_ExtResponses.begin(), m_ExtResponses.end(), compare_response);
    if (it != m_ExtResponses.end()) {
        if ((--it->m_refCount) == 0) {
            sts = ReleaseResponse(response);
            m_ExtResponses.erase(it);
        }
        return sts;
    }

    //internal responsesを検索
    auto it2 = std::find_if(m_responses.begin(), m_responses.end(), compare_response);
    if (it2 != m_responses.end()) {
        sts = ReleaseResponse(response);
        m_responses.erase(it2);
        return sts;
    }

    return MFX_ERR_INVALID_HANDLE;
}

mfxStatus QSVAllocator::Close() {
    for (auto it = m_ExtResponses.begin(); it != m_ExtResponses.end(); it++) {
        ReleaseResponse(&*it);
    }
    m_ExtResponses.clear();

    for (auto it2 = m_responses.begin(); it2 != m_responses.end(); it2++) {
        ReleaseResponse(&*it2);
    }

    return MFX_ERR_NONE;
}
