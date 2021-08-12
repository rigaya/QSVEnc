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
// --------------------------------------------------------------------------------------------

#include <assert.h>
#include <algorithm>
#include "qsv_allocator.h"
#include "qsv_util.h"

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
    mfxStatus sts = MFX_ERR_NONE;
    AddMessage(RGY_LOG_DEBUG, _T("QSVAllocator: FrameAlloc: %s, %d frames.\n"), qsv_memtype_str(request->Type).c_str(), request->NumFrameSuggested);
    if (MFX_ERR_NONE != (sts = CheckRequestType(request))) {
        AddMessage(RGY_LOG_ERROR, _T("QSVAllocator: Failed CheckRequestType: %s\n"), get_err_mes(sts));
        return MFX_ERR_UNSUPPORTED;
    }

    if ( // External Frames
        ((request->Type & MFX_MEMTYPE_EXTERNAL_FRAME) &&
         (request->Type & (MFX_MEMTYPE_FROM_DECODE | MFX_MEMTYPE_FROM_ENC | MFX_MEMTYPE_FROM_PAK)))
         // Exception: Internal Frames for FEI ENC / PAK reconstructs
         ||
        ((request->Type & MFX_MEMTYPE_INTERNAL_FRAME) &&
         (request->Type & (MFX_MEMTYPE_FROM_ENC | MFX_MEMTYPE_FROM_PAK)))
       ) {
        //external
        AddMessage(RGY_LOG_DEBUG, _T("QSVAllocator: Allocate type external.\n"));
        bool foundInCache = false;
        // external decoder allocations
        std::list<UniqueResponse>::iterator
            it = m_ExtResponses.begin(),
            et = m_ExtResponses.end();
        UniqueResponse checker(*response, request->Info.Width, request->Info.Height, request->Type);
        for (; it != et; ++it) {
            // same decoder and same size
            if (request->AllocId == it->AllocId && checker(*it)) {
                // check if enough frames were allocated
                if (request->NumFrameSuggested > it->NumFrameActual) {
                    AddMessage(RGY_LOG_ERROR, _T("QSVAllocator: NumFrameSuggested > it->NumFrameActual\n"));
                    return MFX_ERR_MEMORY_ALLOC;
                }

                it->m_refCount++;
                // return existing response
                *response = (mfxFrameAllocResponse&)*it;
                foundInCache = true;
            }
        }

        if (!foundInCache) {
            if ((AllocImpl(request, response)) == MFX_ERR_NONE) {
                response->AllocId = request->AllocId;
                m_ExtResponses.push_back(UniqueResponse(*response, request->Info.Width, request->Info.Height, UniqueResponse::CropMemoryTypeToStore(request->Type)));
            } else {
                AddMessage(RGY_LOG_ERROR, _T("QSVAllocator: Failed Allocate type external: %s\n"), get_err_mes(sts));
                return sts;
            }
        }
    } else {
        //internal
        AddMessage(RGY_LOG_DEBUG, _T("QSVAllocator: Allocate type internal.\n"));
        m_responses.push_back(mfxFrameAllocResponse());
        sts = AllocImpl(request, response);
        if (sts == MFX_ERR_NONE) {
            m_responses.back() = *response;
        } else {
            m_responses.pop_back();
            AddMessage(RGY_LOG_ERROR, _T("QSVAllocator: Failed Allocate type internal: %s\n"), get_err_mes(sts));
            return sts;
        }
    }
    AddMessage(RGY_LOG_DEBUG, _T("QSVAllocator: FrameAlloc success.\n"));
    return sts;
}

mfxStatus QSVAllocator::FrameFree(mfxFrameAllocResponse *response) {
    if (response == 0)
        return MFX_ERR_INVALID_HANDLE;

    mfxStatus sts = MFX_ERR_NONE;
    AddMessage(RGY_LOG_DEBUG, _T("QSVAllocator: FrameFree...\n"));

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
    AddMessage(RGY_LOG_DEBUG, _T("QSVAllocator: FrameFree external success.\n"));

    //internal responsesを検索
    auto it2 = std::find_if(m_responses.begin(), m_responses.end(), compare_response);
    if (it2 != m_responses.end()) {
        sts = ReleaseResponse(response);
        m_responses.erase(it2);
        return sts;
    }
    AddMessage(RGY_LOG_DEBUG, _T("QSVAllocator: FrameFree internal success.\n"));
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
    AddMessage(RGY_LOG_DEBUG, _T("QSVAllocator: Closed.\n"));
    m_pQSVLog.reset();
    return MFX_ERR_NONE;
}

void QSVAllocator::AddMessage(RGYLogLevel log_level, const tstring &str) {
    if (m_pQSVLog == nullptr || log_level < m_pQSVLog->getLogLevel(RGY_LOGT_DEV)) {
        return;
    }
    auto lines = split(str, _T("\n"));
    for (const auto &line : lines) {
        if (line[0] != _T('\0')) {
            m_pQSVLog->write(log_level, RGY_LOGT_DEV, (m_name + _T(": ") + line + _T("\n")).c_str());
        }
    }
}
void QSVAllocator::AddMessage(RGYLogLevel log_level, const TCHAR *format, ...) {
    if (m_pQSVLog == nullptr || log_level < m_pQSVLog->getLogLevel(RGY_LOGT_DEV)) {
        return;
    }

    va_list args;
    va_start(args, format);
    int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    tstring buffer;
    buffer.resize(len, _T('\0'));
    _vstprintf_s(&buffer[0], len, format, args);
    va_end(args);
    AddMessage(log_level, buffer);
}
