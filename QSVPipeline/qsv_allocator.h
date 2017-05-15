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

#ifndef __QSV_ALLOCATOR_H__
#define __QSV_ALLOCATOR_H__

#include <list>
#include <memory>
#include <cstring>
#include <functional>
#include "mfxvideo.h"
#include "rgy_log.h"

class QSVBufferAllocator : public mfxBufferAllocator {
public:
    QSVBufferAllocator();
    virtual ~QSVBufferAllocator();

    virtual mfxStatus BufAlloc(mfxU32 nbytes, mfxU16 type, mfxMemId *mid) = 0;
    virtual mfxStatus BufLock(mfxMemId mid, mfxU8 **ptr) = 0;
    virtual mfxStatus BufUnlock(mfxMemId mid) = 0;
    virtual mfxStatus BufFree(mfxMemId mid) = 0;

private:
    static mfxStatus MFX_CDECL Alloc_(mfxHDL pthis, mfxU32 nbytes, mfxU16 type, mfxMemId *mid);
    static mfxStatus MFX_CDECL Lock_(mfxHDL pthis, mfxMemId mid, mfxU8 **ptr);
    static mfxStatus MFX_CDECL Unlock_(mfxHDL pthis, mfxMemId mid);
    static mfxStatus MFX_CDECL Free_(mfxHDL pthis, mfxMemId mid);
};

class mfxAllocatorParams {
public:
    virtual ~mfxAllocatorParams(){};
};

class QSVAllocator : public mfxFrameAllocator {
public:
    QSVAllocator();
    virtual ~QSVAllocator();
    virtual mfxStatus Init(mfxAllocatorParams *pParams, shared_ptr<RGYLog> pQSVLog) = 0;
    virtual mfxStatus Close();

    virtual mfxStatus FrameAlloc(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response);
    virtual mfxStatus FrameLock(mfxMemId mid, mfxFrameData *ptr) = 0;
    virtual mfxStatus FrameUnlock(mfxMemId mid, mfxFrameData *ptr) = 0;
    virtual mfxStatus GetFrameHDL(mfxMemId mid, mfxHDL *handle) = 0;
    virtual mfxStatus FrameFree(mfxFrameAllocResponse *response);
private:
    static mfxStatus MFX_CDECL Alloc_(mfxHDL pthis, mfxFrameAllocRequest *request, mfxFrameAllocResponse *response);
    static mfxStatus MFX_CDECL Lock_(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr);
    static mfxStatus MFX_CDECL Unlock_(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr);
    static mfxStatus MFX_CDECL GetHDL_(mfxHDL pthis, mfxMemId mid, mfxHDL *handle);
    static mfxStatus MFX_CDECL Free_(mfxHDL pthis, mfxFrameAllocResponse *response);

protected:
    virtual mfxStatus CheckRequestType(mfxFrameAllocRequest *request);
    virtual mfxStatus ReleaseResponse(mfxFrameAllocResponse *response) = 0;
    virtual mfxStatus AllocImpl(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response) = 0;
    static const auto MEMTYPE_FROM_MASK = MFX_MEMTYPE_FROM_ENCODE | MFX_MEMTYPE_FROM_DECODE | MFX_MEMTYPE_FROM_VPPIN | MFX_MEMTYPE_FROM_VPPOUT;

    struct UniqueResponse : mfxFrameAllocResponse {
        mfxU16 m_cropw;
        mfxU16 m_croph;
        mfxU32 m_refCount;
        mfxU16 m_type;

        UniqueResponse() :
            m_cropw(0),
            m_croph(0),
            m_refCount(0),
            m_type(0) {
            memset(static_cast<mfxFrameAllocResponse*>(this), 0, sizeof(mfxFrameAllocResponse));
        }

        UniqueResponse(const mfxFrameAllocResponse & response, mfxU16 cropw, mfxU16 croph, mfxU16 type) :
            mfxFrameAllocResponse(response),
            m_cropw(cropw),
            m_croph(croph),
            m_refCount(1),
            m_type(type) {
        }
        bool operator() (const UniqueResponse &response) const {
            return m_cropw == response.m_cropw && m_croph == response.m_croph;
        }
    };

    std::list<mfxFrameAllocResponse> m_responses;
    std::list<UniqueResponse> m_ExtResponses;
    shared_ptr<RGYLog> m_pQSVLog;
};

#endif // __QSV_ALLOCATOR_H__