//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef __QSV_ALLOCATOR_H__
#define __QSV_ALLOCATOR_H__

#include <list>
#include <string.h>
#include <functional>
#include "mfxvideo.h"

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
    virtual mfxStatus Init(mfxAllocatorParams *pParams) = 0;
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
};

#endif // __QSV_ALLOCATOR_H__