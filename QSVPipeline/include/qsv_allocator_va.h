//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef __QSV_ALLOCATOR_VA_H__
#define __QSV_ALLOCATOR_VA_H__


#if defined(LIBVA_SUPPORT)

#include <stdlib.h>
#include <va/va.h>

#include "qsv_allocator.h"

// VAAPI Allocator internal Mem ID
struct vaapiMemId {
    VASurfaceID *m_surface;
    VAImage      m_image;
    // variables for VAAPI Allocator inernal color convertion
    unsigned int m_fourcc;
    mfxU8       *m_sys_buffer;
    mfxU8       *m_va_buffer;
};

class vaapiAllocatorParams : public mfxAllocatorParams {
public:
    VADisplay m_dpy;
};

class QSVAllocatorVA : public QSVAllocator {
public:
    QSVAllocatorVA();
    virtual ~QSVAllocatorVA();

    virtual mfxStatus Init(mfxAllocatorParams *pParams) override;
    virtual mfxStatus Close() override;

protected:
    virtual mfxStatus FrameLock(mfxMemId mid, mfxFrameData *ptr) override;
    virtual mfxStatus FrameUnlock(mfxMemId mid, mfxFrameData *ptr) override;
    virtual mfxStatus GetFrameHDL(mfxMemId mid, mfxHDL *handle) override;

    virtual mfxStatus CheckRequestType(mfxFrameAllocRequest *request) override;
    virtual mfxStatus ReleaseResponse(mfxFrameAllocResponse *response) override;
    virtual mfxStatus AllocImpl(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response) override;

    VADisplay m_dpy;
};

#endif //#if defined(LIBVA_SUPPORT)

#endif // __QSV_ALLOCATOR_VA_H__
