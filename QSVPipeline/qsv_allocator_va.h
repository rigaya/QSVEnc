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

class QSVAllocatorParamsVA : public mfxAllocatorParams {
public:
    VADisplay m_dpy;
};

class QSVAllocatorVA : public QSVAllocator {
public:
    QSVAllocatorVA();
    virtual ~QSVAllocatorVA();

    virtual mfxStatus Init(mfxAllocatorParams *pParams, shared_ptr<RGYLog> pQSVLog) override;
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
