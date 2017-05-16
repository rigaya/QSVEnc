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

#ifndef __QSV_HW_VA_H__
#define __QSV_HW_VA_H__

#include "qsv_hw_device.h"

#if defined(LIBVA_DRM_SUPPORT) || defined(LIBVA_X11_SUPPORT)
#include "rgy_osdep.h"
#include <va/va.h>

class CLibVA {
public:
    CLibVA() : m_va_dpy(NULL) {}
    virtual ~CLibVA() {};
    VADisplay GetVADisplay() { return m_va_dpy; }

protected:
    VADisplay m_va_dpy;
};

mfxStatus va_to_mfx_status(VAStatus va_res);

CLibVA *CreateLibVA();
CQSVHWDevice* CreateVAAPIDevice(void);

#endif //#if defined(LIBVA_DRM_SUPPORT) || defined(LIBVA_X11_SUPPORT)


#if defined(LIBVA_DRM_SUPPORT)

#include <va/va_drm.h>

class DRMLibVA : public CLibVA {
public:
    DRMLibVA(void);
    virtual ~DRMLibVA(void);

protected:
    int m_fd;
};

class CQSVHWVADeviceDRM : public CQSVHWDevice {
public:
    CQSVHWVADeviceDRM(){}
    virtual ~CQSVHWVADeviceDRM(void) {}

    virtual mfxStatus Init(mfxHDL hWindow, mfxU32 nAdapterNum, shared_ptr<RGYLog> pQSVLog) override { m_pQSVLog = pQSVLog; return MFX_ERR_NONE; }
    virtual mfxStatus Reset(void) override { return MFX_ERR_NONE; }
    virtual void Close(void) override { m_pQSVLog.reset(); }
    virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL *pHdl) override {
        if ((MFX_HANDLE_VA_DISPLAY == type) && (nullptr != pHdl)) {
            *pHdl = m_DRMLibVA.GetVADisplay();
            return MFX_ERR_NONE;
        }
        return MFX_ERR_UNSUPPORTED;
    }

    virtual mfxStatus RenderFrame(mfxFrameSurface1 *pSurface, mfxFrameAllocator *pmfxAlloc) { return MFX_ERR_NONE; }
    virtual void      UpdateTitle(double fps) { }

protected:
    DRMLibVA m_DRMLibVA;
};

#elif defined(LIBVA_X11_SUPPORT)

#include <va/va_x11.h>

class X11LibVA : public CLibVA {
public:
    X11LibVA(void);
    virtual ~X11LibVA(void);

    void *GetXDisplay(void) { return m_display;}

protected:
    Display* m_display;

private:
};

class CQSVHWVADeviceX11 : public CQSVHWDevice {
public:
    CQSVHWVADeviceX11(){m_window = NULL;}
    virtual ~CQSVHWVADeviceX11(void);

    virtual mfxStatus Init(mfxHDL hWindow, mfxU32 nAdapterNum, shared_ptr<RGYLog> pQSVLog) override;
    virtual mfxStatus Reset(void) override;
    virtual void Close(void) override;

    virtual mfxStatus SetHandle(mfxHandleType type, mfxHDL hdl) override;
    virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL *pHdl) override;

    virtual mfxStatus RenderFrame(mfxFrameSurface1 * pSurface, mfxFrameAllocator * pmfxAlloc) override;

protected:
    mfxHDL m_window;
    X11LibVA m_X11LibVA;
};

#endif

#endif //#ifndef __QSV_HW_VA_H__
