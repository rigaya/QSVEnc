//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#ifndef __QSV_HW_VA_H__
#define __QSV_HW_VA_H__

#include "qsv_hw_device.h"

#if defined(LIBVA_DRM_SUPPORT) || defined(LIBVA_X11_SUPPORT)
#include "qsv_osdep.h"
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

    virtual mfxStatus Init(mfxHDL hWindow, mfxU32 nAdapterNum) override { return MFX_ERR_NONE;}
    virtual mfxStatus Reset(void) override { return MFX_ERR_NONE; }
    virtual void Close(void) override { }
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

    virtual mfxStatus Init(mfxHDL hWindow, mfxU32 nAdapterNum) override;
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
