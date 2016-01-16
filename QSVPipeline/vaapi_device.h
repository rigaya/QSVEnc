/* ****************************************************************************** *\

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2013 Intel Corporation. All Rights Reserved.

\* ****************************************************************************** */

#if defined(LIBVA_DRM_SUPPORT) || defined(LIBVA_X11_SUPPORT) || defined(LIBVA_ANDROID_SUPPORT)

#include "hw_device.h"
#include "vaapi_utils_drm.h"
#include "vaapi_utils_x11.h"
#if defined(LIBVA_ANDROID_SUPPORT)
#include "vaapi_utils_android.h"
#endif

CHWDevice* CreateVAAPIDevice(void);

#if defined(LIBVA_DRM_SUPPORT)
/** VAAPI DRM implementation. */
class CVAAPIDeviceDRM : public CHWDevice
{
public:
    CVAAPIDeviceDRM(){}
    virtual ~CVAAPIDeviceDRM(void) {}

    virtual mfxStatus Init(mfxHDL hWindow, mfxU16 nViews, mfxU32 nAdapterNum) { return MFX_ERR_NONE;}
    virtual mfxStatus Reset(void) { return MFX_ERR_NONE; }
    virtual void Close(void) { }

    virtual mfxStatus SetHandle(mfxHandleType type, mfxHDL hdl) { return MFX_ERR_UNSUPPORTED; }
    virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL *pHdl)
    {
        if ((MFX_HANDLE_VA_DISPLAY == type) && (NULL != pHdl))
        {
            *pHdl = m_DRMLibVA.GetVADisplay();

            return MFX_ERR_NONE;
        }
        return MFX_ERR_UNSUPPORTED;
    }

    virtual mfxStatus RenderFrame(mfxFrameSurface1 * pSurface, mfxFrameAllocator * pmfxAlloc) { return MFX_ERR_NONE; }
    virtual void      UpdateTitle(double fps) { }

protected:
    DRMLibVA m_DRMLibVA;
};

#elif defined(LIBVA_X11_SUPPORT)

/** VAAPI X11 implementation. */
class CVAAPIDeviceX11 : public CHWDevice
{
public:
    CVAAPIDeviceX11(){m_window = NULL;}
    virtual ~CVAAPIDeviceX11(void);

    virtual mfxStatus Init(
            mfxHDL hWindow,
            mfxU16 nViews,
            mfxU32 nAdapterNum);
    virtual mfxStatus Reset(void);
    virtual void Close(void);

    virtual mfxStatus SetHandle(mfxHandleType type, mfxHDL hdl);
    virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL *pHdl);

    virtual mfxStatus RenderFrame(mfxFrameSurface1 * pSurface, mfxFrameAllocator * pmfxAlloc);
    virtual void      UpdateTitle(double fps) { }

protected:
    mfxHDL m_window;
    X11LibVA m_X11LibVA;
};

#elif defined(LIBVA_ANDROID_SUPPORT)

/** VAAPI Android implementation. */
class CVAAPIDeviceAndroid : public CHWDevice
{
public:
    CVAAPIDeviceAndroid(void) {};
    virtual ~CVAAPIDeviceAndroid(void)  { Close();}

    virtual mfxStatus Init(mfxHDL hWindow, mfxU16 nViews, mfxU32 nAdapterNum) { return MFX_ERR_NONE;}
    virtual mfxStatus Reset(void) { return MFX_ERR_NONE; }
    virtual void Close(void) { }

    virtual mfxStatus SetHandle(mfxHandleType type, mfxHDL hdl) { return MFX_ERR_UNSUPPORTED; }
    virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL *pHdl)
    {
        if ((MFX_HANDLE_VA_DISPLAY == type) && (NULL != pHdl))
        {
            *pHdl = m_AndroidLibVA.GetVADisplay();

            return MFX_ERR_NONE;
        }

        return MFX_ERR_UNSUPPORTED;
    }

    virtual mfxStatus RenderFrame(mfxFrameSurface1 * pSurface, mfxFrameAllocator * pmfxAlloc) { return MFX_ERR_NONE; }
    virtual void      UpdateTitle(double fps) { }

protected:
    AndroidLibVA m_AndroidLibVA;
};
#endif
#endif //#if defined(LIBVA_DRM_SUPPORT) || defined(LIBVA_X11_SUPPORT) || defined(LIBVA_ANDROID_SUPPORT)