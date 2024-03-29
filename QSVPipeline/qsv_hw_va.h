﻿/******************************************************************************\
Copyright (c) 2005-2019, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This sample was distributed or derived from the Intel's Media Samples package.
The original version of this sample may be obtained from https://software.intel.com/en-us/intel-media-server-studio
or https://software.intel.com/en-us/media-client-solutions-support.
\**********************************************************************************/

#ifndef __QSV_HW_VA_H__
#define __QSV_HW_VA_H__

#if defined(LIBVA_DRM_SUPPORT) || defined(LIBVA_X11_SUPPORT) || defined(LIBVA_ANDROID_SUPPORT) || defined(LIBVA_WAYLAND_SUPPORT)

#include "qsv_hw_device.h"
#include "qsv_hw_va_utils_drm.h"
#include "qsv_hw_va_utils_x11.h"
#if defined(LIBVA_ANDROID_SUPPORT)
#include "vaapi_utils_android.h"
#endif

CQSVHWDevice* CreateVAAPIDevice(const std::string& devicePath = "", int type = MFX_LIBVA_DRM, std::shared_ptr<RGYLog> log = std::make_shared<RGYLog>(nullptr, RGY_LOG_ERROR));

#if defined(LIBVA_DRM_SUPPORT)
/** VAAPI DRM implementation. */
class CVAAPIDeviceDRM : public CQSVHWDevice
{
public:
    CVAAPIDeviceDRM(const std::string& devicePath, int type, std::shared_ptr<RGYLog> pQSVLog);
    virtual ~CVAAPIDeviceDRM(void);

    virtual mfxStatus Init(mfxHDL hWindow, uint32_t nViews, uint32_t nAdapterNum) override;
    virtual mfxStatus Reset(void) override { return MFX_ERR_NONE; }
    virtual void Close(void) override { }

    virtual mfxStatus SetHandle(mfxHandleType type, mfxHDL hdl) { return MFX_ERR_UNSUPPORTED; }
    virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL *pHdl) override
    {
        if ((MFX_HANDLE_VA_DISPLAY == type) && (NULL != pHdl))
        {
            *pHdl = m_DRMLibVA.GetVADisplay();

            return MFX_ERR_NONE;
        }
        return MFX_ERR_UNSUPPORTED;
    }

    virtual mfxStatus RenderFrame(mfxFrameSurface1 * pSurface, mfxFrameAllocator * pmfxAlloc);
    virtual void      UpdateTitle(double fps) { }
    virtual void      SetMondelloInput(bool isMondelloInputEnabled) { }

    inline drmRenderer* getRenderer() { return m_rndr; }
protected:
    DRMLibVA m_DRMLibVA;
    drmRenderer * m_rndr;
private:
    // no copies allowed
    CVAAPIDeviceDRM(const CVAAPIDeviceDRM &);
    void operator=(const CVAAPIDeviceDRM &);
};

#endif

#if defined(LIBVA_X11_SUPPORT)

/** VAAPI X11 implementation. */
class CVAAPIDeviceX11 : public CQSVHWDevice
{
public:
    CVAAPIDeviceX11(std::shared_ptr<RGYLog> pQSVLog) : CQSVHWDevice(pQSVLog), m_window(NULL), m_X11LibVA(pQSVLog)
    {
        m_name = _T("vaX11");
        m_window = NULL;
        m_nRenderWinX=0;
        m_nRenderWinY=0;
        m_nRenderWinW=0;
        m_nRenderWinH=0;
        m_bRenderWin=false;
#if defined(X11_DRI3_SUPPORT)
        m_dri_fd = 0;
        m_bufmgr = NULL;
        m_xcbconn = NULL;
#endif
    }
    virtual ~CVAAPIDeviceX11(void);

    virtual mfxStatus Init(mfxHDL hWindow, uint32_t nViews, uint32_t nAdapterNum) override;
    virtual mfxStatus Reset(void) override;
    virtual void Close(void) override;

    virtual mfxStatus SetHandle(mfxHandleType type, mfxHDL hdl);
    virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL *pHdl) override;

    virtual mfxStatus RenderFrame(mfxFrameSurface1 * pSurface, mfxFrameAllocator * pmfxAlloc);
    virtual void      UpdateTitle(double fps) { }
    virtual void      SetMondelloInput(bool isMondelloInputEnabled) { }

protected:
    mfxHDL m_window;
    X11LibVA m_X11LibVA;
private:

    bool   m_bRenderWin;
    mfxU32 m_nRenderWinX;
    mfxU32 m_nRenderWinY;
    mfxU32 m_nRenderWinW;
    mfxU32 m_nRenderWinH;
#if defined(X11_DRI3_SUPPORT)
    int m_dri_fd;
    drm_intel_bufmgr* m_bufmgr;
    xcb_connection_t *m_xcbconn;
#endif
    // no copies allowed
    CVAAPIDeviceX11(const CVAAPIDeviceX11 &);
    void operator=(const CVAAPIDeviceX11 &);
};

#endif

#if defined(LIBVA_WAYLAND_SUPPORT)

class Wayland;

class CVAAPIDeviceWayland : public CQSVHWDevice
{
public:
    CVAAPIDeviceWayland(std::shared_ptr<RGYLog> pQSVLog) : CQSVHWDevice(pQSVLog){
        m_name = _T("vaWayland");
        m_nRenderWinX = 0;
        m_nRenderWinY = 0;
        m_nRenderWinW = 0;
        m_nRenderWinH = 0;
        m_isMondelloInputEnabled = false;
        m_Wayland = NULL;
    }
    virtual ~CVAAPIDeviceWayland(void);

    virtual mfxStatus Init(mfxHDL hWindow, uint32_t nViews, uint32_t nAdapterNum) override;
    virtual mfxStatus Reset(void) override { return MFX_ERR_NONE; }
    virtual void Close(void) override;

    virtual mfxStatus SetHandle(mfxHandleType type, mfxHDL hdl) { return MFX_ERR_UNSUPPORTED; }
    virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL *pHdl) override
    {
        if((MFX_HANDLE_VA_DISPLAY == type) && (NULL != pHdl)) {
            *pHdl = m_DRMLibVA.GetVADisplay();
            return MFX_ERR_NONE;
        }

        return MFX_ERR_UNSUPPORTED;
    }
    virtual mfxStatus RenderFrame(mfxFrameSurface1 * pSurface, mfxFrameAllocator * pmfxAlloc);
    virtual void UpdateTitle(double fps) { }

    virtual void SetMondelloInput(bool isMondelloInputEnabled)
    {
        m_isMondelloInputEnabled = isMondelloInputEnabled;
    }

    Wayland * GetWaylandHandle()
    {
        return m_Wayland;
    }
protected:
    DRMLibVA m_DRMLibVA;
    MfxLoader::VA_WaylandClientProxy  m_WaylandClient;
    Wayland *m_Wayland;
private:
    mfxU32 m_nRenderWinX;
    mfxU32 m_nRenderWinY;
    mfxU32 m_nRenderWinW;
    mfxU32 m_nRenderWinH;

    bool m_isMondelloInputEnabled;

    // no copies allowed
    CVAAPIDeviceWayland(const CVAAPIDeviceWayland &);
    void operator=(const CVAAPIDeviceWayland &);
};

#endif

#endif //#if defined(LIBVA_DRM_SUPPORT) || defined(LIBVA_X11_SUPPORT) || defined(LIBVA_ANDROID_SUPPORT)
#endif //__QSV_HW_VA_H__
