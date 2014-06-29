/* ****************************************************************************** *\

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011 - 2013 Intel Corporation. All Rights Reserved.

\* ****************************************************************************** */

#pragma once

#if defined( _WIN32 ) || defined ( _WIN64 )

#include "hw_device.h"

#pragma warning(disable : 4201)
#include <d3d9.h>
#include <dxva2api.h>
#include <dxva.h>
#include <windows.h>

enum {
    MFX_HANDLE_GFXS3DCONTROL = 0x100 /* A handle to the IGFXS3DControl instance */
}; //mfxHandleType

#define OVERLAY_BACKBUFFER_FORMAT D3DFMT_X8R8G8B8
#define VIDEO_MAIN_FORMAT D3DFMT_YUY2

class IGFXS3DControl;

/** Direct3D 9 device implementation.
@note Can be initilized for only 1 or two 2 views. Handle to 
MFX_HANDLE_GFXS3DCONTROL must be set prior if initializing for 2 views.

@note Device always set D3DPRESENT_PARAMETERS::Windowed to TRUE.
*/
class CD3D9Device : public CHWDevice
{
public:
    CD3D9Device();
    virtual ~CD3D9Device();

    virtual mfxStatus Init(
        mfxHDL hWindow,
        mfxU16 nViews,
        mfxU32 nAdapterNum);
    virtual mfxStatus Reset();
    virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL *pHdl);
    virtual mfxStatus SetHandle(mfxHandleType type, mfxHDL hdl);
    virtual mfxStatus RenderFrame(mfxFrameSurface1 * pSurface, mfxFrameAllocator * pmfxAlloc);
    virtual void Close() ;
    
protected:
    mfxStatus CreateVideoProcessors();
    bool CheckOverlaySupport();
    virtual mfxStatus FillD3DPP(mfxHDL hWindow, mfxU16 nViews, D3DPRESENT_PARAMETERS &D3DPP);
private:
    IDirect3D9Ex*               m_pD3D9;
    IDirect3DDevice9Ex*         m_pD3DD9;
    IDirect3DDeviceManager9*    m_pDeviceManager9;
    D3DPRESENT_PARAMETERS       m_D3DPP;
    UINT                        m_resetToken;

    mfxU16                      m_nViews;
    IGFXS3DControl*             m_pS3DControl;


    D3DSURFACE_DESC                 m_backBufferDesc;

    // service required to create video processors
    IDirectXVideoProcessorService*  m_pDXVAVPS;
    //left channel processor
    IDirectXVideoProcessor*         m_pDXVAVP_Left; 
    // right channel processor
    IDirectXVideoProcessor*         m_pDXVAVP_Right;
    // the surface which is passed to render
    IDirect3DSurface9*              m_pRenderSurface; 
    
    // target rectangle
    RECT                            m_targetRect;

    // various structures for DXVA2 calls
    DXVA2_VideoDesc                 m_VideoDesc;
    DXVA2_VideoProcessBltParams     m_BltParams; 
    DXVA2_VideoSample               m_Sample;
};

#endif // #if defined( _WIN32 ) || defined ( _WIN64 )
