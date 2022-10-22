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

#ifndef __QSV_HW_D3D9_H__
#define __QSV_HW_D3D9_H__

#include "qsv_hw_device.h"

#if defined( _WIN32 ) || defined ( _WIN64 )
#pragma warning(disable : 4201)
#include <d3d9.h>
#include <dxva2api.h>
#include <dxva.h>
#include <windows.h>
#include "qsv_util.h"
#pragma comment(lib, "d3d9.lib")
#pragma comment(lib, "dxva2.lib")

struct IUnknown_release {
    void operator()(IUnknown *m_pHandle) const {
        m_pHandle->Release();
    }
};

class CQSVD3D9Device : public CQSVHWDevice {
public:
    CQSVD3D9Device(shared_ptr<RGYLog> pQSVLog);
    virtual ~CQSVD3D9Device();

    virtual mfxStatus Init(mfxHDL hWindow, uint32_t nViews, uint32_t nAdapterNum) override;
    virtual mfxStatus Reset();
    virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL *pHdl) override;
    virtual void      Close() override;
    virtual LUID      GetLUID() override;
protected:
    mfxStatus CreateVideoProcessors();
private:
    unique_ptr<IDirect3D9Ex,            IUnknown_release> m_pD3D9;
    unique_ptr<IDirect3DDevice9Ex,      IUnknown_release> m_pD3DDevice9;
    unique_ptr<IDirect3DDeviceManager9, IUnknown_release> m_pD3DDeviceManager9;

    unique_ptr<IDirectXVideoProcessorService, IUnknown_release> m_pDXVAVProcessorService;
    unique_ptr<IDirectXVideoProcessor,        IUnknown_release> m_pDXVAVProcessor;

    LUID m_devLUID;
    std::wstring m_displayDeviceName;

    D3DPRESENT_PARAMETERS m_D3DPresentPrm;
    UINT                  m_resetToken;
    D3DSURFACE_DESC       m_backBufferDesc;

    RECT                            m_targetRect;

    DXVA2_VideoDesc                 m_VideoDesc;
    DXVA2_VideoProcessBltParams     m_BltParams;
    DXVA2_VideoSample               m_Sample;

    BOOL                            m_bIsA2rgb10;
};

#endif // #if defined( _WIN32 ) || defined ( _WIN64 )
#endif //#ifndef __QSV_HW_D3D9_H__