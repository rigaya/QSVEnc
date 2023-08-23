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

#include "qsv_hw_d3d9.h"
#if defined(WIN32) || defined(WIN64)
#pragma warning(disable:4068)

CQSVD3D9Device::CQSVD3D9Device(shared_ptr<RGYLog> pQSVLog) : CQSVHWDevice(pQSVLog) {
    m_name = _T("d3d9");
    RGY_MEMSET_ZERO(m_D3DPresentPrm);
    m_resetToken = 0;
    m_devLUID = LUID();

    RGY_MEMSET_ZERO(m_backBufferDesc);
    RGY_MEMSET_ZERO(m_targetRect);
    RGY_MEMSET_ZERO(m_VideoDesc);
    RGY_MEMSET_ZERO(m_BltParams);
    RGY_MEMSET_ZERO(m_Sample);

    DXVA2_AYUVSample16 color = {
        0x8000,          // Cr
        0x8000,          // Cb
        0x1000,          // Y
        0xffff           // Alpha
    };

    DXVA2_ExtendedFormat format =   {           // DestFormat
        DXVA2_SampleProgressiveFrame,           // SampleFormat
        DXVA2_VideoChromaSubsampling_MPEG2,     // VideoChromaSubsampling
        DXVA_NominalRange_0_255,                // NominalRange
        DXVA2_VideoTransferMatrix_BT709,        // VideoTransferMatrix
        DXVA2_VideoLighting_bright,             // VideoLighting
        DXVA2_VideoPrimaries_BT709,             // VideoPrimaries
        DXVA2_VideoTransFunc_709                // VideoTransferFunction
    };

    memcpy(&m_VideoDesc.SampleFormat, &format, sizeof(DXVA2_ExtendedFormat));
    m_VideoDesc.SampleWidth                         = 0;
    m_VideoDesc.SampleHeight                        = 0;
    m_VideoDesc.InputSampleFreq.Numerator           = 60;
    m_VideoDesc.InputSampleFreq.Denominator         = 1;
    m_VideoDesc.OutputFrameFreq.Numerator           = 60;
    m_VideoDesc.OutputFrameFreq.Denominator         = 1;

    memcpy(&m_BltParams.DestFormat, &format, sizeof(DXVA2_ExtendedFormat));
    memcpy(&m_BltParams.BackgroundColor, &color, sizeof(DXVA2_AYUVSample16));

    m_Sample.Start = 0;
    m_Sample.End = 1;
    m_Sample.SampleFormat = format;
    m_Sample.PlanarAlpha.Fraction = 0;
    m_Sample.PlanarAlpha.Value = 1;

    m_bIsA2rgb10 = FALSE;
}

mfxStatus CQSVD3D9Device::Init(mfxHDL hWindow, [[maybe_unused]] uint32_t nViews, uint32_t nAdapterNum) {
    mfxStatus sts = MFX_ERR_NONE;
    AddMessage(RGY_LOG_DEBUG, _T("D3D9Device: Init...\n"));

    HRESULT hr = 0;
    IDirect3D9Ex *pD3D9 = nullptr;
    if (FAILED(hr = Direct3DCreate9Ex(D3D_SDK_VERSION, &pD3D9))) {
        AddMessage(RGY_LOG_ERROR, _T("D3D9Device: Failed Direct3DCreate9Ex: %d.\n"), hr);
        return MFX_ERR_DEVICE_FAILED;
    }
    m_pD3D9.reset(pD3D9);
    AddMessage(RGY_LOG_DEBUG, _T("D3D9Device: Direct3DCreate9Ex Success.\n"));

    ZeroMemory(&m_D3DPresentPrm, sizeof(m_D3DPresentPrm));
    m_D3DPresentPrm.Windowed = true;
    m_D3DPresentPrm.hDeviceWindow = (HWND)hWindow;
    m_D3DPresentPrm.Flags                      = D3DPRESENTFLAG_VIDEO;
    m_D3DPresentPrm.FullScreen_RefreshRateInHz = D3DPRESENT_RATE_DEFAULT;
    m_D3DPresentPrm.PresentationInterval       = D3DPRESENT_INTERVAL_ONE;
    m_D3DPresentPrm.BackBufferCount            = 1;
    m_D3DPresentPrm.BackBufferFormat           = (m_bIsA2rgb10) ? D3DFMT_A2R10G10B10 : D3DFMT_X8R8G8B8;
    m_D3DPresentPrm.BackBufferWidth  = GetSystemMetrics(SM_CYSCREEN);
    m_D3DPresentPrm.BackBufferHeight = GetSystemMetrics(SM_CYSCREEN);

    if (hWindow) {
        RECT r;
        GetClientRect((HWND)hWindow, &r);
        m_D3DPresentPrm.BackBufferWidth  = (std::min<uint32_t>)(m_D3DPresentPrm.BackBufferWidth,  r.right - r.left);
        m_D3DPresentPrm.BackBufferHeight = (std::min<uint32_t>)(m_D3DPresentPrm.BackBufferHeight, r.bottom - r.top);
        AddMessage(RGY_LOG_DEBUG, _T("D3D9Device: Set Init() m_D3DPresentPrm.BackBuffer.\n"));
    }
    //
    // Mark the back buffer lockable if software DXVA2 could be used.
    // This is because software DXVA2 device requires a lockable render target
    // for the optimal performance.
    //
    m_D3DPresentPrm.Flags |= D3DPRESENTFLAG_LOCKABLE_BACKBUFFER;
    m_D3DPresentPrm.SwapEffect = D3DSWAPEFFECT_DISCARD;

    //Get LUID
    pD3D9->GetAdapterLUID((UINT)nAdapterNum, &m_devLUID);

    IDirect3DDevice9Ex *pD3DD9 = nullptr;
    if (FAILED(hr = m_pD3D9->CreateDeviceEx(
        nAdapterNum, D3DDEVTYPE_HAL, (HWND)hWindow,
        D3DCREATE_SOFTWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED | D3DCREATE_FPU_PRESERVE,
        &m_D3DPresentPrm, NULL, &pD3DD9))) {
        AddMessage(RGY_LOG_ERROR, _T("D3D9Device: Failed CreateDeviceEx: %d.\n"), hr);
        return MFX_ERR_NULL_PTR;
    }
    m_pD3DDevice9.reset(pD3DD9);
    AddMessage(RGY_LOG_DEBUG, _T("D3D9Device: CreateDeviceEx Success.\n"));

    if (hWindow) {
        if (   FAILED(m_pD3DDevice9->ResetEx(&m_D3DPresentPrm, NULL))
            || FAILED(m_pD3DDevice9->Clear(0, NULL, D3DCLEAR_TARGET, D3DCOLOR_XRGB(0, 0, 0), 1.0f, 0))) {
            return MFX_ERR_UNDEFINED_BEHAVIOR;
        }
    }
    UINT resetToken = 0;
    IDirect3DDeviceManager9 *pDeviceManager9 = nullptr;
    if (FAILED(DXVA2CreateDirect3DDeviceManager9(&resetToken, &pDeviceManager9))) {
        AddMessage(RGY_LOG_ERROR, _T("D3D9Device: Failed DXVA2CreateDirect3DDeviceManager9: %d.\n"), hr);
        return MFX_ERR_NULL_PTR;
    }
    m_pD3DDeviceManager9.reset(pDeviceManager9);
    AddMessage(RGY_LOG_DEBUG, _T("D3D9Device: DXVA2CreateDirect3DDeviceManager9 Success.\n"));

    if (FAILED(hr = m_pD3DDeviceManager9->ResetDevice(m_pD3DDevice9.get(), resetToken))) {
        AddMessage(RGY_LOG_ERROR, _T("D3D9Device: Failed ResetDevice: %d.\n"), hr);
        return MFX_ERR_UNDEFINED_BEHAVIOR;
    }
    AddMessage(RGY_LOG_DEBUG, _T("D3D9Device: ResetDevice Success.\n"));

    m_resetToken = resetToken;

    return sts;
}

mfxStatus CQSVD3D9Device::Reset() {
    m_D3DPresentPrm.BackBufferWidth  = GetSystemMetrics(SM_CXSCREEN);
    m_D3DPresentPrm.BackBufferHeight = GetSystemMetrics(SM_CYSCREEN);
    if (m_D3DPresentPrm.Windowed) {
        RECT r;
        GetClientRect((HWND)m_D3DPresentPrm.hDeviceWindow, &r);
        m_D3DPresentPrm.BackBufferWidth  = (std::min<uint32_t>)(m_D3DPresentPrm.BackBufferWidth,  r.right - r.left);
        m_D3DPresentPrm.BackBufferHeight = (std::min<uint32_t>)(m_D3DPresentPrm.BackBufferHeight, r.bottom - r.top);
        AddMessage(RGY_LOG_DEBUG, _T("D3D9Device: Set Reset() m_D3DPresentPrm.BackBuffer.\n"));
    }

    HRESULT hr = 0;
    D3DPRESENT_PARAMETERS d3dpp = m_D3DPresentPrm;
    if (FAILED(m_pD3DDevice9->ResetEx(&d3dpp, NULL))) {
        AddMessage(RGY_LOG_ERROR, _T("D3D9Device: Failed ResetEx: %d.\n"), hr);
        return MFX_ERR_UNDEFINED_BEHAVIOR;
    } else if (FAILED(m_pD3DDeviceManager9->ResetDevice(m_pD3DDevice9.get(), m_resetToken))) {
        AddMessage(RGY_LOG_ERROR, _T("D3D9Device: Failed ResetDevice: %d.\n"), hr);
        return MFX_ERR_UNDEFINED_BEHAVIOR;
    }
    AddMessage(RGY_LOG_DEBUG, _T("D3D9Device: Reset Success.\n"));
    return MFX_ERR_NONE;
}

void CQSVD3D9Device::Close() {
    m_pDXVAVProcessor.reset();
    m_pDXVAVProcessorService.reset();

    m_pD3DDeviceManager9.reset();
    m_pD3DDevice9.reset();
    m_pD3D9.reset();

    AddMessage(RGY_LOG_DEBUG, _T("D3D9Device: Closed.\n"));
    m_pQSVLog.reset();
}

CQSVD3D9Device::~CQSVD3D9Device() {
    Close();
}

mfxStatus CQSVD3D9Device::GetHandle(mfxHandleType type, mfxHDL *pHdl) {
    if (pHdl != nullptr) {
        if (type == MFX_HANDLE_DIRECT3D_DEVICE_MANAGER9) {
            *pHdl = m_pD3DDeviceManager9.get();
            return MFX_ERR_NONE;
        }
        if (type == MFX_HANDLE_IDIRECT3D9EX) {
            *pHdl = m_pD3DDevice9.get();
            return MFX_ERR_NONE;
        }
    }
    return MFX_ERR_UNSUPPORTED;
}

LUID CQSVD3D9Device::GetLUID() {
    return m_devLUID;
}


mfxStatus CQSVD3D9Device::CreateVideoProcessors() {
    m_pDXVAVProcessor.reset();
    AddMessage(RGY_LOG_DEBUG, _T("D3D9Device: CreateVideoProcessors...\n"));

    ZeroMemory(&m_backBufferDesc, sizeof(m_backBufferDesc));
    IDirect3DSurface9 *backBufferTmp = nullptr;
    auto hr = m_pD3DDevice9->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &backBufferTmp);
    if (NULL != backBufferTmp) {
        backBufferTmp->GetDesc(&m_backBufferDesc);
        backBufferTmp->Release();
    }
    if (FAILED(hr)) {
        AddMessage(RGY_LOG_ERROR, _T("D3D9Device: Failed GetBackBuffer: %d.\n"), hr);
        return MFX_ERR_UNDEFINED_BEHAVIOR;
    }
    AddMessage(RGY_LOG_DEBUG, _T("D3D9Device: GetBackBuffer Success.\n"));

    IDirectXVideoProcessor *pDXVAVP = nullptr;
    if (FAILED(hr = DXVA2CreateVideoService(m_pD3DDevice9.get(), IID_IDirectXVideoProcessorService, (void**)&m_pDXVAVProcessorService))) {
        AddMessage(RGY_LOG_ERROR, _T("D3D9Device: Failed DXVA2CreateVideoService: %d.\n"), hr);
        return MFX_ERR_DEVICE_FAILED;
    } else if (FAILED(hr = m_pDXVAVProcessorService->CreateVideoProcessor(DXVA2_VideoProcProgressiveDevice, &m_VideoDesc, m_D3DPresentPrm.BackBufferFormat, 1, &pDXVAVP))) {
        AddMessage(RGY_LOG_ERROR, _T("D3D9Device: Failed CreateVideoProcessor: %d.\n"), hr);
        return MFX_ERR_DEVICE_FAILED;
    }
    m_pDXVAVProcessor.reset(pDXVAVP);
    AddMessage(RGY_LOG_DEBUG, _T("D3D9Device: CreateVideoProcessor Success.\n"));

    return MFX_ERR_NONE;
}

#endif // #if defined(WIN32) || defined(WIN64)
