//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#include "qsv_hw_d3d11.h"
#if defined(_WIN32) || defined(_WIN64)
#include "qsv_util.h"

#if MFX_D3D11_SUPPORT

CQSVD3D11Device::CQSVD3D11Device():
    m_nViews(0),
    m_bDefaultStereoEnabled(FALSE),
    m_bIsA2rgb10(FALSE),
    m_HandleWindow(NULL) {
}

CQSVD3D11Device::~CQSVD3D11Device() {
    Close();
}

void CQSVD3D11Device::SetSCD1(DXGI_SWAP_CHAIN_DESC1& scd1) {
    scd1.Width              = 0; // Use automatic sizing
    scd1.Height             = 0;
    scd1.Format             = (m_bIsA2rgb10) ? DXGI_FORMAT_R10G10B10A2_UNORM : DXGI_FORMAT_B8G8R8A8_UNORM;
    scd1.Stereo             = FALSE;
    scd1.SampleDesc.Count   = 1; // Don't use multi-sampling
    scd1.SampleDesc.Quality = 0;
    scd1.BufferUsage        = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd1.BufferCount        = 2; // Use double buffering to minimize latency
    scd1.Scaling            = DXGI_SCALING_STRETCH;
    scd1.SwapEffect         = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    scd1.Flags              = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
}

mfxStatus CQSVD3D11Device::Init(mfxHDL hWindow, uint32_t nAdapterNum) {
    m_HandleWindow = (HWND)hWindow;
    if (   FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory2), (void**)(&m_pDXGIFactory)))
        || FAILED(m_pDXGIFactory->EnumAdapters(nAdapterNum, &m_pAdapter))) {
        return MFX_ERR_DEVICE_FAILED;
    }

    static const D3D_FEATURE_LEVEL FeatureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0
    };
    D3D_FEATURE_LEVEL pFeatureLevelsOut;
    if (FAILED(D3D11CreateDevice(m_pAdapter,
        D3D_DRIVER_TYPE_UNKNOWN, NULL, 0, FeatureLevels, _countof(FeatureLevels),
        D3D11_SDK_VERSION, &m_pD3D11Device, &pFeatureLevelsOut, &m_pD3D11Ctx))) {
        return MFX_ERR_DEVICE_FAILED;
    }

    m_pDXGIDev = m_pD3D11Device;
    m_pDX11VideoDevice = m_pD3D11Device;
    m_pVideoContext = m_pD3D11Ctx;

    if (!m_pDXGIDev.p || !m_pDX11VideoDevice.p || !m_pVideoContext.p) {
        return MFX_ERR_NULL_PTR;
    }

    CComQIPtr<ID3D10Multithread> p_mt(m_pVideoContext);
    if (!p_mt) {
        return MFX_ERR_DEVICE_FAILED;
    }
    p_mt->SetMultithreadProtected(true);

    if (hWindow) {
        if (!m_pDXGIFactory.p) {
            return MFX_ERR_NULL_PTR;
        }
        DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {0};
        SetSCD1(swapChainDesc);
        if (FAILED(m_pDXGIFactory->CreateSwapChainForHwnd(m_pD3D11Device,
            (HWND)hWindow, &swapChainDesc, NULL, NULL,
            reinterpret_cast<IDXGISwapChain1**>(&m_pSwapChain)))) {
            return MFX_ERR_DEVICE_FAILED;
        }
    }

    return MFX_ERR_NONE;
}

mfxStatus CQSVD3D11Device::CreateVideoProcessor(mfxFrameSurface1 *pSurface) {
    if (m_VideoProcessorEnum.p || pSurface == nullptr) {
        return MFX_ERR_NONE;
    }

    D3D11_VIDEO_PROCESSOR_CONTENT_DESC ContentDesc;
    QSV_MEMSET_ZERO(ContentDesc);
    ContentDesc.InputFrameFormat            = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
    ContentDesc.InputFrameRate.Numerator    = 30000;
    ContentDesc.InputFrameRate.Denominator  = 1000;
    ContentDesc.InputWidth                  = pSurface->Info.CropW;
    ContentDesc.InputHeight                 = pSurface->Info.CropH;
    ContentDesc.OutputFrameRate.Numerator   = 30000;
    ContentDesc.OutputFrameRate.Denominator = 1000;
    ContentDesc.OutputWidth                 = pSurface->Info.CropW;
    ContentDesc.OutputHeight                = pSurface->Info.CropH;
    ContentDesc.Usage                       = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;

    if (   FAILED(m_pDX11VideoDevice->CreateVideoProcessorEnumerator(&ContentDesc, &m_VideoProcessorEnum))
        || FAILED(m_pDX11VideoDevice->CreateVideoProcessor(m_VideoProcessorEnum, 0, &m_pVideoProcessor))) {
        return MFX_ERR_DEVICE_FAILED;
    }
    return MFX_ERR_NONE;
}

mfxStatus CQSVD3D11Device::Reset() {
    if (!m_pDXGIFactory.p) {
        return MFX_ERR_NULL_PTR;
    }
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {0};
    SetSCD1(swapChainDesc);
    if (FAILED(m_pDXGIFactory->CreateSwapChainForHwnd(m_pD3D11Device,
        (HWND)m_HandleWindow, &swapChainDesc, NULL, NULL,
        reinterpret_cast<IDXGISwapChain1**>(&m_pSwapChain)))) {
        return MFX_ERR_DEVICE_FAILED;
    }

    return MFX_ERR_NONE;
}

mfxStatus CQSVD3D11Device::GetHandle(mfxHandleType type, mfxHDL *pHdl) {
    if (MFX_HANDLE_D3D11_DEVICE == type) {
        *pHdl = m_pD3D11Device.p;
        return MFX_ERR_NONE;
    }
    return MFX_ERR_UNSUPPORTED;
}

void CQSVD3D11Device::Close() {
    m_HandleWindow = NULL;
}

#endif // #if MFX_D3D11_SUPPORT
#endif // #if defined(_WIN32) || defined(_WIN64)
