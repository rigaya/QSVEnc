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

#include "qsv_hw_d3d11.h"
#if defined(_WIN32) || defined(_WIN64)
#include "qsv_util.h"
#include "DeviceId.h"

#if MFX_D3D11_SUPPORT

#define INTEL_VENDOR_ID 0x8086

// The new device dependent counter
#define INTEL_DEVICE_INFO_COUNTERS         "Intel Device Information"

CQSVD3D11Device::CQSVD3D11Device(std::shared_ptr<RGYLog> pQSVLog):
    CQSVHWDevice(pQSVLog),
    m_nViews(0),
    m_bDefaultStereoEnabled(FALSE),
    m_bIsA2rgb10(FALSE),
    m_HandleWindow(NULL),
    m_intelDeviceInfoHeader(),
    m_intelDeviceInfo() {
    m_name = _T("d3d11");
    m_devLUID = LUID();
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

mfxStatus CQSVD3D11Device::Init(mfxHDL hWindow, [[maybe_unused]] uint32_t nViews, uint32_t nAdapterNum) {
    HRESULT hr = 0;
    AddMessage(RGY_LOG_DEBUG, _T("D3D11Device: Init...\n"));
    m_HandleWindow = (HWND)hWindow;
    if (FAILED(hr = CreateDXGIFactory(__uuidof(IDXGIFactory2), (void**)(&m_pDXGIFactory)))) {
        AddMessage(RGY_LOG_ERROR, _T("D3D11Device: CreateDXGIFactory: %d\n"), hr);
        return MFX_ERR_DEVICE_FAILED;
    } else if (FAILED(hr = m_pDXGIFactory->EnumAdapters(nAdapterNum, &m_pAdapter))) {
        AddMessage(RGY_LOG_ERROR, _T("D3D11Device: EnumAdapters: %d\n"), hr);
        return MFX_ERR_DEVICE_FAILED;
    }
    AddMessage(RGY_LOG_DEBUG, _T("D3D11Device: CreateDXGIFactory Success.\n"));

    static const D3D_FEATURE_LEVEL FeatureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0
    };
    D3D_FEATURE_LEVEL pFeatureLevelsOut;
    if (FAILED(hr = D3D11CreateDevice(m_pAdapter,
        D3D_DRIVER_TYPE_UNKNOWN, NULL, 0, FeatureLevels, _countof(FeatureLevels),
        D3D11_SDK_VERSION, &m_pD3D11Device, &pFeatureLevelsOut, &m_pD3D11Ctx))) {
        AddMessage(RGY_LOG_ERROR, _T("D3D11Device: D3D11CreateDevice: %d\n"), hr);
        return MFX_ERR_DEVICE_FAILED;
    }
    AddMessage(RGY_LOG_DEBUG, _T("D3D11Device: D3D11CreateDevice Success.\n"));

    DXGI_ADAPTER_DESC desc;
    m_pAdapter->GetDesc(&desc);
    m_displayDeviceName = desc.Description;
    m_devLUID = desc.AdapterLuid;

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
    AddMessage(RGY_LOG_DEBUG, _T("D3D11Device: SetMultithreadProtected Success.\n"));

    if (hWindow) {
        if (!m_pDXGIFactory.p) {
            return MFX_ERR_NULL_PTR;
        }
        DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {0};
        SetSCD1(swapChainDesc);
        if (FAILED(hr = m_pDXGIFactory->CreateSwapChainForHwnd(m_pD3D11Device,
            (HWND)hWindow, &swapChainDesc, NULL, NULL,
            reinterpret_cast<IDXGISwapChain1**>(&m_pSwapChain)))) {
            AddMessage(RGY_LOG_ERROR, _T("D3D11Device: CreateSwapChainForHwnd: %d\n"), hr);
            return MFX_ERR_DEVICE_FAILED;
        }
        AddMessage(RGY_LOG_DEBUG, _T("D3D11Device: CreateSwapChainForHwnd Success.\n"));
    }
    AddMessage(RGY_LOG_DEBUG, _T("D3D11Device: Init Success.\n"));

    getIntelDeviceInfo();

    return MFX_ERR_NONE;
}

mfxStatus CQSVD3D11Device::getIntelDeviceInfo() {
    HRESULT hr = 0;
    // The counter is in a device dependent counter
    D3D11_COUNTER_INFO counterInfo;
    D3D11_COUNTER_DESC pIntelCounterDesc;
    ZeroMemory(&counterInfo, sizeof(D3D11_COUNTER_INFO));
    ZeroMemory(&pIntelCounterDesc, sizeof(D3D11_COUNTER_DESC));

    // Query the device to find the number of device dependent counters.
    m_pD3D11Device->CheckCounterInfo(&counterInfo);

    if (counterInfo.LastDeviceDependentCounter == 0) {
        return MFX_ERR_NOT_FOUND;
    }

    const int numDependentCounters = counterInfo.LastDeviceDependentCounter - D3D11_COUNTER_DEVICE_DEPENDENT_0 + 1;

    // Search for the apporpriate counter - INTEL_DEVICE_INFO_COUNTERS
    for (int i = 0; i < numDependentCounters; ++i) {
        D3D11_COUNTER_DESC counterDescription;
        counterDescription.Counter = static_cast<D3D11_COUNTER>(i + D3D11_COUNTER_DEVICE_DEPENDENT_0);
        counterDescription.MiscFlags = 0;
        D3D11_COUNTER_TYPE counterType = static_cast<D3D11_COUNTER_TYPE>(0);
        UINT uiSlotsRequired = 0, uiNameLength = 0, uiUnitsLength = 0, uiDescLength = 0;

        if (SUCCEEDED(hr = m_pD3D11Device->CheckCounter(&counterDescription, &counterType, &uiSlotsRequired, NULL, &uiNameLength, NULL, &uiUnitsLength, NULL, &uiDescLength))) {
            std::vector<char> sName(uiNameLength   + 1, 0);
            std::vector<char> sUnits(uiUnitsLength + 1, 0);
            std::vector<char> sDesc(uiDescLength   + 1, 0);

            if (SUCCEEDED(hr = m_pD3D11Device->CheckCounter(&counterDescription, &counterType, &uiSlotsRequired, sName.data(), &uiNameLength, sUnits.data(), &uiUnitsLength, sDesc.data(), &uiDescLength))) {
                if (strcmp(sName.data(), INTEL_DEVICE_INFO_COUNTERS) == 0) {
                    int IntelCounterMajorVersion;
                    int IntelCounterSize;
                    int argsFilled = 0;

                    pIntelCounterDesc.Counter = counterDescription.Counter;

                    argsFilled = sscanf_s(sDesc.data(), "Version %d", &IntelCounterMajorVersion);

                    if (argsFilled != 1 || 1 != sscanf_s(sUnits.data(), "Size %d", &IntelCounterSize)) {
                        // Fall back to version 1.0
                        IntelCounterMajorVersion = 1;
                        IntelCounterSize = sizeof(IntelDeviceInfoV1);
                    }

                    m_intelDeviceInfoHeader = std::make_unique<IntelDeviceInfoHeader>();
                    m_intelDeviceInfoHeader->Version = IntelCounterMajorVersion;
                    m_intelDeviceInfoHeader->Size = IntelCounterSize;
                }
            }
        }
    }

    // Check if device info counter was found
    if (pIntelCounterDesc.Counter == NULL) {
        return MFX_ERR_NOT_FOUND;
    }

    // Intel Device Counter //
    ID3D11Counter *pIntelCounter = NULL;

    // Create the appropriate counter
    hr = m_pD3D11Device->CreateCounter(&pIntelCounterDesc, &pIntelCounter);
    if (FAILED(hr)) {
        return MFX_ERR_NOT_FOUND;
    }

    // Begin and end counter capture
    m_pD3D11Ctx->Begin(pIntelCounter);
    m_pD3D11Ctx->End(pIntelCounter);

    // Check for available data
    hr = m_pD3D11Ctx->GetData(pIntelCounter, NULL, NULL, NULL);
    if (FAILED(hr) || hr == S_FALSE) {
        return MFX_ERR_NOT_FOUND;
    }

    DWORD pData[2] = { 0 };
    // Get pointer to structure
    hr = m_pD3D11Ctx->GetData(pIntelCounter, pData, 2 * sizeof(DWORD), NULL);

    if (FAILED(hr) || hr == S_FALSE) {
        return MFX_ERR_NOT_FOUND;
    }

    //
    // Prepare data to be returned //
    //
    // Copy data to passed in parameter
    void *pDeviceInfoBuffer = *(void**)pData;
    m_intelDeviceInfo = std::make_unique<IntelDeviceInfo>();
    memcpy(m_intelDeviceInfo.get(), pDeviceInfoBuffer, m_intelDeviceInfoHeader->Size);
    return MFX_ERR_NONE;
}

mfxStatus CQSVD3D11Device::CreateVideoProcessor(mfxFrameSurface1 *pSurface) {
    AddMessage(RGY_LOG_DEBUG, _T("D3D11Device: CreateVideoProcessor...\n"));
    if (m_VideoProcessorEnum.p || pSurface == nullptr) {
        return MFX_ERR_NONE;
    }

    D3D11_VIDEO_PROCESSOR_CONTENT_DESC ContentDesc;
    RGY_MEMSET_ZERO(ContentDesc);
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

    HRESULT hr = 0;
    if (FAILED(hr = m_pDX11VideoDevice->CreateVideoProcessorEnumerator(&ContentDesc, &m_VideoProcessorEnum))) {
        AddMessage(RGY_LOG_ERROR, _T("D3D11Device: CreateVideoProcessorEnumerator: %d\n"), hr);
        return MFX_ERR_DEVICE_FAILED;
    } else if (FAILED(hr = m_pDX11VideoDevice->CreateVideoProcessor(m_VideoProcessorEnum, 0, &m_pVideoProcessor))) {
        AddMessage(RGY_LOG_ERROR, _T("D3D11Device: CreateVideoProcessor: %d\n"), hr);
        return MFX_ERR_DEVICE_FAILED;
    }
    AddMessage(RGY_LOG_DEBUG, _T("D3D11Device: CreateVideoProcessor Success.\n"));
    return MFX_ERR_NONE;
}

mfxStatus CQSVD3D11Device::Reset() {
    if (!m_pDXGIFactory.p) {
        return MFX_ERR_NULL_PTR;
    }
    HRESULT hr = 0;
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {0};
    SetSCD1(swapChainDesc);
    if (FAILED(hr = m_pDXGIFactory->CreateSwapChainForHwnd(m_pD3D11Device,
        (HWND)m_HandleWindow, &swapChainDesc, NULL, NULL,
        reinterpret_cast<IDXGISwapChain1**>(&m_pSwapChain)))) {
        AddMessage(RGY_LOG_ERROR, _T("D3D11Device: CreateSwapChainForHwnd: %d\n"), hr);
        return MFX_ERR_DEVICE_FAILED;
    }
    AddMessage(RGY_LOG_DEBUG, _T("D3D11Device: Reset Success.\n"));
    return MFX_ERR_NONE;
}

mfxStatus CQSVD3D11Device::GetHandle(mfxHandleType type, mfxHDL *pHdl) {
    if (MFX_HANDLE_D3D11_DEVICE == type) {
        *pHdl = m_pD3D11Device.p;
        return MFX_ERR_NONE;
    }
    return MFX_ERR_UNSUPPORTED;
}

LUID CQSVD3D11Device::GetLUID() {
    return m_devLUID;
}

tstring CQSVD3D11Device::GetName() {
    return wstring_to_tstring(m_displayDeviceName);
};

void CQSVD3D11Device::Close() {
    m_HandleWindow = NULL;
    AddMessage(RGY_LOG_DEBUG, _T("D3D11Device: Closed.\n"));
    m_pQSVLog.reset();
}

#endif // #if MFX_D3D11_SUPPORT
#endif // #if defined(_WIN32) || defined(_WIN64)
