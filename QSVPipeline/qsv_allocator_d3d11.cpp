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


#if defined(_WIN32) || defined(_WIN64)

#include "rgy_tchar.h"
#include "qsv_hw_device.h"

#if MFX_D3D11_SUPPORT

#include <objbase.h>
#include <initguid.h>
#include <assert.h>
#include <algorithm>
#include <functional>
#include <iterator>
#include "qsv_allocator_d3d11.h"
#include "qsv_util.h"


#define D3DFMT_NV12 (DXGI_FORMAT)MAKEFOURCC('N','V','1','2')
#define D3DFMT_YV12 (DXGI_FORMAT)MAKEFOURCC('Y','V','1','2')

static const std::map<mfxU32, DXGI_FORMAT> fourccToDXGIFormat = {
    { MFX_FOURCC_NV12,       DXGI_FORMAT_NV12 },
    { MFX_FOURCC_YUY2,       DXGI_FORMAT_YUY2 },
    { MFX_FOURCC_RGB4,       DXGI_FORMAT_B8G8R8A8_UNORM },
    { MFX_FOURCC_P8,         DXGI_FORMAT_P8 },
    { MFX_FOURCC_P8_TEXTURE, DXGI_FORMAT_P8 },
    { MFX_FOURCC_P010,       DXGI_FORMAT_P010 },
    { MFX_FOURCC_A2RGB10,    DXGI_FORMAT_R10G10B10A2_UNORM },
    { MFX_FOURCC_AYUV,       DXGI_FORMAT_AYUV },
    { DXGI_FORMAT_AYUV,      DXGI_FORMAT_AYUV },
#if (MFX_VERSION >= 1027)
    { MFX_FOURCC_Y210,       DXGI_FORMAT_Y210 },
    { MFX_FOURCC_Y410,       DXGI_FORMAT_Y410 },
#endif
#if (MFX_VERSION >= 1031)
    { MFX_FOURCC_P016,       DXGI_FORMAT_P016 },
    { MFX_FOURCC_Y216,       DXGI_FORMAT_Y216 },
    { MFX_FOURCC_Y416,       DXGI_FORMAT_Y416 },
#endif
};

QSVAllocatorD3D11::QSVAllocatorD3D11() {
    m_pDeviceContext = nullptr;
    m_name = _T("allocD3D11");
}

QSVAllocatorD3D11::~QSVAllocatorD3D11() {
    Close();
}

QSVAllocatorD3D11::TextureSubResource QSVAllocatorD3D11::GetResourceFromMid(mfxMemId mid) {
    size_t index = (size_t)MFXReadWriteMid(mid).raw() - 1;

    if (m_memIdMap.size() <= index) {
        return TextureSubResource();
    }
    TextureResource *p = &(*m_memIdMap[index]);
    if (!p->bAlloc) {
        return TextureSubResource();
    }
    return TextureSubResource(p, mid);
}

mfxStatus QSVAllocatorD3D11::Init(mfxAllocatorParams *pParams, shared_ptr<RGYLog> pQSVLog) {
    m_pQSVLog = pQSVLog;

    QSVAllocatorParamsD3D11 *pd3d11Params = dynamic_cast<QSVAllocatorParamsD3D11 *>(pParams);
    if (NULL == pd3d11Params ||
        NULL == pd3d11Params->pDevice) {
        return MFX_ERR_NOT_INITIALIZED;
    }

    m_initParams = *pd3d11Params;
    IUnknownSafeRelease(m_pDeviceContext);
    pd3d11Params->pDevice->GetImmediateContext(&m_pDeviceContext);

    return MFX_ERR_NONE;
}

mfxStatus QSVAllocatorD3D11::Close() {
    mfxStatus sts = QSVAllocator::Close();
    for (auto it : m_resourcesByRequest) {
        it.Release();
    }
    m_resourcesByRequest.clear();
    m_memIdMap.clear();
    IUnknownSafeRelease(m_pDeviceContext);
    return sts;
}

mfxStatus QSVAllocatorD3D11::FrameLock(mfxMemId mid, mfxFrameData *ptr) {
    TextureSubResource sr = GetResourceFromMid(mid);
    if (!sr.GetTexture()) {
        return MFX_ERR_LOCK_MEMORY;
    }

    HRESULT hRes = S_OK;
    D3D11_MAP mapType = D3D11_MAP_READ;
    UINT mapFlags = D3D11_MAP_FLAG_DO_NOT_WAIT;
    D3D11_TEXTURE2D_DESC desc = {0};
    D3D11_MAPPED_SUBRESOURCE lockedRect = {0};
    if (NULL == sr.GetStaging()) {
        hRes = m_pDeviceContext->Map(sr.GetTexture(), sr.GetSubResource(), D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &lockedRect);
        desc.Format = DXGI_FORMAT_P8;

        if (FAILED(hRes)) {
            AddMessage(RGY_LOG_ERROR, _T("QSVAllocatorD3D11::FrameLock failed to map surface contxt with subResource: %d.\n"), hRes);
            return MFX_ERR_LOCK_MEMORY;
        }
    } else {
        sr.GetTexture()->GetDesc(&desc);
        static const auto SUPPORTED_FORMATS = make_array<DXGI_FORMAT>(
            DXGI_FORMAT_NV12,
            DXGI_FORMAT_420_OPAQUE,
            DXGI_FORMAT_YUY2,
            DXGI_FORMAT_P8,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            DXGI_FORMAT_R16_UINT,
            DXGI_FORMAT_R16_UNORM,
            DXGI_FORMAT_R10G10B10A2_UNORM,
            DXGI_FORMAT_R16G16B16A16_UNORM,
            DXGI_FORMAT_P010,
#if (MFX_VERSION >= 1027)
            DXGI_FORMAT_Y210,
            DXGI_FORMAT_Y410,
#endif
#if (MFX_VERSION >= 1031)
            DXGI_FORMAT_P016,
            DXGI_FORMAT_Y216,
            DXGI_FORMAT_Y416,
#endif
            DXGI_FORMAT_AYUV
            );
        if (std::find(SUPPORTED_FORMATS.begin(), SUPPORTED_FORMATS.end(), desc.Format) == SUPPORTED_FORMATS.end()) {
            AddMessage(RGY_LOG_ERROR, _T("QSVAllocatorD3D11::FrameLock unsupported format.\n"));
            return MFX_ERR_UNSUPPORTED;
        }

        if (MFXReadWriteMid(mid, MFXReadWriteMid::reuse).isRead()) {
            m_pDeviceContext->CopySubresourceRegion(sr.GetStaging(), 0, 0, 0, 0, sr.GetTexture(), sr.GetSubResource(), NULL);
        }

        do {
            hRes = m_pDeviceContext->Map(sr.GetStaging(), 0, mapType, mapFlags, &lockedRect);
            if (S_OK != hRes && DXGI_ERROR_WAS_STILL_DRAWING != hRes) {
                break;
            }
        } while (DXGI_ERROR_WAS_STILL_DRAWING == hRes);

        if (FAILED(hRes)) {
            AddMessage(RGY_LOG_ERROR, _T("QSVAllocatorD3D11::FrameLock failed to map surface: %d.\n"), hRes);
            return MFX_ERR_LOCK_MEMORY;
        }
    }

    switch (desc.Format) {
        case DXGI_FORMAT_P010:
#if (MFX_VERSION >= 1031)
        case DXGI_FORMAT_P016:
#endif
        case DXGI_FORMAT_NV12:
            ptr->Pitch = (mfxU16)lockedRect.RowPitch;
            ptr->Y = (mfxU8 *)lockedRect.pData;
            ptr->U = (mfxU8 *)lockedRect.pData + desc.Height * lockedRect.RowPitch;
            ptr->V = (desc.Format == DXGI_FORMAT_P010) ? ptr->U + 2 : ptr->U + 1;
            break;
        case DXGI_FORMAT_420_OPAQUE: // can be unsupported by standard ms guid
            ptr->Pitch = (mfxU16)lockedRect.RowPitch;
            ptr->Y = (mfxU8 *)lockedRect.pData;
            ptr->V = ptr->Y + desc.Height * lockedRect.RowPitch;
            ptr->U = ptr->V + (desc.Height * lockedRect.RowPitch) / 4;
            break;
        case DXGI_FORMAT_YUY2:
            ptr->Pitch = (mfxU16)lockedRect.RowPitch;
            ptr->Y = (mfxU8 *)lockedRect.pData;
            ptr->U = ptr->Y + 1;
            ptr->V = ptr->Y + 3;
            break;
        case DXGI_FORMAT_P8 :
            ptr->Pitch = (mfxU16)lockedRect.RowPitch;
            ptr->Y = (mfxU8 *)lockedRect.pData;
            ptr->U = 0;
            ptr->V = 0;
            break;
        case DXGI_FORMAT_AYUV:
        case DXGI_FORMAT_B8G8R8A8_UNORM:
            ptr->Pitch = (mfxU16)lockedRect.RowPitch;
            ptr->B = (mfxU8 *)lockedRect.pData;
            ptr->G = ptr->B + 1;
            ptr->R = ptr->B + 2;
            ptr->A = ptr->B + 3;
            break;
        case DXGI_FORMAT_R10G10B10A2_UNORM :
            ptr->Pitch = (mfxU16)lockedRect.RowPitch;
            ptr->B = (mfxU8 *)lockedRect.pData;
            ptr->G = ptr->B + 1;
            ptr->R = ptr->B + 2;
            ptr->A = ptr->B + 3;
            break;
        case DXGI_FORMAT_R16G16B16A16_UNORM:
            ptr->V16 = (mfxU16*)lockedRect.pData;
            ptr->U16 = ptr->V16 + 1;
            ptr->Y16 = ptr->V16 + 2;
            ptr->A = (mfxU8*)(ptr->V16 + 3);
            ptr->PitchHigh = (mfxU16)((mfxU32)lockedRect.RowPitch / (1 << 16));
            ptr->PitchLow  = (mfxU16)((mfxU32)lockedRect.RowPitch % (1 << 16));
            break;
        case DXGI_FORMAT_R16_UNORM :
        case DXGI_FORMAT_R16_UINT :
            ptr->Pitch = (mfxU16)lockedRect.RowPitch;
            ptr->Y16 = (mfxU16 *)lockedRect.pData;
            ptr->U16 = 0;
            ptr->V16 = 0;
            break;
#if (MFX_VERSION >= 1031)
        case DXGI_FORMAT_Y416:
            ptr->PitchHigh = (mfxU16)(lockedRect.RowPitch / (1 << 16));
            ptr->PitchLow = (mfxU16)(lockedRect.RowPitch % (1 << 16));
            ptr->U16 = (mfxU16*)lockedRect.pData;
            ptr->Y16 = ptr->U16 + 1;
            ptr->V16 = ptr->Y16 + 1;
            ptr->A = (mfxU8 *)(ptr->V16 + 1);
            break;
        case DXGI_FORMAT_Y216:
#endif
#if (MFX_VERSION >= 1027)
        case DXGI_FORMAT_Y210:
            ptr->PitchHigh = (mfxU16)(lockedRect.RowPitch / (1 << 16));
            ptr->PitchLow = (mfxU16)(lockedRect.RowPitch % (1 << 16));
            ptr->Y16 = (mfxU16 *)lockedRect.pData;
            ptr->U16 = ptr->Y16 + 1;
            ptr->V16 = ptr->Y16 + 3;

            break;

        case DXGI_FORMAT_Y410:
            ptr->PitchHigh = (mfxU16)(lockedRect.RowPitch / (1 << 16));
            ptr->PitchLow = (mfxU16)(lockedRect.RowPitch % (1 << 16));
            ptr->Y410 = (mfxY410 *)lockedRect.pData;
            ptr->Y = 0;
            ptr->V = 0;
            ptr->A = 0;

            break;
#endif
        default:
            return MFX_ERR_LOCK_MEMORY;
    }
    AddMessage(RGY_LOG_TRACE, _T("QSVAllocatorD3D11::FrameLock success.\n"));
    return MFX_ERR_NONE;
}

mfxStatus QSVAllocatorD3D11::FrameUnlock(mfxMemId mid, mfxFrameData *ptr) {
    TextureSubResource sr = GetResourceFromMid(mid);
    if (!sr.GetTexture()) {
        return MFX_ERR_LOCK_MEMORY;
    }

    if (NULL == sr.GetStaging()) {
        m_pDeviceContext->Unmap(sr.GetTexture(), sr.GetSubResource());
    } else {
        m_pDeviceContext->Unmap(sr.GetStaging(), 0);
        if (MFXReadWriteMid(mid, MFXReadWriteMid::reuse).isWrite()) {
            m_pDeviceContext->CopySubresourceRegion(sr.GetTexture(), sr.GetSubResource(), 0, 0, 0, sr.GetStaging(), 0, NULL);
        }
    }

    if (ptr) {
        ptr->Pitch = 0;
        ptr->Y     = nullptr;
        ptr->U     = nullptr;
        ptr->V     = nullptr;
        ptr->A     = nullptr;
        ptr->R     = nullptr;
        ptr->G     = nullptr;
        ptr->B     = nullptr;
    }
    AddMessage(RGY_LOG_TRACE, _T("QSVAllocatorD3D11::FrameUnlock success.\n"));
    return MFX_ERR_NONE;
}


mfxStatus QSVAllocatorD3D11::GetFrameHDL(mfxMemId mid, mfxHDL *handle) {
    if (NULL == handle) {
        return MFX_ERR_INVALID_HANDLE;
    }

    TextureSubResource sr = GetResourceFromMid(mid);
    if (!sr.GetTexture()) {
        return MFX_ERR_INVALID_HANDLE;
    }

    mfxHDLPair *pPair  = (mfxHDLPair*)handle;
    pPair->first  = sr.GetTexture();
    pPair->second = (mfxHDL)(UINT_PTR)sr.GetSubResource();

    return MFX_ERR_NONE;
}

mfxStatus QSVAllocatorD3D11::CheckRequestType(mfxFrameAllocRequest *request) {
    mfxStatus sts = QSVAllocator::CheckRequestType(request);
    if (MFX_ERR_NONE != sts) {
        return sts;
    }

    return ((request->Type & (MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET | MFX_MEMTYPE_VIDEO_MEMORY_PROCESSOR_TARGET)) != 0) ?
        MFX_ERR_NONE : MFX_ERR_UNSUPPORTED;
}

mfxStatus QSVAllocatorD3D11::ReleaseResponse(mfxFrameAllocResponse *response) {
    if (NULL == response) {
        return MFX_ERR_NULL_PTR;
    }

    if (response->mids && 0 != response->NumFrameActual) {
        TextureSubResource sr = GetResourceFromMid(response->mids[0]);
        if (!sr.GetTexture()) {
            return MFX_ERR_NULL_PTR;
        }
        sr.Release();

        if (m_resourcesByRequest.end() == std::find_if(m_resourcesByRequest.begin(), m_resourcesByRequest.end(), TextureResource::isAllocated)) {
            m_resourcesByRequest.clear();
            m_memIdMap.clear();
        }
    }
    AddMessage(RGY_LOG_TRACE, _T("QSVAllocatorD3D11::ReleaseResponse success.\n"));
    return MFX_ERR_NONE;
}
mfxStatus QSVAllocatorD3D11::AllocImpl(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response) {
    if (fourccToDXGIFormat.find(request->Info.FourCC) == fourccToDXGIFormat.end()) {
        AddMessage(RGY_LOG_ERROR, _T("QSVAllocatorD3D11::AllocImpl unsupported format.\n"));
        return MFX_ERR_UNSUPPORTED;
    }
    const DXGI_FORMAT colorFormat = fourccToDXGIFormat.at(request->Info.FourCC);

    HRESULT hr = 0;
    TextureResource newTexture;
    if (request->Info.FourCC == MFX_FOURCC_P8) {
        D3D11_BUFFER_DESC desc ={ 0 };
        desc.ByteWidth           = request->Info.Width * request->Info.Height;
        desc.Usage               = D3D11_USAGE_STAGING;
        desc.BindFlags           = 0;
        desc.CPUAccessFlags      = D3D11_CPU_ACCESS_READ;
        desc.MiscFlags           = 0;
        desc.StructureByteStride = 0;

        ID3D11Buffer *buffer = nullptr;
        if (FAILED(hr = m_initParams.pDevice->CreateBuffer(&desc, 0, &buffer))) {
            AddMessage(RGY_LOG_ERROR, _T("QSVAllocatorD3D11::AllocImpl failed to create buffer: %d.\n"), hr);
            return MFX_ERR_MEMORY_ALLOC;
        }
        newTexture.textures.push_back(reinterpret_cast<ID3D11Texture2D *>(buffer));
    } else {
        D3D11_TEXTURE2D_DESC desc = {0};
        desc.Width = request->Info.Width;
        desc.Height =  request->Info.Height;

        desc.MipLevels = 1;
        //number of subresources is 1 in case of not single texture
        desc.ArraySize = m_initParams.bUseSingleTexture ? request->NumFrameSuggested : 1;
        desc.Format = colorFormat;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.MiscFlags = m_initParams.uncompressedResourceMiscFlags | D3D11_RESOURCE_MISC_SHARED;

        desc.BindFlags = D3D11_BIND_DECODER;

        if ( (MFX_MEMTYPE_FROM_VPPIN & request->Type) && (DXGI_FORMAT_YUY2 == desc.Format) ||
             (DXGI_FORMAT_B8G8R8A8_UNORM == desc.Format) ||
             (DXGI_FORMAT_R10G10B10A2_UNORM == desc.Format) ||
             (DXGI_FORMAT_R16G16B16A16_UNORM == desc.Format)) {
            desc.BindFlags = D3D11_BIND_RENDER_TARGET;
            AddMessage(RGY_LOG_DEBUG, _T("QSVAllocatorD3D11::AllocImpl set D3D11_BIND_RENDER_TARGET.\n"));
            if (desc.ArraySize > 2) {
                return MFX_ERR_MEMORY_ALLOC;
            }
        }

        if ( (MFX_MEMTYPE_FROM_VPPOUT & request->Type) ||
             (MFX_MEMTYPE_VIDEO_MEMORY_PROCESSOR_TARGET & request->Type)) {
            desc.BindFlags = D3D11_BIND_RENDER_TARGET;
            AddMessage(RGY_LOG_DEBUG, _T("QSVAllocatorD3D11::AllocImpl set D3D11_BIND_RENDER_TARGET.\n"));
            if (desc.ArraySize > 2) {
                return MFX_ERR_MEMORY_ALLOC;
            }
        }

        if (DXGI_FORMAT_P8 == desc.Format) {
            desc.BindFlags = 0;
        }

        AddMessage(RGY_LOG_DEBUG, _T("QSVAllocatorD3D11::AllocImpl create %d textures, %d staging textures (ArraySize: %d).\n"),
            request->NumFrameSuggested / desc.ArraySize, request->NumFrameSuggested, desc.ArraySize);
        ID3D11Texture2D *pTexture2D = nullptr;

        for (size_t i = 0; i < request->NumFrameSuggested / desc.ArraySize; i++) {
            if (FAILED(hr = m_initParams.pDevice->CreateTexture2D(&desc, NULL, &pTexture2D))) {
                AddMessage(RGY_LOG_ERROR, _T("QSVAllocatorD3D11::AllocImpl failed to CreateTexture2D(textures) #%d: %d.\n"), i, hr);
                return MFX_ERR_MEMORY_ALLOC;
            }
            newTexture.textures.push_back(pTexture2D);
        }

        desc.ArraySize = 1;
        desc.Usage = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        desc.BindFlags = 0;
        desc.MiscFlags = 0;

        for (size_t i = 0; i < request->NumFrameSuggested; i++) {
            if (FAILED(m_initParams.pDevice->CreateTexture2D(&desc, NULL, &pTexture2D))) {
                AddMessage(RGY_LOG_ERROR, _T("QSVAllocatorD3D11::AllocImpl failed to CreateTexture2D(stagingTexture) #%d: %d.\n"), i, hr);
                return MFX_ERR_MEMORY_ALLOC;
            }
            newTexture.stagingTexture.push_back(pTexture2D);
        }
        AddMessage(RGY_LOG_DEBUG, _T("QSVAllocatorD3D11::AllocImpl CreateTexture Success.\n"));
    }

    mfxHDL curId = m_resourcesByRequest.empty() ? 0 :  m_resourcesByRequest.back().outerMids.back();
    auto id_init = [&curId]() {
        auto x = curId;
        curId = (mfxHDL)((size_t)(curId)+1);
        return x;
    };
    id_init();
    std::generate_n(std::back_inserter(newTexture.outerMids), request->NumFrameSuggested, id_init);
    m_resourcesByRequest.push_back(newTexture);
    response->mids = &m_resourcesByRequest.back().outerMids.front();
    response->NumFrameActual = request->NumFrameSuggested;
    auto it_last = m_resourcesByRequest.end();
    std::fill_n(std::back_inserter(m_memIdMap), request->NumFrameSuggested, --it_last);
    AddMessage(RGY_LOG_DEBUG, _T("QSVAllocatorD3D11::AllocImpl Success.\n"));
    return MFX_ERR_NONE;
}

#endif // #if MFX_D3D11_SUPPORT
#endif // #if defined(_WIN32) || defined(_WIN64)
