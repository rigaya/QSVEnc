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

#include "qsv_hw_device.h"
#include <objbase.h>
#include <initguid.h>
#include <assert.h>
#include <d3d9.h>

#include "qsv_allocator_d3d9.h"
#include "qsv_util.h"

#define D3DFMT_NV12 (D3DFORMAT)MAKEFOURCC('N','V','1','2')
#define D3DFMT_YV12 (D3DFORMAT)MAKEFOURCC('Y','V','1','2')
#define D3DFMT_P010 (D3DFORMAT)MAKEFOURCC('P','0','1','0')

static const std::map<mfxU32, D3DFORMAT> fourccToD3DFormat = {
    { MFX_FOURCC_NV12,    D3DFMT_NV12 },
    { MFX_FOURCC_YV12,    D3DFMT_YV12},
    { MFX_FOURCC_YUY2,    D3DFMT_YUY2},
    { MFX_FOURCC_RGB3,    D3DFMT_R8G8B8},
    { MFX_FOURCC_RGB4,    D3DFMT_A8R8G8B8},
    { MFX_FOURCC_P8,      D3DFMT_P8},
    { MFX_FOURCC_P010,    D3DFMT_P010},
    { MFX_FOURCC_A2RGB10, D3DFMT_A2R10G10B10},
};

QSVAllocatorD3D9::QSVAllocatorD3D9() :
    m_decoderService(0), m_processorService(0), m_hDecoder(0), m_hProcessor(0), m_manager(0), m_surfaceUsage(0) {
}

QSVAllocatorD3D9::~QSVAllocatorD3D9() {
    Close();
}

mfxStatus QSVAllocatorD3D9::Init(mfxAllocatorParams *pParams, shared_ptr<RGYLog> pQSVLog) {
    m_pQSVLog = pQSVLog;

    QSVAllocatorParamsD3D9 *pd3d9Params = dynamic_cast<QSVAllocatorParamsD3D9 *>(pParams);
    if (!pd3d9Params)
        return MFX_ERR_NOT_INITIALIZED;

    m_manager = pd3d9Params->pManager;
    m_surfaceUsage = pd3d9Params->surfaceUsage;

    return MFX_ERR_NONE;
}

mfxStatus QSVAllocatorD3D9::Close() {
    if (m_manager && m_hDecoder) {
        m_manager->CloseDeviceHandle(m_hDecoder);
        m_manager = 0;
        m_hDecoder = 0;
    }

    if (m_manager && m_hProcessor) {
        m_manager->CloseDeviceHandle(m_hProcessor);
        m_manager = 0;
        m_hProcessor = 0;
    }

    return QSVAllocator::Close();
}

mfxStatus QSVAllocatorD3D9::FrameLock(mfxMemId mid, mfxFrameData *ptr) {
    if (!ptr || !mid) {
        return MFX_ERR_NULL_PTR;
    }
    IDirect3DSurface9 *pSurface = static_cast<IDirect3DSurface9*>(((mfxHDLPair*)mid)->first);
    if (pSurface == nullptr) {
        return MFX_ERR_INVALID_HANDLE;
    }
    HRESULT hr = 0;
    D3DSURFACE_DESC desc;
    if (FAILED(hr = pSurface->GetDesc(&desc))) {
        m_pQSVLog->write(RGY_LOG_ERROR, _T("QSVAllocatorD3D9::FrameLock Failed GetDesc mid 0x%x: %d\n"), mid, hr);
        return MFX_ERR_LOCK_MEMORY;
    }
    static const auto SUPPORTED_FORMATS = make_array<D3DFORMAT>(
        D3DFMT_NV12,
        D3DFMT_YV12,
        D3DFMT_YUY2,
        D3DFMT_R8G8B8,
        D3DFMT_A8R8G8B8,
        D3DFMT_P8,
        D3DFMT_P010,
        D3DFMT_A2R10G10B10
    );
    if (std::find(SUPPORTED_FORMATS.begin(), SUPPORTED_FORMATS.end(), desc.Format) == SUPPORTED_FORMATS.end()) {
        m_pQSVLog->write(RGY_LOG_ERROR, _T("QSVAllocatorD3D9::Unsupported format.\n"));
        return MFX_ERR_UNSUPPORTED;
    }
    D3DLOCKED_RECT locked;
    if (FAILED(hr = pSurface->LockRect(&locked, 0, D3DLOCK_NOSYSLOCK))) {
        m_pQSVLog->write(RGY_LOG_ERROR, _T("QSVAllocatorD3D9::FrameLock Failed to Lock frame mid 0x%x: %d\n"), mid, hr);
        return MFX_ERR_LOCK_MEMORY;
    }

    switch ((DWORD)desc.Format) {
    case D3DFMT_NV12:
        ptr->Pitch = (mfxU16)locked.Pitch;
        ptr->Y = (mfxU8 *)locked.pBits;
        ptr->U = (mfxU8 *)locked.pBits + desc.Height * locked.Pitch;
        ptr->V = ptr->U + 1;
        break;
    case D3DFMT_YV12:
        ptr->Pitch = (mfxU16)locked.Pitch;
        ptr->Y = (mfxU8 *)locked.pBits;
        ptr->V = ptr->Y + desc.Height * locked.Pitch;
        ptr->U = ptr->V + (desc.Height * locked.Pitch) / 4;
        break;
    case D3DFMT_YUY2:
        ptr->Pitch = (mfxU16)locked.Pitch;
        ptr->Y = (mfxU8 *)locked.pBits;
        ptr->U = ptr->Y + 1;
        ptr->V = ptr->Y + 3;
        break;
    case D3DFMT_R8G8B8:
        ptr->Pitch = (mfxU16)locked.Pitch;
        ptr->B = (mfxU8 *)locked.pBits;
        ptr->G = ptr->B + 1;
        ptr->R = ptr->B + 2;
        break;
    case D3DFMT_A8R8G8B8:
    case D3DFMT_A2R10G10B10:
        ptr->Pitch = (mfxU16)locked.Pitch;
        ptr->B = (mfxU8 *)locked.pBits;
        ptr->G = ptr->B + 1;
        ptr->R = ptr->B + 2;
        ptr->A = ptr->B + 3;
        break;
    case D3DFMT_P8:
        ptr->Pitch = (mfxU16)locked.Pitch;
        ptr->Y = (mfxU8 *)locked.pBits;
        ptr->U = 0;
        ptr->V = 0;
        break;
    case D3DFMT_P010:
        ptr->PitchHigh = (mfxU16)(locked.Pitch >> 16);
        ptr->PitchLow = (mfxU16)(locked.Pitch & (0xffff));
        ptr->Y = (mfxU8 *)locked.pBits;
        ptr->U = (mfxU8 *)locked.pBits + desc.Height * locked.Pitch;
        ptr->V = ptr->U + 1;
        break;
    }

    m_pQSVLog->write(RGY_LOG_TRACE, _T("QSVAllocatorD3D9::FrameLock Success.\n"));
    return MFX_ERR_NONE;
}

mfxStatus QSVAllocatorD3D9::FrameUnlock(mfxMemId mid, mfxFrameData *ptr) {
    if (!mid) {
        return MFX_ERR_NULL_PTR;
    }
    IDirect3DSurface9 *pSurface = static_cast<IDirect3DSurface9*>(((mfxHDLPair*)mid)->first);
    if (pSurface == nullptr) {
        return MFX_ERR_INVALID_HANDLE;
    }
    pSurface->UnlockRect();

    if (ptr != nullptr) {
        ptr->Pitch = 0;
        ptr->Y     = nullptr;
        ptr->U     = nullptr;
        ptr->V     = nullptr;
    }
    m_pQSVLog->write(RGY_LOG_TRACE, _T("QSVAllocatorD3D9::FrameUnlock Success.\n"));
    return MFX_ERR_NONE;
}

mfxStatus QSVAllocatorD3D9::GetFrameHDL(mfxMemId mid, mfxHDL *handle) {
    if (!mid || !handle) {
        return MFX_ERR_NULL_PTR;
    }
    *handle = ((mfxHDLPair*)mid)->first;
    return MFX_ERR_NONE;
}

mfxStatus QSVAllocatorD3D9::CheckRequestType(mfxFrameAllocRequest *request) {
    mfxStatus sts = QSVAllocator::CheckRequestType(request);
    if (MFX_ERR_NONE != sts) {
        return sts;
    }

    return ((request->Type & (MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET | MFX_MEMTYPE_VIDEO_MEMORY_PROCESSOR_TARGET)) != 0) ?
        MFX_ERR_NONE : MFX_ERR_UNSUPPORTED;
}

mfxStatus QSVAllocatorD3D9::ReleaseResponse(mfxFrameAllocResponse *response) {
    if (!response) {
        return MFX_ERR_NULL_PTR;
    }
    if (response->mids) {
        for (mfxU32 i = 0; i < response->NumFrameActual; i++) {
            if (response->mids[i]) {
                directxMemId *dxMids = (directxMemId*)response->mids[i];
                dxMids->m_surface->Release();
            }
        }
        rgy_free(response->mids[0]);
    }
    rgy_free(response->mids);
    m_pQSVLog->write(RGY_LOG_DEBUG, _T("QSVAllocatorD3D9::ReleaseResponse Success.\n"));
    return MFX_ERR_NONE;
}

mfxStatus QSVAllocatorD3D9::AllocImpl(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response) {
    if (request->NumFrameSuggested == 0) {
        return MFX_ERR_UNKNOWN;
    }

    if (fourccToD3DFormat.find(request->Info.FourCC) == fourccToD3DFormat.end()) {
        m_pQSVLog->write(RGY_LOG_ERROR, _T("QSVAllocatorD3D9::AllocImpl unsupported format.\n"));
        return MFX_ERR_UNSUPPORTED;
    }

    D3DFORMAT format = fourccToD3DFormat.at(request->Info.FourCC);
    DWORD target;
    if (MFX_MEMTYPE_DXVA2_DECODER_TARGET & request->Type) {
        m_pQSVLog->write(RGY_LOG_DEBUG, _T("QSVAllocatorD3D9::AllocImpl select DXVA2_VideoDecoderRenderTarget.\n"));
        target = DXVA2_VideoDecoderRenderTarget;
    } else if (MFX_MEMTYPE_DXVA2_PROCESSOR_TARGET & request->Type) {
        m_pQSVLog->write(RGY_LOG_DEBUG, _T("QSVAllocatorD3D9::AllocImpl select DXVA2_VideoProcessorRenderTarget.\n"));
        target = DXVA2_VideoProcessorRenderTarget;
    } else {
        m_pQSVLog->write(RGY_LOG_DEBUG, _T("QSVAllocatorD3D9::AllocImpl unsupported mem type.\n"));
        return MFX_ERR_UNSUPPORTED;
    }

    HRESULT hr = 0;
    IDirectXVideoAccelerationService *videoService = nullptr;

    if (target == DXVA2_VideoProcessorRenderTarget) {
        if (!m_hProcessor) {
            if (FAILED(hr = m_manager->OpenDeviceHandle(&m_hProcessor))) {
                m_pQSVLog->write(RGY_LOG_ERROR, _T("QSVAllocatorD3D9::AllocImpl failed OpenDeviceHandle(Processor): %d.\n"), hr);
                return MFX_ERR_DEVICE_FAILED;
            } else if (FAILED(hr = m_manager->GetVideoService(m_hProcessor, IID_IDirectXVideoProcessorService, (void**)&m_processorService))) {
                m_pQSVLog->write(RGY_LOG_ERROR, _T("QSVAllocatorD3D9::AllocImpl failed GetVideoService(Decoder): %d.\n"), hr);
                return MFX_ERR_DEVICE_FAILED;
            }
        }
        videoService = m_processorService;
    } else {
        if (!m_hDecoder) {
            if (FAILED(hr = m_manager->OpenDeviceHandle(&m_hDecoder))) {
                m_pQSVLog->write(RGY_LOG_ERROR, _T("QSVAllocatorD3D9::AllocImpl failed OpenDeviceHandle(Decoder): %d.\n"), hr);
                return MFX_ERR_DEVICE_FAILED;
            } else if (FAILED(hr = m_manager->GetVideoService(m_hDecoder, IID_IDirectXVideoDecoderService, (void**)&m_decoderService))) {
                m_pQSVLog->write(RGY_LOG_ERROR, _T("QSVAllocatorD3D9::AllocImpl failed GetVideoService(Decoder): %d.\n"), hr);
                return MFX_ERR_DEVICE_FAILED;
            }
        }
        videoService = m_decoderService;
    }
    m_pQSVLog->write(RGY_LOG_DEBUG, _T("QSVAllocatorD3D9::AllocImpl GetVideoService Success.\n"));

    mfxHDLPair  *dxMids    = (mfxHDLPair *)calloc(request->NumFrameSuggested, sizeof(mfxHDLPair));
    mfxHDLPair **dxMidPtrs = (mfxHDLPair**)calloc(request->NumFrameSuggested, sizeof(mfxHDLPair*));

    if (!dxMids || !dxMidPtrs) {
        rgy_free(dxMids);
        rgy_free(dxMidPtrs);
        return MFX_ERR_MEMORY_ALLOC;
    }

    m_pQSVLog->write(RGY_LOG_DEBUG, _T("QSVAllocatorD3D9::AllocImpl allocate surface...\n"));
    response->mids = (mfxMemId*)dxMidPtrs;
    response->NumFrameActual = request->NumFrameSuggested;

    if (request->Type & MFX_MEMTYPE_EXTERNAL_FRAME) {
        for (int i = 0; i < request->NumFrameSuggested; i++) {
            if (FAILED(hr = videoService->CreateSurface(request->Info.Width, request->Info.Height, 0,  format,
                                                D3DPOOL_DEFAULT, m_surfaceUsage, target, (IDirect3DSurface9**)&dxMids[i].first, NULL /*&dxMids[i].second*/))) {
                ReleaseResponse(response);
                rgy_free(dxMids);
                m_pQSVLog->write(RGY_LOG_ERROR, _T("QSVAllocatorD3D9::AllocImpl failed to CreateSurface(external) #%d: %d.\n"), i, hr);
                return MFX_ERR_MEMORY_ALLOC;
            }
            dxMidPtrs[i] = &dxMids[i];
        }
    } else {
        unique_ptr<IDirect3DSurface9*> dxSrf(new IDirect3DSurface9*[request->NumFrameSuggested]);
        if (!dxSrf.get()) {
            rgy_free(dxMids);
            return MFX_ERR_MEMORY_ALLOC;
        }
        if (FAILED(hr = videoService->CreateSurface(request->Info.Width, request->Info.Height, request->NumFrameSuggested - 1,  format,
                                            D3DPOOL_DEFAULT, m_surfaceUsage, target, dxSrf.get(), NULL))) { 
            rgy_free(dxMids);
            m_pQSVLog->write(RGY_LOG_ERROR, _T("QSVAllocatorD3D9::AllocImpl failed to CreateSurface(other) %d frmaes: %d.\n"), request->NumFrameSuggested - 1, hr);
            return MFX_ERR_MEMORY_ALLOC;
        }
        for (int i = 0; i < request->NumFrameSuggested; i++) {
            dxMids[i].first = dxSrf.get()[i];
            dxMidPtrs[i] = &dxMids[i];
        }
    }
    m_pQSVLog->write(RGY_LOG_DEBUG, _T("QSVAllocatorD3D9::AllocImpl Success.\n"));
    return MFX_ERR_NONE;
}

#endif // #if defined(_WIN32) || defined(_WIN64)
