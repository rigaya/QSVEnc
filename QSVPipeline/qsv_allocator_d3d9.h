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

#ifndef __QSV_ALLOCATOR_D3D9_H__
#define __QSV_ALLOCATOR_D3D9_H__

#if defined( _WIN32 ) || defined ( _WIN64 )

#include <map>
#include <atlbase.h>
#include <d3d9.h>
#include <dxva2api.h>
#include "qsv_allocator.h"

enum eTypeHandle {
    DXVA2_PROCESSOR     = 0x00,
    DXVA2_DECODER       = 0x01
};

struct directxMemId {
    IDirect3DSurface9 *m_surface;
    HANDLE m_handle;
};

class QSVAllocatorParamsD3D9 : public mfxAllocatorParams {
public:
    IDirect3DDeviceManager9 *pManager;
    DWORD surfaceUsage;
    bool getSharedHandle;

    QSVAllocatorParamsD3D9() : pManager(), surfaceUsage(), getSharedHandle(true) { }
};

class QSVAllocatorD3D9 : public QSVAllocator {
public:
    QSVAllocatorD3D9();
    virtual ~QSVAllocatorD3D9();

    virtual mfxStatus Init(mfxAllocatorParams *pParams, std::shared_ptr<RGYLog> pQSVLog) override;
    virtual mfxStatus Close() override;

    virtual IDirect3DDeviceManager9* GetDeviceManager() {
        return m_manager;
    };

    virtual mfxStatus FrameLock(mfxMemId mid, mfxFrameData *ptr) override;
    virtual mfxStatus FrameUnlock(mfxMemId mid, mfxFrameData *ptr) override;
    virtual mfxStatus GetFrameHDL(mfxMemId mid, mfxHDL *handle) override;

protected:
    virtual mfxStatus CheckRequestType(mfxFrameAllocRequest *request) override;
    virtual mfxStatus ReleaseResponse(mfxFrameAllocResponse *response) override;
    virtual mfxStatus AllocImpl(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response) override;

    CComPtr<IDirect3DDeviceManager9> m_manager;
    CComPtr<IDirectXVideoDecoderService> m_decoderService;
    CComPtr<IDirectXVideoProcessorService> m_processorService;
    HANDLE m_hDecoder;
    HANDLE m_hProcessor;
    DWORD m_surfaceUsage;
    bool m_getSharedHandle;
};

#endif // #if defined( _WIN32 ) || defined ( _WIN64 )
#endif // __QSV_ALLOCATOR_D3D9_H__
