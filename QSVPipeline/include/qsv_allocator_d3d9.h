//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

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

    QSVAllocatorParamsD3D9() : pManager(), surfaceUsage() { }
};

class QSVAllocatorD3D9 : public QSVAllocator {
public:
    QSVAllocatorD3D9();
    virtual ~QSVAllocatorD3D9();

    virtual mfxStatus Init(mfxAllocatorParams *pParams) override;
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
};

#endif // #if defined( _WIN32 ) || defined ( _WIN64 )
#endif // __QSV_ALLOCATOR_D3D9_H__
