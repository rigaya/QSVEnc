//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

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
    CQSVD3D9Device();
    virtual ~CQSVD3D9Device();

    virtual mfxStatus Init(mfxHDL hWindow, uint32_t nAdapterNum, shared_ptr<CQSVLog> pQSVLog) override;
    virtual mfxStatus Reset();
    virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL *pHdl) override;
    virtual void      Close() override;
protected:
    mfxStatus CreateVideoProcessors();
private:
    unique_ptr<IDirect3D9Ex,            IUnknown_release> m_pD3D9;
    unique_ptr<IDirect3DDevice9Ex,      IUnknown_release> m_pD3DDevice9;
    unique_ptr<IDirect3DDeviceManager9, IUnknown_release> m_pD3DDeviceManager9;

    unique_ptr<IDirectXVideoProcessorService, IUnknown_release> m_pDXVAVProcessorService;
    unique_ptr<IDirectXVideoProcessor,        IUnknown_release> m_pDXVAVProcessor;
    
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