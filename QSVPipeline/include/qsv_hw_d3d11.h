//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#ifndef __QSV_HW_D3D11_H__
#define __QSV_HW_D3D11_H__

#include "qsv_hw_device.h"

#if MFX_D3D11_SUPPORT
#if defined( _WIN32 ) || defined ( _WIN64 )
#pragma warning(disable : 4201)
#include <d3d11.h>
#include <atlbase.h>
#include <windows.h>
#include <dxgi1_2.h>
#include "qsv_util.h"

class CQSVD3D11Device : public CQSVHWDevice {
public:
    CQSVD3D11Device();
    virtual ~CQSVD3D11Device();
    virtual mfxStatus Init(mfxHDL hWindow, uint32_t nAdapterNum) override;
    virtual mfxStatus Reset() override;
    virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL *pHdl) override;
    virtual void      Close() override;
protected:
    void SetSCD1(DXGI_SWAP_CHAIN_DESC1& scd);
    mfxStatus CreateVideoProcessor(mfxFrameSurface1 *pSurface);

    CComPtr<ID3D11Device>                   m_pD3D11Device;
    CComPtr<ID3D11DeviceContext>            m_pD3D11Ctx;
    CComQIPtr<ID3D11VideoDevice>            m_pDX11VideoDevice;
    CComQIPtr<ID3D11VideoContext>           m_pVideoContext;
    CComPtr<ID3D11VideoProcessorEnumerator> m_VideoProcessorEnum;

    CComQIPtr<IDXGIDevice1>                 m_pDXGIDev;
    CComQIPtr<IDXGIAdapter>                 m_pAdapter;

    CComPtr<IDXGIFactory2>                  m_pDXGIFactory;

    CComPtr<IDXGISwapChain1>                m_pSwapChain;
    CComPtr<ID3D11VideoProcessor>           m_pVideoProcessor;

private:
    CComPtr<ID3D11VideoProcessorInputView>  m_pInputViewLeft;
    CComPtr<ID3D11VideoProcessorInputView>  m_pInputViewRight;
    CComPtr<ID3D11VideoProcessorOutputView> m_pOutputView;

    CComPtr<ID3D11Texture2D>                m_pDXGIBackBuffer;
    CComPtr<ID3D11Texture2D>                m_pTempTexture;
    CComPtr<IDXGIDisplayControl>            m_pDisplayControl;
    CComPtr<IDXGIOutput>                    m_pDXGIOutput;
    mfxU16                                  m_nViews;
    BOOL                                    m_bDefaultStereoEnabled;
    BOOL                                    m_bIsA2rgb10;
    HWND                                    m_HandleWindow;
};

#endif //#if defined( _WIN32 ) || defined ( _WIN64 )
#endif //#if MFX_D3D11_SUPPORT
#endif //#ifndef __QSV_HW_D3D11_H__
