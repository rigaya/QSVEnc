//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#ifndef __QSV_HW_DEVICE_H__
#define __QSV_HW_DEVICE_H__

#include <cstdint>
#include "mfxvideo++.h"
#include "qsv_version.h"

#if defined(WIN32) || defined(WIN64)

#if defined(_WIN32) && !defined(MFX_D3D11_SUPPORT)
#include <sdkddkver.h>
#if (NTDDI_VERSION >= NTDDI_VERSION_FROM_WIN32_WINNT2(0x0602)) // >= _WIN32_WINNT_WIN8
#define MFX_D3D11_SUPPORT 1 // Enable D3D11 support if SDK allows
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#else
#define MFX_D3D11_SUPPORT 0
#endif
#endif // #if defined(WIN32) && !defined(MFX_D3D11_SUPPORT)
#endif // #if defined(WIN32) || defined(WIN64)

class CQSVHWDevice {
public:
    CQSVHWDevice() {};
    virtual ~CQSVHWDevice() { }
    virtual mfxStatus Init(mfxHDL hWindow, uint32_t nAdapterNum) = 0;
    virtual mfxStatus Reset() = 0;
    virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL *pHdl) = 0;
    virtual void      Close() = 0;
};

#endif //#ifndef __QSV_HW_DEVICE_H__
