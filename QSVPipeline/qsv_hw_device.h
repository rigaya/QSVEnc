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
#include <memory>
#include "mfxvideo++.h"
#include "qsv_log.h"
#include "qsv_version.h"

#if MFX_D3D11_SUPPORT
#include <sdkddkver.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#endif //#if MFX_D3D11_SUPPORT

class CQSVHWDevice {
public:
    CQSVHWDevice() {};
    virtual ~CQSVHWDevice() { }
    virtual mfxStatus Init(mfxHDL hWindow, uint32_t nAdapterNum, shared_ptr<CQSVLog> pQSVLog) = 0;
    virtual mfxStatus Reset() = 0;
    virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL *pHdl) = 0;
    virtual void      Close() = 0;
protected:
    std::shared_ptr<CQSVLog> m_pQSVLog;
};

#endif //#ifndef __QSV_HW_DEVICE_H__
