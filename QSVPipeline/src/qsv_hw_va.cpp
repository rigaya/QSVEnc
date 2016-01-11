//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#include "vaapi_device.h"

#if defined(LIBVA_X11_SUPPORT)

#include <va/va_x11.h>
#include <X11/Xlib.h>
#include "vaapi_allocator.h"

#define VAAPI_GET_X_DISPLAY(_display) (Display*)(_display)
#define VAAPI_GET_X_WINDOW(_window) (Window*)(_window)

CQSVHWVADeviceX11::~CQSVHWVADeviceX11(void) {
    Close();
}

mfxStatus CQSVHWVADeviceX11::Init(mfxHDL hWindow, mfxU32 nAdapterNum) {
    mfxStatus mfx_res = MFX_ERR_NONE;
    Window* window = NULL;
    return mfx_res;
}

void CQSVHWVADeviceX11::Close() {
    if (m_window) {
        Display* display = VAAPI_GET_X_DISPLAY(m_X11LibVA.GetXDisplay());
        Window* window = VAAPI_GET_X_WINDOW(m_window);
        XDestroyWindow(display, *window);

        free(m_window);
        m_window = NULL;
    }
}

mfxStatus CQSVHWVADeviceX11::Reset() {
    return MFX_ERR_NONE;
}

mfxStatus CQSVHWVADeviceX11::GetHandle(mfxHandleType type, mfxHDL *pHdl) {
    if ((MFX_HANDLE_VA_DISPLAY == type) && (nullptr != pHdl)) {
        *pHdl = m_X11LibVA.GetVADisplay();

        return MFX_ERR_NONE;
    }
    return MFX_ERR_UNSUPPORTED;
}

mfxStatus CVAAPIDeviceX11::SetHandle(mfxHandleType type, mfxHDL hdl) {
    return MFX_ERR_UNSUPPORTED;
}

mfxStatus CVAAPIDeviceX11::RenderFrame(mfxFrameSurface1 *pSurface, mfxFrameAllocator *pmfxAlloc) {
    return MFX_ERR_UNSUPPORTED;
}

#endif //#if defined(LIBVA_X11_SUPPORT)
