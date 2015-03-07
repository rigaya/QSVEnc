//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include "current_date.h"

#ifndef _QSV_VERSION_H_
#define _QSV_VERSION_H_

#ifndef MFX_PRODUCT_VERSION
#define MFX_PRODUCT_VERSION PRODUCT_VERSION_STRING
#endif

#define MSDK_SAMPLE_VERSION MSDK_STRING(MFX_PRODUCT_VERSION)

#define VER_FILEVERSION             0,1,31,0
#define VER_STR_FILEVERSION          "1.31"
#define VER_STR_FILEVERSION_TCHAR _T("1.31")

#ifdef _M_IX86
#define BUILD_ARCH_STR _T("x86")
#else
#define BUILD_ARCH_STR _T("x64")
#endif

#define ENABLE_MVC_ENCODING 0

#define ENABLE_FPS_CONVERSION 0

#define ENABLE_OPENCL 1

#ifndef QSVENC_AUO
#define ENABLE_AVI_READER 1
#define ENABLE_AVISYNTH_READER 1
#define ENABLE_VAPOURSYNTH_READER 1
#else
#define ENABLE_AVI_READER 0
#define ENABLE_AVISYNTH_READER 0
#define ENABLE_VAPOURSYNTH_READER 0
#endif
#endif //_QSV_VERSION_H_
