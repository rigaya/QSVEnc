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

#define VER_FILEVERSION             0,1,14,0
#define VER_STR_FILEVERSION          "1.14"
#define VER_STR_FILEVERSION_TCHAR _T("1.14")

#define ENABLE_MVC_ENCODING 0

//defined(_M_IX86)は32bitであることのチェック
#if (defined(_M_IX86) && !defined(QSVENC_AUO))
#define ENABLE_VAPOURSYNTH_READER 1
#else
#define ENABLE_VAPOURSYNTH_READER 0
#endif

#ifndef QSVENC_AUO
#define ENABLE_AVISYNTH_READER 1
#define ENABLE_AVI_READER 1
#else
#define ENABLE_AVISYNTH_READER 0
#define ENABLE_AVI_READER 0
#endif
#endif //_QSV_VERSION_H_
