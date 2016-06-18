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

#ifndef _QSV_VERSION_H_
#define _QSV_VERSION_H_

#define VER_FILEVERSION             0,2,52,0
#define VER_STR_FILEVERSION          "2.52"
#define VER_STR_FILEVERSION_TCHAR _T("2.52")

#ifdef _M_IX86
#define BUILD_ARCH_STR _T("x86")
#else
#define BUILD_ARCH_STR _T("x64")
#endif

#if _UNICODE
const wchar_t *get_qsvenc_version();
#else
const char *get_qsvenc_version();
#endif

#if defined(_WIN32) || defined(_WIN64)

#define D3D_SURFACES_SUPPORT 1

#define MFX_D3D11_SUPPORT 1

#define ENABLE_FEATURE_COP3_AND_ABOVE 1

#define ENABLE_ADVANCED_DEINTERLACE 0

#define ENABLE_MVC_ENCODING 0

#define ENABLE_FPS_CONVERSION 0

#define ENABLE_OPENCL 1

#define ENABLE_SESSION_THREAD_CONFIG 0

#define ENABLE_AVCODEC_OUT_THREAD 1

#define ENABLE_AVCODEC_AUDPROCESS_THREAD 1

#define ENABLE_CPP_REGEX 1
#define ENABLE_DTL 1

#ifndef QSVENC_AUO
#define ENABLE_AUO_LINK           0
#define ENABLE_AVI_READER         1
#define ENABLE_AVISYNTH_READER    1
#define ENABLE_VAPOURSYNTH_READER 1
#define ENABLE_AVCODEC_QSV_READER 1
#define ENABLE_LIBASS_SUBBURN     1
#define ENABLE_CUSTOM_VPP         1
#ifdef _M_IX86
#define ENABLE_METRIC_FRAMEWORK   0
#else
#define ENABLE_METRIC_FRAMEWORK   1
#endif
#else
#define QSV_AUO_NAME              "QSVEnc.auo"
#define ENABLE_AUO_LINK           1
#define ENABLE_AVI_READER         0
#define ENABLE_AVISYNTH_READER    0
#define ENABLE_VAPOURSYNTH_READER 0
#define ENABLE_AVCODEC_QSV_READER 1
#define ENABLE_CUSTOM_VPP         1
#define ENABLE_LIBASS_SUBBURN     0
#define ENABLE_METRIC_FRAMEWORK   0
#endif

#else //#if defined(WIN32) || defined(WIN64)
#define MFX_D3D11_SUPPORT 0
#define ENABLE_METRIC_FRAMEWORK 0
#include "qsv_config.h"
#endif // #if defined(WIN32) || defined(WIN64)

#endif //_QSV_VERSION_H_
