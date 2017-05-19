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

#ifndef _QSVENCC_VERSION_H_
#define _QSVENCC_VERSION_H_

#include "rgy_version.h"

#ifdef DEBUG
#define VER_DEBUG   VS_FF_DEBUG
#define VER_PRIVATE VS_FF_PRIVATEBUILD
#else
#define VER_DEBUG   0
#define VER_PRIVATE 0
#endif

#ifdef _M_IX86
#define QSVENC_FILENAME "QSVEncC (x86) - QuickSyncVideo Encoder (CUI)"
#else
#define QSVENC_FILENAME "QSVEncC (x64) - QuickSyncVideo Encoder (CUI)"
#endif

#define VER_STR_COMMENTS         "based on Intel Media SDK Sample"
#define VER_STR_COMPANYNAME      ""
#define VER_STR_FILEDESCRIPTION  QSVENC_FILENAME
#define VER_STR_INTERNALNAME     QSVENC_FILENAME
#define VER_STR_ORIGINALFILENAME "QSVEncC.exe"
#define VER_STR_LEGALCOPYRIGHT   "QSVEncC by rigaya"
#define VER_STR_PRODUCTNAME      QSVENC_FILENAME
#define VER_PRODUCTVERSION       VER_FILEVERSION
#define VER_STR_PRODUCTVERSION   VER_STR_FILEVERSION

#endif //_QSVENCC_VERSION_H_
