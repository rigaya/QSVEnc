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

#ifndef _QSV_UTIL_H_
#define _QSV_UTIL_H_

#include "qsv_tchar.h"
#include <emmintrin.h>
#if defined(_WIN32) || defined(_WIN64)
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#endif
#include <vector>
#include <array>
#include <utility>
#include <string>
#include <chrono>
#include <memory>
#include <type_traits>
#include "qsv_osdep.h"
#include "mfxstructures.h"
#include "mfxsession.h"
#include "qsv_version.h"
#include "cpu_info.h"
#include "gpu_info.h"
#include "rgy_util.h"
#include "convert_csp.h"
#include "qsv_prm.h"

using std::vector;
using std::unique_ptr;
using std::shared_ptr;

#define MAP_PAIR_0_1_PROTO(prefix, name0, type0, name1, type1) \
    type1 prefix ## _ ## name0 ## _to_ ## name1(type0 var0); \
    type0 prefix ## _ ## name1 ## _to_ ## name0(type1 var1);

#define MAP_PAIR_0_1(prefix, name0, type0, name1, type1, map_pair, default0, default1) \
    __declspec(noinline) \
    type1 prefix ## _ ## name0 ## _to_ ## name1(type0 var0) {\
        auto ret = std::find_if(map_pair.begin(), map_pair.end(), [var0](std::pair<type0, type1> a) { \
            return a.first == var0; \
        }); \
        return (ret == map_pair.end()) ? default1 : ret->second; \
    } \
    __declspec(noinline)  \
    type0 prefix ## _ ## name1 ## _to_ ## name0(type1 var1) {\
        auto ret = std::find_if(map_pair.begin(), map_pair.end(), [var1](std::pair<type0, type1> a) { \
            return a.second == var1; \
        }); \
        return (ret == map_pair.end()) ? default0 : ret->first; \
    }

MAP_PAIR_0_1_PROTO(codec, rgy, RGY_CODEC, enc, mfxU32);
MAP_PAIR_0_1_PROTO(chromafmt, rgy, RGY_CHROMAFMT, enc, mfxU32);

mfxU16 picstruct_rgy_to_enc(RGY_PICSTRUCT picstruct);
RGY_PICSTRUCT picstruct_enc_to_rgy(mfxU16 picstruct);
mfxFrameInfo frameinfo_rgy_to_enc(VideoInfo info);

static const int RGY_CSP_TO_MFX_FOURCC[] = {
    0, //RGY_CSP_NA
    MFX_FOURCC_NV12, //RGY_CSP_NV12
    MFX_FOURCC_YV12, //RGY_CSP_YV12
    MFX_FOURCC_YUY2, //RGY_CSP_YUY2 
    0, //RGY_CSP_YUV422
    0, //RGY_CSP_YUV444
    MFX_FOURCC_P010, //RGY_CSP_YV12_09
    MFX_FOURCC_P010,
    MFX_FOURCC_P010,
    MFX_FOURCC_P010,
    MFX_FOURCC_P010, //RGY_CSP_YV12_16
    MFX_FOURCC_P010, //RGY_CSP_P010
    MFX_FOURCC_P210, //RGY_CSP_P210
    0, //RGY_CSP_YUV444_09
    0,
    0,
    0,
    0, //RGY_CSP_YUV444_16
    MFX_FOURCC_RGB3,
    MFX_FOURCC_RGB4,
    0 //RGY_CSP_YC48
};

mfxFrameInfo toMFXFrameInfo(VideoInfo info);

#endif //_QSV_UTIL_H_
