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

#ifndef _QSV_QUERY_H_
#define _QSV_QUERY_H_

#include "rgy_tchar.h"
#include <emmintrin.h>
#if defined(_WIN32) || defined(_WIN64)
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#endif
#include <vector>
#include <array>
#include <string>
#include <chrono>
#include <memory>
#include <type_traits>
#include "rgy_osdep.h"
#include "mfxstructures.h"
#include "mfxsession.h"
#include "mfxvideo++.h"
#include "rgy_version.h"
#include "cpu_info.h"
#include "gpu_info.h"
#include "rgy_util.h"
#include "qsv_util.h"

enum {
    CPU_GEN_UNKNOWN = 0,
    CPU_GEN_SANDYBRIDGE,
    CPU_GEN_IVYBRIDGE,
    CPU_GEN_SILVERMONT,
    CPU_GEN_HASWELL,
    CPU_GEN_AIRMONT,
    CPU_GEN_BROADWELL,
    CPU_GEN_SKYLAKE,
    CPU_GEN_GOLDMONT,
    CPU_GEN_KABYLAKE,
    CPU_GEN_GEMINILAKE,
    CPU_GEN_COFFEELAKE,
    CPU_GEN_CANNONLAKE,
    CPU_GEN_ICELAKE,
};

static const TCHAR *const CPU_GEN_STR[] = {
    _T(""),
    _T("SandyBridge"),
    _T("IvyBridge"),
    _T("Silvermont"),
    _T("Haswell"),
    _T("Airmont"),
    _T("Broadwell"),
    _T("Skylake"),
    _T("Apollolake"),
    _T("Kabylake"),
    _T("Geminilake"),
    _T("Coffeelake"),
    _T("Cannonlake"),
    _T("Icelake")
};

int getCPUGenCpuid();
int getCPUGen();
int getCPUGen(MFXVideoSession *pSession);

static const mfxVersion LIB_VER_LIST[] = {
    {  0, 0 },
    {  0, 1 },
    {  1, 1 },
    {  3, 1 },
    {  4, 1 },
    {  6, 1 },
    {  7, 1 },
    {  8, 1 },
    {  9, 1 },
    { 10, 1 },
    { 11, 1 },
    { 13, 1 },
    { 15, 1 },
    { 16, 1 },
    { 17, 1 },
    { 19, 1 },
    { 23, 1 },
    { 26, 1 },
    { 27, 1 },
    { 0, 0 }
};

static const mfxU32 CODEC_LIST_AUO[] = {
    MFX_CODEC_AVC,
    MFX_CODEC_HEVC
};

#define MFX_LIB_VERSION_0_0  LIB_VER_LIST[ 0]
#define MFX_LIB_VERSION_1_0  LIB_VER_LIST[ 1]
#define MFX_LIB_VERSION_1_1  LIB_VER_LIST[ 2]
#define MFX_LIB_VERSION_1_3  LIB_VER_LIST[ 3]
#define MFX_LIB_VERSION_1_4  LIB_VER_LIST[ 4]
#define MFX_LIB_VERSION_1_6  LIB_VER_LIST[ 5]
#define MFX_LIB_VERSION_1_7  LIB_VER_LIST[ 6]
#define MFX_LIB_VERSION_1_8  LIB_VER_LIST[ 7]
#define MFX_LIB_VERSION_1_9  LIB_VER_LIST[ 8]
#define MFX_LIB_VERSION_1_10 LIB_VER_LIST[ 9]
#define MFX_LIB_VERSION_1_11 LIB_VER_LIST[10]
#define MFX_LIB_VERSION_1_13 LIB_VER_LIST[11]
#define MFX_LIB_VERSION_1_15 LIB_VER_LIST[12]
#define MFX_LIB_VERSION_1_16 LIB_VER_LIST[13]
#define MFX_LIB_VERSION_1_17 LIB_VER_LIST[14]
#define MFX_LIB_VERSION_1_19 LIB_VER_LIST[15]
#define MFX_LIB_VERSION_1_23 LIB_VER_LIST[16]
#define MFX_LIB_VERSION_1_26 LIB_VER_LIST[17]
#define MFX_LIB_VERSION_1_27 LIB_VER_LIST[18]

BOOL Check_HWUsed(mfxIMPL impl);
int GetAdapterID(mfxIMPL impl);
int GetAdapterID(mfxSession session);
mfxVersion get_mfx_libhw_version();
mfxVersion get_mfx_libsw_version();
mfxVersion get_mfx_lib_version(mfxIMPL impl);

static BOOL check_lib_version(mfxVersion value, mfxVersion required) {
    if (value.Major < required.Major)
        return FALSE;
    if (value.Major > required.Major)
        return TRUE;
    if (value.Minor < required.Minor)
        return FALSE;
    return TRUE;
}

static BOOL check_lib_version(mfxU32 _value, mfxU32 _required) {
    mfxVersion value, required;
    value.Version = _value;
    required.Version = _required;
    if (value.Major < required.Major)
        return FALSE;
    if (value.Major > required.Major)
        return TRUE;
    if (value.Minor < required.Minor)
        return FALSE;
    return TRUE;
}

static bool inline rc_is_type_lookahead(int rc) {
    return ((rc == MFX_RATECONTROL_LA)
        | (rc == MFX_RATECONTROL_LA_ICQ)
        | (rc == MFX_RATECONTROL_LA_EXT)
        | (rc == MFX_RATECONTROL_LA_HRD));
}

enum : uint64_t {
    ENC_FEATURE_CURRENT_RC             = 0x0000000000000001,
    ENC_FEATURE_AVBR                   = 0x0000000000000002,
    ENC_FEATURE_LA                     = 0x0000000000000004,
    ENC_FEATURE_ICQ                    = 0x0000000000000008,
    ENC_FEATURE_LA_ICQ                 = 0x0000000000000010,
    ENC_FEATURE_VCM                    = 0x0000000000000020,
    ENC_FEATURE_AUD                    = 0x0000000000000040,
    ENC_FEATURE_PIC_STRUCT             = 0x0000000000000080,
    ENC_FEATURE_VUI_INFO               = 0x0000000000000100,
    ENC_FEATURE_CAVLC                  = 0x0000000000000200,
    ENC_FEATURE_RDO                    = 0x0000000000000400,
    ENC_FEATURE_ADAPTIVE_I             = 0x0000000000000800,
    ENC_FEATURE_ADAPTIVE_B             = 0x0000000000001000,
    ENC_FEATURE_B_PYRAMID              = 0x0000000000002000,
    ENC_FEATURE_TRELLIS                = 0x0000000000004000,
    ENC_FEATURE_EXT_BRC                = 0x0000000000008000,
    ENC_FEATURE_MBBRC                  = 0x0000000000010000,
    ENC_FEATURE_LA_DS                  = 0x0000000000020000,
    ENC_FEATURE_INTERLACE              = 0x0000000000040000,
    ENC_FEATURE_EXT_BRC_ADAPTIVE_LTR   = 0x0000000000080000,
    ENC_FEATURE_UNUSED                 = 0x0000000000100000,
    ENC_FEATURE_B_PYRAMID_MANY_BFRAMES = 0x0000000000200000,
    ENC_FEATURE_LA_HRD                 = 0x0000000000400000,
    ENC_FEATURE_LA_EXT                 = 0x0000000000800000,
    ENC_FEATURE_QVBR                   = 0x0000000001000000,
    ENC_FEATURE_INTRA_REFRESH          = 0x0000000002000000,
    ENC_FEATURE_NO_DEBLOCK             = 0x0000000004000000,
    ENC_FEATURE_QP_MINMAX              = 0x0000000008000000,
    ENC_FEATURE_WINBRC                 = 0x0000000010000000,
    ENC_FEATURE_PERMBQP                = 0x0000000020000000,
    ENC_FEATURE_DIRECT_BIAS_ADJUST     = 0x0000000040000000,
    ENC_FEATURE_GLOBAL_MOTION_ADJUST   = 0x0000000080000000,
    ENC_FEATURE_FIXED_FUNC             = 0x0000000100000000,
    ENC_FEATURE_WEIGHT_P               = 0x0000000200000000,
    ENC_FEATURE_WEIGHT_B               = 0x0000000400000000,
    ENC_FEATURE_FADE_DETECT            = 0x0000000800000000,
    ENC_FEATURE_PYRAMID_QP_OFFSET      = 0x0000001000000000,
    ENC_FEATURE_DISABLE_GPB            = 0x0000002000000000,
    ENC_FEATURE_10BIT_DEPTH            = 0x0000004000000000,
    ENC_FEATURE_HEVC_SAO               = 0x0000008000000000,
    ENC_FEATURE_HEVC_CTU               = 0x0000010000000000,
    ENC_FEATURE_HEVC_TSKIP             = 0x0000020000000000,
};

enum : uint64_t {
    VPP_FEATURE_RESIZE                = 0x00000001,
    VPP_FEATURE_DENOISE               = 0x00000002,
    VPP_FEATURE_DETAIL_ENHANCEMENT    = 0x00000004,
    VPP_FEATURE_PROC_AMP              = 0x00000008,
    VPP_FEATURE_IMAGE_STABILIZATION   = 0x00000010,
    VPP_FEATURE_VIDEO_SIGNAL_INFO     = 0x00000020,
    VPP_FEATURE_FPS_CONVERSION        = 0x00000040,
    VPP_FEATURE_FPS_CONVERSION_ADV    = 0x00000080 | VPP_FEATURE_FPS_CONVERSION,
    VPP_FEATURE_DEINTERLACE           = 0x00000100,
    VPP_FEATURE_DEINTERLACE_AUTO      = 0x00000200,
    VPP_FEATURE_DEINTERLACE_IT_MANUAL = 0x00000400,
    VPP_FEATURE_ROTATE                = 0x00000800,
    VPP_FEATURE_MIRROR                = 0x00001000,
    VPP_FEATURE_SCALING_QUALITY       = 0x00002000,
    VPP_FEATURE_MCTF                  = 0x00004000,
};

static const CX_DESC list_rate_control_ry[] = {
    { _T("CBR  "), MFX_RATECONTROL_CBR    },
    { _T("VBR  "), MFX_RATECONTROL_VBR    },
    { _T("AVBR "), MFX_RATECONTROL_AVBR   },
    { _T("QVBR "), MFX_RATECONTROL_QVBR   },
    { _T("CQP  "), MFX_RATECONTROL_CQP    },
    { _T("LA   "), MFX_RATECONTROL_LA     },
    { _T("LAHRD"), MFX_RATECONTROL_LA_HRD },
    { _T("ICQ  "), MFX_RATECONTROL_ICQ    },
    { _T("LAICQ"), MFX_RATECONTROL_LA_ICQ },
    //{ _T("LAEXT"), MFX_RATECONTROL_LA_EXT },
    { _T("VCM  "), MFX_RATECONTROL_VCM    },
};
static const FEATURE_DESC list_enc_feature[] = {
    { _T("RC mode      "), ENC_FEATURE_CURRENT_RC             },
    { _T("10bit depth  "), ENC_FEATURE_10BIT_DEPTH            },
    { _T("Fixed Func   "), ENC_FEATURE_FIXED_FUNC             },
    { _T("Interlace    "), ENC_FEATURE_INTERLACE              },
    { _T("VUI info     "), ENC_FEATURE_VUI_INFO               },
    //{ _T("aud          "), ENC_FEATURE_AUD                    },
    //{ _T("pic_struct   "), ENC_FEATURE_PIC_STRUCT             },
    { _T("Trellis      "), ENC_FEATURE_TRELLIS                },
    //{ _T("rdo          "), ENC_FEATURE_RDO                    },
    //{ _T("CAVLC        "), ENC_FEATURE_CAVLC                  },
    { _T("Adaptive_I   "), ENC_FEATURE_ADAPTIVE_I             },
    { _T("Adaptive_B   "), ENC_FEATURE_ADAPTIVE_B             },
    { _T("WeightP      "), ENC_FEATURE_WEIGHT_P               },
    { _T("WeightB      "), ENC_FEATURE_WEIGHT_B               },
    { _T("FadeDetect   "), ENC_FEATURE_FADE_DETECT            },
    { _T("B_Pyramid    "), ENC_FEATURE_B_PYRAMID              },
    { _T(" +ManyBframes"), ENC_FEATURE_B_PYRAMID_MANY_BFRAMES },
    { _T("PyramQPOffset"), ENC_FEATURE_PYRAMID_QP_OFFSET      },
    { _T("MBBRC        "), ENC_FEATURE_MBBRC                  },
    { _T("ExtBRC       "), ENC_FEATURE_EXT_BRC                },
    { _T("Adaptive_LTR "), ENC_FEATURE_EXT_BRC_ADAPTIVE_LTR   },
    { _T("LA Quality   "), ENC_FEATURE_LA_DS                  },
    { _T("QP Min/Max   "), ENC_FEATURE_QP_MINMAX              },
    { _T("IntraRefresh "), ENC_FEATURE_INTRA_REFRESH          },
    { _T("No Deblock   "), ENC_FEATURE_NO_DEBLOCK             },
    { _T("No GPB       "), ENC_FEATURE_DISABLE_GPB            },
    { _T("Windowed BRC "), ENC_FEATURE_WINBRC                 },
    { _T("PerMBQP(CQP) "), ENC_FEATURE_PERMBQP                },
    { _T("DirectBiasAdj"), ENC_FEATURE_DIRECT_BIAS_ADJUST     },
    { _T("MVCostScaling"), ENC_FEATURE_GLOBAL_MOTION_ADJUST   },
    { _T("SAO          "), ENC_FEATURE_HEVC_SAO               },
    { _T("Max CTU Size "), ENC_FEATURE_HEVC_CTU               },
    { _T("TSkip        "), ENC_FEATURE_HEVC_TSKIP             },
    { NULL, 0 },
};
static const FEATURE_DESC list_vpp_feature[] = {
    { _T("Resize               "), VPP_FEATURE_RESIZE              },
    { _T("Deinterlace          "), VPP_FEATURE_DEINTERLACE         },
    { _T("Scaling Quality      "), VPP_FEATURE_SCALING_QUALITY     },
    { _T("Denoise              "), VPP_FEATURE_DENOISE             },
    { _T("Mctf                 "), VPP_FEATURE_MCTF                },
    { _T("Rotate               "), VPP_FEATURE_ROTATE              },
    { _T("Mirror               "), VPP_FEATURE_MIRROR              },
    { _T("Detail Enhancement   "), VPP_FEATURE_DETAIL_ENHANCEMENT  },
    { _T("Proc Amp.            "), VPP_FEATURE_PROC_AMP            },
    { _T("Image Stabilization  "), VPP_FEATURE_IMAGE_STABILIZATION },
    { _T("Video Signal Info    "), VPP_FEATURE_VIDEO_SIGNAL_INFO   },
    { _T("FPS Conversion       "), VPP_FEATURE_FPS_CONVERSION      },
    { _T("FPS Conversion (Adv.)"), VPP_FEATURE_FPS_CONVERSION_ADV  },
    { NULL, 0 },
};

enum FeatureListStrType {
    FEATURE_LIST_STR_TYPE_UNKNOWN,
    FEATURE_LIST_STR_TYPE_TXT,
    FEATURE_LIST_STR_TYPE_CSV,
    FEATURE_LIST_STR_TYPE_HTML,
};

mfxU64 CheckEncodeFeature(MFXVideoSession& session, mfxVersion ver, mfxU16 ratecontrol, mfxU32 codecId);
mfxU64 CheckEncodeFeatureWithPluginLoad(MFXVideoSession& session, mfxVersion ver, mfxU16 ratecontrol, mfxU32 codecId);
vector<mfxU64> MakeFeatureList(mfxVersion ver, const vector<CX_DESC>& rateControlList, mfxU32 codecId);
vector<vector<mfxU64>> MakeFeatureListPerCodec(const vector<CX_DESC>& rateControlList, const vector<mfxU32>& codecIdList);

tstring MakeFeatureListStr(mfxU64 feature);
vector<std::pair<vector<mfxU64>, tstring>> MakeFeatureListStr(FeatureListStrType outputType);
vector<std::pair<vector<mfxU64>, tstring>> MakeFeatureListStr(FeatureListStrType outputType, const vector<mfxU32>& codecIdList);

mfxU64 CheckVppFeatures(MFXVideoSession& session, mfxVersion ver);
mfxU64 CheckVppFeatures(mfxVersion ver);
tstring MakeVppFeatureStr(FeatureListStrType outputType);

std::vector<RGY_CSP> CheckDecFeaturesInternal(MFXVideoSession& session, mfxVersion mfxVer, mfxU32 codecId);
CodecCsp MakeDecodeFeatureList(mfxVersion ver, const vector<mfxU32>& codecIdList);
tstring MakeDecFeatureStr(FeatureListStrType type);
CodecCsp getHWDecCodecCsp();

#endif //_QSV_QUERY_H_
