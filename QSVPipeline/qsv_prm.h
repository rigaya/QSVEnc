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

#ifndef _QSV_PRM_H_
#define _QSV_PRM_H_

#include <climits>
#include <vector>
#include <optional>
#include "rgy_version.h"
#include "rgy_util.h"
#pragma warning (push)
#pragma warning (disable: 4201) //C4201: 非標準の拡張機能が使用されています: 無名の構造体または共用体です。
#pragma warning (disable: 4996) //C4996: 'MFXInit': が古い形式として宣言されました。
#pragma warning (disable: 4819) //C4819: ファイルは、現在のコード ページ (932) で表示できない文字を含んでいます。データの損失を防ぐために、ファイルを Unicode 形式で保存してください。
RGY_DISABLE_WARNING_PUSH
RGY_DISABLE_WARNING_STR("-Wdeprecated-declarations")
#include "mfxcommon.h"
#include "mfxsession.h"
#include "mfxvp8.h"
#include "mfxstructures.h"
RGY_DISABLE_WARNING_POP
#pragma warning (pop)
#include "convert_csp.h"
#include "rgy_simd.h"
#include "rgy_prm.h"

#define QSVENCC_ABORT_EVENT _T("QSVEncC_abort_%u")

enum {
    MFX_DEINTERLACE_NONE        = 0,
    MFX_DEINTERLACE_NORMAL      = 1,
    MFX_DEINTERLACE_IT          = 2, //inverse telecine, to 24fps
    MFX_DEINTERLACE_BOB         = 3,
    MFX_DEINTERLACE_IT_MANUAL   = 4, //inverse telecine, manual select
    MFX_DEINTERLACE_AUTO_SINGLE = 5,
    MFX_DEINTERLACE_AUTO_DOUBLE = 6,
};

enum {
    MVC_DISABLED          = 0x0,
    MVC_ENABLED           = 0x1,
    MVC_VIEWOUTPUT        = 0x2,    // 2 output bitstreams
};

enum class QSVDeviceNum {
    AUTO,
    NUM1,
    NUM2,
    NUM3,
    NUM4,
    MAX = 64,
};

enum MemType {
    SYSTEM_MEMORY = 0x00,
    VA_MEMORY     = 0x01,
    D3D9_MEMORY   = 0x01,
    D3D11_MEMORY  = 0x02,
    HW_MEMORY     = D3D9_MEMORY | D3D11_MEMORY,
};

static MemType operator~(MemType a) {
    return (MemType)(~(uint32_t)a);
}

static MemType operator|(MemType a, MemType b) {
    return (MemType)((uint32_t)a | (uint32_t)b);
}

static MemType operator|=(MemType& a, MemType b) {
    a = a | b;
    return a;
}

static MemType operator&(MemType a, MemType b) {
    return (MemType)((uint32_t)a & (uint32_t)b);
}

static MemType operator&=(MemType& a, MemType b) {
    a = (MemType)((uint32_t)a & (uint32_t)b);
    return a;
}

enum {
    FPS_CONVERT_NONE = 0,
    FPS_CONVERT_MUL2,
    FPS_CONVERT_MUL2_5,
};

enum {
    QSV_VPP_SUB_SIMPLE = 0,
    QSV_VPP_SUB_COMPLEX,
};

static const mfxDenoiseMode MFX_DENOISE_MODE_LEGACY = (mfxDenoiseMode)-1;

struct VppDenoise {
    bool enable;
    mfxDenoiseMode mode;
    int strength; // 0 - 100

    VppDenoise();
    ~VppDenoise() {};
    bool operator==(const VppDenoise &x) const;
    bool operator!=(const VppDenoise &x) const;
};

struct VppMCTF {
    bool enable;
    int strength; // 0 - 20

    VppMCTF();
    ~VppMCTF() {};
};

struct VppDetailEnhance {
    bool enable;
    int strength; // 0 - 100

    VppDetailEnhance();
    ~VppDetailEnhance() {};
};

struct MFXVppColorspace {
    bool enable;
    struct {
        CspMatrix matrix;
        CspColorRange range;
    } from, to;

    MFXVppColorspace() : enable(false), from(), to() {
        from.matrix = RGY_MATRIX_AUTO;
        from.range = RGY_COLORRANGE_AUTO;
        to.matrix = RGY_MATRIX_AUTO;
        to.range = RGY_COLORRANGE_AUTO;
    }
};

struct MFXVppAISuperRes {
    bool enable;
    int mode;

    MFXVppAISuperRes() : enable(false), mode(MFX_AI_SUPER_RESOLUTION_MODE_DEFAULT) {};
    ~MFXVppAISuperRes() {};
};

struct sVppParams {
    bool bEnable;             //use vpp

    bool bUseResize;          //use vpp resizer
    int resizeInterp;
    int resizeMode;

    MFXVppColorspace colorspace;

    int deinterlace;      //set deinterlace mode
    int telecinePattern;

    int imageStabilizer;  //MFX_IMAGESTAB_MODE_UPSCALE, MFX_IMAGESTAB_MODE_BOXED
    int fpsConversion;    //FPS_CONVERT_xxxx

    int rotate;
    bool halfTurn;
    int mirrorType;

    bool useProAmp;          //not supported

    VppDenoise denoise;
    VppMCTF mctf;
    VppDetailEnhance detail;
    MFXVppAISuperRes aiSuperRes;

    bool percPreEnc;

    sVppParams();
    ~sVppParams() {};
};


struct QSVAV1Params {
    int tile_row;
    int tile_col;

    QSVAV1Params();
};

struct QSVRCParam {
    int start;
    int end;
    int encMode;      // RateControl
    int bitrate;
    int maxBitrate;
    int vbvBufSize;
    int avbrAccuarcy;    // param for AVBR algorithm, for API v1.3
    int avbrConvergence; // param for AVBR algorithm, for API v1.3
    RGYQPSet qp;
    int icqQuality;
    int qvbrQuality;

    QSVRCParam();
    QSVRCParam(int encMode, int bitrate, int maxBitrate, int vbvBufSize, int avbrAccuarcy, int avbrConvergence,
        RGYQPSet qp, int icqQuality, int qvbrQuality);
    tstring print() const;
    bool operator==(const QSVRCParam &x) const;
    bool operator!=(const QSVRCParam &x) const;
};
tstring printParams(const std::vector<QSVRCParam> &dynamicRC);

enum class QSVFunctionMode {
    Auto,
    PG,
    FF,
};

struct sInputParams {
    VideoInfo input;              //入力する動画の情報
    RGYParamInput inprm;
    RGYParamCommon common;
    RGYParamControl ctrl;
    RGYParamVpp vpp;
    sVppParams vppmfx;
    QSVDeviceNum device;
    QSVRCParam rcParam;
    int nTargetUsage;  // Quality
    RGY_CODEC codec;
    int CodecProfile;
    int CodecLevel;
    int outputDepth;
    RGY_CHROMAFMT outputCsp;
    int nIdrInterval;  // Idr frame interval to I frame, not supported
    int nGOPLength;    // (Max) GOP Length
    bool bopenGOP;      // if false, GOP_CLOSED is set
    bool bforceGOPSettings; // if true, GOP_STRICT is set
    int GopRefDist;    // set sequential Bframes num + 1, 0 is auto
    int nRef;          // set ref frames num.
    RGYQPSet qpMin;
    RGYQPSet qpMax;
    std::vector<QSVRCParam> dynamicRC;

    int        nSlices;       // number of slices, 0 is auto

    uint32_t ColorFormat;   //YV12 or NV12

    MemType memType;       //use d3d surface

    mfxHyperMode hyperMode;

    int nInputBufSize; //input buf size

    int        nPAR[2]; //PAR比
    bool       bCAVLC;  //CAVLC
    int        nInterPred;
    int        nIntraPred;
    bool       bRDO;
    int        nMVPrecision;
    std::pair<int,int> MVSearchWindow;

    int        MVC_flags;
    int        nBluray;

    std::optional<bool> bBPyramid;
    std::optional<bool> bMBBRC;
    std::optional<bool> extBRC;
    std::optional<bool> adaptiveRef;
    std::optional<bool> adaptiveLTR;
    std::optional<bool> adaptiveCQM;
    std::optional<bool> bAdaptiveI;
    std::optional<bool> bAdaptiveB;

    int        nLookaheadDepth;
    int        nTrellis;

    int        nAsyncDepth;

    int        nLookaheadDS;
    uint32_t   tuneQuality;
    int        scenarioInfo;

    bool       bDisableTimerPeriodTuning;

    int        intraRefreshCycle;
    bool       bNoDeblock;

    int        maxFrameSize;
    int        maxFrameSizeI;
    int        maxFrameSizeP;
    int        nWinBRCSize;

    int        nMVCostScaling;
    std::optional<bool> bDirectBiasAdjust;
    bool       bGlobalMotionAdjust;
    QSVFunctionMode functionMode;
    bool       gpuCopy;

    int        nSessionThreads;
    int        nSessionThreadPriority;

    int        nVP8Sharpness;

    int        nWeightP;
    int        nWeightB;
    std::optional<bool> nFadeDetect;

    bool       fallbackRC;
    bool       bOutputAud;
    bool       bOutputPicStruct;
    bool       bufPeriodSEI;
    std::optional<bool> repeatHeaders;
    int16_t    pQPOffset[8];

    std::optional<bool> nRepartitionCheck;
    int8_t     padding[2];

    int        hevc_ctu;
    int        hevc_sao;
    int        hevc_tskip;
    int        hevc_tier;
    std::optional<bool> hevc_gpb;

    QSVAV1Params av1;

    tstring    pythonPath;

    bool       bBenchmark;
    mfxU32     nBenchQuality; //ベンチマークの対象

    void applyDOVIProfile(const RGYDOVIProfile inputProfile);

    sInputParams();
    ~sInputParams();
};

enum {
    MFX_CODEC_RAW = MFX_MAKEFOURCC('R', 'A', 'W', ' '),
};

const CX_DESC list_qsv_device[] = {
    { _T("auto"), (int)QSVDeviceNum::AUTO  },
    { _T("1"),    (int)QSVDeviceNum::NUM1 },
    { _T("2"),    (int)QSVDeviceNum::NUM2 },
    { _T("3"),    (int)QSVDeviceNum::NUM3 },
    { _T("4"),    (int)QSVDeviceNum::NUM4 },
    { NULL, 0 }
};

const CX_DESC list_codec_mfx[] = {
    { _T("h264"),     MFX_CODEC_AVC   },
    { _T("hevc"),     MFX_CODEC_HEVC  },
    { _T("mpeg2"),    MFX_CODEC_MPEG2 },
    { _T("vc-1"),     MFX_CODEC_VC1   },
    { _T("vp8"),      MFX_CODEC_VP8   },
    { _T("vp9"),      MFX_CODEC_VP9   },
    { _T("av1"),      MFX_CODEC_AV1   },
    { _T("vvc"),      MFX_CODEC_VVC   },
    { _T("raw"),      MFX_CODEC_RAW   },
    { NULL, 0 }
};

const CX_DESC list_codec_rgy[] = {
    { _T("h264"),     RGY_CODEC_H264   },
    { _T("hevc"),     RGY_CODEC_HEVC  },
    { _T("mpeg2"),    RGY_CODEC_MPEG2 },
    { _T("vc-1"),     RGY_CODEC_VC1   },
    { _T("vp8"),      RGY_CODEC_VP8   },
    { _T("vp9"),      RGY_CODEC_VP9   },
    { _T("av1"),      RGY_CODEC_AV1   },
    { _T("vvc"),      RGY_CODEC_VVC   },
    { _T("raw"),      RGY_CODEC_RAW   },
    { NULL, 0 }
};

const CX_DESC list_rc_mode[] = {
    { _T("CBR"),     MFX_RATECONTROL_CBR    },
    { _T("VBR"),     MFX_RATECONTROL_VBR    },
    { _T("CQP"),     MFX_RATECONTROL_CQP    },
    { _T("AVBR"),    MFX_RATECONTROL_AVBR   },
    { _T("LA"),      MFX_RATECONTROL_LA     },
    { _T("LA-HRD"),  MFX_RATECONTROL_LA_HRD },
    { _T("LA-ICQ"),  MFX_RATECONTROL_LA_ICQ },
    { _T("ICQ"),     MFX_RATECONTROL_ICQ    },
    { _T("QVBR"),    MFX_RATECONTROL_QVBR   },
    { _T("VCM"),     MFX_RATECONTROL_VCM    },
    { NULL, 0 }
};

const CX_DESC list_avc_profile[] = {
    { _T("auto"),     0                        },
    { _T("Baseline"), MFX_PROFILE_AVC_BASELINE },
    { _T("Main"),     MFX_PROFILE_AVC_MAIN     },
    { _T("High"),     MFX_PROFILE_AVC_HIGH     },
    { NULL, 0 }
};

const CX_DESC list_hevc_profile[] = {
    { _T("auto"),     0                       },
    { _T("main"),     MFX_PROFILE_HEVC_MAIN   },
    { _T("main10"),   MFX_PROFILE_HEVC_MAIN10 },
    { _T("main444"),  MFX_PROFILE_HEVC_REXT   },
    { NULL, 0 }
};

const CX_DESC list_hevc_tier[] = {
    { _T("main"),   MFX_TIER_HEVC_MAIN },
    { _T("high"),   MFX_TIER_HEVC_HIGH },
    { NULL, 0 }
};

const CX_DESC list_mpeg2_profile[] = {
    { _T("auto"),     0                        },
    { _T("Simple"),   MFX_PROFILE_MPEG2_SIMPLE },
    { _T("Main"),     MFX_PROFILE_MPEG2_MAIN   },
    { _T("High"),     MFX_PROFILE_MPEG2_HIGH   },
    { NULL, 0 }
};

const CX_DESC list_vc1_profile[] = {
    { _T("auto"),     0                        },
    { _T("Simple"),   MFX_PROFILE_VC1_SIMPLE   },
    { _T("Main"),     MFX_PROFILE_VC1_MAIN     },
    { _T("Advanced"), MFX_PROFILE_VC1_ADVANCED },
    { NULL, 0 }
};

const CX_DESC list_vp8_profile[] = {
    { _T("auto"),     0 },
    { _T("0"),        MFX_PROFILE_VP8_0 },
    { _T("1"),        MFX_PROFILE_VP8_1 },
    { _T("2"),        MFX_PROFILE_VP8_2 },
    { _T("3"),        MFX_PROFILE_VP8_3 },
    { NULL, 0 }
};

const CX_DESC list_vp9_profile[] = {
    { _T("auto"),     0 },
    { _T("0"),        MFX_PROFILE_VP9_0 },
    { _T("1"),        MFX_PROFILE_VP9_1 },
    { _T("2"),        MFX_PROFILE_VP9_2 },
    { _T("3"),        MFX_PROFILE_VP9_3 },
    { NULL, 0 }
};

const CX_DESC list_av1_profile[] = {
    { _T("auto"),     0 },
    { _T("main"),     MFX_PROFILE_AV1_MAIN },
    { _T("high"),     MFX_PROFILE_AV1_HIGH },
    { _T("pro"),      MFX_PROFILE_AV1_PRO  },
    { NULL, 0 }
};

const CX_DESC list_vvc_profile[] = {
    { _T("auto"),     0                        },
    { _T("main10"),   MFX_PROFILE_VVC_MAIN10   },
    { NULL, 0 }
};

const CX_DESC list_output_depth[] = {
    { _T("8"),     8 },
    { _T("10"),   10 },
    { _T("12"),   12 },
    { NULL, 0 }
};

const CX_DESC list_output_csp[] = {
    { _T("yuv420"), RGY_CHROMAFMT_YUV420 },
    { _T("yuv422"), RGY_CHROMAFMT_YUV422 },
    { _T("yuv444"), RGY_CHROMAFMT_YUV444 },
    { _T("rgb"),    RGY_CHROMAFMT_RGB },
    { _T("i420"), RGY_CHROMAFMT_YUV420 },
    { _T("i422"), RGY_CHROMAFMT_YUV422 },
    { _T("i444"), RGY_CHROMAFMT_YUV444 },
    { NULL, 0 }
};

const CX_DESC list_interlaced_mfx[] = {
    { _T("progressive"),     MFX_PICSTRUCT_PROGRESSIVE },
    { _T("interlaced(tff)"), MFX_PICSTRUCT_FIELD_TFF   },
    { _T("interlaced(bff)"), MFX_PICSTRUCT_FIELD_BFF   },
    { NULL, 0 }
};

const CX_DESC list_deinterlace[] = {
    { _T("none"),      MFX_DEINTERLACE_NONE        },
    { _T("normal"),    MFX_DEINTERLACE_NORMAL      },
    { _T("it"),        MFX_DEINTERLACE_IT          },
    { _T("bob"),       MFX_DEINTERLACE_BOB         },
#if ENABLE_ADVANCED_DEINTERLACE
    { _T("it-manual"), MFX_DEINTERLACE_IT_MANUAL   },
    { _T("auto"),      MFX_DEINTERLACE_AUTO_SINGLE },
    { _T("auto-bob"),  MFX_DEINTERLACE_AUTO_DOUBLE },
#endif
    { NULL, 0 }
};

const CX_DESC list_telecine_patterns[] = {
    { _T("32"),     MFX_TELECINE_PATTERN_32 },
    { _T("2332"),   MFX_TELECINE_PATTERN_2332 },
    { _T("repeat"), MFX_TELECINE_PATTERN_FRAME_REPEAT },
    { _T("41"),     MFX_TELECINE_PATTERN_41 },
    { NULL, 0 }
};

const CX_DESC list_avc_level[] = {
    { _T("auto"), 0                 },
    { _T("1"),    MFX_LEVEL_AVC_1   },
    { _T("1b"),   MFX_LEVEL_AVC_1b  },
    { _T("1.1"),  MFX_LEVEL_AVC_11  },
    { _T("1.2"),  MFX_LEVEL_AVC_12  },
    { _T("1.3"),  MFX_LEVEL_AVC_13  },
    { _T("2"),    MFX_LEVEL_AVC_2   },
    { _T("2.1"),  MFX_LEVEL_AVC_21  },
    { _T("2.2"),  MFX_LEVEL_AVC_22  },
    { _T("3"),    MFX_LEVEL_AVC_3   },
    { _T("3.1"),  MFX_LEVEL_AVC_31  },
    { _T("3.2"),  MFX_LEVEL_AVC_32  },
    { _T("4"),    MFX_LEVEL_AVC_4   },
    { _T("4.1"),  MFX_LEVEL_AVC_41  },
    { _T("4.2"),  MFX_LEVEL_AVC_42  },
    { _T("5"),    MFX_LEVEL_AVC_5   },
    { _T("5.1"),  MFX_LEVEL_AVC_51  },
    { _T("5.2"),  MFX_LEVEL_AVC_52  },
#if (MFX_VERSION >= 1035)
    { _T("6"),    MFX_LEVEL_AVC_6   },
    { _T("6.1"),  MFX_LEVEL_AVC_61  },
    { _T("6.2"),  MFX_LEVEL_AVC_62  },
#endif
    { NULL, 0 }
};
const CX_DESC list_hevc_level[] = {
    { _T("auto"), 0                 },
    { _T("1"),    MFX_LEVEL_HEVC_1   },
    { _T("2"),    MFX_LEVEL_HEVC_2   },
    { _T("2.1"),  MFX_LEVEL_HEVC_21  },
    { _T("3"),    MFX_LEVEL_HEVC_3   },
    { _T("3.1"),  MFX_LEVEL_HEVC_31  },
    { _T("4"),    MFX_LEVEL_HEVC_4   },
    { _T("4.1"),  MFX_LEVEL_HEVC_41  },
    { _T("5"),    MFX_LEVEL_HEVC_5   },
    { _T("5.1"),  MFX_LEVEL_HEVC_51  },
    { _T("5.2"),  MFX_LEVEL_HEVC_52  },
    { _T("6"),    MFX_LEVEL_HEVC_6   },
    { _T("6.1"),  MFX_LEVEL_HEVC_61  },
    { _T("6.2"),  MFX_LEVEL_HEVC_62  },
    { NULL, 0 }
};
const CX_DESC list_mpeg2_level[] = {
    { _T("auto"),     0                        },
    { _T("low"),      MFX_LEVEL_MPEG2_LOW      },
    { _T("main"),     MFX_LEVEL_MPEG2_MAIN     },
    { _T("high"),     MFX_LEVEL_MPEG2_HIGH     },
    { _T("high1440"), MFX_LEVEL_MPEG2_HIGH1440 },
    { NULL, 0 }
};
const CX_DESC list_vc1_level[] = {
    { _T("auto"),     0                    },
    { _T("low"),      MFX_LEVEL_VC1_LOW    },
    { _T("median"),   MFX_LEVEL_VC1_MEDIAN },
    { _T("high"),     MFX_LEVEL_VC1_HIGH   },
    { NULL, 0 }
};
const CX_DESC list_vc1_level_adv[] = {
    { _T("auto"),  0               },
    { _T("0"),     MFX_LEVEL_VC1_0 },
    { _T("1"),     MFX_LEVEL_VC1_1 },
    { _T("2"),     MFX_LEVEL_VC1_2 },
    { _T("3"),     MFX_LEVEL_VC1_3 },
    { _T("4"),     MFX_LEVEL_VC1_4 },
    { NULL, 0 }
};
const CX_DESC list_vp8_level[] = {
    { _T("auto"),     0                    },
    { NULL, 0 }
};
const CX_DESC list_vp9_level[] = {
    { _T("auto"),     0                    },
    { NULL, 0 }
};

const CX_DESC list_av1_level[] = {
    { _T("auto"),     0 },
    { _T("2"),        MFX_LEVEL_AV1_2  },
    { _T("2.1"),      MFX_LEVEL_AV1_21 },
    { _T("2.2"),      MFX_LEVEL_AV1_22 },
    { _T("2.3"),      MFX_LEVEL_AV1_23 },
    { _T("3"),        MFX_LEVEL_AV1_3  },
    { _T("3.1"),      MFX_LEVEL_AV1_31 },
    { _T("3.2"),      MFX_LEVEL_AV1_32 },
    { _T("3.3"),      MFX_LEVEL_AV1_33 },
    { _T("4"),        MFX_LEVEL_AV1_4  },
    { _T("4.1"),      MFX_LEVEL_AV1_41 },
    { _T("4.2"),      MFX_LEVEL_AV1_42 },
    { _T("4.3"),      MFX_LEVEL_AV1_43 },
    { _T("5"),        MFX_LEVEL_AV1_5  },
    { _T("5.1"),      MFX_LEVEL_AV1_51 },
    { _T("5.2"),      MFX_LEVEL_AV1_52 },
    { _T("5.3"),      MFX_LEVEL_AV1_53 },
    { _T("6"),        MFX_LEVEL_AV1_6  },
    { _T("6.1"),      MFX_LEVEL_AV1_61 },
    { _T("6.2"),      MFX_LEVEL_AV1_62 },
    { _T("6.3"),      MFX_LEVEL_AV1_63 },
    { _T("7"),        MFX_LEVEL_AV1_7  },
    { _T("7.1"),      MFX_LEVEL_AV1_71 },
    { _T("7.2"),      MFX_LEVEL_AV1_72 },
    { _T("7.3"),      MFX_LEVEL_AV1_73 },
    { NULL, 0 }
};

const CX_DESC list_vvc_level[] = {
    { _T("auto"),     0 },
    { _T("1"),        MFX_LEVEL_VVC_1  },
    { _T("2"),        MFX_LEVEL_VVC_2  },
    { _T("2.1"),      MFX_LEVEL_VVC_21 },
    { _T("3"),        MFX_LEVEL_VVC_3  },
    { _T("3.1"),      MFX_LEVEL_VVC_31 },
    { _T("4"),        MFX_LEVEL_VVC_4  },
    { _T("4.1"),      MFX_LEVEL_VVC_41 },
    { _T("5"),        MFX_LEVEL_VVC_5  },
    { _T("5.1"),      MFX_LEVEL_VVC_51 },
    { _T("5.2"),      MFX_LEVEL_VVC_52 },
    { _T("6"),        MFX_LEVEL_VVC_6  },
    { _T("6.1"),      MFX_LEVEL_VVC_61 },
    { _T("6.2"),      MFX_LEVEL_VVC_62 },
    { NULL, 0 }
};

const CX_DESC list_avc_trellis[] = {
    { _T("Auto"),           MFX_TRELLIS_UNKNOWN },
    { _T("off"),            MFX_TRELLIS_OFF },
    { _T("for I frames"),   MFX_TRELLIS_I   },
    { _T("for I,P frames"), MFX_TRELLIS_I | MFX_TRELLIS_P },
    { _T("for All frames"), MFX_TRELLIS_I | MFX_TRELLIS_P | MFX_TRELLIS_B },
    { NULL, 0 }
};
const CX_DESC list_avc_trellis_for_options[] = {
    { _T("auto"), MFX_TRELLIS_UNKNOWN },
    { _T("off"),  MFX_TRELLIS_OFF },
    { _T("i"),    MFX_TRELLIS_I   },
    { _T("ip"),   MFX_TRELLIS_I | MFX_TRELLIS_P },
    { _T("all"),  MFX_TRELLIS_I | MFX_TRELLIS_P | MFX_TRELLIS_B },
    { _T("ipb"),  MFX_TRELLIS_I | MFX_TRELLIS_P | MFX_TRELLIS_B },
    { _T("ib"),   MFX_TRELLIS_I | MFX_TRELLIS_B },
    { _T("p"),    MFX_TRELLIS_P },
    { _T("pb"),   MFX_TRELLIS_P | MFX_TRELLIS_B },
    { _T("b"),    MFX_TRELLIS_B },
    { NULL, 0 }
};

const CX_DESC list_lookahead_ds[] = {
    { _T("auto"),   MFX_LOOKAHEAD_DS_UNKNOWN },
    { _T("slow"),   MFX_LOOKAHEAD_DS_OFF     },
    { _T("medium"), MFX_LOOKAHEAD_DS_2x      },
    { _T("fast"),   MFX_LOOKAHEAD_DS_4x      },
    { NULL, 0 }
};

const CX_DESC list_mv_cost_scaling[] = {
    { _T("default"), -1 },
    { _T("0"),        0 },
    { _T("1"),        1 },
    { _T("2"),        2 },
    { _T("3"),        3 },
    { _T("4"),        4 },
    { NULL, 0 }
};

const CX_DESC list_priority[] = {
    { _T("low"),    MFX_PRIORITY_LOW    },
    { _T("normal"), MFX_PRIORITY_NORMAL },
    { _T("high"),   MFX_PRIORITY_HIGH   },
};

const CX_DESC list_qsv_function_mode[] = {
    { _T("auto"),  (int)QSVFunctionMode::Auto  },
    { _T("PG"),    (int)QSVFunctionMode::PG    },
    { _T("FF"),    (int)QSVFunctionMode::FF    },
    { NULL, 0 }
};

const CX_DESC list_hyper_mode[] = {
    { _T("off"),      MFX_HYPERMODE_OFF      },
    { _T("on"),       MFX_HYPERMODE_ON       },
    { _T("adaptive"), MFX_HYPERMODE_ADAPTIVE },
    { NULL, 0 }
};

const CX_DESC list_enc_tune_quality_mode[] = {
    { _T("default"),    MFX_ENCODE_TUNE_OFF        },
    { _T("psnr"),       MFX_ENCODE_TUNE_PSNR       },
    { _T("ssim"),       MFX_ENCODE_TUNE_SSIM       },
    { _T("ms_ssim"),    MFX_ENCODE_TUNE_MS_SSIM    },
    { _T("vmaf"),       MFX_ENCODE_TUNE_VMAF       },
    { _T("perceptual"), MFX_ENCODE_TUNE_PERCEPTUAL },
    { NULL, 0 }
};

tstring get_str_of_tune_bitmask(const uint32_t mask);

enum {
    QSV_AUD_ENC_NONE = -1,
    QSV_AUD_ENC_COPY = 0,
    QSV_AUD_ENC_AAC,
    QSV_AUD_ENC_MP3,
    QSV_AUD_ENC_MP2,
    QSV_AUD_ENC_VORBIS,
    QSV_AUD_ENC_WAV,
    QSV_AUD_ENC_OPUS,
    QSV_AUD_ENC_AC3,
};

static inline const CX_DESC *get_level_list(const RGY_CODEC codec) {
    switch (codec) {
        case RGY_CODEC_H264:    return list_avc_level;
        case RGY_CODEC_MPEG2:   return list_mpeg2_level;
        case RGY_CODEC_VC1:     return list_vc1_level;
        case RGY_CODEC_HEVC:    return list_hevc_level;
        case RGY_CODEC_VP8:     return list_vp8_level;
        case RGY_CODEC_VP9:     return list_vp9_level;
        case RGY_CODEC_AV1:     return list_av1_level;
        case RGY_CODEC_VVC:     return list_vvc_level;
        case RGY_CODEC_RAW:     return list_empty;
        default:                return list_empty;
    }
}

static inline const CX_DESC *get_profile_list(const RGY_CODEC codec) {
    switch (codec) {
        case RGY_CODEC_H264:    return list_avc_profile;
        case RGY_CODEC_MPEG2:   return list_mpeg2_profile;
        case RGY_CODEC_VC1:     return list_vc1_profile;
        case RGY_CODEC_HEVC:    return list_hevc_profile;
        case RGY_CODEC_VP8:     return list_vp8_profile;
        case RGY_CODEC_VP9:     return list_vp9_profile;
        case RGY_CODEC_AV1:     return list_av1_profile;
        case RGY_CODEC_VVC:     return list_vvc_profile;
        case RGY_CODEC_RAW:     return list_empty;
        default:                return list_empty;
    }
}

//表示用
const CX_DESC list_quality[] = {
    { _T(" 1 - best quality"), MFX_TARGETUSAGE_BEST_QUALITY },
    { _T(" 2 - higher"),       2                            },
    { _T(" 3 - high quality"), 3                            },
    { _T(" 4 - balanced"),     MFX_TARGETUSAGE_BALANCED     },
    { _T(" 5 - fast"),         5                            },
    { _T(" 6 - faster"),       6                            },
    { _T(" 7 - fastest"),      MFX_TARGETUSAGE_BEST_SPEED   },
    { NULL, 0 }
};

//オプション用
const CX_DESC list_quality_for_option[] = {
    { _T("best"),     MFX_TARGETUSAGE_BEST_QUALITY },
    { _T("higher"),   2                            },
    { _T("high"),     3                            },
    { _T("balanced"), MFX_TARGETUSAGE_BALANCED     },
    { _T("fast"),     5                            },
    { _T("faster"),   6                            },
    { _T("fastest"),  MFX_TARGETUSAGE_BEST_SPEED   },
    { NULL, 0 }
};

const CX_DESC list_scenario_info[] = {
    { _T("unknown"),            MFX_SCENARIO_UNKNOWN            },
    { _T("display_remoting"),   MFX_SCENARIO_DISPLAY_REMOTING   },
    { _T("video_conference"),   MFX_SCENARIO_VIDEO_CONFERENCE   },
    { _T("archive"),            MFX_SCENARIO_ARCHIVE            },
    { _T("live_streaming"),     MFX_SCENARIO_LIVE_STREAMING     },
    { _T("camera_capture"),     MFX_SCENARIO_CAMERA_CAPTURE     },
    { _T("video_surveillance"), MFX_SCENARIO_VIDEO_SURVEILLANCE },
    { _T("game_streaming"),     MFX_SCENARIO_GAME_STREAMING     },
    { _T("remote_gaming"),      MFX_SCENARIO_REMOTE_GAMING      },
    { NULL, 0 }
};

const CX_DESC list_mv_presicion[] = {
    { _T("Auto"),     MFX_MVPRECISION_UNKNOWN    },
    { _T("full-pel"), MFX_MVPRECISION_INTEGER    },
    { _T("half-pel"), MFX_MVPRECISION_HALFPEL    },
    { _T("Q-pel"),    MFX_MVPRECISION_QUARTERPEL },
    { NULL, 0 }
};

const CX_DESC list_pred_block_size[] = {
    { _T("Auto"),          MFX_BLOCKSIZE_UNKNOWN    },
    { _T("16x16"),         MFX_BLOCKSIZE_MIN_16X16  },
    { _T("8x8"),           MFX_BLOCKSIZE_MIN_8X8    },
    { _T("4x4"),           MFX_BLOCKSIZE_MIN_4X4    },
    { NULL, 0 }
};

const CX_DESC list_hevc_ctu[] = {
    { _T("auto"), 0 },
    { _T("16"), 16 },
    { _T("32"), 32 },
    { _T("64"), 64 },
    { NULL, 0 }
};

const CX_DESC list_hevc_sao[] = {
    { _T("auto"),   MFX_SAO_UNKNOWN },
    { _T("none"),   MFX_SAO_DISABLE },
    { _T("luma"),   MFX_SAO_ENABLE_LUMA },
    { _T("chroma"), MFX_SAO_ENABLE_CHROMA },
    { _T("all"),    (uint32_t)MFX_SAO_ENABLE_LUMA | (uint32_t)MFX_SAO_ENABLE_CHROMA },
    { NULL, 0 }
};

const CX_DESC list_vpp_image_stabilizer[] = {
    { _T("none"),    0 },
    { _T("upscale"), MFX_IMAGESTAB_MODE_UPSCALE },
    { _T("box"),     MFX_IMAGESTAB_MODE_BOXING  },
    { NULL, 0 }
};

const CX_DESC list_vpp_fps_conversion[] = {
    { _T("off"),  0 },
    { _T("x2"),   FPS_CONVERT_MUL2 },
    { _T("x2.5"), FPS_CONVERT_MUL2_5  },
    { NULL, 0 }
};

const CX_DESC list_vpp_sub_shaping[] = {
    { _T("simple"),  QSV_VPP_SUB_SIMPLE  },
    { _T("complex"), QSV_VPP_SUB_COMPLEX },
    { NULL, 0 }
};

const CX_DESC list_vpp_mfx_denoise_mode[] = {
    { _T("auto"),            MFX_DENOISE_MODE_DEFAULT  },
    { _T("auto_bdrate"),     MFX_DENOISE_MODE_INTEL_HVS_AUTO_BDRATE },
    { _T("auto_subjective"), MFX_DENOISE_MODE_INTEL_HVS_AUTO_SUBJECTIVE },
    { _T("auto_adjust"),     MFX_DENOISE_MODE_INTEL_HVS_AUTO_ADJUST },
    { _T("pre"),             MFX_DENOISE_MODE_INTEL_HVS_PRE_MANUAL },
    { _T("post"),            MFX_DENOISE_MODE_INTEL_HVS_POST_MANUAL },
    { NULL, 0 }
};

/*
const CX_DESC list_vpp_scaling_quality[] = {
    { _T("auto"),   MFX_SCALING_MODE_DEFAULT  },
    { _T("simple"), MFX_SCALING_MODE_LOWPOWER },
    { _T("fine"),   MFX_SCALING_MODE_QUALITY  },
    { NULL, 0 }
};
*/

//define defaults
const int QSV_DEFAULT_REF = 0;
const int QSV_DEFAULT_GOP_LEN = 0;
const int QSV_DEFAULT_ICQ = 23;
const int QSV_DEFAULT_QVBR = 23;
const int QSV_DEFAULT_QPI = 24;
const int QSV_DEFAULT_QPP = 26;
const int QSV_DEFAULT_QPB = 27;
const int QSV_DEFAULT_BITRATE = 6000;
const int QSV_DEFAULT_MAX_BITRATE = 15000;
const int QSV_GOP_REF_DIST_AUTO = 0;
const int QSV_DEFAULT_H264_GOP_REF_DIST = 4;
const int QSV_DEFAULT_HEVC_GOP_REF_DIST = 4;
const int QSV_DEFAULT_AV1_GOP_REF_DIST = 8;
const int QSV_DEFAULT_QUALITY = MFX_TARGETUSAGE_BALANCED;
const int QSV_DEFAULT_INPUT_BUF_SW = 1;
const int QSV_DEFAULT_INPUT_BUF_HW = 3;
const int QSV_INPUT_BUF_MIN = 1;
const int QSV_INPUT_BUF_MAX = 16;
const int QSV_DEFAULT_CONVERGENCE = 90;
const int QSV_DEFAULT_ACCURACY = 500;
const int QSV_DEFAULT_FORCE_GOP_LEN = 1;
const int QSV_DEFAULT_OUTPUT_BUF_MB = 8;
const uint32_t QSV_DEFAULT_BENCH = (1 << 1) | (1 << 4) | (1 << 7);

const int QSV_DEFAULT_VQP_STRENGTH = 10;
const int QSV_DEFAULT_VQP_SENSITIVITY = 50;
const int QSV_DEFAULT_SC_SENSITIVITY = 80;

const int QSV_DEFAULT_ASYNC_DEPTH = 3;
const int QSV_ASYNC_DEPTH_MAX = 1024;
const int QSV_SESSION_THREAD_MAX = 64;

const int QSV_LOOKAHEAD_DEPTH_MIN = 0;
const int QSV_LOOKAHEAD_DEPTH_MAX = 100;

const uint32_t QSV_DEFAULT_AUDIO_IGNORE_DECODE_ERROR = 10;

const int QSV_DEFAULT_VPP_DELOGO_DEPTH = 128;

const int QSV_DEFAULT_PERF_MONITOR_INTERVAL = 500;

const int QSV_VPP_DENOISE_MIN = 0;
const int QSV_VPP_DENOISE_MAX = 100;
const int QSV_VPP_MCTF_AUTO = 0;
const int QSV_VPP_MCTF_MIN = 1;
const int QSV_VPP_MCTF_MAX = 20;
const int QSV_VPP_DETAIL_ENHANCE_MIN = 0;
const int QSV_VPP_DETAIL_ENHANCE_MAX = 100;

#endif //_QSV_PRM_H_
