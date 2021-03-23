// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
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

#pragma once
#ifndef __RGY_PRM_H__
#define __RGY_PRM_H__

#include "rgy_def.h"
#include "rgy_caption.h"
#include "rgy_simd.h"
#include "rgy_hdr10plus.h"

static const int BITSTREAM_BUFFER_SIZE =  4 * 1024 * 1024;
static const int OUTPUT_BUF_SIZE       = 16 * 1024 * 1024;

static const int RGY_DEFAULT_PERF_MONITOR_INTERVAL = 500;
static const int DEFAULT_IGNORE_DECODE_ERROR = 10;

static const double FILTER_DEFAULT_COLORSPACE_LDRNITS = 100.0;
static const double FILTER_DEFAULT_COLORSPACE_NOMINAL_SOURCE_PEAK = 100.0;
static const double FILTER_DEFAULT_COLORSPACE_HDR_SOURCE_PEAK = 1000.0;

static const double FILTER_DEFAULT_HDR2SDR_DESAT_BASE = 0.18;
static const double FILTER_DEFAULT_HDR2SDR_DESAT_STRENGTH = 0.75;
static const double FILTER_DEFAULT_HDR2SDR_DESAT_EXP = 1.5;

static const double FILTER_DEFAULT_HDR2SDR_HABLE_A = 0.22;
static const double FILTER_DEFAULT_HDR2SDR_HABLE_B = 0.3;
static const double FILTER_DEFAULT_HDR2SDR_HABLE_C = 0.1;
static const double FILTER_DEFAULT_HDR2SDR_HABLE_D = 0.2;
static const double FILTER_DEFAULT_HDR2SDR_HABLE_E = 0.01;
static const double FILTER_DEFAULT_HDR2SDR_HABLE_F = 0.3;
static const double FILTER_DEFAULT_HDR2SDR_HABLE_W = 11.2;

static const double FILTER_DEFAULT_HDR2SDR_MOBIUS_TRANSITION = 0.3;
static const double FILTER_DEFAULT_HDR2SDR_MOBIUS_PEAK = 1.0;

static const double FILTER_DEFAULT_HDR2SDR_REINHARD_CONTRAST = 0.5;
static const double FILTER_DEFAULT_HDR2SDR_REINHARD_PEAK = 1.0;

static const int   FILTER_DEFAULT_AFS_CLIP_TB = 16;
static const int   FILTER_DEFAULT_AFS_CLIP_LR = 32;
static const int   FILTER_DEFAULT_AFS_TB_ORDER = 0;
static const int   FILTER_DEFAULT_AFS_METHOD_SWITCH = 0;
static const int   FILTER_DEFAULT_AFS_COEFF_SHIFT = 192;
static const int   FILTER_DEFAULT_AFS_THRE_SHIFT = 128;
static const int   FILTER_DEFAULT_AFS_THRE_DEINT = 48;
static const int   FILTER_DEFAULT_AFS_THRE_YMOTION = 112;
static const int   FILTER_DEFAULT_AFS_THRE_CMOTION = 224;
static const int   FILTER_DEFAULT_AFS_ANALYZE = 3;
static const bool  FILTER_DEFAULT_AFS_SHIFT = true;
static const bool  FILTER_DEFAULT_AFS_DROP = false;
static const bool  FILTER_DEFAULT_AFS_SMOOTH = false;
static const bool  FILTER_DEFAULT_AFS_FORCE24 = false;
static const bool  FILTER_DEFAULT_AFS_TUNE = false;
static const bool  FILTER_DEFAULT_AFS_RFF = false;
static const int   FILTER_DEFAULT_AFS_TIMECODE = 0;
static const bool  FILTER_DEFAULT_AFS_LOG = false;

static const int   FILTER_DEFAULT_DECIMATE_CYCLE = 5;
static const float FILTER_DEFAULT_DECIMATE_THRE_DUP = 1.1f;
static const float FILTER_DEFAULT_DECIMATE_THRE_SC = 15.0f;
static const int   FILTER_DEFAULT_DECIMATE_BLOCK_X = 32;
static const int   FILTER_DEFAULT_DECIMATE_BLOCK_Y = 32;
static const bool  FILTER_DEFAULT_DECIMATE_PREPROCESSED = false;
static const bool  FILTER_DEFAULT_DECIMATE_CHROMA = true;
static const bool  FILTER_DEFAULT_DECIMATE_LOG = false;

static const int   FILTER_DEFAULT_KNN_RADIUS = 3;
static const float FILTER_DEFAULT_KNN_STRENGTH = 0.08f;
static const float FILTER_DEFAULT_KNN_LERPC = 0.20f;
static const float FILTER_DEFAULT_KNN_WEIGHT_THRESHOLD = 0.01f;
static const float FILTER_DEFAULT_KNN_LERPC_THRESHOLD = 0.80f;

static const float FILTER_DEFAULT_PMD_STRENGTH = 100.0f;
static const float FILTER_DEFAULT_PMD_THRESHOLD = 100.0f;
static const int   FILTER_DEFAULT_PMD_APPLY_COUNT = 2;
static const bool  FILTER_DEFAULT_PMD_USE_EXP = true;

static const int   FILTER_DEFAULT_SMOOTH_QUALITY = 3;
static const int   FILTER_DEFAULT_SMOOTH_QP = 12;
static const float FILTER_DEFAULT_SMOOTH_STRENGTH = 0.0f;
static const float FILTER_DEFAULT_SMOOTH_THRESHOLD = 0.0f;
static const int   FILTER_DEFAULT_SMOOTH_MODE = 0;
static const float FILTER_DEFAULT_SMOOTH_B_RATIO = 0.5f;
static const int   FILTER_DEFAULT_SMOOTH_MAX_QPTABLE_ERR = 10;

static const float FILTER_DEFAULT_TWEAK_BRIGHTNESS = 0.0f;
static const float FILTER_DEFAULT_TWEAK_CONTRAST = 1.0f;
static const float FILTER_DEFAULT_TWEAK_GAMMA = 1.0f;
static const float FILTER_DEFAULT_TWEAK_SATURATION = 1.0f;
static const float FILTER_DEFAULT_TWEAK_HUE = 0.0f;

static const float FILTER_DEFAULT_EDGELEVEL_STRENGTH = 5.0f;
static const float FILTER_DEFAULT_EDGELEVEL_THRESHOLD = 20.0f;
static const float FILTER_DEFAULT_EDGELEVEL_BLACK = 0.0f;
static const float FILTER_DEFAULT_EDGELEVEL_WHITE = 0.0f;

static const int   FILTER_DEFAULT_UNSHARP_RADIUS = 3;
static const float FILTER_DEFAULT_UNSHARP_WEIGHT = 0.5f;
static const float FILTER_DEFAULT_UNSHARP_THRESHOLD = 10.0f;

static const float FILTER_DEFAULT_WARPSHARP_THRESHOLD = 128.0f;
static const int   FILTER_DEFAULT_WARPSHARP_BLUR = 2;
static const int   FILTER_DEFAULT_WARPSHARP_TYPE = 0;
static const float FILTER_DEFAULT_WARPSHARP_DEPTH = 16.0f;
static const int   FILTER_DEFAULT_WARPSHARP_CHROMA = 0;

static const int   FILTER_DEFAULT_DEBAND_RANGE = 15;
static const int   FILTER_DEFAULT_DEBAND_THRE_Y = 15;
static const int   FILTER_DEFAULT_DEBAND_THRE_CB = 15;
static const int   FILTER_DEFAULT_DEBAND_THRE_CR = 15;
static const int   FILTER_DEFAULT_DEBAND_DITHER_Y = 15;
static const int   FILTER_DEFAULT_DEBAND_DITHER_C = 15;
static const int   FILTER_DEFAULT_DEBAND_MODE = 1;
static const int   FILTER_DEFAULT_DEBAND_SEED = 1234;
static const bool  FILTER_DEFAULT_DEBAND_BLUR_FIRST = false;
static const bool  FILTER_DEFAULT_DEBAND_RAND_EACH_FRAME = false;


const CX_DESC list_vpp_denoise[] = {
    { _T("none"),   0 },
    { _T("knn"),    1 },
    { _T("pmd"),    2 },
    { _T("smooth"), 3 },
    { NULL, 0 }
};

const CX_DESC list_vpp_detail_enahance[] = {
    { _T("none"),       0 },
    { _T("unsharp"),    1 },
    { _T("edgelevel"),  2 },
    { _T("warpsharp"),  3 },
    { NULL, 0 }
};

enum HDR2SDRToneMap {
    HDR2SDR_DISABLED,
    HDR2SDR_HABLE,
    HDR2SDR_MOBIUS,
    HDR2SDR_REINHARD,
    HDR2SDR_BT2390,
};

const CX_DESC list_vpp_hdr2sdr[] = {
    { _T("none"),     HDR2SDR_DISABLED },
    { _T("hable"),    HDR2SDR_HABLE },
    { _T("mobius"),   HDR2SDR_MOBIUS },
    { _T("reinhard"), HDR2SDR_REINHARD },
    { _T("bt2390"),   HDR2SDR_BT2390 },
    { NULL, 0 }
};

enum RGY_VPP_RESIZE_MODE {
    RGY_VPP_RESIZE_MODE_DEFAULT,
#if ENCODER_QSV
    RGY_VPP_RESIZE_MODE_MFX_LOWPOWER,
    RGY_VPP_RESIZE_MODE_MFX_QUALITY,
#endif
    RGY_VPP_RESIZE_MODE_UNKNOWN,
};

enum RGY_VPP_RESIZE_ALGO {
    RGY_VPP_RESIZE_AUTO,
    RGY_VPP_RESIZE_BILINEAR,
    RGY_VPP_RESIZE_SPLINE16,
    RGY_VPP_RESIZE_SPLINE36,
    RGY_VPP_RESIZE_SPLINE64,
    RGY_VPP_RESIZE_LANCZOS2,
    RGY_VPP_RESIZE_LANCZOS3,
    RGY_VPP_RESIZE_LANCZOS4,
    RGY_VPP_RESIZE_OPENCL_MAX,
#if ENCODER_QSV
    RGY_VPP_RESIZE_MFX_NEAREST_NEIGHBOR,
    RGY_VPP_RESIZE_MFX_BILINEAR,
    RGY_VPP_RESIZE_MFX_ADVANCED,
    RGY_VPP_RESIZE_MFX_MAX,
#endif
    RGY_VPP_RESIZE_UNKNOWN,
};

enum RGY_VPP_RESIZE_TYPE {
    RGY_VPP_RESIZE_TYPE_NONE,
    RGY_VPP_RESIZE_TYPE_AUTO,
    RGY_VPP_RESIZE_TYPE_OPENCL,
#if ENCODER_QSV
    RGY_VPP_RESIZE_TYPE_MFX,
#endif
    RGY_VPP_RESIZE_TYPE_UNKNOWN,
};

RGY_VPP_RESIZE_TYPE getVppResizeType(RGY_VPP_RESIZE_ALGO resize);


const CX_DESC list_vpp_resize_mode[] = {
    { _T("auto"),     RGY_VPP_RESIZE_MODE_DEFAULT },
#if ENCODER_QSV
    { _T("lowpower"), RGY_VPP_RESIZE_MODE_MFX_LOWPOWER },
    { _T("quality"),  RGY_VPP_RESIZE_MODE_MFX_QUALITY },
#endif
    { NULL, NULL }
};

const CX_DESC list_vpp_resize[] = {
    { _T("auto"),     RGY_VPP_RESIZE_AUTO },
    { _T("spline16"), RGY_VPP_RESIZE_SPLINE16 },
    { _T("spline36"), RGY_VPP_RESIZE_SPLINE36 },
    { _T("spline64"), RGY_VPP_RESIZE_SPLINE64 },
    { _T("lanczos2"), RGY_VPP_RESIZE_LANCZOS2 },
    { _T("lanczos3"), RGY_VPP_RESIZE_LANCZOS3 },
    { _T("lanczos4"), RGY_VPP_RESIZE_LANCZOS4 },
#if ENCODER_QSV
    { _T("bilinear"), RGY_VPP_RESIZE_MFX_BILINEAR },
    { _T("advanced"), RGY_VPP_RESIZE_MFX_ADVANCED },
    { _T("simple"),   RGY_VPP_RESIZE_MFX_NEAREST_NEIGHBOR },
    { _T("fine"),     RGY_VPP_RESIZE_MFX_ADVANCED },
#endif
    { NULL, NULL }
};

enum VppFpPrecision {
    VPP_FP_PRECISION_UNKNOWN = -1,

    VPP_FP_PRECISION_AUTO = 0,
    VPP_FP_PRECISION_FP32,
    VPP_FP_PRECISION_FP16,

    VPP_FP_PRECISION_MAX,
};

const CX_DESC list_vpp_fp_prec[] = {
    { _T("auto"), VPP_FP_PRECISION_AUTO },
    { _T("fp32"), VPP_FP_PRECISION_FP32 },
    { _T("fp16"), VPP_FP_PRECISION_FP16 },
    { NULL, 0 }
};

enum VppNnediField {
    VPP_NNEDI_FIELD_UNKNOWN = 0,
    VPP_NNEDI_FIELD_BOB_AUTO,
    VPP_NNEDI_FIELD_USE_AUTO,
    VPP_NNEDI_FIELD_USE_TOP,
    VPP_NNEDI_FIELD_USE_BOTTOM,
    VPP_NNEDI_FIELD_BOB_TOP_BOTTOM,
    VPP_NNEDI_FIELD_BOB_BOTTOM_TOP,

    VPP_NNEDI_FIELD_MAX,
};

const CX_DESC list_vpp_nnedi_field[] = {
    { _T("bob"),     VPP_NNEDI_FIELD_BOB_AUTO },
    { _T("auto"),    VPP_NNEDI_FIELD_USE_AUTO },
    { _T("top"),     VPP_NNEDI_FIELD_USE_TOP },
    { _T("bottom"),  VPP_NNEDI_FIELD_USE_BOTTOM },
    { _T("bob_tff"), VPP_NNEDI_FIELD_BOB_TOP_BOTTOM },
    { _T("bob_bff"), VPP_NNEDI_FIELD_BOB_BOTTOM_TOP },
    { NULL, 0 }
};

const CX_DESC list_vpp_nnedi_nns[] = {
    { _T("16"),   16 },
    { _T("32"),   32 },
    { _T("64"),   64 },
    { _T("128"), 128 },
    { _T("256"), 256 },
    { NULL, 0 }
};

enum VppNnediNSize {
    VPP_NNEDI_NSIZE_UNKNOWN = -1,

    VPP_NNEDI_NSIZE_8x6 = 0,
    VPP_NNEDI_NSIZE_16x6,
    VPP_NNEDI_NSIZE_32x6,
    VPP_NNEDI_NSIZE_48x6,
    VPP_NNEDI_NSIZE_8x4,
    VPP_NNEDI_NSIZE_16x4,
    VPP_NNEDI_NSIZE_32x4,

    VPP_NNEDI_NSIZE_MAX,
};

const CX_DESC list_vpp_nnedi_nsize[] = {
    { _T("8x6"),  VPP_NNEDI_NSIZE_8x6  },
    { _T("16x6"), VPP_NNEDI_NSIZE_16x6 },
    { _T("32x6"), VPP_NNEDI_NSIZE_32x6 },
    { _T("48x6"), VPP_NNEDI_NSIZE_48x6 },
    { _T("8x4"),  VPP_NNEDI_NSIZE_8x4  },
    { _T("16x4"), VPP_NNEDI_NSIZE_16x4 },
    { _T("32x4"), VPP_NNEDI_NSIZE_32x4 },
    { NULL, 0 }
};

enum VppNnediQuality {
    VPP_NNEDI_QUALITY_UNKNOWN = 0,
    VPP_NNEDI_QUALITY_FAST,
    VPP_NNEDI_QUALITY_SLOW,

    VPP_NNEDI_QUALITY_MAX,
};

const CX_DESC list_vpp_nnedi_quality[] = {
    { _T("fast"), VPP_NNEDI_QUALITY_FAST },
    { _T("slow"), VPP_NNEDI_QUALITY_SLOW },
    { NULL, 0 }
};

enum VppNnediPreScreen : uint32_t {
    VPP_NNEDI_PRE_SCREEN_NONE            = 0x00,
    VPP_NNEDI_PRE_SCREEN_ORIGINAL        = 0x01,
    VPP_NNEDI_PRE_SCREEN_NEW             = 0x02,
    VPP_NNEDI_PRE_SCREEN_MODE            = 0x07,
    VPP_NNEDI_PRE_SCREEN_BLOCK           = 0x10,
    VPP_NNEDI_PRE_SCREEN_ONLY            = 0x20,
    VPP_NNEDI_PRE_SCREEN_ORIGINAL_BLOCK  = VPP_NNEDI_PRE_SCREEN_ORIGINAL | VPP_NNEDI_PRE_SCREEN_BLOCK,
    VPP_NNEDI_PRE_SCREEN_NEW_BLOCK       = VPP_NNEDI_PRE_SCREEN_NEW      | VPP_NNEDI_PRE_SCREEN_BLOCK,
    VPP_NNEDI_PRE_SCREEN_ORIGINAL_ONLY   = VPP_NNEDI_PRE_SCREEN_ORIGINAL | VPP_NNEDI_PRE_SCREEN_ONLY,
    VPP_NNEDI_PRE_SCREEN_NEW_ONLY        = VPP_NNEDI_PRE_SCREEN_NEW      | VPP_NNEDI_PRE_SCREEN_ONLY,

    VPP_NNEDI_PRE_SCREEN_MAX,
};

static VppNnediPreScreen operator|(VppNnediPreScreen a, VppNnediPreScreen b) {
    return (VppNnediPreScreen)((uint32_t)a | (uint32_t)b);
}

static VppNnediPreScreen operator|=(VppNnediPreScreen& a, VppNnediPreScreen b) {
    a = a | b;
    return a;
}

static VppNnediPreScreen operator&(VppNnediPreScreen a, VppNnediPreScreen b) {
    return (VppNnediPreScreen)((uint32_t)a & (uint32_t)b);
}

static VppNnediPreScreen operator&=(VppNnediPreScreen& a, VppNnediPreScreen b) {
    a = (VppNnediPreScreen)((uint32_t)a & (uint32_t)b);
    return a;
}

const CX_DESC list_vpp_nnedi_pre_screen[] = {
    { _T("none"),           VPP_NNEDI_PRE_SCREEN_NONE },
    { _T("original"),       VPP_NNEDI_PRE_SCREEN_ORIGINAL },
    { _T("new"),            VPP_NNEDI_PRE_SCREEN_NEW },
    { _T("original_block"), VPP_NNEDI_PRE_SCREEN_ORIGINAL_BLOCK },
    { _T("new_block"),      VPP_NNEDI_PRE_SCREEN_NEW_BLOCK },
    { _T("original_only"),  VPP_NNEDI_PRE_SCREEN_ORIGINAL_ONLY },
    { _T("new_only"),       VPP_NNEDI_PRE_SCREEN_NEW_ONLY },
    { NULL, 0 }
};

enum VppNnediErrorType {
    VPP_NNEDI_ETYPE_ABS = 0,
    VPP_NNEDI_ETYPE_SQUARE,

    VPP_NNEDI_ETYPE_MAX,
};

const CX_DESC list_vpp_nnedi_error_type[] = {
    { _T("abs"),    VPP_NNEDI_ETYPE_ABS },
    { _T("square"), VPP_NNEDI_ETYPE_SQUARE },
    { NULL, 0 }
};

const CX_DESC list_vpp_deband[] = {
    { _T("0 - 1点参照"),  0 },
    { _T("1 - 2点参照"),  1 },
    { _T("2 - 4点参照"),  2 },
    { NULL, 0 }
};

const CX_DESC list_vpp_rotate[] = {
    { _T("90"),   90 },
    { _T("180"), 180 },
    { _T("270"), 270 },
    { NULL, 0 }
};

const CX_DESC list_vpp_ass_shaping[] = {
    { _T("simple"),  0 },
    { _T("complex"), 1 },
    { NULL, 0 }
};


struct ColorspaceConv {
    VideoVUIInfo from, to;
    double sdr_source_peak;
    bool approx_gamma;
    bool scene_ref;

    ColorspaceConv();
    void set(const VideoVUIInfo& csp_from, const VideoVUIInfo &csp_to) {
        from = csp_from;
        to = csp_to;
    }
    bool operator==(const ColorspaceConv &x) const;
    bool operator!=(const ColorspaceConv &x) const;
};

struct TonemapHable {
    double a, b, c, d, e, f;

    TonemapHable();
    bool operator==(const TonemapHable &x) const;
    bool operator!=(const TonemapHable &x) const;
};

struct TonemapMobius {
    double transition, peak;

    TonemapMobius();
    bool operator==(const TonemapMobius &x) const;
    bool operator!=(const TonemapMobius &x) const;
};

struct TonemapReinhard {
    double contrast, peak;

    TonemapReinhard();
    bool operator==(const TonemapReinhard &x) const;
    bool operator!=(const TonemapReinhard &x) const;
};

struct HDR2SDRParams {
    HDR2SDRToneMap tonemap;
    TonemapHable hable;
    TonemapMobius mobius;
    TonemapReinhard reinhard;
    double ldr_nits;
    double hdr_source_peak;
    double desat_base;
    double desat_strength;
    double desat_exp;

    HDR2SDRParams();
    bool operator==(const HDR2SDRParams &x) const;
    bool operator!=(const HDR2SDRParams &x) const;
};

struct VppColorspace {
    bool enable;
    HDR2SDRParams hdr2sdr;
    vector<ColorspaceConv> convs;

    VppColorspace();
    bool operator==(const VppColorspace &x) const;
    bool operator!=(const VppColorspace &x) const;
};

enum {
    AFS_PRESET_DEFAULT = 0,
    AFS_PRESET_TRIPLE,        //動き重視
    AFS_PRESET_DOUBLE,        //二重化
    AFS_PRESET_ANIME,                     //映画/アニメ
    AFS_PRESET_CINEMA = AFS_PRESET_ANIME, //映画/アニメ
    AFS_PRESET_MIN_AFTERIMG,              //残像最小化
    AFS_PRESET_FORCE24_SD,                //24fps固定
    AFS_PRESET_FORCE24_HD,                //24fps固定 (HD)
    AFS_PRESET_FORCE30,                   //30fps固定
};

const CX_DESC list_afs_preset[] = {
    { _T("default"),      AFS_PRESET_DEFAULT },
    { _T("triple"),       AFS_PRESET_TRIPLE },
    { _T("double"),       AFS_PRESET_DOUBLE },
    { _T("anime/cinema"), AFS_PRESET_ANIME },
    { _T("anime"),        AFS_PRESET_ANIME },
    { _T("cinema"),       AFS_PRESET_CINEMA },
    { _T("min_afterimg"), AFS_PRESET_MIN_AFTERIMG },
    { _T("24fps"),        AFS_PRESET_FORCE24_HD },
    { _T("24fps_sd"),     AFS_PRESET_FORCE24_SD },
    { _T("30fps"),        AFS_PRESET_FORCE30 },
    { NULL, NULL }
};

typedef struct {
    int top, bottom, left, right;
} AFS_SCAN_CLIP;

static inline AFS_SCAN_CLIP scan_clip(int top, int bottom, int left, int right) {
    AFS_SCAN_CLIP clip;
    clip.top = top;
    clip.bottom = bottom;
    clip.left = left;
    clip.right = right;
    return clip;
}

struct VppAfs {
    bool enable;
    int tb_order;
    AFS_SCAN_CLIP clip;    //上下左右
    int method_switch;     //切替点
    int coeff_shift;       //判定比
    int thre_shift;        //縞(ｼﾌﾄ)
    int thre_deint;        //縞(解除)
    int thre_Ymotion;      //Y動き
    int thre_Cmotion;      //C動き
    int analyze;           //解除Lv
    bool shift;            //フィールドシフト
    bool drop;             //間引き
    bool smooth;           //スムージング
    bool force24;          //24fps化
    bool tune;             //調整モード
    bool rff;              //rffフラグを認識して調整
    int timecode;          //timecode出力
    bool log;              //log出力

    VppAfs();
    void set_preset(int preset);
    int read_afs_inifile(const TCHAR *inifile);
    bool operator==(const VppAfs &x) const;
    bool operator!=(const VppAfs &x) const;
    tstring print() const;

    void check();
};

struct VppNnedi {
    bool              enable;
    VppNnediField     field;
    int               nns;
    VppNnediNSize     nsize;
    VppNnediQuality   quality;
    VppFpPrecision precision;
    VppNnediPreScreen pre_screen;
    VppNnediErrorType errortype;
    tstring           weightfile;

    bool isbob();
    VppNnedi();
    bool operator==(const VppNnedi &x) const;
    bool operator!=(const VppNnedi &x) const;
    tstring print() const;
};

const CX_DESC list_vpp_decimate_block[] = {
    { _T("4"),    4 },
    { _T("8"),    8 },
    { _T("16"),  16 },
    { _T("32"),  32 },
    { _T("64"),  64 },
    { NULL, 0 }
};

struct VppDecimate {
    bool enable;
    int cycle;
    float threDuplicate;
    float threSceneChange;
    int blockX;
    int blockY;
    bool preProcessed;
    bool chroma;
    bool log;

    VppDecimate();
    bool operator==(const VppDecimate &x) const;
    bool operator!=(const VppDecimate &x) const;
    tstring print() const;
};

struct VppPad {
    bool enable;
    int left, top, right, bottom;

    VppPad();
    bool operator==(const VppPad &x) const;
    bool operator!=(const VppPad &x) const;
    tstring print() const;
};

struct VppKnn {
    bool  enable;
    int   radius;
    float strength;
    float lerpC;
    float weight_threshold;
    float lerp_threshold;

    VppKnn();
    bool operator==(const VppKnn &x) const;
    bool operator!=(const VppKnn &x) const;
    tstring print() const;
};

struct VppPmd {
    bool  enable;
    float strength;
    float threshold;
    int   applyCount;
    bool  useExp;

    VppPmd();
    bool operator==(const VppPmd &x) const;
    bool operator!=(const VppPmd &x) const;
    tstring print() const;
};

struct VppSmooth {
    bool enable;
    int quality;
    int qp;
    VppFpPrecision prec;
    bool useQPTable;
    float strength;
    float threshold;
    float bratio;
    int maxQPTableErrCount;
    VppSmooth();
    bool operator==(const VppSmooth &x) const;
    bool operator!=(const VppSmooth &x) const;
    tstring print() const;
};

struct VppSubburn {
    bool  enable;
    tstring filename;
    std::string charcode;
    tstring fontsdir;
    int trackId;
    int assShaping;
    float scale;
    float transparency_offset;
    float brightness;
    float contrast;
    double ts_offset;
    bool vid_ts_offset;

    VppSubburn();
    bool operator==(const VppSubburn &x) const;
    bool operator!=(const VppSubburn &x) const;
    tstring print() const;
};

struct VppUnsharp {
    bool  enable;
    int   radius;
    float weight;
    float threshold;

    VppUnsharp();
    bool operator==(const VppUnsharp &x) const;
    bool operator!=(const VppUnsharp &x) const;
    tstring print() const;
};

struct VppEdgelevel {
    bool  enable;
    float strength;
    float threshold;
    float black;
    float white;

    VppEdgelevel();
    bool operator==(const VppEdgelevel &x) const;
    bool operator!=(const VppEdgelevel &x) const;
    tstring print() const;
};

struct VppWarpsharp {
    bool enable;
    float threshold;
    int blur;
    int type;
    float depth;
    int chroma;

    VppWarpsharp();
    bool operator==(const VppWarpsharp& x) const;
    bool operator!=(const VppWarpsharp& x) const;
    tstring print() const;
};

struct VppTweak {
    bool  enable;
    float brightness; // -1.0 - 1.0 (0.0)
    float contrast;   // -2.0 - 2.0 (1.0)
    float gamma;      //  0.1 - 10.0 (1.0)
    float saturation; //  0.0 - 3.0 (1.0)
    float hue;        // -180 - 180 (0.0)

    VppTweak();
    bool operator==(const VppTweak &x) const;
    bool operator!=(const VppTweak &x) const;
    tstring print() const;
};

struct VppTransform {
    bool enable;
    bool transpose;
    bool flipX;
    bool flipY;

    VppTransform();
    int rotate() const;
    bool setRotate(int rotate);
    bool operator==(const VppTransform &x) const;
    bool operator!=(const VppTransform &x) const;
    tstring print() const;
};

struct VppDeband {
    bool enable;
    int range;
    int threY;
    int threCb;
    int threCr;
    int ditherY;
    int ditherC;
    int sample;
    int seed;
    bool blurFirst;
    bool randEachFrame;

    VppDeband();
    bool operator==(const VppDeband &x) const;
    bool operator!=(const VppDeband &x) const;
    tstring print() const;
};

struct RGYParamVpp {
    RGY_VPP_RESIZE_ALGO resize_algo;
    RGY_VPP_RESIZE_MODE resize_mode;
    VppColorspace colorspace;
    VppAfs afs;
    VppNnedi nnedi;
    VppDecimate decimate;
    VppPad pad;
    VppKnn knn;
    VppPmd pmd;
    VppSmooth smooth;
    std::vector<VppSubburn> subburn;
    VppUnsharp unsharp;
    VppEdgelevel edgelevel;
    VppWarpsharp warpsharp;
    VppTweak tweak;
    VppTransform transform;
    VppDeband deband;

    RGYParamVpp();
};


static const char *maxCLLSource = "copy";
static const char *masterDisplaySource = "copy";

static const TCHAR *RGY_METADATA_CLEAR = _T("clear");
static const TCHAR *RGY_METADATA_COPY = _T("copy");

static const int TRACK_SELECT_BY_LANG = -1;

struct AudioSelect {
    int      trackID;         //選択したトラックのリスト 1,2,...(1から連番で指定)
                              // 0 ... 全指定
                              // TRACK_SELECT_BY_LANG ... langによる選択
    tstring  decCodecPrm;     //音声エンコードのデコーダのパラメータ
    tstring  encCodec;        //音声エンコードのコーデック
    tstring  encCodecPrm;     //音声エンコードのコーデックのパラメータ
    tstring  encCodecProfile; //音声エンコードのコーデックのプロファイル
    int      encBitrate;      //音声エンコードに選択した音声トラックのビットレート
    int      encSamplingRate;      //サンプリング周波数
    int      addDelayMs;           //追加する音声の遅延(millisecond)
    tstring  extractFilename;      //抽出する音声のファイル名のリスト
    tstring  extractFormat;        //抽出する音声ファイルのフォーマット
    tstring  filter;               //音声フィルタ
    uint64_t streamChannelSelect[MAX_SPLIT_CHANNELS]; //入力音声の使用するチャンネル
    uint64_t streamChannelOut[MAX_SPLIT_CHANNELS];    //出力音声のチャンネル
    tstring  bsf;                  // 適用するbitstreamfilterの名前
    tstring  disposition;          // 指定のdisposition
    std::string lang;              // 言語選択
    std::vector<tstring> metadata;

    AudioSelect();
    ~AudioSelect() {};
};

struct AudioSource {
    tstring filename;
    std::map<int, AudioSelect> select;

    AudioSource();
    ~AudioSource() {};
};

struct SubtitleSelect {
    int trackID;         // 選択したトラックのリスト 1,2,...(1から連番で指定)
                         //  0 ... 全指定
                         //  TRACK_SELECT_BY_LANG ... langによる選択
    tstring encCodec;
    tstring encCodecPrm;
    tstring decCodecPrm;
    bool asdata;
    tstring bsf;          // 適用するbitstreamfilterの名前
    tstring disposition;  // 指定のdisposition
    std::string lang;         // 言語選択
    std::vector<tstring> metadata;

    SubtitleSelect();
    ~SubtitleSelect() {};
};

struct SubSource {
    tstring filename;
    std::map<int, SubtitleSelect> select;

    SubSource();
    ~SubSource() {};
};

struct DataSelect {
    int trackID;         // 選択したトラックのリスト 1,2,...(1から連番で指定)
                         //  0 ... 全指定
                         //  TRACK_SELECT_BY_LANG ... langによる選択
    tstring disposition; // 指定のdisposition
    std::string lang;    // 言語選択
    std::vector<tstring> metadata;

    DataSelect();
    ~DataSelect() {};
};

using AttachmentSelect = DataSelect;

struct GPUAutoSelectMul {
    float cores;
    float gen;
    float gpu;
    float ve;

    GPUAutoSelectMul();
    bool operator==(const GPUAutoSelectMul &x) const;
    bool operator!=(const GPUAutoSelectMul &x) const;
};

struct RGYParamCommon {
    tstring inputFilename;        //入力ファイル名
    tstring outputFilename;       //出力ファイル名
    tstring muxOutputFormat;      //出力フォーマット
    VideoVUIInfo out_vui;
    RGYOptList inputOpt; //入力オプション

    std::string maxCll;
    std::string masterDisplay;
    CspTransfer atcSei;
    bool hdr10plusMetadataCopy;
    tstring dynamicHdr10plusJson;
    std::string videoCodecTag;
    std::vector<tstring> videoMetadata;
    std::vector<tstring> formatMetadata;
    float seekSec;               //指定された秒数分先頭を飛ばす
    int nSubtitleSelectCount;
    SubtitleSelect **ppSubtitleSelectList;
    std::vector<SubSource> subSource;
    std::vector<AudioSource> audioSource;
    int nAudioSelectCount; //pAudioSelectの数
    AudioSelect **ppAudioSelectList;
    int        nDataSelectCount;
    DataSelect **ppDataSelectList;
    int        nAttachmentSelectCount;
    AttachmentSelect **ppAttachmentSelectList;
    int audioResampler;
    int demuxAnalyzeSec;
    int AVMuxTarget;                       //RGY_MUX_xxx
    int videoTrack;
    int videoStreamId;
    int nTrimCount;
    sTrim *pTrimList;
    bool copyChapter;
    bool keyOnChapter;
    bool chapterNoTrim;
    C2AFormat caption2ass;
    int audioIgnoreDecodeError;
    RGYOptList muxOpt;
    bool disableMp4Opt;
    tstring chapterFile;
    tstring keyFile;
    TCHAR *AVInputFormat;
    RGYAVSync AVSyncMode;     //avsyncの方法 (NV_AVSYNC_xxx)
    bool timecode;
    tstring timecodeFile;

    int outputBufSizeMB;         //出力バッファサイズ

    RGYParamCommon();
    ~RGYParamCommon();
};

struct RGYParamControl {
    int threadCsp;
    int simdCsp;
    tstring logfile;              //ログ出力先
    int loglevel;                 //ログ出力レベル
    bool logFramePosList;     //framePosList出力
    TCHAR *logMuxVidTsFile;
    int threadOutput;
    int threadAudio;
    int threadInput;
    int procSpeedLimit;      //処理速度制限 (0で制限なし)
    int64_t perfMonitorSelect;
    int64_t perfMonitorSelectMatplot;
    int     perfMonitorInterval;
    uint32_t parentProcessID;
    bool lowLatency;
    GPUAutoSelectMul gpuSelect;
    bool skipHWDecodeCheck;
    tstring avsdll;

    RGYParamControl();
    ~RGYParamControl();
};

bool trim_active(const sTrimParam *pTrim);
std::pair<bool, int> frame_inside_range(int frame, const std::vector<sTrim> &trimList);
bool rearrange_trim_list(int frame, int offset, std::vector<sTrim> &trimList);
tstring print_metadata(const std::vector<tstring>& metadata);
bool metadata_copy(const std::vector<tstring> &metadata);
bool metadata_clear(const std::vector<tstring> &metadata);

const CX_DESC list_simd[] = {
    { _T("auto"),     -1  },
    { _T("none"),     NONE },
    { _T("sse2"),     SSE2 },
    { _T("sse3"),     SSE3|SSE2 },
    { _T("ssse3"),    SSSE3|SSE3|SSE2 },
    { _T("sse41"),    SSE41|SSSE3|SSE3|SSE2 },
    { _T("avx"),      AVX|SSE42|SSE41|SSSE3|SSE3|SSE2 },
    { _T("avx2"),     AVX2|AVX|SSE42|SSE41|SSSE3|SSE3|SSE2 },
    { NULL, 0 }
};

template <uint32_t size>
static bool bSplitChannelsEnabled(uint64_t(&streamChannels)[size]) {
    bool bEnabled = false;
    for (uint32_t i = 0; i < size; i++) {
        bEnabled |= streamChannels[i] != 0;
    }
    return bEnabled;
}

template <uint32_t size>
static void setSplitChannelAuto(uint64_t(&streamChannels)[size]) {
    for (uint32_t i = 0; i < size; i++) {
        streamChannels[i] = ((uint64_t)1) << i;
    }
}

template <uint32_t size>
static bool isSplitChannelAuto(uint64_t(&streamChannels)[size]) {
    bool isAuto = true;
    for (uint32_t i = 0; isAuto && i < size; i++) {
        isAuto &= (streamChannels[i] == (((uint64_t)1) << i));
    }
    return isAuto;
}

unique_ptr<RGYHDR10Plus> initDynamicHDR10Plus(const tstring &dynamicHdr10plusJson, shared_ptr<RGYLog> log);

#endif //__RGY_PRM_H__
