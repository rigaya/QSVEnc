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

#include "qsv_prm.h"
#if !FOR_AUO
#include "qsv_query.h"
#endif

static const double FILTER_DEFAULT_COLORSPACE_LDRNITS = 100.0;
static const double FILTER_DEFAULT_COLORSPACE_NOMINAL_SOURCE_PEAK = 100.0;
static const double FILTER_DEFAULT_COLORSPACE_HDR_SOURCE_PEAK = 1000.0;

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

VppDenoise::VppDenoise() :
    enable(false),
    strength(20) {

}

VppMCTF::VppMCTF() :
    enable(false),
    strength(0) {

}

VppDetailEnhance::VppDetailEnhance() :
    enable(false),
    strength(15) {

}

VppDelogo::VppDelogo() :
    pFilePath(nullptr),
    pSelect(nullptr),
    posOffset(std::make_pair(0, 0)),
    depth(QSV_DEFAULT_VPP_DELOGO_DEPTH),
    add(false),
    YOffset(0),
    CbOffset(0),
    CrOffset(0) {
}

VppSubburn::VppSubburn() :
    nTrack(0),
    pFilePath(nullptr),
    pCharEnc(nullptr),
    nShaping(0) {

}

ColorspaceConv::ColorspaceConv() :
    from(),
    to(),
    sdr_source_peak(FILTER_DEFAULT_COLORSPACE_NOMINAL_SOURCE_PEAK),
    approx_gamma(false),
    scene_ref(false) {

}
bool ColorspaceConv::operator==(const ColorspaceConv& x) const {
    return from == x.from
        && to == x.to
        && sdr_source_peak == x.sdr_source_peak
        && approx_gamma == x.approx_gamma
        && scene_ref == x.scene_ref;
}
bool ColorspaceConv::operator!=(const ColorspaceConv& x) const {
    return !(*this == x);
}

TonemapHable::TonemapHable() :
    a(FILTER_DEFAULT_HDR2SDR_HABLE_A),
    b(FILTER_DEFAULT_HDR2SDR_HABLE_B),
    c(FILTER_DEFAULT_HDR2SDR_HABLE_C),
    d(FILTER_DEFAULT_HDR2SDR_HABLE_D),
    e(FILTER_DEFAULT_HDR2SDR_HABLE_E),
    f(FILTER_DEFAULT_HDR2SDR_HABLE_F) {}

bool TonemapHable::operator==(const TonemapHable& x) const {
    return a == x.a
        && b == x.b
        && c == x.c
        && d == x.d
        && e == x.e
        && f == x.f;
}
bool TonemapHable::operator!=(const TonemapHable& x) const {
    return !(*this == x);
}
TonemapMobius::TonemapMobius() :
    transition(FILTER_DEFAULT_HDR2SDR_MOBIUS_TRANSITION),
    peak(FILTER_DEFAULT_HDR2SDR_MOBIUS_PEAK) {
}
bool TonemapMobius::operator==(const TonemapMobius& x) const {
    return transition == x.transition
        && peak == x.peak;
}
bool TonemapMobius::operator!=(const TonemapMobius& x) const {
    return !(*this == x);
}
TonemapReinhard::TonemapReinhard() :
    contrast(FILTER_DEFAULT_HDR2SDR_REINHARD_CONTRAST),
    peak(FILTER_DEFAULT_HDR2SDR_REINHARD_PEAK) {
}
bool TonemapReinhard::operator==(const TonemapReinhard& x) const {
    return contrast == x.contrast
        && peak == x.peak;
}
bool TonemapReinhard::operator!=(const TonemapReinhard& x) const {
    return !(*this == x);
}

HDR2SDRParams::HDR2SDRParams() :
    tonemap(HDR2SDR_DISABLED),
    hable(),
    mobius(),
    reinhard(),
    ldr_nits(FILTER_DEFAULT_COLORSPACE_LDRNITS),
    hdr_source_peak(FILTER_DEFAULT_COLORSPACE_HDR_SOURCE_PEAK) {

}
bool HDR2SDRParams::operator==(const HDR2SDRParams& x) const {
    return tonemap == x.tonemap
        && hable == x.hable
        && mobius == x.mobius
        && reinhard == x.reinhard;
}
bool HDR2SDRParams::operator!=(const HDR2SDRParams& x) const {
    return !(*this == x);
}


VppColorspace::VppColorspace() :
    enable(false),
    hdr2sdr(),
    convs() {

}

bool VppColorspace::operator==(const VppColorspace& x) const {
    if (enable != x.enable
        || x.convs.size() != this->convs.size()) {
        return false;
    }
    for (size_t i = 0; i < x.convs.size(); i++) {
        if (x.convs[i].from != this->convs[i].from
            || x.convs[i].to != this->convs[i].to) {
            return false;
        }
    }
    return true;
}
bool VppColorspace::operator!=(const VppColorspace& x) const {
    return !(*this == x);
}

sVppParams::sVppParams() :
    bEnable(true),
    bUseResize(false),
    scalingQuality(MFX_SCALING_MODE_DEFAULT),
    deinterlace(0),
    telecinePattern(0),
    imageStabilizer(0),
    fpsConversion(0),
    rotate(0),
    halfTurn(false),
    mirrorType(0),
    useProAmp(false),
    denoise(),
    mctf(),
    detail(),
    delogo(),
    subburn() {

}

sInputParams::sInputParams() :
    input(),
    common(),
    ctrl(),
    vpp(),
    vppmfx(),
    nEncMode(MFX_RATECONTROL_CQP),
    nTargetUsage(QSV_DEFAULT_QUALITY),
    CodecId(MFX_CODEC_AVC),
    CodecProfile(0),
    CodecLevel(0),
    nIdrInterval(0),
    nGOPLength(QSV_DEFAULT_GOP_LEN),
    bopenGOP(false),
    bforceGOPSettings(QSV_DEFAULT_FORCE_GOP_LEN),
    nBframes(QSV_BFRAMES_AUTO),
    nRef(QSV_DEFAULT_REF),
    nBitRate(6000),
    nMaxBitrate(15000),
    VBVBufsize(0),
    nQPI(QSV_DEFAULT_QPI),
    nQPP(QSV_DEFAULT_QPP),
    nQPB(QSV_DEFAULT_QPB),
    nQPMin(),
    nQPMax(),
    nAVBRAccuarcy(QSV_DEFAULT_ACCURACY),
    nAVBRConvergence(QSV_DEFAULT_CONVERGENCE),
    nICQQuality(QSV_DEFAULT_ICQ),
    nQVBRQuality(QSV_DEFAULT_QVBR),
    nSlices(0),
    ColorFormat(MFX_FOURCC_NV12),
#if defined(_WIN32) || defined(_WIN64)
    memType(HW_MEMORY),
#else
    memType(SYSTEM_MEMORY),
#endif
    nInputBufSize(QSV_DEFAULT_INPUT_BUF_HW),
    nPAR(),
    bCAVLC(false),
    nInterPred(0),
    nIntraPred(0),
    bRDO(false),
    nMVPrecision(0),
    MVSearchWindow(std::make_pair(0,0)),
    MVC_flags(0),
    nBluray(0),
    bMBBRC(false),
    extBRC(false),
    extBrcAdaptiveLTR(false),
    nLookaheadDepth(0),
    nTrellis(0),
    nAsyncDepth(0),
    nOutputBufSizeMB(QSV_DEFAULT_OUTPUT_BUF_MB),
#if FOR_AUO
    bBPyramid(true),
#else
    bBPyramid(getCPUGenCpuid() >= CPU_GEN_HASWELL),
#endif
    bAdaptiveI(false),
    bAdaptiveB(false),
    nLookaheadDS(),
    bDisableTimerPeriodTuning(false),
    bIntraRefresh(false),
    bNoDeblock(false),
    nWinBRCSize(0),
    nMVCostScaling(0),
    bDirectBiasAdjust(false),
    bGlobalMotionAdjust(false),
    bUseFixedFunc(false),
    nSessionThreads(0),
    nSessionThreadPriority(get_value_from_chr(list_priority, _T("normal"))),
    nVP8Sharpness(0),
    nWeightP(0),
    nWeightB(0),
    nFadeDetect(0),
    nFallback(0),
    bOutputAud(false),
    bOutputPicStruct(false),
    pQPOffset(),
    nRepartitionCheck(0),
    padding(),
    hevc_ctu(0),
    hevc_sao(0),
    hevc_tskip(0),
    hevc_tier(0),
    pythonPath(),
    bBenchmark(false),
    nBenchQuality(QSV_DEFAULT_BENCH)
{
    memset(nQPMin, 0, sizeof(nQPMin));
    memset(nQPMax, 0, sizeof(nQPMax));
    memset(pQPOffset, 0, sizeof(pQPOffset));
    input.vui = VideoVUIInfo();
}

sInputParams::~sInputParams() {

}
