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

sVppParams::sVppParams() :
    bEnable(true),
    bUseResize(false),
    resizeInterp(MFX_INTERPOLATION_DEFAULT),
    resizeMode(MFX_SCALING_MODE_DEFAULT),
    colorspace(),
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
    detail() {

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
    outputDepth(8),
    outputCsp(RGY_CHROMAFMT_YUV420),
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
