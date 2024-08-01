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
#include "rgy_bitstream.h"

tstring get_str_of_tune_bitmask(const uint32_t mask) {
    if (mask == 0) {
        return get_cx_desc(list_enc_tune_quality_mode, mask);
    }
    tstring str;
    for (int i = 0; list_enc_tune_quality_mode[i].desc; i++) {
        if (const uint32_t target = list_enc_tune_quality_mode[i].value; target != 0) {
            if ((mask & target) == target) {
                if (str.length()) str += _T(",");
                str += list_enc_tune_quality_mode[i].desc;
            }
        }
    }
    return str;
}

VppDenoise::VppDenoise() :
    enable(false),
    mode(MFX_DENOISE_MODE_DEFAULT),
    strength(20) {

}

bool VppDenoise::operator==(const VppDenoise &x) const {
    return enable == x.enable
        && mode == x.mode
        && strength == x.strength;
}
bool VppDenoise::operator!=(const VppDenoise &x) const {
    return !(*this == x);
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
    resizeInterp(RGY_VPP_RESIZE_AUTO),
    resizeMode(RGY_VPP_RESIZE_MODE_DEFAULT),
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
    detail(),
    aiSuperRes(),
    percPreEnc(false) {

}

QSVAV1Params::QSVAV1Params() :
    tile_row(0),
    tile_col(0) {

}

tstring printParams(const std::vector<QSVRCParam> &dynamicRC) {
    TStringStream t;
    for (const auto& a : dynamicRC) {
        t << a.print() << std::endl;
    }
    return t.str();
};

QSVRCParam::QSVRCParam() :
    start(-1),
    end(-1),
    encMode(0),
    bitrate(0),
    maxBitrate(0),
    vbvBufSize(0),
    avbrAccuarcy(0),
    avbrConvergence(0),
    qp(),
    icqQuality(-1),
    qvbrQuality(-1) {

}

QSVRCParam::QSVRCParam(
    int encMode_, int bitrate_, int maxBitrate_, int vbvBufSize_,
    int avbrAccuarcy_, int avbrConvergence_,
    RGYQPSet qp_, int icqQuality_, int qvbrQuality_) :
    start(-1),
    end(-1),
    encMode(encMode_),
    bitrate(bitrate_),
    maxBitrate(maxBitrate_),
    vbvBufSize(vbvBufSize_),
    avbrAccuarcy(avbrAccuarcy_),
    avbrConvergence(avbrConvergence_),
    qp(qp_),
    icqQuality(icqQuality_),
    qvbrQuality(qvbrQuality_) {

}

tstring QSVRCParam::print() const {
    TStringStream t;
    if (start >= 0) {
        if (end == INT_MAX || end <= 0) {
            t << "frame=" << start << ":end";
        } else {
            t << "frame=" << start << ":" << end;
        }
        t << ",";
    }
    t << tolowercase(trim(get_chr_from_value(list_rc_mode, encMode))) << "=";
    switch (encMode) {
    case MFX_RATECONTROL_CQP:
        t << qp.qpI << ":" << qp.qpP << ":" << qp.qpB;
        break;
    case MFX_RATECONTROL_ICQ:
    case MFX_RATECONTROL_LA_ICQ:
        t << icqQuality;
        break;
    case MFX_RATECONTROL_AVBR:
        t << bitrate;
        if (avbrAccuarcy > 0) {
            t << ",avbr-accuracy=" << avbrAccuarcy;
        }
        if (avbrConvergence > 0) {
            t << ",avbr-convergence=" << avbrConvergence;
        }
        break;
    case MFX_RATECONTROL_QVBR:
        t << bitrate;
        t << ",qvbr-quality=" << qvbrQuality;
        break;
    case MFX_RATECONTROL_VBR:
    case MFX_RATECONTROL_LA:
    case MFX_RATECONTROL_LA_HRD:
    case MFX_RATECONTROL_VCM:
    default:
        t << bitrate;
        break;

    }
    if (maxBitrate != 0) {
        t << ",max-bitrate=" << maxBitrate;
    }
    return t.str();
}
bool QSVRCParam::operator==(const QSVRCParam &x) const {
    return start == x.start
        && end == x.end
        && encMode == x.encMode
        && bitrate == x.bitrate
        && maxBitrate == x.maxBitrate
        && vbvBufSize == x.vbvBufSize
        && avbrAccuarcy == x.avbrAccuarcy
        && avbrConvergence == x.avbrConvergence
        && qp == x.qp
        && icqQuality == x.icqQuality
        && qvbrQuality == x.qvbrQuality;
}
bool QSVRCParam::operator!=(const QSVRCParam &x) const {
    return !(*this == x);
}

sInputParams::sInputParams() :
    input(),
    inprm(),
    common(),
    ctrl(),
    vpp(),
    vppmfx(),
    device(QSVDeviceNum::AUTO),
    rcParam(QSVRCParam(
        MFX_RATECONTROL_ICQ, QSV_DEFAULT_BITRATE, QSV_DEFAULT_MAX_BITRATE, 0,
        QSV_DEFAULT_ACCURACY, QSV_DEFAULT_CONVERGENCE,
        { QSV_DEFAULT_QPI, QSV_DEFAULT_QPP, QSV_DEFAULT_QPB },
        QSV_DEFAULT_ICQ, QSV_DEFAULT_QVBR)),
    nTargetUsage(QSV_DEFAULT_QUALITY),
    codec(RGY_CODEC_H264),
    CodecProfile(0),
    CodecLevel(0),
    outputDepth(8),
    outputCsp(RGY_CHROMAFMT_YUV420),
    nIdrInterval(0),
    nGOPLength(QSV_DEFAULT_GOP_LEN),
    bopenGOP(false),
    bforceGOPSettings(QSV_DEFAULT_FORCE_GOP_LEN),
    GopRefDist(QSV_GOP_REF_DIST_AUTO),
    nRef(QSV_DEFAULT_REF),
    qpMin(),
    qpMax(),
    dynamicRC(),
    nSlices(0),
    ColorFormat(MFX_FOURCC_NV12),
    memType(HW_MEMORY),
    hyperMode(MFX_HYPERMODE_OFF),
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
    bBPyramid(),
    bMBBRC(),
    extBRC(),
    adaptiveRef(),
    adaptiveLTR(),
    adaptiveCQM(),
    bAdaptiveI(),
    bAdaptiveB(),
    nLookaheadDepth(0),
    nTrellis(0),
    nAsyncDepth(0),
    nLookaheadDS(),
    tuneQuality(MFX_ENCODE_TUNE_OFF),
    scenarioInfo(MFX_SCENARIO_UNKNOWN),
    bDisableTimerPeriodTuning(false),
    intraRefreshCycle(0),
    bNoDeblock(false),
    maxFrameSize(0),
    maxFrameSizeI(0),
    maxFrameSizeP(0),
    nWinBRCSize(0),
    nMVCostScaling(0),
    bDirectBiasAdjust(),
    bGlobalMotionAdjust(false),
    functionMode(QSVFunctionMode::Auto),
    gpuCopy(false),
    nSessionThreads(0),
    nSessionThreadPriority(get_value_from_chr(list_priority, _T("normal"))),
    nVP8Sharpness(0),
    nWeightP(0),
    nWeightB(0),
    nFadeDetect(),
    fallbackRC(true),
    bOutputAud(false),
    bOutputPicStruct(false),
    bufPeriodSEI(false),
    repeatHeaders(),
    pQPOffset(),
    nRepartitionCheck(),
    padding(),
    hevc_ctu(0),
    hevc_sao(0),
    hevc_tskip(0),
    hevc_tier(0),
    hevc_gpb(),
    av1(),
    pythonPath(),
    bBenchmark(false),
    nBenchQuality(QSV_DEFAULT_BENCH) {
    memset(pQPOffset, 0, sizeof(pQPOffset));
    input.vui = VideoVUIInfo();
}

sInputParams::~sInputParams() {

}

void sInputParams::applyDOVIProfile(const RGYDOVIProfile inputProfile) {
#if !FOR_AUO
    if (codec != RGY_CODEC_HEVC) {
        return;
    }
    auto targetDoviProfile = (common.doviProfile == RGY_DOVI_PROFILE_COPY) ? inputProfile : common.doviProfile;
    if (targetDoviProfile == 0) {
        return;
    }
    auto profile = getDOVIProfile(targetDoviProfile);
    if (profile == nullptr) {
        return;
    }

    common.out_vui.setIfUnset(profile->vui);
    if (profile->aud) {
        bOutputAud = true;
    }
    if (profile->HRDSEI) {
        bufPeriodSEI = true;
        bOutputPicStruct = true;
    }
    if (profile->profile == 50) {
        //crQPIndexOffset = 3;
    }
    if (profile->profile == 81) {
        //hdr10sei
        //maxcll
    }
#endif
}
