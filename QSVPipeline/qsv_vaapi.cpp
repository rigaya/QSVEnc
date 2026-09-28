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

#include "qsv_vaapi.h"

#if ENABLE_VAAPI
#include <algorithm>

RGY_ERR qsvVACheckParam(sInputParams& prm, std::shared_ptr<RGYLog> log) {
    const sInputParams defaults;
    auto unsupported = [&](const TCHAR *option) {
        if (log) log->write(RGY_LOG_ERROR, RGY_LOGT_DEV,
            _T("%s is not supported with --backend vaapi.\n"), option);
        return RGY_ERR_UNSUPPORTED;
    };
    if (prm.outputCsp != RGY_CHROMAFMT_YUV420) return unsupported(_T("--output-csp (non-420)"));
    if (prm.outputDepth != 8 && prm.outputDepth != 10) return unsupported(_T("--output-depth (other than 8 / 10)"));
    // 品質指標は既存の QSVMfxDec と MFX セッションを必要とするため、Phase 1 では拒否する。
    if (prm.input.type == RGY_INPUT_FMT_AVHW) return unsupported(_T("--avhw"));
    if (!prm.dynamicRC.empty()) return unsupported(_T("--dynamic-rc"));
    if (prm.rcParam.encMode == MFX_RATECONTROL_LA || prm.rcParam.encMode == MFX_RATECONTROL_LA_ICQ || prm.rcParam.encMode == MFX_RATECONTROL_LA_HRD || prm.rcParam.encMode == MFX_RATECONTROL_VCM) return unsupported(_T("LA / LA-ICQ / LA-HRD / VCM"));
    if (prm.common.adaptResolution != defaults.common.adaptResolution && prm.input.type != RGY_INPUT_FMT_AVSW && prm.input.type != RGY_INPUT_FMT_AVANY)
        return unsupported(_T("--adapt-resolution (non-avsw input)"));
    if (prm.common.adaptResolution != defaults.common.adaptResolution && !prm.ctrl.enableOpenCL)
        return unsupported(_T("--adapt-resolution (OpenCL disabled)"));
    const bool deinterlaceCL = prm.vpp.afs.enable || prm.vpp.nnedi.enable || prm.vpp.yadif.enable
        || prm.vpp.bwdif.enable || prm.vpp.rtgmc.enable || prm.vpp.kfm.enable || prm.vpp.rtgmc_bob.enable
        || prm.vpp.decomb.enable || prm.vpp.onnxDeint.enable || prm.vpp.ivtc.enable;
    if ((prm.input.picstruct & RGY_PICSTRUCT_INTERLACED) != 0 && !deinterlaceCL) return unsupported(_T("--interlace"));
    if (prm.common.metric.enabled()) return unsupported(_T("--ssim / --psnr / --vmaf"));
    if (prm.ctrl.parallelEnc.isEnabled()) return unsupported(_T("--parallel"));
    if (prm.vppmfx.deinterlace != defaults.vppmfx.deinterlace || prm.vppmfx.deinterlaceMode != defaults.vppmfx.deinterlaceMode) return unsupported(_T("--vpp-deinterlace"));
    if (prm.vppmfx.denoise.enable) return unsupported(_T("--vpp-denoise"));
    if (prm.vppmfx.mctf.enable) return unsupported(_T("--vpp-mctf"));
    if (prm.vppmfx.detail.enable) return unsupported(_T("--vpp-detail-enhance"));
    if (prm.vppmfx.imageStabilizer != defaults.vppmfx.imageStabilizer) return unsupported(_T("--vpp-image-stab"));
    if (prm.vppmfx.fpsConversion != defaults.vppmfx.fpsConversion) return unsupported(_T("--vpp-fps-conv"));
    if (prm.vppmfx.rotate != defaults.vppmfx.rotate || prm.vppmfx.halfTurn) return unsupported(_T("--vpp-rotate"));
    if (prm.vppmfx.mirrorType != defaults.vppmfx.mirrorType) return unsupported(_T("--vpp-mirror"));
    if (prm.vppmfx.useProAmp) return unsupported(_T("MFX ProAmp"));
    if (prm.vppmfx.colorspace.enable) return unsupported(_T("--vpp-colorspace (mfx)"));
    if (prm.vppmfx.aiSuperRes.enable) return unsupported(_T("--vpp-ai-superres"));
    if (prm.vppmfx.aiFrameInterpolation.enable) return unsupported(_T("--vpp-ai-frameinterp"));
    if (prm.vppmfx.percPreEnc) return unsupported(_T("--vpp-perc-pre-enc"));
    if (prm.vppmfx.mfxInsertCLCopy != defaults.vppmfx.mfxInsertCLCopy) return unsupported(_T("--vpp-mfx-insert-clcopy"));
    if (isQSVMFXResizeFiter(prm.vpp.resize_algo) || isQSVMFXResizeFiter((RGY_VPP_RESIZE_ALGO)prm.vppmfx.resizeInterp) || prm.vppmfx.resizeMode != defaults.vppmfx.resizeMode) return unsupported(_T("--vpp-resize (mfx)"));
    auto warnUnsupported = [&](const TCHAR *option, bool changed) {
        if (changed && log) log->write(RGY_LOG_WARN, RGY_LOGT_DEV,
            _T("WARN: %s is not supported with --backend vaapi, ignored.\n"), option);
    };
    warnUnsupported(_T("IDR interval"), prm.nIdrInterval != defaults.nIdrInterval);
    warnUnsupported(_T("MVC flags"), prm.MVC_flags != defaults.MVC_flags);
    warnUnsupported(_T("--b-pyramid"), prm.bBPyramid != defaults.bBPyramid);
    warnUnsupported(_T("--mbbrc"), prm.bMBBRC != defaults.bMBBRC);
    warnUnsupported(_T("--extbrc"), prm.extBRC != defaults.extBRC);
    warnUnsupported(_T("--adapt-ref"), prm.adaptiveRef != defaults.adaptiveRef);
    warnUnsupported(_T("--adapt-ltr"), prm.adaptiveLTR != defaults.adaptiveLTR);
    warnUnsupported(_T("--adapt-cqm"), prm.adaptiveCQM != defaults.adaptiveCQM);
    warnUnsupported(_T("--i-adapt"), prm.bAdaptiveI != defaults.bAdaptiveI);
    warnUnsupported(_T("--b-adapt"), prm.bAdaptiveB != defaults.bAdaptiveB);
    warnUnsupported(_T("--weightp"), prm.nWeightP != defaults.nWeightP);
    warnUnsupported(_T("--weightb"), prm.nWeightB != defaults.nWeightB);
    warnUnsupported(_T("--fade-detect"), prm.nFadeDetect != defaults.nFadeDetect);
    warnUnsupported(_T("--trellis"), prm.nTrellis != defaults.nTrellis);
    warnUnsupported(_T("--intra-refresh-cycle"), prm.intraRefreshCycle != defaults.intraRefreshCycle);
    warnUnsupported(_T("--tune"), prm.tuneQuality != defaults.tuneQuality);
    warnUnsupported(_T("--scenario-info"), prm.scenarioInfo != defaults.scenarioInfo);
    warnUnsupported(_T("--open-gop"), prm.openGOP != defaults.openGOP);
    warnUnsupported(_T("--strict-gop"), prm.bforceGOPSettings != defaults.bforceGOPSettings);
    warnUnsupported(_T("--cavlc"), prm.bCAVLC != defaults.bCAVLC);
    warnUnsupported(_T("--rdo"), prm.bRDO != defaults.bRDO);
    warnUnsupported(_T("--bluray"), prm.nBluray != defaults.nBluray);
    warnUnsupported(_T("--repartition-check"), prm.nRepartitionCheck != defaults.nRepartitionCheck);
    warnUnsupported(_T("--max-framesize"), prm.maxFrameSize != defaults.maxFrameSize);
    warnUnsupported(_T("--max-framesize-i"), prm.maxFrameSizeI != defaults.maxFrameSizeI);
    warnUnsupported(_T("--max-framesize-p"), prm.maxFrameSizeP != defaults.maxFrameSizeP);
    warnUnsupported(_T("--la-window-size"), prm.nWinBRCSize != defaults.nWinBRCSize);
    warnUnsupported(_T("--no-deblock"), prm.bNoDeblock != defaults.bNoDeblock);
    warnUnsupported(_T("--ctu"), prm.hevc_ctu != defaults.hevc_ctu);
    warnUnsupported(_T("--sao"), prm.hevc_sao != defaults.hevc_sao);
    warnUnsupported(_T("--tskip"), prm.hevc_tskip != defaults.hevc_tskip);
    warnUnsupported(_T("--hevc-gpb"), prm.hevc_gpb != defaults.hevc_gpb);
    warnUnsupported(_T("--mv-scaling"), prm.nMVCostScaling != defaults.nMVCostScaling || prm.bGlobalMotionAdjust != defaults.bGlobalMotionAdjust);
    warnUnsupported(_T("--direct-bias-adjust"), prm.bDirectBiasAdjust != defaults.bDirectBiasAdjust);
    warnUnsupported(_T("--inter-pred"), prm.nInterPred != defaults.nInterPred);
    warnUnsupported(_T("--intra-pred"), prm.nIntraPred != defaults.nIntraPred);
    warnUnsupported(_T("--mv-precision"), prm.nMVPrecision != defaults.nMVPrecision);
    warnUnsupported(_T("--mv-search"), prm.MVSearchWindow != defaults.MVSearchWindow);
    warnUnsupported(_T("--sharpness"), prm.nVP8Sharpness != defaults.nVP8Sharpness);
    warnUnsupported(_T("--pic-struct"), prm.bOutputPicStruct != defaults.bOutputPicStruct);
    warnUnsupported(_T("--buf-period"), prm.bufPeriodSEI != defaults.bufPeriodSEI);
    warnUnsupported(_T("--repeat-headers"), prm.repeatHeaders != defaults.repeatHeaders);
    warnUnsupported(_T("--la-depth"), prm.nLookaheadDepth != defaults.nLookaheadDepth);
    warnUnsupported(_T("--la-quality"), prm.nLookaheadDS != defaults.nLookaheadDS);
    warnUnsupported(_T("--gpu-copy"), prm.gpuCopy != defaults.gpuCopy);
    warnUnsupported(_T("--session-threads"), prm.nSessionThreads != defaults.nSessionThreads);
    warnUnsupported(_T("--session-thread-priority"), prm.nSessionThreadPriority != defaults.nSessionThreadPriority);
    warnUnsupported(_T("--fallback-rc"), prm.fallbackRC != defaults.fallbackRC);
    warnUnsupported(_T("--workaround-hevc10bit-enctools"), prm.workaroundHevc10bitEnctools != defaults.workaroundHevc10bitEnctools);
    warnUnsupported(_T("--hyper-mode"), prm.hyperMode != defaults.hyperMode);
    warnUnsupported(_T("--avbr-unitsize"), prm.rcParam.avbrConvergence != defaults.rcParam.avbrConvergence);
    warnUnsupported(_T("--avbr-accuracy"), prm.rcParam.avbrAccuarcy != defaults.rcParam.avbrAccuarcy);
    warnUnsupported(_T("--ai-enc-ctrl"), prm.aiEncCtrl.enable != defaults.aiEncCtrl.enable
        || prm.aiEncCtrl.saliencyEncoder != defaults.aiEncCtrl.saliencyEncoder
        || prm.aiEncCtrl.adaptiveTargetUsage != defaults.aiEncCtrl.adaptiveTargetUsage);
    warnUnsupported(_T("--qp-offset"), !std::equal(std::begin(prm.pQPOffset), std::end(prm.pQPOffset), std::begin(defaults.pQPOffset)));
    warnUnsupported(_T("--tile-row"), prm.av1.tile_row != defaults.av1.tile_row);
    warnUnsupported(_T("--tile-col"), prm.av1.tile_col != defaults.av1.tile_col);
    return RGY_ERR_NONE;
}

RGY_ERR qsvVAEncParam(RGYVAEncParam& dst, const sInputParams& prm, RGYDeviceVA *dev,
    int width, int height, rgy_rational<int> fps, rgy_rational<int> sar,
    rgy_rational<int> timebase, const VideoVUIInfo& vui, std::shared_ptr<RGYLog> log) {
    if (dev == nullptr) return RGY_ERR_NULL_PTR;
    dst = RGYVAEncParam();
    dst.codec = prm.codec;
    if (dst.codec != RGY_CODEC_H264 && dst.codec != RGY_CODEC_HEVC
        && dst.codec != RGY_CODEC_VP9 && dst.codec != RGY_CODEC_AV1) {
        if (log) log->write(RGY_LOG_ERROR, RGY_LOGT_DEV, _T("Selected codec is not supported with --backend vaapi.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (!prm.dynamicRC.empty()) {
        if (log) log->write(RGY_LOG_ERROR, RGY_LOGT_DEV, _T("--dynamic-rc is not supported with --backend vaapi.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    const auto& caps = dev->encCaps(prm.codec);
    dst.width = width;
    dst.height = height;
    dst.fps = fps;
    dst.sar = sar;
    dst.timebase = timebase;
    dst.bitdepth = prm.outputDepth;
    dst.vui = vui;
    if (!caps.available || (dst.bitdepth > 8 && !caps.support10bit)) {
        if (log) log->write(RGY_LOG_ERROR, RGY_LOGT_DEV, _T("Selected codec / bit depth is not supported by the VA-API device.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    switch (prm.rcParam.encMode) {
    case MFX_RATECONTROL_CQP: dst.rc = RGY_VA_RC_CQP; break;
    case MFX_RATECONTROL_CBR: dst.rc = RGY_VA_RC_CBR; break;
    case MFX_RATECONTROL_VBR: dst.rc = RGY_VA_RC_VBR; break;
    case MFX_RATECONTROL_ICQ: dst.rc = RGY_VA_RC_ICQ; break;
    case MFX_RATECONTROL_QVBR: dst.rc = RGY_VA_RC_QVBR; break;
    case MFX_RATECONTROL_AVBR: dst.rc = RGY_VA_RC_AVBR; break;
    default:
        if (log) log->write(RGY_LOG_ERROR, RGY_LOGT_DEV, _T("Selected rate control (LA / LA-ICQ / LA-HRD / VCM) is not supported with --backend vaapi.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    dst.qp = prm.rcParam.qp;
    dst.quality = dst.rc == RGY_VA_RC_QVBR ? prm.rcParam.qvbrQuality : prm.rcParam.icqQuality;
    if (!(caps.rcModes & dst.rc)) {
        // ICQ (QSVEnc の既定) を持たないデバイス (i965 など) では、明示の有無にかかわらず CQP に読み替える。
        if (prm.rcParam.encMode == MFX_RATECONTROL_ICQ && (caps.rcModes & RGY_VA_RC_CQP)) {
            if (log) log->write(RGY_LOG_WARN, RGY_LOGT_DEV, _T("ICQ is not supported by the VA-API device; using CQP.\n"));
            dst.rc = RGY_VA_RC_CQP;
            dst.qp = RGYQPSet(prm.rcParam.icqQuality, prm.rcParam.icqQuality, prm.rcParam.icqQuality);
        } else {
            if (log) log->write(RGY_LOG_ERROR, RGY_LOGT_DEV, _T("Selected rate control is not supported by the VA-API device.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }
    // CQP / ICQ ではビットレートは使わないため渡さない (既定値が表示に出てしまうため)。
    if (dst.rc != RGY_VA_RC_CQP && dst.rc != RGY_VA_RC_ICQ) {
        dst.bitrateKbps = prm.rcParam.bitrate;
        dst.maxBitrateKbps = prm.rcParam.maxBitrate;
        dst.vbvBufKbits = prm.rcParam.vbvBufSize;
    }
    dst.compressionLevel = prm.nTargetUsage;
    if (dev->driver() == RGYVADriver::IntelI965 && caps.qualityLevels > 0) {
        dst.compressionLevel = clamp(dst.compressionLevel, 1, caps.qualityLevels);
        if (dst.compressionLevel != prm.nTargetUsage && log)
            log->write(RGY_LOG_WARN, RGY_LOGT_DEV, _T("--quality %d exceeds the i965 quality range; using %d.\n"), prm.nTargetUsage, dst.compressionLevel);
    }
    dst.lowPower = prm.functionMode == QSVFunctionMode::FF ? 1 : prm.functionMode == QSVFunctionMode::PG ? 0 : -1;
    if ((dst.lowPower == 1 && !caps.hasEncSliceLP) || (dst.lowPower == 0 && !caps.hasEncSlice)) {
        if (log) log->write(RGY_LOG_ERROR, RGY_LOGT_DEV, _T("Selected function mode is not supported by the VA-API device.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    dst.gopLen = prm.nGOPLength;
    // QSV の自動値だけ既定の参照距離へ展開する。VA の上限による補正は共通エンコーダが行う。
    const int autoRefDist = prm.codec == RGY_CODEC_AV1 ? QSV_DEFAULT_AV1_GOP_REF_DIST
        : prm.codec == RGY_CODEC_HEVC ? QSV_DEFAULT_HEVC_GOP_REF_DIST : QSV_DEFAULT_H264_GOP_REF_DIST;
    dst.bframes = (prm.GopRefDist == QSV_GOP_REF_DIST_AUTO ? autoRefDist : prm.GopRefDist) - 1;
    dst.refs = prm.nRef > 0 ? prm.nRef : -1;
    dst.slices = prm.nSlices;
    dst.aud = prm.bOutputAud;
    if (prm.qpMin.enable) dst.qpMin = prm.qpMin.qpP;
    if (prm.qpMax.enable) dst.qpMax = prm.qpMax.qpP;
    if ((prm.qpMin.enable && (prm.qpMin.qpI != prm.qpMin.qpP || prm.qpMin.qpB != prm.qpMin.qpP))
        || (prm.qpMax.enable && (prm.qpMax.qpI != prm.qpMax.qpP || prm.qpMax.qpB != prm.qpMax.qpP))) {
        if (log) log->write(RGY_LOG_WARN, RGY_LOGT_DEV, _T("Separate I/P/B --qp-min / --qp-max values are not supported with --backend vaapi; using P values.\n"));
    }
    switch (prm.codec) {
    case RGY_CODEC_H264:
        switch (prm.CodecProfile) {
        case 0: dst.profile = AV_PROFILE_H264_HIGH; break;
        case MFX_PROFILE_AVC_BASELINE: dst.profile = AV_PROFILE_H264_BASELINE; break;
        case MFX_PROFILE_AVC_MAIN: dst.profile = AV_PROFILE_H264_MAIN; break;
        case MFX_PROFILE_AVC_HIGH: dst.profile = AV_PROFILE_H264_HIGH; break;
        default: return RGY_ERR_UNSUPPORTED;
        }
        dst.level = prm.CodecLevel > 0 ? prm.CodecLevel : AV_LEVEL_UNKNOWN;
        break;
    case RGY_CODEC_HEVC:
        switch (prm.CodecProfile) {
        case 0: dst.profile = prm.outputDepth > 8 ? AV_PROFILE_HEVC_MAIN_10 : AV_PROFILE_HEVC_MAIN; break;
        case MFX_PROFILE_HEVC_MAIN: dst.profile = prm.outputDepth > 8 ? AV_PROFILE_HEVC_MAIN_10 : AV_PROFILE_HEVC_MAIN; break;
        case MFX_PROFILE_HEVC_MAIN10: dst.profile = AV_PROFILE_HEVC_MAIN_10; break;
        default: return RGY_ERR_UNSUPPORTED;
        }
        dst.level = prm.CodecLevel > 0 ? (prm.CodecLevel & 0xff) * 3 : AV_LEVEL_UNKNOWN;
        dst.tier = prm.hevc_tier == MFX_TIER_HEVC_HIGH ? 1 : 0;
        break;
    case RGY_CODEC_VP9:
        dst.profile = prm.CodecProfile > 0 ? prm.CodecProfile - MFX_PROFILE_VP9_0
            : prm.outputDepth > 8 ? AV_PROFILE_VP9_2 : AV_PROFILE_VP9_0;
        if (prm.outputDepth > 8 && dst.profile == AV_PROFILE_VP9_0) dst.profile = AV_PROFILE_VP9_2;
        dst.level = prm.CodecLevel > 0 ? prm.CodecLevel : AV_LEVEL_UNKNOWN;
        break;
    case RGY_CODEC_AV1:
        dst.profile = prm.CodecProfile > 0 ? prm.CodecProfile - MFX_PROFILE_AV1_MAIN : AV_PROFILE_AV1_MAIN;
        dst.level = prm.CodecLevel > 0 ? (prm.CodecLevel / 10 - 2) * 4 + prm.CodecLevel % 10 : AV_LEVEL_UNKNOWN;
        break;
    default: return RGY_ERR_UNSUPPORTED;
    }
    if (log) log->write(RGY_LOG_INFO, RGY_LOGT_DEV, _T("VA-API compression_level=%d, low_power=%d.\n"), dst.compressionLevel, dst.lowPower);
    return RGY_ERR_NONE;
}
#endif
