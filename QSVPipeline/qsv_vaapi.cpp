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
