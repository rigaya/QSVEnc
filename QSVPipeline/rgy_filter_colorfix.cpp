// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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
// ------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ColorFix — colour correction filter (three modes)
//
// mode=manual: standard white balance lift/gain (image processing
//   first principles). No external reference source used.
// mode=auto: per-frame chroma mean reduction (YUV domain).
// mode=gray: grayworld color constancy assumption (RGB domain).
//   Based on: E. Land & J. McCann, "Lightness and Retinex Theory" (1971)
//   and the grayworld assumption formulation by G. Buchsbaum (1980).
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include "convert_csp.h"
#include "rgy_avutil.h"
#include "rgy_filter_colorfix.h"
#include "rgy_filter_input_probe.h"

static const int COLORFIX_BLOCK_X = 32;
static const int COLORFIX_BLOCK_Y = 8;
static const int COLORFIX_WG_SIZE = COLORFIX_BLOCK_X * COLORFIX_BLOCK_Y;

RGYFilterColorFix::RGYFilterColorFix(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_colorfix(),
    m_buildOptionsYUV(),
    m_buildOptionsRGB(),
    m_resolvedMatrix(VPP_COLORFIX_MATRIX_BT709),
    m_effectiveSpace(VPP_COLORFIX_SPACE_RGB),
    m_convToRgb(),
    m_convToYuv(),
    m_cspRgb(RGY_CSP_RGB_16),
    m_reducePartials(),
    m_numGroupsLastDispatch(0),
    m_analysisComplete(false),
    m_analysedFrames(0),
    m_skippedFrames(0),
    m_totalSeenFrames(0),
    m_sumA(0), m_sumB(0), m_sumC(0), m_sumY(0), m_sumYsq(0),
    m_rollingVarianceSum(0.0), m_rollingVarianceCount(0),
    m_offsetU(0), m_offsetV(0),
    m_scaleR(1.0f), m_scaleG(1.0f), m_scaleB(1.0f),
    m_prescanUsed(false),
    m_hardCapFrames(0) {
    m_name = _T("colorfix");
}

RGYFilterColorFix::~RGYFilterColorFix() {
    close();
}

RGY_ERR RGYFilterColorFix::checkParam(const std::shared_ptr<RGYFilterParamColorFix> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto &c = prm->colorfix;
    if (c.mode != VPP_COLORFIX_MODE_MANUAL
        && c.mode != VPP_COLORFIX_MODE_AUTO
        && c.mode != VPP_COLORFIX_MODE_GRAY) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid mode=%d: must be 0 (manual), 1 (auto) or 2 (gray).\n"), c.mode);
        return RGY_ERR_INVALID_PARAM;
    }
    if (c.mode == VPP_COLORFIX_MODE_MANUAL) {
        for (int v : { c.whiteR, c.whiteG, c.whiteB, c.blackR, c.blackG, c.blackB }) {
            if (v < 0 || v > 255) {
                AddMessage(RGY_LOG_ERROR, _T("Invalid white/black channel value %d: must be 0..255.\n"), v);
                return RGY_ERR_INVALID_PARAM;
            }
        }
        if (c.whiteR == c.blackR || c.whiteG == c.blackG || c.whiteB == c.blackB) {
            AddMessage(RGY_LOG_ERROR, _T("white and black points must differ on every channel (got R=%d/%d, G=%d/%d, B=%d/%d).\n"),
                c.whiteR, c.blackR, c.whiteG, c.blackG, c.whiteB, c.blackB);
            return RGY_ERR_INVALID_PARAM;
        }
    } else {
        if (c.frames < 10 || c.frames > 5000) {
            AddMessage(RGY_LOG_ERROR, _T("frames=%d must be in [10, 5000].\n"), c.frames);
            return RGY_ERR_INVALID_PARAM;
        }
        if (!(c.strength >= 0.0f && c.strength <= 1.0f)) {
            AddMessage(RGY_LOG_ERROR, _T("strength=%.2f must be in [0.0, 1.0].\n"), c.strength);
            return RGY_ERR_INVALID_PARAM;
        }
        if (!(c.varianceThreshold > 0.0f)) {
            AddMessage(RGY_LOG_ERROR, _T("variance_threshold=%.2f must be > 0.\n"), c.varianceThreshold);
            return RGY_ERR_INVALID_PARAM;
        }
    }
    return RGY_ERR_NONE;
}

int RGYFilterColorFix::resolveMatrix(const VppColorFix &cf, const VideoVUIInfo &vui, int height) const {
    if (cf.matrix != VPP_COLORFIX_MATRIX_AUTO) {
        return cf.matrix;        // CLI override always wins
    }
    // Try stream metadata.
    switch ((int)vui.matrix) {
    case RGY_MATRIX_BT709:        return VPP_COLORFIX_MATRIX_BT709;
    case RGY_MATRIX_ST170_M:
    case RGY_MATRIX_BT470_BG:     return VPP_COLORFIX_MATRIX_BT601;
    case RGY_MATRIX_BT2020_NCL:
    case RGY_MATRIX_BT2020_CL:    return VPP_COLORFIX_MATRIX_BT2020;
    default: break;
    }
    // Resolution fallback.
    if (height <= 576) return VPP_COLORFIX_MATRIX_BT601;
    if (height <= 1200) return VPP_COLORFIX_MATRIX_BT709;
    return VPP_COLORFIX_MATRIX_BT2020;
}

int RGYFilterColorFix::resolveSpace(const VppColorFix &cf) const {
    if (cf.space == VPP_COLORFIX_SPACE_RGB) return VPP_COLORFIX_SPACE_RGB;
    if (cf.space == VPP_COLORFIX_SPACE_YUV) return VPP_COLORFIX_SPACE_YUV;
    // AUTO: rgb for manual, yuv for auto and gray.
    return (cf.mode == VPP_COLORFIX_MODE_MANUAL)
        ? VPP_COLORFIX_SPACE_RGB
        : VPP_COLORFIX_SPACE_YUV;
}

void RGYFilterColorFix::getMatrixCoeffs(int resolvedMatrix, float &Kr, float &Kg, float &Kb) const {
    switch (resolvedMatrix) {
    case VPP_COLORFIX_MATRIX_BT2020: Kr = 0.2627f; Kb = 0.0593f; break;
    case VPP_COLORFIX_MATRIX_BT601:  Kr = 0.299f;  Kb = 0.114f;  break;
    case VPP_COLORFIX_MATRIX_BT709:
    default:                         Kr = 0.2126f; Kb = 0.0722f; break;
    }
    Kg = 1.0f - Kr - Kb;
}

RGY_ERR RGYFilterColorFix::setupCspConverters(const RGYFrameInfo &frameIn, RGY_CSP cspRgb, rgy_rational<int> baseFps) {
    // Hard guard: never instantiate cspconv sub-filters when the
    // effective working space is YUV. The caller is supposed to gate
    // the call (and currently does), but if any future code path
    // ever reaches here with space=yuv, no-op rather than silently
    // adding YUV<->RGB conversions that the YUV pipeline doesn't
    // need. Make sure the converters are null on exit too.
    if (m_effectiveSpace == VPP_COLORFIX_SPACE_YUV) {
        m_convToRgb.reset();
        m_convToYuv.reset();
        AddMessage(RGY_LOG_DEBUG,
            _T("setupCspConverters skipped (effective space=yuv).\n"));
        return RGY_ERR_NONE;
    }

    AddMessage(RGY_LOG_DEBUG,
        _T("setupCspConverters building YUV<->RGB sub-filters (effective space=rgb).\n"));

    auto vuiForMatrix = VideoVUIInfo();
    switch (m_resolvedMatrix) {
    case VPP_COLORFIX_MATRIX_BT601:  vuiForMatrix.matrix = RGY_MATRIX_ST170_M;     break;
    case VPP_COLORFIX_MATRIX_BT709:  vuiForMatrix.matrix = RGY_MATRIX_BT709;       break;
    case VPP_COLORFIX_MATRIX_BT2020: vuiForMatrix.matrix = RGY_MATRIX_BT2020_NCL;  break;
    }

    // YUV -> RGB
    {
        auto filterCrop = std::make_unique<RGYFilterCspCrop>(m_cl);
        auto paramCrop = std::make_shared<RGYFilterParamCrop>();
        paramCrop->frameIn = frameIn;
        paramCrop->frameOut = frameIn;
        paramCrop->frameOut.csp = cspRgb;
        paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->baseFps = baseFps;
        paramCrop->matrix = vuiForMatrix.matrix;
        paramCrop->bOutOverwrite = false;
        auto sts = filterCrop->init(paramCrop, m_pLog);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to init colorfix YUV->RGB converter: %s.\n"), get_err_mes(sts));
            return sts;
        }
        m_convToRgb = std::move(filterCrop);
    }
    // RGB -> YUV
    {
        auto filterCrop = std::make_unique<RGYFilterCspCrop>(m_cl);
        auto paramCrop = std::make_shared<RGYFilterParamCrop>();
        paramCrop->frameIn = frameIn;
        paramCrop->frameIn.csp = cspRgb;
        paramCrop->frameOut = frameIn;
        paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->baseFps = baseFps;
        paramCrop->matrix = vuiForMatrix.matrix;
        paramCrop->bOutOverwrite = false;
        auto sts = filterCrop->init(paramCrop, m_pLog);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to init colorfix RGB->YUV converter: %s.\n"), get_err_mes(sts));
            return sts;
        }
        m_convToYuv = std::move(filterCrop);
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterColorFix::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamColorFix>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    // Colorfix overwrites the frame in place (matches tweak's pattern).
    if (!prm->bOutOverwrite) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid param: colorfix is in-place; bOutOverwrite must be true.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    prm->frameOut = prm->frameIn;
    prm->frameOut.picstruct = prm->frameIn.picstruct;

    const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    const int maxVal   = (1 << bitDepth) - 1;
    m_resolvedMatrix   = resolveMatrix(prm->colorfix, prm->vui, prm->frameIn.height);
    m_effectiveSpace   = resolveSpace(prm->colorfix);

    // auto mode is a chroma-only correction with no meaningful RGB-space
    // equivalent. If the user asked for space=rgb on mode=auto, fall
    // back to YUV with a warning.
    if (prm->colorfix.mode == VPP_COLORFIX_MODE_AUTO
        && m_effectiveSpace == VPP_COLORFIX_SPACE_RGB) {
        AddMessage(RGY_LOG_WARN, _T("mode=auto only supports space=yuv; ignoring space=rgb.\n"));
        m_effectiveSpace = VPP_COLORFIX_SPACE_YUV;
    }

    // Build options. Both YUV and RGB kernels use scalar Type=uchar/ushort;
    // RGB runs over planar R/G/B intermediate frames.
    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamColorFix>(m_param);
    const bool needRebuild = !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
        || prmPrev->colorfix.mode != prm->colorfix.mode
        || resolveSpace(prmPrev->colorfix) != m_effectiveSpace;
    if (needRebuild) {
        // For mode=auto the kernels access the YUV planes directly.
        // For mode=manual/gray the kernels operate on the RGB intermediate;
        // we always build the RGB variant — the YUV-only auto path uses
        // a separate Type=uchar variant for the reduce/apply UV kernels.
        // We'll build BOTH compilations for simplicity. (Two programs.)
        m_buildOptionsYUV = strsprintf(
            "-D Type=%s -D bit_depth=%d -D max_val=%d -D colorfix_block_x=%d -D colorfix_block_y=%d",
            bitDepth > 8 ? "ushort" : "uchar",
            bitDepth, maxVal,
            COLORFIX_BLOCK_X, COLORFIX_BLOCK_Y);
        const int bitDepthRgb = (bitDepth > 8) ? 16 : 8;
        const int maxValRgb   = (1 << bitDepthRgb) - 1;
        m_buildOptionsRGB = strsprintf(
            "-D Type=%s -D bit_depth=%d -D max_val=%d -D colorfix_block_x=%d -D colorfix_block_y=%d",
            bitDepthRgb > 8 ? "ushort" : "uchar",
            bitDepthRgb, maxValRgb,
            COLORFIX_BLOCK_X, COLORFIX_BLOCK_Y);

        // Build the program with whichever defines suit the effective
        // working space. Both RGB and YUV paths operate on scalar planar
        // samples; the bit depth differs for the RGB intermediate.
        const std::string useOptions = (m_effectiveSpace == VPP_COLORFIX_SPACE_RGB)
            ? m_buildOptionsRGB : m_buildOptionsYUV;
        AddMessage(RGY_LOG_DEBUG, _T("Starting async build for RGY_FILTER_COLORFIX_CL (%s).\n"),
            (m_effectiveSpace == VPP_COLORFIX_SPACE_RGB) ? _T("rgb-defines") : _T("yuv-defines"));
        m_colorfix.set(m_cl->buildResourceAsync(_T("RGY_FILTER_COLORFIX_CL"), _T("EXE_DATA"), useOptions.c_str()));
    }

    // Set up the CSP converters only when the effective working space is RGB.
    if (m_effectiveSpace == VPP_COLORFIX_SPACE_RGB) {
        // Pick an RGB CSP matching the source bit depth.
        m_cspRgb = (bitDepth > 8) ? RGY_CSP_RGB_16 : RGY_CSP_RGB;
        sts = setupCspConverters(prm->frameIn, m_cspRgb, prm->baseFps);
        if (sts != RGY_ERR_NONE) return sts;
    } else {
        m_convToRgb.reset();
        m_convToYuv.reset();
    }

    // Allocate the reduction partial-sum buffer for auto / gray modes.
    if (prm->colorfix.mode != VPP_COLORFIX_MODE_MANUAL) {
        // Dispatch grid for the reduce kernel: we'll use the full chroma /
        // RGB resolution (whichever is smaller of the two for safety).
        // For auto: dispatch over chroma plane (W/2 × H/2 for 4:2:0).
        // For gray: dispatch over RGB resolution (W × H).
        // Compute the upper-bound number of work-groups for either case.
        const int rgbWg = ((prm->frameIn.width  + COLORFIX_BLOCK_X - 1) / COLORFIX_BLOCK_X)
                       * ((prm->frameIn.height + COLORFIX_BLOCK_Y - 1) / COLORFIX_BLOCK_Y);
        const int maxLongsPerGroup = 5;       // worst case (reduce_rgb)
        const size_t bufBytes = (size_t)rgbWg * maxLongsPerGroup * sizeof(long long);
        m_reducePartials = m_cl->createBuffer(bufBytes, CL_MEM_READ_WRITE);
        if (!m_reducePartials) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate colorfix reduction buffer (%zu bytes).\n"), bufBytes);
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    // Reset state machine.
    m_analysisComplete    = false;
    m_analysedFrames      = 0;
    m_skippedFrames       = 0;
    m_totalSeenFrames     = 0;
    m_sumA = m_sumB = m_sumC = m_sumY = m_sumYsq = 0;
    m_rollingVarianceSum  = 0.0;
    m_rollingVarianceCount = 0;
    m_offsetU = m_offsetV = 0;
    m_scaleR = m_scaleG = m_scaleB = 1.0f;
    m_prescanUsed   = false;
    m_hardCapFrames = prm->colorfix.frames * 3 + 10;

    // Init-time libav pre-scan for the YUV-space auto / gray paths.
    // The pre-scan opens a private AVFormatContext against the source
    // file, decodes the first `frames=` frames sequentially, and
    // computes m_offsetU / m_offsetV on the host CPU. On success the
    // runtime path applies the correction from frame 1 with no ramp
    // and no analysis discontinuity. On pipe / non-file inputs or any
    // libav failure the pre-scan returns RGY_ERR_UNSUPPORTED and the
    // runtime path falls through to the streaming pre-roll with a
    // strength ramp (see run_filter()). The RGB-space gray path is
    // not pre-scanned because its reduction runs after YUV->RGB
    // conversion on the GPU; that path keeps the streaming pre-roll
    // and gets the ramp as well.
    const bool wantPreScan =
           prm->colorfix.mode == VPP_COLORFIX_MODE_AUTO
        || (prm->colorfix.mode == VPP_COLORFIX_MODE_GRAY
            && m_effectiveSpace == VPP_COLORFIX_SPACE_YUV);
    if (wantPreScan) {
        const auto preErr = runPreScanLibav(prm);
        if (preErr == RGY_ERR_NONE) {
            m_prescanUsed     = true;
            m_analysisComplete = true;
        } else {
            AddMessage(RGY_LOG_DEBUG,
                _T("init-time pre-scan unavailable; ramp fallback at runtime.\n"));
        }
    }

    // Manual mode: compute scale + offset host-side now (no analysis needed).
    if (prm->colorfix.mode == VPP_COLORFIX_MODE_MANUAL) {
        // HBD scaling: user values are 0..255 (8-bit picker).
        const int rgbMax = (RGY_CSP_BIT_DEPTH[m_cspRgb] > 8) ? 65535 : 255;
        const int wR = prm->colorfix.whiteR * rgbMax / 255;
        const int wG = prm->colorfix.whiteG * rgbMax / 255;
        const int wB = prm->colorfix.whiteB * rgbMax / 255;
        const int kR = prm->colorfix.blackR * rgbMax / 255;
        const int kG = prm->colorfix.blackG * rgbMax / 255;
        const int kB = prm->colorfix.blackB * rgbMax / 255;
        m_scaleR = (float)rgbMax / (float)(wR - kR);
        m_scaleG = (float)rgbMax / (float)(wG - kG);
        m_scaleB = (float)rgbMax / (float)(wB - kB);
        m_offsetU = 0;
        m_offsetV = 0;
        // The "offset" used by the RGB-apply kernel is (-black * scale).
        // We pass it as offsetR/offsetG/offsetB in runApplyRGB below;
        // store the precomputed values in m_offsetU / V / scaleX. To keep
        // the cpp tidy, recompute on the fly in runApplyRGB.
        m_analysisComplete = true;
    }

    AddMessage(RGY_LOG_DEBUG,
        _T("colorfix init: src=%dx%d bitDepth=%d, mode=%s, matrix=%s, space=%s\n"),
        prm->frameIn.width, prm->frameIn.height, bitDepth,
        (prm->colorfix.mode == VPP_COLORFIX_MODE_MANUAL) ? _T("manual")
            : (prm->colorfix.mode == VPP_COLORFIX_MODE_AUTO) ? _T("auto") : _T("gray"),
        (m_resolvedMatrix == VPP_COLORFIX_MATRIX_BT601) ? _T("bt601")
            : (m_resolvedMatrix == VPP_COLORFIX_MATRIX_BT709) ? _T("bt709") : _T("bt2020"),
        (m_effectiveSpace == VPP_COLORFIX_SPACE_RGB) ? _T("rgb") : _T("yuv"));

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterColorFix::runApplyRGB(RGYFrameInfo *pTarget,
                                        float scaleR, float scaleG, float scaleB,
                                        float offsetR, float offsetG, float offsetB,
                                        RGYOpenCLQueue &queue,
                                        const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto pR = getPlane(pTarget, RGY_PLANE_R);
    const auto pG = getPlane(pTarget, RGY_PLANE_G);
    const auto pB = getPlane(pTarget, RGY_PLANE_B);
    RGYWorkSize local(COLORFIX_BLOCK_X, COLORFIX_BLOCK_Y);
    RGYWorkSize global(pTarget->width, pTarget->height);
    auto err = m_colorfix.get()->kernel("colorfix_apply_rgb")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)pR.ptr[0],
            (cl_mem)pG.ptr[0],
            (cl_mem)pB.ptr[0], pR.pitch[0],
            pTarget->width, pTarget->height,
            scaleR, scaleG, scaleB,
            offsetR, offsetG, offsetB);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at colorfix_apply_rgb: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterColorFix::runReduceUV(RGYFrameInfo *pSrc,
                                        RGYOpenCLQueue &queue,
                                        const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto pY = getPlane(pSrc, RGY_PLANE_Y);
    const auto pU = getPlane(pSrc, RGY_PLANE_U);
    const auto pV = getPlane(pSrc, RGY_PLANE_V);
    const int subX = std::max(1, pY.width  / pU.width);
    const int subY = std::max(1, pY.height / pU.height);

    RGYWorkSize local(COLORFIX_BLOCK_X, COLORFIX_BLOCK_Y);
    RGYWorkSize global(pU.width, pU.height);
    const int wgX = ((pU.width  + COLORFIX_BLOCK_X - 1) / COLORFIX_BLOCK_X);
    const int wgY = ((pU.height + COLORFIX_BLOCK_Y - 1) / COLORFIX_BLOCK_Y);
    m_numGroupsLastDispatch = wgX * wgY;

    auto err = m_colorfix.get()->kernel("colorfix_reduce_uv")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)pY.ptr[0], pY.pitch[0], pY.width, pY.height,
            (cl_mem)pU.ptr[0], pU.pitch[0], pU.width, pU.height,
            (cl_mem)pV.ptr[0], pV.pitch[0],
            subX, subY,
            m_reducePartials->mem());
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at colorfix_reduce_uv: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterColorFix::runReduceRGB(RGYFrameInfo *pSrc,
                                         RGYOpenCLQueue &queue,
                                         const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto pR = getPlane(pSrc, RGY_PLANE_R);
    const auto pG = getPlane(pSrc, RGY_PLANE_G);
    const auto pB = getPlane(pSrc, RGY_PLANE_B);
    RGYWorkSize local(COLORFIX_BLOCK_X, COLORFIX_BLOCK_Y);
    RGYWorkSize global(pSrc->width, pSrc->height);
    const int wgX = ((pSrc->width  + COLORFIX_BLOCK_X - 1) / COLORFIX_BLOCK_X);
    const int wgY = ((pSrc->height + COLORFIX_BLOCK_Y - 1) / COLORFIX_BLOCK_Y);
    m_numGroupsLastDispatch = wgX * wgY;

    auto err = m_colorfix.get()->kernel("colorfix_reduce_rgb")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)pR.ptr[0],
            (cl_mem)pG.ptr[0],
            (cl_mem)pB.ptr[0], pR.pitch[0],
            pSrc->width, pSrc->height,
            m_reducePartials->mem());
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at colorfix_reduce_rgb: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterColorFix::runApplyUV(RGYFrameInfo *pTarget,
                                       int offsetU, int offsetV,
                                       RGYOpenCLQueue &queue,
                                       const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto pU = getPlane(pTarget, RGY_PLANE_U);
    const auto pV = getPlane(pTarget, RGY_PLANE_V);

    RGYWorkSize local(COLORFIX_BLOCK_X, COLORFIX_BLOCK_Y);
    RGYWorkSize global(pU.width, pU.height);
    auto err = m_colorfix.get()->kernel("colorfix_apply_uv")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)pU.ptr[0], pU.pitch[0],
            (cl_mem)pV.ptr[0], pV.pitch[0],
            pU.width, pU.height,
            offsetU, offsetV);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at colorfix_apply_uv: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterColorFix::runApplyLuma(RGYFrameInfo *pTarget,
                                         float scaleY, float offsetY,
                                         RGYOpenCLQueue &queue,
                                         const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto pY = getPlane(pTarget, RGY_PLANE_Y);
    RGYWorkSize local(COLORFIX_BLOCK_X, COLORFIX_BLOCK_Y);
    RGYWorkSize global(pY.width, pY.height);
    auto err = m_colorfix.get()->kernel("colorfix_apply_luma")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)pY.ptr[0], pY.pitch[0],
            pY.width, pY.height,
            scaleY, offsetY);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at colorfix_apply_luma: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterColorFix::finaliseReduction(RGYOpenCLQueue &queue, int numLongsPerGroup,
                                              std::vector<long long> &out_totals) {
    const size_t bytes = (size_t)m_numGroupsLastDispatch * numLongsPerGroup * sizeof(long long);
    std::vector<long long> host(m_numGroupsLastDispatch * numLongsPerGroup, 0);

    RGYOpenCLEvent readEvent;
    auto clerr = clEnqueueReadBuffer(queue.get(), m_reducePartials->mem(),
        CL_FALSE, 0, bytes, host.data(), 0, nullptr, readEvent.reset_ptr());
    if (clerr != CL_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("readback failed: %s.\n"), cl_errmes(clerr));
        return err_cl_to_rgy(clerr);
    }
    readEvent.wait();

    out_totals.assign(numLongsPerGroup, 0LL);
    for (int g = 0; g < m_numGroupsLastDispatch; g++) {
        for (int i = 0; i < numLongsPerGroup; i++) {
            out_totals[i] += host[g * numLongsPerGroup + i];
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterColorFix::runPreScanLibav(const std::shared_ptr<RGYFilterParamColorFix> &prm) {
    if (!prm || prm->inputFilePath.empty()) {
        return RGY_ERR_UNSUPPORTED;
    }
    std::string fileUtf8;
    if (tchar_to_string(prm->inputFilePath.c_str(), fileUtf8, CP_UTF8) == 0) {
        AddMessage(RGY_LOG_DEBUG, _T("pre-scan: utf-8 conversion failed.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (const char *proto = unsupportedProbeProtocol(fileUtf8); proto != nullptr) {
        AddMessage(RGY_LOG_DEBUG,
            _T("pre-scan: input uses %s protocol; ramp fallback at runtime.\n"),
            char_to_tstring(proto).c_str());
        return RGY_ERR_UNSUPPORTED;
    }

    const int savedAvLogLevel = av_log_get_level();
    av_log_set_level(AV_LOG_FATAL);
    struct AvLogLevelRestorer { int prev; ~AvLogLevelRestorer() { av_log_set_level(prev); } } avGuard{savedAvLogLevel};

    AVFormatContext *fmtCtxRaw = nullptr;
    if (avformat_open_input(&fmtCtxRaw, fileUtf8.c_str(), nullptr, nullptr) < 0) {
        AddMessage(RGY_LOG_DEBUG, _T("pre-scan: avformat_open_input failed.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    std::unique_ptr<AVFormatContext, RGYAVDeleter<AVFormatContext>> fmtGuard(
        fmtCtxRaw, RGYAVDeleter<AVFormatContext>(avformat_close_input));
    AVFormatContext *fmtCtx = fmtGuard.get();

    if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
        AddMessage(RGY_LOG_DEBUG, _T("pre-scan: avformat_find_stream_info failed.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    const int videoIdx = av_find_best_stream(fmtCtx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (videoIdx < 0) {
        AddMessage(RGY_LOG_DEBUG, _T("pre-scan: no video stream.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    AVStream *vst = fmtCtx->streams[videoIdx];
    const AVCodec *codec = avcodec_find_decoder(vst->codecpar->codec_id);
    if (!codec) {
        AddMessage(RGY_LOG_DEBUG, _T("pre-scan: decoder unavailable for stream.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    AVCodecContext *codecCtxRaw = avcodec_alloc_context3(codec);
    if (!codecCtxRaw) return RGY_ERR_NULL_PTR;
    std::unique_ptr<AVCodecContext, RGYAVDeleter<AVCodecContext>> codecGuard(
        codecCtxRaw, RGYAVDeleter<AVCodecContext>(avcodec_free_context));
    AVCodecContext *codecCtx = codecGuard.get();
    if (avcodec_parameters_to_context(codecCtx, vst->codecpar) < 0) {
        AddMessage(RGY_LOG_DEBUG, _T("pre-scan: avcodec_parameters_to_context failed.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    codecCtx->time_base    = vst->time_base;
    codecCtx->pkt_timebase = vst->time_base;
    if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
        AddMessage(RGY_LOG_DEBUG, _T("pre-scan: avcodec_open2 failed.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    AVPacket *pktRaw = av_packet_alloc();
    std::unique_ptr<AVPacket, RGYAVDeleter<AVPacket>> pktGuard(
        pktRaw, RGYAVDeleter<AVPacket>(av_packet_free));
    AVFrame *frameRaw = av_frame_alloc();
    std::unique_ptr<AVFrame, RGYAVDeleter<AVFrame>> frameGuard(
        frameRaw, RGYAVDeleter<AVFrame>(av_frame_free));

    const int wantFrames = prm->colorfix.frames;
    const int seenCap    = wantFrames * 3 + 10;
    const int targetBitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    const double targetMax     = (double)((1 << targetBitDepth) - 1);
    const double neutralTarget = (double)(1 << (targetBitDepth - 1));

    uint64_t sumU = 0, sumV = 0;
    uint64_t totalChromaPx = 0;
    double   rollingVarSum = 0.0;
    int      rollingVarCount = 0;
    int      analysedFrames  = 0;
    int      skippedFrames   = 0;
    int      seenFrames      = 0;
    int      srcBitDepth     = 0;

    // Per-frame host reduction. Uses the AVPixFmtDescriptor to dispatch
    // across planar (YUV420P, YUV422P, YUV444P + LE HBD variants) and
    // semi-planar (NV12, NV21, P010, P016) layouts in a single path.
    // Sums are accumulated in the source's native bit-depth domain via
    // `(raw >> shift) & ((1<<depth)-1)`; final offset is then rescaled
    // to the target encoder bit depth so that mismatched source/target
    // bit depths still produce a valid offset.
    auto processFrame = [&](AVFrame *f) -> RGY_ERR {
        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get((AVPixelFormat)f->format);
        if (!desc || desc->nb_components < 3
            || (desc->flags & AV_PIX_FMT_FLAG_RGB) != 0
            || (desc->flags & AV_PIX_FMT_FLAG_PAL) != 0
            || (desc->flags & AV_PIX_FMT_FLAG_BITSTREAM) != 0
            || (desc->flags & AV_PIX_FMT_FLAG_HWACCEL) != 0) {
            return RGY_ERR_UNSUPPORTED;
        }
        const int depthLuma   = desc->comp[0].depth;
        const int depthChroma = desc->comp[1].depth;
        if (depthLuma == 0 || depthChroma == 0 || depthLuma > 16 || depthChroma > 16) {
            return RGY_ERR_UNSUPPORTED;
        }
        if (srcBitDepth == 0) srcBitDepth = depthLuma;

        const int planeY = desc->comp[0].plane;
        const int planeU = desc->comp[1].plane;
        const int planeV = desc->comp[2].plane;
        const int stepY  = desc->comp[0].step;
        const int stepU  = desc->comp[1].step;
        const int stepV  = desc->comp[2].step;
        const int offY   = desc->comp[0].offset;
        const int offU   = desc->comp[1].offset;
        const int offV   = desc->comp[2].offset;
        const int shY    = desc->comp[0].shift;
        const int shU    = desc->comp[1].shift;
        const int shV    = desc->comp[2].shift;
        const int chromaShiftW = desc->log2_chroma_w;
        const int chromaShiftH = desc->log2_chroma_h;
        const int chromaW = (f->width  + (1 << chromaShiftW) - 1) >> chromaShiftW;
        const int chromaH = (f->height + (1 << chromaShiftH) - 1) >> chromaShiftH;
        const int lumaW   = f->width;
        const int lumaH   = f->height;
        const bool hbdLuma   = depthLuma   > 8;
        const bool hbdChroma = depthChroma > 8;
        const uint32_t maskLuma   = (1U << depthLuma)   - 1U;
        const uint32_t maskChroma = (1U << depthChroma) - 1U;

        if (!f->data[planeU] || !f->data[planeV] || !f->data[planeY]) {
            return RGY_ERR_UNSUPPORTED;
        }

        uint64_t frameSumU = 0, frameSumV = 0;
        for (int y = 0; y < chromaH; ++y) {
            const uint8_t *rowU = f->data[planeU] + (size_t)y * f->linesize[planeU];
            const uint8_t *rowV = f->data[planeV] + (size_t)y * f->linesize[planeV];
            for (int x = 0; x < chromaW; ++x) {
                uint32_t u, v;
                if (hbdChroma) {
                    const uint8_t *pU = rowU + (size_t)x * stepU + offU;
                    const uint8_t *pV = rowV + (size_t)x * stepV + offV;
                    const uint32_t rawU = (uint32_t)pU[0] | ((uint32_t)pU[1] << 8);
                    const uint32_t rawV = (uint32_t)pV[0] | ((uint32_t)pV[1] << 8);
                    u = (rawU >> shU) & maskChroma;
                    v = (rawV >> shV) & maskChroma;
                } else {
                    u = ((uint32_t)rowU[(size_t)x * stepU + offU] >> shU) & maskChroma;
                    v = ((uint32_t)rowV[(size_t)x * stepV + offV] >> shV) & maskChroma;
                }
                frameSumU += u;
                frameSumV += v;
            }
        }

        uint64_t frameSumY = 0, frameSumYsq = 0;
        for (int y = 0; y < lumaH; ++y) {
            const uint8_t *rowY = f->data[planeY] + (size_t)y * f->linesize[planeY];
            for (int x = 0; x < lumaW; ++x) {
                uint32_t yv;
                if (hbdLuma) {
                    const uint8_t *pY = rowY + (size_t)x * stepY + offY;
                    const uint32_t rawY = (uint32_t)pY[0] | ((uint32_t)pY[1] << 8);
                    yv = (rawY >> shY) & maskLuma;
                } else {
                    yv = ((uint32_t)rowY[(size_t)x * stepY + offY] >> shY) & maskLuma;
                }
                frameSumY   += yv;
                frameSumYsq += (uint64_t)yv * (uint64_t)yv;
            }
        }
        const uint64_t npxChroma = (uint64_t)chromaW * (uint64_t)chromaH;
        const uint64_t npxLuma   = (uint64_t)lumaW   * (uint64_t)lumaH;
        if (npxChroma == 0 || npxLuma == 0) return RGY_ERR_UNSUPPORTED;
        const double meanY = (double)frameSumY / (double)npxLuma;
        const double varY  = (double)frameSumYsq / (double)npxLuma - meanY * meanY;

        // Variance guard mirroring the GPU runtime path
        // (rgy_filter_colorfix.cpp:665-672): reject frames whose luma
        // variance lies outside the rolling-average band.
        bool skip = false;
        if (rollingVarCount > 0) {
            const double rollingAvg = rollingVarSum / rollingVarCount;
            const double upper = rollingAvg * prm->colorfix.varianceThreshold;
            const double lower = rollingAvg * 0.1 / prm->colorfix.varianceThreshold;
            if (varY > upper || varY < lower) skip = true;
        }
        if (!skip) {
            sumU += frameSumU;
            sumV += frameSumV;
            totalChromaPx += npxChroma;
            rollingVarSum += varY;
            ++rollingVarCount;
            ++analysedFrames;
        } else {
            ++skippedFrames;
        }
        return RGY_ERR_NONE;
    };

    auto drainDecoder = [&]() -> RGY_ERR {
        while (analysedFrames < wantFrames && seenFrames < seenCap) {
            int rv = avcodec_receive_frame(codecCtx, frameGuard.get());
            if (rv == AVERROR(EAGAIN) || rv == AVERROR_EOF) return RGY_ERR_NONE;
            if (rv < 0) return RGY_ERR_UNKNOWN;
            ++seenFrames;
            auto procErr = processFrame(frameGuard.get());
            av_frame_unref(frameGuard.get());
            if (procErr != RGY_ERR_NONE) return procErr;
        }
        return RGY_ERR_NONE;
    };

    while (analysedFrames < wantFrames && seenFrames < seenCap) {
        int rd = av_read_frame(fmtCtx, pktGuard.get());
        if (rd == AVERROR_EOF) break;
        if (rd < 0) {
            AddMessage(RGY_LOG_DEBUG, _T("pre-scan: av_read_frame error.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        if (pktGuard.get()->stream_index != videoIdx) {
            av_packet_unref(pktGuard.get());
            continue;
        }
        const int sendErr = avcodec_send_packet(codecCtx, pktGuard.get());
        av_packet_unref(pktGuard.get());
        if (sendErr < 0 && sendErr != AVERROR(EAGAIN)) {
            AddMessage(RGY_LOG_DEBUG, _T("pre-scan: avcodec_send_packet error.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        auto rcvErr = drainDecoder();
        if (rcvErr != RGY_ERR_NONE) return rcvErr;
    }
    // Flush trailing frames out of the decoder.
    if (analysedFrames < wantFrames && seenFrames < seenCap) {
        avcodec_send_packet(codecCtx, nullptr);
        auto rcvErr = drainDecoder();
        if (rcvErr != RGY_ERR_NONE) return rcvErr;
    }

    if (analysedFrames == 0 || totalChromaPx == 0 || srcBitDepth == 0) {
        AddMessage(RGY_LOG_DEBUG, _T("pre-scan: no usable frames decoded.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    const double srcMax   = (double)((1ULL << srcBitDepth) - 1ULL);
    const double meanU_src = (double)sumU / (double)totalChromaPx;
    const double meanV_src = (double)sumV / (double)totalChromaPx;
    const double meanU = meanU_src * targetMax / srcMax;
    const double meanV = meanV_src * targetMax / srcMax;
    const double rawOffU = -(meanU - neutralTarget) * prm->colorfix.strength;
    const double rawOffV = -(meanV - neutralTarget) * prm->colorfix.strength;
    m_offsetU        = (int)std::lround(rawOffU);
    m_offsetV        = (int)std::lround(rawOffV);
    m_analysedFrames = analysedFrames;
    m_skippedFrames  = skippedFrames;

    const float offUNorm = (float)((double)m_offsetU / targetMax);
    const float offVNorm = (float)((double)m_offsetV / targetMax);
    AddMessage(RGY_LOG_INFO,
        _T("pre-scan complete -- offsetU=%+.3f, offsetV=%+.3f ")
        _T("(analysed %d frames, skipped %d, src bit_depth=%d -> target bit_depth=%d).\n"),
        offUNorm, offVNorm, analysedFrames, skippedFrames, srcBitDepth, targetBitDepth);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterColorFix::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue_main,
    const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    (void)event;

    if (!pInputFrame || !pInputFrame->ptr[0]) {
        *pOutputFrameNum = 0;
        return RGY_ERR_NONE;
    }
    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("ppOutputFrames[0] must be set (in-place filter).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!m_colorfix.get()) {
        AddMessage(RGY_LOG_ERROR, _T("program failed to build.\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamColorFix>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    RGYFrameInfo *targetFrame = ppOutputFrames[0];

    // ---------------- Manual mode ----------------
    if (prm->colorfix.mode == VPP_COLORFIX_MODE_MANUAL) {
        m_totalSeenFrames++;

        // YUV-space manual: skip CspCrop entirely. Convert the user's
        // RGB white/black points to the equivalent YUV-plane corrections
        // and apply directly on the NV12 planes.
        if (m_effectiveSpace == VPP_COLORFIX_SPACE_YUV) {
            const int bitDepth = RGY_CSP_BIT_DEPTH[targetFrame->csp];
            const int maxVal   = (1 << bitDepth) - 1;
            const float neutral = (float)((maxVal + 1) / 2);
            float Kr = 0.f, Kg = 0.f, Kb = 0.f;
            getMatrixCoeffs(m_resolvedMatrix, Kr, Kg, Kb);

            // User's white/black are in 8-bit RGB. Convert to luma in
            // the working bit depth: Y = Kr*R + Kg*G + Kb*B (full range).
            const float scaleRgb8ToBd = (float)maxVal / 255.0f;
            const float wY = (Kr * (float)prm->colorfix.whiteR
                            + Kg * (float)prm->colorfix.whiteG
                            + Kb * (float)prm->colorfix.whiteB) * scaleRgb8ToBd;
            const float kY = (Kr * (float)prm->colorfix.blackR
                            + Kg * (float)prm->colorfix.blackG
                            + Kb * (float)prm->colorfix.blackB) * scaleRgb8ToBd;
            // Luma stretch: map [kY, wY] -> [0, maxVal].
            const float yDen = (wY - kY != 0.0f) ? (wY - kY) : 1.0f;
            const float scaleY  = (float)maxVal / yDen;
            const float offsetY = -kY * scaleY;

            // Chroma white-balance: shift U and V so that the white-point's
            // chroma maps to neutral. This is the cheap approximation per
            // the "fast" YUV-space contract.
            //   U_wp = (B - Y_wp) / (2*(1 - Kb)) + neutral
            //   V_wp = (R - Y_wp) / (2*(1 - Kr)) + neutral
            const float wB_bd = (float)prm->colorfix.whiteB * scaleRgb8ToBd;
            const float wR_bd = (float)prm->colorfix.whiteR * scaleRgb8ToBd;
            const float U_wp = (wB_bd - wY) / (2.0f * (1.0f - Kb)) + neutral;
            const float V_wp = (wR_bd - wY) / (2.0f * (1.0f - Kr)) + neutral;
            const int offsetU = (int)std::lround(neutral - U_wp);
            const int offsetV = (int)std::lround(neutral - V_wp);

            if (m_totalSeenFrames == 1) {
                AddMessage(RGY_LOG_INFO,
                    _T("manual debug (yuv-space) -- white=(%d,%d,%d) black=(%d,%d,%d)\n")
                    _T("                          Y wp/bp = (%.2f, %.2f)  scaleY=%.6f offsetY=%+.3f\n")
                    _T("                          chroma white-balance offset_U=%d, offset_V=%d\n"),
                    prm->colorfix.whiteR, prm->colorfix.whiteG, prm->colorfix.whiteB,
                    prm->colorfix.blackR, prm->colorfix.blackG, prm->colorfix.blackB,
                    wY, kY, scaleY, offsetY,
                    offsetU, offsetV);
            }

            // Luma stretch on the Y plane in-place.
            auto err = runApplyLuma(targetFrame, scaleY, offsetY, queue_main, wait_events);
            if (err != RGY_ERR_NONE) return err;
            // Chroma offset on U/V planes in-place.
            err = runApplyUV(targetFrame, offsetU, offsetV, queue_main, {});
            if (err != RGY_ERR_NONE) return err;
            return RGY_ERR_NONE;
        }

        // RGB-space manual: original path via CspCrop sub-filters.
        int convOutNum = 0;
        RGYFrameInfo *convOut[1] = { nullptr };
        RGYFrameInfo inFrame = *targetFrame;
        auto err = m_convToRgb->filter(&inFrame, (RGYFrameInfo **)&convOut, &convOutNum, queue_main, wait_events, nullptr);
        if (err != RGY_ERR_NONE || convOut[0] == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("YUV->RGB conversion failed: %s.\n"), get_err_mes(err));
            return err;
        }
        RGYFrameInfo *pRgb = convOut[0];

        const int rgbMax = (RGY_CSP_BIT_DEPTH[m_cspRgb] > 8) ? 65535 : 255;
        const int wR = prm->colorfix.whiteR * rgbMax / 255;
        const int wG = prm->colorfix.whiteG * rgbMax / 255;
        const int wB = prm->colorfix.whiteB * rgbMax / 255;
        const int kR = prm->colorfix.blackR * rgbMax / 255;
        const int kG = prm->colorfix.blackG * rgbMax / 255;
        const int kB = prm->colorfix.blackB * rgbMax / 255;
        const float scaleR  = (float)rgbMax / (float)(wR - kR);
        const float scaleG  = (float)rgbMax / (float)(wG - kG);
        const float scaleB  = (float)rgbMax / (float)(wB - kB);
        const float offsetR = -((float)kR) * scaleR;
        const float offsetG = -((float)kG) * scaleG;
        const float offsetB = -((float)kB) * scaleB;

        // First-frame diagnostic: dump everything between the raw parser
        // output (whiteR/G/B + blackR/G/B) and the floats handed to the
        // OpenCL kernel. Helps spot any divergence between hex- and tuple-
        // form parsing paths.
        if (m_totalSeenFrames == 1) {
            AddMessage(RGY_LOG_INFO,
                _T("manual debug -- white=(%d,%d,%d) black=(%d,%d,%d) rgbMax=%d\n")
                _T("                          wHBD=(%d,%d,%d) kHBD=(%d,%d,%d)\n")
                _T("                          scale=(%.6f, %.6f, %.6f)\n")
                _T("                          offset=(%+.3f, %+.3f, %+.3f)\n"),
                prm->colorfix.whiteR, prm->colorfix.whiteG, prm->colorfix.whiteB,
                prm->colorfix.blackR, prm->colorfix.blackG, prm->colorfix.blackB,
                rgbMax,
                wR, wG, wB, kR, kG, kB,
                scaleR, scaleG, scaleB,
                offsetR, offsetG, offsetB);
        }

        err = runApplyRGB(pRgb, scaleR, scaleG, scaleB, offsetR, offsetG, offsetB, queue_main, {});
        if (err != RGY_ERR_NONE) return err;

        // RGB -> YUV back into ppOutputFrames[0]
        convOutNum = 0;
        RGYFrameInfo *convOut2[1] = { (m_convToYuv->GetFilterParam()->frameOut.csp == targetFrame->csp) ? targetFrame : nullptr };
        RGYFrameInfo inFrameRgb = *pRgb;
        err = m_convToYuv->filter(&inFrameRgb, (RGYFrameInfo **)&convOut2, &convOutNum, queue_main, {}, nullptr);
        if (err != RGY_ERR_NONE || convOut2[0] == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("RGB->YUV conversion failed: %s.\n"), get_err_mes(err));
            return err;
        }
        return RGY_ERR_NONE;
    }

    // ---------------- Auto mode ----------------
    if (prm->colorfix.mode == VPP_COLORFIX_MODE_AUTO) {
        m_totalSeenFrames++;
        if (!m_analysisComplete) {
            auto err = runReduceUV(targetFrame, queue_main, wait_events);
            if (err != RGY_ERR_NONE) return err;

            std::vector<long long> totals;
            err = finaliseReduction(queue_main, 4, totals);
            if (err != RGY_ERR_NONE) return err;

            const long long sumU = totals[0], sumV = totals[1];
            const long long sumY = totals[2], sumYsq = totals[3];

            const auto planeU = getPlane(targetFrame, RGY_PLANE_U);
            const auto planeY = getPlane(targetFrame, RGY_PLANE_Y);
            const long long npxChroma = (long long)planeU.width * planeU.height;
            const long long npxLuma   = (long long)planeY.width * planeY.height;
            const double meanY  = (double)sumY / (double)npxLuma;
            const double varY   = (double)sumYsq / (double)npxLuma - meanY * meanY;

            // Variance guard: skip if outside rollingAvg × threshold.
            bool skip = false;
            if (m_rollingVarianceCount > 0) {
                const double rollingAvg = m_rollingVarianceSum / m_rollingVarianceCount;
                const double upper = rollingAvg * prm->colorfix.varianceThreshold;
                const double lower = rollingAvg * 0.1 / prm->colorfix.varianceThreshold;
                if (varY > upper || varY < lower) skip = true;
            }

            if (!skip) {
                m_sumA += sumU;
                m_sumB += sumV;
                m_sumY += sumY;
                m_sumYsq += sumYsq;
                m_rollingVarianceSum += varY;
                m_rollingVarianceCount++;
                m_analysedFrames++;
                // Track number of chroma pixels accumulated (overload m_sumC for it)
                m_sumC += npxChroma;
            } else {
                m_skippedFrames++;
            }

            // Compute running offsets from the accumulated sums. These
            // are applied at strength_factor < 1.0 during the ramp; the
            // final values become the locked-in offsets at lock-in.
            int runningOffU = 0, runningOffV = 0;
            const int bitDepth = RGY_CSP_BIT_DEPTH[targetFrame->csp];
            const float maxVal = (float)((1 << bitDepth) - 1);
            if (m_analysedFrames > 0 && m_sumC > 0) {
                const double meanU   = (double)m_sumA / (double)m_sumC;
                const double meanV   = (double)m_sumB / (double)m_sumC;
                const double neutral = (double)(1 << (bitDepth - 1));
                runningOffU = (int)std::lround(-(meanU - neutral) * prm->colorfix.strength);
                runningOffV = (int)std::lround(-(meanV - neutral) * prm->colorfix.strength);
            }

            // Lock-in decisions (mirrors chromashift safety net at
            // rgy_filter_chromashift.cpp:379):
            //   - Reached `frames=` accepted frames -> normal lock-in.
            //   - Hit hard cap (3*frames+10 input frames) with at least
            //     one accepted -> emit WARN and freeze running offset.
            //     Prevents the rest of the clip running uncorrected when
            //     a pathological intro keeps tripping the variance guard.
            bool lockNow = false;
            if (m_analysedFrames >= prm->colorfix.frames) {
                lockNow = true;
            } else if (m_totalSeenFrames >= m_hardCapFrames && m_analysedFrames > 0) {
                AddMessage(RGY_LOG_WARN,
                    _T("variance guard rejected too many frames after %d input ")
                    _T("(only %d accepted of %d target). Locking in early offsets.\n"),
                    m_totalSeenFrames, m_analysedFrames, prm->colorfix.frames);
                lockNow = true;
            }
            if (lockNow) {
                m_offsetU = runningOffU;
                m_offsetV = runningOffV;
                m_analysisComplete = true;
                const float offUNorm = m_offsetU / maxVal;
                const float offVNorm = m_offsetV / maxVal;
                AddMessage(RGY_LOG_INFO,
                    _T("auto analysis complete -- offsetU=%+.3f, offsetV=%+.3f ")
                    _T("(skipped %d flash frames)\n"),
                    offUNorm, offVNorm, m_skippedFrames);
            }

            // Ramp: apply correction at reduced strength during analysis
            // so the transition to full-strength at lock-in is smooth.
            // strength_factor = accepted_frames / target_frames; clamped
            // to 1.0 (reached at lock-in). Mathematically, frame N gets
            // the running offset scaled to (N/frames); by the time
            // lock-in fires the factor is 1.0 and the offset is the
            // locked value, so there is no discontinuity at frame N+1.
            const float strengthFactor = std::min(
                (float)m_analysedFrames / (float)prm->colorfix.frames, 1.0f);
            const int applyU = (int)std::lround((double)runningOffU * (double)strengthFactor);
            const int applyV = (int)std::lround((double)runningOffV * (double)strengthFactor);
            return runApplyUV(targetFrame, applyU, applyV, queue_main, {});
        }

        // Post-analysis: apply chroma offsets.
        return runApplyUV(targetFrame, m_offsetU, m_offsetV, queue_main, wait_events);
    }

    // ---------------- Gray mode ----------------
    // (mode == VPP_COLORFIX_MODE_GRAY)
    m_totalSeenFrames++;

    // YUV-space gray: the grayworld assumption ("mean R = mean G = mean B")
    // is fundamentally an RGB-domain assumption. In YUV it reduces to
    // "mean U = mean V = neutral" -- exactly the auto-mode algorithm.
    // Route gray-yuv through the auto-mode code path: reduce chroma sums,
    // compute offsets after `frames` valid frames, then apply on-plane.
    if (m_effectiveSpace == VPP_COLORFIX_SPACE_YUV) {
        if (!m_analysisComplete) {
            auto err = runReduceUV(targetFrame, queue_main, wait_events);
            if (err != RGY_ERR_NONE) return err;

            std::vector<long long> totals;
            err = finaliseReduction(queue_main, 4, totals);
            if (err != RGY_ERR_NONE) return err;

            const long long sumU = totals[0], sumV = totals[1];
            const long long sumY = totals[2], sumYsq = totals[3];

            const auto planeU = getPlane(targetFrame, RGY_PLANE_U);
            const auto planeY = getPlane(targetFrame, RGY_PLANE_Y);
            const long long npxChroma = (long long)planeU.width * planeU.height;
            const long long npxLuma   = (long long)planeY.width * planeY.height;
            const double meanY = (double)sumY / (double)npxLuma;
            const double varY  = (double)sumYsq / (double)npxLuma - meanY * meanY;

            bool skip = false;
            if (m_rollingVarianceCount > 0) {
                const double rollingAvg = m_rollingVarianceSum / m_rollingVarianceCount;
                const double upper = rollingAvg * prm->colorfix.varianceThreshold;
                const double lower = rollingAvg * 0.1 / prm->colorfix.varianceThreshold;
                if (varY > upper || varY < lower) skip = true;
            }
            if (!skip) {
                m_sumA += sumU;
                m_sumB += sumV;
                m_sumY += sumY;
                m_sumYsq += sumYsq;
                m_rollingVarianceSum += varY;
                m_rollingVarianceCount++;
                m_analysedFrames++;
                m_sumC += npxChroma;
            } else {
                m_skippedFrames++;
            }

            // Running offsets for the ramp (same shape as the auto-mode
            // branch above; see that comment block for the strength_factor
            // derivation and the safety-timeout rationale).
            int runningOffU = 0, runningOffV = 0;
            const int bitDepth = RGY_CSP_BIT_DEPTH[targetFrame->csp];
            const float maxVal = (float)((1 << bitDepth) - 1);
            if (m_analysedFrames > 0 && m_sumC > 0) {
                const double meanU   = (double)m_sumA / (double)m_sumC;
                const double meanV   = (double)m_sumB / (double)m_sumC;
                const double neutral = (double)(1 << (bitDepth - 1));
                runningOffU = (int)std::lround(-(meanU - neutral) * prm->colorfix.strength);
                runningOffV = (int)std::lround(-(meanV - neutral) * prm->colorfix.strength);
            }

            bool lockNow = false;
            if (m_analysedFrames >= prm->colorfix.frames) {
                lockNow = true;
            } else if (m_totalSeenFrames >= m_hardCapFrames && m_analysedFrames > 0) {
                AddMessage(RGY_LOG_WARN,
                    _T("variance guard rejected too many frames after %d input ")
                    _T("(only %d accepted of %d target). Locking in early offsets.\n"),
                    m_totalSeenFrames, m_analysedFrames, prm->colorfix.frames);
                lockNow = true;
            }
            if (lockNow) {
                m_offsetU = runningOffU;
                m_offsetV = runningOffV;
                m_analysisComplete = true;
                const float offUNorm = m_offsetU / maxVal;
                const float offVNorm = m_offsetV / maxVal;
                AddMessage(RGY_LOG_INFO,
                    _T("gray (yuv-space) analysis complete -- offsetU=%+.3f, offsetV=%+.3f ")
                    _T("(skipped %d flash frames)\n"),
                    offUNorm, offVNorm, m_skippedFrames);
            }

            const float strengthFactor = std::min(
                (float)m_analysedFrames / (float)prm->colorfix.frames, 1.0f);
            const int applyU = (int)std::lround((double)runningOffU * (double)strengthFactor);
            const int applyV = (int)std::lround((double)runningOffV * (double)strengthFactor);
            return runApplyUV(targetFrame, applyU, applyV, queue_main, {});
        }
        return runApplyUV(targetFrame, m_offsetU, m_offsetV, queue_main, wait_events);
    }

    // RGB-space gray: original path via CspCrop sub-filters.
    {
        // Convert source YUV -> RGB intermediate.
        int convOutNum = 0;
        RGYFrameInfo *convOut[1] = { nullptr };
        RGYFrameInfo inFrame = *targetFrame;
        auto err = m_convToRgb->filter(&inFrame, (RGYFrameInfo **)&convOut, &convOutNum, queue_main, wait_events, nullptr);
        if (err != RGY_ERR_NONE || convOut[0] == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("YUV->RGB conversion failed: %s.\n"), get_err_mes(err));
            return err;
        }
        RGYFrameInfo *pRgb = convOut[0];

        if (!m_analysisComplete) {
            err = runReduceRGB(pRgb, queue_main, {});
            if (err != RGY_ERR_NONE) return err;

            std::vector<long long> totals;
            err = finaliseReduction(queue_main, 5, totals);
            if (err != RGY_ERR_NONE) return err;

            const long long sumR = totals[0], sumG = totals[1], sumB_ = totals[2];
            const long long sumY = totals[3], sumYsq = totals[4];
            const long long npx  = (long long)pRgb->width * pRgb->height;
            const double meanY = (double)sumY / (double)npx;
            const double varY  = (double)sumYsq / (double)npx - meanY * meanY;

            bool skip = false;
            if (m_rollingVarianceCount > 0) {
                const double rollingAvg = m_rollingVarianceSum / m_rollingVarianceCount;
                const double upper = rollingAvg * prm->colorfix.varianceThreshold;
                const double lower = rollingAvg * 0.1 / prm->colorfix.varianceThreshold;
                if (varY > upper || varY < lower) skip = true;
            }

            if (!skip) {
                m_sumA += sumR;
                m_sumB += sumG;
                m_sumC += sumB_;
                m_sumY += sumY;
                m_sumYsq += sumYsq;
                m_rollingVarianceSum += varY;
                m_rollingVarianceCount++;
                m_analysedFrames++;
                // m_totalSeenFrames already tracks pixel count via the analysed count
            } else {
                m_skippedFrames++;
            }

            // Lock-in or safety timeout (3*frames+10). The RGB-space gray
            // path keeps the passthrough-during-analysis behaviour (the
            // round-trip RGB->YUV at end of branch leaves output identical
            // to input); the safety net just lets a clip with pathological
            // intro frames still reach lock-in before the file ends.
            bool lockNow = false;
            if (m_analysedFrames >= prm->colorfix.frames) {
                lockNow = true;
            } else if (m_totalSeenFrames >= m_hardCapFrames && m_analysedFrames > 0) {
                AddMessage(RGY_LOG_WARN,
                    _T("variance guard rejected too many frames after %d input ")
                    _T("(only %d accepted of %d target). Locking in early scales.\n"),
                    m_totalSeenFrames, m_analysedFrames, prm->colorfix.frames);
                lockNow = true;
            }
            if (lockNow) {
                const long long npxTotal = (long long)pRgb->width * pRgb->height * m_analysedFrames;
                const double meanR = (double)m_sumA / (double)npxTotal;
                const double meanG = (double)m_sumB / (double)npxTotal;
                const double meanB = (double)m_sumC / (double)npxTotal;
                const double meanAll = (meanR + meanG + meanB) / 3.0;
                const float strength = prm->colorfix.strength;

                auto safeScale = [&](double mean) -> float {
                    if (mean < 1.0) return 1.0f;     // near-black guard
                    return (float)((meanAll / mean) * strength + (1.0 - strength));
                };
                m_scaleR = safeScale(meanR);
                m_scaleG = safeScale(meanG);
                m_scaleB = safeScale(meanB);
                m_analysisComplete = true;
                AddMessage(RGY_LOG_INFO,
                    _T("gray analysis complete -- scaleR=%.3f, scaleG=%.3f, scaleB=%.3f ")
                    _T("(skipped %d flash frames)\n"),
                    m_scaleR, m_scaleG, m_scaleB, m_skippedFrames);
            }
            // Passthrough during analysis: just round-trip RGB->YUV.
            int yuvOutNum = 0;
            RGYFrameInfo *yuvOut[1] = { (m_convToYuv->GetFilterParam()->frameOut.csp == targetFrame->csp) ? targetFrame : nullptr };
            RGYFrameInfo inFrameRgb = *pRgb;
            err = m_convToYuv->filter(&inFrameRgb, (RGYFrameInfo **)&yuvOut, &yuvOutNum, queue_main, {}, nullptr);
            if (err != RGY_ERR_NONE || yuvOut[0] == nullptr) {
                AddMessage(RGY_LOG_ERROR, _T("RGB->YUV conversion failed (passthrough): %s.\n"), get_err_mes(err));
                return err;
            }
            return RGY_ERR_NONE;
        }

        // Apply gray-world rescale.
        err = runApplyRGB(pRgb, m_scaleR, m_scaleG, m_scaleB, 0.0f, 0.0f, 0.0f, queue_main, {});
        if (err != RGY_ERR_NONE) return err;

        // RGB -> YUV back.
        int yuvOutNum = 0;
        RGYFrameInfo *yuvOut[1] = { (m_convToYuv->GetFilterParam()->frameOut.csp == targetFrame->csp) ? targetFrame : nullptr };
        RGYFrameInfo inFrameRgb = *pRgb;
        err = m_convToYuv->filter(&inFrameRgb, (RGYFrameInfo **)&yuvOut, &yuvOutNum, queue_main, {}, nullptr);
        if (err != RGY_ERR_NONE || yuvOut[0] == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("RGB->YUV conversion failed: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

void RGYFilterColorFix::close() {
    m_colorfix.clear();
    m_buildOptionsYUV.clear();
    m_buildOptionsRGB.clear();
    m_convToRgb.reset();
    m_convToYuv.reset();
    m_reducePartials.reset();
    m_frameBuf.clear();
    m_cl.reset();
}
