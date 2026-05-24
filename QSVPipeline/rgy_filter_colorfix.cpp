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
//   Algorithm also described in FFmpeg grayworld filter (LGPL).
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <vector>
#include "convert_csp.h"
#include "rgy_filter_colorfix.h"

static const int COLORFIX_BLOCK_X = 16;
static const int COLORFIX_BLOCK_Y = 16;
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
    m_scaleR(1.0f), m_scaleG(1.0f), m_scaleB(1.0f) {
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
            _T("colorfix: setupCspConverters skipped (effective space=yuv).\n"));
        return RGY_ERR_NONE;
    }

    AddMessage(RGY_LOG_DEBUG,
        _T("colorfix: setupCspConverters building YUV<->RGB sub-filters (effective space=rgb).\n"));

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
        AddMessage(RGY_LOG_WARN, _T("colorfix: mode=auto only supports space=yuv; ignoring space=rgb.\n"));
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
            (cl_mem)pR.ptr[0], pR.pitch[0],
            (cl_mem)pG.ptr[0], pG.pitch[0],
            (cl_mem)pB.ptr[0], pB.pitch[0],
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
    const int uvInterleaved = (pU.ptr[0] == pV.ptr[0]) ? 1 : 0;
    const int chromaWidth = uvInterleaved ? std::max(1, pU.width / 2) : pU.width;
    const int chromaHeight = pU.height;
    if (!uvInterleaved && (pU.width != pV.width || pU.height != pV.height || pU.pitch[0] != pV.pitch[0])) {
        AddMessage(RGY_LOG_ERROR, _T("colorfix: U/V plane layout mismatch.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    const int subX = std::max(1, pY.width  / chromaWidth);
    const int subY = std::max(1, pY.height / pU.height);

    RGYWorkSize local(COLORFIX_BLOCK_X, COLORFIX_BLOCK_Y);
    RGYWorkSize global(chromaWidth, chromaHeight);
    const int wgX = ((chromaWidth  + COLORFIX_BLOCK_X - 1) / COLORFIX_BLOCK_X);
    const int wgY = ((chromaHeight + COLORFIX_BLOCK_Y - 1) / COLORFIX_BLOCK_Y);
    m_numGroupsLastDispatch = wgX * wgY;

    auto err = m_colorfix.get()->kernel("colorfix_reduce_uv")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)pY.ptr[0], pY.pitch[0], pY.width, pY.height,
            (cl_mem)pU.ptr[0], pU.pitch[0], chromaWidth, chromaHeight,
            (cl_mem)pV.ptr[0], pV.pitch[0],
            uvInterleaved,
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
            (cl_mem)pR.ptr[0], pR.pitch[0],
            (cl_mem)pG.ptr[0], pG.pitch[0],
            (cl_mem)pB.ptr[0], pB.pitch[0],
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
    const int uvInterleaved = (pU.ptr[0] == pV.ptr[0]) ? 1 : 0;
    const int chromaWidth = uvInterleaved ? std::max(1, pU.width / 2) : pU.width;
    const int chromaHeight = pU.height;
    if (!uvInterleaved && (pU.width != pV.width || pU.height != pV.height || pU.pitch[0] != pV.pitch[0])) {
        AddMessage(RGY_LOG_ERROR, _T("colorfix: U/V plane layout mismatch.\n"));
        return RGY_ERR_INVALID_CALL;
    }

    RGYWorkSize local(COLORFIX_BLOCK_X, COLORFIX_BLOCK_Y);
    RGYWorkSize global(chromaWidth, chromaHeight);
    auto err = m_colorfix.get()->kernel("colorfix_apply_uv")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)pU.ptr[0], pU.pitch[0],
            (cl_mem)pV.ptr[0], pV.pitch[0],
            chromaWidth, chromaHeight,
            uvInterleaved,
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
        AddMessage(RGY_LOG_ERROR, _T("colorfix readback failed: %s.\n"), cl_errmes(clerr));
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
        AddMessage(RGY_LOG_ERROR, _T("colorfix program failed to build.\n"));
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
                    _T("colorfix: manual debug (yuv-space) -- white=(%d,%d,%d) black=(%d,%d,%d)\n")
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
                _T("colorfix: manual debug -- white=(%d,%d,%d) black=(%d,%d,%d) rgbMax=%d\n")
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
            const auto planeV = getPlane(targetFrame, RGY_PLANE_V);
            const long long npxChroma = (long long)((planeU.ptr[0] == planeV.ptr[0]) ? std::max(1, planeU.width / 2) : planeU.width) * planeU.height;
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

            if (m_analysedFrames >= prm->colorfix.frames) {
                const double meanU = (double)m_sumA / (double)m_sumC;
                const double meanV = (double)m_sumB / (double)m_sumC;
                const int    bitDepth = RGY_CSP_BIT_DEPTH[targetFrame->csp];
                const float  maxVal  = (float)((1 << bitDepth) - 1);
                const double neutral = (double)(1 << (bitDepth - 1));  // 128 in 8-bit, etc.
                const double rawOffU = -(meanU - neutral) * prm->colorfix.strength;
                const double rawOffV = -(meanV - neutral) * prm->colorfix.strength;
                m_offsetU = (int)std::lround(rawOffU);
                m_offsetV = (int)std::lround(rawOffV);
                m_analysisComplete = true;
                const float offUNorm = m_offsetU / maxVal;
                const float offVNorm = m_offsetV / maxVal;
                AddMessage(RGY_LOG_INFO,
                    _T("colorfix: auto analysis complete -- offsetU=%+.3f, offsetV=%+.3f ")
                    _T("(skipped %d flash frames)\n"),
                    offUNorm, offVNorm, m_skippedFrames);
            }
            // Passthrough during analysis: src and dst are the same buffer
            // (in-place filter), so nothing to do.
            return RGY_ERR_NONE;
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
            const auto planeV = getPlane(targetFrame, RGY_PLANE_V);
            const long long npxChroma = (long long)((planeU.ptr[0] == planeV.ptr[0]) ? std::max(1, planeU.width / 2) : planeU.width) * planeU.height;
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

            if (m_analysedFrames >= prm->colorfix.frames) {
                const double meanU = (double)m_sumA / (double)m_sumC;
                const double meanV = (double)m_sumB / (double)m_sumC;
                const int    bitDepth = RGY_CSP_BIT_DEPTH[targetFrame->csp];
                const float  maxVal  = (float)((1 << bitDepth) - 1);
                const double neutral = (double)(1 << (bitDepth - 1));
                const double rawOffU = -(meanU - neutral) * prm->colorfix.strength;
                const double rawOffV = -(meanV - neutral) * prm->colorfix.strength;
                m_offsetU = (int)std::lround(rawOffU);
                m_offsetV = (int)std::lround(rawOffV);
                m_analysisComplete = true;
                const float offUNorm = m_offsetU / maxVal;
                const float offVNorm = m_offsetV / maxVal;
                AddMessage(RGY_LOG_INFO,
                    _T("colorfix: gray (yuv-space) analysis complete -- offsetU=%+.3f, offsetV=%+.3f ")
                    _T("(skipped %d flash frames)\n"),
                    offUNorm, offVNorm, m_skippedFrames);
            }
            return RGY_ERR_NONE;
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

            if (m_analysedFrames >= prm->colorfix.frames) {
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
                    _T("colorfix: gray analysis complete -- scaleR=%.3f, scaleG=%.3f, scaleB=%.3f ")
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
