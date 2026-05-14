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

#include <algorithm>
#include <cmath>
#include "convert_csp.h"
#include "rgy_filter_maa.h"
#include "rgy_filter_resize.h"

static const int MAA_BLOCK_X = 32;
static const int MAA_BLOCK_Y = 8;

// Round to the nearest multiple of 4. Matches the reference MAA2 script's
// `int(round(c.width * ss / 4.0) * 4)` rounding. Kept aligned to 4 so
// SangNom2's 2-row processing (and the rotation pipeline's stride math)
// stays well-defined for both luma and 4:2:0 chroma planes.
static int alignSsDim(int srcDim, float ss) {
    const float scaled = (float)srcDim * ss / 4.0f;
    return (int)std::lround(scaled) * 4;
}

RGYFilterMaa::RGYFilterMaa(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_maa(),
    m_maaBuildOptions(),
    m_resizeUp(),
    m_resizeDown(),
    m_resizeUpLuma(),
    m_resizeDownLuma(),
    m_supersampled(),
    m_rotated(),
    m_rotatedAA(),
    m_unrotatedAA(),
    m_aaResult(),
    m_edgeMask(),
    m_inflatedMask(),
    m_chromaMask(),
    m_costRawPacked(),
    m_costSmoothPacked(),
    m_costPitch(0),
    m_costSliceBytes(0),
    m_costElemBytes(1),
    m_ssW(0),
    m_ssH(0),
    m_aaf(0.0f),
    m_aacf(0.0f),
    m_mthreshScaled(0) {
    m_name = _T("maa");
}

RGYFilterMaa::~RGYFilterMaa() {
    close();
}

RGY_ERR RGYFilterMaa::checkParam(const std::shared_ptr<RGYFilterParamMaa> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height < 4 || prm->frameOut.width < 4) {
        AddMessage(RGY_LOG_ERROR, _T("MAA requires input width/height >= 4 (got %dx%d).\n"),
            prm->frameOut.width, prm->frameOut.height);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->maa.ss < 1.0f || prm->maa.ss > 4.0f) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid ss=%.3f: must be in [1.0, 4.0].\n"), prm->maa.ss);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->maa.aa < 0 || prm->maa.aa > 255) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid aa=%d: must be in [0, 255].\n"), prm->maa.aa);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->maa.aac < 0 || prm->maa.aac > 255) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid aac=%d: must be in [0, 255].\n"), prm->maa.aac);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->maa.mthresh < 1 || prm->maa.mthresh > 255) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid mthresh=%d: must be in [1, 255].\n"), prm->maa.mthresh);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->maa.show < 0 || prm->maa.show > 2) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid show=%d: must be in [0, 2].\n"), prm->maa.show);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterMaa::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamMaa>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    // MAA is single-frame: 1 output per input, frame-rate preserved, no
    // picstruct/timestamp ownership change. Leave m_pathThrough as-is so
    // upstream metadata flows through unmodified.
    prm->frameOut.picstruct = prm->frameIn.picstruct;

    // Compute supersampled dimensions, rounded to multiple of 4 to match
    // the reference MAA2 script's behaviour
    // (analysis/maa2_investigation/06_formulas_and_constants.md § 5).
    m_ssW = alignSsDim(prm->frameIn.width,  prm->maa.ss);
    m_ssH = alignSsDim(prm->frameIn.height, prm->maa.ss);
    if (m_ssW < 4) m_ssW = 4;
    if (m_ssH < 4) m_ssH = 4;

    // Bit-depth-scaled thresholds. peak = (1 << bitDepth) - 1 — same form
    // used by every other QSVEnc filter that consumes 0..255-style user
    // params (see e.g. rgy_filter_bwdif.cpp:188).
    const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    const int peak     = (1 << bitDepth) - 1;
    m_aaf           = (float)prm->maa.aa  * (float)peak / 256.0f;
    m_aacf          = (float)prm->maa.aac * (float)peak / 256.0f;
    m_mthreshScaled = prm->maa.mthresh * peak / 255;

    // (Re)build the OpenCL program when bit-depth changes. The .cl is
    // currently empty; this wiring is in place so subsequent prompts can
    // add kernels without touching init().
    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamMaa>(m_param);
    if (!prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        const int maxVal = peak;
        m_maaBuildOptions = strsprintf("-D Type=%s -D bit_depth=%d -D max_val=%d -D maa_block_x=%d -D maa_block_y=%d",
            bitDepth > 8 ? "ushort" : "uchar",
            bitDepth, maxVal,
            MAA_BLOCK_X, MAA_BLOCK_Y);
        AddMessage(RGY_LOG_DEBUG, _T("Starting async build for RGY_FILTER_MAA_CL: %s\n"),
            char_to_tstring(m_maaBuildOptions).c_str());
        m_maa.set(m_cl->buildResourceAsync(_T("RGY_FILTER_MAA_CL"), _T("EXE_DATA"), m_maaBuildOptions.c_str()));
    }

    // Output buffer at source resolution. Single buffer — MAA is a 1-in-1-out
    // single-frame filter.
    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output buffer: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // Spline36 resize sub-filters. The MAA pipeline does:
    //   src (W × H) → m_resizeUp → supersampled (ssW × ssH)
    //   ... AA pipeline runs at supersampled resolution ...
    //   AA result (ssW × ssH) → m_resizeDown → output (W × H)
    // Both sub-filters are owned by this filter and torn down in close().
    // chroma=true 時はフル CSP の resize、chroma=false 時は Y8/Y16 CSP の
    // luma-only resize を構築する。chroma=false 時に chroma plane を resize
    // しても結果は最終段で input.chroma の copyPlane に上書きされるため、
    // フル CSP の resize は使わず chroma の処理コスト (≈1/2 of luma cost)
    // を節約する。
    if (prm->maa.chroma) {
        {
            auto prmUp = std::make_shared<RGYFilterParamResize>();
            prmUp->frameIn         = prm->frameIn;
            prmUp->frameOut        = prm->frameIn;          // copy CSP/picstruct/etc, then override w/h
            prmUp->frameOut.width  = m_ssW;
            prmUp->frameOut.height = m_ssH;
            prmUp->interp          = RGY_VPP_RESIZE_SPLINE36;
            prmUp->baseFps         = prm->baseFps;
            prmUp->bOutOverwrite   = false;
            m_resizeUp = std::make_unique<RGYFilterResize>(m_cl);
            sts = m_resizeUp->init(prmUp, m_pLog);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to init upscale sub-filter: %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
        {
            auto prmDn = std::make_shared<RGYFilterParamResize>();
            prmDn->frameIn         = prm->frameIn;          // copy CSP/picstruct/etc
            prmDn->frameIn.width   = m_ssW;
            prmDn->frameIn.height  = m_ssH;
            prmDn->frameOut        = prm->frameOut;
            prmDn->interp          = RGY_VPP_RESIZE_SPLINE36;
            prmDn->baseFps         = prm->baseFps;
            prmDn->bOutOverwrite   = false;
            m_resizeDown = std::make_unique<RGYFilterResize>(m_cl);
            sts = m_resizeDown->init(prmDn, m_pLog);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to init downscale sub-filter: %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
    } else {
        const RGY_CSP lumaCsp = (bitDepth > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;
        {
            auto prmUp = std::make_shared<RGYFilterParamResize>();
            prmUp->frameIn         = prm->frameIn;
            prmUp->frameIn.csp     = lumaCsp;
            prmUp->frameOut        = prmUp->frameIn;
            prmUp->frameOut.width  = m_ssW;
            prmUp->frameOut.height = m_ssH;
            prmUp->interp          = RGY_VPP_RESIZE_SPLINE36;
            prmUp->baseFps         = prm->baseFps;
            prmUp->bOutOverwrite   = false;
            m_resizeUpLuma = std::make_unique<RGYFilterResize>(m_cl);
            sts = m_resizeUpLuma->init(prmUp, m_pLog);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to init luma upscale sub-filter: %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
        {
            auto prmDn = std::make_shared<RGYFilterParamResize>();
            prmDn->frameIn         = prm->frameIn;
            prmDn->frameIn.csp     = lumaCsp;
            prmDn->frameIn.width   = m_ssW;
            prmDn->frameIn.height  = m_ssH;
            prmDn->frameOut        = prm->frameOut;
            prmDn->frameOut.csp    = lumaCsp;
            prmDn->interp          = RGY_VPP_RESIZE_SPLINE36;
            prmDn->baseFps         = prm->baseFps;
            prmDn->bOutOverwrite   = false;
            m_resizeDownLuma = std::make_unique<RGYFilterResize>(m_cl);
            sts = m_resizeDownLuma->init(prmDn, m_pLog);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to init luma downscale sub-filter: %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
    }

    // Allocate intermediate frame buffers. Subsequent prompts will populate
    // them via the AA kernels; for now they exist so the wiring is testable.
    {
        RGYFrameInfo ssInfo = prm->frameIn;
        ssInfo.width  = m_ssW;
        ssInfo.height = m_ssH;
        m_supersampled = m_cl->createFrameBuffer(ssInfo);
        m_aaResult     = m_cl->createFrameBuffer(ssInfo);
        m_unrotatedAA  = m_cl->createFrameBuffer(ssInfo);
        if (!m_supersampled || !m_aaResult || !m_unrotatedAA) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate supersampled intermediate buffers.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    {
        RGYFrameInfo rotInfo = prm->frameIn;
        rotInfo.width  = m_ssH;     // rotated: dimensions swap
        rotInfo.height = m_ssW;
        m_rotated   = m_cl->createFrameBuffer(rotInfo);
        m_rotatedAA = m_cl->createFrameBuffer(rotInfo);
        if (!m_rotated || !m_rotatedAA) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate rotated intermediate buffers.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    {
        m_edgeMask = m_cl->createFrameBuffer(prm->frameIn);
        m_inflatedMask = m_cl->createFrameBuffer(prm->frameIn);
        if (!m_edgeMask || !m_inflatedMask) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate edge-mask buffers.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    // Chroma-resolution mask buffer only needed when chroma=true AND chroma
    // is sub-sampled relative to luma (YV12 / YV16 / NV12). For YV24 the
    // luma mask dimensions already match chroma dimensions and the host
    // reuses m_inflatedMask. We always allocate the buffer when chroma=true
    // so the host code path is uniform.
    if (prm->maa.chroma) {
        m_chromaMask = m_cl->createFrameBuffer(prm->frameIn);
        if (!m_chromaMask) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate chroma-mask buffer.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    // SangNom2 cost buffers: 9 cost slices + 9 smoothed slices, each sized
    // to the LARGER of the two pass shapes. Pass 1 runs on the rotated frame
    // whose luma plane is (ssH × ssW); pass 2 runs on the unrotated luma
    // plane (ssW × ssH). Slice geometry: bufW × bufH =
    //   max(ssH, ssW) × max(ssW, ssH)/2 (covers either shape).
    //
    // PACKED ALLOCATION: rather than 9 separate `RGYCLBuf`s, the 9 slices
    // are stored as contiguous sub-regions of a single `RGYCLBuf`. Slice
    // `i` is at byte offset `i * m_costSliceBytes`. The prepare and
    // finalize kernels take a single packed buffer pointer and compute
    // their own slice base offsets. Smooth still dispatches 9 times (one
    // per slice) but reads/writes the same packed buffers with a
    // `bufIndex` parameter. This collapses 18 individual allocations to
    // 2 and reduces kernel-arg traffic for the prepare and finalize
    // kernels from 9 buffer args each to 1.
    m_costElemBytes = (bitDepth > 8) ? (int)sizeof(uint16_t) : (int)sizeof(uint8_t);
    const int costMaxW       = std::max(m_ssW, m_ssH);
    const int costMaxBufRows = std::max(m_ssW, m_ssH) / 2;
    m_costPitch              = costMaxW * m_costElemBytes;
    m_costSliceBytes         = m_costPitch * costMaxBufRows;
    const size_t totalCostBytes = (size_t)m_costSliceBytes * (size_t)MAA_NUM_COST_BUFFERS;

    m_costRawPacked    = m_cl->createBuffer(totalCostBytes, CL_MEM_READ_WRITE);
    m_costSmoothPacked = m_cl->createBuffer(totalCostBytes, CL_MEM_READ_WRITE);
    if (!m_costRawPacked || !m_costSmoothPacked) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate SangNom packed cost buffers (size=%llu bytes each).\n"),
            (unsigned long long)totalCostBytes);
        return RGY_ERR_MEMORY_ALLOC;
    }

    // Diagnostic: log total intermediate-buffer memory committed at init().
    {
        const size_t bytesSupersampled = (size_t)m_ssW * (size_t)m_ssH * (size_t)m_costElemBytes
            * RGY_CSP_PLANES[prm->frameIn.csp];
        const size_t bytesRotated      = bytesSupersampled;        // m_rotated, m_rotatedAA share shape
        const size_t bytesIntermediate = bytesSupersampled * 5     // m_supersampled, m_aaResult, m_unrotatedAA, m_rotated, m_rotatedAA
                                       + (size_t)prm->frameIn.width * prm->frameIn.height * m_costElemBytes
                                         * RGY_CSP_PLANES[prm->frameIn.csp];   // m_edgeMask at source res
        const size_t bytesCost         = totalCostBytes * 2; // raw + smooth (packed)
        const size_t bytesTotal        = bytesIntermediate + bytesCost;
        AddMessage(RGY_LOG_DEBUG,
            _T("MAA mem: intermediates=%.1f MiB, cost-buffers=%.1f MiB (packed, slice=%dx%d × 9 × 2 × %d B), total=%.1f MiB\n"),
            (double)bytesIntermediate / (1024.0 * 1024.0),
            (double)bytesCost / (1024.0 * 1024.0),
            costMaxW, costMaxBufRows, m_costElemBytes,
            (double)bytesTotal / (1024.0 * 1024.0));
        (void)bytesRotated;
    }

    AddMessage(RGY_LOG_DEBUG, _T("MAA init: src=%dx%d ss=%.2f -> ssDims=%dx%d, bitDepth=%d, peak=%d, aaf=%.3f, aacf=%.3f, mthreshScaled=%d\n"),
        prm->frameIn.width, prm->frameIn.height, prm->maa.ss, m_ssW, m_ssH,
        bitDepth, peak, m_aaf, m_aacf, m_mthreshScaled);

    setFilterInfo(prm->print() + strsprintf(_T(" (ssDims=%dx%d)"), m_ssW, m_ssH));
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterMaa::fturnLeftFrame(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                      RGYOpenCLQueue &queue,
                                      const std::vector<RGYOpenCLEvent> &wait_events,
                                      int planeCount) {
    const int srcPlanes = RGY_CSP_PLANES[pSrc->csp];
    const int planes    = (planeCount < 0) ? srcPlanes : std::min(planeCount, srcPlanes);
    auto waitFirst = wait_events;
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto sP = getPlane(pSrc, (RGY_PLANE)iplane);
        const auto dP = getPlane(pDst, (RGY_PLANE)iplane);
        // After 90° rotation: dst.width = src.height; dst.height = src.width.
        RGYWorkSize local(MAA_BLOCK_X, MAA_BLOCK_Y);
        RGYWorkSize global(sP.height, sP.width);
        const auto &waitHere = (iplane == 0) ? waitFirst : std::vector<RGYOpenCLEvent>();
        auto err = m_maa.get()->kernel("maa_fturn_left").config(queue, local, global, waitHere, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0], sP.width, sP.height,
            (cl_mem)dP.ptr[0], dP.pitch[0]);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at maa_fturn_left (plane %d): %s.\n"),
                iplane, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterMaa::fturnRightFrame(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                       RGYOpenCLQueue &queue,
                                       const std::vector<RGYOpenCLEvent> &wait_events,
                                       int planeCount) {
    const int srcPlanes = RGY_CSP_PLANES[pSrc->csp];
    const int planes    = (planeCount < 0) ? srcPlanes : std::min(planeCount, srcPlanes);
    auto waitFirst = wait_events;
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto sP = getPlane(pSrc, (RGY_PLANE)iplane);
        const auto dP = getPlane(pDst, (RGY_PLANE)iplane);
        RGYWorkSize local(MAA_BLOCK_X, MAA_BLOCK_Y);
        RGYWorkSize global(sP.height, sP.width);
        const auto &waitHere = (iplane == 0) ? waitFirst : std::vector<RGYOpenCLEvent>();
        auto err = m_maa.get()->kernel("maa_fturn_right").config(queue, local, global, waitHere, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0], sP.width, sP.height,
            (cl_mem)dP.ptr[0], dP.pitch[0]);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at maa_fturn_right (plane %d): %s.\n"),
                iplane, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterMaa::sangnomPassPlane(const RGYFrameInfo *pSrc, RGYFrameInfo *pDst,
                                        RGY_PLANE plane, float aaf,
                                        RGYOpenCLQueue &queue,
                                        const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, plane);
    const auto dP = getPlane(pDst, plane);
    const int srcW = sP.width;
    const int srcH = sP.height;
    const int bufW = srcW;
    const int bufH = srcH / 2;     // half-height: one row per missing source row

    // Stage 1 — prepare: 9 raw cost slices from the source plane, written
    // into the single packed cost buffer at sub-buffer offsets 0..8 ×
    // m_costSliceBytes.
    {
        RGYWorkSize local(MAA_BLOCK_X, MAA_BLOCK_Y);
        RGYWorkSize global(bufW, bufH);
        auto err = m_maa.get()->kernel("maa_sangnom_prepare")
            .config(queue, local, global, wait_events, nullptr).launch(
                (cl_mem)sP.ptr[0], sP.pitch[0], srcW, srcH,
                m_costRawPacked->mem(),
                m_costPitch, m_costSliceBytes,
                bufW, bufH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at maa_sangnom_prepare (plane %d): %s.\n"),
                (int)plane, get_err_mes(err));
            return err;
        }
    }

    // Stage 2 — smooth (3×7 spatial blur of each cost slice).
    //
    // [MAA-3D-SMOOTH] Single 3-D dispatch: (bufW, bufH, 9) replaces 9
    // separate (bufW, bufH) dispatches. The kernel now uses
    // get_global_id(2) for the slice index. Each pass drops 8 of the 9
    // per-slice enqueue calls; over 2 passes per frame that is 16 fewer
    // CL enqueue operations per frame. The 2-D fallback kernel
    // `maa_sangnom_smooth` is kept in the .cl for reference but is not
    // invoked here.
    {
        RGYWorkSize local(MAA_BLOCK_X, MAA_BLOCK_Y, 1);
        RGYWorkSize global(bufW, bufH, MAA_NUM_COST_BUFFERS);
        auto err = m_maa.get()->kernel("maa_sangnom_smooth_3d")
            .config(queue, local, global, {}, nullptr).launch(
                m_costRawPacked->mem(),
                m_costSmoothPacked->mem(),
                m_costPitch, m_costSliceBytes,
                bufW, bufH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at maa_sangnom_smooth_3d (plane %d): %s.\n"),
                (int)plane, get_err_mes(err));
            return err;
        }
    }

    // Stage 3 — finalize: per-output-pixel decision cascade. Reads each of
    // the 9 smoothed cost values from the packed smoothed buffer at
    // sub-buffer offsets 0..8 × m_costSliceBytes.
    {
        RGYWorkSize local(MAA_BLOCK_X, MAA_BLOCK_Y);
        RGYWorkSize global(srcW, srcH);
        auto err = m_maa.get()->kernel("maa_sangnom_finalize")
            .config(queue, local, global, {}, nullptr).launch(
                (cl_mem)sP.ptr[0], sP.pitch[0], srcW, srcH,
                m_costSmoothPacked->mem(),
                m_costPitch, m_costSliceBytes,
                bufW, bufH,
                (cl_mem)dP.ptr[0], dP.pitch[0],
                aaf);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at maa_sangnom_finalize (plane %d): %s.\n"),
                (int)plane, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterMaa::runEdgeSobel(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                    int mthreshScaled, RGYOpenCLQueue &queue,
                                    const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, RGY_PLANE_Y);
    RGYWorkSize local(MAA_BLOCK_X, MAA_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_maa.get()->kernel("maa_edge_sobel")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            mthreshScaled);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at maa_edge_sobel: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterMaa::runInflate(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                  RGYOpenCLQueue &queue,
                                  const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, RGY_PLANE_Y);
    RGYWorkSize local(MAA_BLOCK_X, MAA_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_maa.get()->kernel("maa_inflate")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at maa_inflate: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterMaa::runMergePlane(RGYFrameInfo *pDst, const RGYFrameInfo *pSrcA,
                                     const RGYFrameInfo *pSrcB, const RGYFrameInfo *pMask,
                                     RGY_PLANE plane,
                                     RGYOpenCLQueue &queue,
                                     const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto aP = getPlane(pSrcA, plane);
    const auto bP = getPlane(pSrcB, plane);
    // Mask plane is stored at the same plane index so that for chroma
    // we read U (since maskSubsample wrote into m_chromaMask's U plane).
    // For luma we read Y, for chroma we read U. Both have valid mask data
    // in the corresponding plane.
    const RGY_PLANE maskPlane = (plane == RGY_PLANE_Y) ? RGY_PLANE_Y : RGY_PLANE_U;
    const auto mP = getPlane(pMask, maskPlane);
    const auto dP = getPlane(pDst, plane);

    RGYWorkSize local(MAA_BLOCK_X, MAA_BLOCK_Y);
    RGYWorkSize global(dP.width, dP.height);
    auto err = m_maa.get()->kernel("maa_merge")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)aP.ptr[0], aP.pitch[0],
            (cl_mem)bP.ptr[0], bP.pitch[0],
            (cl_mem)mP.ptr[0], mP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            dP.width, dP.height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at maa_merge (plane %d): %s.\n"),
            (int)plane, get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterMaa::runShowOverlay(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                      const RGYFrameInfo *pMask, RGY_PLANE plane,
                                      RGYOpenCLQueue &queue,
                                      const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, plane);
    // Mask is always a luma-plane buffer (the inflated edge mask), so for
    // show overlay we only ever need the Y plane of pMask.
    const auto mP = getPlane(pMask, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, plane);
    RGYWorkSize local(MAA_BLOCK_X, MAA_BLOCK_Y);
    RGYWorkSize global(dP.width, dP.height);
    auto err = m_maa.get()->kernel("maa_show_overlay")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)mP.ptr[0], mP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            dP.width, dP.height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at maa_show_overlay (plane %d): %s.\n"),
            (int)plane, get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterMaa::runShowDarken(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                     RGY_PLANE plane,
                                     RGYOpenCLQueue &queue,
                                     const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, plane);
    const auto dP = getPlane(pDst, plane);
    RGYWorkSize local(MAA_BLOCK_X, MAA_BLOCK_Y);
    RGYWorkSize global(dP.width, dP.height);
    auto err = m_maa.get()->kernel("maa_show_darken")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            dP.width, dP.height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at maa_show_darken (plane %d): %s.\n"),
            (int)plane, get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterMaa::runMaskSubsample(RGYFrameInfo *pChromaMaskDst,
                                        const RGYFrameInfo *pLumaMaskSrc,
                                        RGYOpenCLQueue &queue,
                                        const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto luma   = getPlane(pLumaMaskSrc, RGY_PLANE_Y);
    const auto chroma = getPlane(pChromaMaskDst, RGY_PLANE_U);
    const int subSampleX = (chroma.width  > 0) ? std::max(1, luma.width  / chroma.width)  : 1;
    const int subSampleY = (chroma.height > 0) ? std::max(1, luma.height / chroma.height) : 1;

    RGYWorkSize local(MAA_BLOCK_X, MAA_BLOCK_Y);
    RGYWorkSize global(chroma.width, chroma.height);
    auto err = m_maa.get()->kernel("maa_mask_subsample")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)luma.ptr[0], luma.pitch[0],
            luma.width, luma.height,
            (cl_mem)chroma.ptr[0], chroma.pitch[0],
            chroma.width, chroma.height,
            subSampleX, subSampleY);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at maa_mask_subsample (sub=%dx%d): %s.\n"),
            subSampleX, subSampleY, get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterMaa::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue_main,
    const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    (void)event;
    *pOutputFrameNum  = 0;
    ppOutputFrames[0] = nullptr;

    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }

    if (!m_maa.get()) {
        AddMessage(RGY_LOG_ERROR, _T("MAA OpenCL program failed to build (options: %s).\n"),
            char_to_tstring(m_maaBuildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamMaa>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const bool processChroma = prm->maa.chroma;
    const bool maskOn        = prm->maa.mask;
    const int  showMode      = prm->maa.show;     // 0=normal, 1=mask on input, 2=mask on AA result
    const int  planes        = RGY_CSP_PLANES[pInputFrame->csp];

    // Show modes need the mask regardless of `mask=` setting (because show is
    // a debug overlay of the mask itself). They also need the AA result for
    // show=2 — but the AA result is computed unconditionally by stages 1-6
    // below, so the only addition vs the normal path is the mask construction
    // for show=1 + mask=off and the overlay/darken kernels at the very end.
    const bool needMask = maskOn || (showMode > 0);

    // chroma=false の場合は luma plane のみ AA パイプラインに流す。
    // chroma plane は最終段で input から copyPlane で上書きされるため、
    // resize / FTurn / sangnom 前 copyFrame をすべて Y のみに絞る。
    const int  aaPlanes = processChroma ? planes : 1;
    const RGY_CSP lumaCsp = (RGY_CSP_BIT_DEPTH[pInputFrame->csp] > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;

    // ============== AA PIPELINE ==============
    //
    // Stages 1-6 (per Prompt 2) lift the input to ssW×ssH, run two
    // SangNom2 passes (one per axis, with FTurn between them), and
    // resize back down. With chroma=true the SangNom passes also run
    // on the U and V planes using the chroma threshold m_aacf; with
    // chroma=false the chroma planes are NOT processed in the AA
    // pipeline at all (luma-only resize / 1-plane FTurn / no chroma
    // copyFrame) — the chroma output planes are filled by the
    // final input→output copyPlane step in stages 7c/7e/8.

    // ---- 1. Resize up: source resolution → ssW × ssH ----
    RGYFrameInfo *pSupersampled = &m_supersampled->frame;
    {
        int dummyOutNum = 0;
        RGY_ERR err = RGY_ERR_NONE;
        if (processChroma) {
            RGYFrameInfo *outArr[1] = { pSupersampled };
            err = m_resizeUp->filter(const_cast<RGYFrameInfo *>(pInputFrame),
                (RGYFrameInfo **)&outArr, &dummyOutNum, queue_main, wait_events, nullptr);
        } else {
            RGYFrameInfo inputLuma  = getPlane(pInputFrame,  RGY_PLANE_Y);
            RGYFrameInfo outputLuma = getPlane(pSupersampled, RGY_PLANE_Y);
            inputLuma.csp  = lumaCsp;
            outputLuma.csp = lumaCsp;
            RGYFrameInfo *outArr[1] = { &outputLuma };
            err = m_resizeUpLuma->filter(&inputLuma,
                (RGYFrameInfo **)&outArr, &dummyOutNum, queue_main, wait_events, nullptr);
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("MAA resize-up failed: %s.\n"), get_err_mes(err));
            return err;
        }
    }

    // ---- 2. FTurnLeft: ssW × ssH → ssH × ssW ----
    RGYFrameInfo *pRotated = &m_rotated->frame;
    {
        auto err = fturnLeftFrame(pRotated, pSupersampled, queue_main, {}, aaPlanes);
        if (err != RGY_ERR_NONE) return err;
    }

    // ---- 3. SangNom2 pass 1: anti-aliases the original vertical edges
    //          (which are horizontal in the rotated frame). ----
    RGYFrameInfo *pRotatedAA = &m_rotatedAA->frame;
    {
        // chroma=true 時のみ全プレーン copy (chroma plane は SangNom を
        // 走らせる U/V でも、走らない場合の carry-through でも、書込み元
        // となる pRotated の chroma を pRotatedAA 側に揃えるため必要)。
        // chroma=false 時は pRotatedAA.chroma を後段で読まないのでスキップ。
        if (processChroma) {
            auto err = m_cl->copyFrame(pRotatedAA, pRotated, nullptr, queue_main, {});
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("MAA chroma carry pass1 failed: %s.\n"), get_err_mes(err));
                return err;
            }
        }
        auto err = sangnomPassPlane(pRotated, pRotatedAA, RGY_PLANE_Y, m_aaf, queue_main, {});
        if (err != RGY_ERR_NONE) return err;
        if (processChroma && planes >= 3) {
            err = sangnomPassPlane(pRotated, pRotatedAA, RGY_PLANE_U, m_aacf, queue_main, {});
            if (err != RGY_ERR_NONE) return err;
            err = sangnomPassPlane(pRotated, pRotatedAA, RGY_PLANE_V, m_aacf, queue_main, {});
            if (err != RGY_ERR_NONE) return err;
        }
    }

    // ---- 4. FTurnRight: ssH × ssW → ssW × ssH ----
    RGYFrameInfo *pUnrotatedAA = &m_unrotatedAA->frame;
    {
        auto err = fturnRightFrame(pUnrotatedAA, pRotatedAA, queue_main, {}, aaPlanes);
        if (err != RGY_ERR_NONE) return err;
    }

    // ---- 5. SangNom2 pass 2: anti-aliases the original horizontal edges. ----
    RGYFrameInfo *pAaResult = &m_aaResult->frame;
    {
        if (processChroma) {
            auto err = m_cl->copyFrame(pAaResult, pUnrotatedAA, nullptr, queue_main, {});
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("MAA chroma carry pass2 failed: %s.\n"), get_err_mes(err));
                return err;
            }
        }
        auto err = sangnomPassPlane(pUnrotatedAA, pAaResult, RGY_PLANE_Y, m_aaf, queue_main, {});
        if (err != RGY_ERR_NONE) return err;
        if (processChroma && planes >= 3) {
            err = sangnomPassPlane(pUnrotatedAA, pAaResult, RGY_PLANE_U, m_aacf, queue_main, {});
            if (err != RGY_ERR_NONE) return err;
            err = sangnomPassPlane(pUnrotatedAA, pAaResult, RGY_PLANE_V, m_aacf, queue_main, {});
            if (err != RGY_ERR_NONE) return err;
        }
    }

    // ---- 6. Resize down: ssW × ssH → source resolution.
    //          When mask is OFF we write directly into m_frameBuf[0] (the
    //          output). When mask is ON we route the resized AA result
    //          through a private scratch buffer (m_supersampled is reused
    //          as the destination here — its allocation envelopes any
    //          source-resolution frame and it has the correct CSP) so
    //          the merge stage has both the source (pInputFrame) and the
    //          AA-only result available.
    RGYFrameInfo *pOut       = &m_frameBuf[0]->frame;
    RGYFrameInfo *pAaSrcRes  = nullptr;
    {
        // For mask=on we land the AA-resolution-restored frame into
        // m_inflatedMask's frame-info: it's a same-CSP same-resolution
        // RGYCLFrame allocated at source resolution (see init()) and
        // unused at this point in the pipeline. The mask construction
        // below writes into m_edgeMask/m_inflatedMask AFTER we're done
        // with m_inflatedMask as a temporary — but to keep the code
        // simple and obvious, we use m_supersampled instead. m_supersampled
        // is allocated at ssW×ssH (≥ source dims) and has the right CSP;
        // resizeDown writes into the leading source-shape sub-region.
        // To avoid pitch confusion, just allocate the AA-result-at-
        // source-res by writing through m_frameBuf[0] when mask=off, and
        // for mask=on, we copy m_frameBuf[0] aside before merge. Simpler
        // and uses 1 extra source-res copy per frame.
        int dummyOutNum = 0;
        RGY_ERR err = RGY_ERR_NONE;
        if (processChroma) {
            RGYFrameInfo *outArr[1] = { pOut };
            err = m_resizeDown->filter(pAaResult,
                (RGYFrameInfo **)&outArr, &dummyOutNum, queue_main, {}, nullptr);
        } else {
            RGYFrameInfo aaLuma  = getPlane(pAaResult, RGY_PLANE_Y);
            RGYFrameInfo outLuma = getPlane(pOut, RGY_PLANE_Y);
            aaLuma.csp  = lumaCsp;
            outLuma.csp = lumaCsp;
            RGYFrameInfo *outArr[1] = { &outLuma };
            err = m_resizeDownLuma->filter(&aaLuma,
                (RGYFrameInfo **)&outArr, &dummyOutNum, queue_main, {}, nullptr);
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("MAA resize-down failed: %s.\n"), get_err_mes(err));
            return err;
        }
        pAaSrcRes = pOut;     // After resize-down, pOut == AA result at source res
    }

    // ============== MASK + MERGE ==============
    //
    // When mask=off: pOut already contains the full AA result for luma.
    //   - chroma=false: copy chroma from pInputFrame to pOut (overwrite the
    //     resize-down's chroma which has been processed by the rotation
    //     round-trip but is otherwise identical to source).
    //   - chroma=true:  pOut already contains chroma AA result. Done.
    //
    // When mask=on:
    //   - Build an edge mask on the LUMA plane of pInputFrame:
    //       maa_edge_sobel(input_luma, mthreshScaled) -> m_edgeMask.luma
    //       maa_inflate(m_edgeMask.luma)              -> m_inflatedMask.luma
    //   - Merge for luma: maa_merge(input_luma, aaSrcRes_luma, mask) -> pOut.luma
    //     But pOut.luma already holds the AA result, so we re-route via a
    //     temporary. We copy pOut.luma -> m_inflatedMask.luma's V plane?
    //     No — simpler: write the merge OUTPUT into m_edgeMask.luma, then
    //     copy m_edgeMask.luma -> pOut.luma at the very end. That avoids
    //     any aliasing because the merge needs A=input, B=aa, mask, out=any.
    //   - For chroma=true: sub-sample mask if chroma is sub-sampled, then
    //     merge each chroma plane.
    //   - For chroma=false: copy chroma from pInputFrame to pOut.

    // 7a. Build the mask if either the merge path or show mode wants it.
    if (needMask) {
        auto err = runEdgeSobel(&m_edgeMask->frame, pInputFrame, m_mthreshScaled, queue_main, {});
        if (err != RGY_ERR_NONE) return err;
        err = runInflate(&m_inflatedMask->frame, &m_edgeMask->frame, queue_main, {});
        if (err != RGY_ERR_NONE) return err;
    }

    if (showMode == 1 || showMode == 2) {
        // ============== SHOW MODE (debug overlay) ==============
        //
        // show=1: half-darken the input luma and overlay the inflated edge
        //         mask so the user can SEE which pixels would receive AA.
        // show=2: same, but using the post-AA luma as the underlay so the
        //         user can see both the AA effect AND the mask region.
        //
        // Chroma planes are darkened (no mask) using the SOURCE chroma —
        // this matches the AviSynth MAA2 reference's `mt_lut("x 2 /")`
        // applied to all planes. The colour-tint dimming is the visual
        // cue that this is debug output.

        const RGYFrameInfo *underlayLuma =
            (showMode == 2) ? pAaSrcRes : pInputFrame;

        // Luma overlay: write into m_edgeMask scratch (its content is now
        // free to overwrite; we only needed m_inflatedMask going forward).
        {
            auto err = runShowOverlay(&m_edgeMask->frame, underlayLuma,
                                       &m_inflatedMask->frame, RGY_PLANE_Y,
                                       queue_main, {});
            if (err != RGY_ERR_NONE) return err;
            const auto srcP = getPlane(&m_edgeMask->frame, RGY_PLANE_Y);
            const auto dstP = getPlane(pOut, RGY_PLANE_Y);
            err = m_cl->copyPlane(const_cast<RGYFrameInfo *>(&dstP), &srcP, nullptr, queue_main, {});
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("MAA show-overlay writeback failed: %s.\n"), get_err_mes(err));
                return err;
            }
        }

        // Chroma darken (always from input source — matches the reference).
        if (planes >= 3) {
            for (RGY_PLANE pl : { RGY_PLANE_U, RGY_PLANE_V }) {
                auto err = runShowDarken(&m_edgeMask->frame, pInputFrame, pl, queue_main, {});
                if (err != RGY_ERR_NONE) return err;
                const auto srcP = getPlane(&m_edgeMask->frame, pl);
                const auto dstP = getPlane(pOut, pl);
                err = m_cl->copyPlane(const_cast<RGYFrameInfo *>(&dstP), &srcP, nullptr, queue_main, {});
                if (err != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("MAA show-darken writeback (plane %d) failed: %s.\n"),
                        (int)pl, get_err_mes(err));
                    return err;
                }
            }
        }
    } else if (maskOn) {
        // ============== NORMAL MASK + MERGE ==============
        // 7c. Merge luma. Output goes into m_edgeMask (which is no longer
        //     needed at this point) so we don't read-after-write pOut.
        {
            auto err = runMergePlane(&m_edgeMask->frame,
                                      pInputFrame, pAaSrcRes, &m_inflatedMask->frame,
                                      RGY_PLANE_Y, queue_main, {});
            if (err != RGY_ERR_NONE) return err;
        }

        // 7d. Copy merged luma back into pOut.luma.
        {
            const auto srcP = getPlane(&m_edgeMask->frame, RGY_PLANE_Y);
            const auto dstP = getPlane(pOut, RGY_PLANE_Y);
            auto err = m_cl->copyPlane(const_cast<RGYFrameInfo *>(&dstP), &srcP, nullptr, queue_main, {});
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("MAA merge-luma writeback failed: %s.\n"), get_err_mes(err));
                return err;
            }
        }

        // 7e. Chroma path.
        if (processChroma && planes >= 3) {
            // Sub-sample the luma mask to chroma resolution. m_chromaMask's
            // U plane will hold the down-sampled mask; runMergePlane below
            // reads from there for both U and V (chroma sub-sampling is
            // colour-channel agnostic).
            {
                auto err = runMaskSubsample(&m_chromaMask->frame, &m_inflatedMask->frame, queue_main, {});
                if (err != RGY_ERR_NONE) return err;
            }
            // Merge U into a temp (m_edgeMask's U), then copyPlane to pOut.U.
            // Same for V.
            for (RGY_PLANE pl : { RGY_PLANE_U, RGY_PLANE_V }) {
                auto err = runMergePlane(&m_edgeMask->frame,
                                          pInputFrame, pAaSrcRes, &m_chromaMask->frame,
                                          pl, queue_main, {});
                if (err != RGY_ERR_NONE) return err;
                const auto srcP = getPlane(&m_edgeMask->frame, pl);
                const auto dstP = getPlane(pOut, pl);
                err = m_cl->copyPlane(const_cast<RGYFrameInfo *>(&dstP), &srcP, nullptr, queue_main, {});
                if (err != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("MAA merge-chroma writeback (plane %d) failed: %s.\n"),
                        (int)pl, get_err_mes(err));
                    return err;
                }
            }
        } else if (planes >= 3) {
            // chroma=false + mask=on: copy chroma from input (the AA path
            // round-tripped chroma through rotation, but we want exactly
            // the source's chroma in the output).
            for (RGY_PLANE pl : { RGY_PLANE_U, RGY_PLANE_V }) {
                const auto srcP = getPlane(pInputFrame, pl);
                const auto dstP = getPlane(pOut, pl);
                auto err = m_cl->copyPlane(const_cast<RGYFrameInfo *>(&dstP), &srcP, nullptr, queue_main, {});
                if (err != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("MAA chroma copy (plane %d) failed: %s.\n"),
                        (int)pl, get_err_mes(err));
                    return err;
                }
            }
        }
    } else {
        // mask=off: pOut.luma already has the full AA result. Override the
        // chroma planes per the chroma flag.
        if (!processChroma && planes >= 3) {
            for (RGY_PLANE pl : { RGY_PLANE_U, RGY_PLANE_V }) {
                const auto srcP = getPlane(pInputFrame, pl);
                const auto dstP = getPlane(pOut, pl);
                auto err = m_cl->copyPlane(const_cast<RGYFrameInfo *>(&dstP), &srcP, nullptr, queue_main, {});
                if (err != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("MAA chroma copy (plane %d) failed: %s.\n"),
                        (int)pl, get_err_mes(err));
                    return err;
                }
            }
        }
        // chroma=true + mask=off: pOut already has full AA result for all planes.
    }

    ppOutputFrames[0] = pOut;
    *pOutputFrameNum  = 1;
    return RGY_ERR_NONE;
}

void RGYFilterMaa::close() {
    m_resizeUp.reset();
    m_resizeDown.reset();
    m_resizeUpLuma.reset();
    m_resizeDownLuma.reset();
    m_supersampled.reset();
    m_rotated.reset();
    m_rotatedAA.reset();
    m_unrotatedAA.reset();
    m_aaResult.reset();
    m_edgeMask.reset();
    m_inflatedMask.reset();
    m_chromaMask.reset();
    m_costRawPacked.reset();
    m_costSmoothPacked.reset();
    m_costPitch      = 0;
    m_costSliceBytes = 0;
    m_costElemBytes  = 1;
    m_maa.clear();
    m_maaBuildOptions.clear();
    m_ssW = 0;
    m_ssH = 0;
    m_aaf = 0.0f;
    m_aacf = 0.0f;
    m_mthreshScaled = 0;
    m_frameBuf.clear();
    m_cl.reset();
}
