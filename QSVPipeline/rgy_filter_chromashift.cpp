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
#include "convert_csp.h"
#include "rgy_filter_chromashift.h"

static const int CHROMASHIFT_BLOCK_X = 32;
static const int CHROMASHIFT_BLOCK_Y = 8;

RGYFilterChromaShift::RGYFilterChromaShift(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_chromashift(),
    m_buildOptions(),
    m_srcImagePool(),
    m_signY(),
    m_signUV(),
    m_statsBuf(),
    m_statsHost(),
    m_acceptedDx(),
    m_acceptedDy(),
    m_seenAnalysisFrames(0),
    m_skippedAutoFrames(0),
    m_warmupSkippedFrames(0),
    m_analysisComplete(false),
    m_resolvedShiftX(0.0f),
    m_resolvedShiftY(0.0f) {
    m_name = _T("chromashift");
}

RGYFilterChromaShift::~RGYFilterChromaShift() {
    close();
}

RGY_ERR RGYFilterChromaShift::checkParam(const std::shared_ptr<RGYFilterParamChromaShift> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->chromashift.x < -4.0f || 4.0f < prm->chromashift.x) {
        prm->chromashift.x = clamp(prm->chromashift.x, -4.0f, 4.0f);
        AddMessage(RGY_LOG_WARN, _T("x should be in range of %.1f - %.1f.\n"), -4.0f, 4.0f);
    }
    if (prm->chromashift.y < -4.0f || 4.0f < prm->chromashift.y) {
        prm->chromashift.y = clamp(prm->chromashift.y, -4.0f, 4.0f);
        AddMessage(RGY_LOG_WARN, _T("y should be in range of %.1f - %.1f.\n"), -4.0f, 4.0f);
    }
    if (prm->chromashift.show < 0 || prm->chromashift.show > 1) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid show=%d: must be 0 (normal) or 1 (laplacian).\n"),
            prm->chromashift.show);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->chromashift.auto_frames < 1 || 100 < prm->chromashift.auto_frames) {
        prm->chromashift.auto_frames = clamp(prm->chromashift.auto_frames, 1, 100);
        AddMessage(RGY_LOG_WARN, _T("auto_frames should be in range of %d - %d.\n"), 1, 100);
    }
    if (prm->chromashift.auto_min_pairs < 10 || 10000 < prm->chromashift.auto_min_pairs) {
        prm->chromashift.auto_min_pairs = clamp(prm->chromashift.auto_min_pairs, 10, 10000);
        AddMessage(RGY_LOG_WARN, _T("auto_min_pairs should be in range of %d - %d.\n"), 10, 10000);
    }
    if (prm->chromashift.auto_detect && (prm->chromashift.x != 0.0f || prm->chromashift.y != 0.0f)) {
        AddMessage(RGY_LOG_WARN,
            _T("chromashift: auto=true takes precedence over x=%.2f, y=%.2f. ")
            _T("Those values are used only as a fallback if auto-analysis rejects too many frames.\n"),
            prm->chromashift.x, prm->chromashift.y);
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterChromaShift::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamChromaShift>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    prm->frameOut.picstruct = prm->frameIn.picstruct;

    const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    const int maxVal   = (1 << bitDepth) - 1;

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamChromaShift>(m_param);
    if (!m_chromashift.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        m_buildOptions = strsprintf("-D Type=%s -D bit_depth=%d -D max_val=%d",
            bitDepth > 8 ? "ushort" : "uchar",
            bitDepth, maxVal);
        AddMessage(RGY_LOG_DEBUG, _T("Starting async build for RGY_FILTER_CHROMASHIFT_CL: %s\n"),
            char_to_tstring(m_buildOptions).c_str());
        m_chromashift.set(m_cl->buildResourceAsync(_T("RGY_FILTER_CHROMASHIFT_CL"), _T("EXE_DATA"), m_buildOptions.c_str()));
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output buffer: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // Auto-detect buffers. Allocated only when auto_detect=true and only
    // before analysis completes; freed after lock-in to release ~600KB
    // on 480p sources.
    if (prm->chromashift.auto_detect) {
        const size_t lumaPx = (size_t)prm->frameOut.width * (size_t)prm->frameOut.height;
        m_signY  = m_cl->createBuffer(lumaPx, CL_MEM_READ_WRITE);
        m_signUV = m_cl->createBuffer(lumaPx, CL_MEM_READ_WRITE);
        if (!m_signY || !m_signUV) {
            AddMessage(RGY_LOG_ERROR, _T("chromashift: failed to allocate sign-map buffers.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_statsBuf = m_cl->createBuffer(3 * sizeof(int), CL_MEM_READ_WRITE);
        if (!m_statsBuf) {
            AddMessage(RGY_LOG_ERROR, _T("chromashift: failed to allocate stats buffer.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_statsHost.assign(3, 0);
    } else {
        m_signY.reset();
        m_signUV.reset();
        m_statsBuf.reset();
        m_statsHost.clear();
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterChromaShift::runLapSignY(cl_mem dstSignBuf, const RGYFrameInfo *pSrcY,
                                          RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    const char *kernel_name = "chromashift_lapsign_y";
    RGYWorkSize local(CHROMASHIFT_BLOCK_X, CHROMASHIFT_BLOCK_Y);
    RGYWorkSize global(pSrcY->width, pSrcY->height);
    // signPitch == lumaWidth (sign map allocated with no padding).
    auto err = m_chromashift.get()->kernel(kernel_name).config(queue, local, global, wait_events, nullptr).launch(
        (cl_mem)pSrcY->ptr[0], pSrcY->pitch[0],
        dstSignBuf, pSrcY->width,
        pSrcY->width, pSrcY->height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"),
            char_to_tstring(kernel_name).c_str(), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterChromaShift::runLapSignUV(cl_mem dstSignBuf,
                                            const RGYFrameInfo *pSrcU, const RGYFrameInfo *pSrcV,
                                            int lumaW, int lumaH, int subX, int subY,
                                            RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    const char *kernel_name = "chromashift_lapsign_uv";
    RGYWorkSize local(CHROMASHIFT_BLOCK_X, CHROMASHIFT_BLOCK_Y);
    RGYWorkSize global(lumaW, lumaH);
    auto err = m_chromashift.get()->kernel(kernel_name).config(queue, local, global, wait_events, nullptr).launch(
        (cl_mem)pSrcU->ptr[0], pSrcU->pitch[0],
        (cl_mem)pSrcV->ptr[0], pSrcV->pitch[0],
        pSrcU->width, pSrcU->height,
        subX, subY,
        dstSignBuf, lumaW,
        lumaW, lumaH);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"),
            char_to_tstring(kernel_name).c_str(), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterChromaShift::runCorrelate(cl_mem signYBuf, cl_mem signUVBuf,
                                            int lumaW, int lumaH,
                                            RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    const char *kernel_name = "chromashift_correlate";
    RGYWorkSize local(CHROMASHIFT_BLOCK_X, CHROMASHIFT_BLOCK_Y);
    RGYWorkSize global(lumaW, lumaH);
    auto err = m_chromashift.get()->kernel(kernel_name).config(queue, local, global, wait_events, nullptr).launch(
        signYBuf,  lumaW,
        signUVBuf, lumaW,
        lumaW, lumaH,
        m_statsBuf->mem());
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"),
            char_to_tstring(kernel_name).c_str(), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterChromaShift::runShiftPlane(RGYFrameInfo *pDstPlane, const RGYFrameInfo *pSrcImgPlane,
                                             float shift_x_chroma, float shift_y_chroma,
                                             RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                                             RGYOpenCLEvent *event) {
    const char *kernel_name = "chromashift_shift";
    RGYWorkSize local(CHROMASHIFT_BLOCK_X, CHROMASHIFT_BLOCK_Y);
    RGYWorkSize global(pDstPlane->width, pDstPlane->height);
    auto err = m_chromashift.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pSrcImgPlane->ptr[0],
        (cl_mem)pDstPlane->ptr[0], pDstPlane->pitch[0],
        pDstPlane->width, pDstPlane->height,
        shift_x_chroma, shift_y_chroma);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"),
            char_to_tstring(kernel_name).c_str(), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterChromaShift::runLaplacianToLuma(RGYFrameInfo *pDstY, const RGYFrameInfo *pSrcC,
                                                  int subX, int subY,
                                                  RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                                                  RGYOpenCLEvent *event) {
    const char *kernel_name = "chromashift_laplacian";
    RGYWorkSize local(CHROMASHIFT_BLOCK_X, CHROMASHIFT_BLOCK_Y);
    RGYWorkSize global(pDstY->width, pDstY->height);
    auto err = m_chromashift.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pSrcC->ptr[0], pSrcC->pitch[0],
        pSrcC->width, pSrcC->height,
        subX, subY,
        (cl_mem)pDstY->ptr[0], pDstY->pitch[0],
        pDstY->width, pDstY->height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"),
            char_to_tstring(kernel_name).c_str(), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterChromaShift::runFillNeutral(RGYFrameInfo *pDstPlane,
                                              RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                                              RGYOpenCLEvent *event) {
    const char *kernel_name = "chromashift_fill_neutral";
    RGYWorkSize local(CHROMASHIFT_BLOCK_X, CHROMASHIFT_BLOCK_Y);
    RGYWorkSize global(pDstPlane->width, pDstPlane->height);
    auto err = m_chromashift.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pDstPlane->ptr[0], pDstPlane->pitch[0],
        pDstPlane->width, pDstPlane->height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"),
            char_to_tstring(kernel_name).c_str(), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterChromaShift::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue_main,
    const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    *pOutputFrameNum  = 0;
    ppOutputFrames[0] = nullptr;
    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamChromaShift>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!m_chromashift.get()) {
        AddMessage(RGY_LOG_ERROR, _T("chromashift OpenCL program failed to build (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }
    const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, m_frameBuf[0]->frame.mem_type);
    if (memcpyKind != RGYCLMemcpyD2D) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    RGYFrameInfo *pOut = &m_frameBuf[0]->frame;
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];

    // Compute chroma subsampling factors from the actual plane sizes.
    // YV12: subX=2, subY=2. YUV422: subX=2, subY=1. YUV444: subX=1, subY=1.
    const auto pY = getPlane(pInputFrame, RGY_PLANE_Y);
    int subX = 1;
    int subY = 1;
    if (planes >= 2) {
        const auto pU = getPlane(pInputFrame, RGY_PLANE_U);
        if (pU.width  > 0) subX = std::max(1, pY.width  / pU.width);
        if (pU.height > 0) subY = std::max(1, pY.height / pU.height);
    }

    // ---- Auto-detect analysis phase ----
    // While analysis is in progress, the filter passes chroma through
    // unmodified (effective shift = 0). The analysis path itself reads
    // pInputFrame's Y / U / V planes, accumulates per-frame stats in
    // m_acceptedDx/Dy, and decides when to lock in the final shift.
    // Manual x/y is ignored except as a fallback if analysis rejects
    // too many frames (see lock-in below).
    if (prm->chromashift.auto_detect && !m_analysisComplete && planes >= 2) {
        // m_seenAnalysisFrames is incremented BELOW after the kernel runs,
        // because the warm-up rule (B2 fix) skips counting frames whose
        // kernel returns zero pairs while m_acceptedDx is still empty. See
        // the accounting block after the readback.

        // Reset the [sum_dx, sum_dy, count] global counters to 0.
        const int zero = 0;
        cl_int clerr = clEnqueueFillBuffer(queue_main.get(), m_statsBuf->mem(),
                                            &zero, sizeof(int),
                                            0, 3 * sizeof(int),
                                            0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("chromashift: stats fill failed: %s.\n"), cl_errmes(clerr));
            return err_cl_to_rgy(clerr);
        }

        const auto planeSrcU = getPlane(pInputFrame, RGY_PLANE_U);
        const auto planeSrcV = (planes >= 3) ? getPlane(pInputFrame, RGY_PLANE_V) : planeSrcU;

        auto err = runLapSignY(m_signY->mem(), &pY, queue_main, wait_events);
        if (err != RGY_ERR_NONE) return err;
        err = runLapSignUV(m_signUV->mem(), &planeSrcU, &planeSrcV,
                           pY.width, pY.height, subX, subY, queue_main, {});
        if (err != RGY_ERR_NONE) return err;
        err = runCorrelate(m_signY->mem(), m_signUV->mem(),
                           pY.width, pY.height, queue_main, {});
        if (err != RGY_ERR_NONE) return err;

        // Readback the three counters.
        RGYOpenCLEvent readEvent;
        clerr = clEnqueueReadBuffer(queue_main.get(), m_statsBuf->mem(),
                                     CL_FALSE, 0, 3 * sizeof(int),
                                     m_statsHost.data(),
                                     0, nullptr, readEvent.reset_ptr());
        if (clerr != CL_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("chromashift: stats readback failed: %s.\n"), cl_errmes(clerr));
            return err_cl_to_rgy(clerr);
        }
        readEvent.wait();

        const int sum_dx = m_statsHost[0];
        const int sum_dy = m_statsHost[1];
        const int count  = m_statsHost[2];

        // B2 warm-up rule: frames whose kernel returned zero pairs while
        // m_acceptedDx is still empty are bypassed -- they don't count
        // toward the hardCap. This lets long intros / fades / static
        // logos pass without consuming the budget before real content
        // arrives. Bounded by ABS_SAFETY_CAP below so the encode cannot
        // hang on pathological all-zero content.
        const bool warmupSkip = (count == 0) && m_acceptedDx.empty();
        if (warmupSkip) {
            m_warmupSkippedFrames++;
            AddMessage(RGY_LOG_DEBUG,
                _T("chromashift: analysis frame %d bypassed (0 pairs, no accepted yet)\n"),
                m_warmupSkippedFrames + m_seenAnalysisFrames - 1);
        } else {
            m_seenAnalysisFrames++;
            if (count >= prm->chromashift.auto_min_pairs) {
                const double dx = (double)sum_dx / (double)count;
                const double dy = (double)sum_dy / (double)count;
                m_acceptedDx.push_back(dx);
                m_acceptedDy.push_back(dy);
                AddMessage(RGY_LOG_DEBUG,
                    _T("chromashift: analysis frame %d -> dx=%+.3f dy=%+.3f (%d pairs)\n"),
                    m_seenAnalysisFrames - 1, dx, dy, count);
            } else {
                m_skippedAutoFrames++;
                AddMessage(RGY_LOG_DEBUG,
                    _T("chromashift: analysis frame %d skipped (only %d pairs, min=%d)\n"),
                    m_seenAnalysisFrames - 1, count, prm->chromashift.auto_min_pairs);
            }
        }

        // Lock-in decisions:
        //   - Enough accepted frames -> compute mean, log result, apply
        //     the resolved shift from this frame onwards.
        //   - Hit the per-budget timeout (3x frames + 10 non-warmup
        //     analyses) -> fall back to mean of whatever we have, or to
        //     user x/y if too few accepted to be trustworthy.
        //   - Hit the absolute safety cap on total (warmup + non-warmup)
        //     analyses -> force timeout. Prevents the encode hanging on
        //     pathological content that never produces a single pair.
        static constexpr int ABS_SAFETY_CAP = 1000;
        const int hardCap = prm->chromashift.auto_frames * 3 + 10;
        const bool haveTarget = (int)m_acceptedDx.size() >= prm->chromashift.auto_frames;
        const bool hitTimeout = (m_seenAnalysisFrames >= hardCap)
                             || ((m_seenAnalysisFrames + m_warmupSkippedFrames) >= ABS_SAFETY_CAP);
        if (haveTarget || hitTimeout) {
            const int acceptedCount = (int)m_acceptedDx.size();
            const int minTrusted = std::max(1, prm->chromashift.auto_frames / 2);
            if (acceptedCount >= minTrusted) {
                double sumDx = 0.0;
                double sumDy = 0.0;
                for (double d : m_acceptedDx) sumDx += d;
                for (double d : m_acceptedDy) sumDy += d;
                m_resolvedShiftX = (float)(sumDx / (double)acceptedCount);
                m_resolvedShiftY = (float)(sumDy / (double)acceptedCount);
                AddMessage(RGY_LOG_INFO,
                    _T("chromashift: auto-detected x=%+.2f, y=%+.2f (from %d frames, skipped %d, warmup %d)\n"),
                    m_resolvedShiftX, m_resolvedShiftY, acceptedCount, m_skippedAutoFrames, m_warmupSkippedFrames);
            } else {
                // Fallback: honour user-supplied x/y if non-zero, else 0,0.
                m_resolvedShiftX = prm->chromashift.x;
                m_resolvedShiftY = prm->chromashift.y;
                AddMessage(RGY_LOG_WARN,
                    _T("chromashift: auto-analysis insufficient (only %d of %d target frames accepted ")
                    _T("after %d seen, %d bypassed in warmup). Falling back to x=%+.2f, y=%+.2f%s\n"),
                    acceptedCount, prm->chromashift.auto_frames,
                    m_seenAnalysisFrames, m_warmupSkippedFrames,
                    m_resolvedShiftX, m_resolvedShiftY,
                    (m_resolvedShiftX == 0.0f && m_resolvedShiftY == 0.0f) ? _T(" (no manual shift set)") : _T(""));
            }
            m_analysisComplete = true;
            // Free the analysis-only buffers now that lock-in has happened.
            m_signY.reset();
            m_signUV.reset();
            m_statsBuf.reset();
            m_statsHost.clear();
            m_acceptedDx.clear();
            m_acceptedDy.clear();
        }
    }

    // Determine the effective shift to apply on THIS frame:
    //   auto + analysis-in-progress -> 0, 0 (passthrough; analysis continues next frame)
    //   auto + analysis-complete    -> resolved values
    //   manual                      -> user-supplied x, y
    float effective_x;
    float effective_y;
    if (prm->chromashift.auto_detect) {
        if (m_analysisComplete) {
            effective_x = m_resolvedShiftX;
            effective_y = m_resolvedShiftY;
        } else {
            effective_x = 0.0f;
            effective_y = 0.0f;
        }
    } else {
        effective_x = prm->chromashift.x;
        effective_y = prm->chromashift.y;
    }

    // User shifts are in luma pixels; chroma plane shifts are smaller by
    // the subsampling factor. shift_x_luma=2 on YV12 -> shift_x_chroma=1.
    const float shift_x_chroma = effective_x / (float)subX;
    const float shift_y_chroma = effective_y / (float)subY;

    const bool passthrough = (effective_x == 0.0f && effective_y == 0.0f
                              && prm->chromashift.show == 0);

    auto eventsForFirst = wait_events;

    // --- Luma: always passthrough (unless show=laplacian which we handle below).
    {
        const auto planeSrcY = getPlane(pInputFrame, RGY_PLANE_Y);
        const auto planeDstY = getPlane(pOut, RGY_PLANE_Y);
        if (prm->chromashift.show != 1) {
            auto err = m_cl->copyPlane(const_cast<RGYFrameInfo *>(&planeDstY), &planeSrcY,
                                       nullptr, queue_main, eventsForFirst, (planes <= 1) ? event : nullptr);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("chromashift: luma copyPlane failed: %s.\n"), get_err_mes(err));
                return err;
            }
            eventsForFirst.clear();
        }
    }

    // --- show=laplacian: write |Laplacian(U)| into Y, neutral grey into U/V.
    if (prm->chromashift.show == 1) {
        const auto planeSrcU = getPlane(pInputFrame, RGY_PLANE_U);
        auto       planeDstY = getPlane(pOut, RGY_PLANE_Y);
        auto err = runLaplacianToLuma(&planeDstY, &planeSrcU, subX, subY, queue_main, eventsForFirst, (planes <= 1) ? event : nullptr);
        if (err != RGY_ERR_NONE) return err;
        eventsForFirst.clear();

        for (int i = 1; i < planes; i++) {
            auto planeDst = getPlane(pOut, (RGY_PLANE)i);
            err = runFillNeutral(&planeDst, queue_main, {}, (i == planes - 1) ? event : nullptr);
            if (err != RGY_ERR_NONE) return err;
        }

        pOut->timestamp    = pInputFrame->timestamp;
        pOut->duration     = pInputFrame->duration;
        pOut->inputFrameId = pInputFrame->inputFrameId;
        pOut->picstruct    = pInputFrame->picstruct;
        pOut->flags        = pInputFrame->flags;
        ppOutputFrames[0]  = pOut;
        *pOutputFrameNum   = 1;
        return RGY_ERR_NONE;
    }

    // --- Passthrough fast path: x=0, y=0 -> copy U and V unchanged, no kernel dispatch.
    if (passthrough) {
        for (int i = 1; i < planes; i++) {
            const auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
            auto       planeDst = getPlane(pOut, (RGY_PLANE)i);
            auto err = m_cl->copyPlane(&planeDst, &planeSrc, nullptr, queue_main, eventsForFirst, (i == planes - 1) ? event : nullptr);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("chromashift: chroma passthrough copyPlane (plane %d) failed: %s.\n"),
                    i, get_err_mes(err));
                return err;
            }
            eventsForFirst.clear();
        }
        pOut->timestamp    = pInputFrame->timestamp;
        pOut->duration     = pInputFrame->duration;
        pOut->inputFrameId = pInputFrame->inputFrameId;
        pOut->picstruct    = pInputFrame->picstruct;
        pOut->flags        = pInputFrame->flags;
        ppOutputFrames[0]  = pOut;
        *pOutputFrameNum   = 1;
        return RGY_ERR_NONE;
    }

    // --- Shift mode: bilinear sample from a image2d view of the input chroma.
    auto srcImage = m_cl->createImageFromFrameBuffer(*pInputFrame, true, CL_MEM_READ_ONLY, &m_srcImagePool);
    if (!srcImage) {
        AddMessage(RGY_LOG_ERROR, _T("chromashift: failed to create image for input frame.\n"));
        return RGY_ERR_MEM_OBJECT_ALLOCATION_FAILURE;
    }

    for (int i = 1; i < planes; i++) {
        const auto planeSrcImg = getPlane(&srcImage->frame, (RGY_PLANE)i);
        auto       planeDst    = getPlane(pOut, (RGY_PLANE)i);
        auto err = runShiftPlane(&planeDst, &planeSrcImg, shift_x_chroma, shift_y_chroma,
                                 queue_main, eventsForFirst, (i == planes - 1) ? event : nullptr);
        if (err != RGY_ERR_NONE) return err;
        eventsForFirst.clear();
    }

    pOut->timestamp    = pInputFrame->timestamp;
    pOut->duration     = pInputFrame->duration;
    pOut->inputFrameId = pInputFrame->inputFrameId;
    pOut->picstruct    = pInputFrame->picstruct;
    pOut->flags        = pInputFrame->flags;
    ppOutputFrames[0]  = pOut;
    *pOutputFrameNum   = 1;
    return RGY_ERR_NONE;
}

void RGYFilterChromaShift::close() {
    if (m_skippedAutoFrames > 0) {
        AddMessage(RGY_LOG_INFO,
            _T("chromashift: skipped %d frames during auto-analysis (insufficient zero-crossing pairs).\n"),
            m_skippedAutoFrames);
    }
    m_srcImagePool.clear();
    m_chromashift.clear();
    m_buildOptions.clear();
    m_signY.reset();
    m_signUV.reset();
    m_statsBuf.reset();
    m_statsHost.clear();
    m_acceptedDx.clear();
    m_acceptedDy.clear();
    m_seenAnalysisFrames   = 0;
    m_skippedAutoFrames    = 0;
    m_warmupSkippedFrames  = 0;
    m_analysisComplete   = false;
    m_resolvedShiftX     = 0.0f;
    m_resolvedShiftY     = 0.0f;
    m_frameBuf.clear();
    m_cl.reset();
}
