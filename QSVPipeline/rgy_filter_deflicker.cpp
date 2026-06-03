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
//
// AURORA deflicker -- van Roosmalen 1999 PhD thesis,
// "Restoration of archived film and video".

#include <algorithm>
#include <cmath>
#include "convert_csp.h"
#include "rgy_filter_deflicker.h"

static const int DEFLICKER_REDUCE_X = 32;
static const int DEFLICKER_REDUCE_Y = 8;

// Minimum sigma to clamp the divisor to: prevents division blow-up on
// near-flat frames. Expressed in 8-bit domain; scaled to working depth
// at use site.
static const double DEFLICKER_SIGMA_EPS_8BIT = 0.5;

RGYFilterDeflicker::RGYFilterDeflicker(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_deflicker(),
    m_buildOptions(),
    m_intermediate(),
    m_sumBuf(),
    m_sumSqBuf(),
    m_sumHost(),
    m_sumSqHost(),
    m_statsBufWGCount(0),
    m_rollingMeans(),
    m_rollingSigmas(),
    m_prevMult(1.0),
    m_prevAdd(0.0),
    m_haveDamping(false),
    m_skippedSceneFrames(0) {
    m_name = _T("deflicker");
}

RGYFilterDeflicker::~RGYFilterDeflicker() {
    close();
}

RGY_ERR RGYFilterDeflicker::checkParam(const std::shared_ptr<RGYFilterParamDeflicker> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto chromaFormat = RGY_CSP_CHROMA_FORMAT[prm->frameIn.csp];
    if (rgy_chromafmt_is_rgb(chromaFormat)) {
        AddMessage(RGY_LOG_ERROR, _T("deflicker supports YUV or monochrome formats only: %s.\n"),
            RGY_CSP_NAMES[prm->frameIn.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->deflicker.chroma && RGY_CSP_PLANES[prm->frameIn.csp] < 2) {
        prm->deflicker.chroma = false;
        AddMessage(RGY_LOG_WARN, _T("deflicker chroma processing requires chroma planes; disabled for %s.\n"),
            RGY_CSP_NAMES[prm->frameIn.csp]);
    }
    if (prm->deflicker.strength < 0.0f || 1.0f < prm->deflicker.strength) {
        prm->deflicker.strength = clamp(prm->deflicker.strength, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("strength should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }
    if (prm->deflicker.damping < 0.0f || 1.0f < prm->deflicker.damping) {
        prm->deflicker.damping = clamp(prm->deflicker.damping, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("damping should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }
    if (prm->deflicker.scene_threshold < 0.5f || 5.0f < prm->deflicker.scene_threshold) {
        prm->deflicker.scene_threshold = clamp(prm->deflicker.scene_threshold, 0.5f, 5.0f);
        AddMessage(RGY_LOG_WARN, _T("scene_threshold should be in range of %.1f - %.1f.\n"), 0.5f, 5.0f);
    }
    if (prm->deflicker.frames < 5 || 300 < prm->deflicker.frames) {
        prm->deflicker.frames = clamp(prm->deflicker.frames, 5, 300);
        AddMessage(RGY_LOG_WARN, _T("frames should be in range of %d - %d.\n"), 5, 300);
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDeflicker::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDeflicker>(pParam);
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

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamDeflicker>(m_param);
    const bool rebuild = !m_deflicker.get()
                      || !prmPrev
                      || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp];
    if (rebuild) {
        m_buildOptions = strsprintf("-D Type=%s -D bit_depth=%d -D max_val=%d -D DEFLICKER_REDUCE_X=%d -D DEFLICKER_REDUCE_Y=%d",
            bitDepth > 8 ? "ushort" : "uchar",
            bitDepth, maxVal,
            DEFLICKER_REDUCE_X, DEFLICKER_REDUCE_Y);
        AddMessage(RGY_LOG_DEBUG, _T("Starting async build for RGY_FILTER_DEFLICKER_CL: %s\n"),
            char_to_tstring(m_buildOptions).c_str());
        m_deflicker.set(m_cl->buildResourceAsync(_T("RGY_FILTER_DEFLICKER_CL"), _T("EXE_DATA"), m_buildOptions.c_str()));
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output buffer: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // Intermediate buffer for predictor-corrector first pass. Skip
    // allocation when predictor=off to save the memory.
    if (prm->deflicker.predictor) {
        if (!m_intermediate || cmpFrameInfoCspResolution(&m_intermediate->frame, &prm->frameOut)) {
            m_intermediate = m_cl->createFrameBuffer(prm->frameOut, CL_MEM_READ_WRITE);
            if (!m_intermediate) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate deflicker intermediate buffer.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
    } else {
        m_intermediate.reset();
    }

    // Partial-sums buffers. Sized for the luma plane's workgroup count;
    // chroma planes are smaller and reuse the same buffers (the unused
    // tail entries just stay at whatever the previous launch wrote).
    const int wg_x = (prm->frameOut.width  + DEFLICKER_REDUCE_X - 1) / DEFLICKER_REDUCE_X;
    const int wg_y = (prm->frameOut.height + DEFLICKER_REDUCE_Y - 1) / DEFLICKER_REDUCE_Y;
    const size_t wg_count = (size_t)wg_x * (size_t)wg_y;
    if (!m_sumBuf || m_statsBufWGCount != wg_count) {
        m_sumBuf   = m_cl->createBuffer(wg_count * sizeof(int64_t), CL_MEM_READ_WRITE);
        m_sumSqBuf = m_cl->createBuffer(wg_count * sizeof(int64_t), CL_MEM_READ_WRITE);
        if (!m_sumBuf || !m_sumSqBuf) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate deflicker stats buffers.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_sumHost  .assign(wg_count, 0);
        m_sumSqHost.assign(wg_count, 0);
        m_statsBufWGCount = wg_count;
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterDeflicker::computePlaneStats(const RGYFrameInfo *pPlane, double &meanOut, double &stddevOut,
                                               RGYOpenCLQueue &queue,
                                               const std::vector<RGYOpenCLEvent> &wait_events) {
    const char *kernel_name = "deflicker_reduce";
    const int wg_x = (pPlane->width  + DEFLICKER_REDUCE_X - 1) / DEFLICKER_REDUCE_X;
    const int wg_y = (pPlane->height + DEFLICKER_REDUCE_Y - 1) / DEFLICKER_REDUCE_Y;
    const size_t wg_count = (size_t)wg_x * (size_t)wg_y;
    const size_t bytes_used = wg_count * sizeof(int64_t);

    RGYWorkSize local (DEFLICKER_REDUCE_X, DEFLICKER_REDUCE_Y);
    RGYWorkSize global(wg_x * DEFLICKER_REDUCE_X, wg_y * DEFLICKER_REDUCE_Y);

    auto err = m_deflicker.get()->kernel(kernel_name).config(queue, local, global, wait_events, nullptr).launch(
        (cl_mem)pPlane->ptr[0], pPlane->pitch[0],
        pPlane->width, pPlane->height,
        m_sumBuf->mem(), m_sumSqBuf->mem());
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"),
            char_to_tstring(kernel_name).c_str(), get_err_mes(err));
        return err;
    }

    // Synchronous readback of both partial-sums buffers.
    RGYOpenCLEvent readEvent1, readEvent2;
    cl_int clerr;
    clerr = clEnqueueReadBuffer(queue.get(), m_sumBuf->mem(),   CL_FALSE, 0, bytes_used,
                                m_sumHost.data(),   0, nullptr, readEvent1.reset_ptr());
    if (clerr != CL_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("clEnqueueReadBuffer (deflicker sum) failed: %s.\n"), cl_errmes(clerr));
        return err_cl_to_rgy(clerr);
    }
    clerr = clEnqueueReadBuffer(queue.get(), m_sumSqBuf->mem(), CL_FALSE, 0, bytes_used,
                                m_sumSqHost.data(), 0, nullptr, readEvent2.reset_ptr());
    if (clerr != CL_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("clEnqueueReadBuffer (deflicker sumSq) failed: %s.\n"), cl_errmes(clerr));
        return err_cl_to_rgy(clerr);
    }
    readEvent1.wait();
    readEvent2.wait();

    int64_t sum   = 0;
    int64_t sumSq = 0;
    for (size_t i = 0; i < wg_count; i++) {
        sum   += m_sumHost  [i];
        sumSq += m_sumSqHost[i];
    }
    const double n = (double)((int64_t)pPlane->width * (int64_t)pPlane->height);
    const double mean = (double)sum / n;
    // population variance: E[X^2] - (E[X])^2; clamp at 0 to avoid sqrt
    // of a negative from floating-point round-off on near-flat planes.
    double variance = ((double)sumSq / n) - (mean * mean);
    if (variance < 0.0) variance = 0.0;
    const double stddev = std::sqrt(variance);

    meanOut   = mean;
    stddevOut = stddev;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDeflicker::runApply(RGYFrameInfo *pDstPlane, const RGYFrameInfo *pSrcPlane,
                                      float mult, float add, float blend, int is_chroma,
                                      RGYOpenCLQueue &queue,
                                      const std::vector<RGYOpenCLEvent> &wait_events,
                                      RGYOpenCLEvent *event) {
    const char *kernel_name = "deflicker_apply";
    RGYWorkSize local(DEFLICKER_REDUCE_X, DEFLICKER_REDUCE_Y);
    RGYWorkSize global(pDstPlane->width, pDstPlane->height);
    auto err = m_deflicker.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pSrcPlane->ptr[0], pSrcPlane->pitch[0],
        (cl_mem)pDstPlane->ptr[0], pDstPlane->pitch[0],
        pDstPlane->width, pDstPlane->height,
        mult, add, blend, is_chroma);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"),
            char_to_tstring(kernel_name).c_str(), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDeflicker::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue_main,
    const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    *pOutputFrameNum  = 0;
    ppOutputFrames[0] = nullptr;
    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamDeflicker>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!m_deflicker.get()) {
        AddMessage(RGY_LOG_ERROR, _T("deflicker OpenCL program failed to build (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }
    const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, m_frameBuf[0]->frame.mem_type);
    if (memcpyKind != RGYCLMemcpyD2D) {
        AddMessage(RGY_LOG_ERROR, _T("deflicker only supports device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("deflicker does not support csp conversion.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    const int bitDepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
    const double sigma_eps = DEFLICKER_SIGMA_EPS_8BIT * (double)(1 << std::max(0, bitDepth - 8));

    RGYFrameInfo *pOut = &m_frameBuf[0]->frame;
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    RGYOpenCLEvent *lumaEvent = (planes <= 1) ? event : nullptr;

    // ---- Stage 1: stats on input luma ----
    const auto planeSrcY = getPlane(pInputFrame, RGY_PLANE_Y);
    double mean_in = 0.0, sigma_in = 0.0;
    auto err = computePlaneStats(&planeSrcY, mean_in, sigma_in, queue_main, wait_events);
    if (err != RGY_ERR_NONE) return err;

    // ---- Stage 2: scene-change detection + rolling buffer update ----
    //
    // The scene-change criterion combines a z-score breach with an
    // absolute-luma floor. The z-score check (|diff| > N * std) catches
    // big jumps relative to recent variance. The absolute floor
    // (|diff| > ~10% of max_val) ensures that flicker-target content
    // doesn't trip the gate: archival flicker is typically 5-20 luma
    // units (8-bit) of frame-to-frame mean drift, while real scene cuts
    // are typically 40-80 units. The z-score alone scales with the
    // very flicker we are trying to correct, so on heavily-flickering
    // sources (<archival film maker> film, old VHS transfers, PAL/NTSC conversion
    // artefacts) the std grows with the flicker and the gate triggers
    // on every frame -> rolling buffer never updates -> all subsequent
    // frames pass through unmodified. The absolute floor breaks that
    // self-defeating feedback by requiring a real luma jump regardless
    // of how noisy the rolling baseline is.
    bool sceneChange = false;
    if (m_rollingMeans.size() >= 5) {
        // Compute rolling mean of means and stddev of means.
        double sumM = 0.0;
        for (double v : m_rollingMeans) sumM += v;
        const double rollMeanOfMeans = sumM / (double)m_rollingMeans.size();
        double sumSqDev = 0.0;
        for (double v : m_rollingMeans) {
            const double d = v - rollMeanOfMeans;
            sumSqDev += d * d;
        }
        const double rollStdOfMeans = std::sqrt(sumSqDev / (double)m_rollingMeans.size());
        const double absDiff = std::abs(mean_in - rollMeanOfMeans);
        const double absFloor = 0.10 * (double)((1 << bitDepth) - 1);
        if (rollStdOfMeans > 0.0
            && absDiff > (double)prm->deflicker.scene_threshold * rollStdOfMeans
            && absDiff > absFloor) {
            sceneChange = true;
            m_skippedSceneFrames++;
        }
    }

    if (!sceneChange) {
        m_rollingMeans.push_back(mean_in);
        m_rollingSigmas.push_back(sigma_in);
        while ((int)m_rollingMeans.size() > prm->deflicker.frames) {
            m_rollingMeans.pop_front();
            m_rollingSigmas.pop_front();
        }
    }

    // ---- Stage 3: derive correction ----
    // Reference (mu_ref, sigma_ref) = mean of rolling window. With an
    // empty rolling buffer (first frame, scene change) the correction
    // collapses to identity and the output is the input.
    bool haveReference = !m_rollingMeans.empty();
    double mu_ref = mean_in;
    double sigma_ref = sigma_in;
    if (haveReference) {
        double sM = 0.0, sS = 0.0;
        for (double v : m_rollingMeans)  sM += v;
        for (double v : m_rollingSigmas) sS += v;
        mu_ref    = sM / (double)m_rollingMeans.size();
        sigma_ref = sS / (double)m_rollingSigmas.size();
    }

    double mult_raw = 1.0;
    double add_raw  = 0.0;
    if (haveReference && !sceneChange) {
        // sqrt-damped sigma ratio (van Roosmalen's formula stabilised
        // against over-correction on narrow-sigma frames).
        const double sigma_denom = std::max(sigma_in, sigma_eps);
        mult_raw = std::sqrt(sigma_ref / sigma_denom);
        add_raw  = mu_ref - mult_raw * mean_in;
    }

    // Damping against previous frame's (mult, add).
    double mult_eff = mult_raw;
    double add_eff  = add_raw;
    if (m_haveDamping && !sceneChange) {
        const double d = (double)prm->deflicker.damping;
        mult_eff = d * m_prevMult + (1.0 - d) * mult_raw;
        add_eff  = d * m_prevAdd  + (1.0 - d) * add_raw;
    }

    // ---- Stage 4: apply ----
    auto planeDstY = getPlane(pOut, RGY_PLANE_Y);
    if (sceneChange || !haveReference) {
        // Pass through: copy input -> output unmodified. Damping state
        // is preserved so the next valid frame resumes smoothly.
        err = m_cl->copyPlane(&planeDstY, &planeSrcY, nullptr, queue_main, {}, lumaEvent);
        if (err != RGY_ERR_NONE) return err;
    } else if (!prm->deflicker.predictor) {
        // One-pass mode: apply (mult_eff, add_eff) with strength blend.
        err = runApply(&planeDstY, &planeSrcY,
                       (float)mult_eff, (float)add_eff,
                       prm->deflicker.strength, /*is_chroma=*/0,
                       queue_main, {}, lumaEvent);
        if (err != RGY_ERR_NONE) return err;
    } else {
        // Predictor-corrector path. Two iterations:
        //   1) apply (mult_eff, add_eff) at full strength to m_intermediate
        //   2) recompute stats on intermediate; refine; apply refined to
        //      output with the user-facing strength blend.
        auto planeInter = getPlane(&m_intermediate->frame, RGY_PLANE_Y);
        err = runApply(&planeInter, &planeSrcY,
                       (float)mult_eff, (float)add_eff,
                       1.0f, /*is_chroma=*/0,
                       queue_main, {}, nullptr);
        if (err != RGY_ERR_NONE) return err;

        double mu_1 = 0.0, sigma_1 = 0.0;
        err = computePlaneStats(&planeInter, mu_1, sigma_1, queue_main, {});
        if (err != RGY_ERR_NONE) return err;

        const double sigma_denom2 = std::max(sigma_1, sigma_eps);
        const double mult_refine  = std::sqrt(sigma_ref / sigma_denom2);
        const double add_refine   = mu_ref - mult_refine * mu_1;
        err = runApply(&planeDstY, &planeInter,
                       (float)mult_refine, (float)add_refine,
                       prm->deflicker.strength, /*is_chroma=*/0,
                       queue_main, {}, lumaEvent);
        if (err != RGY_ERR_NONE) return err;
    }

    // Persist effective damping state for next frame (only when correction
    // was actually applied -- not on pass-through/scene-change frames).
    if (!sceneChange && haveReference) {
        m_prevMult    = mult_eff;
        m_prevAdd     = add_eff;
        m_haveDamping = true;
    }

    // ---- Stage 5: chroma ----
    // chroma=false: passthrough copy. chroma=true: apply the same
    // (mult_eff, add_eff but centred form -- the kernel handles centring
    // when is_chroma != 0; add_eff is ignored on that path). Scene-
    // change frames pass through chroma unmodified to match luma.
    for (int i = 1; i < planes; i++) {
        RGYOpenCLEvent *planeEvent = (i == planes - 1) ? event : nullptr;
        const auto planeSrcC = getPlane(pInputFrame, (RGY_PLANE)i);
        auto       planeDstC = getPlane(pOut, (RGY_PLANE)i);
        if (sceneChange || !haveReference || !prm->deflicker.chroma) {
            err = m_cl->copyPlane(&planeDstC, &planeSrcC, nullptr, queue_main, {}, planeEvent);
            if (err != RGY_ERR_NONE) return err;
        } else {
            // Use the effective mult (post-damping) consistent with luma.
            // is_chroma=1 selects centred-multiplicative in the kernel.
            err = runApply(&planeDstC, &planeSrcC,
                           (float)mult_eff, 0.0f,
                           prm->deflicker.strength, /*is_chroma=*/1,
                           queue_main, {}, planeEvent);
            if (err != RGY_ERR_NONE) return err;
        }
    }

    if (sceneChange) {
        AddMessage(RGY_LOG_DEBUG,
            _T("deflicker: scene change at frame %d (mean=%.1f vs rolling=%.1f); passthrough\n"),
            pInputFrame->inputFrameId, mean_in, m_rollingMeans.empty() ? 0.0 : m_rollingMeans.back());
    } else if (haveReference) {
        AddMessage(RGY_LOG_DEBUG,
            _T("deflicker: frame %d mu_in=%.2f sigma_in=%.2f mult=%.4f add=%.2f\n"),
            pInputFrame->inputFrameId, mean_in, sigma_in, mult_eff, add_eff);
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

void RGYFilterDeflicker::close() {
    if (m_skippedSceneFrames > 0) {
        AddMessage(RGY_LOG_INFO, _T("deflicker: skipped %d scene-change frames during analysis.\n"),
            m_skippedSceneFrames);
    }
    m_intermediate.reset();
    m_sumBuf.reset();
    m_sumSqBuf.reset();
    m_sumHost.clear();
    m_sumSqHost.clear();
    m_statsBufWGCount = 0;
    m_rollingMeans.clear();
    m_rollingSigmas.clear();
    m_prevMult = 1.0;
    m_prevAdd  = 0.0;
    m_haveDamping = false;
    m_skippedSceneFrames = 0;
    m_deflicker.clear();
    m_buildOptions.clear();
    m_frameBuf.clear();
    m_cl.reset();
}
