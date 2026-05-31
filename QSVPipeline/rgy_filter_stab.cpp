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
// Phase-correlation stabilisation based on:
// Kuglin & Hines, "The Phase Correlation Image Alignment Method",
// Proc. IEEE Int. Conf. Cybernetics and Society, 1975.
// Clean-room GPU implementation.

#include <algorithm>
#include <cmath>
#include "convert_csp.h"
#include "rgy_filter_stab.h"

// FFT size: whole-frame 2D FFT at 256x256 on the downsampled luma plane.
// 256 is large enough that sub-pixel refinement recovers the precision
// lost to downsampling, and small enough that one 1D FFT fits in 2KB of
// __local memory (256 * sizeof(float2)). FFT_LOG2_N must match -- the
// .cl uses an 8-bit bit-reversal trick that assumes FFT_N == 256.
static const int STAB_FFT_N      = 256;
static const int STAB_FFT_LOG2_N = 8;

// 3-point parabolic interpolation: returns the offset in [-0.5, +0.5]
// of the parabola maximum given samples at integer positions -1, 0, +1.
static float stab_parabolic_refine(float a, float b, float c) {
    const float denom = (a - 2.0f * b + c);
    if (std::abs(denom) < 1e-9f) return 0.0f;
    const float frac = 0.5f * (a - c) / denom;
    if (frac < -0.5f) return -0.5f;
    if (frac >  0.5f) return  0.5f;
    return frac;
}

RGYFilterStab::RGYFilterStab(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_stab(),
    m_buildOptions(),
    m_srcReal(),
    m_curFreq(),
    m_prevFreq(),
    m_corrFreq(),
    m_corrReal(),
    m_corrHost(),
    m_havePrev(false),
    m_smoothShiftX(0.0f),
    m_smoothShiftY(0.0f),
    m_haveSmoothing(false),
    m_lowTrustFrames(0) {
    m_name = _T("stab");
}

RGYFilterStab::~RGYFilterStab() {
    close();
}

RGY_ERR RGYFilterStab::checkParam(const std::shared_ptr<RGYFilterParamStab> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->stab.strength < 0.0f || 1.0f < prm->stab.strength) {
        prm->stab.strength = clamp(prm->stab.strength, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("strength should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }
    if (prm->stab.damping < 0.0f || 1.0f < prm->stab.damping) {
        prm->stab.damping = clamp(prm->stab.damping, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("damping should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }
    if (prm->stab.trust_threshold < 0.0f || 1.0f < prm->stab.trust_threshold) {
        prm->stab.trust_threshold = clamp(prm->stab.trust_threshold, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("trust should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }
    if (prm->stab.max_shift < 1.0f || 256.0f < prm->stab.max_shift) {
        prm->stab.max_shift = clamp(prm->stab.max_shift, 1.0f, 256.0f);
        AddMessage(RGY_LOG_WARN, _T("max_shift should be in range of %.1f - %.1f px.\n"), 1.0f, 256.0f);
    }
    if (prm->stab.border < VPP_STAB_BORDER_BLACK || prm->stab.border > VPP_STAB_BORDER_MIRROR) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid border mode: %d.\n"), prm->stab.border);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterStab::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamStab>(pParam);
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

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamStab>(m_param);
    const bool rebuild = !m_stab.get()
                      || !prmPrev
                      || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp];
    if (rebuild) {
        m_buildOptions = strsprintf(
            "-D Type=%s -D bit_depth=%d -D max_val=%d -D FFT_N=%d -D FFT_LOG2_N=%d",
            bitDepth > 8 ? "ushort" : "uchar",
            bitDepth, maxVal,
            STAB_FFT_N, STAB_FFT_LOG2_N);
        AddMessage(RGY_LOG_DEBUG, _T("Starting async build for RGY_FILTER_STAB_CL: %s\n"),
            char_to_tstring(m_buildOptions).c_str());
        m_stab.set(m_cl->buildResourceAsync(_T("RGY_FILTER_STAB_CL"), _T("EXE_DATA"), m_buildOptions.c_str()));
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output buffer: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // Five 256x256 float2 buffers: source (downsampled luma packed as
    // complex with imag=0), current frame's forward FFT, previous frame's
    // forward FFT, normalised cross-spectrum, and the IFFT correlation
    // output. Total ~2.5 MB of VRAM, negligible.
    const size_t fftBytes = (size_t)STAB_FFT_N * (size_t)STAB_FFT_N * 2 * sizeof(float);
    auto ensureBuf = [&](std::unique_ptr<RGYCLBuf> &buf, const TCHAR *name) -> RGY_ERR {
        if (buf && buf->size() == fftBytes) return RGY_ERR_NONE;
        buf = m_cl->createBuffer(fftBytes, CL_MEM_READ_WRITE);
        if (!buf) {
            AddMessage(RGY_LOG_ERROR, _T("stab: failed to allocate %s buffer (%zu bytes).\n"), name, fftBytes);
            return RGY_ERR_MEMORY_ALLOC;
        }
        return RGY_ERR_NONE;
    };
    if ((sts = ensureBuf(m_srcReal,  _T("srcReal")))  != RGY_ERR_NONE) return sts;
    if ((sts = ensureBuf(m_curFreq,  _T("curFreq")))  != RGY_ERR_NONE) return sts;
    if ((sts = ensureBuf(m_prevFreq, _T("prevFreq"))) != RGY_ERR_NONE) return sts;
    if ((sts = ensureBuf(m_corrFreq, _T("corrFreq"))) != RGY_ERR_NONE) return sts;
    if ((sts = ensureBuf(m_corrReal, _T("corrReal"))) != RGY_ERR_NONE) return sts;

    m_corrHost.assign((size_t)STAB_FFT_N * (size_t)STAB_FFT_N * 2, 0.0f);

    // Reset per-clip state on (re-)init.
    m_havePrev       = false;
    m_smoothShiftX   = 0.0f;
    m_smoothShiftY   = 0.0f;
    m_haveSmoothing  = false;
    m_lowTrustFrames = 0;

    AddMessage(RGY_LOG_DEBUG, _T("stab: FFT %dx%d, source %dx%d.\n"),
        STAB_FFT_N, STAB_FFT_N, prm->frameOut.width, prm->frameOut.height);
    AddMessage(RGY_LOG_INFO,
        _T("stab: phase correlation initialised (FFT_N=%d, trust=%.2f, damping=%.2f).\n"),
        STAB_FFT_N, prm->stab.trust_threshold, prm->stab.damping);

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterStab::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue_main,
    const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    *pOutputFrameNum  = 0;
    ppOutputFrames[0] = nullptr;
    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamStab>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!m_stab.get()) {
        AddMessage(RGY_LOG_ERROR, _T("stab OpenCL program failed to build (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }

    RGYFrameInfo *pOut = &m_frameBuf[0]->frame;
    const auto planeSrcY = getPlane(pInputFrame, RGY_PLANE_Y);

    // ---- Stage 1: downsample luma to STAB_FFT_N x STAB_FFT_N complex buffer.
    {
        const char *kname = "stab_luma_downsample";
        RGYWorkSize local (32, 8);
        RGYWorkSize global((STAB_FFT_N + 31) / 32 * 32, (STAB_FFT_N + 7) / 8 * 8);
        auto err = m_stab.get()->kernel(kname).config(queue_main, local, global, wait_events, nullptr).launch(
            (cl_mem)planeSrcY.ptr[0], planeSrcY.pitch[0],
            planeSrcY.width, planeSrcY.height,
            m_srcReal->mem());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("stab: %s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }

    // ---- Stage 2: forward 2D FFT of current frame.
    // 2a: row FFT (stride 1).
    // 2b: column FFT in-place (stride FFT_N) -- safe because each
    //     workgroup owns one column and reads all of it into SLM before
    //     writing back.
    auto launchFFT1D = [&](cl_mem in, cl_mem out, int stride, float direction) -> RGY_ERR {
        const char *kname = "stab_fft_1d";
        RGYWorkSize local (STAB_FFT_N / 2);
        RGYWorkSize global((size_t)STAB_FFT_N * (size_t)(STAB_FFT_N / 2));
        return m_stab.get()->kernel(kname).config(queue_main, local, global, {}, nullptr).launch(
            in, out, stride, direction);
    };

    auto err = launchFFT1D(m_srcReal->mem(),  m_curFreq->mem(), 1,         +1.0f); // row fwd
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("stab: row FFT failed: %s.\n"), get_err_mes(err)); return err; }
    err      = launchFFT1D(m_curFreq->mem(),  m_curFreq->mem(), STAB_FFT_N, +1.0f); // col fwd (in-place)
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("stab: col FFT failed: %s.\n"), get_err_mes(err)); return err; }

    // ---- Stage 3: if we have a previous spectrum, run phase correlation.
    bool  haveCorrelation = false;
    int   peakX = 0, peakY = 0;
    float peakValue = 0.0f, meanValue = 1.0f;
    float refineX = 0.0f, refineY = 0.0f;

    if (m_havePrev) {
        // 3a: cross-spectrum.
        {
            const char *kname = "stab_cross_spectrum";
            const int total = STAB_FFT_N * STAB_FFT_N;
            RGYWorkSize local (256);
            RGYWorkSize global((total + 255) / 256 * 256);
            err = m_stab.get()->kernel(kname).config(queue_main, local, global, {}, nullptr).launch(
                m_curFreq->mem(), m_prevFreq->mem(), m_corrFreq->mem(), total);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("stab: %s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        // 3b: inverse 2D FFT (rows then columns).
        err = launchFFT1D(m_corrFreq->mem(), m_corrReal->mem(), 1,         -1.0f);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("stab: row IFFT failed: %s.\n"), get_err_mes(err)); return err; }
        err = launchFFT1D(m_corrReal->mem(), m_corrReal->mem(), STAB_FFT_N, -1.0f);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("stab: col IFFT failed: %s.\n"), get_err_mes(err)); return err; }

        // 3c: synchronous readback of the correlation plane (float2,
        // ~512 KB on a 256x256 FFT). The peak scan runs on the CPU.
        {
            const size_t bytes = (size_t)STAB_FFT_N * (size_t)STAB_FFT_N * 2 * sizeof(float);
            RGYOpenCLEvent readEvent;
            cl_int clerr = clEnqueueReadBuffer(queue_main.get(), m_corrReal->mem(),
                                               CL_FALSE, 0, bytes,
                                               m_corrHost.data(), 0, nullptr, readEvent.reset_ptr());
            if (clerr != CL_SUCCESS) {
                AddMessage(RGY_LOG_ERROR, _T("clEnqueueReadBuffer (stab corr) failed: %s.\n"), cl_errmes(clerr));
                return err_cl_to_rgy(clerr);
            }
            readEvent.wait();
        }

        // 3d: scan for the peak in the real part. Phase correlation
        // output is approximately real -- the imaginary part comes from
        // arithmetic error in the FFT/IFFT chain.
        const int N = STAB_FFT_N;
        float pk = m_corrHost[0]; // .x of float2 element 0
        int   pi = 0;
        double sum = 0.0;
        for (int i = 0; i < N * N; i++) {
            const float v = m_corrHost[(size_t)i * 2]; // .x of float2 element i
            sum += (double)v;
            if (v > pk) { pk = v; pi = i; }
        }
        peakX = pi % N;
        peakY = pi / N;
        peakValue = pk;
        meanValue = (float)(sum / (double)(N * N));

        // Sub-pixel parabolic refinement around the integer peak, with
        // wrap-around so peaks near the edges still get refined.
        auto getPx = [&](int x, int y) -> float {
            int xx = x;
            int yy = y;
            if (xx < 0) xx += N; else if (xx >= N) xx -= N;
            if (yy < 0) yy += N; else if (yy >= N) yy -= N;
            return m_corrHost[((size_t)yy * (size_t)N + (size_t)xx) * 2];
        };
        refineX = stab_parabolic_refine(getPx(peakX - 1, peakY), getPx(peakX, peakY), getPx(peakX + 1, peakY));
        refineY = stab_parabolic_refine(getPx(peakX, peakY - 1), getPx(peakX, peakY), getPx(peakX, peakY + 1));

        haveCorrelation = true;
    }

    // ---- Stage 4: convert peak position to a signed shift (centre-origin)
    //               and apply smoothing / trust gate.
    float rawShiftX = 0.0f, rawShiftY = 0.0f, trust = 0.0f;
    if (haveCorrelation) {
        const int N = STAB_FFT_N;
        // Phase correlation wraps cyclically: a peak in the upper half
        // represents a negative shift on that axis. Convert to signed.
        int sx = peakX;
        int sy = peakY;
        if (sx > N / 2) sx -= N;
        if (sy > N / 2) sy -= N;
        const float shiftFftX = (float)sx + refineX;
        const float shiftFftY = (float)sy + refineY;
        // Scale from FFT-pixel units back to source-pixel units.
        const float scaleX = (float)prm->frameOut.width  * (1.0f / (float)N);
        const float scaleY = (float)prm->frameOut.height * (1.0f / (float)N);
        rawShiftX = shiftFftX * scaleX;
        rawShiftY = shiftFftY * scaleY;

        // Trust: peak height vs. correlation surface mean. A sharp single-
        // spike peak gives trust >> 1.0; a noisy surface (scene change,
        // large motion blur) gives trust near 1.0.
        trust = (meanValue > 1e-12f) ? (peakValue / meanValue) : 0.0f;
        // Map into [0, 1) for comparison against the user-facing threshold.
        const float trustNorm = trust / (trust + 100.0f);

        const float maxShift = prm->stab.max_shift;
        if (rawShiftX >  maxShift) rawShiftX =  maxShift;
        if (rawShiftX < -maxShift) rawShiftX = -maxShift;
        if (rawShiftY >  maxShift) rawShiftY =  maxShift;
        if (rawShiftY < -maxShift) rawShiftY = -maxShift;

        if (trustNorm < prm->stab.trust_threshold) {
            m_lowTrustFrames++;
            AddMessage(RGY_LOG_DEBUG,
                _T("stab: frame %d low-trust (trust=%.3f norm=%.3f); keeping previous smoothed shift.\n"),
                pInputFrame->inputFrameId, trust, trustNorm);
        } else if (!m_haveSmoothing) {
            m_smoothShiftX  = rawShiftX;
            m_smoothShiftY  = rawShiftY;
            m_haveSmoothing = true;
        } else {
            const float d = prm->stab.damping;
            m_smoothShiftX = d * m_smoothShiftX + (1.0f - d) * rawShiftX;
            m_smoothShiftY = d * m_smoothShiftY + (1.0f - d) * rawShiftY;
        }
    }

    // ---- Stage 5: warp the input frame by the (strength-scaled) smoothed
    // shift. The gate is m_haveSmoothing: false only on the literal first
    // frame, or on a clip where every observed frame so far has been low-
    // trust. Once a non-low-trust frame has set the initial smoothed
    // value, every subsequent frame -- including low-trust frames --
    // warps by whatever m_smoothShiftX/Y currently holds. Low-trust
    // frames simply did not advance the smoothing this iteration; they
    // continue applying the most recent good trajectory, which avoids
    // visible discontinuities mid-clip.
    const bool apply_warp = m_haveSmoothing;
    if (apply_warp) {
        const int planes   = RGY_CSP_PLANES[pInputFrame->csp];
        const int bitDepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
        // Chroma subsampling factors derived from the actual plane sizes
        // (same idiom as chromashift). 4:2:0 -> (2,2); 4:2:2 -> (2,1);
        // 4:4:4 -> (1,1).
        const auto pY = getPlane(pInputFrame, RGY_PLANE_Y);
        int subX = 1, subY = 1;
        if (planes >= 2) {
            const auto pU = getPlane(pInputFrame, RGY_PLANE_U);
            if (pU.width  > 0) subX = std::max(1, pY.width  / pU.width);
            if (pU.height > 0) subY = std::max(1, pY.height / pU.height);
        }
        const float effShiftX  = prm->stab.strength * m_smoothShiftX;
        const float effShiftY  = prm->stab.strength * m_smoothShiftY;
        const int   chromaFill = 1 << (bitDepth - 1);

        for (int i = 0; i < planes; i++) {
            const auto src = getPlane(pInputFrame, (RGY_PLANE)i);
            auto       dst = getPlane(pOut,        (RGY_PLANE)i);
            const float planeShiftX = (i == 0) ? effShiftX : (effShiftX / (float)subX);
            const float planeShiftY = (i == 0) ? effShiftY : (effShiftY / (float)subY);
            const int   fillValue   = (i == 0) ? 0 : chromaFill;

            const char *kname = "stab_warp";
            RGYWorkSize local (32, 8);
            RGYWorkSize global((src.width + 31) / 32 * 32, (src.height + 7) / 8 * 8);
            err = m_stab.get()->kernel(kname).config(queue_main, local, global, {}, (i == planes - 1) ? event : nullptr).launch(
                (cl_mem)src.ptr[0], src.pitch[0],
                (cl_mem)dst.ptr[0], dst.pitch[0],
                src.width, src.height,
                planeShiftX, planeShiftY,
                prm->stab.border, fillValue);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("stab: %s plane %d failed: %s.\n"),
                    char_to_tstring(kname).c_str(), i, get_err_mes(err));
                return err;
            }
        }
    } else {
        // No usable smoothed shift yet. Pass through unchanged.
        err = m_cl->copyFrame(pOut, pInputFrame, nullptr, queue_main, {}, event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("stab: passthrough copy failed: %s.\n"), get_err_mes(err));
            return err;
        }
    }

    // ---- Stage 6: promote current spectrum to "previous" for the next
    //               frame's correlation. Swap the unique_ptrs so the old
    //               m_prevFreq slot becomes the next scratch buffer.
    std::swap(m_curFreq, m_prevFreq);
    m_havePrev = true;

    if (haveCorrelation) {
        AddMessage(RGY_LOG_DEBUG,
            _T("stab: frame %d raw=(%+.2f,%+.2f) smooth=(%+.2f,%+.2f) trust=%.2f %s.\n"),
            pInputFrame->inputFrameId,
            rawShiftX, rawShiftY,
            m_smoothShiftX, m_smoothShiftY,
            trust,
            apply_warp ? _T("[warp]") : _T("[passthrough]"));
    } else {
        AddMessage(RGY_LOG_DEBUG, _T("stab: frame %d seeded reference spectrum.\n"),
            pInputFrame->inputFrameId);
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

void RGYFilterStab::close() {
    if (m_lowTrustFrames > 0) {
        AddMessage(RGY_LOG_INFO, _T("stab: rejected %d low-trust frames during analysis.\n"),
            m_lowTrustFrames);
    }
    m_srcReal.reset();
    m_curFreq.reset();
    m_prevFreq.reset();
    m_corrFreq.reset();
    m_corrReal.reset();
    m_corrHost.clear();
    m_havePrev = false;
    m_smoothShiftX = 0.0f;
    m_smoothShiftY = 0.0f;
    m_haveSmoothing = false;
    m_lowTrustFrames = 0;
    m_stab.clear();
    m_buildOptions.clear();
    m_frameBuf.clear();
    m_cl.reset();
}
