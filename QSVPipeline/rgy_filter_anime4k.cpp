// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
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

#include "rgy_filter_anime4k.h"
#include "rgy_filter_resize.h" // opt-in end-of-chain resize (anime4k.out_res=/resize=)
#include "rgy_aspect_ratio.h"  // set_auto_resolution() for out_res= negative auto-aspect

#include <fstream>
#include <cstring>

static const int ANIME4K_BLOCK_X = 16;
static const int ANIME4K_BLOCK_Y = 16;

RGYFilterAnime4k::RGYFilterAnime4k(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context), m_anime4k(), m_scratchA(), m_scratchB(),
    m_scratchPitchFloats(0), m_outW(0), m_outH(0),
    m_srcImagePool(), m_frameIdx(0),
    m_fp16Scratch(false),
    m_dtdSrcLuma(), m_dtdSrcLumaPitch(0), m_dtdSrcW(0), m_dtdSrcH(0),
    m_chromaLumaLowres(), m_chromaLowresPitch(0), m_chromaLowresW(0), m_chromaLowresH(0),
    m_prefilterPlane(), m_prefilterPlanePitch(0),
    m_prefilterRef(), m_prefilterRefPitchF4(0),
    m_clampStatsMaxH(), m_clampStatsMax(), m_clampStatsStride(0) {
    m_name = _T("anime4k");
}

RGYFilterAnime4k::~RGYFilterAnime4k() {
    close();
}

RGY_ERR RGYFilterAnime4k::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamAnime4k>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    // mode=darken_hq and mode=thin_hq are aliases for the composable
    // darken=hq / thin=hq post-process flags applied on top of the
    // original base. Set the flag here so dispatch picks it up; leave
    // prm->anime4k.mode untouched so print() shows the user-facing
    // alias name. Dispatch sites that test mode==Original also accept
    // DarkenHQ/ThinHQ as equivalents (see runPlaneY below).
    if (prm->anime4k.mode == VppAnime4kMode::DarkenHQ) {
        prm->anime4k.darken = VppAnime4kDarken::HQ;
    } else if (prm->anime4k.mode == VppAnime4kMode::ThinHQ) {
        prm->anime4k.thin = VppAnime4kThin::HQ;
    }
    // GLSL modes with a fixed scale: dog_sharpen is a 1x sharpener; dog and dtd
    // are 2x DoG upscales; original / deblur / darken_hq / thin_hq keep the
    // default (2x). Auto-correct and inform if the user passed a mismatch.
    const int impliedScale =
          (prm->anime4k.mode == VppAnime4kMode::DogSharpen) ? 1
        : (prm->anime4k.mode == VppAnime4kMode::Dog
           || prm->anime4k.mode == VppAnime4kMode::Dtd)     ? 2
        : 0;
    if (impliedScale != 0 && prm->anime4k.scale != impliedScale) {
        AddMessage(RGY_LOG_INFO, _T("%s implies scale=%d\n"),
            get_cx_desc(list_vpp_anime4k_mode, (int)prm->anime4k.mode),
            impliedScale);
        prm->anime4k.scale = impliedScale;
    }
    if (prm->anime4k.scale != 1 && prm->anime4k.scale != 2) {
        AddMessage(RGY_LOG_ERROR, _T("scale must be 1 or 2 (got %d).\n"), prm->anime4k.scale);
        return RGY_ERR_INVALID_PARAM;
    }

    // mode=dtd bakes its own darken / thin sub-pipelines with tuned-down
    // strengths (1.8 / 0.4); compounding user-supplied darken= / thin=
    // on top would double-apply with values the chain wasn't tuned for.
    // Reject up-front rather than silently produce a different result.
    if (prm->anime4k.mode == VppAnime4kMode::Dtd
     && (prm->anime4k.darken != VppAnime4kDarken::Off
      || prm->anime4k.thin   != VppAnime4kThin::Off)) {
        AddMessage(RGY_LOG_ERROR,
            _T("mode=dtd already includes darken and thin -- remove darken= and thin= flags.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    // denoise= on top of mode=dtd is structurally allowed (DTD's stage
    // C is a sharpen, not a denoise) but the upstream chain wasn't
    // tuned to compose with it. Log and disable to keep behaviour
    // predictable; users wanting denoise + DTD can compose denoise
    // after the encode externally if needed.
    if (prm->anime4k.mode == VppAnime4kMode::Dtd
     && prm->anime4k.denoise != VppAnime4kDenoise::Off) {
        AddMessage(RGY_LOG_INFO,
            _T("mode=dtd ignores denoise= (built-in chain handles sharpening only).\n"));
        prm->anime4k.denoise = VppAnime4kDenoise::Off;
    }
    if (prm->anime4k.strength < FILTER_ANIME4K_STRENGTH_MIN
     || prm->anime4k.strength > FILTER_ANIME4K_STRENGTH_MAX) {
        AddMessage(RGY_LOG_WARN, _T("strength clamped to [%.2f, %.2f] (got %.2f).\n"),
            FILTER_ANIME4K_STRENGTH_MIN, FILTER_ANIME4K_STRENGTH_MAX, prm->anime4k.strength);
        prm->anime4k.strength = clamp(prm->anime4k.strength, FILTER_ANIME4K_STRENGTH_MIN, FILTER_ANIME4K_STRENGTH_MAX);
    }

    const int srcW = prm->frameIn.width;
    const int srcH = prm->frameIn.height;
    const int outW = srcW * prm->anime4k.scale;
    const int outH = srcH * prm->anime4k.scale;
    m_outW = outW;
    m_outH = outH;

    // mode=deblur is Upscale_Original_x2 with REFINE_STRENGTH = 1.0
    // baked in. We expose the user's strength= as the multiplier for
    // both modes; deblur's higher default comes from the CLI default
    // promotion (handled in rgy_cmd.cpp), not from the kernel side.
    const float refineStrength = prm->anime4k.strength;

    // FP16 scratch is available when the device advertises cl_khr_fp16
    // AND the mode actually uses the darken / thin post-process chain
    // (mode=dtd has its own internal chain we leave on the FP32 path
    // to avoid double-converting the stage-C base scratches; CNN modes
    // route through runPlaneYCNN and never touch the post-process).
    // The darken / thin chain kernels then read / write half4 storage
    // via vload_half4 / vstore_half4 (OpenCL 1.2 core builtins, FP32
    // arithmetic internally), halving per-pixel scratch bandwidth.
    // Estimated ~25-35% wall-clock reduction at 4K on Arc A770. The base
    // chain (polynomial P5..P0 evaluation in m_scratchA / m_scratchB)
    // stays FP32 regardless -- the polynomial intermediates reach
    // magnitudes around 170 where FP16 absolute precision drops below
    // the 1/255 8-bit output quantisation step.
    // Codebase precedent: rgy_filter_nnedi.cpp:582 and
    // rgy_filter_denoise_fft3d.cpp:370 use the same checkExtension probe.
    const bool deviceSupportsFP16 =
        RGYOpenCLDevice(m_cl->queue().devid()).checkExtension("cl_khr_fp16");
    const bool modeAllowsFP16 = prm->anime4k.mode != VppAnime4kMode::Dtd;
    // We only enable FP16 scratch when every active tier runs at full
    // output resolution (i.e. HQ or Off). Fast / VeryFast tiers use the
    // existing FP32 m_darkenWork / m_thinWork structs and would need a
    // second program build to coexist with FP16 HQ in the same chain.
    // Restricting to HQ-only keeps the program-build set to one variant
    // and captures the biggest bandwidth win (HQ at 4K is the heaviest
    // path; ~25-35% wall-clock per the investigation report).
    const bool tierAllowsFP16 =
           (prm->anime4k.darken == VppAnime4kDarken::HQ
         || prm->anime4k.darken == VppAnime4kDarken::Off)
        && (prm->anime4k.thin   == VppAnime4kThin::HQ
         || prm->anime4k.thin   == VppAnime4kThin::Off);
    m_fp16Scratch = deviceSupportsFP16 && modeAllowsFP16 && tierAllowsFP16;
    if (m_fp16Scratch) {
        AddMessage(RGY_LOG_DEBUG, _T("FP16 scratch enabled (cl_khr_fp16 available).\n"));
    } else if (deviceSupportsFP16) {
        AddMessage(RGY_LOG_DEBUG, _T("FP16 scratch skipped (Fast / VeryFast / dtd / CNN tier in use); FP32 scratch.\n"));
    } else {
        AddMessage(RGY_LOG_DEBUG, _T("FP16 scratch disabled (cl_khr_fp16 unavailable); using FP32 fallback.\n"));
    }

    // Per-tier sigma coefficients baked into the OCL program at JIT
    // time. The kernels read ANIME4K_DARKEN_SIGMA_REF / THIN_SIGMA_REF
    // and multiply by outH/1080 at runtime; the coefficient selects
    // the tier. Off uses the HQ value as a placeholder (kernels never
    // run when the corresponding tier is Off).
    auto darkenSigmaCoef = [](VppAnime4kDarken d) {
        switch (d) {
        case VppAnime4kDarken::Fast:     return 0.5f;
        case VppAnime4kDarken::VeryFast: return 0.25f;
        case VppAnime4kDarken::HQ:
        case VppAnime4kDarken::Off:
        default:                         return 1.0f;
        }
    };
    auto thinSigmaCoef = [](VppAnime4kThin t) {
        switch (t) {
        case VppAnime4kThin::Fast:     return 1.0f;
        case VppAnime4kThin::VeryFast: return 0.5f;
        case VppAnime4kThin::HQ:
        case VppAnime4kThin::Off:
        default:                       return 2.0f;
        }
    };

    // Clamp denoise parameters and resolve the histReg sentinel before
    // the rebuild trigger so the values that drive both the rebuild
    // check and the JIT -D defines are the final, ready-to-use ones.
    // The published shader defaults are intensity=0.1, spatial=1.0,
    // curve=1.0, hist_reg=0.0 (mean/median) / 0.2 (mode); we expose
    // sensible bounds here.
    prm->anime4k.denoiseIntensity =
        clamp(prm->anime4k.denoiseIntensity, 0.001f, 1.0f);
    prm->anime4k.denoiseSpatial =
        clamp(prm->anime4k.denoiseSpatial, 0.5f, 2.5f);
    prm->anime4k.denoiseCurve =
        clamp(prm->anime4k.denoiseCurve, 0.0f, 2.0f);
    if (prm->anime4k.denoiseHistReg < 0.0f) {
        prm->anime4k.denoiseHistReg =
            (prm->anime4k.denoise == VppAnime4kDenoise::Mode) ? 0.2f : 0.0f;
    }
    prm->anime4k.denoiseHistReg =
        clamp(prm->anime4k.denoiseHistReg, 0.0f, 1.0f);

    // Kernel sizes follow the reference shader formulae, computed once
    // here and baked into the OCL program as -D defines. Mean uses a
    // wider kernel (ceil(s*2)*2+1) because its inner loop is O(K^2);
    // median / mode use a narrower kernel (int(s)*2+1) because their
    // inner loop is O(K^4).
    const float dnSpatial = prm->anime4k.denoiseSpatial;
    const int mean_khalf = std::max((int)std::ceil(dnSpatial * 2.0f), 1);
    const int mmm_khalf  = std::max((int)dnSpatial, 1);
    const int mmm_ksize  = mmm_khalf * 2 + 1;
    const int mmm_klen   = mmm_ksize * mmm_ksize;

    // Rebuild the kernel if any of the build-time defines change. The
    // darken / thin tier affects the sigma -D defines so it joins the
    // rebuild trigger list alongside scale / mode / strength. Denoise
    // parameters likewise drive -D defines (sigma values + kernel half
    // sizes) and need the same treatment.
    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamAnime4k>(m_param);
    const bool rebuild = !m_anime4k.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[prm->frameOut.csp]
        || prmPrev->anime4k.scale            != prm->anime4k.scale
        || prmPrev->anime4k.mode             != prm->anime4k.mode
        || prmPrev->anime4k.strength         != prm->anime4k.strength
        || prmPrev->anime4k.darken           != prm->anime4k.darken
        || prmPrev->anime4k.thin             != prm->anime4k.thin
        || prmPrev->anime4k.denoise          != prm->anime4k.denoise
        || prmPrev->anime4k.denoiseIntensity != prm->anime4k.denoiseIntensity
        || prmPrev->anime4k.denoiseSpatial   != prm->anime4k.denoiseSpatial
        || prmPrev->anime4k.denoiseCurve     != prm->anime4k.denoiseCurve
        || prmPrev->anime4k.denoiseHistReg   != prm->anime4k.denoiseHistReg;
    // Per-mode constants baked into the OCL program at JIT time. The
    // composite mode=dtd path tunes the darken / thin / deblur stages
    // away from the standalone published values; mode=dog uses the
    // Upscale_DoG_x2 strength; mode=dog_sharpen and anything else use
    // the Deblur_DoG defaults. Non-DoG modes never reach the DoG
    // kernels, but the defines still need real values for the program
    // to compile.
    const bool  modeIsDtd = (prm->anime4k.mode == VppAnime4kMode::Dtd);
    const float darkenStrength = modeIsDtd ? 1.8f : 1.5f;
    const float thinStrength   = modeIsDtd ? 0.4f : 0.6f;
    const float dogStrength =
          (prm->anime4k.mode == VppAnime4kMode::Dog) ? 0.8f
        : modeIsDtd                                  ? 0.5f
        :                                              0.6f;
    const float dogBlurCurve     = modeIsDtd ? 0.8f   : 0.6f;
    const float dogNoiseThreshold = modeIsDtd ? 0.004f : 0.001f;

    if (rebuild) {
        const auto options = strsprintf(
            "-D Type=%s -D bit_depth=%d -D SCALE=%d -D ANIME4K_REFINE_STRENGTH=%.6ff"
            " -D ANIME4K_DARKEN_SIGMA_REF=%.6ff -D ANIME4K_THIN_SIGMA_REF=%.6ff"
            " -D ANIME4K_DARKEN_STRENGTH=%.6ff -D ANIME4K_THIN_STRENGTH=%.6ff"
            " -D ANIME4K_DOG_STRENGTH=%.6ff"
            " -D ANIME4K_DOG_BLUR_CURVE=%.6ff"
            " -D ANIME4K_DOG_NOISE_THRESHOLD=%.6ff"
            " -D ANIME4K_DENOISE_SPATIAL_SIGMA=%.6ff"
            " -D ANIME4K_DENOISE_INTENSITY_SIGMA=%.6ff"
            " -D ANIME4K_DENOISE_INTENSITY_CURVE=%.6ff"
            " -D ANIME4K_DENOISE_HIST_REG=%.6ff"
            " -D ANIME4K_DENOISE_MEAN_KHALF=%d"
            " -D ANIME4K_DENOISE_MMM_KHALF=%d"
            " -D ANIME4K_DENOISE_MMM_KLEN=%d"
            " -D ANIME4K_SCRATCH_FP16=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp],
            prm->anime4k.scale,
            refineStrength,
            darkenSigmaCoef(prm->anime4k.darken),
            thinSigmaCoef(prm->anime4k.thin),
            darkenStrength,
            thinStrength,
            dogStrength,
            dogBlurCurve,
            dogNoiseThreshold,
            prm->anime4k.denoiseSpatial,
            prm->anime4k.denoiseIntensity,
            prm->anime4k.denoiseCurve,
            prm->anime4k.denoiseHistReg,
            mean_khalf,
            mmm_khalf,
            mmm_klen,
            m_fp16Scratch ? 1 : 0);
        m_anime4k.set(m_cl->buildResourceAsync(_T("RGY_FILTER_ANIME4K_CL"), _T("EXE_DATA"), options.c_str()));
    }

    // Allocate output frame buffer at scaled dimensions. Chroma planes
    // follow automatically via the csp planar layout (U/V at half luma
    // resolution under YUV420 after scaling).
    RGYFrameInfo frameOut = prm->frameOut;
    frameOut.width = outW;
    frameOut.height = outH;
    prm->frameOut = frameOut;
    auto err = AllocFrameBuf(prm->frameOut, 2);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to allocate output frame buffer: %s.\n"), get_err_mes(err));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // Two float4 scratches sized to the output luma plane. .xy carries
    // sobel partial / gradient direction; .z carries dval propagated
    // forward through the chain so apply() does not need a separate
    // LUMAD buffer. .w is reserved (always zero).
    {
        m_scratchPitchFloats = outW * 4;
        const size_t scratchBytes = (size_t)outW * outH * 4 * sizeof(float);
        m_scratchA = m_cl->createBuffer(scratchBytes, CL_MEM_READ_WRITE);
        m_scratchB = m_cl->createBuffer(scratchBytes, CL_MEM_READ_WRITE);
        if (!m_scratchA || !m_scratchB) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate scratch buffers (%zu bytes each).\n"), scratchBytes);
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    // Downsampled scratches for darken / thin Fast and VeryFast tiers.
    // Sized to half / quarter of the output luma plane in each axis.
    // HQ tiers run at full output res and reuse m_scratchA / m_scratchB
    // above, so these allocations are skipped for HQ and Off.
    auto tierDiv = [](int tier) {
        // 2 for Fast, 4 for VeryFast, 0 for Off / HQ (no allocation)
        return (tier == (int)VppAnime4kDarken::Fast)     ? 2
             : (tier == (int)VppAnime4kDarken::VeryFast) ? 4
             : 0;
    };
    const int typeBytes = (RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8) ? 2 : 1;
    auto allocWork = [&](Anime4kDownscaledScratches &w, int div, const TCHAR *label) -> RGY_ERR {
        if (div <= 0) return RGY_ERR_NONE;
        w.workW = (outW + div - 1) / div;
        w.workH = (outH + div - 1) / div;
        w.pitchFloats = w.workW * 4;
        w.lumaPitch = w.workW * typeBytes;
        const size_t bytesF4 = (size_t)w.workW * w.workH * 4 * sizeof(float);
        const size_t bytesY  = (size_t)w.workH * w.lumaPitch;
        w.A = m_cl->createBuffer(bytesF4, CL_MEM_READ_WRITE);
        w.B = m_cl->createBuffer(bytesF4, CL_MEM_READ_WRITE);
        w.luma = m_cl->createBuffer(bytesY, CL_MEM_READ_WRITE);
        if (!w.A || !w.B || !w.luma) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate %s work scratches (%dx%d).\n"),
                label, w.workW, w.workH);
            return RGY_ERR_MEMORY_ALLOC;
        }
        return RGY_ERR_NONE;
    };
    // FP16 scratch allocator. Same shape as allocWork but with half4
    // storage (2 bytes per half * 4 halfs per pixel = 8 bytes/px,
    // vs 16 bytes/px for float4). pitchFloats stays as "4 elements
    // per pixel" since both half4 and float4 index spaces match.
    auto allocWorkF16 = [&](Anime4kDownscaledScratches &w, int workW, int workH, const TCHAR *label) -> RGY_ERR {
        if (workW <= 0 || workH <= 0) return RGY_ERR_NONE;
        w.workW = workW;
        w.workH = workH;
        w.pitchFloats = w.workW * 4;
        w.lumaPitch = w.workW * typeBytes;
        const size_t bytesH4 = (size_t)w.workW * w.workH * 4 * 2;  // 4 halfs * 2 bytes
        const size_t bytesY  = (size_t)w.workH * w.lumaPitch;
        w.A = m_cl->createBuffer(bytesH4, CL_MEM_READ_WRITE);
        w.B = m_cl->createBuffer(bytesH4, CL_MEM_READ_WRITE);
        w.luma = m_cl->createBuffer(bytesY, CL_MEM_READ_WRITE);
        if (!w.A || !w.B || !w.luma) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate %s FP16 work scratches (%dx%d).\n"),
                label, w.workW, w.workH);
            return RGY_ERR_MEMORY_ALLOC;
        }
        return RGY_ERR_NONE;
    };

    // Allocate the darken / thin work scratches.
    //   m_fp16Scratch=true: only HQ tier is allowed (gated above), so
    //     m_darkenWorkF16 / m_thinWorkF16 are at full output resolution.
    //     m_scratchA / m_scratchB still get allocated (FP32, base chain)
    //     but the HQ darken / thin chain reads / writes the F16 buffers.
    //   m_fp16Scratch=false: legacy behaviour. m_darkenWork / m_thinWork
    //     only allocated for Fast / VeryFast (HQ tier reuses m_scratchA
    //     / m_scratchB). m_darkenWorkF16 / m_thinWorkF16 stay null.
    if (m_fp16Scratch) {
        if (prm->anime4k.darken == VppAnime4kDarken::HQ) {
            if (auto e = allocWorkF16(m_darkenWorkF16, outW, outH, _T("darken HQ FP16")); e != RGY_ERR_NONE) {
                return e;
            }
        }
        if (prm->anime4k.thin == VppAnime4kThin::HQ) {
            if (auto e = allocWorkF16(m_thinWorkF16, outW, outH, _T("thin HQ FP16")); e != RGY_ERR_NONE) {
                return e;
            }
        }
    } else {
        if (auto e = allocWork(m_darkenWork, tierDiv((int)prm->anime4k.darken), _T("darken")); e != RGY_ERR_NONE) {
            return e;
        }
        if (auto e = allocWork(m_thinWork, tierDiv((int)prm->anime4k.thin), _T("thin")); e != RGY_ERR_NONE) {
            return e;
        }
    }

    // mode=dtd needs a writable 1x luma scratch for stage A (which
    // modifies the post-source luma in place). Stages B/C reuse the
    // existing m_scratchA / m_scratchB float4 buffers -- those are
    // allocated at full 2x output dimensions, so the 1x stages use
    // the top-left srcW x srcH region of each buffer (with the same
    // float4 row pitch as the 2x case).
    if (prm->anime4k.mode == VppAnime4kMode::Dtd) {
        m_dtdSrcW = srcW;
        m_dtdSrcH = srcH;
        m_dtdSrcLumaPitch = srcW * typeBytes;
        const size_t bytes = (size_t)m_dtdSrcH * m_dtdSrcLumaPitch;
        m_dtdSrcLuma = m_cl->createBuffer(bytes, CL_MEM_READ_WRITE);
        if (!m_dtdSrcLuma) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate dtd source luma scratch (%dx%d).\n"),
                m_dtdSrcW, m_dtdSrcH);
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    // chroma_resize=joint: allocate the low-res luma guide (source luma box-
    // downscaled to chroma resolution = srcW/2 x srcH/2 for yuv420). Only when
    // the joint-bilateral chroma path is actually selected (scale=2 + chroma).
    if (prm->anime4k.chromaResize == VppAnime4kChromaResize::Joint
            && prm->anime4k.scale == 2 && prm->anime4k.chroma) {
        m_chromaLowresW = (srcW + 1) / 2;
        m_chromaLowresH = (srcH + 1) / 2;
        m_chromaLowresPitch = m_chromaLowresW * typeBytes;
        const size_t bytes = (size_t)m_chromaLowresH * m_chromaLowresPitch;
        m_chromaLumaLowres = m_cl->createBuffer(bytes, CL_MEM_READ_WRITE);
        if (!m_chromaLumaLowres) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate chroma joint-bilateral luma scratch (%dx%d).\n"),
                m_chromaLowresW, m_chromaLowresH);
            return RGY_ERR_MEMORY_ALLOC;
        }
    }


    // Prefilter denoise scratches. Allocated only when the user enabled
    // prefilter_denoise= (Mean / Median / Mode). Sized to the INPUT
    // plane (which equals output for restore modes and dog_sharpen, and
    // is half of output for acnet_*/dog/dtd which scale by 2). The
    // bilateral denoise kernel writes the filtered luma into
    // m_prefilterPlane; the substitute RGYFrameInfo in run_filter
    // routes downstream consumers to read this buffer instead of the
    // source plane. m_prefilterRef is a float4 .x-channel reference
    // built by kernel_anime4k_thin_copy_y_to_ref before each denoise
    // dispatch (matches the post-process denoise's reference-buffer
    // convention).
    if (prm->anime4k.prefilterDenoise != VppAnime4kDenoise::Off) {
        const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
        const int pixBytes = (bitDepth > 8) ? 2 : 1;
        const size_t planeBytes = (size_t)srcW * srcH * pixBytes;
        const size_t refBytes   = (size_t)srcW * srcH * 4 * sizeof(float);
        m_prefilterPlane = m_cl->createBuffer(planeBytes, CL_MEM_READ_WRITE);
        m_prefilterRef   = m_cl->createBuffer(refBytes,   CL_MEM_READ_WRITE);
        if (!m_prefilterPlane || !m_prefilterRef) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate prefilter denoise scratches.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_prefilterPlanePitch  = srcW * pixBytes;
        m_prefilterRefPitchF4  = srcW;
    }

    // Clamp_Highlights scratches: 2 x (1-ch fp16 at source dims). Held for
    // the filter lifetime. Off by default; allocated for ANY active mode
    // when clamp_highlights=true -- the RGB pipeline uses the fp16-input
    // kernels (kernel_anime4k_clamp_*_rgb) and the Y-only pipeline uses the
    // native-Type kernels (kernel_anime4k_clamp_h_max_y /
    // kernel_anime4k_clamp_apply_y) which read/write the Y plane in its
    // native pixel format via the Type macro set at OpenCL compile time.
    // STATSMAX storage is always fp16 regardless of source bit depth.
    if (prm->anime4k.clampHighlights) {
        const size_t bytes_1ch = (size_t)srcW * srcH * sizeof(uint16_t);
        m_clampStatsMaxH = m_cl->createBuffer(bytes_1ch, CL_MEM_READ_WRITE);
        m_clampStatsMax  = m_cl->createBuffer(bytes_1ch, CL_MEM_READ_WRITE);
        if (!m_clampStatsMaxH || !m_clampStatsMax) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate clamp_highlights scratches.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_clampStatsStride = srcW;
    }


    // Opt-in end-of-chain resize. The anime4k mode chain produces output at
    // (outW x outH) = scale*src into m_frameBuf; if the user requested a
    // different final resolution via out_res=, instantiate an internal
    // RGYFilterResize whose input is that scale*src frame and whose output is
    // the requested size. run_filter chains it after the core. This reuses the
    // full resampler family (jinc/nis/lanczos/bicubic/spline) -- no duplicate
    // resize code -- and runs in the correct order (anime4k THEN resize), which
    // the global --vpp-resize stage (which precedes anime4k) cannot do.
    m_postResize.reset();
    if (prm->anime4k.postResizeW != 0 && prm->anime4k.postResizeH != 0) {
        // Resolve the requested target. A negative value on one axis keeps the
        // source aspect (magnitude = rounding step), matching --output-res. The
        // anime4k output (outW x outH) preserves the source aspect through the
        // integer CNN scale, so resolving against it + the input SAR gives the
        // same DAR-correct result the global resizer would (e.g. -2x1080 on a
        // 16:9 source -> 1920, not the raw-pixel 1908).
        int tgtW = prm->anime4k.postResizeW;
        int tgtH = prm->anime4k.postResizeH;
        if (tgtW < 0 || tgtH < 0) {
            sInputCrop nocrop;
            memset(&nocrop, 0, sizeof(nocrop));
            set_auto_resolution(tgtW, tgtH, 1, 1, outW, outH, prm->sar[0], prm->sar[1],
                2, 2, RGYResizeResMode::Normal, false, nocrop);
        }
        if (tgtW > 0 && tgtH > 0 && (tgtW != outW || tgtH != outH)) {
            auto resizeParam = std::make_shared<RGYFilterParamResize>();
            // AUTO is resolved upstream only for the global resize; pick a sane
            // concrete default here so the sub-filter never sees AUTO.
            resizeParam->interp = (prm->anime4k.postResizeAlgo == RGY_VPP_RESIZE_AUTO)
                                  ? RGY_VPP_RESIZE_LANCZOS4 : prm->anime4k.postResizeAlgo;
            resizeParam->frameIn  = prm->frameOut;             // anime4k core output: outW x outH, csp/pitch set above
            resizeParam->frameOut = prm->frameOut;
            resizeParam->frameOut.width  = tgtW;
            resizeParam->frameOut.height = tgtH;
            resizeParam->baseFps       = prm->baseFps;
            resizeParam->bOutOverwrite = false;
            m_postResize = std::make_unique<RGYFilterResize>(m_cl);
            auto rsts = m_postResize->init(resizeParam, m_pLog);
            if (rsts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to init anime4k end-of-chain resize: %s.\n"), get_err_mes(rsts));
                return rsts;
            }
            // Report the FINAL (resized) frame info (dims + pitches) to the
            // pipeline; m_frameBuf stays at outW x outH as the core intermediate.
            prm->frameOut = resizeParam->frameOut;
            AddMessage(RGY_LOG_DEBUG, _T("anime4k: end-of-chain resize %dx%d -> %dx%d (%s).\n"),
                outW, outH, tgtW, tgtH,
                get_cx_desc(list_vpp_resize, (int)resizeParam->interp));
        }
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

// Run the bilateral denoise kernel (Mean/Median/Mode) from the source
// luma plane into m_prefilterPlane, using m_prefilterRef as the float4
// .x-channel reference. Called from run_filter before the per-mode
// dispatch when prefilter_denoise= is active. Wait_events propagate
// into the first kernel; the helper does not signal a completion event
// (downstream consumers chain naturally on the same queue).
RGY_ERR RGYFilterAnime4k::runClampHighlightsY(
    const RGYFrameInfo *pInputPlaneY,
    const RGYFrameInfo *pOutputPlaneY,
    RGYOpenCLQueue &queue) {
    if (!m_anime4k.get() || !m_clampStatsMaxH || !m_clampStatsMax) {
        AddMessage(RGY_LOG_ERROR, _T("clamp_highlights scratches not initialised.\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    const int srcW = pInputPlaneY->width;
    const int srcH = pInputPlaneY->height;
    const int dstW = pOutputPlaneY->width;
    const int dstH = pOutputPlaneY->height;
    const RGYWorkSize local_2d(ANIME4K_BLOCK_X, ANIME4K_BLOCK_Y);
    const RGYWorkSize global_src(srcW, srcH);
    const RGYWorkSize global_dst(dstW, dstH);

    // Pass 1: h-max from native-format Y plane -> fp16 STATSMAX_h.
    // The kernel reads via anime4k_read_y_norm which uses the Type macro
    // (uchar or ushort) set at OpenCL compile time -- so we just pass the
    // raw cl_mem and the native pitch in bytes.
    {
        auto err = m_anime4k.get()->kernel("kernel_anime4k_clamp_h_max_y")
            .config(queue, local_2d, global_src, {}, nullptr).launch(
                m_clampStatsMaxH->mem(), m_clampStatsStride,
                (cl_mem)pInputPlaneY->ptr[0], pInputPlaneY->pitch[0],
                srcW, srcH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("clamp_highlights h-max(y) failed: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    // Pass 2: v-max (shared with RGB path; fp16 -> fp16, no Type involved).
    {
        auto err = m_anime4k.get()->kernel("kernel_anime4k_clamp_v_max")
            .config(queue, local_2d, global_src, {}, nullptr).launch(
                m_clampStatsMax->mem(),  m_clampStatsStride,
                m_clampStatsMaxH->mem(), m_clampStatsStride,
                srcW, srcH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("clamp_highlights v-max failed: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    // Pass 3: apply (in-place modify native-format Y output at dst dims).
    {
        auto err = m_anime4k.get()->kernel("kernel_anime4k_clamp_apply_y")
            .config(queue, local_2d, global_dst, {}, nullptr).launch(
                (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
                m_clampStatsMax->mem(), m_clampStatsStride,
                dstW, dstH,
                srcW, srcH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("clamp_highlights apply(y) failed: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

// PixelClipper anti-ringing post-process (Artoriuz, MIT). Clamps each output
// luma to the 2x2 source min/max envelope, mixed by strength. Reads the source
// Y (pInputPlaneY, pre-upscale) and the output Y (pOutputPlaneY, scale*src) and
// modifies the output in place. No scratch buffers.
RGY_ERR RGYFilterAnime4k::runAntiring(
    const RGYFrameInfo *pInputPlaneY,
    const RGYFrameInfo *pOutputPlaneY,
    float strength,
    RGYOpenCLQueue &queue) {
    if (!m_anime4k.get()) {
        return RGY_ERR_OPENCL_CRUSH;
    }
    const int srcW = pInputPlaneY->width;
    const int srcH = pInputPlaneY->height;
    const int dstW = pOutputPlaneY->width;
    const int dstH = pOutputPlaneY->height;
    const RGYWorkSize local_2d(ANIME4K_BLOCK_X, ANIME4K_BLOCK_Y);
    const RGYWorkSize global_dst(dstW, dstH);
    auto err = m_anime4k.get()->kernel("kernel_anime4k_antiring_y")
        .config(queue, local_2d, global_dst, {}, nullptr).launch(
            (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
            (cl_mem)pInputPlaneY->ptr[0],  pInputPlaneY->pitch[0],
            dstW, dstH, srcW, srcH, strength);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("kernel_anime4k_antiring_y failed: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterAnime4k::runPrefilterDenoise(
    const RGYFrameInfo *pInputPlaneY,
    RGYOpenCLQueue &queue,
    const std::vector<RGYOpenCLEvent> &wait_events) {
    if (!m_anime4k.get() || !m_prefilterPlane || !m_prefilterRef) {
        return RGY_ERR_OPENCL_CRUSH;
    }
    auto prm = std::dynamic_pointer_cast<RGYFilterParamAnime4k>(m_param);
    if (!prm) return RGY_ERR_INVALID_PARAM;

    const int srcW = pInputPlaneY->width;
    const int srcH = pInputPlaneY->height;
    const RGYWorkSize local_2d(ANIME4K_BLOCK_X, ANIME4K_BLOCK_Y);
    const RGYWorkSize global(srcW, srcH);

    // Step 1: copy input Y into the float4 .x channel of m_prefilterRef.
    {
        const char *kname = "kernel_anime4k_thin_copy_y_to_ref";
        auto err = m_anime4k.get()->kernel(kname).config(queue, local_2d, global, wait_events, nullptr).launch(
            m_prefilterRef->mem(), m_prefilterRefPitchF4,
            (cl_mem)pInputPlaneY->ptr[0], pInputPlaneY->pitch[0],
            srcW, srcH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s (prefilter) failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    // Step 2: run the bilateral denoise variant from the ref into the
    // prefiltered luma scratch.
    const char *kname = nullptr;
    switch (prm->anime4k.prefilterDenoise) {
    case VppAnime4kDenoise::Mean:   kname = "kernel_anime4k_denoise_mean";   break;
    case VppAnime4kDenoise::Median: kname = "kernel_anime4k_denoise_median"; break;
    case VppAnime4kDenoise::Mode:   kname = "kernel_anime4k_denoise_mode";   break;
    default: return RGY_ERR_NONE;  // unreachable: caller gates on != Off
    }
    auto err = m_anime4k.get()->kernel(kname).config(queue, local_2d, global, {}, nullptr).launch(
        m_prefilterPlane->mem(), m_prefilterPlanePitch,
        m_prefilterRef->mem(),   m_prefilterRefPitchF4,
        srcW, srcH);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("%s (prefilter) failed: %s.\n"),
            char_to_tstring(kname).c_str(), get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterAnime4k::runModeOriginal(const Anime4kDispatchCtx &ctx, RGYFrameInfo *pOutputPlaneY,
                                          const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    // Base shader chain: Sobel partial + polynomial-refinement
    // edge-blend at the 2x output resolution. Cite
    // Anime4K_Upscale_Original_x2.glsl v3.2.
    {
        const char *kname = "kernel_anime4k_sobel_x";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_x_pass, ctx.global, wait_events, nullptr).launch(
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            ctx.srcImageMem,
            ctx.srcW, ctx.srcH, ctx.outW, ctx.outH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    {
        const char *kname = "kernel_anime4k_sobel_y";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_y_pass, ctx.global, {}, nullptr).launch(
            m_scratchB->mem(), m_scratchPitchFloats / 4,
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            ctx.outW, ctx.outH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    {
        const char *kname = "kernel_anime4k_refine_x";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_x_pass, ctx.global, {}, nullptr).launch(
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            m_scratchB->mem(), m_scratchPitchFloats / 4,
            ctx.outW, ctx.outH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    {
        const char *kname = "kernel_anime4k_refine_y";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_y_pass, ctx.global, {}, nullptr).launch(
            m_scratchB->mem(), m_scratchPitchFloats / 4,
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            ctx.outW, ctx.outH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    {
        const char *kname = "kernel_anime4k_apply";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, ctx.global, {}, event).launch(
            (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
            ctx.srcImageMem,
            m_scratchB->mem(), m_scratchPitchFloats / 4,
            ctx.outW, ctx.outH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterAnime4k::runModeDogSharpen(const Anime4kDispatchCtx &ctx, RGYFrameInfo *pOutputPlaneY, const RGYFrameInfo *pInputPlaneY,
                                             const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    // 1x DoG sharpener: outW == srcW, outH == srcH because init()
    // promoted scale to 1. Cite Anime4K_Deblur_DoG.glsl v3.2.
    {
        const char *kname = "kernel_anime4k_dog_kernel_x";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_x_pass, ctx.global, wait_events, nullptr).launch(
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            (cl_mem)pInputPlaneY->ptr[0], pInputPlaneY->pitch[0],
            ctx.outW, ctx.outH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    {
        const char *kname = "kernel_anime4k_dog_kernel_y";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_y_pass, ctx.global, {}, nullptr).launch(
            m_scratchB->mem(), m_scratchPitchFloats / 4,
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            ctx.outW, ctx.outH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    {
        const char *kname = "kernel_anime4k_dog_apply_soft";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, ctx.global, {}, event).launch(
            (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
            (cl_mem)pInputPlaneY->ptr[0], pInputPlaneY->pitch[0],
            m_scratchB->mem(), m_scratchPitchFloats / 4,
            ctx.outW, ctx.outH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterAnime4k::runModeDog(const Anime4kDispatchCtx &ctx, RGYFrameInfo *pOutputPlaneY, const RGYFrameInfo *pInputPlaneY,
                                      const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    // 2x DoG upscale: DoG kernels at 1x source res (using the
    // top-left srcW x srcH region of the float4 ping-pong scratches),
    // then a 2x apply that bilinear-upsamples both luma and the
    // gauss / minmax scratch. Cite Anime4K_Upscale_DoG_x2.glsl v3.2.
    const RGYWorkSize global_1x(ctx.srcW, ctx.srcH);
    {
        const char *kname = "kernel_anime4k_dog_kernel_x";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_x_pass, global_1x, wait_events, nullptr).launch(
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            (cl_mem)pInputPlaneY->ptr[0], pInputPlaneY->pitch[0],
            ctx.srcW, ctx.srcH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    {
        const char *kname = "kernel_anime4k_dog_kernel_y";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_y_pass, global_1x, {}, nullptr).launch(
            m_scratchB->mem(), m_scratchPitchFloats / 4,
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            ctx.srcW, ctx.srcH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    {
        const char *kname = "kernel_anime4k_dog_apply_upscale";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, ctx.global, {}, event).launch(
            (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
            ctx.srcImageMem,
            m_scratchB->mem(), m_scratchPitchFloats / 4,
            ctx.srcW, ctx.srcH, ctx.outW, ctx.outH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterAnime4k::runModeDtd(const Anime4kDispatchCtx &ctx, RGYFrameInfo *pOutputPlaneY, const RGYFrameInfo *pInputPlaneY,
                                      const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    // 2x composite: Darken -> Thin -> Deblur fused chain. Cite
    // Anime4K_Upscale_DTD_x2.glsl v3.2.
    // Stage A: darken at 1x source res with strength=1.8 baked.
    // Stage B: thin (Sobel/Gauss/Sobel) at 1x, then a 1x->2x warp.
    // Stage C: DoG sharpen at 2x with strength=0.5 baked.
    const RGYWorkSize global_1x(ctx.srcW, ctx.srcH);
    {
        const char *kname = "kernel_anime4k_copy_y_to_y";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, global_1x, wait_events, nullptr).launch(
            m_dtdSrcLuma->mem(), m_dtdSrcLumaPitch,
            (cl_mem)pInputPlaneY->ptr[0], pInputPlaneY->pitch[0],
            ctx.srcW, ctx.srcH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    // Stage A: darken chain at 1x, in-place on m_dtdSrcLuma.
    {
        const char *kname = "kernel_anime4k_darken_gauss1_x";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_x_pass, global_1x, {}, nullptr).launch(
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            m_dtdSrcLuma->mem(), m_dtdSrcLumaPitch,
            ctx.srcW, ctx.srcH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    {
        const char *kname = "kernel_anime4k_darken_dog_y";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_y_pass, global_1x, {}, nullptr).launch(
            m_scratchB->mem(), m_scratchPitchFloats / 4,
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            m_dtdSrcLuma->mem(), m_dtdSrcLumaPitch,
            ctx.srcW, ctx.srcH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    {
        const char *kname = "kernel_anime4k_darken_gauss2_x";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_x_pass, global_1x, {}, nullptr).launch(
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            m_scratchB->mem(), m_scratchPitchFloats / 4,
            ctx.srcW, ctx.srcH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    {
        const char *kname = "kernel_anime4k_darken_apply_y";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_y_pass, global_1x, {}, nullptr).launch(
            m_dtdSrcLuma->mem(), m_dtdSrcLumaPitch,
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            ctx.srcW, ctx.srcH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    // Stage B: thin chain at 1x, then upsample warp to 2x.
    // Fused Sobel-X + Sobel-Y in one kernel; reads m_dtdSrcLuma at
    // a 3x3 stencil and writes shaped magnitude into m_scratchB.x.
    {
        const char *kname = "kernel_anime4k_thin_sobel_xy";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, global_1x, {}, nullptr).launch(
            m_scratchB->mem(), m_scratchPitchFloats / 4,
            m_dtdSrcLuma->mem(), m_dtdSrcLumaPitch,
            ctx.srcW, ctx.srcH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    {
        const char *kname = "kernel_anime4k_thin_gauss_x";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_x_pass, global_1x, {}, nullptr).launch(
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            m_scratchB->mem(), m_scratchPitchFloats / 4,
            ctx.srcW, ctx.srcH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    {
        const char *kname = "kernel_anime4k_thin_gauss_y";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_y_pass, global_1x, {}, nullptr).launch(
            m_scratchB->mem(), m_scratchPitchFloats / 4,
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            ctx.srcW, ctx.srcH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    // Fused Kernel-X + Kernel-Y in one kernel; reads smoothed
    // magnitude from m_scratchB and writes the signed flow field
    // into m_scratchA.xy. Note: the un-fused chain ended with the
    // flow in m_scratchB; the fused chain ends in A, so the warp
    // below reads its pSrcFlow from m_scratchA.
    {
        const char *kname = "kernel_anime4k_thin_kernel_xy";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, global_1x, {}, nullptr).launch(
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            m_scratchB->mem(), m_scratchPitchFloats / 4,
            ctx.srcW, ctx.srcH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    {
        const char *kname = "kernel_anime4k_dtd_warp";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, ctx.global, {}, nullptr).launch(
            (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
            m_dtdSrcLuma->mem(), m_dtdSrcLumaPitch,
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            ctx.srcW, ctx.srcH, ctx.outW, ctx.outH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    // Stage C: DoG sharpen on the post-warp pDstY at 2x, in place.
    {
        const char *kname = "kernel_anime4k_dog_kernel_x";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_x_pass, ctx.global, {}, nullptr).launch(
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
            ctx.outW, ctx.outH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    {
        const char *kname = "kernel_anime4k_dog_kernel_y";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_y_pass, ctx.global, {}, nullptr).launch(
            m_scratchB->mem(), m_scratchPitchFloats / 4,
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            ctx.outW, ctx.outH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    {
        const char *kname = "kernel_anime4k_dog_apply_soft";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, ctx.global, {}, event).launch(
            (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
            (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
            m_scratchB->mem(), m_scratchPitchFloats / 4,
            ctx.outW, ctx.outH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterAnime4k::runDarkenChain(const Anime4kDispatchCtx &ctx, RGYFrameInfo *pOutputPlaneY,
                                          VppAnime4kDarken tier, RGYOpenCLEvent *event) {
    // Darken chain. The HQ tier runs at full output resolution and
    // reuses the base chain's m_scratchA / m_scratchB float4 buffers;
    // its final pass fuses the vertical smoothing Gauss with the apply
    // step that adds STRENGTH * smoothed to pDstY in place. The Fast
    // and VeryFast tiers run at half / quarter output resolution
    // against m_darkenWork.{A, B, luma}; the smoothing and apply steps
    // are split because the final apply must hit the full-resolution Y
    // plane via bilinear upsample of the work-res mask.
    // Chain shape (HQ):
    //   gauss1_x      : pDstY -> A.x  (horiz Gauss of luma)
    //   dog_y         : A.x, pDstY -> B.x  (vert Gauss; min(Y - blur, 0))
    //   gauss2_x      : B.x -> A.x  (horiz Gauss of dark-edge mask)
    //   apply_y       : A.x, pDstY -> pDstY  (vert Gauss + add * STRENGTH)
    // Chain shape (Fast / VeryFast):
    //   downsample_y     : pDstY -> luma   (box=2 or 4)
    //   gauss1_x         : luma  -> A.x
    //   dog_y            : A.x, luma -> B.x
    //   gauss2_x         : B.x   -> A.x
    //   smooth_y         : A.x   -> B.x    (no apply)
    //   upsample_apply   : B.x   -> pDstY  (bilinear upsample + add)
    if (tier == VppAnime4kDarken::HQ) {
        // FP16 path uses the dedicated m_darkenWorkF16 buffers
        // (half4 storage; kernels read/write via vload_half4 /
        // vstore_half4 because the program was built with
        // -D ANIME4K_SCRATCH_FP16=1). The base chain's FP32
        // m_scratchA / m_scratchB are not aliased here so the
        // polynomial intermediates from runPlaneY remain available
        // for downstream stages.
        // FP32 path keeps the legacy behaviour: HQ darken reuses
        // m_scratchA / m_scratchB at full output res.
        cl_mem darkenA = m_fp16Scratch ? m_darkenWorkF16.A->mem() : m_scratchA->mem();
        cl_mem darkenB = m_fp16Scratch ? m_darkenWorkF16.B->mem() : m_scratchB->mem();
        const int darkenPitchF4 = m_fp16Scratch
            ? (m_darkenWorkF16.pitchFloats / 4)
            : (m_scratchPitchFloats / 4);
        {
            const char *kname = "kernel_anime4k_darken_gauss1_x";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_x_pass, ctx.global, {}, nullptr).launch(
                darkenA, darkenPitchF4,
                (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
                ctx.outW, ctx.outH);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        {
            const char *kname = "kernel_anime4k_darken_dog_y";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_y_pass, ctx.global, {}, nullptr).launch(
                darkenB, darkenPitchF4,
                darkenA, darkenPitchF4,
                (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
                ctx.outW, ctx.outH);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        {
            const char *kname = "kernel_anime4k_darken_gauss2_x";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_x_pass, ctx.global, {}, nullptr).launch(
                darkenA, darkenPitchF4,
                darkenB, darkenPitchF4,
                ctx.outW, ctx.outH);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        {
            const char *kname = "kernel_anime4k_darken_apply_y";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_y_pass, ctx.global, {}, event).launch(
                (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
                darkenA, darkenPitchF4,
                ctx.outW, ctx.outH);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
    } else {
        const int wW = m_darkenWork.workW;
        const int wH = m_darkenWork.workH;
        const int wPitchF4 = m_darkenWork.pitchFloats / 4;
        const int box = (tier == VppAnime4kDarken::Fast) ? 2 : 4;
        const RGYWorkSize work_global(wW, wH);
        {
            const char *kname = "kernel_anime4k_downsample_y";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, work_global, {}, nullptr).launch(
                m_darkenWork.luma->mem(), m_darkenWork.lumaPitch,
                (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
                wW, wH, ctx.outW, ctx.outH, box);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        {
            const char *kname = "kernel_anime4k_darken_gauss1_x";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_x_pass, work_global, {}, nullptr).launch(
                m_darkenWork.A->mem(), wPitchF4,
                m_darkenWork.luma->mem(), m_darkenWork.lumaPitch,
                wW, wH);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        {
            const char *kname = "kernel_anime4k_darken_dog_y";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_y_pass, work_global, {}, nullptr).launch(
                m_darkenWork.B->mem(), wPitchF4,
                m_darkenWork.A->mem(), wPitchF4,
                m_darkenWork.luma->mem(), m_darkenWork.lumaPitch,
                wW, wH);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        {
            const char *kname = "kernel_anime4k_darken_gauss2_x";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_x_pass, work_global, {}, nullptr).launch(
                m_darkenWork.A->mem(), wPitchF4,
                m_darkenWork.B->mem(), wPitchF4,
                wW, wH);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        {
            const char *kname = "kernel_anime4k_darken_smooth_y";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_y_pass, work_global, {}, nullptr).launch(
                m_darkenWork.B->mem(), wPitchF4,
                m_darkenWork.A->mem(), wPitchF4,
                wW, wH);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        {
            const char *kname = "kernel_anime4k_darken_upsample_apply";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, ctx.global, {}, event).launch(
                (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
                m_darkenWork.B->mem(), wPitchF4,
                ctx.outW, ctx.outH, wW, wH);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterAnime4k::runThinChain(const Anime4kDispatchCtx &ctx, RGYFrameInfo *pOutputPlaneY,
                                        VppAnime4kThin tier, RGYOpenCLEvent *event) {
    // Thin chain. The HQ tier runs the entire Sobel-Gauss-Sobel-warp
    // chain at full output resolution against m_scratchA / m_scratchB.
    // The Fast and VeryFast tiers run the Sobel/Gauss/kernel passes at
    // half / quarter resolution against m_thinWork.{A, B, luma}; the
    // final copy-to-reference + warp pair stays at full output res so
    // the warp output keeps the upscale's full sharpness. The warp
    // bilinear-samples the (now lower-res) flow field via the
    // flowW / flowH parameters.
    // The flow buffer that the warp will read at the end. For HQ it
    // is m_scratchB at full res; for Fast/VeryFast it is
    // m_thinWork.B at work res.
    cl_mem flowBuf      = nullptr;
    int    flowPitchF4  = 0;
    int    flowW        = 0;
    int    flowH        = 0;
    // For the HQ tier we resolve a (thinA, thinB) cl_mem pair up
    // front. FP16: dedicated half4 buffers m_thinWorkF16.A / .B.
    // FP32: legacy alias to m_scratchA / m_scratchB at full output
    // res. The fused chain ends with the flow field in thinA (one
    // ping-pong less than the original four-pass form), so the
    // copy_y_to_ref below writes the yref into thinB and the warp
    // call passes pSrcA = thinB (yref), pSrcB = thinA (flow).
    // yrefBuf / yrefPitchF4 carry that yref destination (and the
    // pitch the warp uses to read it back). For Fast / VeryFast,
    // the yref buffer is m_scratchA at full output res -- distinct
    // from m_thinWork.{A,B} which only hold the work-res chain.
    cl_mem thinA = nullptr;
    cl_mem thinB = nullptr;
    int    thinPitchF4 = 0;
    cl_mem yrefBuf = nullptr;
    int    yrefPitchF4 = 0;
    if (tier == VppAnime4kThin::HQ) {
        thinA = m_fp16Scratch ? m_thinWorkF16.A->mem() : m_scratchA->mem();
        thinB = m_fp16Scratch ? m_thinWorkF16.B->mem() : m_scratchB->mem();
        thinPitchF4 = m_fp16Scratch
            ? (m_thinWorkF16.pitchFloats / 4)
            : (m_scratchPitchFloats / 4);
        // Fused Sobel-X + Sobel-Y in one kernel; pDstY -> thinB.
        // Replaces two kernels and one intermediate scratch round
        // trip from the original chain.
        {
            const char *kname = "kernel_anime4k_thin_sobel_xy";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, ctx.global, {}, nullptr).launch(
                thinB, thinPitchF4,
                (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
                ctx.outW, ctx.outH);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        {
            const char *kname = "kernel_anime4k_thin_gauss_x";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_x_pass, ctx.global, {}, nullptr).launch(
                thinA, thinPitchF4,
                thinB, thinPitchF4,
                ctx.outW, ctx.outH);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        {
            const char *kname = "kernel_anime4k_thin_gauss_y";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_y_pass, ctx.global, {}, nullptr).launch(
                thinB, thinPitchF4,
                thinA, thinPitchF4,
                ctx.outW, ctx.outH);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        // Fused Kernel-X + Kernel-Y in one kernel; thinB -> thinA.
        // Note the swap from the original chain: flow lives in
        // thinA at the end of this pass (the original two-pass
        // form ended in thinB).
        {
            const char *kname = "kernel_anime4k_thin_kernel_xy";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, ctx.global, {}, nullptr).launch(
                thinA, thinPitchF4,
                thinB, thinPitchF4,
                ctx.outW, ctx.outH);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        flowBuf     = thinA;
        flowPitchF4 = thinPitchF4;
        flowW       = ctx.outW;
        flowH       = ctx.outH;
        yrefBuf     = thinB;
        yrefPitchF4 = thinPitchF4;
    } else {
        const int wW = m_thinWork.workW;
        const int wH = m_thinWork.workH;
        const int wPitchF4 = m_thinWork.pitchFloats / 4;
        const int box = (tier == VppAnime4kThin::Fast) ? 2 : 4;
        const RGYWorkSize work_global(wW, wH);
        {
            const char *kname = "kernel_anime4k_downsample_y";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, work_global, {}, nullptr).launch(
                m_thinWork.luma->mem(), m_thinWork.lumaPitch,
                (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
                wW, wH, ctx.outW, ctx.outH, box);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        // Fused Sobel-X + Sobel-Y in one kernel; luma -> B.
        {
            const char *kname = "kernel_anime4k_thin_sobel_xy";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, work_global, {}, nullptr).launch(
                m_thinWork.B->mem(), wPitchF4,
                m_thinWork.luma->mem(), m_thinWork.lumaPitch,
                wW, wH);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        {
            const char *kname = "kernel_anime4k_thin_gauss_x";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_x_pass, work_global, {}, nullptr).launch(
                m_thinWork.A->mem(), wPitchF4,
                m_thinWork.B->mem(), wPitchF4,
                wW, wH);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        {
            const char *kname = "kernel_anime4k_thin_gauss_y";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_y_pass, work_global, {}, nullptr).launch(
                m_thinWork.B->mem(), wPitchF4,
                m_thinWork.A->mem(), wPitchF4,
                wW, wH);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        // Fused Kernel-X + Kernel-Y in one kernel; B -> A (flow now
        // lives in m_thinWork.A; the original chain ended in B).
        {
            const char *kname = "kernel_anime4k_thin_kernel_xy";
            auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, work_global, {}, nullptr).launch(
                m_thinWork.A->mem(), wPitchF4,
                m_thinWork.B->mem(), wPitchF4,
                wW, wH);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                    char_to_tstring(kname).c_str(), get_err_mes(err));
                return err;
            }
        }
        flowBuf     = m_thinWork.A->mem();
        flowPitchF4 = wPitchF4;
        flowW       = wW;
        flowH       = wH;
        // Fast / VeryFast: yref goes into m_scratchA at full output
        // res (separate from m_thinWork.* which is at work res).
        yrefBuf     = m_scratchA->mem();
        yrefPitchF4 = m_scratchPitchFloats / 4;
    }
    // Copy the current Y plane (post-apply, optionally post-darken)
    // into yrefBuf at full output resolution. The destination
    // differs by tier:
    //   HQ:          thinB (= m_thinWorkF16.B in FP16 mode, or
    //                m_scratchB in FP32 mode). After the fused
    //                kernel_xy pass the flow lives in thinA, so
    //                thinB is the free slot for the yref.
    //   Fast/VeryFast: m_scratchA at full output res, distinct from
    //                  the work-res m_thinWork.{A,B} buffers.
    // The warp pass reads pSrcA = yref via manual bilinear and
    // writes pDstY in place; without this copy the read and write
    // would race.
    {
        const char *kname = "kernel_anime4k_thin_copy_y_to_ref";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, ctx.global, {}, nullptr).launch(
            yrefBuf, yrefPitchF4,
            (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
            ctx.outW, ctx.outH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    {
        const char *kname = "kernel_anime4k_thin_warp";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, ctx.global, {}, event).launch(
            (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
            yrefBuf, yrefPitchF4,
            flowBuf, flowPitchF4,
            ctx.outW, ctx.outH,
            flowW, flowH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterAnime4k::runDenoiseChain(const Anime4kDispatchCtx &ctx, RGYFrameInfo *pOutputPlaneY,
                                           VppAnime4kDenoise tier, RGYOpenCLEvent *event) {
    // Bilateral denoise pass (Mean / Median / Mode). Single OCL kernel
    // each, all read a float Y reference written into m_scratchA.x by
    // the same kernel_anime4k_thin_copy_y_to_ref used by the thin warp
    // path. The chain so far (base apply, optional darken, optional
    // thin) has already written pDstY in place; the copy step turns
    // that into the post-chain Y reference for the denoise kernel
    // to read while it writes back to pDstY without a read-write race.
    {
        const char *kname = "kernel_anime4k_thin_copy_y_to_ref";
        auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, ctx.global, {}, nullptr).launch(
            m_scratchA->mem(), m_scratchPitchFloats / 4,
            (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
            ctx.outW, ctx.outH);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
                char_to_tstring(kname).c_str(), get_err_mes(err));
            return err;
        }
    }
    const char *kname = nullptr;
    switch (tier) {
    case VppAnime4kDenoise::Mean:   kname = "kernel_anime4k_denoise_mean";   break;
    case VppAnime4kDenoise::Median: kname = "kernel_anime4k_denoise_median"; break;
    case VppAnime4kDenoise::Mode:   kname = "kernel_anime4k_denoise_mode";   break;
    default: return RGY_ERR_NONE;  // unreachable: runDenoise gated above
    }
    auto err = m_anime4k.get()->kernel(kname).config(ctx.queue, ctx.local_2d, ctx.global, {}, event).launch(
        (cl_mem)pOutputPlaneY->ptr[0], pOutputPlaneY->pitch[0],
        m_scratchA->mem(), m_scratchPitchFloats / 4,
        ctx.outW, ctx.outH);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
            char_to_tstring(kname).c_str(), get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterAnime4k::runPlaneY(RGYFrameInfo *pOutputPlaneY, const RGYFrameInfo *pInputPlaneY,
                                    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                                    RGYOpenCLEvent *event) {
    if (!m_anime4k.get()) {
        return RGY_ERR_OPENCL_CRUSH;
    }
    const int outW = m_outW;
    const int outH = m_outH;
    // Three workgroup shapes for the anime4k chain:
    //   local_2d      -- 2D-symmetric stencils (apply, warp, copy,
    //                    downsample, upsample-apply, denoise, fused
    //                    sobel_xy / kernel_xy, chroma_resize, CNN).
    //   local_x_pass  -- 32x8 for separable horizontal passes (sobel_x,
    //                    refine_x, gauss1_x, gauss2_x, thin_gauss_x,
    //                    dog_kernel_x). The wider 32-wide work-item row
    //                    matches the row-major scratch stride and lets
    //                    the EU subslices coalesce horizontal reads.
    //   local_y_pass  -- 8x32 mirror for separable vertical passes
    //                    (sobel_y, refine_y, darken_dog_y, apply_y,
    //                    smooth_y, thin_gauss_y, dog_kernel_y); the
    //                    taller 32-wide work-item column matches the
    //                    column reads that walk through pitched rows.
    // Perf rationale: Arc A770 measurements at 4K.
    const RGYWorkSize local_2d(ANIME4K_BLOCK_X, ANIME4K_BLOCK_Y);
    const RGYWorkSize local_x_pass(32, 8);
    const RGYWorkSize local_y_pass(8, 32);
    const RGYWorkSize global(outW, outH);

    // Wrap the input luma plane as a normalised CL_R image so the
    // kernel can use the hardware bilinear sampler. createImageFromFrameBuffer
    // returns a self-managing wrapper; the pool reuses the underlying
    // image2d_from_buffer object across frames when the OpenCL driver
    // supports cl_khr_image2d_from_buffer (zero-copy).
    auto srcImage = m_cl->createImageFromFrameBuffer(*pInputPlaneY, true, CL_MEM_READ_ONLY, &m_srcImagePool);
    if (!srcImage) {
        AddMessage(RGY_LOG_ERROR, _T("failed to wrap input luma plane as image.\n"));
        return RGY_ERR_MEM_OBJECT_ALLOCATION_FAILURE;
    }
    const cl_mem srcImageMem = (cl_mem)srcImage->frame.ptr[0];
    const int srcW = pInputPlaneY->width;
    const int srcH = pInputPlaneY->height;

    // Mode and post-process tiers are needed early to decide which
    // upscale/sharpen path runs and which completion event the final
    // kernel of that path should take. For mode=dtd, runDarken /
    // runThin / runDenoise are guaranteed false (rejected / cleared
    // in init()), so the existing post-process gates below stay
    // dormant and the dtd chain's stage-C apply takes the final event.
    auto prm = std::dynamic_pointer_cast<RGYFilterParamAnime4k>(m_param);
    const VppAnime4kMode    mode        = prm ? prm->anime4k.mode    : VppAnime4kMode::Original;
    const VppAnime4kDarken  darkenTier  = prm ? prm->anime4k.darken  : VppAnime4kDarken::Off;
    const VppAnime4kThin    thinTier    = prm ? prm->anime4k.thin    : VppAnime4kThin::Off;
    const VppAnime4kDenoise denoiseTier = prm ? prm->anime4k.denoise : VppAnime4kDenoise::Off;
    const bool runDarken  = (darkenTier  != VppAnime4kDarken::Off);
    const bool runThin    = (thinTier    != VppAnime4kThin::Off);
    const bool runDenoise = (denoiseTier != VppAnime4kDenoise::Off);
    RGYOpenCLEvent *applyEvent = (runDarken || runThin || runDenoise) ? nullptr : event;

    const Anime4kDispatchCtx ctx{ queue, local_2d, local_x_pass, local_y_pass, global, srcImageMem, srcW, srcH, outW, outH };

    RGY_ERR err = RGY_ERR_NONE;
    if (mode == VppAnime4kMode::Original || mode == VppAnime4kMode::Deblur
     || mode == VppAnime4kMode::DarkenHQ || mode == VppAnime4kMode::ThinHQ) {
        err = runModeOriginal(ctx, pOutputPlaneY, wait_events, applyEvent);
    } else if (mode == VppAnime4kMode::DogSharpen) {
        err = runModeDogSharpen(ctx, pOutputPlaneY, pInputPlaneY, wait_events, applyEvent);
    } else if (mode == VppAnime4kMode::Dog) {
        err = runModeDog(ctx, pOutputPlaneY, pInputPlaneY, wait_events, applyEvent);
    } else if (mode == VppAnime4kMode::Dtd) {
        err = runModeDtd(ctx, pOutputPlaneY, pInputPlaneY, wait_events, applyEvent);
    }
    if (err != RGY_ERR_NONE) return err;

    if (runDarken) {
        err = runDarkenChain(ctx, pOutputPlaneY, darkenTier, (runThin || runDenoise) ? nullptr : event);
        if (err != RGY_ERR_NONE) return err;
    }
    if (runThin) {
        err = runThinChain(ctx, pOutputPlaneY, thinTier, runDenoise ? nullptr : event);
        if (err != RGY_ERR_NONE) return err;
    }
    if (runDenoise) {
        err = runDenoiseChain(ctx, pOutputPlaneY, denoiseTier, event);
        if (err != RGY_ERR_NONE) return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterAnime4k::runPlaneChroma(RGYFrameInfo *pOutputPlaneC, const RGYFrameInfo *pInputPlaneC,
                                          RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    if (!m_anime4k.get()) {
        return RGY_ERR_OPENCL_CRUSH;
    }
    auto prm = std::dynamic_pointer_cast<RGYFilterParamAnime4k>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_PARAM;
    }
    const RGYWorkSize local_2d(ANIME4K_BLOCK_X, ANIME4K_BLOCK_Y);
    const RGYWorkSize global(pOutputPlaneC->width, pOutputPlaneC->height);
    const int chromaMode = (int)prm->anime4k.chromaResize;

    const char *kname = "kernel_anime4k_chroma_resize";
    auto err = m_anime4k.get()->kernel(kname).config(queue, local_2d, global, {}, event).launch(
        (cl_mem)pOutputPlaneC->ptr[0], pOutputPlaneC->pitch[0],
        pOutputPlaneC->width, pOutputPlaneC->height,
        (cl_mem)pInputPlaneC->ptr[0], pInputPlaneC->pitch[0],
        pInputPlaneC->width, pInputPlaneC->height,
        chromaMode);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"),
            char_to_tstring(kname).c_str(), get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

// chroma_resize=joint pass 1: box-downscale the source luma into the chroma-res
// guide scratch m_chromaLumaLowres.
RGY_ERR RGYFilterAnime4k::runChromaLumaLowres(const RGYFrameInfo *pSrcLumaY, RGYOpenCLQueue &queue) {
    if (!m_anime4k.get() || !m_chromaLumaLowres) {
        return RGY_ERR_OPENCL_CRUSH;
    }
    const RGYWorkSize local_2d(ANIME4K_BLOCK_X, ANIME4K_BLOCK_Y);
    const RGYWorkSize global(m_chromaLowresW, m_chromaLowresH);
    const char *kname = "kernel_anime4k_chroma_luma_lowres";
    auto err = m_anime4k.get()->kernel(kname).config(queue, local_2d, global, {}, nullptr).launch(
        m_chromaLumaLowres->mem(), m_chromaLowresPitch,
        m_chromaLowresW, m_chromaLowresH,
        (cl_mem)pSrcLumaY->ptr[0], pSrcLumaY->pitch[0],
        pSrcLumaY->width, pSrcLumaY->height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"), char_to_tstring(kname).c_str(), get_err_mes(err));
    }
    return err;
}

// chroma_resize=joint pass 2: luma-guided joint-bilateral upscale of one chroma
// plane (FastBilateral, MIT). dist/int coeffs are FastBilateral's defaults.
RGY_ERR RGYFilterAnime4k::runPlaneChromaJoint(RGYFrameInfo *pOutputPlaneC, const RGYFrameInfo *pInputPlaneC,
                                             const RGYFrameInfo *pSrcLumaY,
                                             RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    if (!m_anime4k.get() || !m_chromaLumaLowres) {
        return RGY_ERR_OPENCL_CRUSH;
    }
    const RGYWorkSize local_2d(ANIME4K_BLOCK_X, ANIME4K_BLOCK_Y);
    const RGYWorkSize global(pOutputPlaneC->width, pOutputPlaneC->height);
    const float distCoeff = 2.0f;
    const float intCoeff  = 128.0f;
    const char *kname = "kernel_anime4k_chroma_joint_bilateral";
    auto err = m_anime4k.get()->kernel(kname).config(queue, local_2d, global, {}, event).launch(
        (cl_mem)pOutputPlaneC->ptr[0], pOutputPlaneC->pitch[0],
        pOutputPlaneC->width, pOutputPlaneC->height,
        (cl_mem)pInputPlaneC->ptr[0], pInputPlaneC->pitch[0],
        pInputPlaneC->width, pInputPlaneC->height,
        (cl_mem)pSrcLumaY->ptr[0], pSrcLumaY->pitch[0],
        m_chromaLumaLowres->mem(), m_chromaLowresPitch,
        distCoeff, intCoeff);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("%s failed: %s.\n"), char_to_tstring(kname).c_str(), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterAnime4k::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
                                     RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                                     RGYOpenCLEvent *event) {
    // Fast path: no end-of-chain resize -> run the core directly (the output
    // and event semantics are byte-identical to the pre-feature behaviour).
    if (!m_postResize) {
        return runAnime4kCore(pInputFrame, ppOutputFrames, pOutputFrameNum, queue, wait_events, event);
    }
    if (pInputFrame->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        return RGY_ERR_NONE;
    }
    // 1) Run the anime4k mode chain into its own (scale*src) buffer. Pass a null
    //    final event: the in-order OpenCL queue serialises the resize after the
    //    core, and the resize below signals the real completion event. Letting
    //    coreOut[0] start null makes runAnime4kCore assign its own m_frameBuf.
    RGYFrameInfo *coreOut[1] = { nullptr };
    int coreNum = 0;
    auto cerr = runAnime4kCore(pInputFrame, coreOut, &coreNum, queue, wait_events, nullptr);
    if (cerr != RGY_ERR_NONE) {
        return cerr;
    }
    if (coreNum < 1 || coreOut[0] == nullptr) {
        *pOutputFrameNum = 0;
        return RGY_ERR_NONE;
    }
    // 2) Resize the core output to the requested resolution. bOutOverwrite=false
    //    => the sub-filter writes into its own buffer and returns it in
    //    resizeOut[0], which becomes our output frame.
    RGYFrameInfo *resizeOut[1] = { nullptr };
    int resizeNum = 0;
    auto rerr = m_postResize->filter(coreOut[0], resizeOut, &resizeNum, queue, {}, event);
    if (rerr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("anime4k end-of-chain resize failed: %s.\n"), get_err_mes(rerr));
        return rerr;
    }
    ppOutputFrames[0] = resizeOut[0];
    *pOutputFrameNum = 1;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterAnime4k::runAnime4kCore(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
                                     RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                                     RGYOpenCLEvent *event) {
    if (pInputFrame->ptr[0] == nullptr) {
        return RGY_ERR_NONE;
    }
    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto &outFrame = m_frameBuf[(m_frameIdx++) % m_frameBuf.size()];
        ppOutputFrames[0] = &outFrame->frame;
    }
    auto prm = std::dynamic_pointer_cast<RGYFilterParamAnime4k>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_PARAM;
    }

    const int planeCount = RGY_CSP_PLANES[ppOutputFrames[0]->csp];

    auto planeDstY = getPlane(ppOutputFrames[0], RGY_PLANE_Y);
    auto planeSrcY = getPlane(pInputFrame,        RGY_PLANE_Y);
    RGY_ERR err = RGY_ERR_NONE;

    // Optional pre-process denoise. When prefilter_denoise= is active,
    // run the bilateral denoise on the input luma plane and substitute
    // the downstream dispatch's source pointer to read from
    // m_prefilterPlane. The substitution preserves the original input
    // plane's RGYFrameInfo shape (width/height/csp/mem_type/etc.) and
    // only swaps the cl_mem pointer + pitch. Wait_events are consumed
    // by the prefilter dispatch; downstream calls use an empty
    // wait-event list.
    RGYFrameInfo prefilteredSrcY = planeSrcY;
    const RGYFrameInfo *effectiveSrcY = &planeSrcY;
    std::vector<RGYOpenCLEvent> downstreamWaits = wait_events;
    if (prm->anime4k.prefilterDenoise != VppAnime4kDenoise::Off && m_prefilterPlane) {
        err = runPrefilterDenoise(&planeSrcY, queue, wait_events);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("prefilter denoise failed: %s\n"), get_err_mes(err));
            return err;
        }
        prefilteredSrcY.ptr[0]   = (uint8_t *)m_prefilterPlane->mem();
        prefilteredSrcY.pitch[0] = m_prefilterPlanePitch;
        effectiveSrcY            = &prefilteredSrcY;
        downstreamWaits.clear();
    }

    // GLSL Anime4K luma pass. CNN model modes were rejected at init() (they
    // moved to --vpp-onnx), so the only luma dispatch left is the
    // hand-written shader chain in runPlaneY.
    err = runPlaneY(&planeDstY, const_cast<RGYFrameInfo *>(effectiveSrcY), queue, downstreamWaits,
                    (planeCount == 1) ? event : nullptr);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("luma processing failed: %s\n"), get_err_mes(err));
        return err;
    }

    // Clamp_Highlights post-process for Y-only modes (acnet / arnet /
    // artcnn-c/r / fsrcnnx / restore-s/m / dog / dtd / artcnn-r-int8, also
    // luma-pass of artcnn_chroma_*). Reads the source Y via the effective
    // source pointer (which already accounts for prefilter-denoise
    // substitution), computes the separable 5x5 luma max-dilation, and
    // clamps each output pixel's Y at the bilinear-upsampled source
    // envelope. RGB-family modes already had their clamp applied inside
    // runPlaneRGB, so this hook is skipped for them.
    if (prm->anime4k.clampHighlights && m_clampStatsMax) {
        auto cerr = runClampHighlightsY(effectiveSrcY, &planeDstY, queue);
        if (cerr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("clamp_highlights post-process (y) failed: %s\n"), get_err_mes(cerr));
            return cerr;
        }
    }

    // PixelClipper anti-ringing post-process (antiring=<0..1>, 0 = off). Clamps
    // the upscaled luma to the local 2x2 source envelope to remove ringing.
    if (prm->anime4k.antiring > 0.0f) {
        auto aerr = runAntiring(effectiveSrcY, &planeDstY,
                                clamp(prm->anime4k.antiring, 0.0f, 1.0f), queue);
        if (aerr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("antiring post-process (y) failed: %s\n"), get_err_mes(aerr));
            return aerr;
        }
    }

    // chroma_resize=joint: build the low-res luma guide once before the per-plane
    // chroma loop (the bilateral weights depend only on luma, shared by U and V).
    const bool chromaJoint = (prm->anime4k.chromaResize == VppAnime4kChromaResize::Joint
                              && prm->anime4k.scale == 2 && prm->anime4k.chroma && m_chromaLumaLowres);
    if (chromaJoint) {
        err = runChromaLumaLowres(&planeSrcY, queue);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("chroma joint-bilateral luma lowres failed: %s\n"), get_err_mes(err));
            return err;
        }
    }

    for (int i = 1; i < planeCount; i++) {
        auto planeDstC = getPlane(ppOutputFrames[0], (RGY_PLANE)i);
        auto planeSrcC = getPlane(pInputFrame,        (RGY_PLANE)i);
        RGYOpenCLEvent *plane_event = (i == planeCount - 1) ? event : nullptr;
        if (prm->anime4k.scale == 1 || !prm->anime4k.chroma) {
            // scale=1 (or chroma=false): pass chroma through unchanged.
            err = m_cl->copyPlane(&planeDstC, &planeSrcC, nullptr, queue, {}, plane_event);
        } else if (chromaJoint) {
            err = runPlaneChromaJoint(&planeDstC, &planeSrcC, &planeSrcY, queue, plane_event);
        } else {
            err = runPlaneChroma(&planeDstC, &planeSrcC, queue, plane_event);
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("chroma plane %d failed: %s\n"), i, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

void RGYFilterAnime4k::close() {
    m_postResize.reset();
    m_prefilterPlane.reset();
    m_prefilterRef.reset();
    m_prefilterPlanePitch = 0;
    m_prefilterRefPitchF4 = 0;
    m_scratchA.reset();
    m_scratchB.reset();
    m_darkenWork.A.reset();
    m_darkenWork.B.reset();
    m_darkenWork.luma.reset();
    m_darkenWork.workW = m_darkenWork.workH = m_darkenWork.pitchFloats = m_darkenWork.lumaPitch = 0;
    m_thinWork.A.reset();
    m_thinWork.B.reset();
    m_thinWork.luma.reset();
    m_thinWork.workW = m_thinWork.workH = m_thinWork.pitchFloats = m_thinWork.lumaPitch = 0;
    m_darkenWorkF16.A.reset();
    m_darkenWorkF16.B.reset();
    m_darkenWorkF16.luma.reset();
    m_darkenWorkF16.workW = m_darkenWorkF16.workH = m_darkenWorkF16.pitchFloats = m_darkenWorkF16.lumaPitch = 0;
    m_thinWorkF16.A.reset();
    m_thinWorkF16.B.reset();
    m_thinWorkF16.luma.reset();
    m_thinWorkF16.workW = m_thinWorkF16.workH = m_thinWorkF16.pitchFloats = m_thinWorkF16.lumaPitch = 0;
    m_fp16Scratch = false;
    m_dtdSrcLuma.reset();
    m_dtdSrcLumaPitch = m_dtdSrcW = m_dtdSrcH = 0;
    m_chromaLumaLowres.reset();
    m_chromaLowresPitch = m_chromaLowresW = m_chromaLowresH = 0;
    m_srcImagePool.clear();
    m_frameBuf.clear();
    m_anime4k.clear();
    m_cl.reset();
    m_frameIdx = 0;
}
