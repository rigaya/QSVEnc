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

// Anime4K v3.2 hand-tuned luma refinement / 2x upscale chain,
// plus ACNet F8B* CNN upscale via oneDNN.
// Algorithms:
//   * Anime4K v3.2 shader chain  -- bloc97 (MIT, 2019).
//     https://github.com/bloc97/Anime4K
//   * ACNet F8B4/F8B8/F8B18      -- weights from Anime4KCPP
//     (TianZerL, MIT, 2020-2024); inference via Intel oneDNN
//     (Apache-2.0). See ACKNOWLEDGMENTS.md for upstream attribution.
//
// Implementation status:
//   * mode=original  -- Upscale_Original_x2.glsl v3.2 (REFINE_STRENGTH 0.5)
//   * mode=deblur    -- Upscale_Deblur_Original_x2.glsl v3.2 (REFINE_STRENGTH 1.0)
//   * mode=darken_hq -- alias for original+darken=true
//   * mode=thin_hq   -- alias for original+thin=true
//   * mode=acnet_s/m/l -- ACNet F8B4/F8B8/F8B18 CNN 2x upscale
//     (1->8 PReLU, B middle 8->8 PReLU, 8->4 Identity, pixel-shuffle 4->1 at 2x)
//
// Luma-guided chroma upsampling is not used and is not required
// for the YV12 pipeline.

#ifndef __RGY_FILTER_ANIME4K_H__
#define __RGY_FILTER_ANIME4K_H__

#include "rgy_filter_cl.h"
#include "rgy_prm.h"

class RGYFilterResize; // opt-in end-of-chain resize sub-filter (see RGYFilterAnime4k::m_postResize)

class RGYFilterParamAnime4k : public RGYFilterParam {
public:
    VppAnime4k anime4k;
    int sar[2] = { 0, 0 };               // input SAR (set by pipeline) -- used to resolve a negative out_res= (auto-aspect) DAR-correctly
    RGYFilterParamAnime4k() : anime4k() {};
    virtual ~RGYFilterParamAnime4k() {};
    virtual tstring print() const override { return anime4k.print(); };
};

class RGYFilterAnime4k : public RGYFilter {
public:
    RGYFilterAnime4k(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterAnime4k();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
                               RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                               RGYOpenCLEvent *event) override;
    // The anime4k mode chain itself (denoise/restore/upscale/refine). run_filter
    // is now a thin wrapper: when an end-of-chain resize is configured
    // (m_postResize != null) it runs this core into the scale*src buffer with a
    // null completion event, then runs m_postResize to land the final size and
    // signal the real event. When no resize is configured run_filter calls this
    // directly so the path is byte-identical to before.
    RGY_ERR runAnime4kCore(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
                          RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                          RGYOpenCLEvent *event);
    virtual void close() override;

    // Y-plane chain: sobel_x -> sobel_y -> refine_x -> refine_y -> apply.
    // Source reads use the hardware bilinear sampler on the wrapped
    // input plane image; scratch reads use plain buffer indexing.
    RGY_ERR runPlaneY(RGYFrameInfo *pOutputPlaneY, const RGYFrameInfo *pInputPlaneY,
                      RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                      RGYOpenCLEvent *event);

    // Chroma plane resize used when scale=2. Plain geometric resize
    // (bilinear / bicubic / spline36 / lanczos3) on the chroma plane;
    // not luma-guided.
    RGY_ERR runPlaneChroma(RGYFrameInfo *pOutputPlaneC, const RGYFrameInfo *pInputPlaneC,
                           RGYOpenCLQueue &queue, RGYOpenCLEvent *event);

    // chroma_resize=joint (FastBilateral, MIT). runChromaLumaLowres populates
    // m_chromaLumaLowres from the source luma once per frame; runPlaneChromaJoint
    // then luma-guides each chroma plane's upscale.
    RGY_ERR runChromaLumaLowres(const RGYFrameInfo *pSrcLumaY, RGYOpenCLQueue &queue);
    RGY_ERR runPlaneChromaJoint(RGYFrameInfo *pOutputPlaneC, const RGYFrameInfo *pInputPlaneC,
                                const RGYFrameInfo *pSrcLumaY,
                                RGYOpenCLQueue &queue, RGYOpenCLEvent *event);


    RGYOpenCLProgramAsync m_anime4k;
    // Scratch ping-pong buffers sized to the output luma plane. Float
    // throughout to preserve the polynomial-refinement precision.
    std::unique_ptr<RGYCLBuf> m_scratchA;
    std::unique_ptr<RGYCLBuf> m_scratchB;
    int m_scratchPitchFloats;
    int m_outW;
    int m_outH;
    RGYCLFramePool m_srcImagePool;
    int m_frameIdx;

    // Work-resolution scratch set used by the Fast / VeryFast tiers of
    // the darken and thin post-process chains. Each tier reduces the
    // working resolution (full -> half -> quarter) and the sigma in
    // tandem, so the heavy gauss / sobel / kernel passes shrink in
    // both spatial extent and per-pixel cost. The HQ tier reuses the
    // full-resolution m_scratchA / m_scratchB above and leaves these
    // pointers null. The luma buffer is the input to the chain; it
    // holds a 2x or 4x box-averaged copy of pDstY in the matching
    // uchar / ushort type so the existing gauss / sobel kernels can
    // read it the same way they read the full-res Y plane.
    struct Anime4kDownscaledScratches {
        int workW = 0;
        int workH = 0;
        int pitchFloats = 0;    // float4 row pitch, in floats (= workW * 4)
        int lumaPitch = 0;      // luma row pitch in bytes
        std::unique_ptr<RGYCLBuf> A;
        std::unique_ptr<RGYCLBuf> B;
        std::unique_ptr<RGYCLBuf> luma;
    };
    Anime4kDownscaledScratches m_darkenWork;
    Anime4kDownscaledScratches m_thinWork;

    // FP16-storage variants of the darken / thin scratches, allocated
    // when cl_khr_fp16 is advertised by the OpenCL device (detected at
    // init). The OpenCL program is JIT-built with `-D ANIME4K_SCRATCH_FP16=1`
    // in this case, and the darken / thin kernels read / write these
    // buffers via vload_half4 / vstore_half4 (OpenCL 1.2 core, no
    // extension required). Halves the per-pixel scratch bandwidth at
    // the cost of ~5e-4 precision -- well below the 1/255 8-bit
    // quantisation step.
    // The base chain's m_scratchA / m_scratchB (polynomial P5..P0
    // evaluation) is NEVER converted -- the polynomial intermediates
    // reach magnitudes around |P3|=60 and would lose accuracy beyond
    // the 1/255 quantisation floor.
    Anime4kDownscaledScratches m_darkenWorkF16;
    Anime4kDownscaledScratches m_thinWorkF16;
    bool m_fp16Scratch;

    // Working luma scratch at 1x source resolution, allocated only
    // when mode=Dtd. Stage A (darken with strength 1.8) needs to
    // modify the post-source luma in place before stage B reads it;
    // pInputPlaneY is read-only so we stage through this buffer.
    // Format matches the Y plane type (uchar / ushort per bit-depth).
    std::unique_ptr<RGYCLBuf> m_dtdSrcLuma;
    int                       m_dtdSrcLumaPitch;
    int                       m_dtdSrcW;
    int                       m_dtdSrcH;

    // chroma_resize=joint: source luma box-downscaled to chroma resolution,
    // the intensity-similarity guide for the joint-bilateral chroma upscale.
    std::unique_ptr<RGYCLBuf> m_chromaLumaLowres;
    int                       m_chromaLowresPitch;
    int                       m_chromaLowresW;
    int                       m_chromaLowresH;

    // Pre-process denoise scratches (allocated only when
    // prm->anime4k.prefilterDenoise != Off). m_prefilterPlane is a
    // luma-format scratch at INPUT dimensions; the chosen Mean/Median/Mode
    // bilateral denoise kernel writes the prefiltered luma into it. The
    // downstream luma dispatch (runPlaneY / runPlaneYCNN /
    // runPlaneYCNNRestore) reads from this buffer instead of the source
    // plane when the prefilter is active. m_prefilterRef is a float4
    // scratch holding the .x-channel luma reference required by the
    // denoise kernels (matches the post-process denoise's reference-
    // buffer convention; built by kernel_anime4k_thin_copy_y_to_ref).
    std::unique_ptr<RGYCLBuf> m_prefilterPlane;
    int                       m_prefilterPlanePitch;
    std::unique_ptr<RGYCLBuf> m_prefilterRef;
    int                       m_prefilterRefPitchF4;


    // Clamp_Highlights post-process scratches. Allocated only when
    // prm->anime4k.clampHighlights is enabled. Both are 1-channel fp16 at
    // SOURCE spatial dims; clampStatsMaxH is the horizontal-max intermediate,
    // clampStatsMax is the final separable 5x5 max-dilation. Apply pass
    // bilinear-upsamples STATSMAX to output dims and clamps each output
    // pixel's luma at the source-side local max.
    std::unique_ptr<RGYCLBuf> m_clampStatsMaxH;
    std::unique_ptr<RGYCLBuf> m_clampStatsMax;
    int                       m_clampStatsStride;  // halfs per row of STATSMAX (= srcW)

    // Helper: when prefilter_denoise != Off, run the bilateral denoise
    // kernel from pInputPlaneY into m_prefilterPlane (using m_prefilterRef
    // as the .x-channel reference). Idempotent at Off: caller checks the
    // mode first and skips this call.
    RGY_ERR runPrefilterDenoise(const RGYFrameInfo *pInputPlaneY,
                                RGYOpenCLQueue &queue,
                                const std::vector<RGYOpenCLEvent> &wait_events);

    // Clamp_Highlights helpers. Run the 3-pass pipeline (h-max -> v-max ->
    // apply). RGB variant derives luma from 3-ch NCHW source via the
    // active colourmatrix; Y variant operates directly on 1-ch Y planes.
    // Idempotent at clampHighlights=false: caller checks first and skips.
    RGY_ERR runClampHighlightsY(const RGYFrameInfo *pInputPlaneY,
                                const RGYFrameInfo *pOutputPlaneY,
                                RGYOpenCLQueue &queue);

    // PixelClipper anti-ringing (Artoriuz, MIT). Clamps the upscaled output luma
    // to the 2x2 source min/max envelope, mixed by strength in [0,1]. Idempotent
    // at antiring=0: caller checks first and skips.
    RGY_ERR runAntiring(const RGYFrameInfo *pInputPlaneY,
                        const RGYFrameInfo *pOutputPlaneY,
                        float strength,
                        RGYOpenCLQueue &queue);

    // Shared dispatch context passed to each runMode* / runDarkenChain / runThinChain
    // / runDenoiseChain helper so the local variables from runPlaneY's setup
    // block do not appear in every call site.
    struct Anime4kDispatchCtx {
        RGYOpenCLQueue &queue;
        RGYWorkSize local_2d;
        RGYWorkSize local_x_pass;
        RGYWorkSize local_y_pass;
        RGYWorkSize global;      // outW x outH
        cl_mem srcImageMem;
        int srcW, srcH, outW, outH;
    };

    // Upscale/sharpen mode sub-dispatchers. Each holds the kernel-dispatch
    // block for one VppAnime4kMode value extracted from runPlaneY.
    RGY_ERR runModeOriginal(const Anime4kDispatchCtx &ctx, RGYFrameInfo *pOutputPlaneY,
                             const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR runModeDogSharpen(const Anime4kDispatchCtx &ctx, RGYFrameInfo *pOutputPlaneY, const RGYFrameInfo *pInputPlaneY,
                               const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR runModeDog(const Anime4kDispatchCtx &ctx, RGYFrameInfo *pOutputPlaneY, const RGYFrameInfo *pInputPlaneY,
                        const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR runModeDtd(const Anime4kDispatchCtx &ctx, RGYFrameInfo *pOutputPlaneY, const RGYFrameInfo *pInputPlaneY,
                        const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR runDarkenChain(const Anime4kDispatchCtx &ctx, RGYFrameInfo *pOutputPlaneY,
                            VppAnime4kDarken tier, RGYOpenCLEvent *event);
    RGY_ERR runThinChain(const Anime4kDispatchCtx &ctx, RGYFrameInfo *pOutputPlaneY,
                          VppAnime4kThin tier, RGYOpenCLEvent *event);
    RGY_ERR runDenoiseChain(const Anime4kDispatchCtx &ctx, RGYFrameInfo *pOutputPlaneY,
                             VppAnime4kDenoise tier, RGYOpenCLEvent *event);

    // Opt-in end-of-chain resize. Non-null only when anime4k.postResizeW/H > 0.
    // Instantiated and init'd in RGYFilterAnime4k::init() with frameIn = the
    // anime4k mode's output (scale*src) and frameOut = the requested target res;
    // invoked as the last sub-stage of run_filter. Reuses the full resampler
    // family (jinc/nis/lanczos/...) instead of duplicating resize code.
    std::unique_ptr<RGYFilterResize> m_postResize;
};

#endif //__RGY_FILTER_ANIME4K_H__
