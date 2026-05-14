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

#pragma once
#ifndef __RGY_FILTER_MAA_H__
#define __RGY_FILTER_MAA_H__

#include <array>
#include "rgy_filter_cl.h"
#include "rgy_prm.h"

class RGYFilterParamMaa : public RGYFilterParam {
public:
    VppMaa maa;

    RGYFilterParamMaa() : maa() {};
    virtual ~RGYFilterParamMaa() {};
    virtual tstring print() const override { return maa.print(); };
};

class RGYFilterMaa : public RGYFilter {
public:
    RGYFilterMaa(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterMaa();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
                               RGYOpenCLQueue &queue_main, const std::vector<RGYOpenCLEvent> &wait_events,
                               RGYOpenCLEvent *event) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamMaa> pParam);

    // Per-plane FTurn dispatch helpers. dst dimensions (width × height) must
    // equal (src.height × src.width) — the rotation swaps axes.
    // planeCount caps the number of planes processed (default -1 → all planes
    // in src's CSP). Pass 1 to FTurn only the luma plane when chroma is
    // unused, saving the chroma kernel dispatches.
    RGY_ERR fturnLeftFrame(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                           RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                           int planeCount = -1);
    RGY_ERR fturnRightFrame(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                            RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                            int planeCount = -1);

    // Run one full SangNom pass (prepare + 9× smooth + finalize) on a single
    // plane (luma or chroma) of the given source, writing to the matching
    // plane of dst. Caller chooses the plane via `plane` and passes the
    // bit-depth-scaled threshold (m_aaf for luma, m_aacf for chroma).
    RGY_ERR sangnomPassPlane(const RGYFrameInfo *pSrc, RGYFrameInfo *pDst,
                             RGY_PLANE plane, float aaf,
                             RGYOpenCLQueue &queue,
                             const std::vector<RGYOpenCLEvent> &wait_events);

    // Edge-mask construction: simplified Sobel + inflate, both at source
    // luma resolution. mthreshScaled is the bit-depth-scaled threshold.
    RGY_ERR runEdgeSobel(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                         int mthreshScaled, RGYOpenCLQueue &queue,
                         const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR runInflate(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                       RGYOpenCLQueue &queue,
                       const std::vector<RGYOpenCLEvent> &wait_events);

    // Mask-gated alpha-blend merge for one plane. mask must already match
    // the dst plane's dimensions (caller is responsible for sub-sampling
    // when the chroma plane resolution differs from the luma mask).
    RGY_ERR runMergePlane(RGYFrameInfo *pDst, const RGYFrameInfo *pSrcA,
                          const RGYFrameInfo *pSrcB, const RGYFrameInfo *pMask,
                          RGY_PLANE plane,
                          RGYOpenCLQueue &queue,
                          const std::vector<RGYOpenCLEvent> &wait_events);

    // Sub-sample the luma-resolution mask down to chroma resolution.
    // Reads from the luma plane of pLumaMaskSrc and writes to the
    // chroma plane (U) of pChromaMaskDst. (V plane reuses U because
    // mask values are colour-channel agnostic.)
    RGY_ERR runMaskSubsample(RGYFrameInfo *pChromaMaskDst,
                             const RGYFrameInfo *pLumaMaskSrc,
                             RGYOpenCLQueue &queue,
                             const std::vector<RGYOpenCLEvent> &wait_events);

    // Show-mode overlay (one plane). Half-darkens pSrc and adds halved
    // mask, producing `dst = (src >> 1) + (mask >> 1)`. Used for the luma
    // plane only in show=1 / show=2 paths.
    RGY_ERR runShowOverlay(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                            const RGYFrameInfo *pMask, RGY_PLANE plane,
                            RGYOpenCLQueue &queue,
                            const std::vector<RGYOpenCLEvent> &wait_events);

    // Show-mode darken (one plane). `dst = src >> 1`. Used for chroma
    // planes in show mode so the AA-region overlay on luma stands out
    // against a dimmed background.
    RGY_ERR runShowDarken(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                          RGY_PLANE plane,
                          RGYOpenCLQueue &queue,
                          const std::vector<RGYOpenCLEvent> &wait_events);

    // Compiled OpenCL program holding the MAA kernels. Build options are
    // stashed alongside so we can rebuild only when bit-depth or block size
    // changes.
    RGYOpenCLProgramAsync m_maa;
    std::string           m_maaBuildOptions;

    // Spline36 resize sub-filters. m_resizeUp lifts the source-resolution
    // input to the supersampled (ssW × ssH) working surface; m_resizeDown
    // brings the AA result back to source resolution. Both are constructed
    // and init()'d in init() and torn down in close().
    //
    // m_resizeUpLuma / m_resizeDownLuma are luma-only (Y8/Y16 CSP) variants
    // used when chroma=false. The chroma planes are then overridden by an
    // input-chroma copy at the end of run_filter, so resizing chroma at all
    // is wasted work. These pointers are null when chroma=true.
    std::unique_ptr<RGYFilter> m_resizeUp;
    std::unique_ptr<RGYFilter> m_resizeDown;
    std::unique_ptr<RGYFilter> m_resizeUpLuma;
    std::unique_ptr<RGYFilter> m_resizeDownLuma;

    // Intermediate frame buffers used by the MAA pipeline. Allocated lazily
    // in init() so subsequent prompts can populate the AA pass without
    // touching this skeleton's wiring.
    //   m_supersampled  — output of m_resizeUp (ssW × ssH, upright)
    //   m_rotated       — supersampled + 90° turn-left (ssH × ssW)
    //   m_rotatedAA     — m_rotated after one SangNom pass (ssH × ssW)
    //   m_unrotatedAA   — m_rotatedAA + 90° turn-right (ssW × ssH)
    //   m_aaResult      — m_unrotatedAA after second SangNom pass (ssW × ssH)
    //   m_edgeMask      — edge mask at source resolution
    std::unique_ptr<RGYCLFrame> m_supersampled;
    std::unique_ptr<RGYCLFrame> m_rotated;
    std::unique_ptr<RGYCLFrame> m_rotatedAA;
    std::unique_ptr<RGYCLFrame> m_unrotatedAA;
    std::unique_ptr<RGYCLFrame> m_aaResult;
    // Edge-mask path buffers. m_edgeMask holds the raw simplified-Sobel
    // mask; m_inflatedMask holds the post-inflate mask (this is the one
    // consumed by the merge stage). m_chromaMask is allocated only when
    // chroma=true AND the chroma plane is sub-sampled vs luma; for YV24
    // (no sub-sampling) the host reuses m_inflatedMask directly.
    std::unique_ptr<RGYCLFrame> m_edgeMask;
    std::unique_ptr<RGYCLFrame> m_inflatedMask;
    std::unique_ptr<RGYCLFrame> m_chromaMask;

    // SangNom2 cost buffers: 9 cost slices + 9 smoothed slices held in a
    // ping-pong pair so the smooth kernel can read from one set and write
    // into the other.
    //
    // PACKED LAYOUT (since 2026-05-10): the 9 sub-buffers are stored as
    // contiguous slices inside a single `RGYCLBuf` allocation. Slice `i`
    // starts at byte offset `i * m_costSliceBytes` from the base pointer.
    // Both `m_costRawPacked` and `m_costSmoothPacked` share this layout,
    // which lets the prepare and finalize kernels take ONE buffer arg
    // instead of nine — fewer allocations, fewer kernel-arg slots, and
    // a tighter memory footprint.
    //
    // Each slice is sized to cover the LARGER of the two SangNom pass
    // shapes (pass 1 operates at ssH × ssW/2, pass 2 at ssW × ssH/2;
    // both have the same total area = ssW * ssH / 2 elements). The
    // kernels treat each slice as 2-D via (width, height, pitch)
    // parameters passed per dispatch.
    static constexpr int MAA_NUM_COST_BUFFERS = 9;
    std::unique_ptr<RGYCLBuf> m_costRawPacked;     // 9 contiguous slices, each m_costSliceBytes
    std::unique_ptr<RGYCLBuf> m_costSmoothPacked;  // 9 contiguous slices, same shape
    int  m_costPitch;       // bytes per row inside a slice (= max(ssW, ssH) * sizeof(Type))
    int  m_costSliceBytes;  // bytes per slice (= m_costPitch * max(ssW, ssH)/2)
    int  m_costElemBytes;   // sizeof(Type) — 1 for 8-bit, 2 for HBD

    // Derived constants computed at init() time. ssW/ssH are the supersampled
    // dimensions, rounded to the nearest multiple of 4 to keep SangNom2's
    // 2-row processing alignment and SIMD/work-group-friendly widths
    // (per analysis/maa2_investigation/06_formulas_and_constants.md § 5).
    // aaf and aacf are the bit-depth-scaled luma/chroma AA thresholds
    // consumed by the per-direction cost gate. mthreshScaled is the
    // bit-depth-scaled edge threshold.
    int   m_ssW;
    int   m_ssH;
    float m_aaf;
    float m_aacf;
    int   m_mthreshScaled;
};

#endif // __RGY_FILTER_MAA_H__
