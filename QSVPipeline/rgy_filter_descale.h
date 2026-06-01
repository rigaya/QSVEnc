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

#ifndef __RGY_FILTER_DESCALE_H__
#define __RGY_FILTER_DESCALE_H__

#include "rgy_filter_cl.h"
#include "rgy_prm.h"
#include <array>
#include <vector>

// Forward declarations so this header doesn't pull libavformat /
// libavcodec into every translation unit that consumes RGYFilterDescale;
// the full headers are included in the .cpp via rgy_avutil.h.
struct AVFormatContext;
struct AVStream;

// Inverse-kernel resampler: recovers a native lower-resolution image
// from an upscaled distribution by solving the linear system the
// upscaler applied. LDLT decomposition is performed on the host once
// at filter init; per-frame work is two kernels (horizontal then
// vertical) that each compute A' b, forward-substitute, and back-
// substitute per row/column.
class RGYFilterParamDescale : public RGYFilterParam {
public:
    VppDescale descale;
    tstring inputFilePath;   // absolute path to the input media file. Consumed
                             // by the auto-detect pre-decode probe (runProbe)
                             // when descale.autoDetect is true. Same pattern
                             // as RGYFilterParamIvtc::inputFilePath.
    // Optional trim window in coded-frame indices, populated by the
    // pipeline from --trim. When both are > 0, runProbe restricts its
    // seek targets to [probeStartFrame, probeEndFrame). Default 0/0
    // means "sample across the whole file". This makes
    //   --trim 1000:1100 --vpp-descale kernel=auto,width=...,height=...
    // sample the high-detail mid-episode window the user picked,
    // instead of dragging spread-out seeks across the whole runtime.
    int probeStartFrame;
    int probeEndFrame;
    RGYFilterParamDescale() : descale(), inputFilePath(),
                              probeStartFrame(0), probeEndFrame(0) {};
    virtual ~RGYFilterParamDescale() {};
    virtual tstring print() const override { return descale.print(); };
};

// Per-direction compact GPU-resident state. Built once at filter init
// from the host-side LDLT decomposition output (see prepareCore in
// the .cpp). All buffers are read-only on the device.
class RGYFilterDescaleCore {
public:
    RGYFilterDescaleCore() : src_dim(0), dst_dim(0), bandwidth(0), c(0), weights_columns(0),
        weights(), left_idx(), right_idx(), lower(), upper(), diagonal() {}
    ~RGYFilterDescaleCore() {}

    int src_dim;
    int dst_dim;
    int bandwidth;          // support * 4 - 1
    int c;                  // bandwidth / 2 (substitution lookback)
    int weights_columns;    // max non-zero entries per output row

    std::unique_ptr<RGYCLBuf> weights;     // float[dst_dim * weights_columns]
    std::unique_ptr<RGYCLBuf> left_idx;    // int[dst_dim]
    std::unique_ptr<RGYCLBuf> right_idx;   // int[dst_dim]
    std::unique_ptr<RGYCLBuf> lower;       // float[c * dst_dim]
    std::unique_ptr<RGYCLBuf> upper;       // float[c * dst_dim]
    std::unique_ptr<RGYCLBuf> diagonal;    // float[dst_dim]
};

class RGYFilterDescale : public RGYFilter {
public:
    RGYFilterDescale(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterDescale();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    RGY_ERR prepareCore(RGYFilterDescaleCore &core, int src_dim, int dst_dim,
                        VppDescaleKernel kernel, double b, double c_param,
                        double shift, VppDescaleBorder border);

    // Auto-detect probe: open a private libavcodec context against
    // inputFilePath, decode detect_frames frames, score each candidate
    // (10 kernels at the user-supplied width/height for v1) via
    // descale + re-upscale + MSE, lock the kernel with minimum
    // accumulated MSE. Rewrites prm->descale.kernel/.b/.c to the
    // winner and clears prm->descale.autoDetect.
    RGY_ERR runProbe(RGYFilterParamDescale *prm);

    // Sparse forward-upscale weight matrix in compressed-by-row form.
    // Built per candidate at probe time; consumed by kernel_rescale_h
    // and kernel_rescale_v. Separate from the LDLT descale weights
    // (which are A^T compressed by low-res row); these are A compressed
    // by high-res row, the natural form for the forward operation.
    struct ProbeForwardWeights {
        int weights_columns;
        std::unique_ptr<RGYCLBuf> weights;
        std::unique_ptr<RGYCLBuf> left_idx;
        std::unique_ptr<RGYCLBuf> right_idx;
    };

    // Per-candidate state used by the runProbe scoring loop. The scoring
    // path is: orig luma -> descale H -> descale V -> rescale H ->
    // rescale V -> MSE row sums -> reduce. Accumulated across all
    // decoded frames.
    struct ProbeCandidate {
        VppDescaleKernel kernel;
        float b;
        float c;
        int   width;
        int   height;
        double mse;        // filled in by scoreCandidates
        tstring label;
    };

    RGY_ERR buildForwardWeights(ProbeForwardWeights &fw, int src_dim_low, int dst_dim_high,
                                VppDescaleKernel kernel, double b, double c_param,
                                double shift, VppDescaleBorder border);

    // Score every entry in `candidates` against the uploaded luma
    // frames. For each candidate this builds the LDLT cores + forward
    // weights at its width/height, runs the descale -> rescale -> MSE
    // chain, and writes the resulting per-pixel MSE back into
    // `candidate.mse`. All per-candidate device buffers (cores,
    // scratch) are torn down before moving on to the next, so peak
    // VRAM stays bounded by one candidate's footprint regardless of
    // how many are in the list.
    // Score every entry against the uploaded luma frames. Default
    // forward (re-upscale) kernel is fixed to Catmull-Rom for the
    // resolution-detection passes (Stage 1) so wide-tap candidate
    // kernels don't reinject their own ringing into the residual and
    // bias height detection. The Stage 2 kernel tie-break re-scores at
    // the locked dimensions with symmetricForward=true, which restores
    // the natural inverse-problem pairing (forward kernel == candidate
    // descale kernel) for the kernel pick alone.
    RGY_ERR scoreCandidates(std::vector<ProbeCandidate> &candidates,
                            const std::vector<std::unique_ptr<RGYCLBuf>> &lumaBufs,
                            const std::vector<std::unique_ptr<RGYCLBuf>> &edgeWeightsBufs,
                            int src_w, int src_h, int src_pixel_bytes,
                            bool symmetricForward = false);

    // Three-pass coarse-to-fine pyramid that finds both the native
    // height and the upscale kernel for an unknown-resolution source.
    // Runs only when descale.width/height are 0 (user passed auto=true
    // without explicit dims). On return, `candidates` holds the union
    // of all probed candidates with their per-pixel MSEs filled in,
    // and the lowest-MSE entry is the winning configuration. The
    // caller (runProbe) then writes prm->descale.kernel/.b/.c/.width/
    // .height from that winner.
    RGY_ERR runResolutionSearch(RGYFilterParamDescale *prm,
                                std::vector<ProbeCandidate> &candidates,
                                const std::vector<std::unique_ptr<RGYCLBuf>> &lumaBufs,
                                const std::vector<std::unique_ptr<RGYCLBuf>> &edgeWeightsBufs,
                                int src_w, int src_h, int src_pixel_bytes,
                                AVFormatContext *fmtCtx, AVStream *videoStream);
    RGY_ERR runHPlane(RGYFrameInfo *pIntermediateFloat, const RGYFrameInfo *pInputPlane,
                      const RGYFilterDescaleCore &core,
                      RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR runVPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pIntermediateFloat,
                      RGYCLBuf *pVScratch,
                      const RGYFilterDescaleCore &core,
                      RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    RGYOpenCLProgramAsync m_descale;
    // Cores indexed [direction (0=H, 1=V)][plane group (0=luma, 1=chroma)].
    std::array<std::array<RGYFilterDescaleCore, 2>, 2> m_cores;
    // Per-plane float intermediate (dst_w x src_h) for H output, plus
    // per-plane float scratch (dst_w x dst_h) used by V kernel.
    std::array<std::unique_ptr<RGYCLBuf>, 3> m_intermediateH;
    std::array<std::unique_ptr<RGYCLBuf>, 3> m_intermediateV;
    std::array<int, 3> m_intermediatePitchFloats;
    int m_frameIdx;
};

#endif //__RGY_FILTER_DESCALE_H__
