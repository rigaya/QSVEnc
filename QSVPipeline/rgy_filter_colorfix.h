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

// ----- EXTERNAL-ALGORITHM NOTICE -----
// ColorFix — colour correction filter (three modes)
//
// mode=manual: standard white-balance lift / gain in RGB space.
// mode=auto:   per-frame chroma mean reduction (YUV space).
// mode=gray:   grayworld colour-constancy assumption (Buchsbaum 1980).
// --------------------------------------

#pragma once
#ifndef __RGY_FILTER_COLORFIX_H__
#define __RGY_FILTER_COLORFIX_H__

#include "rgy_filter_cl.h"
#include "rgy_prm.h"

class RGYFilterParamColorFix : public RGYFilterParam {
public:
    VppColorFix colorfix;
    VideoVUIInfo vui;             // populated by qsv_pipeline.cpp from input metadata
    tstring      inputFilePath;   // populated by qsv_pipeline.cpp; used by the
                                  // init-time libav pre-scan for mode=auto/gray-yuv
                                  // (empty = no pre-scan; ramp fallback at runtime).

    RGYFilterParamColorFix() : colorfix(), vui(), inputFilePath() {};
    virtual ~RGYFilterParamColorFix() {};
    virtual tstring print() const override { return colorfix.print(); };
};

class RGYFilterColorFix : public RGYFilter {
public:
    RGYFilterColorFix(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterColorFix();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
                               RGYOpenCLQueue &queue_main, const std::vector<RGYOpenCLEvent> &wait_events,
                               RGYOpenCLEvent *event) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamColorFix> pParam);

    // Resolve the effective matrix at init time using the priority:
    // (1) user CLI override → (2) frame metadata → (3) resolution fallback.
    // Returns one of VPP_COLORFIX_MATRIX_{BT601,BT709,BT2020}; never AUTO.
    int  resolveMatrix(const VppColorFix &cf, const VideoVUIInfo &vui, int height) const;

    // Resolve the effective working colour space. Returns RGB or YUV;
    // never AUTO. AUTO -> { manual=RGB, auto=YUV, gray=YUV }.
    int  resolveSpace(const VppColorFix &cf) const;

    // BT.601/709/2020 luma coefficients (Kr, Kg, Kb).
    void getMatrixCoeffs(int resolvedMatrix, float &Kr, float &Kg, float &Kb) const;

    // Build the YUV<->RGB conversion sub-filters (RGYFilterCspCrop) when
    // the colour-space round-trip is needed (manual + gray modes).
    RGY_ERR setupCspConverters(const RGYFrameInfo &frameIn, RGY_CSP cspRgb, rgy_rational<int> baseFps);

    // Apply the RGB-space lift/gain correction (manual or gray).
    RGY_ERR runApplyRGB(RGYFrameInfo *pTarget,
                        float scaleR, float scaleG, float scaleB,
                        float offsetR, float offsetG, float offsetB,
                        RGYOpenCLQueue &queue,
                        const std::vector<RGYOpenCLEvent> &wait_events);

    // Reduce a YUV frame to (sum_U, sum_V, sum_Y, sumSq_Y) per work-group.
    RGY_ERR runReduceUV(RGYFrameInfo *pSrc,
                        RGYOpenCLQueue &queue,
                        const std::vector<RGYOpenCLEvent> &wait_events);
    // Reduce an RGB frame to (sum_R, sum_G, sum_B, sum_Y, sumSq_Y) per work-group.
    RGY_ERR runReduceRGB(RGYFrameInfo *pSrc,
                         RGYOpenCLQueue &queue,
                         const std::vector<RGYOpenCLEvent> &wait_events);

    // Apply the per-frame UV offset (auto mode).
    RGY_ERR runApplyUV(RGYFrameInfo *pTarget,
                       int offsetU, int offsetV,
                       RGYOpenCLQueue &queue,
                       const std::vector<RGYOpenCLEvent> &wait_events);

    // Apply per-plane Y scale+offset (manual YUV-space mode).
    RGY_ERR runApplyLuma(RGYFrameInfo *pTarget,
                         float scaleY, float offsetY,
                         RGYOpenCLQueue &queue,
                         const std::vector<RGYOpenCLEvent> &wait_events);

    // Read back the partial-sum buffer to the host and reduce it.
    // For auto-mode: outputs (sumU, sumV, sumY, sumSqY). For gray-mode:
    // outputs (sumR, sumG, sumB, sumY, sumSqY). Returns the number of
    // pixels covered.
    RGY_ERR finaliseReduction(RGYOpenCLQueue &queue, int numLongsPerGroup,
                              std::vector<long long> &out_totals);

    // Init-time libav pre-scan for mode=auto and mode=gray (yuv-space
    // path). Opens a private AVFormatContext against
    // prm->inputFilePath, sequentially decodes the first frames= frames,
    // accumulates chroma sums on the host CPU, and locks in m_offsetU /
    // m_offsetV before the first encode frame arrives. On success the
    // runtime path applies the correction from frame 1 with no ramp and
    // no discontinuity. On pipe / non-file inputs, libav error, or
    // unsupported pixel format the function returns RGY_ERR_UNSUPPORTED
    // and the caller leaves m_prescanUsed=false so that run_filter()
    // falls back to the streaming pre-roll with strength ramp.
    RGY_ERR runPreScanLibav(const std::shared_ptr<RGYFilterParamColorFix> &prm);

    // OpenCL programs.
    RGYOpenCLProgramAsync m_colorfix;        // our own kernels
    std::string           m_buildOptionsYUV; // build options for YUV-side kernels
    std::string           m_buildOptionsRGB; // build options for RGB-side kernels (Type=uchar4/ushort4)
    int                   m_resolvedMatrix;  // VPP_COLORFIX_MATRIX_{BT601,BT709,BT2020}
    int                   m_effectiveSpace;  // VPP_COLORFIX_SPACE_{RGB,YUV} (never AUTO)

    // YUV<->RGB conversion sub-filters (created lazily for manual + gray).
    std::unique_ptr<RGYFilter> m_convToRgb;   // input YUV -> internal RGB
    std::unique_ptr<RGYFilter> m_convToYuv;   // internal RGB -> output YUV
    RGY_CSP                    m_cspRgb;      // the intermediate RGB CSP

    // Partial-sum buffer (per work-group totals). Sized for the largest
    // analysis kernel reduction we'll do (5 long longs per work-group
    // is the upper bound, for gray-mode reduce_rgb).
    std::unique_ptr<RGYCLBuf> m_reducePartials;
    int                       m_numGroupsLastDispatch;

    // State machine for auto / gray analysis.
    bool      m_analysisComplete;    // true once correction has been computed
    int       m_analysedFrames;      // number of frames that contributed to the sums
    int       m_skippedFrames;       // flash / fade frames dropped by the variance guard
    int       m_totalSeenFrames;     // analysed + skipped (for warning when clip is too short)
    long long m_sumA;                // accumulators (U or R; meaning depends on mode)
    long long m_sumB;                // V or G
    long long m_sumC;                // unused (auto) or B (gray)
    long long m_sumY;                // luma sum (for variance guard)
    long long m_sumYsq;              // luma sum-of-squares (for variance guard)
    double    m_rollingVarianceSum;  // running average of per-frame luma variance
    int       m_rollingVarianceCount;

    // Computed correction values (set once analysis completes).
    int   m_offsetU;        // auto-mode chroma offsets, HBD-scaled
    int   m_offsetV;
    float m_scaleR;         // gray-mode RGB scales
    float m_scaleG;
    float m_scaleB;

    // Init-time libav pre-scan + runtime ramp state.
    // m_prescanUsed: true when runPreScanLibav populated m_offsetU /
    //   m_offsetV before runtime. The auto / gray-yuv runtime paths
    //   then skip analysis entirely and apply from frame 1.
    // m_hardCapFrames: safety timeout for the streaming pre-roll. When
    //   the variance guard keeps rejecting frames (e.g. persistent
    //   flicker / strobe) and m_analysedFrames hasn't reached frames=
    //   by m_totalSeenFrames >= m_hardCapFrames, the runtime locks
    //   in whatever running offset has been accumulated so the rest of
    //   the clip is not left uncorrected. Mirrors the chromashift
    //   safety net at rgy_filter_chromashift.cpp:379.
    bool  m_prescanUsed;
    int   m_hardCapFrames;
};

#endif // __RGY_FILTER_COLORFIX_H__
