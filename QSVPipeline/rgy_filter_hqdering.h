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
#ifndef __RGY_FILTER_HQDERING_H__
#define __RGY_FILTER_HQDERING_H__

#include "rgy_filter_cl.h"
#include "rgy_prm.h"

class RGYFilterParamDering : public RGYFilterParam {
public:
    VppDering dering;

    RGYFilterParamDering() : dering() {};
    virtual ~RGYFilterParamDering() {};
    virtual tstring print() const override { return dering.print(); };
};

class RGYFilterDering : public RGYFilter {
public:
    RGYFilterDering(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterDering();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
                               RGYOpenCLQueue &queue_main, const std::vector<RGYOpenCLEvent> &wait_events,
                               RGYOpenCLEvent *event) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamDering> pParam);

    // Per-kernel host wrappers. All take the plane to operate on; the work
    // buffers are allocated full-frame, so a getPlane() view gives each
    // plane the same processing chain (planes= option, default: Y only).
    RGY_ERR runEdge   (RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                       int mthrHbd, RGY_PLANE plane,
                       RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    // 3×3 morphological max — reused dehalo_expand, kept as a private rebuild
    // to avoid reaching into another sub-filter's program.
    RGY_ERR runExpand3x3(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, RGY_PLANE plane,
                         RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR runBlurH  (RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                       int radius, float sigma, RGY_PLANE plane,
                       RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR runBlurV  (RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                       int radius, float sigma, RGY_PLANE plane,
                       RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    // 3×3 morphological min (minp edge core).
    RGY_ERR runInpand3x3(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, RGY_PLANE plane,
                         RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    // 3×3 mean, RemoveGrain mode20 equivalent (msmooth / sharp>=2).
    RGY_ERR runMean3x3(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, RGY_PLANE plane,
                       RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    // 3×3 binomial blur, RemoveGrain mode11 equivalent (sharp).
    RGY_ERR runRg11   (RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, RGY_PLANE plane,
                       RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    // Contra-sharpen merge stage (sharp).
    RGY_ERR runContra (RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                       const RGYFrameInfo *pSmoothed, const RGYFrameInfo *pMethod, RGY_PLANE plane,
                       RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    // Repair mode1 equivalent: clamp blurred to src 3×3 min/max (drrep).
    RGY_ERR runRepair3x3(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                         const RGYFrameInfo *pBlurred, RGY_PLANE plane,
                         RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    // pCoreMask may be nullptr (minp=0): the kernel then gets a dummy
    // buffer and useCoreMask=0. thrHbd=0 disables the LimitFilter ramp.
    RGY_ERR runCombine(RGYFrameInfo *pDst,
                       const RGYFrameInfo *pSrc, const RGYFrameInfo *pBlurred, const RGYFrameInfo *pMask,
                       const RGYFrameInfo *pEdgeMask, const RGYFrameInfo *pCoreMask,
                       int showmask, int protect,
                       int thrHbd, int darkthrHbd, float elast, RGY_PLANE plane,
                       RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                       RGYOpenCLEvent *event);

    RGYOpenCLProgramAsync m_dering;        // hqdering_edge / blur_h / blur_v / combine
    RGYOpenCLProgramAsync m_dehaloMorph;   // private rebuild of dehalo.cl for expand at r=1
    std::string           m_buildOptions;

    // Resolved edge-operator kernel name. Set in init() from
    // prm->dering.edge; consumed by runEdge(). Default is the 5x5 LoG
    // (hqdering_edge_log); the original Sobel kernel is available via
    // edge=sobel (which resolves to "hqdering_edge").
    std::string           m_edgeKernelName;

    // Intermediate buffers (full-frame; each plane used via getPlane views).
    std::unique_ptr<RGYCLFrame> m_edgeMask;     // Sobel + threshold ramp output
    std::unique_ptr<RGYCLFrame> m_ringMask;     // dilated edge mask (ping side)
    std::unique_ptr<RGYCLFrame> m_morphTmp;     // dilation ping-pong (pong side)
    std::unique_ptr<RGYCLFrame> m_hBlurred;     // horizontal Gaussian pass output
    std::unique_ptr<RGYCLFrame> m_blurred;      // full Gaussian (after vertical pass)
    std::unique_ptr<RGYCLFrame> m_edgeCore;     // minp用: inpandしたエッジ芯 (minp>0のみ確保)
    std::unique_ptr<RGYCLFrame> m_maskTmp2;     // minp ping-pong (minp>0のみ確保)
    std::unique_ptr<RGYCLFrame> m_maskTmp3;     // msmooth ping-pong (msmooth>0のみ確保)
    std::unique_ptr<RGYCLFrame> m_contraTmp;    // sharp用: RG11/RG20チェーンと合成結果 (sharp>0のみ確保)
    std::unique_ptr<RGYCLFrame> m_contraTmp2;
};

#endif // __RGY_FILTER_HQDERING_H__
