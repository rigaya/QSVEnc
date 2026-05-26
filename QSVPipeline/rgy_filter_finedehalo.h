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
#ifndef __RGY_FILTER_FINEDEHALO_H__
#define __RGY_FILTER_FINEDEHALO_H__

#include "rgy_filter_cl.h"
#include "rgy_filter_dehalo.h"
#include "rgy_prm.h"

class RGYFilterParamFineDehalo : public RGYFilterParam {
public:
    VppFineDehalo finedehalo;

    RGYFilterParamFineDehalo() : finedehalo() {};
    virtual ~RGYFilterParamFineDehalo() {};
    virtual tstring print() const override { return finedehalo.print(); };
};

class RGYFilterFineDehalo : public RGYFilter {
public:
    RGYFilterFineDehalo(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterFineDehalo();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
                               RGYOpenCLQueue &queue_main, const std::vector<RGYOpenCLEvent> &wait_events,
                               RGYOpenCLEvent *event) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamFineDehalo> pParam);

    // Per-kernel host wrappers. All operate on the luma plane.
    RGY_ERR runPrewitt   (RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                          int thmiHbd, int thmaHbd,
                          RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR runLimitMask (RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, const RGYFrameInfo *pDehaloed,
                          int thlimiHbd, int thlimaHbd,
                          RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR runCombine   (RGYFrameInfo *pDst,
                          const RGYFrameInfo *pSrc, const RGYFrameInfo *pDehaloed,
                          const RGYFrameInfo *pEm,  const RGYFrameInfo *pLineMask,
                          int showmask,
                          RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                          RGYOpenCLEvent *event);

    // Reuses dehalo_expand / dehalo_inpand kernels at fixed radius 1 (3×3
    // neighbourhood). Source kept in rgy_filter_dehalo.cl; we build a
    // private program instance here to avoid reaching into the sub-filter.
    RGY_ERR runMorph3x3  (const char *kernelName,
                          RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                          RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);

    // Plane-level passthrough copy for chroma planes.
    RGY_ERR copyChromaPlanes(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                             RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                             RGYOpenCLEvent *event);

    // Programs.
    RGYOpenCLProgramAsync m_finedehalo;     // prewitt / limitmask / combine
    RGYOpenCLProgramAsync m_dehaloMorph;    // re-built dehalo.cl, only used for expand/inpand at r=1
    std::string           m_buildOptions;

    // Resolved edge-operator kernel name. Set in init() from
    // prm->finedehalo.edge; consumed by runPrewitt().
    // edge=sobel resolves to "finedehalo_prewitt" (the existing kernel uses
    // centre-weighted 8-tap math, which is identical to textbook Sobel —
    // the "prewitt" label is preserved purely for backward compatibility
    // with the original FineDehalo.avsi AviSynth script).
    std::string           m_edgeKernelName;

    // The DeHalo_alpha pass runs as a sub-filter; we read its output via
    // ppOutputFrames[0] after the call.
    std::unique_ptr<RGYFilterDehalo> m_dehalo;

    // Intermediate buffers (all source-resolution, luma-plane shape).
    std::unique_ptr<RGYCLFrame> m_edges;     // raw Prewitt + threshold ramp
    std::unique_ptr<RGYCLFrame> m_morphTmp;  // scratch between expand and inpand
    std::unique_ptr<RGYCLFrame> m_ey;        // pass-A morph result
    std::unique_ptr<RGYCLFrame> m_em;        // pass-B morph result (edge protection zone)
    std::unique_ptr<RGYCLFrame> m_linemask;  // limit mask
};

#endif // __RGY_FILTER_FINEDEHALO_H__
