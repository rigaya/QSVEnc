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
// AURORA inverse-model deflicker -- van Roosmalen 1999 PhD thesis,
// "Restoration of archived film and video".

#pragma once
#ifndef __RGY_FILTER_DEFLICKER_H__
#define __RGY_FILTER_DEFLICKER_H__

#include <deque>
#include "rgy_filter_cl.h"
#include "rgy_prm.h"

class RGYFilterParamDeflicker : public RGYFilterParam {
public:
    VppDeflicker deflicker;

    RGYFilterParamDeflicker() : deflicker() {};
    virtual ~RGYFilterParamDeflicker() {};
    virtual tstring print() const override { return deflicker.print(); };
};

class RGYFilterDeflicker : public RGYFilter {
public:
    RGYFilterDeflicker(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterDeflicker();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
                               RGYOpenCLQueue &queue_main, const std::vector<RGYOpenCLEvent> &wait_events,
                               RGYOpenCLEvent *event) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamDeflicker> pParam);

    // Reduce-and-readback one plane to (mean, stddev). Synchronous on
    // the readback (clEnqueueReadBuffer + event.wait()).
    RGY_ERR computePlaneStats(const RGYFrameInfo *pPlane, double &meanOut, double &stddevOut,
                              RGYOpenCLQueue &queue,
                              const std::vector<RGYOpenCLEvent> &wait_events);

    // Apply one (mult, add) correction pass. blend=strength on the last
    // pass; intermediate passes should use blend=1.0f (full correction).
    RGY_ERR runApply(RGYFrameInfo *pDstPlane, const RGYFrameInfo *pSrcPlane,
                     float mult, float add, float blend, int is_chroma,
                     RGYOpenCLQueue &queue,
                     const std::vector<RGYOpenCLEvent> &wait_events,
                     RGYOpenCLEvent *event);

    RGYOpenCLProgramAsync m_deflicker;
    std::string           m_buildOptions;

    // Intermediate frame buffer for the predictor-corrector's first
    // pass. Allocated only when predictor=on.
    std::unique_ptr<RGYCLFrame> m_intermediate;

    // Partial-sums buffers for the parallel reduction kernel. One long
    // per workgroup per buffer (sums in m_sumBuf, sums-of-squares in
    // m_sumSqBuf). Two buffers avoids the cl_buffer_region alignment
    // requirements of sub-buffers.
    std::unique_ptr<RGYCLBuf>   m_sumBuf;
    std::unique_ptr<RGYCLBuf>   m_sumSqBuf;
    std::vector<int64_t>        m_sumHost;
    std::vector<int64_t>        m_sumSqHost;
    size_t                      m_statsBufWGCount;

    // Rolling history of past (mean, sigma) over non-scene-change
    // frames. Cap is prm->frames; old entries drop off the back.
    std::deque<double>          m_rollingMeans;
    std::deque<double>          m_rollingSigmas;

    // Damping state from the previous frame's correction.
    double                      m_prevMult;
    double                      m_prevAdd;
    bool                        m_haveDamping; // false until first valid correction lands

    // Diagnostics
    int                         m_skippedSceneFrames;
};

#endif // __RGY_FILTER_DEFLICKER_H__
