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
#ifndef __RGY_FILTER_DEHALO_H__
#define __RGY_FILTER_DEHALO_H__

#include "rgy_filter_cl.h"
#include "rgy_prm.h"

class RGYFilterParamDehalo : public RGYFilterParam {
public:
    VppDehalo dehalo;

    RGYFilterParamDehalo() : dehalo() {};
    virtual ~RGYFilterParamDehalo() {};
    virtual tstring print() const override { return dehalo.print(); };
};

class RGYFilterDehalo : public RGYFilter {
public:
    RGYFilterDehalo(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterDehalo();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
                               RGYOpenCLQueue &queue_main, const std::vector<RGYOpenCLEvent> &wait_events,
                               RGYOpenCLEvent *event) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamDehalo> pParam);

    // Per-kernel host wrappers. All operate on the luma plane of the given
    // RGYFrameInfo. width/height are read from the frame plane info.
    RGY_ERR runExpand (RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                       float rx, float ry,
                       RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR runInpand (RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                       float rx, float ry,
                       RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR runMask   (RGYFrameInfo *pMaskDst,
                       const RGYFrameInfo *pSrc, const RGYFrameInfo *pExpand, const RGYFrameInfo *pInpand,
                       int loScaled, int hiScaled,
                       RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR runApply  (RGYFrameInfo *pDst,
                       const RGYFrameInfo *pSrc, const RGYFrameInfo *pExpand, const RGYFrameInfo *pInpand,
                       const RGYFrameInfo *pMask,
                       float darkstr, float brightstr,
                       RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                       RGYOpenCLEvent *event);

    RGY_ERR copyChromaPlanes(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                             RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                             RGYOpenCLEvent *event);

    RGYOpenCLProgramAsync m_dehalo;
    std::string           m_buildOptions;

    // Spline36 resize sub-filters. Allocated only when ss > 1.0; for ss == 1.0
    // the filter operates at native resolution and these stay null.
    std::unique_ptr<RGYFilter> m_resizeUp;
    std::unique_ptr<RGYFilter> m_resizeDown;

    // Intermediate frame buffers. When ss > 1.0 they live at supersampled
    // resolution (m_ssW × m_ssH); when ss == 1.0 they live at source
    // resolution.
    //   m_supersampled — output of m_resizeUp (only when ss > 1)
    //   m_expanded     — elliptic local max
    //   m_inpand       — elliptic local min
    //   m_mask         — sensitivity-ramp mask in [0, max_val]
    //   m_corrected    — apply output (only when ss > 1; at ss == 1 we
    //                    write directly into m_frameBuf[0])
    std::unique_ptr<RGYCLFrame> m_supersampled;
    std::unique_ptr<RGYCLFrame> m_expanded;
    std::unique_ptr<RGYCLFrame> m_inpand;
    std::unique_ptr<RGYCLFrame> m_mask;
    std::unique_ptr<RGYCLFrame> m_corrected;

    // Cached state.
    int  m_ssW;       // supersampled width  (== sourceW when ss == 1.0)
    int  m_ssH;       // supersampled height (== sourceH when ss == 1.0)
    bool m_ssActive;  // true when ss > 1.0 (resize-up/down dispatched)
};

#endif // __RGY_FILTER_DEHALO_H__
