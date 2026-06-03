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
#ifndef __RGY_FILTER_CHROMASHIFT_H__
#define __RGY_FILTER_CHROMASHIFT_H__

#include <vector>
#include "rgy_filter_cl.h"
#include "rgy_prm.h"

class RGYFilterParamChromaShift : public RGYFilterParam {
public:
    VppChromaShift chromashift;

    RGYFilterParamChromaShift() : chromashift() {};
    virtual ~RGYFilterParamChromaShift() {};
    virtual tstring print() const override { return chromashift.print(); };
};

class RGYFilterChromaShift : public RGYFilter {
public:
    RGYFilterChromaShift(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterChromaShift();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
                               RGYOpenCLQueue &queue_main, const std::vector<RGYOpenCLEvent> &wait_events,
                               RGYOpenCLEvent *event) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamChromaShift> pParam);

    // Shift one chroma plane via hardware-bilinear sampling from an
    // image2d_t view. shift_chroma is already in chroma-plane units.
    RGY_ERR runShiftPlane(RGYFrameInfo *pDstPlane, const RGYFrameInfo *pSrcImgPlane,
                          float shift_x_chroma, float shift_y_chroma,
                          RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                          RGYOpenCLEvent *event);

    // Diagnostic: |Laplacian(U)| -> Y plane (full luma resolution).
    RGY_ERR runLaplacianToLuma(RGYFrameInfo *pDstY, const RGYFrameInfo *pSrcC,
                                int subX, int subY,
                                RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                                RGYOpenCLEvent *event);
    RGY_ERR runFillNeutral(RGYFrameInfo *pDstPlane,
                           RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
                           RGYOpenCLEvent *event);

    // Auto-detect analysis-phase helpers. Sign maps live in plain
    // cl_mem buffers (one signed byte per luma pixel, no padding) so
    // signPitch == lumaWidth in all calls.
    RGY_ERR runLapSignY (cl_mem dstSignBuf, const RGYFrameInfo *pSrcY,
                         RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR runLapSignUV(cl_mem dstSignBuf,
                         const RGYFrameInfo *pSrcU, const RGYFrameInfo *pSrcV,
                         int lumaW, int lumaH, int subX, int subY,
                         RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR runCorrelate(cl_mem signYBuf, cl_mem signUVBuf,
                         int lumaW, int lumaH,
                         RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);

    RGYOpenCLProgramAsync m_chromashift;
    std::string           m_buildOptions;
    RGYCLFramePool        m_srcImagePool;

    // Auto-detect state.
    std::unique_ptr<RGYCLBuf> m_signY;   // Y Laplacian sign map (char, lumaW * lumaH)
    std::unique_ptr<RGYCLBuf> m_signUV;  // UV-combined Laplacian sign map (char, lumaW * lumaH)
    std::unique_ptr<RGYCLBuf> m_statsBuf;// 3 ints: [sum_dx, sum_dy, count]
    std::vector<int>          m_statsHost;
    std::vector<double>       m_acceptedDx;
    std::vector<double>       m_acceptedDy;
    int                       m_seenAnalysisFrames;
    int                       m_skippedAutoFrames;
    // Frames bypassed during the initial warm-up: rejected by the kernel
    // (count == 0) before any accepted frame existed. These don't count
    // against the hardCap budget so long intros / fade-ins don't exhaust
    // the budget before real content arrives. See run_filter() for the
    // accounting; bounded by a separate absolute safety cap.
    int                       m_warmupSkippedFrames;
    bool                      m_analysisComplete;
    float                     m_resolvedShiftX;
    float                     m_resolvedShiftY;
};

#endif // __RGY_FILTER_CHROMASHIFT_H__
