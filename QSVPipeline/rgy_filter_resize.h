// -----------------------------------------------------------------------------------------
//     QSVEnc/VCEEnc/rkmppenc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2019-2021 rigaya
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
// IABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_FILTER_RESIZE_H__
#define __RGY_FILTER_RESIZE_H__

#include <array>
#include "rgy_filter_cl.h"

class RGYFilterParamLibplaceboResample;

class RGYFilterParamResize : public RGYFilterParam {
public:
    RGY_VPP_RESIZE_ALGO interp;
    float gaussP;
    std::shared_ptr<RGYFilterParamLibplaceboResample> libplaceboResample;
    VppResizeFsr1 fsr1;
    RGYFilterParamResize() : interp(RGY_VPP_RESIZE_AUTO), gaussP(2.0f), libplaceboResample(), fsr1() {};
    virtual ~RGYFilterParamResize() {};
};

class RGYFilterLibplaceboResample;

class RGYFilterResize : public RGYFilter {
public:
    RGYFilterResize(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterResize();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    struct RGYResizeGaussPlane {
        std::unique_ptr<RGYCLFrame> tmp;
        RGYResizeGaussPlane();
    };

    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    virtual RGY_ERR resizePlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    virtual RGY_ERR resizePlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const int plane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    virtual RGY_ERR resizePlaneFsr(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
        cl_mem midMem, int midPitchBytes, int midWidth, int midHeight,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    virtual RGY_ERR resizePlaneGauss2Pass(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const int plane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    virtual RGY_ERR resizeFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    virtual RGY_ERR createGaussTmp(RGYResizeGaussPlane& planeTmp, const RGYFrameInfo& planeOut, const RGYFrameInfo& planeIn);
    virtual void clearGaussTmp();

    bool m_bInterlacedWarn;
    std::unique_ptr<RGYCLBuf> m_weightSpline;
    std::array<RGYResizeGaussPlane, RGY_MAX_PLANES> m_gauss2pass;
    std::unique_ptr<RGYFilterLibplaceboResample> m_libplaceboResample;
    std::unique_ptr<RGYCLFrame> m_easuOutput;
    std::array<std::unique_ptr<RGYCLBuf>, RGY_MAX_PLANES> m_easuOutputF16;
    std::array<int, RGY_MAX_PLANES> m_easuOutputF16Width;
    std::array<int, RGY_MAX_PLANES> m_easuOutputF16Height;
    bool m_fp16Easu;
    RGYOpenCLProgramAsync m_resize;
    RGYCLFramePool m_srcImagePool;
};

#endif //__RGY_FILTER_RESIZE_H__
