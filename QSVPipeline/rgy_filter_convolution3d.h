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

#ifndef __RGY_FILTER_CONVOLUTION3D_H__
#define __RGY_FILTER_CONVOLUTION3D_H__

#include "rgy_filter.h"
#include "rgy_prm.h"
#include <array>

class RGYFilterParamConvolution3D : public RGYFilterParam {
public:
    VppConvolution3d convolution3d;
    RGYFilterParamConvolution3D() : convolution3d() {};
    virtual ~RGYFilterParamConvolution3D() {};
    virtual tstring print() const override { return convolution3d.print(); };
};

class RGYFilterConvolution3D : public RGYFilter {
public:
    RGYFilterConvolution3D(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterConvolution3D();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    virtual RGY_ERR denoisePlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pPrevPlane, const RGYFrameInfo *pInputPlane, const RGYFrameInfo *pNextPlane, const float threshold_spatial, const float threshold_temporal, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    virtual RGY_ERR denoiseFrame(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pPrevFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pNextFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    bool m_bInterlacedWarn;
    RGYOpenCLProgramAsync m_convolution3d;
    std::array<std::unique_ptr<RGYCLFrame>, 2> m_prevFrames;
    int m_cacheIdx;
    int m_frameOut;
};

#endif //__RGY_FILTER_CONVOLUTION3D_H__
