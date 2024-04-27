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

#ifndef __RGY_FILTER_DENOISE_NLMEANS_H__
#define __RGY_FILTER_DENOISE_NLMEANS_H__

#include "rgy_filter_cl.h"
#include "rgy_prm.h"
#include <unordered_map>

// dxdyのペアを何並列で同時計算するか
static const int RGY_NLMEANS_DXDY_STEP = 8;

class RGYFilterParamDenoiseNLMeans : public RGYFilterParam {
public:
    VppNLMeans nlmeans;
    RGYFilterParamDenoiseNLMeans() : nlmeans() {};
    virtual ~RGYFilterParamDenoiseNLMeans() {};
    virtual tstring print() const override { return nlmeans.print(); };
};

class RGYFilterDenoiseNLMeans : public RGYFilter {
public:
    RGYFilterDenoiseNLMeans(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterDenoiseNLMeans();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    virtual RGY_ERR denoisePlane(
        RGYFrameInfo *pOutputPlane,
        RGYFrameInfo *pTmpUPlane, RGYFrameInfo *pTmpVPlane,
        RGYFrameInfo *pTmpIWPlane,
        const RGYFrameInfo *pInputPlane,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    virtual RGY_ERR denoiseFrame(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    std::unordered_map<int, std::unique_ptr<RGYOpenCLProgramAsync>> m_nlmeans;
    std::array<std::unique_ptr<RGYCLFrame>, 2 + 1 + RGY_NLMEANS_DXDY_STEP> m_tmpBuf;
};

#endif //__RGY_FILTER_DENOISE_KNN_H__
