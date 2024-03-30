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

#ifndef __RGY_FILTER_DENOISE_DCT_H__
#define __RGY_FILTER_DENOISE_DCT_H__

#include "rgy_filter_cl.h"
#include "rgy_prm.h"

class RGYFilterParamDenoiseDct : public RGYFilterParam {
public:
    VppDenoiseDct dct;
    RGYFilterParamDenoiseDct() : dct() {};
    virtual ~RGYFilterParamDenoiseDct() {};
    virtual tstring print() const;
};

class RGYFilterDenoiseDct : public RGYFilter {
public:
    RGYFilterDenoiseDct(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterDenoiseDct();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    RGY_ERR checkParam(const RGYFilterParamDenoiseDct *prm);

    RGY_ERR colorDecorrelation(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue);
    RGY_ERR colorCorrelation(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue);
    RGY_ERR denoiseDct(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue);
    RGY_ERR denoise(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    int m_step;
    float m_threshold;
    std::unique_ptr<RGYFilterCspCrop> m_srcCrop;
    std::unique_ptr<RGYFilterCspCrop> m_dstCrop;
    std::array<std::unique_ptr<RGYCLFrame>, 2> m_bufImg;
    RGYOpenCLProgramAsync m_dct;
};

#endif //__RGY_FILTER_DENOISE_DCT_H__

