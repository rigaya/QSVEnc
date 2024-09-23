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
#ifndef __RGY_FILTER_CL_H__
#define __RGY_FILTER_CL_H__

#include <cstdint>
#include "rgy_util.h"
#include "rgy_log.h"
#include "rgy_filter.h"
#include "rgy_opencl.h"
#include "convert_csp.h"
#include "rgy_prm.h"

class RGYFilterPerfCL : public RGYFilterPerf {
public:
    RGYFilterPerfCL() : RGYFilterPerf() {};
    virtual ~RGYFilterPerfCL() { };

    virtual RGY_ERR checkPerformace(void *event_start, void *event_fin) override;
protected:
};

class RGYFilter : public RGYFilterBase {
public:
    RGYFilter(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilter();
    RGY_ERR filter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum);
    RGY_ERR filter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue);
    RGY_ERR filter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, RGYOpenCLEvent *event);
    RGY_ERR filter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event = nullptr);

    virtual void setCheckPerformance(const bool check) override;
protected:
    virtual RGY_ERR AllocFrameBuf(const RGYFrameInfo &frame, int frames) override;
    RGY_ERR filter_as_interlaced_pair(const RGYFrameInfo *pInputFrame, RGYFrameInfo *pOutputFrame);
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) = 0;

    std::shared_ptr<RGYOpenCLContext> m_cl;
    std::vector<unique_ptr<RGYCLFrame>> m_frameBuf;
    std::unique_ptr<RGYCLFrame> m_pFieldPairIn;
    std::unique_ptr<RGYCLFrame> m_pFieldPairOut;
};

class RGYFilterDisabled : public RGYFilter {
public:
    RGYFilterDisabled(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context) {};
    virtual ~RGYFilterDisabled() {};
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
protected:
    virtual void close() override;
};

class RGYFilterParamCrop : public RGYFilterParam {
public:
    sInputCrop crop;
    CspMatrix matrix;

    RGYFilterParamCrop() : crop(initCrop()), matrix(RGY_MATRIX_ST170_M) {};
    virtual ~RGYFilterParamCrop() {};
};

class RGYFilterCspCrop : public RGYFilter {
public:
    RGYFilterCspCrop(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterCspCrop();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    RGY_ERR convertYBitDepth(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR convertCspFromNV12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR convertCspFromYV12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR convertCspFromNV16(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR convertCspFromRGB(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR convertCspFromYUV444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR convertCspFromAYUVPacked444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    virtual void close() override;
};

class RGYFilterParamPad : public RGYFilterParam {
public:
    VppPad pad;
    RGY_CSP encoderCsp;
    RGYFilterParamPad() : pad(), encoderCsp(RGY_CSP_NA) {};
    virtual ~RGYFilterParamPad() {};
    virtual tstring print() const override;
};

class RGYFilterPad : public RGYFilter {
public:
    RGYFilterPad(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterPad();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    virtual RGY_ERR procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, int pad_color, const VppPad &pad, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    virtual RGY_ERR procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    RGYOpenCLProgramAsync m_pad;
    bool m_bInterlacedWarn;
};

#endif //__RGY_FILTER_CL_H__
