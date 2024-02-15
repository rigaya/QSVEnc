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

#include "rgy_filter_cl.h"
#include "rgy_prm.h"
#include "rgy_frame.h"
#include <array>

static const int VPP_SMOOTH_MAX_QUALITY_LEVEL = 6;

class RGYFilterParamSmooth : public RGYFilterParam {
public:
    VppSmooth smooth;
    RGYListRef<RGYFrameDataQP> *qpTableRef;

    RGYFilterParamSmooth() : smooth(), qpTableRef(nullptr) {};
    virtual ~RGYFilterParamSmooth() {};
    virtual tstring print() const override { return smooth.print(); };
};

class RGYFilterSmooth : public RGYFilter {
public:
    RGYFilterSmooth(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterSmooth();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    int qp_size(int res) { return divCeil(res + 15, 16); }
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    virtual RGY_ERR procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const RGYFrameInfo *targetQPTable, const int qpBlockShift, const float qpMul, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    virtual RGY_ERR procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *targetQPTable, const float qpMul, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    virtual RGY_ERR setQP(RGYCLFrame *targetQPTable, const int qp, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    float getQPMul(int qp_scale);

    RGYOpenCLProgramAsync m_smooth;
    unique_ptr<RGYCLFrame> m_qp;
    std::shared_ptr<RGYFrameDataQP> m_qpSrc;
    std::shared_ptr<RGYFrameDataQP> m_qpSrcB;
    RGYListRef<RGYFrameDataQP> *m_qpTableRef;
    int m_qpTableErrCount;
    RGYCLFramePool m_srcImagePool;
};
