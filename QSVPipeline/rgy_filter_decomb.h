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
#include <array>

class RGYFilterParamDecomb : public RGYFilterParam {
public:
    VppDecomb decomb;
    rgy_rational<int> timebase;
    RGYFilterParamDecomb() : decomb(), timebase() {};
    virtual ~RGYFilterParamDecomb() {};
    virtual tstring print() const override { return decomb.print(); };
};

class RGYFilterDecomb : public RGYFilter {
public:
    RGYFilterDecomb(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterDecomb();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamDecomb> prm);

    RGY_ERR createMotionMap(
        RGYFrameInfo *pDmaskPlane,
        RGYFrameInfo *pFmaskPlane,
        const RGYFrameInfo *pSrcPlane,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);

    RGY_ERR isCombed(
        RGYCLBuf *pResultIsCombed,
        const RGYFrameInfo *pFmaskPlane,
        RGYOpenCLQueue &queue);

    RGY_ERR deinterlacePlane(
        RGYFrameInfo *pDstPlane,
        const RGYFrameInfo *pSrcPlane,
        const RGYFrameInfo *pDmaskPlane,
        const RGYCLBuf *pResultIsCombed,
        const bool uv420, RGYOpenCLQueue &queue, RGYOpenCLEvent *event
    );

    virtual RGY_ERR procFrame(
        RGYFrameInfo *pOutputFrame,
        RGYFrameInfo *pDmaskFrame,
        RGYFrameInfo *pFmaskFrame,
        RGYCLBuf *pResultIsCombed,
        const RGYFrameInfo *pSrcFrame,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    RGYOpenCLProgramAsync m_decomb;
    std::unique_ptr<RGYCLFrame> m_dmask, m_fmask;
    std::unique_ptr<RGYCLBuf> m_isCombed;
};
