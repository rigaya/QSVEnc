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

#pragma once

#include "rgy_filter_cl.h"
#include "rgy_prm.h"
#include <array>

class RGYFilterParamVinverse : public RGYFilterParam {
public:
    VppVinverse vinverse;
    RGYFilterParamVinverse() : vinverse() {};
    virtual ~RGYFilterParamVinverse() {};
    virtual tstring print() const override { return vinverse.print(); };
};

class RGYFilterVinverse : public RGYFilter {
public:
    RGYFilterVinverse(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterVinverse();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    RGY_ERR procPlaneVblur3(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR procPlaneVblur5(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    // Fused vblur3 + vblur5 (Vinverse mode only). Writes pb3 (vblur3 of
    // src) and pb6 (vblur5 of pb3) at the same coordinate in one
    // dispatch, eliminating the intermediate pb3 R+W cycle through DRAM.
    RGY_ERR procPlaneVblur35(RGYFrameInfo *pPb3Plane, RGYFrameInfo *pPb6Plane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR procPlaneMakediff(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pC1Plane, const RGYFrameInfo *pC2Plane, int h_offset,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR procPlaneSbrCombine(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pSrcPlane, const RGYFrameInfo *pDiffPlane, const RGYFrameInfo *pBlurPlane, int h_offset,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR procPlaneFinalize(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const RGYFrameInfo *pPb3Plane, const RGYFrameInfo *pPb6Plane,
        float sstr, float scl, int thr_hbd, int amnt_hbd,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR procPlane(int planeIdx, RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYFrameInfo *pPb3Plane, RGYFrameInfo *pPb6Plane,
        VppVinverseMode mode, float sstr, float scl, int thr_hbd, int amnt_hbd, int h_offset,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    bool m_bInterlacedWarn;
    RGYOpenCLProgramAsync m_vinverse;
    std::unique_ptr<RGYCLFrame> m_pb3;
    std::unique_ptr<RGYCLFrame> m_pb6;
};
