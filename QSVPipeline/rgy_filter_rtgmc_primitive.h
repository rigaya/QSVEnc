// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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
#include "rgy_filter_resize.h"

enum class RGYRtgmcPrimitiveOp {
    Copy = 0,
    MakeDiff,
    MakeDiffRemoveGrain20,
    MakeDiffRemoveGrain20AddDiff,
    AddDiff,
    AddWeightedDiff,
    RemoveGrain,
    Repair,
    Merge,
    GaussResize,
    VerticalMin5,
    VerticalMax5,
    LogicMin,
    LogicMax,
};

enum class RGYRtgmcPrimitiveRefMode {
    Disabled = 0,
    RemoveGrain20,
};

class RGYFilterParamRtgmcPrimitive : public RGYFilterParam {
public:
    RGYRtgmcPrimitiveOp op;
    RGYRtgmcPrimitiveRefMode refMode;
    int mode;
    float weight;
    int planes;
    bool processChroma;

    RGYFilterParamRtgmcPrimitive();
    virtual ~RGYFilterParamRtgmcPrimitive() {}
    virtual tstring print() const override;
};

class RGYFilterRtgmcPrimitive : public RGYFilter {
public:
    RGYFilterRtgmcPrimitive(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterRtgmcPrimitive();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
    RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    static bool needsRef(RGYRtgmcPrimitiveOp op);
    static const TCHAR *opToStr(RGYRtgmcPrimitiveOp op);
    static const TCHAR *refModeToStr(RGYRtgmcPrimitiveRefMode refMode);

protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamRtgmcPrimitive> &prm);
    RGY_ERR buildKernels(const std::shared_ptr<RGYFilterParamRtgmcPrimitive> &prm);
    RGY_ERR processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame,
        const RGYFilterParamRtgmcPrimitive &prm,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    bool processPlane(int iplane, const RGYFilterParamRtgmcPrimitive &prm) const;
    RGYFrameInfo *generatedRefFrame();

    RGY_ERR setupGaussResize(const RGYFilterParamRtgmcPrimitive &prm);
    RGY_ERR processGaussResize(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
        const RGYFilterParamRtgmcPrimitive &prm,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    RGYOpenCLProgramAsync m_primitive;
    std::string m_buildOptions;
    std::unique_ptr<RGYFilterResize> m_resizeGauss;
    bool m_useKernel;
};
