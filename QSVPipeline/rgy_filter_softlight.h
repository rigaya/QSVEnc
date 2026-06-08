// -----------------------------------------------------------------------------------------
// QSVEnc/VCEEnc/rkmppenc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2026 rigaya
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
// -----------------------------------------------------------------------------------------

#pragma once

#include <array>
#include "rgy_filter_cl.h"
#include "rgy_prm.h"

class RGYFilterParamSoftLight : public RGYFilterParam {
public:
    VppSoftLight softlight;
    VideoVUIInfo vuiInfo;

    RGYFilterParamSoftLight() : softlight(), vuiInfo() {};
    virtual ~RGYFilterParamSoftLight() {};
    virtual tstring print() const override { return softlight.print(); };
};

class RGYFilterSoftLight : public RGYFilter {
public:
    RGYFilterSoftLight(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterSoftLight();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamSoftLight> prm);
    RGY_ERR allocWork(const RGYFrameInfo& rgbFrame);
    RGY_ERR finaliseReduction(RGYOpenCLQueue &queue, std::array<long long, 6>& host);
    RGY_ERR procFrame(RGYFrameInfo *pFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    std::unique_ptr<RGYFilterCspCrop> m_convIn;
    std::unique_ptr<RGYFilterCspCrop> m_convOut;
    std::unique_ptr<RGYCLBuf> m_hsvH;
    std::unique_ptr<RGYCLBuf> m_hsvS;
    std::unique_ptr<RGYCLBuf> m_hsvV;
    std::unique_ptr<RGYCLBuf> m_reduce;
    RGYOpenCLProgramAsync m_softlight;
    int m_numGroupsLastDispatch;
};
