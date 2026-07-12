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
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_FILTER_STDEINT_H__
#define __RGY_FILTER_STDEINT_H__

#include "rgy_filter_cl.h"
#include "rgy_prm.h"
#include "rgy_openvino.h"
#include <array>
#include <memory>
#include <vector>

class RGYFilterParamStDeint : public RGYFilterParam {
public:
    tstring modelFile;
    tstring modelDir;
    tstring device;
    tstring precision;
    VppStDeintMode mode;
    CspMatrix colormatrix;
    CspColorRange colorrange;
    rgy_rational<int> timebase;

    RGYFilterParamStDeint() :
        modelFile(), modelDir(), device(_T("GPU.0")), precision(_T("fp32")), mode(VppStDeintMode::Bob),
        colormatrix(RGY_MATRIX_AUTO), colorrange(RGY_COLORRANGE_AUTO), timebase() {};
    virtual ~RGYFilterParamStDeint() {};
    virtual tstring print() const override;
};

class RGYFilterStDeint : public RGYFilter {
public:
    RGYFilterStDeint(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterStDeint();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    void setOutputFrameProp(RGYFrameInfo *output, const RGYFrameInfo *input) const;
    void setBobTimestamp(const RGYFrameInfo *input, RGYFrameInfo **outputs) const;
    RGYFrameInfo rgbFrame(const std::array<std::unique_ptr<RGYCLBuf>, 3>& planes) const;
    RGY_ERR createRgbPlanes(RGYCLBuf *parent, std::array<std::unique_ptr<RGYCLBuf>, 3>& planes);
    RGY_ERR convertToRgb(const RGYFrameInfo *input, RGYOpenCLQueue& queue,
        const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event);
    RGY_ERR convertFromRgb(RGYFrameInfo *output, RGYOpenCLQueue& queue,
        const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event);
    RGY_ERR runOcl(const RGYFrameInfo *input, RGYFrameInfo **outputs, int outputCount,
        RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event);

    std::unique_ptr<RGYOpenVINO> m_ov;
    std::unique_ptr<RGYFilterCspCrop> m_cropToRgb;
    std::unique_ptr<RGYFilterCspCrop> m_cropFromRgb;
    int m_width;
    int m_height;
    VppStDeintMode m_mode;
    bool m_defaultTff;
    bool m_useOcl;

    std::vector<float> m_inputBuf;
    std::vector<float> m_outputBuf;
    std::unique_ptr<RGYOpenCLProgram> m_program;
    std::unique_ptr<RGYCLBuf> m_inputBufCL;
    std::unique_ptr<RGYCLBuf> m_outputBufCL;
    std::unique_ptr<RGYCLBuf> m_weaveBufCL;
    std::array<std::unique_ptr<RGYCLBuf>, 3> m_inputPlanes;
    std::array<std::unique_ptr<RGYCLBuf>, 3> m_weavePlanes;
};

#endif //__RGY_FILTER_STDEINT_H__
