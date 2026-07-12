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
#include <memory>
#include <vector>

class RGYFilterParamStDeint : public RGYFilterParam {
public:
    tstring modelFile;
    tstring modelDir;
    tstring device;
    tstring precision;
    VppStDeintMode mode;
    tstring colormatrix;
    tstring colorrange;
    rgy_rational<int> timebase;

    RGYFilterParamStDeint() :
        modelFile(), modelDir(), device(_T("GPU.0")), precision(_T("fp32")), mode(VppStDeintMode::Bob),
        colormatrix(_T("auto")), colorrange(_T("auto")), timebase() {};
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

    void yuvToRGB(const RGYFrameInfo& input, float *dst);
    void rgbToYUV(const RGYFrameInfo& output, const float *src);
    void setupColorCoeffs(int matrixSel, bool rangeTV, int pixMax);
    void setOutputFrameProp(RGYFrameInfo *output, const RGYFrameInfo *input) const;
    void setBobTimestamp(const RGYFrameInfo *input, RGYFrameInfo **outputs) const;
    void weaveRestoration(float *dst, const float *restoration, bool frameA) const;
    RGY_ERR writeOutputFrame(RGYFrameInfo *output, const float *rgb, RGYOpenCLQueue& queue, RGYOpenCLEvent *event);
    RGY_ERR runOcl(const RGYFrameInfo *input, RGYFrameInfo **outputs, int outputCount,
        RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event);

    std::unique_ptr<RGYOpenVINO> m_ov;
    int m_width;
    int m_height;
    VppStDeintMode m_mode;
    bool m_defaultTff;
    bool m_useOcl;

    float m_yOff, m_yScale, m_yRange, m_cOff, m_cScale, m_cRange;
    float m_matVR, m_matUG, m_matVG, m_matUB;
    float m_matRY, m_matGY, m_matBY, m_matRU, m_matGU, m_matBU, m_matRV, m_matGV, m_matBV;

    std::vector<float> m_inputBuf;
    std::vector<float> m_outputBuf;
    std::vector<float> m_weaveBuf;
    std::unique_ptr<RGYCLFrame> m_inputStaging;
    std::unique_ptr<RGYCLFrame> m_outputStaging;
    std::unique_ptr<RGYOpenCLProgram> m_program;
    std::unique_ptr<RGYCLBuf> m_inputBufCL;
    std::unique_ptr<RGYCLBuf> m_outputBufCL;
};

#endif //__RGY_FILTER_STDEINT_H__
