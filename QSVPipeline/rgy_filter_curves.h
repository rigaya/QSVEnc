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

#include "rgy_filter.h"
#include "rgy_prm.h"
#include "rgy_input.h"

class RGYFilterParamCurves : public RGYFilterParam {
public:
    VppCurves curves;
    VideoVUIInfo vuiInfo;

    RGYFilterParamCurves() : curves(), vuiInfo() {

    };
    virtual ~RGYFilterParamCurves() {};
    virtual tstring print() const override;
};

class RGYFilterCurves : public RGYFilter {
    struct RGYFilterCurvesLUT {
        std::unique_ptr<RGYCLBuf> r, g, b, master;

        RGYFilterCurvesLUT() : r(), g(), b(), master() {};
    };
public:
    RGYFilterCurves(std::shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterCurves();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue& queue_main, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    RGY_ERR checkParam(const RGYFilterParam *param);
    VppCurveParams getPreset(const VppCurvesPreset preset);

    std::vector<std::pair<double, double>> parsePoints(const tstring& str);
    template<typename Type>
    std::vector<Type> createLUT(const std::vector<std::pair<double, double>>& vec, const int scale);

    template<typename Type>
    RGY_ERR createLUTFromParam(std::vector<Type>& lut, const tstring& str, const RGY_CSP csp, const std::vector<Type> *master);

    template<typename Type>
    RGY_ERR sendLUTToGPU(std::unique_ptr<RGYCLBuf>& mem, const std::vector<Type>& lut);

    template<typename Type>
    RGY_ERR createLUT(const VppCurveParams& prm, const RGY_CSP csp);

    RGY_ERR createLUT(const RGYFilterParamCurves *prm);

    RGY_ERR procPlane(RGYFrameInfo *plane, cl_mem lut,
        RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event);
    RGY_ERR procFrame(RGYFrameInfo *pFrame, RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event);

    std::unique_ptr<RGYFilterCspCrop> m_convIn;
    std::unique_ptr<RGYFilterCspCrop> m_convOut;
    RGYFilterCurvesLUT m_lut;
    RGYOpenCLProgramAsync m_curves;

    bool m_bInterlacedWarn;
};
