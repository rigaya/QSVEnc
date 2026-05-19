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
#include "rgy_filter_rnnedi_field.h"
#include "rgy_filter_rnnedi_weights.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct RGYRnnediParam {
    static constexpr uint32_t WEIGHTS_FILE_SIZE = 13574928u;

    bool enable;
    std::array<bool, 4> processPlane;
    VppRnnediField field;
    VppNnediNSize nsize;
    int nns;
    VppNnediQuality quality;
    int prescreen;
    VppNnediErrorType errortype;
    int clamp;
    bool doubleHeight;
    tstring weightfile;

    RGYRnnediParam();
    bool operator==(const RGYRnnediParam& x) const;
    bool operator!=(const RGYRnnediParam& x) const;
    tstring print() const;
};

struct RGYRnnediNSizeDesc {
    int xdia;
    int ydia;
};

const RGYRnnediNSizeDesc& rgy_rnnedi_nsize_desc(int nsize);
int rgy_rnnedi_nns_value(int nns);
int rgy_rnnedi_nns_index(int nns);

class RGYFilterParamRnnedi : public RGYFilterParam {
public:
    RGYRnnediParam rnnedi;
    HMODULE hModule;
    rgy_rational<int> timebase;

    RGYFilterParamRnnedi();
    virtual ~RGYFilterParamRnnedi() {};
    virtual tstring print() const override { return rnnedi.print(); };
};

class RGYFilterRnnedi : public RGYFilter {
public:
    RGYFilterRnnedi(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterRnnedi();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;

    RGY_ERR validateParam(const RGYRnnediParam& prm);
    std::shared_ptr<const std::vector<uint8_t>> readWeights(const tstring& weightFile, HMODULE hModule);

protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    RGY_ERR initParams(const std::shared_ptr<RGYFilterParamRnnedi> prm);
    bool getInputTff(const RGYFrameInfo *frame) const;
    void setDoubleRateTimestamp(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames) const;
    RGY_ERR prepareFieldReference(const RGYFrameInfo *pInputFrame, int outputSlot, const RGYRnnediFrameMap& frameMap,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR classifyPixelsAndSeedOutput(const RGYFrameInfo *pInputFrame, int outputSlot, const RGYRnnediFrameMap& frameMap,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR resolveClassifiedPixels(const RGYFrameInfo *pInputFrame, int outputSlot, const RGYRnnediFrameMap& frameMap,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    std::shared_ptr<const std::vector<uint8_t>> m_weights;
    RGYFilterRnnediTransformedWeights m_transformedWeights;
    RGYOpenCLProgramAsync m_rnnedi;
    std::string m_rnnediBuildOptions;
    int m_rnnediPredictorSubgroupSize;
    std::vector<std::unique_ptr<RGYCLFrame>> m_refBuf;
    std::unique_ptr<RGYCLBuf> m_prescreenerWeightBuf;
    std::unique_ptr<RGYCLBuf> m_predictorWeightBuf;
    std::vector<std::unique_ptr<RGYCLBuf>> m_workNNBuf;
    std::vector<std::unique_ptr<RGYCLBuf>> m_numBlocksBuf;
    bool m_defaultTff;
};
