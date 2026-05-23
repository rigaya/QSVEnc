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
#include "rgy_filter_nnedi_field.h"
#include "rgy_filter_nnedi_weights.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct RGYNnediParam {
    static constexpr uint32_t WEIGHTS_FILE_SIZE = 13574928u;

    bool enable;
    std::array<bool, 4> processPlane;
    VppNnediField field;
    VppNnediNSize nsize;
    int nns;
    VppNnediQuality quality;
    int prescreen;
    VppNnediErrorType errortype;
    int clamp;
    bool doubleHeight;
    tstring weightfile;

    RGYNnediParam();
    bool operator==(const RGYNnediParam& x) const;
    bool operator!=(const RGYNnediParam& x) const;
    tstring print() const;
};

struct RGYNnediNSizeDesc {
    int xdia;
    int ydia;
};

const RGYNnediNSizeDesc& rgy_nnedi_nsize_desc(int nsize);
int rgy_nnedi_nns_value(int nns);
int rgy_nnedi_nns_index(int nns);

class RGYFilterParamNnedi : public RGYFilterParam {
public:
    RGYNnediParam nnedi;
    HMODULE hModule;
    rgy_rational<int> timebase;

    RGYFilterParamNnedi();
    virtual ~RGYFilterParamNnedi() {};
    virtual tstring print() const override { return nnedi.print(); };
};

class RGYFilterNnedi : public RGYFilter {
public:
    RGYFilterNnedi(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterNnedi();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;

    RGY_ERR validateParam(const RGYNnediParam& prm);
    std::shared_ptr<const std::vector<uint8_t>> readWeights(const tstring& weightFile, HMODULE hModule);

protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    RGY_ERR initParams(const std::shared_ptr<RGYFilterParamNnedi> prm);
    bool getInputTff(const RGYFrameInfo *frame) const;
    void setDoubleRateTimestamp(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames) const;
    RGY_ERR prepareFieldReference(const RGYFrameInfo *pInputFrame, int outputSlot, const RGYNnediFrameMap& frameMap,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR classifyPixelsAndSeedOutput(const RGYFrameInfo *pInputFrame, int outputSlot, const RGYNnediFrameMap& frameMap,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR resolveClassifiedPixels(const RGYFrameInfo *pInputFrame, int outputSlot, const RGYNnediFrameMap& frameMap,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    std::shared_ptr<const std::vector<uint8_t>> m_weights;
    RGYFilterNnediTransformedWeights m_transformedWeights;
    RGYOpenCLProgramAsync m_nnedi;
    std::string m_nnediBuildOptions;
    int m_nnediPredictorSubgroupSize;
    std::vector<std::unique_ptr<RGYCLFrame>> m_refBuf;
    std::unique_ptr<RGYCLBuf> m_prescreenerWeightBuf;
    std::unique_ptr<RGYCLBuf> m_predictorWeightBuf;
    std::vector<std::unique_ptr<RGYCLBuf>> m_workNNBuf;
    std::vector<std::unique_ptr<RGYCLBuf>> m_numBlocksBuf;
    int m_tileGroupsX;
    int m_tileRows;
    int m_predLocalX;
    int m_predLocalY;
    bool m_defaultTff;
};
