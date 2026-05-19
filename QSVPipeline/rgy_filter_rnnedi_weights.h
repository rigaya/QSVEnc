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

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rgy_prm.h"

static constexpr int RNNEDI_NUM_NSIZE = 7;
static constexpr int RNNEDI_NUM_NNS = 5;
static constexpr VppNnediQuality RNNEDI_DEFAULT_QUALITY = VPP_NNEDI_QUALITY_FAST;
static constexpr int RNNEDI_DEFAULT_PRESCREEN = 2;
static constexpr VppNnediErrorType RNNEDI_DEFAULT_ERRORTYPE = VPP_NNEDI_ETYPE_ABS;
static constexpr int RNNEDI_DEFAULT_BITS_PER_PIXEL = 8;

struct RGYFilterRnnediWeightsLayout {
    VppNnediNSize nsize = VPP_NNEDI_NSIZE_8x6;
    int nns = 16;
    int xdia = 0;
    int ydia = 0;
    int neurons = 0;
    int asize = 0;
    int legacyPrescreenerFloats = 0;
    int prescreenerNetworkFloats = 0;
    int predictorPlaneFloats = 0;
    int predictorCatalogFloats = 0;
    int predictorCatalogOffsetFloats = 0;
    size_t rawWeightFloatCount = 0;
    size_t rawPrescreenerOffsetFloats = 0;
    std::array<size_t, 2> rawPredictorOffsetFloats = {};
};

struct RGYFilterRnnediWeightsParam {
    VppNnediNSize nsize = VPP_NNEDI_NSIZE_16x6;
    int nns = 32;
    VppNnediQuality quality = RNNEDI_DEFAULT_QUALITY;
    int prescreen = RNNEDI_DEFAULT_PRESCREEN;
    VppNnediErrorType errortype = RNNEDI_DEFAULT_ERRORTYPE;
    int bitsPerPixel = RNNEDI_DEFAULT_BITS_PER_PIXEL;
};

struct RGYFilterRnnediTransformedWeights {
    RGYFilterRnnediWeightsParam param;
    RGYFilterRnnediWeightsLayout layout;
    std::vector<float> prescreenerFp32;
    std::vector<float> predictorFp32;
};

struct RGYFilterRnnediFloatWeightsSample {
    size_t index = 0;
    float value = 0.0f;
};

struct RGYFilterRnnediFloatBufferDigest {
    size_t floatCount = 0;
    uint64_t fnv1a64 = 0;
    std::array<RGYFilterRnnediFloatWeightsSample, 8> samples = {};
};

struct RGYFilterRnnediWeightsSummary {
    RGYFilterRnnediWeightsParam param;
    RGYFilterRnnediWeightsLayout layout;
    RGYFilterRnnediFloatBufferDigest prescreenerFp32;
    RGYFilterRnnediFloatBufferDigest predictorFp32;
    uint64_t combinedFnv1a64 = 0;
};

struct RGYFilterRnnediWeightsDefaultSampleResult {
    bool success = false;
    std::string message;
    RGYFilterRnnediWeightsSummary defaultSlower;
    RGYFilterRnnediWeightsSummary chroma;
};

struct RGYFilterRnnediWeightsSelfCheckResult {
    bool success = false;
    std::string message;
    RGYFilterRnnediWeightsLayout defaultSlower;
    RGYFilterRnnediWeightsLayout chroma;
};

bool rgy_filter_rnnedi_weights_layout(RGYFilterRnnediWeightsLayout& layout, VppNnediNSize nsize, int nns, int prescreen, VppNnediErrorType errortype, std::string *errorMessage = nullptr);
bool rgy_filter_rnnedi_transform_weights(RGYFilterRnnediTransformedWeights& dst, const float *rawWeights, size_t rawWeightFloatCount, const RGYFilterRnnediWeightsParam& param, std::string *errorMessage = nullptr);
RGYFilterRnnediWeightsSummary rgy_filter_rnnedi_weights_summary(const RGYFilterRnnediTransformedWeights& weights);
RGYFilterRnnediWeightsDefaultSampleResult rgy_filter_rnnedi_weights_default_samples(const float *rawWeights, size_t rawWeightFloatCount);
RGYFilterRnnediWeightsSelfCheckResult rgy_filter_rnnedi_weights_self_check();
