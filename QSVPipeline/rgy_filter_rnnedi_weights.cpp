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

#include "rgy_filter_rnnedi_weights.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace {

static constexpr std::array<int, RNNEDI_NUM_NSIZE> RNNEDI_XDIA = { 8, 16, 32, 48, 8, 16, 32 };
static constexpr std::array<int, RNNEDI_NUM_NSIZE> RNNEDI_YDIA = { 6, 6, 6, 6, 4, 4, 4 };
static constexpr std::array<int, RNNEDI_NUM_NNS> RNNEDI_NNS = { 16, 32, 64, 128, 256 };
static constexpr int RNNEDI_LEGACY_PRESCREENER_FLOATS = 252;
static constexpr int RNNEDI_PRESCREENER_NETWORK_FLOATS = 280;
static constexpr int RNNEDI_PRESCREENER_HIDDEN_COUNT = 4;
static constexpr int RNNEDI_PRESCREENER_SAMPLE_COUNT = 64;
static constexpr int RNNEDI_PRESCREENER_OUTPUT_LANES = 4;
static constexpr int RNNEDI_PREDICTOR_FP32_LANES = 16;
static constexpr uint64_t FNV1A64_OFFSET_BASIS = 14695981039346656037ull;
static constexpr uint64_t FNV1A64_PRIME = 1099511628211ull;
static constexpr size_t WEIGHTBIN_FILE_SIZE = 13574928u;

void set_error(std::string *errorMessage, const std::string& message) {
    if (errorMessage) {
        *errorMessage = message;
    }
}

int rnnedi_nsize_index(const VppNnediNSize nsize) {
    const auto nsizeIndex = static_cast<int>(nsize);
    return (0 <= nsizeIndex && nsizeIndex < RNNEDI_NUM_NSIZE) ? nsizeIndex : -1;
}

int rnnedi_nns_index(const int nns) {
    const auto it = std::find(RNNEDI_NNS.begin(), RNNEDI_NNS.end(), nns);
    return (it == RNNEDI_NNS.end()) ? -1 : static_cast<int>(std::distance(RNNEDI_NNS.begin(), it));
}

bool validate_param(const RGYFilterRnnediWeightsParam& param, std::string *errorMessage) {
    if (rnnedi_nsize_index(param.nsize) < 0) {
        set_error(errorMessage, "RNNEDI weights: nsize must be one of 8x6, 16x6, 32x6, 48x6, 8x4, 16x4, or 32x4.");
        return false;
    }
    if (rnnedi_nns_index(param.nns) < 0) {
        set_error(errorMessage, "RNNEDI weights: nns must be one of 16, 32, 64, 128, or 256.");
        return false;
    }
    if (param.quality != VPP_NNEDI_QUALITY_FAST && param.quality != VPP_NNEDI_QUALITY_SLOW) {
        set_error(errorMessage, "RNNEDI weights: quality must be fast or slow.");
        return false;
    }
    if (param.prescreen < 2 || 4 < param.prescreen) {
        set_error(errorMessage, "RNNEDI weights: supported prescreen values are 2, 3, and 4; prescreen=0/1 use an unsupported prescreener path.");
        return false;
    }
    if (param.errortype < VPP_NNEDI_ETYPE_ABS || VPP_NNEDI_ETYPE_MAX <= param.errortype) {
        set_error(errorMessage, "RNNEDI weights: errortype must be abs or square.");
        return false;
    }
    if (param.bitsPerPixel <= 0 || 16 < param.bitsPerPixel) {
        set_error(errorMessage, "RNNEDI weights: bitsPerPixel must be in [1,16].");
        return false;
    }
    return true;
}

struct RnnediPrescreenerTensorShape {
    int neuronCount = RNNEDI_PRESCREENER_HIDDEN_COUNT;
    int sampleCount = RNNEDI_PRESCREENER_SAMPLE_COUNT;

    int sourcePackedSampleGroupSize = 8;
    int outputSampleStride() const { return neuronCount; }
    int sourcePackedNeuronStride() const { return sourcePackedSampleGroupSize; }
    int sourcePackedGroupStride() const { return neuronCount * sourcePackedSampleGroupSize; }
};

size_t sourcePrescreenerHiddenWeightOffset(const RnnediPrescreenerTensorShape& shape, int neuron, int sample) {
    const int group = sample / shape.sourcePackedSampleGroupSize;
    const int lane = sample % shape.sourcePackedSampleGroupSize;
    return static_cast<size_t>(group * shape.sourcePackedGroupStride() + neuron * shape.sourcePackedNeuronStride() + lane);
}

size_t openclPrescreenerHiddenWeightOffset(const RnnediPrescreenerTensorShape& shape, int neuron, int sample) {
    return static_cast<size_t>(sample) * shape.outputSampleStride() + neuron;
}

struct RnnediOpenCLPrescreenerLayout {
    RnnediPrescreenerTensorShape shape;
    size_t hiddenWeightOffset = 0;
    size_t hiddenWeightFloats = 0;
    size_t hiddenScaleOffset = 0;
    size_t hiddenBiasOffset = 0;
    size_t outputMixOffset = 0;
    size_t outputBiasOffset = 0;
    size_t totalFloats = 0;

    static RnnediOpenCLPrescreenerLayout create() {
        RnnediOpenCLPrescreenerLayout spec;
        spec.hiddenWeightOffset = 0;
        spec.hiddenWeightFloats = static_cast<size_t>(spec.shape.sampleCount) * spec.shape.neuronCount;
        spec.hiddenScaleOffset = spec.hiddenWeightOffset + spec.hiddenWeightFloats;
        spec.hiddenBiasOffset = spec.hiddenScaleOffset + spec.shape.neuronCount;
        spec.outputMixOffset = spec.hiddenBiasOffset + spec.shape.neuronCount;
        spec.outputBiasOffset = spec.outputMixOffset + static_cast<size_t>(RNNEDI_PRESCREENER_OUTPUT_LANES) * spec.shape.neuronCount;
        spec.totalFloats = spec.outputBiasOffset + RNNEDI_PRESCREENER_OUTPUT_LANES;
        return spec;
    }
};

struct RnnediExternalPrescreenerTensorView {
    const float *data = nullptr;
    RnnediPrescreenerTensorShape shape;

    float hiddenWeight(int sample, int neuron) const {
        return data[sourcePrescreenerHiddenWeightOffset(shape, neuron, sample)];
    }
    float hiddenBias(int neuron) const {
        return data[static_cast<size_t>(shape.sampleCount) * shape.neuronCount + neuron];
    }
    float outputMix(int lane, int neuron) const {
        const auto offset = static_cast<size_t>(shape.sampleCount) * shape.neuronCount + shape.neuronCount;
        return data[offset + static_cast<size_t>(neuron) * RNNEDI_PRESCREENER_OUTPUT_LANES + lane];
    }
    float outputBias(int lane) const {
        const auto offset = static_cast<size_t>(shape.sampleCount) * shape.neuronCount
            + shape.neuronCount
            + static_cast<size_t>(shape.neuronCount) * RNNEDI_PRESCREENER_OUTPUT_LANES;
        return data[offset + lane];
    }
};

struct RnnediOpenCLPrescreenerLayoutWriter {
    std::vector<float> *data = nullptr;
    RnnediOpenCLPrescreenerLayout layout;

    void resize() const {
        data->assign(layout.totalFloats, 0.0f);
    }
    void setHiddenWeight(int sample, int neuron, float value) const {
        const auto offset = layout.hiddenWeightOffset + openclPrescreenerHiddenWeightOffset(layout.shape, neuron, sample);
        (*data)[offset] = value;
    }
    void setHiddenScale(int neuron, float value) const {
        (*data)[layout.hiddenScaleOffset + neuron] = value;
    }
    void setHiddenBias(int neuron, float value) const {
        (*data)[layout.hiddenBiasOffset + neuron] = value;
    }
    void setOutputMix(int lane, int neuron, float value) const {
        (*data)[layout.outputMixOffset + static_cast<size_t>(lane) * layout.shape.neuronCount + neuron] = value;
    }
    void setOutputBias(int lane, float value) const {
        (*data)[layout.outputBiasOffset + lane] = value;
    }
};

struct RnnediPrescreenerInputNormalization {
    std::array<double, RNNEDI_PRESCREENER_HIDDEN_COUNT> hiddenMean = {};
    double inputHalf = 0.0;
};

struct RnnediPredictorShape {
    int xdia = 0;
    int ydia = 0;
    int sampleCount = 0;
    int softmaxNeuronCount = 0;
    int neuronCount2 = 0;

    static RnnediPredictorShape create(const RGYFilterRnnediWeightsLayout& layout) {
        RnnediPredictorShape shape;
        shape.xdia = layout.xdia;
        shape.ydia = layout.ydia;
        shape.sampleCount = layout.asize;
        shape.softmaxNeuronCount = layout.neurons;
        shape.neuronCount2 = shape.softmaxNeuronCount * 2;
        return shape;
    }
};

enum class RnnediPredictorHead : int {
    Softmax = 0,
    Elliott = 1,
};

size_t predictor_source_weight_index(int neuron, int sample, int sampleCount);

struct RnnediExternalPredictorTensorView {
    const float *data = nullptr;
    RnnediPredictorShape shape;

    int neuronIndex(RnnediPredictorHead head, int pair) const {
        return pair + (head == RnnediPredictorHead::Elliott ? shape.softmaxNeuronCount : 0);
    }
    float weight(RnnediPredictorHead head, int pair, int sample) const {
        return data[predictor_source_weight_index(neuronIndex(head, pair), sample, shape.sampleCount)];
    }
    float bias(RnnediPredictorHead head, int pair) const {
        return data[static_cast<size_t>(shape.neuronCount2) * shape.sampleCount + neuronIndex(head, pair)];
    }
};

struct RnnediWeightSpan {
    size_t offsetFloats = 0;
    size_t floatCount = 0;
};

struct RnnediWeightFileCatalogKey {
    int nsizeIndex = 0;
    int nnsIndex = 0;
};

RnnediPredictorShape rnnedi_predictor_shape_for_catalog_key(const RnnediWeightFileCatalogKey key) {
    RnnediPredictorShape shape;
    shape.xdia = RNNEDI_XDIA[key.nsizeIndex];
    shape.ydia = RNNEDI_YDIA[key.nsizeIndex];
    shape.softmaxNeuronCount = RNNEDI_NNS[key.nnsIndex];
    shape.neuronCount2 = shape.softmaxNeuronCount * 2;
    shape.sampleCount = shape.xdia * shape.ydia;
    return shape;
}

int rnnedi_predictor_plane_float_count_for_catalog_key(const RnnediWeightFileCatalogKey key) {
    const auto shape = rnnedi_predictor_shape_for_catalog_key(key);
    return (shape.sampleCount + 1) * shape.neuronCount2;
}

struct RnnediWeightFileSectionCursor {
    size_t cursorFloats = 0;

    RnnediWeightSpan take(size_t floatCount) {
        const RnnediWeightSpan span{ cursorFloats, floatCount };
        cursorFloats += floatCount;
        return span;
    }
};

struct RnnediPredictorCatalogLayout {
    RnnediPredictorShape selectedShape;
    int selectedPlaneFloats = 0;
    int selectedPairOffsetFloats = 0;
    int totalFloats = 0;

    static RnnediPredictorCatalogLayout create(const RnnediWeightFileCatalogKey selected) {
        RnnediPredictorCatalogLayout catalog;
        for (int nnsIndex = 0; nnsIndex < RNNEDI_NUM_NNS; nnsIndex++) {
            for (int nsizeIndex = 0; nsizeIndex < RNNEDI_NUM_NSIZE; nsizeIndex++) {
                const RnnediWeightFileCatalogKey key{ nsizeIndex, nnsIndex };
                const int planeFloats = rnnedi_predictor_plane_float_count_for_catalog_key(key);
                if (key.nsizeIndex == selected.nsizeIndex && key.nnsIndex == selected.nnsIndex) {
                    catalog.selectedShape = rnnedi_predictor_shape_for_catalog_key(key);
                    catalog.selectedPlaneFloats = planeFloats;
                    catalog.selectedPairOffsetFloats = catalog.totalFloats;
                }
                catalog.totalFloats += planeFloats * 2;
            }
        }
        return catalog;
    }
};

struct RnnediWeightFileMap {
    RnnediPredictorShape predictorShape;
    int legacyPrescreenerFloats = RNNEDI_LEGACY_PRESCREENER_FLOATS;
    int prescreenerNetworkFloats = RNNEDI_PRESCREENER_NETWORK_FLOATS;
    int predictorPlaneFloats = 0;
    int predictorCatalogFloats = 0;
    int predictorCatalogOffsetFloats = 0;
    size_t rawWeightFloatCount = 0;
    RnnediWeightSpan rawPrescreener = {};
    std::array<RnnediWeightSpan, 2> rawPredictor = {};

    static RnnediWeightFileMap create(int nsizeIndex, int nnsIndex, int prescreen, VppNnediErrorType errortype) {
        const auto catalog = RnnediPredictorCatalogLayout::create(RnnediWeightFileCatalogKey{ nsizeIndex, nnsIndex });

        RnnediWeightFileMap map;
        map.predictorShape = catalog.selectedShape;
        map.predictorPlaneFloats = catalog.selectedPlaneFloats;
        map.predictorCatalogFloats = catalog.totalFloats;
        map.predictorCatalogOffsetFloats = catalog.selectedPairOffsetFloats;

        RnnediWeightFileSectionCursor file;
        file.take(map.legacyPrescreenerFloats);

        std::array<RnnediWeightSpan, 3> prescreenerVariants = {};
        for (auto& span : prescreenerVariants) {
            span = file.take(map.prescreenerNetworkFloats);
        }
        if (2 <= prescreen && prescreen <= 4) {
            map.rawPrescreener = prescreenerVariants[prescreen - 2];
        }

        for (int errorPlane = 0; errorPlane < 2; errorPlane++) {
            const auto predictorCatalog = file.take(map.predictorCatalogFloats);
            if (errorPlane == static_cast<int>(errortype)) {
                map.rawPredictor[0] = RnnediWeightSpan{
                    predictorCatalog.offsetFloats + static_cast<size_t>(map.predictorCatalogOffsetFloats),
                    static_cast<size_t>(map.predictorPlaneFloats)
                };
                map.rawPredictor[1] = RnnediWeightSpan{
                    map.rawPredictor[0].offsetFloats + static_cast<size_t>(map.predictorPlaneFloats),
                    static_cast<size_t>(map.predictorPlaneFloats)
                };
            }
        }
        map.rawWeightFloatCount = file.cursorFloats;
        return map;
    }
};

struct RnnediPredictorNormalizationModel {
    std::vector<double> softmaxCommonModeBySample;
    std::vector<double> perNeuronDcOffset;
    double softmaxBiasCommonMode = 0.0;
};

struct RnnediPredictorTransformSpec {
    RnnediPredictorShape shape;

    static RnnediPredictorTransformSpec create(const RGYFilterRnnediWeightsLayout& layout) {
        RnnediPredictorTransformSpec spec;
        spec.shape = RnnediPredictorShape::create(layout);
        return spec;
    }
};

struct RnnediOpenCLPredictorLayout {
    size_t bodyOffset = 0;
    size_t biasOffset = 0;
    size_t bodyFloats = 0;
    size_t biasFloats = 0;
    size_t totalFloats = 0;

    static RnnediOpenCLPredictorLayout create(const RnnediPredictorShape& shape, VppNnediQuality quality) {
        RnnediOpenCLPredictorLayout layout;
        layout.bodyFloats = static_cast<size_t>((int)quality) * shape.softmaxNeuronCount * shape.sampleCount * 2;
        layout.biasOffset = layout.bodyOffset + layout.bodyFloats;
        layout.biasFloats = static_cast<size_t>((int)quality) * shape.softmaxNeuronCount * 2;
        layout.totalFloats = layout.biasOffset + layout.biasFloats;
        return layout;
    }
};

size_t predictor_fp32_body_index(const RnnediPredictorShape& shape, int q, int pair, int sample);
size_t predictor_fp32_bias_index(const RnnediPredictorShape& shape, const RnnediOpenCLPredictorLayout& layout, int q, int pair);

struct RnnediOpenCLPredictorLayoutWriter {
    std::vector<float> *data = nullptr;
    RnnediPredictorShape shape;
    RnnediOpenCLPredictorLayout layout;

    void resize() const {
        data->assign(layout.totalFloats, 0.0f);
    }
    void setBody(int q, int pair, int sample, RnnediPredictorHead head, float value) const {
        const auto offset = predictor_fp32_body_index(shape, q, pair, sample) + static_cast<int>(head);
        (*data)[offset] = value;
    }
    void setBias(int q, int pair, RnnediPredictorHead head, float value) const {
        const auto offset = predictor_fp32_bias_index(shape, layout, q, pair) + static_cast<int>(head);
        (*data)[offset] = value;
    }
};

size_t predictor_source_weight_index(int neuron, int sample, int sampleCount) {
    return static_cast<size_t>(neuron) * sampleCount + sample;
}

size_t predictor_fp32_body_index(const RnnediPredictorShape& shape, int q, int pair, int sample) {
    const int block = pair / RNNEDI_PREDICTOR_FP32_LANES;
    const int lane = pair - block * RNNEDI_PREDICTOR_FP32_LANES;
    const int blockCount = shape.softmaxNeuronCount / RNNEDI_PREDICTOR_FP32_LANES;
    // fp32 predictor body is laid out as [q][neuronBlock16][sample][lane][softmax/elliott].
    // The kernel's tx=0..15 lanes then read adjacent float2 values for each sample, which is
    // the main GPU coalescing optimization for the predictor dot product.
    return (((static_cast<size_t>(q) * blockCount + block) * shape.sampleCount + sample) * RNNEDI_PREDICTOR_FP32_LANES + lane) * 2;
}

size_t predictor_fp32_bias_index(const RnnediPredictorShape& shape, const RnnediOpenCLPredictorLayout& layout, int q, int pair) {
    return layout.biasOffset + (static_cast<size_t>(q) * shape.softmaxNeuronCount + pair) * 2;
}

uint64_t fnv1a64_update_byte(uint64_t hash, uint8_t value) {
    hash ^= value;
    hash *= FNV1A64_PRIME;
    return hash;
}

uint64_t fnv1a64_update_u64(uint64_t hash, uint64_t value) {
    for (int i = 0; i < 8; i++) {
        hash = fnv1a64_update_byte(hash, static_cast<uint8_t>((value >> (i * 8)) & 0xff));
    }
    return hash;
}

uint64_t fnv1a64_update_i32(uint64_t hash, int value) {
    const auto uvalue = static_cast<uint32_t>(value);
    for (int i = 0; i < 4; i++) {
        hash = fnv1a64_update_byte(hash, static_cast<uint8_t>((uvalue >> (i * 8)) & 0xff));
    }
    return hash;
}

uint64_t fnv1a64_float_buffer(const std::vector<float>& buffer) {
    uint64_t hash = FNV1A64_OFFSET_BASIS;
    for (const auto value : buffer) {
        std::array<uint8_t, sizeof(float)> bytes = {};
        std::memcpy(bytes.data(), &value, sizeof(value));
        for (const auto byte : bytes) {
            hash = fnv1a64_update_byte(hash, byte);
        }
    }
    return hash;
}

RGYFilterRnnediFloatBufferDigest digest_float_buffer(const std::vector<float>& buffer) {
    RGYFilterRnnediFloatBufferDigest digest;
    digest.floatCount = buffer.size();
    digest.fnv1a64 = fnv1a64_float_buffer(buffer);
    if (buffer.empty()) {
        return digest;
    }

    constexpr std::array<size_t, 8> sampleNumerators = { 0, 1, 2, 3, 4, 8, 16, 32 };
    const auto denom = sampleNumerators.back();
    for (size_t i = 0; i < digest.samples.size(); i++) {
        const auto index = (buffer.size() - 1) * sampleNumerators[i] / denom;
        digest.samples[i].index = index;
        digest.samples[i].value = buffer[index];
    }
    return digest;
}

uint64_t fnv1a64_summary(const RGYFilterRnnediTransformedWeights& weights, const RGYFilterRnnediFloatBufferDigest& prescreener, const RGYFilterRnnediFloatBufferDigest& predictor) {
    uint64_t hash = FNV1A64_OFFSET_BASIS;
    hash = fnv1a64_update_i32(hash, weights.param.nsize);
    hash = fnv1a64_update_i32(hash, weights.param.nns);
    hash = fnv1a64_update_i32(hash, (int)weights.param.quality);
    hash = fnv1a64_update_i32(hash, weights.param.prescreen);
    hash = fnv1a64_update_i32(hash, (int)weights.param.errortype);
    hash = fnv1a64_update_i32(hash, weights.param.bitsPerPixel);
    hash = fnv1a64_update_i32(hash, weights.layout.xdia);
    hash = fnv1a64_update_i32(hash, weights.layout.ydia);
    hash = fnv1a64_update_i32(hash, weights.layout.neurons);
    hash = fnv1a64_update_i32(hash, weights.layout.asize);
    hash = fnv1a64_update_u64(hash, static_cast<uint64_t>(prescreener.floatCount));
    hash = fnv1a64_update_u64(hash, prescreener.fnv1a64);
    hash = fnv1a64_update_u64(hash, static_cast<uint64_t>(predictor.floatCount));
    hash = fnv1a64_update_u64(hash, predictor.fnv1a64);
    return hash;
}

RnnediPrescreenerInputNormalization build_prescreener_input_normalization(const RnnediExternalPrescreenerTensorView& src, int bitsPerPixel) {
    RnnediPrescreenerInputNormalization normalization;
    const int prescreenerBits = std::min(bitsPerPixel, 16);
    normalization.inputHalf = (static_cast<double>((1u << prescreenerBits) - 1u)) / 2.0;

    for (int neuron = 0; neuron < src.shape.neuronCount; neuron++) {
        double sum = 0.0;
        for (int sample = 0; sample < src.shape.sampleCount; sample++) {
            sum += src.hiddenWeight(sample, neuron);
        }
        normalization.hiddenMean[neuron] = sum / static_cast<double>(src.shape.sampleCount);
    }
    return normalization;
}

void write_prescreener_hidden_layer(const RnnediOpenCLPrescreenerLayoutWriter& dst, const RnnediExternalPrescreenerTensorView& src, const RnnediPrescreenerInputNormalization& normalization) {
    for (int sample = 0; sample < src.shape.sampleCount; sample++) {
        for (int neuron = 0; neuron < src.shape.neuronCount; neuron++) {
            const auto value = static_cast<float>((src.hiddenWeight(sample, neuron) - normalization.hiddenMean[neuron]) / normalization.inputHalf);
            dst.setHiddenWeight(sample, neuron, value);
        }
    }
    for (int neuron = 0; neuron < src.shape.neuronCount; neuron++) {
        dst.setHiddenScale(neuron, 1.0f);
        dst.setHiddenBias(neuron, src.hiddenBias(neuron));
    }
}

void write_prescreener_output_layer(const RnnediOpenCLPrescreenerLayoutWriter& dst, const RnnediExternalPrescreenerTensorView& src) {
    for (int lane = 0; lane < RNNEDI_PRESCREENER_OUTPUT_LANES; lane++) {
        for (int neuron = 0; neuron < src.shape.neuronCount; neuron++) {
            dst.setOutputMix(lane, neuron, src.outputMix(lane, neuron));
        }
        dst.setOutputBias(lane, src.outputBias(lane));
    }
}

void transform_prescreener_fp32(std::vector<float>& dst, const float *rawWeights, const RGYFilterRnnediWeightsParam& param, const RGYFilterRnnediWeightsLayout& layout) {
    const auto openclLayout = RnnediOpenCLPrescreenerLayout::create();
    const RnnediExternalPrescreenerTensorView src{ rawWeights + layout.rawPrescreenerOffsetFloats, openclLayout.shape };
    const RnnediOpenCLPrescreenerLayoutWriter out{ &dst, openclLayout };
    const auto normalization = build_prescreener_input_normalization(src, param.bitsPerPixel);

    out.resize();
    write_prescreener_hidden_layer(out, src, normalization);
    write_prescreener_output_layer(out, src);
}

RnnediPredictorNormalizationModel build_predictor_normalization_model(const RnnediExternalPredictorTensorView& src) {
    RnnediPredictorNormalizationModel model;
    model.softmaxCommonModeBySample.assign(src.shape.sampleCount, 0.0);
    model.perNeuronDcOffset.assign(src.shape.neuronCount2, 0.0);

    for (const auto head : { RnnediPredictorHead::Softmax, RnnediPredictorHead::Elliott }) {
        for (int pair = 0; pair < src.shape.softmaxNeuronCount; pair++) {
            const int neuron = src.neuronIndex(head, pair);
            for (int sample = 0; sample < src.shape.sampleCount; sample++) {
                model.perNeuronDcOffset[neuron] += src.weight(head, pair, sample);
            }
            model.perNeuronDcOffset[neuron] /= static_cast<double>(src.shape.sampleCount);
        }
    }

    for (int pair = 0; pair < src.shape.softmaxNeuronCount; pair++) {
        const int neuron = src.neuronIndex(RnnediPredictorHead::Softmax, pair);
        for (int sample = 0; sample < src.shape.sampleCount; sample++) {
            const double centered = src.weight(RnnediPredictorHead::Softmax, pair, sample) - model.perNeuronDcOffset[neuron];
            model.softmaxCommonModeBySample[sample] += centered;
        }
        model.softmaxBiasCommonMode += src.bias(RnnediPredictorHead::Softmax, pair);
    }

    for (auto& offset : model.softmaxCommonModeBySample) {
        offset /= static_cast<double>(src.shape.softmaxNeuronCount);
    }
    model.softmaxBiasCommonMode /= static_cast<double>(src.shape.softmaxNeuronCount);
    return model;
}

float predictor_normalized_weight(const RnnediExternalPredictorTensorView& src, const RnnediPredictorNormalizationModel& model, RnnediPredictorHead head, int pair, int sample) {
    const int neuron = src.neuronIndex(head, pair);
    float centered = src.weight(head, pair, sample) - static_cast<float>(model.perNeuronDcOffset[neuron]);
    if (head == RnnediPredictorHead::Softmax) {
        centered -= static_cast<float>(model.softmaxCommonModeBySample[sample]);
    }
    return centered;
}

float predictor_normalized_bias(const RnnediExternalPredictorTensorView& src, const RnnediPredictorNormalizationModel& model, RnnediPredictorHead head, int pair) {
    const auto bias = src.bias(head, pair);
    if (head == RnnediPredictorHead::Softmax) {
        return static_cast<float>(bias - model.softmaxBiasCommonMode);
    }
    return bias;
}

void write_predictor_quality_plane(const RnnediOpenCLPredictorLayoutWriter& dst, int q, const RnnediExternalPredictorTensorView& src, const RnnediPredictorNormalizationModel& model) {
    for (int pair = 0; pair < src.shape.softmaxNeuronCount; pair++) {
        for (int sample = 0; sample < src.shape.sampleCount; sample++) {
            dst.setBody(q, pair, sample, RnnediPredictorHead::Softmax,
                predictor_normalized_weight(src, model, RnnediPredictorHead::Softmax, pair, sample));
            dst.setBody(q, pair, sample, RnnediPredictorHead::Elliott,
                predictor_normalized_weight(src, model, RnnediPredictorHead::Elliott, pair, sample));
        }
        dst.setBias(q, pair, RnnediPredictorHead::Softmax,
            predictor_normalized_bias(src, model, RnnediPredictorHead::Softmax, pair));
        dst.setBias(q, pair, RnnediPredictorHead::Elliott,
            predictor_normalized_bias(src, model, RnnediPredictorHead::Elliott, pair));
    }
}

bool transform_predictor_fp32(std::vector<float>& dst, const float *rawWeights, const RGYFilterRnnediWeightsParam& param, const RGYFilterRnnediWeightsLayout& layout, std::string *errorMessage) {
    const auto spec = RnnediPredictorTransformSpec::create(layout);
    const auto fp32Layout = RnnediOpenCLPredictorLayout::create(spec.shape, param.quality);
    const RnnediOpenCLPredictorLayoutWriter out{ &dst, spec.shape, fp32Layout };

    out.resize();
    for (int q = 0; q < (int)param.quality; q++) {
        const RnnediExternalPredictorTensorView src{ rawWeights + layout.rawPredictorOffsetFloats[q], spec.shape };
        write_predictor_quality_plane(out, q, src, build_predictor_normalization_model(src));
    }
    if (dst.size() != fp32Layout.totalFloats) {
        set_error(errorMessage, "RNNEDI weights: invalid fp32 predictor layout.");
        return false;
    }
    return true;
}

void publish_weight_file_map(RGYFilterRnnediWeightsLayout& layout, const RnnediWeightFileMap& map, VppNnediNSize nsize, int nns) {
    layout = RGYFilterRnnediWeightsLayout();
    layout.nsize = nsize;
    layout.nns = nns;
    layout.xdia = map.predictorShape.xdia;
    layout.ydia = map.predictorShape.ydia;
    layout.neurons = map.predictorShape.softmaxNeuronCount;
    layout.asize = map.predictorShape.sampleCount;
    layout.legacyPrescreenerFloats = map.legacyPrescreenerFloats;
    layout.prescreenerNetworkFloats = map.prescreenerNetworkFloats;
    layout.predictorPlaneFloats = map.predictorPlaneFloats;
    layout.predictorCatalogFloats = map.predictorCatalogFloats;
    layout.predictorCatalogOffsetFloats = map.predictorCatalogOffsetFloats;
    layout.rawWeightFloatCount = map.rawWeightFloatCount;
    layout.rawPrescreenerOffsetFloats = map.rawPrescreener.offsetFloats;
    layout.rawPredictorOffsetFloats[0] = map.rawPredictor[0].offsetFloats;
    layout.rawPredictorOffsetFloats[1] = map.rawPredictor[1].offsetFloats;
}

} // namespace

bool rgy_filter_rnnedi_weights_layout(RGYFilterRnnediWeightsLayout& layout, VppNnediNSize nsize, int nns, int prescreen, VppNnediErrorType errortype, std::string *errorMessage) {
    const auto nsizeIndex = rnnedi_nsize_index(nsize);
    if (nsizeIndex < 0) {
        set_error(errorMessage, "RNNEDI weights layout: nsize must be one of 8x6, 16x6, 32x6, 48x6, 8x4, 16x4, or 32x4.");
        return false;
    }
    const auto nnsIndex = rnnedi_nns_index(nns);
    if (nnsIndex < 0) {
        set_error(errorMessage, "RNNEDI weights layout: nns must be one of 16, 32, 64, 128, or 256.");
        return false;
    }
    if (prescreen < 0 || 4 < prescreen) {
        set_error(errorMessage, "RNNEDI weights layout: prescreen must be in [0,4].");
        return false;
    }
    if (errortype < VPP_NNEDI_ETYPE_ABS || VPP_NNEDI_ETYPE_MAX <= errortype) {
        set_error(errorMessage, "RNNEDI weights layout: errortype must be abs or square.");
        return false;
    }

    const auto weightFileMap = RnnediWeightFileMap::create(nsizeIndex, nnsIndex, prescreen, errortype);
    publish_weight_file_map(layout, weightFileMap, nsize, nns);
    return true;
}

bool rgy_filter_rnnedi_transform_weights(RGYFilterRnnediTransformedWeights& dst, const float *rawWeights, size_t rawWeightFloatCount, const RGYFilterRnnediWeightsParam& param, std::string *errorMessage) {
    if (!validate_param(param, errorMessage)) {
        return false;
    }
    if (!rawWeights) {
        set_error(errorMessage, "RNNEDI weights: rawWeights is null.");
        return false;
    }

    RGYFilterRnnediWeightsLayout layout;
    if (!rgy_filter_rnnedi_weights_layout(layout, param.nsize, param.nns, param.prescreen, param.errortype, errorMessage)) {
        return false;
    }
    if (rawWeightFloatCount < layout.rawWeightFloatCount) {
        set_error(errorMessage, "RNNEDI weights: raw weights buffer is smaller than nnedi3_weights.bin layout.");
        return false;
    }

    RGYFilterRnnediTransformedWeights tmp;
    tmp.param = param;
    tmp.layout = layout;
    transform_prescreener_fp32(tmp.prescreenerFp32, rawWeights, param, layout);
    if (!transform_predictor_fp32(tmp.predictorFp32, rawWeights, param, layout, errorMessage)) {
        return false;
    }

    dst = std::move(tmp);
    return true;
}

RGYFilterRnnediWeightsSummary rgy_filter_rnnedi_weights_summary(const RGYFilterRnnediTransformedWeights& weights) {
    RGYFilterRnnediWeightsSummary summary;
    summary.param = weights.param;
    summary.layout = weights.layout;
    summary.prescreenerFp32 = digest_float_buffer(weights.prescreenerFp32);
    summary.predictorFp32 = digest_float_buffer(weights.predictorFp32);
    summary.combinedFnv1a64 = fnv1a64_summary(weights, summary.prescreenerFp32, summary.predictorFp32);
    return summary;
}

RGYFilterRnnediWeightsDefaultSampleResult rgy_filter_rnnedi_weights_default_samples(const float *rawWeights, size_t rawWeightFloatCount) {
    RGYFilterRnnediWeightsDefaultSampleResult result;
    std::string error;

    RGYFilterRnnediWeightsParam defaultParam;
    defaultParam.nsize = VPP_NNEDI_NSIZE_16x6;
    defaultParam.nns = 32;
    defaultParam.quality = VPP_NNEDI_QUALITY_FAST;
    RGYFilterRnnediTransformedWeights defaultWeights;
    if (!rgy_filter_rnnedi_transform_weights(defaultWeights, rawWeights, rawWeightFloatCount, defaultParam, &error)) {
        result.message = error;
        return result;
    }
    result.defaultSlower = rgy_filter_rnnedi_weights_summary(defaultWeights);

    RGYFilterRnnediWeightsParam chromaParam;
    chromaParam.nsize = VPP_NNEDI_NSIZE_8x4;
    chromaParam.nns = 16;
    chromaParam.quality = VPP_NNEDI_QUALITY_FAST;
    RGYFilterRnnediTransformedWeights chromaWeights;
    if (!rgy_filter_rnnedi_transform_weights(chromaWeights, rawWeights, rawWeightFloatCount, chromaParam, &error)) {
        result.message = error;
        return result;
    }
    result.chroma = rgy_filter_rnnedi_weights_summary(chromaWeights);

    result.success = true;
    result.message = "ok";
    return result;
}

RGYFilterRnnediWeightsSelfCheckResult rgy_filter_rnnedi_weights_self_check() {
    RGYFilterRnnediWeightsSelfCheckResult result;
    std::string error;
    if (!rgy_filter_rnnedi_weights_layout(result.defaultSlower, VPP_NNEDI_NSIZE_16x6, 32, RNNEDI_DEFAULT_PRESCREEN, RNNEDI_DEFAULT_ERRORTYPE, &error)) {
        result.message = error;
        return result;
    }
    if (!rgy_filter_rnnedi_weights_layout(result.chroma, VPP_NNEDI_NSIZE_8x4, 16, RNNEDI_DEFAULT_PRESCREEN, RNNEDI_DEFAULT_ERRORTYPE, &error)) {
        result.message = error;
        return result;
    }

    if (result.defaultSlower.xdia != 16 || result.defaultSlower.ydia != 6 || result.defaultSlower.neurons != 32 || result.defaultSlower.asize != 96 || result.defaultSlower.predictorPlaneFloats != 6208) {
        result.message = "RNNEDI weights self-check: default Slower dimensions mismatch.";
        return result;
    }
    if (result.chroma.xdia != 8 || result.chroma.ydia != 4 || result.chroma.neurons != 16 || result.chroma.asize != 32 || result.chroma.predictorPlaneFloats != 1056) {
        result.message = "RNNEDI weights self-check: chroma dimensions mismatch.";
        return result;
    }
    if (result.defaultSlower.rawPrescreenerOffsetFloats != RNNEDI_LEGACY_PRESCREENER_FLOATS || result.chroma.rawPrescreenerOffsetFloats != RNNEDI_LEGACY_PRESCREENER_FLOATS) {
        result.message = "RNNEDI weights self-check: prescreener offset mismatch.";
        return result;
    }
    if (result.defaultSlower.rawPredictorOffsetFloats[1] != result.defaultSlower.rawPredictorOffsetFloats[0] + static_cast<size_t>(result.defaultSlower.predictorPlaneFloats)) {
        result.message = "RNNEDI weights self-check: default predictor plane offset mismatch.";
        return result;
    }
    if (result.chroma.rawPredictorOffsetFloats[1] != result.chroma.rawPredictorOffsetFloats[0] + static_cast<size_t>(result.chroma.predictorPlaneFloats)) {
        result.message = "RNNEDI weights self-check: chroma predictor plane offset mismatch.";
        return result;
    }
    if (result.defaultSlower.rawWeightFloatCount != result.chroma.rawWeightFloatCount || result.defaultSlower.rawWeightFloatCount * sizeof(float) != WEIGHTBIN_FILE_SIZE) {
        result.message = "RNNEDI weights self-check: raw nnedi3_weights.bin size mismatch.";
        return result;
    }

    result.success = true;
    result.message = "ok";
    return result;
}
