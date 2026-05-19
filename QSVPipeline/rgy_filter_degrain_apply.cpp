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

#include "rgy_filter_degrain.h"
#include "rgy_filter_degrain_common.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <vector>

namespace {
uint32_t degrainDisableMask(const RGYDegrainRefDisableArray &disableRefs, const int temporalDirections) {
    uint32_t mask = 0;
    for (int refDirection = 0; refDirection < std::min(temporalDirections, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS); refDirection++) {
        if (disableRefs[refDirection]) {
            mask |= (1u << refDirection);
        }
    }
    return mask;
}

bool degrainTraceEnvEnabled(const char *name) {
    const auto value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

int degrainTraceEnvInt(const char *name, const int fallback) {
    const auto value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::atoi(value);
}

size_t degrainTemporalMixPlanBytes(const RGYDegrainBlockLayout &layout) {
    return layout.blockCount() * ((size_t)std::max(layout.temporalDirections, 0) + 1u) * sizeof(float);
}

size_t degrainOverlapBlendTableBytes(const int overlapX, const int overlapY) {
    return ((size_t)std::max(overlapX, 0) + (size_t)std::max(overlapY, 0)) * sizeof(float);
}

void degrainFillOverlapBlendAxis(float *dst, const int overlap) {
    if (dst == nullptr || overlap <= 0) {
        return;
    }
    constexpr float pi = 3.14159265358979323846f;
    for (int i = 0; i < overlap; i++) {
        const float t = ((float)i + 0.5f) / (float)overlap;
        dst[i] = 0.5f + 0.5f * std::cos(pi * t);
    }
}

void appendDegrainPixelTraceRefs(std::ostringstream &oss, const int *record) {
    oss << "\"refs\":[";
    for (int refDirection = 0; refDirection < 4; refDirection++) {
        if (refDirection > 0) {
            oss << ',';
        }
        const int traceOffset = 15 + refDirection * 6;
        oss << "{\"ref_slot\":" << refDirection
            << ",\"confidence\":" << record[traceOffset + 0]
            << ",\"sample\":" << record[traceOffset + 1]
            << ",\"dx\":" << record[traceOffset + 2]
            << ",\"dy\":" << record[traceOffset + 3]
            << ",\"sad\":" << record[traceOffset + 4]
            << ",\"valid\":" << record[traceOffset + 5]
            << '}';
    }
    oss << ']';
}

void logDegrainPixelTraceRecords(RGYLog *log, const int *trace, const RGYFrameInfo &planeCur,
    const RGY_PLANE plane, const int currentFrame, const VppDegrainStage stage, const int reqDelta) {
    if (!log || !trace || trace[0] != 0x4d435054) {
        return;
    }
    std::ostringstream head;
    head << "{\"type\":\"degrain_pixel_trace\",\"kind\":\"summary\""
        << ",\"frame\":" << planeCur.inputFrameId
        << ",\"current_frame\":" << currentFrame
        << ",\"pts\":" << (long long)planeCur.timestamp
        << ",\"dur\":" << (long long)planeCur.duration
        << ",\"stage\":" << (int)stage
        << ",\"request_delta\":" << reqDelta
        << ",\"plane\":" << (int)plane
        << ",\"x\":" << trace[1]
        << ",\"y\":" << trace[2]
        << ",\"width\":" << trace[3]
        << ",\"height\":" << trace[4]
        << ",\"fallback\":" << trace[5]
        << ",\"covered\":" << trace[6]
        << ",\"scale_x\":" << trace[7]
        << ",\"scale_y\":" << trace[8]
        << ",\"block_size_x\":" << trace[9]
        << ",\"block_size_y\":" << trace[10]
        << ",\"overlap_x\":" << trace[11]
        << ",\"overlap_y\":" << trace[12]
        << ",\"step_x\":" << trace[13]
        << ",\"primary_bx\":" << trace[14]
        << ",\"primary_by\":" << trace[15]
        << ",\"block_count_x\":" << trace[16]
        << ",\"block_count_y\":" << trace[17]
        << ",\"sample_sum\":" << trace[18]
        << ",\"sample_count\":" << trace[19]
        << ",\"result\":" << trace[20]
        << ",\"sad_limit\":" << trace[21]
        << ",\"disable_mask\":" << trace[22]
        << ",\"blocks_x\":" << trace[23]
        << ",\"blocks_y\":" << trace[24]
        << ",\"records\":" << trace[25]
        << '}';
    log->write(RGY_LOG_INFO, RGY_LOGT_VPP, _T("%s\n"), char_to_tstring(head.str()).c_str());

    const int recordCount = std::min(trace[25], 4);
    for (int i = 0; i < recordCount; i++) {
        const int *record = trace + 32 + i * 48;
        std::ostringstream oss;
        oss << "{\"type\":\"degrain_pixel_trace\",\"kind\":\"block\""
            << ",\"frame\":" << planeCur.inputFrameId
            << ",\"current_frame\":" << currentFrame
            << ",\"pts\":" << (long long)planeCur.timestamp
            << ",\"dur\":" << (long long)planeCur.duration
            << ",\"stage\":" << (int)stage
            << ",\"request_delta\":" << reqDelta
            << ",\"plane\":" << (int)plane
            << ",\"x\":" << trace[1]
            << ",\"y\":" << trace[2]
            << ",\"block_x\":" << record[0]
            << ",\"block_y\":" << record[1]
            << ",\"block\":" << record[2]
            << ",\"base_x\":" << record[3]
            << ",\"base_y\":" << record[4]
            << ",\"local_x\":" << record[5]
            << ",\"local_y\":" << record[6]
            << ",\"window\":" << record[7]
            << ",\"src_sample\":" << record[8]
            << ",\"sample\":" << record[9]
            << ",\"contribution\":" << record[10]
            << ",\"source_mix\":" << record[11]
            << ",\"reference_mix_sum\":" << record[12]
            << ",\"confidence_sum_raw\":" << record[13]
            << ",\"source_confidence_raw\":" << record[14]
            << ',';
        appendDegrainPixelTraceRefs(oss, record);
        oss << '}';
        log->write(RGY_LOG_INFO, RGY_LOGT_VPP, _T("%s\n"), char_to_tstring(oss.str()).c_str());
    }
}

float degrainTraceReferenceAffinityFromSad(const int thresholdSad, const int measuredSad) {
    if (thresholdSad <= measuredSad) {
        return 0.0f;
    }
    const float sadRatio = (float)measuredSad / (float)thresholdSad;
    const float sadRatio2 = sadRatio * sadRatio;
    return (1.0f - sadRatio2) / (1.0f + sadRatio2);
}

int degrainPlaneScaleX(const RGYFrameInfo *frame, const RGY_PLANE plane) {
    if (!frame || plane == RGY_PLANE_Y) {
        return 1;
    }
    switch (RGY_CSP_CHROMA_FORMAT[frame->csp]) {
    case RGY_CHROMAFMT_YUV420:
    case RGY_CHROMAFMT_YUV422:
        return 2;
    default:
        return 1;
    }
}

int degrainPlaneScaleY(const RGYFrameInfo *frame, const RGY_PLANE plane) {
    if (!frame || plane == RGY_PLANE_Y) {
        return 1;
    }
    switch (RGY_CSP_CHROMA_FORMAT[frame->csp]) {
    case RGY_CHROMAFMT_YUV420:
        return 2;
    default:
        return 1;
    }
}

int degrainScaleCovered(const int covered, const int scale) {
    return (covered + std::max(scale, 1) - 1) / std::max(scale, 1);
}

bool degrainCanProcessChroma(const RGYFrameInfo *frame) {
    if (!frame || RGY_CSP_PLANES[frame->csp] < 3) {
        return false;
    }
    switch (RGY_CSP_CHROMA_FORMAT[frame->csp]) {
    case RGY_CHROMAFMT_YUV420:
    case RGY_CHROMAFMT_YUV422:
    case RGY_CHROMAFMT_YUV444:
        return true;
    default:
        return false;
    }
}

std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> degrainRenderRefPlanes(const RGYFilterDegrainFrameSet &frames, const RGY_PLANE plane) {
    auto planes = std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS>();
    const auto planeCur = getPlane(frames.cur, plane);
    planes.fill(planeCur);
    for (int delta = 1; delta <= RGY_DEGRAIN_MAX_DELTA; delta++) {
        const auto backward = frames.backwardRef(delta) ? frames.backwardRef(delta) : frames.cur;
        const auto forward = frames.forwardRef(delta) ? frames.forwardRef(delta) : frames.cur;
        planes[rgy_degrain_ref_index(delta, false)] = getPlane(backward, plane);
        planes[rgy_degrain_ref_index(delta, true)] = getPlane(forward, plane);
    }
    return planes;
}
}

void RGYFilterDegrain::loadDebugEnv() {
    m_debugEnv.applyTrace = degrainTraceEnvEnabled("QSVENC_DEGRAIN_APPLY_TRACE");
    m_debugEnv.applyTraceBlock = degrainTraceEnvInt("QSVENC_DEGRAIN_APPLY_TRACE_BLOCK", -1);
    m_debugEnv.forceDegrainCopy = degrainTraceEnvEnabled("QSVENC_DEGRAIN_DEGRAIN_COPY");
    m_debugEnv.pixelTrace = degrainTraceEnvEnabled("QSVENC_DEGRAIN_PIXEL_TRACE");
    m_debugEnv.pixelTraceX = degrainTraceEnvInt("QSVENC_DEGRAIN_PIXEL_TRACE_X", 0);
    m_debugEnv.pixelTraceY = degrainTraceEnvInt("QSVENC_DEGRAIN_PIXEL_TRACE_Y", 0);
    m_debugEnv.pixelTraceFrame = degrainTraceEnvInt("QSVENC_DEGRAIN_PIXEL_TRACE_FRAME", -1);
}

bool RGYFilterDegrain::degrainApplyTraceEnabled() const {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDegrain>(m_param);
    return m_debugEnv.applyTrace
        && prm
        && prm->degrain.mode == VppDegrainMode::Degrain
        && (prm->degrain.stage == VppDegrainStage::TR1 || prm->degrain.stage == VppDegrainStage::TR2);
}

bool RGYFilterDegrain::analysisSADIncludesChroma(const std::shared_ptr<RGYFilterParamDegrain> &prm) const {
    if (!prm || !prm->degrain.chroma) {
        return false;
    }
    if (m_boundAnalyzeResult.valid()) {
        return (m_boundAnalyzeResult.flags & RGY_DEGRAIN_FRAME_META_FLAG_CHROMA_SAD) != 0;
    }
    return m_lastAnalysisIncludedChroma;
}

void RGYFilterDegrain::logApplyTrace(const std::shared_ptr<RGYFilterParamDegrain> &prm, const RGYFilterDegrainProcessFrameSet &frames,
    const RGYDegrainRefDisableArray &disableRefs, RGYOpenCLQueue &queue) {
    if (!degrainApplyTraceEnabled() || !prm) {
        return;
    }
    auto *mv = analysisMV();
    auto *sad = analysisSAD();
    const auto &layout = analysisLayout();
    const auto *cur = frames.render.cur ? frames.render.cur : frames.analysis.cur;
    if (!mv || !sad || !cur || layout.blockCount() == 0 || layout.temporalDirections <= 0) {
        return;
    }
    const auto entryCount = layout.blockCount() * (size_t)layout.temporalDirections;
    if (layout.mvCount() < entryCount || layout.sadCount() < entryCount) {
        AddMessage(RGY_LOG_INFO, _T("degrain apply trace skipped: invalid entry count (blocks=%llu, stride=%d).\n"),
            (unsigned long long)layout.blockCount(), layout.temporalDirections);
        return;
    }

    std::vector<RGYOpenCLEvent> waitEvents;
    if (analysisEvent()() != nullptr) {
        waitEvents.push_back(analysisEvent());
    }
    auto err = mv->queueMapBuffer(queue, CL_MAP_READ, waitEvents, RGY_CL_MAP_BLOCK_ALL, "degrain.apply.trace.mv");
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_WARN, _T("degrain apply trace failed to map MV buffer: %s.\n"), get_err_mes(err));
        return;
    }
    err = sad->queueMapBuffer(queue, CL_MAP_READ, waitEvents, RGY_CL_MAP_BLOCK_ALL, "degrain.apply.trace.sad");
    if (err != RGY_ERR_NONE) {
        mv->unmapBuffer(queue);
        AddMessage(RGY_LOG_WARN, _T("degrain apply trace failed to map SAD buffer: %s.\n"), get_err_mes(err));
        return;
    }

    const auto *mvValues = reinterpret_cast<const RGYDegrainMV *>(mv->mappedPtr());
    const auto *sadValues = reinterpret_cast<const RGYDegrainSAD *>(sad->mappedPtr());
    if (!mvValues || !sadValues) {
        sad->unmapBuffer(queue);
        mv->unmapBuffer(queue);
        AddMessage(RGY_LOG_WARN, _T("degrain apply trace failed to access mapped MV/SAD buffers.\n"));
        return;
    }

    auto availabilityDisableRefs = analysisAvailabilityDisableRefs(frames.analysis);
    auto useFlagDisableRefs = RGYDegrainRefDisableArray();
    useFlagDisableRefs.fill(false);
    if (prm->degrain.useFlag == 1) {
        for (int delta = 1; delta <= RGY_DEGRAIN_MAX_DELTA; delta++) {
            useFlagDisableRefs[rgy_degrain_ref_index(delta, true)] = true;
        }
    } else if (prm->degrain.useFlag == 2) {
        for (int delta = 1; delta <= RGY_DEGRAIN_MAX_DELTA; delta++) {
            useFlagDisableRefs[rgy_degrain_ref_index(delta, false)] = true;
        }
    }

    const uint32_t scaledThSad = rgy_degrain_scale_sad_threshold(prm->degrain, prm->frameOut, prm->degrain.thsad, analysisSADIncludesChroma(prm));
    const uint32_t disableMask = degrainDisableMask(disableRefs, layout.temporalDirections);
    const bool binomial = prm->degrain.stage != VppDegrainStage::TR2;
    const auto temporalMixPrior = degrainBuildTemporalMixPriorTable(layout.temporalDirections, binomial);
    const float sourceConfidenceRaw = temporalMixPrior[0];
    const int refDirectionCount = std::min(layout.temporalDirections, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS);
    const TCHAR *stageName = get_cx_desc(list_vpp_degrain_stage, (int)prm->degrain.stage);

    std::array<size_t, 4> sampleBlocks = {
        0,
        layout.blockCount() / 2,
        layout.blockCount() - 1,
        layout.blockCount()
    };
    static const TCHAR *sampleNames[] = { _T("first"), _T("mid"), _T("last"), _T("target") };
    const int targetBlock = m_debugEnv.applyTraceBlock;
    if (targetBlock >= 0 && (size_t)targetBlock < layout.blockCount()) {
        sampleBlocks[3] = (size_t)targetBlock;
    }
    for (int sample = 0; sample < (int)sampleBlocks.size(); sample++) {
        const size_t block = sampleBlocks[sample];
        if (block >= layout.blockCount()
            || (sample > 0 && block == sampleBlocks[sample - 1])
            || (sample > 1 && block == sampleBlocks[sample - 2])) {
            continue;
        }
        std::array<float, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> referenceConfidenceRaw = {};
        float confidenceSum = sourceConfidenceRaw;
        float referenceMixTotal = 0.0f;
        for (int refDirection = 0; refDirection < refDirectionCount; refDirection++) {
            const size_t entry = block * (size_t)layout.temporalDirections + (size_t)refDirection;
            const auto &mvValue = mvValues[entry];
            const auto &sadValue = sadValues[entry];
            const bool directionDisabled = disableRefs[refDirection];
            const bool validMotion = (int)mvValue.refdir == refDirection;
            const bool underThSad = sadValue.sad < scaledThSad;
            if (!directionDisabled && validMotion && underThSad) {
                referenceConfidenceRaw[refDirection] = degrainTraceReferenceAffinityFromSad((int)scaledThSad, (int)sadValue.sad)
                    * temporalMixPrior[1 + refDirection];
                confidenceSum += referenceConfidenceRaw[refDirection];
            }
        }
        const float invWeightSum = (confidenceSum > 0.0f) ? (1.0f / confidenceSum) : 0.0f;
        for (int refDirection = 0; refDirection < refDirectionCount; refDirection++) {
            referenceMixTotal += (referenceConfidenceRaw[refDirection] > 0.0f) ? (referenceConfidenceRaw[refDirection] * invWeightSum) : 0.0f;
        }
        const float sourceMixNorm = sourceConfidenceRaw * invWeightSum;

        const int blockX = (layout.blocksX > 0) ? (int)(block % (size_t)layout.blocksX) : 0;
        const int blockY = (layout.blocksX > 0) ? (int)(block / (size_t)layout.blocksX) : 0;
        for (int refDirection = 0; refDirection < refDirectionCount; refDirection++) {
            const size_t entry = block * (size_t)layout.temporalDirections + (size_t)refDirection;
            const auto &mvValue = mvValues[entry];
            const auto &sadValue = sadValues[entry];
            const int delta = rgy_degrain_delta_from_ref_index(refDirection);
            const bool forward = rgy_degrain_ref_index_is_forward(refDirection);
            const bool availabilityDisabled = availabilityDisableRefs[refDirection];
            const bool useFlagDisabled = useFlagDisableRefs[refDirection] && !availabilityDisabled;
            const bool sceneChangeDisabled = disableRefs[refDirection] && !availabilityDisabled && !useFlagDisabled;
            const bool validMotion = (int)mvValue.refdir == refDirection;
            const bool underThSad = sadValue.sad < scaledThSad;
            const float referenceMixNorm = (referenceConfidenceRaw[refDirection] > 0.0f) ? (referenceConfidenceRaw[refDirection] * invWeightSum) : 0.0f;
            AddMessage(RGY_LOG_INFO,
                _T("{\"type\":\"degrain_mix_trace\",\"frame\":%d,\"pts\":%lld,\"dur\":%lld,\"stage\":\"%s\",\"request_delta\":%d,\"delta\":%d,\"ref_slot\":%d,\"temporal_side\":\"%s\",\"sample\":\"%s\",\"block\":%llu,\"block_x\":%d,\"block_y\":%d,\"entry\":%llu,\"motion\":{\"dx\":%d,\"dy\":%d,\"sad\":%u,\"ref_slot\":%u},\"sad_stats\":{\"sad\":%u,\"src_avg\":%u,\"ref_avg\":%u},\"sad_limit\":%u,\"reference_policy\":%d,\"disable\":{\"mask\":%u,\"availability\":%d,\"policy\":%d,\"scene\":%d,\"final\":%d},\"valid\":{\"motion\":%d,\"sad\":%d},\"selected\":%d,\"mix\":{\"confidence_raw\":%.9g,\"reference_norm\":%.9g,\"source_norm\":%.9g,\"reference_norm_sum\":%.9g,\"confidence_sum\":%.9g,\"source_raw\":%.9g},\"layout\":{\"blocks\":%llu,\"blocks_x\":%d,\"blocks_y\":%d,\"directions\":%d,\"block_size\":%d,\"overlap\":%d,\"step\":%d}}\n"),
                cur->inputFrameId, (long long)cur->timestamp, (long long)cur->duration,
                stageName, requestedDelta(), delta, refDirection, forward ? _T("prev") : _T("next"),
                sampleNames[sample],
                (unsigned long long)block, blockX, blockY, (unsigned long long)entry,
                (int)mvValue.dx, (int)mvValue.dy, (unsigned int)mvValue.sad, (unsigned int)mvValue.refdir,
                (unsigned int)sadValue.sad, (unsigned int)sadValue.srcAvg, (unsigned int)sadValue.refAvg,
                (unsigned int)scaledThSad, prm->degrain.useFlag,
                (unsigned int)disableMask, availabilityDisabled ? 1 : 0, useFlagDisabled ? 1 : 0, sceneChangeDisabled ? 1 : 0, disableRefs[refDirection] ? 1 : 0,
                validMotion ? 1 : 0, underThSad ? 1 : 0, referenceMixNorm > 0.0f ? 1 : 0,
                (double)referenceConfidenceRaw[refDirection], (double)referenceMixNorm, (double)sourceMixNorm, (double)referenceMixTotal, (double)confidenceSum, (double)sourceConfidenceRaw,
                (unsigned long long)layout.blockCount(), layout.blocksX, layout.blocksY, layout.temporalDirections, layout.blockSize, layout.overlap, layout.step);
        }
    }

    err = sad->unmapBuffer(queue);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_WARN, _T("degrain apply trace failed to unmap SAD buffer: %s.\n"), get_err_mes(err));
    }
    err = mv->unmapBuffer(queue);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_WARN, _T("degrain apply trace failed to unmap MV buffer: %s.\n"), get_err_mes(err));
    }
}

RGY_ERR RGYFilterDegrain::emitDebugFrame(const RGYFilterDegrainFrameSet &frames, VppDegrainMode mode,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    if (!frames.cur || frames.cur->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return RGY_ERR_NONE;
    }
    auto *mv = analysisMV();
    auto *sad = analysisSAD();
    if (!mv || !sad) {
        AddMessage(RGY_LOG_ERROR, _T("degrain debug output requires analysis buffers.\n"));
        return RGY_ERR_INVALID_CALL;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        ppOutputFrames[0] = &m_frameBuf[0]->frame;
    }

    const auto memcpyKind = getMemcpyKind(frames.cur->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != RGYCLMemcpyD2D) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    RGYOpenCLEvent copyEvent;
    std::vector<RGYOpenCLEvent> copyWaitEvents;
    if (analysisEvent()() != nullptr) {
        copyWaitEvents.push_back(analysisEvent());
    }
    auto err = m_cl->copyFrame(ppOutputFrames[0], frames.cur, nullptr, queue, copyWaitEvents, &copyEvent, RGYFrameCopyMode::FRAME, "degrain.debug_output");
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to prepare degrain debug output: %s.\n"), get_err_mes(err));
        return err;
    }
    copyFramePropWithoutRes(ppOutputFrames[0], frames.cur);

    auto planeDst = getPlane(ppOutputFrames[0], RGY_PLANE_Y);
    const char *kernel_name = nullptr;
    switch (mode) {
    case VppDegrainMode::MV:
        kernel_name = "kernel_degrain_debug_mv";
        break;
    case VppDegrainMode::SAD:
        kernel_name = "kernel_degrain_debug_sad";
        break;
    default:
        AddMessage(RGY_LOG_ERROR, _T("invalid debug mode for degrain: %s.\n"), get_cx_desc(list_vpp_degrain_mode, (int)mode));
        return RGY_ERR_INVALID_CALL;
    }

    RGYWorkSize local(DEGRAIN_DEBUG_BLOCK_X, DEGRAIN_DEBUG_BLOCK_Y);
    RGYWorkSize global(planeDst.width, planeDst.height);
    err = m_degrain.get()->kernel(kernel_name).config(queue, local, global, { copyEvent }, event).launch(
        (cl_mem)planeDst.ptr[0], planeDst.pitch[0],
        planeDst.width, planeDst.height,
        mv->mem(),
        sad->mem(),
        analysisLayout().blocksX,
        analysisLayout().blocksY,
        analysisLayout().blockSize,
        analysisLayout().overlap,
        analysisLayout().step,
        analysisLayout().coveredWidth,
        analysisLayout().coveredHeight);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to render degrain %s debug frame: %s.\n"),
            get_cx_desc(list_vpp_degrain_mode, (int)mode), get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDegrain::emitCompensateFrame(const RGYFilterDegrainFrameSet &frames, VppDegrainMode mode,
    const RGYDegrainRefDisableArray &disableRefs,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    if (!frames.cur || frames.cur->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return RGY_ERR_NONE;
    }
    auto *mv = analysisMV();
    auto *sad = analysisSAD();
    if (!mv || !sad) {
        AddMessage(RGY_LOG_ERROR, _T("degrain compensate output requires analysis buffers.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    if (!m_analysis.temporalMixPrior) {
        AddMessage(RGY_LOG_ERROR, _T("degrain compensate output requires temporal mix prior table.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDegrain>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    RGYDegrainRefDir refDirection = RGYDegrainRefDir::Backward;
    if (!rgy_degrain_refdir_from_mode(mode, &refDirection)) {
        AddMessage(RGY_LOG_ERROR, _T("invalid compensate mode for degrain: %s.\n"), get_cx_desc(list_vpp_degrain_mode, (int)mode));
        return RGY_ERR_INVALID_CALL;
    }
    const int refIndex = rgy_degrain_refdir_index(refDirection);
    if (refIndex >= analysisLayout().temporalDirections) {
        AddMessage(RGY_LOG_ERROR, _T("degrain compensate mode %s is not available for delta=%d.\n"),
            get_cx_desc(list_vpp_degrain_mode, (int)mode), rgy_degrain_delta_from_ref_index(analysisLayout().temporalDirections - 1));
        return RGY_ERR_UNSUPPORTED;
    }

    const int refDelta = rgy_degrain_delta_from_ref_index(refIndex);
    const bool refForward = rgy_degrain_ref_index_is_forward(refIndex);
    const RGYFrameInfo *reference = refForward ? frames.forwardRef(refDelta) : frames.backwardRef(refDelta);
    if (!reference || reference->ptr[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("degrain compensate mode %s requires a valid reference frame.\n"),
            get_cx_desc(list_vpp_degrain_mode, (int)mode));
        return RGY_ERR_INVALID_CALL;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        ppOutputFrames[0] = &m_frameBuf[0]->frame;
    }

    const auto memcpyKind = getMemcpyKind(frames.cur->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != RGYCLMemcpyD2D) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    RGYOpenCLEvent copyEvent;
    std::vector<RGYOpenCLEvent> copyWaitEvents;
    if (analysisEvent()() != nullptr) {
        copyWaitEvents.push_back(analysisEvent());
    }
    auto err = m_cl->copyFrame(ppOutputFrames[0], frames.cur, nullptr, queue, copyWaitEvents, &copyEvent, RGYFrameCopyMode::FRAME, "degrain.compensate_output");
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to prepare degrain %s output: %s.\n"),
            get_cx_desc(list_vpp_degrain_mode, (int)mode), get_err_mes(err));
        return err;
    }
    copyFramePropWithoutRes(ppOutputFrames[0], frames.cur);

    RGYWorkSize local(DEGRAIN_DEBUG_BLOCK_X, DEGRAIN_DEBUG_BLOCK_Y);
    // scene-change/reference availability can still disable
    // a direction, but MotionBack/MotionForw must not use per-block thSAD fallback.
    const uint32_t compensateThSad = std::numeric_limits<uint32_t>::max();
    const uint32_t disableMask = degrainDisableMask(disableRefs, analysisLayout().temporalDirections);
    const bool useOverlapRamp = analysisLayout().overlap > 0;
    auto ensureWindowRamp = [&](RGYDegrainWindowRampState &state, const int planeScaleX, const int planeScaleY) {
        const int planeOverlapX = std::max(analysisLayout().overlap / std::max(planeScaleX, 1), 0);
        const int planeOverlapY = std::max(analysisLayout().overlap / std::max(planeScaleY, 1), 0);
        const auto rampBytes = degrainOverlapBlendTableBytes(planeOverlapX, planeOverlapY);
        if (rampBytes == 0) {
            state.reset();
            return RGY_ERR_NONE;
        }
        if (state.reusable(planeOverlapX, planeOverlapY, rampBytes)) {
            return RGY_ERR_NONE;
        }

        std::vector<float> ramp(planeOverlapX + planeOverlapY);
        degrainFillOverlapBlendAxis(ramp.data(), planeOverlapX);
        degrainFillOverlapBlendAxis(ramp.data() + planeOverlapX, planeOverlapY);
        auto rampBuf = m_cl->copyDataToBuffer(ramp.data(), rampBytes, CL_MEM_READ_ONLY, queue.get());
        if (!rampBuf) {
            state.reset();
            return RGY_ERR_MEMORY_ALLOC;
        }
        state.ramp = std::move(rampBuf);
        state.bytes = rampBytes;
        state.overlapX = planeOverlapX;
        state.overlapY = planeOverlapY;
        return RGY_ERR_NONE;
    };

    auto renderPlane = [&](const RGY_PLANE plane, const uint32_t planeThSad, const std::vector<RGYOpenCLEvent> &waitEvents, RGYOpenCLEvent *planeEvent) {
        const auto planeDst = getPlane(ppOutputFrames[0], plane);
        const auto planeCur = getPlane(frames.cur, plane);
        const auto refPlanes = degrainRenderRefPlanes(frames, plane);
        if (planeDst.ptr[0] == nullptr || planeCur.ptr[0] == nullptr || refPlanes[0].ptr[0] == nullptr || refPlanes[1].ptr[0] == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("degrain compensate mode %s requires valid plane %d.\n"),
                get_cx_desc(list_vpp_degrain_mode, (int)mode), (int)plane);
            return RGY_ERR_INVALID_CALL;
        }
        const int planePitch = planeCur.pitch[0];
        for (int i = 0; i < (int)refPlanes.size(); i++) {
            if (refPlanes[i].pitch[0] != planePitch) {
                AddMessage(RGY_LOG_ERROR, _T("degrain compensate mode %s requires matching plane %d pitch: cur=%d, ref[%d]=%d.\n"),
                    get_cx_desc(list_vpp_degrain_mode, (int)mode), (int)plane, planePitch, i, refPlanes[i].pitch[0]);
                return RGY_ERR_INVALID_PARAM;
            }
        }
        const int planeScaleX = degrainPlaneScaleX(frames.cur, plane);
        const int planeScaleY = degrainPlaneScaleY(frames.cur, plane);
        auto *program = degrainRenderProgram(plane);
        if (program == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build degrain render program for plane %d.\n"), (int)plane);
            return RGY_ERR_NULL_PTR;
        }
        const auto refPlane = refPlanes[refIndex];
        auto planeWaitEvents = waitEvents;
        RGYCLBuf *windowRamp = nullptr;
        if (useOverlapRamp) {
            auto &rampState = (plane == RGY_PLANE_Y) ? m_analysis.windowRampY : m_analysis.windowRampC;
            const auto rampErr = ensureWindowRamp(rampState, planeScaleX, planeScaleY);
            if (rampErr != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to prepare degrain overlap ramp buffer: %s.\n"), get_err_mes(rampErr));
                return rampErr;
            } else if (rampState.ramp) {
                windowRamp = rampState.ramp.get();
            }
        }
        if (windowRamp) {
            return program->kernel("kernel_degrain_compensate_overlap_plane_ramp").config(
                queue, local, RGYWorkSize(planeDst.width, planeDst.height), planeWaitEvents, planeEvent).launch(
                (cl_mem)planeDst.ptr[0], planeDst.pitch[0],
                (cl_mem)planeCur.ptr[0], planePitch,
                (cl_mem)planeCur.ptr[0],
                (cl_mem)refPlane.ptr[0],
                refIndex,
                planeDst.width, planeDst.height,
                mv->mem(),
                sad->mem(),
                analysisLayout().blocksX,
                analysisLayout().blocksY,
                analysisLayout().blockSize,
                analysisLayout().overlap,
                analysisLayout().step,
                degrainScaleCovered(analysisLayout().coveredWidth, planeScaleX),
                degrainScaleCovered(analysisLayout().coveredHeight, planeScaleY),
                planeScaleX,
                planeScaleY,
                planeThSad,
                disableMask,
                windowRamp->mem());
        }
        return program->kernel("kernel_degrain_overlap_plane").config(queue, local, RGYWorkSize(planeDst.width, planeDst.height), planeWaitEvents, planeEvent).launch(
            (cl_mem)planeDst.ptr[0], planeDst.pitch[0],
            (cl_mem)planeCur.ptr[0], planePitch,
            (cl_mem)planeCur.ptr[0],
            (cl_mem)refPlanes[0].ptr[0],
            (cl_mem)refPlanes[1].ptr[0],
            (cl_mem)refPlanes[2].ptr[0],
            (cl_mem)refPlanes[3].ptr[0],
            (cl_mem)refPlanes[4].ptr[0],
            (cl_mem)refPlanes[5].ptr[0],
            (cl_mem)refPlanes[6].ptr[0],
            (cl_mem)refPlanes[7].ptr[0],
            (cl_mem)refPlanes[8].ptr[0],
            (cl_mem)refPlanes[9].ptr[0],
            planeDst.width, planeDst.height,
            mv->mem(),
            sad->mem(),
            m_analysis.temporalMixPrior->mem(),
            analysisLayout().blocksX,
            analysisLayout().blocksY,
            analysisLayout().blockSize,
            analysisLayout().overlap,
            analysisLayout().step,
            degrainScaleCovered(analysisLayout().coveredWidth, planeScaleX),
            degrainScaleCovered(analysisLayout().coveredHeight, planeScaleY),
            planeScaleX,
            planeScaleY,
            (mode == VppDegrainMode::MotionBack || mode == VppDegrainMode::MotionBack2) ? 0 : 1,
            refIndex,
            planeThSad,
            disableMask);
    };

    const bool processChroma = degrainCanProcessChroma(frames.cur);
    const std::array<RGY_PLANE, 3> planes = { RGY_PLANE_Y, RGY_PLANE_U, RGY_PLANE_V };
    auto waitEvents = std::vector<RGYOpenCLEvent>{ copyEvent };
    for (int iplane = 0; iplane < (processChroma ? (int)planes.size() : 1); iplane++) {
        RGYOpenCLEvent renderEvent;
        const auto plane = planes[iplane];
        const bool lastPlane = iplane == (processChroma ? (int)planes.size() : 1) - 1;
        err = renderPlane(plane, compensateThSad, waitEvents, lastPlane ? event : &renderEvent);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to render degrain %s %s output: %s.\n"),
                plane == RGY_PLANE_Y ? _T("luma") : _T("chroma"),
                get_cx_desc(list_vpp_degrain_mode, (int)mode), get_err_mes(err));
            return err;
        }
        if (!lastPlane) {
            waitEvents = { renderEvent };
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDegrain::resolveSceneChange(const std::shared_ptr<RGYFilterParamDegrain> &prm, const RGYFilterDegrainFrameSet &frames, RGYOpenCLQueue &queue,
    bool *disableBackward, bool *disableForward) {
    if (!disableBackward || !disableForward) {
        return RGY_ERR_INVALID_PARAM;
    }
    RGYDegrainRefDisableArray disableRefs;
    auto err = resolveSceneChangeRefs(prm, frames, queue, &disableRefs);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    *disableBackward = disableRefs[rgy_degrain_refdir_index(RGYDegrainRefDir::Backward)];
    *disableForward = disableRefs[rgy_degrain_refdir_index(RGYDegrainRefDir::Forward)];
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDegrain::submitSceneChangeReadback(const std::shared_ptr<RGYFilterParamDegrain> &prm, const RGYFilterDegrainProcessFrameSet &frames,
    RGYOpenCLQueue &queue, PendingSceneChange *pending) {
    if (!pending) {
        return RGY_ERR_INVALID_PARAM;
    }
    *pending = PendingSceneChange();
    pending->prm = prm;
    pending->frames = frames;
    pending->frameAnalysisData = m_frameAnalysisData;
    pending->boundAnalyzeResult = m_boundAnalyzeResult;
    pending->frameAnalysisLayout = m_frameAnalysisLayout;
    pending->layout = analysisLayout();

    const auto availabilityDisableRefs = analysisAvailabilityDisableRefs(frames.analysis);
    auto useFlagDisableRefs = RGYDegrainRefDisableArray();
    useFlagDisableRefs.fill(false);
    auto disableRefs = availabilityDisableRefs;
    if (prm) {
        if (prm->degrain.useFlag == 1) {
            for (int delta = 1; delta <= RGY_DEGRAIN_MAX_DELTA; delta++) {
                const int refIndex = rgy_degrain_ref_index(delta, true);
                useFlagDisableRefs[refIndex] = true;
                disableRefs[refIndex] = true;
            }
        } else if (prm->degrain.useFlag == 2) {
            for (int delta = 1; delta <= RGY_DEGRAIN_MAX_DELTA; delta++) {
                const int refIndex = rgy_degrain_ref_index(delta, false);
                useFlagDisableRefs[refIndex] = true;
                disableRefs[refIndex] = true;
            }
        }
    }

    auto *sad = analysisSAD();
    const bool includeChromaSad = analysisSADIncludesChroma(prm);
    const uint32_t scaledThSad = (prm) ? rgy_degrain_scale_sad_threshold(prm->degrain, prm->frameOut, prm->degrain.thsad, includeChromaSad) : 0;
    const uint32_t scaledThSCD1 = (prm) ? rgy_degrain_scale_sad_threshold(prm->degrain, prm->frameOut, prm->degrain.thscd1, includeChromaSad) : 0;
    const uint64_t scaledThSCD2 = (prm) ? rgy_degrain_scale_scene_change_block_threshold(pending->layout.blockCount(), prm->degrain.thscd2) : 0;
    pending->availabilityDisableRefs = availabilityDisableRefs;
    pending->useFlagDisableRefs = useFlagDisableRefs;
    pending->disableRefs = disableRefs;
    pending->scaledThSad = scaledThSad;
    pending->scaledThSCD1 = scaledThSCD1;
    pending->scaledThSCD2 = scaledThSCD2;
    pending->sad = sad;
    if (!prm || !sad || pending->layout.blockCount() == 0) {
        return RGY_ERR_NONE;
    }
    bool allDirectionsDisabled = true;
    for (int refDirection = 0; refDirection < std::min(pending->layout.temporalDirections, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS); refDirection++) {
        allDirectionsDisabled &= pending->disableRefs[refDirection];
    }
    if (allDirectionsDisabled) {
        return RGY_ERR_NONE;
    }

    std::vector<RGYOpenCLEvent> mapWaitEvents;
    if (analysisEvent()() != nullptr) {
        mapWaitEvents.push_back(analysisEvent());
    }
    auto err = sad->queueMapBuffer(queue, CL_MAP_READ, mapWaitEvents, RGY_CL_MAP_BLOCK_NONE, "degrain.apply.scene.sad");
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to map degrain SAD buffer for scene change detection: %s.\n"), get_err_mes(err));
        return err;
    }
    pending->mapEvent = sad->mapEvent();
    pending->mapSubmitted = true;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDegrain::resolveSceneChangeReadback(PendingSceneChange &pending, RGYOpenCLQueue &queue, RGYDegrainRefDisableArray *disableRefs) {
    if (!disableRefs) {
        return RGY_ERR_INVALID_PARAM;
    }
    *disableRefs = pending.disableRefs;
    if (!pending.prm || !pending.sad || pending.layout.blockCount() == 0) {
        logReferenceGate(pending.prm, pending.frames.analysis, pending.availabilityDisableRefs, pending.useFlagDisableRefs,
            *disableRefs, nullptr, pending.scaledThSad, pending.scaledThSCD1, pending.scaledThSCD2);
        return RGY_ERR_NONE;
    }
    if (!pending.mapSubmitted) {
        logReferenceGate(pending.prm, pending.frames.analysis, pending.availabilityDisableRefs, pending.useFlagDisableRefs,
            *disableRefs, nullptr, pending.scaledThSad, pending.scaledThSCD1, pending.scaledThSCD2);
        return RGY_ERR_NONE;
    }
    pending.mapEvent.wait();

    const auto *sadValues = reinterpret_cast<const RGYDegrainSAD *>(pending.sad->mappedPtr());
    if (!sadValues) {
        pending.sad->unmapBuffer(queue);
        AddMessage(RGY_LOG_ERROR, _T("failed to access degrain SAD buffer for scene change detection.\n"));
        return RGY_ERR_NULL_PTR;
    }

    std::array<size_t, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> sceneChangeBlockCounts = {};
    for (size_t block = 0; block < pending.layout.blockCount(); block++) {
        for (int refDirection = 0; refDirection < std::min(pending.layout.temporalDirections, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS); refDirection++) {
            if (pending.disableRefs[refDirection]) {
                continue;
            }
            const auto &sadValue = sadValues[block * (size_t)pending.layout.temporalDirections + (size_t)refDirection];
            if (sadValue.sad > pending.scaledThSCD1) {
                sceneChangeBlockCounts[refDirection]++;
            }
        }
    }
    auto err = pending.sad->unmapBuffer(queue);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to unmap degrain SAD buffer for scene change detection: %s.\n"), get_err_mes(err));
        return err;
    }
    pending.mapSubmitted = false;

    for (int refDirection = 0; refDirection < std::min(pending.layout.temporalDirections, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS); refDirection++) {
        if (!pending.disableRefs[refDirection]) {
            pending.disableRefs[refDirection] = (uint64_t)sceneChangeBlockCounts[refDirection] > pending.scaledThSCD2;
        }
    }
    *disableRefs = pending.disableRefs;
    logReferenceGate(pending.prm, pending.frames.analysis, pending.availabilityDisableRefs, pending.useFlagDisableRefs,
        *disableRefs, &sceneChangeBlockCounts, pending.scaledThSad, pending.scaledThSCD1, pending.scaledThSCD2);
    return RGY_ERR_NONE;
}

void RGYFilterDegrain::clearPendingSceneChange() {
    if (m_pendingSceneChange && m_pendingSceneChange->mapSubmitted && m_pendingSceneChange->sad) {
        m_pendingSceneChange->mapEvent.wait();
        m_pendingSceneChange->sad->unmapBuffer();
        m_pendingSceneChange->mapSubmitted = false;
    }
    m_pendingSceneChange.reset();
}

void RGYFilterDegrain::applyPendingSceneChangeAnalysisContext(const PendingSceneChange &pending) {
    m_frameAnalysisData = pending.frameAnalysisData;
    m_boundAnalyzeResult = pending.boundAnalyzeResult;
    m_frameAnalysisLayout = pending.frameAnalysisLayout;
}

RGY_ERR RGYFilterDegrain::resolvePendingSceneChangeFrame(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    if (!m_pendingSceneChange) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return RGY_ERR_NONE;
    }
    applyPendingSceneChangeAnalysisContext(*m_pendingSceneChange);
    RGYDegrainRefDisableArray disableRefs;
    auto err = resolveSceneChangeReadback(*m_pendingSceneChange, queue, &disableRefs);
    if (err != RGY_ERR_NONE) {
        m_pendingSceneChange.reset();
        return err;
    }
    logApplyTrace(m_pendingSceneChange->prm, m_pendingSceneChange->frames, disableRefs, queue);
    err = emitDegrainFrame(m_pendingSceneChange->frames.render, disableRefs, ppOutputFrames, pOutputFrameNum, queue, event);
    m_pendingSceneChange.reset();
    return err;
}

RGY_ERR RGYFilterDegrain::resolveSceneChangeRefs(const std::shared_ptr<RGYFilterParamDegrain> &prm, const RGYFilterDegrainFrameSet &frames, RGYOpenCLQueue &queue,
    RGYDegrainRefDisableArray *disableRefs) {
    PendingSceneChange pending;
    RGYFilterDegrainProcessFrameSet processFrames = {};
    processFrames.render = frames;
    processFrames.analysis = frames;
    auto err = submitSceneChangeReadback(prm, processFrames, queue, &pending);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    return resolveSceneChangeReadback(pending, queue, disableRefs);
}

RGY_ERR RGYFilterDegrain::emitDegrainFrame(const RGYFilterDegrainFrameSet &frames,
    const RGYDegrainRefDisableArray &disableRefs,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    if (!frames.cur || frames.cur->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return RGY_ERR_NONE;
    }
    auto *mv = analysisMV();
    auto *sad = analysisSAD();
    if (!mv || !sad) {
        AddMessage(RGY_LOG_ERROR, _T("degrain degrain output requires analysis buffers.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    if (!m_analysis.temporalMixPrior) {
        AddMessage(RGY_LOG_ERROR, _T("degrain degrain output requires temporal mix prior table.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDegrain>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        ppOutputFrames[0] = &m_frameBuf[0]->frame;
    }

    const auto memcpyKind = getMemcpyKind(frames.cur->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != RGYCLMemcpyD2D) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    const auto planeDstY = getPlane(ppOutputFrames[0], RGY_PLANE_Y);
    const auto planeCurY = getPlane(frames.cur, RGY_PLANE_Y);
    if (planeDstY.ptr[0] == nullptr || planeCurY.ptr[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("degrain degrain mode requires valid luma planes.\n"));
        return RGY_ERR_INVALID_CALL;
    }

    // Keep chroma for SAD handling; degrain render should process available YUV chroma planes independently of that option.
    const bool processChroma = degrainCanProcessChroma(frames.cur);
    const bool copyDegrainOutput = m_debugEnv.forceDegrainCopy
        || rgy_csp_has_alpha(frames.cur->csp)
        || RGY_CSP_PLANES[frames.cur->csp] != (processChroma ? 3 : 1)
        || (!processChroma && RGY_CSP_CHROMA_FORMAT[frames.cur->csp] != RGY_CHROMAFMT_MONOCHROME);
    RGYOpenCLEvent copyEvent;
    std::vector<RGYOpenCLEvent> initialWaitEvents;
    if (analysisEvent()() != nullptr) {
        initialWaitEvents.push_back(analysisEvent());
    }
    auto err = RGY_ERR_NONE;
    if (copyDegrainOutput) {
        err = m_cl->copyFrame(ppOutputFrames[0], frames.cur, nullptr, queue, initialWaitEvents, &copyEvent, RGYFrameCopyMode::FRAME, "degrain.degrain_prepare_copy");
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to prepare degrain degrain output: %s.\n"), get_err_mes(err));
            return err;
        }
        initialWaitEvents = { copyEvent };
    }
    copyFramePropWithoutRes(ppOutputFrames[0], frames.cur);

    RGYWorkSize local(DEGRAIN_DEBUG_BLOCK_X, DEGRAIN_DEBUG_BLOCK_Y);
    const uint32_t disableMask = degrainDisableMask(disableRefs, analysisLayout().temporalDirections);
    const bool useOverlapRamp = analysisLayout().overlap > 0;
    const bool pixelTrace = m_debugEnv.pixelTrace;
    const int pixelTraceX = m_debugEnv.pixelTraceX;
    const int pixelTraceY = m_debugEnv.pixelTraceY;
    const int pixelTraceFrame = m_debugEnv.pixelTraceFrame;
    auto ensureWindowRamp = [&](RGYDegrainWindowRampState &state, const int planeScaleX, const int planeScaleY) {
        const int planeOverlapX = std::max(analysisLayout().overlap / std::max(planeScaleX, 1), 0);
        const int planeOverlapY = std::max(analysisLayout().overlap / std::max(planeScaleY, 1), 0);
        const auto rampBytes = degrainOverlapBlendTableBytes(planeOverlapX, planeOverlapY);
        if (rampBytes == 0) {
            state.reset();
            return RGY_ERR_NONE;
        }
        if (state.reusable(planeOverlapX, planeOverlapY, rampBytes)) {
            return RGY_ERR_NONE;
        }

        std::vector<float> ramp(planeOverlapX + planeOverlapY);
        degrainFillOverlapBlendAxis(ramp.data(), planeOverlapX);
        degrainFillOverlapBlendAxis(ramp.data() + planeOverlapX, planeOverlapY);
        auto rampBuf = m_cl->copyDataToBuffer(ramp.data(), rampBytes, CL_MEM_READ_ONLY, queue.get());
        if (!rampBuf) {
            state.reset();
            return RGY_ERR_MEMORY_ALLOC;
        }
        state.ramp = std::move(rampBuf);
        state.bytes = rampBytes;
        state.overlapX = planeOverlapX;
        state.overlapY = planeOverlapY;
        return RGY_ERR_NONE;
    };
    bool temporalMixPlanYReady = false;
    bool temporalMixPlanCReady = false;
    auto ensureTemporalMixPlan = [&](RGYDegrainTemporalMixPlanState &state, bool &ready, const uint32_t scaledThSad,
        const std::vector<RGYOpenCLEvent> &waitEvents) {
        const auto planBytes = degrainTemporalMixPlanBytes(analysisLayout());
        if (ready && state.reusable(planBytes, scaledThSad, disableMask)) {
            return RGY_ERR_NONE;
        }
        if (planBytes == 0) {
            AddMessage(RGY_LOG_ERROR, _T("invalid degrain temporal mix plan buffer geometry.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        if (!state.plan || state.plan->size() != planBytes) {
            state.plan = m_cl->createBuffer(planBytes, CL_MEM_READ_WRITE);
            if (!state.plan) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain temporal mix plan buffer.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
        }

        auto err = m_degrain.get()->kernel("kernel_degrain_build_temporal_mix_plan").config(
            queue,
            RGYWorkSize(256),
            RGYWorkSize((int)analysisLayout().blockCount()),
            waitEvents,
            &state.event).launch(
                state.plan->mem(),
                mv->mem(),
                sad->mem(),
                m_analysis.temporalMixPrior->mem(),
                (int)analysisLayout().blockCount(),
                scaledThSad,
                disableMask);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build degrain temporal mix plan: %s.\n"), get_err_mes(err));
            state.event.reset();
            return err;
        }
        state.bytes = planBytes;
        state.thsad = scaledThSad;
        state.disableMask = disableMask;
        ready = true;
        return RGY_ERR_NONE;
    };
    auto renderPlane = [&](const RGY_PLANE plane, const uint32_t scaledThSad, const std::vector<RGYOpenCLEvent> &waitEvents, RGYOpenCLEvent *planeEvent) {
        const auto planeDst = getPlane(ppOutputFrames[0], plane);
        const auto planeCur = getPlane(frames.cur, plane);
        const auto refPlanes = degrainRenderRefPlanes(frames, plane);
        if (planeDst.ptr[0] == nullptr || planeCur.ptr[0] == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("degrain degrain mode requires valid plane %d.\n"), (int)plane);
            return RGY_ERR_INVALID_CALL;
        }
        const int planePitch = planeCur.pitch[0];
        for (int i = 0; i < (int)refPlanes.size(); i++) {
            if (refPlanes[i].pitch[0] != planePitch) {
                AddMessage(RGY_LOG_ERROR, _T("degrain degrain mode requires matching plane %d pitch: cur=%d, ref[%d]=%d.\n"),
                    (int)plane, planePitch, i, refPlanes[i].pitch[0]);
                return RGY_ERR_INVALID_PARAM;
            }
        }
        const int planeScaleX = degrainPlaneScaleX(frames.cur, plane);
        const int planeScaleY = degrainPlaneScaleY(frames.cur, plane);
        auto *program = degrainRenderProgram(plane);
        if (program == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build degrain render program for plane %d.\n"), (int)plane);
            return RGY_ERR_NULL_PTR;
        }
        auto planeWaitEvents = waitEvents;
        RGYCLBuf *windowRamp = nullptr;
        if (useOverlapRamp) {
            auto &rampState = (plane == RGY_PLANE_Y) ? m_analysis.windowRampY : m_analysis.windowRampC;
            const auto rampErr = ensureWindowRamp(rampState, planeScaleX, planeScaleY);
            if (rampErr != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to prepare degrain overlap ramp buffer: %s.\n"), get_err_mes(rampErr));
                return rampErr;
            } else if (rampState.ramp) {
                windowRamp = rampState.ramp.get();
            }
        }
        RGYWorkSize global(planeDst.width, planeDst.height);
        RGYCLBuf *temporalMixPlan = nullptr;
        if (windowRamp) {
            auto &planState = (plane == RGY_PLANE_Y) ? m_analysis.temporalMixPlanY : m_analysis.temporalMixPlanC;
            auto &planReady = (plane == RGY_PLANE_Y) ? temporalMixPlanYReady : temporalMixPlanCReady;
            auto err = ensureTemporalMixPlan(planState, planReady, scaledThSad, waitEvents);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (planState.plan && planState.event() != nullptr) {
                planeWaitEvents.push_back(planState.event);
                temporalMixPlan = planState.plan.get();
            } else {
                AddMessage(RGY_LOG_ERROR, _T("degrain temporal mix plan buffer was not prepared.\n"));
                return RGY_ERR_INVALID_CALL;
            }
        }
        if (pixelTrace && plane == RGY_PLANE_Y
            && (pixelTraceFrame < 0 || planeCur.inputFrameId == pixelTraceFrame)) {
            auto traceBuf = m_cl->createBuffer(sizeof(cl_int) * 256, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
            if (!traceBuf) {
                AddMessage(RGY_LOG_WARN, _T("failed to allocate degrain pixel trace buffer.\n"));
            } else {
                RGYOpenCLEvent traceEvent;
                auto traceErr = program->kernel("kernel_degrain_pixel_trace").config(queue, RGYWorkSize(1), RGYWorkSize(1), planeWaitEvents, &traceEvent).launch(
                    (cl_mem)planeCur.ptr[0], planePitch,
                    (cl_mem)refPlanes[0].ptr[0],
                    (cl_mem)refPlanes[1].ptr[0],
                    (cl_mem)refPlanes[2].ptr[0],
                    (cl_mem)refPlanes[3].ptr[0],
                    (cl_mem)refPlanes[4].ptr[0],
                    (cl_mem)refPlanes[5].ptr[0],
                    (cl_mem)refPlanes[6].ptr[0],
                    (cl_mem)refPlanes[7].ptr[0],
                    (cl_mem)refPlanes[8].ptr[0],
                    (cl_mem)refPlanes[9].ptr[0],
                    planeDst.width, planeDst.height,
                    mv->mem(),
                    sad->mem(),
                    m_analysis.temporalMixPrior->mem(),
                    analysisLayout().blocksX,
                    analysisLayout().blocksY,
                    analysisLayout().blockSize,
                    analysisLayout().overlap,
                    analysisLayout().step,
                    degrainScaleCovered(analysisLayout().coveredWidth, planeScaleX),
                    degrainScaleCovered(analysisLayout().coveredHeight, planeScaleY),
                    planeScaleX,
                    planeScaleY,
                    scaledThSad,
                    disableMask,
                    pixelTraceX,
                    pixelTraceY,
                    traceBuf->mem());
                if (traceErr != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_WARN, _T("failed to launch degrain pixel trace: %s.\n"), get_err_mes(traceErr));
                } else {
                    traceErr = traceBuf->queueMapBuffer(queue, CL_MAP_READ, { traceEvent }, RGY_CL_MAP_BLOCK_ALL, "degrain.apply.trace.result");
                    if (traceErr != RGY_ERR_NONE) {
                        AddMessage(RGY_LOG_WARN, _T("failed to map degrain pixel trace: %s.\n"), get_err_mes(traceErr));
                    } else {
                        logDegrainPixelTraceRecords(m_pLog.get(), reinterpret_cast<const int *>(traceBuf->mappedPtr()),
                            planeCur, plane, -1, prm->degrain.stage, requestedDelta());
                        traceErr = traceBuf->unmapBuffer(queue);
                        if (traceErr != RGY_ERR_NONE) {
                            AddMessage(RGY_LOG_WARN, _T("failed to unmap degrain pixel trace: %s.\n"), get_err_mes(traceErr));
                        }
                    }
                }
            }
        }
        if (windowRamp) {
            return program->kernel("kernel_degrain_degrain_overlap_plane_preweighted_ramp").config(
                queue, local, global, planeWaitEvents, planeEvent).launch(
                (cl_mem)planeDst.ptr[0], planeDst.pitch[0],
                (cl_mem)planeCur.ptr[0], planePitch,
                (cl_mem)refPlanes[0].ptr[0],
                (cl_mem)refPlanes[1].ptr[0],
                (cl_mem)refPlanes[2].ptr[0],
                (cl_mem)refPlanes[3].ptr[0],
                (cl_mem)refPlanes[4].ptr[0],
                (cl_mem)refPlanes[5].ptr[0],
                (cl_mem)refPlanes[6].ptr[0],
                (cl_mem)refPlanes[7].ptr[0],
                (cl_mem)refPlanes[8].ptr[0],
                (cl_mem)refPlanes[9].ptr[0],
                planeDst.width, planeDst.height,
                mv->mem(),
                analysisLayout().blocksX,
                analysisLayout().blocksY,
                analysisLayout().blockSize,
                analysisLayout().overlap,
                analysisLayout().step,
                degrainScaleCovered(analysisLayout().coveredWidth, planeScaleX),
                degrainScaleCovered(analysisLayout().coveredHeight, planeScaleY),
                planeScaleX,
                planeScaleY,
                windowRamp->mem(),
                temporalMixPlan->mem());
        }
        return program->kernel("kernel_degrain_degrain_overlap_plane").config(queue, local, global, planeWaitEvents, planeEvent).launch(
            (cl_mem)planeDst.ptr[0], planeDst.pitch[0],
            (cl_mem)planeCur.ptr[0], planePitch,
            (cl_mem)refPlanes[0].ptr[0],
            (cl_mem)refPlanes[1].ptr[0],
            (cl_mem)refPlanes[2].ptr[0],
            (cl_mem)refPlanes[3].ptr[0],
            (cl_mem)refPlanes[4].ptr[0],
            (cl_mem)refPlanes[5].ptr[0],
            (cl_mem)refPlanes[6].ptr[0],
            (cl_mem)refPlanes[7].ptr[0],
            (cl_mem)refPlanes[8].ptr[0],
            (cl_mem)refPlanes[9].ptr[0],
            planeDst.width, planeDst.height,
            mv->mem(),
            sad->mem(),
            m_analysis.temporalMixPrior->mem(),
            analysisLayout().blocksX,
            analysisLayout().blocksY,
            analysisLayout().blockSize,
            analysisLayout().overlap,
            analysisLayout().step,
            degrainScaleCovered(analysisLayout().coveredWidth, planeScaleX),
            degrainScaleCovered(analysisLayout().coveredHeight, planeScaleY),
            planeScaleX,
            planeScaleY,
            scaledThSad,
            disableMask);
    };

    const bool includeChromaSad = analysisSADIncludesChroma(prm);
    const uint32_t scaledThSad = rgy_degrain_scale_sad_threshold(prm->degrain, prm->frameOut, prm->degrain.thsad, includeChromaSad);
    const uint32_t scaledThSadC = rgy_degrain_scale_sad_threshold(prm->degrain, prm->frameOut, prm->degrain.thsadc, includeChromaSad);
    const std::array<RGY_PLANE, 3> planes = { RGY_PLANE_Y, RGY_PLANE_U, RGY_PLANE_V };
    auto waitEvents = initialWaitEvents;
    for (int iplane = 0; iplane < (processChroma ? (int)planes.size() : 1); iplane++) {
        RGYOpenCLEvent renderEvent;
        const auto plane = planes[iplane];
        const bool lastPlane = iplane == (processChroma ? (int)planes.size() : 1) - 1;
        err = renderPlane(plane, plane == RGY_PLANE_Y ? scaledThSad : scaledThSadC, waitEvents, lastPlane ? event : &renderEvent);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to render degrain degrain %s output: %s.\n"),
                plane == RGY_PLANE_Y ? _T("luma") : _T("chroma"), get_err_mes(err));
            return err;
        }
        if (!lastPlane) {
            waitEvents = { renderEvent };
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDegrain::runDebugMode(const RGYFilterDegrainProcessFrameSet &frames, const int currentFrame, VppDegrainMode mode, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!bindFrameAnalysisData(frames.render.cur, currentFrame, queue)) {
        auto err = prepareAnalysisState(frames.analysis, queue, wait_events);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    return emitDebugFrame(frames.render, mode, ppOutputFrames, pOutputFrameNum, queue, event);
}

RGY_ERR RGYFilterDegrain::runCompensateMode(const RGYFilterDegrainProcessFrameSet &frames, const int currentFrame, VppDegrainMode mode, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!bindFrameAnalysisData(frames.render.cur, currentFrame, queue)) {
        auto err = prepareAnalysisState(frames.analysis, queue, wait_events);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    RGYDegrainRefDisableArray disableRefs;
    auto err = resolveSceneChangeRefs(std::dynamic_pointer_cast<RGYFilterParamDegrain>(m_param), frames.analysis, queue, &disableRefs);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    return emitCompensateFrame(frames.render, mode, disableRefs, ppOutputFrames, pOutputFrameNum, queue, event);
}

RGY_ERR RGYFilterDegrain::runDegrainMode(const RGYFilterDegrainProcessFrameSet &frames, const int currentFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDegrain>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    std::unique_ptr<PendingSceneChange> pendingOutput;
    if (m_pendingSceneChange) {
        pendingOutput = std::move(m_pendingSceneChange);
        applyPendingSceneChangeAnalysisContext(*pendingOutput);
        RGYDegrainRefDisableArray disableRefs;
        auto err = resolveSceneChangeReadback(*pendingOutput, queue, &disableRefs);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        logApplyTrace(pendingOutput->prm, pendingOutput->frames, disableRefs, queue);
        err = emitDegrainFrame(pendingOutput->frames.render, disableRefs, ppOutputFrames, pOutputFrameNum, queue, event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }

    if (!bindFrameAnalysisData(frames.render.cur, currentFrame, queue)) {
        auto err = prepareAnalysisState(frames.analysis, queue, wait_events);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }

    const bool canDeferSceneChange = !m_boundAnalyzeResult.valid() || m_frameAnalysisData;
    if (!canDeferSceneChange) {
        PendingSceneChange pending;
        auto err = submitSceneChangeReadback(prm, frames, queue, &pending);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        RGYDegrainRefDisableArray disableRefs;
        err = resolveSceneChangeReadback(pending, queue, &disableRefs);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        logApplyTrace(prm, frames, disableRefs, queue);
        return emitDegrainFrame(frames.render, disableRefs, ppOutputFrames, pOutputFrameNum, queue, event);
    }

    auto pending = std::make_unique<PendingSceneChange>();
    auto err = submitSceneChangeReadback(prm, frames, queue, pending.get());
    if (err != RGY_ERR_NONE) {
        return err;
    }
    m_pendingSceneChange = std::move(pending);
    if (pendingOutput) {
        return RGY_ERR_NONE;
    }

    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    return RGY_ERR_NONE;
}
