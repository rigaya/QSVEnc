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

#include "rgy_filter_kfm.h"
#include "rgy_filter_rtgmc.h"
#include "rgy_filter_degrain.h"
#include "rgy_filesystem.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <exception>
#include <limits>

static const std::array<const TCHAR *, 8> KFM_RESOURCE_NAMES = {
    _T("RGY_FILTER_KFM_COMMON_CL"),
    _T("RGY_FILTER_KFM_PAD_CL"),
    _T("RGY_FILTER_KFM_STATIC_CL"),
    _T("RGY_FILTER_KFM_ANALYZE_CL"),
    _T("RGY_FILTER_KFM_RENDER_CL"),
    _T("RGY_FILTER_KFM_MASK_CL"),
    _T("RGY_FILTER_KFM_SWITCH_CL"),
    _T("RGY_FILTER_KFM_UCF_CL"),
};

enum KfmProgramIndex {
    KFM_PROG_COMMON = 0,
    KFM_PROG_PAD,
    KFM_PROG_STATIC,
    KFM_PROG_ANALYZE,
    KFM_PROG_RENDER,
    KFM_PROG_MASK,
    KFM_PROG_SWITCH,
    KFM_PROG_UCF,
};

namespace {
static constexpr int KFM_SOURCE_VPAD = 4;
static constexpr int KFM_FMCOUNT_PAIRS = 9;
static constexpr int KFM_FMCOUNT_COUNT = KFM_FMCOUNT_PAIRS * 2;
static constexpr int KFM_FMCOUNT_SOURCE_FRAMES = KFM_FMCOUNT_PAIRS + 2;
static constexpr int KFM_FMCOUNT_ASYNC_DELAY_CYCLES = 1;
static constexpr int KFM_THRESH_MOVE_Y = 20;
static constexpr int KFM_THRESH_SHIMA_Y = 12;
static constexpr int KFM_THRESH_MOVE_C = 24;
static constexpr int KFM_THRESH_SHIMA_C = 16;
static constexpr int KFM_CLEAN_THRESH_Y = 10;
static constexpr int KFM_CLEAN_THRESH_C = 8;
static constexpr int KFM_REMOVE_COMBE_THRESH_Y = 6;
static constexpr int KFM_REMOVE_COMBE_THRESH_C = 6;
static constexpr int KFM_SWITCH_FLAG_THRESH_Y = 60;
static constexpr int KFM_SWITCH_FLAG_THRESH_C = 80;
static constexpr int KFM_REALTIMEPLUS_SOURCE_CACHE_MARGIN = 64;
static constexpr int KFM_REALTIMEPLUS_DEINT60_CACHE_MARGIN = KFM_REALTIMEPLUS_SOURCE_CACHE_MARGIN * 2;
static constexpr int KFM_VFR_SOURCE_TRIM_LOOKBEHIND = 8;
static constexpr int KFM_VFR_DEINT60_TRIM_LOOKBEHIND = 16;
static constexpr int KFM_UCF_NOISE_LIMIT_NMIN = 1;
static constexpr int KFM_UCF_NOISE_LIMIT_RANGE = 128;
static constexpr int KFM_UCF_SHARED_ANALYSIS_SOURCE_DELAY = 2;
static constexpr double KFM_UCF_GAUSS_P = 2.5;
static constexpr double KFM_UCF_GAUSS_CROP_EPS = 0.0001;

static void resetKfmFrameState(RGYFrameInfo& frame) {
    frame.timestamp = 0;
    frame.duration = 0;
    frame.picstruct = RGY_PICSTRUCT_UNKNOWN;
    frame.flags = RGY_FRAME_FLAG_NONE;
    frame.inputFrameId = -1;
    frame.dataList.clear();
}

static double kfmUcfGaussValue(const double value, const double p) {
    const auto param = std::min(std::max(p, 0.1), 100.0);
    return std::pow(2.0, -(param * 0.1) * value * value);
}

static int kfmFrameParity(const RGYFrameInfo *frame) {
    return (frame && (frame->picstruct & RGY_PICSTRUCT_BFF)) ? 0 : 1;
}

static int kfmFloorDiv2(const int value) {
    const int q = value / 2;
    return q - ((value < 0 && (value & 1)) ? 1 : 0);
}

static bool kfmCspHasInterleavedUV(const RGY_CSP csp) {
    return csp == RGY_CSP_NV12 || csp == RGY_CSP_P010
        || csp == RGY_CSP_NV16 || csp == RGY_CSP_P210;
}

static const char *kfmUcfKernelName(VppKfmMode mode) {
    switch (mode) {
    case VppKfmMode::P60:
        return "kernel_kfm_ucf_60";
    case VppKfmMode::P24:
        return "kernel_kfm_ucf_24";
    case VppKfmMode::VFR:
    default:
        return "kernel_kfm_ucf_param";
    }
}

static RGYFrameInfo *kfmDebugStageFrame(VppKfmDebugStage stage, RGYFrameInfo *switchFlag, RGYFrameInfo *containsCombe, RGYFrameInfo *combeMask) {
    switch (stage) {
    case VppKfmDebugStage::SwitchFlag:
        return switchFlag;
    case VppKfmDebugStage::ContainsCombe:
        return containsCombe;
    case VppKfmDebugStage::CombeMask:
        return combeMask;
    case VppKfmDebugStage::None:
    default:
        return nullptr;
    }
}

static int kfmDepthScale(RGY_CSP csp) {
    return 1 << std::max(0, RGY_CSP_BIT_DEPTH[csp] - 8);
}

static int kfmPow2Shift(int scale) {
    if (scale <= 0 || (scale & (scale - 1)) != 0) {
        return -1;
    }
    int shift = 0;
    while ((1 << shift) < scale) {
        shift++;
    }
    return shift;
}

static bool kfmDeint60BranchEnabled() {
    const char *env = std::getenv("QSVENC_KFM_ENABLE_DEINT60_BRANCH");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

static bool kfmForceEagerRtgmc() {
    const char *env = std::getenv("QSVENC_KFM_FORCE_EAGER_RTGMC");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

static bool kfmDisableCCDuration() {
    const char *env = std::getenv("QSVENC_KFM_DISABLE_CC_DURATION");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

static bool kfmUcfNoGaussForTest() {
    const char *env = std::getenv("QSVENC_KFM_UCF_NO_GAUSS");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

static bool kfmUseFusedFMCount() {
    const char *env = std::getenv("QSVENC_KFM_FMCOUNT_FUSED");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    return _stricmp(env, "0") != 0 && _stricmp(env, "false") != 0 && _stricmp(env, "off") != 0;
}

static bool kfmUseFusedSwitchFlagBinaryExtend() {
    const char *env = std::getenv("QSVENC_KFM_SWITCH_FLAG_BINARY_EXTEND_FUSED");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    return _stricmp(env, "0") != 0 && _stricmp(env, "false") != 0 && _stricmp(env, "off") != 0;
}

static bool kfmUseFusedCleanSuper() {
    const char *env = std::getenv("QSVENC_KFM_CLEAN_SUPER_FUSED");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    return _stricmp(env, "0") != 0 && _stricmp(env, "false") != 0 && _stricmp(env, "off") != 0;
}

static bool kfmUseFusedUcfPreprocess() {
    const char *env = std::getenv("QSVENC_KFM_UCF_PREPROCESS_FUSED");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    return _stricmp(env, "0") != 0 && _stricmp(env, "false") != 0 && _stricmp(env, "off") != 0;
}

static std::string kfmStageDumpName(const char *stage) {
    std::string name = (stage && stage[0]) ? stage : "stage";
    for (auto& c : name) {
        if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
            c = '_';
        }
    }
    return name + ".y4m";
}

struct KfmUcfCalcDumpInfo {
    const char *classification;
    double fieldDiff;
    double diff;
};

static double kfmUcfCalcNoiseDiff(const RGYKFM::UCFNoiseMeta& meta, const RGYKFM::DecombUCFParam& param,
    const RGYKFM::NoiseResult *result0, const RGYKFM::NoiseResult *result1, bool second) {
    const double noisepixels = static_cast<double>(meta.noisew) * meta.noiseh;
    const double noisepixelsUV = static_cast<double>(meta.noiseUVw) * meta.noiseUVh * 2.0;

    const double noise_t_y = (second ? result0[0].noise1 : result0[0].noise0) / noisepixels;
    const double noise_t_uv = (second ? result0[1].noise1 : result0[1].noise0) / noisepixelsUV;
    const double noise_b_y = (second ? result1[0].noise0 : result0[0].noise1) / noisepixels;
    const double noise_b_uv = (second ? result1[1].noise0 : result0[1].noise1) / noisepixelsUV;
    const double diff1_y = noise_t_y - noise_b_y;
    const double diff1_uv = noise_t_uv - noise_b_uv;

    double diff1 = 0.0;
    if (param.chroma == 0) {
        diff1 = diff1_y;
    } else if (param.chroma == 1) {
        diff1 = diff1_uv;
    } else {
        diff1 = (diff1_y + diff1_uv) / 2.0;
    }

    const double absdiff1 = std::abs(diff1);
    return absdiff1 < 1.8 ? diff1 * 10.0
        : absdiff1 < 5.0 ? diff1 * 5.0 + (diff1 / absdiff1) * 9.0
        : absdiff1 < 10.0 ? diff1 * 2.0 + (diff1 / absdiff1) * 24.0
        : diff1 + (diff1 / absdiff1) * 34.0;
}

static KfmUcfCalcDumpInfo kfmUcfCalcDumpInfo(const RGYKFM::UCFNoiseMeta& meta, const RGYKFM::NoiseResult *result0,
    const RGYKFM::NoiseResult *result1, bool second) {
    RGYKFM::DecombUCFParam param;
    const auto classification = RGYKFM::CalcDecombUCF(&meta, &param, result0, result1, second);
    const double pixels = static_cast<double>(meta.srcw) * meta.srch;
    const double fieldDiff = (second
        ? static_cast<double>(result0[0].diff1 + result0[1].diff1)
        : static_cast<double>(result0[0].diff0 + result0[1].diff0)) / (6.0 * pixels) * 100.0;
    return { RGYKFM::decombUCFResultToString(classification), fieldDiff, kfmUcfCalcNoiseDiff(meta, param, result0, result1, second) };
}
}

tstring RGYFilterParamKfm::print() const {
    return kfm.print();
}

RGYFilterKfm::RGYFilterKfm(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_programs(),
    m_rtgmc(),
    m_deint60Rtgmc(),
    m_before60Rtgmc(),
    m_after60Rtgmc(),
    m_nrFilter(),
    m_analyzer(),
    m_kfmFramePool(),
    m_kfmSourceSlotFree(),
    m_kfmSourceSlotRetired(),
    m_sourceCache(),
    m_deint60Lane(),
    m_before60Lane(),
    m_after60Lane(),
    m_ucfNoiseCache(),
    m_pendingUcfNoiseResults(),
    m_fmCountBufPool(),
    m_ucfNoiseResultBufPool(),
    m_ucfNoiseResultCache(),
    m_pendingUcfNoiseDump(),
    m_staticFlag(),
    m_staticWorkFrames(),
    m_analyzeFlags(),
    m_fmCountQueue(),
    m_pendingFMCounts(),
    m_pendingVfrOutputs(),
    m_telecineSuperRaw(),
    m_telecineSuperFrames(),
    m_telecineSuperNeighborFrames(),
    m_switchFlagFrames(),
    m_containsCombeFrames(),
    m_combeMaskFrames(),
    m_patchCombeFrames(),
    m_ucfNoiseFieldFrames(),
    m_ucfNoiseGaussTmpFrames(),
    m_ucfNoiseGaussFrames(),
    m_ucfNoiseGaussVert(),
    m_ucfNoiseGaussHori(),
    m_switchFlagWork(),
    m_switchFlagWorkEvent(),
    m_containsCombeCount(),
    m_fpResult(nullptr),
    m_fpFMCount(nullptr),
    m_fpTimecode(nullptr),
    m_fpFrameInfo(nullptr),
    m_fpContainsCombe(nullptr),
    m_fpUcfNoise(nullptr),
    m_switchDurationPath(),
    m_switchTimecodePath(),
    m_stageDumpDir(),
    m_lastAnalyzeResult(),
    m_analyzerOutputResults(),
    m_hasLastAnalyzeResult(false),
    m_analyzerFinalized(false),
    m_switchTimingDumped(false),
    m_analyzeSourceFrames(0),
    m_nextAnalyzeCycle(0),
    m_nextFMCountSubmitCycle(0),
    m_nextFMCountDumpFrame(0),
    m_cachedSourceFrames(0),
    m_nextSwitchN60(0),
    m_nextSwitchPts(0),
    m_hasLastSwitchTiming(false),
    m_lastSwitchStart60(0),
    m_lastSwitchDuration60(0),
    m_lastSwitchStart120(0),
    m_lastSwitchIsFrame24(false),
    m_switchSingleFrameN60(),
    m_stageDumpFrameCounts(),
    m_stageDumpFrameIndices(),
    m_stageDumpTargetFrames(),
    m_nextTelecine24Frame(0),
    m_nextTelecine24Pts(0),
    m_telecineSuperBufferIndex(0),
    m_maskBranchBufferIndex(0),
    m_patchCombeBufferIndex(0),
    m_stageDumpMaxFrames(0),
    m_timecodeFrameIndex(0),
    m_outputBufferIndex(0),
    m_workFrameBuf(),
    m_workBufferIndex(0) {
    m_name = _T("kfm");
}

RGYFilterKfm::~RGYFilterKfm() {
    close();
}

RGYFilterKfm::KfmRtgmcLane::KfmRtgmcLane() :
    m_owner(nullptr),
    m_rtgmc(nullptr),
    m_stage(nullptr),
    m_cacheLabel(nullptr),
    m_dumpStaticFlag(false),
    m_cache(),
    m_submittedFrames(0),
    m_nextFeedSourceIndex(-1),
    m_nextOutputN60(0),
    m_hotUntilSourceIndex(-1),
    m_cacheFloorN60(0),
    m_feedCount(0),
    m_cacheCopyEvent() {
}

void RGYFilterKfm::KfmRtgmcLane::init(RGYFilterKfm *owner, RGYFilterRtgmc *rtgmc, const char *stage, const TCHAR *cacheLabel, bool dumpStaticFlag) {
    m_owner = owner;
    m_rtgmc = rtgmc;
    m_stage = stage;
    m_cacheLabel = cacheLabel;
    m_dumpStaticFlag = dumpStaticFlag;
    reset();
}

void RGYFilterKfm::KfmRtgmcLane::clear() {
    m_cache.clear();
    m_submittedFrames = 0;
    m_cacheCopyEvent = RGYOpenCLEvent();
}

void RGYFilterKfm::KfmRtgmcLane::reset() {
    clear();
    m_nextFeedSourceIndex = -1;
    m_nextOutputN60 = 0;
    m_hotUntilSourceIndex = -1;
    m_cacheFloorN60 = 0;
    m_feedCount = 0;
}

RGY_ERR RGYFilterKfm::KfmRtgmcLane::feed(const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, int *cachedFrames) {
    if (cachedFrames) {
        *cachedFrames = 0;
    }
    if (!m_rtgmc) {
        return RGY_ERR_NONE;
    }

    int rtgmcOutNum = 0;
    RGYFrameInfo *rtgmcOutFrames[8] = { 0 };
    RGYOpenCLEvent rtgmcEvent;
    if (frame && frame->ptr[0]) {
        m_feedCount++;
    }
    auto sts = m_rtgmc->filter(const_cast<RGYFrameInfo *>(frame), rtgmcOutFrames, &rtgmcOutNum, queue, wait_events, &rtgmcEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    std::vector<RGYOpenCLEvent> cacheWaitEvents;
    if (rtgmcEvent() != nullptr) {
        cacheWaitEvents.push_back(rtgmcEvent);
    }
    for (int i = 0; i < rtgmcOutNum; i++) {
        RGYOpenCLEvent cacheEvent;
        sts = cacheFrame(rtgmcOutFrames[i], queue, cacheWaitEvents, &cacheEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (cacheEvent() != nullptr) {
            m_cacheCopyEvent = cacheEvent;
        }
        if (cachedFrames) {
            (*cachedFrames)++;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::KfmRtgmcLane::drain(RGYOpenCLQueue &queue, int maxDrainIterations, int *cachedFrames) {
    if (cachedFrames) {
        *cachedFrames = 0;
    }
    if (!m_rtgmc) {
        return RGY_ERR_NONE;
    }
    for (int iter = 0; !m_rtgmc->drainComplete(); iter++) {
        if (iter >= maxDrainIterations) {
            m_owner->AddMessage(RGY_LOG_ERROR, _T("KFM %S RTGMC drain did not complete after %d iterations.\n"), m_stage, maxDrainIterations);
            return RGY_ERR_INVALID_CALL;
        }
        int drainedFrames = 0;
        auto sts = feed(nullptr, queue, {}, &drainedFrames);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (cachedFrames) {
            *cachedFrames += drainedFrames;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::KfmRtgmcLane::drainTo(int n60end, RGYOpenCLQueue &queue) {
    if (!m_rtgmc) {
        return RGY_ERR_NONE;
    }
    const auto maxDrainIterations = std::max(256, m_owner ? m_owner->m_cachedSourceFrames * 4 + 256 : 256);
    for (int iter = 0; m_nextOutputN60 < n60end && !m_rtgmc->drainComplete(); iter++) {
        if (iter >= maxDrainIterations) {
            m_owner->AddMessage(RGY_LOG_ERROR, _T("KFM %S RTGMC demand drain did not reach n60=%d after %d iterations.\n"),
                m_stage, n60end, maxDrainIterations);
            return RGY_ERR_INVALID_CALL;
        }
        auto sts = feed(nullptr, queue, {});
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_nextOutputN60 = m_submittedFrames;
    }
    return (m_nextOutputN60 >= n60end) ? RGY_ERR_NONE : RGY_ERR_MORE_DATA;
}

RGY_ERR RGYFilterKfm::KfmRtgmcLane::ensureRange(int n60begin, int n60end, RGYOpenCLQueue &queue) {
    if (!m_rtgmc || n60begin >= n60end) {
        return RGY_ERR_NONE;
    }
    n60begin = std::max(0, n60begin);
    n60end = std::max(n60begin, n60end);

    bool cached = true;
    for (int n60 = n60begin; n60 < n60end; n60++) {
        if (!find(n60, nullptr)) {
            cached = false;
            break;
        }
    }
    if (cached) {
        return RGY_ERR_NONE;
    }

    const int sourceBegin = n60begin >> 1;
    const bool cold = m_nextFeedSourceIndex < 0;
    const bool rewind = m_nextOutputN60 > n60begin;
    const bool feedPastRequest = !cold && m_nextFeedSourceIndex > sourceBegin && m_nextOutputN60 <= n60begin;
    const bool beyondHot = !cold && m_hotUntilSourceIndex >= 0 && sourceBegin > m_hotUntilSourceIndex;
    if (cold || rewind || feedPastRequest || beyondHot) {
        m_rtgmc->resetTemporalState();
        m_cache.clear();
        m_cacheCopyEvent = RGYOpenCLEvent();
        const int primingFrames = m_rtgmc->requiredPrimingSourceFrames();
        const int primeStart = std::max(0, sourceBegin - primingFrames);
        m_nextFeedSourceIndex = primeStart;
        m_submittedFrames = primeStart * 2;
        m_nextOutputN60 = m_submittedFrames;
        m_hotUntilSourceIndex = -1;
    }

    m_cacheFloorN60 = n60begin;
    while (m_nextOutputN60 < n60end) {
        if (!m_owner) {
            return RGY_ERR_INVALID_CALL;
        }
        if (m_nextFeedSourceIndex >= m_owner->m_cachedSourceFrames) {
            if (m_owner->m_analyzerFinalized) {
                return drainTo(n60end, queue);
            }
            return RGY_ERR_MORE_DATA;
        }
        const auto *source = m_owner->findSourceByIndexExact(m_nextFeedSourceIndex);
        if (!source || !source->frame || !source->frame->frame.ptr[0]) {
            if (m_owner->m_analyzerFinalized) {
                return drainTo(n60end, queue);
            }
            return RGY_ERR_MORE_DATA;
        }
        std::vector<RGYOpenCLEvent> waitEvents;
        if (source->event() != nullptr) {
            waitEvents.push_back(source->event);
        }
        if (m_owner && (m_rtgmc == m_owner->m_before60Rtgmc.get() || m_rtgmc == m_owner->m_after60Rtgmc.get())) {
            m_owner->pushDeint60Intermediates(m_rtgmc, source->sourceIndex);
        }
        auto sts = feed(&source->frame->frame, queue, waitEvents);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (m_owner && m_rtgmc == m_owner->m_deint60Rtgmc.get()) {
            m_owner->captureDeint60Intermediates(source->sourceIndex);
        }
        m_nextFeedSourceIndex++;
        m_nextOutputN60 = m_submittedFrames;
    }
    m_hotUntilSourceIndex = std::max(m_hotUntilSourceIndex, m_nextFeedSourceIndex + HOT_KEEP_SOURCE_FRAMES - 1);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::KfmRtgmcLane::feedHot(RGYOpenCLQueue &queue) {
    if (!m_rtgmc || !m_owner || m_nextFeedSourceIndex < 0 || m_nextFeedSourceIndex > m_hotUntilSourceIndex) {
        return RGY_ERR_NONE;
    }
    const auto *source = m_owner->findSourceByIndexExact(m_nextFeedSourceIndex);
    if (!source || !source->frame || !source->frame->frame.ptr[0]) {
        return RGY_ERR_NONE;
    }
    std::vector<RGYOpenCLEvent> waitEvents;
    if (source->event() != nullptr) {
        waitEvents.push_back(source->event);
    }
    if (m_owner && (m_rtgmc == m_owner->m_before60Rtgmc.get() || m_rtgmc == m_owner->m_after60Rtgmc.get())) {
        m_owner->pushDeint60Intermediates(m_rtgmc, source->sourceIndex);
    }
    auto sts = feed(&source->frame->frame, queue, waitEvents);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (m_owner && m_rtgmc == m_owner->m_deint60Rtgmc.get()) {
        m_owner->captureDeint60Intermediates(source->sourceIndex);
    }
    m_nextFeedSourceIndex++;
    m_nextOutputN60 = m_submittedFrames;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::KfmRtgmcLane::cacheFrame(const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!frame || !frame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (!m_owner || !m_stage || !m_stage[0] || !m_cacheLabel || !m_owner->m_staticFlag) {
        return RGY_ERR_INVALID_CALL;
    }

    if (m_owner && (m_rtgmc == m_owner->m_before60Rtgmc.get() || m_rtgmc == m_owner->m_after60Rtgmc.get()) && frame->inputFrameId >= 0) {
        const int frameN60Base = frame->inputFrameId * 2;
        if (m_submittedFrames < frameN60Base || m_submittedFrames > frameN60Base + 1) {
            m_submittedFrames = frameN60Base;
        }
    }

    KfmCachedDeint60 entry;
    entry.n60 = m_submittedFrames++;
    if (entry.n60 < m_cacheFloorN60) {
        return RGY_ERR_NONE;
    }
    entry.inputFrameId = frame->inputFrameId;
    entry.timestamp = frame->timestamp;
    entry.duration = frame->duration;
    entry.frame = m_owner->acquireKfmFrame(*frame, m_cacheLabel);
    if (!entry.frame) {
        return RGY_ERR_MEMORY_ALLOC;
    }

    auto mergeWaitEvents = wait_events;
    const int sourceIndex = entry.n60 >> 1;
    const auto *source = m_owner->findSourceByIndexExact(sourceIndex);
    if (!source) {
        m_owner->AddMessage(RGY_LOG_ERROR, _T("KFM source frame is missing for %S output n60=%d, sourceIndex=%d, inputFrameId=%d.\n"),
            m_stage, entry.n60, sourceIndex, frame->inputFrameId);
        return RGY_ERR_INVALID_CALL;
    }
    if (source->event() != nullptr) {
        mergeWaitEvents.push_back(source->event);
    }

    const auto rawStage = m_dumpStaticFlag ? std::string("rtgmc60-raw") : (std::string(m_stage) + "-raw");
    auto sts = m_owner->dumpStageFrame(rawStage.c_str(), frame, entry.n60, queue, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    RGYOpenCLEvent staticEvent;
    sts = m_owner->analyzeStaticFlag(source->sourceIndex, queue, mergeWaitEvents, &staticEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (staticEvent() != nullptr) {
        mergeWaitEvents.push_back(staticEvent);
    }
    if (m_dumpStaticFlag) {
        sts = m_owner->dumpStageFrame("static-flag", &m_owner->m_staticFlag->frame, sourceIndex, queue,
            (staticEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ staticEvent } : mergeWaitEvents);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    sts = m_owner->mergeStatic(&entry.frame->frame, frame, &source->frame->frame, queue, mergeWaitEvents, &entry.event);
    if (sts != RGY_ERR_NONE) {
        m_owner->AddMessage(RGY_LOG_ERROR, _T("failed to merge/cache KFM %S frame: %s.\n"), m_stage, get_err_mes(sts));
        return sts;
    }
    if (event && entry.event() != nullptr) {
        *event = entry.event;
    }
    m_owner->writeFrameInfoDump(m_stage, &entry.frame->frame);
    sts = m_owner->dumpStageFrame(m_stage, &entry.frame->frame, entry.n60, queue,
        (entry.event() != nullptr) ? std::vector<RGYOpenCLEvent>{ entry.event } : std::vector<RGYOpenCLEvent>());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    m_cache.push_back(std::move(entry));
    m_owner->trimDeint60Cache(m_cache);
    return RGY_ERR_NONE;
}

const RGYFilterKfm::KfmCachedDeint60 *RGYFilterKfm::KfmRtgmcLane::find(int n60, std::vector<RGYOpenCLEvent> *wait_events) const {
    for (auto it = m_cache.rbegin(); it != m_cache.rend(); ++it) {
        if (it->n60 == n60) {
            if (wait_events && it->event() != nullptr) {
                wait_events->push_back(it->event);
            }
            return &(*it);
        }
    }
    return nullptr;
}

void RGYFilterKfm::KfmRtgmcLane::trim(int n60floor, size_t cacheLimit) {
    while (!m_cache.empty() && m_cache.front().n60 < n60floor) {
        m_cache.pop_front();
    }
    while (m_cache.size() > cacheLimit && !m_cache.empty() && m_cache.front().n60 < n60floor) {
        m_cache.pop_front();
    }
}

int RGYFilterKfm::KfmRtgmcLane::requiredPrimingSourceFrames() const {
    return m_rtgmc ? m_rtgmc->requiredPrimingSourceFrames() : 0;
}

RGY_ERR RGYFilterKfm::loadPrograms(const RGYFilterParamKfm& prm) {
    const auto bitDepth = RGY_CSP_BIT_DEPTH[prm.frameOut.csp];
    const auto options = strsprintf("-D Type=%s -D bit_depth=%d",
        bitDepth > 8 ? "ushort" : "uchar",
        bitDepth);
    for (size_t i = 0; i < KFM_RESOURCE_NAMES.size(); ++i) {
        m_programs[i].set(m_cl->buildResourceAsync(KFM_RESOURCE_NAMES[i], _T("EXE_DATA"), options.c_str()));
    }
    return RGY_ERR_NONE;
}

std::shared_ptr<RGYCLFrame> RGYFilterKfm::acquireKfmFrame(const RGYFrameInfo& info, const TCHAR *label, cl_mem_flags flags) {
    if (!m_kfmFramePool) {
        m_kfmFramePool = std::make_shared<RGYCLSharedFramePool>(m_cl);
    }
    auto frame = m_kfmFramePool->acquire(info, flags);
    if (!frame) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM %s frame.\n"), label ? label : _T("cache"));
        return frame;
    }
    resetKfmFrameState(frame->frame);
    return frame;
}

std::shared_ptr<RGYFilterKfm::KfmSourceSlot> RGYFilterKfm::acquireKfmSourceSlot(const RGYFrameInfo& sourceInfo, cl_mem_flags flags) {
    collectRetiredKfmSourceSlots();
    auto matchSlot = [&sourceInfo, flags](const std::shared_ptr<KfmSourceSlot>& slot) {
        return slot && slot->sourceFrame && slot->paddedFrame
            && !cmpFrameInfoCspResolution(&slot->sourceFrame->frame, &sourceInfo)
            && slot->sourceFrame->frame.bitdepth == sourceInfo.bitdepth
            && slot->flags == flags;
    };
    auto pooled = std::find_if(m_kfmSourceSlotFree.begin(), m_kfmSourceSlotFree.end(), matchSlot);
    if (pooled != m_kfmSourceSlotFree.end()) {
        auto slot = std::move(*pooled);
        m_kfmSourceSlotFree.erase(pooled);
        slot->readyEvent.reset();
        resetKfmFrameState(slot->sourceFrame->frame);
        resetKfmFrameState(slot->paddedFrame->frame);
        return slot;
    }

    auto paddedInfo = sourceInfo;
    paddedInfo.height += KFM_SOURCE_VPAD * 2;
    std::shared_ptr<RGYCLFrame> paddedFrame(m_cl->createFrameBuffer(paddedInfo, flags).release());
    if (!paddedFrame) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM padded source cache frame.\n"));
        return nullptr;
    }

    RGYFrameInfo viewInfo = sourceInfo;
    viewInfo.mem_type = RGY_MEM_TYPE_GPU;
    for (int i = 0; i < _countof(viewInfo.ptr); i++) {
        viewInfo.ptr[i] = nullptr;
        viewInfo.pitch[i] = 0;
    }

    const auto memBaseAlignBits = m_cl && m_cl->platform()
        ? std::max(1, m_cl->platform()->dev(0).info().mem_base_addr_align)
        : 8;
    const size_t memBaseAlignBytes = std::max<size_t>(1, (memBaseAlignBits + 7) / 8);
    const int planes = RGY_CSP_PLANES[sourceInfo.csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto parent = getPlane(&paddedFrame->frame, (RGY_PLANE)iplane);
        const auto view = getPlane(&sourceInfo, (RGY_PLANE)iplane);
        const int vpad = (parent.height - view.height) >> 1;
        if (parent.width != view.width || parent.height != view.height + vpad * 2 || vpad <= 0) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM source slot plane size (plane %d, src %dx%d, padded %dx%d).\n"),
                iplane, view.width, view.height, parent.width, parent.height);
            return nullptr;
        }
        const size_t origin = (size_t)parent.pitch[0] * vpad;
        const size_t size = (size_t)parent.pitch[0] * view.height;
        if ((origin % memBaseAlignBytes) != 0) {
            AddMessage(RGY_LOG_ERROR, _T("KFM source sub-buffer offset is not aligned (plane %d, offset %zu, align %zu).\n"),
                iplane, origin, memBaseAlignBytes);
            return nullptr;
        }

        cl_buffer_region region = { origin, size };
        cl_int clerr = CL_SUCCESS;
        cl_mem subbuf = clCreateSubBuffer((cl_mem)parent.ptr[0], flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &clerr);
        if (clerr != CL_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("failed to create KFM source sub-buffer (plane %d): %s.\n"), iplane, cl_errmes(clerr));
            for (int j = 0; j < iplane; j++) {
                if (viewInfo.ptr[j]) {
                    clReleaseMemObject((cl_mem)viewInfo.ptr[j]);
                    viewInfo.ptr[j] = nullptr;
                }
            }
            return nullptr;
        }
        viewInfo.ptr[iplane] = (uint8_t *)subbuf;
        viewInfo.pitch[iplane] = parent.pitch[0];
    }

    auto sourceFrame = std::shared_ptr<RGYCLFrame>(
        new RGYCLFrame(viewInfo, flags),
        [paddedKeepAlive = paddedFrame](RGYCLFrame *frame) {
            delete frame;
        });

    auto slot = std::make_shared<KfmSourceSlot>();
    slot->paddedFrame = paddedFrame;
    slot->sourceFrame = sourceFrame;
    slot->flags = flags;
    resetKfmFrameState(slot->sourceFrame->frame);
    resetKfmFrameState(slot->paddedFrame->frame);
    return slot;
}

void RGYFilterKfm::retireKfmSourceSlot(std::shared_ptr<KfmSourceSlot>&& slot, RGYOpenCLQueue &queue) {
    if (!slot) {
        return;
    }
    slot->sourceFrame->frame.dataList.clear();
    slot->paddedFrame->frame.dataList.clear();
    slot->readyEvent.reset();
    if (queue.getmarker(slot->readyEvent) != RGY_ERR_NONE) {
        slot->readyEvent.reset();
        queue.finish();
        m_kfmSourceSlotFree.emplace_back(std::move(slot));
    } else {
        m_kfmSourceSlotRetired.emplace_back(std::move(slot));
    }
}

void RGYFilterKfm::collectRetiredKfmSourceSlots() {
    for (auto it = m_kfmSourceSlotRetired.begin(); it != m_kfmSourceSlotRetired.end();) {
        auto& slot = *it;
        if (!slot || slot->readyEvent() == nullptr || slot->readyEvent.getInfo().status == CL_COMPLETE) {
            if (slot) {
                slot->readyEvent.reset();
                m_kfmSourceSlotFree.emplace_back(std::move(slot));
            }
            it = m_kfmSourceSlotRetired.erase(it);
        } else {
            ++it;
        }
    }
    trimFreeKfmSourceSlots();
}

void RGYFilterKfm::trimFreeKfmSourceSlots() {
    const auto keep = std::max<size_t>(16, std::min<size_t>(sourceCacheLimit(), 256) + 8);
    while (m_kfmSourceSlotFree.size() > keep) {
        m_kfmSourceSlotFree.pop_front();
    }
}

void RGYFilterKfm::clearKfmSourceSlotPool(bool wait) {
    if (wait) {
        for (auto& slot : m_kfmSourceSlotRetired) {
            if (slot && slot->readyEvent() != nullptr) {
                slot->readyEvent.wait();
                slot->readyEvent.reset();
            }
        }
    }
    m_kfmSourceSlotRetired.clear();
    m_kfmSourceSlotFree.clear();
}

void RGYFilterKfm::trimSourceCache(RGYOpenCLQueue &queue) {
    const auto trimFloor = sourceCacheTrimFloor();
    while (!m_sourceCache.empty() && m_sourceCache.front().sourceIndex < trimFloor) {
        retireKfmSourceSlot(std::move(m_sourceCache.front().slot), queue);
        m_sourceCache.pop_front();
    }
    const auto cacheLimit = sourceCacheLimit();
    while (m_sourceCache.size() > cacheLimit && !m_sourceCache.empty() && m_sourceCache.front().sourceIndex < trimFloor) {
        retireKfmSourceSlot(std::move(m_sourceCache.front().slot), queue);
        m_sourceCache.pop_front();
    }
    collectRetiredKfmSourceSlots();
}

void RGYFilterKfm::trimDeint60Cache(std::deque<KfmCachedDeint60>& cache) {
    const auto trimFloor = deint60CacheTrimFloor();
    while (!cache.empty() && cache.front().n60 < trimFloor) {
        cache.pop_front();
    }
    const auto cacheLimit = deint60CacheLimit();
    while (cache.size() > cacheLimit && !cache.empty() && cache.front().n60 < trimFloor) {
        cache.pop_front();
    }
}

RGY_ERR RGYFilterKfm::allocWorkFrameBuf(const RGYFrameInfo& frame, int frames) {
    if ((int)m_workFrameBuf.size() == frames
        && !m_workFrameBuf.empty()
        && !cmpFrameInfoCspResolution(&m_workFrameBuf[0]->frame, &frame)
        && RGY_CSP_BIT_DEPTH[m_workFrameBuf[0]->frame.csp] == RGY_CSP_BIT_DEPTH[frame.csp]) {
        bool valid = true;
        for (const auto& work : m_workFrameBuf) {
            if (!work) {
                valid = false;
                break;
            }
            for (int iplane = 0; iplane < RGY_CSP_PLANES[work->frame.csp]; iplane++) {
                if (work->frame.ptr[iplane] == nullptr) {
                    valid = false;
                    break;
                }
            }
            if (!valid) {
                break;
            }
        }
        if (valid) {
            m_workBufferIndex = 0;
            return RGY_ERR_NONE;
        }
    }
    m_workFrameBuf.clear();
    for (int i = 0; i < frames; i++) {
        auto work = m_cl->createFrameBuffer(frame);
        if (!work) {
            m_workFrameBuf.clear();
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_workFrameBuf.push_back(std::move(work));
    }
    m_workBufferIndex = 0;
    return RGY_ERR_NONE;
}

RGYFrameInfo *RGYFilterKfm::nextOutputFrame() {
    if (m_frameBuf.empty()) {
        return nullptr;
    }
    auto out = &m_frameBuf[m_outputBufferIndex]->frame;
    m_outputBufferIndex = (m_outputBufferIndex + 1) % (int)m_frameBuf.size();
    return out;
}

RGYFrameInfo *RGYFilterKfm::nextWorkFrame() {
    if (m_workFrameBuf.empty()) {
        return nullptr;
    }
    auto out = &m_workFrameBuf[m_workBufferIndex]->frame;
    m_workBufferIndex = (m_workBufferIndex + 1) % (int)m_workFrameBuf.size();
    return out;
}

RGY_ERR RGYFilterKfm::initAnalyzer(const RGYFilterParamKfm& prm) {
    if (m_cl && !m_pendingUcfNoiseResults.empty()) {
        auto sts = resolveAllUcfNoiseResults(m_cl->queue());
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    flushUcfNoiseResultDump();
    RGYKFM::KFMAnalyzeParam analyzeParam;
    analyzeParam.pastCycles = prm.kfm.pastCycles;
    analyzeParam.NGThresh = prm.kfm.thswitch;
    m_analyzer = std::make_unique<RGYKFM::KFMAnalyze>(analyzeParam);
    m_analyzeSourceFrames = 0;
    m_nextAnalyzeCycle = 0;
    m_nextFMCountSubmitCycle = 0;
    m_nextFMCountDumpFrame = 0;
    m_cachedSourceFrames = 0;
    m_timecodeFrameIndex = 0;
    m_hasLastAnalyzeResult = false;
    m_analyzerFinalized = false;
    m_switchTimingDumped = false;
    m_switchDurationPath.clear();
    m_switchTimecodePath.clear();
    m_stageDumpDir.clear();
    m_analyzerOutputResults.clear();
    m_switchSingleFrameN60.clear();
    m_stageDumpFrameCounts.clear();
    m_stageDumpFrameIndices.clear();
    auto clearFMCountSts = clearPendingFMCounts();
    m_fmCountBufPool.clear();
    if (clearFMCountSts != RGY_ERR_NONE) {
        return clearFMCountSts;
    }
    m_nextSwitchN60 = 0;
    m_nextSwitchPts = 0;
    m_hasLastSwitchTiming = false;
    m_lastSwitchStart60 = 0;
    m_lastSwitchDuration60 = 0;
    m_lastSwitchStart120 = 0;
    m_lastSwitchIsFrame24 = false;
    m_nextTelecine24Frame = 0;
    m_nextTelecine24Pts = 0;
    m_telecineSuperBufferIndex = 0;
    m_maskBranchBufferIndex = 0;
    m_patchCombeBufferIndex = 0;
    m_stageDumpMaxFrames = 0;
    m_deint60Lane.reset();
    m_before60Lane.reset();
    m_after60Lane.reset();
    m_ucfNoiseCache.clear();
    m_pendingUcfNoiseResults.clear();
    m_ucfNoiseResultBufPool.clear();
    m_ucfNoiseResultCache.clear();
    m_pendingUcfNoiseDump = KfmUcfNoiseDumpRecord();

    if (m_fpResult) {
        fclose(m_fpResult);
        m_fpResult = nullptr;
    }
    if (m_fpFMCount) {
        fclose(m_fpFMCount);
        m_fpFMCount = nullptr;
    }
    if (m_fpTimecode) {
        fclose(m_fpTimecode);
        m_fpTimecode = nullptr;
    }
    if (m_fpFrameInfo) {
        fclose(m_fpFrameInfo);
        m_fpFrameInfo = nullptr;
    }
    if (m_fpContainsCombe) {
        fclose(m_fpContainsCombe);
        m_fpContainsCombe = nullptr;
    }
    if (m_fpUcfNoise) {
        fclose(m_fpUcfNoise);
        m_fpUcfNoise = nullptr;
    }
    if (prm.kfm.timecode.length() > 0) {
        if (prm.kfm.mode == VppKfmMode::P24 || prm.kfm.mode == VppKfmMode::VFR) {
            m_switchDurationPath = PathRemoveExtensionS(prm.kfm.timecode) + _T(".duration.txt");
        }
        if (_tfopen_s(&m_fpTimecode, prm.kfm.timecode.c_str(), _T("wb")) != 0 || m_fpTimecode == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to open KFM timecode dump file \"%s\".\n"), prm.kfm.timecode.c_str());
            return RGY_ERR_FILE_OPEN;
        }
        fprintf(m_fpTimecode, "# timecode format v2\n");
        AddMessage(RGY_LOG_DEBUG, _T("opened KFM timecode dump file \"%s\".\n"), prm.kfm.timecode.c_str());
    }
    if (prm.kfm.debug && prm.kfm.timecode.length() > 0) {
        const auto resultPath = PathRemoveExtensionS(prm.kfm.timecode) + _T(".result.dat");
        if (_tfopen_s(&m_fpResult, resultPath.c_str(), _T("wb")) != 0 || m_fpResult == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to open KFM result dump file \"%s\".\n"), resultPath.c_str());
            return RGY_ERR_FILE_OPEN;
        }
        AddMessage(RGY_LOG_DEBUG, _T("opened KFM result dump file \"%s\".\n"), resultPath.c_str());

        const auto fmCountPath = PathRemoveExtensionS(prm.kfm.timecode) + _T(".fmcount.dat");
        if (_tfopen_s(&m_fpFMCount, fmCountPath.c_str(), _T("wb")) != 0 || m_fpFMCount == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to open KFM FMCount dump file \"%s\".\n"), fmCountPath.c_str());
            return RGY_ERR_FILE_OPEN;
        }
        AddMessage(RGY_LOG_DEBUG, _T("opened KFM FMCount dump file \"%s\".\n"), fmCountPath.c_str());

        const auto frameInfoPath = PathRemoveExtensionS(prm.kfm.timecode) + _T(".frameinfo.tsv");
        if (_tfopen_s(&m_fpFrameInfo, frameInfoPath.c_str(), _T("w")) != 0 || m_fpFrameInfo == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to open KFM frame info dump file \"%s\".\n"), frameInfoPath.c_str());
            return RGY_ERR_FILE_OPEN;
        }
        fprintf(m_fpFrameInfo, "#stage\tidx\tinputFrameId\ttimestamp\tduration\ttime_ms\tduration_ms\twidth\theight\tcsp\tpicstruct\tflags\tpattern\tis60p\tscore\tcost\treliability\tkfm_n60\tkfm_n24\tkfm_baseType\tkfm_sourceStart\tkfm_numSourceFrames\tkfm_duration60\tkfm_duration120\tkfm_pattern\tkfm_cost\n");
        AddMessage(RGY_LOG_DEBUG, _T("opened KFM frame info dump file \"%s\".\n"), frameInfoPath.c_str());

        const auto containsCombePath = PathRemoveExtensionS(prm.kfm.timecode) + _T(".contains_combe.tsv");
        if (_tfopen_s(&m_fpContainsCombe, containsCombePath.c_str(), _T("w")) != 0 || m_fpContainsCombe == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to open KFM contains-combe dump file \"%s\".\n"), containsCombePath.c_str());
            return RGY_ERR_FILE_OPEN;
        }
        fprintf(m_fpContainsCombe, "#stage\tidx\tn60\tn24\tbaseType\tsourceStart\tnumSourceFrames\tduration60\tduration120\tcontainsCombeCount\tdurationApplied\tpattern\tcost\n");
        AddMessage(RGY_LOG_DEBUG, _T("opened KFM contains-combe dump file \"%s\".\n"), containsCombePath.c_str());

        if (prm.kfm.ucf) {
            const auto ucfNoisePath = PathRemoveExtensionS(prm.kfm.timecode) + _T(".ucfnoise.tsv");
            if (_tfopen_s(&m_fpUcfNoise, ucfNoisePath.c_str(), _T("w")) != 0 || m_fpUcfNoise == nullptr) {
                AddMessage(RGY_LOG_ERROR, _T("failed to open KFM UCF noise dump file \"%s\".\n"), ucfNoisePath.c_str());
                return RGY_ERR_FILE_OPEN;
            }
            fprintf(m_fpUcfNoise, "frame\tplane\tnoise0\tnoise1\tnoiseR0\tnoiseR1\tdiff0\tdiff1\tclass0\tclass1\tfield_diff0\tfield_diff1\tdiff0_calc\tdiff1_calc\n");
            AddMessage(RGY_LOG_DEBUG, _T("opened KFM UCF noise dump file \"%s\".\n"), ucfNoisePath.c_str());
        }
    }
    return RGY_ERR_NONE;
}

void RGYFilterKfm::initStageDumpConfig(const RGYFilterParamKfm& prm) {
    UNREFERENCED_PARAMETER(prm);
    m_stageDumpDir.clear();
    m_stageDumpMaxFrames = 0;
    m_stageDumpFrameCounts.clear();
    m_stageDumpFrameIndices.clear();
    m_stageDumpTargetFrames.clear();

    const char *dumpDir = std::getenv("QSVENC_KFM_DUMP_DIR");
    if (dumpDir == nullptr || dumpDir[0] == '\0') {
        return;
    }
    m_stageDumpDir = dumpDir;
    if (!CreateDirectoryRecursive(m_stageDumpDir.c_str())) {
        AddMessage(RGY_LOG_ERROR, _T("failed to create KFM stage dump directory \"%s\".\n"),
            char_to_tstring(m_stageDumpDir).c_str());
        m_stageDumpDir.clear();
        return;
    }

    const char *maxFrames = std::getenv("QSVENC_KFM_DUMP_MAX_FRAMES");
    if (maxFrames != nullptr && maxFrames[0] != '\0') {
        char *endptr = nullptr;
        const auto value = std::strtol(maxFrames, &endptr, 10);
        if (endptr != maxFrames) {
            m_stageDumpMaxFrames = (int)std::min<long>(std::max<long>(0, value), std::numeric_limits<int>::max());
        }
    }
    const char *frameList = std::getenv("QSVENC_KFM_DUMP_FRAME_LIST");
    if (frameList != nullptr && frameList[0] != '\0') {
        const char *p = frameList;
        while (*p != '\0') {
            char *endptr = nullptr;
            const auto value = std::strtol(p, &endptr, 10);
            if (endptr != p && value >= 0 && value <= std::numeric_limits<int>::max()) {
                m_stageDumpTargetFrames.insert((int)value);
            }
            p = endptr;
            while (*p == ',' || *p == ';' || *p == ':' || *p == ' ' || *p == '\t') {
                p++;
            }
            if (endptr == p && *p != '\0') {
                p++;
            }
        }
    }
    AddMessage(RGY_LOG_DEBUG, _T("enabled KFM stage dump directory \"%s\", max frames %d, target frames %zu.\n"),
        char_to_tstring(m_stageDumpDir).c_str(), m_stageDumpMaxFrames, m_stageDumpTargetFrames.size());
}

bool RGYFilterKfm::stageDumpRequested(int frame24Index) const {
    return !m_stageDumpDir.empty()
        && ((m_stageDumpMaxFrames > 0 && frame24Index < m_stageDumpMaxFrames)
            || m_stageDumpTargetFrames.find(frame24Index) != m_stageDumpTargetFrames.end());
}

RGY_ERR RGYFilterKfm::dumpStageFrame(const char *stage, const RGYFrameInfo *frame, int frame24Index,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    if (!stage || !frame || !frame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    const std::string stageKey(stage);
    auto& frameCount = m_stageDumpFrameCounts[stageKey];
    const int dumpIndex = (frame24Index >= 0) ? frame24Index : frameCount;
    if (!stageDumpRequested(dumpIndex)) {
        return RGY_ERR_NONE;
    }
    auto *dumpedFrameIndices = (frame24Index >= 0) ? &m_stageDumpFrameIndices[stageKey] : nullptr;
    if (dumpedFrameIndices && dumpedFrameIndices->find(frame24Index) != dumpedFrameIndices->end()) {
        return RGY_ERR_NONE;
    }
    const int bitdepth = RGY_CSP_BIT_DEPTH[frame->csp];
    const bool isYuv420 = RGY_CSP_CHROMA_FORMAT[frame->csp] == RGY_CHROMAFMT_YUV420;
    const bool isMono = RGY_CSP_PLANES[frame->csp] == 1 || RGY_CSP_CHROMA_FORMAT[frame->csp] == RGY_CHROMAFMT_MONOCHROME;
    if (bitdepth > 8 || (!isYuv420 && !isMono)) {
        if (frameCount == 0) {
            AddMessage(RGY_LOG_WARN, _T("KFM stage dump skipped for unsupported csp %s at stage %s.\n"),
                RGY_CSP_NAMES[frame->csp], char_to_tstring(stage).c_str());
        }
        return RGY_ERR_NONE;
    }

    const auto planeY = getPlane(frame, RGY_PLANE_Y);
    if (!planeY.ptr[0] || planeY.width <= 0 || planeY.height <= 0) {
        return RGY_ERR_NONE;
    }
    std::vector<uint8_t> hostY((size_t)planeY.width * planeY.height);
    RGYFrameInfo hostPlaneY(planeY.width, planeY.height, RGY_CSP_Y8, 8, frame->picstruct, RGY_MEM_TYPE_CPU);
    hostPlaneY.ptr[0] = hostY.data();
    hostPlaneY.pitch[0] = planeY.width;
    RGYOpenCLEvent readYEvent;
    auto sts = m_cl->copyPlane(&hostPlaneY, &planeY, nullptr, queue, wait_events, &readYEvent);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to read KFM stage dump Y plane (%s): %s.\n"),
            char_to_tstring(stage).c_str(), get_err_mes(sts));
        return sts;
    }
    sts = readYEvent.wait();
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to wait KFM stage dump Y plane (%s): %s.\n"),
            char_to_tstring(stage).c_str(), get_err_mes(sts));
        return sts;
    }

    const int chromaWidth = (hostPlaneY.width + 1) >> 1;
    const int chromaHeight = (hostPlaneY.height + 1) >> 1;
    std::vector<uint8_t> hostU((size_t)chromaWidth * chromaHeight, 128);
    std::vector<uint8_t> hostV((size_t)chromaWidth * chromaHeight, 128);
    if (isYuv420 && !isMono) {
        if (kfmCspHasInterleavedUV(frame->csp)) {
            const auto planeUV = getPlane(frame, RGY_PLANE_U);
            if (planeUV.ptr[0]) {
                std::vector<uint8_t> hostUV((size_t)planeUV.width * planeUV.height);
                RGYFrameInfo hostPlaneUV(planeUV.width, planeUV.height, RGY_CSP_Y8, 8, frame->picstruct, RGY_MEM_TYPE_CPU);
                hostPlaneUV.ptr[0] = hostUV.data();
                hostPlaneUV.pitch[0] = planeUV.width;
                RGYOpenCLEvent readUVEvent;
                sts = m_cl->copyPlane(&hostPlaneUV, &planeUV, nullptr, queue, {}, &readUVEvent);
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to read KFM stage dump UV plane (%s): %s.\n"),
                        char_to_tstring(stage).c_str(), get_err_mes(sts));
                    return sts;
                }
                sts = readUVEvent.wait();
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to wait KFM stage dump UV plane (%s): %s.\n"),
                        char_to_tstring(stage).c_str(), get_err_mes(sts));
                    return sts;
                }
                for (int y = 0; y < chromaHeight; y++) {
                    const auto *src = hostUV.data() + (size_t)y * hostPlaneUV.pitch[0];
                    auto *dstU = hostU.data() + (size_t)y * chromaWidth;
                    auto *dstV = hostV.data() + (size_t)y * chromaWidth;
                    for (int x = 0; x < chromaWidth; x++) {
                        dstU[x] = src[x * 2 + 0];
                        dstV[x] = src[x * 2 + 1];
                    }
                }
            }
        } else {
            const auto planeU = getPlane(frame, RGY_PLANE_U);
            const auto planeV = getPlane(frame, RGY_PLANE_V);
            if (planeU.ptr[0] && planeV.ptr[0] && planeU.width == chromaWidth && planeU.height == chromaHeight && planeV.width == chromaWidth && planeV.height == chromaHeight) {
                RGYFrameInfo hostPlaneU(chromaWidth, chromaHeight, RGY_CSP_Y8, 8, frame->picstruct, RGY_MEM_TYPE_CPU);
                RGYFrameInfo hostPlaneV(chromaWidth, chromaHeight, RGY_CSP_Y8, 8, frame->picstruct, RGY_MEM_TYPE_CPU);
                hostPlaneU.ptr[0] = hostU.data();
                hostPlaneV.ptr[0] = hostV.data();
                hostPlaneU.pitch[0] = chromaWidth;
                hostPlaneV.pitch[0] = chromaWidth;
                RGYOpenCLEvent readUEvent;
                sts = m_cl->copyPlane(&hostPlaneU, &planeU, nullptr, queue, {}, &readUEvent);
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to read KFM stage dump U plane (%s): %s.\n"),
                        char_to_tstring(stage).c_str(), get_err_mes(sts));
                    return sts;
                }
                sts = readUEvent.wait();
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to wait KFM stage dump U plane (%s): %s.\n"),
                        char_to_tstring(stage).c_str(), get_err_mes(sts));
                    return sts;
                }
                RGYOpenCLEvent readVEvent;
                sts = m_cl->copyPlane(&hostPlaneV, &planeV, nullptr, queue, {}, &readVEvent);
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to read KFM stage dump V plane (%s): %s.\n"),
                        char_to_tstring(stage).c_str(), get_err_mes(sts));
                    return sts;
                }
                sts = readVEvent.wait();
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to wait KFM stage dump V plane (%s): %s.\n"),
                        char_to_tstring(stage).c_str(), get_err_mes(sts));
                    return sts;
                }
            }
        }
    }

    const auto path = PathCombineS(m_stageDumpDir, kfmStageDumpName(stage));
    std::ofstream dump(path, std::ios::out | std::ios::binary | (frameCount == 0 ? std::ios::trunc : std::ios::app));
    if (!dump) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open KFM stage dump \"%s\".\n"), char_to_tstring(path).c_str());
        return RGY_ERR_FILE_OPEN;
    }
    if (frameCount == 0) {
        dump << "YUV4MPEG2 W" << hostPlaneY.width << " H" << hostPlaneY.height << " F30000:1001 Ip A0:0 "
             << (isMono ? "Cmono" : "C420jpeg") << "\n";
    }
    dump << "FRAME\n";
    for (int y = 0; y < hostPlaneY.height; y++) {
        dump.write(reinterpret_cast<const char *>(hostPlaneY.ptr[0] + (size_t)y * hostPlaneY.pitch[0]), hostPlaneY.width);
    }
    if (!isMono) {
        dump.write(reinterpret_cast<const char *>(hostU.data()), hostU.size());
        dump.write(reinterpret_cast<const char *>(hostV.data()), hostV.size());
    }
    if (!dump) {
        AddMessage(RGY_LOG_ERROR, _T("failed to write KFM stage dump \"%s\".\n"), char_to_tstring(path).c_str());
        return RGY_ERR_FILE_OPEN;
    }
    frameCount++;
    if (dumpedFrameIndices) {
        dumpedFrameIndices->insert(frame24Index);
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::initRtgmc(const std::shared_ptr<RGYFilterParamKfm>& prm, std::unique_ptr<RGYFilterRtgmc>& rtgmc, bool updateOutputParam, int useFlag, bool sharedAnalysisMode) {
    auto rtgmcParam = std::make_shared<RGYFilterParamRtgmc>();
    rtgmcParam->rtgmc.enable = true;
    rtgmcParam->rtgmc.preset = prm->kfm.preset;
    apply_vpp_rtgmc_preset(rtgmcParam->rtgmc, prm->kfm.preset, rtgmcParam->rtgmc.tuning);
    if (useFlag > 0) {
        rtgmcParam->rtgmc.tr1.useFlag = useFlag;
        rtgmcParam->rtgmc.tr2.useFlag = useFlag;
        rtgmcParam->rtgmc.analyze.useFlag = useFlag;
    }
    rtgmcParam->frameIn = prm->frameIn;
    rtgmcParam->frameOut = prm->frameOut;
    rtgmcParam->baseFps = prm->baseFps;
    rtgmcParam->timebase = prm->timebase;
    rtgmcParam->bOutOverwrite = false;
    rtgmcParam->sharedAnalysisMode = sharedAnalysisMode;

    rtgmc = std::make_unique<RGYFilterRtgmc>(m_cl);
    auto sts = rtgmc->init(rtgmcParam, m_pLog);
    if (sts != RGY_ERR_NONE) {
        rtgmc.reset();
        return sts;
    }

    if (updateOutputParam) {
        prm->frameOut = rtgmcParam->frameOut;
        prm->baseFps = rtgmcParam->baseFps;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::initNrFilter(const std::shared_ptr<RGYFilterParamKfm>& prm) {
    m_nrFilter.reset();
    if (!prm->kfm.nr) {
        return RGY_ERR_NONE;
    }

    auto nrParam = std::make_shared<RGYFilterParamDegrain>();
    nrParam->frameIn = prm->frameOut;
    nrParam->frameOut = prm->frameOut;
    nrParam->baseFps = prm->baseFps;
    nrParam->bOutOverwrite = false;
    nrParam->degrain.enable = true;
    nrParam->degrain.preset = VppDegrainPreset::Auto;
    nrParam->degrain.mode = VppDegrainMode::Degrain;
    nrParam->degrain.stage = VppDegrainStage::TR1;
    nrParam->degrain.delta = 1;
    nrParam->degrain.search = 4;
    nrParam->degrain.thsad = 300;
    nrParam->degrain.thsadc = 150;
    nrParam->degrain.thscd1 = 1600;
    nrParam->degrain.thscd2 = 130;
    nrParam->degrain.pel = 1;
    nrParam->degrain.blksize = 16;
    nrParam->degrain.overlap = nrParam->degrain.blksize / 2;
    nrParam->degrain.levels = 2;
    nrParam->degrain.chroma = true;
    nrParam->degrain.binomial = 1;
    nrParam->degrain.tvRange = true;

    auto nrFilter = std::make_unique<RGYFilterDegrain>(m_cl);
    auto sts = nrFilter->init(nrParam, m_pLog);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialize KFM NR Auto degrain(TR1) filter: %s.\n"), get_err_mes(sts));
        return sts;
    }
    // This is a Phase4 bridge: the reference applies NR to deint24/deint30 branches
    // before UCF/switch, while this nested filter denoises the final KFM stream.
    AddMessage(RGY_LOG_INFO, _T("--vpp-kfm nr=true is wired to degrain preset=auto,tr=1,blksize=16,levels=2,binomial=true on the final KFM output stream.\n"));
    m_nrFilter = std::move(nrFilter);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamKfm>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (prm->frameOut.width <= 0 || prm->frameOut.height <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (prm->kfm.ucf) {
        AddMessage(RGY_LOG_INFO, _T("--vpp-kfm ucf=true enables the UCF debug field/crop noise pre-stage.\n"));
    }

    m_pathThrough = FILTER_PATHTHROUGH_NONE;
    if (prm->kfm.mode == VppKfmMode::P60) {
        auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamKfm>(m_param);
        if (!prmPrev
            || cmpFrameInfoCspResolution(&prmPrev->frameOut, &prm->frameOut)
            || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[prm->frameOut.csp]
            || prmPrev->kfm != prm->kfm) {
            auto sts = loadPrograms(*prm);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        auto sts = initRtgmc(prm, m_rtgmc, true);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_before60Rtgmc.reset();
        m_after60Rtgmc.reset();
        m_before60Lane.init(this, nullptr, "before60", _T("before60"), false);
        m_after60Lane.init(this, nullptr, "after60", _T("after60"), false);
        if (prm->kfm.ucf) {
            m_rtgmc->enableIntermediateCapture(true);
            sts = initRtgmc(prm, m_before60Rtgmc, false, 1, true);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            sts = initRtgmc(prm, m_after60Rtgmc, false, 2, true);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            auto sharedData = m_rtgmc->getSharedAnalysisData();
            m_before60Rtgmc->setSharedAnalysisData(sharedData);
            m_after60Rtgmc->setSharedAnalysisData(sharedData);
            m_before60Lane.init(this, m_before60Rtgmc.get(), "before60", _T("before60"), false);
            m_after60Lane.init(this, m_after60Rtgmc.get(), "after60", _T("after60"), false);
        }
        sts = initAnalyzer(*prm);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        initStageDumpConfig(*prm);
        sts = AllocFrameBuf(prm->frameOut, 8);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM output buffer: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
        sts = allocWorkFrameBuf(prm->frameOut, 8);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM work buffer: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
        for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
            prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
        }
        sts = initNrFilter(prm);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_staticFlag = m_cl->createFrameBuffer(prm->frameOut);
        if (!m_staticFlag) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM static flag frame.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        for (auto& source : m_sourceCache) {
            retireKfmSourceSlot(std::move(source.slot), m_cl->queue());
        }
        m_sourceCache.clear();
        collectRetiredKfmSourceSlots();
        m_outputBufferIndex = 0;
        setFilterInfo(prm->print());
        m_param = prm;
        return RGY_ERR_NONE;
    }
    m_rtgmc.reset();
    m_deint60Rtgmc.reset();
    m_before60Rtgmc.reset();
    m_after60Rtgmc.reset();
    m_deint60Lane.init(this, nullptr, "deint60", _T("deint60 cache"), true);
    m_before60Lane.init(this, nullptr, "before60", _T("before60"), false);
    m_after60Lane.init(this, nullptr, "after60", _T("after60"), false);
    if (prm->kfm.mode == VppKfmMode::P24) {
        AddMessage(RGY_LOG_INFO, _T("--vpp-kfm mode=24 uses the Phase3 24p render bring-up path.\n"));
    } else if (prm->kfm.mode == VppKfmMode::VFR) {
        AddMessage(RGY_LOG_DEBUG, _T("--vpp-kfm mode=%s uses the Phase3 VFR scheduler with 24p/30p render bring-up.\n"),
            get_cx_desc(list_vpp_kfm_mode, (int)prm->kfm.mode));
    } else {
        AddMessage(RGY_LOG_WARN, _T("--vpp-kfm mode=%s is using the Phase3 source-timeline fallback; IVTC render decisions are analyzed but image selection is not fully wired yet.\n"),
            get_cx_desc(list_vpp_kfm_mode, (int)prm->kfm.mode));
    }

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamKfm>(m_param);
    if (!prmPrev
        || cmpFrameInfoCspResolution(&prmPrev->frameOut, &prm->frameOut)
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[prm->frameOut.csp]
        || prmPrev->kfm != prm->kfm) {
        auto sts = loadPrograms(*prm);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }

    const bool needTelecineWorkFrames = prm->kfm.mode == VppKfmMode::P24
        || prm->kfm.mode == VppKfmMode::VFR;
    const int frameBufCount = needTelecineWorkFrames ? (prm->kfm.ucf ? 16 : 8) : 1;
    auto sts = AllocFrameBuf(prm->frameOut, frameBufCount);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    if (needTelecineWorkFrames) {
        sts = allocWorkFrameBuf(prm->frameOut, frameBufCount);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM work buffer: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    } else {
        m_workFrameBuf.clear();
        m_workBufferIndex = 0;
    }
    if (prm->kfm.mode == VppKfmMode::VFR
        || (prm->kfm.mode == VppKfmMode::P24 && kfmDeint60BranchEnabled())) {
        sts = initRtgmc(prm, m_deint60Rtgmc, false);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_staticFlag = m_cl->createFrameBuffer(prm->frameOut);
        if (!m_staticFlag) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM static flag frame.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_deint60Lane.init(this, m_deint60Rtgmc.get(), "deint60", _T("deint60 cache"), true);
    }
    if (prm->kfm.ucf) {
        RGYFilterRtgmc::RtgmcSharedAnalysisData sharedData;
        bool useSharedAnalysisMode = false;
        if (m_deint60Rtgmc) {
            sharedData = m_deint60Rtgmc->getSharedAnalysisData();
            useSharedAnalysisMode = sharedData.analyzeFilter != nullptr;
            if (useSharedAnalysisMode) {
                m_deint60Rtgmc->enableIntermediateCapture(true);
            }
        }
        sts = initRtgmc(prm, m_before60Rtgmc, false, 1, useSharedAnalysisMode);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = initRtgmc(prm, m_after60Rtgmc, false, 2, useSharedAnalysisMode);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (useSharedAnalysisMode) {
            m_before60Rtgmc->setSharedAnalysisData(sharedData);
            m_after60Rtgmc->setSharedAnalysisData(sharedData);
        }
        m_before60Lane.init(this, m_before60Rtgmc.get(), "before60", _T("before60"), false);
        m_after60Lane.init(this, m_after60Rtgmc.get(), "after60", _T("after60"), false);
        if (!m_staticFlag) {
            m_staticFlag = m_cl->createFrameBuffer(prm->frameOut);
            if (!m_staticFlag) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM static flag frame.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
    }
    sts = initAnalyzer(*prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    initStageDumpConfig(*prm);
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    sts = initNrFilter(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (prm->kfm.mode == VppKfmMode::VFR) {
        prm->baseFps *= 2;
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

int RGYFilterKfm::requiredOutputFrames() const {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamKfm>(m_param);
    if (!prm) {
        return 0;
    }
    switch (prm->kfm.mode) {
    case VppKfmMode::VFR:
    case VppKfmMode::P24:
        return 16;
    default:
        return 0;
    }
}

RGY_ERR RGYFilterKfm::padSourceFrame(RGYFrameInfo *pPaddedFrame, const RGYFrameInfo *pSourceFrame,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event, bool sourceInPaddedFrame) {
    if (!pPaddedFrame || !pSourceFrame || !m_programs[KFM_PROG_PAD].get()) {
        return RGY_ERR_INVALID_CALL;
    }
    if (pPaddedFrame->csp != pSourceFrame->csp || pPaddedFrame->width != pSourceFrame->width) {
        return RGY_ERR_INVALID_PARAM;
    }

    const auto planes = RGY_CSP_PLANES[pPaddedFrame->csp];
    RGYOpenCLEvent prevEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dst = getPlane(pPaddedFrame, (RGY_PLANE)iplane);
        const auto src = getPlane(pSourceFrame, (RGY_PLANE)iplane);
        const int vpad = (dst.height - src.height) >> 1;
        if (dst.width != src.width || dst.height != src.height + vpad * 2 || vpad <= 0) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM padded source plane size (plane %d, src %dx%d, dst %dx%d).\n"),
                iplane, src.width, src.height, dst.width, dst.height);
            return RGY_ERR_INVALID_PARAM;
        }

        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
        RGYOpenCLEvent planeEvent;
        RGYWorkSize local(32, 8);
        const char *kernelName = sourceInPaddedFrame ? "kernel_kfm_padv_inplace" : "kernel_kfm_pad";
        const auto global = sourceInPaddedFrame ? RGYWorkSize(dst.width, vpad) : RGYWorkSize(dst.width, dst.height);
        auto kernel = m_programs[KFM_PROG_PAD].get()->kernel(kernelName).config(queue, local, global, waitHere, &planeEvent);
        auto err = sourceInPaddedFrame
            ? kernel.launch((cl_mem)dst.ptr[0], dst.pitch[0], dst.width, src.height, vpad)
            : kernel.launch((cl_mem)dst.ptr[0], dst.pitch[0], (cl_mem)src.ptr[0], src.pitch[0], dst.width, src.height, vpad);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %S (plane %d): %s.\n"), kernelName, iplane, get_err_mes(err));
            return err;
        }
        prevEvent = planeEvent;
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    copyFramePropWithoutRes(pPaddedFrame, pSourceFrame);
    pPaddedFrame->picstruct = RGY_PICSTRUCT_FRAME;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::cacheSourceFrame(const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    if (!frame || !frame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    KfmCachedSource entry;
    entry.sourceIndex = m_cachedSourceFrames++;
    entry.inputFrameId = frame->inputFrameId;
    entry.timestamp = frame->timestamp;
    entry.slot = acquireKfmSourceSlot(*frame, CL_MEM_READ_WRITE);
    if (!entry.slot || !entry.slot->sourceFrame || !entry.slot->paddedFrame) {
        return RGY_ERR_MEMORY_ALLOC;
    }
    entry.frame = entry.slot->sourceFrame;
    entry.paddedFrame = entry.slot->paddedFrame;
    auto sts = m_cl->copyFrame(&entry.frame->frame, frame, nullptr, queue, wait_events, &entry.event, RGYFrameCopyMode::FRAME, "kfm.source_cache");
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to cache KFM source frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
    copyFramePropWithoutRes(&entry.frame->frame, frame);
    m_sourceCache.push_back(std::move(entry));
    auto& cachedEntry = m_sourceCache.back();

    sts = analyzeAvailableSource(false, queue);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    auto padWaitEvents = wait_events;
    if (cachedEntry.event() != nullptr) {
        padWaitEvents.push_back(cachedEntry.event);
    }
    sts = padSourceFrame(&cachedEntry.paddedFrame->frame, &cachedEntry.frame->frame, queue, padWaitEvents, &cachedEntry.paddedEvent, true);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to pad KFM source frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
    writeFrameInfoDump("source-pad", &cachedEntry.paddedFrame->frame);

    trimSourceCache(queue);
    writeFrameInfoDump("source", frame);
    return RGY_ERR_NONE;
}

size_t RGYFilterKfm::sourceCacheLimit() const {
    const auto prm = std::dynamic_pointer_cast<RGYFilterParamKfm>(m_param);
    if (!prm) {
        return 16;
    }
    if (prm->kfm.timing == VppKfmTiming::Strict) {
        return std::numeric_limits<size_t>::max();
    }
    if (prm->kfm.timing == VppKfmTiming::RealtimePlus) {
        return std::max<size_t>(16, static_cast<size_t>(std::max(0, prm->kfm.pastCycles)) * 5 + KFM_REALTIMEPLUS_SOURCE_CACHE_MARGIN);
    }
    return 16;
}

size_t RGYFilterKfm::deint60CacheLimit() const {
    const auto prm = std::dynamic_pointer_cast<RGYFilterParamKfm>(m_param);
    if (!prm || !m_deint60Rtgmc) {
        return 32;
    }
    if (lazyDeint60Enabled(*prm)) {
        return 32;
    }
    if (prm->kfm.timing == VppKfmTiming::Strict) {
        return std::numeric_limits<size_t>::max();
    }
    if (prm->kfm.timing == VppKfmTiming::RealtimePlus) {
        return std::max<size_t>(32, static_cast<size_t>(std::max(0, prm->kfm.pastCycles)) * 10 + KFM_REALTIMEPLUS_DEINT60_CACHE_MARGIN);
    }
    return 32;
}

int RGYFilterKfm::sourceCacheTrimFloor() const {
    const auto prm = std::dynamic_pointer_cast<RGYFilterParamKfm>(m_param);
    if (!prm || prm->kfm.mode != VppKfmMode::VFR || m_nextSwitchN60 <= 0) {
        return 0;
    }
    int lazyLookbehind = 0;
    if (lazyDeint60Enabled(*prm) && m_deint60Rtgmc) {
        lazyLookbehind = m_deint60Rtgmc->requiredPrimingSourceFrames();
        if (prm->kfm.ucf && (m_before60Rtgmc || m_after60Rtgmc)) {
            lazyLookbehind += std::max(m_before60Lane.requiredPrimingSourceFrames(), m_after60Lane.requiredPrimingSourceFrames())
                + KFM_UCF_SHARED_ANALYSIS_SOURCE_DELAY + 4;
        }
    }
    auto trimFloor = std::max(0, (m_nextSwitchN60 >> 1) - KFM_VFR_SOURCE_TRIM_LOOKBEHIND - lazyLookbehind);
    for (const auto& pending : m_pendingUcfNoiseResults) {
        if (pending.sourceIndex >= 0) {
            trimFloor = std::min(trimFloor, pending.sourceIndex);
        }
    }
    if (m_ucfNoiseCache.size() >= 3) {
        for (const auto& noise : m_ucfNoiseCache) {
            if (noise.fieldIndex >= 0) {
                trimFloor = std::min(trimFloor, noise.fieldIndex >> 1);
            }
        }
    }
    return trimFloor;
}

int RGYFilterKfm::deint60CacheTrimFloor() const {
    const auto prm = std::dynamic_pointer_cast<RGYFilterParamKfm>(m_param);
    if (!prm || prm->kfm.mode != VppKfmMode::VFR || m_nextSwitchN60 <= 0) {
        return 0;
    }
    return std::max(0, m_nextSwitchN60 - KFM_VFR_DEINT60_TRIM_LOOKBEHIND);
}

bool RGYFilterKfm::lazyDeint60Enabled(const RGYFilterParamKfm& prm) const {
    return prm.kfm.mode == VppKfmMode::VFR && !prm.kfm.ucf && !kfmForceEagerRtgmc();
}

RGY_ERR RGYFilterKfm::runDeint60Branch(const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, int *cachedFrames) {
    return m_deint60Lane.feed(frame, queue, wait_events, cachedFrames);
}

RGY_ERR RGYFilterKfm::drainDeint60Branch(RGYOpenCLQueue &queue, int *cachedFrames) {
    const auto maxDrainIterations = std::max(256, m_cachedSourceFrames * 4 + 256);
    return m_deint60Lane.drain(queue, maxDrainIterations, cachedFrames);
}

RGY_ERR RGYFilterKfm::runUcfRtgmcBranches(const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    if (!m_before60Rtgmc && !m_after60Rtgmc) {
        return runUcfNoiseAnalysisFromSource(frame, queue, wait_events);
    }
    if (!frame || !frame->ptr[0]) {
        auto sts = drainUcfRtgmcBranch(m_before60Lane, queue);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = drainUcfRtgmcBranch(m_after60Lane, queue);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        return runUcfNoiseAnalysisFromSource(frame, queue, wait_events);
    }

    auto sts = runUcfRtgmcBranch(m_before60Lane, frame, queue, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = runUcfRtgmcBranch(m_after60Lane, frame, queue, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return runUcfNoiseAnalysisFromSource(frame, queue, wait_events);
}

RGY_ERR RGYFilterKfm::runUcfNoiseAnalysisFromSource(const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto prm = std::dynamic_pointer_cast<RGYFilterParamKfm>(m_param);
    if (!prm || !frame || !frame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    auto sts = RGY_ERR_NONE;
    const int sourceIndex = m_sourceCache.empty() ? m_cachedSourceFrames - 1 : m_sourceCache.back().sourceIndex;
    for (int parity = 0; parity < 2; parity++) {
        const int fieldIndex = sourceIndex * 2 + parity;
        const bool useFusedUcfPreprocess = kfmUseFusedUcfPreprocess() && !stageDumpRequested(fieldIndex);
        RGYFrameInfo *gaussFrame = nullptr;
        RGYOpenCLEvent gaussEvent;
        RGYFrameInfo *fieldFrame = nullptr;
        if (useFusedUcfPreprocess && !kfmUcfNoGaussForTest()) {
            sts = prepareUcfNoiseGaussFrameFromSource(&gaussFrame, sourceIndex, parity, frame, queue, wait_events, &gaussEvent);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        } else {
            RGYOpenCLEvent fieldEvent;
            sts = prepareUcfNoiseFieldCropFrame(&fieldFrame, sourceIndex, parity, frame, queue, wait_events, &fieldEvent);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            std::vector<RGYOpenCLEvent> gaussWaitEvents;
            if (fieldEvent() != nullptr) {
                gaussWaitEvents.push_back(fieldEvent);
            }
            if (kfmUcfNoGaussForTest()) {
                gaussFrame = fieldFrame;
                gaussEvent = fieldEvent;
                writeFrameInfoDump("ucf-noise-gauss", gaussFrame);
                sts = dumpStageFrame("ucf-noise-gauss", gaussFrame, fieldIndex, queue, gaussWaitEvents);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            } else {
                sts = prepareUcfNoiseGaussFrame(&gaussFrame, parity, fieldFrame, queue, gaussWaitEvents, &gaussEvent);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
        }
        std::vector<RGYOpenCLEvent> deintWaitEvents;
        if (gaussEvent() != nullptr) {
            deintWaitEvents.push_back(gaussEvent);
        }
        sts = (useFusedUcfPreprocess && fieldFrame == nullptr)
            ? runUcfNoiseLimitStageFromSource(*prm, frame, gaussFrame, fieldIndex, parity, queue, deintWaitEvents)
            : runUcfNoiseLimitStage(*prm, fieldFrame, gaussFrame, fieldIndex, queue, deintWaitEvents);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::ensureUcfRtgmcRange(KfmUcfLaneType laneType, int n60begin, int n60end, RGYOpenCLQueue &queue) {
    if (laneType == KFM_UCF_LANE_NONE || n60begin >= n60end) {
        return RGY_ERR_NONE;
    }
    // eager (FORCE_EAGER or non-lazy) では intake で全フレーム feed 済みのため ensure 不要。
    // ここで ensureRange を呼ぶと進行中の eager パイプラインを resetTemporalState で
    // 破壊して再 priming が発生する (長尺の eager 実行で性能劣化と出力差の原因になる)。
    const auto prm = std::dynamic_pointer_cast<RGYFilterParamKfm>(m_param);
    if (!prm || !lazyDeint60Enabled(*prm)) {
        return RGY_ERR_NONE;
    }
    auto *lane = (laneType == KFM_UCF_LANE_BEFORE) ? &m_before60Lane : &m_after60Lane;
    if (!m_deint60Rtgmc || (laneType == KFM_UCF_LANE_BEFORE && !m_before60Rtgmc) || (laneType == KFM_UCF_LANE_AFTER && !m_after60Rtgmc)) {
        return RGY_ERR_NONE;
    }
    const int sourceBegin = std::max(0, n60begin) >> 1;
    const int primeStart = std::max(0, sourceBegin - lane->requiredPrimingSourceFrames());
    const int sourceEnd = std::max(primeStart, divCeil(std::max(0, n60end), 2) + KFM_UCF_SHARED_ANALYSIS_SOURCE_DELAY);
    if (!hasDeint60Intermediates(primeStart, sourceEnd)) {
        auto& deint60Cache = m_deint60Lane.cache();
        for (auto it = deint60Cache.begin(); it != deint60Cache.end();) {
            if (primeStart * 2 <= it->n60 && it->n60 < sourceEnd * 2) {
                it = deint60Cache.erase(it);
            } else {
                ++it;
            }
        }
        auto sts = m_deint60Lane.ensureRange(primeStart * 2, sourceEnd * 2, queue);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return lane->ensureRange(n60begin, n60end, queue);
}

void RGYFilterKfm::captureDeint60Intermediates(int sourceIndex) {
    if (!m_deint60Rtgmc || sourceIndex < 0) {
        return;
    }
    const auto& captured = m_deint60Rtgmc->getCapturedIntermediates();
    if (!captured.empty()) {
        for (auto& group : m_deint60IntermediateQueue) {
            if (group.sourceIndex == sourceIndex) {
                group.intermediates = captured;
                m_deint60Rtgmc->clearCapturedIntermediates();
                return;
            }
        }
        KfmMainIntermediateGroup group;
        group.sourceIndex = sourceIndex;
        group.intermediates = captured;
        m_deint60IntermediateQueue.push_back(std::move(group));
        trimDeint60Intermediates();
    }
    m_deint60Rtgmc->clearCapturedIntermediates();
}

bool RGYFilterKfm::hasDeint60Intermediates(int sourceBegin, int sourceEnd) const {
    for (int sourceIndex = sourceBegin; sourceIndex < sourceEnd; sourceIndex++) {
        const auto it = std::find_if(m_deint60IntermediateQueue.begin(), m_deint60IntermediateQueue.end(), [sourceIndex](const KfmMainIntermediateGroup& group) {
            return group.sourceIndex == sourceIndex && !group.intermediates.empty();
        });
        if (it == m_deint60IntermediateQueue.end()) {
            return false;
        }
    }
    return true;
}

void RGYFilterKfm::pushDeint60Intermediates(RGYFilterRtgmc *rtgmc, int sourceIndex) {
    if (!rtgmc || sourceIndex < 0) {
        return;
    }
    for (const auto& group : m_deint60IntermediateQueue) {
        if (group.sourceIndex == sourceIndex) {
            for (const auto& captured : group.intermediates) {
                rtgmc->pushIntermediateInput(captured);
            }
            return;
        }
    }
}

void RGYFilterKfm::trimDeint60Intermediates() {
    const int keepSourceFrames = std::max<int>(32, (int)sourceCacheLimit());
    while (!m_deint60IntermediateQueue.empty() && m_deint60IntermediateQueue.front().sourceIndex + keepSourceFrames < m_cachedSourceFrames) {
        m_deint60IntermediateQueue.pop_front();
    }
}

RGY_ERR RGYFilterKfm::runUcfRtgmcBranch(KfmRtgmcLane& lane, const RGYFrameInfo *frame,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    return lane.feed(frame, queue, wait_events);
}

RGY_ERR RGYFilterKfm::drainUcfRtgmcBranch(KfmRtgmcLane& lane, RGYOpenCLQueue &queue) {
    const auto maxDrainIterations = std::max(256, m_cachedSourceFrames * 4 + 256);
    return lane.drain(queue, maxDrainIterations);
}

const RGYFrameInfo *RGYFilterKfm::findDeint60Frame(int n60, std::vector<RGYOpenCLEvent> *wait_events) const {
    const auto *entry = findCachedDeint60Frame(m_deint60Lane, n60, wait_events);
    return entry && entry->frame ? &entry->frame->frame : nullptr;
}

const RGYFilterKfm::KfmCachedDeint60 *RGYFilterKfm::findCachedDeint60Frame(const KfmRtgmcLane& lane, int n60, std::vector<RGYOpenCLEvent> *wait_events) const {
    return lane.find(n60, wait_events);
}

const RGYFilterKfm::KfmUcfNoiseDumpRecord *RGYFilterKfm::findUcfNoiseResult(int sourceIndex) const {
    for (auto it = m_ucfNoiseResultCache.rbegin(); it != m_ucfNoiseResultCache.rend(); ++it) {
        if (it->sourceIndex == sourceIndex) {
            return &(*it);
        }
    }
    return nullptr;
}

RGY_ERR RGYFilterKfm::copyUcfFrame(const RGYFilterParamKfm& prm, RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!pOutputFrame || !pInputFrame || !m_programs[KFM_PROG_UCF].get()) {
        return RGY_ERR_INVALID_CALL;
    }

    const auto kernelName = kfmUcfKernelName(prm.kfm.mode);
    const auto planes = RGY_CSP_PLANES[pOutputFrame->csp];
    RGYOpenCLEvent prevEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dst = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto src = getPlane(pInputFrame, (RGY_PLANE)iplane);
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
        RGYOpenCLEvent planeEvent;
        RGYWorkSize local(32, 8);
        RGYWorkSize global(dst.width, dst.height);
        auto err = m_programs[KFM_PROG_UCF].get()->kernel(kernelName).config(queue, local, global, waitHere, &planeEvent).launch(
            (cl_mem)dst.ptr[0], dst.pitch[0],
            (cl_mem)src.ptr[0], src.pitch[0],
            dst.width, dst.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %S (plane %d): %s.\n"), kernelName, iplane, get_err_mes(err));
            return err;
        }
        prevEvent = planeEvent;
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    copyFramePropWithoutRes(pOutputFrame, pInputFrame);
    pOutputFrame->dataList = pInputFrame->dataList;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::createUcfGaussProgram(KfmUcfGaussProgram& program, int sourceSize, double cropSize, int targetSize, double p) {
    const double cropStart = 0.0;
    const double filterScale = (double)targetSize / cropSize;
    const double filterStep = std::min(filterScale, 1.0);
    const double filterSupport = 4.0 / filterStep;
    const int filterSize = (int)std::ceil(filterSupport * 2.0);
    if (sourceSize <= filterSupport || targetSize <= 0 || filterSize <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM UCF gaussresize size source=%d target=%d filter=%d.\n"),
            sourceSize, targetSize, filterSize);
        return RGY_ERR_INVALID_PARAM;
    }

    std::vector<int> offset(targetSize);
    std::vector<float> coeff((size_t)targetSize * filterSize);
    double pos = (filterSize == 1) ? cropStart : cropStart + ((cropSize - targetSize) / (targetSize * 2.0));
    const double posStep = cropSize / targetSize;
    for (int i = 0; i < targetSize; i++) {
        int endPos = (int)(pos + filterSupport);
        if (endPos > sourceSize - 1) {
            endPos = sourceSize - 1;
        }
        int startPos = endPos - filterSize + 1;
        if (startPos < 0) {
            startPos = 0;
        }
        offset[i] = startPos;
        const double okPos = std::min(std::max(pos, 0.0), (double)(sourceSize - 1));
        double total = 0.0;
        for (int j = 0; j < filterSize; j++) {
            total += kfmUcfGaussValue((startPos + j - okPos) * filterStep, p);
        }
        if (total == 0.0) {
            total = 1.0;
        }
        double value = 0.0;
        for (int k = 0; k < filterSize; k++) {
            const double newValue = value + kfmUcfGaussValue((startPos + k - okPos) * filterStep, p) / total;
            coeff[(size_t)i * filterSize + k] = (float)(newValue - value);
            value = newValue;
        }
        pos += posStep;
    }
    program.sourceSize = sourceSize;
    program.targetSize = targetSize;
    program.filterSize = filterSize;
    program.offset = m_cl->copyDataToBuffer(offset.data(), offset.size() * sizeof(offset[0]), CL_MEM_READ_ONLY, m_cl->queue().get());
    program.coeff = m_cl->copyDataToBuffer(coeff.data(), coeff.size() * sizeof(coeff[0]), CL_MEM_READ_ONLY, m_cl->queue().get());
    if (!program.offset || !program.offset->mem() || !program.coeff || !program.coeff->mem()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to upload KFM UCF gaussresize coefficients.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::prepareUcfNoiseFieldCropFrame(RGYFrameInfo **ppFieldFrame, int sourceIndex, int parity, const RGYFrameInfo *pInputFrame,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!ppFieldFrame || !pInputFrame || !pInputFrame->ptr[0] || !m_programs[KFM_PROG_UCF].get()) {
        return RGY_ERR_INVALID_CALL;
    }
    *ppFieldFrame = nullptr;
    const int cropX = 4;
    const int cropY = 4;
    RGYFrameInfo fieldInfo = *pInputFrame;
    fieldInfo.width = pInputFrame->width - cropX * 2;
    fieldInfo.height = (pInputFrame->height >> 1) - cropY * 2;
    if (fieldInfo.width <= 0 || fieldInfo.height <= 0 || (fieldInfo.width & 1) || (fieldInfo.height & 1)) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM UCF field/crop size from %dx%d.\n"), pInputFrame->width, pInputFrame->height);
        return RGY_ERR_INVALID_PARAM;
    }
    fieldInfo.picstruct = RGY_PICSTRUCT_FRAME;
    for (int i = 0; i < RGY_MAX_PLANES; i++) {
        fieldInfo.ptr[i] = nullptr;
        fieldInfo.pitch[i] = 0;
    }
    const int frameIndex = parity & 1;
    auto& fieldBuffer = m_ucfNoiseFieldFrames[frameIndex];
    if (!fieldBuffer
        || fieldBuffer->frame.width != fieldInfo.width
        || fieldBuffer->frame.height != fieldInfo.height
        || fieldBuffer->frame.csp != fieldInfo.csp) {
        fieldBuffer = m_cl->createFrameBuffer(fieldInfo);
        if (!fieldBuffer) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM UCF field/crop frame.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    auto *pFieldFrame = &fieldBuffer->frame;
    const int fieldParity = parity & 1;
    const bool interleavedUV = kfmCspHasInterleavedUV(pInputFrame->csp);
    const auto chromaFmt = RGY_CSP_CHROMA_FORMAT[pInputFrame->csp];
    const auto planes = RGY_CSP_PLANES[pFieldFrame->csp];
    RGYOpenCLEvent prevEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto planeType = (RGY_PLANE)iplane;
        const auto dst = getPlane(pFieldFrame, planeType);
        const auto src = getPlane(pInputFrame, planeType);
        if (!dst.ptr[0] || !src.ptr[0]) {
            continue;
        }
        const bool chromaPlane = planeType != RGY_PLANE_Y && chromaFmt != RGY_CHROMAFMT_MONOCHROME;
        const int xShift = (chromaPlane && !interleavedUV
            && (chromaFmt == RGY_CHROMAFMT_YUV420 || chromaFmt == RGY_CHROMAFMT_YUV422)) ? 1 : 0;
        const int yShift = (chromaPlane && chromaFmt == RGY_CHROMAFMT_YUV420) ? 1 : 0;
        const int srcXOffset = cropX >> xShift;
        const int srcYOffset = ((cropY >> yShift) << 1) + fieldParity;
        const int srcYStep = 2;
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
        RGYOpenCLEvent planeEvent;
        auto err = m_programs[KFM_PROG_UCF].get()->kernel("kernel_kfm_ucf_field_crop").config(queue, RGYWorkSize(32, 8), RGYWorkSize(dst.width, dst.height), waitHere, &planeEvent).launch(
            (cl_mem)dst.ptr[0], dst.pitch[0],
            (cl_mem)src.ptr[0], src.pitch[0],
            dst.width, dst.height,
            srcXOffset, srcYOffset, srcYStep);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_ucf_field_crop (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
        prevEvent = planeEvent;
    }

    copyFramePropWithoutRes(pFieldFrame, pInputFrame);
    pFieldFrame->timestamp = pInputFrame->timestamp;
    pFieldFrame->duration = pInputFrame->duration;
    pFieldFrame->inputFrameId = pInputFrame->inputFrameId;
    pFieldFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pFieldFrame->flags = RGY_FRAME_FLAG_NONE;
    pFieldFrame->dataList = pInputFrame->dataList;
    writeFrameInfoDump("ucf-field", pFieldFrame);
    auto sts = dumpStageFrame("ucf-field", pFieldFrame, sourceIndex * 2 + fieldParity, queue,
        (prevEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    *ppFieldFrame = pFieldFrame;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::prepareUcfNoiseGaussFrame(RGYFrameInfo **ppGaussFrame, int parity, const RGYFrameInfo *pInputFrame,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!ppGaussFrame || !pInputFrame || !pInputFrame->ptr[0] || !m_programs[KFM_PROG_UCF].get()) {
        return RGY_ERR_INVALID_CALL;
    }
    *ppGaussFrame = nullptr;
    const int frameIndex = parity & 1;
    RGYFrameInfo gaussInfo = *pInputFrame;
    for (int i = 0; i < RGY_MAX_PLANES; i++) {
        gaussInfo.ptr[i] = nullptr;
        gaussInfo.pitch[i] = 0;
    }
    auto allocFrame = [&](std::unique_ptr<RGYCLFrame>& frame, const TCHAR *label) -> RGY_ERR {
        if (!frame
            || frame->frame.width != gaussInfo.width
            || frame->frame.height != gaussInfo.height
            || frame->frame.csp != gaussInfo.csp) {
            frame = m_cl->createFrameBuffer(gaussInfo);
            if (!frame) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM UCF %s frame.\n"), label);
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
        return RGY_ERR_NONE;
    };
    auto sts = allocFrame(m_ucfNoiseGaussTmpFrames[frameIndex], _T("gauss temporary"));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = allocFrame(m_ucfNoiseGaussFrames[frameIndex], _T("gauss output"));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    auto *pTmpFrame = &m_ucfNoiseGaussTmpFrames[frameIndex]->frame;
    auto *pGaussFrame = &m_ucfNoiseGaussFrames[frameIndex]->frame;
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    const bool interleavedUV = kfmCspHasInterleavedUV(pInputFrame->csp);
    RGYOpenCLEvent prevEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto src = getPlane(pInputFrame, (RGY_PLANE)iplane);
        const auto tmp = getPlane(pTmpFrame, (RGY_PLANE)iplane);
        const auto dst = getPlane(pGaussFrame, (RGY_PLANE)iplane);
        if (!src.ptr[0] || !tmp.ptr[0] || !dst.ptr[0]) {
            continue;
        }
        const int chromaIndex = (iplane == 0) ? 0 : 1;
        const bool interleavedUVPlane = interleavedUV && iplane != 0;
        const int srcWidthForGauss = interleavedUVPlane ? (src.width >> 1) : src.width;
        const int dstWidthForGauss = interleavedUVPlane ? (dst.width >> 1) : dst.width;
        const double cropWidth = (pInputFrame->width > 0)
            ? (double)srcWidthForGauss + KFM_UCF_GAUSS_CROP_EPS * (double)srcWidthForGauss / pInputFrame->width
            : (double)srcWidthForGauss + KFM_UCF_GAUSS_CROP_EPS;
        const double cropHeight = (pInputFrame->height > 0)
            ? (double)src.height + KFM_UCF_GAUSS_CROP_EPS * (double)src.height / pInputFrame->height
            : (double)src.height + KFM_UCF_GAUSS_CROP_EPS;
        auto& progV = m_ucfNoiseGaussVert[frameIndex][chromaIndex];
        auto& progH = m_ucfNoiseGaussHori[frameIndex][chromaIndex];
        if (!progV.offset || progV.sourceSize != src.height || progV.targetSize != dst.height) {
            sts = createUcfGaussProgram(progV, src.height, cropHeight, dst.height, KFM_UCF_GAUSS_P);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if (!progH.offset || progH.sourceSize != srcWidthForGauss || progH.targetSize != dstWidthForGauss) {
            sts = createUcfGaussProgram(progH, srcWidthForGauss, cropWidth, dstWidthForGauss, KFM_UCF_GAUSS_P);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
        RGYOpenCLEvent evV;
        auto err = m_programs[KFM_PROG_UCF].get()->kernel("kernel_kfm_ucf_gaussresize_v")
            .config(queue, RGYWorkSize(32, 8), RGYWorkSize(tmp.width, tmp.height), waitHere, &evV)
            .launch((cl_mem)tmp.ptr[0], tmp.pitch[0],
                (cl_mem)src.ptr[0], src.pitch[0],
                tmp.width, tmp.height,
                progV.offset->mem(), progV.coeff->mem(), progV.filterSize);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_ucf_gaussresize_v (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
        RGYOpenCLEvent evH;
        if (interleavedUVPlane) {
            err = m_programs[KFM_PROG_UCF].get()->kernel("kernel_kfm_ucf_gaussresize_h_uv_interleaved")
                .config(queue, RGYWorkSize(32, 8), RGYWorkSize(dst.width, dst.height), { evV }, &evH)
                .launch((cl_mem)dst.ptr[0], dst.pitch[0],
                    (cl_mem)tmp.ptr[0], tmp.pitch[0],
                    dst.width, dst.height, dstWidthForGauss,
                    progH.offset->mem(), progH.coeff->mem(), progH.filterSize);
        } else {
            err = m_programs[KFM_PROG_UCF].get()->kernel("kernel_kfm_ucf_gaussresize_h")
                .config(queue, RGYWorkSize(32, 8), RGYWorkSize(dst.width, dst.height), { evV }, &evH)
                .launch((cl_mem)dst.ptr[0], dst.pitch[0],
                    (cl_mem)tmp.ptr[0], tmp.pitch[0],
                    dst.width, dst.height,
                    progH.offset->mem(), progH.coeff->mem(), progH.filterSize);
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_ucf_gaussresize_h (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
        prevEvent = evH;
    }

    copyFramePropWithoutRes(pGaussFrame, pInputFrame);
    pGaussFrame->timestamp = pInputFrame->timestamp;
    pGaussFrame->duration = pInputFrame->duration;
    pGaussFrame->inputFrameId = pInputFrame->inputFrameId;
    pGaussFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pGaussFrame->flags = RGY_FRAME_FLAG_NONE;
    pGaussFrame->dataList = pInputFrame->dataList;
    writeFrameInfoDump("ucf-noise-gauss", pGaussFrame);
    sts = dumpStageFrame("ucf-noise-gauss", pGaussFrame, m_timecodeFrameIndex * 2 + frameIndex, queue,
        (prevEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    *ppGaussFrame = pGaussFrame;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::prepareUcfNoiseGaussFrameFromSource(RGYFrameInfo **ppGaussFrame, int sourceIndex, int parity, const RGYFrameInfo *pInputFrame,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!ppGaussFrame || !pInputFrame || !pInputFrame->ptr[0] || !m_programs[KFM_PROG_UCF].get()) {
        return RGY_ERR_INVALID_CALL;
    }
    *ppGaussFrame = nullptr;
    const int cropX = 4;
    const int cropY = 4;
    RGYFrameInfo fieldInfo = *pInputFrame;
    fieldInfo.width = pInputFrame->width - cropX * 2;
    fieldInfo.height = (pInputFrame->height >> 1) - cropY * 2;
    if (fieldInfo.width <= 0 || fieldInfo.height <= 0 || (fieldInfo.width & 1) || (fieldInfo.height & 1)) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM UCF field/crop size from %dx%d.\n"), pInputFrame->width, pInputFrame->height);
        return RGY_ERR_INVALID_PARAM;
    }
    fieldInfo.picstruct = RGY_PICSTRUCT_FRAME;
    fieldInfo.flags = RGY_FRAME_FLAG_NONE;
    fieldInfo.inputFrameId = pInputFrame->inputFrameId;
    fieldInfo.timestamp = pInputFrame->timestamp;
    fieldInfo.duration = pInputFrame->duration;
    writeFrameInfoDump("ucf-field", &fieldInfo);

    const int frameIndex = parity & 1;
    RGYFrameInfo gaussInfo = fieldInfo;
    for (int i = 0; i < RGY_MAX_PLANES; i++) {
        gaussInfo.ptr[i] = nullptr;
        gaussInfo.pitch[i] = 0;
    }
    auto allocFrame = [&](std::unique_ptr<RGYCLFrame>& frame, const TCHAR *label) -> RGY_ERR {
        if (!frame
            || frame->frame.width != gaussInfo.width
            || frame->frame.height != gaussInfo.height
            || frame->frame.csp != gaussInfo.csp) {
            frame = m_cl->createFrameBuffer(gaussInfo);
            if (!frame) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM UCF %s frame.\n"), label);
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
        return RGY_ERR_NONE;
    };
    auto sts = allocFrame(m_ucfNoiseGaussTmpFrames[frameIndex], _T("gauss temporary"));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = allocFrame(m_ucfNoiseGaussFrames[frameIndex], _T("gauss output"));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    auto *pTmpFrame = &m_ucfNoiseGaussTmpFrames[frameIndex]->frame;
    auto *pGaussFrame = &m_ucfNoiseGaussFrames[frameIndex]->frame;
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    const bool interleavedUV = kfmCspHasInterleavedUV(pInputFrame->csp);
    const auto chromaFmt = RGY_CSP_CHROMA_FORMAT[pInputFrame->csp];
    const int fieldParity = parity & 1;
    RGYOpenCLEvent prevEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto planeType = (RGY_PLANE)iplane;
        const auto src = getPlane(pInputFrame, planeType);
        const auto tmp = getPlane(pTmpFrame, planeType);
        const auto dst = getPlane(pGaussFrame, planeType);
        if (!src.ptr[0] || !tmp.ptr[0] || !dst.ptr[0]) {
            continue;
        }
        const bool chromaPlane = planeType != RGY_PLANE_Y && chromaFmt != RGY_CHROMAFMT_MONOCHROME;
        const int xShift = (chromaPlane && !interleavedUV
            && (chromaFmt == RGY_CHROMAFMT_YUV420 || chromaFmt == RGY_CHROMAFMT_YUV422)) ? 1 : 0;
        const int yShift = (chromaPlane && chromaFmt == RGY_CHROMAFMT_YUV420) ? 1 : 0;
        const int srcXOffset = cropX >> xShift;
        const int srcYOffset = ((cropY >> yShift) << 1) + fieldParity;
        const int srcYStep = 2;
        const int chromaIndex = (iplane == 0) ? 0 : 1;
        const bool interleavedUVPlane = interleavedUV && iplane != 0;
        const int dstWidthForGauss = interleavedUVPlane ? (dst.width >> 1) : dst.width;
        const double cropWidth = (fieldInfo.width > 0)
            ? (double)dstWidthForGauss + KFM_UCF_GAUSS_CROP_EPS * (double)dstWidthForGauss / fieldInfo.width
            : (double)dstWidthForGauss + KFM_UCF_GAUSS_CROP_EPS;
        const double cropHeight = (fieldInfo.height > 0)
            ? (double)dst.height + KFM_UCF_GAUSS_CROP_EPS * (double)dst.height / fieldInfo.height
            : (double)dst.height + KFM_UCF_GAUSS_CROP_EPS;
        auto& progV = m_ucfNoiseGaussVert[frameIndex][chromaIndex];
        auto& progH = m_ucfNoiseGaussHori[frameIndex][chromaIndex];
        if (!progV.offset || progV.sourceSize != dst.height || progV.targetSize != dst.height) {
            sts = createUcfGaussProgram(progV, dst.height, cropHeight, dst.height, KFM_UCF_GAUSS_P);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if (!progH.offset || progH.sourceSize != dstWidthForGauss || progH.targetSize != dstWidthForGauss) {
            sts = createUcfGaussProgram(progH, dstWidthForGauss, cropWidth, dstWidthForGauss, KFM_UCF_GAUSS_P);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
        RGYOpenCLEvent evV;
        auto err = m_programs[KFM_PROG_UCF].get()->kernel("kernel_kfm_ucf_field_crop_gaussresize_v")
            .config(queue, RGYWorkSize(32, 8), RGYWorkSize(tmp.width, tmp.height), waitHere, &evV)
            .launch((cl_mem)tmp.ptr[0], tmp.pitch[0],
                (cl_mem)src.ptr[0], src.pitch[0],
                tmp.width, tmp.height,
                srcXOffset, srcYOffset, srcYStep,
                progV.offset->mem(), progV.coeff->mem(), progV.filterSize);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_ucf_field_crop_gaussresize_v (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
        RGYOpenCLEvent evH;
        if (interleavedUVPlane) {
            err = m_programs[KFM_PROG_UCF].get()->kernel("kernel_kfm_ucf_gaussresize_h_uv_interleaved")
                .config(queue, RGYWorkSize(32, 8), RGYWorkSize(dst.width, dst.height), { evV }, &evH)
                .launch((cl_mem)dst.ptr[0], dst.pitch[0],
                    (cl_mem)tmp.ptr[0], tmp.pitch[0],
                    dst.width, dst.height, dstWidthForGauss,
                    progH.offset->mem(), progH.coeff->mem(), progH.filterSize);
        } else {
            err = m_programs[KFM_PROG_UCF].get()->kernel("kernel_kfm_ucf_gaussresize_h")
                .config(queue, RGYWorkSize(32, 8), RGYWorkSize(dst.width, dst.height), { evV }, &evH)
                .launch((cl_mem)dst.ptr[0], dst.pitch[0],
                    (cl_mem)tmp.ptr[0], tmp.pitch[0],
                    dst.width, dst.height,
                    progH.offset->mem(), progH.coeff->mem(), progH.filterSize);
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at KFM UCF fused gauss h (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
        prevEvent = evH;
    }

    copyFramePropWithoutRes(pGaussFrame, &fieldInfo);
    pGaussFrame->dataList = pInputFrame->dataList;
    writeFrameInfoDump("ucf-noise-gauss", pGaussFrame);
    sts = dumpStageFrame("ucf-noise-gauss", pGaussFrame, sourceIndex * 2 + frameIndex, queue,
        (prevEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    *ppGaussFrame = pGaussFrame;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::runUcfNoiseLimitStageFromSource(const RGYFilterParamKfm& prm, const RGYFrameInfo *pSrcFrame, const RGYFrameInfo *pNoiseFrame,
    int fieldIndex, int parity, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    (void)prm;
    if (!pSrcFrame || !pNoiseFrame || !pSrcFrame->ptr[0] || !pNoiseFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (!m_programs[KFM_PROG_UCF].get()) {
        return RGY_ERR_INVALID_CALL;
    }
    const int cropX = 4;
    const int cropY = 4;
    if (pSrcFrame->width - cropX * 2 != pNoiseFrame->width || (pSrcFrame->height >> 1) - cropY * 2 != pNoiseFrame->height
        || pSrcFrame->csp != pNoiseFrame->csp) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM UCF fused noise limit input pair (src %dx%d %s, noise %dx%d %s).\n"),
            pSrcFrame->width, pSrcFrame->height, RGY_CSP_NAMES[pSrcFrame->csp],
            pNoiseFrame->width, pNoiseFrame->height, RGY_CSP_NAMES[pNoiseFrame->csp]);
        return RGY_ERR_INVALID_PARAM;
    }
    KfmCachedUcfNoise entry;
    entry.fieldIndex = fieldIndex;
    entry.inputFrameId = pSrcFrame->inputFrameId;
    entry.timestamp = pSrcFrame->timestamp;
    entry.frame = acquireKfmFrame(*pNoiseFrame, _T("UCF fused noise limit"));
    if (!entry.frame) {
        return RGY_ERR_MEMORY_ALLOC;
    }

    auto *pOutputFrame = &entry.frame->frame;
    const auto planes = RGY_CSP_PLANES[pOutputFrame->csp];
    const bool interleavedUV = kfmCspHasInterleavedUV(pSrcFrame->csp);
    const auto chromaFmt = RGY_CSP_CHROMA_FORMAT[pSrcFrame->csp];
    const int fieldParity = parity & 1;
    RGYOpenCLEvent prevEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto planeType = (RGY_PLANE)iplane;
        const auto dst = getPlane(pOutputFrame, planeType);
        const auto src = getPlane(pSrcFrame, planeType);
        const auto noise = getPlane(pNoiseFrame, planeType);
        if (!dst.ptr[0] || !src.ptr[0] || !noise.ptr[0]) {
            continue;
        }
        const bool chromaPlane = planeType != RGY_PLANE_Y && chromaFmt != RGY_CHROMAFMT_MONOCHROME;
        const int xShift = (chromaPlane && !interleavedUV
            && (chromaFmt == RGY_CHROMAFMT_YUV420 || chromaFmt == RGY_CHROMAFMT_YUV422)) ? 1 : 0;
        const int yShift = (chromaPlane && chromaFmt == RGY_CHROMAFMT_YUV420) ? 1 : 0;
        const int srcXOffset = cropX >> xShift;
        const int srcYOffset = ((cropY >> yShift) << 1) + fieldParity;
        const int srcYStep = 2;
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
        RGYOpenCLEvent planeEvent;
        auto err = m_programs[KFM_PROG_UCF].get()->kernel("kernel_kfm_ucf_source_crop_noise_limit").config(queue, RGYWorkSize(32, 8), RGYWorkSize(dst.width, dst.height), waitHere, &planeEvent).launch(
            (cl_mem)dst.ptr[0], dst.pitch[0],
            (cl_mem)src.ptr[0], src.pitch[0],
            (cl_mem)noise.ptr[0], noise.pitch[0],
            dst.width, dst.height,
            srcXOffset, srcYOffset, srcYStep,
            KFM_UCF_NOISE_LIMIT_NMIN, KFM_UCF_NOISE_LIMIT_RANGE);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_ucf_source_crop_noise_limit (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
        prevEvent = planeEvent;
    }

    copyFramePropWithoutRes(pOutputFrame, pNoiseFrame);
    pOutputFrame->dataList = pSrcFrame->dataList;
    writeFrameInfoDump("ucf-noise-clip", pOutputFrame);
    auto sts = dumpStageFrame("ucf-noise-clip", pOutputFrame, fieldIndex, queue,
        (prevEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (prevEvent() != nullptr) {
        entry.event = prevEvent;
    }
    m_ucfNoiseCache.push_back(std::move(entry));
    while (m_ucfNoiseCache.size() > sourceCacheLimit()) {
        m_ucfNoiseCache.pop_front();
    }
    return analyzeUcfNoiseDebug(queue);
}

RGY_ERR RGYFilterKfm::runUcfNoiseLimitStage(const RGYFilterParamKfm& prm, const RGYFrameInfo *pSrcFrame, const RGYFrameInfo *pNoiseFrame,
    int fieldIndex, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    (void)prm;
    if (!pSrcFrame || !pNoiseFrame || !pSrcFrame->ptr[0] || !pNoiseFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (!m_programs[KFM_PROG_UCF].get()) {
        return RGY_ERR_INVALID_CALL;
    }
    if (pSrcFrame->width != pNoiseFrame->width || pSrcFrame->height != pNoiseFrame->height || pSrcFrame->csp != pNoiseFrame->csp) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM UCF noise limit input pair (src %dx%d %s, noise %dx%d %s).\n"),
            pSrcFrame->width, pSrcFrame->height, RGY_CSP_NAMES[pSrcFrame->csp],
            pNoiseFrame->width, pNoiseFrame->height, RGY_CSP_NAMES[pNoiseFrame->csp]);
        return RGY_ERR_INVALID_PARAM;
    }
    KfmCachedUcfNoise entry;
    entry.fieldIndex = fieldIndex;
    entry.inputFrameId = pSrcFrame->inputFrameId;
    entry.timestamp = pSrcFrame->timestamp;
    entry.frame = acquireKfmFrame(*pSrcFrame, _T("UCF noise limit"));
    if (!entry.frame) {
        return RGY_ERR_MEMORY_ALLOC;
    }

    auto *pOutputFrame = &entry.frame->frame;
    const auto planes = RGY_CSP_PLANES[pOutputFrame->csp];
    RGYOpenCLEvent prevEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dst = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto src = getPlane(pSrcFrame, (RGY_PLANE)iplane);
        const auto noise = getPlane(pNoiseFrame, (RGY_PLANE)iplane);
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
        RGYOpenCLEvent planeEvent;
        auto err = m_programs[KFM_PROG_UCF].get()->kernel("kernel_kfm_ucf_noise_limit").config(queue, RGYWorkSize(32, 8), RGYWorkSize(dst.width, dst.height), waitHere, &planeEvent).launch(
            (cl_mem)dst.ptr[0], dst.pitch[0],
            (cl_mem)src.ptr[0], src.pitch[0],
            (cl_mem)noise.ptr[0], noise.pitch[0],
            dst.width, dst.height,
            KFM_UCF_NOISE_LIMIT_NMIN, KFM_UCF_NOISE_LIMIT_RANGE);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_ucf_noise_limit (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
        prevEvent = planeEvent;
    }

    copyFramePropWithoutRes(pOutputFrame, pSrcFrame);
    pOutputFrame->dataList = pSrcFrame->dataList;
    writeFrameInfoDump("ucf-noise-clip", pOutputFrame);
    auto sts = dumpStageFrame("ucf-noise-clip", pOutputFrame, fieldIndex, queue,
        (prevEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (prevEvent() != nullptr) {
        entry.event = prevEvent;
    }
    m_ucfNoiseCache.push_back(std::move(entry));
    while (m_ucfNoiseCache.size() > sourceCacheLimit()) {
        m_ucfNoiseCache.pop_front();
    }
    return analyzeUcfNoiseDebug(queue);
}

RGY_ERR RGYFilterKfm::analyzeUcfNoiseDebug(RGYOpenCLQueue &queue) {
    if (m_ucfNoiseCache.size() < 3) {
        return RGY_ERR_NONE;
    }
    while (m_ucfNoiseCache.size() >= 3) {
        const auto& noise0 = m_ucfNoiseCache[0];
        const auto& noise1 = m_ucfNoiseCache[1];
        const auto& noise2 = m_ucfNoiseCache[2];
        if (noise0.fieldIndex < 0 || (noise0.fieldIndex & 1) != 0
            || noise1.fieldIndex != noise0.fieldIndex + 1
            || noise2.fieldIndex != noise0.fieldIndex + 2) {
            m_ucfNoiseCache.pop_front();
            continue;
        }
        const int sourceIndex = noise0.fieldIndex >> 1;
        const auto *source0 = findSourceByIndexExact(sourceIndex);
        const auto *source1 = findSourceByIndexExact(sourceIndex + 1);
        if (!source0 || !source1 || !source0->paddedFrame || !source1->paddedFrame) {
            break;
        }
        auto sts = submitUcfNoiseResult(noise0, noise1, noise2, *source0, *source1, queue);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_ucfNoiseCache.pop_front();
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::submitUcfNoiseResult(const KfmCachedUcfNoise& noise0, const KfmCachedUcfNoise& noise1, const KfmCachedUcfNoise& noise2,
    const KfmCachedSource& source0, const KfmCachedSource& source1, RGYOpenCLQueue &queue) {
    if (!m_programs[KFM_PROG_UCF].get() || !noise0.frame || !noise1.frame || !noise2.frame || !source0.paddedFrame || !source1.paddedFrame) {
        return RGY_ERR_INVALID_CALL;
    }

    KfmPendingUcfNoiseResult pending;
    pending.sourceIndex = noise0.fieldIndex >> 1;
    pending.meta.srcw = source0.frame->frame.width;
    pending.meta.srch = source0.frame->frame.height;
    pending.meta.srcUVw = pending.meta.srcw >> 1;
    pending.meta.srcUVh = pending.meta.srch >> 1;
    const auto noiseY = getPlane(&noise0.frame->frame, RGY_PLANE_Y);
    const auto noiseUV = (RGY_CSP_PLANES[noise0.frame->frame.csp] > 1) ? getPlane(&noise0.frame->frame, RGY_PLANE_U) : RGYFrameInfo();
    pending.meta.noisew = noiseY.width;
    pending.meta.noiseh = noiseY.height;
    pending.meta.noiseUVw = noiseUV.ptr[0] ? (kfmCspHasInterleavedUV(noise0.frame->frame.csp) ? (noiseUV.width >> 1) : noiseUV.width) : 0;
    pending.meta.noiseUVh = noiseUV.ptr[0] ? noiseUV.height : 0;

    std::vector<RGYOpenCLEvent> events;
    int partialCount = 0;
    const int localX = 32;
    const int localY = 8;
    const auto reservePartials = [&](int width4, int height) {
        return divCeil(width4, localX) * divCeil(height, localY);
    };
    const auto addPartialCount = [&](const RGYFrameInfo *frame, bool diff) {
        const auto chromaFmt = RGY_CSP_CHROMA_FORMAT[frame->csp];
        const int planes = RGY_CSP_PLANES[frame->csp];
        for (int iplane = 0; iplane < planes; iplane++) {
            const auto plane = getPlane(frame, (RGY_PLANE)iplane);
            const int width4 = plane.width >> 2;
            const bool chromaPlane = iplane != 0 && chromaFmt != RGY_CHROMAFMT_MONOCHROME;
            const int yShift = (chromaPlane && chromaFmt == RGY_CHROMAFMT_YUV420) ? 1 : 0;
            const int vpad = KFM_SOURCE_VPAD >> yShift;
            const int height = diff ? plane.height - vpad * 2 : plane.height;
            if (width4 > 0 && height > 0 && (plane.width & 3) == 0) {
                partialCount += reservePartials(width4, height);
            }
        }
    };
    addPartialCount(&noise0.frame->frame, false);
    addPartialCount(&source0.paddedFrame->frame, true);
    if (partialCount <= 0) {
        RGYKFM::NoiseResult results[2] = {};
        pushUcfNoiseResultDump(pending.sourceIndex, results, pending.meta);
        return RGY_ERR_NONE;
    }
    const size_t requiredBytes = sizeof(RGYKFM::NoiseResult) * (size_t)partialCount;
    pending.resultBuf = acquireUcfNoiseResultBuf(requiredBytes);
    if (!pending.resultBuf) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM UCF noise result buffer.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }

    int offset = 0;
    const auto launchNoise = [&](RGY_PLANE planeType, int resultPlane, std::vector<RGYOpenCLEvent> waitEvents) -> RGY_ERR {
        const auto p0 = getPlane(&noise0.frame->frame, planeType);
        const auto p1 = getPlane(&noise1.frame->frame, planeType);
        const auto p2 = getPlane(&noise2.frame->frame, planeType);
        const int width4 = p0.width >> 2;
        if (!p0.ptr[0] || !p1.ptr[0] || !p2.ptr[0] || width4 <= 0 || p0.height <= 0 || (p0.width & 3) != 0) {
            return RGY_ERR_NONE;
        }
        if (p0.width != p1.width || p0.width != p2.width || p0.height != p1.height || p0.height != p2.height) {
            return RGY_ERR_INVALID_PARAM;
        }
        const int count = reservePartials(width4, p0.height);
        RGYOpenCLEvent event;
        auto err = m_programs[KFM_PROG_UCF].get()->kernel("kernel_kfm_ucf_analyze_noise_partial").config(queue, RGYWorkSize(localX, localY), RGYWorkSize(width4, p0.height), waitEvents, &event).launch(
            (cl_mem)pending.resultBuf->mem(), offset,
            (cl_mem)p0.ptr[0], (cl_mem)p1.ptr[0], (cl_mem)p2.ptr[0], p0.pitch[0],
            width4, p0.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_ucf_analyze_noise_partial (plane %d): %s.\n"), (int)planeType, get_err_mes(err));
            return err;
        }
        pending.segments.push_back({ offset, count, resultPlane });
        offset += count;
        if (event() != nullptr) {
            events.push_back(event);
        }
        return RGY_ERR_NONE;
    };
    const auto launchDiff = [&](RGY_PLANE planeType, int resultPlane, std::vector<RGYOpenCLEvent> waitEvents) -> RGY_ERR {
        const auto p0 = getPlane(&source0.paddedFrame->frame, planeType);
        const auto p1 = getPlane(&source1.paddedFrame->frame, planeType);
        const auto chromaFmt = RGY_CSP_CHROMA_FORMAT[source0.paddedFrame->frame.csp];
        const bool chromaPlane = planeType != RGY_PLANE_Y && chromaFmt != RGY_CHROMAFMT_MONOCHROME;
        const int yShift = (chromaPlane && chromaFmt == RGY_CHROMAFMT_YUV420) ? 1 : 0;
        const int vpad = KFM_SOURCE_VPAD >> yShift;
        const int height = p0.height - vpad * 2;
        const int width4 = p0.width >> 2;
        if (!p0.ptr[0] || !p1.ptr[0] || width4 <= 0 || height <= 0 || (p0.width & 3) != 0) {
            return RGY_ERR_NONE;
        }
        if (p0.width != p1.width || p0.height != p1.height) {
            return RGY_ERR_INVALID_PARAM;
        }
        const int count = reservePartials(width4, height);
        RGYOpenCLEvent event;
        auto err = m_programs[KFM_PROG_UCF].get()->kernel("kernel_kfm_ucf_analyze_diff_partial").config(queue, RGYWorkSize(localX, localY), RGYWorkSize(width4, height), waitEvents, &event).launch(
            (cl_mem)pending.resultBuf->mem(), offset,
            (cl_mem)p0.ptr[0], (cl_mem)p1.ptr[0], p0.pitch[0],
            width4, height, vpad);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_ucf_analyze_diff_partial (plane %d): %s.\n"), (int)planeType, get_err_mes(err));
            return err;
        }
        pending.segments.push_back({ offset, count, resultPlane });
        offset += count;
        if (event() != nullptr) {
            events.push_back(event);
        }
        return RGY_ERR_NONE;
    };

    std::vector<RGYOpenCLEvent> noiseWaits;
    if (noise0.event() != nullptr) noiseWaits.push_back(noise0.event);
    if (noise1.event() != nullptr) noiseWaits.push_back(noise1.event);
    if (noise2.event() != nullptr) noiseWaits.push_back(noise2.event);
    auto sts = launchNoise(RGY_PLANE_Y, 0, noiseWaits);
    if (sts != RGY_ERR_NONE) return sts;
    const bool interleavedUV = kfmCspHasInterleavedUV(noise0.frame->frame.csp);
    if (interleavedUV) {
        sts = launchNoise(RGY_PLANE_U, 1, noiseWaits);
        if (sts != RGY_ERR_NONE) return sts;
    } else if (RGY_CSP_PLANES[noise0.frame->frame.csp] >= 3) {
        sts = launchNoise(RGY_PLANE_U, 1, noiseWaits);
        if (sts != RGY_ERR_NONE) return sts;
        sts = launchNoise(RGY_PLANE_V, 1, noiseWaits);
        if (sts != RGY_ERR_NONE) return sts;
    }

    std::vector<RGYOpenCLEvent> diffWaits;
    if (source0.paddedEvent() != nullptr) diffWaits.push_back(source0.paddedEvent);
    if (source1.paddedEvent() != nullptr) diffWaits.push_back(source1.paddedEvent);
    sts = launchDiff(RGY_PLANE_Y, 0, diffWaits);
    if (sts != RGY_ERR_NONE) return sts;
    if (interleavedUV) {
        sts = launchDiff(RGY_PLANE_U, 1, diffWaits);
        if (sts != RGY_ERR_NONE) return sts;
    } else if (RGY_CSP_PLANES[source0.paddedFrame->frame.csp] >= 3) {
        sts = launchDiff(RGY_PLANE_U, 1, diffWaits);
        if (sts != RGY_ERR_NONE) return sts;
        sts = launchDiff(RGY_PLANE_V, 1, diffWaits);
        if (sts != RGY_ERR_NONE) return sts;
    }

    sts = pending.resultBuf->queueMapBuffer(queue, CL_MAP_READ, events, RGY_CL_MAP_BLOCK_NONE, "kfm.ucf.noise.result");
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to map KFM UCF noise result buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    queue.flush();
    m_pendingUcfNoiseResults.push_back(std::move(pending));
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::resolveUcfNoiseResult(KfmPendingUcfNoiseResult& pending, RGYOpenCLQueue &queue) {
    if (!pending.resultBuf) {
        return RGY_ERR_NULL_PTR;
    }
    auto sts = pending.resultBuf->mapEvent().wait();
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to wait KFM UCF noise result map event: %s.\n"), get_err_mes(sts));
        return sts;
    }
    const auto *partials = reinterpret_cast<const RGYKFM::NoiseResult *>(pending.resultBuf->mappedPtr());
    if (!partials) {
        pending.resultBuf->unmapBuffer(queue);
        return RGY_ERR_NULL_PTR;
    }
    RGYKFM::NoiseResult results[2] = {};
    results[0] = {};
    results[1] = {};
    for (const auto& segment : pending.segments) {
        auto& result = results[segment.plane];
        for (int i = 0; i < segment.count; i++) {
            const auto& partial = partials[segment.offset + i];
            result.noise0 += partial.noise0;
            result.noise1 += partial.noise1;
            result.noiseR0 += partial.noiseR0;
            result.noiseR1 += partial.noiseR1;
            result.diff0 += partial.diff0;
            result.diff1 += partial.diff1;
        }
    }
    sts = pending.resultBuf->unmapBuffer(queue);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to unmap KFM UCF noise result buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    pushUcfNoiseResultDump(pending.sourceIndex, results, pending.meta);
    releaseUcfNoiseResultBuf(std::move(pending.resultBuf));
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::resolveUcfNoiseResults(int sourceIndex, RGYOpenCLQueue &queue) {
    while (!m_pendingUcfNoiseResults.empty() && m_pendingUcfNoiseResults.front().sourceIndex <= sourceIndex) {
        auto sts = resolveUcfNoiseResult(m_pendingUcfNoiseResults.front(), queue);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_pendingUcfNoiseResults.pop_front();
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::resolveAllUcfNoiseResults(RGYOpenCLQueue &queue) {
    while (!m_pendingUcfNoiseResults.empty()) {
        auto sts = resolveUcfNoiseResult(m_pendingUcfNoiseResults.front(), queue);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_pendingUcfNoiseResults.pop_front();
    }
    return RGY_ERR_NONE;
}

std::unique_ptr<RGYCLBuf> RGYFilterKfm::acquireUcfNoiseResultBuf(size_t requiredBytes) {
    auto it = std::find_if(m_ucfNoiseResultBufPool.begin(), m_ucfNoiseResultBufPool.end(),
        [requiredBytes](const std::unique_ptr<RGYCLBuf>& buf) {
            return buf && buf->size() >= requiredBytes;
        });
    if (it != m_ucfNoiseResultBufPool.end()) {
        auto buf = std::move(*it);
        m_ucfNoiseResultBufPool.erase(it);
        return buf;
    }
    return m_cl->createBuffer(requiredBytes, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
}

std::unique_ptr<RGYCLBuf> RGYFilterKfm::acquireFMCountBuf(size_t requiredBytes) {
    auto it = std::find_if(m_fmCountBufPool.begin(), m_fmCountBufPool.end(),
        [requiredBytes](const std::unique_ptr<RGYCLBuf>& buf) {
            return buf && buf->size() >= requiredBytes;
        });
    if (it != m_fmCountBufPool.end()) {
        auto buf = std::move(*it);
        m_fmCountBufPool.erase(it);
        return buf;
    }
    return m_cl->createBuffer(requiredBytes, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
}

void RGYFilterKfm::releaseFMCountBuf(std::unique_ptr<RGYCLBuf>&& buf) {
    if (!buf) {
        return;
    }
    static constexpr size_t KFM_FMCOUNT_BUF_POOL_MAX = KFM_FMCOUNT_PAIRS * 2;
    m_fmCountBufPool.push_back(std::move(buf));
    while (m_fmCountBufPool.size() > KFM_FMCOUNT_BUF_POOL_MAX) {
        m_fmCountBufPool.pop_front();
    }
}

void RGYFilterKfm::releaseUcfNoiseResultBuf(std::unique_ptr<RGYCLBuf>&& buf) {
    if (!buf) {
        return;
    }
    static constexpr size_t KFM_UCF_NOISE_RESULT_BUF_POOL_MAX = 8;
    m_ucfNoiseResultBufPool.push_back(std::move(buf));
    while (m_ucfNoiseResultBufPool.size() > KFM_UCF_NOISE_RESULT_BUF_POOL_MAX) {
        m_ucfNoiseResultBufPool.pop_front();
    }
}

RGY_ERR RGYFilterKfm::clearPendingFMCounts() {
    if (m_pendingFMCounts.empty()) {
        return RGY_ERR_NONE;
    }

    RGY_ERR sts = RGY_ERR_NONE;
    RGYOpenCLQueue *queue = m_fmCountQueue.get() ? &m_fmCountQueue : nullptr;
    for (auto& pending : m_pendingFMCounts) {
        auto& fmCountBuf = pending.countBuf;
        if (!fmCountBuf) {
            continue;
        }
        if (fmCountBuf->isMapped()) {
            const auto waitSts = fmCountBuf->mapEvent().wait();
            if (waitSts != RGY_ERR_NONE && sts == RGY_ERR_NONE) {
                sts = waitSts;
            }
            const auto unmapSts = queue ? fmCountBuf->unmapBuffer(*queue) : fmCountBuf->unmapBuffer();
            if (unmapSts != RGY_ERR_NONE && sts == RGY_ERR_NONE) {
                sts = unmapSts;
            }
        }
    }

    if (queue) {
        const auto finishSts = queue->finish();
        if (finishSts != RGY_ERR_NONE && sts == RGY_ERR_NONE) {
            sts = finishSts;
        }
    }

    for (auto& pending : m_pendingFMCounts) {
        releaseFMCountBuf(std::move(pending.countBuf));
    }
    m_pendingFMCounts.clear();
    return sts;
}

void RGYFilterKfm::pushUcfNoiseResultDump(int sourceIndex, const RGYKFM::NoiseResult (&results)[2], const RGYKFM::UCFNoiseMeta& meta) {
    KfmUcfNoiseDumpRecord record;
    record.sourceIndex = sourceIndex;
    record.results[0] = results[0];
    record.results[1] = results[1];
    record.meta = meta;
    record.valid = true;

    m_ucfNoiseResultCache.push_back(record);
    auto neededByPending = [this](int sourceIndex) {
        for (const auto& pending : m_pendingUcfNoiseResults) {
            if (sourceIndex == pending.sourceIndex || sourceIndex == pending.sourceIndex + 1) {
                return true;
            }
        }
        return false;
    };
    while (m_ucfNoiseResultCache.size() > sourceCacheLimit() && !neededByPending(m_ucfNoiseResultCache.front().sourceIndex)) {
        m_ucfNoiseResultCache.pop_front();
    }

    if (m_fpUcfNoise) {
        if (m_pendingUcfNoiseDump.valid) {
            writeUcfNoiseResultDump(m_pendingUcfNoiseDump, &record);
        }
        m_pendingUcfNoiseDump = record;
    }
}

void RGYFilterKfm::writeUcfNoiseResultDump(const KfmUcfNoiseDumpRecord& record, const KfmUcfNoiseDumpRecord *nextRecord) {
    if (!m_fpUcfNoise) {
        return;
    }
    KfmUcfCalcDumpInfo calc0 = {};
    KfmUcfCalcDumpInfo calc1 = {};
    bool hasCalc0 = false;
    bool hasCalc1 = false;
    try {
        calc0 = kfmUcfCalcDumpInfo(record.meta, record.results, nullptr, false);
        hasCalc0 = true;
        if (nextRecord) {
            calc1 = kfmUcfCalcDumpInfo(record.meta, record.results, nextRecord->results, true);
            hasCalc1 = true;
        }
    } catch (const std::exception& e) {
        AddMessage(RGY_LOG_WARN, _T("failed to calculate KFM UCF classification dump for frame %d: %S.\n"), record.sourceIndex, e.what());
    }
    static const char *planeNames[2] = { "Y", "UV" };
    for (int i = 0; i < 2; i++) {
        const auto& r = record.results[i];
        fprintf(m_fpUcfNoise, "%d\t%s\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%s\t%s\t",
            record.sourceIndex, planeNames[i],
            (unsigned long long)r.noise0,
            (unsigned long long)r.noise1,
            (unsigned long long)r.noiseR0,
            (unsigned long long)r.noiseR1,
            (unsigned long long)r.diff0,
            (unsigned long long)r.diff1,
            hasCalc0 ? calc0.classification : "",
            hasCalc1 ? calc1.classification : "");
        if (hasCalc0) {
            fprintf(m_fpUcfNoise, "%.12g", calc0.fieldDiff);
        }
        fprintf(m_fpUcfNoise, "\t");
        if (hasCalc1) {
            fprintf(m_fpUcfNoise, "%.12g", calc1.fieldDiff);
        }
        fprintf(m_fpUcfNoise, "\t");
        if (hasCalc0) {
            fprintf(m_fpUcfNoise, "%.12g", calc0.diff);
        }
        fprintf(m_fpUcfNoise, "\t");
        if (hasCalc1) {
            fprintf(m_fpUcfNoise, "%.12g", calc1.diff);
        }
        fprintf(m_fpUcfNoise, "\n");
    }
    fflush(m_fpUcfNoise);
}

void RGYFilterKfm::flushUcfNoiseResultDump() {
    if (m_pendingUcfNoiseDump.valid) {
        writeUcfNoiseResultDump(m_pendingUcfNoiseDump, nullptr);
        m_pendingUcfNoiseDump = KfmUcfNoiseDumpRecord();
    }
}

const RGYFrameInfo *RGYFilterKfm::selectUcfDecomb30Frame(int sourceIndex, const RGYFrameInfo *deint30, std::vector<RGYOpenCLEvent> *wait_events) const {
    if (!deint30 || sourceIndex < 0) {
        return deint30;
    }
    const auto selection = planUcfDecomb30Frame(sourceIndex);
    const KfmRtgmcLane *lane = nullptr;
    if (selection.lane == KFM_UCF_LANE_BEFORE) {
        lane = &m_before60Lane;
    } else if (selection.lane == KFM_UCF_LANE_AFTER) {
        lane = &m_after60Lane;
    }
    if (selection.type == KFM_UCF24_SELECT_FRAME && lane && selection.n60 >= 0) {
        const auto *entry = findCachedDeint60Frame(*lane, selection.n60, wait_events);
        return (entry && entry->frame && entry->frame->frame.ptr[0]) ? &entry->frame->frame : deint30;
    }
    return deint30;
}

RGYFilterKfm::KfmUcf24Selection RGYFilterKfm::planUcfDecomb30Frame(int sourceIndex) const {
    KfmUcf24Selection selection;
    if (sourceIndex < 0) {
        return selection;
    }
    const auto *noise = findUcfNoiseResult(sourceIndex);
    if (!noise || !noise->valid) {
        return selection;
    }

    RGYKFM::DecombUCFParam param;
    RGYKFM::DECOMB_UCF_RESULT result = RGYKFM::DECOMB_UCF_CLEAN_1;
    try {
        result = RGYKFM::CalcDecombUCF(&noise->meta, &param, noise->results, nullptr, false);
    } catch (...) {
        return selection;
    }

    const int n60 = sourceIndex * 2;
    if (result == RGYKFM::DECOMB_UCF_USE_0) {
        selection.type = KFM_UCF24_SELECT_FRAME;
        selection.lane = KFM_UCF_LANE_BEFORE;
        selection.n60 = n60;
        return selection;
    }
    if (result == RGYKFM::DECOMB_UCF_USE_1) {
        selection.type = KFM_UCF24_SELECT_FRAME;
        selection.lane = KFM_UCF_LANE_AFTER;
        selection.n60 = n60 + 1;
        return selection;
    }
    return selection;
}

bool RGYFilterKfm::getUcf60FieldDiff(int nstart, double (&diff)[4]) const {
    for (int i = 0; i < 4; i++) {
        const int n = nstart + i;
        const auto *noise = findUcfNoiseResult(n / 2);
        if (!noise || !noise->valid || noise->meta.srcw <= 0 || noise->meta.srch <= 0) {
            return false;
        }
        const double pixels = (double)noise->meta.srcw * noise->meta.srch;
        const uint64_t diffY = (n & 1) ? noise->results[0].diff1 : noise->results[0].diff0;
        const uint64_t diffUV = (n & 1) ? noise->results[1].diff1 : noise->results[1].diff0;
        diff[i] = (double)(diffY + diffUV) / (6.0 * pixels) * 100.0;
    }
    return true;
}

RGYFilterKfm::KfmUcf60Flag RGYFilterKfm::calcUcf60Flag(int n60) const {
    static const RGYKFM::DECOMB_UCF_RESULT replaceResults[2] = {
        RGYKFM::DECOMB_UCF_USE_0,
        RGYKFM::DECOMB_UCF_USE_1,
    };
    static constexpr double UCF60_SC_THRESH = 256.0;
    static constexpr double UCF60_DUP_THRESH = 2.5;

    RGYKFM::DecombUCFParam param;
    int useFrame = n60;
    bool isDirty = false;
    for (int i = 0; i < 2; i++) {
        const int n = n60 + i - 1;
        const auto *noise0 = findUcfNoiseResult(n / 2);
        const auto *noise1 = findUcfNoiseResult(n / 2 + 1);
        if (!noise0 || !noise0->valid || !noise1 || !noise1->valid) {
            continue;
        }

        RGYKFM::DECOMB_UCF_RESULT result = RGYKFM::DECOMB_UCF_CLEAN_1;
        try {
            result = RGYKFM::CalcDecombUCF(&noise0->meta, &param, noise0->results, noise1->results, (n & 1) != 0);
        } catch (...) {
            continue;
        }

        if (result == replaceResults[i]) {
            double diff[4] = {};
            if (i == 0) {
                if (getUcf60FieldDiff(n60 - 3, diff)) {
                    const double sc = diff[3] / (std::max(diff[0], diff[1]) + 0.0001);
                    if (sc > UCF60_DUP_THRESH && diff[3] > UCF60_SC_THRESH) {
                        useFrame = n60 - 1;
                    }
                }
            } else {
                if (getUcf60FieldDiff(n60 - 1, diff)) {
                    const double sc = diff[0] / (std::max(diff[2], diff[3]) + 0.0001);
                    if (sc > UCF60_DUP_THRESH && diff[0] > UCF60_SC_THRESH) {
                        useFrame = n60 + 1;
                    }
                }
            }
        } else if (result == RGYKFM::DECOMB_UCF_NOISY) {
            isDirty = true;
        }
    }

    if (useFrame == n60 && isDirty) {
        return KFM_UCF60_NR;
    }
    if (useFrame < n60) {
        return KFM_UCF60_PREV;
    }
    if (useFrame > n60) {
        return KFM_UCF60_NEXT;
    }
    return KFM_UCF60_NONE;
}

const RGYFrameInfo *RGYFilterKfm::selectUcfDecomb60Frame(int n60, const RGYFrameInfo *deint60, std::vector<RGYOpenCLEvent> *wait_events) const {
    if (!deint60 || n60 < 0) {
        return deint60;
    }
    const auto selection = planUcfDecomb60Frame(n60);
    const KfmRtgmcLane *lane = nullptr;
    if (selection.lane == KFM_UCF_LANE_BEFORE) {
        lane = &m_before60Lane;
    } else if (selection.lane == KFM_UCF_LANE_AFTER) {
        lane = &m_after60Lane;
    }
    if (lane && selection.n60 >= 0) {
        const auto *entry = findCachedDeint60Frame(*lane, selection.n60, wait_events);
        return (entry && entry->frame && entry->frame->frame.ptr[0]) ? &entry->frame->frame : deint60;
    }
    return deint60;
}

RGYFilterKfm::KfmUcf60Selection RGYFilterKfm::planUcfDecomb60Frame(int n60) const {
    KfmUcf60Selection selection;
    if (n60 < 0) {
        return selection;
    }
    const auto centerFlag = calcUcf60Flag(n60);
    KfmUcf60Flag sideFlag = KFM_UCF60_NONE;
    for (int i = -1; i <= 1; i += 2) {
        const auto flag = calcUcf60Flag(n60 + i);
        if (flag == KFM_UCF60_PREV || flag == KFM_UCF60_NEXT) {
            if (i == -1) {
                sideFlag = KFM_UCF60_NEXT;
            } else {
                sideFlag = KFM_UCF60_PREV;
            }
        }
    }

    if (centerFlag == KFM_UCF60_PREV) {
        selection.lane = KFM_UCF_LANE_BEFORE;
        selection.n60 = n60 - 1;
    } else if (centerFlag == KFM_UCF60_NEXT) {
        selection.lane = KFM_UCF_LANE_AFTER;
        selection.n60 = n60 + 1;
    } else if (sideFlag == KFM_UCF60_PREV) {
        selection.lane = KFM_UCF_LANE_BEFORE;
        selection.n60 = n60;
    } else if (sideFlag == KFM_UCF60_NEXT) {
        selection.lane = KFM_UCF_LANE_AFTER;
        selection.n60 = n60;
    }
    return selection;
}

RGYFilterKfm::KfmUcf24Selection RGYFilterKfm::selectUcfDecomb24Frame(const RGYKFM::Frame24Info& frameInfo, const RGYFrameInfo *deint24, std::vector<RGYOpenCLEvent> *wait_events) const {
    auto selection = planUcfDecomb24Frame(frameInfo);
    selection.frame = deint24;
    if (selection.type != KFM_UCF24_SELECT_FRAME || selection.n60 < 0) {
        return selection;
    }
    const KfmRtgmcLane *lane = nullptr;
    if (selection.lane == KFM_UCF_LANE_BEFORE) {
        lane = &m_before60Lane;
    } else if (selection.lane == KFM_UCF_LANE_AFTER) {
        lane = &m_after60Lane;
    }
    if (lane) {
        const auto *entry = findCachedDeint60Frame(*lane, selection.n60, wait_events);
        if (entry && entry->frame && entry->frame->frame.ptr[0]) {
            selection.frame = &entry->frame->frame;
        }
    }
    return selection;
}

RGYFilterKfm::KfmUcf24Selection RGYFilterKfm::planUcfDecomb24Frame(const RGYKFM::Frame24Info& frameInfo) const {
    KfmUcf24Selection selection;
    if (frameInfo.numFields <= 0 || frameInfo.numFields > 6) {
        return selection;
    }

    bool cleanField[6] = { true, true, true, true, true, true };
    RGYKFM::DecombUCFParam param;
    for (int i = 0; i < frameInfo.numFields - 1; i++) {
        const int n60 = frameInfo.cycleIndex * 10 + frameInfo.fieldStartIndex + i;
        const auto *noise0 = findUcfNoiseResult(n60 / 2);
        const auto *noise1 = findUcfNoiseResult(n60 / 2 + 1);
        if (!noise0 || !noise0->valid || !noise1 || !noise1->valid) {
            return selection;
        }

        RGYKFM::DECOMB_UCF_RESULT result = RGYKFM::DECOMB_UCF_CLEAN_1;
        try {
            result = RGYKFM::CalcDecombUCF(&noise0->meta, &param, noise0->results, noise1->results, (n60 & 1) != 0);
        } catch (...) {
            return selection;
        }

        if (result == RGYKFM::DECOMB_UCF_USE_0) {
            cleanField[i + 1] = false;
        } else if (result == RGYKFM::DECOMB_UCF_USE_1) {
            cleanField[i + 0] = false;
        } else if (result == RGYKFM::DECOMB_UCF_NOISY) {
            cleanField[i + 0] = false;
            cleanField[i + 1] = false;
        }
    }

    const bool hasDirty = std::find(cleanField, cleanField + frameInfo.numFields, false) != cleanField + frameInfo.numFields;
    if (!hasDirty) {
        return selection;
    }
    for (int i = 0; i < frameInfo.numFields - 1; i++) {
        if (cleanField[i] && cleanField[i + 1]) {
            selection.type = KFM_UCF24_SELECT_DWEAVE;
            selection.n60 = frameInfo.cycleIndex * 10 + frameInfo.fieldStartIndex + i;
            selection.frame = nullptr;
            return selection;
        }
    }
    if (frameInfo.numFields <= 2) {
        const int n60start = frameInfo.cycleIndex * 10 + frameInfo.fieldStartIndex;
        if (cleanField[0]) {
            selection.type = KFM_UCF24_SELECT_FRAME;
            selection.lane = KFM_UCF_LANE_BEFORE;
            selection.n60 = n60start;
            return selection;
        }
        if (cleanField[1]) {
            selection.type = KFM_UCF24_SELECT_FRAME;
            selection.lane = KFM_UCF_LANE_AFTER;
            selection.n60 = n60start + 1;
            return selection;
        }
    }
    return selection;
}

RGY_ERR RGYFilterKfm::runNrFilter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrame,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!ppOutputFrame) {
        return RGY_ERR_INVALID_PARAM;
    }
    *ppOutputFrame = nullptr;
    if (!m_nrFilter) {
        *ppOutputFrame = pInputFrame;
        return RGY_ERR_NONE;
    }

    RGYFrameInfo *nrOutFrames[2] = {};
    int nrOutNum = 0;
    auto sts = m_nrFilter->filter(pInputFrame, nrOutFrames, &nrOutNum, queue, wait_events, event);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to run KFM NR Auto degrain filter: %s.\n"), get_err_mes(sts));
        return sts;
    }
    if (nrOutNum > 1) {
        AddMessage(RGY_LOG_ERROR, _T("KFM NR Auto degrain returned unexpected output count %d.\n"), nrOutNum);
        return RGY_ERR_INVALID_CALL;
    }
    *ppOutputFrame = (nrOutNum > 0) ? nrOutFrames[0] : nullptr;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::emitOutputFrame(RGYFrameInfo *pFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const RGYOpenCLEvent &frameEvent, RGYOpenCLEvent *event) {
    if (!pFrame || !ppOutputFrames || !pOutputFrameNum) {
        return RGY_ERR_INVALID_PARAM;
    }
    RGYFrameInfo *emitFrame = pFrame;
    RGYOpenCLEvent nrEvent;
    if (m_nrFilter) {
        std::vector<RGYOpenCLEvent> nrWaitEvents;
        if (frameEvent() != nullptr) {
            nrWaitEvents.push_back(frameEvent);
        }
        auto sts = runNrFilter(pFrame, &emitFrame, queue, nrWaitEvents, &nrEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (!emitFrame) {
            if (event && nrEvent() != nullptr) {
                *event = nrEvent;
            }
            return RGY_ERR_NONE;
        }
    }
    auto outputFrame = nextOutputFrame();
    if (!outputFrame || !outputFrame->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    std::vector<RGYOpenCLEvent> copyWaitEvents;
    if (nrEvent() != nullptr) {
        copyWaitEvents.push_back(nrEvent);
    } else if (frameEvent() != nullptr) {
        copyWaitEvents.push_back(frameEvent);
    }
    RGYOpenCLEvent outputEvent;
    if (outputFrame != emitFrame) {
        auto sts = m_cl->copyFrame(outputFrame, emitFrame, nullptr, queue, copyWaitEvents, &outputEvent, RGYFrameCopyMode::FRAME, "kfm.output");
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy KFM output frame: %s.\n"), get_err_mes(sts));
            return sts;
        }
        copyFramePropWithoutRes(outputFrame, emitFrame);
    } else if (!copyWaitEvents.empty()) {
        outputEvent = copyWaitEvents.back();
    }
    if (event) {
        if (outputEvent() != nullptr) {
            *event = outputEvent;
        }
    }
    ppOutputFrames[(*pOutputFrameNum)++] = outputFrame;
    writeFrameInfoDump("output", outputFrame);
    writeFrameTimecode(outputFrame);
    m_timecodeFrameIndex++;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::queueVfrOutputFrame(const RGYFrameInfo *pFrame, RGYOpenCLQueue &queue, const RGYOpenCLEvent &frameEvent) {
    if (!pFrame || !pFrame->ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    KfmPendingVfrOutput pending;
    pending.frame = acquireKfmFrame(*pFrame, _T("VFR delayed output"));
    if (!pending.frame) {
        return RGY_ERR_MEMORY_ALLOC;
    }
    std::vector<RGYOpenCLEvent> copyWaitEvents;
    if (frameEvent() != nullptr) {
        copyWaitEvents.push_back(frameEvent);
    }
    auto sts = m_cl->copyFrame(&pending.frame->frame, pFrame, nullptr, queue, copyWaitEvents, &pending.event, RGYFrameCopyMode::FRAME, "kfm.vfr.delay_output");
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy KFM VFR delayed output frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
    copyFramePropWithoutRes(&pending.frame->frame, pFrame);
    m_pendingVfrOutputs.push_back(std::move(pending));
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::emitPendingVfrOutput(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    if (m_pendingVfrOutputs.empty()) {
        return RGY_ERR_NONE;
    }
    auto pending = std::move(m_pendingVfrOutputs.front());
    m_pendingVfrOutputs.pop_front();
    if (!pending.frame || !pending.frame->frame.ptr[0]) {
        return RGY_ERR_INVALID_CALL;
    }
    return emitOutputFrame(&pending.frame->frame, ppOutputFrames, pOutputFrameNum, queue, pending.event, event);
}

RGY_ERR RGYFilterKfm::emitPendingVfrOutputs(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, RGYOpenCLEvent *event, int keepFrames) {
    keepFrames = std::max(0, keepFrames);
    const int maxOutputFrames = std::min<int>((int)m_frameBuf.size(), 4);
    while ((int)m_pendingVfrOutputs.size() > keepFrames && *pOutputFrameNum < maxOutputFrames) {
        const int outputFrameNumBefore = *pOutputFrameNum;
        auto sts = emitPendingVfrOutput(ppOutputFrames, pOutputFrameNum, queue, event);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (*pOutputFrameNum == outputFrameNumBefore) {
            break;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::drainNrFilter(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    if (!m_nrFilter || !ppOutputFrames || !pOutputFrameNum) {
        return RGY_ERR_NONE;
    }
    RGYFrameInfo *emitFrame = nullptr;
    RGYOpenCLEvent nrEvent;
    auto sts = runNrFilter(nullptr, &emitFrame, queue, {}, &nrEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (!emitFrame) {
        return RGY_ERR_NONE;
    }
    if (event && nrEvent() != nullptr) {
        *event = nrEvent;
    }
    ppOutputFrames[(*pOutputFrameNum)++] = emitFrame;
    writeFrameInfoDump("output", emitFrame);
    writeFrameTimecode(emitFrame);
    m_timecodeFrameIndex++;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::ensureFMCountQueue() {
    if (m_fmCountQueue.get()) {
        return RGY_ERR_NONE;
    }
    m_fmCountQueue = m_cl->createQueue(m_cl->queue().devid(), m_cl->queue().getProperties());
    if (!m_fmCountQueue.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to create KFM FMCount readback queue.\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::submitFMCounts(int cycle, bool drain, RGYOpenCLQueue &queue) {
    if (!m_programs[KFM_PROG_ANALYZE].get()) {
        return RGY_ERR_INVALID_CALL;
    }
    if (std::find_if(m_pendingFMCounts.begin(), m_pendingFMCounts.end(), [cycle](const KfmPendingFMCount& pending) {
        return pending.cycle == cycle;
    }) != m_pendingFMCounts.end()) {
        return RGY_ERR_NONE;
    }
    const int firstSourceIndex = cycle * 5 - 3;
    const int lastSourceIndex = firstSourceIndex + KFM_FMCOUNT_SOURCE_FRAMES - 1;
    if (!drain && m_cachedSourceFrames <= lastSourceIndex) {
        return RGY_ERR_MORE_DATA;
    }
    if (m_cachedSourceFrames <= 0 || m_sourceCache.empty()) {
        return RGY_ERR_MORE_DATA;
    }

    std::array<const KfmCachedSource *, KFM_FMCOUNT_SOURCE_FRAMES> src = {};
    for (int i = 0; i < KFM_FMCOUNT_SOURCE_FRAMES; i++) {
        src[i] = findSourceByIndex(firstSourceIndex + i);
        if (!src[i] || !src[i]->frame || !src[i]->frame->frame.ptr[0]) {
            return RGY_ERR_MORE_DATA;
        }
    }

    auto sts = ensureFMCountQueue();
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    const size_t countBytes = sizeof(RGYKFM::FMCount) * KFM_FMCOUNT_PAIRS * 2;
    KfmPendingFMCount pending;
    pending.cycle = cycle;
    pending.countBuf = acquireFMCountBuf(countBytes);
    if (!pending.countBuf) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM FMCount buffer.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }

    const cl_int zero = 0;
    RGYOpenCLEvent initEvent;
    sts = m_cl->setBuf(&zero, sizeof(zero), countBytes, pending.countBuf.get(), queue, &initEvent);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to clear KFM FMCount buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }

    const bool useFusedFMCount = kfmUseFusedFMCount();
    std::vector<RGYOpenCLEvent> pairCountEvents;
    pairCountEvents.reserve(KFM_FMCOUNT_PAIRS);
    for (int pair = 0; pair < KFM_FMCOUNT_PAIRS; pair++) {
        RGYOpenCLEvent prevCountEvent = initEvent;
        const int dstOffset = pair * 2;
        const auto csp = src[pair + 1]->frame->frame.csp;
        const bool interleavedUV = kfmCspHasInterleavedUV(csp);
        const int targetPlanes = (RGY_CSP_PLANES[csp] >= 3) ? 3 : (interleavedUV ? 3 : 1);
        const int countParity = kfmFrameParity(&src[3]->frame->frame);
        for (int iplane = 0; iplane < targetPlanes; iplane++) {
            const bool interleavedChroma = interleavedUV && iplane > 0;
            const auto plane = interleavedChroma ? RGY_PLANE_U : (RGY_PLANE)iplane;
            const int pixelStep = interleavedChroma ? 2 : 1;
            const int pixelOffset = interleavedChroma ? iplane - 1 : 0;
            const auto prevSrc0 = getPlane(&src[pair + 0]->frame->frame, plane);
            const auto prevSrc1 = getPlane(&src[pair + 1]->frame->frame, plane);
            const auto curSrc0 = prevSrc1;
            const auto curSrc1 = getPlane(&src[pair + 2]->frame->frame, plane);
            if (!prevSrc0.ptr[0] || !prevSrc1.ptr[0] || !curSrc1.ptr[0]) {
                continue;
            }
            if (prevSrc0.width != prevSrc1.width || prevSrc0.height != prevSrc1.height
                || prevSrc0.width != curSrc1.width || prevSrc0.height != curSrc1.height) {
                return RGY_ERR_INVALID_CALL;
            }

            const int gridWidth = (prevSrc0.width / pixelStep) >> 2;
            const int gridHeight = prevSrc0.height >> 2;
            if (gridWidth < 2 || gridHeight < 2) {
                continue;
            }

            const bool chroma = iplane > 0;
            const int threshMove = chroma ? KFM_THRESH_MOVE_C : KFM_THRESH_MOVE_Y;
            const int threshShima = chroma ? KFM_THRESH_SHIMA_C : KFM_THRESH_SHIMA_Y;
            const int cleanThresh = chroma ? KFM_CLEAN_THRESH_C : KFM_CLEAN_THRESH_Y;
            RGYOpenCLEvent countEvent;
            RGYWorkSize countLocal(32, 8);
            RGYWorkSize countGlobal(gridWidth - 1, gridHeight - 1);
            if (useFusedFMCount) {
                std::vector<RGYOpenCLEvent> countWaitEvents;
                countWaitEvents.push_back(prevCountEvent);
                if (src[pair + 0]->event() != nullptr) {
                    countWaitEvents.push_back(src[pair + 0]->event);
                }
                if (src[pair + 1]->event() != nullptr) {
                    countWaitEvents.push_back(src[pair + 1]->event);
                }
                if (src[pair + 2]->event() != nullptr) {
                    countWaitEvents.push_back(src[pair + 2]->event);
                }
                sts = m_programs[KFM_PROG_ANALYZE].get()->kernel("kernel_kfm_analyze_count_cmflags_clean").config(queue, countLocal, countGlobal, countWaitEvents, &countEvent).launch(
                    (cl_mem)pending.countBuf->mem(),
                    dstOffset,
                    (cl_mem)prevSrc0.ptr[0],
                    (cl_mem)prevSrc1.ptr[0],
                    (cl_mem)curSrc0.ptr[0],
                    (cl_mem)curSrc1.ptr[0],
                    prevSrc0.pitch[0],
                    curSrc0.pitch[0],
                    gridWidth - 1, gridHeight - 1,
                    kfmFrameParity(&src[pair + 0]->frame->frame),
                    kfmFrameParity(&src[pair + 1]->frame->frame),
                    countParity,
                    pixelStep, pixelOffset,
                    threshMove, threshShima, threshShima * 3, cleanThresh);
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_analyze_count_cmflags_clean (pair %d, plane %d): %s.\n"), pair, iplane, get_err_mes(sts));
                    return sts;
                }
            } else {
                const int flagHeight = gridHeight * 2;
                const int flagPitch = gridWidth * (int)sizeof(cl_uchar2);
                const size_t flagBytes = (size_t)flagPitch * flagHeight;
                for (auto& flag : m_analyzeFlags) {
                    if (!flag || flag->size() < flagBytes) {
                        flag = m_cl->createBuffer(flagBytes, CL_MEM_READ_WRITE);
                        if (!flag) {
                            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM analyze flag buffer.\n"));
                            return RGY_ERR_MEMORY_ALLOC;
                        }
                    }
                }

                std::vector<RGYOpenCLEvent> prevWaitEvents;
                if (src[pair + 0]->event() != nullptr) {
                    prevWaitEvents.push_back(src[pair + 0]->event);
                }
                if (src[pair + 1]->event() != nullptr) {
                    prevWaitEvents.push_back(src[pair + 1]->event);
                }
                std::vector<RGYOpenCLEvent> curWaitEvents;
                if (src[pair + 1]->event() != nullptr) {
                    curWaitEvents.push_back(src[pair + 1]->event);
                }
                if (src[pair + 2]->event() != nullptr) {
                    curWaitEvents.push_back(src[pair + 2]->event);
                }

                std::array<RGYOpenCLEvent, 2> analyzeEvents;
                RGYWorkSize local(32, 8);
                RGYWorkSize global(gridWidth, gridHeight);
                sts = m_programs[KFM_PROG_ANALYZE].get()->kernel("kernel_kfm_analyze").config(queue, local, global, prevWaitEvents, &analyzeEvents[0]).launch(
                    (cl_mem)m_analyzeFlags[0]->mem(), flagPitch,
                    (cl_mem)prevSrc0.ptr[0],
                    (cl_mem)prevSrc1.ptr[0], prevSrc0.pitch[0],
                    gridWidth, gridHeight, kfmFrameParity(&src[pair + 0]->frame->frame),
                    pixelStep, pixelOffset);
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_analyze (pair %d, plane %d, prev): %s.\n"), pair, iplane, get_err_mes(sts));
                    return sts;
                }
                sts = m_programs[KFM_PROG_ANALYZE].get()->kernel("kernel_kfm_analyze").config(queue, local, global, curWaitEvents, &analyzeEvents[1]).launch(
                    (cl_mem)m_analyzeFlags[1]->mem(), flagPitch,
                    (cl_mem)curSrc0.ptr[0],
                    (cl_mem)curSrc1.ptr[0], curSrc0.pitch[0],
                    gridWidth, gridHeight, kfmFrameParity(&src[pair + 1]->frame->frame),
                    pixelStep, pixelOffset);
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_analyze (pair %d, plane %d, cur): %s.\n"), pair, iplane, get_err_mes(sts));
                    return sts;
                }

                std::vector<RGYOpenCLEvent> countWaitEvents = { prevCountEvent, analyzeEvents[0], analyzeEvents[1] };
                sts = m_programs[KFM_PROG_ANALYZE].get()->kernel("kernel_kfm_count_cmflags_clean").config(queue, countLocal, countGlobal, countWaitEvents, &countEvent).launch(
                    (cl_mem)pending.countBuf->mem(),
                    dstOffset,
                    (cl_mem)m_analyzeFlags[0]->mem(),
                    (cl_mem)m_analyzeFlags[1]->mem(),
                    flagPitch, gridWidth - 1, gridHeight - 1, countParity,
                    threshMove, threshShima, threshShima * 3, cleanThresh);
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_count_cmflags_clean (pair %d, plane %d): %s.\n"), pair, iplane, get_err_mes(sts));
                    return sts;
                }
            }
            prevCountEvent = countEvent;
        }

        if (prevCountEvent() != nullptr) {
            pairCountEvents.push_back(prevCountEvent);
        }
    }
    sts = pending.countBuf->queueMapBuffer(m_fmCountQueue, CL_MAP_READ, pairCountEvents, RGY_CL_MAP_BLOCK_NONE, "kfm.fmcount.cycle");
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to map KFM FMCount buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    queue.flush();
    m_fmCountQueue.flush();
    m_pendingFMCounts.push_back(std::move(pending));
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::readbackFMCounts(std::array<RGYKFM::FMCount, 18>& counts, int cycle, bool drain, RGYOpenCLQueue &queue) {
    if (m_pendingFMCounts.empty() || m_pendingFMCounts.front().cycle != cycle) {
        return RGY_ERR_MORE_DATA;
    }
    if (!drain && m_nextFMCountSubmitCycle - cycle <= KFM_FMCOUNT_ASYNC_DELAY_CYCLES) {
        return RGY_ERR_MORE_DATA;
    }

    auto& pending = m_pendingFMCounts.front();
    counts = {};
    RGYOpenCLQueue& mapQueue = m_fmCountQueue.get() ? m_fmCountQueue : queue;
    RGY_ERR sts = RGY_ERR_NONE;
    auto& fmCountBuf = pending.countBuf;
    if (!fmCountBuf) {
        AddMessage(RGY_LOG_ERROR, _T("KFM FMCount pending buffer is missing.\n"));
        return RGY_ERR_NULL_PTR;
    }
    const auto waitSts = fmCountBuf->mapEvent().wait();
    if (waitSts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to wait KFM FMCount map event: %s.\n"), get_err_mes(waitSts));
        return waitSts;
    }
    const auto *gpuCounts = reinterpret_cast<const RGYKFM::FMCount *>(fmCountBuf->mappedPtr());
    if (!gpuCounts) {
        const auto unmapSts = fmCountBuf->unmapBuffer(mapQueue);
        if (unmapSts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to unmap KFM FMCount buffer after access error: %s.\n"), get_err_mes(unmapSts));
        }
        AddMessage(RGY_LOG_ERROR, _T("failed to access KFM FMCount buffer.\n"));
        return RGY_ERR_NULL_PTR;
    }
    for (int pair = 0; pair < KFM_FMCOUNT_PAIRS; pair++) {
        const int countFrameIndex = pending.cycle * 5 - 3 + pair + 1;
        if (countFrameIndex >= 0) {
            counts[pair * 2 + 0] = gpuCounts[pair * 2 + 0];
            counts[pair * 2 + 1] = gpuCounts[pair * 2 + 1];
        }
    }
    const int firstSourceIndex = pending.cycle * 5 - 3;
    const int firstValidPair = std::max(0, -(firstSourceIndex + 1));
    if (firstValidPair > 0 && firstValidPair < KFM_FMCOUNT_PAIRS) {
        for (int pair = 0; pair < firstValidPair; pair++) {
            counts[pair * 2 + 0] = counts[firstValidPair * 2 + 0];
            counts[pair * 2 + 1] = counts[firstValidPair * 2 + 1];
        }
    }
    sts = fmCountBuf->unmapBuffer(mapQueue);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to unmap KFM FMCount buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = mapQueue.finish();
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to finish KFM FMCount readback queue: %s.\n"), get_err_mes(sts));
        return sts;
    }
    releaseFMCountBuf(std::move(fmCountBuf));
    m_pendingFMCounts.pop_front();
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::analyzeAvailableSource(bool drain, RGYOpenCLQueue &queue) {
    if (!m_analyzer || m_cachedSourceFrames <= 0) {
        return RGY_ERR_NONE;
    }

    const int readyCycles = drain
        ? divCeil(m_cachedSourceFrames, 5)
        : (m_cachedSourceFrames >= 8 ? ((m_cachedSourceFrames - 8) / 5 + 1) : 0);
    const auto prm = std::dynamic_pointer_cast<RGYFilterParamKfm>(m_param);
    const auto timing = prm ? prm->kfm.timing : VppKfmTiming::Realtime;
    while (m_nextFMCountSubmitCycle < readyCycles) {
        auto sts = submitFMCounts(m_nextFMCountSubmitCycle, drain, queue);
        if (sts == RGY_ERR_MORE_DATA) {
            break;
        }
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_nextFMCountSubmitCycle++;
    }

    if (m_nextAnalyzeCycle >= readyCycles) {
        if (drain && timing != VppKfmTiming::Realtime) {
            finalizeAnalyzerResults(timing);
        }
        return RGY_ERR_NONE;
    }

    const auto *frame = m_sourceCache.empty() ? nullptr : &m_sourceCache.back().frame->frame;
    while (m_nextAnalyzeCycle < readyCycles) {
        std::array<RGYKFM::FMCount, 18> counts = {};
        auto sts = readbackFMCounts(counts, m_nextAnalyzeCycle, drain, queue);
        if (sts == RGY_ERR_MORE_DATA) {
            return RGY_ERR_NONE;
        }
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        writeFMCountDump(counts, m_nextAnalyzeCycle);
        try {
            if (timing == VppKfmTiming::Realtime) {
                const auto result = m_analyzer->realtimeFromCounts(counts.data(), frame->width, frame->height);
                writeAnalyzerResult(result, true);
            } else {
                m_analyzer->analyzeCycleFromCounts(counts.data(), frame->width, frame->height);
                if (timing == VppKfmTiming::RealtimePlus) {
                    const auto resultCount = m_analyzer->results().size();
                    const auto delay = static_cast<size_t>(std::max(0, m_analyzer->param().pastCycles));
                    appendAnalyzerResults(resultCount > delay ? resultCount - delay : 0, true, true);
                }
            }
        } catch (const std::exception& e) {
            AddMessage(RGY_LOG_ERROR, _T("failed to analyze KFM cycle %d: %S.\n"), m_nextAnalyzeCycle, e.what());
            return RGY_ERR_INVALID_CALL;
        }
        m_nextAnalyzeCycle++;
    }
    if (drain && timing != VppKfmTiming::Realtime) {
        finalizeAnalyzerResults(timing);
    }
    return RGY_ERR_NONE;
}

void RGYFilterKfm::finalizeAnalyzerResults(VppKfmTiming timing) {
    if (!m_analyzer || m_analyzerFinalized) {
        return;
    }
    const auto resultCount = static_cast<size_t>(m_nextAnalyzeCycle);
    m_analyzer->analyzeTrailingCycles(m_analyzer->param().cycleRange);
    if (timing == VppKfmTiming::Strict) {
        writeAnalyzerResultsFinal(resultCount, true);
    } else if (timing == VppKfmTiming::RealtimePlus) {
        appendAnalyzerResults(resultCount, true, true);
    } else {
        writeAnalyzerResultsFinal(resultCount, false);
    }
    m_analyzerFinalized = true;
}

std::vector<RGYKFM::KFMResult> RGYFilterKfm::analyzerResultsSnapshot(bool mark60p) const {
    std::vector<RGYKFM::KFMResult> results;
    if (!m_analyzer) {
        return results;
    }
    results = m_analyzer->results();
    if (!mark60p || results.empty()) {
        return results;
    }
    for (auto& result : results) {
        result.is60p = false;
    }
    const auto& param = m_analyzer->param();
    bool is60p = true;
    for (int i = 0; i < static_cast<int>(results.size()); ++i) {
        auto& cur = results[i];
        if (is60p) {
            if (cur.cost < param.th24) {
                if (cur.reliability < param.rel24) {
                    is60p = false;
                }
            } else {
                cur.is60p = true;
            }
        } else {
            if (cur.cost >= param.th60) {
                is60p = true;
                for (int t = i; t >= 0; --t) {
                    auto& prev = results[t];
                    if (prev.cost < param.th24) {
                        if (prev.reliability < param.rel24) {
                            break;
                        }
                    } else {
                        prev.is60p = true;
                    }
                }
            }
        }
    }
    return results;
}

void RGYFilterKfm::writeFMCountDump(const std::array<RGYKFM::FMCount, 18>& counts, int cycle) {
    if (!m_fpFMCount) {
        return;
    }
    const int firstCountFrame = cycle * 5 - 2;
    while (m_nextFMCountDumpFrame < m_cachedSourceFrames) {
        const int windowIndex = m_nextFMCountDumpFrame - firstCountFrame;
        if (windowIndex < 0) {
            m_nextFMCountDumpFrame++;
            continue;
        }
        if (windowIndex >= KFM_FMCOUNT_PAIRS) {
            break;
        }
        fwrite(&counts[windowIndex * 2], sizeof(RGYKFM::FMCount), 2, m_fpFMCount);
        m_nextFMCountDumpFrame++;
    }
    fflush(m_fpFMCount);
}

void RGYFilterKfm::writeAnalyzerResult(const RGYKFM::KFMResult& result, bool dump) {
    m_lastAnalyzeResult = result;
    m_hasLastAnalyzeResult = true;
    m_analyzerOutputResults.push_back(result);
    if (dump && m_fpResult) {
        fwrite(&result, sizeof(result), 1, m_fpResult);
        fflush(m_fpResult);
    }
}

void RGYFilterKfm::appendAnalyzerResults(size_t resultCount, bool dump, bool mark60p) {
    if (!m_analyzer) {
        return;
    }
    const auto results = analyzerResultsSnapshot(mark60p);
    resultCount = std::min(resultCount, results.size());
    if (resultCount <= m_analyzerOutputResults.size()) {
        return;
    }
    while (m_analyzerOutputResults.size() < resultCount) {
        const auto& result = results[m_analyzerOutputResults.size()];
        m_analyzerOutputResults.push_back(result);
        m_lastAnalyzeResult = result;
        m_hasLastAnalyzeResult = true;
        if (dump && m_fpResult) {
            fwrite(&result, sizeof(result), 1, m_fpResult);
        }
    }
    if (dump && m_fpResult) {
        fflush(m_fpResult);
    }
}

void RGYFilterKfm::writeAnalyzerResultsFinal(size_t resultCount, bool mark60p) {
    if (!m_analyzer) {
        return;
    }
    const auto results = analyzerResultsSnapshot(mark60p);
    resultCount = std::min(resultCount, results.size());
    if (resultCount == 0) {
        return;
    }
    if (m_fpResult) {
        if (fseek(m_fpResult, 0, SEEK_SET) != 0) {
            AddMessage(RGY_LOG_WARN, _T("failed to seek KFM result dump file.\n"));
        } else {
            fwrite(results.data(), sizeof(results[0]), resultCount, m_fpResult);
            fflush(m_fpResult);
        }
    }
    m_analyzerOutputResults.assign(results.begin(), results.begin() + resultCount);
    m_lastAnalyzeResult = results[resultCount - 1];
    m_hasLastAnalyzeResult = true;
}

void RGYFilterKfm::writeFrameTimecode(const RGYFrameInfo *frame) {
    if (!m_fpTimecode || !frame || frame->timestamp < 0) {
        return;
    }
    const auto prm = std::dynamic_pointer_cast<RGYFilterParamKfm>(m_param);
    if (!prm || !prm->timebase.is_valid()) {
        return;
    }
    const auto timeMs = (double)frame->timestamp * prm->timebase.qdouble() * 1000.0;
    if (prm->kfm.mode == VppKfmMode::VFR) {
        const auto timeMsInt = (int64_t)std::floor(timeMs + 0.5 - 1.0e-9);
        fprintf(m_fpTimecode, "%lld\n", (lls)timeMsInt);
    } else {
        fprintf(m_fpTimecode, "%.6lf\n", timeMs);
    }
    fflush(m_fpTimecode);
}

std::vector<RGYFilterKfm::KfmSwitchTiming> RGYFilterKfm::deriveSwitchTimings(int total60) const {
    std::vector<KfmSwitchTiming> timings;
    if (!m_analyzer || m_analyzerOutputResults.empty() || total60 <= 0) {
        return timings;
    }
    const auto prm = std::dynamic_pointer_cast<RGYFilterParamKfm>(m_param);
    const auto timingMode = prm ? prm->kfm.timing : VppKfmTiming::Realtime;
    const auto thswitch = prm ? prm->kfm.thswitch : 0.5f;
    const auto resultAt = [&](int cycle) -> const RGYKFM::KFMResult& {
        return m_analyzerOutputResults[clamp(cycle, 0, (int)m_analyzerOutputResults.size() - 1)];
    };
    const auto frameInfoAt = [&](int n60, const RGYKFM::KFMResult& fm) {
        KfmSwitchTiming info;
        info.start60 = n60;
        info.start120 = n60 * 2;
        info.duration60 = 1;
        info.duration120 = 2;
        info.sourceStart = n60;
        info.sourceIndex = n60;
        info.numSourceFrames = 1;
        info.baseType = KFM_FRAME_60;
        info.frame24Index = -1;
        info.isFrame24 = false;
        info.isFrame60 = true;

        const bool force60 = (thswitch >= 0.0f)
            && ((timingMode == VppKfmTiming::Realtime && fm.cost > thswitch)
                || (timingMode != VppKfmTiming::Realtime && fm.is60p != 0));
        if (force60) {
            return info;
        }
        if (RGYKFM::PulldownPatterns::is30p(fm.pattern)) {
            info.baseType = KFM_FRAME_30;
            info.sourceStart = n60 >> 1;
            info.sourceIndex = info.sourceStart;
            info.isFrame60 = false;
            return info;
        }
        const auto f = m_analyzer->patterns().getFrame60(fm.pattern, n60);
        int frameIndex = f.frameIndex + f.fieldShift;
        int n24 = f.cycleIndex * 4 + frameIndex;
        if (frameIndex < 0) {
            // The first 60p slot of some pulldown phases belongs to the last
            // 24p frame of the previous cycle. Fall back only when that frame
            // is before the available source window.
            n24 = f.cycleIndex * 4 - 1;
            if (n24 < 0) {
                info.baseType = KFM_FRAME_60;
                info.sourceStart = n60 >> 1;
                info.sourceIndex = info.sourceStart;
                info.isFrame60 = true;
                return info;
            }
        } else if (frameIndex >= 4) {
            const auto next = m_analyzer->patterns().getFrame24(resultAt(n60 / 10 + 1).pattern, 0);
            n24 = f.cycleIndex * 4 + (next.fieldStartIndex > 0 ? 3 : 4);
        }
        info.baseType = KFM_FRAME_24;
        info.frame24Index = n24;
        info.sourceStart = n24;
        info.sourceIndex = n24;
        info.isFrame24 = false;
        info.isFrame60 = false;
        return info;
    };

    int current = 0;
    while (current < total60) {
        auto info = frameInfoAt(current, resultAt(current / 10));
        const bool forceSingle = (info.baseType == KFM_FRAME_24 || info.baseType == KFM_FRAME_30) && isSwitchSingleFrameN60(current);
        const int maxDuration = forceSingle ? 1 : info.baseType == KFM_FRAME_24 ? 4 : info.baseType == KFM_FRAME_30 ? 2 : 1;
        int duration = maxDuration;
        for (int i = 1; i < maxDuration; i++) {
            if (current + i >= total60) {
                duration = i;
                info.isFrame24 = info.baseType == KFM_FRAME_24;
                break;
            }
            const auto next = frameInfoAt(current + i, resultAt((current + i) / 10));
            if (next.baseType != info.baseType || next.sourceIndex != info.sourceIndex) {
                duration = i;
                info.isFrame24 = info.baseType == KFM_FRAME_24;
                break;
            }
        }
        info.duration60 = duration;
        info.duration120 = duration * 2;
        info.numSourceFrames = std::max(1, divCeil(duration, 2));
        timings.push_back(info);
        current += duration;
    }

    if (prm && prm->kfm.is120) {
        for (size_t i = 1; i < timings.size(); ++i) {
            if (timings[i - 1].isFrame24 && timings[i].isFrame24
                && timings[i - 1].duration60 >= 2 && timings[i].duration60 >= 2
                && timings[i - 1].duration60 + timings[i].duration60 == 5) {
                timings[i].start120 = timings[i - 1].start120 + 5;
            }
        }
        for (size_t i = 0; i + 1 < timings.size(); ++i) {
            const int duration120 = timings[i + 1].start120 - timings[i].start120;
            if (duration120 > 0) {
                timings[i].duration120 = duration120;
            }
        }
    }
    return timings;
}

int64_t RGYFilterKfm::sourceFrameDuration(const KfmCachedSource *source) const {
    if (!source || !source->frame) {
        return 1;
    }
    const auto duration = source->frame->frame.duration;
    if (duration > 0) {
        return duration;
    }
    const auto *next = findSourceByIndexExact(source->sourceIndex + 1);
    if (next && next->timestamp > source->timestamp) {
        return next->timestamp - source->timestamp;
    }
    const auto *prev = findSourceByIndexExact(source->sourceIndex - 1);
    if (prev && source->timestamp > prev->timestamp) {
        return source->timestamp - prev->timestamp;
    }
    return 1;
}

bool RGYFilterKfm::isSwitchSingleFrameN60(int n60) const {
    return std::find(m_switchSingleFrameN60.begin(), m_switchSingleFrameN60.end(), n60) != m_switchSingleFrameN60.end();
}

void RGYFilterKfm::markSwitchSingleFrameN60Range(int start60, int duration60) {
    for (int i = 0; i < duration60; i++) {
        const int n60 = start60 + i;
        if (!isSwitchSingleFrameN60(n60)) {
            m_switchSingleFrameN60.push_back(n60);
        }
    }
}

bool RGYFilterKfm::switchSingleFrameDurationEnabled() const {
    const auto prm = std::dynamic_pointer_cast<RGYFilterParamKfm>(m_param);
    return prm && prm->kfm.thswitch >= 0.0f && !kfmDisableCCDuration();
}

void RGYFilterKfm::writeSwitchTimingDump() {
    if (m_switchTimingDumped || m_switchDurationPath.empty()) {
        return;
    }
    const auto timings = deriveSwitchTimings(m_cachedSourceFrames * 2);
    FILE *fp = nullptr;
    if (_tfopen_s(&fp, m_switchDurationPath.c_str(), _T("wb")) != 0 || fp == nullptr) {
        AddMessage(RGY_LOG_WARN, _T("failed to open KFM duration dump file \"%s\".\n"), m_switchDurationPath.c_str());
        return;
    }
    for (const auto& timing : timings) {
        fprintf(fp, "%d\n", timing.duration60);
    }
    fclose(fp);
    m_switchTimingDumped = true;
    AddMessage(RGY_LOG_DEBUG, _T("wrote KFM switch duration dump file \"%s\".\n"), m_switchDurationPath.c_str());
}

void RGYFilterKfm::writeTelecine24DurationDump() {
    if (m_switchTimingDumped || m_switchDurationPath.empty() || !m_analyzer) {
        return;
    }
    FILE *fp = nullptr;
    if (_tfopen_s(&fp, m_switchDurationPath.c_str(), _T("wb")) != 0 || fp == nullptr) {
        AddMessage(RGY_LOG_WARN, _T("failed to open KFM duration dump file \"%s\".\n"), m_switchDurationPath.c_str());
        return;
    }
    for (int frame24Index = 0; frame24Index < m_nextTelecine24Frame; frame24Index++) {
        if (frame24Index / 4 >= (int)m_analyzerOutputResults.size()) {
            break;
        }
        const auto& result = m_analyzerOutputResults[frame24Index / 4];
        try {
            const auto info = m_analyzer->patterns().getFrame24(result.pattern, frame24Index);
            const int firstField = info.cycleIndex * 10 + info.fieldStartIndex;
            const int totalFields = m_cachedSourceFrames * 2;
            const int availableFields = (totalFields > firstField) ? std::min(info.numFields, totalFields - firstField) : info.numFields;
            fprintf(fp, "%d\n", std::max(1, availableFields));
        } catch (...) {
            fprintf(fp, "2\n");
        }
    }
    fclose(fp);
    m_switchTimingDumped = true;
    AddMessage(RGY_LOG_DEBUG, _T("wrote KFM duration dump file \"%s\".\n"), m_switchDurationPath.c_str());
}

void RGYFilterKfm::writeFrameInfoDump(const char *stage, const RGYFrameInfo *frame, const RGYKFM::KFMResult *result) {
    if (!m_fpFrameInfo || !stage || !frame) {
        return;
    }
    const auto prm = std::dynamic_pointer_cast<RGYFilterParamKfm>(m_param);
    const auto timebase = (prm && prm->timebase.is_valid()) ? prm->timebase.qdouble() : 0.0;
    const auto timeMs = (frame->timestamp >= 0 && timebase > 0.0) ? (double)frame->timestamp * timebase * 1000.0 : -1.0;
    const auto durationMs = (frame->duration >= 0 && timebase > 0.0) ? (double)frame->duration * timebase * 1000.0 : -1.0;
    const auto *r = result ? result : (m_hasLastAnalyzeResult ? &m_lastAnalyzeResult : nullptr);
    fprintf(m_fpFrameInfo, "%s\t%d\t%d\t%lld\t%lld\t%.6lf\t%.6lf\t%d\t%d\t%d\t%d\t%d",
        stage,
        m_timecodeFrameIndex,
        frame->inputFrameId,
        (long long)frame->timestamp,
        (long long)frame->duration,
        timeMs,
        durationMs,
        frame->width,
        frame->height,
        (int)frame->csp,
        (int)frame->picstruct,
        (int)frame->flags);
    if (r) {
        fprintf(m_fpFrameInfo, "\t%d\t%d\t%.6f\t%.6f\t%.6f",
            r->pattern, r->is60p, r->score, r->cost, r->reliability);
    } else {
        fprintf(m_fpFrameInfo, "\t-1\t0\t0.000000\t0.000000\t0.000000");
    }
    auto switchData = std::find_if(frame->dataList.begin(), frame->dataList.end(), [](const std::shared_ptr<RGYFrameData>& data) {
        return data && data->dataType() == RGY_FRAME_DATA_KFM_SWITCH;
    });
    if (switchData != frame->dataList.end()) {
        const auto kfm = std::dynamic_pointer_cast<RGYFrameDataKfmSwitch>(*switchData);
        fprintf(m_fpFrameInfo, "\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.6f\n",
            kfm->n60(), kfm->n24(), kfm->baseType(), kfm->sourceStart(), kfm->numSourceFrames(),
            kfm->duration60(), kfm->duration120(), kfm->pattern(), kfm->cost());
    } else {
        fprintf(m_fpFrameInfo, "\t-1\t-1\t-1\t-1\t0\t0\t0\t-1\t0.000000\n");
    }
    fflush(m_fpFrameInfo);
}

void RGYFilterKfm::writeContainsCombeDump(const char *stage, const KfmSwitchTiming& timing, cl_uint containsCombeCount, bool durationApplied, const RGYKFM::KFMResult *result) {
    if (!m_fpContainsCombe || !stage) {
        return;
    }
    fprintf(m_fpContainsCombe, "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%u\t%d\t%d\t%.6f\n",
        stage,
        m_timecodeFrameIndex,
        timing.start60,
        timing.frame24Index,
        timing.baseType,
        timing.sourceStart,
        timing.numSourceFrames,
        timing.duration60,
        timing.duration120,
        containsCombeCount,
        durationApplied ? 1 : 0,
        result ? result->pattern : -1,
        result ? result->cost : 0.0f);
    fflush(m_fpContainsCombe);
}

const RGYFrameInfo *RGYFilterKfm::findSourceFrame(const RGYFrameInfo *frame, std::vector<RGYOpenCLEvent> *wait_events) {
    if (!frame) {
        return nullptr;
    }
    for (auto it = m_sourceCache.rbegin(); it != m_sourceCache.rend(); ++it) {
        if (it->inputFrameId == frame->inputFrameId && it->timestamp == frame->timestamp) {
            if (wait_events && it->event() != nullptr) {
                wait_events->push_back(it->event);
            }
            return &it->frame->frame;
        }
    }
    for (auto it = m_sourceCache.rbegin(); it != m_sourceCache.rend(); ++it) {
        if (it->inputFrameId == frame->inputFrameId) {
            if (wait_events && it->event() != nullptr) {
                wait_events->push_back(it->event);
            }
            return &it->frame->frame;
        }
    }
    return nullptr;
}

const RGYFilterKfm::KfmCachedSource *RGYFilterKfm::findSourceByIndex(int sourceIndex) const {
    if (m_sourceCache.empty()) {
        return nullptr;
    }
    const int clampedIndex = clamp(sourceIndex, m_sourceCache.front().sourceIndex, m_sourceCache.back().sourceIndex);
    for (auto it = m_sourceCache.rbegin(); it != m_sourceCache.rend(); ++it) {
        if (it->sourceIndex == clampedIndex) {
            return &(*it);
        }
    }
    return nullptr;
}

const RGYFilterKfm::KfmCachedSource *RGYFilterKfm::findSourceByIndexExact(int sourceIndex) const {
    for (auto it = m_sourceCache.rbegin(); it != m_sourceCache.rend(); ++it) {
        if (it->sourceIndex == sourceIndex) {
            return &(*it);
        }
    }
    return nullptr;
}

void RGYFilterKfm::attachSwitchFrameData(RGYFrameInfo *frame, const KfmSwitchTiming& timing, const RGYKFM::KFMResult *result) const {
    if (!frame) {
        return;
    }
    frame->dataList.erase(std::remove_if(frame->dataList.begin(), frame->dataList.end(), [](const std::shared_ptr<RGYFrameData>& data) {
        return data && data->dataType() == RGY_FRAME_DATA_KFM_SWITCH;
    }), frame->dataList.end());
    frame->dataList.push_back(std::make_shared<RGYFrameDataKfmSwitch>(
        timing.start60, timing.frame24Index, timing.baseType, timing.sourceStart, timing.numSourceFrames,
        timing.duration60, timing.duration120, result ? result->pattern : -1, result ? result->cost : 0.0f));
}

int RGYFilterKfm::telecine24FrameCount(bool drain) const {
    if (m_analyzerOutputResults.empty()) {
        return 0;
    }
    const int analyzedFrames = (int)m_analyzerOutputResults.size() * 4;
    const int totalFields = m_cachedSourceFrames * 2;
    int readyFrames = 0;
    for (int frame24Index = 0; frame24Index < analyzedFrames; frame24Index++) {
        const auto& result = m_analyzerOutputResults[frame24Index / 4];
        RGYKFM::Frame24Info info;
        try {
            info = m_analyzer->patterns().getFrame24(result.pattern, frame24Index);
        } catch (...) {
            break;
        }
        const int firstField = info.cycleIndex * 10 + info.fieldStartIndex;
        if (totalFields > 0 && firstField >= totalFields) {
            break;
        }
        if (drain) {
            readyFrames++;
            continue;
        }
        const int lastField = firstField + info.numFields - 1;
        const int firstSource = (firstField & ~1) >> 1;
        const int lastSource = std::max(firstSource, lastField >> 1);
        const auto *first = firstSource < 0 ? findSourceByIndex(firstSource) : findSourceByIndexExact(firstSource);
        const auto *last = lastSource < 0 ? findSourceByIndex(lastSource) : findSourceByIndexExact(lastSource);
        if (!first || !last) {
            break;
        }
        readyFrames++;
    }
    return readyFrames;
}

RGY_ERR RGYFilterKfm::clearStaticFlag(RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!m_staticFlag || !m_programs[KFM_PROG_STATIC].get()) {
        return RGY_ERR_INVALID_CALL;
    }
    const auto planes = RGY_CSP_PLANES[m_staticFlag->frame.csp];
    RGYOpenCLEvent prevEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto plane = getPlane(&m_staticFlag->frame, (RGY_PLANE)iplane);
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
        RGYOpenCLEvent planeEvent;
        RGYWorkSize local(32, 8);
        RGYWorkSize global(plane.width, plane.height);
        auto err = m_programs[KFM_PROG_STATIC].get()->kernel("kernel_kfm_zero").config(queue, local, global, waitHere, &planeEvent).launch(
            (cl_mem)plane.ptr[0], plane.pitch[0],
            plane.width, plane.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_zero (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
        prevEvent = planeEvent;
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::analyzeStaticFlag(RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!m_staticFlag || !m_programs[KFM_PROG_STATIC].get()) {
        return RGY_ERR_INVALID_CALL;
    }
    if (m_sourceCache.empty()) {
        return clearStaticFlag(queue, wait_events, event);
    }
    return analyzeStaticFlag(m_sourceCache.back().sourceIndex, queue, wait_events, event);
}

RGY_ERR RGYFilterKfm::analyzeStaticFlag(int sourceIndex, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!m_staticFlag || !m_programs[KFM_PROG_STATIC].get()) {
        return RGY_ERR_INVALID_CALL;
    }
    const auto *cur = findSourceByIndex(sourceIndex);
    if (!cur || !cur->frame || !cur->frame->frame.ptr[0] || !cur->paddedFrame || !cur->paddedFrame->frame.ptr[0]) {
        return clearStaticFlag(queue, wait_events, event);
    }
    for (auto& frame : m_staticWorkFrames) {
        if (!frame
            || frame->frame.width != m_staticFlag->frame.width
            || frame->frame.height != m_staticFlag->frame.height
            || frame->frame.csp != m_staticFlag->frame.csp) {
            frame = m_cl->createFrameBuffer(m_staticFlag->frame);
            if (!frame) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM static work frame.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
    }

    std::vector<RGYOpenCLEvent> sourceWaitEvents = wait_events;
    for (int offset = -3; offset <= 3; offset++) {
        const auto *src = findSourceByIndex(sourceIndex + offset);
        if (!src || !src->frame || !src->frame->frame.ptr[0]) {
            return clearStaticFlag(queue, wait_events, event);
        }
        if (src->event() != nullptr) {
            sourceWaitEvents.push_back(src->event);
        }
    }
    if (cur->paddedEvent() != nullptr) {
        sourceWaitEvents.push_back(cur->paddedEvent);
    }

    const auto planes = RGY_CSP_PLANES[m_staticFlag->frame.csp];
    const auto runCalcCombe = [&](RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const std::vector<RGYOpenCLEvent>& waits, RGYOpenCLEvent *outEvent) -> RGY_ERR {
        RGYOpenCLEvent prevEvent;
        for (int iplane = 0; iplane < planes; iplane++) {
            const auto dst = getPlane(dstFrame, (RGY_PLANE)iplane);
            const auto src = getPlane(srcFrame, (RGY_PLANE)iplane);
            const int srcYOffset = (src.height - dst.height) >> 1;
            if (src.width != dst.width || src.height != dst.height + srcYOffset * 2 || srcYOffset < 2) {
                AddMessage(RGY_LOG_ERROR, _T("invalid KFM static padded source plane size (plane %d, dst %dx%d, src %dx%d, offset %d).\n"),
                    iplane, dst.width, dst.height, src.width, src.height, srcYOffset);
                return RGY_ERR_INVALID_PARAM;
            }
            const int width4 = dst.width >> 2;
            if (width4 <= 0 || (dst.width & 3) != 0) {
                return RGY_ERR_INVALID_PARAM;
            }
            const auto waitHere = (iplane == 0)
                ? waits
                : (prevEvent() != nullptr ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
            RGYOpenCLEvent planeEvent;
            RGYWorkSize local(32, 8);
            RGYWorkSize global(width4, dst.height);
            auto err = m_programs[KFM_PROG_STATIC].get()->kernel("kernel_kfm_calc_combe").config(queue, local, global, waitHere, &planeEvent).launch(
                (cl_mem)dst.ptr[0], dst.pitch[0],
                (cl_mem)src.ptr[0], src.pitch[0],
                width4, dst.height, srcYOffset);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_calc_combe (plane %d): %s.\n"), iplane, get_err_mes(err));
                return err;
            }
            prevEvent = planeEvent;
        }
        if (outEvent && prevEvent() != nullptr) {
            *outEvent = prevEvent;
        }
        return RGY_ERR_NONE;
    };
    const auto runTemporalMinDiff3 = [&](RGYFrameInfo *dstFrame, int centerIndex, const std::vector<RGYOpenCLEvent>& waits, RGYOpenCLEvent *outEvent) -> RGY_ERR {
        std::array<const KfmCachedSource *, 7> src = {};
        for (int i = 0; i < (int)src.size(); i++) {
            src[i] = findSourceByIndex(centerIndex + i - 3);
            if (!src[i] || !src[i]->frame || !src[i]->frame->frame.ptr[0]) {
                return RGY_ERR_MORE_DATA;
            }
        }
        RGYOpenCLEvent prevEvent;
        for (int iplane = 0; iplane < planes; iplane++) {
            const auto dst = getPlane(dstFrame, (RGY_PLANE)iplane);
            const auto src0 = getPlane(&src[0]->frame->frame, (RGY_PLANE)iplane);
            const auto src1 = getPlane(&src[1]->frame->frame, (RGY_PLANE)iplane);
            const auto src2 = getPlane(&src[2]->frame->frame, (RGY_PLANE)iplane);
            const auto src3 = getPlane(&src[3]->frame->frame, (RGY_PLANE)iplane);
            const auto src4 = getPlane(&src[4]->frame->frame, (RGY_PLANE)iplane);
            const auto src5 = getPlane(&src[5]->frame->frame, (RGY_PLANE)iplane);
            const auto src6 = getPlane(&src[6]->frame->frame, (RGY_PLANE)iplane);
            const int width4 = dst.width >> 2;
            if (width4 <= 0 || (dst.width & 3) != 0
                || src0.width != dst.width || src1.width != dst.width || src2.width != dst.width
                || src3.width != dst.width || src4.width != dst.width || src5.width != dst.width || src6.width != dst.width
                || src0.height != dst.height || src1.height != dst.height || src2.height != dst.height
                || src3.height != dst.height || src4.height != dst.height || src5.height != dst.height || src6.height != dst.height) {
                AddMessage(RGY_LOG_ERROR, _T("invalid KFM temporal min-diff source plane size (plane %d).\n"), iplane);
                return RGY_ERR_INVALID_PARAM;
            }
            const auto waitHere = (iplane == 0)
                ? waits
                : (prevEvent() != nullptr ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
            RGYOpenCLEvent planeEvent;
            RGYWorkSize local(32, 8);
            RGYWorkSize global(width4, dst.height);
            auto err = m_programs[KFM_PROG_STATIC].get()->kernel("kernel_kfm_temporal_min_diff5_3").config(queue, local, global, waitHere, &planeEvent).launch(
                (cl_mem)dst.ptr[0], dst.pitch[0],
                (cl_mem)src0.ptr[0],
                (cl_mem)src1.ptr[0],
                (cl_mem)src2.ptr[0],
                (cl_mem)src3.ptr[0],
                (cl_mem)src4.ptr[0],
                (cl_mem)src5.ptr[0],
                (cl_mem)src6.ptr[0],
                src0.pitch[0],
                width4, dst.height);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_temporal_min_diff5_3 (plane %d): %s.\n"), iplane, get_err_mes(err));
                return err;
            }
            prevEvent = planeEvent;
        }
        if (outEvent && prevEvent() != nullptr) {
            *outEvent = prevEvent;
        }
        return RGY_ERR_NONE;
    };
    const auto runMergeUV = [&](RGYFrameInfo *frame, const std::vector<RGYOpenCLEvent>& waits, RGYOpenCLEvent *outEvent) -> RGY_ERR {
        if (planes < 3) {
            if (outEvent && !waits.empty()) {
                *outEvent = waits.back();
            }
            return RGY_ERR_NONE;
        }
        const auto y = getPlane(frame, RGY_PLANE_Y);
        const auto u = getPlane(frame, RGY_PLANE_U);
        const auto v = getPlane(frame, RGY_PLANE_V);
        RGYOpenCLEvent mergeEvent;
        RGYWorkSize local(32, 8);
        RGYWorkSize global(y.width, y.height);
        auto err = m_programs[KFM_PROG_STATIC].get()->kernel("kernel_kfm_merge_uv_coefs").config(queue, local, global, waits, &mergeEvent).launch(
            (cl_mem)y.ptr[0], y.pitch[0],
            (cl_mem)u.ptr[0], (cl_mem)v.ptr[0], u.pitch[0],
            y.width, y.height, 1, 1);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_merge_uv_coefs: %s.\n"), get_err_mes(err));
            return err;
        }
        if (outEvent && mergeEvent() != nullptr) {
            *outEvent = mergeEvent;
        }
        return RGY_ERR_NONE;
    };
    const auto runExtend = [&](RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const std::vector<RGYOpenCLEvent>& waits, RGYOpenCLEvent *outEvent) -> RGY_ERR {
        const auto dst = getPlane(dstFrame, RGY_PLANE_Y);
        const auto src = getPlane(srcFrame, RGY_PLANE_Y);
        const int width4 = dst.width >> 2;
        RGYOpenCLEvent extendEvent;
        RGYWorkSize local(32, 8);
        RGYWorkSize global(width4, dst.height);
        auto err = m_programs[KFM_PROG_STATIC].get()->kernel("kernel_kfm_extend_coefs").config(queue, local, global, waits, &extendEvent).launch(
            (cl_mem)dst.ptr[0], dst.pitch[0],
            (cl_mem)src.ptr[0], src.pitch[0],
            width4, dst.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_extend_coefs: %s.\n"), get_err_mes(err));
            return err;
        }
        if (outEvent && extendEvent() != nullptr) {
            *outEvent = extendEvent;
        }
        return RGY_ERR_NONE;
    };

    RGYOpenCLEvent combEvent;
    auto sts = runCalcCombe(&m_staticWorkFrames[0]->frame, &cur->paddedFrame->frame, sourceWaitEvents, &combEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    std::vector<RGYOpenCLEvent> phaseWait = (combEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ combEvent } : std::vector<RGYOpenCLEvent>();
    RGYOpenCLEvent mergeUvEvent;
    sts = runMergeUV(&m_staticWorkFrames[0]->frame, phaseWait, &mergeUvEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    phaseWait = (mergeUvEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ mergeUvEvent } : phaseWait;
    RGYOpenCLEvent flagcEvent;
    sts = runExtend(&m_staticFlag->frame, &m_staticWorkFrames[0]->frame, phaseWait, &flagcEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    RGYOpenCLEvent prevEvent;
    sts = runTemporalMinDiff3(&m_staticWorkFrames[4]->frame, sourceIndex, sourceWaitEvents, &prevEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    phaseWait = (prevEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>();
    RGYOpenCLEvent temporalMergeUvEvent;
    sts = runMergeUV(&m_staticWorkFrames[4]->frame, phaseWait, &temporalMergeUvEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    phaseWait = (temporalMergeUvEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ temporalMergeUvEvent } : phaseWait;
    RGYOpenCLEvent flagdEvent;
    sts = runExtend(&m_staticWorkFrames[0]->frame, &m_staticWorkFrames[4]->frame, phaseWait, &flagdEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    const auto flagcY = getPlane(&m_staticFlag->frame, RGY_PLANE_Y);
    const auto flagdY = getPlane(&m_staticWorkFrames[0]->frame, RGY_PLANE_Y);
    std::vector<RGYOpenCLEvent> andWaitEvents;
    if (flagcEvent() != nullptr) {
        andWaitEvents.push_back(flagcEvent);
    }
    if (flagdEvent() != nullptr) {
        andWaitEvents.push_back(flagdEvent);
    }
    RGYOpenCLEvent andEvent;
    RGYWorkSize local(32, 8);
    RGYWorkSize global(flagcY.width >> 2, flagcY.height);
    auto err = m_programs[KFM_PROG_STATIC].get()->kernel("kernel_kfm_and_coefs").config(queue, local, global, andWaitEvents, &andEvent).launch(
        (cl_mem)flagcY.ptr[0], flagcY.pitch[0],
        (cl_mem)flagdY.ptr[0], flagdY.pitch[0],
        flagcY.width >> 2, flagcY.height,
        1.0f / 30.0f, 1.0f / 15.0f);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_and_coefs: %s.\n"), get_err_mes(err));
        return err;
    }

    if (planes >= 3) {
        const auto flagU = getPlane(&m_staticFlag->frame, RGY_PLANE_U);
        const auto flagV = getPlane(&m_staticFlag->frame, RGY_PLANE_V);
        RGYOpenCLEvent applyEvent;
        RGYWorkSize uvGlobal(flagU.width, flagU.height);
        err = m_programs[KFM_PROG_STATIC].get()->kernel("kernel_kfm_apply_uv_coefs_420").config(queue, local, uvGlobal,
            (andEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ andEvent } : std::vector<RGYOpenCLEvent>(), &applyEvent).launch(
            (cl_mem)flagcY.ptr[0], flagcY.pitch[0],
            (cl_mem)flagU.ptr[0], (cl_mem)flagV.ptr[0], flagU.pitch[0],
            flagU.width, flagU.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_apply_uv_coefs_420: %s.\n"), get_err_mes(err));
            return err;
        }
        if (event && applyEvent() != nullptr) {
            *event = applyEvent;
        }
        return RGY_ERR_NONE;
    }
    if (event && andEvent() != nullptr) {
        *event = andEvent;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::mergeStatic(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pDeint60Frame, const RGYFrameInfo *pSourceFrame,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!pOutputFrame || !pDeint60Frame || !pSourceFrame || !m_staticFlag || !m_programs[KFM_PROG_STATIC].get()) {
        return RGY_ERR_INVALID_CALL;
    }

    const auto planes = RGY_CSP_PLANES[pOutputFrame->csp];
    RGYOpenCLEvent prevEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dst = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto src60 = getPlane(pDeint60Frame, (RGY_PLANE)iplane);
        const auto src30 = getPlane(pSourceFrame, (RGY_PLANE)iplane);
        const auto flag = getPlane(&m_staticFlag->frame, (RGY_PLANE)iplane);
        const int width4 = dst.width >> 2;
        if (width4 <= 0 || (dst.width & 3) != 0) {
            AddMessage(RGY_LOG_ERROR, _T("KFM mode=60 requires plane width to be multiple of 4 (plane %d, width %d).\n"), iplane, dst.width);
            return RGY_ERR_INVALID_PARAM;
        }
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
        RGYOpenCLEvent planeEvent;
        RGYWorkSize local(32, 8);
        RGYWorkSize global(width4, dst.height);
        auto err = m_programs[KFM_PROG_STATIC].get()->kernel("kernel_kfm_merge_static").config(queue, local, global, waitHere, &planeEvent).launch(
            (cl_mem)dst.ptr[0], dst.pitch[0],
            (cl_mem)src60.ptr[0],
            (cl_mem)src30.ptr[0], src30.pitch[0],
            (cl_mem)flag.ptr[0], flag.pitch[0],
            width4, dst.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_merge_static (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
        prevEvent = planeEvent;
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    copyFramePropWithoutRes(pOutputFrame, pDeint60Frame);
    pOutputFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pOutputFrame->flags = RGY_FRAME_FLAG_NONE;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::renderTelecine24(RGYFrameInfo *pOutputFrame, int frame24Index, bool drain, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!pOutputFrame || !m_analyzer || !m_programs[KFM_PROG_RENDER].get()) {
        return RGY_ERR_INVALID_CALL;
    }
    if (frame24Index < 0 || frame24Index / 4 >= (int)m_analyzerOutputResults.size()) {
        return RGY_ERR_MORE_DATA;
    }
    const auto& result = m_analyzerOutputResults[frame24Index / 4];
    RGYKFM::Frame24Info info;
    try {
        info = m_analyzer->patterns().getFrame24(result.pattern, frame24Index);
    } catch (const std::exception& e) {
        AddMessage(RGY_LOG_ERROR, _T("failed to resolve KFM 24p frame %d: %S.\n"), frame24Index, e.what());
        return RGY_ERR_INVALID_CALL;
    }
    const int firstField = info.cycleIndex * 10 + info.fieldStartIndex;
    const int firstSource = (firstField & ~1) >> 1;
    const int paritySourceIndex = (frame24Index / 4) * 5;
    const auto *paritySource = (drain || paritySourceIndex < 0) ? findSourceByIndex(paritySourceIndex) : findSourceByIndexExact(paritySourceIndex);
    if (!paritySource || !paritySource->frame || !paritySource->frame->frame.ptr[0]) {
        return RGY_ERR_MORE_DATA;
    }
    std::array<const KfmCachedSource *, 3> src = {};
    for (int i = 0; i < (int)src.size(); i++) {
        const int sourceIndex = firstSource + i;
        src[i] = (drain || sourceIndex < 0) ? findSourceByIndex(sourceIndex) : findSourceByIndexExact(sourceIndex);
        if (!src[i] || !src[i]->paddedFrame || !src[i]->paddedFrame->frame.ptr[0]) {
            return RGY_ERR_MORE_DATA;
        }
    }

    std::vector<RGYOpenCLEvent> sourceWaitEvents = wait_events;
    for (const auto *s : src) {
        if (s->paddedEvent() != nullptr) {
            sourceWaitEvents.push_back(s->paddedEvent);
        }
    }

    RGYOpenCLEvent prevEvent;
    const auto planes = RGY_CSP_PLANES[pOutputFrame->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dst = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto src0 = getPlane(&src[0]->paddedFrame->frame, (RGY_PLANE)iplane);
        const auto src1 = getPlane(&src[1]->paddedFrame->frame, (RGY_PLANE)iplane);
        const auto src2 = getPlane(&src[2]->paddedFrame->frame, (RGY_PLANE)iplane);
        const int srcYOffset = (src0.height - dst.height) >> 1;
        if (src0.width != dst.width || src1.width != dst.width || src2.width != dst.width
            || src0.height != src1.height || src0.height != src2.height
            || src0.height != dst.height + srcYOffset * 2 || srcYOffset < 0) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM 24p padded source plane size (plane %d, dst %dx%d, src %dx%d/%dx%d/%dx%d, offset %d).\n"),
                iplane, dst.width, dst.height, src0.width, src0.height, src1.width, src1.height, src2.width, src2.height, srcYOffset);
            return RGY_ERR_INVALID_PARAM;
        }
        const auto waitHere = (iplane == 0)
            ? sourceWaitEvents
            : (prevEvent() != nullptr ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
        RGYOpenCLEvent planeEvent;
        RGYWorkSize local(32, 8);
        RGYWorkSize global(dst.width, dst.height);
        auto err = m_programs[KFM_PROG_RENDER].get()->kernel("kernel_kfm_telecine_weave").config(queue, local, global, waitHere, &planeEvent).launch(
            (cl_mem)dst.ptr[0], dst.pitch[0],
            (cl_mem)src0.ptr[0], src0.pitch[0],
            (cl_mem)src1.ptr[0], src1.pitch[0],
            (cl_mem)src2.ptr[0], src2.pitch[0],
            dst.width, dst.height,
            srcYOffset,
            firstField, info.numFields, kfmFrameParity(&paritySource->frame->frame));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_telecine_weave (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
        prevEvent = planeEvent;
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    copyFramePropWithoutRes(pOutputFrame, &src[0]->frame->frame);
    pOutputFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pOutputFrame->flags = RGY_FRAME_FLAG_NONE;
    if (pOutputFrame->duration > 0) {
        pOutputFrame->duration = std::max<int64_t>(1, (pOutputFrame->duration * 5 + 2) / 4);
    }
    if (m_nextTelecine24Frame == frame24Index) {
        pOutputFrame->timestamp = m_nextTelecine24Pts;
    }
    KfmSwitchTiming timing;
    timing.start60 = firstField;
    timing.start120 = firstField * 2;
    timing.sourceIndex = src[0]->sourceIndex;
    timing.frame24Index = frame24Index;
    timing.baseType = KFM_FRAME_24;
    timing.sourceStart = firstSource;
    timing.numSourceFrames = 3;
    const int totalFields = m_cachedSourceFrames * 2;
    const int availableFields = (totalFields > firstField) ? std::min(info.numFields, totalFields - firstField) : info.numFields;
    timing.duration60 = std::max(1, availableFields);
    timing.duration120 = timing.duration60 * 2;
    timing.isFrame24 = true;
    timing.isFrame60 = false;
    attachSwitchFrameData(pOutputFrame, timing, &result);
    writeFrameInfoDump("deint24", pOutputFrame, &result);
    const auto dumpSts = dumpStageFrame("deint24", pOutputFrame, frame24Index, queue,
        (prevEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
    if (dumpSts != RGY_ERR_NONE) {
        return dumpSts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::renderDoubleWeaveFrame(RGYFrameInfo *pOutputFrame, int firstField, int fieldCount, bool drain, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!pOutputFrame || !m_programs[KFM_PROG_RENDER].get() || firstField < 0 || fieldCount <= 0 || fieldCount > 6) {
        return RGY_ERR_INVALID_CALL;
    }

    const int fieldBase = firstField & ~1;
    const int firstSource = fieldBase >> 1;
    const int lastField = firstField + fieldCount - 1;
    const int lastSourceOffset = std::max(0, ((lastField & ~1) - fieldBase) >> 1);

    auto sourceAt = [this, drain](const int sourceIndex) -> const KfmCachedSource * {
        return drain ? findSourceByIndex(sourceIndex) : findSourceByIndexExact(sourceIndex);
    };

    std::array<const KfmCachedSource *, 3> src = {};
    for (int i = 0; i < (int)src.size(); i++) {
        const int sourceIndex = firstSource + std::min(i, lastSourceOffset);
        src[i] = sourceAt(sourceIndex);
        if (!src[i] || !src[i]->paddedFrame || !src[i]->paddedFrame->frame.ptr[0]) {
            return RGY_ERR_MORE_DATA;
        }
    }

    std::vector<RGYOpenCLEvent> sourceWaitEvents = wait_events;
    for (const auto *s : src) {
        if (s->paddedEvent() != nullptr) {
            sourceWaitEvents.push_back(s->paddedEvent);
        }
    }

    RGYOpenCLEvent prevEvent;
    const auto planes = RGY_CSP_PLANES[pOutputFrame->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dst = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto src0 = getPlane(&src[0]->paddedFrame->frame, (RGY_PLANE)iplane);
        const auto src1 = getPlane(&src[1]->paddedFrame->frame, (RGY_PLANE)iplane);
        const auto src2 = getPlane(&src[2]->paddedFrame->frame, (RGY_PLANE)iplane);
        const int srcYOffset = (src0.height - dst.height) >> 1;
        if (src0.width != dst.width || src1.width != dst.width || src2.width != dst.width
            || src0.height != src1.height || src0.height != src2.height
            || src0.height != dst.height + srcYOffset * 2 || srcYOffset < 0) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM UCF24 DoubleWeave padded source plane size (plane %d, dst %dx%d, src %dx%d/%dx%d/%dx%d, offset %d).\n"),
                iplane, dst.width, dst.height, src0.width, src0.height, src1.width, src1.height, src2.width, src2.height, srcYOffset);
            return RGY_ERR_INVALID_PARAM;
        }
        const auto waitHere = (iplane == 0)
            ? sourceWaitEvents
            : (prevEvent() != nullptr ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
        RGYOpenCLEvent planeEvent;
        RGYWorkSize local(32, 8);
        RGYWorkSize global(dst.width, dst.height);
        auto err = m_programs[KFM_PROG_RENDER].get()->kernel("kernel_kfm_telecine_weave").config(queue, local, global, waitHere, &planeEvent).launch(
            (cl_mem)dst.ptr[0], dst.pitch[0],
            (cl_mem)src0.ptr[0], src0.pitch[0],
            (cl_mem)src1.ptr[0], src1.pitch[0],
            (cl_mem)src2.ptr[0], src2.pitch[0],
            dst.width, dst.height,
            srcYOffset,
            firstField, fieldCount, kfmFrameParity(&src[0]->frame->frame));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_telecine_weave (UCF24 DoubleWeave, plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
        prevEvent = planeEvent;
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    copyFramePropWithoutRes(pOutputFrame, &src[0]->frame->frame);
    pOutputFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pOutputFrame->flags = RGY_FRAME_FLAG_NONE;
    KfmSwitchTiming timing;
    timing.start60 = firstField;
    timing.start120 = firstField * 2;
    timing.sourceIndex = firstSource;
    timing.frame24Index = firstField;
    timing.baseType = KFM_FRAME_UCF;
    timing.sourceStart = firstSource;
    timing.numSourceFrames = lastSourceOffset + 1;
    timing.duration60 = fieldCount;
    timing.duration120 = fieldCount * 2;
    timing.isFrame24 = false;
    timing.isFrame60 = false;
    attachSwitchFrameData(pOutputFrame, timing, nullptr);
    writeFrameInfoDump("ucf24-dweave", pOutputFrame);
    const auto dumpSts = dumpStageFrame("ucf24-dweave", pOutputFrame, firstField, queue,
        (prevEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
    if (dumpSts != RGY_ERR_NONE) {
        return dumpSts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::renderCleanSuperFields(RGYFrameInfo *pOutputFrame, int firstSuperField, int lastSuperField, int propSourceIndex, int outputFrameId, bool drain, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!pOutputFrame || !m_programs[KFM_PROG_ANALYZE].get() || !m_programs[KFM_PROG_RENDER].get()) {
        return RGY_ERR_INVALID_CALL;
    }
    if (firstSuperField > lastSuperField) {
        return RGY_ERR_INVALID_PARAM;
    }

    const bool interleavedUV = kfmCspHasInterleavedUV(pOutputFrame->csp);
    const int targetPlanes = (RGY_CSP_PLANES[pOutputFrame->csp] >= 3) ? 3 : (interleavedUV ? 3 : 1);
    const cl_uchar zero = 0;
    RGYOpenCLEvent prevEvent;

    auto sourceAt = [this, drain](const int sourceIndex) -> const KfmCachedSource * {
        return (drain || sourceIndex < 0) ? findSourceByIndex(sourceIndex) : findSourceByIndexExact(sourceIndex);
    };

    auto renderRawSuper = [&](const int bufIndex, const int sourceIndex, const RGY_PLANE plane, const int pixelStep, const int pixelOffset,
                              const int rawPitch, const int gridWidth, const int gridHeight,
                              const std::vector<RGYOpenCLEvent>& baseWaitEvents, RGYOpenCLEvent *rawEvent) -> RGY_ERR {
        const auto *src0 = sourceAt(sourceIndex);
        const auto *src1 = sourceAt(sourceIndex + 1);
        if (!src0 || !src0->frame || !src0->frame->frame.ptr[0]
            || !src1 || !src1->frame || !src1->frame->frame.ptr[0]) {
            return RGY_ERR_MORE_DATA;
        }
        const auto src0Plane = getPlane(&src0->frame->frame, plane);
        const auto src1Plane = getPlane(&src1->frame->frame, plane);
        if (!src0Plane.ptr[0] || !src1Plane.ptr[0]) {
            return RGY_ERR_MORE_DATA;
        }
        if (src0Plane.width != src1Plane.width || src0Plane.height != src1Plane.height) {
            return RGY_ERR_INVALID_CALL;
        }
        const int srcLogicalWidth = src0Plane.width / pixelStep;
        const int srcGridWidth = srcLogicalWidth >> 2;
        const int srcGridHeight = src0Plane.height >> 2;
        if (srcLogicalWidth <= 0 || (srcLogicalWidth & 3) != 0 || srcGridWidth != gridWidth || srcGridHeight != gridHeight) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM telecine-super raw source size (source %d, plane %d, logical %dx%d, grid %dx%d, expected %dx%d).\n"),
                sourceIndex, (int)plane, srcLogicalWidth, src0Plane.height, srcGridWidth, srcGridHeight, gridWidth, gridHeight);
            return RGY_ERR_INVALID_PARAM;
        }

        const size_t rawBytes = (size_t)rawPitch * gridHeight * 2;
        if (!m_telecineSuperRaw[bufIndex] || m_telecineSuperRaw[bufIndex]->size() < rawBytes) {
            m_telecineSuperRaw[bufIndex] = m_cl->createBuffer(rawBytes, CL_MEM_READ_WRITE);
            if (!m_telecineSuperRaw[bufIndex]) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM telecine-super raw buffer %d.\n"), bufIndex);
                return RGY_ERR_MEMORY_ALLOC;
            }
        }

        std::vector<RGYOpenCLEvent> clearWaitEvents = baseWaitEvents;
        if (src0->event() != nullptr) {
            clearWaitEvents.push_back(src0->event);
        }
        if (src1->event() != nullptr) {
            clearWaitEvents.push_back(src1->event);
        }

        RGYOpenCLEvent clearEvent;
        auto sts = m_cl->setBuf(&zero, sizeof(zero), rawBytes, m_telecineSuperRaw[bufIndex].get(), queue, clearWaitEvents, &clearEvent);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to clear KFM telecine-super raw buffer %d: %s.\n"), bufIndex, get_err_mes(sts));
            return sts;
        }

        RGYWorkSize local(32, 8);
        RGYWorkSize global(gridWidth, gridHeight);
        sts = m_programs[KFM_PROG_ANALYZE].get()->kernel("kernel_kfm_analyze").config(queue, local, global, { clearEvent }, rawEvent).launch(
            (cl_mem)m_telecineSuperRaw[bufIndex]->mem(), rawPitch,
            (cl_mem)src0Plane.ptr[0],
            (cl_mem)src1Plane.ptr[0], src0Plane.pitch[0],
            gridWidth, gridHeight, kfmFrameParity(&src0->frame->frame),
            pixelStep, pixelOffset);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_analyze (telecine-super raw, source %d): %s.\n"), sourceIndex, get_err_mes(sts));
            return sts;
        }
        return RGY_ERR_NONE;
    };

    auto getSuperSourcePlanes = [&](const int sourceIndex, const RGY_PLANE plane, const int pixelStep,
                                    const int gridWidth, const int gridHeight,
                                    RGYFrameInfo *src0Plane, RGYFrameInfo *src1Plane,
                                    const KfmCachedSource **src0, const KfmCachedSource **src1) -> RGY_ERR {
        *src0 = sourceAt(sourceIndex);
        *src1 = sourceAt(sourceIndex + 1);
        if (!*src0 || !(*src0)->frame || !(*src0)->frame->frame.ptr[0]
            || !*src1 || !(*src1)->frame || !(*src1)->frame->frame.ptr[0]) {
            return RGY_ERR_MORE_DATA;
        }
        *src0Plane = getPlane(&(*src0)->frame->frame, plane);
        *src1Plane = getPlane(&(*src1)->frame->frame, plane);
        if (!src0Plane->ptr[0] || !src1Plane->ptr[0]) {
            return RGY_ERR_MORE_DATA;
        }
        if (src0Plane->width != src1Plane->width || src0Plane->height != src1Plane->height) {
            return RGY_ERR_INVALID_CALL;
        }
        const int srcLogicalWidth = src0Plane->width / pixelStep;
        const int srcGridWidth = srcLogicalWidth >> 2;
        const int srcGridHeight = src0Plane->height >> 2;
        if (srcLogicalWidth <= 0 || (srcLogicalWidth & 3) != 0 || srcGridWidth != gridWidth || srcGridHeight != gridHeight) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM telecine-super source size (source %d, plane %d, logical %dx%d, grid %dx%d, expected %dx%d).\n"),
                sourceIndex, (int)plane, srcLogicalWidth, src0Plane->height, srcGridWidth, srcGridHeight, gridWidth, gridHeight);
            return RGY_ERR_INVALID_PARAM;
        }
        return RGY_ERR_NONE;
    };

    auto renderCleanSuperFused = [&](const int prevSourceIndex, const int curSourceIndex, const RGY_PLANE plane, const int pixelStep, const int pixelOffset,
                                     const RGYFrameInfo& dst, const int widthPairs, const int logicalHeight, const int field,
                                     const int cleanThresh, const int maxMode,
                                     const int dstStep, const int dstOffset,
                                     const std::vector<RGYOpenCLEvent>& baseWaitEvents, RGYOpenCLEvent *cleanEvent) -> RGY_ERR {
        RGYFrameInfo prevSrc0Plane, prevSrc1Plane, curSrc0Plane, curSrc1Plane;
        const KfmCachedSource *prevSrc0 = nullptr, *prevSrc1 = nullptr, *curSrc0 = nullptr, *curSrc1 = nullptr;
        auto sts = getSuperSourcePlanes(prevSourceIndex, plane, pixelStep, widthPairs, logicalHeight,
            &prevSrc0Plane, &prevSrc1Plane, &prevSrc0, &prevSrc1);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = getSuperSourcePlanes(curSourceIndex, plane, pixelStep, widthPairs, logicalHeight,
            &curSrc0Plane, &curSrc1Plane, &curSrc0, &curSrc1);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }

        std::vector<RGYOpenCLEvent> cleanWaitEvents = baseWaitEvents;
        for (const auto *src : { prevSrc0, prevSrc1, curSrc0, curSrc1 }) {
            if (src && src->event() != nullptr) {
                cleanWaitEvents.push_back(src->event);
            }
        }

        RGYWorkSize cleanLocal(32, 8);
        RGYWorkSize cleanGlobal(widthPairs, logicalHeight);
        sts = m_programs[KFM_PROG_RENDER].get()->kernel("kernel_kfm_clean_super_direct_max").config(queue, cleanLocal, cleanGlobal, cleanWaitEvents, cleanEvent).launch(
            (cl_mem)dst.ptr[0], dst.pitch[0],
            (cl_mem)prevSrc0Plane.ptr[0],
            (cl_mem)prevSrc1Plane.ptr[0], prevSrc0Plane.pitch[0], kfmFrameParity(&prevSrc0->frame->frame),
            (cl_mem)curSrc0Plane.ptr[0],
            (cl_mem)curSrc1Plane.ptr[0], curSrc0Plane.pitch[0], kfmFrameParity(&curSrc0->frame->frame),
            widthPairs, logicalHeight,
            field & 1,
            cleanThresh,
            maxMode,
            dstStep, dstOffset,
            pixelStep, pixelOffset);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_clean_super_direct_max (field %d, plane %d): %s.\n"), field, (int)plane, get_err_mes(sts));
            return sts;
        }
        return RGY_ERR_NONE;
    };

    const bool useFusedCleanSuper = kfmUseFusedCleanSuper();

    for (int iplane = 0; iplane < targetPlanes; iplane++) {
        const bool interleavedChroma = interleavedUV && iplane > 0;
        const auto plane = interleavedChroma ? RGY_PLANE_U : (RGY_PLANE)iplane;
        const auto dst = getPlane(pOutputFrame, plane);
        if (!dst.ptr[0]) {
            continue;
        }
        const int dstStep = interleavedChroma ? 2 : 1;
        const int dstOffset = interleavedChroma ? iplane - 1 : 0;
        const int logicalWidth = dst.width / dstStep;
        const int logicalHeight = dst.height;
        const int widthPairs = logicalWidth >> 1;
        if (logicalWidth <= 0 || logicalHeight <= 0 || (logicalWidth & 1) != 0) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM telecine-super output plane size (plane %d, logical %dx%d).\n"),
                iplane, logicalWidth, logicalHeight);
            return RGY_ERR_INVALID_PARAM;
        }

        const int rawPitch = widthPairs * (int)sizeof(cl_uchar2);
        const int gridHeight = logicalHeight;
        const int cleanThresh = (iplane > 0) ? KFM_CLEAN_THRESH_C : KFM_CLEAN_THRESH_Y;
        const int pixelStep = interleavedChroma ? 2 : 1;
        const int pixelOffset = interleavedChroma ? iplane - 1 : 0;
        bool firstWrite = true;
        for (int field = firstSuperField; field <= lastSuperField; field++) {
            const int curSourceIndex = kfmFloorDiv2(field);
            const int prevSourceIndex = kfmFloorDiv2(field - 1);
            std::vector<RGYOpenCLEvent> fieldWaitEvents = wait_events;
            if (prevEvent() != nullptr) {
                fieldWaitEvents.push_back(prevEvent);
            }

            RGYOpenCLEvent cleanEvent;
            auto sts = RGY_ERR_NONE;
            if (useFusedCleanSuper) {
                sts = renderCleanSuperFused(prevSourceIndex, curSourceIndex, plane, pixelStep, pixelOffset, dst,
                    widthPairs, logicalHeight, field, cleanThresh, firstWrite ? 0 : 1, dstStep, dstOffset, fieldWaitEvents, &cleanEvent);
            } else {
                RGYOpenCLEvent rawPrevEvent;
                sts = renderRawSuper(0, prevSourceIndex, plane, pixelStep, pixelOffset, rawPitch, widthPairs, gridHeight, fieldWaitEvents, &rawPrevEvent);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                RGYOpenCLEvent rawCurEvent;
                sts = renderRawSuper(1, curSourceIndex, plane, pixelStep, pixelOffset, rawPitch, widthPairs, gridHeight, fieldWaitEvents, &rawCurEvent);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }

                RGYWorkSize cleanLocal(32, 8);
                RGYWorkSize cleanGlobal(widthPairs, logicalHeight);
                sts = m_programs[KFM_PROG_RENDER].get()->kernel("kernel_kfm_clean_separated_super_max").config(queue, cleanLocal, cleanGlobal, { rawPrevEvent, rawCurEvent }, &cleanEvent).launch(
                    (cl_mem)dst.ptr[0], dst.pitch[0],
                    (cl_mem)m_telecineSuperRaw[0]->mem(),
                    (cl_mem)m_telecineSuperRaw[1]->mem(),
                    rawPitch,
                    widthPairs, logicalHeight,
                    field & 1,
                    cleanThresh,
                    firstWrite ? 0 : 1,
                    dstStep, dstOffset);
            }
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at KFM clean super render (field %d, plane %d): %s.\n"), field, iplane, get_err_mes(sts));
                return sts;
            }
            prevEvent = cleanEvent;
            firstWrite = false;
        }
    }

    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    const auto *propSource = sourceAt(propSourceIndex);
    if (propSource && propSource->frame) {
        copyFramePropWithoutRes(pOutputFrame, &propSource->frame->frame);
    }
    pOutputFrame->inputFrameId = outputFrameId;
    pOutputFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pOutputFrame->flags = RGY_FRAME_FLAG_NONE;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::renderTelecineSuper24(RGYFrameInfo *pOutputFrame, int frame24Index, bool drain, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!pOutputFrame || !m_analyzer || !m_programs[KFM_PROG_ANALYZE].get() || !m_programs[KFM_PROG_RENDER].get()) {
        return RGY_ERR_INVALID_CALL;
    }
    if (frame24Index < 0 || frame24Index / 4 >= (int)m_analyzerOutputResults.size()) {
        return RGY_ERR_MORE_DATA;
    }
    const auto& result = m_analyzerOutputResults[frame24Index / 4];
    RGYKFM::Frame24Info info;
    try {
        info = m_analyzer->patterns().getFrame24(result.pattern, frame24Index);
    } catch (const std::exception& e) {
        AddMessage(RGY_LOG_ERROR, _T("failed to resolve KFM 24p super frame %d: %S.\n"), frame24Index, e.what());
        return RGY_ERR_INVALID_CALL;
    }
    const int firstField = info.cycleIndex * 10 + info.fieldStartIndex;
    const int firstSource = (firstField & ~1) >> 1;
    const int lastField = firstField + info.numFields - 2;
    RGYOpenCLEvent superEvent;
    auto sts = renderCleanSuperFields(pOutputFrame, firstField, lastField, firstSource, frame24Index, drain, queue, wait_events, &superEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (event && superEvent() != nullptr) {
        *event = superEvent;
    }
    pOutputFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pOutputFrame->flags = RGY_FRAME_FLAG_NONE;
    writeFrameInfoDump("telecine-super", pOutputFrame, &result);
    const auto dumpSts = dumpStageFrame("telecine-super", pOutputFrame, frame24Index, queue,
        (superEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ superEvent } : std::vector<RGYOpenCLEvent>());
    if (dumpSts != RGY_ERR_NONE) {
        return dumpSts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::renderSuper30(RGYFrameInfo *pOutputFrame, int frame30Index, bool drain, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!pOutputFrame || !m_programs[KFM_PROG_ANALYZE].get() || !m_programs[KFM_PROG_RENDER].get()) {
        return RGY_ERR_INVALID_CALL;
    }
    RGYOpenCLEvent superEvent;
    const int field = frame30Index * 2;
    auto sts = renderCleanSuperFields(pOutputFrame, field, field, frame30Index, frame30Index, drain, queue, wait_events, &superEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (event && superEvent() != nullptr) {
        *event = superEvent;
    }
    pOutputFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pOutputFrame->flags = RGY_FRAME_FLAG_NONE;
    writeFrameInfoDump("super30", pOutputFrame);
    const auto dumpSts = dumpStageFrame("super30", pOutputFrame, frame30Index, queue,
        (superEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ superEvent } : std::vector<RGYOpenCLEvent>());
    if (dumpSts != RGY_ERR_NONE) {
        return dumpSts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::removeCombeFields(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pDeintFrame, const RGYFrameInfo *pTelecineSuperFrame,
    int firstField, int fieldCount, int stageFrameIndex, const char *stageName, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!pOutputFrame || !pDeintFrame || !pTelecineSuperFrame || !m_programs[KFM_PROG_RENDER].get()
        || fieldCount <= 0 || fieldCount > 6 || !stageName) {
        return RGY_ERR_INVALID_CALL;
    }
    const int fieldBase = firstField & ~1;
    const int firstSource = kfmFloorDiv2(fieldBase);
    const int lastField = firstField + fieldCount - 1;
    const int lastSourceOffset = std::max(0, ((lastField & ~1) - fieldBase) >> 1);
    const auto *paritySource = findSourceByIndex(firstSource);
    if (!paritySource || !paritySource->frame || !paritySource->frame->frame.ptr[0]) {
        return RGY_ERR_MORE_DATA;
    }
    std::array<const KfmCachedSource *, 3> teleSrc = {};
    for (int i = 0; i < (int)teleSrc.size(); i++) {
        const int sourceIndex = firstSource + std::min(i, lastSourceOffset);
        teleSrc[i] = findSourceByIndex(sourceIndex);
        if (!teleSrc[i] || !teleSrc[i]->paddedFrame || !teleSrc[i]->paddedFrame->frame.ptr[0]) {
            return RGY_ERR_MORE_DATA;
        }
    }
    std::vector<RGYOpenCLEvent> sourceWaitEvents = wait_events;
    for (const auto *s : teleSrc) {
        if (s->paddedEvent() != nullptr) {
            sourceWaitEvents.push_back(s->paddedEvent);
        }
    }

    RGYOpenCLEvent prevEvent;
    const auto planes = RGY_CSP_PLANES[pOutputFrame->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dst = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto src = getPlane(pDeintFrame, (RGY_PLANE)iplane);
        const auto combe = getPlane(pTelecineSuperFrame, (RGY_PLANE)iplane);
        const auto telePlane0 = getPlane(&teleSrc[0]->paddedFrame->frame, (RGY_PLANE)iplane);
        const auto telePlane1 = getPlane(&teleSrc[1]->paddedFrame->frame, (RGY_PLANE)iplane);
        const auto telePlane2 = getPlane(&teleSrc[2]->paddedFrame->frame, (RGY_PLANE)iplane);
        const int teleSrcYOffset = (telePlane0.height - dst.height) >> 1;
        if (telePlane0.width != dst.width || telePlane1.width != dst.width || telePlane2.width != dst.width
            || telePlane0.height != telePlane1.height || telePlane0.height != telePlane2.height
            || telePlane0.height != dst.height + teleSrcYOffset * 2 || teleSrcYOffset < 0) {
            AddMessage(RGY_LOG_ERROR, _T("invalid padded source plane size (plane %d, dst %dx%d, src %dx%d/%dx%d/%dx%d, offset %d).\n"),
                iplane, dst.width, dst.height, telePlane0.width, telePlane0.height, telePlane1.width, telePlane1.height, telePlane2.width, telePlane2.height, teleSrcYOffset);
            return RGY_ERR_INVALID_PARAM;
        }
        const auto waitHere = (iplane == 0)
            ? sourceWaitEvents
            : (prevEvent() != nullptr ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
        RGYOpenCLEvent planeEvent;
        RGYWorkSize local(32, 8);
        RGYWorkSize global(dst.width, dst.height);
        const bool chroma = iplane > 0;
        const int threshold = (chroma ? KFM_REMOVE_COMBE_THRESH_C : KFM_REMOVE_COMBE_THRESH_Y) * kfmDepthScale(dst.csp);
        auto err = m_programs[KFM_PROG_RENDER].get()->kernel("kernel_kfm_remove_combe_binomial").config(queue, local, global, waitHere, &planeEvent).launch(
            (cl_mem)dst.ptr[0], dst.pitch[0],
            (cl_mem)src.ptr[0], src.pitch[0],
            (cl_mem)combe.ptr[0], combe.pitch[0],
            (cl_mem)telePlane0.ptr[0], telePlane0.pitch[0],
            (cl_mem)telePlane1.ptr[0], telePlane1.pitch[0],
            (cl_mem)telePlane2.ptr[0], telePlane2.pitch[0],
            dst.width, dst.height, threshold,
            1, 0, 1, 0,
            teleSrcYOffset, firstField, fieldCount, kfmFrameParity(&paritySource->frame->frame));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_remove_combe_binomial (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
        prevEvent = planeEvent;
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    copyFramePropWithoutRes(pOutputFrame, pDeintFrame);
    pOutputFrame->dataList = pDeintFrame->dataList;
    pOutputFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pOutputFrame->flags = RGY_FRAME_FLAG_NONE;
    writeFrameInfoDump(stageName, pOutputFrame);
    const auto dumpSts = dumpStageFrame(stageName, pOutputFrame, stageFrameIndex, queue,
        (prevEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
    if (dumpSts != RGY_ERR_NONE) {
        return dumpSts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::removeCombe24(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pDeint24Frame, const RGYFrameInfo *pTelecineSuperFrame, int frame24Index, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!pOutputFrame || !pDeint24Frame || !pTelecineSuperFrame || !m_analyzer) {
        return RGY_ERR_INVALID_CALL;
    }
    if (frame24Index < 0 || frame24Index / 4 >= (int)m_analyzerOutputResults.size()) {
        return RGY_ERR_MORE_DATA;
    }
    const auto& result = m_analyzerOutputResults[frame24Index / 4];
    RGYKFM::Frame24Info info;
    try {
        info = m_analyzer->patterns().getFrame24(result.pattern, frame24Index);
    } catch (const std::exception& e) {
        AddMessage(RGY_LOG_ERROR, _T("failed to resolve KFM 24p frame %d: %S.\n"), frame24Index, e.what());
        return RGY_ERR_INVALID_CALL;
    }
    const int firstField = info.cycleIndex * 10 + info.fieldStartIndex;
    return removeCombeFields(pOutputFrame, pDeint24Frame, pTelecineSuperFrame,
        firstField, info.numFields, frame24Index, "remove-combe", queue, wait_events, event);
}

RGY_ERR RGYFilterKfm::patchCombe(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pBaseFrame, const RGYFrameInfo *pPatchFrame, const RGYFrameInfo *pMaskFrame, int frameIndex, const char *stageName, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!pOutputFrame || !pBaseFrame || !pPatchFrame || !pMaskFrame || !m_programs[KFM_PROG_SWITCH].get()) {
        return RGY_ERR_INVALID_CALL;
    }
    RGYOpenCLEvent prevEvent;
    const auto planes = RGY_CSP_PLANES[pOutputFrame->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dst = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto base = getPlane(pBaseFrame, (RGY_PLANE)iplane);
        const auto patch = getPlane(pPatchFrame, (RGY_PLANE)iplane);
        const auto mask = getPlane(pMaskFrame, (RGY_PLANE)iplane);
        if (dst.width != base.width || dst.height != base.height
            || dst.width != patch.width || dst.height != patch.height
            || dst.width != mask.width || dst.height != mask.height) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM patch-combe plane size (plane %d).\n"), iplane);
            return RGY_ERR_INVALID_PARAM;
        }
        const auto waitHere = (iplane == 0)
            ? wait_events
            : (prevEvent() != nullptr ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
        RGYOpenCLEvent planeEvent;
        RGYWorkSize local(32, 8);
        RGYWorkSize global(dst.width, dst.height);
        auto err = m_programs[KFM_PROG_SWITCH].get()->kernel("kernel_kfm_patch_combe").config(queue, local, global, waitHere, &planeEvent).launch(
            (cl_mem)dst.ptr[0], dst.pitch[0],
            (cl_mem)base.ptr[0], base.pitch[0],
            (cl_mem)patch.ptr[0], patch.pitch[0],
            (cl_mem)mask.ptr[0], mask.pitch[0],
            dst.width, dst.height, 0);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_patch_combe (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
        prevEvent = planeEvent;
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    copyFramePropWithoutRes(pOutputFrame, pBaseFrame);
    pOutputFrame->dataList = pBaseFrame->dataList;
    pOutputFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pOutputFrame->flags = RGY_FRAME_FLAG_NONE;
    const char *stage = (stageName && stageName[0]) ? stageName : "patch-combe";
    writeFrameInfoDump(stage, pOutputFrame);
    const auto dumpSts = dumpStageFrame(stage, pOutputFrame, frameIndex, queue,
        (prevEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ prevEvent } : std::vector<RGYOpenCLEvent>());
    if (dumpSts != RGY_ERR_NONE) {
        return dumpSts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::ensureMaskBranchFrames(RGYFrameInfo **ppSwitchFlagFrame, RGYFrameInfo **ppContainsCombeFrame, RGYFrameInfo **ppCombeMaskFrame, const RGYFrameInfo *pTelecineSuperFrame, const TCHAR *stageLabel) {
    if (!ppSwitchFlagFrame || !ppContainsCombeFrame || !ppCombeMaskFrame || !pTelecineSuperFrame) {
        return RGY_ERR_INVALID_CALL;
    }
    const auto prm = std::dynamic_pointer_cast<RGYFilterParamKfm>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_CALL;
    }
    const auto superY = getPlane(pTelecineSuperFrame, RGY_PLANE_Y);
    const int innerWidth = superY.width >> 2;
    const int innerHeight = superY.height >> 1;
    if (innerWidth <= 0 || innerHeight <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM %s mask source size (%dx%d).\n"), stageLabel ? stageLabel : _T(""), superY.width, superY.height);
        return RGY_ERR_INVALID_PARAM;
    }

    const int index = m_maskBranchBufferIndex++ & 3;
    auto switchInfo = prm->frameOut;
    switchInfo.width = innerWidth + 8;
    switchInfo.height = innerHeight + 4;
    switchInfo.csp = RGY_CSP_Y8;
    switchInfo.bitdepth = RGY_CSP_BIT_DEPTH[RGY_CSP_Y8];
    if (!m_switchFlagFrames[index]
        || m_switchFlagFrames[index]->frame.width != switchInfo.width
        || m_switchFlagFrames[index]->frame.height != switchInfo.height
        || m_switchFlagFrames[index]->frame.csp != switchInfo.csp) {
        m_switchFlagFrames[index] = m_cl->createFrameBuffer(switchInfo);
        if (!m_switchFlagFrames[index]) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM %s switch-flag-min frame.\n"), stageLabel ? stageLabel : _T(""));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    auto containsInfo = switchInfo;
    containsInfo.width = 4;
    containsInfo.height = 1;
    if (!m_containsCombeFrames[index]
        || m_containsCombeFrames[index]->frame.width != containsInfo.width
        || m_containsCombeFrames[index]->frame.height != containsInfo.height
        || m_containsCombeFrames[index]->frame.csp != containsInfo.csp) {
        m_containsCombeFrames[index] = m_cl->createFrameBuffer(containsInfo);
        if (!m_containsCombeFrames[index]) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM %s contains-combe frame.\n"), stageLabel ? stageLabel : _T(""));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    auto combeInfo = prm->frameOut;
    if (!m_combeMaskFrames[index]
        || m_combeMaskFrames[index]->frame.width != combeInfo.width
        || m_combeMaskFrames[index]->frame.height != combeInfo.height
        || m_combeMaskFrames[index]->frame.csp != combeInfo.csp) {
        m_combeMaskFrames[index] = m_cl->createFrameBuffer(combeInfo);
        if (!m_combeMaskFrames[index]) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM %s combe-mask-min frame.\n"), stageLabel ? stageLabel : _T(""));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    *ppSwitchFlagFrame = &m_switchFlagFrames[index]->frame;
    *ppContainsCombeFrame = &m_containsCombeFrames[index]->frame;
    *ppCombeMaskFrame = &m_combeMaskFrames[index]->frame;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::resolveContainsCombeCount(KfmContainsCombeReadback& readback, cl_uint *containsCombeCount) {
    if (!readback.submitted) {
        if (containsCombeCount) {
            *containsCombeCount = 0;
        }
        return RGY_ERR_NONE;
    }
    if (!m_containsCombeCount) {
        readback.submitted = false;
        AddMessage(RGY_LOG_ERROR, _T("KFM contains-combe count buffer is missing.\n"));
        return RGY_ERR_NULL_PTR;
    }
    const auto waitSts = m_containsCombeCount->mapEvent().wait();
    if (waitSts != RGY_ERR_NONE) {
        m_containsCombeCount->unmapBuffer(m_fmCountQueue);
        m_fmCountQueue.finish();
        readback.submitted = false;
        AddMessage(RGY_LOG_ERROR, _T("failed to wait KFM contains-combe count readback: %s.\n"), get_err_mes(waitSts));
        return waitSts;
    }
    const auto *mappedCount = reinterpret_cast<const cl_uint *>(m_containsCombeCount->mappedPtr());
    if (!mappedCount) {
        m_containsCombeCount->unmapBuffer(m_fmCountQueue);
        m_fmCountQueue.finish();
        readback.submitted = false;
        AddMessage(RGY_LOG_ERROR, _T("failed to access KFM contains-combe count readback.\n"));
        return RGY_ERR_NULL_PTR;
    }
    if (containsCombeCount) {
        *containsCombeCount = *mappedCount;
    }
    auto unmapSts = m_containsCombeCount->unmapBuffer(m_fmCountQueue);
    if (unmapSts == RGY_ERR_NONE) {
        unmapSts = m_fmCountQueue.finish();
    }
    readback.submitted = false;
    if (unmapSts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to unmap KFM contains-combe count readback: %s.\n"), get_err_mes(unmapSts));
        return unmapSts;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::renderMaskBranch(RGYFrameInfo *pSwitchFlagFrame, RGYFrameInfo *pContainsCombeFrame, RGYFrameInfo *pCombeMaskFrame, const RGYFrameInfo *pTelecineSuperPrevFrame, const RGYFrameInfo *pTelecineSuperFrame, const RGYFrameInfo *pTelecineSuperNextFrame, const char *switchFlagStage, const char *containsCombeStage, const char *combeMaskStage, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event, KfmContainsCombeReadback *containsCombeReadback) {
    if (!pSwitchFlagFrame || !pContainsCombeFrame || !pCombeMaskFrame
        || !pTelecineSuperPrevFrame || !pTelecineSuperFrame || !pTelecineSuperNextFrame
        || !m_programs[KFM_PROG_MASK].get()) {
        return RGY_ERR_INVALID_CALL;
    }
    if (containsCombeReadback) {
        containsCombeReadback->submitted = false;
    }
    const auto superPrevY = getPlane(pTelecineSuperPrevFrame, RGY_PLANE_Y);
    const auto superY = getPlane(pTelecineSuperFrame, RGY_PLANE_Y);
    const auto superNextY = getPlane(pTelecineSuperNextFrame, RGY_PLANE_Y);
    const bool superInterleavedUV = kfmCspHasInterleavedUV(pTelecineSuperFrame->csp);
    const auto superPrevUV = (RGY_CSP_PLANES[pTelecineSuperPrevFrame->csp] > 1) ? getPlane(pTelecineSuperPrevFrame, RGY_PLANE_U) : RGYFrameInfo();
    const auto superUV = (RGY_CSP_PLANES[pTelecineSuperFrame->csp] > 1) ? getPlane(pTelecineSuperFrame, RGY_PLANE_U) : RGYFrameInfo();
    const auto superNextUV = (RGY_CSP_PLANES[pTelecineSuperNextFrame->csp] > 1) ? getPlane(pTelecineSuperNextFrame, RGY_PLANE_U) : RGYFrameInfo();
    const auto superPrevV = (!superInterleavedUV && RGY_CSP_PLANES[pTelecineSuperPrevFrame->csp] > 2) ? getPlane(pTelecineSuperPrevFrame, RGY_PLANE_V) : superPrevUV;
    const auto superV = (!superInterleavedUV && RGY_CSP_PLANES[pTelecineSuperFrame->csp] > 2) ? getPlane(pTelecineSuperFrame, RGY_PLANE_V) : superUV;
    const auto superNextV = (!superInterleavedUV && RGY_CSP_PLANES[pTelecineSuperNextFrame->csp] > 2) ? getPlane(pTelecineSuperNextFrame, RGY_PLANE_V) : superNextUV;
    const auto switchY = getPlane(pSwitchFlagFrame, RGY_PLANE_Y);
    const int superYPitch = superY.pitch[0];
    if (superPrevY.pitch[0] != superYPitch || superNextY.pitch[0] != superYPitch) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM switch-flag pitch mismatch (Y plane: prev %d, cur %d, next %d).\n"),
            superPrevY.pitch[0], superYPitch, superNextY.pitch[0]);
        return RGY_ERR_INVALID_PARAM;
    }
    const int superUVPitch = superUV.ptr[0] ? superUV.pitch[0] : superYPitch;
    const int superVPitch = superV.ptr[0] ? superV.pitch[0] : superUVPitch;
    if (superUV.ptr[0]) {
        if (superPrevUV.pitch[0] != superUVPitch || superNextUV.pitch[0] != superUVPitch) {
            AddMessage(RGY_LOG_ERROR, _T("invalid KFM switch-flag pitch mismatch (UV plane: prev %d, cur %d, next %d).\n"),
                superPrevUV.pitch[0], superUVPitch, superNextUV.pitch[0]);
            return RGY_ERR_INVALID_PARAM;
        }
        if (!superInterleavedUV && RGY_CSP_PLANES[pTelecineSuperFrame->csp] > 2) {
            if (superPrevV.pitch[0] != superVPitch || superNextV.pitch[0] != superVPitch) {
                AddMessage(RGY_LOG_ERROR, _T("invalid KFM switch-flag pitch mismatch (U/V plane: prev %d/%d, cur %d/%d, next %d/%d).\n"),
                    superPrevUV.pitch[0], superPrevV.pitch[0], superUVPitch, superV.pitch[0], superNextUV.pitch[0], superNextV.pitch[0]);
                return RGY_ERR_INVALID_PARAM;
            }
        }
    }
    const int innerWidth = superY.width >> 2;
    const int innerHeight = superY.height >> 1;
    const int combeWidth = superY.width >> 1;
    const int combeHeight = superY.height;
    const int combeCWidth = superUV.ptr[0] ? (superInterleavedUV ? (superUV.width >> 2) : (superUV.width >> 1)) : 0;
    const int combeCHeight = superUV.ptr[0] ? superUV.height : 0;
    if (switchY.width != innerWidth + 8 || switchY.height != innerHeight + 4) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM switch-flag-min size (super %dx%d, flag %dx%d).\n"),
            superY.width, superY.height, switchY.width, switchY.height);
        return RGY_ERR_INVALID_PARAM;
    }
    if (superPrevY.width != superY.width || superPrevY.height != superY.height
        || superNextY.width != superY.width || superNextY.height != superY.height
        || combeWidth <= 0 || combeHeight <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("invalid KFM switch-flag super triplet size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int maskDumpFrameIndex = pTelecineSuperFrame->inputFrameId >= 0 ? pTelecineSuperFrame->inputFrameId : m_timecodeFrameIndex;

    const int combeYPitch = combeWidth;
    const int combeCPitch = std::max(1, combeCWidth);
    const int flagPitch = switchY.width;
    const size_t combeYBytes = (size_t)combeYPitch * combeHeight;
    const size_t combeCBytes = (size_t)combeCPitch * std::max(1, combeCHeight);
    const size_t flagBytes = (size_t)flagPitch * switchY.height;
    const std::array<size_t, 4> workBytes = { std::max(combeYBytes, flagBytes), std::max(combeCBytes, flagBytes), flagBytes, flagBytes };
    for (int i = 0; i < (int)workBytes.size(); i++) {
        if (!m_switchFlagWork[i] || m_switchFlagWork[i]->size() < workBytes[i]) {
            m_switchFlagWork[i] = m_cl->createBuffer(workBytes[i], CL_MEM_READ_WRITE);
            if (!m_switchFlagWork[i]) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM switch-flag work buffer %d.\n"), i);
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
    }

    auto workWaitEvents = wait_events;
    if (m_switchFlagWorkEvent() != nullptr) {
        workWaitEvents.push_back(m_switchFlagWorkEvent);
    }

    RGYOpenCLEvent combeEvent;
    RGYWorkSize flagLocal(32, 8);
    RGYWorkSize combeGlobal(std::max(combeWidth, combeCWidth), std::max(combeHeight, combeCHeight));
    RGYWorkSize flagGlobal(switchY.width, switchY.height);
    auto dumpSwitchWorkFrame = [&](const char *stage, const std::unique_ptr<RGYCLBuf>& work, int width, int height, int pitch, const RGYOpenCLEvent& waitEvent) -> RGY_ERR {
        if (!stageDumpRequested(maskDumpFrameIndex) || !work) {
            return RGY_ERR_NONE;
        }
        RGYOpenCLEvent copyEvent;
        auto copySts = m_programs[KFM_PROG_MASK].get()->kernel("kernel_kfm_copy_u8_buffer_to_plane").config(queue, flagLocal, flagGlobal, { waitEvent }, &copyEvent).launch(
            (cl_mem)switchY.ptr[0], switchY.pitch[0],
            (cl_mem)work->mem(), pitch,
            width, height);
        if (copySts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_copy_u8_buffer_to_plane (%s): %s.\n"), char_to_tstring(stage).c_str(), get_err_mes(copySts));
            return copySts;
        }
        return dumpStageFrame(stage, pSwitchFlagFrame, maskDumpFrameIndex, queue, { copyEvent });
    };

    auto sts = m_programs[KFM_PROG_MASK].get()->kernel("kernel_kfm_switch_flag_combe_min").config(queue, flagLocal, combeGlobal, workWaitEvents, &combeEvent).launch(
        (cl_mem)m_switchFlagWork[0]->mem(), combeYPitch,
        (cl_mem)m_switchFlagWork[1]->mem(), combeCPitch,
        (cl_mem)(superPrevY.ptr[0]), (cl_mem)(superY.ptr[0]), (cl_mem)(superNextY.ptr[0]), superYPitch,
        (cl_mem)(superPrevUV.ptr[0] ? superPrevUV.ptr[0] : superPrevY.ptr[0]),
        (cl_mem)(superUV.ptr[0] ? superUV.ptr[0] : superY.ptr[0]),
        (cl_mem)(superNextUV.ptr[0] ? superNextUV.ptr[0] : superNextY.ptr[0]), superUVPitch,
        (cl_mem)(superPrevV.ptr[0] ? superPrevV.ptr[0] : superPrevY.ptr[0]),
        (cl_mem)(superV.ptr[0] ? superV.ptr[0] : superY.ptr[0]),
        (cl_mem)(superNextV.ptr[0] ? superNextV.ptr[0] : superNextY.ptr[0]),
        superVPitch,
        combeWidth, combeHeight,
        combeCWidth, combeCHeight,
        superUV.ptr[0] ? 1 : 0,
        superInterleavedUV ? 1 : 0);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_switch_flag_combe_min: %s.\n"), get_err_mes(sts));
        return sts;
    }

    RGYOpenCLEvent fromCombeEvent;
    sts = m_programs[KFM_PROG_MASK].get()->kernel("kernel_kfm_switch_flag_from_combe_min").config(queue, flagLocal, flagGlobal, { combeEvent }, &fromCombeEvent).launch(
        (cl_mem)m_switchFlagWork[2]->mem(), flagPitch,
        (cl_mem)m_switchFlagWork[3]->mem(), flagPitch,
        (cl_mem)m_switchFlagWork[0]->mem(), combeYPitch,
        (cl_mem)m_switchFlagWork[1]->mem(), combeCPitch,
        switchY.width, switchY.height,
        innerWidth, innerHeight,
        combeWidth, combeHeight,
        combeCWidth, combeCHeight);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_switch_flag_from_combe_min: %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = dumpSwitchWorkFrame("switch-from-combe-y", m_switchFlagWork[2], switchY.width, switchY.height, flagPitch, fromCombeEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = dumpSwitchWorkFrame("switch-from-combe-c", m_switchFlagWork[3], switchY.width, switchY.height, flagPitch, fromCombeEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    RGYOpenCLEvent boxY0Event;
    sts = m_programs[KFM_PROG_MASK].get()->kernel("kernel_kfm_switch_flag_box3x3_min").config(queue, flagLocal, flagGlobal, { fromCombeEvent }, &boxY0Event).launch(
        (cl_mem)m_switchFlagWork[0]->mem(), flagPitch,
        (cl_mem)m_switchFlagWork[2]->mem(), flagPitch,
        switchY.width, switchY.height,
        innerWidth, innerHeight);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_switch_flag_box3x3_min (Y pass 0): %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = dumpSwitchWorkFrame("switch-box1-y", m_switchFlagWork[0], switchY.width, switchY.height, flagPitch, boxY0Event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    RGYOpenCLEvent boxY1Event;
    sts = m_programs[KFM_PROG_MASK].get()->kernel("kernel_kfm_switch_flag_box3x3_min").config(queue, flagLocal, flagGlobal, { boxY0Event }, &boxY1Event).launch(
        (cl_mem)m_switchFlagWork[2]->mem(), flagPitch,
        (cl_mem)m_switchFlagWork[0]->mem(), flagPitch,
        switchY.width, switchY.height,
        innerWidth, innerHeight);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_switch_flag_box3x3_min (Y pass 1): %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = dumpSwitchWorkFrame("switch-box2-y", m_switchFlagWork[2], switchY.width, switchY.height, flagPitch, boxY1Event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    RGYOpenCLEvent boxC0Event;
    sts = m_programs[KFM_PROG_MASK].get()->kernel("kernel_kfm_switch_flag_box3x3_min").config(queue, flagLocal, flagGlobal, { fromCombeEvent }, &boxC0Event).launch(
        (cl_mem)m_switchFlagWork[1]->mem(), flagPitch,
        (cl_mem)m_switchFlagWork[3]->mem(), flagPitch,
        switchY.width, switchY.height,
        innerWidth, innerHeight);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_switch_flag_box3x3_min (C pass 0): %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = dumpSwitchWorkFrame("switch-box1-c", m_switchFlagWork[1], switchY.width, switchY.height, flagPitch, boxC0Event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    RGYOpenCLEvent boxC1Event;
    sts = m_programs[KFM_PROG_MASK].get()->kernel("kernel_kfm_switch_flag_box3x3_min").config(queue, flagLocal, flagGlobal, { boxC0Event }, &boxC1Event).launch(
        (cl_mem)m_switchFlagWork[3]->mem(), flagPitch,
        (cl_mem)m_switchFlagWork[1]->mem(), flagPitch,
        switchY.width, switchY.height,
        innerWidth, innerHeight);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_switch_flag_box3x3_min (C pass 1): %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = dumpSwitchWorkFrame("switch-box2-c", m_switchFlagWork[3], switchY.width, switchY.height, flagPitch, boxC1Event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    RGYOpenCLEvent switchEvent;
    if (kfmUseFusedSwitchFlagBinaryExtend()) {
        sts = m_programs[KFM_PROG_MASK].get()->kernel("kernel_kfm_switch_flag_binary_extend_hv_min").config(queue, flagLocal, flagGlobal, { boxY1Event, boxC1Event }, &switchEvent).launch(
            (cl_mem)switchY.ptr[0], switchY.pitch[0],
            (cl_mem)m_switchFlagWork[2]->mem(), flagPitch,
            (cl_mem)m_switchFlagWork[3]->mem(), flagPitch,
            switchY.width, switchY.height,
            innerWidth, innerHeight,
            KFM_SWITCH_FLAG_THRESH_Y, KFM_SWITCH_FLAG_THRESH_C);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_switch_flag_binary_extend_hv_min: %s.\n"), get_err_mes(sts));
            return sts;
        }
    } else {
        RGYOpenCLEvent binaryEvent;
        sts = m_programs[KFM_PROG_MASK].get()->kernel("kernel_kfm_switch_flag_binary_min").config(queue, flagLocal, flagGlobal, { boxY1Event, boxC1Event }, &binaryEvent).launch(
            (cl_mem)switchY.ptr[0], switchY.pitch[0],
            (cl_mem)m_switchFlagWork[2]->mem(), flagPitch,
            (cl_mem)m_switchFlagWork[3]->mem(), flagPitch,
            switchY.width, switchY.height,
            innerWidth, innerHeight,
            KFM_SWITCH_FLAG_THRESH_Y, KFM_SWITCH_FLAG_THRESH_C);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_switch_flag_binary_min: %s.\n"), get_err_mes(sts));
            return sts;
        }
        RGYOpenCLEvent extendHEvent;
        sts = m_programs[KFM_PROG_MASK].get()->kernel("kernel_kfm_switch_flag_extend_h_min").config(queue, flagLocal, flagGlobal, { binaryEvent }, &extendHEvent).launch(
            (cl_mem)m_switchFlagWork[0]->mem(), flagPitch,
            (cl_mem)switchY.ptr[0], switchY.pitch[0],
            innerWidth, innerHeight, 4, 2);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_switch_flag_extend_h_min: %s.\n"), get_err_mes(sts));
            return sts;
        }
        sts = m_programs[KFM_PROG_MASK].get()->kernel("kernel_kfm_switch_flag_extend_v_min").config(queue, flagLocal, flagGlobal, { extendHEvent }, &switchEvent).launch(
            (cl_mem)switchY.ptr[0], switchY.pitch[0],
            (cl_mem)m_switchFlagWork[0]->mem(), flagPitch,
            innerWidth, innerHeight, 4, 2);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_switch_flag_extend_v_min: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }
    m_switchFlagWorkEvent = switchEvent;

    if (!m_containsCombeCount || m_containsCombeCount->size() < sizeof(cl_uint)) {
        m_containsCombeCount = m_cl->createBuffer(sizeof(cl_uint), CL_MEM_READ_WRITE);
        if (!m_containsCombeCount) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM contains-combe count buffer.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    RGYOpenCLEvent initEvent;
    sts = m_programs[KFM_PROG_MASK].get()->kernel("kernel_kfm_contains_combe_init").config(queue, RGYWorkSize(1), RGYWorkSize(1), {}, &initEvent).launch(
        (cl_mem)m_containsCombeCount->mem());
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_contains_combe_init: %s.\n"), get_err_mes(sts));
        return sts;
    }

    RGYOpenCLEvent countEvent;
    sts = m_programs[KFM_PROG_MASK].get()->kernel("kernel_kfm_contains_combe_count").config(queue, RGYWorkSize(32, 8), RGYWorkSize(switchY.width, switchY.height), { switchEvent, initEvent }, &countEvent).launch(
        (cl_mem)switchY.ptr[0], switchY.pitch[0],
        (cl_mem)m_containsCombeCount->mem(),
        switchY.width, switchY.height, 128);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_contains_combe_count: %s.\n"), get_err_mes(sts));
        return sts;
    }

    const auto containsY = getPlane(pContainsCombeFrame, RGY_PLANE_Y);
    RGYOpenCLEvent markEvent;
    sts = m_programs[KFM_PROG_MASK].get()->kernel("kernel_kfm_contains_combe_mark").config(queue, RGYWorkSize(4, 1), RGYWorkSize(4, 1), { countEvent }, &markEvent).launch(
        (cl_mem)containsY.ptr[0], containsY.pitch[0],
        (cl_mem)m_containsCombeCount->mem());
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_contains_combe_mark: %s.\n"), get_err_mes(sts));
        return sts;
    }
    auto cleanupContainsCombeReadback = [&]() {
        if (!containsCombeReadback) {
            return RGY_ERR_NONE;
        }
        return resolveContainsCombeCount(*containsCombeReadback, nullptr);
    };
    if (containsCombeReadback) {
        auto readSts = ensureFMCountQueue();
        if (readSts != RGY_ERR_NONE) {
            return readSts;
        }
        readSts = m_containsCombeCount->queueMapBuffer(m_fmCountQueue, CL_MAP_READ, { countEvent }, RGY_CL_MAP_BLOCK_NONE, "kfm.contains_combe.count");
        if (readSts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to submit KFM contains-combe count readback: %s.\n"), get_err_mes(readSts));
            return readSts;
        }
        queue.flush();
        m_fmCountQueue.flush();
        containsCombeReadback->submitted = true;
    }

    RGYOpenCLEvent prevEvent = markEvent;
    const int planes = RGY_CSP_PLANES[pCombeMaskFrame->csp];
    const bool interleavedUV = kfmCspHasInterleavedUV(pCombeMaskFrame->csp);
    for (int iplane = 0; iplane < planes; iplane++) {
        const bool interleavedChroma = interleavedUV && iplane > 0;
        const auto plane = interleavedChroma ? RGY_PLANE_U : (RGY_PLANE)iplane;
        const auto dst = getPlane(pCombeMaskFrame, plane);
        const int step = interleavedChroma ? 2 : 1;
        const int offset = interleavedChroma ? iplane - 1 : 0;
        const int logicalWidth = dst.width / step;
        const int logicalHeight = dst.height;
        const int scaleX = logicalWidth / innerWidth;
        const int scaleY = logicalHeight / innerHeight;
        const int shiftX = kfmPow2Shift(scaleX);
        const int shiftY = kfmPow2Shift(scaleY);
        if (logicalWidth <= 0 || logicalHeight <= 0 || logicalWidth != innerWidth * scaleX || logicalHeight != innerHeight * scaleY || shiftX < 0 || shiftY < 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported KFM combe-mask-min scale (plane %d, dst %dx%d, flag inner %dx%d).\n"),
                iplane, logicalWidth, logicalHeight, innerWidth, innerHeight);
            auto readSts = cleanupContainsCombeReadback();
            if (readSts != RGY_ERR_NONE) {
                return readSts;
            }
            return RGY_ERR_INVALID_PARAM;
        }
        RGYOpenCLEvent planeEvent;
        sts = m_programs[KFM_PROG_MASK].get()->kernel("kernel_kfm_combe_mask_resize_bilinear_min").config(queue, RGYWorkSize(32, 8), RGYWorkSize(logicalWidth, logicalHeight), { prevEvent }, &planeEvent).launch(
            (cl_mem)dst.ptr[0], dst.pitch[0],
            (cl_mem)switchY.ptr[0], switchY.pitch[0],
            logicalWidth, logicalHeight,
            step, offset,
            scaleX, shiftX,
            scaleY, shiftY,
            innerWidth, innerHeight);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_kfm_combe_mask_resize_bilinear_min (plane %d): %s.\n"), iplane, get_err_mes(sts));
            auto readSts = cleanupContainsCombeReadback();
            if (readSts != RGY_ERR_NONE) {
                return readSts;
            }
            return sts;
        }
        prevEvent = planeEvent;
    }

    copyFramePropWithoutRes(pSwitchFlagFrame, pTelecineSuperFrame);
    copyFramePropWithoutRes(pContainsCombeFrame, pTelecineSuperFrame);
    copyFramePropWithoutRes(pCombeMaskFrame, pTelecineSuperFrame);
    pSwitchFlagFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pContainsCombeFrame->picstruct = RGY_PICSTRUCT_FRAME;
    pCombeMaskFrame->picstruct = RGY_PICSTRUCT_FRAME;
    writeFrameInfoDump(switchFlagStage, pSwitchFlagFrame);
    writeFrameInfoDump(containsCombeStage, pContainsCombeFrame);
    writeFrameInfoDump(combeMaskStage, pCombeMaskFrame);
    sts = dumpStageFrame(switchFlagStage, pSwitchFlagFrame, maskDumpFrameIndex, queue, { switchEvent });
    if (sts != RGY_ERR_NONE) {
        auto readSts = cleanupContainsCombeReadback();
        if (readSts != RGY_ERR_NONE) {
            return readSts;
        }
        return sts;
    }
    sts = dumpStageFrame(containsCombeStage, pContainsCombeFrame, maskDumpFrameIndex, queue, { markEvent });
    if (sts != RGY_ERR_NONE) {
        auto readSts = cleanupContainsCombeReadback();
        if (readSts != RGY_ERR_NONE) {
            return readSts;
        }
        return sts;
    }
    sts = dumpStageFrame(combeMaskStage, pCombeMaskFrame, maskDumpFrameIndex, queue, { prevEvent });
    if (sts != RGY_ERR_NONE) {
        auto readSts = cleanupContainsCombeReadback();
        if (readSts != RGY_ERR_NONE) {
            return readSts;
        }
        return sts;
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::processMainRtgmcOutputs(const RGYFilterParamKfm& prm, RGYFrameInfo **rtgmcOutFrames, int rtgmcOutNum,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (rtgmcOutNum <= 0) {
        return RGY_ERR_NONE;
    }
    std::vector<RGYOpenCLEvent> mergeWaitEvents = wait_events;
    RGYOpenCLEvent staticEvent;
    auto sts = analyzeStaticFlag(queue, wait_events, &staticEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (staticEvent() != nullptr) {
        mergeWaitEvents.push_back(staticEvent);
    }
    for (int i = 0; i < rtgmcOutNum; i++) {
        auto out = nextWorkFrame();
        if (!out) {
            return RGY_ERR_INVALID_CALL;
        }
        auto frameWaitEvents = mergeWaitEvents;
        const auto source = findSourceFrame(rtgmcOutFrames[i], &frameWaitEvents);
        if (!source) {
            AddMessage(RGY_LOG_ERROR, _T("KFM source frame is missing for output inputFrameId=%d.\n"), rtgmcOutFrames[i]->inputFrameId);
            return RGY_ERR_INVALID_CALL;
        }
        RGYOpenCLEvent mergeEvent;
        sts = mergeStatic(out, rtgmcOutFrames[i], source, queue, frameWaitEvents, &mergeEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (prm.kfm.ucf) {
            auto ucfOut = nextWorkFrame();
            if (!ucfOut) {
                return RGY_ERR_INVALID_CALL;
            }
            std::vector<RGYOpenCLEvent> ucfWaitEvents = frameWaitEvents;
            if (mergeEvent() != nullptr) {
                ucfWaitEvents.push_back(mergeEvent);
            }
            sts = resolveUcfNoiseResults((m_timecodeFrameIndex >> 1) + 3, queue);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            const auto *ucfInput = selectUcfDecomb60Frame(m_timecodeFrameIndex, out, &ucfWaitEvents);
            RGYOpenCLEvent ucfEvent;
            sts = copyUcfFrame(prm, ucfOut, ucfInput, queue, ucfWaitEvents, &ucfEvent);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            copyFramePropWithoutRes(ucfOut, out);
            sts = emitOutputFrame(ucfOut, ppOutputFrames, pOutputFrameNum, queue, ucfEvent, event);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        } else {
            sts = emitOutputFrame(out, ppOutputFrames, pOutputFrameNum, queue, mergeEvent, event);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::drainMainRtgmcBranch(const RGYFilterParamKfm& prm, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    if (!m_rtgmc) {
        return RGY_ERR_NONE;
    }
    const auto maxDrainIterations = std::max(256, m_cachedSourceFrames * 4 + 256);
    for (int iter = 0; *pOutputFrameNum == 0 && !m_rtgmc->drainComplete(); iter++) {
        if (iter >= maxDrainIterations) {
            AddMessage(RGY_LOG_ERROR, _T("KFM main RTGMC drain did not complete after %d iterations.\n"), maxDrainIterations);
            return RGY_ERR_INVALID_CALL;
        }
        int rtgmcOutNum = 0;
        RGYFrameInfo *rtgmcOutFrames[8] = { 0 };
        RGYOpenCLEvent rtgmcEvent;
        auto sts = m_rtgmc->filter(nullptr, rtgmcOutFrames, &rtgmcOutNum, queue, {}, &rtgmcEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        std::vector<RGYOpenCLEvent> processWaitEvents;
        if (rtgmcEvent() != nullptr) {
            processWaitEvents.push_back(rtgmcEvent);
        }
        sts = processMainRtgmcOutputs(prm, rtgmcOutFrames, rtgmcOutNum, ppOutputFrames, pOutputFrameNum, queue, processWaitEvents, event);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterKfm::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const auto prm = std::dynamic_pointer_cast<RGYFilterParamKfm>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_CALL;
    }

    if (m_rtgmc) {
        auto sts = cacheSourceFrame(pInputFrame, queue, wait_events);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
            sts = analyzeAvailableSource(true, queue);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        int rtgmcOutNum = 0;
        RGYFrameInfo *rtgmcOutFrames[8] = { 0 };
        RGYOpenCLEvent rtgmcEvent;
        sts = m_rtgmc->filter(const_cast<RGYFrameInfo *>(pInputFrame), rtgmcOutFrames, &rtgmcOutNum, queue, wait_events, &rtgmcEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (prm->kfm.ucf) {
            const bool lazyDeint60 = lazyDeint60Enabled(*prm);
            for (auto &captured : m_rtgmc->getCapturedIntermediates()) {
                if (m_before60Rtgmc) m_before60Rtgmc->pushIntermediateInput(captured);
                if (m_after60Rtgmc) m_after60Rtgmc->pushIntermediateInput(captured);
            }
            m_rtgmc->clearCapturedIntermediates();
            sts = lazyDeint60
                ? runUcfNoiseAnalysisFromSource(pInputFrame, queue, {})
                : runUcfRtgmcBranches(pInputFrame, queue, {});
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
                sts = resolveAllUcfNoiseResults(queue);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
        }
        *pOutputFrameNum = 0;
        std::vector<RGYOpenCLEvent> processWaitEvents;
        if (rtgmcEvent() != nullptr) {
            processWaitEvents.push_back(rtgmcEvent);
        }
        sts = processMainRtgmcOutputs(*prm, rtgmcOutFrames, rtgmcOutNum, ppOutputFrames, pOutputFrameNum, queue, processWaitEvents, event);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if ((pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) && *pOutputFrameNum == 0 && !m_rtgmc->drainComplete()) {
            sts = drainMainRtgmcBranch(*prm, ppOutputFrames, pOutputFrameNum, queue, event);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if ((pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) && *pOutputFrameNum == 0 && m_rtgmc->drainComplete()) {
            sts = drainNrFilter(ppOutputFrames, pOutputFrameNum, queue, event);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        return RGY_ERR_NONE;
    }

    if (prm->kfm.mode == VppKfmMode::VFR) {
        auto sts = RGY_ERR_NONE;
        if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
            sts = analyzeAvailableSource(true, queue);
        } else {
            sts = cacheSourceFrame(pInputFrame, queue, wait_events);
        }
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        const bool lazyDeint60 = lazyDeint60Enabled(*prm);
        sts = lazyDeint60
            ? ((pInputFrame && pInputFrame->ptr[0]) ? m_deint60Lane.feedHot(queue) : RGY_ERR_NONE)
            : ((pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) ? drainDeint60Branch(queue) : runDeint60Branch(pInputFrame, queue, wait_events));
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (!lazyDeint60 && prm->kfm.ucf && m_deint60Rtgmc) {
            for (auto &captured : m_deint60Rtgmc->getCapturedIntermediates()) {
                if (m_before60Rtgmc) m_before60Rtgmc->pushIntermediateInput(captured);
                if (m_after60Rtgmc) m_after60Rtgmc->pushIntermediateInput(captured);
            }
            m_deint60Rtgmc->clearCapturedIntermediates();
        }
        if (prm->kfm.ucf) {
            sts = runUcfRtgmcBranches(pInputFrame, queue, wait_events);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
                sts = resolveAllUcfNoiseResults(queue);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
        }

        *pOutputFrameNum = 0;
        const bool drain = pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr;
        const int rawAvailableN60 = drain
            ? m_cachedSourceFrames * 2
            : std::min(m_cachedSourceFrames * 2, static_cast<int>(m_analyzerOutputResults.size()) * 10);
        // Keep the non-drain tail until the longest VFR duration is known.
        // Otherwise a 24p timing that crosses the current analysis frontier is
        // emitted as a short frame and the same 24p source frame is emitted
        // again later.  The switch single-frame check also needs the next
        // telecine-super frame, so keep one more 24p frame worth of margin.
        const int vfrTailHold60 = switchSingleFrameDurationEnabled() ? 8 : 4;
        const int availableN60 = drain ? rawAvailableN60 : std::max(0, rawAvailableN60 - vfrTailHold60);
        const auto timings = deriveSwitchTimings(availableN60);
        const int maxOutputFrames = std::min<int>((int)m_frameBuf.size(), 4);
        const int vfrOutputDelay = switchSingleFrameDurationEnabled() ? 1 : 0;
        auto emitReadyPending = [&](int keepFrames) -> RGY_ERR {
            return emitPendingVfrOutputs(ppOutputFrames, pOutputFrameNum, queue, event, keepFrames);
        };
        auto ensureDeint60Range = [&](int n60begin, int n60end) -> RGY_ERR {
            return lazyDeint60 ? m_deint60Lane.ensureRange(n60begin, n60end, queue) : RGY_ERR_NONE;
        };
        sts = emitReadyPending(drain ? 0 : vfrOutputDelay);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        while (*pOutputFrameNum < maxOutputFrames) {
            auto itTiming = std::find_if(timings.begin(), timings.end(), [this](const KfmSwitchTiming& timing) {
                return timing.start60 == m_nextSwitchN60;
            });
            if (itTiming == timings.end()) {
                itTiming = std::find_if(timings.begin(), timings.end(), [this](const KfmSwitchTiming& timing) {
                    return timing.start60 < m_nextSwitchN60 && m_nextSwitchN60 < timing.start60 + timing.duration60;
                });
                if (itTiming == timings.end()) {
                    break;
                }
            }
            auto outputTiming = *itTiming;
            if (outputTiming.start60 < m_nextSwitchN60) {
                const auto consumed60 = m_nextSwitchN60 - outputTiming.start60;
                outputTiming.start60 = m_nextSwitchN60;
                outputTiming.start120 += consumed60 * 2;
                outputTiming.duration60 = std::max(1, outputTiming.duration60 - consumed60);
                outputTiming.duration120 = outputTiming.duration60 * 2;
                outputTiming.numSourceFrames = std::max(1, divCeil(outputTiming.duration60, 2));
            }
            if (!drain && outputTiming.start60 + outputTiming.duration60 >= availableN60) {
                break;
            }
            const auto rawStart120 = [](const KfmSwitchTiming& timing) {
                return static_cast<int64_t>(timing.start60) * 2;
            };
            auto sourcePtsFrom120 = [&](const int64_t pos120) {
                if (m_sourceCache.empty()) {
                    return pos120;
                }
                const int64_t sourceIndex = pos120 >> 2;
                const int offset120 = static_cast<int>(pos120 & 3);
                const auto *source = findSourceByIndex(static_cast<int>(sourceIndex));
                const auto duration = sourceFrameDuration(source);
                if (!source || source->timestamp < 0) {
                    return (duration * pos120 + 2) / 4;
                }
                const int64_t sourceOffset120 = (sourceIndex - source->sourceIndex) * 4 + offset120;
                return source->timestamp + (duration * sourceOffset120 + 2) / 4;
            };
            const auto canUse120Cadence = [](bool prevIsFrame24, int prevDuration60, const KfmSwitchTiming& cur) {
                return prevIsFrame24 && cur.isFrame24
                    && prevDuration60 >= 2 && cur.duration60 >= 2
                    && prevDuration60 + cur.duration60 == 5;
            };
            int64_t outputStart120 = rawStart120(outputTiming);
            if (prm->kfm.is120
                && m_hasLastSwitchTiming
                && m_lastSwitchStart60 + m_lastSwitchDuration60 == outputTiming.start60
                && canUse120Cadence(m_lastSwitchIsFrame24, m_lastSwitchDuration60, outputTiming)) {
                outputStart120 = m_lastSwitchStart120 + 5;
            }
            int64_t nextStart120 = outputStart120 + outputTiming.duration60 * 2;
            const auto itNextTiming = std::find_if(timings.begin(), timings.end(), [&outputTiming](const KfmSwitchTiming& timing) {
                return timing.start60 == outputTiming.start60 + outputTiming.duration60;
            });
            if (itNextTiming != timings.end()) {
                nextStart120 = rawStart120(*itNextTiming);
                if (prm->kfm.is120 && canUse120Cadence(outputTiming.isFrame24, outputTiming.duration60, *itNextTiming)) {
                    nextStart120 = outputStart120 + 5;
                }
            }
            outputTiming.start120 = static_cast<int>(outputStart120);
            if (nextStart120 > outputStart120) {
                outputTiming.duration120 = static_cast<int>(nextStart120 - outputStart120);
            }
            const int copySourceIndex = outputTiming.baseType == KFM_FRAME_60 ? (outputTiming.start60 >> 1) : outputTiming.sourceIndex;
            const auto *source = findSourceByIndex(copySourceIndex);
            const auto *switchResult = &m_analyzerOutputResults[clamp(outputTiming.start60 / 10, 0, (int)m_analyzerOutputResults.size() - 1)];
            RGYOpenCLEvent outputEvent;
            RGYFrameInfo *out = nullptr;
            if (outputTiming.baseType == KFM_FRAME_24) {
                const auto savedWorkBufferIndex = m_workBufferIndex;
                const auto savedTelecineSuperBufferIndex = m_telecineSuperBufferIndex;
                auto deint24 = nextWorkFrame();
                out = nextWorkFrame();
                if (!deint24 || !out) {
                    return RGY_ERR_INVALID_CALL;
                }

                const int superIndex = m_telecineSuperBufferIndex++ & 1;
                if (!m_telecineSuperFrames[superIndex]) {
                    auto superInfo = prm->frameOut;
                    superInfo.width = std::max(1, superInfo.width >> 1);
                    superInfo.height = std::max(1, superInfo.height >> 2);
                    m_telecineSuperFrames[superIndex] = m_cl->createFrameBuffer(superInfo);
                    if (!m_telecineSuperFrames[superIndex]) {
                        AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM telecine-super frame.\n"));
                        return RGY_ERR_MEMORY_ALLOC;
                    }
                }
                auto super24 = &m_telecineSuperFrames[superIndex]->frame;

                const auto savedTelecine24Frame = m_nextTelecine24Frame;
                const auto savedTelecine24Pts = m_nextTelecine24Pts;
                RGYOpenCLEvent deintEvent;
                sts = renderTelecine24(deint24, outputTiming.frame24Index, drain, queue, wait_events, &deintEvent);
                m_nextTelecine24Frame = savedTelecine24Frame;
                m_nextTelecine24Pts = savedTelecine24Pts;
                if (sts == RGY_ERR_MORE_DATA) {
                    m_workBufferIndex = savedWorkBufferIndex;
                    m_telecineSuperBufferIndex = savedTelecineSuperBufferIndex;
                    break;
                }
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }

                std::vector<RGYOpenCLEvent> superWaitEvents;
                if (deintEvent() != nullptr) {
                    superWaitEvents.push_back(deintEvent);
                }
                RGYOpenCLEvent superEvent;
                sts = renderTelecineSuper24(super24, outputTiming.frame24Index, drain, queue, superWaitEvents, &superEvent);
                if (sts == RGY_ERR_MORE_DATA) {
                    m_workBufferIndex = savedWorkBufferIndex;
                    m_telecineSuperBufferIndex = savedTelecineSuperBufferIndex;
                    break;
                }
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }

                std::vector<RGYOpenCLEvent> removeWaitEvents = superWaitEvents;
                if (superEvent() != nullptr) {
                    removeWaitEvents.push_back(superEvent);
                }
                RGYFrameInfo *superPrev24 = super24;
                RGYFrameInfo *superNext24 = super24;
                std::vector<RGYOpenCLEvent> maskWaitEvents = removeWaitEvents;
                auto ensureNeighborSuper = [&](int index, RGYFrameInfo **frame) -> RGY_ERR {
                    if (!m_telecineSuperNeighborFrames[index]
                        || m_telecineSuperNeighborFrames[index]->frame.width != super24->width
                        || m_telecineSuperNeighborFrames[index]->frame.height != super24->height
                        || m_telecineSuperNeighborFrames[index]->frame.csp != super24->csp) {
                        auto superInfo = *super24;
                        m_telecineSuperNeighborFrames[index] = m_cl->createFrameBuffer(superInfo);
                        if (!m_telecineSuperNeighborFrames[index]) {
                            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM telecine-super neighbor frame.\n"));
                            return RGY_ERR_MEMORY_ALLOC;
                        }
                    }
                    *frame = &m_telecineSuperNeighborFrames[index]->frame;
                    return RGY_ERR_NONE;
                };
                if (outputTiming.frame24Index > 0) {
                    RGYOpenCLEvent prevSuperEvent;
                    sts = ensureNeighborSuper(0, &superPrev24);
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    sts = renderTelecineSuper24(superPrev24, outputTiming.frame24Index - 1, true, queue, superWaitEvents, &prevSuperEvent);
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    if (prevSuperEvent() != nullptr) {
                        maskWaitEvents.push_back(prevSuperEvent);
                    }
                }
                const int analyzed24Frames = (int)m_analyzerOutputResults.size() * 4;
                if (outputTiming.frame24Index + 1 < analyzed24Frames) {
                    RGYOpenCLEvent nextSuperEvent;
                    sts = ensureNeighborSuper(1, &superNext24);
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    sts = renderTelecineSuper24(superNext24, outputTiming.frame24Index + 1, drain, queue, superWaitEvents, &nextSuperEvent);
                    if (sts == RGY_ERR_MORE_DATA) {
                        m_workBufferIndex = savedWorkBufferIndex;
                        m_telecineSuperBufferIndex = savedTelecineSuperBufferIndex;
                        break;
                    } else if (sts != RGY_ERR_NONE) {
                        return sts;
                    } else if (nextSuperEvent() != nullptr) {
                        maskWaitEvents.push_back(nextSuperEvent);
                    }
                } else if (!drain) {
                    m_workBufferIndex = savedWorkBufferIndex;
                    m_telecineSuperBufferIndex = savedTelecineSuperBufferIndex;
                    break;
                }
                RGYFrameInfo *switchFlag = nullptr;
                RGYFrameInfo *containsCombe = nullptr;
                RGYFrameInfo *combeMask = nullptr;
                sts = ensureMaskBranchFrames(&switchFlag, &containsCombe, &combeMask, super24, _T("24p"));
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                RGYOpenCLEvent maskEvent;
                cl_uint containsCombeCount = 0;
                KfmContainsCombeReadback containsCombeReadback;
                const bool patchCombe24Enabled = kfmDeint60BranchEnabled() && outputTiming.frame24Index >= 0 && m_deint60Rtgmc && m_analyzer;
                const bool needsContainsCombeCount = switchSingleFrameDurationEnabled() || patchCombe24Enabled;
                sts = renderMaskBranch(switchFlag, containsCombe, combeMask, superPrev24, super24, superNext24, "switch-flag-min", "contains-combe", "combe-mask-min", queue, maskWaitEvents, &maskEvent, needsContainsCombeCount ? &containsCombeReadback : nullptr);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                if (maskEvent() != nullptr) {
                    removeWaitEvents.push_back(maskEvent);
                }
                auto resolveContainsCombeDuration = [&]() -> RGY_ERR {
                    auto readSts = resolveContainsCombeCount(containsCombeReadback, needsContainsCombeCount ? &containsCombeCount : nullptr);
                    if (readSts != RGY_ERR_NONE) {
                        return readSts;
                    }
                    writeContainsCombeDump("24p", outputTiming, containsCombeCount, containsCombeCount > 0, switchResult);
                    if (containsCombeCount > 0) {
                        markSwitchSingleFrameN60Range(outputTiming.start60, outputTiming.duration60);
                        outputTiming.duration60 = 1;
                        outputTiming.duration120 = 2;
                        outputTiming.numSourceFrames = 1;
                    }
                    return RGY_ERR_NONE;
                };
                if (auto debugOut = kfmDebugStageFrame(prm->kfm.debugStage, switchFlag, containsCombe, combeMask)) {
                    copyFramePropWithoutRes(debugOut, deint24);
                    debugOut->picstruct = RGY_PICSTRUCT_FRAME;
                    debugOut->flags = RGY_FRAME_FLAG_NONE;
                    out = debugOut;
                    outputEvent = maskEvent;
                    sts = resolveContainsCombeDuration();
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                } else {
                    sts = removeCombe24(out, deint24, super24, outputTiming.frame24Index, queue, removeWaitEvents, &outputEvent);
                    if (sts != RGY_ERR_NONE) {
                        resolveContainsCombeCount(containsCombeReadback, nullptr);
                        return sts;
                    }
                    sts = resolveContainsCombeDuration();
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    int patchN60 = -1;
                    if (patchCombe24Enabled) {
                        try {
                            static const int patchFieldIndex[4] = { 1, 3, 6, 8 };
                            const int frame24Cycle = outputTiming.frame24Index / 4;
                            const int frame24InCycle = outputTiming.frame24Index & 3;
                            const auto& patchResult = m_analyzerOutputResults[clamp(frame24Cycle, 0, (int)m_analyzerOutputResults.size() - 1)];
                            const auto frameInfo = m_analyzer->patterns().getFrame24(patchResult.pattern, outputTiming.frame24Index);
                            patchN60 = clamp(patchFieldIndex[frame24InCycle], frameInfo.fieldStartIndex, frameInfo.fieldStartIndex + frameInfo.numFields - 1) + frameInfo.cycleIndex * 10;
                        } catch (...) {
                            patchN60 = -1;
                        }
                    }
                    if (patchN60 >= 0 && containsCombeCount > 0) {
                        std::vector<RGYOpenCLEvent> patchWaitEvents = removeWaitEvents;
                        if (outputEvent() != nullptr) {
                            patchWaitEvents.push_back(outputEvent);
                        }
                        sts = ensureDeint60Range(patchN60, patchN60 + 1);
                        if (sts == RGY_ERR_MORE_DATA) {
                            m_workBufferIndex = savedWorkBufferIndex;
                            m_telecineSuperBufferIndex = savedTelecineSuperBufferIndex;
                            break;
                        }
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        const auto *deint60 = findDeint60Frame(patchN60, &patchWaitEvents);
                        if (deint60 && deint60->ptr[0]) {
                            const int patchIndex = m_patchCombeBufferIndex++ & 3;
                            if (!m_patchCombeFrames[patchIndex]
                                || m_patchCombeFrames[patchIndex]->frame.width != prm->frameOut.width
                                || m_patchCombeFrames[patchIndex]->frame.height != prm->frameOut.height
                                || m_patchCombeFrames[patchIndex]->frame.csp != prm->frameOut.csp) {
                                m_patchCombeFrames[patchIndex] = m_cl->createFrameBuffer(prm->frameOut);
                                if (!m_patchCombeFrames[patchIndex]) {
                                    AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM patch-combe frame.\n"));
                                    return RGY_ERR_MEMORY_ALLOC;
                                }
                            }
                            RGYOpenCLEvent patchEvent;
                            sts = patchCombe(&m_patchCombeFrames[patchIndex]->frame, out, deint60, combeMask, outputTiming.frame24Index, "patch-combe", queue, patchWaitEvents, &patchEvent);
                            if (sts != RGY_ERR_NONE) {
                                return sts;
                            }
                            out = &m_patchCombeFrames[patchIndex]->frame;
                            outputEvent = patchEvent;
                        }
                    }
                }
                if (prm->kfm.ucf && m_analyzer && !m_analyzerOutputResults.empty() && outputTiming.frame24Index >= 0) {
                    try {
                        const int frame24Cycle = outputTiming.frame24Index / 4;
                        const auto& ucfResult = m_analyzerOutputResults[clamp(frame24Cycle, 0, (int)m_analyzerOutputResults.size() - 1)];
                        const auto frameInfo = m_analyzer->patterns().getFrame24(ucfResult.pattern, outputTiming.frame24Index);
                        std::vector<RGYOpenCLEvent> ucfWaitEvents;
                        if (outputEvent() != nullptr) {
                            ucfWaitEvents.push_back(outputEvent);
                        }
                        const int lastUcfN60 = frameInfo.cycleIndex * 10 + frameInfo.fieldStartIndex + frameInfo.numFields - 2;
                        sts = resolveUcfNoiseResults((lastUcfN60 >> 1) + 1, queue);
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        const auto ucf24Plan = planUcfDecomb24Frame(frameInfo);
                        if (ucf24Plan.type == KFM_UCF24_SELECT_FRAME && ucf24Plan.n60 >= 0) {
                            sts = ensureUcfRtgmcRange(ucf24Plan.lane, ucf24Plan.n60, ucf24Plan.n60 + 1, queue);
                            if (sts == RGY_ERR_MORE_DATA) {
                                m_workBufferIndex = savedWorkBufferIndex;
                                m_telecineSuperBufferIndex = savedTelecineSuperBufferIndex;
                                break;
                            }
                            if (sts != RGY_ERR_NONE) {
                                return sts;
                            }
                        }
                        const auto ucf24 = selectUcfDecomb24Frame(frameInfo, out, &ucfWaitEvents);
                        if (ucf24.type == KFM_UCF24_SELECT_FRAME && ucf24.frame && ucf24.frame != out) {
                            auto ucfOut = nextWorkFrame();
                            if (!ucfOut) {
                                return RGY_ERR_INVALID_CALL;
                            }
                            RGYOpenCLEvent ucfEvent;
                            sts = m_cl->copyFrame(ucfOut, ucf24.frame, nullptr, queue, ucfWaitEvents, &ucfEvent, RGYFrameCopyMode::FRAME, "kfm.vfr.ucf24_copy");
                            if (sts != RGY_ERR_NONE) {
                                AddMessage(RGY_LOG_ERROR, _T("failed to copy KFM VFR UCF24 frame: %s.\n"), get_err_mes(sts));
                                return sts;
                            }
                            copyFramePropWithoutRes(ucfOut, out);
                            out = ucfOut;
                            outputEvent = ucfEvent;
                        } else if (ucf24.type == KFM_UCF24_SELECT_DWEAVE && ucf24.n60 >= 0) {
                            auto dweave = nextWorkFrame();
                            if (!dweave) {
                                return RGY_ERR_INVALID_CALL;
                            }
                            RGYOpenCLEvent dweaveEvent;
                            sts = renderDoubleWeaveFrame(dweave, ucf24.n60, 2, drain, queue, ucfWaitEvents, &dweaveEvent);
                            if (sts == RGY_ERR_MORE_DATA) {
                                sts = RGY_ERR_NONE;
                            } else if (sts != RGY_ERR_NONE) {
                                return sts;
                            } else {
                                const int ucfSuperIndex = m_telecineSuperBufferIndex++ & 1;
                                if (!m_telecineSuperFrames[ucfSuperIndex]) {
                                    auto superInfo = prm->frameOut;
                                    superInfo.width = std::max(1, superInfo.width >> 1);
                                    superInfo.height = std::max(1, superInfo.height >> 2);
                                    m_telecineSuperFrames[ucfSuperIndex] = m_cl->createFrameBuffer(superInfo);
                                    if (!m_telecineSuperFrames[ucfSuperIndex]) {
                                        AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM UCF24 dweave-super frame.\n"));
                                        return RGY_ERR_MEMORY_ALLOC;
                                    }
                                }
                                auto dweaveSuper = &m_telecineSuperFrames[ucfSuperIndex]->frame;
                                std::vector<RGYOpenCLEvent> dweaveSuperWaitEvents = ucfWaitEvents;
                                if (dweaveEvent() != nullptr) {
                                    dweaveSuperWaitEvents.push_back(dweaveEvent);
                                }
                                RGYOpenCLEvent dweaveSuperEvent;
                                sts = renderCleanSuperFields(dweaveSuper, ucf24.n60, ucf24.n60, ucf24.n60 >> 1, ucf24.n60, drain, queue, dweaveSuperWaitEvents, &dweaveSuperEvent);
                                if (sts == RGY_ERR_MORE_DATA) {
                                    sts = RGY_ERR_NONE;
                                } else if (sts != RGY_ERR_NONE) {
                                    return sts;
                                } else {
                                    writeFrameInfoDump("ucf24-dweave-super", dweaveSuper);
                                    sts = dumpStageFrame("ucf24-dweave-super", dweaveSuper, ucf24.n60, queue,
                                        (dweaveSuperEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ dweaveSuperEvent } : std::vector<RGYOpenCLEvent>());
                                    if (sts != RGY_ERR_NONE) {
                                        return sts;
                                    }
                                    auto ucfOut = nextWorkFrame();
                                    if (!ucfOut) {
                                        return RGY_ERR_INVALID_CALL;
                                    }
                                    std::vector<RGYOpenCLEvent> dweaveRemoveWaitEvents = ucfWaitEvents;
                                    if (dweaveEvent() != nullptr) {
                                        dweaveRemoveWaitEvents.push_back(dweaveEvent);
                                    }
                                    if (dweaveSuperEvent() != nullptr) {
                                        dweaveRemoveWaitEvents.push_back(dweaveSuperEvent);
                                    }
                                    RGYOpenCLEvent ucfEvent;
                                    sts = removeCombeFields(ucfOut, dweave, dweaveSuper, ucf24.n60, 2, ucf24.n60, "ucf24-dweave-remove-combe", queue, dweaveRemoveWaitEvents, &ucfEvent);
                                    if (sts != RGY_ERR_NONE) {
                                        return sts;
                                    }
                                    copyFramePropWithoutRes(ucfOut, out);
                                    out = ucfOut;
                                    outputEvent = ucfEvent;
                                }
                            }
                        }
                    } catch (...) {
                    }
                }
            } else if (outputTiming.baseType == KFM_FRAME_60) {
                std::vector<RGYOpenCLEvent> copyWaitEvents = wait_events;
                sts = ensureDeint60Range(outputTiming.start60, outputTiming.start60 + outputTiming.duration60);
                if (sts == RGY_ERR_MORE_DATA) {
                    break;
                }
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                const auto *deint60 = findDeint60Frame(outputTiming.start60, &copyWaitEvents);
                if (!deint60 || !deint60->ptr[0]) {
                    break;
                }
                if (prm->kfm.ucf) {
                    deint60 = selectUcfDecomb60Frame(outputTiming.start60, deint60, &copyWaitEvents);
                }
                out = nextWorkFrame();
                if (!out) {
                    return RGY_ERR_INVALID_CALL;
                }
                sts = m_cl->copyFrame(out, deint60, nullptr, queue, copyWaitEvents, &outputEvent, RGYFrameCopyMode::FRAME, "kfm.vfr.deint60_output");
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to copy KFM VFR deint60 frame: %s.\n"), get_err_mes(sts));
                    return sts;
                }
                copyFramePropWithoutRes(out, deint60);
            } else if (outputTiming.baseType == KFM_FRAME_30) {
                if (!source || !source->frame || !source->frame->frame.ptr[0]) {
                    break;
                }
                std::vector<RGYOpenCLEvent> deintWaitEvents = wait_events;
                if (source->event() != nullptr) {
                    deintWaitEvents.push_back(source->event);
                }
                auto deint30 = nextWorkFrame();
                out = nextWorkFrame();
                if (!deint30 || !out) {
                    return RGY_ERR_INVALID_CALL;
                }

                RGYOpenCLEvent deintEvent;
                sts = m_cl->copyFrame(deint30, &source->frame->frame, nullptr, queue, deintWaitEvents, &deintEvent, RGYFrameCopyMode::FRAME, "kfm.vfr.deint30_source");
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to copy KFM VFR deint30 frame: %s.\n"), get_err_mes(sts));
                    return sts;
                }
                copyFramePropWithoutRes(deint30, &source->frame->frame);
                deint30->picstruct = RGY_PICSTRUCT_FRAME;
                deint30->flags = RGY_FRAME_FLAG_NONE;
                attachSwitchFrameData(deint30, outputTiming, switchResult);
                writeFrameInfoDump("deint30", deint30, switchResult);

                const int superIndex = m_telecineSuperBufferIndex++ & 1;
                if (!m_telecineSuperFrames[superIndex]) {
                    auto superInfo = prm->frameOut;
                    superInfo.width = std::max(1, superInfo.width >> 1);
                    superInfo.height = std::max(1, superInfo.height >> 2);
                    m_telecineSuperFrames[superIndex] = m_cl->createFrameBuffer(superInfo);
                    if (!m_telecineSuperFrames[superIndex]) {
                        AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM super30 frame.\n"));
                        return RGY_ERR_MEMORY_ALLOC;
                    }
                }
                RGYOpenCLEvent superEvent;
                const auto superSts = renderSuper30(&m_telecineSuperFrames[superIndex]->frame, outputTiming.sourceIndex, drain, queue, deintWaitEvents, &superEvent);
                if (superSts != RGY_ERR_NONE && superSts != RGY_ERR_MORE_DATA) {
                    return superSts;
                }

                std::vector<RGYOpenCLEvent> copyWaitEvents;
                if (deintEvent() != nullptr) {
                    copyWaitEvents.push_back(deintEvent);
                }
                const bool patchCombe30Enabled = kfmDeint60BranchEnabled() && m_deint60Rtgmc;
                bool patched30 = false;
                if (superSts == RGY_ERR_NONE) {
                    std::vector<RGYOpenCLEvent> maskWaitEvents;
                    if (superEvent() != nullptr) {
                        maskWaitEvents.push_back(superEvent);
                    }
                    RGYFrameInfo *superPrev30 = &m_telecineSuperFrames[superIndex]->frame;
                    RGYFrameInfo *superNext30 = &m_telecineSuperFrames[superIndex]->frame;
                    auto ensureNeighborSuper = [&](int index, RGYFrameInfo **frame) -> RGY_ERR {
                        if (!m_telecineSuperNeighborFrames[index]
                            || m_telecineSuperNeighborFrames[index]->frame.width != m_telecineSuperFrames[superIndex]->frame.width
                            || m_telecineSuperNeighborFrames[index]->frame.height != m_telecineSuperFrames[superIndex]->frame.height
                            || m_telecineSuperNeighborFrames[index]->frame.csp != m_telecineSuperFrames[superIndex]->frame.csp) {
                            auto superInfo = m_telecineSuperFrames[superIndex]->frame;
                            m_telecineSuperNeighborFrames[index] = m_cl->createFrameBuffer(superInfo);
                            if (!m_telecineSuperNeighborFrames[index]) {
                                AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM super30 neighbor frame.\n"));
                                return RGY_ERR_MEMORY_ALLOC;
                            }
                        }
                        *frame = &m_telecineSuperNeighborFrames[index]->frame;
                        return RGY_ERR_NONE;
                    };
                    if (outputTiming.sourceIndex > 0) {
                        RGYFrameInfo *candidatePrev30 = nullptr;
                        RGYOpenCLEvent prevSuperEvent;
                        sts = ensureNeighborSuper(0, &candidatePrev30);
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        sts = renderSuper30(candidatePrev30, outputTiming.sourceIndex - 1, true, queue, deintWaitEvents, &prevSuperEvent);
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        superPrev30 = candidatePrev30;
                        if (prevSuperEvent() != nullptr) {
                            maskWaitEvents.push_back(prevSuperEvent);
                        }
                    }
                    {
                        RGYFrameInfo *candidateNext30 = nullptr;
                        RGYOpenCLEvent nextSuperEvent;
                        sts = ensureNeighborSuper(1, &candidateNext30);
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        sts = renderSuper30(candidateNext30, outputTiming.sourceIndex + 1, drain, queue, deintWaitEvents, &nextSuperEvent);
                        if (sts != RGY_ERR_NONE && sts != RGY_ERR_MORE_DATA) {
                            return sts;
                        }
                        if (sts == RGY_ERR_NONE) {
                            superNext30 = candidateNext30;
                            if (nextSuperEvent() != nullptr) {
                                maskWaitEvents.push_back(nextSuperEvent);
                            }
                        }
                    }
                    RGYFrameInfo *switchFlag = nullptr;
                    RGYFrameInfo *containsCombe = nullptr;
                    RGYFrameInfo *combeMask = nullptr;
                    sts = ensureMaskBranchFrames(&switchFlag, &containsCombe, &combeMask, &m_telecineSuperFrames[superIndex]->frame, _T("30p"));
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    RGYOpenCLEvent maskEvent;
                    cl_uint containsCombeCount = 0;
                    KfmContainsCombeReadback containsCombeReadback;
                    const bool needsContainsCombeCount = switchSingleFrameDurationEnabled() || patchCombe30Enabled;
                    sts = renderMaskBranch(switchFlag, containsCombe, combeMask, superPrev30, &m_telecineSuperFrames[superIndex]->frame, superNext30, "switch-flag30-min", "contains-combe30", "combe-mask30-min", queue, maskWaitEvents, &maskEvent, needsContainsCombeCount ? &containsCombeReadback : nullptr);
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    sts = resolveContainsCombeCount(containsCombeReadback, needsContainsCombeCount ? &containsCombeCount : nullptr);
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    writeContainsCombeDump("30p", outputTiming, containsCombeCount, containsCombeCount > 0, switchResult);
                    if (containsCombeCount > 0) {
                        markSwitchSingleFrameN60Range(outputTiming.start60, outputTiming.duration60);
                        outputTiming.duration60 = 1;
                        outputTiming.duration120 = 2;
                        outputTiming.numSourceFrames = 1;
                    }
                    if (maskEvent() != nullptr) {
                        copyWaitEvents.push_back(maskEvent);
                    }
                    if (patchCombe30Enabled && containsCombeCount > 0) {
                        std::vector<RGYOpenCLEvent> patchWaitEvents = copyWaitEvents;
                        const int patchN60 = outputTiming.sourceIndex * 2;
                        sts = ensureDeint60Range(patchN60, patchN60 + 1);
                        if (sts == RGY_ERR_MORE_DATA) {
                            break;
                        }
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        const auto *deint60 = findDeint60Frame(patchN60, &patchWaitEvents);
                        if (!deint60 || !deint60->ptr[0]) {
                            break;
                        }
                        sts = patchCombe(out, deint30, deint60, combeMask, outputTiming.sourceIndex, "patch-combe30", queue, patchWaitEvents, &outputEvent);
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        copyFramePropWithoutRes(out, deint30);
                        patched30 = true;
                    }
                }
                if (!patched30) {
                    const RGYFrameInfo *ucf30 = deint30;
                    if (prm->kfm.ucf) {
                        sts = resolveUcfNoiseResults(outputTiming.sourceIndex, queue);
                        if (sts != RGY_ERR_NONE) {
                            return sts;
                        }
                        ucf30 = selectUcfDecomb30Frame(outputTiming.sourceIndex, deint30, &copyWaitEvents);
                    }
                    sts = m_cl->copyFrame(out, ucf30, nullptr, queue, copyWaitEvents, &outputEvent, RGYFrameCopyMode::FRAME, "kfm.vfr.deint30_output");
                    if (sts != RGY_ERR_NONE) {
                        AddMessage(RGY_LOG_ERROR, _T("failed to copy KFM VFR deint30 output frame: %s.\n"), get_err_mes(sts));
                        return sts;
                    }
                    copyFramePropWithoutRes(out, ucf30);
                }
            } else {
                if (!source || !source->frame || !source->frame->frame.ptr[0]) {
                    break;
                }
                std::vector<RGYOpenCLEvent> copyWaitEvents = wait_events;
                if (source->event() != nullptr) {
                    copyWaitEvents.push_back(source->event);
                }
                out = nextWorkFrame();
                if (!out) {
                    return RGY_ERR_INVALID_CALL;
                }
                sts = m_cl->copyFrame(out, &source->frame->frame, nullptr, queue, copyWaitEvents, &outputEvent, RGYFrameCopyMode::FRAME, "kfm.vfr.fallback_output");
                if (sts != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to copy KFM VFR fallback frame: %s.\n"), get_err_mes(sts));
                    return sts;
                }
                copyFramePropWithoutRes(out, &source->frame->frame);
            }
            if (outputTiming.duration60 == 1) {
                outputStart120 = rawStart120(outputTiming);
                outputTiming.start120 = static_cast<int>(outputStart120);
                outputTiming.duration120 = 2;
            }
            const auto outputEnd120 = static_cast<int64_t>(outputTiming.start120) + outputTiming.duration120;
            const auto outputStartPts = sourcePtsFrom120(outputTiming.start120);
            const auto outputEndPts = sourcePtsFrom120(outputEnd120);
            out->timestamp = std::max(outputStartPts, m_nextSwitchPts);
            out->duration = std::max<int64_t>(1, outputEndPts - out->timestamp);
            out->picstruct = RGY_PICSTRUCT_FRAME;
            out->flags = RGY_FRAME_FLAG_NONE;
            attachSwitchFrameData(out, outputTiming, switchResult);
            sts = queueVfrOutputFrame(out, queue, outputEvent);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            m_nextSwitchPts = out->timestamp + out->duration;
            m_hasLastSwitchTiming = true;
            m_lastSwitchStart60 = outputTiming.start60;
            m_lastSwitchDuration60 = outputTiming.duration60;
            m_lastSwitchStart120 = outputStart120;
            m_lastSwitchIsFrame24 = outputTiming.isFrame24;
            m_nextSwitchN60 += outputTiming.duration60;
            sts = emitReadyPending(drain ? 0 : vfrOutputDelay);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if (*pOutputFrameNum == 0 && !m_pendingVfrOutputs.empty()) {
            sts = emitReadyPending(drain ? 0 : (int)m_pendingVfrOutputs.size() - 1);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if (drain && *pOutputFrameNum < maxOutputFrames && !m_pendingVfrOutputs.empty()) {
            sts = emitReadyPending(0);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if (drain && m_pendingVfrOutputs.empty() && (timings.empty() || m_nextSwitchN60 >= m_cachedSourceFrames * 2)) {
            writeSwitchTimingDump();
            if (*pOutputFrameNum == 0) {
                sts = drainNrFilter(ppOutputFrames, pOutputFrameNum, queue, event);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
        }
        return RGY_ERR_NONE;
    }

    if (prm->kfm.mode == VppKfmMode::P24) {
        auto sts = RGY_ERR_NONE;
        if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
            sts = analyzeAvailableSource(true, queue);
        } else {
            sts = cacheSourceFrame(pInputFrame, queue, wait_events);
        }
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr)
            ? drainDeint60Branch(queue)
            : runDeint60Branch(pInputFrame, queue, wait_events);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (prm->kfm.ucf && m_deint60Rtgmc) {
            for (auto &captured : m_deint60Rtgmc->getCapturedIntermediates()) {
                if (m_before60Rtgmc) m_before60Rtgmc->pushIntermediateInput(captured);
                if (m_after60Rtgmc) m_after60Rtgmc->pushIntermediateInput(captured);
            }
            m_deint60Rtgmc->clearCapturedIntermediates();
        }
        if (prm->kfm.ucf) {
            sts = runUcfRtgmcBranches(pInputFrame, queue, wait_events);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
                sts = resolveAllUcfNoiseResults(queue);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
        }

        *pOutputFrameNum = 0;
        const bool drain = pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr;
        const int maxOutputFrames = std::min<int>((int)m_frameBuf.size(), 4);
        while (*pOutputFrameNum < maxOutputFrames && m_nextTelecine24Frame < telecine24FrameCount(drain)) {
            auto deint24 = nextWorkFrame();
            auto out = nextWorkFrame();
            if (!deint24 || !out) {
                return RGY_ERR_INVALID_CALL;
            }

            const int superIndex = m_telecineSuperBufferIndex++ & 1;
            if (!m_telecineSuperFrames[superIndex]) {
                auto superInfo = prm->frameOut;
                superInfo.width = std::max(1, superInfo.width >> 1);
                superInfo.height = std::max(1, superInfo.height >> 2);
                m_telecineSuperFrames[superIndex] = m_cl->createFrameBuffer(superInfo);
                if (!m_telecineSuperFrames[superIndex]) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM telecine-super frame.\n"));
                    return RGY_ERR_MEMORY_ALLOC;
                }
            }
            auto super24 = &m_telecineSuperFrames[superIndex]->frame;

            RGYOpenCLEvent deintEvent;
            sts = renderTelecine24(deint24, m_nextTelecine24Frame, drain, queue, wait_events, &deintEvent);
            if (sts == RGY_ERR_MORE_DATA) {
                break;
            }
            if (sts != RGY_ERR_NONE) {
                return sts;
            }

            std::vector<RGYOpenCLEvent> superWaitEvents;
            if (deintEvent() != nullptr) {
                superWaitEvents.push_back(deintEvent);
            }
            RGYOpenCLEvent superEvent;
            sts = renderTelecineSuper24(super24, m_nextTelecine24Frame, drain, queue, superWaitEvents, &superEvent);
            if (sts == RGY_ERR_MORE_DATA) {
                break;
            }
            if (sts != RGY_ERR_NONE) {
                return sts;
            }

            std::vector<RGYOpenCLEvent> removeWaitEvents = superWaitEvents;
            if (superEvent() != nullptr) {
                removeWaitEvents.push_back(superEvent);
            }
            RGYFrameInfo *superPrev24 = super24;
            RGYFrameInfo *superNext24 = super24;
            std::vector<RGYOpenCLEvent> maskWaitEvents = removeWaitEvents;
            auto ensureNeighborSuper = [&](int index, RGYFrameInfo **frame) -> RGY_ERR {
                if (!m_telecineSuperNeighborFrames[index]
                    || m_telecineSuperNeighborFrames[index]->frame.width != super24->width
                    || m_telecineSuperNeighborFrames[index]->frame.height != super24->height
                    || m_telecineSuperNeighborFrames[index]->frame.csp != super24->csp) {
                    auto superInfo = *super24;
                    m_telecineSuperNeighborFrames[index] = m_cl->createFrameBuffer(superInfo);
                    if (!m_telecineSuperNeighborFrames[index]) {
                        AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM telecine-super neighbor frame.\n"));
                        return RGY_ERR_MEMORY_ALLOC;
                    }
                }
                *frame = &m_telecineSuperNeighborFrames[index]->frame;
                return RGY_ERR_NONE;
            };
            if (m_nextTelecine24Frame > 0) {
                RGYOpenCLEvent prevSuperEvent;
                sts = ensureNeighborSuper(0, &superPrev24);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                sts = renderTelecineSuper24(superPrev24, m_nextTelecine24Frame - 1, true, queue, superWaitEvents, &prevSuperEvent);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                if (prevSuperEvent() != nullptr) {
                    maskWaitEvents.push_back(prevSuperEvent);
                }
            }
            if (m_nextTelecine24Frame + 1 < telecine24FrameCount(drain)) {
                RGYOpenCLEvent nextSuperEvent;
                sts = ensureNeighborSuper(1, &superNext24);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                sts = renderTelecineSuper24(superNext24, m_nextTelecine24Frame + 1, drain, queue, superWaitEvents, &nextSuperEvent);
                if (sts == RGY_ERR_MORE_DATA) {
                    break;
                }
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                if (nextSuperEvent() != nullptr) {
                    maskWaitEvents.push_back(nextSuperEvent);
                }
            } else if (!drain) {
                break;
            }
            RGYFrameInfo *switchFlag = nullptr;
            RGYFrameInfo *containsCombe = nullptr;
            RGYFrameInfo *combeMask = nullptr;
            sts = ensureMaskBranchFrames(&switchFlag, &containsCombe, &combeMask, super24, _T("24p"));
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            RGYOpenCLEvent maskEvent;
            sts = renderMaskBranch(switchFlag, containsCombe, combeMask, superPrev24, super24, superNext24, "switch-flag-min", "contains-combe", "combe-mask-min", queue, maskWaitEvents, &maskEvent);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            if (maskEvent() != nullptr) {
                removeWaitEvents.push_back(maskEvent);
            }
            RGYOpenCLEvent outputEvent;
            if (auto debugOut = kfmDebugStageFrame(prm->kfm.debugStage, switchFlag, containsCombe, combeMask)) {
                copyFramePropWithoutRes(debugOut, deint24);
                debugOut->picstruct = RGY_PICSTRUCT_FRAME;
                debugOut->flags = RGY_FRAME_FLAG_NONE;
                out = debugOut;
                outputEvent = maskEvent;
            } else {
                sts = removeCombe24(out, deint24, super24, m_nextTelecine24Frame, queue, removeWaitEvents, &outputEvent);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                int patchN60 = -1;
                if (kfmDeint60BranchEnabled() && m_deint60Rtgmc && m_analyzer) {
                    try {
                        static const int patchFieldIndex[4] = { 1, 3, 6, 8 };
                        const int frame24Cycle = m_nextTelecine24Frame / 4;
                        const int frame24InCycle = m_nextTelecine24Frame & 3;
                        const auto& patchResult = m_analyzerOutputResults[clamp(frame24Cycle, 0, (int)m_analyzerOutputResults.size() - 1)];
                        const auto frameInfo = m_analyzer->patterns().getFrame24(patchResult.pattern, m_nextTelecine24Frame);
                        patchN60 = clamp(patchFieldIndex[frame24InCycle], frameInfo.fieldStartIndex, frameInfo.fieldStartIndex + frameInfo.numFields - 1) + frameInfo.cycleIndex * 10;
                    } catch (...) {
                        patchN60 = -1;
                    }
                }
                if (patchN60 >= 0) {
                    std::vector<RGYOpenCLEvent> patchWaitEvents = removeWaitEvents;
                    if (outputEvent() != nullptr) {
                        patchWaitEvents.push_back(outputEvent);
                    }
                    const auto *deint60 = findDeint60Frame(patchN60, &patchWaitEvents);
                    if (!deint60 || !deint60->ptr[0]) {
                        break;
                    }
                    const int patchIndex = m_patchCombeBufferIndex++ & 3;
                    if (!m_patchCombeFrames[patchIndex]
                        || m_patchCombeFrames[patchIndex]->frame.width != prm->frameOut.width
                        || m_patchCombeFrames[patchIndex]->frame.height != prm->frameOut.height
                        || m_patchCombeFrames[patchIndex]->frame.csp != prm->frameOut.csp) {
                        m_patchCombeFrames[patchIndex] = m_cl->createFrameBuffer(prm->frameOut);
                        if (!m_patchCombeFrames[patchIndex]) {
                            AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM patch-combe frame.\n"));
                            return RGY_ERR_MEMORY_ALLOC;
                        }
                    }
                    RGYOpenCLEvent patchEvent;
                    sts = patchCombe(&m_patchCombeFrames[patchIndex]->frame, out, deint60, combeMask, m_nextTelecine24Frame, "patch-combe", queue, patchWaitEvents, &patchEvent);
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    out = &m_patchCombeFrames[patchIndex]->frame;
                    outputEvent = patchEvent;
                }
            }
            if (prm->kfm.ucf && m_analyzer && !m_analyzerOutputResults.empty()) {
                try {
                    const int frame24Cycle = m_nextTelecine24Frame / 4;
                    const auto& ucfResult = m_analyzerOutputResults[clamp(frame24Cycle, 0, (int)m_analyzerOutputResults.size() - 1)];
                    const auto frameInfo = m_analyzer->patterns().getFrame24(ucfResult.pattern, m_nextTelecine24Frame);
                    std::vector<RGYOpenCLEvent> ucfWaitEvents;
                    if (outputEvent() != nullptr) {
                        ucfWaitEvents.push_back(outputEvent);
                    }
                    const int lastUcfN60 = frameInfo.cycleIndex * 10 + frameInfo.fieldStartIndex + frameInfo.numFields - 2;
                    sts = resolveUcfNoiseResults((lastUcfN60 >> 1) + 1, queue);
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                    const auto ucf24 = selectUcfDecomb24Frame(frameInfo, out, &ucfWaitEvents);
                    if (ucf24.type == KFM_UCF24_SELECT_FRAME && ucf24.frame && ucf24.frame != out) {
                        auto ucfOut = nextWorkFrame();
                        if (!ucfOut) {
                            return RGY_ERR_INVALID_CALL;
                        }
                        RGYOpenCLEvent ucfEvent;
                        sts = m_cl->copyFrame(ucfOut, ucf24.frame, nullptr, queue, ucfWaitEvents, &ucfEvent, RGYFrameCopyMode::FRAME, "kfm.p24.ucf24_copy");
                        if (sts != RGY_ERR_NONE) {
                            AddMessage(RGY_LOG_ERROR, _T("failed to copy KFM UCF24 frame: %s.\n"), get_err_mes(sts));
                            return sts;
                        }
                        copyFramePropWithoutRes(ucfOut, out);
                        out = ucfOut;
                        outputEvent = ucfEvent;
                    } else if (ucf24.type == KFM_UCF24_SELECT_DWEAVE && ucf24.n60 >= 0) {
                        auto dweave = nextWorkFrame();
                        if (!dweave) {
                            return RGY_ERR_INVALID_CALL;
                        }
                        RGYOpenCLEvent dweaveEvent;
                        sts = renderDoubleWeaveFrame(dweave, ucf24.n60, 2, drain, queue, ucfWaitEvents, &dweaveEvent);
                        if (sts == RGY_ERR_MORE_DATA) {
                            sts = RGY_ERR_NONE;
                        } else if (sts != RGY_ERR_NONE) {
                            return sts;
                        } else {
                            const int ucfSuperIndex = m_telecineSuperBufferIndex++ & 1;
                            if (!m_telecineSuperFrames[ucfSuperIndex]) {
                                auto superInfo = prm->frameOut;
                                superInfo.width = std::max(1, superInfo.width >> 1);
                                superInfo.height = std::max(1, superInfo.height >> 2);
                                m_telecineSuperFrames[ucfSuperIndex] = m_cl->createFrameBuffer(superInfo);
                                if (!m_telecineSuperFrames[ucfSuperIndex]) {
                                    AddMessage(RGY_LOG_ERROR, _T("failed to allocate KFM UCF24 dweave-super frame.\n"));
                                    return RGY_ERR_MEMORY_ALLOC;
                                }
                            }
                            auto dweaveSuper = &m_telecineSuperFrames[ucfSuperIndex]->frame;
                            std::vector<RGYOpenCLEvent> dweaveSuperWaitEvents = ucfWaitEvents;
                            if (dweaveEvent() != nullptr) {
                                dweaveSuperWaitEvents.push_back(dweaveEvent);
                            }
                            RGYOpenCLEvent dweaveSuperEvent;
                            sts = renderCleanSuperFields(dweaveSuper, ucf24.n60, ucf24.n60, ucf24.n60 >> 1, ucf24.n60, drain, queue, dweaveSuperWaitEvents, &dweaveSuperEvent);
                            if (sts == RGY_ERR_MORE_DATA) {
                                sts = RGY_ERR_NONE;
                            } else if (sts != RGY_ERR_NONE) {
                                return sts;
                            } else {
                                writeFrameInfoDump("ucf24-dweave-super", dweaveSuper);
                                sts = dumpStageFrame("ucf24-dweave-super", dweaveSuper, ucf24.n60, queue,
                                    (dweaveSuperEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ dweaveSuperEvent } : std::vector<RGYOpenCLEvent>());
                                if (sts != RGY_ERR_NONE) {
                                    return sts;
                                }
                                auto ucfOut = nextWorkFrame();
                                if (!ucfOut) {
                                    return RGY_ERR_INVALID_CALL;
                                }
                                std::vector<RGYOpenCLEvent> dweaveRemoveWaitEvents = ucfWaitEvents;
                                if (dweaveEvent() != nullptr) {
                                    dweaveRemoveWaitEvents.push_back(dweaveEvent);
                                }
                                if (dweaveSuperEvent() != nullptr) {
                                    dweaveRemoveWaitEvents.push_back(dweaveSuperEvent);
                                }
                                RGYOpenCLEvent ucfEvent;
                                sts = removeCombeFields(ucfOut, dweave, dweaveSuper, ucf24.n60, 2, ucf24.n60, "ucf24-dweave-remove-combe", queue, dweaveRemoveWaitEvents, &ucfEvent);
                                if (sts != RGY_ERR_NONE) {
                                    return sts;
                                }
                                copyFramePropWithoutRes(ucfOut, out);
                                out = ucfOut;
                                outputEvent = ucfEvent;
                            }
                        }
                    }
                } catch (...) {
                }
            }
            sts = emitOutputFrame(out, ppOutputFrames, pOutputFrameNum, queue, outputEvent, event);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            m_nextTelecine24Pts += std::max<int64_t>(1, out->duration);
            m_nextTelecine24Frame++;
        }
        if (drain && m_nextTelecine24Frame >= telecine24FrameCount(true)) {
            writeTelecine24DurationDump();
            if (*pOutputFrameNum == 0) {
                sts = drainNrFilter(ppOutputFrames, pOutputFrameNum, queue, event);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
        }
        return RGY_ERR_NONE;
    }

    if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
        auto sts = analyzeAvailableSource(true, queue);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (prm->kfm.ucf) {
            sts = runUcfRtgmcBranches(pInputFrame, queue, wait_events);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            sts = resolveAllUcfNoiseResults(queue);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        *pOutputFrameNum = 0;
        sts = drainNrFilter(ppOutputFrames, pOutputFrameNum, queue, event);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        return RGY_ERR_NONE;
    }

    auto sts = cacheSourceFrame(pInputFrame, queue, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (prm->kfm.ucf) {
        sts = runUcfRtgmcBranches(pInputFrame, queue, wait_events);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }

    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
    }
    *pOutputFrameNum = 1;

    RGYOpenCLEvent outputEvent;
    if (prm->kfm.ucf) {
        sts = copyUcfFrame(*prm, ppOutputFrames[0], pInputFrame, queue, wait_events, &outputEvent);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy input frame with UCF placeholder: %s.\n"), get_err_mes(sts));
            return sts;
        }
    } else {
        sts = m_cl->copyFrame(ppOutputFrames[0], pInputFrame, nullptr, queue, wait_events, &outputEvent, RGYFrameCopyMode::FRAME, "kfm.fallback_input_copy");
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy input frame: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    ppOutputFrames[0]->timestamp = pInputFrame->timestamp;
    ppOutputFrames[0]->duration = pInputFrame->duration;
    ppOutputFrames[0]->inputFrameId = pInputFrame->inputFrameId;
    ppOutputFrames[0]->picstruct = RGY_PICSTRUCT_FRAME;
    ppOutputFrames[0]->flags = RGY_FRAME_FLAG_NONE;
    ppOutputFrames[0]->dataList = pInputFrame->dataList;

    RGYFrameInfo *out = ppOutputFrames[0];
    *pOutputFrameNum = 0;
    sts = emitOutputFrame(out, ppOutputFrames, pOutputFrameNum, queue, outputEvent, event);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    return RGY_ERR_NONE;
}

void RGYFilterKfm::close() {
    if (m_cl && !m_pendingUcfNoiseResults.empty()) {
        auto sts = resolveAllUcfNoiseResults(m_cl->queue());
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_WARN, _T("failed to resolve pending KFM UCF noise results on close: %s.\n"), get_err_mes(sts));
            m_pendingUcfNoiseResults.clear();
        }
    }
    flushUcfNoiseResultDump();
    AddMessage(RGY_LOG_INFO, _T("KFM RTGMC feed count: deint60=%lld, before60=%lld, after60=%lld.\n"),
        (long long)m_deint60Lane.feedCount(), (long long)m_before60Lane.feedCount(), (long long)m_after60Lane.feedCount());
    m_rtgmc.reset();
    m_deint60Rtgmc.reset();
    m_before60Rtgmc.reset();
    m_after60Rtgmc.reset();
    m_nrFilter.reset();
    m_analyzer.reset();
    if (m_cl) {
        for (auto& source : m_sourceCache) {
            retireKfmSourceSlot(std::move(source.slot), m_cl->queue());
        }
    }
    m_sourceCache.clear();
    m_deint60Lane.init(this, nullptr, "deint60", _T("deint60 cache"), true);
    m_before60Lane.init(this, nullptr, "before60", _T("before60"), false);
    m_after60Lane.init(this, nullptr, "after60", _T("after60"), false);
    m_ucfNoiseCache.clear();
    clearKfmSourceSlotPool(true);
    if (m_kfmFramePool) {
        m_kfmFramePool->clear();
    }
    m_pendingUcfNoiseResults.clear();
    m_ucfNoiseResultBufPool.clear();
    m_ucfNoiseResultCache.clear();
    m_pendingVfrOutputs.clear();
    m_staticFlag.reset();
    for (auto& frame : m_staticWorkFrames) {
        frame.reset();
    }
    for (auto& flag : m_analyzeFlags) {
        flag.reset();
    }
    if (auto clearFMCountSts = clearPendingFMCounts(); clearFMCountSts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to clear KFM pending FMCount buffers: %s.\n"), get_err_mes(clearFMCountSts));
    }
    m_fmCountBufPool.clear();
    for (auto& raw : m_telecineSuperRaw) {
        raw.reset();
    }
    for (auto& frame : m_telecineSuperFrames) {
        frame.reset();
    }
    for (auto& frame : m_telecineSuperNeighborFrames) {
        frame.reset();
    }
    for (auto& frame : m_switchFlagFrames) {
        frame.reset();
    }
    for (auto& frame : m_containsCombeFrames) {
        frame.reset();
    }
    for (auto& frame : m_combeMaskFrames) {
        frame.reset();
    }
    for (auto& frame : m_patchCombeFrames) {
        frame.reset();
    }
    for (auto& frame : m_ucfNoiseFieldFrames) {
        frame.reset();
    }
    for (auto& frame : m_ucfNoiseGaussTmpFrames) {
        frame.reset();
    }
    for (auto& frame : m_ucfNoiseGaussFrames) {
        frame.reset();
    }
    for (auto& programs : m_ucfNoiseGaussVert) {
        for (auto& program : programs) {
            program = KfmUcfGaussProgram();
        }
    }
    for (auto& programs : m_ucfNoiseGaussHori) {
        for (auto& program : programs) {
            program = KfmUcfGaussProgram();
        }
    }
    for (auto& work : m_switchFlagWork) {
        work.reset();
    }
    m_switchFlagWorkEvent = RGYOpenCLEvent();
    m_containsCombeCount.reset();
    m_fmCountQueue.clear();
    if (m_fpResult) {
        fclose(m_fpResult);
        m_fpResult = nullptr;
    }
    if (m_fpFMCount) {
        fclose(m_fpFMCount);
        m_fpFMCount = nullptr;
    }
    if (m_fpTimecode) {
        fclose(m_fpTimecode);
        m_fpTimecode = nullptr;
    }
    if (m_fpFrameInfo) {
        fclose(m_fpFrameInfo);
        m_fpFrameInfo = nullptr;
    }
    if (m_fpContainsCombe) {
        fclose(m_fpContainsCombe);
        m_fpContainsCombe = nullptr;
    }
    if (m_fpUcfNoise) {
        fclose(m_fpUcfNoise);
        m_fpUcfNoise = nullptr;
    }
    m_switchDurationPath.clear();
    m_switchTimecodePath.clear();
    m_stageDumpDir.clear();
    m_analyzerOutputResults.clear();
    m_switchSingleFrameN60.clear();
    m_stageDumpFrameCounts.clear();
    m_stageDumpFrameIndices.clear();
    m_stageDumpTargetFrames.clear();
    m_pendingUcfNoiseDump = KfmUcfNoiseDumpRecord();
    m_hasLastAnalyzeResult = false;
    m_analyzerFinalized = false;
    m_switchTimingDumped = false;
    m_analyzeSourceFrames = 0;
    m_nextAnalyzeCycle = 0;
    m_nextFMCountSubmitCycle = 0;
    m_nextFMCountDumpFrame = 0;
    m_cachedSourceFrames = 0;
    m_deint60Lane.reset();
    m_before60Lane.reset();
    m_after60Lane.reset();
    m_nextSwitchN60 = 0;
    m_nextSwitchPts = 0;
    m_hasLastSwitchTiming = false;
    m_lastSwitchStart60 = 0;
    m_lastSwitchDuration60 = 0;
    m_lastSwitchStart120 = 0;
    m_lastSwitchIsFrame24 = false;
    m_nextTelecine24Frame = 0;
    m_nextTelecine24Pts = 0;
    m_telecineSuperBufferIndex = 0;
    m_maskBranchBufferIndex = 0;
    m_patchCombeBufferIndex = 0;
    m_stageDumpMaxFrames = 0;
    m_timecodeFrameIndex = 0;
    m_outputBufferIndex = 0;
    m_workBufferIndex = 0;
    m_workFrameBuf.clear();
    m_frameBuf.clear();
    for (auto& program : m_programs) {
        program.clear();
    }
    m_cl.reset();
}
