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
#include "rgy_opencl_perf.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <sstream>

namespace {
static uint64_t degrain_cl_perf_now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static uint64_t degrain_cl_perf_begin(const bool enabled) {
    return enabled ? degrain_cl_perf_now_ns() : 0;
}

static uint64_t degrain_cl_perf_end(const bool enabled, const uint64_t start_ns) {
    if (!enabled) {
        return 0;
    }
    const auto end_ns = degrain_cl_perf_now_ns();
    return (end_ns >= start_ns) ? end_ns - start_ns : 0;
}

constexpr int degrainAnalyzePad = 8;

uint32_t degrainAnalyzeFlags(const std::shared_ptr<RGYFilterParamDegrain> &prm, const bool usesAnalysisLuma, const bool includesChromaSad) {
    uint32_t flags = RGY_DEGRAIN_FRAME_META_FLAG_NONE;
    if (prm && usesAnalysisLuma) {
        flags |= RGY_DEGRAIN_FRAME_META_FLAG_ANALYSIS_LUMA;
    }
    if (prm && prm->degrain.levels >= 2) {
        flags |= RGY_DEGRAIN_FRAME_META_FLAG_LEVEL1_REFINE;
    }
    if (prm && includesChromaSad) {
        flags |= RGY_DEGRAIN_FRAME_META_FLAG_CHROMA_SAD;
    }
    return flags;
}

int degrainPelLog2(const int pel) {
    return (pel >= 4) ? 2 : ((pel >= 2) ? 1 : 0);
}

bool degrainEnvFlagEnabled(const char *name) {
    const auto value = std::getenv(name);
    return value && value[0] == '1' && value[1] == '\0';
}

bool degrainEnvFlagNotDisabled(const char *name) {
    const auto value = std::getenv(name);
    return !(value && value[0] == '0' && value[1] == '\0');
}

bool degrainMotionSearchProfileEnabled() {
    static const bool enabled = degrainEnvFlagEnabled("QSVENC_DEGRAIN_MOTION_SEARCH_PROFILE");
    return enabled;
}

bool degrainMotionSearchSubgroupEnabled(const std::shared_ptr<RGYOpenCLContext> &cl) {
    static const int envMode = []() {
        const auto value = std::getenv("QSVENC_DEGRAIN_MOTION_SEARCH_SUBGROUP");
        if (value && value[0] == '0' && value[1] == '\0') {
            return 0;
        }
        if (value && value[0] == '1' && value[1] == '\0') {
            return 1;
        }
        return -1;
    }();
    if (envMode == 0) {
        return false;
    }
    if (envMode == 1) {
        return true;
    }
    if (!cl || !cl->platform()) {
        return false;
    }
    const auto support = cl->platform()->checkSubGroupSupport(cl->queue().devid());
    return support == RGYOpenCLSubGroupSupport::STD22
        || support == RGYOpenCLSubGroupSupport::STD20KHR;
}

bool degrainMotionSearchSubgroupDirectReduceEnabled() {
    static const bool enabled = degrainEnvFlagNotDisabled("QSVENC_DEGRAIN_MOTION_SEARCH_SUBGROUP_DIRECT_REDUCE");
    return enabled;
}

bool degrainMotionSearchRefLocalCacheEnabled() {
    static const bool enabled = degrainEnvFlagNotDisabled("QSVENC_DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE");
    return enabled;
}

bool degrainMotionSearchLazyRefWindowEnabled() {
    static const bool enabled = degrainEnvFlagNotDisabled("QSVENC_DEGRAIN_MOTION_SEARCH_LAZY_REF_WINDOW");
    return enabled;
}

bool degrainMotionSearchSpatialReusePreviousSadEnabled() {
    static const bool enabled = degrainEnvFlagNotDisabled("QSVENC_DEGRAIN_MOTION_SEARCH_SPATIAL_REUSE_PREVIOUS_SAD");
    return enabled;
}

bool degrainMotionSearchBuildOptionsSubgroupEnabled(const std::string &options) {
    return options.find("DEGRAIN_MOTION_SEARCH_SUBGROUP=1") != std::string::npos;
}

bool degrainMotionSearchBuildOptionsHasSubgroupSize(const std::string &options) {
    return options.find("DEGRAIN_MOTION_SEARCH_SUBGROUP_SIZE=") != std::string::npos;
}

size_t degrainMotionSearchLocalSize(const RGYDegrainBlockLayout &layout) {
    return (size_t)std::max(layout.blockSize, 1) * 8u;
}

RGYWorkSize degrainMotionSearchLocalWorkSize(const RGYDegrainBlockLayout &layout) {
    return RGYWorkSize(degrainMotionSearchLocalSize(layout));
}

RGYWorkSize degrainMotionSearchGlobalWorkSize(const RGYDegrainBlockLayout &layout) {
    return RGYWorkSize(layout.blockCount() * degrainMotionSearchLocalSize(layout));
}

struct RGYDegrainAnalyzeChromaPlanes {
    RGYFrameInfo curU;
    RGYFrameInfo curV;
    std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> refU;
    std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> refV;
    int width;
    int height;
    int enable;
};

bool degrainChromaAnalysisFrameSupported(const RGYFrameInfo *frame) {
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

bool degrainValidAnalysisPlane(const RGYFrameInfo &plane) {
    return plane.ptr[0] != nullptr && plane.pitch[0] > 0 && plane.width > 0 && plane.height > 0;
}

RGYDegrainAnalyzeChromaPlanes degrainMakeAnalyzeChromaPlanes(
    const RGYFilterDegrainFrameSet &analysisFrames,
    const RGYFrameInfo &planeCur,
    const std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> &refPlanes,
    const int temporalDirections,
    const int requiredDelta,
    const bool requestChroma,
    const bool usedSearchLuma) {
    RGYDegrainAnalyzeChromaPlanes planes = {};
    planes.curU = getPlane(analysisFrames.cur, RGY_PLANE_U);
    planes.curV = getPlane(analysisFrames.cur, RGY_PLANE_V);
    planes.refU.fill(planeCur);
    planes.refV.fill(planeCur);

    bool enable = requestChroma
        && degrainChromaAnalysisFrameSupported(analysisFrames.cur)
        && degrainValidAnalysisPlane(planes.curU)
        && degrainValidAnalysisPlane(planes.curV);
    for (int delta = 1; delta <= requiredDelta; delta++) {
        const int backwardIndex = rgy_degrain_ref_index(delta, false);
        const int forwardIndex = rgy_degrain_ref_index(delta, true);
        const auto backward = analysisFrames.backwardRef(delta);
        const auto forward = analysisFrames.forwardRef(delta);
        planes.refU[backwardIndex] = getPlane(backward, RGY_PLANE_U);
        planes.refV[backwardIndex] = getPlane(backward, RGY_PLANE_V);
        planes.refU[forwardIndex] = getPlane(forward, RGY_PLANE_U);
        planes.refV[forwardIndex] = getPlane(forward, RGY_PLANE_V);
        enable = enable
            && degrainChromaAnalysisFrameSupported(backward)
            && degrainChromaAnalysisFrameSupported(forward);
    }
    for (int dir = 0; dir < temporalDirections; dir++) {
        enable = enable
            && degrainValidAnalysisPlane(planes.refU[dir])
            && degrainValidAnalysisPlane(planes.refV[dir]);
    }
    (void)usedSearchLuma;

    if (!degrainValidAnalysisPlane(planes.curU)) {
        planes.curU = planeCur;
    }
    if (!degrainValidAnalysisPlane(planes.curV)) {
        planes.curV = planeCur;
    }
    for (int dir = 0; dir < temporalDirections; dir++) {
        if (!degrainValidAnalysisPlane(planes.refU[dir])) {
            planes.refU[dir] = refPlanes[dir];
        }
        if (!degrainValidAnalysisPlane(planes.refV[dir])) {
            planes.refV[dir] = refPlanes[dir];
        }
    }
    planes.width = planes.curU.width;
    planes.height = planes.curU.height;
    planes.enable = enable ? 1 : 0;
    return planes;
}

bool allocDegrainMotionSearchWorkspaceBuffer(
    const std::shared_ptr<RGYOpenCLContext> &cl,
    std::unique_ptr<RGYCLBuf> &buf,
    size_t &currentBytes,
    const size_t requiredBytes) {
    currentBytes = requiredBytes;
    if (requiredBytes == 0) {
        buf.reset();
        return true;
    }
    if (!buf || buf->size() != requiredBytes) {
        buf = cl->createBuffer(requiredBytes, CL_MEM_READ_WRITE);
        if (!buf) {
            return false;
        }
    }
    return true;
}
}

RGYDegrainMotionSearchConfig rgy_degrain_make_motion_search_config(
    const RGYFrameInfo &frameInfo,
    const VppDegrain &degrain,
    const RGYDegrainBlockLayout &layout,
    const int level,
    const int pad) {
    RGYDegrainMotionSearchConfig cfg;
    cfg.pixelBytes = (RGY_CSP_BIT_DEPTH[frameInfo.csp] > 8) ? 2 : 1;
    cfg.bitDepth = RGY_CSP_BIT_DEPTH[frameInfo.csp];
    cfg.pixelMax = (cfg.bitDepth >= 16) ? std::numeric_limits<uint16_t>::max() : ((1 << cfg.bitDepth) - 1);
    cfg.blockSize = layout.blockSize;
    cfg.searchMode = layout.search;
    cfg.pel = degrain.pel;
    cfg.chroma = 0;
    cfg.cpuEmu = 1;
    cfg.searchParam = degrain.searchParam;
    cfg.refs = layout.temporalDirections;
    cfg.trueMotion = degrain.trueMotion ? 1 : 0;
    cfg.globalMotion = degrain.globalMotion ? 1 : 0;
    cfg.subpelInterp = degrain.subpelInterp;
    cfg.width = frameInfo.width;
    cfg.height = frameInfo.height;
    cfg.blocksX = layout.blocksX;
    cfg.blocksY = layout.blocksY;
    cfg.step = layout.step;
    cfg.overlap = layout.overlap;
    cfg.pad = pad;
    cfg.motionCostScale = degrain.lambda;
    auto scaledDegrain = degrain;
    scaledDegrain.blksize = layout.blockSize;
    scaledDegrain.overlap = layout.overlap;
    cfg.lowSadWeightScale = (int)rgy_degrain_scale_sad_threshold(scaledDegrain, frameInfo, degrain.lsad);
    cfg.zeroCandidateCostScale = degrain.pnew;
    cfg.frameAverageCandidateCostScale = 0;
    cfg.predictorCandidateCostScale = degrain.plevel;
    cfg.newCandidateCostScale = degrain.pnew;
    cfg.level = level;
    return cfg;
}

std::string makeDegrainMotionSearchBuildOptions(const RGYDegrainMotionSearchConfig &cfg) {
    std::ostringstream options;
    options
        << "-D TypePixel=" << ((cfg.pixelBytes > 1) ? "ushort" : "uchar")
        << " -D DEGRAIN_PIXEL_TYPE=" << ((cfg.pixelBytes > 1) ? "ushort" : "uchar")
        << " -D DEGRAIN_PIXEL_BYTES=" << cfg.pixelBytes
        << " -D DEGRAIN_BIT_DEPTH=" << cfg.bitDepth
        << " -D DEGRAIN_PIXEL_MAX=" << cfg.pixelMax
        << " -D DEGRAIN_BLK_SIZE=" << cfg.blockSize
        << " -D DEGRAIN_BLOCK_SIZE=" << cfg.blockSize
        << " -D DEGRAIN_SEARCH=" << cfg.searchMode
        << " -D DEGRAIN_SEARCH_MODE=" << cfg.searchMode
        << " -D DEGRAIN_PEL=" << cfg.pel
        << " -D DEGRAIN_NPEL=" << cfg.pel
        << " -D DEGRAIN_CHROMA=" << (cfg.chroma ? 1 : 0)
        << " -D DEGRAIN_CPU_EMU=" << (cfg.cpuEmu ? 1 : 0)
        << " -D DEGRAIN_SEARCH_PARAM=" << cfg.searchParam
        << " -D DEGRAIN_REFS=" << cfg.refs
        << " -D DEGRAIN_TRUE_MOTION=" << (cfg.trueMotion ? 1 : 0)
        << " -D DEGRAIN_GLOBAL_MOTION=" << (cfg.globalMotion ? 1 : 0)
        << " -D DEGRAIN_SUBPEL_INTERP=" << cfg.subpelInterp
        << " -D DEGRAIN_WIDTH=" << cfg.width
        << " -D DEGRAIN_HEIGHT=" << cfg.height
        << " -D DEGRAIN_BLOCKS_X=" << cfg.blocksX
        << " -D DEGRAIN_BLOCKS_Y=" << cfg.blocksY
        << " -D DEGRAIN_STEP=" << cfg.step
        << " -D DEGRAIN_OVERLAP=" << cfg.overlap
        << " -D DEGRAIN_PAD=" << cfg.pad
        << " -D DEGRAIN_MOTION_COST_SCALE=" << cfg.motionCostScale
        << " -D DEGRAIN_LOW_SAD_WEIGHT_SCALE=" << cfg.lowSadWeightScale
        << " -D DEGRAIN_ZERO_CANDIDATE_COST_SCALE=" << cfg.zeroCandidateCostScale
        << " -D DEGRAIN_FRAME_AVERAGE_CANDIDATE_COST_SCALE=" << cfg.frameAverageCandidateCostScale
        << " -D DEGRAIN_PREDICTOR_CANDIDATE_COST_SCALE=" << cfg.predictorCandidateCostScale
        << " -D DEGRAIN_NEW_CANDIDATE_COST_SCALE=" << cfg.newCandidateCostScale
        << " -D DEGRAIN_LEVEL=" << cfg.level
        << " -D DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE=" << (degrainMotionSearchRefLocalCacheEnabled() ? 1 : 0)
        << " -D DEGRAIN_MOTION_SEARCH_LAZY_REF_WINDOW=" << (degrainMotionSearchLazyRefWindowEnabled() ? 1 : 0)
        << " -D DEGRAIN_MOTION_SEARCH_SPATIAL_REUSE_PREVIOUS_SAD=" << (degrainMotionSearchSpatialReusePreviousSadEnabled() ? 1 : 0);
    return options.str();
}

RGY_ERR RGYFilterDegrain::allocAnalysisBuffers(const std::shared_ptr<RGYFilterParamDegrain> &prm) {
    m_analysis.mode = prm->degrain.mode;
    m_analysis.layout = rgy_degrain_make_block_layout(prm->frameOut, prm->degrain);
    m_analysis.layoutLevel1 = rgy_degrain_make_pyramid_block_layout(prm->frameOut, prm->degrain);
    for (auto &analysisLumaEvent : m_analysis.analysisLumaEvents) {
        analysisLumaEvent.reset();
    }
    m_analysis.analysisLumaEvent.reset();
    m_analysis.event.reset();

    if (!modeRequiresAnalysis(prm->degrain.mode)) {
        m_analysis.mv.reset();
        m_analysis.sad.reset();
        m_analysis.windowRampY.reset();
        m_analysis.windowRampC.reset();
        m_analysis.temporalMixPlanY.reset();
        m_analysis.temporalMixPlanC.reset();
        m_analysis.temporalMixPrior.reset();
        m_analysis.motionSearchWorkspace.reset();
        for (auto &luma : m_analysis.analysisLuma) {
            luma.reset();
        }
        for (auto &frame : m_analysis.analysisLumaFrame) {
            frame = RGYFrameInfo();
        }
        m_analysis.analysisLumaFrameNumbers.fill(-1);
        for (auto &analysisLumaEvent : m_analysis.analysisLumaEvents) {
            analysisLumaEvent.reset();
        }
        for (auto &luma : m_analysis.lumaLevel1) {
            luma.reset();
        }
        m_analysis.mvBytes = 0;
        m_analysis.sadBytes = 0;
        m_analysis.analysisLumaBytes = 0;
        m_analysis.lumaLevel1Bytes = 0;
        m_analysis.analysisLumaWidth = 0;
        m_analysis.analysisLumaHeight = 0;
        m_analysis.analysisLumaPitch = 0;
        m_analysis.analysisLumaGeneratedUntil = -1;
        m_analysis.lumaLevel1Width = 0;
        m_analysis.lumaLevel1Height = 0;
        m_analysis.lumaLevel1Pitch = 0;
        m_analysis.lastFrameIndex = -1;
        m_analysis.lastInputFrameId = -1;
        return RGY_ERR_NONE;
    }

    const bool binomial = (prm->degrain.binomial < 0)
        ? (prm->degrain.stage != VppDegrainStage::TR2)
        : (prm->degrain.binomial != 0);
    const auto temporalMixPrior = degrainBuildTemporalMixPriorTable(m_analysis.layout.temporalDirections, binomial);
    m_analysis.temporalMixPrior = m_cl->copyDataToBuffer(temporalMixPrior.data(),
        temporalMixPrior.size() * sizeof(temporalMixPrior[0]), CL_MEM_READ_ONLY, m_cl->queue().get());
    if (!m_analysis.temporalMixPrior || m_analysis.temporalMixPrior->mem() == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain temporal mix prior table buffer.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }

    m_analysis.mvBytes = rgy_degrain_mv_bytes(m_analysis.layout);
    m_analysis.sadBytes = rgy_degrain_sad_bytes(m_analysis.layout);
    if (!m_analysis.mv || m_analysis.mv->size() != m_analysis.mvBytes) {
        m_analysis.mv = m_cl->createBuffer(m_analysis.mvBytes, CL_MEM_READ_WRITE);
        if (!m_analysis.mv) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain MV buffer.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    if (!m_analysis.sad || m_analysis.sad->size() != m_analysis.sadBytes) {
        m_analysis.sad = m_cl->createBuffer(m_analysis.sadBytes, CL_MEM_READ_WRITE);
        if (!m_analysis.sad) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain SAD buffer.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    const auto motionSearchConfig = rgy_degrain_make_motion_search_config(prm->frameOut, prm->degrain, m_analysis.layout, 0, degrainAnalyzePad);
    auto motionSearchConfigLevel1 = rgy_degrain_make_motion_search_config(prm->frameOut, prm->degrain, m_analysis.layoutLevel1, 1, degrainAnalyzePad >> 1);
    motionSearchConfigLevel1.width = std::max(1, (prm->frameOut.width + 1) / 2);
    motionSearchConfigLevel1.height = std::max(1, (prm->frameOut.height + 1) / 2);
    auto &motionSearchWorkspace = m_analysis.motionSearchWorkspace;
    motionSearchWorkspace.buildOptionsLevel0 = makeDegrainMotionSearchBuildOptions(motionSearchConfig);
    motionSearchWorkspace.buildOptionsLevel1 = makeDegrainMotionSearchBuildOptions(motionSearchConfigLevel1);
    if (degrainMotionSearchSubgroupEnabled(m_cl)) {
        const auto subgroupOptions = strsprintf(" -cl-std=CL2.0 -D DEGRAIN_MOTION_SEARCH_SUBGROUP=1 -D DEGRAIN_MOTION_SEARCH_SUBGROUP_DIRECT_REDUCE=%d",
            degrainMotionSearchSubgroupDirectReduceEnabled() ? 1 : 0);
        motionSearchWorkspace.buildOptionsLevel0 += subgroupOptions;
        motionSearchWorkspace.buildOptionsLevel1 += subgroupOptions;
    }
    auto allocLevelWorkspace = [&](RGYDegrainMotionSearchLevelWorkspace &levelWorkspace, const RGYDegrainBlockLayout &layout, const TCHAR *levelName) {
        const size_t planeCount = (size_t)layout.temporalDirections;
        const size_t blockCount = layout.blockCount();
        const size_t vectorCount = (2 + blockCount) * planeCount;
        const size_t finalVectorCount = blockCount * planeCount;
        const size_t sadCount = blockCount * planeCount;
        if (!allocDegrainMotionSearchWorkspaceBuffer(m_cl, levelWorkspace.vectors, levelWorkspace.vectorsBytes,
            vectorCount * RGYDegrainMotionSearchWorkspace::VECTOR_BYTES)) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain motion search %s vectors workspace buffer.\n"), levelName);
            return RGY_ERR_MEMORY_ALLOC;
        }
        if (!allocDegrainMotionSearchWorkspaceBuffer(m_cl, levelWorkspace.vectorsPrev, levelWorkspace.vectorsPrevBytes,
            vectorCount * RGYDegrainMotionSearchWorkspace::VECTOR_BYTES)) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain motion search %s prev vectors workspace buffer.\n"), levelName);
            return RGY_ERR_MEMORY_ALLOC;
        }
        if (!allocDegrainMotionSearchWorkspaceBuffer(m_cl, levelWorkspace.vectorsFinal, levelWorkspace.vectorsFinalBytes,
            finalVectorCount * RGYDegrainMotionSearchWorkspace::VECTOR_BYTES)) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain motion search %s final vectors workspace buffer.\n"), levelName);
            return RGY_ERR_MEMORY_ALLOC;
        }
        if (!allocDegrainMotionSearchWorkspaceBuffer(m_cl, levelWorkspace.sads, levelWorkspace.sadsBytes,
            sadCount * RGYDegrainMotionSearchWorkspace::SAD_BYTES)) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain motion search %s sads workspace buffer.\n"), levelName);
            return RGY_ERR_MEMORY_ALLOC;
        }
        return RGY_ERR_NONE;
    };
    auto err = allocLevelWorkspace(motionSearchWorkspace.level0, m_analysis.layout, _T("level0"));
    if (err != RGY_ERR_NONE) {
        return err;
    }
    if (prm->degrain.levels > 1) {
        err = allocLevelWorkspace(motionSearchWorkspace.level1, m_analysis.layoutLevel1, _T("level1"));
        if (err != RGY_ERR_NONE) {
            return err;
        }
    } else {
        motionSearchWorkspace.level1.reset();
    }
    const size_t motionSearchFrameAverageVectorCount = (size_t)m_analysis.layout.temporalDirections;
    if (!allocDegrainMotionSearchWorkspaceBuffer(m_cl, motionSearchWorkspace.frameAverageMV, motionSearchWorkspace.frameAverageMVBytes,
        motionSearchFrameAverageVectorCount * RGYDegrainMotionSearchWorkspace::FRAME_AVERAGE_MV_BYTES)) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain motion search frameAverageMV workspace buffer.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelBytes = (bitdepth > 8) ? 2 : 1;
    if (degrainRequiresAnalysisLumaCache(prm->degrain)) {
        m_analysis.analysisLumaWidth = prm->frameOut.width;
        m_analysis.analysisLumaHeight = prm->frameOut.height;
        m_analysis.analysisLumaPitch = m_analysis.analysisLumaWidth * pixelBytes;
        m_analysis.analysisLumaBytes = (size_t)m_analysis.analysisLumaPitch * (size_t)m_analysis.analysisLumaHeight;
        const auto analysisCsp = degrainAnalysisLumaCsp(prm->frameOut);
        for (int i = 0; i < (int)m_analysis.analysisLuma.size(); i++) {
            auto &luma = m_analysis.analysisLuma[i];
            if (!luma || luma->size() != m_analysis.analysisLumaBytes) {
                luma = m_cl->createBuffer(m_analysis.analysisLumaBytes, CL_MEM_READ_WRITE);
                if (!luma) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain analysis luma buffer.\n"));
                    return RGY_ERR_MEMORY_ALLOC;
                }
            }
            auto &frame = m_analysis.analysisLumaFrame[i];
            frame = RGYFrameInfo(m_analysis.analysisLumaWidth, m_analysis.analysisLumaHeight, analysisCsp, bitdepth, RGY_PICSTRUCT_FRAME, RGY_MEM_TYPE_GPU);
            frame.ptr[0] = reinterpret_cast<uint8_t *>(m_analysis.analysisLuma[i]->mem());
            frame.pitch[0] = m_analysis.analysisLumaPitch;
        }
        m_analysis.analysisLumaFrameNumbers.fill(-1);
        for (auto &analysisLumaEvent : m_analysis.analysisLumaEvents) {
            analysisLumaEvent.reset();
        }
        m_analysis.analysisLumaGeneratedUntil = -1;
    } else {
        for (auto &luma : m_analysis.analysisLuma) {
            luma.reset();
        }
        for (auto &frame : m_analysis.analysisLumaFrame) {
            frame = RGYFrameInfo();
        }
        m_analysis.analysisLumaFrameNumbers.fill(-1);
        for (auto &analysisLumaEvent : m_analysis.analysisLumaEvents) {
            analysisLumaEvent.reset();
        }
        m_analysis.analysisLumaBytes = 0;
        m_analysis.analysisLumaWidth = 0;
        m_analysis.analysisLumaHeight = 0;
        m_analysis.analysisLumaPitch = 0;
        m_analysis.analysisLumaGeneratedUntil = -1;
    }
    m_analysis.lumaLevel1Width = std::max(1, (prm->frameOut.width + 1) / 2);
    m_analysis.lumaLevel1Height = std::max(1, (prm->frameOut.height + 1) / 2);
    m_analysis.lumaLevel1Pitch = m_analysis.lumaLevel1Width * pixelBytes;
    m_analysis.lumaLevel1Bytes = (size_t)m_analysis.lumaLevel1Pitch * (size_t)m_analysis.lumaLevel1Height;
    const int requiredLevel1Frames = degrainLevel1FrameCount(m_analysis.layout.temporalDirections);
    for (int i = 0; i < (int)m_analysis.lumaLevel1.size(); i++) {
        auto &luma = m_analysis.lumaLevel1[i];
        if (i >= requiredLevel1Frames) {
            luma.reset();
            continue;
        }
        if (!luma || luma->size() != m_analysis.lumaLevel1Bytes) {
            luma = m_cl->createBuffer(m_analysis.lumaLevel1Bytes, CL_MEM_READ_WRITE);
            if (!luma) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain level1 luma buffer.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
    }
    return RGY_ERR_NONE;
}

const RGYFrameInfo *RGYFilterDegrain::resolveAnalysisLumaSourceFrame(const int frameIndex) const {
    if (m_inputCount <= 0) {
        return nullptr;
    }
    const int clampedFrame = clamp(frameIndex, 0, m_inputCount - 1);
    return &m_cacheFrames[cacheIndex(clampedFrame)]->frame;
}

RGYFilterDegrainFrameSet RGYFilterDegrain::resolveAnalysisFrameSet(const int currentFrame) const {
    auto frames = resolveFrameSet(currentFrame);
    if (!useAnalysisLumaCache()) {
        return frames;
    }

    const auto analysisFrame = [this](const int frameIndex) -> const RGYFrameInfo * {
        const int slot = analysisCacheIndex(frameIndex);
        return (m_analysis.analysisLumaFrameNumbers[slot] == frameIndex) ? &m_analysis.analysisLumaFrame[slot] : nullptr;
    };
    frames.cur = analysisFrame(currentFrame);
    for (int delta = 1; delta <= RGY_DEGRAIN_MAX_DELTA; delta++) {
        const int backwardFrame = frames.backwardRefInRange(delta) ? (currentFrame + delta) : currentFrame;
        const int forwardFrame = frames.forwardRefInRange(delta) ? (currentFrame - delta) : currentFrame;
        frames.backward[delta - 1] = analysisFrame(backwardFrame);
        frames.forward[delta - 1] = analysisFrame(forwardFrame);
    }
    return frames;
}

RGY_ERR RGYFilterDegrain::generateAnalysisLumaFrame(const int centerFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDegrain>(m_param);
    if (!prm || !useAnalysisLumaCache()) {
        return RGY_ERR_NONE;
    }
    if (centerFrame < 0 || centerFrame >= m_inputCount) {
        return RGY_ERR_NONE;
    }

    RGYFilterDegrainFrameSet srcFrames = {};
    srcFrames.backward[1] = resolveAnalysisLumaSourceFrame(centerFrame + 2);
    srcFrames.backward[0] = resolveAnalysisLumaSourceFrame(centerFrame + 1);
    srcFrames.cur = resolveAnalysisLumaSourceFrame(centerFrame);
    srcFrames.forward[0] = resolveAnalysisLumaSourceFrame(centerFrame - 1);
    srcFrames.forward[1] = resolveAnalysisLumaSourceFrame(centerFrame - 2);
    if (!srcFrames.backwardRef(2) || !srcFrames.backwardRef(1) || !srcFrames.cur || !srcFrames.forwardRef(1) || !srcFrames.forwardRef(2)) {
        AddMessage(RGY_LOG_ERROR, _T("degrain analysis luma source frames are not ready.\n"));
        return RGY_ERR_INVALID_CALL;
    }

    const auto planePrev2 = getPlane(srcFrames.forwardRef(2), RGY_PLANE_Y);
    const auto planePrev = getPlane(srcFrames.forwardRef(1), RGY_PLANE_Y);
    const auto planeCur = getPlane(srcFrames.cur, RGY_PLANE_Y);
    const auto planeNext = getPlane(srcFrames.backwardRef(1), RGY_PLANE_Y);
    const auto planeNext2 = getPlane(srcFrames.backwardRef(2), RGY_PLANE_Y);
    if (planePrev2.ptr[0] == nullptr || planePrev.ptr[0] == nullptr || planeCur.ptr[0] == nullptr
        || planeNext.ptr[0] == nullptr || planeNext2.ptr[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("degrain analysis luma requires valid source luma planes.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    const int srcPitch = planeCur.pitch[0];
    if (planePrev2.pitch[0] != srcPitch || planePrev.pitch[0] != srcPitch
        || planeNext.pitch[0] != srcPitch || planeNext2.pitch[0] != srcPitch) {
        AddMessage(RGY_LOG_ERROR,
            _T("degrain analysis luma pitch mismatch: prev2=%d, prev=%d, cur=%d, next=%d, next2=%d.\n"),
            planePrev2.pitch[0], planePrev.pitch[0], srcPitch, planeNext.pitch[0], planeNext2.pitch[0]);
        return RGY_ERR_INVALID_PARAM;
    }

    const int slot = analysisCacheIndex(centerFrame);
    RGYOpenCLEvent lumaEvent;
    auto err = m_degrain.get()->kernel("kernel_degrain_temporal_smooth_luma").config(
        queue,
        RGYWorkSize(DEGRAIN_DEBUG_BLOCK_X, DEGRAIN_DEBUG_BLOCK_Y),
        RGYWorkSize(m_analysis.analysisLumaWidth, m_analysis.analysisLumaHeight),
        wait_events,
        &lumaEvent).launch(
            (cl_mem)planePrev2.ptr[0],
            (cl_mem)planePrev.ptr[0],
            (cl_mem)planeCur.ptr[0],
            (cl_mem)planeNext.ptr[0],
            (cl_mem)planeNext2.ptr[0],
            srcPitch,
            m_analysis.analysisLuma[slot]->mem(),
            m_analysis.analysisLumaPitch,
            planeCur.width, planeCur.height,
            prm->degrain.tr0,
            prm->degrain.searchRefine,
            prm->degrain.rep0);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to generate degrain analysis luma: %s.\n"), get_err_mes(err));
        return err;
    }

    auto &analysisFrame = m_analysis.analysisLumaFrame[slot];
    copyFramePropWithoutRes(&analysisFrame, srcFrames.cur);
    analysisFrame.width = m_analysis.analysisLumaWidth;
    analysisFrame.height = m_analysis.analysisLumaHeight;
    m_analysis.analysisLumaFrameNumbers[slot] = centerFrame;
    m_analysis.analysisLumaGeneratedUntil = centerFrame;
    m_analysis.analysisLumaEvents[slot] = lumaEvent;
    m_analysis.analysisLumaEvent = lumaEvent;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDegrain::ensureAnalysisLumaGenerated(int targetFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    if (!useAnalysisLumaCache()) {
        return RGY_ERR_NONE;
    }
    if (m_inputCount <= 0) {
        return RGY_ERR_NONE;
    }

    const int clampedTargetFrame = std::min(targetFrame, m_inputCount - 1);
    if (clampedTargetFrame < 0) {
        return RGY_ERR_NONE;
    }

    const int firstFrame = m_analysis.analysisLumaGeneratedUntil + 1;
    auto chainedWaitEvents = wait_events;
    for (int frame = firstFrame; frame <= clampedTargetFrame; frame++) {
        const auto err = generateAnalysisLumaFrame(frame, queue, chainedWaitEvents);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        chainedWaitEvents.clear();
        if (m_analysis.analysisLumaEvent() != nullptr) {
            chainedWaitEvents.push_back(m_analysis.analysisLumaEvent);
        }
    }
    return RGY_ERR_NONE;
}

void RGYFilterDegrain::clearFrameAnalysisData() {
    m_boundAnalyzeResult = RGYDegrainAnalyzeResult();
    m_frameAnalysisData.reset();
    m_frameAnalysisLayout = {};
}

RGYDegrainAnalyzeResult RGYFilterDegrain::analyzeResult() const {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDegrain>(m_param);
    RGYDegrainAnalyzeResult result;
    if (!prm || !modeRequiresAnalysis(prm->degrain.mode) || !m_analysis.mv || !m_analysis.sad || m_analysis.event() == nullptr) {
        return result;
    }
    result.flags = degrainAnalyzeFlags(prm, useAnalysisLumaCache() || m_lastAnalysisUsedSearchLuma, m_lastAnalysisIncludedChroma);
    result.layout = m_analysis.layout;
    result.mv = m_analysis.mv.get();
    result.sad = m_analysis.sad.get();
    result.event = m_analysis.event;
    result.frameIndex = m_analysis.lastFrameIndex;
    result.inputFrameId = m_analysis.lastInputFrameId;
    result.timestamp = m_analysis.lastTimestamp;
    result.duration = m_analysis.lastDuration;
    result.availabilityDisableRefs = m_analysis.lastAvailabilityDisableRefs;
    return result;
}

RGYDegrainAnalyzeResultSet RGYFilterDegrain::analyzeResultSet() const {
    RGYDegrainAnalyzeResultSet resultSet;
    const auto baseResult = analyzeResult();
    if (!baseResult.valid()) {
        return resultSet;
    }
    const int maxDelta = std::min(RGY_DEGRAIN_MAX_DELTA, std::max(1, baseResult.layout.temporalDirections / 2));
    for (int delta = 1; delta <= maxDelta; delta++) {
        auto slot = baseResult;
        slot.layout.temporalDirections = rgy_degrain_temporal_direction_count(delta);
        resultSet.slots[delta] = slot;
    }
    return resultSet;
}

bool RGYFilterDegrain::setDirectAnalyzeResult(const RGYDegrainAnalyzeResult &result) {
    RGYDegrainAnalyzeResultSet resultSet;
    if (result.valid() && result.hasFrameIdentity()) {
        const int maxDelta = std::min(RGY_DEGRAIN_MAX_DELTA, std::max(1, result.layout.temporalDirections / 2));
        for (int delta = 1; delta <= maxDelta; delta++) {
            auto slot = result;
            slot.layout.temporalDirections = rgy_degrain_temporal_direction_count(delta);
            resultSet.slots[delta] = slot;
        }
    }
    return setDirectAnalyzeResultSet(resultSet);
}

bool RGYFilterDegrain::setDirectAnalyzeResultSet(const RGYDegrainAnalyzeResultSet &resultSet) {
    m_directAnalyzeResultSet = resultSet;
    return m_directAnalyzeResultSet.valid(requestedDelta());
}

void RGYFilterDegrain::clearDirectAnalyzeResult() {
    m_directAnalyzeResultSet = RGYDegrainAnalyzeResultSet();
}

bool RGYFilterDegrain::validateAnalyzeResultFrame(const RGYDegrainAnalyzeResult &result, const RGYFrameInfo *frame, const int currentFrame, const TCHAR *sourceName, const bool requireFrameIndex) {
    if (!frame || !sourceName) {
        return false;
    }
    if (!result.hasFrameIdentity()) {
        AddMessage(RGY_LOG_DEBUG, _T("degrain %s MV/SAD frame identity is missing; falling back to frame data/local analysis.\n"), sourceName);
        return false;
    }
    if ((requireFrameIndex && result.frameIndex != currentFrame)
        || result.inputFrameId != frame->inputFrameId
        || result.timestamp != frame->timestamp
        || result.duration != frame->duration) {
        AddMessage(RGY_LOG_ERROR,
            _T("degrain %s MV/SAD frame mismatch; expected frameIndex=%d, inputFrameId=%d, timestamp=%lld, duration=%lld, got frameIndex=%d, inputFrameId=%d, timestamp=%lld, duration=%lld.\n"),
            sourceName, currentFrame, frame->inputFrameId, (long long)frame->timestamp, (long long)frame->duration,
            result.frameIndex, result.inputFrameId, (long long)result.timestamp, (long long)result.duration);
        assert((!requireFrameIndex || result.frameIndex == currentFrame)
            && result.inputFrameId == frame->inputFrameId
            && result.timestamp == frame->timestamp
            && result.duration == frame->duration);
        return false;
    }
    return true;
}

bool RGYFilterDegrain::bindDirectAnalyzeResult(const RGYFrameInfo *frame, const int currentFrame, RGYOpenCLQueue &queue) {
    const auto result = m_directAnalyzeResultSet.get(requestedDelta());
    if (!result) {
        return false;
    }
    if (!validateAnalyzeResultFrame(*result, frame, currentFrame, _T("direct"), true)) {
        return false;
    }
    if (!rgy_degrain_layout_equal(result->layout, m_analysis.layout)) {
        AddMessage(RGY_LOG_DEBUG, _T("degrain direct MV/SAD layout mismatch; falling back to frame data/local analysis.\n"));
        return false;
    }
    m_boundAnalyzeResult = *result;
    m_frameAnalysisLayout = result->layout;
    logAnalyzeBinding(_T("direct"), frame, *result);
    logAnalysisSamples(_T("direct"), frame, queue);
    return true;
}

bool RGYFilterDegrain::bindFrameAnalysisData(const RGYFrameInfo *frame, const int currentFrame, RGYOpenCLQueue &queue) {
    clearFrameAnalysisData();
    auto frameAnalysis = rgy_degrain_get_frame_data(frame);
    if (frameAnalysis) {
        const auto result = frameAnalysis->analyzeResult();
        const auto layout = result.layout;
        if (result.hasFrameIdentity() && !validateAnalyzeResultFrame(result, frame, currentFrame, _T("attached"), false)) {
            return false;
        }
        if (!rgy_degrain_layout_equal(layout, m_analysis.layout)) {
            AddMessage(RGY_LOG_DEBUG, _T("degrain attached MV/SAD layout mismatch; falling back to local analysis.\n"));
            return false;
        }
        m_boundAnalyzeResult = result;
        m_frameAnalysisLayout = layout;
        m_frameAnalysisData = frameAnalysis;
        logAnalyzeBinding(_T("attached"), frame, result);
        logAnalysisSamples(_T("attached"), frame, queue);
        return true;
    }
    return bindDirectAnalyzeResult(frame, currentFrame, queue);
}

RGYCLBuf *RGYFilterDegrain::analysisMV() const {
    return m_boundAnalyzeResult.valid() ? m_boundAnalyzeResult.mv : m_analysis.mv.get();
}

RGYCLBuf *RGYFilterDegrain::analysisSAD() const {
    return m_boundAnalyzeResult.valid() ? m_boundAnalyzeResult.sad : m_analysis.sad.get();
}

const RGYDegrainBlockLayout &RGYFilterDegrain::analysisLayout() const {
    return m_boundAnalyzeResult.valid() ? m_frameAnalysisLayout : m_analysis.layout;
}

const RGYOpenCLEvent &RGYFilterDegrain::analysisEvent() const {
    return m_boundAnalyzeResult.valid() ? m_boundAnalyzeResult.event : m_analysis.event;
}

RGYDegrainRefDisableArray RGYFilterDegrain::analysisAvailabilityDisableRefs(const RGYFilterDegrainFrameSet &frames) const {
    return m_boundAnalyzeResult.valid() ? m_boundAnalyzeResult.availabilityDisableRefs : degrainReferenceAvailability(frames);
}

RGY_ERR RGYFilterDegrain::attachAnalysisData(const RGYFrameInfo *sourceFrame, RGYFrameInfo *outputFrame,
    const int currentFrame, RGYOpenCLQueue &queue, const RGYOpenCLEvent &frameCopyEvent, RGYOpenCLEvent *event) {
    if (!sourceFrame || !outputFrame || !m_analysis.mv || !m_analysis.sad) {
        return RGY_ERR_INVALID_PARAM;
    }

    auto mv = (m_sideDataBufferPool)
        ? m_sideDataBufferPool->acquire(m_analysis.mvBytes, CL_MEM_READ_WRITE)
        : m_cl->createBuffer(m_analysis.mvBytes, CL_MEM_READ_WRITE);
    if (!mv) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain frame MV side data buffer.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    auto sad = (m_sideDataBufferPool)
        ? m_sideDataBufferPool->acquire(m_analysis.sadBytes, CL_MEM_READ_WRITE)
        : m_cl->createBuffer(m_analysis.sadBytes, CL_MEM_READ_WRITE);
    if (!sad) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate degrain frame SAD side data buffer.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }

    std::vector<RGYOpenCLEvent> mvWaitEvents;
    if (frameCopyEvent() != nullptr) {
        mvWaitEvents.push_back(frameCopyEvent);
    }
    if (m_analysis.event() != nullptr) {
        mvWaitEvents.push_back(m_analysis.event);
    }
    auto& perf_collector = RGYOpenCLPerfCollector::instance();
    const bool perf_enabled = perf_collector.isEnabled();
    const auto mvWaitList = degrainWaitEventList(mvWaitEvents);
    RGYOpenCLEvent mvCopyEvent;
    const auto mvCopyStart = degrain_cl_perf_begin(perf_enabled);
    auto clerr = clEnqueueCopyBuffer(
        queue.get(),
        m_analysis.mv->mem(),
        mv->mem(),
        0, 0,
        m_analysis.mvBytes,
        (cl_uint)mvWaitList.size(),
        mvWaitList.data(),
        mvCopyEvent.reset_ptr());
    const auto mvCopyHostTime = degrain_cl_perf_end(perf_enabled, mvCopyStart);
    if (perf_enabled && clerr == CL_SUCCESS) {
        perf_collector.recordCommand("clEnqueueCopyBuffer:degrain.mv_side_data", m_analysis.mvBytes, mvCopyHostTime, mvCopyEvent,
            mvCopyStart, mvCopyStart + mvCopyHostTime, (uint64_t)(uintptr_t)queue.get());
    }
    auto err = err_cl_to_rgy(clerr);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy degrain MV side data buffer: %s.\n"), get_err_mes(err));
        return err;
    }

    const auto sadWaitList = degrainWaitEventList({ mvCopyEvent });
    RGYOpenCLEvent sadCopyEvent;
    const auto sadCopyStart = degrain_cl_perf_begin(perf_enabled);
    clerr = clEnqueueCopyBuffer(
        queue.get(),
        m_analysis.sad->mem(),
        sad->mem(),
        0, 0,
        m_analysis.sadBytes,
        (cl_uint)sadWaitList.size(),
        sadWaitList.data(),
        sadCopyEvent.reset_ptr());
    const auto sadCopyHostTime = degrain_cl_perf_end(perf_enabled, sadCopyStart);
    if (perf_enabled && clerr == CL_SUCCESS) {
        perf_collector.recordCommand("clEnqueueCopyBuffer:degrain.sad_side_data", m_analysis.sadBytes, sadCopyHostTime, sadCopyEvent,
            sadCopyStart, sadCopyStart + sadCopyHostTime, (uint64_t)(uintptr_t)queue.get());
    }
    err = err_cl_to_rgy(clerr);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy degrain SAD side data buffer: %s.\n"), get_err_mes(err));
        return err;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamDegrain>(m_param);
    const uint32_t flags = degrainAnalyzeFlags(prm, useAnalysisLumaCache() || m_lastAnalysisUsedSearchLuma, m_lastAnalysisIncludedChroma);

    auto frameData = std::make_shared<RGYFrameDataDegrain>(
        rgy_degrain_make_frame_meta_header(m_analysis.layout, flags),
        std::move(mv),
        std::move(sad),
        sadCopyEvent,
        currentFrame,
        sourceFrame->inputFrameId,
        sourceFrame->timestamp,
        sourceFrame->duration,
        m_analysis.lastAvailabilityDisableRefs,
        m_sideDataBufferPool);
    rgy_degrain_erase_frame_data(outputFrame->dataList);
    outputFrame->dataList.push_back(frameData);
    if (event) {
        *event = sadCopyEvent;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDegrain::prepareAnalysisStateMotionSearch(const RGYFrameInfo &planeCur, const std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> &refPlanes,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDegrain>(m_param);
    if (!prm) {
        return RGY_ERR_UNSUPPORTED;
    }
    auto &ws = m_analysis.motionSearchWorkspace;
    auto programL0 = getDegrainMotionSearchProgram(ws.buildOptionsLevel0);
    auto programL1 = getDegrainMotionSearchProgram(ws.buildOptionsLevel1);
    if (!programL0 || !programL1) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build degrain motion search program.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    auto specializeSubgroupSize = [&](RGYOpenCLProgram *program, std::string &buildOptions, const RGYDegrainBlockLayout &layout, const TCHAR *levelName) {
        if (!program || !degrainMotionSearchBuildOptionsSubgroupEnabled(buildOptions)
            || degrainMotionSearchBuildOptionsHasSubgroupSize(buildOptions)
            || layout.blockCount() == 0 || layout.blocksX <= 0) {
            return program;
        }
        const auto local = degrainMotionSearchLocalWorkSize(layout);
        const auto globalParallel = degrainMotionSearchGlobalWorkSize(layout);
        const auto subgroupSizeSearchParallel = program->kernel("kernel_degrain_mv_search_parallel").config(queue, local, globalParallel).subGroupSize();
        const auto subgroupSizeSpatialRefine = program->kernel("kernel_degrain_mv_spatial_refine").config(queue, local, globalParallel).subGroupSize();
        if (subgroupSizeSearchParallel == 0
            || subgroupSizeSearchParallel != subgroupSizeSpatialRefine) {
            AddMessage(RGY_LOG_DEBUG,
                _T("degrain motion search %s subgroup size specialization skipped: search_parallel=%d, spatial_refine=%d.\n"),
                levelName, (int)subgroupSizeSearchParallel, (int)subgroupSizeSpatialRefine);
            return program;
        }
        const auto specializedOptions = buildOptions + strsprintf(" -D DEGRAIN_MOTION_SEARCH_SUBGROUP_SIZE=%d", (int)subgroupSizeSearchParallel);
        auto specializedProgram = getDegrainMotionSearchProgram(specializedOptions);
        if (!specializedProgram) {
            AddMessage(RGY_LOG_DEBUG, _T("degrain motion search %s subgroup size specialization build failed; using generic subgroup build.\n"), levelName);
            return program;
        }
        AddMessage(RGY_LOG_DEBUG, _T("degrain motion search %s subgroup size specialized: %d.\n"), levelName, (int)subgroupSizeSearchParallel);
        buildOptions = specializedOptions;
        return specializedProgram;
    };
    programL0 = specializeSubgroupSize(programL0, ws.buildOptionsLevel0, m_analysis.layout, _T("level0"));
    programL1 = specializeSubgroupSize(programL1, ws.buildOptionsLevel1, m_analysis.layoutLevel1, _T("level1"));
    if (!ws.level0.vectors || !ws.level0.vectorsPrev || !ws.level0.vectorsFinal || !ws.level0.sads
        || !ws.level1.vectors || !ws.level1.vectorsPrev || !ws.level1.vectorsFinal || !ws.level1.sads
        || !ws.frameAverageMV || !m_analysis.mv || !m_analysis.sad) {
        AddMessage(RGY_LOG_ERROR, _T("degrain motion search workspace is not ready.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    auto spatialRefineCount = [&](const int level) {
        if (prm->degrain.mvSpatialRefine >= 0) {
            return prm->degrain.mvSpatialRefine;
        }
        const int innerLevel = (prm->degrain.levels > 1) ? 1 : 0;
        return (level == innerLevel) ? 1 : 0;
    };
    constexpr int vectorSentinelCount = 2;
    auto copyMotionSearchVectors = [&](RGYCLBuf *src, const size_t srcVectorOffset, RGYCLBuf *dst, const size_t dstVectorOffset,
        const int vectorCount, const std::vector<RGYOpenCLEvent> &waitEvents, RGYOpenCLEvent *copyEvent, const TCHAR *stage) {
        const auto waitList = degrainWaitEventList(waitEvents);
        auto& perf_collector = RGYOpenCLPerfCollector::instance();
        const bool perf_enabled = perf_collector.isEnabled();
        const auto bytes = (size_t)vectorCount * RGYDegrainMotionSearchWorkspace::VECTOR_BYTES;
        const auto host_start = degrain_cl_perf_begin(perf_enabled);
        const auto clerr = clEnqueueCopyBuffer(
            queue.get(),
            src->mem(),
            dst->mem(),
            srcVectorOffset * RGYDegrainMotionSearchWorkspace::VECTOR_BYTES,
            dstVectorOffset * RGYDegrainMotionSearchWorkspace::VECTOR_BYTES,
            bytes,
            (cl_uint)waitList.size(),
            waitList.data(),
            copyEvent->reset_ptr());
        const auto host_time = degrain_cl_perf_end(perf_enabled, host_start);
        if (perf_enabled && clerr == CL_SUCCESS) {
            perf_collector.recordCommand("clEnqueueCopyBuffer:degrain.motion_search_vectors", bytes, host_time, *copyEvent,
                host_start, host_start + host_time, (uint64_t)(uintptr_t)queue.get());
        }
        auto err = err_cl_to_rgy(clerr);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy degrain motion search %s vectors: %s.\n"), stage, get_err_mes(err));
        }
        return err;
    };

    using ProfileClock = std::chrono::steady_clock;
    const bool profileEnabled = degrainMotionSearchProfileEnabled();
    const auto profileTotalStart = profileEnabled ? ProfileClock::now() : ProfileClock::time_point();
    double profileDownsampleMs = 0.0;
    double profileInitConstVecMs = 0.0;
    double profileLevel1SeedMs = 0.0;
    double profileLevel1SearchMs = 0.0;
    double profileLevel1ExportSadMs = 0.0;
    double profileInterpolateMs = 0.0;
    double profileLevel0SearchMs = 0.0;
    double profileLevel0ExportSadMs = 0.0;
    const auto profileNow = [&]() {
        return profileEnabled ? ProfileClock::now() : ProfileClock::time_point();
    };
    const auto profileElapsedMs = [](const ProfileClock::time_point &start, const ProfileClock::time_point &end) {
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    };
    const auto profileFinishStep = [&](const TCHAR *stepName, double &totalMs, const ProfileClock::time_point &start, const int dir) {
        if (!profileEnabled) {
            return RGY_ERR_NONE;
        }
        auto err = queue.finish();
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to finish degrain motion search profile step %s: %s.\n"), stepName, get_err_mes(err));
            return err;
        }
        const auto elapsedMs = profileElapsedMs(start, ProfileClock::now());
        totalMs += elapsedMs;
        if (dir >= 0) {
            AddMessage(RGY_LOG_DEBUG, _T("degrain motion search profile refdir=%d %s: %.3f ms.\n"), dir, stepName, elapsedMs);
        } else {
            AddMessage(RGY_LOG_DEBUG, _T("degrain motion search profile %s: %.3f ms.\n"), stepName, elapsedMs);
        }
        return RGY_ERR_NONE;
    };

    const auto planeMem = [](const RGYFrameInfo &plane) { return (cl_mem)plane.ptr[0]; };
    const int refs = m_analysis.layout.temporalDirections;
    const int level1FrameCount = degrainLevel1FrameCount(refs);
    for (int i = 0; i < level1FrameCount; i++) {
        if (!m_analysis.lumaLevel1[i]) {
            AddMessage(RGY_LOG_ERROR, _T("degrain motion search level1 luma workspace is not ready.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }

    RGYOpenCLEvent frameAverageMVEvent;
    {
        const cl_int zeroFrameAverageMV[2] = { 0, 0 };
        const auto fillWaitList = degrainWaitEventList(wait_events);
        auto& perf_collector = RGYOpenCLPerfCollector::instance();
        const bool perf_enabled = perf_collector.isEnabled();
        const auto host_start = degrain_cl_perf_begin(perf_enabled);
        const auto clerr = clEnqueueFillBuffer(
            queue.get(),
            ws.frameAverageMV->mem(),
            zeroFrameAverageMV,
            sizeof(zeroFrameAverageMV),
            0,
            ws.frameAverageMVBytes,
            (cl_uint)fillWaitList.size(),
            fillWaitList.data(),
            frameAverageMVEvent.reset_ptr());
        const auto host_time = degrain_cl_perf_end(perf_enabled, host_start);
        if (perf_enabled && clerr == CL_SUCCESS) {
            perf_collector.recordCommand("clEnqueueFillBuffer:degrain.frame_average_mv", ws.frameAverageMVBytes, host_time, frameAverageMVEvent,
                host_start, host_start + host_time, (uint64_t)(uintptr_t)queue.get());
        }
        auto err = err_cl_to_rgy(clerr);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to clear degrain motion search frameAverageMV buffer: %s.\n"), get_err_mes(err));
            return err;
        }
    }

    std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_LEVEL1_LUMA_FRAMES> level0Planes = {};
    level0Planes[0] = planeCur;
    for (int dir = 0; dir < refs; dir++) {
        level0Planes[dir + 1] = refPlanes[dir];
    }
    std::vector<RGYOpenCLEvent> downsampleEvents(level1FrameCount);
    for (int i = 0; i < level1FrameCount; i++) {
        const auto profileStepStart = profileNow();
        auto err = m_degrain.get()->kernel("kernel_degrain_downsample_luma2x").config(
            queue,
            RGYWorkSize(DEGRAIN_DEBUG_BLOCK_X, DEGRAIN_DEBUG_BLOCK_Y),
            RGYWorkSize(m_analysis.lumaLevel1Width, m_analysis.lumaLevel1Height),
            wait_events,
            &downsampleEvents[i]).launch(
                (cl_mem)level0Planes[i].ptr[0], level0Planes[i].pitch[0],
                m_analysis.lumaLevel1[i]->mem(), m_analysis.lumaLevel1Pitch,
                level0Planes[i].width, level0Planes[i].height,
                m_analysis.lumaLevel1Width, m_analysis.lumaLevel1Height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to downsample degrain motion search level1 luma: %s.\n"), get_err_mes(err));
            return err;
        }
        err = profileFinishStep(_T("downsample"), profileDownsampleMs, profileStepStart, -1);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }

    const auto blockCount0 = (int)m_analysis.layout.blockCount();
    const auto blockCount1 = (int)m_analysis.layoutLevel1.blockCount();
    const auto searchLocal0 = degrainMotionSearchLocalWorkSize(m_analysis.layout);
    const auto searchGlobal0 = degrainMotionSearchGlobalWorkSize(m_analysis.layout);
    const auto searchLocal1 = degrainMotionSearchLocalWorkSize(m_analysis.layoutLevel1);
    const auto searchGlobal1 = degrainMotionSearchGlobalWorkSize(m_analysis.layoutLevel1);
    const int planeStride0 = 2 + blockCount0;
    const int planeStride1 = 2 + blockCount1;
    const auto levelPlaneBase = [](const int dir, const int planeStride) { return dir * planeStride; };
    const auto blockPlaneBase = [](const int dir, const int blockCount) { return dir * blockCount; };

    RGYOpenCLEvent initLevel1Event;
    auto profileStepStart = profileNow();
    auto err = programL1->kernel("kernel_degrain_mv_seed_anchor_vectors").config(
        queue, RGYWorkSize(64), RGYWorkSize(refs), { frameAverageMVEvent }, &initLevel1Event).launch(
            ws.level1.vectors->mem(),
            ws.frameAverageMV->mem(),
            0,
            planeStride1,
            refs);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialize degrain motion search level1 const vectors: %s.\n"), get_err_mes(err));
        return err;
    }
    err = profileFinishStep(_T("init const vec level1"), profileInitConstVecMs, profileStepStart, -1);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    RGYOpenCLEvent initLevel0Event;
    profileStepStart = profileNow();
    err = programL0->kernel("kernel_degrain_mv_seed_anchor_vectors").config(
        queue, RGYWorkSize(64), RGYWorkSize(refs), { frameAverageMVEvent }, &initLevel0Event).launch(
            ws.level0.vectors->mem(),
            ws.frameAverageMV->mem(),
            0,
            planeStride0,
            refs);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialize degrain motion search level0 const vectors: %s.\n"), get_err_mes(err));
        return err;
    }
    err = profileFinishStep(_T("init const vec level0"), profileInitConstVecMs, profileStepStart, -1);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    RGYOpenCLEvent previousEvent;
    for (int dir = 0; dir < refs; dir++) {
        const int planeBase1 = levelPlaneBase(dir, planeStride1);
        const int blockBase1 = blockPlaneBase(dir, blockCount1);

        std::vector<RGYOpenCLEvent> seedLevel1Wait = { initLevel1Event };
        if (previousEvent() != nullptr) {
            seedLevel1Wait.push_back(previousEvent);
        }
        RGYOpenCLEvent seedLevel1Event;
        profileStepStart = profileNow();
        err = programL1->kernel("kernel_degrain_mv_seed_zero_vectors").config(
            queue, RGYWorkSize(256), RGYWorkSize(blockCount1), seedLevel1Wait, &seedLevel1Event).launch(
                ws.level1.vectors->mem(),
                ws.level1.vectorsPrev->mem(),
                ws.level1.sads->mem(),
                planeBase1,
                blockBase1,
                blockCount1);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to seed degrain motion search level1 vectors: %s.\n"), get_err_mes(err));
            return err;
        }
        err = profileFinishStep(_T("level1 seed"), profileLevel1SeedMs, profileStepStart, dir);
        if (err != RGY_ERR_NONE) {
            return err;
        }

        RGYOpenCLEvent searchLevel1Event;
        profileStepStart = profileNow();
        err = programL1->kernel("kernel_degrain_mv_search_parallel").config(
            queue, searchLocal1, searchGlobal1,
            { seedLevel1Event, downsampleEvents[0], downsampleEvents[dir + 1] }, &searchLevel1Event).launch(
                m_analysis.lumaLevel1[0]->mem(),
                m_analysis.lumaLevel1[dir + 1]->mem(),
                ws.level1.vectors->mem(),
                m_analysis.lumaLevel1Pitch,
                m_analysis.lumaLevel1Width,
                m_analysis.lumaLevel1Height,
                planeBase1,
                blockCount1,
                m_analysis.layoutLevel1.blocksX,
                m_analysis.layoutLevel1.blocksY,
                m_analysis.layoutLevel1.step);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run degrain motion search level1 search stub: %s.\n"), get_err_mes(err));
            return err;
        }
        RGYOpenCLEvent level1VectorReadyEvent = searchLevel1Event;
        const int level1SpatialRefineCount = spatialRefineCount(1);
        if (level1SpatialRefineCount <= 0) {
            RGYOpenCLEvent copyEvent;
            err = copyMotionSearchVectors(ws.level1.vectors.get(), (size_t)planeBase1 + vectorSentinelCount,
                ws.level1.vectorsFinal.get(), (size_t)blockBase1, blockCount1, { level1VectorReadyEvent },
                &copyEvent, _T("level1 current-to-final"));
            if (err != RGY_ERR_NONE) {
                return err;
            }
            level1VectorReadyEvent = copyEvent;
        }
        for (int refine = 0; refine < level1SpatialRefineCount; refine++) {
            RGYOpenCLEvent refineEvent;
            err = programL1->kernel("kernel_degrain_mv_spatial_refine").config(
                queue, searchLocal1, searchGlobal1, { level1VectorReadyEvent }, &refineEvent).launch(
                    m_analysis.lumaLevel1[0]->mem(),
                    m_analysis.lumaLevel1[dir + 1]->mem(),
                    ws.level1.vectors->mem(),
                    ws.level1.vectorsPrev->mem(),
                    ws.level1.vectorsFinal->mem(),
                    m_analysis.lumaLevel1Pitch,
                    m_analysis.lumaLevel1Width,
                    m_analysis.lumaLevel1Height,
                    planeBase1,
                    blockBase1,
                    blockCount1,
                    m_analysis.layoutLevel1.blocksX,
                    m_analysis.layoutLevel1.blocksY,
                    m_analysis.layoutLevel1.step);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to refine degrain motion search level1 spatial predictors: %s.\n"), get_err_mes(err));
                return err;
            }
            level1VectorReadyEvent = refineEvent;
            if (refine + 1 < level1SpatialRefineCount) {
                RGYOpenCLEvent copyEvent;
                err = copyMotionSearchVectors(ws.level1.vectorsFinal.get(), (size_t)blockBase1,
                    ws.level1.vectors.get(), (size_t)planeBase1 + vectorSentinelCount, blockCount1, { level1VectorReadyEvent },
                    &copyEvent, _T("level1 final-to-current"));
                if (err != RGY_ERR_NONE) {
                    return err;
                }
                level1VectorReadyEvent = copyEvent;
            }
        }
        err = profileFinishStep(_T("level1 search"), profileLevel1SearchMs, profileStepStart, dir);
        if (err != RGY_ERR_NONE) {
            return err;
        }

        RGYOpenCLEvent exportLevel1Event;
        profileStepStart = profileNow();
        err = programL1->kernel("kernel_degrain_mv_export_sad").config(
            queue, RGYWorkSize(1), RGYWorkSize(blockCount1), { level1VectorReadyEvent }, &exportLevel1Event).launch(
                ws.level1.vectorsFinal->mem(),
                ws.level1.sads->mem(),
                (cl_mem)nullptr,
                (cl_mem)nullptr,
                blockBase1,
                blockBase1,
                blockCount1,
                0,
                dir);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to export degrain motion search level1 SAD: %s.\n"), get_err_mes(err));
            return err;
        }
        err = profileFinishStep(_T("level1 export_sad"), profileLevel1ExportSadMs, profileStepStart, dir);
        if (err != RGY_ERR_NONE) {
            return err;
        }

        const int planeBase0 = levelPlaneBase(dir, planeStride0);
        const int blockBase0 = blockPlaneBase(dir, blockCount0);
        RGYOpenCLEvent interpolateEvent;
        profileStepStart = profileNow();
        err = programL0->kernel("kernel_degrain_mv_expand_coarse_vectors").config(
            queue, RGYWorkSize(256), RGYWorkSize(blockCount0), { exportLevel1Event, initLevel0Event }, &interpolateEvent).launch(
                ws.level1.vectorsFinal->mem(),
                ws.level0.vectors->mem(),
                ws.level0.vectorsPrev->mem(),
                ws.level0.sads->mem(),
                blockBase1,
                planeBase0,
                blockBase0,
                blockCount1,
                blockCount0,
                m_analysis.layoutLevel1.blocksX,
                m_analysis.layoutLevel1.blocksY,
                m_analysis.layout.blocksX);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to interpolate degrain motion search predictor: %s.\n"), get_err_mes(err));
            return err;
        }
        err = profileFinishStep(_T("interpolate"), profileInterpolateMs, profileStepStart, dir);
        if (err != RGY_ERR_NONE) {
            return err;
        }

        RGYOpenCLEvent searchLevel0Event;
        profileStepStart = profileNow();
        err = programL0->kernel("kernel_degrain_mv_search_parallel").config(
            queue, searchLocal0, searchGlobal0, { interpolateEvent }, &searchLevel0Event).launch(
                planeMem(planeCur),
                planeMem(refPlanes[dir]),
                ws.level0.vectors->mem(),
                planeCur.pitch[0],
                planeCur.width,
                planeCur.height,
                planeBase0,
                blockCount0,
                m_analysis.layout.blocksX,
                m_analysis.layout.blocksY,
                m_analysis.layout.step);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run degrain motion search level0 search stub: %s.\n"), get_err_mes(err));
            return err;
        }
        RGYOpenCLEvent level0VectorReadyEvent = searchLevel0Event;
        const int level0SpatialRefineCount = spatialRefineCount(0);
        if (level0SpatialRefineCount <= 0) {
            RGYOpenCLEvent copyEvent;
            err = copyMotionSearchVectors(ws.level0.vectors.get(), (size_t)planeBase0 + vectorSentinelCount,
                ws.level0.vectorsFinal.get(), (size_t)blockBase0, blockCount0, { level0VectorReadyEvent },
                &copyEvent, _T("level0 current-to-final"));
            if (err != RGY_ERR_NONE) {
                return err;
            }
            level0VectorReadyEvent = copyEvent;
        }
        for (int refine = 0; refine < level0SpatialRefineCount; refine++) {
            RGYOpenCLEvent refineEvent;
            err = programL0->kernel("kernel_degrain_mv_spatial_refine").config(
                queue, searchLocal0, searchGlobal0, { level0VectorReadyEvent }, &refineEvent).launch(
                    planeMem(planeCur),
                    planeMem(refPlanes[dir]),
                    ws.level0.vectors->mem(),
                    ws.level0.vectorsPrev->mem(),
                    ws.level0.vectorsFinal->mem(),
                    planeCur.pitch[0],
                    planeCur.width,
                    planeCur.height,
                    planeBase0,
                    blockBase0,
                    blockCount0,
                    m_analysis.layout.blocksX,
                    m_analysis.layout.blocksY,
                    m_analysis.layout.step);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to refine degrain motion search level0 spatial predictors: %s.\n"), get_err_mes(err));
                return err;
            }
            level0VectorReadyEvent = refineEvent;
            if (refine + 1 < level0SpatialRefineCount) {
                RGYOpenCLEvent copyEvent;
                err = copyMotionSearchVectors(ws.level0.vectorsFinal.get(), (size_t)blockBase0,
                    ws.level0.vectors.get(), (size_t)planeBase0 + vectorSentinelCount, blockCount0, { level0VectorReadyEvent },
                    &copyEvent, _T("level0 final-to-current"));
                if (err != RGY_ERR_NONE) {
                    return err;
                }
                level0VectorReadyEvent = copyEvent;
            }
        }
        err = profileFinishStep(_T("level0 search"), profileLevel0SearchMs, profileStepStart, dir);
        if (err != RGY_ERR_NONE) {
            return err;
        }

        RGYOpenCLEvent exportLevel0Event;
        profileStepStart = profileNow();
        err = programL0->kernel("kernel_degrain_mv_export_sad").config(
            queue, RGYWorkSize(1), RGYWorkSize(blockCount0), { level0VectorReadyEvent }, &exportLevel0Event).launch(
                ws.level0.vectorsFinal->mem(),
                ws.level0.sads->mem(),
                m_analysis.mv->mem(),
                m_analysis.sad->mem(),
                blockBase0,
                blockBase0,
                blockCount0,
                0,
                dir);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to export degrain motion search level0 SAD: %s.\n"), get_err_mes(err));
            return err;
        }
        err = profileFinishStep(_T("level0 export_sad"), profileLevel0ExportSadMs, profileStepStart, dir);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        previousEvent = exportLevel0Event;
    }

    if (previousEvent() == nullptr) {
        return RGY_ERR_UNSUPPORTED;
    }
    m_analysis.event = previousEvent;
    m_lastAnalysisIncludedChroma = false;
    if (profileEnabled) {
        const auto profileTotalMs = profileElapsedMs(profileTotalStart, ProfileClock::now());
        AddMessage(RGY_LOG_INFO,
            _T("degrain motion search profile summary: downsample=%.3f ms, seed_anchor_vectors=%.3f ms, level1_seed=%.3f ms, level1_search=%.3f ms, level1_export_sad=%.3f ms, expand_coarse_vectors=%.3f ms, level0_search=%.3f ms, level0_export_sad=%.3f ms, total=%.3f ms.\n"),
            profileDownsampleMs,
            profileInitConstVecMs,
            profileLevel1SeedMs,
            profileLevel1SearchMs,
            profileLevel1ExportSadMs,
            profileInterpolateMs,
            profileLevel0SearchMs,
            profileLevel0ExportSadMs,
            profileTotalMs);
    }
    AddMessage(RGY_LOG_DEBUG, _T("degrain motion search analyze path was used.\n"));
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDegrain::prepareAnalysisState(const RGYFilterDegrainFrameSet &frames, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    const int requiredDelta = std::min(RGY_DEGRAIN_MAX_DELTA, std::max(1, m_analysis.layout.temporalDirections / 2));
    m_lastAnalysisUsedSearchLuma = false;
    m_lastAnalysisIncludedChroma = false;
    if (!frames.cur || frames.cur->ptr[0] == nullptr || !m_analysis.mv || !m_analysis.sad) {
        AddMessage(RGY_LOG_ERROR, _T("degrain analysis buffers are not ready.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    for (int delta = 1; delta <= requiredDelta; delta++) {
        if (!frames.backwardRef(delta) || !frames.forwardRef(delta)
            || frames.backwardRef(delta)->ptr[0] == nullptr || frames.forwardRef(delta)->ptr[0] == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("degrain analysis reference frames for delta=%d are not ready.\n"), delta);
            return RGY_ERR_INVALID_CALL;
        }
    }
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDegrain>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid degrain parameter type in analysis.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto analysisFrames = frames;
    const auto attachedCur = degrainAttachedSearchLumaFrame(frames.cur);
    bool hasAttachedRefs = (attachedCur != nullptr);
    for (int delta = 1; delta <= requiredDelta && hasAttachedRefs; delta++) {
        const auto attachedBackward = degrainAttachedSearchLumaFrame(frames.backwardRef(delta));
        const auto attachedForward = degrainAttachedSearchLumaFrame(frames.forwardRef(delta));
        hasAttachedRefs &= (attachedBackward != nullptr && attachedForward != nullptr);
        if (hasAttachedRefs) {
            analysisFrames.backward[delta - 1] = attachedBackward;
            analysisFrames.forward[delta - 1] = attachedForward;
        }
    }
    if (hasAttachedRefs) {
        analysisFrames.cur = attachedCur;
        m_lastAnalysisUsedSearchLuma = true;
    }
    logLocalAnalysis(_T("local"), analysisFrames);

    const auto planeCur = getPlane(analysisFrames.cur, RGY_PLANE_Y);
    std::array<RGYFrameInfo, RGY_DEGRAIN_MAX_TEMPORAL_DIRECTIONS> refPlanes;
    refPlanes.fill(planeCur);
    for (int delta = 1; delta <= requiredDelta; delta++) {
        refPlanes[rgy_degrain_ref_index(delta, false)] = getPlane(analysisFrames.backwardRef(delta), RGY_PLANE_Y);
        refPlanes[rgy_degrain_ref_index(delta, true)] = getPlane(analysisFrames.forwardRef(delta), RGY_PLANE_Y);
    }
    if (planeCur.ptr[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("degrain analysis requires valid luma planes.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    for (int dir = 0; dir < m_analysis.layout.temporalDirections; dir++) {
        if (refPlanes[dir].ptr[0] == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("degrain analysis reference plane dir=%d is invalid.\n"), dir);
            return RGY_ERR_INVALID_CALL;
        }
    }
    const int pitchY = planeCur.pitch[0];
    for (int dir = 0; dir < m_analysis.layout.temporalDirections; dir++) {
        if (refPlanes[dir].pitch[0] != pitchY) {
            AddMessage(RGY_LOG_ERROR,
                _T("degrain analysis luma pitch mismatch: cur=%d, refdir%d=%d.\n"),
                pitchY, dir, refPlanes[dir].pitch[0]);
            return RGY_ERR_INVALID_PARAM;
        }
    }
    const auto chromaPlanes = degrainMakeAnalyzeChromaPlanes(
        analysisFrames,
        planeCur,
        refPlanes,
        m_analysis.layout.temporalDirections,
        requiredDelta,
        prm->degrain.chroma,
        m_lastAnalysisUsedSearchLuma);
    m_lastAnalysisIncludedChroma = chromaPlanes.enable != 0;
    const int pitchUV = chromaPlanes.curU.pitch[0];
    if (chromaPlanes.enable) {
        if (chromaPlanes.curV.pitch[0] != pitchUV) {
            AddMessage(RGY_LOG_ERROR,
                _T("degrain analysis chroma pitch mismatch: curU=%d, curV=%d.\n"),
                pitchUV, chromaPlanes.curV.pitch[0]);
            return RGY_ERR_INVALID_PARAM;
        }
        for (int dir = 0; dir < m_analysis.layout.temporalDirections; dir++) {
            if (chromaPlanes.refU[dir].pitch[0] != pitchUV || chromaPlanes.refV[dir].pitch[0] != pitchUV) {
                AddMessage(RGY_LOG_ERROR,
                    _T("degrain analysis chroma pitch mismatch: curUV=%d, refdir%d U=%d, V=%d.\n"),
                    pitchUV, dir, chromaPlanes.refU[dir].pitch[0], chromaPlanes.refV[dir].pitch[0]);
                return RGY_ERR_INVALID_PARAM;
            }
        }
    }

    std::vector<RGYOpenCLEvent> analysisWaitEvents = wait_events;
    if (useAnalysisLumaCache()) {
        for (const auto &analysisLumaEvent : m_analysis.analysisLumaEvents) {
            if (analysisLumaEvent() != nullptr) {
                analysisWaitEvents.push_back(analysisLumaEvent);
            }
        }
    }
    const auto motionSearchErr = prepareAnalysisStateMotionSearch(planeCur, refPlanes, queue, analysisWaitEvents);
    if (motionSearchErr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("degrain motion search analysis failed: %s.\n"), get_err_mes(motionSearchErr));
        return motionSearchErr;
    }
    logAnalysisSamples(_T("local"), frames.cur, queue);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDegrain::runAnalyzeMode(const RGYFilterDegrainProcessFrameSet &frames, const int currentFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    clearFrameAnalysisData();
    auto err = prepareAnalysisState(frames.analysis, queue, wait_events);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    m_analysis.lastAvailabilityDisableRefs = degrainReferenceAvailability(frames.analysis);
    m_analysis.lastFrameIndex = currentFrame;
    m_analysis.lastInputFrameId = (frames.render.cur) ? frames.render.cur->inputFrameId : -1;
    m_analysis.lastTimestamp = (frames.render.cur) ? frames.render.cur->timestamp : 0;
    m_analysis.lastDuration = (frames.render.cur) ? frames.render.cur->duration : 0;
    RGYOpenCLEvent copyEvent;
    err = emitSourceFrame(frames.render.cur, ppOutputFrames, pOutputFrameNum, queue, {}, &copyEvent);
    if (err != RGY_ERR_NONE || *pOutputFrameNum <= 0 || ppOutputFrames[0] == nullptr) {
        if (event) {
            *event = copyEvent;
        }
        return err;
    }
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDegrain>(m_param);
    if (prm && !prm->attachAnalysisData) {
        if (event) {
            *event = copyEvent;
        }
        return RGY_ERR_NONE;
    }
    return attachAnalysisData(frames.render.cur, ppOutputFrames[0], currentFrame, queue, copyEvent, event);
}
