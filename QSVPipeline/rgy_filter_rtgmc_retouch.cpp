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

#include "rgy_filter_rtgmc_retouch.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <fstream>
#include <vector>

#include "rgy_filter_resize.h"
#include "rgy_resource.h"
#include "rgy_util.h"

namespace {
static constexpr int RTGMC_RETOUCH_BLOCK_X = 32;
static constexpr int RTGMC_RETOUCH_BLOCK_Y = 8;
static constexpr int RTGMC_RETOUCH_FRAMEBUF_COUNT = 5;
static constexpr float RTGMC_RETOUCH_BACKBLEND_GAUSS_P = 5.0f;
static constexpr float RTGMC_RETOUCH_DETAIL_BASE_GAIN = 0.20f;
static constexpr float RTGMC_RETOUCH_DETAIL_TR1_GAIN = 0.15f;
static constexpr float RTGMC_RETOUCH_DETAIL_TR2_GAIN = 0.25f;
static constexpr float RTGMC_RETOUCH_DETAIL_SMODE1_BIAS = 0.10f;
static constexpr float RTGMC_RETOUCH_EDGE_NARROW_GAIN_SCALE = 6.0f;

float calcRtgmcDetailGain(const VppRtgmcRetouch& retouch) {
    float limitModeBoost = 1.0f;
    if (retouch.slmode == 2 || retouch.slmode == 4) {
        limitModeBoost = 2.0f;
    } else if (retouch.slmode == 1 || retouch.slmode == 3) {
        limitModeBoost = 1.5f;
    }

    float radiusGain = RTGMC_RETOUCH_DETAIL_BASE_GAIN;
    radiusGain += retouch.tr1 * RTGMC_RETOUCH_DETAIL_TR1_GAIN;
    radiusGain += retouch.tr2 * RTGMC_RETOUCH_DETAIL_TR2_GAIN;

    float gain = limitModeBoost * radiusGain;
    if (retouch.smode == 1) {
        gain += RTGMC_RETOUCH_DETAIL_SMODE1_BIAS;
    }
    return retouch.sharpness * gain;
}

float calcRtgmcEdgeNarrowGain(const VppRtgmcRetouch& retouch) {
    return retouch.svthin * RTGMC_RETOUCH_EDGE_NARROW_GAIN_SCALE;
}

bool isFrameCompatible(const RGYFrameInfo *base, const RGYFrameInfo *frame) {
    return base && frame
        && base->csp == frame->csp
        && base->width == frame->width
        && base->height == frame->height
        && base->bitdepth == frame->bitdepth
        && base->mem_type == frame->mem_type;
}

bool isRetouchChromaPlane(const int iplane) {
    return iplane > 0;
}

bool isRtgmcRetouchDumpStageSupported(const std::string& stage) {
    return stage == "input"
        || stage == "detail_boost_edge_ref"
        || stage == "detail_boost_regularized_ref"
        || stage == "detail_boost_blur_ref"
        || stage == "detail_boost"
        || stage == "prelimit_rollback"
        || stage == "prelimit_rollback_delta"
        || stage == "prelimit_rollback_smooth_delta"
        || stage == "prelimit_rollback_soft_delta"
        || stage == "spatial_guard"
        || stage == "postlimit_rollback"
        || stage == "postlimit_rollback_delta"
        || stage == "postlimit_rollback_smooth_delta"
        || stage == "postlimit_rollback_soft_delta"
        || stage == "edge_narrow_delta"
        || stage == "edge_narrow_blur_delta"
        || stage == "edge_narrow_guard_delta"
        || stage == "edge_narrow"
        || stage == "postlimit_spatial_guard_src"
        || stage == "postlimit_spatial_guard_ref"
        || stage == "postlimit_spatial_guard"
        || stage == "temporal_guard_src"
        || stage == "temporal_guard_ref"
        || stage == "temporal_guard_motionback"
        || stage == "temporal_guard_motionforw"
        || stage == "temporal_guard"
        || stage == "postlimit_temporal_guard_src"
        || stage == "postlimit_temporal_guard_ref"
        || stage == "postlimit_temporal_guard_motionback"
        || stage == "postlimit_temporal_guard_motionforw"
        || stage == "postlimit_temporal_guard";
}

bool rtgmcRetouchDumpStageChromaReady(const std::string& stage) {
    return stage == "input"
        || (stage.size() >= 4 && stage.compare(stage.size() - 4, 4, "_ref") == 0)
        || (stage.size() >= 11 && stage.compare(stage.size() - 11, 11, "_motionback") == 0)
        || (stage.size() >= 11 && stage.compare(stage.size() - 11, 11, "_motionforw") == 0);
}

const char *rtgmcRetouchDumpTargetForStage(const std::string& stage) {
    if (stage.find("rollback") != std::string::npos) {
        return "rollback";
    }
    if (stage.rfind("postlimit_", 0) == 0) {
        return "postlimit";
    }
    if (stage.rfind("temporal_guard_", 0) == 0 || stage == "temporal_guard") {
        return "limitover";
    }
    if (stage.rfind("edge_narrow", 0) == 0) {
        return "edge_narrow";
    }
    return "retouch";
}
}

class RGYFilterResizePlaneProxy : public RGYFilterResize {
public:
    using RGYFilterResize::RGYFilterResize;
    using RGYFilterResize::resizePlane;
};

RGYFilterRtgmcRetouch::RGYFilterRtgmcRetouch(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_retouch(),
    m_buildOptions(),
    m_lumaDump(),
    m_lumaDumpPath(),
    m_lumaDumpStage("edge_narrow_blur_delta"),
    m_lumaDumpTarget(),
    m_lumaDumpMaxFrames(0),
    m_lumaDumpFrameCount(0),
    m_lumaDumpEnabled(false),
    m_lumaDumpHeaderWritten(false),
    m_lumaDumpChroma(false),
    m_detailRollbackGauss(),
    m_temporalLimitFrames(),
    m_spatialLimitBaseFrame(nullptr),
    m_loggedTemporalFallback(false) {
    m_name = _T("rtgmc-retouch");
}

RGYFilterRtgmcRetouch::~RGYFilterRtgmcRetouch() {
    close();
}

RGY_ERR RGYFilterRtgmcRetouch::checkParam(const std::shared_ptr<RGYFilterParamRtgmcRetouch> &prm) {
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameIn.width <= 0 || prm->frameIn.height <= 0
        || prm->frameOut.width <= 0 || prm->frameOut.height <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameIn.csp != prm->frameOut.csp
        || prm->frameIn.width != prm->frameOut.width
        || prm->frameIn.height != prm->frameOut.height) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-retouch requires identical input/output csp and resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->rtgmc_retouch.sharpness < 0.0f || prm->rtgmc_retouch.sharpness > 1.0f) {
        prm->rtgmc_retouch.sharpness = clamp(prm->rtgmc_retouch.sharpness, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("rtgmc-retouch sharpness should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }
    if (prm->rtgmc_retouch.limit < 0.0f || prm->rtgmc_retouch.limit > 1.0f) {
        prm->rtgmc_retouch.limit = clamp(prm->rtgmc_retouch.limit, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("rtgmc-retouch limit should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }
    if (prm->rtgmc_retouch.smode < 0 || prm->rtgmc_retouch.smode > 2) {
        prm->rtgmc_retouch.smode = clamp(prm->rtgmc_retouch.smode, 0, 2);
        AddMessage(RGY_LOG_WARN, _T("rtgmc-retouch smode should be in range of 0 - 2.\n"));
    }
    if (prm->rtgmc_retouch.slmode < 0 || prm->rtgmc_retouch.slmode > 4) {
        prm->rtgmc_retouch.slmode = clamp(prm->rtgmc_retouch.slmode, 0, 4);
        AddMessage(RGY_LOG_WARN, _T("rtgmc-retouch slmode should be in range of 0 - 4.\n"));
    }
    if (prm->rtgmc_retouch.slrad < 0 || prm->rtgmc_retouch.slrad > 3) {
        prm->rtgmc_retouch.slrad = clamp(prm->rtgmc_retouch.slrad, 0, 3);
        AddMessage(RGY_LOG_WARN, _T("rtgmc-retouch slrad should be in range of 0 - 3.\n"));
    }
    if (prm->rtgmc_retouch.sovs < 0) {
        prm->rtgmc_retouch.sovs = 0;
        AddMessage(RGY_LOG_WARN, _T("rtgmc-retouch sovs should be 0 or greater.\n"));
    }
    if (prm->rtgmc_retouch.svthin < 0.0f || prm->rtgmc_retouch.svthin > 1.0f) {
        prm->rtgmc_retouch.svthin = clamp(prm->rtgmc_retouch.svthin, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("rtgmc-retouch svthin should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }
    if (prm->rtgmc_retouch.sbb < 0 || prm->rtgmc_retouch.sbb > 3) {
        prm->rtgmc_retouch.sbb = clamp(prm->rtgmc_retouch.sbb, 0, 3);
        AddMessage(RGY_LOG_WARN, _T("rtgmc-retouch sbb should be in range of 0 - 3.\n"));
    }
    if ((prm->rtgmc_retouch.smode == 0 || prm->rtgmc_retouch.sharpness <= 0.0f || prm->rtgmc_retouch.slrad <= 0)
        && prm->rtgmc_retouch.slmode < 3) {
        prm->rtgmc_retouch.slmode = 0;
    }
    if (prm->rtgmc_retouch.slmode >= 3 && prm->skipPostTR2LimitModes) {
        AddMessage(RGY_LOG_DEBUG, _T("rtgmc-retouch slmode=%d is a post-TR2 mode and is not applied in this stage.\n"), prm->rtgmc_retouch.slmode);
    }
    return RGY_ERR_NONE;
}

void RGYFilterRtgmcRetouch::setSpatialLimitBaseFrame(const RGYFrameInfo *frame) {
    m_spatialLimitBaseFrame = frame;
}

void RGYFilterRtgmcRetouch::clearSpatialLimitBaseFrame() {
    m_spatialLimitBaseFrame = nullptr;
}

void RGYFilterRtgmcRetouch::setTemporalLimitFrames(const RGYRtgmcRetouchTemporalLimitFrames &frames) {
    m_temporalLimitFrames = frames;
    m_temporalLimitFrames.useInlineComp = false;
    m_loggedTemporalFallback = false;
}

void RGYFilterRtgmcRetouch::clearTemporalLimitFrames() {
    m_temporalLimitFrames = RGYRtgmcRetouchTemporalLimitFrames();
    m_loggedTemporalFallback = false;
}

void RGYFilterRtgmcRetouch::setTemporalLimitInlineComp(const RGYFrameInfo *ref, const std::array<RGYDegrainCompensateInlineParams, 3> &params, bool processChroma) {
    m_temporalLimitFrames.ref = ref;
    m_temporalLimitFrames.motionBack = nullptr;
    m_temporalLimitFrames.motionForw = nullptr;
    m_temporalLimitFrames.useInlineComp = true;
    m_temporalLimitFrames.inlineCompChroma = processChroma;
    m_temporalLimitFrames.inlineCompParams = params;
    m_loggedTemporalFallback = false;
}

bool RGYFilterRtgmcRetouch::temporalLimitFramesCompatible(const RGYFrameInfo *srcFrame) const {
    const auto &frames = m_temporalLimitFrames;
    if (frames.useInlineComp) {
        return isFrameCompatible(srcFrame, frames.ref);
    }
    return isFrameCompatible(srcFrame, frames.ref)
        && isFrameCompatible(srcFrame, frames.motionBack)
        && isFrameCompatible(srcFrame, frames.motionForw);
}

bool RGYFilterRtgmcRetouch::temporalLimitFramesReady(const RGYFrameInfo *srcFrame) const {
    return m_temporalLimitFrames.valid() && temporalLimitFramesCompatible(srcFrame);
}

RGY_ERR RGYFilterRtgmcRetouch::buildKernels(const std::shared_ptr<RGYFilterParamRtgmcRetouch> &prm) {
    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelMax = (bitdepth >= 16) ? std::numeric_limits<uint16_t>::max() : ((1 << bitdepth) - 1);
    const int rangeHalf = 1 << (bitdepth - 1);
    m_buildOptions = strsprintf(
        "-D Type=%s -D bit_depth=%d -D max_val=%d -D range_half=%d",
        bitdepth > 8 ? "ushort" : "uchar",
        bitdepth,
        pixelMax,
        rangeHalf);
    m_retouch.set(m_cl->buildResourceAsync(_T("RGY_FILTER_RTGMC_RETOUCH_CL"), _T("EXE_DATA"), m_buildOptions.c_str()));
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcRetouch::initLumaDump(const RGYFrameInfo &frameInfo, const RGYFilterParamRtgmcRetouch &prm) {
    UNREFERENCED_PARAMETER(prm);
    m_lumaDumpEnabled = false;
    m_lumaDumpHeaderWritten = false;
    m_lumaDumpFrameCount = 0;
    m_lumaDumpMaxFrames = 0;
    m_lumaDumpPath.clear();
    m_lumaDumpStage = "edge_narrow_blur_delta";
    m_lumaDumpTarget.clear();
    m_lumaDumpChroma = false;
    if (m_lumaDump.is_open()) {
        m_lumaDump.close();
    }

    const char *dumpPathEnv = std::getenv("QSVENC_RTGMC_RETOUCH_LUMA_DUMP_Y4M");
    if (dumpPathEnv == nullptr || dumpPathEnv[0] == '\0') {
        return RGY_ERR_NONE;
    }
    m_lumaDumpPath = dumpPathEnv;

    if (const char *stageEnv = std::getenv("QSVENC_RTGMC_RETOUCH_LUMA_DUMP_STAGE"); stageEnv != nullptr && stageEnv[0] != '\0') {
        m_lumaDumpStage = stageEnv;
    }
    if (!isRtgmcRetouchDumpStageSupported(m_lumaDumpStage)) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported rtgmc retouch luma dump stage: %s.\n"),
            char_to_tstring(m_lumaDumpStage).c_str());
        return RGY_ERR_INVALID_PARAM;
    }

    if (const char *targetEnv = std::getenv("QSVENC_RTGMC_RETOUCH_LUMA_DUMP_TARGET"); targetEnv != nullptr && targetEnv[0] != '\0') {
        m_lumaDumpTarget = targetEnv;
    }
    const char *activeTarget = rtgmcRetouchDumpTargetForStage(m_lumaDumpStage);
    if (!m_lumaDumpTarget.empty() && m_lumaDumpTarget != activeTarget) {
        AddMessage(RGY_LOG_DEBUG, _T("rtgmc retouch luma dump target %s skipped for inactive %s instance.\n"),
            char_to_tstring(m_lumaDumpTarget).c_str(), char_to_tstring(activeTarget).c_str());
        return RGY_ERR_NONE;
    }

    const int bitdepth = RGY_CSP_BIT_DEPTH[frameInfo.csp];
    if (bitdepth > 8) {
        AddMessage(RGY_LOG_WARN, _T("QSVENC_RTGMC_RETOUCH_LUMA_DUMP_Y4M supports only 8bit input, disabling dump for %s.\n"),
            RGY_CSP_NAMES[frameInfo.csp]);
        return RGY_ERR_NONE;
    }
    if (RGY_CSP_CHROMA_FORMAT[frameInfo.csp] != RGY_CHROMAFMT_YUV420 && RGY_CSP_PLANES[frameInfo.csp] != 1) {
        AddMessage(RGY_LOG_WARN, _T("QSVENC_RTGMC_RETOUCH_LUMA_DUMP_Y4M supports only 4:2:0/Y8 input, disabling dump for %s.\n"),
            RGY_CSP_NAMES[frameInfo.csp]);
        return RGY_ERR_NONE;
    }

    if (const char *maxFrames = std::getenv("QSVENC_RTGMC_RETOUCH_LUMA_DUMP_MAX_FRAMES"); maxFrames != nullptr && maxFrames[0] != '\0') {
        char *endptr = nullptr;
        const long parsed = std::strtol(maxFrames, &endptr, 10);
        if (endptr != maxFrames && parsed > 0) {
            m_lumaDumpMaxFrames = (int)std::min<long>(parsed, std::numeric_limits<int>::max());
        }
    }
    if (const char *dumpChroma = std::getenv("QSVENC_RTGMC_RETOUCH_DUMP_CHROMA"); dumpChroma != nullptr && dumpChroma[0] != '\0' && dumpChroma[0] != '0') {
        m_lumaDumpChroma = true;
    }

    m_lumaDump.open(m_lumaDumpPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!m_lumaDump) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open rtgmc retouch luma dump: %s.\n"),
            char_to_tstring(m_lumaDumpPath).c_str());
        return RGY_ERR_FILE_OPEN;
    }
    m_lumaDumpEnabled = true;
    AddMessage(RGY_LOG_INFO, _T("rtgmc retouch luma dump enabled: %s (target=%s, stage=%s).\n"),
        char_to_tstring(m_lumaDumpPath).c_str(), char_to_tstring(activeTarget).c_str(), char_to_tstring(m_lumaDumpStage).c_str());
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcRetouch::dumpLumaFrame(const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, bool dumpChroma) {
    if (!m_lumaDumpEnabled) {
        return RGY_ERR_NONE;
    }
    if (m_lumaDumpMaxFrames > 0 && m_lumaDumpFrameCount >= m_lumaDumpMaxFrames) {
        return RGY_ERR_NONE;
    }
    if (frame == nullptr || frame->ptr[0] == nullptr) {
        return RGY_ERR_NULL_PTR;
    }
    const int bitdepth = RGY_CSP_BIT_DEPTH[frame->csp];
    if (bitdepth > 8 || (RGY_CSP_CHROMA_FORMAT[frame->csp] != RGY_CHROMAFMT_YUV420 && RGY_CSP_PLANES[frame->csp] != 1)) {
        AddMessage(RGY_LOG_WARN, _T("rtgmc retouch luma dump disabled by unsupported frame csp: %s.\n"),
            RGY_CSP_NAMES[frame->csp]);
        m_lumaDumpEnabled = false;
        return RGY_ERR_NONE;
    }

    const auto planeY = getPlane(frame, RGY_PLANE_Y);
    const auto planeU = getPlane(frame, RGY_PLANE_U);
    const auto planeV = getPlane(frame, RGY_PLANE_V);
    std::vector<uint8_t> hostY((size_t)planeY.width * planeY.height);
    std::vector<uint8_t> hostU;
    std::vector<uint8_t> hostV;
    RGYFrameInfo hostFrame(planeY.width, planeY.height, RGY_CSP_Y8, 8, frame->picstruct, RGY_MEM_TYPE_CPU);
    hostFrame.ptr[0] = hostY.data();
    hostFrame.pitch[0] = planeY.width;

    RGYOpenCLEvent readEvent;
    auto err = m_cl->copyPlane(&hostFrame, &planeY, nullptr, queue, wait_events, &readEvent);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to read rtgmc retouch luma dump frame: %s.\n"), get_err_mes(err));
        return err;
    }
    err = readEvent.wait();
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to wait rtgmc retouch luma dump read: %s.\n"), get_err_mes(err));
        return err;
    }
    if (dumpChroma && RGY_CSP_CHROMA_FORMAT[frame->csp] == RGY_CHROMAFMT_YUV420 && planeU.ptr[0] != nullptr) {
        const int chromaWidth = (planeY.width + 1) >> 1;
        const int chromaHeight = (planeY.height + 1) >> 1;
        hostU.resize((size_t)chromaWidth * chromaHeight);
        hostV.resize((size_t)chromaWidth * chromaHeight);
        if (frame->csp == RGY_CSP_NV12) {
            std::vector<uint8_t> hostUV((size_t)planeU.width * planeU.height);
            RGYFrameInfo hostUVPlane(planeU.width, planeU.height, RGY_CSP_Y8, 8, frame->picstruct, RGY_MEM_TYPE_CPU);
            hostUVPlane.ptr[0] = hostUV.data();
            hostUVPlane.pitch[0] = planeU.width;

            RGYOpenCLEvent readUVEvent;
            err = m_cl->copyPlane(&hostUVPlane, &planeU, nullptr, queue, wait_events, &readUVEvent);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to read rtgmc retouch chroma UV dump frame: %s.\n"), get_err_mes(err));
                return err;
            }
            err = readUVEvent.wait();
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to wait rtgmc retouch chroma UV dump read: %s.\n"), get_err_mes(err));
                return err;
            }
            for (int y = 0; y < chromaHeight; y++) {
                const auto *srcUV = hostUV.data() + (size_t)y * hostUVPlane.pitch[0];
                auto *dstU = hostU.data() + (size_t)y * chromaWidth;
                auto *dstV = hostV.data() + (size_t)y * chromaWidth;
                for (int x = 0; x < chromaWidth; x++) {
                    dstU[x] = srcUV[x * 2 + 0];
                    dstV[x] = srcUV[x * 2 + 1];
                }
            }
        } else if (planeV.ptr[0] != nullptr) {
            RGYFrameInfo hostUPlane(chromaWidth, chromaHeight, RGY_CSP_Y8, 8, frame->picstruct, RGY_MEM_TYPE_CPU);
            RGYFrameInfo hostVPlane(chromaWidth, chromaHeight, RGY_CSP_Y8, 8, frame->picstruct, RGY_MEM_TYPE_CPU);
            hostUPlane.ptr[0] = hostU.data();
            hostUPlane.pitch[0] = chromaWidth;
            hostVPlane.ptr[0] = hostV.data();
            hostVPlane.pitch[0] = chromaWidth;

            RGYOpenCLEvent readUEvent;
            err = m_cl->copyPlane(&hostUPlane, &planeU, nullptr, queue, wait_events, &readUEvent);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to read rtgmc retouch chroma U dump frame: %s.\n"), get_err_mes(err));
                return err;
            }
            err = readUEvent.wait();
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to wait rtgmc retouch chroma U dump read: %s.\n"), get_err_mes(err));
                return err;
            }

            RGYOpenCLEvent readVEvent;
            err = m_cl->copyPlane(&hostVPlane, &planeV, nullptr, queue, wait_events, &readVEvent);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to read rtgmc retouch chroma V dump frame: %s.\n"), get_err_mes(err));
                return err;
            }
            err = readVEvent.wait();
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to wait rtgmc retouch chroma V dump read: %s.\n"), get_err_mes(err));
                return err;
            }
        } else {
            hostU.clear();
            hostV.clear();
        }
    }

    if (!m_lumaDumpHeaderWritten) {
        m_lumaDump << "YUV4MPEG2 W" << hostFrame.width << " H" << hostFrame.height << " F30000:1001 Ip A0:0 C420jpeg\n";
        m_lumaDumpHeaderWritten = true;
    }
    m_lumaDump << "FRAME\n";
    for (int y = 0; y < hostFrame.height; y++) {
        m_lumaDump.write(reinterpret_cast<const char *>(hostFrame.ptr[0] + (size_t)y * hostFrame.pitch[0]), hostFrame.width);
    }
    const int chromaWidth = (hostFrame.width + 1) >> 1;
    const int chromaHeight = (hostFrame.height + 1) >> 1;
    if (!hostU.empty() && !hostV.empty()) {
        m_lumaDump.write(reinterpret_cast<const char *>(hostU.data()), hostU.size());
        m_lumaDump.write(reinterpret_cast<const char *>(hostV.data()), hostV.size());
    } else {
        std::vector<uint8_t> neutralUV((size_t)chromaWidth * chromaHeight, 128);
        m_lumaDump.write(reinterpret_cast<const char *>(neutralUV.data()), neutralUV.size());
        m_lumaDump.write(reinterpret_cast<const char *>(neutralUV.data()), neutralUV.size());
    }
    if (!m_lumaDump) {
        AddMessage(RGY_LOG_ERROR, _T("failed to write rtgmc retouch luma dump: %s.\n"),
            char_to_tstring(m_lumaDumpPath).c_str());
        return RGY_ERR_FILE_OPEN;
    }
    m_lumaDumpFrameCount++;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcRetouch::dumpStageFrame(const char *stage, const RGYFrameInfo *frame, const char *target,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    if (!m_lumaDumpEnabled || m_lumaDumpStage != stage || (!m_lumaDumpTarget.empty() && m_lumaDumpTarget != target)) {
        return RGY_ERR_NONE;
    }
    const bool dumpChroma = m_lumaDumpChroma && rtgmcRetouchDumpStageChromaReady(m_lumaDumpStage);
    return dumpLumaFrame(frame, queue, wait_events, dumpChroma);
}

RGY_ERR RGYFilterRtgmcRetouch::setupDetailRollbackGaussFilter(const RGYFilterParamRtgmcRetouch &prm) {
    m_detailRollbackGauss.reset();
    if (prm.rtgmc_retouch.sbb <= 0) {
        return RGY_ERR_NONE;
    }

    auto gaussParam = std::make_shared<RGYFilterParamResize>();
    gaussParam->frameIn = getPlane(&prm.frameOut, RGY_PLANE_Y);
    gaussParam->frameOut = gaussParam->frameIn;
    gaussParam->interp = RGY_VPP_RESIZE_GAUSS;
    gaussParam->gaussP = RTGMC_RETOUCH_BACKBLEND_GAUSS_P;
    auto gaussResize = std::make_unique<RGYFilterResizePlaneProxy>(m_cl);
    auto sts = gaussResize->init(gaussParam, m_pLog);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    m_detailRollbackGauss = std::move(gaussResize);
    return RGY_ERR_NONE;
}

RGYFilterRtgmcRetouch::RtgmcRetouchPlan RGYFilterRtgmcRetouch::buildRtgmcRetouchPlan(
    const VppRtgmcRetouch &retouch, bool chromaPlane, bool skipPostTR2LimitModes, float detailGain) const {
    RtgmcRetouchPlan plan;

    const bool allowDetailBoost = retouch.smode > 0 && detailGain > 0.0f;
    const bool allowSpatialGuard = retouch.slmode == 1 || (retouch.slmode == 3 && !skipPostTR2LimitModes);
    const bool allowTemporalGuard = retouch.slmode == 2 || (retouch.slmode == 4 && !skipPostTR2LimitModes) || retouch.limit > 0.0f;
    if (allowDetailBoost) {
        plan.nodes.push_back({ RtgmcRetouchNodeKind::DetailBoost, 0, 1, 3, 4, "detail_boost" });
    }

    // Chroma planes intentionally never get luma-only retouch nodes.
    if (!chromaPlane) {
        if (retouch.svthin > 0.0f) {
            plan.nodes.push_back({ RtgmcRetouchNodeKind::EdgeNarrowCorrection, 0, 2, 3, 4, "edge_narrow" });
        }
        if (retouch.sbb == 1 || retouch.sbb == 3) {
            plan.nodes.push_back({ RtgmcRetouchNodeKind::PreLimitRollback, 0, 2, 3, 4, "prelimit_rollback" });
        }
    }
    if (allowSpatialGuard) {
        plan.nodes.push_back({ RtgmcRetouchNodeKind::SpatialOvershootGuard, 0, 2, 3, 4,
            (retouch.slmode == 3) ? "postlimit_spatial_guard" : "spatial_guard" });
    }
    if (allowTemporalGuard) {
        plan.nodes.push_back({ RtgmcRetouchNodeKind::TemporalOvershootGuard, 0, 2, 3, 4,
            (retouch.slmode == 4) ? "postlimit_temporal_guard" : "temporal_guard" });
    }
    if (!chromaPlane && retouch.sbb >= 2) {
        plan.nodes.push_back({ RtgmcRetouchNodeKind::PostLimitRollback, 0, 2, 3, 4, "postlimit_rollback" });
    }
    return plan;
}

std::string RGYFilterRtgmcRetouch::describeRtgmcRetouchPlan(const RtgmcRetouchPlan &plan) const {
    if (plan.nodes.empty()) {
        return "copy";
    }
    std::string desc;
    for (const auto &node : plan.nodes) {
        const char *kind = "unknown";
        switch (node.kind) {
        case RtgmcRetouchNodeKind::DetailBoost:
            kind = "detail_boost";
            break;
        case RtgmcRetouchNodeKind::EdgeNarrowCorrection:
            kind = "edge_narrow";
            break;
        case RtgmcRetouchNodeKind::PreLimitRollback:
            kind = "prelimit_rollback";
            break;
        case RtgmcRetouchNodeKind::SpatialOvershootGuard:
            kind = "spatial_guard";
            break;
        case RtgmcRetouchNodeKind::TemporalOvershootGuard:
            kind = "temporal_guard";
            break;
        case RtgmcRetouchNodeKind::PostLimitRollback:
            kind = "postlimit_rollback";
            break;
        }
        if (!desc.empty()) {
            desc += " -> ";
        }
        desc += kind;
        if (node.dumpStage != nullptr && node.dumpStage[0] != '\0') {
            desc += "(";
            desc += node.dumpStage;
            desc += ")";
        }
    }
    return desc;
}

RGY_ERR RGYFilterRtgmcRetouch::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamRtgmcRetouch>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamRtgmcRetouch>(m_param);
    if (!m_retouch.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        sts = buildKernels(prm);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build rtgmc-retouch kernel.\n"));
            return sts;
        }
    }

    sts = AllocFrameBuf(prm->frameOut, RTGMC_RETOUCH_FRAMEBUF_COUNT);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    sts = initLumaDump(m_frameBuf[0]->frame, *prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = setupDetailRollbackGaussFilter(*prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    const float detailGain = calcRtgmcDetailGain(prm->rtgmc_retouch);
    for (int iplane = 0; iplane < RGY_CSP_PLANES[prm->frameOut.csp]; iplane++) {
        const bool chromaPlane = isRetouchChromaPlane(iplane);
        const auto plan = buildRtgmcRetouchPlan(prm->rtgmc_retouch, chromaPlane, prm->skipPostTR2LimitModes, detailGain);
        AddMessage(RGY_LOG_DEBUG, _T("rtgmc-retouch plan plane %d (%s): %s.\n"),
            iplane,
            chromaPlane ? _T("chroma") : _T("luma"),
            char_to_tstring(describeRtgmcRetouchPlan(plan)).c_str());
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcRetouch::processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const RGYFilterParamRtgmcRetouch &prm,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    const float detailGain = calcRtgmcDetailGain(prm.rtgmc_retouch);
    const int bitdepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
    const int scaledSovs = prm.rtgmc_retouch.sovs << std::max(bitdepth - 8, 0);
    const bool hasTemporalLimitFrames = temporalLimitFramesReady(pInputFrame);
    const auto refFrame = hasTemporalLimitFrames ? m_temporalLimitFrames.ref : pInputFrame;
    const auto motionBackFrame = hasTemporalLimitFrames ? m_temporalLimitFrames.motionBack : pInputFrame;
    const auto motionForwFrame = hasTemporalLimitFrames ? m_temporalLimitFrames.motionForw : pInputFrame;
    const auto baseFrame = isFrameCompatible(pInputFrame, m_spatialLimitBaseFrame) ? m_spatialLimitBaseFrame : pInputFrame;
    const char *disableFusionEnv = std::getenv("QSVENC_RTGMC_RETOUCH_DISABLE_FUSION");
    const bool disableFusionByEnv = disableFusionEnv != nullptr && disableFusionEnv[0] != '\0' && disableFusionEnv[0] != '0';
    const bool disableFusion = m_lumaDumpEnabled || disableFusionByEnv;
    const char *mergeDetailLineEnv = std::getenv("QSVENC_RTGMC_KERNEL_MERGE_RETOUCH_DETAIL_LINE");
    const bool enableDetailLineMerge = mergeDetailLineEnv == nullptr || mergeDetailLineEnv[0] != '0';
    const char *mergeRollbackEnv = std::getenv("QSVENC_RTGMC_KERNEL_MERGE_RETOUCH_ROLLBACK");
    const bool enableRollbackMerge = mergeRollbackEnv == nullptr || mergeRollbackEnv[0] != '0';

    auto launchPlane = [&](const char *kernelName, const RGYFrameInfo *dstFrame, const RGYFrameInfo *src0Frame, const int iplane,
        const std::vector<RGYOpenCLEvent> &wait, RGYOpenCLEvent *ev, auto &&...args) {
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto src0Plane = getPlane(src0Frame, (RGY_PLANE)iplane);
        RGYWorkSize local(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        auto err = m_retouch.get()->kernel(kernelName).config(queue, local, global, wait, ev).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
            (cl_mem)src0Plane.ptr[0], src0Plane.pitch[0],
            dstPlane.width, dstPlane.height,
            args...);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                char_to_tstring(kernelName).c_str(), iplane, get_err_mes(err));
        }
        return err;
    };
    auto launchCopy = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const int iplane,
        const std::vector<RGYOpenCLEvent> &wait, RGYOpenCLEvent *ev) {
        return launchPlane("kernel_rtgmc_retouch_copy", dstFrame, srcFrame, iplane, wait, ev);
    };
    auto launchRemoveGrain = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const int iplane, const int smoothingMode) {
        return launchPlane((smoothingMode == 11) ? "kernel_rtgmc_retouch_removegrain11" : "kernel_rtgmc_retouch_removegrain12", dstFrame, srcFrame, iplane, {}, nullptr);
    };
    auto launchRepairMode1 = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const RGYFrameInfo *refFrame, const int iplane) {
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto refPlane = getPlane(refFrame, (RGY_PLANE)iplane);
        RGYWorkSize local(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        auto err = m_retouch.get()->kernel("kernel_rtgmc_retouch_repair1").config(queue, local, global, {}, nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
            (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
            (cl_mem)refPlane.ptr[0], refPlane.pitch[0],
            dstPlane.width, dstPlane.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_retouch_repair1"), iplane, get_err_mes(err));
        }
        return err;
    };
    auto launchRepairMode12 = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const RGYFrameInfo *refFrame, const int iplane) {
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto refPlane = getPlane(refFrame, (RGY_PLANE)iplane);
        RGYWorkSize local(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        auto err = m_retouch.get()->kernel("kernel_rtgmc_retouch_repair12").config(queue, local, global, {}, nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
            (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
            (cl_mem)refPlane.ptr[0], refPlane.pitch[0],
            dstPlane.width, dstPlane.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_retouch_repair12"), iplane, get_err_mes(err));
        }
        return err;
    };
    auto launchDetailBoostEdgeRef = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const int iplane,
        const std::vector<RGYOpenCLEvent> &wait) {
        return launchPlane("kernel_rtgmc_retouch_detail_ref_vertical", dstFrame, srcFrame, iplane, wait, nullptr);
    };
    auto launchPreciseClamp = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const RGYFrameInfo *refFrame, const int iplane) {
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto refPlane = getPlane(refFrame, (RGY_PLANE)iplane);
        RGYWorkSize local(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        auto err = m_retouch.get()->kernel("kernel_rtgmc_retouch_precise_clamp").config(queue, local, global, {}, nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
            (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
            (cl_mem)refPlane.ptr[0], refPlane.pitch[0],
            dstPlane.width, dstPlane.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_retouch_precise_clamp"), iplane, get_err_mes(err));
        }
        return err;
    };
    auto launchBlurH = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const int iplane) {
        return launchPlane("kernel_rtgmc_retouch_blur_h", dstFrame, srcFrame, iplane, {}, nullptr);
    };
    auto launchEdgeNarrowDelta = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const int iplane,
        const std::vector<RGYOpenCLEvent> &wait) {
        return launchPlane("kernel_rtgmc_retouch_edge_narrow_delta", dstFrame, srcFrame, iplane, wait, nullptr, calcRtgmcEdgeNarrowGain(prm.rtgmc_retouch));
    };
    auto launchEdgeNarrowGuardDelta = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const int iplane, const int smoothingMode) {
        return launchPlane((smoothingMode == 11) ? "kernel_rtgmc_retouch_edge_narrow_guard_delta11" : "kernel_rtgmc_retouch_edge_narrow_guard_delta", dstFrame, srcFrame, iplane, {}, nullptr);
    };
    auto launchDetailRollback = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *tmpFrame, const RGYFrameInfo *auxFrame,
        const RGYFrameInfo *srcFrame, const RGYFrameInfo *baseFrame, const int iplane, const char *dumpPrefix) {
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto tmpPlane = getPlane(tmpFrame, (RGY_PLANE)iplane);
        const auto auxPlane = getPlane(auxFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto basePlane = getPlane(baseFrame, (RGY_PLANE)iplane);
        if (!m_detailRollbackGauss) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-retouch detail rollback gauss resize filter is not initialized.\n"));
            return RGY_ERR_NULL_PTR;
        }
        RGYWorkSize local(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        const bool needRollbackDeltaDump = iplane == 0 && dumpPrefix != nullptr && m_lumaDumpEnabled
            && m_lumaDumpStage == std::string(dumpPrefix) + "_delta"
            && (m_lumaDumpTarget.empty() || m_lumaDumpTarget == "rollback");
        const bool mergeRollbackSmoothDelta = enableRollbackMerge && !disableFusionByEnv && !needRollbackDeltaDump;
        auto err = RGY_ERR_NONE;
        if (mergeRollbackSmoothDelta) {
            err = m_retouch.get()->kernel("kernel_rtgmc_retouch_smooth_delta_fused").config(queue, local, global, {}, nullptr).launch(
                (cl_mem)tmpPlane.ptr[0], tmpPlane.pitch[0],
                (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
                (cl_mem)basePlane.ptr[0], basePlane.pitch[0],
                tmpPlane.width, tmpPlane.height);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                    _T("kernel_rtgmc_retouch_smooth_delta_fused"), iplane, get_err_mes(err));
                return err;
            }
        } else {
            err = m_retouch.get()->kernel("kernel_rtgmc_retouch_make_delta").config(queue, local, global, {}, nullptr).launch(
                (cl_mem)auxPlane.ptr[0], auxPlane.pitch[0],
                (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
                (cl_mem)basePlane.ptr[0], basePlane.pitch[0],
                auxPlane.width, auxPlane.height);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                    _T("kernel_rtgmc_retouch_make_delta"), iplane, get_err_mes(err));
                return err;
            }
            if (iplane == 0 && dumpPrefix != nullptr) {
                const auto stage = std::string(dumpPrefix) + "_delta";
                err = dumpStageFrame(stage.c_str(), auxFrame, "rollback", queue, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }

            err = m_retouch.get()->kernel("kernel_rtgmc_retouch_removegrain12").config(queue, local, global, {}, nullptr).launch(
                (cl_mem)tmpPlane.ptr[0], tmpPlane.pitch[0],
                (cl_mem)auxPlane.ptr[0], auxPlane.pitch[0],
                tmpPlane.width, tmpPlane.height);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                    _T("kernel_rtgmc_retouch_removegrain12"), iplane, get_err_mes(err));
                return err;
            }
        }
        if (iplane == 0 && dumpPrefix != nullptr) {
            const auto stage = std::string(dumpPrefix) + "_smooth_delta";
            err = dumpStageFrame(stage.c_str(), tmpFrame, "rollback", queue, {});
            if (err != RGY_ERR_NONE) {
                return err;
            }
        }

        RGYOpenCLEvent evGauss;
        auto tmpGaussPlane = getPlane(tmpFrame, (RGY_PLANE)iplane);
        auto auxGaussPlane = getPlane(auxFrame, (RGY_PLANE)iplane);
        err = m_detailRollbackGauss->resizePlane(&auxGaussPlane, &tmpGaussPlane, queue, {}, &evGauss);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at rtgmc-retouch detail rollback gauss resize (plane %d): %s.\n"),
                iplane, get_err_mes(err));
            return err;
        }
        if (iplane == 0 && dumpPrefix != nullptr) {
            err = evGauss.wait();
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to wait rtgmc retouch detail rollback gauss resize: %s.\n"), get_err_mes(err));
                return err;
            }
            const auto stage = std::string(dumpPrefix) + "_soft_delta";
            err = dumpStageFrame(stage.c_str(), auxFrame, "rollback", queue, {});
            if (err != RGY_ERR_NONE) {
                return err;
            }
        }

        err = m_retouch.get()->kernel("kernel_rtgmc_retouch_make_delta").config(queue, local, global, { evGauss }, nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
            (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
            (cl_mem)auxPlane.ptr[0], auxPlane.pitch[0],
            dstPlane.width, dstPlane.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_retouch_make_delta"), iplane, get_err_mes(err));
        }
        return err;
    };
    auto launchDetailBoost = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const RGYFrameInfo *blurFrame, const int iplane,
        const std::vector<RGYOpenCLEvent> &wait) {
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto blurPlane = getPlane(blurFrame, (RGY_PLANE)iplane);
        RGYWorkSize local(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        auto err = m_retouch.get()->kernel("kernel_rtgmc_retouch_detail_boost").config(queue, local, global, wait, nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
            (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
            (cl_mem)blurPlane.ptr[0], blurPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            detailGain);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_retouch_detail_boost"), iplane, get_err_mes(err));
        }
        return err;
    };
    auto launchDetailBoostFused = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const int iplane,
        const std::vector<RGYOpenCLEvent> &wait) {
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        RGYWorkSize local(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        auto err = m_retouch.get()->kernel("kernel_rtgmc_retouch_detail_boost_fused").config(queue, local, global, wait, nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
            (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            prm.rtgmc_retouch.smode,
            prm.rtgmc_retouch.precise ? 1 : 0,
            detailGain);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_retouch_detail_boost_fused"), iplane, get_err_mes(err));
        }
        return err;
    };
    auto launchDetailBoostEdgeNarrowFused = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const int iplane,
        const std::vector<RGYOpenCLEvent> &wait) {
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        RGYWorkSize local(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        auto err = m_retouch.get()->kernel("kernel_rtgmc_retouch_detail_boost_edge_narrow_fused").config(queue, local, global, wait, nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
            (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            prm.rtgmc_retouch.smode,
            prm.rtgmc_retouch.precise ? 1 : 0,
            detailGain,
            calcRtgmcEdgeNarrowGain(prm.rtgmc_retouch));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_retouch_detail_boost_edge_narrow_fused"), iplane, get_err_mes(err));
        }
        return err;
    };
    auto launchAddDiff = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const RGYFrameInfo *diffFrame, const int iplane) {
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto diffPlane = getPlane(diffFrame, (RGY_PLANE)iplane);
        RGYWorkSize local(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        auto err = m_retouch.get()->kernel("kernel_rtgmc_retouch_adddiff").config(queue, local, global, {}, nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
            (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
            (cl_mem)diffPlane.ptr[0], diffPlane.pitch[0],
            dstPlane.width, dstPlane.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_retouch_adddiff"), iplane, get_err_mes(err));
        }
        return err;
    };
    auto launchEdgeNarrowFused = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const RGYFrameInfo *baseFrame,
        const int iplane, const std::vector<RGYOpenCLEvent> &wait) {
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto basePlane = getPlane(baseFrame, (RGY_PLANE)iplane);
        RGYWorkSize local(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        auto err = m_retouch.get()->kernel("kernel_rtgmc_retouch_edge_narrow_fused").config(queue, local, global, wait, nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
            (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
            (cl_mem)basePlane.ptr[0], basePlane.pitch[0],
            dstPlane.width, dstPlane.height,
            calcRtgmcEdgeNarrowGain(prm.rtgmc_retouch));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_retouch_edge_narrow_fused"), iplane, get_err_mes(err));
        }
        return err;
    };
    auto launchLimitSink = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, const RGYFrameInfo *baseFrame,
        const RGYFrameInfo *refLimitFrame, const RGYFrameInfo *motionBackLimitFrame, const RGYFrameInfo *motionForwLimitFrame,
        const int iplane) {
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto basePlane = getPlane(baseFrame, (RGY_PLANE)iplane);
        const auto refPlane = getPlane(refLimitFrame, (RGY_PLANE)iplane);
        const auto motionBackPlane = getPlane(motionBackLimitFrame, (RGY_PLANE)iplane);
        const auto motionForwPlane = getPlane(motionForwLimitFrame, (RGY_PLANE)iplane);
        RGYWorkSize local(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        auto err = m_retouch.get()->kernel("kernel_rtgmc_retouch_limit").config(queue, local, global, {}, nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
            (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
            (cl_mem)basePlane.ptr[0], basePlane.pitch[0],
            (cl_mem)refPlane.ptr[0], refPlane.pitch[0],
            (cl_mem)motionBackPlane.ptr[0], motionBackPlane.pitch[0],
            (cl_mem)motionForwPlane.ptr[0], motionForwPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            prm.rtgmc_retouch.slmode,
            prm.rtgmc_retouch.slrad,
            scaledSovs,
            prm.rtgmc_retouch.limit,
            hasTemporalLimitFrames ? 1 : 0);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_retouch_limit"), iplane, get_err_mes(err));
        }
        return err;
    };
    auto launchLimitSinkInlineComp = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame,
        const RGYFrameInfo *refLimitFrame, const int iplane) {
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        const auto refPlane = getPlane(refLimitFrame, (RGY_PLANE)iplane);
        const auto &compParams = m_temporalLimitFrames.inlineCompParams[iplane];
        RGYWorkSize local(RTGMC_RETOUCH_BLOCK_X, RTGMC_RETOUCH_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        auto err = m_retouch.get()->kernel("kernel_rtgmc_retouch_limit_inline_comp").config(queue, local, global, {}, nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
            (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
            (cl_mem)refPlane.ptr[0], refPlane.pitch[0],
            (cl_mem)compParams.cur, compParams.cur_pitch,
            (cl_mem)compParams.refBack,
            (cl_mem)compParams.refForw,
            compParams.refDirBack,
            compParams.refDirForw,
            (cl_mem)compParams.mv,
            (cl_mem)compParams.sad,
            compParams.blocksX,
            compParams.blocksY,
            compParams.blockSize,
            compParams.overlap,
            compParams.step,
            compParams.coveredWidth,
            compParams.coveredHeight,
            compParams.planeScaleX,
            compParams.planeScaleY,
            compParams.thsad,
            compParams.disableMask,
            (cl_mem)compParams.windowRamp,
            compParams.width,
            compParams.height,
            compParams.refs,
            compParams.pel,
            compParams.subpelInterp,
            dstPlane.width, dstPlane.height,
            scaledSovs);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_retouch_limit_inline_comp"), iplane, get_err_mes(err));
        }
        return err;
    };

    auto *curA = &m_frameBuf[1]->frame;
    auto *curB = &m_frameBuf[2]->frame;
    auto *work0 = &m_frameBuf[3]->frame;
    auto *work1 = &m_frameBuf[4]->frame;

    for (int iplane = 0; iplane < planes; iplane++) {
        const bool chromaPlane = isRetouchChromaPlane(iplane);
        const int smoothingMode = prm.rtgmc_retouch.precise ? 11 : 12;
        const auto &waitHere = (iplane == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        if (chromaPlane && !prm.processChroma) {
            auto err = launchCopy(pOutputFrame, pInputFrame, iplane, waitHere, (iplane == planes - 1) ? event : nullptr);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            continue;
        }
        const auto plan = buildRtgmcRetouchPlan(prm.rtgmc_retouch, chromaPlane, prm.skipPostTR2LimitModes, detailGain);

        if (iplane == 0) {
            auto err = dumpStageFrame("input", pInputFrame, "retouch", queue, waitHere);
            if (err != RGY_ERR_NONE) {
                return err;
            }
        }

        if (plan.nodes.empty()) {
            auto err = launchCopy(pOutputFrame, pInputFrame, iplane, waitHere, (iplane == planes - 1) ? event : nullptr);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            continue;
        }

        const RGYFrameInfo *curFrame = pInputFrame;
        RGYFrameInfo *curDst = curA;
        RGYFrameInfo *altDst = curB;

        for (size_t inode = 0; inode < plan.nodes.size(); inode++) {
            const auto &node = plan.nodes[inode];
            switch (node.kind) {
            case RtgmcRetouchNodeKind::DetailBoost: {
            const bool mergeDetailLine = enableDetailLineMerge
                && !disableFusion
                && iplane == 0
                && detailGain > 0.0f
                && prm.rtgmc_retouch.svthin > 0.0f
                && prm.rtgmc_retouch.smode > 0
                && inode + 1 < plan.nodes.size()
                && plan.nodes[inode + 1].kind == RtgmcRetouchNodeKind::EdgeNarrowCorrection;
            if (mergeDetailLine) {
                auto err = launchDetailBoostEdgeNarrowFused(curDst, pInputFrame, iplane, waitHere);
                if (err != RGY_ERR_NONE) {
                    return err;
                }
                curFrame = curDst;
                std::swap(curDst, altDst);
                inode++;
                break;
            }
            if (!disableFusion) {
                auto err = launchDetailBoostFused(curDst, pInputFrame, iplane, waitHere);
                if (err != RGY_ERR_NONE) {
                    return err;
                }
                curFrame = curDst;
                std::swap(curDst, altDst);
                break;
            }
            auto err = (prm.rtgmc_retouch.smode == 2)
                ? launchDetailBoostEdgeRef(work0, pInputFrame, iplane, waitHere)
                : RGY_ERR_NONE;
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0 && prm.rtgmc_retouch.smode == 2) {
                err = dumpStageFrame("detail_boost_edge_ref", work0, "retouch", queue, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            const RGYFrameInfo *blurInput = pInputFrame;
            if (prm.rtgmc_retouch.smode == 2) {
                blurInput = work0;
                if (prm.rtgmc_retouch.precise) {
                    err = launchPreciseClamp(work1, work0, pInputFrame, iplane);
                    if (err != RGY_ERR_NONE) {
                        return err;
                    }
                    if (iplane == 0) {
                        err = dumpStageFrame("detail_boost_regularized_ref", work1, "retouch", queue, {});
                        if (err != RGY_ERR_NONE) {
                            return err;
                        }
                    }
                    blurInput = work1;
                }
            }
            err = launchRemoveGrain(work1, blurInput, iplane, smoothingMode);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame("detail_boost_blur_ref", work1, "retouch", queue, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            err = launchDetailBoost(curDst, pInputFrame, work1, iplane, prm.rtgmc_retouch.smode == 2 ? std::vector<RGYOpenCLEvent>() : waitHere);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame("detail_boost", curDst, "retouch", queue, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            curFrame = curDst;
            std::swap(curDst, altDst);
            break;
        }

            case RtgmcRetouchNodeKind::EdgeNarrowCorrection: {
            if (!disableFusion) {
                const auto &thinWait = (curFrame == pInputFrame) ? waitHere : std::vector<RGYOpenCLEvent>();
                auto err = launchEdgeNarrowFused(altDst, curFrame, pInputFrame, iplane, thinWait);
                if (err != RGY_ERR_NONE) {
                    return err;
                }
                curFrame = altDst;
                std::swap(curDst, altDst);
                break;
            }
            auto err = launchEdgeNarrowDelta(work0, pInputFrame, iplane, (curFrame == pInputFrame) ? waitHere : std::vector<RGYOpenCLEvent>());
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame("edge_narrow_delta", work0, "edge_narrow", queue, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            err = launchBlurH(work1, work0, iplane);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame("edge_narrow_blur_delta", work1, "edge_narrow", queue, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            err = launchEdgeNarrowGuardDelta(work0, work1, iplane, smoothingMode);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame("edge_narrow_guard_delta", work0, "edge_narrow", queue, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            err = launchAddDiff(altDst, curFrame, work0, iplane);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame("edge_narrow", altDst, "edge_narrow", queue, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            curFrame = altDst;
            std::swap(curDst, altDst);
            break;
        }

            case RtgmcRetouchNodeKind::PreLimitRollback: {
            auto err = launchDetailRollback(altDst, work0, work1, curFrame, pInputFrame, iplane, node.dumpStage);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame(node.dumpStage, altDst, "rollback", queue, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            curFrame = altDst;
            std::swap(curDst, altDst);
            break;
        }

            case RtgmcRetouchNodeKind::SpatialOvershootGuard: {
            if (iplane == 0 && prm.rtgmc_retouch.slmode == 3) {
                auto err = dumpStageFrame("postlimit_spatial_guard_src", curFrame, "postlimit", queue, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
                err = dumpStageFrame("postlimit_spatial_guard_ref", baseFrame, "postlimit", queue, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            auto err = RGY_ERR_NONE;
            if (prm.rtgmc_retouch.slrad <= 1) {
                err = launchRepairMode1(altDst, curFrame, baseFrame, iplane);
            } else {
                err = launchRepairMode12(work0, curFrame, baseFrame, iplane);
                if (err == RGY_ERR_NONE) {
                    err = launchRepairMode1(altDst, curFrame, work0, iplane);
                }
            }
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame(node.dumpStage,
                    altDst, (prm.rtgmc_retouch.slmode == 3) ? "postlimit" : "retouch", queue, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            curFrame = altDst;
            std::swap(curDst, altDst);
            break;
        }

            case RtgmcRetouchNodeKind::TemporalOvershootGuard: {
            if (iplane == 0 && (prm.rtgmc_retouch.slmode == 2 || prm.rtgmc_retouch.slmode == 4)) {
                const char *dumpTarget = (prm.rtgmc_retouch.slmode == 4) ? "postlimit" : "limitover";
                const char *srcStage = (prm.rtgmc_retouch.slmode == 4) ? "postlimit_temporal_guard_src" : "temporal_guard_src";
                const char *refStage = (prm.rtgmc_retouch.slmode == 4) ? "postlimit_temporal_guard_ref" : "temporal_guard_ref";
                const char *motionBackStage = (prm.rtgmc_retouch.slmode == 4) ? "postlimit_temporal_guard_motionback" : "temporal_guard_motionback";
                const char *motionForwStage = (prm.rtgmc_retouch.slmode == 4) ? "postlimit_temporal_guard_motionforw" : "temporal_guard_motionforw";
                auto err = dumpStageFrame(srcStage, curFrame, dumpTarget, queue, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
                err = dumpStageFrame(refStage, refFrame, dumpTarget, queue, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
                if (!m_temporalLimitFrames.useInlineComp) {
                    err = dumpStageFrame(motionBackStage, motionBackFrame, dumpTarget, queue, {});
                    if (err != RGY_ERR_NONE) {
                        return err;
                    }
                    err = dumpStageFrame(motionForwStage, motionForwFrame, dumpTarget, queue, {});
                    if (err != RGY_ERR_NONE) {
                        return err;
                    }
                }
            }
            RGY_ERR err;
            const bool useInlineCompForPlane = m_temporalLimitFrames.useInlineComp && (!chromaPlane || m_temporalLimitFrames.inlineCompChroma);
            if (useInlineCompForPlane) {
                err = launchLimitSinkInlineComp(altDst, curFrame, refFrame, iplane);
            } else {
                const auto motionBackForPlane = motionBackFrame ? motionBackFrame : refFrame;
                const auto motionForwForPlane = motionForwFrame ? motionForwFrame : refFrame;
                err = launchLimitSink(altDst, curFrame, baseFrame, refFrame, motionBackForPlane, motionForwForPlane, iplane);
            }
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0 && (prm.rtgmc_retouch.slmode == 2 || prm.rtgmc_retouch.slmode == 4)) {
                err = dumpStageFrame(node.dumpStage,
                    altDst, (prm.rtgmc_retouch.slmode == 4) ? "postlimit" : "limitover", queue, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            curFrame = altDst;
            std::swap(curDst, altDst);
            break;
        }

            case RtgmcRetouchNodeKind::PostLimitRollback: {
            auto err = launchDetailRollback(altDst, work0, work1, curFrame, pInputFrame, iplane, node.dumpStage);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (iplane == 0) {
                err = dumpStageFrame(node.dumpStage, altDst, "rollback", queue, {});
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            curFrame = altDst;
            std::swap(curDst, altDst);
            break;
        }
            default:
                AddMessage(RGY_LOG_ERROR, _T("unknown rtgmc retouch node kind.\n"));
                return RGY_ERR_INVALID_PARAM;
            }
        }

        const auto &copyWait = (curFrame == pInputFrame) ? waitHere : std::vector<RGYOpenCLEvent>();
        auto err = launchCopy(pOutputFrame, curFrame, iplane, copyWait, (iplane == planes - 1) ? event : nullptr);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }

    copyFramePropWithoutRes(pOutputFrame, pInputFrame);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcRetouch::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;

    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (!m_retouch.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build/load RGY_FILTER_RTGMC_RETOUCH_CL (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }

    const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, m_frameBuf[0]->frame.mem_type);
    if (memcpyKind != RGYCLMemcpyD2D) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    auto pOutFrame = m_frameBuf[0].get();
    ppOutputFrames[0] = &pOutFrame->frame;
    *pOutputFrameNum = 1;

    auto prm = std::dynamic_pointer_cast<RGYFilterParamRtgmcRetouch>(m_param);
    if (!prm || (prm->rtgmc_retouch.smode == 0
        && prm->rtgmc_retouch.slmode == 0
        && prm->rtgmc_retouch.limit <= 0.0f
        && prm->rtgmc_retouch.svthin <= 0.0f
        && prm->rtgmc_retouch.sbb == 0)) {
        auto copyErr = m_cl->copyFrame(ppOutputFrames[0], pInputFrame, nullptr, queue, wait_events, event, RGYFrameCopyMode::FRAME, "rtgmc_retouch.noop_copy");
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy frame: %s.\n"), get_err_mes(copyErr));
            return copyErr;
        }
        copyFramePropWithoutRes(ppOutputFrames[0], pInputFrame);
        return RGY_ERR_NONE;
    }

    if ((prm->rtgmc_retouch.slmode == 2 || prm->rtgmc_retouch.slmode == 4) && m_temporalLimitFrames.any() && !m_temporalLimitFrames.valid()) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-retouch slmode=%d temporal detail guard requires ref/motionBack/motionForw frames together.\n"), prm->rtgmc_retouch.slmode);
        return RGY_ERR_INVALID_PARAM;
    }
    const bool hasTemporalLimitFrames = temporalLimitFramesReady(pInputFrame);
    if ((prm->rtgmc_retouch.slmode == 2 || prm->rtgmc_retouch.slmode == 4) && m_temporalLimitFrames.valid() && !temporalLimitFramesCompatible(pInputFrame)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-retouch slmode=%d temporal detail guard frames are not compatible with the current source frame.\n"), prm->rtgmc_retouch.slmode);
        return RGY_ERR_INVALID_PARAM;
    }
    if ((prm->rtgmc_retouch.slmode == 2 || prm->rtgmc_retouch.slmode == 4) && !hasTemporalLimitFrames && !m_loggedTemporalFallback) {
        AddMessage(RGY_LOG_DEBUG, _T("rtgmc-retouch slmode=%d temporal detail guard inputs are not wired; using spatial fallback.\n"), prm->rtgmc_retouch.slmode);
        m_loggedTemporalFallback = true;
    }

    return processFrame(ppOutputFrames[0], pInputFrame, *prm, queue, wait_events, event);
}

void RGYFilterRtgmcRetouch::resetTemporalState() {
    clearTemporalLimitFrames();
    clearSpatialLimitBaseFrame();
}

void RGYFilterRtgmcRetouch::close() {
    m_retouch.clear();
    m_buildOptions.clear();
    if (m_lumaDump.is_open()) {
        m_lumaDump.close();
    }
    m_lumaDumpPath.clear();
    m_lumaDumpStage = "edge_narrow_blur_delta";
    m_lumaDumpTarget.clear();
    m_lumaDumpMaxFrames = 0;
    m_lumaDumpFrameCount = 0;
    m_lumaDumpEnabled = false;
    m_lumaDumpHeaderWritten = false;
    m_lumaDumpChroma = false;
    m_detailRollbackGauss.reset();
    clearTemporalLimitFrames();
    clearSpatialLimitBaseFrame();
    m_frameBuf.clear();
    m_cl.reset();
}
