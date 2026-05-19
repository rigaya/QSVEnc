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

#include "rgy_filter_rtgmc_shimmer_repair.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <limits>
#include <vector>

namespace {
static constexpr int RTGMC_SHIMMER_REPAIR_BLOCK_X = 32;
static constexpr int RTGMC_SHIMMER_REPAIR_BLOCK_Y = 8;
static constexpr std::array<std::array<int, 4>, 8> RTGMC_SHIMMER_REPAIR_SUPPORT_RADIUS = {{
    {{ 1, 1, 2, 2 }},
    {{ 1, 1, 2, 2 }},
    {{ 1, 1, 2, 2 }},
    {{ 1, 1, 2, 2 }},
    {{ 2, 2, 2, 2 }},
    {{ 2, 2, 2, 2 }},
    {{ 2, 2, 2, 2 }},
    {{ 2, 2, 2, 2 }}
}};
static constexpr std::array<std::array<int, 4>, 8> RTGMC_SHIMMER_REPAIR_MIN_SUPPORT_PIXELS = {{
    {{ 1, 1, 1, 1 }},
    {{ 1, 1, 1, 1 }},
    {{ 2, 1, 1, 1 }},
    {{ 2, 1, 1, 1 }},
    {{ 3, 2, 2, 2 }},
    {{ 3, 2, 2, 2 }},
    {{ 4, 3, 3, 3 }},
    {{ 4, 3, 3, 3 }}
}};

static const char *rtgmcShimmerRepairTargetName(const RGYRtgmcShimmerRepairStage stage) {
    return (stage == RGYRtgmcShimmerRepairStage::PreRetouch) ? "rep1" : "rep2";
}

static const TCHAR *rtgmcShimmerRepairStageName(const RGYRtgmcShimmerRepairStage stage) {
    return (stage == RGYRtgmcShimmerRepairStage::PreRetouch) ? _T("pre-retouch") : _T("post-tr2");
}

static int rtgmcShimmerRepairSupportRadius(const RGYRtgmcRepairProfile& profile) {
    return RTGMC_SHIMMER_REPAIR_SUPPORT_RADIUS[profile.thinRejectLevel][profile.restorePaddingLevel];
}

static int rtgmcShimmerRepairMinSupportPixels(const RGYRtgmcRepairProfile& profile) {
    return RTGMC_SHIMMER_REPAIR_MIN_SUPPORT_PIXELS[profile.thinRejectLevel][profile.restorePaddingLevel];
}

static void rtgmcShimmerRepairLoadProfile(RGYFilterParamRtgmcShimmerRepair *prm) {
    prm->repairProfile = rgy_rtgmc_repair_profile_from_levels(prm->repairThin, prm->repairPad);
}
}

tstring RGYFilterParamRtgmcShimmerRepair::print() const {
    return strsprintf(_T("rtgmc-shimmer-repair: stage=%s repair-thin=%d repair-pad=%d process_chroma=%s"),
        rtgmcShimmerRepairStageName(stage), repairThin, repairPad, processChroma ? _T("true") : _T("false"));
}

RGYFilterRtgmcShimmerRepair::RGYFilterRtgmcShimmerRepair(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_shimmerRepair(),
    m_buildOptions(),
    m_lumaDump(),
    m_lumaDumpPath(),
    m_lumaDumpStage("shimmer_corrected"),
    m_lumaDumpTarget(),
    m_lumaDumpMaxFrames(0),
    m_lumaDumpFrameCount(0),
    m_lumaDumpEnabled(false),
    m_lumaDumpHeaderWritten(false),
    m_lumaDumpFullYuv(false),
    m_useKernel(false) {
    m_name = _T("rtgmc-shimmer-repair");
}

RGYFilterRtgmcShimmerRepair::~RGYFilterRtgmcShimmerRepair() {
    close();
}

RGY_ERR RGYFilterRtgmcShimmerRepair::checkParam(const std::shared_ptr<RGYFilterParamRtgmcShimmerRepair> &prm) {
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
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-shimmer-repair requires identical input/output csp and resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (!rgy_rtgmc_repair_thin_level_is_valid(prm->repairThin)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-shimmer-repair rep-thin must be 0-7.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!rgy_rtgmc_repair_pad_level_is_valid(prm->repairPad)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-shimmer-repair rep-pad must be 0-3.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcShimmerRepair::buildKernels(const std::shared_ptr<RGYFilterParamRtgmcShimmerRepair> &prm) {
    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelMax = (bitdepth >= 16) ? ((1 << 16) - 1) : ((1 << bitdepth) - 1);
    const int rangeHalf = 1 << (bitdepth - 1);
    const auto profile = prm->repairProfile;
    const int supportRadius = rtgmcShimmerRepairSupportRadius(profile);
    const int minSupportPixels = rtgmcShimmerRepairMinSupportPixels(profile);
    m_buildOptions = strsprintf(
        "-D Type=%s -D bit_depth=%d -D max_val=%d -D range_half=%d -D rtgmc_shimmer_repair_block_x=%d -D rtgmc_shimmer_repair_block_y=%d"
        " -D RTGMC_SHIMMER_REPAIR_SUPPORT_RADIUS=%d -D RTGMC_SHIMMER_REPAIR_MIN_SUPPORT_PIXELS=%d -D RTGMC_SHIMMER_REPAIR_RESTORE_PADDING_LEVEL=%d",
        bitdepth > 8 ? "ushort" : "uchar",
        bitdepth,
        pixelMax,
        rangeHalf,
        RTGMC_SHIMMER_REPAIR_BLOCK_X,
        RTGMC_SHIMMER_REPAIR_BLOCK_Y,
        supportRadius,
        minSupportPixels,
        profile.restorePaddingLevel);
    m_shimmerRepair.set(m_cl->buildResourceAsync(_T("RGY_FILTER_RTGMC_SHIMMER_REPAIR_CL"), _T("EXE_DATA"), m_buildOptions.c_str()));
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcShimmerRepair::initLumaDump(const RGYFrameInfo &frameInfo, const RGYFilterParamRtgmcShimmerRepair &prm) {
    m_lumaDumpEnabled = false;
    m_lumaDumpHeaderWritten = false;
    m_lumaDumpFrameCount = 0;
    m_lumaDumpMaxFrames = 0;
    m_lumaDumpPath.clear();
    m_lumaDumpStage = "shimmer_corrected";
    m_lumaDumpTarget.clear();
    m_lumaDumpFullYuv = false;
    if (m_lumaDump.is_open()) {
        m_lumaDump.close();
    }

    const char *dumpPathEnv = std::getenv("QSVENC_RTGMC_REP_YUV_DUMP_Y4M");
    if (dumpPathEnv != nullptr && dumpPathEnv[0] != '\0') {
        m_lumaDumpFullYuv = true;
    } else {
        dumpPathEnv = std::getenv("QSVENC_RTGMC_REP_LUMA_DUMP_Y4M");
    }
    if (dumpPathEnv == nullptr || dumpPathEnv[0] == '\0') {
        return RGY_ERR_NONE;
    }
    m_lumaDumpPath = dumpPathEnv;

    if (const char *stageEnv = std::getenv("QSVENC_RTGMC_REP_LUMA_DUMP_STAGE"); stageEnv != nullptr && stageEnv[0] != '\0') {
        m_lumaDumpStage = stageEnv;
        std::transform(m_lumaDumpStage.begin(), m_lumaDumpStage.end(), m_lumaDumpStage.begin(),
            [](unsigned char c) { return (char)std::tolower(c); });
    }
    if (m_lumaDumpStage != "correction_delta" && m_lumaDumpStage != "positive_correction_gate"
        && m_lumaDumpStage != "negative_correction_gate" && m_lumaDumpStage != "shimmer_corrected") {
        AddMessage(RGY_LOG_ERROR, _T("unsupported rtgmc rep luma dump stage: %s.\n"),
            char_to_tstring(m_lumaDumpStage).c_str());
        return RGY_ERR_INVALID_PARAM;
    }

    if (const char *targetEnv = std::getenv("QSVENC_RTGMC_REP_LUMA_DUMP_TARGET"); targetEnv != nullptr && targetEnv[0] != '\0') {
        m_lumaDumpTarget = targetEnv;
        std::transform(m_lumaDumpTarget.begin(), m_lumaDumpTarget.end(), m_lumaDumpTarget.begin(),
            [](unsigned char c) { return (char)std::tolower(c); });
    }
    const char *activeTarget = (prm.repairThin > 0) ? rtgmcShimmerRepairTargetName(prm.stage) : "";
    if (!m_lumaDumpTarget.empty() && m_lumaDumpTarget != activeTarget) {
        AddMessage(RGY_LOG_DEBUG, _T("rtgmc rep luma dump target %s skipped for inactive %s instance.\n"),
            char_to_tstring(m_lumaDumpTarget).c_str(), char_to_tstring(activeTarget).c_str());
        return RGY_ERR_NONE;
    }

    const int bitdepth = RGY_CSP_BIT_DEPTH[frameInfo.csp];
    if (bitdepth > 8) {
        AddMessage(RGY_LOG_WARN, _T("rtgmc rep stage dump supports only 8bit input, disabling dump for %s.\n"),
            RGY_CSP_NAMES[frameInfo.csp]);
        return RGY_ERR_NONE;
    }
    if (m_lumaDumpFullYuv && RGY_CSP_CHROMA_FORMAT[frameInfo.csp] != RGY_CHROMAFMT_YUV420) {
        AddMessage(RGY_LOG_WARN, _T("QSVENC_RTGMC_REP_YUV_DUMP_Y4M supports only 4:2:0 input, disabling dump for %s.\n"),
            RGY_CSP_NAMES[frameInfo.csp]);
        return RGY_ERR_NONE;
    }
    if (!m_lumaDumpFullYuv && RGY_CSP_CHROMA_FORMAT[frameInfo.csp] != RGY_CHROMAFMT_YUV420 && RGY_CSP_PLANES[frameInfo.csp] != 1) {
        AddMessage(RGY_LOG_WARN, _T("QSVENC_RTGMC_REP_LUMA_DUMP_Y4M supports only 4:2:0/Y8 input, disabling dump for %s.\n"),
            RGY_CSP_NAMES[frameInfo.csp]);
        return RGY_ERR_NONE;
    }

    const char *maxFrames = std::getenv(m_lumaDumpFullYuv
        ? "QSVENC_RTGMC_REP_YUV_DUMP_MAX_FRAMES"
        : "QSVENC_RTGMC_REP_LUMA_DUMP_MAX_FRAMES");
    if (maxFrames == nullptr || maxFrames[0] == '\0') {
        maxFrames = std::getenv("QSVENC_RTGMC_REP_LUMA_DUMP_MAX_FRAMES");
    }
    if (maxFrames != nullptr && maxFrames[0] != '\0') {
        char *endptr = nullptr;
        const long parsed = std::strtol(maxFrames, &endptr, 10);
        if (endptr != maxFrames && parsed > 0) {
            m_lumaDumpMaxFrames = (int)std::min<long>(parsed, std::numeric_limits<int>::max());
        }
    }

    m_lumaDump.open(m_lumaDumpPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!m_lumaDump) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open rtgmc rep luma dump: %s.\n"),
            char_to_tstring(m_lumaDumpPath).c_str());
        return RGY_ERR_FILE_OPEN;
    }
    m_lumaDumpEnabled = true;
    AddMessage(RGY_LOG_INFO, _T("rtgmc rep %s dump enabled: %s (target=%s, stage=%s).\n"),
        m_lumaDumpFullYuv ? _T("yuv") : _T("luma"),
        char_to_tstring(m_lumaDumpPath).c_str(), char_to_tstring(activeTarget).c_str(), char_to_tstring(m_lumaDumpStage).c_str());
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcShimmerRepair::dumpLumaFrame(const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
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
    if (bitdepth > 8 || (m_lumaDumpFullYuv && RGY_CSP_CHROMA_FORMAT[frame->csp] != RGY_CHROMAFMT_YUV420)
        || (!m_lumaDumpFullYuv && RGY_CSP_CHROMA_FORMAT[frame->csp] != RGY_CHROMAFMT_YUV420 && RGY_CSP_PLANES[frame->csp] != 1)) {
        AddMessage(RGY_LOG_WARN, _T("rtgmc rep luma dump disabled by unsupported frame csp: %s.\n"),
            RGY_CSP_NAMES[frame->csp]);
        m_lumaDumpEnabled = false;
        return RGY_ERR_NONE;
    }

    const auto planeY = getPlane(frame, RGY_PLANE_Y);
    std::vector<uint8_t> hostY((size_t)planeY.width * planeY.height);
    std::vector<uint8_t> hostU;
    std::vector<uint8_t> hostV;
    RGYFrameInfo hostFrame(planeY.width, planeY.height, RGY_CSP_Y8, 8, frame->picstruct, RGY_MEM_TYPE_CPU);
    hostFrame.ptr[0] = hostY.data();
    hostFrame.pitch[0] = planeY.width;
    if (m_lumaDumpFullYuv) {
        const auto planeU = getPlane(frame, RGY_PLANE_U);
        const auto planeV = getPlane(frame, RGY_PLANE_V);
        hostU.resize((size_t)planeU.width * planeU.height);
        hostV.resize((size_t)planeV.width * planeV.height);
        hostFrame.ptr[1] = hostU.data();
        hostFrame.ptr[2] = hostV.data();
        hostFrame.pitch[1] = planeU.width;
        hostFrame.pitch[2] = planeV.width;
    }

    RGYOpenCLEvent readEventY;
    auto err = m_cl->copyPlane(&hostFrame, &planeY, nullptr, queue, wait_events, &readEventY);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to read rtgmc rep luma dump frame: %s.\n"), get_err_mes(err));
        return err;
    }
    err = readEventY.wait();
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to wait rtgmc rep luma dump read: %s.\n"), get_err_mes(err));
        return err;
    }
    if (m_lumaDumpFullYuv) {
        const auto planeU = getPlane(frame, RGY_PLANE_U);
        const auto planeV = getPlane(frame, RGY_PLANE_V);
        RGYFrameInfo hostPlaneU(planeU.width, planeU.height, RGY_CSP_Y8, 8, frame->picstruct, RGY_MEM_TYPE_CPU);
        RGYFrameInfo hostPlaneV(planeV.width, planeV.height, RGY_CSP_Y8, 8, frame->picstruct, RGY_MEM_TYPE_CPU);
        hostPlaneU.ptr[0] = hostU.data();
        hostPlaneV.ptr[0] = hostV.data();
        hostPlaneU.pitch[0] = planeU.width;
        hostPlaneV.pitch[0] = planeV.width;
        RGYOpenCLEvent readEventU, readEventV;
        err = m_cl->copyPlane(&hostPlaneU, &planeU, nullptr, queue, {}, &readEventU);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to read rtgmc rep u dump frame: %s.\n"), get_err_mes(err));
            return err;
        }
        err = readEventU.wait();
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to wait rtgmc rep u dump read: %s.\n"), get_err_mes(err));
            return err;
        }
        err = m_cl->copyPlane(&hostPlaneV, &planeV, nullptr, queue, {}, &readEventV);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to read rtgmc rep v dump frame: %s.\n"), get_err_mes(err));
            return err;
        }
        err = readEventV.wait();
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to wait rtgmc rep v dump read: %s.\n"), get_err_mes(err));
            return err;
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
    if (m_lumaDumpFullYuv) {
        for (int y = 0; y < chromaHeight; y++) {
            m_lumaDump.write(reinterpret_cast<const char *>(hostFrame.ptr[1] + (size_t)y * hostFrame.pitch[1]), chromaWidth);
        }
        for (int y = 0; y < chromaHeight; y++) {
            m_lumaDump.write(reinterpret_cast<const char *>(hostFrame.ptr[2] + (size_t)y * hostFrame.pitch[2]), chromaWidth);
        }
    } else {
        std::vector<uint8_t> neutralUV((size_t)chromaWidth * chromaHeight, 128);
        m_lumaDump.write(reinterpret_cast<const char *>(neutralUV.data()), neutralUV.size());
        m_lumaDump.write(reinterpret_cast<const char *>(neutralUV.data()), neutralUV.size());
    }
    if (!m_lumaDump) {
        AddMessage(RGY_LOG_ERROR, _T("failed to write rtgmc rep luma dump: %s.\n"),
            char_to_tstring(m_lumaDumpPath).c_str());
        return RGY_ERR_FILE_OPEN;
    }
    m_lumaDumpFrameCount++;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcShimmerRepair::dumpStageFrame(const char *stage, const RGYFrameInfo *frame, const char *target,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    if (!m_lumaDumpEnabled || m_lumaDumpStage != stage || (!m_lumaDumpTarget.empty() && m_lumaDumpTarget != target)) {
        return RGY_ERR_NONE;
    }
    return dumpLumaFrame(frame, queue, wait_events);
}

RGY_ERR RGYFilterRtgmcShimmerRepair::launchRtgmcShimmerRepairFused(
    RGYFrameInfo *pOutputFrame,
    RGYFrameInfo *pCorrectionDeltaFrame,
    RGYFrameInfo *pPositiveCorrectionGateFrame,
    RGYFrameInfo *pNegativeCorrectionGateFrame,
    const RGYFrameInfo *pInputFrame,
    const RGYFrameInfo *pRefFrame,
    const RGYFilterParamRtgmcShimmerRepair &prm,
    int iplane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const auto outPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
    const auto deltaPlane = getPlane(pCorrectionDeltaFrame, (RGY_PLANE)iplane);
    const auto positivePlane = getPlane(pPositiveCorrectionGateFrame, (RGY_PLANE)iplane);
    const auto negativePlane = getPlane(pNegativeCorrectionGateFrame, (RGY_PLANE)iplane);
    const auto inputPlane = getPlane(pInputFrame, (RGY_PLANE)iplane);
    const auto refPlane = getPlane(pRefFrame, (RGY_PLANE)iplane);
    RGYWorkSize local(RTGMC_SHIMMER_REPAIR_BLOCK_X, RTGMC_SHIMMER_REPAIR_BLOCK_Y);
    RGYWorkSize global(outPlane.width, outPlane.height);
    auto kernel = m_shimmerRepair.get()->kernel("kernel_rtgmc_shimmer_repair_apply_fused").config(queue, local, global, wait_events, event);
    auto err = kernel.launch(
        (cl_mem)outPlane.ptr[0], outPlane.pitch[0],
        (cl_mem)deltaPlane.ptr[0], deltaPlane.pitch[0],
        (cl_mem)positivePlane.ptr[0], positivePlane.pitch[0],
        (cl_mem)negativePlane.ptr[0], negativePlane.pitch[0],
        (cl_mem)inputPlane.ptr[0], inputPlane.pitch[0],
        (cl_mem)refPlane.ptr[0], refPlane.pitch[0],
        outPlane.width, outPlane.height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
            _T("kernel_rtgmc_shimmer_repair_apply_fused"), iplane, get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterRtgmcShimmerRepair::launchRtgmcShimmerRepairApply(
    RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pInputFrame,
    const RGYFrameInfo *pRefFrame,
    const RGYFilterParamRtgmcShimmerRepair &prm,
    int iplane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const auto outPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
    const auto inputPlane = getPlane(pInputFrame, (RGY_PLANE)iplane);
    const auto refPlane = getPlane(pRefFrame, (RGY_PLANE)iplane);
    RGYWorkSize local(RTGMC_SHIMMER_REPAIR_BLOCK_X, RTGMC_SHIMMER_REPAIR_BLOCK_Y);
    RGYWorkSize global(outPlane.width, outPlane.height);
    auto kernel = m_shimmerRepair.get()->kernel("kernel_rtgmc_shimmer_repair_apply").config(queue, local, global, wait_events, event);
    auto err = kernel.launch(
        (cl_mem)outPlane.ptr[0], outPlane.pitch[0],
        (cl_mem)inputPlane.ptr[0], inputPlane.pitch[0],
        (cl_mem)refPlane.ptr[0], refPlane.pitch[0],
        outPlane.width, outPlane.height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
            _T("kernel_rtgmc_shimmer_repair_apply"), iplane, get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterRtgmcShimmerRepair::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamRtgmcShimmerRepair>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    rtgmcShimmerRepairLoadProfile(prm.get());

    m_pathThrough = FILTER_PATHTHROUGH_ALL;
    m_useKernel = (RGY_CSP_BIT_DEPTH[prm->frameOut.csp] <= 16);

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamRtgmcShimmerRepair>(m_param);
    if (m_useKernel
        && (!m_shimmerRepair.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
        || prmPrev->repairThin != prm->repairThin
        || prmPrev->repairPad != prm->repairPad)) {
        sts = buildKernels(prm);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build rtgmc-shimmer-repair kernel.\n"));
            return sts;
        }
    }

    sts = AllocFrameBuf(prm->frameOut, 8);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    sts = initLumaDump(prm->frameOut, *prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcShimmerRepair::processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame,
    const RGYFilterParamRtgmcShimmerRepair &prm,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    const int repair = prm.repairThin;
    const char *target = (repair > 0) ? rtgmcShimmerRepairTargetName(prm.stage) : "";

    auto launchCopy = [&](const RGYFrameInfo *dstFrame, const RGYFrameInfo *srcFrame, int iplane, const std::vector<RGYOpenCLEvent> &wait, RGYOpenCLEvent *ev) {
        const char *kernelName = "kernel_rtgmc_shimmer_repair_copy";
        const auto dstPlane = getPlane(dstFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(srcFrame, (RGY_PLANE)iplane);
        RGYWorkSize local(RTGMC_SHIMMER_REPAIR_BLOCK_X, RTGMC_SHIMMER_REPAIR_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        auto err = m_shimmerRepair.get()->kernel(kernelName).config(queue, local, global, wait, ev).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
            (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
            dstPlane.width, dstPlane.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                char_to_tstring(kernelName).c_str(), iplane, get_err_mes(err));
        }
        return err;
    };
    auto dumpProcessedStages = [&](const char *stageTarget, RGYFrameInfo *deltaFrame, RGYFrameInfo *positiveGateFrame,
        RGYFrameInfo *negativeGateFrame, RGYFrameInfo *outputFrame, bool dumpNow) -> RGY_ERR {
        if (!dumpNow) {
            return RGY_ERR_NONE;
        }
        auto err = dumpStageFrame("correction_delta", deltaFrame, stageTarget, queue, {});
        if (err != RGY_ERR_NONE) return err;
        err = dumpStageFrame("positive_correction_gate", positiveGateFrame, stageTarget, queue, {});
        if (err != RGY_ERR_NONE) return err;
        err = dumpStageFrame("negative_correction_gate", negativeGateFrame, stageTarget, queue, {});
        if (err != RGY_ERR_NONE) return err;
        err = dumpStageFrame("shimmer_corrected", outputFrame, stageTarget, queue, {});
        if (err != RGY_ERR_NONE) return err;
        return RGY_ERR_NONE;
    };

    for (int iplane = 0; iplane < planes; iplane++) {
        const bool processPlane = (iplane == 0 || prm.processChroma);
        const auto &waitHere = (iplane == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        if (!processPlane || repair == 0) {
            auto err = launchCopy(pOutputFrame, pInputFrame, iplane, waitHere, (iplane == planes - 1) ? event : nullptr);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (m_lumaDumpEnabled && m_lumaDumpFullYuv && repair != 0) {
                err = launchCopy(&m_frameBuf[1]->frame, pInputFrame, iplane, waitHere, nullptr);
                if (err != RGY_ERR_NONE) return err;
                err = launchCopy(&m_frameBuf[2]->frame, pInputFrame, iplane, waitHere, nullptr);
                if (err != RGY_ERR_NONE) return err;
                err = launchCopy(&m_frameBuf[3]->frame, pInputFrame, iplane, waitHere, nullptr);
                if (err != RGY_ERR_NONE) return err;
            }
            continue;
        }

        RGYFrameInfo *correctionDelta = &m_frameBuf[1]->frame;
        RGYFrameInfo *positiveCorrectionGate = &m_frameBuf[2]->frame;
        RGYFrameInfo *negativeCorrectionGate = &m_frameBuf[3]->frame;

        const auto planeEvent = (iplane == planes - 1) ? event : nullptr;
        auto err = m_lumaDumpEnabled
            ? launchRtgmcShimmerRepairFused(
                pOutputFrame, correctionDelta, positiveCorrectionGate, negativeCorrectionGate,
                pInputFrame, pRefFrame, prm, iplane, queue, waitHere, planeEvent)
            : launchRtgmcShimmerRepairApply(
                pOutputFrame, pInputFrame, pRefFrame, prm, iplane, queue, waitHere, planeEvent);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        if (iplane == 0 && !m_lumaDumpFullYuv) {
            err = dumpProcessedStages(target, correctionDelta, positiveCorrectionGate, negativeCorrectionGate, pOutputFrame, true);
            if (err != RGY_ERR_NONE) return err;
        }
    }
    if (m_lumaDumpFullYuv) {
        RGY_ERR err = RGY_ERR_NONE;
        if (m_lumaDumpStage == "correction_delta") {
            err = dumpStageFrame("correction_delta", &m_frameBuf[1]->frame, target, queue, {});
        } else if (m_lumaDumpStage == "positive_correction_gate") {
            err = dumpStageFrame("positive_correction_gate", &m_frameBuf[2]->frame, target, queue, {});
        } else if (m_lumaDumpStage == "negative_correction_gate") {
            err = dumpStageFrame("negative_correction_gate", &m_frameBuf[3]->frame, target, queue, {});
        } else if (m_lumaDumpStage == "shimmer_corrected") {
            err = dumpStageFrame("shimmer_corrected", pOutputFrame, target, queue, {});
        }
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    copyFramePropWithoutRes(pOutputFrame, pInputFrame);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcShimmerRepair::run_filter(const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
    RGYOpenCLEvent *event) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;

    if (!pInputFrame || !pInputFrame->ptr[0] || !pRefFrame || !pRefFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (m_useKernel && !m_shimmerRepair.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build/load RGY_FILTER_RTGMC_SHIMMER_REPAIR_CL (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamRtgmcShimmerRepair>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto pOutFrame = m_frameBuf[0].get();
    ppOutputFrames[0] = &pOutFrame->frame;
    *pOutputFrameNum = 1;

    if (m_useKernel) {
        const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, m_frameBuf[0]->frame.mem_type);
        const auto refMemcpyKind = getMemcpyKind(pRefFrame->mem_type, m_frameBuf[0]->frame.mem_type);
        if (memcpyKind == RGYCLMemcpyD2D && refMemcpyKind == RGYCLMemcpyD2D) {
            auto err = processFrame(&pOutFrame->frame, pInputFrame, pRefFrame, *prm, queue, wait_events, event);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            return RGY_ERR_NONE;
        }
    }

    if (m_useKernel) {
        auto pInputTmp = &m_frameBuf[6]->frame;
        auto pRefTmp = &m_frameBuf[7]->frame;
        RGYOpenCLEvent inputCopyEvent;
        auto copyErr = m_cl->copyFrame(pInputTmp, pInputFrame, nullptr, queue, wait_events, &inputCopyEvent, RGYFrameCopyMode::FRAME, "rtgmc_shimmer.input_tmp");
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-shimmer-repair input frame: %s.\n"), get_err_mes(copyErr));
            return copyErr;
        }
        RGYOpenCLEvent refCopyEvent;
        copyErr = m_cl->copyFrame(pRefTmp, pRefFrame, nullptr, queue, wait_events, &refCopyEvent, RGYFrameCopyMode::FRAME, "rtgmc_shimmer.ref_tmp");
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-shimmer-repair reference frame: %s.\n"), get_err_mes(copyErr));
            return copyErr;
        }
        auto err = processFrame(&pOutFrame->frame, pInputTmp, pRefTmp, *prm, queue, { inputCopyEvent, refCopyEvent }, event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        return RGY_ERR_NONE;
    }

    auto copyErr = m_cl->copyFrame(ppOutputFrames[0], pInputFrame, nullptr, queue, wait_events, event, RGYFrameCopyMode::FRAME, "rtgmc_shimmer.fallback_copy");
    if (copyErr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy frame: %s.\n"), get_err_mes(copyErr));
        return copyErr;
    }
    copyFramePropWithoutRes(ppOutputFrames[0], pInputFrame);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcShimmerRepair::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
    RGYOpenCLEvent *event) {
    return run_filter(pInputFrame, pInputFrame, ppOutputFrames, pOutputFrameNum, queue, wait_events, event);
}

void RGYFilterRtgmcShimmerRepair::close() {
    if (m_lumaDump.is_open()) {
        m_lumaDump.close();
    }
    m_lumaDumpPath.clear();
    m_lumaDumpStage = "shimmer_corrected";
    m_lumaDumpTarget.clear();
    m_lumaDumpMaxFrames = 0;
    m_lumaDumpFrameCount = 0;
    m_lumaDumpEnabled = false;
    m_lumaDumpHeaderWritten = false;
    m_shimmerRepair.clear();
    m_buildOptions.clear();
    m_frameBuf.clear();
    m_useKernel = false;
    m_cl.reset();
}
