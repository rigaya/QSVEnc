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

#include "rgy_filter_rtgmc_mmask.h"

#include <cmath>

namespace {
static constexpr int RTGMC_MMASK_BLOCK_X = 32;
static constexpr int RTGMC_MMASK_BLOCK_Y = 8;
}

RGYFilterParamRtgmcMMask::RGYFilterParamRtgmcMMask() :
    kind(1),
    time(100),
    ml(1.0),
    gamma(1.0) {
}

tstring RGYFilterParamRtgmcMMask::print() const {
    return strsprintf(_T("rtgmc-mmask: kind=%d time=%d ml=%.3f gamma=%.3f"),
        kind, time, ml, gamma);
}

RGYFilterRtgmcMMask::RGYFilterRtgmcMMask(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_mmask(),
    m_buildOptions(),
    m_useKernel(false) {
    m_name = _T("rtgmc-mmask");
}

RGYFilterRtgmcMMask::~RGYFilterRtgmcMMask() {
    close();
}

RGY_ERR RGYFilterRtgmcMMask::checkParam(const std::shared_ptr<RGYFilterParamRtgmcMMask> &prm) {
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
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask requires identical input/output csp and resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->kind != 1) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask supports only kind=1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->time != 100) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask supports only time=100.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->ml <= 0.0) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask ml must be positive.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->gamma <= 0.0) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask gamma must be positive.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.csp == RGY_CSP_NA || RGY_CSP_PLANES[prm->frameOut.csp] <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid colorspace.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcMMask::buildKernel(const std::shared_ptr<RGYFilterParamRtgmcMMask> &prm) {
    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelMax = (bitdepth >= 16) ? ((1 << 16) - 1) : ((1 << bitdepth) - 1);
    const int usePow = (std::abs(prm->gamma - 1.0) > 1.0e-6) ? 1 : 0;
    m_buildOptions = strsprintf(
        "-D Type=%s -D bit_depth=%d -D max_val=%d -D rtgmc_mmask_block_x=%d -D rtgmc_mmask_block_y=%d -D rtgmc_mmask_use_pow=%d",
        bitdepth > 8 ? "ushort" : "uchar",
        bitdepth,
        pixelMax,
        RTGMC_MMASK_BLOCK_X,
        RTGMC_MMASK_BLOCK_Y,
        usePow);
    m_mmask.set(m_cl->buildResourceAsync(_T("RGY_FILTER_RTGMC_MMASK_CL"), _T("EXE_DATA"), m_buildOptions.c_str()));
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcMMask::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamRtgmcMMask>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    m_pathThrough = FILTER_PATHTHROUGH_ALL;
    m_useKernel = (RGY_CSP_BIT_DEPTH[prm->frameOut.csp] <= 16);

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamRtgmcMMask>(m_param);
    if (m_useKernel
        && (!m_mmask.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
        || (std::abs(prmPrev->gamma - 1.0) <= 1.0e-6) != (std::abs(prm->gamma - 1.0) <= 1.0e-6))) {
        sts = buildKernel(prm);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build rtgmc-mmask kernel.\n"));
            return sts;
        }
    }

    sts = AllocFrameBuf(prm->frameOut, 3);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcMMask::checkAnalyzeResult(const RGYDegrainAnalyzeResult &analyzeResult, const RGYFrameInfo *pSourceFrame) {
    if (!analyzeResult.valid() || !analyzeResult.sad || !analyzeResult.sad->mem()) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask requires a valid degrain SAD analysis result.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (analyzeResult.layout.blockSize <= 0 || analyzeResult.layout.blockCount() <= 0 || analyzeResult.layout.temporalDirections <= 0
        || analyzeResult.layout.sadCount() != analyzeResult.layout.blockCount() * (size_t)analyzeResult.layout.temporalDirections) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask has invalid degrain SAD layout.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pSourceFrame
        && (analyzeResult.layout.coveredWidth > pSourceFrame->width || analyzeResult.layout.coveredHeight > pSourceFrame->height)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask degrain SAD layout exceeds frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcMMask::processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pSourceFrame, const RGYFrameInfo *pEdiFrame,
    const RGYDegrainAnalyzeResult &analyzeResult, const RGYFilterParamRtgmcMMask &prm,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const int planes = RGY_CSP_PLANES[pSourceFrame->csp];
    std::vector<RGYOpenCLEvent> waitWithAnalysis = wait_events;
    if (analyzeResult.event() != nullptr) {
        waitWithAnalysis.push_back(analyzeResult.event);
    }

    RGYOpenCLEvent planeEvent;
    std::vector<RGYOpenCLEvent> planeWait = waitWithAnalysis;
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(pSourceFrame, (RGY_PLANE)iplane);
        const auto ediPlane = getPlane(pEdiFrame, (RGY_PLANE)iplane);

        const bool processLuma = iplane == 0;
        const char *kernelName = processLuma ? "kernel_rtgmc_mmask_blend_y" : "kernel_rtgmc_mmask_copy";
        auto err = RGY_ERR_NONE;
        auto kernel = m_mmask.get()->kernel(kernelName)
            .config(queue, RGYWorkSize(RTGMC_MMASK_BLOCK_X, RTGMC_MMASK_BLOCK_Y), RGYWorkSize(dstPlane.width, dstPlane.height),
                planeWait, (iplane == planes - 1) ? event : &planeEvent);
        if (processLuma) {
            err = kernel.launch(
                (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
                (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
                (cl_mem)ediPlane.ptr[0], ediPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                analyzeResult.sad->mem(),
                analyzeResult.layout.blocksX,
                analyzeResult.layout.blocksY,
                analyzeResult.layout.coveredWidth,
                analyzeResult.layout.coveredHeight,
                analyzeResult.layout.step,
                analyzeResult.layout.blockSize,
                analyzeResult.layout.temporalDirections,
                (float)prm.ml,
                (float)prm.gamma);
        } else {
            err = kernel.launch(
                (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
                (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height);
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                char_to_tstring(kernelName).c_str(), iplane, get_err_mes(err));
            return err;
        }
        planeWait.clear();
        if (planeEvent() != nullptr) {
            planeWait.push_back(planeEvent);
        }
    }
    copyFramePropWithoutRes(pOutputFrame, pEdiFrame);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcMMask::run_filter(const RGYFrameInfo *pSourceFrame, const RGYFrameInfo *pEdiFrame, const RGYDegrainAnalyzeResult &analyzeResult,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
    RGYOpenCLEvent *event) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;

    if (!pSourceFrame || !pSourceFrame->ptr[0] || !pEdiFrame || !pEdiFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (pSourceFrame->csp != pEdiFrame->csp || pSourceFrame->width != pEdiFrame->width || pSourceFrame->height != pEdiFrame->height) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask source and edi frames must match csp and resolution.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto prm = std::dynamic_pointer_cast<RGYFilterParamRtgmcMMask>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto err = checkAnalyzeResult(analyzeResult, pSourceFrame);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    if (m_useKernel && !m_mmask.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build/load RGY_FILTER_RTGMC_MMASK_CL (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }

    auto pOutFrame = m_frameBuf[0].get();
    ppOutputFrames[0] = &pOutFrame->frame;
    *pOutputFrameNum = 1;

    if (m_useKernel) {
        const auto sourceMemcpyKind = getMemcpyKind(pSourceFrame->mem_type, pOutFrame->frame.mem_type);
        const auto ediMemcpyKind = getMemcpyKind(pEdiFrame->mem_type, pOutFrame->frame.mem_type);
        if (sourceMemcpyKind == RGYCLMemcpyD2D && ediMemcpyKind == RGYCLMemcpyD2D) {
            return processFrame(&pOutFrame->frame, pSourceFrame, pEdiFrame, analyzeResult, *prm, queue, wait_events, event);
        }

        auto pSourceTmp = &m_frameBuf[1]->frame;
        auto pEdiTmp = &m_frameBuf[2]->frame;
        RGYOpenCLEvent sourceCopyEvent;
        auto copyErr = m_cl->copyFrame(pSourceTmp, pSourceFrame, nullptr, queue, wait_events, &sourceCopyEvent, RGYFrameCopyMode::FRAME, "rtgmc_mmask.source_tmp");
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-mmask source frame: %s.\n"), get_err_mes(copyErr));
            return copyErr;
        }
        RGYOpenCLEvent ediCopyEvent;
        copyErr = m_cl->copyFrame(pEdiTmp, pEdiFrame, nullptr, queue, wait_events, &ediCopyEvent, RGYFrameCopyMode::FRAME, "rtgmc_mmask.edi_tmp");
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-mmask edi frame: %s.\n"), get_err_mes(copyErr));
            return copyErr;
        }
        return processFrame(&pOutFrame->frame, pSourceTmp, pEdiTmp, analyzeResult, *prm, queue, { sourceCopyEvent, ediCopyEvent }, event);
    }

    auto copyErr = m_cl->copyFrame(ppOutputFrames[0], pSourceFrame, nullptr, queue, wait_events, event, RGYFrameCopyMode::FRAME, "rtgmc_mmask.fallback_copy");
    if (copyErr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy frame: %s.\n"), get_err_mes(copyErr));
        return copyErr;
    }
    copyFramePropWithoutRes(ppOutputFrames[0], pEdiFrame);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcMMask::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
    RGYOpenCLEvent *event) {
    RGYDegrainAnalyzeResult analyzeResult;
    return run_filter(pInputFrame, pInputFrame, analyzeResult, ppOutputFrames, pOutputFrameNum, queue, wait_events, event);
}

void RGYFilterRtgmcMMask::close() {
    m_mmask.clear();
    m_buildOptions.clear();
    m_frameBuf.clear();
    m_useKernel = false;
    m_cl.reset();
}
