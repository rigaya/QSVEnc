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

#include "rgy_filter_rtgmc_lossless.h"

namespace {
static constexpr int RTGMC_LOSSLESS_BLOCK_X = 32;
static constexpr int RTGMC_LOSSLESS_BLOCK_Y = 8;
}

tstring RGYFilterParamRtgmcLossless::print() const {
    return strsprintf(_T("rtgmc-lossless: level=%d input_type=%d source_field=%d"),
        level, inputType, sourceField);
}

RGYFilterRtgmcLossless::RGYFilterRtgmcLossless(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_lossless(),
    m_buildOptions(),
    m_useKernel(false) {
    m_name = _T("rtgmc-lossless");
}

RGYFilterRtgmcLossless::~RGYFilterRtgmcLossless() {
    close();
}

RGY_ERR RGYFilterRtgmcLossless::checkParam(const std::shared_ptr<RGYFilterParamRtgmcLossless> &prm) {
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
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-lossless requires identical input/output csp and resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->level != 1 && prm->level != 2) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-lossless level must be 1 or 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->inputType == 1) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-lossless is incompatible with inputType=1.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->inputType < 0 || prm->inputType > 3) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-lossless inputType must be 0-3.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->sourceField != 0 && prm->sourceField != 1) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-lossless sourceField must be 0(top/even) or 1(bottom/odd).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.csp == RGY_CSP_NA || RGY_CSP_PLANES[prm->frameOut.csp] <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid colorspace.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcLossless::buildKernel(const std::shared_ptr<RGYFilterParamRtgmcLossless> &prm) {
    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelMax = (bitdepth >= 16) ? ((1 << 16) - 1) : ((1 << bitdepth) - 1);
    const int rangeHalf = 1 << (bitdepth - 1);
    m_buildOptions = strsprintf(
        "-D Type=%s -D max_val=%d -D range_half=%d -D rtgmc_lossless_block_x=%d -D rtgmc_lossless_block_y=%d",
        bitdepth > 8 ? "ushort" : "uchar",
        pixelMax,
        rangeHalf,
        RTGMC_LOSSLESS_BLOCK_X,
        RTGMC_LOSSLESS_BLOCK_Y);
    m_lossless.set(m_cl->buildResourceAsync(_T("RGY_FILTER_RTGMC_LOSSLESS_CL"), _T("EXE_DATA"), m_buildOptions.c_str()));
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcLossless::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamRtgmcLossless>(pParam);
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

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamRtgmcLossless>(m_param);
    if (m_useKernel
        && (!m_lossless.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp])) {
        sts = buildKernel(prm);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build rtgmc-lossless kernel.\n"));
            return sts;
        }
    }

    sts = AllocFrameBuf(prm->frameOut, 5);
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

RGY_ERR RGYFilterRtgmcLossless::processFrameFused(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pProcessedFrame, const RGYFrameInfo *pSourceFrame, int sourceField,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *kernelName = "kernel_rtgmc_lossless_apply_direct_delta";
    const int planes = RGY_CSP_PLANES[pProcessedFrame->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto processedPlane = getPlane(pProcessedFrame, (RGY_PLANE)iplane);
        const auto sourcePlane = getPlane(pSourceFrame, (RGY_PLANE)iplane);

        RGYWorkSize local(RTGMC_LOSSLESS_BLOCK_X, RTGMC_LOSSLESS_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        const auto &waitHere = (iplane == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        auto err = m_lossless.get()->kernel(kernelName).config(queue, local, global, waitHere, (iplane == planes - 1) ? event : nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
            (cl_mem)processedPlane.ptr[0], processedPlane.pitch[0],
            (cl_mem)sourcePlane.ptr[0], sourcePlane.pitch[0],
            dstPlane.width, dstPlane.height,
            sourceField);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                char_to_tstring(kernelName).c_str(), iplane, get_err_mes(err));
            return err;
        }
    }
    copyFramePropWithoutRes(pOutputFrame, pProcessedFrame);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcLossless::processFramePassSplit(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pProcessedFrame, const RGYFrameInfo *pSourceFrame, int sourceField,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto *pReferenceFrame = &m_frameBuf[3]->frame;
    auto *pDeltaFrame = &m_frameBuf[4]->frame;
    const int planes = RGY_CSP_PLANES[pProcessedFrame->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto processedPlane = getPlane(pProcessedFrame, (RGY_PLANE)iplane);
        const auto sourcePlane = getPlane(pSourceFrame, (RGY_PLANE)iplane);
        const auto referencePlane = getPlane(pReferenceFrame, (RGY_PLANE)iplane);
        const auto deltaPlane = getPlane(pDeltaFrame, (RGY_PLANE)iplane);

        RGYWorkSize local(RTGMC_LOSSLESS_BLOCK_X, RTGMC_LOSSLESS_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        const auto &waitHere = (iplane == 0) ? wait_events : std::vector<RGYOpenCLEvent>();

        auto err = m_lossless.get()->kernel("kernel_rtgmc_lossless_build_reference_frame").config(queue, local, global, waitHere, nullptr).launch(
            (cl_mem)referencePlane.ptr[0], referencePlane.pitch[0],
            (cl_mem)processedPlane.ptr[0], processedPlane.pitch[0],
            (cl_mem)sourcePlane.ptr[0], sourcePlane.pitch[0],
            referencePlane.width, referencePlane.height,
            sourceField);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_lossless_build_reference_frame"), iplane, get_err_mes(err));
            return err;
        }

        err = m_lossless.get()->kernel("kernel_rtgmc_lossless_build_delta_map").config(queue, local, global, {}, nullptr).launch(
            (cl_mem)deltaPlane.ptr[0], deltaPlane.pitch[0],
            (cl_mem)referencePlane.ptr[0], referencePlane.pitch[0],
            deltaPlane.width, deltaPlane.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_lossless_build_delta_map"), iplane, get_err_mes(err));
            return err;
        }

        err = m_lossless.get()->kernel("kernel_rtgmc_lossless_stabilize_delta_map").config(queue, local, global, {}, nullptr).launch(
            (cl_mem)referencePlane.ptr[0], referencePlane.pitch[0],
            (cl_mem)deltaPlane.ptr[0], deltaPlane.pitch[0],
            referencePlane.width, referencePlane.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_lossless_stabilize_delta_map"), iplane, get_err_mes(err));
            return err;
        }

        err = m_lossless.get()->kernel("kernel_rtgmc_lossless_apply_delta").config(queue, local, global, {}, (iplane == planes - 1) ? event : nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
            (cl_mem)processedPlane.ptr[0], processedPlane.pitch[0],
            (cl_mem)sourcePlane.ptr[0], sourcePlane.pitch[0],
            (cl_mem)referencePlane.ptr[0], referencePlane.pitch[0],
            dstPlane.width, dstPlane.height,
            sourceField);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                _T("kernel_rtgmc_lossless_apply_delta"), iplane, get_err_mes(err));
            return err;
        }
    }
    copyFramePropWithoutRes(pOutputFrame, pProcessedFrame);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcLossless::processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pProcessedFrame, const RGYFrameInfo *pSourceFrame, int sourceField,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *forcePassSplitEnv = std::getenv("QSVENC_RTGMC_LOSSLESS_FORCE_PASS_SPLIT");
    const bool forcePassSplit = forcePassSplitEnv != nullptr && forcePassSplitEnv[0] != '\0' && forcePassSplitEnv[0] != '0';
    if (forcePassSplit) {
        return processFramePassSplit(pOutputFrame, pProcessedFrame, pSourceFrame, sourceField, queue, wait_events, event);
    }
    return processFrameFused(pOutputFrame, pProcessedFrame, pSourceFrame, sourceField, queue, wait_events, event);
}

RGY_ERR RGYFilterRtgmcLossless::run_filter(const RGYFrameInfo *pProcessedFrame, const RGYFrameInfo *pSourceFrame, int sourceField, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
    RGYOpenCLEvent *event) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;

    if (!pProcessedFrame || !pProcessedFrame->ptr[0] || !pSourceFrame || !pSourceFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (sourceField != 0 && sourceField != 1) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-lossless sourceField must be 0(top/even) or 1(bottom/odd).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_useKernel && !m_lossless.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build/load RGY_FILTER_RTGMC_LOSSLESS_CL (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamRtgmcLossless>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto pOutFrame = m_frameBuf[0].get();
    ppOutputFrames[0] = &pOutFrame->frame;
    *pOutputFrameNum = 1;

    if (m_useKernel) {
        const auto processedMemcpyKind = getMemcpyKind(pProcessedFrame->mem_type, pOutFrame->frame.mem_type);
        const auto sourceMemcpyKind = getMemcpyKind(pSourceFrame->mem_type, pOutFrame->frame.mem_type);
        if (processedMemcpyKind == RGYCLMemcpyD2D && sourceMemcpyKind == RGYCLMemcpyD2D) {
            return processFrame(&pOutFrame->frame, pProcessedFrame, pSourceFrame, sourceField, queue, wait_events, event);
        }

        auto pProcessedTmp = &m_frameBuf[1]->frame;
        auto pSourceTmp = &m_frameBuf[2]->frame;
        RGYOpenCLEvent processedCopyEvent;
        auto copyErr = m_cl->copyFrame(pProcessedTmp, pProcessedFrame, nullptr, queue, wait_events, &processedCopyEvent, RGYFrameCopyMode::FRAME, "rtgmc_lossless.processed_tmp");
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-lossless processed frame: %s.\n"), get_err_mes(copyErr));
            return copyErr;
        }
        RGYOpenCLEvent sourceCopyEvent;
        copyErr = m_cl->copyFrame(pSourceTmp, pSourceFrame, nullptr, queue, wait_events, &sourceCopyEvent, RGYFrameCopyMode::FRAME, "rtgmc_lossless.source_tmp");
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-lossless source frame: %s.\n"), get_err_mes(copyErr));
            return copyErr;
        }
        return processFrame(&pOutFrame->frame, pProcessedTmp, pSourceTmp, sourceField, queue, { processedCopyEvent, sourceCopyEvent }, event);
    }

    auto copyErr = m_cl->copyFrame(ppOutputFrames[0], pProcessedFrame, nullptr, queue, wait_events, event, RGYFrameCopyMode::FRAME, "rtgmc_lossless.fallback_copy");
    if (copyErr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy frame: %s.\n"), get_err_mes(copyErr));
        return copyErr;
    }
    copyFramePropWithoutRes(ppOutputFrames[0], pProcessedFrame);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcLossless::run_filter(const RGYFrameInfo *pProcessedFrame, const RGYFrameInfo *pSourceFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
    RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamRtgmcLossless>(m_param);
    return run_filter(pProcessedFrame, pSourceFrame, prm ? prm->sourceField : 0, ppOutputFrames, pOutputFrameNum, queue, wait_events, event);
}

RGY_ERR RGYFilterRtgmcLossless::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
    RGYOpenCLEvent *event) {
    return run_filter(pInputFrame, pInputFrame, ppOutputFrames, pOutputFrameNum, queue, wait_events, event);
}

void RGYFilterRtgmcLossless::close() {
    m_lossless.clear();
    m_buildOptions.clear();
    m_frameBuf.clear();
    m_useKernel = false;
    m_cl.reset();
}
