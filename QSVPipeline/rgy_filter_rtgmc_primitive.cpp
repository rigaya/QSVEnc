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

#include "rgy_filter_rtgmc_primitive.h"

#include <algorithm>
#include <vector>

namespace {
static constexpr int RTGMC_PRIMITIVE_BLOCK_X = 32;
static constexpr int RTGMC_PRIMITIVE_BLOCK_Y = 8;
}

RGYFilterParamRtgmcPrimitive::RGYFilterParamRtgmcPrimitive() :
    op(RGYRtgmcPrimitiveOp::Copy),
    refMode(RGYRtgmcPrimitiveRefMode::Disabled),
    mode(0),
    weight(0.5f),
    planes(0x07),
    processChroma(true) {
}

const TCHAR *RGYFilterRtgmcPrimitive::opToStr(RGYRtgmcPrimitiveOp op) {
    switch (op) {
    case RGYRtgmcPrimitiveOp::Copy:        return _T("copy");
    case RGYRtgmcPrimitiveOp::MakeDiff:    return _T("makediff");
    case RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20: return _T("makediff_removegrain20");
    case RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20AddDiff: return _T("makediff_removegrain20_adddiff");
    case RGYRtgmcPrimitiveOp::AddDiff:     return _T("adddiff");
    case RGYRtgmcPrimitiveOp::AddWeightedDiff: return _T("addweighteddiff");
    case RGYRtgmcPrimitiveOp::RemoveGrain: return _T("removegrain");
    case RGYRtgmcPrimitiveOp::Repair:      return _T("repair");
    case RGYRtgmcPrimitiveOp::Merge:       return _T("merge");
    case RGYRtgmcPrimitiveOp::GaussResize: return _T("gaussresize");
    case RGYRtgmcPrimitiveOp::VerticalMin5: return _T("verticalmin5");
    case RGYRtgmcPrimitiveOp::VerticalMax5: return _T("verticalmax5");
    case RGYRtgmcPrimitiveOp::LogicMin:    return _T("logicmin");
    case RGYRtgmcPrimitiveOp::LogicMax:    return _T("logicmax");
    default:                               return _T("unknown");
    }
}

const TCHAR *RGYFilterRtgmcPrimitive::refModeToStr(RGYRtgmcPrimitiveRefMode refMode) {
    switch (refMode) {
    case RGYRtgmcPrimitiveRefMode::Disabled:      return _T("none");
    case RGYRtgmcPrimitiveRefMode::RemoveGrain20: return _T("removegrain20");
    default:                                      return _T("unknown");
    }
}

bool RGYFilterRtgmcPrimitive::needsRef(RGYRtgmcPrimitiveOp op) {
    switch (op) {
    case RGYRtgmcPrimitiveOp::MakeDiff:
    case RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20:
    case RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20AddDiff:
    case RGYRtgmcPrimitiveOp::AddDiff:
    case RGYRtgmcPrimitiveOp::AddWeightedDiff:
    case RGYRtgmcPrimitiveOp::Repair:
    case RGYRtgmcPrimitiveOp::Merge:
    case RGYRtgmcPrimitiveOp::LogicMin:
    case RGYRtgmcPrimitiveOp::LogicMax:
        return true;
    default:
        return false;
    }
}

tstring RGYFilterParamRtgmcPrimitive::print() const {
    return strsprintf(_T("rtgmc-primitive: op=%s ref=%s mode=%d weight=%.3f planes=0x%x chroma=%s"),
        RGYFilterRtgmcPrimitive::opToStr(op), RGYFilterRtgmcPrimitive::refModeToStr(refMode), mode, weight, planes,
        processChroma ? _T("true") : _T("false"));
}

RGYFilterRtgmcPrimitive::RGYFilterRtgmcPrimitive(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_primitive(),
    m_buildOptions(),
    m_resizeGauss(),
    m_useKernel(false) {
    m_name = _T("rtgmc-primitive");
}

RGYFilterRtgmcPrimitive::~RGYFilterRtgmcPrimitive() {
    close();
}

RGY_ERR RGYFilterRtgmcPrimitive::checkParam(const std::shared_ptr<RGYFilterParamRtgmcPrimitive> &prm) {
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
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive requires identical input/output csp and resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->op < RGYRtgmcPrimitiveOp::Copy || prm->op > RGYRtgmcPrimitiveOp::LogicMax) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive: unsupported op.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->refMode < RGYRtgmcPrimitiveRefMode::Disabled || prm->refMode > RGYRtgmcPrimitiveRefMode::RemoveGrain20) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive: unsupported ref mode.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!needsRef(prm->op) && prm->refMode != RGYRtgmcPrimitiveRefMode::Disabled) {
        AddMessage(RGY_LOG_WARN, _T("rtgmc-primitive ref=%s is ignored for op=%s.\n"), refModeToStr(prm->refMode), opToStr(prm->op));
    }
    if (prm->op == RGYRtgmcPrimitiveOp::RemoveGrain) {
        if (!((prm->mode >= 1 && prm->mode <= 4) || prm->mode == 11 || prm->mode == 12 || prm->mode == 20)) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive removegrain mode supports 1-4, 11, 12 and 20.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
    } else if (prm->op == RGYRtgmcPrimitiveOp::Repair) {
        if (!((prm->mode >= 1 && prm->mode <= 4) || prm->mode == 12)) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive repair mode supports 1-4 and 12.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
    }
    if (prm->op == RGYRtgmcPrimitiveOp::Merge) {
        if (prm->weight < 0.0f || prm->weight > 1.0f) {
            AddMessage(RGY_LOG_WARN, _T("rtgmc-primitive merge weight should be 0.0-1.0; clamped.\n"));
            prm->weight = clamp(prm->weight, 0.0f, 1.0f);
        }
    } else if (prm->op == RGYRtgmcPrimitiveOp::AddWeightedDiff) {
        if (prm->weight < -1.0f || prm->weight > 1.0f) {
            AddMessage(RGY_LOG_WARN, _T("rtgmc-primitive addweighteddiff weight should be -1.0-1.0; clamped.\n"));
            prm->weight = clamp(prm->weight, -1.0f, 1.0f);
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcPrimitive::buildKernels(const std::shared_ptr<RGYFilterParamRtgmcPrimitive> &prm) {
    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelMax = (bitdepth >= 16) ? ((1 << 16) - 1) : ((1 << bitdepth) - 1);
    const int rangeHalf = 1 << (bitdepth - 1);
    m_buildOptions = strsprintf(
        "-D Type=%s -D bit_depth=%d -D max_val=%d -D range_half=%d -D rtgmc_primitive_block_x=%d -D rtgmc_primitive_block_y=%d",
        bitdepth > 8 ? "ushort" : "uchar",
        bitdepth,
        pixelMax,
        rangeHalf,
        RTGMC_PRIMITIVE_BLOCK_X,
        RTGMC_PRIMITIVE_BLOCK_Y);
    m_primitive.set(m_cl->buildResourceAsync(_T("RGY_FILTER_RTGMC_PRIMITIVE_CL"), _T("EXE_DATA"), m_buildOptions.c_str()));
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcPrimitive::setupGaussResize(const RGYFilterParamRtgmcPrimitive &prm) {
    if (prm.op != RGYRtgmcPrimitiveOp::GaussResize) {
        m_resizeGauss.reset();
        return RGY_ERR_NONE;
    }

    if (!m_resizeGauss) {
        m_resizeGauss = std::make_unique<RGYFilterResize>(m_cl);
    }
    auto resizePrm = std::make_shared<RGYFilterParamResize>();
    resizePrm->frameIn = prm.frameIn;
    resizePrm->frameOut = prm.frameOut;
    resizePrm->interp = RGY_VPP_RESIZE_GAUSS;
    resizePrm->gaussP = clamp((float)prm.mode, 0.1f, 100.0f);
    return m_resizeGauss->init(resizePrm, m_pLog);
}

RGY_ERR RGYFilterRtgmcPrimitive::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamRtgmcPrimitive>(pParam);
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

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamRtgmcPrimitive>(m_param);
    if (m_useKernel
        && (!m_primitive.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp])) {
        sts = buildKernels(prm);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build rtgmc-primitive kernel.\n"));
            return sts;
        }
    }

    const int frameBufCount = (needsRef(prm->op) && prm->refMode != RGYRtgmcPrimitiveRefMode::Disabled) ? 2 : 1;
    sts = AllocFrameBuf(prm->frameOut, frameBufCount);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    sts = setupGaussResize(*prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

bool RGYFilterRtgmcPrimitive::processPlane(int iplane, const RGYFilterParamRtgmcPrimitive &prm) const {
    return ((prm.planes & (1 << iplane)) != 0) && (iplane == 0 || prm.processChroma);
}

RGYFrameInfo *RGYFilterRtgmcPrimitive::generatedRefFrame() {
    return (m_frameBuf.size() >= 2) ? &m_frameBuf[1]->frame : nullptr;
}

RGY_ERR RGYFilterRtgmcPrimitive::processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame,
    const RGYFilterParamRtgmcPrimitive &prm,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];

    auto launchKernel = [&](RGYRtgmcPrimitiveOp op, const char *kernelName, int iplane, const std::vector<RGYOpenCLEvent> &wait, RGYOpenCLEvent *ev) {
        const auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(pInputFrame, (RGY_PLANE)iplane);
        const auto refPlane = pRefFrame ? getPlane(pRefFrame, (RGY_PLANE)iplane) : RGYFrameInfo();
        RGYWorkSize local(RTGMC_PRIMITIVE_BLOCK_X, RTGMC_PRIMITIVE_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        auto kernel = m_primitive.get()->kernel(kernelName).config(queue, local, global, wait, ev);
        RGY_ERR err = RGY_ERR_NONE;
        switch (op) {
        case RGYRtgmcPrimitiveOp::Copy:
            err = kernel.launch(
                (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
                (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height);
            break;
        case RGYRtgmcPrimitiveOp::AddWeightedDiff:
        case RGYRtgmcPrimitiveOp::Merge:
            err = kernel.launch(
                (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
                (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
                (cl_mem)refPlane.ptr[0], refPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                prm.weight);
            break;
        case RGYRtgmcPrimitiveOp::MakeDiff:
        case RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20:
        case RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20AddDiff:
        case RGYRtgmcPrimitiveOp::AddDiff:
        case RGYRtgmcPrimitiveOp::Repair:
        case RGYRtgmcPrimitiveOp::LogicMin:
        case RGYRtgmcPrimitiveOp::LogicMax:
            err = kernel.launch(
                (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
                (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
                (cl_mem)refPlane.ptr[0], refPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                prm.mode);
            break;
        case RGYRtgmcPrimitiveOp::RemoveGrain:
        case RGYRtgmcPrimitiveOp::VerticalMin5:
        case RGYRtgmcPrimitiveOp::VerticalMax5:
            err = kernel.launch(
                (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
                (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
                dstPlane.width, dstPlane.height,
                prm.mode);
            break;
        default:
            err = RGY_ERR_UNSUPPORTED;
            break;
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                char_to_tstring(kernelName).c_str(), iplane, get_err_mes(err));
        }
        return err;
    };

    std::vector<RGYOpenCLEvent> planeWaitEvents = wait_events;
    RGYOpenCLEvent planeEvent;
    for (int iplane = 0; iplane < planes; iplane++) {
        const bool doProcess = processPlane(iplane, prm);
        auto op = doProcess ? prm.op : RGYRtgmcPrimitiveOp::Copy;
        const char *kernelName = "kernel_rtgmc_primitive_copy";
        if (doProcess) {
            switch (op) {
            case RGYRtgmcPrimitiveOp::MakeDiff:    kernelName = "kernel_rtgmc_primitive_makediff"; break;
            case RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20: kernelName = "kernel_rtgmc_primitive_makediff_removegrain20"; break;
            case RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20AddDiff: kernelName = "kernel_rtgmc_primitive_makediff_removegrain20_adddiff"; break;
            case RGYRtgmcPrimitiveOp::AddDiff:     kernelName = "kernel_rtgmc_primitive_adddiff"; break;
            case RGYRtgmcPrimitiveOp::AddWeightedDiff: kernelName = "kernel_rtgmc_primitive_addweighteddiff"; break;
            case RGYRtgmcPrimitiveOp::RemoveGrain: kernelName = "kernel_rtgmc_primitive_removegrain"; break;
            case RGYRtgmcPrimitiveOp::Repair:      kernelName = "kernel_rtgmc_primitive_repair"; break;
            case RGYRtgmcPrimitiveOp::Merge:       kernelName = "kernel_rtgmc_primitive_merge"; break;
            case RGYRtgmcPrimitiveOp::VerticalMin5: kernelName = "kernel_rtgmc_primitive_vertical_min5"; break;
            case RGYRtgmcPrimitiveOp::VerticalMax5: kernelName = "kernel_rtgmc_primitive_vertical_max5"; break;
            case RGYRtgmcPrimitiveOp::LogicMin:    kernelName = "kernel_rtgmc_primitive_logicmin"; break;
            case RGYRtgmcPrimitiveOp::LogicMax:    kernelName = "kernel_rtgmc_primitive_logicmax"; break;
                default:                               kernelName = "kernel_rtgmc_primitive_copy"; break;
            }
        }
        auto err = launchKernel(op, kernelName, iplane, planeWaitEvents, (iplane == planes - 1) ? event : &planeEvent);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        planeWaitEvents.clear();
        if (planeEvent() != nullptr) {
            planeWaitEvents.push_back(planeEvent);
        }
    }
    copyFramePropWithoutRes(pOutputFrame, pInputFrame);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcPrimitive::processGaussResize(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const RGYFilterParamRtgmcPrimitive &prm,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    int processPlanes = 0;
    for (int iplane = 0; iplane < planes; iplane++) {
        processPlanes += processPlane(iplane, prm) ? 1 : 0;
    }
    if (processPlanes == 0) {
        auto copyPrm = prm;
        copyPrm.op = RGYRtgmcPrimitiveOp::Copy;
        return processFrame(pOutputFrame, pInputFrame, nullptr, copyPrm, queue, wait_events, event);
    }
    if (!m_resizeGauss) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive gaussresize is not initialized.\n"));
        return RGY_ERR_INVALID_CALL;
    }

    RGYFrameInfo inputFrame = *pInputFrame;
    RGYFrameInfo *resizeOutput = pOutputFrame;
    int resizeFrames = 0;
    RGYOpenCLEvent resizeEvent;
    auto err = m_resizeGauss->filter(&inputFrame, &resizeOutput, &resizeFrames, queue, wait_events, (processPlanes == planes) ? event : &resizeEvent);
    if (err == RGY_ERR_NONE && (resizeFrames != 1 || resizeOutput == nullptr)) {
        err = RGY_ERR_INVALID_CALL;
    }
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to run rtgmc-primitive gaussresize: %s.\n"), get_err_mes(err));
        return err;
    }
    if (processPlanes == planes) {
        copyFramePropWithoutRes(pOutputFrame, pInputFrame);
        return RGY_ERR_NONE;
    }

    std::vector<RGYOpenCLEvent> copyWaitEvents;
    if (resizeEvent() != nullptr) {
        copyWaitEvents.push_back(resizeEvent);
    }
    RGYOpenCLEvent planeEvent;
    const int copyPlanes = planes - processPlanes;
    int copiedPlanes = 0;
    for (int iplane = 0; iplane < planes; iplane++) {
        if (processPlane(iplane, prm)) {
            continue;
        }
        copiedPlanes++;
        const auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(pInputFrame, (RGY_PLANE)iplane);
        auto planeOutputEvent = (copiedPlanes == copyPlanes) ? event : &planeEvent;
        err = m_primitive.get()->kernel("kernel_rtgmc_primitive_copy")
            .config(queue, RGYWorkSize(RTGMC_PRIMITIVE_BLOCK_X, RTGMC_PRIMITIVE_BLOCK_Y), RGYWorkSize(dstPlane.width, dstPlane.height), copyWaitEvents, planeOutputEvent)
            .launch((cl_mem)dstPlane.ptr[0], dstPlane.pitch[0], (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0], dstPlane.width, dstPlane.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_rtgmc_primitive_copy (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
        copyWaitEvents.clear();
        if (planeEvent() != nullptr) {
            copyWaitEvents.push_back(planeEvent);
        }
    }
    copyFramePropWithoutRes(pOutputFrame, pInputFrame);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcPrimitive::run_filter(const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
    RGYOpenCLEvent *event) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;

    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    auto prm = std::dynamic_pointer_cast<RGYFilterParamRtgmcPrimitive>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const RGYFrameInfo *actualRefFrame = pRefFrame;
    RGYOpenCLEvent refEvent;
    std::vector<RGYOpenCLEvent> processWaitEvents = wait_events;
    if (needsRef(prm->op) && (!actualRefFrame || !actualRefFrame->ptr[0])) {
        if (prm->refMode == RGYRtgmcPrimitiveRefMode::RemoveGrain20) {
            auto refFrame = generatedRefFrame();
            if (!refFrame || !refFrame->ptr[0]) {
                AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive ref=%s has no frame buffer.\n"), refModeToStr(prm->refMode));
                return RGY_ERR_UNSUPPORTED;
            }
            auto refPrm = *prm;
            refPrm.op = RGYRtgmcPrimitiveOp::RemoveGrain;
            refPrm.mode = 20;
            refPrm.refMode = RGYRtgmcPrimitiveRefMode::Disabled;
            auto refErr = processFrame(refFrame, pInputFrame, nullptr, refPrm, queue, wait_events, &refEvent);
            if (refErr != RGY_ERR_NONE) {
                return refErr;
            }
            actualRefFrame = refFrame;
            processWaitEvents = { refEvent };
        } else {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive op=%s requires a reference frame.\n"), opToStr(prm->op));
            return RGY_ERR_UNSUPPORTED;
        }
    }
    if (needsRef(prm->op) && actualRefFrame && actualRefFrame->ptr[0]
        && (pInputFrame->csp != actualRefFrame->csp || pInputFrame->width != actualRefFrame->width || pInputFrame->height != actualRefFrame->height)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive reference frame must match input csp and resolution.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_useKernel && !m_primitive.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build/load RGY_FILTER_RTGMC_PRIMITIVE_CL (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }

    auto pOutFrame = m_frameBuf[0].get();
    ppOutputFrames[0] = &pOutFrame->frame;
    *pOutputFrameNum = 1;

    if (m_useKernel) {
        const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, m_frameBuf[0]->frame.mem_type);
        const auto refMemcpyKind = actualRefFrame ? getMemcpyKind(actualRefFrame->mem_type, m_frameBuf[0]->frame.mem_type) : RGYCLMemcpyD2D;
        if (memcpyKind == RGYCLMemcpyD2D && refMemcpyKind == RGYCLMemcpyD2D) {
            if (prm->op == RGYRtgmcPrimitiveOp::GaussResize) {
                return processGaussResize(&pOutFrame->frame, pInputFrame, *prm, queue, processWaitEvents, event);
            }
            return processFrame(&pOutFrame->frame, pInputFrame, actualRefFrame, *prm, queue, processWaitEvents, event);
        }
        if (prm->op != RGYRtgmcPrimitiveOp::Copy) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive op=%s requires device-to-device OpenCL frames.\n"), opToStr(prm->op));
            return RGY_ERR_UNSUPPORTED;
        }
    }

    auto copyErr = m_cl->copyFrame(ppOutputFrames[0], pInputFrame, nullptr, queue, wait_events, event, RGYFrameCopyMode::FRAME, "rtgmc_primitive.copy");
    if (copyErr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy frame: %s.\n"), get_err_mes(copyErr));
        return copyErr;
    }
    copyFramePropWithoutRes(ppOutputFrames[0], pInputFrame);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcPrimitive::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
    RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamRtgmcPrimitive>(m_param);
    if (prm && needsRef(prm->op) && prm->refMode == RGYRtgmcPrimitiveRefMode::Disabled) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-primitive op=%s requires a reference frame; call two-input run_filter().\n"), opToStr(prm->op));
        return RGY_ERR_UNSUPPORTED;
    }
    return run_filter(pInputFrame, nullptr, ppOutputFrames, pOutputFrameNum, queue, wait_events, event);
}

void RGYFilterRtgmcPrimitive::close() {
    m_primitive.clear();
    m_buildOptions.clear();
    m_resizeGauss.reset();
    m_frameBuf.clear();
    m_useKernel = false;
    m_cl.reset();
}
