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

#include "convert_csp.h"
#include "rgy_filter_rtgmc_bob.h"

static const int RTGMC_BOB_BLOCK_X = 32;
static const int RTGMC_BOB_BLOCK_Y = 8;

tstring RGYFilterParamRtgmcBob::print() const {
    const TCHAR *orderStr = _T("auto");
    if (order == RGYRtgmcBobFieldOrder::TFF) {
        orderStr = _T("tff");
    } else if (order == RGYRtgmcBobFieldOrder::BFF) {
        orderStr = _T("bff");
    }
    return strsprintf(_T("rtgmc-bob: order=%s"), orderStr);
}

RGYFilterRtgmcBob::RGYFilterRtgmcBob(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_rtgmcBob(),
    m_buildOptions(),
    m_defaultTff(true) {
    m_name = _T("rtgmc-bob");
}

RGYFilterRtgmcBob::~RGYFilterRtgmcBob() {
    close();
}

RGY_ERR RGYFilterRtgmcBob::checkParam(const std::shared_ptr<RGYFilterParamRtgmcBob> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.csp == RGY_CSP_NA || RGY_CSP_PLANES[prm->frameOut.csp] <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid colorspace.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcBob::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamRtgmcBob>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    m_pathThrough &= ~(FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_TIMESTAMP);
    pParam->baseFps *= 2;

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamRtgmcBob>(m_param);
    if (!prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
        const int maxVal = (1 << bitDepth) - 1;
        m_buildOptions = strsprintf("-D Type=%s -D bit_depth=%d -D max_val=%d -D rtgmc_bob_block_x=%d -D rtgmc_bob_block_y=%d",
            bitDepth > 8 ? "ushort" : "uchar",
            bitDepth, maxVal,
            RTGMC_BOB_BLOCK_X, RTGMC_BOB_BLOCK_Y);
        AddMessage(RGY_LOG_DEBUG, _T("Starting async build for RGY_FILTER_RTGMC_BOB_CL: %s\n"),
            char_to_tstring(m_buildOptions).c_str());
        m_rtgmcBob.set(m_cl->buildResourceAsync(_T("RGY_FILTER_RTGMC_BOB_CL"), _T("EXE_DATA"), m_buildOptions.c_str()));
    }

    sts = AllocFrameBuf(prm->frameOut, 2);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    m_defaultTff = (prm->frameIn.picstruct & RGY_PICSTRUCT_BFF) == 0;
    setFilterInfo(prm->print() + _T("\n                         auto-order-fallback=") + (m_defaultTff ? _T("tff") : _T("bff")));
    m_param = prm;
    return RGY_ERR_NONE;
}

bool RGYFilterRtgmcBob::getInputTff(const RGYFrameInfo *frame) const {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamRtgmcBob>(m_param);
    if (!prm) {
        return m_defaultTff;
    }
    if (prm->order == RGYRtgmcBobFieldOrder::TFF) {
        return true;
    }
    if (prm->order == RGYRtgmcBobFieldOrder::BFF) {
        return false;
    }
    if (frame) {
        if (frame->picstruct & RGY_PICSTRUCT_BFF) {
            return false;
        }
        if (frame->picstruct & RGY_PICSTRUCT_TFF) {
            return true;
        }
    }
    return m_defaultTff;
}

RGY_ERR RGYFilterRtgmcBob::processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const int preservedParity,
    const int phaseQuarter,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *kernelName = "kernel_rtgmc_bob";
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto srcPlane = getPlane(pInputFrame, (RGY_PLANE)iplane);

        RGYWorkSize local(RTGMC_BOB_BLOCK_X, RTGMC_BOB_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        const auto &waitHere = (iplane == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        auto err = m_rtgmcBob.get()->kernel(kernelName).config(queue, local, global, waitHere, (iplane == planes - 1) ? event : nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
            (cl_mem)srcPlane.ptr[0], srcPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            preservedParity,
            phaseQuarter);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                char_to_tstring(kernelName).c_str(), iplane, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

void RGYFilterRtgmcBob::setBobTimestamp(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamRtgmcBob>(m_param);
    auto frameDuration = pInputFrame->duration;
    if (frameDuration == 0 && prm && prm->timebase.is_valid()) {
        frameDuration = (decltype(frameDuration))((prm->timebase.inv() / prm->baseFps * 2).qdouble() + 0.5);
    }
    ppOutputFrames[0]->timestamp = pInputFrame->timestamp;
    ppOutputFrames[0]->duration = (frameDuration + 1) / 2;
    ppOutputFrames[1]->timestamp = ppOutputFrames[0]->timestamp + ppOutputFrames[0]->duration;
    ppOutputFrames[1]->duration = frameDuration - ppOutputFrames[0]->duration;
    ppOutputFrames[0]->inputFrameId = pInputFrame->inputFrameId;
    ppOutputFrames[1]->inputFrameId = pInputFrame->inputFrameId;
}

RGY_ERR RGYFilterRtgmcBob::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
    RGYOpenCLEvent *event) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    ppOutputFrames[1] = nullptr;

    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (!m_rtgmcBob.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build/load RGY_FILTER_RTGMC_BOB_CL (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }

    const bool inputTff = getInputTff(pInputFrame);
    const int firstFieldParity = inputTff ? 0 : 1;
    const int secondFieldParity = inputTff ? 1 : 0;
    const int firstFieldPhaseQuarter = (firstFieldParity == 0) ? +1 : -1;
    const int secondFieldPhaseQuarter = (secondFieldParity == 0) ? +1 : -1;

    auto err = processFrame(&m_frameBuf[0]->frame, pInputFrame, firstFieldParity, firstFieldPhaseQuarter, queue, wait_events, nullptr);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = processFrame(&m_frameBuf[1]->frame, pInputFrame, secondFieldParity, secondFieldPhaseQuarter, queue, {}, event);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    for (int i = 0; i < 2; i++) {
        auto pOut = &m_frameBuf[i]->frame;
        pOut->picstruct = RGY_PICSTRUCT_FRAME;
        pOut->flags = RGY_FRAME_FLAG_NONE;
        ppOutputFrames[i] = pOut;
    }
    *pOutputFrameNum = 2;
    setBobTimestamp(pInputFrame, ppOutputFrames);
    return RGY_ERR_NONE;
}

void RGYFilterRtgmcBob::close() {
    m_rtgmcBob.clear();
    m_buildOptions.clear();
    m_frameBuf.clear();
    m_cl.reset();
}
