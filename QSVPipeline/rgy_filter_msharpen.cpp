// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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

#define _USE_MATH_DEFINES
#include <cmath>
#include <map>
#include <array>
#include "rgy_filter_msharpen.h"

RGYFilterMsharpen::RGYFilterMsharpen(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_bInterlacedWarn(false), m_msharpen() {
    m_name = _T("msharpen");
}

RGYFilterMsharpen::~RGYFilterMsharpen() {
    close();
}

RGY_ERR RGYFilterMsharpen::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamMsharpen>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->msharpen.strength < 0.0f || 1.0f < prm->msharpen.strength) {
        prm->msharpen.strength = clamp(prm->msharpen.strength, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("strength should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }
    if (prm->msharpen.threshold < 0.0f || 255.0f < prm->msharpen.threshold) {
        prm->msharpen.threshold = clamp(prm->msharpen.threshold, 0.0f, 255.0f);
        AddMessage(RGY_LOG_WARN, _T("threshold should be in range of %.1f - %.1f.\n"), 0.0f, 255.0f);
    }
    // slope: 0 (default) keeps the original binary edge gate. Positive values
    // enable the sigmoid soft mask. Negative values are invalid and clamped.
    if (prm->msharpen.slope < 0.0f) {
        prm->msharpen.slope = 0.0f;
        AddMessage(RGY_LOG_WARN, _T("slope must be >= 0; clamped to 0 (binary gate).\n"));
    }
    // luma_limit: 0 (default) disables luma-adaptive scaling. Otherwise the
    // useful range is (0, 255]; outside that interval we clamp and warn.
    if (prm->msharpen.luma_limit < 0.0f || 255.0f < prm->msharpen.luma_limit) {
        prm->msharpen.luma_limit = clamp(prm->msharpen.luma_limit, 0.0f, 255.0f);
        AddMessage(RGY_LOG_WARN, _T("luma_limit should be in range of %.1f - %.1f (0 disables).\n"), 0.0f, 255.0f);
    }
    if (prm->msharpen.block_protect < 0.0f || 1.0f < prm->msharpen.block_protect) {
        prm->msharpen.block_protect = clamp(prm->msharpen.block_protect, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("block_protect should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamMsharpen>(m_param);
    if (!m_msharpen.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp]);
        m_msharpen.set(m_cl->buildResourceAsync(_T("RGY_FILTER_MSHARPEN_CL"), _T("EXE_DATA"), options.c_str()));
    }

    auto err = AllocFrameBuf(prm->frameOut, 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    //コピーを保存
    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterMsharpen::procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGY_PLANE plane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamMsharpen>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int bitDepth = RGY_CSP_BIT_DEPTH[pInputPlane->csp];
    if (bitDepth <= 0 || 16 < bitDepth) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp/bit depth: %s.\n"), RGY_CSP_NAMES[pInputPlane->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    const float threshold = prm->msharpen.threshold / (float)((1 << bitDepth) - 1);
    // The user-facing slope and luma_limit are documented in 8-bit terms.
    // Inside the kernel, threshold / gradient / luma are all in [0, 1]
    // normalised space. So we scale by 255 (NOT pixel_max), keeping the
    // sigmoid steepness consistent across 8-bit and HBD inputs.
    const float slope_norm = prm->msharpen.slope * 255.0f;
    const float luma_limit_norm = (plane == RGY_PLANE_Y && prm->msharpen.luma_limit > 0.0f)
        ? prm->msharpen.luma_limit / 255.0f
        : 0.0f;
    const char *kernel_name = "kernel_msharpen";
    RGYWorkSize local(32, 8);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    auto err = m_msharpen.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
        prm->msharpen.strength, threshold,
        slope_norm, luma_limit_norm,
        prm->msharpen.highq ? 1 : 0, prm->msharpen.mask ? 1 : 0,
        prm->msharpen.block_protect);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterMsharpen::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const int planeCount = RGY_CSP_PLANES[pOutputFrame->csp];
    for (int i = 0; i < planeCount; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame,  (RGY_PLANE)i);
        const auto &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == planeCount - 1) ? event : nullptr;
        auto err = procPlane(&planeDst, &planeSrc, (RGY_PLANE)i, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterMsharpen::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (!m_msharpen.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build RGY_FILTER_MSHARPEN_CL(m_msharpen)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != RGYCLMemcpyD2D) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    sts = procFrame(ppOutputFrames[0], pInputFrame, queue, wait_events, event);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at procFrame (%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }

    return sts;
}

void RGYFilterMsharpen::close() {
    m_frameBuf.clear();
    m_msharpen.clear();
    m_cl.reset();
    m_bInterlacedWarn = false;
}
