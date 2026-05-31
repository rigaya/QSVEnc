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
#include <array>
#include "rgy_filter_msmooth.h"

RGYFilterMsmooth::RGYFilterMsmooth(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_bInterlacedWarn(false), m_msmooth(), m_blur(), m_mask(), m_tmp() {
    m_name = _T("msmooth");
}

RGYFilterMsmooth::~RGYFilterMsmooth() {
    close();
}

RGY_ERR RGYFilterMsmooth::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamMsmooth>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->msmooth.strength < 0 || 20 < prm->msmooth.strength) {
        prm->msmooth.strength = clamp(prm->msmooth.strength, 0, 20);
        AddMessage(RGY_LOG_WARN, _T("strength should be in range of %d - %d.\n"), 0, 20);
    }
    if (prm->msmooth.threshold < 0.0f || 255.0f < prm->msmooth.threshold) {
        prm->msmooth.threshold = clamp(prm->msmooth.threshold, 0.0f, 255.0f);
        AddMessage(RGY_LOG_WARN, _T("threshold should be in range of %.1f - %.1f.\n"), 0.0f, 255.0f);
    }

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamMsmooth>(m_param);
    if (!m_msmooth.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp]);
        m_msmooth.set(m_cl->buildResourceAsync(_T("RGY_FILTER_MSMOOTH_CL"), _T("EXE_DATA"), options.c_str()));
    }

    auto err = AllocFrameBuf(prm->frameOut, 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // 中間バッファの確保 (warpsharp のパターンに従う)
    if (!m_blur || cmpFrameInfoCspResolution(&m_blur->frame, &prm->frameOut)) {
        m_blur = m_cl->createFrameBuffer(prm->frameOut, CL_MEM_READ_WRITE);
    }
    if (!m_mask || cmpFrameInfoCspResolution(&m_mask->frame, &prm->frameOut)) {
        m_mask = m_cl->createFrameBuffer(prm->frameOut, CL_MEM_READ_WRITE);
    }
    for (auto& t : m_tmp) {
        if (!t || cmpFrameInfoCspResolution(&t->frame, &prm->frameOut)) {
            t = m_cl->createFrameBuffer(prm->frameOut, CL_MEM_READ_WRITE);
        }
    }

    //コピーを保存
    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterMsmooth::procPlaneBlur(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamMsmooth>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const char *kernel_name = "kernel_msmooth_blur";
    RGYWorkSize local(32, 8);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    auto err = m_msmooth.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0]);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlaneBlur(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterMsmooth::procPlaneEdgeMask(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pBlurPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamMsmooth>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const float threshold = prm->msmooth.threshold / (float)((1 << RGY_CSP_BIT_DEPTH[pBlurPlane->csp]) - 1);
    const char *kernel_name = "kernel_msmooth_edge_mask";
    RGYWorkSize local(32, 8);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    auto err = m_msmooth.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (cl_mem)pBlurPlane->ptr[0], pBlurPlane->pitch[0],
        threshold, prm->msmooth.highq ? 1 : 0);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlaneEdgeMask(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pBlurPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterMsmooth::procPlaneSmooth(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const RGYFrameInfo *pMaskPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *kernel_name = "kernel_msmooth_smooth";
    RGYWorkSize local(32, 8);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    auto err = m_msmooth.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
        (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
        (cl_mem)pMaskPlane->ptr[0], pMaskPlane->pitch[0],
        pOutputPlane->width, pOutputPlane->height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlaneSmooth(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterMsmooth::procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamMsmooth>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto blurPlane = getPlane(&m_blur->frame, RGY_PLANE_Y);
    auto maskPlane = getPlane(&m_mask->frame, RGY_PLANE_Y);
    auto tmpPlane0 = getPlane(&m_tmp[0]->frame, RGY_PLANE_Y);
    auto tmpPlane1 = getPlane(&m_tmp[1]->frame, RGY_PLANE_Y);

    // Step 1: Blur
    auto err = procPlaneBlur(&blurPlane, pInputPlane, queue, wait_events, nullptr);
    if (err != RGY_ERR_NONE) return err;

    // mask mode: edge mask をそのまま出力
    if (prm->msmooth.mask) {
        err = procPlaneEdgeMask(pOutputPlane, &blurPlane, queue, {}, event);
        return err;
    }

    // Step 2: Edge Mask
    err = procPlaneEdgeMask(&maskPlane, &blurPlane, queue, {}, nullptr);
    if (err != RGY_ERR_NONE) return err;

    // Step 3: Iterative Smoothing
    const int iterations = prm->msmooth.strength;
    if (iterations <= 0) {
        // strength=0: just copy input to output
        return m_cl->copyPlane(pOutputPlane, pInputPlane, nullptr, queue, {}, event);
    }

    for (int i = 0; i < iterations; i++) {
        const RGYFrameInfo *srcPlane;
        RGYFrameInfo *dstPlane;
        bool isLast = (i == iterations - 1);
        if (i == 0) {
            srcPlane = pInputPlane;
            dstPlane = isLast ? pOutputPlane : &tmpPlane0;
        } else if (i % 2 == 1) {
            srcPlane = &tmpPlane0;
            dstPlane = isLast ? pOutputPlane : &tmpPlane1;
        } else {
            srcPlane = &tmpPlane1;
            dstPlane = isLast ? pOutputPlane : &tmpPlane0;
        }
        RGYOpenCLEvent *iter_event = isLast ? event : nullptr;
        err = procPlaneSmooth(dstPlane, srcPlane, &maskPlane, queue, {}, iter_event);
        if (err != RGY_ERR_NONE) return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterMsmooth::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[pOutputFrame->csp] - 1) ? event : nullptr;
        auto err = procPlane(&planeDst, &planeSrc, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to msmooth frame(%d): %s\n"), i, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterMsmooth::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
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
    if (!m_msmooth.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build RGY_FILTER_MSMOOTH_CL(m_msmooth)\n"));
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

void RGYFilterMsmooth::close() {
    m_blur.reset();
    m_mask.reset();
    m_tmp[0].reset();
    m_tmp[1].reset();
    m_frameBuf.clear();
    m_msmooth.clear();
    m_cl.reset();
    m_bInterlacedWarn = false;
}
