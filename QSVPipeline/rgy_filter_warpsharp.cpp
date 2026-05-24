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
#include "rgy_filter_warpsharp.h"

static const int WARPSHARP_BLOCK_X = 32;
static const int WARPSHARP_BLOCK_Y = 8;

RGY_ERR RGYFilterWarpsharp::procPlaneSobel(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const float threshold, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *kernel_name = "kernel_warpsharp_sobel";
    RGYWorkSize local(WARPSHARP_BLOCK_X, WARPSHARP_BLOCK_Y);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    auto err = m_warpsharp.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
        (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
        pOutputPlane->width, pOutputPlane->height,
        (int)(threshold * (1 << (RGY_CSP_BIT_DEPTH[pOutputPlane->csp] - 8)) + 0.5f));
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlaneSobel(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pOutputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}
RGY_ERR RGYFilterWarpsharp::procPlaneBlur(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *kernel_name = "kernel_warpsharp_blur";
    RGYWorkSize local(WARPSHARP_BLOCK_X, WARPSHARP_BLOCK_Y);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    auto err = m_warpsharp.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
        (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
        pOutputPlane->width, pOutputPlane->height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlaneBlur(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pOutputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}
RGY_ERR RGYFilterWarpsharp::procPlaneWarp(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputMask, const RGYFrameInfo *pInputPlaneImg, const float depth_min, const float depth_max, const float edge_thr, const float gamma_val, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    // edge_thr is user-facing 8-bit; the kernel works in [0, 1] normalised
    // mask values so we feed it the normalised threshold.
    const float edge_thr_norm = edge_thr / 255.0f;
    const char *kernel_name = "kernel_warpsharp_warp";
    RGYWorkSize local(WARPSHARP_BLOCK_X, WARPSHARP_BLOCK_Y);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    auto err = m_warpsharp.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
        (cl_mem)pInputPlaneImg->ptr[0],
        (cl_mem)pInputMask->ptr[0], pInputMask->pitch[0],
        pOutputPlane->width, pOutputPlane->height,
        depth_min, depth_max, edge_thr_norm, gamma_val);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlaneWarp(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pOutputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterWarpsharp::procPlaneDowscale(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *kernel_name = "kernel_warpsharp_downscale";
    RGYWorkSize local(WARPSHARP_BLOCK_X, WARPSHARP_BLOCK_Y);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    auto err = m_warpsharp.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0]);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlaneDowscale(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterWarpsharp::procPlane(RGYFrameInfo *pOutputPlane, RGYFrameInfo *pMaskPlane0, RGYFrameInfo *pMaskPlane1, const RGYFrameInfo *pInputPlane, const RGYFrameInfo *pInputPlaneImg,
    const float threshold, const float depth_min, const float depth_max, const float edge_thr, const float gamma_val,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamWarpsharp>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    {
        const int blur = prm->warpsharp.blur;
        auto err = procPlaneSobel(pMaskPlane0, pInputPlane, threshold, queue, wait_events, nullptr);
        if (err != RGY_ERR_NONE) {
            return err;
        }

        for (int i = 0; i < blur; i++) {
            err = procPlaneBlur(pMaskPlane1, pMaskPlane0, queue, {}, nullptr);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            std::swap(pMaskPlane1, pMaskPlane0);
        }
        err = procPlaneWarp(pOutputPlane, pMaskPlane0, pInputPlaneImg, depth_min, depth_max, edge_thr, gamma_val, queue, {}, event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterWarpsharp::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamWarpsharp>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const float threshold = prm->warpsharp.threshold;
    const float depth = prm->warpsharp.depth;
    // Resolve the depth_min / depth_max sentinels (1e9f) to the global depth
    // so the back-compat path lands at depth_min == depth_max == depth, which
    // collapses the kernel's per-pixel mix() to a constant.
    const float depth_min = (prm->warpsharp.depth_min == FILTER_DEFAULT_WARPSHARP_DEPTH_MIN) ? depth : prm->warpsharp.depth_min;
    const float depth_max = (prm->warpsharp.depth_max == FILTER_DEFAULT_WARPSHARP_DEPTH_MAX) ? depth : prm->warpsharp.depth_max;
    const float edge_thr  = prm->warpsharp.edge_thr;
    const float gamma_val = prm->warpsharp.gamma;
    auto srcImage = m_cl->createImageFromFrameBuffer(*pInputFrame, true, CL_MEM_READ_ONLY, &m_srcImagePool);
    if (!srcImage) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to create image for input frame.\n"));
        return RGY_ERR_MEM_OBJECT_ALLOCATION_FAILURE;
    }

    const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
    const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
    const auto planeInputImgY = getPlane(&srcImage->frame, RGY_PLANE_Y);
    const auto planeInputImgU = getPlane(&srcImage->frame, RGY_PLANE_U);
    const auto planeInputImgV = getPlane(&srcImage->frame, RGY_PLANE_V);
    auto planeMask0Y = getPlane(&m_mask[0]->frame, RGY_PLANE_Y);
    auto planeMask0U = getPlane(&m_mask[0]->frame, RGY_PLANE_U);
    auto planeMask0V = getPlane(&m_mask[0]->frame, RGY_PLANE_V);
    auto planeMask1Y = getPlane(&m_mask[1]->frame, RGY_PLANE_Y);
    auto planeMask1U = getPlane(&m_mask[1]->frame, RGY_PLANE_U);
    auto planeMask1V = getPlane(&m_mask[1]->frame, RGY_PLANE_V);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
    auto err = procPlane(&planeOutputY, &planeMask0Y, &planeMask1Y, &planeInputY, &planeInputImgY, threshold, depth_min, depth_max, edge_thr, gamma_val, queue, wait_events, nullptr);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    // Chroma plane scaling for YUV420: halve the warp magnitude (same factor
    // as the original code). edge_thr and gamma are unitless ratios so they
    // pass through unchanged.
    const float chroma_scale = (RGY_CSP_CHROMA_FORMAT[pOutputFrame->csp] == RGY_CHROMAFMT_YUV420) ? 0.5f : 1.0f;
    const float depth_min_uv = depth_min * chroma_scale;
    const float depth_max_uv = depth_max * chroma_scale;
    if (prm->warpsharp.chroma == 0) {
        RGYFrameInfo *pMaskUV = &planeMask0Y;
        if (RGY_CSP_CHROMA_FORMAT[pOutputFrame->csp] == RGY_CHROMAFMT_YUV420) {
            err = procPlaneDowscale(&planeMask0U, &planeMask0Y, queue, {}, nullptr);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            pMaskUV = &planeMask0U;
        }
        err = procPlaneWarp(&planeOutputU, pMaskUV, &planeInputImgU, depth_min_uv, depth_max_uv, edge_thr, gamma_val, queue, {}, nullptr);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        err = procPlaneWarp(&planeOutputV, pMaskUV, &planeInputImgV, depth_min_uv, depth_max_uv, edge_thr, gamma_val, queue, {}, event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    } else {
        err = procPlane(&planeOutputU, &planeMask0U, &planeMask1U, &planeInputU, &planeInputImgU, threshold, depth_min_uv, depth_max_uv, edge_thr, gamma_val, queue, {}, nullptr);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        err = procPlane(&planeOutputV, &planeMask0V, &planeMask1V, &planeInputV, &planeInputImgV, threshold, depth_min_uv, depth_max_uv, edge_thr, gamma_val, queue, {}, event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterWarpsharp::RGYFilterWarpsharp(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_warpsharp(), m_mask(), m_srcImagePool() {
    m_name = _T("warpsharp");
}

RGYFilterWarpsharp::~RGYFilterWarpsharp() {
    close();
}

RGY_ERR RGYFilterWarpsharp::checkParam(const std::shared_ptr<RGYFilterParamWarpsharp> prm) {
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->warpsharp.threshold < 0.0f || 255.0f < prm->warpsharp.threshold) {
        prm->warpsharp.threshold = clamp(prm->warpsharp.threshold, 0.0f, 255.0f);
        AddMessage(RGY_LOG_WARN, _T("threshold should be in range of %.1f - %.1f.\n"), 0.0f, 255.0f);
    }
    if (prm->warpsharp.blur < 0) {
        prm->warpsharp.blur = std::max(prm->warpsharp.blur, 0);
        AddMessage(RGY_LOG_WARN, _T("blur should be a positive value.\n"));
    }
    if (prm->warpsharp.type < 0 || 1 < prm->warpsharp.type) {
        prm->warpsharp.type = clamp(prm->warpsharp.type, 0, 1);
        AddMessage(RGY_LOG_WARN, _T("type should be in range of %d - %d.\n"), 0, 1);
    }
    if (prm->warpsharp.depth < -128.0f || 128.0f < prm->warpsharp.depth) {
        prm->warpsharp.depth = clamp(prm->warpsharp.depth, -128.0f, 128.0f);
        AddMessage(RGY_LOG_WARN, _T("depth should be in range of %.1f - %.1f.\n"), -128.0f, 128.0f);
    }
    if (prm->warpsharp.chroma < 0 || 1 < prm->warpsharp.chroma) {
        prm->warpsharp.chroma = clamp(prm->warpsharp.chroma, 0, 1);
        AddMessage(RGY_LOG_WARN, _T("chroma should be in range of %d - %d.\n"), 0, 1);
    }
    // Adaptive-depth params. depth_min / depth_max use 1e9f as the
    // "follow depth" sentinel; any other negative value out of [-128, 128]
    // is clamped to the same range the global depth supports.
    if (prm->warpsharp.depth_min != FILTER_DEFAULT_WARPSHARP_DEPTH_MIN
        && (prm->warpsharp.depth_min < -128.0f || 128.0f < prm->warpsharp.depth_min)) {
        prm->warpsharp.depth_min = clamp(prm->warpsharp.depth_min, -128.0f, 128.0f);
        AddMessage(RGY_LOG_WARN, _T("depth_min should be in range of %.1f - %.1f.\n"), -128.0f, 128.0f);
    }
    if (prm->warpsharp.depth_max != FILTER_DEFAULT_WARPSHARP_DEPTH_MAX
        && (prm->warpsharp.depth_max < -128.0f || 128.0f < prm->warpsharp.depth_max)) {
        prm->warpsharp.depth_max = clamp(prm->warpsharp.depth_max, -128.0f, 128.0f);
        AddMessage(RGY_LOG_WARN, _T("depth_max should be in range of %.1f - %.1f.\n"), -128.0f, 128.0f);
    }
    if (prm->warpsharp.edge_thr <= 0.0f || 255.0f < prm->warpsharp.edge_thr) {
        prm->warpsharp.edge_thr = clamp(prm->warpsharp.edge_thr, 1.0f, 255.0f);
        AddMessage(RGY_LOG_WARN, _T("edge_thr should be in range of (0.0, 255.0]; clamped.\n"));
    }
    if (prm->warpsharp.gamma <= 0.0f || 8.0f < prm->warpsharp.gamma) {
        prm->warpsharp.gamma = clamp(prm->warpsharp.gamma, 0.01f, 8.0f);
        AddMessage(RGY_LOG_WARN, _T("gamma should be in range of (0.0, 8.0]; clamped.\n"));
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterWarpsharp::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamWarpsharp>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if ((sts = checkParam(prm)) != RGY_ERR_NONE) {
        return sts;
    }
    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamWarpsharp>(m_param);
    if (!m_warpsharp.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
        || prmPrev->warpsharp.type != prm->warpsharp.type) {
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d -D blur_range=%d"
            " -D WARPSHARP_BLOCK_X=%d -D WARPSHARP_BLOCK_Y=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp],
            prm->warpsharp.type == 0 ? 6 : 2,
            WARPSHARP_BLOCK_X, WARPSHARP_BLOCK_Y);
        m_warpsharp.set(m_cl->buildResourceAsync(_T("RGY_FILTER_WARPSHARP_CL"), _T("EXE_DATA"), options.c_str()));
    }

    auto err = AllocFrameBuf(prm->frameOut, 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
     if (!m_mask[0] || cmpFrameInfoCspResolution(&m_mask[0]->frame, &prm->frameOut)) {
        for (auto& m : m_mask) {
            m = m_cl->createFrameBuffer(prm->frameOut, CL_MEM_READ_WRITE);
        }
    }

    //コピーを保存
    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterWarpsharp::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
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
    //if (interlaced(*pInputFrame)) {
    //    return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
    //}
    if (!m_warpsharp.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_WARPSHARP_CL(m_warpsharp)\n"));
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

void RGYFilterWarpsharp::close() {
    m_srcImagePool.clear();
    m_frameBuf.clear();
    m_warpsharp.clear();
    m_cl.reset();
    m_bInterlacedWarn = false;
}
