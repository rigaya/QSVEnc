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
#include "rgy_filter_denoise_knn.h"

static const int KNN_RADIUS_MAX = 5;

RGY_ERR RGYFilterDenoiseKnn::denoisePlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseKnn>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    {
        const float strength = 1.0f / (prm->knn.strength * prm->knn.strength);
        const char *kernel_name = "kernel_denoise_knn";
        RGYWorkSize local(32, 8);
        RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
        auto err = m_knn.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
            (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
            (cl_mem)pInputPlane->ptr[0],
            strength, prm->knn.lerpC, prm->knn.weight_threshold, prm->knn.lerp_threshold);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseKnn::denoiseFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto err = m_cl->createImageFromFrameBuffer(m_srcImage, *pInputFrame, true, CL_MEM_READ_ONLY);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to create image from buffer: %s.\n"), get_err_mes(err));
        return err;
    }

    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(&m_srcImage->frame, (RGY_PLANE)i);
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[pOutputFrame->csp] - 1) ? event : nullptr;
        err = denoisePlane(&planeDst, &planeSrc, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to denoise(knn) frame(%d) %s: %s\n"), i, cl_errmes(err));
            return err_cl_to_rgy(err);
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterDenoiseKnn::RGYFilterDenoiseKnn(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_knn(), m_srcImage() {
    m_name = _T("knn");
}

RGYFilterDenoiseKnn::~RGYFilterDenoiseKnn() {
    close();
}

RGY_ERR RGYFilterDenoiseKnn::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto pKnnParam = std::dynamic_pointer_cast<RGYFilterParamDenoiseKnn>(pParam);
    if (!pKnnParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (pKnnParam->frameOut.height <= 0 || pKnnParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.radius <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("radius must be a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.radius > KNN_RADIUS_MAX) {
        AddMessage(RGY_LOG_ERROR, _T("radius must be <= %d.\n"), KNN_RADIUS_MAX);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.strength < 0.0 || 1.0 < pKnnParam->knn.strength) {
        AddMessage(RGY_LOG_ERROR, _T("strength should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.lerpC < 0.0 || 1.0 < pKnnParam->knn.lerpC) {
        AddMessage(RGY_LOG_ERROR, _T("lerpC should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.lerp_threshold < 0.0 || 1.0 < pKnnParam->knn.lerp_threshold) {
        AddMessage(RGY_LOG_ERROR, _T("th_lerp should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.weight_threshold < 0.0 || 1.0 < pKnnParam->knn.weight_threshold) {
        AddMessage(RGY_LOG_ERROR, _T("th_weight should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamDenoiseKnn>(m_param);
    if (!m_knn.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
        || prmPrev->knn.radius != pKnnParam->knn.radius) {
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d -D knn_radius=%d",
            RGY_CSP_BIT_DEPTH[pKnnParam->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[pKnnParam->frameOut.csp],
            pKnnParam->knn.radius);
        m_knn.set(m_cl->buildResourceAsync(_T("RGY_FILTER_DENOISE_KNN_CL"), _T("EXE_DATA"), options.c_str()));
    }

    auto err = AllocFrameBuf(pKnnParam->frameOut, 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        pKnnParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    //コピーを保存
    setFilterInfo(pKnnParam->print());
    m_param = pKnnParam;
    return sts;
}

RGY_ERR RGYFilterDenoiseKnn::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
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
    if (!m_knn.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_DENOISE_KNN_CL(m_knn)\n"));
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

    sts = denoiseFrame(ppOutputFrames[0], pInputFrame, queue, wait_events, event);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at denoiseFrame (%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }

    return sts;
}

void RGYFilterDenoiseKnn::close() {
    m_srcImage.reset();
    m_frameBuf.clear();
    m_knn.clear();
    m_cl.reset();
    m_bInterlacedWarn = false;
}
