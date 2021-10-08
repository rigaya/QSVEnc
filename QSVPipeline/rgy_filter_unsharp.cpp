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
#include "rgy_filter_unsharp.h"

static const int UNSHARP_RADIUS_MAX = 9;

RGY_ERR RGYFilterUnsharp::procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const RGYCLBuf *gaussWeightBuf, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamUnsharp>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    {
        const char *kernel_name = "kernel_unsharp";
        RGYWorkSize local(32, 8);
        RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
        auto err = m_unsharp.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
            (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
            (cl_mem)pInputPlane->ptr[0],
            gaussWeightBuf->mem(),
            prm->unsharp.weight, prm->unsharp.threshold / (1 << RGY_CSP_BIT_DEPTH[pOutputPlane->csp]));
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterUnsharp::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto srcImage = m_cl->createImageFromFrameBuffer(*pInputFrame, true, CL_MEM_READ_ONLY);
    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(&srcImage->frame, (RGY_PLANE)i);
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[pOutputFrame->csp] - 1) ? event : nullptr;
        auto err = procPlane(&planeDst, &planeSrc, (((RGY_PLANE)i) == RGY_PLANE_Y) ? m_pGaussWeightBufY.get() : m_pGaussWeightBufUV.get(), queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to denoise(unsharp) frame(%d) %s: %s\n"), i, cl_errmes(err));
            return err_cl_to_rgy(err);
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterUnsharp::RGYFilterUnsharp(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_unsharp() {
    m_name = _T("unsharp");
}

RGYFilterUnsharp::~RGYFilterUnsharp() {
    close();
}

RGY_ERR RGYFilterUnsharp::setWeight(std::unique_ptr<RGYCLBuf>& pGaussWeightBuf, int radius, float sigma) {
    const int nWeightCount = (2 * radius + 1) * (2 * radius + 1);
    const int nBufferSize = sizeof(float) * nWeightCount;
    vector<float> weight(nWeightCount);
    float *ptr_weight = weight.data();
    double sum = 0.0;
    for (int j = -radius; j <= radius; j++) {
        for (int i = -radius; i <= radius; i++) {
            const double w = 1.0f / (2.0f * (float)M_PI * sigma * sigma) * std::exp(-1.0f * (i * i + j * j) / (2.0f * sigma * sigma));
            *ptr_weight = (float)w;
            sum += (double)w;
            ptr_weight++;
        }
    }
    ptr_weight = weight.data();
    const float inv_sum = (float)(1.0 / sum);
    for (int j = -radius; j <= radius; j++) {
        for (int i = -radius; i <= radius; i++) {
            *ptr_weight *= inv_sum;
            ptr_weight++;
        }
    }

    pGaussWeightBuf = m_cl->copyDataToBuffer(weight.data(), nBufferSize, CL_MEM_READ_ONLY, m_cl->queue().get());
    if (!pGaussWeightBuf) {
        return RGY_ERR_MEMORY_ALLOC;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterUnsharp::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamUnsharp>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->unsharp.radius < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (radius).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->unsharp.radius < 1 && prm->unsharp.radius > UNSHARP_RADIUS_MAX) {
        AddMessage(RGY_LOG_WARN, _T("radius must be in range of 1-%d.\n"), UNSHARP_RADIUS_MAX);
        prm->unsharp.radius = clamp(prm->unsharp.radius, 1, UNSHARP_RADIUS_MAX);
    }
    if (prm->unsharp.weight < 0.0f || 10.0f < prm->unsharp.weight) {
        prm->unsharp.weight = clamp(prm->unsharp.weight, 0.0f, 10.0f);
        AddMessage(RGY_LOG_WARN, _T("weight should be in range of %.1f - %.1f.\n"), 0.0f, 10.0f);
    }
    if (prm->unsharp.threshold < 0.0f || 255.0f < prm->unsharp.threshold) {
        prm->unsharp.threshold = clamp(prm->unsharp.threshold, 0.0f, 255.0f);
        AddMessage(RGY_LOG_WARN, _T("threshold should be in range of %.1f - %.1f.\n"), 0.0f, 255.0f);
    }
    if (!m_unsharp.get()
        || std::dynamic_pointer_cast<RGYFilterParamUnsharp>(m_param)->unsharp != prm->unsharp) {
        const auto options = strsprintf("-D Type=%s -D radius=%d -D bit_depth=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            prm->unsharp.radius,
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp]);
        m_unsharp.set(m_cl->buildResourceAsync(_T("RGY_FILTER_UNSHARP_CL"), _T("EXE_DATA"), options.c_str()));
        float sigmaY = 0.8f + 0.3f * prm->unsharp.radius;
        float sigmaUV = (RGY_CSP_CHROMA_FORMAT[prm->frameIn.csp] == RGY_CHROMAFMT_YUV420) ? 0.8f + 0.3f * (prm->unsharp.radius * 0.5f + 0.25f) : sigmaY;

        if (   RGY_ERR_NONE != (sts = setWeight(m_pGaussWeightBufY,  prm->unsharp.radius, sigmaY))
            || RGY_ERR_NONE != (sts = setWeight(m_pGaussWeightBufUV, prm->unsharp.radius, sigmaUV))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set weight: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
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

RGY_ERR RGYFilterUnsharp::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
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
    if (!m_unsharp.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_UNSHARP_CL(m_unsharp)\n"));
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

void RGYFilterUnsharp::close() {
    m_frameBuf.clear();
    m_unsharp.clear();
    m_cl.reset();
    m_bInterlacedWarn = false;
}
