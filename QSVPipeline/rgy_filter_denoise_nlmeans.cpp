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
#include "rgy_filter_denoise_nlmeans.h"

static const int NLEANS_BLOCK_X = 32;
static const int NLEANS_BLOCK_Y = 8;


enum RGYFilterDenoiseNLMeansTmpBufIdx {
    TMP_U,
    TMP_V,
    TMP_I0,
    TMP_W0,
    TMP_I1,
    TMP_W1,
    TMP_I2,
    TMP_W2,
    TMP_LAST,
};

// https://lcondat.github.io/publis/condat_resreport_NLmeansv3.pdf
RGY_ERR RGYFilterDenoiseNLMeans::denoisePlane(
    RGYFrameInfo *pOutputPlane,
    RGYFrameInfo *pTmpUPlane, RGYFrameInfo *pTmpVPlane,
    RGYFrameInfo *pTmpI0Plane, RGYFrameInfo *pTmpW0Plane,
    RGYFrameInfo *pTmpI1Plane, RGYFrameInfo *pTmpW1Plane,
    RGYFrameInfo *pTmpI2Plane, RGYFrameInfo *pTmpW2Plane,
    const RGYFrameInfo *pInputPlane,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseNLMeans>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    // 一時バッファを初期化
    auto err = m_cl->setPlane(0, pTmpW0Plane, nullptr, queue, wait_events, nullptr);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(setPlane[W0])): %s.\n"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    err = m_cl->setPlane(0, pTmpI0Plane, nullptr, queue);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(setPlane[I0])): %s.\n"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    err = m_cl->setPlane(0, pTmpW1Plane, nullptr, queue);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(setPlane[W1])): %s.\n"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    err = m_cl->setPlane(0, pTmpI1Plane, nullptr, queue);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(setPlane[I1])): %s.\n"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    err = m_cl->setPlane(0, pTmpW2Plane, nullptr, queue);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(setPlane[W2])): %s.\n"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    err = m_cl->setPlane(0, pTmpI2Plane, nullptr, queue);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(setPlane[I2])): %s.\n"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }

    // 計算すべきnx-nyの組み合わせを列挙
    const int support_radius = prm->nlmeans.supportSize / 2;
    std::vector<std::pair<int, int>> nxny;
    for (int ny = -support_radius; ny <= 0; ny++) {
        for (int nx = -support_radius; nx <= support_radius; nx++) {
            if (ny * (2 * support_radius - 1) + nx < 0) {
                nxny.push_back(std::make_pair(nx, ny));
            }
        }
    }
    for (size_t inxny = 0; inxny < nxny.size(); inxny += 2) {
        // nx-nyは2つずつ計算する
        const cl_int2 nx0 = { nxny[inxny+0].first,  (inxny + 1 < nxny.size()) ? nxny[inxny+1].first  : 0 };
        const cl_int2 ny0 = { nxny[inxny+0].second, (inxny + 1 < nxny.size()) ? nxny[inxny+1].second : 0 };
        {
            const char *kernel_name = "kernel_calc_diff_square";
            RGYWorkSize local(NLEANS_BLOCK_X, NLEANS_BLOCK_Y);
            RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
            err = m_nlmeans.get()->kernel(kernel_name).config(queue, local, global, wait_events, nullptr).launch(
                (cl_mem)pTmpUPlane->ptr[0], pTmpUPlane->pitch[0],
                (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
                pOutputPlane->width, pOutputPlane->height,
                nx0, ny0);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(%s)): %s.\n"),
                    char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
                return err;
            }
        }
        {
            const char *kernel_name = "kernel_denoise_nlmeans_calc_v";
            RGYWorkSize local(NLEANS_BLOCK_X, NLEANS_BLOCK_Y);
            RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
            err = m_nlmeans.get()->kernel(kernel_name).config(queue, local, global, wait_events, nullptr).launch(
                (cl_mem)pTmpVPlane->ptr[0], pTmpVPlane->pitch[0],
                (cl_mem)pTmpUPlane->ptr[0], pTmpUPlane->pitch[0],
                pOutputPlane->width, pOutputPlane->height);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(%s)): %s.\n"),
                    char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
                return err;
            }
        }
        {
            const char *kernel_name = "kernel_denoise_nlmeans_calc_weight";
            RGYWorkSize local(NLEANS_BLOCK_X, NLEANS_BLOCK_Y);
            RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
            err = m_nlmeans.get()->kernel(kernel_name).config(queue, local, global, wait_events, nullptr).launch(
                (cl_mem)pTmpI0Plane->ptr[0], (cl_mem)pTmpI1Plane->ptr[0], (cl_mem)pTmpI2Plane->ptr[0],
                (cl_mem)pTmpW0Plane->ptr[0], (cl_mem)pTmpW1Plane->ptr[0], (cl_mem)pTmpW2Plane->ptr[0], pTmpI0Plane->pitch[0],
                (cl_mem)pTmpVPlane->ptr[0], pTmpVPlane->pitch[0],
                (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
                pOutputPlane->width, pOutputPlane->height,
                prm->nlmeans.sigma, 1.0f / (prm->nlmeans.h * prm->nlmeans.h),
                nx0, ny0);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(%s)): %s.\n"),
                    char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
                return err;
            }
        }
    }
    // 最後に規格化
    {
        const char *kernel_name = "kernel_denoise_nlmeans_normalize";
        RGYWorkSize local(NLEANS_BLOCK_X, NLEANS_BLOCK_Y);
        RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
        err = m_nlmeans.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
            (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
            (cl_mem)pTmpI0Plane->ptr[0], (cl_mem)pTmpI1Plane->ptr[0], (cl_mem)pTmpI2Plane->ptr[0],
            (cl_mem)pTmpW0Plane->ptr[0], (cl_mem)pTmpW1Plane->ptr[0], (cl_mem)pTmpW2Plane->ptr[0], pTmpI0Plane->pitch[0],
            pOutputPlane->width, pOutputPlane->height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pOutputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseNLMeans::denoiseFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        auto planeTmpU = getPlane(&m_tmpBuf[TMP_U]->frame, (RGY_PLANE)i);
        auto planeTmpV = getPlane(&m_tmpBuf[TMP_V]->frame, (RGY_PLANE)i);
        auto planeTmpI0 = getPlane(&m_tmpBuf[TMP_I0]->frame, (RGY_PLANE)i);
        auto planeTmpW0 = getPlane(&m_tmpBuf[TMP_W0]->frame, (RGY_PLANE)i);
        auto planeTmpI1 = getPlane(&m_tmpBuf[TMP_I1]->frame, (RGY_PLANE)i);
        auto planeTmpW1 = getPlane(&m_tmpBuf[TMP_W1]->frame, (RGY_PLANE)i);
        auto planeTmpI2 = getPlane(&m_tmpBuf[TMP_I2]->frame, (RGY_PLANE)i);
        auto planeTmpW2 = getPlane(&m_tmpBuf[TMP_W2]->frame, (RGY_PLANE)i);
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[pOutputFrame->csp] - 1) ? event : nullptr;
        auto err = denoisePlane(&planeDst, &planeTmpU, &planeTmpV, &planeTmpI0, &planeTmpW0, &planeTmpI1, &planeTmpW1, &planeTmpI2, &planeTmpW2, &planeSrc,
            queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to denoise(nlmeans) frame(%d) %s: %s\n"), i, cl_errmes(err));
            return err_cl_to_rgy(err);
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterDenoiseNLMeans::RGYFilterDenoiseNLMeans(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_nlmeans(), m_srcImagePool() {
    m_name = _T("nlmeans");
}

RGYFilterDenoiseNLMeans::~RGYFilterDenoiseNLMeans() {
    close();
}

RGY_ERR RGYFilterDenoiseNLMeans::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseNLMeans>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nlmeans.patchSize % 2 == 0) {
        prm->nlmeans.patchSize++; // 奇数にする
    }
    if (prm->nlmeans.patchSize <= 2) {
        AddMessage(RGY_LOG_ERROR, _T("patch must be 3 or bigger.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nlmeans.supportSize % 2 == 0) {
        prm->nlmeans.supportSize++; // 奇数にする
    }
    if (prm->nlmeans.supportSize <= 2) {
        AddMessage(RGY_LOG_ERROR, _T("support must be a 3 or bigger.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //if (pNLMeansParam->nlmeans.radius > KNN_RADIUS_MAX) {
    //    AddMessage(RGY_LOG_ERROR, _T("radius must be <= %d.\n"), KNN_RADIUS_MAX);
    //    return RGY_ERR_INVALID_PARAM;
    //}
    if (prm->nlmeans.sigma < 0.0 || 1.0 < prm->nlmeans.sigma) {
        AddMessage(RGY_LOG_ERROR, _T("sigma should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nlmeans.h < 0.0 || 1.0 < prm->nlmeans.h) {
        AddMessage(RGY_LOG_ERROR, _T("h should be 0.0 - 1.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nlmeans.prec != VPP_FP_PRECISION_FP32) {
        if (!RGYOpenCLDevice(m_cl->queue().devid()).checkExtension("cl_khr_fp16")) {
            AddMessage((!m_param && prm->nlmeans.prec == VPP_FP_PRECISION_FP16) ? RGY_LOG_WARN : RGY_LOG_DEBUG, _T("fp16 not supported on this device, using fp32 mode.\n"));
            prm->nlmeans.prec = VPP_FP_PRECISION_FP32;
        }
    }
    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamDenoiseNLMeans>(m_param);
    if (!m_nlmeans.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
        || prmPrev->nlmeans.patchSize != prm->nlmeans.patchSize
        || prmPrev->nlmeans.supportSize != prm->nlmeans.supportSize
        || prmPrev->nlmeans.prec != prm->nlmeans.prec) {
        const bool use_vtype2_fp16 = prm->nlmeans.prec != VPP_FP_PRECISION_FP32;
        const int support_radius = prm->nlmeans.supportSize / 2;
        const int template_radius = prm->nlmeans.patchSize / 2;
        const int shared_radius = std::max(support_radius, template_radius);
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d"
            " -D TmpVType2=%s -D TmpVTypeFP16=%d -D TmpWPType=float -D TmpWPType2=float2"
            " -D support_radius=%d -D template_radius=%d -D shared_radius=%d"
            " -D NLEANS_BLOCK_X=%d -D NLEANS_BLOCK_Y=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp],
            use_vtype2_fp16 ? "half2" : "float2", use_vtype2_fp16 ? 1 : 0,
            support_radius, template_radius, shared_radius,
            NLEANS_BLOCK_X, NLEANS_BLOCK_Y);
        m_nlmeans.set(m_cl->buildResourceAsync(_T("RGY_FILTER_DENOISE_NLMEANS_CL"), _T("EXE_DATA"), options.c_str()));
    }

    for (size_t i = 0; i < m_tmpBuf.size(); i++) {
        const int tmpBufWidth  = prm->frameOut.width;
        const int tmpBufHeight = prm->frameOut.height;
        if (m_tmpBuf[i]
            && (m_tmpBuf[i]->frame.width != tmpBufWidth || m_tmpBuf[i]->frame.height != tmpBufHeight)) {
            m_tmpBuf[i].reset();
        }
        if (!m_tmpBuf[i]) {
            RGYFrameInfo frameInfo = prm->frameOut;
            frameInfo.width = tmpBufWidth;
            frameInfo.height = tmpBufHeight;
            frameInfo.csp = RGY_CSP_RGB_F32;
            m_tmpBuf[i] = m_cl->createFrameBuffer(frameInfo);
        }
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

RGY_ERR RGYFilterDenoiseNLMeans::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
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
    if (!m_nlmeans.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_DENOISE_NLMEANS_CL(m_nlmeans)\n"));
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

void RGYFilterDenoiseNLMeans::close() {
    m_srcImagePool.clear();
    m_frameBuf.clear();
    m_nlmeans.clear();
    for (auto& f : m_tmpBuf) {
        f.reset();
    }
    m_cl.reset();
    m_bInterlacedWarn = false;
}
