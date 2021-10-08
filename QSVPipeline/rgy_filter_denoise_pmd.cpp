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
#include "rgy_filter_denoise_pmd.h"

static const int KNN_RADIUS_MAX = 5;

static int final_dst_index(int loop_count) {
    return (loop_count - 1) & 1;
}

RGY_ERR RGYFilterDenoisePmd::runGaussPlane(RGYFrameInfo *pGaussPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoisePmd>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    RGYWorkSize local(32, 8);
    RGYWorkSize global(pInputPlane->width, pInputPlane->height);
    const char *kernel_name = "kernel_denoise_pmd_gauss";
    auto err = m_pmd.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pGaussPlane->ptr[0], pGaussPlane->pitch[0], pGaussPlane->width, pGaussPlane->height,
        (cl_mem)pInputPlane->ptr[0]);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoisePmd::runGaussFrame(RGYFrameInfo *pGaussFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    for (int i = 0; i < RGY_CSP_PLANES[pInputFrame->csp]; i++) {
        auto planeDst = getPlane(pGaussFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[pInputFrame->csp] - 1) ? event : nullptr;
        auto err = runGaussPlane(&planeDst, &planeSrc, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to denoise(pmd(gauss)) frame(%d) %s: %s\n"), i, cl_errmes(err));
            return err_cl_to_rgy(err);
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoisePmd::runPmdPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const RGYFrameInfo *pGaussPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoisePmd>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const float range = 4.0f;
    const float strength2 = prm->pmd.strength / (range * 100.0f);
    const float threshold2 = std::pow(2.0f, prm->pmd.threshold / 10.0f - (12 - RGY_CSP_BIT_DEPTH[pInputPlane->csp]) * 2.0f);
    const float inv_threshold2 = 1.0f / threshold2;

    RGYWorkSize local(32, 8);
    RGYWorkSize global(pInputPlane->width, pInputPlane->height);
    const char *kernel_name = "kernel_denoise_pmd";
    auto err = m_pmd.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (cl_mem)pInputPlane->ptr[0], (cl_mem)pGaussPlane->ptr[0],
        strength2, inv_threshold2);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoisePmd::runPmdFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pGaussFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        auto planeGauss = getPlane(pGaussFrame, (RGY_PLANE)i);
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[pOutputFrame->csp] - 1) ? event : nullptr;
        auto err = runPmdPlane(&planeDst, &planeSrc, &planeGauss, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to denoise(pmd) frame(%d) %s: %s\n"), i, cl_errmes(err));
            return err_cl_to_rgy(err);
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoisePmd::denoiseFrame(RGYFrameInfo *pOutputFrame[2], const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoisePmd>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto srcImage = m_cl->createImageFromFrameBuffer(*pInputFrame, true, CL_MEM_READ_ONLY);
    auto ret = runGaussFrame(&m_gauss->frame, &srcImage->frame, queue, wait_events, nullptr);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }
    auto gaussImage = m_cl->createImageFromFrameBuffer(m_gauss->frame, true, CL_MEM_READ_ONLY);

    for (int i = 0; i < prm->pmd.applyCount; i++) {
        const int dst_index = i & 1;
        ret = runPmdFrame(pOutputFrame[dst_index], &srcImage->frame, &gaussImage->frame, queue, {}, (i == prm->pmd.applyCount - 1) ? event : nullptr);
        if (i < prm->pmd.applyCount - 1) {
            srcImage = m_cl->createImageFromFrameBuffer(*(pOutputFrame[dst_index]), true, CL_MEM_READ_ONLY);
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterDenoisePmd::RGYFilterDenoisePmd(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_bInterlacedWarn(false), m_frameIdx(0), m_pmd(), m_gauss() {
    m_name = _T("pmd");
}

RGYFilterDenoisePmd::~RGYFilterDenoisePmd() {
    close();
}

RGY_ERR RGYFilterDenoisePmd::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto pPmdParam = std::dynamic_pointer_cast<RGYFilterParamDenoisePmd>(pParam);
    if (!pPmdParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (pPmdParam->frameOut.height <= 0 || pPmdParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pPmdParam->pmd.applyCount <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, apply_count must be a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pPmdParam->pmd.strength < 0.0f || 100.0f < pPmdParam->pmd.strength) {
        AddMessage(RGY_LOG_WARN, _T("strength must be in range of 0.0 - 100.0.\n"));
        pPmdParam->pmd.strength = clamp(pPmdParam->pmd.strength, 0.0f, 100.0f);
    }
    if (pPmdParam->pmd.threshold < 0.0f || 255.0f < pPmdParam->pmd.threshold) {
        AddMessage(RGY_LOG_WARN, _T("strength must be in range of 0.0 - 255.0.\n"));
        pPmdParam->pmd.threshold = clamp(pPmdParam->pmd.threshold, 0.0f, 255.0f);
    }
    if (!m_pmd.get()
        || std::dynamic_pointer_cast<RGYFilterParamDenoisePmd>(m_param)->pmd != pPmdParam->pmd) {
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d -D useExp=%d",
            RGY_CSP_BIT_DEPTH[pPmdParam->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[pPmdParam->frameOut.csp],
            pPmdParam->pmd.useExp ? 1 : 0);
        m_pmd.set(m_cl->buildResourceAsync(_T("RGY_FILTER_DENOISE_PMD_CL"), _T("EXE_DATA"), options.c_str()));
    }
    if (!m_gauss) {
        m_gauss = m_cl->createFrameBuffer(pPmdParam->frameOut, CL_MEM_READ_WRITE);
    }

    auto err = AllocFrameBuf(pPmdParam->frameOut, 2);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        pPmdParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    //コピーを保存
    setFilterInfo(pParam->print());
    m_param = pParam;
    return sts;
}

RGY_ERR RGYFilterDenoisePmd::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (pInputFrame->ptr[0] == nullptr) {
        return RGY_ERR_NONE;
    }
    auto pPmdParam = std::dynamic_pointer_cast<RGYFilterParamDenoisePmd>(m_param);
    if (!pPmdParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (!m_pmd.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_DENOISE_PMD_CL(m_pmd)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }

    const int out_idx = final_dst_index(pPmdParam->pmd.applyCount);

    *pOutputFrameNum = 1;
    RGYFrameInfo *pOutputFrame[2] = {
        &m_frameBuf[(m_frameIdx++) % m_frameBuf.size()].get()->frame,
        &m_frameBuf[(m_frameIdx++) % m_frameBuf.size()].get()->frame,
    };
    bool frame_swapped = false;
    if (ppOutputFrames[0] != nullptr) {
        //filter_as_interlaced_pair()の時の処理
        frame_swapped = true;
        pOutputFrame[out_idx] = ppOutputFrames[0];
        pOutputFrame[(out_idx + 1) & 1]->width     = pOutputFrame[out_idx]->width;
        pOutputFrame[(out_idx + 1) & 1]->height    = pOutputFrame[out_idx]->height;
        pOutputFrame[(out_idx + 1) & 1]->csp       = pOutputFrame[out_idx]->csp;
        pOutputFrame[(out_idx + 1) & 1]->picstruct = pOutputFrame[out_idx]->picstruct;
        pOutputFrame[(out_idx + 1) & 1]->flags     = pOutputFrame[out_idx]->flags;
    } else {
        ppOutputFrames[0] = pOutputFrame[out_idx];
    }
    //if (interlaced(*pInputFrame)) {
    //    return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
    //}

    return denoiseFrame(pOutputFrame, pInputFrame, queue, wait_events, event);
}

void RGYFilterDenoisePmd::close() {
    m_frameBuf.clear();
    m_gauss.reset();
    m_pmd.clear();
    m_cl.reset();
    m_frameIdx = 0;
    m_bInterlacedWarn = false;
}
