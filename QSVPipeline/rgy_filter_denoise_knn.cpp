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
static const int KNN_TEMPORAL_MAX = 2;

RGY_ERR RGYFilterDenoiseKnn::denoisePlane(RGYFrameInfo *pOutputPlane, const std::array<const RGYFrameInfo *, 5> &pSrcPlanes, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseKnn>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    {
        const float strength = 1.0f / (prm->knn.strength * prm->knn.strength);
        const char *kernel_name = "kernel_denoise_knn";
        RGYWorkSize local(32, 8);
        RGYWorkSize global(ALIGN(pOutputPlane->width, 32), ALIGN(pOutputPlane->height, 8));
        auto err = m_knn.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
            (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
            (cl_mem)pSrcPlanes[0]->ptr[0], (cl_mem)pSrcPlanes[1]->ptr[0], (cl_mem)pSrcPlanes[2]->ptr[0], (cl_mem)pSrcPlanes[3]->ptr[0], (cl_mem)pSrcPlanes[4]->ptr[0],
            strength, prm->knn.lerpC, prm->knn.weight_threshold, prm->knn.lerp_threshold);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pSrcPlanes[2]->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseKnn::denoiseFrame(RGYFrameInfo *pOutputFrame, const std::array<const RGYFrameInfo *, 5> &pSrc, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseKnn>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int temporal_d = prm->knn.d;
    // pSrc[2] が現在フレーム、前後 temporal_d フレーム分の image を作成する。
    // 先頭/末尾のクランプで同じフレームが複数スロットに入る場合は共有する。
    std::array<std::unique_ptr<RGYCLFrame, RGYCLImageFromBufferDeleter>, 5> srcImages;
    std::array<const RGYFrameInfo *, 5> srcImageInfo = { nullptr, nullptr, nullptr, nullptr, nullptr };
    for (int t = 2 - temporal_d; t <= 2 + temporal_d; t++) {
        for (int i = 2 - temporal_d; i < t; i++) {
            if (pSrc[i]->ptr[0] == pSrc[t]->ptr[0]) {
                srcImageInfo[t] = srcImageInfo[i];
                break;
            }
        }
        if (!srcImageInfo[t]) {
            srcImages[t] = m_cl->createImageFromFrameBuffer(*pSrc[t], true, CL_MEM_READ_ONLY, &m_srcImagePool);
            if (!srcImages[t]) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to create image for input frame.\n"));
                return RGY_ERR_MEM_OBJECT_ALLOCATION_FAILURE;
            }
            srcImageInfo[t] = &srcImages[t]->frame;
        }
    }
    for (int t = 0; t < 5; t++) {
        // カーネルから参照されないスロットにも有効な image object を入れておく。
        if (!srcImageInfo[t]) {
            srcImageInfo[t] = srcImageInfo[2];
        }
    }

    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        std::array<RGYFrameInfo, 5> planeSrc;
        std::array<const RGYFrameInfo *, 5> planeSrcPtr;
        for (int t = 0; t < 5; t++) {
            planeSrc[t] = getPlane(srcImageInfo[t], (RGY_PLANE)i);
            planeSrcPtr[t] = &planeSrc[t];
        }
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[pOutputFrame->csp] - 1) ? event : nullptr;
        auto err = denoisePlane(&planeDst, planeSrcPtr, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to denoise(knn) frame(%d) %s: %s\n"), i, cl_errmes(err));
            return err_cl_to_rgy(err);
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterDenoiseKnn::RGYFilterDenoiseKnn(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_knn(), m_srcImagePool(), m_prevFrames(), m_cacheIdx(0), m_frameOut(0) {
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
    if (pKnnParam->knn.d < 0 || pKnnParam->knn.d > KNN_TEMPORAL_MAX) {
        AddMessage(RGY_LOG_ERROR, _T("d must be 0 - %d.\n"), KNN_TEMPORAL_MAX);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pKnnParam->knn.strength <= 0.0 || 1.0 < pKnnParam->knn.strength) {
        // 有効化されたフィルタで strength == 0 は意味がなく、
        // host 側の 1/(strength*strength) 計算で NaN フレームが出力される。
        AddMessage(RGY_LOG_ERROR, _T("strength should be greater than 0.0, up to 1.0.\n"));
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
        || prmPrev->knn.radius != pKnnParam->knn.radius
        || prmPrev->knn.d != pKnnParam->knn.d) {
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d -D knn_radius=%d -D temporal_d=%d",
            RGY_CSP_BIT_DEPTH[pKnnParam->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[pKnnParam->frameOut.csp],
            pKnnParam->knn.radius,
            pKnnParam->knn.d);
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

    if (pKnnParam->knn.d > 0) {
        //convolution3dと同様に前後フレームをキャッシュし、dフレーム遅れで出力する
        const int cacheFrames = 2 * pKnnParam->knn.d + 1;
        if ((int)m_prevFrames.size() != cacheFrames
            || !m_prevFrames.front()
            || cmpFrameInfoCspResolution(&m_prevFrames.front()->frame, &pKnnParam->frameOut)) {
            m_prevFrames.clear();
            m_prevFrames.resize(cacheFrames);
            for (auto& f : m_prevFrames) {
                f = m_cl->createFrameBuffer(pKnnParam->frameOut);
                if (!f) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for frame cache.\n"));
                    return RGY_ERR_NULL_PTR;
                }
            }
            m_cacheIdx = 0;
            m_frameOut = 0;
        }
        //遅延が発生するため、タイムスタンプ等はフィルタ側で設定する
        m_pathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_DATA));
    } else {
        m_prevFrames.clear();
    }

    //コピーを保存
    setFilterInfo(pKnnParam->print());
    m_param = pKnnParam;
    return sts;
}

RGY_ERR RGYFilterDenoiseKnn::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;

    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseKnn>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int temporal_d = prm->knn.d;

    if (pInputFrame->ptr[0] == nullptr
        && (temporal_d == 0 || m_frameOut >= m_cacheIdx)) {
        //終了
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return sts;
    }
    if (!m_knn.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_DENOISE_KNN_CL(m_knn)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    if (temporal_d == 0) {
        //空間のみ(従来)のパス、遅延なし
        *pOutputFrameNum = 1;
        if (ppOutputFrames[0] == nullptr) {
            auto pOutFrame = m_frameBuf[0].get();
            ppOutputFrames[0] = &pOutFrame->frame;
        }
        ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
        //if (interlaced(*pInputFrame)) {
        //    return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
        //}
        const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
        if (memcpyKind != RGYCLMemcpyD2D) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        const std::array<const RGYFrameInfo *, 5> pSrc = { pInputFrame, pInputFrame, pInputFrame, pInputFrame, pInputFrame };
        sts = denoiseFrame(ppOutputFrames[0], pSrc, queue, wait_events, event);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at denoiseFrame (%s): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
            return sts;
        }
        return sts;
    }

    //temporal_d > 0: convolution3dと同様に前後フレームをキャッシュし、temporal_dフレーム遅れで出力する
    std::vector<RGYOpenCLEvent> kernelWaitEvents = wait_events;
    if (pInputFrame->ptr[0]) {
        const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, m_frameBuf[0]->frame.mem_type);
        if (memcpyKind != RGYCLMemcpyD2D) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        // sourceキャッシュにコピー
        auto cacheFrame = &m_prevFrames[m_cacheIdx % m_prevFrames.size()]->frame;
        sts = m_cl->copyFrame(cacheFrame, pInputFrame, nullptr, queue, wait_events, nullptr);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set frame to data cache: %s.\n"), get_err_mes(sts));
            return sts;
        }
        copyFrameProp(cacheFrame, pInputFrame);
        m_cacheIdx++;
        kernelWaitEvents.clear(); // in-order queue ではキャッシュコピーがすでに wait 済み
    }

    //出力するフレームの前後temporal_dフレームがそろうまでは出力しない
    if (pInputFrame->ptr[0] != nullptr && m_cacheIdx < m_frameOut + temporal_d + 1) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return sts;
    }

    RGYCLFrame *pOutFrame = m_frameBuf[0].get();
    *pOutputFrameNum = 1;
    ppOutputFrames[0] = &pOutFrame->frame;

    //出力フレームの前後temporal_dフレームを集める(先頭/末尾はクランプ)
    std::array<const RGYFrameInfo *, 5> pSrc = { nullptr, nullptr, nullptr, nullptr, nullptr };
    for (int t = -2; t <= 2; t++) {
        const int idx = std::max(0, std::min(m_frameOut + t, m_cacheIdx - 1));
        pSrc[t + 2] = &m_prevFrames[idx % m_prevFrames.size()]->frame;
    }
    const RGYFrameInfo *frameCur = pSrc[2];
    pOutFrame->frame.picstruct = frameCur->picstruct;
    copyFramePropWithoutRes(&pOutFrame->frame, frameCur);

    sts = denoiseFrame(&pOutFrame->frame, pSrc, queue, kernelWaitEvents, event);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at denoiseFrame (%s): %s.\n"),
            RGY_CSP_NAMES[frameCur->csp], get_err_mes(sts));
        return sts;
    }
    m_frameOut++;
    return sts;
}

void RGYFilterDenoiseKnn::close() {
    m_srcImagePool.clear();
    m_frameBuf.clear();
    m_prevFrames.clear();
    m_cacheIdx = 0;
    m_frameOut = 0;
    m_knn.clear();
    m_cl.reset();
    m_bInterlacedWarn = false;
}
