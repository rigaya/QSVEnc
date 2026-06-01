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
#if ENCODER_QSV || ENCODER_VCEENC
static const int NLEANS_BLOCK_Y = 8;
#else
static const int NLEANS_BLOCK_Y = 16;
#endif


enum RGYFilterDenoiseNLMeansTmpBufIdx {
    TMP_U,
    TMP_V,
    TMP_IW0,
    TMP_IW1,
    TMP_IW2,
    TMP_IW3,
    TMP_IW4,
    TMP_IW5,
    TMP_IW6,
    TMP_IW7,
    TMP_IW8,
    TMP_LAST = TMP_IW8,
    TMP_TOTAL,
};

std::vector<std::pair<int, int>> nxnylist(const int search_radius) {
    std::vector<std::pair<int, int>> nxny;
    for (int ny = -search_radius; ny <= 0; ny++) {
        for (int nx = -search_radius; nx <= search_radius; nx++) {
            if (ny * (2 * search_radius - 1) + nx < 0) { // nx-nyの対称性を使って半分のみ計算 (0,0)
                nxny.push_back(std::make_pair(nx, ny));
            }
        }
    }
    return nxny;
}

// Full [-r,+r]x[-r,+r] iteration including (0,0). Used by the temporal
// passes; the spatial half-symmetry trick does not apply across frames
// because the patch in the reference frame is not an output pixel.
static std::vector<std::pair<int, int>> nxnylist_full(const int search_radius) {
    std::vector<std::pair<int, int>> nxny;
    nxny.reserve((2 * search_radius + 1) * (2 * search_radius + 1));
    for (int ny = -search_radius; ny <= search_radius; ny++) {
        for (int nx = -search_radius; nx <= search_radius; nx++) {
            nxny.push_back(std::make_pair(nx, ny));
        }
    }
    return nxny;
}

// https://lcondat.github.io/publis/condat_resreport_NLmeansv3.pdf
RGY_ERR RGYFilterDenoiseNLMeans::denoisePlane(
    RGYFrameInfo *pOutputPlane,
    RGYFrameInfo *pTmpUPlane, RGYFrameInfo *pTmpVPlane,
    RGYFrameInfo *pTmpIWPlane,
    const RGYFrameInfo *pInputPlane,
    const std::vector<const RGYFrameInfo *> &refPlanes,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseNLMeans>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    // 一時バッファを初期化
    auto err = m_cl->setPlane(0, &pTmpIWPlane[0], nullptr, queue, wait_events, nullptr);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error setPlane[IW0](%s): %s.\n"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    for (int i = 1; i < RGY_NLMEANS_DXDY_STEP+1; i++) {
        if (pTmpIWPlane[i].ptr[0]) {
            err = m_cl->setPlane(0, &pTmpIWPlane[i], nullptr, queue, {}, nullptr);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error setPlane[IW%d](%s): %s.\n"), i, RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
                return err;
            }
        }
    }

    // 計算すべきnx-nyの組み合わせを列挙
    const int search_radius = prm->nlmeans.searchSize / 2;
    const std::vector<std::pair<int, int>> nxny = nxnylist(search_radius);
    // nx-nyの組み合わせをRGY_NLMEANS_DXDY_STEP個ずつまとめて計算して高速化
    for (size_t inxny = 0; inxny < nxny.size(); inxny += RGY_NLMEANS_DXDY_STEP) {
        const int offset_count = std::min((int)(nxny.size() - inxny), RGY_NLMEANS_DXDY_STEP);
        if (m_nlmeans.find(offset_count) == m_nlmeans.end()) {
            AddMessage(RGY_LOG_ERROR, _T("program for offset_count=%d not found (denoisePlane(%s)).\n"), offset_count, RGY_CSP_NAMES[pInputPlane->csp]);
            return RGY_ERR_UNKNOWN;
        }
        cl_int nx0arr[RGY_NLMEANS_DXDY_STEP], ny0arr[RGY_NLMEANS_DXDY_STEP];
        int nymin = 0;
        for (int i = 0; i < RGY_NLMEANS_DXDY_STEP; i++) {
            nx0arr[i] = (inxny + i < nxny.size()) ? nxny[inxny + i].first : 0;
            ny0arr[i] = (inxny + i < nxny.size()) ? nxny[inxny + i].second : 0;
            nymin = std::min(nymin, ny0arr[i]);
        }
        //kernel引数に渡すために、cl_int8に押し込む
        cl_int8 nx0, ny0;
        memcpy(&nx0, nx0arr, sizeof(nx0));
        memcpy(&ny0, ny0arr, sizeof(ny0));
        {
            const char *kernel_name = "kernel_calc_diff_square";
            RGYWorkSize local(NLEANS_BLOCK_X, NLEANS_BLOCK_Y);
            RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
            err = m_nlmeans[offset_count]->get()->kernel(kernel_name).config(queue, local, global, {}, nullptr).launch(
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
            err = m_nlmeans[offset_count]->get()->kernel(kernel_name).config(queue, local, global, {}, nullptr).launch(
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
            err = m_nlmeans[offset_count]->get()->kernel(kernel_name).config(queue, local, global, {}, nullptr).launch(
                (cl_mem)pTmpIWPlane[0].ptr[0],
                (cl_mem)pTmpIWPlane[1].ptr[0], (cl_mem)pTmpIWPlane[2].ptr[0], (cl_mem)pTmpIWPlane[3].ptr[0], (cl_mem)pTmpIWPlane[4].ptr[0],
                (cl_mem)pTmpIWPlane[5].ptr[0], (cl_mem)pTmpIWPlane[6].ptr[0], (cl_mem)pTmpIWPlane[7].ptr[0], (cl_mem)pTmpIWPlane[8].ptr[0],
                pTmpIWPlane[0].pitch[0],
                (cl_mem)pTmpVPlane->ptr[0], pTmpVPlane->pitch[0],
                (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
                pOutputPlane->width, pOutputPlane->height,
                prm->nlmeans.sigma, 1.0f / (prm->nlmeans.h * prm->nlmeans.h),
                nx0, ny0, nymin);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(%s)): %s.\n"),
                    char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
                return err;
            }
        }
    }
    // Temporal passes — for each reference frame in [-d, +d] except 0,
    // accumulate additional contributions into IW0. The temporal kernel
    // reads patches at offset positions from pRef and weights them into
    // the current-frame output. Uses the full nxny list (no symmetry),
    // but with the user-tunable searchSizeT radius (default 5) instead
    // of the spatial searchSize, because inter-frame motion at +/- k
    // frames is typically small enough that a tight temporal window
    // covers the useful matches.
    if (!refPlanes.empty()) {
        auto prmRef = std::dynamic_pointer_cast<RGYFilterParamDenoiseNLMeans>(m_param);
        const int search_radius_t = prmRef->nlmeans.searchSizeT / 2;
        const std::vector<std::pair<int, int>> nxny_full = nxnylist_full(search_radius_t);
        for (const RGYFrameInfo *pRefPlane : refPlanes) {
            if (!pRefPlane || !pRefPlane->ptr[0]) continue;
            for (size_t inxny = 0; inxny < nxny_full.size(); inxny += RGY_NLMEANS_DXDY_STEP) {
                const int offset_count_t = std::min((int)(nxny_full.size() - inxny), RGY_NLMEANS_DXDY_STEP);
                if (m_nlmeansTemporal.find(offset_count_t) == m_nlmeansTemporal.end()) {
                    AddMessage(RGY_LOG_ERROR, _T("temporal program for offset_count=%d not found (denoisePlane(%s)).\n"),
                        offset_count_t, RGY_CSP_NAMES[pInputPlane->csp]);
                    return RGY_ERR_UNKNOWN;
                }
                cl_int nx0arr_t[RGY_NLMEANS_DXDY_STEP] = {0};
                cl_int ny0arr_t[RGY_NLMEANS_DXDY_STEP] = {0};
                for (int i = 0; i < offset_count_t; i++) {
                    nx0arr_t[i] = nxny_full[inxny + i].first;
                    ny0arr_t[i] = nxny_full[inxny + i].second;
                }
                cl_int8 nx0_t, ny0_t;
                memcpy(&nx0_t, nx0arr_t, sizeof(nx0_t));
                memcpy(&ny0_t, ny0arr_t, sizeof(ny0_t));
                {
                    const char *kernel_name = "kernel_calc_diff_square_temporal";
                    RGYWorkSize local(NLEANS_BLOCK_X, NLEANS_BLOCK_Y);
                    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
                    err = m_nlmeansTemporal[offset_count_t]->get()->kernel(kernel_name).config(queue, local, global, {}, nullptr).launch(
                        (cl_mem)pTmpUPlane->ptr[0], pTmpUPlane->pitch[0],
                        (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
                        (cl_mem)pRefPlane->ptr[0], pRefPlane->pitch[0],
                        pOutputPlane->width, pOutputPlane->height,
                        nx0_t, ny0_t);
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
                    err = m_nlmeansTemporal[offset_count_t]->get()->kernel(kernel_name).config(queue, local, global, {}, nullptr).launch(
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
                    const char *kernel_name = "kernel_denoise_nlmeans_calc_weight_temporal";
                    RGYWorkSize local(NLEANS_BLOCK_X, NLEANS_BLOCK_Y);
                    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
                    err = m_nlmeansTemporal[offset_count_t]->get()->kernel(kernel_name).config(queue, local, global, {}, nullptr).launch(
                        (cl_mem)pTmpIWPlane[0].ptr[0],
                        pTmpIWPlane[0].pitch[0],
                        (cl_mem)pTmpVPlane->ptr[0], pTmpVPlane->pitch[0],
                        (cl_mem)pRefPlane->ptr[0], pRefPlane->pitch[0],
                        pOutputPlane->width, pOutputPlane->height,
                        prmRef->nlmeans.sigma, 1.0f / (prmRef->nlmeans.h * prmRef->nlmeans.h),
                        nx0_t, ny0_t);
                    if (err != RGY_ERR_NONE) {
                        AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(%s)): %s.\n"),
                            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
                        return err;
                    }
                }
            }
        }
    }

    // 最後に規格化
    {
        const char *kernel_name = "kernel_denoise_nlmeans_normalize";
        RGYWorkSize local(NLEANS_BLOCK_X, NLEANS_BLOCK_Y);
        RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
        err = m_nlmeans.begin()->second->get()->kernel(kernel_name).config(queue, local, global, {}, event).launch(
            (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
            (cl_mem)pTmpIWPlane[0].ptr[0],
            (cl_mem)pTmpIWPlane[1].ptr[0], (cl_mem)pTmpIWPlane[2].ptr[0], (cl_mem)pTmpIWPlane[3].ptr[0], (cl_mem)pTmpIWPlane[4].ptr[0],
            (cl_mem)pTmpIWPlane[5].ptr[0], (cl_mem)pTmpIWPlane[6].ptr[0], (cl_mem)pTmpIWPlane[7].ptr[0], (cl_mem)pTmpIWPlane[8].ptr[0],
            pTmpIWPlane[0].pitch[0],
            (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
            pOutputPlane->width, pOutputPlane->height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pOutputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseNLMeans::denoiseFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const std::vector<const RGYFrameInfo *> &refFrames,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseNLMeans>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!prm->nlmeans.processChroma && RGY_CSP_PLANES[pOutputFrame->csp] > 1) {
        RGYOpenCLEvent copyEvent;
        auto copyErr = m_cl->copyFrame(pOutputFrame, pInputFrame, nullptr, queue, wait_events, &copyEvent);
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at copyFrame before luma-only denoise(nlmeans) (%s): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(copyErr));
            return copyErr;
        }
        auto planeDst = getPlane(pOutputFrame, RGY_PLANE_Y);
        auto planeSrc = getPlane(pInputFrame, RGY_PLANE_Y);
        auto planeTmpU = getPlane(&m_tmpBuf[TMP_U]->frame, RGY_PLANE_Y);
        auto planeTmpV = getPlane(&m_tmpBuf[TMP_V]->frame, RGY_PLANE_Y);
        std::array<RGYFrameInfo, RGY_NLMEANS_DXDY_STEP+1> pTmpIWPlane;
        for (size_t j = 0; j < pTmpIWPlane.size(); j++) {
            if (m_tmpBuf[TMP_IW0 + j]) {
                pTmpIWPlane[j] = getPlane(&m_tmpBuf[TMP_IW0 + j]->frame, RGY_PLANE_Y);
            } else {
                pTmpIWPlane[j] = RGYFrameInfo();
            }
        }
        std::vector<RGYFrameInfo> refPlaneInfo;
        std::vector<const RGYFrameInfo *> refPlanePtrs;
        refPlaneInfo.reserve(refFrames.size());
        refPlanePtrs.reserve(refFrames.size());
        for (const RGYFrameInfo *pRef : refFrames) {
            if (pRef && pRef->ptr[0]) {
                refPlaneInfo.push_back(getPlane(pRef, RGY_PLANE_Y));
                refPlanePtrs.push_back(&refPlaneInfo.back());
            }
        }
        auto err = denoisePlane(&planeDst, &planeTmpU, &planeTmpV, pTmpIWPlane.data(), &planeSrc,
            refPlanePtrs, queue, std::vector<RGYOpenCLEvent>{ copyEvent }, event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to denoise(nlmeans) luma plane: %s\n"), cl_errmes(err));
            return err_cl_to_rgy(err);
        }
        return RGY_ERR_NONE;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        auto planeTmpU = getPlane(&m_tmpBuf[TMP_U]->frame, (RGY_PLANE)i);
        auto planeTmpV = getPlane(&m_tmpBuf[TMP_V]->frame, (RGY_PLANE)i);
        std::array<RGYFrameInfo, RGY_NLMEANS_DXDY_STEP+1> pTmpIWPlane;
        for (size_t j = 0; j < pTmpIWPlane.size(); j++) {
            if (m_tmpBuf[TMP_IW0 + j]) {
                pTmpIWPlane[j] = getPlane(&m_tmpBuf[TMP_IW0 + j]->frame, (RGY_PLANE)i);
            } else {
                pTmpIWPlane[j] = RGYFrameInfo();
            }
        }
        // Per-plane reference frame info. Storage backs the const pointers
        // we hand to denoisePlane.
        std::vector<RGYFrameInfo> refPlaneInfo;
        std::vector<const RGYFrameInfo *> refPlanePtrs;
        refPlaneInfo.reserve(refFrames.size());
        refPlanePtrs.reserve(refFrames.size());
        for (const RGYFrameInfo *pRef : refFrames) {
            if (pRef && pRef->ptr[0]) {
                refPlaneInfo.push_back(getPlane(pRef, (RGY_PLANE)i));
                refPlanePtrs.push_back(&refPlaneInfo.back());
            }
        }
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[pOutputFrame->csp] - 1) ? event : nullptr;
        auto err = denoisePlane(&planeDst, &planeTmpU, &planeTmpV, pTmpIWPlane.data(), &planeSrc,
            refPlanePtrs, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to denoise(nlmeans) frame(%d) %s: %s\n"), i, cl_errmes(err));
            return err_cl_to_rgy(err);
        }
    }
    return RGY_ERR_NONE;
}

// Emit the denoised output for cache slot idx_cur. Builds the temporal
// reference frame list by gathering the d frames on each side of idx_cur
// from the cache; missing slots (before stream start / after end) are
// dropped from the list, which effectively shrinks the temporal window
// near boundaries.
RGY_ERR RGYFilterDenoiseNLMeans::emitFrame(int idx_cur, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseNLMeans>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int d = prm->nlmeans.d;
    const int cacheSize = (int)m_cacheFrames.size();
    const RGYFrameInfo *pInputFrame = &m_cacheFrames[idx_cur]->frame;

    // Build refFrames by mapping the centre frame's absolute index +/- k to
    // cache slots, dropping any slot that doesn't currently hold a valid
    // input.
    std::vector<const RGYFrameInfo *> refFrames;
    refFrames.reserve(2 * d);
    // Centre's absolute input index: idx_cur was filled at input number
    // (m_outputCount + 1) under the d-frame-latency rule; reconstruct it
    // from cache occupancy by walking +/- k.
    for (int k = 1; k <= d; k++) {
        // Past reference at frame (centre - k). Cache holds 2d+1 most
        // recent inputs; abs_past >= 0 guarantees the slot is still
        // valid because the centre frame is at most d positions older
        // than the newest input.
        const int abs_past = m_outputCount - k;
        if (abs_past >= 0) {
            refFrames.push_back(&m_cacheFrames[abs_past % cacheSize]->frame);
        }
        // Future reference at frame (centre + k). Missing if we haven't
        // ingested that frame yet (drain phase near EOS).
        const int abs_future = m_outputCount + k;
        if (abs_future < m_inputCount) {
            refFrames.push_back(&m_cacheFrames[abs_future % cacheSize]->frame);
        }
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    ppOutputFrames[0]->timestamp = pInputFrame->timestamp;
    ppOutputFrames[0]->duration = pInputFrame->duration;
    ppOutputFrames[0]->inputFrameId = pInputFrame->inputFrameId;
    ppOutputFrames[0]->flags = pInputFrame->flags;

    auto sts = denoiseFrame(ppOutputFrames[0], pInputFrame, refFrames, queue, wait_events, event);
    if (sts != RGY_ERR_NONE) return sts;
    m_outputCount++;
    return RGY_ERR_NONE;
}

RGYFilterDenoiseNLMeans::RGYFilterDenoiseNLMeans(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_nlmeans(),
    m_nlmeansTemporal(),
    m_tmpBuf(),
    m_cacheFrames(),
    m_inputCount(0),
    m_outputCount(0),
    m_drained(false) {
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
    if (prm->nlmeans.searchSize % 2 == 0) {
        prm->nlmeans.searchSize++; // 奇数にする
    }
    if (prm->nlmeans.searchSize <= 2) {
        AddMessage(RGY_LOG_ERROR, _T("support must be a 3 or bigger.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //if (pNLMeansParam->nlmeans.radius > KNN_RADIUS_MAX) {
    //    AddMessage(RGY_LOG_ERROR, _T("radius must be <= %d.\n"), KNN_RADIUS_MAX);
    //    return RGY_ERR_INVALID_PARAM;
    //}
    if (prm->nlmeans.sigma < 0.0) {
        AddMessage(RGY_LOG_ERROR, _T("sigma should be 0 or larger.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nlmeans.h <= 0.0) {
        AddMessage(RGY_LOG_ERROR, _T("h should be larger than 0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nlmeans.d < 0 || prm->nlmeans.d > FILTER_NLMEANS_D_MAX) {
        prm->nlmeans.d = clamp(prm->nlmeans.d, 0, FILTER_NLMEANS_D_MAX);
        AddMessage(RGY_LOG_WARN, _T("d should be in range of 0 - %d.\n"), FILTER_NLMEANS_D_MAX);
    }
    if (prm->nlmeans.searchSizeT % 2 == 0) {
        prm->nlmeans.searchSizeT++; // 奇数にする
    }
    if (prm->nlmeans.searchSizeT < 3) {
        AddMessage(RGY_LOG_ERROR, _T("search_t must be 3 or bigger.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    // Shared-memory optimisation accumulates spatial-pass weights into
    // a per-workgroup tile that the normalize step never sees from the
    // temporal pass. Force it off whenever d > 0 so both passes write
    // into the same IW0..IW8 layout.
    // SLM tile caching not implemented for temporal path — A770 L2
    // covers the working set (per SLM measurements on this hardware).
    // patch_t= investigated — no benefit on A770; temporal kernel is
    // bandwidth-bound, not compute-bound.
    if (prm->nlmeans.d > 0 && prm->nlmeans.sharedMem) {
        AddMessage(RGY_LOG_DEBUG, _T("disabling sharedMem optimisation because d > 0.\n"));
        prm->nlmeans.sharedMem = false;
    }
    // For d > 0 the centre frame being emitted is NOT the latest input
    // (we run d frames behind). Disable the base class auto-propagation
    // of timestamp / duration / inputFrameId / flags / picstruct from
    // pInputFrame so it doesn't overwrite the values we set in
    // emitFrame() from the cached centre frame. For d = 0 the latest
    // input IS the output, so the default path-through stays on and
    // the d = 0 code path remains byte-identical to prior builds.
    if (prm->nlmeans.d > 0) {
        m_pathThrough &= ~(FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_TIMESTAMP);
    }
    if (prm->nlmeans.fp16 != VppNLMeansFP16Opt::NoOpt) {
        if (!RGYOpenCLDevice(m_cl->queue().devid()).checkExtension("cl_khr_fp16")) {
            AddMessage((!m_param) ? RGY_LOG_INFO : RGY_LOG_DEBUG, _T("fp16 not supported on this device, using fp32 mode.\n"));
            prm->nlmeans.fp16 = VppNLMeansFP16Opt::NoOpt;
        }
    }
    const bool use_vtype_fp16 = prm->nlmeans.fp16 != VppNLMeansFP16Opt::NoOpt;
    const bool use_wptype_fp16 = prm->nlmeans.fp16 == VppNLMeansFP16Opt::All;

    const int search_radius = prm->nlmeans.searchSize / 2;
    // メモリへの書き込みが衝突しないよう、ブロックごとに書き込み先のバッファを分けるが、それがブロックサイズを超えてはいけない
    // x方向は正負両方向にsearch_radius分はみ出し、y方向は負方向にのみsearch_radius分はみ出す
    const bool shared_mem_opt_possible = search_radius * 2 <= NLEANS_BLOCK_X && search_radius <= NLEANS_BLOCK_Y;
    if (prm->nlmeans.sharedMem && !shared_mem_opt_possible) {
        prm->nlmeans.sharedMem = false;
    }
    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamDenoiseNLMeans>(m_param);
    if (m_nlmeans.size() == 0
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
        || prmPrev->nlmeans.patchSize != prm->nlmeans.patchSize
        || prmPrev->nlmeans.searchSize != prm->nlmeans.searchSize
        || prmPrev->nlmeans.sharedMem != prm->nlmeans.sharedMem
        || prmPrev->nlmeans.fp16 != prm->nlmeans.fp16
        || prmPrev->nlmeans.d != prm->nlmeans.d
        || prmPrev->nlmeans.searchSizeT != prm->nlmeans.searchSizeT) {
        const int search_radius_t = prm->nlmeans.searchSizeT / 2;
        std::vector<std::pair<int, int>> nxny = nxnylist(search_radius);
        std::vector<std::pair<int, int>> nxny_full = nxnylist_full(search_radius_t);
        auto add_program = [&](const int offset_count, bool temporal) {
            const int template_radius = prm->nlmeans.patchSize / 2;
            const int shared_radius = std::max(search_radius, template_radius);
            const auto options = strsprintf("-D Type=%s -D bit_depth=%d"
                " -D TmpVType8=%s -D TmpVTypeFP16=%d"
                " -D TmpWPType=%s -D TmpWPType2=%s -D TmpWPType8=%s -D TmpWPTypeFP16=%d"
                " -D search_radius=%d -D template_radius=%d -D shared_radius=%d -D SHARED_OPT=%d"
                " -D NLEANS_BLOCK_X=%d -D NLEANS_BLOCK_Y=%d -D offset_count=%d%s",
                RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
                RGY_CSP_BIT_DEPTH[prm->frameOut.csp],
                use_vtype_fp16 ? "half8" : "float8",
                use_vtype_fp16 ? 1 : 0,
                use_wptype_fp16 ? "half" : "float",
                use_wptype_fp16 ? "half2" : "float2",
                use_wptype_fp16 ? "half8" : "float8",
                use_wptype_fp16 ? 1 : 0,
                search_radius, template_radius, shared_radius,
                prm->nlmeans.sharedMem ? 1 : 0,
                NLEANS_BLOCK_X, NLEANS_BLOCK_Y, offset_count,
                temporal ? " -D TEMPORAL=1" : "");
            auto &target = temporal ? m_nlmeansTemporal : m_nlmeans;
            target[offset_count] = std::make_unique<RGYOpenCLProgramAsync>();
            target[offset_count]->set(m_cl->buildResourceAsync(_T("RGY_FILTER_DENOISE_NLMEANS_CL"), _T("EXE_DATA"), options.c_str()));
        };
        m_nlmeans.clear();
        m_nlmeansTemporal.clear();
        if (nxny.size() >= RGY_NLMEANS_DXDY_STEP) add_program(RGY_NLMEANS_DXDY_STEP, false);
        if (nxny.size() % RGY_NLMEANS_DXDY_STEP) add_program(nxny.size() % RGY_NLMEANS_DXDY_STEP, false);
        if (prm->nlmeans.d > 0) {
            if (nxny_full.size() >= RGY_NLMEANS_DXDY_STEP) add_program(RGY_NLMEANS_DXDY_STEP, true);
            if (nxny_full.size() % RGY_NLMEANS_DXDY_STEP) add_program(nxny_full.size() % RGY_NLMEANS_DXDY_STEP, true);
        }
    }

    for (size_t i = 0; i < m_tmpBuf.size(); i++) {
        int tmpBufWidth = 0;
        if (i == TMP_U || i == TMP_V) {
            tmpBufWidth = prm->frameOut.width * ((use_vtype_fp16) ? 16 /*half8*/ : 32/*float8*/);
        } else {
            tmpBufWidth = prm->frameOut.width * ((use_wptype_fp16) ? 4 /*half2*/ : 8 /*float2*/);
        }
        // sharedメモリを使う場合、TMP_U, TMP_VとTMP_IW0～TMP_IW3のみ使用する(TMP_IW4以降は不要)
        if (prm->nlmeans.sharedMem && i >= 6) {
            m_tmpBuf[i].reset();
            continue;
        }
        const int tmpBufHeight = prm->frameOut.height;
        if (m_tmpBuf[i]
            && (m_tmpBuf[i]->frame.width != tmpBufWidth || m_tmpBuf[i]->frame.height != tmpBufHeight)) {
            m_tmpBuf[i].reset();
        }
        if (!m_tmpBuf[i]) {
            RGYFrameInfo frameInfo = prm->frameOut;
            frameInfo.width = tmpBufWidth;
            frameInfo.height = tmpBufHeight;
            switch (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp]) {
                case RGY_CHROMAFMT_RGB:
                    frameInfo.csp = RGY_CSP_RGB;
                    break;
                case RGY_CHROMAFMT_YUV444:
                    frameInfo.csp = RGY_CSP_YUV444;
                    break;
                case RGY_CHROMAFMT_YUV420:
                    frameInfo.csp = RGY_CSP_YV12;
                    break;
                default:
                    AddMessage(RGY_LOG_ERROR, _T("unsupported csp.\n"));
                    return RGY_ERR_UNSUPPORTED;
            }
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

    // Frame cache for the temporal extension. Capacity = 2*d + 1.
    // Allocated only when d > 0; the d = 0 path stays byte-identical
    // to the existing spatial-only filter (no cache, no output delay).
    // Slot validity is derived from m_inputCount / m_outputCount, not
    // from any per-slot flag — emitFrame computes whether each k-th
    // past/future neighbour is in range.
    const int requiredCacheSize = (prm->nlmeans.d > 0) ? (2 * prm->nlmeans.d + 1) : 0;
    if ((int)m_cacheFrames.size() != requiredCacheSize
        || (requiredCacheSize > 0 && !m_cacheFrames.empty()
            && cmpFrameInfoCspResolution(&m_cacheFrames[0]->frame, &prm->frameIn))) {
        m_cacheFrames.clear();
        for (int i = 0; i < requiredCacheSize; i++) {
            auto clframe = m_cl->createFrameBuffer(prm->frameIn, CL_MEM_READ_WRITE);
            if (!clframe) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate cache frame %d.\n"), i);
                return RGY_ERR_MEMORY_ALLOC;
            }
            m_cacheFrames.push_back(std::move(clframe));
        }
        m_inputCount = 0;
        m_outputCount = 0;
        m_drained = false;
    }

    //コピーを保存
    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterDenoiseNLMeans::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    //if (interlaced(*pInputFrame)) {
    //    return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
    //}
    for (auto& program : m_nlmeans) {
        if (!program.second.get()) {
            AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_DENOISE_NLMEANS_CL(m_nlmeans)\n"));
            return RGY_ERR_OPENCL_CRUSH;
        }
    }
    for (auto& program : m_nlmeansTemporal) {
        if (!program.second.get()) {
            AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_DENOISE_NLMEANS_CL(m_nlmeansTemporal)\n"));
            return RGY_ERR_OPENCL_CRUSH;
        }
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseNLMeans>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const bool hasInput = (pInputFrame && pInputFrame->ptr[0]);

    // d == 0 path: byte-identical to the prior spatial-only filter.
    // No caching, no output delay, no temporal kernel dispatch.
    if (prm->nlmeans.d == 0) {
        if (!hasInput) return RGY_ERR_NONE;
        *pOutputFrameNum = 1;
        if (ppOutputFrames[0] == nullptr) {
            auto pOutFrame = m_frameBuf[0].get();
            ppOutputFrames[0] = &pOutFrame->frame;
        }
        ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
        const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
        if (memcpyKind != RGYCLMemcpyD2D) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        sts = denoiseFrame(ppOutputFrames[0], pInputFrame, std::vector<const RGYFrameInfo *>(), queue, wait_events, event);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at denoiseFrame (%s): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        }
        return sts;
    }

    // d > 0 path: cache + delayed output following the bwdif pattern.
    const int d = prm->nlmeans.d;
    const int cacheSize = (int)m_cacheFrames.size();

    if (hasInput) {
        const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, m_cacheFrames[0]->frame.mem_type);
        if (memcpyKind != RGYCLMemcpyD2D) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        const int slot = m_inputCount % cacheSize;
        RGYFrameInfo *pSlot = &m_cacheFrames[slot]->frame;
        auto copyErr = m_cl->copyFrame(pSlot, pInputFrame, nullptr, queue, wait_events);
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy input to cache slot %d: %s.\n"), slot, get_err_mes(copyErr));
            return copyErr;
        }
        pSlot->timestamp    = pInputFrame->timestamp;
        pSlot->duration     = pInputFrame->duration;
        pSlot->inputFrameId = pInputFrame->inputFrameId;
        pSlot->picstruct    = pInputFrame->picstruct;
        pSlot->flags        = pInputFrame->flags;

        m_inputCount++;
        // Still filling the cache (need d future frames before the
        // centre frame can be emitted).
        if (m_inputCount <= d) {
            return RGY_ERR_NONE;
        }
        const int idx_cur = m_outputCount % cacheSize;
        return emitFrame(idx_cur, ppOutputFrames, pOutputFrameNum, queue, std::vector<RGYOpenCLEvent>(), event);
    }

    // No input + not yet drained: emit one remaining cached frame per call.
    if (m_outputCount < m_inputCount) {
        const int idx_cur = m_outputCount % cacheSize;
        return emitFrame(idx_cur, ppOutputFrames, pOutputFrameNum, queue, wait_events, event);
    }
    m_drained = true;
    return RGY_ERR_NONE;
}

void RGYFilterDenoiseNLMeans::close() {
    m_frameBuf.clear();
    m_nlmeans.clear();
    m_nlmeansTemporal.clear();
    for (auto& f : m_tmpBuf) {
        f.reset();
    }
    m_cacheFrames.clear();
    m_inputCount = 0;
    m_outputCount = 0;
    m_drained = false;
    m_cl.reset();
}
