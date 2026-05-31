// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
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

#include <algorithm>
#include <cmath>
#include "convert_csp.h"
#include "rgy_filter_hqdering.h"

static const int DERING_BLOCK_X = 32;
static const int DERING_BLOCK_Y = 8;

// Compile-time upper bound on the Gaussian radius. sigma=5.0 → radius=10
// (kernel width 21). The kernel uses this as a static loop bound; the
// runtime `radius` arg short-circuits the actual work for smaller sigma.
static const int DERING_KERNEL_RADIUS_MAX = 10;

RGYFilterDering::RGYFilterDering(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_dering(),
    m_dehaloMorph(),
    m_buildOptions(),
    m_edgeMask(),
    m_ringMask(),
    m_morphTmp(),
    m_hBlurred(),
    m_blurred() {
    m_name = _T("dering");
}

RGYFilterDering::~RGYFilterDering() {
    close();
}

RGY_ERR RGYFilterDering::checkParam(const std::shared_ptr<RGYFilterParamDering> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height < 4 || prm->frameOut.width < 4) {
        AddMessage(RGY_LOG_ERROR, _T("dering requires input width/height >= 4 (got %dx%d).\n"),
            prm->frameOut.width, prm->frameOut.height);
        return RGY_ERR_INVALID_PARAM;
    }
    const auto csp = prm->frameIn.csp;
    const auto chromaFormat = RGY_CSP_CHROMA_FORMAT[csp];
    const auto dataType = RGY_CSP_DATA_TYPE[csp];
    if (dataType != RGY_DATA_TYPE_U8 && dataType != RGY_DATA_TYPE_U16) {
        AddMessage(RGY_LOG_ERROR, _T("dering requires 8-16bit integer input.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (chromaFormat != RGY_CHROMAFMT_YUV420
        && chromaFormat != RGY_CHROMAFMT_YUV422
        && chromaFormat != RGY_CHROMAFMT_YUV444
        && chromaFormat != RGY_CHROMAFMT_MONOCHROME) {
        AddMessage(RGY_LOG_ERROR, _T("dering requires YUV or monochrome input.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (RGY_CSP_PLANES[csp] <= 1 && chromaFormat != RGY_CHROMAFMT_MONOCHROME) {
        AddMessage(RGY_LOG_ERROR, _T("dering does not support packed YUV input.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dering.mrad < 1 || prm->dering.mrad > 3) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid mrad=%d: must be in [1, 3].\n"), prm->dering.mrad);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dering.mthr < 0 || prm->dering.mthr > 255) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid mthr=%d: must be in [0, 255].\n"), prm->dering.mthr);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dering.sigma >= 0.5f && prm->dering.sigma <= 5.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid sigma=%.2f: must be in [0.5, 5.0].\n"), prm->dering.sigma);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dering.edge != _T("log")
        && prm->dering.edge != _T("sobel")
        && prm->dering.edge != _T("prewitt")
        && prm->dering.edge != _T("scharr")
        && prm->dering.edge != _T("kirsch")
        && prm->dering.edge != _T("laplacian")) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid edge=%s: must be log|sobel|prewitt|scharr|kirsch|laplacian.\n"),
            prm->dering.edge.c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDering::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDering>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    // Resolve edge operator → kernel name. edge=log is the new default
    // (5x5 LoG, better ring-detection in flat regions). edge=sobel
    // preserves the original hqdering_edge kernel.
    if      (prm->dering.edge == _T("log"))       m_edgeKernelName = "hqdering_edge_log";
    else if (prm->dering.edge == _T("sobel"))     m_edgeKernelName = "hqdering_edge";
    else if (prm->dering.edge == _T("prewitt"))   m_edgeKernelName = "hqdering_edge_prewitt";
    else if (prm->dering.edge == _T("scharr"))    m_edgeKernelName = "hqdering_edge_scharr";
    else if (prm->dering.edge == _T("kirsch"))    m_edgeKernelName = "hqdering_edge_kirsch";
    else if (prm->dering.edge == _T("laplacian")) m_edgeKernelName = "hqdering_edge_laplacian";

    prm->frameOut.picstruct = prm->frameIn.picstruct;

    const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    const int maxVal   = (1 << bitDepth) - 1;

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamDering>(m_param);
    if (!prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        m_buildOptions = strsprintf("-D Type=%s -D bit_depth=%d -D max_val=%d -D dering_block_x=%d -D dering_block_y=%d -D DERING_KERNEL_RADIUS_MAX=%d",
            bitDepth > 8 ? "ushort" : "uchar",
            bitDepth, maxVal,
            DERING_BLOCK_X, DERING_BLOCK_Y,
            DERING_KERNEL_RADIUS_MAX);
        AddMessage(RGY_LOG_DEBUG, _T("Starting async build for RGY_FILTER_HQDERING_CL: %s\n"),
            char_to_tstring(m_buildOptions).c_str());
        m_dering.set(m_cl->buildResourceAsync(_T("RGY_FILTER_HQDERING_CL"), _T("EXE_DATA"), m_buildOptions.c_str()));

        // Private rebuild of dehalo.cl so we can call dehalo_expand at
        // fixed radius 1 without a cross-filter program dependency.
        const std::string dehaloOpts = strsprintf("-D Type=%s -D bit_depth=%d -D max_val=%d -D dehalo_block_x=%d -D dehalo_block_y=%d",
            bitDepth > 8 ? "ushort" : "uchar",
            bitDepth, maxVal,
            DERING_BLOCK_X, DERING_BLOCK_Y);
        m_dehaloMorph.set(m_cl->buildResourceAsync(_T("RGY_FILTER_DEHALO_CL"), _T("EXE_DATA"), dehaloOpts.c_str()));
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output buffer: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    m_edgeMask = m_cl->createFrameBuffer(prm->frameIn);
    m_ringMask = m_cl->createFrameBuffer(prm->frameIn);
    m_morphTmp = m_cl->createFrameBuffer(prm->frameIn);
    m_hBlurred = m_cl->createFrameBuffer(prm->frameIn);
    m_blurred  = m_cl->createFrameBuffer(prm->frameIn);
    if (!m_edgeMask || !m_ringMask || !m_morphTmp || !m_hBlurred || !m_blurred) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate dering intermediate buffers.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }

    AddMessage(RGY_LOG_DEBUG,
        _T("dering init: src=%dx%d bitDepth=%d, mrad=%d, mthr=%d, sigma=%.2f, showmask=%s, protect=%s\n"),
        prm->frameIn.width, prm->frameIn.height, bitDepth,
        prm->dering.mrad, prm->dering.mthr, prm->dering.sigma,
        prm->dering.showmask ? _T("on") : _T("off"),
        prm->dering.protect  ? _T("on") : _T("off"));

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterDering::runEdge(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                  int mthrHbd,
                                  RGYOpenCLQueue &queue,
                                  const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, RGY_PLANE_Y);
    RGYWorkSize local(DERING_BLOCK_X, DERING_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_dering.get()->kernel(m_edgeKernelName.c_str())
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            mthrHbd);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"),
            char_to_tstring(m_edgeKernelName).c_str(), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDering::runExpand3x3(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                       RGYOpenCLQueue &queue,
                                       const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, RGY_PLANE_Y);
    RGYWorkSize local(DERING_BLOCK_X, DERING_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    const float r = 1.0f;
    auto err = m_dehaloMorph.get()->kernel("dehalo_expand")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            r, r);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at dehalo_expand (3x3): %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDering::runBlurH(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                   int radius, float sigma,
                                   RGYOpenCLQueue &queue,
                                   const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, RGY_PLANE_Y);
    RGYWorkSize local(DERING_BLOCK_X, DERING_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_dering.get()->kernel("hqdering_blur_h")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            radius, sigma);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at hqdering_blur_h: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDering::runBlurV(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                   int radius, float sigma,
                                   RGYOpenCLQueue &queue,
                                   const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, RGY_PLANE_Y);
    // Vertical 1-D Gaussian: each work-item reads (2*radius + 1) rows
    // at its column. An 8x32 work-group (tall vs the default 32x8)
    // keeps the per-column row reads inside one cache tile so the
    // 32 work-items in a column share the radius-window's row reads
    // through the L1 cache instead of pulling from L2 / DRAM.
    RGYWorkSize local(DERING_BLOCK_Y, DERING_BLOCK_X);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_dering.get()->kernel("hqdering_blur_v")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            radius, sigma);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at hqdering_blur_v: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDering::runCombine(RGYFrameInfo *pDst,
                                     const RGYFrameInfo *pSrc, const RGYFrameInfo *pBlurred, const RGYFrameInfo *pMask,
                                     const RGYFrameInfo *pEdgeMask,
                                     int showmask, int protect,
                                     RGYOpenCLQueue &queue,
                                     const std::vector<RGYOpenCLEvent> &wait_events,
                                     RGYOpenCLEvent *event) {
    const auto sP = getPlane(pSrc,      RGY_PLANE_Y);
    const auto bP = getPlane(pBlurred,  RGY_PLANE_Y);
    const auto mP = getPlane(pMask,     RGY_PLANE_Y);
    const auto eP = getPlane(pEdgeMask, RGY_PLANE_Y);
    const auto dP = getPlane(pDst,      RGY_PLANE_Y);
    RGYWorkSize local(DERING_BLOCK_X, DERING_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_dering.get()->kernel("hqdering_combine")
        .config(queue, local, global, wait_events, event).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)bP.ptr[0], bP.pitch[0],
            (cl_mem)mP.ptr[0], mP.pitch[0],
            (cl_mem)eP.ptr[0], eP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            showmask, protect);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at hqdering_combine: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDering::copyChromaPlanes(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                           RGYOpenCLQueue &queue,
                                           const std::vector<RGYOpenCLEvent> &wait_events,
                                           RGYOpenCLEvent *event) {
    const int planes = RGY_CSP_PLANES[pDst->csp];
    if (planes <= 1) {
        return RGY_ERR_NONE;
    }
    auto waitFirst = wait_events;
    for (int i = 1; i < planes; i++) {
        const auto pl = (RGY_PLANE)i;
        const auto srcP = getPlane(pSrc, pl);
        const auto dstP = getPlane(pDst, pl);
        auto err = m_cl->copyPlane(const_cast<RGYFrameInfo *>(&dstP), &srcP, nullptr, queue, waitFirst, (i == planes - 1) ? event : nullptr);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("dering chroma copy (plane %d) failed: %s.\n"),
                i, get_err_mes(err));
            return err;
        }
        waitFirst.clear();
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDering::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue_main,
    const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    *pOutputFrameNum  = 0;
    ppOutputFrames[0] = nullptr;

    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (!m_dering.get()) {
        AddMessage(RGY_LOG_ERROR, _T("dering OpenCL program failed to build (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }
    if (!m_dehaloMorph.get()) {
        AddMessage(RGY_LOG_ERROR, _T("dering dehalo-morph OpenCL program failed to build.\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamDering>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const int bitDepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
    const int maxVal   = (1 << bitDepth) - 1;
    const int mthrHbd  = (int)((long long)prm->dering.mthr * maxVal / 255);

    // Gaussian kernel radius: ceil(2 sigma), clamped to the compile-time max.
    int kernelRadius = (int)std::ceil(2.0f * prm->dering.sigma);
    if (kernelRadius < 1) kernelRadius = 1;
    if (kernelRadius > DERING_KERNEL_RADIUS_MAX) kernelRadius = DERING_KERNEL_RADIUS_MAX;

    RGYFrameInfo *pOut = &m_frameBuf[0]->frame;
    const int planes = RGY_CSP_PLANES[pOut->csp];
    const bool hasChroma = planes > 1;

    // ---- 1. Sobel + threshold → m_edgeMask ----
    {
        auto err = runEdge(&m_edgeMask->frame, pInputFrame, mthrHbd, queue_main, wait_events);
        if (err != RGY_ERR_NONE) return err;
    }

    // ---- 2. mrad-pass dilation, ping-pong between m_ringMask and m_morphTmp ----
    // First iteration reads from m_edgeMask. Subsequent iterations alternate.
    // After the loop, pRingMask points to the buffer holding the final result.
    RGYFrameInfo *pRingMask = nullptr;
    {
        const RGYFrameInfo *pIn = &m_edgeMask->frame;
        for (int i = 0; i < prm->dering.mrad; i++) {
            RGYFrameInfo *pOutMorph = ((i & 1) == 0) ? &m_ringMask->frame : &m_morphTmp->frame;
            auto err = runExpand3x3(pOutMorph, pIn, queue_main, {});
            if (err != RGY_ERR_NONE) return err;
            pIn = pOutMorph;
        }
        // pIn now holds the final ring mask.
        pRingMask = const_cast<RGYFrameInfo *>(pIn);
    }

    // ---- 3+4. Separable Gaussian: src → m_hBlurred → m_blurred ----
    {
        auto err = runBlurH(&m_hBlurred->frame, pInputFrame,
                            kernelRadius, prm->dering.sigma, queue_main, {});
        if (err != RGY_ERR_NONE) return err;
        err = runBlurV(&m_blurred->frame, &m_hBlurred->frame,
                       kernelRadius, prm->dering.sigma, queue_main, {});
        if (err != RGY_ERR_NONE) return err;
    }

    // ---- 5. Fused mask + alpha-blend (or showmask write-through) ----
    // m_edgeMask is the pre-dilation Sobel mask from step 1; combine
    // optionally subtracts it from the dilated ring mask when protect=1.
    {
        const int showmask = prm->dering.showmask ? 1 : 0;
        const int protect  = prm->dering.protect  ? 1 : 0;
        auto err = runCombine(pOut, pInputFrame, &m_blurred->frame, pRingMask,
                              &m_edgeMask->frame,
                              showmask, protect, queue_main, {}, hasChroma ? nullptr : event);
        if (err != RGY_ERR_NONE) return err;
    }

    // Chroma planes pass through from source.
    if (hasChroma) {
        auto err = copyChromaPlanes(pOut, pInputFrame, queue_main, {}, event);
        if (err != RGY_ERR_NONE) return err;
    }

    pOut->timestamp    = pInputFrame->timestamp;
    pOut->duration     = pInputFrame->duration;
    pOut->inputFrameId = pInputFrame->inputFrameId;
    pOut->picstruct    = pInputFrame->picstruct;
    pOut->flags        = pInputFrame->flags;

    ppOutputFrames[0] = pOut;
    *pOutputFrameNum  = 1;
    return RGY_ERR_NONE;
}

void RGYFilterDering::close() {
    m_dering.clear();
    m_dehaloMorph.clear();
    m_buildOptions.clear();
    m_edgeMask.reset();
    m_ringMask.reset();
    m_morphTmp.reset();
    m_hBlurred.reset();
    m_blurred.reset();
    m_frameBuf.clear();
    m_cl.reset();
}
