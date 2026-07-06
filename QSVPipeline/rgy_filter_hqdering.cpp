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
    m_blurred(),
    m_edgeCore(),
    m_maskTmp2(),
    m_maskTmp3(),
    m_contraTmp(),
    m_contraTmp2() {
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
    if (prm->dering.thr < 0 || prm->dering.thr > 255) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid thr=%d: must be in [0, 255].\n"), prm->dering.thr);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dering.elast < 1.0f || prm->dering.elast > 3.0f) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid elast=%.2f: must be in [1.0, 3.0].\n"), prm->dering.elast);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dering.darkthr < -1 || prm->dering.darkthr > 255) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid darkthr=%d: must be in [-1, 255].\n"), prm->dering.darkthr);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dering.minp < 0 || prm->dering.minp > 3) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid minp=%d: must be in [0, 3].\n"), prm->dering.minp);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dering.msmooth < 0 || prm->dering.msmooth > 3) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid msmooth=%d: must be in [0, 3].\n"), prm->dering.msmooth);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dering.drrep < 0 || prm->dering.drrep > 1) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid drrep=%d: must be 0 or 1.\n"), prm->dering.drrep);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dering.sharp < 0 || prm->dering.sharp > 3) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid sharp=%d: must be in [0, 3].\n"), prm->dering.sharp);
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
    // Optional-knob buffers, allocated only when the knob is active so the
    // default path keeps the previous memory footprint.
    if (prm->dering.minp > 0) {
        m_edgeCore = m_cl->createFrameBuffer(prm->frameIn);
        m_maskTmp2 = m_cl->createFrameBuffer(prm->frameIn);
        if (!m_edgeCore || !m_maskTmp2) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate dering minp buffers.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    } else {
        m_edgeCore.reset();
        m_maskTmp2.reset();
    }
    if (prm->dering.msmooth > 0) {
        m_maskTmp3 = m_cl->createFrameBuffer(prm->frameIn);
        if (!m_maskTmp3) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate dering msmooth buffer.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    } else {
        m_maskTmp3.reset();
    }
    if (prm->dering.sharp > 0) {
        m_contraTmp  = m_cl->createFrameBuffer(prm->frameIn);
        m_contraTmp2 = m_cl->createFrameBuffer(prm->frameIn);
        if (!m_contraTmp || !m_contraTmp2) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate dering sharp buffers.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    } else {
        m_contraTmp.reset();
        m_contraTmp2.reset();
    }

    AddMessage(RGY_LOG_DEBUG,
        _T("dering init: src=%dx%d bitDepth=%d, mrad=%d, mthr=%d, sigma=%.2f, showmask=%s, protect=%s, ")
        _T("thr=%d, elast=%.2f, darkthr=%d, minp=%d, msmooth=%d, drrep=%d, sharp=%d, planes=%d%d%d\n"),
        prm->frameIn.width, prm->frameIn.height, bitDepth,
        prm->dering.mrad, prm->dering.mthr, prm->dering.sigma,
        prm->dering.showmask ? _T("on") : _T("off"),
        prm->dering.protect  ? _T("on") : _T("off"),
        prm->dering.thr, prm->dering.elast, prm->dering.darkthr,
        prm->dering.minp, prm->dering.msmooth, prm->dering.drrep, prm->dering.sharp,
        prm->dering.planes[0] ? 1 : 0, prm->dering.planes[1] ? 1 : 0, prm->dering.planes[2] ? 1 : 0);

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterDering::runEdge(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                  int mthrHbd, RGY_PLANE plane,
                                  RGYOpenCLQueue &queue,
                                  const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, plane);
    const auto dP = getPlane(pDst, plane);
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

RGY_ERR RGYFilterDering::runExpand3x3(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, RGY_PLANE plane,
                                       RGYOpenCLQueue &queue,
                                       const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, plane);
    const auto dP = getPlane(pDst, plane);
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
                                   int radius, float sigma, RGY_PLANE plane,
                                   RGYOpenCLQueue &queue,
                                   const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, plane);
    const auto dP = getPlane(pDst, plane);
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
                                   int radius, float sigma, RGY_PLANE plane,
                                   RGYOpenCLQueue &queue,
                                   const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, plane);
    const auto dP = getPlane(pDst, plane);
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

// ---------------------------------------------------------------------------
// The mask/limit extensions below (inpand, 3x3 mean, min/max repair and the
// LimitFilter-style change limit in the combine kernel) follow the reference
// HQDering parameter set. Their building blocks are the same primitives that
// the --vpp-finedehalo rework (#777) introduced in rgy_filter_dehalo.cl /
// rgy_filter_finedehalo.cl (square min/max morphology, RemoveGrain-mode-20
// style 3x3 mean, min/max range clamp). They are implemented here as
// hqdering-local kernels on purpose, so that this patch does not touch the
// freshly reworked dehalo sources - if a shared header for these morphology
// primitives is preferred, they can be unified later.
// ---------------------------------------------------------------------------

//expand3x3の対 (min morph) - #777のsquare morph(min)と同じプリミティブ
RGY_ERR RGYFilterDering::runInpand3x3(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, RGY_PLANE plane,
                                       RGYOpenCLQueue &queue,
                                       const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, plane);
    const auto dP = getPlane(pDst, plane);
    RGYWorkSize local(DERING_BLOCK_X, DERING_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_dering.get()->kernel("hqdering_inpand3x3")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at hqdering_inpand3x3: %s.\n"), get_err_mes(err));
    }
    return err;
}

//3x3平均 (RemoveGrain mode20相当) - msmooth / sharp>=2で使用
RGY_ERR RGYFilterDering::runMean3x3(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, RGY_PLANE plane,
                                     RGYOpenCLQueue &queue,
                                     const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, plane);
    const auto dP = getPlane(pDst, plane);
    RGYWorkSize local(DERING_BLOCK_X, DERING_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_dering.get()->kernel("hqdering_mean3x3")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at hqdering_mean3x3: %s.\n"), get_err_mes(err));
    }
    return err;
}

//RemoveGrain mode11相当: 3x3二項フィルタ - コントラシャープのぼかし段
RGY_ERR RGYFilterDering::runRg11(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, RGY_PLANE plane,
                                  RGYOpenCLQueue &queue,
                                  const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, plane);
    const auto dP = getPlane(pDst, plane);
    RGYWorkSize local(DERING_BLOCK_X, DERING_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_dering.get()->kernel("hqdering_rg11")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at hqdering_rg11: %s.\n"), get_err_mes(err));
    }
    return err;
}

//コントラシャープの合成段: ブラーで失われた線をsrcに実在した変化の範囲でのみ戻す
RGY_ERR RGYFilterDering::runContra(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                    const RGYFrameInfo *pSmoothed, const RGYFrameInfo *pMethod, RGY_PLANE plane,
                                    RGYOpenCLQueue &queue,
                                    const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP  = getPlane(pSrc,      plane);
    const auto smP = getPlane(pSmoothed, plane);
    const auto mtP = getPlane(pMethod,   plane);
    const auto dP  = getPlane(pDst,      plane);
    RGYWorkSize local(DERING_BLOCK_X, DERING_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_dering.get()->kernel("hqdering_contra")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0],  sP.pitch[0],
            (cl_mem)smP.ptr[0], smP.pitch[0],
            (cl_mem)mtP.ptr[0], mtP.pitch[0],
            (cl_mem)dP.ptr[0],  dP.pitch[0],
            sP.width, sP.height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at hqdering_contra: %s.\n"), get_err_mes(err));
    }
    return err;
}

//repair mode1相当: ブラー結果をsrcの3x3 min/maxへクランプ
RGY_ERR RGYFilterDering::runRepair3x3(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                       const RGYFrameInfo *pBlurred, RGY_PLANE plane,
                                       RGYOpenCLQueue &queue,
                                       const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc,     plane);
    const auto bP = getPlane(pBlurred, plane);
    const auto dP = getPlane(pDst,     plane);
    RGYWorkSize local(DERING_BLOCK_X, DERING_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_dering.get()->kernel("hqdering_repair3x3")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)bP.ptr[0], bP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at hqdering_repair3x3: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDering::runCombine(RGYFrameInfo *pDst,
                                     const RGYFrameInfo *pSrc, const RGYFrameInfo *pBlurred, const RGYFrameInfo *pMask,
                                     const RGYFrameInfo *pEdgeMask, const RGYFrameInfo *pCoreMask,
                                     int showmask, int protect,
                                     int thrHbd, int darkthrHbd, float elast, RGY_PLANE plane,
                                     RGYOpenCLQueue &queue,
                                     const std::vector<RGYOpenCLEvent> &wait_events,
                                     RGYOpenCLEvent *event) {
    const auto sP = getPlane(pSrc,      plane);
    const auto bP = getPlane(pBlurred,  plane);
    const auto mP = getPlane(pMask,     plane);
    const auto eP = getPlane(pEdgeMask, plane);
    const auto dP = getPlane(pDst,      plane);
    // OpenCLではCUDAのようなnullポインタ判定が使えないため、minp=0のときは
    // edgeMaskをダミーとして渡し、useCoreMask=0で参照を止める
    const auto cP = getPlane((pCoreMask) ? pCoreMask : pEdgeMask, plane);
    const int useCoreMask = (pCoreMask) ? 1 : 0;
    RGYWorkSize local(DERING_BLOCK_X, DERING_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_dering.get()->kernel("hqdering_combine")
        .config(queue, local, global, wait_events, event).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)bP.ptr[0], bP.pitch[0],
            (cl_mem)mP.ptr[0], mP.pitch[0],
            (cl_mem)eP.ptr[0], eP.pitch[0],
            (cl_mem)cP.ptr[0], cP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            showmask, protect, useCoreMask,
            thrHbd, darkthrHbd, elast);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at hqdering_combine: %s.\n"), get_err_mes(err));
    }
    return err;
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
    const int thrHbd   = (int)((long long)prm->dering.thr  * maxVal / 255);
    const int darkthrHbd = (prm->dering.darkthr >= 0) ? (int)((long long)prm->dering.darkthr * maxVal / 255) : -1;

    // Gaussian kernel radius: ceil(2 sigma), clamped to the compile-time max.
    int kernelRadius = (int)std::ceil(2.0f * prm->dering.sigma);
    if (kernelRadius < 1) kernelRadius = 1;
    if (kernelRadius > DERING_KERNEL_RADIUS_MAX) kernelRadius = DERING_KERNEL_RADIUS_MAX;

    RGYFrameInfo *pOut = &m_frameBuf[0]->frame;
    const int planes = RGY_CSP_PLANES[pOut->csp];
    const int showmask = prm->dering.showmask ? 1 : 0;
    const int protect  = prm->dering.protect  ? 1 : 0;

    //処理バッファは入力cspのフル形式で確保しているため、getPlaneで各プレーンのビューを
    //取れば同じ処理チェーンがそのまま色差プレーンにも使える (planes指定, default=yのみ)
    auto waitFirst = wait_events;
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto plane = (RGY_PLANE)iplane;
        RGYOpenCLEvent *planeEvent = (iplane == planes - 1) ? event : nullptr;
        const bool process = (iplane < (int)prm->dering.planes.size()) && prm->dering.planes[iplane];
        if (!process) {
            // Disabled planes pass through from source.
            const auto srcP = getPlane(pInputFrame, plane);
            auto dstP = getPlane(pOut, plane);
            auto err = m_cl->copyPlane(&dstP, &srcP, nullptr, queue_main, waitFirst, planeEvent);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("dering plane copy (plane %d) failed: %s.\n"),
                    iplane, get_err_mes(err));
                return err;
            }
            waitFirst.clear();
            continue;
        }

        // ---- 1. edge operator + threshold → m_edgeMask ----
        {
            auto err = runEdge(&m_edgeMask->frame, pInputFrame, mthrHbd, plane, queue_main, waitFirst);
            if (err != RGY_ERR_NONE) return err;
            waitFirst.clear();
        }

        // ---- 2. mrad-pass dilation, ping-pong between m_ringMask and m_morphTmp ----
        // First iteration reads from m_edgeMask. Subsequent iterations alternate.
        // After the loop, pRingMask points to the buffer holding the final result.
        RGYFrameInfo *pRingMask = nullptr;
        {
            const RGYFrameInfo *pIn = &m_edgeMask->frame;
            for (int i = 0; i < prm->dering.mrad; i++) {
                RGYFrameInfo *pOutMorph = ((i & 1) == 0) ? &m_ringMask->frame : &m_morphTmp->frame;
                auto err = runExpand3x3(pOutMorph, pIn, plane, queue_main, {});
                if (err != RGY_ERR_NONE) return err;
                pIn = pOutMorph;
            }
            // pIn now holds the final ring mask.
            pRingMask = const_cast<RGYFrameInfo *>(pIn);
        }

        //minp: エッジマスクをinpandして「エッジ芯」を作る (combineでリングマスクから除外)
        const RGYFrameInfo *pCoreMask = nullptr;
        if (prm->dering.minp > 0 && m_edgeCore && m_maskTmp2) {
            const RGYFrameInfo *pInM = &m_edgeMask->frame;
            for (int i = 0; i < prm->dering.minp; i++) {
                RGYFrameInfo *pOutM = ((i & 1) == 0) ? &m_edgeCore->frame : &m_maskTmp2->frame;
                auto err = runInpand3x3(pOutM, pInM, plane, queue_main, {});
                if (err != RGY_ERR_NONE) return err;
                pInM = pOutM;
            }
            pCoreMask = pInM;
        }

        //msmooth: リングマスクの境界を3x3平均でなじませて処理境界の段差を防ぐ
        if (prm->dering.msmooth > 0 && m_maskTmp3) {
            //pRingMaskはringMask/morphTmpのどちらかで終わるため、もう一方を空きバッファとして使う
            RGYFrameInfo *pSmoothA = &m_maskTmp3->frame;
            RGYFrameInfo *pSmoothB = (pRingMask == &m_ringMask->frame) ? &m_morphTmp->frame : &m_ringMask->frame;
            const RGYFrameInfo *pInS = pRingMask;
            for (int i = 0; i < prm->dering.msmooth; i++) {
                RGYFrameInfo *pOutS = ((i & 1) == 0) ? pSmoothA : pSmoothB;
                auto err = runMean3x3(pOutS, pInS, plane, queue_main, {});
                if (err != RGY_ERR_NONE) return err;
                pInS = pOutS;
            }
            pRingMask = const_cast<RGYFrameInfo *>(pInS);
        }

        // ---- 3+4. Separable Gaussian: src → m_hBlurred → m_blurred ----
        {
            auto err = runBlurH(&m_hBlurred->frame, pInputFrame,
                                kernelRadius, prm->dering.sigma, plane, queue_main, {});
            if (err != RGY_ERR_NONE) return err;
            err = runBlurV(&m_blurred->frame, &m_hBlurred->frame,
                           kernelRadius, prm->dering.sigma, plane, queue_main, {});
            if (err != RGY_ERR_NONE) return err;
        }

        //sharp: コントラシャープ (リファレンスの順序どおり drrep/thr制限の前に適用)
        //  method = RG11(smoothed) [sharp>=2: ->RG20, sharp>=3: ->RG20] (RG20=mean3x3)
        const RGYFrameInfo *pSmoothed = &m_blurred->frame;
        if (prm->dering.sharp > 0 && m_contraTmp && m_contraTmp2) {
            auto err = runRg11(&m_contraTmp->frame, pSmoothed, plane, queue_main, {});
            if (err != RGY_ERR_NONE) return err;
            const RGYFrameInfo *pMethod = &m_contraTmp->frame;
            for (int i = 1; i < prm->dering.sharp; i++) {
                RGYFrameInfo *pOutM = ((i & 1) != 0) ? &m_contraTmp2->frame : &m_contraTmp->frame;
                err = runMean3x3(pOutM, pMethod, plane, queue_main, {});
                if (err != RGY_ERR_NONE) return err;
                pMethod = pOutM;
            }
            RGYFrameInfo *pContraOut = (pMethod == &m_contraTmp->frame) ? &m_contraTmp2->frame : &m_contraTmp->frame;
            err = runContra(pContraOut, pInputFrame, pSmoothed, pMethod, plane, queue_main, {});
            if (err != RGY_ERR_NONE) return err;
            pSmoothed = pContraOut;
        }

        //drrep: ブラー結果が元画像の3x3 min/max範囲の外の値を作らないよう修復してから合成する
        const RGYFrameInfo *pBlurUse = pSmoothed;
        if (prm->dering.drrep > 0) {
            //hBlurredはblur_v完了後は空きバッファとして再利用できる
            auto err = runRepair3x3(&m_hBlurred->frame, pInputFrame, pSmoothed, plane, queue_main, {});
            if (err != RGY_ERR_NONE) return err;
            pBlurUse = &m_hBlurred->frame;
        }

        // ---- 5. Fused mask + alpha-blend (or showmask write-through) ----
        // m_edgeMask is the pre-dilation edge mask from step 1; combine
        // optionally subtracts it from the dilated ring mask when protect=1,
        // subtracts the minp edge core when set, and applies the optional
        // thr/darkthr/elast LimitFilter ramp (thrHbd=0 keeps the plain blend).
        {
            auto err = runCombine(pOut, pInputFrame, pBlurUse, pRingMask,
                                  &m_edgeMask->frame, pCoreMask,
                                  showmask, protect, thrHbd, darkthrHbd, prm->dering.elast,
                                  plane, queue_main, {}, planeEvent);
            if (err != RGY_ERR_NONE) return err;
        }
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
    m_edgeCore.reset();
    m_maskTmp2.reset();
    m_maskTmp3.reset();
    m_contraTmp.reset();
    m_contraTmp2.reset();
    m_frameBuf.clear();
    m_cl.reset();
}
