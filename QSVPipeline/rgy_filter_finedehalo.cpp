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
#include "convert_csp.h"
#include "rgy_filter_finedehalo.h"

static const int FINEDEHALO_BLOCK_X = 32;
static const int FINEDEHALO_BLOCK_Y = 8;

RGYFilterFineDehalo::RGYFilterFineDehalo(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_finedehalo(),
    m_dehaloMorph(),
    m_buildOptions(),
    m_dehalo(),
    m_edges(),
    m_morphTmp(),
    m_ey(),
    m_em(),
    m_linemask() {
    m_name = _T("finedehalo");
}

RGYFilterFineDehalo::~RGYFilterFineDehalo() {
    close();
}

RGY_ERR RGYFilterFineDehalo::checkParam(const std::shared_ptr<RGYFilterParamFineDehalo> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height < 4 || prm->frameOut.width < 4) {
        AddMessage(RGY_LOG_ERROR, _T("finedehalo requires input width/height >= 4 (got %dx%d).\n"),
            prm->frameOut.width, prm->frameOut.height);
        return RGY_ERR_INVALID_PARAM;
    }
    const auto csp = prm->frameIn.csp;
    const auto chromaFormat = RGY_CSP_CHROMA_FORMAT[csp];
    const auto dataType = RGY_CSP_DATA_TYPE[csp];
    if (dataType != RGY_DATA_TYPE_U8 && dataType != RGY_DATA_TYPE_U16) {
        AddMessage(RGY_LOG_ERROR, _T("finedehalo requires 8-16bit integer input.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (chromaFormat != RGY_CHROMAFMT_YUV420
        && chromaFormat != RGY_CHROMAFMT_YUV422
        && chromaFormat != RGY_CHROMAFMT_YUV444
        && chromaFormat != RGY_CHROMAFMT_MONOCHROME) {
        AddMessage(RGY_LOG_ERROR, _T("finedehalo requires YUV or monochrome input.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (RGY_CSP_PLANES[csp] <= 1 && chromaFormat != RGY_CHROMAFMT_MONOCHROME) {
        AddMessage(RGY_LOG_ERROR, _T("finedehalo does not support packed YUV input.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto &p = prm->finedehalo;
    if (!(p.rx >= 0.5f && p.rx <= 10.0f) || !(p.ry >= 0.5f && p.ry <= 10.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid rx=%.2f or ry=%.2f: must be in [0.5, 10.0].\n"), p.rx, p.ry);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(p.darkstr >= 0.0f && p.darkstr <= 1.0f) || !(p.brightstr >= 0.0f && p.brightstr <= 1.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid darkstr=%.2f or brightstr=%.2f: must be in [0.0, 1.0].\n"),
            p.darkstr, p.brightstr);
        return RGY_ERR_INVALID_PARAM;
    }
    if (p.lowsens < 0 || p.lowsens > 100 || p.highsens < 0 || p.highsens > 100) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid lowsens=%d or highsens=%d: must be in [0, 100].\n"),
            p.lowsens, p.highsens);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(p.ss >= 1.0f && p.ss <= 4.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid ss=%.2f: must be in [1.0, 4.0].\n"), p.ss);
        return RGY_ERR_INVALID_PARAM;
    }
    if (p.thmi < 0 || p.thmi > 255 || p.thma < 0 || p.thma > 255) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid thmi=%d or thma=%d: must be in [0, 255].\n"), p.thmi, p.thma);
        return RGY_ERR_INVALID_PARAM;
    }
    if (p.thlimi < 0 || p.thlimi > 255 || p.thlima < 0 || p.thlima > 255) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid thlimi=%d or thlima=%d: must be in [0, 255].\n"),
            p.thlimi, p.thlima);
        return RGY_ERR_INVALID_PARAM;
    }
    if (p.showmask < 0 || p.showmask > 4) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid showmask=%d: must be in [0, 4].\n"), p.showmask);
        return RGY_ERR_INVALID_PARAM;
    }
    if (p.edge != _T("prewitt")
        && p.edge != _T("sobel")
        && p.edge != _T("scharr")
        && p.edge != _T("kirsch")
        && p.edge != _T("laplacian")) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid edge=%s: must be prewitt|sobel|scharr|kirsch|laplacian.\n"),
            p.edge.c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterFineDehalo::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamFineDehalo>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    // Resolve edge operator → kernel name. edge=sobel aliases prewitt
    // because the existing finedehalo_prewitt kernel is already
    // centre-weighted (textbook Sobel math); preserving the prewitt name
    // for backward compatibility with the AviSynth FineDehalo script.
    if      (prm->finedehalo.edge == _T("prewitt"))   m_edgeKernelName = "finedehalo_prewitt";
    else if (prm->finedehalo.edge == _T("sobel"))     m_edgeKernelName = "finedehalo_prewitt";
    else if (prm->finedehalo.edge == _T("scharr"))    m_edgeKernelName = "finedehalo_scharr";
    else if (prm->finedehalo.edge == _T("kirsch"))    m_edgeKernelName = "finedehalo_kirsch";
    else if (prm->finedehalo.edge == _T("laplacian")) m_edgeKernelName = "finedehalo_laplacian";

    prm->frameOut.picstruct = prm->frameIn.picstruct;

    const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    const int maxVal   = (1 << bitDepth) - 1;

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamFineDehalo>(m_param);
    if (!prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        m_buildOptions = strsprintf("-D Type=%s -D bit_depth=%d -D max_val=%d -D finedehalo_block_x=%d -D finedehalo_block_y=%d",
            bitDepth > 8 ? "ushort" : "uchar",
            bitDepth, maxVal,
            FINEDEHALO_BLOCK_X, FINEDEHALO_BLOCK_Y);
        AddMessage(RGY_LOG_DEBUG, _T("Starting async build for RGY_FILTER_FINEDEHALO_CL: %s\n"),
            char_to_tstring(m_buildOptions).c_str());
        m_finedehalo.set(m_cl->buildResourceAsync(_T("RGY_FILTER_FINEDEHALO_CL"), _T("EXE_DATA"), m_buildOptions.c_str()));

        // Re-build dehalo.cl as a private program so we can call its
        // expand/inpand kernels at fixed radius 1 without reaching into
        // the dehalo sub-filter's program.
        const std::string dehaloOpts = strsprintf("-D Type=%s -D bit_depth=%d -D max_val=%d -D dehalo_block_x=%d -D dehalo_block_y=%d",
            bitDepth > 8 ? "ushort" : "uchar",
            bitDepth, maxVal,
            FINEDEHALO_BLOCK_X, FINEDEHALO_BLOCK_Y);
        m_dehaloMorph.set(m_cl->buildResourceAsync(_T("RGY_FILTER_DEHALO_CL"), _T("EXE_DATA"), dehaloOpts.c_str()));
    }

    // Output buffer at source resolution.
    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output buffer: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // Instantiate the DeHalo_alpha sub-filter. Its output lives in the
    // sub-filter's own m_frameBuf[0] and we read it back via outArr[0]
    // after each call (bOutOverwrite=false; see filter() semantics in
    // rgy_filter_cl.cpp).
    {
        auto prmDh = std::make_shared<RGYFilterParamDehalo>();
        prmDh->dehalo.enable    = true;
        prmDh->dehalo.rx        = prm->finedehalo.rx;
        prmDh->dehalo.ry        = prm->finedehalo.ry;
        prmDh->dehalo.darkstr   = prm->finedehalo.darkstr;
        prmDh->dehalo.brightstr = prm->finedehalo.brightstr;
        prmDh->dehalo.lowsens   = prm->finedehalo.lowsens;
        prmDh->dehalo.highsens  = prm->finedehalo.highsens;
        prmDh->dehalo.ss        = prm->finedehalo.ss;
        prmDh->frameIn          = prm->frameIn;
        prmDh->frameOut         = prm->frameIn;
        prmDh->baseFps          = prm->baseFps;
        prmDh->bOutOverwrite    = false;
        m_dehalo = std::make_unique<RGYFilterDehalo>(m_cl);
        sts = m_dehalo->init(prmDh, m_pLog);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to init dehalo sub-filter: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    // All intermediates at source resolution.
    m_edges    = m_cl->createFrameBuffer(prm->frameIn);
    m_morphTmp = m_cl->createFrameBuffer(prm->frameIn);
    m_ey       = m_cl->createFrameBuffer(prm->frameIn);
    m_em       = m_cl->createFrameBuffer(prm->frameIn);
    m_linemask = m_cl->createFrameBuffer(prm->frameIn);
    if (!m_edges || !m_morphTmp || !m_ey || !m_em || !m_linemask) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate finedehalo intermediate buffers.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }

    AddMessage(RGY_LOG_DEBUG,
        _T("finedehalo init: src=%dx%d bitDepth=%d, rx=%.2f, ry=%.2f, thmi=%d, thma=%d, thlimi=%d, thlima=%d, darkstr=%.2f, brightstr=%.2f, lowsens=%d, highsens=%d, ss=%.2f, showmask=%d\n"),
        prm->frameIn.width, prm->frameIn.height, bitDepth,
        prm->finedehalo.rx, prm->finedehalo.ry,
        prm->finedehalo.thmi, prm->finedehalo.thma,
        prm->finedehalo.thlimi, prm->finedehalo.thlima,
        prm->finedehalo.darkstr, prm->finedehalo.brightstr,
        prm->finedehalo.lowsens, prm->finedehalo.highsens,
        prm->finedehalo.ss, prm->finedehalo.showmask);

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterFineDehalo::runPrewitt(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                         int thmiHbd, int thmaHbd,
                                         RGYOpenCLQueue &queue,
                                         const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, RGY_PLANE_Y);
    RGYWorkSize local(FINEDEHALO_BLOCK_X, FINEDEHALO_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_finedehalo.get()->kernel(m_edgeKernelName.c_str())
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            thmiHbd, thmaHbd);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"),
            char_to_tstring(m_edgeKernelName).c_str(), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterFineDehalo::runLimitMask(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc, const RGYFrameInfo *pDehaloed,
                                           int thlimiHbd, int thlimaHbd,
                                           RGYOpenCLQueue &queue,
                                           const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc,      RGY_PLANE_Y);
    const auto dhP = getPlane(pDehaloed, RGY_PLANE_Y);
    const auto dP = getPlane(pDst,      RGY_PLANE_Y);
    RGYWorkSize local(FINEDEHALO_BLOCK_X, FINEDEHALO_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_finedehalo.get()->kernel("finedehalo_limitmask")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0],  sP.pitch[0],
            (cl_mem)dhP.ptr[0], dhP.pitch[0],
            (cl_mem)dP.ptr[0],  dP.pitch[0],
            sP.width, sP.height,
            thlimiHbd, thlimaHbd);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at finedehalo_limitmask: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterFineDehalo::runCombine(RGYFrameInfo *pDst,
                                         const RGYFrameInfo *pSrc, const RGYFrameInfo *pDehaloed,
                                         const RGYFrameInfo *pEm,  const RGYFrameInfo *pLineMask,
                                         int showmask,
                                         RGYOpenCLQueue &queue,
                                         const std::vector<RGYOpenCLEvent> &wait_events,
                                         RGYOpenCLEvent *event) {
    const auto sP  = getPlane(pSrc,      RGY_PLANE_Y);
    const auto dhP = getPlane(pDehaloed, RGY_PLANE_Y);
    const auto eP  = getPlane(pEm,       RGY_PLANE_Y);
    const auto lP  = getPlane(pLineMask, RGY_PLANE_Y);
    const auto dP  = getPlane(pDst,      RGY_PLANE_Y);
    RGYWorkSize local(FINEDEHALO_BLOCK_X, FINEDEHALO_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_finedehalo.get()->kernel("finedehalo_combine")
        .config(queue, local, global, wait_events, event).launch(
            (cl_mem)sP.ptr[0],  sP.pitch[0],
            (cl_mem)dhP.ptr[0], dhP.pitch[0],
            (cl_mem)eP.ptr[0],  eP.pitch[0],
            (cl_mem)lP.ptr[0],  lP.pitch[0],
            (cl_mem)dP.ptr[0],  dP.pitch[0],
            sP.width, sP.height,
            showmask);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at finedehalo_combine: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterFineDehalo::runMorph3x3(const char *kernelName,
                                          RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                          RGYOpenCLQueue &queue,
                                          const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, RGY_PLANE_Y);
    RGYWorkSize local(FINEDEHALO_BLOCK_X, FINEDEHALO_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    const float r = 1.0f;
    auto err = m_dehaloMorph.get()->kernel(kernelName)
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            r, r);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (3x3): %s.\n"),
            char_to_tstring(kernelName).c_str(), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterFineDehalo::copyChromaPlanes(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
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
            AddMessage(RGY_LOG_ERROR, _T("finedehalo chroma copy (plane %d) failed: %s.\n"),
                i, get_err_mes(err));
            return err;
        }
        waitFirst.clear();
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterFineDehalo::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue_main,
    const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    *pOutputFrameNum  = 0;
    ppOutputFrames[0] = nullptr;

    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }

    if (!m_finedehalo.get()) {
        AddMessage(RGY_LOG_ERROR, _T("finedehalo OpenCL program failed to build (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }
    if (!m_dehaloMorph.get()) {
        AddMessage(RGY_LOG_ERROR, _T("finedehalo dehalo-morph OpenCL program failed to build.\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamFineDehalo>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const int bitDepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
    const int maxVal   = (1 << bitDepth) - 1;
    const int thmiHbd   = (int)((long long)prm->finedehalo.thmi   * maxVal / 255);
    const int thmaHbd   = (int)((long long)prm->finedehalo.thma   * maxVal / 255);
    const int thlimiHbd = (int)((long long)prm->finedehalo.thlimi * maxVal / 255);
    const int thlimaHbd = (int)((long long)prm->finedehalo.thlima * maxVal / 255);

    RGYFrameInfo *pOut = &m_frameBuf[0]->frame;
    const int planes = RGY_CSP_PLANES[pOut->csp];
    const bool hasChroma = planes > 1;

    // ---- 1. DeHalo_alpha sub-filter ----
    RGYFrameInfo *pDehaloed = nullptr;
    {
        RGYFrameInfo *outArr[1] = { nullptr };
        int outCount = 0;
        auto err = m_dehalo->filter(const_cast<RGYFrameInfo *>(pInputFrame),
            outArr, &outCount, queue_main, wait_events, nullptr);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("finedehalo: sub-filter dehalo failed: %s.\n"), get_err_mes(err));
            return err;
        }
        if (outCount != 1 || outArr[0] == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("finedehalo: dehalo sub-filter returned no frame.\n"));
            return RGY_ERR_UNKNOWN;
        }
        pDehaloed = outArr[0];
    }

    // ---- 2. Prewitt edge detection on the ORIGINAL source ----
    {
        auto err = runPrewitt(&m_edges->frame, pInputFrame, thmiHbd, thmaHbd, queue_main, {});
        if (err != RGY_ERR_NONE) return err;
    }

    // ---- 3. Edge zone morphology pass A: expand → inpand ----
    {
        auto err = runMorph3x3("dehalo_expand", &m_morphTmp->frame, &m_edges->frame, queue_main, {});
        if (err != RGY_ERR_NONE) return err;
        err = runMorph3x3("dehalo_inpand", &m_ey->frame, &m_morphTmp->frame, queue_main, {});
        if (err != RGY_ERR_NONE) return err;
    }

    // ---- 4. Edge zone morphology pass B: expand → inpand ----
    {
        auto err = runMorph3x3("dehalo_expand", &m_morphTmp->frame, &m_ey->frame, queue_main, {});
        if (err != RGY_ERR_NONE) return err;
        err = runMorph3x3("dehalo_inpand", &m_em->frame, &m_morphTmp->frame, queue_main, {});
        if (err != RGY_ERR_NONE) return err;
    }

    // ---- 5. Limit mask from src ↔ dehaloed difference ----
    {
        auto err = runLimitMask(&m_linemask->frame, pInputFrame, pDehaloed,
                                thlimiHbd, thlimaHbd, queue_main, {});
        if (err != RGY_ERR_NONE) return err;
    }

    // ---- 6. Final merge / mask debug ----
    const int showmask = prm->finedehalo.showmask;
    if (showmask == 1 || showmask == 2 || showmask == 3) {
        // Debug: copy the requested intermediate luma to output. Combine
        // kernel is skipped; chroma copy below greyscales the frame
        // implicitly via the pass-through (chroma carries source colour,
        // but the luma is the mask, so on a greyscale viewer the mask
        // shape is what's visible).
        const RGYFrameInfo *pSrcMask = (showmask == 1) ? &m_edges->frame
                                     : (showmask == 2) ? &m_em->frame
                                                       : &m_linemask->frame;
        const auto srcP = getPlane(pSrcMask, RGY_PLANE_Y);
        const auto dstP = getPlane(pOut,     RGY_PLANE_Y);
        auto err = m_cl->copyPlane(const_cast<RGYFrameInfo *>(&dstP), &srcP, nullptr, queue_main, {}, hasChroma ? nullptr : event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("finedehalo showmask copy failed: %s.\n"), get_err_mes(err));
            return err;
        }
    } else {
        // Normal path (showmask==0) and final-mask path (showmask==4) both
        // go through the combine kernel — the kernel branches on showmask
        // internally to either emit blended output or final_mask.
        auto err = runCombine(pOut, pInputFrame, pDehaloed,
                              &m_em->frame, &m_linemask->frame,
                              showmask, queue_main, {}, hasChroma ? nullptr : event);
        if (err != RGY_ERR_NONE) return err;
    }

    // Chroma planes always come from the original source (luma-only filter).
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

void RGYFilterFineDehalo::close() {
    m_finedehalo.clear();
    m_dehaloMorph.clear();
    m_buildOptions.clear();
    m_dehalo.reset();
    m_edges.reset();
    m_morphTmp.reset();
    m_ey.reset();
    m_em.reset();
    m_linemask.reset();
    m_frameBuf.clear();
    m_cl.reset();
}
