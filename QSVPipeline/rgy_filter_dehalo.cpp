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
#include "rgy_filter_dehalo.h"
#include "rgy_filter_resize.h"

static const int DEHALO_BLOCK_X = 32;
static const int DEHALO_BLOCK_Y = 8;

RGYFilterDehalo::RGYFilterDehalo(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_dehalo(),
    m_buildOptions(),
    m_resizeUp(),
    m_resizeDown(),
    m_supersampled(),
    m_expanded(),
    m_inpand(),
    m_mask(),
    m_corrected(),
    m_ssW(0),
    m_ssH(0),
    m_ssActive(false) {
    m_name = _T("dehalo");
}

RGYFilterDehalo::~RGYFilterDehalo() {
    close();
}

RGY_ERR RGYFilterDehalo::checkParam(const std::shared_ptr<RGYFilterParamDehalo> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height < 4 || prm->frameOut.width < 4) {
        AddMessage(RGY_LOG_ERROR, _T("dehalo requires input width/height >= 4 (got %dx%d).\n"),
            prm->frameOut.width, prm->frameOut.height);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.rx >= 0.5f && prm->dehalo.rx <= 10.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid rx=%.2f: must be in [0.5, 10.0].\n"), prm->dehalo.rx);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.ry >= 0.5f && prm->dehalo.ry <= 10.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid ry=%.2f: must be in [0.5, 10.0].\n"), prm->dehalo.ry);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.darkstr >= 0.0f && prm->dehalo.darkstr <= 1.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid darkstr=%.2f: must be in [0.0, 1.0].\n"), prm->dehalo.darkstr);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.brightstr >= 0.0f && prm->dehalo.brightstr <= 1.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid brightstr=%.2f: must be in [0.0, 1.0].\n"), prm->dehalo.brightstr);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dehalo.lowsens < 0 || prm->dehalo.lowsens > 100) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid lowsens=%d: must be in [0, 100].\n"), prm->dehalo.lowsens);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dehalo.highsens < 0 || prm->dehalo.highsens > 100) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid highsens=%d: must be in [0, 100].\n"), prm->dehalo.highsens);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.ss >= 1.0f && prm->dehalo.ss <= 4.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid ss=%.2f: must be in [1.0, 4.0].\n"), prm->dehalo.ss);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDehalo::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDehalo>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    // Single-frame, 1-in/1-out, no fps change.
    prm->frameOut.picstruct = prm->frameIn.picstruct;

    // Bit-depth setup.
    const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    const int maxVal   = (1 << bitDepth) - 1;

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamDehalo>(m_param);
    if (!prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        m_buildOptions = strsprintf("-D Type=%s -D bit_depth=%d -D max_val=%d -D dehalo_block_x=%d -D dehalo_block_y=%d",
            bitDepth > 8 ? "ushort" : "uchar",
            bitDepth, maxVal,
            DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
        AddMessage(RGY_LOG_DEBUG, _T("Starting async build for RGY_FILTER_DEHALO_CL: %s\n"),
            char_to_tstring(m_buildOptions).c_str());
        m_dehalo.set(m_cl->buildResourceAsync(_T("RGY_FILTER_DEHALO_CL"), _T("EXE_DATA"), m_buildOptions.c_str()));
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

    // Decide working resolution. ss == 1.0 → process at source resolution and
    // skip the resize sub-filters entirely; ss > 1.0 → instantiate Spline36
    // resize-up and resize-down sub-filters and run the morphology /
    // mask / apply pipeline at the supersampled resolution.
    const float ssRatio = prm->dehalo.ss;
    m_ssActive = (ssRatio > 1.0f + 1e-6f);
    if (m_ssActive) {
        // Round to even for safer chroma subsampling alignment, mirroring MAA.
        m_ssW = ((int)std::lround(prm->frameIn.width  * ssRatio) + 1) & ~1;
        m_ssH = ((int)std::lround(prm->frameIn.height * ssRatio) + 1) & ~1;
    } else {
        m_ssW = prm->frameIn.width;
        m_ssH = prm->frameIn.height;
    }

    if (m_ssActive) {
        // Spline36 resize sub-filters: src → m_supersampled, m_corrected → output.
        {
            auto prmUp = std::make_shared<RGYFilterParamResize>();
            prmUp->frameIn        = prm->frameIn;
            prmUp->frameOut       = prm->frameIn;
            prmUp->frameOut.width  = m_ssW;
            prmUp->frameOut.height = m_ssH;
            prmUp->interp         = RGY_VPP_RESIZE_SPLINE36;
            prmUp->baseFps        = prm->baseFps;
            prmUp->bOutOverwrite  = false;
            m_resizeUp = std::make_unique<RGYFilterResize>(m_cl);
            sts = m_resizeUp->init(prmUp, m_pLog);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to init dehalo upscale sub-filter: %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
        {
            auto prmDn = std::make_shared<RGYFilterParamResize>();
            prmDn->frameIn        = prm->frameIn;
            prmDn->frameIn.width   = m_ssW;
            prmDn->frameIn.height  = m_ssH;
            prmDn->frameOut       = prm->frameOut;
            prmDn->interp         = RGY_VPP_RESIZE_SPLINE36;
            prmDn->baseFps        = prm->baseFps;
            prmDn->bOutOverwrite  = false;
            m_resizeDown = std::make_unique<RGYFilterResize>(m_cl);
            sts = m_resizeDown->init(prmDn, m_pLog);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to init dehalo downscale sub-filter: %s.\n"), get_err_mes(sts));
                return sts;
            }
        }

        // Working buffers at supersampled resolution.
        RGYFrameInfo ssInfo = prm->frameIn;
        ssInfo.width  = m_ssW;
        ssInfo.height = m_ssH;
        m_supersampled = m_cl->createFrameBuffer(ssInfo);
        m_expanded     = m_cl->createFrameBuffer(ssInfo);
        m_inpand       = m_cl->createFrameBuffer(ssInfo);
        m_mask         = m_cl->createFrameBuffer(ssInfo);
        m_corrected    = m_cl->createFrameBuffer(ssInfo);
        if (!m_supersampled || !m_expanded || !m_inpand || !m_mask || !m_corrected) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate dehalo supersampled buffers.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    } else {
        // Native-resolution path. m_supersampled and m_corrected stay null;
        // the morph / mask passes write into source-res scratch buffers and
        // the apply pass writes directly into m_frameBuf[0].
        m_expanded = m_cl->createFrameBuffer(prm->frameIn);
        m_inpand   = m_cl->createFrameBuffer(prm->frameIn);
        m_mask     = m_cl->createFrameBuffer(prm->frameIn);
        if (!m_expanded || !m_inpand || !m_mask) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate dehalo native-res buffers.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    AddMessage(RGY_LOG_DEBUG,
        _T("dehalo init: src=%dx%d bitDepth=%d, rx=%.2f, ry=%.2f, darkstr=%.2f, brightstr=%.2f, lowsens=%d, highsens=%d, ss=%.2f%s\n"),
        prm->frameIn.width, prm->frameIn.height, bitDepth,
        prm->dehalo.rx, prm->dehalo.ry, prm->dehalo.darkstr, prm->dehalo.brightstr,
        prm->dehalo.lowsens, prm->dehalo.highsens, prm->dehalo.ss,
        m_ssActive ? _T(" (supersampled)") : _T(" (native)"));

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterDehalo::runExpand(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                    float rx, float ry,
                                    RGYOpenCLQueue &queue,
                                    const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, RGY_PLANE_Y);
    RGYWorkSize local(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    RGYWorkSize global(ALIGN(sP.width, DEHALO_BLOCK_X), ALIGN(sP.height, DEHALO_BLOCK_Y));
    auto err = m_dehalo.get()->kernel("dehalo_expand")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            rx, ry);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at dehalo_expand: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDehalo::runInpand(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                    float rx, float ry,
                                    RGYOpenCLQueue &queue,
                                    const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, RGY_PLANE_Y);
    RGYWorkSize local(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    RGYWorkSize global(ALIGN(sP.width, DEHALO_BLOCK_X), ALIGN(sP.height, DEHALO_BLOCK_Y));
    auto err = m_dehalo.get()->kernel("dehalo_inpand")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            rx, ry);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at dehalo_inpand: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDehalo::runMask(RGYFrameInfo *pMaskDst,
                                  const RGYFrameInfo *pSrc, const RGYFrameInfo *pExpand, const RGYFrameInfo *pInpand,
                                  int loScaled, int hiScaled,
                                  RGYOpenCLQueue &queue,
                                  const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc,    RGY_PLANE_Y);
    const auto eP = getPlane(pExpand, RGY_PLANE_Y);
    const auto iP = getPlane(pInpand, RGY_PLANE_Y);
    const auto mP = getPlane(pMaskDst, RGY_PLANE_Y);
    RGYWorkSize local(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_dehalo.get()->kernel("dehalo_mask")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)eP.ptr[0], eP.pitch[0],
            (cl_mem)iP.ptr[0], iP.pitch[0],
            (cl_mem)mP.ptr[0], mP.pitch[0],
            sP.width, sP.height,
            loScaled, hiScaled);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at dehalo_mask: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDehalo::runApply(RGYFrameInfo *pDst,
                                   const RGYFrameInfo *pSrc, const RGYFrameInfo *pExpand, const RGYFrameInfo *pInpand,
                                   const RGYFrameInfo *pMask,
                                   float darkstr, float brightstr,
                                   RGYOpenCLQueue &queue,
                                   const std::vector<RGYOpenCLEvent> &wait_events,
                                   RGYOpenCLEvent *event) {
    const auto sP = getPlane(pSrc,    RGY_PLANE_Y);
    const auto eP = getPlane(pExpand, RGY_PLANE_Y);
    const auto iP = getPlane(pInpand, RGY_PLANE_Y);
    const auto mP = getPlane(pMask,   RGY_PLANE_Y);
    const auto dP = getPlane(pDst,    RGY_PLANE_Y);
    RGYWorkSize local(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_dehalo.get()->kernel("dehalo_apply")
        .config(queue, local, global, wait_events, event).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)eP.ptr[0], eP.pitch[0],
            (cl_mem)iP.ptr[0], iP.pitch[0],
            (cl_mem)mP.ptr[0], mP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            darkstr, brightstr);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at dehalo_apply: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDehalo::copyChromaPlanes(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
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
            AddMessage(RGY_LOG_ERROR, _T("dehalo chroma copy (plane %d) failed: %s.\n"),
                i, get_err_mes(err));
            return err;
        }
        waitFirst.clear();
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDehalo::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue_main,
    const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    *pOutputFrameNum  = 0;
    ppOutputFrames[0] = nullptr;

    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }

    if (!m_dehalo.get()) {
        AddMessage(RGY_LOG_ERROR, _T("dehalo OpenCL program failed to build (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamDehalo>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    // Pre-scale lowsens / highsens to working bit depth at the host:
    //   lo = lowsens  * max_val / 100
    //   hi = highsens * max_val / 100
    // Kernel uses these as integer ramp anchors.
    const int bitDepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
    const int maxVal   = (1 << bitDepth) - 1;
    const int loScaled = (int)((long long)prm->dehalo.lowsens  * maxVal / 100);
    const int hiScaled = (int)((long long)prm->dehalo.highsens * maxVal / 100);

    RGYFrameInfo *pOut = &m_frameBuf[0]->frame;

    // Choose the source for the morph passes: supersampled buffer (ss>1)
    // or the input frame directly (ss==1).
    const RGYFrameInfo *pMorphSrc = nullptr;

    // queue_main is in-order, so only the first enqueue needs the upstream
    // wait_events, and only the last enqueue needs to export event.
    const int planes = RGY_CSP_PLANES[pOut->csp];
    const bool hasChroma = planes > 1;

    // ---- 1. (ss > 1 only) Spline36 resize-up ----
    if (m_ssActive) {
        int dummyOutNum = 0;
        RGYFrameInfo *outArr[1] = { &m_supersampled->frame };
        auto err = m_resizeUp->filter(const_cast<RGYFrameInfo *>(pInputFrame),
            (RGYFrameInfo **)&outArr, &dummyOutNum, queue_main, wait_events, nullptr);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("dehalo resize-up failed: %s.\n"), get_err_mes(err));
            return err;
        }
        pMorphSrc = &m_supersampled->frame;
    } else {
        pMorphSrc = pInputFrame;
    }

    // The wait_events list is consumed by either the resize-up or by the
    // first morph pass.
    const std::vector<RGYOpenCLEvent> initialWait = m_ssActive ? std::vector<RGYOpenCLEvent>() : wait_events;

    // ---- 2. Elliptic local maximum (expand) ----
    {
        auto err = runExpand(&m_expanded->frame, pMorphSrc,
                             prm->dehalo.rx, prm->dehalo.ry,
                             queue_main,
                             initialWait);
        if (err != RGY_ERR_NONE) return err;
    }

    // ---- 3. Elliptic local minimum (inpand) ----
    {
        auto err = runInpand(&m_inpand->frame, pMorphSrc,
                             prm->dehalo.rx, prm->dehalo.ry,
                             queue_main, {});
        if (err != RGY_ERR_NONE) return err;
    }

    // ---- 4. Sensitivity-ramp mask ----
    {
        auto err = runMask(&m_mask->frame,
                           pMorphSrc, &m_expanded->frame, &m_inpand->frame,
                           loScaled, hiScaled,
                           queue_main, {});
        if (err != RGY_ERR_NONE) return err;
    }

    // ---- 5. Alpha-blend correction ----
    //   ss > 1 → write into m_corrected (then resize-down to pOut)
    //   ss == 1 → write directly into pOut (m_frameBuf[0])
    RGYFrameInfo *pApplyDst = m_ssActive ? &m_corrected->frame : pOut;
    {
        auto err = runApply(pApplyDst,
                            pMorphSrc, &m_expanded->frame, &m_inpand->frame, &m_mask->frame,
                            prm->dehalo.darkstr, prm->dehalo.brightstr,
                            queue_main, {}, (!m_ssActive && !hasChroma) ? event : nullptr);
        if (err != RGY_ERR_NONE) return err;
    }

    // ---- 6. (ss > 1 only) Spline36 resize-down to source resolution ----
    if (m_ssActive) {
        int dummyOutNum = 0;
        RGYFrameInfo *outArr[1] = { pOut };
        auto err = m_resizeDown->filter(&m_corrected->frame,
            (RGYFrameInfo **)&outArr, &dummyOutNum, queue_main, {}, hasChroma ? nullptr : event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("dehalo resize-down failed: %s.\n"), get_err_mes(err));
            return err;
        }
    }

    // Chroma planes are always copied from the original source.
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

void RGYFilterDehalo::close() {
    m_dehalo.clear();
    m_buildOptions.clear();
    m_resizeUp.reset();
    m_resizeDown.reset();
    m_supersampled.reset();
    m_expanded.reset();
    m_inpand.reset();
    m_mask.reset();
    m_corrected.reset();
    m_ssW = 0;
    m_ssH = 0;
    m_ssActive = false;
    m_frameBuf.clear();
    m_cl.reset();
}
