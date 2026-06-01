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
#include <algorithm>
#include "rgy_filter_vinverse.h"

RGYFilterVinverse::RGYFilterVinverse(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_bInterlacedWarn(false), m_vinverse(), m_pb3(), m_pb6() {
    m_name = _T("vinverse");
}

RGYFilterVinverse::~RGYFilterVinverse() {
    close();
}

RGY_ERR RGYFilterVinverse::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamVinverse>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->vinverse.sstr < 0.0f || 8.0f < prm->vinverse.sstr) {
        prm->vinverse.sstr = clamp(prm->vinverse.sstr, 0.0f, 8.0f);
        AddMessage(RGY_LOG_WARN, _T("sstr should be in range of %.1f - %.1f.\n"), 0.0f, 8.0f);
    }
    if (prm->vinverse.amnt < 0.0f || 255.0f < prm->vinverse.amnt) {
        prm->vinverse.amnt = clamp(prm->vinverse.amnt, 0.0f, 255.0f);
        AddMessage(RGY_LOG_WARN, _T("amnt should be in range of %.1f - %.1f.\n"), 0.0f, 255.0f);
    }
    if (prm->vinverse.scl < 0.0f || 4.0f < prm->vinverse.scl) {
        prm->vinverse.scl = clamp(prm->vinverse.scl, 0.0f, 4.0f);
        AddMessage(RGY_LOG_WARN, _T("scl should be in range of %.1f - %.1f.\n"), 0.0f, 4.0f);
    }
    if (prm->vinverse.thr < 0.0f || 255.0f < prm->vinverse.thr) {
        prm->vinverse.thr = clamp(prm->vinverse.thr, 0.0f, 255.0f);
        AddMessage(RGY_LOG_WARN, _T("thr should be in range of %.1f - %.1f.\n"), 0.0f, 255.0f);
    }

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamVinverse>(m_param);
    if (!m_vinverse.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp]);
        m_vinverse.set(m_cl->buildResourceAsync(_T("RGY_FILTER_VINVERSE_CL"), _T("EXE_DATA"), options.c_str()));
    }

    auto err = AllocFrameBuf(prm->frameOut, 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    if (!m_pb3 || cmpFrameInfoCspResolution(&m_pb3->frame, &prm->frameOut)) {
        m_pb3 = m_cl->createFrameBuffer(prm->frameOut, CL_MEM_READ_WRITE);
    }
    if (!m_pb6 || cmpFrameInfoCspResolution(&m_pb6->frame, &prm->frameOut)) {
        m_pb6 = m_cl->createFrameBuffer(prm->frameOut, CL_MEM_READ_WRITE);
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterVinverse::procPlaneVblur3(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *kernel_name = "kernel_vinverse_vblur3";
    // 8x32 work-group favours vertical stencils: adjacent work-items
    // in a column share the ±1 row reads through the L1 cache.
    RGYWorkSize local(8, 32);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    auto err = m_vinverse.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0]);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlaneVblur3(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterVinverse::procPlaneVblur5(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *kernel_name = "kernel_vinverse_vblur5";
    RGYWorkSize local(8, 32);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    auto err = m_vinverse.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0]);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlaneVblur5(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterVinverse::procPlaneVblur35(RGYFrameInfo *pPb3Plane, RGYFrameInfo *pPb6Plane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *kernel_name = "kernel_vinverse_vblur35";
    RGYWorkSize local(8, 32);
    RGYWorkSize global(pPb3Plane->width, pPb3Plane->height);
    auto err = m_vinverse.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pPb3Plane->ptr[0], pPb3Plane->pitch[0],
        (cl_mem)pPb6Plane->ptr[0], pPb6Plane->pitch[0],
        pPb3Plane->width, pPb3Plane->height,
        (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0]);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlaneVblur35(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterVinverse::procPlaneMakediff(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pC1Plane, const RGYFrameInfo *pC2Plane, int h_offset,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *kernel_name = "kernel_vinverse_makediff";
    RGYWorkSize local(32, 8);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    auto err = m_vinverse.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (cl_mem)pC1Plane->ptr[0], pC1Plane->pitch[0],
        (cl_mem)pC2Plane->ptr[0], pC2Plane->pitch[0],
        h_offset);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlaneMakediff(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pC1Plane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterVinverse::procPlaneSbrCombine(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pSrcPlane, const RGYFrameInfo *pDiffPlane, const RGYFrameInfo *pBlurPlane, int h_offset,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *kernel_name = "kernel_vinverse_sbr_combine";
    RGYWorkSize local(32, 8);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    auto err = m_vinverse.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (cl_mem)pSrcPlane->ptr[0], pSrcPlane->pitch[0],
        (cl_mem)pDiffPlane->ptr[0], pDiffPlane->pitch[0],
        (cl_mem)pBlurPlane->ptr[0], pBlurPlane->pitch[0],
        h_offset);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlaneSbrCombine(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pSrcPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterVinverse::procPlaneFinalize(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const RGYFrameInfo *pPb3Plane, const RGYFrameInfo *pPb6Plane,
    float sstr, float scl, int thr_hbd, int amnt_hbd,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *kernel_name = "kernel_vinverse_finalize";
    RGYWorkSize local(32, 8);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    auto err = m_vinverse.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
        (cl_mem)pPb3Plane->ptr[0], pPb3Plane->pitch[0],
        (cl_mem)pPb6Plane->ptr[0], pPb6Plane->pitch[0],
        sstr, scl, thr_hbd, amnt_hbd);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlaneFinalize(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterVinverse::procPlane(int planeIdx, RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYFrameInfo *pPb3Plane, RGYFrameInfo *pPb6Plane,
    VppVinverseMode mode, float sstr, float scl, int thr_hbd, int amnt_hbd, int h_offset,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR err = RGY_ERR_NONE;
    if (mode == VppVinverseMode::Vinverse) {
        // Fused vblur3 + vblur5: one dispatch writes pb3 (vblur3 of src)
        // and pb6 (vblur5 of pb3) at the same coordinate, eliminating
        // the intermediate pb3 DRAM R+W cycle the two-pass chain
        // incurred. The fused kernel performs five vblur3 evaluations
        // per output pixel from the source plane; cache locality in the
        // 8x32 work-group absorbs the column re-reads.
        err = procPlaneVblur35(pPb3Plane, pPb6Plane, pInputPlane, queue, wait_events, nullptr);
        if (err != RGY_ERR_NONE) return err;
    } else {
        // Vinverse2:
        //   luma  : pb3 = sbr(src) via { pb6 = vblur3(src);
        //                                 pb3 = makediff(src, pb6, h);
        //                                 pb6 = vblur3(pb3);
        //                                 pb3 = sbr_combine(src, pb3, pb6, h) }
        //   chroma: pb3 = copy(src)                    (reference is luma-only)
        //   then    : pb6 = vblur3(pb3)
        if (planeIdx == 0) {
            err = procPlaneVblur3(pPb6Plane, pInputPlane, queue, wait_events, nullptr);
            if (err != RGY_ERR_NONE) return err;
            err = procPlaneMakediff(pPb3Plane, pInputPlane, pPb6Plane, h_offset, queue, {}, nullptr);
            if (err != RGY_ERR_NONE) return err;
            err = procPlaneVblur3(pPb6Plane, pPb3Plane, queue, {}, nullptr);
            if (err != RGY_ERR_NONE) return err;
            err = procPlaneSbrCombine(pPb3Plane, pInputPlane, pPb3Plane, pPb6Plane, h_offset, queue, {}, nullptr);
            if (err != RGY_ERR_NONE) return err;
        } else {
            err = m_cl->copyPlane(pPb3Plane, pInputPlane, nullptr, queue, wait_events, nullptr);
            if (err != RGY_ERR_NONE) return err;
        }
        err = procPlaneVblur3(pPb6Plane, pPb3Plane, queue, {}, nullptr);
        if (err != RGY_ERR_NONE) return err;
    }
    err = procPlaneFinalize(pOutputPlane, pInputPlane, pPb3Plane, pPb6Plane, sstr, scl, thr_hbd, amnt_hbd, queue, {}, event);
    return err;
}

RGY_ERR RGYFilterVinverse::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamVinverse>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int bit_depth = RGY_CSP_BIT_DEPTH[pOutputFrame->csp];
    const int peak = (1 << bit_depth) - 1;
    const int hbd_shift = bit_depth - 8;
    // amnt is exposed in 8-bit units (0..255). The reference's default
    // behaviour is "no per-pixel cap" — we get that by saturating amnt at
    // peak when the user-facing value reaches the full 255.
    int amnt_hbd = (int)(prm->vinverse.amnt * (float)(1 << hbd_shift));
    if (prm->vinverse.amnt >= 255.0f || amnt_hbd > peak) amnt_hbd = peak;
    if (amnt_hbd < 0) amnt_hbd = 0;
    int thr_hbd = (int)(prm->vinverse.thr * (float)(1 << hbd_shift));
    if (thr_hbd < 0) thr_hbd = 0;
    if (thr_hbd > peak) thr_hbd = peak;
    // h_offset is the "diff centring" constant used by makediff and
    // sbr_combine in Vinverse2 mode: 1 << (bit_depth - 1) = half-peak.
    const int h_offset = 1 << (bit_depth - 1);

    const int nPlanes = RGY_CSP_PLANES[pOutputFrame->csp];
    for (int i = 0; i < nPlanes; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        auto planePb3 = getPlane(&m_pb3->frame, (RGY_PLANE)i);
        auto planePb6 = getPlane(&m_pb6->frame, (RGY_PLANE)i);

        const bool process = (i == 0) || prm->vinverse.chroma;
        const std::vector<RGYOpenCLEvent> &plane_wait = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == nPlanes - 1) ? event : nullptr;

        if (!process) {
            auto err = m_cl->copyPlane(&planeDst, &planeSrc, nullptr, queue, plane_wait, plane_event);
            if (err != RGY_ERR_NONE) return err;
            continue;
        }

        auto err = procPlane(i, &planeDst, &planeSrc, &planePb3, &planePb6,
            prm->vinverse.mode, prm->vinverse.sstr, prm->vinverse.scl, thr_hbd, amnt_hbd, h_offset,
            queue, plane_wait, plane_event);
        if (err != RGY_ERR_NONE) return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterVinverse::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
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
    if (!m_vinverse.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build RGY_FILTER_VINVERSE_CL(m_vinverse)\n"));
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

void RGYFilterVinverse::close() {
    m_pb3.reset();
    m_pb6.reset();
    m_frameBuf.clear();
    m_vinverse.clear();
    m_cl.reset();
    m_bInterlacedWarn = false;
}
