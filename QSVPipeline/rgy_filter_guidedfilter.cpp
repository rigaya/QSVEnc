// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2026 rigaya
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

#include "rgy_filter_guidedfilter.h"

RGY_ERR RGYFilterGuidedfilter::procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamGuidedfilter>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pOutputPlane->width > m_abAllocW || pOutputPlane->height > m_abAllocH) {
        AddMessage(RGY_LOG_ERROR, _T("plane size %dx%d exceeds allocated intermediate buffer %dx%d.\n"),
            pOutputPlane->width, pOutputPlane->height, m_abAllocW, m_abAllocH);
        return RGY_ERR_INVALID_PARAM;
    }
    const int abPitch = m_abAllocW;
    const int planeW  = pOutputPlane->width;
    const int planeH  = pOutputPlane->height;
    RGYWorkSize local(32, 8);
    RGYWorkSize global(ALIGN(planeW, 32), ALIGN(planeH, 8));

    // Pass 1: I -> (a, b)
    {
        const char *kernel_name = "kernel_guidedfilter_calc_ab";
        auto err = m_guidedfilter.get()->kernel(kernel_name).config(queue, local, global, wait_events, nullptr).launch(
            m_bufA->mem(),
            m_bufB->mem(),
            abPitch,
            (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
            planeW, planeH,
            (int)prm->guidedfilter.radius,
            prm->guidedfilter.eps);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }

    // Pass 2: I + (a, b) -> q
    {
        const char *kernel_name = "kernel_guidedfilter_calc_q";
        auto err = m_guidedfilter.get()->kernel(kernel_name).config(queue, local, global, {}, event).launch(
            (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
            (cl_mem)pInputPlane->ptr[0],  pInputPlane->pitch[0],
            planeW, planeH,
            m_bufA->mem(),
            m_bufB->mem(),
            abPitch,
            (int)prm->guidedfilter.radius);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterGuidedfilter::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamGuidedfilter>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int numPlanes = RGY_CSP_PLANES[pOutputFrame->csp];
    for (int i = 0; i < numPlanes; i++) {
        if (i > 0 && !prm->guidedfilter.chroma) {
            auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
            auto planeSrc = getPlane(pInputFrame,  (RGY_PLANE)i);
            auto sts = m_cl->copyPlane(&planeDst, &planeSrc, nullptr, queue, {}, (i == numPlanes - 1) ? event : nullptr);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            continue;
        }
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame,  (RGY_PLANE)i);
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == numPlanes - 1) ? event : nullptr;
        auto err = procPlane(&planeDst, &planeSrc, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to guidedfilter frame plane %d %s: %s\n"),
                i, RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterGuidedfilter::RGYFilterGuidedfilter(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context), m_guidedfilter(), m_bufA(), m_bufB(), m_abAllocW(0), m_abAllocH(0) {
    m_name = _T("guidedfilter");
}

RGYFilterGuidedfilter::~RGYFilterGuidedfilter() {
    close();
}

RGY_ERR RGYFilterGuidedfilter::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamGuidedfilter>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->guidedfilter.radius < 1 || 32 < prm->guidedfilter.radius) {
        prm->guidedfilter.radius = clamp(prm->guidedfilter.radius, 1, 32);
        AddMessage(RGY_LOG_WARN, _T("radius should be in range of %d - %d.\n"), 1, 32);
    }
    if (prm->guidedfilter.eps < 0.0001f || 1.0f < prm->guidedfilter.eps) {
        prm->guidedfilter.eps = clamp(prm->guidedfilter.eps, 0.0001f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("eps should be in range of %.4f - %.1f.\n"), 0.0001f, 1.0f);
    }

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamGuidedfilter>(m_param);
    if (!m_guidedfilter.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp]);
        m_guidedfilter.set(m_cl->buildResourceAsync(_T("RGY_FILTER_GUIDEDFILTER_CL"), _T("EXE_DATA"), options.c_str()));
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // Float intermediate (a, b) buffers sized for the luma plane (the
    // largest plane). Re-used across U/V when chroma=true since planes
    // are filtered sequentially.
    const int abW = prm->frameOut.width;
    const int abH = prm->frameOut.height;
    const bool resizeIntermediate = !m_bufA || abW > m_abAllocW || abH > m_abAllocH;
    if (resizeIntermediate) {
        const size_t bufBytes = (size_t)abW * (size_t)abH * sizeof(float);
        m_bufA = m_cl->createBuffer(bufBytes, CL_MEM_READ_WRITE);
        m_bufB = m_cl->createBuffer(bufBytes, CL_MEM_READ_WRITE);
        if (!m_bufA || !m_bufB) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate intermediate a, b buffers.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_abAllocW = abW;
        m_abAllocH = abH;
    }
    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterGuidedfilter::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
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
    if (!m_guidedfilter.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build RGY_FILTER_GUIDEDFILTER_CL(m_guidedfilter)\n"));
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
        AddMessage(RGY_LOG_ERROR, _T("error at guidedfilterFrame (%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }
    return sts;
}

void RGYFilterGuidedfilter::close() {
    m_frameBuf.clear();
    m_bufA.reset();
    m_bufB.reset();
    m_abAllocW = 0;
    m_abAllocH = 0;
    m_guidedfilter.clear();
    m_cl.reset();
}
