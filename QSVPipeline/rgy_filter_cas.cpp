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
#include "rgy_filter_cas.h"

RGYFilterCas::RGYFilterCas(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_bInterlacedWarn(false), m_cas() {
    m_name = _T("cas");
}

RGYFilterCas::~RGYFilterCas() {
    close();
}

RGY_ERR RGYFilterCas::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamCas>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto chromaFormat = RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp];
    const bool supportedCsp = chromaFormat == RGY_CHROMAFMT_MONOCHROME
        || (RGY_CSP_PLANES[prm->frameOut.csp] > 1
            && (chromaFormat == RGY_CHROMAFMT_YUV420 || chromaFormat == RGY_CHROMAFMT_YUV422 || chromaFormat == RGY_CHROMAFMT_YUV444));
    if (!supportedCsp) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp for cas: %s.\n"), RGY_CSP_NAMES[prm->frameOut.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->cas.sharpness < 0.0f || 1.0f < prm->cas.sharpness) {
        prm->cas.sharpness = clamp(prm->cas.sharpness, 0.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("sharpness should be in range of %.1f - %.1f.\n"), 0.0f, 1.0f);
    }

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamCas>(m_param);
    if (!m_cas.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp]);
        m_cas.set(m_cl->buildResourceAsync(_T("RGY_FILTER_CAS_CL"), _T("EXE_DATA"), options.c_str()));
    }

    auto err = AllocFrameBuf(prm->frameOut, 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterCas::procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, float peak, int apply_gamma2, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *kernel_name = "kernel_cas";
    RGYWorkSize local(32, 8);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    auto err = m_cas.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
        peak, apply_gamma2);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterCas::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamCas>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    // peak = -1 / mix(8.0, 5.0, sharpness). Negative scalar; the kernel
    // multiplies the adaptive amplitude by this to get the negative-lobe
    // weight applied to the cross neighbours.
    const float sharp = std::min(std::max(prm->cas.sharpness, 0.0f), 1.0f);
    const float lerp = 8.0f + (5.0f - 8.0f) * sharp;
    const float peak = -1.0f / lerp;
    const bool use_gamma2 = !prm->cas.hdr;

    const int nPlanes = RGY_CSP_PLANES[pOutputFrame->csp];
    auto planeDstY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeSrcY = getPlane(pInputFrame, RGY_PLANE_Y);
    auto err = procPlane(&planeDstY, &planeSrcY, peak, use_gamma2 ? 1 : 0, queue, wait_events, (nPlanes == 1) ? event : nullptr);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    for (int i = 1; i < nPlanes; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        err = m_cl->copyPlane(&planeDst, &planeSrc, nullptr, queue, std::vector<RGYOpenCLEvent>(), (i == nPlanes - 1) ? event : nullptr);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("cas chroma copy (plane %d) failed: %s.\n"), i, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterCas::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
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
    if (!m_cas.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build RGY_FILTER_CAS_CL(m_cas)\n"));
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

void RGYFilterCas::close() {
    m_frameBuf.clear();
    m_cas.clear();
    m_cl.reset();
    m_bInterlacedWarn = false;
}
