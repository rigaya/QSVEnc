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

#include "rgy_filter_dehaze.h"
#include <cmath>

RGY_ERR RGYFilterDehaze::procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDehaze>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto local = RGYWorkSize(32, 8);
    const auto global = RGYWorkSize(ALIGN(pOutputPlane->width, 32), ALIGN(pOutputPlane->height, 8));
    auto minHorizontal = getPlane(&m_minHorizontal->frame, RGY_PLANE_Y);
    const char *kernelHorizontal = "kernel_dehaze_min_horizontal";
    auto err = m_dehaze.get()->kernel(kernelHorizontal).config(queue, local, global, wait_events, nullptr).launch(
        (cl_mem)minHorizontal.ptr[0], minHorizontal.pitch[0],
        (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
        pInputPlane->width, pInputPlane->height, prm->dehaze.patch_radius);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
            char_to_tstring(kernelHorizontal).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }

    const char *kernelVertical = "kernel_dehaze";
    err = m_dehaze.get()->kernel(kernelVertical).config(queue, local, global, {}, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
        (cl_mem)minHorizontal.ptr[0], minHorizontal.pitch[0],
        prm->dehaze.patch_radius, prm->dehaze.omega, prm->dehaze.t_floor, prm->dehaze.atm_light);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
            char_to_tstring(kernelVertical).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
    }
    return err;
}

RGYFilterDehaze::RGYFilterDehaze(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context), m_dehaze(), m_minHorizontal() {
    m_name = _T("dehaze");
}

RGYFilterDehaze::~RGYFilterDehaze() {
    close();
}

RGY_ERR RGYFilterDehaze::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDehaze>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (RGY_CSP_DATA_TYPE[prm->frameOut.csp] != RGY_DATA_TYPE_U8
        && RGY_CSP_DATA_TYPE[prm->frameOut.csp] != RGY_DATA_TYPE_U16) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp for dehaze: %s.\n"), RGY_CSP_NAMES[prm->frameOut.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (!std::isfinite(prm->dehaze.omega) || !std::isfinite(prm->dehaze.t_floor) || !std::isfinite(prm->dehaze.atm_light)) {
        AddMessage(RGY_LOG_ERROR, _T("dehaze parameters must be finite.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dehaze.patch_radius < 1 || 15 < prm->dehaze.patch_radius) {
        prm->dehaze.patch_radius = clamp(prm->dehaze.patch_radius, 1, 15);
        AddMessage(RGY_LOG_WARN, _T("patch_radius should be in range of %d - %d.\n"), 1, 15);
    }
    if (prm->dehaze.omega < 0.5f || 1.0f < prm->dehaze.omega) {
        prm->dehaze.omega = clamp(prm->dehaze.omega, 0.5f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("omega should be in range of %.2f - %.2f.\n"), 0.5f, 1.0f);
    }
    if (prm->dehaze.t_floor < 0.01f || 0.5f < prm->dehaze.t_floor) {
        prm->dehaze.t_floor = clamp(prm->dehaze.t_floor, 0.01f, 0.5f);
        AddMessage(RGY_LOG_WARN, _T("t_floor should be in range of %.2f - %.2f.\n"), 0.01f, 0.5f);
    }
    if (prm->dehaze.atm_light < 0.1f || 1.0f < prm->dehaze.atm_light) {
        prm->dehaze.atm_light = clamp(prm->dehaze.atm_light, 0.1f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("atm_light should be in range of %.2f - %.2f.\n"), 0.1f, 1.0f);
    }

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamDehaze>(m_param);
    const int storageBitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    if (!m_dehaze.get() || !prmPrev || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != storageBitdepth) {
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d",
            storageBitdepth > 8 ? "ushort" : "uchar", storageBitdepth);
        m_dehaze.set(m_cl->buildResourceAsync(_T("RGY_FILTER_DEHAZE_CL"), _T("EXE_DATA"), options.c_str()));
    }

    auto sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    auto frameY = getPlane(&prm->frameOut, RGY_PLANE_Y);
    if (!m_minHorizontal || cmpFrameInfoCspResolution(&m_minHorizontal->frame, &frameY)) {
        m_minHorizontal = m_cl->createFrameBuffer(frameY, CL_MEM_READ_WRITE);
        if (!m_minHorizontal) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate dehaze horizontal-min buffer.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDehaze::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (pInputFrame->ptr[0] == nullptr) {
        return RGY_ERR_NONE;
    }
    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        ppOutputFrames[0] = &m_frameBuf[0]->frame;
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (!m_dehaze.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build RGY_FILTER_DEHAZE_CL(m_dehaze)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    if (getMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type) != RGYCLMemcpyD2D) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    const int numPlanes = RGY_CSP_PLANES[ppOutputFrames[0]->csp];
    for (int i = 0; i < numPlanes; i++) {
        auto planeDst = getPlane(ppOutputFrames[0], (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        RGY_ERR sts = RGY_ERR_NONE;
        if (i == 0) {
            sts = procPlane(&planeDst, &planeSrc, queue, wait_events, (i == numPlanes - 1) ? event : nullptr);
        } else {
            sts = m_cl->copyPlane(&planeDst, &planeSrc, nullptr, queue, {}, (i == numPlanes - 1) ? event : nullptr);
        }
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at dehaze frame plane %d (%s): %s.\n"),
                i, RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

void RGYFilterDehaze::close() {
    m_frameBuf.clear();
    m_minHorizontal.reset();
    m_dehaze.clear();
    m_cl.reset();
}
