// -----------------------------------------------------------------------------------------
// QSVEnc/VCEEnc/rkmppenc by rigaya
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
// -----------------------------------------------------------------------------------------

#define _USE_MATH_DEFINES
#include <cmath>
#include "rgy_filter_detailsharpen.h"

RGYFilterDetailSharpen::RGYFilterDetailSharpen(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_detailsharpen(),
    m_blur(),
    m_buildBitDepth(0),
    m_buildMode(-1),
    m_buildMed(false) {
    m_name = _T("detailsharpen");
}

RGYFilterDetailSharpen::~RGYFilterDetailSharpen() {
    close();
}

RGY_ERR RGYFilterDetailSharpen::checkParam(const std::shared_ptr<RGYFilterParamDetailSharpen> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp] == RGY_CHROMAFMT_RGB) {
        AddMessage(RGY_LOG_ERROR, _T("detailsharpen is not supported on RGB csp %s.\n"), RGY_CSP_NAMES[prm->frameOut.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (RGY_CSP_DATA_TYPE[prm->frameOut.csp] != RGY_DATA_TYPE_U8 && RGY_CSP_DATA_TYPE[prm->frameOut.csp] != RGY_DATA_TYPE_U16) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp for detailsharpen: %s.\n"), RGY_CSP_NAMES[prm->frameOut.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->detailsharpen.z < 0.001f || 64.0f < prm->detailsharpen.z) {
        prm->detailsharpen.z = clamp(prm->detailsharpen.z, 0.001f, 64.0f);
        AddMessage(RGY_LOG_WARN, _T("z should be in range of %.3f - %.1f.\n"), 0.001f, 64.0f);
    }
    if (prm->detailsharpen.sstr < 0.0f || 16.0f < prm->detailsharpen.sstr) {
        prm->detailsharpen.sstr = clamp(prm->detailsharpen.sstr, 0.0f, 16.0f);
        AddMessage(RGY_LOG_WARN, _T("sstr should be in range of %.1f - %.1f.\n"), 0.0f, 16.0f);
    }
    if (prm->detailsharpen.power < 1.0f || 16.0f < prm->detailsharpen.power) {
        prm->detailsharpen.power = clamp(prm->detailsharpen.power, 1.0f, 16.0f);
        AddMessage(RGY_LOG_WARN, _T("power should be in range of %.1f - %.1f.\n"), 1.0f, 16.0f);
    }
    if (prm->detailsharpen.ldmp < 0.0f || 1000.0f < prm->detailsharpen.ldmp) {
        prm->detailsharpen.ldmp = clamp(prm->detailsharpen.ldmp, 0.0f, 1000.0f);
        AddMessage(RGY_LOG_WARN, _T("ldmp should be in range of %.1f - %.1f.\n"), 0.0f, 1000.0f);
    }
    if (prm->detailsharpen.mode < 0 || 1 < prm->detailsharpen.mode) {
        AddMessage(RGY_LOG_ERROR, _T("mode should be 0 or 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDetailSharpen::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDetailSharpen>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    const int bitDepth = prm->frameOut.bitdepth > 0 ? prm->frameOut.bitdepth : RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    if (!m_detailsharpen.get()
        || m_buildBitDepth != bitDepth
        || m_buildMode != prm->detailsharpen.mode
        || m_buildMed != prm->detailsharpen.med) {
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d -D detailsharpen_mode=%d -D detailsharpen_med=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            bitDepth,
            prm->detailsharpen.mode,
            prm->detailsharpen.med ? 1 : 0);
        m_detailsharpen.set(m_cl->buildResourceAsync(_T("RGY_FILTER_DETAILSHARPEN_CL"), _T("EXE_DATA"), options.c_str()));
        m_buildBitDepth = bitDepth;
        m_buildMode = prm->detailsharpen.mode;
        m_buildMed = prm->detailsharpen.med;
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    const auto planeY = getPlane(&prm->frameOut, RGY_PLANE_Y);
    const auto blurCsp = (RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;
    if (!m_blur
        || m_blur->frame.width != planeY.width
        || m_blur->frame.height != planeY.height
        || m_blur->frame.csp != blurCsp) {
        m_blur = m_cl->createFrameBuffer(planeY.width, planeY.height, blurCsp, RGY_CSP_BIT_DEPTH[blurCsp]);
        if (!m_blur) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate detailsharpen blur buffer.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDetailSharpen::procPlane(RGYFrameInfo *pOutputPlane, RGYFrameInfo *pBlurPlane, const RGYFrameInfo *pInputPlane,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDetailSharpen>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    RGYWorkSize local(32, 8);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    auto err = m_detailsharpen.get()->kernel("kernel_detailsharpen_blur").config(queue, local, global, wait_events).launch(
        (cl_mem)pBlurPlane->ptr[0], pBlurPlane->pitch[0], pBlurPlane->width, pBlurPlane->height,
        (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0]);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_detailsharpen_blur (%s): %s.\n"),
            RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }

    const int bitDepth = pInputPlane->bitdepth > 0 ? pInputPlane->bitdepth : RGY_CSP_BIT_DEPTH[pInputPlane->csp];
    const int shift = clamp(bitDepth, 8, 16) - 8;
    const float i = (float)(1 << shift);
    err = m_detailsharpen.get()->kernel("kernel_detailsharpen_apply").config(queue, local, global, {}, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
        (cl_mem)pBlurPlane->ptr[0], pBlurPlane->pitch[0],
        prm->detailsharpen.z,
        1.0f / prm->detailsharpen.power,
        prm->detailsharpen.ldmp,
        prm->detailsharpen.sstr * prm->detailsharpen.z * i,
        1.0f / i);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_detailsharpen_apply (%s): %s.\n"),
            RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDetailSharpen::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeBlurY = getPlane(&m_blur->frame, RGY_PLANE_Y);
    const int planes = RGY_CSP_PLANES[pOutputFrame->csp];
    auto err = procPlane(&planeOutputY, &planeBlurY, &planeInputY, queue, wait_events, (planes == 1) ? event : nullptr);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    for (int i = 1; i < planes; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        err = m_cl->copyPlane(&planeDst, &planeSrc, nullptr, queue, {}, (i == planes - 1) ? event : nullptr);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("detailsharpen plane copy (plane %d) failed: %s.\n"), i, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDetailSharpen::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
        return RGY_ERR_NONE;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        ppOutputFrames[0] = &m_frameBuf[0]->frame;
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (!m_detailsharpen.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build RGY_FILTER_DETAILSHARPEN_CL(m_detailsharpen)\n"));
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

    auto sts = procFrame(ppOutputFrames[0], pInputFrame, queue, wait_events, event);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at detailsharpen(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }
    return sts;
}

void RGYFilterDetailSharpen::close() {
    m_frameBuf.clear();
    m_blur.reset();
    m_detailsharpen.clear();
    m_cl.reset();
    m_buildBitDepth = 0;
    m_buildMode = -1;
    m_buildMed = false;
}
