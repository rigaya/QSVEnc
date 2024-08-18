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

#include "rgy_filter_cl.h"

tstring RGYFilterParamPad::print() const {
    return strsprintf(_T("pad: [%dx%d]->[%dx%d] "),
        frameIn.width, frameIn.height,
        frameOut.width, frameOut.height)
        + pad.print();
}

RGY_ERR RGYFilterPad::procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, int pad_color, const VppPad& pad, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    {
        const char *kernel_name = "kernel_pad";
        RGYWorkSize local(32, 8);
        RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
        auto err = m_pad.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
            (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
            (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0], pInputPlane->width, pInputPlane->height,
            pad.left, pad.top, pad_color);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterPad::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamPad>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
    const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
    const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
    auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);

    const int padColorY = (RGY_CSP_CHROMA_FORMAT[prm->frameIn.csp] == RGY_CHROMAFMT_RGB) ? 0 : (uint16_t)(16 << (RGY_CSP_BIT_DEPTH[prm->frameIn.csp] - 8));
    const int padColorC = (RGY_CSP_CHROMA_FORMAT[prm->frameIn.csp] == RGY_CHROMAFMT_RGB) ? 0 : (uint16_t)(128 << (RGY_CSP_BIT_DEPTH[prm->frameIn.csp] - 8));

    auto sts = procPlane(&planeOutputY, &planeInputY, padColorY, prm->pad, queue, wait_events, nullptr);
    if (sts != RGY_ERR_NONE) return sts;

    auto uvPad = prm->pad;
    if (RGY_CSP_CHROMA_FORMAT[prm->frameIn.csp] == RGY_CHROMAFMT_YUV420) {
        uvPad.right >>= 1;
        uvPad.left >>= 1;
        uvPad.top >>= 1;
        uvPad.bottom >>= 1;
    } else if (RGY_CSP_CHROMA_FORMAT[prm->frameIn.csp] == RGY_CHROMAFMT_YUV444) {
        //特に何もしない
    } else {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp.\n"));
        sts = RGY_ERR_UNSUPPORTED;
    }

    sts = procPlane(&planeOutputU, &planeInputU, padColorC, uvPad, queue, {}, nullptr);
    if (sts != RGY_ERR_NONE) return sts;

    sts = procPlane(&planeOutputV, &planeInputV, padColorC, uvPad, queue, {}, event);
    if (sts != RGY_ERR_NONE) return sts;

    return sts;
}

RGYFilterPad::RGYFilterPad(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_pad() {
    m_name = _T("pad");
}

RGYFilterPad::~RGYFilterPad() {
    close();
}

RGY_ERR RGYFilterPad::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamPad>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    //パラメータチェック
    if (RGY_CSP_CHROMA_FORMAT[prm->frameIn.csp] == RGY_CHROMAFMT_YUV420
        && (prm->pad.left   % 2 != 0
         || prm->pad.top    % 2 != 0
         || prm->pad.right  % 2 != 0
         || prm->pad.bottom % 2 != 0)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, --vpp-pad only supports values which is multiple of 2 in YUV420.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->frameOut.width != pParam->frameIn.width + prm->pad.right + prm->pad.left
        || pParam->frameOut.height != pParam->frameIn.height + prm->pad.top + prm->pad.bottom) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (RGY_CSP_CHROMA_FORMAT[prm->encoderCsp] == RGY_CHROMAFMT_YUV420
        && (pParam->frameOut.width  % 2 != 0
         || pParam->frameOut.height % 2 != 0)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, output resolution must be multiple of 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamPad>(m_param);
    if (!m_pad.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        const auto options = strsprintf("-D Type=%s", RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar");
        m_pad.set(m_cl->buildResourceAsync(_T("RGY_FILTER_PAD_CL"), _T("EXE_DATA"), options.c_str()));
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    //コピーを保存
    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterPad::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        return sts;
    }
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
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    if (!m_pad.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_PAD_CL(m_pad)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    sts = procFrame(ppOutputFrames[0], pInputFrame, queue, wait_events, event);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at denoiseFrame (%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }

    return sts;
}

void RGYFilterPad::close() {
    m_frameBuf.clear();
    m_pad.clear();
    m_cl.reset();
    m_bInterlacedWarn = false;
}
