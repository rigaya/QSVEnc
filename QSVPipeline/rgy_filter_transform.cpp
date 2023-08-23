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
#include <map>
#include <array>
#include "rgy_filter_transform.h"

static const int TRASNPOSE_BLOCK_DIM = 16;
static const int TRASNPOSE_TILE_DIM = 64;
static_assert((TRASNPOSE_TILE_DIM % TRASNPOSE_BLOCK_DIM) == 0, "Invalid TRASNPOSE_TILE_DIM");
static_assert(TRASNPOSE_TILE_DIM / TRASNPOSE_BLOCK_DIM > 0, "Invalid TRASNPOSE_TILE_DIM");

static const int FLIP_BLOCK_DIM = 16;

RGY_ERR RGYFilterTransform::procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamTransform>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->trans.transpose) {
        const char *kernel_name = "kernel_transpose_plane";
        RGYWorkSize local(TRASNPOSE_BLOCK_DIM, TRASNPOSE_BLOCK_DIM);
        RGYWorkSize global(
            divCeil(pOutputPlane->width,  TRASNPOSE_TILE_DIM / TRASNPOSE_BLOCK_DIM),
            divCeil(pOutputPlane->height, TRASNPOSE_TILE_DIM / TRASNPOSE_BLOCK_DIM));
        auto err = m_transform.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
            (cl_mem)pOutputPlane->ptr[0],
            pOutputPlane->pitch[0],
            pOutputPlane->width,  // = srcHeight
            pOutputPlane->height, // = srcWidth
            (cl_mem)pInputPlane->ptr[0],
            pInputPlane->pitch[0]);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    } else {
        const char *kernel_name = "kernel_flip_plane";
        RGYWorkSize local(FLIP_BLOCK_DIM, FLIP_BLOCK_DIM);
        RGYWorkSize global(divCeil(pOutputPlane->width, 4), pOutputPlane->height);
        auto err = m_transform.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
            (cl_mem)pOutputPlane->ptr[0],
            pOutputPlane->pitch[0],
            pOutputPlane->width,
            pOutputPlane->height,
            (cl_mem)pInputPlane->ptr[0],
            pInputPlane->pitch[0]);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterTransform::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[pOutputFrame->csp] - 1) ? event : nullptr;
        auto err = procPlane(&planeDst, &planeSrc, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to transform frame(%d) %s: %s\n"), i, cl_errmes(err));
            return err_cl_to_rgy(err);
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterTransform::RGYFilterTransform(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_transform() {
    m_name = _T("transform");
}

RGYFilterTransform::~RGYFilterTransform() {
    close();
}

RGY_ERR RGYFilterTransform::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamTransform>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->trans.transpose) {
        prm->frameOut.width = prm->frameIn.height;
        prm->frameOut.height = prm->frameIn.width;
    }
    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamTransform>(m_param);
    if (!m_transform.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
        || prmPrev->trans.flipX != prm->trans.flipX
        || prmPrev->trans.flipY != prm->trans.flipY) {
        const auto options = strsprintf("-D TypePixel=%s -D TypePixel4=%s -D flipX=%d -D flipY=%d -D FLIP_BLOCK_DIM=%d -D TRASNPOSE_BLOCK_DIM=%d -D TRASNPOSE_TILE_DIM=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort"  : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort4" : "uchar4",
            prm->trans.flipX ? 1 : 0,
            prm->trans.flipY ? 1 : 0,
            FLIP_BLOCK_DIM, TRASNPOSE_BLOCK_DIM, TRASNPOSE_TILE_DIM);
        m_transform.set(m_cl->buildResourceAsync(_T("RGY_FILTER_TRANSFORM_CL"), _T("EXE_DATA"), options.c_str()));
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

RGY_ERR RGYFilterTransform::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
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
    //if (interlaced(*pInputFrame)) {
    //    return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
    //}
    if (!m_transform.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_TRANSFORM_CL(m_transform)\n"));
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

void RGYFilterTransform::close() {
    m_frameBuf.clear();
    m_transform.clear();
    m_cl.reset();
    m_bInterlacedWarn = false;
}
