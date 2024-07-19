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

#include "rgy_filter_decomb.h"

static const int DECOMB_BLOCK_X = 32; //work groupサイズ(x) = スレッド数/work group
static const int DECOMB_BLOCK_Y = 8; //work groupサイズ(y) = スレッド数/work group

static const int BOX_X_LOG2 = 2;
static const int BOX_Y_LOG2 = 3;

static const int DECOMB_BLOCK_IS_COMBED_X = 1 << BOX_Y_LOG2;
static const int DECOMB_BLOCK_IS_COMBED_Y = 32;

RGY_ERR RGYFilterDecomb::createMotionMap(
    RGYFrameInfo *pDmaskPlane,
    RGYFrameInfo *pFmaskPlane,
    const RGYFrameInfo *pSrcPlane,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDecomb>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const float thre_mul = RGY_CSP_BIT_DEPTH[pSrcPlane->csp] > 8 ? 65535.0f : 1.0f;
    const float thre = (float)prm->decomb.threshold * thre_mul;
    const float dthre = (float)prm->decomb.dthreshold * thre_mul;
    const char *kernel_name = "kernel_motion_map";
    RGYWorkSize local(DECOMB_BLOCK_X, DECOMB_BLOCK_Y);
    RGYWorkSize global(pSrcPlane->width, pSrcPlane->height);
    auto err = m_decomb.get()->kernel(kernel_name).config(queue, local, global, wait_events, nullptr).launch(
        (cl_mem)pDmaskPlane->ptr[0],
        (cl_mem)pFmaskPlane->ptr[0], pDmaskPlane->pitch[0],
        (cl_mem)pSrcPlane->ptr[0], pSrcPlane->pitch[0],
        pSrcPlane->width, pSrcPlane->height,
        thre, dthre);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (createMotionMap(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pSrcPlane->csp], get_err_mes(err));
        return err;
    }
    return err;
}

RGY_ERR RGYFilterDecomb::isCombed(
    RGYCLBuf *pResultIsCombed,
    const RGYFrameInfo *pFmaskPlane,
    RGYOpenCLQueue &queue) {
    static const int CT = 15;

    int value = 0;
    auto sts = m_cl->setBuf(&value, sizeof(int), sizeof(int), pResultIsCombed, queue);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    const char *kernel_name = "kernel_is_combed";
    RGYWorkSize local(DECOMB_BLOCK_IS_COMBED_X, DECOMB_BLOCK_IS_COMBED_Y);
    RGYWorkSize global(pFmaskPlane->width, pFmaskPlane->height);
    auto err = m_decomb.get()->kernel(kernel_name).config(queue, local, global, {}, nullptr).launch(
        pResultIsCombed->mem(),
        (cl_mem)pFmaskPlane->ptr[0], pFmaskPlane->pitch[0],
        pFmaskPlane->width, pFmaskPlane->height, CT);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (isCombed(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pFmaskPlane->csp], get_err_mes(err));
        return err;
    }
    return err;
}

RGY_ERR RGYFilterDecomb::deinterlacePlane(
    RGYFrameInfo *pDstPlane,
    const RGYFrameInfo *pSrcPlane,
    const RGYFrameInfo *pDmaskPlane,
    const RGYCLBuf *pResultIsCombed,
    const bool uv420, RGYOpenCLQueue &queue, RGYOpenCLEvent *event
) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDecomb>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const char *kernel_name = "kernel_deinterlace";
    RGYWorkSize local(DECOMB_BLOCK_X, DECOMB_BLOCK_Y);
    RGYWorkSize global(pSrcPlane->width, pSrcPlane->height);
    auto err = m_decomb.get()->kernel(kernel_name).config(queue, local, global, {}, event).launch(
        (cl_mem)pDstPlane->ptr[0],
        (cl_mem)pSrcPlane->ptr[0], pSrcPlane->pitch[0],
        pSrcPlane->width, pSrcPlane->height,
        (cl_mem)pDmaskPlane->ptr[0], pDmaskPlane->pitch[0],
        pResultIsCombed->mem(), uv420 ? 1 : 0);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (createMotionMap(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pSrcPlane->csp], get_err_mes(err));
        return err;
    }
    return err;
}

RGY_ERR RGYFilterDecomb::procFrame(
    RGYFrameInfo * pOutputFrame,
    RGYFrameInfo * pDmaskFrame,
    RGYFrameInfo * pFmaskFrame,
    RGYCLBuf * pResultIsCombed,
    const RGYFrameInfo * pSrcFrame,
    RGYOpenCLQueue & queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent * event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDecomb>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = createMotionMap(pDmaskFrame, pFmaskFrame, pSrcFrame, queue, wait_events);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    if (!prm->decomb.full) {
        sts = isCombed(pResultIsCombed, pFmaskFrame, queue);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }

    for (int iplane = 0; iplane < RGY_CSP_PLANES[pSrcFrame->csp]; iplane++) {
        const auto planeSrc = getPlane(pSrcFrame, (RGY_PLANE)iplane);
        auto planeOutput = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        RGYOpenCLEvent *plane_event = (iplane == RGY_CSP_PLANES[pOutputFrame->csp] - 1) ? event : nullptr;
        sts = deinterlacePlane(&planeOutput, &planeSrc, pDmaskFrame, pResultIsCombed,
            iplane > 0 && RGY_CSP_CHROMA_FORMAT[pSrcFrame->csp] == RGY_CHROMAFMT_YUV420, queue, plane_event);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterDecomb::RGYFilterDecomb(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_decomb(), m_dmask(), m_fmask(), m_isCombed() {
    m_name = _T("decomb");
}

RGYFilterDecomb::~RGYFilterDecomb() {
    close();
}

RGY_ERR RGYFilterDecomb::checkParam(const std::shared_ptr<RGYFilterParamDecomb> prm) {
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int hight_mul = (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp] == RGY_CHROMAFMT_YUV420) ? 4 : 2;
    if ((prm->frameOut.height % hight_mul) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Height must be multiple of %d.\n"), hight_mul);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->decomb.threshold < 0 && 255 <= prm->decomb.threshold) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (threshold).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->decomb.dthreshold < 0 && 255 <= prm->decomb.dthreshold) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (dthreshold).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDecomb::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDecomb>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if ((sts = checkParam(prm)) != RGY_ERR_NONE) {
        return sts;
    }
    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamDecomb>(m_param);
    if (!m_decomb.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
    }
    if (!m_decomb.get()
        || !prmPrev
        || cmpFrameInfoCspResolution(&prmPrev->frameOut, &pParam->frameOut)
        || prmPrev->decomb.full != prm->decomb.full) {
        const auto options = strsprintf("-D TypePixel=%s -D TypeMask=%s -D TypeMaskVec=%s"
            " -D DECOMB_BLOCK_IS_COMBED_X=%d -D DECOMB_BLOCK_IS_COMBED_Y=%d"
            " -D BOX_X_LOG2=%d -D BOX_Y_LOG2=%d -D full=%d -D blend=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            "uchar", "uchar4",
            DECOMB_BLOCK_IS_COMBED_X, DECOMB_BLOCK_IS_COMBED_Y,
            BOX_X_LOG2, BOX_Y_LOG2,
            prm->decomb.full ? 1 : 0,
            prm->decomb.blend ? 1 : 0);
        m_decomb.set(m_cl->buildResourceAsync(_T("RGY_FILTER_DECOMB_CL"), _T("EXE_DATA"), options.c_str()));

        auto err = AllocFrameBuf(prm->frameOut, 1);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
            return RGY_ERR_MEMORY_ALLOC;
        }
        for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
            prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
        }
    }

    if (!m_dmask
        || m_dmask->width() != prm->frameOut.width
        || m_dmask->height() != prm->frameOut.height) {
        auto frame = prm->frameOut;
        frame.csp = RGY_CSP_Y8;
        m_dmask = m_cl->createFrameBuffer(frame);
        if (!m_dmask) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }
    if (!m_fmask
        || m_fmask->width() != prm->frameOut.width
        || m_fmask->height() != prm->frameOut.height) {
        auto frame = prm->frameOut;
        frame.csp = RGY_CSP_Y8;
        m_fmask = m_cl->createFrameBuffer(frame);
        if (!m_fmask) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }
    if (!m_isCombed) {
        m_isCombed = m_cl->createBuffer(sizeof(int));
        if (!m_isCombed) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    m_pathThrough &= (~(FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS));

    //コピーを保存
    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterDecomb::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        return RGY_ERR_NONE;
    }
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDecomb>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!m_decomb.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_DECOMB_CL(m_decomb)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }


    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
    }

    if (interlaced(*pInputFrame)) {
        sts = procFrame(ppOutputFrames[0],
            &m_dmask->frame, &m_fmask->frame, m_isCombed.get(), pInputFrame, queue, wait_events, event);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to deinterlace frame: %s.\n"), get_err_mes(sts));
            return sts;
        }
    } else {
        //ppOutputFrames[0]にコピー
        sts = m_cl->copyFrame(ppOutputFrames[0], pInputFrame, nullptr, queue, wait_events, event);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy frame: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }
    ppOutputFrames[0]->picstruct = RGY_PICSTRUCT_FRAME;

    return sts;
}

void RGYFilterDecomb::close() {
    m_frameBuf.clear();
    m_decomb.clear();
    m_cl.reset();
}
