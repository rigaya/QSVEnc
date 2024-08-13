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

#include "rgy_filter_yadif.h"

static const int YADIF_BLOCK_X = 32;
static const int YADIF_BLOCK_Y = 8;

RGYFilterYadifSource::RGYFilterYadifSource(std::shared_ptr<RGYOpenCLContext> cl) : m_cl(cl), m_nFramesInput(0), m_nFramesOutput(0), m_buf() {

}

RGYFilterYadifSource::~RGYFilterYadifSource() {
    clear();
}

void RGYFilterYadifSource::clear() {
    for (auto& buf : m_buf) {
        buf.reset();
    }
    m_nFramesInput = 0;
    m_nFramesOutput = 0;
}

RGY_ERR RGYFilterYadifSource::alloc(const RGYFrameInfo& frameInfo) {
    if (m_buf.begin()->get() != nullptr
        && !cmpFrameInfoCspResolution(&m_buf.begin()->get()->frame, &frameInfo)) {
        //すべて確保されているか確認
        bool allocated = true;
        for (auto& buf : m_buf) {
            if (buf->frame.ptr[0] == nullptr) {
                allocated = false;
                break;
            }
        }
        if (allocated) {
            return RGY_ERR_NONE;
        }
    }
    for (auto& buf : m_buf) {
        buf = m_cl->createFrameBuffer(frameInfo.width, frameInfo.height, frameInfo.csp, frameInfo.bitdepth);
        if (!buf) {
            return RGY_ERR_NULL_PTR;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterYadifSource::add(const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue) {
    const int iframe = m_nFramesInput++;
    auto pDstFrame = get(iframe);
    auto err = m_cl->copyFrame(&pDstFrame->frame, pInputFrame, nullptr, queue);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    copyFrameProp(&pDstFrame->frame, pInputFrame);
    return err;
}

RGY_ERR RGYFilterYadif::procPlane(
    RGYFrameInfo *pOutputPlane,
    const RGYFrameInfo *pInputPlane0,
    const RGYFrameInfo *pInputPlane1,
    const RGYFrameInfo *pInputPlane2,
    const YadifTargetField targetField,
    const RGY_PICSTRUCT picstruct,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *kernel_name = "kernel_yadif";
    RGYWorkSize local(YADIF_BLOCK_X, YADIF_BLOCK_Y);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    auto err = m_yadif.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0],
        pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (cl_mem)pInputPlane0->ptr[0], (cl_mem)pInputPlane1->ptr[0], (cl_mem)pInputPlane2->ptr[0],
        pInputPlane1->pitch[0], pInputPlane1->width, pInputPlane1->height,
        targetField, picstruct);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pOutputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterYadif::procFrame(
    RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pInputFrame0,
    const RGYFrameInfo *pInputFrame1,
    const RGYFrameInfo *pInputFrame2,
    const YadifTargetField targetField,
    const RGY_PICSTRUCT picstruct,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamYadif>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        const auto planeSrc0 = getPlane(pInputFrame0, (RGY_PLANE)i);
        const auto planeSrc1 = getPlane(pInputFrame1, (RGY_PLANE)i);
        const auto planeSrc2 = getPlane(pInputFrame2, (RGY_PLANE)i);
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[pOutputFrame->csp] - 1) ? event : nullptr;
        auto err = procPlane(&planeDst, &planeSrc0, &planeSrc1, &planeSrc2, targetField, picstruct, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to process yadif frame(%d) %s: %s\n"), i, cl_errmes(err));
            return err_cl_to_rgy(err);
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterYadif::RGYFilterYadif(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_yadif(), m_nFrame(0), m_pts(0), m_source(context) {
    m_name = _T("yadif");
}

RGYFilterYadif::~RGYFilterYadif() {
    close();
}

RGY_ERR RGYFilterYadif::checkParam(const std::shared_ptr<RGYFilterParamYadif> prm) {
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
    if (prm->yadif.mode >= VPP_YADIF_MODE_MAX) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (mode).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterYadif::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamYadif>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if ((sts = checkParam(prm)) != RGY_ERR_NONE) {
        return sts;
    }
    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamYadif>(m_param);
    if (!m_yadif.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d"
            " -D YADIF_GEN_FIELD_TOP=%d -D YADIF_GEN_FIELD_BOTTOM=%d -D RGY_PICSTRUCT_TFF=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp],
            YADIF_GEN_FIELD_TOP, YADIF_GEN_FIELD_BOTTOM, RGY_PICSTRUCT_TFF);
        m_yadif.set(m_cl->buildResourceAsync(_T("RGY_FILTER_YADIF_CL"), _T("EXE_DATA"), options.c_str()));
    }
    if (!prmPrev
        || cmpFrameInfoCspResolution(&prmPrev->frameOut, &pParam->frameOut)) {
        m_source.alloc(pParam->frameOut);
        auto err = AllocFrameBuf(prm->frameOut, 2);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
            return RGY_ERR_MEMORY_ALLOC;
        }
        for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
            prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
        }
    }


    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    m_nFrame = 0;
    m_pts = 0;
    m_pathThrough &= (~(FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_TIMESTAMP));
    if (prm->yadif.mode & VPP_YADIF_MODE_BOB) {
        prm->baseFps *= 2;
    }

    //コピーを保存
    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterYadif::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;

    auto prm = std::dynamic_pointer_cast<RGYFilterParamYadif>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!m_yadif.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_YADIF_CL(m_yadif)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }

    const int iframe = m_source.inframe();
    if (pInputFrame->ptr[0] == nullptr && m_nFrame >= iframe) {
        //終了
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return sts;
    } else if (pInputFrame->ptr[0] != nullptr) {
        //エラーチェック
        if (prm->frameOut.csp != prm->frameIn.csp) {
            AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        //sourceキャッシュにコピー
        auto err = m_source.add(pInputFrame, queue);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to add frame to source buffer: %s.\n"), get_err_mes(err));
            return err;
        }
    }

    //十分な数のフレームがたまった、あるいはdrainモードならフレームを出力
    if (iframe >= 1 || pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
        //出力先のフレーム
        RGYCLFrame *pOutFrame = nullptr;
        *pOutputFrameNum = 1;
        if (ppOutputFrames[0] == nullptr) {
            pOutFrame = m_frameBuf[0].get();
            ppOutputFrames[0] = &pOutFrame->frame;
            ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
            if (prm->yadif.mode & VPP_YADIF_MODE_BOB) {
                pOutFrame = m_frameBuf[1].get();
                ppOutputFrames[1] = &pOutFrame->frame;
                ppOutputFrames[1]->picstruct = pInputFrame->picstruct;
                *pOutputFrameNum = 2;
            }
        }

        const auto *const pSourceFrame = &m_source.get(m_nFrame)->frame;
        pOutFrame->frame.flags = pSourceFrame->flags & (~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_BFF | RGY_FRAME_FLAG_RFF_TFF));

        YadifTargetField targetField = YADIF_GEN_FIELD_UNKNOWN;
        if (prm->yadif.mode & VPP_YADIF_MODE_AUTO) {
            //エラーチェック
            const auto memcpyKind = getMemcpyKind(pSourceFrame->mem_type, ppOutputFrames[0]->mem_type);
            if (memcpyKind != RGYCLMemcpyD2D) {
                AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
                return RGY_ERR_INVALID_CALL;
            }
            if ((pSourceFrame->picstruct & RGY_PICSTRUCT_INTERLACED) == 0) {
                ppOutputFrames[0]->picstruct = RGY_PICSTRUCT_FRAME;
                ppOutputFrames[0]->timestamp = pSourceFrame->timestamp;
                ppOutputFrames[0]->inputFrameId = pSourceFrame->inputFrameId;
                m_cl->copyFrame(ppOutputFrames[0], pSourceFrame, nullptr, queue);
                if (prm->yadif.mode & VPP_YADIF_MODE_BOB) {
                    m_cl->copyFrame(ppOutputFrames[1], pSourceFrame, nullptr, queue);
                    setBobTimestamp(iframe, ppOutputFrames);
                }
                m_nFrame++;
                return RGY_ERR_NONE;
            } else if ((pSourceFrame->picstruct & RGY_PICSTRUCT_FRAME_TFF) == RGY_PICSTRUCT_FRAME_TFF) {
                targetField = YADIF_GEN_FIELD_BOTTOM;
            } else if ((pSourceFrame->picstruct & RGY_PICSTRUCT_FRAME_BFF) == RGY_PICSTRUCT_FRAME_BFF) {
                targetField = YADIF_GEN_FIELD_TOP;
            }
        } else if (prm->yadif.mode & VPP_YADIF_MODE_TFF) {
            targetField = YADIF_GEN_FIELD_BOTTOM;
        } else if (prm->yadif.mode & VPP_YADIF_MODE_BFF) {
            targetField = YADIF_GEN_FIELD_TOP;
        } else {
            AddMessage(RGY_LOG_ERROR, _T("Not implemented yet.\n"));
            return RGY_ERR_INVALID_PARAM;
        }

        auto err = procFrame(ppOutputFrames[0],
            &m_source.get(m_nFrame - 1)->frame,
            &m_source.get(m_nFrame + 0)->frame,
            &m_source.get(m_nFrame + 1)->frame,
            targetField,
            pSourceFrame->picstruct,
            queue, wait_events, event
            );
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to proc frame: %s.\n"), get_err_mes(err));
            return err;
        }

        ppOutputFrames[0]->picstruct = RGY_PICSTRUCT_FRAME;
        ppOutputFrames[0]->timestamp = pSourceFrame->timestamp;
        ppOutputFrames[0]->inputFrameId = pSourceFrame->inputFrameId;
        if (prm->yadif.mode & VPP_YADIF_MODE_BOB) {
            targetField = (targetField == YADIF_GEN_FIELD_BOTTOM) ? YADIF_GEN_FIELD_TOP : YADIF_GEN_FIELD_BOTTOM;
            err = procFrame(ppOutputFrames[1],
                &m_source.get(m_nFrame - 1)->frame,
                &m_source.get(m_nFrame + 0)->frame,
                &m_source.get(m_nFrame + 1)->frame,
                targetField,
                pSourceFrame->picstruct,
                queue, wait_events, event
                );
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to proc frame (2nd field): %s.\n"), get_err_mes(err));
                return err;
            }
            setBobTimestamp(iframe, ppOutputFrames);
        }
        m_nFrame++;
    } else {
        //出力フレームなし
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
    }
    return sts;
}

void RGYFilterYadif::setBobTimestamp(const int iframe, RGYFrameInfo **ppOutputFrames) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamYadif>(m_param);

    auto frameDuration = m_source.get(m_nFrame + 0)->frame.duration;
    if (frameDuration == 0) {
        if (iframe <= 1) {
            frameDuration = (decltype(frameDuration))((prm->timebase.inv() / prm->baseFps * 2).qdouble() + 0.5);
        } else if (m_nFrame + 1 >= iframe) {
            frameDuration = m_source.get(m_nFrame + 0)->frame.timestamp - m_source.get(m_nFrame - 1)->frame.timestamp;
        } else {
            frameDuration = m_source.get(m_nFrame + 1)->frame.timestamp - m_source.get(m_nFrame + 0)->frame.timestamp;
        }
    }
    ppOutputFrames[1]->picstruct = RGY_PICSTRUCT_FRAME;
    ppOutputFrames[0]->timestamp = m_source.get(m_nFrame + 0)->frame.timestamp;
    ppOutputFrames[0]->duration = (frameDuration + 1) / 2;
    ppOutputFrames[1]->timestamp = ppOutputFrames[0]->timestamp + ppOutputFrames[0]->duration;
    ppOutputFrames[1]->duration = frameDuration - ppOutputFrames[0]->duration;
    ppOutputFrames[1]->inputFrameId = m_source.get(m_nFrame + 0)->frame.inputFrameId;
}

void RGYFilterYadif::close() {
    m_frameBuf.clear();
    m_yadif.clear();
    m_cl.reset();
    m_nFrame = 0;
    m_pts = 0;
}
