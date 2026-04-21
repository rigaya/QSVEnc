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
#include "convert_csp.h"
#include "rgy_filter_bwdif.h"

static const int BWDIF_BLOCK_X    = 32;
static const int BWDIF_BLOCK_Y    = 8;
static const int BWDIF_CACHE_SIZE = 3;

RGYFilterBwdif::RGYFilterBwdif(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_bwdif(),
    m_bwdifBuildOptions(),
    m_cacheFrames(),
    m_inputCount(0),
    m_drained(false),
    m_defaultTff(true) {
    m_name = _T("bwdif");
}

RGYFilterBwdif::~RGYFilterBwdif() {
    close();
}

RGY_ERR RGYFilterBwdif::checkParam(const std::shared_ptr<RGYFilterParamBwdif> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int height_mul = (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp] == RGY_CHROMAFMT_YUV420) ? 4 : 2;
    if ((prm->frameOut.height % height_mul) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Height must be multiple of %d.\n"), height_mul);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->bwdif.thr < 0.0f || prm->bwdif.thr > 100.0f) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid thr=%.3f: must be in [0.0, 100.0].\n"), prm->bwdif.thr);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterBwdif::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamBwdif>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    m_pathThrough &= ~(FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_TIMESTAMP);
    if (prm->bwdif.isbob()) {
        pParam->baseFps *= 2;
    }

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamBwdif>(m_param);
    if (!prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
        const int maxVal   = (1 << bitDepth) - 1;
        m_bwdifBuildOptions = strsprintf("-D Type=%s -D bit_depth=%d -D max_val=%d -D bwdif_block_x=%d -D bwdif_block_y=%d",
            bitDepth > 8 ? "ushort" : "uchar",
            bitDepth, maxVal,
            BWDIF_BLOCK_X, BWDIF_BLOCK_Y);
        AddMessage(RGY_LOG_DEBUG, _T("Starting async build for RGY_FILTER_BWDIF_CL: %s\n"),
            char_to_tstring(m_bwdifBuildOptions).c_str());
        m_bwdif.set(m_cl->buildResourceAsync(_T("RGY_FILTER_BWDIF_CL"), _T("EXE_DATA"), m_bwdifBuildOptions.c_str()));
    }

    sts = AllocFrameBuf(prm->frameOut, prm->bwdif.isbob() ? 2 : 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    if ((int)m_cacheFrames.size() != BWDIF_CACHE_SIZE
        || !prmPrev
        || cmpFrameInfoCspResolution(&m_cacheFrames[0]->frame, &prm->frameIn)) {
        m_cacheFrames.clear();
        for (int i = 0; i < BWDIF_CACHE_SIZE; i++) {
            auto clframe = m_cl->createFrameBuffer(prm->frameIn);
            if (!clframe) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate cache frame %d.\n"), i);
                return RGY_ERR_MEMORY_ALLOC;
            }
            m_cacheFrames.push_back(std::move(clframe));
        }
    }

    m_defaultTff = (prm->frameIn.picstruct & RGY_PICSTRUCT_BFF) == 0;

    if (!prmPrev || prmPrev->bwdif != prm->bwdif) {
        m_inputCount = 0;
        m_drained = false;
    }

    setFilterInfo(prm->print() + _T("\n                         auto-order-fallback=") + (m_defaultTff ? _T("tff") : _T("bff")));
    m_param = prm;
    return sts;
}

bool RGYFilterBwdif::getInputTff(const RGYFrameInfo *frame) const {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamBwdif>(m_param);
    if (!prm) {
        return m_defaultTff;
    }
    if (prm->bwdif.order == VppBwdifOrder::TFF) {
        return true;
    }
    if (prm->bwdif.order == VppBwdifOrder::BFF) {
        return false;
    }
    if (frame) {
        if (frame->picstruct & RGY_PICSTRUCT_BFF) {
            return false;
        }
        if (frame->picstruct & RGY_PICSTRUCT_TFF) {
            return true;
        }
    }
    return m_defaultTff;
}

bool RGYFilterBwdif::shouldPassthrough(const RGYFrameInfo *frame) const {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamBwdif>(m_param);
    return prm
        && prm->bwdif.order == VppBwdifOrder::Auto
        && frame
        && (frame->picstruct & RGY_PICSTRUCT_INTERLACED) == 0;
}

RGY_ERR RGYFilterBwdif::reconstructFrame(int idx_prev, int idx_cur, int idx_next,
    bool inputTff, int preserveTopField, int outputSlot,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamBwdif>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const bool xorFlag = (preserveTopField != (inputTff ? 1 : 0));
    const int idx_prev2 = xorFlag ? idx_prev : idx_cur;
    const int idx_next2 = xorFlag ? idx_cur  : idx_next;

    const RGYFrameInfo *prev2 = &m_cacheFrames[idx_prev2]->frame;
    const RGYFrameInfo *prev  = &m_cacheFrames[idx_prev ]->frame;
    const RGYFrameInfo *cur   = &m_cacheFrames[idx_cur  ]->frame;
    const RGYFrameInfo *next  = &m_cacheFrames[idx_next ]->frame;
    const RGYFrameInfo *next2 = &m_cacheFrames[idx_next2]->frame;

    const int bitDepth = RGY_CSP_BIT_DEPTH[cur->csp];
    const int maxVal   = (1 << bitDepth) - 1;
    const int thr      = (int)(prm->bwdif.thr * (float)maxVal / 100.0f + 0.5f);

    RGYFrameInfo *dst = &m_frameBuf[outputSlot]->frame;
    const char *kernel_name = "kernel_bwdif_frame";
    const int planes = RGY_CSP_PLANES[cur->csp];

    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dPlane  = getPlane(dst,   (RGY_PLANE)iplane);
        const auto p2Plane = getPlane(prev2, (RGY_PLANE)iplane);
        const auto pPlane  = getPlane(prev,  (RGY_PLANE)iplane);
        const auto cPlane  = getPlane(cur,   (RGY_PLANE)iplane);
        const auto nPlane  = getPlane(next,  (RGY_PLANE)iplane);
        const auto n2Plane = getPlane(next2, (RGY_PLANE)iplane);

        RGYWorkSize local(BWDIF_BLOCK_X, BWDIF_BLOCK_Y);
        RGYWorkSize global(dPlane.width, dPlane.height);
        const auto &waitHere = (iplane == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        auto err = m_bwdif.get()->kernel(kernel_name).config(queue, local, global, waitHere, nullptr).launch(
            (cl_mem)dPlane.ptr[0], dPlane.pitch[0],
            (cl_mem)p2Plane.ptr[0],
            (cl_mem)pPlane.ptr[0],
            (cl_mem)cPlane.ptr[0],
            (cl_mem)nPlane.ptr[0],
            (cl_mem)n2Plane.ptr[0],
            cPlane.pitch[0],
            dPlane.width, dPlane.height,
            preserveTopField,
            thr);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                char_to_tstring(kernel_name).c_str(), iplane, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterBwdif::generateOutput(int idx_prev, int idx_cur, int idx_next,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamBwdif>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const bool bob = prm->bwdif.isbob();
    const RGYFrameInfo *curF = &m_cacheFrames[idx_cur]->frame;
    const bool inputTff = getInputTff(curF);
    const int firstFieldParity  = inputTff ? 1 : 0;
    const int secondFieldParity = inputTff ? 0 : 1;

    if (shouldPassthrough(curF)) {
        auto err = m_cl->copyFrame(&m_frameBuf[0]->frame, curF, nullptr, queue, wait_events);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy progressive frame: %s.\n"), get_err_mes(err));
            return err;
        }
        auto pOut0 = &m_frameBuf[0]->frame;
        pOut0->picstruct = RGY_PICSTRUCT_FRAME;
        pOut0->flags = curF->flags;
        ppOutputFrames[0] = pOut0;
        *pOutputFrameNum = 1;
        if (bob) {
            err = m_cl->copyFrame(&m_frameBuf[1]->frame, curF, nullptr, queue, {});
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to duplicate progressive frame for bob: %s.\n"), get_err_mes(err));
                return err;
            }
            auto pOut1 = &m_frameBuf[1]->frame;
            pOut1->picstruct = RGY_PICSTRUCT_FRAME;
            pOut1->flags = curF->flags;
            ppOutputFrames[1] = pOut1;
            *pOutputFrameNum = 2;
            setBobTimestamp(curF, ppOutputFrames);
        } else {
            pOut0->timestamp = curF->timestamp;
            pOut0->duration = curF->duration;
            pOut0->inputFrameId = curF->inputFrameId;
        }
        return RGY_ERR_NONE;
    }

    auto err = reconstructFrame(idx_prev, idx_cur, idx_next, inputTff, firstFieldParity, 0, queue, wait_events);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    auto pOut0 = &m_frameBuf[0]->frame;
    pOut0->picstruct = RGY_PICSTRUCT_FRAME;
    pOut0->flags     = curF->flags;
    ppOutputFrames[0] = pOut0;
    *pOutputFrameNum  = 1;

    if (bob) {
        err = reconstructFrame(idx_prev, idx_cur, idx_next, inputTff, secondFieldParity, 1, queue, {});
        if (err != RGY_ERR_NONE) {
            return err;
        }
        auto pOut1 = &m_frameBuf[1]->frame;
        pOut1->picstruct = RGY_PICSTRUCT_FRAME;
        pOut1->flags     = curF->flags;
        ppOutputFrames[1] = pOut1;
        *pOutputFrameNum  = 2;
        setBobTimestamp(curF, ppOutputFrames);
    } else {
        pOut0->timestamp    = curF->timestamp;
        pOut0->duration     = curF->duration;
        pOut0->inputFrameId = curF->inputFrameId;
    }
    return RGY_ERR_NONE;
}

void RGYFilterBwdif::setBobTimestamp(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamBwdif>(m_param);
    auto frameDuration = pInputFrame->duration;
    if (frameDuration == 0 && prm && prm->timebase.is_valid()) {
        frameDuration = (decltype(frameDuration))((prm->timebase.inv() / prm->baseFps * 2).qdouble() + 0.5);
    }
    ppOutputFrames[0]->timestamp    = pInputFrame->timestamp;
    ppOutputFrames[0]->duration     = (frameDuration + 1) / 2;
    ppOutputFrames[1]->timestamp    = ppOutputFrames[0]->timestamp + ppOutputFrames[0]->duration;
    ppOutputFrames[1]->duration     = frameDuration - ppOutputFrames[0]->duration;
    ppOutputFrames[0]->inputFrameId = pInputFrame->inputFrameId;
    ppOutputFrames[1]->inputFrameId = pInputFrame->inputFrameId;
}

RGY_ERR RGYFilterBwdif::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue_main,
    const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    (void)event;
    *pOutputFrameNum  = 0;
    ppOutputFrames[0] = nullptr;
    ppOutputFrames[1] = nullptr;

    if (!m_bwdif.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build/load RGY_FILTER_BWDIF_CL (options: %s).\n"),
            char_to_tstring(m_bwdifBuildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamBwdif>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const bool hasInput = (pInputFrame && pInputFrame->ptr[0]);

    if (hasInput) {
        const int slot = m_inputCount % BWDIF_CACHE_SIZE;
        RGYFrameInfo *pSlot = &m_cacheFrames[slot]->frame;
        auto copyErr = m_cl->copyFrame(pSlot, pInputFrame, nullptr, queue_main, wait_events);
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy input to cache slot %d: %s.\n"), slot, get_err_mes(copyErr));
            return copyErr;
        }
        pSlot->timestamp    = pInputFrame->timestamp;
        pSlot->duration     = pInputFrame->duration;
        pSlot->inputFrameId = pInputFrame->inputFrameId;
        pSlot->picstruct    = pInputFrame->picstruct;
        pSlot->flags        = pInputFrame->flags;

        m_inputCount++;
        if (m_inputCount < 2) {
            return RGY_ERR_NONE;
        }

        const int idx_cur  = (m_inputCount - 2) % BWDIF_CACHE_SIZE;
        const int idx_next = (m_inputCount - 1) % BWDIF_CACHE_SIZE;
        const int idx_prev = (m_inputCount >= 3) ? (m_inputCount - 3) % BWDIF_CACHE_SIZE : idx_cur;
        return generateOutput(idx_prev, idx_cur, idx_next, ppOutputFrames, pOutputFrameNum, queue_main, wait_events);
    }

    if (!m_drained && m_inputCount >= 1) {
        m_drained = true;
        const int idx_cur  = (m_inputCount - 1) % BWDIF_CACHE_SIZE;
        const int idx_next = idx_cur;
        const int idx_prev = (m_inputCount >= 2) ? (m_inputCount - 2) % BWDIF_CACHE_SIZE : idx_cur;
        return generateOutput(idx_prev, idx_cur, idx_next, ppOutputFrames, pOutputFrameNum, queue_main, wait_events);
    }

    return RGY_ERR_NONE;
}

void RGYFilterBwdif::close() {
    m_bwdif.clear();
    m_bwdifBuildOptions.clear();
    m_cacheFrames.clear();
    m_inputCount = 0;
    m_drained    = false;
    m_frameBuf.clear();
    m_cl.reset();
}
