// -----------------------------------------------------------------------------------------
//  QSVEnc/VCEEnc/rkmppenc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2019-2021 rigaya
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
// IABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#include "rgy_filter_cl.h"

RGY_ERR RGYFilterPerfCL::checkPerformace(void *event_start, void *event_fin) {
    uint64_t time_start = 0;
    auto sts = ((RGYOpenCLEvent *)event_start)->getProfilingTimeEnd(time_start);
    if (sts != RGY_ERR_NONE) return sts;
    uint64_t time_end = 0;
    sts = ((RGYOpenCLEvent *)event_fin)->getProfilingTimeStart(time_end);
    if (sts != RGY_ERR_NONE) return sts;
    setTime((time_end - time_start) * 1e-6 /*ns -> ms*/);
    return RGY_ERR_NONE;
}

RGYFilter::RGYFilter(shared_ptr<RGYOpenCLContext> context) :
    RGYFilterBase(),
    m_cl(context),
    m_frameBuf(),
    m_pFieldPairIn(),
    m_pFieldPairOut() {

}

RGYFilter::~RGYFilter() {
    m_frameBuf.clear();
    m_pFieldPairIn.reset();
    m_pFieldPairOut.reset();
    m_param.reset();
}

RGY_ERR RGYFilter::AllocFrameBuf(const RGYFrameInfo &frame, int frames) {
    if ((int)m_frameBuf.size() == frames
        && !cmpFrameInfoCspResolution(&m_frameBuf[0]->frame, &frame)) {
        //すべて確保されているか確認
        bool allocated = true;
        for (size_t i = 0; i < m_frameBuf.size(); i++) {
            for (int iplane = 0; iplane < RGY_CSP_PLANES[m_frameBuf[i]->frame.csp]; iplane++) {
                if (m_frameBuf[i]->frame.ptr[iplane] == nullptr) {
                    allocated = false;
                    break;
                }
            }
        }
        if (allocated) {
            return RGY_ERR_NONE;
        }
    }
    m_frameBuf.clear();

    for (int i = 0; i < frames; i++) {
        auto uptr = m_cl->createFrameBuffer(frame);
        if (!uptr) {
            m_frameBuf.clear();
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_frameBuf.push_back(std::move(uptr));
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilter::filter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    return filter(pInputFrame, ppOutputFrames, pOutputFrameNum, m_cl->queue());
}
RGY_ERR RGYFilter::filter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue& queue) {
    return filter(pInputFrame, ppOutputFrames, pOutputFrameNum, queue, nullptr);
}
RGY_ERR RGYFilter::filter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue& queue, RGYOpenCLEvent *event) {
    return filter(pInputFrame, ppOutputFrames, pOutputFrameNum, queue, {}, event);
}
RGY_ERR RGYFilter::filter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (pInputFrame == nullptr) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
    }
    if (m_param
        && m_param->bOutOverwrite //上書きか?
        && pInputFrame != nullptr && pInputFrame->ptr[0] != nullptr //入力が存在するか?
        && ppOutputFrames != nullptr && ppOutputFrames[0] == nullptr) { //出力先がセット可能か?
        ppOutputFrames[0] = pInputFrame;
        *pOutputFrameNum = 1;
    }
    RGYOpenCLEvent queueRunStart;
    if (m_perfMonitor) {
        queue.getmarker(queueRunStart);
    }
    const auto ret = run_filter(pInputFrame, ppOutputFrames, pOutputFrameNum, queue, wait_events, event);
    const int nOutFrame = *pOutputFrameNum;
    if (!m_param->bOutOverwrite && nOutFrame > 0) {
        if (m_pathThrough & FILTER_PATHTHROUGH_TIMESTAMP) {
            if (nOutFrame != 1) {
                AddMessage(RGY_LOG_ERROR, _T("timestamp path through can only be applied to 1-in/1-out filter.\n"));
                return RGY_ERR_INVALID_CALL;
            } else {
                ppOutputFrames[0]->timestamp = pInputFrame->timestamp;
                ppOutputFrames[0]->duration = pInputFrame->duration;
                ppOutputFrames[0]->inputFrameId = pInputFrame->inputFrameId;
            }
        }
        for (int i = 0; i < nOutFrame; i++) {
            if (m_pathThrough & FILTER_PATHTHROUGH_FLAGS)     ppOutputFrames[i]->flags = pInputFrame->flags;
            if (m_pathThrough & FILTER_PATHTHROUGH_PICSTRUCT) ppOutputFrames[i]->picstruct = pInputFrame->picstruct;
            if (m_pathThrough & FILTER_PATHTHROUGH_DATA)      ppOutputFrames[i]->dataList  = pInputFrame->dataList;
        }
    }
    if (m_perfMonitor) {
        RGYOpenCLEvent queueRunEnd;
        queue.getmarker(queueRunEnd);
        queueRunEnd.wait();
        m_perfMonitor->checkPerformace(&queueRunStart, &queueRunEnd);
    }
    return ret;
}

void RGYFilter::setCheckPerformance(const bool check) {
    if (check) m_perfMonitor = std::make_unique<RGYFilterPerfCL>();
    else       m_perfMonitor.reset();
}

RGY_ERR RGYFilter::filter_as_interlaced_pair(const RGYFrameInfo *pInputFrame, RGYFrameInfo *pOutputFrame) {
#if 0
    if (!m_pFieldPairIn) {
        unique_ptr<CUFrameBuf> uptr(new CUFrameBuf(*pInputFrame));
        uptr->frame.ptr = nullptr;
        uptr->frame.pitch = 0;
        uptr->frame.height >>= 1;
        uptr->frame.picstruct = RGY_PICSTRUCT_FRAME;
        uptr->frame.flags &= ~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF);
        auto ret = uptr->alloc();
        if (ret != cudaSuccess) {
            m_frameBuf.clear();
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_pFieldPairIn = std::move(uptr);
    }
    if (!m_pFieldPairOut) {
        unique_ptr<CUFrameBuf> uptr(new CUFrameBuf(*pOutputFrame));
        uptr->frame.ptr = nullptr;
        uptr->frame.pitch = 0;
        uptr->frame.height >>= 1;
        uptr->frame.picstruct = RGY_PICSTRUCT_FRAME;
        uptr->frame.flags &= ~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF);
        auto ret = uptr->alloc();
        if (ret != cudaSuccess) {
            m_frameBuf.clear();
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_pFieldPairOut = std::move(uptr);
    }
    const auto inputFrameInfoEx = getFrameInfoExtra(pInputFrame);
    const auto outputFrameInfoEx = getFrameInfoExtra(pOutputFrame);

    for (int i = 0; i < 2; i++) {
        auto cudaerr = cudaMemcpy2DAsync(m_pFieldPairIn->frame.ptr, m_pFieldPairIn->frame.pitch,
            pInputFrame->ptr + pInputFrame->pitch * i, pInputFrame->pitch * 2,
            inputFrameInfoEx.width_byte, inputFrameInfoEx.height_total >> 1,
            cudaMemcpyDeviceToDevice, stream);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed to seprate field(0): %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_CUDA;
        }
        int nFieldOut = 0;
        auto pFieldOut = &m_pFieldPairOut->frame;
        auto err = run_filter(&m_pFieldPairIn->frame, &pFieldOut, &nFieldOut);
        if (err != NV_ENC_SUCCESS) {
            return err;
        }
        cudaerr = cudaMemcpy2DAsync(pOutputFrame->ptr + pOutputFrame->pitch * i, pOutputFrame->pitch * 2,
            pFieldOut->ptr, pFieldOut->pitch,
            outputFrameInfoEx.width_byte, outputFrameInfoEx.height_total >> 1,
            cudaMemcpyDeviceToDevice, stream);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("failed to merge field(1): %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_CUDA;
        }
    }
    return RGY_ERR_NONE;
#else
    UNREFERENCED_PARAMETER(pInputFrame);
    UNREFERENCED_PARAMETER(pOutputFrame);
    return RGY_ERR_UNSUPPORTED;
#endif
}

#pragma warning(push)
#pragma warning(disable:4100)
RGY_ERR RGYFilterDisabled::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    AddMessage(RGY_LOG_ERROR, _T("This build doesn't support this filter.\n"));
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR RGYFilterDisabled::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    AddMessage(RGY_LOG_ERROR, _T("This build doesn't support this filter.\n"));
    return RGY_ERR_UNSUPPORTED;
}
#pragma warning(pop)

void RGYFilterDisabled::close() {
    m_pLog.reset();
}
