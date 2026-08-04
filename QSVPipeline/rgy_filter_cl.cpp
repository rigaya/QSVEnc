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

//インタレフレームをtop/bottomの2フィールドに分離し、各フィールドを単体のprogressiveフレームとしてrun_filter()に通してから、
//結果を元のインタレ配置に戻す。フィールドをまたいで画素を混ぜてはいけないフィルタ(resize等)をインタレ入力に適用するために使う。
//run_filter()を1フレームにつき2回呼ぶため、run_filter()側は「1入力→1出力」であることが前提(nFieldOutで確認)。
RGY_ERR RGYFilter::filter_as_interlaced_pair(const RGYFrameInfo *pInputFrame, RGYFrameInfo *pOutputFrame, RGYOpenCLQueue &queue) {
    //フィールドペア用バッファは、入力仕様が変化した場合にも再確保する。
    //(入力途中の解像度変更でこの関数への入力解像度が変わりうるため、初回のみ確保する実装では足りない)
    auto allocFieldPairBuf = [this](std::unique_ptr<RGYCLFrame>& fieldPairBuf, const RGYFrameInfo *frameInfo, const TCHAR *bufName) {
        //高さ半分・progressive扱いのフレームとして確保する。RFF関連のフラグはフィールド単体では意味を持たないので落とす
        RGYFrameInfo fieldFrame = *frameInfo;
        fieldFrame.height >>= 1;
        fieldFrame.picstruct = RGY_PICSTRUCT_FRAME;
        fieldFrame.flags &= ~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_TFF | RGY_FRAME_FLAG_RFF_BFF);
        if (fieldPairBuf && !cmpFrameInfoCspResolution(&fieldPairBuf->frame, &fieldFrame)) {
            return RGY_ERR_NONE;
        }
        auto uptr = m_cl->createFrameBuffer(fieldFrame);
        if (!uptr) {
            return RGY_ERR_MEMORY_ALLOC;
        }
        AddMessage(RGY_LOG_DEBUG, _T("allocated OpenCL field pair buffer(%s): %dx%d %s.\n"),
            bufName, uptr->frame.width, uptr->frame.height, RGY_CSP_NAMES[uptr->frame.csp]);
        //確保に成功してから差し替え、失敗時には既存バッファを維持する。
        fieldPairBuf = std::move(uptr);
        return RGY_ERR_NONE;
    };
    if (auto err = allocFieldPairBuf(m_pFieldPairIn, pInputFrame, _T("in")); err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate OpenCL field pair buffer(in): %s.\n"), get_err_mes(err));
        return err;
    }
    if (auto err = allocFieldPairBuf(m_pFieldPairOut, pOutputFrame, _T("out")); err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate OpenCL field pair buffer(out): %s.\n"), get_err_mes(err));
        return err;
    }

    for (int i = 0; i < 2; i++) {
        const auto fieldMode = (i == 0) ? RGYFrameCopyMode::FIELD_TOP : RGYFrameCopyMode::FIELD_BOTTOM;
        //src側はインタレフレームの片フィールド(1行おき)、dst側は詰まったフレーム全体。src/dstで異なるモードを指定する必要がある
        auto err = m_cl->copyFrameField(&m_pFieldPairIn->frame, pInputFrame,
            fieldMode, RGYFrameCopyMode::FRAME, nullptr, queue);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to separate field(%d): %s.\n"), i, get_err_mes(err));
            return err;
        }
        int nFieldOut = 0;
        auto pFieldOut = &m_pFieldPairOut->frame;
        err = run_filter(&m_pFieldPairIn->frame, &pFieldOut, &nFieldOut, queue, {}, nullptr);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        if (nFieldOut != 1 || pFieldOut == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("unexpected field output count: %d.\n"), nFieldOut);
            return RGY_ERR_UNKNOWN;
        }
        //分離時とは逆に、詰まったフレームを出力フレームの片フィールドへ書き戻す
        err = m_cl->copyFrameField(pOutputFrame, pFieldOut,
            RGYFrameCopyMode::FRAME, fieldMode, nullptr, queue);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to merge field(%d): %s.\n"), i, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
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
