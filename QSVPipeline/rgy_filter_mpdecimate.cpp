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

#include <map>
#include <array>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <type_traits>
#define _USE_MATH_DEFINES
#include <cmath>
#include "convert_csp.h"
#include "rgy_filter_mpdecimate.h"

#define MPDECIMATE_BLOCK_X (32)
#define MPDECIMATE_BLOCK_Y (8)

RGY_ERR RGYFilterMpdecimate::procPlane(const RGYFrameInfo *p0, const RGYFrameInfo *p1, RGYFrameInfo *tmp, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const int width = p0->width;
    const int height = p0->height;
    const char *kernel_name = "kernel_mpdecimate_block_diff";
    RGYWorkSize local(MPDECIMATE_BLOCK_X, MPDECIMATE_BLOCK_Y);
    RGYWorkSize global(divCeil(width, 8), height);
    auto err = m_mpdecimate.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)p0->ptr[0], p0->pitch[0],
        (cl_mem)p1->ptr[0], p1->pitch[0],
        width, height,
        (cl_mem)tmp->ptr[0], tmp->pitch[0]);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[p0->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterMpdecimate::procFrame(const RGYFrameInfo *p0, const RGYFrameInfo *p1, RGYFrameInfo *tmp, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    for (int i = 0; i < RGY_CSP_PLANES[p0->csp]; i++) {
        const auto plane0 = getPlane(p0, (RGY_PLANE)i);
        const auto plane1 = getPlane(p1, (RGY_PLANE)i);
        auto planeTmp = getPlane(tmp, (RGY_PLANE)i);
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[p0->csp] - 1) ? event : nullptr;
        auto err = procPlane(&plane0, &plane1, &planeTmp, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to procPlane(diff) frame(%d) %s: %s\n"), i, cl_errmes(err));
            return err_cl_to_rgy(err);
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterMpdecimate::calcDiff(RGYFilterMpdecimateFrameData *target, const RGYFilterMpdecimateFrameData *ref, RGYOpenCLQueue& queue_main) {
    if (m_streamDiff.get()) { // 別途キューを用意して並列実行する場合
        auto err = procFrame(&target->get()->frame, &ref->get()->frame, &target->tmp()->frame, m_streamDiff, { m_eventDiff }, &m_eventTransfer);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run calcDiff: %s.\n"), get_err_mes(err));
            return err;
        }
        if ((err = target->tmp()->queueMapBuffer(m_streamTransfer, CL_MAP_READ, { m_eventTransfer })) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to queueMapBuffer in calcDiff: %s.\n"), get_err_mes(err));
            return err;
        }
    } else {
        //QSV:Broadwell以前の環境では、なぜか上記のように別のキューで実行しようとすると、永遠にqueueMapBufferが開始されず、フリーズしてしまう
        //こういうケースでは標準のキューを使って逐次実行する
        auto err = procFrame(&target->get()->frame, &ref->get()->frame, &target->tmp()->frame, queue_main, { }, nullptr);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run calcDiff: %s.\n"), get_err_mes(err));
            return err;
        }
        if ((err = target->tmp()->queueMapBuffer(queue_main, CL_MAP_READ)) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to queueMapBuffer in calcDiff: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterMpdecimateFrameData::RGYFilterMpdecimateFrameData(shared_ptr<RGYOpenCLContext> context, std::shared_ptr<RGYLog> log) :
    m_cl(context),
    m_log(log),
    m_inFrameId(-1),
    m_buf(),
    m_tmp() {

}

RGYFilterMpdecimateFrameData::~RGYFilterMpdecimateFrameData() {
    m_buf.reset();
    m_tmp.reset();
}

RGY_ERR RGYFilterMpdecimateFrameData::set(const RGYFrameInfo *pInputFrame, int inputFrameId, RGYOpenCLQueue& queue, RGYOpenCLEvent& event) {
    m_inFrameId = inputFrameId;
    if (!m_buf) {
        m_buf = m_cl->createFrameBuffer(pInputFrame->width, pInputFrame->height, pInputFrame->csp, pInputFrame->bitdepth);
    }
    if (!m_tmp) {
        m_tmp = m_cl->createFrameBuffer(divCeil(pInputFrame->width, 8), divCeil(pInputFrame->height, 8), RGY_CSP_YUV444_32, RGY_CSP_BIT_DEPTH[RGY_CSP_YUV444_32]);
    }
    copyFrameProp(&m_buf->frame, pInputFrame);

    auto err = m_cl->copyFrame(&m_buf->frame, pInputFrame, nullptr, queue, &event);
    if (err != RGY_ERR_NONE) {
        m_log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to set frame to data cache: %s.\n"), get_err_mes(err));
        return RGY_ERR_CUDA;
    }
    return err;
}

bool RGYFilterMpdecimateFrameData::checkIfFrameCanbeDropped(const int hi, const int lo, const float factor) {
    m_tmp->mapEvent().wait();
    auto tmpMappedHost = m_tmp->mappedHost();
    const int threshold = (int)((float)tmpMappedHost.width * tmpMappedHost.height * factor + 0.5f);
    int loCount = 0;
    for (int iplane = 0; iplane < RGY_CSP_PLANES[tmpMappedHost.csp]; iplane++) {
        const auto plane = getPlane(&tmpMappedHost, (RGY_PLANE)iplane);
        const int blockw = divCeil(plane.width, 8);
        const int blockh = divCeil(plane.height, 8);
        for (int j = 0; j < blockh; j++) {
            const int *ptrResult = (const int *)(tmpMappedHost.ptr[0] + j * tmpMappedHost.pitch[0]);
            for (int i = 0; i < blockw; i++) {
                const int result = ptrResult[i];
                if (result > hi) {
                    return false;
                }
                if (result > lo) {
                    loCount++;
                    if (loCount > threshold) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

RGYFilterMpdecimateCache::RGYFilterMpdecimateCache(shared_ptr<RGYOpenCLContext> context) : m_cl(context), m_inputFrames(0), m_frames() {

}

RGYFilterMpdecimateCache::~RGYFilterMpdecimateCache() {
    m_frames.clear();
}

void RGYFilterMpdecimateCache::init(int bufCount, std::shared_ptr<RGYLog> log) {
    m_log = log;
    m_frames.clear();
    for (int i = 0; i < bufCount; i++) {
        m_frames.push_back(std::make_unique<RGYFilterMpdecimateFrameData>(m_cl, log));
    }
}

RGY_ERR RGYFilterMpdecimateCache::add(const RGYFrameInfo *pInputFrame, RGYOpenCLQueue& queue, RGYOpenCLEvent& event) {
    const int id = m_inputFrames++;
    return getEmpty()->set(pInputFrame, id, queue, event);
}

RGYFilterMpdecimate::RGYFilterMpdecimate(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_dropCount(0), m_ref(-1), m_target(-1), m_mpdecimate(), m_cache(context), m_eventDiff(), m_streamDiff(), m_streamTransfer() {
    m_name = _T("mpdecimate");
}

RGYFilterMpdecimate::~RGYFilterMpdecimate() {
    close();
}

RGY_ERR RGYFilterMpdecimate::checkParam(const std::shared_ptr<RGYFilterParamMpdecimate> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->mpdecimate.lo <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("\"lo\" must a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->mpdecimate.hi <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("\"hi\" must a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->mpdecimate.frac < 0.0) {
        AddMessage(RGY_LOG_ERROR, _T("\"frac\" must a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterMpdecimate::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamMpdecimate>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if ((sts = checkParam(prm)) != RGY_ERR_NONE) {
        return sts;
    }

    if (!m_param
        || std::dynamic_pointer_cast<RGYFilterParamMpdecimate>(m_param)->mpdecimate != prm->mpdecimate) {

        const auto options = strsprintf("-D Type=%s -D Type4=%s -D MPDECIMATE_BLOCK_X=%d -D MPDECIMATE_BLOCK_Y=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort"  : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort4" : "uchar4",
            MPDECIMATE_BLOCK_X, MPDECIMATE_BLOCK_Y);
        m_mpdecimate.set(std::move(m_cl->buildResourceAsync(_T("RGY_FILTER_MPDECIMATE_CL"), _T("EXE_DATA"), options.c_str())));

        m_cache.init(2, m_pLog);

        if (prm->useSeparateQueue) {
            m_streamDiff = m_cl->createQueue(m_cl->queue().devid(), m_cl->queue().getProperties());
            if (!m_streamDiff.get()) {
                AddMessage(RGY_LOG_ERROR, _T("failed to createQueue.\n"));
                return RGY_ERR_UNKNOWN;
            }
            AddMessage(RGY_LOG_DEBUG, _T("Create OpenCL queue: Success.\n"));

            m_streamTransfer = m_cl->createQueue(m_cl->queue().devid(), m_cl->queue().getProperties());
            if (!m_streamTransfer.get()) {
                AddMessage(RGY_LOG_ERROR, _T("failed to createQueue.\n"));
                return RGY_ERR_UNKNOWN;
            }
            AddMessage(RGY_LOG_DEBUG, _T("Create OpenCL queue: Success.\n"));
        } else {
            AddMessage(RGY_LOG_DEBUG, _T("Use main queue for data transfer, this might lead to poor performance.\n"));
        }
        for (int i = 0; i < _countof(prm->frameIn.pitch); i++) {
            prm->frameOut.pitch[i] = prm->frameIn.pitch[i];
        }

        m_fpLog.reset();
        if (prm->mpdecimate.log) {
            const tstring logfilename = prm->outfilename + _T(".mpdecimate.log.txt");
            m_fpLog = std::unique_ptr<FILE, fp_deleter>(_tfopen(logfilename.c_str(), _T("w")), fp_deleter());
            AddMessage(RGY_LOG_DEBUG, _T("Opened log file: %s.\n"), logfilename.c_str());
        }

        const int max_value = (1 << RGY_CSP_BIT_DEPTH[prm->frameIn.csp]) - 1;
        m_pathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP));
        m_dropCount = 0;
        m_ref = -1;
        m_target = -1;

        setFilterInfo(pParam->print());
    }
    m_param = pParam;
    return sts;
}

tstring RGYFilterParamMpdecimate::print() const {
    auto str = mpdecimate.print();
    if (!useSeparateQueue) {
        str += _T(", no queue opt");
    }
    return str;
}

bool RGYFilterMpdecimate::dropFrame(RGYFilterMpdecimateFrameData *targetFrame) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamMpdecimate>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return false;
    }
    if (prm->mpdecimate.max > 0 &&
        m_dropCount >= prm->mpdecimate.max) {
        return false;
    }
    if (prm->mpdecimate.max < 0 &&
        (m_dropCount - 1) > prm->mpdecimate.max) {
        return false;
    }
    const int bit_depth = RGY_CSP_BIT_DEPTH[targetFrame->get()->frame.csp];
    auto err = targetFrame->checkIfFrameCanbeDropped(prm->mpdecimate.hi << (bit_depth - 8), prm->mpdecimate.lo << (bit_depth - 8), prm->mpdecimate.frac);
    targetFrame->tmp()->unmapBuffer();
    return err;
}

RGY_ERR RGYFilterMpdecimate::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue& queue_main, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamMpdecimate>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pInputFrame->ptr[0] == nullptr && m_ref < 0) {
        //終了
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return sts;
    }
    if (!m_mpdecimate.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_MPDECIMATE_CL(m_mpdecimate)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    if (m_ref < 0) {
        m_ref = m_cache.inframe();
        auto err = m_cache.add(pInputFrame, queue_main, m_eventDiff);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to add frame to cache: %s.\n"), get_err_mes(err));
            return err;
        }
        *pOutputFrameNum = 1;
        ppOutputFrames[0] = &m_cache.get(m_ref)->frame;
        if (m_fpLog) {
            fprintf(m_fpLog.get(), "  %8d: %10lld\n", m_ref, (long long)ppOutputFrames[0]->timestamp);
        }
        return sts;
    }
    if (m_target >= 0) {
        auto targetFrame = m_cache.frame(m_target);
        const bool drop = dropFrame(targetFrame) && pInputFrame->ptr[0] != nullptr; //最終フレームは必ず出力する
        if (m_fpLog) {
            fprintf(m_fpLog.get(), "%s %8d: %10lld\n", (drop) ? "d" : " ", m_target, (long long)targetFrame->get()->frame.timestamp);
        }
        if (drop) {
            targetFrame->reset();
            m_target = -1;
            m_dropCount = std::max(1, m_dropCount + 1);
            *pOutputFrameNum = 0;
            ppOutputFrames[0] = nullptr;
        } else {
            m_dropCount = std::min(-1, m_dropCount - 1);
            m_cache.frame(m_ref)->reset();
            m_ref = m_target;
            m_target = -1;
            *pOutputFrameNum = 1;
            ppOutputFrames[0] = &targetFrame->get()->frame;
        }
    }
    if (pInputFrame->ptr[0] != nullptr) {
        m_target = m_cache.inframe();
        auto err = m_cache.add(pInputFrame, queue_main, m_eventDiff);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to add frame to cache: %s.\n"), get_err_mes(err));
            return err;
        }
        err = calcDiff(m_cache.frame(m_target), m_cache.frame(m_ref), queue_main);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run calcDiff: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

void RGYFilterMpdecimate::close() {
    m_mpdecimate.clear();
    m_eventDiff.reset();
    m_eventTransfer.reset();
    m_fpLog.reset();
}
