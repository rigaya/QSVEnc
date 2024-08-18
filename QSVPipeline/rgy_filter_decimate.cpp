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
#include "rgy_avutil.h"
#include "rgy_filter_decimate.h"

#define DECIMATE_BLOCK_MAX (32)
#define DECIMATE_K2_THREAD_BLOCK_X (32)
#define DECIMATE_K2_THREAD_BLOCK_Y (8)

//blockxがこの値以下なら、kernel2を使用する
static const int DECIMATE_KERNEL2_BLOCK_X_THRESHOLD = 4;

struct int2 {
    int x, y;
};

RGY_ERR RGYFilterDecimate::procPlane(const bool useKernel2, const bool firstPlane, const RGYFrameInfo *p0, const RGYFrameInfo *p1, std::unique_ptr<RGYCLBuf>& tmp, const int blockHalfX, const int blockHalfY,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const int width = p0->width;
    const int height = p0->height;
    const char *kernel_name = nullptr;
    RGYWorkSize local, global;
    if (useKernel2) {
        local = RGYWorkSize(DECIMATE_K2_THREAD_BLOCK_X, DECIMATE_K2_THREAD_BLOCK_Y);
        global = RGYWorkSize(divCeil(width, blockHalfX), divCeil(width, blockHalfY));
        switch (blockHalfX) {
        case 1:  kernel_name = "kernel_block_diff2_1"; break;
        case 2:  kernel_name = "kernel_block_diff2_2"; break;
        case 4:  kernel_name = "kernel_block_diff2_4"; break;
        case 8:  kernel_name = "kernel_block_diff2_8"; break;
        case 16: kernel_name = "kernel_block_diff2_16"; break;
        default:
            AddMessage(RGY_LOG_ERROR, _T("Unsupported blockHalfX=%d for useKernel2 true\n"), blockHalfX);
            return RGY_ERR_UNSUPPORTED;
        }
    } else {
        local = RGYWorkSize(blockHalfX / 4, blockHalfY);
        global = RGYWorkSize(divCeil(width, 4), height);
        kernel_name = "kernel_block_diff";
        if (blockHalfX < 4 || 64 < blockHalfX) {
            AddMessage(RGY_LOG_ERROR, _T("useKernel2 false do not support blockHalfX=%d\n"), blockHalfX);
            return RGY_ERR_UNSUPPORTED;
        }
    }

    const size_t grid_count = global.groups(local).total();
    const size_t bufsize = (useKernel2) ? grid_count * sizeof(int2) : grid_count * sizeof(int);
    if (!tmp
        || tmp->size() < bufsize) {
        tmp = m_cl->createBuffer(bufsize, CL_MEM_READ_WRITE);
        if (!tmp) {
            return RGY_ERR_NULL_PTR;
        }
        int zero = 0;
        auto err = m_cl->setBuf(&zero, sizeof(zero), bufsize, tmp.get(), queue, wait_events);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    auto err = m_decimate.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)p0->ptr[0], p0->pitch[0],
        (cl_mem)p1->ptr[0], p1->pitch[0],
        width, height,
        blockHalfX, firstPlane ? 1 : 0,
        tmp->mem());
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[p0->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDecimate::procFrame(const RGYFrameInfo *p0, const RGYFrameInfo *p1, std::unique_ptr<RGYCLBuf>& tmp,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDecimate>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const bool useKernel2 = (prm->decimate.blockX / 2 <= DECIMATE_KERNEL2_BLOCK_X_THRESHOLD);

    const int targetPlanes = (prm->decimate.chroma || RGY_CSP_CHROMA_FORMAT[p0->csp] == RGY_CHROMAFMT_RGB) ? (int)(RGY_CSP_PLANES[p0->csp]) : 1;
    for (int i = 0; i < targetPlanes; i++) {
        const auto plane0 = getPlane(p0, (RGY_PLANE)i);
        const auto plane1 = getPlane(p1, (RGY_PLANE)i);
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[p0->csp] - 1) ? event : nullptr;
        int blockHalfX = prm->decimate.blockX / 2;
        int blockHalfY = prm->decimate.blockY / 2;
        if (i > 0 && RGY_CSP_CHROMA_FORMAT[p0->csp] == RGY_CHROMAFMT_YUV420) {
            blockHalfX /= 2;
            blockHalfY /= 2;
        }
        auto err = procPlane(useKernel2, i == 0, &plane0, &plane1, tmp, blockHalfX, blockHalfY, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterDecimateFrameData::RGYFilterDecimateFrameData(std::shared_ptr<RGYOpenCLContext> context, std::shared_ptr<RGYLog> log) :
    m_cl(context),
    m_log(log),
    m_inFrameId(-1),
    m_buf(),
    m_tmp(),
    m_diffMaxBlock(std::numeric_limits<int64_t>::max()),
    m_diffTotal(std::numeric_limits<int64_t>::max()) {

}

RGYFilterDecimateFrameData::~RGYFilterDecimateFrameData() {
    m_buf.reset();
    m_tmp.reset();
    m_log.reset();
}

RGY_ERR RGYFilterDecimateFrameData::set(const RGYFrameInfo *pInputFrame, int inputFrameId, int blockSizeX, int blockSizeY, RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent& event) {
    m_inFrameId = inputFrameId;
    m_blockX = blockSizeX;
    m_blockY = blockSizeY;
    m_diffMaxBlock = std::numeric_limits<int64_t>::max();
    m_diffTotal = std::numeric_limits<int64_t>::max();
    if (!m_buf) {
        m_buf = m_cl->createFrameBuffer(pInputFrame->width, pInputFrame->height, pInputFrame->csp, pInputFrame->bitdepth);
    }
    auto err = m_cl->copyFrame(&m_buf->frame, pInputFrame, nullptr, queue, wait_events, &event);
    if (err != RGY_ERR_NONE) {
        m_log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to set frame to data cache: %s.\n"), get_err_mes(err));
        return RGY_ERR_CUDA;
    }
    copyFrameProp(&m_buf->frame, pInputFrame);
    return err;
}

RGY_ERR RGYFilterDecimate::calcDiff(RGYFilterDecimateFrameData *current, const RGYFilterDecimateFrameData *prev, RGYOpenCLQueue& queue_main) {
    if (m_streamDiff.get()) { // 別途キューを用意して並列実行する場合
        auto err = procFrame(&current->get()->frame, &prev->get()->frame, current->tmp(), m_streamDiff, { m_eventDiff }, &m_eventTransfer);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        if ((err = current->tmp()->queueMapBuffer(m_streamTransfer, CL_MAP_READ, { m_eventTransfer })) != RGY_ERR_NONE) {
            m_pLog->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to queueMapBuffer in calcDiff: %s.\n"), get_err_mes(err));
            return err;
        }
    } else {
        //QSV:Broadwell以前の環境では、なぜか上記のように別のキューで実行しようとすると、永遠にqueueMapBufferが開始されず、フリーズしてしまう
        //こういうケースでは標準のキューを使って逐次実行する
        auto err = procFrame(&current->get()->frame, &prev->get()->frame, current->tmp(), queue_main, { }, nullptr);
        if (err != RGY_ERR_NONE) {
            return err;
        }
#if ENCODER_VCEENC
        //非同期モードで転送するとなぜかエラー終了したり、予期せぬ結果を招くため、同期転送する
        RGYCLMapBlock map_mode = RGY_CL_MAP_BLOCK_LAST;
#else
        RGYCLMapBlock map_mode = RGY_CL_MAP_BLOCK_NONE;
#endif
        if ((err = current->tmp()->queueMapBuffer(queue_main, CL_MAP_READ, {}, map_mode)) != RGY_ERR_NONE) {
            m_pLog->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to queueMapBuffer in calcDiff: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

void RGYFilterDecimateFrameData::calcDiffFromTmp() {
    if (m_inFrameId == 0) { //最初のフレームは差分をとる対象がない
        m_diffMaxBlock = std::numeric_limits<int64_t>::max();
        m_diffTotal = std::numeric_limits<int64_t>::max();
        if (m_tmp) m_tmp->unmapBuffer();
        return;
    }
    m_tmp->mapEvent().wait();
    const int blockHalfX = m_blockX / 2;
    const int blockHalfY = m_blockY / 2;
    const bool useKernel2 = (m_blockX / 2 <= DECIMATE_KERNEL2_BLOCK_X_THRESHOLD);
    if (useKernel2) {
        int2 *const tmpHost = (int2 *)m_tmp->mappedPtr();
        const size_t count = m_tmp->size() / sizeof(int2);
        m_diffMaxBlock = -1;
        m_diffTotal = 0;
        for (size_t i = 0; i < count; i++) {
            m_diffTotal += tmpHost[i].x;
            m_diffMaxBlock = std::max<int64_t>(m_diffMaxBlock, tmpHost[i].y);
        }
    } else {
        const int blockXHalfCount = divCeil(m_buf->frame.width, blockHalfX);
        const int blockYHalfCount = divCeil(m_buf->frame.height, blockHalfY);
        const int blockXYHalfCount = blockXHalfCount * blockYHalfCount;

        int *const tmpHost = (int *)m_tmp->mappedPtr();

        m_diffMaxBlock = -1;
        for (int i = 0; i < blockYHalfCount - 1; i++) {
            for (int j = 0; j < blockXHalfCount - 1; j++) {
                int64_t tmp = tmpHost[(i + 0) * blockXHalfCount + j + 0]
                            + tmpHost[(i + 0) * blockXHalfCount + j + 1]
                            + tmpHost[(i + 1) * blockXHalfCount + j + 0]
                            + tmpHost[(i + 1) * blockXHalfCount + j + 1];
                m_diffMaxBlock = std::max(m_diffMaxBlock, tmp);
            }
        }
        m_diffTotal = std::accumulate(tmpHost, tmpHost + blockXYHalfCount, (int64_t)0);
    }
    m_tmp->unmapBuffer();
}

RGYFilterDecimateCache::RGYFilterDecimateCache(shared_ptr<RGYOpenCLContext> context) : m_cl(context), m_inputFrames(0), m_frames() {

}

RGYFilterDecimateCache::~RGYFilterDecimateCache() {
    m_frames.clear();
    m_log.reset();
}

void RGYFilterDecimateCache::init(int bufCount, int blockX, int blockY, std::shared_ptr<RGYLog> log) {
    m_log = log;
    m_blockX = blockX;
    m_blockY = blockY;
    m_frames.clear();
    for (int i = 0; i < bufCount; i++) {
        m_frames.push_back(std::make_unique<RGYFilterDecimateFrameData>(m_cl, log));
    }
}

RGY_ERR RGYFilterDecimateCache::add(const RGYFrameInfo *pInputFrame, RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent& event) {
    const int id = m_inputFrames++;
    return frame(id)->set(pInputFrame, id, m_blockX, m_blockY, queue, wait_events, event);
}

RGYFilterDecimate::RGYFilterDecimate(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_flushed(false), m_frameLastDropped(-1), m_frameLastInputDuration(0), m_decimate(), m_cache(context), m_eventDiff(), m_streamDiff(), m_streamTransfer() {
    m_name = _T("decimate");
}

RGYFilterDecimate::~RGYFilterDecimate() {
    close();
}

RGY_ERR RGYFilterDecimate::checkParam(const std::shared_ptr<RGYFilterParamDecimate> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->decimate.cycle <= 1) {
        AddMessage(RGY_LOG_ERROR, _T("cycle must be 2 or bigger: cycle = %d.\n"), prm->decimate.cycle);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->decimate.cycle >= (int)DecimateSelectResult::ORDER) {
        AddMessage(RGY_LOG_ERROR, _T("cycle must be less than %d.\n"), (int)DecimateSelectResult::ORDER);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->decimate.drop <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("drop must be 1 or bigger: drop = %d.\n"), prm->decimate.drop);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->decimate.drop >= prm->decimate.cycle) {
        AddMessage(RGY_LOG_ERROR, _T("cycle must be larger than drop: cycle = %d, drop = %d.\n"), prm->decimate.cycle, prm->decimate.drop);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->decimate.blockX < 4 || 64 < prm->decimate.blockX || (prm->decimate.blockX & (prm->decimate.blockX-1)) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid blockX: %d.\n"), prm->decimate.blockX);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->decimate.blockY < 4 || 64 < prm->decimate.blockY || (prm->decimate.blockY & (prm->decimate.blockY - 1)) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid blockY: %d.\n"), prm->decimate.blockY);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

#pragma warning(push)
#pragma warning(disable: 4127) //C4127: 条件式が定数です。
RGY_ERR RGYFilterDecimate::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDecimate>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if ((sts = checkParam(prm)) != RGY_ERR_NONE) {
        return sts;
    }

    if (!m_param
        || std::dynamic_pointer_cast<RGYFilterParamDecimate>(m_param)->decimate != prm->decimate) {
        auto options = strsprintf("-D Type=%s -D Type2=%s -D Type4=%s"
            " -D DTB_X=%d -D DTB_Y=%d -D DECIMATE_BLOCK_MAX=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort2" : "uchar2",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort4" : "uchar4",
            DECIMATE_K2_THREAD_BLOCK_X,
            DECIMATE_K2_THREAD_BLOCK_Y,
            DECIMATE_BLOCK_MAX);
        m_decimate.set(std::async(std::launch::async, [cl = m_cl, options]() {
            auto build_options = options;
            const auto sub_group_ext_avail = cl->platform()->checkSubGroupSupport(cl->queue().devid());
            if (ENCODER_QSV && sub_group_ext_avail != RGYOpenCLSubGroupSupport::NONE) { // VCEではこれを使用するとかえって遅くなる
                if (   sub_group_ext_avail == RGYOpenCLSubGroupSupport::STD22
                    || sub_group_ext_avail == RGYOpenCLSubGroupSupport::STD20KHR) {
                    build_options += " -cl-std=CL2.0 ";
                }
                //subgroup情報を得るため一度コンパイル
                auto decimate = cl->buildResource(_T("RGY_FILTER_DECIMATE_CL"), _T("EXE_DATA"), build_options.c_str());
                if (!decimate) {
                    return std::unique_ptr<RGYOpenCLProgram>();
                }

                auto getKernelSubGroupInfo = clGetKernelSubGroupInfo != nullptr ? clGetKernelSubGroupInfo : clGetKernelSubGroupInfoKHR;
                RGYWorkSize local(DECIMATE_K2_THREAD_BLOCK_X, DECIMATE_K2_THREAD_BLOCK_Y);
                size_t subgroup_size = 0;
                auto err = getKernelSubGroupInfo(decimate->kernel("kernel_block_diff2_4").get()->get(), cl->platform()->dev(0).id(), CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR,
                    sizeof(local.w[0]) * 2, &local.w[0], sizeof(subgroup_size), &subgroup_size, nullptr);
                if (err == 0) {
                    build_options += strsprintf(" -D SUB_GROUP_SIZE=%u", subgroup_size);
                }
            }
            return cl->buildResource(_T("RGY_FILTER_DECIMATE_CL"), _T("EXE_DATA"), build_options.c_str());
        }));

        m_cache.init(prm->decimate.cycle + 1, prm->decimate.blockX, prm->decimate.blockY, m_pLog);

        pParam->baseFps *= rgy_rational<int>(prm->decimate.cycle - prm->decimate.drop, prm->decimate.cycle);

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
        if (prm->decimate.log) {
            const tstring logfilename = prm->outfilename + _T(".decimate.log.txt");
            m_fpLog = std::unique_ptr<FILE, fp_deleter>(_tfopen(logfilename.c_str(), _T("w")), fp_deleter());
            AddMessage(RGY_LOG_DEBUG, _T("Opened log file: %s.\n"), logfilename.c_str());
        }

        const int max_value = (1 << RGY_CSP_BIT_DEPTH[prm->frameIn.csp]) - 1;
        m_pathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP));
        m_threSceneChange = (int64_t)(((double)max_value * prm->frameIn.width * prm->frameIn.height * (double)prm->decimate.threSceneChange) / 100);
        m_threDuplicate = (int64_t)(((double)max_value * prm->decimate.blockX * prm->decimate.blockY * (double)prm->decimate.threDuplicate) / 100);
        m_frameLastDropped = -1;
        m_flushed = false;

        setFilterInfo(pParam->print());
    }
    m_param = pParam;
    return sts;
}
#pragma warning(pop)

tstring RGYFilterParamDecimate::print() const {
    auto str = decimate.print();
    if (!useSeparateQueue) {
        str += _T(", no queue opt");
    }
    return str;
}

std::vector<DecimateSelectResult> RGYFilterDecimate::selectDropFrame(const int iframeStart) {
    std::vector<DecimateSelectResult> selectResults(m_cache.inframe() - iframeStart, DecimateSelectResult::NONE);

    auto prm = std::dynamic_pointer_cast<RGYFilterParamDecimate>(m_param);
    int cycle = prm->decimate.cycle;
    for (int idrop = 0; idrop < prm->decimate.drop; idrop++, cycle--) {
        int frameLowest = iframeStart;
        int frameDuplicate = -1;
        int frameSceneChange = -1;
        for (int iframe = iframeStart; iframe < m_cache.inframe(); iframe++) {
            if (selectResults[iframe - iframeStart] & DecimateSelectResult::DROP) {
                if (iframe == frameLowest) {
                    frameLowest++;
                }
            } else {
                if (m_cache.frame(iframe)->diffTotal() > m_threSceneChange) {
                    frameSceneChange = iframe;
                }
                if (m_cache.frame(iframe)->diffMaxBlock() < m_cache.frame(frameLowest)->diffMaxBlock()) {
                    frameLowest = iframe;
                }
            }
        }
        if (m_cache.frame(frameLowest)->diffMaxBlock() < m_threDuplicate) {
            frameDuplicate = frameLowest;
        }
        //判定結果を設定
        if (frameDuplicate >= 0) selectResults[frameDuplicate - iframeStart] |= DecimateSelectResult::DUPLICATE;
        if (frameSceneChange >= 0) selectResults[frameSceneChange - iframeStart] |= DecimateSelectResult::SCENE_CHANGE;
        if (frameLowest >= 0) selectResults[frameLowest - iframeStart] |= (uint32_t)(idrop + 1); //そのままだと0がフラグとして認識できないので+1する
        //ドロップするフレームの選択
        const int frameDrop = (frameSceneChange >= 0 && frameDuplicate < 0) ? frameSceneChange : frameLowest;
        //cycle分のフレームがそろっている場合は、必ずいずれかのフレームをドロップする
        if (m_cache.inframe() - iframeStart == cycle) {
            ;
        } else if (m_frameLastDropped + cycle >= m_cache.inframe()) {
            //cycle分のフレームがそろっていない(flushする)場合は、
            //dropすべきものがなければ、dropしない(-1)とする
            break;
        }
        selectResults[frameDrop - iframeStart] |= DecimateSelectResult::DROP;
    }
    return selectResults;

}

RGY_ERR RGYFilterDecimate::setOutputFrame(int64_t nextTimestamp, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDecimate>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int iframeStart = (int)((m_cache.inframe() + prm->decimate.cycle - 1) / prm->decimate.cycle) * prm->decimate.cycle - prm->decimate.cycle;
    //GPU->CPUの転送終了を待機
    m_eventTransfer.wait();
    //CPUに転送された情報の後処理
    for (int iframe = iframeStart; iframe < m_cache.inframe(); iframe++) {
        m_cache.frame(iframe)->calcDiffFromTmp();
    }

    //判定
    const auto selectResults = selectDropFrame(iframeStart);
    if ((int)selectResults.size() != m_cache.inframe() - iframeStart) {
        AddMessage(RGY_LOG_ERROR, _T("NVEncFilterDecimate::setOutputFrame: unexpected error, %d != %d - %d.\n"), (int)selectResults.size(), m_cache.inframe(), iframeStart);
        return RGY_ERR_UNKNOWN;
    }
    const auto dropFrameCount = std::count_if(selectResults.begin(), selectResults.end(),
        [](const auto& flag) { return (flag & DecimateSelectResult::DROP) != DecimateSelectResult::NONE; });

    //入力フレームのtimestamp取得
    bool ptsInvalid = false;
    std::vector<int64_t> cycleInPts;
    cycleInPts.reserve(prm->decimate.cycle+1);
    for (int iframe = iframeStart; iframe < m_cache.inframe(); iframe++) {
        auto timestamp = m_cache.frame(iframe)->get()->frame.timestamp;
        if (timestamp == AV_NOPTS_VALUE) {
            ptsInvalid = true;
        }
        cycleInPts.push_back(timestamp);
    }
    if (nextTimestamp == AV_NOPTS_VALUE && !ptsInvalid) {
        if (cycleInPts.size() > 1) {
            nextTimestamp = cycleInPts.back() + (cycleInPts.back() - cycleInPts.front()) * cycleInPts.size() / (cycleInPts.size() - 1); // 単純外挿
        } else {
            nextTimestamp = cycleInPts.back() + m_frameLastInputDuration;
        }
    }
    cycleInPts.push_back(nextTimestamp);
    if (selectResults.size() == 0 && !ptsInvalid) {
        if (cycleInPts.size() > 1) {
            cycleInPts.push_back(cycleInPts.back() + (cycleInPts.back() - cycleInPts.front()) * cycleInPts.size() / (cycleInPts.size() - 1)); // 単純外挿
        } else {
            cycleInPts.push_back(cycleInPts.back() + m_frameLastInputDuration);
        }
    }
    if (cycleInPts.size() > 1) {
        m_frameLastInputDuration = (cycleInPts.back() - cycleInPts.front()) / (cycleInPts.size() - 1);
    }

    //出力フレームのtimestampの調整
    std::vector<int64_t> cycleOutPts(m_cache.inframe() - iframeStart - dropFrameCount + 1, AV_NOPTS_VALUE);
    if (!ptsInvalid) {
        // dropしたフレームの合計時間の計算
        int64_t dropFramesDuration = 0;
        for (size_t i = 0; i < selectResults.size(); i++) {
            if (selectResults[i] & DecimateSelectResult::DROP) {
                dropFramesDuration += cycleInPts[i + 1] - cycleInPts[i];
            }
        }
        //フレームの時間分配の最小単位は cycle - drop
        const int timeFrameDivBase = prm->decimate.cycle - prm->decimate.drop;
        //この最小単位を基準に、フレームを進めるごとに追加する時間はdrop
        const int timeFrameDivInc = prm->decimate.drop;
        int timeFrameOffset = 0;
        int outframe = 0;
        for (size_t i = 0; i < selectResults.size(); i++) {
            if (selectResults[i] & DecimateSelectResult::DROP) {
                timeFrameOffset -= timeFrameDivBase;
            } else {
                cycleOutPts[outframe++] = cycleInPts[i] + (timeFrameOffset * dropFramesDuration) / (prm->decimate.drop * timeFrameDivBase);
                timeFrameOffset += timeFrameDivInc;
            }
        }
        cycleOutPts[outframe++] = cycleInPts.back();
        if (outframe != (int)cycleOutPts.size()) {
            AddMessage(RGY_LOG_ERROR, _T("RGYFilterDecimate::setOutputFrame: unexpected error, outframe = %d, cycleOutPts.size() = %d.\n"), outframe, (int)cycleOutPts.size());
            return RGY_ERR_UNKNOWN;
        }
    }

    //cycleをあらわすのに必要な桁数
    const int cycle_digit_num = (int)std::log10(prm->decimate.cycle) + 1;
    std::string cycle_digit_space(cycle_digit_num + 1, ' ');

    //出力フレームの設定
    *pOutputFrameNum = 0;
    for (int i = 0, iout = 0, iframe = iframeStart; iframe < m_cache.inframe(); iframe++, i++) {
        auto iframeData = m_cache.frame(iframe);
        if (selectResults[i] & DecimateSelectResult::DROP) {
            m_frameLastDropped = iframe;
        } else {
            auto frame = &iframeData->get()->frame;
            frame->timestamp = cycleOutPts[iout];
            frame->duration = cycleOutPts[iout + 1] - cycleOutPts[iout];
            if (frame->duration < 0) {
                AddMessage(RGY_LOG_WARN, _T("Unexpected frame duration %d for frame = %d.\n"), frame->duration, iframe);
            }
            ppOutputFrames[iout++] = frame;
            *pOutputFrameNum = iout;
        }
        if (m_fpLog) {
            fprintf(m_fpLog.get(), "[%s%s%s%s] %8d: %10lld: diff total %10lld, max %10lld\n",
                (selectResults[i] & DecimateSelectResult::SCENE_CHANGE) ? "S" : " ",
                (selectResults[i] & DecimateSelectResult::DUPLICATE)    ? "P" : " ",
                (selectResults[i] & DecimateSelectResult::DROP)         ? "D" : " ",
                (selectResults[i] & DecimateSelectResult::ORDER)        ? strsprintf("L%0d", (int)(selectResults[i] & DecimateSelectResult::ORDER)).c_str() : cycle_digit_space.c_str(),
                iframe,
                (selectResults[i] & DecimateSelectResult::DROP)         ? -1ll : cycleOutPts[iout-1],
                (long long int)iframeData->diffTotal(),
                (long long int)iframeData->diffMaxBlock());
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDecimate::calcDiffWithPrevFrameAndSetDiffToCurr(const int curr, const int prev, RGYOpenCLQueue& queue_main) {
    //前のフレームとの差分をとる
    auto frameCurrent = m_cache.frame(curr);
    auto framePrev = m_cache.frame(prev);
    auto err = calcDiff(frameCurrent, framePrev, queue_main);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at calc_block_diff_frame: %s.\n"),
            get_err_mes(err));
        return err;
    }
    return err;
}

RGY_ERR RGYFilterDecimate::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue& queue_main, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDecimate>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (pInputFrame->ptr[0] == nullptr && m_flushed) {
        //終了
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return sts;
    }
    if (!m_decimate.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_DECIMATE_CL(m_decimate)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }

    const int inframeId = m_cache.inframe();
    *pOutputFrameNum = 0;
    if (m_cache.inframe() > 0 && (m_cache.inframe() % prm->decimate.cycle == 0 || pInputFrame->ptr[0] == nullptr)) { //cycle分のフレームがそろったら
        //dropFrameの計算が終わっている時点でフレームの準備は完了、待機するものはない
        event = nullptr;
        auto ret = setOutputFrame((pInputFrame) ? pInputFrame->timestamp : AV_NOPTS_VALUE, ppOutputFrames, pOutputFrameNum);
        if (ret != RGY_ERR_NONE) {
            return ret;
        }

        if (pInputFrame->ptr[0] == nullptr) {
            m_flushed = true;
            return sts;
        }
    }

    auto err = m_cache.add(pInputFrame, queue_main, wait_events, m_eventDiff);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to add frame to cache: %s.\n"), get_err_mes(err));
        return RGY_ERR_CUDA;
    }

    if (inframeId > 0) {
        auto ret = calcDiffWithPrevFrameAndSetDiffToCurr(inframeId + 0, inframeId - 1, queue_main);
        if (ret != RGY_ERR_NONE) {
            return ret;
        }
    }
    return sts;
}

void RGYFilterDecimate::close() {
    m_decimate.clear();
    m_eventDiff.reset();
    m_eventTransfer.reset();
    m_fpLog.reset();
}
