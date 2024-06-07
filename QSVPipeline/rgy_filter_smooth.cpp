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
#include "rgy_filter_smooth.h"

#define SPP_THREAD_BLOCK_X (8) //blockDim.x
#define SPP_THREAD_BLOCK_Y (8) //blockDim.y

#define SPP_BLOCK_SIZE_X (8) //ひとつのスレッドブロックの担当するx方向の8x8ブロックの数

#define SPP_SHARED_BLOCK_NUM_X (SPP_BLOCK_SIZE_X+2) //sharedメモリ上のx方向の8x8ブロックの数
#define SPP_SHARED_BLOCK_NUM_Y (2)                  //sharedメモリ上のy方向の8x8ブロックの数

#define SPP_LOOP_COUNT_BLOCK (8)

#define DCT_IDCT_BARRIER_ENABLE (1)

RGY_ERR RGYFilterSmooth::procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const RGYFrameInfo *targetQPTable, const int qpBlockShift, const float qpMul, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSmooth>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    {
        const char *kernel_name = "kernel_smooth";
        RGYWorkSize local(SPP_THREAD_BLOCK_X, SPP_THREAD_BLOCK_Y);
        RGYWorkSize global(divCeil(pOutputPlane->width, SPP_BLOCK_SIZE_X), divCeil(pOutputPlane->height, SPP_LOOP_COUNT_BLOCK));

        const float W = 5.0f;
        const float thresh_a = (prm->smooth.threshold + W) / (2.0f * W);
        const float thresh_b = (W * W - prm->smooth.threshold * prm->smooth.threshold) / (2.0f * W);
        auto err = m_smooth.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
            (cl_mem)pOutputPlane->ptr[0],
            (cl_mem)pInputPlane->ptr[0],
            pOutputPlane->pitch[0],
            pOutputPlane->width,
            pOutputPlane->height,
            (cl_mem)targetQPTable->ptr[0],
            targetQPTable->pitch[0],
            targetQPTable->width,
            targetQPTable->height,
            qpBlockShift, qpMul,
            prm->smooth.quality, prm->smooth.strength,
            thresh_a, thresh_b);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSmooth::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *targetQPTable, const float qpMul, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto srcImage = m_cl->createImageFromFrameBuffer(*pInputFrame, true, CL_MEM_READ_ONLY, &m_srcImagePool);
    if (!srcImage) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to create image for input frame.\n"));
        return RGY_ERR_MEM_OBJECT_ALLOCATION_FAILURE;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(&srcImage->frame, (RGY_PLANE)i);
        const int qpBlockShift = (i > 0 && RGY_CSP_CHROMA_FORMAT[pOutputFrame->csp] == RGY_CHROMAFMT_YUV420) ? 0 : 1;
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[pOutputFrame->csp] - 1) ? event : nullptr;
        auto err = procPlane(&planeDst, &planeSrc, targetQPTable, qpBlockShift, qpMul, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to denoise(smooth) frame(%d): %s\n"), i, cl_errmes(err));
            return err_cl_to_rgy(err);
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSmooth::setQP(RGYCLFrame *targetQPTable, const int qp, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *kernel_name = "kernel_smooth_set_qp";
    RGYWorkSize local(64, 4);
    RGYWorkSize global(divCeil(targetQPTable->frame.width, 4), targetQPTable->frame.height);
    auto err = m_smooth.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)targetQPTable->frame.ptr[0], targetQPTable->frame.pitch[0], targetQPTable->frame.width, targetQPTable->frame.height, qp);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (setQP(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[targetQPTable->frame.csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}


RGYFilterSmooth::RGYFilterSmooth(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_smooth(), m_qp(), m_qpSrc(), m_qpSrcB(), m_qpTableRef(nullptr), m_qpTableErrCount(0), m_srcImagePool() {
    m_name = _T("smooth");
}

RGYFilterSmooth::~RGYFilterSmooth() {
    close();
}

float RGYFilterSmooth::getQPMul(int qpScaleType) {
    switch (qpScaleType) {
    case 0/*mpeg1*/: return 4.0f;
    case 1/*mpeg2*/: return 2.0f;
    case 2/*h264*/:  return 1.0f;
    case 3/*VP56*/:  //return (63 - qscale + 2);
    default:
        return 0.0f;
    }
}

RGY_ERR RGYFilterSmooth::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSmooth>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->smooth.quality < 0 || prm->smooth.quality > VPP_SMOOTH_MAX_QUALITY_LEVEL) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (quality).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->smooth.qp <= 0 || prm->smooth.qp > 63) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (qp).\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamSmooth>(m_param);
    if (!m_param
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
        || prmPrev->smooth.prec != prm->smooth.prec) {
        if (prm->smooth.prec != VPP_FP_PRECISION_FP32) {
            if (!RGYOpenCLDevice(m_cl->queue().devid()).checkExtension("cl_khr_fp16")) {
                AddMessage((prm->smooth.prec == VPP_FP_PRECISION_FP16) ? RGY_LOG_WARN : RGY_LOG_DEBUG, _T("fp16 not supported on this device, using fp32 mode.\n"));
                prm->smooth.prec = VPP_FP_PRECISION_FP32;
            }
        }
        const auto sub_group_ext_avail = m_cl->platform()->checkSubGroupSupport(m_cl->queue().devid());
        const bool cl_fp16_support = prm->smooth.prec != VPP_FP_PRECISION_FP32;
        const bool usefp16DctFirst = cl_fp16_support
            && sub_group_ext_avail != RGYOpenCLSubGroupSupport::NONE
            && prm->smooth.prec == VPP_FP_PRECISION_FP16
            && prm->smooth.quality > 0; // quality = 0の時には適用してはならない
        m_smooth.set(std::async(std::launch::async,
            [cl = m_cl, log = m_pLog, cl_fp16_support, sub_group_ext_avail, usefp16DctOrg = usefp16DctFirst, frameOut = prm->frameOut]() {

            const int dct_idct_barrier_mode = (DCT_IDCT_BARRIER_ENABLE) ? (sub_group_ext_avail != RGYOpenCLSubGroupSupport::NONE ? 2 : 1) : 0;
            auto gen_options = [&](bool enable_fp16, bool cl_fp16_support) {
                auto options = strsprintf("-D TypePixel=%s -D bit_depth=%d"
                    " -D usefp16Dct=%d -D usefp16IO=%d -D TypeQP=uchar -D TypeQP4=uchar4"
                    " -D SPP_BLOCK_SIZE_X=%d"
                    " -D SPP_THREAD_BLOCK_X=%d -D SPP_THREAD_BLOCK_Y=%d"
                    " -D SPP_SHARED_BLOCK_NUM_X=%d -D SPP_SHARED_BLOCK_NUM_Y=%d"
                    " -D SPP_LOOP_COUNT_BLOCK=%d -D DCT_IDCT_BARRIER_MODE=%d",
                    RGY_CSP_BIT_DEPTH[frameOut.csp] > 8 ? "ushort" : "uchar",
                    RGY_CSP_BIT_DEPTH[frameOut.csp],
                    (enable_fp16) ? 1 : 0,
                    (cl_fp16_support) ? 1 : 0,
                    SPP_BLOCK_SIZE_X,
                    SPP_THREAD_BLOCK_X, SPP_THREAD_BLOCK_Y,
                    SPP_SHARED_BLOCK_NUM_X, SPP_SHARED_BLOCK_NUM_Y,
                    SPP_LOOP_COUNT_BLOCK,
                    dct_idct_barrier_mode
                );
                if (dct_idct_barrier_mode > 0 && sub_group_ext_avail == RGYOpenCLSubGroupSupport::STD20KHR) {
                    options += " -cl-std=CL2.0";
                }
                return options;
            };
            auto usefp16Dct = usefp16DctOrg;
            auto smooth = cl->buildResource(_T("RGY_FILTER_SMOOTH_CL"), _T("EXE_DATA"), gen_options(usefp16Dct, cl_fp16_support).c_str());
            if (!smooth) {
                log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to load RGY_FILTER_SMOOTH_CL(m_smooth)\n"));
                return std::unique_ptr<RGYOpenCLProgram>();
            }
            if (usefp16Dct) {
                RGYWorkSize local(SPP_THREAD_BLOCK_X, SPP_THREAD_BLOCK_Y);
                RGYWorkSize global(divCeil(frameOut.width, SPP_BLOCK_SIZE_X), divCeil(frameOut.height, SPP_LOOP_COUNT_BLOCK));
                const auto subGroupSize = smooth->kernel("kernel_smooth").config(cl->queue(), local, global).subGroupSize();
                if (subGroupSize == 0) {
                    if (usefp16Dct) {
                        log->write(RGY_LOG_WARN, RGY_LOGT_VPP, _T("Could not get subGroupSize for kernel, fp16 dct disabled.\n"));
                        usefp16Dct = false;
                    }
                } else if ((subGroupSize & (subGroupSize - 1)) != 0) {
                    log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("subGroupSize(%d) is not pow2!\n"), subGroupSize);
                    return std::unique_ptr<RGYOpenCLProgram>();
                } else if (subGroupSize < 8) {
                    log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("subGroupSize(%d) < 8 !\n"), subGroupSize);
                    return std::unique_ptr<RGYOpenCLProgram>();
                } else if (subGroupSize < 32) {
                    if (usefp16Dct) {
                        log->write(RGY_LOG_WARN, RGY_LOGT_VPP, _T("subGroupSize(%d) < 32, fp16 dct disabled.\n"), subGroupSize);
                        usefp16Dct = false;
                    }
                }
                if (usefp16DctOrg && !usefp16Dct) {
                    log->write(RGY_LOG_DEBUG, RGY_LOGT_VPP, _T("Use fp16 opt: subGroupSize=%d.\n"), subGroupSize);
                    smooth = cl->buildResource(_T("RGY_FILTER_SMOOTH_CL"), _T("EXE_DATA"), gen_options(false, cl_fp16_support).c_str());
                    if (!smooth) {
                        return std::unique_ptr<RGYOpenCLProgram>();
                    }
                }
            }
            return smooth;
        }));
    }
    const auto qpframe = RGYFrameInfo(qp_size(pParam->frameIn.width), qp_size(pParam->frameIn.height), RGY_CSP_Y8, RGY_CSP_BIT_DEPTH[RGY_CSP_Y8], RGY_PICSTRUCT_FRAME, RGY_MEM_TYPE_CPU);
    if (!m_qp || m_qp->frame.width != qpframe.width || m_qp->frame.height != qpframe.height) {
        m_qp = m_cl->createFrameBuffer(qpframe);
        if (!m_qp) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for qp table.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        AddMessage(RGY_LOG_DEBUG, _T("allocated qp table buffer: %dx%d, pitch %d, %s.\n"),
                   m_qp->frame.width, m_qp->frame.height, m_qp->frame.pitch, RGY_CSP_NAMES[m_qp->frame.csp]);
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

RGY_ERR RGYFilterSmooth::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
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
    if (!m_smooth.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build RGY_FILTER_SMOOTH_CL(m_smooth)\n"));
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

    auto prm = std::dynamic_pointer_cast<RGYFilterParamSmooth>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //入力フレームのQPテーブルへの参照を取得
    std::shared_ptr<RGYFrameDataQP> qpInput;
    if (prm->smooth.useQPTable) {
        for (auto &data : pInputFrame->dataList) {
            if (data->dataType() == RGY_FRAME_DATA_QP) {
                auto ptr = dynamic_cast<RGYFrameDataQP *>(data.get());
                if (ptr == nullptr) {
                    AddMessage(RGY_LOG_ERROR, _T("Failed to get RGYFrameDataQP.\n"));
                    return RGY_ERR_UNSUPPORTED;
                }
                auto ptrRef = m_qpTableRef->get(ptr);
                if (!ptrRef) {
                    AddMessage(RGY_LOG_ERROR, _T("Failed to get ref to RGYFrameDataQP.\n"));
                    return RGY_ERR_UNSUPPORTED;
                }
                qpInput = std::move(ptrRef);
            }
        }
        if (!qpInput) {
            m_qpTableErrCount++;
            AddMessage(RGY_LOG_DEBUG, _T("Failed to get qp table from input file %d: inputID %d, %lld\n"), m_qpTableErrCount, pInputFrame->inputFrameId, pInputFrame->timestamp);
            if (m_qpTableErrCount >= prm->smooth.maxQPTableErrCount) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to get qp table from input file for more than %d times, please specify \"qp\" for --vpp-smooth.\n"), m_qpTableErrCount);
                return RGY_ERR_UNSUPPORTED;
            }
            //ひとまず、前のQPテーブルで代用する
            qpInput = m_qpSrc;
        } else {
            m_qpTableErrCount = 0;
        }
    }

    std::vector<RGYOpenCLEvent> wait_events_copy = wait_events;

    //実際に計算用に使用するQPテーブルの選択、あるいは作成
    RGYCLFrame *targetQPTable = nullptr;
    float qpMul = 1.0f;
#if 0
    if (!!qpInput) {
        auto cudaerr = cudaStreamWaitEvent(stream, qpInput->event(), 0);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("error in cudaStreamWaitEvent(): %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
            return RGY_ERR_MEMORY_ALLOC;
        }
        qpMul = getQPMul(qpInput->qpScaleType());
        if (qpMul <= 0.0f) {
            AddMessage(RGY_LOG_ERROR, _T("Unsupported qp scale type.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        const bool isBFrame = qpInput->frameType() == 3;
        if (isBFrame) {
            m_qpSrcB = qpInput;
            targetQPTable = m_qp.get();
            cudaerr = run_gen_qp_table<uchar4>(&m_qp.frame, &m_qpSrc->qpDev()->frame, &m_qpSrcB->qpDev()->frame, qpMul, prm->smooth.bratio, stream);
            if (cudaerr != cudaSuccess) {
                AddMessage(RGY_LOG_ERROR, _T("error in run_set_qp(): %s.\n"), char_to_tstring(cudaGetErrorName(cudaerr)).c_str());
                return RGY_ERR_MEMORY_ALLOC;
            }
            qpMul = 1.0f; //run_gen_qp_tableの中で反映済み
        } else {
            m_qpSrc = qpInput;
            targetQPTable = m_qpSrc->qpDev();
        }
    } else
#endif
    {
        targetQPTable = m_qp.get();
        auto err = setQP(targetQPTable, prm->smooth.qp, queue, wait_events_copy, event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error in setQP(): %s.\n"), get_err_mes(err));
            return RGY_ERR_MEMORY_ALLOC;
        }
        wait_events_copy.clear();
    }

    sts = procFrame(ppOutputFrames[0], pInputFrame, &targetQPTable->frame, qpMul, queue, wait_events_copy, event);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at procFrame (%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }

    return sts;
}

void RGYFilterSmooth::close() {
    m_srcImagePool.clear();
    m_frameBuf.clear();
    m_smooth.clear();
    m_qp.reset();
    m_qpSrc.reset();
    m_qpSrcB.reset();
    m_cl.reset();
}
