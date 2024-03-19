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

#include <array>
#include <map>
#include "convert_csp.h"
#include "rgy_filter_denoise_dct.h"

#define DENOISE_BLOCK_SIZE_X (8) //ひとつのスレッドブロックの担当するx方向の8x8ブロックの数

#define DENOISE_SHARED_BLOCK_NUM_X (DENOISE_BLOCK_SIZE_X+2) //sharedメモリ上のx方向の8x8ブロックの数
#define DENOISE_SHARED_BLOCK_NUM_Y (2)                      //sharedメモリ上のy方向の8x8ブロックの数

#define DENOISE_LOOP_COUNT_BLOCK (8)

#define DCT_IDCT_BARRIER_ENABLE (1)

RGY_ERR RGYFilterDenoiseDct::denoiseDct(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseDct>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto planeInputR = getPlane(pInputFrame, RGY_PLANE_R);
    const auto planeInputG = getPlane(pInputFrame, RGY_PLANE_G);
    const auto planeInputB = getPlane(pInputFrame, RGY_PLANE_B);
    auto planeOutputR = getPlane(pOutputFrame, RGY_PLANE_R);
    auto planeOutputG = getPlane(pOutputFrame, RGY_PLANE_G);
    auto planeOutputB = getPlane(pOutputFrame, RGY_PLANE_B);
    if (planeInputR.pitch[0] != planeInputG.pitch[0] || planeInputR.pitch[0] != planeInputB.pitch[0]
        || planeOutputR.pitch[0] != planeOutputG.pitch[0] || planeOutputR.pitch[0] != planeOutputB.pitch[0]) {
        return RGY_ERR_UNKNOWN;
    }
    {
        const char *kernel_name = "kernel_denoise_dct";
        RGYWorkSize local(prm->dct.block_size, DENOISE_BLOCK_SIZE_X);
        RGYWorkSize global(divCeil(planeInputR.width, DENOISE_BLOCK_SIZE_X), divCeil(planeInputR.height, DENOISE_LOOP_COUNT_BLOCK), 3);
        auto err = m_dct.get()->kernel(kernel_name).config(queue, local, global).launch(
            (cl_mem)planeOutputR.ptr[0], (cl_mem)planeOutputG.ptr[0], (cl_mem)planeOutputB.ptr[0], planeOutputR.pitch[0],
            (cl_mem)planeInputR.ptr[0], (cl_mem)planeInputG.ptr[0], (cl_mem)planeInputB.ptr[0], planeInputR.pitch[0],
            planeInputR.width, planeInputR.height, m_threshold);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (denoiseDct): %s.\n"),
                char_to_tstring(kernel_name).c_str(), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseDct::colorDecorrelation(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue) {
    const auto planeInputR = getPlane(pInputFrame, RGY_PLANE_R);
    const auto planeInputG = getPlane(pInputFrame, RGY_PLANE_G);
    const auto planeInputB = getPlane(pInputFrame, RGY_PLANE_B);
    auto planeOutputR = getPlane(pOutputFrame, RGY_PLANE_R);
    auto planeOutputG = getPlane(pOutputFrame, RGY_PLANE_G);
    auto planeOutputB = getPlane(pOutputFrame, RGY_PLANE_B);
    if (   cmpFrameInfoCspResolution(&planeInputR, &planeOutputR)
        || cmpFrameInfoCspResolution(&planeInputG, &planeOutputG)
        || cmpFrameInfoCspResolution(&planeInputB, &planeOutputB)
        || cmpFrameInfoCspResolution(&planeInputR, &planeInputG)
        || cmpFrameInfoCspResolution(&planeInputR, &planeInputB)) {
        return RGY_ERR_UNKNOWN;
    }
    if (planeInputR.pitch[0] != planeInputG.pitch[0] || planeInputR.pitch[0] != planeInputB.pitch[0]
        || planeOutputR.pitch[0] != planeOutputG.pitch[0] || planeOutputR.pitch[0] != planeOutputB.pitch[0]) {
        return RGY_ERR_UNKNOWN;
    }
    {
        const char *kernel_name = "kernel_color_decorrelation";
        RGYWorkSize local(64, 4);
        RGYWorkSize global(planeInputR.width, planeInputR.height);
        auto err = m_dct.get()->kernel(kernel_name).config(queue, local, global).launch(
            (cl_mem)planeOutputR.ptr[0], (cl_mem)planeOutputG.ptr[0], (cl_mem)planeOutputB.ptr[0], planeOutputR.pitch[0],
            (cl_mem)planeInputR.ptr[0], (cl_mem)planeInputG.ptr[0], (cl_mem)planeInputB.ptr[0], planeInputR.pitch[0],
            planeInputR.width, planeInputR.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (colorDecorrelation): %s.\n"),
                char_to_tstring(kernel_name).c_str(), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseDct::colorCorrelation(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue) {
    const auto planeInputR = getPlane(pInputFrame, RGY_PLANE_R);
    const auto planeInputG = getPlane(pInputFrame, RGY_PLANE_G);
    const auto planeInputB = getPlane(pInputFrame, RGY_PLANE_B);
    auto planeOutputR = getPlane(pOutputFrame, RGY_PLANE_R);
    auto planeOutputG = getPlane(pOutputFrame, RGY_PLANE_G);
    auto planeOutputB = getPlane(pOutputFrame, RGY_PLANE_B);
    if (   cmpFrameInfoCspResolution(&planeInputR, &planeOutputR)
        || cmpFrameInfoCspResolution(&planeInputG, &planeOutputG)
        || cmpFrameInfoCspResolution(&planeInputB, &planeOutputB)
        || cmpFrameInfoCspResolution(&planeInputR, &planeInputG)
        || cmpFrameInfoCspResolution(&planeInputR, &planeInputB)) {
        return RGY_ERR_UNKNOWN;
    }
    if (planeInputR.pitch[0] != planeInputG.pitch[0] || planeInputR.pitch[0] != planeInputB.pitch[0]
        || planeOutputR.pitch[0] != planeOutputG.pitch[0] || planeOutputR.pitch[0] != planeOutputB.pitch[0]) {
        return RGY_ERR_UNKNOWN;
    }
    {
        const char *kernel_name = "kernel_color_correlation";
        RGYWorkSize local(64, 4);
        RGYWorkSize global(planeInputR.width, planeInputR.height);
        auto err = m_dct.get()->kernel(kernel_name).config(queue, local, global).launch(
            (cl_mem)planeOutputR.ptr[0], (cl_mem)planeOutputG.ptr[0], (cl_mem)planeOutputB.ptr[0], planeOutputR.pitch[0],
            (cl_mem)planeInputR.ptr[0], (cl_mem)planeInputG.ptr[0], (cl_mem)planeInputB.ptr[0], planeInputR.pitch[0],
            planeInputR.width, planeInputR.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (colorCorrelation): %s.\n"),
                char_to_tstring(kernel_name).c_str(), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseDct::denoise(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseDct>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    RGYCLFrame *bufDst = m_bufImg[0].get();
    {
        RGYFrameInfo srcImgInfo = m_srcCrop->GetFilterParam()->frameOut;
        int cropFilterOutputNum = 0;
        RGYFrameInfo *outInfo[1] = { &bufDst->frame };
        RGYFrameInfo cropInput = *pInputFrame;
        auto sts_filter = m_srcCrop->filter(&cropInput, (RGYFrameInfo **)&outInfo, &cropFilterOutputNum, queue, wait_events, nullptr);
        if (outInfo[0] == nullptr || cropFilterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_srcCrop->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || cropFilterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_srcCrop->name().c_str());
            return sts_filter;
        }
    }
    RGYCLFrame *bufSrc = bufDst;
    bufDst = m_bufImg[1].get();
    auto sts = colorDecorrelation(&bufDst->frame, &bufSrc->frame, queue);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    std::swap(bufSrc, bufDst);
    sts = denoiseDct(&bufDst->frame, &bufSrc->frame, queue);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    std::swap(bufSrc, bufDst);
    sts = colorCorrelation(&bufDst->frame, &bufSrc->frame, queue);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    std::swap(bufSrc, bufDst);
    {
        int cropFilterOutputNum = 0;
        RGYFrameInfo *outInfo[1] = { pOutputFrame };
        auto sts_filter = m_dstCrop->filter(&bufSrc->frame, outInfo, &cropFilterOutputNum, queue, event);
        if (outInfo[0] == nullptr || cropFilterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_dstCrop->name().c_str());
            return sts_filter;
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterDenoiseDct::RGYFilterDenoiseDct(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_step(-1),
    m_threshold(0.0f),
    m_srcCrop(),
    m_dstCrop(),
    m_bufImg(),
    m_dct() {
    m_name = _T("denoise-dct");
}

RGYFilterDenoiseDct::~RGYFilterDenoiseDct() {
    close();
}

RGY_ERR RGYFilterDenoiseDct::checkParam(const RGYFilterParamDenoiseDct *prm) {
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dct.sigma < 0.0f) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, sigma must be a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (get_cx_index(list_vpp_denoise_dct_block_size, prm->dct.block_size) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid block_size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseDct::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseDct>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if ((sts = checkParam(prm.get())) != RGY_ERR_NONE) {
        return sts;
    }
    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamDenoiseDct>(m_param);
    if (!m_dct.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
        || prmPrev->dct.block_size != prm->dct.block_size
        || prmPrev->dct.step != prm->dct.step) {
        {
            AddMessage(RGY_LOG_DEBUG, _T("Create input csp conversion filter.\n"));
            std::unique_ptr<RGYFilterCspCrop> filter(new RGYFilterCspCrop(m_cl));
            std::shared_ptr<RGYFilterParamCrop> paramCrop(new RGYFilterParamCrop());
            paramCrop->frameIn = pParam->frameIn;
            paramCrop->frameOut = paramCrop->frameIn;
            paramCrop->frameOut.csp = RGY_CSP_RGB_F32;
            paramCrop->baseFps = pParam->baseFps;
            paramCrop->frameIn.mem_type = RGY_MEM_TYPE_GPU;
            paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
            paramCrop->bOutOverwrite = false;
            sts = filter->init(paramCrop, m_pLog);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            m_srcCrop = std::move(filter);
            AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_srcCrop->GetInputMessage().c_str());
        }
        {
            AddMessage(RGY_LOG_DEBUG, _T("Create output csp conversion filter.\n"));
            std::unique_ptr<RGYFilterCspCrop> filter(new RGYFilterCspCrop(m_cl));
            std::shared_ptr<RGYFilterParamCrop> paramCrop(new RGYFilterParamCrop());
            paramCrop->frameIn = m_srcCrop->GetFilterParam()->frameOut;
            paramCrop->frameOut = pParam->frameOut;
            paramCrop->baseFps = pParam->baseFps;
            paramCrop->frameIn.mem_type = RGY_MEM_TYPE_GPU;
            paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
            paramCrop->bOutOverwrite = false;
            sts = filter->init(paramCrop, m_pLog);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            m_dstCrop = std::move(filter);
            AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_dstCrop->GetInputMessage().c_str());
        }
        for (auto& buf : m_bufImg) {
            if (!buf || cmpFrameInfoCspResolution(&buf->frame, &m_srcCrop->GetFilterParam()->frameOut)) {
                buf = m_cl->createFrameBuffer(m_srcCrop->GetFilterParam()->frameOut);
                if (!buf) {
                    return RGY_ERR_NULL_PTR;
                }
            }
        }
        const auto sub_group_ext_avail = m_cl->platform()->checkSubGroupSupport(m_cl->queue().devid());
        const int dct_idct_barrier_mode = (DCT_IDCT_BARRIER_ENABLE) ? (sub_group_ext_avail != RGYOpenCLSubGroupSupport::NONE ? 2 : 1) : 0;
        auto options = strsprintf("-D TypePixel=float -D bit_depth=32 -D TypeTmp=float -D BLOCK_SIZE=%d -D STEP=%d"
            " -D DENOISE_BLOCK_SIZE_X=%d -D DENOISE_SHARED_BLOCK_NUM_X=%d -D DENOISE_SHARED_BLOCK_NUM_Y=%d -D DENOISE_LOOP_COUNT_BLOCK=%d -D DCT_IDCT_BARRIER_MODE=%d",
            prm->dct.block_size, prm->dct.step,
            DENOISE_BLOCK_SIZE_X, DENOISE_SHARED_BLOCK_NUM_X, DENOISE_SHARED_BLOCK_NUM_Y, DENOISE_LOOP_COUNT_BLOCK, dct_idct_barrier_mode);
        if (dct_idct_barrier_mode > 0 && sub_group_ext_avail == RGYOpenCLSubGroupSupport::STD20KHR) {
            options += " -cl-std=CL2.0";
        }
        m_dct.set(m_cl->buildResourceAsync(_T("RGY_FILTER_DENOISE_DCT_CL"), _T("EXE_DATA"), options.c_str()));

        auto err = AllocFrameBuf(prm->frameOut, 1);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
            return RGY_ERR_MEMORY_ALLOC;
        }
        for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
            prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
        }

        m_step = prm->dct.step;
        m_threshold = prm->dct.sigma * 3.0f / 255.0f;
    }

    setFilterInfo(pParam->print());
    m_param = pParam;
    return sts;
}

tstring RGYFilterParamDenoiseDct::print() const {
    return dct.print();
}

RGY_ERR RGYFilterDenoiseDct::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
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
    if (!m_dct.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_DENOISE_DCT_CL(m_dct)\n"));
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

    sts = denoise(ppOutputFrames[0], pInputFrame, queue, wait_events, event);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at denoiseFrame (%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }

    return sts;
}

void RGYFilterDenoiseDct::close() {
    m_srcCrop.reset();
    m_dstCrop.reset();
}
