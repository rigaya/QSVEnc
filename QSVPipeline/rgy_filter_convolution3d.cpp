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
#include "rgy_filter_convolution3d.h"

static const int C3D_THRESHOLD_MIN = 0;
static const int C3D_THRESHOLD_MAX = 255;
static const int C3D_BLOCK_X       = 32;
static const int C3D_BLOCK_Y       = 8;

RGY_ERR RGYFilterConvolution3D::denoisePlane(
    RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pPrevPlane, const RGYFrameInfo *pInputPlane, const RGYFrameInfo *pNextPlane,
    const float threshold_spatial, const float threshold_temporal, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamConvolution3D>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    {
        const char *kernel_name = "kernel_convolution3d";
        RGYWorkSize local(C3D_BLOCK_X, C3D_BLOCK_Y);
        RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
        auto err = m_convolution3d.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
            (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
            (cl_mem)pPrevPlane->ptr[0], (cl_mem)pInputPlane->ptr[0], (cl_mem)pNextPlane->ptr[0], pInputPlane->pitch[0],
            pInputPlane->width, pInputPlane->height,
            threshold_spatial, threshold_temporal);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterConvolution3D::denoiseFrame(
    RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pPrevFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pNextFrame,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamConvolution3D>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        const RGY_PLANE ip = (RGY_PLANE)i;
        auto planeDst   = getPlane(pOutputFrame, ip);
        auto planePrev  = getPlane(pPrevFrame,   ip);
        auto planeInput = getPlane(pInputFrame,  ip);
        auto planeNext  = getPlane(pNextFrame,   ip);
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[pOutputFrame->csp] - 1) ? event : nullptr;
        const auto threshold_spatial  = (ip == RGY_PLANE_Y) ? prm->convolution3d.threshYspatial  : prm->convolution3d.threshCspatial;
        const auto threshold_temporal = (ip == RGY_PLANE_Y) ? prm->convolution3d.threshYtemporal : prm->convolution3d.threshCtemporal;
        const auto thresholdMul = (float)(1 << (RGY_CSP_BIT_DEPTH[planeInput.csp] - 8));
        auto err = denoisePlane(&planeDst, &planePrev, &planeInput, &planeNext, threshold_spatial * thresholdMul, threshold_temporal * thresholdMul, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to denoise(convolution3d) frame(%d) %s: %s\n"), i, RGY_CSP_NAMES[pInputFrame->csp], cl_errmes(err));
            return err_cl_to_rgy(err);
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterConvolution3D::RGYFilterConvolution3D(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_convolution3d(), m_prevFrames(), m_cacheIdx(0), m_frameOut(0) {
    m_name = _T("convolution3d");
}

RGYFilterConvolution3D::~RGYFilterConvolution3D() {
    close();
}

RGY_ERR RGYFilterConvolution3D::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamConvolution3D>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->convolution3d.threshYspatial < C3D_THRESHOLD_MIN || C3D_THRESHOLD_MAX < prm->convolution3d.threshYspatial) {
        AddMessage(RGY_LOG_ERROR, _T("ythresh must be a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->convolution3d.threshCspatial < C3D_THRESHOLD_MIN || C3D_THRESHOLD_MAX < prm->convolution3d.threshCspatial) {
        AddMessage(RGY_LOG_ERROR, _T("cthresh must be a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->convolution3d.threshYtemporal < C3D_THRESHOLD_MIN || C3D_THRESHOLD_MAX < prm->convolution3d.threshYtemporal) {
        AddMessage(RGY_LOG_ERROR, _T("t_ythresh must be a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->convolution3d.threshCtemporal < C3D_THRESHOLD_MIN || C3D_THRESHOLD_MAX < prm->convolution3d.threshCtemporal) {
        AddMessage(RGY_LOG_ERROR, _T("t_cthresh must be a positive value.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!m_convolution3d.get()
        || std::dynamic_pointer_cast<RGYFilterParamConvolution3D>(m_param)->convolution3d != prm->convolution3d) {
        int s0 = 0, s1 = 0, s2 = 0, t0 = 0, t1 = 0, t2 = 0;
        if (prm->convolution3d.matrix == VppConvolution3dMatrix::Standard) {
            s0 = 1, s1 = 2, s2 = 1;
            t0 = 1, t1 = 2, t2 = 1;
        } else if (prm->convolution3d.matrix == VppConvolution3dMatrix::Simple) {
            s0 = 1, s1 = 1, s2 = 1;
            t0 = 1, t1 = 1, t2 = 1;
        } else {
            AddMessage(RGY_LOG_ERROR, _T("Unknown matrix type: %d.\n"), (int)prm->convolution3d.matrix);
            return RGY_ERR_UNSUPPORTED;
        }
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d -D C3D_BLOCK_X=%d -D C3D_BLOCK_Y=%d -D C3D_FAST=%d "
                                        "-D C3D_S0=%d -D C3D_S1=%d -D C3D_S2=%d -D C3D_T0=%d -D C3D_T1=%d -D C3D_T2=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp],
            C3D_BLOCK_X, C3D_BLOCK_Y, prm->convolution3d.fast ? 1 : 0, s0, s1, s2, t0, t1, t2);
        m_convolution3d.set(m_cl->buildResourceAsync(_T("RGY_FILTER_CONVOLUTION3D_CL"), _T("EXE_DATA"), options.c_str()));
    }

    if (!m_prevFrames.front() ||
        cmpFrameInfoCspResolution(&m_prevFrames.front()->frame, &prm->frameOut)) {
        for (auto& f : m_prevFrames) {
            f = m_cl->createFrameBuffer(prm->frameOut);
            if (!f) {
                return RGY_ERR_NULL_PTR;
            }
        }
        m_cacheIdx = 0;
        m_frameOut = 0;
    }


    auto err = AllocFrameBuf(prm->frameOut, 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    m_pathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP));

    //コピーを保存
    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterConvolution3D::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;
    //if (interlaced(*pInputFrame)) {
    //    return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
    //}
    if (!m_convolution3d.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_CONVOLUTION3D_CL(m_convolution3d)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    auto param = std::dynamic_pointer_cast<RGYFilterParamConvolution3D>(m_param);
    if (!param) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pInputFrame->ptr[0] == nullptr && m_frameOut >= m_cacheIdx) {
        //終了
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return sts;
    }

    auto frameNext = pInputFrame;

    //十分な数のフレームがたまった、あるいはdrainモードならフレームを出力
    if (m_cacheIdx >= 1) {
        //出力先のフレーム
        RGYCLFrame *pOutFrame = nullptr;
        *pOutputFrameNum = 1;
        if (ppOutputFrames[0] == nullptr) {
            pOutFrame = m_frameBuf[0].get();
            ppOutputFrames[0] = &pOutFrame->frame;
        }
        if (pInputFrame->ptr[0]) {
            const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
            if (memcpyKind != RGYCLMemcpyD2D) {
                AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
                return RGY_ERR_UNSUPPORTED;
            }
            if (m_param->frameOut.csp != m_param->frameIn.csp) {
                AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
                return RGY_ERR_UNSUPPORTED;
            }
        }

        auto framePrev = &m_prevFrames[std::max(m_cacheIdx-2, 0) % m_prevFrames.size()]->frame;
        auto frameCur  = &m_prevFrames[        (m_cacheIdx-1)    % m_prevFrames.size()]->frame;
        if (frameNext->ptr[0] == nullptr) {
            frameNext = frameCur;
        }

        pOutFrame->frame.inputFrameId = frameCur->inputFrameId;
        pOutFrame->frame.duration     = frameCur->duration;
        pOutFrame->frame.timestamp    = frameCur->timestamp;

        sts = denoiseFrame(&pOutFrame->frame, framePrev, frameCur, frameNext, queue, wait_events, event);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at denoiseFrame(convolution3d)(%s): %s.\n"),
                       RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
            return sts;
        }
        m_frameOut++;
    } else {
        //出力フレームなし
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
    }
    //sourceキャッシュにコピー
    if (pInputFrame->ptr[0]) {
        auto cacheFrame = &m_prevFrames[m_cacheIdx++ % m_prevFrames.size()]->frame;
        sts = m_cl->copyFrame(cacheFrame, frameNext, nullptr, queue);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set frame to data cache: %s.\n"), get_err_mes(sts));
            return RGY_ERR_CUDA;
        }
        copyFrameProp(cacheFrame, frameNext);
    }
    return sts;
}

void RGYFilterConvolution3D::close() {
    m_frameBuf.clear();
    for (auto& f : m_prevFrames) {
        f.reset();
    }
    m_convolution3d.clear();
    m_cl.reset();
    m_bInterlacedWarn = false;
}
