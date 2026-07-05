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
#include <cmath>
#include "convert_csp.h"
#include "rgy_filter_denoise_fft3d.h"
#include "rgy_resource.h"

#define FFT_M_PI (3.14159265358979323846f)

static const int MAX_BLOCK = 64;

static constexpr int log2u(int n) {
    int x = -1;
    while (n > 0) {
        x++;
        n >>= 1;
    }
    return x;
}

// intのbitを逆順に並び替える
static int bitreverse(const int bitlength, int x) {
    int y = 0;
    for (int i = 0; i < bitlength; i++) {
        y = (y << 1) + (x & 1);
        x >>= 1;
    }
    return y;
}

static int getDenoiseBlockSizeX(const int block_size) {
    return std::max(64 / block_size, 1);
}

static std::pair<int, int> getBlockCount(const int width, const int height, const int block_size, const int ov1, const int ov2) {
    const int block_eff = block_size - ov1 * 2 - ov2;
    const int block_count_x = (width + block_eff - 1) / block_eff;
    const int block_count_y = (height + block_eff - 1) / block_eff;
    return std::make_pair(block_count_x, block_count_y);
}

// Resolve the effective temporal radius bt. bt overrides 'temporal' when set
// (!= 0); otherwise fall back to the legacy temporal flag (0 -> bt1 spatial,
// 1 -> bt3 prev+cur+next). bt=-1 = sharpen/degrid only (no denoising), which
// processes a single frame like bt=1. Frame layout per bt:
//   bt1 [cur]                 nPast=0 nFuture=0 curIdx=0
//   bt2 [prev,cur]            nPast=1 nFuture=0 curIdx=1
//   bt3 [prev,cur,next]       nPast=1 nFuture=1 curIdx=1
//   bt4 [prev2,prev,cur,next] nPast=2 nFuture=1 curIdx=2
static int fft3d_bt(const VppDenoiseFFT3D &f) {
    const int bt = (f.bt != 0) ? f.bt : (f.temporal ? 3 : 1);
    return (bt < -1) ? -1 : (bt > 4) ? 4 : bt;
}

// number of frames processed together (bt=-1 -> 1)
static int fft3d_bt_frames(const VppDenoiseFFT3D &f) {
    return std::max(fft3d_bt(f), 1);
}

RGY_ERR RGYFilterDenoiseFFT3D::denoiseFFT(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseFFT3D>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const char *kernel_name = "kernel_fft";
    const int denoiseBlockSizeX = getDenoiseBlockSizeX(prm->fft3d.block_size);
    {
        auto block_count = getBlockCount(pInputFrame->width, pInputFrame->height, prm->fft3d.block_size, m_ov1, m_ov2);
        const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
        auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
        RGYWorkSize local(prm->fft3d.block_size, denoiseBlockSizeX);
        RGYWorkSize global(divCeil(block_count.first, denoiseBlockSizeX) * local(0), block_count.second * local(1), 1);
        auto err = m_fft3d.get()->kernel(kernel_name).config(queue, local, global, wait_events).launch(
            (cl_mem)planeOutputY.ptr[0], (cl_mem)nullptr, planeOutputY.pitch[0],
            (cl_mem)planeInputY.ptr[0], (cl_mem)nullptr, planeInputY.pitch[0],
            planeInputY.width, planeInputY.height, block_count.first,
            m_windowBuf->mem(),
            m_ov1, m_ov2
        );
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    {
        const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
        const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
        auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
        auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
        if (!prm->processChroma) {
            return RGY_ERR_NONE;
        }
        if (planeOutputU.pitch[0] != planeOutputV.pitch[0]) {
            return RGY_ERR_UNKNOWN;
        }
        auto block_count = getBlockCount(planeInputU.width, planeInputU.height, prm->fft3d.block_size, m_ov1, m_ov2);
        RGYWorkSize local(prm->fft3d.block_size, denoiseBlockSizeX);
        RGYWorkSize global(divCeil(block_count.first, denoiseBlockSizeX) * local(0), block_count.second * local(1), 2);
        auto err = m_fft3d.get()->kernel(kernel_name).config(queue, local, global).launch(
            (cl_mem)planeOutputU.ptr[0], (cl_mem)planeOutputV.ptr[0], planeOutputU.pitch[0],
            (cl_mem)planeInputU.ptr[0], (cl_mem)planeInputV.ptr[0], planeInputU.pitch[0],
            planeInputU.width, planeInputU.height, block_count.first,
            m_windowBuf->mem(),
            m_ov1, m_ov2
        );
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseFFT3D::denoiseTFFTFilterIFFT(RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pInputFrameA, const RGYFrameInfo *pInputFrameB, const RGYFrameInfo *pInputFrameC, const RGYFrameInfo *pInputFrameD,
    RGYOpenCLQueue &queue) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseFFT3D>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const char *kernel_name = "kernel_tfft_filter_ifft";
    const int denoiseBlockSizeX = getDenoiseBlockSizeX(prm->fft3d.block_size);
    const float limit = 1.0f - prm->fft3d.amount;
    // sharpen limits, same 8bit-referenced normalisation as sigma (signorm: real noise-power units)
    const float scale = (1.0f / ((1 << 8) - 1));
    const float nGain = (prm->fft3d.signorm) ? m_noisePowerGain : 1.0f;
    const float sminSq = (prm->fft3d.smin * scale) * (prm->fft3d.smin * scale) * nGain;
    const float smaxSq = (prm->fft3d.smax * scale) * (prm->fft3d.smax * scale) * nGain;
    const float degridFactor = (m_gridBuf && m_gridDC > 0.0f) ? prm->fft3d.degrid / m_gridDC : 0.0f;
    const cl_mem sigmaMem    = m_sigmaBuf->mem();
    const cl_mem wsharpenMem = (m_wsharpenBuf) ? m_wsharpenBuf->mem() : (cl_mem)nullptr;
    const cl_mem gridMem     = (m_gridBuf) ? m_gridBuf->mem() : (cl_mem)nullptr;
    {
        const auto block_count = getBlockCount(prm->frameOut.width, prm->frameOut.height, prm->fft3d.block_size, m_ov1, m_ov2);
        const auto planeInputYA = (pInputFrameA) ? getPlane(pInputFrameA, RGY_PLANE_Y) : RGYFrameInfo();
        const auto planeInputYB = (pInputFrameB) ? getPlane(pInputFrameB, RGY_PLANE_Y) : RGYFrameInfo();
        const auto planeInputYC = (pInputFrameC) ? getPlane(pInputFrameC, RGY_PLANE_Y) : RGYFrameInfo();
        const auto planeInputYD = (pInputFrameD) ? getPlane(pInputFrameD, RGY_PLANE_Y) : RGYFrameInfo();
        auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
        RGYWorkSize local(prm->fft3d.block_size, denoiseBlockSizeX);
        RGYWorkSize global(divCeil(block_count.first, denoiseBlockSizeX) * local(0), block_count.second * local(1), 1);
        auto err = m_fft3d.get()->kernel(kernel_name).config(queue, local, global).launch(
            (cl_mem)planeOutputY.ptr[0], (cl_mem)nullptr,
            planeOutputY.pitch[0],
            (cl_mem)planeInputYA.ptr[0], (cl_mem)nullptr,
            (cl_mem)planeInputYB.ptr[0], (cl_mem)nullptr,
            (cl_mem)planeInputYC.ptr[0], (cl_mem)nullptr,
            (cl_mem)planeInputYD.ptr[0], (cl_mem)nullptr,
            planeInputYA.pitch[0],
            block_count.first,
            m_windowBufInverse->mem(),
            m_ov1, m_ov2,
            sigmaMem, limit,
            wsharpenMem, sminSq, smaxSq, // sharpen: luma only
            gridMem, degridFactor
        );
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    {
        auto planeUV = getPlane(&prm->frameOut, RGY_PLANE_U);
        const auto planeInputUA = (pInputFrameA) ? getPlane(pInputFrameA, RGY_PLANE_U) : RGYFrameInfo();
        const auto planeInputVA = (pInputFrameA) ? getPlane(pInputFrameA, RGY_PLANE_V) : RGYFrameInfo();
        const auto planeInputUB = (pInputFrameB) ? getPlane(pInputFrameB, RGY_PLANE_U) : RGYFrameInfo();
        const auto planeInputVB = (pInputFrameB) ? getPlane(pInputFrameB, RGY_PLANE_V) : RGYFrameInfo();
        const auto planeInputUC = (pInputFrameC) ? getPlane(pInputFrameC, RGY_PLANE_U) : RGYFrameInfo();
        const auto planeInputVC = (pInputFrameC) ? getPlane(pInputFrameC, RGY_PLANE_V) : RGYFrameInfo();
        const auto planeInputUD = (pInputFrameD) ? getPlane(pInputFrameD, RGY_PLANE_U) : RGYFrameInfo();
        const auto planeInputVD = (pInputFrameD) ? getPlane(pInputFrameD, RGY_PLANE_V) : RGYFrameInfo();
        auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
        auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
        if (!prm->processChroma) {
            return RGY_ERR_NONE;
        }
        if (planeOutputU.pitch[0] != planeOutputV.pitch[0]) {
            return RGY_ERR_UNKNOWN;
        }
        const auto block_count = getBlockCount(planeUV.width, planeUV.height, prm->fft3d.block_size, m_ov1, m_ov2);
        RGYWorkSize local(prm->fft3d.block_size, denoiseBlockSizeX);
        RGYWorkSize global(divCeil(block_count.first, denoiseBlockSizeX) * local(0), block_count.second * local(1), 2);
        auto err = m_fft3d.get()->kernel(kernel_name).config(queue, local, global).launch(
            (cl_mem)planeOutputU.ptr[0], (cl_mem)planeOutputV.ptr[0],
            planeOutputU.pitch[0],
            (cl_mem)planeInputUA.ptr[0], (cl_mem)planeInputVA.ptr[0],
            (cl_mem)planeInputUB.ptr[0], (cl_mem)planeInputVB.ptr[0],
            (cl_mem)planeInputUC.ptr[0], (cl_mem)planeInputVC.ptr[0],
            (cl_mem)planeInputUD.ptr[0], (cl_mem)planeInputVD.ptr[0],
            planeInputUA.pitch[0],
            block_count.first,
            m_windowBufInverse->mem(),
            m_ov1, m_ov2,
            sigmaMem, limit,
            (cl_mem)nullptr, sminSq, smaxSq, // no sharpening on chroma (matches the original filter's luma-default; avoids chroma ringing)
            gridMem, degridFactor
        );
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseFFT3D::denoiseMerge(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseFFT3D>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const char *kernel_name = "kernel_merge";
    {
        const auto planeInputY = getPlane(pInputFrame, RGY_PLANE_Y);
        auto planeOutputY = getPlane(pOutputFrame, RGY_PLANE_Y);
        const auto block_count = getBlockCount(planeOutputY.width, planeOutputY.height, prm->fft3d.block_size, m_ov1, m_ov2);
        RGYWorkSize local(32, 8);
        RGYWorkSize global(planeOutputY.width, planeOutputY.height, 1);
        auto err = m_fft3d.get()->kernel(kernel_name).config(queue, local, global, wait_events, prm->processChroma ? nullptr : event).launch(
            (cl_mem)planeOutputY.ptr[0], (cl_mem)nullptr, planeOutputY.pitch[0],
            (cl_mem)planeInputY.ptr[0], (cl_mem)nullptr, planeInputY.pitch[0],
            planeOutputY.width, planeOutputY.height, block_count.first, block_count.second, m_ov1, m_ov2);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    {
        const auto planeInputU = getPlane(pInputFrame, RGY_PLANE_U);
        const auto planeInputV = getPlane(pInputFrame, RGY_PLANE_V);
        auto planeOutputU = getPlane(pOutputFrame, RGY_PLANE_U);
        auto planeOutputV = getPlane(pOutputFrame, RGY_PLANE_V);
        if (!prm->processChroma) {
            return RGY_ERR_NONE;
        }
        if (planeOutputU.pitch[0] != planeOutputV.pitch[0]) {
            return RGY_ERR_UNKNOWN;
        }
        const auto block_count = getBlockCount(planeOutputU.width, planeOutputU.height, prm->fft3d.block_size, m_ov1, m_ov2);
        RGYWorkSize local(32, 8);
        RGYWorkSize global(planeOutputU.width, planeOutputU.height, 2);
        auto err = m_fft3d.get()->kernel(kernel_name).config(queue, local, global, event).launch(
            (cl_mem)planeOutputU.ptr[0], (cl_mem)planeOutputV.ptr[0], planeOutputU.pitch[0],
            (cl_mem)planeInputU.ptr[0], (cl_mem)planeInputV.ptr[0], planeInputU.pitch[0],
            planeOutputU.width, planeOutputU.height, block_count.first, block_count.second, m_ov1, m_ov2);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    return RGY_ERR_NONE;
}


RGY_ERR RGYFilterDenoiseFFT3DBuffer::alloc(int width, int height, RGY_CSP csp, int frames) {
    m_bufFFT.resize(frames);
    for (auto& buf : m_bufFFT) {
        if (!buf || buf->frame.width != width || buf->frame.height != height || buf->frame.csp != csp) {
            buf = m_cl->createFrameBuffer(width, height, csp, RGY_CSP_BIT_DEPTH[csp]);
            if (!buf) {
                return RGY_ERR_NULL_PTR;
            }
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterDenoiseFFT3D::RGYFilterDenoiseFFT3D(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_frameIdx(0),
    m_bufIdx(0),
    m_ov1(0),
    m_ov2(0),
    m_bufFFT(context),
    m_srcBuf(context),
    m_filteredBlocks(),
    m_windowBuf(),
    m_windowBufInverse(),
    m_sigmaBuf(),
    m_wsharpenBuf(),
    m_gridBuf(),
    m_gridDC(0.0f),
    m_noisePowerGain(1.0f),
    m_fft3d() {
    m_name = _T("fft3d");
}

RGYFilterDenoiseFFT3D::~RGYFilterDenoiseFFT3D() {
    close();
}

RGY_ERR RGYFilterDenoiseFFT3D::checkParam(const RGYFilterParamDenoiseFFT3D *prm) {
    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.sigma < 0.0f || 100.0f < prm->fft3d.sigma) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, sigma must be 0 - 100.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (   prm->fft3d.sigma2 < 0.0f || 100.0f < prm->fft3d.sigma2
        || prm->fft3d.sigma3 < 0.0f || 100.0f < prm->fft3d.sigma3
        || prm->fft3d.sigma4 < 0.0f || 100.0f < prm->fft3d.sigma4) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, sigma2/sigma3/sigma4 must be 0 - 100 (0 = follow sigma).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.amount < 0.0f || 1.0f < prm->fft3d.amount) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, strength must be 0 - 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (get_cx_index(list_vpp_fft3d_block_size, prm->fft3d.block_size) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid block_size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.block_size > MAX_BLOCK) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid block_size %d: MAX_BLOCK = %d.\n"), prm->fft3d.block_size, MAX_BLOCK);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.overlap < 0.2f || 0.8f < prm->fft3d.overlap) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, overlap must be 0.2 - 0.8.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.overlap2 < 0.0f || 0.8f < prm->fft3d.overlap2) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, overlap2 must be 0 - 0.8.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (0.8f < prm->fft3d.overlap + prm->fft3d.overlap2) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, sum of overlap and overlap2 must be below 0.8.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.method != 0 && prm->fft3d.method != 1) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, method must be 0 or 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.temporal != 0 && prm->fft3d.temporal != 1) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, temporal must be 0 or 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.bt < -1 || prm->fft3d.bt > 4) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, bt must be 0 (follow temporal), 1 - 4, or -1 (sharpen/degrid only).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.bt == -1 && prm->fft3d.sharpen == 0.0f) {
        // degrid alone with bt=-1 would be an exact no-op (subtract + add back with
        // no filtering in between), so sharpen is required.
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, bt=-1 requires sharpen.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.sharpen < -10.0f || 10.0f < prm->fft3d.sharpen) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, sharpen must be -10 - 10.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.scutoff <= 0.0f || 1.0f < prm->fft3d.scutoff) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, scutoff must be greater than 0, up to 1.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.svr < 0.0f || 10.0f < prm->fft3d.svr) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, svr must be 0 - 10.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.smin < 0.0f || prm->fft3d.smax <= 0.0f || prm->fft3d.smax < prm->fft3d.smin) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, smin/smax must satisfy 0 <= smin <= smax, smax > 0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->fft3d.degrid < 0.0f || 2.0f < prm->fft3d.degrid) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter, degrid must be 0 - 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (get_cx_index(list_vpp_fp_prec, prm->fft3d.precision) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid precision.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseFFT3D::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseFFT3D>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if ((sts = checkParam(prm.get())) != RGY_ERR_NONE) {
        return sts;
    }
    if (prm->fft3d.precision != VppFpPrecision::VPP_FP_PRECISION_FP32 && !RGYOpenCLDevice(m_cl->queue().devid()).checkExtension("cl_khr_fp16")) {
        AddMessage(RGY_LOG_DEBUG, _T("fp16 disabled.\n"));
        prm->fft3d.precision = VppFpPrecision::VPP_FP_PRECISION_FP32;
    }
    if (!m_param
        || prm->fft3d.block_size != std::dynamic_pointer_cast<RGYFilterParamDenoiseFFT3D>(m_param)->fft3d.block_size
        || prm->fft3d.overlap != std::dynamic_pointer_cast<RGYFilterParamDenoiseFFT3D>(m_param)->fft3d.overlap
        || prm->fft3d.overlap2 != std::dynamic_pointer_cast<RGYFilterParamDenoiseFFT3D>(m_param)->fft3d.overlap2
        || prm->fft3d.temporal != std::dynamic_pointer_cast<RGYFilterParamDenoiseFFT3D>(m_param)->fft3d.temporal
        || prm->fft3d.bt != std::dynamic_pointer_cast<RGYFilterParamDenoiseFFT3D>(m_param)->fft3d.bt
        || prm->fft3d.method != std::dynamic_pointer_cast<RGYFilterParamDenoiseFFT3D>(m_param)->fft3d.method
        || (prm->fft3d.sharpen != 0.0f) != (std::dynamic_pointer_cast<RGYFilterParamDenoiseFFT3D>(m_param)->fft3d.sharpen != 0.0f)
        || (prm->fft3d.degrid > 0.0f) != (std::dynamic_pointer_cast<RGYFilterParamDenoiseFFT3D>(m_param)->fft3d.degrid > 0.0f)
        || prm->fft3d.precision != std::dynamic_pointer_cast<RGYFilterParamDenoiseFFT3D>(m_param)->fft3d.precision
        || prm->processChroma != std::dynamic_pointer_cast<RGYFilterParamDenoiseFFT3D>(m_param)->processChroma
        || cmpFrameInfoCspResolution(&m_param->frameOut, &prm->frameOut)) {
        m_ov1 = (int)(prm->fft3d.block_size * 0.5 * prm->fft3d.overlap + 0.5);
        m_ov2 = (int)(prm->fft3d.block_size * 0.5 * (prm->fft3d.overlap + prm->fft3d.overlap2) + 0.5) - m_ov1;
        AddMessage(RGY_LOG_DEBUG, _T("ov1:ov2 = %d:%d.\n"), m_ov1, m_ov2);

        //より小さいUVに合わせてブロック数を計算し、そこから確保するメモリを決める
        auto planeUV = getPlane(&prm->frameOut, RGY_PLANE_U);
        const auto blocksUV = getBlockCount(planeUV.width, planeUV.height, prm->fft3d.block_size, m_ov1, m_ov2);
        const int complexSize = (prm->fft3d.precision == VppFpPrecision::VPP_FP_PRECISION_FP32) ? 8 : 4;

        const auto sub_group_ext_avail = m_cl->platform()->checkSubGroupSupport(m_cl->queue().devid());
        const int fft_barrier_mode = (true) ? (sub_group_ext_avail != RGYOpenCLSubGroupSupport::NONE ? 2 : 1) : 0;
        m_fft3d.set(m_cl->threadPool()->enqueue(
            [cl = m_cl, log = m_pLog, prm = prm, ov1 = m_ov1, ov2 = m_ov2, fft_barrier_mode, sub_group_ext_avail]() {
                // 必要な定数テーブルを作成
                // bitreverse_BLOCK_SIZE
                std::string fft_constants_str = "__constant int bitreverse_BLOCK_SIZE[BLOCK_SIZE] = {\n";
                for (int i = 0; i < prm->fft3d.block_size; i++) {
                    const int value = bitreverse(log2u(prm->fft3d.block_size), i);
                    fft_constants_str += strsprintf("    %d,\n", value);
                }
                fft_constants_str += "};\n\n";
                // FW_BLOCK_K_N
                for (int forward = 0; forward < 2; forward++) {
                    for (int block_size = 2; block_size <= MAX_BLOCK; block_size *= 2) {
                        fft_constants_str += strsprintf("__constant TypeComplex FW_BLOCK_K_%d%s[] = {\n", block_size, forward ? "false" : "true");
                        for (int k = 0; k < block_size / 2; k++) {
                            if (k > 0) {
                                fft_constants_str += ",\n";
                            }
                            const float theta = ((forward) ? -2.0f : +2.0f) * (float)k * FFT_M_PI / (float)block_size;
                            const float cos_theta = std::cos(theta);
                            const float sin_theta = std::sin(theta);
                            fft_constants_str += strsprintf("    (TypeComplex)(%.8ff, %.8ff)", cos_theta, sin_theta);
                        }
                        fft_constants_str += "\n};\n\n";
                    }
                }
                // FW_TEMPORAL1..4 (twiddle tables for the temporal DFT; the kernel
                // selects the one matching temporalCount by token pasting)
                for (int fwTemporal = 1; fwTemporal <= 4; fwTemporal++) {
                    fft_constants_str += strsprintf("__constant TypeComplex FW_TEMPORAL%d[2][%d] = {\n", fwTemporal, (fwTemporal - 1) * (fwTemporal - 1) + 1);
                    for (int forward = 0; forward < 2; forward++) {
                        if (forward > 0) {
                            fft_constants_str += ",\n";
                        }
                        fft_constants_str += "  { ";
                        for (int i = 0; i <= (fwTemporal-1)*(fwTemporal-1); i++) {
                            if (i > 0) {
                                fft_constants_str += ",";
                            }
                            const float theta = ((forward) ? -2.0f : +2.0f) * (float)(i) * FFT_M_PI / (float)(fwTemporal);
                            const float cos_theta = std::cos(theta);
                            const float sin_theta = std::sin(theta);
                            fft_constants_str += strsprintf("(TypeComplex)(%.8ff, %.8ff)", cos_theta, sin_theta);
                        }
                        fft_constants_str += " }";
                    }
                    fft_constants_str += "\n};\n";
                }
                log->write(RGY_LOG_DEBUG, RGY_LOGT_VPP, _T("fft_constants_str.\n%s\n"), char_to_tstring(fft_constants_str).c_str());

                auto gen_options = [&](const int sub_group_size) {
                    const int btFrames = fft3d_bt_frames(prm->fft3d);
                    auto options = strsprintf("-D TypePixel=%s -D bit_depth=%d -D USE_FP16=%d"
                        " -D TypeComplex=%s -D BLOCK_SIZE=%d -D DENOISE_BLOCK_SIZE_X=%d"
                        " -D temporalCurrentIdx=%d -D temporalCount=%d"
                        " -D FFT_BARRIER_MODE=%d -D SUB_GROUP_SIZE=%d -D filterMethod=%d"
                        " -D useSharpen=%d -D useDegrid=%d",
                        RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
                        RGY_CSP_BIT_DEPTH[prm->frameOut.csp],
                        prm->fft3d.precision != VppFpPrecision::VPP_FP_PRECISION_FP32 ? 1 : 0,
                        prm->fft3d.precision != VppFpPrecision::VPP_FP_PRECISION_FP32 ? "half2" : "float2",
                        prm->fft3d.block_size, getDenoiseBlockSizeX(prm->fft3d.block_size),
                        btFrames / 2, btFrames,
                        fft_barrier_mode, sub_group_size,
                        (fft3d_bt(prm->fft3d) < 0) ? -1 : prm->fft3d.method,
                        (prm->fft3d.sharpen != 0.0f) ? 1 : 0,
                        (prm->fft3d.degrid > 0.0f) ? 1 : 0);
                    if (fft_barrier_mode > 0 && sub_group_ext_avail == RGYOpenCLSubGroupSupport::STD20KHR) {
                        options += " -cl-std=CL2.0";
                    }
                    return options;
                };
                auto fft3d_cl = getEmbeddedResourceStr(_T("RGY_FILTER_DENOISE_FFT3D_CL"), _T("EXE_DATA"), cl->getModuleHandle());
                // まず一度、sub_group_size=0でビルドを試み、sub_group_sizeが取得できたら、そのサイズで再ビルドする
                auto fft3d = cl->build(fft_constants_str + fft3d_cl, gen_options(0).c_str());
                if (!fft3d) {
                    log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to load or build RGY_FILTER_DENOISE_FFT3D_CL(fft3d)\n"));
                    return std::unique_ptr<RGYOpenCLProgram>();
                }
                if (sub_group_ext_avail != RGYOpenCLSubGroupSupport::NONE) {
                    auto block_count = getBlockCount(prm->frameOut.width, prm->frameOut.height, prm->fft3d.block_size, ov1, ov2);
                    RGYWorkSize local(prm->fft3d.block_size, getDenoiseBlockSizeX(prm->fft3d.block_size));
                    RGYWorkSize global(divCeil(block_count.first, getDenoiseBlockSizeX(prm->fft3d.block_size)) * local(0), block_count.second * local(1), 1);
                    const auto subGroupSizeFFT = fft3d->kernel("kernel_fft").config(cl->queue(), local, global).subGroupSize();
                    const auto subGroupSizeIFFT = fft3d->kernel("kernel_tfft_filter_ifft").config(cl->queue(), local, global).subGroupSize();
                    bool setSubGroupSize = (subGroupSizeFFT > 0 && subGroupSizeFFT == subGroupSizeIFFT);
                    if ((subGroupSizeFFT & (subGroupSizeFFT - 1)) != 0) {
                        log->write(RGY_LOG_DEBUG, RGY_LOGT_VPP, _T("subGroupSize(%d) is not pow2, sub group opt disabled!\n"), subGroupSizeFFT);
                        setSubGroupSize = false;
                    }
                    if (setSubGroupSize) {
                        log->write(RGY_LOG_DEBUG, RGY_LOGT_VPP, _T("Use sub group opt: subGroupSize=%d.\n"), subGroupSizeFFT);
                        fft3d = cl->build(fft_constants_str + fft3d_cl, gen_options((int)subGroupSizeFFT).c_str());
                        if (!fft3d) {
                            return std::unique_ptr<RGYOpenCLProgram>();
                        }
                    }
                }
                return fft3d;
            }));

        RGY_CSP fft_csp = RGY_CSP_NA;
        int blockGlobalWidth = 0, blockGlobalHeight = 0;
        if (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp] == RGY_CHROMAFMT_YUV420) {
            fft_csp = RGY_CSP_YV12;
            blockGlobalWidth = blocksUV.first * prm->fft3d.block_size * 2;
            blockGlobalHeight = blocksUV.second * prm->fft3d.block_size * 2;
        } else if (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp] == RGY_CHROMAFMT_YUV444) {
            fft_csp = RGY_CSP_YUV444;
            blockGlobalWidth = blocksUV.first * prm->fft3d.block_size;
            blockGlobalHeight = blocksUV.second * prm->fft3d.block_size;
        } else {
            AddMessage(RGY_LOG_ERROR, _T("Invalid colorformat: %s.\n"), RGY_CSP_NAMES[prm->frameOut.csp]);
            return RGY_ERR_UNSUPPORTED;
        }

        if ((sts = m_bufFFT.alloc(blockGlobalWidth * complexSize, blockGlobalHeight * complexSize, fft_csp, fft3d_bt_frames(prm->fft3d))) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for FFT: %s.\n"), get_err_mes(sts));
            return sts;
        }
        if (!prm->processChroma) {
            if ((sts = m_srcBuf.alloc(prm->frameOut.width, prm->frameOut.height, prm->frameOut.csp, fft3d_bt_frames(prm->fft3d))) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for luma-only FFT3D source frames: %s.\n"), get_err_mes(sts));
                return sts;
            }
        } else {
            m_srcBuf.clear();
        }

        m_filteredBlocks = m_cl->createFrameBuffer(blockGlobalWidth, blockGlobalHeight, prm->frameOut.csp, RGY_CSP_BIT_DEPTH[prm->frameOut.csp]);
        if (!m_filteredBlocks) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for filtered blocks: %s.\n"), get_err_mes(sts));
            return RGY_ERR_NULL_PTR;
        }

        sts = AllocFrameBuf(prm->frameOut, 1);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return sts;
        }
        for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
            prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
        }

        if (!m_param || !m_windowBuf || prm->fft3d.block_size != std::dynamic_pointer_cast<RGYFilterParamDenoiseFFT3D>(m_param)->fft3d.block_size) {
            std::vector<float> blockWindow(prm->fft3d.block_size);
            std::vector<float> blockWindowInv(prm->fft3d.block_size);
            auto winFunc = [block_size = prm->fft3d.block_size](const int x) { return 0.50f - 0.50f * std::cos(2.0f * FFT_M_PI * x / (float)block_size); };
            //auto winFunc = [block_size = prm->fft3d.block_size](const int x) { return 0.54f - 0.46f * std::cos(2.0f * FFT_M_PI * x / (float)block_size); };
            for (int i = 0; i < prm->fft3d.block_size; i++) {
                blockWindow[i] = winFunc(i);
                blockWindowInv[i] = 1.0f / blockWindow[i];
            }

            m_windowBuf = m_cl->createBuffer(blockWindow.size() * sizeof(blockWindow[0]));
            m_windowBufInverse = m_cl->createBuffer(blockWindowInv.size() * sizeof(blockWindowInv[0]));

            if (!m_windowBuf) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for FFT window: %s.\n"), get_err_mes(sts));
                return RGY_ERR_NULL_PTR;
            }
            if (!m_windowBufInverse) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for FFT window (inverse): %s.\n"), get_err_mes(sts));
                return RGY_ERR_NULL_PTR;
            }
            //CPUが書き込むためにMapする
            if ((sts = m_windowBuf->queueMapBuffer(m_cl->queue(), CL_MAP_WRITE)) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to map memory for FFT window: %s.\n"), get_err_mes(sts));
                return sts;
            }
            if ((sts = m_windowBufInverse->queueMapBuffer(m_cl->queue(), CL_MAP_WRITE)) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to map memory for FFT window (inv): %s.\n"), get_err_mes(sts));
                return sts;
            }
            if ((sts = m_cl->queue().finish()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to finish queue (map): %s.\n"), get_err_mes(sts));
                return sts;
            }
            memcpy(m_windowBuf->mappedPtr(), blockWindow.data(), blockWindow.size() * sizeof(blockWindow[0]));
            memcpy(m_windowBufInverse->mappedPtr(), blockWindowInv.data(), blockWindowInv.size() * sizeof(blockWindowInv[0]));
            if ((sts = m_windowBuf->unmapBuffer()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to unmap memory for FFT window: %s.\n"), get_err_mes(sts));
                return sts;
            }
            if ((sts = m_windowBufInverse->unmapBuffer()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to unmap memory for FFT window (inv): %s.\n"), get_err_mes(sts));
                return sts;
            }
            if ((sts = m_cl->queue().finish()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to finish queue (unmap): %s.\n"), get_err_mes(sts));
                return sts;
            }
        }
    }

    // Per-frequency-bin host tables, rebuilt every init() (cheap: block_size^2
    // floats) so they always reflect the current parameters even when the
    // (gated) buffer reallocation above is skipped. Clean-room from the
    // documented FFT3DFilter parameter semantics.
    {
        const int bs = prm->fft3d.block_size;
        auto uploadTable = [&](std::unique_ptr<RGYCLBuf>& buf, const std::vector<float>& table, const TCHAR *name) {
            buf = m_cl->createBuffer(table.size() * sizeof(table[0]));
            if (!buf) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for FFT3D %s table.\n"), name);
                return RGY_ERR_NULL_PTR;
            }
            auto err = buf->queueMapBuffer(m_cl->queue(), CL_MAP_WRITE);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to map memory for FFT3D %s table: %s.\n"), name, get_err_mes(err));
                return err;
            }
            if ((err = m_cl->queue().finish()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to finish queue (map %s): %s.\n"), name, get_err_mes(err));
                return err;
            }
            memcpy(buf->mappedPtr(), table.data(), table.size() * sizeof(table[0]));
            if ((err = buf->unmapBuffer()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to unmap memory for FFT3D %s table: %s.\n"), name, get_err_mes(err));
                return err;
            }
            if ((err = m_cl->queue().finish()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to finish queue (unmap %s): %s.\n"), name, get_err_mes(err));
                return err;
            }
            return RGY_ERR_NONE;
        };
        // frequency coordinate of bin i, normalised to [0,1] (0 = DC, 1 = Nyquist),
        // with the negative frequencies at the upper indices mirrored down.
        auto fnorm = [bs](int i) { const int f = (i < bs - i) ? i : (bs - i); return (float)f / (float)(bs / 2); };

        // analysis window (same function the fft kernel uses) and its power sum.
        // For white noise of std s (normalised pixels), each windowed FFT bin has
        // expected power s^2 * sum(w(x)^2) * sum(w(y)^2) - the reference for
        // signorm, which lets the user give sigma as the actual noise level.
        std::vector<float> win(bs);
        auto winFunc = [bs](const int x) { return 0.50f - 0.50f * std::cos(2.0f * FFT_M_PI * x / (float)bs); };
        double sw2 = 0.0;
        for (int i = 0; i < bs; i++) {
            win[i] = winFunc(i);
            sw2 += win[i] * win[i];
        }
        m_noisePowerGain = (float)(sw2 * sw2); // 2D separable window power gain

        // (1) sigma table: the 4 documented anchors (sigma = highest .. sigma4 =
        // lowest frequency) interpolated across the normalised radial frequency.
        {
            const float s1 = prm->fft3d.sigma;                                     // highest freq
            const float s2 = (prm->fft3d.sigma2 > 0.0f) ? prm->fft3d.sigma2 : s1;  // mid-high
            const float s3 = (prm->fft3d.sigma3 > 0.0f) ? prm->fft3d.sigma3 : s1;  // mid-low
            const float s4 = (prm->fft3d.sigma4 > 0.0f) ? prm->fft3d.sigma4 : s1;  // lowest
            const float anchors[4] = { s4, s3, s2, s1 }; // radial 0 -> 1
            std::vector<float> sigmaTable((size_t)bs * bs);
            for (int by = 0; by < bs; by++) {
                const float fy = fnorm(by);
                for (int bx = 0; bx < bs; bx++) {
                    const float fx = fnorm(bx);
                    float radial = std::sqrt(fx * fx + fy * fy) * 0.70710678f; // /sqrt(2) -> [0,1]
                    if (radial > 1.0f) radial = 1.0f;
                    const float t = radial * 3.0f; // 3 linear segments across the 4 anchors
                    int seg = (int)t; if (seg > 2) seg = 2;
                    const float frac = t - (float)seg;
                    const float sval = anchors[seg] * (1.0f - frac) + anchors[seg + 1] * frac;
                    if (prm->fft3d.signorm) {
                        // sigma given as the noise level (8-bit scale): threshold at the
                        // per-bin noise power that this noise level actually produces.
                        // The forward temporal DFT is un-normalized, so iid per-frame
                        // noise power is multiplied by btFrames in the temporal bins
                        // (cf. the original FFT3DFilter's btcur factor) - scale to match.
                        // (smin/smax must NOT get this factor: the sharpen psd is taken
                        // after the 1/N-normalized inverse temporal DFT, per-frame units.)
                        const float snorm = sval * (1.0f / ((1 << 8) - 1));
                        sigmaTable[(size_t)by * bs + bx] = snorm * snorm * m_noisePowerGain * (float)fft3d_bt_frames(prm->fft3d);
                    } else {
                        // backward compatible scale (compared against unnormalised bin power)
                        sigmaTable[(size_t)by * bs + bx] = sval * (1.0f / ((1 << 8) - 1)); // match scalar /255
                    }
                }
            }
            if ((sts = uploadTable(m_sigmaBuf, sigmaTable, _T("sigma"))) != RGY_ERR_NONE) {
                return sts;
            }
        }

        // (2) sharpen weight table: strength x gaussian high-pass frequency
        // weight, 1 - exp(-f^2 / (2*scutoff^2)), with the vertical frequency
        // contribution scaled by svr (svr = 0 -> no vertical sharpening).
        if (prm->fft3d.sharpen != 0.0f) {
            const float scutoff = std::max(prm->fft3d.scutoff, 0.01f);
            std::vector<float> wsharpenTable((size_t)bs * bs);
            for (int by = 0; by < bs; by++) {
                const float fy = fnorm(by) * prm->fft3d.svr;
                for (int bx = 0; bx < bs; bx++) {
                    const float fx = fnorm(bx);
                    const float f2 = fx * fx + fy * fy;
                    const float weight = 1.0f - std::exp(-f2 / (2.0f * scutoff * scutoff));
                    wsharpenTable[(size_t)by * bs + bx] = prm->fft3d.sharpen * weight;
                }
            }
            if ((sts = uploadTable(m_wsharpenBuf, wsharpenTable, _T("sharpen"))) != RGY_ERR_NONE) {
                return sts;
            }
        } else {
            m_wsharpenBuf.reset();
        }

        // (3) gridsample spectrum for degrid: the 2D spectrum of the analysis
        // window itself (the spectrum a flat, featureless block produces). It is
        // separable, so it is built from the 1D DFT of the window function.
        // The kernel scales it by each block's DC / gridDC to reconstruct and
        // subtract the window bias before filtering.
        if (prm->fft3d.degrid > 0.0f) {
            std::vector<std::pair<float, float>> w1(bs); // 1D DFT of the window
            for (int k = 0; k < bs; k++) {
                double re = 0.0, im = 0.0;
                for (int x = 0; x < bs; x++) {
                    const double theta = -2.0 * FFT_M_PI * k * x / (double)bs;
                    re += win[x] * std::cos(theta);
                    im += win[x] * std::sin(theta);
                }
                w1[k] = { (float)re, (float)im };
            }
            std::vector<float> gridTable((size_t)bs * bs * 2);
            for (int by = 0; by < bs; by++) {
                for (int bx = 0; bx < bs; bx++) {
                    // complex product W1[by] * W1[bx]
                    const float re = w1[by].first * w1[bx].first - w1[by].second * w1[bx].second;
                    const float im = w1[by].first * w1[bx].second + w1[by].second * w1[bx].first;
                    gridTable[((size_t)by * bs + bx) * 2 + 0] = re;
                    gridTable[((size_t)by * bs + bx) * 2 + 1] = im;
                }
            }
            m_gridDC = w1[0].first * w1[0].first; // (sum of window)^2, DC of the 2D spectrum
            if ((sts = uploadTable(m_gridBuf, gridTable, _T("gridsample"))) != RGY_ERR_NONE) {
                return sts;
            }
        } else {
            m_gridBuf.reset();
            m_gridDC = 0.0f;
        }
    }

    setFilterInfo(pParam->print());
    m_pathThrough = FILTER_PATHTHROUGH_ALL;
    if (fft3d_bt(prm->fft3d) > 1) {
        m_pathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_DATA));
    }
    m_param = pParam;
    return sts;
}

tstring RGYFilterParamDenoiseFFT3D::print() const {
    return fft3d.print() + strsprintf(_T(", chroma %s"), processChroma ? _T("on") : _T("off"));
}

RGY_ERR RGYFilterDenoiseFFT3D::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;

    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseFFT3D>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const int bt = fft3d_bt(prm->fft3d);
    const bool finalOutput = pInputFrame->ptr[0] == nullptr;
    if (finalOutput) {
        if (bt <= 1 || m_frameIdx >= m_bufIdx) {
            //終了
            *pOutputFrameNum = 0;
            ppOutputFrames[0] = nullptr;
            return sts;
        }
        for (auto w : wait_events) {
            m_cl->queue().wait(w);
        }
    } else {
        //if (interlaced(*pInputFrame)) {
        //    return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
        //}
        if (m_param->frameOut.csp != m_param->frameIn.csp) {
            AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        const int curBufIdx = m_bufIdx++;
        auto fftBuf = m_bufFFT.get(curBufIdx);
        if (!fftBuf || !fftBuf->frame.ptr[0]) {
            AddMessage(RGY_LOG_ERROR, _T("failed to get fft buffer.\n"));
            return RGY_ERR_NULL_PTR;
        }
        if (!prm->processChroma) {
            auto srcBuf = m_srcBuf.get(curBufIdx);
            if (!srcBuf || !srcBuf->frame.ptr[0]) {
                AddMessage(RGY_LOG_ERROR, _T("failed to get luma-only FFT3D source buffer.\n"));
                return RGY_ERR_NULL_PTR;
            }
            auto copyErr = m_cl->copyFrame(&srcBuf->frame, pInputFrame, nullptr, queue, wait_events, nullptr);
            if (copyErr != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy luma-only FFT3D source frame: %s.\n"), get_err_mes(copyErr));
                return copyErr;
            }
            copyFramePropWithoutRes(&srcBuf->frame, pInputFrame);
        }
        sts = denoiseFFT(&fftBuf->frame, pInputFrame, queue, wait_events);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run fft: %s.\n"), get_err_mes(sts));
            return RGY_ERR_NONE;
        }
        copyFramePropWithoutRes(&fftBuf->frame, pInputFrame);
    }

    auto planeUV = getPlane(&prm->frameOut, RGY_PLANE_U);

    const RGYFrameInfo *srcCurFrame = nullptr;
    {
        // bt frame layout (see fft3d_bt): [prev.. , cur, ..next], cur at index nPast.
        const int btFrames = std::max(bt, 1);      // bt=-1 processes a single frame
        const int nPast = btFrames / 2;
        const int nFuture = (btFrames - 1) - nPast; // bt1:0 bt2:0 bt3:1 bt4:1
        const int curIdx = nPast;                   // temporalCurrentIdx (baked into the program)

        // Not enough future frames buffered yet to emit the next output frame.
        // (nFuture==0 for bt=1/2 -> emit immediately; nFuture==1 for bt=3/4 ->
        //  one frame of look-ahead, flushed at finalOutput.)
        if (!finalOutput && m_bufIdx < m_frameIdx + nFuture + 1) {
            *pOutputFrameNum = 0;
            ppOutputFrames[0] = nullptr;
            return sts;
        }

        const int outFrameIdx = m_frameIdx; // frame to output now
        // Gather the btFrames frames [outFrameIdx-nPast .. outFrameIdx+nFuture],
        // repeating boundary frames at the start of stream and during flush by
        // clamping to the valid buffered range (reproduces the previous
        // prev=cur / next=cur edge handling for bt=3).
        RGYCLFrame *frames[4] = { nullptr, nullptr, nullptr, nullptr };
        for (int k = 0; k < btFrames; k++) {
            int idx = outFrameIdx + (k - nPast);
            if (idx < 0) idx = 0;
            if (idx > m_bufIdx - 1) idx = m_bufIdx - 1;
            frames[k] = m_bufFFT.get(idx);
        }
        auto fftCur = frames[curIdx];
        if (!prm->processChroma) {
            auto srcCur = m_srcBuf.get(outFrameIdx);
            srcCurFrame = srcCur ? &srcCur->frame : nullptr;
        }
        sts = denoiseTFFTFilterIFFT(&m_filteredBlocks->frame,
            frames[0] ? &frames[0]->frame : nullptr,
            frames[1] ? &frames[1]->frame : nullptr,
            frames[2] ? &frames[2]->frame : nullptr,
            frames[3] ? &frames[3]->frame : nullptr,
            queue);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run tfft_filter_ifft(%d, %d): %s.\n"), curIdx, btFrames, get_err_mes(sts));
            return RGY_ERR_NONE;
        }
        if (btFrames > 1) {
            copyFramePropWithoutRes(ppOutputFrames[0], &fftCur->frame);
        }
    }
    std::vector<RGYOpenCLEvent> mergeWaitEvents;
    if (!prm->processChroma) {
        if (!srcCurFrame || !srcCurFrame->ptr[0]) {
            AddMessage(RGY_LOG_ERROR, _T("missing luma-only FFT3D source frame.\n"));
            return RGY_ERR_INVALID_CALL;
        }
        RGYOpenCLEvent copyEvent;
        auto copyErr = m_cl->copyFrame(ppOutputFrames[0], srcCurFrame, nullptr, queue, {}, &copyEvent);
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy luma-only FFT3D output base frame: %s.\n"), get_err_mes(copyErr));
            return copyErr;
        }
        if (copyEvent() != nullptr) {
            mergeWaitEvents.push_back(copyEvent);
        }
    }
    sts = denoiseMerge(ppOutputFrames[0], &m_filteredBlocks->frame, queue, mergeWaitEvents, event);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to run merge: %s.\n"), get_err_mes(sts));
        return RGY_ERR_NONE;
    }

    m_frameIdx++;
    return sts;
}

void RGYFilterDenoiseFFT3D::close() {
    m_frameBuf.clear();
    m_bufFFT.clear();
    m_srcBuf.clear();
    m_windowBuf.reset();
    m_windowBufInverse.reset();
    m_sigmaBuf.reset();
    m_wsharpenBuf.reset();
    m_gridBuf.reset();
}
