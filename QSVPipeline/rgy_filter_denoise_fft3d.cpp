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
    const float sigma = prm->fft3d.sigma / (float)((1 << 8) - 1); // sigmaは8bit基準なので 0 - 1 に変換する
    const float limit = 1.0f - prm->fft3d.amount;
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
            sigma, limit
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
            sigma, limit
        );
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseFFT3D::denoiseMerge(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
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
        auto err = m_fft3d.get()->kernel(kernel_name).config(queue, local, global).launch(
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
    m_filteredBlocks(),
    m_windowBuf(),
    m_windowBufInverse(),
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
        || prm->fft3d.precision != std::dynamic_pointer_cast<RGYFilterParamDenoiseFFT3D>(m_param)->fft3d.precision
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
        m_fft3d.set(std::async(std::launch::async,
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
                // FW_TEMPORAL3
                fft_constants_str += "__constant TypeComplex FW_TEMPORAL3[2][5] = {\n";
                for (int forward = 0; forward < 2; forward++) {
                    if (forward > 0) {
                        fft_constants_str += ",\n";
                    }
                    fft_constants_str += "  { ";
                    const int fwTemporal = 3;
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
                log->write(RGY_LOG_DEBUG, RGY_LOGT_VPP, _T("fft_constants_str.\n%s\n"), char_to_tstring(fft_constants_str).c_str());

                auto gen_options = [&](const int sub_group_size) {
                    auto options = strsprintf("-D TypePixel=%s -D bit_depth=%d -D USE_FP16=%d"
                        " -D TypeComplex=%s -D BLOCK_SIZE=%d -D DENOISE_BLOCK_SIZE_X=%d"
                        " -D temporalCurrentIdx=%d -D temporalCount=%d"
                        " -D FFT_BARRIER_MODE=%d -D SUB_GROUP_SIZE=%d -D filterMethod=%d",
                        RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
                        RGY_CSP_BIT_DEPTH[prm->frameOut.csp],
                        prm->fft3d.precision != VppFpPrecision::VPP_FP_PRECISION_FP32 ? 1 : 0,
                        prm->fft3d.precision != VppFpPrecision::VPP_FP_PRECISION_FP32 ? "half2" : "float2",
                        prm->fft3d.block_size, getDenoiseBlockSizeX(prm->fft3d.block_size),
                        prm->fft3d.temporal ? 1 : 0, prm->fft3d.temporal ? 3 : 1,
                        fft_barrier_mode, sub_group_size,
                        prm->fft3d.method);
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

        if ((sts = m_bufFFT.alloc(blockGlobalWidth * complexSize, blockGlobalHeight * complexSize, fft_csp, prm->fft3d.temporal ? 3 : 1)) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for FFT: %s.\n"), get_err_mes(sts));
            return sts;
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

    setFilterInfo(pParam->print());
    m_pathThrough = FILTER_PATHTHROUGH_ALL;
    if (prm->fft3d.temporal) {
        m_pathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_DATA));
    }
    m_param = pParam;
    return sts;
}

tstring RGYFilterParamDenoiseFFT3D::print() const {
    return fft3d.print();
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

    const bool finalOutput = pInputFrame->ptr[0] == nullptr;
    if (finalOutput) {
        if (!prm->fft3d.temporal || m_frameIdx >= m_bufIdx) {
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
        auto fftBuf = m_bufFFT.get(m_bufIdx++);
        if (!fftBuf || !fftBuf->frame.ptr[0]) {
            AddMessage(RGY_LOG_ERROR, _T("failed to get fft buffer.\n"));
            return RGY_ERR_NULL_PTR;
        }
        sts = denoiseFFT(&fftBuf->frame, pInputFrame, queue, wait_events);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run fft: %s.\n"), get_err_mes(sts));
            return RGY_ERR_NONE;
        }
        copyFramePropWithoutRes(&fftBuf->frame, pInputFrame);
    }

    auto planeUV = getPlane(&prm->frameOut, RGY_PLANE_U);

    if (prm->fft3d.temporal) {
        if (m_bufIdx <= 1) {
            //出力フレームなし
            *pOutputFrameNum = 0;
            ppOutputFrames[0] = nullptr;
            return sts;
        }
        auto fftPrev = m_bufFFT.get(std::max(m_bufIdx - ((finalOutput) ? 2 : 3), 0));
        auto fftCur = m_bufFFT.get(m_bufIdx - ((finalOutput) ? 1 : 2));
        auto fftNext = m_bufFFT.get(m_bufIdx - 1);
        sts = denoiseTFFTFilterIFFT(&m_filteredBlocks->frame, &fftPrev->frame, &fftCur->frame, &fftNext->frame, nullptr, queue);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run tfft_filter_ifft(1, 3): %s.\n"), get_err_mes(sts));
            return RGY_ERR_NONE;
        }
        copyFramePropWithoutRes(ppOutputFrames[0], &fftCur->frame);
    } else {
        auto fftCur = m_bufFFT.get(m_bufIdx - 1);
        sts = denoiseTFFTFilterIFFT(&m_filteredBlocks->frame, &fftCur->frame, nullptr, nullptr, nullptr, queue);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to run tfft_filter_ifft(0, 1): %s.\n"), get_err_mes(sts));
            return RGY_ERR_NONE;
        }
    }
    sts = denoiseMerge(ppOutputFrames[0], &m_filteredBlocks->frame, queue, event);
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
    m_windowBuf.reset();
    m_windowBufInverse.reset();
}
