// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2026 rigaya
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

#include "rgy_filter_clahe.h"

RGY_ERR RGYFilterClahe::procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamClahe>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int planeW = pOutputPlane->width;
    const int planeH = pOutputPlane->height;
    const int tilesX = prm->clahe.tiles_x;
    const int tilesY = prm->clahe.tiles_y;

    // Pass 1: 複製したSLMヒストグラムでatomic競合を抑える。
    {
        const char *kernel_name = "kernel_clahe_hist";
        RGYWorkSize local(32, 8);
        RGYWorkSize global((size_t)tilesX * 32, (size_t)tilesY * 8);
        auto err = m_clahe.get()->kernel(kernel_name).config(queue, local, global, wait_events, nullptr).launch(
            m_histBuf->mem(),
            (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
            planeW, planeH,
            tilesX, tilesY);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }

    // Pass 2: per-tile contrast-limit + CDF -> transform table.
    {
        const char *kernel_name = "kernel_clahe_cdf";
        RGYWorkSize local(1, 1);
        RGYWorkSize global(tilesX, tilesY);
        auto err = m_clahe.get()->kernel(kernel_name).config(queue, local, global, {}, nullptr).launch(
            m_transformBuf->mem(),
            m_histBuf->mem(),
            planeW, planeH,
            tilesX, tilesY,
            prm->clahe.slope);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }

    // Pass 3: bilinear interpolated transform apply.
    {
        const char *kernel_name = "kernel_clahe_apply";
        RGYWorkSize local(32, 8);
        RGYWorkSize global(ALIGN(planeW, 32), ALIGN(planeH, 8));
        auto err = m_clahe.get()->kernel(kernel_name).config(queue, local, global, {}, event).launch(
            (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
            (cl_mem)pInputPlane->ptr[0],  pInputPlane->pitch[0],
            planeW, planeH,
            m_transformBuf->mem(),
            tilesX, tilesY);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterClahe::RGYFilterClahe(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context), m_clahe(), m_histBuf(), m_transformBuf(), m_tilesX(0), m_tilesY(0), m_binBitdepth(0) {
    m_name = _T("clahe");
}

RGYFilterClahe::~RGYFilterClahe() {
    close();
}

RGY_ERR RGYFilterClahe::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamClahe>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->clahe.tiles_x < 2 || 32 < prm->clahe.tiles_x) {
        prm->clahe.tiles_x = clamp(prm->clahe.tiles_x, 2, 32);
        AddMessage(RGY_LOG_WARN, _T("tiles_x should be in range of %d - %d.\n"), 2, 32);
    }
    if (prm->clahe.tiles_y < 2 || 32 < prm->clahe.tiles_y) {
        prm->clahe.tiles_y = clamp(prm->clahe.tiles_y, 2, 32);
        AddMessage(RGY_LOG_WARN, _T("tiles_y should be in range of %d - %d.\n"), 2, 32);
    }
    if (prm->clahe.slope < 1.0f || 40.0f < prm->clahe.slope) {
        prm->clahe.slope = clamp(prm->clahe.slope, 1.0f, 40.0f);
        AddMessage(RGY_LOG_WARN, _T("slope should be in range of %.1f - %.1f.\n"), 1.0f, 40.0f);
    }

    const int storageBitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int histBitdepth = prm->histBitdepth;
    if (histBitdepth < 8 || storageBitdepth < histBitdepth) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported bit depth combination: histogram %d-bit, storage %d-bit.\n"),
            histBitdepth, storageBitdepth);
        return RGY_ERR_UNSUPPORTED;
    }
    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamClahe>(m_param);
    if (!m_clahe.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != storageBitdepth
        || prmPrev->histBitdepth != histBitdepth) {
        const auto options = strsprintf("-D Type=%s -D hist_bit_depth=%d -D storage_bit_depth=%d",
            storageBitdepth > 8 ? "ushort" : "uchar",
            histBitdepth, storageBitdepth);
        m_clahe.set(m_cl->buildResourceAsync(_T("RGY_FILTER_CLAHE_CL"), _T("EXE_DATA"), options.c_str()));
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // タイル数と実効bitdepthに応じたヒストグラム・変換表を確保する。
    const int tilesX = prm->clahe.tiles_x;
    const int tilesY = prm->clahe.tiles_y;
    const int binBitdepth = std::min(histBitdepth, 10);
    const int bins = 1 << binBitdepth;
    if (!m_histBuf || tilesX > m_tilesX || tilesY > m_tilesY || binBitdepth > m_binBitdepth) {
        const size_t histBytes      = (size_t)tilesX * (size_t)tilesY * bins * sizeof(uint32_t);
        const size_t transformBytes = (size_t)tilesX * (size_t)tilesY * bins * sizeof(uint16_t);
        m_histBuf      = m_cl->createBuffer(histBytes,      CL_MEM_READ_WRITE);
        m_transformBuf = m_cl->createBuffer(transformBytes, CL_MEM_READ_WRITE);
        if (!m_histBuf || !m_transformBuf) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate hist / transform buffers.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_tilesX = tilesX;
        m_tilesY = tilesY;
        m_binBitdepth = binBitdepth;
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterClahe::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
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
    if (!m_clahe.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build RGY_FILTER_CLAHE_CL(m_clahe)\n"));
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

    // CLAHE: Y-plane only by design (luminance contrast enhancement).
    // Chroma planes pass through unmodified.
    const int numPlanes = RGY_CSP_PLANES[ppOutputFrames[0]->csp];
    for (int i = 0; i < numPlanes; i++) {
        auto planeDst = getPlane(ppOutputFrames[0], (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame,       (RGY_PLANE)i);
        if (i == 0) {
            sts = procPlane(&planeDst, &planeSrc, queue, wait_events, (i == numPlanes - 1) ? event : nullptr);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at claheFrame (%s): %s.\n"),
                    RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
                return sts;
            }
        } else {
            sts = m_cl->copyPlane(&planeDst, &planeSrc, nullptr, queue, {}, (i == numPlanes - 1) ? event : nullptr);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
    }
    return sts;
}

void RGYFilterClahe::close() {
    m_frameBuf.clear();
    m_histBuf.reset();
    m_transformBuf.reset();
    m_tilesX = 0;
    m_tilesY = 0;
    m_binBitdepth = 0;
    m_clahe.clear();
    m_cl.reset();
}
