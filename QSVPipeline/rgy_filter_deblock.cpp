// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
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

#include <algorithm>
#include "convert_csp.h"
#include "rgy_filter_deblock.h"

static const int DEBLOCK_BLOCK_X = 32;
static const int DEBLOCK_BLOCK_Y = 8;

// =============================================================================
// ITU-T Rec. H.264 (V14, 2022-08) Table 8-16 and Table 8-17.
// Values are factual numeric constants from the published international
// standard, indexed by the [0, 51] quantisation parameter range defined
// in the spec. Lookup indices outside [0, 51] are clamped by the host
// before access; the standard does not define values above 51.
// =============================================================================

// Table 8-16, alpha[indexA] -- outer threshold (|p0 - q0| < alpha).
static const int H264_ALPHA[52] = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      4,   4,   5,   6,   7,   8,   9,  10,  12,  13,  15,  17,  20,  22,  25,  28,
     32,  36,  40,  45,  50,  56,  63,  71,  80,  90, 101, 113, 127, 144, 162, 182,
    203, 226, 255, 255
};

// Table 8-16, beta[indexB] -- inner threshold (|p1 - p0| < beta etc.).
static const int H264_BETA[52] = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      2,   2,   2,   3,   3,   3,   3,   4,   4,   4,   6,   6,   7,   7,   8,   8,
      9,   9,  10,  10,  11,  11,  12,  12,  13,  13,  14,  14,  15,  15,  16,  16,
     17,  17,  18,  18
};

// Table 8-17, column bS=1 -- non-strong filter clip value tc0[indexA].
// (Columns for bS=2 and bS=3 are not used; this filter implements the
// bS=1 path only, the gentlest non-strong setting suited to spatial
// post-processing of an already-decoded image.)
static const int H264_TC0_BS1[52] = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   2,
      2,   2,   3,   3
};

RGYFilterDeblock::RGYFilterDeblock(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_deblock(),
    m_buildOptions() {
    m_name = _T("deblock");
}

RGYFilterDeblock::~RGYFilterDeblock() {
    close();
}

RGY_ERR RGYFilterDeblock::checkParam(const std::shared_ptr<RGYFilterParamDeblock> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->deblock.qp < 0 || 51 < prm->deblock.qp) {
        prm->deblock.qp = clamp(prm->deblock.qp, 0, 51);
        AddMessage(RGY_LOG_WARN, _T("qp should be in range of [0, 51] (ITU-T Rec. H.264 §8.7); clamped.\n"));
    }
    if (prm->deblock.alpha < -6 || 6 < prm->deblock.alpha) {
        prm->deblock.alpha = clamp(prm->deblock.alpha, -6, 6);
        AddMessage(RGY_LOG_WARN, _T("alpha offset should be in range of [-6, 6]; clamped.\n"));
    }
    if (prm->deblock.beta < -6 || 6 < prm->deblock.beta) {
        prm->deblock.beta = clamp(prm->deblock.beta, -6, 6);
        AddMessage(RGY_LOG_WARN, _T("beta offset should be in range of [-6, 6]; clamped.\n"));
    }
    const auto chromaFormat = RGY_CSP_CHROMA_FORMAT[prm->frameIn.csp];
    if (rgy_chromafmt_is_rgb(chromaFormat)
        || (RGY_CSP_PLANES[prm->frameIn.csp] == 1 && chromaFormat != RGY_CHROMAFMT_MONOCHROME)) {
        AddMessage(RGY_LOG_ERROR, _T("deblock supports planar/semi-planar YUV or monochrome formats only: %s.\n"),
            RGY_CSP_NAMES[prm->frameIn.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->deblock.chroma && RGY_CSP_PLANES[prm->frameIn.csp] < 3) {
        prm->deblock.chroma = false;
        AddMessage(RGY_LOG_WARN, _T("deblock chroma processing requires planar chroma; disabled for %s.\n"),
            RGY_CSP_NAMES[prm->frameIn.csp]);
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDeblock::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDeblock>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    prm->frameOut.picstruct = prm->frameIn.picstruct;

    const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    const int maxVal   = (1 << bitDepth) - 1;

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamDeblock>(m_param);
    if (!m_deblock.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        m_buildOptions = strsprintf("-D Type=%s -D bit_depth=%d -D max_val=%d",
            bitDepth > 8 ? "ushort" : "uchar",
            bitDepth, maxVal);
        AddMessage(RGY_LOG_DEBUG, _T("Starting async build for RGY_FILTER_DEBLOCK_CL: %s\n"),
            char_to_tstring(m_buildOptions).c_str());
        m_deblock.set(m_cl->buildResourceAsync(_T("RGY_FILTER_DEBLOCK_CL"), _T("EXE_DATA"), m_buildOptions.c_str()));
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output buffer: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterDeblock::runPassVertical(RGYFrameInfo *pDstPlane,
                                           int alpha, int beta, int tc0, int is_chroma,
                                           RGYOpenCLQueue &queue,
                                           const std::vector<RGYOpenCLEvent> &wait_events,
                                           RGYOpenCLEvent *event) {
    const char *kernel_name = "deblock_vertical";
    // One thread per (edge, row). Edges are at columns 4, 8, ..., (W/4 - 1)*4.
    const int num_edges = (pDstPlane->width / 4) - 1;
    if (num_edges <= 0) return RGY_ERR_NONE;
    // Vertical-edge pass has a tall narrow iteration space
    // (num_edges = W/4 - 1 wide, full height tall), so a Y-major
    // workgroup (8 wide, 32 tall) fits the grid shape better than the
    // X-major (32, 8) used for the horizontal-edge pass.
    RGYWorkSize local(8, 32);
    RGYWorkSize global(num_edges, pDstPlane->height);
    auto err = m_deblock.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pDstPlane->ptr[0], pDstPlane->pitch[0],
        pDstPlane->width, pDstPlane->height,
        alpha, beta, tc0, is_chroma);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"),
            char_to_tstring(kernel_name).c_str(), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDeblock::runPassHorizontal(RGYFrameInfo *pDstPlane,
                                             int alpha, int beta, int tc0, int is_chroma,
                                             RGYOpenCLQueue &queue,
                                             const std::vector<RGYOpenCLEvent> &wait_events,
                                             RGYOpenCLEvent *event) {
    const char *kernel_name = "deblock_horizontal";
    // One thread per (column, edge). Edges at rows 4, 8, ..., (H/4 - 1)*4.
    const int num_edges = (pDstPlane->height / 4) - 1;
    if (num_edges <= 0) return RGY_ERR_NONE;
    RGYWorkSize local(DEBLOCK_BLOCK_X, DEBLOCK_BLOCK_Y);
    RGYWorkSize global(pDstPlane->width, num_edges);
    auto err = m_deblock.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pDstPlane->ptr[0], pDstPlane->pitch[0],
        pDstPlane->width, pDstPlane->height,
        alpha, beta, tc0, is_chroma);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"),
            char_to_tstring(kernel_name).c_str(), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDeblock::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue_main,
    const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    *pOutputFrameNum  = 0;
    ppOutputFrames[0] = nullptr;
    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamDeblock>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!m_deblock.get()) {
        AddMessage(RGY_LOG_ERROR, _T("deblock OpenCL program failed to build (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }
    const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, m_frameBuf[0]->frame.mem_type);
    if (memcpyKind != RGYCLMemcpyD2D) {
        AddMessage(RGY_LOG_ERROR, _T("deblock only supports device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("deblock does not support csp conversion.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    // Resolve effective table lookup indices.
    //   indexA = clamp(qp + alpha_offset, 0, 51) -> alpha and tc0
    //   indexB = clamp(qp + beta_offset , 0, 51) -> beta
    // The standard tables stop at 51; clamping back to 51 saturates at
    // the strongest defined setting. The host then scales each table
    // entry by 1 << (bit_depth - 8) so the kernel works in the source's
    // pixel domain.
    const int indexA = std::min(51, std::max(0, prm->deblock.qp + prm->deblock.alpha));
    const int indexB = std::min(51, std::max(0, prm->deblock.qp + prm->deblock.beta));
    const int bitDepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
    const int bdShift  = std::max(0, bitDepth - 8);
    const int alpha_scaled = H264_ALPHA  [indexA] << bdShift;
    const int beta_scaled  = H264_BETA   [indexB] << bdShift;
    const int tc0_scaled   = H264_TC0_BS1[indexA] << bdShift;

    RGYFrameInfo *pOut = &m_frameBuf[0]->frame;
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];

    // 同一queue_main上のin-order実行に依存し、stage間/plane間のevent連鎖は省略する。
    // 最後にenqueueされる作業にだけeventを紐付け、呼び出し元へ返す。
    const int planeMax = prm->deblock.chroma ? planes : 1;
    int lastDeblockPlane = -1;
    for (int i = 0; i < planeMax; i++) {
        const auto planeDst = getPlane(pOut, (RGY_PLANE)i);
        if (planeDst.width >= 8 && planeDst.height >= 8) lastDeblockPlane = i;
    }

    // Stage 1: copy every plane src -> dst. The kernels then modify
    // edge pixels in place. Planes we choose not to deblock (chroma when
    // chroma=false) end up identical to source, which is exactly what
    // the copy already gave us.
    for (int i = 0; i < planes; i++) {
        const auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        auto       planeDst = getPlane(pOut, (RGY_PLANE)i);
        const std::vector<RGYOpenCLEvent> &copyWait = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        // deblock kernelが一切走らないケースでは、最後のcopyPlaneにeventを紐付ける。
        RGYOpenCLEvent *copyEvent = (lastDeblockPlane < 0 && i == planes - 1) ? event : nullptr;
        auto err = m_cl->copyPlane(&planeDst, &planeSrc, nullptr, queue_main, copyWait, copyEvent);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("deblock: src->dst copyPlane (plane %d) failed: %s.\n"),
                i, get_err_mes(err));
            return err;
        }
    }

    // Stage 2 and 3: pass 1 (vertical edges) then pass 2 (horizontal
    // edges), in-place on dst. Luma always; chroma only when enabled.
    for (int i = 0; i < planeMax; i++) {
        auto planeDst = getPlane(pOut, (RGY_PLANE)i);
        const int is_chroma = (i == 0) ? 0 : 1;
        // Need at least 8 pixels on each axis to have an interior 4x4
        // block boundary that the filter can safely sample (read range
        // is +/- 3 of the boundary).
        if (planeDst.width < 8 || planeDst.height < 8) continue;

        auto err = runPassVertical(&planeDst,
                                    alpha_scaled, beta_scaled, tc0_scaled, is_chroma,
                                    queue_main, {}, nullptr);
        if (err != RGY_ERR_NONE) return err;
        RGYOpenCLEvent *passEvent = (i == lastDeblockPlane) ? event : nullptr;
        err = runPassHorizontal(&planeDst,
                                 alpha_scaled, beta_scaled, tc0_scaled, is_chroma,
                                 queue_main, {}, passEvent);
        if (err != RGY_ERR_NONE) return err;
    }

    pOut->timestamp    = pInputFrame->timestamp;
    pOut->duration     = pInputFrame->duration;
    pOut->inputFrameId = pInputFrame->inputFrameId;
    pOut->picstruct    = pInputFrame->picstruct;
    pOut->flags        = pInputFrame->flags;
    ppOutputFrames[0]  = pOut;
    *pOutputFrameNum   = 1;
    return RGY_ERR_NONE;
}

void RGYFilterDeblock::close() {
    m_deblock.clear();
    m_buildOptions.clear();
    m_frameBuf.clear();
    m_cl.reset();
}
