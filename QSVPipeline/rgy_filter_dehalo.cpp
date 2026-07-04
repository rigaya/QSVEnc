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
#include <cmath>
#include <utility>
#include "convert_csp.h"
#include "rgy_filter_dehalo.h"
#include "rgy_filter_resize.h"

static const int DEHALO_BLOCK_X = 32;
static const int DEHALO_BLOCK_Y = 8;

static int dehalo_alpha_auto_search_radius(const VppDehalo& prm) {
    return std::max((int)std::lround(std::max(prm.rx, prm.ry)), 3);
}

static std::pair<int, int> dehalo_alpha_search_radius(const VppDehalo& prm) {
    int searchRade = prm.searchRade;
    int searchRadi = prm.searchRadi;
    if (searchRade < 0) {
        searchRade = dehalo_alpha_auto_search_radius(prm);
    }
    if (searchRadi < 0) {
        searchRadi = searchRade;
    }
    return std::make_pair(searchRade, searchRadi);
}

RGYFilterDehalo::RGYFilterDehalo(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_dehalo(),
    m_buildOptions(),
    m_resizeUp(),
    m_resizeDown(),
    m_resizeAlphaHaloDown(),
    m_resizeAlphaHaloUp(),
    m_resizeAlphaUp(),
    m_resizeAlphaDown(),
    m_supersampled(),
    m_expanded(),
    m_inpand(),
    m_mask(),
    m_corrected(),
    m_alphaHalosSmall(),
    m_alphaHalos(),
    m_alphaAre(),
    m_alphaUgly(),
    m_alphaLets(),
    m_alphaLimitLow(),
    m_alphaLimitHigh(),
    m_alphaLimitLowSS(),
    m_alphaLimitHighSS(),
    m_alphaRemoved(),
    m_ssW(0),
    m_ssH(0),
    m_alphaHaloW(0),
    m_alphaHaloH(0),
    m_ssActive(false) {
    m_name = _T("dehalo");
}

RGYFilterDehalo::~RGYFilterDehalo() {
    close();
}

RGY_ERR RGYFilterDehalo::checkParam(const std::shared_ptr<RGYFilterParamDehalo> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height < 4 || prm->frameOut.width < 4) {
        AddMessage(RGY_LOG_ERROR, _T("dehalo requires input width/height >= 4 (got %dx%d).\n"),
            prm->frameOut.width, prm->frameOut.height);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.rx >= 0.5f && prm->dehalo.rx <= 10.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid rx=%.2f: must be in [0.5, 10.0].\n"), prm->dehalo.rx);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.ry >= 0.5f && prm->dehalo.ry <= 10.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid ry=%.2f: must be in [0.5, 10.0].\n"), prm->dehalo.ry);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.darkstr >= 0.0f && prm->dehalo.darkstr <= 1.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid darkstr=%.2f: must be in [0.0, 1.0].\n"), prm->dehalo.darkstr);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.brightstr >= 0.0f && prm->dehalo.brightstr <= 1.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid brightstr=%.2f: must be in [0.0, 1.0].\n"), prm->dehalo.brightstr);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dehalo.lowsens < 0 || prm->dehalo.lowsens > 100) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid lowsens=%d: must be in [0, 100].\n"), prm->dehalo.lowsens);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dehalo.highsens < 0 || prm->dehalo.highsens > 100) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid highsens=%d: must be in [0, 100].\n"), prm->dehalo.highsens);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!(prm->dehalo.ss >= 1.0f && prm->dehalo.ss <= 4.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid ss=%.2f: must be in [1.0, 4.0].\n"), prm->dehalo.ss);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!((prm->dehalo.searchRade == FILTER_DEFAULT_DEHALO_SEARCH_RADIUS_AUTO) || (prm->dehalo.searchRade >= 1 && prm->dehalo.searchRade <= 10))) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid search_rade=%d: must be auto or in [1, 10].\n"), prm->dehalo.searchRade);
        return RGY_ERR_INVALID_PARAM;
    }
    if (!((prm->dehalo.searchRadi == FILTER_DEFAULT_DEHALO_SEARCH_RADIUS_AUTO) || (prm->dehalo.searchRadi >= 1 && prm->dehalo.searchRadi <= 10))) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid search_radi=%d: must be auto or in [1, 10].\n"), prm->dehalo.searchRadi);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDehalo::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDehalo>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    // Single-frame, 1-in/1-out, no fps change.
    prm->frameOut.picstruct = prm->frameIn.picstruct;

    // Bit-depth setup.
    const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    const int maxVal   = (1 << bitDepth) - 1;

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamDehalo>(m_param);
    if (!prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        m_buildOptions = strsprintf("-D Type=%s -D bit_depth=%d -D max_val=%d -D dehalo_block_x=%d -D dehalo_block_y=%d",
            bitDepth > 8 ? "ushort" : "uchar",
            bitDepth, maxVal,
            DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
        AddMessage(RGY_LOG_DEBUG, _T("Starting async build for RGY_FILTER_DEHALO_CL: %s\n"),
            char_to_tstring(m_buildOptions).c_str());
        m_dehalo.set(m_cl->buildResourceAsync(_T("RGY_FILTER_DEHALO_CL"), _T("EXE_DATA"), m_buildOptions.c_str()));
    }

    // Output buffer at source resolution.
    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output buffer: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // Decide working resolution. ss == 1.0 → process at source resolution and
    // skip the resize sub-filters entirely; ss > 1.0 → instantiate Spline36
    // resize-up and resize-down sub-filters and run the morphology /
    // mask / apply pipeline at the supersampled resolution.
    const float ssRatio = prm->dehalo.ss;
    m_ssActive = (ssRatio > 1.0f + 1e-6f);
    if (m_ssActive) {
        // Round to even for safer chroma subsampling alignment, mirroring MAA.
        m_ssW = ((int)std::lround(prm->frameIn.width  * ssRatio) + 1) & ~1;
        m_ssH = ((int)std::lround(prm->frameIn.height * ssRatio) + 1) & ~1;
    } else {
        m_ssW = prm->frameIn.width;
        m_ssH = prm->frameIn.height;
    }

    auto initResize = [&](std::unique_ptr<RGYFilter>& filter, const RGYFrameInfo& frameIn, const RGYFrameInfo& frameOut,
        RGY_VPP_RESIZE_ALGO interp, const TCHAR *label, const float bicubicB = FILTER_DEFAULT_RESIZE_BICUBIC_B, const float bicubicC = FILTER_DEFAULT_RESIZE_BICUBIC_C) {
        auto resizePrm = std::make_shared<RGYFilterParamResize>();
        resizePrm->frameIn = frameIn;
        resizePrm->frameOut = frameOut;
        resizePrm->interp = interp;
        resizePrm->bicubic.b = bicubicB;
        resizePrm->bicubic.c = bicubicC;
        resizePrm->baseFps = prm->baseFps;
        resizePrm->bOutOverwrite = false;
        filter = std::make_unique<RGYFilterResize>(m_cl);
        const auto resizeSts = filter->init(resizePrm, m_pLog);
        if (resizeSts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to init dehalo %s sub-filter: %s.\n"), label, get_err_mes(resizeSts));
        }
        return resizeSts;
    };

    const auto lumaCsp = (bitDepth > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;
    auto lumaInfo = prm->frameIn;
    lumaInfo.csp = lumaCsp;

    if (prm->dehalo.mode == VPP_DEHALO_MODE_LEGACY) {
        m_resizeAlphaHaloDown.reset();
        m_resizeAlphaHaloUp.reset();
        m_resizeAlphaUp.reset();
        m_resizeAlphaDown.reset();
        m_alphaHalosSmall.reset();
        m_alphaHalos.reset();
        m_alphaAre.reset();
        m_alphaUgly.reset();
        m_alphaLets.reset();
        m_alphaLimitLow.reset();
        m_alphaLimitHigh.reset();
        m_alphaLimitLowSS.reset();
        m_alphaLimitHighSS.reset();
        m_alphaRemoved.reset();
        if (m_ssActive) {
            RGYFrameInfo ssInfo = prm->frameIn;
            ssInfo.width  = m_ssW;
            ssInfo.height = m_ssH;
            sts = initResize(m_resizeUp, prm->frameIn, ssInfo, RGY_VPP_RESIZE_SPLINE36, _T("upscale"));
            if (sts != RGY_ERR_NONE) return sts;
            sts = initResize(m_resizeDown, ssInfo, prm->frameOut, RGY_VPP_RESIZE_SPLINE36, _T("downscale"));
            if (sts != RGY_ERR_NONE) return sts;

            m_supersampled = m_cl->createFrameBuffer(ssInfo);
            m_expanded     = m_cl->createFrameBuffer(ssInfo);
            m_inpand       = m_cl->createFrameBuffer(ssInfo);
            m_mask         = m_cl->createFrameBuffer(ssInfo);
            m_corrected    = m_cl->createFrameBuffer(ssInfo);
            if (!m_supersampled || !m_expanded || !m_inpand || !m_mask || !m_corrected) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate dehalo supersampled buffers.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
        } else {
            m_resizeUp.reset();
            m_resizeDown.reset();
            m_supersampled.reset();
            m_corrected.reset();
            m_expanded = m_cl->createFrameBuffer(prm->frameIn);
            m_inpand   = m_cl->createFrameBuffer(prm->frameIn);
            m_mask     = m_cl->createFrameBuffer(prm->frameIn);
            if (!m_expanded || !m_inpand || !m_mask) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate dehalo native-res buffers.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
    } else {
        m_resizeUp.reset();
        m_resizeDown.reset();
        m_expanded.reset();
        m_inpand.reset();
        m_mask.reset();
        m_alphaHaloW = std::max(4, (int)std::lround((double)prm->frameIn.width / (double)prm->dehalo.rx));
        m_alphaHaloH = std::max(4, (int)std::lround((double)prm->frameIn.height / (double)prm->dehalo.ry));

        RGYFrameInfo fullInfo = lumaInfo;
        RGYFrameInfo smallInfo = lumaInfo;
        smallInfo.width = m_alphaHaloW;
        smallInfo.height = m_alphaHaloH;
        RGYFrameInfo ssInfo = lumaInfo;
        ssInfo.width = m_ssW;
        ssInfo.height = m_ssH;

        m_alphaHalosSmall = m_cl->createFrameBuffer(smallInfo);
        m_alphaHalos      = m_cl->createFrameBuffer(fullInfo);
        m_alphaAre        = m_cl->createFrameBuffer(fullInfo);
        m_alphaUgly       = m_cl->createFrameBuffer(fullInfo);
        m_alphaLets       = m_cl->createFrameBuffer(fullInfo);
        m_alphaLimitLow   = m_cl->createFrameBuffer(fullInfo);
        m_alphaLimitHigh  = m_cl->createFrameBuffer(fullInfo);
        m_alphaRemoved    = m_cl->createFrameBuffer(fullInfo);
        if (!m_alphaHalosSmall || !m_alphaHalos || !m_alphaAre || !m_alphaUgly || !m_alphaLets || !m_alphaLimitLow || !m_alphaLimitHigh || !m_alphaRemoved) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate dehalo alpha buffers.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        sts = initResize(m_resizeAlphaHaloDown, fullInfo, smallInfo, RGY_VPP_RESIZE_BICUBIC, _T("alpha halo downscale"), 1.0f / 3.0f, 1.0f / 3.0f);
        if (sts != RGY_ERR_NONE) return sts;
        sts = initResize(m_resizeAlphaHaloUp, smallInfo, fullInfo, RGY_VPP_RESIZE_BICUBIC, _T("alpha halo upscale"), 1.0f, 0.0f);
        if (sts != RGY_ERR_NONE) return sts;

        if (m_ssActive) {
            m_supersampled      = m_cl->createFrameBuffer(ssInfo);
            m_corrected         = m_cl->createFrameBuffer(ssInfo);
            m_alphaLimitLowSS   = m_cl->createFrameBuffer(ssInfo);
            m_alphaLimitHighSS  = m_cl->createFrameBuffer(ssInfo);
            if (!m_supersampled || !m_corrected || !m_alphaLimitLowSS || !m_alphaLimitHighSS) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate dehalo alpha supersampled buffers.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
            sts = initResize(m_resizeAlphaUp, fullInfo, ssInfo, RGY_VPP_RESIZE_LANCZOS3, _T("alpha upscale"));
            if (sts != RGY_ERR_NONE) return sts;
            sts = initResize(m_resizeAlphaDown, ssInfo, fullInfo, RGY_VPP_RESIZE_LANCZOS3, _T("alpha downscale"));
            if (sts != RGY_ERR_NONE) return sts;
        } else {
            m_supersampled.reset();
            m_corrected.reset();
            m_alphaLimitLowSS.reset();
            m_alphaLimitHighSS.reset();
            m_resizeAlphaUp.reset();
            m_resizeAlphaDown.reset();
        }
    }

    AddMessage(RGY_LOG_DEBUG,
        _T("dehalo init: src=%dx%d bitDepth=%d, rx=%.2f, ry=%.2f, darkstr=%.2f, brightstr=%.2f, lowsens=%d, highsens=%d, ss=%.2f%s\n"),
        prm->frameIn.width, prm->frameIn.height, bitDepth,
        prm->dehalo.rx, prm->dehalo.ry, prm->dehalo.darkstr, prm->dehalo.brightstr,
        prm->dehalo.lowsens, prm->dehalo.highsens, prm->dehalo.ss,
        m_ssActive ? _T(" (supersampled)") : _T(" (native)"));

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterDehalo::runExpand(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                    float rx, float ry,
                                    RGYOpenCLQueue &queue,
                                    const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, RGY_PLANE_Y);
    RGYWorkSize local(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    RGYWorkSize global(ALIGN(sP.width, DEHALO_BLOCK_X), ALIGN(sP.height, DEHALO_BLOCK_Y));
    auto err = m_dehalo.get()->kernel("dehalo_expand")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            rx, ry);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at dehalo_expand: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDehalo::runInpand(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                    float rx, float ry,
                                    RGYOpenCLQueue &queue,
                                    const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, RGY_PLANE_Y);
    RGYWorkSize local(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    RGYWorkSize global(ALIGN(sP.width, DEHALO_BLOCK_X), ALIGN(sP.height, DEHALO_BLOCK_Y));
    auto err = m_dehalo.get()->kernel("dehalo_inpand")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            rx, ry);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at dehalo_inpand: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDehalo::runMask(RGYFrameInfo *pMaskDst,
                                  const RGYFrameInfo *pSrc, const RGYFrameInfo *pExpand, const RGYFrameInfo *pInpand,
                                  int loScaled, int hiScaled,
                                  RGYOpenCLQueue &queue,
                                  const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc,    RGY_PLANE_Y);
    const auto eP = getPlane(pExpand, RGY_PLANE_Y);
    const auto iP = getPlane(pInpand, RGY_PLANE_Y);
    const auto mP = getPlane(pMaskDst, RGY_PLANE_Y);
    RGYWorkSize local(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_dehalo.get()->kernel("dehalo_mask")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)eP.ptr[0], eP.pitch[0],
            (cl_mem)iP.ptr[0], iP.pitch[0],
            (cl_mem)mP.ptr[0], mP.pitch[0],
            sP.width, sP.height,
            loScaled, hiScaled);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at dehalo_mask: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDehalo::runApply(RGYFrameInfo *pDst,
                                   const RGYFrameInfo *pSrc, const RGYFrameInfo *pExpand, const RGYFrameInfo *pInpand,
                                   const RGYFrameInfo *pMask,
                                   float darkstr, float brightstr,
                                   RGYOpenCLQueue &queue,
                                   const std::vector<RGYOpenCLEvent> &wait_events,
                                   RGYOpenCLEvent *event) {
    const auto sP = getPlane(pSrc,    RGY_PLANE_Y);
    const auto eP = getPlane(pExpand, RGY_PLANE_Y);
    const auto iP = getPlane(pInpand, RGY_PLANE_Y);
    const auto mP = getPlane(pMask,   RGY_PLANE_Y);
    const auto dP = getPlane(pDst,    RGY_PLANE_Y);
    RGYWorkSize local(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    RGYWorkSize global(sP.width, sP.height);
    auto err = m_dehalo.get()->kernel("dehalo_apply")
        .config(queue, local, global, wait_events, event).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)eP.ptr[0], eP.pitch[0],
            (cl_mem)iP.ptr[0], iP.pitch[0],
            (cl_mem)mP.ptr[0], mP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            darkstr, brightstr);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at dehalo_apply: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDehalo::runAlphaRange(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                        int radiusExpand, int radiusInpand,
                                        RGYOpenCLQueue &queue,
                                        const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, RGY_PLANE_Y);
    RGYWorkSize local(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    RGYWorkSize global(ALIGN(sP.width, DEHALO_BLOCK_X), ALIGN(sP.height, DEHALO_BLOCK_Y));
    auto err = m_dehalo.get()->kernel("dehalo_square_range")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            radiusExpand, radiusInpand);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at dehalo_square_range: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDehalo::runAlphaMorph(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                        int radius, bool expand,
                                        RGYOpenCLQueue &queue,
                                        const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, RGY_PLANE_Y);
    RGYWorkSize local(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    RGYWorkSize global(ALIGN(sP.width, DEHALO_BLOCK_X), ALIGN(sP.height, DEHALO_BLOCK_Y));
    auto err = m_dehalo.get()->kernel("dehalo_square_morph")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            radius, expand ? 1 : 0);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at dehalo_square_morph: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDehalo::runAlphaLets(RGYFrameInfo *pDst,
                                       const RGYFrameInfo *pSrc, const RGYFrameInfo *pHalos, const RGYFrameInfo *pAre, const RGYFrameInfo *pUgly,
                                       int loScaled, int highsens,
                                       RGYOpenCLQueue &queue,
                                       const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, RGY_PLANE_Y);
    const auto hP = getPlane(pHalos, RGY_PLANE_Y);
    const auto aP = getPlane(pAre, RGY_PLANE_Y);
    const auto uP = getPlane(pUgly, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, RGY_PLANE_Y);
    RGYWorkSize local(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    RGYWorkSize global(ALIGN(sP.width, DEHALO_BLOCK_X), ALIGN(sP.height, DEHALO_BLOCK_Y));
    auto err = m_dehalo.get()->kernel("dehalo_alpha_lets")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)hP.ptr[0], hP.pitch[0],
            (cl_mem)aP.ptr[0], aP.pitch[0],
            (cl_mem)uP.ptr[0], uP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            loScaled, highsens);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at dehalo_alpha_lets: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDehalo::runAlphaClamp(RGYFrameInfo *pDst,
                                        const RGYFrameInfo *pSrc, const RGYFrameInfo *pLimitLow, const RGYFrameInfo *pLimitHigh,
                                        RGYOpenCLQueue &queue,
                                        const std::vector<RGYOpenCLEvent> &wait_events) {
    const auto sP = getPlane(pSrc, RGY_PLANE_Y);
    const auto lP = getPlane(pLimitLow, RGY_PLANE_Y);
    const auto hP = getPlane(pLimitHigh, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, RGY_PLANE_Y);
    RGYWorkSize local(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    RGYWorkSize global(ALIGN(sP.width, DEHALO_BLOCK_X), ALIGN(sP.height, DEHALO_BLOCK_Y));
    auto err = m_dehalo.get()->kernel("dehalo_alpha_clamp")
        .config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)lP.ptr[0], lP.pitch[0],
            (cl_mem)hP.ptr[0], hP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at dehalo_alpha_clamp: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDehalo::runAlphaThem(RGYFrameInfo *pDst,
                                       const RGYFrameInfo *pSrc, const RGYFrameInfo *pRemove,
                                       const VppDehalo& prm,
                                       RGYOpenCLQueue &queue,
                                       const std::vector<RGYOpenCLEvent> &wait_events,
                                       RGYOpenCLEvent *event) {
    const auto sP = getPlane(pSrc, RGY_PLANE_Y);
    const auto rP = getPlane(pRemove, RGY_PLANE_Y);
    const auto dP = getPlane(pDst, RGY_PLANE_Y);
    RGYWorkSize local(DEHALO_BLOCK_X, DEHALO_BLOCK_Y);
    RGYWorkSize global(ALIGN(sP.width, DEHALO_BLOCK_X), ALIGN(sP.height, DEHALO_BLOCK_Y));
    auto err = m_dehalo.get()->kernel("dehalo_alpha_them")
        .config(queue, local, global, wait_events, event).launch(
            (cl_mem)sP.ptr[0], sP.pitch[0],
            (cl_mem)rP.ptr[0], rP.pitch[0],
            (cl_mem)dP.ptr[0], dP.pitch[0],
            sP.width, sP.height,
            prm.darkstr, prm.brightstr);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at dehalo_alpha_them: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterDehalo::copyChromaPlanes(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
                                           RGYOpenCLQueue &queue,
                                           const std::vector<RGYOpenCLEvent> &wait_events,
                                           RGYOpenCLEvent *event) {
    const int planes = RGY_CSP_PLANES[pDst->csp];
    if (planes <= 1) {
        return RGY_ERR_NONE;
    }
    auto waitFirst = wait_events;
    for (int i = 1; i < planes; i++) {
        const auto pl = (RGY_PLANE)i;
        const auto srcP = getPlane(pSrc, pl);
        const auto dstP = getPlane(pDst, pl);
        auto err = m_cl->copyPlane(const_cast<RGYFrameInfo *>(&dstP), &srcP, nullptr, queue, waitFirst, (i == planes - 1) ? event : nullptr);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("dehalo chroma copy (plane %d) failed: %s.\n"),
                i, get_err_mes(err));
            return err;
        }
        waitFirst.clear();
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDehalo::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue_main,
    const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    *pOutputFrameNum  = 0;
    ppOutputFrames[0] = nullptr;

    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }

    if (!m_dehalo.get()) {
        AddMessage(RGY_LOG_ERROR, _T("dehalo OpenCL program failed to build (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamDehalo>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const int bitDepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
    const int maxVal   = (1 << bitDepth) - 1;
    const int loScaled = (int)((long long)prm->dehalo.lowsens  * maxVal / 100);
    const int hiScaled = (int)((long long)prm->dehalo.highsens * maxVal / 100);
    const int loScaledAlpha = (int)((long long)prm->dehalo.lowsens * maxVal / 255);

    RGYFrameInfo *pOut = &m_frameBuf[0]->frame;

    const int planes = RGY_CSP_PLANES[pOut->csp];
    const bool hasChroma = planes > 1;

    if (prm->dehalo.mode == VPP_DEHALO_MODE_LEGACY) {
        const RGYFrameInfo *pMorphSrc = nullptr;
        if (m_ssActive) {
            int dummyOutNum = 0;
            RGYFrameInfo *outArr[1] = { &m_supersampled->frame };
            auto err = m_resizeUp->filter(const_cast<RGYFrameInfo *>(pInputFrame),
                (RGYFrameInfo **)&outArr, &dummyOutNum, queue_main, wait_events, nullptr);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("dehalo resize-up failed: %s.\n"), get_err_mes(err));
                return err;
            }
            pMorphSrc = &m_supersampled->frame;
        } else {
            pMorphSrc = pInputFrame;
        }

        const std::vector<RGYOpenCLEvent> initialWait = m_ssActive ? std::vector<RGYOpenCLEvent>() : wait_events;
        {
            auto err = runExpand(&m_expanded->frame, pMorphSrc,
                                 prm->dehalo.rx, prm->dehalo.ry,
                                 queue_main,
                                 initialWait);
            if (err != RGY_ERR_NONE) return err;
        }

        {
            auto err = runInpand(&m_inpand->frame, pMorphSrc,
                                 prm->dehalo.rx, prm->dehalo.ry,
                                 queue_main, {});
            if (err != RGY_ERR_NONE) return err;
        }

        {
            auto err = runMask(&m_mask->frame,
                               pMorphSrc, &m_expanded->frame, &m_inpand->frame,
                               loScaled, hiScaled,
                               queue_main, {});
            if (err != RGY_ERR_NONE) return err;
        }

        RGYFrameInfo *pApplyDst = m_ssActive ? &m_corrected->frame : pOut;
        {
            auto err = runApply(pApplyDst,
                                pMorphSrc, &m_expanded->frame, &m_inpand->frame, &m_mask->frame,
                                prm->dehalo.darkstr, prm->dehalo.brightstr,
                                queue_main, {}, (!m_ssActive && !hasChroma) ? event : nullptr);
            if (err != RGY_ERR_NONE) return err;
        }

        if (m_ssActive) {
            int dummyOutNum = 0;
            RGYFrameInfo *outArr[1] = { pOut };
            auto err = m_resizeDown->filter(&m_corrected->frame,
                (RGYFrameInfo **)&outArr, &dummyOutNum, queue_main, {}, hasChroma ? nullptr : event);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("dehalo resize-down failed: %s.\n"), get_err_mes(err));
                return err;
            }
        }
    } else {
        const auto lumaCsp = (bitDepth > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;
        auto inputLuma = getPlane(pInputFrame, RGY_PLANE_Y);
        auto outputLuma = getPlane(pOut, RGY_PLANE_Y);
        inputLuma.csp = lumaCsp;
        outputLuma.csp = lumaCsp;

        auto runResize = [&](RGYFilter *filter, RGYFrameInfo *pIn, RGYFrameInfo *pOutFrame, const TCHAR *label,
            const std::vector<RGYOpenCLEvent>& wait = std::vector<RGYOpenCLEvent>()) {
            int resizeOutNum = 0;
            RGYFrameInfo *resizeOut[1] = { pOutFrame };
            const auto resizeSts = filter->filter(pIn, resizeOut, &resizeOutNum, queue_main, wait, nullptr);
            if (resizeSts != RGY_ERR_NONE || resizeOutNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("dehalo %s failed: %s.\n"), label, get_err_mes(resizeSts));
                return (resizeSts != RGY_ERR_NONE) ? resizeSts : RGY_ERR_UNKNOWN;
            }
            return RGY_ERR_NONE;
        };

        auto err = runResize(m_resizeAlphaHaloDown.get(), &inputLuma, &m_alphaHalosSmall->frame, _T("alpha halo downscale"), wait_events);
        if (err != RGY_ERR_NONE) return err;
        err = runResize(m_resizeAlphaHaloUp.get(), &m_alphaHalosSmall->frame, &m_alphaHalos->frame, _T("alpha halo upscale"));
        if (err != RGY_ERR_NONE) return err;

        const auto searchRadius = dehalo_alpha_search_radius(prm->dehalo);
        err = runAlphaRange(&m_alphaAre->frame, &inputLuma, searchRadius.first, searchRadius.second, queue_main, {});
        if (err != RGY_ERR_NONE) return err;
        err = runAlphaRange(&m_alphaUgly->frame, &m_alphaHalos->frame, searchRadius.first, searchRadius.second, queue_main, {});
        if (err != RGY_ERR_NONE) return err;
        err = runAlphaLets(&m_alphaLets->frame, &inputLuma, &m_alphaHalos->frame, &m_alphaAre->frame, &m_alphaUgly->frame,
            loScaledAlpha, prm->dehalo.highsens, queue_main, {});
        if (err != RGY_ERR_NONE) return err;
        err = runAlphaMorph(&m_alphaLimitLow->frame, &m_alphaLets->frame, 1, false, queue_main, {});
        if (err != RGY_ERR_NONE) return err;
        err = runAlphaMorph(&m_alphaLimitHigh->frame, &m_alphaLets->frame, 1, true, queue_main, {});
        if (err != RGY_ERR_NONE) return err;

        if (m_ssActive) {
            err = runResize(m_resizeAlphaUp.get(), &inputLuma, &m_supersampled->frame, _T("alpha upscale"));
            if (err != RGY_ERR_NONE) return err;
            err = runResize(m_resizeAlphaUp.get(), &m_alphaLimitLow->frame, &m_alphaLimitLowSS->frame, _T("alpha limit-low upscale"));
            if (err != RGY_ERR_NONE) return err;
            err = runResize(m_resizeAlphaUp.get(), &m_alphaLimitHigh->frame, &m_alphaLimitHighSS->frame, _T("alpha limit-high upscale"));
            if (err != RGY_ERR_NONE) return err;
            err = runAlphaClamp(&m_corrected->frame, &m_supersampled->frame, &m_alphaLimitLowSS->frame, &m_alphaLimitHighSS->frame, queue_main, {});
            if (err != RGY_ERR_NONE) return err;
            err = runResize(m_resizeAlphaDown.get(), &m_corrected->frame, &m_alphaRemoved->frame, _T("alpha downscale"));
            if (err != RGY_ERR_NONE) return err;
        } else {
            err = runAlphaClamp(&m_alphaRemoved->frame, &inputLuma, &m_alphaLimitLow->frame, &m_alphaLimitHigh->frame, queue_main, {});
            if (err != RGY_ERR_NONE) return err;
        }
        err = runAlphaThem(&outputLuma, &inputLuma, &m_alphaRemoved->frame, prm->dehalo, queue_main, {}, hasChroma ? nullptr : event);
        if (err != RGY_ERR_NONE) return err;
    }

    // Chroma planes are always copied from the original source.
    if (hasChroma) {
        auto err = copyChromaPlanes(pOut, pInputFrame, queue_main, {}, event);
        if (err != RGY_ERR_NONE) return err;
    }

    pOut->timestamp    = pInputFrame->timestamp;
    pOut->duration     = pInputFrame->duration;
    pOut->inputFrameId = pInputFrame->inputFrameId;
    pOut->picstruct    = pInputFrame->picstruct;
    pOut->flags        = pInputFrame->flags;

    ppOutputFrames[0] = pOut;
    *pOutputFrameNum  = 1;
    return RGY_ERR_NONE;
}

void RGYFilterDehalo::close() {
    m_dehalo.clear();
    m_buildOptions.clear();
    m_resizeUp.reset();
    m_resizeDown.reset();
    m_resizeAlphaHaloDown.reset();
    m_resizeAlphaHaloUp.reset();
    m_resizeAlphaUp.reset();
    m_resizeAlphaDown.reset();
    m_supersampled.reset();
    m_expanded.reset();
    m_inpand.reset();
    m_mask.reset();
    m_corrected.reset();
    m_alphaHalosSmall.reset();
    m_alphaHalos.reset();
    m_alphaAre.reset();
    m_alphaUgly.reset();
    m_alphaLets.reset();
    m_alphaLimitLow.reset();
    m_alphaLimitHigh.reset();
    m_alphaLimitLowSS.reset();
    m_alphaLimitHighSS.reset();
    m_alphaRemoved.reset();
    m_ssW = 0;
    m_ssH = 0;
    m_alphaHaloW = 0;
    m_alphaHaloH = 0;
    m_ssActive = false;
    m_frameBuf.clear();
    m_cl.reset();
}
