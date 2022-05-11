// -----------------------------------------------------------------------------------------
// QSVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2019-2021 rigaya
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
#include "convert_csp.h"
#include "rgy_filter.h"
#include "rgy_prm.h"


RGY_ERR RGYFilterResize::resizePlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const float ratioX = pInputPlane->width / (float)(pOutputPlane->width);
    const float ratioY = pInputPlane->height / (float)(pOutputPlane->height);
    const float ratioDistX = (pInputPlane->width <= pOutputPlane->width) ? 1.0f : pOutputPlane->width / (float)(pInputPlane->width);
    const float ratioDistY = (pInputPlane->height <= pOutputPlane->height) ? 1.0f : pOutputPlane->height / (float)(pInputPlane->height);

    auto pResizeParam = std::dynamic_pointer_cast<RGYFilterParamResize>(m_param);
    if (!pResizeParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    {
        const char *kernel_name = nullptr;
        RGY_ERR err = RGY_ERR_NONE;
        RGYWorkSize local(32, 8);
        RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
        switch (pResizeParam->interp) {
        case RGY_VPP_RESIZE_BILINEAR:
            kernel_name = "kernel_resize_texture_bilinear";
            err = m_resize.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
                (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
                (cl_mem)pInputPlane->ptr[0],
                ratioX, ratioY
            );
            break;
        case RGY_VPP_RESIZE_LANCZOS2:
        case RGY_VPP_RESIZE_LANCZOS3:
        case RGY_VPP_RESIZE_LANCZOS4:
            kernel_name = "kernel_resize_lanczos";
            err = m_resize.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
                (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
                (cl_mem)pInputPlane->ptr[0],
                ratioX, ratioY, ratioDistX, ratioDistY);
            break;
        case RGY_VPP_RESIZE_SPLINE16:
        case RGY_VPP_RESIZE_SPLINE36:
        case RGY_VPP_RESIZE_SPLINE64:
        case RGY_VPP_RESIZE_AUTO:
        default:
            kernel_name = "kernel_resize_spline";
            err = m_resize.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
                (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
                (cl_mem)pInputPlane->ptr[0],
                ratioX, ratioY, ratioDistX, ratioDistY,
                (cl_mem)m_weightSpline->mem());
            break;
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (resizePlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterResize::resizeFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto err = m_cl->createImageFromFrameBuffer(m_srcImage, *pInputFrame, true, CL_MEM_READ_ONLY);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to create image from buffer: %s.\n"), get_err_mes(err));
        return err;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(&m_srcImage->frame, (RGY_PLANE)i);
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[pOutputFrame->csp] - 1) ? event : nullptr;
        err = resizePlane(&planeDst, &planeSrc, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to resize frame(%d) %s: %s\n"), i, cl_errmes(err));
            return err_cl_to_rgy(err);
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterResize::RGYFilterResize(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_bInterlacedWarn(false), m_weightSpline(), m_resize(), m_srcImage() {
    m_name = _T("resize");
}

RGYFilterResize::~RGYFilterResize() {
    close();
}

RGY_ERR RGYFilterResize::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto pResizeParam = std::dynamic_pointer_cast<RGYFilterParamResize>(pParam);
    if (!pResizeParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (pResizeParam->frameOut.height <= 0 || pResizeParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto err = AllocFrameBuf(pResizeParam->frameOut, 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < 4; i++) {
        pResizeParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamResize>(m_param);
    if (!m_resize.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
        || prmPrev->interp != pResizeParam->interp) {
        int radius = 1;
        switch (pResizeParam->interp) {
        case RGY_VPP_RESIZE_LANCZOS2:
        case RGY_VPP_RESIZE_SPLINE16:
            radius = 2;
            break;
        case RGY_VPP_RESIZE_SPLINE36:
        case RGY_VPP_RESIZE_LANCZOS3:
            radius = 3;
            break;
        case RGY_VPP_RESIZE_LANCZOS4:
        case RGY_VPP_RESIZE_SPLINE64:
            radius = 4;
            break;
        case RGY_VPP_RESIZE_BILINEAR:
        default:
            break;
        }
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d -D radius=%d",
            RGY_CSP_BIT_DEPTH[pResizeParam->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[pResizeParam->frameOut.csp],
            radius);
        m_resize.set(std::move(m_cl->buildResourceAsync(_T("RGY_FILTER_RESIZE_CL"), _T("EXE_DATA"), options.c_str())));
        if (!m_weightSpline
            && (   pResizeParam->interp == RGY_VPP_RESIZE_SPLINE16
                || pResizeParam->interp == RGY_VPP_RESIZE_SPLINE36
                || pResizeParam->interp == RGY_VPP_RESIZE_SPLINE64)) {
            static const auto SPLINE16_WEIGHT = std::vector<float>{
                1.0f,       -9.0f/5.0f,  -1.0f/5.0f, 1.0f,
                -1.0f/3.0f,  9.0f/5.0f, -46.0f/15.0f, 8.0f/5.0f
            };
            static const auto SPLINE36_WEIGHT = std::vector<float>{
                13.0f/11.0f, -453.0f/209.0f,    -3.0f/209.0f,  1.0f,
                -6.0f/11.0f,  612.0f/209.0f, -1038.0f/209.0f,  540.0f/209.0f,
                 1.0f/11.0f, -159.0f/209.0f,   434.0f/209.0f, -384.0f/209.0f
            };
            static const auto SPLINE64_WEIGHT = std::vector<float>{
                 49.0f/41.0f, -6387.0f/2911.0f,     -3.0f/2911.0f,  1.0f,
                -24.0f/41.0f,  9144.0f/2911.0f, -15504.0f/2911.0f,  8064.0f/2911.0f,
                  6.0f/41.0f, -3564.0f/2911.0f,   9726.0f/2911.0f, -8604.0f/2911.0f,
                 -1.0f/41.0f,   807.0f/2911.0f,  -3022.0f/2911.0f,  3720.0f/2911.0f
            };
            const std::vector<float> *weight = nullptr;
            switch (pResizeParam->interp) {
            case RGY_VPP_RESIZE_SPLINE16: weight = &SPLINE16_WEIGHT; break;
            case RGY_VPP_RESIZE_SPLINE36: weight = &SPLINE36_WEIGHT; break;
            case RGY_VPP_RESIZE_SPLINE64: weight = &SPLINE64_WEIGHT; break;
            default: {
                AddMessage(RGY_LOG_ERROR, _T("unknown interpolation type: %d.\n"), pResizeParam->interp);
                return RGY_ERR_INVALID_PARAM;
            }
            }

            m_weightSpline = m_cl->copyDataToBuffer(weight->data(), sizeof((*weight)[0]) * weight->size(), CL_MEM_READ_ONLY);
            if (!m_weightSpline) {
                AddMessage(RGY_LOG_ERROR, _T("failed to send weight to gpu memory.\n"));
                return RGY_ERR_NULL_PTR;
            }
        }
    }

    m_infoStr = strsprintf(_T("resize(%s): %dx%d -> %dx%d"),
        get_chr_from_value(list_vpp_resize, pResizeParam->interp),
        pResizeParam->frameIn.width, pResizeParam->frameIn.height,
        pResizeParam->frameOut.width, pResizeParam->frameOut.height);

    //コピーを保存
    m_param = pResizeParam;
    return sts;
}

RGY_ERR RGYFilterResize::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
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

    if (!m_resize.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_RESIZE_CL(resize)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }

    //if (interlaced(*pInputFrame)) {
    //    return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
    //}
    const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != RGYCLMemcpyD2D) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    static const auto supportedCspYV12   = make_array<RGY_CSP>(RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
    static const auto supportedCspYUV444 = make_array<RGY_CSP>(RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);

    auto pResizeParam = std::dynamic_pointer_cast<RGYFilterParamResize>(m_param);
    if (!pResizeParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    sts = resizeFrame(ppOutputFrames[0], pInputFrame, queue, wait_events, event);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at resizeFrame (%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }

    return sts;
}

void RGYFilterResize::close() {
    m_srcImage.reset();
    m_frameBuf.clear();
    m_resize.clear();
    m_weightSpline.reset();
    m_cl.reset();
    m_bInterlacedWarn = false;
}
