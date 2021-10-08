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
#include "rgy_filter_tweak.h"

static const int TWEAK_BLOCK_X = 64;
static const int TWEAK_BLOCK_Y = 4;

RGY_ERR RGYFilterTweak::procFrame(RGYFrameInfo *pFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamTweak>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const float contrast   = prm->tweak.contrast;
    const float brightness = prm->tweak.brightness;
    const float saturation = prm->tweak.saturation;
    const float gamma      = prm->tweak.gamma;
    const float hue_degree = prm->tweak.hue;

    auto planeInputY = getPlane(pFrame, RGY_PLANE_Y);
    auto planeInputU = getPlane(pFrame, RGY_PLANE_U);
    auto planeInputV = getPlane(pFrame, RGY_PLANE_V);

    auto wait_events_copy = wait_events;

    //Y
    if (   contrast   != 1.0f
        || brightness != 0.0f
        || gamma      != 1.0f) {
        RGYWorkSize local(TWEAK_BLOCK_X, TWEAK_BLOCK_Y);
        RGYWorkSize global(divCeil(planeInputY.width, 4), planeInputY.height);
        const char *kernel_name = "kernel_tweak_y";
        auto err = m_tweak.get()->kernel(kernel_name).config(queue, local, global, wait_events_copy, event).launch(
            (cl_mem)planeInputY.ptr[0], planeInputY.pitch[0], planeInputY.width, planeInputY.height,
            contrast, brightness, 1.0f / gamma);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (procFrame(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pFrame->csp], get_err_mes(err));
            return err;
        }
        wait_events_copy.clear();
    }

    //UV
    if (   saturation != 1.0f
        || hue_degree != 0.0f) {
        if (   planeInputU.width    != planeInputV.width
            || planeInputU.height   != planeInputV.height
            || planeInputU.pitch[0] != planeInputV.pitch[0]) {
            return RGY_ERR_INVALID_CALL;
        }
        RGYWorkSize local(TWEAK_BLOCK_X, TWEAK_BLOCK_Y);
        RGYWorkSize global(divCeil(planeInputU.width, 4), planeInputU.height);
        const float hue = hue_degree * (float)M_PI / 180.0f;
        const char *kernel_name = "kernel_tweak_uv";
        auto err = m_tweak.get()->kernel(kernel_name).config(queue, local, global, wait_events_copy, event).launch(
            (cl_mem)planeInputU.ptr[0], (cl_mem)planeInputV.ptr[0], planeInputU.pitch[0], planeInputU.width, planeInputU.height,
            saturation, std::sin(hue) * saturation, std::cos(hue) * saturation);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (procFrame(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pFrame->csp], get_err_mes(err));
            return err;
        }
        wait_events_copy.clear();
    }
    return RGY_ERR_NONE;
}

RGYFilterTweak::RGYFilterTweak(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_tweak() {
    m_name = _T("tweak");
}

RGYFilterTweak::~RGYFilterTweak() {
    close();
}

RGY_ERR RGYFilterTweak::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamTweak>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //tweakは常に元のフレームを書き換え
    if (!prm->bOutOverwrite) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid param, tweak will overwrite input frame.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    prm->frameOut = prm->frameIn;


    //パラメータチェック
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->tweak.brightness < -1.0f || 1.0f < prm->tweak.brightness) {
        prm->tweak.brightness = clamp(prm->tweak.brightness, -1.0f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("brightness should be in range of %.1f - %.1f.\n"), -1.0f, 1.0f);
    }
    if (prm->tweak.contrast < -2.0f || 2.0f < prm->tweak.contrast) {
        prm->tweak.contrast = clamp(prm->tweak.contrast, -2.0f, 2.0f);
        AddMessage(RGY_LOG_WARN, _T("contrast should be in range of %.1f - %.1f.\n"), -2.0f, 2.0f);
    }
    if (prm->tweak.saturation < 0.0f || 3.0f < prm->tweak.saturation) {
        prm->tweak.saturation = clamp(prm->tweak.saturation, 0.0f, 3.0f);
        AddMessage(RGY_LOG_WARN, _T("saturation should be in range of %.1f - %.1f.\n"), 0.0f, 3.0f);
    }
    if (prm->tweak.gamma < 0.1f || 10.0f < prm->tweak.gamma) {
        prm->tweak.gamma = clamp(prm->tweak.gamma, 0.1f, 10.0f);
        AddMessage(RGY_LOG_WARN, _T("gamma should be in range of %.1f - %.1f.\n"), 0.1f, 10.0f);
    }

    if (!m_tweak.get()
        || std::dynamic_pointer_cast<RGYFilterParamTweak>(m_param)->tweak != prm->tweak) {
        const auto options = strsprintf("-D Type=%s -D Type4=%s -D bit_depth=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort4" : "uchar4",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp]);
        m_tweak.set(m_cl->buildResourceAsync(_T("RGY_FILTER_TWEAK_CL"), _T("EXE_DATA"), options.c_str()));
    }

    //コピーを保存
    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterTweak::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("ppOutputFrames[0] must be set.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (!m_tweak.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_TWEAK_CL(m_tweak)\n"));
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

    sts = procFrame(ppOutputFrames[0], queue, wait_events, event);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at procFrame (%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }

    return sts;
}

void RGYFilterTweak::close() {
    m_frameBuf.clear();
    m_tweak.clear();
    m_cl.reset();
    m_bInterlacedWarn = false;
}
