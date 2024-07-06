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
    const int   swapuv     = prm->tweak.swapuv ? 1 : 0;

    auto planeInputY = getPlane(pFrame, RGY_PLANE_Y);
    auto planeInputU = getPlane(pFrame, RGY_PLANE_U);
    auto planeInputV = getPlane(pFrame, RGY_PLANE_V);

    auto wait_events_copy = wait_events;

    //Y
    if (   contrast   != 1.0f
        || brightness != 0.0f
        || gamma      != 1.0f
        || prm->tweak.y.enabled()) {
        RGYWorkSize local(TWEAK_BLOCK_X, TWEAK_BLOCK_Y);
        RGYWorkSize global(divCeil(planeInputY.width, 4), planeInputY.height);
        const char *kernel_name = "kernel_tweak_y";
        auto err = m_tweak.get()->kernel(kernel_name).config(queue, local, global, wait_events_copy, event).launch(
            (cl_mem)planeInputY.ptr[0], planeInputY.pitch[0], planeInputY.width, planeInputY.height,
            contrast, brightness, 1.0f / gamma,
            prm->tweak.y.gain, prm->tweak.y.offset);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (procFrame(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pFrame->csp], get_err_mes(err));
            return err;
        }
        wait_events_copy.clear();
    }

    //UV
    if (   saturation != 1.0f
        || hue_degree != 0.0f
        || swapuv
        || prm->tweak.cb.enabled()
        || prm->tweak.cr.enabled()) {
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
            saturation, std::sin(hue) * saturation, std::cos(hue) * saturation, swapuv,
            prm->tweak.cb.gain, prm->tweak.cb.offset,
            prm->tweak.cr.gain, prm->tweak.cr.offset);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (procFrame(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pFrame->csp], get_err_mes(err));
            return err;
        }
        wait_events_copy.clear();
    }
    return RGY_ERR_NONE;
}


RGY_ERR RGYFilterTweak::procFrameRGB(RGYFrameInfo *pFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamTweak>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    std::vector<std::pair<const VppTweakChannel *, RGYFrameInfo>> targetPlanes;
    if (prm->tweak.r.enabled()) { targetPlanes.push_back(std::make_pair(&prm->tweak.r, getPlane(pFrame, RGY_PLANE_R))); }
    if (prm->tweak.g.enabled()) { targetPlanes.push_back(std::make_pair(&prm->tweak.g, getPlane(pFrame, RGY_PLANE_G))); }
    if (prm->tweak.b.enabled()) { targetPlanes.push_back(std::make_pair(&prm->tweak.b, getPlane(pFrame, RGY_PLANE_B))); }

    auto wait_events_copy = wait_events;

    for (auto& target : targetPlanes) {
        const auto& plane = target.second;
        const RGYWorkSize local(TWEAK_BLOCK_X, TWEAK_BLOCK_Y);
        const RGYWorkSize global(divCeil(plane.width, 4), plane.height);
        const char *kernel_name = "kernel_tweak_y";
        auto err = m_tweakRGB.get()->kernel(kernel_name).config(queue, local, global, wait_events_copy, event).launch(
            (cl_mem)plane.ptr[0], plane.pitch[0], plane.width, plane.height,
            target.first->gain, target.first->offset, 1.0f / target.first->gamma,
            0.0f, 0.0f);
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
    for (auto prmtweak : { &prm->tweak.r, &prm->tweak.g, &prm->tweak.b, &prm->tweak.y, &prm->tweak.cb, &prm->tweak.cr }) {
        if (prmtweak->offset < -1.0f || 1.0f < prmtweak->offset) {
            prmtweak->offset = clamp(prmtweak->offset, -1.0f, 1.0f);
            AddMessage(RGY_LOG_WARN, _T("offset should be in range of %.1f - %.1f.\n"), -1.0f, 1.0f);
        }
        if (prmtweak->gain < -2.0f || 2.0f < prmtweak->gain) {
            prmtweak->gain = clamp(prmtweak->gain, -2.0f, 2.0f);
            AddMessage(RGY_LOG_WARN, _T("gain should be in range of %.1f - %.1f.\n"), -2.0f, 2.0f);
        }
    }
    for (auto prmtweak : { &prm->tweak.r, &prm->tweak.g, &prm->tweak.b }) {
        if (prmtweak->gamma < 0.1f || 10.0f < prmtweak->gamma) {
            prmtweak->gamma = clamp(prmtweak->gamma, 0.1f, 10.0f);
            AddMessage(RGY_LOG_WARN, _T("gamma should be in range of %.1f - %.1f.\n"), 0.1f, 10.0f);
        }
    }

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamTweak>(m_param);
    if (!m_tweak.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
        || prmPrev->tweak.y.enabled()  != prm->tweak.y.enabled()
        || prmPrev->tweak.cb.enabled() != prm->tweak.cb.enabled()
        || prmPrev->tweak.cr.enabled() != prm->tweak.cr.enabled()) {
        const auto options = strsprintf("-D Type=%s -D Type4=%s -D bit_depth=%d"
            " -D TWEAK_Y=%d -D TWEAK_CB=%d -D TWEAK_CR=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort4" : "uchar4",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp],
            prm->tweak.y.enabled() ? 1 : 0,
            prm->tweak.cb.enabled() ? 1 : 0,
            prm->tweak.cr.enabled() ? 1 : 0);
        m_tweak.set(m_cl->buildResourceAsync(_T("RGY_FILTER_TWEAK_CL"), _T("EXE_DATA"), options.c_str()));
    }
    if ((prm->tweak.r.enabled() || prm->tweak.g.enabled() || prm->tweak.b.enabled())
        && (!m_convSrc || !m_convDst)) {
        const auto csp_rgb = RGY_CSP_RGB_16;
        VideoVUIInfo vui = prm->vui;
        vui.setIfUnset(VideoVUIInfo().to((CspMatrix)COLOR_VALUE_AUTO_RESOLUTION).to((CspColorprim)COLOR_VALUE_AUTO_RESOLUTION).to((CspTransfer)COLOR_VALUE_AUTO_RESOLUTION));
        vui.apply_auto(VideoVUIInfo(), pParam->frameIn.height);
        {
            auto filterCrop = std::make_unique<RGYFilterCspCrop>(m_cl);
            auto paramCrop = std::make_shared<RGYFilterParamCrop>();
            paramCrop->frameIn = pParam->frameIn;
            paramCrop->frameOut = pParam->frameIn;
            paramCrop->frameOut.csp = csp_rgb;
            paramCrop->baseFps = pParam->baseFps;
            paramCrop->matrix = vui.matrix;
            paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
            paramCrop->bOutOverwrite = false;
            sts = filterCrop->init(paramCrop, m_pLog);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            m_convSrc = std::move(filterCrop);
        }
        {
            auto filterCrop = std::make_unique<RGYFilterCspCrop>(m_cl);
            auto paramCrop = std::make_shared<RGYFilterParamCrop>();
            paramCrop->frameIn = pParam->frameIn;
            paramCrop->frameIn.csp = csp_rgb;
            paramCrop->frameOut = pParam->frameIn;
            paramCrop->baseFps = pParam->baseFps;
            paramCrop->matrix = vui.matrix;
            paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
            paramCrop->bOutOverwrite = false;
            sts = filterCrop->init(paramCrop, m_pLog);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            m_convDst = std::move(filterCrop);
        }

        const auto options = strsprintf("-D Type=%s -D Type4=%s -D bit_depth=%d"
            " -D TWEAK_Y=%d -D TWEAK_CB=%d -D TWEAK_CR=%d",
            RGY_CSP_BIT_DEPTH[csp_rgb] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[csp_rgb] > 8 ? "ushort4" : "uchar4",
            RGY_CSP_BIT_DEPTH[csp_rgb], 0, 0, 0);
        m_tweakRGB.set(m_cl->buildResourceAsync(_T("RGY_FILTER_TWEAK_CL"), _T("EXE_DATA"), options.c_str()));
    }

    //コピーを保存
    if (prm->tweak.r.enabled() || prm->tweak.g.enabled() || prm->tweak.b.enabled()) {
        tstring str = prm->tweak.print(false) + _T("\n");
        const tstring indent = _T("       ");
        str += indent + m_convSrc->GetInputMessage() + _T("\n");
        if (prm->tweak.r.enabled()) str += indent + _T("r: ") + prm->tweak.r.print() + _T("\n");
        if (prm->tweak.g.enabled()) str += indent + _T("g: ") + prm->tweak.g.print() + _T("\n");
        if (prm->tweak.b.enabled()) str += indent + _T("b: ") + prm->tweak.b.print() + _T("\n");
        str += indent + m_convDst->GetInputMessage();
        setFilterInfo(str);
    } else {
        setFilterInfo(prm->print());
    }
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
    auto prm = std::dynamic_pointer_cast<RGYFilterParamTweak>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->tweak.r.enabled() || prm->tweak.g.enabled() || prm->tweak.b.enabled()) {
        if (!m_tweakRGB.get()) {
            AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_TWEAK_CL(m_tweakRGB)\n"));
            return RGY_ERR_OPENCL_CRUSH;
        }
        RGYFrameInfo *targetFrame = nullptr;
        { // YUV -> RGB
            int cropFilterOutputNum = 0;
            RGYFrameInfo *pCropFilterOutput[1] = { nullptr };
            RGYFrameInfo inFrame = *(ppOutputFrames[0]);
            auto sts_filter = m_convSrc->filter(&inFrame, (RGYFrameInfo **)&pCropFilterOutput, &cropFilterOutputNum, queue, {}, nullptr);
            if (pCropFilterOutput[0] == nullptr || cropFilterOutputNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_convSrc->name().c_str());
                return sts_filter;
            }
            if (sts_filter != RGY_ERR_NONE || cropFilterOutputNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_convSrc->name().c_str());
                return sts_filter;
            }
            targetFrame = pCropFilterOutput[0];
        }
        // RGBでの処理を実行
        sts = procFrameRGB(targetFrame, queue, {}, nullptr);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at procFrameRGB (%s): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
            return sts;
        }
        { // RGB -> YUV
            int cropFilterOutputNum = 0;
            RGYFrameInfo *pCropFilterOutput[1] = { nullptr };
            RGYFrameInfo inFrame = *targetFrame;
            auto sts_filter = m_convDst->filter(&inFrame, (RGYFrameInfo **)&pCropFilterOutput, &cropFilterOutputNum, queue, {}, event);
            if (pCropFilterOutput[0] == nullptr || cropFilterOutputNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_convDst->name().c_str());
                return sts_filter;
            }
            if (sts_filter != RGY_ERR_NONE || cropFilterOutputNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_convDst->name().c_str());
                return sts_filter;
            }
            ppOutputFrames[0] = pCropFilterOutput[0];
        }
    }

    return sts;
}

void RGYFilterTweak::close() {
    m_convSrc.reset();
    m_convDst.reset();
    m_frameBuf.clear();
    m_tweak.clear();
    m_tweakRGB.clear();
    m_cl.reset();
    m_bInterlacedWarn = false;
}
