// -----------------------------------------------------------------------------------------
// QSVEnc/VCEEnc/rkmppenc by rigaya
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
// -----------------------------------------------------------------------------------------

#include <array>
#include <vector>
#include "convert_csp.h"
#include "rgy_filter_softlight.h"

static const int SOFTLIGHT_BLOCK_X = 32;
static const int SOFTLIGHT_BLOCK_Y = 8;

RGYFilterSoftLight::RGYFilterSoftLight(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_convIn(),
    m_convOut(),
    m_hsvH(),
    m_hsvS(),
    m_hsvV(),
    m_reduce(),
    m_softlight(),
    m_numGroupsLastDispatch(0) {
    m_name = _T("softlight");
}

RGYFilterSoftLight::~RGYFilterSoftLight() {
    close();
}

RGY_ERR RGYFilterSoftLight::checkParam(const std::shared_ptr<RGYFilterParamSoftLight> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (rgy_csp_has_alpha(prm->frameOut.csp)) {
        AddMessage(RGY_LOG_ERROR, _T("softlight is not supported on alpha csp %s.\n"), RGY_CSP_NAMES[prm->frameOut.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSoftLight::allocWork(const RGYFrameInfo& rgbFrame) {
    const auto frameSize = (size_t)rgbFrame.width * rgbFrame.height * sizeof(float);
    auto allocBuf = [&](std::unique_ptr<RGYCLBuf>& buf, const size_t size, const TCHAR *name) {
        if (!buf || buf->size() < size) {
            buf = m_cl->createBuffer(size, CL_MEM_READ_WRITE);
            if (!buf) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate %s buffer.\n"), name);
                return RGY_ERR_MEMORY_ALLOC;
            }
        }
        return RGY_ERR_NONE;
    };
    auto sts = allocBuf(m_hsvH, frameSize, _T("HSV H"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocBuf(m_hsvS, frameSize, _T("HSV S"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = allocBuf(m_hsvV, frameSize, _T("HSV V"));
    if (sts != RGY_ERR_NONE) return sts;

    const int wgX = (rgbFrame.width  + SOFTLIGHT_BLOCK_X - 1) / SOFTLIGHT_BLOCK_X;
    const int wgY = (rgbFrame.height + SOFTLIGHT_BLOCK_Y - 1) / SOFTLIGHT_BLOCK_Y;
    const size_t reduceBytes = (size_t)wgX * wgY * 6 * sizeof(long long);
    return allocBuf(m_reduce, reduceBytes, _T("reduce"));
}

RGY_ERR RGYFilterSoftLight::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSoftLight>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if ((sts = checkParam(prm)) != RGY_ERR_NONE) {
        return sts;
    }

    const auto rgbCsp = RGY_CSP_RGB_16;
    if (pParam->frameIn.csp != rgbCsp) {
        VideoVUIInfo vui = prm->vuiInfo;
        if (vui.matrix == RGY_MATRIX_UNSPECIFIED) {
            vui.matrix = (CspMatrix)COLOR_VALUE_AUTO_RESOLUTION;
        }
        vui.apply_auto(vui, pParam->frameIn.height);
        {
            auto filter = std::make_unique<RGYFilterCspCrop>(m_cl);
            auto paramCrop = std::make_shared<RGYFilterParamCrop>();
            paramCrop->frameIn = pParam->frameIn;
            paramCrop->frameOut = pParam->frameIn;
            paramCrop->frameOut.csp = rgbCsp;
            paramCrop->matrix = vui.matrix;
            paramCrop->baseFps = pParam->baseFps;
            paramCrop->frameIn.mem_type = RGY_MEM_TYPE_GPU;
            paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
            paramCrop->bOutOverwrite = false;
            sts = filter->init(paramCrop, m_pLog);
            if (sts != RGY_ERR_NONE) return sts;
            m_convIn = std::move(filter);
        }
        {
            auto filter = std::make_unique<RGYFilterCspCrop>(m_cl);
            auto paramCrop = std::make_shared<RGYFilterParamCrop>();
            paramCrop->frameIn = pParam->frameIn;
            paramCrop->frameIn.csp = rgbCsp;
            paramCrop->frameOut = pParam->frameOut;
            paramCrop->matrix = vui.matrix;
            paramCrop->baseFps = pParam->baseFps;
            paramCrop->frameIn.mem_type = RGY_MEM_TYPE_GPU;
            paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
            paramCrop->bOutOverwrite = false;
            sts = filter->init(paramCrop, m_pLog);
            if (sts != RGY_ERR_NONE) return sts;
            m_convOut = std::move(filter);
        }
    } else {
        m_convIn.reset();
        m_convOut.reset();
    }

    RGYFrameInfo rgbFrame = pParam->frameIn;
    rgbFrame.csp = rgbCsp;
    if ((sts = allocWork(rgbFrame)) != RGY_ERR_NONE) {
        return sts;
    }
    const auto options = strsprintf("-D softlight_block_x=%d -D softlight_block_y=%d", SOFTLIGHT_BLOCK_X, SOFTLIGHT_BLOCK_Y);
    m_softlight.set(m_cl->buildResourceAsync(_T("RGY_FILTER_SOFTLIGHT_CL"), _T("EXE_DATA"), options.c_str()));

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    tstring info = _T("softlight: ");
    if (m_convIn) {
        info += m_convIn->GetInputMessage() + _T("\n");
    }
    const tstring INFO_INDENT = _T("               ");
    const auto extraIndent = tstring(_tcslen(_T("softlight: ")), _T(' '));
    info += tstring(INFO_INDENT) + extraIndent + pParam->print();
    if (m_convOut) {
        info += tstring(_T("\n")) + tstring(INFO_INDENT) + extraIndent + m_convOut->GetInputMessage();
    }
    setFilterInfo(info);
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterSoftLight::finaliseReduction(RGYOpenCLQueue &queue, std::array<long long, 6>& host) {
    const size_t bytes = (size_t)m_numGroupsLastDispatch * host.size() * sizeof(host[0]);
    std::vector<long long> partials(m_numGroupsLastDispatch * host.size(), 0);

    RGYOpenCLEvent readEvent;
    auto clerr = clEnqueueReadBuffer(queue.get(), m_reduce->mem(),
        CL_FALSE, 0, bytes, partials.data(), 0, nullptr, readEvent.reset_ptr());
    if (clerr != CL_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("softlight reduction readback failed: %s.\n"), cl_errmes(clerr));
        return err_cl_to_rgy(clerr);
    }
    auto sts = readEvent.wait();
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("softlight reduction readback wait failed: %s.\n"), get_err_mes(sts));
        return sts;
    }

    host = {};
    for (int g = 0; g < m_numGroupsLastDispatch; g++) {
        for (int i = 0; i < (int)host.size(); i++) {
            host[i] += partials[g * host.size() + i];
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSoftLight::procFrame(RGYFrameInfo *pFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSoftLight>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pFrame->csp != RGY_CSP_RGB_16) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }

    const auto planeR = getPlane(pFrame, RGY_PLANE_R);
    const auto planeG = getPlane(pFrame, RGY_PLANE_G);
    const auto planeB = getPlane(pFrame, RGY_PLANE_B);
    const int width = planeR.width;
    const int height = planeR.height;
    if ((int64_t)width * height <= 0) {
        return RGY_ERR_NONE;
    }
    if (auto sts = allocWork(*pFrame); sts != RGY_ERR_NONE) {
        return sts;
    }

    const auto mode = prm->softlight.mode;
    const auto formula = prm->softlight.formula;
    const int modeInt = (int)mode;
    const int formulaInt = (int)formula;
    const bool neutralize =
        mode == VppSoftLightMode::NEUTRALIZE
        || mode == VppSoftLightMode::LIGHTNESS
        || mode == VppSoftLightMode::NEUTRALIZE_BOOST_SAT
        || mode == VppSoftLightMode::NEUTRALIZE_FULL
        || mode == VppSoftLightMode::NEUTRALIZE_BOOST;
    const bool rgbBoost =
        mode == VppSoftLightMode::NEUTRALIZE_BOOST
        || mode == VppSoftLightMode::BOOST;

    RGYWorkSize local(SOFTLIGHT_BLOCK_X, SOFTLIGHT_BLOCK_Y);
    RGYWorkSize global(width, height);
    std::vector<RGYOpenCLEvent> kernelWait = wait_events;
    RGYOpenCLEvent prevEvent;

    auto setPrevEvent = [&](RGYOpenCLEvent *ev) {
        kernelWait.clear();
        if (ev) {
            kernelWait.push_back(*ev);
        }
    };
    auto launchErr = [&](const char *kernel_name, RGY_ERR err) {
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"), char_to_tstring(kernel_name).c_str(), get_err_mes(err));
        }
        return err;
    };

    if (mode == VppSoftLightMode::NEUTRALIZE || mode == VppSoftLightMode::NEUTRALIZE_BOOST_SAT) {
        const char *kernel_name = "kernel_rgb_to_v_u16";
        auto err = m_softlight.get()->kernel(kernel_name).config(queue, local, global, kernelWait, &prevEvent).launch(
            (cl_mem)planeR.ptr[0], planeR.pitch[0], (cl_mem)planeG.ptr[0], planeG.pitch[0], (cl_mem)planeB.ptr[0], planeB.pitch[0],
            width, height, m_hsvV->mem());
        if (launchErr(kernel_name, err) != RGY_ERR_NONE) return err;
        setPrevEvent(&prevEvent);
    } else if (mode == VppSoftLightMode::LIGHTNESS) {
        const char *kernel_name = "kernel_rgb_to_hs_u16";
        auto err = m_softlight.get()->kernel(kernel_name).config(queue, local, global, kernelWait, &prevEvent).launch(
            (cl_mem)planeR.ptr[0], planeR.pitch[0], (cl_mem)planeG.ptr[0], planeG.pitch[0], (cl_mem)planeB.ptr[0], planeB.pitch[0],
            width, height, m_hsvH->mem(), m_hsvS->mem());
        if (launchErr(kernel_name, err) != RGY_ERR_NONE) return err;
        setPrevEvent(&prevEvent);
    }

    if (neutralize) {
        const int wgX = (width  + SOFTLIGHT_BLOCK_X - 1) / SOFTLIGHT_BLOCK_X;
        const int wgY = (height + SOFTLIGHT_BLOCK_Y - 1) / SOFTLIGHT_BLOCK_Y;
        m_numGroupsLastDispatch = wgX * wgY;
        const char *kernel_name = "kernel_reduce_rgb_u16";
        auto err = m_softlight.get()->kernel(kernel_name).config(queue, local, global, kernelWait, &prevEvent).launch(
            (cl_mem)planeR.ptr[0], planeR.pitch[0], (cl_mem)planeG.ptr[0], planeG.pitch[0], (cl_mem)planeB.ptr[0], planeB.pitch[0],
            width, height, m_reduce->mem());
        if (launchErr(kernel_name, err) != RGY_ERR_NONE) return err;
        setPrevEvent(&prevEvent);

        std::array<long long, 6> host = {};
        auto sts = finaliseReduction(queue, host);
        if (sts != RGY_ERR_NONE) return sts;
        kernelWait.clear();

        const double totalPx = (double)((int64_t)width * height);
        std::array<float, 3> b = {};
        for (int i = 0; i < 3; i++) {
            const double denom = totalPx - (prm->softlight.skipblack ? (double)host[3 + i] : 0.0);
            const double mean = (denom > 0.0) ? ((double)host[i] / denom) / 65535.0 : 0.0;
            b[i] = (float)(1.0 - mean);
        }
        const struct {
            const char *name;
            const RGYFrameInfo *plane;
            float b;
        } channels[] = {
            { "kernel_softlight_scalar_u16", &planeR, b[0] },
            { "kernel_softlight_scalar_u16", &planeG, b[1] },
            { "kernel_softlight_scalar_u16", &planeB, b[2] },
        };
        for (const auto& ch : channels) {
            err = m_softlight.get()->kernel(ch.name).config(queue, local, global, kernelWait, &prevEvent).launch(
                (cl_mem)ch.plane->ptr[0], ch.plane->pitch[0], width, height, ch.b, formulaInt);
            if (launchErr(ch.name, err) != RGY_ERR_NONE) return err;
            setPrevEvent(&prevEvent);
        }
    }

    if (rgbBoost) {
        const RGYFrameInfo *planes[] = { &planeR, &planeG, &planeB };
        for (const auto plane : planes) {
            const char *kernel_name = "kernel_softlight_self_u16";
            auto err = m_softlight.get()->kernel(kernel_name).config(queue, local, global, kernelWait, &prevEvent).launch(
                (cl_mem)plane->ptr[0], plane->pitch[0], width, height, formulaInt);
            if (launchErr(kernel_name, err) != RGY_ERR_NONE) return err;
            setPrevEvent(&prevEvent);
        }
    }

    if (mode == VppSoftLightMode::NEUTRALIZE || mode == VppSoftLightMode::NEUTRALIZE_BOOST_SAT) {
        {
            const char *kernel_name = "kernel_rgb_to_hs_u16";
            auto err = m_softlight.get()->kernel(kernel_name).config(queue, local, global, kernelWait, &prevEvent).launch(
                (cl_mem)planeR.ptr[0], planeR.pitch[0], (cl_mem)planeG.ptr[0], planeG.pitch[0], (cl_mem)planeB.ptr[0], planeB.pitch[0],
                width, height, m_hsvH->mem(), m_hsvS->mem());
            if (launchErr(kernel_name, err) != RGY_ERR_NONE) return err;
            setPrevEvent(&prevEvent);
        }
        if (mode == VppSoftLightMode::NEUTRALIZE_BOOST_SAT) {
            const char *kernel_name = "kernel_softlight_self_f32";
            auto err = m_softlight.get()->kernel(kernel_name).config(queue, local, global, kernelWait, &prevEvent).launch(
                m_hsvS->mem(), width, height, formulaInt);
            if (launchErr(kernel_name, err) != RGY_ERR_NONE) return err;
            setPrevEvent(&prevEvent);
        }
        const char *kernel_name = "kernel_hsv_to_rgb_u16";
        auto err = m_softlight.get()->kernel(kernel_name).config(queue, local, global, kernelWait, event).launch(
            (cl_mem)planeR.ptr[0], planeR.pitch[0], (cl_mem)planeG.ptr[0], planeG.pitch[0], (cl_mem)planeB.ptr[0], planeB.pitch[0],
            width, height, m_hsvH->mem(), m_hsvS->mem(), m_hsvV->mem());
        if (launchErr(kernel_name, err) != RGY_ERR_NONE) return err;
    } else if (mode == VppSoftLightMode::LIGHTNESS) {
        {
            const char *kernel_name = "kernel_rgb_to_v_u16";
            auto err = m_softlight.get()->kernel(kernel_name).config(queue, local, global, kernelWait, &prevEvent).launch(
                (cl_mem)planeR.ptr[0], planeR.pitch[0], (cl_mem)planeG.ptr[0], planeG.pitch[0], (cl_mem)planeB.ptr[0], planeB.pitch[0],
                width, height, m_hsvV->mem());
            if (launchErr(kernel_name, err) != RGY_ERR_NONE) return err;
            setPrevEvent(&prevEvent);
        }
        const char *kernel_name = "kernel_hsv_to_rgb_u16";
        auto err = m_softlight.get()->kernel(kernel_name).config(queue, local, global, kernelWait, event).launch(
            (cl_mem)planeR.ptr[0], planeR.pitch[0], (cl_mem)planeG.ptr[0], planeG.pitch[0], (cl_mem)planeB.ptr[0], planeB.pitch[0],
            width, height, m_hsvH->mem(), m_hsvS->mem(), m_hsvV->mem());
        if (launchErr(kernel_name, err) != RGY_ERR_NONE) return err;
    } else if (mode == VppSoftLightMode::SATURATION) {
        {
            const char *kernel_name = "kernel_rgb_to_hsv_u16";
            auto err = m_softlight.get()->kernel(kernel_name).config(queue, local, global, kernelWait, &prevEvent).launch(
                (cl_mem)planeR.ptr[0], planeR.pitch[0], (cl_mem)planeG.ptr[0], planeG.pitch[0], (cl_mem)planeB.ptr[0], planeB.pitch[0],
                width, height, m_hsvH->mem(), m_hsvS->mem(), m_hsvV->mem());
            if (launchErr(kernel_name, err) != RGY_ERR_NONE) return err;
            setPrevEvent(&prevEvent);
        }
        {
            const char *kernel_name = "kernel_softlight_self_f32";
            auto err = m_softlight.get()->kernel(kernel_name).config(queue, local, global, kernelWait, &prevEvent).launch(
                m_hsvS->mem(), width, height, formulaInt);
            if (launchErr(kernel_name, err) != RGY_ERR_NONE) return err;
            setPrevEvent(&prevEvent);
        }
        const char *kernel_name = "kernel_hsv_to_rgb_u16";
        auto err = m_softlight.get()->kernel(kernel_name).config(queue, local, global, kernelWait, event).launch(
            (cl_mem)planeR.ptr[0], planeR.pitch[0], (cl_mem)planeG.ptr[0], planeG.pitch[0], (cl_mem)planeB.ptr[0], planeB.pitch[0],
            width, height, m_hsvH->mem(), m_hsvS->mem(), m_hsvV->mem());
        if (launchErr(kernel_name, err) != RGY_ERR_NONE) return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSoftLight::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }
    if (!m_softlight.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_SOFTLIGHT_CL(m_softlight)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != RGYCLMemcpyD2D) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    RGYFrameInfo targetFrame = *pInputFrame;
    if (m_convIn) {
        int cropFilterOutputNum = 0;
        RGYFrameInfo *outInfo[1] = { nullptr };
        auto sts_filter = m_convIn->filter(&targetFrame, (RGYFrameInfo **)&outInfo, &cropFilterOutputNum, queue, wait_events, event);
        if (outInfo[0] == nullptr || cropFilterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_convIn->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || cropFilterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_convIn->name().c_str());
            return sts_filter;
        }
        targetFrame = *outInfo[0];
    } else {
        auto sts_copy = m_cl->copyFrame(ppOutputFrames[0], pInputFrame, nullptr, queue, wait_events, event);
        if (sts_copy != RGY_ERR_NONE) {
            return sts_copy;
        }
        targetFrame = *ppOutputFrames[0];
    }

    if ((sts = procFrame(&targetFrame, queue, wait_events, event)) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at softlight(%s): %s.\n"),
            RGY_CSP_NAMES[targetFrame.csp], get_err_mes(sts));
        return sts;
    }

    if (m_convOut) {
        auto sts_filter = m_convOut->filter(&targetFrame, ppOutputFrames, pOutputFrameNum, queue, wait_events, event);
        if (ppOutputFrames[0] == nullptr || *pOutputFrameNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_convOut->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || *pOutputFrameNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_convOut->name().c_str());
            return sts_filter;
        }
    }
    return sts;
}

void RGYFilterSoftLight::close() {
    m_convIn.reset();
    m_convOut.reset();
    m_hsvH.reset();
    m_hsvS.reset();
    m_hsvV.reset();
    m_reduce.reset();
    m_softlight.clear();
    m_frameBuf.clear();
    m_cl.reset();
    m_numGroupsLastDispatch = 0;
}
