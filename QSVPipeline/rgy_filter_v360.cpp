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
//
// The projection ray maths (equirectangular / rectilinear / cubemap) is standard textbook
// cartography; this is an independent implementation from that published maths (no third-party
// filter source was used).

#define _USE_MATH_DEFINES
#include <cmath>
#include <map>
#include "rgy_filter_v360.h"

static const int V360_BLOCK_X = 32;
static const int V360_BLOCK_Y = 8;

static void matmul3(const float A[9], const float B[9], float C[9]) {
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            C[r * 3 + c] = A[r * 3 + 0] * B[0 * 3 + c] + A[r * 3 + 1] * B[1 * 3 + c] + A[r * 3 + 2] * B[2 * 3 + c];
        }
    }
}

// R = Ryaw * Rpitch * Rroll (rotates the output ray into the world/input frame)
static void computeRot(float yawDeg, float pitchDeg, float rollDeg, float R[9]) {
    const float a = yawDeg   * (float)M_PI / 180.0f;
    const float b = pitchDeg * (float)M_PI / 180.0f;
    const float c = rollDeg  * (float)M_PI / 180.0f;
    const float ca = cosf(a), sa = sinf(a), cb = cosf(b), sb = sinf(b), cc = cosf(c), sc = sinf(c);
    const float Ry[9] = { ca, 0.0f, sa,  0.0f, 1.0f, 0.0f,  -sa, 0.0f, ca };
    const float Rp[9] = { 1.0f, 0.0f, 0.0f,  0.0f, cb, -sb,  0.0f, sb, cb };
    const float Rr[9] = { cc, -sc, 0.0f,  sc, cc, 0.0f,  0.0f, 0.0f, 1.0f };
    float RpRr[9];
    matmul3(Rp, Rr, RpRr);
    matmul3(Ry, RpRr, R);
}

RGY_ERR RGYFilterV360::procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, float fillValue, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamV360>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    float R[9];
    computeRot(prm->v360.yaw, prm->v360.pitch, prm->v360.roll, R);
    const float out_hfov = prm->v360.out_hfov * (float)M_PI / 180.0f;
    const float in_hfov  = prm->v360.in_hfov  * (float)M_PI / 180.0f;
    const char *kernel_name = "kernel_v360";
    RGYWorkSize local(V360_BLOCK_X, V360_BLOCK_Y);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    auto err = m_v360.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (cl_mem)pInputPlane->ptr[0],  pInputPlane->pitch[0],  pInputPlane->width,  pInputPlane->height,
        R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8],
        out_hfov, in_hfov, fillValue);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterV360::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[pOutputFrame->csp] - 1) ? event : nullptr;
        const float fillValue = (i == 0) ? 0.0f : (float)(1 << (RGY_CSP_BIT_DEPTH[pOutputFrame->csp] - 1));
        auto err = procPlane(&planeDst, &planeSrc, fillValue, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to apply v360 frame(%d) %s: %s\n"), i, RGY_CSP_NAMES[planeSrc.csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterV360::RGYFilterV360(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_v360() {
    m_name = _T("v360");
}

RGYFilterV360::~RGYFilterV360() {
    close();
}

RGY_ERR RGYFilterV360::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamV360>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameIn.height <= 0 || prm->frameIn.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    // output size: explicit w/h, else same as input
    int outW = (prm->v360.w > 0) ? prm->v360.w : prm->frameIn.width;
    int outH = (prm->v360.h > 0) ? prm->v360.h : prm->frameIn.height;
    outW &= ~1; outH &= ~1; // keep even for chroma subsampling
    prm->frameOut.width = outW;
    prm->frameOut.height = outH;

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamV360>(m_param);
    if (!m_v360.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
        || prmPrev->v360.in_proj  != prm->v360.in_proj
        || prmPrev->v360.out_proj != prm->v360.out_proj) {
        const auto options = strsprintf("-D Type=%s -D IN_PROJ=%d -D OUT_PROJ=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            prm->v360.in_proj, prm->v360.out_proj);
        m_v360.set(m_cl->buildResourceAsync(_T("RGY_FILTER_V360_CL"), _T("EXE_DATA"), options.c_str()));
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterV360::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
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
    if (!m_v360.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_V360_CL(m_v360)\n"));
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

    sts = procFrame(ppOutputFrames[0], pInputFrame, queue, wait_events, event);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at procFrame (%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }
    return sts;
}

void RGYFilterV360::close() {
    m_frameBuf.clear();
    m_v360.clear();
    m_cl.reset();
}
