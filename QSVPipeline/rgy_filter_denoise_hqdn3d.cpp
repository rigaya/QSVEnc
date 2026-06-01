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
#include <array>
#include <cstdint>
#include <vector>
#include "rgy_filter_denoise_hqdn3d.h"

// Workgroup shapes. The spatial passes launch one work-item per
// row (H) / column (V) so the inner loop runs sequentially per
// work-item. The temporal pass is a standard 2D parallel kernel.
static const int HQDN3D_BLOCK_LINEAR = 32;
static const int HQDN3D_TBLOCK_X = 32;
static const int HQDN3D_TBLOCK_Y = 8;

void RGYFilterDenoiseHqdn3d::precalcCoefs(std::vector<float> &table, double dist25) {
    table.assign(2 * HQDN3D_LUT_RADIUS, 0.0f);
    if (dist25 <= 0.0) {
        return;
    }
    // IIR attenuation coefficient derived from:
    //   dist25 = pixel distance at which IIR retains 25% of prior value.
    //   sigma  = -1 / log(0.25). Choosing the rate scale as
    //   (dist25 * sigma) makes exp(-|f|/(dist25*sigma)) = 0.25 exactly
    //   at |f| = dist25. See algorithm description.
    const double clamped = std::min(dist25, 253.9);
    const double sigma   = -1.0 / std::log(0.25);
    const double scale   = clamped * sigma + 1e-7;
    for (int i = 0; i < 2 * HQDN3D_LUT_RADIUS; ++i) {
        const double f = (double)(i - HQDN3D_LUT_RADIUS);
        const double attenuation = std::exp(-std::fabs(f) / scale);
        // Normalize to the [0, 1] float domain. Output of the IIR step
        // is `cur + table[idx]` where idx is signed pixel delta + LUT
        // radius, so `f / 256` puts the additive term back in the
        // normalised luma scale used by the GPU kernel.
        table[i] = (float)(attenuation * f / 256.0);
    }
}

RGYFilterDenoiseHqdn3d::RGYFilterDenoiseHqdn3d(shared_ptr<RGYOpenCLContext> context)
    : RGYFilter(context), m_hqdn3d(), m_coefs(), m_framePrev(), m_framePrevPitchFloats{},
      m_tmpH(), m_tmpHV(), m_tmpPitchFloats(0), m_frameIdx(0), m_firstFrame(true),
      m_fp16Scratch(false) {
    m_name = _T("hqdn3d");
}

RGYFilterDenoiseHqdn3d::~RGYFilterDenoiseHqdn3d() {
    close();
}

RGY_ERR RGYFilterDenoiseHqdn3d::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseHqdn3d>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    // Strength range guard (matches reference)
    auto clampStrength = [&](float &v, const TCHAR *name) {
        if (v < 0.0f || v > 255.0f) {
            AddMessage(RGY_LOG_WARN, _T("%s must be in range 0.0 - 255.0.\n"), name);
            v = clamp(v, 0.0f, 255.0f);
        }
    };
    clampStrength(prm->hqdn3d.luma_spatial,    _T("luma_spatial"));
    clampStrength(prm->hqdn3d.chroma_spatial,  _T("chroma_spatial"));
    clampStrength(prm->hqdn3d.luma_temporal,   _T("luma_temporal"));
    clampStrength(prm->hqdn3d.chroma_temporal, _T("chroma_temporal"));

    // FP16 scratch: m_tmpH / m_tmpHV / m_framePrev[] move from FP32 to
    // FP16 storage when the device advertises cl_khr_fp16. The kernel
    // arithmetic remains float (vload_half / vstore_half convert at the
    // storage boundary). Safe because
    // the delta-LUT add already quantises to integer 8-bit pixel units,
    // so the FP16 ULP error (~6e-5 in [0, 1]) cannot accumulate as
    // temporal drift in m_framePrev. Halves the per-frame scratch
    // bandwidth on Pass 2 (reads m_tmpH) and Pass 3 (reads m_tmpHV and
    // reads / writes m_framePrev). Codebase precedent for the probe:
    // rgy_filter_anime4k.cpp:584 and rgy_filter_denoise_fft3d.cpp:370.
    m_fp16Scratch =
        RGYOpenCLDevice(m_cl->queue().devid()).checkExtension("cl_khr_fp16");
    if (m_fp16Scratch) {
        AddMessage(RGY_LOG_DEBUG, _T("FP16 scratch enabled (cl_khr_fp16 available).\n"));
    } else {
        AddMessage(RGY_LOG_DEBUG, _T("FP16 scratch disabled (cl_khr_fp16 unavailable); using FP32 fallback.\n"));
    }

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamDenoiseHqdn3d>(m_param);
    if (!m_hqdn3d.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[prm->frameOut.csp]) {
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d -D LUT_RADIUS=%d -D HQDN3D_SCRATCH_FP16=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp],
            HQDN3D_LUT_RADIUS,
            m_fp16Scratch ? 1 : 0);
        m_hqdn3d.set(m_cl->buildResourceAsync(_T("RGY_FILTER_DENOISE_HQDN3D_CL"), _T("EXE_DATA"), options.c_str()));
    }

    // (Re-)precalc the four coefficient LUTs. Strengths are the four
    // user-tunable parameters; order: 0=luma_spatial, 1=luma_temporal,
    // 2=chroma_spatial, 3=chroma_temporal. The denoisePlane() caller
    // selects 0/1 for luma and 2/3 for chroma.
    const float strengths[4] = {
        prm->hqdn3d.luma_spatial,
        prm->hqdn3d.luma_temporal,
        prm->hqdn3d.chroma_spatial,
        prm->hqdn3d.chroma_temporal
    };
    for (int i = 0; i < 4; ++i) {
        std::vector<float> table;
        precalcCoefs(table, (double)strengths[i]);
        m_coefs[i] = m_cl->createBuffer(table.size() * sizeof(float),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, table.data());
        if (!m_coefs[i] || !m_coefs[i]->mem()) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate coefficient buffer %d.\n"), i);
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    // Allocate per-plane prev-state and the shared H / HV scratch
    // buffers. Dimensions are derived from the output csp; luma is
    // the largest plane and the scratch buffers are sized to it.
    auto err = AllocFrameBuf(prm->frameOut, 2);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to allocate memory: %s.\n"), get_err_mes(err));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // Element size for m_framePrev / m_tmpH / m_tmpHV. Pitch values
    // remain "elements per row" (unchanged); only the per-element byte
    // count shrinks when FP16 storage is in use.
    const size_t scratchElemBytes = m_fp16Scratch ? sizeof(uint16_t) : sizeof(float);
    const int planeCount = RGY_CSP_PLANES[m_frameBuf[0]->frame.csp];
    for (int i = 0; i < planeCount && i < (int)m_framePrev.size(); ++i) {
        auto planeInfo = getPlane(&m_frameBuf[0]->frame, (RGY_PLANE)i);
        const int pw = planeInfo.width;
        const int ph = planeInfo.height;
        m_framePrevPitchFloats[i] = pw;
        m_framePrev[i] = m_cl->createBuffer((size_t)pw * ph * scratchElemBytes, CL_MEM_READ_WRITE);
        if (!m_framePrev[i] || !m_framePrev[i]->mem()) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate prev buffer plane %d.\n"), i);
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    {
        auto lumaInfo = getPlane(&m_frameBuf[0]->frame, RGY_PLANE_Y);
        m_tmpPitchFloats = lumaInfo.width;
        const size_t tmpBytes = (size_t)lumaInfo.width * lumaInfo.height * scratchElemBytes;
        m_tmpH  = m_cl->createBuffer(tmpBytes, CL_MEM_READ_WRITE);
        m_tmpHV = m_cl->createBuffer(tmpBytes, CL_MEM_READ_WRITE);
        if (!m_tmpH || !m_tmpHV) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate scratch buffers.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    m_firstFrame = true;

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterDenoiseHqdn3d::denoisePlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
    RGYCLBuf *pCoefSpatial, RGYCLBuf *pCoefTemporal,
    RGYCLBuf *pPrev, int prevPitchFloats,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!m_hqdn3d.get()) {
        return RGY_ERR_OPENCL_CRUSH;
    }
    const int w = pOutputPlane->width;
    const int h = pOutputPlane->height;

    // Pass 1: horizontal IIR. One work-item per row.
    {
        const char *kernel_name = "kernel_hqdn3d_h";
        RGYWorkSize local(HQDN3D_BLOCK_LINEAR, 1);
        RGYWorkSize global(h, 1);
        auto err = m_hqdn3d.get()->kernel(kernel_name).config(queue, local, global, wait_events, nullptr).launch(
            m_tmpH->mem(), m_tmpPitchFloats,
            (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
            w, h,
            pCoefSpatial->mem());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    // Pass 2: vertical IIR. One work-item per column.
    {
        const char *kernel_name = "kernel_hqdn3d_v";
        RGYWorkSize local(HQDN3D_BLOCK_LINEAR, 1);
        RGYWorkSize global(w, 1);
        auto err = m_hqdn3d.get()->kernel(kernel_name).config(queue, local, global, {}, nullptr).launch(
            m_tmpHV->mem(), m_tmpPitchFloats,
            m_tmpH->mem(),  m_tmpPitchFloats,
            w, h,
            pCoefSpatial->mem());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    // Pass 3: temporal IIR.
    {
        const char *kernel_name = "kernel_hqdn3d_t";
        RGYWorkSize local(HQDN3D_TBLOCK_X, HQDN3D_TBLOCK_Y);
        RGYWorkSize global(w, h);
        auto err = m_hqdn3d.get()->kernel(kernel_name).config(queue, local, global, {}, event).launch(
            (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
            pPrev->mem(), prevPitchFloats,
            m_tmpHV->mem(), m_tmpPitchFloats,
            w, h,
            pCoefTemporal->mem(),
            m_firstFrame ? 1 : 0);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (denoisePlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseHqdn3d::denoiseFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const int planeCount = RGY_CSP_PLANES[pOutputFrame->csp];
    for (int i = 0; i < planeCount && i < (int)m_framePrev.size(); ++i) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame,  (RGY_PLANE)i);
        const bool isChroma = (i > 0);
        RGYCLBuf *coefSpat = m_coefs[isChroma ? 2 : 0].get();
        RGYCLBuf *coefTemp = m_coefs[isChroma ? 3 : 1].get();
        const auto &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event  = (i == planeCount - 1) ? event : nullptr;
        auto err = denoisePlane(&planeDst, &planeSrc,
            coefSpat, coefTemp,
            m_framePrev[i].get(), m_framePrevPitchFloats[i],
            queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to denoise(hqdn3d) plane %d: %s\n"), i, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseHqdn3d::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (pInputFrame->ptr[0] == nullptr) {
        return RGY_ERR_NONE;
    }
    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto &outFrame = m_frameBuf[(m_frameIdx++) % m_frameBuf.size()];
        ppOutputFrames[0] = &outFrame->frame;
    }
    auto err = denoiseFrame(ppOutputFrames[0], pInputFrame, queue, wait_events, event);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    m_firstFrame = false;
    return RGY_ERR_NONE;
}

void RGYFilterDenoiseHqdn3d::close() {
    for (auto &b : m_framePrev) b.reset();
    for (auto &c : m_coefs)     c.reset();
    m_tmpH.reset();
    m_tmpHV.reset();
    m_frameBuf.clear();
    m_hqdn3d.clear();
    m_cl.reset();
    m_frameIdx = 0;
    m_firstFrame = true;
}
