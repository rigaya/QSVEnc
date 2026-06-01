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

#ifndef __RGY_FILTER_DENOISE_HQDN3D_H__
#define __RGY_FILTER_DENOISE_HQDN3D_H__

#include "rgy_filter_cl.h"
#include "rgy_prm.h"
#include <array>

// LUT bin count per side. coef[i] indexed by signed delta in
// [-LUT_RADIUS, +LUT_RADIUS); table size = 2 * LUT_RADIUS entries.
// 256 covers the full 8-bit delta range; HBD inputs are mapped to
// the same range by left-shifting (16-bit canonical), so 256 is
// sufficient for all supported bit depths.
static const int HQDN3D_LUT_RADIUS = 256;

class RGYFilterParamDenoiseHqdn3d : public RGYFilterParam {
public:
    VppHqdn3d hqdn3d;
    RGYFilterParamDenoiseHqdn3d() : hqdn3d() {};
    virtual ~RGYFilterParamDenoiseHqdn3d() {};
    virtual tstring print() const override { return hqdn3d.print(); };
};

class RGYFilterDenoiseHqdn3d : public RGYFilter {
public:
    RGYFilterDenoiseHqdn3d(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterDenoiseHqdn3d();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    RGY_ERR denoisePlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
        RGYCLBuf *pCoefSpatial, RGYCLBuf *pCoefTemporal,
        RGYCLBuf *pPrev, int prevPitchFloats,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR denoiseFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    static void precalcCoefs(std::vector<float> &table, double dist25);

    RGYOpenCLProgramAsync m_hqdn3d;
    // Coefficient LUTs (one per plane-direction combination): luma-spatial,
    // luma-temporal, chroma-spatial, chroma-temporal. Each table holds
    // 2 * HQDN3D_LUT_RADIUS float entries.
    std::array<std::unique_ptr<RGYCLBuf>, 4> m_coefs;
    // Per-plane previous-frame state (float full-plane buffers) and the
    // intermediate H / HV scratch buffers. Allocated lazily on first
    // run() so we have the post-init frame dimensions.
    // Per-plane float prev-state buffers (luma + 2 chroma). Allocated
    // at init() once the output csp/dimensions are known. Each holds
    // plane_width * plane_height float entries.
    std::array<std::unique_ptr<RGYCLBuf>, 3> m_framePrev;
    std::array<int, 3> m_framePrevPitchFloats;
    // Intermediate H / HV float scratch buffers. Sized to the luma
    // plane and reused for chroma (which is smaller).
    std::unique_ptr<RGYCLBuf> m_tmpH;
    std::unique_ptr<RGYCLBuf> m_tmpHV;
    int m_tmpPitchFloats;
    int m_frameIdx;
    bool m_firstFrame;
    // True when the device advertises cl_khr_fp16 and m_tmpH / m_tmpHV /
    // m_framePrev[] use 16-bit half storage instead of 32-bit float. The
    // kernel arithmetic stays float; only the buffer reads / writes go
    // through vload_half / vstore_half. Safe because the delta-LUT add
    // already quantises to integer 8-bit pixel units, so FP16 ULP error
    // (~6e-5 in [0, 1]) cannot accumulate temporal drift.
    bool m_fp16Scratch;
};

#endif //__RGY_FILTER_DENOISE_HQDN3D_H__
