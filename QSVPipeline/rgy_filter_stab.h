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
// Camera-shake stabilisation -- Phase Correlation
// (Kuglin & Hines 1975, "The phase correlation image alignment method",
//  Proc. Int. Conf. on Cybernetics and Society).
//
// Hand-rolled OpenCL FFT on a downsampled luma plane. Same shape as
// rigaya's rgy_filter_denoise_fft3d.cl, but a whole-frame radix-2 2D
// FFT at FFT_N x FFT_N instead of small tiled blocks.

#pragma once
#ifndef __RGY_FILTER_STAB_H__
#define __RGY_FILTER_STAB_H__

#include <vector>
#include "rgy_filter_cl.h"
#include "rgy_prm.h"

class RGYFilterParamStab : public RGYFilterParam {
public:
    VppStab stab;

    RGYFilterParamStab() : stab() {};
    virtual ~RGYFilterParamStab() {};
    virtual tstring print() const override { return stab.print(); };
};

class RGYFilterStab : public RGYFilter {
public:
    RGYFilterStab(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterStab();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
                               RGYOpenCLQueue &queue_main, const std::vector<RGYOpenCLEvent> &wait_events,
                               RGYOpenCLEvent *event) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamStab> pParam);

    // Lazily compiled program: holds all four stab kernels.
    RGYOpenCLProgramAsync m_stab;
    std::string           m_buildOptions;

    // Buffers (all sized to FFT_N x FFT_N).
    // m_srcReal      : downsampled luma, real, float
    // m_curFreq      : forward-FFT of the current frame, complex (float2)
    // m_prevFreq     : forward-FFT of the previous frame, complex (float2)
    // m_corrFreq     : normalised cross-spectrum, complex (float2)
    // m_corrReal     : inverse-FFT result (real part), float -- the
    //                  correlation plane that gets read back to host
    std::unique_ptr<RGYCLBuf> m_srcReal;
    std::unique_ptr<RGYCLBuf> m_curFreq;
    std::unique_ptr<RGYCLBuf> m_prevFreq;
    std::unique_ptr<RGYCLBuf> m_corrFreq;
    std::unique_ptr<RGYCLBuf> m_corrReal;

    // Host staging for the IFFT correlation plane and the peak scan.
    std::vector<float> m_corrHost;

    // True once m_prevFreq holds a usable spectrum from the previous
    // frame. False on the first frame after init.
    bool m_havePrev;

    // Smoothing state.
    float m_smoothShiftX;
    float m_smoothShiftY;
    bool  m_haveSmoothing;

    // Diagnostics.
    int m_lowTrustFrames;
};

#endif // __RGY_FILTER_STAB_H__
