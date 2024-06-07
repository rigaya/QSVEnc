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

#ifndef __RGY_FILTER_DENOISE_FFT3D_H__
#define __RGY_FILTER_DENOISE_FFT3D_H__

#include "rgy_filter_cl.h"
#include "rgy_prm.h"

class RGYFilterParamDenoiseFFT3D : public RGYFilterParam {
public:
    VppDenoiseFFT3D fft3d;
    RGYFilterParamDenoiseFFT3D() : fft3d() {};
    virtual ~RGYFilterParamDenoiseFFT3D() {};
    virtual tstring print() const;
};

class RGYFilterDenoiseFFT3DBuffer {
public:
    RGYFilterDenoiseFFT3DBuffer(shared_ptr<RGYOpenCLContext> context) : m_cl(context), m_bufFFT() {};
    ~RGYFilterDenoiseFFT3DBuffer() {};
    RGY_ERR alloc(int width, int height, RGY_CSP csp, int frames);
    RGYCLFrame *get(const int index) { return m_bufFFT[index % m_bufFFT.size()].get(); }
    void clear() { m_bufFFT.clear(); }
protected:
    shared_ptr<RGYOpenCLContext> m_cl;
    std::vector<std::unique_ptr<RGYCLFrame>> m_bufFFT;
};

class RGYFilterDenoiseFFT3D : public RGYFilter {
public:
    RGYFilterDenoiseFFT3D(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterDenoiseFFT3D();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    RGY_ERR checkParam(const RGYFilterParamDenoiseFFT3D *prm);

    RGY_ERR denoiseFFT(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR denoiseTFFTFilterIFFT(RGYFrameInfo *pOutputFrame,
        const RGYFrameInfo *pInputFrameA, const RGYFrameInfo *pInputFrameB, const RGYFrameInfo *pInputFrameC, const RGYFrameInfo *pInputFrameD,
        RGYOpenCLQueue &queue);
    RGY_ERR denoiseMerge(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, RGYOpenCLEvent *event);

    int m_frameIdx;
    int m_bufIdx;
    int m_ov1;
    int m_ov2;

    RGYFilterDenoiseFFT3DBuffer m_bufFFT;
    std::unique_ptr<RGYCLFrame> m_filteredBlocks;
    std::unique_ptr<RGYCLBuf> m_windowBuf;
    std::unique_ptr<RGYCLBuf> m_windowBufInverse;
    RGYOpenCLProgramAsync m_fft3d;
};

#endif //__RGY_FILTER_DENOISE_FFT3D_H__

