// -----------------------------------------------------------------------------------------
//     QSVEnc/VCEEnc/rkmppenc by rigaya
// -----------------------------------------------------------------------------------------
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

#pragma once
#ifndef __RGY_FILTER_RIFE_OV_H__
#define __RGY_FILTER_RIFE_OV_H__

#include "rgy_filter_cl.h"
#include "rgy_prm.h"
#include "rgy_openvino.h"
#include <vector>
#include <memory>
#include <array>

// Fresh, standalone OpenVINO-backed RIFE frame-interpolation filter. It does NOT
// reuse the older oneDNN RIFE code: the network is loaded directly from a vs-mlrt
// RIFE v4.x ONNX and run via OpenVINO (the same backend as --vpp-onnx). The only
// piece beyond a plain --vpp-onnx run is the temporal pipeline: it buffers the
// previous frame, builds the 11-channel RIFE input, runs the network once per
// interpolated time point, and emits the interpolated plus passthrough frames in
// order with correct timestamps.
//
// RIFE v4.x ONNX I/O (verified, vs-mlrt export):
//   input  [1, 11, H, W] : img0(3) + img1(3) + timestep(1) + base_grid(2) + multiplier(2)
//   output [1,  3, H, W] : one interpolated RGB frame
//   base_grid  = normalised [-1,1] backwarp mesh (horizontal, vertical)
//   multiplier = per-axis pixel-flow -> grid scale (2/(W-1), 2/(H-1))
//   H and W must be a multiple of 32.

class RGYFilterParamRifeOV : public RGYFilterParam {
public:
    tstring modelFile;        // path to a RIFE v4.x .onnx / .xml
    tstring modelDir;
    tstring device;           // OpenVINO device ("GPU.0" default)
    int     multi;            // frame-rate multiplier (>=2; 2 = double the frame rate)
    tstring colormatrix;      // auto / bt601 / bt709 / bt2020
    tstring colorrange;       // auto / tv / pc
    RGYFilterParamRifeOV() : modelFile(), modelDir(), device(_T("GPU.0")), multi(2), colormatrix(_T("auto")), colorrange(_T("auto")) {};
    virtual ~RGYFilterParamRifeOV() {};
    virtual tstring print() const override;
};

class RGYFilterRifeOV : public RGYFilter {
public:
    RGYFilterRifeOV(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterRifeOV();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    // ホスト経路で時刻tの補間フレームを推論する。
    RGY_ERR interpolate(float t);
    RGY_ERR interpolateOcl(float t, RGYOpenCLQueue &queue);
    RGY_ERR runHost(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR runOcl(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR createRgbPlanes(RGYCLBuf *parent, int channelOffset,
        std::array<std::unique_ptr<RGYCLBuf>, 3>& planes);
    RGYFrameInfo rgbFrame(const std::array<std::unique_ptr<RGYCLBuf>, 3>& planes) const;
    RGY_ERR readRgbStaging(RGYOpenCLQueue &queue, std::vector<float>& dst);
    RGY_ERR writeRgbStaging(RGYOpenCLQueue &queue, const std::vector<float>& src);

    std::unique_ptr<RGYOpenVINO> m_ov;
    int   m_W, m_H;           // working resolution (frame size; must be /32)
    int   m_multi;            // frame-rate multiplier
    bool  m_useOcl;

    // temporal state
    bool    m_havePrev;
    int64_t m_prevTimestamp;
    int64_t m_prevDuration;

    // host buffers
    std::vector<float> m_prevRGB;     // previous frame, planar RGB CHW (3*W*H)
    std::vector<float> m_currRGB;     // current  frame, planar RGB CHW (3*W*H)
    std::vector<float> m_inBuf;       // network input  (11*W*H, CHW)
    std::vector<float> m_outBuf;      // network output ( 3*W*H, CHW)
    std::vector<float> m_baseGrid;    // precomputed base_grid (2*W*H)
    std::vector<float> m_multiplier;  // precomputed multiplier (2*W*H)

    std::unique_ptr<RGYCLFrame> m_inStaging;
    std::unique_ptr<RGYCLFrame> m_outStaging;
    std::unique_ptr<RGYFilterCspCrop> m_cropToRgb;
    std::unique_ptr<RGYFilterCspCrop> m_cropFromRgb;

    std::unique_ptr<RGYCLBuf> m_inBufCL;
    std::unique_ptr<RGYCLBuf> m_outBufCL;
    std::array<std::unique_ptr<RGYCLBuf>, 3> m_prevRgbPlanes;
    std::array<std::unique_ptr<RGYCLBuf>, 3> m_currRgbPlanes;
    std::array<std::unique_ptr<RGYCLBuf>, 3> m_outRgbPlanes;
};

#endif //__RGY_FILTER_RIFE_OV_H__
