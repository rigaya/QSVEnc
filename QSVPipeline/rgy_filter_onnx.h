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
// IABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_FILTER_ONNX_H__
#define __RGY_FILTER_ONNX_H__

#include "rgy_filter_cl.h"
#include "rgy_prm.h"
#include "rgy_openvino.h"
#include <vector>
#include <memory>
#include <deque>
#include <array>

class RGYFilterResize; // opt-in end-of-chain resize sub-filter (out_res=/resize=)

class RGYFilterParamOnnx : public RGYFilterParam {
public:
    VppOnnx onnx;
    tstring modelDir;
    int sar[2] = { 0, 0 };  // input SAR (set by pipeline) -- resolves a negative out_res= (auto-aspect) DAR-correctly
    RGYFilterParamOnnx() : onnx(), modelDir() {};
    virtual ~RGYFilterParamOnnx() {};
    virtual tstring print() const override;
};

// The pre/post a model needs, inferred from its input/output channel count.
// OpenVINO runs the network for any architecture with no per-model code; this
// enum only selects how pixels are packed into / unpacked from the tensor.
enum class OnnxIO {
    LumaSR,     // in1  -> out1 : Y plane through the net, integer scale, chroma bilinear-resampled (ArtCNN, vgg7-Y)
    GrayNoise,  // in2  -> out1 : [Y, sigma] through the net, scale=1, chroma copied (DRUNet gray)
    Chroma,     // in3  -> out2 : [Y, Cb, Cr] -> refined [Cb, Cr], scale=1, luma copied (ArtCNN Chroma)
    RGB,        // in3  -> out3 : YUV<->RGB bookend, integer scale (Real-ESRGAN / waifu2x / Real-CUGAN / BSRGAN / ArtCNN RGB)
    RGBNoise,   // in4  -> out3 : [R, G, B, sigma] -> RGB, integer scale (DPSR, DRUNet color)
};

// Standalone OpenVINO-backed CNN VPP filter. This whole class is the onnx
// experiment: one generic OpenVINO load-and-run replaces the per-network
// build*()/forward*() oneDNN graphs of the native anime4k filter. The network's
// I/O convention is inferred from its channel count (OnnxIO), so the same
// load-and-run covers every anime4k model family with no per-model code.
//
// ホスト経路はすべてのI/O形式とCPUを含む各デバイスに対応する。
// GPU共有経路は1ch輝度モデル、3ch RGBモデル、4ch RGB+Noiseモデルで自動的に使用する。
class RGYFilterOnnx : public RGYFilter {
public:
    RGYFilterOnnx(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterOnnx();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamOnnx> prm);
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    // host-readback path (correctness baseline; works on any device incl. CPU,
    // and handles every OnnxIO mode)
    RGY_ERR runHost(const RGYFrameInfo *in, RGYFrameInfo *out,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    // 1ch輝度モデルのゼロコピー経路。
    RGY_ERR runOcl(const RGYFrameInfo *in, RGYFrameInfo *out,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    // RGBとRGB+Noiseモデルのゼロコピー経路。
    RGY_ERR runOclRGB(const RGYFrameInfo *in, RGYFrameInfo *out,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR createRgbPlanes(RGYCLBuf *parent, int width, int height,
        std::array<std::unique_ptr<RGYCLBuf>, 3>& planes);
    RGYFrameInfo rgbFrame(const std::array<std::unique_ptr<RGYCLBuf>, 3>& planes,
        int width, int height) const;

    // host pre/post, dispatched on m_io. fillInputHost packs the mapped input
    // frame into m_inBuf (inC*inW*inH, CHW); writeOutputHost unpacks m_outBuf
    // (outC*outW*outH, CHW) into the mapped output frame.
    void fillInputHost(const RGYFrameInfo &hin);
    void writeOutputHost(const RGYFrameInfo &hout, const RGYFrameInfo &hin);
    // 8/16-bit bodies of the above (TPix = uint8_t / uint16_t); the plain
    // versions dispatch on m_bitdepth.
    template<typename TPix> void fillInputHostT(const RGYFrameInfo &hin);
    template<typename TPix> void writeOutputHostT(const RGYFrameInfo &hout, const RGYFrameInfo &hin);
    // compute the YUV<->RGB matrix + range coefficients (mirrors the native
    // anime4k RGB bookend exactly).
    void setupColorCoeffs(int matrixSelIn, int matrixSelOut, bool rangeTV, int pixMax);

    // --- multi-frame temporal window (frames= > 1) ---------------------------
    // Pack one mapped input frame's YUV as planar RGB [0,1] CHW
    // (3*modelInW*modelInH floats) into dst; same maths as the RGB case of
    // fillInputHost, but writing a single frame's worth to an arbitrary offset.
    void packFrameRGB(const RGYFrameInfo &hin, float *dst);
    template<typename TPix> void packFrameRGBT(const RGYFrameInfo &hin, float *dst);
    // Temporal run path: buffer the last T RGB frames and emit a centred window
    // with edge replication (1-in-1-out with (T-1)/2 delay); flush drains the tail.
    RGY_ERR runTemporal(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    // Build the T-frame window centred on outIdx, run the network and write the
    // result into m_frameBuf[0] (plus optional post-resize), stamped from the centre frame.
    RGY_ERR emitTemporalOutput(int64_t outIdx, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, RGYOpenCLEvent *event);

    // --- two-input inpainting mode (mask=) -----------------------------------
    // The model receives the frame and a user mask as separate tensors and
    // returns the frame with the masked region filled; only the masked pixels
    // are composited back, everything else keeps the frame's own values.
    RGY_ERR initMask(std::shared_ptr<RGYFilterParamOnnx> prm, const int inW, const int inH, const RGY_CSP inCsp);
    RGY_ERR runMask(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    std::unique_ptr<RGYOpenVINO> m_ov;
    OnnxIO m_io;                          // I/O convention inferred from channel counts
    int   m_inC, m_outC;                        // model input / output channel counts
    int   m_scale;                              // integer upscale factor from the model (out/in)
    int   m_modelInW, m_modelInH;               // tensor input size; can be padded from the frame size
    int   m_padL, m_padT;                       // edge-replication padding before packing host input
    float m_maxval;                             // (1<<bitdepth)-1
    int   m_bitdepth;                           // 8 (yv12/nv12) or 16 (yv12(16bit)/p010)
    bool  m_useOcl;                             // 初期化時に選択したゼロコピー経路を使用するか
    bool  m_ycbcr;                              // 3ch model fed as planar YCbCr instead of RGB
    float m_sigmaNorm;                          // noise sigma / 255 for the conditioning channel

    // colour coefficients (computed once at init)
    float m_yOff, m_yScale, m_yRange, m_cOff, m_cScale, m_cRange;
    float m_matVR, m_matUG, m_matVG, m_matUB;                                   // YUV -> RGB
    float m_matRY, m_matGY, m_matBY, m_matRU, m_matGU, m_matBU, m_matRV, m_matGV, m_matBV; // RGB -> YUV

    // host-readback path scratch
    std::unique_ptr<RGYCLFrame>  m_inStaging;   // host-mappable copy of the input frame
    std::unique_ptr<RGYCLFrame>  m_outStaging;  // host-mappable scratch for the output frame
    std::vector<float>           m_inBuf;       // network input tensor  (inC*inW*inH, CHW)
    std::vector<float>           m_outBuf;      // network output tensor (outC*outW*outH, CHW)
    std::vector<float>           m_u444, m_v444;// normalised chroma at output luma res (for 4:2:0 downsample)

    // ゼロコピー経路で使用するリソース
    std::unique_ptr<RGYOpenCLProgram> m_program;  // pack / unpack / chroma kernels
    std::unique_ptr<RGYCLBuf>         m_inBufCL;  // f32 network input  (inC*inW*inH)
    std::unique_ptr<RGYCLBuf>         m_outBufCL; // f32 network output (outC*outW*outH)
    std::array<std::unique_ptr<RGYCLBuf>, 3> m_inRgbPlanes;
    std::array<std::unique_ptr<RGYCLBuf>, 3> m_outRgbPlanes;
    std::unique_ptr<RGYFilterCspCrop> m_cropToRgb;
    std::unique_ptr<RGYFilterCspCrop> m_cropFromRgb;

    // --- multi-frame temporal window state (only used when m_temporalT > 1) ---
    int m_temporalT;                 // input frames per window (1 = single-frame path, disabled)
    struct RingFrame {
        std::vector<float> rgb;      // one frame, planar RGB [0,1] CHW (3*modelInW*modelInH)
        int64_t         timestamp;
        int64_t         duration;
        RGY_PICSTRUCT   picstruct;
        RGY_FRAME_FLAGS flags;
        int             inputFrameId;
    };
    std::deque<RingFrame> m_ring;    // most recent up-to-T frames (oldest at front)
    int64_t m_ringBaseIdx;           // global frame index of m_ring.front()
    int64_t m_recvCount;             // real frames received so far
    int64_t m_emitCount;             // outputs emitted so far

    // --- two-input inpainting mode state (only used when mask= is set) --------
    std::unique_ptr<RGYOpenVINOMultiIO> m_ovm; // multi-port backend (replaces m_ov)
    int m_maskModelW, m_maskModelH;  // the model's own (static) spatial size
    int m_imgPortIdx, m_mskPortIdx;  // input port order of the frame / mask tensors
    float m_outScale;                // model output range: 1.0 = [0,1], 1/255 = [0,255] (settled on the first frame; 0 = not yet)
    std::vector<float> m_maskFrame;  // mask at frame resolution, 0/1 (composite selector)
    std::vector<float> m_maskModel;  // mask at model resolution, 0/1 (network input)
    std::vector<float> m_frameRGB;   // current frame, planar RGB [0,1] CHW at frame resolution
    std::vector<float> m_modelIn;    // network image input (3*maskModelW*maskModelH)
    std::vector<float> m_modelOut;   // network output      (3*maskModelW*maskModelH)

    // opt-in end-of-chain resize (out_res=): runs after the network core, fitting
    // the integer-scaled output to the requested final resolution. Reuses the
    // shared resampler family; null when out_res= is not used.
    std::unique_ptr<RGYFilterResize>  m_postResize;
};

#endif //__RGY_FILTER_ONNX_H__
