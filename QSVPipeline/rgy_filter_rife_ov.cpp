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

#include "rgy_filter_rife_ov.h"
#include "rgy_filesystem.h"
#include <algorithm>
#include <cmath>

static inline uint8_t clamp_u8(int v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }
static inline float   clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

RGYFilterRifeOV::RGYFilterRifeOV(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context), m_ov(), m_W(0), m_H(0), m_multi(2), m_maxval(255.0f),
    m_yOff(0), m_yScale(1), m_yRange(255), m_cOff(128), m_cScale(1), m_cRange(255),
    m_matVR(0), m_matUG(0), m_matVG(0), m_matUB(0),
    m_matRY(0), m_matGY(0), m_matBY(0), m_matRU(0), m_matGU(0), m_matBU(0), m_matRV(0), m_matGV(0), m_matBV(0),
    m_havePrev(false), m_prevTimestamp(0), m_prevDuration(0),
    m_prevRGB(), m_currRGB(), m_inBuf(), m_outBuf(), m_baseGrid(), m_multiplier(),
    m_inStaging(), m_outStaging() {
    m_name = _T("rife-ov");
}

RGYFilterRifeOV::~RGYFilterRifeOV() { close(); }

void RGYFilterRifeOV::close() {
    m_ov.reset();
    m_inStaging.reset();
    m_outStaging.reset();
    m_frameBuf.clear();
}

tstring RGYFilterParamRifeOV::print() const {
    return strsprintf(_T("rife-ov: %s, x%d, device %s"), modelFile.c_str(), multi, device.c_str());
}

void RGYFilterRifeOV::setupColorCoeffs(int matrixSel, bool rangeTV, int pixMax) {
    float Kr = 0.2126f, Kb = 0.0722f; // BT.709
    if (matrixSel == 601)  { Kr = 0.299f;  Kb = 0.114f; }
    if (matrixSel == 2020) { Kr = 0.2627f; Kb = 0.0593f; }
    const float Kg = 1.0f - Kr - Kb;
    m_matVR = 2.0f * (1.0f - Kr);
    m_matUG = -2.0f * Kb * (1.0f - Kb) / Kg;
    m_matVG = -2.0f * Kr * (1.0f - Kr) / Kg;
    m_matUB = 2.0f * (1.0f - Kb);
    m_matRY = Kr;                            m_matGY = Kg;                            m_matBY = Kb;
    m_matRU = -Kr / (2.0f * (1.0f - Kb));    m_matGU = -Kg / (2.0f * (1.0f - Kb));    m_matBU = 0.5f;
    m_matRV = 0.5f;                          m_matGV = -Kg / (2.0f * (1.0f - Kr));    m_matBV = -Kb / (2.0f * (1.0f - Kr));
    m_yOff   = rangeTV ? (16.0f  * pixMax / 255.0f) : 0.0f;
    m_yRange = rangeTV ? (219.0f * pixMax / 255.0f) : (float)pixMax;
    m_yScale = 1.0f / m_yRange;
    m_cOff   = rangeTV ? (128.0f * pixMax / 255.0f) : ((float)pixMax / 2.0f);
    m_cRange = rangeTV ? (224.0f * pixMax / 255.0f) : (float)pixMax;
    m_cScale = 1.0f / m_cRange;
}

RGY_ERR RGYFilterRifeOV::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamRifeOV>(pParam);
    if (!prm) { AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n")); return RGY_ERR_INVALID_PARAM; }
    if (!RGYOpenVINO::available()) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: this build was compiled without OpenVINO support.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->modelFile.empty() || !rgy_file_exists(prm->modelFile)) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: model= (path to a RIFE .onnx) is required and must exist.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->multi < 2) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: multi must be >= 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto inCsp = prm->frameIn.csp;
    if ((inCsp != RGY_CSP_YV12 && inCsp != RGY_CSP_NV12) || prm->frameIn.bitdepth != 8) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: supports 8-bit yuv420 (yv12/nv12) only; got %s %dbit.\n"),
            RGY_CSP_NAMES[inCsp], prm->frameIn.bitdepth);
        return RGY_ERR_UNSUPPORTED;
    }
    m_W = prm->frameIn.width;
    m_H = prm->frameIn.height;
    if ((m_W % 32) != 0 || (m_H % 32) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: RIFE requires width/height a multiple of 32 (got %dx%d). "
            "Pad/crop the input first (e.g. --vpp-pad / --crop).\n"), m_W, m_H);
        return RGY_ERR_UNSUPPORTED;
    }
    m_multi  = prm->multi;
    m_maxval = (float)((1 << prm->frameIn.bitdepth) - 1);

    // load + compile the RIFE ONNX (input reshaped to [1,11,H,W]).
    m_ov = std::make_unique<RGYOpenVINO>();
    std::string errMsg;
    const std::string modelPathA = tchar_to_string(prm->modelFile);
    const std::string deviceA    = tchar_to_string(prm->device);
    int peekIn = 0, peekOut = 0;
    RGY_ERR err = m_ov->peekChannels(modelPathA, peekIn, peekOut, errMsg);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: failed to read model %s: %s\n"), prm->modelFile.c_str(), char_to_tstring(errMsg).c_str());
        return err;
    }
    err = m_ov->init(modelPathA, deviceA, m_H, m_W, errMsg);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: failed to load/compile model on %s: %s\n"), prm->device.c_str(), char_to_tstring(errMsg).c_str());
        return err;
    }
    if (m_ov->inChannels() != 11 || m_ov->outChannels() != 3) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: not a RIFE model (expected 11ch in / 3ch out, got %dch / %dch).\n"),
            m_ov->inChannels(), m_ov->outChannels());
        return RGY_ERR_UNSUPPORTED;
    }

    // colour matrix + range (auto: BT.601 for SD, BT.709 for HD; TV range).
    int matrixSel;
    if      (prm->colormatrix == _T("bt601"))  matrixSel = 601;
    else if (prm->colormatrix == _T("bt2020")) matrixSel = 2020;
    else if (prm->colormatrix == _T("bt709"))  matrixSel = 709;
    else                                       matrixSel = (m_H <= 576) ? 601 : 709;
    const bool rangeTV = (prm->colorrange != _T("pc"));
    setupColorCoeffs(matrixSel, rangeTV, 255);

    // precompute base_grid (normalised [-1,1] mesh) and multiplier (2/(W-1), 2/(H-1)).
    const size_t plane = (size_t)m_W * m_H;
    m_baseGrid.resize(2 * plane);
    m_multiplier.resize(2 * plane);
    const float multH = 2.0f / (float)(m_W - 1);
    const float multV = 2.0f / (float)(m_H - 1);
    for (int y = 0; y < m_H; y++) {
        const float vy = (m_H > 1) ? (-1.0f + 2.0f * (float)y / (float)(m_H - 1)) : 0.0f;
        for (int x = 0; x < m_W; x++) {
            const float vx = (m_W > 1) ? (-1.0f + 2.0f * (float)x / (float)(m_W - 1)) : 0.0f;
            const size_t idx = (size_t)y * m_W + x;
            m_baseGrid[idx]           = vx;          // ch0: horizontal
            m_baseGrid[plane + idx]   = vy;          // ch1: vertical
            m_multiplier[idx]         = multH;       // ch0
            m_multiplier[plane + idx] = multV;       // ch1
        }
    }

    // host buffers
    m_prevRGB.resize(3 * plane);
    m_currRGB.resize(3 * plane);
    m_inBuf.resize(11 * plane);
    m_outBuf.resize(3 * plane);

    // output frame info: same resolution, frame rate multiplied by `multi`.
    auto frameOut = prm->frameOut;
    frameOut.csp    = inCsp;
    frameOut.width  = m_W;
    frameOut.height = m_H;
    prm->frameOut   = frameOut;

    // Multi-out filter (1-in / multi-out): the framework's auto path-through for
    // timestamp / picstruct / flags only works for 1-in/1-out, so clear those bits;
    // run_filter stamps timestamp / duration / picstruct / inputFrameId per output.
    m_pathThrough = (FILTER_PATHTHROUGH_FRAMEINFO)(m_pathThrough &
        (~(uint32_t)(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS)));

    prm->baseFps   *= m_multi;   // interpolated output runs at multi x the input rate

    // pool: up to `multi` output frames per input frame.
    err = AllocFrameBuf(prm->frameOut, m_multi);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: failed to allocate output frame buffer: %s.\n"), get_err_mes(err));
        return err;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    m_inStaging  = m_cl->createFrameBuffer(m_W, m_H, inCsp, prm->frameIn.bitdepth, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
    m_outStaging = m_cl->createFrameBuffer(m_W, m_H, inCsp, prm->frameIn.bitdepth, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
    if (!m_inStaging || !m_outStaging) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: failed to allocate staging frame buffers.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }

    m_havePrev = false;
    m_param = prm;
    AddMessage(RGY_LOG_DEBUG, _T("rife-ov: %s, %dx%d, x%d, device %s.\n"),
        prm->modelFile.c_str(), m_W, m_H, m_multi, prm->device.c_str());
    return RGY_ERR_NONE;
}

// YUV (yv12/nv12 8-bit 4:2:0) -> planar RGB [0,1] CHW (3*W*H). Chroma bilinear-upsampled.
void RGYFilterRifeOV::yuvToRGB(const RGYFrameInfo &hin, float *dst) {
    const int W = m_W, H = m_H;
    const size_t plane = (size_t)W * H;
    const bool nv12 = (hin.csp == RGY_CSP_NV12);
    const int cw = W / 2, ch = H / 2;
    const uint8_t *pU = hin.ptr[1];
    const uint8_t *pV = nv12 ? (hin.ptr[1] + 1) : hin.ptr[2];
    const int cStride = nv12 ? 2 : 1;
    const int cPitchU = hin.pitch[1];
    const int cPitchV = nv12 ? hin.pitch[1] : hin.pitch[2];
    float *R = dst, *G = dst + plane, *B = dst + 2 * plane;
    for (int y = 0; y < H; y++) {
        const uint8_t *yrow = hin.ptr[0] + (size_t)y * hin.pitch[0];
        const int cy = std::min(y / 2, ch - 1);
        for (int x = 0; x < W; x++) {
            const int cx = std::min(x / 2, cw - 1);
            const float yn = ((float)yrow[x] - m_yOff) * m_yScale;
            const float un = ((float)pU[(size_t)cy * cPitchU + (size_t)cx * cStride] - m_cOff) * m_cScale;
            const float vn = ((float)pV[(size_t)cy * cPitchV + (size_t)cx * cStride] - m_cOff) * m_cScale;
            const size_t i = (size_t)y * W + x;
            R[i] = clampf(yn + m_matVR * vn, 0.0f, 1.0f);
            G[i] = clampf(yn + m_matUG * un + m_matVG * vn, 0.0f, 1.0f);
            B[i] = clampf(yn + m_matUB * un, 0.0f, 1.0f);
        }
    }
}

// planar RGB [0,1] CHW (3*W*H) -> yv12/nv12 8-bit into the mapped output frame.
void RGYFilterRifeOV::rgbToYUV(const RGYFrameInfo &hout, const float *src) {
    const int W = m_W, H = m_H;
    const size_t plane = (size_t)W * H;
    const bool nv12 = (hout.csp == RGY_CSP_NV12);
    const int cw = W / 2, chh = H / 2;
    const float *R = src, *G = src + plane, *B = src + 2 * plane;
    uint8_t *oU = hout.ptr[1];
    uint8_t *oV = nv12 ? (hout.ptr[1] + 1) : hout.ptr[2];
    const int oStride = nv12 ? 2 : 1;
    const int oPitchU = hout.pitch[1];
    const int oPitchV = nv12 ? hout.pitch[1] : hout.pitch[2];
    // luma + accumulate chroma at full res, then 4:2:0 box-average.
    for (int y = 0; y < H; y++) {
        uint8_t *yd = hout.ptr[0] + (size_t)y * hout.pitch[0];
        for (int x = 0; x < W; x++) {
            const size_t i = (size_t)y * W + x;
            const float r = R[i], g = G[i], b = B[i];
            const float Yn = m_matRY * r + m_matGY * g + m_matBY * b;
            yd[x] = clamp_u8((int)(Yn * m_yRange + m_yOff + 0.5f));
        }
    }
    for (int cy = 0; cy < chh; cy++) {
        for (int cx = 0; cx < cw; cx++) {
            float u = 0.0f, v = 0.0f;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    const size_t i = (size_t)(cy * 2 + dy) * W + (cx * 2 + dx);
                    const float r = R[i], g = G[i], b = B[i];
                    u += m_matRU * r + m_matGU * g + m_matBU * b;
                    v += m_matRV * r + m_matGV * g + m_matBV * b;
                }
            }
            u *= 0.25f; v *= 0.25f;
            oU[(size_t)cy * oPitchU + (size_t)cx * oStride] = clamp_u8((int)(u * m_cRange + m_cOff + 0.5f));
            oV[(size_t)cy * oPitchV + (size_t)cx * oStride] = clamp_u8((int)(v * m_cRange + m_cOff + 0.5f));
        }
    }
}

// build the 11-channel input for time t and run the network -> m_outBuf.
RGY_ERR RGYFilterRifeOV::interpolate(float t) {
    const size_t plane = (size_t)m_W * m_H;
    float *p = m_inBuf.data();
    memcpy(p + 0 * plane, m_prevRGB.data(), 3 * plane * sizeof(float)); // img0 (3)
    memcpy(p + 3 * plane, m_currRGB.data(), 3 * plane * sizeof(float)); // img1 (3)
    std::fill(p + 6 * plane, p + 7 * plane, t);                         // timestep (1)
    memcpy(p + 7 * plane, m_baseGrid.data(), 2 * plane * sizeof(float));// base_grid (2)
    memcpy(p + 9 * plane, m_multiplier.data(), 2 * plane * sizeof(float)); // multiplier (2)
    return m_ov->infer(m_inBuf.data(), m_outBuf.data());
}

RGY_ERR RGYFilterRifeOV::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (pInputFrame->ptr[0] == nullptr) { *pOutputFrameNum = 0; return RGY_ERR_NONE; } // flush: drop trailing single frame

    // copy input -> host-mappable staging, map, convert to currRGB.
    auto err = m_cl->copyFrame(&m_inStaging->frame, pInputFrame, nullptr, queue, wait_events, nullptr);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("rife-ov: copy input to staging failed: %s.\n"), get_err_mes(err)); return err; }
    err = m_inStaging->queueMapBuffer(queue, CL_MAP_READ, {}, RGY_CL_MAP_BLOCK_ALL);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("rife-ov: map input staging failed: %s.\n"), get_err_mes(err)); return err; }
    yuvToRGB(m_inStaging->mappedHost()->host(), m_currRGB.data());
    m_inStaging->unmapBuffer(queue);

    if (!m_havePrev) {
        // first frame: emit it unchanged; it becomes the previous frame.
        ppOutputFrames[0] = &m_frameBuf[0]->frame;
        err = m_cl->copyFrame(ppOutputFrames[0], pInputFrame, nullptr, queue, {}, event);
        if (err != RGY_ERR_NONE) return err;
        ppOutputFrames[0]->timestamp = pInputFrame->timestamp;
        ppOutputFrames[0]->duration  = pInputFrame->duration;
        ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
        ppOutputFrames[0]->inputFrameId = pInputFrame->inputFrameId;
        *pOutputFrameNum = 1;
        m_prevRGB = m_currRGB;
        m_prevTimestamp = pInputFrame->timestamp;
        m_prevDuration  = pInputFrame->duration;
        m_havePrev = true;
        return RGY_ERR_NONE;
    }

    const int64_t spanDur = pInputFrame->timestamp - m_prevTimestamp;
    // (multi-1) interpolated frames between prev and curr.
    for (int k = 1; k < m_multi; k++) {
        const float t = (float)k / (float)m_multi;
        err = interpolate(t);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("rife-ov: inference failed at t=%.3f.\n"), t); return err; }
        err = m_outStaging->queueMapBuffer(queue, CL_MAP_WRITE, {}, RGY_CL_MAP_BLOCK_ALL);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("rife-ov: map output staging failed: %s.\n"), get_err_mes(err)); return err; }
        rgbToYUV(m_outStaging->mappedHost()->host(), m_outBuf.data());
        m_outStaging->unmapBuffer(queue);
        RGYFrameInfo *out = &m_frameBuf[k - 1]->frame;
        err = m_cl->copyFrame(out, &m_outStaging->frame, nullptr, queue, {}, nullptr);
        if (err != RGY_ERR_NONE) return err;
        out->timestamp = m_prevTimestamp + (spanDur > 0 ? spanDur * (int64_t)k / (int64_t)m_multi : 0);
        out->duration  = (spanDur > 0) ? (spanDur / m_multi) : pInputFrame->duration;
        out->picstruct = pInputFrame->picstruct;
        out->inputFrameId = pInputFrame->inputFrameId;
        ppOutputFrames[k - 1] = out;
    }
    // passthrough of the current frame (copied unchanged, no RGB round-trip).
    RGYFrameInfo *passthru = &m_frameBuf[m_multi - 1]->frame;
    err = m_cl->copyFrame(passthru, pInputFrame, nullptr, queue, {}, event);
    if (err != RGY_ERR_NONE) return err;
    passthru->timestamp = pInputFrame->timestamp;
    passthru->duration  = (spanDur > 0) ? (spanDur / m_multi) : pInputFrame->duration;
    passthru->picstruct = pInputFrame->picstruct;
    passthru->inputFrameId = pInputFrame->inputFrameId;
    ppOutputFrames[m_multi - 1] = passthru;

    *pOutputFrameNum = m_multi;
    m_prevRGB.swap(m_currRGB);
    m_prevTimestamp = pInputFrame->timestamp;
    m_prevDuration  = pInputFrame->duration;
    return RGY_ERR_NONE;
}
