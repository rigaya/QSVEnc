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
#include "rgy_model_registry.h"
#include <algorithm>
#include <cstring>

RGYFilterRifeOV::RGYFilterRifeOV(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context), m_ov(), m_W(0), m_H(0), m_multi(2), m_useOcl(false),
    m_havePrev(false), m_prevTimestamp(0), m_prevDuration(0),
    m_prevRGB(), m_currRGB(), m_inBuf(), m_outBuf(), m_baseGrid(), m_multiplier(),
    m_inStaging(), m_outStaging(), m_cropToRgb(), m_cropFromRgb(),
    m_inBufCL(), m_outBufCL(), m_prevRgbPlanes(), m_currRgbPlanes(), m_outRgbPlanes() {
    m_name = _T("rife-ov");
}

RGYFilterRifeOV::~RGYFilterRifeOV() { close(); }

void RGYFilterRifeOV::close() {
    m_cropToRgb.reset();
    m_cropFromRgb.reset();
    for (auto& plane : m_prevRgbPlanes) plane.reset();
    for (auto& plane : m_currRgbPlanes) plane.reset();
    for (auto& plane : m_outRgbPlanes) plane.reset();
    m_inBufCL.reset();
    m_outBufCL.reset();
    m_inStaging.reset();
    m_outStaging.reset();
    m_ov.reset();
    m_frameBuf.clear();
    m_havePrev = false;
    m_useOcl = false;
}

tstring RGYFilterParamRifeOV::print() const {
    return strsprintf(_T("rife-ov: %s, x%d, device %s"), modelFile.c_str(), multi, device.c_str());
}

RGY_ERR RGYFilterRifeOV::createRgbPlanes(RGYCLBuf *parent, const int channelOffset,
    std::array<std::unique_ptr<RGYCLBuf>, 3>& planes) {
    const size_t planeBytes = (size_t)m_W * m_H * sizeof(float);
    for (int i = 0; i < 3; i++) {
        cl_buffer_region region = { (size_t)(channelOffset + i) * planeBytes, planeBytes };
        cl_int clerr = CL_SUCCESS;
        auto subbuf = clCreateSubBuffer(parent->mem(), CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &clerr);
        if (clerr != CL_SUCCESS || subbuf == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("rife-ov: RGBテンソル平面%dの作成に失敗しました: %s。\n"), i, cl_errmes(clerr));
            return (clerr == CL_SUCCESS) ? RGY_ERR_MEMORY_ALLOC : err_cl_to_rgy(clerr);
        }
        planes[i] = std::make_unique<RGYCLBuf>(subbuf, CL_MEM_READ_WRITE, planeBytes);
    }
    return RGY_ERR_NONE;
}

RGYFrameInfo RGYFilterRifeOV::rgbFrame(const std::array<std::unique_ptr<RGYCLBuf>, 3>& planes) const {
    RGYFrameInfo frame;
    frame.width = m_W;
    frame.height = m_H;
    frame.csp = RGY_CSP_RGB_F32;
    frame.bitdepth = 32;
    frame.mem_type = RGY_MEM_TYPE_GPU;
    frame.picstruct = RGY_PICSTRUCT_FRAME;
    for (int i = 0; i < 3; i++) {
        frame.ptr[i] = (uint8_t *)planes[i]->mem();
        frame.pitch[i] = m_W * sizeof(float);
    }
    return frame;
}

RGY_ERR RGYFilterRifeOV::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamRifeOV>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: パラメータ型が不正です。\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!RGYOpenVINO::available()) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: OpenVINOが有効でないビルドです。\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->modelFile.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: model=に登録済みモデル名またはRIFEモデルのパスが必要です。\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->modelFile.find_first_of(_T("/\\\\.")) == tstring::npos && !prm->modelDir.empty()) {
        RGYModelRegistry registry;
        const auto err = registry.load(PathCombineS(prm->modelDir, _T("rife_ov_models.json")), m_pLog);
        if (err != RGY_ERR_NONE) return err;
        if (!registry.find(prm->modelFile)) {
            AddMessage(RGY_LOG_ERROR, _T("rife-ov: rife_ov_models.jsonにモデル\"%s\"がありません。\n"), prm->modelFile.c_str());
            return RGY_ERR_NOT_FOUND;
        }
        prm->modelFile = registry.resolveModelPath(prm->modelFile);
    }
    if (!rgy_file_exists(prm->modelFile)) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: モデルファイルがありません: %s\n"), prm->modelFile.c_str());
        return RGY_ERR_FILE_OPEN;
    }
    if (prm->multi < 2) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: multiは2以上である必要があります。\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto inCsp = prm->frameIn.csp;
    if ((inCsp != RGY_CSP_YV12 && inCsp != RGY_CSP_NV12 && inCsp != RGY_CSP_YV12_16 && inCsp != RGY_CSP_P010)
        || (prm->frameIn.bitdepth != 8 && prm->frameIn.bitdepth != 16)) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: 8bit yuv420（yv12/nv12）または16bit（yv12(16bit)/p010）のみ対応します: %s %dbit。\n"),
            RGY_CSP_NAMES[inCsp], prm->frameIn.bitdepth);
        return RGY_ERR_UNSUPPORTED;
    }
    m_W = prm->frameIn.width;
    m_H = prm->frameIn.height;
    if ((m_W % 32) != 0 || (m_H % 32) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: 幅と高さは32の倍数である必要があります（%dx%d）。\n"), m_W, m_H);
        return RGY_ERR_UNSUPPORTED;
    }
    m_multi = prm->multi;

    m_ov = std::make_unique<RGYOpenVINO>();
    tstring errMsg;
    int peekIn = 0, peekOut = 0;
    auto err = m_ov->peekChannels(prm->modelFile, peekIn, peekOut, errMsg);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: モデルの読み込みに失敗しました: %s。\n"), errMsg.c_str());
        return err;
    }
    if (peekIn != 11 || peekOut != 3) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: RIFEモデルではありません（入力%dch、出力%dch）。\n"), peekIn, peekOut);
        return RGY_ERR_UNSUPPORTED;
    }

    const bool deviceWantsGpu = (prm->device.substr(0, 3) == _T("GPU") || prm->device == _T("AUTO"));
    if (deviceWantsGpu && m_cl) {
        // 共有テンソルが使える場合は、同一OpenCLキュー上で変換と推論を直列化する。
        err = m_ov->initShared(prm->modelFile, (void *)m_cl->queue().get(), (void *)m_cl->context(), m_H, m_W, errMsg);
        if (err == RGY_ERR_NONE) {
            m_useOcl = true;
        } else {
            AddMessage(RGY_LOG_WARN, _T("rife-ov: OpenCL共有テンソルを利用できないためホスト経路へ切り替えます: %s。\n"), errMsg.c_str());
            // 共有テンソルAPIだけが不足する場合も、OpenVINOの実行先は選択中のOpenCL GPUを優先する。
            m_ov = std::make_unique<RGYOpenVINO>();
            errMsg.clear();
            err = m_ov->initFromOpenCLQueue(prm->modelFile, (void *)m_cl->queue().get(), (void *)m_cl->context(), m_H, m_W, errMsg);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_DEBUG, _T("rife-ov: OpenCLキューからの初期化に失敗したためdevice=%sを使用します: %s。\n"),
                    prm->device.c_str(), errMsg.c_str());
                m_ov = std::make_unique<RGYOpenVINO>();
                errMsg.clear();
                err = m_ov->init(prm->modelFile, prm->device, m_H, m_W, errMsg);
            }
        }
    } else {
        err = m_ov->init(prm->modelFile, prm->device, m_H, m_W, errMsg);
    }
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: モデルのコンパイルに失敗しました（%s）: %s。\n"), prm->device.c_str(), errMsg.c_str());
        return err;
    }
    if (m_ov->inChannels() != 11 || m_ov->outChannels() != 3) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: コンパイル後の入出力形状が不正です（入力%dch、出力%dch）。\n"),
            m_ov->inChannels(), m_ov->outChannels());
        return RGY_ERR_UNSUPPORTED;
    }

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
            m_baseGrid[idx] = vx;
            m_baseGrid[plane + idx] = vy;
            m_multiplier[idx] = multH;
            m_multiplier[plane + idx] = multV;
        }
    }

    auto frameOut = prm->frameOut;
    frameOut.csp = inCsp;
    frameOut.width = m_W;
    frameOut.height = m_H;
    prm->frameOut = frameOut;
    m_pathThrough = (FILTER_PATHTHROUGH_FRAMEINFO)(m_pathThrough &
        (~(uint32_t)(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS)));
    prm->baseFps *= m_multi;

    err = AllocFrameBuf(prm->frameOut, m_multi);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("rife-ov: 出力フレームの確保に失敗しました: %s。\n"), get_err_mes(err));
        return err;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    CspMatrix matrix = RGY_MATRIX_BT709;
    if (prm->colormatrix == _T("bt601")) matrix = RGY_MATRIX_ST170_M;
    else if (prm->colormatrix == _T("bt2020")) matrix = RGY_MATRIX_BT2020_NCL;
    else if (prm->colormatrix == _T("bt709")) matrix = RGY_MATRIX_BT709;
    else matrix = (m_H <= 576) ? RGY_MATRIX_ST170_M : RGY_MATRIX_BT709;
    const auto colorrange = (prm->colorrange == _T("pc")) ? RGY_COLORRANGE_FULL : RGY_COLORRANGE_LIMITED;

    RGYFrameInfo rgbIn;
    RGYFrameInfo rgbOut;
    if (m_useOcl) {
        m_inBufCL = m_cl->createBuffer(11 * plane * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
        m_outBufCL = m_cl->createBuffer(3 * plane * sizeof(float));
        if (!m_inBufCL || !m_outBufCL) {
            AddMessage(RGY_LOG_ERROR, _T("rife-ov: 共有テンソルバッファの確保に失敗しました。\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        err = createRgbPlanes(m_inBufCL.get(), 0, m_prevRgbPlanes);
        if (err == RGY_ERR_NONE) err = createRgbPlanes(m_inBufCL.get(), 3, m_currRgbPlanes);
        if (err == RGY_ERR_NONE) err = createRgbPlanes(m_outBufCL.get(), 0, m_outRgbPlanes);
        if (err != RGY_ERR_NONE) return err;
        rgbIn = rgbFrame(m_currRgbPlanes);
        rgbOut = rgbFrame(m_outRgbPlanes);

        err = m_inBufCL->queueMapBuffer(m_cl->queue(), CL_MAP_WRITE, {}, RGY_CL_MAP_BLOCK_ALL);
        if (err != RGY_ERR_NONE) return err;
        auto inPtr = (float *)m_inBufCL->mappedPtr();
        std::fill(inPtr, inPtr + 11 * plane, 0.0f);
        std::memcpy(inPtr + 7 * plane, m_baseGrid.data(), 2 * plane * sizeof(float));
        std::memcpy(inPtr + 9 * plane, m_multiplier.data(), 2 * plane * sizeof(float));
        err = m_inBufCL->unmapBuffer(m_cl->queue());
        if (err != RGY_ERR_NONE) return err;
        err = m_cl->queue().finish();
        if (err != RGY_ERR_NONE) return err;
        err = m_ov->setSharedIO((void *)m_inBufCL->mem(), (void *)m_outBufCL->mem());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("rife-ov: 共有テンソルのバインドに失敗しました。\n"));
            return err;
        }
    } else {
        m_prevRGB.resize(3 * plane);
        m_currRGB.resize(3 * plane);
        m_inBuf.resize(11 * plane);
        m_outBuf.resize(3 * plane);
        std::memcpy(m_inBuf.data() + 7 * plane, m_baseGrid.data(), 2 * plane * sizeof(float));
        std::memcpy(m_inBuf.data() + 9 * plane, m_multiplier.data(), 2 * plane * sizeof(float));
        m_inStaging = m_cl->createFrameBuffer(m_W, m_H, RGY_CSP_RGB_F32, 32, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
        m_outStaging = m_cl->createFrameBuffer(m_W, m_H, RGY_CSP_RGB_F32, 32, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
        if (!m_inStaging || !m_outStaging) {
            AddMessage(RGY_LOG_ERROR, _T("rife-ov: RGBステージングバッファの確保に失敗しました。\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        rgbIn = m_inStaging->frame;
        rgbOut = m_outStaging->frame;
    }

    auto cropToRgbParam = std::make_shared<RGYFilterParamCrop>();
    cropToRgbParam->frameIn = prm->frameIn;
    cropToRgbParam->frameOut = rgbIn;
    cropToRgbParam->baseFps = prm->baseFps;
    cropToRgbParam->matrix = matrix;
    cropToRgbParam->colorrange = colorrange;
    cropToRgbParam->chroma420Interpolate = true;
    m_cropToRgb = std::make_unique<RGYFilterCspCrop>(m_cl);
    err = m_cropToRgb->init(cropToRgbParam, m_pLog);
    if (err != RGY_ERR_NONE) return err;

    auto cropFromRgbParam = std::make_shared<RGYFilterParamCrop>();
    cropFromRgbParam->frameIn = rgbOut;
    cropFromRgbParam->frameOut = frameOut;
    cropFromRgbParam->baseFps = prm->baseFps;
    cropFromRgbParam->matrix = matrix;
    cropFromRgbParam->colorrange = colorrange;
    m_cropFromRgb = std::make_unique<RGYFilterCspCrop>(m_cl);
    err = m_cropFromRgb->init(cropFromRgbParam, m_pLog);
    if (err != RGY_ERR_NONE) return err;

    m_havePrev = false;
    m_param = prm;
    AddMessage(RGY_LOG_DEBUG, _T("rife-ov: %s、%dx%d、x%d、device=%s、path=%s。\n"),
        prm->modelFile.c_str(), m_W, m_H, m_multi, prm->device.c_str(), m_useOcl ? _T("ocl") : _T("host"));
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRifeOV::readRgbStaging(RGYOpenCLQueue &queue, std::vector<float>& dst) {
    auto err = m_inStaging->queueMapBuffer(queue, CL_MAP_READ, {}, RGY_CL_MAP_BLOCK_ALL);
    if (err != RGY_ERR_NONE) return err;
    const auto& host = m_inStaging->mappedHost()->host();
    const size_t rowBytes = (size_t)m_W * sizeof(float);
    const size_t plane = (size_t)m_W * m_H;
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < m_H; y++) {
            std::memcpy(dst.data() + c * plane + (size_t)y * m_W,
                host.ptr[c] + (size_t)y * host.pitch[c], rowBytes);
        }
    }
    return m_inStaging->unmapBuffer(queue);
}

RGY_ERR RGYFilterRifeOV::writeRgbStaging(RGYOpenCLQueue &queue, const std::vector<float>& src) {
    auto err = m_outStaging->queueMapBuffer(queue, CL_MAP_WRITE, {}, RGY_CL_MAP_BLOCK_ALL);
    if (err != RGY_ERR_NONE) return err;
    auto& host = m_outStaging->mappedHost()->host();
    const size_t rowBytes = (size_t)m_W * sizeof(float);
    const size_t plane = (size_t)m_W * m_H;
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < m_H; y++) {
            std::memcpy(host.ptr[c] + (size_t)y * host.pitch[c],
                src.data() + c * plane + (size_t)y * m_W, rowBytes);
        }
    }
    return m_outStaging->unmapBuffer(queue);
}

RGY_ERR RGYFilterRifeOV::interpolate(float t) {
    const size_t plane = (size_t)m_W * m_H;
    std::memcpy(m_inBuf.data(), m_prevRGB.data(), 3 * plane * sizeof(float));
    std::memcpy(m_inBuf.data() + 3 * plane, m_currRGB.data(), 3 * plane * sizeof(float));
    std::fill(m_inBuf.data() + 6 * plane, m_inBuf.data() + 7 * plane, t);
    return m_ov->infer(m_inBuf.data(), m_outBuf.data());
}

RGY_ERR RGYFilterRifeOV::interpolateOcl(float t, RGYOpenCLQueue &queue) {
    const size_t planeBytes = (size_t)m_W * m_H * sizeof(float);
    const auto clerr = clEnqueueFillBuffer(queue.get(), m_inBufCL->mem(), &t, sizeof(t),
        6 * planeBytes, planeBytes, 0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) return err_cl_to_rgy(clerr);
    return m_ov->inferShared();
}

RGY_ERR RGYFilterRifeOV::runHost(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto inputYuv = *pInputFrame;
    inputYuv.picstruct = RGY_PICSTRUCT_FRAME;
    RGYFrameInfo *rgbOut[1] = { &m_inStaging->frame };
    int outputCount = 0;
    auto err = m_cropToRgb->filter(&inputYuv, rgbOut, &outputCount, queue, wait_events, nullptr);
    if (err != RGY_ERR_NONE) return err;
    err = readRgbStaging(queue, m_currRGB);
    if (err != RGY_ERR_NONE) return err;

    if (!m_havePrev) {
        ppOutputFrames[0] = &m_frameBuf[0]->frame;
        err = m_cl->copyFrame(ppOutputFrames[0], pInputFrame, nullptr, queue, {}, event);
        if (err != RGY_ERR_NONE) return err;
        ppOutputFrames[0]->timestamp = pInputFrame->timestamp;
        ppOutputFrames[0]->duration = pInputFrame->duration;
        ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
        ppOutputFrames[0]->inputFrameId = pInputFrame->inputFrameId;
        *pOutputFrameNum = 1;
        m_prevRGB = m_currRGB;
        m_prevTimestamp = pInputFrame->timestamp;
        m_prevDuration = pInputFrame->duration;
        m_havePrev = true;
        return RGY_ERR_NONE;
    }

    const int64_t spanDur = pInputFrame->timestamp - m_prevTimestamp;
    for (int k = 1; k < m_multi; k++) {
        const float t = (float)k / (float)m_multi;
        err = interpolate(t);
        if (err != RGY_ERR_NONE) return err;
        err = writeRgbStaging(queue, m_outBuf);
        if (err != RGY_ERR_NONE) return err;
        auto rgb = m_outStaging->frame;
        RGYFrameInfo *out = &m_frameBuf[k - 1]->frame;
        RGYFrameInfo *yuvOut[1] = { out };
        outputCount = 0;
        err = m_cropFromRgb->filter(&rgb, yuvOut, &outputCount, queue, {}, nullptr);
        if (err != RGY_ERR_NONE) return err;
        out->timestamp = m_prevTimestamp + (spanDur > 0 ? spanDur * (int64_t)k / (int64_t)m_multi : 0);
        out->duration = (spanDur > 0) ? (spanDur / m_multi) : pInputFrame->duration;
        out->picstruct = pInputFrame->picstruct;
        out->inputFrameId = pInputFrame->inputFrameId;
        ppOutputFrames[k - 1] = out;
    }
    RGYFrameInfo *passthru = &m_frameBuf[m_multi - 1]->frame;
    err = m_cl->copyFrame(passthru, pInputFrame, nullptr, queue, {}, event);
    if (err != RGY_ERR_NONE) return err;
    passthru->timestamp = pInputFrame->timestamp;
    passthru->duration = (spanDur > 0) ? (spanDur / m_multi) : pInputFrame->duration;
    passthru->picstruct = pInputFrame->picstruct;
    passthru->inputFrameId = pInputFrame->inputFrameId;
    ppOutputFrames[m_multi - 1] = passthru;
    *pOutputFrameNum = m_multi;
    m_prevRGB.swap(m_currRGB);
    m_prevTimestamp = pInputFrame->timestamp;
    m_prevDuration = pInputFrame->duration;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRifeOV::runOcl(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto inputYuv = *pInputFrame;
    inputYuv.picstruct = RGY_PICSTRUCT_FRAME;
    auto currRgb = rgbFrame(m_currRgbPlanes);
    RGYFrameInfo *rgbOut[1] = { &currRgb };
    int outputCount = 0;
    auto err = m_cropToRgb->filter(&inputYuv, rgbOut, &outputCount, queue, wait_events, nullptr);
    if (err != RGY_ERR_NONE) return err;

    const size_t planeBytes = (size_t)m_W * m_H * sizeof(float);
    if (!m_havePrev) {
        const auto clerr = clEnqueueCopyBuffer(queue.get(), m_inBufCL->mem(), m_inBufCL->mem(),
            3 * planeBytes, 0, 3 * planeBytes, 0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) return err_cl_to_rgy(clerr);
        ppOutputFrames[0] = &m_frameBuf[0]->frame;
        err = m_cl->copyFrame(ppOutputFrames[0], pInputFrame, nullptr, queue, {}, event);
        if (err != RGY_ERR_NONE) return err;
        ppOutputFrames[0]->timestamp = pInputFrame->timestamp;
        ppOutputFrames[0]->duration = pInputFrame->duration;
        ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
        ppOutputFrames[0]->inputFrameId = pInputFrame->inputFrameId;
        *pOutputFrameNum = 1;
        m_prevTimestamp = pInputFrame->timestamp;
        m_prevDuration = pInputFrame->duration;
        m_havePrev = true;
        return RGY_ERR_NONE;
    }

    const int64_t spanDur = pInputFrame->timestamp - m_prevTimestamp;
    for (int k = 1; k < m_multi; k++) {
        const float t = (float)k / (float)m_multi;
        err = interpolateOcl(t, queue);
        if (err != RGY_ERR_NONE) return err;
        auto rgb = rgbFrame(m_outRgbPlanes);
        RGYFrameInfo *out = &m_frameBuf[k - 1]->frame;
        RGYFrameInfo *yuvOut[1] = { out };
        outputCount = 0;
        err = m_cropFromRgb->filter(&rgb, yuvOut, &outputCount, queue, {}, nullptr);
        if (err != RGY_ERR_NONE) return err;
        out->timestamp = m_prevTimestamp + (spanDur > 0 ? spanDur * (int64_t)k / (int64_t)m_multi : 0);
        out->duration = (spanDur > 0) ? (spanDur / m_multi) : pInputFrame->duration;
        out->picstruct = pInputFrame->picstruct;
        out->inputFrameId = pInputFrame->inputFrameId;
        ppOutputFrames[k - 1] = out;
    }
    auto clerr = clEnqueueCopyBuffer(queue.get(), m_inBufCL->mem(), m_inBufCL->mem(),
        3 * planeBytes, 0, 3 * planeBytes, 0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) return err_cl_to_rgy(clerr);
    RGYFrameInfo *passthru = &m_frameBuf[m_multi - 1]->frame;
    err = m_cl->copyFrame(passthru, pInputFrame, nullptr, queue, {}, event);
    if (err != RGY_ERR_NONE) return err;
    passthru->timestamp = pInputFrame->timestamp;
    passthru->duration = (spanDur > 0) ? (spanDur / m_multi) : pInputFrame->duration;
    passthru->picstruct = pInputFrame->picstruct;
    passthru->inputFrameId = pInputFrame->inputFrameId;
    ppOutputFrames[m_multi - 1] = passthru;
    *pOutputFrameNum = m_multi;
    m_prevTimestamp = pInputFrame->timestamp;
    m_prevDuration = pInputFrame->duration;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRifeOV::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (pInputFrame->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        return RGY_ERR_NONE;
    }
    return m_useOcl
        ? runOcl(pInputFrame, ppOutputFrames, pOutputFrameNum, queue, wait_events, event)
        : runHost(pInputFrame, ppOutputFrames, pOutputFrameNum, queue, wait_events, event);
}
