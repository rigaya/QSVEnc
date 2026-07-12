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

#include "rgy_filter_stdeint.h"
#include "rgy_filesystem.h"
#include "rgy_model_registry.h"
#include <algorithm>

static inline uint8_t stdeint_clamp_u8(int value) {
    return (uint8_t)(value < 0 ? 0 : (value > 255 ? 255 : value));
}

static inline float stdeint_clampf(float value, float low, float high) {
    return value < low ? low : (value > high ? high : value);
}

RGYFilterStDeint::RGYFilterStDeint(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context), m_ov(), m_width(0), m_height(0), m_mode(VppStDeintMode::Bob), m_defaultTff(true), m_useOcl(false),
    m_yOff(0), m_yScale(1), m_yRange(255), m_cOff(128), m_cScale(1), m_cRange(255),
    m_matVR(0), m_matUG(0), m_matVG(0), m_matUB(0),
    m_matRY(0), m_matGY(0), m_matBY(0), m_matRU(0), m_matGU(0), m_matBU(0), m_matRV(0), m_matGV(0), m_matBV(0),
    m_inputBuf(), m_outputBuf(), m_weaveBuf(), m_inputStaging(), m_outputStaging(),
    m_program(), m_inputBufCL(), m_outputBufCL() {
    m_name = _T("stdeint");
}

RGYFilterStDeint::~RGYFilterStDeint() {
    close();
}

void RGYFilterStDeint::close() {
    m_ov.reset();
    m_inputStaging.reset();
    m_outputStaging.reset();
    m_program.reset();
    m_inputBufCL.reset();
    m_outputBufCL.reset();
    m_inputBuf.clear();
    m_outputBuf.clear();
    m_weaveBuf.clear();
    m_frameBuf.clear();
}

tstring RGYFilterParamStDeint::print() const {
    return strsprintf(_T("stdeint: %s, mode %s, device %s, precision %s"), modelFile.c_str(),
        get_cx_desc(list_vpp_stdeint_mode, (int)mode), device.c_str(), precision.c_str());
}

void RGYFilterStDeint::setupColorCoeffs(int matrixSel, bool rangeTV, int pixMax) {
    float kr = 0.2126f, kb = 0.0722f;
    if (matrixSel == 601)  { kr = 0.299f;  kb = 0.114f; }
    if (matrixSel == 2020) { kr = 0.2627f; kb = 0.0593f; }
    const float kg = 1.0f - kr - kb;
    m_matVR = 2.0f * (1.0f - kr);
    m_matUG = -2.0f * kb * (1.0f - kb) / kg;
    m_matVG = -2.0f * kr * (1.0f - kr) / kg;
    m_matUB = 2.0f * (1.0f - kb);
    m_matRY = kr;                            m_matGY = kg;                            m_matBY = kb;
    m_matRU = -kr / (2.0f * (1.0f - kb));    m_matGU = -kg / (2.0f * (1.0f - kb));    m_matBU = 0.5f;
    m_matRV = 0.5f;                          m_matGV = -kg / (2.0f * (1.0f - kr));    m_matBV = -kb / (2.0f * (1.0f - kr));
    m_yOff   = rangeTV ? (16.0f  * pixMax / 255.0f) : 0.0f;
    m_yRange = rangeTV ? (219.0f * pixMax / 255.0f) : (float)pixMax;
    m_yScale = 1.0f / m_yRange;
    m_cOff   = rangeTV ? (128.0f * pixMax / 255.0f) : ((float)pixMax / 2.0f);
    m_cRange = rangeTV ? (224.0f * pixMax / 255.0f) : (float)pixMax;
    m_cScale = 1.0f / m_cRange;
}

RGY_ERR RGYFilterStDeint::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamStDeint>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!RGYOpenVINO::available()) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: this build was compiled without OpenVINO support.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->modelFile.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: model= (a registered model name or ST-DeInt .onnx path) is required.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->modelFile.find_first_of(_T("/\\.")) == tstring::npos && !prm->modelDir.empty()) {
        RGYModelRegistry registry;
        const auto err = registry.load(PathCombineS(prm->modelDir, _T("stdeint_ov_models.json")), m_pLog);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        if (!registry.find(prm->modelFile)) {
            AddMessage(RGY_LOG_ERROR, _T("stdeint: model \"%s\" not found in stdeint_ov_models.json\n"), prm->modelFile.c_str());
            return RGY_ERR_NOT_FOUND;
        }
        prm->modelFile = registry.resolveModelPath(prm->modelFile);
    }
    if (!rgy_file_exists(prm->modelFile)) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: model file not found: %s\n"), prm->modelFile.c_str());
        return RGY_ERR_FILE_OPEN;
    }

    const auto inputCsp = prm->frameIn.csp;
    if ((inputCsp != RGY_CSP_YV12 && inputCsp != RGY_CSP_NV12) || prm->frameIn.bitdepth != 8) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: supports 8-bit yuv420 (yv12/nv12) only; got %s %dbit.\n"),
            RGY_CSP_NAMES[inputCsp], prm->frameIn.bitdepth);
        return RGY_ERR_UNSUPPORTED;
    }
    m_width = prm->frameIn.width;
    m_height = prm->frameIn.height;
    if (m_height < 4 || (m_height & 1) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: height must be an even value of at least 4 (got %d).\n"), m_height);
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->mode != VppStDeintMode::Bob && prm->mode != VppStDeintMode::Normal) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: invalid output mode.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    m_mode = prm->mode;
    m_defaultTff = (prm->frameIn.picstruct & RGY_PICSTRUCT_BFF) == 0;

    m_ov = std::make_unique<RGYOpenVINO>();
    tstring errorMessage;
    int inputChannels = 0, outputChannels = 0;
    auto err = m_ov->peekChannels(prm->modelFile, inputChannels, outputChannels, errorMessage);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: failed to read model %s: %s\n"), prm->modelFile.c_str(), errorMessage.c_str());
        return err;
    }
    const auto deviceLower = tolowercase(prm->device);
    const bool deviceWantsGpu = deviceLower.substr(0, 3) == _T("gpu") || deviceLower == _T("auto");
    m_useOcl = deviceWantsGpu && m_cl;
    if (m_useOcl) {
        err = m_ov->initShared(prm->modelFile, (void *)m_cl->queue().get(), (void *)m_cl->context(),
            m_height, m_width, errorMessage, prm->precision);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_WARN, _T("stdeint: OpenCL zero-copy initialization failed; falling back to host path: %s\n"),
                errorMessage.c_str());
            m_useOcl = false;
            errorMessage.clear();
        }
    }
    if (!m_useOcl) {
        err = m_ov->init(prm->modelFile, prm->device, m_height, m_width, errorMessage, prm->precision);
    }
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: failed to load/compile model on %s: %s\n"), prm->device.c_str(), errorMessage.c_str());
        return err;
    }
    if (m_ov->inChannels() != 3 || m_ov->outChannels() != 6) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: invalid model (expected 3ch input / 6ch output, got %dch / %dch).\n"),
            m_ov->inChannels(), m_ov->outChannels());
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_ov->outHeight() == m_height && m_ov->outWidth() == m_width) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: this model contains the legacy ONNX weave output; re-export it with the current export_stdeint.py.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_ov->outHeight() != m_height / 2 || m_ov->outWidth() != m_width) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: restoration output must be 6ch with half input height (expected %dx%d, got %dx%d).\n"),
            m_width, m_height / 2, m_ov->outWidth(), m_ov->outHeight());
        return RGY_ERR_UNSUPPORTED;
    }

    int matrixSel;
    if      (prm->colormatrix == _T("bt601"))  matrixSel = 601;
    else if (prm->colormatrix == _T("bt2020")) matrixSel = 2020;
    else if (prm->colormatrix == _T("bt709"))  matrixSel = 709;
    else                                         matrixSel = (m_height <= 576) ? 601 : 709;
    setupColorCoeffs(matrixSel, prm->colorrange != _T("pc"), 255);

    prm->frameOut.csp = inputCsp;
    prm->frameOut.width = m_width;
    prm->frameOut.height = m_height;
    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    m_pathThrough = (FILTER_PATHTHROUGH_FRAMEINFO)(m_pathThrough &
        (~(uint32_t)(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS)));
    if (m_mode == VppStDeintMode::Bob) {
        prm->baseFps *= 2;
    }

    err = AllocFrameBuf(prm->frameOut, (m_mode == VppStDeintMode::Bob) ? 2 : 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: failed to allocate output frame buffer: %s.\n"), get_err_mes(err));
        return err;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    const size_t plane = (size_t)m_width * m_height;
    if (m_useOcl) {
        m_program = m_cl->buildResource(_T("RGY_FILTER_STDEINT_CL"), _T("EXE_DATA"), std::string());
        m_inputBufCL = m_cl->createBuffer(3 * plane * sizeof(float));
        m_outputBufCL = m_cl->createBuffer(3 * plane * sizeof(float));
        if (!m_program || !m_inputBufCL || !m_outputBufCL) {
            AddMessage(RGY_LOG_WARN, _T("stdeint: failed to prepare OpenCL zero-copy buffers; falling back to host path.\n"));
            m_useOcl = false;
        } else {
            err = m_ov->setSharedIO((void *)m_inputBufCL->mem(), (void *)m_outputBufCL->mem());
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_WARN, _T("stdeint: failed to bind OpenCL remote tensors; falling back to host path: %s.\n"),
                    get_err_mes(err));
                m_useOcl = false;
            }
        }
        if (!m_useOcl) {
            m_program.reset();
            m_inputBufCL.reset();
            m_outputBufCL.reset();
            errorMessage.clear();
            err = m_ov->init(prm->modelFile, prm->device, m_height, m_width, errorMessage, prm->precision);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("stdeint: host fallback model initialization failed on %s: %s\n"),
                    prm->device.c_str(), errorMessage.c_str());
                return err;
            }
        }
    }
    if (!m_useOcl) {
        m_inputBuf.resize(3 * plane);
        m_outputBuf.resize(3 * plane);
        m_weaveBuf.resize(3 * plane);
        m_inputStaging = m_cl->createFrameBuffer(m_width, m_height, inputCsp, prm->frameIn.bitdepth,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
        m_outputStaging = m_cl->createFrameBuffer(m_width, m_height, inputCsp, prm->frameIn.bitdepth,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
        if (!m_inputStaging || !m_outputStaging) {
            AddMessage(RGY_LOG_ERROR, _T("stdeint: failed to allocate staging frame buffers.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    m_param = prm;
    setFilterInfo(prm->print() + strsprintf(_T(", path %s"), m_useOcl ? _T("ocl") : _T("host")));
    AddMessage(RGY_LOG_DEBUG, _T("stdeint: %s, %dx%d, mode %s, device %s, path %s.\n"),
        prm->modelFile.c_str(), m_width, m_height, get_cx_desc(list_vpp_stdeint_mode, (int)m_mode),
        prm->device.c_str(), m_useOcl ? _T("ocl") : _T("host"));
    return RGY_ERR_NONE;
}

void RGYFilterStDeint::yuvToRGB(const RGYFrameInfo& input, float *dst) {
    const size_t plane = (size_t)m_width * m_height;
    const bool nv12 = input.csp == RGY_CSP_NV12;
    const int chromaWidth = m_width / 2;
    const int chromaHeight = m_height / 2;
    const uint8_t *uPlane = input.ptr[1];
    const uint8_t *vPlane = nv12 ? input.ptr[1] + 1 : input.ptr[2];
    const int chromaStride = nv12 ? 2 : 1;
    const int uPitch = input.pitch[1];
    const int vPitch = nv12 ? input.pitch[1] : input.pitch[2];
    float *red = dst;
    float *green = dst + plane;
    float *blue = dst + 2 * plane;
    for (int y = 0; y < m_height; y++) {
        const uint8_t *yRow = input.ptr[0] + (size_t)y * input.pitch[0];
        const int cy = std::min(y / 2, chromaHeight - 1);
        for (int x = 0; x < m_width; x++) {
            const int cx = std::min(x / 2, chromaWidth - 1);
            const float yn = ((float)yRow[x] - m_yOff) * m_yScale;
            const float un = ((float)uPlane[(size_t)cy * uPitch + (size_t)cx * chromaStride] - m_cOff) * m_cScale;
            const float vn = ((float)vPlane[(size_t)cy * vPitch + (size_t)cx * chromaStride] - m_cOff) * m_cScale;
            const size_t index = (size_t)y * m_width + x;
            red[index] = stdeint_clampf(yn + m_matVR * vn, 0.0f, 1.0f);
            green[index] = stdeint_clampf(yn + m_matUG * un + m_matVG * vn, 0.0f, 1.0f);
            blue[index] = stdeint_clampf(yn + m_matUB * un, 0.0f, 1.0f);
        }
    }
}

void RGYFilterStDeint::rgbToYUV(const RGYFrameInfo& output, const float *src) {
    const size_t plane = (size_t)m_width * m_height;
    const bool nv12 = output.csp == RGY_CSP_NV12;
    const int chromaWidth = m_width / 2;
    const int chromaHeight = m_height / 2;
    const float *red = src;
    const float *green = src + plane;
    const float *blue = src + 2 * plane;
    uint8_t *uPlane = output.ptr[1];
    uint8_t *vPlane = nv12 ? output.ptr[1] + 1 : output.ptr[2];
    const int chromaStride = nv12 ? 2 : 1;
    const int uPitch = output.pitch[1];
    const int vPitch = nv12 ? output.pitch[1] : output.pitch[2];
    for (int y = 0; y < m_height; y++) {
        uint8_t *yRow = output.ptr[0] + (size_t)y * output.pitch[0];
        for (int x = 0; x < m_width; x++) {
            const size_t index = (size_t)y * m_width + x;
            const float luma = m_matRY * red[index] + m_matGY * green[index] + m_matBY * blue[index];
            yRow[x] = stdeint_clamp_u8((int)(luma * m_yRange + m_yOff + 0.5f));
        }
    }
    for (int cy = 0; cy < chromaHeight; cy++) {
        for (int cx = 0; cx < chromaWidth; cx++) {
            float u = 0.0f, v = 0.0f;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    const size_t index = (size_t)(cy * 2 + dy) * m_width + (cx * 2 + dx);
                    u += m_matRU * red[index] + m_matGU * green[index] + m_matBU * blue[index];
                    v += m_matRV * red[index] + m_matGV * green[index] + m_matBV * blue[index];
                }
            }
            u *= 0.25f;
            v *= 0.25f;
            uPlane[(size_t)cy * uPitch + (size_t)cx * chromaStride] = stdeint_clamp_u8((int)(u * m_cRange + m_cOff + 0.5f));
            vPlane[(size_t)cy * vPitch + (size_t)cx * chromaStride] = stdeint_clamp_u8((int)(v * m_cRange + m_cOff + 0.5f));
        }
    }
}

void RGYFilterStDeint::setOutputFrameProp(RGYFrameInfo *output, const RGYFrameInfo *input) const {
    copyFramePropWithoutRes(output, input);
    output->picstruct = RGY_PICSTRUCT_FRAME;
    output->flags = (RGY_FRAME_FLAGS)(input->flags &
        ~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_BFF | RGY_FRAME_FLAG_RFF_TFF));
}

void RGYFilterStDeint::setBobTimestamp(const RGYFrameInfo *input, RGYFrameInfo **outputs) const {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamStDeint>(m_param);
    auto frameDuration = input->duration;
    if (frameDuration == 0 && prm && prm->timebase.is_valid()) {
        frameDuration = (decltype(frameDuration))((prm->timebase.inv() / prm->baseFps * 2).qdouble() + 0.5);
    }
    outputs[0]->timestamp = input->timestamp;
    outputs[0]->duration = (frameDuration + 1) / 2;
    outputs[1]->timestamp = outputs[0]->timestamp + outputs[0]->duration;
    outputs[1]->duration = frameDuration - outputs[0]->duration;
    outputs[0]->inputFrameId = input->inputFrameId;
    outputs[1]->inputFrameId = input->inputFrameId;
}

void RGYFilterStDeint::weaveRestoration(float *dst, const float *restoration, bool frameA) const {
    const size_t plane = (size_t)m_width * m_height;
    const size_t halfPlane = plane / 2;
    for (int channel = 0; channel < 3; channel++) {
        const auto inputPlane = m_inputBuf.data() + (size_t)channel * plane;
        const auto restorePlane = restoration + (size_t)channel * halfPlane;
        auto outputPlane = dst + (size_t)channel * plane;
        for (int y = 0; y < m_height / 2; y++) {
            const auto inputRow = inputPlane + (size_t)(y * 2 + (frameA ? 0 : 1)) * m_width;
            const auto restoreRow = restorePlane + (size_t)y * m_width;
            auto upperRow = outputPlane + (size_t)(y * 2) * m_width;
            auto lowerRow = upperRow + m_width;
            std::copy_n(frameA ? inputRow : restoreRow, m_width, upperRow);
            std::copy_n(frameA ? restoreRow : inputRow, m_width, lowerRow);
        }
    }
}

RGY_ERR RGYFilterStDeint::writeOutputFrame(RGYFrameInfo *output, const float *rgb,
    RGYOpenCLQueue& queue, RGYOpenCLEvent *event) {
    auto err = m_outputStaging->queueMapBuffer(queue, CL_MAP_WRITE, {}, RGY_CL_MAP_BLOCK_ALL);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: failed to map output staging: %s.\n"), get_err_mes(err));
        return err;
    }
    rgbToYUV(m_outputStaging->mappedHost()->host(), rgb);
    m_outputStaging->unmapBuffer(queue);
    err = m_cl->copyFrame(output, &m_outputStaging->frame, nullptr, queue, {}, event);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: failed to copy output from staging: %s.\n"), get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterStDeint::runOcl(const RGYFrameInfo *input, RGYFrameInfo **outputs, int outputCount,
    RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event) {
    const bool nv12 = input->csp == RGY_CSP_NV12;
    const auto inputU = (cl_mem)input->ptr[1];
    const auto inputV = nv12 ? inputU : (cl_mem)input->ptr[2];
    const int inputVPitch = nv12 ? input->pitch[1] : input->pitch[2];
    RGYWorkSize local(32, 8);
    RGYWorkSize global(m_width, m_height);
    auto err = m_program->kernel("stdeint_pack_rgb").config(queue, local, global, wait_events, nullptr).launch(
        (cl_mem)input->ptr[0], input->pitch[0], inputU, input->pitch[1], inputV, inputVPitch, nv12 ? 1 : 0,
        m_inputBufCL->mem(), m_width, m_height,
        m_yOff, m_yScale, m_cOff, m_cScale, m_matVR, m_matUG, m_matVG, m_matUB);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: OpenCL RGB pack failed: %s.\n"), get_err_mes(err));
        return err;
    }

    err = m_ov->inferShared();
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: shared OpenCL inference failed: %s.\n"), get_err_mes(err));
        return err;
    }

    bool inputTff = m_defaultTff;
    if (input->picstruct & RGY_PICSTRUCT_BFF) {
        inputTff = false;
    } else if (input->picstruct & RGY_PICSTRUCT_TFF) {
        inputTff = true;
    }
    const int firstIndex = inputTff ? 0 : 1;
    const int sourceIndices[2] = { firstIndex, 1 - firstIndex };
    for (int i = 0; i < outputCount; i++) {
        auto output = &m_frameBuf[i]->frame;
        const bool outputNv12 = output->csp == RGY_CSP_NV12;
        const auto outputU = (cl_mem)output->ptr[1];
        const auto outputV = outputNv12 ? outputU : (cl_mem)output->ptr[2];
        const int outputVPitch = outputNv12 ? output->pitch[1] : output->pitch[2];
        const int frameA = sourceIndices[i] == 0 ? 1 : 0;
        err = m_program->kernel("stdeint_weave_yuv").config(queue, local, global, {},
            (i == outputCount - 1) ? event : nullptr).launch(
                m_inputBufCL->mem(), m_outputBufCL->mem(),
                (cl_mem)output->ptr[0], output->pitch[0], outputU, output->pitch[1], outputV, outputVPitch,
                outputNv12 ? 1 : 0, m_width, m_height, frameA,
                m_yOff, m_yRange, m_cOff, m_cRange,
                m_matRY, m_matGY, m_matBY, m_matRU, m_matGU, m_matBU, m_matRV, m_matGV, m_matBV);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("stdeint: OpenCL weave failed: %s.\n"), get_err_mes(err));
            return err;
        }
        setOutputFrameProp(output, input);
        outputs[i] = output;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterStDeint::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events,
    RGYOpenCLEvent *event) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    ppOutputFrames[1] = nullptr;
    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }

    const bool bob = m_mode == VppStDeintMode::Bob;
    const int outputCount = bob ? 2 : 1;
    if ((pInputFrame->picstruct & RGY_PICSTRUCT_INTERLACED) == 0) {
        for (int i = 0; i < outputCount; i++) {
            auto output = &m_frameBuf[i]->frame;
            const auto err = m_cl->copyFrame(output, pInputFrame, nullptr, queue,
                (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>(),
                (i == outputCount - 1) ? event : nullptr);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("stdeint: failed to copy progressive input: %s.\n"), get_err_mes(err));
                return err;
            }
            setOutputFrameProp(output, pInputFrame);
            ppOutputFrames[i] = output;
        }
        *pOutputFrameNum = outputCount;
        if (bob) {
            setBobTimestamp(pInputFrame, ppOutputFrames);
        }
        return RGY_ERR_NONE;
    }

    if (m_useOcl) {
        const auto err = runOcl(pInputFrame, ppOutputFrames, outputCount, queue, wait_events, event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        *pOutputFrameNum = outputCount;
        if (bob) {
            setBobTimestamp(pInputFrame, ppOutputFrames);
        }
        return RGY_ERR_NONE;
    }

    auto err = m_cl->copyFrame(&m_inputStaging->frame, pInputFrame, nullptr, queue, wait_events, nullptr);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: failed to copy input to staging: %s.\n"), get_err_mes(err));
        return err;
    }
    err = m_inputStaging->queueMapBuffer(queue, CL_MAP_READ, {}, RGY_CL_MAP_BLOCK_ALL);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: failed to map input staging: %s.\n"), get_err_mes(err));
        return err;
    }
    yuvToRGB(m_inputStaging->mappedHost()->host(), m_inputBuf.data());
    m_inputStaging->unmapBuffer(queue);

    err = m_ov->infer(m_inputBuf.data(), m_outputBuf.data());
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("stdeint: inference failed: %s.\n"), get_err_mes(err));
        return err;
    }
    const size_t restorationElements = (size_t)3 * m_width * (m_height / 2);
    bool inputTff = m_defaultTff;
    if (pInputFrame->picstruct & RGY_PICSTRUCT_BFF) {
        inputTff = false;
    } else if (pInputFrame->picstruct & RGY_PICSTRUCT_TFF) {
        inputTff = true;
    }
    const int firstIndex = inputTff ? 0 : 1;
    const int secondIndex = 1 - firstIndex;
    const int sourceIndices[2] = { firstIndex, secondIndex };
    for (int i = 0; i < outputCount; i++) {
        auto output = &m_frameBuf[i]->frame;
        const int frameIndex = sourceIndices[i];
        weaveRestoration(m_weaveBuf.data(), m_outputBuf.data() + (size_t)frameIndex * restorationElements, frameIndex == 0);
        err = writeOutputFrame(output, m_weaveBuf.data(),
            queue, (i == outputCount - 1) ? event : nullptr);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        setOutputFrameProp(output, pInputFrame);
        ppOutputFrames[i] = output;
    }
    *pOutputFrameNum = outputCount;
    if (bob) {
        setBobTimestamp(pInputFrame, ppOutputFrames);
    }
    return RGY_ERR_NONE;
}
