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

#include "rgy_filter_onnx_deint.h"
#include "rgy_filesystem.h"
#include "rgy_model_registry.h"
#include <algorithm>
#include <cstring>

static const int ONNX_DEINT_TEMPORAL_IN_CHANNELS = 9;
static const int ONNX_DEINT_TEMPORAL_OUT_CHANNELS = 3;

static const TCHAR *onnx_deint_cx_desc_or_unknown(const CX_DESC *list, int value) {
    const auto desc = get_cx_desc(list, value);
    return (desc != nullptr) ? desc : _T("unknown");
}

static bool onnx_deint_resolve_matrix(CspMatrix matrix, int inputHeight, CspMatrix& resolved) {
    if (matrix == RGY_MATRIX_AUTO || (int)matrix == COLOR_VALUE_AUTO_RESOLUTION) {
        resolved = (inputHeight <= 576) ? RGY_MATRIX_ST170_M : RGY_MATRIX_BT709;
        return true;
    }
    switch (matrix) {
    case RGY_MATRIX_ST170_M:
    case RGY_MATRIX_BT470_BG:
        resolved = RGY_MATRIX_ST170_M;
        return true;
    case RGY_MATRIX_BT709:
        resolved = RGY_MATRIX_BT709;
        return true;
    case RGY_MATRIX_BT2020_NCL:
        resolved = RGY_MATRIX_BT2020_NCL;
        return true;
    default:
        return false;
    }
}

static bool onnx_deint_supported_colorrange(CspColorRange range) {
    return range == RGY_COLORRANGE_AUTO
        || range == RGY_COLORRANGE_LIMITED
        || range == RGY_COLORRANGE_FULL;
}

RGYFilterOnnxDeint::RGYFilterOnnxDeint(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context), m_ov(), m_cropToRgb(), m_cropFromRgb(), m_width(0), m_height(0),
    m_mode(VppOnnxDeintMode::Bob), m_defaultTff(true), m_useOcl(false),
    m_inputBuf(), m_outputBuf(), m_program(), m_inputBufCL(), m_outputBufCL(), m_weaveBufCL(),
    m_inputPlanes(), m_weavePlanes(),
    m_temporal(false), m_framesIn(0), m_frameOut(0), m_temporalRing() {
    m_name = _T("onnx-deint");
}

RGYFilterOnnxDeint::~RGYFilterOnnxDeint() {
    close();
}

void RGYFilterOnnxDeint::close() {
    m_ov.reset();
    m_cropToRgb.reset();
    m_cropFromRgb.reset();
    m_program.reset();
    for (auto& plane : m_inputPlanes) plane.reset();
    for (auto& plane : m_weavePlanes) plane.reset();
    m_inputBufCL.reset();
    m_outputBufCL.reset();
    m_weaveBufCL.reset();
    m_inputBuf.clear();
    m_outputBuf.clear();
    for (auto& slot : m_temporalRing) {
        slot.frame.reset();
        slot.rgb.clear();
    }
    m_framesIn = 0;
    m_frameOut = 0;
    m_frameBuf.clear();
}

tstring RGYFilterParamOnnxDeint::print() const {
    return strsprintf(_T("onnx-deint: %s, mode %s, device %s, precision %s, colormatrix %s, colorrange %s"), modelFile.c_str(),
        get_cx_desc(list_vpp_onnx_deint_mode, (int)mode), device.c_str(), precision.c_str(),
        onnx_deint_cx_desc_or_unknown(list_colormatrix, colormatrix), onnx_deint_cx_desc_or_unknown(list_colorrange, colorrange));
}

RGY_ERR RGYFilterOnnxDeint::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamOnnxDeint>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!RGYOpenVINO::available()) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: this build was compiled without OpenVINO support.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->modelFile.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: model= (registered model name) is required.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->modelFile.find_first_of(_T("/\\.")) != tstring::npos) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: model must be a registered name, not a path: %s\n"), prm->modelFile.c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->modelDir.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: --vpp-onnx-model-dir is required for model registry.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    RGYModelRegistry registry;
    const auto registryErr = registry.load(PathCombineS(prm->modelDir, _T("onnx_deint_models.json")), m_pLog);
    if (registryErr != RGY_ERR_NONE) {
        return registryErr;
    }
    const auto modelEntry = registry.find(prm->modelFile);
    if (!modelEntry) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: model \"%s\" is not registered in onnx_deint_models.json.\n"), prm->modelFile.c_str());
        return RGY_ERR_NOT_FOUND;
    }
    if (!modelEntry->onnxDeintArchitecturePresent) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: model \"%s\" has no architecture in onnx_deint_models.json.\n"), prm->modelFile.c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    if (!modelEntry->onnxDeintArchitectureTypeValid || !modelEntry->onnxDeintArchitecture) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: model \"%s\" architecture must be a string.\n"), prm->modelFile.c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    if (*modelEntry->onnxDeintArchitecture == _T("stdeint")) {
        prm->architecture = VppOnnxDeintArchitecture::StDeint;
    } else if (*modelEntry->onnxDeintArchitecture == _T("ddd")) {
        prm->architecture = VppOnnxDeintArchitecture::DDD;
    } else {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: model \"%s\" has unknown architecture \"%s\" (expected stdeint or ddd).\n"),
            prm->modelFile.c_str(), modelEntry->onnxDeintArchitecture->c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    prm->modelFile = registry.resolveModelPath(prm->modelFile);
    if (!rgy_file_exists(prm->modelFile)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: model file not found: %s\n"), prm->modelFile.c_str());
        return RGY_ERR_FILE_OPEN;
    }

    const auto inputCsp = prm->frameIn.csp;
    if ((inputCsp != RGY_CSP_YV12 && inputCsp != RGY_CSP_NV12) || prm->frameIn.bitdepth != 8) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: supports 8-bit yuv420 (yv12/nv12) only; got %s %dbit.\n"),
            RGY_CSP_NAMES[inputCsp], prm->frameIn.bitdepth);
        return RGY_ERR_UNSUPPORTED;
    }
    m_width = prm->frameIn.width;
    m_height = prm->frameIn.height;
    if (m_height < 4 || (m_height & 1) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: height must be an even value of at least 4 (got %d).\n"), m_height);
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->mode != VppOnnxDeintMode::Bob && prm->mode != VppOnnxDeintMode::Normal) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: invalid output mode.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    CspMatrix matrix = RGY_MATRIX_UNSPECIFIED;
    if (!onnx_deint_resolve_matrix(prm->colormatrix, m_height, matrix)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: unsupported colormatrix %s.\n"),
            onnx_deint_cx_desc_or_unknown(list_colormatrix, prm->colormatrix));
        return RGY_ERR_UNSUPPORTED;
    }
    if (!onnx_deint_supported_colorrange(prm->colorrange)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: unsupported colorrange %s.\n"),
            onnx_deint_cx_desc_or_unknown(list_colorrange, prm->colorrange));
        return RGY_ERR_UNSUPPORTED;
    }
    m_mode = prm->mode;
    m_defaultTff = (prm->frameIn.picstruct & RGY_PICSTRUCT_BFF) == 0;

    m_ov = std::make_unique<RGYOpenVINO>();
    tstring errorMessage;
    int inputChannels = 0, outputChannels = 0;
    auto err = m_ov->peekChannels(prm->modelFile, inputChannels, outputChannels, errorMessage);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to read model %s: %s\n"), prm->modelFile.c_str(), errorMessage.c_str());
        return err;
    }
    //モデル方式はマニフェストのarchitectureで確定し、チャンネル数から推測しない。
    m_temporal = (prm->architecture == VppOnnxDeintArchitecture::DDD);
    //temporalモデルはフィールドを転置して渡すため、モデルの縦横は(フレーム幅, フィールド高さ)となる。
    const int modelHeight = m_temporal ? m_width : m_height;
    const int modelWidth = m_temporal ? m_height / 2 : m_width;
    const auto deviceLower = tolowercase(prm->device);
    const bool deviceWantsGpu = deviceLower.substr(0, 3) == _T("gpu") || deviceLower == _T("auto");
    //zero-copy経路のRGBバッファは3ch/フレーム解像度専用なので、temporalモデルではhost経路を使う。
    m_useOcl = deviceWantsGpu && m_cl && !m_temporal;
    if (m_useOcl) {
        err = m_ov->initShared(prm->modelFile, (void *)m_cl->queue().get(), (void *)m_cl->context(),
            modelHeight, modelWidth, errorMessage, prm->precision);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_WARN, _T("onnx-deint: OpenCL zero-copy initialization failed; falling back to host path: %s\n"),
                errorMessage.c_str());
            m_useOcl = false;
            errorMessage.clear();
        }
    }
    if (!m_useOcl) {
        err = m_ov->init(prm->modelFile, prm->device, modelHeight, modelWidth, errorMessage, prm->precision);
    }
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to load/compile model on %s: %s\n"), prm->device.c_str(), errorMessage.c_str());
        return err;
    }
    if (m_temporal) {
        if (m_ov->inChannels() != ONNX_DEINT_TEMPORAL_IN_CHANNELS || m_ov->outChannels() != ONNX_DEINT_TEMPORAL_OUT_CHANNELS) {
            AddMessage(RGY_LOG_ERROR, _T("onnx-deint: invalid temporal model (expected %dch input / %dch output, got %dch / %dch).\n"),
                ONNX_DEINT_TEMPORAL_IN_CHANNELS, ONNX_DEINT_TEMPORAL_OUT_CHANNELS, m_ov->inChannels(), m_ov->outChannels());
            return RGY_ERR_UNSUPPORTED;
        }
        if (m_ov->outHeight() != modelHeight || m_ov->outWidth() != modelWidth) {
            AddMessage(RGY_LOG_ERROR, _T("onnx-deint: temporal output must keep the transposed field size (expected %dx%d, got %dx%d).\n"),
                modelWidth, modelHeight, m_ov->outWidth(), m_ov->outHeight());
            return RGY_ERR_UNSUPPORTED;
        }
    } else {
        if (m_ov->inChannels() != 3 || m_ov->outChannels() != 6) {
            AddMessage(RGY_LOG_ERROR, _T("onnx-deint: invalid model (expected 3ch input / 6ch output, got %dch / %dch).\n"),
                m_ov->inChannels(), m_ov->outChannels());
            return RGY_ERR_UNSUPPORTED;
        }
        if (m_ov->outHeight() != m_height / 2 || m_ov->outWidth() != m_width) {
            AddMessage(RGY_LOG_ERROR, _T("onnx-deint: restoration output must be 6ch with half input height (expected %dx%d, got %dx%d).\n"),
                m_width, m_height / 2, m_ov->outWidth(), m_ov->outHeight());
            return RGY_ERR_UNSUPPORTED;
        }
    }

    prm->frameOut.csp = inputCsp;
    prm->frameOut.width = m_width;
    prm->frameOut.height = m_height;
    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    m_pathThrough = (FILTER_PATHTHROUGH_FRAMEINFO)(m_pathThrough &
        (~(uint32_t)(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS)));
    if (m_mode == VppOnnxDeintMode::Bob) {
        prm->baseFps *= 2;
    }

    err = AllocFrameBuf(prm->frameOut, (m_mode == VppOnnxDeintMode::Bob) ? 2 : 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to allocate output frame buffer: %s.\n"), get_err_mes(err));
        return err;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    const size_t plane = (size_t)m_width * m_height;
    m_program = m_cl->buildResource(_T("RGY_FILTER_ONNX_DEINT_CL"), _T("EXE_DATA"), std::string());
    m_inputBufCL = m_cl->createBuffer(3 * plane * sizeof(float));
    m_outputBufCL = m_cl->createBuffer(3 * plane * sizeof(float));
    m_weaveBufCL = m_cl->createBuffer(3 * plane * sizeof(float));
    if (!m_program || !m_inputBufCL || !m_outputBufCL || !m_weaveBufCL
        || createRgbPlanes(m_inputBufCL.get(), m_inputPlanes) != RGY_ERR_NONE
        || createRgbPlanes(m_weaveBufCL.get(), m_weavePlanes) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to prepare OpenCL RGB tensor buffers.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    if (m_useOcl) {
        err = m_ov->setSharedIO((void *)m_inputBufCL->mem(), (void *)m_outputBufCL->mem());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_WARN, _T("onnx-deint: failed to bind OpenCL remote tensors; falling back to host path: %s.\n"), get_err_mes(err));
            m_useOcl = false;
            errorMessage.clear();
            err = m_ov->init(prm->modelFile, prm->device, modelHeight, modelWidth, errorMessage, prm->precision);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("onnx-deint: host fallback model initialization failed on %s: %s\n"), prm->device.c_str(), errorMessage.c_str());
                return err;
            }
        }
    }
    if (!m_useOcl) {
        const size_t modelPlane = (size_t)modelHeight * modelWidth;
        m_inputBuf.resize(m_temporal ? ONNX_DEINT_TEMPORAL_IN_CHANNELS * modelPlane : 3 * plane);
        m_outputBuf.resize(m_temporal ? ONNX_DEINT_TEMPORAL_OUT_CHANNELS * modelPlane : 3 * plane);
    }
    if (m_temporal) {
        err = allocTemporalRing(inputCsp, prm->frameIn.bitdepth);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }

    auto rgbInfo = rgbFrame(m_inputPlanes);
    auto cropToRgbParam = std::make_shared<RGYFilterParamCrop>();
    cropToRgbParam->frameIn = prm->frameIn;
    cropToRgbParam->frameIn.picstruct = RGY_PICSTRUCT_FRAME;
    cropToRgbParam->frameOut = rgbInfo;
    cropToRgbParam->baseFps = prm->baseFps;
    cropToRgbParam->matrix = matrix;
    cropToRgbParam->colorrange = (prm->colorrange == RGY_COLORRANGE_FULL) ? RGY_COLORRANGE_FULL : RGY_COLORRANGE_LIMITED;
    cropToRgbParam->chroma420Interpolate = false;
    m_cropToRgb = std::make_unique<RGYFilterCspCrop>(m_cl);
    err = m_cropToRgb->init(cropToRgbParam, m_pLog);
    if (err != RGY_ERR_NONE) return err;

    auto cropFromRgbParam = std::make_shared<RGYFilterParamCrop>();
    cropFromRgbParam->frameIn = rgbFrame(m_weavePlanes);
    cropFromRgbParam->frameOut = prm->frameOut;
    cropFromRgbParam->baseFps = prm->baseFps;
    cropFromRgbParam->matrix = matrix;
    cropFromRgbParam->colorrange = cropToRgbParam->colorrange;
    m_cropFromRgb = std::make_unique<RGYFilterCspCrop>(m_cl);
    err = m_cropFromRgb->init(cropFromRgbParam, m_pLog);
    if (err != RGY_ERR_NONE) return err;

    m_param = prm;
    setFilterInfo(prm->print() + strsprintf(_T(", path %s%s"), m_useOcl ? _T("ocl") : _T("host"), m_temporal ? _T(", temporal") : _T("")));
    AddMessage(RGY_LOG_DEBUG, _T("onnx-deint: %s, %dx%d, mode %s, device %s, path %s, model %dch/%dch %dx%d.\n"),
        prm->modelFile.c_str(), m_width, m_height, get_cx_desc(list_vpp_onnx_deint_mode, (int)m_mode),
        prm->device.c_str(), m_useOcl ? _T("ocl") : _T("host"),
        m_ov->inChannels(), m_ov->outChannels(), modelWidth, modelHeight);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterOnnxDeint::createRgbPlanes(RGYCLBuf *parent, std::array<std::unique_ptr<RGYCLBuf>, 3>& planes) {
    const size_t planeBytes = (size_t)m_width * m_height * sizeof(float);
    for (int i = 0; i < 3; i++) {
        cl_buffer_region region = { (size_t)i * planeBytes, planeBytes };
        cl_int clerr = CL_SUCCESS;
        auto subbuf = clCreateSubBuffer(parent->mem(), CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &clerr);
        if (clerr != CL_SUCCESS || subbuf == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to create RGB plane sub-buffer %d: %s.\n"), i, cl_errmes(clerr));
            return err_cl_to_rgy(clerr);
        }
        planes[i] = std::make_unique<RGYCLBuf>(subbuf, CL_MEM_READ_WRITE, planeBytes);
    }
    return RGY_ERR_NONE;
}

RGYFrameInfo RGYFilterOnnxDeint::rgbFrame(const std::array<std::unique_ptr<RGYCLBuf>, 3>& planes) const {
    RGYFrameInfo frame;
    frame.width = m_width;
    frame.height = m_height;
    frame.csp = RGY_CSP_RGB_F32;
    frame.bitdepth = 32;
    frame.mem_type = RGY_MEM_TYPE_GPU;
    frame.picstruct = RGY_PICSTRUCT_FRAME;
    for (int i = 0; i < 3; i++) {
        frame.ptr[i] = (uint8_t *)planes[i]->mem();
        frame.pitch[i] = m_width * sizeof(float);
    }
    return frame;
}

RGY_ERR RGYFilterOnnxDeint::convertToRgb(const RGYFrameInfo *input, RGYOpenCLQueue& queue,
    const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event) {
    auto inputFrame = *input;
    inputFrame.picstruct = RGY_PICSTRUCT_FRAME;
    auto outputFrame = rgbFrame(m_inputPlanes);
    RGYFrameInfo *outputs[1] = { &outputFrame };
    int outputCount = 0;
    return m_cropToRgb->filter(&inputFrame, outputs, &outputCount, queue, wait_events, event);
}

RGY_ERR RGYFilterOnnxDeint::convertFromRgb(RGYFrameInfo *output, RGYOpenCLQueue& queue,
    const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event) {
    auto inputFrame = rgbFrame(m_weavePlanes);
    RGYFrameInfo *outputs[1] = { output };
    int outputCount = 0;
    return m_cropFromRgb->filter(&inputFrame, outputs, &outputCount, queue, wait_events, event);
}

void RGYFilterOnnxDeint::setOutputFrameProp(RGYFrameInfo *output, const RGYFrameInfo *input) const {
    copyFramePropWithoutRes(output, input);
    output->picstruct = RGY_PICSTRUCT_FRAME;
    output->flags = (RGY_FRAME_FLAGS)(input->flags &
        ~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_BFF | RGY_FRAME_FLAG_RFF_TFF));
}

void RGYFilterOnnxDeint::setBobTimestamp(const RGYFrameInfo *input, RGYFrameInfo **outputs) const {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamOnnxDeint>(m_param);
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

RGY_ERR RGYFilterOnnxDeint::runOcl(const RGYFrameInfo *input, RGYFrameInfo **outputs, int outputCount,
    RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event) {
    auto err = convertToRgb(input, queue, wait_events, nullptr);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: OpenCL YUV-to-RGB conversion failed: %s.\n"), get_err_mes(err));
        return err;
    }
    // 共有バッファへの推論は1フレームにつき1回だけ実行する。
    err = m_ov->inferShared();
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: shared OpenCL inference failed: %s.\n"), get_err_mes(err));
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
    RGYWorkSize local(32, 8);
    RGYWorkSize global(m_width, m_height);
    for (int i = 0; i < outputCount; i++) {
        auto output = &m_frameBuf[i]->frame;
        const int frameA = sourceIndices[i] == 0 ? 1 : 0;
        err = m_program->kernel("onnx_deint_weave_rgb").config(queue, local, global, {}, nullptr).launch(
            m_inputBufCL->mem(), m_outputBufCL->mem(), m_weaveBufCL->mem(), m_width, m_height, frameA);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("onnx-deint: OpenCL weave failed: %s.\n"), get_err_mes(err));
            return err;
        }
        err = convertFromRgb(output, queue, {}, (i == outputCount - 1) ? event : nullptr);
        if (err != RGY_ERR_NONE) return err;
        setOutputFrameProp(output, input);
        outputs[i] = output;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterOnnxDeint::allocTemporalRing(const RGY_CSP csp, const int bitdepth) {
    const size_t plane = (size_t)m_width * m_height;
    for (auto& slot : m_temporalRing) {
        slot.frame = m_cl->createFrameBuffer(m_width, m_height, csp, bitdepth);
        if (!slot.frame) {
            AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to allocate the temporal frame ring.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        slot.rgb.assign(3 * plane, 0.0f);
        slot.tff = m_defaultTff;
        slot.interlaced = false;
    }
    m_framesIn = 0;
    m_frameOut = 0;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterOnnxDeint::addTemporalFrame(const RGYFrameInfo *input, RGYOpenCLQueue& queue,
    const std::vector<RGYOpenCLEvent>& wait_events) {
    auto& slot = m_temporalRing[m_framesIn % m_temporalRing.size()];
    auto err = m_cl->copyFrame(&slot.frame->frame, input, nullptr, queue, wait_events, nullptr);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to copy input into the temporal frame ring: %s.\n"), get_err_mes(err));
        return err;
    }
    copyFrameProp(&slot.frame->frame, input);

    err = convertToRgb(input, queue, {}, nullptr);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to convert input to RGB: %s.\n"), get_err_mes(err));
        return err;
    }
    err = m_inputBufCL->queueMapBuffer(queue, CL_MAP_READ, {}, RGY_CL_MAP_BLOCK_ALL);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to map RGB input tensor: %s.\n"), get_err_mes(err));
        return err;
    }
    std::memcpy(slot.rgb.data(), m_inputBufCL->mappedPtr(), slot.rgb.size() * sizeof(float));
    m_inputBufCL->unmapBuffer(queue);

    slot.tff = m_defaultTff;
    if (input->picstruct & RGY_PICSTRUCT_BFF) {
        slot.tff = false;
    } else if (input->picstruct & RGY_PICSTRUCT_TFF) {
        slot.tff = true;
    }
    slot.interlaced = (input->picstruct & RGY_PICSTRUCT_INTERLACED) != 0;
    m_framesIn++;
    return RGY_ERR_NONE;
}

// フィールドはフレームごとに表示順(0が先, 1が後)で数える。
// 戻り値は0がtop field(フレームの偶数行), 1がbottom field(フレームの奇数行)。
int RGYFilterOnnxDeint::temporalFieldParity(const int fieldIndex) const {
    const auto& slot = m_temporalRing[(fieldIndex / 2) % m_temporalRing.size()];
    const int fieldPos = fieldIndex & 1;
    return slot.tff ? fieldPos : 1 - fieldPos;
}

// 出力フィールドn (= frameIndex*2 + fieldPos) 用の9chテンソルを組み立てる。
//  1. フィールドn-1, n, n+1を使う。端はミラーリング (index<0 -> -index, index>last -> 2*last-index)。
//  2. 各フィールド(h, W, 3)を転置して(W, h, 3)にする。モデルは縦方向のインターレースで学習されているため、
//     モデルのテンソルは(1, 9, W, h) - つまりモデルの高さ=フレーム幅, モデルの幅=フィールド高さとなる。
//  3. 既知フィールドnがtop field(parity==0)ならflip = true。このとき3フィールドすべての行軸
//     (モデルの最終軸)を反転する。
//  4. [prev R,G,B, cur R,G,B, next R,G,B] の順にフィールド単位で9chへ積む。
void RGYFilterOnnxDeint::buildTemporalInput(const int frameIndex, const int fieldPos) {
    const int fieldHeight = m_height / 2;
    const int fieldIndex = frameIndex * 2 + fieldPos;
    const int lastField = (m_framesIn - 1) * 2 + 1;
    const bool flip = temporalFieldParity(fieldIndex) == 0;
    for (int i = 0; i < 3; i++) {
        int refIndex = fieldIndex - 1 + i;
        if (refIndex < 0) refIndex = -refIndex;
        if (refIndex > lastField) refIndex = 2 * lastField - refIndex;
        const auto& slot = m_temporalRing[(refIndex / 2) % m_temporalRing.size()];
        const int parity = temporalFieldParity(refIndex);
        for (int c = 0; c < 3; c++) {
            const float *srcPlane = slot.rgb.data() + (size_t)c * m_width * m_height;
            float *dstPlane = m_inputBuf.data() + (size_t)(i * 3 + c) * m_width * fieldHeight;
            for (int x = 0; x < m_width; x++) {
                float *dstLine = dstPlane + (size_t)x * fieldHeight;
                for (int y = 0; y < fieldHeight; y++) {
                    const int srcRow = 2 * (flip ? (fieldHeight - 1 - y) : y) + parity;
                    dstLine[y] = srcPlane[(size_t)srcRow * m_width + x];
                }
            }
        }
    }
}

// モデル出力(1, 3, W, h)と中央のフィールドを1枚のフレームへ戻す。
//  6. モデルの最終軸を2倍にして合成する: out[..., 0::2] = モデル出力,
//     out[..., 1::2] = 入力したときのままの中央フィールド。
//  7. flipなら2倍にした軸を反転して戻す。
//  8. 転置を戻して(H, W)のフレームにする。
// 奇数側は結局フレームの行yそのものになる - 既知フィールドが元の行にビットイグザクトで戻る。
void RGYFilterOnnxDeint::combineTemporalOutput(const int frameIndex, const int fieldPos, float *dst) const {
    const int fieldHeight = m_height / 2;
    const int fieldIndex = frameIndex * 2 + fieldPos;
    const int parity = temporalFieldParity(fieldIndex);
    const bool flip = (parity == 0);
    const auto& slot = m_temporalRing[frameIndex % m_temporalRing.size()];
    for (int c = 0; c < 3; c++) {
        const float *modelPlane = m_outputBuf.data() + (size_t)c * m_width * fieldHeight;
        const float *midPlane = slot.rgb.data() + (size_t)c * m_width * m_height;
        float *dstPlane = dst + (size_t)c * m_width * m_height;
        for (int y = 0; y < m_height; y++) {
            const int pos = flip ? (m_height - 1 - y) : y;
            float *dstLine = dstPlane + (size_t)y * m_width;
            if ((pos & 1) == 0) {
                const int row = pos / 2;
                for (int x = 0; x < m_width; x++) {
                    dstLine[x] = modelPlane[(size_t)x * fieldHeight + row];
                }
            } else {
                const int row = flip ? (fieldHeight - 1 - pos / 2) : (pos / 2);
                std::memcpy(dstLine, midPlane + (size_t)(2 * row + parity) * m_width, (size_t)m_width * sizeof(float));
            }
        }
    }
}

RGY_ERR RGYFilterOnnxDeint::procTemporalField(const int frameIndex, const int fieldPos, RGYFrameInfo *output,
    RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event) {
    buildTemporalInput(frameIndex, fieldPos);
    auto err = m_ov->infer(m_inputBuf.data(), m_outputBuf.data());
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: inference failed: %s.\n"), get_err_mes(err));
        return err;
    }
    err = m_weaveBufCL->queueMapBuffer(queue, CL_MAP_WRITE, wait_events, RGY_CL_MAP_BLOCK_ALL);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to map RGB output tensor: %s.\n"), get_err_mes(err));
        return err;
    }
    combineTemporalOutput(frameIndex, fieldPos, (float *)m_weaveBufCL->mappedPtr());
    m_weaveBufCL->unmapBuffer(queue);
    return convertFromRgb(output, queue, {}, event);
}

RGY_ERR RGYFilterOnnxDeint::emitTemporalFrame(const int frameIndex, RGYFrameInfo **outputs, int *outputFrameNum,
    RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event) {
    const auto& slot = m_temporalRing[frameIndex % m_temporalRing.size()];
    const auto *source = &slot.frame->frame;
    const bool bob = m_mode == VppOnnxDeintMode::Bob;
    const int outputCount = bob ? 2 : 1;
    for (int i = 0; i < outputCount; i++) {
        auto output = &m_frameBuf[i]->frame;
        RGYOpenCLEvent *frame_event = (i == outputCount - 1) ? event : nullptr;
        if (slot.interlaced) {
            const auto err = procTemporalField(frameIndex, i, output, queue,
                (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>(), frame_event);
            if (err != RGY_ERR_NONE) {
                return err;
            }
        } else {
            const auto err = m_cl->copyFrame(output, source, nullptr, queue,
                (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>(), frame_event);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to copy progressive input: %s.\n"), get_err_mes(err));
                return err;
            }
        }
        setOutputFrameProp(output, source);
        outputs[i] = output;
    }
    *outputFrameNum = outputCount;
    if (bob) {
        setBobTimestamp(source, outputs);
    }
    return RGY_ERR_NONE;
}

// 出力フィールドは前後のフィールドを参照するため、フレームfの出力はf+1が到着してから確定する。
// 残った末尾のフレームはdrain (ptr[0] == nullptr) で1フレームずつ吐き出す。
// パイプラインは出力が0になるまでdrainを繰り返し呼ぶ。
RGY_ERR RGYFilterOnnxDeint::runTemporal(const RGYFrameInfo *input, RGYFrameInfo **outputs, int *outputFrameNum,
    RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event) {
    if (input->ptr[0] != nullptr) {
        auto err = addTemporalFrame(input, queue, wait_events);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        if (m_frameOut + 1 < m_framesIn) {
            const int frameIndex = m_frameOut++;
            return emitTemporalFrame(frameIndex, outputs, outputFrameNum, queue, {}, event);
        }
        return RGY_ERR_NONE;
    }
    if (m_frameOut < m_framesIn) {
        const int frameIndex = m_frameOut++;
        return emitTemporalFrame(frameIndex, outputs, outputFrameNum, queue, wait_events, event);
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterOnnxDeint::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events,
    RGYOpenCLEvent *event) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    ppOutputFrames[1] = nullptr;
    if (!pInputFrame) {
        return RGY_ERR_NONE;
    }
    if (m_temporal) {
        return runTemporal(pInputFrame, ppOutputFrames, pOutputFrameNum, queue, wait_events, event);
    }
    if (!pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }

    const bool bob = m_mode == VppOnnxDeintMode::Bob;
    const int outputCount = bob ? 2 : 1;
    if ((pInputFrame->picstruct & RGY_PICSTRUCT_INTERLACED) == 0) {
        for (int i = 0; i < outputCount; i++) {
            auto output = &m_frameBuf[i]->frame;
            const auto err = m_cl->copyFrame(output, pInputFrame, nullptr, queue,
                (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>(),
                (i == outputCount - 1) ? event : nullptr);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to copy progressive input: %s.\n"), get_err_mes(err));
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

    auto err = convertToRgb(pInputFrame, queue, wait_events, nullptr);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to convert input to RGB: %s.\n"), get_err_mes(err));
        return err;
    }
    err = m_inputBufCL->queueMapBuffer(queue, CL_MAP_READ, {}, RGY_CL_MAP_BLOCK_ALL);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: failed to map RGB input tensor: %s.\n"), get_err_mes(err));
        return err;
    }
    std::memcpy(m_inputBuf.data(), m_inputBufCL->mappedPtr(), m_inputBuf.size() * sizeof(float));
    m_inputBufCL->unmapBuffer(queue);

    err = m_ov->infer(m_inputBuf.data(), m_outputBuf.data());
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx-deint: inference failed: %s.\n"), get_err_mes(err));
        return err;
    }
    err = m_outputBufCL->queueMapBuffer(queue, CL_MAP_WRITE, {}, RGY_CL_MAP_BLOCK_ALL);
    if (err != RGY_ERR_NONE) return err;
    std::memcpy(m_outputBufCL->mappedPtr(), m_outputBuf.data(), m_outputBuf.size() * sizeof(float));
    m_outputBufCL->unmapBuffer(queue);
    bool inputTff = m_defaultTff;
    if (pInputFrame->picstruct & RGY_PICSTRUCT_BFF) {
        inputTff = false;
    } else if (pInputFrame->picstruct & RGY_PICSTRUCT_TFF) {
        inputTff = true;
    }
    const int firstIndex = inputTff ? 0 : 1;
    const int secondIndex = 1 - firstIndex;
    const int sourceIndices[2] = { firstIndex, secondIndex };
    RGYWorkSize local(32, 8);
    RGYWorkSize global(m_width, m_height);
    for (int i = 0; i < outputCount; i++) {
        auto output = &m_frameBuf[i]->frame;
        const int frameIndex = sourceIndices[i];
        err = m_program->kernel("onnx_deint_weave_rgb").config(queue, local, global, {}, nullptr).launch(
            m_inputBufCL->mem(), m_outputBufCL->mem(), m_weaveBufCL->mem(),
            m_width, m_height, frameIndex == 0 ? 1 : 0);
        if (err != RGY_ERR_NONE) return err;
        err = convertFromRgb(output, queue, {}, (i == outputCount - 1) ? event : nullptr);
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
