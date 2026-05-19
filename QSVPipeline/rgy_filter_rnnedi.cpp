// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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

#include "rgy_filter_rnnedi.h"
#include "convert_csp.h"
#include "rgy_filesystem.h"
#include "rgy_resource.h"
#include <algorithm>
#include <fstream>

namespace {

static constexpr std::array<RGYRnnediNSizeDesc, 7> RNNEDI_NSIZE_DESC = {
    RGYRnnediNSizeDesc{  8, 6 },
    RGYRnnediNSizeDesc{ 16, 6 },
    RGYRnnediNSizeDesc{ 32, 6 },
    RGYRnnediNSizeDesc{ 48, 6 },
    RGYRnnediNSizeDesc{  8, 4 },
    RGYRnnediNSizeDesc{ 16, 4 },
    RGYRnnediNSizeDesc{ 32, 4 },
};

static constexpr std::array<int, 5> RNNEDI_NNS_VALUE = { 16, 32, 64, 128, 256 };
static const TCHAR *RNNEDI_DEFAULT_WEIGHT_FILE = _T("nnedi3_weights.bin");

struct RGYRnnediPlaneValueRange {
    int planeRangeMode;
    int valMin;
    int valMax;
};

static const TCHAR *rnnedi_nsize_name(const int nsize) {
    static const TCHAR *names[] = {
        _T("8x6"), _T("16x6"), _T("32x6"), _T("48x6"), _T("8x4"), _T("16x4"), _T("32x4")
    };
    return (0 <= nsize && nsize < (int)_countof(names)) ? names[nsize] : _T("unknown");
}

static RGYRnnediPlaneValueRange rnnediPlaneValueRange(const int clamp, const RGY_CSP csp, const RGY_PLANE plane, const int bitDepth) {
    const bool isAlpha = plane == RGY_PLANE_A;
    const bool isYuv = csp != RGY_CSP_Y8 && RGY_CSP_PLANES[csp] >= 3;
    int planeRangeMode = clamp;
    if (isAlpha) {
        planeRangeMode = 1;
    } else if (clamp != 1 && clamp != 4) {
        if (isYuv) {
            planeRangeMode = (plane == RGY_PLANE_Y) ? 2 : 3;
        } else {
            planeRangeMode = (clamp == 0) ? 2 : clamp;
        }
    }

    const int fullMax = (1 << bitDepth) - 1;
    const int scale = 1 << std::max(0, bitDepth - 8);
    RGYRnnediPlaneValueRange valueRange;
    valueRange.planeRangeMode = planeRangeMode;
    switch (planeRangeMode) {
    case 2:
        valueRange.valMin = 16 * scale;
        valueRange.valMax = 235 * scale;
        break;
    case 3:
        valueRange.valMin = 16 * scale;
        valueRange.valMax = 240 * scale;
        break;
    case 4:
        valueRange.valMin = 16 * scale;
        valueRange.valMax = fullMax;
        break;
    case 1:
    default:
        valueRange.valMin = 0;
        valueRange.valMax = fullMax;
        break;
    }
    return valueRange;
}

static bool rnnediPlaneEnabled(const RGYRnnediParam& prm, const int plane) {
    return 0 <= plane && plane < (int)prm.processPlane.size() && prm.processPlane[plane];
}

static bool rnnediSupportedPlanarCsp(const RGY_CSP csp) {
    switch (csp) {
    case RGY_CSP_Y8:
    case RGY_CSP_Y16:
    case RGY_CSP_YV12:
    case RGY_CSP_YV12_16:
    case RGY_CSP_YUV422:
    case RGY_CSP_YUV422_16:
    case RGY_CSP_YUV444:
    case RGY_CSP_YUV444_16:
        return true;
    default:
        return false;
    }
}

static int rnnediBytesPerSample(const RGY_CSP csp) {
    return (RGY_CSP_BIT_DEPTH[csp] > 8) ? 2 : 1;
}

static int rnnediFindEnabledPlane(const RGYRnnediParam& prm, const RGY_CSP csp, const bool reverse = false) {
    const int planes = RGY_CSP_PLANES[csp];
    for (int i = 0; i < planes; i++) {
        const int plane = reverse ? (planes - 1 - i) : i;
        if (rnnediPlaneEnabled(prm, plane)) {
            return plane;
        }
    }
    return -1;
}

#if defined(_WIN32) || defined(_WIN64)
static tstring rnnediDefaultWeightFilePath(HMODULE hModule) {
    const tstring filename = RNNEDI_DEFAULT_WEIGHT_FILE;
    if (rgy_file_exists(filename.c_str())) {
        return filename;
    }

    const auto modulePath = getModulePath(hModule);
    if (modulePath.length() > 0) {
        const auto moduleDir = PathRemoveFileSpecFixed(modulePath).second;
        const auto path = PathCombineS(moduleDir, filename);
        if (rgy_file_exists(path.c_str())) {
            return path;
        }
    }

    const auto exeDir = getExeDir();
    if (exeDir.length() > 0) {
        const auto path = PathCombineS(exeDir, filename);
        if (rgy_file_exists(path.c_str())) {
            return path;
        }
    }

    return filename;
}
#endif

} // namespace

RGYRnnediParam::RGYRnnediParam() :
    enable(false),
    processPlane{ true, true, true, false },
    field(VPP_RNNEDI_FIELD_BOB),
    nsize(VPP_NNEDI_NSIZE_16x6),
    nns(32),
    quality(VPP_NNEDI_QUALITY_FAST),
    prescreen(2),
    errortype(VPP_NNEDI_ETYPE_ABS),
    doubleHeight(false),
    weightfile(_T("")) {
    clamp = 1;
}

bool RGYRnnediParam::operator==(const RGYRnnediParam& x) const {
    return enable == x.enable
        && processPlane == x.processPlane
        && field == x.field
        && nsize == x.nsize
        && nns == x.nns
        && quality == x.quality
        && prescreen == x.prescreen
        && errortype == x.errortype
        && clamp == x.clamp
        && doubleHeight == x.doubleHeight
        && weightfile == x.weightfile;
}

bool RGYRnnediParam::operator!=(const RGYRnnediParam& x) const {
    return !(*this == x);
}

tstring RGYRnnediParam::print() const {
    const auto nsizeIndex = (int)nsize;
    return strsprintf(
        _T("rnnedi: field %s, nsize %s, nns %d, quality %s\n")
        _T("                         prescreen %d, errortype %s, clamp %d, double_height %s, weight \"%s\""),
        get_cx_desc(list_vpp_rnnedi_field, field),
        rnnedi_nsize_name(nsizeIndex),
        nns,
        get_cx_desc(list_vpp_nnedi_quality, quality),
        prescreen,
        get_cx_desc(list_vpp_nnedi_error_type, errortype),
        clamp,
        doubleHeight ? _T("on") : _T("off"),
        ((weightfile.length()) ? weightfile.c_str() : _T("default")));
}

const RGYRnnediNSizeDesc& rgy_rnnedi_nsize_desc(const int nsize) {
    return RNNEDI_NSIZE_DESC[nsize];
}

int rgy_rnnedi_nns_value(const int nns) {
    return (0 <= nns && nns < (int)RNNEDI_NNS_VALUE.size()) ? RNNEDI_NNS_VALUE[nns] : 0;
}

int rgy_rnnedi_nns_index(const int nns) {
    const auto it = std::find(RNNEDI_NNS_VALUE.begin(), RNNEDI_NNS_VALUE.end(), nns);
    return (it == RNNEDI_NNS_VALUE.end()) ? -1 : (int)std::distance(RNNEDI_NNS_VALUE.begin(), it);
}

RGYFilterParamRnnedi::RGYFilterParamRnnedi() :
    rnnedi(),
    hModule(NULL),
    timebase() {
}

RGYFilterRnnedi::RGYFilterRnnedi(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_weights(),
    m_rnnedi(),
    m_rnnediBuildOptions(),
    m_rnnediPredictorSubgroupSize(0),
    m_refBuf(),
    m_prescreenerWeightBuf(),
    m_predictorWeightBuf(),
    m_workNNBuf(),
    m_numBlocksBuf(),
    m_defaultTff(true) {
    m_name = _T("rnnedi");
}

RGYFilterRnnedi::~RGYFilterRnnedi() {
    close();
}

RGY_ERR RGYFilterRnnedi::validateParam(const RGYRnnediParam& prm) {
    const auto field = (int)prm.field;
    if (field < -2 || 3 < field) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI invalid field=%d.\n"), field);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm.doubleHeight && (field < -1 || 1 < field)) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI double_height=true supports only field=auto,top,bottom.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto nsizeIndex = (int)prm.nsize;
    if (nsizeIndex < 0 || (int)RNNEDI_NSIZE_DESC.size() <= nsizeIndex) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI invalid nsize=%d.\n"), nsizeIndex);
        return RGY_ERR_INVALID_PARAM;
    }
    if (rgy_rnnedi_nns_index(prm.nns) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI invalid nns=%d, expected 16, 32, 64, 128, or 256.\n"), prm.nns);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm.quality != VPP_NNEDI_QUALITY_FAST && prm.quality != VPP_NNEDI_QUALITY_SLOW) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI invalid quality=%d.\n"), (int)prm.quality);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm.clamp < 0 || 4 < prm.clamp) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI invalid clamp=%d, expected 0-4.\n"), prm.clamp);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm.prescreen < 2 || 4 < prm.prescreen) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI unsupported prescreen=%d, supported prescreen values are 2, 3, and 4; prescreen=0/1 use an unsupported prescreener path.\n"), prm.prescreen);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm.errortype < VPP_NNEDI_ETYPE_ABS || VPP_NNEDI_ETYPE_MAX <= prm.errortype) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI invalid errortype=%d.\n"), (int)prm.errortype);
        return RGY_ERR_INVALID_PARAM;
    }
    if (std::none_of(prm.processPlane.begin(), prm.processPlane.end(), [](const bool enabled) { return enabled; })) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI requires at least one target plane to be enabled.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

std::shared_ptr<const std::vector<uint8_t>> RGYFilterRnnedi::readWeights(const tstring& weightFile, HMODULE hModule) {
    const auto expectedFileSize = RGYRnnediParam::WEIGHTS_FILE_SIZE;
    uint64_t weightFileSize = 0;

#if !(defined(_WIN32) || defined(_WIN64))
    if (weightFile.length() == 0) {
        void *pDataPtr = nullptr;
        weightFileSize = getEmbeddedResource(&pDataPtr, _T("NNEDI_WEIGHTBIN"), _T("EXE_DATA"), hModule);
        if (pDataPtr == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get RNNEDI weights data.\n"));
            return nullptr;
        }
        if (expectedFileSize != weightFileSize) {
            AddMessage(RGY_LOG_ERROR, _T("RNNEDI weights data has unexpected size %lld [expected: %u].\n"),
                (long long int)weightFileSize, expectedFileSize);
            return nullptr;
        }
        return std::make_shared<const std::vector<uint8_t>>(
            (const uint8_t *)pDataPtr, (const uint8_t *)pDataPtr + weightFileSize);
    }
    const auto weightFilePath = weightFile;
#else
    const auto weightFilePath = (weightFile.length() > 0) ? weightFile : rnnediDefaultWeightFilePath(hModule);
#endif

    if (!rgy_file_exists(weightFilePath.c_str())) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI weight file \"%s\" does not exist.\n"), weightFilePath.c_str());
        return nullptr;
    }
    if (!rgy_get_filesize(weightFilePath.c_str(), &weightFileSize)) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to get filesize of RNNEDI weight file \"%s\".\n"), weightFilePath.c_str());
        return nullptr;
    }
    if (weightFileSize != expectedFileSize) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI weights file \"%s\" has unexpected file size %lld [expected: %u].\n"),
            weightFilePath.c_str(), (long long int)weightFileSize, expectedFileSize);
        return nullptr;
    }

    auto weights = std::make_shared<std::vector<uint8_t>>(weightFileSize);
    std::ifstream fin(weightFilePath, std::ios::in | std::ios::binary);
    if (!fin.good()) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to open RNNEDI weights file \"%s\".\n"), weightFilePath.c_str());
        return nullptr;
    }
    if (fin.read((char *)weights->data(), weights->size()).gcount() != (int64_t)weights->size()) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to read RNNEDI weights file \"%s\".\n"), weightFilePath.c_str());
        return nullptr;
    }
    return weights;
}

RGY_ERR RGYFilterRnnedi::initParams(const std::shared_ptr<RGYFilterParamRnnedi> prm) {
    auto err = validateParam(prm->rnnedi);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    if (bitDepth != 8 && bitDepth != 16) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI supports only 8-bit or 16-bit planar input.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    m_weights = readWeights(prm->rnnedi.weightfile, prm->hModule);
    if (!m_weights) {
        return RGY_ERR_INVALID_PARAM;
    }
    if ((m_weights->size() % sizeof(float)) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI weights data is not float-aligned.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto selfCheck = rgy_filter_rnnedi_weights_self_check();
    if (!selfCheck.success) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI weights self-check failed: %s.\n"), char_to_tstring(selfCheck.message).c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    RGYFilterRnnediWeightsParam weightsParam;
    weightsParam.nsize = prm->rnnedi.nsize;
    weightsParam.nns = prm->rnnedi.nns;
    weightsParam.quality = prm->rnnedi.quality;
    weightsParam.prescreen = prm->rnnedi.prescreen;
    weightsParam.errortype = prm->rnnedi.errortype;
    weightsParam.bitsPerPixel = bitDepth;
    std::string weightsError;
    if (!rgy_filter_rnnedi_transform_weights(m_transformedWeights,
        reinterpret_cast<const float *>(m_weights->data()), m_weights->size() / sizeof(float), weightsParam, &weightsError)) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI weights transform failed: %s.\n"), char_to_tstring(weightsError).c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_transformedWeights.prescreenerFp32.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI fp32 prescreener weights are empty.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_transformedWeights.predictorFp32.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI fp32 predictor weights are empty.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRnnedi::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamRnnedi>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    auto err = initParams(prm);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    if (bitDepth != 8 && bitDepth != 16) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI supports only 8-bit or 16-bit planar input.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (!rnnediSupportedPlanarCsp(prm->frameIn.csp)) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI supports only planar Y/YV12/YUV422/YUV444 8-bit or 16-bit input.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (rnnediFindEnabledPlane(prm->rnnedi, prm->frameIn.csp) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI requires at least one target plane present in the input format.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->rnnedi.doubleHeight) {
        const int planes = RGY_CSP_PLANES[prm->frameIn.csp];
        for (int iplane = 0; iplane < planes; iplane++) {
            if (!rnnediPlaneEnabled(prm->rnnedi, iplane)) {
                AddMessage(RGY_LOG_ERROR, _T("RNNEDI double_height=true changes output height, so all existing input planes must be enabled; plane %d is disabled.\n"), iplane);
                return RGY_ERR_INVALID_PARAM;
            }
        }
    }
    if ((prm->frameIn.height & 1) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("RNNEDI requires even input height.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    m_defaultTff = (prm->frameIn.picstruct & RGY_PICSTRUCT_BFF) == 0;
    RGYRnnediTopology topology;
    err = rgy_rnnedi_resolve_topology(&topology, (int)prm->rnnedi.field, m_defaultTff);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    if (prm->rnnedi.doubleHeight) {
        prm->frameOut.height *= 2;
    }
    prm->baseFps *= topology.fpsMultiplier;
    m_pathThrough &= ~FILTER_PATHTHROUGH_PICSTRUCT;
    if (topology.doubleRate) {
        m_pathThrough &= ~FILTER_PATHTHROUGH_TIMESTAMP;
    }

    const auto &layout = m_transformedWeights.layout;
    const auto typeName = (bitDepth > 8) ? "ushort" : "uchar";
    m_rnnediBuildOptions = strsprintf("-D Type=%s -D Type2=%s2 -D Type4=%s4 -D Type8=%s8 -D RNNEDI_BIT_DEPTH=%d"
        " -D RNNEDI_PRED_XDIA=%d"
        " -D RNNEDI_PRED_YDIA=%d"
        " -D RNNEDI_PRED_K=%d"
        " -D RNNEDI_PRED_NNS=%d"
        " -D RNNEDI_PRED_QUAL=%d",
        typeName,
        typeName,
        typeName,
        typeName,
        bitDepth,
        layout.xdia,
        layout.ydia,
        layout.xdia * layout.ydia,
        layout.neurons,
        (int)prm->rnnedi.quality);
    m_rnnediPredictorSubgroupSize = 0;
    AddMessage(RGY_LOG_DEBUG, _T("Starting async build for RGY_FILTER_RNNEDI_CL: %s\n"),
        char_to_tstring(m_rnnediBuildOptions).c_str());
    m_rnnedi.set(m_cl->buildResourceAsync(_T("RGY_FILTER_RNNEDI_CL"), _T("EXE_DATA"), m_rnnediBuildOptions.c_str()));
    setFilterInfo(prm->rnnedi.print());

    const int outputSlots = topology.frameMultiplier;
    err = AllocFrameBuf(prm->frameOut, outputSlots);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output frames: %s.\n"), get_err_mes(err));
        return err;
    }
    RGYFrameInfo refFrame = prm->frameOut;
    refFrame.width += RGY_RNNEDI_HPAD * 4;
    refFrame.height = (refFrame.height >> 1) + RGY_RNNEDI_VPAD * 4;
    m_refBuf.clear();
    for (int i = 0; i < outputSlots; i++) {
        auto ref = m_cl->createFrameBuffer(refFrame);
        if (!ref) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate RNNEDI ref buffer %d.\n"), i);
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_refBuf.push_back(std::move(ref));
    }

    m_prescreenerWeightBuf = m_cl->copyDataToBuffer(m_transformedWeights.prescreenerFp32.data(),
        m_transformedWeights.prescreenerFp32.size() * sizeof(m_transformedWeights.prescreenerFp32[0]), CL_MEM_READ_ONLY, m_cl->queue().get());
    if (!m_prescreenerWeightBuf) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate RNNEDI prescreener weights buffer.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    m_predictorWeightBuf = m_cl->copyDataToBuffer(m_transformedWeights.predictorFp32.data(),
        m_transformedWeights.predictorFp32.size() * sizeof(m_transformedWeights.predictorFp32[0]), CL_MEM_READ_ONLY, m_cl->queue().get());
    if (!m_predictorWeightBuf) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate RNNEDI predictor weights buffer.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }

    int maxWidth4 = 0;
    int maxHeight = 0;
    const auto planes = RGY_CSP_PLANES[prm->frameIn.csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto plane = getPlane(&prm->frameOut, (RGY_PLANE)iplane);
        maxWidth4 = std::max(maxWidth4, (plane.width + 3) >> 2);
        maxHeight = std::max(maxHeight, plane.height >> 1);
    }
    const int preBlockW = 32;
    const int preBlockH = 16;
    const int blocksX = (maxWidth4 + preBlockW - 1) / preBlockW;
    const int blocksY = (maxHeight + preBlockH - 1) / preBlockH;
    const size_t numBlocks = std::max(1, blocksX * blocksY);
    const size_t candidateMaskBytes = numBlocks * preBlockW * preBlockH;
    const size_t numBlocksBytes = numBlocks * sizeof(int);
    m_workNNBuf.clear();
    m_numBlocksBuf.clear();
    for (int i = 0; i < outputSlots * planes; i++) {
        auto workNN = m_cl->createBuffer(candidateMaskBytes, CL_MEM_READ_WRITE);
        auto numBlock = m_cl->createBuffer(numBlocksBytes, CL_MEM_READ_WRITE);
        if (!workNN || !numBlock) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate RNNEDI prescreen work buffers.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_workNNBuf.push_back(std::move(workNN));
        m_numBlocksBuf.push_back(std::move(numBlock));
    }

    m_param = pParam;
    return RGY_ERR_NONE;
}

bool RGYFilterRnnedi::getInputTff(const RGYFrameInfo *frame) const {
    if (frame) {
        if (frame->picstruct & RGY_PICSTRUCT_BFF) {
            return false;
        }
        if (frame->picstruct & RGY_PICSTRUCT_TFF) {
            return true;
        }
    }
    return m_defaultTff;
}

void RGYFilterRnnedi::setDoubleRateTimestamp(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames) const {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamRnnedi>(m_param);
    auto frameDuration = pInputFrame->duration;
    if (frameDuration == 0 && prm && prm->timebase.is_valid()) {
        frameDuration = (decltype(frameDuration))((prm->timebase.inv() / prm->baseFps * 2).qdouble() + 0.5);
    }
    ppOutputFrames[0]->timestamp    = pInputFrame->timestamp;
    ppOutputFrames[0]->duration     = (frameDuration + 1) / 2;
    ppOutputFrames[1]->timestamp    = ppOutputFrames[0]->timestamp + ppOutputFrames[0]->duration;
    ppOutputFrames[1]->duration     = frameDuration - ppOutputFrames[0]->duration;
    ppOutputFrames[0]->inputFrameId = pInputFrame->inputFrameId;
    ppOutputFrames[1]->inputFrameId = pInputFrame->inputFrameId;
}

RGY_ERR RGYFilterRnnedi::prepareFieldReference(const RGYFrameInfo *pInputFrame, int outputSlot, const RGYRnnediFrameMap& frameMap,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!m_rnnedi.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build/load RGY_FILTER_RNNEDI_CL (options: %s).\n"),
            char_to_tstring(m_rnnediBuildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }
    if (outputSlot < 0 || outputSlot >= (int)m_frameBuf.size() || outputSlot >= (int)m_refBuf.size()) {
        return RGY_ERR_INVALID_PARAM;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamRnnedi>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_PARAM;
    }
    const bool doubleHeight = prm->rnnedi.doubleHeight;
    auto pOutputFrame = &m_frameBuf[outputSlot]->frame;
    const auto planes = RGY_CSP_PLANES[pInputFrame->csp];
    const int firstEnabledPlane = rnnediFindEnabledPlane(prm->rnnedi, pInputFrame->csp);
    const int lastEnabledPlane = rnnediFindEnabledPlane(prm->rnnedi, pInputFrame->csp, true);
    for (int iplane = 0; iplane < planes; iplane++) {
        if (!rnnediPlaneEnabled(prm->rnnedi, iplane)) {
            continue;
        }
        const auto srcPlane = getPlane(pInputFrame, (RGY_PLANE)iplane);
        auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        auto refPlane = getPlane(&m_refBuf[outputSlot]->frame, (RGY_PLANE)iplane);
        if (srcPlane.ptr[0] == nullptr || dstPlane.ptr[0] == nullptr || refPlane.ptr[0] == nullptr) {
            continue;
        }
        const auto dstOffset = frameMap.sourceFieldOffset * dstPlane.pitch[0];
        const auto srcOffset = doubleHeight ? 0 : frameMap.sourceFieldOffset * srcPlane.pitch[0];
        const auto srcPitch = doubleHeight ? srcPlane.pitch[0] : srcPlane.pitch[0] * 2;
        const auto fieldHeight = doubleHeight ? srcPlane.height : (srcPlane.height >> 1);
        const auto refBaseHpad = (refPlane.width - srcPlane.width) >> 1;
        const auto refBaseVpad = (refPlane.height - fieldHeight) >> 1;
        const auto refOffset = refBaseVpad * refPlane.pitch[0] + refBaseHpad * rnnediBytesPerSample(pInputFrame->csp);
        const auto mirrorHpad = RGY_RNNEDI_HPAD;
        const auto mirrorVpad = RGY_RNNEDI_VPAD;
        RGYWorkSize local(32, 8);
        RGYWorkSize global((srcPlane.width + mirrorHpad * 2 + 3) >> 2, fieldHeight + mirrorVpad * 2);
        const auto &waitHere = (iplane == firstEnabledPlane) ? wait_events : std::vector<RGYOpenCLEvent>();
        const auto kernelName = "kernel_rnnedi_pad_ref_and_copy_half_scalar";
        auto err = m_rnnedi.get()->kernel(kernelName).config(queue, local, global, waitHere, (iplane == lastEnabledPlane) ? event : nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0] * 2, (int)dstOffset,
            (cl_mem)refPlane.ptr[0], refPlane.pitch[0], (int)refOffset,
            (cl_mem)srcPlane.ptr[0], (int)srcPitch, (int)srcOffset,
            srcPlane.width, fieldHeight,
            mirrorHpad, mirrorVpad);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"), char_to_tstring(kernelName).c_str(), iplane, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRnnedi::classifyPixelsAndSeedOutput(const RGYFrameInfo *pInputFrame, int outputSlot, const RGYRnnediFrameMap& frameMap,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!m_rnnedi.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build/load RGY_FILTER_RNNEDI_CL (options: %s).\n"),
            char_to_tstring(m_rnnediBuildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }
    if (!m_prescreenerWeightBuf || outputSlot < 0 || outputSlot >= (int)m_frameBuf.size() || outputSlot >= (int)m_refBuf.size()) {
        return RGY_ERR_INVALID_PARAM;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamRnnedi>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_PARAM;
    }
    auto pOutputFrame = &m_frameBuf[outputSlot]->frame;
    const auto planes = RGY_CSP_PLANES[pInputFrame->csp];
    const int firstEnabledPlane = rnnediFindEnabledPlane(prm->rnnedi, pInputFrame->csp);
    const int lastEnabledPlane = rnnediFindEnabledPlane(prm->rnnedi, pInputFrame->csp, true);
    for (int iplane = 0; iplane < planes; iplane++) {
        if (!rnnediPlaneEnabled(prm->rnnedi, iplane)) {
            continue;
        }
        auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        auto refPlane = getPlane(&m_refBuf[outputSlot]->frame, (RGY_PLANE)iplane);
        if (dstPlane.ptr[0] == nullptr || refPlane.ptr[0] == nullptr) {
            continue;
        }
        const int workIndex = outputSlot * planes + iplane;
        if (workIndex < 0 || workIndex >= (int)m_workNNBuf.size() || workIndex >= (int)m_numBlocksBuf.size()) {
            return RGY_ERR_INVALID_PARAM;
        }

        const auto refBaseHpad = (refPlane.width - dstPlane.width) >> 1;
        const auto refBaseVpad = (refPlane.height - (dstPlane.height >> 1)) >> 1;
        const int bytesPerSample = rnnediBytesPerSample(pInputFrame->csp);
        const auto refOriginalOffset = (refBaseVpad + RGY_RNNEDI_VPAD) * refPlane.pitch[0] + (refBaseHpad + RGY_RNNEDI_HPAD) * bytesPerSample;
        const auto refEvalOffset = refOriginalOffset + frameMap.evalRefOffsetY * refPlane.pitch[0] - refPlane.pitch[0] - 8 * bytesPerSample;
        const auto dstOffset = (int)frameMap.generateField * dstPlane.pitch[0];
        const int width4 = (dstPlane.width + 3) >> 2;
        const int height = dstPlane.height >> 1;
        const auto valueRange = rnnediPlaneValueRange(prm->rnnedi.clamp, pInputFrame->csp, (RGY_PLANE)iplane, RGY_CSP_BIT_DEPTH[pInputFrame->csp]);

        RGYWorkSize local(32, 16);
        RGYWorkSize global(((width4 + 31) / 32) * 32, ((height + 15) / 16) * 16);
        const auto &waitHere = (iplane == firstEnabledPlane) ? wait_events : std::vector<RGYOpenCLEvent>();
        auto err = m_rnnedi.get()->kernel("kernel_rnnedi_prescreen_cubic").config(queue, local, global, waitHere, (iplane == lastEnabledPlane) ? event : nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0] * 2, (int)dstOffset,
            (cl_mem)refPlane.ptr[0], refPlane.pitch[0], (int)refEvalOffset,
            (cl_mem)m_prescreenerWeightBuf->mem(),
            (cl_mem)m_workNNBuf[workIndex]->mem(), (cl_mem)m_numBlocksBuf[workIndex]->mem(),
            width4, dstPlane.width, height, valueRange.valMin, valueRange.valMax);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_rnnedi_prescreen_cubic (plane %d): %s.\n"), iplane, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRnnedi::resolveClassifiedPixels(const RGYFrameInfo *pInputFrame, int outputSlot, const RGYRnnediFrameMap& frameMap,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto rnnediProgram = m_rnnedi.get();
    if (!rnnediProgram) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build/load RGY_FILTER_RNNEDI_CL (options: %s).\n"),
            char_to_tstring(m_rnnediBuildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }
    if (!m_predictorWeightBuf || outputSlot < 0 || outputSlot >= (int)m_frameBuf.size() || outputSlot >= (int)m_refBuf.size()) {
        return RGY_ERR_INVALID_PARAM;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamRnnedi>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_PARAM;
    }

    auto pOutputFrame = &m_frameBuf[outputSlot]->frame;
    const auto planes = RGY_CSP_PLANES[pInputFrame->csp];
    const int firstEnabledPlane = rnnediFindEnabledPlane(prm->rnnedi, pInputFrame->csp);
    const int lastEnabledPlane = rnnediFindEnabledPlane(prm->rnnedi, pInputFrame->csp, true);
    const auto kernelName = "kernel_rnnedi_predictor_network";
    RGYWorkSize local(16, 32);
    if (m_rnnediPredictorSubgroupSize == 0) {
        const auto subgroupSize = (int)rnnediProgram->kernel(kernelName).config(queue, local, local).subGroupSize();
        if (subgroupSize == 16 || subgroupSize == 32) {
            const auto subgroupBuildOptions = m_rnnediBuildOptions + strsprintf(
                " -cl-std=CL2.0 -D RNNEDI_PRED_SUBGROUP_OPT=1 -D RNNEDI_PRED_SUBGROUP_SIZE=%d",
                subgroupSize);
            AddMessage(RGY_LOG_DEBUG, _T("Rebuilding RGY_FILTER_RNNEDI_CL with predictor subgroup size %d.\n"), subgroupSize);
            m_rnnedi.set(m_cl->buildResourceAsync(_T("RGY_FILTER_RNNEDI_CL"), _T("EXE_DATA"), subgroupBuildOptions.c_str()));
            m_rnnediBuildOptions = subgroupBuildOptions;
            m_rnnediPredictorSubgroupSize = subgroupSize;
            rnnediProgram = m_rnnedi.get();
            if (!rnnediProgram) {
                AddMessage(RGY_LOG_ERROR, _T("failed to build/load subgroup RGY_FILTER_RNNEDI_CL (options: %s).\n"),
                    char_to_tstring(m_rnnediBuildOptions).c_str());
                return RGY_ERR_OPENCL_CRUSH;
            }
        } else {
            m_rnnediPredictorSubgroupSize = -1;
            AddMessage(RGY_LOG_DEBUG, _T("RNNEDI predictor subgroup optimization skipped: subgroup size %d.\n"), subgroupSize);
        }
    }
    for (int iplane = 0; iplane < planes; iplane++) {
        if (!rnnediPlaneEnabled(prm->rnnedi, iplane)) {
            continue;
        }
        auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        auto refPlane = getPlane(&m_refBuf[outputSlot]->frame, (RGY_PLANE)iplane);
        if (dstPlane.ptr[0] == nullptr || refPlane.ptr[0] == nullptr) {
            continue;
        }
        const int workIndex = outputSlot * planes + iplane;
        if (workIndex < 0 || workIndex >= (int)m_workNNBuf.size() || workIndex >= (int)m_numBlocksBuf.size()) {
            return RGY_ERR_INVALID_PARAM;
        }

        const auto refBaseHpad = (refPlane.width - dstPlane.width) >> 1;
        const auto refBaseVpad = (refPlane.height - (dstPlane.height >> 1)) >> 1;
        const int bytesPerSample = rnnediBytesPerSample(pInputFrame->csp);
        const auto refOriginalOffset = (refBaseVpad + RGY_RNNEDI_VPAD) * refPlane.pitch[0] + (refBaseHpad + RGY_RNNEDI_HPAD) * bytesPerSample;
        const auto refEvalOffset = refOriginalOffset + frameMap.evalRefOffsetY * refPlane.pitch[0]
            - (((m_transformedWeights.layout.ydia >> 1) - 1) * refPlane.pitch[0] + ((m_transformedWeights.layout.xdia >> 1) - 1) * bytesPerSample);
        const auto dstOffset = (int)frameMap.generateField * dstPlane.pitch[0];
        const int width4 = (dstPlane.width + 3) >> 2;
        const int height = dstPlane.height >> 1;
        const auto valueRange = rnnediPlaneValueRange(prm->rnnedi.clamp, pInputFrame->csp, (RGY_PLANE)iplane, RGY_CSP_BIT_DEPTH[pInputFrame->csp]);

        RGYWorkSize global(((width4 + 31) / 32) * 16, ((height + 15) / 16) * 32);
        const auto &waitHere = (iplane == firstEnabledPlane) ? wait_events : std::vector<RGYOpenCLEvent>();
        auto err = rnnediProgram->kernel(kernelName).config(queue, local, global, waitHere, (iplane == lastEnabledPlane) ? event : nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0] * 2, (int)dstOffset,
            (cl_mem)refPlane.ptr[0], refPlane.pitch[0], (int)refEvalOffset,
            (cl_mem)m_workNNBuf[workIndex]->mem(), (cl_mem)m_numBlocksBuf[workIndex]->mem(),
            (cl_mem)m_predictorWeightBuf->mem(),
            width4, height, valueRange.valMin, valueRange.valMax);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"), char_to_tstring(kernelName).c_str(), iplane, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRnnedi::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    ppOutputFrames[1] = nullptr;
    if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
        return RGY_ERR_NONE;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamRnnedi>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_PARAM;
    }
    RGYRnnediTopology topology;
    auto err = rgy_rnnedi_resolve_topology(&topology, (int)prm->rnnedi.field, getInputTff(pInputFrame));
    if (err != RGY_ERR_NONE) {
        return err;
    }

    const bool doubleHeight = prm->rnnedi.doubleHeight;
    const int outputFrames = topology.frameMultiplier;
    for (int i = 0; i < outputFrames; i++) {
        if (!doubleHeight) {
            err = m_cl->copyFrame(&m_frameBuf[i]->frame, pInputFrame, nullptr, queue, (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>(), nullptr, RGYFrameCopyMode::FRAME, "rnnedi.output_init_copy");
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to initialize RNNEDI output frame %d: %s.\n"), i, get_err_mes(err));
                return err;
            }
        }
        RGYRnnediFrameMap frameMap;
        err = rgy_rnnedi_map_output_frame(&frameMap, topology, i);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        err = prepareFieldReference(pInputFrame, i, frameMap, queue, (doubleHeight && i == 0) ? wait_events : std::vector<RGYOpenCLEvent>(), nullptr);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        err = classifyPixelsAndSeedOutput(pInputFrame, i, frameMap, queue, {}, nullptr);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        err = resolveClassifiedPixels(pInputFrame, i, frameMap, queue, {}, (i == outputFrames - 1) ? event : nullptr);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        auto pOut = &m_frameBuf[i]->frame;
        copyFramePropWithoutRes(pOut, pInputFrame);
        pOut->picstruct = RGY_PICSTRUCT_FRAME;
        ppOutputFrames[i] = pOut;
    }
    *pOutputFrameNum = outputFrames;
    if (topology.doubleRate) {
        setDoubleRateTimestamp(pInputFrame, ppOutputFrames);
    }
    return RGY_ERR_NONE;
}

void RGYFilterRnnedi::close() {
    m_weights.reset();
    m_transformedWeights = RGYFilterRnnediTransformedWeights();
    m_rnnedi.clear();
    m_rnnediBuildOptions.clear();
    m_rnnediPredictorSubgroupSize = 0;
    m_refBuf.clear();
    m_prescreenerWeightBuf.reset();
    m_predictorWeightBuf.reset();
    m_workNNBuf.clear();
    m_numBlocksBuf.clear();
    m_frameBuf.clear();
}
