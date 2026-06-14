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

#include "rgy_filter_rtgmc_edi.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

namespace {
static constexpr int RTGMC_EDI_BLOCK_X = 32;
static constexpr int RTGMC_EDI_BLOCK_Y = 8;

static bool rtgmcEdiModeIsBob(const VppRtgmcEdiMode mode) {
    return mode == VppRtgmcEdiMode::Bob || mode == VppRtgmcEdiMode::BobChromaMerge;
}

static bool rtgmcEdiModeIsLightweight(const VppRtgmcEdiMode mode) {
    return rtgmcEdiModeIsBob(mode)
        || mode == VppRtgmcEdiMode::Yadif
        || mode == VppRtgmcEdiMode::cYadif
        || mode == VppRtgmcEdiMode::RepYadif
        || mode == VppRtgmcEdiMode::RepcYadif;
}

static bool rtgmcEdiModeUsesTemporalYadif(const VppRtgmcEdiMode mode) {
    return mode == VppRtgmcEdiMode::Yadif
        || mode == VppRtgmcEdiMode::cYadif
        || mode == VppRtgmcEdiMode::RepYadif
        || mode == VppRtgmcEdiMode::RepcYadif;
}

static const char *rtgmcEdiKernelName(const VppRtgmcEdiMode mode) {
    switch (mode) {
    case VppRtgmcEdiMode::Bob:
        return "kernel_rtgmc_edi_bob";
    case VppRtgmcEdiMode::Yadif:
        return "kernel_rtgmc_edi_yadif";
    case VppRtgmcEdiMode::cYadif:
        return "kernel_rtgmc_edi_cyadif";
    case VppRtgmcEdiMode::TDeint:
        return "kernel_rtgmc_edi_tdeint";
    case VppRtgmcEdiMode::RepYadif:
        return "kernel_rtgmc_edi_rep_yadif";
    case VppRtgmcEdiMode::RepcYadif:
        return "kernel_rtgmc_edi_rep_cyadif";
    case VppRtgmcEdiMode::Passthrough:
    default:
        return "kernel_rtgmc_edi_passthrough";
    }
}

static const TCHAR *rtgmcEdiModeDetail(const VppRtgmcEdiMode mode) {
    switch (mode) {
    case VppRtgmcEdiMode::Bob:
        return _T(" (bob chroma + selected luma)");
    case VppRtgmcEdiMode::Yadif:
        return _T(" (Yadifmod2 temporal)");
    case VppRtgmcEdiMode::cYadif:
        return _T(" (Yadif temporal)");
    case VppRtgmcEdiMode::TDeint:
        return _T(" (TDeint cubic)");
    case VppRtgmcEdiMode::RepYadif:
        return _T(" (Yadifmod2 planar)");
    case VppRtgmcEdiMode::RepcYadif:
        return _T(" (Yadif temporal + Repair mode2)");
    case VppRtgmcEdiMode::NNEDI3:
        return _T(" (nnedi3 adapter)");
    default:
        return _T("");
    }
}

static void rtgmcEdiCopyFrameProps(RGYFrameInfo *dst, const RGYFrameInfo *src, const VppRtgmcEdiMode mode) {
    copyFramePropWithoutRes(dst, src);
    dst->picstruct = rtgmcEdiModeIsLightweight(mode) ? RGY_PICSTRUCT_FRAME : src->picstruct;
}

static std::string rtgmcEdiDumpJoinPath(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    const auto last = dir.back();
    return (last == '/' || last == '\\') ? dir + name : dir + "\\" + name;
}

static std::string rtgmcEdiDumpName(const char *label, int frameIndex, int inputFrameId, int targetField) {
    std::ostringstream oss;
    oss << "edi_" << std::setw(4) << std::setfill('0') << frameIndex
        << "_tf" << targetField << "_" << label << "_if" << inputFrameId << ".y4m";
    return oss.str();
}

static std::string rtgmcEdiJsonEscape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (const auto c : value) {
        if (c == '\\' || c == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(c);
    }
    return escaped;
}

static int rtgmcEdiEnvInt(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    char *endptr = nullptr;
    const long parsed = std::strtol(value, &endptr, 10);
    return (endptr != value && parsed > 0) ? (int)std::min<long>(parsed, std::numeric_limits<int>::max()) : 0;
}

static std::string rtgmcEdiEnvString(const char *name) {
    const char *value = std::getenv(name);
    return (value != nullptr && value[0] != '\0') ? value : "";
}

static RGY_ERR rtgmcEdiDumpY4MFrame(const RGYFrameInfo *frame, const std::string& path,
    std::shared_ptr<RGYOpenCLContext> cl, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &waitEvents) {
    if (frame == nullptr || frame->ptr[0] == nullptr) {
        return RGY_ERR_NONE;
    }
    const int bitdepth = RGY_CSP_BIT_DEPTH[frame->csp];
    if (bitdepth > 8 || (RGY_CSP_CHROMA_FORMAT[frame->csp] != RGY_CHROMAFMT_YUV420 && RGY_CSP_PLANES[frame->csp] != 1)) {
        return RGY_ERR_NONE;
    }
    const auto planeY = getPlane(frame, RGY_PLANE_Y);
    if (planeY.ptr[0] == nullptr || planeY.width <= 0 || planeY.height <= 0) {
        return RGY_ERR_INVALID_CALL;
    }
    std::vector<uint8_t> hostY((size_t)planeY.width * planeY.height);
    RGYFrameInfo hostFrame(planeY.width, planeY.height, RGY_CSP_Y8, 8, frame->picstruct, RGY_MEM_TYPE_CPU);
    hostFrame.ptr[0] = hostY.data();
    hostFrame.pitch[0] = planeY.width;

    RGYOpenCLEvent readEvent;
    auto err = cl->copyPlane(&hostFrame, &planeY, nullptr, queue, waitEvents, &readEvent);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    err = readEvent.wait();
    if (err != RGY_ERR_NONE) {
        return err;
    }

    std::ofstream dump(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!dump) {
        return RGY_ERR_FILE_OPEN;
    }
    dump << "YUV4MPEG2 W" << hostFrame.width << " H" << hostFrame.height << " F30000:1001 Ip A0:0 C420jpeg\n";
    dump << "FRAME\n";
    for (int y = 0; y < hostFrame.height; y++) {
        dump.write(reinterpret_cast<const char *>(hostFrame.ptr[0] + (size_t)y * hostFrame.pitch[0]), hostFrame.width);
    }
    const int chromaWidth = (hostFrame.width + 1) >> 1;
    const int chromaHeight = (hostFrame.height + 1) >> 1;
    std::vector<uint8_t> neutralUV((size_t)chromaWidth * chromaHeight, 128);
    dump.write(reinterpret_cast<const char *>(neutralUV.data()), neutralUV.size());
    dump.write(reinterpret_cast<const char *>(neutralUV.data()), neutralUV.size());
    return dump ? RGY_ERR_NONE : RGY_ERR_FILE_OPEN;
}

static RGY_ERR rtgmcEdiDumpTrace(const std::string& dumpDir, const std::string& jsonlPath, int frameIndex, int targetField,
    const RGYFrameInfo *bob, const RGYFrameInfo *srcPrev, const RGYFrameInfo *srcCur, const RGYFrameInfo *srcNext,
    const RGYFrameInfo *out, std::shared_ptr<RGYOpenCLContext> cl, RGYOpenCLQueue &queue,
    const std::vector<RGYOpenCLEvent> &waitEvents) {
    struct DumpItem {
        const char *label;
        const RGYFrameInfo *frame;
    };
    const DumpItem items[] = {
        { "bob", bob },
        { "src_prev", srcPrev },
        { "src_cur", srcCur },
        { "src_next", srcNext },
        { "out", out },
    };
    std::vector<std::pair<std::string, std::string>> dumped;
    for (const auto& item : items) {
        if (item.frame == nullptr || item.frame->ptr[0] == nullptr) {
            continue;
        }
        const auto name = rtgmcEdiDumpName(item.label, frameIndex, item.frame->inputFrameId, targetField);
        const auto path = rtgmcEdiDumpJoinPath(dumpDir, name);
        auto err = rtgmcEdiDumpY4MFrame(item.frame, path, cl, queue, waitEvents);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        dumped.push_back({ item.label, path });
    }

    if (!jsonlPath.empty()) {
        std::ofstream jsonl(jsonlPath, std::ios::out | std::ios::binary | std::ios::app);
        if (!jsonl) {
            return RGY_ERR_FILE_OPEN;
        }
        jsonl << "{\"frame\":" << frameIndex
              << ",\"target_field\":" << targetField
              << ",\"bob_input_frame_id\":" << (bob ? bob->inputFrameId : -1)
              << ",\"src_prev_input_frame_id\":" << (srcPrev ? srcPrev->inputFrameId : -1)
              << ",\"src_cur_input_frame_id\":" << (srcCur ? srcCur->inputFrameId : -1)
              << ",\"src_next_input_frame_id\":" << (srcNext ? srcNext->inputFrameId : -1)
              << ",\"dumps\":{";
        for (size_t i = 0; i < dumped.size(); i++) {
            jsonl << (i ? "," : "")
                  << "\"" << dumped[i].first << "\":\"" << rtgmcEdiJsonEscape(dumped[i].second) << "\"";
        }
        jsonl << "}}\n";
    }
    return RGY_ERR_NONE;
}
}

void RGYFilterRtgmcEdi::loadDumpEnv() {
    m_dumpDir = rtgmcEdiEnvString("QSVENC_RTGMC_EDI_DUMP_DIR");
    m_dumpJsonl = rtgmcEdiEnvString("QSVENC_RTGMC_EDI_DUMP_JSONL");
    m_dumpMaxFrames = rtgmcEdiEnvInt("QSVENC_RTGMC_EDI_DUMP_MAX_FRAMES");
}

bool RGYFilterRtgmcEdi::dumpRequested(int frameIndex) const {
    return !m_dumpDir.empty() && (m_dumpMaxFrames <= 0 || frameIndex < m_dumpMaxFrames);
}

RGYFilterRtgmcEdi::FrameSource::FrameSource() :
    m_nFramesInput(0),
    m_buf() {
}

void RGYFilterRtgmcEdi::FrameSource::clear() {
    for (auto& buf : m_buf) {
        buf.reset();
    }
    m_nFramesInput = 0;
}

void RGYFilterRtgmcEdi::FrameSource::resetFrames() {
    m_nFramesInput = 0;
}

RGY_ERR RGYFilterRtgmcEdi::FrameSource::alloc(std::shared_ptr<RGYOpenCLContext> cl, const RGYFrameInfo& frameInfo) {
    if (m_buf[0]
        && !cmpFrameInfoCspResolution(&m_buf[0]->frame, &frameInfo)) {
        bool allocated = true;
        for (auto& buf : m_buf) {
            if (!buf || buf->frame.ptr[0] == nullptr) {
                allocated = false;
                break;
            }
        }
        if (allocated) {
            return RGY_ERR_NONE;
        }
    }
    for (auto& buf : m_buf) {
        buf = cl->createFrameBuffer(frameInfo.width, frameInfo.height, frameInfo.csp, frameInfo.bitdepth);
        if (!buf) {
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    m_nFramesInput = 0;
    return RGY_ERR_NONE;
}

RGYCLFrame *RGYFilterRtgmcEdi::FrameSource::get(int iframe) {
    iframe = clamp(iframe, 0, m_nFramesInput - 1);
    return m_buf[iframe % m_buf.size()].get();
}

int RGYFilterRtgmcEdi::FrameSource::findIndexByInputFrameId(int inputFrameId) const {
    const int start = std::max(0, m_nFramesInput - (int)m_buf.size());
    for (int iframe = start; iframe < m_nFramesInput; iframe++) {
        const auto& buf = m_buf[iframe % m_buf.size()];
        if (buf && buf->frame.ptr[0] && buf->frame.inputFrameId == inputFrameId) {
            return iframe;
        }
    }
    return -1;
}

RGY_ERR RGYFilterRtgmcEdi::FrameSource::add(std::shared_ptr<RGYOpenCLContext> cl, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue) {
    const int iframe = m_nFramesInput++;
    auto pDstFrame = get(iframe);
    auto err = cl->copyFrame(&pDstFrame->frame, pInputFrame, nullptr, queue, {}, nullptr, RGYFrameCopyMode::FRAME, "rtgmc_edi_adapter.source_copy");
    if (err != RGY_ERR_NONE) {
        return err;
    }
    copyFramePropWithoutRes(&pDstFrame->frame, pInputFrame);
    return RGY_ERR_NONE;
}

tstring RGYFilterParamRtgmcEdi::print() const {
    return strsprintf(_T("rtgmc-edi: mode=%s%s, nnsize=%d, nneurons=%d, ediqual=%d, chroma_edi=%s"),
        get_cx_desc(list_vpp_rtgmc_edi_mode, (int)mode),
        rtgmcEdiModeIsLightweight(mode) ? rtgmcEdiModeDetail(mode) : _T(""),
        nnsize, nneurons, ediqual,
        get_cx_desc(list_vpp_rtgmc_chroma_edi_mode, (int)chromaEdi));
}

RGYFilterRtgmcEdi::NnediAdapterState::NnediAdapterState() :
    filter(),
    outputCsp(),
    cachedFrames({ nullptr, nullptr }),
    cachedKey(),
    cachedEvent(),
    cacheValid(false) {
}

void RGYFilterRtgmcEdi::NnediAdapterState::clear() {
    filter.reset();
    outputCsp.reset();
    cachedFrames = { nullptr, nullptr };
    cachedKey = FrameKey();
    cachedEvent.reset();
    cacheValid = false;
}

RGYFilterRtgmcEdi::RGYFilterRtgmcEdi(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_edi(),
    m_buildOptions(),
    m_dumpDir(),
    m_dumpJsonl(),
    m_dumpMaxFrames(0),
    m_bobSource(),
    m_ediSource(),
    m_inputSource(),
    m_nnediStates(),
    m_nnediAdapterCopyEvent(),
    m_nFrame(0),
    m_lastInputFrameId(-1),
    m_pairFrameIndex(0),
    m_fallbackFrameIndex(0),
    m_useKernel(false) {
    m_name = _T("rtgmc-edi");
    m_pathThrough &= ~(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_DATA);
}

RGYFilterRtgmcEdi::~RGYFilterRtgmcEdi() {
    close();
}

RGY_ERR RGYFilterRtgmcEdi::checkParam(const std::shared_ptr<RGYFilterParamRtgmcEdi> &prm) {
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameIn.width <= 0 || prm->frameIn.height <= 0
        || prm->frameOut.width <= 0 || prm->frameOut.height <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameIn.csp != prm->frameOut.csp
        || prm->frameIn.width != prm->frameOut.width
        || prm->frameIn.height != prm->frameOut.height) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi requires identical input/output csp and resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->chromaEdi == VppRtgmcChromaEdiMode::NNEDI3) {
        if (prm->mode != VppRtgmcEdiMode::NNEDI3) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi chroma_edi=nnedi3 requires mode=nnedi3.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        if (RGY_CSP_PLANES[prm->frameOut.csp] < 3) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi chroma_edi=nnedi3 requires a planar YUV format with chroma planes.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }
    if (prm->mode == VppRtgmcEdiMode::NNEDI3
        && (prm->sourceFrameIn.csp == RGY_CSP_NA || prm->sourceFrameIn.width <= 0 || prm->sourceFrameIn.height <= 0)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi mode nnedi3 requires original source frame info; use it through --vpp-rtgmc, not standalone --vpp-rtgmc-edi.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->mode != VppRtgmcEdiMode::Passthrough && !rtgmcEdiModeIsLightweight(prm->mode)) {
        if (prm->mode != VppRtgmcEdiMode::NNEDI3) {
            AddMessage(RGY_LOG_ERROR, _T("Invalid rtgmc-edi mode.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
    }
    if (prm->nnsize < 0 || prm->nnsize > 6 || prm->nneurons < 0 || prm->nneurons > 4 || prm->ediqual < 1 || prm->ediqual > 2) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid rtgmc-edi NNEDI3 parameter: nnsize %d, nneurons %d, ediqual %d.\n"),
            prm->nnsize, prm->nneurons, prm->ediqual);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcEdi::checkInputs(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame) {
    if (pBobInputFrame->csp != pEdiInputFrame->csp
        || pBobInputFrame->width != pEdiInputFrame->width
        || pBobInputFrame->height != pEdiInputFrame->height) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi requires bob/edi inputs to match in csp and resolution.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcEdi::buildKernels(const std::shared_ptr<RGYFilterParamRtgmcEdi> &prm) {
    const int bitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int pixelMax = (bitdepth >= 16) ? ((1 << 16) - 1) : ((1 << bitdepth) - 1);
    m_buildOptions = strsprintf(
        "-D Type=%s -D bit_depth=%d -D max_val=%d -D rtgmc_edi_block_x=%d -D rtgmc_edi_block_y=%d",
        bitdepth > 8 ? "ushort" : "uchar",
        bitdepth,
        pixelMax,
        RTGMC_EDI_BLOCK_X,
        RTGMC_EDI_BLOCK_Y);
    m_edi.set(m_cl->buildResourceAsync(_T("RGY_FILTER_RTGMC_EDI_CL"), _T("EXE_DATA"), m_buildOptions.c_str()));
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcEdi::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamRtgmcEdi>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    m_pathThrough = FILTER_PATHTHROUGH_NONE;
    m_useKernel = (RGY_CSP_BIT_DEPTH[prm->frameOut.csp] <= 16);
    m_lastInputFrameId = -1;
    m_pairFrameIndex = 0;
    m_fallbackFrameIndex = 0;
    m_nFrame = 0;
    for (auto &state : m_nnediStates) {
        state.clear();
    }
    m_nnediAdapterCopyEvent.reset();

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamRtgmcEdi>(m_param);
    if (prm->mode == VppRtgmcEdiMode::NNEDI3) {
        sts = initNnediAdapterState(m_nnediStates[0], prm, false);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (prm->chromaEdi == VppRtgmcChromaEdiMode::NNEDI3) {
            sts = initNnediAdapterState(m_nnediStates[1], prm, true);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        m_edi.clear();
        m_buildOptions.clear();
        m_useKernel = false;
    } else if (m_useKernel
        && (!m_edi.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp])) {
        sts = buildKernels(prm);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build rtgmc-edi kernel.\n"));
            return sts;
        }
    }
    if (rtgmcEdiModeUsesTemporalYadif(prm->mode)) {
        sts = m_bobSource.alloc(m_cl, prm->frameOut);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate rtgmc-edi bob source buffer: %s.\n"), get_err_mes(sts));
            return sts;
        }
        sts = m_ediSource.alloc(m_cl, prm->frameOut);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate rtgmc-edi source buffer: %s.\n"), get_err_mes(sts));
            return sts;
        }
        sts = m_inputSource.alloc(m_cl, prm->frameIn);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate rtgmc-edi input source buffer: %s.\n"), get_err_mes(sts));
            return sts;
        }
    } else if (prm->mode == VppRtgmcEdiMode::NNEDI3) {
        m_bobSource.clear();
        m_ediSource.clear();
        sts = m_inputSource.alloc(m_cl, prm->sourceFrameIn);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate rtgmc-edi NNEDI source buffer: %s.\n"), get_err_mes(sts));
            return sts;
        }
    } else {
        m_bobSource.clear();
        m_ediSource.clear();
        m_inputSource.clear();
    }

    sts = AllocFrameBuf(prm->frameOut, rtgmcEdiModeUsesTemporalYadif(prm->mode) ? 2 : 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    loadDumpEnv();
    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcEdi::initNnediAdapterState(NnediAdapterState &state, const std::shared_ptr<RGYFilterParamRtgmcEdi> &prm, const bool chroma) {
    auto filter = std::make_unique<RGYFilterNnedi>(m_cl);
    auto nnedi = std::make_shared<RGYFilterParamNnedi>();
    nnedi->nnedi.enable = true;
    nnedi->nnedi.field = VPP_NNEDI_FIELD_BOB;
    nnedi->nnedi.nsize = (VppNnediNSize)(chroma ? VPP_NNEDI_NSIZE_8x4 : prm->nnsize);
    nnedi->nnedi.nns = rgy_nnedi_nns_value(chroma ? 0 : prm->nneurons);
    nnedi->nnedi.quality = (VppNnediQuality)(chroma ? VPP_NNEDI_QUALITY_FAST : prm->ediqual);
    nnedi->nnedi.processPlane = chroma
        ? std::array<bool, 4>{ false, true, true, false }
        : std::array<bool, 4>{ true, false, false, false };
    nnedi->nnedi.prescreen = 2;
    nnedi->nnedi.errortype = VPP_NNEDI_ETYPE_ABS;
    nnedi->nnedi.clamp = 1;
    nnedi->nnedi.doubleHeight = false;
    nnedi->hModule = prm->hModule ? prm->hModule : m_cl->getModuleHandle();
    nnedi->frameIn = prm->sourceFrameIn;
    nnedi->frameOut = prm->sourceFrameIn;
    nnedi->baseFps = prm->sourceBaseFps;
    nnedi->timebase = prm->sourceTimebase;
    nnedi->bOutOverwrite = false;
    auto sts = filter->init(nnedi, m_pLog);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialize rtgmc-edi %s NNEDI adapter: %s.\n"),
            chroma ? _T("chroma") : _T("main"), get_err_mes(sts));
        return sts;
    }
    if (nnedi->frameOut.width != prm->frameOut.width || nnedi->frameOut.height != prm->frameOut.height) {
        AddMessage(RGY_LOG_ERROR,
            _T("rtgmc-edi %s NNEDI adapter output size does not match Bob frame size: NNEDI %dx%d, Bob %dx%d.\n"),
            chroma ? _T("chroma") : _T("main"),
            nnedi->frameOut.width, nnedi->frameOut.height, prm->frameOut.width, prm->frameOut.height);
        return RGY_ERR_UNSUPPORTED;
    }
    if (nnedi->frameOut.csp != prm->frameOut.csp) {
        auto cspconv = std::make_unique<RGYFilterCspCrop>(m_cl);
        auto cspPrm = std::make_shared<RGYFilterParamCrop>();
        cspPrm->frameIn = nnedi->frameOut;
        cspPrm->frameOut = nnedi->frameOut;
        cspPrm->frameOut.csp = prm->frameOut.csp;
        cspPrm->baseFps = nnedi->baseFps;
        cspPrm->bOutOverwrite = false;
        sts = cspconv->init(cspPrm, m_pLog);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to initialize rtgmc-edi %s NNEDI output csp conversion (%s -> %s): %s.\n"),
                chroma ? _T("chroma") : _T("main"),
                RGY_CSP_NAMES[nnedi->frameOut.csp], RGY_CSP_NAMES[prm->frameOut.csp], get_err_mes(sts));
            return sts;
        }
        state.outputCsp = std::move(cspconv);
    } else {
        state.outputCsp.reset();
    }
    state.filter = std::move(filter);
    state.cachedFrames = { nullptr, nullptr };
    state.cachedKey = FrameKey();
    state.cachedEvent.reset();
    state.cacheValid = false;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcEdi::runNnediAdapterState(NnediAdapterState &state, const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pSourceInputFrame,
    RGYFrameInfo **ppOutputFrame, const RGYFrameInfo **ppSelectedFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
    RGYOpenCLEvent *event, const RGYFilterParamRtgmcEdi &prm, const bool chroma) {
    UNREFERENCED_PARAMETER(prm);
    if (ppOutputFrame) {
        *ppOutputFrame = nullptr;
    }
    if (ppSelectedFrame) {
        *ppSelectedFrame = nullptr;
    }
    if (!pBobInputFrame || !pBobInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (!pSourceInputFrame || !pSourceInputFrame->ptr[0]) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi %s nnedi3 requires original source input frame.\n"),
            chroma ? _T("chroma") : _T("main"));
        return RGY_ERR_INVALID_CALL;
    }
    if (!state.filter) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi %s NNEDI adapter is not initialized.\n"),
            chroma ? _T("chroma") : _T("main"));
        return RGY_ERR_INVALID_CALL;
    }
    if (pSourceInputFrame && pSourceInputFrame->ptr[0]
        && m_inputSource.findIndexByInputFrameId(pSourceInputFrame->inputFrameId) < 0) {
        auto err = m_inputSource.add(m_cl, pSourceInputFrame, queue);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to add rtgmc-edi %s NNEDI source frame: %s.\n"),
                chroma ? _T("chroma") : _T("main"), get_err_mes(err));
            return err;
        }
    }
    const int sourceIndex = m_inputSource.findIndexByInputFrameId(pBobInputFrame->inputFrameId);
    if (sourceIndex < 0) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi %s NNEDI adapter source frame is missing for Bob inputFrameId=%d.\n"),
            chroma ? _T("chroma") : _T("main"), pBobInputFrame->inputFrameId);
        return RGY_ERR_INVALID_CALL;
    }
    const auto *pNnediSourceFrame = &m_inputSource.get(sourceIndex)->frame;
    const FrameKey sourceKey(pNnediSourceFrame);
    if (!state.cacheValid || !state.cachedKey.matches(pNnediSourceFrame)) {
        RGYFrameInfo *nnediOut[2] = { nullptr, nullptr };
        int nnediOutNum = 0;
        RGYOpenCLEvent nnediEvent;
        auto nnediWaitEvents = wait_events;
        if (m_nnediAdapterCopyEvent() != nullptr) {
            nnediWaitEvents.push_back(m_nnediAdapterCopyEvent);
        }
        auto sts = state.filter->filter(const_cast<RGYFrameInfo *>(pNnediSourceFrame), nnediOut, &nnediOutNum, queue, nnediWaitEvents, &nnediEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (nnediOutNum != 2 || !nnediOut[0] || !nnediOut[1] || !nnediOut[0]->ptr[0] || !nnediOut[1]->ptr[0]) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi %s NNEDI adapter expected 2 output frames, got %d.\n"),
                chroma ? _T("chroma") : _T("main"), nnediOutNum);
            return RGY_ERR_INVALID_CALL;
        }
        state.cachedFrames = { nnediOut[0], nnediOut[1] };
        state.cachedKey = sourceKey;
        state.cachedEvent = nnediEvent;
        state.cacheValid = true;
    }

    RGYFrameInfo *selected = nullptr;
    for (auto *candidate : state.cachedFrames) {
        if (candidate && candidate->inputFrameId == pBobInputFrame->inputFrameId
            && candidate->timestamp == pBobInputFrame->timestamp
            && candidate->duration == pBobInputFrame->duration) {
            selected = candidate;
            break;
        }
    }
    if (selected == nullptr) {
        for (auto *candidate : state.cachedFrames) {
            if (candidate && candidate->inputFrameId == pBobInputFrame->inputFrameId
                && candidate->timestamp == pBobInputFrame->timestamp) {
                selected = candidate;
                break;
            }
        }
    }
    if (selected == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi %s NNEDI adapter could not match Bob frame timestamp to cached NNEDI output.\n"),
            chroma ? _T("chroma") : _T("main"));
        AddMessage(RGY_LOG_ERROR, _T("  Bob frame: inputFrameId=%d, timestamp=%lld, duration=%lld.\n"),
            pBobInputFrame->inputFrameId, (long long)pBobInputFrame->timestamp, (long long)pBobInputFrame->duration);
        for (int i = 0; i < (int)state.cachedFrames.size(); i++) {
            const auto *candidate = state.cachedFrames[i];
            if (candidate) {
                AddMessage(RGY_LOG_ERROR, _T("  NNEDI candidate[%d]: inputFrameId=%d, timestamp=%lld, duration=%lld.\n"),
                    i, candidate->inputFrameId, (long long)candidate->timestamp, (long long)candidate->duration);
            }
        }
        return RGY_ERR_INVALID_CALL;
    }
    if (ppSelectedFrame) {
        *ppSelectedFrame = selected;
    }

    std::vector<RGYOpenCLEvent> copyWaitEvents;
    RGYOpenCLEvent outputEvent = state.cachedEvent;
    if (state.cachedEvent() != nullptr) {
        copyWaitEvents.push_back(state.cachedEvent);
    }
    RGYFrameInfo *copySrc = selected;
    if (state.outputCsp) {
        RGYFrameInfo *converted[1] = { nullptr };
        int convertedNum = 0;
        RGYOpenCLEvent convertEvent;
        auto sts = state.outputCsp->filter(selected, converted, &convertedNum, queue, copyWaitEvents, &convertEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (convertedNum != 1 || !converted[0] || !converted[0]->ptr[0]) {
            AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi %s NNEDI adapter csp conversion expected 1 output frame, got %d.\n"),
                chroma ? _T("chroma") : _T("main"), convertedNum);
            return RGY_ERR_INVALID_CALL;
        }
        copySrc = converted[0];
        copyWaitEvents.clear();
        if (convertEvent() != nullptr) {
            outputEvent = convertEvent;
        }
    }
    if (ppOutputFrame) {
        *ppOutputFrame = copySrc;
    }
    if (event) {
        *event = outputEvent;
    }
    return RGY_ERR_NONE;
}

int RGYFilterRtgmcEdi::targetField(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pParityFrame) {
    int pairIndex = 0;
    if (pBobInputFrame != nullptr && pBobInputFrame->inputFrameId >= 0) {
        if (pBobInputFrame->inputFrameId != m_lastInputFrameId) {
            m_lastInputFrameId = pBobInputFrame->inputFrameId;
            m_pairFrameIndex = 0;
        }
        pairIndex = m_pairFrameIndex++;
    } else {
        pairIndex = m_fallbackFrameIndex++;
    }
    const auto *fieldOrderFrame = (pParityFrame != nullptr) ? pParityFrame : pBobInputFrame;
    const bool bff = fieldOrderFrame != nullptr && (fieldOrderFrame->picstruct & RGY_PICSTRUCT_BFF);
    const int firstTargetField = bff ? 0 : 1;
    return (pairIndex & 1) ? (1 - firstTargetField) : firstTargetField;
}

RGY_ERR RGYFilterRtgmcEdi::processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pBobInputFrame,
    const RGYFrameInfo *pEdiPrevFrame, const RGYFrameInfo *pEdiInputFrame, const RGYFrameInfo *pEdiNextFrame,
    const RGYFilterParamRtgmcEdi &prm,
    const int targetField,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *kernelName = rtgmcEdiKernelName(prm.mode);
    const RGYFrameInfo *pSrcForProps = (rtgmcEdiModeUsesTemporalYadif(prm.mode) && pBobInputFrame != nullptr)
        ? pBobInputFrame
        : (pEdiInputFrame != nullptr) ? pEdiInputFrame : pBobInputFrame;
    const auto *fieldOrderFrame = (pEdiInputFrame != nullptr) ? pEdiInputFrame : pBobInputFrame;
    const bool tff = fieldOrderFrame == nullptr || (fieldOrderFrame->picstruct & RGY_PICSTRUCT_BFF) == 0;
    const int fieldSecond = ((targetField == 0) == tff) ? 1 : 0;
    const int planes = RGY_CSP_PLANES[pSrcForProps->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto dstPlane = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto bobPlane = getPlane(pBobInputFrame, (RGY_PLANE)iplane);
        const auto ediPrevPlane = getPlane(pEdiPrevFrame, (RGY_PLANE)iplane);
        const auto ediPlane = getPlane(pEdiInputFrame, (RGY_PLANE)iplane);
        const auto ediNextPlane = getPlane(pEdiNextFrame, (RGY_PLANE)iplane);
        RGYWorkSize local(RTGMC_EDI_BLOCK_X, RTGMC_EDI_BLOCK_Y);
        RGYWorkSize global(dstPlane.width, dstPlane.height);
        const auto &waitHere = (iplane == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        auto err = m_edi.get()->kernel(kernelName).config(queue, local, global, waitHere, (iplane == planes - 1) ? event : nullptr).launch(
            (cl_mem)dstPlane.ptr[0], dstPlane.pitch[0],
            (cl_mem)bobPlane.ptr[0], bobPlane.pitch[0],
            (cl_mem)ediPrevPlane.ptr[0], ediPrevPlane.pitch[0],
            (cl_mem)ediPlane.ptr[0], ediPlane.pitch[0],
            (cl_mem)ediNextPlane.ptr[0], ediNextPlane.pitch[0],
            dstPlane.width, dstPlane.height,
            iplane,
            targetField,
            fieldSecond);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"),
                char_to_tstring(kernelName).c_str(), iplane, get_err_mes(err));
            return err;
        }
    }
    rtgmcEdiCopyFrameProps(pOutputFrame, pSrcForProps, prm.mode);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcEdi::runTemporalYadif(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame, const RGYFrameInfo *pSourceInputFrame,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event,
    const RGYFilterParamRtgmcEdi &prm) {
    const bool draining = !pBobInputFrame || !pBobInputFrame->ptr[0];
    if (pBobInputFrame && pBobInputFrame->ptr[0]) {
        auto err = m_bobSource.add(m_cl, pBobInputFrame, queue);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to add rtgmc-edi bob source frame: %s.\n"), get_err_mes(err));
            return err;
        }
        err = m_ediSource.add(m_cl, pEdiInputFrame, queue);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to add rtgmc-edi source frame: %s.\n"), get_err_mes(err));
            return err;
        }
        if (pSourceInputFrame && pSourceInputFrame->ptr[0]
            && m_inputSource.findIndexByInputFrameId(pSourceInputFrame->inputFrameId) < 0) {
            err = m_inputSource.add(m_cl, pSourceInputFrame, queue);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to add rtgmc-edi input source frame: %s.\n"), get_err_mes(err));
                return err;
            }
        }
    }

    while (*pOutputFrameNum < (int)m_frameBuf.size() && m_nFrame < m_ediSource.inframe()) {
        const auto *pBobCur = &m_bobSource.get(m_nFrame)->frame;
        const int srcIndex = m_inputSource.findIndexByInputFrameId(pBobCur->inputFrameId);
        if (srcIndex < 0 || (!draining && srcIndex + 1 >= m_inputSource.inframe())) {
            break;
        }

        auto pOutFrame = m_frameBuf[*pOutputFrameNum].get();
        ppOutputFrames[*pOutputFrameNum] = &pOutFrame->frame;

        const int prevIndex = (srcIndex == 0 && m_inputSource.inframe() > 1) ? srcIndex + 1 : srcIndex - 1;
        const auto *pSrcPrev = &m_inputSource.get(prevIndex)->frame;
        const auto *pSrcCur = &m_inputSource.get(srcIndex + 0)->frame;
        const auto *pSrcNext = &m_inputSource.get(srcIndex + 1)->frame;
        const int target = targetField(pBobCur, pSrcCur);
        const auto &frameWaitEvents = (*pOutputFrameNum == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        auto err = processFrame(&pOutFrame->frame, pBobCur, pSrcPrev, pSrcCur, pSrcNext, prm, target, queue, frameWaitEvents, event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        (*pOutputFrameNum)++;
        if (dumpRequested(m_nFrame)) {
            auto dumpErr = rtgmcEdiDumpTrace(m_dumpDir, m_dumpJsonl, m_nFrame, target,
                pBobCur, pSrcPrev, pSrcCur, pSrcNext, &pOutFrame->frame, m_cl, queue, {});
            if (dumpErr != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to dump rtgmc-edi diagnostic frames: %s.\n"), get_err_mes(dumpErr));
                return dumpErr;
            }
        }
        m_nFrame++;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcEdi::runNnediAdapter(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pSourceInputFrame,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event,
    const RGYFilterParamRtgmcEdi &prm) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    if (!pBobInputFrame || !pBobInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (!pSourceInputFrame || !pSourceInputFrame->ptr[0]) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi nnedi3 requires original source input frame.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    if (pSourceInputFrame && pSourceInputFrame->ptr[0]
        && m_inputSource.findIndexByInputFrameId(pSourceInputFrame->inputFrameId) < 0) {
        auto err = m_inputSource.add(m_cl, pSourceInputFrame, queue);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to add rtgmc-edi NNEDI source frame: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    const int sourceIndex = m_inputSource.findIndexByInputFrameId(pBobInputFrame->inputFrameId);
    if (sourceIndex < 0) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-edi NNEDI adapter source frame is missing for Bob inputFrameId=%d.\n"),
            pBobInputFrame->inputFrameId);
        return RGY_ERR_INVALID_CALL;
    }
    auto pOutFrame = m_frameBuf[0].get();
    RGYFrameInfo *mainFrame = nullptr;
    const RGYFrameInfo *mainSelected = nullptr;
    RGYOpenCLEvent mainEvent;
    auto sts = runNnediAdapterState(m_nnediStates[0], pBobInputFrame, pSourceInputFrame, &mainFrame, &mainSelected, queue, wait_events, &mainEvent, prm, false);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    RGYFrameInfo *chromaFrame = nullptr;
    RGYOpenCLEvent chromaEvent;
    const bool useChroma = prm.chromaEdi == VppRtgmcChromaEdiMode::NNEDI3;
    if (useChroma) {
        sts = runNnediAdapterState(m_nnediStates[1], pBobInputFrame, pSourceInputFrame, &chromaFrame, nullptr, queue, wait_events, &chromaEvent, prm, true);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }

    RGYOpenCLEvent copyEvent;
    auto dstY = getPlane(&pOutFrame->frame, RGY_PLANE_Y);
    const auto srcY = getPlane(mainFrame, RGY_PLANE_Y);
    auto err = m_cl->copyPlane(&dstY, &srcY, nullptr, queue,
        (mainEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ mainEvent } : std::vector<RGYOpenCLEvent>(), &copyEvent);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-edi NNEDI adapter luma output: %s.\n"), get_err_mes(err));
        return err;
    }

    RGYOpenCLEvent finalCopyEvent = copyEvent;
    if (useChroma) {
        const int chromaPlanes = std::min(3, (int)RGY_CSP_PLANES[pOutFrame->frame.csp]);
        for (int iplane = 1; iplane < chromaPlanes; iplane++) {
            auto dstPlane = getPlane(&pOutFrame->frame, (RGY_PLANE)iplane);
            const auto srcPlane = getPlane(chromaFrame, (RGY_PLANE)iplane);
            if (dstPlane.ptr[0] == nullptr || srcPlane.ptr[0] == nullptr) {
                continue;
            }
            std::vector<RGYOpenCLEvent> chromaWaitEvents;
            if (chromaEvent() != nullptr) {
                chromaWaitEvents.push_back(chromaEvent);
            }
            if (finalCopyEvent() != nullptr) {
                chromaWaitEvents.push_back(finalCopyEvent);
            }
            RGYOpenCLEvent planeEvent;
            err = m_cl->copyPlane(&dstPlane, &srcPlane, nullptr, queue, chromaWaitEvents, &planeEvent);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-edi NNEDI chroma plane %d: %s.\n"), iplane, get_err_mes(err));
                return err;
            }
            finalCopyEvent = planeEvent;
        }
    } else {
        const int chromaPlanes = std::min(3, (int)RGY_CSP_PLANES[pOutFrame->frame.csp]);
        for (int iplane = 1; iplane < chromaPlanes; iplane++) {
            auto dstPlane = getPlane(&pOutFrame->frame, (RGY_PLANE)iplane);
            const auto srcPlane = getPlane(pBobInputFrame, (RGY_PLANE)iplane);
            if (dstPlane.ptr[0] == nullptr || srcPlane.ptr[0] == nullptr) {
                continue;
            }
            RGYOpenCLEvent planeEvent;
            err = m_cl->copyPlane(&dstPlane, &srcPlane, nullptr, queue,
                (finalCopyEvent() != nullptr) ? std::vector<RGYOpenCLEvent>{ finalCopyEvent } : std::vector<RGYOpenCLEvent>(), &planeEvent);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed to copy rtgmc-edi bob chroma plane %d: %s.\n"), iplane, get_err_mes(err));
                return err;
            }
            finalCopyEvent = planeEvent;
        }
    }

    m_nnediAdapterCopyEvent = finalCopyEvent;
    if (event) {
        *event = finalCopyEvent;
    }
    copyFramePropWithoutRes(&pOutFrame->frame, pBobInputFrame);
    pOutFrame->frame.picstruct = RGY_PICSTRUCT_FRAME;
    if (dumpRequested(m_nFrame)) {
        auto dumpWaitEvents = wait_events;
        if (mainEvent() != nullptr) {
            dumpWaitEvents.push_back(mainEvent);
        }
        if (useChroma && chromaEvent() != nullptr) {
            dumpWaitEvents.push_back(chromaEvent);
        }
        if (finalCopyEvent() != nullptr) {
            dumpWaitEvents.push_back(finalCopyEvent);
        }
        const int target = targetField(pBobInputFrame, pSourceInputFrame);
        const auto *dumpSrc = (mainSelected != nullptr) ? mainSelected : mainFrame;
        auto dumpErr = rtgmcEdiDumpTrace(m_dumpDir, m_dumpJsonl, m_nFrame, target,
            pBobInputFrame, pSourceInputFrame, pSourceInputFrame, dumpSrc, &pOutFrame->frame, m_cl, queue, dumpWaitEvents);
        if (dumpErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to dump rtgmc-edi NNEDI adapter diagnostic frames: %s.\n"), get_err_mes(dumpErr));
            return dumpErr;
        }
    }
    m_nFrame++;
    ppOutputFrames[0] = &pOutFrame->frame;
    *pOutputFrameNum = 1;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcEdi::run_filter_impl(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame, const RGYFrameInfo *pSourceInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
    RGYOpenCLEvent *event) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;

    auto prm = std::dynamic_pointer_cast<RGYFilterParamRtgmcEdi>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->mode == VppRtgmcEdiMode::NNEDI3) {
        return runNnediAdapter(pBobInputFrame, pSourceInputFrame, ppOutputFrames, pOutputFrameNum, queue, wait_events, event, *prm);
    }

    if (!pBobInputFrame || !pBobInputFrame->ptr[0] || !pEdiInputFrame || !pEdiInputFrame->ptr[0]) {
        if (rtgmcEdiModeUsesTemporalYadif(prm->mode)) {
            return runTemporalYadif(nullptr, nullptr, nullptr, ppOutputFrames, pOutputFrameNum, queue, wait_events, event, *prm);
        }
        return RGY_ERR_NONE;
    }
    auto inputErr = checkInputs(pBobInputFrame, pEdiInputFrame);
    if (inputErr != RGY_ERR_NONE) {
        return inputErr;
    }
    if (m_useKernel && !m_edi.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build/load RGY_FILTER_RTGMC_EDI_CL (options: %s).\n"),
            char_to_tstring(m_buildOptions).c_str());
        return RGY_ERR_OPENCL_CRUSH;
    }

    auto pOutFrame = m_frameBuf[0].get();

    if (m_useKernel) {
        const auto bobMemcpyKind = getMemcpyKind(pBobInputFrame->mem_type, m_frameBuf[0]->frame.mem_type);
        const auto ediMemcpyKind = getMemcpyKind(pEdiInputFrame->mem_type, m_frameBuf[0]->frame.mem_type);
        if (bobMemcpyKind == RGYCLMemcpyD2D && ediMemcpyKind == RGYCLMemcpyD2D) {
            if (rtgmcEdiModeUsesTemporalYadif(prm->mode)) {
                return runTemporalYadif(pBobInputFrame, pEdiInputFrame, pSourceInputFrame, ppOutputFrames, pOutputFrameNum, queue, wait_events, event, *prm);
            }
            ppOutputFrames[0] = &pOutFrame->frame;
            *pOutputFrameNum = 1;
            const int target = targetField(pBobInputFrame);
            auto err = processFrame(&pOutFrame->frame, pBobInputFrame, pEdiInputFrame, pEdiInputFrame, pEdiInputFrame, *prm, target, queue, wait_events, event);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            return RGY_ERR_NONE;
        }
    }

    ppOutputFrames[0] = &pOutFrame->frame;
    *pOutputFrameNum = 1;
    auto copyErr = m_cl->copyFrame(ppOutputFrames[0], pEdiInputFrame, nullptr, queue, wait_events, event, RGYFrameCopyMode::FRAME, "rtgmc_edi.fallback_copy");
    if (copyErr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy frame: %s.\n"), get_err_mes(copyErr));
        return copyErr;
    }
    rtgmcEdiCopyFrameProps(ppOutputFrames[0], pEdiInputFrame, prm->mode);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterRtgmcEdi::run_filter(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
    RGYOpenCLEvent *event) {
    return run_filter_impl(pBobInputFrame, pEdiInputFrame, pEdiInputFrame, ppOutputFrames, pOutputFrameNum, queue, wait_events, event);
}

RGY_ERR RGYFilterRtgmcEdi::run_filter(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame, const RGYFrameInfo *pSourceInputFrame,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
    RGYOpenCLEvent *event) {
    return run_filter_impl(pBobInputFrame, pEdiInputFrame, pSourceInputFrame, ppOutputFrames, pOutputFrameNum, queue, wait_events, event);
}

RGY_ERR RGYFilterRtgmcEdi::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events,
    RGYOpenCLEvent *event) {
    return run_filter_impl(pInputFrame, pInputFrame, pInputFrame, ppOutputFrames, pOutputFrameNum, queue, wait_events, event);
}

void RGYFilterRtgmcEdi::resetTemporalState() {
    m_bobSource.resetFrames();
    m_ediSource.resetFrames();
    m_inputSource.resetFrames();
    for (auto &state : m_nnediStates) {
        state.cachedFrames = { nullptr, nullptr };
        state.cachedKey = FrameKey();
        state.cachedEvent.reset();
        state.cacheValid = false;
    }
    m_nnediAdapterCopyEvent.reset();
    m_nFrame = 0;
    m_lastInputFrameId = -1;
    m_pairFrameIndex = 0;
    m_fallbackFrameIndex = 0;
}

void RGYFilterRtgmcEdi::close() {
    m_edi.clear();
    m_buildOptions.clear();
    m_dumpDir.clear();
    m_dumpJsonl.clear();
    m_dumpMaxFrames = 0;
    m_bobSource.clear();
    m_ediSource.clear();
    m_inputSource.clear();
    for (auto &state : m_nnediStates) {
        state.clear();
    }
    m_nnediAdapterCopyEvent.reset();
    m_frameBuf.clear();
    m_nFrame = 0;
    m_lastInputFrameId = -1;
    m_pairFrameIndex = 0;
    m_fallbackFrameIndex = 0;
    m_useKernel = false;
    m_cl.reset();
}
