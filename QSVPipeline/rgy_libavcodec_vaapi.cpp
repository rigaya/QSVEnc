// -----------------------------------------------------------------------------------------
//     VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2014-2017 rigaya
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
// ------------------------------------------------------------------------------------------

#include "rgy_libavcodec_vaapi.h"
#include "rgy_frame.h"
#include "rgy_bitstream.h"
#include "rgy_input_avcodec.h"

#if ENABLE_VAAPI

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <unistd.h>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vaapi.h>
}

#include <va/va.h>
#include <va/va_drm.h>

namespace {

constexpr int VA_DEFAULT_FRAME_RATE = 30;
constexpr int VA_DEFAULT_BIT_DEPTH = 8;
constexpr int VA_DEFAULT_GOP_LENGTH = 30;
constexpr int VA_DEFAULT_GOP_SECONDS = 2;
constexpr int VA_DEFAULT_BITRATE = 1000000;
constexpr int VA_PROBE_WIDTH = 640;
constexpr int VA_PROBE_HEIGHT = 360;
constexpr int VA_PROBE_FRAME_POOL_SIZE = 2;
constexpr int VA_ENCODER_FRAME_POOL_SIZE = 8;

const char *rc_mode_name(const RGYVAEncRCMode mode) {
    switch (mode) {
    case RGY_VA_RC_CBR:  return "CBR";
    case RGY_VA_RC_VBR:  return "VBR";
    case RGY_VA_RC_CQP:  return "CQP";
    case RGY_VA_RC_QVBR: return "QVBR";
    case RGY_VA_RC_ICQ:  return "ICQ";
    case RGY_VA_RC_AVBR: return "AVBR";
    default: return nullptr;
    }
}

const char *codec_name(const RGY_CODEC codec) {
    switch (codec) {
    case RGY_CODEC_H264: return "h264_vaapi";
    case RGY_CODEC_HEVC: return "hevc_vaapi";
    case RGY_CODEC_AV1:  return "av1_vaapi";
    case RGY_CODEC_VP9:  return "vp9_vaapi";
    default:             return nullptr;
    }
}

VAProfile va_profile(const RGY_CODEC codec) {
    switch (codec) {
    case RGY_CODEC_H264: return VAProfileH264High;
    case RGY_CODEC_HEVC: return VAProfileHEVCMain;
    case RGY_CODEC_AV1:  return VAProfileAV1Profile0;
    case RGY_CODEC_VP9:  return VAProfileVP9Profile0;
    default:             return VAProfileNone;
    }
}

bool has_encoding_entrypoint(VADisplay display, const VAProfile profile, VAEntrypoint& selected) {
    const int maxEntrypoints = vaMaxNumEntrypoints(display);
    if (maxEntrypoints <= 0) return false;
    std::vector<VAEntrypoint> entrypoints(maxEntrypoints);
    int numEntrypoints = 0;
    if (vaQueryConfigEntrypoints(display, profile, entrypoints.data(), &numEntrypoints) != VA_STATUS_SUCCESS) {
        return false;
    }
    for (int i = 0; i < numEntrypoints; i++) {
        if (entrypoints[i] == VAEntrypointEncSlice) {
            selected = VAEntrypointEncSlice;
            return true;
        }
    }
    for (int i = 0; i < numEntrypoints; i++) {
        if (entrypoints[i] == VAEntrypointEncSliceLP) {
            selected = VAEntrypointEncSliceLP;
            return true;
        }
    }
    return false;
}

bool has_profile_entrypoint(VADisplay display, const VAProfile profile, const VAEntrypoint target) {
    const int maxEntrypoints = vaMaxNumEntrypoints(display);
    if (maxEntrypoints <= 0) return false;
    std::vector<VAEntrypoint> entrypoints(maxEntrypoints);
    int numEntrypoints = 0;
    if (vaQueryConfigEntrypoints(display, profile, entrypoints.data(), &numEntrypoints) != VA_STATUS_SUCCESS) {
        return false;
    }
    return std::find(entrypoints.begin(), entrypoints.begin() + numEntrypoints, target) != entrypoints.begin() + numEntrypoints;
}

uint32_t query_decode_rt_format(VADisplay display, const VAProfile profile) {
    VAConfigAttrib attrib = { VAConfigAttribRTFormat, VA_ATTRIB_NOT_SUPPORTED };
    if (vaGetConfigAttributes(display, profile, VAEntrypointVLD, &attrib, 1) != VA_STATUS_SUCCESS) {
        return 0;
    }
    return attrib.value == VA_ATTRIB_NOT_SUPPORTED ? 0 : attrib.value;
}

CodecCsp query_decode_caps(VADisplay display) {
    CodecCsp caps;
    const int maxProfiles = vaMaxNumProfiles(display);
    if (maxProfiles <= 0) return caps;
    std::vector<VAProfile> profiles(maxProfiles);
    int numProfiles = 0;
    if (vaQueryConfigProfiles(display, profiles.data(), &numProfiles) != VA_STATUS_SUCCESS) return caps;
    profiles.resize(numProfiles);

    const auto add = [&caps, display, &profiles](const RGY_CODEC codec, const VAProfile profile, const RGY_CSP csp, const uint32_t requiredFormat) {
        if (std::find(profiles.begin(), profiles.end(), profile) == profiles.end()
            || !has_profile_entrypoint(display, profile, VAEntrypointVLD)) {
            return;
        }
        const auto rtFormat = query_decode_rt_format(display, profile);
        if (codec == RGY_CODEC_AV1 && rtFormat == 0) return;
        if (rtFormat != 0 && requiredFormat != 0 && (rtFormat & requiredFormat) == 0) return;
        auto& csps = caps[codec];
        const auto addCsp = [&csps](const RGY_CSP outputCsp) {
            if (std::find(csps.begin(), csps.end(), outputCsp) == csps.end()) {
                csps.push_back(outputCsp);
            }
        };
        addCsp(csp);
        if (csp == RGY_CSP_NV12) addCsp(RGY_CSP_YV12);
        else if (csp == RGY_CSP_P010) addCsp(RGY_CSP_YV12_10);
    };

    add(RGY_CODEC_H264, VAProfileH264ConstrainedBaseline, RGY_CSP_NV12, VA_RT_FORMAT_YUV420);
    add(RGY_CODEC_H264, VAProfileH264Main, RGY_CSP_NV12, VA_RT_FORMAT_YUV420);
    add(RGY_CODEC_H264, VAProfileH264High, RGY_CSP_NV12, VA_RT_FORMAT_YUV420);
#if VA_CHECK_VERSION(1, 18, 0) // libva 2.18 未満には VAProfileH264High10 がない
    add(RGY_CODEC_H264, VAProfileH264High10, RGY_CSP_P010, VA_RT_FORMAT_YUV420_10);
#endif
    add(RGY_CODEC_HEVC, VAProfileHEVCMain, RGY_CSP_NV12, VA_RT_FORMAT_YUV420);
    add(RGY_CODEC_HEVC, VAProfileHEVCMain10, RGY_CSP_P010, VA_RT_FORMAT_YUV420_10);
    add(RGY_CODEC_AV1, VAProfileAV1Profile0, RGY_CSP_NV12, VA_RT_FORMAT_YUV420);
    add(RGY_CODEC_AV1, VAProfileAV1Profile0, RGY_CSP_P010, VA_RT_FORMAT_YUV420_10);
    add(RGY_CODEC_VP9, VAProfileVP9Profile0, RGY_CSP_NV12, VA_RT_FORMAT_YUV420);
    add(RGY_CODEC_VP9, VAProfileVP9Profile2, RGY_CSP_P010, VA_RT_FORMAT_YUV420_10);
    add(RGY_CODEC_MPEG2, VAProfileMPEG2Simple, RGY_CSP_NV12, VA_RT_FORMAT_YUV420);
    add(RGY_CODEC_MPEG2, VAProfileMPEG2Main, RGY_CSP_NV12, VA_RT_FORMAT_YUV420);
    add(RGY_CODEC_VC1, VAProfileVC1Simple, RGY_CSP_NV12, VA_RT_FORMAT_YUV420);
    add(RGY_CODEC_VC1, VAProfileVC1Main, RGY_CSP_NV12, VA_RT_FORMAT_YUV420);
    add(RGY_CODEC_VC1, VAProfileVC1Advanced, RGY_CSP_NV12, VA_RT_FORMAT_YUV420);
    return caps;
}

bool query_profile_attributes(VADisplay display, const VAProfile profile, const VAEntrypoint entrypoint,
    uint32_t& rcModes, int& maxRefL0, int& maxRefL1, int& maxWidth, int& maxHeight, uint32_t& rtFormat, int& qualityLevels) {
    std::array<VAConfigAttrib, 6> attrs = {{
        { VAConfigAttribRateControl, VA_ATTRIB_NOT_SUPPORTED },
        { VAConfigAttribEncMaxRefFrames, VA_ATTRIB_NOT_SUPPORTED },
        { VAConfigAttribMaxPictureWidth, VA_ATTRIB_NOT_SUPPORTED },
        { VAConfigAttribMaxPictureHeight, VA_ATTRIB_NOT_SUPPORTED },
        { VAConfigAttribRTFormat, VA_ATTRIB_NOT_SUPPORTED },
        { VAConfigAttribEncQualityRange, VA_ATTRIB_NOT_SUPPORTED },
    }};
    if (vaGetConfigAttributes(display, profile, entrypoint, attrs.data(), (int)attrs.size()) != VA_STATUS_SUCCESS) {
        return false;
    }
    if (attrs[0].value != VA_ATTRIB_NOT_SUPPORTED) {
        if (attrs[0].value & VA_RC_CBR)  rcModes |= RGY_VA_RC_CBR;
        if (attrs[0].value & VA_RC_VBR)  rcModes |= RGY_VA_RC_VBR;
        if (attrs[0].value & VA_RC_CQP)  rcModes |= RGY_VA_RC_CQP;
        if (attrs[0].value & VA_RC_QVBR) rcModes |= RGY_VA_RC_QVBR;
        if (attrs[0].value & VA_RC_ICQ)  rcModes |= RGY_VA_RC_ICQ;
        if (attrs[0].value & VA_RC_AVBR) rcModes |= RGY_VA_RC_AVBR;
    }
    if (attrs[1].value != VA_ATTRIB_NOT_SUPPORTED) {
        maxRefL0 = (int)(attrs[1].value & 0xffffu);
        maxRefL1 = (int)((attrs[1].value >> 16) & 0xffffu);
    }
    if (attrs[2].value != VA_ATTRIB_NOT_SUPPORTED) maxWidth = (int)attrs[2].value;
    if (attrs[3].value != VA_ATTRIB_NOT_SUPPORTED) maxHeight = (int)attrs[3].value;
    if (attrs[4].value != VA_ATTRIB_NOT_SUPPORTED) rtFormat = attrs[4].value;
    if (attrs[5].value != VA_ATTRIB_NOT_SUPPORTED) qualityLevels = (int)attrs[5].value;
    return true;
}

bool supports_10bit(VADisplay display, const RGY_CODEC codec, const VAEntrypoint entrypoint) {
    if (codec == RGY_CODEC_H264) return false;
    VAProfile profile = va_profile(codec);
    if (codec == RGY_CODEC_HEVC) {
        profile = VAProfileHEVCMain10;
    } else if (codec == RGY_CODEC_VP9) {
        profile = VAProfileVP9Profile2;
    }
    uint32_t rcModes = 0;
    uint32_t rtFormat = 0;
    int maxRefL0 = 0, maxRefL1 = 0, maxWidth = 0, maxHeight = 0, qualityLevels = 0;
    return query_profile_attributes(display, profile, entrypoint, rcModes, maxRefL0, maxRefL1, maxWidth, maxHeight, rtFormat, qualityLevels)
        && (rtFormat & VA_RT_FORMAT_YUV420_10) != 0;
}

tstring get_device_name(const char *vendorString) {
    if (vendorString == nullptr || vendorString[0] == '\0') {
        return _T("VA-API device");
    }
    std::string name(vendorString);
    const auto forPos = name.find(" for ");
    if (forPos != std::string::npos) {
        name = name.substr(forPos + 5);
        const auto detailPos = name.find(" (");
        if (detailPos != std::string::npos) name.resize(detailPos);
    }
    return char_to_tstring(name);
}

std::string read_pci_bus_id(const std::filesystem::path& renderNode) {
    const auto sysfsDevice = std::filesystem::path("/sys/class/drm") / renderNode.filename() / "device";
    std::error_code ec;
    const auto resolved = std::filesystem::canonical(sysfsDevice, ec);
    return ec ? std::string() : resolved.filename().string();
}

void va_info_callback(void *userContext, const char *message) {
    auto *log = (RGYLog *)userContext;
    if (log != nullptr && message != nullptr && message[0] != '\0') {
        log->write(RGY_LOG_DEBUG, RGY_LOGT_DEV, _T("libva: %s\n"), char_to_tstring(message).c_str());
    }
}

void va_error_callback(void *userContext, const char *message) {
    auto *log = (RGYLog *)userContext;
    if (log != nullptr && message != nullptr && message[0] != '\0') {
        log->write(RGY_LOG_WARN, RGY_LOGT_DEV, _T("libva: %s\n"), char_to_tstring(message).c_str());
    }
}

uint32_t read_pci_id(const std::filesystem::path& renderNode, const char *attribute) {
    const auto path = std::filesystem::path("/sys/class/drm") / renderNode.filename() / "device" / attribute;
    std::ifstream file(path);
    uint32_t value = 0;
    return file >> std::hex >> value ? value : 0;
}

bool is_vendor_render_node(const std::filesystem::path& renderNode, const uint32_t vendorId) {
    return vendorId != 0 && read_pci_id(renderNode, "vendor") == vendorId;
}

// openErrno: render node を開けなかったときの errno (開けた場合は 0)
bool probe_va_device(const std::filesystem::path& renderNode, tstring& name, int& openErrno, RGYLog *log) {
    openErrno = 0;
    const int fd = open(renderNode.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        openErrno = errno;
        if (log != nullptr) {
            log->write(RGY_LOG_DEBUG, RGY_LOGT_DEV, _T("Failed to open %s: %s.\n"),
                char_to_tstring(renderNode.string()).c_str(), char_to_tstring(strerror(openErrno)).c_str());
        }
        return false;
    }
    VADisplay display = vaGetDisplayDRM(fd);
    if (display != nullptr) {
        vaSetInfoCallback(display, va_info_callback, log);
        vaSetErrorCallback(display, va_error_callback, log);
    }
    int major = 0, minor = 0;
    const auto status = display ? vaInitialize(display, &major, &minor) : VA_STATUS_ERROR_INVALID_DISPLAY;
    if (status == VA_STATUS_SUCCESS) {
        name = get_device_name(vaQueryVendorString(display));
    }
    // vaInitialize に失敗した場合も vaGetDisplayDRM で確保した display は vaTerminate で解放する
    if (display != nullptr) {
        vaTerminate(display);
    }
    close(fd);
    return status == VA_STATUS_SUCCESS;
}

bool test_encoder_open(AVBufferRef *hwdevice, const AVCodec *codec, const bool tenBit) {
    if (hwdevice == nullptr || codec == nullptr) return false;
    AVBufferRef *framesRefRaw = av_hwframe_ctx_alloc(hwdevice);
    if (framesRefRaw == nullptr) return false;
    std::unique_ptr<AVBufferRef, RGYAVDeleter<AVBufferRef>> framesRef(framesRefRaw, RGYAVDeleter<AVBufferRef>(av_buffer_unref));
    auto *frames = (AVHWFramesContext *)framesRef->data;
    frames->format = AV_PIX_FMT_VAAPI;
    frames->sw_format = tenBit ? AV_PIX_FMT_P010 : AV_PIX_FMT_NV12;
    frames->width = VA_PROBE_WIDTH;
    frames->height = VA_PROBE_HEIGHT;
    frames->initial_pool_size = VA_PROBE_FRAME_POOL_SIZE;
    if (av_hwframe_ctx_init(framesRef.get()) < 0) return false;

    AVCodecContext *contextRaw = avcodec_alloc_context3(codec);
    if (contextRaw == nullptr) return false;
    std::unique_ptr<AVCodecContext, RGYAVDeleter<AVCodecContext>> context(contextRaw, RGYAVDeleter<AVCodecContext>(avcodec_free_context));
    context->width = frames->width;
    context->height = frames->height;
    context->time_base = AVRational{ 1, VA_DEFAULT_FRAME_RATE };
    context->framerate = AVRational{ VA_DEFAULT_FRAME_RATE, 1 };
    context->bit_rate = VA_DEFAULT_BITRATE;
    context->gop_size = VA_DEFAULT_GOP_LENGTH;
    context->max_b_frames = 0;
    context->pix_fmt = AV_PIX_FMT_VAAPI;
    if (tenBit && codec->id == AV_CODEC_ID_HEVC) {
        context->profile = AV_PROFILE_HEVC_MAIN_10;
    } else if (tenBit && codec->id == AV_CODEC_ID_VP9) {
        context->profile = AV_PROFILE_VP9_2;
    }
    context->hw_frames_ctx = av_buffer_ref(framesRef.get());
    if (context->hw_frames_ctx == nullptr) return false;

    // av_log のレベルはプロセス共通なので、並列子の試し開きを直列化する。
    static std::mutex probeLogMutex;
    const std::lock_guard<std::mutex> probeLogLock(probeLogMutex);
    const int previousLogLevel = av_log_get_level();
    av_log_set_level(AV_LOG_QUIET);
    struct AvLogLevelRestorer { int prev; ~AvLogLevelRestorer() { av_log_set_level(prev); } } avGuard{ previousLogLevel };
    const int openResult = avcodec_open2(context.get(), codec, nullptr);
    return openResult >= 0;
}

} // namespace

RGYVAEncCaps::RGYVAEncCaps() :
    available(false),
    support10bit(false),
    hasEncSlice(false),
    hasEncSliceLP(false),
    rcModes(0),
    maxRefL0(0),
    maxRefL1(0),
    maxWidth(0),
    maxHeight(0),
    qualityLevels(0) {
}

std::vector<RGYVADeviceInfo> enumerateVADevices(uint32_t vendorId, int idBase, RGYLog *log, tstring *openErrorMessage,
    const std::map<int, std::string>& preferredIds) {
    if (openErrorMessage != nullptr) openErrorMessage->clear();
    std::vector<std::filesystem::path> nodes;
    std::error_code ec;
    const std::filesystem::path drmPath("/dev/dri");
    for (std::filesystem::directory_iterator it(drmPath, ec), end; !ec && it != end; it.increment(ec)) {
        const auto name = it->path().filename().string();
        if (name.compare(0, 7, "renderD") == 0 && name.size() > 7
            && std::all_of(name.begin() + 7, name.end(), [](const char c) { return c >= '0' && c <= '9'; })) {
            nodes.push_back(it->path());
        }
    }
    std::sort(nodes.begin(), nodes.end(), [](const auto& a, const auto& b) {
        return std::stoi(a.filename().string().substr(7)) < std::stoi(b.filename().string().substr(7));
    });

    std::vector<RGYVADeviceInfo> devices;
    int lastOpenErrno = 0;
    std::filesystem::path lastOpenFailedNode;
    for (const auto& node : nodes) {
        if (!is_vendor_render_node(node, vendorId)) continue;
        tstring name;
        int openErrno = 0;
        if (!probe_va_device(node, name, openErrno, log)) {
            if (openErrno != 0) {
                lastOpenErrno = openErrno;
                lastOpenFailedNode = node;
            }
            continue;
        }
        RGYVADeviceInfo info;
        info.id = idBase + (int)devices.size();
        info.renderNode = char_to_tstring(node.string());
        info.pciBusId = read_pci_bus_id(node);
        info.vendorId = vendorId;
        info.deviceId = read_pci_id(node, "device");
        info.name = std::move(name);
        devices.push_back(std::move(info));
    }
    if (!preferredIds.empty()) {
        int nextId = std::max(idBase, preferredIds.rbegin()->first + 1);
        for (auto& device : devices) {
            const auto match = std::find_if(preferredIds.begin(), preferredIds.end(), [&device](const auto& preferred) {
                return preferred.second == device.pciBusId;
            });
            device.id = match != preferredIds.end() ? match->first : nextId++;
        }
        std::sort(devices.begin(), devices.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
    }
    if (log != nullptr) {
        for (const auto& device : devices) {
            log->write(RGY_LOG_DEBUG, RGY_LOGT_DEV, _T("VA-API device #%d: %s (%s, PCI %s)\n"),
                device.id, device.name.c_str(), device.renderNode.c_str(), char_to_tstring(device.pciBusId).c_str());
        }
    }
    // 指定ベンダーの render node がすべて使えず、その原因が open() の失敗だった場合は、
    // "VA-API unavailable" だけでは原因が分からないため、errno とヒントを出す。
    // ほかのベンダーの node は対象外。
    if (devices.empty() && lastOpenErrno != 0) {
        auto message = strsprintf(_T("Failed to open render node %s: %s.\n"),
            char_to_tstring(lastOpenFailedNode.string()).c_str(), char_to_tstring(strerror(lastOpenErrno)).c_str());
        if (lastOpenErrno == EACCES || lastOpenErrno == EPERM) {
            message += _T("  Add the user to the \"render\" group (e.g. sudo usermod -aG render $USER) and log in again.\n");
        }
        if (log != nullptr) {
            log->write(RGY_LOG_WARN, RGY_LOGT_DEV, _T("%s"), message.c_str());
        }
        if (openErrorMessage != nullptr) {
            *openErrorMessage = message;
        }
    }
    return devices;
}

RGYDeviceVA::RGYDeviceVA() :
    m_info(),
    m_vendorString(),
    m_hwdevice(nullptr, RGYAVDeleter<AVBufferRef>(av_buffer_unref)),
    m_display(nullptr),
    m_encCaps(),
    m_decCaps(),
    m_decCapsQueried(false),
    m_log() {
}

RGYDeviceVA::~RGYDeviceVA() {
}

RGY_ERR RGYDeviceVA::open(const RGYVADeviceInfo& info, std::shared_ptr<RGYLog> log) {
    m_info = info;
    m_log = std::move(log);
    AVBufferRef *deviceRaw = nullptr;
    const auto devicePath = tchar_to_string(m_info.renderNode);
    const int result = av_hwdevice_ctx_create(&deviceRaw, AV_HWDEVICE_TYPE_VAAPI, devicePath.c_str(), nullptr, 0);
    if (result < 0 || deviceRaw == nullptr) {
        if (m_log != nullptr) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE] = {};
            av_strerror(result, errbuf, sizeof(errbuf));
            m_log->write(RGY_LOG_ERROR, RGY_LOGT_DEV, _T("Failed to open VA-API device %s (%s).\n"),
                m_info.renderNode.c_str(), char_to_tstring(errbuf).c_str());
        }
        return RGY_ERR_DEVICE_NOT_FOUND;
    }
    m_hwdevice.reset(deviceRaw);
    const auto *deviceCtx = (const AVHWDeviceContext *)m_hwdevice->data;
    const auto *vaCtx = (const AVVAAPIDeviceContext *)deviceCtx->hwctx;
    m_display = vaCtx->display;
    if (m_display == nullptr) {
        m_hwdevice.reset();
        return RGY_ERR_DEVICE_NOT_FOUND;
    }
    const char *vendor = vaQueryVendorString((VADisplay)m_display);
    m_vendorString = vendor ? char_to_tstring(vendor) : _T("unknown");
    return RGY_ERR_NONE;
}

RGYVADriver RGYDeviceVA::driver() const {
    auto vendor = tchar_to_string(m_vendorString);
    std::transform(vendor.begin(), vendor.end(), vendor.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    if (vendor.find("mesa") != std::string::npos) return RGYVADriver::Mesa;
    if (vendor.find("ihd") != std::string::npos) return RGYVADriver::IntelIHD;
    if (vendor.find("i965") != std::string::npos) return RGYVADriver::IntelI965;
    return RGYVADriver::Unknown;
}

const RGYVAEncCaps& RGYDeviceVA::encCaps(RGY_CODEC codec) {
    const auto cached = m_encCaps.find(codec);
    if (cached != m_encCaps.end()) return cached->second;

    RGYVAEncCaps caps;
    const auto name = codec_name(codec);
    VAEntrypoint entrypoint = VAEntrypointEncSlice;
    if (m_display != nullptr && name != nullptr && has_encoding_entrypoint((VADisplay)m_display, va_profile(codec), entrypoint)) {
        caps.hasEncSlice = has_profile_entrypoint((VADisplay)m_display, va_profile(codec), VAEntrypointEncSlice);
        caps.hasEncSliceLP = has_profile_entrypoint((VADisplay)m_display, va_profile(codec), VAEntrypointEncSliceLP);
        uint32_t rtFormat = 0;
        query_profile_attributes((VADisplay)m_display, va_profile(codec), entrypoint,
            caps.rcModes, caps.maxRefL0, caps.maxRefL1, caps.maxWidth, caps.maxHeight, rtFormat, caps.qualityLevels);
        const bool supports10bit = supports_10bit((VADisplay)m_display, codec, entrypoint);

        const AVCodec *avcodec = avcodec_find_encoder_by_name(name);
        const bool opened8bit = test_encoder_open(m_hwdevice.get(), avcodec, false);
        const bool opened10bit = supports10bit && test_encoder_open(m_hwdevice.get(), avcodec, true);
        caps.support10bit = opened8bit && opened10bit;
        caps.available = avcodec != nullptr && opened8bit;
        if (m_log != nullptr) {
            m_log->write(RGY_LOG_DEBUG, RGY_LOGT_DEV, _T("VA-API encoder trial open %s (8-bit): %s\n"),
                char_to_tstring(name).c_str(), opened8bit ? _T("succeeded") : _T("failed"));
            if (supports10bit) {
                m_log->write(RGY_LOG_DEBUG, RGY_LOGT_DEV, _T("VA-API encoder trial open %s (10-bit): %s\n"),
                    char_to_tstring(name).c_str(), opened10bit ? _T("succeeded") : _T("failed"));
            }
        }
    }
    return m_encCaps.emplace(codec, caps).first->second;
}

tstring RGYDeviceVA::capsString(RGY_CODEC codec) {
    const auto& caps = encCaps(codec);
    tstring rcModes;
    const auto appendMode = [&rcModes](const uint32_t flag, const TCHAR *name) {
        if (flag) {
            if (!rcModes.empty()) rcModes += _T(", ");
            rcModes += name;
        }
    };
    appendMode(caps.rcModes & RGY_VA_RC_CBR, _T("CBR"));
    appendMode(caps.rcModes & RGY_VA_RC_VBR, _T("VBR"));
    appendMode(caps.rcModes & RGY_VA_RC_CQP, _T("CQP"));
    appendMode(caps.rcModes & RGY_VA_RC_QVBR, _T("QVBR"));
    appendMode(caps.rcModes & RGY_VA_RC_ICQ, _T("ICQ"));
    appendMode(caps.rcModes & RGY_VA_RC_AVBR, _T("AVBR"));
    if (rcModes.empty()) rcModes = _T("none");
    return strsprintf(_T("  available: %s\n  10-bit: %s\n  rate control: %s\n  max ref (L0/L1): %d/%d\n  max resolution: %dx%d\n  EncSlice/EncSliceLP: %s/%s\n  quality levels: %d"),
        caps.available ? _T("yes") : _T("no"), caps.support10bit ? _T("yes") : _T("no"), rcModes.c_str(), caps.maxRefL0, caps.maxRefL1, caps.maxWidth, caps.maxHeight,
        caps.hasEncSlice ? _T("yes") : _T("no"), caps.hasEncSliceLP ? _T("yes") : _T("no"), caps.qualityLevels);
}

const CodecCsp& RGYDeviceVA::decCaps() {
    if (!m_decCapsQueried) {
        m_decCaps = m_display != nullptr ? query_decode_caps((VADisplay)m_display) : CodecCsp();
        m_decCapsQueried = true;
    }
    return m_decCaps;
}

tstring RGYDeviceVA::decCapsString(RGY_CODEC codec) {
    const auto& caps = decCaps();
    const auto it = caps.find(codec);
    const bool supported = it != caps.end() && !it->second.empty();
    const bool support8bit = supported && std::find(it->second.begin(), it->second.end(), RGY_CSP_NV12) != it->second.end();
    const bool support10bit = supported && std::find(it->second.begin(), it->second.end(), RGY_CSP_P010) != it->second.end();
    tstring outputFormats;
    if (support8bit) outputFormats += _T("NV12");
    if (support10bit) {
        if (!outputFormats.empty()) outputFormats += _T(", ");
        outputFormats += _T("P010");
    }
    if (outputFormats.empty()) outputFormats = _T("none");
    return strsprintf(_T("  available:     %s\n  8bit depth:    %s\n  10bit depth:   %s\n  output format: %s"),
        supported ? _T("yes") : _T("no"), support8bit ? _T("yes") : _T("no"), support10bit ? _T("yes") : _T("no"), outputFormats.c_str());
}

RGYEncoderVA::RGYEncoderVA() :
    m_codecCtx(nullptr, RGYAVDeleter<AVCodecContext>(avcodec_free_context)),
    m_hwframes(nullptr, RGYAVDeleter<AVBufferRef>(av_buffer_unref)),
    m_frameHW(nullptr, RGYAVDeleter<AVFrame>(av_frame_free)),
    m_frameSW(nullptr, RGYAVDeleter<AVFrame>(av_frame_free)),
    m_pkt(nullptr, RGYAVDeleter<AVPacket>(av_packet_free)),
    m_log(), m_codec(RGY_CODEC_UNKNOWN), m_width(0), m_height(0), m_bitdepth(VA_DEFAULT_BIT_DEPTH), m_rateControl(RGY_VA_RC_CQP),
    m_qp(0), m_bframes(0), m_refs(0), m_compressionLevel(-1), m_tier(0), m_qpMin(), m_qpMax() {
}

RGYEncoderVA::~RGYEncoderVA() = default;

RGY_ERR RGYEncoderVA::init(RGYDeviceVA *dev, const RGYVAEncParam& prm, std::shared_ptr<RGYLog> log) {
    if (dev == nullptr || dev->hwdevice() == nullptr || prm.width <= 0 || prm.height <= 0
        || (prm.bitdepth != 8 && prm.bitdepth != 10) || prm.fps.n() <= 0 || prm.fps.d() <= 0
        || prm.timebase.n() <= 0 || prm.timebase.d() <= 0 || prm.sar.d() <= 0
        || prm.bframes < 0 || prm.refs < -1 || prm.lowPower < -1 || prm.lowPower > 1
        || prm.compressionLevel < -1 || prm.slices < 0) return RGY_ERR_INVALID_PARAM;
    m_log = std::move(log);
    m_codec = prm.codec;
    m_width = prm.width;
    m_height = prm.height;
    m_bitdepth = prm.bitdepth;
    m_rateControl = prm.rc;
    const AVRational avSar{ prm.sar.n(), prm.sar.d() };
    const AVRational avFps{ prm.fps.n(), prm.fps.d() };
    const AVRational avTimebase{ prm.timebase.n(), prm.timebase.d() };
    const auto& caps = dev->encCaps(prm.codec);
    const char *rcMode = rc_mode_name(prm.rc);
    if (rcMode == nullptr) return RGY_ERR_INVALID_PARAM;
    if (!caps.available || (prm.bitdepth > 8 && !caps.support10bit)) return RGY_ERR_UNSUPPORTED;
    if (!(caps.rcModes & prm.rc)) {
        if (m_log) m_log->write(RGY_LOG_ERROR, RGY_LOGT_DEV, _T("%s is not supported by this VA-API device.\n"), char_to_tstring(rcMode).c_str());
        return RGY_ERR_UNSUPPORTED;
    }
    if ((prm.lowPower == 0 && !caps.hasEncSlice) || (prm.lowPower == 1 && !caps.hasEncSliceLP)) {
        if (m_log) m_log->write(RGY_LOG_ERROR, RGY_LOGT_DEV, _T("VA-API low_power=%d is not supported by this device.\n"), prm.lowPower);
        return RGY_ERR_UNSUPPORTED;
    }
    const char *name = codec_name(prm.codec);
    const AVCodec *codec = name ? avcodec_find_encoder_by_name(name) : nullptr;
    if (codec == nullptr) {
        if (m_log) m_log->write(RGY_LOG_ERROR, RGY_LOGT_DEV, _T("VA-API encoder %s was not found.\n"), char_to_tstring(name ? name : "unknown").c_str());
        return RGY_ERR_UNSUPPORTED;
    }
    int maxBFrames = prm.bframes;
    if (maxBFrames > 0 && caps.maxRefL1 <= 0) {
        if (m_log) m_log->write(RGY_LOG_WARN, RGY_LOGT_DEV, _T("WARN: --bframes is not supported with --backend vaapi, ignored (device reports maxRefL1=0).\n"));
        maxBFrames = 0;
    }
    m_bframes = maxBFrames;
    m_refs = prm.refs >= 0 ? prm.refs : caps.maxRefL0;
    if (m_refs > caps.maxRefL0) {
        if (m_log) m_log->write(RGY_LOG_WARN, RGY_LOGT_DEV, _T("WARN: --ref %d exceeds the VA-API device limit (%d), using %d.\n"), m_refs, caps.maxRefL0, caps.maxRefL0);
        m_refs = caps.maxRefL0;
    }
    m_qp = prm.rc == RGY_VA_RC_QVBR || prm.rc == RGY_VA_RC_ICQ ? prm.quality : prm.qp.qpP;
    m_qpMin = prm.qpMin;
    m_qpMax = prm.qpMax;

    AVBufferRef *framesRaw = av_hwframe_ctx_alloc(dev->hwdevice());
    if (framesRaw == nullptr) return RGY_ERR_NULL_PTR;
    m_hwframes.reset(framesRaw);
    auto *frames = (AVHWFramesContext *)m_hwframes->data;
    frames->format = AV_PIX_FMT_VAAPI;
    frames->sw_format = prm.bitdepth > 8 ? AV_PIX_FMT_P010 : AV_PIX_FMT_NV12;
    frames->width = prm.width;
    frames->height = prm.height;
    frames->initial_pool_size = VA_ENCODER_FRAME_POOL_SIZE;
    int ret = av_hwframe_ctx_init(m_hwframes.get());
    if (ret < 0) return RGY_ERR_DEVICE_FAILED;

    AVCodecContext *ctxRaw = avcodec_alloc_context3(codec);
    if (ctxRaw == nullptr) return RGY_ERR_NULL_PTR;
    m_codecCtx.reset(ctxRaw);
    auto *ctx = m_codecCtx.get();
    ctx->width = prm.width;
    ctx->height = prm.height;
    ctx->bit_rate = (int64_t)prm.bitrateKbps * 1000;
    ctx->rc_max_rate = (int64_t)prm.maxBitrateKbps * 1000;
    ctx->rc_buffer_size = prm.vbvBufKbits * 1000;
    ctx->time_base = avTimebase;
    ctx->framerate = avFps;
    ctx->pix_fmt = AV_PIX_FMT_VAAPI;
    ctx->gop_size = prm.gopLen > 0 ? prm.gopLen : avFps.num * VA_DEFAULT_GOP_SECONDS / avFps.den;
    ctx->max_b_frames = maxBFrames;
    ctx->refs = m_refs;
    ctx->sample_aspect_ratio = avSar;
    ctx->profile = prm.profile;
    ctx->level = prm.level;
    if (prm.qpMin.has_value()) ctx->qmin = prm.qpMin.value();
    if (prm.qpMax.has_value()) ctx->qmax = prm.qpMax.value();
    ctx->color_primaries = (AVColorPrimaries)prm.vui.colorprim;
    ctx->color_trc = (AVColorTransferCharacteristic)prm.vui.transfer;
    ctx->colorspace = (AVColorSpace)prm.vui.matrix;
    ctx->color_range = (AVColorRange)prm.vui.colorrange;
    ctx->hw_frames_ctx = av_buffer_ref(m_hwframes.get());
    if (ctx->hw_frames_ctx == nullptr) return RGY_ERR_NULL_PTR;

    AVDictionary *opts = nullptr;
    av_dict_set(&opts, "rc_mode", rcMode, 0);
    // refs の未指定時はオプションを渡さず、従来の VCEEnc と同じ FFmpeg の既定値を保つ。
    if (prm.refs >= 0) av_dict_set_int(&opts, "refs", m_refs, 0);
    m_compressionLevel = prm.compressionLevel;
    if (prm.compressionLevel >= 0) ctx->compression_level = prm.compressionLevel;
    if (prm.lowPower >= 0) av_dict_set_int(&opts, "low_power", prm.lowPower, 0);
    if (prm.slices > 0) ctx->slices = prm.slices;
    if ((prm.codec == RGY_CODEC_H264 || prm.codec == RGY_CODEC_HEVC) && prm.aud) av_dict_set(&opts, "aud", "1", 0);
    m_tier = prm.tier;
    if (prm.codec == RGY_CODEC_HEVC || prm.codec == RGY_CODEC_AV1) av_dict_set_int(&opts, "tier", m_tier, 0);
    if (prm.rc == RGY_VA_RC_CQP) {
        ctx->flags |= AV_CODEC_FLAG_QSCALE;
        ctx->global_quality = prm.qp.qpP * FF_QP2LAMBDA;
        ctx->i_quant_factor = (float)prm.qp.qpI / (float)(std::max)(1, prm.qp.qpP);
        ctx->b_quant_factor = (float)prm.qp.qpB / (float)(std::max)(1, prm.qp.qpP);
    } else if (prm.rc == RGY_VA_RC_QVBR) {
        av_dict_set_int(&opts, "qp", prm.quality, 0);
    } else if (prm.rc == RGY_VA_RC_ICQ) {
        // QSCALE を立てない ICQ では FFmpeg は global_quality を品質値としてそのまま読む。
        ctx->global_quality = prm.quality;
    }
    ret = avcodec_open2(ctx, codec, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE] = {};
        av_strerror(ret, errbuf, sizeof(errbuf));
        if (m_log) m_log->write(RGY_LOG_ERROR, RGY_LOGT_DEV, _T("Failed to open VA-API encoder %s: %s.\n"), char_to_tstring(name).c_str(), char_to_tstring(errbuf).c_str());
        return RGY_ERR_UNSUPPORTED;
    }
    m_frameHW.reset(av_frame_alloc());
    m_frameSW.reset(av_frame_alloc());
    m_pkt.reset(av_packet_alloc());
    if (!m_frameHW || !m_frameSW || !m_pkt) return RGY_ERR_NULL_PTR;
    m_frameHW->format = AV_PIX_FMT_VAAPI;
    m_frameHW->width = prm.width;
    m_frameHW->height = prm.height;
    if (m_log) m_log->write(RGY_LOG_INFO, RGY_LOGT_DEV, _T("VA-API encoder initialized: %s %dx%d, %dbit, %d/%d fps.\n"),
        char_to_tstring(name).c_str(), prm.width, prm.height, m_bitdepth, avFps.num, avFps.den);
    return RGY_ERR_NONE;
}

RGY_ERR RGYEncoderVA::submit(RGYFrame *frame) {
    if (!m_codecCtx) return RGY_ERR_NOT_INITIALIZED;
    if (frame == nullptr) {
        const int ret = avcodec_send_frame(m_codecCtx.get(), nullptr);
        return ret == AVERROR(EAGAIN) ? RGY_ERR_MORE_DATA : (ret < 0 ? RGY_ERR_DEVICE_FAILED : RGY_ERR_NONE);
    }
    if (auto *direct = dynamic_cast<RGYFrameHWAVFrame *>(frame); direct && direct->avframe()->buf[0]) {
        const auto *source = direct->avframe();
        const auto *sourceFrames = source && source->hw_frames_ctx ? (const AVHWFramesContext *)source->hw_frames_ctx->data : nullptr;
        const auto *encoderFrames = m_hwframes ? (const AVHWFramesContext *)m_hwframes->data : nullptr;
        const auto *sourceDevice = sourceFrames && sourceFrames->device_ctx ? (const AVVAAPIDeviceContext *)sourceFrames->device_ctx->hwctx : nullptr;
        const auto *encoderDevice = encoderFrames && encoderFrames->device_ctx ? (const AVVAAPIDeviceContext *)encoderFrames->device_ctx->hwctx : nullptr;
        if (!source || source->format != AV_PIX_FMT_VAAPI || !sourceDevice || !encoderDevice
            || sourceDevice->display != encoderDevice->display
            || source->width != m_width || source->height != m_height
            || sourceFrames->sw_format != encoderFrames->sw_format) return RGY_ERR_UNSUPPORTED;
        av_frame_unref(m_frameHW.get());
        if (av_frame_ref(m_frameHW.get(), source) < 0) return RGY_ERR_NULL_PTR;
        m_frameHW->pts = direct->timestamp();
        m_frameHW->duration = direct->duration();
        m_frameHW->pict_type = AV_PICTURE_TYPE_NONE;
        m_frameHW->flags &= ~AV_FRAME_FLAG_KEY;
        const int ret = avcodec_send_frame(m_codecCtx.get(), m_frameHW.get());
        return ret == AVERROR(EAGAIN) ? RGY_ERR_MORE_DATA : (ret < 0 ? RGY_ERR_DEVICE_FAILED : RGY_ERR_NONE);
    }
    auto *sys = dynamic_cast<RGYSysFrame *>(frame);
    if (sys == nullptr) return RGY_ERR_UNSUPPORTED;
    const auto& info = sys->frameInfo();
    if (info.width != m_width || info.height != m_height) return RGY_ERR_INVALID_VIDEO_PARAM;
    av_frame_unref(m_frameSW.get());
    m_frameSW->format = (m_bitdepth > 8) ? AV_PIX_FMT_P010 : AV_PIX_FMT_NV12;
    m_frameSW->width = m_width;
    m_frameSW->height = m_height;
    for (int plane = 0; plane < 4; plane++) {
        m_frameSW->data[plane] = info.ptr[plane];
        m_frameSW->linesize[plane] = info.pitch[plane];
    }
    m_frameSW->pts = info.timestamp;
    m_frameSW->duration = (int64_t)info.duration;
    m_frameSW->pict_type = AV_PICTURE_TYPE_NONE;
    av_frame_unref(m_frameHW.get());
    m_frameHW->format = AV_PIX_FMT_VAAPI;
    m_frameHW->width = m_width;
    m_frameHW->height = m_height;
    int ret = av_hwframe_get_buffer(m_hwframes.get(), m_frameHW.get(), 0);
    if (ret < 0) return RGY_ERR_DEVICE_FAILED;
    ret = av_hwframe_transfer_data(m_frameHW.get(), m_frameSW.get(), 0);
    if (ret < 0) return RGY_ERR_DEVICE_FAILED;
    m_frameHW->pts = info.timestamp;
    m_frameHW->duration = (int64_t)info.duration;
    ret = avcodec_send_frame(m_codecCtx.get(), m_frameHW.get());
    return ret == AVERROR(EAGAIN) ? RGY_ERR_MORE_DATA : (ret < 0 ? RGY_ERR_DEVICE_FAILED : RGY_ERR_NONE);
}

RGY_ERR RGYEncoderVA::receive(std::shared_ptr<RGYBitstream>& bs) {
    bs.reset();
    if (!m_codecCtx || !m_pkt) return RGY_ERR_NOT_INITIALIZED;
    const int ret = avcodec_receive_packet(m_codecCtx.get(), m_pkt.get());
    if (ret == AVERROR(EAGAIN)) return RGY_ERR_MORE_DATA;
    if (ret == AVERROR_EOF) return RGY_ERR_MORE_BITSTREAM;
    if (ret < 0) return RGY_ERR_DEVICE_FAILED;
    auto output = std::make_shared<RGYBitstream>(RGYBitstreamInit());
    const int64_t pts = m_pkt->pts == AV_NOPTS_VALUE ? 0 : m_pkt->pts;
    const int64_t dts = m_pkt->dts == AV_NOPTS_VALUE ? pts : m_pkt->dts;
    const auto duration = m_pkt->duration;
    const auto copyErr = output->copy(m_pkt->data, m_pkt->size);
    if (copyErr != RGY_ERR_NONE) return copyErr;
    if (m_codec == RGY_CODEC_AV1) {
        const auto units = parse_unit_av1(output->data(), output->size());
        const auto hasTemporalDelimiter = std::find_if(units.begin(), units.end(), [](const auto& unit) {
            return unit->type == OBU_TEMPORAL_DELIMITER;
        }) != units.end();
        if (!hasTemporalDelimiter) {
            // 後段はTemporal DelimiterでAV1フレームを分割するため、VAAPI出力に無い場合は補う。
            std::vector<uint8_t> packetWithTemporalDelimiter{ 0x12, 0x00 };
            packetWithTemporalDelimiter.insert(packetWithTemporalDelimiter.end(), output->data(), output->data() + output->size());
            const auto prependErr = output->copy(packetWithTemporalDelimiter.data(), packetWithTemporalDelimiter.size());
            if (prependErr != RGY_ERR_NONE) return prependErr;
        }
    }
    // 各アプリの copy() の引数差に依存せず、共通の setter で時刻と長さを引き継ぐ。
    output->setPts(pts);
    output->setDts(dts);
    output->setDuration(duration);
    output->setFrametype((m_pkt->flags & AV_PKT_FLAG_KEY) ? RGY_FRAMETYPE_IDR : RGY_FRAMETYPE_P);
    bs = std::move(output);
    av_packet_unref(m_pkt.get());
    return RGY_ERR_NONE;
}

tstring RGYEncoderVA::profileString() const {
    if (!m_codecCtx) return _T("auto");
    const char *profile = avcodec_profile_name(m_codecCtx->codec_id, m_codecCtx->profile);
    return profile ? char_to_tstring(profile) : _T("auto");
}

tstring RGYEncoderVA::levelString() const {
    if (!m_codecCtx || m_codecCtx->level == AV_LEVEL_UNKNOWN) return _T("auto");
    const int level = m_codecCtx->level;
    if (m_codec == RGY_CODEC_AV1) return strsprintf(_T("%d.%d"), 2 + level / 4, level % 4);
    const int scale = m_codec == RGY_CODEC_HEVC ? 30 : 10;
    if (m_codec == RGY_CODEC_H264 && level == 9) return _T("1b");
    return strsprintf(_T("%d.%d"), level / scale, (level % scale) / (scale / 10));
}

tstring RGYEncoderVA::tierString() const {
    return m_codec == RGY_CODEC_HEVC ? (m_tier ? _T("high") : _T("main")) : _T("");
}

tstring RGYEncoderVA::paramString() const {
    if (!m_codecCtx) return _T("VA-API encoder is not initialized.\n");
    tstring mes = strsprintf(_T("Compression:   %d\n"), m_compressionLevel);
    if (m_rateControl == RGY_VA_RC_CQP) {
        const int qpP = m_codecCtx->global_quality / FF_QP2LAMBDA;
        const int qpI = (int)(qpP * m_codecCtx->i_quant_factor);
        const int qpB = (int)(qpP * m_codecCtx->b_quant_factor);
        mes += strsprintf(_T("CQP:           %s:%d, %s:%d"),
            m_codec == RGY_CODEC_AV1 ? _T("Intra") : _T("I"), qpI,
            m_codec == RGY_CODEC_AV1 ? _T("Inter") : _T("P"), qpP);
        if (m_bframes > 0) {
            mes += strsprintf(_T(", %s:%d"), m_codec == RGY_CODEC_AV1 ? _T("InterB") : _T("B"), qpB);
        }
        mes += _T("\n");
    } else {
        if (m_rateControl == RGY_VA_RC_ICQ) {
            mes += strsprintf(_T("ICQ:           Quality %d\n"), m_qp);
        } else {
            mes += strsprintf(_T("%-15s%lld kbps\n"), (char_to_tstring(rc_mode_name(m_rateControl)) + _T(":")).c_str(), (long long)(m_codecCtx->bit_rate / 1000));
        }
        if (m_rateControl == RGY_VA_RC_QVBR) {
            mes += strsprintf(_T("Quality level: %d\n"), m_qp);
        }
        if (m_codecCtx->rc_max_rate > 0) {
            mes += strsprintf(_T("Max bitrate:   %lld kbps\n"), (long long)(m_codecCtx->rc_max_rate / 1000));
        }
        if (m_qpMin.has_value() || m_qpMax.has_value()) {
            const auto qmin = m_qpMin.has_value() ? strsprintf(_T("%d"), m_qpMin.value()) : tstring(_T("auto"));
            const auto qmax = m_qpMax.has_value() ? strsprintf(_T("%d"), m_qpMax.value()) : tstring(_T("auto"));
            mes += strsprintf(_T("QP:            Min: %s:%s, Max: %s:%s\n"), qmin.c_str(), qmin.c_str(), qmax.c_str(), qmax.c_str());
        }
    }
    if (m_codecCtx->rc_buffer_size > 0) {
        mes += strsprintf(_T("VBV Bufsize:   %d kb\n"), m_codecCtx->rc_buffer_size / 1000);
    }
    // H.264でBフレームを使わない場合はBframesの行を出さない。
    if (m_bframes > 0 || m_codec != RGY_CODEC_H264) {
        mes += strsprintf(_T("Bframes:       %d frames\n"), m_bframes);
    }
    mes += strsprintf(_T("Ref frames:    %d frames\nGOP Len:       %d frames\n"), m_refs, m_codecCtx->gop_size);
    return mes;
}

#endif // ENABLE_VAAPI
