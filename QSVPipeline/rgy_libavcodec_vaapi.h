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

#pragma once

#include "rgy_version.h"

#if ENABLE_VAAPI

#include <optional>
#include <map>
#include <unordered_map>
#include <vector>
#include "rgy_avutil.h"
#include "rgy_def.h"
#include "rgy_err.h"
#include "rgy_log.h"
#include "rgy_prm.h"

class RGYFrame;
struct RGYBitstream;

enum RGYVAEncRCMode : uint32_t {
    RGY_VA_RC_CBR  = 1u << 0,
    RGY_VA_RC_VBR  = 1u << 1,
    RGY_VA_RC_CQP  = 1u << 2,
    RGY_VA_RC_QVBR = 1u << 3,
    RGY_VA_RC_ICQ  = 1u << 4,
    RGY_VA_RC_AVBR = 1u << 5,
};

enum class RGYVADriver { Unknown, Mesa, IntelIHD, IntelI965 };

struct RGYVADeviceInfo {
    int         id = -1;
    tstring     renderNode;
    std::string pciBusId;
    uint32_t    vendorId = 0;
    uint32_t    deviceId = 0;
    tstring     name;
};

struct RGYVAEncCaps {
    bool     available;
    bool     support10bit;
    bool     hasEncSlice;
    bool     hasEncSliceLP;
    uint32_t rcModes;
    int      maxRefL0;
    int      maxRefL1;
    int      maxWidth;
    int      maxHeight;
    int      qualityLevels;

    RGYVAEncCaps();
};

// 指定ベンダーの render node を idBase 始まりで列挙する。preferredIds は既存 backend の PCI バス ID と番号の対応。
// openErrorMessage は対象 node を開けなかった理由を返す。
// probeDevices=false は VA を開かず、ベンダーが一致した全 node を番号付きで返す (名前は空)。
std::vector<RGYVADeviceInfo> enumerateVADevices(uint32_t vendorId, int idBase, RGYLog *log, tstring *openErrorMessage = nullptr,
    const std::map<int, std::string>& preferredIds = {}, bool probeDevices = true);

class RGYDeviceVA {
public:
    RGYDeviceVA();
    ~RGYDeviceVA();
    RGY_ERR open(const RGYVADeviceInfo& info, std::shared_ptr<RGYLog> log, RGYLogLevel errorLogLevel = RGY_LOG_ERROR, tstring *openErrorMessage = nullptr);
    const RGYVAEncCaps& encCaps(RGY_CODEC codec);
    tstring capsString(RGY_CODEC codec);
    const CodecCsp& decCaps();
    tstring decCapsString(RGY_CODEC codec);
    RGYVADriver driver() const;
    void *display() const { return m_display; }
    const RGYVADeviceInfo& info() const { return m_info; }
    const tstring& vendorString() const { return m_vendorString; }
    AVBufferRef *hwdevice() { return m_hwdevice.get(); }
protected:
    RGYVADeviceInfo m_info;
    tstring m_vendorString;
    std::unique_ptr<AVBufferRef, RGYAVDeleter<AVBufferRef>> m_hwdevice;
    void *m_display;
    std::unordered_map<RGY_CODEC, RGYVAEncCaps> m_encCaps;
    CodecCsp m_decCaps;
    bool m_decCapsQueried;
    std::shared_ptr<RGYLog> m_log;
};

// 値は FFmpeg の表記で指定する。アプリ固有の既定値・preset・profile の変換は呼び出し側で行う。
struct RGYVAEncParam {
    RGY_CODEC codec = RGY_CODEC_UNKNOWN;
    int width = 0, height = 0, bitdepth = 8;
    rgy_rational<int> sar = { 1, 1 }, fps = { 30, 1 }, timebase = { 1, 30 };
    RGYVAEncRCMode rc = RGY_VA_RC_CQP;
    RGYQPSet qp = { 23, 23, 23 };
    int quality = 23;
    int bitrateKbps = 0, maxBitrateKbps = 0, vbvBufKbits = 0;
    int gopLen = 0; // 0 は fps から2秒分を設定する。
    int bframes = 0;
    int refs = -1; // -1 はデバイスの上限を使い、FFmpeg の refs オプションを明示しない。
    int profile = AV_PROFILE_UNKNOWN, level = AV_LEVEL_UNKNOWN, tier = 0;
    std::optional<int> qpMin, qpMax;
    int compressionLevel = -1; // -1 は指定なし。Mesa のビット値の解釈は呼び出し側の担当。
    int lowPower = -1; // -1 は FFmpeg に任せる。
    int slices = 0; // 0 は指定なし。
    bool aud = false;
    VideoVUIInfo vui;
};

class RGYEncoderVA {
public:
    RGYEncoderVA();
    ~RGYEncoderVA();
    RGY_ERR init(RGYDeviceVA *dev, const RGYVAEncParam& prm, std::shared_ptr<RGYLog> log);
    RGY_ERR submit(RGYFrame *frame);
    RGY_ERR receive(std::shared_ptr<RGYBitstream>& bs);
    tstring profileString() const;
    tstring levelString() const;
    tstring tierString() const;
    int width() const { return m_width; }
    int height() const { return m_height; }
    int bitdepth() const { return m_bitdepth; }
    RGY_CODEC codec() const { return m_codec; }
    RGYVAEncRCMode rateControl() const { return m_rateControl; }
    int quality() const { return m_qp; }
    int bframes() const { return m_bframes; }
    int refs() const { return m_refs; }
    int compressionLevel() const { return m_compressionLevel; }
    std::optional<int> qpMin() const { return m_qpMin; }
    std::optional<int> qpMax() const { return m_qpMax; }
    RGYQPSet qp() const;
    int64_t bitrateKbps() const;
    int64_t maxBitrateKbps() const;
    int vbvBufKbits() const;
    int gopLen() const;
    int asyncDepth() const;
    int profile() const;
    int lowPower() const;
    int videoDelay() const { return (m_codec != RGY_CODEC_AV1 && m_bframes > 0) ? 1 : 0; }
protected:
    std::unique_ptr<AVCodecContext, RGYAVDeleter<AVCodecContext>> m_codecCtx;
    std::unique_ptr<AVBufferRef, RGYAVDeleter<AVBufferRef>> m_hwframes;
    std::unique_ptr<AVFrame, RGYAVDeleter<AVFrame>> m_frameHW;
    std::unique_ptr<AVFrame, RGYAVDeleter<AVFrame>> m_frameSW;
    std::unique_ptr<AVPacket, RGYAVDeleter<AVPacket>> m_pkt;
    std::shared_ptr<RGYLog> m_log;
    RGY_CODEC m_codec;
    int m_width, m_height, m_bitdepth;
    RGYVAEncRCMode m_rateControl;
    int m_qp;
    int m_bframes;
    int m_refs;
    int m_compressionLevel;
    int m_tier;
    std::optional<int> m_qpMin;
    std::optional<int> m_qpMax;
};

#endif // ENABLE_VAAPI
