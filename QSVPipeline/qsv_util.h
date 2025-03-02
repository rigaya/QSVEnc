// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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
// --------------------------------------------------------------------------------------------

#ifndef _QSV_UTIL_H_
#define _QSV_UTIL_H_

#include "rgy_tchar.h"
#include <emmintrin.h>
#include <vector>
#include <array>
#include <utility>
#include <string>
#include <chrono>
#include <memory>
#include <type_traits>
#include "rgy_osdep.h"
#include "rgy_version.h"
#include "cpu_info.h"
#include "gpu_info.h"
#include "rgy_util.h"
#include "convert_csp.h"
#include "qsv_prm.h"
#include "rgy_err.h"
#include "rgy_frame_info.h"
#include "rgy_opencl.h"

using std::vector;
using std::unique_ptr;
using std::shared_ptr;

class RGYFrameData;

#define MFX_HANDLE_IDIRECT3D9EX ((mfxHandleType)-1)

#define INIT_MFX_EXT_BUFFER(x, id) { RGY_MEMSET_ZERO(x); (x).Header.BufferId = (id); (x).Header.BufferSz = sizeof(x); }

MAP_PAIR_0_1_PROTO(codec, rgy, RGY_CODEC, enc, mfxU32);
//MAP_PAIR_0_1_PROTO(chromafmt, rgy, RGY_CHROMAFMT, enc, mfxU16);
MAP_PAIR_0_1_PROTO(csp, rgy, RGY_CSP, enc, mfxU32);
MAP_PAIR_0_1_PROTO(resize_algo, rgy, RGY_VPP_RESIZE_ALGO, enc, int);
MAP_PAIR_0_1_PROTO(resize_mode, rgy, RGY_VPP_RESIZE_MODE, enc, int);

mfxU16 picstruct_rgy_to_enc(RGY_PICSTRUCT picstruct);
RGY_PICSTRUCT picstruct_enc_to_rgy(mfxU16 picstruct);
mfxU16 mfx_fourcc_to_chromafmt(mfxU32 fourcc);
mfxFrameInfo frameinfo_rgy_to_enc(VideoInfo info);
mfxFrameInfo frameinfo_rgy_to_enc(const RGYFrameInfo& info, const rgy_rational<int> fps, const rgy_rational<int> sar, const int blockSize);
VideoInfo videooutputinfo(const mfxInfoMFX& mfx, const mfxExtVideoSignalInfo& vui, const mfxExtChromaLocInfo& chromaloc);
VideoInfo videooutputinfo(const mfxFrameInfo& frameinfo);

static inline RGY_FRAMETYPE frametype_enc_to_rgy(const uint32_t frametype) {
    RGY_FRAMETYPE type = RGY_FRAMETYPE_UNKNOWN;
    type |=  (MFX_FRAMETYPE_IDR  & frametype) ? RGY_FRAMETYPE_IDR : RGY_FRAMETYPE_UNKNOWN;
    type |=  (MFX_FRAMETYPE_I    & frametype) ? RGY_FRAMETYPE_I   : RGY_FRAMETYPE_UNKNOWN;
    type |=  (MFX_FRAMETYPE_P    & frametype) ? RGY_FRAMETYPE_P   : RGY_FRAMETYPE_UNKNOWN;
    type |=  (MFX_FRAMETYPE_B    & frametype) ? RGY_FRAMETYPE_B   : RGY_FRAMETYPE_UNKNOWN;
    type |=  (MFX_FRAMETYPE_xIDR & frametype) ? RGY_FRAMETYPE_xIDR : RGY_FRAMETYPE_UNKNOWN;
    type |=  (MFX_FRAMETYPE_xI   & frametype) ? RGY_FRAMETYPE_xI   : RGY_FRAMETYPE_UNKNOWN;
    type |=  (MFX_FRAMETYPE_xP   & frametype) ? RGY_FRAMETYPE_xP   : RGY_FRAMETYPE_UNKNOWN;
    type |=  (MFX_FRAMETYPE_xB   & frametype) ? RGY_FRAMETYPE_xB   : RGY_FRAMETYPE_UNKNOWN;
    return type;
}

static inline uint16_t frametype_rgy_to_enc(const RGY_FRAMETYPE frametype) {
    uint32_t type = MFX_FRAMETYPE_UNKNOWN;
    type |=  (RGY_FRAMETYPE_IDR  & frametype) ? MFX_FRAMETYPE_IDR : MFX_FRAMETYPE_UNKNOWN;
    type |=  (RGY_FRAMETYPE_I    & frametype) ? MFX_FRAMETYPE_I   : MFX_FRAMETYPE_UNKNOWN;
    type |=  (RGY_FRAMETYPE_P    & frametype) ? MFX_FRAMETYPE_P   : MFX_FRAMETYPE_UNKNOWN;
    type |=  (RGY_FRAMETYPE_B    & frametype) ? MFX_FRAMETYPE_B   : MFX_FRAMETYPE_UNKNOWN;
    type |=  (RGY_FRAMETYPE_xIDR & frametype) ? MFX_FRAMETYPE_xIDR : MFX_FRAMETYPE_UNKNOWN;
    type |=  (RGY_FRAMETYPE_xI   & frametype) ? MFX_FRAMETYPE_xI   : MFX_FRAMETYPE_UNKNOWN;
    type |=  (RGY_FRAMETYPE_xP   & frametype) ? MFX_FRAMETYPE_xP   : MFX_FRAMETYPE_UNKNOWN;
    type |=  (RGY_FRAMETYPE_xB   & frametype) ? MFX_FRAMETYPE_xB   : MFX_FRAMETYPE_UNKNOWN;
    return (uint16_t)type;
}

static bool fourccShiftUsed(const uint32_t fourcc) {
    const auto csp = csp_enc_to_rgy(fourcc);
    return cspShiftUsed(csp);
}

static int getEncoderBitdepth(const sInputParams *pParams) {
    switch (pParams->codec) {
    case RGY_CODEC_HEVC:
    case RGY_CODEC_VP9:
    case RGY_CODEC_AV1:
    case RGY_CODEC_RAW:
        return pParams->outputDepth;
    case RGY_CODEC_H264:
    case RGY_CODEC_VP8:
    case RGY_CODEC_MPEG2:
    case RGY_CODEC_VC1:
        break;
    default:
        return 0;
    }
    return 8;
}

static bool gopRefDistAsBframe(const RGY_CODEC codec) {
    return codec == RGY_CODEC_H264 || codec == RGY_CODEC_HEVC || codec == RGY_CODEC_MPEG2;
}

// QSVでRGBエンコードの際、RGY_CSP_VUYA扱いとする色空間
static const RGY_CSP RGY_CSP_MFX_RGB = RGY_CSP_RBGA32;

static RGY_CSP getMFXCsp(const RGY_CHROMAFMT chroma, const int bitdepth) {
    if (bitdepth > 8) {
        switch (chroma) {
        case RGY_CHROMAFMT_YUV420: return RGY_CSP_P010;
        case RGY_CHROMAFMT_YUV422: return RGY_CSP_Y210;
        case RGY_CHROMAFMT_YUV444: return (bitdepth > 10) ? RGY_CSP_Y416 : RGY_CSP_Y410;
        case RGY_CHROMAFMT_RGB:    return (bitdepth > 10) ? RGY_CSP_RBGA64 : RGY_CSP_RBGA64_10;
        default: return RGY_CSP_NA;
        }
    }
    switch (chroma) {
    case RGY_CHROMAFMT_YUV420: return RGY_CSP_NV12;
    case RGY_CHROMAFMT_YUV422: return RGY_CSP_YUY2;
    case RGY_CHROMAFMT_YUV444: return RGY_CSP_VUYA;
    case RGY_CHROMAFMT_RGB:    return RGY_CSP_MFX_RGB;
    default: return RGY_CSP_NA;
    }
}

static RGY_CSP getMFXCsp(const RGY_CSP csp) {
    return getMFXCsp(RGY_CSP_CHROMA_FORMAT[csp], RGY_CSP_BIT_DEPTH[csp]);
}

mfxFrameInfo toMFXFrameInfo(VideoInfo info);

tstring qsv_memtype_str(uint32_t memtype);

mfxHandleType mfxHandleTypeFromMemType(const MemType memType, const bool forOpenCLInterop);

static inline uint16_t check_coding_option(uint16_t value) {
    if (value == MFX_CODINGOPTION_UNKNOWN
        || value == MFX_CODINGOPTION_ON
        || value == MFX_CODINGOPTION_OFF
        || value == MFX_CODINGOPTION_ADAPTIVE) {
        return value;
    }
    return MFX_CODINGOPTION_UNKNOWN;
}

template<typename T>
static inline T get3state(const std::optional<bool>& value, const T valDefault, const T valOn, const T valOff) {
    if (!value.has_value()) {
        return valDefault;
    }
    return value.value() ? valOn : valOff;
}

static inline uint16_t get_codingopt(const std::optional<bool>& value) {
    return (uint16_t)get3state(value, MFX_CODINGOPTION_UNKNOWN, MFX_CODINGOPTION_ON, MFX_CODINGOPTION_OFF);
}

VideoInfo videooutputinfo(const mfxInfoMFX& mfx, const mfxExtVideoSignalInfo& vui);

bool isRCBitrateMode(int encmode);

struct RGYBitstream {
private:
    mfxBitstream m_bitstream;
    RGYFrameData **frameDataList;
    int frameDataNum;
    int64_t frameIndex;

public:
    mfxBitstream *bsptr() {
        return &m_bitstream;
    }
    const mfxBitstream *bsptr() const {
        return &m_bitstream;
    }
    mfxBitstream& bitstream() {
        return m_bitstream;
    }
    const mfxBitstream &bitstream() const {
        return m_bitstream;
    }

    uint8_t *bufptr() const {
        return m_bitstream.Data;
    }

    uint8_t *data() const {
        return m_bitstream.Data + m_bitstream.DataOffset;
    }

    uint32_t dataflag() const {
        return m_bitstream.DataFlag;
    }

    void setDataflag(uint32_t flag) {
        m_bitstream.DataFlag = (uint16_t)flag;
    }

    RGY_FRAMETYPE frametype() const {
        return frametype_enc_to_rgy(m_bitstream.FrameType);
    }

    void setFrametype(RGY_FRAMETYPE frametype) {
        m_bitstream.FrameType = frametype_rgy_to_enc(frametype);
    }

    RGY_PICSTRUCT picstruct() const {
        return picstruct_enc_to_rgy(m_bitstream.PicStruct);
    }

    void setPicstruct(RGY_PICSTRUCT picstruct) {
        m_bitstream.PicStruct = picstruct_rgy_to_enc(picstruct);
    }

    int duration() {
        return 0;
    }

    void setDuration(int64_t duration) {
        UNREFERENCED_PARAMETER(duration);
    }

    int64_t frameIdx() {
        return frameIndex;
    }

    void setFrameIdx(int64_t frameIdx) {
        frameIndex = frameIdx;
    }

    size_t size() const {
        return m_bitstream.DataLength;
    }

    void setSize(size_t size) {
        m_bitstream.DataLength = (uint32_t)size;
    }

    size_t offset() const {
        return m_bitstream.DataOffset;
    }

    void addOffset(size_t add) {
        m_bitstream.DataOffset += (uint32_t)add;
    }

    void setOffset(size_t offset) {
        m_bitstream.DataOffset = (uint32_t)offset;
    }

    size_t bufsize() const {
        return m_bitstream.MaxLength;
    }

    void setPts(int64_t pts) {
        m_bitstream.TimeStamp = pts;
    }

    int64_t pts() const {
        return m_bitstream.TimeStamp;
    }

    void setDts(int64_t dts) {
        m_bitstream.DecodeTimeStamp = dts;
    }

    int64_t dts() const {
        return m_bitstream.DecodeTimeStamp;
    }

    uint32_t avgQP() {
        return 0;
    }

    void setAvgQP(uint32_t avgQP) {
        UNREFERENCED_PARAMETER(avgQP);
    }

    void free_mem() {
        if (m_bitstream.Data) {
            _aligned_free(m_bitstream.Data);
            m_bitstream.Data = nullptr;
        }
    }

    void clear() {
        free_mem();
        memset(&m_bitstream, 0, sizeof(m_bitstream));
    }

    RGY_ERR init(size_t nSize) {
        free_mem();

        if (nSize > 0) {
            if (nullptr == (m_bitstream.Data = (uint8_t *)_aligned_malloc(nSize, 32))) {
                return RGY_ERR_NULL_PTR;
            }

            m_bitstream.MaxLength = (uint32_t)nSize;
        }
        return RGY_ERR_NONE;
    }

    void trim() {
        if (m_bitstream.DataOffset > 0 && m_bitstream.DataLength > 0) {
            memmove(m_bitstream.Data, m_bitstream.Data + m_bitstream.DataOffset, m_bitstream.DataLength);
            m_bitstream.DataOffset = 0;
        }
    }

    RGY_ERR copy(const uint8_t *setData, size_t setSize) {
        if (setData == nullptr || setSize == 0) {
            return RGY_ERR_MORE_BITSTREAM;
        }
        if (m_bitstream.MaxLength < setSize) {
            free_mem();
            auto sts = init(setSize);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        m_bitstream.DataLength = (uint32_t)setSize;
        m_bitstream.DataOffset = 0;
        memcpy(m_bitstream.Data, setData, setSize);
        return RGY_ERR_NONE;
    }

    RGY_ERR copy(const uint8_t *setData, size_t setSize, int64_t dts, int64_t pts) {
        auto sts = copy(setData, setSize);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_bitstream.DecodeTimeStamp = dts;
        m_bitstream.TimeStamp = pts;
        return RGY_ERR_NONE;
    }

    RGY_ERR copy(const RGYBitstream *pBitstream) {
        auto sts = copy(pBitstream->data(), pBitstream->size());
        if (sts != RGY_ERR_NONE) {
            return sts;
        }

        auto ptr = m_bitstream.Data;
        auto offset = m_bitstream.DataOffset;
        auto datalength = m_bitstream.DataLength;
        auto maxLength = m_bitstream.MaxLength;

        memcpy(&m_bitstream, pBitstream, sizeof(pBitstream[0]));

        m_bitstream.Data = ptr;
        m_bitstream.DataLength = datalength;
        m_bitstream.DataOffset = offset;
        m_bitstream.MaxLength = maxLength;
        return RGY_ERR_NONE;
    }

    RGY_ERR resize(size_t nNewSize) {
        if (m_bitstream.MaxLength < nNewSize) {
            uint8_t *pData = (uint8_t *)_aligned_malloc(nNewSize, 32);
            if (pData == nullptr) {
                return RGY_ERR_NULL_PTR;
            }
            if (m_bitstream.DataLength > 0) {
                uint32_t copyLength = std::min<uint32_t>((uint32_t)m_bitstream.DataLength, (uint32_t)nNewSize);
                memcpy(pData, m_bitstream.Data + m_bitstream.DataOffset, copyLength);
            }
            free_mem();
            m_bitstream.Data = pData;
            m_bitstream.DataOffset = 0;
            m_bitstream.DataLength = (uint32_t)nNewSize;
            m_bitstream.MaxLength = (uint32_t)nNewSize;
            return RGY_ERR_NONE;
        }
        if (m_bitstream.DataLength > 0 && m_bitstream.MaxLength < nNewSize + m_bitstream.DataOffset) {
            uint32_t copyLength = std::min<uint32_t>((uint32_t)m_bitstream.DataLength, (uint32_t)nNewSize);
            memmove(m_bitstream.Data, m_bitstream.Data + m_bitstream.DataOffset, copyLength);
            m_bitstream.DataOffset = 0;
            m_bitstream.DataLength = (uint32_t)nNewSize;
            return RGY_ERR_NONE;
        }
        if (m_bitstream.DataLength == 0) {
            m_bitstream.DataOffset = 0;
        }
        m_bitstream.DataLength = (uint32_t)nNewSize;
        return RGY_ERR_NONE;
    }

    RGY_ERR changeSize(size_t nNewSize) {
        uint8_t *pData = (uint8_t *)_aligned_malloc(nNewSize, 32);
        if (pData == nullptr) {
            return RGY_ERR_NULL_PTR;
        }

        size_t nDataLen = m_bitstream.DataLength;
        if (m_bitstream.DataLength) {
            memcpy(pData, m_bitstream.Data + m_bitstream.DataOffset, (std::min)(nDataLen, nNewSize));
        }
        free_mem();

        m_bitstream.Data       = pData;
        m_bitstream.DataOffset = 0;
        m_bitstream.DataLength = (uint32_t)nDataLen;
        m_bitstream.MaxLength  = (uint32_t)nNewSize;

        return RGY_ERR_NONE;
    }

    RGY_ERR append(const uint8_t *appendData, size_t appendSize) {
        if (appendData && appendSize > 0) {
            const auto new_data_length = appendSize + m_bitstream.DataLength;
            if (m_bitstream.MaxLength < new_data_length) {
                auto sts = changeSize(new_data_length);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }

            if (m_bitstream.MaxLength < new_data_length + m_bitstream.DataOffset) {
                memmove(m_bitstream.Data, m_bitstream.Data + m_bitstream.DataOffset, m_bitstream.DataLength);
                m_bitstream.DataOffset = 0;
            }
            memcpy(m_bitstream.Data + m_bitstream.DataLength + m_bitstream.DataOffset, appendData, appendSize);
            m_bitstream.DataLength = (uint32_t)new_data_length;
        }
        return RGY_ERR_NONE;
    }

    RGY_ERR append(const RGYBitstream *pBitstream) {
        if (pBitstream != nullptr && pBitstream->size() > 0) {
            return append(pBitstream->data(), pBitstream->size());
        }
        return RGY_ERR_NONE;
    }
    void addFrameData(RGYFrameData *frameData);
    void clearFrameDataList();
    std::vector<RGYFrameData *> getFrameDataList();
};

static inline RGYBitstream RGYBitstreamInit() {
    RGYBitstream bitstream;
    memset(&bitstream, 0, sizeof(bitstream));
    return bitstream;
}

static_assert(std::is_trivially_copyable<RGYBitstream>::value == true, "RGYBitstream should be trivially copyable.");

static inline RGYFrameInfo frameinfo_enc_to_rgy(const mfxFrameSurface1& mfx) {
    RGYFrameInfo info;
    info.width = mfx.Info.CropW;
    info.height = mfx.Info.CropH;
    info.csp = csp_enc_to_rgy(mfx.Info.FourCC);
    info.bitdepth = (mfx.Info.BitDepthLuma == 0) ? 8 : mfx.Info.BitDepthLuma;
    info.picstruct = picstruct_enc_to_rgy(mfx.Info.PicStruct);
    info.mem_type = RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED;
    info.timestamp = mfx.Data.TimeStamp;
    info.duration = 0;
    info.inputFrameId = mfx.Data.FrameOrder;
    info.flags = (RGY_FRAME_FLAGS)mfx.Data.DataFlag;
    memset(info.ptr, 0, sizeof(info.ptr));
    if (mfx.Info.FourCC == MFX_FOURCC_Y410) {
        info.ptr[0] = (uint8_t *)mfx.Data.Y410;
    } else if (mfx.Info.FourCC == MFX_MAKEFOURCC('R','G','B','3') // MFX_FOURCC_RGB3
            || mfx.Info.FourCC == MFX_FOURCC_RGB4) {
        info.ptr[0] = (uint8_t *)(std::min)((std::min)(mfx.Data.R, mfx.Data.G), mfx.Data.B);
    } else {
        info.ptr[0] = (uint8_t *)mfx.Data.Y;
        info.ptr[1] = (uint8_t *)mfx.Data.UV;
        info.ptr[2] = (uint8_t *)mfx.Data.V;
    }
    memset(info.pitch, 0, sizeof(info.pitch));
    for (int i = 0; i < RGY_CSP_PLANES[info.csp]; i++) {
        info.pitch[i] = mfx.Data.Pitch;
    }
    return info;
}

#if !FOR_AUO
class RGYFrameMFXSurf : public RGYFrame {
protected:
    mfxFrameSurface1 m_surface;
    uint64_t m_duration;
    int m_inputFrameId;
    std::vector<std::shared_ptr<RGYFrameData>> m_dataList;
public:
    RGYFrameMFXSurf(mfxFrameSurface1& s) : m_surface(s), m_duration(0), m_inputFrameId(-1), m_dataList() { };
    virtual mfxFrameSurface1 *surf() { return &m_surface; };
    virtual const mfxFrameSurface1 *surf() const { return &m_surface; };
    virtual bool isempty() const { return false; };
    virtual sInputCrop crop() const {
        sInputCrop cr;
        cr.e.left = m_surface.Info.CropX;
        cr.e.up = m_surface.Info.CropY;
        cr.e.right = m_surface.Info.Width - m_surface.Info.CropW - m_surface.Info.CropX;
        cr.e.bottom = m_surface.Info.Height - m_surface.Info.CropH - m_surface.Info.CropY;
        return cr;
    }
    virtual void setTimestamp(uint64_t timestamp) override { m_surface.Data.TimeStamp = timestamp; }
    virtual void setDuration(uint64_t frame_duration) override { m_duration = frame_duration; }
    virtual void setPicstruct(RGY_PICSTRUCT picstruct) override { m_surface.Info.PicStruct = picstruct_rgy_to_enc(picstruct); }
    virtual void setInputFrameId(int inputFrameId) override { m_inputFrameId = inputFrameId; }
    virtual void setFlags(RGY_FRAME_FLAGS flag) override { m_surface.Data.DataFlag = (decltype(m_surface.Data.DataFlag))flag; };
    virtual void clearDataList() override { m_dataList.clear(); };
    virtual const std::vector<std::shared_ptr<RGYFrameData>>& dataList() const override { return m_dataList; };
    virtual std::vector<std::shared_ptr<RGYFrameData>>& dataList() override { return m_dataList; };
    virtual void setDataList(const std::vector<std::shared_ptr<RGYFrameData>>& dataList) override { m_dataList = dataList; };
    RGYFrameInfo getInfoCopy() const { return getInfo(); }
    uint32_t locked() const { return m_surface.Data.Locked; }
protected:
    virtual RGYFrameInfo getInfo() const override {
        RGYFrameInfo info = frameinfo_enc_to_rgy(m_surface);
        info.duration = m_duration;
        info.inputFrameId = m_inputFrameId;
        info.dataList = m_dataList;
        return info;
    }
};
#endif

const TCHAR *get_low_power_str(uint32_t LowPower);
const TCHAR *get_err_mes(int sts);
static void print_err_mes(int sts) {
    _ftprintf(stderr, _T("%s"), get_err_mes(sts));
}

const TCHAR *ChromaFormatToStr(uint32_t format);
const TCHAR *ColorFormatToStr(uint32_t format);
const TCHAR *TargetUsageToStr(uint16_t tu);
const TCHAR *EncmodeToStr(uint32_t enc_mode);
const TCHAR *MemTypeToStr(uint32_t memType);
tstring MFXPicStructToStr(uint32_t picstruct);
tstring MFXImplToStr(uint32_t impl);
tstring MFXAccelerationModeToStr(mfxAccelerationMode impl);
tstring MFXImplTypeToStr(mfxImplType impl);

mfxStatus mfxBitstreamInit(mfxBitstream *pBitstream, uint32_t nSize);
mfxStatus mfxBitstreamCopy(mfxBitstream *pBitstreamCopy, const mfxBitstream *pBitstream);
mfxStatus mfxBitstreamExtend(mfxBitstream *pBitstream, uint32_t nSize);
mfxStatus mfxBitstreamAppend(mfxBitstream *pBitstream, const uint8_t *data, uint32_t size);
void mfxBitstreamClear(mfxBitstream *pBitstream);

#define QSV_IGNORE_STS(sts, err)                { if ((err) == (sts)) {(sts) = MFX_ERR_NONE; } }
#define RGY_IGNORE_STS(sts, err)                { if ((err) == (sts)) {(sts) = RGY_ERR_NONE; } }

mfxExtBuffer *GetExtBuffer(mfxExtBuffer **ppExtBuf, int nCount, uint32_t targetBufferId);

const TCHAR *get_vpp_image_stab_mode_str(int mode);

int getCPUInfoQSV(TCHAR *buffer, size_t nSize);
int getCPUInfoQSV(TCHAR *buffer, size_t nSize, mfxSession session);

int64_t rational_rescale(int64_t v, rgy_rational<int> from, rgy_rational<int> to);

#endif //_QSV_UTIL_H_
