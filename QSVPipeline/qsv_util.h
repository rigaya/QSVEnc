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
#if defined(_WIN32) || defined(_WIN64)
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#endif
#include <vector>
#include <array>
#include <utility>
#include <string>
#include <chrono>
#include <memory>
#include <type_traits>
#include "rgy_osdep.h"
#include "mfxstructures.h"
#include "mfxcommon.h"
#include "mfxsession.h"
#include "rgy_version.h"
#include "cpu_info.h"
#include "gpu_info.h"
#include "rgy_util.h"
#include "convert_csp.h"
#include "qsv_prm.h"
#include "rgy_err.h"

using std::vector;
using std::unique_ptr;
using std::shared_ptr;

#define INIT_MFX_EXT_BUFFER(x, id) { RGY_MEMSET_ZERO(x); (x).Header.BufferId = (id); (x).Header.BufferSz = sizeof(x); }

MAP_PAIR_0_1_PROTO(codec, rgy, RGY_CODEC, enc, mfxU32);
MAP_PAIR_0_1_PROTO(chromafmt, rgy, RGY_CHROMAFMT, enc, mfxU32);
MAP_PAIR_0_1_PROTO(csp, rgy, RGY_CSP, enc, mfxU32);

mfxU16 picstruct_rgy_to_enc(RGY_PICSTRUCT picstruct);
RGY_PICSTRUCT picstruct_enc_to_rgy(mfxU16 picstruct);
mfxFrameInfo frameinfo_rgy_to_enc(VideoInfo info);
VideoInfo videooutputinfo(const mfxInfoMFX& mfx, const mfxExtVideoSignalInfo& vui);

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

static const int RGY_CSP_TO_MFX_FOURCC[] = {
    0, //RGY_CSP_NA
    MFX_FOURCC_NV12, //RGY_CSP_NV12
    MFX_FOURCC_YV12, //RGY_CSP_YV12
    MFX_FOURCC_YUY2, //RGY_CSP_YUY2 
    0, //RGY_CSP_YUV422
    0, //RGY_CSP_YUV444
    MFX_FOURCC_P010, //RGY_CSP_YV12_09
    MFX_FOURCC_P010,
    MFX_FOURCC_P010,
    MFX_FOURCC_P010,
    MFX_FOURCC_P010, //RGY_CSP_YV12_16
    MFX_FOURCC_P010, //RGY_CSP_P010
    MFX_FOURCC_P210, //RGY_CSP_P210
    0, //RGY_CSP_YUV444_09
    0,
    0,
    0,
    0, //RGY_CSP_YUV444_16
    MFX_FOURCC_RGB3,
    MFX_FOURCC_RGB4,
    0 //RGY_CSP_YC48
};

mfxFrameInfo toMFXFrameInfo(VideoInfo info);

tstring qsv_memtype_str(uint16_t memtype);

static inline uint16_t check_coding_option(uint16_t value) {
    if (value == MFX_CODINGOPTION_UNKNOWN
        || value == MFX_CODINGOPTION_ON
        || value == MFX_CODINGOPTION_OFF
        || value == MFX_CODINGOPTION_ADAPTIVE) {
        return value;
    }
    return MFX_CODINGOPTION_UNKNOWN;
}

VideoInfo videooutputinfo(const mfxInfoMFX& mfx, const mfxExtVideoSignalInfo& vui);


struct RGYBitstream {
private:
    mfxBitstream m_bitstream;

public:
    mfxBitstream& bitstream() {
        return m_bitstream;
    }

    uint8_t *bufptr() const {
        return m_bitstream.Data;
    }

    const uint8_t *data() const {
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

    void setDuration(int duration) {
        UNREFERENCED_PARAMETER(duration);
    }

    int frameIdx() {
        return 0;
    }

    void setFrameIdx(int frameIdx) {
        UNREFERENCED_PARAMETER(frameIdx);
    }

    uint32_t size() const {
        return m_bitstream.DataLength;
    }

    void setSize(uint32_t size) {
        m_bitstream.DataLength = size;
    }

    uint32_t offset() const {
        return m_bitstream.DataOffset;
    }

    void addOffset(uint32_t add) {
        m_bitstream.DataOffset += add;
    }

    void setOffset(uint32_t offset) {
        m_bitstream.DataOffset = offset;
    }

    uint32_t bufsize() const {
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

    void clear() {
        if (m_bitstream.Data) {
            _aligned_free(m_bitstream.Data);
        }
        memset(&m_bitstream, 0, sizeof(m_bitstream));
    }

    RGY_ERR init(uint32_t nSize) {
        clear();

        if (nSize > 0) {
            if (nullptr == (m_bitstream.Data = (uint8_t *)_aligned_malloc(nSize, 32))) {
                return RGY_ERR_NULL_PTR;
            }

            m_bitstream.MaxLength = nSize;
        }
        return RGY_ERR_NONE;
    }

    RGY_ERR copy(const uint8_t *setData, uint32_t setSize) {
        if (setData == nullptr || setSize == 0) {
            return RGY_ERR_MORE_BITSTREAM;
        }
        if (m_bitstream.MaxLength < setSize) {
            clear();
            auto sts = init(setSize);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        m_bitstream.DataLength = setSize;
        m_bitstream.DataOffset = 0;
        memcpy(m_bitstream.Data, setData, setSize);
        return RGY_ERR_NONE;
    }

    RGY_ERR copy(const uint8_t *setData, uint32_t setSize, int64_t dts, int64_t pts) {
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

    RGY_ERR changeSize(uint32_t nNewSize) {
        uint8_t *pData = (uint8_t *)_aligned_malloc(nNewSize, 32);
        if (pData == nullptr) {
            return RGY_ERR_NULL_PTR;
        }

        auto nDataLen = m_bitstream.DataLength;
        if (m_bitstream.DataLength) {
            memcpy(pData, m_bitstream.Data + m_bitstream.DataOffset, (std::min)(nDataLen, nNewSize));
        }
        clear();

        m_bitstream.Data       = pData;
        m_bitstream.DataOffset = 0;
        m_bitstream.DataLength = nDataLen;
        m_bitstream.MaxLength  = nNewSize;

        return RGY_ERR_NONE;
    }

    RGY_ERR append(const uint8_t *appendData, uint32_t appendSize) {
        if (appendData) {
            const uint32_t new_data_length = appendSize + m_bitstream.DataLength;
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
            m_bitstream.DataLength = new_data_length;
        }
        return RGY_ERR_NONE;
    }

    RGY_ERR append(RGYBitstream *pBitstream) {
        return append(pBitstream->data(), pBitstream->size());
    }
};

static inline RGYBitstream RGYBitstreamInit() {
    RGYBitstream bitstream;
    memset(&bitstream, 0, sizeof(bitstream));
    return bitstream;
}

static_assert(sizeof(mfxBitstream) == sizeof(RGYBitstream), "RGYFrame size should equal to mfxFrameSurface1 size.");
static_assert(std::is_pod<RGYBitstream>::value == true, "RGYBitstream should be POD type.");

struct RGYFrame {
private:
    mfxFrameSurface1 m_surface;
public:
    mfxFrameSurface1& frame() {
        return m_surface;
    }
    void ptrArray(void *array[3]) {
        array[0] = m_surface.Data.Y;
        array[1] = m_surface.Data.UV;
        array[2] = m_surface.Data.V;
    }
    uint8_t *ptrY() {
        return m_surface.Data.Y;
    }
    uint8_t *ptrUV() {
        return m_surface.Data.UV;
    }
    uint8_t *ptrU() {
        return m_surface.Data.U;
    }
    uint8_t *ptrV() {
        return m_surface.Data.V;
    }
    uint8_t *ptrRGB() {
        return (std::min)((std::min)(m_surface.Data.R, m_surface.Data.G), m_surface.Data.B);
    }
    uint32_t pitch() {
        return m_surface.Data.Pitch;
    }
    uint32_t width() {
        return m_surface.Info.CropW;
    }
    uint32_t height() {
        return m_surface.Info.CropH;
    }
    sInputCrop crop() {
        sInputCrop cr;
        cr.e.left = m_surface.Info.CropX;
        cr.e.up = m_surface.Info.CropY;
        cr.e.right = m_surface.Info.Width - m_surface.Info.CropW - m_surface.Info.CropX;
        cr.e.bottom = m_surface.Info.Height - m_surface.Info.CropH - m_surface.Info.CropY;
        return cr;
    }
    RGY_CSP csp() {
        return csp_enc_to_rgy(m_surface.Info.FourCC);
    }
    int locked() {
        return m_surface.Data.Locked;
    }
    void lockIncrement() {
        m_surface.Data.Locked++;
    }
    void lockDecrement() {
        m_surface.Data.Locked--;
    }
    void setLocked(int locked) {
        m_surface.Data.Locked = (uint16_t)locked;
    }
    uint64_t timestamp() {
        return m_surface.Data.TimeStamp;
    }
    void setTimestamp(uint64_t timestamp) {
        m_surface.Data.TimeStamp = timestamp;
    }
};

static inline RGYFrame RGYFrameInit() {
    RGYFrame frame;
    memset(&frame, 0, sizeof(frame));
    return frame;
}

static_assert(sizeof(RGYFrame) == sizeof(mfxFrameSurface1), "RGYFrame size should equal to mfxFrameSurface1 size.");
static_assert(std::is_pod<RGYFrame>::value == true, "RGYFrame should be POD type.");

const TCHAR *get_low_power_str(mfxU16 LowPower);
const TCHAR *get_err_mes(int sts);
static void print_err_mes(int sts) {
    _ftprintf(stderr, _T("%s"), get_err_mes(sts));
}

const TCHAR *ColorFormatToStr(uint32_t format);
const TCHAR *CodecIdToStr(uint32_t nFourCC);
const TCHAR *TargetUsageToStr(uint16_t tu);
const TCHAR *EncmodeToStr(uint32_t enc_mode);
const TCHAR *MemTypeToStr(uint32_t memType);

mfxStatus mfxBitstreamInit(mfxBitstream *pBitstream, uint32_t nSize);
mfxStatus mfxBitstreamCopy(mfxBitstream *pBitstreamCopy, const mfxBitstream *pBitstream);
mfxStatus mfxBitstreamExtend(mfxBitstream *pBitstream, uint32_t nSize);
mfxStatus mfxBitstreamAppend(mfxBitstream *pBitstream, const uint8_t *data, uint32_t size);
void mfxBitstreamClear(mfxBitstream *pBitstream);

#define QSV_IGNORE_STS(sts, err)                { if ((err) == (sts)) {(sts) = MFX_ERR_NONE; } }

mfxExtBuffer *GetExtBuffer(mfxExtBuffer **ppExtBuf, int nCount, uint32_t targetBufferId);

const TCHAR *get_vpp_image_stab_mode_str(int mode);

#if defined(_WIN32) || defined(_WIN64)
bool check_if_d3d11_necessary();
#endif

#endif //_QSV_UTIL_H_
