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
#include "rgy_opencl.h"

using std::vector;
using std::unique_ptr;
using std::shared_ptr;

class RGYFrameData;

#define MFX_HANDLE_IDIRECT3D9EX ((mfxHandleType)-1)

#define INIT_MFX_EXT_BUFFER(x, id) { RGY_MEMSET_ZERO(x); (x).Header.BufferId = (id); (x).Header.BufferSz = sizeof(x); }

MAP_PAIR_0_1_PROTO(codec, rgy, RGY_CODEC, enc, mfxU32);
MAP_PAIR_0_1_PROTO(chromafmt, rgy, RGY_CHROMAFMT, enc, mfxU32);
MAP_PAIR_0_1_PROTO(csp, rgy, RGY_CSP, enc, mfxU32);
MAP_PAIR_0_1_PROTO(resize_algo, rgy, RGY_VPP_RESIZE_ALGO, enc, int);
MAP_PAIR_0_1_PROTO(resize_mode, rgy, RGY_VPP_RESIZE_MODE, enc, int);

mfxU16 picstruct_rgy_to_enc(RGY_PICSTRUCT picstruct);
RGY_PICSTRUCT picstruct_enc_to_rgy(mfxU16 picstruct);
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
    switch (pParams->CodecId) {
    case MFX_CODEC_HEVC:
    case MFX_CODEC_VP9:
    case MFX_CODEC_AV1:
        return pParams->outputDepth;
    case MFX_CODEC_AVC:
    case MFX_CODEC_VP8:
    case MFX_CODEC_MPEG2:
    case MFX_CODEC_VC1:
        break;
    default:
        return 0;
    }
    return 8;
}

static RGY_CSP getMFXCsp(const RGY_CHROMAFMT chroma, const int bitdepth) {
    if (bitdepth > 8) {
        switch (chroma) {
        case RGY_CHROMAFMT_YUV420: return RGY_CSP_P010;
        case RGY_CHROMAFMT_YUV422: return RGY_CSP_Y210;
        case RGY_CHROMAFMT_YUV444: return (bitdepth > 10) ? RGY_CSP_Y416 : RGY_CSP_Y410;
        default: return RGY_CSP_NA;
        }
    }
    switch (chroma) {
    case RGY_CHROMAFMT_YUV420: return RGY_CSP_NV12;
    case RGY_CHROMAFMT_YUV422: return RGY_CSP_YUY2;
    case RGY_CHROMAFMT_YUV444: return RGY_CSP_AYUV;
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

VideoInfo videooutputinfo(const mfxInfoMFX& mfx, const mfxExtVideoSignalInfo& vui);

bool isRCBitrateMode(int encmode);

struct RGYBitstream {
private:
    mfxBitstream m_bitstream;
    RGYFrameData **frameDataList;
    int frameDataNum;

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

    void setDuration(int duration) {
        UNREFERENCED_PARAMETER(duration);
    }

    int frameIdx() {
        return 0;
    }

    void setFrameIdx(int frameIdx) {
        UNREFERENCED_PARAMETER(frameIdx);
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

    void clear() {
        if (m_bitstream.Data) {
            _aligned_free(m_bitstream.Data);
        }
        memset(&m_bitstream, 0, sizeof(m_bitstream));
    }

    RGY_ERR init(size_t nSize) {
        clear();

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
            clear();
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

    RGY_ERR changeSize(size_t nNewSize) {
        uint8_t *pData = (uint8_t *)_aligned_malloc(nNewSize, 32);
        if (pData == nullptr) {
            return RGY_ERR_NULL_PTR;
        }

        size_t nDataLen = m_bitstream.DataLength;
        if (m_bitstream.DataLength) {
            memcpy(pData, m_bitstream.Data + m_bitstream.DataOffset, (std::min)(nDataLen, nNewSize));
        }
        clear();

        m_bitstream.Data       = pData;
        m_bitstream.DataOffset = 0;
        m_bitstream.DataLength = (uint32_t)nDataLen;
        m_bitstream.MaxLength  = (uint32_t)nNewSize;

        return RGY_ERR_NONE;
    }

    RGY_ERR append(const uint8_t *appendData, size_t appendSize) {
        if (appendData) {
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
        if (pBitstream != nullptr) {
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

static_assert(std::is_pod<RGYBitstream>::value == true, "RGYBitstream should be POD type.");

struct RGYCLFrame;

class RGYFrame {
protected:
    std::vector<std::shared_ptr<RGYFrameData>> frameDataList;
public:
    RGYFrame() : frameDataList() {};
    virtual ~RGYFrame() {};
    virtual mfxFrameSurface1 *surf() { return nullptr; };
    virtual const mfxFrameSurface1 *surf() const { return nullptr; };
    virtual RGYCLFrame *clframe() { return nullptr; };
    virtual const RGYCLFrame *clframe() const { return nullptr; };
    virtual void ptrArray(void *array[3], bool bRGB) = 0;
    virtual uint8_t *ptrY() = 0;
    virtual uint8_t *ptrUV() = 0;
    virtual uint8_t *ptrU() = 0;
    virtual uint8_t *ptrV() = 0;
    virtual uint8_t *ptrRGB() = 0;
    virtual uint32_t pitch() const = 0;
    virtual uint32_t width() const = 0;
    virtual uint32_t height() const = 0;
    virtual sInputCrop crop() const = 0;
    virtual RGY_CSP csp() const = 0;
    virtual int64_t timestamp() const = 0;
    virtual void setTimestamp(int64_t timestamp) = 0;
    virtual int64_t duration() const = 0;
    virtual void setDuration(int64_t frame_duration) = 0;
    virtual RGY_PICSTRUCT picstruct() const = 0;
    virtual void setPicstruct(RGY_PICSTRUCT picstruct) = 0;
    virtual int inputFrameId() const = 0;
    virtual void setInputFrameId(int inputFrameId) = 0;
    virtual uint64_t flags() const { return RGY_FRAME_FLAG_NONE; };
    virtual void setFlags(uint64_t flag) { UNREFERENCED_PARAMETER(flag); };
    virtual const std::vector<std::shared_ptr<RGYFrameData>>& dataList() const { return frameDataList; }
    virtual std::vector<std::shared_ptr<RGYFrameData>>& dataList() { return frameDataList; }
    virtual void setDataList(std::vector<std::shared_ptr<RGYFrameData>>& dataList) { frameDataList = dataList; }
    virtual void clearDataList() { frameDataList.clear(); }
};

class RGYFrameMFXSurf : public RGYFrame {
protected:
    mfxFrameSurface1 m_surface;
public:
    RGYFrameMFXSurf(mfxFrameSurface1& s) : m_surface(s) { };
    virtual mfxFrameSurface1 *surf() override { return &m_surface; };
    virtual const mfxFrameSurface1 *surf() const override { return &m_surface; };
    virtual int64_t timestamp() const override { return m_surface.Data.TimeStamp; }
    virtual void setTimestamp(int64_t timestamp) override { m_surface.Data.TimeStamp = timestamp; }
    virtual int inputFrameId() const override { return m_surface.Data.FrameOrder; }
    virtual void setInputFrameId(int inputFrameId) override { m_surface.Data.FrameOrder = inputFrameId; }
    virtual uint64_t flags() const  override { return m_surface.Data.DataFlag; }
    virtual void setFlags(uint64_t flag) override { m_surface.Data.DataFlag = (decltype(m_surface.Data.DataFlag))flag; };

    virtual void ptrArray(void *array[3], bool bRGB) override {
        array[0] = (m_surface.Info.FourCC == MFX_FOURCC_Y410) ? (void *)m_surface.Data.Y410 : ((bRGB) ? ptrRGB() : m_surface.Data.Y);
        array[1] = m_surface.Data.UV;
        array[2] = m_surface.Data.V;
    }
    virtual uint8_t *ptrY() override { return m_surface.Data.Y; }
    virtual uint8_t *ptrUV() override { return m_surface.Data.UV; }
    virtual uint8_t *ptrU() override { return m_surface.Data.U; }
    virtual uint8_t *ptrV() override { return m_surface.Data.V; }
    virtual uint8_t *ptrRGB() override { return (std::min)((std::min)(m_surface.Data.R, m_surface.Data.G), m_surface.Data.B); }
    virtual uint32_t pitch() const override { return m_surface.Data.Pitch; }
    virtual uint32_t width() const override { return m_surface.Info.CropW; }
    virtual uint32_t height() const override { return m_surface.Info.CropH; }
    virtual sInputCrop crop() const override {
        sInputCrop cr;
        cr.e.left = m_surface.Info.CropX;
        cr.e.up = m_surface.Info.CropY;
        cr.e.right = m_surface.Info.Width - m_surface.Info.CropW - m_surface.Info.CropX;
        cr.e.bottom = m_surface.Info.Height - m_surface.Info.CropH - m_surface.Info.CropY;
        return cr;
    }
    virtual RGY_CSP csp() const override { return csp_enc_to_rgy(m_surface.Info.FourCC); }
    virtual int64_t duration() const override { return m_surface.Data.FrameOrder; }
    virtual void setDuration(int64_t frame_duration) override { m_surface.Data.FrameOrder = (decltype(m_surface.Data.FrameOrder))frame_duration; }
    virtual RGY_PICSTRUCT picstruct() const override { return picstruct_enc_to_rgy(m_surface.Info.PicStruct); }
    virtual void setPicstruct(RGY_PICSTRUCT picstruct) override { m_surface.Info.PicStruct = picstruct_rgy_to_enc(picstruct); }
};

#if !FOR_AUO
class RGYFrameCL : public RGYFrame {
protected:
    std::unique_ptr<RGYCLFrame> openclframe;
public:
    RGYFrameCL(std::unique_ptr<RGYCLFrame>& f) : openclframe(std::move(f)) {};
    virtual RGYCLFrame *clframe() override { return openclframe.get(); };
    virtual const RGYCLFrame *clframe() const override { return openclframe.get(); };
    virtual int64_t timestamp() const override { return openclframe->frameInfo().timestamp; }
    virtual void setTimestamp(int64_t timestamp) override { openclframe->frameInfo().timestamp = timestamp; };
    virtual int inputFrameId() const override { return openclframe->frameInfo().inputFrameId; }
    virtual void setInputFrameId(int inputFrameId) override { openclframe->frameInfo().inputFrameId = inputFrameId; }
    virtual uint64_t flags() const override { return openclframe->frameInfo().flags; }
    virtual void setFlags(uint64_t flag) override { openclframe->frameInfo().flags = (RGY_FRAME_FLAGS)flag; };
    virtual void ptrArray(void *array[3], bool bRGB) override {
        UNREFERENCED_PARAMETER(bRGB);
        auto frame = openclframe->frameInfo();
        array[0] = frame.ptr[0];
        array[1] = frame.ptr[1];
        array[2] = frame.ptr[2];
    }
    virtual uint8_t *ptrY() override {
        auto frame = openclframe->frameInfo();
        auto plane = getPlane(&frame, RGY_PLANE_Y);
        return plane.ptr[0];
    }
    virtual uint8_t *ptrUV() override {
        auto frame = openclframe->frameInfo();
        auto plane = getPlane(&frame, RGY_PLANE_C);
        return plane.ptr[0];
    }
    virtual uint8_t *ptrU() override {
        auto frame = openclframe->frameInfo();
        auto plane = getPlane(&frame, RGY_PLANE_U);
        return plane.ptr[0];
    }
    virtual uint8_t *ptrV() override {
        auto frame = openclframe->frameInfo();
        auto plane = getPlane(&frame, RGY_PLANE_U);
        return plane.ptr[0];
    }
    virtual uint8_t *ptrRGB() override { return nullptr; }
    virtual uint32_t pitch() const override { return openclframe->frameInfo().pitch[0]; }
    virtual uint32_t width() const override { return openclframe->frameInfo().width; }
    virtual uint32_t height() const override { return openclframe->frameInfo().height; }
    virtual sInputCrop crop() const override { return sInputCrop(); }
    virtual RGY_CSP csp() const override { return openclframe->frameInfo().csp; }
    virtual int64_t duration() const override { return openclframe->frameInfo().duration; }
    virtual void setDuration(int64_t frame_duration) override { openclframe->frameInfo().duration = frame_duration; }
    virtual RGY_PICSTRUCT picstruct() const override { return openclframe->frameInfo().picstruct; }
    virtual void setPicstruct(RGY_PICSTRUCT picstruct) override { openclframe->frameInfo().picstruct = picstruct; }
};
#endif

class RGYFrameRef : public RGYFrame {
protected:
    RGYFrameInfo frame;
public:
    RGYFrameRef(const RGYFrameInfo& f) : frame(f) {};
    virtual int64_t timestamp() const override { return frame.timestamp; }
    virtual void setTimestamp(int64_t timestamp) override { frame.timestamp = timestamp; };
    virtual int inputFrameId() const override { return frame.inputFrameId; }
    virtual void setInputFrameId(int inputFrameId) override { frame.inputFrameId = inputFrameId; }
    virtual uint64_t flags() const override { return frame.flags; }
    virtual void setFlags(uint64_t flag) override { frame.flags = (RGY_FRAME_FLAGS)flag; };
    virtual void ptrArray(void *array[3], bool bRGB) override {
        UNREFERENCED_PARAMETER(bRGB);
        array[0] = frame.ptr[0];
        array[1] = frame.ptr[1];
        array[2] = frame.ptr[2];
    }
    virtual uint8_t *ptrY() override {
        auto plane = getPlane(&frame, RGY_PLANE_Y);
        return plane.ptr[0];
    }
    virtual uint8_t *ptrUV() override {
        auto plane = getPlane(&frame, RGY_PLANE_C);
        return plane.ptr[0];
    }
    virtual uint8_t *ptrU() override {
        auto plane = getPlane(&frame, RGY_PLANE_U);
        return plane.ptr[0];
    }
    virtual uint8_t *ptrV() override {
        auto plane = getPlane(&frame, RGY_PLANE_U);
        return plane.ptr[0];
    }
    virtual uint8_t *ptrRGB() override { return nullptr; }
    virtual uint32_t pitch() const override { return frame.pitch[0]; }
    virtual uint32_t width() const override { return frame.width; }
    virtual uint32_t height() const override { return frame.height; }
    virtual sInputCrop crop() const override { return sInputCrop(); }
    virtual RGY_CSP csp() const override { return frame.csp; }
    virtual int64_t duration() const override { return frame.duration; }
    virtual void setDuration(int64_t frame_duration) override { frame.duration = frame_duration; }
    virtual RGY_PICSTRUCT picstruct() const override { return frame.picstruct; }
    virtual void setPicstruct(RGY_PICSTRUCT picstruct) override { frame.picstruct = picstruct; }
};
const TCHAR *get_low_power_str(uint32_t LowPower);
const TCHAR *get_err_mes(int sts);
static void print_err_mes(int sts) {
    _ftprintf(stderr, _T("%s"), get_err_mes(sts));
}

const TCHAR *ChromaFormatToStr(uint32_t format);
const TCHAR *ColorFormatToStr(uint32_t format);
const TCHAR *CodecIdToStr(uint32_t nFourCC);
const TCHAR *TargetUsageToStr(uint16_t tu);
const TCHAR *EncmodeToStr(uint32_t enc_mode);
const TCHAR *MemTypeToStr(uint32_t memType);
tstring MFXPicStructToStr(uint32_t picstruct);
tstring MFXImplToStr(uint32_t impl);
tstring MFXAccelerationModeToStr(mfxAccelerationMode impl);

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
