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

#include "qsv_util.h"
#include "rgy_err.h"
#include "rgy_osdep.h"
#include <mfxjpeg.h>

static const auto RGY_CODEC_TO_MFX = make_array<std::pair<RGY_CODEC, mfxU32>>(
    std::make_pair(RGY_CODEC_H264,  MFX_CODEC_AVC),
    std::make_pair(RGY_CODEC_HEVC,  MFX_CODEC_HEVC),
    std::make_pair(RGY_CODEC_MPEG2, MFX_CODEC_MPEG2),
    std::make_pair(RGY_CODEC_VP8,   MFX_CODEC_VP8),
    std::make_pair(RGY_CODEC_VP9,   MFX_CODEC_VP9),
    std::make_pair(RGY_CODEC_VC1,   MFX_CODEC_VC1)
);

MAP_PAIR_0_1(codec, rgy, RGY_CODEC, enc, mfxU32, RGY_CODEC_TO_MFX, RGY_CODEC_UNKNOWN, 0u);

static const auto RGY_CHROMAFMT_TO_MFX = make_array<std::pair<RGY_CHROMAFMT, mfxU32>>(
    std::make_pair(RGY_CHROMAFMT_MONOCHROME, MFX_CHROMAFORMAT_MONOCHROME),
    std::make_pair(RGY_CHROMAFMT_YUV420,     MFX_CHROMAFORMAT_YUV420),
    std::make_pair(RGY_CHROMAFMT_YUV422,     MFX_CHROMAFORMAT_YUV422),
    std::make_pair(RGY_CHROMAFMT_YUV444,     MFX_CHROMAFORMAT_YUV444)
    );

MAP_PAIR_0_1(chromafmt, rgy, RGY_CHROMAFMT, enc, mfxU32, RGY_CHROMAFMT_TO_MFX, RGY_CHROMAFMT_UNKNOWN, 0u);

static const auto RGY_CSP_TO_MFX = make_array<std::pair<RGY_CSP, mfxU32>>(
    std::make_pair(RGY_CSP_NA,        0),
    std::make_pair(RGY_CSP_NV12,      MFX_FOURCC_NV12),
    std::make_pair(RGY_CSP_YV12,      MFX_FOURCC_YV12),
    std::make_pair(RGY_CSP_YUY2,      MFX_FOURCC_YUY2),
    std::make_pair(RGY_CSP_YUV422,    0),
    std::make_pair(RGY_CSP_YUV444,    0),
    std::make_pair(RGY_CSP_YV12_09,   MFX_FOURCC_P010),
    std::make_pair(RGY_CSP_YV12_10,   MFX_FOURCC_P010),
    std::make_pair(RGY_CSP_YV12_12,   MFX_FOURCC_P010),
    std::make_pair(RGY_CSP_YV12_14,   MFX_FOURCC_P010),
    std::make_pair(RGY_CSP_YV12_16,   MFX_FOURCC_P010),
    std::make_pair(RGY_CSP_P010,      MFX_FOURCC_P010),
    std::make_pair(RGY_CSP_P210,      MFX_FOURCC_P210),
    std::make_pair(RGY_CSP_YUV444_09, 0),
    std::make_pair(RGY_CSP_YUV444_10, 0),
    std::make_pair(RGY_CSP_YUV444_12, 0),
    std::make_pair(RGY_CSP_YUV444_14, 0),
    std::make_pair(RGY_CSP_YUV444_16, 0),
    std::make_pair(RGY_CSP_RGB24,     MFX_FOURCC_RGB3),
    std::make_pair(RGY_CSP_RGB32,     MFX_FOURCC_RGB4),
    std::make_pair(RGY_CSP_YC48,      0)
    );

MAP_PAIR_0_1(csp, rgy, RGY_CSP, enc, mfxU32, RGY_CSP_TO_MFX, RGY_CSP_NA, 0);

RGY_NOINLINE
mfxU16 picstruct_rgy_to_enc(RGY_PICSTRUCT picstruct) {
    if (picstruct & RGY_PICSTRUCT_TFF) return (mfxU16)MFX_PICSTRUCT_FIELD_TFF;
    if (picstruct & RGY_PICSTRUCT_BFF) return (mfxU16)MFX_PICSTRUCT_FIELD_BFF;
    return (mfxU16)MFX_PICSTRUCT_PROGRESSIVE;
}

RGY_NOINLINE
RGY_PICSTRUCT picstruct_enc_to_rgy(mfxU16 picstruct) {
    if (picstruct & MFX_PICSTRUCT_FIELD_TFF) return RGY_PICSTRUCT_FRAME_TFF;
    if (picstruct & MFX_PICSTRUCT_FIELD_BFF) return RGY_PICSTRUCT_FRAME_BFF;
    return RGY_PICSTRUCT_FRAME;
}

RGY_NOINLINE
mfxFrameInfo frameinfo_rgy_to_enc(VideoInfo info) {
    mfxFrameInfo mfx = { 0 };
    mfx.FourCC = RGY_CSP_TO_MFX_FOURCC[info.csp];
    mfx.ChromaFormat = (mfxU16)chromafmt_rgy_to_enc(RGY_CSP_CHROMA_FORMAT[info.csp]);
    mfx.BitDepthLuma = (mfxU16)(RGY_CSP_BIT_DEPTH[info.csp] - info.shift);
    mfx.BitDepthChroma = (mfxU16)(RGY_CSP_BIT_DEPTH[info.csp] - info.shift);
    mfx.Shift = info.shift ? 1 : 0;
    mfx.Width = (mfxU16)info.srcWidth;
    mfx.Height = (mfxU16)info.srcHeight;
    mfx.CropX = (mfxU16)info.crop.e.left;
    mfx.CropY = (mfxU16)info.crop.e.up;
    mfx.CropW = (mfxU16)(mfx.Width - info.crop.e.left - info.crop.e.right);
    mfx.CropH = (mfxU16)(mfx.Height - info.crop.e.up - info.crop.e.bottom);
    mfx.FrameRateExtN = info.fpsN;
    mfx.FrameRateExtD = info.fpsD;
    mfx.AspectRatioW = (mfxU16)info.sar[0];
    mfx.AspectRatioH = (mfxU16)info.sar[1];
    mfx.PicStruct = picstruct_rgy_to_enc(info.picstruct);
    return mfx;
}

RGY_NOINLINE
VideoInfo videooutputinfo(const mfxInfoMFX& mfx, const mfxExtVideoSignalInfo& vui) {
    VideoInfo info;
    info.codec = codec_enc_to_rgy(mfx.CodecId);
    info.codecLevel = mfx.CodecLevel;
    info.codecProfile = mfx.CodecProfile;
    info.videoDelay = ((mfx.GopRefDist - 1) > 0) + (((mfx.GopRefDist - 1) > 0) & ((mfx.GopRefDist - 1) > 2));
    info.dstWidth = mfx.FrameInfo.CropW;
    info.dstHeight = mfx.FrameInfo.CropH;
    info.fpsN = mfx.FrameInfo.FrameRateExtN;
    info.fpsD = mfx.FrameInfo.FrameRateExtD;
    info.sar[0] = mfx.FrameInfo.AspectRatioW;
    info.sar[1] = mfx.FrameInfo.AspectRatioH;
    info.vui.descriptpresent = vui.ColourDescriptionPresent;
    info.vui.colorprim = vui.ColourPrimaries;
    info.vui.matrix = vui.MatrixCoefficients;
    info.vui.transfer = vui.TransferCharacteristics;
    info.vui.fullrange = vui.VideoFullRange;
    info.vui.format = vui.VideoFormat;
    info.picstruct = picstruct_enc_to_rgy(mfx.FrameInfo.PicStruct);
    info.shift = (mfx.FrameInfo.Shift) ? 16 - mfx.FrameInfo.BitDepthLuma : 0;
    info.csp = csp_enc_to_rgy(mfx.FrameInfo.FourCC);
    return info;
}

RGY_NOINLINE
const TCHAR *get_err_mes(int sts) {
    switch (sts) {
    case MFX_ERR_NONE:                     return _T("no error.");
    case MFX_ERR_UNKNOWN:                  return _T("unknown error.");
    case MFX_ERR_NULL_PTR:                 return _T("null pointer.");
    case MFX_ERR_UNSUPPORTED:              return _T("undeveloped feature.");
    case MFX_ERR_MEMORY_ALLOC:             return _T("failed to allocate memory.");
    case MFX_ERR_NOT_ENOUGH_BUFFER:        return _T("insufficient buffer at input/output.");
    case MFX_ERR_INVALID_HANDLE:           return _T("invalid handle.");
    case MFX_ERR_LOCK_MEMORY:              return _T("failed to lock the memory block.");
    case MFX_ERR_NOT_INITIALIZED:          return _T("member function called before initialization.");
    case MFX_ERR_NOT_FOUND:                return _T("the specified object is not found.");
    case MFX_ERR_MORE_DATA:                return _T("expect more data at input.");
    case MFX_ERR_MORE_SURFACE:             return _T("expect more surface at output.");
    case MFX_ERR_ABORTED:                  return _T("operation aborted.");
    case MFX_ERR_DEVICE_LOST:              return _T("lose the HW acceleration device.");
    case MFX_ERR_INCOMPATIBLE_VIDEO_PARAM: return _T("incompatible video parameters.");
    case MFX_ERR_INVALID_VIDEO_PARAM:      return _T("invalid video parameters.");
    case MFX_ERR_UNDEFINED_BEHAVIOR:       return _T("undefined behavior.");
    case MFX_ERR_DEVICE_FAILED:            return _T("device operation failure.");
    case MFX_ERR_GPU_HANG:                 return _T("gpu hang.");
    case MFX_ERR_REALLOC_SURFACE:          return _T("failed to realloc surface.");

    case MFX_WRN_IN_EXECUTION:             return _T("the previous asynchrous operation is in execution.");
    case MFX_WRN_DEVICE_BUSY:              return _T("the HW acceleration device is busy.");
    case MFX_WRN_VIDEO_PARAM_CHANGED:      return _T("the video parameters are changed during decoding.");
    case MFX_WRN_PARTIAL_ACCELERATION:     return _T("SW is used.");
    case MFX_WRN_INCOMPATIBLE_VIDEO_PARAM: return _T("incompatible video parameters.");
    case MFX_WRN_VALUE_NOT_CHANGED:        return _T("the value is saturated based on its valid range.");
    case MFX_WRN_OUT_OF_RANGE:             return _T("the value is out of valid range.");
    default:                               return _T("unknown error.");
    }
}

RGY_NOINLINE
const TCHAR *get_low_power_str(mfxU16 LowPower) {
    switch (LowPower) {
    case MFX_CODINGOPTION_OFF: return _T(" PG");
    case MFX_CODINGOPTION_ON:  return _T(" FF");
    default: return _T("");
    }
}

RGY_NOINLINE
tstring qsv_memtype_str(uint16_t memtype) {
    tstring str;
    if (memtype & MFX_MEMTYPE_INTERNAL_FRAME)         str += _T("internal,");
    if (memtype & MFX_MEMTYPE_EXTERNAL_FRAME)         str += _T("external,");
    if (memtype & MFX_MEMTYPE_OPAQUE_FRAME)           str += _T("opaque,");
    if (memtype & MFX_MEMTYPE_DXVA2_DECODER_TARGET)   str += _T("dxvadec,");
    if (memtype & MFX_MEMTYPE_DXVA2_PROCESSOR_TARGET) str += _T("dxvaproc,");
    if (memtype & MFX_MEMTYPE_SYSTEM_MEMORY)          str += _T("system,");
    if (memtype & MFX_MEMTYPE_FROM_ENCODE)            str += _T("enc,");
    if (memtype & MFX_MEMTYPE_FROM_DECODE)            str += _T("dec,");
    if (memtype & MFX_MEMTYPE_FROM_VPPIN)             str += _T("vppin,");
    if (memtype & MFX_MEMTYPE_FROM_VPPOUT)            str += _T("vppout,");
    if (memtype == 0)                                 str += _T("none,");
    return str.substr(0, str.length()-1);
}


mfxStatus mfxBitstreamInit(mfxBitstream *pBitstream, uint32_t nSize) {
    mfxBitstreamClear(pBitstream);

    if (nullptr == (pBitstream->Data = (uint8_t *)_aligned_malloc(nSize, 32))) {
        return MFX_ERR_NULL_PTR;
    }

    pBitstream->MaxLength = nSize;
    return MFX_ERR_NONE;
}

mfxStatus mfxBitstreamCopy(mfxBitstream *pBitstreamCopy, const mfxBitstream *pBitstream) {
    memcpy(pBitstreamCopy, pBitstream, sizeof(pBitstreamCopy[0]));
    pBitstreamCopy->Data = nullptr;
    pBitstreamCopy->DataLength = 0;
    pBitstreamCopy->DataOffset = 0;
    pBitstreamCopy->MaxLength = 0;
    auto sts = mfxBitstreamInit(pBitstreamCopy, pBitstream->MaxLength);
    if (sts == MFX_ERR_NONE) {
        memcpy(pBitstreamCopy->Data, pBitstream->Data, pBitstreamCopy->DataLength);
    }
    return sts;
}

mfxStatus mfxBitstreamExtend(mfxBitstream *pBitstream, uint32_t nSize) {
    uint8_t *pData = (uint8_t *)_aligned_malloc(nSize, 32);
    if (nullptr == pData) {
        return MFX_ERR_NULL_PTR;
    }

    auto nDataLen = pBitstream->DataLength;
    if (nDataLen) {
        memmove(pData, pBitstream->Data + pBitstream->DataOffset, nDataLen);
    }

    mfxBitstreamClear(pBitstream);

    pBitstream->Data       = pData;
    pBitstream->DataOffset = 0;
    pBitstream->DataLength = nDataLen;
    pBitstream->MaxLength  = nSize;

    return MFX_ERR_NONE;
}

void mfxBitstreamClear(mfxBitstream *pBitstream) {
    if (pBitstream->Data) {
        _aligned_free(pBitstream->Data);
    }
    memset(pBitstream, 0, sizeof(pBitstream[0]));
}

mfxStatus mfxBitstreamAppend(mfxBitstream *pBitstream, const uint8_t *data, uint32_t size) {
    mfxStatus sts = MFX_ERR_NONE;
    if (data) {
        const uint32_t new_data_length = pBitstream->DataLength + size;
        if (pBitstream->MaxLength < new_data_length) {
            if (MFX_ERR_NONE != (sts = mfxBitstreamExtend(pBitstream, new_data_length))) {
                return sts;
            }
        }

        if (pBitstream->MaxLength < new_data_length + pBitstream->DataOffset) {
            memmove(pBitstream->Data, pBitstream->Data + pBitstream->DataOffset, pBitstream->DataLength);
            pBitstream->DataOffset = 0;
        }
        memcpy(pBitstream->Data + pBitstream->DataLength + pBitstream->DataOffset, data, size);
        pBitstream->DataLength = new_data_length;
    }
    return sts;
}

mfxExtBuffer *GetExtBuffer(mfxExtBuffer **ppExtBuf, int nCount, uint32_t targetBufferId) {
    if (ppExtBuf) {
        for (int i = 0; i < nCount; i++) {
            if (ppExtBuf[i] && ppExtBuf[i]->BufferId == targetBufferId) {
                return ppExtBuf[i];
            }
        }
    }
    return nullptr;
}

RGY_NOINLINE
const TCHAR *ColorFormatToStr(uint32_t format) {
    switch (format) {
    case MFX_FOURCC_NV12:
        return _T("nv12");
    case MFX_FOURCC_NV16:
        return _T("nv16");
    case MFX_FOURCC_YV12:
        return _T("yv12");
    case MFX_FOURCC_YUY2:
        return _T("yuy2");
    case MFX_FOURCC_RGB3:
        return _T("rgb24");
    case MFX_FOURCC_RGB4:
        return _T("rgb32");
    case MFX_FOURCC_BGR4:
        return _T("bgr32");
    case MFX_FOURCC_P010:
        return _T("nv12(10bit)");
    case MFX_FOURCC_P210:
        return _T("nv16(10bit)");
    default:
        return _T("unsupported");
    }
}

RGY_NOINLINE
const TCHAR *CodecIdToStr(uint32_t nFourCC) {
    switch (nFourCC) {
    case MFX_CODEC_AVC:
        return _T("H.264/AVC");
    case MFX_CODEC_VC1:
        return _T("VC-1");
    case MFX_CODEC_HEVC:
        return _T("HEVC");
    case MFX_CODEC_MPEG2:
        return _T("MPEG2");
    case MFX_CODEC_VP8:
        return _T("VP8");
    case MFX_CODEC_VP9:
        return _T("VP9");
    case MFX_CODEC_JPEG:
        return _T("JPEG");
    default:
        return _T("NOT_SUPPORTED");
    }
}

RGY_NOINLINE
const TCHAR *TargetUsageToStr(uint16_t tu) {
    switch (tu) {
    case MFX_TARGETUSAGE_BEST_QUALITY: return _T("1 - best");
    case 2:                            return _T("2 - higher");
    case 3:                            return _T("3 - high");
    case MFX_TARGETUSAGE_BALANCED:     return _T("4 - balanced");
    case 5:                            return _T("5 - fast");
    case 6:                            return _T("6 - faster");
    case MFX_TARGETUSAGE_BEST_SPEED:   return _T("7 - fastest");
    case MFX_TARGETUSAGE_UNKNOWN:      return _T("unknown");
    default:                           return _T("unsupported");
    }
}

RGY_NOINLINE
const TCHAR *EncmodeToStr(uint32_t enc_mode) {
    switch (enc_mode) {
    case MFX_RATECONTROL_CBR:
        return _T("Bitrate Mode - CBR");
    case MFX_RATECONTROL_VBR:
        return _T("Bitrate Mode - VBR");
    case MFX_RATECONTROL_AVBR:
        return _T("Bitrate Mode - AVBR");
    case MFX_RATECONTROL_CQP:
        return _T("Constant QP (CQP)");
    case MFX_RATECONTROL_LA:
        return _T("Bitrate Mode - Lookahead");
    case MFX_RATECONTROL_ICQ:
        return _T("ICQ (Intelligent Const. Quality)");
    case MFX_RATECONTROL_VCM:
        return _T("VCM (Video Conference Mode)");
    case MFX_RATECONTROL_LA_ICQ:
        return _T("LA-ICQ (Intelligent Const. Quality with Lookahead)");
    case MFX_RATECONTROL_LA_EXT:
        return _T("LA-EXT (Extended Lookahead)");
    case MFX_RATECONTROL_LA_HRD:
        return _T("LA-HRD (HRD compliant Lookahead)");
    case MFX_RATECONTROL_QVBR:
        return _T("Quality VBR bitrate");
    case MFX_RATECONTROL_VQP:
        return _T("Variable QP (VQP)");
    default:
        return _T("unsupported");
    }
}

RGY_NOINLINE
const TCHAR *MemTypeToStr(uint32_t memType) {
    switch (memType) {
    case SYSTEM_MEMORY:
        return _T("system");
#if D3D_SURFACES_SUPPORT
    case D3D9_MEMORY:
        return _T("d3d9");
#if MFX_D3D11_SUPPORT
    case D3D11_MEMORY:
        return _T("d3d11");
    case HW_MEMORY:
        return _T("d3d11+d3d9");
#endif //#if MFX_D3D11_SUPPORT
#endif //#if D3D_SURFACES_SUPPORT
#ifdef LIBVA_SUPPORT
    case VA_MEMORY:
    case HW_MEMORY:
        return _T("va");
#endif
    default:
        return _T("unsupported");
    }
}

RGY_NOINLINE
const TCHAR *get_vpp_image_stab_mode_str(int mode) {
    switch (mode) {
    case MFX_IMAGESTAB_MODE_UPSCALE: return _T("upscale");
    case MFX_IMAGESTAB_MODE_BOXING:  return _T("boxing");
    default: return _T("unknown");
    }
}

#if defined(_WIN32) || defined(_WIN64)
bool check_if_d3d11_necessary();
#endif