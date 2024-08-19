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
#include "rgy_frame.h"

#pragma warning (push)
#pragma warning (disable: 4201) //C4201: 非標準の拡張機能が使用されています: 無名の構造体または共用体です。
#pragma warning (disable: 4996) //C4996: 'MFXInit': が古い形式として宣言されました。
#pragma warning (disable: 4819) //C4819: ファイルは、現在のコード ページ (932) で表示できない文字を含んでいます。データの損失を防ぐために、ファイルを Unicode 形式で保存してください。
#include <mfxjpeg.h>
#pragma warning (pop)

static const auto RGY_CODEC_TO_MFX = make_array<std::pair<RGY_CODEC, mfxU32>>(
    std::make_pair(RGY_CODEC_H264,  MFX_CODEC_AVC),
    std::make_pair(RGY_CODEC_HEVC,  MFX_CODEC_HEVC),
    std::make_pair(RGY_CODEC_MPEG2, MFX_CODEC_MPEG2),
    std::make_pair(RGY_CODEC_VP8,   MFX_CODEC_VP8),
    std::make_pair(RGY_CODEC_VP9,   MFX_CODEC_VP9),
    std::make_pair(RGY_CODEC_AV1,   MFX_CODEC_AV1),
    std::make_pair(RGY_CODEC_VC1,   MFX_CODEC_VC1),
    std::make_pair(RGY_CODEC_VVC,   MFX_CODEC_VVC),
    std::make_pair(RGY_CODEC_RAW,   MFX_CODEC_RAW)
);

MAP_PAIR_0_1(codec, rgy, RGY_CODEC, enc, mfxU32, RGY_CODEC_TO_MFX, RGY_CODEC_UNKNOWN, 0u);

static const auto RGY_CHROMAFMT_TO_MFX = make_array<std::pair<RGY_CHROMAFMT, mfxU16>>(
    std::make_pair(RGY_CHROMAFMT_MONOCHROME, (mfxU16)MFX_CHROMAFORMAT_MONOCHROME),
    std::make_pair(RGY_CHROMAFMT_YUV420,     (mfxU16)MFX_CHROMAFORMAT_YUV420),
    std::make_pair(RGY_CHROMAFMT_YUV422,     (mfxU16)MFX_CHROMAFORMAT_YUV422),
    std::make_pair(RGY_CHROMAFMT_YUV444,     (mfxU16)MFX_CHROMAFORMAT_YUV444)
    );

MAP_PAIR_0_1(chromafmt, rgy, RGY_CHROMAFMT, enc, mfxU16, RGY_CHROMAFMT_TO_MFX, RGY_CHROMAFMT_UNKNOWN, 0u);

#define MFX_EXT_MAKEFOURCC(A,B,C,D)    (MFX_MAKEFOURCC(A,B,C,D) | 0x80808080)
enum {
    MFX_EXT_FOURCC_YUV420_16 = MFX_EXT_MAKEFOURCC('Y', '0', '1', '6'),
    MFX_EXT_FOURCC_YUV420_12 = MFX_EXT_MAKEFOURCC('Y', '0', '1', '2'),
    MFX_EXT_FOURCC_YUV420_10 = MFX_EXT_MAKEFOURCC('Y', '0', '1', '0'),
    MFX_EXT_FOURCC_YUV422_16 = MFX_EXT_MAKEFOURCC('Y', '2', '1', '6'),
    MFX_EXT_FOURCC_YUV422_12 = MFX_EXT_MAKEFOURCC('Y', '2', '1', '2'),
    MFX_EXT_FOURCC_YUV422_10 = MFX_EXT_MAKEFOURCC('Y', '2', '1', '0'),
    MFX_EXT_FOURCC_YUV444_16 = MFX_EXT_MAKEFOURCC('Y', '4', '1', '6'),
    MFX_EXT_FOURCC_YUV444_12 = MFX_EXT_MAKEFOURCC('Y', '4', '1', '2'),
    MFX_EXT_FOURCC_YUV444_10 = MFX_EXT_MAKEFOURCC('Y', '4', '1', '0'),
    MFX_EXT_FOURCC_RGBP      = MFX_EXT_MAKEFOURCC('R', 'G', 'B', 'P'),
};
#undef MFX_EXT_MAKEFOURCC

static const auto RGY_CSP_TO_MFX = make_array<std::pair<RGY_CSP, mfxU32>>(
    std::make_pair(RGY_CSP_NA,        0),
    std::make_pair(RGY_CSP_NV12,      MFX_FOURCC_NV12),
    std::make_pair(RGY_CSP_YV12,      MFX_FOURCC_YV12),
    std::make_pair(RGY_CSP_YUY2,      MFX_FOURCC_YUY2),
    std::make_pair(RGY_CSP_YUV422,    0),
    std::make_pair(RGY_CSP_YUV444,    0),
    std::make_pair(RGY_CSP_NV16,      MFX_FOURCC_NV16),
    std::make_pair(RGY_CSP_YV12_09,   0),
    std::make_pair(RGY_CSP_YV12_10,   MFX_EXT_FOURCC_YUV420_10),
    std::make_pair(RGY_CSP_YV12_12,   0),
    std::make_pair(RGY_CSP_YV12_14,   0),
    std::make_pair(RGY_CSP_YV12_16,   MFX_EXT_FOURCC_YUV420_16),
    std::make_pair(RGY_CSP_P010,      MFX_FOURCC_P010),
    std::make_pair(RGY_CSP_YUV422_09, 0),
    std::make_pair(RGY_CSP_YUV422_10, MFX_EXT_FOURCC_YUV422_10),
    std::make_pair(RGY_CSP_YUV422_12, 0),
    std::make_pair(RGY_CSP_YUV422_14, 0),
    std::make_pair(RGY_CSP_YUV422_16, MFX_EXT_FOURCC_YUV422_16),
    std::make_pair(RGY_CSP_P210,      MFX_FOURCC_P210),
    std::make_pair(RGY_CSP_YUV444_09, 0),
    std::make_pair(RGY_CSP_YUV444_10, MFX_EXT_FOURCC_YUV444_10),
    std::make_pair(RGY_CSP_YUV444_12, 0),
    std::make_pair(RGY_CSP_YUV444_14, 0),
    std::make_pair(RGY_CSP_YUV444_16, MFX_EXT_FOURCC_YUV444_16),
    std::make_pair(RGY_CSP_VUYA,      MFX_FOURCC_AYUV),
    std::make_pair(RGY_CSP_Y210,      MFX_FOURCC_Y210),
    std::make_pair(RGY_CSP_Y216,      MFX_FOURCC_Y216),
    std::make_pair(RGY_CSP_Y410,      MFX_FOURCC_Y410),
    std::make_pair(RGY_CSP_Y416,      MFX_FOURCC_Y416),
    std::make_pair(RGY_CSP_RBGA64_10, MFX_FOURCC_Y410),
    std::make_pair(RGY_CSP_RBGA64,    MFX_FOURCC_Y416),
    std::make_pair(RGY_CSP_BGR32,     MFX_FOURCC_RGB4),
    std::make_pair(RGY_CSP_RGB32,     MFX_FOURCC_BGR4),
    std::make_pair(RGY_CSP_MFX_RGB,   MFX_FOURCC_AYUV),
    std::make_pair(RGY_CSP_RGB,       MFX_EXT_FOURCC_RGBP),
    std::make_pair(RGY_CSP_YC48,      0)
    );

MAP_PAIR_0_1(csp, rgy, RGY_CSP, enc, mfxU32, RGY_CSP_TO_MFX, RGY_CSP_NA, 0);

static const auto RGY_RESIZE_ALGO_TO_MFX = make_array<std::pair<RGY_VPP_RESIZE_ALGO, int>>(
    std::make_pair(RGY_VPP_RESIZE_AUTO,                 MFX_INTERPOLATION_DEFAULT),
    std::make_pair(RGY_VPP_RESIZE_MFX_NEAREST_NEIGHBOR, MFX_INTERPOLATION_NEAREST_NEIGHBOR),
    std::make_pair(RGY_VPP_RESIZE_MFX_BILINEAR,         MFX_INTERPOLATION_BILINEAR),
    std::make_pair(RGY_VPP_RESIZE_MFX_ADVANCED,         MFX_INTERPOLATION_ADVANCED)
);

MAP_PAIR_0_1(resize_algo, rgy, RGY_VPP_RESIZE_ALGO, enc, int, RGY_RESIZE_ALGO_TO_MFX, RGY_VPP_RESIZE_UNKNOWN, -1);


static const auto RGY_SCALING_MODE_TO_MFX = make_array<std::pair<RGY_VPP_RESIZE_MODE, int>>(
    std::make_pair(RGY_VPP_RESIZE_MODE_DEFAULT,      MFX_SCALING_MODE_DEFAULT),
    std::make_pair(RGY_VPP_RESIZE_MODE_MFX_LOWPOWER, MFX_SCALING_MODE_LOWPOWER),
    std::make_pair(RGY_VPP_RESIZE_MODE_MFX_QUALITY,  MFX_SCALING_MODE_QUALITY)
);

MAP_PAIR_0_1(resize_mode, rgy, RGY_VPP_RESIZE_MODE, enc, int, RGY_SCALING_MODE_TO_MFX, RGY_VPP_RESIZE_MODE_UNKNOWN, -1);

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
mfxU16 mfx_fourcc_to_chromafmt(mfxU32 fourcc) {
    switch (fourcc) {
    case MFX_FOURCC_AYUV:
    case MFX_FOURCC_RGB4:
    case MFX_FOURCC_BGR4:
    case MFX_FOURCC_Y410:
    case MFX_FOURCC_Y416:
        return MFX_CHROMAFORMAT_YUV444;
    case MFX_FOURCC_YUY2:
    case MFX_FOURCC_NV16:
    case MFX_FOURCC_P210:
    case MFX_FOURCC_Y210:
    case MFX_FOURCC_Y216:
        return MFX_CHROMAFORMAT_YUV422;
    case MFX_FOURCC_NV12:
    case MFX_FOURCC_YV12:
    case MFX_FOURCC_P010:
    default:
        return MFX_CHROMAFORMAT_YUV420;
    }
}

RGY_NOINLINE
mfxFrameInfo frameinfo_rgy_to_enc(VideoInfo info) {
    mfxFrameInfo mfx = { 0 };
    mfx.FourCC = csp_rgy_to_enc(info.csp);
    mfx.ChromaFormat = mfx_fourcc_to_chromafmt(mfx.FourCC);
    mfx.BitDepthLuma = (mfxU16)(info.bitdepth > 8 ? info.bitdepth : 0);
    mfx.BitDepthChroma = (mfxU16)(info.bitdepth > 8 ? info.bitdepth : 0);
    mfx.Shift = (fourccShiftUsed(mfx.FourCC) && RGY_CSP_BIT_DEPTH[info.csp] - info.bitdepth > 0) ? 1 : 0;
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
mfxFrameInfo frameinfo_rgy_to_enc(const RGYFrameInfo& info, const rgy_rational<int> fps, const rgy_rational<int> sar, const int blockSize) {
    mfxFrameInfo mfx = { 0 };
    mfx.FourCC = csp_rgy_to_enc(info.csp);
    mfx.ChromaFormat = mfx_fourcc_to_chromafmt(mfx.FourCC);
    mfx.BitDepthLuma = (mfxU16)(info.bitdepth > 8 ? info.bitdepth : 0);
    mfx.BitDepthChroma = (mfxU16)(info.bitdepth > 8 ? info.bitdepth : 0);
    mfx.Shift = (fourccShiftUsed(mfx.FourCC) && RGY_CSP_BIT_DEPTH[info.csp] - info.bitdepth > 0) ? 1 : 0;
    mfx.Width = (mfxU16)ALIGN(info.width, blockSize);
    mfx.Height = (mfxU16)ALIGN(info.height, blockSize);
    mfx.CropX = (mfxU16)0;
    mfx.CropY = (mfxU16)0;
    mfx.CropW = (mfxU16)info.width;
    mfx.CropH = (mfxU16)info.height;
    mfx.FrameRateExtN = fps.n();
    mfx.FrameRateExtD = fps.d();
    mfx.AspectRatioW = (mfxU16)sar.n();
    mfx.AspectRatioH = (mfxU16)sar.d();
    mfx.PicStruct = picstruct_rgy_to_enc(info.picstruct);
    return mfx;
}

RGY_NOINLINE
VideoInfo videooutputinfo(const mfxInfoMFX& mfx, const mfxExtVideoSignalInfo& vui, const mfxExtChromaLocInfo& chromaloc) {
    VideoInfo info;
    info.codec = codec_enc_to_rgy(mfx.CodecId);
    info.codecLevel = mfx.CodecLevel;
    info.codecProfile = mfx.CodecProfile;
    if (info.codec == RGY_CODEC_AV1) {
        info.videoDelay = 0;
    } else {
        info.videoDelay = ((mfx.GopRefDist - 1) > 0) + (((mfx.GopRefDist - 1) > 0) & ((mfx.GopRefDist - 1) > 2));
    }
    info.dstWidth = mfx.FrameInfo.CropW;
    info.dstHeight = mfx.FrameInfo.CropH;
    info.fpsN = mfx.FrameInfo.FrameRateExtN;
    info.fpsD = mfx.FrameInfo.FrameRateExtD;
    info.sar[0] = mfx.FrameInfo.AspectRatioW;
    info.sar[1] = mfx.FrameInfo.AspectRatioH;
    info.vui.descriptpresent = vui.ColourDescriptionPresent;
    info.vui.colorprim = (CspColorprim)vui.ColourPrimaries;
    info.vui.matrix = (CspMatrix)vui.MatrixCoefficients;
    info.vui.transfer = (CspTransfer)vui.TransferCharacteristics;
    info.vui.colorrange = vui.VideoFullRange ? RGY_COLORRANGE_FULL : RGY_COLORRANGE_UNSPECIFIED;
    info.vui.format = vui.VideoFormat;
    info.vui.chromaloc = (chromaloc.ChromaLocInfoPresentFlag && chromaloc.ChromaSampleLocTypeTopField) ? (CspChromaloc)(chromaloc.ChromaSampleLocTypeTopField+1) : RGY_CHROMALOC_UNSPECIFIED;
    info.picstruct = picstruct_enc_to_rgy(mfx.FrameInfo.PicStruct);
    info.bitdepth = (mfx.FrameInfo.BitDepthLuma == 0) ? 8 : mfx.FrameInfo.BitDepthLuma;
    info.csp = csp_enc_to_rgy(mfx.FrameInfo.FourCC);
    return info;
}

RGY_NOINLINE
VideoInfo videooutputinfo(const mfxFrameInfo& frameinfo) {
    VideoInfo info;
    info.codec = RGY_CODEC_RAW;
    info.dstWidth = frameinfo.CropW;
    info.dstHeight = frameinfo.CropH;
    info.fpsN = frameinfo.FrameRateExtN;
    info.fpsD = frameinfo.FrameRateExtD;
    info.sar[0] = frameinfo.AspectRatioW;
    info.sar[1] = frameinfo.AspectRatioH;
    info.picstruct = picstruct_enc_to_rgy(frameinfo.PicStruct);
    info.bitdepth = (frameinfo.BitDepthLuma == 0) ? 8 : frameinfo.BitDepthLuma;
    info.csp = csp_enc_to_rgy(frameinfo.FourCC);
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
const TCHAR *get_low_power_str(uint32_t LowPower) {
    switch (LowPower) {
    case MFX_CODINGOPTION_OFF: return _T(" PG");
    case MFX_CODINGOPTION_ON:  return _T(" FF");
    default: return _T("");
    }
}

RGY_NOINLINE
tstring qsv_memtype_str(uint32_t memtype) {
    tstring str;
    if (memtype & MFX_MEMTYPE_INTERNAL_FRAME)         str += _T("internal,");
    if (memtype & MFX_MEMTYPE_EXTERNAL_FRAME)         str += _T("external,");
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

RGY_NOINLINE
mfxHandleType mfxHandleTypeFromMemType(const MemType memType, const bool forOpenCLInterop) {
#if LIBVA_SUPPORT
    //VAではメモリタイプによらずhwデバイスの初期化とハンドルの取得が必要
    return MFX_HANDLE_VA_DISPLAY;
#else
    mfxHandleType hdl_t = (mfxHandleType)0;
    switch (memType) {
#if D3D_SURFACES_SUPPORT
    case D3D9_MEMORY:  hdl_t = (forOpenCLInterop) ? MFX_HANDLE_IDIRECT3D9EX : MFX_HANDLE_D3D9_DEVICE_MANAGER; break;
#if MFX_D3D11_SUPPORT
    case D3D11_MEMORY: hdl_t = MFX_HANDLE_D3D11_DEVICE; break;
#endif
#endif
#if LIBVA_SUPPORT
    case VA_MEMORY: hdl_t = MFX_HANDLE_VA_DISPLAY; break;
#endif
    default:
        break;
    }
    return hdl_t;
#endif
}

//ビットレート指定モードかどうか
bool isRCBitrateMode(int encmode) {
    static const auto RC_NON_BITRATE = make_array<int>(MFX_RATECONTROL_CQP, MFX_RATECONTROL_ICQ, MFX_RATECONTROL_LA_ICQ);
    return (std::find(RC_NON_BITRATE.begin(), RC_NON_BITRATE.end(), encmode) == RC_NON_BITRATE.end());
}

void RGYBitstream::addFrameData(RGYFrameData *frameData) {
    if (frameData != nullptr) {
        frameDataNum++;
        frameDataList = (RGYFrameData **)realloc(frameDataList, frameDataNum * sizeof(frameDataList[0]));
        frameDataList[frameDataNum - 1] = frameData;
    }
}

void RGYBitstream::clearFrameDataList() {
    frameDataNum = 0;
    if (frameDataList) {
        for (int i = 0; i < frameDataNum; i++) {
            if (frameDataList[i]) {
                delete frameDataList[i];
            }
        }
        free(frameDataList);
        frameDataList = nullptr;
    }
}
std::vector<RGYFrameData *> RGYBitstream::getFrameDataList() {
    return make_vector(frameDataList, frameDataNum);
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
    case MFX_FOURCC_RGB4: // -> RGY_CSP_BGR32
        return _T("bgr32");
    case MFX_FOURCC_BGR4: // -> RGY_CSP_RGB32
        return _T("rgb32");
    case MFX_FOURCC_AYUV:
        return _T("AYUV");
    case MFX_FOURCC_P010:
        return _T("p010");
    case MFX_FOURCC_P016:
        return _T("p016");
    case MFX_FOURCC_P210:
        return _T("p210");
    case MFX_FOURCC_Y210:
        return _T("y210");
    case MFX_FOURCC_Y216:
        return _T("y216");
    case MFX_FOURCC_Y410:
        return _T("y410");
    case MFX_FOURCC_Y416:
        return _T("y416");
    default:
        return _T("unsupported");
    }
}

RGY_NOINLINE
const TCHAR *ChromaFormatToStr(uint32_t format) {
    switch (format) {
    case MFX_CHROMAFORMAT_YUV400:     return _T("yuv400");
    case MFX_CHROMAFORMAT_YUV420:     return _T("yuv420");
    case MFX_CHROMAFORMAT_YUV422:     return _T("yuv422");
    case MFX_CHROMAFORMAT_YUV444:     return _T("yuv444");
    case MFX_CHROMAFORMAT_YUV411:     return _T("yuv411");
    case MFX_CHROMAFORMAT_YUV422V:    return _T("yuv422v");
    default:
        return _T("unsupported");
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
    case MFX_RATECONTROL_LA_HRD:
        return _T("LA-HRD (HRD compliant Lookahead)");
    case MFX_RATECONTROL_QVBR:
        return _T("Quality VBR bitrate");
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
#if LIBVA_SUPPORT
    case VA_MEMORY:
    case HW_MEMORY:
        return _T("va");
#endif
    default:
        return _T("unsupported");
    }
}

RGY_NOINLINE
tstring MFXPicStructToStr(uint32_t picstruct) {
    if (picstruct == 0) return _T("unknown");
    tstring str;
    if (picstruct & MFX_PICSTRUCT_PROGRESSIVE)       str += _T(",prog");
    if (picstruct & MFX_PICSTRUCT_FIELD_TFF)         str += _T(",tff");
    if (picstruct & MFX_PICSTRUCT_FIELD_BFF)         str += _T(",bff");
    if (picstruct & MFX_PICSTRUCT_FIELD_REPEATED)    str += _T(",repeat");
    if (picstruct & MFX_PICSTRUCT_FRAME_DOUBLING)    str += _T(",double");
    if (picstruct & MFX_PICSTRUCT_FRAME_TRIPLING)    str += _T(",triple");
    if (picstruct & MFX_PICSTRUCT_FIELD_SINGLE)      str += _T(",single");
    if (picstruct & MFX_PICSTRUCT_FIELD_PAIRED_PREV) str += _T(",pair_prev");
    if (picstruct & MFX_PICSTRUCT_FIELD_PAIRED_NEXT) str += _T(",pair_next");
    return str.substr(1);
}

RGY_NOINLINE
tstring MFXImplToStr(uint32_t impl) {
    if (impl == 0) return _T("auto");
    tstring str;
    if ((impl & 0x00ff) == MFX_IMPL_SOFTWARE)      str += _T(",sw");
    if ((impl & 0x00ff) == MFX_IMPL_HARDWARE)      str += _T(",hw");
    if ((impl & 0x00ff) == MFX_IMPL_AUTO_ANY)      str += _T(",auto_any");
    if ((impl & 0x00ff) == MFX_IMPL_HARDWARE_ANY)  str += _T(",hw_any");
    if ((impl & 0x00ff) == MFX_IMPL_HARDWARE2)     str += _T(",hw2");
    if ((impl & 0x00ff) == MFX_IMPL_HARDWARE3)     str += _T(",hw3");
    if ((impl & 0x00ff) == MFX_IMPL_HARDWARE4)     str += _T(",hw4");
    if ((impl & 0xff00) == MFX_IMPL_RUNTIME)       str += _T(",runtime");
    if ((impl & 0xff00) == MFX_IMPL_VIA_ANY)       str += _T(",via_any");
    if ((impl & 0xff00) == MFX_IMPL_VIA_D3D9)      str += _T(",via_d3d9");
    if ((impl & 0xff00) == MFX_IMPL_VIA_D3D11)     str += _T(",via_d3d11");
    if ((impl & 0xff00) == MFX_IMPL_VIA_VAAPI)     str += _T(",via_va");
    if ((impl & 0xff00) == MFX_IMPL_VIA_HDDLUNITE) str += _T(",via_hddlunite");
    return str.substr(1);
}

RGY_NOINLINE
tstring MFXAccelerationModeToStr(mfxAccelerationMode impl) {
    if (impl == 0) return _T("auto");
    tstring str;
    if ((impl & 0x0fff) == MFX_ACCEL_MODE_VIA_D3D9)              str += _T(",d3d9");
    if ((impl & 0x0fff) == MFX_ACCEL_MODE_VIA_D3D11)             str += _T(",d3d11");
    if ((impl & 0x0fff) == MFX_ACCEL_MODE_VIA_VAAPI)             str += _T(",vaapi");
    if ((impl & 0x0fff) == MFX_ACCEL_MODE_VIA_VAAPI_DRM_MODESET) str += _T(",vaapi_drm");
    if ((impl & 0x0fff) == MFX_ACCEL_MODE_VIA_VAAPI_GLX)         str += _T(",vaapi_glx");
    if ((impl & 0x0fff) == MFX_ACCEL_MODE_VIA_VAAPI_X11)         str += _T(",vaapi_x11");
    if ((impl & 0x0fff) == MFX_ACCEL_MODE_VIA_VAAPI_WAYLAND)     str += _T(",vaapi_wayland");
    if ((impl & 0xff00) == MFX_ACCEL_MODE_VIA_HDDLUNITE)         str += _T(",hddlunite");
    return str.substr(1);
}

RGY_NOINLINE
tstring MFXImplTypeToStr(mfxImplType impl) {
    if (impl == 0) return _T("auto");
    tstring str;
    if ((impl & MFX_IMPL_TYPE_SOFTWARE) == MFX_IMPL_TYPE_SOFTWARE) str += _T(",sw");
    if ((impl & MFX_IMPL_TYPE_HARDWARE) == MFX_IMPL_TYPE_HARDWARE) str += _T(",hw");
    return str.substr(1);
}

RGY_NOINLINE
const TCHAR *get_vpp_image_stab_mode_str(int mode) {
    switch (mode) {
    case MFX_IMAGESTAB_MODE_UPSCALE: return _T("upscale");
    case MFX_IMAGESTAB_MODE_BOXING:  return _T("boxing");
    default: return _T("unknown");
    }
}

#if !ENABLE_AVSW_READER
#define TTMATH_NOASM
#include "ttmath/ttmath.h"

int64_t rational_rescale(int64_t v, rgy_rational<int> from, rgy_rational<int> to) {
    auto mul = rgy_rational<int64_t>((int64_t)from.n() * (int64_t)to.d(), (int64_t)from.d() * (int64_t)to.n());

#if _M_IX86
#define RESCALE_INT_SIZE 4
#else
#define RESCALE_INT_SIZE 2
#endif
    ttmath::Int<RESCALE_INT_SIZE> tmp1 = v;
    tmp1 *= mul.n();
    ttmath::Int<RESCALE_INT_SIZE> tmp2 = mul.d();

    tmp1 = (tmp1 + tmp2 - 1) / tmp2;
    int64_t ret;
    tmp1.ToInt(ret);
    return ret;
}

#endif
