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
    std::make_pair(RGY_CSP_RGB3,      MFX_FOURCC_RGB3),
    std::make_pair(RGY_CSP_RGB4,      MFX_FOURCC_RGB4),
    std::make_pair(RGY_CSP_YC48,      0)
    );

MAP_PAIR_0_1(csp, rgy, RGY_CSP, enc, mfxU32, RGY_CSP_TO_MFX, RGY_CSP_NA, 0);

__declspec(noinline)
mfxU16 picstruct_rgy_to_enc(RGY_PICSTRUCT picstruct) {
    if (picstruct & RGY_PICSTRUCT_TFF) return (mfxU16)MFX_PICSTRUCT_FIELD_TFF;
    if (picstruct & RGY_PICSTRUCT_BFF) return (mfxU16)MFX_PICSTRUCT_FIELD_BFF;
    return (mfxU16)MFX_PICSTRUCT_PROGRESSIVE;
}

__declspec(noinline)
RGY_PICSTRUCT picstruct_enc_to_rgy(mfxU16 picstruct) {
    if (picstruct & MFX_PICSTRUCT_FIELD_TFF) return RGY_PICSTRUCT_FRAME_TFF;
    if (picstruct & MFX_PICSTRUCT_FIELD_BFF) return RGY_PICSTRUCT_FRAME_BFF;
    return RGY_PICSTRUCT_FRAME;
}

__declspec(noinline)
mfxFrameInfo frameinfo_rgy_to_enc(VideoInfo info) {
    mfxFrameInfo mfx = { 0 };
    mfx.FourCC = RGY_CSP_TO_MFX_FOURCC[info.csp];
    mfx.ChromaFormat = (mfxU16)chromafmt_rgy_to_enc(RGY_CSP_CHROMA_FORMAT[info.csp]);
    mfx.BitDepthLuma = (mfxU16)RGY_CSP_BIT_DEPTH[info.csp];
    mfx.BitDepthChroma = (mfxU16)RGY_CSP_BIT_DEPTH[info.csp];
    mfx.Shift = (mfxU16)info.shift;
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

__declspec(noinline)
VideoInfo videooutputinfo(const mfxInfoMFX& mfx, const mfxExtVideoSignalInfo& vui) {
    VideoInfo info;
    info.codec = codec_enc_to_rgy(mfx.CodecId);
    info.codecLevel = mfx.CodecLevel;
    info.codecProfile = mfx.CodecProfile;
    info.videoDelay = ((mfx.GopRefDist - 1) > 0) + (((mfx.GopRefDist - 1) > 0) & ((mfx.GopRefDist - 1) > 2));;
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
    info.shift = mfx.FrameInfo.Shift;
    info.csp = csp_enc_to_rgy(mfx.FrameInfo.FourCC);
    return info;
}
