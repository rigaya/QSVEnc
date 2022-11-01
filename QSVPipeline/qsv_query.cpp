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

#include <stdio.h>
#include <vector>
#include <numeric>
#include <memory>
#include <sstream>
#include <future>
#include <algorithm>
#include <optional>
#include <type_traits>
#include "rgy_osdep.h"
#ifndef _MSC_VER
#include <sys/sysinfo.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <iconv.h>
#endif
#include "rgy_tchar.h"
#include "rgy_util.h"
#include "rgy_avutil.h"
#include "rgy_arch.h"
#include "qsv_util.h"
#include "qsv_prm.h"
#include "qsv_device.h"
#include "rgy_osdep.h"
#include "rgy_env.h"
#include "qsv_query.h"
#include "qsv_session.h"
#include "qsv_hw_device.h"
#include "cpu_info.h"
#pragma warning (push)
#pragma warning (disable: 4201) //C4201: 非標準の拡張機能が使用されています: 無名の構造体または共用体です。
#pragma warning (disable: 4996) //C4996: 'MFXInit': が古い形式として宣言されました。
#pragma warning (disable: 4819) //C4819: ファイルは、現在のコード ページ (932) で表示できない文字を含んでいます。データの損失を防ぐために、ファイルを Unicode 形式で保存してください。
#include "mfxjpeg.h"
#pragma warning (pop)

#if D3D_SURFACES_SUPPORT
#include "qsv_hw_d3d9.h"
#include "qsv_hw_d3d11.h"

#include "qsv_allocator_d3d9.h"
#include "qsv_allocator_d3d11.h"
#endif

#if LIBVA_SUPPORT
#include "qsv_hw_va.h"
#include "qsv_allocator_va.h"
#endif

#if 1
QSV_CPU_GEN getCPUGenCpuid() {
#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
    int CPUInfo[4] = {-1};
    __cpuid(CPUInfo, 0x01);
    const bool bMOVBE  = !!(CPUInfo[2] & (1<<22));
    const bool bRDRand = !!(CPUInfo[2] & (1<<30));

    __cpuid(CPUInfo, 0x07);
    const bool bSHA        = !!(CPUInfo[1] & (1<<29));
    const bool bClflushOpt = !!(CPUInfo[1] & (1<<23));
    const bool bADX        = !!(CPUInfo[1] & (1<<19));
    const bool bRDSeed     = !!(CPUInfo[1] & (1<<18));
    const bool bFsgsbase   = !!(CPUInfo[1] & (1));

    if (bSHA && !bADX)       return CPU_GEN_GOLDMONT;
    if (bClflushOpt)         return CPU_GEN_SKYLAKE;
    if (bRDSeed)             return CPU_GEN_BROADWELL;
    if (bMOVBE && bFsgsbase) return CPU_GEN_HASWELL;
    if (bFsgsbase)           return CPU_GEN_IVYBRIDGE;

    if (bRDRand) {
        __cpuid(CPUInfo, 0x02);
        return (CPUInfo[0] == 0x61B4A001) ? CPU_GEN_AIRMONT : CPU_GEN_SILVERMONT;
    }
#endif //#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
    return CPU_GEN_SANDYBRIDGE;
}
#endif

static const auto RGY_CPU_GEN_TO_MFX = make_array<std::pair<QSV_CPU_GEN, uint32_t>>(
    std::make_pair(CPU_GEN_UNKNOWN, MFX_PLATFORM_UNKNOWN),
    std::make_pair(CPU_GEN_SANDYBRIDGE, MFX_PLATFORM_SANDYBRIDGE),
    std::make_pair(CPU_GEN_IVYBRIDGE, MFX_PLATFORM_IVYBRIDGE),
    std::make_pair(CPU_GEN_HASWELL, MFX_PLATFORM_HASWELL),
    std::make_pair(CPU_GEN_SILVERMONT, MFX_PLATFORM_BAYTRAIL),
    std::make_pair(CPU_GEN_BROADWELL, MFX_PLATFORM_BROADWELL),
    std::make_pair(CPU_GEN_AIRMONT, MFX_PLATFORM_CHERRYTRAIL),
    std::make_pair(CPU_GEN_SKYLAKE, MFX_PLATFORM_SKYLAKE),
    std::make_pair(CPU_GEN_GOLDMONT, MFX_PLATFORM_APOLLOLAKE),
    std::make_pair(CPU_GEN_KABYLAKE, MFX_PLATFORM_KABYLAKE),
    std::make_pair(CPU_GEN_GEMINILAKE, MFX_PLATFORM_GEMINILAKE),
    std::make_pair(CPU_GEN_COFFEELAKE, MFX_PLATFORM_COFFEELAKE),
    std::make_pair(CPU_GEN_CANNONLAKE, MFX_PLATFORM_CANNONLAKE),
    std::make_pair(CPU_GEN_ICELAKE, MFX_PLATFORM_ICELAKE),
    std::make_pair(CPU_GEN_JASPERLAKE, MFX_PLATFORM_JASPERLAKE),
    std::make_pair(CPU_GEN_ELKHARTLAKE, MFX_PLATFORM_ELKHARTLAKE),
    std::make_pair(CPU_GEN_TIGERLAKE, MFX_PLATFORM_TIGERLAKE),
    std::make_pair(CPU_GEN_ROCKETLAKE, MFX_PLATFORM_ROCKETLAKE),
    std::make_pair(CPU_GEN_ALDERLAKE_S, MFX_PLATFORM_ALDERLAKE_S),
    std::make_pair(CPU_GEN_ALDERLAKE_P, MFX_PLATFORM_ALDERLAKE_P),
    std::make_pair(CPU_GEN_ARCTICSOUND_P, MFX_PLATFORM_ARCTICSOUND_P),
    std::make_pair(CPU_GEN_XEHP_SDV, MFX_PLATFORM_XEHP_SDV),
    std::make_pair(CPU_GEN_DG2, MFX_PLATFORM_DG2),
    std::make_pair(CPU_GEN_ATS_M, MFX_PLATFORM_ATS_M),
    std::make_pair(CPU_GEN_ALDERLAKE_N, MFX_PLATFORM_ALDERLAKE_N),
    std::make_pair(CPU_GEN_KEEMBAY, MFX_PLATFORM_KEEMBAY)
    );
MAP_PAIR_0_1(cpu_gen, rgy, QSV_CPU_GEN, enc, uint32_t, RGY_CPU_GEN_TO_MFX, CPU_GEN_UNKNOWN, MFX_PLATFORM_UNKNOWN);


BOOL Check_HWUsed(mfxIMPL impl) {
    static const int HW_list[] = {
        MFX_IMPL_HARDWARE,
        MFX_IMPL_HARDWARE_ANY,
        MFX_IMPL_HARDWARE2,
        MFX_IMPL_HARDWARE3,
        MFX_IMPL_HARDWARE4,
        0
    };
    for (int i = 0; HW_list[i]; i++)
        if (HW_list[i] == (HW_list[i] & (int)impl))
            return TRUE;
    return FALSE;
}

int GetAdapterID(mfxIMPL impl) {
    return (std::max)(0, MFX_IMPL_BASETYPE(impl) - MFX_IMPL_HARDWARE_ANY);
}

int GetAdapterID(mfxSession session) {
    mfxIMPL impl;
    MFXQueryIMPL(session, &impl);
    return GetAdapterID(impl);
}

int GetAdapterID(MFXVideoSession *session) {
    mfxIMPL impl;
    MFXQueryIMPL(*session, &impl);
    return GetAdapterID(impl);
}

mfxVersion get_mfx_lib_version(const mfxIMPL impl, const QSVDeviceNum deviceNum) {
    if (impl == MFX_IMPL_SOFTWARE) {
        return LIB_VER_LIST[0];
    }
#if 1
    { // 新手法
        std::unique_ptr<CQSVHWDevice> hwdev;
        MFXVideoSession2 session;
        auto memType = HW_MEMORY;
        MFXVideoSession2Params params;
        auto log = std::make_shared<RGYLog>(nullptr, RGY_LOG_ERROR);
        auto err = InitSessionAndDevice(hwdev, session, memType, deviceNum, params, log);
        if (err == RGY_ERR_NONE) {
            mfxVersion ver;
            auto sts = session.QueryVersion(&ver);
            if (sts != MFX_ERR_NONE) {
                return LIB_VER_LIST[0];
            }
            return ver;
        }
    }
#endif
    mfxVersion ver = MFX_LIB_VERSION_1_1;
    auto session_deleter = [](MFXVideoSession *session) { session->Close(); };
    std::unique_ptr<MFXVideoSession, decltype(session_deleter)> test(new MFXVideoSession(), session_deleter);
    mfxStatus sts = test->Init(impl, &ver);
    if (sts != MFX_ERR_NONE) {
        return LIB_VER_LIST[0];
    }
    sts = test->QueryVersion(&ver);
    if (sts != MFX_ERR_NONE) {
        return LIB_VER_LIST[0];
    }
    auto log = std::make_shared<RGYLog>(nullptr, RGY_LOG_ERROR);
#if D3D_SURFACES_SUPPORT
#if MFX_D3D11_SUPPORT
    if ((impl & MFX_IMPL_VIA_D3D11) == MFX_IMPL_VIA_D3D11) {
        auto hwdev = std::make_unique<CQSVD3D11Device>(log);
        sts = hwdev->Init(NULL, 0, GetAdapterID(*test.get()));
    } else
#endif // #if MFX_D3D11_SUPPORT
    if ((impl & MFX_IMPL_VIA_D3D9) == MFX_IMPL_VIA_D3D9) {
        auto hwdev = std::make_unique<CQSVD3D9Device>(log);
        sts = hwdev->Init(NULL, 0, GetAdapterID(*test.get()));
    }
#elif LIBVA_SUPPORT
     {
        auto hwdev = std::unique_ptr<CQSVHWDevice>(CreateVAAPIDevice("", MFX_LIBVA_DRM, log));
        sts = hwdev->Init(NULL, 0, GetAdapterID(*test.get()));
    }
#endif
    return (sts == MFX_ERR_NONE) ? ver : LIB_VER_LIST[0];
}

mfxVersion get_mfx_libhw_version(const QSVDeviceNum deviceNum) {
    static const mfxU32 impl_list[] = {
        MFX_IMPL_HARDWARE_ANY | MFX_IMPL_VIA_D3D11,
        MFX_IMPL_HARDWARE_ANY,
        MFX_IMPL_HARDWARE,
    };
    mfxVersion test = { 0 };
    //Win7でD3D11のチェックをやると、
    //デスクトップコンポジションが切られてしまう問題が発生すると報告を頂いたので、
    //D3D11をWin8以降に限定
    for (int i = (check_OS_Win8orLater() ? 0 : 1); i < _countof(impl_list); i++) {
        test = get_mfx_lib_version(impl_list[i], deviceNum);
        if (check_lib_version(test, MFX_LIB_VERSION_1_1))
            break;
    }
    return test;
}
mfxVersion get_mfx_libsw_version() {
    return LIB_VER_LIST[0];
}

QSVVideoParam::QSVVideoParam(uint32_t CodecId, mfxVersion mfxver_) :
    mfxVer(mfxver_), isVppParam(false), videoPrmVpp(), videoPrm(), buf(), spsbuf(), ppsbuf(), spspps(),
    cop(), cop2(), cop3(), copVp8(), vp9Prm(), hevcPrm(), av1BitstreamPrm(), av1ResolutionPrm(), av1TilePrm(), hyperModePrm() {
    memset(spsbuf, 0, sizeof(spsbuf));
    memset(ppsbuf, 0, sizeof(ppsbuf));
    INIT_MFX_EXT_BUFFER(spspps, MFX_EXTBUFF_CODING_OPTION_SPSPPS);
    spspps.SPSBuffer = spsbuf;
    spspps.SPSBufSize = sizeof(spsbuf);
    spspps.PPSBuffer = ppsbuf;
    spspps.PPSBufSize = sizeof(ppsbuf);

    INIT_MFX_EXT_BUFFER(cop, MFX_EXTBUFF_CODING_OPTION);
    INIT_MFX_EXT_BUFFER(cop2, MFX_EXTBUFF_CODING_OPTION2);
    INIT_MFX_EXT_BUFFER(cop3, MFX_EXTBUFF_CODING_OPTION3);
    INIT_MFX_EXT_BUFFER(copVp8, MFX_EXTBUFF_VP8_CODING_OPTION);
    INIT_MFX_EXT_BUFFER(hevcPrm, MFX_EXTBUFF_HEVC_PARAM);
    INIT_MFX_EXT_BUFFER(vp9Prm, MFX_EXTBUFF_VP9_PARAM);
    INIT_MFX_EXT_BUFFER(av1BitstreamPrm, MFX_EXTBUFF_AV1_BITSTREAM_PARAM);
    INIT_MFX_EXT_BUFFER(av1ResolutionPrm, MFX_EXTBUFF_AV1_RESOLUTION_PARAM);
    INIT_MFX_EXT_BUFFER(av1TilePrm, MFX_EXTBUFF_AV1_TILE_PARAM);
    INIT_MFX_EXT_BUFFER(hyperModePrm, MFX_EXTBUFF_HYPER_MODE_PARAM);

    if (add_cop(CodecId)) {
        buf.push_back((mfxExtBuffer *)&cop);
    }
    if (CodecId == MFX_CODEC_AVC || CodecId == MFX_CODEC_HEVC) {
        buf.push_back((mfxExtBuffer *)&spspps);
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_6)) {
        buf.push_back((mfxExtBuffer *)&cop2);
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_11)) {
        buf.push_back((mfxExtBuffer *)&cop3);
    }
    if (CodecId == MFX_CODEC_VP8) {
        buf.push_back((mfxExtBuffer *)&copVp8);
    }
    if (CodecId == MFX_CODEC_VP9 && check_lib_version(mfxVer, MFX_LIB_VERSION_1_26)) {
        buf.push_back((mfxExtBuffer *)&vp9Prm);
    }
    if (CodecId == MFX_CODEC_HEVC && check_lib_version(mfxVer, MFX_LIB_VERSION_1_26)) {
        buf.push_back((mfxExtBuffer *)&hevcPrm);
    }
    if (CodecId == MFX_CODEC_AV1 && check_lib_version(mfxVer, MFX_LIB_VERSION_2_5)) {
        buf.push_back((mfxExtBuffer *)&av1BitstreamPrm);
        //buf.push_back((mfxExtBuffer *)&av1ResolutionPrm);
        //buf.push_back((mfxExtBuffer *)&av1TilePrm);
    }
    if (ENABLE_HYPER_MODE
        && (CodecId == MFX_CODEC_AVC || CodecId == MFX_CODEC_HEVC || CodecId == MFX_CODEC_AV1)
        && check_lib_version(mfxVer, MFX_LIB_VERSION_2_5)) {
        buf.push_back((mfxExtBuffer *)&hyperModePrm);
    }

    RGY_MEMSET_ZERO(videoPrm);
    videoPrm.NumExtParam = (mfxU16)buf.size();
    videoPrm.ExtParam = &buf[0];

    RGY_MEMSET_ZERO(videoPrmVpp);
};

std::vector<RGY_CSP> CheckDecFeaturesInternal(MFXVideoSession& session, mfxVersion mfxVer, mfxU32 codecId) {
    std::vector<RGY_CSP> supportedCsp;
    mfxIMPL impl;
    session.QueryIMPL(&impl);
    const auto HARDWARE_IMPL = make_array<mfxIMPL>(MFX_IMPL_HARDWARE, MFX_IMPL_HARDWARE_ANY, MFX_IMPL_HARDWARE2, MFX_IMPL_HARDWARE3, MFX_IMPL_HARDWARE4);
    //const bool bHardware = HARDWARE_IMPL.end() != std::find(HARDWARE_IMPL.begin(), HARDWARE_IMPL.end(), MFX_IMPL_BASETYPE(impl));

    mfxVideoParam videoPrm, videoPrmOut;
    memset(&videoPrm,  0, sizeof(videoPrm));
    videoPrm.AsyncDepth                  = 3;
    videoPrm.IOPattern                   = MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    videoPrm.mfx.CodecId                 = codecId;
    switch (codecId) {
    case MFX_CODEC_AVC:
        videoPrm.mfx.CodecLevel          = MFX_LEVEL_AVC_41;
        videoPrm.mfx.CodecProfile        = MFX_PROFILE_AVC_HIGH;
        break;
    case MFX_CODEC_HEVC:
        videoPrm.mfx.CodecLevel          = MFX_LEVEL_HEVC_4;
        videoPrm.mfx.CodecProfile        = MFX_PROFILE_HEVC_MAIN;
        break;
    case MFX_CODEC_MPEG2:
        videoPrm.mfx.CodecLevel          = MFX_LEVEL_MPEG2_MAIN;
        videoPrm.mfx.CodecProfile        = MFX_PROFILE_MPEG2_MAIN;
        break;
    case MFX_CODEC_VC1:
        videoPrm.mfx.CodecLevel          = 0;
        videoPrm.mfx.CodecProfile        = MFX_PROFILE_VC1_ADVANCED;
        break;
    case MFX_CODEC_VP8:
        break;
    case MFX_CODEC_VP9:
        videoPrm.mfx.CodecProfile        = MFX_PROFILE_VP9_0;
        break;
    case MFX_CODEC_AV1:
        videoPrm.mfx.CodecProfile        = MFX_PROFILE_AV1_MAIN;
        videoPrm.mfx.CodecLevel          = MFX_LEVEL_AV1_4;
        break;
    default:
        return supportedCsp;
    }
    videoPrm.mfx.EncodedOrder            = 0;
    videoPrm.mfx.FrameInfo.FrameRateExtN = 30000;
    videoPrm.mfx.FrameInfo.FrameRateExtD = 1001;
    videoPrm.mfx.FrameInfo.FourCC        = MFX_FOURCC_NV12;
    videoPrm.mfx.FrameInfo.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    videoPrm.mfx.FrameInfo.PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
    videoPrm.mfx.FrameInfo.AspectRatioW  = 1;
    videoPrm.mfx.FrameInfo.AspectRatioH  = 1;
    videoPrm.mfx.FrameInfo.Width         = 1920;
    videoPrm.mfx.FrameInfo.Height        = 1088;
    videoPrm.mfx.FrameInfo.CropX         = 0;
    videoPrm.mfx.FrameInfo.CropY         = 0;
    videoPrm.mfx.FrameInfo.CropW         = 1920;
    videoPrm.mfx.FrameInfo.CropH         = 1080;

    memcpy(&videoPrmOut, &videoPrm, sizeof(videoPrm));

    switch (codecId) {
    //デフォルトでデコード可能なもの
    case MFX_CODEC_AVC:
    case MFX_CODEC_MPEG2:
    case MFX_CODEC_VC1:
        break;
    //不明なものはテストする
    default:
        {
        mfxStatus ret = MFXVideoDECODE_Query(session , &videoPrm, &videoPrmOut);
        if (ret < MFX_ERR_NONE) { // QSV_WRN_XXX (> 0) は無視する
            return supportedCsp;
        }
        break;
        }
    }
    supportedCsp.push_back(RGY_CSP_NV12);
    supportedCsp.push_back(RGY_CSP_YV12);


#define CHECK_FEATURE(rgy_csp, required_ver) { \
        if (check_lib_version(mfxVer, (required_ver))) { \
            memcpy(&videoPrmOut, &videoPrm, sizeof(videoPrm)); \
            if (MFXVideoDECODE_Query(session, &videoPrm, &videoPrmOut) >= MFX_ERR_NONE) { \
                supportedCsp.push_back(rgy_csp); \
            } \
        } \
    }

    static const auto test_csp = make_array<RGY_CSP>(
        RGY_CSP_YV12_09,
        RGY_CSP_YV12_10,
        RGY_CSP_YV12_12,
        RGY_CSP_YV12_14,
        RGY_CSP_YV12_16,
        RGY_CSP_YUV422,
        RGY_CSP_YUV422_09,
        RGY_CSP_YUV422_10,
        RGY_CSP_YUV422_12,
        RGY_CSP_YUV422_14,
        RGY_CSP_YUV422_16,
        RGY_CSP_YUV444,
        RGY_CSP_YUV444_09,
        RGY_CSP_YUV444_10,
        RGY_CSP_YUV444_12,
        RGY_CSP_YUV444_14,
        RGY_CSP_YUV444_16
        );

    mfxVideoParam videoPrmTmp = videoPrm;
    for (const auto& test : test_csp) {
        switch (RGY_CSP_CHROMA_FORMAT[test]) {
        case RGY_CHROMAFMT_YUV420: videoPrm.mfx.FrameInfo.FourCC = (RGY_CSP_BIT_DEPTH[test] >  8) ? MFX_FOURCC_P010 : MFX_FOURCC_NV12; break;
        case RGY_CHROMAFMT_YUV422: videoPrm.mfx.FrameInfo.FourCC = (RGY_CSP_BIT_DEPTH[test] >  8) ? MFX_FOURCC_Y210 : MFX_FOURCC_YUY2; break;
        case RGY_CHROMAFMT_YUV444: videoPrm.mfx.FrameInfo.FourCC = (RGY_CSP_BIT_DEPTH[test] > 10) ? MFX_FOURCC_Y416 : ((RGY_CSP_BIT_DEPTH[test] > 8) ? MFX_FOURCC_Y410 : MFX_FOURCC_AYUV); break;
        default: videoPrm.mfx.FrameInfo.FourCC = MFX_FOURCC_NV12;
            break;
        }
        
        videoPrm.mfx.FrameInfo.ChromaFormat = (mfxU16)chromafmt_rgy_to_enc(RGY_CSP_CHROMA_FORMAT[test]);
        if (codecId == MFX_CODEC_HEVC) {
            if (RGY_CSP_CHROMA_FORMAT[test] == RGY_CHROMAFMT_YUV420) {
                videoPrm.mfx.CodecProfile = (mfxU16)((RGY_CSP_BIT_DEPTH[test] > 8) ? MFX_PROFILE_HEVC_MAIN10 : MFX_PROFILE_HEVC_MAIN);
            } else {
                videoPrm.mfx.CodecProfile = (mfxU16)MFX_PROFILE_HEVC_REXT;
            }
        } else if (codecId == MFX_CODEC_VP9) {
            videoPrm.mfx.CodecProfile = (mfxU16)((RGY_CSP_BIT_DEPTH[test] > 8) ? MFX_PROFILE_VP9_2 : MFX_PROFILE_VP9_0);
        } else if (codecId == MFX_CODEC_AV1) {
            if (RGY_CSP_CHROMA_FORMAT[test] == RGY_CHROMAFMT_YUV420) {
                videoPrm.mfx.CodecProfile = MFX_PROFILE_AV1_MAIN;
            } else {
                videoPrm.mfx.CodecProfile = MFX_PROFILE_AV1_PRO;
            }
        } else {
            break;
        }
        if (fourccShiftUsed(videoPrm.mfx.FrameInfo.FourCC)) {
            videoPrm.mfx.FrameInfo.BitDepthLuma = (mfxU16)((RGY_CSP_BIT_DEPTH[test] > 8) ? RGY_CSP_BIT_DEPTH[test] : 0);
            videoPrm.mfx.FrameInfo.BitDepthChroma = (mfxU16)((RGY_CSP_BIT_DEPTH[test] > 8) ? RGY_CSP_BIT_DEPTH[test] : 0);
            videoPrm.mfx.FrameInfo.Shift = (RGY_CSP_BIT_DEPTH[test] > 8) ? 1 : 0;
        } else {
            videoPrm.mfx.FrameInfo.BitDepthLuma = 0;
            videoPrm.mfx.FrameInfo.BitDepthChroma = 0;
            videoPrm.mfx.FrameInfo.Shift = 0;
        }
        CHECK_FEATURE(test, MFX_LIB_VERSION_1_19);
        videoPrm = videoPrmTmp;
    }

#undef CHECK_FEATURE
    return supportedCsp;
}

mfxU64 CheckVppFeaturesInternal(MFXVideoSession& session, mfxVersion mfxVer) {
    using namespace std;

    mfxU64 result = 0x00;
    result |= VPP_FEATURE_RESIZE;
    result |= VPP_FEATURE_DEINTERLACE;
    result |= VPP_FEATURE_DENOISE;
    result |= VPP_FEATURE_DETAIL_ENHANCEMENT;
    result |= VPP_FEATURE_PROC_AMP;
    if (!check_lib_version(mfxVer, MFX_LIB_VERSION_1_3))
        return result;

    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_13)) {
        result |= VPP_FEATURE_DEINTERLACE_AUTO;
        result |= VPP_FEATURE_DEINTERLACE_IT_MANUAL;
    }
    mfxIMPL impl;
    session.QueryIMPL(&impl);
    const auto HARDWARE_IMPL = make_array<mfxIMPL>(MFX_IMPL_HARDWARE, MFX_IMPL_HARDWARE_ANY, MFX_IMPL_HARDWARE2, MFX_IMPL_HARDWARE3, MFX_IMPL_HARDWARE4);
    const bool bHardware = HARDWARE_IMPL.end() != std::find(HARDWARE_IMPL.begin(), HARDWARE_IMPL.end(), MFX_IMPL_BASETYPE(impl));

    const bool bSetDoNotUseTag = getCPUGen(&session) < CPU_GEN_HASWELL;

    mfxExtVPPDoUse vppDoUse;
    mfxExtVPPDoUse vppDoNotUse;
    mfxExtVPPFrameRateConversion vppFpsConv;
    mfxExtVPPImageStab vppImageStab;
    mfxExtVPPVideoSignalInfo vppVSI;
    mfxExtVPPRotation vppRotate;
    mfxExtVPPMirroring vppMirror;
    mfxExtVPPScaling vppScaleQuality;
    mfxExtVppMctf vppMctf;
    mfxExtVPPDenoise2 vppDenoise2;
    INIT_MFX_EXT_BUFFER(vppDoUse,        MFX_EXTBUFF_VPP_DOUSE);
    INIT_MFX_EXT_BUFFER(vppDoNotUse,     MFX_EXTBUFF_VPP_DONOTUSE);
    INIT_MFX_EXT_BUFFER(vppFpsConv,      MFX_EXTBUFF_VPP_FRAME_RATE_CONVERSION);
    INIT_MFX_EXT_BUFFER(vppImageStab,    MFX_EXTBUFF_VPP_IMAGE_STABILIZATION);
    INIT_MFX_EXT_BUFFER(vppVSI,          MFX_EXTBUFF_VPP_VIDEO_SIGNAL_INFO);
    INIT_MFX_EXT_BUFFER(vppRotate,       MFX_EXTBUFF_VPP_ROTATION);
    INIT_MFX_EXT_BUFFER(vppMirror,       MFX_EXTBUFF_VPP_MIRRORING);
    INIT_MFX_EXT_BUFFER(vppScaleQuality, MFX_EXTBUFF_VPP_SCALING);
    INIT_MFX_EXT_BUFFER(vppMctf,         MFX_EXTBUFF_VPP_MCTF);
    INIT_MFX_EXT_BUFFER(vppDenoise2,     MFX_EXTBUFF_VPP_DENOISE2);

    vppFpsConv.Algorithm = MFX_FRCALGM_FRAME_INTERPOLATION;
    vppImageStab.Mode = MFX_IMAGESTAB_MODE_UPSCALE;
    vppVSI.In.TransferMatrix = MFX_TRANSFERMATRIX_BT601;
    vppVSI.Out.TransferMatrix = MFX_TRANSFERMATRIX_BT709;
    vppVSI.In.NominalRange = MFX_NOMINALRANGE_16_235;
    vppVSI.Out.NominalRange = MFX_NOMINALRANGE_0_255;
    vppRotate.Angle = MFX_ANGLE_180;
    vppMirror.Type = MFX_MIRRORING_HORIZONTAL;
    vppScaleQuality.ScalingMode = MFX_SCALING_MODE_LOWPOWER;
    vppMctf.FilterStrength = 0;

    vector<mfxExtBuffer*> buf;
    buf.push_back((mfxExtBuffer *)&vppDoUse);
    if (bSetDoNotUseTag) {
        buf.push_back((mfxExtBuffer *)&vppDoNotUse);
    }
    buf.push_back((mfxExtBuffer *)nullptr);

    mfxVideoParam videoPrm;
    RGY_MEMSET_ZERO(videoPrm);

    videoPrm.NumExtParam = (mfxU16)buf.size();
    videoPrm.ExtParam = (buf.size()) ? &buf[0] : NULL;
    videoPrm.AsyncDepth           = 3;
    videoPrm.IOPattern            = (bHardware) ? MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY : MFX_IOPATTERN_IN_SYSTEM_MEMORY | MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
    videoPrm.vpp.In.FrameRateExtN = 24000;
    videoPrm.vpp.In.FrameRateExtD = 1001;
    videoPrm.vpp.In.FourCC        = MFX_FOURCC_NV12;
    videoPrm.vpp.In.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    videoPrm.vpp.In.PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
    videoPrm.vpp.In.AspectRatioW  = 1;
    videoPrm.vpp.In.AspectRatioH  = 1;
    videoPrm.vpp.In.Width         = 1920;
    videoPrm.vpp.In.Height        = 1088;
    videoPrm.vpp.In.CropX         = 0;
    videoPrm.vpp.In.CropY         = 0;
    videoPrm.vpp.In.CropW         = 1920;
    videoPrm.vpp.In.CropH         = 1080;
    memcpy(&videoPrm.vpp.Out, &videoPrm.vpp.In, sizeof(videoPrm.vpp.In));
    videoPrm.vpp.Out.Width        = 1280;
    videoPrm.vpp.Out.Height       = 720;
    videoPrm.vpp.Out.CropW        = 1280;
    videoPrm.vpp.Out.CropH        = 720;

    mfxExtVPPDoUse vppDoUseOut;
    mfxExtVPPDoUse vppDoNotUseOut;
    mfxExtVPPFrameRateConversion vppFpsConvOut;
    mfxExtVPPImageStab vppImageStabOut;
    mfxExtVPPVideoSignalInfo vppVSIOut;
    mfxExtVPPRotation vppRotateOut;
    mfxExtVPPMirroring vppMirrorOut;
    mfxExtVPPScaling vppScaleQualityOut;
    mfxExtVppMctf vppMctfOut;
    mfxExtVPPDenoise2 vppDenoise2Out;

    memcpy(&vppDoUseOut,        &vppDoUse,        sizeof(vppDoUse));
    memcpy(&vppDoNotUseOut,     &vppDoNotUse,     sizeof(vppDoNotUse));
    memcpy(&vppFpsConvOut,      &vppFpsConv,      sizeof(vppFpsConv));
    memcpy(&vppImageStabOut,    &vppImageStab,    sizeof(vppImageStab));
    memcpy(&vppVSIOut,          &vppVSI,          sizeof(vppVSI));
    memcpy(&vppRotateOut,       &vppRotate,       sizeof(vppRotate));
    memcpy(&vppMirrorOut,       &vppMirror,       sizeof(vppMirror));
    memcpy(&vppScaleQualityOut, &vppScaleQuality, sizeof(vppScaleQuality));
    memcpy(&vppMctfOut,         &vppMctf,         sizeof(vppMctf));
    memcpy(&vppDenoise2Out,     &vppDenoise2,     sizeof(vppDenoise2));

    vector<mfxExtBuffer *> bufOut;
    bufOut.push_back((mfxExtBuffer *)&vppDoUse);
    if (bSetDoNotUseTag) {
        bufOut.push_back((mfxExtBuffer *)&vppDoNotUse);
    }
    bufOut.push_back((mfxExtBuffer *)nullptr);

    mfxVideoParam videoPrmOut;
    memcpy(&videoPrmOut, &videoPrm, sizeof(videoPrm));
    videoPrmOut.NumExtParam = (mfxU16)bufOut.size();
    videoPrmOut.ExtParam = (bufOut.size()) ? &bufOut[0] : NULL;

    static const mfxU32 vppList[] = {
        MFX_EXTBUFF_VPP_PROCAMP,
        MFX_EXTBUFF_VPP_DENOISE,
        MFX_EXTBUFF_VPP_DETAIL,
        MFX_EXTBUFF_VPP_AUXDATA
    };
    auto check_feature = [&](mfxExtBuffer *structIn, mfxExtBuffer *structOut, mfxVersion requiredVer, mfxU64 featureNoErr, mfxU64 featureWarn) {
        if (check_lib_version(mfxVer, requiredVer)) {
            const mfxU32 target = structIn->BufferId;
            //vppDoUseListとvppDoNotUseListを構築する
            vector<mfxU32> vppDoUseList;
            vector<mfxU32> vppDoNotUseList;
            vppDoUseList.push_back(target);
            for (int i = 0; i < _countof(vppList); i++)
                vppDoNotUseList.push_back(vppList[i]);
            //出力側に同じものをコピー
            vector<mfxU32> vppDoUseListOut(vppDoUseList.size());
            vector<mfxU32> vppDoNotUseListOut(vppDoNotUseList.size());
            copy(vppDoUseList.begin(), vppDoUseList.end(), vppDoUseListOut.begin());
            copy(vppDoNotUseList.begin(), vppDoNotUseList.end(), vppDoNotUseListOut.begin());
            //入力側の設定
            vppDoUse.NumAlg     = (mfxU32)vppDoUseList.size();
            vppDoUse.AlgList    = &vppDoUseList[0];
            vppDoNotUse.NumAlg  = (mfxU32)vppDoNotUseList.size();
            vppDoNotUse.AlgList = &vppDoNotUseList[0];
            //出力側の設定
            vppDoUseOut.NumAlg     = (mfxU32)vppDoUseListOut.size();
            vppDoUseOut.AlgList    = &vppDoUseListOut[0];
            vppDoNotUseOut.NumAlg  = (mfxU32)vppDoNotUseListOut.size();
            vppDoNotUseOut.AlgList = &vppDoNotUseListOut[0];
            //bufの一番端はチェック用に開けてあるので、そこに構造体へのポインタを入れる
            *(buf.end()    - 1) = (mfxExtBuffer *)structIn;
            *(bufOut.end() - 1) = (mfxExtBuffer *)structOut;
            mfxStatus ret = MFXVideoVPP_Query(session, &videoPrm, &videoPrmOut);
            if (ret >= MFX_ERR_NONE) {// QSV_WRN_XXX (> 0) は無視する
                result |= (MFX_ERR_NONE == ret || MFX_WRN_PARTIAL_ACCELERATION == ret) ? featureNoErr : featureWarn;
            }
        }
    };

    check_feature((mfxExtBuffer *)&vppImageStab,    (mfxExtBuffer *)&vppImageStabOut,    MFX_LIB_VERSION_1_6,  VPP_FEATURE_IMAGE_STABILIZATION, 0x00);
    check_feature((mfxExtBuffer *)&vppVSI,          (mfxExtBuffer *)&vppVSIOut,          MFX_LIB_VERSION_1_8,  VPP_FEATURE_VIDEO_SIGNAL_INFO,   0x00);
#if defined(_WIN32) || defined(_WIN64)
    check_feature((mfxExtBuffer *)&vppRotate,       (mfxExtBuffer *)&vppRotateOut,       MFX_LIB_VERSION_1_17, VPP_FEATURE_ROTATE,              0x00);
#endif //#if defined(_WIN32) || defined(_WIN64)
    check_feature((mfxExtBuffer *)&vppMirror,       (mfxExtBuffer *)&vppMirrorOut,       MFX_LIB_VERSION_1_19,  VPP_FEATURE_MIRROR,             0x00);
    check_feature((mfxExtBuffer *)&vppScaleQuality, (mfxExtBuffer *)&vppScaleQualityOut, MFX_LIB_VERSION_1_19,  VPP_FEATURE_SCALING_QUALITY,    0x00);
    check_feature((mfxExtBuffer *)&vppMctf,         (mfxExtBuffer *)&vppMctfOut,         MFX_LIB_VERSION_1_26,  VPP_FEATURE_MCTF,               0x00);
    check_feature((mfxExtBuffer *)&vppDenoise2,     (mfxExtBuffer *)&vppDenoise2Out,     MFX_LIB_VERSION_2_5,   VPP_FEATURE_DENOISE2,           0x00);

    videoPrm.vpp.Out.FrameRateExtN    = 60000;
    videoPrm.vpp.Out.FrameRateExtD    = 1001;
    videoPrmOut.vpp.Out.FrameRateExtN = 60000;
    videoPrmOut.vpp.Out.FrameRateExtD = 1001;
    check_feature((mfxExtBuffer *)&vppFpsConv,   (mfxExtBuffer *)&vppFpsConvOut,   MFX_LIB_VERSION_1_3,  VPP_FEATURE_FPS_CONVERSION_ADV,  VPP_FEATURE_FPS_CONVERSION);
    return result;
}

mfxU64 CheckVppFeatures(MFXVideoSession& session) {
    mfxU64 feature = 0x00;
    mfxVersion ver = MFX_LIB_VERSION_0_0;
    session.QueryVersion(&ver);
    if (!check_lib_version(ver, MFX_LIB_VERSION_1_3)) {
        //API v1.3未満で実際にチェックする必要は殆ど無いので、
        //コードで決められた値を返すようにする
        feature |= VPP_FEATURE_RESIZE;
        feature |= VPP_FEATURE_DEINTERLACE;
        feature |= VPP_FEATURE_DENOISE;
        feature |= VPP_FEATURE_DETAIL_ENHANCEMENT;
        feature |= VPP_FEATURE_PROC_AMP;
    } else {
        feature = CheckVppFeaturesInternal(session, ver);
    }

    return feature;
}

mfxU64 CheckVppFeatures(const QSVDeviceNum deviceNum, std::shared_ptr<RGYLog> log) {
    mfxU64 feature = 0x00;
    MemType memType = HW_MEMORY;
    std::unique_ptr<CQSVHWDevice> hwdev;
    MFXVideoSession2 session;
    MFXVideoSession2Params params;
    bool bexternalAlloc = true;
    std::unique_ptr<QSVAllocator> allocator;
    auto err = RGY_ERR_NONE;
    if ((err = InitSessionAndDevice(hwdev, session, memType, deviceNum, params, log)) != RGY_ERR_NONE) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_DEV, _T("InitSessionAndDevice: failed to initialize: %s.\n"), get_err_mes(err));
    } else if ((err = CreateAllocator(allocator, bexternalAlloc, memType, hwdev.get(), session, log)) != RGY_ERR_NONE) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_DEV, _T("CreateAllocator: failed to create allocator: %s.\n"), get_err_mes(err));
    } else {
        mfxVersion ver = MFX_LIB_VERSION_0_0;
        session.QueryVersion(&ver);
        if (!check_lib_version(ver, MFX_LIB_VERSION_1_3)) {
            //API v1.3未満で実際にチェックする必要は殆ど無いので、
            //コードで決められた値を返すようにする
            feature |= VPP_FEATURE_RESIZE;
            feature |= VPP_FEATURE_DEINTERLACE;
            feature |= VPP_FEATURE_DENOISE;
            feature |= VPP_FEATURE_DETAIL_ENHANCEMENT;
            feature |= VPP_FEATURE_PROC_AMP;
        } else {
            log->write(RGY_LOG_DEBUG, RGY_LOGT_DEV, _T("InitSession: initialized allocator.\n"));
            feature = CheckVppFeaturesInternal(session, ver);
        }
    }

    return feature;
}

uint64_t CheckEncodeFeature(MFXVideoSession& session, const int ratecontrol, const RGY_CODEC codec, const bool lowPower) {
    const mfxU32 codecId = codec_rgy_to_enc(codec);
    mfxVersion mfxVer;
    session.QueryVersion(&mfxVer);
    if (codecId == MFX_CODEC_HEVC && !check_lib_version(mfxVer, MFX_LIB_VERSION_1_15)) {
        return 0x00;
    }
    if (codecId == MFX_CODEC_AV1 && !check_lib_version(mfxVer, MFX_LIB_VERSION_2_5)) {
        return 0x00;
    }
    const std::vector<std::pair<int, mfxVersion>> rc_list = {
        { MFX_RATECONTROL_VBR,    MFX_LIB_VERSION_1_1  },
        { MFX_RATECONTROL_CBR,    MFX_LIB_VERSION_1_1  },
        { MFX_RATECONTROL_CQP,    MFX_LIB_VERSION_1_1  },
        { MFX_RATECONTROL_AVBR,   MFX_LIB_VERSION_1_3  },
        { MFX_RATECONTROL_LA,     MFX_LIB_VERSION_1_7  },
        { MFX_RATECONTROL_LA_ICQ, MFX_LIB_VERSION_1_8  },
        { MFX_RATECONTROL_VCM,    MFX_LIB_VERSION_1_8  },
        //{ MFX_RATECONTROL_LA_EXT, MFX_LIB_VERSION_1_11 },
        { MFX_RATECONTROL_LA_HRD, MFX_LIB_VERSION_1_11 },
        { MFX_RATECONTROL_QVBR,   MFX_LIB_VERSION_1_11 },
    };
    for (auto rc : rc_list) {
        if ((mfxU16)ratecontrol == rc.first) {
            if (!check_lib_version(mfxVer, rc.second)) {
                return 0x00;
            }
            break;
        }
    }

    static const auto HYPER_MODE_ENABLED_CODECS = make_array<RGY_CODEC>(
        RGY_CODEC_H264, RGY_CODEC_HEVC, RGY_CODEC_AV1
    );

    mfxExtCodingOption cop;
    mfxExtCodingOption2 cop2;
    mfxExtCodingOption3 cop3;
    mfxExtHEVCParam hevc;
    mfxExtVP9Param vp9;
    mfxExtAV1BitstreamParam av1;
    mfxExtHyperModeParam hyperMode;
    mfxExtVideoSignalInfo videoSignalInfo;
    INIT_MFX_EXT_BUFFER(cop,  MFX_EXTBUFF_CODING_OPTION);
    INIT_MFX_EXT_BUFFER(cop2, MFX_EXTBUFF_CODING_OPTION2);
    INIT_MFX_EXT_BUFFER(cop3, MFX_EXTBUFF_CODING_OPTION3);
    INIT_MFX_EXT_BUFFER(hevc, MFX_EXTBUFF_HEVC_PARAM);
    INIT_MFX_EXT_BUFFER(vp9, MFX_EXTBUFF_VP9_PARAM);
    INIT_MFX_EXT_BUFFER(av1, MFX_EXTBUFF_AV1_BITSTREAM_PARAM);
    INIT_MFX_EXT_BUFFER(hyperMode, MFX_EXTBUFF_HYPER_MODE_PARAM);
    INIT_MFX_EXT_BUFFER(videoSignalInfo, MFX_EXTBUFF_VIDEO_SIGNAL_INFO);

    std::vector<mfxExtBuffer *> buf;
    if (add_cop(codecId)) { // VP9ではmfxExtCodingOptionはチェックしないようにしないと正常に動作しない
        buf.push_back((mfxExtBuffer *)&cop);
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_3)
        && add_vui(codecId)) {
        buf.push_back((mfxExtBuffer*)&videoSignalInfo);
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_6)) {
        buf.push_back((mfxExtBuffer *)&cop2);
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_11)) {
        buf.push_back((mfxExtBuffer *)&cop3);
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_15)
        && codecId == MFX_CODEC_HEVC) {
        buf.push_back((mfxExtBuffer *)&hevc);
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_26)
        && codecId == MFX_CODEC_VP9) {
        buf.push_back((mfxExtBuffer *)&vp9);
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_2_5)
        && codecId == MFX_CODEC_AV1) {
        buf.push_back((mfxExtBuffer *)&av1);
    }
    if (ENABLE_HYPER_MODE && check_lib_version(mfxVer, MFX_LIB_VERSION_2_5)
        && (codecId == MFX_CODEC_AVC || codecId == MFX_CODEC_HEVC || codecId == MFX_CODEC_AV1)) {
        buf.push_back((mfxExtBuffer *)&hyperMode);
    }

    mfxVideoParam videoPrm;
    RGY_MEMSET_ZERO(videoPrm);

    auto set_default_quality_prm = [&videoPrm]() {
        if (   videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_VBR
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_AVBR
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_CBR
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_LA
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_LA_HRD
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_VCM
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_QVBR) {
            videoPrm.mfx.TargetKbps = 3000;
            videoPrm.mfx.MaxKbps    = 3000; //videoPrm.mfx.MaxKbpsはvideoPrm.mfx.TargetKbpsと一致させないとCBRの時に失敗する
            if (videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_AVBR) {
                videoPrm.mfx.Accuracy     = 500;
                videoPrm.mfx.Convergence  = 90;
            }
        } else if (videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_CQP) {
            videoPrm.mfx.QPI = 23;
            videoPrm.mfx.QPP = 23;
            videoPrm.mfx.QPB = 23;
        } else {
            //MFX_RATECONTROL_ICQ
            //MFX_RATECONTROL_LA_ICQ
            videoPrm.mfx.ICQQuality = 23;
        }
    };

    const auto lowPowerMode = (decltype(videoPrm.mfx.LowPower))((lowPower) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF);

    videoPrm.NumExtParam = (mfxU16)buf.size();
    videoPrm.ExtParam = (buf.size()) ? &buf[0] : NULL;
    videoPrm.AsyncDepth                  = 3;
    videoPrm.IOPattern                   = MFX_IOPATTERN_IN_SYSTEM_MEMORY;
    videoPrm.mfx.CodecId                 = codecId;
    videoPrm.mfx.RateControlMethod       = (mfxU16)ratecontrol;
    videoPrm.mfx.TargetUsage             = MFX_TARGETUSAGE_BALANCED;
    videoPrm.mfx.EncodedOrder            = 0;
    videoPrm.mfx.NumSlice                = 1;
    videoPrm.mfx.NumRefFrame             = 3;
    videoPrm.mfx.GopPicSize              = 30;
    videoPrm.mfx.IdrInterval             = 0;
    videoPrm.mfx.GopOptFlag              = 0;
    videoPrm.mfx.GopRefDist              = 1;
    videoPrm.mfx.FrameInfo.FrameRateExtN = 30000;
    videoPrm.mfx.FrameInfo.FrameRateExtD = 1001;
    videoPrm.mfx.FrameInfo.FourCC        = MFX_FOURCC_NV12;
    videoPrm.mfx.FrameInfo.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    videoPrm.mfx.FrameInfo.PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
    videoPrm.mfx.FrameInfo.AspectRatioW  = 1;
    videoPrm.mfx.FrameInfo.AspectRatioH  = 1;
    videoPrm.mfx.FrameInfo.Width         = 1920;
    videoPrm.mfx.FrameInfo.Height        = 1088;
    videoPrm.mfx.FrameInfo.CropX         = 0;
    videoPrm.mfx.FrameInfo.CropY         = 0;
    videoPrm.mfx.FrameInfo.CropW         = 1920;
    videoPrm.mfx.FrameInfo.CropH         = 1080;
    videoPrm.mfx.LowPower                = lowPowerMode;
    switch (codecId) {
    case MFX_CODEC_HEVC:
        videoPrm.mfx.CodecLevel = MFX_LEVEL_UNKNOWN;
        videoPrm.mfx.CodecProfile = MFX_PROFILE_HEVC_MAIN;
        break;
    case MFX_CODEC_VP8:
        break;
    case MFX_CODEC_VP9:
        videoPrm.mfx.CodecLevel = MFX_LEVEL_UNKNOWN;
        videoPrm.mfx.CodecProfile = MFX_PROFILE_VP9_0;
        videoPrm.mfx.GopRefDist = 1;
        videoPrm.mfx.NumRefFrame = 1;
        //videoPrm.mfx.GopPicSize = 65535;
        //videoPrm.mfx.NumThread = 1;
        //videoPrm.AsyncDepth = 4;
        //videoPrm.mfx.BRCParamMultiplier = 1;
        //videoPrm.mfx.FrameInfo.BufferSize=47187200;
        videoPrm.mfx.Interleaved = 0;
        //なぜか1280x720でqueryしないと成功しない
        videoPrm.mfx.FrameInfo.Width         = 1280;
        videoPrm.mfx.FrameInfo.Height        = 720;
        videoPrm.mfx.FrameInfo.CropW         = 1280;
        videoPrm.mfx.FrameInfo.CropH         = 720;
        if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_9)) {
            videoPrm.mfx.FrameInfo.BitDepthLuma = 8;
            videoPrm.mfx.FrameInfo.BitDepthChroma = 8;
        }
        //videoPrm.mfx.LowPower = MFX_CODINGOPTION_ON;
        break;
    case MFX_CODEC_AV1:
        videoPrm.mfx.CodecLevel = MFX_LEVEL_UNKNOWN;
        videoPrm.mfx.CodecProfile = MFX_PROFILE_AV1_MAIN;
        videoPrm.mfx.FrameInfo.Width = 1280;
        videoPrm.mfx.FrameInfo.Height = 720;
        videoPrm.mfx.FrameInfo.CropW = 1280;
        videoPrm.mfx.FrameInfo.CropH = 720;
        videoPrm.mfx.GopRefDist = 1;
        av1.WriteIVFHeaders = MFX_CODINGOPTION_OFF;
        if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_9)) {
            videoPrm.mfx.FrameInfo.BitDepthLuma = 8;
            videoPrm.mfx.FrameInfo.BitDepthChroma = 8;
        }
        break;
    default:
    case MFX_CODEC_AVC:
        videoPrm.mfx.CodecLevel = MFX_LEVEL_UNKNOWN;
        videoPrm.mfx.CodecProfile = MFX_PROFILE_AVC_HIGH;
        break;
    }
    set_default_quality_prm();

    auto ret = MFXVideoENCODE_Query(session, &videoPrm, &videoPrm);
    //_ftprintf(stderr, _T("error checking %s: %s\n"), CodecIdToStr(codecId), get_err_mes(err_to_rgy(ret)));

    uint64_t result = (ret >= MFX_ERR_NONE && videoPrm.mfx.RateControlMethod == ratecontrol && videoPrm.mfx.LowPower == lowPowerMode) ? ENC_FEATURE_CURRENT_RC : 0x00;
    if (result) {

        //まず、エンコードモードについてチェック
        auto check_enc_mode = [&](mfxU16 mode, mfxU64 flag, mfxVersion required_ver) {
            if (check_lib_version(mfxVer, required_ver)) {
                mfxU16 original_method = videoPrm.mfx.RateControlMethod;
                videoPrm.mfx.RateControlMethod = mode;
                set_default_quality_prm();
                if (MFXVideoENCODE_Query(session, &videoPrm, &videoPrm) >= MFX_ERR_NONE && videoPrm.mfx.RateControlMethod == ratecontrol)
                    result |= flag;
                videoPrm.mfx.RateControlMethod = original_method;
                set_default_quality_prm();
            }
        };
        check_enc_mode(MFX_RATECONTROL_AVBR,   ENC_FEATURE_AVBR,   MFX_LIB_VERSION_1_3);
        check_enc_mode(MFX_RATECONTROL_LA,     ENC_FEATURE_LA,     MFX_LIB_VERSION_1_7);
        check_enc_mode(MFX_RATECONTROL_ICQ,    ENC_FEATURE_ICQ,    MFX_LIB_VERSION_1_8);
        check_enc_mode(MFX_RATECONTROL_VCM,    ENC_FEATURE_VCM,    MFX_LIB_VERSION_1_8);
        check_enc_mode(MFX_RATECONTROL_LA_HRD, ENC_FEATURE_LA_HRD, MFX_LIB_VERSION_1_11);
        check_enc_mode(MFX_RATECONTROL_QVBR,   ENC_FEATURE_QVBR,   MFX_LIB_VERSION_1_11);

#define CHECK_FEATURE(membersIn, flag, value, required_ver) { \
        if (check_lib_version(mfxVer, (required_ver))) { \
            const decltype(membersIn) orig = (membersIn); \
            (membersIn) = (value); \
            auto check_ret = MFXVideoENCODE_Query(session, &videoPrm, &videoPrm); \
            if (check_ret >= MFX_ERR_NONE \
                && (membersIn) == (value) \
                && videoPrm.mfx.RateControlMethod == ratecontrol) { \
                result |= (flag); \
            } else if (false) { \
                _ftprintf(stderr, _T("error checking %s %s " # flag ": %s\n"), CodecIdToStr(codecId), EncmodeToStr(ratecontrol), get_err_mes(err_to_rgy(check_ret))); \
            } \
            (membersIn) = orig; \
        } \
    }
        if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_3) && add_vui(codecId)) {
            if (true) {
                //これはもう単純にAPIチェックでOK
                result |= ENC_FEATURE_VUI_INFO;
            } else {
                videoSignalInfo.ColourDescriptionPresent = 1; //"1"と設定しないと正しく反映されない
                CHECK_FEATURE(videoSignalInfo.MatrixCoefficients, ENC_FEATURE_VUI_INFO, (decltype(videoSignalInfo.MatrixCoefficients))RGY_MATRIX_BT709, MFX_LIB_VERSION_1_3);
                videoSignalInfo.ColourDescriptionPresent = 0;
            }
        }
        //とりあえずAV1ではBフレームのチェックはしない
        //ひとつひとつパラメータを入れ替えて試していく
#pragma warning(push)
#pragma warning(disable:4244) //'mfxU16' から 'mfxU8' への変換です。データが失われる可能性があります。
#define PICTYPE mfx.FrameInfo.PicStruct
        const mfxU32 MFX_TRELLIS_ALL = MFX_TRELLIS_I | MFX_TRELLIS_P | MFX_TRELLIS_B;
        CHECK_FEATURE(videoPrm.PICTYPE,          ENC_FEATURE_INTERLACE,     MFX_PICSTRUCT_FIELD_TFF, MFX_LIB_VERSION_1_1);
        if (add_cop(codecId)) { // VP9ではmfxExtCodingOptionはチェックしない
            CHECK_FEATURE(cop.AUDelimiter,           ENC_FEATURE_AUD,           MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_1);
            CHECK_FEATURE(cop.PicTimingSEI,          ENC_FEATURE_PIC_STRUCT,    MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_1);
            CHECK_FEATURE(cop.RateDistortionOpt,     ENC_FEATURE_RDO,           MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_1);
            CHECK_FEATURE(cop.CAVLC,                 ENC_FEATURE_CAVLC,         MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_1);
        }
        CHECK_FEATURE(videoPrm.mfx.GopRefDist,   ENC_FEATURE_GOPREFDIST,    4,                       MFX_LIB_VERSION_1_1);
        CHECK_FEATURE(cop2.ExtBRC,               ENC_FEATURE_EXT_BRC,       MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_6);
        CHECK_FEATURE(cop2.MBBRC,                ENC_FEATURE_MBBRC,         MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_6);
        CHECK_FEATURE(cop2.Trellis,              ENC_FEATURE_TRELLIS,       MFX_TRELLIS_ALL,         MFX_LIB_VERSION_1_7);
        cop2.IntRefCycleSize = 16;
        CHECK_FEATURE(cop2.IntRefType,           ENC_FEATURE_INTRA_REFRESH, 1,                       MFX_LIB_VERSION_1_7);
        cop2.IntRefCycleSize = 0;
        CHECK_FEATURE(cop2.AdaptiveI,            ENC_FEATURE_ADAPTIVE_I,    MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_8);
        const auto bframesCheck = ((result & ENC_FEATURE_GOPREFDIST) != 0);
        if (bframesCheck) {
            CHECK_FEATURE(cop2.AdaptiveB, ENC_FEATURE_ADAPTIVE_B, MFX_CODINGOPTION_ON, MFX_LIB_VERSION_1_8);
            const auto orig_ref_dist = videoPrm.mfx.GopRefDist;
            videoPrm.mfx.GopRefDist = 4;
            CHECK_FEATURE(cop2.BRefType, ENC_FEATURE_B_PYRAMID, MFX_B_REF_PYRAMID, MFX_LIB_VERSION_1_8);
            videoPrm.mfx.GopRefDist = orig_ref_dist;
            CHECK_FEATURE(cop3.WeightedBiPred, ENC_FEATURE_WEIGHT_B, MFX_WEIGHTED_PRED_DEFAULT, MFX_LIB_VERSION_1_16);
        }
        CHECK_FEATURE(cop3.WeightedPred, ENC_FEATURE_WEIGHT_P, MFX_WEIGHTED_PRED_DEFAULT, MFX_LIB_VERSION_1_16);
        if (rc_is_type_lookahead(ratecontrol)) {
            CHECK_FEATURE(cop2.LookAheadDS,      ENC_FEATURE_LA_DS,         MFX_LOOKAHEAD_DS_2x,     MFX_LIB_VERSION_1_8);
        }
        CHECK_FEATURE(cop2.DisableDeblockingIdc, ENC_FEATURE_NO_DEBLOCK,    MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_9);
        CHECK_FEATURE(cop2.MaxQPI,               ENC_FEATURE_QP_MINMAX,     48,                      MFX_LIB_VERSION_1_9);
        cop3.WinBRCMaxAvgKbps = 3000;
        CHECK_FEATURE(cop3.WinBRCSize,           ENC_FEATURE_WINBRC,        10,                      MFX_LIB_VERSION_1_11);
        cop3.WinBRCMaxAvgKbps = 0;
        CHECK_FEATURE(cop3.EnableMBQP,                 ENC_FEATURE_PERMBQP,                    MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_13);
        CHECK_FEATURE(cop3.DirectBiasAdjustment,       ENC_FEATURE_DIRECT_BIAS_ADJUST,         MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_13);
        CHECK_FEATURE(cop3.GlobalMotionBiasAdjustment, ENC_FEATURE_GLOBAL_MOTION_ADJUST,       MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_13);
        if (ENABLE_HYPER_MODE) {
            if (!LIMIT_HYPER_MODE_TO_KNOWN_CODECS
                || std::find(HYPER_MODE_ENABLED_CODECS.begin(), HYPER_MODE_ENABLED_CODECS.end(), codec) != HYPER_MODE_ENABLED_CODECS.end()) {
                CHECK_FEATURE(hyperMode.Mode, ENC_FEATURE_HYPER_MODE, MFX_HYPERMODE_ON, MFX_LIB_VERSION_2_5);
            }
        }
        CHECK_FEATURE(cop3.FadeDetection,        ENC_FEATURE_FADE_DETECT,   MFX_CODINGOPTION_ON,           MFX_LIB_VERSION_1_17);
        CHECK_FEATURE(cop3.AdaptiveLTR,          ENC_FEATURE_ADAPTIVE_LTR,  MFX_CODINGOPTION_ON,           MFX_LIB_VERSION_2_4);
        CHECK_FEATURE(cop3.AdaptiveRef,          ENC_FEATURE_ADAPTIVE_REF,  MFX_CODINGOPTION_ON,           MFX_LIB_VERSION_2_4);
        CHECK_FEATURE(cop3.AdaptiveCQM,          ENC_FEATURE_ADAPTIVE_CQM,  MFX_CODINGOPTION_ON,           MFX_LIB_VERSION_2_2);
        if (codecId == MFX_CODEC_HEVC) {
            CHECK_FEATURE(cop3.GPB,              ENC_FEATURE_DISABLE_GPB,       MFX_CODINGOPTION_ON,  MFX_LIB_VERSION_1_19);
            CHECK_FEATURE(cop3.EnableQPOffset,   ENC_FEATURE_PYRAMID_QP_OFFSET, MFX_CODINGOPTION_ON,  MFX_LIB_VERSION_1_19);
            videoPrm.mfx.FrameInfo.BitDepthLuma = 10;
            videoPrm.mfx.FrameInfo.BitDepthChroma = 10;
            videoPrm.mfx.FrameInfo.Shift = 1;
            videoPrm.mfx.CodecProfile = MFX_PROFILE_HEVC_MAIN10;
            CHECK_FEATURE(videoPrm.mfx.FrameInfo.FourCC, ENC_FEATURE_10BIT_DEPTH, MFX_FOURCC_P010, MFX_LIB_VERSION_1_19);
            videoPrm.mfx.FrameInfo.BitDepthLuma = 0;
            videoPrm.mfx.FrameInfo.BitDepthChroma = 0;
            videoPrm.mfx.FrameInfo.Shift = 0;
            videoPrm.mfx.CodecProfile = MFX_PROFILE_HEVC_MAIN;
            CHECK_FEATURE(cop3.TransformSkip, ENC_FEATURE_HEVC_TSKIP, MFX_CODINGOPTION_ON, MFX_LIB_VERSION_1_26);
            CHECK_FEATURE(hevc.SampleAdaptiveOffset, ENC_FEATURE_HEVC_SAO, MFX_SAO_ENABLE_LUMA, MFX_LIB_VERSION_1_26);
            CHECK_FEATURE(hevc.LCUSize, ENC_FEATURE_HEVC_CTU, 32, MFX_LIB_VERSION_1_26);
        } else if (codecId == MFX_CODEC_VP9) {
            videoPrm.mfx.FrameInfo.BitDepthLuma = 10;
            videoPrm.mfx.FrameInfo.BitDepthChroma = 10;
            videoPrm.mfx.FrameInfo.Shift = 1;
            videoPrm.mfx.CodecProfile = MFX_PROFILE_VP9_2;
            CHECK_FEATURE(videoPrm.mfx.FrameInfo.FourCC, ENC_FEATURE_10BIT_DEPTH, MFX_FOURCC_P010, MFX_LIB_VERSION_1_19);
            videoPrm.mfx.FrameInfo.BitDepthLuma = 0;
            videoPrm.mfx.FrameInfo.BitDepthChroma = 0;
            videoPrm.mfx.FrameInfo.Shift = 0;
            videoPrm.mfx.CodecProfile = MFX_PROFILE_VP9_0;
        } else if (codecId == MFX_CODEC_AV1) {
            videoPrm.mfx.FrameInfo.BitDepthLuma = 10;
            videoPrm.mfx.FrameInfo.BitDepthChroma = 10;
            videoPrm.mfx.FrameInfo.Shift = 1;
            CHECK_FEATURE(videoPrm.mfx.FrameInfo.FourCC, ENC_FEATURE_10BIT_DEPTH, MFX_FOURCC_P010, MFX_LIB_VERSION_1_19);
            videoPrm.mfx.FrameInfo.BitDepthLuma = 8;
            videoPrm.mfx.FrameInfo.BitDepthChroma = 8;
            videoPrm.mfx.FrameInfo.Shift = 0;
        }
#undef PICTYPE
#pragma warning(pop)
        //付随オプション
        if (result & ENC_FEATURE_B_PYRAMID) {
            result |= ENC_FEATURE_B_PYRAMID_MANY_BFRAMES;
        }
        //以下特殊な場合
        if (rc_is_type_lookahead(ratecontrol)) {
            result &= ~ENC_FEATURE_RDO;
            result &= ~ENC_FEATURE_MBBRC;
            result &= ~ENC_FEATURE_EXT_BRC;
            if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_8)) {
                //API v1.8以降、LA + 多すぎるBフレームは不安定(フリーズ)
                result &= ~ENC_FEATURE_B_PYRAMID_MANY_BFRAMES;
            }
        } else if (MFX_RATECONTROL_CQP == ratecontrol) {
            result &= ~ENC_FEATURE_MBBRC;
            result &= ~ENC_FEATURE_EXT_BRC;
        }
        //Kabylake以前では、不安定でエンコードが途中で終了あるいはフリーズしてしまう
        const auto cpu_gen = getCPUGen(&session);
        if ((result & ENC_FEATURE_FADE_DETECT) && cpu_gen < CPU_GEN_KABYLAKE) {
            result &= ~ENC_FEATURE_FADE_DETECT;
        }
        //Kabylake以降では、10bit depthに対応しているはずだが、これが正常に判定されないことがある
        if (codecId == MFX_CODEC_HEVC && cpu_gen >= CPU_GEN_KABYLAKE) {
            result |= ENC_FEATURE_10BIT_DEPTH;
        }
    }
#undef CHECK_FEATURE
    return result;
}

//サポートする機能のチェックをAPIバージョンのみで行う
//API v1.6以降はCheckEncodeFeatureを使うべき
//同一のAPIバージョンでも環境により異なることが多くなるため
static uint64_t CheckEncodeFeatureStatic(const mfxVersion mfxVer, const int ratecontrol, const RGY_CODEC codec) {
    const mfxU32 codecId = codec_rgy_to_enc(codec);
    uint64_t feature = 0x00;
    if (codecId != MFX_CODEC_AVC && codecId != MFX_CODEC_MPEG2) {
        return feature;
    }
    //まずレート制御モードをチェック
    BOOL rate_control_supported = false;
    switch (ratecontrol) {
    case MFX_RATECONTROL_CBR:
    case MFX_RATECONTROL_VBR:
    case MFX_RATECONTROL_CQP:
        rate_control_supported = true;
        break;
    case MFX_RATECONTROL_AVBR:
        rate_control_supported = check_lib_version(mfxVer, MFX_LIB_VERSION_1_3);
        break;
    case MFX_RATECONTROL_LA:
        rate_control_supported = check_lib_version(mfxVer, MFX_LIB_VERSION_1_7);
        break;
    case MFX_RATECONTROL_ICQ:
    case MFX_RATECONTROL_LA_ICQ:
    case MFX_RATECONTROL_VCM:
        rate_control_supported = check_lib_version(mfxVer, MFX_LIB_VERSION_1_8);
        break;
    case MFX_RATECONTROL_LA_HRD:
    case MFX_RATECONTROL_QVBR:
        rate_control_supported = check_lib_version(mfxVer, MFX_LIB_VERSION_1_11);
        break;
    default:
        break;
    }
    if (!rate_control_supported) {
        return feature;
    }

    //各モードをチェック
    feature |= ENC_FEATURE_CURRENT_RC;

    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_1)) {
        feature |= ENC_FEATURE_AUD;
        feature |= ENC_FEATURE_PIC_STRUCT;
        feature |= ENC_FEATURE_RDO;
        feature |= ENC_FEATURE_CAVLC;
        feature |= ENC_FEATURE_INTERLACE;
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_3)) {
        feature |= ENC_FEATURE_VUI_INFO;
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_6)) {
        feature |= ENC_FEATURE_EXT_BRC;
        feature |= ENC_FEATURE_MBBRC;
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_7)) {
        feature |= ENC_FEATURE_TRELLIS;
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_8)) {
        feature |= ENC_FEATURE_ADAPTIVE_I;
        feature |= ENC_FEATURE_ADAPTIVE_B;
        feature |= ENC_FEATURE_B_PYRAMID;
        feature |= ENC_FEATURE_B_PYRAMID_MANY_BFRAMES;
        feature |= ENC_FEATURE_VUI_INFO;
        if (rc_is_type_lookahead(ratecontrol)) {
            feature |= ENC_FEATURE_LA_DS;
            feature &= ~ENC_FEATURE_B_PYRAMID_MANY_BFRAMES;
        }
    }

    //以下特殊な場合の制限
    if (rc_is_type_lookahead(ratecontrol)) {
        feature &= ~ENC_FEATURE_RDO;
        feature &= ~ENC_FEATURE_MBBRC;
        feature &= ~ENC_FEATURE_EXT_BRC;
    } else if (MFX_RATECONTROL_CQP == ratecontrol) {
        feature &= ~ENC_FEATURE_MBBRC;
        feature &= ~ENC_FEATURE_EXT_BRC;
    }

    return feature;
}

uint64_t CheckEncodeFeatureWithPluginLoad(MFXVideoSession& session, const int ratecontrol, const RGY_CODEC codec, const bool lowPower) {
    uint64_t feature = 0x00;
    mfxVersion ver = MFX_LIB_VERSION_0_0;
    session.QueryVersion(&ver);
    if (!check_lib_version(ver, MFX_LIB_VERSION_1_0)) {
        ; //特にすることはない
    } else if (lowPower && !check_lib_version(ver, MFX_LIB_VERSION_1_15)) {
        ; // lowepowerはAPI 1.15以降の対応
        ; //特にすることはない
    } else if (!check_lib_version(ver, MFX_LIB_VERSION_1_6)) {
        //API v1.6未満で実際にチェックする必要は殆ど無いので、
        //コードで決められた値を返すようにする
        feature = CheckEncodeFeatureStatic(ver, ratecontrol, codec);
    } else {
        feature = CheckEncodeFeature(session, ratecontrol, codec, lowPower);
    }

    return feature;
}

const TCHAR *EncFeatureStr(mfxU64 enc_feature) {
    for (const FEATURE_DESC *ptr = list_enc_feature; ptr->desc; ptr++)
        if (enc_feature == (mfxU64)ptr->value)
            return ptr->desc;
    return NULL;
}

QSVEncFeatureData MakeFeatureList(const QSVDeviceNum deviceNum, const std::vector<CX_DESC>& rateControlList, const RGY_CODEC codec, const bool lowPower, std::shared_ptr<RGYLog> log) {
    QSVEncFeatureData availableFeatureForEachRC;
    availableFeatureForEachRC.codec = codec;
    availableFeatureForEachRC.dev = deviceNum;
    availableFeatureForEachRC.lowPwer = lowPower;
#if LIBVA_SUPPORT
    if (codec != RGY_CODEC_MPEG2) {
#endif
        MemType memType = HW_MEMORY;
        std::unique_ptr<CQSVHWDevice> hwdev;
        MFXVideoSession2 session;
        MFXVideoSession2Params params;
        bool bexternalAlloc = true;
        std::unique_ptr<QSVAllocator> allocator;
        auto err = RGY_ERR_NONE;
        if ((err = InitSessionAndDevice(hwdev, session, memType, deviceNum, params, log)) != RGY_ERR_NONE) {
            log->write(RGY_LOG_ERROR, RGY_LOGT_DEV, _T("InitSessionAndDevice: failed to initialize: %s.\n"), get_err_mes(err));
        } else if ((err = CreateAllocator(allocator, bexternalAlloc, memType, hwdev.get(), session, log)) != RGY_ERR_NONE) {
            log->write(RGY_LOG_ERROR, RGY_LOGT_DEV, _T("CreateAllocator: failed to create allocator: %s.\n"), get_err_mes(err));
        } else {
            mfxVersion ver = MFX_LIB_VERSION_0_0;
            session.QueryVersion(&ver);
            log->write(RGY_LOG_DEBUG, RGY_LOGT_DEV, _T("InitSession: initialized allocator.\n"));
            for (const auto& ratecontrol : rateControlList) {
                const uint64_t ret = CheckEncodeFeatureWithPluginLoad(session, (mfxU16)ratecontrol.value, codec, lowPower);
                if (ret == 0 && ratecontrol.value == MFX_RATECONTROL_CQP) {
                    ver = MFX_LIB_VERSION_0_0;
                }
                availableFeatureForEachRC.feature[ratecontrol.value] = ret;
            }
        }
#if LIBVA_SUPPORT
    }
#endif
    return availableFeatureForEachRC;
}

std::vector<QSVEncFeatureData> MakeFeatureListPerCodec(const QSVDeviceNum deviceNum, const vector<CX_DESC>& rateControlList, const vector<RGY_CODEC>& codecIdList, std::shared_ptr<RGYLog> log) {
    std::vector<QSVEncFeatureData> codecFeatures;
    vector<std::future<QSVEncFeatureData>> futures;
    if (true) {
        for (auto codec : codecIdList) {
            auto f0 = std::async(MakeFeatureList, deviceNum, rateControlList, codec, false, log);
            futures.push_back(std::move(f0));
            auto f1 = std::async(MakeFeatureList, deviceNum, rateControlList, codec, true, log);
            futures.push_back(std::move(f1));
        }
        for (size_t i = 0; i < futures.size(); i++) {
            codecFeatures.push_back(futures[i].get());
        }
    } else {
        for (auto codec : codecIdList) {
            codecFeatures.push_back(MakeFeatureList(deviceNum, rateControlList, codec, false, log));
            codecFeatures.push_back(MakeFeatureList(deviceNum, rateControlList, codec, true, log));
        }
    }
    // HEVCのhyper modeのチェックは使用できる場合でもなぜか成功しない
    // 原因不明だが、まずはH.264の結果を参照するようにする
    if (ENABLE_HYPER_MODE && OVERRIDE_HYPER_MODE_HEVC_FROM_H264) {
        for (auto& lowPower : { false, true }) {
            const auto it_h264 = std::find_if(codecFeatures.begin(), codecFeatures.end(), [lowPower](const QSVEncFeatureData& feature) {
                return feature.codec == RGY_CODEC_H264 && feature.lowPwer == lowPower;
            });
            auto it_hevc = std::find_if(codecFeatures.begin(), codecFeatures.end(), [lowPower](const QSVEncFeatureData& feature) {
                return feature.codec == RGY_CODEC_HEVC && feature.lowPwer == lowPower;
            });
            if (it_h264 != codecFeatures.end()
                && it_hevc != codecFeatures.end()
                && it_hevc->available()) {
                for (const auto& [rc, feature] : it_hevc->feature) {
                    if ((feature & ENC_FEATURE_HYPER_MODE) == 0) { // HEVCのHyperModeがオフで
                        if ((it_h264->feature[rc] & ENC_FEATURE_HYPER_MODE) != 0) { //H.264のHyperModeは有効なら
                            it_hevc->feature[rc] |= ENC_FEATURE_HYPER_MODE;
                        }
                    }
                }
            }
        }
    }
    return codecFeatures;
}

std::vector<RGY_CSP> CheckDecodeFeature(MFXVideoSession& session, mfxVersion ver, mfxU32 codecId) {
    std::vector<RGY_CSP> supportedCsp;
    switch (codecId) {
    case MFX_CODEC_HEVC:
        if (!check_lib_version(ver, MFX_LIB_VERSION_1_8)) {
            return supportedCsp;
        }
        break;
    case MFX_CODEC_VP8:
    case MFX_CODEC_VP9:
    case MFX_CODEC_JPEG:
        if (!check_lib_version(ver, MFX_LIB_VERSION_1_13)) {
            return supportedCsp;
        }
        break;
    case MFX_CODEC_AV1:
        if (!check_lib_version(ver, MFX_LIB_VERSION_1_34)) {
            return supportedCsp;
        }
        break;
    default:
        break;
    }

    return CheckDecFeaturesInternal(session, ver, codecId);
}

CodecCsp MakeDecodeFeatureList(MFXVideoSession& session, const vector<RGY_CODEC>& codecIdList, std::shared_ptr<RGYLog> log, const bool skipHWDecodeCheck) {
    CodecCsp codecFeatures;
    mfxVersion ver = MFX_LIB_VERSION_0_0;
    session.QueryVersion(&ver);
    for (auto codec : codecIdList) {
        if (skipHWDecodeCheck) {
            codecFeatures[codec] = {
                RGY_CSP_NV12, RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16,
                RGY_CSP_YUV422, RGY_CSP_YUV422_09, RGY_CSP_YUV422_10, RGY_CSP_YUV422_12, RGY_CSP_YUV422_14, RGY_CSP_YUV422_16,
                RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16
            };
        } else {
            auto features = CheckDecodeFeature(session, ver, codec_rgy_to_enc(codec));
            if (features.size() > 0) {
                codecFeatures[codec] = features;
            }
        }
    }
    return codecFeatures;
}

CodecCsp MakeDecodeFeatureList(const QSVDeviceNum deviceNum, const vector<RGY_CODEC>& codecIdList, std::shared_ptr<RGYLog> log, const bool skipHWDecodeCheck) {
    CodecCsp codecFeatures;
    MemType memType = HW_MEMORY;
    std::unique_ptr<CQSVHWDevice> hwdev;
    MFXVideoSession2 session;
    MFXVideoSession2Params params;
    bool bexternalAlloc = true;
    std::unique_ptr<QSVAllocator> allocator;
    auto err = RGY_ERR_NONE;
    if ((err = InitSessionAndDevice(hwdev, session, memType, deviceNum, params, log)) != RGY_ERR_NONE) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_DEV, _T("InitSessionAndDevice: failed to initialize: %s.\n"), get_err_mes(err));
    } else if ((err = CreateAllocator(allocator, bexternalAlloc, memType, hwdev.get(), session, log)) != RGY_ERR_NONE) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_DEV, _T("CreateAllocator: failed to create allocator: %s.\n"), get_err_mes(err));
    } else {
        log->write(RGY_LOG_DEBUG, RGY_LOGT_DEV, _T("InitSession: initialized allocator.\n"));
        codecFeatures = MakeDecodeFeatureList(session, codecIdList, log, skipHWDecodeCheck);
    }
    return codecFeatures;
}

static const TCHAR *const QSV_FEATURE_MARK_YES_NO[] = { _T("×"), _T("○") };
static const TCHAR *const QSV_FEATURE_MARK_YES_NO_WITH_SPACE[] = { _T(" x    "), _T(" o    ") };

tstring MakeFeatureListStr(const uint64_t feature) {
    tstring str;
    for (const FEATURE_DESC *ptr = list_enc_feature; ptr->desc; ptr++) {
        str += ptr->desc;
        str += QSV_FEATURE_MARK_YES_NO_WITH_SPACE[!!(feature & ptr->value)];
        str += _T("\n");
    }
    str += _T("\n");
    return str;
}

std::vector<std::pair<QSVEncFeatureData, tstring>> MakeFeatureListStr(const QSVDeviceNum deviceNum, const FeatureListStrType type, const vector<RGY_CODEC>& codecLists, std::shared_ptr<RGYLog> log) {
    const auto featurePerCodec = MakeFeatureListPerCodec(deviceNum, make_vector(list_rate_control_ry), codecLists, log);

    std::vector<std::pair<QSVEncFeatureData, tstring>> strPerCodec;

    // H.264がサポートされているかチェック
    const bool h264Supported = std::accumulate(featurePerCodec.begin(), featurePerCodec.end(), (uint64_t)0, [](uint64_t sum, const QSVEncFeatureData& value) {
        if (value.codec == RGY_CODEC_H264 && value.available()) {
            sum++;
        }
        return sum;
    }) != 0;

    for (const auto& availableFeatureForEachRC : featurePerCodec) {
        tstring str;
        //H.264以外で、ひとつもフラグが立っていなかったら、スキップする
        if ((availableFeatureForEachRC.codec != RGY_CODEC_H264 || h264Supported || availableFeatureForEachRC.lowPwer)
            && !availableFeatureForEachRC.available()) {
            continue;
        }
        str += _T("Codec: ") + tstring(CodecToStr(availableFeatureForEachRC.codec)) + _T(" ") + (availableFeatureForEachRC.lowPwer ? _T("FF") : _T("PG")) + _T("\n");

        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += _T("<table class=simpleOrange>");
        }

        switch (type) {
        case FEATURE_LIST_STR_TYPE_HTML: str += _T("<tr><th></th>"); break;
        case FEATURE_LIST_STR_TYPE_TXT:
        default:
            //ヘッダ部分
            const size_t row_header_length = _tcslen(list_enc_feature[0].desc);
            for (size_t i = 1; i < row_header_length; i++)
                str += _T(" ");
            break;
        }

        for (size_t i = 0; i < _countof(list_rate_control_ry); i++) {
            const auto ratecontrol = list_rate_control_ry[i].value;
            switch (type) {
            case FEATURE_LIST_STR_TYPE_CSV: str += _T(","); break;
            case FEATURE_LIST_STR_TYPE_HTML: str += _T("<th>"); break;
            case FEATURE_LIST_STR_TYPE_TXT:
            default: str += _T(" "); break;
            }
            str += get_cx_desc(list_rate_control_ry, ratecontrol);
            if (type == FEATURE_LIST_STR_TYPE_HTML) {
                str += _T("</th>");
            }
        }
        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += _T("</tr>");
        }
        str += _T("\n");

        //モードがサポートされているか
        for (const FEATURE_DESC *ptr = list_enc_feature; ptr->desc; ptr++) {
            if (type == FEATURE_LIST_STR_TYPE_HTML) {
                str += _T("<tr><td>");
            }
            str += ptr->desc;
            switch (type) {
            case FEATURE_LIST_STR_TYPE_CSV: str += _T(","); break;
            case FEATURE_LIST_STR_TYPE_HTML: str += _T("</td>"); break;
            default: break;
            }
            for (size_t i = 0; i < _countof(list_rate_control_ry); i++) {
                const auto ratecontrol = list_rate_control_ry[i].value;
                const auto feature = availableFeatureForEachRC.feature.count(ratecontrol) > 0 ? availableFeatureForEachRC.feature.at(ratecontrol) : 0;
                if (type == FEATURE_LIST_STR_TYPE_HTML) {
                    str += !!(feature & ptr->value) ? _T("<td class=ok>") : _T("<td class=fail>");
                }
                if (type == FEATURE_LIST_STR_TYPE_TXT) {
                    str += QSV_FEATURE_MARK_YES_NO_WITH_SPACE[!!(feature & ptr->value)];
                } else {
                    str += QSV_FEATURE_MARK_YES_NO[!!(feature & ptr->value)];
                }
                switch (type) {
                case FEATURE_LIST_STR_TYPE_CSV: str += _T(","); break;
                case FEATURE_LIST_STR_TYPE_HTML: str += _T("</td>"); break;
                default: break;
                }
            }
            if (type == FEATURE_LIST_STR_TYPE_HTML) {
                str += _T("</tr>");
            }
            str += _T("\n");
        }
        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += _T("</table><br>");
        }
        str += _T("\n");
        strPerCodec.push_back(std::make_pair(availableFeatureForEachRC, str));
    }
    return strPerCodec;
}

std::vector<std::pair<QSVEncFeatureData, tstring>> MakeFeatureListStr(const QSVDeviceNum deviceNum, const FeatureListStrType type, std::shared_ptr<RGYLog> log) {
    const std::vector<RGY_CODEC> codecLists = { RGY_CODEC_H264, RGY_CODEC_HEVC, RGY_CODEC_MPEG2, RGY_CODEC_VP8, RGY_CODEC_VP9, RGY_CODEC_AV1 };
    return MakeFeatureListStr(deviceNum, type, codecLists, log);
}

tstring MakeVppFeatureStr(const QSVDeviceNum deviceNum, FeatureListStrType type, std::shared_ptr<RGYLog> log) {
    uint64_t features = CheckVppFeatures(deviceNum, log);
    const TCHAR *MARK_YES_NO[] = { _T(" x"), _T(" o") };
    tstring str;
    if (type == FEATURE_LIST_STR_TYPE_HTML) {
        str += _T("<table class=simpleOrange>");
    }
    for (const FEATURE_DESC *ptr = list_vpp_feature; ptr->desc; ptr++) {
        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += _T("<tr><td>");
        }
        str += ptr->desc;
        switch (type) {
        case FEATURE_LIST_STR_TYPE_CSV: str += _T(","); break;
        case FEATURE_LIST_STR_TYPE_HTML: str += _T("</td>"); break;
        default: break;
        }
        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += (features & ptr->value) ? _T("<td class=ok>") : _T("<td class=fail>");
        }
        if (type == FEATURE_LIST_STR_TYPE_TXT) {
            str += MARK_YES_NO[ptr->value == (features & ptr->value)];
        } else {
            str += QSV_FEATURE_MARK_YES_NO[ptr->value == (features & ptr->value)];
        }
        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += _T("</td></tr>");
        }
        str += _T("\n");
    }
    if (type == FEATURE_LIST_STR_TYPE_HTML) {
        str += _T("</table>\n");
    }
    return str;
}

tstring MakeDecFeatureStr(const QSVDeviceNum deviceNum, FeatureListStrType type, std::shared_ptr<RGYLog> log) {
#if ENABLE_AVSW_READER
    vector<RGY_CODEC> codecLists;
    for (int i = 0; i < _countof(HW_DECODE_LIST); i++) {
        codecLists.push_back(HW_DECODE_LIST[i].rgy_codec);
    }
    auto decodeCodecCsp = MakeDecodeFeatureList(deviceNum, codecLists, log, false);

    const auto chromafmts = make_array<RGY_CHROMAFMT>(RGY_CHROMAFMT_YUV420, RGY_CHROMAFMT_YUV422, RGY_CHROMAFMT_YUV444);
    std::map<RGY_CODEC, std::vector<int>> featurePerCodec;
    for (int i = 0; i < _countof(HW_DECODE_LIST); i++) {
        const auto target_codec = HW_DECODE_LIST[i].rgy_codec;
        std::vector<int> chromafmt_bitdepth(chromafmts.size(), 0);
        if (decodeCodecCsp.count(target_codec) > 0) {
            auto codecCsps = decodeCodecCsp[target_codec];
            for (size_t icfmt = 0; icfmt < chromafmts.size(); icfmt++) {
                int max_bitdepth = 0;
                for (auto csp : codecCsps) {
                    if (RGY_CSP_CHROMA_FORMAT[csp] == chromafmts[icfmt]) {
                        max_bitdepth = std::max(max_bitdepth, (int)RGY_CSP_BIT_DEPTH[csp]);
                    }
                }
                chromafmt_bitdepth[icfmt] = max_bitdepth;
            }
        }
        featurePerCodec[target_codec] = chromafmt_bitdepth;
    }


    const TCHAR *MARK_YES_NO[] = { _T("   x "), _T("   o ") };
    tstring str;
    if (type == FEATURE_LIST_STR_TYPE_HTML) {
        str += _T("<table class=simpleOrange>");
    }
    if (type == FEATURE_LIST_STR_TYPE_HTML) {
        str += _T("<tr><td></td>");
    }

    int maxFeatureStrLen = 0;
    for (const auto& cfmt : chromafmts) {
        maxFeatureStrLen = (std::max<int>)(maxFeatureStrLen, (int)_tcslen(RGY_CHROMAFMT_NAMES[cfmt]));
    }

    if (type != FEATURE_LIST_STR_TYPE_HTML) {
        for (int i = 0; i < maxFeatureStrLen+2; i++) {
            str += _T(" ");
        }
    }
    for (uint32_t i_codec = 0; i_codec < codecLists.size(); i_codec++) {
        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += _T("<td>");
        }
        tstring codecStr = CodecToStr(codecLists[i_codec]);
        codecStr = str_replace(codecStr, _T("H.264/AVC"), _T("H.264"));
        codecStr = str_replace(codecStr, _T("H.265/HEVC"), _T("HEVC"));
        while (codecStr.length() < 6) {
            codecStr += _T(" ");
        }
        str += codecStr;
        switch (type) {
        case FEATURE_LIST_STR_TYPE_TXT: str += _T(" ");
            break;
        case FEATURE_LIST_STR_TYPE_CSV:
            str += _T(",");
            break;
        case FEATURE_LIST_STR_TYPE_HTML: str += _T("</td>"); break;
        default: break;
        }
    }
    if (type == FEATURE_LIST_STR_TYPE_HTML) {
        str += _T("</tr>");
    }
    str += _T("\n");

    for (size_t icfmt = 0; icfmt < chromafmts.size(); icfmt++) {
        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += _T("<tr><td>");
        }
        str += RGY_CHROMAFMT_NAMES[chromafmts[icfmt]];
        switch (type) {
        case FEATURE_LIST_STR_TYPE_CSV: str += _T(","); break;
        case FEATURE_LIST_STR_TYPE_HTML: str += _T("</td>"); break;
        default: break;
        }
        for (uint32_t i_codec = 0; i_codec < codecLists.size(); i_codec++) {
            auto codecFmts = featurePerCodec[codecLists[i_codec]];
            if (type == FEATURE_LIST_STR_TYPE_HTML) {
                str += (codecFmts[icfmt] > 0) ? _T("<td class=ok>") : _T("<td class=fail>");
            }
            if (codecFmts[icfmt] > 0) {
                str += strsprintf(_T(" %2dbit "), codecFmts[icfmt]);
            } else {
                str += _T("       ");
            }
            if (type == FEATURE_LIST_STR_TYPE_HTML) {
                str += _T("</td>");
            }
        }
        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += _T("</tr>");
        }
        str += _T("\n");
    }
    if (type == FEATURE_LIST_STR_TYPE_HTML) {
        str += _T("</table>\n");
    }
    return str;
#else
    return _T("");
#endif
}

CodecCsp getHWDecCodecCsp(const QSVDeviceNum deviceNum, std::shared_ptr<RGYLog> log, const bool skipHWDecodeCheck) {
#if ENABLE_AVSW_READER
    vector<RGY_CODEC> codecLists;
    for (int i = 0; i < _countof(HW_DECODE_LIST); i++) {
        codecLists.push_back(HW_DECODE_LIST[i].rgy_codec);
    }
    if (skipHWDecodeCheck) {
        CodecCsp codecFeatures;
        for (auto codec : codecLists) {
            codecFeatures[codec] = {
                RGY_CSP_NV12, RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16,
                RGY_CSP_YUV422, RGY_CSP_YUV422_09, RGY_CSP_YUV422_10, RGY_CSP_YUV422_12, RGY_CSP_YUV422_14, RGY_CSP_YUV422_16,
                RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16
            };
        }
        return codecFeatures;
    }
    return MakeDecodeFeatureList(deviceNum, codecLists, log, skipHWDecodeCheck);
#else
    return CodecCsp();
#endif
}

QSV_CPU_GEN getCPUGen(MFXVideoSession *pSession) {
    if (pSession == nullptr) {
        return getCPUGenCpuid();
    }
    mfxVersion mfxVer;
    pSession->QueryVersion(&mfxVer);
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_19)) {
        mfxPlatform platform;
        memset(&platform, 0, sizeof(platform));
        pSession->QueryPlatform(&platform);
        return cpu_gen_enc_to_rgy(platform.CodeName);
    } else {
        return getCPUGenCpuid();
    }
}

int GetImplListStr(tstring& str) {
    auto log = std::make_shared<RGYLog>(nullptr, RGY_LOG_INFO);
    const auto implList = getVPLImplList(log);
    str.clear();
    for (const auto& impl : implList) {
        for (int iacc = 0; iacc == 0 || iacc < (int)impl.AccelerationModeDescription.NumAccelerationModes; iacc++) {
            const auto accMode = (impl.AccelerationModeDescription.NumAccelerationModes > 0) ? impl.AccelerationModeDescription.Mode[iacc] : impl.AccelerationMode;
            str += strsprintf(_T("API %d.%02d: %s(%s), Acceleration %s\n"),
                impl.ApiVersion.Major, impl.ApiVersion.Minor,
                (impl.Impl == MFX_IMPL_TYPE_HARDWARE) ? _T("hw") : _T("sw"),
                char_to_tstring(impl.ImplName).c_str(),
                MFXAccelerationModeToStr(accMode).c_str());
        }
    }
    return (int)implList.size();
}

std::vector<tstring> getDeviceNameList() {
    std::vector<tstring> result;
    auto log = std::make_shared<RGYLog>(nullptr, RGY_LOG_QUIET);
    for (int idev = 1; idev <= (int)QSVDeviceNum::MAX; idev++) {
        auto dev = std::make_unique<QSVDevice>();
        if (dev->init((QSVDeviceNum)idev, true, true) != RGY_ERR_NONE) {
            break;
        }
        auto info = dev->devInfo();
        if (info && info->name.length() > 0) {
            auto gpu_name = info->name;
            gpu_name = str_replace(gpu_name, "(R)", "");
            gpu_name = str_replace(gpu_name, "(TM)", "");
            result.push_back(strsprintf(_T("Device #%d: %s"), idev, char_to_tstring(gpu_name).c_str()));
        } else {
            tstring name;
            if (dev->hwdev()) {
                name = dev->hwdev()->GetName();
            }
            if (!name.empty()) {
                result.push_back(strsprintf(_T("Device #%d: %s"), idev, name.c_str()));
            } else {
                result.push_back(strsprintf(_T("Device #%d"), idev));
            }
        }
    }
    return result;
}
