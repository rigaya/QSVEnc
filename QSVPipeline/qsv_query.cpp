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
#include <type_traits>
#include "rgy_osdep.h"
#ifndef _MSC_VER
#include <sys/sysinfo.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <iconv.h>
#endif
#include "mfxstructures.h"
#include "mfxvideo.h"
#include "mfxvideo++.h"
#include "mfxplugin.h"
#include "mfxplugin++.h"
#include "mfxjpeg.h"
#include "rgy_tchar.h"
#include "rgy_util.h"
#include "rgy_avutil.h"
#include "qsv_util.h"
#include "qsv_prm.h"
#include "qsv_plugin.h"
#include "rgy_osdep.h"
#include "qsv_query.h"
#include "qsv_hw_device.h"
#include "cpu_info.h"

#if D3D_SURFACES_SUPPORT
#include "qsv_hw_d3d9.h"
#include "qsv_hw_d3d11.h"

#include "qsv_allocator_d3d9.h"
#include "qsv_allocator_d3d11.h"
#endif

#ifdef LIBVA_SUPPORT
#include "qsv_hw_va.h"
#include "qsv_allocator_va.h"
#endif

#if 1
int getCPUGenCpuid() {
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

    return CPU_GEN_SANDYBRIDGE;
}
#endif

static const auto RGY_CPU_GEN_TO_MFX = make_array<std::pair<int, uint32_t>>(
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
    std::make_pair(CPU_GEN_TIGERLAKE, MFX_PLATFORM_TIGERLAKE)
    );
MAP_PAIR_0_1(cpu_gen, rgy, int, enc, uint32_t, RGY_CPU_GEN_TO_MFX, CPU_GEN_UNKNOWN, MFX_PLATFORM_UNKNOWN);


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

mfxVersion get_mfx_lib_version(mfxIMPL impl) {
    if (impl == MFX_IMPL_SOFTWARE) {
        return LIB_VER_LIST[0];
    }
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

mfxVersion get_mfx_libhw_version() {
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
        test = get_mfx_lib_version(impl_list[i]);
        if (check_lib_version(test, MFX_LIB_VERSION_1_1))
            break;
    }
    return test;
}
bool check_if_d3d11_necessary() {
    bool check_d3d11 = (0 != check_lib_version(get_mfx_lib_version(MFX_IMPL_HARDWARE_ANY | MFX_IMPL_VIA_D3D11), MFX_LIB_VERSION_1_1));
    bool check_d3d9  = (0 != check_lib_version(get_mfx_lib_version(MFX_IMPL_HARDWARE_ANY | MFX_IMPL_VIA_D3D9), MFX_LIB_VERSION_1_1));

    return (check_d3d11 == true && check_d3d9 == false);
}
mfxVersion get_mfx_libsw_version() {
    return get_mfx_lib_version(MFX_IMPL_SOFTWARE);
}

std::vector<RGY_CSP> CheckDecFeaturesInternal(MFXVideoSession& session, mfxVersion mfxVer, mfxU32 codecId) {
    std::vector<RGY_CSP> supportedCsp;
    MFXVideoDECODE dec(session);
    mfxIMPL impl;
    session.QueryIMPL(&impl);
    const auto HARDWARE_IMPL = make_array<mfxIMPL>(MFX_IMPL_HARDWARE, MFX_IMPL_HARDWARE_ANY, MFX_IMPL_HARDWARE2, MFX_IMPL_HARDWARE3, MFX_IMPL_HARDWARE4);
    const bool bHardware = HARDWARE_IMPL.end() != std::find(HARDWARE_IMPL.begin(), HARDWARE_IMPL.end(), MFX_IMPL_BASETYPE(impl));

    auto sessionPlugins = std::unique_ptr<CSessionPlugins>(new CSessionPlugins(session));
    auto plugin = getMFXPluginUID(MFXComponentType::DECODE, codecId, false);
    if (plugin != nullptr) {
        if (MFX_ERR_NONE != sessionPlugins->LoadPlugin(MFX_PLUGINTYPE_VIDEO_DECODE, *plugin, 1)) {
            return supportedCsp;
        }
    }
    mfxVideoParam videoPrm, videoPrmOut;
    memset(&videoPrm,  0, sizeof(videoPrm));
    videoPrm.AsyncDepth                  = 3;
    videoPrm.IOPattern                   = MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    videoPrm.mfx.CodecId                 = codecId;
    switch (codecId) {
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
    default:
    case MFX_CODEC_AVC:
        videoPrm.mfx.CodecLevel          = MFX_LEVEL_AVC_41;
        videoPrm.mfx.CodecProfile        = MFX_PROFILE_AVC_HIGH;
        break;
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
        mfxStatus ret = dec.Query(&videoPrm, &videoPrmOut);
        if (ret != MFX_ERR_NONE) {
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
            if (MFX_ERR_NONE <= dec.Query(&videoPrm, &videoPrmOut)) { \
                supportedCsp.push_back(rgy_csp); \
            } \
        } \
    }

    static const auto test_yuv420_highbit_depth = make_array<std::pair<int, RGY_CSP>>(
        std::make_pair( 9, RGY_CSP_YV12_09),
        std::make_pair(10, RGY_CSP_YV12_10),
        std::make_pair(12, RGY_CSP_YV12_12),
        std::make_pair(14, RGY_CSP_YV12_14),
        std::make_pair(14, RGY_CSP_YV12_16)
        );
    static const auto test_yuv444 = make_array<std::pair<int, RGY_CSP>>(
        std::make_pair( 8, RGY_CSP_YUV444),
        std::make_pair( 9, RGY_CSP_YUV444_09),
        std::make_pair(10, RGY_CSP_YUV444_10),
        std::make_pair(12, RGY_CSP_YUV444_12),
        std::make_pair(14, RGY_CSP_YUV444_14),
        std::make_pair(16, RGY_CSP_YUV444_16)
        );

    mfxVideoParam videoPrmTmp = videoPrm;
    for (const auto& test : test_yuv420_highbit_depth) {
        videoPrm.mfx.FrameInfo.FourCC = MFX_FOURCC_P010;
        if (codecId == MFX_CODEC_HEVC) {
            videoPrm.mfx.CodecProfile = (mfxU16)((test.first > 8) ? MFX_PROFILE_HEVC_MAIN10 : MFX_PROFILE_HEVC_MAIN);
        } else if (codecId == MFX_CODEC_VP9) {
            videoPrm.mfx.CodecProfile = (mfxU16)((test.first > 8) ? MFX_PROFILE_VP9_2 : MFX_PROFILE_VP9_0);
        } else {
            break;
        }
        videoPrm.mfx.FrameInfo.BitDepthLuma = (mfxU16)((test.first > 8) ? test.first : 0);
        videoPrm.mfx.FrameInfo.BitDepthChroma = (mfxU16)((test.first > 8) ? test.first : 0);
        videoPrm.mfx.FrameInfo.Shift = (test.first > 8) ? 1 : 0;
        CHECK_FEATURE(test.second, MFX_LIB_VERSION_1_19);
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
    MFXVideoVPP vpp(session);
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
    INIT_MFX_EXT_BUFFER(vppDoUse,        MFX_EXTBUFF_VPP_DOUSE);
    INIT_MFX_EXT_BUFFER(vppDoNotUse,     MFX_EXTBUFF_VPP_DONOTUSE);
    INIT_MFX_EXT_BUFFER(vppFpsConv,      MFX_EXTBUFF_VPP_FRAME_RATE_CONVERSION);
    INIT_MFX_EXT_BUFFER(vppImageStab,    MFX_EXTBUFF_VPP_IMAGE_STABILIZATION);
    INIT_MFX_EXT_BUFFER(vppVSI,          MFX_EXTBUFF_VPP_VIDEO_SIGNAL_INFO);
    INIT_MFX_EXT_BUFFER(vppRotate,       MFX_EXTBUFF_VPP_ROTATION);
    INIT_MFX_EXT_BUFFER(vppMirror,       MFX_EXTBUFF_VPP_MIRRORING);
    INIT_MFX_EXT_BUFFER(vppScaleQuality, MFX_EXTBUFF_VPP_SCALING);
    INIT_MFX_EXT_BUFFER(vppMctf,         MFX_EXTBUFF_VPP_MCTF);

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

    memcpy(&vppDoUseOut,        &vppDoUse,        sizeof(vppDoUse));
    memcpy(&vppDoNotUseOut,     &vppDoNotUse,     sizeof(vppDoNotUse));
    memcpy(&vppFpsConvOut,      &vppFpsConv,      sizeof(vppFpsConv));
    memcpy(&vppImageStabOut,    &vppImageStab,    sizeof(vppImageStab));
    memcpy(&vppVSIOut,          &vppVSI,          sizeof(vppVSI));
    memcpy(&vppRotateOut,       &vppRotate,       sizeof(vppRotate));
    memcpy(&vppMirrorOut,       &vppMirror,       sizeof(vppMirror));
    memcpy(&vppScaleQualityOut, &vppScaleQuality, sizeof(vppScaleQuality));
    memcpy(&vppMctfOut,         &vppMctf,         sizeof(vppMctf));

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
            mfxStatus ret = vpp.Query(&videoPrm, &videoPrmOut);
            if (MFX_ERR_NONE <= ret) {
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

    videoPrm.vpp.Out.FrameRateExtN    = 60000;
    videoPrm.vpp.Out.FrameRateExtD    = 1001;
    videoPrmOut.vpp.Out.FrameRateExtN = 60000;
    videoPrmOut.vpp.Out.FrameRateExtD = 1001;
    check_feature((mfxExtBuffer *)&vppFpsConv,   (mfxExtBuffer *)&vppFpsConvOut,   MFX_LIB_VERSION_1_3,  VPP_FEATURE_FPS_CONVERSION_ADV,  VPP_FEATURE_FPS_CONVERSION);
    return result;
}

mfxStatus InitSession(MFXVideoSession& session, bool useHWLib, MemType& memType) {
    mfxStatus sts = MFX_ERR_INVALID_HANDLE;
    if (useHWLib) {
        //とりあえず、MFX_IMPL_HARDWARE_ANYでの初期化を試みる
        mfxIMPL impl = MFX_IMPL_HARDWARE_ANY;
#if MFX_D3D11_SUPPORT
        //Win7でD3D11のチェックをやると、
        //デスクトップコンポジションが切られてしまう問題が発生すると報告を頂いたので、
        //D3D11をWin8以降に限定
        if (!check_OS_Win8orLater()) {
            memType &= (MemType)(~D3D11_MEMORY);
        }
        if (HW_MEMORY == (memType & HW_MEMORY) && false == check_if_d3d11_necessary()) {
            memType &= (MemType)(~D3D11_MEMORY);
        }

#endif //#if MFX_D3D11_SUPPORT
        //まずd3d11モードを試すよう設定されていれば、ますd3d11を試して、失敗したらd3d9での初期化を試みる
        for (int i_try_d3d11 = 0; i_try_d3d11 < 1 + (HW_MEMORY == (memType & HW_MEMORY)); i_try_d3d11++) {
#if D3D_SURFACES_SUPPORT
#if MFX_D3D11_SUPPORT
            if (D3D11_MEMORY & memType) {
                if (0 == i_try_d3d11) {
                    impl |= MFX_IMPL_VIA_D3D11; //d3d11モードも試す場合は、まずd3d11モードをチェック
                    memType = D3D11_MEMORY;
                } else {
                    impl &= ~MFX_IMPL_VIA_D3D11; //d3d11をオフにして再度テストする
                    impl |= MFX_IMPL_VIA_D3D9;
                    memType = D3D9_MEMORY;
                }
            } else
#endif //#if MFX_D3D11_SUPPORT
            if (D3D9_MEMORY & memType) {
                impl |= MFX_IMPL_VIA_D3D9; //d3d11モードも試す場合は、まずd3d11モードをチェック
            }
#endif //#if D3D_SURFACES_SUPPORT
            mfxVersion verRequired = MFX_LIB_VERSION_1_1;
            sts = session.Init(impl, &verRequired);

            //MFX_IMPL_HARDWARE_ANYがサポートされない場合もあり得るので、失敗したらこれをオフにしてもう一回試す
            if (MFX_ERR_NONE != sts) {
                sts = session.Init((impl & (~MFX_IMPL_HARDWARE_ANY)) | MFX_IMPL_HARDWARE, &verRequired);
            }

            //成功したらループを出る
            if (MFX_ERR_NONE == sts) {
                break;
            }
        }
    } else {
        mfxIMPL impl = MFX_IMPL_SOFTWARE;
        mfxVersion verRequired = MFX_LIB_VERSION_1_1;
        sts = session.Init(impl, &verRequired);
        memType = SYSTEM_MEMORY;
    }
    return sts;
}

std::unique_ptr<CQSVHWDevice> InitHWDevice(MFXVideoSession& session, MemType& memType, std::shared_ptr<RGYLog> log) {
    mfxStatus sts = MFX_ERR_NONE;
    std::unique_ptr<CQSVHWDevice> hwdev;
#if D3D_SURFACES_SUPPORT
    POINT point = {0, 0};
    HWND window = WindowFromPoint(point);

    if (memType) {
#if MFX_D3D11_SUPPORT
        if (memType == D3D11_MEMORY
            && (hwdev = std::make_unique<CQSVD3D11Device>(log))) {
            memType = D3D11_MEMORY;

            sts = hwdev->Init(NULL, 0, GetAdapterID(session));
            if (sts != MFX_ERR_NONE) {
                hwdev.reset();
            }
        }
#endif // #if MFX_D3D11_SUPPORT
        if (!hwdev && (hwdev = std::make_unique<CQSVD3D9Device>(log))) {
            //もし、d3d11要求で失敗したら自動的にd3d9に切り替える
            //sessionごと切り替える必要がある
            if (memType != D3D9_MEMORY) {
                memType = D3D9_MEMORY;
                InitSession(session, true, memType);
            }

            sts = hwdev->Init(window, 0, GetAdapterID(session));
        }
    }

#elif LIBVA_SUPPORT
    hwdev.reset(CreateVAAPIDevice("", MFX_LIBVA_DRM, log));
    if (hwdev) {
        sts = hwdev->Init(NULL, 0, GetAdapterID(session));
    }
    mfxHDL hdl = NULL;
    sts = hwdev->GetHandle(MFX_HANDLE_VA_DISPLAY, &hdl);

    //ハンドルを渡す
    sts = session.SetHandle(MFX_HANDLE_VA_DISPLAY, hdl);
#endif
    if (sts != MFX_ERR_NONE) {
        hwdev.reset();
    }
    return hwdev;
}

mfxU64 CheckVppFeatures(MFXVideoSession& session, mfxVersion ver) {
    mfxU64 feature = 0x00;
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

mfxU64 CheckVppFeatures(mfxVersion ver, std::shared_ptr<RGYLog> log) {
    mfxU64 feature = 0x00;
    if (!check_lib_version(ver, MFX_LIB_VERSION_1_3)) {
        //API v1.3未満で実際にチェックする必要は殆ど無いので、
        //コードで決められた値を返すようにする
        feature |= VPP_FEATURE_RESIZE;
        feature |= VPP_FEATURE_DEINTERLACE;
        feature |= VPP_FEATURE_DENOISE;
        feature |= VPP_FEATURE_DETAIL_ENHANCEMENT;
        feature |= VPP_FEATURE_PROC_AMP;
    } else {
        MemType memType = HW_MEMORY;
        MFXVideoSession session;
        if (InitSession(session, true, memType) == MFX_ERR_NONE) {
            if (auto hwdevice = InitHWDevice(session, memType, log)) {
                feature = CheckVppFeaturesInternal(session, ver);
            }
        }

    }

    return feature;
}

mfxU64 CheckEncodeFeature(MFXVideoSession& session, mfxVersion mfxVer, int ratecontrol, mfxU32 codecId) {
    if (codecId == MFX_CODEC_HEVC && !check_lib_version(mfxVer, MFX_LIB_VERSION_1_15)) {
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

    MFXVideoENCODE encode(session);

    mfxExtCodingOption cop;
    mfxExtCodingOption2 cop2;
    mfxExtCodingOption3 cop3;
    mfxExtHEVCParam hevc;
    mfxExtVP9Param vp9;
    INIT_MFX_EXT_BUFFER(cop,  MFX_EXTBUFF_CODING_OPTION);
    INIT_MFX_EXT_BUFFER(cop2, MFX_EXTBUFF_CODING_OPTION2);
    INIT_MFX_EXT_BUFFER(cop3, MFX_EXTBUFF_CODING_OPTION3);
    INIT_MFX_EXT_BUFFER(hevc, MFX_EXTBUFF_HEVC_PARAM);
    INIT_MFX_EXT_BUFFER(vp9, MFX_EXTBUFF_VP9_PARAM);

    std::vector<mfxExtBuffer *> buf;
    buf.push_back((mfxExtBuffer *)&cop);
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_6)) {
        buf.push_back((mfxExtBuffer *)&cop2);
    }
#if ENABLE_FEATURE_COP3_AND_ABOVE
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
#endif //#if ENABLE_FEATURE_COP3_AND_ABOVE

    mfxVideoParam videoPrm;
    RGY_MEMSET_ZERO(videoPrm);

    auto set_default_quality_prm = [&videoPrm]() {
        if (   videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_VBR
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_AVBR
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_CBR
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_LA
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_LA_HRD
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_LA_EXT
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
    videoPrm.mfx.GopRefDist              = 4;
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
        videoPrm.AsyncDepth = 0;
        videoPrm.IOPattern = MFX_IOPATTERN_IN_OPAQUE_MEMORY;
        if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_9)) {
            videoPrm.mfx.FrameInfo.BitDepthLuma = 8;
            videoPrm.mfx.FrameInfo.BitDepthChroma = 8;
        }
        break;
    default:
    case MFX_CODEC_AVC:
        videoPrm.mfx.CodecLevel = MFX_LEVEL_AVC_41;
        videoPrm.mfx.CodecProfile = MFX_PROFILE_AVC_HIGH;
        break;
    }
    set_default_quality_prm();

    mfxExtCodingOption copOut;
    mfxExtCodingOption2 cop2Out;
    mfxExtCodingOption3 cop3Out;
    mfxExtHEVCParam hevcOut;
    mfxExtVP9Param vp9Out;
    std::vector<mfxExtBuffer *> bufOut;
    bufOut.push_back((mfxExtBuffer *)&copOut);
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_6)) {
        bufOut.push_back((mfxExtBuffer *)&cop2Out);
    }
#if ENABLE_FEATURE_COP3_AND_ABOVE
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_11)) {
        bufOut.push_back((mfxExtBuffer *)&cop3Out);
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_15)
        && codecId == MFX_CODEC_HEVC) {
        hevc.PicWidthInLumaSamples  = videoPrm.mfx.FrameInfo.CropW;
        hevc.PicHeightInLumaSamples = videoPrm.mfx.FrameInfo.CropH;
        bufOut.push_back((mfxExtBuffer*)&hevcOut);
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_26)
        && codecId == MFX_CODEC_VP9) {
        vp9.FrameWidth = videoPrm.mfx.FrameInfo.Width;
        vp9.FrameHeight = videoPrm.mfx.FrameInfo.Height;
        vp9.NumTileRows = 1;
        vp9.NumTileColumns = 1;
        bufOut.push_back((mfxExtBuffer *)&vp9Out);
    }
#endif //#if ENABLE_FEATURE_COP3_AND_ABOVE
    mfxVideoParam videoPrmOut;
    //In, Outのパラメータが同一となっているようにきちんとコピーする
    //そうしないとQueryが失敗する
    memcpy(&copOut,  &cop,  sizeof(cop));
    memcpy(&cop2Out, &cop2, sizeof(cop2));
    memcpy(&cop3Out, &cop3, sizeof(cop3));
    memcpy(&hevcOut, &hevc, sizeof(hevc));
    memcpy(&vp9Out, &vp9, sizeof(vp9));
    memcpy(&videoPrmOut, &videoPrm, sizeof(videoPrm));
    videoPrm.NumExtParam = (mfxU16)bufOut.size();
    videoPrm.ExtParam = &bufOut[0];

    mfxStatus ret = encode.Query(&videoPrm, &videoPrmOut);

    mfxU64 result = (MFX_ERR_NONE <= ret && videoPrm.mfx.RateControlMethod == videoPrmOut.mfx.RateControlMethod) ? ENC_FEATURE_CURRENT_RC : 0x00;
    if (result) {

        //まず、エンコードモードについてチェック
        auto check_enc_mode = [&](mfxU16 mode, mfxU64 flag, mfxVersion required_ver) {
            if (check_lib_version(mfxVer, required_ver)) {
                mfxU16 original_method = videoPrm.mfx.RateControlMethod;
                videoPrm.mfx.RateControlMethod = mode;
                set_default_quality_prm();
                memcpy(&copOut,  &cop,  sizeof(cop));
                memcpy(&cop2Out, &cop2, sizeof(cop2));
                memcpy(&cop3Out, &cop3, sizeof(cop3));
                memcpy(&hevcOut, &hevc, sizeof(hevc));
                memcpy(&vp9Out,  &vp9,  sizeof(vp9));
                memcpy(&videoPrmOut, &videoPrm, sizeof(videoPrm));
                videoPrm.NumExtParam = (mfxU16)bufOut.size();
                videoPrm.ExtParam = &bufOut[0];
                if (MFX_ERR_NONE <= encode.Query(&videoPrm, &videoPrmOut) && videoPrm.mfx.RateControlMethod == videoPrmOut.mfx.RateControlMethod)
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

#define CHECK_FEATURE(membersIn, membersOut, flag, value, required_ver) { \
        if (check_lib_version(mfxVer, (required_ver))) { \
            mfxU16 temp = (membersIn); \
            (membersIn) = (value); \
            memcpy(&videoPrmOut,  &videoPrm,  sizeof(videoPrm)); \
            memcpy(&copOut,  &cop,  sizeof(cop)); \
            memcpy(&cop2Out, &cop2, sizeof(cop2)); \
            memcpy(&cop3Out, &cop3, sizeof(cop3)); \
            memcpy(&hevcOut, &hevc, sizeof(hevc)); \
            auto check_ret = encode.Query(&videoPrm, &videoPrmOut); \
            if (MFX_ERR_NONE <= check_ret \
                && (membersIn) == (membersOut) \
                && videoPrm.mfx.RateControlMethod == videoPrmOut.mfx.RateControlMethod) { \
                result |= (flag); \
            } else { \
                /*_ftprintf(stderr, _T("error checking " # flag ": %s\n"), get_err_mes(err_to_rgy(check_ret)));*/ \
            } \
            (membersIn) = temp; \
        } \
    }
        //これはもう単純にAPIチェックでOK
        if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_3)) {
            result |= ENC_FEATURE_VUI_INFO;
        }
        //ひとつひとつパラメータを入れ替えて試していく
#pragma warning(push)
#pragma warning(disable:4244) //'mfxU16' から 'mfxU8' への変換です。データが失われる可能性があります。
#define PICTYPE mfx.FrameInfo.PicStruct
        const mfxU32 MFX_TRELLIS_ALL = MFX_TRELLIS_I | MFX_TRELLIS_P | MFX_TRELLIS_B;
        CHECK_FEATURE(cop.AUDelimiter,           copOut.AUDelimiter,           ENC_FEATURE_AUD,           MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_1);
        CHECK_FEATURE(videoPrm.PICTYPE,          videoPrmOut.PICTYPE,          ENC_FEATURE_INTERLACE,     MFX_PICSTRUCT_FIELD_TFF, MFX_LIB_VERSION_1_1);
        CHECK_FEATURE(cop.PicTimingSEI,          copOut.PicTimingSEI,          ENC_FEATURE_PIC_STRUCT,    MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_1);
        CHECK_FEATURE(cop.RateDistortionOpt,     copOut.RateDistortionOpt,     ENC_FEATURE_RDO,           MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_1);
        CHECK_FEATURE(cop.CAVLC,                 copOut.CAVLC,                 ENC_FEATURE_CAVLC,         MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_1);
        CHECK_FEATURE(cop2.ExtBRC,               cop2Out.ExtBRC,               ENC_FEATURE_EXT_BRC,       MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_6);
        CHECK_FEATURE(cop2.MBBRC,                cop2Out.MBBRC,                ENC_FEATURE_MBBRC,         MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_6);
        CHECK_FEATURE(cop2.Trellis,              cop2Out.Trellis,              ENC_FEATURE_TRELLIS,       MFX_TRELLIS_ALL,         MFX_LIB_VERSION_1_7);
        cop2.IntRefCycleSize = 16;
        CHECK_FEATURE(cop2.IntRefType,           cop2Out.IntRefType,           ENC_FEATURE_INTRA_REFRESH, 1,                       MFX_LIB_VERSION_1_7);
        cop2.IntRefCycleSize = 0;
        CHECK_FEATURE(cop2.AdaptiveI,            cop2Out.AdaptiveI,            ENC_FEATURE_ADAPTIVE_I,    MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_8);
        CHECK_FEATURE(cop2.AdaptiveB,            cop2Out.AdaptiveB,            ENC_FEATURE_ADAPTIVE_B,    MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_8);
        CHECK_FEATURE(cop2.BRefType,             cop2Out.BRefType,             ENC_FEATURE_B_PYRAMID,     MFX_B_REF_PYRAMID,       MFX_LIB_VERSION_1_8);
        if (rc_is_type_lookahead(ratecontrol)) {
            CHECK_FEATURE(cop2.LookAheadDS,      cop2Out.LookAheadDS,          ENC_FEATURE_LA_DS,         MFX_LOOKAHEAD_DS_2x,     MFX_LIB_VERSION_1_8);
        }
        CHECK_FEATURE(cop2.DisableDeblockingIdc, cop2Out.DisableDeblockingIdc, ENC_FEATURE_NO_DEBLOCK,    MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_9);
        CHECK_FEATURE(cop2.MaxQPI,               cop2Out.MaxQPI,               ENC_FEATURE_QP_MINMAX,     48,                      MFX_LIB_VERSION_1_9);
        cop3.WinBRCMaxAvgKbps = 3000;
        CHECK_FEATURE(cop3.WinBRCSize,           cop3Out.WinBRCSize,           ENC_FEATURE_WINBRC,        10,                      MFX_LIB_VERSION_1_11);
        cop3.WinBRCMaxAvgKbps = 0;
        CHECK_FEATURE(cop3.EnableMBQP,                 cop3Out.EnableMBQP,                 ENC_FEATURE_PERMBQP,                    MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_13);
        CHECK_FEATURE(cop3.DirectBiasAdjustment,       cop3Out.DirectBiasAdjustment,       ENC_FEATURE_DIRECT_BIAS_ADJUST,         MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_13);
        CHECK_FEATURE(cop3.GlobalMotionBiasAdjustment, cop3Out.GlobalMotionBiasAdjustment, ENC_FEATURE_GLOBAL_MOTION_ADJUST,       MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_13);
        videoPrm.mfx.GopRefDist = 1;
        CHECK_FEATURE(videoPrm.mfx.LowPower,     videoPrmOut.mfx.LowPower,     ENC_FEATURE_FIXED_FUNC,    MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_15);
        videoPrm.mfx.GopRefDist = 4;
        CHECK_FEATURE(cop3.WeightedPred,         cop3Out.WeightedPred,         ENC_FEATURE_WEIGHT_P,      MFX_WEIGHTED_PRED_DEFAULT,     MFX_LIB_VERSION_1_16);
        CHECK_FEATURE(cop3.WeightedBiPred,       cop3Out.WeightedBiPred,       ENC_FEATURE_WEIGHT_B,      MFX_WEIGHTED_PRED_DEFAULT,     MFX_LIB_VERSION_1_16);
        CHECK_FEATURE(cop3.FadeDetection,        cop3Out.FadeDetection,        ENC_FEATURE_FADE_DETECT,   MFX_CODINGOPTION_ON,           MFX_LIB_VERSION_1_17);
        cop2.ExtBRC = MFX_CODINGOPTION_ON;
        cop2.BitrateLimit = MFX_CODINGOPTION_OFF;
        CHECK_FEATURE(cop3.ExtBrcAdaptiveLTR,    cop3Out.ExtBrcAdaptiveLTR,    ENC_FEATURE_EXT_BRC_ADAPTIVE_LTR, MFX_CODINGOPTION_ON,    MFX_LIB_VERSION_1_26);
        cop2.ExtBRC = MFX_CODINGOPTION_UNKNOWN;
        cop2.BitrateLimit = MFX_CODINGOPTION_UNKNOWN;
        if (codecId == MFX_CODEC_HEVC) {
            CHECK_FEATURE(cop3.GPB,              cop3Out.GPB,                  ENC_FEATURE_DISABLE_GPB,       MFX_CODINGOPTION_ON,  MFX_LIB_VERSION_1_19);
            CHECK_FEATURE(cop3.EnableQPOffset,   cop3Out.EnableQPOffset,       ENC_FEATURE_PYRAMID_QP_OFFSET, MFX_CODINGOPTION_ON,  MFX_LIB_VERSION_1_19);
            videoPrm.mfx.FrameInfo.BitDepthLuma = 10;
            videoPrm.mfx.FrameInfo.BitDepthChroma = 10;
            videoPrm.mfx.FrameInfo.Shift = 1;
            videoPrm.mfx.CodecProfile = MFX_PROFILE_HEVC_MAIN10;
            CHECK_FEATURE(videoPrm.mfx.FrameInfo.FourCC, videoPrmOut.mfx.FrameInfo.FourCC, ENC_FEATURE_10BIT_DEPTH, MFX_FOURCC_P010, MFX_LIB_VERSION_1_19);
            videoPrm.mfx.FrameInfo.BitDepthLuma = 0;
            videoPrm.mfx.FrameInfo.BitDepthChroma = 0;
            videoPrm.mfx.FrameInfo.Shift = 0;
            videoPrm.mfx.CodecProfile = MFX_PROFILE_HEVC_MAIN;
            CHECK_FEATURE(cop3.TransformSkip, cop3Out.TransformSkip, ENC_FEATURE_HEVC_TSKIP, MFX_CODINGOPTION_ON, MFX_LIB_VERSION_1_26);
            CHECK_FEATURE(hevc.SampleAdaptiveOffset, hevcOut.SampleAdaptiveOffset, ENC_FEATURE_HEVC_SAO, MFX_SAO_ENABLE_LUMA, MFX_LIB_VERSION_1_26);
            CHECK_FEATURE(hevc.LCUSize, hevcOut.LCUSize, ENC_FEATURE_HEVC_CTU, 32, MFX_LIB_VERSION_1_26);
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
        auto cpu_gen = getCPUGen(&session);
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
static mfxU64 CheckEncodeFeatureStatic(mfxVersion mfxVer, int ratecontrol, mfxU32 codecId) {
    mfxU64 feature = 0x00;
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
    case MFX_RATECONTROL_LA_EXT:
        rate_control_supported = check_lib_version(mfxVer, MFX_LIB_VERSION_1_10);
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

mfxU64 CheckEncodeFeatureWithPluginLoad(MFXVideoSession& session, mfxVersion ver, int ratecontrol, mfxU32 codecId) {
    mfxU64 feature = 0x00;
    if (!check_lib_version(ver, MFX_LIB_VERSION_1_0)) {
        ; //特にすることはない
    } else if (!check_lib_version(ver, MFX_LIB_VERSION_1_6)) {
        //API v1.6未満で実際にチェックする必要は殆ど無いので、
        //コードで決められた値を返すようにする
        feature = CheckEncodeFeatureStatic(ver, ratecontrol, codecId);
    } else {

        CSessionPlugins sessionPlugins(session);
        auto plugin = getMFXPluginUID(MFXComponentType::ENCODE, codecId, false);
        if (plugin != nullptr) {
            sessionPlugins.LoadPlugin(MFX_PLUGINTYPE_VIDEO_ENCODE, *plugin, 1);
        }
        feature = CheckEncodeFeature(session, ver, ratecontrol, codecId);
        sessionPlugins.UnloadPlugins();
    }

    return feature;
}

const TCHAR *EncFeatureStr(mfxU64 enc_feature) {
    for (const FEATURE_DESC *ptr = list_enc_feature; ptr->desc; ptr++)
        if (enc_feature == (mfxU64)ptr->value)
            return ptr->desc;
    return NULL;
}

vector<mfxU64> MakeFeatureList(mfxVersion ver, const vector<CX_DESC>& rateControlList, mfxU32 codecId, std::shared_ptr<RGYLog> log) {
    vector<mfxU64> availableFeatureForEachRC;
    availableFeatureForEachRC.reserve(rateControlList.size());
#if LIBVA_SUPPORT
    if (codecId != MFX_CODEC_MPEG2) {
#endif
        MemType memType = HW_MEMORY;
        MFXVideoSession session;
        if (InitSession(session, true, memType) == MFX_ERR_NONE) {
            if (auto hwdevice = InitHWDevice(session, memType, log)) {
                for (const auto& ratecontrol : rateControlList) {
                    mfxU64 ret = CheckEncodeFeatureWithPluginLoad(session, ver, (mfxU16)ratecontrol.value, codecId);
                    if (ret == 0 && ratecontrol.value == MFX_RATECONTROL_CQP) {
                        ver = MFX_LIB_VERSION_0_0;
                    }
                    availableFeatureForEachRC.push_back(ret);
                }
            }
        }
#if LIBVA_SUPPORT
    }
#endif
    return availableFeatureForEachRC;
}

vector<vector<mfxU64>> MakeFeatureListPerCodec(mfxVersion ver, const vector<CX_DESC>& rateControlList, const vector<mfxU32>& codecIdList, std::shared_ptr<RGYLog> log) {
    vector<vector<mfxU64>> codecFeatures;
    vector<std::future<vector<mfxU64>>> futures;
    for (auto codec : codecIdList) {
        auto f = std::async(MakeFeatureList, ver, rateControlList, codec, log);
        futures.push_back(std::move(f));
    }
    for (uint32_t i = 0; i < futures.size(); i++) {
        codecFeatures.push_back(futures[i].get());
    }
    return codecFeatures;
}

vector<vector<mfxU64>> MakeFeatureListPerCodec(const vector<CX_DESC>& rateControlList, const vector<mfxU32>& codecIdList, std::shared_ptr<RGYLog> log) {
    mfxVersion ver = get_mfx_libhw_version();
    return MakeFeatureListPerCodec(ver, rateControlList, codecIdList, log);
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
    default:
        break;
    }

    return CheckDecFeaturesInternal(session, ver, codecId);
}

CodecCsp MakeDecodeFeatureList(mfxVersion ver, const vector<RGY_CODEC>& codecIdList, std::shared_ptr<RGYLog> log, const bool skipHWDecodeCheck) {
    CodecCsp codecFeatures;
    MFXVideoSession session;
    MemType memtype = HW_MEMORY;
    if (InitSession(session, true, memtype) == MFX_ERR_NONE) {
        if (auto hwdevice = InitHWDevice(session, memtype, log)) {
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
        }
    }
    return codecFeatures;
}

static const TCHAR *const QSV_FEATURE_MARK_YES_NO[] = { _T("×"), _T("○") };
static const TCHAR *const QSV_FEATURE_MARK_YES_NO_WITH_SPACE[] = { _T(" x    "), _T(" o    ") };

tstring MakeFeatureListStr(mfxU64 feature) {
    tstring str;
    for (const FEATURE_DESC *ptr = list_enc_feature; ptr->desc; ptr++) {
        str += ptr->desc;
        str += QSV_FEATURE_MARK_YES_NO_WITH_SPACE[!!(feature & ptr->value)];
        str += _T("\n");
    }
    str += _T("\n");
    return str;
}

vector<std::pair<vector<mfxU64>, tstring>> MakeFeatureListStr(FeatureListStrType type, const vector<mfxU32>& codecLists, std::shared_ptr<RGYLog> log) {
    auto featurePerCodec = MakeFeatureListPerCodec(make_vector(list_rate_control_ry), codecLists, log);

    vector<std::pair<vector<mfxU64>, tstring>> strPerCodec;

    for (mfxU32 i_codec = 0; i_codec < codecLists.size(); i_codec++) {
        tstring str;
        auto& availableFeatureForEachRC = featurePerCodec[i_codec];
        //H.264以外で、ひとつもフラグが立っていなかったら、スキップする
        if (codecLists[i_codec] != MFX_CODEC_AVC
            && 0 == std::accumulate(availableFeatureForEachRC.begin(), availableFeatureForEachRC.end(), 0,
            [](mfxU32 sum, mfxU64 value) { return sum | (mfxU32)(value & 0xffffffff) | (mfxU32)(value >> 32); })) {
            continue;
        }
        str += _T("Codec: ") + tstring(CodecIdToStr(codecLists[i_codec])) + _T("\n");

        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += _T("<table class=simpleOrange>");
        }

        switch (type) {
        case FEATURE_LIST_STR_TYPE_HTML: str += _T("<tr><th></th>"); break;
        case FEATURE_LIST_STR_TYPE_TXT:
        default:
            //ヘッダ部分
            const mfxU32 row_header_length = (mfxU32)_tcslen(list_enc_feature[0].desc);
            for (mfxU32 i = 1; i < row_header_length; i++)
                str += _T(" ");
            break;
        }

        for (mfxU32 i = 0; i < _countof(list_rate_control_ry); i++) {
            switch (type) {
            case FEATURE_LIST_STR_TYPE_CSV: str += _T(","); break;
            case FEATURE_LIST_STR_TYPE_HTML: str += _T("<th>"); break;
            case FEATURE_LIST_STR_TYPE_TXT:
            default: str += _T(" "); break;
            }
            str += list_rate_control_ry[i].desc;
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
            for (mfxU32 i = 0; i < _countof(list_rate_control_ry); i++) {
                if (type == FEATURE_LIST_STR_TYPE_HTML) {
                    str += !!(availableFeatureForEachRC[i] & ptr->value) ? _T("<td class=ok>") : _T("<td class=fail>");
                }
                if (type == FEATURE_LIST_STR_TYPE_TXT) {
                    str += QSV_FEATURE_MARK_YES_NO_WITH_SPACE[!!(availableFeatureForEachRC[i] & ptr->value)];
                } else {
                    str += QSV_FEATURE_MARK_YES_NO[!!(availableFeatureForEachRC[i] & ptr->value)];
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

vector<std::pair<vector<mfxU64>, tstring>> MakeFeatureListStr(FeatureListStrType type, std::shared_ptr<RGYLog> log) {
    const vector<mfxU32> codecLists = { MFX_CODEC_AVC, MFX_CODEC_HEVC, MFX_CODEC_MPEG2, MFX_CODEC_VP8, MFX_CODEC_VP9 };
    return MakeFeatureListStr(type, codecLists, log);
}

tstring MakeVppFeatureStr(FeatureListStrType type, std::shared_ptr<RGYLog> log) {
    mfxVersion ver = get_mfx_libhw_version();
    uint64_t features = CheckVppFeatures(ver, log);
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

tstring MakeDecFeatureStr(FeatureListStrType type, std::shared_ptr<RGYLog> log) {
#if ENABLE_AVSW_READER
    mfxVersion ver = get_mfx_libhw_version();
    vector<RGY_CODEC> codecLists;
    for (int i = 0; i < _countof(HW_DECODE_LIST); i++) {
        codecLists.push_back(HW_DECODE_LIST[i].rgy_codec);
    }
    auto decodeCodecCsp = MakeDecodeFeatureList(ver, codecLists, log, false);

    enum : uint32_t {
        DEC_FEATURE_HW    = 0x00000001,
        DEC_FEATURE_10BIT = 0x00000002,
    };

    static const FEATURE_DESC list_dec_feature[] = {
        { _T("HW Decode   "), DEC_FEATURE_HW    },
        { _T("10bit depth "), DEC_FEATURE_10BIT },
        { NULL, 0 },
    };

    std::vector<uint32_t> featurePerCodec;
    for (int i = 0; i < _countof(HW_DECODE_LIST); i++) {
        uint32_t feature = 0x00;
        if (decodeCodecCsp.count(HW_DECODE_LIST[i].rgy_codec) > 0) {
            feature |= DEC_FEATURE_HW;
            const auto& cspList = decodeCodecCsp.at(HW_DECODE_LIST[i].rgy_codec);
            for (auto csp : cspList) {
                if (RGY_CSP_BIT_DEPTH[csp] > 8) {
                    feature |= DEC_FEATURE_10BIT;
                }
            }
        }
        featurePerCodec.push_back(feature);
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
    for (const FEATURE_DESC *ptr = list_dec_feature; ptr->desc; ptr++) {
        maxFeatureStrLen = (std::max<int>)(maxFeatureStrLen, (int)_tcslen(ptr->desc));
    }

    if (type != FEATURE_LIST_STR_TYPE_HTML) {
        for (int i = 0; i < maxFeatureStrLen; i++) {
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
        while (codecStr.length() < 4) {
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

    for (const FEATURE_DESC *ptr = list_dec_feature; ptr->desc; ptr++) {
        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += _T("<tr><td>");
        }
        str += ptr->desc;
        switch (type) {
        case FEATURE_LIST_STR_TYPE_CSV: str += _T(","); break;
        case FEATURE_LIST_STR_TYPE_HTML: str += _T("</td>"); break;
        default: break;
        }
        for (uint32_t i_codec = 0; i_codec < codecLists.size(); i_codec++) {
            if (type == FEATURE_LIST_STR_TYPE_HTML) {
                str += (featurePerCodec[i_codec] & ptr->value) ? _T("<td class=ok>") : _T("<td class=fail>");
            }
            if (type == FEATURE_LIST_STR_TYPE_TXT) {
                str += MARK_YES_NO[ptr->value == (featurePerCodec[i_codec] & ptr->value)];
            } else {
                str += QSV_FEATURE_MARK_YES_NO[ptr->value == (featurePerCodec[i_codec] & ptr->value)];
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

CodecCsp getHWDecCodecCsp(std::shared_ptr<RGYLog> log, const bool skipHWDecodeCheck) {
#if ENABLE_AVSW_READER
    vector<RGY_CODEC> codecLists;
    for (int i = 0; i < _countof(HW_DECODE_LIST); i++) {
        codecLists.push_back(HW_DECODE_LIST[i].rgy_codec);
    }
    return MakeDecodeFeatureList(get_mfx_libhw_version(), codecLists, log, skipHWDecodeCheck);
#else
    return CodecCsp();
#endif
}

int getCPUGen() {
    mfxPlatform platform;
    memset(&platform, 0, sizeof(platform));
    MemType memtype = HW_MEMORY;
    MFXVideoSession session;
    InitSession(session, true, memtype);
    mfxVersion mfxVer;
    session.QueryVersion(&mfxVer);
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_19)) {
        session.QueryPlatform(&platform);
        return cpu_gen_enc_to_rgy(platform.CodeName);
    } else {
        return getCPUGenCpuid();
    }
}

int getCPUGen(MFXVideoSession *pSession) {
    if (pSession == nullptr || (mfxSession)(*pSession) == nullptr) {
        return getCPUGen();
    }
    mfxPlatform platform;
    memset(&platform, 0, sizeof(platform));
    mfxVersion mfxVer;
    pSession->QueryVersion(&mfxVer);
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_19)) {
        pSession->QueryPlatform(&platform);
        return cpu_gen_enc_to_rgy(platform.CodeName);
    } else {
        return getCPUGenCpuid();
    }
}
