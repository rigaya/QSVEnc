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
// ------------------------------------------------------------------------------------------

#include "rgy_tchar.h"
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <climits>
#include <deque>
#include <mutex>
#pragma warning(push)
#pragma warning(disable: 4244)
#pragma warning(disable: 4834)
#define TTMATH_NOASM
#include "ttmath/ttmath.h"
#pragma warning(pop)
#include "rgy_osdep.h"
#include "qsv_pipeline.h"
#include "qsv_pipeline_ctrl.h"
#include "qsv_query.h"
#include "rgy_input.h"
#include "rgy_output.h"
#include "rgy_input_raw.h"
#include "rgy_input_vpy.h"
#include "rgy_input_avs.h"
#include "rgy_input_avi.h"
#include "rgy_input_sm.h"
#include "rgy_input_avcodec.h"
#include "rgy_filter.h"
#include "rgy_filter_denoise_knn.h"
#include "rgy_output_avcodec.h"
#include "rgy_bitstream.h"
#include "qsv_hw_device.h"
#include "qsv_allocator.h"
#include "qsv_allocator_sys.h"
#include "rgy_avlog.h"
#include "rgy_chapter.h"
#include "rgy_timecode.h"
#include "rgy_codepage.h"
#if defined(_WIN32) || defined(_WIN64)
#include "api_hook.h"
#endif

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

#define RGY_ERR_MES(ret, MES)    {if (RGY_ERR_NONE > (ret)) { PrintMes(RGY_LOG_ERROR, _T("%s : %s\n"), MES, get_err_mes(ret)); return err_to_mfx(ret);}}
#define RGY_ERR(ret, MES)    {if (RGY_ERR_NONE > (ret)) { PrintMes(RGY_LOG_ERROR, _T("%s : %s\n"), MES, get_err_mes(ret)); return ret;}}
#define QSV_ERR_MES(sts, MES)    {if (MFX_ERR_NONE > (sts)) { PrintMes(RGY_LOG_ERROR, _T("%s : %s\n"), MES, get_err_mes((int)sts)); return sts;}}
#define CHECK_RANGE_LIST(value, list, name)    { if (CheckParamList((value), (list), (name)) != RGY_ERR_NONE) { return RGY_ERR_INVALID_VIDEO_PARAM; } }

int CQSVPipeline::clamp_param_int(int value, int low, int high, const TCHAR *param_name) {
    auto value_old = value;
    value = clamp(value, low, high);
    if (value != value_old) {
        PrintMes(RGY_LOG_WARN, _T("%s value changed %d -> %d, must be in range of %d-%d\n"), param_name, value_old, value, low, high);
    }
    return value;
}

bool CQSVPipeline::CompareParam(const mfxParamSet& prmIn, const mfxParamSet& prmOut) {
    bool ret = false;
#define COMPARE_INT(member, ignoreIfInput) { \
    if (prmIn.member != prmOut.member) { \
        ret = true;\
        PrintMes(((int64_t)prmIn.member == (int64_t)ignoreIfInput) ? RGY_LOG_DEBUG : RGY_LOG_WARN, _T("%s value changed %d -> %d by driver\n"), _T(#member), (int)prmIn.member, (int)prmOut.member); \
    }}
#define TRI_STATE(x) ((x == 0) ? _T("auto") : ((x == MFX_CODINGOPTION_ON) ? _T("on") : _T("off")))
#define COMPARE_TRI(member, ignoreIfInput) { \
    if (prmIn.member != prmOut.member) { \
        ret = true;\
        PrintMes((prmIn.member == ignoreIfInput) ? RGY_LOG_DEBUG : RGY_LOG_WARN, _T("%s value changed %s -> %s by driver\n"), _T(#member), TRI_STATE(prmIn.member), TRI_STATE(prmOut.member)); \
    }}
#define COMPARE_HEX(member, ignoreIfInput) { \
    if (prmIn.member != prmOut.member) { \
        ret = true;\
        PrintMes((prmIn.member == ignoreIfInput) ? RGY_LOG_DEBUG : RGY_LOG_WARN, _T("%s value changed 0x%x -> 0x%x by driver\n"), _T(#member), (int)prmIn.member, (int)prmOut.member); \
    }}
#define COMPARE_DBL(member, ignoreIfInput) { \
    if (prmIn.member != prmOut.member) { \
        ret = true;\
        PrintMes((prmIn.member == ignoreIfInput) ? RGY_LOG_DEBUG : RGY_LOG_WARN, _T("%s value changed %lf -> %lf by driver\n"), _T(#member), (double)prmIn.member, (double)prmOut.member); \
    }}
#define COMPARE_STR(member, ignoreIfInput, printMethod) { \
    if (prmIn.member != prmOut.member) { \
        ret = true;\
        PrintMes((prmIn.member == ignoreIfInput) ? RGY_LOG_DEBUG : RGY_LOG_WARN, _T("%s value changed %s -> %s by driver\n"), _T(#member), printMethod(prmIn.member), printMethod(prmOut.member)); \
    }}
#define COMPARE_LST(member, ignoreIfInput, list) { \
    if (prmIn.member != prmOut.member) { \
        ret = true;\
        PrintMes((prmIn.member == ignoreIfInput) ? RGY_LOG_DEBUG : RGY_LOG_WARN, _T("%s value changed %s -> %s by driver\n"), _T(#member), get_chr_from_value(list, prmIn.member), get_chr_from_value(list, prmOut.member)); \
    }}
    COMPARE_INT(vidprm.AsyncDepth,             0);
    COMPARE_HEX(vidprm.IOPattern,              0);
    COMPARE_INT(vidprm.mfx.NumThread,          0);
    COMPARE_INT(vidprm.mfx.BRCParamMultiplier, 0);
    COMPARE_INT(vidprm.mfx.LowPower,           0);
    COMPARE_STR(vidprm.mfx.CodecId,            0, CodecIdToStr);
    COMPARE_LST(vidprm.mfx.CodecProfile,       0, get_profile_list(prmIn.vidprm.mfx.CodecId));
    COMPARE_LST(vidprm.mfx.CodecLevel,         0, get_level_list(prmIn.vidprm.mfx.CodecId));
    COMPARE_INT(vidprm.mfx.NumThread,          0);
    COMPARE_INT(vidprm.mfx.TargetUsage,       -1);
    COMPARE_INT(vidprm.mfx.GopPicSize,         0);
    COMPARE_INT(vidprm.mfx.GopRefDist,         0);
    COMPARE_INT(vidprm.mfx.GopOptFlag,         0);
    COMPARE_INT(vidprm.mfx.IdrInterval,        0);
    COMPARE_STR(vidprm.mfx.RateControlMethod,  0, EncmodeToStr);
    if (prmIn.vidprm.mfx.RateControlMethod == MFX_RATECONTROL_CQP) {
        COMPARE_INT(vidprm.mfx.QPI, -1);
        COMPARE_INT(vidprm.mfx.QPP, -1);
        COMPARE_INT(vidprm.mfx.QPB, -1);
    } else if (rc_is_type_lookahead(m_mfxEncParams.mfx.RateControlMethod)) {
        COMPARE_INT(cop2.LookAheadDepth, -1);
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)) {
            COMPARE_LST(cop2.LookAheadDS, 0, list_lookahead_ds);
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_11)) {
            COMPARE_INT(cop3.WinBRCSize,       0);
            COMPARE_INT(cop3.WinBRCMaxAvgKbps, 0);
        }
        if (MFX_RATECONTROL_LA_ICQ == m_mfxEncParams.mfx.RateControlMethod) {
            COMPARE_INT(vidprm.mfx.ICQQuality, -1);
        }
    } else if (MFX_RATECONTROL_ICQ == m_mfxEncParams.mfx.RateControlMethod) {
        COMPARE_INT(vidprm.mfx.ICQQuality, -1);
    } else {
        COMPARE_INT(vidprm.mfx.TargetKbps, 0);
        if (m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_AVBR) {
            COMPARE_INT(vidprm.mfx.TargetKbps, 0);
        } else {
            COMPARE_INT(vidprm.mfx.MaxKbps, 0);
            if (m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_QVBR) {
                COMPARE_INT(cop3.QVBRQuality, -1);
            }
        }
    }
    COMPARE_INT(vidprm.mfx.NumSlice,             0);
    COMPARE_INT(vidprm.mfx.NumRefFrame,          0);
    COMPARE_INT(vidprm.mfx.EncodedOrder,         0);
    COMPARE_INT(vidprm.mfx.ExtendedPicStruct,    0);
    COMPARE_INT(vidprm.mfx.TimeStampCalc,        0);
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_6)) {
        COMPARE_INT(vidprm.mfx.SliceGroupsPresent, 0);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_15)) {
        COMPARE_TRI(vidprm.mfx.LowPower, 0);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_16)) {
        COMPARE_INT(vidprm.mfx.MaxDecFrameBuffering, 0);
    }

    COMPARE_TRI(cop.RateDistortionOpt,    0);
    COMPARE_INT(cop.MECostType,           0);
    COMPARE_INT(cop.MESearchType,         0);
    COMPARE_TRI(cop.EndOfSequence,        0);
    COMPARE_TRI(cop.FramePicture,         0);
    COMPARE_TRI(cop.CAVLC,                0);
    COMPARE_TRI(cop.ViewOutput,           0);
    COMPARE_TRI(cop.VuiVclHrdParameters,  0);
    COMPARE_TRI(cop.RefPicListReordering, 0);
    COMPARE_TRI(cop.ResetRefList,         0);
    COMPARE_INT(cop.MaxDecFrameBuffering, 0);
    COMPARE_TRI(cop.AUDelimiter,          0);
    COMPARE_TRI(cop.EndOfStream,          0);
    COMPARE_TRI(cop.PicTimingSEI,         0);
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_3)) {
        COMPARE_TRI(cop.RefPicMarkRep,       0);
        COMPARE_TRI(cop.FieldOutput,         0);
        COMPARE_TRI(cop.NalHrdConformance,   0);
        COMPARE_TRI(cop.SingleSeiNalUnit,    0);
        COMPARE_TRI(cop.VuiNalHrdParameters, 0);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_6)) {
        COMPARE_TRI(cop.RecoveryPointSEI, 0);

        COMPARE_INT(cop2.MaxFrameSize,    0);
        COMPARE_INT(cop2.MaxSliceSize,    0);
        COMPARE_TRI(cop2.BitrateLimit,    0);
        COMPARE_TRI(cop2.MBBRC,           0);
        COMPARE_TRI(cop2.ExtBRC,          0);
    }

    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)) {
        COMPARE_TRI(cop2.RepeatPPS,           0);
        COMPARE_INT(cop2.BRefType,            0);
        COMPARE_TRI(cop2.AdaptiveI,           0);
        COMPARE_TRI(cop2.AdaptiveB,           0);
        COMPARE_INT(cop2.NumMbPerSlice,       0);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_9)) {
        COMPARE_INT(cop2.MaxSliceSize,        0);
        COMPARE_INT(cop2.SkipFrame,           0);
        COMPARE_INT(cop2.MinQPI,              0);
        COMPARE_INT(cop2.MaxQPI,              0);
        COMPARE_INT(cop2.MinQPP,              0);
        COMPARE_INT(cop2.MaxQPP,              0);
        COMPARE_INT(cop2.MinQPB,              0);
        COMPARE_INT(cop2.MaxQPB,              0);
        COMPARE_INT(cop2.FixedFrameRate,      0);
        COMPARE_INT(cop2.DisableDeblockingIdc,0);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_10)) {
        COMPARE_INT(cop2.DisableVUI,         0);
        COMPARE_INT(cop2.BufferingPeriodSEI, 0);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_11)) {
        COMPARE_TRI(cop2.EnableMAD, 0);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_13)) {
        COMPARE_TRI(cop2.UseRawRef, 0);
    }

    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_11)) {
        COMPARE_INT(cop3.NumSliceI,                  0);
        COMPARE_INT(cop3.NumSliceP,                  0);
        COMPARE_INT(cop3.NumSliceB,                  0);
        if (rc_is_type_lookahead(m_mfxEncParams.mfx.RateControlMethod)) {
            COMPARE_INT(cop3.WinBRCMaxAvgKbps,       0);
            COMPARE_INT(cop3.WinBRCSize,             0);
        }
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_13)) {
        COMPARE_TRI(cop3.EnableMBQP,                 0);
        COMPARE_TRI(cop3.DirectBiasAdjustment,       0);
        COMPARE_TRI(cop3.GlobalMotionBiasAdjustment, 0);
        COMPARE_INT(cop3.MVCostScalingFactor,        0);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_16)) {
        COMPARE_INT(cop3.IntRefCycleDist,            0);
        COMPARE_TRI(cop3.MBDisableSkipMap,           0);
        COMPARE_INT(cop3.WeightedPred,               0);
        COMPARE_INT(cop3.WeightedBiPred,             0);
        COMPARE_TRI(cop3.AspectRatioInfoPresent,     0);
        COMPARE_TRI(cop3.OverscanInfoPresent,        0);
        COMPARE_TRI(cop3.OverscanAppropriate,        0);
        COMPARE_TRI(cop3.TimingInfoPresent,          0);
        COMPARE_TRI(cop3.BitstreamRestriction,       0);
        COMPARE_INT(cop3.PRefType,                   0);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_17)) {
        COMPARE_TRI(cop3.FadeDetection,              0);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_19)) {
        COMPARE_TRI(cop3.LowDelayHrd,                0);
        COMPARE_TRI(cop3.MotionVectorsOverPicBoundaries, 0);
        COMPARE_TRI(cop3.MaxFrameSizeI,      0);
        COMPARE_TRI(cop3.MaxFrameSizeP,      0);
        COMPARE_TRI(cop3.EnableQPOffset,     0);
        COMPARE_TRI(cop3.TransformSkip,      0);
        COMPARE_INT(cop3.QPOffset[0],        0);
        COMPARE_INT(cop3.QPOffset[1],        0);
        COMPARE_INT(cop3.QPOffset[2],        0);
        COMPARE_INT(cop3.NumRefActiveP[0],   0);
        COMPARE_INT(cop3.NumRefActiveP[1],   0);
        COMPARE_INT(cop3.NumRefActiveP[2],   0);
        COMPARE_INT(cop3.NumRefActiveBL0[0], 0);
        COMPARE_INT(cop3.NumRefActiveBL0[1], 0);
        COMPARE_INT(cop3.NumRefActiveBL0[2], 0);
        COMPARE_INT(cop3.NumRefActiveBL1[0], 0);
        COMPARE_INT(cop3.NumRefActiveBL1[1], 0);
        COMPARE_INT(cop3.NumRefActiveBL1[2], 0);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_26)) {
        COMPARE_TRI(hevc.SampleAdaptiveOffset,  MFX_SAO_UNKNOWN);
        COMPARE_TRI(hevc.LCUSize, 0);
    }
    return ret;
}

//範囲チェック
RGY_ERR CQSVPipeline::CheckParamList(int value, const CX_DESC *list, const char *param_name) {
    for (int i = 0; list[i].desc; i++)
        if (list[i].value == value)
            return RGY_ERR_NONE;
    PrintMes(RGY_LOG_ERROR, _T("%s=%d, is not valid param.\n"), param_name, value);
    return RGY_ERR_INVALID_VIDEO_PARAM;
};

RGY_ERR CQSVPipeline::InitMfxDecParams(sInputParams *pInParams) {
#if ENABLE_AVSW_READER
    RGY_ERR sts = RGY_ERR_NONE;
    if (m_pFileReader->getInputCodec()) {
        m_DecInputBitstream.init(AVCODEC_READER_INPUT_BUF_SIZE);
        //TimeStampはQSVに自動的に計算させる
        m_DecInputBitstream.setPts(MFX_TIMESTAMP_UNKNOWN);

        sts = m_pFileReader->GetHeader(&m_DecInputBitstream);
        RGY_ERR(sts, _T("InitMfxDecParams: Failed to get stream header from reader."));

        const bool bGotHeader = m_DecInputBitstream.size() > 0;
        if (!bGotHeader) {
            //最初のフレームそのものをヘッダーとして使用する。
            //ここで読み込みんだ第1フレームのデータを読み込み側から消してしまうと、
            //メインループでは第2フレームのデータがmfxBitstreamに追加されてしまい、
            //第1・第2フレームの両方のデータが存在することになってしまう。
            //VP8/VP9のデコードでは、mfxBitstreamに複数のフレームのデータがあるとうまく動作しないことがあるためこれを回避する。
            //ここで読み込んだ第1フレームは読み込み側から消さないようにすることで、
            //メインループで再び第1フレームのデータとして読み込むことができる。
            m_pFileReader->GetNextBitstreamNoDelete(&m_DecInputBitstream);
        }

        //デコーダの作成
        m_pmfxDEC.reset(new MFXVideoDECODE(m_mfxSession));
        if (!m_pmfxDEC) {
            return RGY_ERR_MEMORY_ALLOC;
        }

        static const auto codecPluginList = make_array<std::pair<RGY_CODEC, mfxPluginUID>>(
            std::make_pair(RGY_CODEC_HEVC, MFX_PLUGINID_HEVCD_HW),
            std::make_pair(RGY_CODEC_VP8,  MFX_PLUGINID_VP8D_HW),
            std::make_pair(RGY_CODEC_VP9,  MFX_PLUGINID_VP9D_HW)
        );
        const auto inputCodec = m_pFileReader->getInputCodec();
        const auto plugin = std::find_if(codecPluginList.begin(), codecPluginList.end(),
                [inputCodec](decltype((codecPluginList[0])) codecPlugin) {
            return codecPlugin.first == inputCodec;
        });
        if (plugin != codecPluginList.end()) {
            PrintMes(RGY_LOG_DEBUG, _T("InitMfxDecParams: Loading %s decoder plugin..."), CodecToStr(plugin->first).c_str());
            if (MFX_ERR_NONE != m_SessionPlugins->LoadPlugin(MFX_PLUGINTYPE_VIDEO_DECODE, plugin->second, 1)) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to load hw %s decoder.\n"), CodecToStr(plugin->first).c_str());
                return RGY_ERR_UNSUPPORTED;
            }
            PrintMes(RGY_LOG_DEBUG, _T("InitMfxDecParams: Loaded %s decoder plugin.\n"), CodecToStr(plugin->first).c_str());
        }

        if (m_pFileReader->getInputCodec() == RGY_CODEC_H264 || m_pFileReader->getInputCodec() == RGY_CODEC_HEVC) {
            //これを付加しないとMFXVideoDECODE_DecodeHeaderが成功しない
            const uint32_t IDR = 0x65010000;
            m_DecInputBitstream.append((uint8_t *)&IDR, sizeof(IDR));
        }
        memset(&m_mfxDecParams, 0, sizeof(m_mfxDecParams));
        m_mfxDecParams.mfx.CodecId = codec_rgy_to_enc(m_pFileReader->getInputCodec());
        m_mfxDecParams.IOPattern = (uint16_t)((pInParams->memType != SYSTEM_MEMORY) ? MFX_IOPATTERN_OUT_VIDEO_MEMORY : MFX_IOPATTERN_OUT_SYSTEM_MEMORY);
        sts = err_to_rgy(m_pmfxDEC->DecodeHeader(&m_DecInputBitstream.bitstream(), &m_mfxDecParams));
        QSV_ERR_MES(sts, _T("InitMfxDecParams: Failed to DecodeHeader."));

        //DecodeHeaderした結果をreaderにも反映
        //VPPにInputFrameInfoを渡す時などに、high bit depthなどの時にshiftの取得しておく必要がある
        auto inputVideoInfo = m_pFileReader->GetInputFrameInfo();
        m_mfxDecParams.mfx.FrameInfo.BitDepthLuma = (mfxU16)(RGY_CSP_BIT_DEPTH[inputVideoInfo.csp] - inputVideoInfo.shift);
        m_mfxDecParams.mfx.FrameInfo.BitDepthChroma = (mfxU16)(RGY_CSP_BIT_DEPTH[inputVideoInfo.csp] - inputVideoInfo.shift);
        if (m_mfxDecParams.mfx.FrameInfo.BitDepthLuma > 16 || m_mfxDecParams.mfx.FrameInfo.BitDepthChroma > 16) {
            PrintMes(RGY_LOG_DEBUG, _T("InitMfxDecParams: Invalid bitdepth.\n"));
            return RGY_ERR_INVALID_VIDEO_PARAM;
        }
        m_mfxDecParams.mfx.FrameInfo.Shift = inputVideoInfo.shift ? 1 : 0; //mfxFrameInfoのShiftはシフトすべきかどうかの 1 か 0 のみ。
        if (!check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_9)
            || (inputCodec != RGY_CODEC_VP8 && inputCodec != RGY_CODEC_VP9)) { // VP8/VP9ではこの処理は不要
            if (m_mfxDecParams.mfx.FrameInfo.BitDepthLuma == 8)   m_mfxDecParams.mfx.FrameInfo.BitDepthLuma = 0;
            if (m_mfxDecParams.mfx.FrameInfo.BitDepthChroma == 8) m_mfxDecParams.mfx.FrameInfo.BitDepthChroma = 0;
        }
        if (m_mfxDecParams.mfx.FrameInfo.Shift
            && m_mfxDecParams.mfx.FrameInfo.BitDepthLuma == 0
            && m_mfxDecParams.mfx.FrameInfo.BitDepthChroma == 0) {
            PrintMes(RGY_LOG_DEBUG, _T("InitMfxDecParams: Bit shift required but bitdepth not set.\n"));
            return RGY_ERR_INVALID_VIDEO_PARAM;
        }
        if (m_mfxDecParams.mfx.FrameInfo.FrameRateExtN == 0
            && m_mfxDecParams.mfx.FrameInfo.FrameRateExtD == 0) {
            auto inputFrameInfo = m_pFileReader->GetInputFrameInfo();
            if (inputFrameInfo.fpsN > 0 && inputFrameInfo.fpsD > 0) {
                m_mfxDecParams.mfx.FrameInfo.FrameRateExtN = inputFrameInfo.fpsN;
                m_mfxDecParams.mfx.FrameInfo.FrameRateExtD = inputFrameInfo.fpsD;
            }
        }

        if (!bGotHeader) {
            //最初のフレームそのものをヘッダーとして使用している場合、一度データをクリアする
            //メインループに入った際に再度第1フレームを読み込むようにする。
            m_DecInputBitstream.clear();
        }

        PrintMes(RGY_LOG_DEBUG, _T("")
            _T("InitMfxDecParams: QSVDec prm: %s, Level %d, Profile %d\n")
            _T("InitMfxDecParams: Frame: %s, %dx%d%s [%d,%d,%d,%d] %d:%d\n")
            _T("InitMfxDecParams: color format %s, chroma %s, bitdepth %d, shift %d, picstruct %s\n"),
            CodecIdToStr(m_mfxDecParams.mfx.CodecId), m_mfxDecParams.mfx.CodecLevel, m_mfxDecParams.mfx.CodecProfile,
            ColorFormatToStr(m_mfxDecParams.mfx.FrameInfo.FourCC), m_mfxDecParams.mfx.FrameInfo.Width, m_mfxDecParams.mfx.FrameInfo.Height,
            (m_mfxDecParams.mfx.FrameInfo.PicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF)) ? _T("i") : _T("p"),
            m_mfxDecParams.mfx.FrameInfo.CropX, m_mfxDecParams.mfx.FrameInfo.CropY, m_mfxDecParams.mfx.FrameInfo.CropW, m_mfxDecParams.mfx.FrameInfo.CropH,
            m_mfxDecParams.mfx.FrameInfo.AspectRatioW, m_mfxDecParams.mfx.FrameInfo.AspectRatioH,
            ColorFormatToStr(m_mfxDecParams.mfx.FrameInfo.FourCC), ChromaFormatToStr(m_mfxDecParams.mfx.FrameInfo.ChromaFormat),
            m_mfxDecParams.mfx.FrameInfo.BitDepthLuma, m_mfxDecParams.mfx.FrameInfo.Shift,
            MFXPicStructToStr(m_mfxDecParams.mfx.FrameInfo.PicStruct).c_str());

        memset(&m_DecVidProc, 0, sizeof(m_DecVidProc));
        m_DecExtParams.clear();
#if 0
        const auto enc_fourcc = csp_rgy_to_enc(getEncoderCsp(pInParams, nullptr));
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_23)
            && ( m_mfxDecParams.mfx.FrameInfo.CropW  != pInParams->input.dstWidth
              || m_mfxDecParams.mfx.FrameInfo.CropH  != pInParams->input.dstHeight
              || m_mfxDecParams.mfx.FrameInfo.FourCC != enc_fourcc)
            && pInParams->vpp.nScalingQuality == MFX_SCALING_MODE_LOWPOWER
            && enc_fourcc == MFX_FOURCC_NV12
            && m_mfxDecParams.mfx.FrameInfo.FourCC == MFX_FOURCC_NV12
            && m_mfxDecParams.mfx.FrameInfo.ChromaFormat == MFX_CHROMAFORMAT_YUV420
            && !cropEnabled(pInParams->sInCrop)) {
            m_DecVidProc.Header.BufferId = MFX_EXTBUFF_DEC_VIDEO_PROCESSING;
            m_DecVidProc.Header.BufferSz = sizeof(m_DecVidProc);
            m_DecVidProc.In.CropX = 0;
            m_DecVidProc.In.CropY = 0;
            m_DecVidProc.In.CropW = m_mfxDecParams.mfx.FrameInfo.CropW;
            m_DecVidProc.In.CropH = m_mfxDecParams.mfx.FrameInfo.CropH;

            m_DecVidProc.Out.FourCC = enc_fourcc;
            m_DecVidProc.Out.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
            m_DecVidProc.Out.Width  = std::max<mfxU16>(ALIGN16(pInParams->input.dstWidth), m_mfxDecParams.mfx.FrameInfo.Width);
            m_DecVidProc.Out.Height = std::max<mfxU16>(ALIGN16(pInParams->input.dstHeight), m_mfxDecParams.mfx.FrameInfo.Height);
            m_DecVidProc.Out.CropX = 0;
            m_DecVidProc.Out.CropY = 0;
            m_DecVidProc.Out.CropW = pInParams->input.dstWidth;
            m_DecVidProc.Out.CropH = pInParams->input.dstHeight;

            m_DecExtParams.push_back((mfxExtBuffer *)&m_DecVidProc);
            m_mfxDecParams.ExtParam = &m_DecExtParams[0];
            m_mfxDecParams.NumExtParam = (mfxU16)m_DecExtParams.size();

            pInParams->input.srcWidth = pInParams->input.dstWidth;
            pInParams->input.srcHeight = pInParams->input.dstHeight;
        }
#endif
    }
#endif
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::InitMfxEncodeParams(sInputParams *pInParams) {
    if (pInParams->CodecId == MFX_CODEC_RAW) {
        PrintMes(RGY_LOG_DEBUG, _T("Raw codec is selected, disable encode.\n"));
        return RGY_ERR_NONE;
    }
    const mfxU32 blocksz = (pInParams->CodecId == MFX_CODEC_HEVC) ? 32 : 16;
    auto print_feature_warnings = [this](int log_level, const TCHAR *feature_name) {
        PrintMes(log_level, _T("%s is not supported on current platform, disabled.\n"), feature_name);
    };

    if (pInParams->CodecId == MFX_CODEC_HEVC) {
        if (RGY_ERR_NONE != m_SessionPlugins->LoadPlugin(MFX_PLUGINTYPE_VIDEO_ENCODE, MFX_PLUGINID_HEVCE_HW, 1)) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to load hw hevc encoder.\n"));
            PrintMes(RGY_LOG_ERROR, _T("hevc encoding is not supported on current platform.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    } else if (pInParams->CodecId == MFX_CODEC_VP8) {
        if (RGY_ERR_NONE != m_SessionPlugins->LoadPlugin(MFX_PLUGINTYPE_VIDEO_ENCODE, MFX_PLUGINID_VP8E_HW, 1)) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to load hw vp8 encoder.\n"));
            PrintMes(RGY_LOG_ERROR, _T("vp8 encoding is not supported on current platform.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    } else if (pInParams->CodecId == MFX_CODEC_VP9) {
        if (RGY_ERR_NONE != m_SessionPlugins->LoadPlugin(MFX_PLUGINTYPE_VIDEO_ENCODE, MFX_PLUGINID_VP9E_HW, 1)) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to load hw vp9 encoder.\n"));
            PrintMes(RGY_LOG_ERROR, _T("vp9 encoding is not supported on current platform.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }
    const int encodeBitDepth = getEncoderBitdepth(pInParams);
    if (encodeBitDepth <= 0) {
        PrintMes(RGY_LOG_ERROR, _T("Unknown codec.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    const int codecMaxQP = 51 + (encodeBitDepth - 8) * 6;
    PrintMes(RGY_LOG_DEBUG, _T("encodeBitDepth: %d, codecMaxQP: %d.\n"), encodeBitDepth, codecMaxQP);

    //エンコードモードのチェック
    auto availableFeaures = CheckEncodeFeature(m_mfxSession, m_mfxVer, pInParams->nEncMode, pInParams->CodecId);
    PrintMes(RGY_LOG_DEBUG, _T("Detected avaliable features for hw API v%d.%d, %s, %s\n%s\n"),
        m_mfxVer.Major, m_mfxVer.Minor,
        CodecIdToStr(pInParams->CodecId), EncmodeToStr(pInParams->nEncMode), MakeFeatureListStr(availableFeaures).c_str());
    if (!(availableFeaures & ENC_FEATURE_CURRENT_RC)) {
        //このコーデックがサポートされているかどうか確認する
        if (   pInParams->nEncMode == MFX_RATECONTROL_CQP
            || pInParams->nEncMode == MFX_RATECONTROL_VBR
            || pInParams->nEncMode == MFX_RATECONTROL_CBR
            || !(CheckEncodeFeature(m_mfxSession, m_mfxVer, MFX_RATECONTROL_CQP, pInParams->CodecId) & ENC_FEATURE_CURRENT_RC)) {
            PrintMes(RGY_LOG_ERROR, _T("%s encoding is not supported on current platform.\n"), CodecIdToStr(pInParams->CodecId));
            return RGY_ERR_INVALID_VIDEO_PARAM;
        }
        const int rc_error_log_level = (pInParams->nFallback) ? RGY_LOG_WARN : RGY_LOG_ERROR;
        PrintMes(rc_error_log_level, _T("%s mode is not supported on current platform.\n"), EncmodeToStr(pInParams->nEncMode));
        if (MFX_RATECONTROL_LA == pInParams->nEncMode) {
            if (!check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_7)) {
                PrintMes(rc_error_log_level, _T("Lookahead mode is only supported by API v1.7 or later.\n"));
            }
        }
        if (   MFX_RATECONTROL_ICQ    == pInParams->nEncMode
            || MFX_RATECONTROL_LA_ICQ == pInParams->nEncMode
            || MFX_RATECONTROL_VCM    == pInParams->nEncMode) {
            if (!check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)) {
                PrintMes(rc_error_log_level, _T("%s mode is only supported by API v1.8 or later.\n"), EncmodeToStr(pInParams->nEncMode));
            }
        }
        if (   MFX_RATECONTROL_LA_EXT == pInParams->nEncMode
            || MFX_RATECONTROL_LA_HRD == pInParams->nEncMode
            || MFX_RATECONTROL_QVBR   == pInParams->nEncMode) {
            if (!check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_11)) {
                PrintMes(rc_error_log_level, _T("%s mode is only supported by API v1.11 or later.\n"), EncmodeToStr(pInParams->nEncMode));
            }
        }
        if (!pInParams->nFallback) {
            return RGY_ERR_INVALID_VIDEO_PARAM;
        }
        //fallback
        //fallbackの候補リスト、優先度の高い順にセットする
        vector<int> check_rc_list;
        //現在のレート制御モードは使用できないので、それ以外を確認する
        auto check_rc_add = [pInParams, &check_rc_list](int rc_mode) {
            if (pInParams->nEncMode != rc_mode) {
                check_rc_list.push_back(rc_mode);
            }
        };

        //品質指定系の場合、若干補正をかけた値を設定する
        int nAdjustedQP[3] = { QSV_DEFAULT_QPI, QSV_DEFAULT_QPP, QSV_DEFAULT_QPB };
        if (isRCBitrateMode(pInParams->nEncMode)) {
            //ビットレートモードなら、QVBR->VBRをチェックする
            check_rc_add(MFX_RATECONTROL_QVBR);
            check_rc_add(MFX_RATECONTROL_VBR);
        } else {
            //固定品質モードなら、ICQ->CQPをチェックする
            check_rc_add(MFX_RATECONTROL_ICQ);
            check_rc_add(MFX_RATECONTROL_CQP);
            //品質指定系の場合、若干補正をかけた値を設定する
            if (pInParams->nEncMode == MFX_RATECONTROL_LA_ICQ) {
                nAdjustedQP[0] = pInParams->nICQQuality - 8;
                nAdjustedQP[1] = pInParams->nICQQuality - 6;
                nAdjustedQP[2] = pInParams->nICQQuality - 3;
            } else if (pInParams->nEncMode == MFX_RATECONTROL_ICQ) {
                nAdjustedQP[0] = pInParams->nICQQuality - 1;
                nAdjustedQP[1] = pInParams->nICQQuality + 1;
                nAdjustedQP[2] = pInParams->nICQQuality + 4;
            } else if (pInParams->nEncMode == MFX_RATECONTROL_CQP) {
                nAdjustedQP[0] = pInParams->nQPI;
                nAdjustedQP[1] = pInParams->nQPP;
                nAdjustedQP[2] = pInParams->nQPB;
            }
        }
        //check_rc_listに設定したfallbackの候補リストをチェックする
        bool bFallbackSuccess = false;
        for (uint32_t i = 0; i < (uint32_t)check_rc_list.size(); i++) {
            auto availRCFeatures = CheckEncodeFeature(m_mfxSession, m_mfxVer, (uint16_t)check_rc_list[i], pInParams->CodecId);
            if (availRCFeatures & ENC_FEATURE_CURRENT_RC) {
                pInParams->nEncMode = (uint16_t)check_rc_list[i];
                if (pInParams->nEncMode == MFX_RATECONTROL_LA_ICQ) {
                    pInParams->nICQQuality = (uint16_t)clamp(nAdjustedQP[1] + 6, 1, codecMaxQP);
                } else if (pInParams->nEncMode == MFX_RATECONTROL_LA_ICQ) {
                    pInParams->nICQQuality = (uint16_t)clamp(nAdjustedQP[1], 1, codecMaxQP);
                } else if (pInParams->nEncMode == MFX_RATECONTROL_CQP) {
                    pInParams->nQPI = (uint16_t)clamp(nAdjustedQP[0], 0, codecMaxQP);
                    pInParams->nQPP = (uint16_t)clamp(nAdjustedQP[1], 0, codecMaxQP);
                    pInParams->nQPB = (uint16_t)clamp(nAdjustedQP[2], 0, codecMaxQP);
                }
                bFallbackSuccess = true;
                availableFeaures = availRCFeatures;
                PrintMes(rc_error_log_level, _T("Falling back to %s mode.\n"), EncmodeToStr(pInParams->nEncMode));
                break;
            }
        }
        //なんらかの理由でフォールバックできなかったらエラー終了
        if (!bFallbackSuccess) {
            return RGY_ERR_INVALID_VIDEO_PARAM;
        }
    }
    if (pInParams->nBframes == QSV_BFRAMES_AUTO) {
        pInParams->nBframes = (pInParams->CodecId == MFX_CODEC_HEVC) ? QSV_DEFAULT_HEVC_BFRAMES : QSV_DEFAULT_H264_BFRAMES;
    }
    //その他機能のチェック
    if (pInParams->bAdaptiveI && !(availableFeaures & ENC_FEATURE_ADAPTIVE_I)) {
        PrintMes(RGY_LOG_WARN, _T("Adaptve I-frame insert is not supported on current platform, disabled.\n"));
        pInParams->bAdaptiveI = false;
    }
    if (pInParams->bAdaptiveB && !(availableFeaures & ENC_FEATURE_ADAPTIVE_B)) {
        PrintMes(RGY_LOG_WARN, _T("Adaptve B-frame insert is not supported on current platform, disabled.\n"));
        pInParams->bAdaptiveB = false;
    }
    if (pInParams->bBPyramid && !(availableFeaures & ENC_FEATURE_B_PYRAMID)) {
        print_feature_warnings(RGY_LOG_WARN, _T("B pyramid"));
        pInParams->bBPyramid = false;
    }
    if (pInParams->bCAVLC && !(availableFeaures & ENC_FEATURE_CAVLC)) {
        print_feature_warnings(RGY_LOG_WARN, _T("CAVLC"));
        pInParams->bCAVLC = false;
    }
    if (pInParams->extBRC && !(availableFeaures & ENC_FEATURE_EXT_BRC)) {
        print_feature_warnings(RGY_LOG_WARN, _T("ExtBRC"));
        pInParams->extBRC = false;
    }
    if (pInParams->extBrcAdaptiveLTR && !(availableFeaures & ENC_FEATURE_EXT_BRC_ADAPTIVE_LTR)) {
        print_feature_warnings(RGY_LOG_WARN, _T("AdaptiveLTR"));
        pInParams->extBrcAdaptiveLTR = false;
    }
    if (pInParams->bMBBRC && !(availableFeaures & ENC_FEATURE_MBBRC)) {
        print_feature_warnings(RGY_LOG_WARN, _T("MBBRC"));
        pInParams->bMBBRC = false;
    }
    if (   (MFX_RATECONTROL_LA     == pInParams->nEncMode
         || MFX_RATECONTROL_LA_ICQ == pInParams->nEncMode)
        && pInParams->nLookaheadDS != MFX_LOOKAHEAD_DS_UNKNOWN
        && !(availableFeaures & ENC_FEATURE_LA_DS)) {
        print_feature_warnings(RGY_LOG_WARN, _T("Lookahead qaulity setting"));
        pInParams->nLookaheadDS = MFX_LOOKAHEAD_DS_UNKNOWN;
    }
    if (pInParams->nTrellis != MFX_TRELLIS_UNKNOWN && !(availableFeaures & ENC_FEATURE_TRELLIS)) {
        print_feature_warnings(RGY_LOG_WARN, _T("trellis"));
        pInParams->nTrellis = MFX_TRELLIS_UNKNOWN;
    }
    if (pInParams->bRDO && !(availableFeaures & ENC_FEATURE_RDO)) {
        print_feature_warnings(RGY_LOG_WARN, _T("RDO"));
        pInParams->bRDO = false;
    }
    if (((m_encPicstruct & RGY_PICSTRUCT_INTERLACED) != 0)
        && !(availableFeaures & ENC_FEATURE_INTERLACE)) {
        PrintMes(RGY_LOG_ERROR, _T("Interlaced encoding is not supported on current rate control mode.\n"));
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }
    if (pInParams->CodecId == MFX_CODEC_AVC
        && ((m_encPicstruct & RGY_PICSTRUCT_INTERLACED) != 0)
        && pInParams->nBframes > 0
        && getCPUGen(&m_mfxSession) == CPU_GEN_HASWELL
        && m_memType == D3D11_MEMORY) {
        PrintMes(RGY_LOG_WARN, _T("H.264 interlaced encoding with B frames on d3d11 mode results fuzzy outputs on Haswell CPUs.\n"));
        PrintMes(RGY_LOG_WARN, _T("B frames will be disabled.\n"));
        pInParams->nBframes = 0;
    }
    //最近のドライバでは問題ない模様
    //if (pInParams->nBframes > 2 && pInParams->CodecId == MFX_CODEC_HEVC) {
    //    PrintMes(RGY_LOG_WARN, _T("HEVC encoding + B-frames > 2 might cause artifacts, please check the output.\n"));
    //}
    if (pInParams->bBPyramid && pInParams->nBframes >= 10 && !(availableFeaures & ENC_FEATURE_B_PYRAMID_MANY_BFRAMES)) {
        PrintMes(RGY_LOG_WARN, _T("B pyramid with too many bframes is not supported on current platform, B pyramid disabled.\n"));
        pInParams->bBPyramid = false;
    }
    if (pInParams->bBPyramid && getCPUGen(&m_mfxSession) < CPU_GEN_HASWELL) {
        PrintMes(RGY_LOG_WARN, _T("B pyramid on IvyBridge generation might cause artifacts, please check your encoded video.\n"));
    }
    if (pInParams->bNoDeblock && !(availableFeaures & ENC_FEATURE_NO_DEBLOCK)) {
        print_feature_warnings(RGY_LOG_WARN, _T("No deblock"));
        pInParams->bNoDeblock = false;
    }
    if (pInParams->bIntraRefresh && !(availableFeaures & ENC_FEATURE_INTRA_REFRESH)) {
        print_feature_warnings(RGY_LOG_WARN, _T("Intra Refresh"));
        pInParams->bIntraRefresh = false;
    }
    if (0 != (pInParams->nQPMin[0] | pInParams->nQPMin[1] | pInParams->nQPMin[2]
            | pInParams->nQPMax[0] | pInParams->nQPMax[1] | pInParams->nQPMax[2]) && !(availableFeaures & ENC_FEATURE_QP_MINMAX)) {
        print_feature_warnings(RGY_LOG_WARN, _T("Min/Max QP"));
        memset(pInParams->nQPMin, 0, sizeof(pInParams->nQPMin));
        memset(pInParams->nQPMax, 0, sizeof(pInParams->nQPMax));
    }
    if (0 != pInParams->nWinBRCSize) {
        if (!(availableFeaures & ENC_FEATURE_WINBRC)) {
            print_feature_warnings(RGY_LOG_WARN, _T("WinBRC"));
            pInParams->nWinBRCSize = 0;
        } else if (0 == pInParams->nMaxBitrate) {
            print_feature_warnings(RGY_LOG_WARN, _T("Min/Max QP"));
            PrintMes(RGY_LOG_WARN, _T("WinBRC requires Max bitrate to be set, disabled.\n"));
            pInParams->nWinBRCSize = 0;
        }
    }
    if (pInParams->bDirectBiasAdjust && !(availableFeaures & ENC_FEATURE_DIRECT_BIAS_ADJUST)) {
        print_feature_warnings(RGY_LOG_WARN, _T("Direct Bias Adjust"));
        pInParams->bDirectBiasAdjust = 0;
    }
    if (pInParams->bGlobalMotionAdjust && !(availableFeaures & ENC_FEATURE_GLOBAL_MOTION_ADJUST)) {
        print_feature_warnings(RGY_LOG_WARN, _T("MV Cost Scaling"));
        pInParams->bGlobalMotionAdjust = 0;
        pInParams->nMVCostScaling = 0;
    }
    if (pInParams->bUseFixedFunc && !(availableFeaures & ENC_FEATURE_FIXED_FUNC)) {
        print_feature_warnings(RGY_LOG_WARN, _T("Fixed Func"));
        pInParams->bUseFixedFunc = 0;
    }
    if (pInParams->nWeightP && !(availableFeaures & ENC_FEATURE_WEIGHT_P)) {
        if (pInParams->nWeightP == MFX_CODINGOPTION_ON) {
            print_feature_warnings(RGY_LOG_WARN, _T("WeightP"));
        }
        pInParams->nWeightP = 0;
    }
    if (pInParams->nWeightB && !(availableFeaures & ENC_FEATURE_WEIGHT_B)) {
        if (pInParams->nWeightB == MFX_CODINGOPTION_ON) {
            print_feature_warnings(RGY_LOG_WARN, _T("WeightB"));
        }
        pInParams->nWeightB = 0;
    }
#if !ENABLE_FADE_DETECT
    if (pInParams->nFadeDetect == MFX_CODINGOPTION_ON) {
        PrintMes(RGY_LOG_WARN, _T("fade-detect will be disabled due to instability.\n"));
        pInParams->nFadeDetect = MFX_CODINGOPTION_UNKNOWN;
    }
#endif
    if (pInParams->nFadeDetect != MFX_CODINGOPTION_UNKNOWN && !(availableFeaures & ENC_FEATURE_FADE_DETECT)) {
        if (pInParams->nFadeDetect == MFX_CODINGOPTION_ON) {
            print_feature_warnings(RGY_LOG_WARN, _T("FadeDetect"));
        }
        pInParams->nFadeDetect = MFX_CODINGOPTION_UNKNOWN;
    }
    if (pInParams->CodecId == MFX_CODEC_HEVC) {
        if (pInParams->hevc_ctu > 0 && !(availableFeaures & ENC_FEATURE_HEVC_CTU)) {
            print_feature_warnings(RGY_LOG_WARN, _T("HEVC CTU"));
            pInParams->hevc_ctu = 0;
        }
        if (pInParams->hevc_sao != MFX_SAO_UNKNOWN && !(availableFeaures & ENC_FEATURE_HEVC_SAO)) {
            print_feature_warnings(RGY_LOG_WARN, _T("HEVC SAO"));
            pInParams->hevc_sao = MFX_SAO_UNKNOWN;
        }
        if (pInParams->hevc_tskip != MFX_CODINGOPTION_UNKNOWN && !(availableFeaures & ENC_FEATURE_HEVC_TSKIP)) {
            print_feature_warnings(RGY_LOG_WARN, _T("HEVC tskip"));
            pInParams->hevc_tskip = MFX_CODINGOPTION_UNKNOWN;
        }
    }
    bool bQPOffsetUsed = false;
    std::for_each(pInParams->pQPOffset, pInParams->pQPOffset + _countof(pInParams->pQPOffset), [&bQPOffsetUsed](decltype(pInParams->pQPOffset[0]) v){ bQPOffsetUsed |= (v != 0); });
    if (bQPOffsetUsed && !(availableFeaures & ENC_FEATURE_PYRAMID_QP_OFFSET)) {
        print_feature_warnings(RGY_LOG_WARN, _T("QPOffset"));
        memset(pInParams->pQPOffset, 0, sizeof(pInParams->pQPOffset));
        bQPOffsetUsed = false;
    }

    if (!(availableFeaures & ENC_FEATURE_VUI_INFO)) {
        if (m_encVUI.colorrange == RGY_COLORRANGE_FULL) {
            print_feature_warnings(RGY_LOG_WARN, _T("fullrange"));
            m_encVUI.colorrange = RGY_COLORRANGE_UNSPECIFIED;
        }
        if (m_encVUI.transfer != get_cx_value(list_transfer, _T("undef"))) {
            print_feature_warnings(RGY_LOG_WARN, _T("transfer"));
            m_encVUI.transfer = (CspTransfer)get_cx_value(list_transfer, _T("undef"));
        }
        if (m_encVUI.format != get_cx_value(list_videoformat, _T("undef"))) {
            print_feature_warnings(RGY_LOG_WARN, _T("videoformat"));
            m_encVUI.format = get_cx_value(list_videoformat, _T("undef"));
        }
        if (m_encVUI.matrix != get_cx_value(list_colormatrix, _T("undef"))) {
            print_feature_warnings(RGY_LOG_WARN, _T("colormatrix"));
            m_encVUI.matrix = (CspMatrix)get_cx_value(list_colormatrix, _T("undef"));
        }
        if (m_encVUI.colorprim != get_cx_value(list_colorprim, _T("undef"))) {
            print_feature_warnings(RGY_LOG_WARN, _T("colorprim"));
            m_encVUI.colorprim = (CspColorprim)get_cx_value(list_colorprim, _T("undef"));
        }
    }
    m_encVUI.descriptpresent =
           (int)m_encVUI.matrix != get_cx_value(list_colormatrix, _T("undef"))
        || (int)m_encVUI.colorprim != get_cx_value(list_colorprim, _T("undef"))
        || (int)m_encVUI.transfer != get_cx_value(list_transfer, _T("undef"));

    //Intra Refereshが指定された場合は、GOP関連の設定を自動的に上書き
    if (pInParams->bIntraRefresh) {
        pInParams->bforceGOPSettings = true;
    }
    //profileを守るための調整
    if (pInParams->CodecProfile == MFX_PROFILE_AVC_BASELINE) {
        pInParams->nBframes = 0;
        pInParams->bCAVLC = true;
    }
    if (pInParams->bCAVLC) {
        pInParams->bRDO = false;
    }

    CHECK_RANGE_LIST(pInParams->CodecId,      list_codec,   "codec");
    CHECK_RANGE_LIST(pInParams->CodecLevel,   get_level_list(pInParams->CodecId),   "level");
    CHECK_RANGE_LIST(pInParams->CodecProfile, get_profile_list(pInParams->CodecId), "profile");
    CHECK_RANGE_LIST(pInParams->nEncMode,     list_rc_mode, "rc mode");

    //設定開始
    m_mfxEncParams.mfx.CodecId                 = pInParams->CodecId;
    m_mfxEncParams.mfx.RateControlMethod       = (mfxU16)pInParams->nEncMode;
    if (MFX_RATECONTROL_CQP == m_mfxEncParams.mfx.RateControlMethod) {
        //CQP
        m_mfxEncParams.mfx.QPI             = (mfxU16)clamp_param_int(pInParams->nQPI, 0, codecMaxQP, _T("qp-i"));
        m_mfxEncParams.mfx.QPP             = (mfxU16)clamp_param_int(pInParams->nQPP, 0, codecMaxQP, _T("qp-p"));
        m_mfxEncParams.mfx.QPB             = (mfxU16)clamp_param_int(pInParams->nQPB, 0, codecMaxQP, _T("qp-b"));
    } else if (MFX_RATECONTROL_ICQ    == m_mfxEncParams.mfx.RateControlMethod
            || MFX_RATECONTROL_LA_ICQ == m_mfxEncParams.mfx.RateControlMethod) {
        m_mfxEncParams.mfx.ICQQuality      = (mfxU16)clamp_param_int(pInParams->nICQQuality, 1, codecMaxQP, _T("icq"));
        m_mfxEncParams.mfx.MaxKbps         = 0;
    } else {
        auto maxBitrate = (std::max)((std::max)(pInParams->nBitRate, pInParams->nMaxBitrate),
            pInParams->VBVBufsize / 8 /*これはbyte単位の指定*/);
        if (maxBitrate > USHRT_MAX) {
            m_mfxEncParams.mfx.BRCParamMultiplier = (mfxU16)(maxBitrate / USHRT_MAX) + 1;
            pInParams->nBitRate    /= m_mfxEncParams.mfx.BRCParamMultiplier;
            pInParams->nMaxBitrate /= m_mfxEncParams.mfx.BRCParamMultiplier;
            pInParams->VBVBufsize  /= m_mfxEncParams.mfx.BRCParamMultiplier;
        }
        m_mfxEncParams.mfx.TargetKbps      = (mfxU16)pInParams->nBitRate; // in kbps
        if (m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_AVBR) {
            //AVBR
            //m_mfxEncParams.mfx.Accuracy        = pInParams->nAVBRAccuarcy;
            m_mfxEncParams.mfx.Accuracy        = 500;
            m_mfxEncParams.mfx.Convergence     = (mfxU16)pInParams->nAVBRConvergence;
        } else {
            //CBR, VBR
            m_mfxEncParams.mfx.MaxKbps         = (mfxU16)pInParams->nMaxBitrate;
            m_mfxEncParams.mfx.BufferSizeInKB  = (mfxU16)pInParams->VBVBufsize / 8; //これはbyte単位の指定
            m_mfxEncParams.mfx.InitialDelayInKB = m_mfxEncParams.mfx.BufferSizeInKB / 2;
        }
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_15)) {
        m_mfxEncParams.mfx.LowPower = (mfxU16)((pInParams->bUseFixedFunc) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF);
    }
    m_mfxEncParams.mfx.TargetUsage             = (mfxU16)clamp_param_int(pInParams->nTargetUsage, MFX_TARGETUSAGE_BEST_QUALITY, MFX_TARGETUSAGE_BEST_SPEED, _T("quality")); // trade-off between quality and speed

    PrintMes(RGY_LOG_DEBUG, _T("InitMfxEncParams: Output FPS %d/%d\n"), m_encFps.n(), m_encFps.d());
    if (pInParams->nGOPLength == 0) {
        pInParams->nGOPLength = (mfxU16)((m_encFps.n() + m_encFps.d() - 1) / m_encFps.d()) * 10;
        PrintMes(RGY_LOG_DEBUG, _T("InitMfxEncParams: Auto GOP Length: %d\n"), pInParams->nGOPLength);
    }
    m_mfxEncParams.mfx.FrameInfo.FrameRateExtN = m_encFps.n();
    m_mfxEncParams.mfx.FrameInfo.FrameRateExtD = m_encFps.d();
    m_mfxEncParams.mfx.EncodedOrder            = 0;
    m_mfxEncParams.mfx.NumSlice                = (mfxU16)pInParams->nSlices;

    m_mfxEncParams.mfx.NumRefFrame             = (mfxU16)clamp_param_int(pInParams->nRef, 0, 16, _T("ref"));
    m_mfxEncParams.mfx.CodecLevel              = (mfxU16)pInParams->CodecLevel;
    m_mfxEncParams.mfx.CodecProfile            = (mfxU16)pInParams->CodecProfile;
    m_mfxEncParams.mfx.GopOptFlag              = 0;
    m_mfxEncParams.mfx.GopOptFlag             |= (!pInParams->bopenGOP) ? MFX_GOP_CLOSED : 0x00;

    /* For H.264, IdrInterval specifies IDR-frame interval in terms of I-frames; if IdrInterval = 0, then every I-frame is an IDR-frame. If IdrInterval = 1, then every other I-frame is an IDR-frame, etc.
     * For HEVC, if IdrInterval = 0, then only first I-frame is an IDR-frame. If IdrInterval = 1, then every I-frame is an IDR-frame. If IdrInterval = 2, then every other I-frame is an IDR-frame, etc.
     * For MPEG2, IdrInterval defines sequence header interval in terms of I-frames. If IdrInterval = N, SDK inserts the sequence header before every Nth I-frame. If IdrInterval = 0 (default), SDK inserts the sequence header once at the beginning of the stream.
     * If GopPicSize or GopRefDist is zero, IdrInterval is undefined. */
    if (pInParams->CodecId == MFX_CODEC_HEVC) {
        m_mfxEncParams.mfx.IdrInterval = (mfxU16)((!pInParams->bopenGOP) ? 1 : 1 + ((m_encFps.n() + m_encFps.d() - 1) / m_encFps.d()) * 20 / pInParams->nGOPLength);
    } else if (pInParams->CodecId == MFX_CODEC_AVC) {
        m_mfxEncParams.mfx.IdrInterval = (mfxU16)((!pInParams->bopenGOP) ? 0 : ((m_encFps.n() + m_encFps.d() - 1) / m_encFps.d()) * 20 / pInParams->nGOPLength);
    } else {
        m_mfxEncParams.mfx.IdrInterval = 0;
    }
    //MFX_GOP_STRICTにより、インタレ保持時にフレームが壊れる場合があるため、無効とする
    //m_mfxEncParams.mfx.GopOptFlag             |= (pInParams->bforceGOPSettings) ? MFX_GOP_STRICT : NULL;

    m_mfxEncParams.mfx.GopPicSize              = (pInParams->bIntraRefresh) ? 0 : (mfxU16)pInParams->nGOPLength;
    m_mfxEncParams.mfx.GopRefDist              = (mfxU16)(clamp_param_int(pInParams->nBframes, -1, 16, _T("bframes")) + 1);

    // specify memory type
    m_mfxEncParams.IOPattern = (mfxU16)((pInParams->memType != SYSTEM_MEMORY) ? MFX_IOPATTERN_IN_VIDEO_MEMORY : MFX_IOPATTERN_IN_SYSTEM_MEMORY);

    // frame info parameters
    m_mfxEncParams.mfx.FrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
    m_mfxEncParams.mfx.FrameInfo.PicStruct    = picstruct_rgy_to_enc(m_encPicstruct);

    // set sar info
    auto par = std::make_pair(pInParams->nPAR[0], pInParams->nPAR[1]);
    if ((!pInParams->nPAR[0] || !pInParams->nPAR[1]) //SAR比の指定がない
        && pInParams->input.sar[0] && pInParams->input.sar[1] //入力側からSAR比を取得ずみ
        && (pInParams->input.dstWidth == pInParams->input.srcWidth && pInParams->input.dstHeight == pInParams->input.srcHeight)) {//リサイズは行われない
        par = std::make_pair(pInParams->input.sar[0], pInParams->input.sar[1]);
    }
    adjust_sar(&par.first, &par.second, pInParams->input.dstWidth, pInParams->input.dstHeight);
    m_mfxEncParams.mfx.FrameInfo.AspectRatioW = (mfxU16)par.first;
    m_mfxEncParams.mfx.FrameInfo.AspectRatioH = (mfxU16)par.second;

    RGY_MEMSET_ZERO(m_CodingOption);
    m_CodingOption.Header.BufferId = MFX_EXTBUFF_CODING_OPTION;
    m_CodingOption.Header.BufferSz = sizeof(mfxExtCodingOption);
    //if (!pInParams->bUseHWLib) {
    //    //swライブラリ使用時のみ
    //    m_CodingOption.InterPredBlockSize = pInParams->nInterPred;
    //    m_CodingOption.IntraPredBlockSize = pInParams->nIntraPred;
    //    m_CodingOption.MVSearchWindow     = pInParams->MVSearchWindow;
    //    m_CodingOption.MVPrecision        = pInParams->nMVPrecision;
    //}
    //if (!pInParams->bUseHWLib || pInParams->CodecProfile == MFX_PROFILE_AVC_BASELINE) {
    //    //swライブラリ使用時かbaselineを指定した時
    //    m_CodingOption.RateDistortionOpt  = (mfxU16)((pInParams->bRDO) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_UNKNOWN);
    //    m_CodingOption.CAVLC              = (mfxU16)((pInParams->bCAVLC) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_UNKNOWN);
    //}
    //m_CodingOption.FramePicture = MFX_CODINGOPTION_ON;
    //m_CodingOption.FieldOutput = MFX_CODINGOPTION_ON;
    //m_CodingOption.VuiVclHrdParameters = MFX_CODINGOPTION_ON;
    //m_CodingOption.VuiNalHrdParameters = MFX_CODINGOPTION_ON;
    m_CodingOption.AUDelimiter = (mfxU16)((pInParams->bOutputAud) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF);
    m_CodingOption.PicTimingSEI = (mfxU16)((pInParams->bOutputPicStruct) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF);
    //m_CodingOption.SingleSeiNalUnit = MFX_CODINGOPTION_OFF;

    //API v1.6の機能
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_6)) {
        INIT_MFX_EXT_BUFFER(m_CodingOption2, MFX_EXTBUFF_CODING_OPTION2);
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)) {
            m_CodingOption2.AdaptiveI   = (mfxU16)((pInParams->bAdaptiveI) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_UNKNOWN);
            m_CodingOption2.AdaptiveB   = (mfxU16)((pInParams->bAdaptiveB) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_UNKNOWN);
            m_CodingOption2.BRefType    = (mfxU16)((pInParams->bBPyramid)  ? MFX_B_REF_PYRAMID   : MFX_B_REF_OFF);

            CHECK_RANGE_LIST(pInParams->nLookaheadDS, list_lookahead_ds, "la-quality");
            m_CodingOption2.LookAheadDS = (mfxU16)pInParams->nLookaheadDS;
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_7)) {
            m_CodingOption2.LookAheadDepth = (mfxU16)((pInParams->nLookaheadDepth == 0) ? pInParams->nLookaheadDepth : clamp_param_int(pInParams->nLookaheadDepth, QSV_LOOKAHEAD_DEPTH_MIN, QSV_LOOKAHEAD_DEPTH_MAX, _T("la-depth")));

            CHECK_RANGE_LIST(pInParams->nTrellis, list_avc_trellis_for_options, "trellis");
            m_CodingOption2.Trellis = (mfxU16)pInParams->nTrellis;
        }
        if (pInParams->bMBBRC) {
            m_CodingOption2.MBBRC = MFX_CODINGOPTION_ON;
        }

        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_26)
            && pInParams->extBrcAdaptiveLTR) {
            m_CodingOption2.BitrateLimit = MFX_CODINGOPTION_OFF;
        }
        if (pInParams->extBRC) {
            m_CodingOption2.ExtBRC = MFX_CODINGOPTION_ON;
        }
        if (pInParams->bIntraRefresh) {
            m_CodingOption2.IntRefType = 1;
            m_CodingOption2.IntRefCycleSize = (mfxU16)((pInParams->nGOPLength >= 2) ? pInParams->nGOPLength : ((m_encFps.n() + m_encFps.d() - 1) / m_encFps.d()) * 10);
        }
        if (pInParams->bNoDeblock) {
            m_CodingOption2.DisableDeblockingIdc = MFX_CODINGOPTION_ON;
        }
        for (int i = 0; i < 3; i++) {
            pInParams->nQPMin[i] = clamp_param_int(pInParams->nQPMin[i], 0, codecMaxQP, _T("qp min"));
            pInParams->nQPMax[i] = clamp_param_int(pInParams->nQPMax[i], 0, codecMaxQP, _T("qp max"));
            const int qpMin = (std::min)(pInParams->nQPMin[i], pInParams->nQPMax[i]);
            const int qpMax = (std::max)(pInParams->nQPMin[i], pInParams->nQPMax[i]);
            pInParams->nQPMin[i] = (0 == pInParams->nQPMin[i]) ? 0 : qpMin;
            pInParams->nQPMax[i] = (0 == pInParams->nQPMax[i]) ? 0 : qpMax;
        }
        m_CodingOption2.MaxQPI = (mfxU8)pInParams->nQPMax[0];
        m_CodingOption2.MaxQPP = (mfxU8)pInParams->nQPMax[1];
        m_CodingOption2.MaxQPB = (mfxU8)pInParams->nQPMax[2];
        m_CodingOption2.MinQPI = (mfxU8)pInParams->nQPMin[0];
        m_CodingOption2.MinQPP = (mfxU8)pInParams->nQPMin[1];
        m_CodingOption2.MinQPB = (mfxU8)pInParams->nQPMin[2];
        m_EncExtParams.push_back((mfxExtBuffer *)&m_CodingOption2);
    }

    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)) {
        if (m_mfxEncParams.mfx.CodecId == MFX_CODEC_HEVC) {
            if (pInParams->hevc_tier != 0) {
                m_mfxEncParams.mfx.CodecLevel |= (mfxU16)pInParams->hevc_tier;
            }
        }
    }

    //API v1.11の機能
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_11)) {
        INIT_MFX_EXT_BUFFER(m_CodingOption3, MFX_EXTBUFF_CODING_OPTION3);
        if (MFX_RATECONTROL_QVBR == m_mfxEncParams.mfx.RateControlMethod) {
            m_CodingOption3.QVBRQuality = (mfxU16)clamp_param_int(pInParams->nQVBRQuality, 1, codecMaxQP, _T("qvbr-q"));
        }
        //WinBRCの対象のレート制御モードかどうかをチェックする
        //これを行わないとInvalid Parametersとなる場合がある
        static const auto WinBRCTargetRC = make_array<int>(MFX_RATECONTROL_VBR, MFX_RATECONTROL_LA, MFX_RATECONTROL_LA_HRD, MFX_RATECONTROL_QVBR);
        if (std::find(WinBRCTargetRC.begin(), WinBRCTargetRC.end(), pInParams->nEncMode) != WinBRCTargetRC.end()
            && pInParams->nMaxBitrate != 0
            && !pInParams->extBRC) { // extbrcはWinBRCと併用できない模様
            m_CodingOption3.WinBRCSize = (mfxU16)((0 != pInParams->nWinBRCSize) ? pInParams->nWinBRCSize : ((m_encFps.n() + m_encFps.d() - 1) / m_encFps.d()));
            m_CodingOption3.WinBRCMaxAvgKbps = (mfxU16)pInParams->nMaxBitrate;
        }

        //API v1.13の機能
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_13)) {
            m_CodingOption3.DirectBiasAdjustment       = (mfxU16)((pInParams->bDirectBiasAdjust)   ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF);
            m_CodingOption3.GlobalMotionBiasAdjustment = (mfxU16)((pInParams->bGlobalMotionAdjust) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF);
            if (pInParams->bGlobalMotionAdjust) {
                CHECK_RANGE_LIST(pInParams->nMVCostScaling, list_mv_cost_scaling, "mv-scaling");
                m_CodingOption3.MVCostScalingFactor    = (mfxU16)pInParams->nMVCostScaling;
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_16)) {
            m_CodingOption3.WeightedBiPred = (mfxU16)pInParams->nWeightB;
            m_CodingOption3.WeightedPred   = (mfxU16)pInParams->nWeightP;
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_17)) {
            m_CodingOption3.FadeDetection = check_coding_option((mfxU16)pInParams->nFadeDetect);
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_19)) {
            if (bQPOffsetUsed) {
                m_CodingOption3.EnableQPOffset = MFX_CODINGOPTION_ON;
                memcpy(m_CodingOption3.QPOffset, pInParams->pQPOffset, sizeof(pInParams->pQPOffset));
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_23)) {
            m_CodingOption3.RepartitionCheckEnable = (mfxU16)pInParams->nRepartitionCheck;
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_26)) {
            m_CodingOption3.ExtBrcAdaptiveLTR = (mfxU16)(pInParams->extBrcAdaptiveLTR ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_UNKNOWN);
        }
        m_EncExtParams.push_back((mfxExtBuffer *)&m_CodingOption3);
    }

    //Bluray互換出力
    if (pInParams->nBluray) {
        if (   m_mfxEncParams.mfx.RateControlMethod != MFX_RATECONTROL_CBR
            && m_mfxEncParams.mfx.RateControlMethod != MFX_RATECONTROL_VBR
            && m_mfxEncParams.mfx.RateControlMethod != MFX_RATECONTROL_LA
            && m_mfxEncParams.mfx.RateControlMethod != MFX_RATECONTROL_LA_HRD) {
                if (pInParams->nBluray == 1) {
                    PrintMes(RGY_LOG_ERROR, _T("")
                        _T("Current encode mode (%s) is not preferred for Bluray encoding,\n")
                        _T("since it cannot set Max Bitrate.\n")
                        _T("Please consider using Lookahead/VBR/CBR mode for Bluray encoding.\n"), EncmodeToStr(m_mfxEncParams.mfx.RateControlMethod));
                    return RGY_ERR_INCOMPATIBLE_VIDEO_PARAM;
                } else {
                    //pInParams->nBluray == 2 -> force Bluray
                    PrintMes(RGY_LOG_WARN, _T("")
                        _T("Current encode mode (%s) is not preferred for Bluray encoding,\n")
                        _T("since it cannot set Max Bitrate.\n")
                        _T("This output might not be able to be played on a Bluray Player.\n")
                        _T("Please consider using Lookahead/VBR/CBR mode for Bluray encoding.\n"), EncmodeToStr(m_mfxEncParams.mfx.RateControlMethod));
                }
        }
        if (   m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_CBR
            || m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_VBR
            || m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_LA
            || m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_LA_HRD) {
                m_mfxEncParams.mfx.MaxKbps    = (std::min)(m_mfxEncParams.mfx.MaxKbps, (uint16_t)40000);
                m_mfxEncParams.mfx.TargetKbps = (std::min)(m_mfxEncParams.mfx.TargetKbps, m_mfxEncParams.mfx.MaxKbps);
                if (m_mfxEncParams.mfx.BufferSizeInKB == 0) {
                    m_mfxEncParams.mfx.BufferSizeInKB = m_mfxEncParams.mfx.MaxKbps / 8;
                }
                if (m_mfxEncParams.mfx.InitialDelayInKB == 0) {
                    m_mfxEncParams.mfx.InitialDelayInKB = m_mfxEncParams.mfx.BufferSizeInKB / 2;
                }
        } else {
            m_mfxEncParams.mfx.BufferSizeInKB = 25000 / 8;
        }
        m_mfxEncParams.mfx.CodecLevel = (m_mfxEncParams.mfx.CodecLevel == 0) ? MFX_LEVEL_AVC_41 : ((std::min)(m_mfxEncParams.mfx.CodecLevel, (uint16_t)MFX_LEVEL_AVC_41));
        m_mfxEncParams.mfx.NumSlice   = (std::max)(m_mfxEncParams.mfx.NumSlice, (uint16_t)4);
        m_mfxEncParams.mfx.GopOptFlag &= (~MFX_GOP_STRICT);
        m_mfxEncParams.mfx.GopRefDist = (std::min)(m_mfxEncParams.mfx.GopRefDist, (uint16_t)(3+1));
        m_mfxEncParams.mfx.GopPicSize = (int)((std::min)(m_mfxEncParams.mfx.GopPicSize, (uint16_t)30) / m_mfxEncParams.mfx.GopRefDist) * m_mfxEncParams.mfx.GopRefDist;
        m_mfxEncParams.mfx.NumRefFrame = (std::min)(m_mfxEncParams.mfx.NumRefFrame, (uint16_t)6);
        m_CodingOption.MaxDecFrameBuffering = m_mfxEncParams.mfx.NumRefFrame;
        m_CodingOption.VuiNalHrdParameters = MFX_CODINGOPTION_ON;
        m_CodingOption.VuiVclHrdParameters = MFX_CODINGOPTION_ON;
        m_CodingOption.AUDelimiter  = MFX_CODINGOPTION_ON;
        m_CodingOption.PicTimingSEI = MFX_CODINGOPTION_ON;
        m_CodingOption.ResetRefList = MFX_CODINGOPTION_ON;
        //m_CodingOption.EndOfSequence = MFX_CODINGOPTION_ON; //hwモードでは効果なし 0x00, 0x00, 0x01, 0x0a
        //m_CodingOption.EndOfStream   = MFX_CODINGOPTION_ON; //hwモードでは効果なし 0x00, 0x00, 0x01, 0x0b
        PrintMes(RGY_LOG_DEBUG, _T("InitMfxEncParams: Adjusted param for Bluray encoding.\n"));
    }

    m_EncExtParams.push_back((mfxExtBuffer *)&m_CodingOption);

    //m_mfxEncParams.mfx.TimeStampCalc = (mfxU16)((pInParams->vpp.nDeinterlace == MFX_DEINTERLACE_IT) ? MFX_TIMESTAMPCALC_TELECINE : MFX_TIMESTAMPCALC_UNKNOWN);
    //m_mfxEncParams.mfx.ExtendedPicStruct = pInParams->nPicStruct;

    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_3) &&
        (m_encVUI.format    != get_cx_value(list_videoformat, _T("undef")) ||
         m_encVUI.colorprim != get_cx_value(list_colorprim, _T("undef")) ||
         m_encVUI.transfer  != get_cx_value(list_transfer, _T("undef")) ||
         m_encVUI.matrix    != get_cx_value(list_colormatrix, _T("undef")) ||
         m_encVUI.colorrange == RGY_COLORRANGE_FULL
        ) ) {
#define GET_COLOR_PRM(v, list) (mfxU16)((v == COLOR_VALUE_AUTO) ? ((pInParams->input.dstHeight >= HD_HEIGHT_THRESHOLD) ? list[HD_INDEX].value : list[SD_INDEX].value) : v)
            //色設定 (for API v1.3)
            CHECK_RANGE_LIST(m_encVUI.format,    list_videoformat, "videoformat");
            CHECK_RANGE_LIST(m_encVUI.colorprim, list_colorprim,   "colorprim");
            CHECK_RANGE_LIST(m_encVUI.transfer,  list_transfer,    "transfer");
            CHECK_RANGE_LIST(m_encVUI.matrix,    list_colormatrix, "colormatrix");

            INIT_MFX_EXT_BUFFER(m_VideoSignalInfo, MFX_EXTBUFF_VIDEO_SIGNAL_INFO);
            m_VideoSignalInfo.ColourDescriptionPresent = 1; //"1"と設定しないと正しく反映されない
            m_VideoSignalInfo.VideoFormat              = (mfxU16)m_encVUI.format;
            m_VideoSignalInfo.VideoFullRange           = m_encVUI.colorrange == RGY_COLORRANGE_FULL;
            m_VideoSignalInfo.ColourPrimaries          = (mfxU16)m_encVUI.colorprim;
            m_VideoSignalInfo.TransferCharacteristics  = (mfxU16)m_encVUI.transfer;
            m_VideoSignalInfo.MatrixCoefficients       = (mfxU16)m_encVUI.matrix;
#undef GET_COLOR_PRM
            m_EncExtParams.push_back((mfxExtBuffer *)&m_VideoSignalInfo);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_13)
        && m_encVUI.chromaloc != RGY_CHROMALOC_UNSPECIFIED) {
        INIT_MFX_EXT_BUFFER(m_chromalocInfo, MFX_EXTBUFF_CHROMA_LOC_INFO);
        m_chromalocInfo.ChromaLocInfoPresentFlag = 1;
        m_chromalocInfo.ChromaSampleLocTypeTopField = (mfxU16)(m_encVUI.chromaloc-1);
        m_chromalocInfo.ChromaSampleLocTypeBottomField = (mfxU16)(m_encVUI.chromaloc-1);
        ////HWエンコーダではこれはサポートされていない模様なので無効化する
        //m_EncExtParams.push_back((mfxExtBuffer *)&m_chromalocInfo);
    }

    m_mfxEncParams.mfx.FrameInfo.FourCC = MFX_FOURCC_NV12;
    if (encodeBitDepth > 8) {
        m_mfxEncParams.mfx.FrameInfo.FourCC = MFX_FOURCC_P010;
        m_mfxEncParams.mfx.FrameInfo.BitDepthLuma = (mfxU16)encodeBitDepth;
        m_mfxEncParams.mfx.FrameInfo.BitDepthChroma = (mfxU16)encodeBitDepth;
        m_mfxEncParams.mfx.FrameInfo.Shift = 1;
    }
    m_mfxEncParams.mfx.FrameInfo.Width  = (mfxU16)ALIGN(pInParams->input.dstWidth, blocksz);
    m_mfxEncParams.mfx.FrameInfo.Height = (mfxU16)((MFX_PICSTRUCT_PROGRESSIVE == m_mfxEncParams.mfx.FrameInfo.PicStruct)?
        ALIGN(pInParams->input.dstHeight, blocksz) : ALIGN(pInParams->input.dstHeight, blocksz * 2));

    m_mfxEncParams.mfx.FrameInfo.CropX = 0;
    m_mfxEncParams.mfx.FrameInfo.CropY = 0;
    m_mfxEncParams.mfx.FrameInfo.CropW = (mfxU16)pInParams->input.dstWidth;
    m_mfxEncParams.mfx.FrameInfo.CropH = (mfxU16)pInParams->input.dstHeight;

    // In case of HEVC when height and/or width divided with 8 but not divided with 16
    // add extended parameter to increase performance
    if ( ( !((m_mfxEncParams.mfx.FrameInfo.CropW & 15 ) ^ 8 ) ||
           !((m_mfxEncParams.mfx.FrameInfo.CropH & 15 ) ^ 8 ) ) &&
             (m_mfxEncParams.mfx.CodecId == MFX_CODEC_HEVC) ) {
        INIT_MFX_EXT_BUFFER(m_ExtHEVCParam, MFX_EXTBUFF_HEVC_PARAM);
        m_ExtHEVCParam.PicWidthInLumaSamples = m_mfxEncParams.mfx.FrameInfo.CropW;
        m_ExtHEVCParam.PicHeightInLumaSamples = m_mfxEncParams.mfx.FrameInfo.CropH;
        m_EncExtParams.push_back((mfxExtBuffer*)&m_ExtHEVCParam);
    }

    if (m_mfxEncParams.mfx.CodecId == MFX_CODEC_VP8) {
        INIT_MFX_EXT_BUFFER(m_ExtVP8CodingOption, MFX_EXTBUFF_VP8_CODING_OPTION);
        m_ExtVP8CodingOption.SharpnessLevel = (mfxU16)clamp_param_int(pInParams->nVP8Sharpness, 0, 8, _T("sharpness"));
        m_EncExtParams.push_back((mfxExtBuffer*)&m_ExtVP8CodingOption);
    }

    if (!m_EncExtParams.empty()) {
        m_mfxEncParams.ExtParam = &m_EncExtParams[0];
        m_mfxEncParams.NumExtParam = (mfxU16)m_EncExtParams.size();
        for (const auto& extParam : m_EncExtParams) {
            PrintMes(RGY_LOG_DEBUG, _T("InitMfxEncParams: set ext param %s.\n"), fourccToStr(extParam->BufferId).c_str());
        }
    }

    PrintMes(RGY_LOG_DEBUG, _T("InitMfxEncParams: enc input frame %dx%d (%d,%d,%d,%d)\n"),
        m_mfxEncParams.mfx.FrameInfo.Width, m_mfxEncParams.mfx.FrameInfo.Height,
        m_mfxEncParams.mfx.FrameInfo.CropX, m_mfxEncParams.mfx.FrameInfo.CropY, m_mfxEncParams.mfx.FrameInfo.CropW, m_mfxEncParams.mfx.FrameInfo.CropH);
    PrintMes(RGY_LOG_DEBUG, _T("InitMfxEncParams: enc input color format %s, chroma %s, bitdepth %d, shift %d, picstruct %s\n"),
        ColorFormatToStr(m_mfxEncParams.mfx.FrameInfo.FourCC), ChromaFormatToStr(m_mfxEncParams.mfx.FrameInfo.ChromaFormat),
        m_mfxEncParams.mfx.FrameInfo.BitDepthLuma, m_mfxEncParams.mfx.FrameInfo.Shift, MFXPicStructToStr(m_mfxEncParams.mfx.FrameInfo.PicStruct).c_str());
    PrintMes(RGY_LOG_DEBUG, _T("InitMfxEncParams: set all enc params.\n"));

    m_pmfxENC.reset(new MFXVideoENCODE(m_mfxSession));
    if (!m_pmfxENC) {
        return RGY_ERR_MEMORY_ALLOC;
    }
    return RGY_ERR_NONE;
}

#if 0
mfxStatus CQSVPipeline::InitMfxVppParams(sInputParams *pInParams) {
    const mfxU32 blocksz = (pInParams->CodecId == MFX_CODEC_HEVC) ? 32 : 16;
    mfxU64 availableFeaures = CheckVppFeatures(m_mfxSession, m_mfxVer);
#if ENABLE_FPS_CONVERSION
    if (FPS_CONVERT_NONE != pInParams->vpp.nFPSConversion && !(availableFeaures & VPP_FEATURE_FPS_CONVERSION_ADV)) {
        PrintMes(RGY_LOG_WARN, _T("FPS Conversion not supported on this platform, disabled.\n"));
        pInParams->vpp.nFPSConversion = FPS_CONVERT_NONE;
    }
#else
    if (pInParams->vpp.rotate) {
        if (!(availableFeaures & VPP_FEATURE_ROTATE)) {
            PrintMes(RGY_LOG_ERROR, _T("vpp-rotate is not supported on this platform.\n"));
            return MFX_ERR_UNSUPPORTED;
        }
        if (pInParams->input.picstruct & RGY_PICSTRUCT_INTERLACED) {
            PrintMes(RGY_LOG_ERROR, _T("vpp-rotate is not supported with interlaced output.\n"));
            return MFX_ERR_INVALID_VIDEO_PARAM;
        }
    }
    //現時点ではうまく動いてなさそうなので無効化
    if (FPS_CONVERT_NONE != pInParams->vpp.fpsConversion) {
        PrintMes(RGY_LOG_WARN, _T("FPS Conversion not supported on this build, disabled.\n"));
        pInParams->vpp.fpsConversion = FPS_CONVERT_NONE;
    }
#endif

    if (pInParams->vpp.imageStabilizer && !(availableFeaures & VPP_FEATURE_IMAGE_STABILIZATION)) {
        PrintMes(RGY_LOG_WARN, _T("Image Stabilizer not supported on this platform, disabled.\n"));
        pInParams->vpp.imageStabilizer = 0;
    }

    if (pInParams->input.picstruct & RGY_PICSTRUCT_INTERLACED) {
        switch (pInParams->vpp.deinterlace) {
        case MFX_DEINTERLACE_IT_MANUAL:
            if (!(availableFeaures & VPP_FEATURE_DEINTERLACE_IT_MANUAL)) {
                PrintMes(RGY_LOG_ERROR, _T("Deinterlace \"it-manual\" is not supported on this platform.\n"));
                return MFX_ERR_INVALID_VIDEO_PARAM;
            }
            break;
        case MFX_DEINTERLACE_AUTO_SINGLE:
        case MFX_DEINTERLACE_AUTO_DOUBLE:
            if (!(availableFeaures & VPP_FEATURE_DEINTERLACE_AUTO)) {
                PrintMes(RGY_LOG_ERROR, _T("Deinterlace \"auto\" is not supported on this platform.\n"));
                return MFX_ERR_INVALID_VIDEO_PARAM;
            }
            break;
        default:
            break;
        }
    }

    if (pInParams->vpp.scalingQuality != MFX_SCALING_MODE_DEFAULT
        && !(availableFeaures & VPP_FEATURE_SCALING_QUALITY)) {
        PrintMes(RGY_LOG_WARN, _T("vpp scaling quality is not supported on this platform, disabled.\n"));
        pInParams->vpp.scalingQuality = MFX_SCALING_MODE_DEFAULT;
    }

    if (pInParams->vpp.mirrorType != MFX_MIRRORING_DISABLED
        && !(availableFeaures & VPP_FEATURE_MIRROR)) {
        PrintMes(RGY_LOG_ERROR, _T("vpp mirroring is not supported on this platform, disabled.\n"));
        return MFX_ERR_UNSUPPORTED;
    }

    if (!pInParams) {
        return MFX_ERR_MEMORY_ALLOC;
    }

    m_mfxVppParams.IOPattern = (pInParams->memType != SYSTEM_MEMORY) ?
        MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY :
        MFX_IOPATTERN_IN_SYSTEM_MEMORY | MFX_IOPATTERN_OUT_SYSTEM_MEMORY;

    m_mfxVppParams.vpp.In.PicStruct = picstruct_rgy_to_enc(pInParams->input.picstruct);
    m_mfxVppParams.vpp.In.FrameRateExtN = pInParams->input.fpsN;
    m_mfxVppParams.vpp.In.FrameRateExtD = pInParams->input.fpsD;
    m_mfxVppParams.vpp.In.AspectRatioW  = (mfxU16)pInParams->nPAR[0];
    m_mfxVppParams.vpp.In.AspectRatioH  = (mfxU16)pInParams->nPAR[1];

    mfxFrameInfo inputFrameInfo = frameinfo_rgy_to_enc(m_pFileReader->GetInputFrameInfo());
    if (inputFrameInfo.FourCC == 0 || inputFrameInfo.FourCC == MFX_FOURCC_NV12) {
        m_mfxVppParams.vpp.In.FourCC       = MFX_FOURCC_NV12;
        m_mfxVppParams.vpp.In.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
        m_mfxVppParams.vpp.In.Width     = (mfxU16)ALIGN(pInParams->input.srcWidth, blocksz);
        m_mfxVppParams.vpp.In.Height    = (mfxU16)((MFX_PICSTRUCT_PROGRESSIVE == m_mfxVppParams.vpp.In.PicStruct)?
            ALIGN(pInParams->input.srcHeight, blocksz) : ALIGN(pInParams->input.srcHeight, blocksz));
    } else {
        m_mfxVppParams.vpp.In.FourCC         = inputFrameInfo.FourCC;
        m_mfxVppParams.vpp.In.ChromaFormat   = inputFrameInfo.ChromaFormat;
        m_mfxVppParams.vpp.In.BitDepthLuma   = inputFrameInfo.BitDepthLuma;
        m_mfxVppParams.vpp.In.BitDepthChroma = inputFrameInfo.BitDepthChroma;
        if (m_mfxVppParams.vpp.In.ChromaFormat == MFX_CHROMAFORMAT_YUV422) {
            //10を指定しないとおかしな変換が行われる
            if (m_mfxVppParams.vpp.In.BitDepthLuma > 8) m_mfxVppParams.vpp.In.BitDepthLuma = 10;
            if (m_mfxVppParams.vpp.In.BitDepthChroma > 8) m_mfxVppParams.vpp.In.BitDepthChroma = 10;
            if (m_mfxVppParams.vpp.In.BitDepthLuma > 8) m_mfxVppParams.vpp.In.Shift = 1;
        }
        //QSVデコーダは特別にShiftパラメータを使う可能性がある
        if (inputFrameInfo.Shift) {
            m_mfxVppParams.vpp.In.Shift      = inputFrameInfo.Shift;
        }
        m_mfxVppParams.vpp.In.Width     = (mfxU16)ALIGN(inputFrameInfo.CropW, blocksz);
        m_mfxVppParams.vpp.In.Height    = (mfxU16)((MFX_PICSTRUCT_PROGRESSIVE == m_mfxVppParams.vpp.In.PicStruct) ?
            ALIGN(inputFrameInfo.CropH, blocksz) : ALIGN(inputFrameInfo.CropH, blocksz * 2));
    }

    m_mfxVppParams.vpp.In.CropW = (mfxU16)pInParams->input.srcWidth;
    m_mfxVppParams.vpp.In.CropH = (mfxU16)pInParams->input.srcHeight;

    //QSVデコードを行う場合、CropはVppで行う
    if (m_pFileReader->getInputCodec() != RGY_CODEC_UNKNOWN) {
        m_mfxVppParams.vpp.In.CropX = (mfxU16)pInParams->input.crop.e.left;
        m_mfxVppParams.vpp.In.CropY = (mfxU16)pInParams->input.crop.e.up;
        m_mfxVppParams.vpp.In.CropW -= (mfxU16)(pInParams->input.crop.e.left   + pInParams->input.crop.e.right);
        m_mfxVppParams.vpp.In.CropH -= (mfxU16)(pInParams->input.crop.e.bottom + pInParams->input.crop.e.up);
        PrintMes(RGY_LOG_DEBUG, _T("InitMfxVppParams: vpp crop enabled.\n"));
    }
    PrintMes(RGY_LOG_DEBUG, _T("InitMfxVppParams: vpp input frame %dx%d (%d,%d,%d,%d)\n"),
        m_mfxVppParams.vpp.In.Width, m_mfxVppParams.vpp.In.Height, m_mfxVppParams.vpp.In.CropX, m_mfxVppParams.vpp.In.CropY, m_mfxVppParams.vpp.In.CropW, m_mfxVppParams.vpp.In.CropH);
    PrintMes(RGY_LOG_DEBUG, _T("InitMfxVppParams: vpp input color format %s, chroma %d, bitdepth %d, shift %d\n"),
        ColorFormatToStr(m_mfxVppParams.vpp.In.FourCC), m_mfxVppParams.vpp.In.ChromaFormat, m_mfxVppParams.vpp.In.BitDepthLuma, m_mfxVppParams.vpp.In.Shift);

    memcpy(&m_mfxVppParams.vpp.Out, &m_mfxVppParams.vpp.In, sizeof(mfxFrameInfo));

    m_mfxEncParams.mfx.FrameInfo.FourCC = MFX_FOURCC_NV12;
    if (pInParams->CodecId == MFX_CODEC_HEVC && pInParams->CodecProfile == MFX_PROFILE_HEVC_MAIN10) {
        m_mfxVppParams.vpp.Out.FourCC = MFX_FOURCC_P010;
        m_mfxVppParams.vpp.Out.BitDepthLuma = 10;
        m_mfxVppParams.vpp.Out.BitDepthChroma = 10;
        m_mfxVppParams.vpp.Out.Shift = 1;
    } else {
        m_mfxVppParams.vpp.Out.FourCC = MFX_FOURCC_NV12;
        m_mfxVppParams.vpp.Out.BitDepthLuma = 0;
        m_mfxVppParams.vpp.Out.BitDepthChroma = 0;
        m_mfxVppParams.vpp.Out.Shift = 0;
    }
    m_mfxVppParams.vpp.Out.ChromaFormat   = MFX_CHROMAFORMAT_YUV420;
    m_mfxVppParams.vpp.Out.PicStruct = (pInParams->vpp.deinterlace) ? MFX_PICSTRUCT_PROGRESSIVE : picstruct_rgy_to_enc(pInParams->input.picstruct);

    if ((pInParams->input.picstruct & RGY_PICSTRUCT_INTERLACED) != 0) {
        INIT_MFX_EXT_BUFFER(m_ExtDeinterlacing, MFX_EXTBUFF_VPP_DEINTERLACING);
        switch (pInParams->vpp.deinterlace) {
        case MFX_DEINTERLACE_NORMAL:
        case MFX_DEINTERLACE_AUTO_SINGLE:
            m_ExtDeinterlacing.Mode = (uint16_t)((pInParams->vpp.deinterlace == MFX_DEINTERLACE_NORMAL) ? MFX_DEINTERLACING_30FPS_OUT : MFX_DEINTERLACING_AUTO_SINGLE);
            break;
        case MFX_DEINTERLACE_IT:
        case MFX_DEINTERLACE_IT_MANUAL:
            if (pInParams->vpp.deinterlace == MFX_DEINTERLACE_IT_MANUAL) {
                m_ExtDeinterlacing.Mode = MFX_DEINTERLACING_FIXED_TELECINE_PATTERN;
                m_ExtDeinterlacing.TelecinePattern = (mfxU16)pInParams->vpp.telecinePattern;
            } else {
                m_ExtDeinterlacing.Mode = MFX_DEINTERLACING_24FPS_OUT;
            }
            m_mfxVppParams.vpp.Out.FrameRateExtN = (m_mfxVppParams.vpp.Out.FrameRateExtN * 4) / 5;
            break;
        case MFX_DEINTERLACE_BOB:
        case MFX_DEINTERLACE_AUTO_DOUBLE:
            m_ExtDeinterlacing.Mode = (uint16_t)((pInParams->vpp.deinterlace == MFX_DEINTERLACE_BOB) ? MFX_DEINTERLACING_BOB : MFX_DEINTERLACING_AUTO_DOUBLE);
            m_mfxVppParams.vpp.Out.FrameRateExtN *= 2;
            break;
        case MFX_DEINTERLACE_NONE:
        default:
            break;
        }
        if (pInParams->vpp.deinterlace != MFX_DEINTERLACE_NONE) {
#if ENABLE_ADVANCED_DEINTERLACE
            if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_13)) {
                m_VppExtParams.push_back((mfxExtBuffer *)&m_ExtDeinterlacing);
                m_VppDoUseList.push_back(MFX_EXTBUFF_VPP_DEINTERLACING);
            }
#endif
            m_mfxVppParams.vpp.Out.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
            VppExtMes += _T("Deinterlace (");
            VppExtMes += get_chr_from_value(list_deinterlace, pInParams->vpp.deinterlace);
            if (pInParams->vpp.deinterlace == MFX_DEINTERLACE_IT_MANUAL) {
                VppExtMes += _T(", ");
                VppExtMes += get_chr_from_value(list_telecine_patterns, pInParams->vpp.telecinePattern);
            }
            VppExtMes += _T(")\n");
            PrintMes(RGY_LOG_DEBUG, _T("InitMfxVppParams: vpp deinterlace enabled.\n"));
        }
        pInParams->vpp.fpsConversion = FPS_CONVERT_NONE;
    } else {
        switch (pInParams->vpp.fpsConversion) {
        case FPS_CONVERT_MUL2:
            m_mfxVppParams.vpp.Out.FrameRateExtN *= 2;
            break;
        case FPS_CONVERT_MUL2_5:
            m_mfxVppParams.vpp.Out.FrameRateExtN = m_mfxVppParams.vpp.Out.FrameRateExtN * 5 / 2;
            break;
        default:
            break;
        }
    }
    m_mfxVppParams.vpp.Out.CropX = 0;
    m_mfxVppParams.vpp.Out.CropY = 0;
    m_mfxVppParams.vpp.Out.CropW = (mfxU16)pInParams->input.dstWidth;
    m_mfxVppParams.vpp.Out.CropH = (mfxU16)pInParams->input.dstHeight;
    m_mfxVppParams.vpp.Out.Width = (mfxU16)ALIGN(pInParams->input.dstWidth, blocksz);
    m_mfxVppParams.vpp.Out.Height = (mfxU16)((MFX_PICSTRUCT_PROGRESSIVE == m_mfxVppParams.vpp.Out.PicStruct)?
        ALIGN(pInParams->input.dstHeight, blocksz) : ALIGN(pInParams->input.dstHeight, blocksz));
    PrintMes(RGY_LOG_DEBUG, _T("InitMfxVppParams: vpp output frame %dx%d (%d,%d,%d,%d)\n"),
        m_mfxVppParams.vpp.Out.Width, m_mfxVppParams.vpp.Out.Height, m_mfxVppParams.vpp.Out.CropX, m_mfxVppParams.vpp.Out.CropY, m_mfxVppParams.vpp.Out.CropW, m_mfxVppParams.vpp.Out.CropH);
    PrintMes(RGY_LOG_DEBUG, _T("InitMfxVppParams: vpp output color format %s, chroma %d, bitdepth %d, shift %d\n"),
        ColorFormatToStr(m_mfxVppParams.vpp.Out.FourCC), m_mfxVppParams.vpp.Out.ChromaFormat, m_mfxVppParams.vpp.Out.BitDepthLuma, m_mfxVppParams.vpp.Out.Shift);
    PrintMes(RGY_LOG_DEBUG, _T("InitMfxVppParams: set all vpp params.\n"));
    return MFX_ERR_NONE;
}

mfxStatus CQSVPipeline::CreateVppExtBuffers(sInputParams *pParams) {
    m_VppExtParams.clear();
    m_VppDoUseList.clear();
    m_VppDoNotUseList.clear();
    m_VppDoNotUseList.push_back(MFX_EXTBUFF_VPP_PROCAMP);
    auto vppExtAddMes = [this](tstring str) {
        VppExtMes += str;
        PrintMes(RGY_LOG_DEBUG, _T("CreateVppExtBuffers: %s"), str.c_str());
    };

    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)
        && (   MFX_FOURCC_RGB3 == m_mfxVppParams.vpp.In.FourCC
            || MFX_FOURCC_RGB4 == m_mfxVppParams.vpp.In.FourCC
            || pParams->vpp.colorspace.enable)) {

        const bool inputRGB = m_mfxVppParams.vpp.In.FourCC == MFX_FOURCC_RGB3 || m_mfxVppParams.vpp.In.FourCC == MFX_FOURCC_RGB4;
        VideoVUIInfo vuiFrom = VideoVUIInfo();
        VideoVUIInfo vuiTo   = VideoVUIInfo();
        if (pParams->vpp.colorspace.enable && pParams->vpp.colorspace.convs.size() > 0) {
            vuiFrom = pParams->vpp.colorspace.convs.begin()->from;
            vuiTo = pParams->vpp.colorspace.convs.begin()->to;
        }
        if (vuiTo.colorrange == RGY_COLORRANGE_UNSPECIFIED) {
            vuiTo.colorrange = m_encVUI.colorrange;
            if (vuiTo.colorrange == RGY_COLORRANGE_UNSPECIFIED) {
                vuiTo.colorrange = RGY_COLORRANGE_AUTO;
            }
        }
        if (vuiTo.matrix == RGY_MATRIX_UNSPECIFIED) {
            vuiTo.matrix = m_encVUI.matrix;
            if (vuiTo.matrix == RGY_MATRIX_UNSPECIFIED) {
                vuiTo.matrix = RGY_MATRIX_AUTO;
            }
        }
        vuiFrom.apply_auto(pParams->input.vui, pParams->input.srcHeight);
        vuiTo.apply_auto(vuiFrom, pParams->input.dstHeight);
        if (inputRGB) {
            vuiFrom.colorrange = RGY_COLORRANGE_FULL;
        }

        INIT_MFX_EXT_BUFFER(m_ExtVppVSI, MFX_EXTBUFF_VPP_VIDEO_SIGNAL_INFO);
        m_ExtVppVSI.In.NominalRange    = (mfxU16)((vuiFrom.colorrange == RGY_COLORRANGE_FULL) ? MFX_NOMINALRANGE_0_255 : MFX_NOMINALRANGE_16_235);
        m_ExtVppVSI.In.TransferMatrix  = (mfxU16)((vuiFrom.matrix == RGY_MATRIX_ST170_M) ? MFX_TRANSFERMATRIX_BT601 : MFX_TRANSFERMATRIX_BT709);
        m_ExtVppVSI.Out.NominalRange   = (mfxU16)((vuiTo.colorrange == RGY_COLORRANGE_FULL) ? MFX_NOMINALRANGE_0_255 : MFX_NOMINALRANGE_16_235);
        m_ExtVppVSI.Out.TransferMatrix = (mfxU16)((vuiTo.matrix == RGY_MATRIX_ST170_M) ? MFX_TRANSFERMATRIX_BT601 : MFX_TRANSFERMATRIX_BT709);
        m_encVUI.apply_auto(vuiFrom, pParams->input.dstHeight);
        m_VppExtParams.push_back((mfxExtBuffer *)&m_ExtVppVSI);
        m_VppDoUseList.push_back(MFX_EXTBUFF_VPP_VIDEO_SIGNAL_INFO);
        PrintMes(RGY_LOG_DEBUG, _T("InitMfxVppParams: vpp colorspace conversion enabled.\n"));
    } else if(check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_17)) { //なんかMFX_EXTBUFF_VPP_VIDEO_SIGNAL_INFOを設定すると古い環境ではvppの初期化に失敗するらしい。
        m_VppDoNotUseList.push_back(MFX_EXTBUFF_VPP_VIDEO_SIGNAL_INFO);
    }

    if (pParams->vpp.detail.enable) {
        INIT_MFX_EXT_BUFFER(m_ExtDetail, MFX_EXTBUFF_VPP_DETAIL);
        m_ExtDetail.DetailFactor = (mfxU16)clamp_param_int(pParams->vpp.detail.strength, QSV_VPP_DETAIL_ENHANCE_MIN, QSV_VPP_DETAIL_ENHANCE_MAX, _T("vpp-detail-enhance"));
        m_VppExtParams.push_back((mfxExtBuffer*)&m_ExtDetail);

        vppExtAddMes(strsprintf(_T("Detail Enhancer, strength %d\n"), m_ExtDetail.DetailFactor));
        m_VppDoUseList.push_back(MFX_EXTBUFF_VPP_DETAIL);
    } else {
        m_VppDoNotUseList.push_back(MFX_EXTBUFF_VPP_DETAIL);
    }

    switch (pParams->vpp.rotate) {
    case MFX_ANGLE_90:
    case MFX_ANGLE_180:
    case MFX_ANGLE_270:
        INIT_MFX_EXT_BUFFER(m_ExtRotate, MFX_EXTBUFF_VPP_ROTATION);
        m_ExtRotate.Angle = (mfxU16)pParams->vpp.rotate;
        m_VppExtParams.push_back((mfxExtBuffer*)&m_ExtRotate);

        vppExtAddMes(strsprintf(_T("rotate %d\n"), pParams->vpp.rotate));
        m_VppDoUseList.push_back(MFX_EXTBUFF_VPP_ROTATION);
        break;
    default:
        break;
    }

    if (pParams->vpp.mirrorType != MFX_MIRRORING_DISABLED) {
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_19)) {
            PrintMes(RGY_LOG_ERROR, _T("--vpp-mirror not supported on this platform, disabled.\n"));
            return MFX_ERR_UNSUPPORTED;
        }
        INIT_MFX_EXT_BUFFER(m_ExtMirror, MFX_EXTBUFF_VPP_MIRRORING);
        m_ExtMirror.Type = (mfxU16)pParams->vpp.mirrorType;
        m_VppExtParams.push_back((mfxExtBuffer*)&m_ExtMirror);

        vppExtAddMes(strsprintf(_T("mirroring %s\n"), get_chr_from_value(list_vpp_mirroring, pParams->vpp.mirrorType)));
        m_VppDoUseList.push_back(MFX_EXTBUFF_VPP_MIRRORING);
    }

    if ( (    pParams->input.srcWidth  != pParams->input.dstWidth
           || pParams->input.srcHeight != pParams->input.dstHeight)
        && pParams->vpp.scalingQuality != MFX_SCALING_MODE_DEFAULT) {
        INIT_MFX_EXT_BUFFER(m_ExtScaling, MFX_EXTBUFF_VPP_SCALING);
        m_ExtScaling.ScalingMode = (mfxU16)pParams->vpp.scalingQuality;
        m_VppExtParams.push_back((mfxExtBuffer*)&m_ExtScaling);

        m_VppDoUseList.push_back(MFX_EXTBUFF_VPP_SCALING);
    }

    if (pParams->vpp.fpsConversion != FPS_CONVERT_NONE) {
        INIT_MFX_EXT_BUFFER(m_ExtFrameRateConv, MFX_EXTBUFF_VPP_FRAME_RATE_CONVERSION);
        m_ExtFrameRateConv.Algorithm = MFX_FRCALGM_FRAME_INTERPOLATION;
        m_VppExtParams.push_back((mfxExtBuffer*)&m_ExtFrameRateConv);

        vppExtAddMes(_T("fps conversion with interpolation\n"));
        m_VppDoUseList.push_back(MFX_EXTBUFF_VPP_FRAME_RATE_CONVERSION);
    }

    if (pParams->vpp.denoise.enable) {
        INIT_MFX_EXT_BUFFER(m_ExtDenoise, MFX_EXTBUFF_VPP_DENOISE);
        m_ExtDenoise.DenoiseFactor = (mfxU16)clamp_param_int(pParams->vpp.denoise.strength, QSV_VPP_DENOISE_MIN, QSV_VPP_DENOISE_MAX, _T("vpp-denoise"));
        m_VppExtParams.push_back((mfxExtBuffer*)&m_ExtDenoise);

        vppExtAddMes(strsprintf(_T("Denoise, strength %d\n"), m_ExtDenoise.DenoiseFactor));
        m_VppDoUseList.push_back(MFX_EXTBUFF_VPP_DENOISE);
    } else {
        m_VppDoNotUseList.push_back(MFX_EXTBUFF_VPP_DENOISE);
    }

    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_26)) {
        if (pParams->vpp.mctf.enable) {
            INIT_MFX_EXT_BUFFER(m_ExtMctf, MFX_EXTBUFF_VPP_MCTF);
            m_ExtMctf.FilterStrength = (mfxU16)clamp_param_int(pParams->vpp.mctf.strength, 0, QSV_VPP_MCTF_MAX, _T("vpp-mctf"));
            m_VppExtParams.push_back((mfxExtBuffer*)&m_ExtMctf);

            if (m_ExtMctf.FilterStrength == 0) {
                vppExtAddMes(_T("mctf, strength auto\n"));
            } else {
                vppExtAddMes(strsprintf(_T("mctf, strength %d\n"), m_ExtMctf.FilterStrength));
            }
            m_VppDoUseList.push_back(MFX_EXTBUFF_VPP_MCTF);
        } else {
            m_VppDoNotUseList.push_back(MFX_EXTBUFF_VPP_MCTF);
        }
    } else {
        if (pParams->vpp.mctf.enable) {
            PrintMes(RGY_LOG_WARN, _T("--vpp-mctf not supported on this platform, disabled.\n"));
            pParams->vpp.mctf.enable = false;
        }
    }

    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_6)) {
        if (pParams->vpp.imageStabilizer) {
            CHECK_RANGE_LIST(pParams->vpp.imageStabilizer, list_vpp_image_stabilizer, "vpp-image-stab");
            INIT_MFX_EXT_BUFFER(m_ExtImageStab, MFX_EXTBUFF_VPP_IMAGE_STABILIZATION);
            m_ExtImageStab.Mode = (mfxU16)pParams->vpp.imageStabilizer;
            m_VppExtParams.push_back((mfxExtBuffer*)&m_ExtImageStab);

            vppExtAddMes(strsprintf(_T("Stabilizer, mode %s\n"), get_vpp_image_stab_mode_str(m_ExtImageStab.Mode)));
            m_VppDoUseList.push_back(MFX_EXTBUFF_VPP_IMAGE_STABILIZATION);
        }
    } else {
        if (pParams->vpp.imageStabilizer) {
            PrintMes(RGY_LOG_WARN, _T("--vpp-image-stab not supported on this platform, disabled.\n"));
            pParams->vpp.imageStabilizer = false;
        }
    }

    m_VppDoNotUseList.push_back(MFX_EXTBUFF_VPP_SCENE_ANALYSIS);

    if (   check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_3)
        && (pParams->input.picstruct & RGY_PICSTRUCT_INTERLACED)) {
            switch (pParams->vpp.deinterlace) {
            case MFX_DEINTERLACE_IT:
            case MFX_DEINTERLACE_IT_MANUAL:
            case MFX_DEINTERLACE_BOB:
            case MFX_DEINTERLACE_AUTO_DOUBLE:
                INIT_MFX_EXT_BUFFER(m_ExtFrameRateConv, MFX_EXTBUFF_VPP_FRAME_RATE_CONVERSION);
                m_ExtFrameRateConv.Algorithm = MFX_FRCALGM_DISTRIBUTED_TIMESTAMP;

                m_VppDoUseList.push_back(MFX_EXTBUFF_VPP_FRAME_RATE_CONVERSION);
                break;
            default:
                break;
            }
    }

    if (m_VppDoUseList.size()) {
        INIT_MFX_EXT_BUFFER(m_VppDoUse, MFX_EXTBUFF_VPP_DOUSE);
        m_VppDoUse.NumAlg = (mfxU32)m_VppDoUseList.size();
        m_VppDoUse.AlgList = &m_VppDoUseList[0];

        m_VppExtParams.insert(m_VppExtParams.begin(), (mfxExtBuffer *)&m_VppDoUse);
        for (const auto& extParam : m_VppDoUseList) {
            PrintMes(RGY_LOG_DEBUG, _T("CreateVppExtBuffers: set DoUse %s.\n"), fourccToStr(extParam).c_str());
        }
    }

    //Haswell以降では、DONOTUSEをセットするとdetail enhancerの効きが固定になるなど、よくわからない挙動を示す。
    if (m_VppDoNotUseList.size() && getCPUGen(&m_mfxSession) < CPU_GEN_HASWELL) {
        AllocAndInitVppDoNotUse();
        m_VppExtParams.push_back((mfxExtBuffer *)&m_VppDoNotUse);
        for (const auto& extParam : m_VppDoNotUseList) {
            PrintMes(RGY_LOG_DEBUG, _T("CreateVppExtBuffers: set DoNotUse %s.\n"), fourccToStr(extParam).c_str());
        }
    }

    m_mfxVppParams.ExtParam = (m_VppExtParams.size()) ? &m_VppExtParams[0] : nullptr;
    m_mfxVppParams.NumExtParam = (mfxU16)m_VppExtParams.size();

    return MFX_ERR_NONE;
}

#pragma warning (push)
#pragma warning (disable: 4100)
mfxStatus CQSVPipeline::InitVppPrePlugins(sInputParams *pParams) {
    mfxStatus sts = MFX_ERR_NONE;
#if ENABLE_CUSTOM_VPP
    tstring vppPreMes = _T("");
#if ENABLE_AVSW_READER && ENABLE_LIBASS_SUBBURN
    if (pParams->vpp.subburn.nTrack || pParams->vpp.subburn.pFilePath) {
        AVDemuxStream targetSubStream = { 0 };
        auto pAVCodecReader = std::dynamic_pointer_cast<RGYInputAvcodec>(m_pFileReader);
        if (pParams->vpp.subburn.nTrack) {
            if (pAVCodecReader == nullptr) {
                PrintMes(RGY_LOG_ERROR, _T("--vpp-sub-burn from track required --avqsv reader.\n"));
                return MFX_ERR_UNSUPPORTED;
            }
            for (const auto& stream : pAVCodecReader->GetInputStreamInfo()) {
                if (trackID(stream.trackId) == pParams->vpp.subburn.nTrack) {
                    targetSubStream = stream;
                    break;
                }
            }
            if (targetSubStream.stream == nullptr) {
                PrintMes(RGY_LOG_ERROR, _T("--vpp-sub-burn: subtitile track #%d not found.\n"), pParams->vpp.subburn.nTrack);
                return MFX_ERR_UNSUPPORTED;
            }
        }

        unique_ptr<CVPPPlugin> filter(new CVPPPlugin());
        SubBurnParam param(m_pMFXAllocator.get(), m_memType,
            pParams->vpp.subburn.pFilePath,
            pParams->vpp.subburn.pCharEnc,
            pParams->vpp.subburn.nShaping,
            frameinfo_rgy_to_enc(m_pFileReader->GetInputFrameInfo()),
            (pAVCodecReader) ? pAVCodecReader->GetInputVideoStream() : nullptr,
            //ファイルからの読み込みの時は最初のpts分の補正が必要
            //トラックからの読み込みなら不要
            (pAVCodecReader && !pParams->vpp.subburn.nTrack) ? pAVCodecReader->GetVideoFirstKeyPts() : 0,
            targetSubStream,
            //avqsvリーダー使用時以外はcropは読み込み段階ですでに行われている
            (pAVCodecReader) ? &(pParams->input.crop) : nullptr);
        uint16_t nVppSubAsyncDepth = (pParams->nAsyncDepth) ? (uint16_t)(std::min<int>)(pParams->nAsyncDepth, 6) : 3;
        sts = filter->Init(m_mfxVer, _T("subburn"), &param, sizeof(param), true, m_memType, m_hwdev, m_pMFXAllocator.get(), nVppSubAsyncDepth, m_mfxVppParams.vpp.In, m_mfxVppParams.IOPattern, m_pQSVLog);
        if (sts != MFX_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("%s\n"), filter->getMessage().c_str());
            return sts;
        } else {
            sts = MFXJoinSession(m_mfxSession, filter->getSession());
            QSV_ERR_MES(sts, _T("Failed to join vpp pre filter session."));
            tstring mes = filter->getMessage();
            PrintMes(RGY_LOG_DEBUG, _T("InitVppPrePlugins: add filter: %s\n"), mes.c_str());
            vppPreMes += mes;
            m_VppPrePlugins.push_back(std::move(filter));
        }
    }
#endif //#if ENABLE_AVSW_READER && ENABLE_LIBASS_SUBBURN
    if (pParams->vpp.delogo.pFilePath) {
        unique_ptr<CVPPPlugin> filter(new CVPPPlugin());
        DelogoParam param(m_pMFXAllocator.get(), m_memType, pParams->vpp.delogo.pFilePath, pParams->vpp.delogo.pSelect, pParams->common.inputFilename.c_str(),
            (short)pParams->vpp.delogo.posOffset.first, (short)pParams->vpp.delogo.posOffset.second, (short)pParams->vpp.delogo.depth,
            (short)pParams->vpp.delogo.YOffset, (short)pParams->vpp.delogo.CbOffset, (short)pParams->vpp.delogo.CrOffset, pParams->vpp.delogo.add);
        sts = filter->Init(m_mfxVer, _T("delogo"), &param, sizeof(param), true, m_memType, m_hwdev, m_pMFXAllocator.get(), 3, m_mfxVppParams.vpp.In, m_mfxVppParams.IOPattern, m_pQSVLog);
        if (sts == MFX_ERR_ABORTED) {
            PrintMes(RGY_LOG_WARN, _T("%s\n"), filter->getMessage().c_str());
            sts = MFX_ERR_NONE;
        } else if (sts != MFX_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("%s\n"), filter->getMessage().c_str());
            return sts;
        } else {
            sts = MFXJoinSession(m_mfxSession, filter->getSession());
            QSV_ERR_MES(sts, _T("Failed to join vpp pre filter session."));
            tstring mes = filter->getMessage();
            PrintMes(RGY_LOG_DEBUG, _T("InitVppPrePlugins: add filter: %s\n"), mes.c_str());
            vppPreMes += mes;
            m_VppPrePlugins.push_back(std::move(filter));
        }
    }
    if (pParams->vpp.halfTurn) {
        unique_ptr<CVPPPlugin> filter(new CVPPPlugin());
        RotateParam param(180);
        sts = filter->Init(m_mfxVer, _T("rotate"), &param, sizeof(param), true, m_memType, m_hwdev, m_pMFXAllocator.get(), 3, m_mfxVppParams.vpp.In, m_mfxVppParams.IOPattern, m_pQSVLog);
        if (sts != MFX_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("%s\n"), filter->getMessage().c_str());
            return sts;
        } else {
            sts = MFXJoinSession(m_mfxSession, filter->getSession());
            QSV_ERR_MES(sts, _T("Failed to join vpp pre filter session."));
            tstring mes = filter->getMessage();
            PrintMes(RGY_LOG_DEBUG, _T("InitVppPrePlugins: add filter: %s\n"), mes.c_str());
            vppPreMes += mes;
            m_VppPrePlugins.push_back(std::move(filter));
        }
    }
    VppExtMes = vppPreMes + VppExtMes;
#endif
    return sts;
}

mfxStatus CQSVPipeline::InitVppPostPlugins(sInputParams *pParams) {
    return MFX_ERR_NONE;
}
#pragma warning (pop)
#endif

RGY_ERR CQSVPipeline::InitOpenCL() {
    const mfxHandleType hdl_t = mfxHandleTypeFromMemType(m_memType);
    mfxHDL hdl = nullptr;
    if (hdl_t) {
        auto sts = err_to_rgy(m_hwdev->GetHandle(hdl_t, &hdl));
        RGY_ERR(sts, _T("Failed to get HW device handle."));
        PrintMes(RGY_LOG_DEBUG, _T("Got HW device handle: %p.\n"), hdl);
    }

    RGYOpenCL cl(m_pQSVLog);
    auto platforms = cl.getPlatforms("Intel");
    if (platforms.size() == 0) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to find OpenCL platforms.\n"));
        return RGY_ERR_DEVICE_LOST;
    }
    PrintMes(RGY_LOG_DEBUG, _T("Created Intel OpenCL platform.\n"));

    auto& platform = platforms[0];
    if (m_memType == D3D9_MEMORY) {
        if (platform->createDeviceListD3D9(CL_DEVICE_TYPE_GPU, (void *)hdl) != CL_SUCCESS || platform->devs().size() == 0) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to find d3d9 device.\n"));
            return RGY_ERR_DEVICE_LOST;
        }
    } else if (m_memType == D3D11_MEMORY) {
        if (platform->createDeviceListD3D11(CL_DEVICE_TYPE_GPU, (void *)hdl) != CL_SUCCESS || platform->devs().size() == 0) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to find d3d11 device.\n"));
            return RGY_ERR_DEVICE_LOST;
        }
    } else {
        if (platform->createDeviceList(CL_DEVICE_TYPE_GPU) != CL_SUCCESS || platform->devs().size() == 0) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to find gpu device.\n"));
            return RGY_ERR_DEVICE_LOST;
        }
    }
    auto devices = platform->devs();
    if ((int)devices.size() == 0) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to OpenCL device.\n"));
        return RGY_ERR_DEVICE_LOST;
    }
    platform->setDev(devices[0]);

    m_cl = std::make_shared<RGYOpenCLContext>(platform, m_pQSVLog);
    if (m_cl->createContext() != CL_SUCCESS) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to create OpenCL context.\n"));
        return RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::CreateHWDevice() {
    auto sts = RGY_ERR_NONE;

#if D3D_SURFACES_SUPPORT
    POINT point = {0, 0};
    HWND window = WindowFromPoint(point);
    m_hwdev.reset();

    if (m_memType) {
#if MFX_D3D11_SUPPORT
        if (m_memType == D3D11_MEMORY
            && (m_hwdev = std::make_shared<CQSVD3D11Device>(m_pQSVLog))) {
            m_memType = D3D11_MEMORY;
            PrintMes(RGY_LOG_DEBUG, _T("HWDevice: d3d11 - initializing...\n"));

            sts = err_to_rgy(m_hwdev->Init(NULL, 0, GetAdapterID(m_mfxSession)));
            if (sts != MFX_ERR_NONE) {
                m_hwdev.reset();
                PrintMes(RGY_LOG_DEBUG, _T("HWDevice: d3d11 - initializing failed.\n"));
            }
        }
#endif // #if MFX_D3D11_SUPPORT
        if (!m_hwdev && (m_hwdev = std::make_shared<CQSVD3D9Device>(m_pQSVLog))) {
            //もし、d3d11要求で失敗したら自動的にd3d9に切り替える
            //sessionごと切り替える必要がある
            if (m_memType != D3D9_MEMORY) {
                PrintMes(RGY_LOG_DEBUG, _T("Retry openning device, chaging to d3d9 mode, re-init session.\n"));
                InitSession(true, D3D9_MEMORY);
                m_memType = m_memType;
            }

            PrintMes(RGY_LOG_DEBUG, _T("HWDevice: d3d9 - initializing...\n"));
            sts = err_to_rgy(m_hwdev->Init(window, 0, GetAdapterID(m_mfxSession)));
        }
    }
    RGY_ERR(sts, _T("Failed to initialize HW Device."));
    PrintMes(RGY_LOG_DEBUG, _T("HWDevice: initializing device success.\n"));

#elif LIBVA_SUPPORT
    m_hwdev.reset(CreateVAAPIDevice("", MFX_LIBVA_DRM, m_pQSVLog));
    if (!m_hwdev) {
        return MFX_ERR_MEMORY_ALLOC;
    }
    sts = m_hwdev->Init(NULL, 0, GetAdapterID(m_mfxSession));
    QSV_ERR_MES(sts, _T("Failed to initialize HW Device."));
#endif
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::ResetDevice() {
    if (m_memType & (D3D9_MEMORY | D3D11_MEMORY)) {
        PrintMes(RGY_LOG_DEBUG, _T("HWDevice: reset.\n"));
        return err_to_rgy(m_hwdev->Reset());
    }
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::AllocFrames() {
    if (m_pipelineTasks.size() == 0) {
        PrintMes(RGY_LOG_ERROR, _T("allocFrames: pipeline not defined!\n"));
        return RGY_ERR_INVALID_CALL;
    }

    PrintMes(RGY_LOG_DEBUG, _T("allocFrames: m_nAsyncDepth - %d frames\n"), m_nAsyncDepth);

#if 0
    mfxFrameAllocRequest DecRequest;
    mfxFrameAllocRequest EncRequest;
    mfxFrameAllocRequest VppRequest[2];

    //各要素が要求するフレーム数を調べる
    if (m_pmfxENC) {
        auto sts = err_to_rgy(m_pmfxENC->QueryIOSurf(&m_mfxEncParams, &EncRequest));
        RGY_ERR(sts, _T("Failed to get required buffer size for encoder."));
        PrintMes(RGY_LOG_DEBUG, _T("allocFrames: Enc query - %d frames\n"), EncRequest.NumFrameSuggested);
    }

    if (m_pmfxVPP) {
        // VppRequest[0]はvppへの入力, VppRequest[1]はvppからの出力
        auto sts = err_to_rgy(m_pmfxVPP->QueryIOSurf(&m_mfxVppParams, VppRequest));
        RGY_ERR(sts, _T("Failed to get required buffer size for vpp."));
        PrintMes(RGY_LOG_DEBUG, _T("allocFrames: Vpp query[0] - %d frames\n"), VppRequest[0].NumFrameSuggested);
        PrintMes(RGY_LOG_DEBUG, _T("allocFrames: Vpp query[1] - %d frames\n"), VppRequest[1].NumFrameSuggested);
    }

    if (m_pmfxDEC) {
        auto sts = err_to_rgy(m_pmfxDEC->QueryIOSurf(&m_mfxDecParams, &DecRequest));
        RGY_ERR(sts, _T("Failed to get required buffer size for decoder."));
        PrintMes(RGY_LOG_DEBUG, _T("allocFrames: Dec query - %d frames\n"), DecRequest.NumFrameSuggested);
    }
#endif

    PipelineTask *t0 = m_pipelineTasks[0].get();
    for (size_t ip = 1; ip < m_pipelineTasks.size(); ip++) {
        if (t0->isPassThrough()) {
            PrintMes(RGY_LOG_ERROR, _T("allocFrames: t0 cannot be path through task!\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        // 次のtaskを見つける
        PipelineTask *t1 = nullptr;
        for (; ip < m_pipelineTasks.size(); ip++) {
            if (!m_pipelineTasks[ip]->isPassThrough()) { // isPassThroughがtrueなtaskはスキップ
                t1 = m_pipelineTasks[ip].get();
                break;
            }
        }
        if (t1 == nullptr) {
            PrintMes(RGY_LOG_ERROR, _T("AllocFrames: invalid pipeline, t1 not found!\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        PrintMes(RGY_LOG_DEBUG, _T("AllocFrames: %s-%s\n"), t0->print().c_str(), t1->print().c_str());

        const auto t0Alloc = t0->requiredSurfOut();
        const auto t1Alloc = t1->requiredSurfIn();
        int t0RequestNumFrame = 0;
        int t1RequestNumFrame = 0;
        mfxFrameAllocRequest allocRequest;
        if (t0Alloc.has_value() && t1Alloc.has_value()) {
            t0RequestNumFrame = t0Alloc.value().NumFrameSuggested;
            t1RequestNumFrame = t1Alloc.value().NumFrameSuggested;
            allocRequest = (t0->workSurfacesAllocPriority() >= t1->workSurfacesAllocPriority()) ? t0Alloc.value() : t1Alloc.value();
            allocRequest.Info.Width = std::max(t0Alloc.value().Info.Width, t1Alloc.value().Info.Width);
            allocRequest.Info.Height = std::max(t0Alloc.value().Info.Height, t1Alloc.value().Info.Height);
        } else if (t0Alloc.has_value()) {
            allocRequest = t0Alloc.value();
            t0RequestNumFrame = t0Alloc.value().NumFrameSuggested;
        } else if (t1Alloc.has_value()) {
            allocRequest = t1Alloc.value();
            t1RequestNumFrame = t1Alloc.value().NumFrameSuggested;
        } else {
            PrintMes(RGY_LOG_ERROR, _T("AllocFrames: invalid pipeline: cannot get request from either t0 or t1!\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        switch (t0->taskType()) {
        case PipelineTaskType::MFXDEC:    allocRequest.Type |= MFX_MEMTYPE_FROM_DECODE; break;
        case PipelineTaskType::MFXVPP:    allocRequest.Type |= MFX_MEMTYPE_FROM_VPPOUT; break;
        case PipelineTaskType::OPENCL:    allocRequest.Type |= MFX_MEMTYPE_FROM_VPPOUT; break;
        case PipelineTaskType::MFXENC:    allocRequest.Type |= MFX_MEMTYPE_FROM_ENC;    break;
        case PipelineTaskType::MFXENCODE: allocRequest.Type |= MFX_MEMTYPE_FROM_ENCODE; break;
        default: break;
        }
        switch (t1->taskType()) {
        case PipelineTaskType::MFXDEC:    allocRequest.Type |= MFX_MEMTYPE_FROM_DECODE; break;
        case PipelineTaskType::MFXVPP:    allocRequest.Type |= MFX_MEMTYPE_FROM_VPPIN;  break;
        case PipelineTaskType::OPENCL:    allocRequest.Type |= MFX_MEMTYPE_FROM_VPPIN;  break;
        case PipelineTaskType::MFXENC:    allocRequest.Type |= MFX_MEMTYPE_FROM_ENC;    break;
        case PipelineTaskType::MFXENCODE: allocRequest.Type |= MFX_MEMTYPE_FROM_ENCODE; break;
        default: break;
        }

        allocRequest.NumFrameSuggested = (mfxU16)std::max(1, t0RequestNumFrame + t1RequestNumFrame + m_nAsyncDepth + 1);
        allocRequest.NumFrameMin = allocRequest.NumFrameSuggested;
        PrintMes(RGY_LOG_DEBUG, _T("AllocFrames: %s-%s, type: %s, %dx%d [%d,%d,%d,%d], request %d frames\n"),
            t0->print().c_str(), t1->print().c_str(), qsv_memtype_str(allocRequest.Type).c_str(),
            allocRequest.Info.Width, allocRequest.Info.Height, allocRequest.Info.CropX, allocRequest.Info.CropY, allocRequest.Info.CropW, allocRequest.Info.CropH,
            allocRequest.NumFrameSuggested);

        auto sts = t0->workSurfacesAlloc(allocRequest, m_bExternalAlloc, m_pMFXAllocator.get());
        if (sts != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("AllocFrames:   Failed to allocate frames for %s-%s: %s."), t0->print().c_str(), t1->print().c_str(), get_err_mes(sts));
            return sts;
        }
        t0 = t1;
    }
#if 0
    int nEncSurfNum = 0; // enc用のフレーム数
    int nVppSurfNum = 0; // vpp用のフレーム数

    int nInputSurfAdd   = 0;
    int nDecSurfAdd     = 0; // dec用のフレーム数
    int nVppPreSurfAdd  = 0; // vpp pre用のフレーム数
    int nVppSurfAdd     = 0;
    int nVppPostSurfAdd = 0; // vpp post用のフレーム数

    RGY_MEMSET_ZERO(DecRequest);
    RGY_MEMSET_ZERO(EncRequest);
    RGY_MEMSET_ZERO(VppRequest[0]);
    RGY_MEMSET_ZERO(VppRequest[1]);
    for (const auto& filter : m_VppPrePlugins) {
        RGY_MEMSET_ZERO(filter->m_PluginResponse);
    }
    for (const auto& filter : m_VppPostPlugins) {
        RGY_MEMSET_ZERO(filter->m_PluginResponse);
    }
    RGY_MEMSET_ZERO(NextRequest);

    nInputSurfAdd = std::max(m_EncThread.m_nFrameBuffer, 1);

    nDecSurfAdd = DecRequest.NumFrameSuggested;

    //vppの出力用のフレームとencの入力用のフレームは共有される
    nEncSurfNum = EncRequest.NumFrameSuggested + m_nAsyncDepth;

    //m_nAsyncDepthを考慮して、vppの入力用のフレーム数を決める
    nVppSurfNum = VppRequest[0].NumFrameSuggested + m_nAsyncDepth;

    PrintMes(RGY_LOG_DEBUG, _T("AllocFrames: nInputSurfAdd %d frames\n"), nInputSurfAdd);
    PrintMes(RGY_LOG_DEBUG, _T("AllocFrames: nDecSurfAdd   %d frames\n"), nDecSurfAdd);

    if (m_pmfxDEC) {
        NextRequest = DecRequest;
    }

    //VppPrePlugins
    if (m_VppPrePlugins.size()) {
        for (int i = 0; i < (int)m_VppPrePlugins.size(); i++) {
            uint32_t mem_type = ((HW_MEMORY & m_memType) ? MFX_MEMTYPE_EXTERNAL_FRAME : MFX_MEMTYPE_SYSTEM_MEMORY);
            m_VppPrePlugins[i]->m_nSurfNum += m_nAsyncDepth;
            if (i == 0) {
                mem_type |= (nDecSurfAdd) ? (MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET | MFX_MEMTYPE_FROM_DECODE) : (MFX_MEMTYPE_VIDEO_MEMORY_PROCESSOR_TARGET | MFX_MEMTYPE_FROM_VPPOUT);
                m_VppPrePlugins[i]->m_nSurfNum += (std::max)(1, (int)nInputSurfAdd + nDecSurfAdd - m_nAsyncDepth + 1);
            } else {
                //surfaceが2つの要素c1とc2に共有されるとき、NumSurf = c1_out + c2_in - AsyncDepth + 1
                mem_type |= MFX_MEMTYPE_VIDEO_MEMORY_PROCESSOR_TARGET | MFX_MEMTYPE_FROM_VPPOUT;
                m_VppPrePlugins[i]->m_nSurfNum += m_VppPrePlugins[i-1]->m_nSurfNum - m_nAsyncDepth + 1;
            }
            m_VppPrePlugins[i]->m_PluginRequest.Type = (mfxU16)mem_type;
            m_VppPrePlugins[i]->m_PluginRequest.NumFrameMin = (mfxU16)m_VppPrePlugins[i]->m_nSurfNum;
            m_VppPrePlugins[i]->m_PluginRequest.NumFrameSuggested = (mfxU16)m_VppPrePlugins[i]->m_nSurfNum;
            memcpy(&m_VppPrePlugins[i]->m_PluginRequest.Info, &(m_VppPrePlugins[i]->m_pluginVideoParams.mfx.FrameInfo), sizeof(mfxFrameInfo));
            if (m_pmfxDEC && nDecSurfAdd) {
                m_VppPrePlugins[i]->m_PluginRequest.Info.Width  = DecRequest.Info.Width;
                m_VppPrePlugins[i]->m_PluginRequest.Info.Height = DecRequest.Info.Height;
                m_VppPrePlugins[i]->m_pluginVideoParams.mfx.FrameInfo.Width  = DecRequest.Info.Width;
                m_VppPrePlugins[i]->m_pluginVideoParams.mfx.FrameInfo.Height = DecRequest.Info.Height;
            }
            NextRequest = m_VppPrePlugins[i]->m_PluginRequest;
            memcpy(&NextRequest.Info, &(m_VppPrePlugins[i]->m_pluginVideoParams.vpp.Out), sizeof(mfxFrameInfo));
            PrintMes(RGY_LOG_DEBUG, _T("AllocFrames: PrePlugins[%d] %s, type: %s, %dx%d [%d,%d,%d,%d], request %d frames\n"),
                i, m_VppPrePlugins[i]->getFilterName().c_str(), qsv_memtype_str(mem_type).c_str(),
                m_VppPrePlugins[i]->m_PluginRequest.Info.Width, m_VppPrePlugins[i]->m_PluginRequest.Info.Height,
                m_VppPrePlugins[i]->m_PluginRequest.Info.CropX, m_VppPrePlugins[i]->m_PluginRequest.Info.CropY,
                m_VppPrePlugins[i]->m_PluginRequest.Info.CropW, m_VppPrePlugins[i]->m_PluginRequest.Info.CropH,
                m_VppPrePlugins[i]->m_PluginRequest.NumFrameSuggested);
        }

        //後始末
        nDecSurfAdd = 0;
        nInputSurfAdd = 0;
        nVppPreSurfAdd = m_VppPrePlugins.back()->m_nSurfNum;
    }

    //Vpp
    if (m_pmfxVPP) {
        nVppSurfNum += (std::max)(1, (int)nInputSurfAdd + nDecSurfAdd + nVppPreSurfAdd - m_nAsyncDepth + 1);

        //VppRequest[0]の準備
        VppRequest[0].NumFrameMin = (mfxU16)nVppSurfNum;
        VppRequest[0].NumFrameSuggested = (mfxU16)nVppSurfNum;
        memcpy(&VppRequest[0].Info, &(m_mfxVppParams.mfx.FrameInfo), sizeof(mfxFrameInfo));
        if (m_pmfxDEC && nDecSurfAdd) {
            VppRequest[0].Type = DecRequest.Type;
            VppRequest[0].Info.Width  = DecRequest.Info.Width;
            VppRequest[0].Info.Height = DecRequest.Info.Height;
            m_mfxVppParams.mfx.FrameInfo.Width = DecRequest.Info.Width;
            m_mfxVppParams.mfx.FrameInfo.Height = DecRequest.Info.Height;
            //フレームのリクエストを出す時点でCropの値を入れておくと、
            //DecFrameAsyncでMFX_ERR_UNDEFINED_BEHAVIORを出してしまう
            //Cropの値はVppFrameAsyncの直前に渡すようにする
            VppRequest[0].Info.CropX = DecRequest.Info.CropX;
            VppRequest[0].Info.CropY = DecRequest.Info.CropY;
            VppRequest[0].Info.CropW = DecRequest.Info.CropW;
            VppRequest[0].Info.CropH = DecRequest.Info.CropH;
        }

        //後始末
        nInputSurfAdd = 0;
        nDecSurfAdd = 0;
        nVppPreSurfAdd = 0;
        nVppSurfAdd = (std::max<int>)(VppRequest[1].NumFrameSuggested, 1);
        NextRequest = VppRequest[1];
        memcpy(&NextRequest.Info, &(m_mfxVppParams.vpp.Out), sizeof(mfxFrameInfo));
        PrintMes(RGY_LOG_DEBUG, _T("AllocFrames: Vpp type: %s, %dx%d [%d,%d,%d,%d], request %d frames\n"),
            qsv_memtype_str(VppRequest[0].Type).c_str(),
            VppRequest[0].Info.Width, VppRequest[0].Info.Height,
            VppRequest[0].Info.CropX, VppRequest[0].Info.CropY, VppRequest[0].Info.CropW, VppRequest[0].Info.CropH, VppRequest[0].NumFrameSuggested);
    }

    //VppPostPlugins
    if (m_VppPostPlugins.size()) {
        for (int i = 0; i < (int)m_VppPostPlugins.size(); i++) {
            uint32_t mem_type = ((HW_MEMORY & m_memType) ? MFX_MEMTYPE_EXTERNAL_FRAME : MFX_MEMTYPE_SYSTEM_MEMORY);
            m_VppPostPlugins[i]->m_nSurfNum += m_nAsyncDepth;
            if (i == 0) {
                mem_type |= (nDecSurfAdd) ? (MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET | MFX_MEMTYPE_FROM_DECODE) : (MFX_MEMTYPE_VIDEO_MEMORY_PROCESSOR_TARGET | MFX_MEMTYPE_FROM_VPPOUT);
                m_VppPostPlugins[i]->m_nSurfNum += (std::max)(1, (int)nInputSurfAdd + nDecSurfAdd + nVppPreSurfAdd + nVppSurfAdd - m_nAsyncDepth + 1);
            } else {
                //surfaceが2つの要素c1とc2に共有されるとき、NumSurf = c1_out + c2_in - AsyncDepth + 1
                mem_type |= MFX_MEMTYPE_VIDEO_MEMORY_PROCESSOR_TARGET | MFX_MEMTYPE_FROM_VPPOUT;
                m_VppPostPlugins[i]->m_nSurfNum += m_VppPostPlugins[i-1]->m_nSurfNum - m_nAsyncDepth + 1;
            }
            m_VppPostPlugins[i]->m_PluginRequest.Type = (mfxU16)mem_type;
            m_VppPostPlugins[i]->m_PluginRequest.NumFrameMin = (mfxU16)m_VppPostPlugins[i]->m_nSurfNum;
            m_VppPostPlugins[i]->m_PluginRequest.NumFrameSuggested = (mfxU16)m_VppPostPlugins[i]->m_nSurfNum;
            memcpy(&m_VppPostPlugins[i]->m_PluginRequest.Info, &(m_VppPostPlugins[i]->m_pluginVideoParams.mfx.FrameInfo), sizeof(mfxFrameInfo));
            if (m_pmfxDEC && nDecSurfAdd) {
                m_VppPostPlugins[i]->m_PluginRequest.Type = DecRequest.Type;
                m_VppPostPlugins[i]->m_PluginRequest.Info.Width  = DecRequest.Info.Width;
                m_VppPostPlugins[i]->m_PluginRequest.Info.Height = DecRequest.Info.Height;
                m_VppPostPlugins[i]->m_pluginVideoParams.mfx.FrameInfo.Width  = DecRequest.Info.Width;
                m_VppPostPlugins[i]->m_pluginVideoParams.mfx.FrameInfo.Height = DecRequest.Info.Height;
            }
            NextRequest = m_VppPostPlugins[i]->m_PluginRequest;
            memcpy(&NextRequest.Info, &(m_VppPostPlugins[i]->m_pluginVideoParams.vpp.Out), sizeof(mfxFrameInfo));
            PrintMes(RGY_LOG_DEBUG, _T("AllocFrames: PostPlugins[%d] %s, type: %s, %dx%d [%d,%d,%d,%d], request %d frames\n"),
                i, m_VppPostPlugins[i]->getFilterName().c_str(), qsv_memtype_str(mem_type).c_str(),
                m_VppPostPlugins[i]->m_PluginRequest.Info.Width, m_VppPostPlugins[i]->m_PluginRequest.Info.Height,
                m_VppPostPlugins[i]->m_PluginRequest.Info.CropX, m_VppPostPlugins[i]->m_PluginRequest.Info.CropY,
                m_VppPostPlugins[i]->m_PluginRequest.Info.CropW, m_VppPostPlugins[i]->m_PluginRequest.Info.CropH,
                m_VppPostPlugins[i]->m_PluginRequest.NumFrameSuggested);
        }

        //後始末
        nInputSurfAdd = 0;
        nDecSurfAdd = 0;
        nVppPreSurfAdd = 0;
        nVppSurfAdd = 0;
        nVppPostSurfAdd = m_VppPostPlugins.back()->m_nSurfNum;
    }

    //Enc、エンコーダが有効でない場合は出力フレーム
    {
        nEncSurfNum += (std::max)(1, (int)nInputSurfAdd + nDecSurfAdd + nVppPreSurfAdd + nVppSurfAdd + nVppPostSurfAdd - m_nAsyncDepth + 1);
        if (m_pmfxENC == nullptr) {
            EncRequest = NextRequest;
            nEncSurfNum += (m_nAsyncDepth - 1);
        } else {
            memcpy(&EncRequest.Info, &(m_mfxEncParams.mfx.FrameInfo), sizeof(mfxFrameInfo));
        }
        EncRequest.NumFrameMin = (mfxU16)nEncSurfNum;
        EncRequest.NumFrameSuggested = (mfxU16)nEncSurfNum;
        if (m_pmfxDEC && nDecSurfAdd) {
            EncRequest.Type |= MFX_MEMTYPE_FROM_DECODE;
            EncRequest.Info.Width = std::max(EncRequest.Info.Width, DecRequest.Info.Width);
            EncRequest.Info.Height = std::max(EncRequest.Info.Height, DecRequest.Info.Height);
        }
        if (nVppPreSurfAdd || nVppSurfAdd || nVppPostSurfAdd) {
            EncRequest.Type |= MFX_MEMTYPE_FROM_VPPOUT;
        }

        //後始末
        nInputSurfAdd = 0;
        nDecSurfAdd = 0;
        nVppPreSurfAdd = 0;
        nVppSurfAdd = 0;
        nVppPostSurfAdd = 0;
        PrintMes(RGY_LOG_DEBUG, _T("AllocFrames: %s type: %s, %dx%d [%d,%d,%d,%d], request %d frames\n"),
            (m_pmfxENC) ? _T("Enc") : _T("Out"),
            qsv_memtype_str(EncRequest.Type).c_str(),
            EncRequest.Info.Width, EncRequest.Info.Height,
            EncRequest.Info.CropX, EncRequest.Info.CropY, EncRequest.Info.CropW, EncRequest.Info.CropH, EncRequest.NumFrameSuggested);
    }

    // エンコーダ用のメモリ確保
    sts = m_pMFXAllocator->Alloc(m_pMFXAllocator->pthis, &EncRequest, &m_EncResponse);
    QSV_ERR_MES(sts, _T("Failed to allocate frames for encoder."));
    PrintMes(RGY_LOG_DEBUG, _T("AllocFrames: Allocated EncRequest %d\n"), m_EncResponse.NumFrameActual);

    // vppを使用するなら、vpp用のメモリを確保する
    if (m_pmfxVPP) {
        sts = m_pMFXAllocator->Alloc(m_pMFXAllocator->pthis, &(VppRequest[0]), &m_VppResponse);
        QSV_ERR_MES(sts, _T("Failed to allocate frames for vpp."));
        PrintMes(RGY_LOG_DEBUG, _T("AllocFrames: Allocated VppRequest %d\n"), m_VppResponse.NumFrameActual);
    }

    //エンコーダ用のmfxFrameSurface1配列を作成する
    m_pEncSurfaces.resize(m_EncResponse.NumFrameActual);

    for (int i = 0; i < m_EncResponse.NumFrameActual; i++) {
        memset(&(m_pEncSurfaces[i]), 0, sizeof(mfxFrameSurface1));
        memcpy(&m_pEncSurfaces[i].Info, &(EncRequest.Info), sizeof(mfxFrameInfo));

        if (m_bExternalAlloc) {
            m_pEncSurfaces[i].Data.MemId = m_EncResponse.mids[i];
        } else {
            sts = m_pMFXAllocator->Lock(m_pMFXAllocator->pthis, m_EncResponse.mids[i], &(m_pEncSurfaces[i].Data));
            QSV_ERR_MES(sts, _T("Failed to allocate surfaces for encoder."));
        }
    }

    //vpp用のmfxFrameSurface1配列を作成する
    if (m_pmfxVPP) {
        m_pVppSurfaces.resize(m_VppResponse.NumFrameActual);

        for (int i = 0; i < m_VppResponse.NumFrameActual; i++) {
            RGY_MEMSET_ZERO(m_pVppSurfaces[i]);
            memcpy(&m_pVppSurfaces[i].Info, &(VppRequest[0].Info), sizeof(mfxFrameInfo));

            if (m_bExternalAlloc) {
                m_pVppSurfaces[i].Data.MemId = m_VppResponse.mids[i];
            } else {
                sts = m_pMFXAllocator->Lock(m_pMFXAllocator->pthis, m_VppResponse.mids[i], &(m_pVppSurfaces[i].Data));
                QSV_ERR_MES(sts, _T("Failed to allocate surfaces for vpp."));
            }
        }
    }

    //vpp pre用のmfxFrameSurface1配列を作成する
    for (const auto& filter : m_VppPrePlugins) {
        if (MFX_ERR_NONE != (sts = filter->AllocSurfaces(m_pMFXAllocator.get(), m_bExternalAlloc))) {
            PrintMes(RGY_LOG_ERROR, _T("AllocFrames: Failed to alloc surface for %s\n"), filter->getFilterName().c_str());
            return sts;
        }
        PrintMes(RGY_LOG_DEBUG, _T("AllocFrames: Allocated surface for %s\n"), filter->getFilterName().c_str());
    }

    //vpp post用のmfxFrameSurface1配列を作成する
    for (const auto& filter : m_VppPostPlugins) {
        if (MFX_ERR_NONE != (sts = filter->AllocSurfaces(m_pMFXAllocator.get(), m_bExternalAlloc))) {
            PrintMes(RGY_LOG_ERROR, _T("AllocFrames: Failed to alloc surface for %s\n"), filter->getFilterName().c_str());
            return sts;
        }
        PrintMes(RGY_LOG_DEBUG, _T("AllocFrames: Allocated surface for %s\n"), filter->getFilterName().c_str());
    }
#endif
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::CreateAllocator() {
    auto sts = RGY_ERR_NONE;
    PrintMes(RGY_LOG_DEBUG, _T("CreateAllocator: MemType: %s\n"), MemTypeToStr(m_memType));

    if (D3D9_MEMORY == m_memType || D3D11_MEMORY == m_memType || VA_MEMORY == m_memType || HW_MEMORY == m_memType) {
#if D3D_SURFACES_SUPPORT
        sts = CreateHWDevice();
        RGY_ERR(sts, _T("Failed to CreateHWDevice."));
        PrintMes(RGY_LOG_DEBUG, _T("CreateAllocator: CreateHWDevice success.\n"));

        const mfxHandleType hdl_t = mfxHandleTypeFromMemType(m_memType);
        mfxHDL hdl = NULL;
        sts = err_to_rgy(m_hwdev->GetHandle(hdl_t, &hdl));
        RGY_ERR(sts, _T("Failed to get HW device handle."));
        PrintMes(RGY_LOG_DEBUG, _T("CreateAllocator: HW device GetHandle success.\n"));

        mfxIMPL impl = 0;
        m_mfxSession.QueryIMPL(&impl);
        if (impl != MFX_IMPL_SOFTWARE) {
            // hwエンコード時のみハンドルを渡す
            sts = err_to_rgy(m_mfxSession.SetHandle(hdl_t, hdl));
            RGY_ERR(sts, _T("Failed to set HW device handle to encode session."));
            PrintMes(RGY_LOG_DEBUG, _T("CreateAllocator: set HW device handle to encode session.\n"));
        }

        //D3D allocatorを作成
#if MFX_D3D11_SUPPORT
        if (D3D11_MEMORY == m_memType) {
            PrintMes(RGY_LOG_DEBUG, _T("CreateAllocator: Create d3d11 allocator.\n"));
            m_pMFXAllocator.reset(new QSVAllocatorD3D11);
            if (!m_pMFXAllocator) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to allcate memory for D3D11FrameAllocator.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }

            QSVAllocatorParamsD3D11 *pd3dAllocParams = new QSVAllocatorParamsD3D11;
            if (!pd3dAllocParams) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to allcate memory for D3D11AllocatorParams.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
            pd3dAllocParams->pDevice = reinterpret_cast<ID3D11Device *>(hdl);
            PrintMes(RGY_LOG_DEBUG, _T("CreateAllocator: d3d11...\n"));

            m_pmfxAllocatorParams.reset(pd3dAllocParams);
        } else
#endif // #if MFX_D3D11_SUPPORT
        {
            PrintMes(RGY_LOG_DEBUG, _T("CreateAllocator: Create d3d9 allocator.\n"));
            m_pMFXAllocator.reset(new QSVAllocatorD3D9);
            if (!m_pMFXAllocator) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to allcate memory for D3DFrameAllocator.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }

            QSVAllocatorParamsD3D9 *pd3dAllocParams = new QSVAllocatorParamsD3D9;
            if (!pd3dAllocParams) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to allcate memory for pd3dAllocParams.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
            pd3dAllocParams->pManager = reinterpret_cast<IDirect3DDeviceManager9 *>(hdl);
            PrintMes(RGY_LOG_DEBUG, _T("CreateAllocator: d3d9...\n"));

            m_pmfxAllocatorParams.reset(pd3dAllocParams);
        }

        //GPUメモリ使用時には external allocatorを使用する必要がある
        //mfxSessionにallocatorを渡してやる必要がある
        sts = err_to_rgy(m_mfxSession.SetFrameAllocator(m_pMFXAllocator.get()));
        RGY_ERR(sts, _T("Failed to set frame allocator to encode session."));
        PrintMes(RGY_LOG_DEBUG, _T("CreateAllocator: frame allocator set to session.\n"));

        m_bExternalAlloc = true;
#endif
#ifdef LIBVA_SUPPORT
        sts = CreateHWDevice();
        RGY_ERR(sts, _T("Failed to CreateHWDevice."));

        mfxHDL hdl = NULL;
        sts = m_hwdev->GetHandle(MFX_HANDLE_VA_DISPLAY, &hdl);
        RGY_ERR(sts, _T("Failed to get HW device handle."));
        PrintMes(RGY_LOG_DEBUG, _T("CreateAllocator: HW device GetHandle success. : 0x%x\n"), (uint32_t)(size_t)hdl);

        //ハンドルを渡す
        sts = err_to_rgy(m_mfxSession.SetHandle(MFX_HANDLE_VA_DISPLAY, hdl));
        RGY_ERR(sts, _T("Failed to set HW device handle to encode session."));

        //VAAPI allocatorを作成
        m_pMFXAllocator.reset(new QSVAllocatorVA());
        if (!m_pMFXAllocator) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to allcate memory for vaapiFrameAllocator.\n"));
            return MFX_ERR_MEMORY_ALLOC;
        }

        QSVAllocatorParamsVA *p_vaapiAllocParams = new QSVAllocatorParamsVA();
        if (!p_vaapiAllocParams) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to allcate memory for vaapiAllocatorParams.\n"));
            return MFX_ERR_MEMORY_ALLOC;
        }

        p_vaapiAllocParams->m_dpy = (VADisplay)hdl;
        m_pmfxAllocatorParams.reset(p_vaapiAllocParams);

        //GPUメモリ使用時には external allocatorを使用する必要がある
        //mfxSessionにallocatorを渡してやる必要がある
        sts = err_to_rgy(m_mfxSession.SetFrameAllocator(m_pMFXAllocator.get()));
        RGY_ERR(sts, _T("Failed to set frame allocator to encode session."));
        PrintMes(RGY_LOG_DEBUG, _T("CreateAllocator: frame allocator set to session.\n"));

        m_bExternalAlloc = true;
#endif
    } else {
#ifdef LIBVA_SUPPORT
        //システムメモリ使用でも MFX_HANDLE_VA_DISPLAYをHW libraryに渡してやる必要がある
        mfxIMPL impl;
        m_mfxSession.QueryIMPL(&impl);

        if (MFX_IMPL_HARDWARE == MFX_IMPL_BASETYPE(impl)) {
            sts = CreateHWDevice();
            RGY_ERR(sts, _T("Failed to CreateHWDevice."));

            mfxHDL hdl = NULL;
            sts = err_to_rgy(m_hwdev->GetHandle(MFX_HANDLE_VA_DISPLAY, &hdl));
            RGY_ERR(sts, _T("Failed to get HW device handle."));
            PrintMes(RGY_LOG_DEBUG, _T("CreateAllocator: HW device GetHandle success. : 0x%x\n"), (uint32_t)(size_t)hdl);

            //ハンドルを渡す
            sts = err_to_rgy(m_mfxSession.SetHandle(MFX_HANDLE_VA_DISPLAY, hdl));
            RGY_ERR(sts, _T("Failed to set HW device handle to encode session."));
        }
#endif
        //system memory allocatorを作成
        m_pMFXAllocator.reset(new QSVAllocatorSys);
        if (!m_pMFXAllocator) {
            return RGY_ERR_MEMORY_ALLOC;
        }
        PrintMes(RGY_LOG_DEBUG, _T("CreateAllocator: sys mem allocator...\n"));
    }

    //メモリallocatorの初期化
    if (MFX_ERR_NONE > (sts = err_to_rgy(m_pMFXAllocator->Init(m_pmfxAllocatorParams.get(), m_pQSVLog)))) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to initialize %s memory allocator. : %s\n"), MemTypeToStr(m_memType), get_err_mes(sts));
        return sts;
    }
    PrintMes(RGY_LOG_DEBUG, _T("CreateAllocator: frame allocator initialized.\n"));

    return RGY_ERR_NONE;
}

void CQSVPipeline::DeleteHWDevice() {
    m_hwdev.reset();
}

void CQSVPipeline::DeleteAllocator() {
    m_pMFXAllocator.reset();
    m_pmfxAllocatorParams.reset();

    DeleteHWDevice();
}

CQSVPipeline::CQSVPipeline() :
    m_mfxVer({ 0 }),
    m_pStatus(),
    m_pPerfMonitor(),
    m_EncThread(),
    m_encWidth(0),
    m_encHeight(0),
    m_encPicstruct(RGY_PICSTRUCT_UNKNOWN),
    m_inputFps(),
    m_encFps(),
    m_outputTimebase(),
    m_encVUI(),
    m_bTimerPeriodTuning(false),
    m_pFileWriterListAudio(),
    m_pFileWriter(),
    m_AudioReaders(),
    m_pFileReader(),
    m_TaskPool(),
    m_nAsyncDepth(0),
    m_nAVSyncMode(RGY_AVSYNC_ASSUME_CFR),
    m_outputTimestamp(),
    m_InitParam(),
    m_pInitParamExtBuf(),
    m_ThreadsParam(),
    m_VideoSignalInfo(),
    m_chromalocInfo(),
    m_CodingOption(),
    m_CodingOption2(),
    m_CodingOption3(),
    m_ExtVP8CodingOption(),
    m_ExtHEVCParam(),
    m_mfxSession(),
    m_pmfxDEC(),
    m_pmfxENC(),
    m_mfxVPP(),
    m_SessionPlugins(),
    m_trimParam(),
    m_mfxDecParams(),
    m_mfxEncParams(),
    m_prmSetIn(),
    m_DecExtParams(),
    m_EncExtParams(),
    m_DecVidProc(),
#if ENABLE_AVSW_READER
    m_Chapters(),
#endif
    m_timecode(),
    m_HDRSei(),
    m_pMFXAllocator(),
    m_pmfxAllocatorParams(),
    m_nMFXThreads(-1),
    m_memType(SYSTEM_MEMORY),
    m_bExternalAlloc(false),
    m_nProcSpeedLimit(0),
    m_pAbortByUser(nullptr),
    m_heAbort(),
    m_DecInputBitstream(),
    m_cl(),
    m_vpFilters(),
    m_hwdev(),
    m_pipelineTasks() {
    m_trimParam.offset = 0;

    for (size_t i = 0; i < _countof(m_pInitParamExtBuf); i++) {
        m_pInitParamExtBuf[i] = nullptr;
    }

#if ENABLE_MVC_ENCODING
    m_bIsMVC = false;
    m_MVCflags = MVC_DISABLED;
    m_nNumView = 0;
    RGY_MEMSET_ZERO(m_MVCSeqDesc);
    m_MVCSeqDesc.Header.BufferId = MFX_EXTBUFF_MVC_SEQ_DESC;
    m_MVCSeqDesc.Header.BufferSz = sizeof(m_MVCSeqDesc);
#endif
    RGY_MEMSET_ZERO(m_InitParam);
    INIT_MFX_EXT_BUFFER(m_VideoSignalInfo,    MFX_EXTBUFF_VIDEO_SIGNAL_INFO);
    INIT_MFX_EXT_BUFFER(m_chromalocInfo,      MFX_EXTBUFF_CHROMA_LOC_INFO);
    INIT_MFX_EXT_BUFFER(m_CodingOption,       MFX_EXTBUFF_CODING_OPTION);
    INIT_MFX_EXT_BUFFER(m_CodingOption2,      MFX_EXTBUFF_CODING_OPTION2);
    INIT_MFX_EXT_BUFFER(m_CodingOption3,      MFX_EXTBUFF_CODING_OPTION3);
    INIT_MFX_EXT_BUFFER(m_ExtVP8CodingOption, MFX_EXTBUFF_VP8_CODING_OPTION);
    INIT_MFX_EXT_BUFFER(m_ExtHEVCParam,       MFX_EXTBUFF_HEVC_PARAM);
    INIT_MFX_EXT_BUFFER(m_ThreadsParam,       MFX_EXTBUFF_THREADS_PARAM);

    RGY_MEMSET_ZERO(m_DecInputBitstream);

    RGY_MEMSET_ZERO(m_mfxDecParams);
    RGY_MEMSET_ZERO(m_mfxEncParams);

    RGY_MEMSET_ZERO(m_DecVidProc);
}

CQSVPipeline::~CQSVPipeline() {
    Close();
}

void CQSVPipeline::SetAbortFlagPointer(bool *abortFlag) {
    m_pAbortByUser = abortFlag;
}

RGY_ERR CQSVPipeline::readChapterFile(tstring chapfile) {
#if ENABLE_AVSW_READER
    ChapterRW chapter;
    auto err = chapter.read_file(chapfile.c_str(), CODE_PAGE_UNSET, 0.0);
    if (err != AUO_CHAP_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("failed to %s chapter file: \"%s\".\n"), (err == AUO_CHAP_ERR_FILE_OPEN) ? _T("open") : _T("read"), chapfile.c_str());
        return RGY_ERR_UNKNOWN;
    }
    if (chapter.chapterlist().size() == 0) {
        PrintMes(RGY_LOG_ERROR, _T("no chapter found from chapter file: \"%s\".\n"), chapfile.c_str());
        return RGY_ERR_UNKNOWN;
    }
    m_Chapters.clear();
    const auto& chapter_list = chapter.chapterlist();
    tstring chap_log;
    for (size_t i = 0; i < chapter_list.size(); i++) {
        unique_ptr<AVChapter> avchap(new AVChapter);
        avchap->time_base = av_make_q(1, 1000);
        avchap->start = chapter_list[i]->get_ms();
        avchap->end = (i < chapter_list.size()-1) ? chapter_list[i+1]->get_ms() : avchap->start + 1;
        avchap->id = (int)m_Chapters.size();
        avchap->metadata = nullptr;
        av_dict_set(&avchap->metadata, "title", chapter_list[i]->name.c_str(), 0); //chapter_list[i]->nameはUTF-8になっている
        chap_log += strsprintf(_T("chapter #%02d [%d.%02d.%02d.%03d]: %s.\n"),
            avchap->id, chapter_list[i]->h, chapter_list[i]->m, chapter_list[i]->s, chapter_list[i]->ms,
            char_to_tstring(chapter_list[i]->name, CODE_PAGE_UTF8).c_str()); //chapter_list[i]->nameはUTF-8になっている
        m_Chapters.push_back(std::move(avchap));
    }
    PrintMes(RGY_LOG_DEBUG, _T("%s"), chap_log.c_str());
    return RGY_ERR_NONE;
#else
    PrintMes(RGY_LOG_ERROR, _T("chater reading unsupported in this build"));
    return RGY_ERR_UNKNOWN;
#endif //#if ENABLE_AVSW_READER
}

RGY_ERR CQSVPipeline::InitChapters(const sInputParams *inputParam) {
#if ENABLE_AVSW_READER
    m_Chapters.clear();
    if (inputParam->common.chapterFile.length() > 0) {
        //チャプターファイルを読み込む
        auto chap_sts = readChapterFile(inputParam->common.chapterFile);
        if (chap_sts != RGY_ERR_NONE) {
            return chap_sts;
        }
    }
    if (m_Chapters.size() == 0) {
        auto pAVCodecReader = std::dynamic_pointer_cast<RGYInputAvcodec>(m_pFileReader);
        if (pAVCodecReader != nullptr) {
            auto chapterList = pAVCodecReader->GetChapterList();
            //入力ファイルのチャプターをコピーする
            for (uint32_t i = 0; i < chapterList.size(); i++) {
                unique_ptr<AVChapter> avchap(new AVChapter);
                *avchap = *chapterList[i];
                m_Chapters.push_back(std::move(avchap));
            }
        }
    }
    if (m_Chapters.size() > 0) {
        //if (inputParam->common.keyOnChapter && m_trimParam.list.size() > 0) {
        //    PrintMes(RGY_LOG_WARN, _T("--key-on-chap not supported when using --trim.\n"));
        //} else {
        //    m_keyOnChapter = inputParam->common.keyOnChapter;
        //}
    }
#endif //#if ENABLE_AVSW_READER
    return RGY_ERR_NONE;
}

int CQSVPipeline::getEncoderBitdepth(const sInputParams *pParams) {
    int encodeBitDepth = 8;
    switch (pParams->CodecId) {
    case MFX_CODEC_AVC: break;
    case MFX_CODEC_VP8: break;
    case MFX_CODEC_VP9: break;
    case MFX_CODEC_MPEG2: break;
    case MFX_CODEC_VC1: break;
    case MFX_CODEC_HEVC:
        if (pParams->CodecProfile == MFX_PROFILE_HEVC_MAIN10) {
            encodeBitDepth = 10;
        }
        break;
    default:
        PrintMes(RGY_LOG_ERROR, _T("Unknown codec.\n"));
        return 0;
    }
    return encodeBitDepth;
}

RGY_CSP CQSVPipeline::getEncoderCsp(const sInputParams *pParams, int *pShift) {
    if (pParams->CodecId == MFX_CODEC_HEVC && pParams->CodecProfile == MFX_PROFILE_HEVC_MAIN10) {
        if (pShift) {
            *pShift = 6;
        }
        return RGY_CSP_P010;
    }
    if (pShift) {
        *pShift = 0;
    }
    return RGY_CSP_NV12;
}

RGY_ERR CQSVPipeline::InitOutput(sInputParams *inputParams) {
    auto [err, outFrameInfo] = GetOutputVideoInfo();
    if (err != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to get output frame info!\n"));
        return err;
    }
    const auto outputVideoInfo = (outFrameInfo->isVppParam) ? videooutputinfo(outFrameInfo->videoPrmVpp.vpp.Out) : videooutputinfo(outFrameInfo->videoPrm.mfx, m_VideoSignalInfo, m_chromalocInfo);
    if (outputVideoInfo.codec == RGY_CODEC_UNKNOWN) {
        inputParams->common.AVMuxTarget &= ~RGY_MUX_VIDEO;
    }
    m_HDRSei = createHEVCHDRSei(inputParams->common.maxCll, inputParams->common.masterDisplay, inputParams->common.atcSei, m_pFileReader.get());
    if (!m_HDRSei) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to parse HEVC HDR10 metadata.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    err = initWriters(m_pFileWriter, m_pFileWriterListAudio, m_pFileReader, m_AudioReaders,
        &inputParams->common, &inputParams->input, &inputParams->ctrl, outputVideoInfo,
        m_trimParam, m_outputTimebase,
#if ENABLE_AVSW_READER
        m_Chapters,
#endif //#if ENABLE_AVSW_READER
        m_HDRSei.get(),
        !check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_6),
        inputParams->bBenchmark,
        m_pStatus, m_pPerfMonitor, m_pQSVLog);
    if (err != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("failed to initialize file reader(s).\n"));
        return err;
    }
    if (inputParams->common.timecode) {
        m_timecode = std::make_unique<RGYTimecode>();
        const auto tcfilename = (inputParams->common.timecodeFile.length() > 0) ? inputParams->common.timecodeFile : PathRemoveExtensionS(inputParams->common.outputFilename) + _T(".timecode.txt");
        err = m_timecode->init(tcfilename);
        if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("failed to open timecode file: \"%s\".\n"), tcfilename.c_str());
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::InitInput(sInputParams *inputParam) {
#if ENABLE_RAW_READER
#if ENABLE_AVSW_READER
    DeviceCodecCsp HWDecCodecCsp;
    HWDecCodecCsp.push_back(std::make_pair(0, getHWDecCodecCsp(m_pQSVLog, inputParam->ctrl.skipHWDecodeCheck)));
#endif
    m_pStatus.reset(new EncodeStatus());

    int subburnTrackId = 0;
#if ENCODER_NVENC
    if (inputParam->common.nSubtitleSelectCount > 0 && inputParam->vpp.subburn.size() > 0) {
        PrintMes(RGY_LOG_ERROR, _T("--sub-copy and --vpp-subburn should not be set at the same time.\n"));
        return MFX_ERR_UNKNOWN;
    }
    for (const auto &subburn : inputParam->vpp.subburn) {
        if (subburn.trackId > 0) {
            subburnTrackId = subburn.trackId;
            break;
        }
    }
#endif

    //--input-cspの値 (raw読み込み用の入力色空間)
    //この後上書きするので、ここで保存する
    const auto inputCspOfRawReader = inputParam->input.csp;

    //入力モジュールが、エンコーダに返すべき色空間をセット
    inputParam->input.csp = getEncoderCsp(inputParam, &inputParam->input.shift);

    auto sts = initReaders(m_pFileReader, m_AudioReaders, &inputParam->input, inputCspOfRawReader,
        m_pStatus, &inputParam->common, &inputParam->ctrl, HWDecCodecCsp, subburnTrackId,
#if ENCODER_NVENC
        inputParam->vpp.rff, inputParam->vpp.afs.enable,
#else
        false, false,
#endif
        nullptr, m_pPerfMonitor.get(), m_pQSVLog);
    if (sts != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("failed to initialize file reader(s).\n"));
        return sts;
    }
    PrintMes(RGY_LOG_DEBUG, _T("initReaders: Success.\n"));

    m_inputFps = rgy_rational<int>(inputParam->input.fpsN, inputParam->input.fpsD);
    m_outputTimebase = m_inputFps.inv() * rgy_rational<int>(1, 4);
    if (m_nAVSyncMode & RGY_AVSYNC_VFR) {
        //avsync vfr時は、入力streamのtimebaseをそのまま使用する
        m_outputTimebase = m_pFileReader->getInputTimebase();
    }

    if (
#if ENABLE_AVSW_READER
        std::dynamic_pointer_cast<RGYInputAvcodec>(m_pFileReader) == nullptr &&
#endif
        inputParam->common.pTrimList && inputParam->common.nTrimCount > 0) {
        //avhw/avswリーダー以外は、trimは自分ではセットされないので、ここでセットする
        sTrimParam trimParam;
        trimParam.list = make_vector(inputParam->common.pTrimList, inputParam->common.nTrimCount);
        trimParam.offset = 0;
        m_pFileReader->SetTrimParam(trimParam);
    }
    //trim情報をリーダーから取得する
    m_trimParam = m_pFileReader->GetTrimParam();
    if (m_trimParam.list.size() > 0) {
        PrintMes(RGY_LOG_DEBUG, _T("Input: trim options\n"));
        for (int i = 0; i < (int)m_trimParam.list.size(); i++) {
            PrintMes(RGY_LOG_DEBUG, _T("%d-%d "), m_trimParam.list[i].start, m_trimParam.list[i].fin);
        }
        PrintMes(RGY_LOG_DEBUG, _T(" (offset: %d)\n"), m_trimParam.offset);
    }

#if ENABLE_AVSW_READER
    auto pAVCodecReader = std::dynamic_pointer_cast<RGYInputAvcodec>(m_pFileReader);
    if ((m_nAVSyncMode & (RGY_AVSYNC_VFR | RGY_AVSYNC_FORCE_CFR))
#if ENCODER_NVENC
        || inputParam->vpp.rff
#endif
        ) {
        tstring err_target;
        if (m_nAVSyncMode & RGY_AVSYNC_VFR)       err_target += _T("avsync vfr, ");
        if (m_nAVSyncMode & RGY_AVSYNC_FORCE_CFR) err_target += _T("avsync forcecfr, ");
#if ENCODER_NVENC
        if (inputParam->vpp.rff)                  err_target += _T("vpp-rff, ");
#endif
        err_target = err_target.substr(0, err_target.length()-2);

        if (pAVCodecReader) {
            //timestampになんらかの問題がある場合、vpp-rffとavsync vfrは使用できない
            const auto timestamp_status = pAVCodecReader->GetFramePosList()->getStreamPtsStatus();
            if ((timestamp_status & (~RGY_PTS_NORMAL)) != 0) {

                tstring err_sts;
                if (timestamp_status & RGY_PTS_SOMETIMES_INVALID) err_sts += _T("SOMETIMES_INVALID, "); //時折、無効なptsを得る
                if (timestamp_status & RGY_PTS_HALF_INVALID)      err_sts += _T("HALF_INVALID, "); //PAFFなため、半分のフレームのptsやdtsが無効
                if (timestamp_status & RGY_PTS_ALL_INVALID)       err_sts += _T("ALL_INVALID, "); //すべてのフレームのptsやdtsが無効
                if (timestamp_status & RGY_PTS_NONKEY_INVALID)    err_sts += _T("NONKEY_INVALID, "); //キーフレーム以外のフレームのptsやdtsが無効
                if (timestamp_status & RGY_PTS_DUPLICATE)         err_sts += _T("PTS_DUPLICATE, "); //重複するpts/dtsが存在する
                if (timestamp_status & RGY_DTS_SOMETIMES_INVALID) err_sts += _T("DTS_SOMETIMES_INVALID, "); //時折、無効なdtsを得る
                err_sts = err_sts.substr(0, err_sts.length()-2);

                PrintMes(RGY_LOG_ERROR, _T("timestamp not acquired successfully from input stream, %s cannot be used. \n  [0x%x] %s\n"),
                    err_target.c_str(), (uint32_t)timestamp_status, err_sts.c_str());
                return RGY_ERR_UNKNOWN;
            }
            PrintMes(RGY_LOG_DEBUG, _T("timestamp check: 0x%x\n"), timestamp_status);
        } else if (m_outputTimebase.n() == 0 || !m_outputTimebase.is_valid()) {
            PrintMes(RGY_LOG_ERROR, _T("%s cannot be used with current reader.\n"), err_target.c_str());
            return RGY_ERR_UNKNOWN;
        }
    } else if (pAVCodecReader && ((pAVCodecReader->GetFramePosList()->getStreamPtsStatus() & (~RGY_PTS_NORMAL)) == 0)) {
#if !ENCODER_QSV
        m_nAVSyncMode |= RGY_AVSYNC_VFR;
        const auto timebaseStreamIn = to_rgy(pAVCodecReader->GetInputVideoStream()->time_base);
        if ((timebaseStreamIn.inv() * m_inputFps.inv()).d() == 1 || timebaseStreamIn.n() > 1000) { //fpsを割り切れるtimebaseなら
#if ENCODER_NVENC
            if (!inputParam->vpp.afs.enable && !inputParam->vpp.rff) {
                m_outputTimebase = m_inputFps.inv() * rgy_rational<int>(1, 8);
            }
#endif
        }
        PrintMes(RGY_LOG_DEBUG, _T("vfr mode automatically enabled with timebase %d/%d\n"), m_outputTimebase.n(), m_outputTimebase.d());
#endif
    }
#if 0
    if (inputParam->common.dynamicHdr10plusJson.length() > 0) {
        m_hdr10plus = initDynamicHDR10Plus(inputParam->common.dynamicHdr10plusJson, m_pNVLog);
        if (!m_hdr10plus) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to initialize hdr10plus reader.\n"));
            return RGY_ERR_UNKNOWN;
        }
    }
#endif
#endif //#if ENABLE_AVSW_READER
    return RGY_ERR_NONE;
#else
    return RGY_ERR_UNSUPPORTED;
#endif //#if ENABLE_RAW_READER
}

RGY_ERR CQSVPipeline::CheckParam(sInputParams *inputParam) {
    const auto inputFrameInfo = m_pFileReader->GetInputFrameInfo();

    if ((inputParam->memType & HW_MEMORY)
        && (inputFrameInfo.csp == RGY_CSP_NV16 || inputFrameInfo.csp == RGY_CSP_P210)) {
        PrintMes(RGY_LOG_WARN, _T("Currently yuv422 surfaces are not supported by d3d9/d3d11 memory.\n"));
        PrintMes(RGY_LOG_WARN, _T("Switching to system memory.\n"));
        inputParam->memType = SYSTEM_MEMORY;
    }

    //デコードを行う場合は、入力バッファサイズを常に1に設定する (そうしないと正常に動かない)
    //また、バッファサイズを拡大しても特に高速化しない
    if (m_pFileReader->getInputCodec() != RGY_CODEC_UNKNOWN) {
        inputParam->nInputBufSize = 1;
        //Haswell以前はHEVCデコーダを使用する場合はD3D11メモリを使用しないと正常に稼働しない (4080ドライバ)
        if (getCPUGen(&m_mfxSession) <= CPU_GEN_HASWELL && m_pFileReader->getInputCodec() == RGY_CODEC_HEVC) {
            if (inputParam->memType & D3D9_MEMORY) {
                inputParam->memType &= ~D3D9_MEMORY;
                inputParam->memType |= D3D11_MEMORY;
            }
            PrintMes(RGY_LOG_DEBUG, _T("Switched to d3d11 mode for HEVC decoding on Haswell.\n"));
        }
    }

    // 解像度の条件とcrop
    int h_mul = 2;
    bool output_interlaced = ((inputParam->input.picstruct & RGY_PICSTRUCT_INTERLACED) != 0 && !inputParam->vppmfx.deinterlace);
    if (output_interlaced) {
        h_mul *= 2;
    }
    // crop設定の確認
    if (inputParam->input.crop.e.left % 2 != 0 || inputParam->input.crop.e.right % 2 != 0) {
        PrintMes(RGY_LOG_ERROR, _T("crop width should be a multiple of 2.\n"));
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }
    if (inputParam->input.crop.e.bottom % h_mul != 0 || inputParam->input.crop.e.up % h_mul != 0) {
        PrintMes(RGY_LOG_ERROR, _T("crop height should be a multiple of %d.\n"));
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }
    if (0 == inputParam->input.srcWidth || 0 == inputParam->input.srcHeight) {
        PrintMes(RGY_LOG_ERROR, _T("--input-res must be specified with raw input.\n"));
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }
    if (inputParam->input.fpsN == 0 || inputParam->input.fpsD == 0) {
        PrintMes(RGY_LOG_ERROR, _T("--fps must be specified with raw input.\n"));
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }
    if (inputParam->input.srcWidth < (inputParam->input.crop.e.left + inputParam->input.crop.e.right)
        || inputParam->input.srcHeight < (inputParam->input.crop.e.bottom + inputParam->input.crop.e.up)) {
        PrintMes(RGY_LOG_ERROR, _T("crop size is too big.\n"));
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }

    //解像度の自動設定
    auto outpar = std::make_pair(inputParam->nPAR[0], inputParam->nPAR[1]);
    if ((!inputParam->nPAR[0] || !inputParam->nPAR[1]) //SAR比の指定がない
        && inputParam->input.sar[0] && inputParam->input.sar[1] //入力側からSAR比を取得ずみ
        && (inputParam->input.dstWidth == inputParam->input.srcWidth && inputParam->input.dstHeight == inputParam->input.srcHeight)) {//リサイズは行われない
        outpar = std::make_pair(inputParam->input.sar[0], inputParam->input.sar[1]);
    }
    if (inputParam->input.dstWidth < 0 && inputParam->input.dstHeight < 0) {
        PrintMes(RGY_LOG_ERROR, _T("Either one of output resolution must be positive value.\n"));
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }

    set_auto_resolution(inputParam->input.dstWidth, inputParam->input.dstHeight, outpar.first, outpar.second,
        inputParam->input.srcWidth, inputParam->input.srcHeight, inputParam->input.sar[0], inputParam->input.sar[1], inputParam->input.crop);

    // 解像度の条件とcrop
    if (inputParam->input.dstWidth % 2 != 0) {
        PrintMes(RGY_LOG_ERROR, _T("output width should be a multiple of 2.\n"));
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }

    if (inputParam->input.dstHeight % h_mul != 0) {
        PrintMes(RGY_LOG_ERROR, _T("output height should be a multiple of %d.\n"), h_mul);
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }

    //入力バッファサイズの範囲チェック
    inputParam->nInputBufSize = (mfxU16)clamp_param_int(inputParam->nInputBufSize, QSV_INPUT_BUF_MIN, QSV_INPUT_BUF_MAX, _T("input-buf"));

    return RGY_ERR_NONE;
}

std::vector<VppType> CQSVPipeline::InitFiltersCreateVppList(sInputParams *inputParam, const bool cropRequired, const bool resizeRequired) {
    std::vector<VppType> filterPipeline;
    filterPipeline.reserve((size_t)VppType::CL_MAX);

    if (cropRequired)                      filterPipeline.push_back(VppType::CL_CROP);
    if (inputParam->vpp.colorspace.enable) filterPipeline.push_back(VppType::CL_COLORSPACE);
    if (inputParam->vpp.afs.enable)        filterPipeline.push_back(VppType::CL_AFS);
    if (inputParam->vpp.nnedi.enable)      filterPipeline.push_back(VppType::CL_NNEDI);
    if (inputParam->vppmfx.deinterlace != MFX_DEINTERLACE_NONE)  filterPipeline.push_back(VppType::MFX_DEINTERLACE);
    if (inputParam->vpp.knn.enable)        filterPipeline.push_back(VppType::CL_DENOISE_KNN);
    if (inputParam->vpp.pmd.enable)        filterPipeline.push_back(VppType::CL_DENOISE_PMD);
    if (inputParam->vpp.smooth.enable)     filterPipeline.push_back(VppType::CL_DENOISE_SMOOTH);
    if (inputParam->vppmfx.denoise.enable) filterPipeline.push_back(VppType::MFX_DENOISE);
    if (inputParam->vpp.subburn.size()>0)  filterPipeline.push_back(VppType::CL_SUBBURN);
    if (resizeRequired)                    filterPipeline.push_back(VppType::CL_RESIZE);
    if (inputParam->vpp.unsharp.enable)    filterPipeline.push_back(VppType::CL_UNSHARP);
    if (inputParam->vpp.edgelevel.enable)  filterPipeline.push_back(VppType::CL_EDGELEVEL);
    if (inputParam->vpp.warpsharp.enable)  filterPipeline.push_back(VppType::CL_WARPSHARP);
    if (inputParam->vpp.tweak.enable)      filterPipeline.push_back(VppType::CL_TWEAK);
    if (inputParam->vppmfx.detail.enable)  filterPipeline.push_back(VppType::MFX_DETAIL_ENHANCE);
    if (inputParam->vppmfx.mctf.enable)    filterPipeline.push_back(VppType::MFX_MCTF);
    if (inputParam->vpp.transform.enable)  filterPipeline.push_back(VppType::CL_TRANSFORM);
    if (inputParam->vpp.tweak.enable)      filterPipeline.push_back(VppType::CL_TWEAK);
    if (inputParam->vpp.deband.enable)     filterPipeline.push_back(VppType::CL_DEBAND);

    if (filterPipeline.size() == 0) {
        return filterPipeline;
    }

    // cropとresizeはmfxとopencl両方ともあるので、前後のフィルタのどちらかがmfxだったら、そちらに合わせる
    for (size_t i = 0; i < filterPipeline.size(); i++) {
        if (filterPipeline[i] == VppType::CL_CROP
            || filterPipeline[i] == VppType::CL_RESIZE) {
            const VppFilterType prev = (i >= 1)                        ? getVppFilterType(filterPipeline[i - 1]) : VppFilterType::FILTER_NONE;
            const VppFilterType next = (i + 1 < filterPipeline.size()) ? getVppFilterType(filterPipeline[i + 1]) : VppFilterType::FILTER_NONE;
            if (prev == VppFilterType::FILTER_MFX
                || next == VppFilterType::FILTER_MFX
                || (prev == next && prev == VppFilterType::FILTER_NONE)) {
                if (filterPipeline[i] == VppType::CL_CROP) {
                    filterPipeline[i] = VppType::MFX_CROP;
                }
                if (filterPipeline[i] == VppType::CL_RESIZE) {
                    filterPipeline[i] = VppType::MFX_RESIZE;
                }
            }
        }
    }
    return filterPipeline;
}

std::pair<RGY_ERR, std::unique_ptr<QSVVppMfx>> CQSVPipeline::AddFilterMFX(
    FrameInfo& frameInfo, VideoVUIInfo& vuiIn, rgy_rational<int>& fps,
    const VppType vppType, const sInputParams *inputParam, sInputCrop *crop, const int blockSize) {
    auto params = &inputParam->vppmfx;
    FrameInfo frameIn = frameInfo;
    sVppParams vppParams;
    vppParams.bEnable = true;
    switch (vppType) {
    case VppType::MFX_CROP: break;
    case VppType::MFX_DEINTERLACE:         vppParams.deinterlace = params->deinterlace; break;
    case VppType::MFX_DENOISE:             vppParams.denoise = params->denoise; break;
    case VppType::MFX_DETAIL_ENHANCE:      vppParams.detail = params->detail; break;
    case VppType::MFX_IMAGE_STABILIZATION: vppParams.imageStabilizer = params->imageStabilizer; break;
    case VppType::MFX_ROTATE:              vppParams.rotate = params->rotate; break;
    case VppType::MFX_MIRROR:              vppParams.mirrorType = params->mirrorType; break;
    case VppType::MFX_MCTF:                vppParams.mctf = params->mctf; break;
    case VppType::MFX_RESIZE:              vppParams.bUseResize = true;
                                           vppParams.scalingQuality = params->scalingQuality;
                                           frameInfo.width = inputParam->input.dstWidth;
                                           frameInfo.height = inputParam->input.dstHeight; break;

    case VppType::MFX_FPS_CONV:
    case VppType::MFX_COLORSPACE:
    default:
        return { RGY_ERR_UNSUPPORTED, std::unique_ptr<QSVVppMfx>() };
    }

    mfxIMPL impl;
    m_mfxSession.QueryIMPL(&impl);
    auto mfxvpp = std::make_unique<QSVVppMfx>(m_hwdev, m_pMFXAllocator.get(), m_mfxVer, impl, m_memType, m_nAsyncDepth, m_pQSVLog);
    auto err = mfxvpp->SetParam(vppParams, {}, frameInfo, m_encVUI, frameIn, vuiIn, (vppType == VppType::MFX_CROP) ? crop : nullptr,
        fps, rgy_rational<int>(1,1), blockSize);
    if (err != RGY_ERR_NONE) {
        return { err, std::unique_ptr<QSVVppMfx>() };
    }

    if (mfxvpp->GetVppList().size() == 0) {
        PrintMes(RGY_LOG_WARN, _T("filtering has no action.\n"));
        return { err, std::unique_ptr<QSVVppMfx>() };
    }

    //入力フレーム情報を更新
    frameInfo = mfxvpp->GetFrameOut();
    fps = mfxvpp->GetOutFps();

    return { RGY_ERR_NONE, std::move(mfxvpp) };
}

std::pair<RGY_ERR, std::unique_ptr<RGYFilter>> CQSVPipeline::AddFilterOpenCL(FrameInfo& inputFrame, rgy_rational<int>& fps, const VppType vppType, sInputParams *inputParam) {
    auto params = &inputParam->vpp;
    
    //afs
    if (vppType == VppType::CL_AFS) {
#if 0
        if ((inputParam->input.picstruct & (RGY_PICSTRUCT_TFF | RGY_PICSTRUCT_BFF)) == 0) {
            PrintMes(RGY_LOG_ERROR, _T("Please set input interlace field order (--interlace tff/bff) for vpp-afs.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        unique_ptr<RGYFilter> filter(new RGYFilterAfs(m_cl));
        shared_ptr<RGYFilterParamAfs> param(new RGYFilterParamAfs());
        param->afs = params->afs;
        param->afs.tb_order = (inputParam->input.picstruct & RGY_PICSTRUCT_TFF) != 0;
        if (inputParam->common.timecode && param->afs.timecode) {
            param->afs.timecode = 2;
        }
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->inFps = m_inputFps;
        param->inTimebase = m_outputTimebase;
        param->outTimebase = m_outputTimebase;
        param->baseFps = m_encFps;
        param->outFilename = inputParam->common.outputFilename;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return { sts, std::unique_ptr<RGYFilter>() };
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        return { RGY_ERR_NONE, std::move(filter) };
#else
        PrintMes(RGY_LOG_ERROR, _T("vpp-afs not suported yet.\n"));
        return { RGY_ERR_UNSUPPORTED, std::unique_ptr<RGYFilter>() };
#endif
    }
    //nnedi
    if (vppType == VppType::CL_NNEDI) {
#if 0
        if ((inputParam->input.picstruct & (RGY_PICSTRUCT_TFF | RGY_PICSTRUCT_BFF)) == 0) {
            PrintMes(RGY_LOG_ERROR, _T("Please set input interlace field order (--interlace tff/bff) for vpp-nnedi.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        unique_ptr<RGYFilter> filter(new RGYFilterNnedi(m_cl));
        shared_ptr<RGYFilterParamNnedi> param(new RGYFilterParamNnedi());
        param->nnedi = params->nnedi;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return { sts, std::unique_ptr<RGYFilter>() };
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        return { RGY_ERR_NONE, std::move(filter) };
#else
        PrintMes(RGY_LOG_ERROR, _T("vpp-nnedi not suported yet.\n"));
        return { RGY_ERR_UNSUPPORTED, std::unique_ptr<RGYFilter>() };
#endif
    }
    //回転
    if (vppType == VppType::CL_TRANSFORM) {
#if 0
        unique_ptr<RGYFilter> filter(new RGYFilterTransform(m_cl));
        shared_ptr<RGYFilterParamTransform> param(new RGYFilterParamTransform());
        param->trans = params->transform;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return { sts, std::unique_ptr<RGYFilter>() };
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        return { RGY_ERR_NONE, std::move(filter) };
#else
        PrintMes(RGY_LOG_ERROR, _T("vpp-transform not suported yet.\n"));
        return { RGY_ERR_UNSUPPORTED, std::unique_ptr<RGYFilter>() };
#endif
    }
    //knn
    if (vppType == VppType::CL_DENOISE_KNN) {
#if 1
        unique_ptr<RGYFilter> filter(new RGYFilterDenoiseKnn(m_cl));
        shared_ptr<RGYFilterParamDenoiseKnn> param(new RGYFilterParamDenoiseKnn());
        param->knn = params->knn;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return { sts, std::unique_ptr<RGYFilter>() };
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        return { RGY_ERR_NONE, std::move(filter) };
#else
        PrintMes(RGY_LOG_ERROR, _T("vpp-knn not suported yet.\n"));
        return { RGY_ERR_UNSUPPORTED, std::unique_ptr<RGYFilter>() };
#endif
    }
    //pmd
    if (vppType == VppType::CL_DENOISE_PMD) {
#if 0
        unique_ptr<RGYFilter> filter(new RGYFilterDenoisePmd(m_cl));
        shared_ptr<RGYFilterParamDenoisePmd> param(new RGYFilterParamDenoisePmd());
        param->pmd = params->pmd;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        return { RGY_ERR_NONE, std::move(filter) };
#else
        PrintMes(RGY_LOG_ERROR, _T("vpp-pmd not suported yet.\n"));
        return { RGY_ERR_UNSUPPORTED, std::unique_ptr<RGYFilter>() };
#endif
    }
    //smooth
    if (vppType == VppType::CL_DENOISE_SMOOTH) {
#if 0
        unique_ptr<RGYFilter> filter(new RGYFilterSmooth(m_cl));
        shared_ptr<RGYFilterParamSmooth> param(new RGYFilterParamSmooth());
        param->smooth = params->smooth;
        param->qpTableRef = nullptr;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return { sts, std::unique_ptr<RGYFilter>() };
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        return { RGY_ERR_NONE, std::move(filter) };
#else
        PrintMes(RGY_LOG_ERROR, _T("vpp-smooth not suported yet.\n"));
        return { RGY_ERR_UNSUPPORTED, std::unique_ptr<RGYFilter>() };
#endif
    }
    //字幕焼きこみ
    if (vppType == VppType::CL_SUBBURN) {
        for (const auto& subburn : params->subburn) {
#if 0 && ENABLE_AVSW_READER
            if (subburn.filename.length() > 0
                && m_trimParam.list.size() > 0) {
                PrintMes(RGY_LOG_ERROR, _T("--vpp-subburn with input as file cannot be used with --trim.\n"));
                return RGY_ERR_UNSUPPORTED;
            }
            unique_ptr<RGYFilter> filter(new RGYFilterSubburn(m_cl));
            shared_ptr<RGYFilterParamSubburn> param(new RGYFilterParamSubburn());
            param->subburn = subburn;

            auto pAVCodecReader = std::dynamic_pointer_cast<RGYInputAvcodec>(m_pFileReader);
            if (pAVCodecReader != nullptr) {
                param->videoInputStream = pAVCodecReader->GetInputVideoStream();
                param->videoInputFirstKeyPts = pAVCodecReader->GetVideoFirstKeyPts();
                for (const auto &stream : pAVCodecReader->GetInputStreamInfo()) {
                    if (stream.trackId == trackFullID(AVMEDIA_TYPE_SUBTITLE, param->subburn.trackId)) {
                        param->streamIn = stream;
                        break;
                    }
                }
                param->attachmentStreams = pAVCodecReader->GetInputAttachmentStreams();
            }
            param->videoInfo = m_pFileReader->GetInputFrameInfo();
            if (param->subburn.trackId != 0 && param->streamIn.stream == nullptr) {
                PrintMes(RGY_LOG_WARN, _T("Could not find subtitle track #%d, vpp-subburn for track #%d will be disabled.\n"),
                    param->subburn.trackId, param->subburn.trackId);
            } else {
                param->bOutOverwrite = true;
                param->videoOutTimebase = av_make_q(m_outputTimebase);
                param->frameIn = inputFrame;
                param->frameOut = inputFrame;
                param->baseFps = m_encFps;
                param->crop = inputParam->input.crop;
                auto sts = filter->init(param, m_pLog);
                if (sts != RGY_ERR_NONE) {
                    return { sts, std::unique_ptr<RGYFilter>() };
                }
                //入力フレーム情報を更新
                inputFrame = param->frameOut;
                m_encFps = param->baseFps;
                //登録
                filterPipeline.push_back(VppFilterMFXCL(VppType::CL_SUBBURN, std::move(filter)));
            }
#else
            PrintMes(RGY_LOG_ERROR, _T("--vpp-subburn not supported in this build.\n"));
            return { RGY_ERR_UNSUPPORTED, std::unique_ptr<RGYFilter>() };
#endif
        }
    }
    //リサイズ
    if (vppType == VppType::CL_RESIZE) {
#if 0
        auto filterResizeCL = std::make_unique<RGYFilterResize>(m_cl);
        {
            shared_ptr<RGYFilterParamResize> param(new RGYFilterParamResize());
            param->interp = (params->resize != RGY_VPP_RESIZE_AUTO) ? params->resize : RGY_VPP_RESIZE_SPLINE36;
            param->frameIn = inputFrame;
            param->frameOut = inputFrame;
            param->frameOut.width = resizeWidth;
            param->frameOut.height = resizeHeight;
            param->baseFps = m_encFps;
            param->bOutOverwrite = false;
            auto sts = filterResizeCL->init(param, m_pQSVLog);
            if (sts != RGY_ERR_NONE) {
                return { sts, std::unique_ptr<RGYFilter>() };
            }
            //パラメータ情報を更新
            m_pLastFilterParam = std::dynamic_pointer_cast<RGYFilterParam>(param);
            //入力フレーム情報を更新
            inputFrame = param->frameOut;
            m_encFps = param->baseFps;
        }
        //登録
        return { RGY_ERR_NONE, std::move(filter) };
#else
        PrintMes(RGY_LOG_ERROR, _T("--vpp-resize not suported yet.\n"));
        return { RGY_ERR_UNSUPPORTED, std::unique_ptr<RGYFilter>() };
#endif
    }
    //unsharp
    if (vppType == VppType::CL_UNSHARP) {
#if 0
        unique_ptr<RGYFilter> filter(new RGYFilterUnsharp(m_cl));
        shared_ptr<RGYFilterParamUnsharp> param(new RGYFilterParamUnsharp());
        param->unsharp = params->unsharp;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return { sts, std::unique_ptr<RGYFilter>() };
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        return { RGY_ERR_NONE, std::move(filter) };
#else
        PrintMes(RGY_LOG_ERROR, _T("--vpp-unsharp not suported yet.\n"));
        return { RGY_ERR_UNSUPPORTED, std::unique_ptr<RGYFilter>() };
#endif
    }
    //edgelevel
    if (vppType == VppType::CL_EDGELEVEL) {
#if 0
        unique_ptr<RGYFilter> filter(new RGYFilterEdgelevel(m_cl));
        shared_ptr<RGYFilterParamEdgelevel> param(new RGYFilterParamEdgelevel());
        param->edgelevel = params->edgelevel;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return { sts, std::unique_ptr<RGYFilter>() };
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        return { RGY_ERR_NONE, std::move(filter) };
#else
        PrintMes(RGY_LOG_ERROR, _T("--vpp-edgelevel not suported yet.\n"));
        return { RGY_ERR_UNSUPPORTED, std::unique_ptr<RGYFilter>() };
#endif
    }
    //warpsharp
    if (vppType == VppType::CL_WARPSHARP) {
#if 0
        unique_ptr<RGYFilter> filter(new RGYFilterWarpsharp(m_cl));
        shared_ptr<RGYFilterParamWarpsharp> param(new RGYFilterParamWarpsharp());
        param->warpsharp = params->warpsharp;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        return { RGY_ERR_NONE, std::move(filter) };
#else
        PrintMes(RGY_LOG_ERROR, _T("--vpp-warpsharp not suported yet.\n"));
        return { RGY_ERR_UNSUPPORTED, std::unique_ptr<RGYFilter>() };
#endif
    }

    //tweak
    if (vppType == VppType::CL_TWEAK) {
#if 0
        unique_ptr<RGYFilter> filter(new RGYFilterTweak(m_cl));
        shared_ptr<RGYFilterParamTweak> param(new RGYFilterParamTweak());
        param->tweak = params->tweak;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = true;
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return { sts, std::unique_ptr<RGYFilter>() };
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        return { RGY_ERR_NONE, std::move(filter) };
#else
        PrintMes(RGY_LOG_ERROR, _T("--vpp-tweak not suported yet.\n"));
        return { RGY_ERR_UNSUPPORTED, std::unique_ptr<RGYFilter>() };
#endif
    }
    //deband
    if (vppType == VppType::CL_DEBAND) {
#if 0
        unique_ptr<RGYFilter> filter(new RGYFilterDeband(m_cl));
        shared_ptr<RGYFilterParamDeband> param(new RGYFilterParamDeband());
        param->deband = params->deband;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return { sts, std::unique_ptr<RGYFilter>() };
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        return { RGY_ERR_NONE, std::move(filter) };
#else
        PrintMes(RGY_LOG_ERROR, _T("--vpp-deband not suported yet.\n"));
        return { RGY_ERR_UNSUPPORTED, std::unique_ptr<RGYFilter>() };
#endif
    }
    //padding
    if (vppType == VppType::CL_PAD) {
#if 0
        unique_ptr<RGYFilter> filter(new RGYFilterPad(m_cl));
        shared_ptr<RGYFilterParamPad> param(new RGYFilterParamPad());
        param->pad = params->pad;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->frameOut.width = m_encWidth;
        param->frameOut.height = m_encHeight;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return { sts, std::unique_ptr<RGYFilter>() };
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        return { RGY_ERR_NONE, std::move(filter) };
#else
        PrintMes(RGY_LOG_ERROR, _T("--vpp-pad not suported yet.\n"));
        return { RGY_ERR_UNSUPPORTED, std::unique_ptr<RGYFilter>() };
#endif
    }

    PrintMes(RGY_LOG_ERROR, _T("Unknown filter type.\n"));
    return { RGY_ERR_UNSUPPORTED, std::unique_ptr<RGYFilter>() };
}

RGY_ERR CQSVPipeline::InitFilters(sInputParams *inputParam) {
    const bool cropRequired = cropEnabled(inputParam->input.crop)
        && m_pFileReader->getInputCodec() != RGY_CODEC_UNKNOWN;

    FrameInfo inputFrame;
    inputFrame.width = inputParam->input.srcWidth;
    inputFrame.height = inputParam->input.srcHeight;
    inputFrame.csp = inputParam->input.csp;
    inputFrame.picstruct = inputParam->input.picstruct;
    inputFrame.bitdepth = RGY_CSP_BIT_DEPTH[inputParam->input.csp] - inputParam->input.shift;
    const auto input_sar = rgy_rational<int>(inputParam->input.sar[0], inputParam->input.sar[1]);
    const int croppedWidth = inputFrame.width - inputParam->input.crop.e.left - inputParam->input.crop.e.right;
    const int croppedHeight = inputFrame.height - inputParam->input.crop.e.bottom - inputParam->input.crop.e.up;
    if (!cropRequired) {
        //入力時にcrop済み
        inputFrame.width = croppedWidth;
        inputFrame.height = croppedHeight;
    }

    //出力解像度が設定されていない場合は、入力解像度と同じにする
    if (inputParam->input.dstWidth == 0) {
        inputParam->input.dstWidth = croppedWidth;
    }
    if (inputParam->input.dstHeight == 0) {
        inputParam->input.dstHeight = croppedHeight;
    }
    if (m_pFileReader->getInputCodec() != RGY_CODEC_UNKNOWN) {
        inputFrame.mem_type = RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED;
    }

    //リサイザの出力すべきサイズ
    int resizeWidth = croppedWidth;
    int resizeHeight = croppedHeight;
    m_encWidth = resizeWidth;
    m_encHeight = resizeHeight;
    if (inputParam->vpp.pad.enable) {
        m_encWidth += inputParam->vpp.pad.right + inputParam->vpp.pad.left;
        m_encHeight += inputParam->vpp.pad.bottom + inputParam->vpp.pad.top;
    }

    //指定のリサイズがあればそのサイズに設定する
    if (inputParam->input.dstWidth > 0 && inputParam->input.dstHeight > 0) {
        m_encWidth = inputParam->input.dstWidth;
        m_encHeight = inputParam->input.dstHeight;
        resizeWidth = m_encWidth;
        resizeHeight = m_encHeight;
        if (inputParam->vpp.pad.enable) {
            resizeWidth -= (inputParam->vpp.pad.right + inputParam->vpp.pad.left);
            resizeHeight -= (inputParam->vpp.pad.bottom + inputParam->vpp.pad.top);
        }
    }
    bool resizeRequired = false;
    if (croppedWidth != resizeWidth || croppedHeight != resizeHeight) {
        resizeRequired = true;
    }

    //フレームレートのチェック
    if (inputParam->input.fpsN == 0 || inputParam->input.fpsD == 0) {
        PrintMes(RGY_LOG_ERROR, _T("unable to parse fps data.\n"));
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }
    m_encFps = rgy_rational<int>(inputParam->input.fpsN, inputParam->input.fpsD);

    if (inputParam->input.picstruct & RGY_PICSTRUCT_INTERLACED) {
        if (CheckParamList(inputParam->vppmfx.deinterlace, list_deinterlace, "vpp-deinterlace") != MFX_ERR_NONE) {
            return RGY_ERR_INVALID_VIDEO_PARAM;
        }
        if (inputParam->common.AVSyncMode == RGY_AVSYNC_FORCE_CFR
            && (inputParam->vppmfx.deinterlace == MFX_DEINTERLACE_IT
                || inputParam->vppmfx.deinterlace == MFX_DEINTERLACE_IT_MANUAL
                || inputParam->vppmfx.deinterlace == MFX_DEINTERLACE_BOB
                || inputParam->vppmfx.deinterlace == MFX_DEINTERLACE_AUTO_DOUBLE)) {
            PrintMes(RGY_LOG_ERROR, _T("--avsync forcecfr cannnot be used with deinterlace %s.\n"), get_chr_from_value(list_deinterlace, inputParam->vppmfx.deinterlace));
            return RGY_ERR_INVALID_VIDEO_PARAM;
        }
    }

    //インタレ解除の個数をチェック
    int deinterlacer = 0;
    if (inputParam->vppmfx.deinterlace != MFX_DEINTERLACE_NONE) deinterlacer++;
    if (inputParam->vpp.afs.enable) deinterlacer++;
    if (inputParam->vpp.nnedi.enable) deinterlacer++;
    //if (inputParam->vpp.yadif.enable) deinterlacer++;
    if (deinterlacer >= 2) {
        PrintMes(RGY_LOG_ERROR, _T("Activating 2 or more deinterlacer is not supported.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    //picStructの設定
    m_encPicstruct = (deinterlacer > 0) ? RGY_PICSTRUCT_FRAME : inputParam->input.picstruct;

    std::vector<VppType> filterPipeline = InitFiltersCreateVppList(inputParam, cropRequired, resizeRequired);
    if (filterPipeline.size() == 0) {
        PrintMes(RGY_LOG_DEBUG, _T("No filters required.\n"));
        return RGY_ERR_NONE;
    }

    //VUI情報
    auto VuiFiltered = inputParam->input.vui;

    m_encVUI = inputParam->common.out_vui;
    m_encVUI.apply_auto(inputParam->input.vui, m_encHeight);
    // blocksize
    const int blocksize = inputParam->CodecId == MFX_CODEC_HEVC ? 32 : 16;
    //読み込み時のcrop
    sInputCrop *inputCrop = (cropRequired) ? &inputParam->input.crop : nullptr;

    m_vpFilters.clear();
    std::vector<std::unique_ptr<RGYFilter>> vppOpenCLFilters;
    for (size_t i = 0; i < filterPipeline.size(); i++) {
        const VppFilterType ftype0 = (i >= 1)                      ? getVppFilterType(filterPipeline[i-1]) : VppFilterType::FILTER_NONE;
        const VppFilterType ftype1 =                                 getVppFilterType(filterPipeline[i+0]);
        const VppFilterType ftype2 = (i+1 < filterPipeline.size()) ? getVppFilterType(filterPipeline[i+1]) : VppFilterType::FILTER_NONE;
        if (ftype1 == VppFilterType::FILTER_MFX) {
            auto [err, vppmfx] = AddFilterMFX(inputFrame, VuiFiltered, m_encFps, filterPipeline[i], inputParam, inputCrop, blocksize);
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (vppmfx) {
                m_vpFilters.push_back(std::move(VppVilterBlock(vppmfx)));
            }
        } else if (ftype1 == VppFilterType::FILTER_OPENCL) {
            if (ftype0 != VppFilterType::FILTER_OPENCL || filterPipeline[i] == VppType::CL_CROP) { // 前のfilterがOpenCLでない場合、変換が必要
                auto filterCrop = std::make_unique<RGYFilterCspCrop>(m_cl);
                shared_ptr<RGYFilterParamCrop> param(new RGYFilterParamCrop());
                param->frameIn = inputFrame;
                param->frameOut = inputFrame;
                switch (param->frameOut.csp) {
                case RGY_CSP_NV12: param->frameOut.csp = RGY_CSP_YV12; break;
                case RGY_CSP_P010: param->frameOut.csp = RGY_CSP_YV12_16; break;
                default:
                    break;
                }
                if (inputCrop) {
                    param->crop = *inputCrop;
                    inputCrop = nullptr;
                }
                param->baseFps = m_encFps;
                param->frameOut.mem_type = RGY_MEM_TYPE_GPU;
                param->bOutOverwrite = false;
                auto sts = filterCrop->init(param, m_pQSVLog);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                //入力フレーム情報を更新
                inputFrame = param->frameOut;
                m_encFps = param->baseFps;
                vppOpenCLFilters.push_back(std::move(filterCrop));
            }
            if (filterPipeline[i] != VppType::CL_CROP) {
                auto [err, vppcl] = AddFilterOpenCL(inputFrame, m_encFps, filterPipeline[i], inputParam);
                if (err != RGY_ERR_NONE) {
                    return err;
                }
                if (vppcl) {
                    vppOpenCLFilters.push_back(std::move(vppcl));
                }
            }
            if (ftype2 != VppFilterType::FILTER_OPENCL) { // 次のfilterがOpenCLでない場合、変換が必要
                std::unique_ptr<RGYFilter> filterCrop(new RGYFilterCspCrop(m_cl));
                std::shared_ptr<RGYFilterParamCrop> param(new RGYFilterParamCrop());
                param->frameIn = inputFrame;
                param->frameOut.csp = getEncoderCsp(inputParam);
                param->frameOut.mem_type = RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED;
                param->baseFps = m_encFps;
                param->bOutOverwrite = false;
                auto sts = filterCrop->init(param, m_pQSVLog);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                //入力フレーム情報を更新
                inputFrame = param->frameOut;
                m_encFps = param->baseFps;
                //登録
                vppOpenCLFilters.push_back(std::move(filterCrop));
                // ブロックに追加する
                m_vpFilters.push_back(std::move(VppVilterBlock(vppOpenCLFilters)));
            }
        } else {
            PrintMes(RGY_LOG_ERROR, _T("Unsupported vpp filter type.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }

    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::InitSessionInitParam(int threads, int priority) {
    INIT_MFX_EXT_BUFFER(m_ThreadsParam, MFX_EXTBUFF_THREADS_PARAM);
    m_ThreadsParam.NumThread = (mfxU16)clamp_param_int(threads, 0, QSV_SESSION_THREAD_MAX, _T("session-threads"));
    m_ThreadsParam.Priority = (mfxU16)clamp_param_int(priority, MFX_PRIORITY_LOW, MFX_PRIORITY_HIGH, _T("priority"));
    m_pInitParamExtBuf[0] = &m_ThreadsParam.Header;

    RGY_MEMSET_ZERO(m_InitParam);
    m_InitParam.ExtParam = m_pInitParamExtBuf;
    m_InitParam.NumExtParam = 1;
    return RGY_ERR_NONE;
}

#if defined(_WIN32) || defined(_WIN64)
typedef decltype(GetSystemInfo)* funcGetSystemInfo;
static int nGetSystemInfoHookThreads = -1;
static std::mutex mtxGetSystemInfoHook;
static funcGetSystemInfo origGetSystemInfoFunc = nullptr;
void __stdcall GetSystemInfoHook(LPSYSTEM_INFO lpSystemInfo) {
    origGetSystemInfoFunc(lpSystemInfo);
    if (lpSystemInfo && nGetSystemInfoHookThreads > 0) {
        decltype(lpSystemInfo->dwActiveProcessorMask) mask = 0;
        const int nThreads = std::max(1, std::min(nGetSystemInfoHookThreads, (int)sizeof(lpSystemInfo->dwActiveProcessorMask) * 8));
        for (int i = 0; i < nThreads; i++) {
            mask |= ((size_t)1<<i);
        }
        lpSystemInfo->dwActiveProcessorMask = mask;
        lpSystemInfo->dwNumberOfProcessors = nThreads;
    }
}
#endif

RGY_ERR CQSVPipeline::InitSession(bool useHWLib, uint32_t memType) {
    auto err = RGY_ERR_NONE;
    m_SessionPlugins.reset();
    m_mfxSession.Close();
    PrintMes(RGY_LOG_DEBUG, _T("InitSession: Start initilaizing... memType: %s\n"), MemTypeToStr(memType));
#if defined(_WIN32) || defined(_WIN64)
    //コードの簡略化のため、静的フィールドを使うので、念のためロックをかける
    {
        std::lock_guard<std::mutex> lock(mtxGetSystemInfoHook);
        {
            nGetSystemInfoHookThreads = m_nMFXThreads;
            apihook api_hook;
            api_hook.hook(_T("kernel32.dll"), "GetSystemInfo", GetSystemInfoHook, (void **)&origGetSystemInfoFunc);
#endif

            auto InitSessionEx = [&](mfxIMPL impl, mfxVersion *verRequired) {
#if ENABLE_SESSION_THREAD_CONFIG
                if (m_ThreadsParam.NumThread != 0 || m_ThreadsParam.Priority != get_value_from_chr(list_priority, _T("normal"))) {
                    m_InitParam.Implementation = impl;
                    m_InitParam.Version = MFX_LIB_VERSION_1_15;
                    if (useHWLib) {
                        m_InitParam.GPUCopy = MFX_GPUCOPY_ON;
                    }
                    if (MFX_ERR_NONE == m_mfxSession.InitEx(m_InitParam)) {
                        return MFX_ERR_NONE;
                    } else {
                        m_ThreadsParam.NumThread = 0;
                        m_ThreadsParam.Priority = get_value_from_chr(list_priority, _T("normal"));
                    }
                }
#endif
                return err_to_rgy(m_mfxSession.Init(impl, verRequired));
            };

            if (useHWLib) {
                //とりあえず、MFX_IMPL_HARDWARE_ANYでの初期化を試みる
                mfxIMPL impl = MFX_IMPL_HARDWARE_ANY;
                m_memType = (memType) ? D3D9_MEMORY : SYSTEM_MEMORY;
#if MFX_D3D11_SUPPORT
                //Win7でD3D11のチェックをやると、
                //デスクトップコンポジションが切られてしまう問題が発生すると報告を頂いたので、
                //D3D11をWin8以降に限定
                if (!check_OS_Win8orLater()) {
                    memType &= (~D3D11_MEMORY);
                    PrintMes(RGY_LOG_DEBUG, _T("InitSession: OS is Win7, do not check for d3d11 mode.\n"));
                }

                //D3D11モードは基本的には遅い模様なので、自動モードなら切る
                if (HW_MEMORY == (memType & HW_MEMORY) && false == check_if_d3d11_necessary()) {
                    memType &= (~D3D11_MEMORY);
                    PrintMes(RGY_LOG_DEBUG, _T("InitSession: d3d11 memory mode not required, switching to d3d9 memory mode.\n"));
                }
#endif //#if MFX_D3D11_SUPPORT
                //まずd3d11モードを試すよう設定されていれば、ますd3d11を試して、失敗したらd3d9での初期化を試みる
                for (int i_try_d3d11 = 0; i_try_d3d11 < 1 + (HW_MEMORY == (memType & HW_MEMORY)); i_try_d3d11++) {
#if D3D_SURFACES_SUPPORT
#if MFX_D3D11_SUPPORT
                    if (D3D11_MEMORY & memType) {
                        if (0 == i_try_d3d11) {
                            impl |= MFX_IMPL_VIA_D3D11; //d3d11モードも試す場合は、まずd3d11モードをチェック
                            impl &= (~MFX_IMPL_HARDWARE_ANY); //d3d11モードでは、MFX_IMPL_HARDWAREをまず試す
                            impl |= MFX_IMPL_HARDWARE;
                            m_memType = D3D11_MEMORY;
                            PrintMes(RGY_LOG_DEBUG, _T("InitSession: trying to init session for d3d11 mode.\n"));
                        } else {
                            impl &= ~MFX_IMPL_VIA_D3D11; //d3d11をオフにして再度テストする
                            impl |= MFX_IMPL_VIA_D3D9;
                            m_memType = D3D9_MEMORY;
                            PrintMes(RGY_LOG_DEBUG, _T("InitSession: trying to init session for d3d9 mode.\n"));
                        }
                    } else
#endif //#if MFX_D3D11_SUPPORT
                    if (D3D9_MEMORY & memType) {
                        impl |= MFX_IMPL_VIA_D3D9; //d3d11モードも試す場合は、まずd3d11モードをチェック
                    }
#endif //#if D3D_SURFACES_SUPPORT
                    mfxVersion verRequired = MFX_LIB_VERSION_1_1;

                    err = InitSessionEx(impl, &verRequired);
                    if (err != RGY_ERR_NONE) {
                        if (impl & MFX_IMPL_HARDWARE_ANY) {  //MFX_IMPL_HARDWARE_ANYがサポートされない場合もあり得るので、失敗したらこれをオフにしてもう一回試す
                            impl &= (~MFX_IMPL_HARDWARE_ANY);
                            impl |= MFX_IMPL_HARDWARE;
                        } else if (impl & MFX_IMPL_HARDWARE) {  //MFX_IMPL_HARDWAREで失敗したら、MFX_IMPL_HARDWARE_ANYでもう一回試す
                            impl &= (~MFX_IMPL_HARDWARE);
                            impl |= MFX_IMPL_HARDWARE_ANY;
                        }
                        PrintMes(RGY_LOG_DEBUG, _T("InitSession: failed to init session for multi GPU mode, retry by single GPU mode.\n"));
                        err = err_to_rgy(m_mfxSession.Init(impl, &verRequired));
                    }

                    //成功したらループを出る
                    if (err == RGY_ERR_NONE) {
                        break;
                    }
                }
                PrintMes(RGY_LOG_DEBUG, _T("InitSession: initialized using %s memory.\n"), MemTypeToStr(m_memType));
            } else {
                mfxIMPL impl = MFX_IMPL_SOFTWARE;
                mfxVersion verRequired = MFX_LIB_VERSION_1_1;
                err = InitSessionEx(impl, &verRequired);
                m_memType = SYSTEM_MEMORY;
                PrintMes(RGY_LOG_DEBUG, _T("InitSession: initialized with system memory.\n"));
            }
#if defined(_WIN32) || defined(_WIN64)
        }
    }
#endif
    if (err != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_DEBUG, _T("InitSession: Failed to initialize session using %s memory: %s.\n"), MemTypeToStr(m_memType), get_err_mes(err));
        return err;
    }

    //使用できる最大のversionをチェック
    m_mfxSession.QueryVersion(&m_mfxVer);
    mfxIMPL impl;
    m_mfxSession.QueryIMPL(&impl);
    PrintMes(RGY_LOG_DEBUG, _T("InitSession: mfx lib version: %d.%d, impl 0x%x\n"), m_mfxVer.Major, m_mfxVer.Minor, impl);
    return err;
}

RGY_ERR CQSVPipeline::InitLog(sInputParams *pParams) {
    //ログの初期化
    m_pQSVLog.reset(new RGYLog(pParams->ctrl.logfile.c_str(), pParams->ctrl.loglevel));
    if ((pParams->ctrl.logfile.length() > 0 || pParams->common.outputFilename.length() > 0) && pParams->input.type != RGY_INPUT_FMT_SM) {
        m_pQSVLog->writeFileHeader(pParams->common.outputFilename.c_str());
    }
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::InitPerfMonitor(const sInputParams *inputParam) {
    const bool bLogOutput = inputParam->ctrl.perfMonitorSelect || inputParam->ctrl.perfMonitorSelectMatplot;
    tstring perfMonLog;
    if (bLogOutput) {
        perfMonLog = inputParam->common.outputFilename + _T("_perf.csv");
    }
    CPerfMonitorPrm perfMonitorPrm;
    if (m_pPerfMonitor->init(perfMonLog.c_str(), inputParam->pythonPath.c_str(), (bLogOutput) ? inputParam->ctrl.perfMonitorInterval : 1000,
        (int)inputParam->ctrl.perfMonitorSelect, (int)inputParam->ctrl.perfMonitorSelectMatplot,
#if defined(_WIN32) || defined(_WIN64)
        std::unique_ptr<void, handle_deleter>(OpenThread(SYNCHRONIZE | THREAD_QUERY_INFORMATION, false, GetCurrentThreadId()), handle_deleter()),
#else
        nullptr,
#endif
        m_pQSVLog, &perfMonitorPrm)) {
        PrintMes(RGY_LOG_WARN, _T("Failed to initialize performance monitor, disabled.\n"));
        m_pPerfMonitor.reset();
    }
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::SetPerfMonitorThreadHandles() {
#if ENABLE_AVSW_READER
    if (m_pPerfMonitor) {
        HANDLE thOutput = NULL;
        HANDLE thInput = NULL;
        HANDLE thAudProc = NULL;
        HANDLE thAudEnc = NULL;
        auto pAVCodecReader = std::dynamic_pointer_cast<RGYInputAvcodec>(m_pFileReader);
        if (pAVCodecReader != nullptr) {
            thInput = pAVCodecReader->getThreadHandleInput();
        }
        auto pAVCodecWriter = std::dynamic_pointer_cast<RGYOutputAvcodec>(m_pFileWriter);
        if (pAVCodecWriter != nullptr) {
            thOutput = pAVCodecWriter->getThreadHandleOutput();
            thAudProc = pAVCodecWriter->getThreadHandleAudProcess();
            thAudEnc = pAVCodecWriter->getThreadHandleAudEncode();
        }
        m_pPerfMonitor->SetThreadHandles((HANDLE)(m_EncThread.GetHandleEncThread().native_handle()), thInput, thOutput, thAudProc, thAudEnc);
    }
#endif //#if ENABLE_AVSW_READER
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::Init(sInputParams *pParams) {
    if (pParams == nullptr) {
        return RGY_ERR_NULL_PTR;
    }

    InitLog(pParams);

    RGY_ERR sts = RGY_ERR_NONE;

    if (pParams->bBenchmark) {
        pParams->common.AVMuxTarget = RGY_MUX_NONE;
        if (pParams->common.nAudioSelectCount) {
            for (int i = 0; i < pParams->common.nAudioSelectCount; i++) {
                rgy_free(pParams->common.ppAudioSelectList[i]);
            }
            rgy_free(pParams->common.ppAudioSelectList);
            pParams->common.nAudioSelectCount = 0;
            PrintMes(RGY_LOG_WARN, _T("audio copy or audio encoding disabled on benchmark mode.\n"));
        }
        if (pParams->common.nSubtitleSelectCount) {
            pParams->common.nSubtitleSelectCount = 0;
            PrintMes(RGY_LOG_WARN, _T("subtitle copy disabled on benchmark mode.\n"));
        }
        if (pParams->ctrl.perfMonitorSelect || pParams->ctrl.perfMonitorSelectMatplot) {
            pParams->ctrl.perfMonitorSelect = 0;
            pParams->ctrl.perfMonitorSelectMatplot = 0;
            PrintMes(RGY_LOG_WARN, _T("performance monitor disabled on benchmark mode.\n"));
        }
        pParams->common.muxOutputFormat = _T("raw");
        PrintMes(RGY_LOG_DEBUG, _T("Param adjusted for benchmark mode.\n"));
    }

    //メモリの指定が自動の場合、出力コーデックがrawなら、systemメモリを自動的に使用する
    if (HW_MEMORY == (pParams->memType & HW_MEMORY) && pParams->CodecId == MFX_CODEC_RAW) {
        PrintMes(RGY_LOG_DEBUG, _T("Automatically selecting system memory for output raw frames.\n"));
        pParams->memType = SYSTEM_MEMORY;
    }

    m_nMFXThreads = pParams->nSessionThreads;
    m_nAVSyncMode = pParams->common.AVSyncMode;

    sts = InitSessionInitParam(pParams->nSessionThreads, pParams->nSessionThreadPriority);
    if (sts < RGY_ERR_NONE) return sts;
    PrintMes(RGY_LOG_DEBUG, _T("InitSessionInitParam: Success.\n"));

    m_pPerfMonitor = std::make_unique<CPerfMonitor>();

    sts = InitInput(pParams);
    if (sts < RGY_ERR_NONE) return sts;
    PrintMes(RGY_LOG_DEBUG, _T("InitInput: Success.\n"));

    sts = CheckParam(pParams);
    if (sts != RGY_ERR_NONE) return sts;
    PrintMes(RGY_LOG_DEBUG, _T("CheckParam: Success.\n"));

    sts = InitSession(true, pParams->memType);
    QSV_ERR_MES(sts, _T("Failed to initialize encode session."));
    PrintMes(RGY_LOG_DEBUG, _T("InitSession: Success.\n"));

    m_SessionPlugins = std::unique_ptr<CSessionPlugins>(new CSessionPlugins(m_mfxSession));

    sts = CreateAllocator();
    if (sts < RGY_ERR_NONE) return sts;

    sts = InitOpenCL();
    if (sts < RGY_ERR_NONE) return sts;

    sts = InitMfxDecParams(pParams);
    if (sts < RGY_ERR_NONE) return sts;

    sts = InitFilters(pParams);
    if (sts < RGY_ERR_NONE) return sts;

    sts = InitMfxEncodeParams(pParams);
    if (sts < RGY_ERR_NONE) return sts;

    sts = InitChapters(pParams);
    if (sts < RGY_ERR_NONE) return sts;

    sts = InitPerfMonitor(pParams);
    if (sts < RGY_ERR_NONE) return sts;

    sts = InitOutput(pParams);
    if (sts < RGY_ERR_NONE) return sts;

    const int nPipelineElements = !!m_pmfxDEC + (int)m_vpFilters.size() + !!m_pmfxENC;
    if (nPipelineElements == 0) {
        PrintMes(RGY_LOG_ERROR, _T("None of the pipeline element (DEC,VPP,ENC) are activated!\n"));
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }
    PrintMes(RGY_LOG_DEBUG, _T("pipeline element count: %d\n"), nPipelineElements);

    m_nProcSpeedLimit = pParams->ctrl.procSpeedLimit;
    m_nAsyncDepth = clamp_param_int((pParams->ctrl.lowLatency) ? 1 : pParams->nAsyncDepth, 0, QSV_ASYNC_DEPTH_MAX, _T("async-depth"));
    if (m_nAsyncDepth == 0) {
        m_nAsyncDepth = (std::min)(QSV_DEFAULT_ASYNC_DEPTH + (nPipelineElements - 1), 8);
        PrintMes(RGY_LOG_DEBUG, _T("async depth automatically set to %d\n"), m_nAsyncDepth);
    }
    if (pParams->ctrl.lowLatency) {
        pParams->bDisableTimerPeriodTuning = false;
    }

#if defined(_WIN32) || defined(_WIN64)
    if (!pParams->bDisableTimerPeriodTuning) {
        m_bTimerPeriodTuning = true;
        timeBeginPeriod(1);
        PrintMes(RGY_LOG_DEBUG, _T("timeBeginPeriod(1)\n"));
    }
#endif //#if defined(_WIN32) || defined(_WIN64)

    if ((sts = ResetMFXComponents(pParams)) != RGY_ERR_NONE) {
        return sts;
    }
    if ((sts = SetPerfMonitorThreadHandles()) != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to set thread handles to perf monitor!\n"));
        return sts;
    }
    return RGY_ERR_NONE;
}

void CQSVPipeline::Close() {
    PrintMes(RGY_LOG_DEBUG, _T("Closing pipeline...\n"));
    m_pipelineTasks.clear();
    //PrintMes(RGY_LOG_INFO, _T("Frame number: %hd\r"), m_pFileWriter.m_nProcessedFramesNum);

    PrintMes(RGY_LOG_DEBUG, _T("Closing filters...\n"));
    m_vpFilters.clear();

    PrintMes(RGY_LOG_DEBUG, _T("Closing enc status...\n"));
    m_pStatus.reset();

    PrintMes(RGY_LOG_DEBUG, _T("Closing m_EncThread...\n"));
    m_EncThread.Close();

    PrintMes(RGY_LOG_DEBUG, _T("Closing Plugins...\n"));
    m_SessionPlugins.reset();

    m_trimParam.list.clear();
    m_trimParam.offset = 0;

    PrintMes(RGY_LOG_DEBUG, _T("Closing m_pmfxDEC/ENC/VPP...\n"));
    m_pmfxDEC.reset();
    m_pmfxENC.reset();
    m_mfxVPP.clear();
    m_vpFilters.clear();

#if ENABLE_MVC_ENCODING
    FreeMVCSeqDesc();
#endif

    m_EncExtParams.clear();

    m_DecInputBitstream.clear();

    PrintMes(RGY_LOG_DEBUG, _T("Closing TaskPool...\n"));
    m_TaskPool.Close();

    PrintMes(RGY_LOG_DEBUG, _T("Closing mfxSession...\n"));
    m_mfxSession.Close();

    PrintMes(RGY_LOG_DEBUG, _T("DeleteAllocator...\n"));
    // allocator if used as external for MediaSDK must be deleted after SDK components
    DeleteAllocator();

    m_cl.reset();

    PrintMes(RGY_LOG_DEBUG, _T("Closing audio readers (if used)...\n"));
    m_AudioReaders.clear();

    for (auto pWriter : m_pFileWriterListAudio) {
        if (pWriter) {
            if (pWriter != m_pFileWriter) {
                pWriter->Close();
                pWriter.reset();
            }
        }
    }
    m_pFileWriterListAudio.clear();

    PrintMes(RGY_LOG_DEBUG, _T("Closing writer...\n"));
    if (m_pFileWriter) {
        m_pFileWriter->Close();
        m_pFileWriter.reset();
    }

    PrintMes(RGY_LOG_DEBUG, _T("Closing reader...\n"));
    if (m_pFileReader) {
        m_pFileReader->Close();
        m_pFileReader.reset();
    }
#if defined(_WIN32) || defined(_WIN64)
    if (m_bTimerPeriodTuning) {
        timeEndPeriod(1);
        m_bTimerPeriodTuning = false;
        PrintMes(RGY_LOG_DEBUG, _T("timeEndPeriod(1)\n"));
    }
#endif //#if defined(_WIN32) || defined(_WIN64)

    m_timecode.reset();

    PrintMes(RGY_LOG_DEBUG, _T("Closing perf monitor...\n"));
    m_pPerfMonitor.reset();

    m_nMFXThreads = -1;
    m_pAbortByUser = nullptr;
    m_nAVSyncMode = RGY_AVSYNC_ASSUME_CFR;
    m_nProcSpeedLimit = 0;
#if ENABLE_AVSW_READER
    av_qsv_log_free();
#endif //#if ENABLE_AVSW_READER
    PrintMes(RGY_LOG_DEBUG, _T("Closed pipeline.\n"));
    if (m_pQSVLog.get() != nullptr) {
        m_pQSVLog->writeFileFooter();
        m_pQSVLog.reset();
    }
}

int CQSVPipeline::logTemporarilyIgnoreErrorMes() {
    //MediaSDK内のエラーをRGY_LOG_DEBUG以下の時以外には一時的に無視するようにする。
    //RGY_LOG_DEBUG以下の時にも、「無視できるエラーが発生するかもしれない」ことをログに残す。
    const auto log_level = m_pQSVLog->getLogLevel();
    if (log_level >= RGY_LOG_MORE) {
        m_pQSVLog->setLogLevel(RGY_LOG_QUIET); //一時的にエラーを無視
    } else {
        PrintMes(RGY_LOG_DEBUG, _T("ResetMFXComponents: there might be error below, but it might be internal error which could be ignored.\n"));
    }
    return log_level;
}

RGY_ERR CQSVPipeline::InitMfxEncode() {
    if (!m_pmfxENC) {
        return RGY_ERR_NONE;
    }
    const auto log_level = logTemporarilyIgnoreErrorMes();
    m_prmSetIn.vidprm = m_mfxEncParams;
    m_prmSetIn.cop = m_CodingOption;
    m_prmSetIn.cop2 = m_CodingOption2;
    m_prmSetIn.cop3 = m_CodingOption3;
    m_prmSetIn.hevc = m_ExtHEVCParam;
    auto sts = err_to_rgy(m_pmfxENC->Init(&m_mfxEncParams));
    m_pQSVLog->setLogLevel(log_level);
    if (sts == RGY_WRN_PARTIAL_ACCELERATION) {
        PrintMes(RGY_LOG_WARN, _T("partial acceleration on Encoding.\n"));
        sts = RGY_ERR_NONE;
    }
    RGY_ERR(sts, _T("Failed to initialize encoder."));
    PrintMes(RGY_LOG_DEBUG, _T("Encoder initialized.\n"));
    return sts;
}

RGY_ERR CQSVPipeline::InitMfxVpp() {
    for (auto& filterBlock : m_vpFilters) {
        if (filterBlock.type == VppFilterType::FILTER_MFX) {
            auto err = filterBlock.vppmfx->Init();
            if (err != RGY_ERR_NONE) {
                return err;
            }
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::InitMfxDec() {
    if (!m_pmfxDEC) {
        return RGY_ERR_NONE;
    }
    const auto log_level = logTemporarilyIgnoreErrorMes();
    auto sts = err_to_rgy(m_pmfxDEC->Init(&m_mfxDecParams));
    m_pQSVLog->setLogLevel(log_level);
    if (sts == MFX_WRN_PARTIAL_ACCELERATION) {
        PrintMes(RGY_LOG_WARN, _T("partial acceleration on decoding.\n"));
        sts = RGY_ERR_NONE;
    }
    RGY_ERR(sts, _T("Failed to initialize decoder.\n"));
    PrintMes(RGY_LOG_DEBUG, _T("Dec initialized.\n"));
    return sts;
}

RGY_ERR CQSVPipeline::ResetMFXComponents(sInputParams* pParams) {
    if (!pParams) {
        return RGY_ERR_NULL_PTR;
    }

    auto err = RGY_ERR_NONE;
    PrintMes(RGY_LOG_DEBUG, _T("ResetMFXComponents: Start...\n"));

    m_pipelineTasks.clear();

    if (m_pmfxENC) {
        err = err_to_rgy(m_pmfxENC->Close());
        RGY_IGNORE_STS(err, RGY_ERR_NOT_INITIALIZED);
        RGY_ERR(err, _T("Failed to reset encoder (fail on closing)."));
        PrintMes(RGY_LOG_DEBUG, _T("ResetMFXComponents: Enc closed.\n"));
    }

    for (auto& filterBlock : m_vpFilters) {
        if (filterBlock.type == VppFilterType::FILTER_MFX) {
            err = filterBlock.vppmfx->Close();
            RGY_IGNORE_STS(err, RGY_ERR_NOT_INITIALIZED);
            RGY_ERR(err, _T("Failed to reset vpp (fail on closing)."));
            PrintMes(RGY_LOG_DEBUG, _T("ResetMFXComponents: Vpp closed.\n"));
        }
    }

    if (m_pmfxDEC) {
        err = err_to_rgy(m_pmfxDEC->Close());
        RGY_IGNORE_STS(err, RGY_ERR_NOT_INITIALIZED);
        RGY_ERR(err, _T("Failed to reset decoder (fail on closing)."));
        PrintMes(RGY_LOG_DEBUG, _T("ResetMFXComponents: Dec closed.\n"));
    }

    // free allocated frames
    //DeleteFrames();
    //PrintMes(RGY_LOG_DEBUG, _T("ResetMFXComponents: Frames deleted.\n"));

    m_TaskPool.Close();

    if ((err = CreatePipeline()) != RGY_ERR_NONE) {
        return err;
    }
    if ((err = AllocFrames()) != RGY_ERR_NONE) {
        return err;
    }
    if ((err = InitMfxEncode()) != RGY_ERR_NONE) {
        return err;
    }
    if ((err = InitMfxVpp()) != RGY_ERR_NONE) {
        return err;
    }
    if ((err = InitMfxDec()) != RGY_ERR_NONE) {
        return err;
    }
#if 0
    mfxU32 nEncodedDataBufferSize = m_mfxEncParams.mfx.FrameInfo.Width * m_mfxEncParams.mfx.FrameInfo.Height * 4;
    PrintMes(RGY_LOG_DEBUG, _T("ResetMFXComponents: Creating task pool, poolSize %d, bufsize %d KB.\n"), m_nAsyncDepth, nEncodedDataBufferSize >> 10);
    err = m_TaskPool.Init(&m_mfxSession, m_pMFXAllocator.get(), m_pFileWriter, m_nAsyncDepth, nEncodedDataBufferSize);
    QSV_ERR_MES(err, _T("Failed to initialize task pool for encoding."));
    PrintMes(RGY_LOG_DEBUG, _T("ResetMFXComponents: Created task pool.\n"));
#endif
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::AllocateSufficientBuffer(mfxBitstream *pBS) {
    if (!pBS) {
        return RGY_ERR_NULL_PTR;
    }

    mfxVideoParam par = { 0 };
    auto err = err_to_rgy(m_pmfxENC->GetVideoParam(&par));
    RGY_ERR(err, _T("Failed to get required output buffer size from encoder."));

    err = err_to_rgy(mfxBitstreamExtend(pBS, par.mfx.BufferSizeInKB * 1000 * (std::max)(1, (int)par.mfx.BRCParamMultiplier)));
    if (err != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to allocate memory for output bufffer: %s\n"), get_err_mes(err));
        mfxBitstreamClear(pBS);
        return err;
    }

    return RGY_ERR_NONE;
}

//この関数がMFX_ERR_NONE以外を返すことでRunEncodeは終了処理に入る
mfxStatus CQSVPipeline::GetNextFrame(mfxFrameSurface1 **pSurface) {
    const int inputBufIdx = m_EncThread.m_nFrameGet % m_EncThread.m_nFrameBuffer;
    sInputBufSys *pInputBuf = &m_EncThread.m_InputBuf[inputBufIdx];

    //_ftprintf(stderr, "GetNextFrame: wait for %d\n", m_EncThread.m_nFrameGet);
    //_ftprintf(stderr, "wait for heInputDone, %d\n", m_EncThread.m_nFrameGet);
    PrintMes(RGY_LOG_TRACE, _T("Enc Thread: Wait Done %d.\n"), m_EncThread.m_nFrameGet);
    //HWデコードの場合、本来ここで待機する必要はなく、またここでRGY_ERR_MORE_DATAを返す必要もない
    while (WaitForSingleObject(pInputBuf->heInputDone, (m_pFileReader->getInputCodec() == RGY_CODEC_UNKNOWN) ? INFINITE : 1) == WAIT_TIMEOUT) {
        //ここに入るのはHWデコードの場合のみ
        //HWデコードの時は、本来このロックはフレーム読み込みには使用しておらず、
        //CQSVPipeline::Run()内のm_pFileReader->LoadNextFrame()による進捗管理のために行っているに過ぎない
        //そのためCQSVPipeline::Run()が終了している、
        //すなわちm_EncThread.m_stsThreadがMFX_ERR_MORE_DATAであれば、
        //特に待機せずMFX_ERR_NONEを返して終了する
        if (m_EncThread.m_stsThread == MFX_ERR_MORE_DATA) {
            return MFX_ERR_NONE;
        }
    }
    //エラー・中断要求などでの終了
    if (m_EncThread.m_bthForceAbort) {
        PrintMes(RGY_LOG_DEBUG, _T("GetNextFrame: Encode Aborted...\n"));
        return m_EncThread.m_stsThread;
    }
    //読み込み完了による終了
    if (m_EncThread.m_stsThread == MFX_ERR_MORE_DATA && m_EncThread.m_nFrameGet == m_pStatus->m_sData.frameIn) {
        PrintMes(RGY_LOG_DEBUG, _T("GetNextFrame: Frame read finished.\n"));
        return m_EncThread.m_stsThread;
    }
    //フレーム読み込みでない場合は、フレーム関連の処理は行わない
    if (m_pFileReader->getInputCodec() == RGY_CODEC_UNKNOWN) {
        *pSurface = (mfxFrameSurface1 *)pInputBuf->pFrameSurface;
        if ((m_nAVSyncMode & (RGY_AVSYNC_VFR | RGY_AVSYNC_FORCE_CFR)) == 0) {
            (*pSurface)->Data.TimeStamp = inputBufIdx;
        }
        (*pSurface)->Data.Locked = FALSE;
        m_EncThread.m_nFrameGet++;
    }
    return MFX_ERR_NONE;
}

mfxStatus CQSVPipeline::SetNextSurface(mfxFrameSurface1 *pSurface) {
    const int inputBufIdx = m_EncThread.m_nFrameSet % m_EncThread.m_nFrameBuffer;
    sInputBufSys *pInputBuf = &m_EncThread.m_InputBuf[inputBufIdx];
    //フレーム読み込みでない場合は、フレーム関連の処理は行わない
    if (m_pFileReader->getInputCodec() == RGY_CODEC_UNKNOWN) {
        //_ftprintf(stderr, "Set heInputStart: %d\n", m_EncThread.m_nFrameSet);
        pSurface->Data.Locked = TRUE;
        //_ftprintf(stderr, "set surface %d, set event heInputStart %d\n", pSurface, m_EncThread.m_nFrameSet);
        pInputBuf->pFrameSurface = (RGYFrame *)pSurface;
    }
    SetEvent(pInputBuf->heInputStart);
    PrintMes(RGY_LOG_TRACE, _T("Enc Thread: Set Start %d.\n"), m_EncThread.m_nFrameSet);
    m_EncThread.m_nFrameSet++;
    return MFX_ERR_NONE;
}

mfxStatus CQSVPipeline::GetFreeTask(QSVTask **ppTask) {
    mfxStatus sts = MFX_ERR_NONE;

    sts = m_TaskPool.GetFreeTask(ppTask);
    if (MFX_ERR_NOT_FOUND == sts) {
        sts = SynchronizeFirstTask();
        QSV_ERR_MES(sts, _T("Failed to SynchronizeFirstTask"));

        // try again
        sts = m_TaskPool.GetFreeTask(ppTask);
    }

    return sts;
}

mfxStatus CQSVPipeline::SynchronizeFirstTask() {
    mfxStatus sts = m_TaskPool.SynchronizeFirstTask();

    return sts;
}

void CQSVPipeline::RunEncThreadLauncher(void *pParam) {
    //reinterpret_cast<CQSVPipeline*>(pParam)->RunEncode();
}

RGY_ERR CQSVPipeline::Run() {
    return RunEncode2();
}

#if 0
RGY_ERR CQSVPipeline::Run(size_t SubThreadAffinityMask) {
    mfxStatus sts = MFX_ERR_NONE;

    PrintMes(RGY_LOG_DEBUG, _T("Main Thread: Lauching encode thread...\n"));
    sts = m_EncThread.RunEncFuncbyThread(&RunEncThreadLauncher, this, SubThreadAffinityMask);
    QSV_ERR_MES(sts, _T("Failed to start encode thread."));
    PrintMes(RGY_LOG_DEBUG, _T("Main Thread: Starting Encode...\n"));

#if ENABLE_AVSW_READER
    if (m_pPerfMonitor) {
        HANDLE thOutput = NULL;
        HANDLE thInput = NULL;
        HANDLE thAudProc = NULL;
        HANDLE thAudEnc = NULL;
        auto pAVCodecReader = std::dynamic_pointer_cast<RGYInputAvcodec>(m_pFileReader);
        if (pAVCodecReader != nullptr) {
            thInput = pAVCodecReader->getThreadHandleInput();
        }
        auto pAVCodecWriter = std::dynamic_pointer_cast<RGYOutputAvcodec>(m_pFileWriter);
        if (pAVCodecWriter != nullptr) {
            thOutput = pAVCodecWriter->getThreadHandleOutput();
            thAudProc = pAVCodecWriter->getThreadHandleAudProcess();
            thAudEnc = pAVCodecWriter->getThreadHandleAudEncode();
        }
        m_pPerfMonitor->SetThreadHandles((HANDLE)(m_EncThread.GetHandleEncThread().native_handle()), thInput, thOutput, thAudProc, thAudEnc);
    }
#endif //#if ENABLE_AVSW_READER

    const int bufferSize = m_EncThread.m_nFrameBuffer;
    sInputBufSys *pArrayInputBuf = m_EncThread.m_InputBuf;
    sInputBufSys *pInputBuf;
    //入力ループ
    if (m_pFileReader->getInputCodec() == RGY_CODEC_UNKNOWN) {
        for (int i = 0; sts == MFX_ERR_NONE; i++) {
            pInputBuf = &pArrayInputBuf[i % bufferSize];

            //空いているフレームがセットされるのを待機
            PrintMes(RGY_LOG_TRACE, _T("Main Thread: Wait Start %d.\n"), i);
            while (WAIT_TIMEOUT == WaitForSingleObject(pInputBuf->heInputStart, 5000)) {
                //エンコードスレッドが異常終了していたら、それを検知してこちらも終了
                if (!CheckThreadAlive(m_EncThread.GetHandleEncThread())) {
                    PrintMes(RGY_LOG_ERROR, _T("error at encode thread.\n"));
                    sts = MFX_ERR_INVALID_HANDLE;
                }
            }

            //フレームを読み込み
            PrintMes(RGY_LOG_TRACE, _T("Main Thread: LoadNextFrame %d.\n"), i);
            if (sts == MFX_ERR_NONE) {
                sts = err_to_mfx(m_pFileReader->LoadNextFrame(pInputBuf->pFrameSurface));
            }
            if (m_pAbortByUser != nullptr && *m_pAbortByUser) {
                PrintMes(RGY_LOG_INFO, _T("                                                                              \r"));
                sts = MFX_ERR_ABORTED;
            } else if (sts == MFX_ERR_MORE_DATA) {
                m_EncThread.m_stsThread = sts;
            }

            //フレームの読み込み終了を通知
            SetEvent(pInputBuf->heInputDone);
            PrintMes(RGY_LOG_TRACE, _T("Main Thread: Set Done %d.\n"), i);
        }
    } else {
        while (sts == MFX_ERR_NONE) {
            std::this_thread::sleep_for(std::chrono::milliseconds(800));
            if (!CheckThreadAlive(m_EncThread.GetHandleEncThread())) {
                //読み込みが完了しているか、確認
                if (MFX_ERR_MORE_DATA == err_to_mfx(m_pFileReader->LoadNextFrame(nullptr))) {
                    //正常に読み込みを終了
                    sts = MFX_ERR_NONE;
                } else {
                    PrintMes(RGY_LOG_ERROR, _T("error at encode thread.\n"));
                    sts = MFX_ERR_UNKNOWN;
                }
                break;
            }
            //進捗表示 & 読み込み状況確認
            sts = err_to_mfx(m_pFileReader->LoadNextFrame(nullptr));
            if (m_pAbortByUser != nullptr && *m_pAbortByUser) {
                PrintMes(RGY_LOG_INFO, _T("                                                                              \r"));
                sts = MFX_ERR_ABORTED;
            } else if (sts == MFX_ERR_MORE_DATA) {
                m_EncThread.m_stsThread = sts;
            }
        }
    }
    m_EncThread.WaitToFinish(sts, m_pQSVLog);
    PrintMes(RGY_LOG_DEBUG, _T("Main Thread: Finished Main Loop...\n"));

    sts = (std::min)(sts, m_EncThread.m_stsThread);
    QSV_IGNORE_STS(sts, MFX_ERR_MORE_DATA);

    m_EncThread.Close();

    //ここでファイル出力の完了を確認してから、結果表示(m_pStatus->WriteResults)を行う
    m_pFileWriter->WaitFin();
    m_pStatus->WriteResults();

    PrintMes(RGY_LOG_DEBUG, _T("Main Thread: finished.\n"));
    return sts;
}
#endif

RGY_ERR CQSVPipeline::CreatePipeline() {
    m_outputTimestamp.clear();
    m_pipelineTasks.clear();

    if (m_pFileReader->getInputCodec() == RGY_CODEC_UNKNOWN) {
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskInput>(&m_mfxSession, m_pMFXAllocator.get(), 0, m_pFileReader.get(), m_mfxVer, m_pQSVLog));
    } else {
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskMFXDecode>(&m_mfxSession, 1, m_pmfxDEC.get(), m_mfxDecParams, m_pFileReader.get(), m_mfxVer, m_pQSVLog));
    }
    if (m_pFileWriterListAudio.size() > 0) {
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskAudio>(m_pFileReader.get(), m_AudioReaders, m_pFileWriterListAudio, 0, m_mfxVer, m_pQSVLog));
    }
    if (m_trimParam.list.size() > 0) {
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskTrim>(m_trimParam, 0, m_mfxVer, m_pQSVLog));
    }

    const int64_t outFrameDuration = std::max<int64_t>(1, rational_rescale(1, m_inputFps.inv(), m_outputTimebase)); //固定fpsを仮定した時の1フレームのduration (スケール: m_outputTimebase)
    const auto inputFrameInfo = m_pFileReader->GetInputFrameInfo();
    const auto inputFpsTimebase = rgy_rational<int>((int)inputFrameInfo.fpsD, (int)inputFrameInfo.fpsN);
    const auto srcTimebase = (m_pFileReader->getInputTimebase().n() > 0 && m_pFileReader->getInputTimebase().is_valid()) ? m_pFileReader->getInputTimebase() : inputFpsTimebase;
    m_pipelineTasks.push_back(std::make_unique<PipelineTaskCheckPTS>(srcTimebase, m_outputTimebase, m_outputTimestamp, outFrameDuration, m_nAVSyncMode, 0, m_mfxVer, m_pQSVLog));

    for (auto& filterBlock : m_vpFilters) {
        if (filterBlock.type == VppFilterType::FILTER_MFX) {
            auto err = err_to_rgy(m_mfxSession.JoinSession(filterBlock.vppmfx->GetSession()));
            if (err != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to join mfx vpp session: %s.\n"), get_err_mes(err));
                return err;
            }
            m_pipelineTasks.push_back(std::make_unique<PipelineTaskMFXVpp>(&m_mfxSession, 1, filterBlock.vppmfx->mfxvpp(), filterBlock.vppmfx->mfxparams(), filterBlock.vppmfx->mfxver(), m_pQSVLog));
        } else if (filterBlock.type == VppFilterType::FILTER_OPENCL) {
            m_pipelineTasks.push_back(std::make_unique<PipelineTaskOpenCL>(filterBlock.vppcl, m_cl, m_memType, m_pMFXAllocator.get(), &m_mfxSession, 1, m_pQSVLog));
        } else {
            PrintMes(RGY_LOG_ERROR, _T("Unknown filter type.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }

    if (m_pmfxENC) {
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskMFXEncode>(&m_mfxSession, 1, m_pmfxENC.get(), m_mfxVer, m_mfxEncParams, m_timecode.get(), m_outputTimebase, m_outputTimestamp, m_pQSVLog));
    }

    if (m_pipelineTasks.size() == 0) {
        PrintMes(RGY_LOG_DEBUG, _T("Failed to create pipeline: size = 0.\n"));
        return RGY_ERR_INVALID_OPERATION;
    }

    PrintMes(RGY_LOG_DEBUG, _T("Created pipeline.\n"));
    for (auto& p : m_pipelineTasks) {
        PrintMes(RGY_LOG_DEBUG, _T("  %s\n"), p->print().c_str());
    }
    PrintMes(RGY_LOG_DEBUG, _T("\n"));
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::RunEncode2() {
    PrintMes(RGY_LOG_DEBUG, _T("Encode Thread: RunEncode2...\n"));
    if (m_pipelineTasks.size() == 0) {
        PrintMes(RGY_LOG_DEBUG, _T("Failed to create pipeline: size = 0.\n"));
        return RGY_ERR_INVALID_OPERATION;
    }

#if defined(_WIN32) || defined(_WIN64)
    TCHAR handleEvent[256];
    _stprintf_s(handleEvent, QSVENCC_ABORT_EVENT, GetCurrentProcessId());
    auto heAbort = std::unique_ptr<std::remove_pointer<HANDLE>::type, handle_deleter>((HANDLE)CreateEvent(nullptr, TRUE, FALSE, handleEvent));
    auto checkAbort = [pabort = m_pAbortByUser, &heAbort]() { return ((pabort != nullptr && *pabort) || WaitForSingleObject(heAbort.get(), 0) == WAIT_OBJECT_0) ? true : false; };
#else
    auto checkAbort = [pabort = m_pAbortByUser]() { return  (pabort != nullptr && *pabort); }
#endif
    m_pStatus->SetStart();

    CProcSpeedControl speedCtrl(m_nProcSpeedLimit);

    auto requireSync = [this](const size_t itask) {
        if (itask + 1 >= m_pipelineTasks.size()) return true; // 次が最後のタスクの時

        for (size_t nexttask = itask+1; nexttask < m_pipelineTasks.size(); nexttask++) {
            if (!m_pipelineTasks[nexttask]->isPassThrough()) {
                return m_pipelineTasks[itask]->requireSync(m_pipelineTasks[nexttask]->taskType());
            }
        }
        return true;
    };

    RGY_ERR err = RGY_ERR_NONE;
    {
        auto checkContinue = [&checkAbort](RGY_ERR& err) {
            if (checkAbort()) { err = RGY_ERR_ABORTED; return false; }
            return err >= RGY_ERR_NONE || err == RGY_ERR_MORE_DATA || err == RGY_ERR_MORE_SURFACE;
        };
        while (checkContinue(err)) {
            speedCtrl.wait(m_pipelineTasks.front()->outputFrames());

            std::vector<std::unique_ptr<PipelineTaskOutput>> data;
            data.push_back(nullptr); // デコード実行用
            for (size_t itask = 0; checkContinue(err) && itask < m_pipelineTasks.size(); itask++) {
                err = RGY_ERR_NONE;
                auto& task = m_pipelineTasks[itask];
                for (auto& d : data) {
                    err = task->sendFrame(d);
                    if (!checkContinue(err)) break;
                }
                data.clear();
                if (err == RGY_ERR_NONE) {
                    data = task->getOutput(requireSync(itask));
                    if (data.size() == 0) break;
                }
            }
            for (auto& d : data) { // pipelineの最終的なデータを出力
                if ((err = d->write(m_pFileWriter.get(), m_pMFXAllocator.get())) != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("failed to write output: %s.\n"), get_err_mes(err));
                    break;
                }
            }
        }
    }
    // flush
    if (err == RGY_ERR_MORE_BITSTREAM) { // 読み込みの完了を示すフラグ
        err = RGY_ERR_NONE;
        for (auto& task : m_pipelineTasks) {
            task->setOutputMaxQueueSize(0); //flushのため
        }
        auto checkContinue = [&checkAbort](RGY_ERR& err) {
            if (checkAbort()) { err = RGY_ERR_ABORTED; return false; }
            return err >= RGY_ERR_NONE || err == RGY_ERR_MORE_SURFACE;
        };
        for (size_t flushedTask = 0; flushedTask < m_pipelineTasks.size(); ) { // taskを前方からひとつづつflushしていく
            err = RGY_ERR_NONE;
            std::vector<std::unique_ptr<PipelineTaskOutput>> data;
            data.push_back(nullptr); // flush用
            for (size_t itask = flushedTask; checkContinue(err) && itask < m_pipelineTasks.size(); itask++) {
                err = RGY_ERR_NONE;
                auto& task = m_pipelineTasks[itask];
                for (auto& d : data) {
                    err = task->sendFrame(d);
                    if (!checkContinue(err)) {
                        if (itask == flushedTask) flushedTask++;
                        break;
                    };
                }
                data.clear();

                data = task->getOutput(requireSync(itask));
                if (data.size() == 0) {
                    break;
                }
                RGY_IGNORE_STS(err, RGY_ERR_MORE_DATA); //VPPなどでsendFrameがRGY_ERR_MORE_DATAだったが、フレームが出てくる場合がある
            }
            for (auto& d : data) { // pipelineの最終的なデータを出力
                if ((err = d->write(m_pFileWriter.get(), m_pMFXAllocator.get())) != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("failed to write output: %s.\n"), get_err_mes(err));
                    break;
                }
            }
        }
    }

    PrintMes(RGY_LOG_DEBUG, _T("Clear pipeline tasks...\n"));
    m_pipelineTasks.clear();
    PrintMes(RGY_LOG_DEBUG, _T("Clear vpp filters...\n"));
    m_vpFilters.clear();
    PrintMes(RGY_LOG_DEBUG, _T("Waiting for writer to finish...\n"));
    m_pFileWriter->WaitFin();
    PrintMes(RGY_LOG_DEBUG, _T("Write results...\n"));
    m_pStatus->WriteResults();
    PrintMes(RGY_LOG_DEBUG, _T("RunEncode2: finished.\n"));
    return RGY_ERR_NONE;
}

#if 0
mfxStatus CQSVPipeline::RunEncode() {
    PrintMes(RGY_LOG_DEBUG, _T("Encode Thread: Starting Encode...\n"));

    mfxStatus sts = MFX_ERR_NONE;

    mfxFrameSurface1 *pSurfInputBuf = nullptr;
    mfxFrameSurface1 *pSurfEncIn = nullptr;
    mfxFrameSurface1 *pSurfVppIn = nullptr;
    mfxFrameSurface1 *pSurfCheckPts = nullptr; //checkptsから出てきて、他の要素に投入するフレーム / 投入後、ロックを解除する必要がある
    vector<mfxFrameSurface1 *>pSurfVppPreFilter(m_VppPrePlugins.size() + 1, nullptr);
    vector<mfxFrameSurface1 *>pSurfVppPostFilter(m_VppPostPlugins.size() + 1, nullptr);
    mfxFrameSurface1 *pNextFrame = nullptr;
    mfxSyncPoint lastSyncP = nullptr;
    bool bVppRequireMoreFrame = false;
    int nFramePutToEncoder = 0; //エンコーダに投入したフレーム数 (TimeStamp計算用)

    QSVTask *pCurrentTask = nullptr; //現在のタスクへのポインタ
    int nEncSurfIdx = -1; //使用するフレームのインデックス encoder input (vpp output)
    int nVppSurfIdx = -1; //使用するフレームのインデックス vpp input

    bool bVppMultipleOutput = false;  //bob化などの際にvppが余分にフレームを出力するフラグ
    bool bCheckPtsMultipleOutput = false; //dorcecfrなどにともなって、checkptsが余分にフレームを出力するフラグ

    int nLastCheckPtsFrame = -1;
    int nInputFrameCount = -1; //入力されたフレームの数 (最初のフレームが0になるよう、-1で初期化する)  Trimの反映に使用する

    struct frameData {
        mfxSyncPoint syncp;
        mfxFrameSurface1 *pSurface;
        int64_t timestamp;
    };
    std::deque<frameData> qDecodeFrames; //デコードされて出てきたsyncpとframe, timestamp

    const auto inputFrameInfo = m_pFileReader->GetInputFrameInfo();
    //QSVEncでは、常にTimestampは90kHzベースとする(m_outputTimebaseに設定済み)
    int64_t nOutFirstPts = -1;  //入力のptsに対する補正 (スケール: m_outputTimebase)
    int64_t nOutEstimatedPts = 0; //固定fpsを仮定した時のfps (スケール: m_outputTimebase)
    const auto hw_timebase = rgy_rational<int>(1, HW_TIMEBASE);
    const auto inputFpsTimebase = rgy_rational<int>((int)inputFrameInfo.fpsD, (int)inputFrameInfo.fpsN);
#if ENABLE_AVSW_READER
    const bool bAVutilDll = check_avcodec_dll();
    const AVStream *pStreamIn = nullptr;
    RGYInputAvcodec *pAVCodecReader = dynamic_cast<RGYInputAvcodec *>(m_pFileReader.get());
    if (pAVCodecReader != nullptr) {
        pStreamIn = pAVCodecReader->GetInputVideoStream();
    }
    FramePosList *framePosList = (pAVCodecReader != nullptr) ? pAVCodecReader->GetFramePosList() : nullptr;
    uint32_t framePosListIndex = (uint32_t)-1;
    const auto srcTimebase = (m_pFileReader->getInputTimebase().n() > 0 && m_pFileReader->getInputTimebase().is_valid()) ? m_pFileReader->getInputTimebase() : inputFpsTimebase;
    vector<AVPacket> packetList;
#else
    m_nAVSyncMode = RGY_AVSYNC_ASSUME_CFR;
#endif
    const auto nOutFrameDuration = std::max<int64_t>(1, rational_rescale(1, inputFpsTimebase, m_outputTimebase)); //固定fpsを仮定した時の1フレームのduration (スケール: m_outputTimebase)

    CProcSpeedControl speedCtrl(m_nProcSpeedLimit);

    m_pStatus->SetStart();

#if ENABLE_AVSW_READER
    //streamのindexから必要なwriteへのポインタを返すテーブルを作成
    std::map<int, shared_ptr<RGYOutputAvcodec>> pWriterForAudioStreams;
    for (auto pWriter : m_pFileWriterListAudio) {
        auto pAVCodecWriter = std::dynamic_pointer_cast<RGYOutputAvcodec>(pWriter);
        if (pAVCodecWriter) {
            auto trackIdList = pAVCodecWriter->GetStreamTrackIdList();
            for (auto trackID : trackIdList) {
                pWriterForAudioStreams[trackID] = pAVCodecWriter;
            }
        }
    }
    //streamのtrackIdからパケットを送信するvppフィルタへのポインタを返すテーブルを作成
    std::map<int, shared_ptr<QSVEncPlugin>> pFilterForStreams;
    for (const auto& pPlugins : m_VppPrePlugins) {
        const int trackId = pPlugins->getTargetTrack();
        if (trackId != 0) {
            pFilterForStreams[trackId] = pPlugins->getPluginHandle();
        }
    }
    for (const auto& pPlugins : m_VppPostPlugins) {
        const int trackId = pPlugins->getTargetTrack();
        if (trackId != 0) {
            pFilterForStreams[trackId] = pPlugins->getPluginHandle();
        }
    }
#endif

    sts = MFX_ERR_NONE;

    auto get_all_free_surface =[&](mfxFrameSurface1 *pSurfEncInput) {
        //パイプラインの後ろからたどっていく
        pSurfInputBuf = pSurfEncInput; //pSurfEncInにはパイプラインを後ろからたどった順にフレームポインタを更新していく
        pSurfVppPostFilter[m_VppPostPlugins.size()] = pSurfInputBuf; //pSurfVppPreFilterの最後はその直前のステップのフレームに出力される
        for (int i_filter = (int)m_VppPostPlugins.size()-1; i_filter >= 0; i_filter--) {
            int freeSurfIdx = GetFreeSurface(m_VppPostPlugins[i_filter]->m_pPluginSurfaces.get(), m_VppPostPlugins[i_filter]->m_PluginResponse.NumFrameActual);
            if (freeSurfIdx == MSDK_INVALID_SURF_IDX) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to get free surface for vpp post.\n"));
                return MFX_ERR_MEMORY_ALLOC;
            }
            pSurfVppPostFilter[i_filter] = &m_VppPostPlugins[i_filter]->m_pPluginSurfaces[freeSurfIdx];
            pSurfInputBuf = pSurfVppPostFilter[i_filter];
        }
        //vppが有効ならvpp用のフレームも用意する
        if (m_pmfxVPP) {
            //空いているフレームバッファを取得、空いていない場合は待機して、空くまで待ってから取得
            nVppSurfIdx = GetFreeSurface(m_pVppSurfaces.data(), m_VppResponse.NumFrameActual);
            if (nVppSurfIdx == MSDK_INVALID_SURF_IDX) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to get free surface for vpp.\n"));
                return MFX_ERR_MEMORY_ALLOC;
            }
            pSurfVppIn = &m_pVppSurfaces[nVppSurfIdx];
            pSurfInputBuf = pSurfVppIn;
        }
        pSurfVppPreFilter[m_VppPrePlugins.size()] = pSurfInputBuf; //pSurfVppPreFilterの最後はその直前のステップのフレームに出力される
        for (int i_filter = (int)m_VppPrePlugins.size()-1; i_filter >= 0; i_filter--) {
            int freeSurfIdx = GetFreeSurface(m_VppPrePlugins[i_filter]->m_pPluginSurfaces.get(), m_VppPrePlugins[i_filter]->m_PluginResponse.NumFrameActual);
            if (freeSurfIdx == MSDK_INVALID_SURF_IDX) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to get free surface for vpp pre.\n"));
                return MFX_ERR_MEMORY_ALLOC;
            }
            pSurfVppPreFilter[i_filter] = &m_VppPrePlugins[i_filter]->m_pPluginSurfaces[freeSurfIdx];
            pSurfInputBuf = pSurfVppPreFilter[i_filter];
        }
        //最終的にpSurfInputBufには一番最初のステップのフレームポインタが入る
        return MFX_ERR_NONE;
    };

    auto set_surface_to_input_buffer = [&]() {
        mfxStatus sts_set_buffer = MFX_ERR_NONE;
        for (int i = 0; i < m_EncThread.m_nFrameBuffer; i++) {
            get_all_free_surface(&m_pEncSurfaces[GetFreeSurface(m_pEncSurfaces.data(), m_EncResponse.NumFrameActual)]);

            //フレーム読み込みでない場合には、ここでロックする必要はない
            if (m_bExternalAlloc && m_pFileReader->getInputCodec() == RGY_CODEC_UNKNOWN) {
                if (MFX_ERR_NONE != (sts_set_buffer = m_pMFXAllocator->Lock(m_pMFXAllocator->pthis, pSurfInputBuf->Data.MemId, &(pSurfInputBuf->Data))))
                    break;
        }
        //空いているフレームを読み込み側に渡し、該当フレームの読み込み開始イベントをSetする(pInputBuf->heInputStart)
        SetNextSurface(pSurfInputBuf);
    }
        return sts_set_buffer;
    };

    //先読みバッファ用フレームを読み込み側に提供する
    set_surface_to_input_buffer();
    PrintMes(RGY_LOG_DEBUG, _T("Encode Thread: Set surface to input buffer...\n"));

    auto copy_crop_info = [](mfxFrameSurface1 *dst, const mfxFrameInfo *src) {
        if (NULL != dst) {
            dst->Info.CropX = src->CropX;
            dst->Info.CropY = src->CropY;
            dst->Info.CropW = src->CropW;
            dst->Info.CropH = src->CropH;
        }
    };

    auto extract_audio = [&](int inputFrames) {
        RGY_ERR ret = RGY_ERR_NONE;
#if ENABLE_AVSW_READER
        if (m_pFileWriterListAudio.size() + pFilterForStreams.size() > 0) {
#if ENABLE_SM_READER
            RGYInputSM *pReaderSM = dynamic_cast<RGYInputSM *>(m_pFileReader.get());
            const int droppedInAviutl = (pReaderSM != nullptr) ? pReaderSM->droppedFrames() : 0;
#else
            const int droppedInAviutl = 0;
#endif

            packetList = m_pFileReader->GetStreamDataPackets(inputFrames + droppedInAviutl);

            //音声ファイルリーダーからのトラックを結合する
            for (const auto& reader : m_AudioReaders) {
                vector_cat(packetList, reader->GetStreamDataPackets(inputFrames + droppedInAviutl));
            }
            //パケットを各Writerに分配する
            for (uint32_t i = 0; i < packetList.size(); i++) {
                const int nTrackId = (int)((uint32_t)packetList[i].flags >> 16);
                if (pWriterForAudioStreams.count(nTrackId)) {
                    auto pWriter = pWriterForAudioStreams[nTrackId];
                    if (pWriter == nullptr) {
                        PrintMes(RGY_LOG_ERROR, _T("Invalid writer found for track %d\n"), nTrackId);
                        return RGY_ERR_NULL_PTR;
                    }
                    if (RGY_ERR_NONE != (ret = pWriter->WriteNextPacket(&packetList[i]))) {
                        return ret;
                    }
                } else if (pFilterForStreams.count(nTrackId)) {
                    auto pFilter = pFilterForStreams[nTrackId];
                    if (pFilter == nullptr) {
                        PrintMes(RGY_LOG_ERROR, _T("Invalid filter found for track %d\n"), nTrackId);
                        return RGY_ERR_NULL_PTR;
                    }
                    auto sts = pFilter->SendData(PLUGIN_SEND_DATA_AVPACKET, &packetList[i]);
                    if (sts != MFX_ERR_NONE) {
                        return err_to_rgy(sts);
                    }
                } else {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to find writer for track %d\n"), nTrackId);
                    return RGY_ERR_NOT_FOUND;
                }
            }
        }
#endif //ENABLE_AVSW_READER
        return ret;
    };

    auto decode_one_frame = [&](bool getNextBitstream) {
        mfxStatus dec_sts = MFX_ERR_NONE;
        if (m_pmfxDEC) {
            if (getNextBitstream
                //m_DecInputBitstream.size() > 0のときにbitstreamを連結してしまうと
                //環境によっては正常にフレームが取り出せなくなることがある
                //これを避けるため、m_DecInputBitstream.size() == 0のときのみbitstreamを取得する
                //これにより GetNextFrame / SetNextFrame の回数が異常となり、
                //GetNextFrameのロックが抜けれらなくなる場合がある。
                //HWデコード時、本来GetNextFrameのロックは必要ないので、
                //これを無視する実装も併せて行った。
                && (m_DecInputBitstream.size() <= 1)) {
                //この関数がMFX_ERR_NONE以外を返せば、入力ビットストリームは終了
                auto ret = m_pFileReader->GetNextBitstream(&m_DecInputBitstream);
                if (ret == RGY_ERR_MORE_BITSTREAM) {
                    return err_to_mfx(ret); //入力ビットストリームは終了
                }
                RGY_ERR_MES(ret, _T("Error on getting video bitstream."));
            }

            getNextBitstream |= m_DecInputBitstream.size() > 0;

            //デコードも行う場合は、デコード用のフレームをpSurfVppInかpSurfEncInから受け取る
            mfxFrameSurface1 *pSurfDecWork = pNextFrame;
            mfxFrameSurface1 *pSurfDecOut = NULL;
            mfxBitstream *pInputBitstream = (getNextBitstream) ? &m_DecInputBitstream.bitstream() : nullptr;

            if (!m_mfxDecParams.mfx.FrameInfo.FourCC) {
                //デコード前には、デコード用のパラメータでFrameInfoを更新
                copy_crop_info(pSurfDecWork, &m_mfxDecParams.mfx.FrameInfo);
            }
            if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_9)
                && (m_mfxDecParams.mfx.CodecId == MFX_CODEC_VP8 || m_mfxDecParams.mfx.CodecId == MFX_CODEC_VP9)) { // VP8/VP9ではこの処理が必要
                if (pSurfDecWork->Info.BitDepthLuma == 0 || pSurfDecWork->Info.BitDepthChroma == 0) {
                    pSurfDecWork->Info.BitDepthLuma = m_mfxDecParams.mfx.FrameInfo.BitDepthLuma;
                    pSurfDecWork->Info.BitDepthChroma = m_mfxDecParams.mfx.FrameInfo.BitDepthChroma;
                }
            }
            if (pInputBitstream != nullptr) {
                if (pInputBitstream->TimeStamp == (mfxU64)AV_NOPTS_VALUE) {
                    pInputBitstream->TimeStamp = (mfxU64)MFX_TIMESTAMP_UNKNOWN;
                }
                pInputBitstream->DecodeTimeStamp = MFX_TIMESTAMP_UNKNOWN;
            }
            pSurfDecWork->Data.TimeStamp = (mfxU64)MFX_TIMESTAMP_UNKNOWN;
            pSurfDecWork->Data.DataFlag |= MFX_FRAMEDATA_ORIGINAL_TIMESTAMP;

            for (int i = 0; ; i++) {
                const auto inputDataLen = (pInputBitstream) ? pInputBitstream->DataLength : 0;
                mfxSyncPoint DecSyncPoint = NULL;
                dec_sts = m_pmfxDEC->DecodeFrameAsync(pInputBitstream, pSurfDecWork, &pSurfDecOut, &DecSyncPoint);
                lastSyncP = DecSyncPoint;

                if (MFX_ERR_NONE < dec_sts && !DecSyncPoint) {
                    if (MFX_WRN_DEVICE_BUSY == dec_sts)
                        sleep_hybrid(i);
                    if (i > 1024 * 1024 * 30) {
                        PrintMes(RGY_LOG_ERROR, _T("device kept on busy for 30s, unknown error occurred.\n"));
                        return MFX_ERR_UNKNOWN;
                    }
                } else if (MFX_ERR_NONE < dec_sts && DecSyncPoint) {
                    dec_sts = MFX_ERR_NONE; //出力があれば、警告は無視する
                    break;
                } else if (dec_sts < MFX_ERR_NONE && (dec_sts != MFX_ERR_MORE_DATA && dec_sts != MFX_ERR_MORE_SURFACE)) {
                    PrintMes(RGY_LOG_ERROR, _T("DecodeFrameAsync error: %s.\n"), get_err_mes(dec_sts));
                    break;
                } else {
                    //pInputBitstreamの長さがDecodeFrameAsyncを経ても全く変わっていない場合は、そのデータは捨てる
                    //これを行わないとデコードが止まってしまう
                    if (dec_sts == MFX_ERR_MORE_DATA && pInputBitstream && pInputBitstream->DataLength == inputDataLen) {
                        PrintMes((inputDataLen >= 10) ? RGY_LOG_WARN : RGY_LOG_DEBUG,
                            _T("DecodeFrameAsync: removing %d bytes from input bitstream not read by decoder.\n"), inputDataLen);
                        pInputBitstream->DataLength = 0;
                        pInputBitstream->DataOffset = 0;
                    }
                    break;
                }
            }

            //次のステップのフレームをデコードの出力に設定
            pNextFrame = pSurfDecOut;
            nInputFrameCount += (pSurfDecOut != nullptr && lastSyncP != nullptr);
        } else {
            //デコードがオンでなくても、フレームは入力してるはずなので加算する
            //Trimの反映に使用する
            nInputFrameCount++;
        }
        return dec_sts;
    };

    int64_t prevPts = 0; //(m_outputTimebase基準)
    auto check_pts = [&]() {
        int64_t outDuration = nOutFrameDuration; //入力fpsに従ったduration (m_outputTimebase基準)
        int64_t outPts = nOutEstimatedPts; //(m_outputTimebase基準)
#if ENABLE_AVSW_READER
        if (m_nAVSyncMode & (RGY_AVSYNC_VFR | RGY_AVSYNC_FORCE_CFR)) {
            if (pNextFrame) {
                if (framePosList) {
                    auto pos = framePosList->copy(nInputFrameCount, &framePosListIndex);
                    if (pos.poc == FRAMEPOS_POC_INVALID) {
                        PrintMes(RGY_LOG_ERROR, _T("Encode Thread: failed to get timestamp.\n"));
                        return MFX_ERR_UNKNOWN;
                    }
                    outPts = rational_rescale(pos.pts, srcTimebase, m_outputTimebase);
                    if ((m_nAVSyncMode & RGY_AVSYNC_VFR) && pos.duration > 0) {
                        outDuration = rational_rescale(pos.duration, srcTimebase, m_outputTimebase);
                    }
                    if (nOutFirstPts >= 0 && !frame_inside_range(nInputFrameCount - 1, m_trimParam.list).first) {
                        nOutFirstPts += (outPts - prevPts);
                    }
                } else {
                    outPts = rational_rescale(pNextFrame->Data.TimeStamp, srcTimebase, m_outputTimebase);
                    outDuration = rational_rescale(((RGYFrame *)pNextFrame)->duration(), srcTimebase, m_outputTimebase);
                    ((RGYFrame *)pNextFrame)->setDuration(0);
                }
            }
        }
        nLastCheckPtsFrame = nInputFrameCount;
        if (nOutFirstPts == -1) {
            nOutFirstPts = outPts; //最初のpts
        }
        //最初のptsを0に修正
        outPts -= nOutFirstPts;
        if (outPts < 0) {
            outPts = nOutEstimatedPts;
        }

        if (m_nAVSyncMode & RGY_AVSYNC_FORCE_CFR) {
            if (!bCheckPtsMultipleOutput) {
                //ひとまずデコード結果をキューに格納
                if (pNextFrame) {
                    //ここでロックしないとキューにためているフレームが勝手に使われてしまう
                    pNextFrame->Data.Locked++;
                    qDecodeFrames.push_back({ lastSyncP, pNextFrame, outPts });
                }
                //queueが空になったら終了
                if (qDecodeFrames.size() == 0) {
                    return MFX_ERR_MORE_DATA;
                }
            }
            auto queueFirstFrame = qDecodeFrames.front();
            auto queueFirstPts = queueFirstFrame.timestamp;
            if (queueFirstPts == AV_NOPTS_VALUE) {
                PrintMes(RGY_LOG_ERROR, _T("Invalid timestamp provided from input.\n"));
                return MFX_ERR_UNSUPPORTED;
            }

            auto ptsDiff = queueFirstPts - nOutEstimatedPts;
            if (std::abs(ptsDiff) >= CHECK_PTS_MAX_INSERT_FRAMES * nOutFrameDuration) {
                //timestampに一定以上の差があればそれを無視する
                nOutFirstPts += (queueFirstPts - nOutEstimatedPts); //今後の位置合わせのための補正
                outPts = nOutEstimatedPts;
                //PrintMes(RGY_LOG_WARN, _T("Big Gap was found between 2 frames (%d - %d), avsync might be corrupted.\n"), nInputFrameCount, nInputFrameCount+1);
            } else if (ptsDiff >= std::max<int64_t>(1, nOutFrameDuration * 3 / 4)) {
                //水増しが必要 -> 何も(pop)しない
                bCheckPtsMultipleOutput = true;
                queueFirstFrame.pSurface->Data.Locked++;
                framePosListIndex--;
                rearrange_trim_list(nInputFrameCount, -1, m_trimParam.list);
            } else {
                bCheckPtsMultipleOutput = false;
                qDecodeFrames.pop_front();
                if (ptsDiff <= std::min<int64_t>(-1, -1 * nOutFrameDuration * 3 / 4)) {
                    //間引きが必要 -> フレームを後段に渡さず破棄
                    queueFirstFrame.pSurface->Data.Locked--;
                    pSurfCheckPts = nullptr;
                    rearrange_trim_list(nInputFrameCount, 1, m_trimParam.list);
                    return MFX_ERR_MORE_SURFACE;
                }
            }
            lastSyncP = queueFirstFrame.syncp;
            pNextFrame    = queueFirstFrame.pSurface;
            pSurfCheckPts = queueFirstFrame.pSurface;
            outPts = nOutEstimatedPts;
        }
#endif //#if ENABLE_AVSW_READER
        nOutEstimatedPts += outDuration;
        prevPts = outPts + outDuration;
        pNextFrame->Data.TimeStamp = rational_rescale(outPts, m_outputTimebase, hw_timebase);
        pNextFrame->Data.DataFlag &= (~MFX_FRAMEDATA_ORIGINAL_TIMESTAMP);
        m_outputTimestamp.add(pNextFrame->Data.TimeStamp, rational_rescale(outDuration, m_outputTimebase, hw_timebase));
        return MFX_ERR_NONE;
    };

    auto filter_one_frame = [&](const unique_ptr<CVPPPlugin>& filter, mfxFrameSurface1 **ppSurfIn, mfxFrameSurface1 **ppSurfOut) {
        mfxStatus filter_sts = MFX_ERR_NONE;
        mfxSyncPoint filterSyncPoint = NULL;

        for (int i = 0; ; i++) {
            mfxHDL *h1 = (mfxHDL *)ppSurfIn;
            mfxHDL *h2 = (mfxHDL *)ppSurfOut;

            filter_sts = MFXVideoUSER_ProcessFrameAsync(filter->getSession(), h1, 1, h2, 1, &filterSyncPoint);

            if (MFX_WRN_DEVICE_BUSY == filter_sts) {
                sleep_hybrid(i);
                if (i > 1024 * 1024 * 30) {
                    PrintMes(RGY_LOG_ERROR, _T("device kept on busy for 30s, unknown error occurred.\n"));
                    return MFX_ERR_UNKNOWN;
                }
            } else {
                break;
            }
        }
        // save the id of preceding vpp task which will produce input data for the encode task
        if (filterSyncPoint) {
            lastSyncP = filterSyncPoint;
            //pCurrentTask->vppSyncPoint.push_back(filterSyncPoint);
            filterSyncPoint = NULL;
        }
        return filter_sts;
    };

    auto vpp_one_frame =[&](mfxFrameSurface1* pSurfVppIn, mfxFrameSurface1* pSurfVppOut) {
        mfxStatus vpp_sts = MFX_ERR_NONE;
        if (m_pmfxVPP) {
            mfxSyncPoint VppSyncPoint = nullptr;
            bVppMultipleOutput = false;
            bVppRequireMoreFrame = false;

            //vpp前に、vpp用のパラメータでFrameInfoを更新
            copy_crop_info(pSurfVppIn, &m_mfxVppParams.mfx.FrameInfo);

            for (int i = 0; ; i++) {
                //bob化の際、pSurfVppInに連続で同じフレーム(同じtimestamp)を投入すると、
                //最初のフレームには設定したtimestamp、次のフレームにはMFX_TIMESTAMP_UNKNOWNが設定されて出てくる
                //特別pSurfVppOut側のTimestampを設定する必要はなさそう
                vpp_sts = m_pmfxVPP->RunFrameVPPAsync(pSurfVppIn, pSurfVppOut, NULL, &VppSyncPoint);
                lastSyncP = VppSyncPoint;

                if (MFX_ERR_NONE < vpp_sts && !VppSyncPoint) {
                    if (MFX_WRN_DEVICE_BUSY == vpp_sts)
                        sleep_hybrid(i);
                    if (i > 1024 * 1024 * 30) {
                        PrintMes(RGY_LOG_ERROR, _T("device kept on busy for 30s, unknown error occurred.\n"));
                        return MFX_ERR_UNKNOWN;
                    }
                } else if (MFX_ERR_NONE < vpp_sts && VppSyncPoint) {
                    vpp_sts = MFX_ERR_NONE;
                    break;
                } else {
                    break;
                }
            }

            if (MFX_ERR_MORE_DATA == vpp_sts) {
                bVppRequireMoreFrame = true;
            } else if (MFX_ERR_MORE_SURFACE == vpp_sts) {
                bVppMultipleOutput = true;
                vpp_sts = MFX_ERR_NONE;
            }

            if (VppSyncPoint) {
                pCurrentTask->vppSyncPoint.push_back(VppSyncPoint);
                VppSyncPoint = NULL;
                pNextFrame = pSurfVppOut;
            }
        }
        return vpp_sts;
    };

    auto encode_one_frame =[&](mfxFrameSurface1* pSurfEncIn) {
        if (m_pmfxENC == nullptr) {
            //エンコードが有効でない場合、このフレームデータを出力する
            //パイプラインの最後のSyncPointをセットする
            pCurrentTask->encSyncPoint = lastSyncP;
            //フレームデータが出力されるまで空きフレームとして使われないようLockを加算しておく
            //TaskのWriteBitstreamで減算され、解放される
            pSurfEncIn->Data.Locked++;
            //フレームのポインタを出力用にセット
            pCurrentTask->mfxSurf = pSurfEncIn;
            return MFX_ERR_NONE;
        }

        mfxStatus enc_sts = MFX_ERR_NONE;

        //以下の処理は
        if (pSurfEncIn) {
            nFramePutToEncoder++;
            //TimeStampをMFX_TIMESTAMP_UNKNOWNにしておくと、きちんと設定される
            pCurrentTask->mfxBS.TimeStamp = (uint64_t)MFX_TIMESTAMP_UNKNOWN;
            pCurrentTask->mfxBS.DecodeTimeStamp = (uint64_t)MFX_TIMESTAMP_UNKNOWN;
            //bob化の際に増えたフレームのTimeStampには、MFX_TIMESTAMP_UNKNOWNが設定されているのでこれを補間して修正する
            pSurfEncIn->Data.TimeStamp = (uint64_t)m_outputTimestamp.check(pSurfEncIn->Data.TimeStamp);
            if (m_timecode) {
                m_timecode->write(pSurfEncIn->Data.TimeStamp, m_outputTimebase);
            }
        }

        bool bDeviceBusy = false;
        for (int i = 0; ; i++) {
            enc_sts = m_pmfxENC->EncodeFrameAsync(nullptr, pSurfEncIn, &pCurrentTask->mfxBS, &pCurrentTask->encSyncPoint);
            bDeviceBusy = false;

            if (MFX_ERR_NONE < enc_sts && !pCurrentTask->encSyncPoint) {
                bDeviceBusy = true;
                if (MFX_WRN_DEVICE_BUSY == enc_sts)
                sleep_hybrid(i);
                if (i > 65536 * 1024 * 30) {
                    PrintMes(RGY_LOG_ERROR, _T("device kept on busy for 30s, unknown error occurred.\n"));
                    return MFX_ERR_UNKNOWN;
                }
            } else if (MFX_ERR_NONE < enc_sts && pCurrentTask->encSyncPoint) {
                enc_sts = MFX_ERR_NONE;
                break;
            } else if (MFX_ERR_NOT_ENOUGH_BUFFER == enc_sts) {
                enc_sts = AllocateSufficientBuffer(&pCurrentTask->mfxBS);
                if (enc_sts < MFX_ERR_NONE) return enc_sts;
            } else if (enc_sts < MFX_ERR_NONE && (enc_sts != MFX_ERR_MORE_DATA && enc_sts != MFX_ERR_MORE_SURFACE)) {
                PrintMes(RGY_LOG_ERROR, _T("EncodeFrameAsync error: %s.\n"), get_err_mes(enc_sts));
                break;
            } else {
                QSV_IGNORE_STS(enc_sts, MFX_ERR_MORE_BITSTREAM);
                break;
            }
        }
        return enc_sts;
    };

    //メインループ
    while (MFX_ERR_NONE <= sts || MFX_ERR_MORE_DATA == sts || MFX_ERR_MORE_SURFACE == sts) {
        if (pSurfCheckPts) {
            //pSurfCheckPtsはcheckptsから出てきて、他の要素に投入するフレーム
            //投入後、ロックを解除する必要がある
            pSurfCheckPts->Data.Locked--;
            pSurfCheckPts = nullptr;
        }
        speedCtrl.wait(m_pStatus->m_sData.frameIn);
#if defined(_WIN32) || defined(_WIN64)
        //中断オブジェクトのチェック
        if (WaitForSingleObject(m_heAbort.get(), 0) == WAIT_OBJECT_0) {
            m_EncThread.m_bthForceAbort = true;
        }
#endif

        //空いているフレームバッファを取得、空いていない場合は待機して、出力ストリームの書き出しを待ってから取得
        //encTaskによるsyncPointを完了してsyncPointがnullになっているものを探す
        if (MFX_ERR_NONE != (sts = GetFreeTask(&pCurrentTask)))
            break;

        //空いているフレームバッファを取得、空いていない場合は待機して、空くまで待ってから取得
        //具体的には、Lockされていないフレームを取得する
        nEncSurfIdx = GetFreeSurface(m_pEncSurfaces.data(), m_EncResponse.NumFrameActual);
        if (nEncSurfIdx == MSDK_INVALID_SURF_IDX) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to get free surface for enc.\n"));
            return MFX_ERR_MEMORY_ALLOC;
        }

        // point pSurf to encoder surface
        pSurfEncIn = &m_pEncSurfaces[nEncSurfIdx];

        if (!bVppMultipleOutput) {
            if (MFX_ERR_NONE != (sts = get_all_free_surface(pSurfEncIn))) {
                break;
            }
            if (!bCheckPtsMultipleOutput) {
                //if (m_VppPrePlugins.size()) {
                //    pSurfInputBuf = pSurfVppPreFilter[0];
                //    //ppNextFrame = &;
                //} else if (m_pmfxVPP) {
                //    pSurfInputBuf = &m_pVppSurfaces[nVppSurfIdx];
                //    //ppNextFrame = &pSurfVppIn;
                //} else if (m_VppPostPlugins.size()) {
                //    pSurfInputBuf = &pSurfVppPostFilter[0];
                //    //ppNextFrame = &;
                //} else {
                //    pSurfInputBuf = pSurfEncIn;
                //    //ppNextFrame = &pSurfEncIn;
                //}

                if (m_pFileReader->getInputCodec() == RGY_CODEC_UNKNOWN) {
                    //読み込み側の該当フレームの読み込み終了を待機(pInputBuf->heInputDone)して、読み込んだフレームを取得
                    //この関数がRGY_ERR_NONE以外を返すことでRunEncodeは終了処理に入る
                    sts = GetNextFrame(&pNextFrame);
                    if (sts != MFX_ERR_NONE) {
                        break;
                    }
                    //フレーム読み込みの場合には、必要ならここでロックする
                    if (m_bExternalAlloc) {
                        if (MFX_ERR_NONE != (sts = m_pMFXAllocator->Unlock(m_pMFXAllocator->pthis, (pNextFrame)->Data.MemId, &((pNextFrame)->Data))))
                            break;

                        if (MFX_ERR_NONE != (sts = m_pMFXAllocator->Lock(m_pMFXAllocator->pthis, pSurfInputBuf->Data.MemId, &(pSurfInputBuf->Data))))
                            break;
                    }

                    //空いているフレームを読み込み側に渡す
                    SetNextSurface(pSurfInputBuf);
                } else {
                    //フレーム読み込みでない場合には、フレームバッファをm_pFileReaderを通さずに直接渡す
                    pNextFrame = pSurfInputBuf;
                    if (m_EncThread.m_bthForceAbort) {
                        sts = m_EncThread.m_stsThread;
                        break;
                    }
                }

                auto ret = extract_audio(nInputFrameCount);
                if (ret != RGY_ERR_NONE) {
                    sts = err_to_mfx(ret);
                    break;
                }

                //この関数がMFX_ERR_MORE_BITSTREAMを返せば、入力は終了
                sts = decode_one_frame(true);
                if (sts == MFX_ERR_MORE_DATA || sts == MFX_ERR_MORE_SURFACE)
                    continue;
                if (sts != MFX_ERR_NONE)
                    break;
            }

            if (!frame_inside_range(nInputFrameCount, m_trimParam.list).first)
                continue;

            sts = check_pts();
            if (sts == MFX_ERR_MORE_SURFACE)
                continue;
            if (sts != MFX_ERR_NONE)
                break;

            for (int i_filter = 0; i_filter < (int)m_VppPrePlugins.size(); i_filter++) {
                mfxFrameSurface1 *pSurfFilterOut = pSurfVppPreFilter[i_filter + 1];
                if (MFX_ERR_NONE != (sts = filter_one_frame(m_VppPrePlugins[i_filter], &pNextFrame, &pSurfFilterOut)))
                    break;
                pNextFrame = pSurfFilterOut;
            }
            if (sts != MFX_ERR_NONE)
                break;

            pSurfVppIn = pNextFrame;
        }

        sts = vpp_one_frame(pSurfVppIn, (m_VppPostPlugins.size()) ? pSurfVppPostFilter[0] : pSurfEncIn);
        if (bVppRequireMoreFrame)
            continue;
        if (sts != MFX_ERR_NONE)
            break;

        for (int i_filter = 0; i_filter < (int)m_VppPostPlugins.size(); i_filter++) {
            mfxFrameSurface1 *pSurfFilterOut = pSurfVppPostFilter[i_filter + 1];
            if (MFX_ERR_NONE != (sts = filter_one_frame(m_VppPostPlugins[i_filter], &pNextFrame, &pSurfFilterOut)))
                break;
            pNextFrame = pSurfFilterOut;
        }
        if (sts != MFX_ERR_NONE)
            break;

        sts = encode_one_frame(pNextFrame);
    }

    //MFX_ERR_MORE_DATA/MFX_ERR_MORE_BITSTREAMは入力が終了したことを示す
    QSV_IGNORE_STS(sts, (m_pFileReader->getInputCodec() != RGY_CODEC_UNKNOWN) ? MFX_ERR_MORE_BITSTREAM : MFX_ERR_MORE_DATA);
    //エラーチェック
    m_EncThread.m_stsThread = sts;
    QSV_ERR_MES(sts, _T("Error in encoding pipeline."));
    PrintMes(RGY_LOG_DEBUG, _T("Encode Thread: finished main loop.\n"));

    if (m_pmfxDEC) {
        auto ret = extract_audio(nInputFrameCount);
        RGY_ERR_MES(ret, _T("Error on extracting audio."));

        pNextFrame = NULL;

        while (MFX_ERR_NONE <= sts || sts == MFX_ERR_MORE_SURFACE) {
            if (pSurfCheckPts) {
                //pSurfCheckPtsはcheckptsから出てきて、他の要素に投入するフレーム
                //投入後、ロックを解除する必要がある
                pSurfCheckPts->Data.Locked--;
                pSurfCheckPts = nullptr;
            }

            //空いているフレームバッファを取得、空いていない場合は待機して、出力ストリームの書き出しを待ってから取得
            if (MFX_ERR_NONE != (sts = GetFreeTask(&pCurrentTask)))
                break;

            //空いているフレームバッファを取得、空いていない場合は待機して、空くまで待ってから取得
            nEncSurfIdx = GetFreeSurface(m_pEncSurfaces.data(), m_EncResponse.NumFrameActual);
            if (nEncSurfIdx == MSDK_INVALID_SURF_IDX) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to get free surface for enc.\n"));
                return MFX_ERR_MEMORY_ALLOC;
            }

            pSurfEncIn = &m_pEncSurfaces[nEncSurfIdx];

            if (!bVppMultipleOutput) {
                get_all_free_surface(pSurfEncIn);
                pNextFrame = pSurfInputBuf;

                if (!bCheckPtsMultipleOutput) {
                    sts = decode_one_frame(false);
                    if (sts == MFX_ERR_MORE_SURFACE)
                        continue;
                    if (sts != MFX_ERR_NONE)
                        break;
                }

                sts = check_pts();
                if (sts == MFX_ERR_MORE_SURFACE)
                    continue;
                if (sts != MFX_ERR_NONE)
                    break;

                for (int i_filter = 0; i_filter < (int)m_VppPrePlugins.size(); i_filter++) {
                    mfxFrameSurface1 *pSurfFilterOut = pSurfVppPreFilter[i_filter + 1];
                    if (MFX_ERR_NONE != (sts = filter_one_frame(m_VppPrePlugins[i_filter], &pNextFrame, &pSurfFilterOut)))
                        break;
                    pNextFrame = pSurfFilterOut;
                }
                if (sts != MFX_ERR_NONE)
                    break;

                pSurfVppIn = pNextFrame;
            }

            if (!frame_inside_range(nInputFrameCount, m_trimParam.list).first)
                continue;

            sts = vpp_one_frame(pSurfVppIn, (m_VppPostPlugins.size()) ? pSurfVppPostFilter[0] : pSurfEncIn);
            if (bVppRequireMoreFrame)
                continue;
            if (sts != MFX_ERR_NONE)
                break;

            for (int i_filter = 0; i_filter < (int)m_VppPostPlugins.size(); i_filter++) {
                mfxFrameSurface1 *pSurfFilterOut = pSurfVppPostFilter[i_filter + 1];
                if (MFX_ERR_NONE != (sts = filter_one_frame(m_VppPostPlugins[i_filter], &pNextFrame, &pSurfFilterOut)))
                    break;
                pNextFrame = pSurfFilterOut;
            }
            if (sts != MFX_ERR_NONE)
                break;

            sts = encode_one_frame(pNextFrame);
        }

        //MFX_ERR_MORE_DATAはデコーダにもうflushするべきフレームがないことを示す
        QSV_IGNORE_STS(sts, MFX_ERR_MORE_DATA);
        //エラーチェック
        m_EncThread.m_stsThread = sts;
        QSV_ERR_MES(sts, _T("Error in getting buffered frames from decoder."));
        PrintMes(RGY_LOG_DEBUG, _T("Encode Thread: finished getting buffered frames from decoder.\n"));
    }

#if ENABLE_AVSW_READER
    if (m_pmfxDEC && (m_nAVSyncMode & RGY_AVSYNC_FORCE_CFR)) {

        pNextFrame = NULL;

        while (MFX_ERR_NONE <= sts || sts == MFX_ERR_MORE_SURFACE) {
            if (pSurfCheckPts) {
                //pSurfCheckPtsはcheckptsから出てきて、他の要素に投入するフレーム
                //投入後、ロックを解除する必要がある
                pSurfCheckPts->Data.Locked--;
                pSurfCheckPts = nullptr;
            }

            //空いているフレームバッファを取得、空いていない場合は待機して、出力ストリームの書き出しを待ってから取得
            if (MFX_ERR_NONE != (sts = GetFreeTask(&pCurrentTask)))
                break;

            //空いているフレームバッファを取得、空いていない場合は待機して、空くまで待ってから取得
            nEncSurfIdx = GetFreeSurface(m_pEncSurfaces.data(), m_EncResponse.NumFrameActual);
            if (nEncSurfIdx == MSDK_INVALID_SURF_IDX) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to get free surface for enc.\n"));
                return MFX_ERR_MEMORY_ALLOC;
            }

            pSurfEncIn = &m_pEncSurfaces[nEncSurfIdx];

            if (!bVppMultipleOutput) {
                get_all_free_surface(pSurfEncIn);
                pNextFrame = nullptr;
                lastSyncP = nullptr;

                sts = check_pts();
                if (sts == MFX_ERR_MORE_SURFACE)
                    continue;
                if (sts != MFX_ERR_NONE)
                    break;

                for (int i_filter = 0; i_filter < (int)m_VppPrePlugins.size(); i_filter++) {
                    mfxFrameSurface1 *pSurfFilterOut = pSurfVppPreFilter[i_filter + 1];
                    if (MFX_ERR_NONE != (sts = filter_one_frame(m_VppPrePlugins[i_filter], &pNextFrame, &pSurfFilterOut)))
                        break;
                    pNextFrame = pSurfFilterOut;
                }
                if (sts != MFX_ERR_NONE)
                    break;

                pSurfVppIn = pNextFrame;
            }

            sts = vpp_one_frame(pSurfVppIn, (m_VppPostPlugins.size()) ? pSurfVppPostFilter[0] : pSurfEncIn);
            if (bVppRequireMoreFrame)
                continue;
            if (sts != MFX_ERR_NONE)
                break;

            for (int i_filter = 0; i_filter < (int)m_VppPostPlugins.size(); i_filter++) {
                mfxFrameSurface1 *pSurfFilterOut = pSurfVppPostFilter[i_filter + 1];
                if (MFX_ERR_NONE != (sts = filter_one_frame(m_VppPostPlugins[i_filter], &pNextFrame, &pSurfFilterOut)))
                    break;
                pNextFrame = pSurfFilterOut;
            }
            if (sts != MFX_ERR_NONE)
                break;

            sts = encode_one_frame(pNextFrame);
        }

        //MFX_ERR_MORE_DATAはcheck_ptsにもうflushするべきフレームがないことを示す
        QSV_IGNORE_STS(sts, MFX_ERR_MORE_DATA);
        // exit in case of other errors
        m_EncThread.m_stsThread = sts;
        QSV_ERR_MES(sts, _T("Error in getting buffered frames from avsync buffer."));
        PrintMes(RGY_LOG_DEBUG, _T("Encode Thread: finished getting buffered frames from avsync buffer.\n"));
    }

    for (const auto& writer : m_pFileWriterListAudio) {
        auto pAVCodecWriter = std::dynamic_pointer_cast<RGYOutputAvcodec>(writer);
        if (pAVCodecWriter != nullptr) {
            //エンコーダなどにキャッシュされたパケットを書き出す
            pAVCodecWriter->WriteNextPacket(nullptr);
        }
    }
#endif //ENABLE_AVSW_READER

    if (m_pmfxVPP) {
        //vppのフレームをflush
        while (MFX_ERR_NONE <= sts || MFX_ERR_MORE_DATA == sts || MFX_ERR_MORE_SURFACE == sts) {
            // MFX_ERR_MORE_SURFACE can be returned only by RunFrameVPPAsync
            // MFX_ERR_MORE_DATA is accepted only from EncodeFrameAsync
            if (pSurfCheckPts) {
                //pSurfCheckPtsはcheckptsから出てきて、他の要素に投入するフレーム
                //投入後、ロックを解除する必要がある
                pSurfCheckPts->Data.Locked--;
                pSurfCheckPts = nullptr;
            }

            pNextFrame = nullptr;

            nEncSurfIdx = GetFreeSurface(m_pEncSurfaces.data(), m_EncResponse.NumFrameActual);
            if (nEncSurfIdx == MSDK_INVALID_SURF_IDX) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to get free surface for enc.\n"));
                return MFX_ERR_MEMORY_ALLOC;
            }

            pSurfEncIn = &m_pEncSurfaces[nEncSurfIdx];

            if (MFX_ERR_NONE != (sts = GetFreeTask(&pCurrentTask)))
                break;

            get_all_free_surface(pSurfEncIn);

            //for (int i_filter = 0; i_filter < (int)m_VppPrePlugins.size(); i_filter++) {
            //    bVppAllFiltersFlushed &= m_VppPrePlugins[i_filter]->m_bPluginFlushed;
            //    if (!m_VppPrePlugins[i_filter]->m_bPluginFlushed) {
            //        mfxFrameSurface1 *pSurfFilterOut = pSurfVppPreFilter[i_filter + 1];
            //        sts = filter_one_frame(m_VppPrePlugins[i_filter], &pNextFrame, &pSurfFilterOut);
            //        if (sts == MFX_ERR_MORE_DATA) {
            //            m_VppPrePlugins[i_filter]->m_bPluginFlushed = true;
            //            sts = MFX_ERR_NONE;
            //        }
            //        MSDK_BREAK_ON_ERROR(sts);
            //        pNextFrame = pSurfFilterOut;
            //    }
            //}
            //MSDK_BREAK_ON_ERROR(sts);

            sts = vpp_one_frame(pNextFrame, (m_VppPostPlugins.size()) ? pSurfVppPostFilter[0] : pSurfEncIn);
            if (bVppRequireMoreFrame) {
                break;
            }
            if (sts != MFX_ERR_NONE)
                break;

            for (int i_filter = 0; i_filter < (int)m_VppPostPlugins.size(); i_filter++) {
                mfxFrameSurface1 *pSurfFilterOut = pSurfVppPostFilter[i_filter + 1];
                if (MFX_ERR_NONE != (sts = filter_one_frame(m_VppPostPlugins[i_filter], &pNextFrame, &pSurfFilterOut)))
                    break;
                pNextFrame = pSurfFilterOut;
            }
            if (sts != MFX_ERR_NONE)
                break;

            sts = encode_one_frame(pNextFrame);
        }

        //MFX_ERR_MORE_DATAはvppにもうflushするべきフレームがないことを示す
        QSV_IGNORE_STS(sts, MFX_ERR_MORE_DATA);
        //エラーチェック
        m_EncThread.m_stsThread = sts;
        QSV_ERR_MES(sts, _T("Error in getting buffered frames from vpp."));
        PrintMes(RGY_LOG_DEBUG, _T("Encode Thread: finished getting buffered frames from vpp.\n"));
    }

    //encのフレームをflush
    while (MFX_ERR_NONE <= sts && m_pmfxENC) {
        if (pSurfCheckPts) {
            pSurfCheckPts->Data.Locked--;
            pSurfCheckPts = nullptr;
        }

        if (MFX_ERR_NONE != (sts = GetFreeTask(&pCurrentTask)))
            break;

        sts = encode_one_frame(NULL);
    }
    PrintMes(RGY_LOG_DEBUG, _T("Encode Thread: finished getting buffered frames from encoder.\n"));

    //MFX_ERR_MORE_DATAはencにもうflushするべきフレームがないことを示す
    QSV_IGNORE_STS(sts, MFX_ERR_MORE_DATA);
    //エラーチェック
    m_EncThread.m_stsThread = sts;
    QSV_ERR_MES(sts, _T("Error in getting buffered frames from encoder."));

    //タスクプールのすべてのタスクの終了を確認
    while (MFX_ERR_NONE == sts) {
        sts = m_TaskPool.SynchronizeFirstTask();
    }

    // MFX_ERR_NOT_FOUNDは、正しい終了ステータス
    QSV_IGNORE_STS(sts, MFX_ERR_NOT_FOUND);
    //エラーチェック
    m_EncThread.m_stsThread = sts;
    QSV_ERR_MES(sts, _T("Error in encoding pipeline, synchronizing pipeline."));

    PrintMes(RGY_LOG_DEBUG, _T("Encode Thread: finished.\n"));
    return sts;
}
#endif

void CQSVPipeline::PrintMes(int log_level, const TCHAR *format, ...) {
    if (m_pQSVLog.get() == nullptr) {
        if (log_level <= RGY_LOG_INFO) {
            return;
        }
    } else if (log_level < m_pQSVLog->getLogLevel()) {
        return;
    }

    va_list args;
    va_start(args, format);

    int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    vector<TCHAR> buffer(len, 0);
    _vstprintf_s(buffer.data(), len, format, args);
    va_end(args);

    if (m_pQSVLog.get() != nullptr) {
        m_pQSVLog->write(log_level, buffer.data());
    } else {
        _ftprintf(stderr, _T("%s"), buffer.data());
    }
}

void CQSVPipeline::GetEncodeLibInfo(mfxVersion *ver, bool *hardware) {
    if (NULL != ver && NULL != hardware) {
        mfxIMPL impl;
        m_mfxSession.QueryIMPL(&impl);
        *hardware = !!Check_HWUsed(impl);
        *ver = m_mfxVer;
    }

}

MemType CQSVPipeline::GetMemType() {
    return m_memType;
}

RGY_ERR CQSVPipeline::GetEncodeStatusData(EncodeStatusData *data) {
    if (data == nullptr)
        return RGY_ERR_NULL_PTR;

    if (m_pStatus == nullptr)
        return RGY_ERR_NOT_INITIALIZED;

    *data = m_pStatus->GetEncodeData();
    return RGY_ERR_NONE;
}

const TCHAR *CQSVPipeline::GetInputMessage() {
    return m_pFileReader->GetInputMessage();
}

std::pair<RGY_ERR, std::unique_ptr<QSVVideoParam>> CQSVPipeline::GetOutputVideoInfo() {
    auto prmset = std::make_unique<QSVVideoParam>(m_mfxEncParams.mfx.CodecId, m_mfxVer);
    if (m_pmfxENC) {
        auto sts = err_to_rgy(m_pmfxENC->GetVideoParam(&prmset->videoPrm));
        if (sts == RGY_ERR_NOT_INITIALIZED) { // 未初期化の場合、設定しようとしたパラメータで代用する
            prmset->videoPrm = m_mfxEncParams;
            sts = RGY_ERR_NONE;
        }
        return { sts, std::move(prmset) };
    }
    if (m_vpFilters.size() > 0) {
        prmset->isVppParam = true;
        auto& lastFilter = m_vpFilters.back();
        if (lastFilter.type == VppFilterType::FILTER_MFX) {
            auto sts = err_to_rgy(lastFilter.vppmfx->mfxvpp()->GetVideoParam(&prmset->videoPrmVpp));
            if (sts == RGY_ERR_NOT_INITIALIZED) { // 未初期化の場合、設定しようとしたパラメータで代用する
                prmset->videoPrm = lastFilter.vppmfx->mfxparams();
                sts = RGY_ERR_NONE;
            }
            return { sts, std::move(prmset) };
        } else if (lastFilter.type == VppFilterType::FILTER_OPENCL) {
            auto& frameOut = lastFilter.vppcl.back()->GetFilterParam()->frameOut;
            const int blockSize = (m_mfxEncParams.mfx.CodecId == MFX_CODEC_HEVC) ? 32 : 16;
            prmset->videoPrmVpp.vpp.Out = frameinfo_rgy_to_enc(frameOut, m_encFps, rgy_rational<int>(0, 0), blockSize);
        } else {
            PrintMes(RGY_LOG_ERROR, _T("GetOutputVideoInfo: Unknown VPP filter type.\n"));
            return { RGY_ERR_UNSUPPORTED, std::move(prmset) };
        }
    }
    if (m_pmfxDEC) {
        auto sts = err_to_rgy(m_pmfxDEC->GetVideoParam(&prmset->videoPrm));
        if (sts == RGY_ERR_NOT_INITIALIZED) { // 未初期化の場合、設定しようとしたパラメータで代用する
            prmset->videoPrm = m_mfxDecParams;
            sts = RGY_ERR_NONE;
        }
        return { sts, std::move(prmset) };
    }
    PrintMes(RGY_LOG_ERROR, _T("GetOutputVideoInfo: None of the pipeline elements are detected!\n"));
    return { RGY_ERR_UNSUPPORTED, std::move(prmset) };
}

RGY_ERR CQSVPipeline::CheckCurrentVideoParam(TCHAR *str, mfxU32 bufSize) {
    mfxIMPL impl;
    m_mfxSession.QueryIMPL(&impl);

    mfxFrameInfo DstPicInfo = m_mfxEncParams.mfx.FrameInfo;

    auto [ err, outFrameInfo ] = GetOutputVideoInfo();
    if (err != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to get output frame info!\n"));
        return err;
    }

    DstPicInfo = (outFrameInfo->isVppParam) ? outFrameInfo->videoPrmVpp.vpp.Out : outFrameInfo->videoPrm.mfx.FrameInfo;

    const int workSurfaceCount = std::accumulate(m_pipelineTasks.begin(), m_pipelineTasks.end(), 0, [](int sum, std::unique_ptr<PipelineTask>& task) {
        return sum + (int)task->workSurfacesCount();
        });


    if (m_pmfxENC) {
        mfxParamSet prmSetOut;
        prmSetOut.vidprm = outFrameInfo->videoPrm;
        prmSetOut.cop    = outFrameInfo->cop;
        prmSetOut.cop2   = outFrameInfo->cop2;
        prmSetOut.cop3   = outFrameInfo->cop3;
        prmSetOut.hevc   = outFrameInfo->hevcPrm;

        CompareParam(m_prmSetIn, prmSetOut);
    }

    TCHAR cpuInfo[256] = { 0 };
    getCPUInfo(cpuInfo, _countof(cpuInfo), &m_mfxSession);

    TCHAR gpu_info[1024] = { 0 };
    if (Check_HWUsed(impl)) {
        getGPUInfo("Intel", gpu_info, _countof(gpu_info));
    }
    TCHAR info[4096] = { 0 };
    mfxU32 info_len = 0;

#define PRINT_INFO(fmt, ...) { info_len += _stprintf_s(info + info_len, _countof(info) - info_len, fmt, __VA_ARGS__); }
#define PRINT_INT_AUTO(fmt, i) { if ((i) != 0) { info_len += _stprintf_s(info + info_len, _countof(info) - info_len, fmt, i); } else { info_len += _stprintf_s(info + info_len, _countof(info) - info_len, (fmt[_tcslen(fmt)-1]=='\n') ? _T("Auto\n") : _T("Auto")); } }
    PRINT_INFO(    _T("%s\n"), get_encoder_version());
#if defined(_WIN32) || defined(_WIN64)
    OSVERSIONINFOEXW osversioninfo = { 0 };
    tstring osversionstr = getOSVersion(&osversioninfo);
    PRINT_INFO(    _T("OS             %s %s (%d) [%s]\n"), osversionstr.c_str(), rgy_is_64bit_os() ? _T("x64") : _T("x86"), osversioninfo.dwBuildNumber, getACPCodepageStr().c_str());
#else
    PRINT_INFO(    _T("OS             %s %s\n"), getOSVersion().c_str(), rgy_is_64bit_os() ? _T("x64") : _T("x86"));
#endif
    PRINT_INFO(    _T("CPU Info       %s\n"), cpuInfo);
    if (Check_HWUsed(impl)) {
        PRINT_INFO(_T("GPU Info       %s\n"), gpu_info);
    }
    if (Check_HWUsed(impl)) {
        static const TCHAR * const NUM_APPENDIX[] = { _T("st"), _T("nd"), _T("rd"), _T("th")};
        mfxU32 iGPUID = GetAdapterID(m_mfxSession);
        PRINT_INFO(    _T("Media SDK      QuickSyncVideo (hardware encoder)%s, %d%s GPU, API v%d.%d\n"),
            get_low_power_str(outFrameInfo->videoPrm.mfx.LowPower), iGPUID + 1, NUM_APPENDIX[clamp(iGPUID, 0, _countof(NUM_APPENDIX) - 1)], m_mfxVer.Major, m_mfxVer.Minor);
    } else {
        PRINT_INFO(    _T("Media SDK      software encoder, API v%d.%d\n"), m_mfxVer.Major, m_mfxVer.Minor);
    }
    PRINT_INFO(    _T("Async Depth    %d frames\n"), m_nAsyncDepth);
    PRINT_INFO(    _T("Buffer Memory  %s, %d work buffer\n"), MemTypeToStr(m_memType), workSurfaceCount);
    //PRINT_INFO(    _T("Input Frame Format   %s\n"), ColorFormatToStr(m_pFileReader->m_ColorFormat));
    //PRINT_INFO(    _T("Input Frame Type     %s\n"), list_interlaced_mfx[get_cx_index(list_interlaced_mfx, SrcPicInfo.PicStruct)].desc);
    tstring inputMes = m_pFileReader->GetInputMessage();
    for (const auto& reader : m_AudioReaders) {
        inputMes += _T("\n") + tstring(reader->GetInputMessage());
    }
    auto inputMesSplitted = split(inputMes, _T("\n"));
    for (mfxU32 i = 0; i < inputMesSplitted.size(); i++) {
        PRINT_INFO(_T("%s%s\n"), (i == 0) ? _T("Input Info     ") : _T("               "), inputMesSplitted[i].c_str());
    }

    if (m_vpFilters.size() > 0) {
        const TCHAR *m = _T("VPP Enabled    ");
        tstring vppstr;
        for (auto& block : m_vpFilters) {
            if (block.type == VppFilterType::FILTER_MFX) {
                vppstr += block.vppmfx->print();
            } else if (block.type == VppFilterType::FILTER_OPENCL) {
                for (auto& clfilter : block.vppcl) {
                    vppstr += clfilter->GetInputMessage();
                }
            } else {
                PrintMes(RGY_LOG_ERROR, _T("CheckCurrentVideoParam: Unknown VPP filter type.\n"));
                return RGY_ERR_UNSUPPORTED;
            }
        }
        std::vector<TCHAR> vpp_mes(vppstr.length() + 1, _T('\0'));
        memcpy(vpp_mes.data(), vppstr.c_str(), vpp_mes.size() * sizeof(vpp_mes[0]));
        for (TCHAR *p = vpp_mes.data(), *q; (p = _tcstok_s(p, _T("\n"), &q)) != NULL; ) {
            PRINT_INFO(_T("%s%s\n"), m, p);
            m    = _T("               ");
            p = NULL;
        }
    }
    if (m_trimParam.list.size()
        && !(m_trimParam.list[0].start == 0 && m_trimParam.list[0].fin == TRIM_MAX)) {
        PRINT_INFO(_T("%s"), _T("Trim           "));
        for (auto trim : m_trimParam.list) {
            if (trim.fin == TRIM_MAX) {
                PRINT_INFO(_T("%d-fin "), trim.start + m_trimParam.offset);
            } else {
                PRINT_INFO(_T("%d-%d "), trim.start + m_trimParam.offset, trim.fin + m_trimParam.offset);
            }
        }
        PRINT_INFO(_T("[offset: %d]\n"), m_trimParam.offset);
    }
    PRINT_INFO(_T("AVSync         %s\n"), get_chr_from_value(list_avsync, m_nAVSyncMode));
    if (m_pmfxENC) {
        PRINT_INFO(_T("Output         %s %s @ Level %s%s\n"), CodecIdToStr(outFrameInfo->videoPrm.mfx.CodecId),
            get_profile_list(outFrameInfo->videoPrm.mfx.CodecId)[get_cx_index(get_profile_list(outFrameInfo->videoPrm.mfx.CodecId), outFrameInfo->videoPrm.mfx.CodecProfile)].desc,
            get_level_list(outFrameInfo->videoPrm.mfx.CodecId)[get_cx_index(get_level_list(outFrameInfo->videoPrm.mfx.CodecId), outFrameInfo->videoPrm.mfx.CodecLevel & 0xff)].desc,
            (outFrameInfo->videoPrm.mfx.CodecId == MFX_CODEC_HEVC && (outFrameInfo->videoPrm.mfx.CodecLevel & MFX_TIER_HEVC_HIGH)) ? _T(" (high tier)") : _T(""));
    }
    PRINT_INFO(_T("%s         %dx%d%s %d:%d %0.3ffps (%d/%dfps)%s%s\n"),
        (m_pmfxENC) ? _T("      ") : _T("Output"),
        DstPicInfo.CropW, DstPicInfo.CropH, (DstPicInfo.PicStruct & MFX_PICSTRUCT_PROGRESSIVE) ? _T("p") : _T("i"),
        outFrameInfo->videoPrm.mfx.FrameInfo.AspectRatioW, outFrameInfo->videoPrm.mfx.FrameInfo.AspectRatioH,
        DstPicInfo.FrameRateExtN / (double)DstPicInfo.FrameRateExtD, DstPicInfo.FrameRateExtN, DstPicInfo.FrameRateExtD,
        (DstPicInfo.PicStruct & MFX_PICSTRUCT_PROGRESSIVE) ? _T("") : _T(", "),
        (DstPicInfo.PicStruct & MFX_PICSTRUCT_PROGRESSIVE) ? _T("") : list_interlaced_mfx[get_cx_index(list_interlaced_mfx, DstPicInfo.PicStruct)].desc);
    if (m_pFileWriter) {
        inputMesSplitted = split(m_pFileWriter->GetOutputMessage(), _T("\n"));
        for (auto mes : inputMesSplitted) {
            if (mes.length()) {
                PRINT_INFO(_T("%s%s\n"), _T("               "), mes.c_str());
            }
        }
    }
    for (auto pWriter : m_pFileWriterListAudio) {
        if (pWriter && pWriter != m_pFileWriter) {
            inputMesSplitted = split(pWriter->GetOutputMessage(), _T("\n"));
            for (auto mes : inputMesSplitted) {
                if (mes.length()) {
                    PRINT_INFO(_T("%s%s\n"), _T("               "), mes.c_str());
                }
            }
        }
    }

    if (m_pmfxENC) {
        PRINT_INFO(_T("Target usage   %s\n"), TargetUsageToStr(outFrameInfo->videoPrm.mfx.TargetUsage));
        PRINT_INFO(_T("Encode Mode    %s\n"), EncmodeToStr(outFrameInfo->videoPrm.mfx.RateControlMethod));
        if (m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_CQP) {
            PRINT_INFO(_T("CQP Value      I:%d  P:%d  B:%d\n"), outFrameInfo->videoPrm.mfx.QPI, outFrameInfo->videoPrm.mfx.QPP, outFrameInfo->videoPrm.mfx.QPB);
        } else if (rc_is_type_lookahead(m_mfxEncParams.mfx.RateControlMethod)) {
            if (m_mfxEncParams.mfx.RateControlMethod != MFX_RATECONTROL_LA_ICQ) {
                PRINT_INFO(_T("Bitrate        %d kbps\n"), outFrameInfo->videoPrm.mfx.TargetKbps * (std::max<int>)(m_mfxEncParams.mfx.BRCParamMultiplier, 1));
                PRINT_INFO(_T("%s"), _T("Max Bitrate    "));
                PRINT_INT_AUTO(_T("%d kbps\n"), outFrameInfo->videoPrm.mfx.MaxKbps * (std::max<int>)(m_mfxEncParams.mfx.BRCParamMultiplier, 1));
            }
            PRINT_INFO(_T("Lookahead      depth %d frames"), outFrameInfo->cop2.LookAheadDepth);
            if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)) {
                PRINT_INFO(_T(", quality %s"), list_lookahead_ds[get_cx_index(list_lookahead_ds, outFrameInfo->cop2.LookAheadDS)].desc);
            }
            PRINT_INFO(_T("%s"), _T("\n"));
            if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_11)) {
                if (outFrameInfo->cop3.WinBRCSize) {
                    PRINT_INFO(_T("Windowed RC    %d frames, Max %d kbps\n"), outFrameInfo->cop3.WinBRCSize, outFrameInfo->cop3.WinBRCMaxAvgKbps);
                } else {
                    PRINT_INFO(_T("%s"), _T("Windowed RC    off\n"));
                }
            }
            if (m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_LA_ICQ) {
                PRINT_INFO(_T("ICQ Quality    %d\n"), outFrameInfo->videoPrm.mfx.ICQQuality);
            }
        } else if (m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_ICQ) {
            PRINT_INFO(_T("ICQ Quality    %d\n"), outFrameInfo->videoPrm.mfx.ICQQuality);
        } else {
            PRINT_INFO(_T("Bitrate        %d kbps\n"), outFrameInfo->videoPrm.mfx.TargetKbps * (std::max<int>)(m_mfxEncParams.mfx.BRCParamMultiplier, 1));
            if (m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_AVBR) {
                //PRINT_INFO(_T("AVBR Accuracy range\t%.01lf%%"), m_mfxEncParams.mfx.Accuracy / 10.0);
                PRINT_INFO(_T("AVBR Converge  %d frames unit\n"), outFrameInfo->videoPrm.mfx.Convergence * 100);
            } else {
                PRINT_INFO(_T("%s"), _T("Max Bitrate    "));
                PRINT_INT_AUTO(_T("%d kbps\n"), outFrameInfo->videoPrm.mfx.MaxKbps * (std::max<int>)(m_mfxEncParams.mfx.BRCParamMultiplier, 1));
                if (m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_QVBR) {
                    PRINT_INFO(_T("QVBR Quality   %d\n"), outFrameInfo->cop3.QVBRQuality);
                }
            }
            if (outFrameInfo->videoPrm.mfx.BufferSizeInKB > 0) {
                PRINT_INFO(_T("VBV Bufsize    %d kbps\n"), outFrameInfo->videoPrm.mfx.BufferSizeInKB * 8 * (std::max<int>)(m_mfxEncParams.mfx.BRCParamMultiplier, 1));
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_9)) {
            auto qp_limit_str = [](mfxU8 limitI, mfxU8 limitP, mfxU8 limitB) {
                mfxU8 limit[3] = { limitI, limitP, limitB };
                if (0 == (limit[0] | limit[1] | limit[2])) {
                    return tstring(_T("none"));
                }
                if (limit[0] == limit[1] && limit[0] == limit[2]) {
                    return strsprintf(_T("%d"), limit[0]);
                }

                tstring buf;
                for (int i = 0; i < 3; i++) {
                    buf += ((i) ? _T(":") : _T(""));
                    if (limit[i]) {
                        buf += strsprintf(_T("%d"), limit[i]);
                    } else {
                        buf += _T("-");
                    }
                }
                return buf;
            };
            PRINT_INFO(_T("QP Limit       min: %s, max: %s\n"),
                qp_limit_str(outFrameInfo->cop2.MinQPI, outFrameInfo->cop2.MinQPP, outFrameInfo->cop2.MinQPB).c_str(),
                qp_limit_str(outFrameInfo->cop2.MaxQPI, outFrameInfo->cop2.MaxQPP, outFrameInfo->cop2.MaxQPB).c_str());
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_7)) {
            PRINT_INFO(_T("Trellis        %s\n"), list_avc_trellis[get_cx_index(list_avc_trellis_for_options, outFrameInfo->cop2.Trellis)].desc);
        }

        if (outFrameInfo->videoPrm.mfx.CodecId == MFX_CODEC_AVC && !Check_HWUsed(impl)) {
            PRINT_INFO(_T("CABAC          %s\n"), (outFrameInfo->cop.CAVLC == MFX_CODINGOPTION_ON) ? _T("off") : _T("on"));
            PRINT_INFO(_T("RDO            %s\n"), (outFrameInfo->cop.RateDistortionOpt == MFX_CODINGOPTION_ON) ? _T("on") : _T("off"));
            if ((outFrameInfo->cop.MVSearchWindow.x | outFrameInfo->cop.MVSearchWindow.y) == 0) {
                PRINT_INFO(_T("mv search      precision: %s\n"), list_mv_presicion[get_cx_index(list_mv_presicion, outFrameInfo->cop.MVPrecision)].desc);
            } else {
                PRINT_INFO(_T("mv search      precision: %s, window size:%dx%d\n"), list_mv_presicion[get_cx_index(list_mv_presicion, outFrameInfo->cop.MVPrecision)].desc, outFrameInfo->cop.MVSearchWindow.x, outFrameInfo->cop.MVSearchWindow.y);
            }
            PRINT_INFO(_T("min pred size  inter: %s   intra: %s\n"), list_pred_block_size[get_cx_index(list_pred_block_size, outFrameInfo->cop.InterPredBlockSize)].desc, list_pred_block_size[get_cx_index(list_pred_block_size, outFrameInfo->cop.IntraPredBlockSize)].desc);
        }
        PRINT_INFO(_T("%s"), _T("Ref frames     "));
        PRINT_INT_AUTO(_T("%d frames\n"), outFrameInfo->videoPrm.mfx.NumRefFrame);

        PRINT_INFO(_T("%s"), _T("Bframes        "));
        switch (outFrameInfo->videoPrm.mfx.GopRefDist) {
        case 0:  PRINT_INFO(_T("%s"), _T("Auto\n")); break;
        case 1:  PRINT_INFO(_T("%s"), _T("none\n")); break;
        default: PRINT_INFO(_T("%d frame%s%s%s\n"),
            outFrameInfo->videoPrm.mfx.GopRefDist - 1, (outFrameInfo->videoPrm.mfx.GopRefDist > 2) ? _T("s") : _T(""),
            check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8) ? _T(", B-pyramid: ") : _T(""),
            (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8) ? ((MFX_B_REF_PYRAMID == outFrameInfo->cop2.BRefType) ? _T("on") : _T("off")) : _T(""))); break;
        }

        //PRINT_INFO(    _T("Idr Interval    %d\n"), outFrameInfo->videoPrm.mfx.IdrInterval);
        PRINT_INFO(_T("%s"), _T("Max GOP Length "));
        PRINT_INT_AUTO(_T("%d frames\n"), outFrameInfo->videoPrm.mfx.GopPicSize);
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)) {
            //PRINT_INFO(    _T("GOP Structure           "));
            //bool adaptiveIOn = (MFX_CODINGOPTION_ON == outFrameInfo->cop2.AdaptiveI);
            //bool adaptiveBOn = (MFX_CODINGOPTION_ON == outFrameInfo->cop2.AdaptiveB);
            //if (!adaptiveIOn && !adaptiveBOn) {
            //    PRINT_INFO(_T("fixed\n"))
            //} else {
            //    PRINT_INFO(_T("Adaptive %s%s%s insert\n"),
            //        (adaptiveIOn) ? _T("I") : _T(""),
            //        (adaptiveIOn && adaptiveBOn) ? _T(",") : _T(""),
            //        (adaptiveBOn) ? _T("B") : _T(""));
            //}
        }
        if (outFrameInfo->videoPrm.mfx.NumSlice >= 2) {
            PRINT_INFO(_T("Slices         %d\n"), outFrameInfo->videoPrm.mfx.NumSlice);
        }

        if (outFrameInfo->videoPrm.mfx.CodecId == MFX_CODEC_VP8) {
            PRINT_INFO(_T("Sharpness      %d\n"), outFrameInfo->copVp8.SharpnessLevel);
        }
        { const auto &vui_str = m_encVUI.print_all();
        if (vui_str.length() > 0) {
            PRINT_INFO(_T("VUI            %s\n"), vui_str.c_str());
        }
        }
        if (m_HDRSei) {
            const auto masterdisplay = m_HDRSei->print_masterdisplay();
            const auto maxcll = m_HDRSei->print_maxcll();
            if (masterdisplay.length() > 0) {
                const tstring tstr = char_to_tstring(masterdisplay);
                const auto splitpos = tstr.find(_T("WP("));
                if (splitpos == std::string::npos) {
                    PRINT_INFO(_T("MasteringDisp  %s\n"), tstr.c_str());
                } else {
                    PRINT_INFO(_T("MasteringDisp  %s\n")
                               _T("               %s\n"),
                        tstr.substr(0, splitpos-1).c_str(), tstr.substr(splitpos).c_str());
                }
            }
            if (maxcll.length() > 0) {
                PRINT_INFO(_T("MaxCLL/MaxFALL %s\n"), char_to_tstring(maxcll).c_str());
            }
        }

        //last line
        tstring extFeatures;
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_6)) {
            if (outFrameInfo->cop2.MBBRC  == MFX_CODINGOPTION_ON) {
                extFeatures += _T("PerMBRC ");
            }
            if (outFrameInfo->cop2.ExtBRC == MFX_CODINGOPTION_ON) {
                extFeatures += _T("ExtBRC ");
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_9)) {
            if (outFrameInfo->cop2.DisableDeblockingIdc) {
                extFeatures += _T("No-Deblock ");
            }
            if (outFrameInfo->cop2.IntRefType) {
                extFeatures += _T("Intra-Refresh ");
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_13)) {
            if (outFrameInfo->cop3.DirectBiasAdjustment == MFX_CODINGOPTION_ON) {
                extFeatures += _T("DirectBiasAdjust ");
            }
            if (outFrameInfo->cop3.GlobalMotionBiasAdjustment == MFX_CODINGOPTION_ON) {
                extFeatures += strsprintf(_T("MVCostScaling=%d "), outFrameInfo->cop3.MVCostScalingFactor);
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_16)) {
            if (outFrameInfo->cop3.WeightedPred != MFX_WEIGHTED_PRED_UNKNOWN) {
                extFeatures += _T("WeightP ");
            }
            if (outFrameInfo->cop3.WeightedBiPred != MFX_WEIGHTED_PRED_UNKNOWN) {
                extFeatures += _T("WeightB ");
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_17)) {
            if (outFrameInfo->cop3.FadeDetection == MFX_CODINGOPTION_ON) {
                extFeatures += _T("FadeDetect ");
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_19)) {
            if (outFrameInfo->cop3.EnableQPOffset == MFX_CODINGOPTION_ON) {
                extFeatures += _T("QPOffset ");
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_26)) {
            if (outFrameInfo->cop3.ExtBrcAdaptiveLTR == MFX_CODINGOPTION_ON) {
                extFeatures += _T("AdaptiveLTR ");
            }
        }
        //if (outFrameInfo->cop.AUDelimiter == MFX_CODINGOPTION_ON) {
        //    extFeatures += _T("aud ");
        //}
        //if (outFrameInfo->cop.PicTimingSEI == MFX_CODINGOPTION_ON) {
        //    extFeatures += _T("pic_struct ");
        //}
        //if (outFrameInfo->cop.SingleSeiNalUnit == MFX_CODINGOPTION_ON) {
        //    extFeatures += _T("SingleSEI ");
        //}
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_23)) {
            if (outFrameInfo->cop3.RepartitionCheckEnable == MFX_CODINGOPTION_ON) {
                extFeatures += _T("RepartitionCheck ");
            }
        }
        if (m_mfxEncParams.mfx.CodecId == MFX_CODEC_HEVC) {
            if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_26)) {
                if (outFrameInfo->cop3.TransformSkip == MFX_CODINGOPTION_ON) {
                    extFeatures += _T("tskip ");
                }
                if (outFrameInfo->hevcPrm.LCUSize != 0) {
                    extFeatures += strsprintf(_T("ctu:%d "), outFrameInfo->hevcPrm.LCUSize);
                }
                if (outFrameInfo->hevcPrm.SampleAdaptiveOffset != 0) {
                    extFeatures += strsprintf(_T("sao:%s "), get_chr_from_value(list_hevc_sao, outFrameInfo->hevcPrm.SampleAdaptiveOffset));
                }
            }
        }
        if (extFeatures.length() > 0) {
            PRINT_INFO(_T("Ext. Features  %s\n"), extFeatures.c_str());
        }
    }

    PrintMes(RGY_LOG_INFO, info);
    if (str && bufSize > 0) {
        _tcscpy_s(str, bufSize, info);
    }

    return RGY_ERR_NONE;
#undef PRINT_INFO
#undef PRINT_INT_AUTO
}

