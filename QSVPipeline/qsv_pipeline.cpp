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
#include "rgy_osdep.h"
#include "rgy_util.h"
#pragma warning(push)
#pragma warning(disable: 4244)
#pragma warning(disable: 4834)
RGY_DISABLE_WARNING_PUSH
RGY_DISABLE_WARNING_STR("-Wunused-result")
#define TTMATH_NOASM
#include "ttmath/ttmath.h"
RGY_DISABLE_WARNING_POP
#pragma warning(pop)
#include "qsv_pipeline.h"
#include "qsv_pipeline_ctrl.h"
#include "qsv_session.h"
#include "qsv_query.h"
#include "rgy_def.h"
#include "rgy_env.h"
#include "rgy_device_info_cache.h"
#include "rgy_filesystem.h"
#include "rgy_input.h"
#include "rgy_output.h"
#include "rgy_input_raw.h"
#include "rgy_input_vpy.h"
#include "rgy_input_avs.h"
#include "rgy_input_avi.h"
#include "rgy_input_sm.h"
#include "rgy_input_avcodec.h"
#include "rgy_filter.h"
#include "rgy_filter_colorspace.h"
#include "rgy_filter_rff.h"
#include "rgy_filter_afs.h"
#include "rgy_filter_nnedi.h"
#include "rgy_filter_yadif.h"
#include "rgy_filter_mpdecimate.h"
#include "rgy_filter_decimate.h"
#include "rgy_filter_decomb.h"
#include "rgy_filter_delogo.h"
#include "rgy_filter_convolution3d.h"
#include "rgy_filter_denoise_dct.h"
#include "rgy_filter_smooth.h"
#include "rgy_filter_denoise_fft3d.h"
#include "rgy_filter_denoise_knn.h"
#include "rgy_filter_denoise_nlmeans.h"
#include "rgy_filter_denoise_pmd.h"
#include "rgy_filter_subburn.h"
#include "rgy_filter_resize.h"
#include "rgy_filter_libplacebo.h"
#include "rgy_filter_transform.h"
#include "rgy_filter_unsharp.h"
#include "rgy_filter_edgelevel.h"
#include "rgy_filter_warpsharp.h"
#include "rgy_filter_deband.h"
#include "rgy_filter_ssim.h"
#include "rgy_filter_overlay.h"
#include "rgy_filter_curves.h"
#include "rgy_filter_tweak.h"
#include "rgy_output_avcodec.h"
#include "rgy_bitstream.h"
#include "qsv_hw_device.h"
#include "qsv_allocator.h"
#include "qsv_allocator_sys.h"
#include "rgy_avlog.h"
#include "rgy_chapter.h"
#include "rgy_timecode.h"
#include "rgy_aspect_ratio.h"
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

#if LIBVA_SUPPORT
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

bool CQSVPipeline::CompareParam(const QSVVideoParam& prmIn, const QSVVideoParam& prmOut) {
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
    COMPARE_INT(videoPrm.AsyncDepth,             0);
    COMPARE_HEX(videoPrm.IOPattern,              0);
    COMPARE_INT(videoPrm.mfx.NumThread,          0);
    if (prmOut.videoPrm.mfx.BRCParamMultiplier > 1) {
        COMPARE_INT(videoPrm.mfx.BRCParamMultiplier, -1);
    }
    COMPARE_INT(videoPrm.mfx.LowPower,           0);
    COMPARE_LST(videoPrm.mfx.CodecId,            0, list_codec_mfx);
    COMPARE_LST(videoPrm.mfx.CodecProfile,       0, get_profile_list(codec_enc_to_rgy(prmIn.videoPrm.mfx.CodecId)));
    COMPARE_LST(videoPrm.mfx.CodecLevel,         0, get_level_list(codec_enc_to_rgy(prmIn.videoPrm.mfx.CodecId)));
    COMPARE_INT(videoPrm.mfx.NumThread,          0);
    COMPARE_INT(videoPrm.mfx.TargetUsage,       -1);
    COMPARE_INT(videoPrm.mfx.GopPicSize,         0);
    COMPARE_INT(videoPrm.mfx.GopRefDist,         0);
    COMPARE_INT(videoPrm.mfx.GopOptFlag,         0);
    COMPARE_INT(videoPrm.mfx.IdrInterval,        0);
    COMPARE_STR(videoPrm.mfx.RateControlMethod,  0, EncmodeToStr);
    if (prmIn.videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_CQP) {
        COMPARE_INT(videoPrm.mfx.QPI, -1);
        COMPARE_INT(videoPrm.mfx.QPP, -1);
        COMPARE_INT(videoPrm.mfx.QPB, -1);
    } else if (rc_is_type_lookahead(m_encParams.videoPrm.mfx.RateControlMethod)) {
        COMPARE_INT(cop2.LookAheadDepth, -1);
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)) {
            COMPARE_LST(cop2.LookAheadDS, 0, list_lookahead_ds);
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_11)) {
            COMPARE_INT(cop3.WinBRCSize,       0);
            COMPARE_INT(cop3.WinBRCMaxAvgKbps, 0);
        }
        if (MFX_RATECONTROL_LA_ICQ == m_encParams.videoPrm.mfx.RateControlMethod) {
            COMPARE_INT(videoPrm.mfx.ICQQuality, -1);
        }
    } else if (MFX_RATECONTROL_ICQ == m_encParams.videoPrm.mfx.RateControlMethod) {
        COMPARE_INT(videoPrm.mfx.ICQQuality, -1);
    } else {
        COMPARE_INT(videoPrm.mfx.TargetKbps, 0);
        //COMPARE_INT(videoPrm.mfx.BufferSizeInKB, 0);
        if (m_encParams.videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_AVBR) {
            COMPARE_INT(videoPrm.mfx.TargetKbps, 0);
        } else {
            COMPARE_INT(videoPrm.mfx.MaxKbps, 0);
            if (m_encParams.videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_QVBR) {
                COMPARE_INT(cop3.QVBRQuality, -1);
            }
        }
    }
    COMPARE_INT(videoPrm.mfx.NumSlice,             0);
    COMPARE_INT(videoPrm.mfx.NumRefFrame,          0);
    COMPARE_INT(videoPrm.mfx.EncodedOrder,         0);
    COMPARE_INT(videoPrm.mfx.ExtendedPicStruct,    0);
    COMPARE_INT(videoPrm.mfx.TimeStampCalc,        0);
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_6)) {
        COMPARE_INT(videoPrm.mfx.SliceGroupsPresent, 0);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_15)) {
        COMPARE_TRI(videoPrm.mfx.LowPower, 0);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_16)) {
        COMPARE_INT(videoPrm.mfx.MaxDecFrameBuffering, 0);
    }

    COMPARE_TRI(cop.RateDistortionOpt,    0);
    COMPARE_INT(cop.MECostType,           0);
    COMPARE_INT(cop.MESearchType,         0);
    //Deprecated: COMPARE_TRI(cop.EndOfSequence,        0);
    COMPARE_TRI(cop.FramePicture,         0);
    COMPARE_TRI(cop.CAVLC,                0);
    COMPARE_TRI(cop.ViewOutput,           0);
    COMPARE_TRI(cop.VuiVclHrdParameters,  0);
    COMPARE_TRI(cop.RefPicListReordering, 0);
    COMPARE_TRI(cop.ResetRefList,         0);
    COMPARE_INT(cop.MaxDecFrameBuffering, 0);
    COMPARE_TRI(cop.AUDelimiter,          0);
    //Deprecated: COMPARE_TRI(cop.EndOfStream,          0);
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
        //COMPARE_TRI(cop2.BitrateLimit,    0);
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
        if (rc_is_type_lookahead(m_encParams.videoPrm.mfx.RateControlMethod)) {
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
        COMPARE_TRI(cop3.TransformSkip,      0);
        COMPARE_INT(hevcPrm.SampleAdaptiveOffset,  MFX_SAO_UNKNOWN);
        COMPARE_INT(hevcPrm.LCUSize, 0);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_2_2)) {
        COMPARE_TRI(cop3.AdaptiveCQM, 0);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_2_4)) {
        COMPARE_TRI(cop3.AdaptiveRef, 0);
        COMPARE_TRI(cop3.AdaptiveLTR, 0);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_2_5)) {
        COMPARE_TRI(av1BitstreamPrm.WriteIVFHeaders, 0);
        COMPARE_INT(av1ResolutionPrm.FrameWidth, 0);
        COMPARE_INT(av1ResolutionPrm.FrameHeight, 0);
        COMPARE_INT(av1TilePrm.NumTileRows, 0);
        COMPARE_INT(av1TilePrm.NumTileColumns, 0);
        COMPARE_INT(av1TilePrm.NumTileGroups, 0);
        COMPARE_INT(hyperModePrm.Mode, 0);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_2_9)) {
        COMPARE_INT(tuneEncQualityPrm.TuneQuality, 0);
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

RGY_ERR CQSVPipeline::InitMfxDecParams() {
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
        mfxIMPL impl;
        m_device->mfxSession().QueryIMPL(&impl);
        m_mfxDEC = std::make_unique<QSVMfxDec>(m_device->hwdev(), m_device->allocator(), m_mfxVer, impl, m_device->memType(), m_sessionParams, m_device->deviceNum(), m_pQSVLog);

        sts = m_mfxDEC->InitMFXSession();
        RGY_ERR(sts, _T("InitMfxDecParams: Failed init session for hw decoder."));

        sts = m_mfxDEC->SetParam(m_pFileReader->getInputCodec(), m_DecInputBitstream, m_pFileReader->GetInputFrameInfo());
        RGY_ERR(sts, _T("InitMfxDecParams: Failed set param for hw decoder."));

        if (!bGotHeader) {
            //最初のフレームそのものをヘッダーとして使用している場合、一度データをクリアする
            //メインループに入った際に再度第1フレームを読み込むようにする。
            m_DecInputBitstream.clear();
        }
    }
#endif
    return RGY_ERR_NONE;
}

std::pair<RGY_ERR, QSVEncFeatures> CQSVPipeline::CheckMFXRCMode(QSVRCParam& rcParam, sInputParams *pInParams, const int codecMaxQP) {
    //エンコードモードのチェック
    QSVEncFeatures availableFeaures;
    if (pInParams->functionMode == QSVFunctionMode::Auto) {
        auto availableFeauresPG = m_device->getEncodeFeature(rcParam.encMode, pInParams->codec, false);
        auto availableFeauresFF = m_device->getEncodeFeature(rcParam.encMode, pInParams->codec, true);
        if (!!availableFeauresPG && !!availableFeauresFF) {
            // 両方サポートされている場合
            if (pInParams->codec == RGY_CODEC_H264) {
                availableFeaures = availableFeauresPG;
                pInParams->functionMode = QSVFunctionMode::PG;
            } else {
                availableFeaures = availableFeauresFF;
                pInParams->functionMode = QSVFunctionMode::FF;
            }
        } else if (!!availableFeauresPG) {
            // PGのみがサポートされている場合
            availableFeaures = availableFeauresPG;
            pInParams->functionMode = QSVFunctionMode::PG;
        } else if (!!availableFeauresFF) {
            // FFのみがサポートされている場合
            availableFeaures = availableFeauresFF;
            pInParams->functionMode = QSVFunctionMode::FF;
        } else {
            // どちらもサポートされていない場合
            if (pInParams->codec == RGY_CODEC_H264) {
                availableFeaures = availableFeauresPG;
                pInParams->functionMode = QSVFunctionMode::PG;
            } else {
                availableFeaures = availableFeauresFF;
                pInParams->functionMode = QSVFunctionMode::FF;
            }
        }
        PrintMes(RGY_LOG_DEBUG, _T("Auto select function mode for %s %s: %s\n"), CodecToStr(pInParams->codec).c_str(), EncmodeToStr(rcParam.encMode),
            get_cx_desc(list_qsv_function_mode, (int)pInParams->functionMode));
    } else {
        availableFeaures = m_device->getEncodeFeature(rcParam.encMode, pInParams->codec, pInParams->functionMode == QSVFunctionMode::FF);
        if (!availableFeaures) {
            availableFeaures = m_device->getEncodeFeature(rcParam.encMode, pInParams->codec, pInParams->functionMode != QSVFunctionMode::FF);
            if (!!availableFeaures) {
                auto reverseMode = QSVFunctionMode::Auto;
                switch (pInParams->functionMode) {
                case QSVFunctionMode::FF: reverseMode = QSVFunctionMode::PG; break;
                case QSVFunctionMode::PG: reverseMode = QSVFunctionMode::FF; break;
                default: return { RGY_ERR_UNKNOWN, availableFeaures };
                }
                PrintMes(RGY_LOG_WARN, _T("%s is not supported on this platform, switched to %s mode.\n"),
                    get_cx_desc(list_qsv_function_mode, (int)pInParams->functionMode), get_cx_desc(list_qsv_function_mode, (int)reverseMode));
                pInParams->functionMode = reverseMode;
            }
        }
    }
    PrintMes(RGY_LOG_DEBUG, _T("Detected avaliable features for hw API v%d.%02d, %s%s, %s\n%s\n"),
        m_mfxVer.Major, m_mfxVer.Minor,
        CodecToStr(pInParams->codec).c_str(), get_cx_desc(list_qsv_function_mode, (int)pInParams->functionMode),
        EncmodeToStr(rcParam.encMode), MakeFeatureListStr(availableFeaures).c_str());

    // サポートされていたらOK
    if (availableFeaures & ENC_FEATURE_CURRENT_RC) {
        return { RGY_ERR_NONE, availableFeaures };
    }
    //そもそもこのコーデックがサポートされているかどうか確認する
    if (   rcParam.encMode == MFX_RATECONTROL_CQP
        || rcParam.encMode == MFX_RATECONTROL_VBR
        || rcParam.encMode == MFX_RATECONTROL_CBR
        || !(m_device->getEncodeFeature(MFX_RATECONTROL_CQP, pInParams->codec, pInParams->functionMode == QSVFunctionMode::FF) & ENC_FEATURE_CURRENT_RC)) {
        if (!(m_device->getEncodeFeature(MFX_RATECONTROL_CQP, pInParams->codec, pInParams->functionMode != QSVFunctionMode::FF) & ENC_FEATURE_CURRENT_RC)) {
            PrintMes(RGY_LOG_ERROR, _T("%s encoding is not supported on current platform.\n"), CodecToStr(pInParams->codec).c_str());
            return { RGY_ERR_INVALID_VIDEO_PARAM, availableFeaures };
        }
        auto reverseMode = QSVFunctionMode::Auto;
        switch (pInParams->functionMode) {
        case QSVFunctionMode::FF: reverseMode = QSVFunctionMode::PG; break;
        case QSVFunctionMode::PG: reverseMode = QSVFunctionMode::FF; break;
        default: return { RGY_ERR_UNKNOWN, availableFeaures };
        }
        PrintMes(RGY_LOG_WARN, _T("%s is not supported on this platform, switched to %s mode.\n"),
            get_cx_desc(list_qsv_function_mode, (int)pInParams->functionMode), get_cx_desc(list_qsv_function_mode, (int)reverseMode));
        pInParams->functionMode = reverseMode;
    }
    const auto rc_error_log_level = (pInParams->fallbackRC) ? RGY_LOG_WARN : RGY_LOG_ERROR;
    PrintMes(rc_error_log_level, _T("%s mode is not supported on current platform.\n"), EncmodeToStr(rcParam.encMode));
    if (MFX_RATECONTROL_LA == rcParam.encMode) {
        if (!check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_7)) {
            PrintMes(rc_error_log_level, _T("Lookahead mode is only supported by API v1.7 or later.\n"));
        }
    }
    if (   MFX_RATECONTROL_ICQ    == rcParam.encMode
        || MFX_RATECONTROL_LA_ICQ == rcParam.encMode
        || MFX_RATECONTROL_VCM    == rcParam.encMode) {
        if (!check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)) {
            PrintMes(rc_error_log_level, _T("%s mode is only supported by API v1.8 or later.\n"), EncmodeToStr(rcParam.encMode));
        }
    }
    if (   MFX_RATECONTROL_LA_HRD == rcParam.encMode
        || MFX_RATECONTROL_QVBR   == rcParam.encMode) {
        if (!check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_11)) {
            PrintMes(rc_error_log_level, _T("%s mode is only supported by API v1.11 or later.\n"), EncmodeToStr(rcParam.encMode));
        }
    }
    if (!pInParams->fallbackRC) {
        return { RGY_ERR_INVALID_VIDEO_PARAM, availableFeaures };
    }
    //fallback
    //fallbackの候補リスト、優先度の高い順にセットする
    std::vector<int> check_rc_list;
    //現在のレート制御モードは使用できないので、それ以外を確認する
    auto check_rc_add = [pInParams, &rcParam, &check_rc_list](const int rc_mode) {
        if (rcParam.encMode != rc_mode) {
            check_rc_list.push_back(rc_mode);
        }
    };

    //品質指定系の場合、若干補正をかけた値を設定する
    int nAdjustedQP[3] = { QSV_DEFAULT_QPI, QSV_DEFAULT_QPP, QSV_DEFAULT_QPB };
    if (isRCBitrateMode(rcParam.encMode)) {
        //ビットレートモードなら、QVBR->VBRをチェックする
        check_rc_add(MFX_RATECONTROL_QVBR);
        check_rc_add(MFX_RATECONTROL_VBR);
    } else {
        //固定品質モードなら、ICQ->CQPをチェックする
        check_rc_add(MFX_RATECONTROL_ICQ);
        check_rc_add(MFX_RATECONTROL_CQP);
        //品質指定系の場合、若干補正をかけた値を設定する
        if (rcParam.encMode == MFX_RATECONTROL_LA_ICQ) {
            nAdjustedQP[0] = rcParam.icqQuality - 8;
            nAdjustedQP[1] = rcParam.icqQuality - 6;
            nAdjustedQP[2] = rcParam.icqQuality - 3;
        } else if (rcParam.encMode == MFX_RATECONTROL_ICQ) {
            nAdjustedQP[0] = rcParam.icqQuality - 1;
            nAdjustedQP[1] = rcParam.icqQuality + 1;
            nAdjustedQP[2] = rcParam.icqQuality + 4;
        } else if (rcParam.encMode == MFX_RATECONTROL_CQP) {
            nAdjustedQP[0] = rcParam.qp.qpI;
            nAdjustedQP[1] = rcParam.qp.qpP;
            nAdjustedQP[2] = rcParam.qp.qpB;
        }
    }
    //check_rc_listに設定したfallbackの候補リストをチェックする
    bool bFallbackSuccess = false;
    for (uint32_t i = 0; i < (uint32_t)check_rc_list.size(); i++) {
        auto availRCFeatures = m_device->getEncodeFeature(check_rc_list[i], pInParams->codec, pInParams->functionMode == QSVFunctionMode::FF);
        if (availRCFeatures & ENC_FEATURE_CURRENT_RC) {
            rcParam.encMode = (uint16_t)check_rc_list[i];
            if (rcParam.encMode == MFX_RATECONTROL_LA_ICQ) {
                rcParam.icqQuality = (uint16_t)clamp(nAdjustedQP[1] + 6, 1, codecMaxQP);
            } else if (rcParam.encMode == MFX_RATECONTROL_LA_ICQ) {
                rcParam.icqQuality = (uint16_t)clamp(nAdjustedQP[1], 1, codecMaxQP);
            } else if (rcParam.encMode == MFX_RATECONTROL_CQP) {
                rcParam.qp.qpI = clamp(nAdjustedQP[0], 0, codecMaxQP);
                rcParam.qp.qpP = clamp(nAdjustedQP[1], 0, codecMaxQP);
                rcParam.qp.qpB = clamp(nAdjustedQP[2], 0, codecMaxQP);
            }
            bFallbackSuccess = true;
            availableFeaures = availRCFeatures;
            PrintMes(rc_error_log_level, _T("Falling back to %s mode.\n%s\n"), EncmodeToStr(rcParam.encMode), MakeFeatureListStr(availableFeaures).c_str());
            break;
        }
    }
    //なんらかの理由でフォールバックできなかったらエラー終了
    if (!bFallbackSuccess) {
        return { RGY_ERR_INVALID_VIDEO_PARAM, availableFeaures };
    }
    return { RGY_ERR_NONE, availableFeaures };
}

RGY_ERR CQSVPipeline::InitMfxEncodeParams(sInputParams *pInParams, std::vector<std::unique_ptr<QSVDevice>>& devList) {
    if (pInParams->codec == RGY_CODEC_RAW) {
        PrintMes(RGY_LOG_DEBUG, _T("Raw codec is selected, disable encode.\n"));
        return RGY_ERR_NONE;
    }
    const mfxU32 blocksz = (pInParams->codec == RGY_CODEC_HEVC) ? 32 : 16;
    auto print_feature_warnings = [this](RGYLogLevel log_level, const TCHAR *feature_name) {
        PrintMes(log_level, _T("%s is not supported on current platform, disabled.\n"), feature_name);
    };

    const int encodeBitDepth = getEncoderBitdepth(pInParams);
    if (encodeBitDepth <= 0) {
        PrintMes(RGY_LOG_ERROR, _T("Unknown codec.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    const int codecMaxQP = (pInParams->codec == RGY_CODEC_AV1) ? 255 : 51 + (encodeBitDepth - 8) * 6;
    PrintMes(RGY_LOG_DEBUG, _T("encodeBitDepth: %d, codecMaxQP: %d.\n"), encodeBitDepth, codecMaxQP);

    {
        const auto encCsp = getEncoderCsp(pInParams);
        if (RGY_CSP_CHROMA_FORMAT[encCsp] == RGY_CHROMAFMT_YUV444) {
            if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_15)) {
                if (pInParams->functionMode != QSVFunctionMode::FF) {
                    PrintMes(RGY_LOG_WARN, _T("Switched to fixed function (FF) mode, as encoding in YUV444 requires FF mode.\n"));
                    pInParams->functionMode = QSVFunctionMode::FF;
                    m_encParams.videoPrm.mfx.LowPower = (mfxU16)MFX_CODINGOPTION_ON;
                }
            } else {
                PrintMes(RGY_LOG_ERROR, _T("Encoding in YUV444 is not supported on this platform.\n"));
                return RGY_ERR_UNSUPPORTED;
            }
        }
    }

    auto [ err, availableFeaures ] = CheckMFXRCMode(pInParams->rcParam, pInParams, codecMaxQP);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    for (auto& rc : pInParams->dynamicRC) {
        auto [err2, availableFeaures2 ] = CheckMFXRCMode(rc, pInParams, codecMaxQP);
        if (err2 != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_WARN, _T("Unsupported dynamic rc param for frame %d-%d, will be disabled.\n"), rc.start, rc.end);
            PrintMes(RGY_LOG_WARN, _T("  paramter was %s.\n"), rc.print().c_str());
            rc.start = rc.end = -1;
        }
    }
    m_dynamicRC = pInParams->dynamicRC;

    // 並列エンコード時のチェック
    if (pInParams->ctrl.parallelEnc.isEnabled()) {
        pInParams->hyperMode = MFX_HYPERMODE_OFF; // 並列エンコード時はHyperModeは使用しない
        for (auto& dev2 : devList) {
            if (dev2) {
                const auto dev2Feature = dev2->getEncodeFeature(pInParams->rcParam.encMode, pInParams->codec, pInParams->functionMode == QSVFunctionMode::FF);
                availableFeaures &= dev2Feature;
            }
        }
    }

    // HyperModeがらみのチェック
    if (pInParams->hyperMode != MFX_HYPERMODE_OFF) {
        if (!(availableFeaures & ENC_FEATURE_HYPER_MODE)) {
            if (ENABLE_HYPER_MODE
                && pInParams->codec == RGY_CODEC_HEVC
                && OVERRIDE_HYPER_MODE_HEVC_FROM_H264
                && check_lib_version(m_mfxVer, MFX_LIB_VERSION_2_5)) {
                // HEVCのhyper modeのチェックは使用できる場合でもなぜか成功しない
                // 原因不明だが、まずはH.264の結果を参照するようにする
                const auto availRCFeaturesH264 = m_device->getEncodeFeature(pInParams->rcParam.encMode, RGY_CODEC_H264, pInParams->functionMode == QSVFunctionMode::FF);
                if (availRCFeaturesH264 & ENC_FEATURE_HYPER_MODE) {
                    availableFeaures |= ENC_FEATURE_HYPER_MODE;
                }
            }
        }
        if (!(availableFeaures & ENC_FEATURE_HYPER_MODE)) {
            if (pInParams->hyperMode == MFX_HYPERMODE_ON) {
                print_feature_warnings(RGY_LOG_WARN, _T("HyperMode"));
            }
            pInParams->hyperMode = MFX_HYPERMODE_OFF;
        }
    }
    if (pInParams->hyperMode == MFX_HYPERMODE_ON) {
        //HyperModeの対象となるGPUのfeature取得を行い、andをとる
        for (auto& dev2 : devList) {
            if (dev2) { // 自分自身はすでにm_deviceにmoveして、devListにはいなくなっている
                const auto dev2Feature = dev2->getEncodeFeature(pInParams->rcParam.encMode, pInParams->codec, pInParams->functionMode == QSVFunctionMode::FF);
                if (dev2Feature & ENC_FEATURE_HYPER_MODE) { // HyperModeに対応するGPUを選択
                    PrintMes(RGY_LOG_DEBUG, _T("Detected avaliable features for hyper mode, dev %d, %s\n%s\n"), (int)dev2->deviceNum(), EncmodeToStr(pInParams->rcParam.encMode), MakeFeatureListStr(dev2Feature).c_str());
                    availableFeaures &= dev2Feature;
                }
            }
        }
        if (pInParams->bopenGOP) {
            PrintMes(RGY_LOG_WARN, _T("OpenGOP is not supported with hyper-mode on, disabled.\n"));
            pInParams->bopenGOP = false;
        }
    }

    const bool gopRefDistAsBframes = gopRefDistAsBframe(pInParams->codec);
    if (pInParams->GopRefDist != 1 && !(availableFeaures & ENC_FEATURE_GOPREFDIST)) {
        print_feature_warnings(RGY_LOG_WARN, (gopRefDistAsBframes) ? _T("B frame") : _T("GopRefDist"));
        pInParams->GopRefDist = 1; //Bframe = 0
    }
    if (pInParams->GopRefDist == QSV_GOP_REF_DIST_AUTO) {
        switch (pInParams->codec) {
        case RGY_CODEC_HEVC: pInParams->GopRefDist = QSV_DEFAULT_HEVC_GOP_REF_DIST; break;
        case RGY_CODEC_AV1:  pInParams->GopRefDist = QSV_DEFAULT_AV1_GOP_REF_DIST; break;
        case RGY_CODEC_H264:
        default:             pInParams->GopRefDist = QSV_DEFAULT_H264_GOP_REF_DIST; break;
        }
    }
    //その他機能のチェック
    if (pInParams->bAdaptiveI && !(availableFeaures & ENC_FEATURE_ADAPTIVE_I)) {
        PrintMes(RGY_LOG_WARN, _T("Adaptve I-frame insert is not supported on current platform, disabled.\n"));
        pInParams->bAdaptiveI.reset();
    }
    if (pInParams->bAdaptiveB && !(availableFeaures & ENC_FEATURE_ADAPTIVE_B)) {
        PrintMes(RGY_LOG_WARN, _T("Adaptve B-frame insert is not supported on current platform, disabled.\n"));
        pInParams->bAdaptiveB.reset();
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
        pInParams->extBRC.reset();
    }
    if (pInParams->tuneQuality != MFX_ENCODE_TUNE_OFF && !(availableFeaures & ENC_FEATURE_TUNE_ENCODE_QUALITY)) {
        print_feature_warnings(RGY_LOG_WARN, _T("Tune Quality"));
        pInParams->tuneQuality = MFX_ENCODE_TUNE_OFF;
    }
    if (pInParams->scenarioInfo != MFX_SCENARIO_UNKNOWN && !(availableFeaures & ENC_FEATURE_SCENARIO_INFO)) {
        print_feature_warnings(RGY_LOG_WARN, _T("Scenario Info"));
        pInParams->scenarioInfo = MFX_SCENARIO_UNKNOWN;
    }
    if (pInParams->adaptiveRef && !(availableFeaures & ENC_FEATURE_ADAPTIVE_REF)) {
        print_feature_warnings(RGY_LOG_WARN, _T("adaptiveRef"));
        pInParams->adaptiveRef.reset();
    }
    if (pInParams->adaptiveLTR && !(availableFeaures & ENC_FEATURE_ADAPTIVE_LTR)) {
        print_feature_warnings(RGY_LOG_WARN, _T("AdaptiveLTR"));
        pInParams->adaptiveLTR.reset();
    }
    if (pInParams->adaptiveCQM && !(availableFeaures & ENC_FEATURE_ADAPTIVE_CQM)) {
        print_feature_warnings(RGY_LOG_WARN, _T("AdaptiveCQM"));
        pInParams->adaptiveCQM.reset();
    }
    if (pInParams->bMBBRC && !(availableFeaures & ENC_FEATURE_MBBRC)) {
        print_feature_warnings(RGY_LOG_WARN, _T("MBBRC"));
        pInParams->bMBBRC.reset();
    }
    if (   (MFX_RATECONTROL_LA     == pInParams->rcParam.encMode
         || MFX_RATECONTROL_LA_ICQ == pInParams->rcParam.encMode)
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
    if (pInParams->codec == RGY_CODEC_H264
        && ((m_encPicstruct & RGY_PICSTRUCT_INTERLACED) != 0)
        && pInParams->GopRefDist > 1 //Bframes > 0
        && m_device->CPUGen() == CPU_GEN_HASWELL
        && m_device->memType() == D3D11_MEMORY) {
        PrintMes(RGY_LOG_WARN, _T("H.264 interlaced encoding with B frames on d3d11 mode results fuzzy outputs on Haswell CPUs.\n"));
        PrintMes(RGY_LOG_WARN, _T("B frames will be disabled.\n"));
        pInParams->GopRefDist = 1; //Bframes = 0
    }
    //最近のドライバでは問題ない模様
    //if (pInParams->nBframes > 2 && pInParams->CodecId == MFX_CODEC_HEVC) {
    //    PrintMes(RGY_LOG_WARN, _T("HEVC encoding + B-frames > 2 might cause artifacts, please check the output.\n"));
    //}
    if (!pInParams->bBPyramid.value_or(true) && pInParams->GopRefDist-1 >= 10 && !(availableFeaures & ENC_FEATURE_B_PYRAMID_MANY_BFRAMES)) {
        if (pInParams->bBPyramid.value_or(false)) {
            PrintMes(RGY_LOG_WARN, _T("B pyramid with too many bframes is not supported on current platform, B pyramid disabled.\n"));
        }
        pInParams->bBPyramid = false;
    }
    if (!pInParams->bBPyramid.value_or(true) && m_device->CPUGen() < CPU_GEN_HASWELL) {
        if (pInParams->bBPyramid.value_or(false)) {
            PrintMes(RGY_LOG_WARN, _T("B pyramid on IvyBridge generation might cause artifacts, please check your encoded video.\n"));
        }
    }
    if (pInParams->bNoDeblock && !(availableFeaures & ENC_FEATURE_NO_DEBLOCK)) {
        print_feature_warnings(RGY_LOG_WARN, _T("No deblock"));
        pInParams->bNoDeblock = false;
    }
    if (pInParams->intraRefreshCycle > 0 && !(availableFeaures & ENC_FEATURE_INTRA_REFRESH)) {
        print_feature_warnings(RGY_LOG_WARN, _T("Intra Refresh"));
        pInParams->intraRefreshCycle = 0;
    }
    if ((pInParams->qpMin != RGYQPSet() || pInParams->qpMax != RGYQPSet()) && !(availableFeaures & ENC_FEATURE_QP_MINMAX)) {
        print_feature_warnings(RGY_LOG_WARN, _T("Min/Max QP"));
        pInParams->qpMin = RGYQPSet();
        pInParams->qpMax = RGYQPSet();
    }
    if (0 != pInParams->nWinBRCSize) {
        if (!(availableFeaures & ENC_FEATURE_WINBRC)) {
            print_feature_warnings(RGY_LOG_WARN, _T("WinBRC"));
            pInParams->nWinBRCSize = 0;
        } else if (0 == pInParams->rcParam.maxBitrate) {
            print_feature_warnings(RGY_LOG_WARN, _T("Min/Max QP"));
            PrintMes(RGY_LOG_WARN, _T("WinBRC requires Max bitrate to be set, disabled.\n"));
            pInParams->nWinBRCSize = 0;
        }
    }
    if (pInParams->bDirectBiasAdjust && !(availableFeaures & ENC_FEATURE_DIRECT_BIAS_ADJUST)) {
        print_feature_warnings(RGY_LOG_WARN, _T("Direct Bias Adjust"));
        pInParams->bDirectBiasAdjust.reset();
    }
    if (pInParams->bGlobalMotionAdjust && !(availableFeaures & ENC_FEATURE_GLOBAL_MOTION_ADJUST)) {
        print_feature_warnings(RGY_LOG_WARN, _T("MV Cost Scaling"));
        pInParams->bGlobalMotionAdjust = 0;
        pInParams->nMVCostScaling = 0;
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
    if (pInParams->nFadeDetect.value_or(false)) {
        PrintMes(RGY_LOG_WARN, _T("fade-detect will be disabled due to instability.\n"));
        pInParams->nFadeDetect = MFX_CODINGOPTION_OFF;
    }
#endif
    if (pInParams->nFadeDetect.has_value() && !(availableFeaures & ENC_FEATURE_FADE_DETECT)) {
        if (pInParams->nFadeDetect.value_or(false)) {
            print_feature_warnings(RGY_LOG_WARN, _T("FadeDetect"));
        }
        pInParams->nFadeDetect.reset();
    }
    if (pInParams->codec == RGY_CODEC_HEVC) {
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
        if (pInParams->hevc_gpb.has_value() && !(availableFeaures & ENC_FEATURE_DISABLE_GPB)) {
            print_feature_warnings(RGY_LOG_WARN, _T("HEVC GPB"));
            pInParams->hevc_gpb.reset();
        }
    }
    if (pInParams->codec == RGY_CODEC_VP9) {
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_15)) {
            if (pInParams->functionMode != QSVFunctionMode::FF) {
                PrintMes(RGY_LOG_WARN, _T("Switched to fixed function (FF) mode, as VP9 encoding requires FF mode.\n"));
                pInParams->functionMode = QSVFunctionMode::FF;
            }
        } else {
            PrintMes(RGY_LOG_ERROR, _T("VP9 encoding not supported on this platform.\n"));
            return RGY_ERR_UNSUPPORTED;
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
    m_encVUI.setDescriptPreset();

    if (pInParams->bOutputAud && !(availableFeaures & ENC_FEATURE_AUD)) {
        print_feature_warnings(RGY_LOG_WARN, _T("aud"));
        pInParams->bOutputAud = false;
    }
    if (pInParams->bOutputPicStruct && !(availableFeaures & ENC_FEATURE_PIC_STRUCT)) {
        print_feature_warnings(RGY_LOG_WARN, _T("pic-strcut"));
        pInParams->bOutputPicStruct = false;
    }

    //profileを守るための調整
    if (pInParams->codec == RGY_CODEC_H264) {
        if (pInParams->CodecProfile == MFX_PROFILE_AVC_BASELINE) {
            pInParams->GopRefDist = 1; //Bframe=0
            pInParams->bCAVLC = true;
        }
        if (pInParams->bCAVLC) {
            pInParams->bRDO = false;
        }
    }

    CHECK_RANGE_LIST(pInParams->codec,        list_codec_rgy,   "codec");
    CHECK_RANGE_LIST(pInParams->CodecLevel,   get_level_list(pInParams->codec),   "level");
    CHECK_RANGE_LIST(pInParams->CodecProfile, get_profile_list(pInParams->codec), "profile");
    CHECK_RANGE_LIST(pInParams->rcParam.encMode,     list_rc_mode, "rc mode");

    //設定開始
    m_encParams.videoPrm.mfx.CodecId                 = codec_rgy_to_enc(pInParams->codec);
    m_encParams.videoPrm.mfx.RateControlMethod       = (mfxU16)pInParams->rcParam.encMode;
    if (MFX_RATECONTROL_CQP == m_encParams.videoPrm.mfx.RateControlMethod) {
        //CQP
        m_encParams.videoPrm.mfx.QPI             = (mfxU16)clamp_param_int(pInParams->rcParam.qp.qpI, 0, codecMaxQP, _T("qp-i"));
        m_encParams.videoPrm.mfx.QPP             = (mfxU16)clamp_param_int(pInParams->rcParam.qp.qpP, 0, codecMaxQP, _T("qp-p"));
        m_encParams.videoPrm.mfx.QPB             = (mfxU16)clamp_param_int(pInParams->rcParam.qp.qpB, 0, codecMaxQP, _T("qp-b"));
    } else if (MFX_RATECONTROL_ICQ    == m_encParams.videoPrm.mfx.RateControlMethod
            || MFX_RATECONTROL_LA_ICQ == m_encParams.videoPrm.mfx.RateControlMethod) {
        m_encParams.videoPrm.mfx.ICQQuality      = (mfxU16)clamp_param_int(pInParams->rcParam.icqQuality, 1, codecMaxQP, _T("icq"));
        m_encParams.videoPrm.mfx.MaxKbps         = 0;
    } else {
        auto maxBitrate = (std::max)((std::max)(pInParams->rcParam.bitrate, pInParams->rcParam.maxBitrate),
            pInParams->rcParam.vbvBufSize / 8 /*これはbyte単位の指定*/);
        if (maxBitrate > USHRT_MAX) {
            m_encParams.videoPrm.mfx.BRCParamMultiplier = (mfxU16)(maxBitrate / USHRT_MAX) + 1;
            pInParams->rcParam.bitrate    /= m_encParams.videoPrm.mfx.BRCParamMultiplier;
            pInParams->rcParam.maxBitrate /= m_encParams.videoPrm.mfx.BRCParamMultiplier;
            pInParams->rcParam.vbvBufSize /= m_encParams.videoPrm.mfx.BRCParamMultiplier;
        }
        m_encParams.videoPrm.mfx.TargetKbps      = (mfxU16)pInParams->rcParam.bitrate; // in kbps
        if (m_encParams.videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_AVBR) {
            //AVBR
            //m_encParams.videoPrm.mfx.Accuracy        = pInParams->nAVBRAccuarcy;
            m_encParams.videoPrm.mfx.Accuracy        = 500;
            m_encParams.videoPrm.mfx.Convergence     = (mfxU16)pInParams->rcParam.avbrConvergence;
        } else {
            //CBR, VBR
            m_encParams.videoPrm.mfx.MaxKbps         = (mfxU16)pInParams->rcParam.maxBitrate;
            m_encParams.videoPrm.mfx.BufferSizeInKB  = (mfxU16)(pInParams->rcParam.vbvBufSize / 8); //これはbyte単位の指定
            m_encParams.videoPrm.mfx.InitialDelayInKB = m_encParams.videoPrm.mfx.BufferSizeInKB / 2;
        }
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_15)) {
        m_encParams.videoPrm.mfx.LowPower = (mfxU16)((pInParams->functionMode == QSVFunctionMode::FF) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF);
    }
    m_encParams.videoPrm.mfx.TargetUsage             = (mfxU16)clamp_param_int(pInParams->nTargetUsage, MFX_TARGETUSAGE_BEST_QUALITY, MFX_TARGETUSAGE_BEST_SPEED, _T("quality")); // trade-off between quality and speed

    PrintMes(RGY_LOG_DEBUG, _T("InitMfxEncParams: Output FPS %d/%d\n"), m_encFps.n(), m_encFps.d());
    if (pInParams->nGOPLength == 0) {
        pInParams->nGOPLength = (mfxU16)((m_encFps.n() + m_encFps.d() - 1) / m_encFps.d()) * 10;
        PrintMes(RGY_LOG_DEBUG, _T("InitMfxEncParams: Auto GOP Length: %d\n"), pInParams->nGOPLength);
    }
    m_encParams.videoPrm.mfx.FrameInfo.FrameRateExtN = m_encFps.n();
    m_encParams.videoPrm.mfx.FrameInfo.FrameRateExtD = m_encFps.d();
    m_encParams.videoPrm.mfx.EncodedOrder            = 0;
    m_encParams.videoPrm.mfx.NumSlice                = (mfxU16)pInParams->nSlices;

    m_encParams.videoPrm.mfx.NumRefFrame             = (mfxU16)clamp_param_int(pInParams->nRef, 0, 16, _T("ref"));
    m_encParams.videoPrm.mfx.CodecLevel              = (mfxU16)pInParams->CodecLevel;
    m_encParams.videoPrm.mfx.CodecProfile            = (mfxU16)pInParams->CodecProfile;
    m_encParams.videoPrm.mfx.GopOptFlag              = 0;
    m_encParams.videoPrm.mfx.GopOptFlag             |= (!pInParams->bopenGOP) ? MFX_GOP_CLOSED : 0x00;

    /* For H.264, IdrInterval specifies IDR-frame interval in terms of I-frames; if IdrInterval = 0, then every I-frame is an IDR-frame. If IdrInterval = 1, then every other I-frame is an IDR-frame, etc.
     * For HEVC, if IdrInterval = 0, then only first I-frame is an IDR-frame. If IdrInterval = 1, then every I-frame is an IDR-frame. If IdrInterval = 2, then every other I-frame is an IDR-frame, etc.
     * For MPEG2, IdrInterval defines sequence header interval in terms of I-frames. If IdrInterval = N, SDK inserts the sequence header before every Nth I-frame. If IdrInterval = 0 (default), SDK inserts the sequence header once at the beginning of the stream.
     * If GopPicSize or GopRefDist is zero, IdrInterval is undefined. */
    if (pInParams->codec == RGY_CODEC_HEVC) {
        m_encParams.videoPrm.mfx.IdrInterval = (mfxU16)((!pInParams->bopenGOP) ? 1 : 1 + ((m_encFps.n() + m_encFps.d() - 1) / m_encFps.d()) * 600 / pInParams->nGOPLength);
    } else if (pInParams->codec == RGY_CODEC_H264) {
        m_encParams.videoPrm.mfx.IdrInterval = (mfxU16)((!pInParams->bopenGOP) ? 0 : ((m_encFps.n() + m_encFps.d() - 1) / m_encFps.d()) * 600 / pInParams->nGOPLength);
    } else {
        m_encParams.videoPrm.mfx.IdrInterval = 0;
    }
    //MFX_GOP_STRICTにより、インタレ保持時にフレームが壊れる場合があるため、無効とする
    //m_encParams.videoPrm.mfx.GopOptFlag             |= (pInParams->bforceGOPSettings) ? MFX_GOP_STRICT : NULL;

    m_encParams.videoPrm.mfx.GopPicSize              = (mfxU16)pInParams->nGOPLength;
    if (gopRefDistAsBframes) {
        m_encParams.videoPrm.mfx.GopRefDist = (mfxU16)(clamp_param_int(pInParams->GopRefDist-1, 0, 16, _T("bframes"))+1);
    } else {
        m_encParams.videoPrm.mfx.GopRefDist = (mfxU16)clamp_param_int(pInParams->GopRefDist, 1, 33, _T("GopRefDist"));
    }

    // specify memory type
    m_encParams.videoPrm.IOPattern = (mfxU16)((pInParams->memType != SYSTEM_MEMORY) ? MFX_IOPATTERN_IN_VIDEO_MEMORY : MFX_IOPATTERN_IN_SYSTEM_MEMORY);

    // frame info parameters
    m_encParams.videoPrm.mfx.FrameInfo.ChromaFormat = mfx_fourcc_to_chromafmt(csp_rgy_to_enc(getEncoderCsp(pInParams)));
    m_encParams.videoPrm.mfx.FrameInfo.PicStruct    = (mfxU16)picstruct_rgy_to_enc(m_encPicstruct);

    // set sar info
    auto par = std::make_pair(pInParams->nPAR[0], pInParams->nPAR[1]);
    if ((!pInParams->nPAR[0] || !pInParams->nPAR[1]) //SAR比の指定がない
        && pInParams->input.sar[0] && pInParams->input.sar[1] //入力側からSAR比を取得ずみ
        && (m_encWidth == pInParams->input.srcWidth && m_encHeight == pInParams->input.srcHeight)) {//リサイズは行われない
        par = std::make_pair(pInParams->input.sar[0], pInParams->input.sar[1]);
    }
    adjust_sar(&par.first, &par.second, m_encWidth, m_encHeight);
    m_encParams.videoPrm.mfx.FrameInfo.AspectRatioW = (mfxU16)par.first;
    m_encParams.videoPrm.mfx.FrameInfo.AspectRatioH = (mfxU16)par.second;

    RGY_MEMSET_ZERO(m_encParams.cop);
    m_encParams.cop.Header.BufferId = MFX_EXTBUFF_CODING_OPTION;
    m_encParams.cop.Header.BufferSz = sizeof(mfxExtCodingOption);
    //if (!pInParams->bUseHWLib) {
    //    //swライブラリ使用時のみ
    //    m_encParams.cop.InterPredBlockSize = pInParams->nInterPred;
    //    m_encParams.cop.IntraPredBlockSize = pInParams->nIntraPred;
    //    m_encParams.cop.MVSearchWindow     = pInParams->MVSearchWindow;
    //    m_encParams.cop.MVPrecision        = pInParams->nMVPrecision;
    //}
    //if (!pInParams->bUseHWLib || pInParams->CodecProfile == MFX_PROFILE_AVC_BASELINE) {
    //    //swライブラリ使用時かbaselineを指定した時
    //    m_encParams.cop.RateDistortionOpt  = (mfxU16)((pInParams->bRDO) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_UNKNOWN);
    //    m_encParams.cop.CAVLC              = (mfxU16)((pInParams->bCAVLC) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_UNKNOWN);
    //}
    //m_encParams.cop.FramePicture = MFX_CODINGOPTION_ON;
    //m_encParams.cop.FieldOutput = MFX_CODINGOPTION_ON;
    //m_encParams.cop.VuiVclHrdParameters = MFX_CODINGOPTION_ON;
    //m_encParams.cop.VuiNalHrdParameters = MFX_CODINGOPTION_ON;
    m_encParams.cop.AUDelimiter = (mfxU16)((pInParams->bOutputAud) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF);
    m_encParams.cop.PicTimingSEI = (mfxU16)((pInParams->bOutputPicStruct) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF);
    m_encParams.cop.SingleSeiNalUnit = MFX_CODINGOPTION_OFF;

    const auto VBR_RC_LIST = make_array<int>(MFX_RATECONTROL_CBR, MFX_RATECONTROL_VBR, MFX_RATECONTROL_AVBR, MFX_RATECONTROL_VCM, MFX_RATECONTROL_QVBR, MFX_RATECONTROL_LA, MFX_RATECONTROL_LA_HRD);
    const auto DOVI_RC_LIST = make_array<int>(MFX_RATECONTROL_CBR, MFX_RATECONTROL_VBR, MFX_RATECONTROL_AVBR, MFX_RATECONTROL_VCM, MFX_RATECONTROL_QVBR);
    if (auto profile = getDOVIProfile(pInParams->common.doviProfile); profile != nullptr && profile->HRDSEI) {
        if (std::find(DOVI_RC_LIST.begin(), DOVI_RC_LIST.end(), pInParams->rcParam.encMode) != DOVI_RC_LIST.end()) {
            if (m_encParams.videoPrm.mfx.BufferSizeInKB == 0) {
                m_encParams.videoPrm.mfx.BufferSizeInKB = m_encParams.videoPrm.mfx.MaxKbps / 8;
            }
            if (m_encParams.videoPrm.mfx.InitialDelayInKB == 0) {
                m_encParams.videoPrm.mfx.InitialDelayInKB = m_encParams.videoPrm.mfx.BufferSizeInKB / 2;
            }
        }
    }


    //API v1.6の機能
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_6)
        && (availableFeaures & ENC_FEATURE_EXT_COP2)) {
        INIT_MFX_EXT_BUFFER(m_encParams.cop2, MFX_EXTBUFF_CODING_OPTION2);
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)) {
            m_encParams.cop2.AdaptiveI   = get_codingopt(pInParams->bAdaptiveI);
            m_encParams.cop2.AdaptiveB   = get_codingopt(pInParams->bAdaptiveB);
            m_encParams.cop2.BRefType    = (decltype(m_encParams.cop2.BRefType))get3state(pInParams->bBPyramid, MFX_B_REF_UNKNOWN, MFX_B_REF_PYRAMID, MFX_B_REF_OFF);

            CHECK_RANGE_LIST(pInParams->nLookaheadDS, list_lookahead_ds, "la-quality");
            m_encParams.cop2.LookAheadDS = (mfxU16)pInParams->nLookaheadDS;
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_7)) {
            m_encParams.cop2.LookAheadDepth = (mfxU16)((pInParams->nLookaheadDepth == 0) ? pInParams->nLookaheadDepth : clamp_param_int(pInParams->nLookaheadDepth, QSV_LOOKAHEAD_DEPTH_MIN, QSV_LOOKAHEAD_DEPTH_MAX, _T("la-depth")));

            CHECK_RANGE_LIST(pInParams->nTrellis, list_avc_trellis_for_options, "trellis");
            m_encParams.cop2.Trellis = (mfxU16)pInParams->nTrellis;
        }
        m_encParams.cop2.MBBRC = get_codingopt(pInParams->bMBBRC);
        m_encParams.cop2.ExtBRC = get_codingopt(pInParams->extBRC);
        if (pInParams->intraRefreshCycle > 0) {
            m_encParams.cop2.IntRefType = MFX_REFRESH_VERTICAL;
            if (pInParams->intraRefreshCycle == 1) {
                PrintMes(RGY_LOG_ERROR, _T("--intra-refresh-cycle must be 2 or larger.\n"));
                return RGY_ERR_UNSUPPORTED;
            }
            m_encParams.cop2.IntRefCycleSize = (mfxU16)pInParams->intraRefreshCycle;
        }
        if (pInParams->bNoDeblock) {
            m_encParams.cop2.DisableDeblockingIdc = MFX_CODINGOPTION_ON;
        }
        if (pInParams->maxFrameSize) {
            m_encParams.cop2.MaxFrameSize = (decltype(m_encParams.cop2.MaxFrameSize))pInParams->maxFrameSize;
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)) {
            if (m_hdr10plus || m_hdr10plusMetadataCopy || m_dovirpu || pInParams->common.doviProfile != 0 || (m_hdrseiOut && m_hdrseiOut->gen_nal().size() > 0) || m_parallelEnc) {
                m_encParams.cop2.RepeatPPS = MFX_CODINGOPTION_ON;
            } else if (pInParams->repeatHeaders.has_value()) {
                m_encParams.cop2.RepeatPPS = (mfxU16)((pInParams->repeatHeaders.value()) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF);
            } else {
                m_encParams.cop2.RepeatPPS = MFX_CODINGOPTION_UNKNOWN;
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_10)) {
            m_encParams.cop2.BufferingPeriodSEI = (mfxU16)((pInParams->bufPeriodSEI) ? MFX_BPSEI_IFRAME : MFX_BPSEI_DEFAULT);
        }
        for (int i = 0; i < 3; i++) {
            pInParams->qpMin.qp(i) = clamp_param_int(pInParams->qpMin.qp(i), 0, codecMaxQP, _T("qp min"));
            pInParams->qpMax.qp(i) = clamp_param_int(pInParams->qpMax.qp(i), 0, codecMaxQP, _T("qp max"));
            int qpMin = pInParams->qpMin.qp(i);
            int qpMax = pInParams->qpMax.qp(i);
            if (pInParams->qpMin.qp(i) > 0 && pInParams->qpMax.qp(i) > 0) {
                qpMin = (std::min)(qpMin, pInParams->qpMax.qp(i));
                qpMax = (std::max)(qpMax, pInParams->qpMin.qp(i));
            }
            pInParams->qpMin.qp(i) = (pInParams->qpMin.qp(i) == 0) ? 0 : qpMin;
            pInParams->qpMax.qp(i) = (pInParams->qpMax.qp(i) == 0) ? 0 : qpMax;
        }
        m_encParams.cop2.MaxQPI = (mfxU8)pInParams->qpMax.qpI;
        m_encParams.cop2.MaxQPP = (mfxU8)pInParams->qpMax.qpP;
        m_encParams.cop2.MaxQPB = (mfxU8)pInParams->qpMax.qpB;
        m_encParams.cop2.MinQPI = (mfxU8)pInParams->qpMin.qpI;
        m_encParams.cop2.MinQPP = (mfxU8)pInParams->qpMin.qpP;
        m_encParams.cop2.MinQPB = (mfxU8)pInParams->qpMin.qpB;
        m_encParams.addExtParams(&m_encParams.cop2);
    }

    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)) {
        if (m_encParams.videoPrm.mfx.CodecId == MFX_CODEC_HEVC) {
            if (pInParams->hevc_tier != 0) {
                m_encParams.videoPrm.mfx.CodecLevel |= (mfxU16)pInParams->hevc_tier;
            }
        }
    }

    //API v1.11の機能
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_11)
        && (availableFeaures & ENC_FEATURE_EXT_COP3)) {
        INIT_MFX_EXT_BUFFER(m_encParams.cop3, MFX_EXTBUFF_CODING_OPTION3);
        if (MFX_RATECONTROL_QVBR == m_encParams.videoPrm.mfx.RateControlMethod) {
            m_encParams.cop3.QVBRQuality = (mfxU16)clamp_param_int(pInParams->rcParam.qvbrQuality, 1, codecMaxQP, _T("qvbr-q"));
        }
        //WinBRCの対象のレート制御モードかどうかをチェックする
        //これを行わないとInvalid Parametersとなる場合がある
        //なお、WinBRCを有効にすると、ビットレートが著しく低下したままになってしまう場合がある
        //なので、オプションを指定しない限り有効にならないようにする
        static const auto WinBRCTargetRC = make_array<int>(MFX_RATECONTROL_VBR, MFX_RATECONTROL_LA, MFX_RATECONTROL_LA_HRD, MFX_RATECONTROL_QVBR);
        if (pInParams->nWinBRCSize != 0
            && std::find(WinBRCTargetRC.begin(), WinBRCTargetRC.end(), pInParams->rcParam.encMode) != WinBRCTargetRC.end()
            && pInParams->rcParam.maxBitrate != 0
            && !pInParams->extBRC) { // extbrcはWinBRCと併用できない模様
            m_encParams.cop3.WinBRCSize = (mfxU16)pInParams->nWinBRCSize;
            m_encParams.cop3.WinBRCMaxAvgKbps = (mfxU16)pInParams->rcParam.maxBitrate;
        }

        //API v1.13の機能
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_13)) {
            m_encParams.cop3.DirectBiasAdjustment = get_codingopt(pInParams->bDirectBiasAdjust);
            if (pInParams->bGlobalMotionAdjust) {
                m_encParams.cop3.GlobalMotionBiasAdjustment = MFX_CODINGOPTION_ON;
                CHECK_RANGE_LIST(pInParams->nMVCostScaling, list_mv_cost_scaling, "mv-scaling");
                m_encParams.cop3.MVCostScalingFactor    = (mfxU16)pInParams->nMVCostScaling;
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_16)) {
            m_encParams.cop3.WeightedBiPred = (mfxU16)pInParams->nWeightB;
            m_encParams.cop3.WeightedPred   = (mfxU16)pInParams->nWeightP;
            m_encParams.cop3.ScenarioInfo   = (mfxU16)(pInParams->scenarioInfo);
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_17)) {
            m_encParams.cop3.FadeDetection = get_codingopt(pInParams->nFadeDetect);
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_18)) {
            if (m_encParams.videoPrm.mfx.CodecId == MFX_CODEC_HEVC) {
                m_encParams.cop3.GPB = get_codingopt(pInParams->hevc_gpb);
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_19)) {
            if (bQPOffsetUsed) {
                m_encParams.cop3.EnableQPOffset = MFX_CODINGOPTION_ON;
                memcpy(m_encParams.cop3.QPOffset, pInParams->pQPOffset, sizeof(pInParams->pQPOffset));
            }
            if (pInParams->maxFrameSize || pInParams->maxFrameSizeI) {
                m_encParams.cop3.MaxFrameSizeI = (decltype(m_encParams.cop3.MaxFrameSizeI))((pInParams->maxFrameSizeI) ? pInParams->maxFrameSizeI : pInParams->maxFrameSize);
            }
            if (pInParams->maxFrameSize || pInParams->maxFrameSizeP) {
                m_encParams.cop3.MaxFrameSizeP = (decltype(m_encParams.cop3.MaxFrameSizeP))((pInParams->maxFrameSizeP) ? pInParams->maxFrameSizeP : pInParams->maxFrameSize);
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_23)) {
            m_encParams.cop3.RepartitionCheckEnable = get_codingopt(pInParams->nRepartitionCheck);
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_26)) {
            if (m_encParams.videoPrm.mfx.CodecId == MFX_CODEC_HEVC) {
                m_encParams.cop3.TransformSkip = (mfxU16)pInParams->hevc_tskip;
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_2_2)) {
            m_encParams.cop3.AdaptiveCQM = get_codingopt(pInParams->adaptiveCQM);
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_2_4)) {
            m_encParams.cop3.AdaptiveRef = get_codingopt(pInParams->adaptiveRef);
            m_encParams.cop3.AdaptiveLTR = get_codingopt(pInParams->adaptiveLTR);
        }
        m_encParams.addExtParams(&m_encParams.cop3);
    }


    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_2_9)
        && pInParams->tuneQuality != MFX_ENCODE_TUNE_OFF
        && (availableFeaures & ENC_FEATURE_EXT_TUNE_ENC_QUALITY)) {
        INIT_MFX_EXT_BUFFER(m_encParams.tuneEncQualityPrm, MFX_EXTBUFF_TUNE_ENCODE_QUALITY);
        m_encParams.tuneEncQualityPrm.TuneQuality = (decltype(m_encParams.tuneEncQualityPrm.TuneQuality))(pInParams->tuneQuality);
        m_encParams.addExtParams(&m_encParams.tuneEncQualityPrm);
    }

    //Bluray互換出力
    if (pInParams->nBluray) {
        if (   m_encParams.videoPrm.mfx.RateControlMethod != MFX_RATECONTROL_CBR
            && m_encParams.videoPrm.mfx.RateControlMethod != MFX_RATECONTROL_VBR
            && m_encParams.videoPrm.mfx.RateControlMethod != MFX_RATECONTROL_LA
            && m_encParams.videoPrm.mfx.RateControlMethod != MFX_RATECONTROL_LA_HRD) {
                if (pInParams->nBluray == 1) {
                    PrintMes(RGY_LOG_ERROR, _T("")
                        _T("Current encode mode (%s) is not preferred for Bluray encoding,\n")
                        _T("since it cannot set Max Bitrate.\n")
                        _T("Please consider using Lookahead/VBR/CBR mode for Bluray encoding.\n"), EncmodeToStr(m_encParams.videoPrm.mfx.RateControlMethod));
                    return RGY_ERR_INCOMPATIBLE_VIDEO_PARAM;
                } else {
                    //pInParams->nBluray == 2 -> force Bluray
                    PrintMes(RGY_LOG_WARN, _T("")
                        _T("Current encode mode (%s) is not preferred for Bluray encoding,\n")
                        _T("since it cannot set Max Bitrate.\n")
                        _T("This output might not be able to be played on a Bluray Player.\n")
                        _T("Please consider using Lookahead/VBR/CBR mode for Bluray encoding.\n"), EncmodeToStr(m_encParams.videoPrm.mfx.RateControlMethod));
                }
        }
        if (   m_encParams.videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_CBR
            || m_encParams.videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_VBR
            || m_encParams.videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_LA
            || m_encParams.videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_LA_HRD) {
                m_encParams.videoPrm.mfx.MaxKbps    = (std::min)(m_encParams.videoPrm.mfx.MaxKbps, (uint16_t)40000);
                m_encParams.videoPrm.mfx.TargetKbps = (std::min)(m_encParams.videoPrm.mfx.TargetKbps, m_encParams.videoPrm.mfx.MaxKbps);
                if (m_encParams.videoPrm.mfx.BufferSizeInKB == 0) {
                    m_encParams.videoPrm.mfx.BufferSizeInKB = m_encParams.videoPrm.mfx.MaxKbps / 8;
                }
                if (m_encParams.videoPrm.mfx.InitialDelayInKB == 0) {
                    m_encParams.videoPrm.mfx.InitialDelayInKB = m_encParams.videoPrm.mfx.BufferSizeInKB / 2;
                }
        } else {
            m_encParams.videoPrm.mfx.BufferSizeInKB = 25000 / 8;
        }
        m_encParams.videoPrm.mfx.CodecLevel = (m_encParams.videoPrm.mfx.CodecLevel == 0) ? MFX_LEVEL_AVC_41 : ((std::min)(m_encParams.videoPrm.mfx.CodecLevel, (uint16_t)MFX_LEVEL_AVC_41));
        m_encParams.videoPrm.mfx.NumSlice   = (std::max)(m_encParams.videoPrm.mfx.NumSlice, (uint16_t)4);
        m_encParams.videoPrm.mfx.GopOptFlag &= (~MFX_GOP_STRICT);
        m_encParams.videoPrm.mfx.GopRefDist = (std::min)(m_encParams.videoPrm.mfx.GopRefDist, (uint16_t)(3+1));
        m_encParams.videoPrm.mfx.GopPicSize = (int)((std::min)(m_encParams.videoPrm.mfx.GopPicSize, (uint16_t)30) / m_encParams.videoPrm.mfx.GopRefDist) * m_encParams.videoPrm.mfx.GopRefDist;
        m_encParams.videoPrm.mfx.NumRefFrame = (std::min)(m_encParams.videoPrm.mfx.NumRefFrame, (uint16_t)6);
        m_encParams.cop.MaxDecFrameBuffering = m_encParams.videoPrm.mfx.NumRefFrame;
        m_encParams.cop.VuiNalHrdParameters = MFX_CODINGOPTION_ON;
        m_encParams.cop.VuiVclHrdParameters = MFX_CODINGOPTION_ON;
        m_encParams.cop.AUDelimiter  = MFX_CODINGOPTION_ON;
        m_encParams.cop.PicTimingSEI = MFX_CODINGOPTION_ON;
        m_encParams.cop.ResetRefList = MFX_CODINGOPTION_ON;
        //m_encParams.cop.EndOfSequence = MFX_CODINGOPTION_ON; //hwモードでは効果なし 0x00, 0x00, 0x01, 0x0a
        //m_encParams.cop.EndOfStream   = MFX_CODINGOPTION_ON; //hwモードでは効果なし 0x00, 0x00, 0x01, 0x0b
        PrintMes(RGY_LOG_DEBUG, _T("InitMfxEncParams: Adjusted param for Bluray encoding.\n"));
    }
    if (availableFeaures & ENC_FEATURE_EXT_COP) { // VP9ではmfxExtCodingOptionはチェックしないようにしないと正常に動作しない
        m_encParams.addExtParams(&m_encParams.cop);
    }

    //m_encParams.videoPrm.mfx.TimeStampCalc = MFX_TIMESTAMPCALC_UNKNOWN;
    //m_encParams.videoPrm.mfx.TimeStampCalc = (mfxU16)((pInParams->vpp.nDeinterlace == MFX_DEINTERLACE_IT) ? MFX_TIMESTAMPCALC_TELECINE : MFX_TIMESTAMPCALC_UNKNOWN);
    //m_encParams.videoPrm.mfx.ExtendedPicStruct = pInParams->nPicStruct;

    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_3)
        && (availableFeaures & ENC_FEATURE_EXT_VIDEO_SIGNAL_INFO) &&
        (m_encVUI.format    != get_cx_value(list_videoformat, _T("undef")) ||
         m_encVUI.colorprim != get_cx_value(list_colorprim, _T("undef")) ||
         m_encVUI.transfer  != get_cx_value(list_transfer, _T("undef")) ||
         m_encVUI.matrix    != get_cx_value(list_colormatrix, _T("undef")) ||
         m_encVUI.colorrange == RGY_COLORRANGE_FULL
        ) ) {
#define GET_COLOR_PRM(v, list) (mfxU16)((v == COLOR_VALUE_AUTO) ? ((m_encHeight >= HD_HEIGHT_THRESHOLD) ? list[HD_INDEX].value : list[SD_INDEX].value) : v)
            //色設定 (for API v1.3)
            CHECK_RANGE_LIST(m_encVUI.format,    list_videoformat, "videoformat");
            CHECK_RANGE_LIST(m_encVUI.colorprim, list_colorprim,   "colorprim");
            CHECK_RANGE_LIST(m_encVUI.transfer,  list_transfer,    "transfer");
            CHECK_RANGE_LIST(m_encVUI.matrix,    list_colormatrix, "colormatrix");

            INIT_MFX_EXT_BUFFER(m_encParams.videoSignalInfo, MFX_EXTBUFF_VIDEO_SIGNAL_INFO);
            m_encParams.videoSignalInfo.ColourDescriptionPresent = 1; //"1"と設定しないと正しく反映されない
            m_encParams.videoSignalInfo.VideoFormat              = (mfxU16)m_encVUI.format;
            m_encParams.videoSignalInfo.VideoFullRange           = m_encVUI.colorrange == RGY_COLORRANGE_FULL;
            m_encParams.videoSignalInfo.ColourPrimaries          = (mfxU16)m_encVUI.colorprim;
            m_encParams.videoSignalInfo.TransferCharacteristics  = (mfxU16)m_encVUI.transfer;
            m_encParams.videoSignalInfo.MatrixCoefficients       = (mfxU16)m_encVUI.matrix;
#undef GET_COLOR_PRM
            m_encParams.addExtParams(&m_encParams.videoSignalInfo);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_13)
        && m_encVUI.chromaloc != RGY_CHROMALOC_UNSPECIFIED
        && (availableFeaures & ENC_FEATURE_EXT_CHROMALOC)) {
        INIT_MFX_EXT_BUFFER(m_encParams.chromaLocInfo, MFX_EXTBUFF_CHROMA_LOC_INFO);
        m_encParams.chromaLocInfo.ChromaLocInfoPresentFlag = 1;
        m_encParams.chromaLocInfo.ChromaSampleLocTypeTopField = (mfxU16)(m_encVUI.chromaloc-1);
        m_encParams.chromaLocInfo.ChromaSampleLocTypeBottomField = (mfxU16)(m_encVUI.chromaloc-1);
        ////HWエンコーダではこれはサポートされていない模様なので無効化する
        //m_EncExtParams.push_back((mfxExtBuffer *)&m_encParams.chromaLocInfo);
    }

    const int encBitdepth = getEncoderBitdepth(pInParams);
    const auto encCsp = getEncoderCsp(pInParams);
    m_encParams.videoPrm.mfx.FrameInfo.FourCC = csp_rgy_to_enc(encCsp);
    m_encParams.videoPrm.mfx.FrameInfo.ChromaFormat = mfx_fourcc_to_chromafmt(m_encParams.videoPrm.mfx.FrameInfo.FourCC);
    m_encParams.videoPrm.mfx.FrameInfo.BitDepthLuma = (mfxU16)encBitdepth;
    m_encParams.videoPrm.mfx.FrameInfo.BitDepthChroma = (mfxU16)encBitdepth;
    m_encParams.videoPrm.mfx.FrameInfo.Shift = (cspShiftUsed(encCsp) && RGY_CSP_BIT_DEPTH[encCsp] - encBitdepth > 0) ? 1 : 0;
    m_encParams.videoPrm.mfx.FrameInfo.Width  = (mfxU16)ALIGN(m_encWidth, blocksz);
    m_encParams.videoPrm.mfx.FrameInfo.Height = (mfxU16)ALIGN(m_encHeight, blocksz * ((MFX_PICSTRUCT_PROGRESSIVE == m_encParams.videoPrm.mfx.FrameInfo.PicStruct) ? 1:2));

    m_encParams.videoPrm.mfx.FrameInfo.CropX = 0;
    m_encParams.videoPrm.mfx.FrameInfo.CropY = 0;
    m_encParams.videoPrm.mfx.FrameInfo.CropW = (mfxU16)m_encWidth;
    m_encParams.videoPrm.mfx.FrameInfo.CropH = (mfxU16)m_encHeight;

    if (m_encParams.videoPrm.mfx.FrameInfo.ChromaFormat == MFX_CHROMAFORMAT_YUV444) {
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_15)) {
            if (pInParams->functionMode != QSVFunctionMode::FF) {
                PrintMes(RGY_LOG_WARN, _T("Switched to fixed function (FF) mode, as encoding in YUV444/RGB requires FF mode.\n"));
                pInParams->functionMode = QSVFunctionMode::FF;
                m_encParams.videoPrm.mfx.LowPower = (mfxU16)MFX_CODINGOPTION_ON;
            }
            if (m_encParams.videoPrm.mfx.CodecId == MFX_CODEC_AVC) {
                PrintMes(RGY_LOG_ERROR, _T("Encoding in H.264 YUV444/RGB is not supported.\n"));
                return RGY_ERR_UNSUPPORTED;
            } else if (m_encParams.videoPrm.mfx.CodecId == MFX_CODEC_HEVC) {
                m_encParams.videoPrm.mfx.CodecProfile = MFX_PROFILE_HEVC_REXT;
            }
        } else {
            PrintMes(RGY_LOG_ERROR, _T("Encoding in YUV444/RGB is not supported on this platform.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }

    // In case of HEVC when height and/or width divided with 8 but not divided with 16
    // add extended parameter to increase performance
    if (m_encParams.videoPrm.mfx.CodecId == MFX_CODEC_HEVC
        && (availableFeaures & ENC_FEATURE_EXT_HEVC_PRM)) {
        INIT_MFX_EXT_BUFFER(m_encParams.hevcPrm, MFX_EXTBUFF_HEVC_PARAM);
        m_encParams.hevcPrm.PicWidthInLumaSamples = m_encParams.videoPrm.mfx.FrameInfo.CropW;
        m_encParams.hevcPrm.PicHeightInLumaSamples = m_encParams.videoPrm.mfx.FrameInfo.CropH;
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_26)) {
            m_encParams.hevcPrm.SampleAdaptiveOffset = (mfxU16)pInParams->hevc_sao;
            m_encParams.hevcPrm.LCUSize = (mfxU16)pInParams->hevc_ctu;
        }
        m_encParams.addExtParams(&m_encParams.hevcPrm);
    }

    if (m_encParams.videoPrm.mfx.CodecId == MFX_CODEC_VP8
        && (availableFeaures & ENC_FEATURE_EXT_COP_VP8)) {
        INIT_MFX_EXT_BUFFER(m_encParams.copVp8, MFX_EXTBUFF_VP8_CODING_OPTION);
        m_encParams.copVp8.SharpnessLevel = (mfxU16)clamp_param_int(pInParams->nVP8Sharpness, 0, 8, _T("sharpness"));
        m_encParams.addExtParams(&m_encParams.copVp8);
    }

    if (m_encParams.videoPrm.mfx.CodecId == MFX_CODEC_VP9
        && check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_26)
        && (availableFeaures & ENC_FEATURE_EXT_VP9_PRM)) {
        INIT_MFX_EXT_BUFFER(m_encParams.vp9Prm, MFX_EXTBUFF_VP9_PARAM);
        //m_encParams.vp9Prm.FrameWidth = m_encParams.videoPrm.mfx.FrameInfo.Width;
        //m_encParams.vp9Prm.FrameHeight = m_encParams.videoPrm.mfx.FrameInfo.Height;
        m_encParams.vp9Prm.WriteIVFHeaders = MFX_CODINGOPTION_OFF;
        //m_encParams.vp9Prm.NumTileColumns = 2;
        //m_encParams.vp9Prm.NumTileRows = 2;
        m_encParams.addExtParams(&m_encParams.vp9Prm);
    }

    if (m_encParams.videoPrm.mfx.CodecId == MFX_CODEC_AV1
        && check_lib_version(m_mfxVer, MFX_LIB_VERSION_2_5)) {
        if (availableFeaures & ENC_FEATURE_EXT_AV1_BITSTREAM_PRM) {
            INIT_MFX_EXT_BUFFER(m_encParams.av1BitstreamPrm, MFX_EXTBUFF_AV1_BITSTREAM_PARAM);
            //m_encParams.vp9Prm.FrameWidth = m_encParams.videoPrm.mfx.FrameInfo.Width;
            //m_encParams.vp9Prm.FrameHeight = m_encParams.videoPrm.mfx.FrameInfo.Height;
            m_encParams.av1BitstreamPrm.WriteIVFHeaders = MFX_CODINGOPTION_OFF;
            //m_encParams.vp9Prm.NumTileColumns = 2;
            //m_encParams.vp9Prm.NumTileRows = 2;
            m_encParams.addExtParams(&m_encParams.av1BitstreamPrm);
        }

        if (availableFeaures & ENC_FEATURE_EXT_AV1_RESOLUTION_PRM) {
            INIT_MFX_EXT_BUFFER(m_encParams.av1ResolutionPrm, MFX_EXTBUFF_AV1_RESOLUTION_PARAM);
            INIT_MFX_EXT_BUFFER(m_encParams.av1TilePrm, MFX_EXTBUFF_AV1_TILE_PARAM);
            //m_EncExtParams.push_back((mfxExtBuffer*)&m_encParams.av1ResolutionPrm);
        }

        if (availableFeaures & ENC_FEATURE_EXT_AV1_TILE_PRM) {
            if (pInParams->av1.tile_row != 0 || pInParams->av1.tile_col != 0) {
                m_encParams.av1TilePrm.NumTileRows = (mfxU16)std::max(pInParams->av1.tile_row, 1);
                m_encParams.av1TilePrm.NumTileColumns = (mfxU16)std::max(pInParams->av1.tile_col, 1);

                m_encParams.addExtParams(&m_encParams.av1TilePrm);
            }
        }
    }
    if (pInParams->hyperMode != MFX_HYPERMODE_OFF
        && (availableFeaures & ENC_FEATURE_EXT_HYPER_MODE)) {
        INIT_MFX_EXT_BUFFER(m_encParams.hyperModePrm, MFX_EXTBUFF_HYPER_MODE_PARAM);
        m_encParams.hyperModePrm.Mode = pInParams->hyperMode;
        m_encParams.addExtParams(&m_encParams.hyperModePrm);
    }

    m_encParams.setExtParams();
    for (const auto& extParam : m_encParams.buf) {
        PrintMes(RGY_LOG_DEBUG, _T("InitMfxEncParams: set ext param %s.\n"), fourccToStr(extParam->BufferId).c_str());
    }

    PrintMes(RGY_LOG_DEBUG, _T("InitMfxEncParams: enc input frame %dx%d (%d,%d,%d,%d)\n"),
        m_encParams.videoPrm.mfx.FrameInfo.Width, m_encParams.videoPrm.mfx.FrameInfo.Height,
        m_encParams.videoPrm.mfx.FrameInfo.CropX, m_encParams.videoPrm.mfx.FrameInfo.CropY, m_encParams.videoPrm.mfx.FrameInfo.CropW, m_encParams.videoPrm.mfx.FrameInfo.CropH);
    PrintMes(RGY_LOG_DEBUG, _T("InitMfxEncParams: enc input color format %s, chroma %s, bitdepth %d, shift %d, picstruct %s\n"),
        ColorFormatToStr(m_encParams.videoPrm.mfx.FrameInfo.FourCC), ChromaFormatToStr(m_encParams.videoPrm.mfx.FrameInfo.ChromaFormat),
        m_encParams.videoPrm.mfx.FrameInfo.BitDepthLuma, m_encParams.videoPrm.mfx.FrameInfo.Shift, MFXPicStructToStr(m_encParams.videoPrm.mfx.FrameInfo.PicStruct).c_str());
    PrintMes(RGY_LOG_DEBUG, _T("InitMfxEncParams: set all enc params.\n"));

    m_pmfxENC.reset(new MFXVideoENCODE(m_device->mfxSession()));
    if (!m_pmfxENC) {
        return RGY_ERR_MEMORY_ALLOC;
    }
    //のちの使用のために保存
    m_encFeatures = availableFeaures;
    return RGY_ERR_NONE;
}

bool CQSVPipeline::CPUGenOpenCLSupported(const QSV_CPU_GEN cpu_gen) {
    //SandyBridgeではOpenCLフィルタをサポートしない
    return cpu_gen != CPU_GEN_SANDYBRIDGE;
}

RGY_ERR CQSVPipeline::InitOpenCL(const bool enableOpenCL, const int openCLBuildThreads, const bool checkVppPerformance) {
    if (!enableOpenCL) {
        PrintMes(RGY_LOG_DEBUG, _T("OpenCL disabled.\n"));
        return RGY_ERR_NONE;
    }
    if (!CPUGenOpenCLSupported(m_device->CPUGen())) {
        PrintMes(RGY_LOG_DEBUG, _T("Skip OpenCL init as OpenCL is not supported in %s platform.\n"), CPU_GEN_STR[m_device->CPUGen()]);
        return RGY_ERR_NONE;
    }
    const mfxHandleType hdl_t = mfxHandleTypeFromMemType(m_device->memType(), true);
    mfxHDL hdl = nullptr;
    if (hdl_t) {
        auto sts = err_to_rgy(m_device->hwdev()->GetHandle((hdl_t == MFX_HANDLE_DIRECT3D_DEVICE_MANAGER9) ? (mfxHandleType)0 : hdl_t, &hdl));
        RGY_ERR(sts, _T("Failed to get HW device handle."));
        PrintMes(RGY_LOG_DEBUG, _T("Got HW device handle: %p.\n"), hdl);
    }

    RGYOpenCL cl(m_pQSVLog);
    if (!RGYOpenCL::openCLloaded()) {
        PrintMes(RGY_LOG_WARN, _T("Skip OpenCL init as OpenCL is not supported on this platform.\n"));
        return RGY_ERR_NONE;
    }
    auto platforms = cl.getPlatforms("Intel");
    if (platforms.size() == 0) {
        PrintMes(RGY_LOG_WARN, _T("Skip OpenCL init as OpenCL platforms not found.\n"));
        return RGY_ERR_NONE;
    }
    PrintMes(RGY_LOG_DEBUG, _T("Created Intel OpenCL platform.\n"));

    std::shared_ptr<RGYOpenCLPlatform> selectedPlatform;
    tstring clErrMessage;
    for (auto& platform : platforms) {
        if (m_device->memType() == D3D9_MEMORY && ENABLE_RGY_OPENCL_D3D9) {
            if (platform->createDeviceListD3D9(CL_DEVICE_TYPE_GPU, (void *)hdl) != CL_SUCCESS || platform->devs().size() == 0) {
                auto mes = strsprintf(_T("Failed to find d3d9 device in platform %s.\n"), char_to_tstring(platform->info().name).c_str());
                PrintMes(RGY_LOG_DEBUG, mes.c_str());
                clErrMessage += mes;
                continue;
            }
        } else if (m_device->memType() == D3D11_MEMORY && ENABLE_RGY_OPENCL_D3D11) {
            if (platform->createDeviceListD3D11(CL_DEVICE_TYPE_GPU, (void *)hdl) != CL_SUCCESS || platform->devs().size() == 0) {
                auto mes = strsprintf(_T("Failed to find d3d11 device in platform %s.\n"), char_to_tstring(platform->info().name).c_str());
                PrintMes(RGY_LOG_DEBUG, mes.c_str());
                clErrMessage += mes;
                continue;
            }
        } else if (m_device->memType() == VA_MEMORY && ENABLE_RGY_OPENCL_VA) {
            if (platform->createDeviceListVA(CL_DEVICE_TYPE_GPU, (void *)hdl) != CL_SUCCESS || platform->devs().size() == 0) {
                auto mes = strsprintf(_T("Failed to find va device in platform %s.\n"), char_to_tstring(platform->info().name).c_str());
                PrintMes(RGY_LOG_DEBUG, mes.c_str());
                clErrMessage += mes;
                continue;
            }
        } else {
            if (platform->createDeviceList(CL_DEVICE_TYPE_GPU) != CL_SUCCESS || platform->devs().size() == 0) {
                auto mes = _T("Failed to find gpu device.\n");
                PrintMes(RGY_LOG_DEBUG, mes);
                clErrMessage += mes;
                continue;
            }
        }
        selectedPlatform = platform;
        break;
    }
    if (!selectedPlatform) {
        PrintMes(RGY_LOG_WARN, clErrMessage.c_str());
        return RGY_ERR_NONE;
    }
    auto devices = selectedPlatform->devs();
    if ((int)devices.size() == 0) {
        PrintMes(RGY_LOG_WARN, _T("Failed to find OpenCL device.\n"));
        return RGY_ERR_NONE;
    }
    selectedPlatform->setDev(devices[0]);

    m_cl = std::make_shared<RGYOpenCLContext>(selectedPlatform, openCLBuildThreads, m_pQSVLog);
    if (m_cl->createContext((checkVppPerformance) ? CL_QUEUE_PROFILING_ENABLE : 0) != CL_SUCCESS) {
        PrintMes(RGY_LOG_WARN, _T("Failed to create OpenCL context.\n"));
        m_cl.reset();
        return RGY_ERR_NONE;
    }
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::ResetDevice() {
    if (m_device->memType() & (D3D9_MEMORY | D3D11_MEMORY)) {
        PrintMes(RGY_LOG_DEBUG, _T("HWDevice: reset.\n"));
        return err_to_rgy(m_device->hwdev()->Reset());
    }
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::AllocFrames() {
    if (m_pipelineTasks.size() == 0) {
        PrintMes(RGY_LOG_ERROR, _T("allocFrames: pipeline not defined!\n"));
        return RGY_ERR_INVALID_CALL;
    }

    PrintMes(RGY_LOG_DEBUG, _T("allocFrames: m_nAsyncDepth - %d frames\n"), m_nAsyncDepth);

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
        mfxFrameAllocRequest allocRequest = { 0 };
        bool allocateOpenCLFrame = false;
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
        } else if (t0->getOutputFrameInfo(allocRequest.Info) == RGY_ERR_NONE) {
            t0RequestNumFrame = std::max(t0->outputMaxQueueSize(), 1);
            t1RequestNumFrame = 1;
            if (   t0->taskType() == PipelineTaskType::OPENCL // openclとraw出力がつながっているような場合
                || t1->taskType() == PipelineTaskType::OPENCL // inputとopenclがつながっているような場合
            ) {
                if (!m_cl) {
                    PrintMes(RGY_LOG_ERROR, _T("AllocFrames: OpenCL filter not enabled.\n"));
                    return RGY_ERR_UNSUPPORTED;
                }
                allocateOpenCLFrame = true; // inputとopenclがつながっているような場合
            }
            if (t0->taskType() == PipelineTaskType::OPENCL) {
                t0RequestNumFrame += 4; // 内部でフレームが増える場合に備えて
            }
        } else {
            PrintMes(RGY_LOG_ERROR, _T("AllocFrames: invalid pipeline: cannot get request from either t0 or t1!\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        const int requestNumFrames = std::max(1, t0RequestNumFrame + t1RequestNumFrame + m_nAsyncDepth + 1);
        if (allocateOpenCLFrame) { // OpenCLフレームを介してやり取りする場合
            const RGYFrameInfo frame(allocRequest.Info.CropW, allocRequest.Info.CropH,
                csp_enc_to_rgy(allocRequest.Info.FourCC),
                (allocRequest.Info.BitDepthLuma > 0) ? allocRequest.Info.BitDepthLuma : 8,
                picstruct_enc_to_rgy(allocRequest.Info.PicStruct));
            PrintMes(RGY_LOG_DEBUG, _T("AllocFrames: %s-%s, type: CL, %s %dx%d, request %d frames\n"),
                t0->print().c_str(), t1->print().c_str(), RGY_CSP_NAMES[frame.csp],
                frame.width, frame.height, requestNumFrames);
            auto sts = t0->workSurfacesAllocCL(requestNumFrames, frame, m_cl.get());
            if (sts != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("AllocFrames:   Failed to allocate frames for %s-%s: %s."), t0->print().c_str(), t1->print().c_str(), get_err_mes(sts));
                return sts;
            }
        } else {
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

            allocRequest.AllocId = (m_device->externalAlloc()) ? m_device->allocator()->getExtAllocCounts() : 0u;
            allocRequest.NumFrameSuggested = (mfxU16)requestNumFrames;
            allocRequest.NumFrameMin = allocRequest.NumFrameSuggested;
            PrintMes(RGY_LOG_DEBUG, _T("AllocFrames: Id: %d, %s-%s, type: %s, %s %dx%d [%d,%d,%d,%d], request %d frames\n"),
                allocRequest.AllocId, t0->print().c_str(), t1->print().c_str(), qsv_memtype_str(allocRequest.Type).c_str(), ColorFormatToStr(allocRequest.Info.FourCC),
                allocRequest.Info.Width, allocRequest.Info.Height, allocRequest.Info.CropX, allocRequest.Info.CropY, allocRequest.Info.CropW, allocRequest.Info.CropH,
                allocRequest.NumFrameSuggested);

            auto sts = t0->workSurfacesAlloc(allocRequest, m_device->externalAlloc(), m_device->allocator());
            if (sts != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("AllocFrames:   Failed to allocate frames for %s-%s: %s."), t0->print().c_str(), t1->print().c_str(), get_err_mes(sts));
                return sts;
            }
        }
        t0 = t1;
    }
    return RGY_ERR_NONE;
}

CQSVPipeline::CQSVPipeline() :
    m_mfxVer({ 0 }),
    m_device(),
    m_devNames(),
    m_pStatus(),
    m_pPerfMonitor(),
    m_deviceUsage(),
    m_parallelEnc(),
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
    m_poolPkt(),
    m_poolFrame(),
    m_nAsyncDepth(0),
    m_nAVSyncMode(RGY_AVSYNC_AUTO),
    m_timestampPassThrough(false),
    m_encParams(MFX_LIB_VERSION_0_0),
    m_mfxDEC(),
    m_pmfxENC(),
    m_mfxVPP(),
    m_dynamicRC(),
    m_encFeatures(),
    m_trimParam(),
    m_prmSetIn(MFX_LIB_VERSION_0_0),
#if ENABLE_AVSW_READER
    m_Chapters(),
#endif
    m_timecode(),
    m_hdrseiIn(),
    m_hdrseiOut(),
    m_hdr10plus(),
    m_hdr10plusMetadataCopy(false),
    m_dovirpu(),
    m_dovirpuMetadataCopy(false),
    m_doviProfile(RGY_DOVI_PROFILE_UNSET),
    m_encTimestamp(),
    m_sessionParams(),
    m_nProcSpeedLimit(0),
    m_taskPerfMonitor(false),
    m_dummyLoad(),
    m_pAbortByUser(nullptr),
    m_heAbort(),
    m_DecInputBitstream(),
    m_cl(),
    m_vpFilters(),
    m_videoQualityMetric(),
    m_pipelineTasks() {
    m_trimParam.offset = 0;

#if ENABLE_MVC_ENCODING
    m_bIsMVC = false;
    m_MVCflags = MVC_DISABLED;
    m_nNumView = 0;
    RGY_MEMSET_ZERO(m_MVCSeqDesc);
    m_MVCSeqDesc.Header.BufferId = MFX_EXTBUFF_MVC_SEQ_DESC;
    m_MVCSeqDesc.Header.BufferSz = sizeof(m_MVCSeqDesc);
#endif

    RGY_MEMSET_ZERO(m_DecInputBitstream);

    RGY_MEMSET_ZERO(m_encParams.videoPrm);
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

RGY_CSP CQSVPipeline::getEncoderCsp(const sInputParams *pParams, int *pShift) const {
    auto csp = getMFXCsp(pParams->outputCsp, getEncoderBitdepth(pParams));
    if (pShift && fourccShiftUsed(csp_rgy_to_enc(csp))) {
        *pShift = (getEncoderBitdepth(pParams) > 8) ? 16 - pParams->outputDepth : 0;
    }
    return csp;
}

RGY_ERR CQSVPipeline::InitOutput(sInputParams *inputParams) {
    auto [err, outFrameInfo] = GetOutputVideoInfo();
    if (err != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to get output frame info!\n"));
        return err;
    }
    if (!m_pmfxENC) {
        outFrameInfo->videoPrm.mfx.CodecId = MFX_CODEC_RAW; //エンコードしない場合は出力コーデックはraw(=0)
    }
    const auto outputVideoInfo = (outFrameInfo->isVppParam) ? videooutputinfo(outFrameInfo->videoPrmVpp.vpp.Out) : videooutputinfo(outFrameInfo->videoPrm.mfx, m_encParams.videoSignalInfo, m_encParams.chromaLocInfo);
    if (outputVideoInfo.codec == RGY_CODEC_RAW) {
        inputParams->common.AVMuxTarget &= ~RGY_MUX_VIDEO;
    }
    m_hdrseiIn = createHEVCHDRSei(maxCLLSource, masterDisplaySource, RGY_TRANSFER_UNKNOWN, m_pFileReader.get());
    if (!m_hdrseiIn) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to parse HEVC HDR10 metadata.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    m_hdrseiOut = createHEVCHDRSei(inputParams->common.maxCll, inputParams->common.masterDisplay, inputParams->common.atcSei, m_pFileReader.get());
    if (!m_hdrseiOut) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to parse HEVC HDR10 metadata.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    err = initWriters(m_pFileWriter, m_pFileWriterListAudio, m_pFileReader, m_AudioReaders,
        &inputParams->common, &inputParams->input, &inputParams->ctrl, outputVideoInfo,
        m_trimParam, m_outputTimebase,
#if ENABLE_AVSW_READER
        m_Chapters,
#endif //#if ENABLE_AVSW_READER
        m_hdrseiOut.get(), m_hdr10plus.get(), m_dovirpu.get(), m_encTimestamp.get(),
        !check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_6),
        inputParams->bBenchmark, false, 0,
        m_poolPkt.get(), m_poolFrame.get(),
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

DeviceCodecCsp CQSVPipeline::getHWDecCodecCsp(const bool skipHWDecodeCheck, std::vector<std::unique_ptr<QSVDevice>>& devList) {
    DeviceCodecCsp HWDecCodecCsp;
    for (const auto& dev : devList) {
        HWDecCodecCsp.push_back(std::make_pair((int)dev->deviceNum(), dev->getDecodeCodecCsp(skipHWDecodeCheck)));
    }
    return HWDecCodecCsp;
}

RGY_ERR CQSVPipeline::InitInput(sInputParams *inputParam, DeviceCodecCsp& HWDecCodecCsp) {
#if ENABLE_RAW_READER
    m_pStatus = std::make_shared<EncodeStatus>();

    int subburnTrackId = 0;
    for (const auto &subburn : inputParam->vpp.subburn) {
        if (subburn.trackId > 0) {
            subburnTrackId = subburn.trackId;
            break;
        }
    }

    //--input-cspの値 (raw読み込み用の入力色空間)
    //この後上書きするので、ここで保存する
    const auto inputCspOfRawReader = inputParam->input.csp;

    //入力モジュールが、エンコーダに返すべき色空間をセット
    inputParam->input.csp = getEncoderCsp(inputParam, &inputParam->input.bitdepth);

    // インタレ解除が指定され、かつインタレの指定がない場合は、自動的にインタレの情報取得を行う
    int deinterlacer = 0;
    if (inputParam->vppmfx.deinterlace) deinterlacer++;
    if (inputParam->vpp.afs.enable) deinterlacer++;
    if (inputParam->vpp.nnedi.enable) deinterlacer++;
    if (inputParam->vpp.yadif.enable) deinterlacer++;
    if (inputParam->vpp.decomb.enable) deinterlacer++;
    if (deinterlacer > 0 && ((inputParam->input.picstruct & RGY_PICSTRUCT_INTERLACED) == 0)) {
        inputParam->input.picstruct = RGY_PICSTRUCT_AUTO;
    }

    m_poolPkt = std::make_unique<RGYPoolAVPacket>();
    m_poolFrame = std::make_unique<RGYPoolAVFrame>();

    auto sts = initReaders(m_pFileReader, m_AudioReaders, &inputParam->input, &inputParam->inprm, inputCspOfRawReader,
        m_pStatus, &inputParam->common, &inputParam->ctrl, HWDecCodecCsp, subburnTrackId,
        (ENABLE_VPP_FILTER_RFF) ? inputParam->vpp.rff.enable : false,
        (ENABLE_VPP_FILTER_AFS) ? inputParam->vpp.afs.enable : false,
        inputParam->vpp.libplacebo_tonemapping.enable,
        m_poolPkt.get(), m_poolFrame.get(),
        nullptr, m_pPerfMonitor.get(), m_pQSVLog);
    if (sts != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("failed to initialize file reader(s).\n"));
        return sts;
    }
    PrintMes(RGY_LOG_DEBUG, _T("initReaders: Success.\n"));

    m_inputFps = rgy_rational<int>(inputParam->input.fpsN, inputParam->input.fpsD);
    m_outputTimebase = (inputParam->common.timebase.is_valid()) ? inputParam->common.timebase : m_inputFps.inv() * rgy_rational<int>(1, 4);
    m_timestampPassThrough = inputParam->common.timestampPassThrough;
    if (inputParam->common.timestampPassThrough) {
        PrintMes(RGY_LOG_DEBUG, _T("Switching to VFR mode as --timestamp-paththrough is used.\n"));
        m_nAVSyncMode = RGY_AVSYNC_VFR;
    }
    if (inputParam->common.tcfileIn.length() > 0) {
        PrintMes(RGY_LOG_DEBUG, _T("Switching to VFR mode as --tcfile-in is used.\n"));
        m_nAVSyncMode |= RGY_AVSYNC_VFR;
    }
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
#if ENABLE_VPP_FILTER_RFF
        || inputParam->vpp.rff.enable
#endif
        ) {
        tstring err_target;
        if (m_nAVSyncMode & RGY_AVSYNC_VFR)       err_target += _T("avsync vfr, ");
        if (m_nAVSyncMode & RGY_AVSYNC_FORCE_CFR) err_target += _T("avsync forcecfr, ");
#if ENABLE_VPP_FILTER_RFF
        if (inputParam->vpp.rff.enable) {
            err_target += _T("vpp-rff, ");
            m_nAVSyncMode = RGY_AVSYNC_VFR;
        }
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
        if (!ENCODER_QSV) {
            m_nAVSyncMode |= RGY_AVSYNC_VFR;
            const auto timebaseStreamIn = to_rgy(pAVCodecReader->GetInputVideoStream()->time_base);
            if (!inputParam->common.timebase.is_valid()
                && ((timebaseStreamIn.inv() * m_inputFps.inv()).d() == 1 || timebaseStreamIn.n() > 1000)) { //fpsを割り切れるtimebaseなら
                if (!inputParam->vpp.afs.enable && !inputParam->vpp.rff.enable) {
                    m_outputTimebase = m_inputFps.inv() * rgy_rational<int>(1, 4);
                }
            }
        }
        PrintMes(RGY_LOG_DEBUG, _T("vfr mode automatically enabled with timebase %d/%d\n"), m_outputTimebase.n(), m_outputTimebase.d());
    }
    if (inputParam->common.dynamicHdr10plusJson.length() > 0) {
        m_hdr10plus = initDynamicHDR10Plus(inputParam->common.dynamicHdr10plusJson, m_pQSVLog);
        if (!m_hdr10plus) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to initialize hdr10plus reader.\n"));
            return RGY_ERR_UNKNOWN;
        }
    } else if (inputParam->common.hdr10plusMetadataCopy) {
        m_hdr10plusMetadataCopy = true;
        if (pAVCodecReader != nullptr) {
            const auto timestamp_status = pAVCodecReader->GetFramePosList()->getStreamPtsStatus();
            if ((timestamp_status & (~RGY_PTS_NORMAL)) != 0) {
                PrintMes(RGY_LOG_ERROR, _T("HDR10+ dynamic metadata cannot be copied from input file using avhw reader, as timestamp was not properly got from input file.\n"));
                PrintMes(RGY_LOG_ERROR, _T("Please consider using avsw reader.\n"));
                return RGY_ERR_UNSUPPORTED;
            }
        }
    }
    if (inputParam->common.doviRpuFile.length() > 0) {
        m_dovirpu = std::make_unique<DOVIRpu>();
        if (m_dovirpu->init(inputParam->common.doviRpuFile.c_str()) != 0) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to open dovi rpu \"%s\".\n"), inputParam->common.doviRpuFile.c_str());
            return RGY_ERR_FILE_OPEN;
        }
    } else if (inputParam->common.doviRpuMetadataCopy) {
        m_dovirpuMetadataCopy = true;
    }
    m_doviProfile = inputParam->common.doviProfile;
#endif //#if ENABLE_AVSW_READER
    return RGY_ERR_NONE;
#else
    return RGY_ERR_UNSUPPORTED;
#endif //#if ENABLE_RAW_READER
}

RGY_ERR CQSVPipeline::CheckParam(sInputParams *inputParam) {
    const auto inputFrameInfo = m_pFileReader->GetInputFrameInfo();

    //いろいろなチェックの前提となる
    applyInputVUIToColorspaceParams(inputParam);

    if ((inputParam->memType & HW_MEMORY) == HW_MEMORY) { //自動モードの場合
        //OpenCLフィルタを使う場合はd3d11を使用する
        if (preferD3D11Mode(inputParam)) {
            inputParam->memType = D3D11_MEMORY;
            PrintMes(RGY_LOG_DEBUG, _T("d3d11 mode prefered, switched to d3d11 mode.\n"));
        //出力コーデックがrawなら、systemメモリを自動的に使用する
        } else if (inputParam->codec == RGY_CODEC_RAW) {
            inputParam->memType = SYSTEM_MEMORY;
            PrintMes(RGY_LOG_DEBUG, _T("Automatically selecting system memory for output raw frames.\n"));
        }
    }

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
        if (m_device->CPUGen() <= CPU_GEN_HASWELL && m_pFileReader->getInputCodec() == RGY_CODEC_HEVC) {
            if (inputParam->memType & D3D9_MEMORY) {
                inputParam->memType &= ~D3D9_MEMORY;
                inputParam->memType |= D3D11_MEMORY;
            }
            PrintMes(RGY_LOG_DEBUG, _T("Switched to d3d11 mode for HEVC decoding on Haswell.\n"));
        }
        if (m_pFileReader->getInputCodec() == RGY_CODEC_AV1) {
            if (inputParam->memType & D3D9_MEMORY) {
                inputParam->memType &= ~D3D9_MEMORY;
                inputParam->memType |= D3D11_MEMORY;
            }
            PrintMes(RGY_LOG_DEBUG, _T("Switched to d3d11 mode for AV1 decoding.\n"));
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
        && ((inputParam->input.dstWidth == inputParam->input.srcWidth && inputParam->input.dstHeight == inputParam->input.srcHeight)
            || (inputParam->input.dstWidth == 0 || inputParam->input.dstHeight == 0))) {//リサイズは行われない
        outpar = std::make_pair(inputParam->input.sar[0], inputParam->input.sar[1]);
    }
    if (inputParam->input.dstWidth < 0 && inputParam->input.dstHeight < 0) {
        PrintMes(RGY_LOG_ERROR, _T("Either one of output resolution must be positive value.\n"));
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }

    set_auto_resolution(inputParam->input.dstWidth, inputParam->input.dstHeight, outpar.first, outpar.second,
        inputParam->input.srcWidth, inputParam->input.srcHeight, inputParam->input.sar[0], inputParam->input.sar[1], 2, 2, inputParam->inprm.resizeResMode, inputParam->inprm.ignoreSAR, inputParam->input.crop);

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

void CQSVPipeline::applyInputVUIToColorspaceParams(sInputParams *inputParam) {
    auto currentVUI = inputParam->input.vui;
    for (size_t i = 0; i < inputParam->vpp.colorspace.convs.size(); i++) {
        auto conv_from = inputParam->vpp.colorspace.convs[i].from;
        conv_from.apply_auto(currentVUI, inputParam->input.srcHeight);
        if (i == 0) {
            inputParam->vppmfx.colorspace.from.matrix = conv_from.matrix;
            inputParam->vppmfx.colorspace.from.range = conv_from.colorrange;
        }

        auto conv_to = inputParam->vpp.colorspace.convs[i].to;
        const bool is_last_conversion = i == (inputParam->vpp.colorspace.convs.size() - 1);
        if (is_last_conversion) {
            conv_to.apply_auto(m_encVUI, m_encHeight);
            inputParam->vppmfx.colorspace.to.matrix = conv_to.matrix;
            inputParam->vppmfx.colorspace.to.range = conv_to.colorrange;
        } else {
            conv_to.apply_auto(conv_from, inputParam->input.srcHeight);
        }
    }
}

std::vector<VppType> CQSVPipeline::InitFiltersCreateVppList(const sInputParams *inputParam, const bool cspConvRequired, const bool cropRequired, const RGY_VPP_RESIZE_TYPE resizeRequired) {
    std::vector<VppType> filterPipeline;
    filterPipeline.reserve((size_t)VppType::CL_MAX);

    if (cspConvRequired || cropRequired) {
        filterPipeline.push_back((inputParam->outputCsp == RGY_CHROMAFMT_RGB) ? VppType::CL_CROP : VppType::MFX_CROP);
    }
    if (inputParam->vpp.colorspace.enable) {
        bool requireOpenCL = inputParam->vpp.colorspace.hdr2sdr.tonemap != HDR2SDR_DISABLED || inputParam->vpp.colorspace.lut3d.table_file.length() > 0;
        if (!requireOpenCL) {
            auto currentVUI = inputParam->input.vui;
            for (size_t i = 0; i < inputParam->vpp.colorspace.convs.size(); i++) {
                auto conv_from = inputParam->vpp.colorspace.convs[i].from;
                auto conv_to = inputParam->vpp.colorspace.convs[i].to;
                if (conv_from.chromaloc != conv_to.chromaloc
                    || conv_from.colorprim != conv_to.colorprim
                    || conv_from.transfer != conv_to.transfer) {
                    requireOpenCL = true;
                } else if (conv_from.matrix != conv_to.matrix
                    && (conv_from.matrix != RGY_MATRIX_ST170_M && conv_from.matrix != RGY_MATRIX_BT709)
                    && (conv_to.matrix != RGY_MATRIX_ST170_M && conv_to.matrix != RGY_MATRIX_BT709)) {
                    requireOpenCL = true;
                }
            }
        }
        filterPipeline.push_back((requireOpenCL) ? VppType::CL_COLORSPACE : VppType::MFX_COLORSPACE);
    }
    if (inputParam->vpp.libplacebo_tonemapping.enable) filterPipeline.push_back(VppType::CL_LIBPLACEBO_TONEMAP);
    if (inputParam->vpp.rff.enable)        filterPipeline.push_back(VppType::CL_RFF);
    if (inputParam->vpp.delogo.enable)     filterPipeline.push_back(VppType::CL_DELOGO);
    if (inputParam->vpp.afs.enable)        filterPipeline.push_back(VppType::CL_AFS);
    if (inputParam->vpp.nnedi.enable)      filterPipeline.push_back(VppType::CL_NNEDI);
    if (inputParam->vpp.yadif.enable)      filterPipeline.push_back(VppType::CL_YADIF);
    if (inputParam->vpp.decomb.enable)     filterPipeline.push_back(VppType::CL_DECOMB);
    if (inputParam->vppmfx.deinterlace != MFX_DEINTERLACE_NONE)  filterPipeline.push_back(VppType::MFX_DEINTERLACE);
    if (inputParam->vpp.decimate.enable)   filterPipeline.push_back(VppType::CL_DECIMATE);
    if (inputParam->vpp.mpdecimate.enable) filterPipeline.push_back(VppType::CL_MPDECIMATE);
    if (inputParam->vpp.convolution3d.enable) filterPipeline.push_back(VppType::CL_CONVOLUTION3D);
    if (inputParam->vpp.smooth.enable)     filterPipeline.push_back(VppType::CL_DENOISE_SMOOTH);
    if (inputParam->vpp.dct.enable)        filterPipeline.push_back(VppType::CL_DENOISE_DCT);
    if (inputParam->vpp.fft3d.enable)      filterPipeline.push_back(VppType::CL_DENOISE_FFT3D);
    if (inputParam->vpp.knn.enable)        filterPipeline.push_back(VppType::CL_DENOISE_KNN);
    if (inputParam->vpp.nlmeans.enable)    filterPipeline.push_back(VppType::CL_DENOISE_NLMEANS);
    if (inputParam->vpp.pmd.enable)        filterPipeline.push_back(VppType::CL_DENOISE_PMD);
    if (inputParam->vppmfx.denoise.enable) filterPipeline.push_back(VppType::MFX_DENOISE);
    if (inputParam->vppmfx.imageStabilizer != 0) filterPipeline.push_back(VppType::MFX_IMAGE_STABILIZATION);
    if (inputParam->vppmfx.mctf.enable)    filterPipeline.push_back(VppType::MFX_MCTF);
    if (inputParam->vpp.subburn.size()>0)  filterPipeline.push_back(VppType::CL_SUBBURN);
    if (inputParam->vpp.libplacebo_shader.size()>0)  filterPipeline.push_back(VppType::CL_LIBPLACEBO_SHADER);
    if (     resizeRequired == RGY_VPP_RESIZE_TYPE_OPENCL
#if ENABLE_LIBPLACEBO
        || resizeRequired == RGY_VPP_RESIZE_TYPE_LIBPLACEBO
#endif
        ) filterPipeline.push_back(VppType::CL_RESIZE);
    else if (resizeRequired != RGY_VPP_RESIZE_TYPE_NONE)   filterPipeline.push_back(VppType::MFX_RESIZE);
    if (inputParam->vpp.unsharp.enable)    filterPipeline.push_back(VppType::CL_UNSHARP);
    if (inputParam->vpp.edgelevel.enable)  filterPipeline.push_back(VppType::CL_EDGELEVEL);
    if (inputParam->vpp.warpsharp.enable)  filterPipeline.push_back(VppType::CL_WARPSHARP);
    if (inputParam->vppmfx.detail.enable)  filterPipeline.push_back(VppType::MFX_DETAIL_ENHANCE);
    if (inputParam->vppmfx.mirrorType != MFX_MIRRORING_DISABLED) filterPipeline.push_back(VppType::MFX_MIRROR);
    if (inputParam->vpp.transform.enable)  filterPipeline.push_back(VppType::CL_TRANSFORM);
    if (inputParam->vpp.curves.enable)     filterPipeline.push_back(VppType::CL_CURVES);
    if (inputParam->vpp.tweak.enable)      filterPipeline.push_back(VppType::CL_TWEAK);
    if (inputParam->vpp.deband.enable)     filterPipeline.push_back(VppType::CL_DEBAND);
    if (inputParam->vpp.libplacebo_deband.enable)     filterPipeline.push_back(VppType::CL_LIBPLACEBO_DEBAND);
    if (inputParam->vpp.pad.enable)        filterPipeline.push_back(VppType::CL_PAD);
    if (inputParam->vppmfx.percPreEnc)     filterPipeline.push_back(VppType::MFX_PERC_ENC_PREFILTER);
    if (inputParam->vpp.overlay.size() > 0)  filterPipeline.push_back(VppType::CL_OVERLAY);

    if (filterPipeline.size() == 0) {
        return filterPipeline;
    }

    //OpenCLが使用できない場合
    if (!m_cl) {
        //置き換え
        for (auto& filter : filterPipeline) {
            if (filter == VppType::CL_RESIZE) filter = VppType::MFX_RESIZE;
        }
        //削除
        decltype(filterPipeline) newPipeline;
        for (auto& filter : filterPipeline) {
            if (getVppFilterType(filter) != VppFilterType::FILTER_OPENCL) {
                newPipeline.push_back(filter);
            }
        }
        if (filterPipeline.size() != newPipeline.size()) {
            PrintMes(RGY_LOG_WARN, _T("OpenCL disabled, OpenCL based vpp filters will be disabled!\n"));
        }
        filterPipeline = newPipeline;
    }

    // cropとresizeはmfxとopencl両方ともあるので、前後のフィルタがどちらもOpenCLだったら、そちらに合わせる
    for (size_t i = 0; i < filterPipeline.size(); i++) {
        const VppFilterType prev = (i >= 1)                        ? getVppFilterType(filterPipeline[i - 1]) : VppFilterType::FILTER_NONE;
        const VppFilterType next = (i + 1 < filterPipeline.size()) ? getVppFilterType(filterPipeline[i + 1]) : VppFilterType::FILTER_NONE;
        if (filterPipeline[i] == VppType::MFX_RESIZE) {
            if (resizeRequired == RGY_VPP_RESIZE_TYPE_AUTO // 自動以外の指定があれば、それに従うので、自動の場合のみ変更
                && m_cl
                && prev == VppFilterType::FILTER_OPENCL
                && next == VppFilterType::FILTER_OPENCL) {
                filterPipeline[i] = VppType::CL_RESIZE; // OpenCLに挟まれていたら、OpenCLのresizeを優先する
            }
        } else if (filterPipeline[i] == VppType::MFX_CROP) {
            if (m_cl
                && (prev == VppFilterType::FILTER_OPENCL || next == VppFilterType::FILTER_OPENCL)
                && (prev != VppFilterType::FILTER_MFX    || next != VppFilterType::FILTER_MFX)) {
                filterPipeline[i] = VppType::CL_CROP; // OpenCLに挟まれていたら、OpenCLのcropを優先する
            }
        } else if (filterPipeline[i] == VppType::MFX_COLORSPACE) {
            if (m_cl
                && prev == VppFilterType::FILTER_OPENCL
                && next == VppFilterType::FILTER_OPENCL) {
                filterPipeline[i] = VppType::CL_COLORSPACE; // OpenCLに挟まれていたら、OpenCLのcolorspaceを優先する
            }
        }
    }
    return filterPipeline;
}

std::pair<RGY_ERR, std::unique_ptr<QSVVppMfx>> CQSVPipeline::AddFilterMFX(
    RGYFrameInfo& frameInfo, rgy_rational<int>& fps,
    const VppType vppType, const sVppParams *params, const RGY_CSP outCsp, const int outBitdepth, const sInputCrop *crop, const std::pair<int,int> resize, const int blockSize) {
    RGYFrameInfo frameIn = frameInfo;
    sVppParams vppParams;
    vppParams.bEnable = true;
    switch (vppType) {
    case VppType::MFX_COPY: break;
    case VppType::MFX_DEINTERLACE:         vppParams.deinterlace = params->deinterlace; break;
    case VppType::MFX_DENOISE:             vppParams.denoise = params->denoise; break;
    case VppType::MFX_DETAIL_ENHANCE:      vppParams.detail = params->detail; break;
    case VppType::MFX_COLORSPACE:          vppParams.colorspace = params->colorspace; vppParams.colorspace.enable = true; break;
    case VppType::MFX_IMAGE_STABILIZATION: vppParams.imageStabilizer = params->imageStabilizer; break;
    case VppType::MFX_ROTATE:              vppParams.rotate = params->rotate; break;
    case VppType::MFX_MIRROR:              vppParams.mirrorType = params->mirrorType; break;
    case VppType::MFX_MCTF:                vppParams.mctf = params->mctf; break;
    case VppType::MFX_PERC_ENC_PREFILTER:  vppParams.percPreEnc = params->percPreEnc; break;
    case VppType::MFX_AISUPRERES:
    case VppType::MFX_RESIZE:              vppParams.bUseResize = true;
                                           vppParams.resizeInterp = params->resizeInterp;
                                           vppParams.resizeMode = params->resizeMode;
                                           vppParams.aiSuperRes.enable = params->aiSuperRes.enable;
                                           frameInfo.width = resize.first;
                                           frameInfo.height = resize.second;
                                           if (resize.first == 0 || resize.second == 0
                                               || (frameInfo.width == frameIn.width && frameInfo.height == frameIn.height)) {
                                               return { RGY_ERR_NONE, std::unique_ptr<QSVVppMfx>() };
                                           }
                                           break;
    case VppType::MFX_CROP:                frameInfo.width  -= (crop) ? (crop->e.left + crop->e.right) : 0;
                                           frameInfo.height -= (crop) ? (crop->e.up + crop->e.bottom)  : 0; break;
    case VppType::MFX_FPS_CONV:
    default:
        return { RGY_ERR_UNSUPPORTED, std::unique_ptr<QSVVppMfx>() };
    }

    frameInfo.csp = outCsp; // 常に適用
    frameInfo.bitdepth = outBitdepth;

    mfxIMPL impl;
    m_device->mfxSession().QueryIMPL(&impl);
    auto mfxvpp = std::make_unique<QSVVppMfx>(m_device->hwdev(), m_device->allocator(), m_mfxVer, impl, m_device->memType(), m_sessionParams, m_device->deviceNum(), m_nAsyncDepth, m_pQSVLog);
    auto err = mfxvpp->SetParam(vppParams, frameInfo, frameIn, (vppType == VppType::MFX_CROP) ? crop : nullptr,
        fps, rgy_rational<int>(1,1), blockSize);
    if (err != RGY_ERR_NONE) {
        return { err, std::unique_ptr<QSVVppMfx>() };
    }

    if (vppType != VppType::MFX_COPY // copyの時は意図的にアクションがない
        && mfxvpp->GetVppList().size() == 0) {
        PrintMes(RGY_LOG_WARN, _T("filtering has no action.\n"));
        return { err, std::unique_ptr<QSVVppMfx>() };
    }

    //入力フレーム情報を更新
    frameInfo = mfxvpp->GetFrameOut();
    fps = mfxvpp->GetOutFps();

    return { RGY_ERR_NONE, std::move(mfxvpp) };
}

RGY_ERR CQSVPipeline::AddFilterOpenCL(std::vector<std::unique_ptr<RGYFilter>>& clfilters,
    RGYFrameInfo& inputFrame, const VppType vppType, const sInputParams *params, const sInputCrop *crop, const std::pair<int, int> resize, VideoVUIInfo& vuiInfo) {
    //colorspace
    if (vppType == VppType::CL_COLORSPACE) {
        unique_ptr<RGYFilterColorspace> filter(new RGYFilterColorspace(m_cl));
        shared_ptr<RGYFilterParamColorspace> param(new RGYFilterParamColorspace());
        param->colorspace = params->vpp.colorspace;
        param->encCsp = inputFrame.csp;
        param->VuiIn = vuiInfo;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        vuiInfo = filter->VuiOut();
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //libplacebo-tonemap
    if (vppType == VppType::CL_LIBPLACEBO_TONEMAP) {
        unique_ptr<RGYFilterLibplaceboToneMapping> filter(new RGYFilterLibplaceboToneMapping(m_cl));
        shared_ptr<RGYFilterParamLibplaceboToneMapping> param(new RGYFilterParamLibplaceboToneMapping());
        param->toneMapping = params->vpp.libplacebo_tonemapping;
        param->hdrMetadataIn = m_hdrseiIn.get();
        param->hdrMetadataOut = m_hdrseiOut.get();
        param->vk = m_device->vulkan();
        param->vui = vuiInfo;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
#if ENABLE_LIBPLACEBO
        vuiInfo = filter->VuiOut();
#endif
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //rff
    if (vppType == VppType::CL_RFF) {
        unique_ptr<RGYFilter> filter(new RGYFilterRff(m_cl));
        shared_ptr<RGYFilterParamRff> param(new RGYFilterParamRff());
        param->rff = params->vpp.rff;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->inFps = m_inputFps;
        param->timebase = m_outputTimebase;
        param->outFilename = params->common.outputFilename;
        param->bOutOverwrite = true;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //delogo
    if (vppType == VppType::CL_DELOGO) {
        unique_ptr<RGYFilter> filter(new RGYFilterDelogo(m_cl));
        shared_ptr<RGYFilterParamDelogo> param(new RGYFilterParamDelogo());
        param->inputFileName = params->common.inputFilename.c_str();
        param->delogo = params->vpp.delogo;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = true;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //afs
    if (vppType == VppType::CL_AFS) {
        unique_ptr<RGYFilter> filter(new RGYFilterAfs(m_cl));
        shared_ptr<RGYFilterParamAfs> param(new RGYFilterParamAfs());
        param->afs = params->vpp.afs;
        param->afs.tb_order = 1;
        if (params->input.picstruct & RGY_PICSTRUCT_BFF) param->afs.tb_order = 0;
        if (params->input.picstruct & RGY_PICSTRUCT_TFF) param->afs.tb_order = 1;
        if (params->common.timecode && param->afs.timecode) {
            param->afs.timecode = 2;
        }
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->inFps = m_inputFps;
        param->inTimebase = m_outputTimebase;
        param->outTimebase = m_outputTimebase;
        param->baseFps = m_encFps;
        param->outFilename = params->common.outputFilename;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //nnedi
    if (vppType == VppType::CL_NNEDI) {
        unique_ptr<RGYFilter> filter(new RGYFilterNnedi(m_cl));
        shared_ptr<RGYFilterParamNnedi> param(new RGYFilterParamNnedi());
        param->nnedi = params->vpp.nnedi;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->timebase = m_outputTimebase;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //yadif
    if (vppType == VppType::CL_YADIF) {
        unique_ptr<RGYFilter> filter(new RGYFilterYadif(m_cl));
        shared_ptr<RGYFilterParamYadif> param(new RGYFilterParamYadif());
        param->yadif = params->vpp.yadif;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->timebase = m_outputTimebase;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //decomb
    if (vppType == VppType::CL_DECOMB) {
        unique_ptr<RGYFilter> filter(new RGYFilterDecomb(m_cl));
        shared_ptr<RGYFilterParamDecomb> param(new RGYFilterParamDecomb());
        param->decomb = params->vpp.decomb;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //decimate
    if (vppType == VppType::CL_DECIMATE) {
        unique_ptr<RGYFilter> filter(new RGYFilterDecimate(m_cl));
        shared_ptr<RGYFilterParamDecimate> param(new RGYFilterParamDecimate());
        param->decimate = params->vpp.decimate;
        //QSV:Broadwell以前の環境では、なぜか別のキューで実行しようとすると、永遠にqueueMapBufferが開始されず、フリーズしてしまう
        //こういうケースでは標準のキューを使って逐次実行する
        param->useSeparateQueue = m_device->CPUGen() >= CPU_GEN_SKYLAKE;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //mpdecimate
    if (vppType == VppType::CL_MPDECIMATE) {
        unique_ptr<RGYFilter> filter(new RGYFilterMpdecimate(m_cl));
        shared_ptr<RGYFilterParamMpdecimate> param(new RGYFilterParamMpdecimate());
        param->mpdecimate = params->vpp.mpdecimate;
        //QSV:Broadwell以前の環境では、なぜか別のキューで実行しようとすると、永遠にqueueMapBufferが開始されず、フリーズしてしまう
        //こういうケースでは標準のキューを使って逐次実行する
        param->useSeparateQueue = m_device->CPUGen() >= CPU_GEN_SKYLAKE;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //回転
    if (vppType == VppType::CL_TRANSFORM) {
        unique_ptr<RGYFilter> filter(new RGYFilterTransform(m_cl));
        shared_ptr<RGYFilterParamTransform> param(new RGYFilterParamTransform());
        param->trans = params->vpp.transform;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //convolution3D
    if (vppType == VppType::CL_CONVOLUTION3D) {
        unique_ptr<RGYFilter> filter(new RGYFilterConvolution3D(m_cl));
        shared_ptr<RGYFilterParamConvolution3D> param(new RGYFilterParamConvolution3D());
        param->convolution3d = params->vpp.convolution3d;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //smooth
    if (vppType == VppType::CL_DENOISE_SMOOTH) {
        unique_ptr<RGYFilter> filter(new RGYFilterSmooth(m_cl));
        shared_ptr<RGYFilterParamSmooth> param(new RGYFilterParamSmooth());
        param->smooth = params->vpp.smooth;
        param->qpTableRef = nullptr;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //denoise-dct
    if (vppType == VppType::CL_DENOISE_DCT) {
        unique_ptr<RGYFilter> filter(new RGYFilterDenoiseDct(m_cl));
        shared_ptr<RGYFilterParamDenoiseDct> param(new RGYFilterParamDenoiseDct());
        param->dct = params->vpp.dct;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //fft3d
    if (vppType == VppType::CL_DENOISE_FFT3D) {
        unique_ptr<RGYFilter> filter(new RGYFilterDenoiseFFT3D(m_cl));
        shared_ptr<RGYFilterParamDenoiseFFT3D> param(new RGYFilterParamDenoiseFFT3D());
        param->fft3d = params->vpp.fft3d;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //knn
    if (vppType == VppType::CL_DENOISE_KNN) {
        unique_ptr<RGYFilter> filter(new RGYFilterDenoiseKnn(m_cl));
        shared_ptr<RGYFilterParamDenoiseKnn> param(new RGYFilterParamDenoiseKnn());
        param->knn = params->vpp.knn;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //nlmeans
    if (vppType == VppType::CL_DENOISE_NLMEANS) {
        unique_ptr<RGYFilter> filter(new RGYFilterDenoiseNLMeans(m_cl));
        shared_ptr<RGYFilterParamDenoiseNLMeans> param(new RGYFilterParamDenoiseNLMeans());
        param->nlmeans = params->vpp.nlmeans;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //pmd
    if (vppType == VppType::CL_DENOISE_PMD) {
        unique_ptr<RGYFilter> filter(new RGYFilterDenoisePmd(m_cl));
        shared_ptr<RGYFilterParamDenoisePmd> param(new RGYFilterParamDenoisePmd());
        param->pmd = params->vpp.pmd;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //字幕焼きこみ
    if (vppType == VppType::CL_SUBBURN) {
        std::vector<std::unique_ptr<RGYFilter>> filters;
        for (const auto& subburn : params->vpp.subburn) {
#if ENABLE_AVSW_READER
            if (subburn.filename.length() > 0
                && m_trimParam.list.size() > 0) {
                PrintMes(RGY_LOG_ERROR, _T("--vpp-subburn with input as file cannot be used with --trim.\n"));
                return RGY_ERR_UNSUPPORTED;
            }
            unique_ptr<RGYFilter> filter(new RGYFilterSubburn(m_cl));
            shared_ptr<RGYFilterParamSubburn> param(new RGYFilterParamSubburn());
            param->subburn = subburn;
            if (m_timestampPassThrough) {
                param->subburn.vid_ts_offset = false;
            }

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
                param->poolPkt = m_poolPkt.get();
                if (crop) param->crop = *crop;
                auto sts = filter->init(param, m_pQSVLog);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                //入力フレーム情報を更新
                inputFrame = param->frameOut;
                m_encFps = param->baseFps;
                clfilters.push_back(std::move(filter));
            }
#endif //#if ENABLE_AVSW_READER
        }
        return RGY_ERR_NONE;
    }
    if (vppType == VppType::CL_LIBPLACEBO_SHADER) {
        for (const auto& shader : params->vpp.libplacebo_shader) {
            unique_ptr<RGYFilter> filter(new RGYFilterLibplaceboShader(m_cl));
            shared_ptr<RGYFilterParamLibplaceboShader> param(new RGYFilterParamLibplaceboShader());
            param->shader = shader;
            param->frameIn = inputFrame;
            param->frameOut = inputFrame;
            if (param->shader.width > 0 && param->shader.height > 0) {
                param->frameOut.width = param->shader.width;
                param->frameOut.height = param->shader.height;
            }
            param->baseFps = m_encFps;
            param->bOutOverwrite = false;
            param->vk = m_device->vulkan();
            auto sts = filter->init(param, m_pQSVLog);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            //入力フレーム情報を更新
            inputFrame = param->frameOut;
            m_encFps = param->baseFps;
            //登録
            clfilters.push_back(std::move(filter));
        }
        return RGY_ERR_NONE;
    }
    //リサイズ
    if (vppType == VppType::CL_RESIZE) {
        if (resize.first > 0 && resize.second > 0
            && (resize.first != inputFrame.width || resize.second != inputFrame.height)) {
            auto filter = std::make_unique<RGYFilterResize>(m_cl);
            shared_ptr<RGYFilterParamResize> param(new RGYFilterParamResize());
            param->interp = (params->vpp.resize_algo != RGY_VPP_RESIZE_AUTO) ? params->vpp.resize_algo : RGY_VPP_RESIZE_SPLINE36;
            param->frameIn = inputFrame;
            param->frameOut = inputFrame;
            param->frameOut.width = resize.first;
            param->frameOut.height = resize.second;
            param->baseFps = m_encFps;
            param->bOutOverwrite = false;
            if (isLibplaceboResizeFiter(params->vpp.resize_algo)) {
                param->libplaceboResample = std::make_shared<RGYFilterParamLibplaceboResample>();
                param->libplaceboResample->resample = params->vpp.resize_libplacebo;
                param->libplaceboResample->resize_algo = param->interp;
                param->libplaceboResample->vk = m_device->vulkan();
            }
            auto sts = filter->init(param, m_pQSVLog);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            //入力フレーム情報を更新
            inputFrame = param->frameOut;
            m_encFps = param->baseFps;
            //登録
            clfilters.push_back(std::move(filter));
        }
        return RGY_ERR_NONE;
    }
    //unsharp
    if (vppType == VppType::CL_UNSHARP) {
        unique_ptr<RGYFilter> filter(new RGYFilterUnsharp(m_cl));
        shared_ptr<RGYFilterParamUnsharp> param(new RGYFilterParamUnsharp());
        param->unsharp = params->vpp.unsharp;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //edgelevel
    if (vppType == VppType::CL_EDGELEVEL) {
        unique_ptr<RGYFilter> filter(new RGYFilterEdgelevel(m_cl));
        shared_ptr<RGYFilterParamEdgelevel> param(new RGYFilterParamEdgelevel());
        param->edgelevel = params->vpp.edgelevel;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //warpsharp
    if (vppType == VppType::CL_WARPSHARP) {
        unique_ptr<RGYFilter> filter(new RGYFilterWarpsharp(m_cl));
        shared_ptr<RGYFilterParamWarpsharp> param(new RGYFilterParamWarpsharp());
        param->warpsharp = params->vpp.warpsharp;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //curves
    if (vppType == VppType::CL_CURVES) {
        unique_ptr<RGYFilter> filter(new RGYFilterCurves(m_cl));
        shared_ptr<RGYFilterParamCurves> param(new RGYFilterParamCurves());
        param->curves = params->vpp.curves;
        param->vuiInfo = vuiInfo;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = true;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //tweak
    if (vppType == VppType::CL_TWEAK) {
        unique_ptr<RGYFilter> filter(new RGYFilterTweak(m_cl));
        shared_ptr<RGYFilterParamTweak> param(new RGYFilterParamTweak());
        param->tweak = params->vpp.tweak;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->vui = vuiInfo;
        param->baseFps = m_encFps;
        param->bOutOverwrite = true;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //deband
    if (vppType == VppType::CL_DEBAND) {
        unique_ptr<RGYFilter> filter(new RGYFilterDeband(m_cl));
        shared_ptr<RGYFilterParamDeband> param(new RGYFilterParamDeband());
        param->deband = params->vpp.deband;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //libplacebo deband
    if (vppType == VppType::CL_LIBPLACEBO_DEBAND) {
        auto filter = std::make_unique<RGYFilterLibplaceboDeband>(m_cl);
        shared_ptr<RGYFilterParamLibplaceboDeband> param(new RGYFilterParamLibplaceboDeband());
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        param->deband = params->vpp.libplacebo_deband;
        param->vk = m_device->vulkan();
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //padding
    if (vppType == VppType::CL_PAD) {
        unique_ptr<RGYFilter> filter(new RGYFilterPad(m_cl));
        shared_ptr<RGYFilterParamPad> param(new RGYFilterParamPad());
        param->pad = params->vpp.pad;
        param->frameIn = inputFrame;
        param->frameOut = inputFrame;
        param->frameOut.width += params->vpp.pad.left + params->vpp.pad.right;
        param->frameOut.height += params->vpp.pad.top + params->vpp.pad.bottom;
        param->encoderCsp = getEncoderCsp(params);
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        auto sts = filter->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        //入力フレーム情報を更新
        inputFrame = param->frameOut;
        m_encFps = param->baseFps;
        //登録
        clfilters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    }
    //overlay
    if (vppType == VppType::CL_OVERLAY) {
        for (const auto& overlay : params->vpp.overlay) {
            unique_ptr<RGYFilter> filter(new RGYFilterOverlay(m_cl));
            shared_ptr<RGYFilterParamOverlay> param(new RGYFilterParamOverlay());
            param->overlay = overlay;
            param->frameIn = inputFrame;
            param->frameOut = inputFrame;
            param->baseFps = m_encFps;
            param->bOutOverwrite = true;
            auto sts = filter->init(param, m_pQSVLog);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            //入力フレーム情報を更新
            inputFrame = param->frameOut;
            m_encFps = param->baseFps;
            //登録
            clfilters.push_back(std::move(filter));
        }
        return RGY_ERR_NONE;
    }

    PrintMes(RGY_LOG_ERROR, _T("Unknown filter type.\n"));
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR CQSVPipeline::createOpenCLCopyFilterForPreVideoMetric() {
    auto [err, outFrameInfo] = GetOutputVideoInfo();
    if (err != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to get output frame info!\n"));
        return err;
    }

    const auto formatOut = videooutputinfo(outFrameInfo->videoPrm.mfx, m_encParams.videoSignalInfo, m_encParams.chromaLocInfo);
    std::unique_ptr<RGYFilter> filterCrop(new RGYFilterCspCrop(m_cl));
    std::shared_ptr<RGYFilterParamCrop> param(new RGYFilterParamCrop());
    param->frameOut = RGYFrameInfo(formatOut.dstWidth, formatOut.dstHeight, formatOut.csp, formatOut.bitdepth, formatOut.picstruct, RGY_MEM_TYPE_GPU);
    param->frameIn = param->frameOut;
    param->frameIn.bitdepth = RGY_CSP_BIT_DEPTH[param->frameIn.csp];
    param->baseFps = m_encFps;
    param->bOutOverwrite = false;
    auto sts = filterCrop->init(param, m_pQSVLog);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    //登録
    std::vector<std::unique_ptr<RGYFilter>> filters;
    filters.push_back(std::move(filterCrop));
    if (m_vpFilters.size() > 0) {
        PrintMes(RGY_LOG_ERROR, _T("Unknown error, not expected that m_vpFilters has size.\n"));
        return RGY_ERR_UNDEFINED_BEHAVIOR;
    }
    m_vpFilters.push_back(VppVilterBlock(filters));
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::InitFilters(sInputParams *inputParam) {
    const bool cropRequired = cropEnabled(inputParam->input.crop)
        && m_pFileReader->getInputCodec() != RGY_CODEC_UNKNOWN;

    RGYFrameInfo inputFrame(inputParam->input.srcWidth, inputParam->input.srcHeight,
        (m_pFileReader->getInputCodec() == RGY_CODEC_UNKNOWN) ? inputParam->input.csp : m_mfxDEC->GetFrameOut().csp,
        (m_pFileReader->getInputCodec() == RGY_CODEC_UNKNOWN) ? inputParam->input.bitdepth : m_mfxDEC->GetFrameOut().bitdepth,
        inputParam->input.picstruct,
        RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED);
    const auto input_sar = rgy_rational<int>(inputParam->input.sar[0], inputParam->input.sar[1]);
    const int croppedWidth = inputFrame.width - inputParam->input.crop.e.left - inputParam->input.crop.e.right;
    const int croppedHeight = inputFrame.height - inputParam->input.crop.e.bottom - inputParam->input.crop.e.up;
    if (!cropRequired) {
        //入力時にcrop済み
        inputFrame.width = croppedWidth;
        inputFrame.height = croppedHeight;
    }

    const bool cspConvRequired = inputFrame.csp != getEncoderCsp(inputParam);

    m_encWidth = croppedWidth;
    m_encHeight = croppedHeight;
    //リサイザの出力すべきサイズ
    int resizeWidth = 0;
    int resizeHeight = 0;
    //指定のリサイズがあればそのサイズに設定する
    if (inputParam->input.dstWidth > 0 && inputParam->input.dstHeight > 0) {
        m_encWidth = resizeWidth = inputParam->input.dstWidth;
        m_encHeight = resizeHeight = inputParam->input.dstHeight;
        if (inputParam->vpp.pad.enable) {
            resizeWidth -= (inputParam->vpp.pad.right + inputParam->vpp.pad.left);
            resizeHeight -= (inputParam->vpp.pad.bottom + inputParam->vpp.pad.top);
        }
    }

    RGY_VPP_RESIZE_TYPE resizeRequired = RGY_VPP_RESIZE_TYPE_NONE;
    if (resizeWidth > 0 && resizeHeight > 0) {
        resizeRequired = getVppResizeType(inputParam->vpp.resize_algo);
        if (resizeRequired == RGY_VPP_RESIZE_TYPE_UNKNOWN) {
            PrintMes(RGY_LOG_ERROR, _T("Unknown resize type.\n"));
            return RGY_ERR_INVALID_VIDEO_PARAM;
        }
    }
    //リサイズアルゴリズムのパラメータはvpp側に設定されているので、設定をvppmfxに転写する
    inputParam->vppmfx.resizeInterp = resize_algo_rgy_to_enc(inputParam->vpp.resize_algo);
    inputParam->vppmfx.resizeMode = resize_mode_rgy_to_enc(inputParam->vpp.resize_mode);
    inputParam->vppmfx.aiSuperRes.enable = inputParam->vpp.resize_algo == RGY_VPP_RESIZE_MFX_AI_SUPRERES;

    //フレームレートのチェック
    if (inputParam->input.fpsN == 0 || inputParam->input.fpsD == 0) {
        PrintMes(RGY_LOG_ERROR, _T("unable to parse fps data.\n"));
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }
    m_encFps = rgy_rational<int>(inputParam->input.fpsN, inputParam->input.fpsD);

    if (inputParam->input.picstruct & RGY_PICSTRUCT_INTERLACED) {
        if (CheckParamList(inputParam->vppmfx.deinterlace, list_deinterlace, "vpp-deinterlace") != RGY_ERR_NONE) {
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
    if (inputParam->vpp.yadif.enable) deinterlacer++;
    if (inputParam->vpp.decomb.enable) deinterlacer++;
    if (deinterlacer >= 2) {
        PrintMes(RGY_LOG_ERROR, _T("Activating 2 or more deinterlacer is not supported.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    //vpp-rffの制約事項
    if (inputParam->vpp.rff.enable) {
        if (trim_active(&m_trimParam)) {
            PrintMes(RGY_LOG_ERROR, _T("vpp-rff cannot be used with trim.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }
    //picStructの設定
    m_encPicstruct = (deinterlacer > 0) ? RGY_PICSTRUCT_FRAME : inputParam->input.picstruct;

    //VUI情報
    auto VuiFiltered = inputParam->input.vui;

    m_encVUI = inputParam->common.out_vui;
    m_encVUI.apply_auto(inputParam->input.vui, m_encHeight);
    m_encVUI.setDescriptPreset();

    if (inputParam->outputCsp == RGY_CHROMAFMT_RGB) {
        m_encVUI.descriptpresent = 1;
        if (m_encVUI.matrix     == RGY_MATRIX_UNSPECIFIED     || m_encVUI.matrix     == RGY_MATRIX_AUTO)     m_encVUI.matrix     = RGY_MATRIX_RGB;
        if (m_encVUI.colorprim  == RGY_PRIM_UNSPECIFIED       || m_encVUI.colorprim  == RGY_PRIM_AUTO)       m_encVUI.colorprim  = RGY_PRIM_BT709;
        if (m_encVUI.transfer   == RGY_TRANSFER_UNSPECIFIED   || m_encVUI.transfer   == RGY_TRANSFER_AUTO)   m_encVUI.transfer   = RGY_TRANSFER_IEC61966_2_1;
        if (m_encVUI.colorrange == RGY_COLORRANGE_UNSPECIFIED || m_encVUI.colorrange == RGY_COLORRANGE_AUTO) m_encVUI.colorrange = RGY_COLORRANGE_FULL;
    }

    m_vpFilters.clear();

    std::vector<VppType> filterPipeline = InitFiltersCreateVppList(inputParam, cspConvRequired, cropRequired, resizeRequired);
    if (filterPipeline.size() == 0) {
        PrintMes(RGY_LOG_DEBUG, _T("No filters required.\n"));
        return RGY_ERR_NONE;
    }
    const auto clfilterCount = std::count_if(filterPipeline.begin(), filterPipeline.end(), [](VppType type) { return getVppFilterType(type) == VppFilterType::FILTER_OPENCL; });
    if (!m_cl && clfilterCount > 0) {
        if (!inputParam->ctrl.enableOpenCL) {
            PrintMes(RGY_LOG_ERROR, _T("OpenCL filter not enabled.\n"));
        } else {
            PrintMes(RGY_LOG_ERROR, _T("OpenCL filter not supported on this platform: %s.\n"), CPU_GEN_STR[m_device->CPUGen()]);
        }
        return RGY_ERR_UNSUPPORTED;
    }
    // blocksize
    const int blocksize = inputParam->codec == RGY_CODEC_HEVC ? 32 : 16;
    //読み込み時のcrop
    sInputCrop *inputCrop = (cropRequired) ? &inputParam->input.crop : nullptr;
    const auto resize = std::make_pair(resizeWidth, resizeHeight);

    std::vector<std::unique_ptr<RGYFilter>> vppOpenCLFilters;
    for (size_t i = 0; i < filterPipeline.size(); i++) {
        const VppFilterType ftype0 = (i >= 1)                      ? getVppFilterType(filterPipeline[i-1]) : VppFilterType::FILTER_NONE;
        const VppFilterType ftype1 =                                 getVppFilterType(filterPipeline[i+0]);
        const VppFilterType ftype2 = (i+1 < filterPipeline.size()) ? getVppFilterType(filterPipeline[i+1]) : VppFilterType::FILTER_NONE;
        if (ftype1 == VppFilterType::FILTER_MFX) {
            auto [err, vppmfx] = AddFilterMFX(inputFrame, m_encFps, filterPipeline[i], &inputParam->vppmfx,
                getEncoderCsp(inputParam), getEncoderBitdepth(inputParam), inputCrop, resize, blocksize);
            inputCrop = nullptr;
            if (err != RGY_ERR_NONE) {
                return err;
            }
            if (vppmfx) {
                m_vpFilters.push_back(VppVilterBlock(vppmfx));
            }
        } else if (ftype1 == VppFilterType::FILTER_OPENCL) {
            if (ftype0 != VppFilterType::FILTER_OPENCL || filterPipeline[i] == VppType::CL_CROP) { // 前のfilterがOpenCLでない場合、変換が必要
                auto filterCrop = std::make_unique<RGYFilterCspCrop>(m_cl);
                shared_ptr<RGYFilterParamCrop> param(new RGYFilterParamCrop());
                param->frameIn = inputFrame;
                param->frameOut = inputFrame;
                param->frameOut.csp = getEncoderCsp(inputParam);
                switch (param->frameOut.csp) { // OpenCLフィルタの内部形式への変換
                case RGY_CSP_NV12: param->frameOut.csp = RGY_CSP_YV12; break;
                case RGY_CSP_P010: param->frameOut.csp = RGY_CSP_YV12_16; break;
                case RGY_CSP_VUYA: param->frameOut.csp = RGY_CSP_YUV444; break;
                case RGY_CSP_Y410: param->frameOut.csp = RGY_CSP_YUV444_16; break;
                case RGY_CSP_Y416: param->frameOut.csp = RGY_CSP_YUV444_16; break;
                case RGY_CSP_MFX_RGB: param->frameOut.csp = RGY_CSP_RGB; break;
                default:
                    break;
                }
                param->frameOut.bitdepth = RGY_CSP_BIT_DEPTH[param->frameOut.csp];
                if (inputCrop) {
                    param->crop = *inputCrop;
                    inputCrop = nullptr;
                }
                param->baseFps = m_encFps;
                param->frameIn.mem_type = RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED;
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
                auto err = AddFilterOpenCL(vppOpenCLFilters, inputFrame, filterPipeline[i], inputParam, inputCrop, resize, VuiFiltered);
                if (err != RGY_ERR_NONE) {
                    return err;
                }
            }
            if (ftype2 != VppFilterType::FILTER_OPENCL) { // 次のfilterがOpenCLでない場合、変換が必要
                std::unique_ptr<RGYFilter> filterCrop(new RGYFilterCspCrop(m_cl));
                std::shared_ptr<RGYFilterParamCrop> param(new RGYFilterParamCrop());
                param->frameIn = inputFrame;
                param->frameOut = inputFrame;
                param->frameOut.csp = getEncoderCsp(inputParam);
                param->frameOut.bitdepth = getEncoderBitdepth(inputParam);
                param->frameIn.mem_type = RGY_MEM_TYPE_GPU;
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
                m_vpFilters.push_back(VppVilterBlock(vppOpenCLFilters));
                vppOpenCLFilters.clear();
            }
        } else {
            PrintMes(RGY_LOG_ERROR, _T("Unsupported vpp filter type.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }

    if (inputParam->vpp.checkPerformance) {
        for (auto& block : m_vpFilters) {
            if (block.type == VppFilterType::FILTER_OPENCL) {
                for (auto& filter : block.vppcl) {
                    filter->setCheckPerformance(inputParam->vpp.checkPerformance);
                }
            }
        }
    }

    m_encWidth  = inputFrame.width;
    m_encHeight = inputFrame.height;
    return RGY_ERR_NONE;
}

#if 0 && (defined(_WIN32) || defined(_WIN64))
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

bool CQSVPipeline::preferD3D11Mode(const sInputParams *inputParam) {
#if defined(_WIN32) || defined(_WIN64)
    if (!check_OS_Win8orLater() || MFX_D3D11_SUPPORT == 0) {
        return false;
    }

    const auto filters = InitFiltersCreateVppList(inputParam, inputParam->vpp.colorspace.convs.size() > 0, true, getVppResizeType(inputParam->vpp.resize_algo));
    const bool clfilterexists = std::find_if(filters.begin(), filters.end(), [](VppType filter) {
        return getVppFilterType(filter) == VppFilterType::FILTER_OPENCL;
    }) != filters.end();
    return clfilterexists;
#else
    return false;
#endif
}

RGY_ERR CQSVPipeline::checkGPUListByEncoder(const sInputParams *prm, std::vector<std::unique_ptr<QSVDevice>>& gpuList) {
    PrintMes(RGY_LOG_DEBUG, _T("Check GPU List by Encoder from %d devices.\n"), (int)gpuList.size());
    if (prm->codec == RGY_CODEC_RAW) {
        return RGY_ERR_NONE;
    }

    //const auto enc_csp = getEncoderCsp(prm);
    const auto enc_bitdepth = getEncoderBitdepth(prm);
    const auto rate_control = prm->rcParam.encMode;
    tstring message;
    for (auto gpu = gpuList.begin(); gpu != gpuList.end(); ) {
        PrintMes(RGY_LOG_DEBUG, _T("Checking GPU #%d (%s) for codec %s.\n"),
            (*gpu)->deviceNum(), (*gpu)->name().c_str(), CodecToStr(prm->codec).c_str());
        const bool lowPower = prm->codec != RGY_CODEC_H264;
        QSVEncFeatures deviceFeature;
        //コーデックのチェック
        if (   !(deviceFeature = (*gpu)->getEncodeFeature(rate_control, prm->codec, lowPower))
            && !(deviceFeature = (*gpu)->getEncodeFeature(rate_control, prm->codec, !lowPower))
            && !(deviceFeature = (*gpu)->getEncodeFeature(MFX_RATECONTROL_CQP, prm->codec, lowPower))
            && !(deviceFeature = (*gpu)->getEncodeFeature(MFX_RATECONTROL_CQP, prm->codec, !lowPower))) {
            message += strsprintf(_T("GPU #%d (%s) does not support %s encoding.\n"),
                (*gpu)->deviceNum(), (*gpu)->name().c_str(), CodecToStr(prm->codec).c_str());
            gpu = gpuList.erase(gpu);
            continue;
        }
        //10bit深度のチェック
        if (enc_bitdepth > 8 && (deviceFeature & ENC_FEATURE_10BIT_DEPTH) != ENC_FEATURE_10BIT_DEPTH) {
            message += strsprintf(_T("GPU #%d (%s) does not support %s %d bit encoding.\n"),
                (*gpu)->deviceNum(), (*gpu)->name().c_str(), CodecToStr(prm->codec).c_str(), enc_bitdepth);
            gpu = gpuList.erase(gpu);
            continue;
        }
        //インタレ保持のチェック
        const bool interlacedEncoding =
            (prm->input.picstruct & RGY_PICSTRUCT_INTERLACED)
            && prm->vppmfx.deinterlace == MFX_DEINTERLACE_NONE
            && !prm->vpp.afs.enable
            && !prm->vpp.nnedi.enable
            && !prm->vpp.yadif.enable
            && !prm->vpp.decomb.enable;
        if (interlacedEncoding && (deviceFeature & ENC_FEATURE_INTERLACE) != ENC_FEATURE_INTERLACE) {
            message += strsprintf(_T("GPU #%d (%s) does not support %s interlaced encoding.\n"),
                (*gpu)->deviceNum(), (*gpu)->name().c_str(), CodecToStr(prm->codec).c_str());
            gpu = gpuList.erase(gpu);
            continue;
        }
        PrintMes(RGY_LOG_DEBUG, _T("GPU #%d (%s) available for %s encode.\n"), (*gpu)->deviceNum(), (*gpu)->name().c_str(), CodecToStr(prm->codec).c_str());
        gpu++;
    }
    PrintMes((gpuList.size() == 0) ? RGY_LOG_ERROR : RGY_LOG_DEBUG, _T("%s\n"), message.c_str());
    if (gpuList.size() == 0) {
        return RGY_ERR_UNSUPPORTED;
    }
    if (gpuList.size() == 1) {
        return RGY_ERR_NONE;
    }
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::deviceAutoSelect(const sInputParams *prm, std::vector<std::unique_ptr<QSVDevice>>& gpuList, const RGYDeviceUsageLockManager *devUsageLock) {
    if (gpuList.size() <= 1) {
        return RGY_ERR_NONE;
    }
    int maxDeviceUsageCount = 1;
    std::vector<std::pair<int, int64_t>> deviceUsage;
    if (gpuList.size() > 1) {
        deviceUsage = m_deviceUsage->getUsage(devUsageLock);
        for (size_t i = 0; i < deviceUsage.size(); i++) {
            maxDeviceUsageCount = std::max(maxDeviceUsageCount, deviceUsage[i].first);
            if (deviceUsage[i].first > 0) {
                PrintMes(RGY_LOG_INFO, _T("Device #%d: %d usage.\n"), i, deviceUsage[i].first);
            }
        }
    }
    const tstring PEPrefix = (prm->ctrl.parallelEnc.isChild()) ? strsprintf(_T("Parallel Enc %d: "), prm->ctrl.parallelEnc.parallelId) : _T("");
#if ENABLE_PERF_COUNTER
    PrintMes(RGY_LOG_DEBUG, _T("Auto select device from %d devices.\n"), (int)gpuList.size());
    bool counterIsIntialized = m_pPerfMonitor->isPerfCounterInitialized();
    for (int i = 0; i < 4 && !counterIsIntialized; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        counterIsIntialized = m_pPerfMonitor->isPerfCounterInitialized();
    }
    if (!counterIsIntialized) {
        return RGY_ERR_NONE;
    }
    while (!m_pPerfMonitor->isPerfCounterRefreshed()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    auto entries = m_pPerfMonitor->GetPerfCountersSystem();
#endif //#if ENABLE_PERF_COUNTER

    std::map<QSVDeviceNum, double> gpuscore;
    for (const auto &gpu : gpuList) {
#if ENABLE_PERF_COUNTER
        auto counters = RGYGPUCounterWinEntries(entries).filter_luid(gpu->luid()).get();
        auto ve_utilization = std::max(
            RGYGPUCounterWinEntries(counters).filter_type(L"codec").sum(),
            RGYGPUCounterWinEntries(counters).filter_type(L"encode").max());
        auto gpu_utilization = std::max(std::max(std::max(
            RGYGPUCounterWinEntries(counters).filter_type(L"cuda").max(), //nvenc
            RGYGPUCounterWinEntries(counters).filter_type(L"compute").max()), //vce-opencl
            RGYGPUCounterWinEntries(counters).filter_type(L"3d").max()), //qsv
            RGYGPUCounterWinEntries(counters).filter_type(L"videoprocessing").max());
        double ve_score = 100.0 * (1.0 - std::pow(std::min(ve_utilization / 100.0, 1.0), 1.0)) * prm->ctrl.gpuSelect.ve;
        double gpu_score = 100.0 * (1.0 - std::pow(std::min(gpu_utilization / 100.0, 1.0), 1.5)) * prm->ctrl.gpuSelect.gpu;
#else
        double ve_score = 0.0;
        double gpu_score = 0.0;
#endif
        double core_score = 0.0;
        double cc_score = 0.0;
        double cl_score = gpu->devInfo() ? 0.0 : maxDeviceUsageCount * -100.0; // openclの初期化に成功したか?
        const int deviceUsageCount = (int)gpu->deviceNum() < (int)deviceUsage.size() ? deviceUsage[(int)gpu->deviceNum()].first : 0;
        double usage_score = 100.0 * (maxDeviceUsageCount - deviceUsageCount) / (double)maxDeviceUsageCount;
        ve_score /= (double)maxDeviceUsageCount;
        gpu_score /= (double)maxDeviceUsageCount;
        core_score /= (double)maxDeviceUsageCount;
        cl_score /= (double)maxDeviceUsageCount;

        gpuscore[gpu->deviceNum()] = usage_score + cc_score + ve_score + gpu_score + core_score + cl_score;
        m_pQSVLog->write(RGY_LOG_DEBUG, RGY_LOGT_CORE_GPU_SELECT, _T("%sGPU #%d (%s) score: %.1f: Use: %.1f, VE %.1f, GPU %.1f, CC %.1f, Core %.1f, CL %.1f.\n"), PEPrefix.c_str(), gpu->deviceNum(), gpu->name().c_str(),
            gpuscore[gpu->deviceNum()], usage_score, ve_score, gpu_score, cc_score, core_score, cl_score);
    }
    std::sort(gpuList.begin(), gpuList.end(), [&](const std::unique_ptr<QSVDevice> &a, const std::unique_ptr<QSVDevice> &b) {
        if (gpuscore.at(a->deviceNum()) != gpuscore.at(b->deviceNum())) {
            return gpuscore.at(a->deviceNum()) > gpuscore.at(b->deviceNum());
        }
        return a->deviceNum() < b->deviceNum();
        });

    PrintMes(RGY_LOG_DEBUG, _T("GPU Priority\n"));
    for (const auto &gpu : gpuList) {
        PrintMes(RGY_LOG_DEBUG, _T("%sGPU #%d (%s): score %.1f\n"), PEPrefix.c_str(), gpu->deviceNum(), gpu->name().c_str(), gpuscore[gpu->deviceNum()]);
    }
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::InitSession(const sInputParams *inputParam, std::vector<std::unique_ptr<QSVDevice>>& deviceList) {
    auto err = RGY_ERR_NONE;
    std::unique_ptr<RGYDeviceUsageLockManager> devUsageLock;
    if (deviceList.size() > 1) {
        m_deviceUsage = std::make_unique<RGYDeviceUsage>();
        devUsageLock = m_deviceUsage->lock(); // ロックは親プロセス側でとる
    }
    if (deviceList.size() == 0) {
        PrintMes(RGY_LOG_DEBUG, _T("No device found for QSV encoding!\n"));
        return RGY_ERR_DEVICE_NOT_FOUND;
    } else if (deviceList.size() == 1) {
        m_device = std::move(deviceList.front());
    } else {
        if ((err = checkGPUListByEncoder(inputParam, deviceList)) != RGY_ERR_NONE) {
            return err;
        }
        if ((err = deviceAutoSelect(inputParam, deviceList, devUsageLock.get())) != RGY_ERR_NONE) {
            return err;
        }
        m_device = std::move(deviceList.front());
        PrintMes(RGY_LOG_DEBUG, _T("InitSession: selected device #%d: %s.\n"), (int)m_device->deviceNum(), m_device->name().c_str());
    }
    if (m_deviceUsage) {
        // 登録を解除するプロセスを起動
        const auto [err_run_proc, child_pid] = m_deviceUsage->startProcessMonitor((int)m_device->deviceNum());
        if (err_run_proc == RGY_ERR_NONE) {
            // プロセスが起動できたら、その子プロセスのIDを登録する
            m_deviceUsage->add((int)m_device->deviceNum(), child_pid, devUsageLock.get());
        }
    }
    devUsageLock.reset();

    //使用できる最大のversionをチェック
    m_device->mfxSession().QueryVersion(&m_mfxVer);
    mfxIMPL impl;
    m_device->mfxSession().QueryIMPL(&impl);
    PrintMes(RGY_LOG_DEBUG, _T("InitSession: mfx lib version: %d.%02d, impl %s\n"), m_mfxVer.Major, m_mfxVer.Minor, MFXImplToStr(impl).c_str());
    return err;
}

RGY_ERR CQSVPipeline::InitVideoQualityMetric(sInputParams *prm) {
    if (prm->common.metric.enabled()) {
        if (!m_pmfxENC) {
            PrintMes(RGY_LOG_WARN, _T("Encoder not enabled, %s calculation will be disabled.\n"), prm->common.metric.enabled_metric().c_str());
            return RGY_ERR_NONE;
        }
        auto [err, outFrameInfo] = GetOutputVideoInfo();
        if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to get output frame info!\n"));
            return err;
        }
        mfxIMPL impl;
        m_device->mfxSession().QueryIMPL(&impl);
        //場合によっては2つ目のhwデコーダを動作させることになる
        //このとき、個別のallocatorを持たないと正常に動作しないので、内部で独自のallocatorを作るようにする
        auto mfxdec = std::make_unique<QSVMfxDec>(m_device->hwdev(), nullptr /*内部で独自のallocatorを作る必要がある*/, m_mfxVer, impl, m_device->memType(), m_sessionParams, m_device->deviceNum(), m_pQSVLog);

        const auto formatOut = videooutputinfo(outFrameInfo->videoPrm.mfx, m_encParams.videoSignalInfo, m_encParams.chromaLocInfo);
        unique_ptr<RGYFilterSsim> filterSsim(new RGYFilterSsim(m_cl));
        shared_ptr<RGYFilterParamSsim> param(new RGYFilterParamSsim());
        param->input = formatOut;
        param->input.srcWidth = m_encWidth;
        param->input.srcHeight = m_encHeight;
        param->bitDepth = prm->outputDepth;
        param->frameIn = RGYFrameInfo(formatOut.dstWidth, formatOut.dstHeight, formatOut.csp, formatOut.bitdepth, formatOut.picstruct, RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED);
        param->frameOut = param->frameIn;
        param->baseFps = m_encFps;
        param->bOutOverwrite = false;
        param->threadParam = prm->ctrl.threadParams.get(RGYThreadType::VIDEO_QUALITY);
        param->mfxDEC = std::move(mfxdec);
        param->metric = prm->common.metric;
        auto sts = filterSsim->init(param, m_pQSVLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_videoQualityMetric = std::move(filterSsim);
        PrintMes(RGY_LOG_DEBUG, _T("Initalized %s calculation.\n"), prm->common.metric.enabled_metric().c_str());
    }
    return RGY_ERR_NONE;
}

//Power throttolingは消費電力削減に有効だが、
//fpsが高い場合やvppフィルタを使用する場合は、速度に悪影響がある場合がある
//そのあたりを適当に考慮し、throttolingのauto/onを自動的に切り替え
RGY_ERR CQSVPipeline::InitPowerThrottoling(sInputParams *pParams) {
    //解像度が低いほど、fpsが出やすい
    int score_resolution = 0;
    const int outputResolution = m_encWidth * m_encHeight;
    if (       outputResolution <= 1024*576) {
        score_resolution += 4;
    } else if (outputResolution <= 1280*720) {
        score_resolution += 3;
    } else if (outputResolution <= 1920*1080) {
        score_resolution += 2;
    } else if (outputResolution <= 2560*1440) {
        score_resolution += 1;
    }
    int score_codec = 0;
    int score_tu = 0;
    if (m_pmfxENC) {
        //MPEG2/H.264エンコードは高速
        switch (pParams->codec) {
        case RGY_CODEC_MPEG2:
        case RGY_CODEC_H264:
            score_codec += 2;
            break;
        case RGY_CODEC_HEVC:
        default:
            score_codec += 0;
            break;
        }
        //TUによる重みづけ
        if (m_encParams.videoPrm.mfx.TargetUsage <= 2) {
            score_tu += 0;
        } else if (m_encParams.videoPrm.mfx.TargetUsage <= 4) {
            score_tu += 2;
        } else {
            score_tu += 3;
        }
    }
    //MFX filterがある場合、RGYThreadType::ENCのthrottolingを有効にするとわずかに遅くなることが多い
    int score_filter = 0;
    const auto filterMFX = std::count_if(m_vpFilters.begin(), m_vpFilters.end(), [](const VppVilterBlock& block) { return block.type == VppFilterType::FILTER_MFX; });
    if (filterMFX > 0) {
        score_filter += 4;
    }
    //OpenCLフィルタは比較的重いのでfpsが低下しやすい
    const auto filterOpenCL = std::count_if(m_vpFilters.begin(), m_vpFilters.end(), [](const VppVilterBlock& block) { return block.type == VppFilterType::FILTER_OPENCL; });
    if (filterOpenCL > 0) {
        score_filter -= 1;
    }
    const int parallelMul = (pParams->ctrl.parallelEnc.isEnabled()) ? pParams->ctrl.parallelEnc.parallelCount : 1;
    const bool speedLimit = pParams->ctrl.procSpeedLimit > 0 && pParams->ctrl.procSpeedLimit <= 240;
    const int score = (speedLimit) ? 0 : (score_codec + score_resolution + score_tu + score_filter) * parallelMul;

    //一定以上のスコアなら、throttolingをAuto、それ以外はthrottolingを有効にして消費電力を削減
    const int score_threshold = 7;
    const auto mode = (score >= score_threshold) ? RGYThreadPowerThrottlingMode::Auto : RGYThreadPowerThrottlingMode::Enabled;
    PrintMes(RGY_LOG_DEBUG, _T("selected mode %s : score %d: codec %d, resolution %d, tu %d, filter %d, speed limit %s, parallelMul: %d.\n"),
        rgy_thread_power_throttoling_mode_to_str(mode), score, score_codec, score_resolution, score_tu, score_filter, speedLimit ? _T("on") : _T("off"), parallelMul);

    if (pParams->ctrl.parallelEnc.isEnabled()) {
        // 並列エンコード時には音声スレッドと出力スレッドが重要なので、throttolingを有効にはならないように
        auto& target = pParams->ctrl.threadParams.get(RGYThreadType::AUDIO);
        if (target.throttling == RGYThreadPowerThrottlingMode::Unset) {
            target.throttling = RGYThreadPowerThrottlingMode::Auto;
        }
        target = pParams->ctrl.threadParams.get(RGYThreadType::OUTPUT);
        if (target.throttling == RGYThreadPowerThrottlingMode::Unset) {
            target.throttling = RGYThreadPowerThrottlingMode::Auto;
        }
    }

    //Unsetのままの設定について自動決定したモードを適用
    for (int i = (int)RGYThreadType::ALL + 1; i < (int)RGYThreadType::END; i++) {
        auto& target = pParams->ctrl.threadParams.get((RGYThreadType)i);
        if (target.throttling == RGYThreadPowerThrottlingMode::Unset) {
            target.throttling = mode;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::InitAvoidIdleClock(const sInputParams *pParams) {
    if (!m_cl) return RGY_ERR_NONE;
    if (pParams->ctrl.avoidIdleClock.mode == RGYParamAvoidIdleClockMode::Disabled) {
        PrintMes(RGY_LOG_DEBUG, _T("avoid Idle clock is disabled.\n"));
        return RGY_ERR_NONE;
    }
    if (pParams->ctrl.avoidIdleClock.mode == RGYParamAvoidIdleClockMode::Auto) {
        // OpenCLフィルタが使用されている場合
        if (std::count_if(m_vpFilters.begin(), m_vpFilters.end(), [](const VppVilterBlock& block) { return block.type == VppFilterType::FILTER_OPENCL; }) > 0) {
            PrintMes(RGY_LOG_DEBUG, _T("OpenCL filter is used, avoid Idle clock is disabled.\n"));
            return RGY_ERR_NONE;
        }

        // 内蔵GPUの場合
        if (m_device->adapterType() == MFX_MEDIA_INTEGRATED) {
            PrintMes(RGY_LOG_DEBUG, _T("Integrated GPU is used, avoid Idle clock is disabled.\n"));
            return RGY_ERR_NONE;
        }

        // max-procfpsを使用している場合
        if (pParams->ctrl.procSpeedLimit > 0) {
            PrintMes(RGY_LOG_DEBUG, _T("max-procfps is used, avoid Idle clock is disabled.\n"));
            return RGY_ERR_NONE;
        }

        if (pParams->codec != RGY_CODEC_RAW) { // エンコードする場合
            // PGモードが使用されている場合
            if (m_encParams.videoPrm.mfx.LowPower != MFX_CODINGOPTION_ON) {
                PrintMes(RGY_LOG_DEBUG, _T("PG mode is used, avoid Idle clock is disabled.\n"));
                return RGY_ERR_NONE;
            }

            // lowlatencyモードが使用されている場合
            if (pParams->ctrl.lowLatency) {
                PrintMes(RGY_LOG_DEBUG, _T("low latency is used, avoid Idle clock is disabled.\n"));
                return RGY_ERR_NONE;
            }
        }
    }

    PrintMes(RGY_LOG_DEBUG, _T("Enable avoid Idle clock, target load %.2f.\n"), pParams->ctrl.avoidIdleClock.loadPercent);
    m_dummyLoad = std::make_unique<RGYDummyLoadCL>(m_cl);
    m_dummyLoad->run(pParams->ctrl.avoidIdleClock.loadPercent, m_pQSVLog);
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::InitLog(sInputParams *pParams) {
    //ログの初期化
    m_pQSVLog.reset(new RGYLog(pParams->ctrl.logfile.c_str(), pParams->ctrl.loglevel, pParams->ctrl.logAddTime));
    if (pParams->ctrl.parallelEnc.isChild() && pParams->ctrl.parallelEnc.sendData) {
        m_pQSVLog->setLock(pParams->ctrl.parallelEnc.sendData->logMutex);
    }
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
        inputParam->ctrl.threadParams.get(RGYThreadType::PERF_MONITOR),
        m_pQSVLog, &perfMonitorPrm)) {
        PrintMes(RGY_LOG_WARN, _T("Failed to initialize performance monitor, disabled.\n"));
        m_pPerfMonitor.reset();
    }
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::InitParallelEncode(sInputParams *inputParam, const int maxEncoders) {
    if (!inputParam->ctrl.parallelEnc.isEnabled()) {
        return RGY_ERR_NONE;
    }
    const bool isChild = inputParam->ctrl.parallelEnc.isChild();
    auto [sts, errmes] = RGYParallelEnc::isParallelEncPossible(inputParam, m_pFileReader.get());
    if (sts != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_WARN, _T("%s"), errmes);
        inputParam->ctrl.parallelEnc.parallelCount = 0;
        inputParam->ctrl.parallelEnc.parallelId = -1;
        return (isChild) ? sts : RGY_ERR_NONE; // 子スレッド側でエラーが起こった場合はエラー、親の場合は正常終了(並列動作を無効化して継続)を返す
    }
    // 並列処理が有効の場合、メインスレッドではエンコードは行わないので、m_deviceUsageは解放する
    if (inputParam->ctrl.parallelEnc.isParent() && m_deviceUsage) {
        m_deviceUsage->close();
    }
    if (inputParam->ctrl.parallelEnc.isParent()) {
        // とんでもなく大きい値にする人がいそうなので、適当に制限する
        const int maxParallelCount = std::max(4, maxEncoders * 2);
        if (inputParam->ctrl.parallelEnc.parallelCount < 0) {
            inputParam->ctrl.parallelEnc.parallelCount = maxEncoders;
            PrintMes(RGY_LOG_DEBUG, _T("parallelCount set to %d\n"), inputParam->ctrl.parallelEnc.parallelCount);
        } else if (inputParam->ctrl.parallelEnc.parallelCount > maxParallelCount) {
            inputParam->ctrl.parallelEnc.parallelCount = maxParallelCount;
            PrintMes(RGY_LOG_WARN, _T("Parallel count limited to %d\n"), inputParam->ctrl.parallelEnc.parallelCount);
        }
        if (inputParam->ctrl.parallelEnc.parallelCount <= 1) { // 並列数が1以下ならparallelを無効化
            inputParam->ctrl.parallelEnc.parallelCount = 0;
            inputParam->ctrl.parallelEnc.parallelId = -1;
            PrintMes(RGY_LOG_DEBUG, _T("Parallel encoding disabled, as parallel count id set to %d.\n"), inputParam->ctrl.parallelEnc.parallelCount);
            return RGY_ERR_NONE;
        }
    }
    m_parallelEnc = std::make_unique<RGYParallelEnc>(m_pQSVLog);
    if ((sts = m_parallelEnc->parallelRun(inputParam, m_pFileReader.get(), m_outputTimebase, inputParam->ctrl.parallelEnc.parallelCount > maxEncoders, m_pStatus.get())) != RGY_ERR_NONE) {
        if (inputParam->ctrl.parallelEnc.isChild()) {
            return sts; // 子スレッド側でエラーが起こった場合はエラー
        }
        // うまくいかなかった場合、並列処理を無効化して続行する
        PrintMes(RGY_LOG_WARN, _T("Failed to initialize parallel encoding, disabled.\n"));
        m_parallelEnc.reset();
        // m_deviceUsageはいったん解放したので、登録を再追加
        if (m_deviceUsage) {
            m_deviceUsage = std::make_unique<RGYDeviceUsage>();
            auto devUsageLock = m_deviceUsage->lock(); // ロックは親プロセス側でとる
            // 登録を解除するプロセスを起動
            const auto [err_run_proc, child_pid] = m_deviceUsage->startProcessMonitor((int)m_device->deviceNum());
            if (err_run_proc == RGY_ERR_NONE) {
                // プロセスが起動できたら、その子プロセスのIDを登録する
                m_deviceUsage->add((int)m_device->deviceNum(), child_pid, devUsageLock.get());
            }
        }
        return RGY_ERR_NONE; // 親の場合は正常終了(並列動作を無効化して継続)を返す
    }
    if (inputParam->ctrl.parallelEnc.isChild()) {
        m_pQSVLog->write(RGY_LOG_DEBUG, RGY_LOGT_CORE_GPU_SELECT, _T("Parallel Enc %d: Selected GPU #%d (%s)\n"), inputParam->ctrl.parallelEnc.parallelId, m_device->deviceNum(), m_device->name().c_str());
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
        m_pPerfMonitor->SetThreadHandles((HANDLE)NULL, thInput, thOutput, thAudProc, thAudEnc);
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

#if ENABLE_VULKAN
    if (pParams->ctrl.enableVulkan == RGYParamInitVulkan::TargetVendor) {
        setenv("VK_LOADER_DRIVERS_SELECT", "*intel*", 1);
    }
#endif

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

    m_pPerfMonitor = std::make_unique<CPerfMonitor>();
#if ENABLE_PERF_COUNTER
    m_pPerfMonitor->runCounterThread();
#endif

    if (const auto affinity = pParams->ctrl.threadParams.get(RGYThreadType::PROCESS).affinity; affinity.mode != RGYThreadAffinityMode::ALL) {
        SetProcessAffinityMask(GetCurrentProcess(), affinity.getMask());
        PrintMes(RGY_LOG_DEBUG, _T("Set Process Affinity Mask: %s (0x%llx).\n"), affinity.to_string().c_str(), affinity.getMask());
    }
    if (const auto priority = pParams->ctrl.threadParams.get(RGYThreadType::PROCESS).priority; priority != RGYThreadPriority::Normal) {
        SetPriorityClass(GetCurrentProcess(), pParams->ctrl.threadParams.get(RGYThreadType::PROCESS).getPriorityCalss());
        PrintMes(RGY_LOG_DEBUG, _T("Set Process priority: %s.\n"), rgy_thread_priority_mode_to_str(priority));
    }

    RGYParamThread threadParamThrottleDsiabled;
    threadParamThrottleDsiabled.set(pParams->ctrl.threadParams.get(RGYThreadType::PROCESS).affinity, RGYThreadPriority::Normal, RGYThreadPowerThrottlingMode::Disabled);
    threadParamThrottleDsiabled.apply(GetCurrentThread());

#if defined(_WIN32) || defined(_WIN64)
    if (!pParams->bDisableTimerPeriodTuning) {
        m_bTimerPeriodTuning = true;
        timeBeginPeriod(1);
        PrintMes(RGY_LOG_DEBUG, _T("timeBeginPeriod(1)\n"));
    }
#endif //#if defined(_WIN32) || defined(_WIN64)

    m_sessionParams.threads = pParams->nSessionThreads;
    m_sessionParams.deviceCopy = pParams->gpuCopy;
    m_nAVSyncMode = pParams->common.AVSyncMode;

    DeviceCodecCsp HWDecCodecCsp;
    auto deviceInfoCache = std::make_shared<QSVDeviceInfoCache>();
    if ((sts = deviceInfoCache->loadCacheFile()) != RGY_ERR_NONE) {
        if (sts == RGY_ERR_FILE_OPEN) { // ファイルは存在するが開けない
            deviceInfoCache.reset(); // キャッシュの存在を無視して進める
        }
    } else {
        HWDecCodecCsp = deviceInfoCache->getDeviceDecCodecCsp();
        PrintMes(RGY_LOG_DEBUG, _T("HW dec codec csp support read from cache file.\n"));
    }
    std::vector<std::unique_ptr<QSVDevice>> deviceList;
    auto getDevIdName = [&deviceList]() {
        std::map<int, std::string> devIdName;
        for (const auto& dev : deviceList) {
            devIdName[(int)dev->deviceNum()] = tchar_to_string(dev->name());
        }
        return devIdName;
    };
    if (deviceInfoCache
        && (deviceInfoCache->getDeviceIds().size() == 0
        || (pParams->device == QSVDeviceNum::AUTO && deviceInfoCache->getDeviceIds().size() != HWDecCodecCsp.size())
        || (pParams->device != QSVDeviceNum::AUTO && 
            std::find_if(HWDecCodecCsp.begin(), HWDecCodecCsp.end(), [dev = (int)pParams->device](const std::pair<int, CodecCsp>& data) { return data.first == dev; }) == HWDecCodecCsp.end()))) {
        deviceList = getDeviceList((deviceInfoCache->getDeviceIds().size() == 0) ? QSVDeviceNum::AUTO : pParams->device, pParams->ctrl.enableOpenCL, pParams->ctrl.enableVulkan, pParams->memType, m_sessionParams, deviceInfoCache, m_pQSVLog);
        if (deviceList.size() == 0) {
            PrintMes(RGY_LOG_DEBUG, _T("No device found for QSV encoding!\n"));
            return RGY_ERR_DEVICE_NOT_FOUND;
        }
        HWDecCodecCsp = getHWDecCodecCsp(pParams->ctrl.skipHWDecodeCheck, deviceList);
        deviceInfoCache->setDecCodecCsp(getDevIdName(), HWDecCodecCsp);
        deviceInfoCache->saveCacheFile();
        PrintMes(RGY_LOG_DEBUG, _T("HW dec codec csp support saved to cache file.\n"));
        if (pParams->device != QSVDeviceNum::AUTO) {
            for (auto it = deviceList.begin(); it != deviceList.end();) {
                if ((*it)->deviceNum() != pParams->device) {
                    it = deviceList.erase(it);
                } else {
                    it++;
                }
            }
        }
    }

    auto input_ret = std::async(std::launch::async, [&] {
        threadParamThrottleDsiabled.apply(GetCurrentThread());
        auto sts = InitInput(pParams, HWDecCodecCsp);
        if (sts == RGY_ERR_NONE) {
            pParams->applyDOVIProfile(m_pFileReader->getInputDOVIProfile());
        }
        return sts;
    });

    if (deviceList.size() == 0) {
        deviceList = getDeviceList(pParams->device, pParams->ctrl.enableOpenCL, pParams->ctrl.enableVulkan, pParams->memType, m_sessionParams, deviceInfoCache, m_pQSVLog);
        if (deviceList.size() == 0) {
            PrintMes(RGY_LOG_INFO, _T("No device found for QSV encoding!\n"));
            return RGY_ERR_DEVICE_NOT_FOUND;
        }
        if (deviceInfoCache) {
            deviceInfoCache->setDeviceIds(getDevIdName());
        }
    }

    m_devNames.clear();
    for (const auto& dev : deviceList) {
        m_devNames.push_back(dev->name());
    }

    sts = InitSession(pParams, deviceList);
    RGY_ERR(sts, _T("Failed to initialize encode session."));
    PrintMes(RGY_LOG_DEBUG, _T("InitSession: Success.\n"));

    sts = InitOpenCL(pParams->ctrl.enableOpenCL, pParams->ctrl.parallelEnc.isParent() ? 1 : pParams->ctrl.openclBuildThreads, pParams->vpp.checkPerformance);
    if (sts < RGY_ERR_NONE) return sts;
    PrintMes(RGY_LOG_DEBUG, _T("InitOpenCL: Success.\n"));

    sts = input_ret.get();
    if (sts < RGY_ERR_NONE) return sts;
    PrintMes(RGY_LOG_DEBUG, _T("InitInput: Success.\n"));

    // 並列動作の子は読み込みが終了したらすぐに並列動作を呼び出し
    // ただし、親-子間のデータやり取りを少し遅らせる場合(delayChildSync)は親と同じタイミングで処理する
    if (pParams->ctrl.parallelEnc.isChild() && !pParams->ctrl.parallelEnc.delayChildSync) {
        sts = InitParallelEncode(pParams, (int)m_devNames.size());
        if (sts < RGY_ERR_NONE) return sts;
    }

    sts = CheckParam(pParams);
    if (sts != RGY_ERR_NONE) return sts;
    PrintMes(RGY_LOG_DEBUG, _T("CheckParam: Success.\n"));

    sts = InitMfxDecParams();
    if (sts < RGY_ERR_NONE) return sts;
    PrintMes(RGY_LOG_DEBUG, _T("InitMfxDecParams: Success.\n"));

    sts = InitFilters(pParams);
    if (sts < RGY_ERR_NONE) return sts;
    PrintMes(RGY_LOG_DEBUG, _T("InitFilters: Success.\n"));

    sts = InitMfxEncodeParams(pParams, deviceList);
    if (sts < RGY_ERR_NONE) return sts;
    PrintMes(RGY_LOG_DEBUG, _T("InitMfxEncodeParams: Success.\n"));

    deviceList.clear();
    if (deviceInfoCache) deviceInfoCache->updateCacheFile();

    sts = InitPowerThrottoling(pParams);
    if (sts < RGY_ERR_NONE) return sts;

    sts = InitChapters(pParams);
    if (sts < RGY_ERR_NONE) return sts;

    sts = InitPerfMonitor(pParams);
    if (sts < RGY_ERR_NONE) return sts;

    // 親はエンコード設定が完了してから並列動作を呼び出し
    if (pParams->ctrl.parallelEnc.isParent() || (pParams->ctrl.parallelEnc.isChild() && pParams->ctrl.parallelEnc.delayChildSync)) {
        sts = InitParallelEncode(pParams, (int)m_devNames.size());
        if (sts < RGY_ERR_NONE) return sts;
    }

    m_encTimestamp = std::make_unique<RGYTimestamp>(pParams->common.timestampPassThrough, pParams->ctrl.parallelEnc.isParent() /*durationは子エンコーダで修正済み*/);

    sts = InitOutput(pParams);
    if (sts < RGY_ERR_NONE) return sts;

    m_nProcSpeedLimit = pParams->ctrl.procSpeedLimit;
    m_taskPerfMonitor = pParams->ctrl.taskPerfMonitor;
    m_nAsyncDepth = clamp_param_int((pParams->ctrl.lowLatency) ? 1 : pParams->nAsyncDepth, 0, QSV_ASYNC_DEPTH_MAX, _T("async-depth"));
    if (m_nAsyncDepth == 0) {
        m_nAsyncDepth = QSV_DEFAULT_ASYNC_DEPTH;
        PrintMes(RGY_LOG_DEBUG, _T("async depth automatically set to %d\n"), m_nAsyncDepth);
    }
    if (pParams->ctrl.lowLatency) {
        pParams->bDisableTimerPeriodTuning = false;
    }

    const int nPipelineElements = !!m_mfxDEC + (int)m_vpFilters.size() + !!m_pmfxENC;
    if (nPipelineElements == 0) {
        PrintMes(RGY_LOG_ERROR, _T("None of the pipeline element (DEC,VPP,ENC) are activated!\n"));
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }
    PrintMes(RGY_LOG_DEBUG, _T("pipeline element count: %d\n"), nPipelineElements);

    sts = InitVideoQualityMetric(pParams);
    if (sts < RGY_ERR_NONE) return sts;

    if ((sts = InitAvoidIdleClock(pParams)) != RGY_ERR_NONE) {
        return sts;
    }
    if ((sts = ResetMFXComponents(pParams)) != RGY_ERR_NONE) {
        return sts;
    }

    {
        const auto& threadParam = pParams->ctrl.threadParams.get(RGYThreadType::MAIN);
        threadParam.apply(GetCurrentThread());
        PrintMes(RGY_LOG_DEBUG, _T("Set main thread param: %s.\n"), threadParam.desc().c_str());
    }
    {
        const auto& threadParam = pParams->ctrl.threadParams.get(RGYThreadType::ENC);
#if defined(_WIN32) || defined(_WIN64)
#ifdef _M_IX86
        const TCHAR* dll_mfx_platform = _T("libmfxhw32.dll");
        const TCHAR* dll_vpl_platform = _T("libmfx32-gen.dll");
#else
        const TCHAR* dll_mfx_platform = _T("libmfxhw64.dll");
        const TCHAR* dll_vpl_platform = _T("libmfx64-gen.dll");
#endif
        const TCHAR* target_dll = check_lib_version(m_mfxVer, MFX_LIB_VERSION_2_0) ? dll_vpl_platform : dll_mfx_platform;
        if (const auto affinity = threadParam.affinity; affinity.mode != RGYThreadAffinityMode::ALL) {
            SetThreadAffinityForModule(GetCurrentProcessId(), target_dll, affinity.getMask());
            PrintMes(RGY_LOG_DEBUG, _T("Set mfx thread Affinity Mask: %s (0x%llx).\n"), affinity.to_string().c_str(), affinity.getMask());
        }
        if (threadParam.priority != RGYThreadPriority::Normal) {
            SetThreadPriorityForModule(GetCurrentProcessId(), target_dll, threadParam.priority);
            PrintMes(RGY_LOG_DEBUG, _T("Set mfx thread priority: %s.\n"), threadParam.to_string(RGYParamThreadType::priority).c_str());
        }
        if (threadParam.throttling != RGYThreadPowerThrottlingMode::Auto) {
            SetThreadPowerThrottolingModeForModule(GetCurrentProcessId(), target_dll, threadParam.throttling);
            PrintMes(RGY_LOG_DEBUG, _T("Set mfx thread throttling mode: %s.\n"), threadParam.to_string(RGYParamThreadType::throttling).c_str());
        }
#endif //#if defined(_WIN32) || defined(_WIN64)
    }
    if ((sts = SetPerfMonitorThreadHandles()) != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to set thread handles to perf monitor!\n"));
        return sts;
    }
    return RGY_ERR_NONE;
}

void CQSVPipeline::Close() {
    // MFXのコンポーネントをm_pipelineTasksの解放(フレームの解放)前に実施する
    PrintMes(RGY_LOG_DEBUG, _T("Clear vpp filters...\n"));
    m_videoQualityMetric.reset();
    m_vpFilters.clear();
    PrintMes(RGY_LOG_DEBUG, _T("Closing m_pmfxDEC/ENC/VPP...\n"));
    m_mfxDEC.reset();
    m_pmfxENC.reset();
    m_mfxVPP.clear();
    //この中でフレームの解放がなされる
    PrintMes(RGY_LOG_DEBUG, _T("Clear pipeline tasks and allocated frames...\n"));
    m_pipelineTasks.clear();

    m_dummyLoad.reset();

    PrintMes(RGY_LOG_DEBUG, _T("Closing enc status...\n"));
    m_pStatus.reset();

#if ENABLE_MVC_ENCODING
    FreeMVCSeqDesc();
#endif

    m_DecInputBitstream.clear();

    PrintMes(RGY_LOG_DEBUG, _T("Closing device...\n"));
    m_device.reset();
    m_deviceUsage.reset();
    m_parallelEnc.reset();

    m_trimParam.list.clear();
    m_trimParam.offset = 0;

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
    m_poolFrame.reset();
    m_poolPkt.reset();
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

    m_encTimestamp.reset();
    m_dovirpu.reset();
    m_hdr10plusMetadataCopy = false;
    m_hdr10plus.reset();
    m_hdrseiOut.reset();

    m_sessionParams.threads = 0;
    m_sessionParams.deviceCopy = false;
    m_pAbortByUser = nullptr;
    m_nAVSyncMode = RGY_AVSYNC_AUTO;
    m_nProcSpeedLimit = 0;
    m_taskPerfMonitor = false;
#if ENABLE_AVSW_READER
    av_qsv_log_free();
#endif //#if ENABLE_AVSW_READER
    PrintMes(RGY_LOG_DEBUG, _T("Closed pipeline.\n"));
    if (m_pQSVLog.get() != nullptr) {
        m_pQSVLog->writeFileFooter();
        m_pQSVLog.reset();
    }
}

RGYParamLogLevel CQSVPipeline::logTemporarilyIgnoreErrorMes() {
    //MediaSDK内のエラーをRGY_LOG_DEBUG以下の時以外には一時的に無視するようにする。
    //RGY_LOG_DEBUG以下の時にも、「無視できるエラーが発生するかもしれない」ことをログに残す。
    const auto log_level = m_pQSVLog->getLogLevelAll();
    if (   log_level.get(RGY_LOGT_CORE) >= RGY_LOG_MORE
        || log_level.get(RGY_LOGT_DEC)  >= RGY_LOG_MORE
        || log_level.get(RGY_LOGT_VPP)  >= RGY_LOG_MORE
        || log_level.get(RGY_LOGT_DEV)  >= RGY_LOG_MORE) {
        m_pQSVLog->setLogLevel(RGY_LOG_QUIET, RGY_LOGT_ALL); //一時的にエラーを無視
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
    m_prmSetIn = m_encParams;
    auto sts = err_to_rgy(m_pmfxENC->Init(&m_encParams.videoPrm));
    m_pQSVLog->setLogLevelAll(log_level);
    if (sts == RGY_WRN_PARTIAL_ACCELERATION) {
        PrintMes(RGY_LOG_WARN, _T("partial acceleration on Encoding.\n"));
        sts = RGY_ERR_NONE;
    }
    RGY_ERR(sts, _T("Failed to initialize encoder."));
    PrintMes(RGY_LOG_DEBUG, _T("Encoder initialized.\n"));
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::InitMfxVpp() {
    for (auto& filterBlock : m_vpFilters) {
        if (filterBlock.type == VppFilterType::FILTER_MFX) {
            auto err = filterBlock.vppmfx->Init();
            if (err < RGY_ERR_NONE) { //RGY_WRN_xxx ( > 0) は無視する
                return err;
            }
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR CQSVPipeline::InitMfxDec() {
    if (!m_mfxDEC) {
        return RGY_ERR_NONE;
    }
    const auto log_level = logTemporarilyIgnoreErrorMes();
    auto sts = m_mfxDEC->Init();
    m_pQSVLog->setLogLevelAll(log_level);
    if (sts == RGY_WRN_PARTIAL_ACCELERATION) {
        PrintMes(RGY_LOG_WARN, _T("partial acceleration on decoding.\n"));
        sts = RGY_ERR_NONE;
    }
    RGY_ERR(sts, _T("Failed to initialize decoder.\n"));
    PrintMes(RGY_LOG_DEBUG, _T("Dec initialized.\n"));
    return RGY_ERR_NONE;
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

    if (m_mfxDEC) {
        err = m_mfxDEC->Close();
        RGY_IGNORE_STS(err, RGY_ERR_NOT_INITIALIZED);
        RGY_ERR(err, _T("Failed to reset decoder (fail on closing)."));
        PrintMes(RGY_LOG_DEBUG, _T("ResetMFXComponents: Dec closed.\n"));
    }

    // free allocated frames
    //DeleteFrames();
    //PrintMes(RGY_LOG_DEBUG, _T("ResetMFXComponents: Frames deleted.\n"));

    if ((err = CreatePipeline(pParams)) != RGY_ERR_NONE) {
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

RGY_ERR CQSVPipeline::Run() {
    return RunEncode2();
}

bool CQSVPipeline::VppAfsRffAware() const {
    //vpp-afsのrffが使用されているか
    bool vpp_afs_rff_aware = false;
    for (const auto& filter_block : m_vpFilters) {
        if (filter_block.type == VppFilterType::FILTER_OPENCL) {
            const auto vpp_afs_filter = std::find_if(filter_block.vppcl.begin(), filter_block.vppcl.end(),
                [](const unique_ptr<RGYFilter>& filter) { return typeid(*filter) == typeid(RGYFilterAfs); });
            if (vpp_afs_filter == filter_block.vppcl.end()) continue;
            auto afs_prm = reinterpret_cast<const RGYFilterParamAfs *>((*vpp_afs_filter)->GetFilterParam());
            if (afs_prm != nullptr) {
                vpp_afs_rff_aware |= afs_prm->afs.rff;
            }
        }
    }
    return vpp_afs_rff_aware;
}

RGY_ERR CQSVPipeline::CreatePipeline(const sInputParams* prm) {
    m_pipelineTasks.clear();

    if (m_parallelEnc && m_parallelEnc->id() < 0) {
        // 親プロセスの子プロセスのデータ回収用
        std::unique_ptr<PipelineTaskAudio> taskAudio;
        if (m_pFileWriterListAudio.size() > 0) {
            taskAudio = std::make_unique<PipelineTaskAudio>(m_pFileReader.get(), m_AudioReaders, m_pFileWriterListAudio, m_vpFilters, 0, m_mfxVer, m_pQSVLog);
        }
        const auto encOutputTimebase = (ENCODER_QSV) ? to_rgy(HW_NATIVE_TIMEBASE) : m_outputTimebase;
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskParallelEncBitstream>(m_pFileReader.get(), m_encTimestamp.get(), m_timecode.get(), m_parallelEnc.get(), m_pStatus.get(), m_encFps, encOutputTimebase, taskAudio, 0, m_mfxVer, m_pQSVLog));
        return RGY_ERR_NONE;
    }

    // 並列処理時用の終了時刻 (この時刻は含まないようにする) -1の場合は制限なし(最後まで)
    const auto parallelEncEndPts = (m_parallelEnc) ? m_parallelEnc->getVideoEndKeyPts() : -1ll;
    if (m_pFileReader->getInputCodec() == RGY_CODEC_UNKNOWN) {
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskInput>(&m_device->mfxSession(), m_device->allocator(), parallelEncEndPts, 0, m_pFileReader.get(), m_mfxVer, m_cl, m_pQSVLog));
    } else {
        auto err = err_to_rgy(m_device->mfxSession().JoinSession(m_mfxDEC->GetSession()));
        if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to join mfx vpp session: %s.\n"), get_err_mes(err));
            return err;
        }
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskMFXDecode>(&m_device->mfxSession(), 1, m_mfxDEC->mfxdec(), m_mfxDEC->mfxparams(), m_mfxDEC->skipAV1C(), parallelEncEndPts, m_pFileReader.get(), m_mfxVer, m_pQSVLog));
    }
    if (m_pFileWriterListAudio.size() > 0) {
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskAudio>(m_pFileReader.get(), m_AudioReaders, m_pFileWriterListAudio, m_vpFilters, 0, m_mfxVer, m_pQSVLog));
    }

    const int64_t outFrameDuration = std::max<int64_t>(1, rational_rescale(1, m_inputFps.inv(), m_outputTimebase)); //固定fpsを仮定した時の1フレームのduration (スケール: m_outputTimebase)
    const auto inputFrameInfo = m_pFileReader->GetInputFrameInfo();
    const auto inputFpsTimebase = rgy_rational<int>((int)inputFrameInfo.fpsD, (int)inputFrameInfo.fpsN);
    const auto srcTimebase = (m_pFileReader->getInputTimebase().n() > 0 && m_pFileReader->getInputTimebase().is_valid()) ? m_pFileReader->getInputTimebase() : inputFpsTimebase;
    if (m_trimParam.list.size() > 0 || prm->common.seekToSec > 0.0f || m_parallelEnc) {
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskTrim>(m_trimParam, m_pFileReader.get(), m_parallelEnc.get(), srcTimebase, 0, m_mfxVer, m_pQSVLog));
    }
    m_pipelineTasks.push_back(std::make_unique<PipelineTaskCheckPTS>(&m_device->mfxSession(), srcTimebase, m_outputTimebase, outFrameDuration, m_nAVSyncMode, m_timestampPassThrough, VppAfsRffAware() && m_pFileReader->rffAware(), m_mfxVer, m_pQSVLog));

    for (auto& filterBlock : m_vpFilters) {
        if (filterBlock.type == VppFilterType::FILTER_MFX) {
            auto err = err_to_rgy(m_device->mfxSession().JoinSession(filterBlock.vppmfx->GetSession()));
            if (err != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to join mfx vpp session: %s.\n"), get_err_mes(err));
                return err;
            }
            m_pipelineTasks.push_back(std::make_unique<PipelineTaskMFXVpp>(&m_device->mfxSession(), 1, filterBlock.vppmfx.get(), filterBlock.vppmfx->mfxparams(), filterBlock.vppmfx->mfxver(), m_outputTimebase, m_timestampPassThrough, m_pQSVLog));
        } else if (filterBlock.type == VppFilterType::FILTER_OPENCL) {
            if (!m_cl) {
                PrintMes(RGY_LOG_ERROR, _T("OpenCL not enabled, OpenCL filters cannot be used.\n"), CPU_GEN_STR[m_device->CPUGen()]);
                return RGY_ERR_UNSUPPORTED;
            }
            m_pipelineTasks.push_back(std::make_unique<PipelineTaskOpenCL>(filterBlock.vppcl, nullptr, m_cl, m_device->memType(), m_device->allocator(), &m_device->mfxSession(), 1, m_pQSVLog));
        } else {
            PrintMes(RGY_LOG_ERROR, _T("Unknown filter type.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
    }

    if (m_videoQualityMetric) {
        int prevtask = -1;
        for (int itask = (int)m_pipelineTasks.size() - 1; itask >= 0; itask--) {
            if (!m_pipelineTasks[itask]->isPassThrough()) {
                prevtask = itask;
                break;
            }
        }
        if (m_pipelineTasks[prevtask]->taskType() == PipelineTaskType::INPUT) {
            //inputと直接つながる場合はうまく処理できなくなる(うまく同期がとれない)
            //そこで、CopyのOpenCLフィルタを挟んでその中で処理する
            auto err = createOpenCLCopyFilterForPreVideoMetric();
            if (err != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to join mfx vpp session: %s.\n"), get_err_mes(err));
                return err;
            } else if (m_vpFilters.size() != 1) {
                PrintMes(RGY_LOG_ERROR, _T("m_vpFilters.size() != 1.\n"));
                return RGY_ERR_UNDEFINED_BEHAVIOR;
            }
            m_pipelineTasks.push_back(std::make_unique<PipelineTaskOpenCL>(m_vpFilters.front().vppcl, m_videoQualityMetric.get(), m_cl, m_device->memType(), m_device->allocator(), &m_device->mfxSession(), 1, m_pQSVLog));
        } else if (m_pipelineTasks[prevtask]->taskType() == PipelineTaskType::OPENCL) {
            auto taskOpenCL = dynamic_cast<PipelineTaskOpenCL*>(m_pipelineTasks[prevtask].get());
            if (taskOpenCL == nullptr) {
                PrintMes(RGY_LOG_ERROR, _T("taskOpenCL == nullptr.\n"));
                return RGY_ERR_UNDEFINED_BEHAVIOR;
            }
            taskOpenCL->setVideoQualityMetricFilter(m_videoQualityMetric.get());
        } else {
            m_pipelineTasks.push_back(std::make_unique<PipelineTaskVideoQualityMetric>(m_videoQualityMetric.get(), m_cl, m_device->memType(), m_device->allocator(), &m_device->mfxSession(), 0, m_mfxVer, m_pQSVLog));
        }
    }
    if (m_pmfxENC) {
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskMFXEncode>(&m_device->mfxSession(), 1, m_pmfxENC.get(), m_mfxVer, m_encParams, m_timecode.get(), m_encTimestamp.get(), m_outputTimebase, m_dynamicRC, m_hdr10plus.get(), m_dovirpu.get(), m_pQSVLog));
    } else {
        m_pipelineTasks.push_back(std::make_unique<PipelineTaskOutputRaw>(&m_device->mfxSession(), 1, m_mfxVer, m_pQSVLog));
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
    auto checkAbort = [pabort = m_pAbortByUser]() { return  (pabort != nullptr && *pabort); };
#endif
    m_pStatus->SetStart();

    CProcSpeedControl speedCtrl(m_nProcSpeedLimit);
    for (auto& task : m_pipelineTasks) {
        if (m_taskPerfMonitor) {
            task->setStopWatch();
        }
    }

    auto requireSync = [this](const size_t itask) {
        if (itask + 1 >= m_pipelineTasks.size()) return true; // 次が最後のタスクの時

        size_t srctask = itask;
        if (m_pipelineTasks[srctask]->isPassThrough()) {
            for (size_t prevtask = srctask-1; prevtask >= 0; prevtask--) {
                if (!m_pipelineTasks[prevtask]->isPassThrough()) {
                    srctask = prevtask;
                    break;
                }
            }
        }
        for (size_t nexttask = itask+1; nexttask < m_pipelineTasks.size(); nexttask++) {
            if (!m_pipelineTasks[nexttask]->isPassThrough()) {
                return m_pipelineTasks[srctask]->requireSync(m_pipelineTasks[nexttask]->taskType());
            }
        }
        return true;
    };
    auto time_prev = std::chrono::high_resolution_clock::now();

    RGY_ERR err = RGY_ERR_NONE;
    auto setloglevel = [](RGY_ERR err) {
        if (err == RGY_ERR_NONE || err == RGY_ERR_MORE_DATA || err == RGY_ERR_MORE_SURFACE || err == RGY_ERR_MORE_BITSTREAM) return RGY_LOG_DEBUG;
        if (err > RGY_ERR_NONE) return RGY_LOG_WARN;
        return RGY_LOG_ERROR;
        };
    struct PipelineTaskData {
        size_t task;
        std::unique_ptr<PipelineTaskOutput> data;
        PipelineTaskData(size_t t) : task(t), data() {};
        PipelineTaskData(size_t t, std::unique_ptr<PipelineTaskOutput>& d) : task(t), data(std::move(d)) {};
    };
    std::deque<PipelineTaskData> dataqueue;
    {
        auto checkContinue = [&checkAbort](RGY_ERR& err) {
            if (checkAbort() || stdInAbort()) { err = RGY_ERR_ABORTED; return false; }
            return err >= RGY_ERR_NONE || err == RGY_ERR_MORE_DATA || err == RGY_ERR_MORE_SURFACE;
            };
        while (checkContinue(err)) {
            if (dataqueue.empty()) {
                speedCtrl.wait(m_pipelineTasks.front()->outputFrames());
                dataqueue.push_back(PipelineTaskData(0)); // デコード実行用
            }
            while (!dataqueue.empty()) {
                auto d = std::move(dataqueue.front());
                dataqueue.pop_front();
                if (d.task < m_pipelineTasks.size()) {
                    err = RGY_ERR_NONE;
                    auto& task = m_pipelineTasks[d.task];
                    err = task->sendFrame(d.data);
                    if (!checkContinue(err)) {
                        PrintMes(setloglevel(err), _T("Break in task %s: %s.\n"), task->print().c_str(), get_err_mes(err));
                        break;
                    }
                    if (err == RGY_ERR_NONE) {
                        auto output = task->getOutput(requireSync(d.task));
                        if (output.size() == 0) break;
                        //出てきたものは先頭に追加していく
                        std::for_each(output.rbegin(), output.rend(), [itask = d.task, &dataqueue](auto&& o) {
                            dataqueue.push_front(PipelineTaskData(itask + 1, o));
                            });
                    }
                } else { // pipelineの最終的なデータを出力
                    if ((err = d.data->write(m_pFileWriter.get(), m_device->allocator(), (m_cl) ? &m_cl->queue() : nullptr, m_videoQualityMetric.get())) != RGY_ERR_NONE) {
                        PrintMes(RGY_LOG_ERROR, _T("failed to write output: %s.\n"), get_err_mes(err));
                        break;
                    }
                }
            }
            if (dataqueue.empty()) {
                // taskを前方からひとつづつ出力が残っていないかチェック(主にcheckptsの処理のため)
                for (size_t itask = 0; itask < m_pipelineTasks.size(); itask++) {
                    auto& task = m_pipelineTasks[itask];
                    auto output = task->getOutput(requireSync(itask));
                    if (output.size() > 0) {
                        //出てきたものは先頭に追加していく
                        std::for_each(output.rbegin(), output.rend(), [itask, &dataqueue](auto&& o) {
                            dataqueue.push_front(PipelineTaskData(itask + 1, o));
                            });
                        //checkptsの処理上、でてきたフレームはすぐに後続処理に渡したいのでbreak
                        break;
                    }
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
        for (size_t flushedTaskSend = 0, flushedTaskGet = 0; flushedTaskGet < m_pipelineTasks.size(); ) { // taskを前方からひとつづつflushしていく
            err = RGY_ERR_NONE;
            if (flushedTaskSend == flushedTaskGet) {
                dataqueue.push_back(PipelineTaskData(flushedTaskSend)); //flush用
            }
            while (!dataqueue.empty() && checkContinue(err)) {
                auto d = std::move(dataqueue.front());
                dataqueue.pop_front();
                if (d.task < m_pipelineTasks.size()) {
                    err = RGY_ERR_NONE;
                    auto& task = m_pipelineTasks[d.task];
                    err = task->sendFrame(d.data);
                    if (!checkContinue(err)) {
                        if (d.task == flushedTaskSend) flushedTaskSend++;
                        break;
                    }
                    auto output = task->getOutput(requireSync(d.task));
                    if (output.size() == 0) break;
                    //出てきたものは先頭に追加していく
                    std::for_each(output.rbegin(), output.rend(), [itask = d.task, &dataqueue](auto&& o) {
                        dataqueue.push_front(PipelineTaskData(itask + 1, o));
                        });
                    RGY_IGNORE_STS(err, RGY_ERR_MORE_DATA); //VPPなどでsendFrameがRGY_ERR_MORE_DATAだったが、フレームが出てくる場合がある
                } else { // pipelineの最終的なデータを出力
                    if ((err = d.data->write(m_pFileWriter.get(), m_device->allocator(), (m_cl) ? &m_cl->queue() : nullptr, m_videoQualityMetric.get())) != RGY_ERR_NONE) {
                        PrintMes(RGY_LOG_ERROR, _T("failed to write output: %s.\n"), get_err_mes(err));
                        break;
                    }
                }
            }
            if (dataqueue.empty()) {
                // taskを前方からひとつづつ出力が残っていないかチェック(主にcheckptsの処理のため)
                for (size_t itask = flushedTaskGet; itask < m_pipelineTasks.size(); itask++) {
                    auto& task = m_pipelineTasks[itask];
                    auto output = task->getOutput(requireSync(itask));
                    if (output.size() > 0) {
                        //出てきたものは先頭に追加していく
                        std::for_each(output.rbegin(), output.rend(), [itask, &dataqueue](auto&& o) {
                            dataqueue.push_front(PipelineTaskData(itask + 1, o));
                            });
                        //checkptsの処理上、でてきたフレームはすぐに後続処理に渡したいのでbreak
                        break;
                    } else if (itask == flushedTaskGet && flushedTaskGet < flushedTaskSend) {
                        flushedTaskGet++;
                    }
                }
            }
        }
    }
    // エラー終了の場合も含めキューをすべて開放する (m_pipelineTasksを解放する前に行う)
    dataqueue.clear();

    if (m_videoQualityMetric) {
        PrintMes(RGY_LOG_DEBUG, _T("Flushing video quality metric calc.\n"));
        m_videoQualityMetric->addBitstream(nullptr);
    }

    //vpp-perf-monitor
    std::vector<std::pair<tstring, double>> filter_result;
    for (auto& block : m_vpFilters) {
        if (block.type == VppFilterType::FILTER_OPENCL) {
            for (auto& filter : block.vppcl) {
                auto avgtime = filter->GetAvgTimeElapsed();
                if (avgtime > 0.0) {
                    filter_result.push_back({ filter->name(), avgtime });
                }
            }
        }
    }
    // MFXのコンポーネントをm_pipelineTasksの解放(フレームの解放)前に実施する
    PrintMes(RGY_LOG_DEBUG, _T("Clear vpp filters...\n"));
    m_vpFilters.clear();
    PrintMes(RGY_LOG_DEBUG, _T("Closing m_pmfxDEC/ENC/VPP...\n"));
    m_mfxDEC.reset();
    m_pmfxENC.reset();
    m_mfxVPP.clear();
    // taskの集計結果を表示
    if (m_taskPerfMonitor) {
        PrintMes(RGY_LOG_INFO, _T("\nTask Performance\n"));
        const int64_t totalTicks = std::accumulate(m_pipelineTasks.begin(), m_pipelineTasks.end(), 0LL, [](int64_t total, const std::unique_ptr<PipelineTask>& task) {
            return total + task->getStopWatchTotal();
        });
        if (totalTicks > 0) {
            const size_t maxWorkStrLenLen = std::accumulate(m_pipelineTasks.begin(), m_pipelineTasks.end(), (size_t)0, [](size_t maxStrLength, const std::unique_ptr<PipelineTask>& task) {
                return std::max(maxStrLength, task->getStopWatchMaxWorkStrLen());
            });
            const size_t maxTaskStrLen = std::accumulate(m_pipelineTasks.begin(), m_pipelineTasks.end(), (size_t)0, [](size_t maxStrLength, const std::unique_ptr<PipelineTask>& task) {
                return std::max(maxStrLength, _tcslen(getPipelineTaskTypeName(task->taskType())));
            });
            for (auto& task : m_pipelineTasks) {
                task->printStopWatch(totalTicks, maxWorkStrLenLen + maxTaskStrLen - _tcslen(getPipelineTaskTypeName(task->taskType())));
            }
        }
    }
    //この中でフレームの解放がなされる
    PrintMes(RGY_LOG_DEBUG, _T("Clear pipeline tasks and allocated frames...\n"));
    // 依存関係があるため、m_pipelineTasksを後ろから解放する
    for (auto it = m_pipelineTasks.rbegin(); it != m_pipelineTasks.rend(); ++it) {
        it->reset();
    }
    m_pipelineTasks.clear();
    PrintMes(RGY_LOG_DEBUG, _T("Waiting for writer to finish...\n"));
    m_pFileWriter->WaitFin();
    PrintMes(RGY_LOG_DEBUG, _T("Write results...\n"));
    if (m_videoQualityMetric) {
        PrintMes(RGY_LOG_DEBUG, _T("Write video quality metric results...\n"));
        m_videoQualityMetric->showResult();
    }
    if (m_deviceUsage) {
        m_deviceUsage->close();
    }
    m_pStatus->WriteResults();
    if (filter_result.size()) {
        PrintMes(RGY_LOG_INFO, _T("\nVpp Filter Performance\n"));
        const auto max_len = std::accumulate(filter_result.begin(), filter_result.end(), 0u, [](uint32_t max_length, std::pair<tstring, double> info) {
            return std::max(max_length, (uint32_t)info.first.length());
            });
        for (const auto& info : filter_result) {
            tstring str = info.first + _T(":");
            for (uint32_t i = (uint32_t)info.first.length(); i < max_len; i++) {
                str += _T(" ");
            }
            PrintMes(RGY_LOG_INFO, _T("%s %8.1f us\n"), str.c_str(), info.second * 1000.0);
        }
    }
    PrintMes(RGY_LOG_DEBUG, _T("RunEncode2: finished.\n"));
    err = (err == RGY_ERR_NONE || err == RGY_ERR_MORE_DATA || err == RGY_ERR_MORE_SURFACE || err == RGY_ERR_MORE_BITSTREAM || err > RGY_ERR_NONE) ? RGY_ERR_NONE : err;
    if (m_parallelEnc) {
        m_parallelEnc->close(err == RGY_ERR_NONE);
    }
    return err;
}

void CQSVPipeline::PrintMes(RGYLogLevel log_level, const TCHAR *format, ...) {
    if (m_pQSVLog.get() == nullptr) {
        if (log_level <= RGY_LOG_INFO) {
            return;
        }
    } else if (log_level < m_pQSVLog->getLogLevel(RGY_LOGT_CORE)) {
        return;
    }

    va_list args;
    va_start(args, format);

    int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    vector<TCHAR> buffer(len, 0);
    _vstprintf_s(buffer.data(), len, format, args);
    va_end(args);

    if (m_pQSVLog.get() != nullptr) {
        m_pQSVLog->write(log_level, RGY_LOGT_CORE, buffer.data());
    } else {
        _ftprintf(stderr, _T("%s"), buffer.data());
    }
}

void CQSVPipeline::GetEncodeLibInfo(mfxVersion *ver, bool *hardware) {
    if (NULL != ver && NULL != hardware) {
        mfxIMPL impl;
        m_device->mfxSession().QueryIMPL(&impl);
        *hardware = !!Check_HWUsed(impl);
        *ver = m_mfxVer;
    }

}

MemType CQSVPipeline::GetMemType() {
    return m_device->memType();
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
    if (m_pmfxENC) {
        auto prmset = std::make_unique<QSVVideoParam>(m_encParams);
        auto sts = err_to_rgy(m_pmfxENC->GetVideoParam(&prmset->videoPrm));
        if (sts == RGY_ERR_NOT_INITIALIZED) { // 未初期化の場合、設定しようとしたパラメータで代用する
            prmset->videoPrm = m_encParams.videoPrm;
            sts = RGY_ERR_NONE;
        }
        return { sts, std::move(prmset) };
    }
    auto prmset = std::make_unique<QSVVideoParam>(m_mfxVer);
    if (m_vpFilters.size() > 0) {
        prmset->isVppParam = true;
        auto& lastFilter = m_vpFilters.back();
        if (lastFilter.type == VppFilterType::FILTER_MFX) {
            auto sts = err_to_rgy(lastFilter.vppmfx->mfxvpp()->GetVideoParam(&prmset->videoPrmVpp));
            if (sts == RGY_ERR_NOT_INITIALIZED) { // 未初期化の場合、設定しようとしたパラメータで代用する
                prmset->videoPrmVpp = lastFilter.vppmfx->mfxparams();
                sts = RGY_ERR_NONE;
            }
            return { sts, std::move(prmset) };
        } else if (lastFilter.type == VppFilterType::FILTER_OPENCL) {
            auto& frameOut = lastFilter.vppcl.back()->GetFilterParam()->frameOut;
            const int blockSize = (m_encParams.videoPrm.mfx.CodecId == MFX_CODEC_HEVC) ? 32 : 16;
            prmset->videoPrmVpp.vpp.Out = frameinfo_rgy_to_enc(frameOut, m_encFps, rgy_rational<int>(0, 0), blockSize);
        } else {
            PrintMes(RGY_LOG_ERROR, _T("GetOutputVideoInfo: Unknown VPP filter type.\n"));
            return { RGY_ERR_UNSUPPORTED, std::move(prmset) };
        }
    }
    if (m_mfxDEC) {
        prmset->videoPrm = m_mfxDEC->mfxparams();
        return { RGY_ERR_NONE, std::move(prmset) };
    }
    if (m_pFileReader) {
        auto frameInfo = m_pFileReader->GetInputFrameInfo();
        prmset->videoPrm.mfx.CodecId = 0;
        // frameInfo から prmset->videoPrm.mfx.FrameInfo に値をコピーする
        prmset->videoPrm.mfx.FrameInfo.FourCC = csp_rgy_to_enc(frameInfo.csp);
        prmset->videoPrm.mfx.FrameInfo.ChromaFormat = mfx_fourcc_to_chromafmt(prmset->videoPrm.mfx.FrameInfo.FourCC);
        prmset->videoPrm.mfx.FrameInfo.PicStruct = picstruct_rgy_to_enc(frameInfo.picstruct);
        prmset->videoPrm.mfx.FrameInfo.BitDepthLuma = RGY_CSP_BIT_DEPTH[frameInfo.csp];
        prmset->videoPrm.mfx.FrameInfo.BitDepthChroma = RGY_CSP_BIT_DEPTH[frameInfo.csp];
        prmset->videoPrm.mfx.FrameInfo.Shift = 0;
        prmset->videoPrm.mfx.FrameInfo.Width = (mfxU16)frameInfo.srcWidth;
        prmset->videoPrm.mfx.FrameInfo.Height = (mfxU16)frameInfo.srcHeight;
        prmset->videoPrm.mfx.FrameInfo.CropW = (mfxU16)frameInfo.srcWidth;
        prmset->videoPrm.mfx.FrameInfo.CropH = (mfxU16)frameInfo.srcHeight;
        prmset->videoPrm.mfx.FrameInfo.FrameRateExtN = frameInfo.fpsN;
        prmset->videoPrm.mfx.FrameInfo.FrameRateExtD = frameInfo.fpsD;
        prmset->videoPrm.mfx.FrameInfo.AspectRatioW = (mfxU16)frameInfo.sar[0];
        prmset->videoPrm.mfx.FrameInfo.AspectRatioH = (mfxU16)frameInfo.sar[1];
        return { RGY_ERR_NONE, std::move(prmset) };
    }
    PrintMes(RGY_LOG_ERROR, _T("GetOutputVideoInfo: None of the pipeline elements are detected!\n"));
    return { RGY_ERR_UNSUPPORTED, std::move(prmset) };
}

RGY_ERR CQSVPipeline::CheckCurrentVideoParam(TCHAR *str, mfxU32 bufSize) {
    mfxIMPL impl;
    m_device->mfxSession().QueryIMPL(&impl);

    mfxFrameInfo DstPicInfo = m_encParams.videoPrm.mfx.FrameInfo;

    auto [ err, outFrameInfo ] = GetOutputVideoInfo();
    if (err != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to get output frame info!: %s\n"), get_err_mes(err));
        return err;
    }

    DstPicInfo = (outFrameInfo->isVppParam) ? outFrameInfo->videoPrmVpp.vpp.Out : outFrameInfo->videoPrm.mfx.FrameInfo;

    const int workSurfaceCount = std::accumulate(m_pipelineTasks.begin(), m_pipelineTasks.end(), 0, [](int sum, std::unique_ptr<PipelineTask>& task) {
        return sum + (int)task->workSurfacesCount();
        });


    if (m_pmfxENC) {
        CompareParam(m_prmSetIn, *outFrameInfo);

        m_encParams = *outFrameInfo;
    }

    TCHAR cpuInfo[256] = { 0 };
    getCPUInfo(cpuInfo, _countof(cpuInfo), &m_device->mfxSession());

    TCHAR gpu_info[1024] = { 0 };
    if (Check_HWUsed(impl)) {
        getGPUInfo("Intel", gpu_info, _countof(gpu_info), GetAdapterID(m_device->mfxSession().get()), (m_cl) ? m_cl->platform() : nullptr, (m_cl) ? false : true, m_device->intelDeviceInfo());
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
        for (const auto& devName : m_devNames) {
            if (devName != m_device->name()) {
                PRINT_INFO(_T("               %s\n"), devName.c_str());
            }
        }

        auto gpu_num_str = [](int id, int adaptor_type) {
            const TCHAR *adaptorTypeStr = nullptr;
            switch (adaptor_type) {
            case MFX_MEDIA_INTEGRATED: adaptorTypeStr = _T("i"); break;
            case MFX_MEDIA_DISCRETE: adaptorTypeStr = _T("d"); break;
            }
            static const TCHAR * const NUM_APPENDIX[] = { _T("st"), _T("nd"), _T("rd"), _T("th") };
            auto str = strsprintf(_T("%d%s"), id, NUM_APPENDIX[clamp(id-1, 0, _countof(NUM_APPENDIX) - 1)]);
            if (adaptorTypeStr) {
                str += strsprintf(_T("(%s)"), adaptorTypeStr);
            }
            return str;
        };
        tstring gpuNumStr = gpu_num_str((int)m_device->deviceNum(), m_device->adapterType());
        PRINT_INFO(_T("Media SDK      QuickSyncVideo API v%d.%02d,%s, %s GPU\n"), m_mfxVer.Major, m_mfxVer.Minor,
            get_low_power_str(outFrameInfo->videoPrm.mfx.LowPower), gpuNumStr.c_str());
    }
    PRINT_INFO(    _T("Async Depth    %d frames\n"), m_nAsyncDepth);
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_2_5)) {
        PRINT_INFO(_T("Hyper Mode     %s\n"), get_cx_desc(list_hyper_mode, outFrameInfo->hyperModePrm.Mode));
    }
    PRINT_INFO(    _T("Buffer Memory  %s, %d work buffer\n"), MemTypeToStr(m_device->memType()), workSurfaceCount);
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

    if (m_vpFilters.size() > 0 || m_videoQualityMetric) {
        const TCHAR *m = _T("VPP            ");
        if (m_vpFilters.size() > 0) {
            tstring vppstr;
            for (auto& block : m_vpFilters) {
                if (block.type == VppFilterType::FILTER_MFX) {
                    vppstr += block.vppmfx->print();
                } else if (block.type == VppFilterType::FILTER_OPENCL) {
                    for (auto& clfilter : block.vppcl) {
                        vppstr += str_replace(clfilter->GetInputMessage(), _T("\n               "), _T("\n")) + _T("\n");
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
                m = _T("               ");
                p = NULL;
            }
        }
        if (m_videoQualityMetric) {
            PRINT_INFO(_T("%s%s\n"), m, m_videoQualityMetric->GetInputMessage().c_str());
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
        const auto enc_codec = codec_enc_to_rgy(outFrameInfo->videoPrm.mfx.CodecId);
        PRINT_INFO(_T("Output         %s%s %s @ Level %s%s\n"), CodecToStr(enc_codec).c_str(),
            (outFrameInfo->videoPrm.mfx.FrameInfo.BitDepthLuma > 8) ? strsprintf(_T("(%s %dbit)"), ChromaFormatToStr(outFrameInfo->videoPrm.mfx.FrameInfo.ChromaFormat), outFrameInfo->videoPrm.mfx.FrameInfo.BitDepthLuma).c_str()
                                                                    : strsprintf(_T("(%s)"), ChromaFormatToStr(outFrameInfo->videoPrm.mfx.FrameInfo.ChromaFormat)).c_str(),
            get_profile_list(enc_codec)[get_cx_index(get_profile_list(enc_codec), outFrameInfo->videoPrm.mfx.CodecProfile)].desc,
            get_level_list(enc_codec)[get_cx_index(get_level_list(enc_codec), outFrameInfo->videoPrm.mfx.CodecLevel & 0xff)].desc,
            (enc_codec == RGY_CODEC_HEVC && (outFrameInfo->videoPrm.mfx.CodecLevel & MFX_TIER_HEVC_HIGH)) ? _T(" (high tier)") : _T(""));
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
        const auto enc_codec = codec_enc_to_rgy(outFrameInfo->videoPrm.mfx.CodecId);
        PRINT_INFO(_T("Target usage   %s\n"), TargetUsageToStr(outFrameInfo->videoPrm.mfx.TargetUsage));
        PRINT_INFO(_T("Encode Mode    %s\n"), EncmodeToStr(outFrameInfo->videoPrm.mfx.RateControlMethod));
        if (m_encParams.videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_CQP) {
            PRINT_INFO(_T("CQP Value      I:%d  P:%d  B:%d\n"), outFrameInfo->videoPrm.mfx.QPI, outFrameInfo->videoPrm.mfx.QPP, outFrameInfo->videoPrm.mfx.QPB);
        } else if (rc_is_type_lookahead(m_encParams.videoPrm.mfx.RateControlMethod)) {
            if (m_encParams.videoPrm.mfx.RateControlMethod != MFX_RATECONTROL_LA_ICQ) {
                PRINT_INFO(_T("Bitrate        %d kbps\n"), outFrameInfo->videoPrm.mfx.TargetKbps * (std::max<int>)(m_encParams.videoPrm.mfx.BRCParamMultiplier, 1));
                PRINT_INFO(_T("%s"), _T("Max Bitrate    "));
                PRINT_INT_AUTO(_T("%d kbps\n"), outFrameInfo->videoPrm.mfx.MaxKbps * (std::max<int>)(m_encParams.videoPrm.mfx.BRCParamMultiplier, 1));
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
            if (m_encParams.videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_LA_ICQ) {
                PRINT_INFO(_T("ICQ Quality    %d\n"), outFrameInfo->videoPrm.mfx.ICQQuality);
            }
        } else if (m_encParams.videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_ICQ) {
            PRINT_INFO(_T("ICQ Quality    %d\n"), outFrameInfo->videoPrm.mfx.ICQQuality);
        } else {
            PRINT_INFO(_T("Bitrate        %d kbps\n"), outFrameInfo->videoPrm.mfx.TargetKbps * (std::max<int>)(m_encParams.videoPrm.mfx.BRCParamMultiplier, 1));
            if (m_encParams.videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_AVBR) {
                //PRINT_INFO(_T("AVBR Accuracy range\t%.01lf%%"), m_encParams.videoPrm.mfx.Accuracy / 10.0);
                PRINT_INFO(_T("AVBR Converge  %d frames unit\n"), outFrameInfo->videoPrm.mfx.Convergence * 100);
            } else {
                PRINT_INFO(_T("%s"), _T("Max Bitrate    "));
                PRINT_INT_AUTO(_T("%d kbps\n"), outFrameInfo->videoPrm.mfx.MaxKbps * (std::max<int>)(m_encParams.videoPrm.mfx.BRCParamMultiplier, 1));
                if (m_encParams.videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_QVBR) {
                    PRINT_INFO(_T("QVBR Quality   %d\n"), outFrameInfo->cop3.QVBRQuality);
                }
            }
            if (outFrameInfo->videoPrm.mfx.BufferSizeInKB > 0) {
                int bufSizeInKB = outFrameInfo->videoPrm.mfx.BufferSizeInKB;
                if (enc_codec == RGY_CODEC_AV1) {
                    // AV1では、BufferSizeInKBの値がtemporal layersの値の分(=ceil(log2(GopRefDist)))だけ乗算されていると思われるので、
                    // フレーム単位の値に戻して表示する
                    int mul = 1;
                    for (int val = 1; val <= 128; mul++, val *= 2) {
                        if (val >= outFrameInfo->videoPrm.mfx.GopRefDist) {
                            break;
                        }
                    }
                    bufSizeInKB /= mul;
                }
                PRINT_INFO(_T("VBV Bufsize    %d kb\n"), bufSizeInKB * 8 * (std::max<int>)(m_encParams.videoPrm.mfx.BRCParamMultiplier, 1));
            }
            if (outFrameInfo->cop2.LookAheadDepth > 0) {
                PRINT_INFO(_T("LookaheadDepth %d\n"), outFrameInfo->cop2.LookAheadDepth);
            }
        }
        if (m_dynamicRC.size() > 0) {
            tstring strDynamicRC = tstring(_T("DynamicRC      ")) + m_dynamicRC[0].print();
            for (int i = 1; i < (int)m_dynamicRC.size(); i++) {
                strDynamicRC += _T("\n               ") + m_dynamicRC[i].print();
            }
            PRINT_INFO(_T("%s\n"), strDynamicRC.c_str());
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_16)
            && outFrameInfo->cop3.ScenarioInfo != MFX_SCENARIO_UNKNOWN
            && get_cx_desc(list_scenario_info, outFrameInfo->cop3.ScenarioInfo) != nullptr) {
            PRINT_INFO(_T("Scenario Info  %s\n"), get_cx_desc(list_scenario_info, outFrameInfo->cop3.ScenarioInfo));
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_2_9)
            && outFrameInfo->tuneEncQualityPrm.TuneQuality != MFX_ENCODE_TUNE_OFF
            && get_cx_desc(list_enc_tune_quality_mode, outFrameInfo->tuneEncQualityPrm.TuneQuality) != nullptr) {
            PRINT_INFO(_T("Tune Quality   %s\n"), get_str_of_tune_bitmask(outFrameInfo->tuneEncQualityPrm.TuneQuality).c_str());
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

        if (enc_codec == RGY_CODEC_H264 && !Check_HWUsed(impl)) {
            if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_7)) {
                PRINT_INFO(_T("Trellis        %s\n"), list_avc_trellis[get_cx_index(list_avc_trellis_for_options, outFrameInfo->cop2.Trellis)].desc);
            }
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

        const bool showAsBframes = gopRefDistAsBframe(enc_codec);
        PRINT_INFO(_T("%s     "), (showAsBframes) ? _T("Bframes   ") : _T("GopRefDist"));
        const bool showBpyramid = check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8) && outFrameInfo->videoPrm.mfx.GopRefDist >= 2;
        if (showAsBframes) {
            switch (outFrameInfo->videoPrm.mfx.GopRefDist) {
            case 0:  PRINT_INFO(_T("%s"), _T("Auto\n")); break;
            case 1:  PRINT_INFO(_T("%s"), _T("none\n")); break;
            default: PRINT_INFO(_T("%d frame%s%s%s\n"),
                outFrameInfo->videoPrm.mfx.GopRefDist - 1, (outFrameInfo->videoPrm.mfx.GopRefDist > 2) ? _T("s") : _T(""),
                showBpyramid ? _T(", B-pyramid: ") : _T(""),
                showBpyramid ? ((MFX_B_REF_PYRAMID == outFrameInfo->cop2.BRefType) ? _T("on") : _T("off")) : _T("")); break;
            }
        } else {
            PRINT_INFO(_T("%d%s%s\n"), outFrameInfo->videoPrm.mfx.GopRefDist,
                showBpyramid ? _T(", B-pyramid: ") : _T(""),
                showBpyramid ? ((MFX_B_REF_PYRAMID == outFrameInfo->cop2.BRefType) ? _T("on") : _T("off")) : _T(""));
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

        if (enc_codec == RGY_CODEC_VP8) {
            PRINT_INFO(_T("Sharpness      %d\n"), outFrameInfo->copVp8.SharpnessLevel);
        }
        { const auto &vui_str = m_encVUI.print_all();
        if (vui_str.length() > 0) {
            PRINT_INFO(_T("VUI            %s\n"), vui_str.c_str());
        }
        }
        if (m_hdrseiOut) {
            const auto masterdisplay = m_hdrseiOut->print_masterdisplay();
            const auto maxcll = m_hdrseiOut->print_maxcll();
            const auto atcsei = (enc_codec == RGY_CODEC_HEVC) ? m_hdrseiOut->print_atcsei() : "";
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
            if (atcsei.length() > 0) {
                PRINT_INFO(_T("atcsei         %s\n"), char_to_tstring(atcsei).c_str());
            }
        }
        if (m_hdr10plus) {
            PRINT_INFO(_T("Dynamic HDR10  %s\n"), m_hdr10plus->inputJson().c_str());
        } else if (m_hdr10plusMetadataCopy) {
            PRINT_INFO(_T("Dynamic HDR10  %s\n"), _T("copy"));
        }
        if (m_doviProfile != RGY_DOVI_PROFILE_UNSET) {
            tstring profile_copy;
            if (m_doviProfile == RGY_DOVI_PROFILE_COPY) {
                profile_copy = tstring(_T(" (")) + get_cx_desc(list_dovi_profile, m_pFileReader->getInputDOVIProfile()) + tstring(_T(")"));
            }
            PRINT_INFO(_T("dovi profile   %s%s\n"), get_cx_desc(list_dovi_profile, m_doviProfile), profile_copy.c_str());
        }
        if (m_dovirpu) {
            PRINT_INFO(_T("dovi rpu       %s\n"), m_dovirpu->get_filepath().c_str());
        } else if (m_dovirpuMetadataCopy) {
            PRINT_INFO(_T("dovi rpu       %s\n"), _T("copy"));
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
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)) {
            if (outFrameInfo->cop2.RepeatPPS == MFX_CODINGOPTION_ON) {
                extFeatures += _T("RepeatPPS ");
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_9)) {
            if (outFrameInfo->cop2.DisableDeblockingIdc) {
                extFeatures += _T("No-Deblock ");
            }
            if (outFrameInfo->cop2.IntRefType != MFX_REFRESH_NO) {
                extFeatures += strsprintf(_T("Intra-Refresh:%d "), outFrameInfo->cop2.IntRefCycleSize);
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
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_18) && enc_codec == RGY_CODEC_HEVC) {
            if (outFrameInfo->cop3.GPB == MFX_CODINGOPTION_ON) {
                extFeatures += _T("GPB ");
            } else if (outFrameInfo->cop3.GPB == MFX_CODINGOPTION_OFF) {
                extFeatures += _T("NoGPB ");
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_19)) {
            if (outFrameInfo->cop3.EnableQPOffset == MFX_CODINGOPTION_ON) {
                extFeatures += _T("QPOffset ");
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_2_4)) {
            if (outFrameInfo->cop3.AdaptiveRef == MFX_CODINGOPTION_ON) {
                extFeatures += _T("AdaptiveRef ");
            }
            if (outFrameInfo->cop3.AdaptiveLTR == MFX_CODINGOPTION_ON) {
                extFeatures += _T("AdaptiveLTR ");
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_2_2)) {
            if (outFrameInfo->cop3.AdaptiveCQM == MFX_CODINGOPTION_ON) {
                extFeatures += _T("AdaptiveCQM ");
            }
        }
        if (outFrameInfo->cop.AUDelimiter == MFX_CODINGOPTION_ON) {
            extFeatures += _T("aud ");
        }
        if (outFrameInfo->cop.PicTimingSEI == MFX_CODINGOPTION_ON) {
            extFeatures += _T("pic_struct ");
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_10)) {
            if (outFrameInfo->cop2.BufferingPeriodSEI == MFX_CODINGOPTION_ON) {
                extFeatures += _T("BufPeriod ");
            }
        }
        //if (outFrameInfo->cop.SingleSeiNalUnit == MFX_CODINGOPTION_ON) {
        //    extFeatures += _T("SingleSEI ");
        //}
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_23)) {
            if (outFrameInfo->cop3.RepartitionCheckEnable == MFX_CODINGOPTION_ON) {
                extFeatures += _T("RepartitionCheck ");
            }
        }
        if (m_encParams.videoPrm.mfx.CodecId == MFX_CODEC_HEVC) {
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

