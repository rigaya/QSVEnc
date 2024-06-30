// -----------------------------------------------------------------------------------------
// x264guiEx/x265guiEx/svtAV1guiEx/ffmpegOut/QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2010-2022 rigaya
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

#pragma once

using namespace System;
using namespace System::Data;
using namespace System::Threading;
using namespace System::IO;
using namespace System::Collections::Generic;

#include "qsv_prm.h"
#include "qsv_util.h"
#include "qsv_query.h"
#include "mfxstructures.h"

static const WCHAR *use_default_exe_path = L"exe_files内の実行ファイルを自動選択";

const int fcgTBQualityTimerLatency = 600;
const int fcgTBQualityTimerPeriod = 40;
const int fcgTXCmdfulloffset = 57;
const int fcgCXAudioEncModeSmallWidth = 189;
const int fcgCXAudioEncModeLargeWidth = 237;


static const WCHAR * const list_aspect_ratio[] = {
    L"SAR(PAR, 画素比)で指定",
    L"DAR(画面比)で指定",
    NULL
};

static const WCHAR * const list_tempdir[] = {
    L"出力先と同じフォルダ (デフォルト)",
    L"システムの一時フォルダ",
    L"カスタム",
    NULL
};

static const WCHAR * const list_audtempdir[] = {
    L"変更しない",
    L"カスタム",
    NULL
};

static const WCHAR * const list_mp4boxtempdir[] = {
    L"指定しない",
    L"カスタム",
    NULL
};




static const ENC_OPTION_STR aspect_desc[] = {
    { NULL, AUO_CONFIG_CX_ASPECT_SAR, L"SAR比を指定 (デフォルト)" },
    { NULL, AUO_CONFIG_CX_ASPECT_DAR, L"画面比から自動計算"       },
    { NULL, AUO_MES_UNKNOWN, NULL }
};

static const ENC_OPTION_STR tempdir_desc[] = {
    { NULL, AUO_CONFIG_CX_TEMPDIR_OUTDIR, L"出力先と同じフォルダ (デフォルト)" },
    { NULL, AUO_CONFIG_CX_TEMPDIR_SYSTEM, L"システムの一時フォルダ"            },
    { NULL, AUO_CONFIG_CX_TEMPDIR_CUSTOM, L"カスタム"                          },
    { NULL, AUO_MES_UNKNOWN, NULL }
};

static const ENC_OPTION_STR audtempdir_desc[] = {
    { NULL, AUO_CONFIG_CX_AUDTEMP_DEFAULT, L"変更しない" },
    { NULL, AUO_CONFIG_CX_AUDTEMP_CUSTOM,  L"カスタム"   },
    { NULL, AUO_MES_UNKNOWN, NULL }
};

static const ENC_OPTION_STR mp4boxtempdir_desc[] = {
    { NULL, AUO_CONFIG_CX_MP4BOXTEMP_DEFAULT, L"指定しない" },
    { NULL, AUO_CONFIG_CX_MP4BOXTEMP_CUSTOM,  L"カスタム"   },
    { NULL, AUO_MES_UNKNOWN, NULL }
};

#if ENCODER_QSV
static const ENC_OPTION_STR2 list_interlaced_mfx_gui[] = {
    { AUO_CONFIG_CX_INTERLACE_PROGRESSIVE, L"progressive",     MFX_PICSTRUCT_PROGRESSIVE },
    { AUO_CONFIG_CX_INTERLACE_TFF,         L"interlaced(tff)", MFX_PICSTRUCT_FIELD_TFF   },
    { AUO_CONFIG_CX_INTERLACE_BFF,         L"interlaced(bff)", MFX_PICSTRUCT_FIELD_BFF   },
    { AUO_MES_UNKNOWN, NULL, 0 }
};
#endif

static const ENC_OPTION_STR audio_enc_timing_desc[] = {
    { NULL, AUO_CONFIG_CX_AUD_ENC_ORDER_AFTER,    L"後"   },
    { NULL, AUO_CONFIG_CX_AUD_ENC_ORDER_BEFORE,   L"前"   },
    { NULL, AUO_CONFIG_CX_AUD_ENC_ORDER_PARALLEL, L"同時" },
    { NULL, AUO_MES_UNKNOWN, NULL }
};

static const ENC_OPTION_STR2 list_deinterlace_gui[] = {
    { AUO_CONFIG_CX_DEINTERLACE_NONE,   L"なし",                   0 },
#if ENCODER_QSV
    { AUO_CONFIG_CX_DEINTERLACE_NORMAL, L"インタレ解除 (通常)",     MFX_DEINTERLACE_NORMAL      },
    { AUO_CONFIG_CX_DEINTERLACE_IT,     L"インタレ解除 (24fps化)",  MFX_DEINTERLACE_IT          },
    { AUO_CONFIG_CX_DEINTERLACE_BOB,    L"インタレ解除 (Bob化)",    MFX_DEINTERLACE_BOB         },
#elif ENCODER_NVENC
#elif ENCODER_VCEENC
#else
    static_assert(false);
#endif
    { AUO_CONFIG_CX_DEINTERLACE_AFS,    L"自動フィールドシフト",    100 },
    { AUO_CONFIG_CX_DEINTERLACE_NNEDI,  L"nnedi",                   101 },
    { AUO_CONFIG_CX_DEINTERLACE_YADIF,  L"yadif",                   102 },
    { AUO_CONFIG_CX_DEINTERLACE_DECOMB, L"decomb",                  103 },
    { AUO_MES_UNKNOWN, NULL, NULL }
};

static const ENC_OPTION_STR list_vpp_afs_analyze[] = {
    { NULL, AUO_CONFIG_CX_AFS_ANALYZE0, L"0 - 解除なし"         },
    { NULL, AUO_CONFIG_CX_AFS_ANALYZE1, L"1 - フィールド三重化" },
    { NULL, AUO_CONFIG_CX_AFS_ANALYZE2, L"2 - 縞検出二重化"     },
    { NULL, AUO_CONFIG_CX_AFS_ANALYZE3, L"3 - 動き検出二重化"   },
    { NULL, AUO_CONFIG_CX_AFS_ANALYZE4, L"4 - 動き検出補間"     },
    { NULL, AUO_MES_UNKNOWN, NULL},
};

static const ENC_OPTION_STR2 list_vpp_nnedi_pre_screen_gui[] = {
    { AUO_CONFIG_CX_NNEDI_PRESCREEN_NONE,           L"none",           VPP_NNEDI_PRE_SCREEN_NONE           },
    { AUO_CONFIG_CX_NNEDI_PRESCREEN_ORIGINAL,       L"original",       VPP_NNEDI_PRE_SCREEN_ORIGINAL       },
    { AUO_CONFIG_CX_NNEDI_PRESCREEN_NEW,            L"new",            VPP_NNEDI_PRE_SCREEN_NEW            },
    { AUO_CONFIG_CX_NNEDI_PRESCREEN_ORIGINAL_BLOCK, L"original_block", VPP_NNEDI_PRE_SCREEN_ORIGINAL_BLOCK },
    { AUO_CONFIG_CX_NNEDI_PRESCREEN_NEW_BLOCK,      L"new_block",      VPP_NNEDI_PRE_SCREEN_NEW_BLOCK      },
    { AUO_MES_UNKNOWN, NULL, NULL }
};

static const ENC_OPTION_STR2 list_vpp_yadif_mode_gui[] = {
    { AUO_CONFIG_CX_YADIF_MODE_NORMAL, L"normal",        VPP_YADIF_MODE_AUTO     },
    { AUO_CONFIG_CX_YADIF_MODE_BOB,    L"bob",           VPP_YADIF_MODE_BOB_AUTO },
    { AUO_MES_UNKNOWN, NULL, NULL }
};

static const ENC_OPTION_STR2 list_vpp_denoise_dct_step_gui[] = {
    { AUO_CONFIG_CX_DENOISE_DCT_STEP_1, L"1 - 高品質",  1 },
    { AUO_CONFIG_CX_DENOISE_DCT_STEP_2, L"2",           2 },
    { AUO_CONFIG_CX_DENOISE_DCT_STEP_4, L"4",           4 },
    { AUO_CONFIG_CX_DENOISE_DCT_STEP_8, L"8 - 高速",    8 },
    { AUO_MES_UNKNOWN, NULL, NULL }
};
static_assert(_countof(list_vpp_denoise_dct_step) == _countof(list_vpp_denoise_dct_step_gui), "list_vpp_denoise_dct_step size");

static const ENC_OPTION_STR2 list_encmode[] = {
#if ENCODER_QSV
    { AUO_CONFIG_CX_RC_CBR,    L"ビットレート指定 - CBR",           MFX_RATECONTROL_CBR    },
    { AUO_CONFIG_CX_RC_VBR,    L"ビットレート指定 - VBR",           MFX_RATECONTROL_VBR    },
    { AUO_CONFIG_CX_RC_AVBR,   L"ビットレート指定 - AVBR",          MFX_RATECONTROL_AVBR   },
    { AUO_CONFIG_CX_RC_QVBR,   L"ビットレート指定 - QVBR",          MFX_RATECONTROL_QVBR   },
    { AUO_CONFIG_CX_RC_CQP,    L"固定量子化量 (CQP)",               MFX_RATECONTROL_CQP    },
    { AUO_CONFIG_CX_RC_LA,     L"先行探索レート制御",               MFX_RATECONTROL_LA     },
    { AUO_CONFIG_CX_RC_LA_HRD, L"先行探索レート制御 (HRD準拠)",     MFX_RATECONTROL_LA_HRD },
    { AUO_CONFIG_CX_RC_ICQ,    L"固定品質モード",                   MFX_RATECONTROL_ICQ    },
    { AUO_CONFIG_CX_RC_LA_ICQ, L"先行探索付き固定品質モード",       MFX_RATECONTROL_LA_ICQ },
    { AUO_CONFIG_CX_RC_VCM,    L"ビデオ会議モード",                 MFX_RATECONTROL_VCM    },
#elif ENCODER_NVENC
#elif ENCODER_VCEENC
#else
    static_assert(false);
#endif
    { AUO_MES_UNKNOWN, NULL, NULL }
};

static const ENC_OPTION_STR2 list_vpp_deband_gui[] = {
    { AUO_CONFIG_CX_DEBAND_0, L"0 - 1点参照",  0 },
    { AUO_CONFIG_CX_DEBAND_1, L"1 - 2点参照",  1 },
    { AUO_CONFIG_CX_DEBAND_2, L"2 - 4点参照",  2 },
    { AUO_MES_UNKNOWN, NULL, 0 }
};

static const ENC_OPTION_STR2 list_vpp_fft3d_temporal_gui[] = {
    { AUO_MES_UNKNOWN, L"off",  0 },
    { AUO_MES_UNKNOWN, L"on",  1 },
    { AUO_MES_UNKNOWN, NULL, 0 }
};

#if ENCODER_QSV
static const ENC_OPTION_STR2 list_rotate_angle_ja[] = {
    { AUO_MES_UNKNOWN,   L"0°",  MFX_ANGLE_0    },
    { AUO_MES_UNKNOWN,  L"90°",  MFX_ANGLE_90   },
    { AUO_MES_UNKNOWN, L"180°",  MFX_ANGLE_180  },
    { AUO_MES_UNKNOWN, L"270°",  MFX_ANGLE_270  },
    { AUO_MES_UNKNOWN, NULL, 0 }
};

static const ENC_OPTION_STR2 list_out_enc_codec[] = {
    { AUO_MES_UNKNOWN, L"H.264 / AVC",  RGY_CODEC_H264  },
    { AUO_MES_UNKNOWN, L"H.265 / HEVC", RGY_CODEC_HEVC },
    //{ AUO_MES_UNKNOWN,"VC-1", RGY_CODEC_VC1 },
    { AUO_MES_UNKNOWN, L"VP9", RGY_CODEC_VP9 },
    { AUO_MES_UNKNOWN, L"AV1", RGY_CODEC_AV1 },
    { AUO_MES_UNKNOWN, NULL, NULL }
};
//下記は一致していないといけない
static_assert(_countof(list_out_enc_codec)-1/*NULLの分*/ == _countof(CODEC_LIST_AUO));
#endif

static RGY_CODEC get_out_enc_codec_by_index(const int index) {
    return (index >= 0 && index < _countof(list_out_enc_codec)-1/*NULLの分*/) ? (RGY_CODEC)list_out_enc_codec[index].value : RGY_CODEC_UNKNOWN;
}

static const ENC_OPTION_STR2 list_log_level_jp[] = {
    { AUO_CONFIG_CX_LOG_LEVEL_INFO,  L"通常",                  RGY_LOG_INFO  },
    { AUO_CONFIG_CX_LOG_LEVEL_MORE,  L"音声/muxのログも表示 ", RGY_LOG_MORE  },
    { AUO_CONFIG_CX_LOG_LEVEL_DEBUG, L"デバッグ用出力も表示 ", RGY_LOG_DEBUG },
    { AUO_MES_UNKNOWN, NULL, NULL }
};

const WCHAR * const bit_depth_desc[] = {
    L"8bit",
    L"10bit",
    L"12bit",
    NULL
};
static int get_bit_depth(const WCHAR *str) {
    if (0 == wcscmp(str, bit_depth_desc[0])) return 8;
    if (0 == wcscmp(str, bit_depth_desc[1])) return 10;
    if (0 == wcscmp(str, bit_depth_desc[2])) return 12;
    return 8;
}
static int get_bit_depth(int index) {
    return get_bit_depth(bit_depth_desc[index]);
}
static int get_bit_depth_idx(int bit_depth) {
    switch (bit_depth) {
    case 10: return 1;
    case 12: return 2;
    case 8:
    default: return 0;
    }
}


//メモ表示用 RGB
const int StgNotesColor[][3] = {
    {  80,  72,  92 },
    { 120, 120, 120 }
};

const WCHAR * const DefaultStgNotes = L"メモ...";


namespace QSVEnc {

    ref class LocalSettings {
    public:
        String^ vidEncName;
        String^ vidEncPath;
        List<String^>^ audEncName;
        List<String^>^ audEncExeName;
        List<String^>^ audEncPath;
        String^ MP4MuxerExeName;
        String^ MP4MuxerPath;
        String^ MKVMuxerExeName;
        String^ MKVMuxerPath;
        String^ TC2MP4ExeName;
        String^ TC2MP4Path;
        String^ MP4RawExeName;
        String^ MP4RawPath;
        String^ CustomTmpDir;
        String^ CustomAudTmpDir;
        String^ CustomMP4TmpDir;
        String^ LastAppDir;
        String^ LastBatDir;
        String^ AuoLinkSrcPath;

        LocalSettings() {
            audEncName = gcnew List<String^>();
            audEncExeName = gcnew List<String^>();
            audEncPath = gcnew List<String^>();
        }
        ~LocalSettings() {
            delete audEncName;
            delete audEncExeName;
            delete audEncPath;
        }
    };

    value struct ExeControls {
        String^ Name;
        String^ Path;
        const char* args;
    };

    value struct TrackBarNU {
        TrackBar ^TB;
        NumericUpDown ^NU;
    };

    ref class CodecFeature {
        array<UInt64>^ features_;
        DataTable^ dataTableCodec_;
    public:
        CodecFeature() {
            features_ = nullptr;
            dataTableCodec_ = nullptr;
        };
        ~CodecFeature() {
            delete features_;
            delete dataTableCodec_;
        }
        void initColumns() {
            dataTableCodec_ = gcnew DataTable();
            dataTableCodec_->Columns->Add(L"機能");
            for (int i_rc = 0; i_rc < _countof(list_rate_control_ry); i_rc++) {
                dataTableCodec_->Columns->Add(String(list_rate_control_ry[i_rc].desc).ToString()->TrimEnd());
            }
        }
        void initFeatureDataArray() {
            features_ = gcnew array<UInt64>(_countof(list_rate_control_ry));
            for (int i_rc = 0; i_rc < _countof(list_rate_control_ry); i_rc++) {
                features_[i_rc] = 0;
            }
        }
        void setFeatures(array<UInt64>^ features) {
            features_ = features;

            UInt64 rc_features = 0;
            for (int i_rc = 0; i_rc < features_->Length; i_rc++) {
                features_[i_rc] |= rc_features;
            }
            GenerateTable();
        }
        UInt64 features(int rc_index) {
            if (rc_index < 0 || features_->Length <= rc_index) {
                return 0;
            }
            return features_[rc_index];
        }
        bool avail() {
            for (int i = 0; i < features_->Length; i++) {
                if ((features_[i] & ENC_FEATURE_CURRENT_RC) != 0) {
                    return true;
                }
            }
            return false;
        }
        DataTable^ table() { return dataTableCodec_; }

        System::Void GenerateTable() {
            static const ENC_OPTION_STR3 list_enc_feature_jp[] = {
                { AUO_CONFIG_CX_FEATURE_CURRENT_RC,             L"モード有効       ", ENC_FEATURE_CURRENT_RC               },
                { AUO_CONFIG_CX_FEATURE_10BIT_DEPTH,            L"10bit深度        ", ENC_FEATURE_10BIT_DEPTH            },
                { AUO_CONFIG_CX_FEATURE_HYPER_MODE,             L"Deep Link        ", ENC_FEATURE_HYPER_MODE             },
                { AUO_CONFIG_CX_FEATURE_INTERLACE,              L"インタレ保持     ", ENC_FEATURE_INTERLACE              },
                { AUO_CONFIG_CX_FEATURE_VUI_INFO,               L"色設定等出力     ", ENC_FEATURE_VUI_INFO               },
                //{ AUO_CONFIG_CX_FEATURE_AUD,                    L"aud              ", ENC_FEATURE_AUD                    },
                //{ AUO_CONFIG_CX_FEATURE_PIC_STRUCT,             L"pic_struct       ", ENC_FEATURE_PIC_STRUCT             },
                //{ AUO_CONFIG_CX_FEATURE_RDO,                    L"rdo              ", ENC_FEATURE_RDO                    },
                //{ AUO_CONFIG_CX_FEATURE_CAVLC,                  L"CAVLC            ", ENC_FEATURE_CAVLC                  },
                { AUO_CONFIG_CX_FEATURE_GOPREFDIST,             L"Bフレーム/GopRef ", ENC_FEATURE_GOPREFDIST             },
                { AUO_CONFIG_CX_FEATURE_ADAPTIVE_I,             L"適応的Iフレーム  ", ENC_FEATURE_ADAPTIVE_I             },
                { AUO_CONFIG_CX_FEATURE_ADAPTIVE_B,             L"適応的Bフレーム  ", ENC_FEATURE_ADAPTIVE_B             },
                { AUO_CONFIG_CX_FEATURE_WEIGHT_P,               L"重み付きPフレーム", ENC_FEATURE_WEIGHT_P               },
                { AUO_CONFIG_CX_FEATURE_WEIGHT_B,               L"重み付きBフレーム", ENC_FEATURE_WEIGHT_B               },
                { AUO_CONFIG_CX_FEATURE_FADE_DETECT,            L"フェード検出     ", ENC_FEATURE_FADE_DETECT            },
                { AUO_CONFIG_CX_FEATURE_B_PYRAMID,              L"ピラミッド参照   ", ENC_FEATURE_B_PYRAMID              },
                { AUO_CONFIG_CX_FEATURE_B_PYRAMID_MANY_BFRAMES, L" +多Bframe       ", ENC_FEATURE_B_PYRAMID_MANY_BFRAMES },
                { AUO_CONFIG_CX_FEATURE_SCENARIO_INFO,          L"ScenarioInfo     ", ENC_FEATURE_SCENARIO_INFO          },
                { AUO_CONFIG_CX_FEATURE_MBBRC,                  L"MB単位レート制御 ", ENC_FEATURE_MBBRC                  },
                //{ AUO_CONFIG_CX_FEATURE_EXT_BRC,                L"ExtBRC           ", ENC_FEATURE_EXT_BRC                },
                { AUO_CONFIG_CX_FEATURE_LA_DS,                  L"先行探索品質     ", ENC_FEATURE_LA_DS                  },
                { AUO_CONFIG_CX_FEATURE_QP_MINMAX,              L"最大/最小 QP     ", ENC_FEATURE_QP_MINMAX              },
                { AUO_CONFIG_CX_FEATURE_TRELLIS,                L"Trellis          ", ENC_FEATURE_TRELLIS                },
                { AUO_CONFIG_CX_FEATURE_INTRA_REFRESH,          L"周期的ｲﾝﾄﾗ更新   ", ENC_FEATURE_INTRA_REFRESH          },
                { AUO_CONFIG_CX_FEATURE_NO_DEBLOCK,             L"No-Deblock       ", ENC_FEATURE_NO_DEBLOCK             },
                { AUO_CONFIG_CX_FEATURE_PERMBQP,                L"MBQP(CQP)        ", ENC_FEATURE_PERMBQP                },
                { AUO_CONFIG_CX_FEATURE_DIRECT_BIAS_ADJUST,     L"ﾀﾞｲﾚｸﾄﾓｰﾄﾞ最適化 ", ENC_FEATURE_DIRECT_BIAS_ADJUST     },
                { AUO_CONFIG_CX_FEATURE_GLOBAL_MOTION_ADJUST,   L"MVコスト調整     ", ENC_FEATURE_GLOBAL_MOTION_ADJUST   },
                { AUO_CONFIG_CX_FEATURE_HEVC_SAO,               L"SAO              ", ENC_FEATURE_HEVC_SAO               },
                { AUO_CONFIG_CX_FEATURE_HEVC_CTU,               L"最大 CTU Size    ", ENC_FEATURE_HEVC_CTU               },
                { AUO_CONFIG_CX_FEATURE_HEVC_TSKIP,             L"TSkip            ", ENC_FEATURE_HEVC_TSKIP             },
                { AUO_MES_UNKNOWN, NULL, 0 },
            };

            dataTableCodec_ = gcnew DataTable();
            dataTableCodec_->Columns->Add(L"機能");
            for (int i_rc = 0; i_rc < _countof(list_rate_control_ry); i_rc++) {
                dataTableCodec_->Columns->Add(String(list_rate_control_ry[i_rc].desc).ToString()->TrimEnd());
            }

            //第2行以降を連続で追加していく
            for (int i = 0; list_enc_feature_jp[i].desc; i++) {
                DataRow^ drb = dataTableCodec_->NewRow();

                String^ string = nullptr;
                if (list_enc_feature_jp[i].mes != AUO_MES_UNKNOWN) {
                    string = LOAD_CLI_STRING(list_enc_feature_jp[i].mes);
                }
                if (string == nullptr || string->Length == 0) {
                    string = String(list_enc_feature_jp[i].desc).ToString();
                }
                drb[0] = string;
                for (int j = 1; j < dataTableCodec_->Columns->Count; j++) {
                    drb[j] = String((features_[j - 1] & list_enc_feature_jp[i].value) ? L"○" : L"×").ToString();
                }
                dataTableCodec_->Rows->Add(drb);
            }
        }
    };

    ref class CodecData {
        RGY_CODEC codec_;
        CodecFeature^ featureFF;
        CodecFeature^ featurePG;
    public:
        CodecData(RGY_CODEC codec) {
            codec_ = codec;
            featureFF = gcnew CodecFeature();
            featurePG = gcnew CodecFeature();
        };
        ~CodecData() {
            delete featureFF;
            delete featurePG;
        };
        RGY_CODEC codec() { return codec_; }
        bool codecAvail(bool lowpower) { return feature(lowpower)->avail(); }
        bool codecAvail() { return featureFF->avail() || featurePG->avail(); }
        void setFeatures(const bool lowpower, array<UInt64>^ features) {
            feature(lowpower)->setFeatures(features);
        }
        CodecFeature^ feature(const bool lowpower) { return (lowpower) ? featureFF : featurePG; }
        UInt64 feature(const bool lowpower, const int rc_index) {
            return feature(lowpower)->features(rc_index);
        }
        DataTable^ table(const bool lowpower, const bool reGenerateTable) {
            if (reGenerateTable) {
                generateTable();
            }
            return feature(lowpower)->table();
        }

        void initDataTableColumns() {
            featureFF->initColumns();
            featurePG->initColumns();
        }
        void initFeatureDataArray() {
            featureFF->initFeatureDataArray();
            featurePG->initFeatureDataArray();
        }
        void generateTable() {
            featureFF->GenerateTable();
            featurePG->GenerateTable();
        }
        void init() {
            initDataTableColumns();
            initFeatureDataArray();
        }
    };

    ref class QSVDevFeatures {
        QSVDeviceNum devID_;
        String^ devName_;
        String^ exePath_;
        array<String^>^ environmentInfo_;
        array<CodecData^>^ codecFeatureList_;
        Thread^ thGetFeatures;
        bool getFeaturesFinished_;
    public:
        QSVDevFeatures(QSVDeviceNum devID, String^ devName, String^ exePath, array<String^>^ environmentInfo) {
            devID_ = devID;
            devName_ = devName;
            exePath_ = exePath;
            environmentInfo_ = environmentInfo;
            getFeaturesFinished_ = false;
            codecFeatureList_ = nullptr;
        }
        ~QSVDevFeatures() {
            if (codecFeatureList_) delete codecFeatureList_;
            if (thGetFeatures != nullptr && thGetFeatures->IsAlive) {
                thGetFeatures->Join();
                delete thGetFeatures;
            }
        };
        QSVDeviceNum devID() { return devID_; }
        String^ devName() { return devName_; }
        bool getFeaturesFinished() { return getFeaturesFinished_; }
        void init() {
            int codecCount = 0;
            while (list_out_enc_codec[codecCount].desc)
                codecCount++;

            codecFeatureList_ = gcnew array<CodecData^>(codecCount);
            for (int i_codec = 0; i_codec < codecCount; i_codec++) {
                codecFeatureList_[i_codec] = gcnew CodecData(get_out_enc_codec_by_index(i_codec));
                codecFeatureList_[i_codec]->init();
            }
        }
        CodecData^ getCodecData(const RGY_CODEC codec) {
            for (int i = 0; i < (int)codecFeatureList_->Length; i++) {
                if (codecFeatureList_[i]->codec() == codec) {
                    return codecFeatureList_[i];
                }
            }
            return nullptr;
        }
        bool codecAvail(const RGY_CODEC codec) {
            if (thGetFeatures != nullptr && thGetFeatures->IsAlive) {
                thGetFeatures->Join();
            }
            CodecData^ cdata = getCodecData(codec);
            if (cdata == nullptr) {
                return false;
            }
            return cdata->codecAvail();
        }
        bool codecAvail(const RGY_CODEC codec, bool lowpower) {
            if (thGetFeatures != nullptr && thGetFeatures->IsAlive) {
                thGetFeatures->Join();
            }
            CodecData^ cdata = getCodecData(codec);
            if (cdata == nullptr) {
                return false;
            }
            return cdata->codecAvail(lowpower);
        }
        UInt64 featureOfRC(const int rc_index, const RGY_CODEC codec, const bool lowpower) {
            if (thGetFeatures != nullptr && thGetFeatures->IsAlive) {
                thGetFeatures->Join();
            }
            CodecData^ cdata = getCodecData(codec);
            if (cdata == nullptr) {
                return false;
            }
            return cdata->feature(lowpower, rc_index);
        }
        DataTable^ getTable(const RGY_CODEC codec, const bool lowpower, const bool reGenerateTable) {
            if (thGetFeatures != nullptr && thGetFeatures->IsAlive) {
                thGetFeatures->Join();
            }
            CodecData^ cdata = getCodecData(codec);
            if (cdata == nullptr) {
                return nullptr;
            }
            return cdata->table(lowpower, reGenerateTable);
        }
        void startGetFeatures() {
            thGetFeatures = gcnew Thread(gcnew ThreadStart(this, &QSVDevFeatures::getFeatures));
            thGetFeatures->Start();
        }
        System::Void getFeatures() {
            if (exePath_ == nullptr || !System::IO::File::Exists(exePath_)) {
                getFeaturesFinished_ = true;
                MessageBox::Show(L"getFeaturesFinished_");
                return;
            }

            //feature情報のキャッシュファイル
            //先頭に--check-environment-auoの結果
            //後半に--check-features-auoの結果
            String^ cacheFile = exePath_ + L".dev" + ((int)devID_).ToString() + L".featureCache.txt";

            bool use_cache = false;
            auto featureDataLines = gcnew List<String^>();
            //キャッシュファイルがあれば、そちらを読むことを考える
            if (environmentInfo_ != nullptr && environmentInfo_->Length > 0) {
                if (System::IO::File::Exists(cacheFile)) {
                    try {
                        StreamReader^ sr = gcnew StreamReader(cacheFile);
                        String ^line;
                        while ((line = sr->ReadLine()) != nullptr) {
                            featureDataLines->Add(line);
                        }
                        sr->Close();
                    } catch (...) {

                    }
                    //まず、--check-environment-auoの出力がキャッシュファイルと一致しているか
                    use_cache = true;
                    for (int i = 0; i < environmentInfo_->Length; i++) {
                        //不一致だったらキャッシュファイルは使用しない
                        if (featureDataLines->Count <= i) {
                            use_cache = false;
                            break;
                        }
                        int compare_len = environmentInfo_[i]->Length;
                        if (environmentInfo_[i]->Contains(L"RAM: ")) {
                            continue; //使用メモリ量は実行するたびに変わる
                        }
                        if (environmentInfo_[i]->Contains(L"CPU:")) {
                            //TBの周波数は取得するタイミングで変わりうる
                            auto len = environmentInfo_[i]->IndexOf(L"@");
                            if (len < 0) {
                                len = environmentInfo_[i]->IndexOf(L"[");
                            }
                            if (len < 0) {
                                len = environmentInfo_[i]->IndexOf(L"(");
                            }
                            if (len > 0) {
                                compare_len = len;
                            }
                        }
                        if (String::Compare(environmentInfo_[i], 0, featureDataLines[i], 0, compare_len) != 0) {
                            use_cache = false;
                            break;
                        }
                    }
                    if (!use_cache) {
                        featureDataLines = gcnew List<String^>();
                        featureDataLines->AddRange(environmentInfo_);
                    }
                } else {
                    featureDataLines->AddRange(environmentInfo_);
                }
            }

            //キャッシュを使用できない場合は、実際に情報を取得する(時間がかかる)
            if (!use_cache) {
                char exe_path[1024];
                GetCHARfromString(exe_path, sizeof(exe_path), exePath_);
                char cmd[128];
                if (devID_ != QSVDeviceNum::AUTO) {
                    sprintf_s(cmd, "--check-features-auo -d %d", (int)devID_);
                } else {
                    strcpy_s(cmd, "--check-features-auo");
                }
                std::vector<char> buffer(256 * 1024);
                if (get_exe_message(exe_path, cmd, buffer.data(), buffer.size(), AUO_PIPE_MUXED) == RP_SUCCESS) {
                    auto lines = String(buffer.data()).ToString()->Split(String(L"\r\n").ToString()->ToCharArray(), System::StringSplitOptions::RemoveEmptyEntries);
                    featureDataLines->AddRange(lines);
                }
            }

            if (featureDataLines->Count > 0) {
                int i_feature = 0;
                CodecData^ codecData = nullptr;
                bool lowPower = false;
                array<UInt64>^ codecAvailableFeatures = nullptr;
                for (int iline = 0; iline < featureDataLines->Count; iline++) {
                    if (featureDataLines[iline]->Contains(L"Codec:")) {
                        if (codecData != nullptr) {
                            codecData->setFeatures(lowPower, codecAvailableFeatures);
                            codecAvailableFeatures = nullptr;
                        }
                        lowPower = featureDataLines[iline]->Contains(L"FF");
                        codecAvailableFeatures = gcnew array<UInt64>(_countof(list_rate_control_ry));
                        for (int i_rate_control = 0; i_rate_control < _countof(list_rate_control_ry); i_rate_control++) {
                            codecAvailableFeatures[i_rate_control] = 0;
                        }
                        for (int icodec = 0; list_out_enc_codec[icodec].desc; icodec++) {
                            array<wchar_t>^ delimiterChars = { L' ', L'/' };
                            String^ codecName = String(list_out_enc_codec[icodec].desc).ToString();
                            auto codecNames = codecName->Split(delimiterChars);
                            for (int in = 0; in < codecNames->Length; in++) {
                                if (codecNames[in]->Length > 0 && featureDataLines[iline]->Contains(codecNames[in])) {
                                    const auto codec = get_out_enc_codec_by_index(icodec);
                                    codecData = getCodecData(codec);
                                    break;
                                }
                            }
                        }
                        i_feature = 0;
                        iline++;
                    } else if (codecData != nullptr) {
                        int i_rate_control = 0;
                        for (int j = _tcslen(list_enc_feature_params[0].desc)+1; j < featureDataLines[iline]->Length; j++) {
                            auto line = featureDataLines[iline];
                            auto c = line[j];
                            if (c == L'o') {
                                codecAvailableFeatures[i_rate_control] |= list_enc_feature_params[i_feature].value;
                                i_rate_control++;
                            } else if (c == L'x') {
                                i_rate_control++;
                            }
                        }
                        i_feature++;
                    }
                }
                if (codecData != nullptr) {
                    codecData->setFeatures(lowPower, codecAvailableFeatures);
                    codecAvailableFeatures = nullptr;
                }
            }
            getFeaturesFinished_ = true;
            if (!use_cache && featureDataLines->Count > 0) {
                StreamWriter^ sw = gcnew StreamWriter(cacheFile, false);
                for (int iline = 0; iline < featureDataLines->Count; iline++) {
                    sw->WriteLine(featureDataLines[iline]);
                }
                sw->Close();
            }
        }
    };

    ref class QSVFeatures {
    private:
        Thread^ thGetLibVersion;

        array<String^>^ environmentInfo;
        array<QSVDevFeatures^>^ devList;
        UInt64 availableVppFeatures;
        bool hardware;
        bool getLibVerFinished;
        bool getFeaturesFinished;
        mfxU32 mfxVer;
        String^ exePath;
        String^ gpuname;
    public:
        QSVFeatures(bool _hardware, String^ _exePath) {

            thGetLibVersion = nullptr;
            environmentInfo = nullptr;
            hardware = _hardware;
            exePath = _exePath;
            getLibVerFinished = false;
            getFeaturesFinished = false;
            gpuname = nullptr;

            devList = nullptr;

            thGetLibVersion = gcnew Thread(gcnew ThreadStart(this, &QSVFeatures::getLibVersion));
            thGetLibVersion->Start();
        }
        ~QSVFeatures() {
            if (thGetLibVersion != nullptr && thGetLibVersion->IsAlive) {
                thGetLibVersion->Join();
            }
            delete devList;
        }
        int getRCIdx(mfxU32 rc) {
            for (int i_rate_control = 0; i_rate_control < _countof(list_rate_control_ry); i_rate_control++) {
                if ((mfxU32)(list_rate_control_ry[i_rate_control].value) == rc) {
                    return i_rate_control;
                }
            }
            return -1;
        }
        QSVDevFeatures ^getDevFeatures(String^ dev) {
            if (devList) {
                for (int i = 0; i < (int)devList->Length; i++) {
                    if (devList[i]->devName() == dev) {
                        return devList[i];
                    }
                }
            }
            return nullptr;
        }
        int devCount() {
            return (devList == nullptr) ? 0 : devList->Length;
        }
        QSVDevFeatures ^getDevFeatures(const int dev_index) {
            if (devList == nullptr || dev_index < 0 || devList->Length <= dev_index) {
                return nullptr;
            }
            return devList[dev_index];
        }
        QSVDeviceNum getDevID(String^ dev) {
            QSVDevFeatures^ devf = getDevFeatures(dev);
            if (devf == nullptr) {
                return QSVDeviceNum::AUTO;
            }
            return devf->devID();
        }
        int getDevIndex(QSVDeviceNum num) {
            if (devList) {
                for (int i = 0; i < (int)devList->Length; i++) {
                    if (devList[i]->devID() == num) {
                        return i;
                    }
                }
            }
            return 0;
        }
        QSVDeviceNum getDevID(const int dev_index) {
            QSVDevFeatures^ devf = getDevFeatures(dev_index);
            if (devf == nullptr) {
                return QSVDeviceNum::AUTO;
            }
            return devf->devID();
        }
        bool getCodecAvail(const int dev_index, const RGY_CODEC codec) {
            QSVDevFeatures^ devf = getDevFeatures(dev_index);
            if (devf == nullptr) {
                return false;
            }
            return devf->codecAvail(codec);
        }
        bool getCodecAvail(const int dev_index, const RGY_CODEC codec, bool lowpower) {
            QSVDevFeatures^ devf = getDevFeatures(dev_index);
            if (devf == nullptr) {
                return false;
            }
            return devf->codecAvail(codec, lowpower);
        }
        QSVFunctionMode getAutoSelectFunctionMode(const int dev_index, const int rc_index, const RGY_CODEC codec) {
            QSVDevFeatures^ devf = getDevFeatures(dev_index);
            if (devf == nullptr) {
                return QSVFunctionMode::FF;
            }
            auto feature = devf->featureOfRC(rc_index, codec, true);
            return ((feature & ENC_FEATURE_CURRENT_RC) != 0) ? QSVFunctionMode::FF : QSVFunctionMode::PG;
        }
        bool getRCAvail(const int dev_index, const int rc_index, const RGY_CODEC codec, const QSVFunctionMode mode) {
            return (getFeatureOfRC(dev_index, rc_index, codec, mode) & ENC_FEATURE_CURRENT_RC) != 0;
        }
        UInt64 getFeatureOfRC(const int dev_index, const int rc_index, const RGY_CODEC codec, const QSVFunctionMode mode) {
            QSVDevFeatures^ devf = getDevFeatures(dev_index);
            if (devf == nullptr) {
                return false;
            }
            if (mode == QSVFunctionMode::Auto) {
                auto feature = devf->featureOfRC(rc_index, codec, true);
                if ((feature & ENC_FEATURE_CURRENT_RC) == 0) {
                    feature = devf->featureOfRC(rc_index, codec, false);
                }
                return feature;
            } else {
                return devf->featureOfRC(rc_index, codec, mode == QSVFunctionMode::FF);
            }
        }
        DataTable^ getFeatureTable(const int dev_index, const RGY_CODEC codec, const bool lowpower, const bool reGenerateTable) {
            QSVDevFeatures^ devf = getDevFeatures(dev_index);
            if (devf == nullptr) {
                return nullptr;
            }
            return devf->getTable(codec, lowpower, reGenerateTable);
        }
        bool checkIfGetFeaturesFinished() {
            if (devList) {
                getFeaturesFinished = true;
                for (int i = 0; i < (int)devList->Length; i++) {
                    getFeaturesFinished &= devList[i]->getFeaturesFinished();
                }
            }
            return getFeaturesFinished;
        }
        UInt64 getVppFeatures() {
            return 0; //未実装
        }
        bool checkIfGetLibVerAvailable() {
            return getLibVerFinished;
        }
        UInt32 GetmfxLibVer() {
            if (!getLibVerFinished) {
                thGetLibVersion->Join();
                getLibVerFinished = true;
            }
            return mfxVer;
        }
        String^ GetGPUName() {
            if (!getLibVerFinished) {
                thGetLibVersion->Join();
                getLibVerFinished = true;
            }
            return gpuname;
        }
        array<String^>^ GetDeviceNames() {
            if (!getLibVerFinished) {
                thGetLibVersion->Join();
                getLibVerFinished = true;
            }
            array<String^>^ devNames = nullptr;
            if (devList) {
                devNames = gcnew array<String^>(devList->Length);
                for (int i = 0; i < devList->Length; i++) {
                    devNames[i] = devList[i]->devName();
                }
            }
            return devNames;
        }
    private:
        System::Void getLibVersion() {
            if (exePath == nullptr || !System::IO::File::Exists(exePath)) {
                getFeaturesFinished = true;
                return;
            }

            char exe_path[1024];
            GetCHARfromString(exe_path, sizeof(exe_path), exePath);

            List<String^>^ devNames = gcnew List<String^>();
            std::vector<char> buffer(64 * 1024);
            if (get_exe_message(exe_path, "--check-environment-auo", buffer.data(), buffer.size(), AUO_PIPE_MUXED) == RP_SUCCESS) {
                environmentInfo = String(buffer.data()).ToString()->Split(String(L"\r\n").ToString()->ToCharArray(), System::StringSplitOptions::RemoveEmptyEntries);
                for (int i = 0; i < environmentInfo->Length; i++) {
                    if (environmentInfo[i]->Contains(L"Hardware API")) {
                        auto ver = environmentInfo[i]->Substring(environmentInfo[i]->IndexOf(L"Hardware API") + String(L"Hardware API").ToString()->Length + 1);
                        auto ver2 = ver->Substring(ver->IndexOf(L"v") + 1)->Split(String(L".").ToString()->ToCharArray(), System::StringSplitOptions::RemoveEmptyEntries);
                        try {
                            mfxVersion version;
                            version.Major = System::Int16::Parse(ver2[0]);
                            version.Minor = System::Int16::Parse(ver2[1]);
                            mfxVer = version.Version;
                        } catch (...) {
                            mfxVer = 0;
                        }
                    }
                    if (environmentInfo[i]->Contains(L"GPU: ")) {
                        gpuname = environmentInfo[i]->Substring(String(L"GPU: ").ToString()->Length);
                    }
                    if (environmentInfo[i]->Contains("Device")) {
                        devNames->Add(environmentInfo[i]->Replace("Device ", ""));
                    }
                }
            }

            devList = gcnew array<QSVDevFeatures^>(devNames->Count > 0 ? devNames->Count : 1);
            for (int i = 0; i < devList->Length; i++) {
                QSVDeviceNum id = QSVDeviceNum::AUTO;
                String^ deviceName = "dev#0";
                if (devNames->Count > 0) {
                    deviceName = devNames[i];
                    const int idstart = devNames[i]->IndexOf(L"#") + 1;
                    const int idend = devNames[i]->IndexOf(L":");
                    try {
                        id = (QSVDeviceNum)System::Int32::Parse(devNames[i]->Substring(idstart, idend - idstart));
                    } catch (...) {
                        id = QSVDeviceNum(i + 1);
                    }
                }
                devList[i] = gcnew QSVDevFeatures(id, deviceName, exePath, environmentInfo);
                devList[i]->init();
                devList[i]->startGetFeatures();
            }
        }
    };
};
