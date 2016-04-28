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

#pragma once

using namespace System;
using namespace System::Data;
using namespace System::Threading;
using namespace System::IO;
using namespace System::Collections::Generic;

#include "qsv_prm.h"
#include "qsv_util.h"
#include "mfxstructures.h"

#define HIDE_MPEG2

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

const CX_DESC list_deinterlace_ja[] = {
    { "なし",                       MFX_DEINTERLACE_NONE        },
    { "インタレ解除 (通常)",        MFX_DEINTERLACE_NORMAL      },
    { "インタレ解除 (24fps化)",     MFX_DEINTERLACE_IT          },
    { "インタレ解除 (Bob化)",       MFX_DEINTERLACE_BOB         },
#if ENABLE_ADVANCED_DEINTERLACE
    { "インタレ解除 (固定24fps化)", MFX_DEINTERLACE_IT_MANUAL   },
    { "インタレ解除 (自動)",        MFX_DEINTERLACE_AUTO_SINGLE },
    { "インタレ解除 (自動Bob化)",   MFX_DEINTERLACE_AUTO_DOUBLE },
#endif
    { NULL, NULL } 
};

const CX_DESC list_encmode[] = {
    { "ビットレート指定 - CBR",           MFX_RATECONTROL_CBR    },
    { "ビットレート指定 - VBR",           MFX_RATECONTROL_VBR    },
    { "ビットレート指定 - AVBR",          MFX_RATECONTROL_AVBR   },
    { "ビットレート指定 - QVBR",          MFX_RATECONTROL_QVBR   },
    { "固定量子化量 (CQP)",               MFX_RATECONTROL_CQP    },
    { "可変QP (VQP, プログレッシブのみ)", MFX_RATECONTROL_VQP    },
    { "先行探索レート制御",               MFX_RATECONTROL_LA     },
    { "先行探索レート制御 (HRD準拠)",     MFX_RATECONTROL_LA_HRD },
    { "固定品質モード",                   MFX_RATECONTROL_ICQ    },
    { "先行探索付き固定品質モード",       MFX_RATECONTROL_LA_ICQ },
    { "ビデオ会議モード",                 MFX_RATECONTROL_VCM    },
    { NULL, NULL }
};

const CX_DESC list_rotate_angle_ja[] = {
    { _T("0°"),     MFX_ANGLE_0    },
    { _T("90°"),    MFX_ANGLE_90   },
    { _T("180°"),   MFX_ANGLE_180  },
    { _T("270°"),   MFX_ANGLE_270  },
    { NULL, 0 }
};

const WCHAR * const audio_enc_timing_desc[] = {
    L"後",
    L"前",
    L"同時",
    NULL
};

const CX_DESC list_outtype[] = {
    { "H.264 / AVC",  MFX_CODEC_AVC  },
    { "H.265 / HEVC", MFX_CODEC_HEVC },
#ifndef HIDE_MPEG2
    { "MPEG2", MFX_CODEC_MPEG2 },
#endif
    //{ "VC-1", MFX_CODEC_VC1 },
    { NULL, NULL }
};

const CX_DESC list_log_level_jp[] = {
    { "通常",                  QSV_LOG_INFO  },
    { "音声/muxのログも表示 ", QSV_LOG_MORE  },
    { "デバッグ用出力も表示 ", QSV_LOG_DEBUG },
    { NULL, NULL }
};

//メモ表示用 RGB
const int StgNotesColor[][3] = {
    {  80,  72,  92 },
    { 120, 120, 120 }
};

const WCHAR * const DefaultStgNotes = L"メモ...";


namespace QSVEnc {

    ref class LocalSettings {
    public:
        List<String^>^ audEncName;
        List<String^>^ audEncExeName;
        List<String^>^ audEncPath;
        String^ MP4MuxerExeName;
        String^ MP4MuxerPath;
        String^ MKVMuxerExeName;
        String^ MKVMuxerPath;
        String^ TC2MP4ExeName;
        String^ TC2MP4Path;
        String^ MPGMuxerExeName;
        String^ MPGMuxerPath;
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

    ref class QSVFeatures {
    private:
        Thread^ thGetLibVersion;
        Thread^ thGetFeatures;

        array<mfxU32>^ codecIdList;
        array<array<UInt64>^>^ availableFeatures;
        UInt64 availableVppFeatures;
        bool hardware;
        bool getLibVerFinished;
        bool getFeaturesFinished;
        mfxU32 mfxVer;
        array<DataTable^>^ dataTableQsvCodecFeatures;
    public:
        QSVFeatures(bool _hardware) {

            thGetLibVersion = nullptr;
            thGetFeatures = nullptr;
            hardware = _hardware;
            availableFeatures = nullptr;
            getLibVerFinished = false;
            getFeaturesFinished = false;

            int codecCount = 0;
            while (list_outtype[codecCount].desc)
                codecCount++;

            codecIdList = gcnew array<mfxU32>(codecCount);
            dataTableQsvCodecFeatures = gcnew array<DataTable^>(codecCount);
            for (int i_codec = 0; i_codec < codecCount; i_codec++) {
                auto codecDataTable = gcnew DataTable();
                codecDataTable->Columns->Add(L"機能");
                for (int i_rc = 0; i_rc < _countof(list_rate_control_ry); i_rc++)
                    codecDataTable->Columns->Add(String(list_rate_control_ry[i_rc].desc).ToString()->TrimEnd());
                dataTableQsvCodecFeatures[i_codec] = codecDataTable;
                codecIdList[i_codec] = list_outtype[i_codec].value;
            }

            thGetLibVersion = gcnew Thread(gcnew ThreadStart(this, &QSVFeatures::getLibVersion));
            thGetLibVersion->Start();
        }
        ~QSVFeatures() {
            if (thGetLibVersion != nullptr && thGetLibVersion->IsAlive) {
                thGetLibVersion->Join();
            }
            if (thGetFeatures != nullptr && thGetFeatures->IsAlive) {
                thGetFeatures->Join();
            }
            delete dataTableQsvCodecFeatures;
            delete availableFeatures;
        }
        int getCodecIdIdx(mfxU32 codecId) {
            for (int i = 0; i < (int)codecIdList->Length; i++) {
                if (codecIdList[i] == codecId) {
                    return i;
                }
            }
            return -1;
        }
        bool checkIfGetFeaturesFinished() {
            return getFeaturesFinished;
        }
        UInt64 getFeatureOfRC(int rc_index, mfxU32 codecId) {
            if (getFeaturesFinished) {
                return availableFeatures[getCodecIdIdx(codecId)][rc_index];
            }
            mfxVersion version;
            version.Version = mfxVer;
            return CheckEncodeFeature(hardware, version, (mfxU16)list_rate_control_ry[rc_index].value, codecId);
        }
        UInt64 getVppFeatures() {
            if (getFeaturesFinished) {
                return availableVppFeatures;
            }
            mfxVersion version;
            version.Version = mfxVer;
            return CheckVppFeatures(hardware, version);
        }
        DataTable^ getFeatureTable(mfxU32 codecId) {
            return dataTableQsvCodecFeatures[getCodecIdIdx(codecId)];
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
    private:
        System::Void getLibVersion() {
            mfxVer = (hardware) ? get_mfx_libhw_version().Version : get_mfx_libsw_version().Version;
            thGetFeatures = gcnew Thread(gcnew ThreadStart(this, &QSVFeatures::getFeatures));
            thGetFeatures->Start();
        }
        System::Void getFeatures() {
            if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_1.Version)) {
                if (availableFeatures == nullptr) {
                    //MakeFeatureListが少し時間かかるので非同期にする必要がある
                    mfxVersion version;
                    version.Version = mfxVer;
                    std::vector<mfxU32> _codecIdList;
                    for each(auto codec in codecIdList) {
                        _codecIdList.push_back(codec);
                    }
                    auto featuresPerCodec = MakeFeatureListPerCodec(hardware, version, make_vector(list_rate_control_ry), _codecIdList);

                    availableFeatures = gcnew array<array<UInt64>^>(codecIdList->Length);
                    for (int j = 0; j < (int)featuresPerCodec.size(); j++) {
                        auto codecAvailableFeatures = gcnew array<UInt64>(_countof(list_rate_control_ry));
                        for (int i = 0; i < _countof(list_rate_control_ry); i++) {
                            codecAvailableFeatures[i] = featuresPerCodec[j][i];
                        }
                        availableFeatures[j] = codecAvailableFeatures;
                        GenerateTable(codecIdList[j]);
                    }
                    availableVppFeatures = CheckVppFeatures(hardware, version);
                    getFeaturesFinished = true;
                }
            }
        }
        System::Void GenerateTable(mfxU32 codecId) {
            static const FEATURE_DESC list_enc_feature_jp[] = {
                { _T("モード有効      "), ENC_FEATURE_CURRENT_RC             },
                { _T("完全HWモード    "), ENC_FEATURE_FIXED_FUNC             },
                { _T("インタレ保持    "), ENC_FEATURE_INTERLACE              },
                { _T("シーンチェンジ  "), ENC_FEATURE_SCENECHANGE            },
                { _T("色設定等出力    "), ENC_FEATURE_VUI_INFO               },
                //{ _T("aud             "), ENC_FEATURE_AUD                    },
                //{ _T("pic_struct      "), ENC_FEATURE_PIC_STRUCT             },
                { _T("Trellis         "), ENC_FEATURE_TRELLIS                },
                //{ _T("rdo             "), ENC_FEATURE_RDO                    },
                //{ _T("CAVLC           "), ENC_FEATURE_CAVLC                  },
                { _T("適応的Iフレーム "), ENC_FEATURE_ADAPTIVE_I             },
                { _T("適応的Bフレーム "), ENC_FEATURE_ADAPTIVE_B             },
                { _T("重み付きPフレーム"), ENC_FEATURE_WEIGHT_P              },
                { _T("重み付きBフレーム"), ENC_FEATURE_WEIGHT_B              },
                { _T("フェード検出    "), ENC_FEATURE_FADE_DETECT            },
                { _T("ピラミッド参照  "), ENC_FEATURE_B_PYRAMID              },
                { _T(" +シーンチェンジ"), ENC_FEATURE_B_PYRAMID_AND_SC       },
                { _T(" +多Bframe     "),  ENC_FEATURE_B_PYRAMID_MANY_BFRAMES },
                { _T("MB単位レート制御"), ENC_FEATURE_MBBRC                  },
                { _T("ExtBRC          "), ENC_FEATURE_EXT_BRC                },
                { _T("先行探索品質    "), ENC_FEATURE_LA_DS                  },
                { _T("最大/最小 QP    "), ENC_FEATURE_QP_MINMAX              },
                { _T("周期的ｲﾝﾄﾗ更新  "), ENC_FEATURE_INTRA_REFRESH          },
                { _T("No-Deblock      "), ENC_FEATURE_NO_DEBLOCK             },
                { _T("MBQP(CQP)       "), ENC_FEATURE_PERMBQP                },
                { _T("ﾀﾞｲﾚｸﾄﾓｰﾄﾞ最適化"), ENC_FEATURE_DIRECT_BIAS_ADJUST     },
                { _T("MVコスト調整    "), ENC_FEATURE_GLOBAL_MOTION_ADJUST   },
                { NULL, 0 },
            };

            const int codecIdx = getCodecIdIdx(codecId);

            //第2行以降を連続で追加していく
            for (int i = 0; list_enc_feature_jp[i].desc; i++) {
                DataRow^ drb = dataTableQsvCodecFeatures[codecIdx]->NewRow();
                drb[0] = String(list_enc_feature_jp[i].desc).ToString();
                for (int j = 1; j < dataTableQsvCodecFeatures[codecIdx]->Columns->Count; j++) {
                    drb[j] = String((availableFeatures[codecIdx][j-1] & list_enc_feature_jp[i].value) ? L"○" : L"×").ToString();
                }
                dataTableQsvCodecFeatures[codecIdx]->Rows->Add(drb);
            }
        }
    };
};
