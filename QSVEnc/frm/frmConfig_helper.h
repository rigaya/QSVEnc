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
#include "qsv_query.h"
#include "mfxstructures.h"

#define HIDE_MPEG2

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

const CX_DESC list_deinterlace_ja[] = {
    { "なし",                       MFX_DEINTERLACE_NONE        },
    { "インタレ解除 (通常)",        MFX_DEINTERLACE_NORMAL      },
    { "インタレ解除 (24fps化)",     MFX_DEINTERLACE_IT          },
    { "インタレ解除 (Bob化)",       MFX_DEINTERLACE_BOB         },
    { "自動フィールドシフト",       100 },
    { "nnedi",                      101 },
    { NULL, NULL }
};

static const wchar_t *const list_vpp_afs_analyze[] = {
    L"0 - 解除なし",
    L"1 - フィールド三重化",
    L"2 - 縞検出二重化",
    L"3 - 動き検出二重化",
    L"4 - 動き検出補間",
    NULL
};

const CX_DESC list_vpp_nnedi_pre_screen_gui[] = {
    { _T("none"),           VPP_NNEDI_PRE_SCREEN_NONE },
    { _T("original"),       VPP_NNEDI_PRE_SCREEN_ORIGINAL },
    { _T("new"),            VPP_NNEDI_PRE_SCREEN_NEW },
    { _T("original_block"), VPP_NNEDI_PRE_SCREEN_ORIGINAL_BLOCK },
    { _T("new_block"),      VPP_NNEDI_PRE_SCREEN_NEW_BLOCK },
    { NULL, NULL }
};

const CX_DESC list_vpp_yadif_mode_gui[] = {
    //{ _T("normal"),        VPP_YADIF_MODE_AUTO },
    //{ _T("bob"),           VPP_YADIF_MODE_BOB_AUTO },
    { NULL, NULL }
};

const CX_DESC list_encmode[] = {
    { "ビットレート指定 - CBR",           MFX_RATECONTROL_CBR    },
    { "ビットレート指定 - VBR",           MFX_RATECONTROL_VBR    },
    { "ビットレート指定 - AVBR",          MFX_RATECONTROL_AVBR   },
    { "ビットレート指定 - QVBR",          MFX_RATECONTROL_QVBR   },
    { "固定量子化量 (CQP)",               MFX_RATECONTROL_CQP    },
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
    { "VP9", MFX_CODEC_VP9 },
    { "AV1", MFX_CODEC_AV1 },
    { NULL, NULL }
};
//下記は一致していないといけない
static_assert(_countof(list_outtype)-1/*NULLの分*/ == _countof(CODEC_LIST_AUO));

const CX_DESC list_log_level_jp[] = {
    { "通常",                  RGY_LOG_INFO  },
    { "音声/muxのログも表示 ", RGY_LOG_MORE  },
    { "デバッグ用出力も表示 ", RGY_LOG_DEBUG },
    { NULL, NULL }
};

static const wchar_t *const list_vpp_deinterlacer[] = {
    L"なし",
    L"自動フィールドシフト",
    L"nnedi",
    NULL
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

    value struct TrackBarNU {
        TrackBar ^TB;
        NumericUpDown ^NU;
    };

    ref class QSVFeatures {
    private:
        Thread^ thGetLibVersion;
        Thread^ thGetFeatures;

        array<String^>^ environmentInfo;
        array<mfxU32>^ codecIdList;
        array<bool>^ codecAvail;
        array<array<UInt64>^>^ availableFeatures;
        UInt64 availableVppFeatures;
        bool hardware;
        bool getLibVerFinished;
        bool getFeaturesFinished;
        mfxU32 mfxVer;
        array<DataTable^>^ dataTableQsvCodecFeatures;
        String^ exePath;
        String^ gpuname;
    public:
        QSVFeatures(bool _hardware, String^ _exePath) {

            thGetLibVersion = nullptr;
            thGetFeatures = nullptr;
            environmentInfo = nullptr;
            hardware = _hardware;
            exePath = _exePath;
            availableFeatures = nullptr;
            getLibVerFinished = false;
            getFeaturesFinished = false;
            gpuname = nullptr;

            int codecCount = 0;
            while (list_outtype[codecCount].desc)
                codecCount++;

            codecIdList = gcnew array<mfxU32>(codecCount);
            dataTableQsvCodecFeatures = gcnew array<DataTable^>(codecCount);
            for (int i_codec = 0; i_codec < codecCount; i_codec++) {
                auto codecDataTable = gcnew DataTable();
                codecDataTable->Columns->Add(L"機能");
                for (int i_rc = 0; i_rc < _countof(list_rate_control_ry); i_rc++) {
                    codecDataTable->Columns->Add(String(list_rate_control_ry[i_rc].desc).ToString()->TrimEnd());
                }
                dataTableQsvCodecFeatures[i_codec] = codecDataTable;
                codecIdList[i_codec] = list_outtype[i_codec].value;
            }

            availableFeatures = gcnew array<array<UInt64>^>(codecCount);
            for (int i_codec = 0; i_codec < codecCount; i_codec++) {
                availableFeatures[i_codec] = gcnew array<UInt64>(_countof(list_rate_control_ry));
                for (int i_rc = 0; i_rc < _countof(list_rate_control_ry); i_rc++) {
                    availableFeatures[i_codec][i_rc] = 0;
                }
            }

            codecAvail = gcnew array<bool>(codecCount);
            for (int i_codec = 0; i_codec < codecCount; i_codec++) {
                codecAvail[i_codec] = true;
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
        int getRCIdx(mfxU32 rc) {
            for (int i_rate_control = 0; i_rate_control < _countof(list_rate_control_ry); i_rate_control++) {
                if ((mfxU32)(list_rate_control_ry[i_rate_control].value) == rc) {
                    return i_rate_control;
                }
            }
            return -1;
        }
        bool checkIfGetFeaturesFinished() {
            return getFeaturesFinished;
        }
        bool getCodecAvail(mfxU32 codecId) {
            if (thGetFeatures != nullptr && thGetFeatures->IsAlive) {
                thGetFeatures->Join();
            }
            int codecIdx = getCodecIdIdx(codecId);
            if (codecIdx < 0 || availableFeatures->Length <= codecIdx) {
                return false;
            }
            return codecAvail[codecIdx];
        }
        UInt64 getFeatureOfRC(int rc_index, mfxU32 codecId) {
            if (thGetFeatures != nullptr && thGetFeatures->IsAlive) {
                thGetFeatures->Join();
            }
            int codecIdx = getCodecIdIdx(codecId);
            if (codecIdx < 0 || availableFeatures->Length <= codecIdx) {
                return 0;
            }
            if (rc_index < 0 || availableFeatures[codecIdx]->Length <= rc_index) {
                return 0;
            }
            return availableFeatures[codecIdx][rc_index];
        }
        UInt64 getVppFeatures() {
            return 0; //未実装
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
        String^ GetGPUName() {
            if (!getLibVerFinished) {
                thGetLibVersion->Join();
                getLibVerFinished = true;
            }
            return gpuname;
        }
    private:
        System::Void getLibVersion() {
            if (exePath == nullptr || !System::IO::File::Exists(exePath)) {
                getFeaturesFinished = true;
                return;
            }

            char exe_path[1024];
            std::vector<char> buffer(64 * 1024);
            GetCHARfromString(exe_path, sizeof(exe_path), exePath);
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
                        break;
                    }
                    if (environmentInfo[i]->Contains(L"GPU: ")) {
                        gpuname = environmentInfo[i]->Substring(String(L"GPU: ").ToString()->Length);
                    }
                }
            }
            thGetFeatures = gcnew Thread(gcnew ThreadStart(this, &QSVFeatures::getFeatures));
            thGetFeatures->Start();
        }
        System::Void getFeatures() {
            if (exePath == nullptr || !System::IO::File::Exists(exePath)) {
                getFeaturesFinished = true;
                return;
            }

            char exe_path[1024];
            GetCHARfromString(exe_path, sizeof(exe_path), exePath);
            std::vector<char> buffer(128 * 1024);

            //feature情報のキャッシュファイル
            //先頭に--check-environment-auoの結果
            //後半に--check-features-auoの結果
            String^ cacheFile = exePath + L".featureCache.txt";

            bool use_cache = false;
            auto featureDataLines = gcnew List<String^>();
            //キャッシュファイルがあれば、そちらを読むことを考える
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
                for (int i = 0; i < environmentInfo->Length; i++) {
                    //不一致だったらキャッシュファイルは使用しない
                    if (featureDataLines->Count <= i) {
                        use_cache = false;
                        break;
                    }
                    int compare_len = environmentInfo[i]->Length;
                    if (environmentInfo[i]->Contains(L"RAM: ")) {
                        continue; //使用メモリ量は実行するたびに変わる
                    }
                    if (environmentInfo[i]->Contains(L"CPU:")) {
                        //TBの周波数は取得するタイミングで変わりうる
                        auto len = environmentInfo[i]->IndexOf(L"@");
                        if (len < 0) {
                            len = environmentInfo[i]->IndexOf(L"[");
                        }
                        if (len < 0) {
                            len = environmentInfo[i]->IndexOf(L"(");
                        }
                        if (len > 0) {
                            compare_len = len;
                        }
                    }
                    if (String::Compare(environmentInfo[i], 0, featureDataLines[i], 0, compare_len) != 0) {
                        use_cache = false;
                        break;
                    }
                }
                if (!use_cache) {
                    featureDataLines = gcnew List<String^>();
                    featureDataLines->AddRange(environmentInfo);
                }
            } else {
                featureDataLines->AddRange(environmentInfo);
            }
            //キャッシュを使用できない場合は、実際に情報を取得する(時間がかかる)
            if (!use_cache
                && get_exe_message(exe_path, "--check-features-auo", buffer.data(), buffer.size(), AUO_PIPE_MUXED) == RP_SUCCESS) {
                auto lines = String(buffer.data()).ToString()->Split(String(L"\r\n").ToString()->ToCharArray(), System::StringSplitOptions::RemoveEmptyEntries);
                featureDataLines->AddRange(lines);
            }
            if (featureDataLines->Count > 0) {
                int i_feature = 0;
                mfxU32 codec = 0;
                for (int i_codec = 0; i_codec < codecAvail->Length; i_codec++) {
                    codecAvail[i_codec] = false;
                }

                array<UInt64>^ codecAvailableFeatures = nullptr;
                for (int iline = 0; iline < featureDataLines->Count; iline++) {
                    if (featureDataLines[iline]->Contains(L"Codec:")) {
                        if (codec != 0) {
                            int codecIdx = getCodecIdIdx(codec);
                            availableFeatures[codecIdx] = codecAvailableFeatures;
                            codecAvail[codecIdx] = true;
                        }
                        codecAvailableFeatures = gcnew array<UInt64>(_countof(list_rate_control_ry));
                        for (int i_rate_control = 0; i_rate_control < _countof(list_rate_control_ry); i_rate_control++) {
                            codecAvailableFeatures[i_rate_control] = 0;
                        }
                        for (int icodec = 0; list_outtype[icodec].desc; icodec++) {
                            array<wchar_t>^ delimiterChars = { L' ', L'/' };
                            String^ codecName = String(list_outtype[icodec].desc).ToString();
                            auto codecNames = codecName->Split(delimiterChars);
                            for (int in = 0; in < codecNames->Length; in++) {
                                if (codecNames[in]->Length > 0 && featureDataLines[iline]->Contains(codecNames[in])) {
                                    codec = list_outtype[icodec].value;
                                    break;
                                }
                            }
                        }
                        i_feature = 0;
                        iline++;
                    } else if (codec != 0) {
                        int i_rate_control = 0;
                        for (int j = _tcslen(list_enc_feature[0].desc); j < featureDataLines[iline]->Length; j++) {
                            auto line = featureDataLines[iline];
                            auto c = line[j];
                            if (c == L'o') {
                                codecAvailableFeatures[i_rate_control] |= list_enc_feature[i_feature].value;
                                i_rate_control++;
                            } else if (c == L'x') {
                                i_rate_control++;
                            }
                        }
                        i_feature++;
                    }
                }
                if (codec != 0) {
                    int codecIdx = getCodecIdIdx(codec);
                    availableFeatures[codecIdx] = codecAvailableFeatures;
                    codecAvail[codecIdx] = true;
                }

                for (int codecIdx = 0; codecIdx < availableFeatures->Length; codecIdx++) {
                    uint64_t feat = 0;
                    feat |= (availableFeatures[codecIdx][getRCIdx(MFX_RATECONTROL_AVBR)]   & ENC_FEATURE_CURRENT_RC) ? ENC_FEATURE_AVBR   : 0;
                    feat |= (availableFeatures[codecIdx][getRCIdx(MFX_RATECONTROL_QVBR)]   & ENC_FEATURE_CURRENT_RC) ? ENC_FEATURE_QVBR   : 0;
                    feat |= (availableFeatures[codecIdx][getRCIdx(MFX_RATECONTROL_LA)]     & ENC_FEATURE_CURRENT_RC) ? ENC_FEATURE_LA     : 0;
                    feat |= (availableFeatures[codecIdx][getRCIdx(MFX_RATECONTROL_LA_HRD)] & ENC_FEATURE_CURRENT_RC) ? ENC_FEATURE_LA_HRD : 0;
                    feat |= (availableFeatures[codecIdx][getRCIdx(MFX_RATECONTROL_ICQ)]    & ENC_FEATURE_CURRENT_RC) ? ENC_FEATURE_ICQ    : 0;
                    feat |= (availableFeatures[codecIdx][getRCIdx(MFX_RATECONTROL_VCM)]    & ENC_FEATURE_CURRENT_RC) ? ENC_FEATURE_VCM    : 0;
                    for (int i_rate_control = 0; i_rate_control < _countof(list_rate_control_ry); i_rate_control++) {
                        availableFeatures[codecIdx][i_rate_control] |= feat;
                    }
                }
            }
            for (int codecIdx = 0; list_outtype[codecIdx].desc; codecIdx++) {
                GenerateTable(list_outtype[codecIdx].value);
            }
            getFeaturesFinished = true;
            if (!use_cache && featureDataLines->Count > 0) {
                StreamWriter^ sw = gcnew StreamWriter(cacheFile, false);
                for (int iline = 0; iline < featureDataLines->Count; iline++) {
                    sw->WriteLine(featureDataLines[iline]);
                }
                sw->Close();
            }
        }
        System::Void GenerateTable(mfxU32 codecId) {
            static const FEATURE_DESC list_enc_feature_jp[] = {
                { _T("モード有効      "), ENC_FEATURE_CURRENT_RC             },
                { _T("10bit深度       "), ENC_FEATURE_10BIT_DEPTH            },
                { _T("完全HWモード    "), ENC_FEATURE_FIXED_FUNC             },
                { _T("インタレ保持    "), ENC_FEATURE_INTERLACE              },
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
                { _T(" +多Bframe     "),  ENC_FEATURE_B_PYRAMID_MANY_BFRAMES },
                { _T("MB単位レート制御"), ENC_FEATURE_MBBRC                  },
                //{ _T("ExtBRC          "), ENC_FEATURE_EXT_BRC                },
                { _T("先行探索品質    "), ENC_FEATURE_LA_DS                  },
                { _T("最大/最小 QP    "), ENC_FEATURE_QP_MINMAX              },
                { _T("周期的ｲﾝﾄﾗ更新  "), ENC_FEATURE_INTRA_REFRESH          },
                { _T("No-Deblock      "), ENC_FEATURE_NO_DEBLOCK             },
                { _T("MBQP(CQP)       "), ENC_FEATURE_PERMBQP                },
                { _T("ﾀﾞｲﾚｸﾄﾓｰﾄﾞ最適化"), ENC_FEATURE_DIRECT_BIAS_ADJUST     },
                { _T("MVコスト調整    "), ENC_FEATURE_GLOBAL_MOTION_ADJUST   },
                { _T("SAO             "), ENC_FEATURE_HEVC_SAO               },
                { _T("最大 CTU Size   "), ENC_FEATURE_HEVC_CTU               },
                { _T("TSkip           "), ENC_FEATURE_HEVC_TSKIP             },
                { NULL, 0 },
            };

            const int codecIdx = getCodecIdIdx(codecId);
            if (codecIdx < 0) {
                return;
            }

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
