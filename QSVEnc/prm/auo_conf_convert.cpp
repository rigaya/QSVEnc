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

#include "rgy_osdep.h"
#include <string.h>
#include <stdio.h>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "auo_util.h"
#include "auo_conf.h"
#include "qsv_query.h"
#include "rgy_version.h"
#include "rgy_perf_monitor.h"
#include "rgy_avutil.h"
#include "qsv_cmd.h"

const CX_DESC list_vpp_scaling_quality_auo_conf_old[] = {
    { _T("auto"),   MFX_SCALING_MODE_DEFAULT  },
    { _T("simple"), MFX_SCALING_MODE_LOWPOWER },
    { _T("fine"),   MFX_SCALING_MODE_QUALITY  },
    { NULL, 0 }
};

typedef struct sAudioSelectOld {
    int    nAudioSelect;               //選択した音声トラックのリスト 1,2,...(1から連番で指定)
    TCHAR *pAVAudioEncodeCodec;        //音声エンコードのコーデック
    TCHAR *pAVAudioEncodeCodecPrm;     //音声エンコードのコーデックのパラメータ
    TCHAR *pAVAudioEncodeCodecProfile; //音声エンコードのコーデックのプロファイル
    int    nAVAudioEncodeBitrate;      //音声エンコードに選択した音声トラックのビットレート
    int    nAudioSamplingRate;         //サンプリング周波数
    TCHAR *pAudioExtractFilename;      //抽出する音声のファイル名のリスト
    TCHAR *pAudioExtractFormat;        //抽出する音声ファイルのフォーマット
    TCHAR *pAudioFilter;               //音声フィルタ
    uint64_t pnStreamChannelSelect[MAX_SPLIT_CHANNELS]; //入力音声の使用するチャンネル
    uint64_t pnStreamChannelOut[MAX_SPLIT_CHANNELS];    //出力音声のチャンネル
} sAudioSelectOld;

#pragma pack(push, 4)
typedef struct {
    bool bEnable;             //use vpp

    bool bUseResize;          //use vpp resizer

    bool __unsed2;
    bool __unsed;
    mfxU16 nRotate;
    bool bUseProAmp;          //not supported
    bool bUseDenoise;         //use vpp denoise
    mfxU16 nDenoise;          // 0 - 100 Denoise Strength
    bool bUseDetailEnhance;   //use vpp detail enhancer
    bool __unsed4;
    mfxU16 nDetailEnhance;    // 0 - 100
    mfxU16 nDeinterlace;      //set deinterlace mode

    mfxU16 nImageStabilizer;  //MFX_IMAGESTAB_MODE_UPSCALE, MFX_IMAGESTAB_MODE_BOXED
    mfxU16 nFPSConversion;    //FPS_CONVERT_xxxx

    mfxU16 nTelecinePattern;

    bool bHalfTurn;
    bool __unsed3;

    struct {
        TCHAR     *pFilePath; //ロゴファイル名へのポインタ
        TCHAR     *pSelect; //選択するロゴ
        mfxI16Pair nPosOffset;
        uint8_t    nDepth;
        uint8_t    bAdd;
        mfxI16     nYOffset;
        mfxI16     nCbOffset;
        mfxI16     nCrOffset;
    } delogo;

    struct {
        int    nTrack;    //動画ファイルから字幕を抽出する場合の字幕トラック (0で無効)
        TCHAR *pFilePath; //字幕を別ファイルから読み込む場合のファイルの場所
        TCHAR *pCharEnc;  //字幕の文字コード
        int    nShaping;  //字幕を焼きこむときのモード
    } subburn;

    mfxU16 nMirrorType;  //MFX_MIRRORING_xxx
    mfxU16 nScalingQuality; //MFX_SCALING_MODE_xxx
    mfxU8 Reserved[84];
} sVppParamsOld;

struct sInputParamsOld {
    mfxU16 nInputFmt;     // RGY_INUPT_FMT_xxx
    mfxU16 nEncMode;      // RateControl
    mfxU16 nTargetUsage;  // Quality
    mfxU32 CodecId;       // H.264 only for this
    mfxU16 CodecProfile;
    mfxU16 CodecLevel;
    mfxU16 nIdrInterval;  // Idr frame interval to I frame, not supported
    mfxU16 nGOPLength;    // (Max) GOP Length
    bool   bopenGOP;      // if false, GOP_CLOSED is set
    bool   bforceGOPSettings; // if true, GOP_STRICT is set
    mfxI16 nBframes;      // set sequential Bframes num, -1 is auto.
    mfxU16 nRef;          // set ref frames num.
    mfxU16 nQPI;          // QP for I frames
    mfxU16 nQPP;          // QP for P frames
    mfxU16 nQPB;          // QP for B frames
    mfxU16 nAVBRAccuarcy;    // param for AVBR algorithm, for API v1.3
    mfxU16 nAVBRConvergence; // param for AVBR algorithm, for API v1.3

    mfxU16 nSlices;       // number of slices, 0 is auto

    mfxU16 VideoFormat;   //for API v1.3
    mfxU16 ColorMatrix;   //for API v1.3
    mfxU16 ColorPrim;     //for API v1.3
    mfxU16 Transfer;      //for API v1.3
    bool   bFullrange;    //for API v1.3

    mfxU32 ColorFormat;   //YV12 or NV12
    mfxU16 nPicStruct;    //Progressive or interlaced, and other flags
    mfxU16 nWidth;        //width of input
    mfxU16 nHeight;       //height of input
    mfxU32 nFPSRate;      //fps rate of input
    mfxU32 nFPSScale;     //fps scale of input
    mfxU16 __nBitRate;
    mfxU16 __nMaxBitrate;
    mfxI16 nLogLevel;     //ログレベル

    mfxU16 nDstWidth;     //output width
    mfxU16 nDstHeight;    //input width

    mfxU8 memType;       //use d3d surface
    bool bUseHWLib;       //use QSV (hw encoding)

    mfxU16 nInputBufSize; //input buf size

    bool   __unused;
    void  *pPrivatePrm;


    mfxI32     nPAR[2]; //PAR比
    bool       bCAVLC;  //CAVLC
    mfxU16     nInterPred;
    mfxU16     nIntraPred;
    bool       bRDO;
    mfxU16     nMVPrecision;
    mfxI16Pair MVSearchWindow;

    mfxU16     MVC_flags;
    mfxU8      nBluray;

    mfxU16     nVQPStrength;
    mfxU16     nVQPSensitivity;

    mfxU16     reserved__[4];

    mfxU16     nQuality; // quality parameter for JPEG encoder

    mfxU8      bMBBRC;
    mfxU8      bExtBRC;

    mfxU16     nLookaheadDepth;
    mfxU16     nTrellis;

    TCHAR     *pStrLogFile; //ログファイル名へのポインタ
    mfxU16     nAsyncDepth;
    mfxI16     nOutputBufSizeMB;

    mfxU16     bBPyramid;
    mfxU8      bAdaptiveI;
    mfxU8      bAdaptiveB;
    mfxU16     nLookaheadDS;

    mfxU16     nICQQuality;
    mfxU8      bBenchmark;

    mfxU8      bDisableTimerPeriodTuning;

    mfxU16     nQVBRQuality;

    mfxU8      bIntraRefresh;
    mfxU8      bNoDeblock;
    mfxU8      nQPMin[3];
    mfxU8      nQPMax[3];

    mfxU16     nWinBRCSize;

    mfxU8      nMVCostScaling;
    mfxU8      bDirectBiasAdjust;
    mfxU8      bGlobalMotionAdjust;
    mfxU8      bUseFixedFunc;
    mfxU32     nBitRate;
    mfxU32     nMaxBitrate;

    mfxU16     nTrimCount;
    sTrim     *pTrimList;
    mfxU16     inputBitDepthLuma;
    mfxU16     inputBitDepthChroma;
    mfxU8      nAVMux; //RGY_MUX_xxx
    mfxU16     nAVDemuxAnalyzeSec;

    TCHAR     *pAVMuxOutputFormat;

    mfxU8      nAudioSelectCount; //pAudioSelectの数
    sAudioSelectOld **ppAudioSelectList;

    mfxI16     nSessionThreads;
    mfxU16     nSessionThreadPriority;

    mfxU8      bCopyChapter;
    mfxU8      nAudioResampler;
    mfxU8      nVP8Sharpness;
    mfxU8      nAudioSourceCount;
    TCHAR      **ppAudioSourceList;

    mfxU16     nWeightP;
    mfxU16     nWeightB;
    mfxU16     nFadeDetect;
    mfxU16     nSubtitleSelectCount;
    int       *pSubtitleSelect;
    int64_t    nPerfMonitorSelect;
    int64_t    nPerfMonitorSelectMatplot;
    int        nPerfMonitorInterval;
    TCHAR     *pPythonPath;
    mfxU32     nBenchQuality; //ベンチマークの対象
    int8_t     nOutputThread;
    int8_t     nAudioThread;

    RGYOptList *pMuxOpt;
    TCHAR     *pChapterFile;
    uint32_t   nAudioIgnoreDecodeError;
    RGYAVSync  nAVSyncMode;     //avsyncの方法 (RGY_AVSYNC_xxx)
    uint16_t   nProcSpeedLimit; //プリデコードする場合の処理速度制限 (0で制限なし)
    int8_t     nInputThread;
    int8_t     unused;
    float      fSeekSec; //指定された秒数分先頭を飛ばす
    TCHAR     *pFramePosListLog;
    uint32_t   nFallback;
    int        nVideoStreamId;
    int8_t     nVideoTrack;
    int8_t     bOutputAud;
    int8_t     bOutputPicStruct;
    int8_t     bChapterNoTrim;
    int16_t    pQPOffset[8];
    TCHAR     *pMuxVidTsLogFile;
    TCHAR     *pAVInputFormat;
    TCHAR     *pLogCopyFrameData;

    sInputCrop sInCrop;

    mfxU16     nRepartitionCheck;
    int8_t     padding[2];
    char      *sMaxCll;
    char      *sMasterDisplay;
    int8_t     Reserved[1000];

    TCHAR strSrcFile[MAX_FILENAME_LEN];
    TCHAR strDstFile[MAX_FILENAME_LEN];

    mfxU16 nRotationAngle; //not supported

    sVppParamsOld vpp;
};

typedef struct {
    BOOL afs;                      //自動フィールドシフトの使用
    BOOL auo_tcfile_out;           //auo側でタイムコードを出力する
    int  reserved[2];
} CONF_VIDEO_OLD_V5; //動画用設定(qsv以外)

typedef struct {
    int  encoder;             //使用する音声エンコーダ
    int  bitrate;             //ビットレート指定モード
} CONF_AUDIO_DIRECT;

typedef struct {
    char        conf_name[CONF_NAME_BLOCK_LEN];  //保存時に使用
    int         size_all;                        //保存時: CONF_GUIEXの全サイズ / 設定中、エンコ中: CONF_INITIALIZED
    int         head_size;                       //ヘッダ部分の全サイズ
    int         block_count;                     //ヘッダ部を除いた設定のブロック数
    int         block_size[CONF_BLOCK_MAX];      //各ブロックのサイズ
    size_t      block_head_p[CONF_BLOCK_MAX];    //各ブロックのポインタ位置
    sInputParamsOld qsv;                         //qsvについての設定
    CONF_VIDEO_OLD_V5 vid;                       //その他動画についての設定
    CONF_AUDIO  aud;                             //音声についての設定
    CONF_MUX    mux;                             //muxについての設定
    CONF_OTHER  oth;                             //その他の設定
    CONF_AUDIO_DIRECT aud_avqsv;                 //音声についての設定
} CONF_GUIEX_OLD_V5;

#pragma pack(pop)

const int conf_block_data_old5[CONF_BLOCK_COUNT] = {
    sizeof(sInputParamsOld),
    sizeof(CONF_VIDEO_OLD_V5),
    sizeof(CONF_AUDIO),
    sizeof(CONF_MUX),
    sizeof(CONF_OTHER),
    sizeof(CONF_AUDIO_DIRECT)
};

const size_t conf_block_pointer_old5[CONF_BLOCK_COUNT] = {
    offsetof(CONF_GUIEX_OLD_V5, qsv),
    offsetof(CONF_GUIEX_OLD_V5, vid),
    offsetof(CONF_GUIEX_OLD_V5, aud),
    offsetof(CONF_GUIEX_OLD_V5, mux),
    offsetof(CONF_GUIEX_OLD_V5, oth),
    offsetof(CONF_GUIEX_OLD_V5, aud_avqsv)
};

void write_conf_header_old5(CONF_GUIEX_OLD_V5 *save_conf) {
    sprintf_s(save_conf->conf_name, sizeof(save_conf->conf_name), CONF_NAME_OLD_5);
    save_conf->size_all = sizeof(CONF_GUIEX_OLD_V5);
    save_conf->head_size = CONF_HEAD_SIZE;
    save_conf->block_count = CONF_BLOCK_COUNT;
    for (int i = 0; i < CONF_BLOCK_COUNT; ++i) {
        save_conf->block_size[i] = conf_block_data_old5[i];
        save_conf->block_head_p[i] = conf_block_pointer_old5[i];
    }
}

void *guiEx_config::convert_qsvstgv1_to_stgv3(void *_conf, int size) {
    CONF_GUIEX_OLD_V5 *conf = (CONF_GUIEX_OLD_V5 *)calloc(sizeof(CONF_GUIEX_OLD_V5), 1);
    write_conf_header_old5(conf);
    static_assert(sizeof(conf->qsv) == 3560, "sizeof(conf->enc) not equal to 3560, which will break convert_qsvstgv2_to_stgv3().");
    static_assert(sizeof(conf->vid) == 16,   "sizeof(conf->vid) not equal to 16,   which will break convert_qsvstgv2_to_stgv3().");

    //ブロック部分のコピー
    for (int i = 0; i < ((CONF_GUIEX_OLD_V5 *)_conf)->block_count; ++i) {
        BYTE *filedat = (BYTE *)_conf + ((CONF_GUIEX_OLD_V5 *)_conf)->block_head_p[i];
        BYTE *dst = (BYTE *)conf + conf_block_pointer_old5[i];
        memcpy(dst, filedat, std::min<int>(((CONF_GUIEX_OLD_V5 *)_conf)->block_size[i], conf_block_data_old5[i]));
    }

    conf->qsv.nBitRate = conf->qsv.__nBitRate;
    conf->qsv.nMaxBitrate = conf->qsv.__nMaxBitrate;
    conf->qsv.__nBitRate = 0;
    conf->qsv.__nMaxBitrate = 0;
    strcpy_s(conf->conf_name, CONF_NAME_OLD_2);

    memset(((BYTE *)conf) + size - 2056, 0, 2056);
    strcpy_s(conf->conf_name, CONF_NAME_OLD_3);
    return conf;
}

void *guiEx_config::convert_qsvstgv2_to_stgv3(void *_conf) {
    CONF_GUIEX_OLD_V5 *conf = (CONF_GUIEX_OLD_V5 *)calloc(sizeof(CONF_GUIEX_OLD_V5), 1);
    write_conf_header_old5(conf);

    //ブロック部分のコピー
    for (int i = 0; i < ((CONF_GUIEX_OLD_V5 *)_conf)->block_count; ++i) {
        BYTE *filedat = (BYTE *)_conf + ((CONF_GUIEX_OLD_V5 *)_conf)->block_head_p[i];
        BYTE *dst = (BYTE *)conf + conf_block_pointer_old5[i];
        memcpy(dst, filedat, std::min<int>(((CONF_GUIEX_OLD_V5 *)_conf)->block_size[i], conf_block_data_old5[i]));
    }

    static const DWORD OLD_FLAG_AFTER  = 0x01;
    static const DWORD OLD_FLAG_BEFORE = 0x02;

    char bat_path_before_process[1024];
    char bat_path_after_process[1024];
    strcpy_s(bat_path_after_process,  conf->oth.batfiles[0]);
    strcpy_s(bat_path_before_process, conf->oth.batfiles[2]);

    DWORD old_run_bat_flags = conf->oth.run_bat;
    conf->oth.run_bat  = 0x00;
    conf->oth.run_bat |= (old_run_bat_flags & OLD_FLAG_BEFORE) ? RUN_BAT_BEFORE_PROCESS : 0x00;
    conf->oth.run_bat |= (old_run_bat_flags & OLD_FLAG_AFTER)  ? RUN_BAT_AFTER_PROCESS  : 0x00;

    memset(&conf->oth.batfiles[0], 0, sizeof(conf->oth.batfiles));
    strcpy_s(conf->oth.batfile.before_process, bat_path_before_process);
    strcpy_s(conf->oth.batfile.after_process,  bat_path_after_process);
    strcpy_s(conf->conf_name, CONF_NAME_OLD_3);

    free(_conf);
    return conf;
}

void *guiEx_config::convert_qsvstgv3_to_stgv4(void *_conf) {
    CONF_GUIEX_OLD_V5 *conf = (CONF_GUIEX_OLD_V5 *)calloc(sizeof(CONF_GUIEX_OLD_V5), 1);
    write_conf_header_old5(conf);

    //ブロック部分のコピー
    for (int i = 0; i < ((CONF_GUIEX_OLD_V5 *)_conf)->block_count; ++i) {
        BYTE *filedat = (BYTE *)_conf + ((CONF_GUIEX_OLD_V5 *)_conf)->block_head_p[i];
        BYTE *dst = (BYTE *)conf + conf_block_pointer_old5[i];
        memcpy(dst, filedat, std::min<int>(((CONF_GUIEX_OLD_V5 *)_conf)->block_size[i], conf_block_data_old5[i]));
    }
    if (conf->qsv.nOutputBufSizeMB == 0) {
        conf->qsv.nOutputBufSizeMB = QSV_DEFAULT_OUTPUT_BUF_MB;
    } else {
        conf->qsv.nOutputBufSizeMB = clamp(conf->qsv.nOutputBufSizeMB, 0, RGY_OUTPUT_BUF_MB_MAX);
    }
    strcpy_s(conf->conf_name, CONF_NAME_OLD_4);
    free(_conf);
    return conf;
}

void *guiEx_config::convert_qsvstgv4_to_stgv5(void *_conf) {
    CONF_GUIEX_OLD_V5 *conf = (CONF_GUIEX_OLD_V5 *)calloc(sizeof(CONF_GUIEX_OLD_V5), 1);
    write_conf_header_old5(conf);

    //ブロック部分のコピー
    for (int i = 0; i < ((CONF_GUIEX_OLD_V5 *)_conf)->block_count; ++i) {
        BYTE *filedat = (BYTE *)_conf + ((CONF_GUIEX_OLD_V5 *)_conf)->block_head_p[i];
        BYTE *dst = (BYTE *)conf + conf_block_pointer_old5[i];
        memcpy(dst, filedat, std::min<int>(((CONF_GUIEX_OLD_V5 *)_conf)->block_size[i], conf_block_data_old5[i]));
    }
    if (conf->qsv.nOutputThread == 0) {
        conf->qsv.nOutputThread = RGY_OUTPUT_THREAD_AUTO;
    }
    if (conf->qsv.nAudioThread == 0) {
        conf->qsv.nAudioThread = RGY_AUDIO_THREAD_AUTO;
    }
    strcpy_s(conf->conf_name, CONF_NAME_OLD_5);
    free(_conf);
    return conf;
}

static tstring gen_cmd_oldv5(const sInputParamsOld *pParams, bool save_disabled_prm);


static RGY_CODEC conv_codec_mfx_to_rgy(const uint32_t codecId) {
    switch (codecId) {
    case MFX_CODEC_AVC:   return RGY_CODEC_H264;
    case MFX_CODEC_HEVC:  return RGY_CODEC_HEVC;
    case MFX_CODEC_MPEG2: return RGY_CODEC_MPEG2;
    case MFX_CODEC_VC1:   return RGY_CODEC_VC1;
    case MFX_CODEC_VP8:   return RGY_CODEC_VP8;
    case MFX_CODEC_VP9:   return RGY_CODEC_VP9;
    case MFX_CODEC_AV1:   return RGY_CODEC_AV1;
    default:              return RGY_CODEC_UNKNOWN;
    }
}

const CX_DESC auo_list_log_level[7] = {
    { _T("trace"), RGY_LOG_TRACE },
    { _T("debug"), RGY_LOG_DEBUG },
    { _T("more"),  RGY_LOG_MORE  },
    { _T("info"),  RGY_LOG_INFO  },
    { _T("warn"),  RGY_LOG_WARN  },
    { _T("error"), RGY_LOG_ERROR },
    { NULL, 0 }
};

void *guiEx_config::convert_qsvstgv5_to_stgv6(void *_conf) {
    CONF_GUIEX_OLD_V5 *conf_old = (CONF_GUIEX_OLD_V5 *)calloc(sizeof(CONF_GUIEX_OLD_V5), 1);
    write_conf_header_old5(conf_old);

    //ブロック部分のコピー
    for (int i = 0; i < ((CONF_GUIEX_OLD_V5 *)_conf)->block_count; ++i) {
        BYTE *filedat = (BYTE *)_conf + ((CONF_GUIEX_OLD_V5 *)_conf)->block_head_p[i];
        BYTE *dst = (BYTE *)conf_old + conf_block_pointer_old5[i];
        memcpy(dst, filedat, std::min<int>(((CONF_GUIEX_OLD_V5 *)_conf)->block_size[i], conf_block_data_old5[i]));
    }

    CONF_GUIEX *conf = (CONF_GUIEX *)calloc(sizeof(CONF_GUIEX), 1);
    write_conf_header(conf);

    //まずそのままコピーするブロックはそうする
#define COPY_BLOCK(block, block_idx) { memcpy(&conf->block, ((BYTE *)conf_old) + conf_old->block_head_p[block_idx], std::min<int>(sizeof(conf->block), conf_old->block_size[block_idx])); }
    COPY_BLOCK(aud, 2);
    COPY_BLOCK(mux, 3);
    COPY_BLOCK(oth, 4);
#undef COPY_BLOCK

    conf->enc.codec_rgy      = conv_codec_mfx_to_rgy(conf_old->qsv.CodecId);
    conf->vid.auo_tcfile_out = conf_old->vid.auo_tcfile_out;
    conf->vid.afs            = conf_old->vid.afs;

    if (!conf_old->qsv.vpp.bEnable) {
        conf_old->qsv.vpp.bUseDenoise = false;
        conf_old->qsv.vpp.bUseDetailEnhance = false;
        conf_old->qsv.vpp.bUseProAmp = false;
        conf_old->qsv.vpp.bUseResize = false;
        conf_old->qsv.vpp.bHalfTurn = 0;
        conf_old->qsv.vpp.nDeinterlace = 0;
        conf_old->qsv.vpp.nMirrorType = MFX_MIRRORING_DISABLED;
        conf_old->qsv.vpp.nRotate = 0;
    }
    conf_old->qsv.bCAVLC = FALSE;
    conf_old->qsv.bRDO = FALSE;
    conf_old->qsv.MVSearchWindow.x = 0;
    conf_old->qsv.MVSearchWindow.y = 0;
    conf_old->qsv.nMVPrecision = MFX_MVPRECISION_UNKNOWN;
    conf_old->qsv.nInterPred = MFX_BLOCKSIZE_UNKNOWN;
    conf_old->qsv.nIntraPred = MFX_BLOCKSIZE_UNKNOWN;
    conf_old->qsv.bDirectBiasAdjust = FALSE;
    conf_old->qsv.bNoDeblock = FALSE;
    conf_old->qsv.bIntraRefresh = FALSE;

    tstring cmd_full = gen_cmd_oldv5(&conf_old->qsv, true);

    sInputParams prm;
    parse_cmd(&prm, cmd_full.c_str(), true);
    strcpy_s(conf->enc.cmd, gen_cmd(&prm, true).c_str());

    strcpy_s(conf->conf_name, CONF_NAME_OLD_6);

    conf->vid.resize_enable = conf_old->qsv.vpp.bUseResize ? TRUE : FALSE;
    conf->vid.resize_width = conf_old->qsv.nDstWidth;
    conf->vid.resize_height = conf_old->qsv.nDstWidth;
    free(_conf);
    free(conf_old);
    return conf;
}

static void init_qsvp_prm_oldv5(sInputParamsOld *prm) {
    memset(prm, 0, sizeof(sInputParamsOld));
    prm->CodecId           = MFX_CODEC_AVC;
    prm->nTargetUsage      = QSV_DEFAULT_QUALITY;
    prm->nEncMode          = MFX_RATECONTROL_CQP;
    prm->bUseHWLib         = true;
#if defined(_WIN32) || defined(_WIN64)
    prm->memType           = HW_MEMORY;
#else
    prm->memType           = SYSTEM_MEMORY;
#endif
    prm->ColorFormat       = MFX_FOURCC_NV12;
    prm->nPicStruct        = MFX_PICSTRUCT_PROGRESSIVE;
    prm->nBitRate          = 3000;
    prm->nMaxBitrate       = 15000;
    prm->nFPSRate          = 0;
    prm->nFPSScale         = 0;
    prm->nQPI              = QSV_DEFAULT_QPI;
    prm->nQPP              = QSV_DEFAULT_QPP;
    prm->nQPB              = QSV_DEFAULT_QPB;
    prm->nICQQuality       = QSV_DEFAULT_ICQ;
    prm->nQVBRQuality      = QSV_DEFAULT_QVBR;
    prm->nAVBRAccuarcy     = QSV_DEFAULT_ACCURACY;
    prm->nAVBRConvergence  = QSV_DEFAULT_CONVERGENCE;
    prm->nIdrInterval      = 0;
    prm->nBframes          = QSV_GOP_REF_DIST_AUTO-1;
    prm->nGOPLength        = QSV_DEFAULT_GOP_LEN;
    prm->nRef              = QSV_DEFAULT_REF;
    prm->bopenGOP          = false;
    prm->bBPyramid         = false;
    prm->bforceGOPSettings = QSV_DEFAULT_FORCE_GOP_LEN;
    prm->ColorPrim         = (mfxU16)list_colorprim[0].value;
    prm->ColorMatrix       = (mfxU16)list_colormatrix[0].value;
    prm->Transfer          = (mfxU16)list_transfer[0].value;
    prm->VideoFormat       = (mfxU16)list_videoformat[0].value;
    prm->bRDO              = false;

    prm->nVQPStrength      = QSV_DEFAULT_VQP_STRENGTH;
    prm->nVQPSensitivity   = QSV_DEFAULT_VQP_SENSITIVITY;
    prm->nPerfMonitorInterval = QSV_DEFAULT_PERF_MONITOR_INTERVAL;
    prm->nOutputBufSizeMB  = QSV_DEFAULT_OUTPUT_BUF_MB;
    prm->nInputBufSize     = QSV_DEFAULT_INPUT_BUF_HW;
    prm->nOutputThread     = RGY_OUTPUT_THREAD_AUTO;
    prm->nAudioThread      = RGY_AUDIO_THREAD_AUTO;
    prm->nAudioIgnoreDecodeError = QSV_DEFAULT_AUDIO_IGNORE_DECODE_ERROR;

    prm->nDstWidth          = 1280;
    prm->nDstHeight         = 720;
    prm->vpp.nDenoise       = 20;
    prm->vpp.nDetailEnhance = 20;
    prm->vpp.delogo.nDepth  = QSV_DEFAULT_VPP_DELOGO_DEPTH;

    prm->nSessionThreadPriority = (mfxU16)get_value_from_chr(list_priority, _T("normal"));
}

#pragma warning(push)
#pragma warning(disable: 4127) //C4127: 条件式が定数です。
static tstring gen_cmd_oldv5(const sInputParamsOld *pParams, bool save_disabled_prm) {
    std::basic_stringstream<TCHAR> tmp;
    std::basic_stringstream<TCHAR> cmd;
    sInputParamsOld encPrmDefault;
    init_qsvp_prm_oldv5(&encPrmDefault);

#define OPT_FLOAT(str, opt, prec) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << (str) << _T(" ") << std::setprecision(prec) << (pParams->opt);
#define OPT_NUM(str, opt) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << (str) << _T(" ") << (int)(pParams->opt);
#define OPT_TRI(str_true, str_false, opt, val_true, val_false) \
    if ((pParams->opt) != (encPrmDefault.opt)) { \
        if ((pParams->opt) == (val_true)) { \
            cmd << _T(" ") << (str_true); \
        } else if ((pParams->opt) == (val_false)) { \
            cmd << _T(" ") << (str_false); \
        } \
    }

#define OPT_LST(str, opt, list) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << (str) << _T(" ") << get_chr_from_value(list, (pParams->opt));
#define OPT_QP(str, force, qpi, qpp, qpb) { \
    if ((force) \
    || (pParams->qpi) != (encPrmDefault.qpi) \
    || (pParams->qpp) != (encPrmDefault.qpp) \
    || (pParams->qpb) != (encPrmDefault.qpb)) { \
        if ((pParams->qpi) == (pParams->qpp) && (pParams->qpi) == (pParams->qpb)) { \
            cmd << _T(" ") << (str) << _T(" ") << (int)(pParams->qpi); \
        } else { \
            cmd << _T(" ") << (str) << _T(" ") << (int)(pParams->qpi) << _T(":") << (int)(pParams->qpp) << _T(":") << (int)(pParams->qpb); \
        } \
    } \
}
#define OPT_BOOL(str_true, str_false, opt) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << ((pParams->opt) ? (str_true) : (str_false));
#define OPT_BOOL_VAL(str_true, str_false, opt, val) { \
    if ((pParams->opt) != (encPrmDefault.opt) || (save_disabled_prm && (pParams->val) != (encPrmDefault.val))) { \
        cmd << _T(" ") << ((pParams->opt) ? (str_true) : (str_false)) <<  _T(" ") << (pParams->val); \
    } \
}
#define OPT_CHAR(str, opt) if ((pParams->opt) && (pParams->opt[0] != 0)) cmd << _T(" ") << str << _T(" ") << (pParams->opt);
#define OPT_STR(str, opt) if (pParams->opt.length() > 0) cmd << _T(" ") << str << _T(" ") << (pParams->opt.c_str());
#define OPT_CHAR_PATH(str, opt) if ((pParams->opt) && (pParams->opt[0] != 0)) cmd << _T(" ") << str << _T(" \"") << (pParams->opt) << _T("\"");
#define OPT_STR_PATH(str, opt) if (pParams->opt.length() > 0) cmd << _T(" ") << str << _T(" \"") << (pParams->opt.c_str()) << _T("\"");

    cmd << _T(" -c ") << get_chr_from_value(list_codec_mfx, pParams->CodecId);
    OPT_CHAR_PATH(_T("-i"), strSrcFile);
    OPT_CHAR_PATH(_T("-o"), strDstFile);
    switch (pParams->nInputFmt) {
    case RGY_INPUT_FMT_RAW:    cmd << _T(" --raw"); break;
    case RGY_INPUT_FMT_Y4M:    cmd << _T(" --y4m"); break;
    case RGY_INPUT_FMT_AVI:    cmd << _T(" --avi"); break;
    case RGY_INPUT_FMT_AVS:    cmd << _T(" --avs"); break;
    case RGY_INPUT_FMT_VPY:    cmd << _T(" --vpy"); break;
    case RGY_INPUT_FMT_VPY_MT: cmd << _T(" --vpy-mt"); break;
    case RGY_INPUT_FMT_AVHW:   cmd << _T(" --avhw"); break;
    case RGY_INPUT_FMT_AVSW:   cmd << _T(" --avsw"); break;
    default: break;
    }
    if (save_disabled_prm || pParams->nPicStruct != RGY_PICSTRUCT_FRAME) {
        OPT_LST(_T("--interlace"), nPicStruct, list_interlaced);
    }
    if (cropEnabled(pParams->sInCrop)) {
        cmd << _T(" --crop ") << pParams->sInCrop.e.left << _T(",") << pParams->sInCrop.e.up
            << _T(",") << pParams->sInCrop.e.right << _T(",") << pParams->sInCrop.e.bottom;
    }
    if (pParams->nFPSRate * pParams->nFPSScale > 0) {
        cmd << _T(" --fps ") << pParams->nFPSRate << _T("/") << pParams->nFPSScale;
    }
    if (pParams->nWidth * pParams->nHeight > 0) {
        cmd << _T(" --input-res ") << pParams->nWidth << _T("x") << pParams->nHeight;
    }
    if (pParams->nDstWidth * pParams->nDstHeight > 0) {
        cmd << _T(" --output-res ") << pParams->nDstWidth << _T("x") << pParams->nDstHeight;
    }
    OPT_LST(_T("--quality"), nTargetUsage, list_quality_for_option);
    OPT_BOOL(_T("--fixed-func"), _T("--no-fixed-func"), bUseFixedFunc);
    OPT_NUM(_T("--async-depth"), nAsyncDepth);
    if (save_disabled_prm || ((pParams->memType) != (encPrmDefault.memType))) {
        switch (pParams->memType) {
#if D3D_SURFACES_SUPPORT
        case SYSTEM_MEMORY: cmd << _T(" --disable-d3d"); break;
        case HW_MEMORY:   cmd << _T(" --d3d"); break;
        case D3D9_MEMORY: cmd << _T(" --d3d9"); break;
#if MFX_D3D11_SUPPORT
        case D3D11_MEMORY: cmd << _T(" --d3d11"); break;
#endif
#endif
#if LIBVA_SUPPORT
        case SYSTEM_MEMORY: cmd << _T(" --disable-va"); break;
        case D3D11_MEMORY: cmd << _T(" --va"); break;
#endif
        default: break;
        }
    }
    if (save_disabled_prm || pParams->nEncMode == MFX_RATECONTROL_QVBR) {
        OPT_NUM(_T("--qvbr-q"), nQVBRQuality);
    }
    if (save_disabled_prm) {
        switch (pParams->nEncMode) {
        case MFX_RATECONTROL_CBR:
        case MFX_RATECONTROL_VBR:
        case MFX_RATECONTROL_AVBR:
        case MFX_RATECONTROL_QVBR:
        case MFX_RATECONTROL_LA:
        case MFX_RATECONTROL_LA_HRD:
        case MFX_RATECONTROL_VCM: {
            OPT_QP(_T("--cqp"), true, nQPI, nQPP, nQPB);
            cmd << _T(" --icq ") << pParams->nICQQuality;
        } break;
        case MFX_RATECONTROL_ICQ:
        case MFX_RATECONTROL_LA_ICQ: {
            OPT_QP(_T("--cqp"), true, nQPI, nQPP, nQPB);
            cmd << _T(" --vbr ") << pParams->nBitRate;
        } break;
        case MFX_RATECONTROL_CQP:
        default: {
            cmd << _T(" --icq ") << pParams->nICQQuality;
            cmd << _T(" --vbr ") << pParams->nBitRate;
        } break;
        }
    }
    switch (pParams->nEncMode) {
    case MFX_RATECONTROL_CBR: {
        cmd << _T(" --cbr ") << pParams->nBitRate;
    } break;
    case MFX_RATECONTROL_VBR: {
        cmd << _T(" --vbr ") << pParams->nBitRate;
    } break;
    case MFX_RATECONTROL_AVBR: {
        cmd << _T(" --avbr ") << pParams->nBitRate;
    } break;
    case MFX_RATECONTROL_QVBR: {
        cmd << _T(" --qvbr ") << pParams->nBitRate;
    } break;
    case MFX_RATECONTROL_LA: {
        cmd << _T(" --la ") << pParams->nBitRate;
    } break;
    case MFX_RATECONTROL_LA_HRD: {
        cmd << _T(" --la-hrd ") << pParams->nBitRate;
    } break;
    case MFX_RATECONTROL_VCM: {
        cmd << _T(" --vcm ") << pParams->nBitRate;
    } break;
    case MFX_RATECONTROL_ICQ: {
        cmd << _T(" --icq ") << pParams->nICQQuality;
    } break;
    case MFX_RATECONTROL_LA_ICQ: {
        cmd << _T(" --la-icq ") << pParams->nICQQuality;
    } break;
    case MFX_RATECONTROL_CQP:
    default: {
        OPT_QP(_T("--cqp"), true, nQPI, nQPP, nQPB);
    } break;
    }
    OPT_NUM(_T("--la-depth"), nLookaheadDepth);
    OPT_NUM(_T("--la-window-size"), nWinBRCSize);
    OPT_LST(_T("--la-quality"), nLookaheadDS, list_lookahead_ds);
    OPT_NUM(_T("--avbr-unitsize"), nAVBRConvergence);
    OPT_BOOL(_T("--fallback-rc"), _T(""), nFallback);
    OPT_NUM(_T("--max-bitrate"), nMaxBitrate);
    OPT_NUM(_T("--la-depth"), nLookaheadDepth);
    OPT_QP(_T("--qp-min"), true, nQPMin[0], nQPMin[1], nQPMin[2]);
    OPT_QP(_T("--qp-max"), true, nQPMax[0], nQPMax[1], nQPMax[2]);
    if (memcmp(pParams->pQPOffset, encPrmDefault.pQPOffset, sizeof(encPrmDefault.pQPOffset))) {
        tmp.str(tstring());
        bool exit_loop = false;
        for (int i = 0; i < _countof(pParams->pQPOffset) && !exit_loop; i++) {
            tmp << pParams->pQPOffset[i] << _T(":");
            exit_loop = true;
            for (int j = i+1; j < _countof(pParams->pQPOffset); j++) {
                if (pParams->pQPOffset[i] != pParams->pQPOffset[j]) {
                    exit_loop = false;
                    break;
                }
            }
        }
        cmd << _T(" --qp-offset ") << tmp.str().substr(1);
    }

    OPT_NUM(_T("--slices"), nSlices);
    OPT_NUM(_T("--ref"), nRef);
    OPT_NUM(_T("-b"), nBframes);
    OPT_BOOL(_T("--b-pyramid"), _T("--no-b-pyramid"), bBPyramid);
    OPT_BOOL(_T("--open-gop"), _T("--no-open-gop"), bopenGOP);
    OPT_BOOL(_T("--strict-gop"), _T(""), bforceGOPSettings);
    OPT_BOOL(_T("--i-adapt"), _T("--no-i-adapt"), bAdaptiveI);
    OPT_BOOL(_T("--b-adapt"), _T("--no-b-adapt"), bAdaptiveB);
    OPT_TRI(_T("--weightb"), _T("--no-weightb"), nWeightB, MFX_WEIGHTED_PRED_DEFAULT, MFX_WEIGHTED_PRED_UNKNOWN);
    OPT_TRI(_T("--weightp"), _T("--no-weightp"), nWeightB, MFX_WEIGHTED_PRED_DEFAULT, MFX_WEIGHTED_PRED_UNKNOWN);
    OPT_TRI(_T("--repartition-check"), _T("--no-repartition-check"), nRepartitionCheck, MFX_CODINGOPTION_ON, MFX_CODINGOPTION_OFF);
    OPT_TRI(_T("--fade-detect"), _T("--no-fade-detect"), nFadeDetect, MFX_CODINGOPTION_ON, MFX_CODINGOPTION_OFF);
    if (pParams->nGOPLength == 0) {
        cmd << _T(" --gop-len auto");
    } else {
        OPT_NUM(_T("--gop-len"), nGOPLength);
    }
    OPT_BOOL(_T("--weightp"), _T(""), nWeightP);
    OPT_LST(_T("--mv-precision"), nMVPrecision, list_mv_presicion);
    OPT_NUM(_T("--mv-search"), MVSearchWindow.x);
    if (pParams->bGlobalMotionAdjust) {
        cmd << _T(" --mv-scaling ") << pParams->nMVCostScaling;
    }
    if (pParams->nPAR[0] > 0 && pParams->nPAR[1] > 0) {
        cmd << _T(" --sar ") << pParams->nPAR[0] << _T(":") << pParams->nPAR[1];
    } else if (pParams->nPAR[0] < 0 && pParams->nPAR[1] < 0) {
        cmd << _T(" --dar ") << -1 * pParams->nPAR[0] << _T(":") << -1 * pParams->nPAR[1];
    }

    OPT_BOOL(_T("--extbrc"), _T("--no-extbrc"), bExtBRC);
    OPT_BOOL(_T("--mbbrc"), _T("--no-mbbrc"), bMBBRC);
    OPT_BOOL(_T("--intra-refresh"), _T("--no-intra-refresh"), bIntraRefresh);
    OPT_BOOL(_T("--direct-bias-adjust"), _T("--no-direct-bias-adjust"), bDirectBiasAdjust);
    OPT_LST(_T("--intra-pred"), nIntraPred, list_pred_block_size);
    OPT_LST(_T("--inter-pred"), nInterPred, list_pred_block_size);
    OPT_BOOL(_T("--aud"), _T(""), bOutputAud);
    OPT_BOOL(_T("--pic-struct"), _T(""), bOutputPicStruct);
    OPT_BOOL(_T("--fullrange"), _T(""), bFullrange);
    OPT_LST(_T("--videoformat"), VideoFormat, list_videoformat);
    OPT_LST(_T("--colormatrix"), ColorMatrix, list_colormatrix);
    OPT_LST(_T("--colorprim"), ColorPrim, list_colorprim);
    OPT_LST(_T("--transfer"), Transfer, list_transfer);
    OPT_LST(_T("--level"), CodecLevel, get_level_list(conv_codec_mfx_to_rgy(pParams->CodecId)));
    OPT_LST(_T("--profile"), CodecProfile, get_profile_list(conv_codec_mfx_to_rgy(pParams->CodecId)));
    if (save_disabled_prm || pParams->CodecId == MFX_CODEC_HEVC) {
        OPT_CHAR(_T("--max-cll"), sMaxCll);
        OPT_CHAR(_T("--master-display"), sMasterDisplay);
    }
    if (save_disabled_prm || pParams->CodecId == MFX_CODEC_AVC) {
        OPT_LST(_T("--trellis"), nTrellis, list_avc_trellis_for_options);
        switch (pParams->nBluray) {
        case 1: cmd << _T(" --bluray"); break;
        case 2: cmd << _T(" --force-bluray"); break;
        case 0:
        default: break;
        }
        OPT_BOOL(_T("--rdo"), _T(""), bRDO);
        OPT_BOOL(_T("--cavlc"), _T(""), bCAVLC);
        OPT_BOOL(_T("--no-deblock"), _T(""), bNoDeblock);
    }
    if (save_disabled_prm || pParams->CodecId == MFX_CODEC_VP8) {
        OPT_NUM(_T("--sharpness"), nVP8Sharpness);
    }
#if ENABLE_SESSION_THREAD_CONFIG
    OPT_NUM(_T("--session-threads"), nSessionThreads);
    OPT_LST(_T("--session-thread-priority"), nSessionThreadPriority, list_priority);
#endif //#if ENABLE_SESSION_THREAD_CONFIG

#if ENABLE_AVSW_READER
    OPT_NUM(_T("--input-analyze"), nAVDemuxAnalyzeSec);
    if (pParams->nTrimCount > 0) {
        cmd << _T(" --trim ");
        for (int i = 0; i < pParams->nTrimCount; i++) {
            if (i > 0) cmd << _T(",");
            cmd << pParams->pTrimList[i].start << _T(":") << pParams->pTrimList[i].fin;
        }
    }
    OPT_FLOAT(_T("--seek"), fSeekSec, 2);
    OPT_CHAR(_T("--input-format"), pAVInputFormat);
    OPT_CHAR(_T("--output-format"), pAVMuxOutputFormat);
    OPT_NUM(_T("--video-track"), nVideoTrack);
    OPT_NUM(_T("--video-streamid"), nVideoStreamId);
    if (pParams->pMuxOpt) {
        for (uint32_t i = 0; i < pParams->pMuxOpt->size(); i++) {
            cmd << _T(" -m ") << pParams->pMuxOpt->at(i).first << _T(":") << pParams->pMuxOpt->at(i).second;
        }
    }
    tmp.str(tstring());
    for (uint32_t i = 0; i < pParams->nAudioSelectCount; i++) {
        const sAudioSelectOld *pAudioSelect = pParams->ppAudioSelectList[i];
        if (_tcscmp(pAudioSelect->pAVAudioEncodeCodec, RGY_AVCODEC_COPY) == 0) {
            tmp << pAudioSelect->nAudioSelect << _T(",");
        }
    }
    if (tmp.str().empty()) {
        cmd << _T(" --audio-copy");
    } else {
        cmd << _T(" --audio-copy ") << tmp.str().substr(1);
    }
    tmp.str(tstring());

    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const sAudioSelectOld *pAudioSelect = pParams->ppAudioSelectList[i];
        if (_tcscmp(pAudioSelect->pAVAudioEncodeCodec, RGY_AVCODEC_COPY) != 0) {
            cmd << _T(" --audio-codec ") << pAudioSelect->nAudioSelect;
            if (_tcscmp(pAudioSelect->pAVAudioEncodeCodec, RGY_AVCODEC_AUTO) != 0) {
                cmd << _T("?") << pAudioSelect->pAVAudioEncodeCodec;
            }
        }
    }

    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const sAudioSelectOld *pAudioSelect = pParams->ppAudioSelectList[i];
        if (_tcscmp(pAudioSelect->pAVAudioEncodeCodec, RGY_AVCODEC_COPY) != 0) {
            cmd << _T(" --audio-bitrate ") << pAudioSelect->nAudioSelect << _T("?") << pAudioSelect->nAVAudioEncodeBitrate;
        }
    }

    //QSVEnc.auoでは、libavutilの関数 av_get_channel_layout_string()を実行してはならない
    //for (int i = 0; i < pParams->nAudioSelectCount; i++) {
    //    const sAudioSelectOld *pAudioSelect = pParams->ppAudioSelectList[i];
    //    cmd << _T(" --audio-stream ") << pAudioSelect->nAudioSelect;
    //    for (int j = 0; j < MAX_SPLIT_CHANNELS; j++) {
    //        if (pAudioSelect->pnStreamChannelSelect[j] == 0) {
    //            break;
    //        }
    //        if (j > 0) cmd << _T(",");
    //        if (pAudioSelect->pnStreamChannelSelect[j] != RGY_CHANNEL_AUTO) {
    //            char buf[256];
    //            av_get_channel_layout_string(buf, _countof(buf), 0, pAudioSelect->pnStreamChannelOut[j]);
    //            cmd << char_to_tstring(buf);
    //        }
    //        if (pAudioSelect->pnStreamChannelOut[j] != RGY_CHANNEL_AUTO) {
    //            cmd << _T(":");
    //            char buf[256];
    //            av_get_channel_layout_string(buf, _countof(buf), 0, pAudioSelect->pnStreamChannelOut[j]);
    //            cmd << char_to_tstring(buf);
    //        }
    //    }
    //}

    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const sAudioSelectOld *pAudioSelect = pParams->ppAudioSelectList[i];
        if (_tcscmp(pAudioSelect->pAVAudioEncodeCodec, RGY_AVCODEC_COPY) != 0) {
            cmd << _T(" --audio-samplerate ") << pAudioSelect->nAudioSelect << _T("?") << pAudioSelect->nAudioSamplingRate;
        }
    }
    OPT_LST(_T("--audio-resampler"), nAudioResampler, list_resampler);

    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const sAudioSelectOld *pAudioSelect = pParams->ppAudioSelectList[i];
        if (_tcscmp(pAudioSelect->pAVAudioEncodeCodec, RGY_AVCODEC_COPY) != 0) {
            cmd << _T(" --audio-filter ") << pAudioSelect->nAudioSelect << _T("?") << pAudioSelect->pAudioFilter;
        }
    }
    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const sAudioSelectOld *pAudioSelect = pParams->ppAudioSelectList[i];
        if (pAudioSelect->pAudioExtractFilename) {
            cmd << _T(" --audio-file ") << pAudioSelect->nAudioSelect << _T("?");
            if (pAudioSelect->pAudioExtractFormat) {
                cmd << pAudioSelect->pAudioExtractFormat << _T(":");
            }
            cmd << _T("\"") << pAudioSelect->pAudioExtractFilename << _T("\"");
        }
    }
    for (int i = 0; i < pParams->nAudioSourceCount; i++) {
        cmd << _T(" --audio-source ") << _T("\"") << pParams->ppAudioSourceList[i] << _T("\"");
    }
    OPT_NUM(_T("--audio-ignore-decode-error"), nAudioIgnoreDecodeError);
    if (pParams->pMuxOpt) {
        for (uint32_t i = 0; i < pParams->pMuxOpt->size(); i++) {
            cmd << _T(" -m ") << (*pParams->pMuxOpt)[i].first << _T(":") << (*pParams->pMuxOpt)[i].second;
        }
    }

    tmp.str(tstring());
    for (int i = 0; i < pParams->nSubtitleSelectCount; i++) {
        tmp << pParams->pSubtitleSelect[i] << _T(",");
    }
    if (!tmp.str().empty()) {
        cmd << _T(" --sub-copy ") << tmp.str().substr(1);
    }
    tmp.str(tstring());
    OPT_CHAR_PATH(_T("--chapter"), pChapterFile);
    OPT_BOOL(_T("--chapter-copy"), _T(""), bCopyChapter);
    OPT_BOOL(_T("--chapter-no-trim"), _T(""), bChapterNoTrim);
    OPT_LST(_T("--avsync"), nAVSyncMode, list_avsync);
#endif //#if ENABLE_AVSW_READER

    OPT_LST(_T("--vpp-deinterlace"), vpp.nDeinterlace, list_deinterlace);
    OPT_BOOL_VAL(_T("--vpp-detail-enhance"), _T("--no-vpp-detail-enhance"), vpp.bUseDetailEnhance, vpp.nDetailEnhance);
    OPT_BOOL_VAL(_T("--vpp-denoise"), _T("--no-vpp-denoise"), vpp.bUseDenoise, vpp.nDenoise);
    OPT_BOOL(_T("--vpp-half-turn"), _T(""), vpp.bHalfTurn);
    OPT_LST(_T("--vpp-rotate"), vpp.nRotate, list_vpp_rotate);
    OPT_LST(_T("--vpp-mirror"), vpp.nMirrorType, list_vpp_mirroring);
    OPT_LST(_T("--vpp-scaling"), vpp.nScalingQuality, list_vpp_scaling_quality_auo_conf_old);
    OPT_LST(_T("--vpp-fps-conv"), vpp.nFPSConversion, list_vpp_fps_conversion);
    OPT_LST(_T("--vpp-image-stab"), vpp.nImageStabilizer, list_vpp_image_stabilizer);
#if ENABLE_CUSTOM_VPP
#if ENABLE_AVSW_READER && ENABLE_LIBASS_SUBBURN
    OPT_CHAR_PATH(_T("--vpp-sub"), vpp.subburn.pFilePath);
    OPT_CHAR_PATH(_T("--vpp-sub-charset"), vpp.subburn.pCharEnc);
    OPT_LST(_T("--vpp-sub-shaping"), vpp.subburn.nShaping, list_vpp_sub_shaping);
#endif //#if ENABLE_AVSW_READER && ENABLE_LIBASS_SUBBURN
    OPT_CHAR_PATH(_T("--vpp-delogo"), vpp.delogo.pFilePath);
    OPT_CHAR(_T("--vpp-delogo-select"), vpp.delogo.pSelect);
    OPT_NUM(_T("--vpp-delogo-depth"), vpp.delogo.nDepth);
    if (pParams->vpp.delogo.nPosOffset.x > 0 || pParams->vpp.delogo.nPosOffset.y > 0) {
        cmd << _T(" --vpp-delogo-pos ") << pParams->vpp.delogo.nPosOffset.x << _T("x") << pParams->vpp.delogo.nPosOffset.y;
    }
    OPT_NUM(_T("--vpp-delogo-y"), vpp.delogo.nYOffset);
    OPT_NUM(_T("--vpp-delogo-cb"), vpp.delogo.nCbOffset);
    OPT_NUM(_T("--vpp-delogo-cr"), vpp.delogo.nCrOffset);
#endif //#if ENABLE_CUSTOM_VPP
#if defined(_WIN32) || defined(_WIN64)
    OPT_NUM(_T("--mfx-thread"), nSessionThreads);
#endif //#if defined(_WIN32) || defined(_WIN64)
    OPT_NUM(_T("--input-buf"), nInputBufSize);
    OPT_NUM(_T("--output-buf"), nOutputBufSizeMB);
    OPT_NUM(_T("--output-thread"), nOutputThread);
    OPT_NUM(_T("--input-thread"), nInputThread);
    OPT_NUM(_T("--audio-thread"), nAudioThread);
    OPT_NUM(_T("--max-procfps"), nProcSpeedLimit);
    OPT_CHAR_PATH(_T("--log"), pStrLogFile);
    OPT_LST(_T("--log-level"), nLogLevel, auo_list_log_level);
    OPT_CHAR_PATH(_T("--log-framelist"), pFramePosListLog);
    OPT_CHAR_PATH(_T("--log-mux-ts"), pMuxVidTsLogFile);
    OPT_CHAR_PATH(_T("--log-copy-framedata"), pLogCopyFrameData);
    if (pParams->nPerfMonitorSelect != encPrmDefault.nPerfMonitorSelect) {
        auto select = (int)pParams->nPerfMonitorSelect;
        tmp.str(tstring());
        for (int i = 0; list_pref_monitor[i].desc; i++) {
            auto check = list_pref_monitor[i].value;
            if ((select & check) == check) {
                tmp << _T(",") << list_pref_monitor[i].desc;
                select &= (~check);
            }
        }
        if (tmp.str().empty()) {
            cmd << _T(" --perf-monitor");
        } else {
            cmd << _T(" --perf-monitor ") << tmp.str().substr(1);
        }
    }
    if (pParams->nPerfMonitorSelectMatplot != encPrmDefault.nPerfMonitorSelectMatplot) {
        auto select = (int)pParams->nPerfMonitorSelectMatplot;
        tmp.str(tstring());
        for (int i = 0; list_pref_monitor[i].desc; i++) {
            auto check = list_pref_monitor[i].value;
            if ((select & check) == check) {
                tmp << _T(",") << list_pref_monitor[i].desc;
                select &= (~check);
            }
        }
        if (tmp.str().empty()) {
            cmd << _T(" --perf-monitor-plot");
        } else {
            cmd << _T(" --perf-monitor-plot ") << tmp.str().substr(1);
        }
    }
    OPT_NUM(_T("--perf-monitor-interval"), nPerfMonitorInterval);
    OPT_CHAR_PATH(_T("--python"), pLogCopyFrameData);
    OPT_BOOL(_T("--timer-period-tuning"), _T("--no-timer-period-tuning"), bDisableTimerPeriodTuning);
    return cmd.str();
}
#pragma warning(pop)
