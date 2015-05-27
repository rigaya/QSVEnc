//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _QSV_PRM_H_
#define _QSV_PRM_H_

#include <Windows.h>

#include "sample_defs.h"
#include "qsv_version.h"
#include "qsv_util.h"

typedef enum {
	MFX_DEINTERLACE_NONE        = 0,
	MFX_DEINTERLACE_NORMAL      = 1,
	MFX_DEINTERLACE_IT          = 2, //inverse telecine, to 24fps
	MFX_DEINTERLACE_BOB         = 3,
	MFX_DEINTERLACE_IT_MANUAL   = 4, //inverse telecine, manual select
	MFX_DEINTERLACE_AUTO_SINGLE = 5,
	MFX_DEINTERLACE_AUTO_DOUBLE = 6,
} mfxDeinterlace;

typedef enum {
	MVC_DISABLED          = 0x0,
	MVC_ENABLED           = 0x1,
	MVC_VIEWOUTPUT        = 0x2,    // 2 output bitstreams
};

typedef enum {
	INPUT_FMT_AUTO = 0,
	INPUT_FMT_RAW,
	INPUT_FMT_Y4M,
	INPUT_FMT_AVI,
	INPUT_FMT_AVS,
	INPUT_FMT_VPY,
	INPUT_FMT_VPY_MT,
	INPUT_FMT_AVCODEC_QSV,
};

typedef enum MemType {
	SYSTEM_MEMORY = 0x00,
	D3D9_MEMORY   = 0x01,
	D3D11_MEMORY  = 0x02,
	HW_MEMORY = D3D9_MEMORY | D3D11_MEMORY,
};

typedef struct {
	mfxU16 left, up, right, bottom;
} sInputCrop;

typedef struct {
	int start, fin;
} sTrim;

typedef struct {
	std::vector<sTrim> list;
	int offset;
} sTrimParam;

static const int TRIM_MAX = INT_MAX;

static bool inline frame_inside_range(int frame, const std::vector<sTrim>& trimList) {
	if (trimList.size() == 0)
		return true;
	if (frame < 0)
		return false;
	for (auto trim : trimList) {
		if (trim.start <= frame && frame <= trim.fin) {
			return true;
		}
	}
	return false;
}

typedef enum {
	FPS_CONVERT_NONE = 0,
	FPS_CONVERT_MUL2,
	FPS_CONVERT_MUL2_5,
};

typedef enum {
	QSVENC_MUX_NONE  = 0x00,
	QSVENC_MUX_VIDEO = 0x01,
	QSVENC_MUX_AUDIO = 0x02,
};

typedef struct {
	bool bEnable;             //use vpp

	bool bUseResize;          //use vpp resizer

	bool __unsed2;
	bool __unsed;
	mfxU16 reserved;      
	bool bUseProAmp;          //not supported
	bool bUseDenoise;         //use vpp denoise
	mfxU16 nDenoise;          // 0 - 100 Denoise Strength
	bool bUseDetailEnhance;   //use vpp detail enhancer
	mfxU16 nDetailEnhance;    // 0 - 100 
	mfxU16 nDeinterlace;      //set deinterlace mode

	mfxU16 nImageStabilizer;  //MFX_IMAGESTAB_MODE_UPSCALE, MFX_IMAGESTAB_MODE_BOXED
	mfxU16 nFPSConversion;    //FPS_CONVERT_xxxx

	mfxU16 nTelecinePattern;

	mfxU8 Reserved[124];
} sVppParams;

struct sInputParams
{
	mfxU16 nInputFmt;     // 0 - raw, 1 - y4m, 2 - avi/avs
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
	mfxU16 __nBitRate;    //old field of target bitrate used in bitrate mode
	mfxU16 __nMaxBitrate; //max bitrate
	mfxU16 __nThreads;    //max bitrate

	mfxU16 nDstWidth;     //output width 
	mfxU16 nDstHeight;    //input width

	mfxU8 memType;       //use d3d surface
	bool bUseHWLib;       //use QSV (hw encoding)

	mfxU16 nInputBufSize; //input buf size

	bool   bIsMVC; // true if Multi-View-Codec is in use
	mfxU32 numViews; // number of views for Multi-View-Codec
	

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

	sInputCrop sInCrop;

	mfxU16     nQuality; // quality parameter for JPEG encoder

	mfxU8      bMBBRC;
	mfxU8      bExtBRC;

	mfxU16     nLookaheadDepth;
	mfxU16     nTrellis;

	TCHAR     *pStrLogFile; //ログファイル名へのポインタ
#ifdef _M_IX86
	mfxU32     reserved;
#endif

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
	mfxU8      __unused;
	mfxU32     nBitRate;
	mfxU32     nMaxBitrate;

	mfxU16     nTrimCount;
	sTrim     *pTrimList;
#ifdef _M_IX86
	mfxU32     reserved2;
#endif
	TCHAR     *pAudioFilename;
#ifdef _M_IX86
	mfxU32     reserved3;
#endif
	mfxU16     inputBitDepthLuma;
	mfxU16     inputBitDepthChroma;
	mfxU8      nAVMux; //QSVENC_MUX_xxx
	mfxU16     nAVDemuxAnalyzeSec;

	mfxU8      Reserved[1157];

	TCHAR strSrcFile[MAX_FILENAME_LEN];
	TCHAR strDstFile[MAX_FILENAME_LEN];

	mfxU16 nRotationAngle; //not supported

	sVppParams vpp;
};

const int MFX_COLOR_VALUE_AUTO = 0x0000ffff; //max of 16bit-integer (unsigned)

const CX_DESC list_codec[] = {
	{ _T("h264"),     MFX_CODEC_AVC   },
	{ _T("mpeg2"),    MFX_CODEC_MPEG2 },
	{ _T("vc-1"),     MFX_CODEC_VC1   },
	{ NULL, NULL }
};

const CX_DESC list_avc_profile[] = {
	{ _T("auto"),     0                        },
	{ _T("Baseline"), MFX_PROFILE_AVC_BASELINE },
	{ _T("Main"),     MFX_PROFILE_AVC_MAIN     },
	{ _T("High"),     MFX_PROFILE_AVC_HIGH     },
	{ NULL, NULL }
};

const CX_DESC list_mpeg2_profile[] = {
	{ _T("auto"),     0                        },
	{ _T("Simple"),   MFX_PROFILE_MPEG2_SIMPLE },
	{ _T("Main"),     MFX_PROFILE_MPEG2_MAIN   },
	{ _T("High"),     MFX_PROFILE_MPEG2_HIGH   },
	{ NULL, NULL }
};

const CX_DESC list_vc1_profile[] = {
	{ _T("auto"),     0                        },
	{ _T("Simple"),   MFX_PROFILE_VC1_SIMPLE   },
	{ _T("Main"),     MFX_PROFILE_VC1_MAIN     },
	{ _T("Advanced"), MFX_PROFILE_VC1_ADVANCED },
	{ NULL, NULL }
};

const CX_DESC list_interlaced[] = {
	{ _T("progressive"),     MFX_PICSTRUCT_PROGRESSIVE },
	{ _T("interlaced(tff)"), MFX_PICSTRUCT_FIELD_TFF   },
	{ _T("interlaced(bff)"), MFX_PICSTRUCT_FIELD_BFF   },
	{ NULL, NULL }
};

const CX_DESC list_deinterlace[] = {
	{ _T("none"),      MFX_DEINTERLACE_NONE        },
	{ _T("normal"),    MFX_DEINTERLACE_NORMAL      },
	{ _T("it"),        MFX_DEINTERLACE_IT          },
	{ _T("bob"),       MFX_DEINTERLACE_BOB         },
#if ENABLE_ADVANCED_DEINTERLACE
	{ _T("it-manual"), MFX_DEINTERLACE_IT_MANUAL   },
	{ _T("auto"),      MFX_DEINTERLACE_AUTO_SINGLE },
	{ _T("auto-bob"),  MFX_DEINTERLACE_AUTO_DOUBLE },
#endif
	{ NULL, NULL }
};

const CX_DESC list_telecine_patterns[] = {
	{ _T("32"),     MFX_TELECINE_PATTERN_32 },
	{ _T("2332"),   MFX_TELECINE_PATTERN_2332 },
	{ _T("repeat"), MFX_TELECINE_PATTERN_FRAME_REPEAT },
	{ _T("41"),     MFX_TELECINE_PATTERN_41 },
	{ NULL, NULL }
};

const CX_DESC list_avc_level[] = { 
	{ _T("auto"), 0                 },
	{ _T("1"),    MFX_LEVEL_AVC_1   },
	{ _T("1b"),   MFX_LEVEL_AVC_1b  },
	{ _T("1.1"),  MFX_LEVEL_AVC_11  },
	{ _T("1.2"),  MFX_LEVEL_AVC_12  },
	{ _T("1.3"),  MFX_LEVEL_AVC_13  },
	{ _T("2"),    MFX_LEVEL_AVC_2   },
	{ _T("2.1"),  MFX_LEVEL_AVC_21  },
	{ _T("2.2"),  MFX_LEVEL_AVC_22  },
	{ _T("3"),    MFX_LEVEL_AVC_3   },
	{ _T("3.1"),  MFX_LEVEL_AVC_31  },
	{ _T("3.2"),  MFX_LEVEL_AVC_32  },
	{ _T("4"),    MFX_LEVEL_AVC_4   },
	{ _T("4.1"),  MFX_LEVEL_AVC_41  },
	{ _T("4.2"),  MFX_LEVEL_AVC_42  },
	{ _T("5"),    MFX_LEVEL_AVC_5   },
	{ _T("5.1"),  MFX_LEVEL_AVC_51  },
	{ NULL, NULL }
};
const CX_DESC list_mpeg2_level[] = { 
	{ _T("auto"),     0                        },
	{ _T("low"),      MFX_LEVEL_MPEG2_LOW      },
	{ _T("main"),     MFX_LEVEL_MPEG2_MAIN     },
	{ _T("high"),     MFX_LEVEL_MPEG2_HIGH     },
	{ _T("high1440"), MFX_LEVEL_MPEG2_HIGH1440 },
	{ NULL, NULL }
};
const CX_DESC list_vc1_level[] = { 
	{ _T("auto"),     0                    },
	{ _T("low"),      MFX_LEVEL_VC1_LOW    },
	{ _T("median"),   MFX_LEVEL_VC1_MEDIAN },
	{ _T("high"),     MFX_LEVEL_VC1_HIGH   },
	{ NULL, NULL }
};
const CX_DESC list_vc1_level_adv[] = { 
	{ _T("auto"),  0               },
	{ _T("0"),     MFX_LEVEL_VC1_0 },
	{ _T("1"),     MFX_LEVEL_VC1_1 },
	{ _T("2"),     MFX_LEVEL_VC1_2 },
	{ _T("3"),     MFX_LEVEL_VC1_3 },
	{ _T("4"),     MFX_LEVEL_VC1_4 },
	{ NULL, NULL }
};
const CX_DESC list_avc_trellis[] = {
	{ _T("Auto"),           MFX_TRELLIS_UNKNOWN },
	{ _T("off"),            MFX_TRELLIS_OFF },
	{ _T("for I frames"),   MFX_TRELLIS_I   },
	{ _T("for I,P frames"), MFX_TRELLIS_I | MFX_TRELLIS_P },
	{ _T("for All frames"), MFX_TRELLIS_I | MFX_TRELLIS_P | MFX_TRELLIS_B },
	{ NULL, NULL }
};
const CX_DESC list_avc_trellis_for_options[] = { 
	{ _T("auto"), MFX_TRELLIS_UNKNOWN },
	{ _T("off"),  MFX_TRELLIS_OFF },
	{ _T("i"),    MFX_TRELLIS_I   },
	{ _T("ip"),   MFX_TRELLIS_I | MFX_TRELLIS_P },
	{ _T("all"),  MFX_TRELLIS_I | MFX_TRELLIS_P | MFX_TRELLIS_B },
	{ _T("ipb"),  MFX_TRELLIS_I | MFX_TRELLIS_P | MFX_TRELLIS_B },
	{ _T("ib"),   MFX_TRELLIS_I | MFX_TRELLIS_B },
	{ _T("p"),    MFX_TRELLIS_P },
	{ _T("pb"),   MFX_TRELLIS_P | MFX_TRELLIS_B },
	{ _T("b"),    MFX_TRELLIS_B },
	{ NULL, NULL }
};

const CX_DESC list_lookahead_ds[] = {
	{ _T("auto"),   MFX_LOOKAHEAD_DS_UNKNOWN },
	{ _T("slow"),   MFX_LOOKAHEAD_DS_OFF     },
	{ _T("medium"), MFX_LOOKAHEAD_DS_2x      },
	{ _T("fast"),   MFX_LOOKAHEAD_DS_4x      },
	{ NULL, NULL }
};

const CX_DESC list_mv_cost_scaling[] = {
	{ _T("default"), -1 },
	{ _T("0"),        0 },
	{ _T("1"),        1 },
	{ _T("2"),        2 },
	{ _T("3"),        3 },
	{ _T("4"),        4 },
	{ NULL, NULL }
};

static inline const CX_DESC *get_level_list(int CodecID) {
	switch (CodecID) {
		case MFX_CODEC_MPEG2: return list_mpeg2_level;
		case MFX_CODEC_VC1:   return list_vc1_level;
		case MFX_CODEC_AVC:
		default:              return list_avc_level;
	}
}

static inline const CX_DESC *get_profile_list(int CodecID) {
	switch (CodecID) {
		case MFX_CODEC_MPEG2: return list_mpeg2_profile;
		case MFX_CODEC_VC1:   return list_vc1_profile;
		case MFX_CODEC_AVC:
		default:              return list_avc_profile;
	}
}

const int HD_HEIGHT_THRESHOLD = 720;
const int HD_INDEX = 2;
const int SD_INDEX = 3;
const CX_DESC list_colorprim[] = {
	{ _T("undef"),     2  },
	{ _T("auto"),      MFX_COLOR_VALUE_AUTO },
	{ _T("bt709"),     1  },
	{ _T("smpte170m"), 6  },
	{ _T("bt470m"),    4  },
	{ _T("bt470bg"),   5  },
	{ _T("smpte240m"), 7  },
	{ _T("film"),      8  },
	{ NULL, NULL }
};
const CX_DESC list_transfer[] = {
	{ _T("undef"),     2  },
	{ _T("auto"),      MFX_COLOR_VALUE_AUTO },
	{ _T("bt709"),     1  },
	{ _T("smpte170m"), 6  },
	{ _T("bt470m"),    4  },
	{ _T("bt470bg"),   5  },
	{ _T("smpte240m"), 7  },
	{ _T("linear"),    8  },
	{ _T("log100"),    9  },
	{ _T("log316"),    10 },
	{ NULL, NULL }
};
const CX_DESC list_colormatrix[] = {
	{ _T("undef"),     2  },
	{ _T("auto"),      MFX_COLOR_VALUE_AUTO },
	{ _T("bt709"),     1  },
	{ _T("smpte170m"), 6  },
	{ _T("bt470bg"),   5  },
	{ _T("smpte240m"), 7  },
	{ _T("YCgCo"),     8  },
	{ _T("fcc"),       4  },
	{ _T("GBR"),       0  },
	{ NULL, NULL }
};
const CX_DESC list_videoformat[] = {
	{ _T("undef"),     5  },
	{ _T("ntsc"),      2  },
	{ _T("component"), 0  },
	{ _T("pal"),       1  },
	{ _T("secam"),     3  },
	{ _T("mac"),       4  },
	{ NULL, NULL } 
};

//表示用
const CX_DESC list_quality[] = {
	{ _T(" 1 - best quality"), MFX_TARGETUSAGE_BEST_QUALITY },
	{ _T(" 2 - higher"),       2                            },
	{ _T(" 3 - high quality"), 3                            },
	{ _T(" 4 - balanced"),     MFX_TARGETUSAGE_BALANCED     },
	{ _T(" 5 - fast"),         5                            },
	{ _T(" 6 - faster"),       6                            },
	{ _T(" 7 - fastest"),      MFX_TARGETUSAGE_BEST_SPEED   },
	{ NULL, NULL }
};

//オプション用
const CX_DESC list_quality_for_option[] = {
	{ _T("best"),     MFX_TARGETUSAGE_BEST_QUALITY },
	{ _T("higher"),   2                            },
	{ _T("high"),     3                            },
	{ _T("balanced"), MFX_TARGETUSAGE_BALANCED     },
	{ _T("fast"),     5                            },
	{ _T("faster"),   6                            },
	{ _T("fastest"),  MFX_TARGETUSAGE_BEST_SPEED   },
	{ NULL, NULL }
};

const CX_DESC list_mv_presicion[] = {
	{ _T("Auto"),     MFX_MVPRECISION_UNKNOWN    },
	{ _T("full-pel"), MFX_MVPRECISION_INTEGER    },
	{ _T("half-pel"), MFX_MVPRECISION_HALFPEL    },
	{ _T("Q-pel"),    MFX_MVPRECISION_QUARTERPEL },
	{ NULL, NULL }
};

const CX_DESC list_pred_block_size[] = {
	{ _T("Auto"),          MFX_BLOCKSIZE_UNKNOWN    },
	{ _T("16x16"),         MFX_BLOCKSIZE_MIN_16X16  },
	{ _T("8x8"),           MFX_BLOCKSIZE_MIN_8X8    },
	{ _T("4x4"),           MFX_BLOCKSIZE_MIN_4X4    },
	{ NULL, NULL }
};

const CX_DESC list_vpp_image_stabilizer[] = {
	{ _T("none"),    0 },
	{ _T("upscale"), MFX_IMAGESTAB_MODE_UPSCALE },
	{ _T("box"),     MFX_IMAGESTAB_MODE_BOXING  },
	{ NULL, NULL }
};

const CX_DESC list_vpp_fps_conversion[] = {
	{ _T("off"),  0 },
	{ _T("x2"),   FPS_CONVERT_MUL2 },
	{ _T("x2.5"), FPS_CONVERT_MUL2_5  },
	{ NULL, NULL }
};

static int get_cx_index(const CX_DESC * list, int v) {
	for (int i = 0; list[i].desc; i++)
		if (list[i].value == v)
			return i;
	return -1;
}
static int get_cx_index(const CX_DESC * list, const TCHAR *chr) {
	for (int i = 0; list[i].desc; i++)
		if (0 == _tcsicmp(list[i].desc, chr))
			return i;
	return -1;
}

static int PARSE_ERROR_FLAG = INT_MIN;
static int get_value_from_chr(const CX_DESC *list, const TCHAR *chr) {
	for (int i = 0; list[i].desc; i++)
		if (0 == _tcsicmp(list[i].desc, chr))
			return list[i].value;
	return PARSE_ERROR_FLAG;
}

//define defaults
const int QSV_DEFAULT_REF = 0;
const int QSV_DEFAULT_GOP_LEN = 0;
const int QSV_DEFAULT_ICQ = 23;
const int QSV_DEFAULT_QVBR = 23;
const int QSV_DEFAULT_QPI = 24;
const int QSV_DEFAULT_QPP = 26;
const int QSV_DEFAULT_QPB = 27;
const int QSV_DEFAULT_BFRAMES = 3;
const int QSV_DEFAULT_QUALITY = MFX_TARGETUSAGE_BALANCED;
const int QSV_DEFAULT_INPUT_BUF_SW = 1;
const int QSV_DEFAULT_INPUT_BUF_HW = 3;
const int QSV_INPUT_BUF_MIN = 1;
const int QSV_INPUT_BUF_MAX = 16;
const int QSV_DEFAULT_CONVERGENCE = 90;
const int QSV_DEFAULT_ACCURACY = 500;
const int QSV_DEFAULT_FORCE_GOP_LEN = 1;

const mfxU16 QSV_DEFAULT_VQP_STRENGTH = 10;
const mfxU16 QSV_DEFAULT_VQP_SENSITIVITY = 50;
const mfxU16 QSV_DEFAULT_SC_SENSITIVITY = 80;

const int QSV_LOOKAHEAD_DEPTH_MIN = 10;
const int QSV_LOOKAHEAD_DEPTH_MAX = 100;

void init_qsvp_prm(sInputParams *prm);

#endif //_QSV_PRM_H_