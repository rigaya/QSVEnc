#ifndef _QSV_UTIL_H_
#define _QSV_UTIL_H_

#include <Windows.h>
#include <tchar.h>
#include <shlwapi.h>
#include <emmintrin.h>
#pragma comment(lib, "shlwapi.lib")
#include <vector>
#include <string>
#include "vm/strings_defs.h"
#include "mfxstructures.h"
#include "mfxSession.h"

#ifndef MIN3
#define MIN3(a,b,c) (min((a), min((b), (c))))
#endif
#ifndef MAX3
#define MAX3(a,b,c) (max((a), max((b), (c))))
#endif

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

mfxU32 GCD(mfxU32 a, mfxU32 b);
mfxI64 GCDI64(mfxI64 a, mfxI64 b);

typedef struct CX_DESC {
	TCHAR *desc;
	int value;
} CX_DESC;

static const mfxVersion LIB_VER_LIST[] = {
	{ 0, 0 },
	{ 0, 1 },
	{ 1, 1 },
	{ 3, 1 },
	{ 4, 1 },
	{ 6, 1 },
	{ 7, 1 },
	{ 8, 1 },
	{ NULL, NULL } 
};

#define MFX_LIB_VERSION_0_0 LIB_VER_LIST[0]
#define MFX_LIB_VERSION_1_1 LIB_VER_LIST[2]
#define MFX_LIB_VERSION_1_3 LIB_VER_LIST[3]
#define MFX_LIB_VERSION_1_4 LIB_VER_LIST[4]
#define MFX_LIB_VERSION_1_6 LIB_VER_LIST[5]
#define MFX_LIB_VERSION_1_7 LIB_VER_LIST[6]
#define MFX_LIB_VERSION_1_8 LIB_VER_LIST[7]

BOOL Check_HWUsed(mfxIMPL impl);
mfxVersion get_mfx_libhw_version();
mfxVersion get_mfx_libsw_version();
mfxVersion get_mfx_lib_version(mfxIMPL impl);
BOOL check_lib_version(mfxVersion value, mfxVersion required);
BOOL check_lib_version(mfxU32 _value, mfxU32 _required);

enum {
	ENC_FEATURE_CURRENT_RC = 0x00000001,
	ENC_FEATURE_AVBR       = 0x00000002,
	ENC_FEATURE_LA         = 0x00000004,
	ENC_FEATURE_ICQ        = 0x00000008,
	ENC_FEATURE_LA_ICQ     = 0x00000010,
	ENC_FEATURE_VCM        = 0x00000020,
	ENC_FEATURE_AUD        = 0x00000040,
	ENC_FEATURE_PIC_STRUCT = 0x00000080,
	ENC_FEATURE_VUI_INFO   = 0x00000100,
	ENC_FEATURE_CAVLC      = 0x00000200,
	ENC_FEATURE_RDO        = 0x00000400,
	ENC_FEATURE_ADAPTIVE_I = 0x00000800,
	ENC_FEATURE_ADAPTIVE_B = 0x00001000,
	ENC_FEATURE_B_PYRAMID  = 0x00002000,
	ENC_FEATURE_TRELLIS    = 0x00004000,
	ENC_FEATURE_EXT_BRC    = 0x00008000,
	ENC_FEATURE_MBBRC      = 0x00010000,
	ENC_FEATURE_LA_DS      = 0x00020000,
};

static const CX_DESC list_rate_control_ry[] = {
	{ _T("CBR  "), MFX_RATECONTROL_CBR    },
	{ _T("VBR  "), MFX_RATECONTROL_VBR    },
	{ _T("AVBR "), MFX_RATECONTROL_AVBR   },
	{ _T("CQP  "), MFX_RATECONTROL_CQP    },
	{ _T("VQP  "), MFX_RATECONTROL_VQP    },
	{ _T("LA   "), MFX_RATECONTROL_LA     },
	{ _T("ICQ  "), MFX_RATECONTROL_ICQ    },
	{ _T("LAICQ"), MFX_RATECONTROL_LA_ICQ },
	{ _T("VCM  "), MFX_RATECONTROL_VCM    },
};
static const CX_DESC list_enc_feature[] = {
	{ _T("RC mode available "), ENC_FEATURE_CURRENT_RC },
	{ _T("VUI info output   "), ENC_FEATURE_VUI_INFO   },
	//{ _T("aud               "), ENC_FEATURE_AUD        },
	//{ _T("pic_struct        "), ENC_FEATURE_PIC_STRUCT },
	{ _T("Trellis           "), ENC_FEATURE_TRELLIS    },
	//{ _T("rdo               "), ENC_FEATURE_RDO        },
	//{ _T("CAVLC             "), ENC_FEATURE_CAVLC      },
	{ _T("Adaptive_I        "), ENC_FEATURE_ADAPTIVE_I },
	{ _T("Adaptive_B        "), ENC_FEATURE_ADAPTIVE_B },
	{ _T("B_Pyramid         "), ENC_FEATURE_B_PYRAMID  },
	{ _T("Ext_BRC           "), ENC_FEATURE_EXT_BRC    },
	{ _T("MBBRC             "), ENC_FEATURE_MBBRC      },
	{ _T("Lookahead Quality "), ENC_FEATURE_LA_DS      },
	{ NULL, 0 },
};

mfxU32 CheckEncodeFeature(mfxSession session, mfxU16 ratecontrol = MFX_RATECONTROL_VBR);
mfxU32 CheckEncodeFeature(bool hardware, mfxU16 ratecontrol, mfxVersion ver);
mfxU32 CheckEncodeFeature(bool hardware, mfxU16 ratecontrol = MFX_RATECONTROL_VBR);
void MakeFeatureList(bool hardware, mfxVersion ver, const CX_DESC *rateControlList, int rateControlCount, std::vector<mfxU32>& availableFeatureForEachRC);
void MakeFeatureList(bool hardware, const CX_DESC *rateControlList, int rateControlCount, std::vector<mfxU32>& availableFeatureForEachRC);
void MakeFeatureListStr(bool hardware, std::basic_string<msdk_char>& str);

bool check_if_d3d11_necessary();

int getCPUInfo(TCHAR *buffer, size_t nSize);

void adjust_sar(int *sar_w, int *sar_h, int width, int height);

//拡張子が一致するか確認する
static BOOL _tcheck_ext(const TCHAR *filename, const TCHAR *ext) {
	return (_tcsicmp(PathFindExtension(filename), ext) == NULL) ? TRUE : FALSE;
}

BOOL check_OS_Win8orLater();
bool isHaswellOrLater();

mfxStatus ParseY4MHeader(char *buf, mfxFrameInfo *info);

const TCHAR *get_err_mes(int sts);

static void __forceinline sse_memcpy(BYTE *dst, const BYTE *src, int size) {
	BYTE *dst_fin = dst + size;
	BYTE *dst_aligned_fin = (BYTE *)(((size_t)dst_fin & ~15) - 64);
	__m128 x0, x1, x2, x3;
	const int start_align_diff = (int)((size_t)dst & 15);
	if (start_align_diff) {
		x0 = _mm_loadu_ps((float*)src);
		_mm_storeu_ps((float*)dst, x0);
		dst += start_align_diff;
		src += start_align_diff;
	}
	for ( ; dst < dst_aligned_fin; dst += 64, src += 64) {
		x0 = _mm_loadu_ps((float*)(src +  0));
		x1 = _mm_loadu_ps((float*)(src + 16));
		x2 = _mm_loadu_ps((float*)(src + 32));
		x3 = _mm_loadu_ps((float*)(src + 48));
		_mm_store_ps((float*)(dst +  0), x0);
		_mm_store_ps((float*)(dst + 16), x1);
		_mm_store_ps((float*)(dst + 32), x2);
		_mm_store_ps((float*)(dst + 48), x3);
	}
	BYTE *dst_tmp = dst_fin - 64;
	src -= (dst - dst_tmp);
	x0 = _mm_loadu_ps((float*)(src +  0));
	x1 = _mm_loadu_ps((float*)(src + 16));
	x2 = _mm_loadu_ps((float*)(src + 32));
	x3 = _mm_loadu_ps((float*)(src + 48));
	_mm_storeu_ps((float*)(dst_tmp +  0), x0);
	_mm_storeu_ps((float*)(dst_tmp + 16), x1);
	_mm_storeu_ps((float*)(dst_tmp + 32), x2);
	_mm_storeu_ps((float*)(dst_tmp + 48), x3);
}

const int MAX_FILENAME_LEN = 1024;

#endif //_QSV_UTIL_H_
