#ifndef _QSV_UTIL_H_
#define _QSV_UTIL_H_

#include <Windows.h>
#include <tchar.h>
#include <shlwapi.h>
#include <emmintrin.h>
#pragma comment(lib, "shlwapi.lib")
#include <vector>
#include <string>
#include <type_traits>
#include "vm/strings_defs.h"
#include "mfxstructures.h"
#include "mfxSession.h"
#include "qsv_version.h"
#include "cpu_info.h"
#include "gpu_info.h"

using std::vector;

#ifndef MIN3
#define MIN3(a,b,c) (min((a), min((b), (c))))
#endif
#ifndef MAX3
#define MAX3(a,b,c) (max((a), max((b), (c))))
#endif

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

template<typename T, size_t size>
std::vector<T> to_vector(T(&ptr)[size]) {
    return std::vector<T>(ptr, ptr + size);
}
template<typename T, size_t size>
std::vector<const T> to_vector(const T(&ptr)[size]) {
    return std::vector<const T>(ptr, ptr + size);
}


template<typename T>
static inline T qsv_gcd(T a, T b) {
    static_assert(std::is_integral<T>::value, "qsv_gcd is defined only for integer.");
    if (a == 0) return b;
    if (b == 0) return a;
    T c;
    while ((c = a % b) != 0)
        a = b, b = c;
    return b;
}

#if UNICODE
#define to_tstring to_wstring
#else
#define to_tstring to_string
#endif

typedef std::basic_string<TCHAR> tstring;
typedef std::basic_stringstream<TCHAR> TStringStream;

unsigned int tchar_to_string(const TCHAR *tstr, std::string& str, DWORD codepage = CP_THREAD_ACP);
std::string tchar_to_string(const TCHAR *tstr, DWORD codepage = CP_THREAD_ACP);
std::string tchar_to_string(const tstring& tstr, DWORD codepage = CP_THREAD_ACP);
unsigned int char_to_tstring(tstring& tstr, const char *str, DWORD codepage = CP_THREAD_ACP);
tstring char_to_tstring(const char *str, DWORD codepage = CP_THREAD_ACP);
tstring char_to_tstring(const std::string& str, DWORD codepage = CP_THREAD_ACP);
std::string strsprintf(const char* format, ...);
std::wstring strsprintf(const WCHAR* format, ...);
std::vector<tstring> split(const tstring &str, const tstring &delim);
std::string GetFullPath(const char *path);
std::wstring GetFullPath(const WCHAR *path);

static inline mfxU16 readUB16(const void *ptr) {
    mfxU16 i = *(mfxU16 *)ptr;
    return (i >> 8) | (i << 8);
}

static inline mfxU32 readUB32(const void *ptr) {
    mfxU32 i = *(mfxU32 *)ptr;
    return (i >> 24) | ((i & 0xff0000) >> 8) | ((i & 0xff00) << 8) | ((i & 0xff) << 24);
}

static inline mfxU32 check_range_unsigned(mfxU32 value, mfxU32 min, mfxU32 max) {
    return (value - min) <= (max - min);
}

static int popcnt32(mfxU32 bits) {
    bits = (bits & 0x55555555) + (bits >> 1 & 0x55555555);
    bits = (bits & 0x33333333) + (bits >> 2 & 0x33333333);
    bits = (bits & 0x0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f);
    bits = (bits & 0x00ff00ff) + (bits >> 8 & 0x00ff00ff);
    return (bits & 0x0000ffff) + (bits >>16 & 0x0000ffff);
}

static TCHAR *alloc_str(const TCHAR *str, size_t length = 0) {
    const size_t count = (length) ? length : _tcslen(str);
    TCHAR *ptr = (TCHAR *)calloc(count + 1, sizeof(str[0]));
    memcpy(ptr, str, sizeof(str[0]) * count);
    return ptr;
}

template<typename type>
static std::basic_string<type> repeatStr(std::basic_string<type> str, int count) {
    std::basic_string<type> ret;
    for (int i = 0; i < count; i++) {
        ret += str;
    }
    return ret;
}

static tstring fourccToStr(mfxU32 nFourCC) {
    tstring fcc;
    for (int i = 0; i < 4; i++) {
        fcc.push_back((msdk_char)*(i + (char*)&nFourCC));
    }
    return fcc;
}

bool check_ext(const TCHAR *filename, const std::vector<const char*>& ext_list);

typedef struct CX_DESC {
    TCHAR *desc;
    int value;
} CX_DESC;

typedef struct FEATURE_DESC {
    TCHAR *desc;
    uint64_t value;
} FEATURE_DESC;

static const int QSV_TIMEBASE = 90000;

#define INIT_MFX_EXT_BUFFER(x, id) { MSDK_ZERO_MEMORY(x); (x).Header.BufferId = (id); (x).Header.BufferSz = sizeof(x); }

static const mfxVersion LIB_VER_LIST[] = {
    {  0, 0 },
    {  0, 1 },
    {  1, 1 },
    {  3, 1 },
    {  4, 1 },
    {  6, 1 },
    {  7, 1 },
    {  8, 1 },
    {  9, 1 },
    { 10, 1 },
    { 11, 1 },
    { 13, 1 },
    { 15, 1 },
    { NULL, NULL } 
};

#define MFX_LIB_VERSION_0_0  LIB_VER_LIST[ 0]
#define MFX_LIB_VERSION_1_0  LIB_VER_LIST[ 1]
#define MFX_LIB_VERSION_1_1  LIB_VER_LIST[ 2]
#define MFX_LIB_VERSION_1_3  LIB_VER_LIST[ 3]
#define MFX_LIB_VERSION_1_4  LIB_VER_LIST[ 4]
#define MFX_LIB_VERSION_1_6  LIB_VER_LIST[ 5]
#define MFX_LIB_VERSION_1_7  LIB_VER_LIST[ 6]
#define MFX_LIB_VERSION_1_8  LIB_VER_LIST[ 7]
#define MFX_LIB_VERSION_1_9  LIB_VER_LIST[ 8]
#define MFX_LIB_VERSION_1_10 LIB_VER_LIST[ 9]
#define MFX_LIB_VERSION_1_11 LIB_VER_LIST[10]
#define MFX_LIB_VERSION_1_13 LIB_VER_LIST[11]
#define MFX_LIB_VERSION_1_15 LIB_VER_LIST[12]

BOOL Check_HWUsed(mfxIMPL impl);
mfxVersion get_mfx_libhw_version();
mfxVersion get_mfx_libsw_version();
mfxVersion get_mfx_lib_version(mfxIMPL impl);
BOOL check_lib_version(mfxVersion value, mfxVersion required);
BOOL check_lib_version(mfxU32 _value, mfxU32 _required);

static bool inline rc_is_type_lookahead(int rc) {
    return ((rc == MFX_RATECONTROL_LA)
        | (rc == MFX_RATECONTROL_LA_ICQ)
        | (rc == MFX_RATECONTROL_LA_EXT)
        | (rc == MFX_RATECONTROL_LA_HRD));
}

enum : uint64_t {
    ENC_FEATURE_CURRENT_RC             = 0x0000000000000001,
    ENC_FEATURE_AVBR                   = 0x0000000000000002,
    ENC_FEATURE_LA                     = 0x0000000000000004,
    ENC_FEATURE_ICQ                    = 0x0000000000000008,
    ENC_FEATURE_LA_ICQ                 = 0x0000000000000010,
    ENC_FEATURE_VCM                    = 0x0000000000000020,
    ENC_FEATURE_AUD                    = 0x0000000000000040,
    ENC_FEATURE_PIC_STRUCT             = 0x0000000000000080,
    ENC_FEATURE_VUI_INFO               = 0x0000000000000100,
    ENC_FEATURE_CAVLC                  = 0x0000000000000200,
    ENC_FEATURE_RDO                    = 0x0000000000000400,
    ENC_FEATURE_ADAPTIVE_I             = 0x0000000000000800,
    ENC_FEATURE_ADAPTIVE_B             = 0x0000000000001000,
    ENC_FEATURE_B_PYRAMID              = 0x0000000000002000,
    ENC_FEATURE_TRELLIS                = 0x0000000000004000,
    ENC_FEATURE_EXT_BRC                = 0x0000000000008000,
    ENC_FEATURE_MBBRC                  = 0x0000000000010000,
    ENC_FEATURE_LA_DS                  = 0x0000000000020000,
    ENC_FEATURE_INTERLACE              = 0x0000000000040000,
    ENC_FEATURE_SCENECHANGE            = 0x0000000000080000,
    ENC_FEATURE_B_PYRAMID_AND_SC       = 0x0000000000100000,
    ENC_FEATURE_B_PYRAMID_MANY_BFRAMES = 0x0000000000200000,
    ENC_FEATURE_LA_HRD                 = 0x0000000000400000,
    ENC_FEATURE_LA_EXT                 = 0x0000000000800000,
    ENC_FEATURE_QVBR                   = 0x0000000001000000,
    ENC_FEATURE_INTRA_REFRESH          = 0x0000000002000000,
    ENC_FEATURE_NO_DEBLOCK             = 0x0000000004000000,
    ENC_FEATURE_QP_MINMAX              = 0x0000000008000000,
    ENC_FEATURE_WINBRC                 = 0x0000000010000000,
    ENC_FEATURE_PERMBQP                = 0x0000000020000000,
    ENC_FEATURE_DIRECT_BIAS_ADJUST     = 0x0000000040000000,
    ENC_FEATURE_GLOBAL_MOTION_ADJUST   = 0x0000000080000000,
};

enum : uint64_t {
    VPP_FEATURE_RESIZE                = 0x00000001,
    VPP_FEATURE_DENOISE               = 0x00000002,
    VPP_FEATURE_DETAIL_ENHANCEMENT    = 0x00000004,
    VPP_FEATURE_PROC_AMP              = 0x00000008,
    VPP_FEATURE_IMAGE_STABILIZATION   = 0x00000010,
    VPP_FEATURE_VIDEO_SIGNAL_INFO     = 0x00000020,
    VPP_FEATURE_FPS_CONVERSION        = 0x00000040,
    VPP_FEATURE_FPS_CONVERSION_ADV    = 0x00000080 | VPP_FEATURE_FPS_CONVERSION,
    VPP_FEATURE_DEINTERLACE           = 0x00000100,
    VPP_FEATURE_DEINTERLACE_AUTO      = 0x00000200,
    VPP_FEATURE_DEINTERLACE_IT_MANUAL = 0x00000400,
};

static const CX_DESC list_rate_control_ry[] = {
    { _T("CBR  "), MFX_RATECONTROL_CBR    },
    { _T("VBR  "), MFX_RATECONTROL_VBR    },
    { _T("AVBR "), MFX_RATECONTROL_AVBR   },
    { _T("QVBR "), MFX_RATECONTROL_QVBR   },
    { _T("CQP  "), MFX_RATECONTROL_CQP    },
    { _T("VQP  "), MFX_RATECONTROL_VQP    },
    { _T("LA   "), MFX_RATECONTROL_LA     },
    { _T("LAHRD"), MFX_RATECONTROL_LA_HRD },
    { _T("ICQ  "), MFX_RATECONTROL_ICQ    },
    { _T("LAICQ"), MFX_RATECONTROL_LA_ICQ },
    //{ _T("LAEXT"), MFX_RATECONTROL_LA_EXT },
    { _T("VCM  "), MFX_RATECONTROL_VCM    },
};
static const FEATURE_DESC list_enc_feature[] = {
    { _T("RC mode      "), ENC_FEATURE_CURRENT_RC             },
    { _T("Interlace    "), ENC_FEATURE_INTERLACE              },
    { _T("SceneChange  "), ENC_FEATURE_SCENECHANGE            },
    { _T("VUI info     "), ENC_FEATURE_VUI_INFO               },
    //{ _T("aud          "), ENC_FEATURE_AUD                    },
    //{ _T("pic_struct   "), ENC_FEATURE_PIC_STRUCT             },
    { _T("Trellis      "), ENC_FEATURE_TRELLIS                },
    //{ _T("rdo          "), ENC_FEATURE_RDO                    },
    //{ _T("CAVLC        "), ENC_FEATURE_CAVLC                  },
    { _T("Adaptive_I   "), ENC_FEATURE_ADAPTIVE_I             },
    { _T("Adaptive_B   "), ENC_FEATURE_ADAPTIVE_B             },
    { _T("B_Pyramid    "), ENC_FEATURE_B_PYRAMID              },
    { _T(" +Scenechange"), ENC_FEATURE_B_PYRAMID_AND_SC       },
    { _T(" +ManyBframes"), ENC_FEATURE_B_PYRAMID_MANY_BFRAMES },
    { _T("Ext_BRC      "), ENC_FEATURE_EXT_BRC                },
    { _T("MBBRC        "), ENC_FEATURE_MBBRC                  },
    { _T("LA Quality   "), ENC_FEATURE_LA_DS                  },
    { _T("QP Min/Max   "), ENC_FEATURE_QP_MINMAX              },
    { _T("IntraRefresh "), ENC_FEATURE_INTRA_REFRESH          },
    { _T("No Debloc    "), ENC_FEATURE_NO_DEBLOCK             },
    { _T("Windowed BRC "), ENC_FEATURE_WINBRC                 },
    { _T("PerMBQP(CQP) "), ENC_FEATURE_PERMBQP                },
    { _T("DirectBiasAdj"), ENC_FEATURE_DIRECT_BIAS_ADJUST     },
    { _T("MVCostScaling"), ENC_FEATURE_GLOBAL_MOTION_ADJUST   },
    { NULL, 0 },
};
static const FEATURE_DESC list_vpp_feature[] = {
    { _T("Resize               "), VPP_FEATURE_RESIZE              },
    { _T("Deinterlace          "), VPP_FEATURE_DEINTERLACE         },
    { _T("Denoise              "), VPP_FEATURE_DENOISE             },
    { _T("Detail Enhancement   "), VPP_FEATURE_DETAIL_ENHANCEMENT  },
    { _T("Proc Amp.            "), VPP_FEATURE_PROC_AMP            },
    { _T("Image Stabilization  "), VPP_FEATURE_IMAGE_STABILIZATION },
    { _T("Video Signal Info    "), VPP_FEATURE_VIDEO_SIGNAL_INFO   },
    { _T("FPS Conversion       "), VPP_FEATURE_FPS_CONVERSION      },
    { _T("FPS Conversion (Adv.)"), VPP_FEATURE_FPS_CONVERSION_ADV  },
    { NULL, 0 },
};

mfxU64 CheckEncodeFeature(mfxSession session, mfxVersion ver, mfxU16 ratecontrol, mfxU32 codecId);
mfxU64 CheckEncodeFeature(bool hardware, mfxVersion ver, mfxU16 ratecontrol, mfxU32 codecId);
mfxU64 CheckEncodeFeature(bool hardware, mfxU16 ratecontrol, mfxU32 codecId);
vector<mfxU64> MakeFeatureList(bool hardware, mfxVersion ver, const vector<const CX_DESC>& rateControlList, mfxU32 codecId);
vector<vector<mfxU64>> MakeFeatureListPerCodec(bool hardware, const vector<const CX_DESC>& rateControlList, const vector<mfxU32>& codecIdList);
vector<vector<mfxU64>> MakeFeatureListPerCodec(bool hardware, mfxVersion ver, const vector<const CX_DESC>& rateControlList, const vector<mfxU32>& codecIdList);
tstring MakeFeatureListStr(mfxU64 feature);
tstring MakeFeatureListStr(bool hardware);

mfxU64 CheckVppFeatures(bool hardware, mfxVersion ver);
tstring MakeVppFeatureStr(bool hardware);

bool check_if_d3d11_necessary();

const TCHAR *getOSVersion();
BOOL is_64bit_os();
UINT64 getPhysicalRamSize(UINT64 *ramUsed);
void getEnviromentInfo(TCHAR *buf, unsigned int buffer_size, bool add_ram_info = true);

void adjust_sar(int *sar_w, int *sar_h, int width, int height);

//拡張子が一致するか確認する
static BOOL _tcheck_ext(const TCHAR *filename, const TCHAR *ext) {
    return (_tcsicmp(PathFindExtension(filename), ext) == NULL) ? TRUE : FALSE;
}

const TCHAR *get_vpp_image_stab_mode_str(int mode);

BOOL check_OS_Win8orLater();

mfxStatus ParseY4MHeader(char *buf, mfxFrameInfo *info);

const TCHAR *get_err_mes(int sts);

mfxStatus AppendMfxBitstream(mfxBitstream *bitstream, const mfxU8 *data, mfxU32 size);

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

static void __forceinline sleep_hybrid(int count) {
    _mm_pause();
    if ((count & 255) == 255) {
         Sleep((count & 1023) == 1023);
    }
}

const int MAX_FILENAME_LEN = 1024;


enum {
    QSV_LOG_TRACE = -3,
    QSV_LOG_DEBUG = -2,
    QSV_LOG_MORE = -1,
    QSV_LOG_INFO = 0,
    QSV_LOG_WARN,
    QSV_LOG_ERROR,
};

int qsv_print_stderr(int log_level, const TCHAR *mes, HANDLE handle = NULL);

class CQSVLog {
protected:
    int m_nLogLevel = QSV_LOG_INFO;
    const TCHAR *m_pStrLog = nullptr;
    CRITICAL_SECTION cs;
public:
    CQSVLog(const TCHAR *pLogFile, int log_level = QSV_LOG_INFO) {
        InitializeCriticalSection(&cs);
        init(pLogFile, log_level);
    };
    virtual ~CQSVLog() {
        DeleteCriticalSection(&cs);
    };
    void init(const TCHAR *pLogFile, int log_level = QSV_LOG_INFO) {
        m_pStrLog = pLogFile;
        m_nLogLevel = log_level;
        if (m_nLogLevel == QSV_LOG_DEBUG) {
            TCHAR cpuInfo[256];
            TCHAR gpu_info[1024] = { 0 };
            getCPUInfo(cpuInfo, _countof(cpuInfo));
            getGPUInfo("Intel", gpu_info, _countof(gpu_info));
            (*this)(QSV_LOG_DEBUG, _T("QSVEnc    %s (%s)\n"), VER_STR_FILEVERSION_TCHAR, BUILD_ARCH_STR);
            (*this)(QSV_LOG_DEBUG, _T("OS        %s (%s)\n"), getOSVersion(), is_64bit_os() ? _T("x64") : _T("x86"));
            (*this)(QSV_LOG_DEBUG, _T("CPU Info  %s\n"), cpuInfo);
            (*this)(QSV_LOG_DEBUG, _T("GPU Info  %s\n"), gpu_info);
        }
    };
    int getLogLevel() {
        return m_nLogLevel;
    }
    virtual void operator()(int log_level, const TCHAR *format, ...);
};

#endif //_QSV_UTIL_H_
