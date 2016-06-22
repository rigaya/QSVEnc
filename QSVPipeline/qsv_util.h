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

#ifndef _QSV_UTIL_H_
#define _QSV_UTIL_H_

#include "qsv_tchar.h"
#include <emmintrin.h>
#if defined(_WIN32) || defined(_WIN64)
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#endif
#include <vector>
#include <array>
#include <string>
#include <chrono>
#include <memory>
#include <type_traits>
#include "qsv_osdep.h"
#include "mfxstructures.h"
#include "mfxsession.h"
#include "qsv_version.h"
#include "cpu_info.h"
#include "gpu_info.h"

using std::vector;
using std::unique_ptr;
using std::shared_ptr;

#ifndef MIN3
#define MIN3(a,b,c) (min((a), min((b), (c))))
#endif
#ifndef MAX3
#define MAX3(a,b,c) (max((a), max((b), (c))))
#endif

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

#define ALIGN(x,align) (((x)+((align)-1))&(~((align)-1)))
#define ALIGN16(x) (((x)+15)&(~15))
#define ALIGN32(x) (((x)+31)&(~31))

#define QSV_MEMSET_ZERO(x) { memset(&(x), 0, sizeof(x)); }

template<typename T, size_t size>
std::vector<T> make_vector(T(&ptr)[size]) {
    return std::vector<T>(ptr, ptr + size);
}
template<typename T, size_t size>
std::vector<T> make_vector(const T(&ptr)[size]) {
    return std::vector<T>(ptr, ptr + size);
}
template<typename T0, typename T1>
std::vector<T0> make_vector(const T0 *ptr, T1 size) {
    static_assert(std::is_integral<T1>::value == true, "T1 should be integral");
    return (ptr && size) ? std::vector<T0>(ptr, ptr + size) : std::vector<T0>();
}
template<typename T0, typename T1>
std::vector<T0> make_vector(T0 *ptr, T1 size) {
    static_assert(std::is_integral<T1>::value == true, "T1 should be integral");
    return (ptr && size) ? std::vector<T0>(ptr, ptr + size) : std::vector<T0>();
}
template<typename T, typename ...Args>
constexpr std::array<T, sizeof...(Args)> make_array(Args&&... args) {
    return std::array<T, sizeof...(Args)>{ static_cast<Args&&>(args)... };
}
template<typename T, std::size_t N>
constexpr std::size_t array_size(const std::array<T, N>&) {
    return N;
}
template<typename T, std::size_t N>
constexpr std::size_t array_size(T(&)[N]) {
    return N;
}
template<typename T>
void vector_cat(vector<T>& v1, const vector<T>& v2) {
    if (v2.size()) {
        v1.insert(v1.end(), v2.begin(), v2.end());
    }
}
template<typename T>
static void qsv_free(T& ptr) {
    static_assert(std::is_pointer<T>::value == true, "T should be pointer");
    if (ptr) {
        free(ptr);
        ptr = nullptr;
    }
}

struct aligned_malloc_deleter {
    void operator()(void* ptr) const {
        _aligned_free(ptr);
    }
};

struct malloc_deleter {
    void operator()(void* ptr) const {
        free(ptr);
    }
};

struct fp_deleter {
    void operator()(FILE* fp) const {
        if (fp) {
            fflush(fp);
            fclose(fp);
        }
    }
};

struct handle_deleter {
    void operator()(HANDLE handle) const {
        if (handle) {
#if defined(_WIN32) || defined(_WIN64)
            CloseHandle(handle);
#endif //#if defined(_WIN32) || defined(_WIN64)
        }
    }
};

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

unsigned int wstring_to_string(const wchar_t *wstr, std::string& str, uint32_t codepage = CP_THREAD_ACP);
std::string wstring_to_string(const wchar_t *wstr, uint32_t codepage = CP_THREAD_ACP);
std::string wstring_to_string(const std::wstring& wstr, uint32_t codepage = CP_THREAD_ACP);
unsigned int char_to_wstring(std::wstring& wstr, const char *str, uint32_t codepage = CP_THREAD_ACP);
std::wstring char_to_wstring(const char *str, uint32_t = CP_THREAD_ACP);
std::wstring char_to_wstring(const std::string& str, uint32_t codepage = CP_THREAD_ACP);
#if defined(_WIN32) || defined(_WIN64)
std::wstring strsprintf(const WCHAR* format, ...);

std::wstring str_replace(std::wstring str, const std::wstring& from, const std::wstring& to);
std::wstring GetFullPath(const WCHAR *path);
bool qsv_get_filesize(const WCHAR *filepath, uint64_t *filesize);
std::pair<int, std::wstring> PathRemoveFileSpecFixed(const std::wstring& path);
std::wstring PathCombineS(const std::wstring& dir, const std::wstring& filename);
std::string PathCombineS(const std::string& dir, const std::string& filename);
bool CreateDirectoryRecursive(const WCHAR *dir);
#endif

std::wstring tchar_to_wstring(const tstring& tstr, uint32_t codepage = CP_THREAD_ACP);
std::wstring tchar_to_wstring(const TCHAR *tstr, uint32_t codepage = CP_THREAD_ACP);
unsigned int tchar_to_string(const TCHAR *tstr, std::string& str, uint32_t codepage = CP_THREAD_ACP);
std::string tchar_to_string(const TCHAR *tstr, uint32_t codepage = CP_THREAD_ACP);
std::string tchar_to_string(const tstring& tstr, uint32_t codepage = CP_THREAD_ACP);
unsigned int char_to_tstring(tstring& tstr, const char *str, uint32_t codepage = CP_THREAD_ACP);
tstring char_to_tstring(const char *str, uint32_t codepage = CP_THREAD_ACP);
tstring char_to_tstring(const std::string& str, uint32_t codepage = CP_THREAD_ACP);
unsigned int wstring_to_tstring(const WCHAR *wstr, tstring& tstr, uint32_t codepage = CP_THREAD_ACP);
tstring wstring_to_tstring(const WCHAR *wstr, uint32_t codepage = CP_THREAD_ACP);
tstring wstring_to_tstring(const std::wstring& wstr, uint32_t codepage = CP_THREAD_ACP);
std::string strsprintf(const char* format, ...);
std::vector<std::wstring> split(const std::wstring &str, const std::wstring &delim, bool bTrim = false);
std::vector<std::string> split(const std::string &str, const std::string &delim, bool bTrim = false);
std::string lstrip(const std::string& string, const char* trim = " \t\v\r\n");
std::string rstrip(const std::string& string, const char* trim = " \t\v\r\n");
std::string trim(const std::string& string, const char* trim = " \t\v\r\n");
std::wstring lstrip(const std::wstring& string, const WCHAR* trim = L" \t\v\r\n");
std::wstring rstrip(const std::wstring& string, const WCHAR* trim = L" \t\v\r\n");
std::wstring trim(const std::wstring& string, const WCHAR* trim = L" \t\v\r\n");

std::string str_replace(std::string str, const std::string& from, const std::string& to);
std::string GetFullPath(const char *path);
bool qsv_get_filesize(const char *filepath, uint64_t *filesize);
std::pair<int, std::string> PathRemoveFileSpecFixed(const std::string& path);
bool CreateDirectoryRecursive(const char *dir);

tstring print_time(double time);

tstring qsv_memtype_str(mfxU16 memtype);

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

static inline uint16_t check_coding_option(uint16_t value) {
    if (   value == MFX_CODINGOPTION_UNKNOWN
        || value == MFX_CODINGOPTION_ON
        || value == MFX_CODINGOPTION_OFF
        || value == MFX_CODINGOPTION_ADAPTIVE) {
        return value;
    }
    return MFX_CODINGOPTION_UNKNOWN;
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
        fcc.push_back((TCHAR)*(i + (char*)&nFourCC));
    }
    return fcc;
}

bool check_ext(const TCHAR *filename, const std::vector<const char*>& ext_list);

typedef struct CX_DESC {
    const TCHAR *desc;
    int value;
} CX_DESC;

typedef struct FEATURE_DESC {
    const TCHAR *desc;
    uint64_t value;
} FEATURE_DESC;

static const int QSV_TIMEBASE = 90000;

#define INIT_MFX_EXT_BUFFER(x, id) { QSV_MEMSET_ZERO(x); (x).Header.BufferId = (id); (x).Header.BufferSz = sizeof(x); }

#if defined(_WIN32) || defined(_WIN64)
bool check_if_d3d11_necessary();
tstring getOSVersion(OSVERSIONINFOEXW *osinfo = nullptr);
#else
tstring getOSVersion();
#endif
BOOL is_64bit_os();
uint64_t getPhysicalRamSize(uint64_t *ramUsed);
tstring getEnviromentInfo(bool add_ram_info = true);
void adjust_sar(int *sar_w, int *sar_h, int width, int height);

//拡張子が一致するか確認する
static BOOL _tcheck_ext(const TCHAR *filename, const TCHAR *ext) {
    return (_tcsicmp(PathFindExtension(filename), ext) == 0) ? TRUE : FALSE;
}

const TCHAR *get_vpp_image_stab_mode_str(int mode);

BOOL check_OS_Win8orLater();

mfxStatus ParseY4MHeader(char *buf, mfxFrameInfo *info);
mfxStatus WriteY4MHeader(FILE *fp, const mfxFrameInfo *info);

const TCHAR *get_err_mes(int sts);

const TCHAR *get_low_power_str(mfxU16 LowPower);

static void QSV_FORCEINLINE sse_memcpy(uint8_t *dst, const uint8_t *src, int size) {
    uint8_t *dst_fin = dst + size;
    uint8_t *dst_aligned_fin = (uint8_t *)(((size_t)dst_fin & ~15) - 64);
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
    uint8_t *dst_tmp = dst_fin - 64;
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

//確保できなかったら、サイズを小さくして再度確保を試みる (最終的にnMinSizeも確保できなかったら諦める)
size_t malloc_degeneracy(void **ptr, size_t nSize, size_t nMinSize);

int qsv_print_stderr(int log_level, const TCHAR *mes, HANDLE handle = NULL);

const int MAX_FILENAME_LEN = 1024;

const TCHAR *ColorFormatToStr(uint32_t format);
const TCHAR *CodecIdToStr(uint32_t nFourCC);
const TCHAR *TargetUsageToStr(uint16_t tu);
const TCHAR *EncmodeToStr(uint32_t enc_mode);
const TCHAR *MemTypeToStr(uint32_t memType);

mfxStatus mfxBitstreamInit(mfxBitstream *pBitstream, uint32_t nSize);
mfxStatus mfxBitstreamCopy(mfxBitstream *pBitstreamCopy, const mfxBitstream *pBitstream);
mfxStatus mfxBitstreamExtend(mfxBitstream *pBitstream, uint32_t nSize);
mfxStatus mfxBitstreamAppend(mfxBitstream *pBitstream, const uint8_t *data, uint32_t size);
void mfxBitstreamClear(mfxBitstream *pBitstream);

mfxExtBuffer *GetExtBuffer(mfxExtBuffer **ppExtBuf, int nCount, uint32_t targetBufferId);

const TCHAR *get_err_mes(int sts);
static void print_err_mes(int sts) {
    _ftprintf(stderr, _T("%s"), get_err_mes(sts));
}
#define QSV_IGNORE_STS(sts, err)                { if ((err) == (sts)) {(sts) = MFX_ERR_NONE; } }

int qsv_avx_dummy_if_avail(int bAVXAvail);

#endif //_QSV_UTIL_H_
