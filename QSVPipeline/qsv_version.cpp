//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include "qsv_version.h"
#include "qsv_osdep.h"
#include "qsv_util.h"

#define SSTRING(str) STRING(str)
#define STRING(str) #str

const TCHAR *get_qsvenc_version() {
    return
#if QSVENC_AUO
        _T("QSVEnc (")
#else
        _T("QSVEncC (")
#endif
        BUILD_ARCH_STR _T(") ") VER_STR_FILEVERSION_TCHAR _T(" by rigaya, ")  _T(__DATE__) _T(" ") _T(__TIME__)
#if defined(_MSC_VER)
        _T(" (VC ") _T(SSTRING(_MSC_VER))
#elif defined(__clang__)
        _T(" (clang ") _T(SSTRING(__clang_major__)) _T(".") _T(SSTRING(__clang_minor__)) _T(".") _T(SSTRING(__clang_patchlevel__))
#elif defined(__GNUC__)
        _T(" (gcc ") _T(SSTRING(__GNUC__)) _T(".") _T(SSTRING(__GNUC_MINOR__)) _T(".") _T(SSTRING(__GNUC_PATCHLEVEL__))
#else
        _T(" (unknown")
#endif
        _T("/")
#ifdef _WIN32
        _T("Win/avx2")
#elif  __linux
        _T("Linux")
  #if defined(__AVX2__)
        _T("/avx2")
  #elif defined(__AVX__)
        _T("/avx")
  #else
        _T("/sse4.2")
  #endif
#else
        _T("unknown")
#endif
        _T(")");
}
