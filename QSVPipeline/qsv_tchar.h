//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef __QSV_TCHAR_H__
#define __QSV_TCHAR_H__

#if defined(_WIN32) || defined(_WIN64)
#include <tchar.h>
#else
#include <cstddef>
#include <cstring>

typedef char TCHAR;
#define _T(x) x
#define _tmain main
#define _tcslen strlen
#define _ftprintf fprintf
#define _stscanf_s sscanf
#define _stscanf sscanf
#define _tcscmp strcmp
#define _tcsnccmp strncmp
#define _tcsicmp strcasecmp
#define _tcschr strchr
#define _tcsstr strstr
#define _tcstol strtol
#define _tfopen fopen
#define _tfopen_s fopen_s
#define _stprintf_s sprintf_s
#define _vsctprintf _vscprintf
#define _vstprintf_s _vsprintf_s
#define _tcstok_s strtok_s
#define _tcserror strerror
#define _fgetts fgets
#define _tcscpy strcpy

static inline char *_tcscpy_s(TCHAR *dst, const TCHAR *src) {
    return strcpy(dst, src);
}

static inline char *_tcscpy_s(TCHAR *dst, size_t size, const TCHAR *src) {
    return strcpy(dst, src);
}
#endif //#if defined(_WIN32) || defined(_WIN64)
#endif // __QSV_TCHAR_H__
