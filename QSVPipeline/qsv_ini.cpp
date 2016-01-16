//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#if !(defined(_WIN32) || defined(_WIN64))

#include <cstdio>
#include <regex>
#include <string>
#include "qsv_tchar.h"
#include "qsv_osdep.h"

uint32_t GetPrivateProfileString(const TCHAR *Section, const TCHAR *Key, const TCHAR *Default, TCHAR *buf, size_t nSize, const TCHAR *IniFile) {
    FILE *fp = fopen(IniFile, "r");
    if (fp != NULL) {
        TCHAR buffer[1024];

        auto tsection = std::basic_string<TCHAR>("[");
        tsection += Section;
        tsection += "]";

        bool bTargetSection = false;
        while (_fgetts(buffer, _countof(buffer), fp) != NULL) {
            if (buffer[0] == _T('[')) {
                bTargetSection = (_tcscmp(buffer, tsection.c_str()) == 0);
            } else if (bTargetSection) {
                char *pDelim = _tcschr(buffer, _T('='));
                if (pDelim != NULL) {
                    *pDelim = _T('\0');
                    if (_tcscmp(buffer, Key) == 0) {
                        _tcscpy(buf, pDelim+1);
                        return _tcslen(buf);
                    }
                }
            }
        }
    }
    _tcscpy(buf, Default);
    return _tcslen(buf);
}

#endif //#if !(defined(_WIN32) || defined(_WIN64))
