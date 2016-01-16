//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef __QSV_INI_H__
#define __QSV_INI_H__

#if !(defined(_WIN32) || defined(_WIN64))

uint32_t GetPrivateProfileString(const TCHAR *Section, const TCHAR *Key, const TCHAR *Default, TCHAR *buf, size_t nSize, const TCHAR *IniFile);

#define GetPrivateProfileStringA GetPrivateProfileString

#endif //#if !(defined(_WIN32) || defined(_WIN64))

#endif //__QSV_INI_H__

