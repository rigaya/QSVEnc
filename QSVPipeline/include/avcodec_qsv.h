//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------
#ifndef _AVCODEC_QSV_H_
#define _AVCODEC_QSV_H_

#include "qsv_version.h"

#if ENABLE_AVCODEC_QSV_READER
#include <Windows.h>

#pragma warning (push)
#pragma warning (disable: 4244)
extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/error.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}
#pragma comment (lib, "avcodec.lib")
#pragma comment (lib, "avformat.lib")
#pragma comment (lib, "avutil.lib")
#pragma warning (pop)

#if _DEBUG
#define QSV_AV_LOG_LEVEL AV_LOG_WARNING
#else
#define QSV_AV_LOG_LEVEL AV_LOG_FATAL
#endif

//必要なavcodecのdllがそろっているかを確認
static bool check_avcodec_dll() {
	std::vector<HMODULE> hDllList;
	static const TCHAR *AVCODEC_DLL_NAME[] = {
		_T("avcodec-56.dll"), _T("avformat-56.dll"), _T("avutil-54.dll")
	};
	bool check = true;
	for (int i = 0; i < _countof(AVCODEC_DLL_NAME); i++) {
		HMODULE hDll = NULL;
		if (NULL == (hDll = LoadLibrary(AVCODEC_DLL_NAME[i]))) {
			check = false;
			break;
		}
		hDllList.push_back(hDll);
	}
	for (auto hDll : hDllList) {
		FreeLibrary(hDll);
	}
	return check;
}

//avcodecのライセンスがLGPLであるかどうかを確認
static bool checkAvcodecLicense() {
	auto check = [](const char *license) {
		std::string str(license);
		transform(str.begin(), str.end(), str.begin(), [](char in) -> char {return (char)tolower(in); });
		return std::string::npos != str.find("lgpl");
	};
	return (check(avutil_license()) && check(avcodec_license()) && check(avformat_license()));
}

#endif //ENABLE_AVCODEC_QSV_READER

#endif //_AVCODEC_QSV_H_
