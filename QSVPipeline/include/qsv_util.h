#ifndef _QSV_UTIL_H_
#define _QSV_UTIL_H_

#include <Windows.h>
#include <tchar.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include "mfxstructures.h"

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

static const mfxVersion LIB_VER_LIST[] = {
	{ 0, 0 },
	{ 0, 1 },
	{ 1, 1 },
	{ 3, 1 },
	{ 4, 1 },
	{ 6, 1 },
	{ 7, 1 },
	{ NULL, NULL } 
};

#define MFX_LIB_VERSION_0_0 LIB_VER_LIST[0]
#define MFX_LIB_VERSION_1_1 LIB_VER_LIST[2]
#define MFX_LIB_VERSION_1_3 LIB_VER_LIST[3]
#define MFX_LIB_VERSION_1_4 LIB_VER_LIST[4]
#define MFX_LIB_VERSION_1_6 LIB_VER_LIST[5]
#define MFX_LIB_VERSION_1_7 LIB_VER_LIST[6]

BOOL Check_HWUsed(mfxIMPL impl);
mfxVersion get_mfx_libhw_version();
mfxVersion get_mfx_libsw_version();
mfxVersion get_mfx_lib_version(mfxIMPL impl);
BOOL check_lib_version(mfxVersion value, mfxVersion required);
BOOL check_lib_version(mfxU32 _value, mfxU32 _required);

bool check_if_d3d11_necessary();

void adjust_sar(int *sar_w, int *sar_h, int width, int height);

//拡張子が一致するか確認する
static BOOL _tcheck_ext(const TCHAR *filename, const TCHAR *ext) {
	return (_tcsicmp(PathFindExtension(filename), ext) == NULL) ? TRUE : FALSE;
}


static BOOL check_OS_Win8orLater() {
	OSVERSIONINFO osvi = { 0 };
	osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
	GetVersionEx(&osvi);
	return ((osvi.dwPlatformId == VER_PLATFORM_WIN32_NT) && ((osvi.dwMajorVersion == 6 && osvi.dwMinorVersion >= 2) || osvi.dwMajorVersion > 6));
}

const int MAX_FILENAME_LEN = 1024;

#endif //_QSV_UTIL_H_
