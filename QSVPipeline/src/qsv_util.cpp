//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include "mfxStructures.h"
#include "mfxvideo.h"
#include "mfxvideo++.h"
#include "qsv_util.h"

BOOL Check_HWUsed(mfxIMPL impl) {
	static const int HW_list[] = {
		MFX_IMPL_HARDWARE,
		MFX_IMPL_HARDWARE_ANY,
		MFX_IMPL_HARDWARE2,
		MFX_IMPL_HARDWARE3,
		MFX_IMPL_HARDWARE4,
		NULL
	};
	for (int i = 0; HW_list[i]; i++)
		if (HW_list[i] == (HW_list[i] & (int)impl))
			return TRUE;
	return FALSE;
}

mfxVersion get_mfx_lib_version(mfxIMPL impl) {
	int i;
	for (i = 1; LIB_VER_LIST[i].Major; i++) {
		MFXVideoSession test;
		mfxVersion ver;
		memcpy(&ver, &LIB_VER_LIST[i], sizeof(mfxVersion));
		mfxStatus sts = test.Init(impl, &ver);
		if (sts != MFX_ERR_NONE)
			break;
	}
	return LIB_VER_LIST[i-1];
}

mfxVersion get_mfx_libhw_version() {
	static const mfxU32 impl_list[] = {
		MFX_IMPL_HARDWARE_ANY | MFX_IMPL_VIA_D3D11,
		MFX_IMPL_HARDWARE_ANY,
		MFX_IMPL_HARDWARE,
	};
	mfxVersion test = { 0 };
	//Win7でD3D11のチェックをやると、
	//デスクトップコンポジションが切られてしまう問題が発生すると報告を頂いたので、
	//D3D11をWin8以降に限定
	for (int i = (check_OS_Win8orLater() ? 0 : 1); i < _countof(impl_list); i++) {
		test = get_mfx_lib_version(impl_list[i]);
		if (check_lib_version(test, MFX_LIB_VERSION_1_1))
			break;
	}
	return test;
}
bool check_if_d3d11_necessary() {
	bool check_d3d11 = (0 != check_lib_version(get_mfx_lib_version(MFX_IMPL_HARDWARE_ANY | MFX_IMPL_VIA_D3D11), MFX_LIB_VERSION_1_1));
	bool check_d3d9  = (0 != check_lib_version(get_mfx_lib_version(MFX_IMPL_HARDWARE_ANY), MFX_LIB_VERSION_1_1));

	return (check_d3d11 == true && check_d3d9 == false);
}
mfxVersion get_mfx_libsw_version() {
	return get_mfx_lib_version(MFX_IMPL_SOFTWARE);
}

BOOL check_lib_version(mfxVersion value, mfxVersion required) {
	if (value.Major < required.Major)
		return FALSE;
	if (value.Major > required.Major)
		return TRUE;
	if (value.Minor < required.Minor)
		return FALSE;
	return TRUE;
}

BOOL check_lib_version(mfxU32 _value, mfxU32 _required) {
	mfxVersion value, required;
	value.Version = _value;
	required.Version = _required;
	if (value.Major < required.Major)
		return FALSE;
	if (value.Major > required.Major)
		return TRUE;
	if (value.Minor < required.Minor)
		return FALSE;
	return TRUE;
}

void adjust_sar(int *sar_w, int *sar_h, int width, int height) {
	int aspect_w = *sar_w;
	int aspect_h = *sar_h;
	//正負チェック
	if (aspect_w * aspect_h <= 0)
		aspect_w = aspect_h = 0;
	else if (aspect_w < 0) {
		//負で与えられている場合はDARでの指定
		//SAR比に変換する
		int dar_x = -1 * aspect_w;
		int dar_y = -1 * aspect_h;
		int x = dar_x * height;
		int y = dar_y * width;
		//多少のづれは容認する
		if (abs(y - x) > 16 * dar_y) {
			//gcd
			int a = x, b = y, c;
			while ((c = a % b) != 0)
				a = b, b = c;
			*sar_w = x / b;
			*sar_h = y / b;
		} else {
			 *sar_w = *sar_h = 1;
		}
	} else {
		//sarも一応gcdをとっておく
		int a = aspect_w, b = aspect_h, c;
		while ((c = a % b) != 0)
			a = b, b = c;
		*sar_w = aspect_w / b;
		*sar_h = aspect_h / b;
	}
}
