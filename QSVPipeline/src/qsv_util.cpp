//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <stdio.h>
#include <vector>
#include <sstream>
#include <algorithm>
#include "mfxStructures.h"
#include "mfxvideo.h"
#include "mfxvideo++.h"
#include "sample_defs.h"
#include "qsv_util.h"
#include "qsv_prm.h"

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
		MFXVideoSession *test = new MFXVideoSession();
		mfxVersion ver;
		memcpy(&ver, &LIB_VER_LIST[i], sizeof(mfxVersion));
		mfxStatus sts = test->Init(impl, &ver);
		test->Close();
		delete test;
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
	bool check_d3d9  = (0 != check_lib_version(get_mfx_lib_version(MFX_IMPL_HARDWARE_ANY | MFX_IMPL_VIA_D3D9), MFX_LIB_VERSION_1_1));

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

mfxU32 CheckEncodeFeature(mfxSession session, mfxVersion mfxVer, mfxU16 ratecontrol) {
	const std::vector<std::pair<mfxU16, mfxVersion>> rc_list = {
		{ MFX_RATECONTROL_VBR,    MFX_LIB_VERSION_1_1 },
		{ MFX_RATECONTROL_CBR,    MFX_LIB_VERSION_1_1 },
		{ MFX_RATECONTROL_CQP,    MFX_LIB_VERSION_1_1 },
		{ MFX_RATECONTROL_VQP,    MFX_LIB_VERSION_1_1 },
		{ MFX_RATECONTROL_AVBR,   MFX_LIB_VERSION_1_3 },
		{ MFX_RATECONTROL_LA,     MFX_LIB_VERSION_1_7 },
		{ MFX_RATECONTROL_LA_ICQ, MFX_LIB_VERSION_1_8 },
		{ MFX_RATECONTROL_VCM,    MFX_LIB_VERSION_1_8 },
	};
	for (auto rc : rc_list) {
		if (ratecontrol == rc.first) {
			if (!check_lib_version(mfxVer, rc.second)) {
				return 0x00;
			}
			break;
		}
	}

	MFXVideoENCODE encode(session);

	mfxExtCodingOption cop;
	MSDK_ZERO_MEMORY(cop);
	cop.Header.BufferId = MFX_EXTBUFF_CODING_OPTION;
	cop.Header.BufferSz = sizeof(mfxExtCodingOption);

	mfxExtCodingOption2 cop2;
	MSDK_ZERO_MEMORY(cop2);
	cop2.Header.BufferId = MFX_EXTBUFF_CODING_OPTION2;
	cop2.Header.BufferSz = sizeof(mfxExtCodingOption2);

	std::vector<mfxExtBuffer *> buf;
	buf.push_back((mfxExtBuffer *)&cop);
	if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_6)) {
		buf.push_back((mfxExtBuffer *)&cop2);
	}

	mfxVideoParam videoPrm;
	MSDK_ZERO_MEMORY(videoPrm);

	auto set_default_quality_prm = [&videoPrm]() {
		if (   videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_VBR
			|| videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_AVBR
			|| videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_CBR
			|| videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_LA
			|| videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_VCM) {
			videoPrm.mfx.TargetKbps = 3000;
			videoPrm.mfx.MaxKbps    = 3000; //videoPrm.mfx.MaxKbpsはvideoPrm.mfx.TargetKbpsと一致させないとCBRの時に失敗する
			if (videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_AVBR) {
				videoPrm.mfx.Accuracy     = 500;
				videoPrm.mfx.Convergence  = 90;
			}
		} else if (
			   videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_CQP
			|| videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_VQP) {
			videoPrm.mfx.QPI = 23;
			videoPrm.mfx.QPP = 23;
			videoPrm.mfx.QPB = 23;
		} else {
			videoPrm.mfx.ICQQuality = 23;
		}
	};

	videoPrm.NumExtParam = (mfxU16)buf.size();
	videoPrm.ExtParam = (buf.size()) ? &buf[0] : NULL;
	videoPrm.AsyncDepth                  = 3;
	videoPrm.IOPattern                   = MFX_IOPATTERN_IN_SYSTEM_MEMORY;
	videoPrm.mfx.CodecId                 = MFX_CODEC_AVC;
	videoPrm.mfx.RateControlMethod       = (ratecontrol == MFX_RATECONTROL_VQP) ? MFX_RATECONTROL_CQP : ratecontrol;
	videoPrm.mfx.CodecLevel              = MFX_LEVEL_AVC_41;
	videoPrm.mfx.CodecProfile            = MFX_PROFILE_AVC_HIGH;
	videoPrm.mfx.TargetUsage             = MFX_TARGETUSAGE_BEST_QUALITY;
	videoPrm.mfx.EncodedOrder            = 0;
	videoPrm.mfx.NumSlice                = 1;
	videoPrm.mfx.NumRefFrame             = 2;
	videoPrm.mfx.GopPicSize              = USHRT_MAX;
	videoPrm.mfx.GopOptFlag              = MFX_GOP_CLOSED;
	videoPrm.mfx.GopRefDist              = 3;
	videoPrm.mfx.FrameInfo.FrameRateExtN = 30000;
	videoPrm.mfx.FrameInfo.FrameRateExtD = 1001;
	videoPrm.mfx.FrameInfo.FourCC        = MFX_FOURCC_NV12;
	videoPrm.mfx.FrameInfo.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
	videoPrm.mfx.FrameInfo.PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
	videoPrm.mfx.FrameInfo.AspectRatioW  = 1;
	videoPrm.mfx.FrameInfo.AspectRatioH  = 1;
	videoPrm.mfx.FrameInfo.Width         = 1280;
	videoPrm.mfx.FrameInfo.Height        = 720;
	videoPrm.mfx.FrameInfo.CropX         = 0;
	videoPrm.mfx.FrameInfo.CropY         = 0;
	videoPrm.mfx.FrameInfo.CropW         = 1280;
	videoPrm.mfx.FrameInfo.CropH         = 720;
	set_default_quality_prm();

	mfxExtCodingOption copOut;
	mfxExtCodingOption2 cop2Out;
	std::vector<mfxExtBuffer *> bufOut;
	bufOut.push_back((mfxExtBuffer *)&copOut);
	if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_6)) {
		bufOut.push_back((mfxExtBuffer *)&cop2Out);
	}
	mfxVideoParam videoPrmOut;
	//In, Outのパラメータが同一となっているようにきちんとコピーする
	//そうしないとQueryが失敗する
	MSDK_MEMCPY(&copOut, &cop, sizeof(cop));
	MSDK_MEMCPY(&cop2Out, &cop2, sizeof(cop2));
	MSDK_MEMCPY(&videoPrmOut, &videoPrm, sizeof(videoPrm));
	videoPrm.NumExtParam = (mfxU16)bufOut.size();
	videoPrm.ExtParam = &bufOut[0];

	mfxStatus ret = encode.Query(&videoPrm, &videoPrmOut);
	
	mfxU32 result = (MFX_ERR_NONE <= ret && videoPrm.mfx.RateControlMethod == videoPrmOut.mfx.RateControlMethod) ? ENC_FEATURE_CURRENT_RC : 0x00;
	if (MFX_ERR_NONE <= ret) {

		//まず、エンコードモードについてチェック
		auto check_enc_mode = [&](mfxU16 mode, mfxU32 flag, mfxVersion required_ver) {
			if (check_lib_version(mfxVer, required_ver)) {
				mfxU16 original_method = videoPrm.mfx.RateControlMethod;
				videoPrm.mfx.RateControlMethod = mode;
				set_default_quality_prm();
				MSDK_MEMCPY(&copOut, &cop, sizeof(cop));
				MSDK_MEMCPY(&cop2Out, &cop2, sizeof(cop2));
				MSDK_MEMCPY(&videoPrmOut, &videoPrm, sizeof(videoPrm));
				videoPrm.NumExtParam = (mfxU16)bufOut.size();
				videoPrm.ExtParam = &bufOut[0];
				if (MFX_ERR_NONE <= encode.Query(&videoPrm, &videoPrmOut) && videoPrm.mfx.RateControlMethod == videoPrmOut.mfx.RateControlMethod)
					result |= flag;
				videoPrm.mfx.RateControlMethod = original_method;
				set_default_quality_prm();
			}
		};
		check_enc_mode(MFX_RATECONTROL_AVBR, ENC_FEATURE_AVBR, MFX_LIB_VERSION_1_3);
		check_enc_mode(MFX_RATECONTROL_LA,   ENC_FEATURE_LA,   MFX_LIB_VERSION_1_7);
		check_enc_mode(MFX_RATECONTROL_ICQ,  ENC_FEATURE_ICQ,  MFX_LIB_VERSION_1_8);
		check_enc_mode(MFX_RATECONTROL_VCM,  ENC_FEATURE_VCM,  MFX_LIB_VERSION_1_8);
	
#define CHECK_FEATURE(membersIn, membersOut, flag, value, required_ver) { \
		if (check_lib_version(mfxVer, (required_ver))) { \
			(membersIn) = (value); \
			MSDK_MEMCPY(&copOut, &cop, sizeof(cop)); \
			MSDK_MEMCPY(&cop2Out, &cop2, sizeof(cop2)); \
			if (MFX_ERR_NONE <= encode.Query(&videoPrm, &videoPrmOut) \
				&& (membersIn) == (membersOut) \
				&& videoPrm.mfx.RateControlMethod == videoPrmOut.mfx.RateControlMethod) \
				result |= (flag); \
			(membersIn) = 0; \
		} \
	}\
		//これはもう単純にAPIチェックでOK
		if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_3)) {
			result |= ENC_FEATURE_VUI_INFO;
		}
		//ひとつひとつパラメータを入れ替えて試していく
		const mfxU32 MFX_TRELLIS_ALL = MFX_TRELLIS_I | MFX_TRELLIS_P | MFX_TRELLIS_B;
		CHECK_FEATURE(cop.AUDelimiter,       copOut.AUDelimiter,       ENC_FEATURE_AUD,        MFX_CODINGOPTION_ON, MFX_LIB_VERSION_1_1);
		CHECK_FEATURE(cop.PicTimingSEI,      copOut.PicTimingSEI,      ENC_FEATURE_PIC_STRUCT, MFX_CODINGOPTION_ON, MFX_LIB_VERSION_1_1);
		CHECK_FEATURE(cop.RateDistortionOpt, copOut.RateDistortionOpt, ENC_FEATURE_RDO,        MFX_CODINGOPTION_ON, MFX_LIB_VERSION_1_1);
		CHECK_FEATURE(cop.CAVLC,             copOut.CAVLC,             ENC_FEATURE_CAVLC,      MFX_CODINGOPTION_ON, MFX_LIB_VERSION_1_1);
		CHECK_FEATURE(cop2.ExtBRC,           cop2Out.ExtBRC,           ENC_FEATURE_EXT_BRC,    MFX_CODINGOPTION_ON, MFX_LIB_VERSION_1_6);
		CHECK_FEATURE(cop2.MBBRC,            cop2Out.MBBRC,            ENC_FEATURE_MBBRC,      MFX_CODINGOPTION_ON, MFX_LIB_VERSION_1_6);
		CHECK_FEATURE(cop2.Trellis,          cop2Out.Trellis,          ENC_FEATURE_TRELLIS,    MFX_TRELLIS_ALL,     MFX_LIB_VERSION_1_7);
		CHECK_FEATURE(cop2.AdaptiveI,        cop2Out.AdaptiveI,        ENC_FEATURE_ADAPTIVE_I, MFX_CODINGOPTION_ON, MFX_LIB_VERSION_1_8);
		CHECK_FEATURE(cop2.AdaptiveB,        cop2Out.AdaptiveB,        ENC_FEATURE_ADAPTIVE_B, MFX_CODINGOPTION_ON, MFX_LIB_VERSION_1_8);
		CHECK_FEATURE(cop2.BRefType,         cop2Out.BRefType,         ENC_FEATURE_B_PYRAMID,  MFX_B_REF_PYRAMID,   MFX_LIB_VERSION_1_8);
		if (MFX_RATECONTROL_LA == ratecontrol || MFX_RATECONTROL_LA_ICQ == ratecontrol) {
			CHECK_FEATURE(cop2.LookAheadDS,  cop2Out.LookAheadDS,      ENC_FEATURE_LA_DS,      MFX_LOOKAHEAD_DS_2x, MFX_LIB_VERSION_1_8);
		}

		//以下特殊な場合
		if (   MFX_RATECONTROL_LA     == ratecontrol
			|| MFX_RATECONTROL_LA_ICQ == ratecontrol) {
			result &= ~ENC_FEATURE_RDO;
			result &= ~ENC_FEATURE_MBBRC;
			result &= ~ENC_FEATURE_EXT_BRC;
		} else if (MFX_RATECONTROL_CQP == ratecontrol
			    || MFX_RATECONTROL_VQP == ratecontrol) {
			result &= ~ENC_FEATURE_MBBRC;
			result &= ~ENC_FEATURE_EXT_BRC;
		}
	}
#undef CHECK_FEATURE
	return result;
}

//サポートする機能のチェックをAPIバージョンのみで行う
//API v1.6以降はCheckEncodeFeatureを使うべき
//同一のAPIバージョンでも環境により異なることが多くなるため
static mfxU32 CheckEncodeFeatureStatic(mfxVersion mfxVer, mfxU16 ratecontrol) {
	mfxU32 feature = 0x00;
	//まずレート制御モードをチェック
	if (!check_lib_version(mfxVer, MFX_LIB_VERSION_1_8)
		&& (MFX_RATECONTROL_ICQ    == ratecontrol
		 || MFX_RATECONTROL_LA_ICQ == ratecontrol
		 || MFX_RATECONTROL_VCM    == ratecontrol)) {
		return feature;
	}
	if (!check_lib_version(mfxVer, MFX_LIB_VERSION_1_7)
		&& MFX_RATECONTROL_LA == ratecontrol) {
		return feature;
	}
	if (!check_lib_version(mfxVer, MFX_LIB_VERSION_1_3)
		&& MFX_RATECONTROL_AVBR == ratecontrol) {
		return feature;
	}

	//各モードをチェック
	feature |= ENC_FEATURE_CURRENT_RC;

	if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_1)) {
		feature |= ENC_FEATURE_AUD;
		feature |= ENC_FEATURE_PIC_STRUCT;
		feature |= ENC_FEATURE_RDO;
		feature |= ENC_FEATURE_CAVLC;
	}
	if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_3)) {
		feature |= ENC_FEATURE_VUI_INFO;
	}
	if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_6)) {
		feature |= ENC_FEATURE_EXT_BRC;
		feature |= ENC_FEATURE_MBBRC;
	}
	if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_7)) {
		feature |= ENC_FEATURE_TRELLIS;
	}
	if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_8)) {
		feature |= ENC_FEATURE_ADAPTIVE_I;
		feature |= ENC_FEATURE_ADAPTIVE_B;
		feature |= ENC_FEATURE_B_PYRAMID;
		feature |= ENC_FEATURE_VUI_INFO;
		if (MFX_RATECONTROL_LA == ratecontrol || MFX_RATECONTROL_LA_ICQ == ratecontrol) {
			feature |= ENC_FEATURE_LA_DS;
		}
	}

	//以下特殊な場合の制限
	if (   MFX_RATECONTROL_LA     == ratecontrol
		|| MFX_RATECONTROL_LA_ICQ == ratecontrol) {
		feature &= ~ENC_FEATURE_RDO;
		feature &= ~ENC_FEATURE_MBBRC;
		feature &= ~ENC_FEATURE_EXT_BRC;
	} else if (MFX_RATECONTROL_CQP == ratecontrol
			|| MFX_RATECONTROL_VQP == ratecontrol) {
		feature &= ~ENC_FEATURE_MBBRC;
		feature &= ~ENC_FEATURE_EXT_BRC;
	}

	return feature;
}

mfxU32 CheckEncodeFeature(bool hardware, mfxVersion ver, mfxU16 ratecontrol) {
	mfxU32 feature = 0x00;
	if (!check_lib_version(ver, MFX_LIB_VERSION_1_6)) {
		//API v1.6未満で実際にチェックする必要は殆ど無いので、
		//コードで決められた値を返すようにする
		feature = CheckEncodeFeatureStatic(ver, ratecontrol);
	} else {
		mfxSession session;

		mfxStatus ret = MFXInit((hardware) ? MFX_IMPL_HARDWARE_ANY : MFX_IMPL_SOFTWARE, &ver, &session);

		feature = (MFX_ERR_NONE == ret) ? CheckEncodeFeature(session, ver, ratecontrol) : 0x00;

		MFXClose(session);
	}

	return feature;
}

mfxU32 CheckEncodeFeature(bool hardware, mfxU16 ratecontrol) {
	mfxVersion ver = (hardware) ? get_mfx_libhw_version() : get_mfx_libsw_version();
	return CheckEncodeFeature(hardware, ver, ratecontrol);
}

const msdk_char *EncFeatureStr(mfxU32 enc_feature) {
	for (const CX_DESC *ptr = list_enc_feature; ptr->desc; ptr++)
		if (enc_feature == (mfxU32)ptr->value)
			return ptr->desc;
	return NULL;
}

void MakeFeatureList(bool hardware, mfxVersion ver, const CX_DESC *rateControlList, int rateControlCount, std::vector<mfxU32>& availableFeatureForEachRC) {
	availableFeatureForEachRC.resize(rateControlCount, 0);

	for (int i_rc = 0; i_rc < rateControlCount; i_rc++) {
		availableFeatureForEachRC[i_rc] = CheckEncodeFeature(hardware, ver, (mfxU16)rateControlList[i_rc].value);
	}
}

void MakeFeatureList(bool hardware, const CX_DESC *rateControlList, int rateControlCount, std::vector<mfxU32>& availableFeatureForEachRC) {
	MakeFeatureList(hardware, (hardware) ? get_mfx_libhw_version() : get_mfx_libsw_version(), rateControlList, rateControlCount, availableFeatureForEachRC);
}

void MakeFeatureListStr(bool hardware, std::basic_string<msdk_char>& str) {

	std::vector<mfxU32> availableFeatureForEachRC;
	MakeFeatureList(hardware, list_rate_control_ry, _countof(list_rate_control_ry), availableFeatureForEachRC);
	
	str.clear();
	
	//ヘッダ部分
	const mfxU32 row_header_length = (mfxU32)_tcslen(list_enc_feature[0].desc);
	for (mfxU32 i = 1; i < row_header_length; i++)
		str += _T(" ");

	for (mfxU32 i = 0; i < _countof(list_rate_control_ry); i++) {
		str += _T(" ");
		str += list_rate_control_ry[i].desc;
	}
	str += _T("\n");
	
	//モードがサポートされているか
	TCHAR *MARK_YES_NO[] = {  _T(" x    "), _T(" o    ") };
	for (const CX_DESC *ptr = list_enc_feature; ptr->desc; ptr++) {
		str += ptr->desc;
		for (mfxU32 i = 0; i < _countof(list_rate_control_ry); i++) {
			str += MARK_YES_NO[!!(availableFeatureForEachRC[i] & ptr->value)];
		}
		str += _T("\n");
	}
}

mfxU32 GCD(mfxU32 a, mfxU32 b)
{
	if (0 == a)
		return b;
	else if (0 == b)
		return a;

	mfxU32 a1, b1;

	if (a >= b)
	{
		a1 = a;
		b1 = b;
	}
	else
	{
		a1 = b;
		b1 = a;
	}

	// a1 >= b1;
	mfxU32 r = a1 % b1;

	while (0 != r)
	{
		a1 = b1;
		b1 = r;
		r = a1 % b1;
	}

	return b1;
}
mfxI64 GCDI64(mfxI64 a, mfxI64 b)
{
	if (0 == a)
		return b;
	else if (0 == b)
		return a;

	mfxI64 a1, b1;

	if (a >= b)
	{
		a1 = a;
		b1 = b;
	}
	else
	{
		a1 = b;
		b1 = a;
	}

	// a1 >= b1;
	mfxI64 r = a1 % b1;

	while (0 != r)
	{
		a1 = b1;
		b1 = r;
		r = a1 % b1;
	}

	return b1;
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

const TCHAR *get_err_mes(int sts) {
	switch (sts) {
		case MFX_ERR_NONE:                     return _T("no error.");
		case MFX_ERR_UNKNOWN:                  return _T("unknown error.");
		case MFX_ERR_NULL_PTR:                 return _T("null pointer.");
		case MFX_ERR_UNSUPPORTED:              return _T("undeveloped feature.");
		case MFX_ERR_MEMORY_ALLOC:             return _T("failed to allocate memory.");
		case MFX_ERR_NOT_ENOUGH_BUFFER:        return _T("insufficient buffer at input/output.");
		case MFX_ERR_INVALID_HANDLE:           return _T("invalid handle.");
		case MFX_ERR_LOCK_MEMORY:              return _T("failed to lock the memory block.");
		case MFX_ERR_NOT_INITIALIZED:          return _T("member function called before initialization.");
		case MFX_ERR_NOT_FOUND:                return _T("the specified object is not found.");
		case MFX_ERR_MORE_DATA:                return _T("expect more data at input.");
		case MFX_ERR_MORE_SURFACE:             return _T("expect more surface at output.");
		case MFX_ERR_ABORTED:                  return _T("operation aborted.");
		case MFX_ERR_DEVICE_LOST:              return _T("lose the HW acceleration device.");
		case MFX_ERR_INCOMPATIBLE_VIDEO_PARAM: return _T("incompatible video parameters.");
		case MFX_ERR_INVALID_VIDEO_PARAM:      return _T("invalid video parameters.");
		case MFX_ERR_UNDEFINED_BEHAVIOR:       return _T("undefined behavior.");
		case MFX_ERR_DEVICE_FAILED:            return _T("device operation failure.");
		
		case MFX_WRN_IN_EXECUTION:             return _T("the previous asynchrous operation is in execution.");
		case MFX_WRN_DEVICE_BUSY:              return _T("the HW acceleration device is busy.");
		case MFX_WRN_VIDEO_PARAM_CHANGED:      return _T("the video parameters are changed during decoding.");
		case MFX_WRN_PARTIAL_ACCELERATION:     return _T("SW is used.");
		case MFX_WRN_INCOMPATIBLE_VIDEO_PARAM: return _T("incompatible video parameters.");
		case MFX_WRN_VALUE_NOT_CHANGED:        return _T("the value is saturated based on its valid range.");
		case MFX_WRN_OUT_OF_RANGE:             return _T("the value is out of valid range.");
		default:                               return _T("unknown error."); 
	}
}

mfxStatus ParseY4MHeader(char *buf, mfxFrameInfo *info) {
	char *p, *q = NULL;
	memset(info, 0, sizeof(mfxFrameInfo));
	for (p = buf; (p = strtok_s(p, " ", &q)) != NULL; ) {
		switch (*p) {
			case 'W':
				{
					char *eptr = NULL;
					int w = strtol(p+1, &eptr, 10);
					if (*eptr == '\0' && w)
						info->Width = (mfxU16)w;
				}
				break;
			case 'H':
				{
					char *eptr = NULL;
					int h = strtol(p+1, &eptr, 10);
					if (*eptr == '\0' && h)
						info->Height = (mfxU16)h;
				}
				break;
			case 'F':
				{
					int rate = 0, scale = 0;
					if (   (info->FrameRateExtN == 0 || info->FrameRateExtD == 0)
						&& sscanf_s(p+1, "%d:%d", &rate, &scale) == 2) {
							if (rate && scale) {
								info->FrameRateExtN = rate;
								info->FrameRateExtD = scale;
							}
					}
				}
				break;
			case 'A':
				{
					int sar_x = 0, sar_y = 0;
					if (   (info->AspectRatioW == 0 || info->AspectRatioH == 0)
						&& sscanf_s(p+1, "%d:%d", &sar_x, &sar_y) == 2) {
							if (sar_x && sar_y) {
								info->AspectRatioW = (mfxU16)sar_x;
								info->AspectRatioH = (mfxU16)sar_y;
							}
					}
				}
				break;
			case 'I':
				switch (*(p+1)) {
			case 'b':
				info->PicStruct = MFX_PICSTRUCT_FIELD_BFF;
				break;
			case 't':
			case 'm':
				info->PicStruct = MFX_PICSTRUCT_FIELD_TFF;
				break;
			default:
				break;
				}
				break;
			case 'C':
				if (   0 != _strnicmp(p+1, "420",      strlen("420"))
					&& 0 != _strnicmp(p+1, "420mpeg2", strlen("420mpeg2"))
					&& 0 != _strnicmp(p+1, "420jpeg",  strlen("420jpeg"))
					&& 0 != _strnicmp(p+1, "420paldv", strlen("420paldv"))) {
					return MFX_PRINT_OPTION_ERR;
				}
				break;
			default:
				break;
		}
		p = NULL;
	}
	return MFX_ERR_NONE;
}

#include <Windows.h>
#include <process.h>

BOOL check_OS_Win8orLater() {
	OSVERSIONINFO osvi = { 0 };
	osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
	GetVersionEx(&osvi);
	return ((osvi.dwPlatformId == VER_PLATFORM_WIN32_NT) && ((osvi.dwMajorVersion == 6 && osvi.dwMinorVersion >= 2) || osvi.dwMajorVersion > 6));
}

const TCHAR *getOSVersion() {
	const TCHAR *ptr = _T("Unknown");
	OSVERSIONINFO info = { 0 };
	info.dwOSVersionInfoSize = sizeof(info);
	GetVersionEx(&info);
	switch (info.dwPlatformId) {
	case VER_PLATFORM_WIN32_WINDOWS:
		if (4 <= info.dwMajorVersion) {
			switch (info.dwMinorVersion) {
			case 0:  ptr = _T("Windows 95"); break;
			case 10: ptr = _T("Windows 98"); break;
			case 90: ptr = _T("Windows Me"); break;
			default: break;
			}
		}
		break;
	case VER_PLATFORM_WIN32_NT:
		switch (info.dwMajorVersion) {
		case 3:
			switch (info.dwMinorVersion) {
			case 0:  ptr = _T("Windows NT 3"); break;
			case 1:  ptr = _T("Windows NT 3.1"); break;
			case 5:  ptr = _T("Windows NT 3.5"); break;
			case 51: ptr = _T("Windows NT 3.51"); break;
			default: break;
			}
			break;
		case 4:
			if (0 == info.dwMinorVersion)
				ptr = _T("Windows NT 4.0");
			break;
		case 5:
			switch (info.dwMinorVersion) {
			case 0:  ptr = _T("Windows 2000"); break;
			case 1:  ptr = _T("Windows XP"); break;
			case 2:  ptr = _T("Windows Server 2003"); break;
			default: break;
			}
			break;
		case 6:
			switch (info.dwMinorVersion) {
			case 0:  ptr = _T("Windows Vista"); break;
			case 1:  ptr = _T("Windows 7"); break;
			case 2:  ptr = _T("Windows 8"); break;
			case 3:  ptr = _T("Windows 8.1"); break;
			default:
				if (4 <= info.dwMinorVersion) {
					ptr = _T("Later than Windows 8.1");
				}
				break;
			}
			break;
		default:
			if (7 <= info.dwPlatformId) {
				ptr = _T("Later than Windows 8.1");
			}
			break;
		}
		break;
	default:
		break;
	}
	return ptr;
}

BOOL is_64bit_os() {
	SYSTEM_INFO sinfo = { 0 };
	GetNativeSystemInfo(&sinfo);
	return sinfo.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64;
}

UINT64 getPhysicalRamSize(UINT64 *ramUsed) {
	MEMORYSTATUSEX msex ={ 0 };
	msex.dwLength = sizeof(msex);
	GlobalMemoryStatusEx(&msex);
	if (NULL != ramUsed)
		*ramUsed = msex.ullTotalPhys - msex.ullAvailPhys;
	return msex.ullTotalPhys;
}

void getEnviromentInfo(TCHAR *buf, unsigned int buffer_size) {
	ZeroMemory(buf, sizeof(buf[0]) * buffer_size);

	TCHAR cpu_info[1024] = { 0 };
	getCPUInfo(cpu_info, _countof(cpu_info));

	TCHAR gpu_info[1024] = { 0 };
	getGPUInfo("Intel", gpu_info, _countof(gpu_info));

	UINT64 UsedRamSize = 0;
	UINT64 totalRamsize = getPhysicalRamSize(&UsedRamSize);

	auto add_tchar_to_buf = [buf, buffer_size](const TCHAR *fmt, ...) {
		unsigned int buf_length = (unsigned int)_tcslen(buf);
		va_list args = { 0 };
		va_start(args, fmt);
		_vstprintf_s(buf + buf_length, buffer_size - buf_length, fmt, args);
		va_end(args);
	};

	add_tchar_to_buf(_T("Environment Info\n"));
	add_tchar_to_buf(_T("OS : %s (%s)\n"), getOSVersion(), is_64bit_os() ? _T("x64") : _T("x86"));
	add_tchar_to_buf(_T("CPU: %s\n"), cpu_info);
	add_tchar_to_buf(_T("GPU: %s\n"), gpu_info);
	add_tchar_to_buf(_T("RAM: DDRx-xxxx / x channel (Total %d MB / Used %d MB)\n"), (UINT)(totalRamsize >> 20), (UINT)(UsedRamSize >> 20));
}
