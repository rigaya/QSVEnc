//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#pragma once

using namespace System;
using namespace System::IO;
using namespace System::Collections::Generic;

#include "qsv_prm.h"
#include "mfxstructures.h"

#define HIDE_MPEG2

namespace QSVEnc {

	ref class LocalSettings 
	{
	public:
		List<String^>^ audEncName;
		List<String^>^ audEncExeName;
		List<String^>^ audEncPath;
		String^ MP4MuxerExeName;
		String^ MP4MuxerPath;
		String^ MKVMuxerExeName;
		String^ MKVMuxerPath;
		String^ TC2MP4ExeName;
		String^ TC2MP4Path;
		String^ MPGMuxerExeName;
		String^ MPGMuxerPath;
		String^ MP4RawExeName;
		String^ MP4RawPath;
		String^ CustomTmpDir;
		String^ CustomAudTmpDir;
		String^ CustomMP4TmpDir;
		String^ LastAppDir;
		String^ LastBatDir;

		LocalSettings() {
			audEncName = gcnew List<String^>();
			audEncExeName = gcnew List<String^>();
			audEncPath = gcnew List<String^>();
		}
		~LocalSettings() {
			delete audEncName;
			delete audEncExeName;
			delete audEncPath;
		}
	};

	value struct ExeControls
	{
		String^ Name;
		String^ Path;
		const char* args;
	};
};

const int fcgTBQualityTimerLatency = 600;
const int fcgTBQualityTimerPeriod = 40;
const int fcgTXCmdfulloffset = 57;
const int fcgCXAudioEncModeSmallWidth = 189;
const int fcgCXAudioEncModeLargeWidth = 237;


static const WCHAR * const list_aspect_ratio[] = {
	L"SAR(PAR, 画素比)で指定",
	L"DAR(画面比)で指定",
	NULL
};

static const WCHAR * const list_tempdir[] = {
	L"出力先と同じフォルダ (デフォルト)",
	L"システムの一時フォルダ",
	L"カスタム",
	NULL
};

static const WCHAR * const list_audtempdir[] = {
	L"変更しない",
	L"カスタム",
	NULL
};

static const WCHAR * const list_mp4boxtempdir[] = {
	L"指定しない",
	L"カスタム",
	NULL
};

const CX_DESC list_deinterlace[] = {
	{ "なし",                   MFX_DEINTERLACE_NONE  },
	{ "インタレ解除 (通常)",    MFX_DEINTERLACE_NORMAL  },
	{ "インタレ解除 (24fps化)", MFX_DEINTERLACE_IT  },
	{ "インタレ解除 (Bob化)",   MFX_DEINTERLACE_BOB },
	{ NULL, NULL } 
};

const CX_DESC list_encmode[] = {
	{ "ビットレート指定 - CBR",           MFX_RATECONTROL_CBR },
	{ "ビットレート指定 - VBR",           MFX_RATECONTROL_VBR },
	{ "ビットレート指定 - AVBR",          MFX_RATECONTROL_AVBR},
	{ "固定量子化量 (CQP)",               MFX_RATECONTROL_CQP },
	{ "可変QP (VQP, プログレッシブのみ)", MFX_RATECONTROL_VQP },
	{ "先行探索レート制御",               MFX_RATECONTROL_LA  },
	{ NULL, NULL }
};

const WCHAR * const audio_enc_timing_desc[] = {
	L"後",
	L"前",
	L"同時",
	NULL
};

const CX_DESC list_outtype[] = {
	{ "H.264 / AVC", MFX_CODEC_AVC },
#ifndef HIDE_MPEG2
	{ "MPEG2", MFX_CODEC_MPEG2 },
#endif
	//{ "VC-1", MFX_CODEC_VC1 },
	{ NULL, NULL }
};

//メモ表示用 RGB
const int StgNotesColor[][3] = {
	{  80,  72,  92 },
	{ 120, 120, 120 }
};

const WCHAR * const DefaultStgNotes = L"メモ...";