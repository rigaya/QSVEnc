//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#include <algorithm>
#include <cctype>
#include "avcodec_reader.h"

#if ENABLE_AVCODEC_QSV_READER

#if 0
#define AVDEBUG_PRINT(fmt, ...) _ftprintf(stderr, _T(fmt), __VA_ARGS__)
#else
#define AVDEBUG_PRINT(fmt, ...)
#endif

CAvcodecReader::CAvcodecReader()  {
	MSDK_ZERO_MEMORY(demux);
	memset(m_sPacket, 0, sizeof(m_sPacket));
}

CAvcodecReader::~CAvcodecReader() {

}

void CAvcodecReader::Close() {
	//リソースの解放
	for (int i = 0; i < _countof(m_sPacket); i++) {
		if (m_sPacket[i].data) {
			av_free_packet(&m_sPacket[i]);
		}
	}
	memset(m_sPacket, 0, sizeof(m_sPacket));

	//close bitstreamfilter
	if (demux.bsfc) {
		av_bitstream_filter_close(demux.bsfc);
	}
	
	//close codec
	if (demux.pCodecCtx) {
		avcodec_close(demux.pCodecCtx);
	}
	
	//close video file
	if (demux.pFormatCtx) {
		avformat_close_input(&demux.pFormatCtx);
	}
	
	if (demux.extradata) {
		av_free(demux.extradata);
	}

	//free input buffer (使用していない)
	//if (buffer) {
	//	free(buffer);
	//	buffer = nullptr;
	//}

	MSDK_ZERO_MEMORY(demux);
}

bool CAvcodecReader::checkAvcodecDll() {
	std::vector<HMODULE> hDllList;
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

bool CAvcodecReader::checkAvcodecLicense() {
	auto check = [](const char *license) {
		std::string str(license);
		transform(str.begin(), str.end(), str.begin(), [](char in) -> char {return (char)tolower(in); });
		return std::string::npos != str.find("lgpl");
	};
	return (check(avutil_license()) && check(avcodec_license()) && check(avformat_license()));
}

mfxU32 CAvcodecReader::getQSVFourcc(mfxU32 id) {
	for (int i = 0; i < _countof(QSV_LIST); i++)
		if (QSV_LIST[i].codec_id == id)
			return QSV_LIST[i].qsv_fourcc;
	return 0;
}

int CAvcodecReader::getVideoStream() {
	int videoIndex = -1;
	const int n_streams = demux.pFormatCtx->nb_streams;
	for (int i = 0; i < n_streams; i++) {
		if (demux.pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
			videoIndex = i;
			break;
		}
	}
	return videoIndex;
}

#pragma warning(push)
#pragma warning(disable:4100)
mfxStatus CAvcodecReader::Init(const TCHAR *strFileName, mfxU32 ColorFormat, int option, CEncodingThread *pEncThread, CEncodeStatusInfo *pEncSatusInfo, sInputCrop *pInputCrop) {
	if (!checkAvcodecDll()) {
		m_strInputInfo += _T("avcodec: failed to load dlls.\n");
		return MFX_ERR_NULL_PTR;
	}
	//if (!checkAvcodecLicense()) {
	//	m_strInputInfo += _T("avcodec: invalid dlls for QSVEncC.\n");
	//	return MFX_ERR_NULL_PTR;
	//}

	Close();

	MSDK_CHECK_POINTER(pEncThread, MFX_ERR_NULL_PTR);
	m_pEncThread = pEncThread;

	MSDK_CHECK_POINTER(pEncSatusInfo, MFX_ERR_NULL_PTR);
	m_pEncSatusInfo = pEncSatusInfo;

	MSDK_CHECK_POINTER(pInputCrop, MFX_ERR_NULL_PTR);
	memcpy(&m_sInputCrop, pInputCrop, sizeof(m_sInputCrop));

	av_register_all();
	avcodec_register_all();
	av_log_set_level(AV_LOG_FATAL);
	
	std::string filename_char;
	if (0 == tchar_to_string(strFileName, filename_char)) {
		m_strInputInfo += _T("avcodec: failed to convert to ansi characters.\n");
		return MFX_ERR_INVALID_HANDLE;
	}
	if (avformat_open_input(&(demux.pFormatCtx), filename_char.c_str(), nullptr, nullptr)) {
		m_strInputInfo += _T("avcodec: error opening file\n");
		return MFX_ERR_NULL_PTR; // Couldn't open file
	}
	AVDEBUG_PRINT("avcodec: opened file.\n");
	if (avformat_find_stream_info(demux.pFormatCtx, nullptr) < 0) {
		m_strInputInfo += _T("avcodec: error finding stream information!\n");
		return MFX_ERR_NULL_PTR; // Couldn't find stream information
	}
	AVDEBUG_PRINT("avcodec: got stream information.\n");
	//dump_format(dec.pFormatCtx, 0, argv[1], 0);
	
	//動画ストリームを探す
	if (-1 == (demux.videoIndex = getVideoStream())) {
		m_strInputInfo += _T("avcodec: unable to find video stream!\n");
		return MFX_ERR_NULL_PTR; // Didn't find a video stream
	}
	AVDEBUG_PRINT("avcodec: found video stream.\n");

	demux.pCodecCtx = demux.pFormatCtx->streams[demux.videoIndex]->codec;

	//QSVでデコード可能かチェック
	if (0 == (m_nInputCodec = getQSVFourcc(demux.pCodecCtx->codec_id))) {
		m_strInputInfo += _T("avcodec: unable to decode by qsv.\n");
		return MFX_ERR_NULL_PTR;
	}
	AVDEBUG_PRINT("avcodec: can be decoded by qsv.\n");

	//必要ならbitstream filterを初期化
	if (m_nInputCodec == MFX_CODEC_AVC && demux.pCodecCtx->extradata && demux.pCodecCtx->extradata[0] == 1) {
		if (NULL == (demux.bsfc = av_bitstream_filter_init("h264_mp4toannexb"))) {
			m_strInputInfo += _T("avcodec: unable to init h264_mp4toannexb.\n");
			return MFX_ERR_NULL_PTR;
		}
		AVDEBUG_PRINT("avcodec: success to init h264_mp4toannexb.\n");
	}
	AVDEBUG_PRINT("avcodec: start demuxing... \n");
	
	mfxStatus decHeaderSts = MFX_ERR_NONE;
	mfxBitstream bitstream = { 0 };
	if (MFX_ERR_NONE != (decHeaderSts = GetHeader(&bitstream))) {
		m_strInputInfo += _T("avcodec: failed to get header.\n");
		return decHeaderSts;
	}
	
	if (m_nInputCodec == MFX_CODEC_AVC) {
		//これを付加しないとMFXVideoDECODE_DecodeHeaderが成功しない
		const mfxU32 IDR = 0x65010000;
		AppendMfxBitstream(&bitstream, (mfxU8 *)&IDR, sizeof(IDR));
	}

	mfxSession session = { 0 };
	mfxVersion version = MFX_LIB_VERSION_1_1;
	if (MFX_ERR_NONE != (decHeaderSts = MFXInit(MFX_IMPL_HARDWARE_ANY, &version, &session))) {
		m_strInputInfo += _T("avcodec: unable to init qsv decoder.\n");
		return decHeaderSts;
	}

	memset(&m_sDecParam, 0, sizeof(m_sDecParam));
	m_sDecParam.mfx.CodecId = m_nInputCodec;
	m_sDecParam.IOPattern = (mfxU16)((option) ? MFX_IOPATTERN_OUT_VIDEO_MEMORY : MFX_IOPATTERN_OUT_SYSTEM_MEMORY);
	decHeaderSts = MFXVideoDECODE_DecodeHeader(session, &bitstream, &m_sDecParam);
	MFXClose(session);
	if (MFX_ERR_NONE != decHeaderSts) {
		m_strInputInfo += _T("avcodec: failed to decode header.\n");
		return decHeaderSts;
	}
	WipeMfxBitstream(&bitstream);

	const mfxU32 fps_gcd = GCD(m_sDecParam.mfx.FrameInfo.FrameRateExtN, m_sDecParam.mfx.FrameInfo.FrameRateExtD);
	m_sDecParam.mfx.FrameInfo.FrameRateExtN /= fps_gcd;
	m_sDecParam.mfx.FrameInfo.FrameRateExtD /= fps_gcd;

	//情報を格納
	memcpy(&m_inputFrameInfo, &m_sDecParam.mfx.FrameInfo, sizeof(m_inputFrameInfo));
	//m_inputFrameInfoのWidthとHeightには解像度をそのまま入れて、
	//他の読み込みに合わせる
	//もともとは16ないし32でアラインされた数字が入っている
	m_inputFrameInfo.Width = m_inputFrameInfo.CropW;
	m_inputFrameInfo.Height = m_inputFrameInfo.CropH;
	//フレーム数は未定
	*(DWORD*)&m_inputFrameInfo.FrameId = 0;
	
	TCHAR mes[256];
	_stprintf_s(mes, _countof(mes), _T("avcodec (%s), %dx%d, %d/%d fps"), CodecIdToStr(m_nInputCodec).c_str(),
		m_inputFrameInfo.Width, m_inputFrameInfo.Height, m_inputFrameInfo.FrameRateExtN, m_inputFrameInfo.FrameRateExtD);
	m_strInputInfo += mes;
	m_tmLastUpdate = timeGetTime();
	return MFX_ERR_NONE;
}
#pragma warning(pop)

int CAvcodecReader::getSample(AVPacket *pkt) {
	av_init_packet(pkt);
	while (av_read_frame(demux.pFormatCtx, pkt) >= 0) {
		if (pkt->stream_index == demux.videoIndex) {
			if (demux.bsfc) {
				mfxU8 *data = NULL;
				int dataSize = 0;
				std::swap(demux.extradata,     demux.pCodecCtx->extradata);
				std::swap(demux.extradataSize, demux.pCodecCtx->extradata_size);
				av_bitstream_filter_filter(demux.bsfc, demux.pCodecCtx, nullptr,
					&data, &dataSize, pkt->data, pkt->size, 0);
				std::swap(demux.extradata,     demux.pCodecCtx->extradata);
				std::swap(demux.extradataSize, demux.pCodecCtx->extradata_size);
				av_free_packet(pkt); //メモリ解放を忘れない
				av_packet_from_data(pkt, data, dataSize);
			}
			return 0;
		}
		av_free_packet(pkt);
	}
	//ファイルの終わりに到達
	pkt->data = nullptr;
	pkt->size = 0;
	return 1;
}

mfxStatus CAvcodecReader::setToMfxBitstream(mfxBitstream *bitstream, AVPacket *pkt) {
	mfxStatus sts = MFX_ERR_NONE;
	if (pkt->data) {
		sts = AppendMfxBitstream(bitstream, pkt->data, pkt->size);
	} else {
		sts = MFX_ERR_MORE_DATA;
	}
	return sts;
}

mfxStatus CAvcodecReader::GetNextBitstream(mfxBitstream *bitstream) {
	mfxStatus sts = setToMfxBitstream(bitstream, &m_sPacket[demux.sampleGetCount % _countof(m_sPacket)]);
	demux.sampleGetCount++;
	return sts;
}

mfxStatus CAvcodecReader::GetHeader(mfxBitstream *bitstream) {
	if (bitstream == nullptr)
		return MFX_ERR_NULL_PTR;
	if (bitstream->Data == nullptr)
		InitMfxBitstream(bitstream, AVCODEC_READER_INPUT_BUF_SIZE);

	if (demux.extradata == nullptr) {
		demux.extradataSize = demux.pCodecCtx->extradata_size;
		//ここでav_mallocを使用しないと正常に動作しない
		demux.extradata = (mfxU8 *)av_malloc(demux.pCodecCtx->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE);
		//ヘッダのデータをコピーしておく
		memcpy(demux.extradata, demux.pCodecCtx->extradata, demux.extradataSize);
		memset(demux.extradata + demux.extradataSize, 0, FF_INPUT_BUFFER_PADDING_SIZE);

		if (demux.bsfc && demux.extradata[0] == 1) {
			mfxU8 *dummy = NULL;
			int dummy_size = 0;
			std::swap(demux.extradata,     demux.pCodecCtx->extradata);
			std::swap(demux.extradataSize, demux.pCodecCtx->extradata_size);
			av_bitstream_filter_filter(demux.bsfc, demux.pCodecCtx, nullptr, &dummy, &dummy_size, nullptr, 0, 0);
			std::swap(demux.extradata,     demux.pCodecCtx->extradata);
			std::swap(demux.extradataSize, demux.pCodecCtx->extradata_size);
		}
	}
	
	memcpy(bitstream->Data, demux.extradata, demux.extradataSize);
	bitstream->DataLength = demux.extradataSize;
	return MFX_ERR_NONE;
}

#pragma warning(push)
#pragma warning(disable:4100)
mfxStatus CAvcodecReader::LoadNextFrame(mfxFrameSurface1* pSurface) {
	AVPacket *pkt = &m_sPacket[demux.sampleLoadCount % _countof(m_sPacket)];
	if (pkt->data) {
		av_free_packet(pkt);
		pkt->data = nullptr;
		pkt->size = 0;
	}
	if (getSample(pkt)) {
		av_free_packet(pkt);
		pkt->data = nullptr;
		pkt->size = 0;
		return MFX_ERR_MORE_DATA; //ファイルの終わりに到達
	}
	demux.sampleLoadCount++;
	m_pEncSatusInfo->m_nInputFrames++;
	mfxU32 tm = timeGetTime();
	if (tm - m_tmLastUpdate > UPDATE_INTERVAL) {
		m_tmLastUpdate = tm;
		m_pEncSatusInfo->UpdateDisplay(tm, 0);
	}
	return MFX_ERR_NONE;
}
#pragma warning(pop)

#endif //ENABLE_AVCODEC_QSV_READER
