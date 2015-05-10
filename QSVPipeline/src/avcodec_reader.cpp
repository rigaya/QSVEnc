//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#include <algorithm>
#include <numeric>
#include <cctype>
#include <memory>
#include "mfxplugin.h"
#include "mfxplugin++.h"
#include "plugin_utils.h"
#include "plugin_loader.h"
#include "avcodec_reader.h"

#if ENABLE_AVCODEC_QSV_READER

#if _DEBUG
#define AVDEBUG_PRINT(fmt, ...) _ftprintf(stderr, _T(fmt), __VA_ARGS__)
#else
#define AVDEBUG_PRINT(fmt, ...)
#endif

tstring getAVQSVSupportedCodecList() {
	tstring codecs;
	for (int i = 0; i < _countof(QSV_DECODE_LIST); i++) {
		if (i) codecs += _T(", ");
		codecs += CodecIdToStr(QSV_DECODE_LIST[i].qsv_fourcc);
	}
	return codecs;
}

static inline void extend_array_size(VideoFrameData *dataset) {
	static int default_capacity = 8 * 1024;
	int current_cap = dataset->capacity;
	dataset->capacity = (current_cap) ? current_cap * 2 : default_capacity;
	dataset->frame = (FramePos *)realloc(dataset->frame, dataset->capacity * sizeof(dataset->frame[0]));
	memset(dataset->frame + current_cap, 0, sizeof(dataset->frame[0]) * (dataset->capacity - current_cap));
}

CAvcodecReader::CAvcodecReader()  {
	MSDK_ZERO_MEMORY(demux);
	MSDK_ZERO_MEMORY(m_sDecParam);
}

CAvcodecReader::~CAvcodecReader() {

}

void CAvcodecReader::clearAudioPacketList(std::vector<AVPacket>& pktList) {
	for (mfxU32 i_pkt = 0; i_pkt < pktList.size(); i_pkt++) {
		if (pktList[i_pkt].data) {
			av_free_packet(&pktList[i_pkt]);
		}
	}
	pktList.clear();
}

void CAvcodecReader::Close() {
	//リソースの解放
	for (int i = 0; i < _countof(m_AudioPacketsBufferL1); i++) {
		m_AudioPacketsBufferL1[i].clear();
	}
	clearAudioPacketList(m_AudioPacketsBufferL2);
	m_AudioPacketsBufferL2Used = 0;

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

	m_sTrimParam.list.clear();
	m_sTrimParam.offset = 0;

	//free input buffer (使用していない)
	//if (buffer) {
	//	free(buffer);
	//	buffer = nullptr;
	//}

	if (demux.videoFrameData.cs_initialized) {
		DeleteCriticalSection(&demux.videoFrameData.cs);
	}

	MSDK_ZERO_MEMORY(demux);
	MSDK_ZERO_MEMORY(m_sDecParam);
}

mfxU32 CAvcodecReader::getQSVFourcc(mfxU32 id) {
	for (int i = 0; i < _countof(QSV_DECODE_LIST); i++)
		if (QSV_DECODE_LIST[i].codec_id == id)
			return QSV_DECODE_LIST[i].qsv_fourcc;
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

int CAvcodecReader::getAudioStream() {
	int audioIndex = -1;
	const int n_streams = demux.pFormatCtx->nb_streams;
	for (int i = 0; i < n_streams; i++) {
		if (demux.pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) {
			audioIndex = i;
			break;
		}
	}
	return audioIndex;
}

void CAvcodecReader::sortVideoPtsList() {
	//フレーム順序が確定していないところをソートする
	FramePos *ptr = demux.videoFrameData.frame;
	std::sort(ptr + demux.videoFrameData.fixed_num, ptr + demux.videoFrameData.num,
		[](const FramePos& posA, const FramePos& posB) {
		return (abs(posA.pts - posB.pts) < 0xFFFFFFFF) ? posA.pts < posB.pts : posB.pts < posA.pts; });
}

void CAvcodecReader::addVideoPtsToList(FramePos pos) {
	if (demux.videoFrameData.capacity <= demux.videoFrameData.num+1) {
		EnterCriticalSection(&demux.videoFrameData.cs);
		extend_array_size(&demux.videoFrameData);
		LeaveCriticalSection(&demux.videoFrameData.cs);
	}
	demux.videoFrameData.frame[demux.videoFrameData.num] = pos;
	demux.videoFrameData.num++;

	if (demux.videoFrameData.fixed_num + 32 < demux.videoFrameData.num) {
		sortVideoPtsList();
		demux.videoFrameData.fixed_num += 16;
	}
}

mfxStatus CAvcodecReader::getFirstFramePosAndFrameRate(AVRational fpsDecoder) {
	const int max_check = 256;
	std::vector<FramePos> framePosList;
	framePosList.reserve(max_check);
	
	FramePos firstKeyframePos = { 0 };
	AVPacket pkt;
	for (int i = 0; i < max_check && !getSample(&pkt); i++) {
		FramePos pos = { pkt.pts, pkt.duration };
		framePosList.push_back(pos);
		if (firstKeyframePos.duration == 0 && pkt.flags & AV_PKT_FLAG_KEY) {
			firstKeyframePos = pos;
			//キーフレームに到達するまでQSVではフレームが出てこない
			//そのぶんのずれを記録しておき、Trim値などに補正をかける
			m_sTrimParam.offset = i;
		}
		av_free_packet(&pkt);
	}
	if (firstKeyframePos.duration == 0) {
		m_strInputInfo += _T("avcodec: failed to get first frame pos.\n");
		return MFX_ERR_UNKNOWN;
	}

	//durationを再計算する
	std::sort(framePosList.begin(), framePosList.end(), [](const FramePos& posA, const FramePos& posB) { return posA.pts < posB.pts; });
	for (int i = 0; i < (int)framePosList.size() - 1; i++) {
		int duration = (int)(framePosList[i+1].pts - framePosList[i].pts);
		if (abs(framePosList[i].duration - duration) <= 1) {
			framePosList[i].duration = duration;
		}
	}

	//durationのヒストグラムを作成
	std::vector<std::pair<int, int>> durationHistgram;
	for (auto pos : framePosList) {
		auto target = std::find_if(durationHistgram.begin(), durationHistgram.end(), [pos](const std::pair<int, int>& pair) { return pair.first == pos.duration; });
		if (target != durationHistgram.end()) {
			target->second++;
		} else {
			durationHistgram.push_back(std::make_pair(pos.duration, 1));
		}
	}
	//多い順にソートする
	std::sort(durationHistgram.begin(), durationHistgram.end(), [](const std::pair<int, int>& pairA, const std::pair<int, int>& pairB) { return pairA.second < pairB.second; });

	//durationの平均を求める
	auto avgDuration = std::accumulate(framePosList.begin(), framePosList.end(), 0, [](const int sum, const FramePos& pos) { return sum + pos.duration; }) / (double)framePosList.size();
	//durationから求めた平均fpsを計算する
	AVRational estimatedAvgFps = { 0 };
	estimatedAvgFps.num = demux.pCodecCtx->pkt_timebase.den;
	if (abs(avgDuration / durationHistgram[0].first - 1.0) < 5e-4) {
		estimatedAvgFps.den = demux.pCodecCtx->pkt_timebase.num * durationHistgram[0].first;
	} else {
		estimatedAvgFps.den = demux.pCodecCtx->pkt_timebase.num * (int)(avgDuration + 0.5);
	}

	//TimeStampベースで最初のフレームに戻す
	if (0 <= av_seek_frame(demux.pFormatCtx, demux.videoIndex, demux.videoStreamFirstPts, AVSEEK_FLAG_BACKWARD)) {
		addVideoPtsToList(firstKeyframePos);
		double dFpsDecoder = fpsDecoder.num / (double)fpsDecoder.den;
		double dEstimatedAvgFps = estimatedAvgFps.num / (double)estimatedAvgFps.den;
		double dEstimatedAvgFpsCompare = estimatedAvgFps.num / (double)(estimatedAvgFps.den + ((dFpsDecoder < dEstimatedAvgFps) ? 1 : -1));
		//durationから求めた平均fpsがデコーダの出したfpsの近似値と分かれば、デコーダの出したfpsを採用する
		demux.videoAvgFramerate = (abs(dEstimatedAvgFps - dFpsDecoder) < abs(dEstimatedAvgFpsCompare - dFpsDecoder)) ? fpsDecoder : estimatedAvgFps;
	} else {
		//失敗したら、Byte単位でのシークを試み、最初に戻す
		demux.videoStreamPtsInvalid = true;
		demux.videoAvgFramerate = fpsDecoder;
		if (0 <= av_seek_frame(demux.pFormatCtx, demux.videoIndex, 0, AVSEEK_FLAG_BACKWARD|AVSEEK_FLAG_BYTE)) {
			//ptsとdurationをpkt_timebaseで適当に作成する
			addVideoPtsToList({ 0, (int)av_rescale_q(1, demux.pCodecCtx->time_base, demux.pCodecCtx->pkt_timebase) });
		} else {
			m_strInputInfo += _T("avcodec: failed to seek backward.\n");
			return MFX_ERR_UNSUPPORTED;
		}
	}
	return MFX_ERR_NONE;
}

#pragma warning(push)
#pragma warning(disable:4100)
mfxStatus CAvcodecReader::Init(const TCHAR *strFileName, mfxU32 ColorFormat, const void *option, CEncodingThread *pEncThread, CEncodeStatusInfo *pEncSatusInfo, sInputCrop *pInputCrop) {
	if (!check_avcodec_dll()) {
		m_strInputInfo += error_mes_avcodec_dll_not_found();
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
	av_log_set_level(QSV_AV_LOG_LEVEL);
	InitializeCriticalSection(&demux.videoFrameData.cs);
	demux.videoFrameData.cs_initialized = true;

	const AvcodecReaderPrm *input_prm = (const AvcodecReaderPrm *)option;
	
	std::string filename_char;
	if (0 == tchar_to_string(strFileName, filename_char)) {
		m_strInputInfo += _T("avcodec reader: failed to convert filename to ansi characters.\n");
		return MFX_ERR_INVALID_HANDLE;
	}
	demux.pFormatCtx = avformat_alloc_context();
	//if (av_opt_set_int(demux.pFormatCtx, "probesize", 60 * AV_TIME_BASE, 0)) {
	//	AVDEBUG_PRINT("avcodec reader: faield to set probesize.\n");
	//}
	if (avformat_open_input(&(demux.pFormatCtx), filename_char.c_str(), nullptr, nullptr)) {
		m_strInputInfo += _T("avcodec reader: error opening file.\n");
		return MFX_ERR_NULL_PTR; // Couldn't open file
	}

	AVDEBUG_PRINT("avcodec reader: opened file.\n");
	//if (av_opt_set_int(demux.pFormatCtx, "analyzeduration", 60 * AV_TIME_BASE, 0)) {
	//	AVDEBUG_PRINT("avcodec reader: faield to set analyzeduration.\n");
	//}
	if (avformat_find_stream_info(demux.pFormatCtx, nullptr) < 0) {
		m_strInputInfo += _T("avcodec reader: error finding stream information.\n");
		return MFX_ERR_NULL_PTR; // Couldn't find stream information
	}
	AVDEBUG_PRINT("avcodec reader: got stream information.\n");
	//dump_format(dec.pFormatCtx, 0, argv[1], 0);
	
	//動画ストリームを探す
	if (-1 == (demux.videoIndex = getVideoStream())) {
		m_strInputInfo += _T("avcodec reader: unable to find video stream.\n");
		return MFX_ERR_NULL_PTR; // Didn't find a video stream
	}
	AVDEBUG_PRINT("avcodec reader: found video stream.\n");

	demux.pCodecCtx = demux.pFormatCtx->streams[demux.videoIndex]->codec;

	//QSVでデコード可能かチェック
	if (0 == (m_nInputCodec = getQSVFourcc(demux.pCodecCtx->codec_id))) {
		m_strInputInfo += _T("avcodec reader: codec ");
		m_strInputInfo += char_to_tstring(demux.pCodecCtx->codec->name);
		m_strInputInfo += _T(" unable to decode by qsv.\n");
		return MFX_ERR_NULL_PTR;
	}
	AVDEBUG_PRINT("avcodec reader: can be decoded by qsv.\n");

	//必要ならbitstream filterを初期化
	if ((m_nInputCodec == MFX_CODEC_AVC || m_nInputCodec == MFX_CODEC_HEVC) && demux.pCodecCtx->extradata && demux.pCodecCtx->extradata[0] == 1) {
		if (NULL == (demux.bsfc = av_bitstream_filter_init("h264_mp4toannexb"))) {
			m_strInputInfo += _T("avcodec reader: unable to init h264_mp4toannexb.\n");
			return MFX_ERR_NULL_PTR;
		}
		AVDEBUG_PRINT("avcodec reader: success to init h264_mp4toannexb.\n");
	}
	AVDEBUG_PRINT("avcodec reader: start demuxing... \n");
	
	mfxStatus decHeaderSts = MFX_ERR_NONE;
	mfxBitstream bitstream = { 0 };
	if (MFX_ERR_NONE != (decHeaderSts = GetHeader(&bitstream))) {
		m_strInputInfo += _T("avcodec reader: failed to get header.\n");
		return decHeaderSts;
	}
	
	if (m_nInputCodec == MFX_CODEC_AVC || m_nInputCodec == MFX_CODEC_HEVC) {
		//これを付加しないとMFXVideoDECODE_DecodeHeaderが成功しない
		const mfxU32 IDR = 0x65010000;
		AppendMfxBitstream(&bitstream, (mfxU8 *)&IDR, sizeof(IDR));
	}

	mfxSession session = { 0 };
	mfxVersion version = MFX_LIB_VERSION_1_1;
	if (MFX_ERR_NONE != (decHeaderSts = MFXInit(MFX_IMPL_HARDWARE_ANY, &version, &session))) {
		m_strInputInfo += _T("avcodec reader: unable to init qsv decoder.\n");
		return decHeaderSts;
	}

	std::unique_ptr<MFXPlugin> pPlugin;
	if (m_nInputCodec == MFX_CODEC_HEVC) {
		pPlugin.reset(LoadPlugin(MFX_PLUGINTYPE_VIDEO_DECODE, session, MFX_PLUGINID_HEVCD_HW, 1));
		if (pPlugin.get() == NULL) {
			m_strInputInfo += _T("avcodec reader: failed to load hw hevc decoder.\n");
			return MFX_ERR_UNSUPPORTED;
		}
	}
	
	memset(&m_sDecParam, 0, sizeof(m_sDecParam));
	m_sDecParam.mfx.CodecId = m_nInputCodec;
	m_sDecParam.IOPattern = (mfxU16)((input_prm->memType != SYSTEM_MEMORY) ? MFX_IOPATTERN_OUT_VIDEO_MEMORY : MFX_IOPATTERN_OUT_SYSTEM_MEMORY);
	if (MFX_ERR_NONE != (decHeaderSts = MFXVideoDECODE_DecodeHeader(session, &bitstream, &m_sDecParam))) {
		m_strInputInfo += _T("avcodec reader: failed to decode header.\n");
	} else if (MFX_ERR_NONE != (decHeaderSts = getFirstFramePosAndFrameRate({ m_sDecParam.mfx.FrameInfo.FrameRateExtN, m_sDecParam.mfx.FrameInfo.FrameRateExtD }))) {
		m_strInputInfo += _T("avcodec reader: failed to get first frame position.\n");
	}
	pPlugin.reset(); //必ずsessionをクローズする前に開放すること
	MFXClose(session);
	if (MFX_ERR_NONE != decHeaderSts) {
		m_strInputInfo += _T("avcodec reader: unable to decode by qsv, please consider using other input method.\n");
		return decHeaderSts;
	}
	WipeMfxBitstream(&bitstream);

	m_sTrimParam.list = std::vector<sTrim>(input_prm->pTrimList, input_prm->pTrimList + input_prm->nTrimCount);
	//キーフレームに到達するまでQSVではフレームが出てこない
	//そのぶんのずれを記録しておき、Trim値などに補正をかける
	if (m_sTrimParam.offset) {
		for (int i = (int)m_sTrimParam.list.size() - 1; i >= 0; i--) {
			if (m_sTrimParam.list[i].fin - m_sTrimParam.offset < 0) {
				m_sTrimParam.list.erase(m_sTrimParam.list.begin() + i);
			} else {
				m_sTrimParam.list[i].start = (std::max)(0, m_sTrimParam.list[i].start - m_sTrimParam.offset);
				m_sTrimParam.list[i].fin   = (std::max)(0, m_sTrimParam.list[i].fin   - m_sTrimParam.offset);
			}
		}
	}

	//getFirstFramePosAndFrameRateをもとにfpsを決定
	m_sDecParam.mfx.FrameInfo.FrameRateExtN = demux.videoAvgFramerate.num;
	m_sDecParam.mfx.FrameInfo.FrameRateExtD = demux.videoAvgFramerate.den;
	const mfxU32 fps_gcd = GCD(m_sDecParam.mfx.FrameInfo.FrameRateExtN, m_sDecParam.mfx.FrameInfo.FrameRateExtD);
	m_sDecParam.mfx.FrameInfo.FrameRateExtN /= fps_gcd;
	m_sDecParam.mfx.FrameInfo.FrameRateExtD /= fps_gcd;

	//近似値であれば、分母1001に合わせる
	if (m_sDecParam.mfx.FrameInfo.FrameRateExtD != 1001) {
		double fps = m_sDecParam.mfx.FrameInfo.FrameRateExtN / (double)m_sDecParam.mfx.FrameInfo.FrameRateExtD;
		double fps_n = fps * 1001;
		int fps_n_int = (int)(fps + 0.5) * 1000;
		if (abs(fps_n / (double)fps_n_int - 1.0) < 1e-4) {
			m_sDecParam.mfx.FrameInfo.FrameRateExtN = fps_n_int;
			m_sDecParam.mfx.FrameInfo.FrameRateExtD = 1001;
		}
	}

	//音声ストリームを探す
	if (input_prm->bReadAudio && -1 != (demux.audioIndex = getAudioStream())) {
		demux.pCodecCtxAudio = demux.pFormatCtx->streams[demux.audioIndex]->codec;
	}

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
	_stprintf_s(mes, _countof(mes), _T("avcodec video: %s, %dx%d, %d/%d fps"), CodecIdToStr(m_nInputCodec).c_str(),
		m_inputFrameInfo.Width, m_inputFrameInfo.Height, m_inputFrameInfo.FrameRateExtN, m_inputFrameInfo.FrameRateExtD);
	m_strInputInfo += mes;

	m_tmLastUpdate = timeGetTime();
	return MFX_ERR_NONE;
}
#pragma warning(pop)

int CAvcodecReader::getVideoFrameIdx(mfxI64 pts, AVRational timebase, int i_start) {
	const int frame_n = demux.videoFrameData.num;
	for (int i = max(0, i_start); i < frame_n; i++) {
		//pts < demux.videoFramePts[i]であるなら、その前のフレームを返す
		if (0 > av_compare_ts(pts, timebase, demux.videoFrameData.frame[i].pts, demux.pCodecCtx->pkt_timebase)) {
			return i - 1;
		}
	}
	return frame_n;
}

mfxI64 CAvcodecReader::convertTimebaseVidToAud(mfxI64 pts) {
	return av_rescale_q(pts, demux.pCodecCtx->pkt_timebase, demux.pCodecCtxAudio->pkt_timebase);
}

bool CAvcodecReader::checkAudioPacketToAdd(const AVPacket *pkt) {
	demux.lastVidIndex = getVideoFrameIdx(pkt->pts, demux.pCodecCtxAudio->pkt_timebase, demux.lastVidIndex);

	//該当フレームが-1フレーム未満なら、その音声はこの動画には含まれない
	if (demux.lastVidIndex < -1) {
		return false;
	}

	const FramePos *vidFramePos = &demux.videoFrameData.frame[max(demux.lastVidIndex, 0)];
	const mfxI64 vid_fin = convertTimebaseVidToAud(vidFramePos->pts + ((demux.lastVidIndex >= 0) ? vidFramePos->duration : 0));

	const mfxI64 aud_start = pkt->pts;
	const mfxI64 aud_fin   = pkt->pts + pkt->duration;

	const bool frame_is_in_range = frame_inside_range(demux.lastVidIndex,     m_sTrimParam.list);
	const bool next_is_in_range  = frame_inside_range(demux.lastVidIndex + 1, m_sTrimParam.list);

	bool result = true; //動画に含まれる音声かどうか

	if (frame_is_in_range) {
		if (aud_fin < vid_fin || next_is_in_range) {
			; //完全に動画フレームの範囲内か、次のフレームも範囲内なら、その音声パケットは含まれる
		//              vid_fin
		//動画 <-----------|
		//音声      |-----------|
		//     aud_start     aud_fin
		} else if (pkt->duration / 2 > (aud_fin - vid_fin + demux.audExtractErrExcess)) {
			//はみ出した領域が少ないなら、その音声パケットは含まれる
			demux.audExtractErrExcess += aud_fin - vid_fin;
		} else {
			//はみ出した領域が多いなら、その音声パケットは含まれない
			demux.audExtractErrExcess -= vid_fin - aud_start;
			result = false;
		}
	} else if (next_is_in_range && aud_fin > vid_fin) {
		//             vid_fin
		//動画             |------------>
		//音声      |-----------|
		//     aud_start     aud_fin
		if (pkt->duration / 2 > (vid_fin - aud_start + demux.audExtractErrExcess)) {
			demux.audExtractErrExcess += vid_fin - aud_start;
		} else {
			demux.audExtractErrExcess -= aud_fin - vid_fin;
			result = false;
		}
	} else {
		result = false;
	}
	return result;
}

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
			//最初のptsが格納されていたら( = getFirstFramePosAndFrameRate()が実行済み)、後続のptsを格納していく
			if (demux.videoFrameData.num) {
				//最初のキーフレームを取得するまではスキップする
				if (!demux.videoGotFirstKeyframe && !(pkt->flags & AV_PKT_FLAG_KEY)) {
					av_free_packet(pkt);
					continue;
				} else {
					demux.videoGotFirstKeyframe = true;
					//AVPacketのもたらすptsが無効であれば、CFRを仮定して適当にptsとdurationを突っ込んでいく
					//0フレーム目は格納されているので、その次からを格納する
					if (demux.videoStreamPtsInvalid && demux.sampleLoadCount) {
						int duration = demux.videoFrameData.frame[0].duration;
						addVideoPtsToList({ demux.sampleLoadCount * duration, duration });
					//最初のptsは格納されているので、その次からを格納する
					} else {
						addVideoPtsToList({ pkt->pts, pkt->duration });
					}
				}
			}
			return 0;
		}
		if (pkt->stream_index == demux.audioIndex) {
			//音声パケットはひとまずすべてバッファに格納する
			m_AudioPacketsBufferL1[demux.sampleLoadCount % _countof(demux.videoPacket)].push_back(*pkt);
		} else {
			av_free_packet(pkt);
		}
	}
	//ファイルの終わりに到達
	pkt->data = nullptr;
	pkt->size = 0;
	sortVideoPtsList();
	demux.videoFrameData.fixed_num = demux.videoFrameData.num - 1;
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
	mfxStatus sts = setToMfxBitstream(bitstream, &demux.videoPacket[demux.sampleGetCount % _countof(demux.videoPacket)]);
	demux.sampleGetCount++;
	return sts;
}

std::vector<AVPacket> CAvcodecReader::GetAudioDataPackets() {
	//すでに使用した音声バッファはクリアする
	for (mfxU32 i = 0; i < m_AudioPacketsBufferL2Used; i++) {
		av_free_packet(&m_AudioPacketsBufferL2[i]);
	}
	if (m_AudioPacketsBufferL2Used) {
		m_AudioPacketsBufferL2.erase(m_AudioPacketsBufferL2.begin(), m_AudioPacketsBufferL2.begin() + m_AudioPacketsBufferL2Used);
	}
	m_AudioPacketsBufferL2Used = 0;

	//別スレッドで使用されていないほうを連結する
	const auto& packetsL1 = m_AudioPacketsBufferL1[demux.sampleGetCount % _countof(m_AudioPacketsBufferL1)];
	m_AudioPacketsBufferL2.insert(m_AudioPacketsBufferL2.end(), packetsL1.begin(), packetsL1.end());

	//出力するパケットを選択する
	std::vector<AVPacket> packets;
	EnterCriticalSection(&demux.videoFrameData.cs);
	for (const auto& pkt : m_AudioPacketsBufferL2) {
		//音声のptsが映像の終わりのptsを行きすぎたらやめる
		if (0 < av_compare_ts(pkt.pts, demux.pCodecCtxAudio->pkt_timebase, demux.videoFrameData.frame[demux.videoFrameData.fixed_num].pts, demux.pCodecCtx->pkt_timebase)) {
			break;
		}
		m_AudioPacketsBufferL2Used++;
		if (checkAudioPacketToAdd(&pkt)) {
			packets.push_back(pkt);
		}
	}
	LeaveCriticalSection(&demux.videoFrameData.cs);
	return std::move(packets);
}

const AVCodecContext *CAvcodecReader::GetAudioCodecCtx() {
	return demux.pCodecCtxAudio;
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
	AVPacket *pkt = &demux.videoPacket[demux.sampleLoadCount % _countof(demux.videoPacket)];
	m_AudioPacketsBufferL1[demux.sampleLoadCount % _countof(m_AudioPacketsBufferL1)].clear();

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
