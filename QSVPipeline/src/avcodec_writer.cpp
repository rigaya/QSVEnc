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
#include <memory>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include "qsv_util.h"
#include "avcodec_writer.h"

#if ENABLE_AVCODEC_QSV_READER

CAvcodecWriter::CAvcodecWriter() {
	MSDK_ZERO_MEMORY(m_Muxer);
}

CAvcodecWriter::~CAvcodecWriter() {

}

void CAvcodecWriter::Close() {
	//close audio file
	if (m_Muxer.pFormatCtx) {
		av_write_trailer(m_Muxer.pFormatCtx);
		avio_close(m_Muxer.pFormatCtx->pb);
		avformat_free_context(m_Muxer.pFormatCtx);
	}

	m_strOutputInfo.clear();

	MSDK_ZERO_MEMORY(m_Muxer);
}

mfxStatus CAvcodecWriter::Init(const msdk_char *strFileName, const void *option, CEncodeStatusInfo *pEncSatusInfo) {
	if (!check_avcodec_dll()) {
		m_strOutputInfo += error_mes_avcodec_dll_not_found();
		return MFX_ERR_NULL_PTR;
	}
	
	const AvcodecWriterPrm *prm = (const AvcodecWriterPrm *)option;

	std::string filename;
	if (0 == tchar_to_string(strFileName, filename)) {
		m_strOutputInfo += _T("avcodec writer: failed to convert audio filename to ansi characters.\n");
		return MFX_ERR_NULL_PTR;
	}

	av_register_all();
	avcodec_register_all();
	av_log_set_level(QSV_AV_LOG_LEVEL);

	if (NULL == (m_Muxer.pOutputFmt = av_guess_format(NULL, filename.c_str(), NULL))) {
		m_strOutputInfo += _T("avcodec writer: failed to assume format from audio filename.\n");
		m_strOutputInfo += _T("                please set proper extension for audio file.\n");
		return MFX_ERR_NULL_PTR;
	}
	m_Muxer.pFormatCtx = avformat_alloc_context();
	m_Muxer.pFormatCtx->oformat = m_Muxer.pOutputFmt;

	if (0 > avio_open2(&m_Muxer.pFormatCtx->pb, filename.c_str(), AVIO_FLAG_WRITE, NULL, NULL)) {
		m_strOutputInfo += _T("avcodec writer: failed to open audio output file.\n");
		return MFX_ERR_NULL_PTR; // Couldn't open file
	}

	if (NULL == (m_Muxer.pStreamAudio = avformat_new_stream(m_Muxer.pFormatCtx, NULL))) {
		m_strOutputInfo += _T("avcodec writer: failed to create new stream for audio.\n");
		return MFX_ERR_NULL_PTR;
	}

	//パラメータのコピー
	avcodec_copy_context(m_Muxer.pStreamAudio->codec, prm->pCodecCtxAudioIn);
	sprintf_s(m_Muxer.pFormatCtx->filename, filename.c_str());
	m_Muxer.pStreamAudio->time_base = av_make_q(1, m_Muxer.pStreamAudio->codec->sample_rate);
	m_Muxer.pStreamAudio->codec->time_base = m_Muxer.pStreamAudio->time_base;

	if (m_Muxer.pOutputFmt->flags & AVFMT_GLOBALHEADER)
		m_Muxer.pStreamAudio->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;

	int ret = 0;
	if (0 > (ret = avformat_write_header(m_Muxer.pFormatCtx, NULL))) {
		m_strOutputInfo += _T("avcodec writer: failed to write header for audio.\n");
		char buf[1024];
		av_make_error_string(buf, sizeof(buf), ret);
		m_strOutputInfo += _T("         ");
		m_strOutputInfo += char_to_tstring(buf);
		m_strOutputInfo += _T("\n");
		return MFX_ERR_UNKNOWN;
	}
	av_dump_format(m_Muxer.pFormatCtx, 0, m_Muxer.pFormatCtx->filename, 1);

	TCHAR mes[256];
	_stprintf_s(mes, _countof(mes), _T("avcodec audio: %s -> %s (%s)"),
		char_to_tstring(avcodec_get_name(m_Muxer.pStreamAudio->codec->codec_id)).c_str(),
		char_to_tstring(PathFindFileNameA(m_Muxer.pFormatCtx->filename)).c_str(),
		char_to_tstring(m_Muxer.pFormatCtx->oformat->name).c_str());
	m_strOutputInfo += mes;

	m_pEncSatusInfo = pEncSatusInfo;

	return MFX_ERR_NONE;
}

mfxStatus CAvcodecWriter::WriteNextFrame(AVPacket *pkt) {
	m_Muxer.nPacketWritten++;
	pkt->stream_index = m_Muxer.pStreamAudio->id;
	//durationについて、パケットのtimebaseから出力ストリームのtimebaseに変更する
	const int duration = (int)av_rescale_q(pkt->duration, m_Muxer.pStreamAudio->codec->pkt_timebase, m_Muxer.pStreamAudio->time_base);
	m_Muxer.nLastPktDtsAudio += duration;

	pkt->duration = duration;
	pkt->dts      = m_Muxer.nLastPktDtsAudio;
	pkt->pts      = m_Muxer.nLastPktDtsAudio;
	return 0 == av_write_frame(m_Muxer.pFormatCtx, pkt) ? MFX_ERR_NONE : MFX_ERR_UNKNOWN;
}

#endif //ENABLE_AVCODEC_QSV_READER
