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
#if USE_CUSTOM_IO
static int funcReadPacket(void *opaque, uint8_t *buf, int buf_size) {
	CAvcodecWriter *writer = reinterpret_cast<CAvcodecWriter *>(opaque);
	return writer->readPacket(buf, buf_size);
}
static int funcWritePacket(void *opaque, uint8_t *buf, int buf_size) {
	CAvcodecWriter *writer = reinterpret_cast<CAvcodecWriter *>(opaque);
	return writer->writePacket(buf, buf_size);
}
static int64_t funcSeek(void *opaque, int64_t offset, int whence) {
	CAvcodecWriter *writer = reinterpret_cast<CAvcodecWriter *>(opaque);
	return writer->seek(offset, whence);
}
#endif //USE_CUSTOM_IO

CAvcodecWriter::CAvcodecWriter() {
	MSDK_ZERO_MEMORY(m_Muxer);
}

CAvcodecWriter::~CAvcodecWriter() {

}

void CAvcodecWriter::Close() {
	//close decoder
	if (m_Muxer.pAudioOutCodecDecodeCtx) {
		avcodec_close(m_Muxer.pAudioOutCodecDecodeCtx);
		av_free(m_Muxer.pAudioOutCodecDecodeCtx);
	}

	//close encoder
	if (m_Muxer.pAudioOutCodecEncodeCtx) {
		avcodec_close(m_Muxer.pAudioOutCodecEncodeCtx);
		av_free(m_Muxer.pAudioOutCodecEncodeCtx);
	}

	//free packet
	if (m_Muxer.audioOutPacket.data) {
		av_free_packet(&m_Muxer.audioOutPacket);
	}

	//close audio file
	AVIOContext *pAVIOContext = m_Muxer.pFormatCtx->pb;
	if (m_Muxer.pFormatCtx) {
		if (!m_Muxer.bStreamError) {
			av_write_trailer(m_Muxer.pFormatCtx);
		}
		avformat_free_context(m_Muxer.pFormatCtx);
	}
	if (m_Muxer.pStreamAudio) {
		av_free(m_Muxer.pStreamAudio);
	}
	if (pAVIOContext) {
		avio_close(pAVIOContext);
		av_free(pAVIOContext);
	}
#if USE_CUSTOM_IO
	if (m_Muxer.fpOutput) {
		fflush(m_Muxer.fpOutput);
		fclose(m_Muxer.fpOutput);
	}

	if (m_Muxer.pAVOutBuffer) {
		av_free(m_Muxer.pAVOutBuffer);
	}

	if (m_Muxer.pOutputBuffer) {
		free(m_Muxer.pOutputBuffer);
	}
#endif //USE_CUSTOM_IO

	m_strOutputInfo.clear();

	MSDK_ZERO_MEMORY(m_Muxer);
}

AVCodecID CAvcodecWriter::PCMRequiresConversion(const AVCodecContext *audioCtx) {
	static const std::pair<AVCodecID, AVCodecID> pcmConvertCodecs[] = {
		{ AV_CODEC_ID_FIRST_AUDIO,      AV_CODEC_ID_FIRST_AUDIO },
		{ AV_CODEC_ID_PCM_DVD,          AV_CODEC_ID_FIRST_AUDIO },
		{ AV_CODEC_ID_PCM_BLURAY,       AV_CODEC_ID_FIRST_AUDIO },
		{ AV_CODEC_ID_PCM_S8_PLANAR,    AV_CODEC_ID_PCM_S8      },
		{ AV_CODEC_ID_PCM_S16LE_PLANAR, AV_CODEC_ID_PCM_S16LE   },
		{ AV_CODEC_ID_PCM_S16BE_PLANAR, AV_CODEC_ID_PCM_S16LE   },
		{ AV_CODEC_ID_PCM_S16BE,        AV_CODEC_ID_PCM_S16LE   },
		{ AV_CODEC_ID_PCM_S24LE_PLANAR, AV_CODEC_ID_PCM_S24LE   },
		{ AV_CODEC_ID_PCM_S24BE,        AV_CODEC_ID_PCM_S24LE   },
		{ AV_CODEC_ID_PCM_S32LE_PLANAR, AV_CODEC_ID_PCM_S32LE   },
		{ AV_CODEC_ID_PCM_S32BE,        AV_CODEC_ID_PCM_S32LE   },
		{ AV_CODEC_ID_PCM_F32BE,        AV_CODEC_ID_PCM_S32LE   },
		{ AV_CODEC_ID_PCM_F64BE,        AV_CODEC_ID_PCM_S32LE   },
	};
	for (int i = 0; i < _countof(pcmConvertCodecs); i++) {
		if (pcmConvertCodecs[i].first == audioCtx->codec_id) {
			if (pcmConvertCodecs[i].second != AV_CODEC_ID_FIRST_AUDIO) {
				return pcmConvertCodecs[i].second;
			}
			switch (audioCtx->bits_per_raw_sample) {
			case 32: return AV_CODEC_ID_PCM_S32LE;
			case 24: return AV_CODEC_ID_PCM_S24LE;
			case 8:  return AV_CODEC_ID_PCM_S16LE;
			case 16:
			default: return AV_CODEC_ID_PCM_S16LE;
			}
		}
	}
	return AV_CODEC_ID_NONE;
}

mfxStatus CAvcodecWriter::Init(const msdk_char *strFileName, const void *option, CEncodeStatusInfo *pEncSatusInfo) {
	if (!check_avcodec_dll()) {
		m_strOutputInfo += error_mes_avcodec_dll_not_found();
		return MFX_ERR_NULL_PTR;
	}
	
	m_Muxer.bStreamError = true;
	const AvcodecWriterPrm *prm = (const AvcodecWriterPrm *)option;
	m_Muxer.pktTimebase = prm->pCodecCtxAudioIn->pkt_timebase;

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

#if USE_CUSTOM_IO
	m_Muxer.nAVOutBufferSize = 1024 * 1024;
	m_Muxer.nOutputBufferSize = 16 * 1024 * 1024;

	if (NULL == (m_Muxer.pAVOutBuffer = (mfxU8 *)av_malloc(m_Muxer.nAVOutBufferSize))) {
		m_strOutputInfo += _T("avcodec writer: failed to allocate muxer buffer.\n");
		return MFX_ERR_NULL_PTR;
	}

	if (fopen_s(&m_Muxer.fpOutput, filename.c_str(), "wb")) {
		m_strOutputInfo += _T("avcodec writer: failed to open audio output file.\n");
		return MFX_ERR_NULL_PTR; // Couldn't open file
	}
	if (NULL != (m_Muxer.pOutputBuffer = (char *)malloc(m_Muxer.nOutputBufferSize))) {
		setvbuf(m_Muxer.fpOutput, m_Muxer.pOutputBuffer, _IOFBF, m_Muxer.nOutputBufferSize);
	}

	if (NULL == (m_Muxer.pFormatCtx->pb = avio_alloc_context(m_Muxer.pAVOutBuffer, m_Muxer.nAVOutBufferSize, 1, this, funcReadPacket, funcWritePacket, funcSeek))) {
		m_strOutputInfo += _T("avcodec writer: failed to alloc avio context.\n");
		return MFX_ERR_NULL_PTR;
	}
#else
	if (0 > avio_open2(&m_Muxer.pFormatCtx->pb, filename.c_str(), AVIO_FLAG_WRITE, NULL, NULL)) {
		m_strOutputInfo += _T("avcodec writer: failed to open audio output file.\n");
		return MFX_ERR_NULL_PTR; // Couldn't open file
	}
#endif
	if (NULL == (m_Muxer.pStreamAudio = avformat_new_stream(m_Muxer.pFormatCtx, NULL))) {
		m_strOutputInfo += _T("avcodec writer: failed to create new stream for audio.\n");
		return MFX_ERR_NULL_PTR;
	}

	//音声がwavの場合、フォーマット変換が必要な場合がある
	AVCodecID codecId = AV_CODEC_ID_NONE;
	if (AV_CODEC_ID_NONE != (codecId = PCMRequiresConversion(prm->pCodecCtxAudioIn))) {
		auto error_mes =[](const TCHAR *mes, AVCodecID targetCodec) {
			return mes + tstring(_T(" for ")) + char_to_tstring(avcodec_get_name(targetCodec)).c_str() + tstring(_T(".\n"));
		};
		//PCM decoder
		if (NULL == (m_Muxer.pAudioOutCodecDecode = avcodec_find_decoder(prm->pCodecCtxAudioIn->codec_id))) {
			m_strOutputInfo += error_mes(_T("avcodec writer: failed to find decoder"), prm->pCodecCtxAudioIn->codec_id);
			return MFX_ERR_NULL_PTR;
		}
		if (NULL == (m_Muxer.pAudioOutCodecDecodeCtx = avcodec_alloc_context3(m_Muxer.pAudioOutCodecDecode))) {
			m_strOutputInfo += error_mes( _T("avcodec writer: failed to get decode codec context"), prm->pCodecCtxAudioIn->codec_id);
			return MFX_ERR_NULL_PTR;
		}
#define COPY_IF_ZERO(dst, src) { if ((dst)==0) (dst)=(src); }
		COPY_IF_ZERO(m_Muxer.pAudioOutCodecDecodeCtx->sample_rate,    prm->pCodecCtxAudioIn->sample_rate);
		COPY_IF_ZERO(m_Muxer.pAudioOutCodecDecodeCtx->channels,       prm->pCodecCtxAudioIn->channels);
		COPY_IF_ZERO(m_Muxer.pAudioOutCodecDecodeCtx->channel_layout, prm->pCodecCtxAudioIn->channel_layout);
#undef COPY_IF_ZERO
		if (0 > avcodec_open2(m_Muxer.pAudioOutCodecDecodeCtx, m_Muxer.pAudioOutCodecDecode, NULL)) {
			m_strOutputInfo += error_mes( _T("avcodec writer: failed to open decoder"), prm->pCodecCtxAudioIn->codec_id);
			return MFX_ERR_NULL_PTR;
		}
		av_new_packet(&m_Muxer.audioOutPacket, 512 * 1024);
		m_Muxer.audioOutPacket.size = 0;

		//PCM encoder
		if (NULL == (m_Muxer.pAudioOutCodecEncode = avcodec_find_encoder(codecId))) {
			m_strOutputInfo += error_mes( _T("avcodec writer: failed to find encoder"), codecId);
			return MFX_ERR_NULL_PTR;
		}
		if (NULL == (m_Muxer.pAudioOutCodecEncodeCtx = avcodec_alloc_context3(m_Muxer.pAudioOutCodecEncode))) {
			m_strOutputInfo += error_mes( _T("avcodec writer: failed to get encode codec context"), codecId);
			return MFX_ERR_NULL_PTR;
		}
		m_Muxer.pAudioOutCodecEncodeCtx->sample_fmt          = prm->pCodecCtxAudioIn->sample_fmt;
		m_Muxer.pAudioOutCodecEncodeCtx->sample_rate         = prm->pCodecCtxAudioIn->sample_rate;
		m_Muxer.pAudioOutCodecEncodeCtx->channels            = prm->pCodecCtxAudioIn->channels;
		m_Muxer.pAudioOutCodecEncodeCtx->channel_layout      = prm->pCodecCtxAudioIn->channel_layout;
		m_Muxer.pAudioOutCodecEncodeCtx->bits_per_raw_sample = prm->pCodecCtxAudioIn->bits_per_raw_sample;
		if (0 > avcodec_open2(m_Muxer.pAudioOutCodecEncodeCtx, m_Muxer.pAudioOutCodecEncode, NULL)) {
			m_strOutputInfo += error_mes( _T("avcodec writer: failed to open encoder"), codecId);
			return MFX_ERR_NULL_PTR;
		}
	}

	//パラメータのコピー
	avcodec_copy_context(m_Muxer.pStreamAudio->codec, (m_Muxer.pAudioOutCodecEncodeCtx) ? m_Muxer.pAudioOutCodecEncodeCtx : prm->pCodecCtxAudioIn);
	sprintf_s(m_Muxer.pFormatCtx->filename, filename.c_str());
	m_Muxer.pStreamAudio->time_base = av_make_q(1, m_Muxer.pStreamAudio->codec->sample_rate);
	m_Muxer.pStreamAudio->codec->time_base = m_Muxer.pStreamAudio->time_base;

	if (m_Muxer.pOutputFmt->flags & AVFMT_GLOBALHEADER)
		m_Muxer.pStreamAudio->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;

	int ret = 0;
	if (0 > (ret = avformat_write_header(m_Muxer.pFormatCtx, NULL))) {
		m_strOutputInfo += _T("avcodec writer: failed to write header for audio. :");
		m_strOutputInfo += qsv_av_err2str(ret) + tstring(_T("\n"));
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
	m_Muxer.bStreamError = false;

	return MFX_ERR_NONE;
}

mfxStatus CAvcodecWriter::WriteNextFrame(AVPacket *pkt) {
	m_Muxer.nPacketWritten++;
	
	AVPacket encodePkt = { 0 };
	int duration = 0;
	BOOL got_result = TRUE;
	if (!m_Muxer.pAudioOutCodecDecodeCtx) {
		//durationについて、パケットのtimebaseから出力ストリームのtimebaseに変更する
		duration = (int)av_rescale_q(pkt->duration, m_Muxer.pktTimebase, m_Muxer.pStreamAudio->time_base);
	} else {
		AVFrame *decodedFrame = av_frame_alloc();
		AVPacket *pktIn = pkt;
		if (m_Muxer.audioOutPacket.size != 0) {
			int currentSize = m_Muxer.audioOutPacket.size;
			if (m_Muxer.audioOutPacket.buf->size < currentSize + pkt->size) {
				av_grow_packet(&m_Muxer.audioOutPacket, currentSize + pkt->size);
				m_Muxer.audioOutPacket.size = currentSize;
			}
			memcpy(m_Muxer.audioOutPacket.data, pkt->data, pkt->size);
			m_Muxer.audioOutPacket.size += pkt->size;
			pktIn = &m_Muxer.audioOutPacket;
		}
		//PCM decode
		int len = 0;
		if (0 > (len = avcodec_decode_audio4(m_Muxer.pAudioOutCodecDecodeCtx, decodedFrame, &got_result, pkt))) {
			m_strOutputInfo += _T("avcodec writer: failed to convert pcm format(1). :");
			m_strOutputInfo += qsv_av_err2str(len) + tstring(_T("\n"));
			m_Muxer.bStreamError = true;
		} else if (pkt->size != len) {
			int newLen = pkt->size - len;
			memmove(m_Muxer.audioOutPacket.data, pkt->data + len, newLen);
			m_Muxer.audioOutPacket.size = newLen;
		} else {
			m_Muxer.audioOutPacket.size = 0;
		}
		if (got_result) {
			//PCM encode
			av_init_packet(&encodePkt);
			int ret = avcodec_encode_audio2(m_Muxer.pAudioOutCodecEncodeCtx, &encodePkt, decodedFrame, &got_result);
			if (ret < 0) {
				m_strOutputInfo += _T("avcodec writer: failed to convert pcm format(2). :");
				m_strOutputInfo += qsv_av_err2str(ret) + tstring(_T("\n"));
				m_Muxer.bStreamError = true;
			} else if (got_result) {
				duration = encodePkt.duration;
				pkt = &encodePkt;
			}
		}
		if (decodedFrame) {
			av_frame_free(&decodedFrame);
		}
	}

	if (duration) {
		pkt->stream_index  = m_Muxer.pStreamAudio->id;
		pkt->duration      = duration;
		pkt->dts           = m_Muxer.nLastPktDtsAudio;
		pkt->pts           = m_Muxer.nLastPktDtsAudio;
		m_Muxer.bStreamError = 0 != av_write_frame(m_Muxer.pFormatCtx, pkt);
		m_Muxer.nLastPktDtsAudio += duration;
	}
	if (encodePkt.data) {
		av_free_packet(&encodePkt);
	}
	return (m_Muxer.bStreamError) ? MFX_ERR_UNKNOWN : MFX_ERR_NONE;
}

#if USE_CUSTOM_IO
int CAvcodecWriter::readPacket(uint8_t *buf, int buf_size) {
	return (int)fread(buf, 1, buf_size, m_Muxer.fpOutput);
}
int CAvcodecWriter::writePacket(uint8_t *buf, int buf_size) {
	return (int)fwrite(buf, 1, buf_size, m_Muxer.fpOutput);
}
int64_t CAvcodecWriter::seek(int64_t offset, int whence) {
	return _fseeki64(m_Muxer.fpOutput, offset, whence);
}
#endif //USE_CUSTOM_IO

#endif //ENABLE_AVCODEC_QSV_READER
