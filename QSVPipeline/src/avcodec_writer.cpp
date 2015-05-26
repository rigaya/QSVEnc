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
	MSDK_ZERO_MEMORY(m_Mux.format);
	MSDK_ZERO_MEMORY(m_Mux.video);
	MSDK_ZERO_MEMORY(m_Mux.audio);
}

CAvcodecWriter::~CAvcodecWriter() {

}

void CAvcodecWriter::CloseAudio(AVMuxAudio *pMuxAudio) {
	//close decoder
	if (pMuxAudio->pOutCodecDecodeCtx) {
		avcodec_close(pMuxAudio->pOutCodecDecodeCtx);
		av_free(pMuxAudio->pOutCodecDecodeCtx);
	}

	//close encoder
	if (pMuxAudio->pOutCodecEncodeCtx) {
		avcodec_close(pMuxAudio->pOutCodecEncodeCtx);
		av_free(pMuxAudio->pOutCodecEncodeCtx);
	}

	//free packet
	if (pMuxAudio->OutPacket.data) {
		av_free_packet(&pMuxAudio->OutPacket);
	}
	if (pMuxAudio->pAACBsfc) {
		av_bitstream_filter_close(pMuxAudio->pAACBsfc);
	}
	if (pMuxAudio->pCodecCtxIn) {
		avcodec_free_context(&pMuxAudio->pCodecCtxIn);
	}
	memset(pMuxAudio, 0, sizeof(pMuxAudio[0]));
}

void CAvcodecWriter::CloseVideo(AVMuxVideo *pMuxVideo) {
	memset(pMuxVideo, 0, sizeof(pMuxVideo[0]));
}

void CAvcodecWriter::CloseFormat(AVMuxFormat *pMuxFormat) {
	if (pMuxFormat->pFormatCtx) {
		if (!pMuxFormat->bStreamError) {
			av_write_trailer(pMuxFormat->pFormatCtx);
		}
#if USE_CUSTOM_IO
		if (!pMuxFormat->fpOutput)
#endif
			avio_close(pMuxFormat->pFormatCtx->pb);
		avformat_free_context(pMuxFormat->pFormatCtx);
	}
#if USE_CUSTOM_IO
	if (pMuxFormat->fpOutput) {
		fflush(pMuxFormat->fpOutput);
		fclose(pMuxFormat->fpOutput);
	}

	if (pMuxFormat->pAVOutBuffer) {
		av_free(pMuxFormat->pAVOutBuffer);
	}

	if (pMuxFormat->pOutputBuffer) {
		free(pMuxFormat->pOutputBuffer);
	}
#endif //USE_CUSTOM_IO
	memset(pMuxFormat, 0, sizeof(pMuxFormat[0]));
}

void CAvcodecWriter::Close() {
	CloseFormat(&m_Mux.format);
	CloseAudio(&m_Mux.audio);
	CloseVideo(&m_Mux.video);

	m_strOutputInfo.clear();
}

tstring CAvcodecWriter::errorMesForCodec(const TCHAR *mes, AVCodecID targetCodec) {
	return mes + tstring(_T(" for ")) + char_to_tstring(avcodec_get_name(targetCodec)) + tstring(_T(".\n"));
};

AVCodecID CAvcodecWriter::getAVCodecId(mfxU32 QSVFourcc) {
	for (int i = 0; i < _countof(QSV_DECODE_LIST); i++)
		if (QSV_DECODE_LIST[i].qsv_fourcc == QSVFourcc)
			return (AVCodecID)QSV_DECODE_LIST[i].codec_id;
	return AV_CODEC_ID_NONE;
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

void CAvcodecWriter::SetExtraData(AVCodecContext *codecCtx, const mfxU8 *data, mfxU32 size) {
	if (codecCtx->extradata)
		av_free(codecCtx->extradata);
	codecCtx->extradata_size = size;
	codecCtx->extradata      = (uint8_t *)av_malloc(codecCtx->extradata_size);
	memcpy(codecCtx->extradata, data, size);
};

mfxStatus CAvcodecWriter::Init(const msdk_char *strFileName, const void *option, CEncodeStatusInfo *pEncSatusInfo) {
	if (!check_avcodec_dll()) {
		m_strOutputInfo += error_mes_avcodec_dll_not_found();
		return MFX_ERR_NULL_PTR;
	}
	
	m_Mux.format.bStreamError = true;
	const AvcodecWriterPrm *prm = (const AvcodecWriterPrm *)option;

	std::string filename;
	if (0 == tchar_to_string(strFileName, filename)) {
		m_strOutputInfo += _T("avcodec writer: failed to convert audio filename to ansi characters.\n");
		return MFX_ERR_NULL_PTR;
	}

	av_register_all();
	avcodec_register_all();
	av_log_set_level(QSV_AV_LOG_LEVEL);

	if (NULL == (m_Mux.format.pOutputFmt = av_guess_format(NULL, filename.c_str(), NULL))) {
		m_strOutputInfo += _T("avcodec writer: failed to assume format from audio filename.\n");
		m_strOutputInfo += _T("                please set proper extension for audio file.\n");
		return MFX_ERR_NULL_PTR;
	}
	m_Mux.format.pFormatCtx = avformat_alloc_context();
	m_Mux.format.pFormatCtx->oformat = m_Mux.format.pOutputFmt;
	const bool isMatroska = 0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "matroska");

#if USE_CUSTOM_IO
	m_Mux.format.nAVOutBufferSize = 1024 * 1024;
	m_Mux.format.nOutputBufferSize = 16 * 1024 * 1024;
	if (prm->pVideoInfo) {
		m_Mux.format.nAVOutBufferSize *= 8;
		m_Mux.format.nOutputBufferSize *= 4;
	}

	if (NULL == (m_Mux.format.pAVOutBuffer = (mfxU8 *)av_malloc(m_Mux.format.nAVOutBufferSize))) {
		m_strOutputInfo += _T("avcodec writer: failed to allocate muxer buffer.\n");
		return MFX_ERR_NULL_PTR;
	}

	if (fopen_s(&m_Mux.format.fpOutput, filename.c_str(), "wb")) {
		m_strOutputInfo += _T("avcodec writer: failed to open audio output file.\n");
		return MFX_ERR_NULL_PTR; // Couldn't open file
	}
	//確保できなかったら、サイズを小さくして再度確保を試みる (最終的に1MBも確保できなかったら諦める)
	for ( ; m_Mux.format.nOutputBufferSize >= 1024 * 1024; m_Mux.format.nOutputBufferSize >>= 1) {
		if (NULL != (m_Mux.format.pOutputBuffer = (char *)malloc(m_Mux.format.nOutputBufferSize))) {
			setvbuf(m_Mux.format.fpOutput, m_Mux.format.pOutputBuffer, _IOFBF, m_Mux.format.nOutputBufferSize);
			break;
		}
	}

	if (NULL == (m_Mux.format.pFormatCtx->pb = avio_alloc_context(m_Mux.format.pAVOutBuffer, m_Mux.format.nAVOutBufferSize, 1, this, funcReadPacket, funcWritePacket, funcSeek))) {
		m_strOutputInfo += _T("avcodec writer: failed to alloc avio context.\n");
		return MFX_ERR_NULL_PTR;
	}
#else
	if (0 > avio_open2(&m_Mux.format.pFormatCtx->pb, filename.c_str(), AVIO_FLAG_WRITE, NULL, NULL)) {
		m_strOutputInfo += _T("avcodec writer: failed to open audio output file.\n");
		return MFX_ERR_NULL_PTR; // Couldn't open file
	}
#endif
	if (prm->pVideoInfo) {
		m_Mux.format.pFormatCtx->video_codec_id = getAVCodecId(prm->pVideoInfo->CodecId);
		if (m_Mux.format.pFormatCtx->video_codec_id == AV_CODEC_ID_NONE) {
			m_strOutputInfo += _T("avcodec writer: failed to find codec id for video.\n");
			return MFX_ERR_NULL_PTR;
		}
		m_Mux.format.pFormatCtx->oformat->video_codec = m_Mux.format.pFormatCtx->video_codec_id;
		if (NULL == (m_Mux.video.pCodec = avcodec_find_decoder(m_Mux.format.pFormatCtx->video_codec_id))) {
			m_strOutputInfo += _T("avcodec writer: failed to codec for video.\n");
			return MFX_ERR_NULL_PTR;
		}
		if (NULL == (m_Mux.video.pStream = avformat_new_stream(m_Mux.format.pFormatCtx, m_Mux.video.pCodec))) {
			m_strOutputInfo += _T("avcodec writer: failed to create new stream for video.\n");
			return MFX_ERR_NULL_PTR;
		}
		m_Mux.video.nFPS = av_make_q(prm->pVideoInfo->FrameInfo.FrameRateExtN, prm->pVideoInfo->FrameInfo.FrameRateExtD);

		m_Mux.video.pCodecCtx = m_Mux.video.pStream->codec;
		m_Mux.video.pCodecCtx->codec_id                = m_Mux.format.pFormatCtx->video_codec_id;
		m_Mux.video.pCodecCtx->width                   = prm->pVideoInfo->FrameInfo.CropW;
		m_Mux.video.pCodecCtx->height                  = prm->pVideoInfo->FrameInfo.CropH;
		m_Mux.video.pCodecCtx->time_base               = av_inv_q(m_Mux.video.nFPS);
		m_Mux.video.pCodecCtx->pix_fmt                 = AV_PIX_FMT_YUV420P;
		m_Mux.video.pCodecCtx->compression_level       = FF_COMPRESSION_DEFAULT;
		m_Mux.video.pCodecCtx->level                   = prm->pVideoInfo->CodecLevel;
		m_Mux.video.pCodecCtx->profile                 = prm->pVideoInfo->CodecProfile;
		m_Mux.video.pCodecCtx->refs                    = prm->pVideoInfo->NumRefFrame;
		m_Mux.video.pCodecCtx->gop_size                = prm->pVideoInfo->GopPicSize;
		m_Mux.video.pCodecCtx->max_b_frames            = prm->pVideoInfo->GopRefDist - 1;
		m_Mux.video.pCodecCtx->chroma_sample_location  = AVCHROMA_LOC_LEFT;
		m_Mux.video.pCodecCtx->slice_count             = prm->pVideoInfo->NumSlice;
		m_Mux.video.pCodecCtx->sample_aspect_ratio.num = prm->pVideoInfo->FrameInfo.AspectRatioW;
		m_Mux.video.pCodecCtx->sample_aspect_ratio.den = prm->pVideoInfo->FrameInfo.AspectRatioH;
		if (prm->pVideoSignalInfo->ColourDescriptionPresent) {
			m_Mux.video.pCodecCtx->colorspace          = (AVColorSpace)prm->pVideoSignalInfo->MatrixCoefficients;
			m_Mux.video.pCodecCtx->color_primaries     = (AVColorPrimaries)prm->pVideoSignalInfo->ColourPrimaries;
			m_Mux.video.pCodecCtx->color_range         = (AVColorRange)(prm->pVideoSignalInfo->VideoFullRange ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG);
			m_Mux.video.pCodecCtx->color_trc           = (AVColorTransferCharacteristic)prm->pVideoSignalInfo->TransferCharacteristics;
		}
		if (0 > avcodec_open2(m_Mux.video.pCodecCtx, m_Mux.video.pCodec, NULL)) {
			m_strOutputInfo += _T("avcodec writer: failed to open codec for video.\n");
			return MFX_ERR_NULL_PTR;
		}
		if (isMatroska) {
			m_Mux.video.pCodecCtx->time_base = av_make_q(1, 1000);
		}
		m_Mux.video.pStream->time_base = m_Mux.video.pCodecCtx->time_base;
		m_Mux.video.pStream->codec->pkt_timebase = m_Mux.video.pStream->time_base;
		m_Mux.video.pStream->codec->time_base = m_Mux.video.pStream->time_base;
		m_Mux.video.pStream->codec->framerate = m_Mux.video.nFPS;

		m_Mux.video.bDtsUnavailable = prm->bVideoDtsUnavailable;
	}

	if (prm->pCodecCtxAudioIn) {
		m_Mux.audio.pCodecCtxIn = avcodec_alloc_context3(NULL);
		avcodec_copy_context(m_Mux.audio.pCodecCtxIn, prm->pCodecCtxAudioIn);
		if (NULL == (m_Mux.audio.pStream = avformat_new_stream(m_Mux.format.pFormatCtx, NULL))) {
			m_strOutputInfo += _T("avcodec writer: failed to create new stream for audio.\n");
			return MFX_ERR_NULL_PTR;
		}

		//音声がwavの場合、フォーマット変換が必要な場合がある
		AVCodecID codecId = AV_CODEC_ID_NONE;
		if (AV_CODEC_ID_NONE != (codecId = PCMRequiresConversion(m_Mux.audio.pCodecCtxIn))) {
			//PCM decoder
			if (NULL == (m_Mux.audio.pOutCodecDecode = avcodec_find_decoder(m_Mux.audio.pCodecCtxIn->codec_id))) {
				m_strOutputInfo += errorMesForCodec(_T("avcodec writer: failed to find decoder"), prm->pCodecCtxAudioIn->codec_id);
				return MFX_ERR_NULL_PTR;
			}
			if (NULL == (m_Mux.audio.pOutCodecDecodeCtx = avcodec_alloc_context3(m_Mux.audio.pOutCodecDecode))) {
				m_strOutputInfo += errorMesForCodec(_T("avcodec writer: failed to get decode codec context"), prm->pCodecCtxAudioIn->codec_id);
				return MFX_ERR_NULL_PTR;
			}
			//設定されていない必須情報があれば設定する
#define COPY_IF_ZERO(dst, src) { if ((dst)==0) (dst)=(src); }
			COPY_IF_ZERO(m_Mux.audio.pOutCodecDecodeCtx->sample_rate,    prm->pCodecCtxAudioIn->sample_rate);
			COPY_IF_ZERO(m_Mux.audio.pOutCodecDecodeCtx->channels,       prm->pCodecCtxAudioIn->channels);
			COPY_IF_ZERO(m_Mux.audio.pOutCodecDecodeCtx->channel_layout, prm->pCodecCtxAudioIn->channel_layout);
#undef COPY_IF_ZERO
			if (0 > avcodec_open2(m_Mux.audio.pOutCodecDecodeCtx, m_Mux.audio.pOutCodecDecode, NULL)) {
				m_strOutputInfo += errorMesForCodec(_T("avcodec writer: failed to open decoder"), prm->pCodecCtxAudioIn->codec_id);
				return MFX_ERR_NULL_PTR;
			}
			av_new_packet(&m_Mux.audio.OutPacket, 512 * 1024);
			m_Mux.audio.OutPacket.size = 0;

			//PCM encoder
			if (NULL == (m_Mux.audio.pOutCodecEncode = avcodec_find_encoder(codecId))) {
				m_strOutputInfo += errorMesForCodec(_T("avcodec writer: failed to find encoder"), codecId);
				return MFX_ERR_NULL_PTR;
			}
			if (NULL == (m_Mux.audio.pOutCodecEncodeCtx = avcodec_alloc_context3(m_Mux.audio.pOutCodecEncode))) {
				m_strOutputInfo += errorMesForCodec(_T("avcodec writer: failed to get encode codec context"), codecId);
				return MFX_ERR_NULL_PTR;
			}
			m_Mux.audio.pOutCodecEncodeCtx->sample_fmt          = prm->pCodecCtxAudioIn->sample_fmt;
			m_Mux.audio.pOutCodecEncodeCtx->sample_rate         = prm->pCodecCtxAudioIn->sample_rate;
			m_Mux.audio.pOutCodecEncodeCtx->channels            = prm->pCodecCtxAudioIn->channels;
			m_Mux.audio.pOutCodecEncodeCtx->channel_layout      = prm->pCodecCtxAudioIn->channel_layout;
			m_Mux.audio.pOutCodecEncodeCtx->bits_per_raw_sample = prm->pCodecCtxAudioIn->bits_per_raw_sample;
			if (0 > avcodec_open2(m_Mux.audio.pOutCodecEncodeCtx, m_Mux.audio.pOutCodecEncode, NULL)) {
				m_strOutputInfo += errorMesForCodec(_T("avcodec writer: failed to open encoder"), codecId);
				return MFX_ERR_NULL_PTR;
			}
		} else if (m_Mux.audio.pCodecCtxIn->codec_id == AV_CODEC_ID_AAC && m_Mux.audio.pCodecCtxIn->extradata == NULL && m_Mux.video.pStream) {
			if (NULL == (m_Mux.audio.pAACBsfc = av_bitstream_filter_init("aac_adtstoasc"))) {
				m_strOutputInfo += _T("avcodec writer: failed to open bitstream filter for AAC audio.");
				return MFX_ERR_NULL_PTR;
			}
			if (prm->pAudioPktSample) {
				//mkvではavformat_write_headerまでにAVCodecContextにextradataをセットしておく必要がある
				AVPacket *audpkt = prm->pAudioPktSample;
				if (0 > av_bitstream_filter_filter(m_Mux.audio.pAACBsfc, m_Mux.audio.pCodecCtxIn, NULL, &audpkt->data, &audpkt->size, audpkt->data, audpkt->size, 0)) {
					m_strOutputInfo += _T("avcodec writer: failed to run bitstream filter for AAC audio.");
					return MFX_ERR_UNKNOWN;
				}
			}
		}

		//パラメータのコピー
		//下記のようにavcodec_copy_contextを使用するとavformat_write_header()が
		//Tag mp4a/0x6134706d incompatible with output codec id '86018' ([64][0][0][0])のようなエラーを出すことがある
		//そのため、必要な値だけをひとつづつコピーする
		//avcodec_copy_context(m_Mux.audio.pStream->codec, srcCodecCtx);
		const AVCodecContext *srcCodecCtx            = (m_Mux.audio.pOutCodecEncodeCtx) ? m_Mux.audio.pOutCodecEncodeCtx : prm->pCodecCtxAudioIn;
		m_Mux.audio.pStream->codec->codec_type      = srcCodecCtx->codec_type;
		m_Mux.audio.pStream->codec->codec_id        = srcCodecCtx->codec_id;
		m_Mux.audio.pStream->codec->frame_size      = srcCodecCtx->frame_size;
		m_Mux.audio.pStream->codec->channels        = srcCodecCtx->channels;
		m_Mux.audio.pStream->codec->channel_layout  = srcCodecCtx->channel_layout;
		m_Mux.audio.pStream->codec->ticks_per_frame = srcCodecCtx->ticks_per_frame;
		m_Mux.audio.pStream->codec->sample_rate     = srcCodecCtx->sample_rate;
		m_Mux.audio.pStream->codec->sample_fmt      = srcCodecCtx->sample_fmt;
		if (srcCodecCtx->extradata_size) {
			SetExtraData(m_Mux.audio.pStream->codec, srcCodecCtx->extradata, srcCodecCtx->extradata_size);
		} else if (m_Mux.audio.pCodecCtxIn->extradata_size) {
			//aac_adtstoascから得たヘッダをコピーする
			//これをしておかないと、avformat_write_headerで"Error parsing AAC extradata, unable to determine samplerate."という
			//意味不明なエラーメッセージが表示される
			SetExtraData(m_Mux.audio.pStream->codec, m_Mux.audio.pCodecCtxIn->extradata, m_Mux.audio.pCodecCtxIn->extradata_size);
		}
		m_Mux.audio.pStream->time_base = av_make_q(1, m_Mux.audio.pStream->codec->sample_rate);
		m_Mux.audio.pStream->codec->time_base = m_Mux.audio.pStream->time_base;
	}
	
	sprintf_s(m_Mux.format.pFormatCtx->filename, filename.c_str());
	if (m_Mux.format.pOutputFmt->flags & AVFMT_GLOBALHEADER) {
		if (m_Mux.video.pStream) { m_Mux.video.pStream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER; }
		if (m_Mux.audio.pStream) { m_Mux.audio.pStream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER; }
	}

	//QSVEncCでエンコーダしたことを記録してみる
	//これは直接metadetaにセットする
	sprintf_s(m_Mux.format.metadataStr, "QSVEncC (%s) %s", tchar_to_string(BUILD_ARCH_STR).c_str(), VER_STR_FILEVERSION);
	av_dict_set(&m_Mux.format.pFormatCtx->metadata, "encoding_tool", m_Mux.format.metadataStr, 0); //mp4
	//encoderではなく、encoding_toolを使用する。mp4はcomment, titleなどは設定可能, mkvではencode_byも可能

	//mp4のmajor_brandをisonからmp42に変更
	//これはmetadataではなく、avformat_write_headerのoptionsに渡す
	//この差ははっきり言って謎
	AVDictionary *avdict = NULL;
	if (m_Mux.video.pStream && 0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "mp4")) {
		av_dict_set(&avdict, "brand", "mp42", 0);
	}

	//なんらかの問題があると、ここでよく死ぬ
	int ret = 0;
	if (0 > (ret = avformat_write_header(m_Mux.format.pFormatCtx, &avdict))) {
		m_strOutputInfo += _T("avcodec writer: failed to write header for output file. :");
		m_strOutputInfo += qsv_av_err2str(ret) + tstring(_T("\n"));
		if (avdict) av_dict_free(&avdict);
		return MFX_ERR_UNKNOWN;
	}
	//不正なオプションを渡していないかチェック
	for (const AVDictionaryEntry *t = NULL; NULL != (t = av_dict_get(avdict, "", t, AV_DICT_IGNORE_SUFFIX));) {
		m_strOutputInfo += _T("avcodec writer: Unknown option to muxer. :");
		m_strOutputInfo += char_to_tstring(t->key);
		return MFX_ERR_UNKNOWN;
	}
	if (avdict) {
		av_dict_free(&avdict);
	}
	av_dump_format(m_Mux.format.pFormatCtx, 0, m_Mux.format.pFormatCtx->filename, 1);

	//API v1.6以下でdtsがQSVが提供されない場合、自前で計算する必要がある
	//API v1.6ではB-pyramidが存在しないので、Bフレームがあるかないかだけ考慮するればよい
	if (m_Mux.video.bDtsUnavailable) {
		m_Mux.video.nFpsBaseNextDts = (m_Mux.video.pCodecCtx->max_b_frames == 0) ? 0 : -1;
	}
	
	TCHAR mes[256];
	_stprintf_s(mes, _T("avcodec writer: %s%s%s -> %s"),
		(m_Mux.video.pStream) ? char_to_tstring(avcodec_get_name(m_Mux.video.pStream->codec->codec_id)).c_str() : _T(""),
		(m_Mux.video.pStream && m_Mux.audio.pStream) ? _T(", ") : _T(""),
		(m_Mux.audio.pStream) ? char_to_tstring(avcodec_get_name(m_Mux.audio.pStream->codec->codec_id)).c_str() : _T(""),
		char_to_tstring(m_Mux.format.pFormatCtx->oformat->name).c_str());
	m_strOutputInfo += mes;

	m_pEncSatusInfo = pEncSatusInfo;
	m_Mux.format.bStreamError = false;

	m_bInited = true;

	return MFX_ERR_NONE;
}

mfxStatus CAvcodecWriter::WriteNextFrame(mfxBitstream *pMfxBitstream) {
	AVPacket pkt = { 0 };
	av_init_packet(&pkt);
	av_new_packet(&pkt, pMfxBitstream->DataLength);
	memcpy(pkt.data, pMfxBitstream->Data + pMfxBitstream->DataOffset, pMfxBitstream->DataLength);
	pkt.size = pMfxBitstream->DataLength;

	const AVRational fpsTimebase = av_inv_q(m_Mux.video.nFPS);
	const AVRational streamTimebase = m_Mux.video.pStream->codec->pkt_timebase;
	pkt.stream_index = m_Mux.video.pStream->index;
	pkt.flags        = (pMfxBitstream->FrameType & MFX_FRAMETYPE_IDR) >> 7;
	pkt.duration     = (int)av_rescale_q(1, fpsTimebase, streamTimebase);
	pkt.pts          = av_rescale_q(av_rescale_q(pMfxBitstream->TimeStamp, QSV_NATIVE_TIMEBASE, fpsTimebase), fpsTimebase, streamTimebase);
	if (!m_Mux.video.bDtsUnavailable) {
		pkt.dts = av_rescale_q(av_rescale_q(pMfxBitstream->DecodeTimeStamp, QSV_NATIVE_TIMEBASE, fpsTimebase), fpsTimebase, streamTimebase);
	} else {
		pkt.dts = av_rescale_q(m_Mux.video.nFpsBaseNextDts, fpsTimebase, streamTimebase);
		m_Mux.video.nFpsBaseNextDts++;
	}
	m_Mux.format.bStreamError |= 0 != av_interleaved_write_frame(m_Mux.format.pFormatCtx, &pkt);

	m_pEncSatusInfo->SetOutputData(pMfxBitstream->DataLength, pMfxBitstream->FrameType);
	pMfxBitstream->DataLength = 0;
	return (m_Mux.format.bStreamError) ? MFX_ERR_UNKNOWN : MFX_ERR_NONE;
}

void CAvcodecWriter::applyBitstreamFilterAAC(AVPacket *pkt) {
	uint8_t *data = NULL;
	int dataSize = 0;
	//毎回bitstream filterを初期化して、extradataに新しいヘッダを供給する
	//動画とmuxする際に必須
	av_bitstream_filter_close(m_Mux.audio.pAACBsfc);
	m_Mux.audio.pAACBsfc = av_bitstream_filter_init("aac_adtstoasc");
	if (0 > av_bitstream_filter_filter(m_Mux.audio.pAACBsfc, m_Mux.audio.pCodecCtxIn,
		nullptr, &data, &dataSize, pkt->data, pkt->size, 0)) {
		m_Mux.format.bStreamError = (m_Mux.audio.nPacketWritten > 1);
		pkt->duration = 0; //書き込み処理が行われないように
	} else {
		if (pkt->buf->size < dataSize) {
			av_grow_packet(pkt, dataSize);
		}
		if (pkt->data != data) {
			memmove(pkt->data, data, dataSize);
		}
		pkt->size = dataSize;
	}
}

mfxStatus CAvcodecWriter::WriteNextFrame(AVPacket *pkt) {
	m_Mux.audio.nPacketWritten++;
	
	AVRational samplerate = { 1, m_Mux.audio.pCodecCtxIn->sample_rate };
	AVPacket encodePkt = { 0 };
	int samples = 0;
	BOOL got_result = TRUE;
	if (m_Mux.audio.pAACBsfc) {
		applyBitstreamFilterAAC(pkt);
	}
	if (!m_Mux.audio.pOutCodecDecodeCtx) {
		samples = (int)av_rescale_q(pkt->duration, m_Mux.audio.pCodecCtxIn->pkt_timebase, samplerate);
	} else {
		AVFrame *decodedFrame = av_frame_alloc();
		AVPacket *pktIn = pkt;
		if (m_Mux.audio.OutPacket.size != 0) {
			int currentSize = m_Mux.audio.OutPacket.size;
			if (m_Mux.audio.OutPacket.buf->size < currentSize + pkt->size) {
				av_grow_packet(&m_Mux.audio.OutPacket, currentSize + pkt->size);
				m_Mux.audio.OutPacket.size = currentSize;
			}
			memcpy(m_Mux.audio.OutPacket.data, pkt->data, pkt->size);
			m_Mux.audio.OutPacket.size += pkt->size;
			pktIn = &m_Mux.audio.OutPacket;
		}
		//PCM decode
		int len = 0;
		if (0 > (len = avcodec_decode_audio4(m_Mux.audio.pOutCodecDecodeCtx, decodedFrame, &got_result, pkt))) {
			m_strOutputInfo += _T("avcodec writer: failed to convert pcm format(1). :");
			m_strOutputInfo += qsv_av_err2str(len) + tstring(_T("\n"));
			m_Mux.format.bStreamError = true;
		} else if (pkt->size != len) {
			int newLen = pkt->size - len;
			memmove(m_Mux.audio.OutPacket.data, pkt->data + len, newLen);
			m_Mux.audio.OutPacket.size = newLen;
		} else {
			m_Mux.audio.OutPacket.size = 0;
		}
		if (got_result) {
			//PCM encode
			av_init_packet(&encodePkt);
			int ret = avcodec_encode_audio2(m_Mux.audio.pOutCodecEncodeCtx, &encodePkt, decodedFrame, &got_result);
			if (ret < 0) {
				m_strOutputInfo += _T("avcodec writer: failed to convert pcm format(2). :");
				m_strOutputInfo += qsv_av_err2str(ret) + tstring(_T("\n"));
				m_Mux.format.bStreamError = true;
			} else if (got_result) {
				samples = (int)av_rescale_q(encodePkt.duration, m_Mux.audio.pOutCodecEncodeCtx->pkt_timebase, samplerate);
				pkt = &encodePkt;
			}
		}
		if (decodedFrame) {
			av_frame_free(&decodedFrame);
		}
	}

	if (samples) {
		//durationについて、sample数から出力ストリームのtimebaseに変更する
		pkt->stream_index = m_Mux.audio.pStream->index;
		pkt->flags        = AV_PKT_FLAG_KEY;
		pkt->dts          = av_rescale_q(m_Mux.audio.nOutputSamples, samplerate, m_Mux.audio.pStream->time_base);
		pkt->pts          = pkt->dts;
		pkt->duration     = (int)(pkt->pts - m_Mux.audio.nLastPts);
		if (pkt->duration == 0)
			pkt->duration = (int)av_rescale_q(samples, samplerate, m_Mux.audio.pStream->time_base);
		m_Mux.audio.nLastPts = pkt->pts;
		m_Mux.format.bStreamError |= 0 != av_interleaved_write_frame(m_Mux.format.pFormatCtx, pkt);
		m_Mux.audio.nOutputSamples += samples;
	} else {
		//av_interleaved_write_frameに渡ったパケットは開放する必要がないが、
		//それ以外は解放してやる必要がある
		av_free_packet(pkt);
	}
	return (m_Mux.format.bStreamError) ? MFX_ERR_UNKNOWN : MFX_ERR_NONE;
}

#if USE_CUSTOM_IO
int CAvcodecWriter::readPacket(uint8_t *buf, int buf_size) {
	return (int)fread(buf, 1, buf_size, m_Mux.format.fpOutput);
}
int CAvcodecWriter::writePacket(uint8_t *buf, int buf_size) {
	return (int)fwrite(buf, 1, buf_size, m_Mux.format.fpOutput);
}
int64_t CAvcodecWriter::seek(int64_t offset, int whence) {
	return _fseeki64(m_Mux.format.fpOutput, offset, whence);
}
#endif //USE_CUSTOM_IO

#endif //ENABLE_AVCODEC_QSV_READER
