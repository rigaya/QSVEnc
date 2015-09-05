//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#include <io.h>
#include <fcntl.h>
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
    m_strWriterName = _T("avout");
}

CAvcodecWriter::~CAvcodecWriter() {

}

void CAvcodecWriter::CloseAudio(AVMuxAudio *pMuxAudio) {
    //close resampler
    if (pMuxAudio->pSwrContext) {
        swr_free(&pMuxAudio->pSwrContext);
        AddMessage(QSV_LOG_DEBUG, _T("Closed pSwrContext.\n"));
    }
    if (pMuxAudio->pSwrBuffer) {
        if (pMuxAudio->pSwrBuffer[0]) {
            av_free(pMuxAudio->pSwrBuffer[0]);
        }
        av_free(pMuxAudio->pSwrBuffer);
    }

    //close decoder
    if (pMuxAudio->pOutCodecDecodeCtx) {
        avcodec_close(pMuxAudio->pOutCodecDecodeCtx);
        av_free(pMuxAudio->pOutCodecDecodeCtx);
        AddMessage(QSV_LOG_DEBUG, _T("Closed pOutCodecDecodeCtx.\n"));
    }

    //close encoder
    if (pMuxAudio->pOutCodecEncodeCtx) {
        avcodec_close(pMuxAudio->pOutCodecEncodeCtx);
        av_free(pMuxAudio->pOutCodecEncodeCtx);
        AddMessage(QSV_LOG_DEBUG, _T("Closed pOutCodecEncodeCtx.\n"));
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
        AddMessage(QSV_LOG_DEBUG, _T("Closed AVCodecConetxt.\n"));
    }
    memset(pMuxAudio, 0, sizeof(pMuxAudio[0]));
    AddMessage(QSV_LOG_DEBUG, _T("Closed audio.\n"));
}

void CAvcodecWriter::CloseVideo(AVMuxVideo *pMuxVideo) {
    memset(pMuxVideo, 0, sizeof(pMuxVideo[0]));
    AddMessage(QSV_LOG_DEBUG, _T("Closed video.\n"));
}

void CAvcodecWriter::CloseFormat(AVMuxFormat *pMuxFormat) {
    if (pMuxFormat->pFormatCtx) {
        if (!pMuxFormat->bStreamError) {
            av_write_trailer(pMuxFormat->pFormatCtx);
        }
#if USE_CUSTOM_IO
        if (!pMuxFormat->fpOutput) {
#endif
            avio_close(pMuxFormat->pFormatCtx->pb);
            AddMessage(QSV_LOG_DEBUG, _T("Closed AVIO Context.\n"));
#if USE_CUSTOM_IO
        }
#endif
        avformat_free_context(pMuxFormat->pFormatCtx);
        AddMessage(QSV_LOG_DEBUG, _T("Closed avformat context.\n"));
    }
#if USE_CUSTOM_IO
    if (pMuxFormat->fpOutput) {
        fflush(pMuxFormat->fpOutput);
        fclose(pMuxFormat->fpOutput);
        AddMessage(QSV_LOG_DEBUG, _T("Closed File Pointer.\n"));
    }

    if (pMuxFormat->pAVOutBuffer) {
        av_free(pMuxFormat->pAVOutBuffer);
    }

    if (pMuxFormat->pOutputBuffer) {
        free(pMuxFormat->pOutputBuffer);
    }
#endif //USE_CUSTOM_IO
    memset(pMuxFormat, 0, sizeof(pMuxFormat[0]));
    AddMessage(QSV_LOG_DEBUG, _T("Closed format.\n"));
}

void CAvcodecWriter::Close() {
    AddMessage(QSV_LOG_DEBUG, _T("Closing...\n"));
    CloseFormat(&m_Mux.format);
    for (int i = 0; i < (int)m_Mux.audio.size(); i++) {
        CloseAudio(&m_Mux.audio[i]);
    }
    m_Mux.audio.clear();
    CloseVideo(&m_Mux.video);
    m_strOutputInfo.clear();
    AddMessage(QSV_LOG_DEBUG, _T("Closed.\n"));
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
bool CAvcodecWriter::codecIDIsPCM(AVCodecID targetCodec) {
    const std::vector<AVCodecID> pcmCodecs = {
        AV_CODEC_ID_FIRST_AUDIO,
        AV_CODEC_ID_PCM_S16LE,
        AV_CODEC_ID_PCM_S16BE,
        AV_CODEC_ID_PCM_U16LE,
        AV_CODEC_ID_PCM_U16BE,
        AV_CODEC_ID_PCM_S8,
        AV_CODEC_ID_PCM_U8,
        AV_CODEC_ID_PCM_MULAW,
        AV_CODEC_ID_PCM_ALAW,
        AV_CODEC_ID_PCM_S32LE,
        AV_CODEC_ID_PCM_S32BE,
        AV_CODEC_ID_PCM_U32LE,
        AV_CODEC_ID_PCM_U32BE,
        AV_CODEC_ID_PCM_S24LE,
        AV_CODEC_ID_PCM_S24BE,
        AV_CODEC_ID_PCM_U24LE,
        AV_CODEC_ID_PCM_U24BE,
        AV_CODEC_ID_PCM_S24DAUD,
        AV_CODEC_ID_PCM_ZORK,
        AV_CODEC_ID_PCM_S16LE_PLANAR,
        AV_CODEC_ID_PCM_DVD,
        AV_CODEC_ID_PCM_F32BE,
        AV_CODEC_ID_PCM_F32LE,
        AV_CODEC_ID_PCM_F64BE,
        AV_CODEC_ID_PCM_F64LE,
        AV_CODEC_ID_PCM_BLURAY,
        AV_CODEC_ID_PCM_LXF,
        AV_CODEC_ID_S302M,
        AV_CODEC_ID_PCM_S8_PLANAR,
        AV_CODEC_ID_PCM_S24LE_PLANAR_DEPRECATED,
        AV_CODEC_ID_PCM_S32LE_PLANAR_DEPRECATED,
        AV_CODEC_ID_PCM_S16BE_PLANAR_DEPRECATED,
        AV_CODEC_ID_PCM_S24LE_PLANAR,
        AV_CODEC_ID_PCM_S32LE_PLANAR,
        AV_CODEC_ID_PCM_S16BE_PLANAR
    };
    return (pcmCodecs.end() != std::find(pcmCodecs.begin(), pcmCodecs.end(), targetCodec));
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
    AVCodecID prmCodec = AV_CODEC_ID_NONE;
    for (int i = 0; i < _countof(pcmConvertCodecs); i++) {
        if (pcmConvertCodecs[i].first == audioCtx->codec_id) {
            if (pcmConvertCodecs[i].second != AV_CODEC_ID_FIRST_AUDIO) {
                return pcmConvertCodecs[i].second;
            }
            switch (audioCtx->bits_per_raw_sample) {
            case 32: prmCodec = AV_CODEC_ID_PCM_S32LE; break;
            case 24: prmCodec = AV_CODEC_ID_PCM_S24LE; break;
            case 8:  prmCodec = AV_CODEC_ID_PCM_S16LE; break;
            case 16:
            default: prmCodec = AV_CODEC_ID_PCM_S16LE; break;
            }
        }
    }
    if (prmCodec != AV_CODEC_ID_NONE) {
        AddMessage(QSV_LOG_DEBUG, _T("PCM requires conversion...\n"));
    }
    return prmCodec;
}

void CAvcodecWriter::SetExtraData(AVCodecContext *codecCtx, const mfxU8 *data, mfxU32 size) {
    if (data == nullptr || size == 0)
        return;
    if (codecCtx->extradata)
        av_free(codecCtx->extradata);
    codecCtx->extradata_size = size;
    codecCtx->extradata      = (uint8_t *)av_malloc(codecCtx->extradata_size);
    memcpy(codecCtx->extradata, data, size);
};

//音声のchannel_layoutを自動選択する
uint64_t CAvcodecWriter::AutoSelectChannelLayout(const uint64_t *pChannelLayout, const AVCodecContext *pSrcAudioCtx) {
    int srcChannels = av_get_channel_layout_nb_channels(pSrcAudioCtx->channel_layout);
    if (srcChannels == 0) {
        srcChannels = pSrcAudioCtx->channels;
    }
    if (pChannelLayout == nullptr) {
        switch (srcChannels) {
        case 1:  return AV_CH_LAYOUT_MONO;
        case 2:  return AV_CH_LAYOUT_STEREO;
        case 3:  return AV_CH_LAYOUT_2_1;
        case 4:  return AV_CH_LAYOUT_QUAD;
        case 5:  return AV_CH_LAYOUT_5POINT0;
        case 6:  return AV_CH_LAYOUT_5POINT1;
        case 7:  return AV_CH_LAYOUT_6POINT1;
        case 8:  return AV_CH_LAYOUT_7POINT1;
        default: return AV_CH_LAYOUT_NATIVE;
        }
    }

    for (int i = 0; pChannelLayout[i]; i++) {
        if (srcChannels == av_get_channel_layout_nb_channels(pChannelLayout[i])) {
            return pChannelLayout[i];
        }
    }
    return pChannelLayout[0];
}

int CAvcodecWriter::AutoSelectSamplingRate(const int *pSamplingRateList, int nSrcSamplingRate) {
    if (pSamplingRateList == nullptr) {
        return nSrcSamplingRate;
    }
    //一致するものがあれば、それを返す
    int i = 0;
    for (; pSamplingRateList[i]; i++) {
        if (nSrcSamplingRate == pSamplingRateList[i]) {
            return nSrcSamplingRate;
        }
    }
    //相対誤差が最も小さいものを選択する
    vector<double> diffrate(i);
    for (i = 0; pSamplingRateList[i]; i++) {
        diffrate[i] = abs(1 - pSamplingRateList[i] / (double)nSrcSamplingRate);
    }
    return pSamplingRateList[std::distance(diffrate.begin(), std::min_element(diffrate.begin(), diffrate.end()))];
}

AVSampleFormat CAvcodecWriter::AutoSelectSampleFmt(const AVSampleFormat *pSamplefmtList, const AVCodecContext *pSrcAudioCtx) {
    AVSampleFormat srcFormat = pSrcAudioCtx->sample_fmt;
    if (pSamplefmtList == nullptr) {
        return srcFormat;
    }
    if (srcFormat == AV_SAMPLE_FMT_NONE) {
        return pSamplefmtList[0];
    }
    for (int i = 0; pSamplefmtList[i] >= 0; i++) {
        if (srcFormat == pSamplefmtList[i]) {
            return pSamplefmtList[i];
        }
    }
    vector<std::pair<AVSampleFormat, int>> sampleFmtLevel = {
        { AV_SAMPLE_FMT_DBLP, 8 },
        { AV_SAMPLE_FMT_DBL,  8 },
        { AV_SAMPLE_FMT_FLTP, 6 },
        { AV_SAMPLE_FMT_FLT,  6 },
        { AV_SAMPLE_FMT_S32P, 4 },
        { AV_SAMPLE_FMT_S32,  4 },
        { AV_SAMPLE_FMT_S16P, 2 },
        { AV_SAMPLE_FMT_S16,  2 },
        { AV_SAMPLE_FMT_U8P,  1 },
        { AV_SAMPLE_FMT_U8,   1 },
    };
    int srcFormatLevel = std::find_if(sampleFmtLevel.begin(), sampleFmtLevel.end(),
        [srcFormat](const std::pair<AVSampleFormat, int>& targetFormat) { return targetFormat.first == srcFormat;})->second;
    auto foundFormat = std::find_if(sampleFmtLevel.begin(), sampleFmtLevel.end(),
        [srcFormatLevel](const std::pair<AVSampleFormat, int>& targetFormat) { return targetFormat.second == srcFormatLevel; });
    for (; foundFormat != sampleFmtLevel.end(); foundFormat++) {
        for (int i = 0; pSamplefmtList[i] >= 0; i++) {
            if (foundFormat->first == pSamplefmtList[i]) {
                return pSamplefmtList[i];
            }
        }
    }
    return pSamplefmtList[0];
}

mfxStatus CAvcodecWriter::InitVideo(const AvcodecWriterPrm *prm) {
    m_Mux.format.pFormatCtx->video_codec_id = getAVCodecId(prm->pVideoInfo->CodecId);
    if (m_Mux.format.pFormatCtx->video_codec_id == AV_CODEC_ID_NONE) {
        AddMessage(QSV_LOG_ERROR, _T("failed to find codec id for video.\n"));
        return MFX_ERR_NULL_PTR;
    }
    m_Mux.format.pFormatCtx->oformat->video_codec = m_Mux.format.pFormatCtx->video_codec_id;
    if (NULL == (m_Mux.video.pCodec = avcodec_find_decoder(m_Mux.format.pFormatCtx->video_codec_id))) {
        AddMessage(QSV_LOG_ERROR, _T("failed to codec for video.\n"));
        return MFX_ERR_NULL_PTR;
    }
    if (NULL == (m_Mux.video.pStream = avformat_new_stream(m_Mux.format.pFormatCtx, m_Mux.video.pCodec))) {
        AddMessage(QSV_LOG_ERROR, _T("failed to create new stream for video.\n"));
        return MFX_ERR_NULL_PTR;
    }
    m_Mux.video.nFPS = av_make_q(prm->pVideoInfo->FrameInfo.FrameRateExtN, prm->pVideoInfo->FrameInfo.FrameRateExtD);
    AddMessage(QSV_LOG_DEBUG, _T("output video stream fps: %d/%d\n"), prm->pVideoInfo->FrameInfo.FrameRateExtN, prm->pVideoInfo->FrameInfo.FrameRateExtD);

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
        AddMessage(QSV_LOG_ERROR, _T("failed to open codec for video.\n"));
        return MFX_ERR_NULL_PTR;
    }
    AddMessage(QSV_LOG_DEBUG, _T("opened video avcodec\n"));

    if (m_Mux.format.bIsMatroska) {
        m_Mux.video.pCodecCtx->time_base = av_make_q(1, 1000);
    }
    m_Mux.video.pStream->time_base           = m_Mux.video.pCodecCtx->time_base;
    m_Mux.video.pStream->codec->pkt_timebase = m_Mux.video.pStream->time_base;
    m_Mux.video.pStream->codec->time_base    = m_Mux.video.pStream->time_base;
    m_Mux.video.pStream->codec->framerate    = m_Mux.video.nFPS;
    m_Mux.video.pStream->start_time          = 0;

    m_Mux.video.bDtsUnavailable = prm->bVideoDtsUnavailable;

    AddMessage(QSV_LOG_DEBUG, _T("output video stream timebase: %d/%d\n"), m_Mux.video.pStream->time_base.num, m_Mux.video.pStream->time_base.den);
    AddMessage(QSV_LOG_DEBUG, _T("bDtsUnavailable: %s\n"), (m_Mux.video.bDtsUnavailable) ? _T("on") : _T("off"));
    return MFX_ERR_NONE;
}

mfxStatus CAvcodecWriter::InitAudio(AVMuxAudio *pMuxAudio, AVOutputAudioPrm *pInputAudio) {
    pMuxAudio->pCodecCtxIn = avcodec_alloc_context3(NULL);
    avcodec_copy_context(pMuxAudio->pCodecCtxIn, pInputAudio->src.pCodecCtx);
    AddMessage(QSV_LOG_DEBUG, _T("start initializing audio ouput...\n"));
    AddMessage(QSV_LOG_DEBUG, _T("output stream index %d, trackId %d, delay %d, \n"), pInputAudio->src.nIndex, pInputAudio->src.nTrackId, pMuxAudio->nDelaySamplesOfAudio);
    AddMessage(QSV_LOG_DEBUG, _T("samplerate %d, stream pkt_timebase %d/%d\n"), pMuxAudio->pCodecCtxIn->sample_rate, pMuxAudio->pCodecCtxIn->pkt_timebase.num, pMuxAudio->pCodecCtxIn->pkt_timebase.den);

    if (NULL == (pMuxAudio->pStream = avformat_new_stream(m_Mux.format.pFormatCtx, NULL))) {
        AddMessage(QSV_LOG_ERROR, _T("failed to create new stream for audio.\n"));
        return MFX_ERR_NULL_PTR;
    }
    pMuxAudio->nInTrackId = pInputAudio->src.nTrackId;
    pMuxAudio->nStreamIndexIn = pInputAudio->src.nIndex;
    pMuxAudio->nLastPtsIn = AV_NOPTS_VALUE;

    //音声がwavの場合、フォーマット変換が必要な場合がある
    AVCodecID codecId = AV_CODEC_ID_NONE;
    if (!avcodecIsCopy(pInputAudio->pEncodeCodec) || AV_CODEC_ID_NONE != (codecId = PCMRequiresConversion(pMuxAudio->pCodecCtxIn))) {
        //setup decoder
        if (NULL == (pMuxAudio->pOutCodecDecode = avcodec_find_decoder(pMuxAudio->pCodecCtxIn->codec_id))) {
            AddMessage(QSV_LOG_ERROR, errorMesForCodec(_T("failed to find decoder"), pInputAudio->src.pCodecCtx->codec_id));
            AddMessage(QSV_LOG_ERROR, _T("Please use --check-decoders to check available decoder.\n"));
            return MFX_ERR_NULL_PTR;
        }
        if (NULL == (pMuxAudio->pOutCodecDecodeCtx = avcodec_alloc_context3(pMuxAudio->pOutCodecDecode))) {
            AddMessage(QSV_LOG_ERROR, errorMesForCodec(_T("failed to get decode codec context"), pInputAudio->src.pCodecCtx->codec_id));
            return MFX_ERR_NULL_PTR;
        }
        //設定されていない必須情報があれば設定する
#define COPY_IF_ZERO(dst, src) { if ((dst)==0) (dst)=(src); }
        COPY_IF_ZERO(pMuxAudio->pOutCodecDecodeCtx->sample_rate,         pInputAudio->src.pCodecCtx->sample_rate);
        COPY_IF_ZERO(pMuxAudio->pOutCodecDecodeCtx->channels,            pInputAudio->src.pCodecCtx->channels);
        COPY_IF_ZERO(pMuxAudio->pOutCodecDecodeCtx->channel_layout,      pInputAudio->src.pCodecCtx->channel_layout);
        COPY_IF_ZERO(pMuxAudio->pOutCodecDecodeCtx->bits_per_raw_sample, pInputAudio->src.pCodecCtx->bits_per_raw_sample);
#undef COPY_IF_ZERO
        pMuxAudio->pOutCodecDecodeCtx->pkt_timebase = pInputAudio->src.pCodecCtx->pkt_timebase;
        SetExtraData(pMuxAudio->pOutCodecDecodeCtx, pInputAudio->src.pCodecCtx->extradata, pInputAudio->src.pCodecCtx->extradata_size);
        if (nullptr != strstr(pMuxAudio->pOutCodecDecode->name, "wma")) {
            pMuxAudio->pOutCodecDecodeCtx->block_align = pInputAudio->src.pCodecCtx->block_align;
        }
        int ret;
        if (0 > (ret = avcodec_open2(pMuxAudio->pOutCodecDecodeCtx, pMuxAudio->pOutCodecDecode, NULL))) {
            AddMessage(QSV_LOG_ERROR, _T("failed to open decoder for %s: %s\n"),
                char_to_tstring(avcodec_get_name(pInputAudio->src.pCodecCtx->codec_id)).c_str(), qsv_av_err2str(ret).c_str());
            return MFX_ERR_NULL_PTR;
        }
        AddMessage(QSV_LOG_DEBUG, _T("Audio Decoder opened\n"));
        AddMessage(QSV_LOG_DEBUG, _T("Audio Decode Info: %s, %dch[0x%02x], %.1fkHz, %s, %d/%d\n"), char_to_tstring(avcodec_get_name(pMuxAudio->pCodecCtxIn->codec_id)).c_str(),
            pMuxAudio->pOutCodecDecodeCtx->channels, (uint32_t)pMuxAudio->pOutCodecDecodeCtx->channel_layout, pMuxAudio->pOutCodecDecodeCtx->sample_rate / 1000.0,
            char_to_tstring(av_get_sample_fmt_name(pMuxAudio->pOutCodecDecodeCtx->sample_fmt)).c_str(),
            pMuxAudio->pOutCodecDecodeCtx->pkt_timebase.num, pMuxAudio->pOutCodecDecodeCtx->pkt_timebase.den);

        av_new_packet(&pMuxAudio->OutPacket, 512 * 1024);
        pMuxAudio->OutPacket.size = 0;

        if (codecId != AV_CODEC_ID_NONE) {
            //PCM encoder
            if (NULL == (pMuxAudio->pOutCodecEncode = avcodec_find_encoder(codecId))) {
                AddMessage(QSV_LOG_ERROR, errorMesForCodec(_T("failed to find encoder"), codecId));
                return MFX_ERR_NULL_PTR;
            }
            pInputAudio->pEncodeCodec = AVQSV_CODEC_COPY;
        } else {
            if (avcodecIsAuto(pInputAudio->pEncodeCodec)) {
                //エンコーダを探す (自動)
                if (NULL == (pMuxAudio->pOutCodecEncode = avcodec_find_encoder(m_Mux.format.pOutputFmt->audio_codec))) {
                    AddMessage(QSV_LOG_ERROR, errorMesForCodec(_T("failed to find encoder"), m_Mux.format.pOutputFmt->audio_codec));
                    AddMessage(QSV_LOG_ERROR, _T("Please use --check-encoders to find available encoder.\n"));
                    return MFX_ERR_NULL_PTR;
                }
                AddMessage(QSV_LOG_DEBUG, _T("found encoder for codec %s for audio track %d\n"), char_to_tstring(pMuxAudio->pOutCodecEncode->name).c_str(), pInputAudio->src.nTrackId);
            } else {
                //エンコーダを探す (指定のもの)
                if (NULL == (pMuxAudio->pOutCodecEncode = avcodec_find_encoder_by_name(tchar_to_string(pInputAudio->pEncodeCodec).c_str()))) {
                    AddMessage(QSV_LOG_ERROR, _T("failed to find encoder for codec %s\n"), pInputAudio->pEncodeCodec);
                    AddMessage(QSV_LOG_ERROR, _T("Please use --check-encoders to find available encoder.\n"));
                    return MFX_ERR_NULL_PTR;
                }
                AddMessage(QSV_LOG_DEBUG, _T("found encoder for codec %s selected for audio track %d\n"), char_to_tstring(pMuxAudio->pOutCodecEncode->name).c_str(), pInputAudio->src.nTrackId);
            }
            pInputAudio->pEncodeCodec = _T("codec_something");
        }
        if (NULL == (pMuxAudio->pOutCodecEncodeCtx = avcodec_alloc_context3(pMuxAudio->pOutCodecEncode))) {
            AddMessage(QSV_LOG_ERROR, errorMesForCodec(_T("failed to get encode codec context"), codecId));
            return MFX_ERR_NULL_PTR;
        }
        //select samplefmt
        pMuxAudio->pOutCodecEncodeCtx->sample_fmt          = AutoSelectSampleFmt(pMuxAudio->pOutCodecEncode->sample_fmts, pMuxAudio->pOutCodecDecodeCtx);
        pMuxAudio->pOutCodecEncodeCtx->sample_rate         = AutoSelectSamplingRate(pMuxAudio->pOutCodecEncode->supported_samplerates, pMuxAudio->pOutCodecDecodeCtx->sample_rate);
        pMuxAudio->pOutCodecEncodeCtx->channel_layout      = AutoSelectChannelLayout(pMuxAudio->pOutCodecEncode->channel_layouts, pMuxAudio->pOutCodecDecodeCtx);
        pMuxAudio->pOutCodecEncodeCtx->channels            = av_get_channel_layout_nb_channels(pMuxAudio->pOutCodecEncodeCtx->channel_layout);
        pMuxAudio->pOutCodecEncodeCtx->bits_per_raw_sample = pMuxAudio->pOutCodecDecodeCtx->bits_per_raw_sample;
        pMuxAudio->pOutCodecEncodeCtx->pkt_timebase        = av_make_q(1, pMuxAudio->pOutCodecDecodeCtx->sample_rate);
        if (!avcodecIsCopy(pInputAudio->pEncodeCodec)) {
            pMuxAudio->pOutCodecEncodeCtx->bit_rate        = ((pInputAudio->nBitrate) ? pInputAudio->nBitrate : AVQSV_DEFAULT_AUDIO_BITRATE) * 1000;
        }
        AddMessage(QSV_LOG_DEBUG, _T("Audio Encoder Param: %s, %dch[0x%02x], %.1fkHz, %s, %d/%d\n"), char_to_tstring(pMuxAudio->pOutCodecEncode->name).c_str(),
            pMuxAudio->pOutCodecEncodeCtx->channels, (uint32_t)pMuxAudio->pOutCodecEncodeCtx->channel_layout, pMuxAudio->pOutCodecEncodeCtx->sample_rate / 1000.0,
            char_to_tstring(av_get_sample_fmt_name(pMuxAudio->pOutCodecEncodeCtx->sample_fmt)).c_str(),
            pMuxAudio->pOutCodecEncodeCtx->pkt_timebase.num, pMuxAudio->pOutCodecEncodeCtx->pkt_timebase.den);
        if (pMuxAudio->pOutCodecEncode->capabilities & CODEC_CAP_EXPERIMENTAL) { 
            //誰がなんと言おうと使うと言ったら使うのだ
            av_opt_set(pMuxAudio->pOutCodecEncodeCtx, "strict", "experimental", 0);
        }
        if (0 > avcodec_open2(pMuxAudio->pOutCodecEncodeCtx, pMuxAudio->pOutCodecEncode, NULL)) {
            AddMessage(QSV_LOG_ERROR, errorMesForCodec(_T("failed to open encoder"), codecId));
            return MFX_ERR_NULL_PTR;
        }
        if ((!codecIDIsPCM(codecId) //PCM系のコーデックに出力するなら、sample_fmtのresampleは不要
            && pMuxAudio->pOutCodecEncodeCtx->sample_fmt   != pMuxAudio->pOutCodecDecodeCtx->sample_fmt)
             || pMuxAudio->pOutCodecEncodeCtx->sample_rate != pMuxAudio->pOutCodecDecodeCtx->sample_rate
             || pMuxAudio->pOutCodecEncodeCtx->channels    != pMuxAudio->pOutCodecDecodeCtx->channels) {
            pMuxAudio->pSwrContext = swr_alloc();
            av_opt_set_int       (pMuxAudio->pSwrContext, "in_channel_count",   pMuxAudio->pOutCodecDecodeCtx->channels,       0);
            av_opt_set_int       (pMuxAudio->pSwrContext, "in_channel_layout",  pMuxAudio->pOutCodecDecodeCtx->channel_layout, 0);
            av_opt_set_int       (pMuxAudio->pSwrContext, "in_sample_rate",     pMuxAudio->pOutCodecDecodeCtx->sample_rate,    0);
            av_opt_set_sample_fmt(pMuxAudio->pSwrContext, "in_sample_fmt",      pMuxAudio->pOutCodecDecodeCtx->sample_fmt,     0);
            av_opt_set_int       (pMuxAudio->pSwrContext, "out_channel_count",  pMuxAudio->pOutCodecEncodeCtx->channels,       0);
            av_opt_set_int       (pMuxAudio->pSwrContext, "out_channel_layout", pMuxAudio->pOutCodecEncodeCtx->channel_layout, 0);
            av_opt_set_int       (pMuxAudio->pSwrContext, "out_sample_rate",    pMuxAudio->pOutCodecEncodeCtx->sample_rate,    0);
            av_opt_set_sample_fmt(pMuxAudio->pSwrContext, "out_sample_fmt",     pMuxAudio->pOutCodecEncodeCtx->sample_fmt,     0);
            //av_opt_set           (pMuxAudio->pSwrContext, "resampler",          "sox",                                         0);

            ret = swr_init(pMuxAudio->pSwrContext);
            if (ret < 0) {
                AddMessage(QSV_LOG_ERROR, _T("Failed to initialize the resampling context: %s\n"), qsv_av_err2str(ret).c_str());
                return MFX_ERR_UNKNOWN;
            }
            pMuxAudio->nSwrBufferSize = 16384;
            if (0 > (ret = av_samples_alloc_array_and_samples(&pMuxAudio->pSwrBuffer, &pMuxAudio->nSwrBufferLinesize,
                pMuxAudio->pOutCodecEncodeCtx->channels, pMuxAudio->nSwrBufferSize, pMuxAudio->pOutCodecEncodeCtx->sample_fmt, 0))) {
                AddMessage(QSV_LOG_ERROR, _T("Failed to allocate buffer for resampling: %s\n"), qsv_av_err2str(ret).c_str());
                return MFX_ERR_UNKNOWN;
            }
            AddMessage(QSV_LOG_DEBUG, _T("Created audio resampler: %s, %dch, %.1fkHz -> %s, %dch, %.1fkHz\n"),
                char_to_tstring(av_get_sample_fmt_name(pMuxAudio->pOutCodecDecodeCtx->sample_fmt)).c_str(), pMuxAudio->pOutCodecDecodeCtx->channels, pMuxAudio->pOutCodecDecodeCtx->sample_rate / 1000.0,
                char_to_tstring(av_get_sample_fmt_name(pMuxAudio->pOutCodecEncodeCtx->sample_fmt)).c_str(), pMuxAudio->pOutCodecEncodeCtx->channels, pMuxAudio->pOutCodecEncodeCtx->sample_rate / 1000.0);
        }
    } else if (pMuxAudio->pCodecCtxIn->codec_id == AV_CODEC_ID_AAC && pMuxAudio->pCodecCtxIn->extradata == NULL && m_Mux.video.pStream) {
        AddMessage(QSV_LOG_DEBUG, _T("start initialize aac_adtstoasc filter...\n"));
        if (NULL == (pMuxAudio->pAACBsfc = av_bitstream_filter_init("aac_adtstoasc"))) {
            AddMessage(QSV_LOG_ERROR, _T("failed to open bitstream filter for AAC audio.\n"));
            return MFX_ERR_NULL_PTR;
        }
        if (pInputAudio->src.pktSample.data) {
            //mkvではavformat_write_headerまでにAVCodecContextにextradataをセットしておく必要がある
            AVPacket *audpkt = &pInputAudio->src.pktSample;
            if (0 > av_bitstream_filter_filter(pMuxAudio->pAACBsfc, pMuxAudio->pCodecCtxIn, NULL, &audpkt->data, &audpkt->size, audpkt->data, audpkt->size, 0)) {
                AddMessage(QSV_LOG_ERROR, _T("failed to run bitstream filter for AAC audio.\n"));
                return MFX_ERR_UNKNOWN;
            }
            AddMessage(QSV_LOG_DEBUG, _T("successfully attached packet sample from AAC\n."));
        }
    }

    //パラメータのコピー
    //下記のようにavcodec_copy_contextを使用するとavformat_write_header()が
    //Tag mp4a/0x6134706d incompatible with output codec id '86018' ([64][0][0][0])のようなエラーを出すことがある
    //そのため、必要な値だけをひとつづつコピーする
    //avcodec_copy_context(pMuxAudio->pStream->codec, srcCodecCtx);
    const AVCodecContext *srcCodecCtx = (pMuxAudio->pOutCodecEncodeCtx) ? pMuxAudio->pOutCodecEncodeCtx : pInputAudio->src.pCodecCtx;
    pMuxAudio->pStream->codec->codec_type      = srcCodecCtx->codec_type;
    pMuxAudio->pStream->codec->codec_id        = srcCodecCtx->codec_id;
    pMuxAudio->pStream->codec->frame_size      = srcCodecCtx->frame_size;
    pMuxAudio->pStream->codec->channels        = srcCodecCtx->channels;
    pMuxAudio->pStream->codec->channel_layout  = srcCodecCtx->channel_layout;
    pMuxAudio->pStream->codec->ticks_per_frame = srcCodecCtx->ticks_per_frame;
    pMuxAudio->pStream->codec->sample_rate     = srcCodecCtx->sample_rate;
    pMuxAudio->pStream->codec->sample_fmt      = srcCodecCtx->sample_fmt;
    pMuxAudio->pStream->codec->block_align     = srcCodecCtx->block_align;
    if (srcCodecCtx->extradata_size) {
        AddMessage(QSV_LOG_DEBUG, _T("set extradata from stream codec...\n"));
        SetExtraData(pMuxAudio->pStream->codec, srcCodecCtx->extradata, srcCodecCtx->extradata_size);
    } else if (pMuxAudio->pCodecCtxIn->extradata_size) {
        //aac_adtstoascから得たヘッダをコピーする
        //これをしておかないと、avformat_write_headerで"Error parsing AAC extradata, unable to determine samplerate."という
        //意味不明なエラーメッセージが表示される
        AddMessage(QSV_LOG_DEBUG, _T("set extradata from original packet...\n"));
        SetExtraData(pMuxAudio->pStream->codec, pMuxAudio->pCodecCtxIn->extradata, pMuxAudio->pCodecCtxIn->extradata_size);
    }
    pMuxAudio->pStream->time_base = av_make_q(1, pMuxAudio->pStream->codec->sample_rate);
    pMuxAudio->pStream->codec->time_base = pMuxAudio->pStream->time_base;
    if (m_Mux.video.pStream) {
        pMuxAudio->pStream->start_time = (int)av_rescale_q(pInputAudio->src.nDelayOfAudio, pMuxAudio->pCodecCtxIn->pkt_timebase, pMuxAudio->pStream->time_base);
        pMuxAudio->nDelaySamplesOfAudio = (int)pMuxAudio->pStream->start_time;
        pMuxAudio->nLastPtsOut = pMuxAudio->pStream->start_time;

        AddMessage(QSV_LOG_DEBUG, _T("delay      %6d (timabase %d/%d)\n"), pInputAudio->src.nDelayOfAudio, pMuxAudio->pCodecCtxIn->pkt_timebase.num, pMuxAudio->pCodecCtxIn->pkt_timebase.den);
        AddMessage(QSV_LOG_DEBUG, _T("start_time %6d (timabase %d/%d)\n"), pMuxAudio->pStream->start_time, pMuxAudio->pStream->codec->time_base.num, pMuxAudio->pStream->codec->time_base.den);
    }
    return MFX_ERR_NONE;
}

mfxStatus CAvcodecWriter::SetChapters(const vector<const AVChapter *>& pChapterList) {
    vector<AVChapter *> outChapters;
    for (int i = 0; i < (int)pChapterList.size(); i++) {
        int64_t start = AdjustTimestampTrimmed(pChapterList[i]->start, pChapterList[i]->time_base, pChapterList[i]->time_base, true);
        int64_t end   = AdjustTimestampTrimmed(pChapterList[i]->end,   pChapterList[i]->time_base, pChapterList[i]->time_base, true);
        if (start < end) {
            AVChapter *pChap = (AVChapter *)av_mallocz(sizeof(pChap[0]));
            pChap->start     = start;
            pChap->end       = end;
            pChap->id        = pChapterList[i]->id;
            pChap->time_base = pChapterList[i]->time_base;
            av_dict_copy(&pChap->metadata, pChapterList[i]->metadata, 0);
            outChapters.push_back(pChap);
        }
    }
    if (outChapters.size() > 0) {
        m_Mux.format.pFormatCtx->nb_chapters = (uint32_t)outChapters.size();
        m_Mux.format.pFormatCtx->chapters = (AVChapter **)av_realloc_f(m_Mux.format.pFormatCtx->chapters, outChapters.size(), sizeof(m_Mux.format.pFormatCtx->chapters[0]) * outChapters.size());
        for (int i = 0; i < (int)outChapters.size(); i++) {
            m_Mux.format.pFormatCtx->chapters[i] = outChapters[i];

            AddMessage(QSV_LOG_DEBUG, _T("chapter #%d: id %d, start %I64d, end %I64d\n, timebase %d/%d\n"),
                outChapters[i]->id, outChapters[i]->start, outChapters[i]->end, outChapters[i]->time_base.num, outChapters[i]->time_base.den);
        }
    }
    return MFX_ERR_NONE;
}

mfxStatus CAvcodecWriter::Init(const msdk_char *strFileName, const void *option, CEncodeStatusInfo *pEncSatusInfo) {
    m_Mux.format.bStreamError = true;
    AvcodecWriterPrm *prm = (AvcodecWriterPrm *)option;

    if (!check_avcodec_dll()) {
        AddMessage(QSV_LOG_ERROR, error_mes_avcodec_dll_not_found());
        return MFX_ERR_NULL_PTR;
    }

    std::string filename;
    if (0 == tchar_to_string(strFileName, filename, CP_UTF8)) {
        AddMessage(QSV_LOG_ERROR, _T("failed to convert output filename to utf-8 characters.\n"));
        return MFX_ERR_NULL_PTR;
    }

    av_register_all();
    avcodec_register_all();
    av_log_set_level((m_pPrintMes->getLogLevel() == QSV_LOG_DEBUG) ?  AV_LOG_DEBUG : QSV_AV_LOG_LEVEL);

    if (prm->pOutputFormat != nullptr) {
        AddMessage(QSV_LOG_DEBUG, _T("output format specified: %s\n"), prm->pOutputFormat);
    }
    AddMessage(QSV_LOG_DEBUG, _T("output filename: \"%s\"\n"), strFileName);
    if (NULL == (m_Mux.format.pOutputFmt = av_guess_format((prm->pOutputFormat) ? tchar_to_string(prm->pOutputFormat).c_str() : NULL, filename.c_str(), NULL))) {
        AddMessage(QSV_LOG_ERROR,
            _T("failed to assume format from output filename.\n")
            _T("please set proper extension for output file, or specify format using option %s.\n"), (prm->pVideoInfo) ? _T("--format") : _T("--audio-file <format>:<filename>"));
        if (prm->pOutputFormat != nullptr) {
            AddMessage(QSV_LOG_ERROR, _T("Please use --check-formats to check available formats.\n"));
        }
        return MFX_ERR_NULL_PTR;
    }
    m_Mux.format.pFormatCtx = avformat_alloc_context();
    m_Mux.format.pFormatCtx->oformat = m_Mux.format.pOutputFmt;
    m_Mux.format.bIsMatroska = 0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "matroska");
    m_Mux.format.bIsPipe = (0 == strcmp(filename.c_str(), "-")) || filename.c_str() == strstr(filename.c_str(), R"(\\.\pipe\)");

    if (m_Mux.format.bIsPipe) {
        AddMessage(QSV_LOG_DEBUG, _T("output is pipe\n"));
        if (_setmode(_fileno(stdout), _O_BINARY) < 0) {
            AddMessage(QSV_LOG_ERROR, _T("failed to switch stdout to binary mode.\n"));
            return MFX_ERR_UNKNOWN;
        }
        if (0 == strcmp(filename.c_str(), "-")) {
            m_bOutputIsStdout = true;
            filename = "pipe:1";
            AddMessage(QSV_LOG_DEBUG, _T("output is set to stdout\n"));
        } else if (m_pPrintMes->getLogLevel() == QSV_LOG_DEBUG) {
            AddMessage(QSV_LOG_DEBUG, _T("file name is %sunc path.\n"), (PathIsUNC(strFileName)) ? _T("") : _T("not "));
            if (PathFileExists(strFileName)) {
                AddMessage(QSV_LOG_DEBUG, _T("file already exists and will overwrite.\n"));
            }
        }
        int err;
        if (0 > (err = avio_open2(&m_Mux.format.pFormatCtx->pb, filename.c_str(), AVIO_FLAG_WRITE, NULL, NULL))) {
            AddMessage(QSV_LOG_ERROR, _T("failed to avio_open2 file \"%s\": %s\n"), char_to_tstring(filename, CP_UTF8).c_str(), qsv_av_err2str(err).c_str());
            return MFX_ERR_NULL_PTR; // Couldn't open file
        }
        AddMessage(QSV_LOG_DEBUG, _T("Opened file \"%s\".\n"), char_to_tstring(filename, CP_UTF8).c_str());
    } else {
        m_Mux.format.nAVOutBufferSize = 1024 * 1024;
        m_Mux.format.nOutputBufferSize = 16 * 1024 * 1024;
        if (prm->pVideoInfo) {
            m_Mux.format.nAVOutBufferSize *= 8;
            m_Mux.format.nOutputBufferSize *= 4;
        }

        if (NULL == (m_Mux.format.pAVOutBuffer = (mfxU8 *)av_malloc(m_Mux.format.nAVOutBufferSize))) {
            AddMessage(QSV_LOG_ERROR, _T("failed to allocate muxer buffer of %d MB.\n"), m_Mux.format.nAVOutBufferSize / (1024 * 1024));
            return MFX_ERR_NULL_PTR;
        }
        AddMessage(QSV_LOG_DEBUG, _T("allocated internal buffer %d MB.\n"), m_Mux.format.nAVOutBufferSize / (1024 * 1024));

        errno_t error;
        if (0 != (error = _tfopen_s(&m_Mux.format.fpOutput, strFileName, _T("wb"))) || m_Mux.format.fpOutput == NULL) {
            AddMessage(QSV_LOG_ERROR, _T("failed to open %soutput file \"%s\": %s.\n"), (prm->pVideoInfo) ? _T("") : _T("audio "), strFileName, _tcserror(error));
            return MFX_ERR_NULL_PTR; // Couldn't open file
        }
        //確保できなかったら、サイズを小さくして再度確保を試みる (最終的に1MBも確保できなかったら諦める)
        for (; m_Mux.format.nOutputBufferSize >= 1024 * 1024; m_Mux.format.nOutputBufferSize >>= 1) {
            if (NULL != (m_Mux.format.pOutputBuffer = (char *)malloc(m_Mux.format.nOutputBufferSize))) {
                setvbuf(m_Mux.format.fpOutput, m_Mux.format.pOutputBuffer, _IOFBF, m_Mux.format.nOutputBufferSize);
                AddMessage(QSV_LOG_DEBUG, _T("set external output buffer %d MB.\n"), m_Mux.format.nOutputBufferSize / (1024 * 1024));
                break;
            }
        }

        if (NULL == (m_Mux.format.pFormatCtx->pb = avio_alloc_context(m_Mux.format.pAVOutBuffer, m_Mux.format.nAVOutBufferSize, 1, this, funcReadPacket, funcWritePacket, funcSeek))) {
            AddMessage(QSV_LOG_ERROR, _T("failed to alloc avio context.\n"));
            return MFX_ERR_NULL_PTR;
        }
    }

    m_Mux.trim = prm->trimList;

    if (prm->pVideoInfo) {
        mfxStatus sts = InitVideo(prm);
        if (sts != MFX_ERR_NONE) {
            return sts;
        }
        AddMessage(QSV_LOG_DEBUG, _T("Initialized video output.\n"));
    }
    
    const int audioStreamCount = (int)prm->inputAudioList.size();
    if (audioStreamCount) {
        m_Mux.audio.resize(audioStreamCount, { 0 });
        for (int i = 0; i < audioStreamCount; i++) {
            mfxStatus sts = InitAudio(&m_Mux.audio[i], &prm->inputAudioList[i]);
            if (sts != MFX_ERR_NONE) {
                return sts;
            }
            AddMessage(QSV_LOG_DEBUG, _T("Initialized audio output - %d.\n"), i);
        }
    }

    SetChapters(prm->chapterList);
    
    sprintf_s(m_Mux.format.pFormatCtx->filename, filename.c_str());
    if (m_Mux.format.pOutputFmt->flags & AVFMT_GLOBALHEADER) {
        if (m_Mux.video.pStream) { m_Mux.video.pStream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER; }
        for (auto audio : m_Mux.audio) {
            if (audio.pStream) { audio.pStream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER; }
        }
    }

    m_pEncSatusInfo = pEncSatusInfo;
    //音声のみの出力を行う場合、SetVideoParamは呼ばれないので、ここで最後まで初期化をすませてしまう
    if (!m_Mux.video.pStream) {
        return SetVideoParam(NULL, NULL);
    }
    return MFX_ERR_NONE;
}

mfxStatus CAvcodecWriter::SetSPSPPSToExtraData(const mfxVideoParam *pMfxVideoPrm) {
    //SPS/PPSをセット
    if (m_Mux.video.pStream) {
        mfxExtCodingOptionSPSPPS *pSpsPPS = NULL;
        for (int iExt = 0; iExt < pMfxVideoPrm->NumExtParam; iExt++) {
            if (pMfxVideoPrm->ExtParam[iExt]->BufferId == MFX_EXTBUFF_CODING_OPTION_SPSPPS) {
                pSpsPPS = (mfxExtCodingOptionSPSPPS *)(pMfxVideoPrm->ExtParam[iExt]);
                break;
            }
        }
        if (pSpsPPS) {
            m_Mux.video.pCodecCtx->extradata_size = pSpsPPS->SPSBufSize + pSpsPPS->PPSBufSize;
            m_Mux.video.pCodecCtx->extradata = (mfxU8 *)av_malloc(m_Mux.video.pCodecCtx->extradata_size);
            memcpy(m_Mux.video.pCodecCtx->extradata,                       pSpsPPS->SPSBuffer, pSpsPPS->SPSBufSize);
            memcpy(m_Mux.video.pCodecCtx->extradata + pSpsPPS->SPSBufSize, pSpsPPS->PPSBuffer, pSpsPPS->PPSBufSize);
            AddMessage(QSV_LOG_DEBUG, _T("copied video header from QSV encoder.\n"));
        } else {
            AddMessage(QSV_LOG_ERROR, _T("failed to get video header from QSV encoder.\n"));
            return MFX_ERR_UNKNOWN;
        }
        m_Mux.video.bIsPAFF = 0 != (pMfxVideoPrm->mfx.FrameInfo.PicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF));
        if (m_Mux.video.bIsPAFF) {
            AddMessage(QSV_LOG_DEBUG, _T("output is PAFF.\n"));
        }
    }
    return MFX_ERR_NONE;
}

//extradataにHEVCのヘッダーを追加する
mfxStatus CAvcodecWriter::AddHEVCHeaderToExtraData(const mfxBitstream *pMfxBitstream) {
    mfxU8 *ptr = pMfxBitstream->Data;
    mfxU8 *vps_start_ptr = nullptr;
    mfxU8 *vps_fin_ptr = nullptr;
    const int i_fin = pMfxBitstream->DataOffset + pMfxBitstream->DataLength - 3;
    for (int i = pMfxBitstream->DataOffset; i < i_fin; i++) {
        if (ptr[i+0] == 0 && ptr[i+1] == 0 && ptr[i+2] == 1) {
            mfxU8 nalu_type = (ptr[i+3] & 0x7f) >> 1;
            if (nalu_type == 32 && vps_start_ptr == nullptr) {
                vps_start_ptr = ptr + i - (i > 0 && ptr[i-1] == 0);
                i += 3;
            } else if (nalu_type != 32 && vps_start_ptr && vps_fin_ptr == nullptr) {
                vps_fin_ptr = ptr + i - (i > 0 && ptr[i-1] == 0);
                break;
            }
        }
    }
    if (vps_fin_ptr == nullptr) {
        vps_fin_ptr = ptr + pMfxBitstream->DataOffset + pMfxBitstream->DataLength;
    }
    if (vps_start_ptr) {
        const mfxU32 vps_length = (mfxU32)(vps_fin_ptr - vps_start_ptr);
        mfxU8 *new_ptr = (mfxU8 *)av_malloc(m_Mux.video.pCodecCtx->extradata_size + vps_length);
        memcpy(new_ptr, vps_start_ptr, vps_length);
        memcpy(new_ptr + vps_length, m_Mux.video.pCodecCtx->extradata, m_Mux.video.pCodecCtx->extradata_size);
        m_Mux.video.pCodecCtx->extradata_size += vps_length;
        av_free(m_Mux.video.pCodecCtx->extradata);
        m_Mux.video.pCodecCtx->extradata = new_ptr;
    }
    return MFX_ERR_NONE;
}

mfxStatus CAvcodecWriter::WriteFileHeader(const mfxVideoParam *pMfxVideoPrm, const mfxExtCodingOption2 *cop2, const mfxBitstream *pMfxBitstream) {
    if ((m_Mux.video.pCodecCtx && m_Mux.video.pCodecCtx->codec_id == AV_CODEC_ID_HEVC) && pMfxBitstream) {
        mfxStatus sts = AddHEVCHeaderToExtraData(pMfxBitstream);
        if (sts != MFX_ERR_NONE) {
            return sts;
        }
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
        AddMessage(QSV_LOG_DEBUG, _T("set format brand \"mp42\".\n"));
    }

    //なんらかの問題があると、ここでよく死ぬ
    int ret = 0;
    if (0 > (ret = avformat_write_header(m_Mux.format.pFormatCtx, &avdict))) {
        AddMessage(QSV_LOG_ERROR, _T("failed to write header for output file: %s\n"), qsv_av_err2str(ret).c_str());
        if (avdict) av_dict_free(&avdict);
        return MFX_ERR_UNKNOWN;
    }
    //不正なオプションを渡していないかチェック
    for (const AVDictionaryEntry *t = NULL; NULL != (t = av_dict_get(avdict, "", t, AV_DICT_IGNORE_SUFFIX));) {
        AddMessage(QSV_LOG_ERROR, _T("Unknown option to muxer: ") + char_to_tstring(t->key) + _T("\n"));
        return MFX_ERR_UNKNOWN;
    }
    if (avdict) {
        av_dict_free(&avdict);
    }
    m_Mux.format.bFileHeaderWritten = true;

    av_dump_format(m_Mux.format.pFormatCtx, 0, m_Mux.format.pFormatCtx->filename, 1);

    //frame_sizeを表示
    for (const auto& audio : m_Mux.audio) {
        if (audio.pOutCodecDecodeCtx || audio.pOutCodecEncodeCtx) {
            tstring audioFrameSize = strsprintf(_T("audio track #%d:"), audio.nInTrackId);
            if (audio.pOutCodecDecodeCtx) {
                audioFrameSize += strsprintf(_T(" %s frame_size %d sample/byte"), char_to_tstring(audio.pOutCodecDecode->name).c_str(), audio.pOutCodecDecodeCtx->frame_size);
            }
            if (audio.pOutCodecEncodeCtx) {
                audioFrameSize += strsprintf(_T(" -> %s frame_size %d sample/byte"), char_to_tstring(audio.pOutCodecEncode->name).c_str(), audio.pOutCodecEncodeCtx->frame_size);
            }
            AddMessage(QSV_LOG_DEBUG, audioFrameSize);
        }
    }

    //API v1.6以下でdtsがQSVが提供されない場合、自前で計算する必要がある
    //API v1.6ではB-pyramidが存在しないので、Bフレームがあるかないかだけ考慮するればよい
    if (m_Mux.video.pStream) {
        if (m_Mux.video.bDtsUnavailable) {
            m_Mux.video.nFpsBaseNextDts = (0 - (pMfxVideoPrm->mfx.GopRefDist > 0) - (cop2->BRefType == MFX_B_REF_PYRAMID)) * (1 + m_Mux.video.bIsPAFF);
            AddMessage(QSV_LOG_DEBUG, _T("calc dts, first dts %d x (timebase).\n"), m_Mux.video.nFpsBaseNextDts);
        }
    }
    return MFX_ERR_NONE;
}

mfxStatus CAvcodecWriter::SetVideoParam(const mfxVideoParam *pMfxVideoPrm, const mfxExtCodingOption2 *cop2) {
    mfxStatus sts = SetSPSPPSToExtraData(pMfxVideoPrm);
    if (sts != MFX_ERR_NONE) {
        return sts;
    }

    if (pMfxVideoPrm) m_Mux.video.mfxParam = *pMfxVideoPrm;
    if (cop2) m_Mux.video.mfxCop2 = *cop2;

    if (m_Mux.video.pCodecCtx == nullptr || m_Mux.video.pCodecCtx->codec_id != AV_CODEC_ID_HEVC) {
        if (MFX_ERR_NONE != (sts = WriteFileHeader(pMfxVideoPrm, cop2, nullptr))) {
            return sts;
        }
    }

    tstring mes = GetWriterMes();
    AddMessage(QSV_LOG_DEBUG, mes);
    m_strOutputInfo += mes;
    m_Mux.format.bStreamError = false;

    m_bInited = true;

    return MFX_ERR_NONE;
}

int64_t CAvcodecWriter::AdjustTimestampTrimmed(int64_t nTimeIn, AVRational timescaleIn, AVRational timescaleOut, bool lastValidFrame) {
    AVRational timescaleFps = av_inv_q(m_Mux.video.nFPS);
    const int vidFrameIdx = (int)av_rescale_q(nTimeIn, timescaleIn, timescaleFps);
    int cutFrames = 0;
    if (m_Mux.trim.size()) {
        int nLastFinFrame = 0;
        for (const auto& trim : m_Mux.trim) {
            if (vidFrameIdx < trim.start) {
                if (lastValidFrame) {
                    cutFrames += (vidFrameIdx - nLastFinFrame);
                    nLastFinFrame = vidFrameIdx;
                    break;
                }
                return AV_NOPTS_VALUE;
            }
            cutFrames += trim.start - nLastFinFrame;
            if (vidFrameIdx <= trim.fin) {
                nLastFinFrame = vidFrameIdx;
                break;
            }
            nLastFinFrame = trim.fin;
        }
        cutFrames += vidFrameIdx - nLastFinFrame;
    }
    int64_t tsTimeOut = av_rescale_q(nTimeIn,   timescaleIn,  timescaleOut);
    int64_t tsTrim    = av_rescale_q(cutFrames, timescaleFps, timescaleOut);
    return tsTimeOut - tsTrim;
}

tstring CAvcodecWriter::GetWriterMes() {
    int iStream = 0;
    std::string mes = "avcodec writer: ";
    if (m_Mux.video.pStream) {
        mes += strsprintf("%s", avcodec_get_name(m_Mux.video.pStream->codec->codec_id), m_Mux.video.pStream->codec->width, m_Mux.video.pStream->codec->height);
        iStream++;
    }
    for (const auto& audioStream : m_Mux.audio) {
        if (audioStream.pStream) {
            if (iStream) {
                mes += ", ";
            }
            if (audioStream.pOutCodecEncodeCtx) {
                mes += strsprintf("(%s -> %s", audioStream.pOutCodecDecode->name, audioStream.pOutCodecEncode->name);
            } else {
                mes += strsprintf("%s", avcodec_get_name(audioStream.pStream->codec->codec_id));
            }
            if (audioStream.pOutCodecEncodeCtx) {
                mes += strsprintf(", %dkbps)", audioStream.pOutCodecEncodeCtx->bit_rate / 1000);
            }
            iStream++;
        }
    }
    if (m_Mux.format.pFormatCtx->nb_chapters > 0) {
        mes += ", chap";
    }
    mes += " -> ";
    mes += m_Mux.format.pFormatCtx->oformat->name;
    return char_to_tstring(mes.c_str());
}

mfxU32 CAvcodecWriter::getH264PAFFFieldLength(mfxU8 *ptr, mfxU32 size) {
    int sliceNalu = 0;
    mfxU8 a = ptr[0], b = ptr[1], c = ptr[2], d = 0;
    for (mfxU32 i = 3; i < size; i++) {
        d = ptr[i];
        if (((a | b) == 0) & (c == 1)) {
            if (sliceNalu) {
                return i-3-(ptr[i-4]==0)+1;
            }
            int nalType = d & 0x1F;
            sliceNalu += ((nalType == 1) | (nalType == 5));
        }
        a = b, b = c, c = d;
    }
    return size;
}

mfxStatus CAvcodecWriter::WriteNextFrame(mfxBitstream *pMfxBitstream) {
    if (!m_Mux.format.bFileHeaderWritten) {
        //HEVCエンコードでは、DecodeTimeStampが正しく設定されない
        if (pMfxBitstream->DecodeTimeStamp == MFX_TIMESTAMP_UNKNOWN) {
            m_Mux.video.bDtsUnavailable = true;
        }
        mfxStatus sts = WriteFileHeader(&m_Mux.video.mfxParam, &m_Mux.video.mfxCop2, pMfxBitstream);
        if (sts != MFX_ERR_NONE) {
            return sts;
        }
    }

    const int bIsPAFF = !!m_Mux.video.bIsPAFF;
    for (mfxU32 i = 0, frameSize = pMfxBitstream->DataLength; frameSize > 0; i++) {
        const mfxU32 bytesToWrite = (bIsPAFF) ? getH264PAFFFieldLength(pMfxBitstream->Data + pMfxBitstream->DataOffset, frameSize) : frameSize;
        AVPacket pkt = { 0 };
        av_init_packet(&pkt);
        av_new_packet(&pkt, bytesToWrite);
        memcpy(pkt.data, pMfxBitstream->Data + pMfxBitstream->DataOffset, bytesToWrite);
        pkt.size = bytesToWrite;

        const AVRational fpsTimebase = av_div_q({1, 1 + bIsPAFF}, m_Mux.video.nFPS);
        const AVRational streamTimebase = m_Mux.video.pStream->codec->pkt_timebase;
        pkt.stream_index = m_Mux.video.pStream->index;
        pkt.flags        = !!(pMfxBitstream->FrameType & (MFX_FRAMETYPE_IDR << (i<<3)));
        pkt.duration     = (int)av_rescale_q(1, fpsTimebase, streamTimebase);
        pkt.pts          = av_rescale_q(av_rescale_q(pMfxBitstream->TimeStamp, QSV_NATIVE_TIMEBASE, fpsTimebase), fpsTimebase, streamTimebase) + bIsPAFF * i * pkt.duration;
        if (!m_Mux.video.bDtsUnavailable) {
            pkt.dts = av_rescale_q(av_rescale_q(pMfxBitstream->DecodeTimeStamp, QSV_NATIVE_TIMEBASE, fpsTimebase), fpsTimebase, streamTimebase) + bIsPAFF * i * pkt.duration;
        } else {
            pkt.dts = av_rescale_q(m_Mux.video.nFpsBaseNextDts, fpsTimebase, streamTimebase);
            m_Mux.video.nFpsBaseNextDts++;
        }
        m_Mux.format.bStreamError |= 0 != av_interleaved_write_frame(m_Mux.format.pFormatCtx, &pkt);

        frameSize -= bytesToWrite;
        pMfxBitstream->DataOffset += bytesToWrite;
    }
    m_pEncSatusInfo->SetOutputData(pMfxBitstream->DataLength, pMfxBitstream->FrameType);
    pMfxBitstream->DataLength = 0;
    pMfxBitstream->DataOffset = 0;
    return (m_Mux.format.bStreamError) ? MFX_ERR_UNKNOWN : MFX_ERR_NONE;
}

vector<int> CAvcodecWriter::GetAudioStreamIndex() {
    vector<int> audioStreamIndexes;
    audioStreamIndexes.reserve(m_Mux.audio.size());
    for (auto audio : m_Mux.audio) {
        audioStreamIndexes.push_back(audio.nStreamIndexIn);
    }
    return std::move(audioStreamIndexes);
}

AVMuxAudio *CAvcodecWriter::getAudioPacketStreamData(const AVPacket *pkt) {
    const int streamIndex = pkt->stream_index;
    //privには、trackIdへのポインタが格納してある…はず
    const int inTrackId = (pkt->priv) ? *(int *)(pkt->priv) : -1;
    for (int i = 0; i < (int)m_Mux.audio.size(); i++) {
        //streamIndexの一致とtrackIdの一致を確認する
        if (m_Mux.audio[i].nStreamIndexIn == streamIndex
            && (inTrackId < 0 || m_Mux.audio[i].nInTrackId == inTrackId)) {
            return &m_Mux.audio[i];
        }
    }
    return NULL;
}

void CAvcodecWriter::applyBitstreamFilterAAC(AVPacket *pkt, AVMuxAudio *pMuxAudio) {
    uint8_t *data = NULL;
    int dataSize = 0;
    //毎回bitstream filterを初期化して、extradataに新しいヘッダを供給する
    //動画とmuxする際に必須
    av_bitstream_filter_close(pMuxAudio->pAACBsfc);
    pMuxAudio->pAACBsfc = av_bitstream_filter_init("aac_adtstoasc");
    if (0 > av_bitstream_filter_filter(pMuxAudio->pAACBsfc, pMuxAudio->pCodecCtxIn,
        nullptr, &data, &dataSize, pkt->data, pkt->size, 0)) {
        m_Mux.format.bStreamError = (pMuxAudio->nPacketWritten > 1);
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

void CAvcodecWriter::WriteNextPacket(AVMuxAudio *pMuxAudio, AVPacket *pkt, int samples) {
    AVRational samplerate = { 1, pMuxAudio->pCodecCtxIn->sample_rate };
    if (samples) {
        //durationについて、sample数から出力ストリームのtimebaseに変更する
        pkt->stream_index = pMuxAudio->pStream->index;
        pkt->flags        = AV_PKT_FLAG_KEY;
        pkt->dts          = av_rescale_q(pMuxAudio->nOutputSamples + pMuxAudio->nDelaySamplesOfAudio, samplerate, pMuxAudio->pStream->time_base);
        pkt->pts          = pkt->dts;
        pkt->duration     = (int)(pkt->pts - pMuxAudio->nLastPtsOut);
        if (pkt->duration == 0)
            pkt->duration = (int)av_rescale_q(samples, samplerate, pMuxAudio->pStream->time_base);
        pMuxAudio->nLastPtsOut = pkt->pts;
        m_Mux.format.bStreamError |= 0 != av_interleaved_write_frame(m_Mux.format.pFormatCtx, pkt);
        pMuxAudio->nOutputSamples += samples;
    } else {
        //av_interleaved_write_frameに渡ったパケットは開放する必要がないが、
        //それ以外は解放してやる必要がある
        av_free_packet(pkt);
    }
}

AVFrame *CAvcodecWriter::AudioDecodePacket(AVMuxAudio *pMuxAudio, const AVPacket *pkt, int *got_result) {
    const AVPacket *pktIn = pkt;
    if (pMuxAudio->OutPacket.size != 0) {
        int currentSize = pMuxAudio->OutPacket.size;
        if (pMuxAudio->OutPacket.buf->size < currentSize + pkt->size) {
            av_grow_packet(&pMuxAudio->OutPacket, currentSize + pkt->size);
        }
        memcpy(pMuxAudio->OutPacket.data + currentSize, pkt->data, pkt->size);
        pMuxAudio->OutPacket.size = currentSize + pkt->size;
        pktIn = &pMuxAudio->OutPacket;
        av_packet_copy_props(&pMuxAudio->OutPacket, pkt);
    }
    AVFrame *decodedFrame = av_frame_alloc();
    *got_result = FALSE;
    while (!got_result || decodedFrame->nb_samples == 0) {
        int len = avcodec_decode_audio4(pMuxAudio->pOutCodecDecodeCtx, decodedFrame, got_result, pktIn);
        if (len < 0) {
            AddMessage(QSV_LOG_ERROR, _T("avcodec writer: failed to decode audio #%d: %s\n"), pMuxAudio->nInTrackId, qsv_av_err2str(len).c_str());
            m_Mux.format.bStreamError = true;
        } else if (pktIn->size != len) {
            int newLen = pktIn->size - len;
            memmove(pMuxAudio->OutPacket.data, pktIn->data + len, newLen);
            pMuxAudio->OutPacket.size = newLen;
            pktIn = &pMuxAudio->OutPacket;
        } else {
            pMuxAudio->OutPacket.size = 0;
            break;
        }
        if (pMuxAudio->pOutCodecDecodeCtx->block_align > 0
            && pMuxAudio->OutPacket.size < pMuxAudio->pOutCodecDecodeCtx->block_align) {
            break;
        }
    }
    return decodedFrame;
}

//音声をresample
int CAvcodecWriter::AudioResampleFrame(AVMuxAudio *pMuxAudio, AVFrame **frame) {
    int ret = 0;
    if (pMuxAudio->pSwrContext) {
        const uint32_t dst_nb_samples = (uint32_t)av_rescale_rnd(
            swr_get_delay(pMuxAudio->pSwrContext, pMuxAudio->pOutCodecEncodeCtx->sample_rate) + ((*frame) ? (*frame)->nb_samples : 0),
            pMuxAudio->pOutCodecEncodeCtx->sample_rate, pMuxAudio->pOutCodecEncodeCtx->sample_rate, AV_ROUND_UP);
        if (dst_nb_samples > 0) {
            if (dst_nb_samples > pMuxAudio->nSwrBufferSize) {
                av_free(pMuxAudio->pSwrBuffer[0]);
                av_samples_alloc(pMuxAudio->pSwrBuffer, &pMuxAudio->nSwrBufferLinesize,
                    pMuxAudio->pOutCodecEncodeCtx->channels, dst_nb_samples * 2, pMuxAudio->pOutCodecEncodeCtx->sample_fmt, 0);
                pMuxAudio->nSwrBufferSize = dst_nb_samples * 2;
            }
            if (0 > (ret = swr_convert(pMuxAudio->pSwrContext,
                pMuxAudio->pSwrBuffer, dst_nb_samples,
                (*frame) ? (const uint8_t **)(*frame)->data : nullptr,
                (*frame) ? (*frame)->nb_samples : 0))) {
                AddMessage(QSV_LOG_ERROR, _T("avcodec writer: failed to convert sample format #%d: %s\n"), pMuxAudio->nInTrackId, qsv_av_err2str(ret).c_str());
                m_Mux.format.bStreamError = true;
            }
            if (*frame) {
                av_frame_free(frame);
            }

            if (ret >= 0 && dst_nb_samples > 0) {
                AVFrame *pResampledFrame        = av_frame_alloc();
                pResampledFrame->nb_samples     = ret;
                pResampledFrame->channels       = pMuxAudio->pOutCodecEncodeCtx->channels;
                pResampledFrame->channel_layout = pMuxAudio->pOutCodecEncodeCtx->channel_layout;
                pResampledFrame->sample_rate    = pMuxAudio->pOutCodecEncodeCtx->sample_rate;
                pResampledFrame->format         = pMuxAudio->pOutCodecEncodeCtx->sample_fmt;
                av_frame_get_buffer(pResampledFrame, 32); //format, channel_layout, nb_samplesを埋めて、av_frame_get_buffer()により、メモリを確保する
                const int bytes_per_sample = av_get_bytes_per_sample(pMuxAudio->pOutCodecEncodeCtx->sample_fmt)
                    * (av_sample_fmt_is_planar(pMuxAudio->pOutCodecEncodeCtx->sample_fmt) ? 1 : pMuxAudio->pOutCodecEncodeCtx->channels);
                const int channel_loop_count = av_sample_fmt_is_planar(pMuxAudio->pOutCodecEncodeCtx->sample_fmt) ? pMuxAudio->pOutCodecEncodeCtx->channels : 1;
                for (int i = 0; i < channel_loop_count; i++) {
                    memcpy(pResampledFrame->data[i], pMuxAudio->pSwrBuffer[i], pResampledFrame->nb_samples * bytes_per_sample);
                }
                (*frame) = pResampledFrame;
            }
        }
    }
    return ret;
}

//音声をエンコード
int CAvcodecWriter::AudioEncodeFrame(AVMuxAudio *pMuxAudio, AVPacket *pEncPkt, const AVFrame *frame, int *got_result) {
    av_init_packet(pEncPkt);
    int samples = 0;
    int ret = avcodec_encode_audio2(pMuxAudio->pOutCodecEncodeCtx, pEncPkt, frame, got_result);
    if (ret < 0) {
        AddMessage(QSV_LOG_ERROR, _T("avcodec writer: failed to encode audio #%d: %s\n"), pMuxAudio->nInTrackId, qsv_av_err2str(ret).c_str());
        m_Mux.format.bStreamError = true;
    } else if (*got_result) {
        samples = (int)av_rescale_q(pEncPkt->duration, pMuxAudio->pOutCodecEncodeCtx->pkt_timebase, { 1, pMuxAudio->pCodecCtxIn->sample_rate });
    }
    return samples;
}

void CAvcodecWriter::AudioFlushStream(AVMuxAudio *pMuxAudio) {
    while (pMuxAudio->pOutCodecDecodeCtx) {
        int samples = 0;
        int got_result = 0;
        AVPacket pkt = { 0 };
        AVFrame *decodedFrame = AudioDecodePacket(pMuxAudio, &pkt, &got_result);
        if (!got_result && decodedFrame != nullptr) {
            break;
        }
        if (0 == AudioResampleFrame(pMuxAudio, &decodedFrame)) {
            samples = AudioEncodeFrame(pMuxAudio, &pkt, decodedFrame, &got_result);
        }
        if (decodedFrame != nullptr) {
            av_frame_free(&decodedFrame);
        }
        WriteNextPacket(pMuxAudio, &pkt, samples);
    }
    while (pMuxAudio->pSwrContext) {
        int samples = 0;
        int got_result = 0;
        AVPacket pkt = { 0 };
        AVFrame *decodedFrame = nullptr;
        if (0 != AudioResampleFrame(pMuxAudio, &decodedFrame) || decodedFrame == nullptr) {
            break;
        }
        samples = AudioEncodeFrame(pMuxAudio, &pkt, decodedFrame, &got_result);
        WriteNextPacket(pMuxAudio, &pkt, samples);
    }
    while (pMuxAudio->pOutCodecEncodeCtx) {
        int got_result = 0;
        AVPacket pkt = { 0 };
        int samples = AudioEncodeFrame(pMuxAudio, &pkt, nullptr, &got_result);
        if (samples == 0)
            break;
        WriteNextPacket(pMuxAudio, &pkt, samples);
    }
}

mfxStatus CAvcodecWriter::WriteNextPacket(AVPacket *pkt) {
    if (!m_Mux.format.bFileHeaderWritten) {
        //まだフレームヘッダーが書かれていなければ、パケットをキャッシュして終了
        m_AudPktBufFileHead.push_back(pkt);
        return MFX_ERR_NONE;
    }
    //m_AudPktBufFileHeadにキャッシュしてあるパケットかどうかを調べる
    if (m_AudPktBufFileHead.end() == std::find(m_AudPktBufFileHead.begin(), m_AudPktBufFileHead.end(), pkt)) {
        //キャッシュしてあるパケットでないなら、キャッシュしてあるパケットをまず処理する
        for (auto bufPkt : m_AudPktBufFileHead) {
            mfxStatus sts = WriteNextPacket(bufPkt);
            if (sts != MFX_ERR_NONE) {
                return sts;
            }
        }
        //キャッシュをすべて書き出したらクリア
        m_AudPktBufFileHead.clear();
    }

    if (pkt == nullptr) {
        for (uint32_t i = 0; i < m_Mux.audio.size(); i++) {
            AudioFlushStream(&m_Mux.audio[i]);
        }
        AddMessage(QSV_LOG_DEBUG, _T("Flushed audio buffer.\n"));
        return (m_Mux.format.bStreamError) ? MFX_ERR_UNKNOWN : MFX_ERR_NONE;
    }

    int samples = 0;
    AVMuxAudio *pMuxAudio = getAudioPacketStreamData(pkt);
    if (pMuxAudio == NULL) {
        AddMessage(QSV_LOG_ERROR, _T("failed to get stream for input stream.\n"));
        m_Mux.format.bStreamError = true;
        av_free_packet(pkt);
        return MFX_ERR_NULL_PTR;
    }

    pMuxAudio->nPacketWritten++;
    
    AVRational samplerate = { 1, pMuxAudio->pCodecCtxIn->sample_rate };
    if (pMuxAudio->pAACBsfc) {
        applyBitstreamFilterAAC(pkt, pMuxAudio);
    }
    if (!pMuxAudio->pOutCodecDecodeCtx) {
        samples = (int)av_rescale_q(pkt->duration, pMuxAudio->pCodecCtxIn->pkt_timebase, samplerate);
        // 1/1000 timebaseは信じるに値しないので、frame_sizeがあればその値を使用する
        if (0 == av_cmp_q(pMuxAudio->pCodecCtxIn->pkt_timebase, { 1, 1000 })
            && pMuxAudio->pCodecCtxIn->frame_size) {
            samples = pMuxAudio->pCodecCtxIn->frame_size;
        } else {
            //このdurationから計算したsampleが信頼できるか計算する
            //mkvではたまにptsの差分とdurationが一致しないことがある
            //ptsDiffが動画の1フレーム分より小さいときのみ対象とする (カット編集によるものを混同する可能性がある)
            mfxI64 ptsDiff = pkt->pts - pMuxAudio->nLastPtsIn;
            if (0 < ptsDiff
                && ptsDiff < av_rescale_q(1, av_inv_q(m_Mux.video.nFPS), samplerate)
                && pMuxAudio->nLastPtsIn != AV_NOPTS_VALUE
                && 1 < abs(ptsDiff - pkt->duration)) {
                //ptsの差分から計算しなおす
                samples = (int)av_rescale_q(ptsDiff, pMuxAudio->pCodecCtxIn->pkt_timebase, samplerate);
            }
        }
        pMuxAudio->nLastPtsIn = pkt->pts;
        WriteNextPacket(pMuxAudio, pkt, samples);
    } else {
        int got_result = 0;
        AVFrame *decodedFrame = AudioDecodePacket(pMuxAudio, pkt, &got_result);
        if (pkt != nullptr) {
            av_free_packet(pkt);
        }
        if (got_result && decodedFrame != nullptr) {
            if (0 <= AudioResampleFrame(pMuxAudio, &decodedFrame)) {
                if (pMuxAudio->pDecodedFrameCache == nullptr && (decodedFrame->nb_samples == pMuxAudio->pOutCodecEncodeCtx->frame_size || pMuxAudio->pOutCodecEncodeCtx->frame_size == 0)) {
                    //デコードの出力サンプル数とエンコーダのframe_sizeが一致していれば、そのままエンコードする
                    samples = AudioEncodeFrame(pMuxAudio, pkt, decodedFrame, &got_result);
                    if (got_result && samples) {
                        WriteNextPacket(pMuxAudio, pkt, samples);
                    }
                } else {
                    const int bytes_per_sample = av_get_bytes_per_sample(pMuxAudio->pOutCodecEncodeCtx->sample_fmt)
                        * (av_sample_fmt_is_planar(pMuxAudio->pOutCodecEncodeCtx->sample_fmt) ? 1 : pMuxAudio->pOutCodecEncodeCtx->channels);
                    const int channel_loop_count = av_sample_fmt_is_planar(pMuxAudio->pOutCodecEncodeCtx->sample_fmt) ? pMuxAudio->pOutCodecEncodeCtx->channels : 1;
                    //それまでにたまっているキャッシュがあれば、それを結合する
                    if (pMuxAudio->pDecodedFrameCache) {
                        //pMuxAudio->pDecodedFrameCacheとdecodedFrameを結合
                        AVFrame *pCombinedFrame = av_frame_alloc();
                        pCombinedFrame->format = pMuxAudio->pOutCodecEncodeCtx->sample_fmt;
                        pCombinedFrame->channel_layout = pMuxAudio->pOutCodecEncodeCtx->channel_layout;
                        pCombinedFrame->nb_samples = decodedFrame->nb_samples + pMuxAudio->pDecodedFrameCache->nb_samples;
                        av_frame_get_buffer(pCombinedFrame, 32); //format, channel_layout, nb_samplesを埋めて、av_frame_get_buffer()により、メモリを確保する
                        for (int i = 0; i < channel_loop_count; i++) {
                            mfxU32 cachedBytes = pMuxAudio->pDecodedFrameCache->nb_samples * bytes_per_sample;
                            memcpy(pCombinedFrame->data[i], pMuxAudio->pDecodedFrameCache->data[i], cachedBytes);
                            memcpy(pCombinedFrame->data[i] + cachedBytes, decodedFrame->data[i], decodedFrame->nb_samples * bytes_per_sample);
                        }
                        //結合し終わっていらないものは破棄
                        av_frame_free(&pMuxAudio->pDecodedFrameCache);
                        av_frame_free(&decodedFrame);
                        decodedFrame = pCombinedFrame;
                    }
                    //frameにエンコーダのframe_size分だけ切り出しながら、エンコードを進める
                    AVFrame *pCutFrame = av_frame_alloc();
                    pCutFrame->format = pMuxAudio->pOutCodecEncodeCtx->sample_fmt;
                    pCutFrame->channel_layout = pMuxAudio->pOutCodecEncodeCtx->channel_layout;
                    pCutFrame->nb_samples = pMuxAudio->pOutCodecEncodeCtx->frame_size;
                    av_frame_get_buffer(pCutFrame, 32);

                    int samplesRemain = decodedFrame->nb_samples; //残りのサンプル数
                    int samplesWritten = 0; //エンコーダに渡したサンプル数
                    //残りサンプル数がframe_size未満になるまで回す
                    for (; samplesRemain >= pMuxAudio->pOutCodecEncodeCtx->frame_size;
                        samplesWritten += pMuxAudio->pOutCodecEncodeCtx->frame_size, samplesRemain -= pMuxAudio->pOutCodecEncodeCtx->frame_size) {
                        for (int i = 0; i < channel_loop_count; i++) {
                            memcpy(pCutFrame->data[i], decodedFrame->data[i] + samplesWritten * bytes_per_sample, pCutFrame->nb_samples * bytes_per_sample);
                        }
                        samples = AudioEncodeFrame(pMuxAudio, pkt, pCutFrame, &got_result);
                        if (got_result && samples) {
                            WriteNextPacket(pMuxAudio, pkt, samples);
                        }
                    }
                    if (samplesRemain) {
                        pCutFrame->nb_samples = samplesRemain;
                        for (int i = 0; i < channel_loop_count; i++) {
                            memcpy(pCutFrame->data[i], decodedFrame->data[i] + samplesWritten * bytes_per_sample, pCutFrame->nb_samples * bytes_per_sample);
                        }
                        pMuxAudio->pDecodedFrameCache = pCutFrame;
                    }
                }
            }
        }
        if (decodedFrame != nullptr) {
            av_frame_free(&decodedFrame);
        }
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
