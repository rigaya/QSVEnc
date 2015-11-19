//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#include <fcntl.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include "qsv_osdep.h"
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

void CAvcodecWriter::CloseSubtitle(AVMuxSub *pMuxSub) {
    //close decoder
    if (pMuxSub->pOutCodecDecodeCtx) {
        avcodec_close(pMuxSub->pOutCodecDecodeCtx);
        av_free(pMuxSub->pOutCodecDecodeCtx);
        AddMessage(QSV_LOG_DEBUG, _T("Closed pOutCodecDecodeCtx.\n"));
    }

    //close encoder
    if (pMuxSub->pOutCodecEncodeCtx) {
        avcodec_close(pMuxSub->pOutCodecEncodeCtx);
        av_free(pMuxSub->pOutCodecEncodeCtx);
        AddMessage(QSV_LOG_DEBUG, _T("Closed pOutCodecEncodeCtx.\n"));
    }
    if (pMuxSub->pBuf) {
        av_free(pMuxSub->pBuf);
    }

    memset(pMuxSub, 0, sizeof(pMuxSub[0]));
    AddMessage(QSV_LOG_DEBUG, _T("Closed subtitle.\n"));
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

void CAvcodecWriter::CloseQueues() {
#if ENABLE_AVCODEC_OUT_THREAD
    m_Mux.thread.bAbort = true;
    m_Mux.thread.qVideobitstream.clear();
    m_Mux.thread.qVideobitstreamFreeI.clear([](mfxBitstream *bitstream) { WipeMfxBitstream(bitstream); });
    m_Mux.thread.qVideobitstreamFreePB.clear([](mfxBitstream *bitstream) { WipeMfxBitstream(bitstream); });
    m_Mux.thread.qAudioPacket.clear();
#endif
}

void CAvcodecWriter::CloseThread() {
#if ENABLE_AVCODEC_OUT_THREAD
    m_Mux.thread.bAbort = true;
    if (m_Mux.thread.thOutput.joinable()) {
        //ここに来た時に、まだメインスレッドがループ中の可能性がある
        //その場合、SetEvent(m_Mux.thread.heEventPktAdded)を一度やるだけだと、
        //そのあとにResetEvent(m_Mux.thread.heEventPktAdded)が発生してしまい、
        //ここでスレッドが停止してしまう。
        //これを回避するため、m_Mux.thread.heEventClosingがセットされるまで、
        //SetEvent(m_Mux.thread.heEventPktAdded)を実行し続ける必要がある。
        while (WAIT_TIMEOUT == WaitForSingleObject(m_Mux.thread.heEventClosing, 100)) {
            SetEvent(m_Mux.thread.heEventPktAdded);
        }
        m_Mux.thread.thOutput.join();
        CloseEvent(m_Mux.thread.heEventPktAdded);
    }
    CloseQueues();
    m_Mux.thread.bAbort = false;
#endif
}

void CAvcodecWriter::Close() {
    AddMessage(QSV_LOG_DEBUG, _T("Closing...\n"));
    CloseThread();
    CloseFormat(&m_Mux.format);
    for (int i = 0; i < (int)m_Mux.audio.size(); i++) {
        CloseAudio(&m_Mux.audio[i]);
    }
    m_Mux.audio.clear();
    for (int i = 0; i < (int)m_Mux.sub.size(); i++) {
        CloseSubtitle(&m_Mux.sub[i]);
    }
    m_Mux.sub.clear();
    CloseVideo(&m_Mux.video);
    m_strOutputInfo.clear();
    m_pEncSatusInfo.reset();
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
    codecCtx->extradata      = (uint8_t *)av_malloc(codecCtx->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
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
        diffrate[i] = std::abs(1 - pSamplingRateList[i] / (double)nSrcSamplingRate);
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
    m_Mux.video.nInputFirstPts  = prm->nVideoInputFirstPts;
    m_Mux.video.pInputCodecCtx  = prm->pVideoInputCodecCtx;

    AddMessage(QSV_LOG_DEBUG, _T("output video stream timebase: %d/%d\n"), m_Mux.video.pStream->time_base.num, m_Mux.video.pStream->time_base.den);
    AddMessage(QSV_LOG_DEBUG, _T("bDtsUnavailable: %s\n"), (m_Mux.video.bDtsUnavailable) ? _T("on") : _T("off"));
    return MFX_ERR_NONE;
}

mfxStatus CAvcodecWriter::InitAudio(AVMuxAudio *pMuxAudio, AVOutputStreamPrm *pInputAudio) {
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
        pMuxAudio->pStream->start_time = (int)av_rescale_q(pInputAudio->src.nDelayOfStream, pMuxAudio->pCodecCtxIn->pkt_timebase, pMuxAudio->pStream->time_base);
        pMuxAudio->nDelaySamplesOfAudio = (int)pMuxAudio->pStream->start_time;
        pMuxAudio->nLastPtsOut = pMuxAudio->pStream->start_time;

        AddMessage(QSV_LOG_DEBUG, _T("delay      %6d (timabase %d/%d)\n"), pInputAudio->src.nDelayOfStream, pMuxAudio->pCodecCtxIn->pkt_timebase.num, pMuxAudio->pCodecCtxIn->pkt_timebase.den);
        AddMessage(QSV_LOG_DEBUG, _T("start_time %6d (timabase %d/%d)\n"), pMuxAudio->pStream->start_time,  pMuxAudio->pStream->codec->time_base.num, pMuxAudio->pStream->codec->time_base.den);
    }

    if (pInputAudio->src.pStream->metadata) {
        for (AVDictionaryEntry *pEntry = nullptr;
        nullptr != (pEntry = av_dict_get(pInputAudio->src.pStream->metadata, "", pEntry, AV_DICT_IGNORE_SUFFIX));) {
            av_dict_set(&pMuxAudio->pStream->metadata, pEntry->key, pEntry->value, AV_DICT_IGNORE_SUFFIX);
            AddMessage(QSV_LOG_DEBUG, _T("Copy Audio Metadata: key %s, value %s\n"), char_to_tstring(pEntry->key).c_str(), char_to_tstring(pEntry->value).c_str());
        }
        auto language_data = av_dict_get(pInputAudio->src.pStream->metadata, "language", NULL, AV_DICT_MATCH_CASE);
        if (language_data) {
            av_dict_set(&pMuxAudio->pStream->metadata, language_data->key, language_data->value, AV_DICT_IGNORE_SUFFIX);
            AddMessage(QSV_LOG_DEBUG, _T("Set Audio language: key %s, value %s\n"), char_to_tstring(language_data->key).c_str(), char_to_tstring(language_data->value).c_str());
        }
    }
    return MFX_ERR_NONE;
}

mfxStatus CAvcodecWriter::InitSubtitle(AVMuxSub *pMuxSub, AVOutputStreamPrm *pInputSubtitle) {
    AddMessage(QSV_LOG_DEBUG, _T("start initializing subtitle ouput...\n"));
    AddMessage(QSV_LOG_DEBUG, _T("output stream index %d, pkt_timebase %d/%d, trackId %d\n"),
        pInputSubtitle->src.nIndex, pInputSubtitle->src.pCodecCtx->pkt_timebase.num, pInputSubtitle->src.pCodecCtx->pkt_timebase.den, pInputSubtitle->src.nTrackId);

    if (NULL == (pMuxSub->pStream = avformat_new_stream(m_Mux.format.pFormatCtx, NULL))) {
        AddMessage(QSV_LOG_ERROR, _T("failed to create new stream for subtitle.\n"));
        return MFX_ERR_NULL_PTR;
    }

    AVCodecID codecId = pInputSubtitle->src.pCodecCtx->codec_id;
    if (   0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "mp4")
        || 0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "mov")
        || 0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "3gp")
        || 0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "3g2")
        || 0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "psp")
        || 0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "ipod")
        || 0 == strcmp(m_Mux.format.pFormatCtx->oformat->name, "f4v")) {
        if (avcodec_descriptor_get(codecId)->props & AV_CODEC_PROP_TEXT_SUB) {
            codecId = AV_CODEC_ID_MOV_TEXT;
        }
    } else if (codecId == AV_CODEC_ID_MOV_TEXT) {
        codecId = AV_CODEC_ID_ASS;
    }

    auto copy_subtitle_header = [](AVCodecContext *pDstCtx, const AVCodecContext *pSrcCtx) {
        if (pSrcCtx->subtitle_header_size) {
            pDstCtx->subtitle_header_size = pSrcCtx->subtitle_header_size;
            pDstCtx->subtitle_header = (uint8_t *)av_mallocz(pDstCtx->subtitle_header_size + AV_INPUT_BUFFER_PADDING_SIZE);
            memcpy(pDstCtx->subtitle_header, pSrcCtx->subtitle_header, pSrcCtx->subtitle_header_size);
        }
    };

    if (codecId != pInputSubtitle->src.pCodecCtx->codec_id || codecId == AV_CODEC_ID_MOV_TEXT) {
        //setup decoder
        if (NULL == (pMuxSub->pOutCodecDecode = avcodec_find_decoder(pInputSubtitle->src.pCodecCtx->codec_id))) {
            AddMessage(QSV_LOG_ERROR, errorMesForCodec(_T("failed to find decoder"), pInputSubtitle->src.pCodecCtx->codec_id));
            AddMessage(QSV_LOG_ERROR, _T("Please use --check-decoders to check available decoder.\n"));
            return MFX_ERR_NULL_PTR;
        }
        if (NULL == (pMuxSub->pOutCodecDecodeCtx = avcodec_alloc_context3(pMuxSub->pOutCodecDecode))) {
            AddMessage(QSV_LOG_ERROR, errorMesForCodec(_T("failed to get decode codec context"), pInputSubtitle->src.pCodecCtx->codec_id));
            return MFX_ERR_NULL_PTR;
        }
        //設定されていない必須情報があれば設定する
#define COPY_IF_ZERO(dst, src) { if ((dst)==0) (dst)=(src); }
        COPY_IF_ZERO(pMuxSub->pOutCodecDecodeCtx->width,  pInputSubtitle->src.pCodecCtx->width);
        COPY_IF_ZERO(pMuxSub->pOutCodecDecodeCtx->height, pInputSubtitle->src.pCodecCtx->height);
#undef COPY_IF_ZERO
        pMuxSub->pOutCodecDecodeCtx->pkt_timebase = pInputSubtitle->src.pCodecCtx->pkt_timebase;
        SetExtraData(pMuxSub->pOutCodecDecodeCtx, pInputSubtitle->src.pCodecCtx->extradata, pInputSubtitle->src.pCodecCtx->extradata_size);
        int ret;
        if (0 > (ret = avcodec_open2(pMuxSub->pOutCodecDecodeCtx, pMuxSub->pOutCodecDecode, NULL))) {
            AddMessage(QSV_LOG_ERROR, _T("failed to open decoder for %s: %s\n"),
                char_to_tstring(avcodec_get_name(pInputSubtitle->src.pCodecCtx->codec_id)).c_str(), qsv_av_err2str(ret).c_str());
            return MFX_ERR_NULL_PTR;
        }
        AddMessage(QSV_LOG_DEBUG, _T("Subtitle Decoder opened\n"));
        AddMessage(QSV_LOG_DEBUG, _T("Subtitle Decode Info: %s, %dx%d\n"), char_to_tstring(avcodec_get_name(pInputSubtitle->src.pCodecCtx->codec_id)).c_str(),
            pMuxSub->pOutCodecDecodeCtx->width, pMuxSub->pOutCodecDecodeCtx->height);

        //エンコーダを探す
        if (NULL == (pMuxSub->pOutCodecEncode = avcodec_find_encoder(codecId))) {
            AddMessage(QSV_LOG_ERROR, errorMesForCodec(_T("failed to find encoder"), codecId));
            AddMessage(QSV_LOG_ERROR, _T("Please use --check-encoders to find available encoder.\n"));
            return MFX_ERR_NULL_PTR;
        }
        AddMessage(QSV_LOG_DEBUG, _T("found encoder for codec %s for subtitle track %d\n"), char_to_tstring(pMuxSub->pOutCodecEncode->name).c_str(), pInputSubtitle->src.nTrackId);

        if (NULL == (pMuxSub->pOutCodecEncodeCtx = avcodec_alloc_context3(pMuxSub->pOutCodecEncode))) {
            AddMessage(QSV_LOG_ERROR, errorMesForCodec(_T("failed to get encode codec context"), codecId));
            return MFX_ERR_NULL_PTR;
        }
        pMuxSub->pOutCodecEncodeCtx->time_base = av_make_q(1, 1000);
        copy_subtitle_header(pMuxSub->pOutCodecEncodeCtx, pInputSubtitle->src.pCodecCtx);

        AddMessage(QSV_LOG_DEBUG, _T("Subtitle Encoder Param: %s, %dx%d\n"), char_to_tstring(pMuxSub->pOutCodecEncode->name).c_str(),
            pMuxSub->pOutCodecEncodeCtx->width, pMuxSub->pOutCodecEncodeCtx->height);
        if (pMuxSub->pOutCodecEncode->capabilities & CODEC_CAP_EXPERIMENTAL) {
            //問答無用で使うのだ
            av_opt_set(pMuxSub->pOutCodecEncodeCtx, "strict", "experimental", 0);
        }
        if (0 > (ret = avcodec_open2(pMuxSub->pOutCodecEncodeCtx, pMuxSub->pOutCodecEncode, NULL))) {
            AddMessage(QSV_LOG_ERROR, errorMesForCodec(_T("failed to open encoder"), codecId));
            AddMessage(QSV_LOG_ERROR, _T("%s\n"), qsv_av_err2str(ret).c_str());
            return MFX_ERR_NULL_PTR;
        }
        AddMessage(QSV_LOG_DEBUG, _T("Opened Subtitle Encoder Param: %s\n"), char_to_tstring(pMuxSub->pOutCodecEncode->name).c_str());
        if (nullptr == (pMuxSub->pBuf = (uint8_t *)av_malloc(SUB_ENC_BUF_MAX_SIZE))) {
            AddMessage(QSV_LOG_ERROR, _T("failed to allocate buffer memory for subtitle encoding.\n"));
            return MFX_ERR_NULL_PTR;
        }
        pMuxSub->pStream->codec->codec = pMuxSub->pOutCodecEncodeCtx->codec;
    }

    pMuxSub->nInTrackId     = pInputSubtitle->src.nTrackId;
    pMuxSub->nStreamIndexIn = pInputSubtitle->src.nIndex;
    pMuxSub->pCodecCtxIn    = pInputSubtitle->src.pCodecCtx;

    const AVCodecContext *srcCodecCtx = (pMuxSub->pOutCodecEncodeCtx) ? pMuxSub->pOutCodecEncodeCtx : pMuxSub->pCodecCtxIn;
    avcodec_get_context_defaults3(pMuxSub->pStream->codec, NULL);
    copy_subtitle_header(pMuxSub->pStream->codec, srcCodecCtx);
    SetExtraData(pMuxSub->pStream->codec, srcCodecCtx->extradata, srcCodecCtx->extradata_size);
    pMuxSub->pStream->codec->codec_type      = srcCodecCtx->codec_type;
    pMuxSub->pStream->codec->codec_id        = srcCodecCtx->codec_id;
    if (!pMuxSub->pStream->codec->codec_tag) {
        uint32_t codec_tag = 0;
        if (!m_Mux.format.pFormatCtx->oformat->codec_tag
            || av_codec_get_id(m_Mux.format.pFormatCtx->oformat->codec_tag, srcCodecCtx->codec_tag) == srcCodecCtx->codec_id
            || !av_codec_get_tag2(m_Mux.format.pFormatCtx->oformat->codec_tag, srcCodecCtx->codec_id, &codec_tag)) {
            pMuxSub->pStream->codec->codec_tag = srcCodecCtx->codec_tag;
        }
    }
    pMuxSub->pStream->codec->width           = srcCodecCtx->width;
    pMuxSub->pStream->codec->height          = srcCodecCtx->height;
    pMuxSub->pStream->time_base              = srcCodecCtx->time_base;
    pMuxSub->pStream->codec->time_base       = pMuxSub->pStream->time_base;
    pMuxSub->pStream->start_time             = 0;
    pMuxSub->pStream->codec->framerate       = srcCodecCtx->framerate;

    if (pInputSubtitle->src.nTrackId == -1) {
        pMuxSub->pStream->disposition |= AV_DISPOSITION_DEFAULT;
    }
    if (pInputSubtitle->src.pStream->metadata) {
        for (AVDictionaryEntry *pEntry = nullptr;
        nullptr != (pEntry = av_dict_get(pInputSubtitle->src.pStream->metadata, "", pEntry, AV_DICT_IGNORE_SUFFIX));) {
            av_dict_set(&pMuxSub->pStream->metadata, pEntry->key, pEntry->value, AV_DICT_IGNORE_SUFFIX);
            AddMessage(QSV_LOG_DEBUG, _T("Copy Subtitle Metadata: key %s, value %s\n"), char_to_tstring(pEntry->key).c_str(), char_to_tstring(pEntry->value).c_str());
        }
        auto language_data = av_dict_get(pInputSubtitle->src.pStream->metadata, "language", NULL, AV_DICT_MATCH_CASE);
        if (language_data) {
            av_dict_set(&pMuxSub->pStream->metadata, language_data->key, language_data->value, AV_DICT_IGNORE_SUFFIX);
            AddMessage(QSV_LOG_DEBUG, _T("Set Subtitle language: key %s, value %s\n"), char_to_tstring(language_data->key).c_str(), char_to_tstring(language_data->value).c_str());
        }
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

mfxStatus CAvcodecWriter::Init(const msdk_char *strFileName, const void *option, shared_ptr<CEncodeStatusInfo> pEncSatusInfo) {
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
#if defined(_WIN32) || defined(_WIN64)
        if (_setmode(_fileno(stdout), _O_BINARY) < 0) {
            AddMessage(QSV_LOG_ERROR, _T("failed to switch stdout to binary mode.\n"));
            return MFX_ERR_UNKNOWN;
        }
#endif //#if defined(_WIN32) || defined(_WIN64)
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

    const int audioStreamCount = (int)count_if(prm->inputStreamList.begin(), prm->inputStreamList.end(), [](AVOutputStreamPrm prm) { return prm.src.nTrackId > 0; });
    if (audioStreamCount) {
        m_Mux.audio.resize(audioStreamCount, { 0 });
        int iAudioIdx = 0;
        for (int iStream = 0; iStream < (int)prm->inputStreamList.size(); iStream++) {
            if (prm->inputStreamList[iStream].src.nTrackId > 0) {
                mfxStatus sts = InitAudio(&m_Mux.audio[iAudioIdx], &prm->inputStreamList[iStream]);
                if (sts != MFX_ERR_NONE) {
                    return sts;
                }
                AddMessage(QSV_LOG_DEBUG, _T("Initialized audio output - %d.\n"), iAudioIdx);
                iAudioIdx++;
            }
        }
    }
    const int subStreamCount = (int)count_if(prm->inputStreamList.begin(), prm->inputStreamList.end(), [](AVOutputStreamPrm prm) { return prm.src.nTrackId < 0; });
    if (subStreamCount) {
        m_Mux.sub.resize(subStreamCount, { 0 });
        int iSubIdx = 0;
        for (int iStream = 0; iStream < (int)prm->inputStreamList.size(); iStream++) {
            if (prm->inputStreamList[iStream].src.nTrackId < 0) {
                mfxStatus sts = InitSubtitle(&m_Mux.sub[iSubIdx], &prm->inputStreamList[iStream]);
                if (sts != MFX_ERR_NONE) {
                    return sts;
                }
                AddMessage(QSV_LOG_DEBUG, _T("Initialized subtitle output - %d.\n"), iSubIdx);
                iSubIdx++;
            }
        }
    }

    SetChapters(prm->chapterList);
    
    sprintf_s(m_Mux.format.pFormatCtx->filename, filename.c_str());
    if (m_Mux.format.pOutputFmt->flags & AVFMT_GLOBALHEADER) {
        if (m_Mux.video.pStream) { m_Mux.video.pStream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER; }
        for (uint32_t i = 0; i < m_Mux.audio.size(); i++) {
            if (m_Mux.audio[i].pStream) { m_Mux.audio[i].pStream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER; }
        }
        for (uint32_t i = 0; i < m_Mux.sub.size(); i++) {
            if (m_Mux.sub[i].pStream) { m_Mux.sub[i].pStream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER; }
        }
    }

    if (m_Mux.format.pFormatCtx->metadata) {
        av_dict_copy(&m_Mux.format.pFormatCtx->metadata, prm->pInputFormatMetadata, AV_DICT_DONT_OVERWRITE);
        av_dict_set(&m_Mux.format.pFormatCtx->metadata, "duration", NULL, 0);
        av_dict_set(&m_Mux.format.pFormatCtx->metadata, "creation_time", NULL, 0);
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
        mfxU8 *new_ptr = (mfxU8 *)av_malloc(m_Mux.video.pCodecCtx->extradata_size + vps_length + AV_INPUT_BUFFER_PADDING_SIZE);
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

#if ENABLE_AVCODEC_OUT_THREAD
    m_Mux.thread.bAbort = false;
    m_Mux.thread.qAudioPacket.init(3000);
    m_Mux.thread.qVideobitstream.init(1600, (size_t)(m_Mux.video.nFPS.num * 10.0 / m_Mux.video.nFPS.den + 0.5));
    m_Mux.thread.qVideobitstreamFreeI.init(100);
    m_Mux.thread.qVideobitstreamFreePB.init(1500);
    m_Mux.thread.heEventPktAdded = CreateEvent(NULL, TRUE, FALSE, NULL);
    m_Mux.thread.heEventClosing  = CreateEvent(NULL, TRUE, FALSE, NULL);
    m_Mux.thread.thOutput = std::thread(&CAvcodecWriter::WriteThreadFunc, this);
#endif
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
    for (const auto& subtitleStream : m_Mux.sub) {
        if (subtitleStream.pStream) {
            if (iStream) {
                mes += ", ";
            }
            mes += strsprintf("sub#%d", std::abs(subtitleStream.nInTrackId));
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
#if ENABLE_AVCODEC_OUT_THREAD
    mfxBitstream copyStream = { 0 };
    bool bFrameI = (pMfxBitstream->FrameType & MFX_FRAMETYPE_I) != 0;
    bool bFrameP = (pMfxBitstream->FrameType & MFX_FRAMETYPE_P) != 0;
    //IフレームかPBフレームかでサイズが大きく違うため、空きのmfxBistreamは異なるキューで管理する
    auto& qVideoQueueFree = (bFrameI) ? m_Mux.thread.qVideobitstreamFreeI : m_Mux.thread.qVideobitstreamFreePB;
    //空いているmfxBistreamを取り出す
    if (!qVideoQueueFree.front_copy_and_pop_no_lock(&copyStream) || copyStream.MaxLength < pMfxBitstream->DataLength) {
        //空いているmfxBistreamがない、あるいはそのバッファサイズが小さい場合は、領域を取り直す
        if (MFX_ERR_NONE != InitMfxBitstream(&copyStream, (bFrameI) ? pMfxBitstream->MaxLength : pMfxBitstream->DataLength * ((bFrameP) ? 2 : 6))) {
            AddMessage(QSV_LOG_ERROR, _T("Failed to allocate memory for video bitstream output buffer.\n"));
            m_Mux.format.bStreamError = true;
            return MFX_ERR_MEMORY_ALLOC;
        }
    }
    //必要な情報をコピー
    copyStream.DataFlag = pMfxBitstream->DataFlag;
    copyStream.TimeStamp = pMfxBitstream->TimeStamp;
    copyStream.DecodeTimeStamp = pMfxBitstream->DecodeTimeStamp;
    copyStream.FrameType = pMfxBitstream->FrameType;
    copyStream.DataLength = pMfxBitstream->DataLength;
    copyStream.DataOffset = 0;
    memcpy(copyStream.Data, pMfxBitstream->Data + pMfxBitstream->DataOffset, copyStream.DataLength);
    //キューに押し込む
    if (!m_Mux.thread.qVideobitstream.push(copyStream)) {
        AddMessage(QSV_LOG_ERROR, _T("Failed to allocate memory for video bitstream queue.\n"));
        m_Mux.format.bStreamError = true;
    }
    pMfxBitstream->DataLength = 0;
    pMfxBitstream->DataOffset = 0;
    SetEvent(m_Mux.thread.heEventPktAdded);
    return (m_Mux.format.bStreamError) ? MFX_ERR_UNKNOWN : MFX_ERR_NONE;
#else
    int64_t dts = 0;
    return WriteNextFrameInternal(pMfxBitstream, &dts);
#endif
}

mfxStatus CAvcodecWriter::WriteNextFrameInternal(mfxBitstream *pMfxBitstream, int64_t *pWrittenDts) {
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
        *pWrittenDts = av_rescale_q(pkt.dts, streamTimebase, QSV_NATIVE_TIMEBASE);
        m_Mux.format.bStreamError |= 0 != av_interleaved_write_frame(m_Mux.format.pFormatCtx, &pkt);

        frameSize -= bytesToWrite;
        pMfxBitstream->DataOffset += bytesToWrite;
    }
    m_pEncSatusInfo->SetOutputData(pMfxBitstream->DataLength, pMfxBitstream->FrameType);
#if ENABLE_AVCODEC_OUT_THREAD
    //確保したメモリ領域を使いまわすためにスタックに格納
    auto& qVideoQueueFree = (pMfxBitstream->FrameType & MFX_FRAMETYPE_I) ? m_Mux.thread.qVideobitstreamFreeI : m_Mux.thread.qVideobitstreamFreePB;
    qVideoQueueFree.push(*pMfxBitstream);
#else
    pMfxBitstream->DataLength = 0;
    pMfxBitstream->DataOffset = 0;
#endif
    return (m_Mux.format.bStreamError) ? MFX_ERR_UNKNOWN : MFX_ERR_NONE;
}

vector<int> CAvcodecWriter::GetStreamTrackIdList() {
    vector<int> streamTrackId;
    streamTrackId.reserve(m_Mux.audio.size());
    for (auto audio : m_Mux.audio) {
        streamTrackId.push_back(audio.nInTrackId);
    }
    for (auto sub : m_Mux.sub) {
        streamTrackId.push_back(sub.nInTrackId);
    }
    return std::move(streamTrackId);
}

AVMuxAudio *CAvcodecWriter::getAudioPacketStreamData(const AVPacket *pkt) {
    const int streamIndex = pkt->stream_index;
    //privには、trackIdへのポインタが格納してある…はず
    const int inTrackId = (int16_t)(pkt->flags >> 16);
    for (int i = 0; i < (int)m_Mux.audio.size(); i++) {
        //streamIndexの一致とtrackIdの一致を確認する
        if (m_Mux.audio[i].nStreamIndexIn == streamIndex
            && m_Mux.audio[i].nInTrackId == inTrackId) {
            return &m_Mux.audio[i];
        }
    }
    return NULL;
}

AVMuxSub *CAvcodecWriter::getSubPacketStreamData(const AVPacket *pkt) {
    const int streamIndex = pkt->stream_index;
    //privには、trackIdへのポインタが格納してある…はず
    const int inTrackId = (int16_t)(pkt->flags >> 16);
    for (int i = 0; i < (int)m_Mux.sub.size(); i++) {
        //streamIndexの一致とtrackIdの一致を確認する
        if (m_Mux.sub[i].nStreamIndexIn == streamIndex
            && m_Mux.sub[i].nInTrackId == inTrackId) {
            return &m_Mux.sub[i];
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

void CAvcodecWriter::WriteNextPacket(AVMuxAudio *pMuxAudio, AVPacket *pkt, int samples, int64_t *pWrittenDts) {
    AVRational samplerate = { 1, pMuxAudio->pCodecCtxIn->sample_rate };
    if (samples) {
        //durationについて、sample数から出力ストリームのtimebaseに変更する
        pkt->stream_index = pMuxAudio->pStream->index;
        pkt->flags        = AV_PKT_FLAG_KEY; //元のpacketの上位16bitにはトラック番号を紛れ込ませているので、av_interleaved_write_frame前に消すこと
        pkt->dts          = av_rescale_q(pMuxAudio->nOutputSamples + pMuxAudio->nDelaySamplesOfAudio, samplerate, pMuxAudio->pStream->time_base);
        pkt->pts          = pkt->dts;
        pkt->duration     = (int)av_rescale_q(samples, samplerate, pMuxAudio->pStream->time_base);
        if (pkt->duration == 0)
            pkt->duration = (int)(pkt->pts - pMuxAudio->nLastPtsOut);
        pMuxAudio->nLastPtsOut = pkt->pts;
        *pWrittenDts = av_rescale_q(pkt->dts, pMuxAudio->pStream->time_base, QSV_NATIVE_TIMEBASE);
        m_Mux.format.bStreamError |= 0 != av_interleaved_write_frame(m_Mux.format.pFormatCtx, pkt);
        pMuxAudio->nOutputSamples += samples;
    } else {
        //av_interleaved_write_frameに渡ったパケットは開放する必要がないが、
        //それ以外は解放してやる必要がある
        av_free_packet(pkt);
    }
}

AVFrame *CAvcodecWriter::AudioDecodePacket(AVMuxAudio *pMuxAudio, const AVPacket *pkt, int *got_result) {
    *got_result = FALSE;
    if (pMuxAudio->bDecodeError) {
        return nullptr;
    }
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
    while (!(*got_result) || pktIn->size > 0) {
        AVFrame *decodedData = av_frame_alloc();
        int len = avcodec_decode_audio4(pMuxAudio->pOutCodecDecodeCtx, decodedData, got_result, pktIn);
        if (decodedFrame->nb_samples && decodedData->nb_samples) {
            AVFrame *decodedFrameNew        = av_frame_alloc();
            decodedFrameNew->nb_samples     = decodedFrame->nb_samples + decodedData->nb_samples;
            decodedFrameNew->channels       = decodedData->channels;
            decodedFrameNew->channel_layout = decodedData->channel_layout;
            decodedFrameNew->sample_rate    = decodedData->sample_rate;
            decodedFrameNew->format         = decodedData->format;
            av_frame_get_buffer(decodedFrameNew, 32); //format, channel_layout, nb_samplesを埋めて、av_frame_get_buffer()により、メモリを確保する
            const int bytes_per_sample = av_get_bytes_per_sample((AVSampleFormat)decodedFrameNew->format)
                * (av_sample_fmt_is_planar((AVSampleFormat)decodedFrameNew->format) ? 1 : decodedFrameNew->channels);
            const int channel_loop_count = av_sample_fmt_is_planar((AVSampleFormat)decodedFrameNew->format) ? decodedFrameNew->channels : 1;
            for (int i = 0; i < channel_loop_count; i++) {
                if (decodedFrame->nb_samples > 0) {
                    memcpy(decodedFrameNew->data[i], decodedFrame->data[i], decodedFrame->nb_samples * bytes_per_sample);
                }
                if (decodedData->nb_samples > 0) {
                    memcpy(decodedFrameNew->data[i] + decodedFrame->nb_samples * bytes_per_sample,
                        decodedData->data[i], decodedData->nb_samples * bytes_per_sample);
                }
            }
            av_frame_free(&decodedFrame);
            decodedFrame = decodedFrameNew;
        } else if (decodedData->nb_samples) {
            av_frame_free(&decodedFrame);
            decodedFrame = decodedData;
        }
        if (len < 0) {
            AddMessage(QSV_LOG_WARN, _T("avcodec writer: failed to decode audio #%d: %s\n"), pMuxAudio->nInTrackId, qsv_av_err2str(len).c_str());
            pMuxAudio->bDecodeError = true;
            break;
        } else if (pktIn->size != len) {
            int newLen = pktIn->size - len;
            memmove(pMuxAudio->OutPacket.data, pktIn->data + len, newLen);
            pMuxAudio->OutPacket.size = newLen;
            pktIn = &pMuxAudio->OutPacket;
        } else {
            pMuxAudio->OutPacket.size = 0;
            break;
        }
    }
    *got_result = decodedFrame->nb_samples > 0;
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
        AddMessage(QSV_LOG_WARN, _T("avcodec writer: failed to encode audio #%d: %s\n"), pMuxAudio->nInTrackId, qsv_av_err2str(ret).c_str());
        pMuxAudio->bEncodeError = true;
    } else if (*got_result) {
        samples = (int)av_rescale_q(pEncPkt->duration, pMuxAudio->pOutCodecEncodeCtx->pkt_timebase, { 1, pMuxAudio->pCodecCtxIn->sample_rate });
    }
    return samples;
}

void CAvcodecWriter::AudioFlushStream(AVMuxAudio *pMuxAudio, int64_t *pWrittenDts) {
    while (pMuxAudio->pOutCodecDecodeCtx && !pMuxAudio->bEncodeError) {
        int samples = 0;
        int got_result = 0;
        AVPacket pkt = { 0 };
        AVFrame *decodedFrame = AudioDecodePacket(pMuxAudio, &pkt, &got_result);
        if (!got_result && (decodedFrame != nullptr || pMuxAudio->bDecodeError)) {
            if (decodedFrame != nullptr) {
                av_frame_free(&decodedFrame);
            }
            break;
        }
        if (0 == AudioResampleFrame(pMuxAudio, &decodedFrame)) {
            samples = AudioEncodeFrame(pMuxAudio, &pkt, decodedFrame, &got_result);
        }
        if (decodedFrame != nullptr) {
            av_frame_free(&decodedFrame);
        }
        WriteNextPacket(pMuxAudio, &pkt, samples, pWrittenDts);
    }
    while (pMuxAudio->pSwrContext && !pMuxAudio->bEncodeError) {
        int samples = 0;
        int got_result = 0;
        AVPacket pkt = { 0 };
        AVFrame *decodedFrame = nullptr;
        if (0 != AudioResampleFrame(pMuxAudio, &decodedFrame) || decodedFrame == nullptr) {
            break;
        }
        samples = AudioEncodeFrame(pMuxAudio, &pkt, decodedFrame, &got_result);
        WriteNextPacket(pMuxAudio, &pkt, samples, pWrittenDts);
    }
    while (pMuxAudio->pOutCodecEncodeCtx) {
        int got_result = 0;
        AVPacket pkt = { 0 };
        int samples = AudioEncodeFrame(pMuxAudio, &pkt, nullptr, &got_result);
        if (samples == 0 || pMuxAudio->bDecodeError)
            break;
        WriteNextPacket(pMuxAudio, &pkt, samples, pWrittenDts);
    }
}

mfxStatus CAvcodecWriter::SubtitleTranscode(const AVMuxSub *pMuxSub, AVPacket *pkt) {
    int got_sub = 0;
    AVSubtitle sub = { 0 };
    if (0 > avcodec_decode_subtitle2(pMuxSub->pOutCodecDecodeCtx, &sub, &got_sub, pkt)) {
        AddMessage(QSV_LOG_ERROR, _T("Failed to decode subtitle.\n"));
        m_Mux.format.bStreamError = true;
    }
    if (!pMuxSub->pBuf) {
        AddMessage(QSV_LOG_ERROR, _T("No buffer for encoding subtitle.\n"));
        m_Mux.format.bStreamError = true;
    }
    av_free_packet(pkt);
    if (m_Mux.format.bStreamError)
        return MFX_ERR_UNKNOWN;
    if (!got_sub || sub.num_rects == 0)
        return MFX_ERR_NONE;

    const int nOutPackets = 1 + (pMuxSub->pOutCodecEncodeCtx->codec_id == AV_CODEC_ID_DVB_SUBTITLE);
    for (int i = 0; i < nOutPackets; i++) {
        sub.pts               += av_rescale_q(sub.start_display_time, av_make_q(1, 1000), av_make_q(1, AV_TIME_BASE));
        sub.end_display_time  -= sub.start_display_time;
        sub.start_display_time = 0;
        if (i > 0) {
            sub.num_rects = 0;
        }

        int sub_out_size = avcodec_encode_subtitle(pMuxSub->pOutCodecEncodeCtx, pMuxSub->pBuf, SUB_ENC_BUF_MAX_SIZE, &sub);
        if (sub_out_size < 0) {
            AddMessage(QSV_LOG_ERROR, _T("failed to encode subtitle.\n"));
            m_Mux.format.bStreamError = true;
            return MFX_ERR_UNKNOWN;
        }

        AVPacket pktOut;
        av_init_packet(&pktOut);
        pktOut.data = pMuxSub->pBuf;
        pktOut.stream_index = pMuxSub->pStream->index;
        pktOut.size = sub_out_size;
        pktOut.duration = (int)av_rescale_q(sub.end_display_time, av_make_q(1, 1000), pMuxSub->pStream->time_base);
        pktOut.pts  = av_rescale_q(sub.pts, av_make_q(1, AV_TIME_BASE), pMuxSub->pStream->time_base);
        if (pMuxSub->pOutCodecEncodeCtx->codec_id == AV_CODEC_ID_DVB_SUBTITLE) {
            pktOut.pts += 90 * ((i == 0) ? sub.start_display_time : sub.end_display_time);
        }
        pktOut.dts = pktOut.pts;
        m_Mux.format.bStreamError |= 0 != av_interleaved_write_frame(m_Mux.format.pFormatCtx, &pktOut);
    }
    return (m_Mux.format.bStreamError) ? MFX_ERR_UNKNOWN : MFX_ERR_NONE;
}

mfxStatus CAvcodecWriter::SubtitleWritePacket(AVPacket *pkt) {
    //字幕を処理する
    const AVMuxSub *pMuxSub = getSubPacketStreamData(pkt);
    int64_t pts_adjust = av_rescale_q(m_Mux.video.nInputFirstPts, m_Mux.video.pInputCodecCtx->pkt_timebase, pMuxSub->pCodecCtxIn->pkt_timebase);
    //ptsが存在しない場合はないものとすると、AdjustTimestampTrimmedの結果がAV_NOPTS_VALUEとなるのは、
    //Trimによりカットされたときのみ
    int64_t pts_orig = pkt->pts;
    if (AV_NOPTS_VALUE != (pkt->pts = AdjustTimestampTrimmed(std::max(INT64_C(0), pkt->pts - pts_adjust), pMuxSub->pCodecCtxIn->pkt_timebase, pMuxSub->pStream->time_base, false))) {
        if (pMuxSub->pOutCodecEncodeCtx) {
            return SubtitleTranscode(pMuxSub, pkt);
        }
        //dts側にもpts側に加えたのと同じ分だけの補正をかける
        pkt->dts = pkt->dts + (av_rescale_q(pkt->pts, pMuxSub->pStream->time_base, pMuxSub->pCodecCtxIn->pkt_timebase) - pts_orig);
        //timescaleの変換を行い、負の値をとらないようにする
        pkt->dts = std::max(INT64_C(0), av_rescale_q(pkt->dts, pMuxSub->pCodecCtxIn->pkt_timebase, pMuxSub->pStream->time_base));
        pkt->flags &= 0x0000ffff; //元のpacketの上位16bitにはトラック番号を紛れ込ませているので、av_interleaved_write_frame前に消すこと
        pkt->duration = (int)av_rescale_q(pkt->duration, pMuxSub->pCodecCtxIn->pkt_timebase, pMuxSub->pStream->time_base);
        pkt->stream_index = pMuxSub->pStream->index;
        pkt->pos = -1;
        m_Mux.format.bStreamError |= 0 != av_interleaved_write_frame(m_Mux.format.pFormatCtx, pkt);
    }
    return (m_Mux.format.bStreamError) ? MFX_ERR_UNKNOWN : MFX_ERR_NONE;
}

mfxStatus CAvcodecWriter::WriteNextPacket(AVPacket *pkt) {
#if ENABLE_AVCODEC_OUT_THREAD
    //pkt = nullptrの代理として、pkt.buf == nullptrなパケットを投入
    AVPacket zeroFilled = { 0 };
    if (!m_Mux.thread.qAudioPacket.push((pkt == nullptr) ? zeroFilled : *pkt)) {
        AddMessage(QSV_LOG_ERROR, _T("Failed to allocate memory for audio packet queue.\n"));
        m_Mux.format.bStreamError = true;
    }
    SetEvent(m_Mux.thread.heEventPktAdded);
    return (m_Mux.format.bStreamError) ? MFX_ERR_UNKNOWN : MFX_ERR_NONE;
#else
    int64_t dts = 0;
    return WriteNextPacketInternal(pkt, &dts);
#endif
}

mfxStatus CAvcodecWriter::WriteNextPacketInternal(AVPacket *pkt, int64_t *pWrittenDts) {
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
            AudioFlushStream(&m_Mux.audio[i], pWrittenDts);
        }
        *pWrittenDts = INT64_MAX;
        AddMessage(QSV_LOG_DEBUG, _T("Flushed audio buffer.\n"));
        return (m_Mux.format.bStreamError) ? MFX_ERR_UNKNOWN : MFX_ERR_NONE;
    }

    if (((int16_t)(pkt->flags >> 16)) < 0) {
        return SubtitleWritePacket(pkt);
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
                && 1 < std::abs(ptsDiff - pkt->duration)) {
                //ptsの差分から計算しなおす
                samples = (int)av_rescale_q(ptsDiff, pMuxAudio->pCodecCtxIn->pkt_timebase, samplerate);
            }
        }
        pMuxAudio->nLastPtsIn = pkt->pts;
        WriteNextPacket(pMuxAudio, pkt, samples, pWrittenDts);
    } else if (!pMuxAudio->bDecodeError && !pMuxAudio->bEncodeError) {
        int got_result = 0;
        AVFrame *decodedFrame = AudioDecodePacket(pMuxAudio, pkt, &got_result);
        if (pkt != nullptr) {
            av_free_packet(pkt);
        }
        if (got_result && (pMuxAudio->bDecodeError || decodedFrame != nullptr)) {
            if (0 <= AudioResampleFrame(pMuxAudio, &decodedFrame)) {
                if (pMuxAudio->pDecodedFrameCache == nullptr && (decodedFrame->nb_samples == pMuxAudio->pOutCodecEncodeCtx->frame_size || pMuxAudio->pOutCodecEncodeCtx->frame_size == 0)) {
                    //デコードの出力サンプル数とエンコーダのframe_sizeが一致していれば、そのままエンコードする
                    samples = AudioEncodeFrame(pMuxAudio, pkt, decodedFrame, &got_result);
                    if (got_result && samples) {
                        WriteNextPacket(pMuxAudio, pkt, samples, pWrittenDts);
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
                            WriteNextPacket(pMuxAudio, pkt, samples, pWrittenDts);
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

mfxStatus CAvcodecWriter::WriteThreadFunc() {
#if ENABLE_AVCODEC_OUT_THREAD
    //映像と音声の同期をとる際に、それをあきらめるまでの閾値
    const size_t videoPacketThreshold = std::min<size_t>(512, m_Mux.thread.qVideobitstream.capacity());
    const size_t audioPacketThreshold = std::min<size_t>(2048, m_Mux.thread.qAudioPacket.capacity());
    //現在のdts、"-1"は無視することを映像と音声の同期を行う必要がないことを意味する
    int64_t audioDts = (m_Mux.audio.size()) ? -1 : INT64_MAX;
    int64_t videoDts = (m_Mux.video.pCodecCtx) ? -1 : INT64_MAX;
    //キューにデータが存在するか
    bool bAudioExists = false;
    bool bVideoExists = false;
    const auto fpsTimebase = av_inv_q(m_Mux.video.nFPS);
    const auto dtsThreshold = std::max<int64_t>(av_rescale_q(8, fpsTimebase, QSV_NATIVE_TIMEBASE), QSV_TIMEBASE / 4);
    WaitForSingleObject(m_Mux.thread.heEventPktAdded, INFINITE);
    while (!m_Mux.thread.bAbort) {
        do {
            //映像・音声の同期待ちが必要な場合、falseとなってループから抜けるよう、ここでfalseに設定する
            bAudioExists = false;
            bVideoExists = false;
            AVPacket pkt = { 0 };
            while ((videoDts < 0 || audioDts <= videoDts + dtsThreshold)
                && false != (bAudioExists = m_Mux.thread.qAudioPacket.front_copy_and_pop_no_lock(&pkt))) {
                int64_t pktDts = 0;
                //pkt.buf == nullptrはpkt = nullptrの代理として格納してあることに注意
                WriteNextPacketInternal((pkt.buf == nullptr) ? nullptr : &pkt, &pktDts);
                audioDts = (std::max)(audioDts, pktDts);
            }
            mfxBitstream bitstream = { 0 };
            while ((audioDts < 0 || videoDts <= audioDts + dtsThreshold)
                && false != (bVideoExists = m_Mux.thread.qVideobitstream.front_copy_and_pop_no_lock(&bitstream))) {
                WriteNextFrameInternal(&bitstream, &videoDts);
            }
            //一定以上の動画フレームがキューにたまっており、音声キューになにもなければ、
            //音声を無視して動画フレームの処理を開始させる
            if (m_Mux.thread.qAudioPacket.size() == 0 && m_Mux.thread.qVideobitstream.size() > videoPacketThreshold) {
                audioDts = -1;
            }
            //一定以上の音声フレームがキューにたまっており、動画キューになにもなければ、
            //動画を無視して音声フレームの処理を開始させる
            if (m_Mux.thread.qVideobitstream.size() == 0 && m_Mux.thread.qAudioPacket.size() > audioPacketThreshold) {
                videoDts = -1;
            }
        } while (bAudioExists || bVideoExists); //両方のキューがひとまず空になるか、映像・音声の同期待ちが必要になるまで回す
                                                //次のフレーム・パケットが送られてくるまで待機する
        ResetEvent(m_Mux.thread.heEventPktAdded);
        WaitForSingleObject(m_Mux.thread.heEventPktAdded, INFINITE);
    }
    //メインループを抜けたことを通知する
    SetEvent(m_Mux.thread.heEventClosing);
    bAudioExists = !m_Mux.thread.qAudioPacket.empty();
    bVideoExists = !m_Mux.thread.qVideobitstream.empty();
    //まずは映像と音声の同期をとって出力するためのループ
    while (bAudioExists && bVideoExists) {
        AVPacket pkt = { 0 };
        while (audioDts <= videoDts + dtsThreshold
            && false != (bAudioExists = m_Mux.thread.qAudioPacket.front_copy_and_pop_no_lock(&pkt))) {
            int64_t pktDts = 0;
            WriteNextPacketInternal((pkt.buf == nullptr) ? nullptr : &pkt, &pktDts);
            audioDts = (std::max)(audioDts, pktDts);
        }
        mfxBitstream bitstream = { 0 };
        while (videoDts <= audioDts + dtsThreshold
            && false != (bVideoExists = m_Mux.thread.qVideobitstream.front_copy_and_pop_no_lock(&bitstream))) {
            WriteNextFrameInternal(&bitstream, &videoDts);
        }
    }
    { //音声を書き出す
        AVPacket pkt = { 0 };
        while (m_Mux.thread.qAudioPacket.front_copy_and_pop_no_lock(&pkt)) {
            int64_t pktDts = 0;
            WriteNextPacketInternal((pkt.buf == nullptr) ? nullptr : &pkt, &pktDts);
        }
    }
    { //動画を書き出す
        mfxBitstream bitstream = { 0 };
        while (m_Mux.thread.qVideobitstream.front_copy_and_pop_no_lock(&bitstream)) {
            WriteNextFrameInternal(&bitstream, &videoDts);
        }
    }
#endif
    return (m_Mux.format.bStreamError) ? MFX_ERR_UNKNOWN : MFX_ERR_NONE;
}

HANDLE CAvcodecWriter::getThreadHandle() {
#if ENABLE_AVCODEC_OUT_THREAD
    return (HANDLE)m_Mux.thread.thOutput.native_handle();
#else
    return NULL;
#endif
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
