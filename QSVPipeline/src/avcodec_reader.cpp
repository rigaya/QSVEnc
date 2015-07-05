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
    MSDK_ZERO_MEMORY(m_Demux.format);
    MSDK_ZERO_MEMORY(m_Demux.video);
    MSDK_ZERO_MEMORY(m_sDecParam);
    m_AudioPacketsBufferL2Used = 0;
}

CAvcodecReader::~CAvcodecReader() {

}

void CAvcodecReader::clearAudioPacketList(vector<AVPacket>& pktList) {
    for (mfxU32 i_pkt = 0; i_pkt < pktList.size(); i_pkt++) {
        if (pktList[i_pkt].data) {
            av_free_packet(&pktList[i_pkt]);
        }
    }
    pktList.clear();
}

void CAvcodecReader::CloseFormat(AVDemuxFormat *pFormat) {
    //close video file
    if (pFormat->pFormatCtx) {
        avformat_close_input(&pFormat->pFormatCtx);
    }
    memset(pFormat, 0, sizeof(pFormat[0]));
}

void CAvcodecReader::CloseVideo(AVDemuxVideo *pVideo) {
    //close bitstreamfilter
    if (pVideo->pH264Bsfc) {
        av_bitstream_filter_close(pVideo->pH264Bsfc);
    }
    
    if (pVideo->pExtradata) {
        av_free(pVideo->pExtradata);
    }

    if (pVideo->frameData.cs_initialized) {
        DeleteCriticalSection(&pVideo->frameData.cs);
    }
    memset(pVideo, 0, sizeof(pVideo[0]));
    pVideo->nIndex = -1;
}

void CAvcodecReader::CloseAudio(AVDemuxAudio *pAudio) {
    if (pAudio->pktSample.data) {
        av_free_packet(&pAudio->pktSample);
    }
    memset(pAudio, 0, sizeof(pAudio[0]));
    pAudio->nIndex = -1;
}

void CAvcodecReader::Close() {
    //リソースの解放
    for (int i = 0; i < _countof(m_AudioPacketsBufferL1); i++) {
        m_AudioPacketsBufferL1[i].clear();
    }
    if (m_AudioPacketsBufferL2Used) {
        //使用済みパケットを削除する
        //これらのパケットはすでにWriter側に渡っているか、解放されているので、av_free_packetは不要
        m_AudioPacketsBufferL2.erase(m_AudioPacketsBufferL2.begin(), m_AudioPacketsBufferL2.begin() + m_AudioPacketsBufferL2Used);
    }
    clearAudioPacketList(m_AudioPacketsBufferL2);
    m_AudioPacketsBufferL2Used = 0;

    CloseFormat(&m_Demux.format);
    CloseVideo(&m_Demux.video);
    for (int i = 0; i < (int)m_Demux.audio.size(); i++)
        CloseAudio(&m_Demux.audio[i]);
    m_Demux.audio.clear();

    m_sTrimParam.list.clear();
    m_sTrimParam.offset = 0;

    m_hevcMp42AnnexbBuffer.clear();

    //free input buffer (使用していない)
    //if (buffer) {
    //    free(buffer);
    //    buffer = nullptr;
    //}

    MSDK_ZERO_MEMORY(m_sDecParam);
}

mfxU32 CAvcodecReader::getQSVFourcc(mfxU32 id) {
    for (int i = 0; i < _countof(QSV_DECODE_LIST); i++)
        if (QSV_DECODE_LIST[i].codec_id == id)
            return QSV_DECODE_LIST[i].qsv_fourcc;
    return 0;
}

vector<int> CAvcodecReader::getStreamIndex(AVMediaType type) {
    vector<int> streams;
    const int n_streams = m_Demux.format.pFormatCtx->nb_streams;
    for (int i = 0; i < n_streams; i++) {
        if (m_Demux.format.pFormatCtx->streams[i]->codec->codec_type == type) {
            streams.push_back(i);
        }
    }
    return std::move(streams);
}

void CAvcodecReader::sortVideoPtsList() {
    //フレーム順序が確定していないところをソートする
    FramePos *ptr = m_Demux.video.frameData.frame;
    std::sort(ptr + m_Demux.video.frameData.fixed_num, ptr + m_Demux.video.frameData.num,
        [](const FramePos& posA, const FramePos& posB) {
        return (abs(posA.pts - posB.pts) < 0xFFFFFFFF) ? posA.pts < posB.pts : posB.pts < posA.pts; });
}

void CAvcodecReader::addVideoPtsToList(FramePos pos) {
    if (m_Demux.video.frameData.capacity <= m_Demux.video.frameData.num+1) {
        EnterCriticalSection(&m_Demux.video.frameData.cs);
        extend_array_size(&m_Demux.video.frameData);
        LeaveCriticalSection(&m_Demux.video.frameData.cs);
    }
    if (pos.pts == AV_NOPTS_VALUE && m_Demux.video.nStreamPtsInvalid & AVQSV_PTS_HALF_INVALID) {
        //ptsがないのは音声抽出で、正常に抽出されない問題が生じる
        //半分PTSがないPAFFのような動画については、前のフレームからの補完を行う
        const FramePos *lastFrame = &m_Demux.video.frameData.frame[m_Demux.video.frameData.num-1];
        pos.dts = lastFrame->dts + lastFrame->duration;
        pos.pts = lastFrame->pts + lastFrame->duration;
    }
    m_Demux.video.frameData.frame[m_Demux.video.frameData.num] = pos;
    m_Demux.video.frameData.num++;

    if (m_Demux.video.frameData.fixed_num + 32 < m_Demux.video.frameData.num) {
        if (m_Demux.video.nStreamPtsInvalid & AVQSV_PTS_SOMETIMES_INVALID) {
            //ptsがあてにならない時は、dtsから適当に生成する
            FramePos *frame = m_Demux.video.frameData.frame;
            const int i_fin = m_Demux.video.frameData.fixed_num + 16;
            for (int i = m_Demux.video.frameData.fixed_num + 1; i <= i_fin; i++) {
                if (frame[i].dts == AV_NOPTS_VALUE) {
                    //まずdtsがない場合は、前のフレームからコピーする
                    frame[i].dts = frame[i-1].dts;
                }
            }
            int64_t firstFramePtsDtsDiff = frame[0].pts - frame[0].dts;
            for (int i = m_Demux.video.frameData.fixed_num + 1; i <= i_fin; i++) {
                frame[i].pts = frame[i].dts + firstFramePtsDtsDiff;
            }
        }
        //m_Demux.video.frameData.fixed_numから16フレーム分、pts順にソートを行う
        sortVideoPtsList();
        //進捗表示用のdurationの計算を行う
        const FramePos *pos = m_Demux.video.frameData.frame + m_Demux.video.frameData.fixed_num;
        int64_t duration = pos[16].pts - pos[0].pts;
        if (duration < 0 || duration > 0xFFFFFFFF) {
            duration = 0;
            for (int i = 1; i < 16; i++) {
                int64_t diff = MSDK_MAX(0, pos[i].pts - pos[i-1].pts);
                int64_t last_frame_dur = MSDK_MAX(0, pos[i-1].duration);
                duration += (diff > 0xFFFFFFFF) ? last_frame_dur : diff;
            }
        }
        m_Demux.video.frameData.duration += duration;
        m_Demux.video.frameData.fixed_num += 16;
    }
}

void CAvcodecReader::hevcMp42Annexb(AVPacket *pkt) {
    static const mfxU8 SC[] = { 0, 0, 0, 1 };
    const mfxU8 *ptr, *ptr_fin;
    if (pkt == NULL) {
        m_hevcMp42AnnexbBuffer.reserve(m_Demux.video.nExtradataSize + 128);
        ptr = m_Demux.video.pExtradata;
        ptr_fin = ptr + m_Demux.video.nExtradataSize;
        ptr += 0x16;
    } else {
        m_hevcMp42AnnexbBuffer.reserve(pkt->size + 128);
        ptr = pkt->data;
        ptr_fin = ptr + pkt->size;
    }
    const int numOfArrays = *ptr;
    ptr += !!numOfArrays;

    while (ptr + 6 < ptr_fin) {
        ptr += !!numOfArrays;
        const int count = readUB16(ptr); ptr += 2;
        int units = (numOfArrays) ? count : 1;
        for (int i = MSDK_MAX(1, units); i; i--) {
            uint32_t size = readUB16(ptr); ptr += 2;
            uint32_t uppper = count << 16;
            size += (numOfArrays) ? 0 : uppper;
            m_hevcMp42AnnexbBuffer.insert(m_hevcMp42AnnexbBuffer.end(), SC, SC+4);
            m_hevcMp42AnnexbBuffer.insert(m_hevcMp42AnnexbBuffer.end(), ptr, ptr+size); ptr += size;
        }
    }
    if (pkt) {
        if (pkt->buf->size < (int)m_hevcMp42AnnexbBuffer.size()) {
            av_grow_packet(pkt, (int)m_hevcMp42AnnexbBuffer.size());
        }
        memcpy(pkt->data, m_hevcMp42AnnexbBuffer.data(), m_hevcMp42AnnexbBuffer.size());
        pkt->size = (int)m_hevcMp42AnnexbBuffer.size();
    } else {
        if (m_Demux.video.pExtradata) {
            av_free(m_Demux.video.pExtradata);
        }
        m_Demux.video.pExtradata = (mfxU8 *)av_malloc(m_hevcMp42AnnexbBuffer.size());
        m_Demux.video.nExtradataSize = (int)m_hevcMp42AnnexbBuffer.size();
        memcpy(m_Demux.video.pExtradata, m_hevcMp42AnnexbBuffer.data(), m_hevcMp42AnnexbBuffer.size());
    }
    m_hevcMp42AnnexbBuffer.clear();
}

mfxStatus CAvcodecReader::getFirstFramePosAndFrameRate(AVRational fpsDecoder, mfxSession session, mfxBitstream *bitstream, const sTrim *pTrimList, int nTrimCount) {
    mfxStatus sts = MFX_ERR_NONE;
    const bool fpsDecoderInvalid = (fpsDecoder.den == 0 || fpsDecoder.num == 0);
    const int maxCheckFrames = (m_Demux.format.nAnalyzeSec == 0) ? 256 : 9000;
    const int maxCheckSec = (m_Demux.format.nAnalyzeSec == 0) ? INT_MAX : m_Demux.format.nAnalyzeSec;
    vector<FramePos> framePosList;
    framePosList.reserve(maxCheckFrames);

    mfxVideoParam param = { 0 };
    param.mfx.CodecId = m_nInputCodec;
    //よくわからないが、ここは1としないほうがよい模様
    //下記のフレームレート推定が怪しくなる(主にPAFFの判定に失敗する)
    //フレーム確保量の問題から、原因不明の問題が発生すると思われる
    //param.AsyncDepth = 1;
    param.IOPattern = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
    if (MFX_ERR_NONE != (sts = MFXVideoDECODE_DecodeHeader(session, bitstream, &param))) {
        m_strInputInfo += _T("avcodec reader: failed to decode header(2).\n");
        return sts;
    }

    mfxFrameAllocRequest request = { 0 };
    if (MFX_ERR_NONE > (sts = MFXVideoDECODE_QueryIOSurf(session, &param, &request))) {
        m_strInputInfo += _T("avcodec reader: failed to get required frame.\n");
        return sts;
    }

    int numSurfaces = request.NumFrameSuggested;
    int surfaceWidth = MSDK_ALIGN32(request.Info.Width);
    int surfaceHeight = MSDK_ALIGN32(request.Info.Height);
    int surfaceSize = surfaceWidth * surfaceHeight * 3 / 2;
    vector<mfxU8> surfaceBuffers(numSurfaces * surfaceSize);
    std::unique_ptr<mfxFrameSurface1[]> pmfxSurfaces(new mfxFrameSurface1[numSurfaces]);

    for (int i = 0; i < numSurfaces; i++) {
        MSDK_ZERO_MEMORY(pmfxSurfaces[i]);
        MSDK_MEMCPY(&pmfxSurfaces[i].Info, &param.mfx.FrameInfo, sizeof(param.mfx.FrameInfo));
        pmfxSurfaces[i].Data.Y = surfaceBuffers.data() + i * surfaceSize;
        pmfxSurfaces[i].Data.UV = pmfxSurfaces[i].Data.Y + surfaceWidth * surfaceHeight;
        pmfxSurfaces[i].Data.Pitch = (mfxU16)surfaceWidth;
    }

    if (MFX_ERR_NONE != (sts = MFXVideoDECODE_Init(session, &param))) {
        m_strInputInfo += _T("avcodec reader: failed to init decoder.\n");
        m_strInputInfo += _T("                decoding of this video is not supported.\n");
        return sts;
    }

    int gotFrameCount = 0; //デコーダの出力フレーム
    int moreDataCount = 0; //出力が始まってから、デコーダが余剰にフレームを求めた回数
    bool gotFirstKeyframePos = false;
    FramePos firstKeyframePos = { 0 };
    AVPacket pkt;
    auto getTotalDuration =[&]() -> int {
        if (firstKeyframePos.duration)
            return 0;
        int diff = 0;
        if (pkt.dts != AV_NOPTS_VALUE) {
            diff = (int)(pkt.dts - firstKeyframePos.dts);
        } else if (pkt.pts != AV_NOPTS_VALUE) {
            diff = (int)(pkt.pts - firstKeyframePos.pts);
        }
        return diff * m_Demux.video.pCodecCtx->pkt_timebase.num / m_Demux.video.pCodecCtx->pkt_timebase.den;
    };

    m_Demux.format.nPreReadBufferIdx = UINT32_MAX; //PreReadBufferからの読み込みを禁止にする
    clearAudioPacketList(m_PreReadBuffer);
    m_PreReadBuffer.reserve(maxCheckFrames);

    for (int i = 0; i < maxCheckFrames && getTotalDuration() < maxCheckSec && !getSample(&pkt); i++) {
        int64_t pts = pkt.pts, dts = pkt.dts;
        FramePos pos = { (pts == AV_NOPTS_VALUE) ? dts : pts, dts, pkt.duration, pkt.flags };
        framePosList.push_back(pos);
        if (i == 0 && pts != AV_NOPTS_VALUE) {
            m_Demux.video.nStreamFirstPts = pkt.pts;
        }
        if (!gotFirstKeyframePos && pkt.flags & AV_PKT_FLAG_KEY) {
            firstKeyframePos = pos;
            //キーフレームに到達するまでQSVではフレームが出てこない
            //そのぶんのずれを記録しておき、Trim値などに補正をかける
            m_sTrimParam.offset = i;
            gotFirstKeyframePos = true;
        }
        ///キーフレーム取得済み
        if (gotFirstKeyframePos) {
            AppendMfxBitstream(bitstream, pkt.data, pkt.size);

            mfxStatus decsts = MFX_ERR_MORE_SURFACE;
            while (MFX_ERR_MORE_SURFACE == decsts) {
                auto getFreeSurface = [&]() -> mfxFrameSurface1* {
                    for (int i = 0; i < numSurfaces; i++) {
                        if (!pmfxSurfaces[i].Data.Locked) {
                            return &pmfxSurfaces[i];
                        }
                    }
                    return NULL;
                };
                mfxSyncPoint syncp = NULL;
                mfxFrameSurface1 *pmfxOut = NULL;
                decsts = MFXVideoDECODE_DecodeFrameAsync(session, bitstream, getFreeSurface(), &pmfxOut, &syncp);
                if (MFX_ERR_NONE <= decsts && syncp) {
                    decsts = MFXVideoCORE_SyncOperation(session, syncp, 60 * 1000);
                    gotFrameCount += (MFX_ERR_NONE == decsts);
                } else if (gotFrameCount && decsts == MFX_ERR_MORE_DATA) {
                    moreDataCount++;
                }
            }
            if (decsts < MFX_ERR_NONE && decsts != MFX_ERR_MORE_DATA) {
                m_strInputInfo += _T("avcodec reader: failed to decode stream.\n");
                return decsts;
            }
        }
        m_PreReadBuffer.push_back(pkt);
    }
    //上記ループで最後まで行ってしまうと値が入ってしまうので初期化
    m_Demux.video.frameData.duration = 0;
    m_Demux.video.frameData.fixed_num = 0;
    m_Demux.video.frameData.num = 0;

    if (!gotFirstKeyframePos) {
        m_strInputInfo += _T("avcodec reader: failed to get first frame pos.\n");
        return MFX_ERR_UNKNOWN;
    }

    //PAFFの場合、2フィールド目のpts, dtsが存在しないことがある
    mfxU32 dts_pts_no_value_in_between = 0;
    mfxU32 dts_pts_invalid_count = 0;
    mfxU32 dts_pts_invalid_keyframe_count = 0;
    mfxU32 keyframe_count = 0;
    for (mfxU32 i = 0; i < framePosList.size(); i++) {
        //自分のpts/dtsがともにAV_NOPTS_VALUEで、それが前後ともにAV_NOPTS_VALUEでない場合に修正する
        if (i > 0
            && framePosList[i].dts == AV_NOPTS_VALUE
            && framePosList[i].pts == AV_NOPTS_VALUE) {
            if (   framePosList[i-1].dts != AV_NOPTS_VALUE
                && framePosList[i-1].pts != AV_NOPTS_VALUE
                && (i + 1 >= framePosList.size()
                    || (framePosList[i+1].pts != AV_NOPTS_VALUE
                     && framePosList[i+1].pts != AV_NOPTS_VALUE))) {
                framePosList[i].dts = framePosList[i-1].dts;
                framePosList[i].pts = framePosList[i-1].pts;
                dts_pts_no_value_in_between++;
            }
        }
        if (   framePosList[i].dts == AV_NOPTS_VALUE
            && framePosList[i].pts == AV_NOPTS_VALUE) {
            //自分のpts/dtsがともにAV_NOPTS_VALUEだが、規則的にはさまれているわけではなく、
            //データが無効な場合、そのままになる
            dts_pts_invalid_count++;
            dts_pts_invalid_keyframe_count += (framePosList[i].flags & 1);
        }
        keyframe_count += (framePosList[i].flags & 1);
    }

    //ほとんどのpts, dtsがあてにならない
    m_Demux.video.nStreamPtsInvalid  = (dts_pts_invalid_count >= framePosList.size() - 5 && dts_pts_invalid_keyframe_count >= MSDK_MAX(1, keyframe_count-2)) ? AVQSV_PTS_ALL_INVALID : 0;
    m_Demux.video.nStreamPtsInvalid |= (dts_pts_invalid_count >= (framePosList.size() - keyframe_count) - 5 && dts_pts_invalid_keyframe_count == 0)          ? AVQSV_PTS_NONKEY_INVALID : 0;

    //PAFFっぽさ (適当)
    const bool seemsLikePAFF =
        (framePosList.size() * 9 / 20 <= dts_pts_no_value_in_between)
        || (abs(1.0 - moreDataCount / (double)gotFrameCount) <= 0.2);
    m_Demux.video.nStreamPtsInvalid |= (seemsLikePAFF) ? AVQSV_PTS_HALF_INVALID : 0;

    if (m_Demux.video.nStreamPtsInvalid & AVQSV_PTS_NONKEY_INVALID) {
        //キーフレームだけptsが得られている場合は、他のフレームのptsをdurationをもとに推定する
        for (int i = m_sTrimParam.offset + 1; i < (int)framePosList.size(); i++) {
            if (framePosList[i].dts == AV_NOPTS_VALUE && framePosList[i].pts == AV_NOPTS_VALUE) {
                framePosList[i].dts = framePosList[i-1].dts + framePosList[i-1].duration;
                framePosList[i].pts = framePosList[i-1].pts + framePosList[i-1].duration;
            }
        }
    } else if (dts_pts_invalid_count > framePosList.size() / 20) {
        //pts/dtsがともにAV_NOPTS_VALUEがある場合にはdurationの再計算を行わない
        m_Demux.video.nStreamPtsInvalid |= AVQSV_PTS_SOMETIMES_INVALID;
    } else {
        //durationを再計算する (主にmkvのくそな時間解像度への対策)
        std::sort(framePosList.begin(), framePosList.end(), [](const FramePos& posA, const FramePos& posB) { return posA.pts < posB.pts; });
        for (int i = 0; i < (int)framePosList.size() - 1; i++) {
            if (framePosList[i+1].pts != AV_NOPTS_VALUE && framePosList[i].pts != AV_NOPTS_VALUE) {
                int duration = (int)(framePosList[i+1].pts - framePosList[i].pts);
                if (duration >= 0) {
                    framePosList[i].duration = duration;
                }
            }
        }
    }
    //より正確なduration計算のため、最初と最後の数フレームは落とす
    //最初と最後のフレームはBフレームによりまだ並べ替えが必要な場合があり、正確なdurationを算出しない
    const int cutoff = (framePosList.size() >= 64) ? 16 : ((uint32_t)framePosList.size() / 4);
    if (framePosList.size() >= 32) {
        vector<FramePos> newList;
        newList.reserve(framePosList.size() - cutoff - m_sTrimParam.offset);
        //最初のキーフレームからのフレームリストを構築する
        //最初のキーフレームからのリストであることは後段で、nDelayOfAudioを正確に計算するために重要
        //Bフレーム等の並べ替えを考慮し、キーフレームのpts以前のptsを持つものを削除、
        //また、最後の16フレームもBフレームを考慮してカット
        std::for_each(framePosList.begin() + m_sTrimParam.offset, framePosList.end() - cutoff,
            [&newList, firstKeyframePos](const FramePos& pos) {
            if (pos.pts == AV_NOPTS_VALUE || firstKeyframePos.pts <= pos.pts) newList.push_back(pos);
        });
        framePosList = std::move(newList);
    }

    //durationのヒストグラムを作成 (ただし、先頭は信頼ならないので、cutoff分は計算に含めない)
    vector<std::pair<int, int>> durationHistgram;
    std::for_each(framePosList.begin() + cutoff, framePosList.end(), [&durationHistgram](const FramePos& pos) {
        auto target = std::find_if(durationHistgram.begin(), durationHistgram.end(), [pos](const std::pair<int, int>& pair) { return pair.first == pos.duration; });
        if (target != durationHistgram.end()) {
            target->second++;
        } else {
            durationHistgram.push_back(std::make_pair(pos.duration, 1));
        }
    });
    //多い順にソートする
    std::sort(durationHistgram.begin(), durationHistgram.end(), [](const std::pair<int, int>& pairA, const std::pair<int, int>& pairB) { return pairA.second > pairB.second; });
    //durationが0でなく、最も頻繁に出てきたもの
    auto mostPopularDuration = durationHistgram[durationHistgram.size() > 1 && durationHistgram[0].first == 0];

    AVRational estimatedAvgFps = { 0 };
    if (mostPopularDuration.first == 0) {
        m_Demux.video.nStreamPtsInvalid |= AVQSV_PTS_ALL_INVALID;
    } else {
        //avgFpsとtargetFpsが近いかどうか
        auto fps_near = [](double avgFps, double targetFps) { return abs(1 - avgFps / targetFps) < 0.5; };
        //durationの平均を求める (ただし、先頭は信頼ならないので、cutoff分は計算に含めない)
        double avgDuration = std::accumulate(framePosList.begin() + cutoff, framePosList.end(), 0, [](const int sum, const FramePos& pos) { return sum + pos.duration; }) / (double)(framePosList.size() - cutoff);
        double avgFps = m_Demux.video.pCodecCtx->pkt_timebase.den / (double)(avgDuration * m_Demux.video.pCodecCtx->time_base.num);
        double torrelance = (fps_near(avgFps, 25.0) || fps_near(avgFps, 50.0)) ? 0.01 : 0.0008; //25fps, 50fps近辺は基準が甘くてよい
        if (mostPopularDuration.second / (double)(framePosList.size() - cutoff) > 0.95 && abs(1 - mostPopularDuration.first / avgDuration) < torrelance) {
            avgDuration = mostPopularDuration.first;
        }
        //入力フレームに対し、出力フレームが半分程度なら、フレームのdurationを倍と見積もる
        avgDuration *= (seemsLikePAFF) ? 2.0 : 1.0;
        //durationから求めた平均fpsを計算する
        const int mul = (int)ceil(1001.0 / m_Demux.video.pCodecCtx->time_base.num);
        estimatedAvgFps.num = (int)(m_Demux.video.pCodecCtx->pkt_timebase.den / avgDuration * (double)m_Demux.video.pCodecCtx->time_base.num * mul + 0.5);
        estimatedAvgFps.den = m_Demux.video.pCodecCtx->time_base.num * mul;
    }

    if (m_Demux.video.nStreamPtsInvalid & AVQSV_PTS_ALL_INVALID) {
        //ptsとdurationをpkt_timebaseで適当に作成する
        addVideoPtsToList({ 0, 0, (int)av_rescale_q(1, m_Demux.video.pCodecCtx->time_base, m_Demux.video.pCodecCtx->pkt_timebase), firstKeyframePos.flags });
        m_Demux.video.nAvgFramerate = (fpsDecoderInvalid) ? estimatedAvgFps : fpsDecoder;
    } else {
        addVideoPtsToList(firstKeyframePos);
        if (fpsDecoderInvalid) {
            m_Demux.video.nAvgFramerate = estimatedAvgFps;
        } else {
            double dFpsDecoder = fpsDecoder.num / (double)fpsDecoder.den;
            double dEstimatedAvgFps = estimatedAvgFps.num / (double)estimatedAvgFps.den;
            //2フレーム分程度がもたらす誤差があっても許容する
            if (abs(dFpsDecoder / dEstimatedAvgFps - 1.0) < (2.0 / framePosList.size())) {
                m_Demux.video.nAvgFramerate = fpsDecoder;
            } else {
                double dEstimatedAvgFpsCompare = estimatedAvgFps.num / (double)(estimatedAvgFps.den + ((dFpsDecoder < dEstimatedAvgFps) ? 1 : -1));
                //durationから求めた平均fpsがデコーダの出したfpsの近似値と分かれば、デコーダの出したfpsを採用する
                m_Demux.video.nAvgFramerate = (abs(dEstimatedAvgFps - dFpsDecoder) < abs(dEstimatedAvgFpsCompare - dFpsDecoder)) ? fpsDecoder : estimatedAvgFps;
            }
        }
    }

    const mfxU32 fps_gcd = GCD(m_Demux.video.nAvgFramerate.num, m_Demux.video.nAvgFramerate.den);
    m_Demux.video.nAvgFramerate.num /= fps_gcd;
    m_Demux.video.nAvgFramerate.den /= fps_gcd;

    //近似値であれば、分母1001/分母1に合わせる
    double fps = m_Demux.video.nAvgFramerate.num / (double)m_Demux.video.nAvgFramerate.den;
    double fps_n = fps * 1001;
    int fps_n_int = (int)(fps + 0.5) * 1000;
    if (abs(fps_n / (double)fps_n_int - 1.0) < 1e-4) {
        m_Demux.video.nAvgFramerate.num = fps_n_int;
        m_Demux.video.nAvgFramerate.den = 1001;
    } else {
        fps_n = fps * 1000;
        int fps_n_int = (int)(fps + 0.5) * 1000;
        if (abs(fps_n / (double)fps_n_int - 1.0) < 1e-4) {
            m_Demux.video.nAvgFramerate.num = fps_n_int / 1000;
            m_Demux.video.nAvgFramerate.den = 1;
        }
    }
    auto trimList = vector<sTrim>(pTrimList, pTrimList + nTrimCount);
    //出力時の音声解析用に1パケットコピーしておく
    auto& audioBuffer = m_AudioPacketsBufferL1[m_Demux.video.nSampleLoadCount % _countof(m_Demux.video.packet)];
    if (audioBuffer.size()) {
        for (auto audioInfo = m_Demux.audio.begin(); audioInfo != m_Demux.audio.end(); audioInfo++) {
            const AVPacket *pkt1 = NULL; //最初のパケット
            const AVPacket *pkt2 = NULL; //2番目のパケット
            for (int j = 0; j < (int)audioBuffer.size(); j++) {
                if (audioBuffer[j].stream_index == audioInfo->nIndex) {
                    if (pkt1) {
                        pkt2 = &audioBuffer[j];
                        break;
                    }
                    pkt1 = &audioBuffer[j];
                }
            }
            if (pkt1 != NULL) {
                //1パケット目はたまにおかしいので、可能なら2パケット目を使用する
                av_copy_packet(&audioInfo->pktSample, (pkt2) ? pkt2 : pkt1);
                if (m_Demux.video.nStreamPtsInvalid & AVQSV_PTS_ALL_INVALID) {
                    audioInfo->nDelayOfAudio = 0;
                } else {
                    //その音声の属する動画フレーム番号
                    const int vidIndex = getVideoFrameIdx(pkt1->pts, audioInfo->pCodecCtx->pkt_timebase, framePosList.data(), (int)framePosList.size(), 0);
                    if (vidIndex >= 0) {
                        //音声の遅れているフレーム数分のdurationを足し上げる
                        int delayOfAudio = (frame_inside_range(vidIndex, trimList)) ? (int)(pkt1->pts - framePosList[vidIndex].pts) : 0;
                        for (int iFrame = m_sTrimParam.offset; iFrame < vidIndex; iFrame++) {
                            if (frame_inside_range(iFrame, trimList)) {
                                delayOfAudio += framePosList[iFrame].duration;
                            }
                        }
                        audioInfo->nDelayOfAudio = delayOfAudio;
                    }
                }
            } else {
                //音声の最初のサンプルを取得できていない
                audioInfo = m_Demux.audio.erase(audioInfo) - 1;
            }
        }
        if (audioBuffer.size() == 0) {
            //音声の最初のサンプルを取得できていないため、音声がすべてなくなってしまった
            m_strInputInfo += _T("avcodec reader: failed to find audio stream in preread.\n");
            return MFX_ERR_UNDEFINED_BEHAVIOR;
        }
    }
    //音声パケットもm_PreReadBufferに接続する
    m_PreReadBuffer.insert(m_PreReadBuffer.end(), audioBuffer.begin(), audioBuffer.end());
    audioBuffer.clear();
    m_Demux.format.nPreReadBufferIdx = 0; //PreReadBufferからの読み込み優先にする

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
    //    m_strInputInfo += _T("avcodec: invalid dlls for QSVEncC.\n");
    //    return MFX_ERR_NULL_PTR;
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
    InitializeCriticalSection(&m_Demux.video.frameData.cs);
    m_Demux.video.frameData.cs_initialized = true;

    const AvcodecReaderPrm *input_prm = (const AvcodecReaderPrm *)option;
    
    std::string filename_char;
    if (0 == tchar_to_string(strFileName, filename_char, CP_UTF8)) {
        m_strInputInfo += _T("avcodec reader: failed to convert filename to ansi characters.\n");
        return MFX_ERR_INVALID_HANDLE;
    }
    m_Demux.format.bIsPipe = (0 == strcmp(filename_char.c_str(), "-")) || filename_char.c_str() == strstr(filename_char.c_str(), R"(\\.\pipe\)");
    m_Demux.format.pFormatCtx = avformat_alloc_context();
    m_Demux.format.nAnalyzeSec = input_prm->nAnalyzeSec;
    if (m_Demux.format.nAnalyzeSec) {
        if (av_opt_set_int(m_Demux.format.pFormatCtx, "probesize", m_Demux.format.nAnalyzeSec * AV_TIME_BASE, 0)) {
            AVDEBUG_PRINT("avcodec reader: failed to set probesize.\n");
        }
    }
    if (0 == strcmp(filename_char.c_str(), "-")) {
        if (_setmode(_fileno(stdin), _O_BINARY) < 0) {
            m_strInputInfo += _T("avcodec reader: failed to switch stdin to binary mode.");
            return MFX_ERR_UNKNOWN;
        }
        filename_char = "pipe:0";
    }
    if (avformat_open_input(&(m_Demux.format.pFormatCtx), filename_char.c_str(), nullptr, nullptr)) {
        m_strInputInfo += _T("avcodec reader: error opening file.\n");
        return MFX_ERR_NULL_PTR; // Couldn't open file
    }

    AVDEBUG_PRINT("avcodec reader: opened file.\n");
    if (m_Demux.format.nAnalyzeSec) {
        if (av_opt_set_int(m_Demux.format.pFormatCtx, "analyzeduration", m_Demux.format.nAnalyzeSec * AV_TIME_BASE, 0)) {
            AVDEBUG_PRINT("avcodec reader: failed to set analyzeduration.\n");
        }
    }
    if (avformat_find_stream_info(m_Demux.format.pFormatCtx, nullptr) < 0) {
        m_strInputInfo += _T("avcodec reader: error finding stream information.\n");
        return MFX_ERR_NULL_PTR; // Couldn't find stream information
    }
    AVDEBUG_PRINT("avcodec reader: got stream information.\n");
    //dump_format(dec.pFormatCtx, 0, argv[1], 0);
    
    //動画ストリームを探す
    auto videoStreams = getStreamIndex(AVMEDIA_TYPE_VIDEO);
    if (videoStreams.size() == 0) {
        m_strInputInfo += _T("avcodec reader: unable to find video stream.\n");
        return MFX_ERR_NULL_PTR; // Didn't find a video stream
    }
    m_Demux.video.nIndex = videoStreams[0];
    AVDEBUG_PRINT("avcodec reader: found video stream.\n");

    m_Demux.video.pCodecCtx = m_Demux.format.pFormatCtx->streams[m_Demux.video.nIndex]->codec;

    //QSVでデコード可能かチェック
    if (0 == (m_nInputCodec = getQSVFourcc(m_Demux.video.pCodecCtx->codec_id))) {
        m_strInputInfo += _T("avcodec reader: codec ");
        if (m_Demux.video.pCodecCtx->codec && m_Demux.video.pCodecCtx->codec->name) {
            m_strInputInfo += char_to_tstring(m_Demux.video.pCodecCtx->codec->name);
            m_strInputInfo += _T(" ");
        }
        m_strInputInfo += _T("unable to decode by qsv.\n");
        return MFX_ERR_NULL_PTR;
    }
    AVDEBUG_PRINT("avcodec reader: can be decoded by qsv.\n");

    //音声ストリームを探す
    if (input_prm->nReadAudio) {
        auto audioStreams = getStreamIndex(AVMEDIA_TYPE_AUDIO);
        if (audioStreams.size() == 0) {
            m_strInputInfo += _T("avcodec reader: --audio-file or --copy-audio is set, but no audio stream found.\n");
            return MFX_ERR_NOT_FOUND;
        } else {
            for (int iAudTrack = 0; iAudTrack < (int)audioStreams.size(); iAudTrack++) {
                //特に指定なし = 全指定かどうか
                bool useStream = (input_prm->nReadAudio & AVQSV_AUDIO_MUX) && (input_prm->pAudioSelect == NULL || input_prm->nAudioSelectCount == 0);
                //選択されているかをチェック
                for (int i = 0; !useStream && i < input_prm->nAudioSelectCount; i++) {
                    useStream = (input_prm->pAudioSelect[i] == (iAudTrack+1)); //トラック番号は1から連番
                }
                for (int i = 0; !useStream && i < input_prm->nAudioExtractFileCount; i++) {
                    useStream = (input_prm->pAudioExtractFileSelect[i] == (iAudTrack+1)); //トラック番号は1から連番
                }
                if (useStream) {
                    AVDemuxAudio audio = { 0 };
                    audio.nTrackId = iAudTrack+1; //トラック番号は1から連番
                    audio.nIndex = audioStreams[iAudTrack];
                    audio.pCodecCtx = m_Demux.format.pFormatCtx->streams[audio.nIndex]->codec;
                    m_Demux.audio.push_back(audio);
                }
            }
        }
    }

    //必要ならbitstream filterを初期化
    if (m_Demux.video.pCodecCtx->extradata && m_Demux.video.pCodecCtx->extradata[0] == 1) {
        if (m_nInputCodec == MFX_CODEC_AVC) {
            if (NULL == (m_Demux.video.pH264Bsfc = av_bitstream_filter_init("h264_mp4toannexb"))) {
                m_strInputInfo += _T("avcodec reader: unable to init h264_mp4toannexb.\n");
                return MFX_ERR_NULL_PTR;
            }
            AVDEBUG_PRINT("avcodec reader: success to init h264_mp4toannexb.\n");
        } else if (m_nInputCodec == MFX_CODEC_HEVC) {
            m_Demux.video.bUseHEVCmp42AnnexB = true;
        }
    } else if (m_Demux.video.pCodecCtx->extradata == NULL && m_Demux.video.pCodecCtx->extradata_size == 0) {
        m_strInputInfo += _T("avcodec reader: video header not extracted by libavcodec.\n");
        return MFX_ERR_UNSUPPORTED;
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
    } else if (MFX_ERR_NONE != (decHeaderSts = getFirstFramePosAndFrameRate({ m_sDecParam.mfx.FrameInfo.FrameRateExtN, m_sDecParam.mfx.FrameInfo.FrameRateExtD }, session, &bitstream, input_prm->pTrimList, input_prm->nTrimCount))) {
        m_strInputInfo += _T("avcodec reader: failed to get first frame position.\n");
    }
    MFXVideoDECODE_Close(session);
    pPlugin.reset(); //必ずsessionをクローズする前に開放すること
    MFXClose(session);
    if (MFX_ERR_NONE != decHeaderSts) {
        m_strInputInfo += _T("avcodec reader: unable to decode by qsv, please consider using other input method.\n");
        return decHeaderSts;
    }
    WipeMfxBitstream(&bitstream);

    m_sTrimParam.list = vector<sTrim>(input_prm->pTrimList, input_prm->pTrimList + input_prm->nTrimCount);
    //キーフレームに到達するまでQSVではフレームが出てこない
    //そのぶんのずれを記録しておき、Trim値などに補正をかける
    if (m_sTrimParam.offset) {
        for (int i = (int)m_sTrimParam.list.size() - 1; i >= 0; i--) {
            if (m_sTrimParam.list[i].fin - m_sTrimParam.offset < 0) {
                m_sTrimParam.list.erase(m_sTrimParam.list.begin() + i);
            } else {
                m_sTrimParam.list[i].start = (std::max)(0, m_sTrimParam.list[i].start - m_sTrimParam.offset);
                if (m_sTrimParam.list[i].fin != TRIM_MAX) {
                    m_sTrimParam.list[i].fin = (std::max)(0, m_sTrimParam.list[i].fin - m_sTrimParam.offset);
                }
            }
        }
        //ずれが存在し、範囲指定がない場合はダミーの全域指定を追加する
        //これにより、自動的に音声側との同期がとれるようになる
        if (m_sTrimParam.list.size() == 0) {
            m_sTrimParam.list.push_back({ 0, TRIM_MAX });
        }
    }

    //getFirstFramePosAndFrameRateをもとにfpsを決定
    m_sDecParam.mfx.FrameInfo.FrameRateExtN = m_Demux.video.nAvgFramerate.num;
    m_sDecParam.mfx.FrameInfo.FrameRateExtD = m_Demux.video.nAvgFramerate.den;

    //情報を格納
    memcpy(&m_inputFrameInfo, &m_sDecParam.mfx.FrameInfo, sizeof(m_inputFrameInfo));
    //m_inputFrameInfoのWidthとHeightには解像度をそのまま入れて、
    //他の読み込みに合わせる
    //もともとは16ないし32でアラインされた数字が入っている
    m_inputFrameInfo.Width          = m_inputFrameInfo.CropW;
    m_inputFrameInfo.Height         = m_inputFrameInfo.CropH;
    m_inputFrameInfo.BitDepthLuma   = m_inputFrameInfo.BitDepthLuma;
    m_inputFrameInfo.BitDepthChroma = m_inputFrameInfo.BitDepthChroma;
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

int CAvcodecReader::getVideoFrameIdx(mfxI64 pts, AVRational timebase, const FramePos *framePos, int framePosCount, int iStart) {
    for (int i = max(0, iStart); i < framePosCount; i++) {
        //pts < demux.videoFramePts[i]であるなら、その前のフレームを返す
        if (0 > av_compare_ts(pts, timebase, framePos[i].pts, m_Demux.video.pCodecCtx->pkt_timebase)) {
            return i - 1;
        }
    }
    return framePosCount;
}

mfxI64 CAvcodecReader::convertTimebaseVidToAud(mfxI64 pts, const AVDemuxAudio *pAudio) {
    return av_rescale_q(pts, m_Demux.video.pCodecCtx->pkt_timebase, pAudio->pCodecCtx->pkt_timebase);
}

bool CAvcodecReader::checkAudioPacketToAdd(const AVPacket *pkt, AVDemuxAudio *pAudio) {
    pAudio->nLastVidIndex = getVideoFrameIdx(pkt->pts, pAudio->pCodecCtx->pkt_timebase, m_Demux.video.frameData.frame, m_Demux.video.frameData.num, pAudio->nLastVidIndex);

    //該当フレームが-1フレーム未満なら、その音声はこの動画には含まれない
    if (pAudio->nLastVidIndex < -1) {
        return false;
    }

    const FramePos *vidFramePos = &m_Demux.video.frameData.frame[(std::max)(pAudio->nLastVidIndex, 0)];
    const mfxI64 vid_fin = convertTimebaseVidToAud(vidFramePos->pts + ((pAudio->nLastVidIndex >= 0) ? vidFramePos->duration : 0), pAudio);

    const mfxI64 aud_start = pkt->pts;
    const mfxI64 aud_fin   = pkt->pts + pkt->duration;

    const bool frame_is_in_range = frame_inside_range(pAudio->nLastVidIndex,     m_sTrimParam.list);
    const bool next_is_in_range  = frame_inside_range(pAudio->nLastVidIndex + 1, m_sTrimParam.list);

    bool result = true; //動画に含まれる音声かどうか

    if (frame_is_in_range) {
        if (aud_fin < vid_fin || next_is_in_range) {
            ; //完全に動画フレームの範囲内か、次のフレームも範囲内なら、その音声パケットは含まれる
        //              vid_fin
        //動画 <-----------|
        //音声      |-----------|
        //     aud_start     aud_fin
        } else if (pkt->duration / 2 > (aud_fin - vid_fin + pAudio->nExtractErrExcess)) {
            //はみ出した領域が少ないなら、その音声パケットは含まれる
            pAudio->nExtractErrExcess += aud_fin - vid_fin;
        } else {
            //はみ出した領域が多いなら、その音声パケットは含まれない
            pAudio->nExtractErrExcess -= vid_fin - aud_start;
            result = false;
        }
    } else if (next_is_in_range && aud_fin > vid_fin) {
        //             vid_fin
        //動画             |------------>
        //音声      |-----------|
        //     aud_start     aud_fin
        if (pkt->duration / 2 > (vid_fin - aud_start + pAudio->nExtractErrExcess)) {
            pAudio->nExtractErrExcess += vid_fin - aud_start;
        } else {
            pAudio->nExtractErrExcess -= aud_fin - vid_fin;
            result = false;
        }
    } else {
        result = false;
    }
    return result;
}

AVDemuxAudio *CAvcodecReader::getAudioPacketStreamData(const AVPacket *pkt) {
    int streamIndex = pkt->stream_index;
    for (int i = 0; i < (int)m_Demux.audio.size(); i++) {
        if (m_Demux.audio[i].nIndex == streamIndex) {
            return &m_Demux.audio[i];
        }
    }
    return NULL;
}

int CAvcodecReader::getSample(AVPacket *pkt) {
    av_init_packet(pkt);
    auto get_sample = [this](AVPacket *pkt, bool *fromPreReadBuffer) {
        if (m_Demux.format.nPreReadBufferIdx < m_PreReadBuffer.size()) {
            *pkt = m_PreReadBuffer[m_Demux.format.nPreReadBufferIdx];
            m_Demux.format.nPreReadBufferIdx++;
            if (m_Demux.format.nPreReadBufferIdx >= m_PreReadBuffer.size()) {
                m_PreReadBuffer.clear();
                m_Demux.format.nPreReadBufferIdx = UINT32_MAX;
            }
            *fromPreReadBuffer = true;
            return 0;
        } else {
            return av_read_frame(m_Demux.format.pFormatCtx, pkt);
        }
    };
    bool fromPreReadBuffer = false;
    while (get_sample(pkt, &fromPreReadBuffer) >= 0) {
        if (pkt->stream_index == m_Demux.video.nIndex) {
            if (!fromPreReadBuffer) {
                if (m_Demux.video.pH264Bsfc) {
                    mfxU8 *data = NULL;
                    int dataSize = 0;
                    std::swap(m_Demux.video.pExtradata, m_Demux.video.pCodecCtx->extradata);
                    std::swap(m_Demux.video.nExtradataSize, m_Demux.video.pCodecCtx->extradata_size);
                    av_bitstream_filter_filter(m_Demux.video.pH264Bsfc, m_Demux.video.pCodecCtx, nullptr,
                        &data, &dataSize, pkt->data, pkt->size, 0);
                    std::swap(m_Demux.video.pExtradata, m_Demux.video.pCodecCtx->extradata);
                    std::swap(m_Demux.video.nExtradataSize, m_Demux.video.pCodecCtx->extradata_size);
                    av_free_packet(pkt); //メモリ解放を忘れない
                    av_packet_from_data(pkt, data, dataSize);
                }
                if (m_Demux.video.bUseHEVCmp42AnnexB) {
                    hevcMp42Annexb(pkt);
                }
            }
            //最初のptsが格納されていたら( = getFirstFramePosAndFrameRate()が実行済み)、後続のptsを格納していく
            if (m_Demux.video.frameData.num) {
                //最初のキーフレームを取得するまではスキップする
                if (!m_Demux.video.bGotFirstKeyframe && !(pkt->flags & AV_PKT_FLAG_KEY)) {
                    av_free_packet(pkt);
                    continue;
                } else {
                    m_Demux.video.bGotFirstKeyframe = true;
                    //AVPacketのもたらすptsが無効であれば、CFRを仮定して適当にptsとdurationを突っ込んでいく
                    //0フレーム目は格納されているので、その次からを格納する
                    if ((m_Demux.video.nStreamPtsInvalid & AVQSV_PTS_ALL_INVALID) && m_Demux.video.nSampleLoadCount) {
                        int duration = m_Demux.video.frameData.frame[0].duration;
                        int64_t pts = m_Demux.video.nSampleLoadCount * duration;
                        addVideoPtsToList({ pts, pts, duration, pkt->flags });
                        //最初のptsは格納されているので、その次からを格納する
                    } else {
                        int64_t pts = pkt->pts, dts = pkt->dts;
                        if ((m_Demux.video.nStreamPtsInvalid & AVQSV_PTS_NONKEY_INVALID) && pts == AV_NOPTS_VALUE) {
                            //キーフレーム以外のptsとdtsが無効な場合は、適当に推定する
                            const FramePos *lastFrame = &m_Demux.video.frameData.frame[m_Demux.video.frameData.num-1];
                            int duration = lastFrame->duration;
                            pts = lastFrame->pts + duration;
                            dts = lastFrame->dts + duration;
                        }
                        addVideoPtsToList({ (pts == AV_NOPTS_VALUE) ? dts : pts, dts, pkt->duration, pkt->flags });
                    }
                }
            }
            return 0;
        }
        if (getAudioPacketStreamData(pkt) != NULL) {
            //音声パケットはひとまずすべてバッファに格納する
            m_AudioPacketsBufferL1[m_Demux.video.nSampleLoadCount % _countof(m_Demux.video.packet)].push_back(*pkt);
        } else {
            av_free_packet(pkt);
        }
    }
    //ファイルの終わりに到達
    pkt->data = nullptr;
    pkt->size = 0;
    sortVideoPtsList();
    //動画の終端を表す最後のptsを挿入する
    int64_t videoFinPts = 0;
    if (m_Demux.video.nStreamPtsInvalid & AVQSV_PTS_ALL_INVALID) {
        videoFinPts = m_Demux.video.nSampleLoadCount * m_Demux.video.frameData.frame[0].duration;
    } else if (m_Demux.video.frameData.num) {
        const FramePos *lastFrame = &m_Demux.video.frameData.frame[m_Demux.video.frameData.num - 1];
        videoFinPts = lastFrame->pts + lastFrame->duration;
    }
    //もし選択範囲が手動で決定されていないのなら、音声を最大限取得する
    if (m_sTrimParam.list.size() == 0 || m_sTrimParam.list.back().fin == TRIM_MAX) {
        if (m_AudioPacketsBufferL2.size()) {
            videoFinPts = (std::max)(videoFinPts, m_AudioPacketsBufferL2.back().pts + m_AudioPacketsBufferL2.back().duration);
        }
        for (auto packetsL1 : m_AudioPacketsBufferL1) {
            if (packetsL1.size()) {
                videoFinPts = (std::max)(videoFinPts, packetsL1.back().pts + packetsL1.back().duration);
            }
        }
    }
    addVideoPtsToList({ videoFinPts, videoFinPts, 0 });
    m_Demux.video.frameData.fixed_num = m_Demux.video.frameData.num - 1;
    m_Demux.video.frameData.duration = m_Demux.format.pFormatCtx->duration;
    m_pEncSatusInfo->UpdateDisplay(timeGetTime(), 0, 100.0);
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
    mfxStatus sts = setToMfxBitstream(bitstream, &m_Demux.video.packet[m_Demux.video.nSampleGetCount % _countof(m_Demux.video.packet)]);
    m_Demux.video.nSampleGetCount++;
    return sts;
}

vector<AVPacket> CAvcodecReader::GetAudioDataPackets() {
    //すでに使用した音声バッファはクリアする
    if (m_AudioPacketsBufferL2Used) {
        //使用済みパケットを削除する
        //これらのパケットはすでにWriter側に渡っているか、解放されているので、av_free_packetは不要
        m_AudioPacketsBufferL2.erase(m_AudioPacketsBufferL2.begin(), m_AudioPacketsBufferL2.begin() + m_AudioPacketsBufferL2Used);
    }
    m_AudioPacketsBufferL2Used = 0;

    //別スレッドで使用されていないほうを連結する
    const auto& packetsL1 = m_AudioPacketsBufferL1[m_Demux.video.nSampleGetCount % _countof(m_AudioPacketsBufferL1)];
    m_AudioPacketsBufferL2.insert(m_AudioPacketsBufferL2.end(), packetsL1.begin(), packetsL1.end());

    //出力するパケットを選択する
    vector<AVPacket> packets;
    EnterCriticalSection(&m_Demux.video.frameData.cs);
    for (mfxU32 i = 0; i < m_AudioPacketsBufferL2.size(); i++) {
        AVPacket *pkt = &m_AudioPacketsBufferL2[i];
        AVDemuxAudio *pAudio = getAudioPacketStreamData(pkt);
        //音声のptsが映像の終わりのptsを行きすぎたらやめる
        if (0 < av_compare_ts(pkt->pts, pAudio->pCodecCtx->pkt_timebase, m_Demux.video.frameData.frame[m_Demux.video.frameData.fixed_num].pts, m_Demux.video.pCodecCtx->pkt_timebase)) {
            break;
        }
        m_AudioPacketsBufferL2Used++;
        if (checkAudioPacketToAdd(pkt, pAudio)) {
            packets.push_back(*pkt); //Writer側に渡したパケットはWriter側で開放する
        } else {
            av_free_packet(pkt); //Writer側に渡さないパケットはここで開放する
            pkt->data = NULL;
            pkt->size = 0;
        }
    }
    LeaveCriticalSection(&m_Demux.video.frameData.cs);
    return std::move(packets);
}

vector<AVDemuxAudio> CAvcodecReader::GetInputAudioInfo() {
    return vector<AVDemuxAudio>(m_Demux.audio.begin(), m_Demux.audio.end());
}

mfxStatus CAvcodecReader::GetHeader(mfxBitstream *bitstream) {
    if (bitstream == nullptr)
        return MFX_ERR_NULL_PTR;
    if (bitstream->Data == nullptr)
        if (MFX_ERR_NONE != InitMfxBitstream(bitstream, AVCODEC_READER_INPUT_BUF_SIZE))
            return MFX_ERR_NULL_PTR;

    if (m_Demux.video.pExtradata == nullptr) {
        m_Demux.video.nExtradataSize = m_Demux.video.pCodecCtx->extradata_size;
        //ここでav_mallocを使用しないと正常に動作しない
        m_Demux.video.pExtradata = (mfxU8 *)av_malloc(m_Demux.video.pCodecCtx->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE);
        //ヘッダのデータをコピーしておく
        memcpy(m_Demux.video.pExtradata, m_Demux.video.pCodecCtx->extradata, m_Demux.video.nExtradataSize);
        memset(m_Demux.video.pExtradata + m_Demux.video.nExtradataSize, 0, FF_INPUT_BUFFER_PADDING_SIZE);

        if (m_Demux.video.bUseHEVCmp42AnnexB) {
            hevcMp42Annexb(NULL);
        } else if (m_Demux.video.pH264Bsfc && m_Demux.video.pExtradata[0] == 1) {
            mfxU8 *dummy = NULL;
            int dummy_size = 0;
            std::swap(m_Demux.video.pExtradata,     m_Demux.video.pCodecCtx->extradata);
            std::swap(m_Demux.video.nExtradataSize, m_Demux.video.pCodecCtx->extradata_size);
            av_bitstream_filter_filter(m_Demux.video.pH264Bsfc, m_Demux.video.pCodecCtx, nullptr, &dummy, &dummy_size, nullptr, 0, 0);
            std::swap(m_Demux.video.pExtradata,     m_Demux.video.pCodecCtx->extradata);
            std::swap(m_Demux.video.nExtradataSize, m_Demux.video.pCodecCtx->extradata_size);
        }
    }
    
    memcpy(bitstream->Data, m_Demux.video.pExtradata, m_Demux.video.nExtradataSize);
    bitstream->DataLength = m_Demux.video.nExtradataSize;
    return MFX_ERR_NONE;
}

#pragma warning(push)
#pragma warning(disable:4100)
mfxStatus CAvcodecReader::LoadNextFrame(mfxFrameSurface1* pSurface) {
    AVPacket *pkt = &m_Demux.video.packet[m_Demux.video.nSampleLoadCount % _countof(m_Demux.video.packet)];
    m_AudioPacketsBufferL1[m_Demux.video.nSampleLoadCount % _countof(m_AudioPacketsBufferL1)].clear();

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
    m_Demux.video.nSampleLoadCount++;
    m_pEncSatusInfo->m_nInputFrames++;
    mfxU32 tm = timeGetTime();
    if (tm - m_tmLastUpdate > UPDATE_INTERVAL) {
        double progressPercent = 0.0;
        if (m_Demux.format.pFormatCtx->duration) {
            progressPercent = m_Demux.video.frameData.duration * (m_Demux.video.pCodecCtx->pkt_timebase.num / (double)m_Demux.video.pCodecCtx->pkt_timebase.den) / (m_Demux.format.pFormatCtx->duration * (1.0 / (double)AV_TIME_BASE)) * 100.0;
        }
        m_tmLastUpdate = tm;
        m_pEncSatusInfo->UpdateDisplay(tm, 0, progressPercent);
    }
    return MFX_ERR_NONE;
}
#pragma warning(pop)

#endif //ENABLE_AVCODEC_QSV_READER
