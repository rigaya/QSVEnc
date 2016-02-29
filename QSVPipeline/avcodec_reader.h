//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------
#ifndef _AVCODEC_READER_H_
#define _AVCODEC_READER_H_

#include "qsv_input.h"

#if ENABLE_AVCODEC_QSV_READER
#include "avcodec_qsv.h"
#include "qsv_queue.h"
#include <cassert>

using std::vector;
using std::pair;

static const uint32_t AVCODEC_READER_INPUT_BUF_SIZE = 16 * 1024 * 1024;
static const uint32_t AVQSV_FRAME_MAX_REORDER = 16;
static const uint32_t AVQSV_POC_INVALID = UINT32_MAX;

enum {
    AVQSV_AUDIO_NONE         = 0x00,
    AVQSV_AUDIO_MUX          = 0x01,
    AVQSV_AUDIO_COPY_TO_FILE = 0x02,
};

enum : uint32_t {
    AVQSV_PTS_SOMETIMES_INVALID = 0x01, //時折、無効なptsやdtsを得る
    AVQSV_PTS_HALF_INVALID      = 0x02, //PAFFなため、半分のフレームのptsやdtsが無効
    AVQSV_PTS_ALL_INVALID       = 0x04, //すべてのフレームのptsやdtsが無効
    AVQSV_PTS_NONKEY_INVALID    = 0x08, //キーフレーム以外のフレームのptsやdtsが無効
};

//フレームの位置情報と長さを格納する
typedef struct FramePos {
    int64_t pts;  //pts
    int64_t dts;  //dts
    int duration;  //該当フレーム/フィールドの表示時間
    int duration2; //ペアフィールドの表示時間
    uint32_t poc; //
    uint8_t flags;    //flags
    uint8_t pic_struct;
    uint8_t repeat_pict;
    uint8_t pict_type;
} FramePos;

static FramePos framePos(int64_t pts, int64_t dts,
    int duration, int duration2 = 0,
    uint32_t poc = AVQSV_POC_INVALID,
    uint8_t flags = 0, uint8_t pic_struct = 0, uint8_t repeat_pict = 0, uint8_t pict_type = 0) {
    FramePos pos;
    pos.pts = pts;
    pos.dts = dts;
    pos.duration = duration;
    pos.duration2 = duration2;
    pos.poc = poc;
    pos.flags = flags;
    pos.pic_struct = pic_struct;
    pos.repeat_pict = repeat_pict;
    pos.pict_type = pict_type;
    return pos;
}

class FramePosList {
public:
    FramePosList() :
        m_list(),
        m_nNextFixNumIndex(0),
        m_bInputFin(false),
        m_nDuration(0),
        m_nDurationNum(0),
        m_nStreamPtsInvalid(0),
        m_nLastPoc(0),
        m_nFirstKeyframePts(AV_NOPTS_VALUE) {
        m_list.init();
    };
    virtual ~FramePosList() {
        clear();
    }
    int printList(const TCHAR *filename) {
        const uint32_t nList = (uint32_t)m_list.size();
        if (nList == 0) {
            return 0;
        }
        if (filename == nullptr) {
            return 1;
        }
        FILE *fp = NULL;
        if (0 != _tfopen_s(&fp, filename, _T("wb"))) {
            return 1;
        }
        fprintf(fp, "pts,dts,duration,duration2,poc,flags,pic_struct,repeat_pict,pict_type\r\n");
        for (uint32_t i = 0; i < nList; i++) {
            fprintf(fp, "%I64d,%I64d,%d,%d,%d,%d,%d,%d,%d\r\n",
                m_list[i].data.pts, m_list[i].data.dts,
                m_list[i].data.duration, m_list[i].data.duration2,
                m_list[i].data.poc,
                (int)m_list[i].data.flags, (int)m_list[i].data.pic_struct, (int)m_list[i].data.repeat_pict, (int)m_list[i].data.pict_type);
        }
        fclose(fp);
        return 0;
    }
    //indexの位置への参照を返す
    // !! push側のスレッドからのみ有効 !!
    FramePos& list(uint32_t index) {
        return m_list[index].data;
    }
    void clear() {
        m_list.close();
        m_nNextFixNumIndex = 0;
        m_bInputFin = false;
        m_nDuration = 0;
        m_nDurationNum = 0;
        m_nStreamPtsInvalid = 0;
        m_nLastPoc = 0;
        m_nFirstKeyframePts = AV_NOPTS_VALUE;
        m_list.init();
    }
    void setStreamPtsCondition(uint32_t nStreamPtsInvalid) {
        m_nStreamPtsInvalid = nStreamPtsInvalid;
    }
    //ここまで計算したdurationを返す
    int64_t duration() const {
        return m_nDuration;
    }
    //登録された(ptsの確定していないものを含む)フレーム数を返す
    uint32_t frameNum() const {
        return (uint32_t)m_list.size();
    }
    //ptsが確定したフレーム数を返す
    uint32_t fixedNum() const {
        return m_nNextFixNumIndex;
    }
    //FramePosを追加し、内部状態を変更する
    void add(const FramePos& pos) {
        m_list.push(pos);
        const int nListSize = m_list.size();
        //自分のフレームのインデックス
        const uint32_t nIndex = nListSize-1;
        //ptsの補正
        if (m_list[nIndex].data.pts == AV_NOPTS_VALUE) {
            if (m_nStreamPtsInvalid & AVQSV_PTS_NONKEY_INVALID) {
                //AVPacketのもたらすptsが無効であれば、CFRを仮定して適当にptsとdurationを突っ込んでいく
                int duration = m_list[0].data.duration;
                m_list[nIndex].data.pts = nIndex * (int64_t)duration;
                m_list[nIndex].data.dts = m_list[nIndex].data.pts;
            } else if (m_nStreamPtsInvalid & AVQSV_PTS_NONKEY_INVALID) {
                //キーフレーム以外のptsとdtsが無効な場合は、適当に推定する
                int duration = m_list[nIndex-1].data.duration;
                m_list[nIndex].data.pts = m_list[nIndex-1].data.pts + duration;
                m_list[nIndex].data.dts = m_list[nIndex-1].data.dts + duration;
            } else if (m_nStreamPtsInvalid & AVQSV_PTS_HALF_INVALID) {
                //ptsがないのは音声抽出で、正常に抽出されない問題が生じる
                //半分PTSがないPAFFのような動画については、前のフレームからの補完を行う
                if (m_list[nIndex].data.dts == AV_NOPTS_VALUE) {
                    m_list[nIndex].data.dts = m_list[nIndex-1].data.dts + m_list[nIndex-1].data.duration;
                }
                m_list[nIndex].data.pts = m_list[nIndex-1].data.pts + m_list[nIndex-1].data.duration;
            }
        }
        //最初のキーフレームの位置を記憶しておく
        if (m_nFirstKeyframePts == AV_NOPTS_VALUE && (pos.flags & AV_PKT_FLAG_KEY) && nIndex == 0) {
            m_nFirstKeyframePts = m_list[nIndex].data.pts;
        }
        if (m_bInputFin || nListSize - m_nNextFixNumIndex > AVQSV_FRAME_MAX_REORDER) {
            //ソートの前にptsなどが設定されていない場合など適当に調整する
            adjustInfo();
            //ptsでソート
            sortPts(m_nNextFixNumIndex, std::min<uint32_t>(nListSize - m_nNextFixNumIndex, AVQSV_FRAME_MAX_REORDER));

            if (m_list[m_nNextFixNumIndex].data.pts < m_nFirstKeyframePts) {
                //ソートの先頭のptsが塚下キーフレームの先頭のptsよりも小さいことがある(opengop)
                //これはフレームリストから取り除く
                m_list.pop();
            } else {
                //ソートにより確定したptsに対して、pocを設定する
                setPoc(m_nNextFixNumIndex);
                m_nNextFixNumIndex++;
            }
        }
        calcDuration();
    };
    //pocの一致するフレームの情報のコピーを返す
    FramePos copy(uint32_t poc, uint32_t *lastIndex) {
        assert(lastIndex != nullptr);
        for (uint32_t index = *lastIndex + 1; ; index++) {
            FramePos pos;
            if (!m_list.copy(&pos, index)) {
                break;
            }
            if (pos.poc == poc) {
                return pos;
            }
        }
        //エラー
        FramePos pos = { 0 };
        pos.poc = AVQSV_POC_INVALID;
        return pos;
    }
    //入力が終了した際に使用し、内部状態を変更する
    void fin(const FramePos& pos, int64_t total_duration) {
        m_bInputFin = true;
        const int nFrame = m_list.size();
        adjustInfo();
        sortPts(m_nNextFixNumIndex, nFrame - m_nNextFixNumIndex);
        for (int i = m_nNextFixNumIndex; i < nFrame; i++) {
            setPoc(i);
        }
        m_nNextFixNumIndex = nFrame;
        add(pos);
        m_nDuration = total_duration;
        m_nDurationNum = m_nNextFixNumIndex;
    }
protected:
    //ptsでソート
    void sortPts(uint32_t index, uint32_t len) {
        std::sort(m_list.get(index), m_list.get(index + len), [](const auto& posA, const auto& posB) {
            return ((uint32_t)std::abs(posA.data.pts - posB.data.pts) < 0xFFFFFFFF) ? posA.data.pts < posB.data.pts : posB.data.pts < posA.data.pts; });
    }
    //ソートの前にptsなどが設定されていない場合など適当に調整する
    void adjustInfo() {
        const int nListSize = m_list.size();
        if (m_nStreamPtsInvalid & AVQSV_PTS_SOMETIMES_INVALID) {
            //ptsがあてにならない時は、dtsから適当に生成する
            for (int i = m_nNextFixNumIndex; i < nListSize; i++) {
                if (m_list[i].data.dts == AV_NOPTS_VALUE) {
                    //まずdtsがない場合は、前のフレームからコピーする
                    m_list[i].data.dts = m_list[i-1].data.dts;
                }
            }
            int64_t firstFramePtsDtsDiff = m_list[0].data.pts - m_list[0].data.dts;
            for (int i = m_nNextFixNumIndex; i < nListSize; i++) {
                m_list[i].data.pts = m_list[i].data.dts + firstFramePtsDtsDiff;
            }
        } else {
            for (int i = m_nNextFixNumIndex; i < nListSize; i++) {
                if (i > 0 && m_list[i].data.pts == AV_NOPTS_VALUE) {
                    m_list[i].data.pts = m_list[i-1].data.pts + m_list[i-1].data.duration;
                }
            }
        }
    }
    //ソートにより確定したptsに対して、pocを設定する
    void setPoc(int index) {
        if (m_list[index].data.pic_struct) {
            if (index > 0 && (m_list[index-1].data.poc != AVQSV_POC_INVALID && m_list[index-1].data.pic_struct)) {
                m_list[index].data.poc = AVQSV_POC_INVALID;
                m_list[index-1].data.duration2 = m_list[index].data.duration;
            } else {
                m_list[index].data.poc = m_nLastPoc++;
            }
        } else {
            m_list[index].data.poc = m_nLastPoc++;
        }
    }
    void calcDuration() {
        //進捗表示用のdurationの計算を行う
        //これは16フレームに1回行う
        int nNonDurationCalculatedFrames = m_nNextFixNumIndex - m_nDurationNum;
        if (nNonDurationCalculatedFrames >= 16) {
            const auto *pos_fixed = m_list.get(m_nDurationNum);
            int64_t duration = pos_fixed[nNonDurationCalculatedFrames-1].data.pts - pos_fixed[0].data.pts;
            if (duration < 0 || duration > 0xFFFFFFFF) {
                duration = 0;
                for (int i = 1; i < nNonDurationCalculatedFrames; i++) {
                    int64_t diff = (std::max<int64_t>)(0, pos_fixed[i].data.pts - pos_fixed[i-1].data.pts);
                    int64_t last_frame_dur = (std::max<int64_t>)(0, pos_fixed[i-1].data.duration);
                    duration += (diff > 0xFFFFFFFF) ? last_frame_dur : diff;
                }
            }
            m_nDuration += duration;
            m_nDurationNum += nNonDurationCalculatedFrames;
        }
    }
protected:
    CQueueSPSP<FramePos, 1> m_list; //内部データサイズとFramePosのデータサイズを一致させるため、alignを1に設定
    int m_nNextFixNumIndex; //次にptsを確定させるフレームのインデックス
    bool m_bInputFin; //入力が終了したことを示すフラグ
    int64_t m_nDuration; //m_nDurationNumのフレーム数分のdurationの総和
    int m_nDurationNum; //durationを計算したフレーム数
    uint32_t m_nStreamPtsInvalid; //入力から提供されるptsの状態 (AVQSV_PTS_xxx)
    uint32_t m_nLastPoc; //ptsが確定したフレームのうち、直近のpoc
    int64_t m_nFirstKeyframePts; //最初のキーフレームのpts
};

//動画フレームのデータ
typedef struct VideoFrameData {
    FramePos *frame;      //各フレームの情報への配列 (デコードが開始されるフレームから取得する)
    int fixed_num;        //frame配列でフレーム順序が確定したフレーム数
    int num;              //frame配列で現在格納されているデータ数
    int capacity;         //frame配列を確保した数
    int64_t duration;     //合計の動画の長さ
} VideoFrameData;

typedef struct AVDemuxFormat {
    AVFormatContext          *pFormatCtx;            //動画ファイルのformatContext
    int                       nAnalyzeSec;           //動画ファイルを先頭から分析する時間
    bool                      bIsPipe;               //入力がパイプ
    uint32_t                  nPreReadBufferIdx;     //先読みバッファの読み込み履歴
    int                       nAudioTracks;          //存在する音声のトラック数
    int                       nSubtitleTracks;       //存在する字幕のトラック数
    QSVAVSync                 nAVSyncMode;           //音声・映像同期モード
} AVDemuxFormat;

typedef struct AVDemuxVideo {
                                                     //動画は音声のみ抽出する場合でも同期のため参照することがあり、
                                                     //pCodecCtxのチェックだけでは読み込むかどうか判定できないので、
                                                     //実際に使用するかどうかはこのフラグをチェックする
    bool                      bReadVideo;
    AVCodecContext           *pCodecCtx;             //動画のcodecContext, 動画を読み込むかどうかの判定には使用しないこと (bReadVideoを使用)
    int                       nIndex;                //動画のストリームID
    int64_t                   nStreamFirstPts;       //動画ファイルの最初のpts
    uint32_t                  nStreamPtsInvalid;     //動画ファイルのptsが無効 (H.264/ES, 等)
    int                       nRFFEstimate;          //動画がRFFの可能性がある
    bool                      bGotFirstKeyframe;     //動画の最初のキーフレームを取得済み
    AVBitStreamFilterContext *pH264Bsfc;             //必要なら使用するbitstreamfilter
    uint8_t                  *pExtradata;            //動画のヘッダ情報
    int                       nExtradataSize;        //動画のヘッダサイズ
    AVPacket                  packet[2];             //取得した動画ストリームの1フレーム分のデータ
    AVRational                nAvgFramerate;         //動画のフレームレート
    bool                      bUseHEVCmp42AnnexB;    //HEVCのmp4->AnnexB変換

    uint32_t                  nSampleLoadCount;      //sampleをLoadNextFrameでロードした数
    uint32_t                  nSampleGetCount;       //sampleをGetNextBitstreamで取得した数

    AVCodecParserContext     *pParserCtx;            //動画ストリームのParser
} AVDemuxVideo;

typedef struct AVDemuxStream {
    int                       nIndex;                 //音声・字幕のストリームID (libavのストリームID)
    int                       nTrackId;               //音声のトラックID (QSVEncC独自, 1,2,3,...)、字幕は0
    int                       nSubStreamId;           //通常は0、音声のチャンネルを分離する際に複製として作成
    AVCodecContext           *pCodecCtx;              //音声・字幕のcodecContext
    AVStream                 *pStream;                //音声・字幕のストリーム
    int                       nLastVidIndex;          //音声の直前の相当する動画の位置
    int64_t                   nExtractErrExcess;      //音声抽出のあまり (音声が多くなっていれば正、足りなくなっていれば負)
    AVPacket                  pktSample;              //サンプル用の音声・字幕データ
    int                       nDelayOfStream;         //音声側の遅延 (pkt_timebase基準)
    uint64_t                  pnStreamChannelSelect[MAX_SPLIT_CHANNELS]; //入力音声の使用するチャンネル
    uint64_t                  pnStreamChannelOut[MAX_SPLIT_CHANNELS];    //出力音声のチャンネル
} AVDemuxStream;

typedef struct AVDemuxer {
    AVDemuxFormat            format;
    AVDemuxVideo             video;
    FramePosList             frames;
    vector<AVDemuxStream>    stream;
    vector<const AVChapter*> chapter;
} AVDemuxer;

typedef struct AvcodecReaderPrm {
    uint8_t        memType;                 //使用するメモリの種類
    bool           bReadVideo;              //映像の読み込みを行うかどうか
    uint32_t       nReadAudio;              //音声の読み込みを行うかどうか (AVQSV_AUDIO_xxx)
    bool           bReadSubtitle;           //字幕の読み込みを行うかどうか
    bool           bReadChapter;            //チャプターの読み込みを行うかどうか
    pair<int,int>  nVideoAvgFramerate;      //動画のフレームレート
    uint16_t       nAnalyzeSec;             //入力ファイルを分析する秒数
    uint16_t       nTrimCount;              //Trimする動画フレームの領域の数
    sTrim         *pTrimList;               //Trimする動画フレームの領域のリスト
    int            nAudioTrackStart;        //音声のトラック番号の開始点
    int            nSubtitleTrackStart;     //字幕のトラック番号の開始点
    int            nAudioSelectCount;       //muxする音声のトラック数
    sAudioSelect **ppAudioSelect;           //muxする音声のトラック番号のリスト 1,2,...(1から連番で指定)
    int            nSubtitleSelectCount;    //muxする字幕のトラック数
    const int     *pSubtitleSelect;         //muxする字幕のトラック番号のリスト 1,2,...(1から連番で指定)
    int            nProcSpeedLimit;         //プリデコードする場合の処理速度制限 (0で制限なし)
    QSVAVSync      nAVSyncMode;             //音声・映像同期モード
    float          fSeekSec;                //指定された秒数分先頭を飛ばす
    const TCHAR   *pFramePosListLog;        //FramePosListの内容を入力終了時に出力する (デバッグ用)
} AvcodecReaderPrm;


class CAvcodecReader : public CQSVInput
{
public:
    CAvcodecReader();
    virtual ~CAvcodecReader();

    virtual mfxStatus Init(const TCHAR *strFileName, uint32_t ColorFormat, const void *option, CEncodingThread *pEncThread, shared_ptr<CEncodeStatusInfo> pEncSatusInfo, sInputCrop *pInputCrop) override;

    virtual void Close();

    //動画ストリームの1フレーム分のデータをm_sPacketに格納する
    //m_sPacketからの取得はGetNextBitstreamで行う
    virtual mfxStatus LoadNextFrame(mfxFrameSurface1 *pSurface) override;

    //動画ストリームの1フレーム分のデータをbitstreamに追加する
    virtual mfxStatus GetNextBitstream(mfxBitstream *bitstream) override;

    //ストリームのヘッダ部分を取得する
    virtual mfxStatus GetHeader(mfxBitstream *bitstream) override;

    //入力ファイルのグローバルメタデータを取得する
    const AVDictionary *GetInputFormatMetadata();

    //動画の入力情報を取得する
    const AVCodecContext *GetInputVideoCodecCtx();
    
    //音声・字幕パケットの配列を取得する
    vector<AVPacket> GetStreamDataPackets();

    //音声・字幕のコーデックコンテキストを取得する
    vector<AVDemuxStream> GetInputStreamInfo();

    //チャプターリストを取得する
    vector<const AVChapter *> GetChapterList();

    //入力ファイルに存在する音声のトラック数を返す
    int GetAudioTrackCount() override;

    //入力ファイルに存在する字幕のトラック数を返す
    int GetSubtitleTrackCount() override;

    //デコードするストリームの情報を取得する
    void GetDecParam(mfxVideoParam *decParam) {
        memcpy(decParam, &m_sDecParam, sizeof(m_sDecParam));
    }

    //動画の最初のフレームのptsを取得する
    int64_t GetVideoFirstPts();
private:
    //avcodecのコーデックIDからIntel Media SDKのコーデックのFourccを取得
    uint32_t getQSVFourcc(uint32_t id);

    //avcodecのストリームIDを取得 (typeはAVMEDIA_TYPE_xxxxx)
    vector<int> getStreamIndex(AVMediaType type);

    //VC-1のスタートコードの確認
    bool vc1StartCodeExists(uint8_t *ptr);

    //対象ストリームのパケットを取得
    int getSample(AVPacket *pkt);

    //対象・字幕の音声パケットを追加するかどうか
    bool checkStreamPacketToAdd(const AVPacket *pkt, AVDemuxStream *pStream);

    //対象のパケットの必要な対象のストリーム情報へのポインタ
    AVDemuxStream *getPacketStreamData(const AVPacket *pkt);

    //bitstreamにpktの内容を追加する
    mfxStatus setToMfxBitstream(mfxBitstream *bitstream, AVPacket *pkt);

    //音声パケットリストを開放
    void clearStreamPacketList(std::vector<AVPacket>& pktList);

    //音声パケットの配列を取得する (映像を読み込んでいないときに使用)
    vector<AVPacket> GetAudioDataPacketsWhenNoVideoRead();

    //QSVでデコードした際の最初のフレームのptsを取得する
    //さらに、平均フレームレートを推定する
    //fpsDecoderはdecoderの推定したfps
    mfxStatus getFirstFramePosAndFrameRate(AVRational fpsDecoder, mfxSession session, mfxBitstream *bitstream, const sTrim *pTrimList, int nTrimCount, int nProcSpeedLimit);

    //指定したptsとtimebaseから、該当する動画フレームを取得する
    int getVideoFrameIdx(int64_t pts, AVRational timebase, int iStart);

    //ptsを動画のtimebaseから音声のtimebaseに変換する
    int64_t convertTimebaseVidToStream(int64_t pts, const AVDemuxStream *pStream);

    //HEVCのmp4->AnnexB簡易変換
    void hevcMp42Annexb(AVPacket *pkt);

    //VC-1のヘッダの修正を行う
    void vc1FixHeader(int nLengthFix = -1);

    //VC-1のフレームヘッダを追加
    void vc1AddFrameHeader(AVPacket *pkt);

    void CloseStream(AVDemuxStream *pAudio);
    void CloseVideo(AVDemuxVideo *pVideo);
    void CloseFormat(AVDemuxFormat *pFormat);

    AVDemuxer        m_Demux;                      //デコード用情報
    tstring          m_sFramePosListLog;           //FramePosListの内容を入力終了時に出力する (デバッグ用)
    vector<uint8_t>  m_hevcMp42AnnexbBuffer;       //HEVCのmp4->AnnexB簡易変換用バッファ
    vector<AVPacket> m_PreReadBuffer;              //解析用に先行取得した映像パケット
    vector<AVPacket> m_StreamPacketsBufferL1[2];    //音声のAVPacketのバッファ (マルチスレッドで追加されてくることに注意する)
    vector<AVPacket> m_StreamPacketsBufferL2;       //音声のAVPacketのバッファ
    uint32_t         m_StreamPacketsBufferL2Used;   //m_StreamPacketsBufferL2のパケットのうち、すでに使用したもの
};

#endif //ENABLE_AVCODEC_QSV_READER

#endif //_AVCODEC_READER_H_
