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

#include "sample_utils.h"

#if ENABLE_AVCODEC_QSV_READER
#include "avcodec_qsv.h"

static const mfxU32 AVCODEC_READER_INPUT_BUF_SIZE = 16 * 1024 * 1024;

//フレームの位置情報と長さを格納する
typedef struct FramePos {
	int64_t pts;  //pts
	int duration; //該当フレームの表示時間
} FramePos;

//動画フレームのデータ
typedef struct VideoFrameData {
	FramePos *frame;      //各フレームの情報への配列 (デコードが開始されるフレームから取得する)
	int fixed_num;        //frame配列でフレーム順序が確定したフレーム数
	int num;              //frame配列で現在格納されているデータ数
	int capacity;         //frame配列を確保した数
	CRITICAL_SECTION cs;  //frame配列アクセスへの排他制御用
	bool cs_initialized;  //csが初期化されているかどうか
} VideoFrameData;

typedef struct AVDemuxer {
	AVFormatContext          *pFormatCtx;                 //動画ファイルのformatContext
	AVCodecContext           *pCodecCtx;                  //動画のcodecContext
	int                       videoIndex;                 //動画のストリームID
	int64_t                   videoStreamFirstPts;        //動画ファイルの最初のpts
	bool                      videoStreamPtsInvalid;      //動画ファイルのptsが無効 (H.264/ES等)
	VideoFrameData            videoFrameData;             //動画フレームのptsのリスト
	AVBitStreamFilterContext *bsfc;                       //必要なら使用するbitstreamfilter
	mfxU8                    *extradata;                  //動画のヘッダ情報
	int                       extradataSize;              //動画のヘッダサイズ
	AVPacket                  videoPacket[2];             //取得した動画ストリームの1フレーム分のデータ
	AVRational                videoAvgFramerate;          //動画のフレームレート

	int                       audioIndex;                 //音声のストリームID
	AVCodecContext           *pCodecCtxAudio;             //音声のcodecContext
	int                       lastVidIndex;               //音声の直前の相当する動画の位置
	mfxI64                    audExtractErrExcess;        //音声抽出のあまり (音声が多くなっていれば正、足りなくなっていれば負)

	mfxU32                    sampleLoadCount;            //sampleをLoadNextFrameでロードした数
	mfxU32                    sampleGetCount;             //sampleをGetNextBitstreamで取得した数
} AVDemuxer;

typedef struct QSVCodec {
	mfxU32 codec_id;   //avcodecのコーデックID
	mfxU32 qsv_fourcc; //QSVのfourcc
} QSVCodec;

//QSVでデコード可能なコーデックのリスト
static const QSVCodec QSV_DECODE_LIST[] = { 
	{ AV_CODEC_ID_MPEG2VIDEO, MFX_CODEC_MPEG2 },
	{ AV_CODEC_ID_H264,       MFX_CODEC_AVC   },
	//{ AV_CODEC_ID_HEVC,       MFX_CODEC_HEVC  },
	//{ AV_CODEC_ID_VC1,        MFX_CODEC_VC1   },
};

typedef struct AvcodecReaderPrm {
	mfxU8      memType;
	bool       bReadAudio;
	mfxU16     nTrimCount;
	sTrim     *pTrimList;
} AvcodecReaderPrm;


class CAvcodecReader : public CSmplYUVReader
{
public:
	CAvcodecReader();
	virtual ~CAvcodecReader();

	virtual mfxStatus Init(const TCHAR *strFileName, mfxU32 ColorFormat, const void *option, CEncodingThread *pEncThread, CEncodeStatusInfo *pEncSatusInfo, sInputCrop *pInputCrop) override;

	virtual void Close();

	//動画ストリームの1フレーム分のデータをm_sPacketに格納する
	//m_sPacketからの取得はGetNextBitstreamで行う
	virtual mfxStatus LoadNextFrame(mfxFrameSurface1* pSurface) override;

	//動画ストリームの1フレーム分のデータをbitstreamに追加する
	virtual mfxStatus GetNextBitstream(mfxBitstream *bitstream) override;

	//ストリームのヘッダ部分を取得する
	virtual mfxStatus GetHeader(mfxBitstream *bitstream) override;
	
	//音声パケットの配列を取得する
	std::vector<AVPacket> GetAudioDataPackets();

	//音声のコーデックコンテキストを取得する
	const AVCodecContext *GetAudioCodecCtx();

	//デコードするストリームの情報を取得する
	void GetDecParam(mfxVideoParam *decParam) {
		memcpy(decParam, &m_sDecParam, sizeof(m_sDecParam));
	}
private:
	//avcodecのコーデックIDからIntel Media SDKのコーデックのFourccを取得
	mfxU32 getQSVFourcc(mfxU32 id);

	//avcodecのvideoのストリームIDを取得
	int getVideoStream();

	//avcodecのaudioのストリームIDを取得
	int getAudioStream();

	//動画のptsをソートする
	void sortVideoPtsList();

	//動画のptsをリストに加える
	void addVideoPtsToList(FramePos pos);

	//対象ストリームのパケットを取得
	int getSample(AVPacket *pkt);

	//対象の音声パケットを追加するかどうか
	bool checkAudioPacketToAdd(const AVPacket *pkt);

	//bitstreamにpktの内容を追加する
	mfxStatus setToMfxBitstream(mfxBitstream *bitstream, AVPacket *pkt);

	//音声パケットリストを開放
	void clearAudioPacketList(std::vector<AVPacket>& pktList);

	//QSVでデコードした際の最初のフレームのptsを取得する
	//さらに、平均フレームレートを推定する
	//fpsDecoderはdecoderの推定したfps
	mfxStatus getFirstFramePosAndFrameRate(AVRational fpsDecoder);

	//指定したptsとtimebaseから、該当する動画フレームを取得する
	int getVideoFrameIdx(mfxI64 pts, AVRational timebase, int i_start);

	//ptsを動画のtimebaseから音声のtimebaseに変換する
	mfxI64 convertTimebaseVidToAud(mfxI64 pts);

	//gcdを取得
	int getGcd(int a, int b) {
		if (a == 0 || b == 0)
			return 1;
		int c;
		while ((c = a % b) != 0)
			a = b, b = c;
		return b;
	}
	AVDemuxer demux; //デコード用情報
	std::vector<AVPacket> m_AudioPacketsBufferL1[2];    //音声のAVPacketのバッファ (マルチスレッドで追加されてくることに注意する)
	std::vector<AVPacket> m_AudioPacketsBufferL2;       //音声のAVPacketのバッファ
	mfxU32                m_AudioPacketsBufferL2Used;   //m_AudioPacketsBufferL2のパケットのうち、すでに使用したもの

	std::vector<sTrim> m_sTrimList;
};

#endif //ENABLE_AVCODEC_QSV_READER

#endif //_AVCODEC_READER_H_
