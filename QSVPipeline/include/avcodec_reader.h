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

typedef struct AVDemuxer {
	AVFormatContext          *pFormatCtx;      //動画ファイルのformatContext
	AVCodecContext           *pCodecCtx;       //動画のcodecContext
	int                       videoIndex;      //動画のストリームID
	mfxU32                    sampleLoadCount; //sampleをLoadNextFrameでロードした数
	mfxU32                    sampleGetCount;  //sampleをGetNextBitstreamで取得した数
	AVBitStreamFilterContext *bsfc;            //必要なら使用するbitstreamfilter
	mfxU8                    *extradata;       //ヘッダ情報
	int                       extradataSize;   //ヘッダサイズ
} AVDemuxer;

typedef struct QSVCodec {
	mfxU32 codec_id;   //avcodecのコーデックID
	mfxU32 qsv_fourcc; //QSVのfourcc
} QSVCodec;

//QSVでデコード可能なコーデックのリスト
static const QSVCodec QSV_LIST[] = { 
	{ AV_CODEC_ID_MPEG2VIDEO, MFX_CODEC_MPEG2 },
	{ AV_CODEC_ID_H264,       MFX_CODEC_AVC   },
	{ AV_CODEC_ID_HEVC,       MFX_CODEC_HEVC  },
	//{ AV_CODEC_ID_VC1,        MFX_CODEC_VC1   },
};

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
private:


	//avcodecのコーデックIDからIntel Media SDKのコーデックのFourccを取得
	mfxU32 getQSVFourcc(mfxU32 id);

	//avcodecのvideoのストリームIDを取得
	int getVideoStream();

	//対象ストリームのパケットを取得
	int getSample(AVPacket *pkt);

	//bitstreamにpktの内容を追加する
	mfxStatus setToMfxBitstream(mfxBitstream *bitstream, AVPacket *pkt);

	//gcdを取得
	int getGcd(int a, int b) {
		if (a == 0 || b == 0)
			return 1;
		int c;
		while ((c = a % b) != 0)
			a = b, b = c;
		return b;
	}

	AVPacket m_sPacket[2]; //取得した動画ストリームの1フレーム分のデータ
	AVDemuxer demux; //デコード用情報
};

#endif //ENABLE_AVCODEC_QSV_READER

#endif //_AVCODEC_READER_H_
