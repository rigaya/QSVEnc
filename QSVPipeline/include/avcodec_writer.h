//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------
#ifndef _AVCODEC_WRITER_H_
#define _AVCODEC_WRITER_H_

#include "sample_utils.h"

#if ENABLE_AVCODEC_QSV_READER
#include "avcodec_qsv.h"
#include "avcodec_reader.h"

using std::vector;

#define USE_CUSTOM_IO 1

typedef struct AVMuxFormat {
	AVFormatContext      *pFormatCtx;           //出力ファイルのformatContext
	char                  metadataStr[256];     //出力ファイルのエンコーダ名
	AVOutputFormat       *pOutputFmt;           //出力ファイルのoutputFormat

#if USE_CUSTOM_IO
	mfxU8                *pAVOutBuffer;         //avio_alloc_context用のバッファ
	mfxU32                nAVOutBufferSize;     //avio_alloc_context用のバッファサイズ
	FILE                 *fpOutput;             //出力ファイルポインタ
	char                 *pOutputBuffer;        //出力ファイルポインタ用のバッファ
	mfxU32                nOutputBufferSize;    //出力ファイルポインタ用のバッファサイズ
#endif //USE_CUSTOM_IO
	bool                  bStreamError;         //エラーが発生
	bool                  bIsMatroska;          //mkvかどうか
} AVMuxFormat;

typedef struct AVMuxVideo {
	AVCodec              *pCodec;               //出力映像のCodec
	AVCodecContext       *pCodecCtx;            //出力映像のCodecContext
	AVRational            nFPS;                 //出力映像のフレームレート
	AVStream             *pStream;              //出力ファイルの映像ストリーム
	bool                  bDtsUnavailable;      //出力映像のdtsが無効 (API v1.6以下)
	int                   nFpsBaseNextDts;      //出力映像のfpsベースでのdts (API v1.6以下でdtsが計算されない場合に使用する)
	bool                  bIsPAFF;              //出力映像がPAFFである
} AVMuxVideo;

typedef struct AVMuxAudio {
	AVCodecContext       *pCodecCtxIn;          //入力音声のCodecContextのコピー
	int                   nStreamIndexIn;       //入力音声のStreamのindex
	int                   nDelaySamplesOfAudio; //入力音声の遅延 (pkt_timebase基準)
	AVStream             *pStream;              //出力ファイルの音声ストリーム
	int                   nPacketWritten;       //出力したパケットの数

	//PCMの変換用
	AVCodec              *pOutCodecDecode;      //変換するPCMの元のコーデック
	AVCodecContext       *pOutCodecDecodeCtx;   //変換するPCMの元のCodecContext
	AVCodec              *pOutCodecEncode;      //変換先のPCMの音声のコーデック
	AVCodecContext       *pOutCodecEncodeCtx;   //変換先のPCMの音声のCodecContext
	AVPacket              OutPacket;            //変換用の音声バッファ
	//AACの変換用
	AVBitStreamFilterContext *pAACBsfc;         //必要なら使用するbitstreamfilter

	int                   nOutputSamples;       //出力音声の出力済みsample数
	mfxI64                nLastPts;             //出力音声の前パケットのpts
} AVMuxAudio;

typedef struct AVMux {
	AVMuxFormat         format;
	AVMuxVideo          video;
	vector<AVMuxAudio>  audio;
} AVMux;

typedef struct AvcodecWriterPrm {
	const mfxInfoMFX            *pVideoInfo;              //出力映像の情報
	bool                         bVideoDtsUnavailable;    //出力映像のdtsが無効 (API v1.6以下)
	const mfxExtVideoSignalInfo *pVideoSignalInfo;        //出力映像の情報
	vector<AVDemuxAudio>         inputAudioList;          //入力ファイルの音声の情報
} AvcodecWriterPrm;

class CAvcodecWriter : public CSmplBitstreamWriter
{
public:
	CAvcodecWriter();
	virtual ~CAvcodecWriter();

	virtual mfxStatus Init(const msdk_char *strFileName, const void *option, CEncodeStatusInfo *pEncSatusInfo) override;

	virtual mfxStatus SetVideoParam(mfxVideoParam *pMfxVideoPrm) override;

	virtual mfxStatus WriteNextFrame(mfxBitstream *pMfxBitstream) override;

	virtual mfxStatus WriteNextFrame(AVPacket *pkt);

	virtual vector<int> GetAudioStreamIndex();

	virtual void Close();

#if USE_CUSTOM_IO
	int readPacket(uint8_t *buf, int buf_size);
	int writePacket(uint8_t *buf, int buf_size);
	int64_t seek(int64_t offset, int whence);
#endif //USE_CUSTOM_IO
private:
	//PCMのコーデックがwav出力時に変換を必要とするかを判定する
	AVCodecID PCMRequiresConversion(const AVCodecContext *audioCtx);

	//QSVのコーデックFourccからAVCodecのCodecIDを返す
	AVCodecID getAVCodecId(mfxU32 QSVFourcc);

	//"<mes> for codec"型のエラーメッセージを作成する  
	tstring errorMesForCodec(const TCHAR *mes, AVCodecID targetCodec);

	//AAC音声にBitstreamフィルターを適用する
	void applyBitstreamFilterAAC(AVPacket *pkt, AVMuxAudio *pMuxAudio);

	//H.264ストリームからPAFFのフィールドの長さを返す
	mfxU32 getH264PAFFFieldLength(mfxU8 *ptr, mfxU32 size);

	//extradataをコピーする
	void SetExtraData(AVCodecContext *codecCtx, const mfxU8 *data, mfxU32 size);
	
	//映像の初期化
	mfxStatus InitVideo(const AvcodecWriterPrm *prm);

	//音声の初期化
	mfxStatus InitAudio(AVMuxAudio *pMuxAudio, AVDemuxAudio *pInputAudio);

	//メッセージを作成
	tstring GetWriterMes();

	//対象のパケットの必要な対象のストリーム情報へのポインタ
	AVMuxAudio *getAudioPacketStreamData(const AVPacket *pkt);

	void CloseAudio(AVMuxAudio *pMuxAudio);
	void CloseVideo(AVMuxVideo *pMuxVideo);
	void CloseFormat(AVMuxFormat *pMuxFormat);

	AVMux m_Mux;
};

#endif //ENABLE_AVCODEC_QSV_READER

#endif //_AVCODEC_WRITER_H_
