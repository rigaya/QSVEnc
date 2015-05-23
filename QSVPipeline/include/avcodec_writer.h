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

#define USE_CUSTOM_IO 1

typedef struct AVMuxer {
	AVFormatContext      *pFormatCtx;           //出力ファイルのformatContext
	char                  metadataStr[256];     //出力ファイルのエンコーダ名
	AVOutputFormat       *pOutputFmt;           //出力ファイルのoutputFormat

	AVCodec              *pVideoCodec;          //出力映像のCodec
	AVCodecContext       *pVideoCodecCtx;       //出力映像のCodecContext
	AVRational            nVideoFPS;            //出力映像のフレームレート
	AVStream             *pVideoStream;         //出力ファイルの映像ストリーム
	bool                  bVideoDtsUnavailable; //出力映像のdtsが無効 (API v1.6以下)
	int                   nVideoFpsBaseNextDts; //出力映像のfpsベースでのdts (API v1.6以下でdtsが計算されない場合に使用する)

	AVCodecContext       *pAudioCodecCtxIn;     //入力音声のCodecContextのコピー
	AVStream             *pAudioStream;         //出力ファイルの音声ストリーム
	int                   nAudioPacketWritten;  //出力したパケットの数
#if USE_CUSTOM_IO
	mfxU8                *pAVOutBuffer;         //avio_alloc_context用のバッファ
	mfxU32                nAVOutBufferSize;     //avio_alloc_context用のバッファサイズ
	FILE                 *fpOutput;             //出力ファイルポインタ
	char                 *pOutputBuffer;        //出力ファイルポインタ用のバッファ
	mfxU32                nOutputBufferSize;    //出力ファイルポインタ用のバッファサイズ
#endif //USE_CUSTOM_IO

	//PCMの変換用
	AVCodec              *pAudioOutCodecDecode;     //変換するPCMの元のコーデック
	AVCodecContext       *pAudioOutCodecDecodeCtx;  //変換するPCMの元のCodecContext
	AVCodec              *pAudioOutCodecEncode;     //変換先のPCMの音声のコーデック
	AVCodecContext       *pAudioOutCodecEncodeCtx;  //変換先のPCMの音声のCodecContext
	AVPacket              audioOutPacket;           //変換用の音声バッファ
	//AACの変換用
	AVBitStreamFilterContext *pAudioAACBsfc;        //必要なら使用するbitstreamfilter

	int                   nAudioOutputSamples; //出力音声の出力済みsample数
	mfxI64                nAudioLastPts;       //出力音声の前パケットのpts

	bool                  bStreamError;     //エラーが発生
} AVMuxer;

typedef struct AvcodecWriterPrm {
	const mfxInfoMFX            *pVideoInfo;           //出力映像の情報
	bool                         bVideoDtsUnavailable; //出力映像のdtsが無効 (API v1.6以下)
	const mfxExtVideoSignalInfo *pVideoSignalInfo;     //出力映像の情報
	const AVCodecContext        *pCodecCtxAudioIn;     //入力ファイルの音声のcodecContext
} AvcodecWriterPrm;

class CAvcodecWriter : public CSmplBitstreamWriter
{
public:
	CAvcodecWriter();
	virtual ~CAvcodecWriter();

	virtual mfxStatus Init(const msdk_char *strFileName, const void *option, CEncodeStatusInfo *pEncSatusInfo) override;

	virtual mfxStatus WriteNextFrame(mfxBitstream *pMfxBitstream) override;

	virtual mfxStatus WriteNextFrame(AVPacket *pkt);

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
	void applyBitstreamFilterAAC(AVPacket *pkt);

	AVMuxer m_Muxer;
};

#endif //ENABLE_AVCODEC_QSV_READER

#endif //_AVCODEC_WRITER_H_
