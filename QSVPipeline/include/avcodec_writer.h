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

typedef struct AVMuxer {
	AVFormatContext      *pFormatCtx;      //出力ファイルのformatContext
	AVOutputFormat       *pOutputFmt;      //出力ファイルのoutputFormat
	AVStream             *pStreamAudio;    //出力ファイルの音声ストリーム
	int                   nPacketWritten;  //出力したパケットの数
	mfxI64                nFirstPktPts;    //最初のパケットのpts
	mfxI64                nFirstPktDts;    //最初のパケットのdts
} AVMuxer;

typedef struct AvcodecWriterPrm {
	const AVCodecContext *pCodecCtxAudioIn;    //入力ファイルの音声のcodecContext
} AvcodecWriterPrm;

class CAvcodecWriter : public CSmplBitstreamWriter
{
public:
	CAvcodecWriter();
	virtual ~CAvcodecWriter();

	virtual mfxStatus Init(const msdk_char *strFileName, const void *option, CEncodeStatusInfo *pEncSatusInfo) override;

	virtual mfxStatus WriteNextFrame(AVPacket *pkt);

	virtual void Close();
private:
	AVMuxer m_Muxer;
};

#endif //ENABLE_AVCODEC_QSV_READER

#endif //_AVCODEC_WRITER_H_
