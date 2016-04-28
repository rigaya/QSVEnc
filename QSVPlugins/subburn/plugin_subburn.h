//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#pragma once
#ifndef __PLUGIN_SUB_BURN_H__
#define __PLUGIN_SUB_BURN_H__

#include <stdlib.h>
#include <vector>
#include <mfxplugin++.h>

#include "qsv_version.h"

#include "qsv_prm.h"
#include "qsv_queue.h"
#include "../base/plugin_base.h"

#if ENABLE_AVCODEC_QSV_READER && ENABLE_LIBASS_SUBBURN
#include "avcodec_reader.h"
#include "ass/ass.h"

struct ProcessDataSubBurn {
    MemType               memType;                //使用するメモリの種類
    const TCHAR          *pFilePath;              //入力字幕ファイル (nullptrの場合は入力映像ファイルのトラックから読み込む)
    std::string           sCharEnc;               //字幕の文字コード
    ASS_ShapingLevel      nAssShaping;            //assのレンダリング品質
    const AVCodecContext *pVideoInputCodecCtx;    //入力映像のコーデック情報
    int64_t               nVideoInputFirstKeyPts; //入力映像の最初のpts
    sInputCrop            sCrop;                  //crop

    int                   nInTrackId;             //ソースファイルの入力トラック番号
    mfxFrameInfo          frameInfo;              //フレーム情報
    AVCodecContext       *pCodecCtxIn;            //入力字幕のCodecContextのコピー
    int                   nStreamIndexIn;         //入力字幕のStreamのindex

    //変換用
    AVFormatContext      *pFormatCtx;             //ファイル読み込みの際に使用する(トラックを受け取る場合はnullptr)
    int                   nSubtitleStreamIndex;   //ファイル読み込みの際に使用する(トラックを受け取る場合は-1)
    AVCodec              *pOutCodecDecode;        //変換する元のコーデック
    AVCodecContext       *pOutCodecDecodeCtx;     //変換する元のCodecContext
    AVCodec              *pOutCodecEncode;        //変換先の音声のコーデック
    AVCodecContext       *pOutCodecEncodeCtx;     //変換先の音声のCodecContext

    uint8_t              *pBuf;                   //変換用のバッファ

    int                   nType;                  //字幕の種類
    ASS_Library          *pAssLibrary;            //libassのコンテキスト
    ASS_Renderer         *pAssRenderer;           //libassのレンダラ
    ASS_Track            *pAssTrack;              //libassのトラック
    
    CQueueSPSP<AVPacket>  qSubPackets;            //入力から得られた字幕パケット

    uint32_t              nSimdAvail;

    ProcessDataSubBurn() :
        memType(SYSTEM_MEMORY),
        pFilePath(nullptr),
        sCharEnc(),
        nAssShaping(ASS_SHAPING_SIMPLE),
        pVideoInputCodecCtx(nullptr),
        nVideoInputFirstKeyPts(0),
        sCrop({ 0 }),
        nInTrackId(0),
        frameInfo({ 0 }),
        pCodecCtxIn(nullptr),
        nStreamIndexIn(-1),
        pFormatCtx(nullptr),
        nSubtitleStreamIndex(-1),
        pOutCodecDecode(nullptr),
        pOutCodecDecodeCtx(nullptr),
        pOutCodecEncode(nullptr),
        pOutCodecEncodeCtx(nullptr),
        pBuf(nullptr),
        nType(0),
        pAssLibrary(nullptr),
        pAssRenderer(nullptr),
        pAssTrack(nullptr),
        qSubPackets(),
        nSimdAvail(0) {
        qSubPackets.init();
    }
};

class ProcessorSubBurn : public Processor
{
public:
    virtual mfxStatus Init(mfxFrameSurface1 *frame_in, mfxFrameSurface1 *frame_out, const void *data) override;
    virtual mfxStatus Process(DataChunk *chunk, uint8_t *pBuffer) override;

protected:
#if ENABLE_AVCODEC_QSV_READER && ENABLE_LIBASS_SUBBURN
    virtual void CopyFrameY();
    virtual void CopyFrameUV();
    virtual void BlendSubY(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcolory, uint8_t subTransparency, uint8_t *pBuf);
    virtual void BlendSubUV(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcoloru, uint8_t subcolorv, uint8_t subTransparency, uint8_t *pBuf);
    template<bool forUV> mfxStatus SubBurn(ASS_Image *pImage, uint8_t *pBuffer);
#endif
    ProcessDataSubBurn *m_pProcData;
};

struct SubBurnParam {
    mfxFrameAllocator    *pAllocator;             //メインパイプラインのアロケータ
    MemType               memType;                //アロケータのメモリタイプ
    const TCHAR          *pFilePath;              //入力字幕ファイル (nullptrの場合は入力映像ファイルのトラックから読み込む)
    const TCHAR          *pCharEnc;               //字幕の文字コード
    int                   nShaping;               //レンダリング品質 (QSV_VPP_SUB_xxx)
    mfxFrameInfo          frameInfo;              //フレーム情報
    const AVCodecContext *pVideoInputCodecCtx;    //入力映像のコーデック情報
    int64_t               nVideoInputFirstKeyPts; //入力映像の最初のpts
    AVDemuxStream         src;                    //焼きこむ字幕の情報
    sInputCrop            sCrop;                  //crop

    SubBurnParam() : pAllocator(nullptr), memType(SYSTEM_MEMORY), pFilePath(nullptr), pCharEnc(nullptr), nShaping(QSV_VPP_SUB_SIMPLE), frameInfo({ 0 }), pVideoInputCodecCtx(nullptr), nVideoInputFirstKeyPts(0), src(), sCrop({ 0 }) {
        memset(&src, 0, sizeof(src));
    }

    SubBurnParam(mfxFrameAllocator *allocator, MemType memtype, const TCHAR *pSubFilePath, const TCHAR *pSubCharEnc, int nSubShaping, mfxFrameInfo inputFrameInfo, const AVCodecContext *pSrcVideoInputCodecCtx, int64_t nSrcVideoInputFirstKeyPts, AVDemuxStream srcStream, const sInputCrop *pSrcCrop) :
        pAllocator(allocator), memType(memtype),
        pFilePath(pSubFilePath),
        pCharEnc(pSubCharEnc),
        nShaping(nSubShaping),
        frameInfo(inputFrameInfo),
        pVideoInputCodecCtx(pSrcVideoInputCodecCtx),
        nVideoInputFirstKeyPts(nSrcVideoInputFirstKeyPts),
        src(srcStream),
        sCrop({ 0 }) {
        if (pSrcCrop) {
            sCrop = *pSrcCrop;
        }
    }
};

class SubBurn : public QSVEncPlugin
{
public:
    SubBurn();
    virtual ~SubBurn();

    // methods to be called by Media SDK
    virtual mfxStatus Init(mfxVideoParam *mfxParam) override;
    virtual mfxStatus SetAuxParams(void *auxParam, int auxParamSize) override;
    virtual mfxStatus Submit(const mfxHDL *in, mfxU32 in_num, const mfxHDL *out, mfxU32 out_num, mfxThreadTask *task) override;
    // methods to be called by application
    static MFXGenericPlugin *CreateGenericPlugin() {
        return new SubBurn();
    }

    virtual mfxStatus SendData(int nType, void *pData) override;

    virtual mfxStatus Close();

    virtual int getTargetTrack() override {
        return m_vProcessData[0].nInTrackId;
    }

protected:
    mfxStatus InitLibAss(ProcessDataSubBurn *pProcData);
    mfxStatus InitAvcodec(ProcessDataSubBurn *pProcData);
    tstring errorMesForCodec(const TCHAR *mes, AVCodecID targetCodec);
    void SetExtraData(AVCodecContext *codecCtx, const uint8_t *data, uint32_t size);
    mfxStatus ProcSub(ProcessDataSubBurn *pProcData);

    int m_nCpuGen;
    uint32_t m_nSimdAvail;
    SubBurnParam m_SubBurnParam;
    vector<ProcessDataSubBurn> m_vProcessData;
};
#endif //#if ENABLE_AVCODEC_QSV_READER && ENABLE_LIBASS_SUBBURN

#endif // __PLUGIN_SUB_BURN_H__
