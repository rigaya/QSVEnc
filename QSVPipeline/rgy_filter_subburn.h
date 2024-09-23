// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc/rkmppenc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2020-2021 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#include "rgy_filter_cl.h"
#include "rgy_prm.h"
#include "rgy_frame.h"
#include "rgy_filter_resize.h"
#include "rgy_input_avcodec.h"
#include <array>

#if ENABLE_AVSW_READER

class RGYFilterParamSubburn : public RGYFilterParam {
public:
    VppSubburn      subburn;
    AVRational      videoOutTimebase;
    const AVStream *videoInputStream;
    int64_t         videoInputFirstKeyPts;
    VideoInfo       videoInfo;
    AVDemuxStream   streamIn;
    sInputCrop      crop;
    RGYPoolAVPacket *poolPkt;
    std::vector<const AVStream *> attachmentStreams;

    RGYFilterParamSubburn() : subburn(), videoOutTimebase(), videoInputStream(nullptr), videoInputFirstKeyPts(0), videoInfo(), streamIn(), crop(), poolPkt(nullptr), attachmentStreams() {};
    virtual ~RGYFilterParamSubburn() {};
    virtual tstring print() const override {
        return subburn.print();
    }
};


#if ENABLE_LIBASS_SUBBURN

#include "ass/ass.h"

struct subtitle_deleter {
    void operator()(AVSubtitle *subtitle) const {
        avsubtitle_free(subtitle);
        delete subtitle;
    }
};

struct SubImageData {
    unique_ptr<RGYCLFrame> image;
    unique_ptr<RGYCLFrame> imageTemp;
    int x, y;

    SubImageData(unique_ptr<RGYCLFrame> img, unique_ptr<RGYCLFrame> imgTemp, int posX, int posY) :
        image(std::move(img)), imageTemp(std::move(imgTemp)), x(posX), y(posY) { }
};

class RGYFilterSubburn : public RGYFilter {
public:
    RGYFilterSubburn(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterSubburn();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
    virtual RGY_ERR addStreamPacket(AVPacket *pkt) override;
    virtual int targetTrackIdx() override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamSubburn> prm);
    virtual RGY_ERR initAVCodec(const std::shared_ptr<RGYFilterParamSubburn> prm);
    virtual RGY_ERR InitLibAss(const std::shared_ptr<RGYFilterParamSubburn> prm);
    void SetExtraData(AVCodecContext *codecCtx, const uint8_t *data, uint32_t size);
    RGY_ERR readSubFile();
    SubImageData textRectToImage(const ASS_Image *image, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    SubImageData bitmapRectToImage(const AVSubtitleRect *rect, const RGYFrameInfo *outputFrame, const sInputCrop &crop, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR procFrameText(RGYFrameInfo *pOutputFrame, int64_t frameTimeMs, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR procFrameBitmap(RGYFrameInfo *pOutputFrame, const int64_t frameTimeMs, const sInputCrop &crop, const bool forced_subs_only, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR procFrame(RGYFrameInfo *pFrame,
        const RGYFrameInfo *pSubImg,
        int pos_x, int pos_y,
        float transparency_offset, float brightness, float contrast,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *even);
    RGY_ERR procFrame(RGYFrameInfo *pOutputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    int m_subType; //字幕の種類
    unique_ptr<AVFormatContext, RGYAVDeleter<AVFormatContext>> m_formatCtx;     //ファイル読み込みの際に使用する(トラックを受け取る場合はnullptr)
    int m_subtitleStreamIndex; //ファイル読み込みの際に使用する(トラックを受け取る場合は-1)
    const AVCodec *m_outCodecDecode; //変換する元のコーデック
    unique_ptr<AVCodecContext, RGYAVDeleter<AVCodecContext>> m_outCodecDecodeCtx;     //変換する元のCodecContext

    unique_ptr<AVSubtitle, subtitle_deleter> m_subData;
    vector<SubImageData> m_subImages;

    unique_ptr<ASS_Library, decltype(&ass_library_done)> m_assLibrary; //libassのコンテキスト
    unique_ptr<ASS_Renderer, decltype(&ass_renderer_done)> m_assRenderer; //libassのレンダラ
    unique_ptr<ASS_Track, decltype(&ass_free_track)> m_assTrack; //libassのトラック

    unique_ptr<RGYFilterResize> m_resize;

    RGYPoolAVPacket *m_poolPkt;
    RGYQueueMPMP<AVPacket*> m_queueSubPackets; //入力から得られた字幕パケット

    RGYOpenCLProgramAsync m_subburn;
};

#else //ENABLE_LIBASS_SUBBURN

class RGYFilterSubburn : public RGYFilter {
public:
    RGYFilterSubburn(shared_ptr<RGYOpenCLContext> context) {};
    virtual ~RGYFilterSubburn() {};
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override {
        AddMessage(RGY_LOG_ERROR, _T("subburn not supported in this build.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override {
        AddMessage(RGY_LOG_ERROR, _T("subburn not supported in this build.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    virtual void close() override {};
};

#endif //#if ENABLE_LIBASS_SUBBURN

#endif //#if ENABLE_AVSW_READER
