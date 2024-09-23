// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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

#pragma once

#include "rgy_filter_cl.h"
#include "rgy_filter_resize.h"
#include "rgy_prm.h"
#include "rgy_input.h"

class RGYFilterParamOverlay : public RGYFilterParam {
public:
    VppOverlay overlay;
    RGYParamThread threadPrm;

    RGYFilterParamOverlay() : overlay(), threadPrm() {

    };
    virtual ~RGYFilterParamOverlay() {};
    virtual tstring print() const override;
};

class RGYFilterOverlay : public RGYFilter {
    struct RGYFilterOverlayFrame {
        std::unique_ptr<RGYFilterCspCrop> crop;
        std::unique_ptr<RGYFilterResize> resize;
        std::unique_ptr<RGYCLFrame> dev;
        RGYFrameInfo *inputPtr;

        RGYFilterOverlayFrame() : crop(), resize(), dev(), inputPtr(nullptr) {};
        void close();
    };
public:
    RGYFilterOverlay(std::shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterOverlay();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR initInput(RGYFilterParamOverlay *prm);
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue& queue_main, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    std::tuple<RGY_ERR, std::unique_ptr<AVPacket, RGYAVDeleter<AVPacket>>> getFramePkt();
    RGY_ERR getFrame(RGYOpenCLQueue& queue);
    RGY_ERR prepareFrameDev(RGYFilterOverlayFrame& target, RGYOpenCLQueue& queue);
    RGY_ERR overlayPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const RGYFrameInfo *pOverlay, const RGYFrameInfo *pAlpha, const int posX, const int posY,
        RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event);
    RGY_ERR overlayFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event);

    std::unique_ptr<AVFormatContext, decltype(&avformat_free_context)> m_formatCtx;
    std::unique_ptr<AVCodecContext, RGYAVDeleter<AVCodecContext>> m_codecCtxDec;
    int m_inputFrames;
    std::unique_ptr<RGYConvertCSP> m_convert;
    RGY_CSP m_inputCsp;
    AVStream *m_stream;
    RGYFilterOverlayFrame m_frame;
    RGYFilterOverlayFrame m_alpha;
    RGYOpenCLProgramAsync m_overlay;

    bool m_bInterlacedWarn;
};
