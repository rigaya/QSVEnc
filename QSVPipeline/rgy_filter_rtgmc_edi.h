// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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

#include <array>

#include "rgy_filter_cl.h"
#include "rgy_filter_nnedi.h"
#include "rgy_prm.h"

class RGYFilterParamRtgmcEdi : public RGYFilterParam {
public:
    // Bob input supplies chroma; EDI input can supply luma.
    VppRtgmcEdiMode mode;
    VppRtgmcChromaEdiMode chromaEdi;
    int nnsize;
    int nneurons;
    int ediqual;
    RGYFrameInfo sourceFrameIn;
    rgy_rational<int> sourceBaseFps;
    rgy_rational<int> sourceTimebase;
    HMODULE hModule;

    RGYFilterParamRtgmcEdi() : mode(VppRtgmcEdiMode::BobChromaMerge), chromaEdi(VppRtgmcChromaEdiMode::None), nnsize(1), nneurons(1), ediqual(1), sourceFrameIn(), sourceBaseFps(), sourceTimebase(), hModule(NULL) {}
    virtual ~RGYFilterParamRtgmcEdi() {}
    virtual tstring print() const override;
};

class RGYFilterRtgmcEdi : public RGYFilter {
public:
    RGYFilterRtgmcEdi(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterRtgmcEdi();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;

    RGY_ERR run_filter(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR run_filter(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame, const RGYFrameInfo *pSourceInputFrame,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;
public:
    virtual void resetTemporalState() override;
protected:

    class FrameSource {
    public:
        FrameSource();
        RGY_ERR alloc(std::shared_ptr<RGYOpenCLContext> cl, const RGYFrameInfo& frameInfo);
        RGY_ERR add(std::shared_ptr<RGYOpenCLContext> cl, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, bool copyChroma = true);
        RGYCLFrame *get(int iframe);
        int findIndexByInputFrameId(int inputFrameId) const;
        int inframe() const { return m_nFramesInput; }
        void clear();
        void resetFrames();
    private:
        int m_nFramesInput;
        std::array<std::unique_ptr<RGYCLFrame>, 4> m_buf;
    };

    struct FrameKey {
        int inputFrameId;
        int64_t timestamp;
        int64_t duration;

        FrameKey() : inputFrameId(-1), timestamp(0), duration(0) {}
        explicit FrameKey(const RGYFrameInfo *frame) :
            inputFrameId(frame ? frame->inputFrameId : -1),
            timestamp(frame ? frame->timestamp : 0),
            duration(frame ? frame->duration : 0) {}
        bool matches(const RGYFrameInfo *frame) const {
            return frame
                && inputFrameId == frame->inputFrameId
                && timestamp == frame->timestamp
                && duration == frame->duration;
        }
    };

    struct NnediAdapterState {
        std::unique_ptr<RGYFilterNnedi> filter;
        std::unique_ptr<RGYFilterCspCrop> outputCsp;
        std::array<RGYFrameInfo *, 2> cachedFrames;
        FrameKey cachedKey;
        RGYOpenCLEvent cachedEvent;
        bool cacheValid;

        NnediAdapterState();
        void clear();
    };

    RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamRtgmcEdi> &prm);
    RGY_ERR checkInputs(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame);
    RGY_ERR buildKernels(const std::shared_ptr<RGYFilterParamRtgmcEdi> &prm);
    RGY_ERR run_filter_impl(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame, const RGYFrameInfo *pSourceInputFrame,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pBobInputFrame,
        const RGYFrameInfo *pEdiPrevFrame, const RGYFrameInfo *pEdiInputFrame, const RGYFrameInfo *pEdiNextFrame,
        const RGYFilterParamRtgmcEdi &prm,
        const int targetField,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR runTemporalYadif(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame, const RGYFrameInfo *pSourceInputFrame,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event,
        const RGYFilterParamRtgmcEdi &prm);
    RGY_ERR initNnediAdapterState(NnediAdapterState &state, const std::shared_ptr<RGYFilterParamRtgmcEdi> &prm, const bool chroma);
    RGY_ERR runNnediAdapterState(NnediAdapterState &state, const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pSourceInputFrame,
        RGYFrameInfo **ppOutputFrame, const RGYFrameInfo **ppSelectedFrame,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event,
        const RGYFilterParamRtgmcEdi &prm, const bool chroma);
    RGY_ERR runNnediAdapter(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pSourceInputFrame,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event,
        const RGYFilterParamRtgmcEdi &prm);
    int targetField(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pParityFrame = nullptr);
    void loadDumpEnv();
    bool dumpRequested(int frameIndex) const;

    RGYOpenCLProgramAsync m_edi;
    std::string m_buildOptions;
    std::string m_dumpDir;
    std::string m_dumpJsonl;
    int m_dumpMaxFrames;
    FrameSource m_bobSource;
    FrameSource m_ediSource;
    FrameSource m_inputSource;
    std::array<NnediAdapterState, 2> m_nnediStates;
    RGYOpenCLEvent m_nnediAdapterCopyEvent;
    int m_nFrame;
    int m_lastInputFrameId;
    int m_pairFrameIndex;
    int m_fallbackFrameIndex;
    bool m_useKernel;
};
