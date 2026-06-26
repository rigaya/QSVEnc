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
#include <cstdint>
#include <deque>
#include <vector>

#include "rgy_filter_cl.h"
#include "rgy_filter_rtgmc_bob.h"
#include "rgy_filter_rtgmc_common.h"
#include "rgy_filter_rtgmc_search_prefilter.h"
#include "rgy_filter_rtgmc_edi.h"
#include "rgy_filter_denoise_nlmeans.h"
#include "rgy_filter_denoise_fft3d.h"
#include "rgy_filter_degrain.h"
#include "rgy_filter_rtgmc_retouch.h"
#include "rgy_filter_rtgmc_shimmer_repair.h"
#include "rgy_filter_rtgmc_lossless.h"
#include "rgy_filter_rtgmc_primitive.h"
#if __has_include("rgy_filter_rtgmc_mmask.h")
#include "rgy_filter_rtgmc_mmask.h"
#define RGY_HAS_RTGMC_MMASK_FILTER 1
#endif
#include "rgy_prm.h"

#ifndef RGY_HAS_RTGMC_MMASK_FILTER
#define RGY_HAS_RTGMC_MMASK_FILTER 0
#endif

class RGYFilterParamRtgmc : public RGYFilterParam {
public:
    VppRtgmc rtgmc;
    rgy_rational<int> timebase;
    bool sharedAnalysisMode;

    RGYFilterParamRtgmc() : rtgmc(), timebase(), sharedAnalysisMode(false) {}
    virtual ~RGYFilterParamRtgmc() {}
    virtual tstring print() const override { return rtgmc.print(); }
};

class RGYFilterRtgmc : public RGYFilter {
public:
    RGYFilterRtgmc(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterRtgmc();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
    bool draining() const;
    bool drainComplete() const;

protected:
    struct RtgmcFrameKey {
        int inputFrameId;
        int64_t timestamp;
        int64_t duration;

        RtgmcFrameKey() : inputFrameId(-1), timestamp(0), duration(0) {}
        explicit RtgmcFrameKey(const RGYFrameInfo *frame) :
            inputFrameId(frame ? frame->inputFrameId : -1),
            timestamp(frame ? frame->timestamp : 0),
            duration(frame ? frame->duration : 0) {}
        bool matches(const RGYFrameInfo *frame) const {
            return frame
                && inputFrameId == frame->inputFrameId
                && timestamp == frame->timestamp
                && duration == frame->duration;
        }
        bool matchesFrameIdentity(const RGYFrameInfo *frame) const {
            return frame
                && inputFrameId == frame->inputFrameId
                && timestamp == frame->timestamp;
        }
    };

    struct RtgmcPendingCompRef {
        RtgmcFrameKey key;
        std::shared_ptr<RGYFrameData> backward;
        std::shared_ptr<RGYFrameData> forward;
        RGYOpenCLEvent backwardEvent;
        RGYOpenCLEvent forwardEvent;
        bool hasInlineParams;
        bool inlineParamsChroma;
        std::array<RGYDegrainCompensateInlineParams, 3> backwardInlineParams;
        std::array<RGYDegrainCompensateInlineParams, 3> forwardInlineParams;

        RtgmcPendingCompRef() :
            key(),
            backward(),
            forward(),
            backwardEvent(),
            forwardEvent(),
            hasInlineParams(false),
            inlineParamsChroma(false),
            backwardInlineParams(),
            forwardInlineParams() {
        }
    };

    struct RtgmcPendingEdiRef {
        RtgmcFrameKey key;
        std::shared_ptr<RGYFrameData> edi;
        RGYOpenCLEvent event;
    };

    struct RtgmcSourceCacheFrame {
        RtgmcFrameKey key;
        std::unique_ptr<RGYCLFrame> frame;
        RGYOpenCLEvent event;
    };

    struct RtgmcPendingFrameRef {
        RtgmcFrameKey key;
        std::shared_ptr<RGYCLFrame> frame;
        RGYOpenCLEvent event;
    };

    struct RtgmcMatchCorrectionPass {
        std::unique_ptr<RGYFilterRtgmcEdi> interpolator;
        std::unique_ptr<RGYFilterRtgmcPrimitive> correctionBuild;
        std::unique_ptr<RGYFilterRtgmcPrimitive> correctionSpatialPrepass;
        bool fusedCorrectionBuild = false;
        bool fusedCorrectionApply = false;
        std::unique_ptr<RGYFilterDegrain> correctionTemporalFilter;
        std::unique_ptr<RGYFilterRtgmcPrimitive> correctionApply;
        std::unique_ptr<RGYFilterRtgmcRetouch> correctionEnhance;
        std::deque<RtgmcPendingFrameRef> composeBaseRefs;
    };

public:
    struct RtgmcSharedAnalysisData {
        RGYFilterDegrain *analyzeFilter;
        std::deque<RtgmcPendingEdiRef> *pendingEdiRefs;
        std::array<RtgmcSourceCacheFrame, 256> *sourceCache;
        std::deque<RtgmcPendingCompRef> *pendingCompRefs;
        std::deque<RtgmcPendingFrameRef> *pendingNoiseRefs;
        std::shared_ptr<RGYCLSharedFramePool> sharedFramePool;

        RtgmcSharedAnalysisData() : analyzeFilter(nullptr), pendingEdiRefs(nullptr),
            sourceCache(nullptr), pendingCompRefs(nullptr), pendingNoiseRefs(nullptr), sharedFramePool() {}
    };

    void setSharedAnalysisData(const RtgmcSharedAnalysisData &data);
    RtgmcSharedAnalysisData getSharedAnalysisData();

    struct RtgmcCapturedIntermediate {
        std::shared_ptr<RGYCLFrame> frame;
        RGYFrameInfo frameInfo;
        RGYOpenCLEvent event;
    };

    void enableIntermediateCapture(bool enable);
    const std::vector<RtgmcCapturedIntermediate>& getCapturedIntermediates() const;
    void clearCapturedIntermediates();
    void pushIntermediateInput(const RtgmcCapturedIntermediate &input);

protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;
public:
    virtual void resetTemporalState() override;
    int requiredPrimingSourceFrames() const;
protected:

    RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamRtgmc> &prm);
    RGY_ERR initFilters(const std::shared_ptr<RGYFilterParamRtgmc> &prm);
    RGY_ERR initRetouchCompFilters(const std::shared_ptr<RGYFilterParamRtgmc> &prm, const RGYFrameInfo &frameInfo, const rgy_rational<int> &baseFps);
    RGY_ERR initSourceMatchCorrectionFilters(const std::shared_ptr<RGYFilterParamRtgmc> &prm, const RGYFrameInfo &sourceFrameIn,
        const RGYFrameInfo &frameInfo, const rgy_rational<int> &sourceBaseFps, const rgy_rational<int> &sourceTimebase,
        const rgy_rational<int> &baseFps);
    RGY_ERR runNestedFilter(size_t filterIdx, RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR runSourceMatchCorrectionPasses(int firstStage, int lastStage, RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR runSourceMatchCorrectionPass(int stage, RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR runThrough(size_t filterIdx, RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event, bool storePending);
    RGY_ERR drainFrom(size_t filterIdx, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, RGYOpenCLEvent *event);
    RGY_ERR returnPendingFrames(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum);
    RGY_ERR initBorderFrame(const RGYFrameInfo &frameInfo);
    RGY_ERR buildBorderCopyProgram(const RGYFrameInfo &frameInfo);
    RGY_ERR runBorderCopy(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, bool crop,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR addBorderToInput(const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR copyFinalOutput(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    std::shared_ptr<RGYCLFrame> getSharedFrameBuffer(const RGYFrameInfo *frame);
    bool noiseRestoreEnabled() const;
    RGY_ERR storeNoiseReference(const RGYFrameInfo *baseFrame, RGYFrameInfo *denoisedFrame,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RtgmcPendingFrameRef *findNoiseReference(const RGYFrameInfo *frame);
    void clearNoiseReference(const RGYFrameInfo *frame);
    RGY_ERR cacheSourceFrame(const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    const RtgmcSourceCacheFrame *findCachedSourceEntry(const RGYFrameInfo *frame) const;
    const RGYFrameInfo *findCachedSourceFrame(const RGYFrameInfo *frame, std::vector<RGYOpenCLEvent> *wait_events);
    int sourceFieldForFrame(const RGYFrameInfo *frame) const;
    size_t sourceCacheCapacity() const;
    RGY_ERR storePostLimitBaseReference(const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RtgmcPendingFrameRef *findPostLimitBaseReference(const RGYFrameInfo *frame);
    void clearPostLimitBaseReference(const RGYFrameInfo *frame);
    RGY_ERR storeMatchCorrectionBaseReference(int stage, const RGYFrameInfo *keyFrame, const RGYFrameInfo *baseFrame,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RtgmcPendingFrameRef *findMatchCorrectionBaseReference(int stage, const RGYFrameInfo *frame);
    void clearMatchCorrectionBaseReference(int stage, const RGYFrameInfo *frame);
    void enqueueSourceMatchFrameProp(const RGYFrameInfo *frame);
    RGY_ERR applySourceMatchFrameProp(RGYFrameInfo *frame);
    RGY_ERR attachEdiReference(RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR updateCompReferenceStore(const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR drainCompReferenceStore(RGYOpenCLQueue &queue);
    void storeEdiReference(const RGYFrameInfo *frame, const std::shared_ptr<RGYFrameDataRtgmcEdi> &edi, const RGYOpenCLEvent &event);
    RtgmcPendingEdiRef *findStoredEdiReference(const RGYFrameInfo *frame);
    void clearStoredEdiReference(const RGYFrameInfo *frame);
    void storeCompReference(const RGYFrameInfo *frame, const std::shared_ptr<RGYFrameData> &backward, const std::shared_ptr<RGYFrameData> &forward,
        const RGYOpenCLEvent &backwardEvent, const RGYOpenCLEvent &forwardEvent);
    RtgmcPendingCompRef *findStoredCompReference(const RGYFrameInfo *frame);
    void clearStoredCompReferences(const RGYFrameInfo *frame);
    void attachStoredEdiReference(RGYFrameInfo *frame, std::vector<RGYOpenCLEvent> *wait_events);
    void attachStoredCompReferences(RGYFrameInfo *frame, std::vector<RGYOpenCLEvent> *wait_events);

    std::vector<std::unique_ptr<RGYFilter>> m_filters;
    RGYFilter *m_noiseFilter;
    std::array<std::unique_ptr<RGYFilterDegrain>, 2> m_retouchCompFilters;
    std::array<RtgmcMatchCorrectionPass, 3> m_matchCorrectionPasses;
    std::deque<RtgmcPendingEdiRef> m_pendingEdiRefs;
    std::deque<RtgmcPendingCompRef> m_pendingCompRefs;
    std::deque<RtgmcPendingFrameRef> m_pendingPostLimitBaseRefs;
    std::deque<int> m_pendingOutputFrames;
    std::array<RtgmcSourceCacheFrame, 256> m_sourceCache;
    std::shared_ptr<RGYCLSharedFramePool> m_sharedFramePool;
    std::shared_ptr<RGYCLSharedFramePool> m_ediSideDataFramePool;
    std::unique_ptr<RGYCLFrame> m_borderFrame;
    RGYOpenCLProgramAsync m_borderCopy;
    std::unique_ptr<RGYFilterRtgmcPrimitive> m_noiseDiffFilter;
    std::deque<RtgmcPendingFrameRef> m_pendingNoiseRefs;
    RGYFrameInfo m_inputFrame;
    int m_sourceCacheNext;
    int m_outputBufferIndex;
    std::deque<RtgmcFrameKey> m_pendingSourceMatchFrameProps;
    size_t m_drainFilterIdx;
    bool m_draining;
    bool m_drainComplete;
    int m_debugResetAtFrame;
    int m_nFrame;
    bool m_attachRetouchCompRefs;
    bool m_enablePostTR2Limit;
    bool m_sharedAnalysisMode;
    RtgmcSharedAnalysisData m_sharedData;
    bool m_captureIntermediate;
    std::vector<RtgmcCapturedIntermediate> m_capturedIntermediates;
    std::deque<RtgmcCapturedIntermediate> m_pendingIntermediateInputs;
};
