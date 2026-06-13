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
#ifndef __RGY_FILTER_KFM_H__
#define __RGY_FILTER_KFM_H__

#include <array>
#include <cstdio>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "rgy_filter_cl.h"
#include "rgy_filter_kfm_analyze.h"

class RGYFilterRtgmc;
class RGYFilterDegrain;
class RGYCLSharedFramePool;

class RGYFilterParamKfm : public RGYFilterParam {
public:
    VppKfm kfm;
    rgy_rational<int> timebase;

    RGYFilterParamKfm() : kfm(), timebase() {};
    virtual ~RGYFilterParamKfm() {};
    virtual tstring print() const override;
};

class RGYFrameDataKfmSwitch : public RGYFrameData {
public:
    RGYFrameDataKfmSwitch(int n60, int n24, int baseType, int sourceStart, int numSourceFrames, int duration60, int duration120, int pattern, float cost) :
        m_n60(n60),
        m_n24(n24),
        m_baseType(baseType),
        m_sourceStart(sourceStart),
        m_numSourceFrames(numSourceFrames),
        m_duration60(duration60),
        m_duration120(duration120),
        m_pattern(pattern),
        m_cost(cost) {
        m_dataType = RGY_FRAME_DATA_KFM_SWITCH;
    }
    virtual ~RGYFrameDataKfmSwitch() {};

    int n60() const { return m_n60; }
    int n24() const { return m_n24; }
    int baseType() const { return m_baseType; }
    int sourceStart() const { return m_sourceStart; }
    int numSourceFrames() const { return m_numSourceFrames; }
    int duration60() const { return m_duration60; }
    int duration120() const { return m_duration120; }
    int pattern() const { return m_pattern; }
    float cost() const { return m_cost; }

protected:
    int m_n60;
    int m_n24;
    int m_baseType;
    int m_sourceStart;
    int m_numSourceFrames;
    int m_duration60;
    int m_duration120;
    int m_pattern;
    float m_cost;
};

class RGYFilterKfm : public RGYFilter {
public:
    RGYFilterKfm(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterKfm();

    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
    virtual int requiredOutputFrames() const override;

protected:
    struct KfmCachedSource;
    struct KfmCachedDeint60;
    struct KfmCachedUcfNoise;
    struct KfmUcfNoiseDumpRecord;
    struct KfmPendingUcfNoiseResult;
    struct KfmUcfGaussProgram;
    struct KfmPendingVfrOutput;
    enum KfmFrameType {
        KFM_FRAME_60 = 1,
        KFM_FRAME_30 = 2,
        KFM_FRAME_24 = 3,
        KFM_FRAME_UCF = 4,
    };
    enum KfmUcf60Flag {
        KFM_UCF60_NONE = 0,
        KFM_UCF60_NR = 1,
        KFM_UCF60_PREV = 2,
        KFM_UCF60_NEXT = 3,
    };
    enum KfmUcf24SelectType {
        KFM_UCF24_SELECT_DEINT24 = 0,
        KFM_UCF24_SELECT_FRAME = 1,
        KFM_UCF24_SELECT_DWEAVE = 2,
    };
    struct KfmUcf24Selection {
        KfmUcf24SelectType type;
        int n60;
        const RGYFrameInfo *frame;

        KfmUcf24Selection() : type(KFM_UCF24_SELECT_DEINT24), n60(-1), frame(nullptr) {};
    };
    struct KfmSwitchTiming {
        int start60;
        int start120;
        int sourceIndex;
        int frame24Index;
        int baseType;
        int sourceStart;
        int numSourceFrames;
        int duration60;
        int duration120;
        bool isFrame24;
        bool isFrame60;

        KfmSwitchTiming() : start60(0), start120(0), sourceIndex(0), frame24Index(-1), baseType(KFM_FRAME_60), sourceStart(0), numSourceFrames(1), duration60(1), duration120(2), isFrame24(false), isFrame60(false) {};
    };
    struct KfmContainsCombeReadback {
        bool submitted;

        KfmContainsCombeReadback() : submitted(false) {};
    };

    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    RGY_ERR loadPrograms(const RGYFilterParamKfm& prm);
    RGY_ERR initRtgmc(const std::shared_ptr<RGYFilterParamKfm>& prm, std::unique_ptr<RGYFilterRtgmc>& rtgmc, bool updateOutputParam, int useFlag = 0, bool sharedAnalysisMode = false);
    RGY_ERR initNrFilter(const std::shared_ptr<RGYFilterParamKfm>& prm);
    RGY_ERR initAnalyzer(const RGYFilterParamKfm& prm);
    RGY_ERR padSourceFrame(RGYFrameInfo *pPaddedFrame, const RGYFrameInfo *pSourceFrame,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event, bool sourceInPaddedFrame = false);
    RGY_ERR cacheSourceFrame(const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR runDeint60Branch(const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, int *cachedFrames = nullptr);
    RGY_ERR drainDeint60Branch(RGYOpenCLQueue &queue, int *cachedFrames = nullptr);
    RGY_ERR cacheDeint60Frame(const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR runUcfRtgmcBranches(const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR runUcfRtgmcBranch(RGYFilterRtgmc *rtgmc, const char *stage, const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, std::deque<KfmCachedDeint60>& cache, int& submittedFrames, RGYOpenCLEvent& cacheCopyEvent);
    RGY_ERR drainUcfRtgmcBranch(RGYFilterRtgmc *rtgmc, const char *stage, RGYOpenCLQueue &queue, std::deque<KfmCachedDeint60>& cache, int& submittedFrames, RGYOpenCLEvent& cacheCopyEvent);
    RGY_ERR processMainRtgmcOutputs(const RGYFilterParamKfm& prm, RGYFrameInfo **rtgmcOutFrames, int rtgmcOutNum,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR drainMainRtgmcBranch(const RGYFilterParamKfm& prm, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, RGYOpenCLEvent *event);
    RGY_ERR cacheUcfRtgmcFrame(const char *stage, const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, std::deque<KfmCachedDeint60>& cache, int& submittedFrames, RGYOpenCLEvent *event);
    size_t sourceCacheLimit() const;
    size_t deint60CacheLimit() const;
    int sourceCacheTrimFloor() const;
    int deint60CacheTrimFloor() const;
    const RGYFrameInfo *findDeint60Frame(int n60, std::vector<RGYOpenCLEvent> *wait_events) const;
    const RGYFrameInfo *findSourceFrame(const RGYFrameInfo *frame, std::vector<RGYOpenCLEvent> *wait_events);
    const KfmCachedSource *findSourceByIndex(int sourceIndex) const;
    const KfmCachedSource *findSourceByIndexExact(int sourceIndex) const;
    const KfmCachedDeint60 *findCachedDeint60Frame(const std::deque<KfmCachedDeint60>& cache, int n60, std::vector<RGYOpenCLEvent> *wait_events) const;
    const KfmUcfNoiseDumpRecord *findUcfNoiseResult(int sourceIndex) const;
    RGY_ERR ensureFMCountQueue();
    RGY_ERR submitFMCounts(int cycle, bool drain, RGYOpenCLQueue &queue);
    RGY_ERR readbackFMCounts(std::array<RGYKFM::FMCount, 18>& counts, int cycle, bool drain, RGYOpenCLQueue &queue);
    RGY_ERR analyzeAvailableSource(bool drain, RGYOpenCLQueue &queue);
    void finalizeAnalyzerResults(VppKfmTiming timing);
    std::vector<RGYKFM::KFMResult> analyzerResultsSnapshot(bool mark60p) const;
    void appendAnalyzerResults(size_t resultCount, bool dump, bool mark60p);
    std::vector<KfmSwitchTiming> deriveSwitchTimings(int total60) const;
    int64_t sourceFrameDuration(const KfmCachedSource *source) const;
    bool isSwitchSingleFrameN60(int n60) const;
    void markSwitchSingleFrameN60Range(int start60, int duration60);
    bool switchSingleFrameDurationEnabled() const;
    void writeSwitchTimingDump();
    void writeTelecine24DurationDump();
    void writeFMCountDump(const std::array<RGYKFM::FMCount, 18>& counts, int cycle);
    void writeAnalyzerResult(const RGYKFM::KFMResult& result, bool dump);
    void writeAnalyzerResultsFinal(size_t resultCount, bool mark60p);
    void writeFrameTimecode(const RGYFrameInfo *frame);
    void writeFrameInfoDump(const char *stage, const RGYFrameInfo *frame, const RGYKFM::KFMResult *result = nullptr);
    void writeContainsCombeDump(const char *stage, const KfmSwitchTiming& timing, cl_uint containsCombeCount, bool durationApplied, const RGYKFM::KFMResult *result);
    void attachSwitchFrameData(RGYFrameInfo *frame, const KfmSwitchTiming& timing, const RGYKFM::KFMResult *result) const;
    void initStageDumpConfig(const RGYFilterParamKfm& prm);
    bool stageDumpRequested(int frame24Index) const;
    RGY_ERR dumpStageFrame(const char *stage, const RGYFrameInfo *frame, int frame24Index,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR copyUcfFrame(const RGYFilterParamKfm& prm, RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR createUcfGaussProgram(KfmUcfGaussProgram& program, int sourceSize, double cropSize, int targetSize, double p);
    RGY_ERR prepareUcfNoiseFieldCropFrame(RGYFrameInfo **ppFieldFrame, int sourceIndex, int parity, const RGYFrameInfo *pInputFrame,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR prepareUcfNoiseGaussFrameFromSource(RGYFrameInfo **ppGaussFrame, int sourceIndex, int parity, const RGYFrameInfo *pInputFrame,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR prepareUcfNoiseGaussFrame(RGYFrameInfo **ppGaussFrame, int parity, const RGYFrameInfo *pInputFrame,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR runUcfNoiseLimitStageFromSource(const RGYFilterParamKfm& prm, const RGYFrameInfo *pSrcFrame, const RGYFrameInfo *pNoiseFrame,
        int fieldIndex, int parity, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR runUcfNoiseLimitStage(const RGYFilterParamKfm& prm, const RGYFrameInfo *pSrcFrame, const RGYFrameInfo *pNoiseFrame,
        int fieldIndex, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR analyzeUcfNoiseDebug(RGYOpenCLQueue &queue);
    RGY_ERR submitUcfNoiseResult(const KfmCachedUcfNoise& noise0, const KfmCachedUcfNoise& noise1, const KfmCachedUcfNoise& noise2,
        const KfmCachedSource& source0, const KfmCachedSource& source1, RGYOpenCLQueue &queue);
    RGY_ERR resolveUcfNoiseResult(KfmPendingUcfNoiseResult& pending, RGYOpenCLQueue &queue);
    RGY_ERR resolveUcfNoiseResults(int sourceIndex, RGYOpenCLQueue &queue);
    RGY_ERR resolveAllUcfNoiseResults(RGYOpenCLQueue &queue);
    RGY_ERR clearPendingFMCounts();
    std::unique_ptr<RGYCLBuf> acquireFMCountBuf(size_t requiredBytes);
    void releaseFMCountBuf(std::unique_ptr<RGYCLBuf>&& buf);
    std::unique_ptr<RGYCLBuf> acquireUcfNoiseResultBuf(size_t requiredBytes);
    void releaseUcfNoiseResultBuf(std::unique_ptr<RGYCLBuf>&& buf);
    void pushUcfNoiseResultDump(int sourceIndex, const RGYKFM::NoiseResult (&results)[2], const RGYKFM::UCFNoiseMeta& meta);
    void writeUcfNoiseResultDump(const KfmUcfNoiseDumpRecord& record, const KfmUcfNoiseDumpRecord *nextRecord);
    void flushUcfNoiseResultDump();
    const RGYFrameInfo *selectUcfDecomb30Frame(int sourceIndex, const RGYFrameInfo *deint30, std::vector<RGYOpenCLEvent> *wait_events) const;
    bool getUcf60FieldDiff(int nstart, double (&diff)[4]) const;
    KfmUcf60Flag calcUcf60Flag(int n60) const;
    const RGYFrameInfo *selectUcfDecomb60Frame(int n60, const RGYFrameInfo *deint60, std::vector<RGYOpenCLEvent> *wait_events) const;
    KfmUcf24Selection selectUcfDecomb24Frame(const RGYKFM::Frame24Info& frameInfo, const RGYFrameInfo *deint24, std::vector<RGYOpenCLEvent> *wait_events) const;
    RGY_ERR runNrFilter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrame,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR emitOutputFrame(RGYFrameInfo *pFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const RGYOpenCLEvent &frameEvent, RGYOpenCLEvent *event);
    RGY_ERR queueVfrOutputFrame(const RGYFrameInfo *pFrame, RGYOpenCLQueue &queue, const RGYOpenCLEvent &frameEvent);
    RGY_ERR emitPendingVfrOutput(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, RGYOpenCLEvent *event);
    RGY_ERR emitPendingVfrOutputs(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, RGYOpenCLEvent *event, int keepFrames);
    RGY_ERR drainNrFilter(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, RGYOpenCLEvent *event);
    RGY_ERR clearStaticFlag(RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event = nullptr);
    RGY_ERR analyzeStaticFlag(int sourceIndex, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR analyzeStaticFlag(RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR mergeStatic(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pDeint60Frame, const RGYFrameInfo *pSourceFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR renderTelecine24(RGYFrameInfo *pOutputFrame, int frame24Index, bool drain, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR renderDoubleWeaveFrame(RGYFrameInfo *pOutputFrame, int firstField, int fieldCount, bool drain, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR renderCleanSuperFields(RGYFrameInfo *pOutputFrame, int firstSuperField, int lastSuperField, int propSourceIndex, int outputFrameId, bool drain, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR renderTelecineSuper24(RGYFrameInfo *pOutputFrame, int frame24Index, bool drain, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR renderSuper30(RGYFrameInfo *pOutputFrame, int frame30Index, bool drain, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR removeCombeFields(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pDeintFrame, const RGYFrameInfo *pTelecineSuperFrame,
        int firstField, int fieldCount, int stageFrameIndex, const char *stageName, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR removeCombe24(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pDeint24Frame, const RGYFrameInfo *pTelecineSuperFrame, int frame24Index, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR ensureMaskBranchFrames(RGYFrameInfo **ppSwitchFlagFrame, RGYFrameInfo **ppContainsCombeFrame, RGYFrameInfo **ppCombeMaskFrame, const RGYFrameInfo *pTelecineSuperFrame, const TCHAR *stageLabel);
    RGY_ERR renderMaskBranch(RGYFrameInfo *pSwitchFlagFrame, RGYFrameInfo *pContainsCombeFrame, RGYFrameInfo *pCombeMaskFrame, const RGYFrameInfo *pTelecineSuperPrevFrame, const RGYFrameInfo *pTelecineSuperFrame, const RGYFrameInfo *pTelecineSuperNextFrame, const char *switchFlagStage, const char *containsCombeStage, const char *combeMaskStage, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event, KfmContainsCombeReadback *containsCombeReadback = nullptr);
    RGY_ERR resolveContainsCombeCount(KfmContainsCombeReadback& readback, cl_uint *containsCombeCount);
    RGY_ERR patchCombe(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pBaseFrame, const RGYFrameInfo *pPatchFrame, const RGYFrameInfo *pMaskFrame, int frameIndex, const char *stageName, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    int telecine24FrameCount(bool drain) const;
    std::shared_ptr<RGYCLFrame> acquireKfmFrame(const RGYFrameInfo& info, const TCHAR *label, cl_mem_flags flags = CL_MEM_READ_WRITE);
    struct KfmSourceSlot;
    std::shared_ptr<KfmSourceSlot> acquireKfmSourceSlot(const RGYFrameInfo& sourceInfo, cl_mem_flags flags);
    void retireKfmSourceSlot(std::shared_ptr<KfmSourceSlot>&& slot, RGYOpenCLQueue &queue);
    void collectRetiredKfmSourceSlots();
    void trimFreeKfmSourceSlots();
    void clearKfmSourceSlotPool(bool wait);
    void trimSourceCache(RGYOpenCLQueue &queue);
    void trimDeint60Cache(std::deque<KfmCachedDeint60>& cache);
    RGY_ERR allocWorkFrameBuf(const RGYFrameInfo& frame, int frames);
    RGYFrameInfo *nextOutputFrame();
    RGYFrameInfo *nextWorkFrame();

    struct KfmSourceSlot {
        std::shared_ptr<RGYCLFrame> paddedFrame;
        std::shared_ptr<RGYCLFrame> sourceFrame;
        RGYOpenCLEvent readyEvent;
        cl_mem_flags flags;

        KfmSourceSlot() : paddedFrame(), sourceFrame(), readyEvent(), flags(0) {};
    };

    struct KfmCachedSource {
        int sourceIndex;
        int inputFrameId;
        int64_t timestamp;
        std::shared_ptr<KfmSourceSlot> slot;
        std::shared_ptr<RGYCLFrame> frame;
        std::shared_ptr<RGYCLFrame> paddedFrame;
        RGYOpenCLEvent event;
        RGYOpenCLEvent paddedEvent;

        KfmCachedSource() : sourceIndex(-1), inputFrameId(-1), timestamp(0), slot(), frame(), paddedFrame(), event(), paddedEvent() {};
    };

    struct KfmCachedDeint60 {
        int n60;
        int inputFrameId;
        int64_t timestamp;
        int64_t duration;
        std::shared_ptr<RGYCLFrame> frame;
        RGYOpenCLEvent event;

        KfmCachedDeint60() : n60(-1), inputFrameId(-1), timestamp(0), duration(0), frame(), event() {};
    };

    struct KfmCachedUcfNoise {
        int fieldIndex;
        int inputFrameId;
        int64_t timestamp;
        std::shared_ptr<RGYCLFrame> frame;
        RGYOpenCLEvent event;

        KfmCachedUcfNoise() : fieldIndex(-1), inputFrameId(-1), timestamp(0), frame(), event() {};
    };

    struct KfmUcfNoiseDumpRecord {
        int sourceIndex;
        RGYKFM::NoiseResult results[2];
        RGYKFM::UCFNoiseMeta meta;
        bool valid;

        KfmUcfNoiseDumpRecord() : sourceIndex(-1), results(), meta(), valid(false) {};
    };

    struct KfmPendingUcfNoiseResult {
        struct Segment {
            int offset;
            int count;
            int plane;
        };
        int sourceIndex;
        std::unique_ptr<RGYCLBuf> resultBuf;
        std::vector<Segment> segments;
        RGYKFM::UCFNoiseMeta meta;

        KfmPendingUcfNoiseResult() : sourceIndex(-1), resultBuf(), segments(), meta() {};
    };

    struct KfmUcfGaussProgram {
        int sourceSize;
        int targetSize;
        int filterSize;
        std::unique_ptr<RGYCLBuf> offset;
        std::unique_ptr<RGYCLBuf> coeff;

        KfmUcfGaussProgram() : sourceSize(0), targetSize(0), filterSize(0), offset(), coeff() {};
    };

    struct KfmPendingFMCount {
        int cycle;
        std::unique_ptr<RGYCLBuf> countBuf;

        KfmPendingFMCount() : cycle(-1), countBuf() {};
    };

    struct KfmPendingVfrOutput {
        std::shared_ptr<RGYCLFrame> frame;
        RGYOpenCLEvent event;

        KfmPendingVfrOutput() : frame(), event() {};
    };

    std::array<RGYOpenCLProgramAsync, 8> m_programs;
    std::unique_ptr<RGYFilterRtgmc> m_rtgmc;
    std::unique_ptr<RGYFilterRtgmc> m_deint60Rtgmc;
    std::unique_ptr<RGYFilterRtgmc> m_before60Rtgmc;
    std::unique_ptr<RGYFilterRtgmc> m_after60Rtgmc;
    std::unique_ptr<RGYFilterDegrain> m_nrFilter;
    std::unique_ptr<RGYKFM::KFMAnalyze> m_analyzer;
    std::shared_ptr<RGYCLSharedFramePool> m_kfmFramePool;
    std::deque<std::shared_ptr<KfmSourceSlot>> m_kfmSourceSlotFree;
    std::deque<std::shared_ptr<KfmSourceSlot>> m_kfmSourceSlotRetired;
    std::deque<KfmCachedSource> m_sourceCache;
    std::deque<KfmCachedDeint60> m_deint60Cache;
    std::deque<KfmCachedDeint60> m_before60Cache;
    std::deque<KfmCachedDeint60> m_after60Cache;
    std::deque<KfmCachedUcfNoise> m_ucfNoiseCache;
    std::deque<KfmPendingUcfNoiseResult> m_pendingUcfNoiseResults;
    std::deque<std::unique_ptr<RGYCLBuf>> m_fmCountBufPool;
    std::deque<std::unique_ptr<RGYCLBuf>> m_ucfNoiseResultBufPool;
    std::deque<KfmUcfNoiseDumpRecord> m_ucfNoiseResultCache;
    KfmUcfNoiseDumpRecord m_pendingUcfNoiseDump;
    int m_deint60SubmittedSourceFrames;
    int m_before60SubmittedSourceFrames;
    int m_after60SubmittedSourceFrames;
    RGYOpenCLEvent m_deint60CacheCopyEvent;
    RGYOpenCLEvent m_before60CacheCopyEvent;
    RGYOpenCLEvent m_after60CacheCopyEvent;
    std::unique_ptr<RGYCLFrame> m_staticFlag;
    std::array<std::unique_ptr<RGYCLFrame>, 5> m_staticWorkFrames;
    std::array<std::unique_ptr<RGYCLBuf>, 2> m_analyzeFlags;
    RGYOpenCLQueue m_fmCountQueue;
    std::deque<KfmPendingFMCount> m_pendingFMCounts;
    std::deque<KfmPendingVfrOutput> m_pendingVfrOutputs;
    std::array<std::unique_ptr<RGYCLBuf>, 2> m_telecineSuperRaw;
    std::array<std::unique_ptr<RGYCLFrame>, 2> m_telecineSuperFrames;
    std::array<std::unique_ptr<RGYCLFrame>, 2> m_telecineSuperNeighborFrames;
    std::array<std::unique_ptr<RGYCLFrame>, 4> m_switchFlagFrames;
    std::array<std::unique_ptr<RGYCLFrame>, 4> m_containsCombeFrames;
    std::array<std::unique_ptr<RGYCLFrame>, 4> m_combeMaskFrames;
    std::array<std::unique_ptr<RGYCLFrame>, 4> m_patchCombeFrames;
    std::array<std::unique_ptr<RGYCLFrame>, 2> m_ucfNoiseFieldFrames;
    std::array<std::unique_ptr<RGYCLFrame>, 2> m_ucfNoiseGaussTmpFrames;
    std::array<std::unique_ptr<RGYCLFrame>, 2> m_ucfNoiseGaussFrames;
    std::array<std::array<KfmUcfGaussProgram, 2>, 2> m_ucfNoiseGaussVert;
    std::array<std::array<KfmUcfGaussProgram, 2>, 2> m_ucfNoiseGaussHori;
    std::array<std::unique_ptr<RGYCLBuf>, 4> m_switchFlagWork;
    RGYOpenCLEvent m_switchFlagWorkEvent;
    std::unique_ptr<RGYCLBuf> m_containsCombeCount;
    FILE *m_fpResult;
    FILE *m_fpFMCount;
    FILE *m_fpTimecode;
    FILE *m_fpFrameInfo;
    FILE *m_fpContainsCombe;
    FILE *m_fpUcfNoise;
    tstring m_switchDurationPath;
    tstring m_switchTimecodePath;
    std::string m_stageDumpDir;
    RGYKFM::KFMResult m_lastAnalyzeResult;
    std::vector<RGYKFM::KFMResult> m_analyzerOutputResults;
    bool m_hasLastAnalyzeResult;
    bool m_analyzerFinalized;
    bool m_switchTimingDumped;
    int m_analyzeSourceFrames;
    int m_nextAnalyzeCycle;
    int m_nextFMCountSubmitCycle;
    int m_nextFMCountDumpFrame;
    int m_cachedSourceFrames;
    int m_nextSwitchN60;
    int64_t m_nextSwitchPts;
    bool m_hasLastSwitchTiming;
    int m_lastSwitchStart60;
    int m_lastSwitchDuration60;
    int64_t m_lastSwitchStart120;
    bool m_lastSwitchIsFrame24;
    std::vector<int> m_switchSingleFrameN60;
    std::unordered_map<std::string, int> m_stageDumpFrameCounts;
    std::unordered_map<std::string, std::unordered_set<int>> m_stageDumpFrameIndices;
    std::unordered_set<int> m_stageDumpTargetFrames;
    int m_nextTelecine24Frame;
    int64_t m_nextTelecine24Pts;
    int m_telecineSuperBufferIndex;
    int m_maskBranchBufferIndex;
    int m_patchCombeBufferIndex;
    int m_stageDumpMaxFrames;
    int m_timecodeFrameIndex;
    int m_outputBufferIndex;
    std::vector<std::unique_ptr<RGYCLFrame>> m_workFrameBuf;
    int m_workBufferIndex;
};

#endif //__RGY_FILTER_KFM_H__
