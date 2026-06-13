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
#include <deque>
#include <fstream>
#include <string>
#include <vector>

#include "rgy_filter_cl.h"
#include "rgy_filter_rtgmc_repair_profile.h"

class RGYFilterResizePlaneProxy;

class RGYFilterParamRtgmcSearchPrefilter : public RGYFilterParam {
public:
    int tr0;
    int searchRefine;
    int rep0Thin;
    int rep0Pad;
    RGYRtgmcRepairProfile repairProfile;
    bool tvRange;
    bool chromaMotion;
    bool attachSearchLuma;
    tstring dumpY4m;
    tstring dumpStage;
    int dumpMaxFrames;

    RGYFilterParamRtgmcSearchPrefilter() : tr0(1), searchRefine(1), rep0Thin(1), rep0Pad(0), repairProfile(), tvRange(true), chromaMotion(false), attachSearchLuma(false), dumpY4m(), dumpStage(), dumpMaxFrames(0) {}
    virtual ~RGYFilterParamRtgmcSearchPrefilter() {}
    virtual tstring print() const override;
};

class RGYFrameDataRtgmcSearchLuma : public RGYFrameData {
public:
    RGYFrameDataRtgmcSearchLuma(std::shared_ptr<RGYCLFrame> frame, int bitdepth);
    virtual ~RGYFrameDataRtgmcSearchLuma() {}
    const RGYFrameInfo *frame() const;
    RGYCLFrame *clFrame() const { return m_frame.get(); }
    int bitdepth() const { return m_bitdepth; }
protected:
    std::shared_ptr<RGYCLFrame> m_frame;
    int m_bitdepth;
};

class RGYFilterRtgmcSearchPrefilter : public RGYFilter {
public:
    static constexpr int RTGMC_SEARCH_PREFILTER_CACHE_SIZE = 5;

    RGYFilterRtgmcSearchPrefilter(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterRtgmcSearchPrefilter();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;

protected:
    struct SearchRefine1PlaneResources {
        std::unique_ptr<RGYCLFrame> motionGuide;
        std::unique_ptr<RGYCLFrame> halfSearchBase;
        std::unique_ptr<RGYCLFrame> halfSearchSmoothed;

        void clear() {
            motionGuide.reset();
            halfSearchBase.reset();
            halfSearchSmoothed.reset();
        }
    };
    struct SearchRefine2PlaneResources {
        std::unique_ptr<RGYCLFrame> motionGuide;
        std::unique_ptr<RGYCLFrame> searchSmoothed3x3;
        std::unique_ptr<RGYCLFrame> edgeSoftenedSearch;
        std::unique_ptr<RGYCLFrame> preStabilizedSearch;

        void clear() {
            motionGuide.reset();
            searchSmoothed3x3.reset();
            edgeSoftenedSearch.reset();
            preStabilizedSearch.reset();
        }
    };

    struct SharedFramePool : public std::enable_shared_from_this<SharedFramePool> {
        struct Entry {
            std::unique_ptr<RGYCLFrame> frame;
            RGYOpenCLEvent readyEvent;
        };
        std::shared_ptr<RGYOpenCLContext> cl;
        std::deque<Entry> frames;

        SharedFramePool(std::shared_ptr<RGYOpenCLContext> context) : cl(context), frames() {};
        std::shared_ptr<RGYCLFrame> get(const RGYFrameInfo &frameInfo);
        void recycle(RGYCLFrame *frame);
        void clear();
    };
    struct PendingSceneChangePlane {
        RGY_PLANE plane;
        tstring planeName;
        int smoothRadius;
        int groupCount;
        uint64_t sceneThreshold;
        std::array<int, 4> flags;
        std::unique_ptr<RGYCLBuf> partial;
        RGYOpenCLEvent mapEvent;
        bool mapSubmitted;

        PendingSceneChangePlane() :
            plane(RGY_PLANE_Y), planeName(), smoothRadius(0), groupCount(0), sceneThreshold(0), flags(),
            partial(), mapEvent(), mapSubmitted(false) {
            flags.fill(0);
        }
    };
    struct PendingSearchPrefilterFrame {
        int currentFrame;
        std::array<std::shared_ptr<RGYCLFrame>, RTGMC_SEARCH_PREFILTER_CACHE_SIZE> refs;
        std::vector<PendingSceneChangePlane> scenePlanes;

        PendingSearchPrefilterFrame() : currentFrame(-1), refs(), scenePlanes() {}
    };

    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;
public:
    virtual void resetTemporalState() override;
protected:

    RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamRtgmcSearchPrefilter> &prm);
    RGY_ERR buildKernel(const std::shared_ptr<RGYFilterParamRtgmcSearchPrefilter> &prm);
    RGY_ERR allocCacheFrames(const RGYFrameInfo &frameInfo);
    RGY_ERR setupSearchRefine1Resources(const RGYFrameInfo &frameInfo, bool processChroma);
    RGY_ERR setupSearchRefine2Resources(const RGYFrameInfo &frameInfo, bool processChroma);
    std::shared_ptr<RGYCLFrame> createSearchLumaFrame(const RGYFrameInfo &frameInfo, bool includeChroma);
    std::unique_ptr<RGYCLFrame> createPlaneFrame(const RGYFrameInfo &frameInfo);
    RGY_ERR pushCacheFrame(const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR checkSameResolutionPlanePitches(const TCHAR *stageName, const std::vector<const RGYFrameInfo *> &planes);
    RGY_ERR checkTemporalPlanePitches(const TCHAR *planeName,
        const RGYFrameInfo *prev2, const RGYFrameInfo *prev, const RGYFrameInfo *cur, const RGYFrameInfo *next, const RGYFrameInfo *next2);
    std::unique_ptr<RGYCLBuf> getSceneChangeBuffer(size_t requiredSize);
    void recycleSceneChangeBuffer(std::unique_ptr<RGYCLBuf> &&buf);
    RGY_ERR submitSceneChangePlane(PendingSceneChangePlane *pending,
        const RGYFrameInfo *prev2, const RGYFrameInfo *prev, const RGYFrameInfo *cur, const RGYFrameInfo *next, const RGYFrameInfo *next2,
        RGY_PLANE plane, const TCHAR *planeName, int smoothRadius, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR resolveSceneChangePlane(PendingSceneChangePlane *pending, RGYOpenCLQueue &queue);
    RGY_ERR submitPendingSearchPrefilterFrame(int currentFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR resolvePendingSearchPrefilterFrame(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, RGYOpenCLEvent *event);
    void clearPendingSearchPrefilterFrames();
    std::array<int, 4> sceneChangeFlagsForPlane(const PendingSearchPrefilterFrame &pending, RGY_PLANE plane) const;
    RGY_ERR emitPrefilteredFrame(PendingSearchPrefilterFrame &pending, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR initSearchLumaDump(const RGYFrameInfo &frameInfo, const RGYFilterParamRtgmcSearchPrefilter &prm);
    RGY_ERR dumpSearchLumaFrame(RGYCLFrame *searchLuma, const RGYFrameInfo &sourceFrame,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR dumpSearchYuvFrame(const RGYFrameInfo &yFrame, const RGYFrameInfo *chromaFrame,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    const RGYFrameInfo *resolveCacheFrame(int frameIndex) const;
    std::shared_ptr<RGYCLFrame> resolveCacheFrameShared(int frameIndex) const;
    int cacheIndex(int frame) const;
    int outputDelay() const;
    int drainFrameCount() const;

    std::array<std::shared_ptr<RGYCLFrame>, RTGMC_SEARCH_PREFILTER_CACHE_SIZE> m_cacheFrames;
    std::deque<std::unique_ptr<RGYCLBuf>> m_sceneChangeBufferPool;
    std::deque<PendingSearchPrefilterFrame> m_pendingSearchPrefilterFrames;
    std::array<SearchRefine1PlaneResources, 2> m_searchRefine1PlaneResources;
    std::array<SearchRefine2PlaneResources, 2> m_searchRefine2PlaneResources;
    std::array<std::unique_ptr<RGYFilterResizePlaneProxy>, 2> m_searchRefine1ResizeDown;
    std::array<std::unique_ptr<RGYFilterResizePlaneProxy>, 2> m_searchRefine1ResizeUp;
    std::array<std::unique_ptr<RGYFilterResizePlaneProxy>, 2> m_searchRefine2ResizeEdgeSoftenedSearch;
    std::shared_ptr<SharedFramePool> m_cacheFramePool;
    std::shared_ptr<SharedFramePool> m_searchLumaPool;
    RGYOpenCLProgramAsync m_prefilter;
    std::string m_buildOptions;
    std::ofstream m_searchLumaDump;
    std::string m_searchLumaDumpPath;
    std::string m_searchLumaDumpStage;
    int m_searchLumaDumpMaxFrames;
    int m_searchLumaDumpFrameCount;
    bool m_searchLumaDumpEnabled;
    bool m_searchLumaDumpHeaderWritten;
    int m_inputCount;
    int m_drainCount;
};
