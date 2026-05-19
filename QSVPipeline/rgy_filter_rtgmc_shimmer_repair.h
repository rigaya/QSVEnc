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

#include "rgy_filter_cl.h"
#include "rgy_filter_rtgmc_repair_profile.h"

#include <fstream>
#include <vector>

enum class RGYRtgmcShimmerRepairStage {
    PreRetouch,
    PostTR2,
};

class RGYFilterParamRtgmcShimmerRepair : public RGYFilterParam {
public:
    RGYRtgmcShimmerRepairStage stage;
    int repairThin;
    int repairPad;
    bool processChroma;
    RGYRtgmcRepairProfile repairProfile;

    RGYFilterParamRtgmcShimmerRepair() : stage(RGYRtgmcShimmerRepairStage::PreRetouch), repairThin(0), repairPad(0), processChroma(true), repairProfile() {}
    virtual ~RGYFilterParamRtgmcShimmerRepair() {}
    virtual tstring print() const override;
};

class RGYFilterRtgmcShimmerRepair : public RGYFilter {
public:
    RGYFilterRtgmcShimmerRepair(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterRtgmcShimmerRepair();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
    RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamRtgmcShimmerRepair> &prm);
    RGY_ERR buildKernels(const std::shared_ptr<RGYFilterParamRtgmcShimmerRepair> &prm);
    RGY_ERR processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const RGYFrameInfo *pRefFrame,
        const RGYFilterParamRtgmcShimmerRepair &prm,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

private:
    RGY_ERR launchRtgmcShimmerRepairFused(
        RGYFrameInfo *pOutputFrame,
        RGYFrameInfo *pCorrectionDeltaFrame,
        RGYFrameInfo *pPositiveCorrectionGateFrame,
        RGYFrameInfo *pNegativeCorrectionGateFrame,
        const RGYFrameInfo *pInputFrame,
        const RGYFrameInfo *pRefFrame,
        const RGYFilterParamRtgmcShimmerRepair &prm,
        int iplane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR launchRtgmcShimmerRepairApply(
        RGYFrameInfo *pOutputFrame,
        const RGYFrameInfo *pInputFrame,
        const RGYFrameInfo *pRefFrame,
        const RGYFilterParamRtgmcShimmerRepair &prm,
        int iplane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

protected:
    RGYOpenCLProgramAsync m_shimmerRepair;
    std::string m_buildOptions;
    std::ofstream m_lumaDump;
    std::string m_lumaDumpPath;
    std::string m_lumaDumpStage;
    std::string m_lumaDumpTarget;
    int m_lumaDumpMaxFrames;
    int m_lumaDumpFrameCount;
    bool m_lumaDumpEnabled;
    bool m_lumaDumpHeaderWritten;
    bool m_lumaDumpFullYuv;
    bool m_useKernel;

    RGY_ERR initLumaDump(const RGYFrameInfo &frameInfo, const RGYFilterParamRtgmcShimmerRepair &prm);
    RGY_ERR dumpLumaFrame(const RGYFrameInfo *frame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR dumpStageFrame(const char *stage, const RGYFrameInfo *frame, const char *target,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
};
