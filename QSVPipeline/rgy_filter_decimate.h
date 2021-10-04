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
#include "rgy_filter.h"
#include "rgy_prm.h"

class RGYFilterParamDecimate : public RGYFilterParam {
public:
    VppDecimate decimate;
    bool useSeparateQueue;
    tstring outfilename;

    RGYFilterParamDecimate() : decimate(), useSeparateQueue(true), outfilename() {};
    virtual ~RGYFilterParamDecimate() {};
    virtual tstring print() const override;
};

class RGYFilterDecimateFrameData {
public:
    RGYFilterDecimateFrameData(std::shared_ptr<RGYOpenCLContext> context, std::shared_ptr<RGYLog> log);
    ~RGYFilterDecimateFrameData();

    RGYCLFrame *get() { return m_buf.get(); }
    const RGYCLFrame *get() const { return m_buf.get(); }
    std::unique_ptr<RGYCLBuf>& tmp() { return m_tmp; }
    RGY_ERR set(const RGYFrameInfo *pInputFrame, int inputFrameId, int blockSizeX, int blockSizeY, RGYOpenCLQueue& queue, RGYOpenCLEvent& event);
    int id() const { return m_inFrameId; }
    void calcDiffFromTmp();

    int64_t diffMaxBlock() const { return m_diffMaxBlock; }
    int64_t diffTotal() const { return m_diffTotal; }
private:
    std::shared_ptr<RGYOpenCLContext> m_cl;
    std::shared_ptr<RGYLog> m_log;
    int m_inFrameId;
    int m_blockX;
    int m_blockY;
    std::unique_ptr<RGYCLFrame> m_buf;
    std::unique_ptr<RGYCLBuf> m_tmp;
    int64_t m_diffMaxBlock;
    int64_t m_diffTotal;
};


class RGYFilterDecimateCache {
public:
    RGYFilterDecimateCache(shared_ptr<RGYOpenCLContext> context);
    ~RGYFilterDecimateCache();
    void init(int bufCount, int blockX, int blockY, std::shared_ptr<RGYLog> log);
    RGY_ERR add(const RGYFrameInfo *pInputFrame, RGYOpenCLQueue& queue, RGYOpenCLEvent& event);
    RGYFilterDecimateFrameData *frame(int iframe) {
        iframe = clamp(iframe, 0, m_inputFrames - 1);
        return m_frames[iframe % m_frames.size()].get();
    }
    RGYCLFrame *get(int iframe) {
        return frame(iframe)->get();
    }
    int inframe() const { return m_inputFrames; }
private:
    shared_ptr<RGYOpenCLContext> m_cl;
    std::shared_ptr<RGYLog> m_log;
    int m_blockX;
    int m_blockY;
    int m_inputFrames;
    std::vector<std::unique_ptr<RGYFilterDecimateFrameData>> m_frames;
};

class RGYFilterDecimate : public RGYFilter {
public:
    RGYFilterDecimate(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterDecimate();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo* pInputFrame, RGYFrameInfo** ppOutputFrames, int* pOutputFrameNum, RGYOpenCLQueue& queue_main, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent* event) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamDecimate> pParam);
    RGY_ERR setOutputFrame(int64_t nextTimestamp, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum);

    RGY_ERR calcDiff(RGYFilterDecimateFrameData *current, const RGYFilterDecimateFrameData *prev, RGYOpenCLQueue& queue_main);
    RGY_ERR procPlane(const bool useKernel2, const bool firstPlane, const RGYFrameInfo *p0, const RGYFrameInfo *p1, std::unique_ptr<RGYCLBuf>& tmp, const int blockHalfX, const int blockHalfY,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR procFrame(const RGYFrameInfo *p0, const RGYFrameInfo *p1, std::unique_ptr<RGYCLBuf>& tmp,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    bool m_flushed;
    int m_frameLastDropped;
    int64_t m_threSceneChange;
    int64_t m_threDuplicate;
    RGYOpenCLProgramAsync m_decimate;
    RGYFilterDecimateCache m_cache;
    RGYOpenCLEvent m_eventDiff;
    RGYOpenCLEvent m_eventTransfer;
    RGYOpenCLQueue m_streamDiff;
    RGYOpenCLQueue m_streamTransfer;
    unique_ptr<FILE, fp_deleter> m_fpLog;
};
