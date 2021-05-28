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

class RGYFilterParamMpdecimate : public RGYFilterParam {
public:
    VppMpdecimate mpdecimate;
    bool useSeparateQueue;
    tstring outfilename;

    RGYFilterParamMpdecimate() : mpdecimate(), useSeparateQueue(true), outfilename() {};
    virtual ~RGYFilterParamMpdecimate() {};
    virtual tstring print() const override;
};

class RGYFilterMpdecimateFrameData {
public:
    RGYFilterMpdecimateFrameData(shared_ptr<RGYOpenCLContext> context, std::shared_ptr<RGYLog> log);
    ~RGYFilterMpdecimateFrameData();

    RGYCLFrame *get() { return m_buf.get(); }
    RGYCLFrame *tmp() { return m_tmp.get(); }
    const RGYCLFrame *get() const { return m_buf.get(); }
    RGY_ERR set(const RGYFrameInfo *pInputFrame, int inputFrameId, RGYOpenCLQueue& queue, RGYOpenCLEvent& event);
    int id() const { return m_inFrameId; }
    void reset() { m_inFrameId = -1; }
    bool checkIfFrameCanbeDropped(const int hi, const int lo, const float factor);
private:
    shared_ptr<RGYOpenCLContext> m_cl;
    std::shared_ptr<RGYLog> m_log;
    int m_inFrameId;
    std::unique_ptr<RGYCLFrame> m_buf;
    std::unique_ptr<RGYCLFrame> m_tmp;
};

class RGYFilterMpdecimateCache {
public:
    RGYFilterMpdecimateCache(shared_ptr<RGYOpenCLContext> context);
    ~RGYFilterMpdecimateCache();
    void init(int bufCount, std::shared_ptr<RGYLog> log);
    RGY_ERR add(const RGYFrameInfo *pInputFrame, RGYOpenCLQueue& queue, RGYOpenCLEvent& event);
    void removeFromCache(int iframe) {
        for (auto &f : m_frames) {
            if (f->id() == iframe) {
                f->reset();
                return;
            }
        }
    }
    RGYFilterMpdecimateFrameData *frame(int iframe) {
        for (auto &f : m_frames) {
            if (f->id() == iframe) {
                return f.get();
            }
        }
        return nullptr;
    }
    RGYFilterMpdecimateFrameData *getEmpty() {
        for (auto &f : m_frames) {
            if (f->id() < 0) {
                return f.get();
            }
        }
        return nullptr;
    }
    RGYCLFrame *get(int iframe) {
        return frame(iframe)->get();
    }
    int inframe() const { return m_inputFrames; }
private:
    shared_ptr<RGYOpenCLContext> m_cl;
    std::shared_ptr<RGYLog> m_log;
    int m_inputFrames;
    std::vector<std::unique_ptr<RGYFilterMpdecimateFrameData>> m_frames;
};

class RGYFilterMpdecimate : public RGYFilter {
public:
    RGYFilterMpdecimate(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterMpdecimate();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo* pInputFrame, RGYFrameInfo** ppOutputFrames, int* pOutputFrameNum, RGYOpenCLQueue& queue_main, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent* event) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamMpdecimate> pParam);
    bool dropFrame(RGYFilterMpdecimateFrameData *targetFrame);

    RGY_ERR procPlane(const RGYFrameInfo *p0, const RGYFrameInfo *p1, RGYFrameInfo *tmp, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR procFrame(const RGYFrameInfo *p0, const RGYFrameInfo *p1, RGYFrameInfo *tmp, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR calcDiff(RGYFilterMpdecimateFrameData *target, const RGYFilterMpdecimateFrameData *ref, RGYOpenCLQueue& queue_main);

    int m_dropCount;
    int m_ref;
    int m_target;
    std::unique_ptr<RGYOpenCLProgram> m_mpdecimate;
    RGYFilterMpdecimateCache m_cache;
    RGYOpenCLEvent m_eventDiff;
    RGYOpenCLEvent m_eventTransfer;
    RGYOpenCLQueue m_streamDiff;
    RGYOpenCLQueue m_streamTransfer;
    unique_ptr<FILE, fp_deleter> m_fpLog;
};
