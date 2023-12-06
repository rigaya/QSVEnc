// -----------------------------------------------------------------------------------------
//     QSVEnc/VCEEnc/rkmppenc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2019-2021 rigaya
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
// IABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include "rgy_util.h"
#include "rgy_log.h"
#include "rgy_opencl.h"
#include "convert_csp.h"
#include "rgy_prm.h"

struct AVPacket;

class RGYFilterParam {
public:
    RGYFrameInfo frameIn;
    RGYFrameInfo frameOut;
    rgy_rational<int> baseFps;
    bool bOutOverwrite;

    RGYFilterParam() : frameIn(), frameOut(), baseFps(), bOutOverwrite(false) {};
    virtual ~RGYFilterParam() {};
    virtual tstring print() const { return _T(""); };
};


enum FILTER_PATHTHROUGH_FRAMEINFO : uint32_t {
    FILTER_PATHTHROUGH_NONE      = 0x00u,
    FILTER_PATHTHROUGH_TIMESTAMP = 0x01u,
    FILTER_PATHTHROUGH_FLAGS     = 0x02u,
    FILTER_PATHTHROUGH_PICSTRUCT = 0x04u,
    FILTER_PATHTHROUGH_DATA      = 0x07u,

    FILTER_PATHTHROUGH_ALL       = 0x0fu,
};

static FILTER_PATHTHROUGH_FRAMEINFO operator|(FILTER_PATHTHROUGH_FRAMEINFO a, FILTER_PATHTHROUGH_FRAMEINFO b) {
    return (FILTER_PATHTHROUGH_FRAMEINFO)((uint32_t)a | (uint32_t)b);
}

static FILTER_PATHTHROUGH_FRAMEINFO operator|=(FILTER_PATHTHROUGH_FRAMEINFO &a, FILTER_PATHTHROUGH_FRAMEINFO b) {
    a = a | b;
    return a;
}

static FILTER_PATHTHROUGH_FRAMEINFO operator&(FILTER_PATHTHROUGH_FRAMEINFO a, FILTER_PATHTHROUGH_FRAMEINFO b) {
    return (FILTER_PATHTHROUGH_FRAMEINFO)((uint32_t)a & (uint32_t)b);
}

static FILTER_PATHTHROUGH_FRAMEINFO operator&=(FILTER_PATHTHROUGH_FRAMEINFO &a, FILTER_PATHTHROUGH_FRAMEINFO b) {
    a = a & b;
    return a;
}

static FILTER_PATHTHROUGH_FRAMEINFO operator~(FILTER_PATHTHROUGH_FRAMEINFO a) {
    return (FILTER_PATHTHROUGH_FRAMEINFO)(~((uint32_t)a));
}

class RGYFilterPerf {
public:
    RGYFilterPerf() : m_checkPerformance(false), m_filterTimeMs(0.0), m_runCount(0) {};
    ~RGYFilterPerf() { };

    bool checkPerformanceEnabled() const { return m_checkPerformance; }
    void setCheckPerformance(const bool check) { m_checkPerformance = check; }
    RGY_ERR checkPerformace(RGYOpenCLEvent *event_start, RGYOpenCLEvent *event_fin) {
        uint64_t time_start = 0;
        auto sts = event_start->getProfilingTimeEnd(time_start);
        if (sts != RGY_ERR_NONE) return sts;
        uint64_t time_end = 0;
        sts = event_fin->getProfilingTimeStart(time_end);
        if (sts != RGY_ERR_NONE) return sts;
        m_runCount++;
        m_filterTimeMs += (time_end - time_start) * 1e-6 /*ns -> ms*/;
        return RGY_ERR_NONE;
    }
    double GetAvgTimeElapsed() const {
        return (m_runCount > 0) ? m_filterTimeMs / (double)m_runCount : 0.0;
    }
protected:
    bool m_checkPerformance;
    double m_filterTimeMs;
    int m_runCount;
};

class RGYFilter {
public:
    RGYFilter(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilter();
    tstring name() {
        return m_name;
    }
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> param, shared_ptr<RGYLog> pPrintMes) = 0;
    virtual RGY_ERR addStreamPacket(AVPacket *pkt) { UNREFERENCED_PARAMETER(pkt); return RGY_ERR_UNSUPPORTED; };
    RGY_ERR filter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum);
    RGY_ERR filter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue);
    RGY_ERR filter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, RGYOpenCLEvent *event);
    RGY_ERR filter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event = nullptr);
    const tstring GetInputMessage() const {
        return m_infoStr;
    }
    const RGYFilterParam *GetFilterParam() const {
        return m_param.get();
    }
    virtual RGY_ERR AllocFrameBuf(const RGYFrameInfo &frame, int frames);
    //virtual RGY_ERR addStreamPacket(AVPacket *pkt) { UNREFERENCED_PARAMETER(pkt); return RGY_ERR_UNSUPPORTED; };
    virtual int targetTrackIdx() { return 0; };
    void setCheckPerformance(const bool check) { m_perfMonitor.setCheckPerformance(check); }
    double GetAvgTimeElapsed() { return m_perfMonitor.GetAvgTimeElapsed(); }
protected:
    RGY_ERR filter_as_interlaced_pair(const RGYFrameInfo *pInputFrame, RGYFrameInfo *pOutputFrame);
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) = 0;
    virtual void close() = 0;

    void AddMessage(RGYLogLevel log_level, const tstring &str) {
        if (m_pLog == nullptr || log_level < m_pLog->getLogLevel(RGY_LOGT_VPP)) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto &line : lines) {
            if (line[0] != _T('\0')) {
                m_pLog->write(log_level, RGY_LOGT_VPP, (m_name + _T(": ") + line + _T("\n")).c_str());
            }
        }
    }
    void AddMessage(RGYLogLevel log_level, const TCHAR *format, ...) {
        if (m_pLog == nullptr || log_level < m_pLog->getLogLevel(RGY_LOGT_VPP)) {
            return;
        }

        va_list args;
        va_start(args, format);
        int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
        tstring buffer;
        buffer.resize(len, _T('\0'));
        _vstprintf_s(&buffer[0], len, format, args);
        va_end(args);
        AddMessage(log_level, buffer);
    }
    void setFilterInfo(const tstring &info) {
        m_infoStr = info;
        AddMessage(RGY_LOG_DEBUG, info);
    }

    tstring m_name;
    tstring m_infoStr;
    shared_ptr<RGYLog> m_pLog;  //ログ出力
    shared_ptr<RGYOpenCLContext> m_cl;
    vector<unique_ptr<RGYCLFrame>> m_frameBuf;
    unique_ptr<RGYCLFrame> m_pFieldPairIn;
    unique_ptr<RGYCLFrame> m_pFieldPairOut;
    shared_ptr<RGYFilterParam> m_param;
    FILTER_PATHTHROUGH_FRAMEINFO m_pathThrough;
    RGYFilterPerf m_perfMonitor;
};

class RGYFilterParamCrop : public RGYFilterParam {
public:
    sInputCrop crop;
    CspMatrix matrix;

    RGYFilterParamCrop() : crop(initCrop()), matrix(RGY_MATRIX_ST170_M) {};
    virtual ~RGYFilterParamCrop() {};
};

class RGYFilterCspCrop : public RGYFilter {
public:
    RGYFilterCspCrop(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterCspCrop();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    RGY_ERR convertYBitDepth(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR convertCspFromNV12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR convertCspFromYV12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR convertCspFromNV16(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR convertCspFromRGB(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR convertCspFromYUV444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    RGY_ERR convertCspFromAYUVPacked444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    virtual void close() override;
};

class RGYFilterParamResize : public RGYFilterParam {
public:
    RGY_VPP_RESIZE_ALGO interp;
    RGYFilterParamResize() : interp(RGY_VPP_RESIZE_AUTO) {};
    virtual ~RGYFilterParamResize() {};
};

class RGYFilterResize : public RGYFilter {
public:
    RGYFilterResize(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterResize();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    virtual RGY_ERR resizePlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    virtual RGY_ERR resizeFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    bool m_bInterlacedWarn;
    unique_ptr<RGYCLBuf> m_weightSpline;
    RGYOpenCLProgramAsync m_resize;
    RGYCLFramePool m_srcImagePool;
};

class RGYFilterParamPad : public RGYFilterParam {
public:
    VppPad pad;
    RGY_CSP encoderCsp;
    RGYFilterParamPad() : pad(), encoderCsp(RGY_CSP_NA) {};
    virtual ~RGYFilterParamPad() {};
    virtual tstring print() const override;
};

class RGYFilterPad : public RGYFilter {
public:
    RGYFilterPad(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterPad();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    virtual RGY_ERR procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, int pad_color, const VppPad &pad, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    virtual RGY_ERR procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    RGYOpenCLProgramAsync m_pad;
    bool m_bInterlacedWarn;
};
