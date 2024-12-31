// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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
// --------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_PARALLEL_ENC_H__
#define __RGY_PARALLEL_ENC_H__

#include <thread>
#include "rgy_osdep.h"
#include "rgy_err.h"
#include "rgy_event.h"
#include "rgy_log.h"
#include "rgy_queue.h"

struct RGYOutputRawPEExtHeader;
class RGYInput;
#if ENCODER_QSV
struct sInputParams;
using encParams = sInputParams;

class CQSVPipeline;

#elif ENCODER_NVENC
#elif ENCODER_VCEENC
#elif ENCODER_RKMPP
#endif

struct RGYParallelEncSendData {
    unique_event eventChildHasSentFirstKeyPts;
    unique_event eventParentHasSentFinKeyPts;
    int64_t videoFirstKeyPts;
    int64_t videoFinKeyPts;
    RGYQueueMPMP<RGYOutputRawPEExtHeader*> *qFirstProcessData;

    RGYParallelEncSendData() :
        eventChildHasSentFirstKeyPts(unique_event(nullptr, nullptr)),
        eventParentHasSentFinKeyPts(unique_event(nullptr, nullptr)),
        videoFirstKeyPts(-1),
        videoFinKeyPts(-1),
        qFirstProcessData(nullptr) {};
};;

struct RGYParallelEncProcessData {
    tstring tmppath;
    int64_t ptsOffset;
};

class RGYParallelEncProcess {
public:
    RGYParallelEncProcess(const int id, const tstring& tmpfile, std::shared_ptr<RGYLog> log);
    ~RGYParallelEncProcess();
    RGY_ERR run(const encParams& peParams);
    int id() const { return m_id; }
    int64_t getVideoFirstKeyPts(const int timeout);
    RGY_ERR sendEndPts(const int64_t endPts);
    RGY_ERR close();
    RGYParallelEncProcessData tmpfile() const { return { m_tmpfile, m_sendData.videoFirstKeyPts }; }
    RGY_ERR getNextPacket(RGYOutputRawPEExtHeader **ptr);
    RGY_ERR pushPacket(RGYOutputRawPEExtHeader *ptr);
protected:
    void AddMessage(RGYLogLevel log_level, const tstring &str) {
        if (m_log == nullptr || log_level < m_log->getLogLevel(RGY_LOGT_APP)) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto &line : lines) {
            if (line[0] != _T('\0')) {
                m_log->write(log_level, RGY_LOGT_APP, strsprintf(_T("PE%d: %s\n"), m_id, line.c_str()).c_str());
            }
        }
    }
    void AddMessage(RGYLogLevel log_level, const TCHAR *format, ...) {
        if (m_log == nullptr || log_level < m_log->getLogLevel(RGY_LOGT_APP)) {
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
    int m_id;
    std::unique_ptr<CQSVPipeline> m_process;
    std::unique_ptr<RGYQueueMPMP<RGYOutputRawPEExtHeader*>> m_qFirstProcessData;
    RGYParallelEncSendData m_sendData;
    tstring m_tmpfile;
    std::thread m_thRunProcess;
    bool m_thAbort;
    std::shared_ptr<RGYLog> m_log;
};

class RGYParallelEnc {
public:
    RGYParallelEnc(std::shared_ptr<RGYLog> log);
    virtual ~RGYParallelEnc();
    bool isParallelEncPossible(const encParams *prm, const RGYInput *input) const;
    RGY_ERR parallelRun(const encParams *prm, const RGYInput *input);
    void close();
    int64_t getVideofirstKeyPts(const int processID);
    int64_t getVideoEndKeyPts() const { return m_videoEndKeyPts; }
    void setVideoFinished() { m_videoFinished = true; }
    bool videoFinished() const { return m_videoFinished; }
    int id() const { return m_id; }
    size_t parallelCount() const { return m_encProcess.size(); }
    std::vector<RGYParallelEncProcessData> peRawFilePaths() const;
    RGY_ERR getNextPacketFromFirst(RGYOutputRawPEExtHeader **ptr);
    RGY_ERR pushNextPacket(RGYOutputRawPEExtHeader *ptr);
protected:
    encParams genPEParam(const int ip, const encParams *prm, const tstring& tmpfile);
    RGY_ERR parallelChild(const encParams *prm, const RGYInput *input);

    void AddMessage(RGYLogLevel log_level, const tstring &str) {
        if (m_log == nullptr || log_level < m_log->getLogLevel(RGY_LOGT_APP)) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto &line : lines) {
            if (line[0] != _T('\0')) {
                m_log->write(log_level, RGY_LOGT_APP, (_T("replace: ") + line + _T("\n")).c_str());
            }
        }
    }
    void AddMessage(RGYLogLevel log_level, const TCHAR *format, ...) {
        if (m_log == nullptr || log_level < m_log->getLogLevel(RGY_LOGT_APP)) {
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

    int m_id;
    std::vector<std::unique_ptr<RGYParallelEncProcess>> m_encProcess;
    std::shared_ptr<RGYLog> m_log;
    int64_t m_videoEndKeyPts;
    bool m_videoFinished;
};


#endif //__RGY_PARALLEL_ENC_H__
