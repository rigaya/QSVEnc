// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
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

#include "rgy_parallel_enc.h"
#include "rgy_filesystem.h"
#include "rgy_input.h"
#if ENCODER_QSV
#include "qsv_cmd.h"
#elif ENCODER_NVENC
#include "nvenc_cmd.h"
#elif ENCODER_VCEENC
#include "vce_cmd.h"
#elif ENCODER_RKMPP
#include "rkmppenc_cmd.h"
#endif

static const TCHAR *RGYParallelEncProcessFirstPtsKey = _T("RGYParallelEncProcessFirstPtsKey");

RGYParallelEncProcess::RGYParallelEncProcess(const int id, std::shared_ptr<RGYLog> log) :
    m_id(id),
    m_process(nullptr),
    m_eventGotVideoFirstKeyPts(unique_event(nullptr, nullptr)),
    m_thRecvStderr(),
    m_videoFirstKeyPts(-1),
    m_log(log) {
}

RGYParallelEncProcess::~RGYParallelEncProcess() {
    close();
}

RGY_ERR RGYParallelEncProcess::close() {
    auto err = RGY_ERR_NONE;
    if (m_process) {
        err = (RGY_ERR)m_process->waitAndGetExitCode();
        m_process.reset();
    }
    return err;
}

RGY_ERR RGYParallelEncProcess::run(const std::vector<tstring>& cmd) {
    m_process = std::make_unique<RGYPipeProcess>();
    m_process->init(PIPE_MODE_ENABLE | PIPE_MODE_ENABLE_FP, PIPE_MODE_DISABLE, PIPE_MODE_ENABLE | PIPE_MODE_ENABLE_FP);
    if (m_process->run(cmd, nullptr, 0, true, true) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to run encoder %d.\n"), m_id);
        return RGY_ERR_UNKNOWN;
    }
    m_thRecvStderr = std::thread([&]() {
        recvStdErr();
    });
    return RGY_ERR_NONE;
}

RGY_ERR RGYParallelEncProcess::recvStdErr() {
    AddMessage(RGY_LOG_DEBUG, _T("PE%d: Start thread to receive messages from encoder.\n"), m_id);
    std::vector<uint8_t> buffer;
    while (m_process->stdErrRead(buffer) >= 0) {
        if (buffer.size() > 0) {
            auto str = std::string(buffer.data(), buffer.data() + buffer.size());
            auto tstr = char_to_tstring(str);
            if (tstr.find(RGYParallelEncProcessFirstPtsKey) == 0) {
                try {
                    m_videoFirstKeyPts = std::stoll(tstr.substr(_tcslen(RGYParallelEncProcessFirstPtsKey) + 1));
                } catch (...) {
                    m_videoFirstKeyPts = -1;
                }
                AddMessage(RGY_LOG_DEBUG, _T("PE%d: Got first key pts: %lld.\n"), m_id, m_videoFirstKeyPts);
            }
            m_log->write(RGY_LOG_INFO, RGY_LOGT_APP, _T("%s"), tstr.c_str());
            buffer.clear();
        }
    }
    m_process->stdErrRead(buffer);
    if (buffer.size() > 0) {
        auto str = std::string(buffer.data(), buffer.data() + buffer.size());
        m_log->write(RGY_LOG_INFO, RGY_LOGT_APP, _T("%s"), char_to_tstring(str).c_str());
        buffer.clear();
    }
    AddMessage(RGY_LOG_DEBUG, _T("PE%d: Reached encoder stderr EOF.\n"), m_id);
    return RGY_ERR_NONE;
}

int64_t RGYParallelEncProcess::getVideofirstKeyPts(const int timeout) {
    if (!m_eventGotVideoFirstKeyPts) {
        return -1;
    }
    if (WaitForSingleObject(m_eventGotVideoFirstKeyPts.get(), timeout) == WAIT_OBJECT_0) {
        return m_videoFirstKeyPts;
    }
    return -1;
}

RGY_ERR RGYParallelEncProcess::sendEndPts(const int64_t endPts) {
    if (m_process->processAlive()) {
        m_process->stdInFpWrite(&endPts, sizeof(endPts));
    }
}

RGYParallelEncReadPacket::RGYParallelEncReadPacket() :
    m_fp(nullptr, fclose) {
}

RGYParallelEncReadPacket::~RGYParallelEncReadPacket() {
    m_fp.reset();
}

RGY_ERR RGYParallelEncReadPacket::init(const tstring &filename) {
    if (filename.length() > 0) {
        FILE *fp = nullptr;
        if (_tfopen_s(&fp, filename.c_str(), _T("rb"))) {
            return RGY_ERR_FILE_OPEN;
        }
        m_fp = std::unique_ptr<FILE, decltype(&fclose)>(fp, fclose);
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYParallelEncReadPacket::getSample(int64_t& pts, int& key, std::vector<uint8_t>& buffer) {
    int64_t size = 0;
    pts = -1;
    key = 0;
    if (m_fp) {
        if (fread(&pts, sizeof(pts), 1, m_fp.get()) != 1) {
            return RGY_ERR_MORE_DATA;
        }
        if (fread(&key, sizeof(key), 1, m_fp.get()) != 1) {
            return RGY_ERR_MORE_DATA;
        }
        if (fread(&size, sizeof(size), 1, m_fp.get()) != 1) {
            return RGY_ERR_MORE_DATA;
        }
        buffer.resize(size);
        if (fread(buffer.data(), 1, size, m_fp.get()) != size) {
            return RGY_ERR_MORE_DATA;
        }
    }
    return RGY_ERR_NONE;
}

RGYParallelEnc::RGYParallelEnc(std::shared_ptr<RGYLog> log) :
    m_encProcess(),
    m_log(log) {}

RGYParallelEnc::~RGYParallelEnc() {}

bool RGYParallelEnc::isParallelEncPossible(const RGYInput *input) const {
    return input->seekable() && input->GetVideoFirstKeyPts() >= 0;
}

RGY_ERR RGYParallelEnc::parallelChild(const sInputParams *prm, const RGYInput *input) {
    const auto firstKeyPts = input->GetVideoFirstKeyPts();
    AddMessage(RGY_LOG_ERROR, _T("%s: %lld\n"), RGYParallelEncProcessFirstPtsKey, input->GetVideoFirstKeyPts());
    if (firstKeyPts < 0) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to get first key pts from encoder.\n"));
        return RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

std::vector<tstring> RGYParallelEnc::genCmd(const int ip, const sInputParams *prm) {
    std::vector<tstring> args;
    if (ip <= 0) {
        return args;
    }
    sInputParams prmParallel = *prm;
    prmParallel.ctrl.parallelEnc.multiProcessId = ip;
    prmParallel.ctrl.parentProcessID = GetCurrentProcessId();
    prmParallel.ctrl.loglevel = RGYParamLogLevel(RGY_LOG_ERROR);
    auto cmdt = gen_cmd(&prmParallel, false);
    args = splitCommandLine(cmdt.c_str());
    if (args.size() == 0) {
        return args;
    }
    args[0] = getExePath();
    return args;
}

RGY_ERR RGYParallelEnc::parallelRun(const sInputParams *prm, const RGYInput *input) {
    if (prm->ctrl.parallelEnc.multiProcess <= 1) {
        return RGY_ERR_NONE;
    }
    if (!isParallelEncPossible(input)) {
        AddMessage(RGY_LOG_ERROR, _T("Parallel encoding is not possible.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->ctrl.parentProcessID > 0) { // 子プロセス
        return parallelChild(prm, input);
    }
    for (int ip = 1; ip < prm->ctrl.parallelEnc.multiProcess; ip++) {
        const auto cmd = genCmd(ip, prm);
        auto process = std::make_unique<RGYParallelEncProcess>(ip, m_log);
        if (process->run(cmd) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to run encoder.\n"));
            return RGY_ERR_UNKNOWN;
        }
        const auto firstKeyPts = process->getVideofirstKeyPts(10 * 1000);
        if (firstKeyPts < 0) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get first key pts from encoder.\n"));
            return RGY_ERR_UNKNOWN;
        }
        m_encProcess.push_back(std::move(process));
    }
    return RGY_ERR_NONE;
}