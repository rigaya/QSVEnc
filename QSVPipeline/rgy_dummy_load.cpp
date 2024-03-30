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

#include <thread>
#include <chrono>
#include <random>
#include "rgy_dummy_load.h"
#include "rgy_thread_affinity.h"
#include "rgy_filter_cl.h"

RGYDummyLoadCL::RGYDummyLoadCL(std::shared_ptr<RGYOpenCLContext> cl) :
    m_cl(cl),
    m_clQueue(),
    m_prog(),
    m_thread(),
    m_event(CreateEventUnique(nullptr, false, false)),
    m_log(),
    m_abort(false),
    m_bufElemSize(0),
    m_clBuf() {
}

RGYDummyLoadCL::~RGYDummyLoadCL() {
    close();
}

std::pair<RGY_ERR, double> RGYDummyLoadCL::runKernel(const int count, const int innerLoop, const float valA, const float valB) {
    const char *kernel_name = "kernel_dummy_load";
    RGYWorkSize local(256);
    RGYWorkSize global(m_bufElemSize);
    if (m_bufElemSize <= 0) {
        return { RGY_ERR_NOT_INITIALIZED, 0.0 };
    }

    RGYOpenCLEvent queueRunStart;
    m_clQueue.getmarker(queueRunStart);
    for (int i = 0; i < count; i++) {
        auto err = m_prog.get()->kernel(kernel_name).config(m_clQueue, local, global).launch(
            m_clBuf->mem(), m_bufElemSize, innerLoop, valA, valB);
        if (err != RGY_ERR_NONE) {
            m_log->write(RGY_LOG_ERROR, RGY_LOGT_PERF_MONITOR, _T("error at %s: %s.\n"),
                char_to_tstring(kernel_name).c_str(), get_err_mes(err));
            return { err, 0.0 };
        }
    }
    RGYOpenCLEvent queueRunEnd;
    m_clQueue.getmarker(queueRunEnd);
    queueRunEnd.wait();
    auto perf = std::make_unique<RGYFilterPerfCL>();
    perf->checkPerformace(&queueRunStart, &queueRunEnd);
    return { RGY_ERR_NONE, perf->GetAvgTimeElapsed() };
}

RGY_ERR RGYDummyLoadCL::run(const float targetLoadPercent, std::shared_ptr<RGYLog> log) {
    m_log = log;
    m_log->write(RGY_LOG_DEBUG, RGY_LOGT_PERF_MONITOR, _T("RGYDummyLoadCL::run() started.\n"));
    m_clQueue = m_cl->createQueue(m_cl->queue().devid(), CL_QUEUE_PROFILING_ENABLE);
    if (!m_clQueue.get()) {
        m_log->write(RGY_LOG_ERROR, RGY_LOGT_PERF_MONITOR, _T("Failed to create OpenCL queue for dummy load.\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    m_bufElemSize = 32 * 1024;
    m_clBuf = m_cl->createBuffer(m_bufElemSize * sizeof(uint8_t), CL_MEM_READ_WRITE);

    m_thread = std::thread([&](float targetLoadPercent) {
#if defined(_WIN32) || defined(_WIN64)
        RGYParamThread param;
        param.throttling = RGYThreadPowerThrottlingMode::Enabled;
        param.apply(GetCurrentThread());
#endif //#if defined(_WIN32) || defined(_WIN64)

        //最初のうちは高速なので、5秒ほど待つ
        std::this_thread::sleep_for(std::chrono::seconds(5));

        m_prog.set(m_cl->buildResourceAsync(_T("RGY_DUMMY_LOAD_CL"), _T("EXE_DATA"), ""));

        {
            uint8_t val = 128;
            clEnqueueFillBuffer(m_clQueue.get(), m_clBuf->mem(), &val, sizeof(val), 0, m_clBuf->size(), 0, nullptr, nullptr);
        }
        m_clQueue.finish();

        if (!m_prog.get()) {
            m_log->write(RGY_LOG_ERROR, RGY_LOGT_PERF_MONITOR, _T("failed to load RGY_DUMMY_LOAD_CL(m_pmd)\n"));
            return RGY_ERR_OPENCL_CRUSH;
        }

        std::random_device seed_gen;
        std::default_random_engine engine(seed_gen());

        std::uniform_real_distribution<> dist(0.0, 1.0);

        const int minSleepMs = 10; // 最小sleep
        int nextSleepMs = 500; // 次sleepする量
        const int runIntervalMs = 250; // 目標実行間隔
        const double targetRuntimeMs = runIntervalMs * clamp(targetLoadPercent * 0.01, 1e-9, 1.0);
        const int runKernelCount = 1;
        int innerLoop = 100;
        while (WaitForSingleObject(m_event.get(), nextSleepMs) == WAIT_TIMEOUT) {
            if (m_abort) break;
            const auto [ sts, timeMs ] = runKernel(runKernelCount, innerLoop, (float)dist(engine), (float)dist(engine));
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            int oldInnerLoop = innerLoop;
            if (timeMs > 0.0) {
                innerLoop = std::max((int)(innerLoop * targetRuntimeMs / timeMs), 1);
            } else {
                innerLoop += 100;
            }
            m_log->write(RGY_LOG_TRACE, RGY_LOGT_PERF_MONITOR, _T("oldInnerLoop = %d, timeMs = %.3f -> innerLoop = %d.\n"), oldInnerLoop, timeMs, innerLoop);
            nextSleepMs = std::max(runIntervalMs - (int)timeMs, minSleepMs);
        }
        m_log->write(RGY_LOG_DEBUG, RGY_LOGT_PERF_MONITOR, _T("RGYDummyLoadCL::run() finished.\n"));
        return RGY_ERR_NONE;
    }, targetLoadPercent);
    return RGY_ERR_NONE;
}

void RGYDummyLoadCL::close() {
    m_abort = true;
    if (m_thread.joinable()) {
        SetEvent(m_event.get());
        m_thread.join();
    }
    m_clBuf.reset();
    m_event.reset();
    m_clQueue.clear();
}
