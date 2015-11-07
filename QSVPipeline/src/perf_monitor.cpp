//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <chrono>
#include <thread>
#include <memory>
#include <cstring>
#include <cstdio>
#include <ctime>
#include <string>
#include "perf_monitor.h"
#include "cpu_info.h"
#include "qsv_osdep.h"
#include "qsv_control.h"
#include "qsv_util.h"
#if defined(_WIN32) || defined(_WIN64)
#include <psapi.h>
#endif //#if defined(_WIN32) || defined(_WIN64)

CPerfMonitor::CPerfMonitor() {
    memset(m_info, 0, sizeof(m_info));

    cpu_info_t cpu_info;
    get_cpu_info(&cpu_info);
    m_nLogicalCPU = cpu_info.logical_cores;
}

CPerfMonitor::~CPerfMonitor() {
    clear();
}

void CPerfMonitor::clear() {
    if (m_thCheck.joinable()) {
        m_bAbort = true;
        m_thCheck.join();
    }
    memset(m_info, 0, sizeof(m_info));
    m_nStep = 0;
    m_thMainThread.reset();
    m_thEncThread = nullptr;
    m_bAbort = false;
    m_bEncStarted = false;
    if (m_fpLog) {
        fprintf(m_fpLog.get(), "\n\n");
    }
    m_fpLog.reset();
}

int CPerfMonitor::init(tstring filename,
    int interval, bool bUseMatplotLib, int nSelectOutputLog,
    std::unique_ptr<void, handle_deleter> thMainThread) {
    clear();

    m_nCreateTime100ns = clock() * (1e7 / CLOCKS_PER_SEC);
    m_sMonitorFilename = filename;
    m_nInterval = interval;
    m_bUseMatplotLib = bUseMatplotLib;
    m_nSelectOutputLog = nSelectOutputLog;
    m_thMainThread = std::move(thMainThread);

    if (!m_fpLog) {
        m_fpLog = std::unique_ptr<FILE, fp_deleter>(_tfopen(m_sMonitorFilename.c_str(), _T("a")));
        if (!m_fpLog) {
            return 1;
        }
    }

    //未実装
    m_nSelectOutputLog &= (~PERF_MONITOR_FRAME_IN);

    //未実装
#if !(defined(_WIN32) || defined(_WIN64))
    m_nSelectOutputLog &= (~PERF_MONITOR_CPU);
    m_nSelectOutputLog &= (~PERF_MONITOR_CPU_KERNEL);
    m_nSelectOutputLog &= (~PERF_MONITOR_THREAD_MAIN);
    m_nSelectOutputLog &= (~PERF_MONITOR_THREAD_ENC);
    m_nSelectOutputLog &= (~PERF_MONITOR_MEM_PRIVATE);
    m_nSelectOutputLog &= (~PERF_MONITOR_MEM_VIRTUAL);
    m_nSelectOutputLog &= (~PERF_MONITOR_IO_READ);
    m_nSelectOutputLog &= (~PERF_MONITOR_IO_WRITE);
#endif //#if defined(_WIN32) || defined(_WIN64)

    std::string str;
    if (m_nSelectOutputLog & PERF_MONITOR_CPU) {
        str += ",cpu (%)";
    }
    if (m_nSelectOutputLog & PERF_MONITOR_CPU_KERNEL) {
        str += ",cpu kernel (%)";
    }
    if (m_nSelectOutputLog & PERF_MONITOR_THREAD_MAIN) {
        str += ",cpu main thread (%)";
    }
    if (m_nSelectOutputLog & PERF_MONITOR_THREAD_ENC) {
        str += ",cpu enc thread (%)";
    }
    if (m_nSelectOutputLog & PERF_MONITOR_MEM_PRIVATE) {
        str += ",mem private (MB)";
    }
    if (m_nSelectOutputLog & PERF_MONITOR_MEM_VIRTUAL) {
        str += ",mem virtual (MB)";
    }
    if (m_nSelectOutputLog & PERF_MONITOR_FRAME_IN) {
        str += ",frame in";
    }
    if (m_nSelectOutputLog & PERF_MONITOR_FRAME_OUT) {
        str += ",frame out";
    }
    if (m_nSelectOutputLog & PERF_MONITOR_FPS) {
        str += ",enc speed (fps)";
    }
    if (m_nSelectOutputLog & PERF_MONITOR_FPS_AVG) {
        str += ",enc speed avg (fps)";
    }
    if (m_nSelectOutputLog & PERF_MONITOR_BITRATE) {
        str += ",bitrate (kbps)";
    }
    if (m_nSelectOutputLog & PERF_MONITOR_BITRATE_AVG) {
        str += ",bitrate avg (kbps)";
    }
    if (m_nSelectOutputLog & PERF_MONITOR_IO_READ) {
        str += ",read (MB/s)";
    }
    if (m_nSelectOutputLog & PERF_MONITOR_IO_WRITE) {
        str += ",write (MB/s)";
    }
    str += "\n";
    fwrite(str.c_str(), 1, str.length(), m_fpLog.get());

    m_thCheck = std::thread(loader, this);
    return 0;
}

void CPerfMonitor::SetEncStatus(std::shared_ptr<CEncodeStatusInfo> encStatus) {
    m_pEncStatus = encStatus;
    m_nOutputFPSScale = encStatus->m_nOutputFPSScale;
    m_nOutputFPSRate = encStatus->m_nOutputFPSRate;
}

void CPerfMonitor::SetEncThread(HANDLE thEncThread) {
    m_thEncThread = thEncThread;
}

void CPerfMonitor::check() {
    PerfInfo *pInfoNew = &m_info[(m_nStep + 1) & 1];
    PerfInfo *pInfoOld = &m_info[ m_nStep      & 1];
    memcpy(pInfoNew, pInfoOld, sizeof(pInfoNew[0]));

#if defined(_WIN32) || defined(_WIN64)
    const auto hProcess = GetCurrentProcess();
    auto getThreadTime = [](HANDLE hThread, PROCESS_TIME *time) {
        GetThreadTimes(hThread, (FILETIME *)&time->creation, (FILETIME *)&time->exit, (FILETIME *)&time->kernel, (FILETIME *)&time->user);
    };

    //メモリ情報
    PROCESS_MEMORY_COUNTERS mem_counters ={ 0 };
    mem_counters.cb = sizeof(PROCESS_MEMORY_COUNTERS);
    GetProcessMemoryInfo(hProcess, &mem_counters, sizeof(mem_counters));
    pInfoNew->mem_private = mem_counters.WorkingSetSize;
    pInfoNew->mem_virtual = mem_counters.PagefileUsage;

    //IO情報
    IO_COUNTERS io_counters = { 0 };
    GetProcessIoCounters(hProcess, &io_counters);
    pInfoNew->io_total_read = io_counters.ReadTransferCount;
    pInfoNew->io_total_write = io_counters.WriteTransferCount;

    //現在時刻
    uint64_t current_time = 0;
    SYSTEMTIME systime = { 0 };
    GetSystemTime(&systime);
    SystemTimeToFileTime(&systime, (FILETIME *)&current_time);

    //CPU情報
    PROCESS_TIME pt = { 0 };
    GetProcessTimes(hProcess, (FILETIME *)&pt.creation, (FILETIME *)&pt.exit, (FILETIME *)&pt.kernel, (FILETIME *)&pt.user);
    pInfoNew->time_us = (current_time - pt.creation) / 10;
    const double time_diff_inv = 1.0 / (pInfoNew->time_us - pInfoOld->time_us);
    if (pInfoNew->time_us > pInfoOld->time_us) {
        pInfoNew->cpu_total_us = (pt.user + pt.kernel) / 10;
        pInfoNew->cpu_total_kernel_us = pt.kernel / 10;

        //CPU使用率
        const double logical_cpu_inv       = 1.0 / m_nLogicalCPU;
        pInfoNew->cpu_percent        = (pInfoNew->cpu_total_us        - pInfoOld->cpu_total_us) * 100.0 * logical_cpu_inv * time_diff_inv;
        pInfoNew->cpu_kernel_percent = (pInfoNew->cpu_total_kernel_us - pInfoOld->cpu_total_kernel_us) * 100.0 * logical_cpu_inv * time_diff_inv;

        //スレッドCPU使用率
        if (m_thMainThread) {
            getThreadTime(m_thMainThread.get(), &pt);
            pInfoNew->main_thread_total_active_us = (pt.user + pt.kernel) / 10;
            pInfoNew->main_thread_percent = (pInfoNew->main_thread_total_active_us - pInfoOld->main_thread_total_active_us) * 100.0 * logical_cpu_inv * time_diff_inv;
        }

        if (m_thEncThread) {
            DWORD exit_code = 0;
            if (0 != GetExitCodeThread(m_thEncThread, &exit_code) && exit_code == STILL_ACTIVE) {
                getThreadTime(m_thEncThread, &pt);
                pInfoNew->enc_thread_total_active_us = (pt.user + pt.kernel) / 10;
                pInfoNew->enc_thread_percent  = (pInfoNew->enc_thread_total_active_us  - pInfoOld->enc_thread_total_active_us) * 100.0 * logical_cpu_inv * time_diff_inv;
            } else {
                pInfoNew->enc_thread_percent = 0.0;
            }
        }

        //IO情報
        pInfoNew->io_read_per_sec = (pInfoNew->io_total_read - pInfoOld->io_total_read) * time_diff_inv * 1e6;
        pInfoNew->io_write_per_sec = (pInfoNew->io_total_write - pInfoOld->io_total_write) * time_diff_inv * 1e6;
    }
#else
    uint64_t current_time = clock() * (1e7 / CLOCKS_PER_SEC);
    pInfoNew->time_us = (current_time - m_nCreateTime100ns) / 10;
    const double time_diff_inv = 1.0 / (pInfoNew->time_us - pInfoOld->time_us);
#endif //#if defined(_WIN32) || defined(_WIN64)

    if (!m_bEncStarted && m_pEncStatus) {
        m_bEncStarted = m_pEncStatus->getEncStarted();
        if (m_bEncStarted) {
            m_nEncStartTime = m_pEncStatus->getStartTimeMicroSec();
        }
    }

    pInfoNew->bitrate_kbps = 0;
    pInfoNew->frames_out_byte = 0;
    pInfoNew->fps = 0.0;
    if (m_bEncStarted && m_pEncStatus) {
        sEncodeStatusData data = { 0 };
        m_pEncStatus->GetEncodeData(&data);

        //fps情報
        pInfoNew->frames_out = data.nProcessedFramesNum;
        if (pInfoNew->frames_out > pInfoOld->frames_out) {
            pInfoNew->fps_avg = pInfoNew->frames_out / (double)(current_time / 10 - m_nEncStartTime) * 1e6;
            if (pInfoNew->time_us > pInfoOld->time_us) {
                pInfoNew->fps     = (pInfoNew->frames_out - pInfoOld->frames_out) * time_diff_inv * 1e6;
            }

            //ビットレート情報
            double videoSec     = pInfoNew->frames_out * m_nOutputFPSScale / (double)m_nOutputFPSRate;
            double videoSecDiff = (pInfoNew->frames_out - pInfoOld->frames_out) * m_nOutputFPSScale / (double)m_nOutputFPSRate;

            pInfoNew->frames_out_byte = data.nWrittenBytes;
            pInfoNew->bitrate_kbps_avg =  pInfoNew->frames_out_byte * 8.0 / videoSec * 1e-3;
            if (pInfoNew->time_us > pInfoOld->time_us) {
                pInfoNew->bitrate_kbps     = (pInfoNew->frames_out_byte - pInfoOld->frames_out_byte) * 8.0 / videoSecDiff * 1e-3;
            }
        }
    }

    m_nStep++;
}

void CPerfMonitor::write() {
    const PerfInfo *pInfo = &m_info[m_nStep & 1];
    std::string str = strsprintf("%lf", pInfo->time_us * 1e-6);
    if (m_nSelectOutputLog & PERF_MONITOR_CPU) {
        str += strsprintf(",%lf", pInfo->cpu_percent);
    }
    if (m_nSelectOutputLog & PERF_MONITOR_CPU_KERNEL) {
        str += strsprintf(",%lf", pInfo->cpu_kernel_percent);
    }
    if (m_nSelectOutputLog & PERF_MONITOR_THREAD_MAIN) {
        str += strsprintf(",%lf", pInfo->main_thread_percent);
    }
    if (m_nSelectOutputLog & PERF_MONITOR_THREAD_ENC) {
        str += strsprintf(",%lf", pInfo->enc_thread_percent);
    }
    if (m_nSelectOutputLog & PERF_MONITOR_MEM_PRIVATE) {
        str += strsprintf(",%.2lf", pInfo->mem_private / (double)(1024 * 1024));
    }
    if (m_nSelectOutputLog & PERF_MONITOR_MEM_VIRTUAL) {
        str += strsprintf(",%.2lf", pInfo->mem_virtual / (double)(1024 * 1024));
    }
    if (m_nSelectOutputLog & PERF_MONITOR_FRAME_IN) {
        str += strsprintf(",%d", pInfo->frames_in);
    }
    if (m_nSelectOutputLog & PERF_MONITOR_FRAME_OUT) {
        str += strsprintf(",%d", pInfo->frames_out);
    }
    if (m_nSelectOutputLog & PERF_MONITOR_FPS) {
        str += strsprintf(",%lf", pInfo->fps);
    }
    if (m_nSelectOutputLog & PERF_MONITOR_FPS_AVG) {
        str += strsprintf(",%lf", pInfo->fps_avg);
    }
    if (m_nSelectOutputLog & PERF_MONITOR_BITRATE) {
        str += strsprintf(",%lf", pInfo->bitrate_kbps);
    }
    if (m_nSelectOutputLog & PERF_MONITOR_BITRATE_AVG) {
        str += strsprintf(",%lf", pInfo->bitrate_kbps_avg);
    }
    if (m_nSelectOutputLog & PERF_MONITOR_IO_READ) {
        str += strsprintf(",%lf", pInfo->io_read_per_sec / (double)(1024 * 1024));
    }
    if (m_nSelectOutputLog & PERF_MONITOR_IO_WRITE) {
        str += strsprintf(",%lf", pInfo->io_write_per_sec / (double)(1024 * 1024));
    }
    str += "\n";
    fwrite(str.c_str(), 1, str.length(), m_fpLog.get());
}

void CPerfMonitor::loader(void *prm) {
    reinterpret_cast<CPerfMonitor*>(prm)->run();
}

void CPerfMonitor::run() {
    while (!m_bAbort) {
        check();
        write();
        std::this_thread::sleep_for(std::chrono::milliseconds(m_nInterval));
    }
    check();
    write();
}
