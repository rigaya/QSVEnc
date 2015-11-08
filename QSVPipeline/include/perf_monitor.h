//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef __PERF_MONITOR_H__
#define __PERF_MONITOR_H__

#include <thread>
#include <cstdint>
#include <climits>
#include <memory>
#include "cpu_info.h"
#include "qsv_util.h"
#ifndef HANDLE
typedef void * HANDLE;
#endif

class CEncodeStatusInfo;

enum : int {
    PERF_MONITOR_CPU         = 0x00000001,
    PERF_MONITOR_CPU_KERNEL  = 0x00000002,
    PERF_MONITOR_MEM_PRIVATE = 0x00000004,
    PERF_MONITOR_MEM_VIRTUAL = 0x00000008,
    PERF_MONITOR_FPS         = 0x00000010,
    PERF_MONITOR_FPS_AVG     = 0x00000020,
    PERF_MONITOR_BITRATE     = 0x00000040,
    PERF_MONITOR_BITRATE_AVG = 0x00000080,
    PERF_MONITOR_IO_READ     = 0x00000100,
    PERF_MONITOR_IO_WRITE    = 0x00000200,
    PERF_MONITOR_THREAD_MAIN = 0x00000400,
    PERF_MONITOR_THREAD_ENC  = 0x00000800,
    PERF_MONITOR_FRAME_IN    = 0x00001000,
    PERF_MONITOR_FRAME_OUT   = 0x00002000,

    PERF_MONITOR_ALL         = (int)UINT_MAX,
};

const CX_DESC list_pref_monitor[] ={
    { _T("all"),         PERF_MONITOR_ALL },
    { _T("cpu"),         PERF_MONITOR_CPU | PERF_MONITOR_CPU_KERNEL | PERF_MONITOR_THREAD_MAIN | PERF_MONITOR_THREAD_ENC },
    { _T("cpu_total"),   PERF_MONITOR_CPU },
    { _T("cpu_kernel"),  PERF_MONITOR_CPU_KERNEL },
    { _T("cpu_main"),    PERF_MONITOR_THREAD_MAIN },
    { _T("cpu_enc"),     PERF_MONITOR_THREAD_ENC },
    { _T("mem"),         PERF_MONITOR_MEM_PRIVATE | PERF_MONITOR_MEM_VIRTUAL },
    { _T("mem_private"), PERF_MONITOR_MEM_PRIVATE },
    { _T("mem_virtual"), PERF_MONITOR_MEM_VIRTUAL },
    { _T("io"),          PERF_MONITOR_IO_READ | PERF_MONITOR_IO_WRITE },
    { _T("io_read"),     PERF_MONITOR_IO_READ },
    { _T("io_write"),    PERF_MONITOR_IO_WRITE },
    { _T("fps"),         PERF_MONITOR_FPS },
    { _T("fps_avg"),     PERF_MONITOR_FPS_AVG },
    { _T("bitrate"),     PERF_MONITOR_BITRATE },
    { _T("bitrate_avg"), PERF_MONITOR_BITRATE_AVG },
    { _T("frame_out"),   PERF_MONITOR_FRAME_OUT },
    { NULL, 0 }
};

struct PerfInfo {
    int64_t time_us;
    int64_t cpu_total_us;
    int64_t cpu_total_kernel_us;

    int64_t main_thread_total_active_us;
    int64_t enc_thread_total_active_us;

    int64_t mem_private;
    int64_t mem_virtual;

    int64_t io_total_read;
    int64_t io_total_write;

    int64_t frames_in;
    int64_t frames_out;
    int64_t frames_out_byte;

    double  fps;
    double  fps_avg;

    double  bitrate_kbps;
    double  bitrate_kbps_avg;

    double  io_read_per_sec;
    double  io_write_per_sec;

    double  cpu_percent;
    double  cpu_kernel_percent;

    double  main_thread_percent;
    double  enc_thread_percent;
};

struct PerfOutputInfo {
    int flag;
    const TCHAR *fmt;
    ptrdiff_t offset;
};

class CPerfMonitor {
public:
    CPerfMonitor();
    int init(tstring filename,
        int interval, bool bUseMatplotLib, int nSelectOutputLog,
        std::unique_ptr<void, handle_deleter> thMainThread);
    ~CPerfMonitor();

    void SetEncStatus(std::shared_ptr<CEncodeStatusInfo> encStatus);
    void SetEncThread(HANDLE thEncThread);

    void clear();
protected:
    void check();
    void run();
    void write();

    static void loader(void *prm);

    int m_nStep;
    PerfInfo m_info[2];
    std::thread m_thCheck;
    std::unique_ptr<void, handle_deleter> m_thMainThread;
    HANDLE m_thEncThread;
    int m_nLogicalCPU;
    std::shared_ptr<CEncodeStatusInfo> m_pEncStatus;
    int64_t m_nEncStartTime;
    int64_t m_nOutputFPSRate;
    int64_t m_nOutputFPSScale;
    int64_t m_nCreateTime100ns;
    bool m_bAbort;
    bool m_bEncStarted;
    int m_nInterval;
    tstring m_sMonitorFilename;
    std::unique_ptr<FILE, fp_deleter> m_fpLog;
    bool m_bUseMatplotLib;
    int m_nSelectOutputLog;
};


#endif //#ifndef __PERF_MONITOR_H__
