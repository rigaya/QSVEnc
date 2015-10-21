//  -----------------------------------------------------------------------------------------
//    ram_speed by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <stdio.h>
#include <cstdint>
#include <vector>
#include <numeric>
#include <chrono>
#include <climits>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "cpu_info.h"
#include "qsv_simd.h"
#include "ram_speed.h"

typedef struct {
    int mode;
    uint32_t check_size_bytes;
    uint32_t thread_id;
    uint32_t physical_cores;
    double megabytes_per_sec;
} RAM_SPEED_THREAD;


typedef struct {
    bool ready;
    std::mutex mtx;
    std::condition_variable cv;
} RAM_SPEED_THREAD_WAKE;

#ifdef __cplusplus
extern "C" {
#endif
extern void __stdcall read_sse(uint8_t *src, uint32_t size, uint32_t count_n);
extern void __stdcall read_avx(uint8_t *src, uint32_t size, uint32_t count_n);
extern void __stdcall write_sse(uint8_t *dst, uint32_t size, uint32_t count_n);
extern void __stdcall write_avx(uint8_t *dst, uint32_t size, uint32_t count_n);
#ifdef __cplusplus
}
#endif


typedef void(__stdcall *func_ram_test)(uint8_t *dst, uint32_t size, uint32_t count_n);

void ram_speed_func(RAM_SPEED_THREAD *thread_prm, RAM_SPEED_THREAD_WAKE *thread_wk) {
    const int TEST_COUNT = 4;
    uint32_t check_size_bytes = (thread_prm->check_size_bytes + 255) & ~255;
    const uint32_t test_kilo_bytes   = (uint32_t)(((thread_prm->mode == RAM_SPEED_MODE_READ) ? 1 : 0.5) * thread_prm->physical_cores * 1024 * 1024 / (std::max)(1.0, log2(check_size_bytes / 1024.0)) + 0.5);
    const uint32_t warmup_kilo_bytes = test_kilo_bytes * 2;
    uint8_t *ptr = (uint8_t *)_aligned_malloc(check_size_bytes, 64);
    uint32_t count_n = (int)(test_kilo_bytes * 1024.0 / check_size_bytes + 0.5);
    int avx = 0 != (get_availableSIMD() & AVX);
    int64_t result[TEST_COUNT];
    static const func_ram_test RAM_TEST_LIST[][2] = {
        {read_sse, write_sse},
        {read_avx, write_avx},
    };

    const func_ram_test ram_test = RAM_TEST_LIST[avx][thread_prm->mode];

    {
        std::unique_lock<std::mutex> uniq_lk(thread_wk->mtx); // ここでロックされる
        thread_wk->cv.wait(uniq_lk, [&thread_wk] { return thread_wk->ready; });
    }

    ram_test(ptr, check_size_bytes, (int)(warmup_kilo_bytes * 1024.0 / check_size_bytes + 0.5));
    for (int i = 0; i < TEST_COUNT; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        ram_test(ptr, check_size_bytes, count_n);
        auto fin = std::chrono::high_resolution_clock::now();
        result[i] = std::chrono::duration_cast<std::chrono::microseconds>(fin - start).count();
    }
    ram_test(ptr, check_size_bytes, (int)(warmup_kilo_bytes * 1024.0 / check_size_bytes + 0.5));
    _aligned_free(ptr);

    int64_t time_min = LLONG_MAX;
    for (int i = 0; i < TEST_COUNT; i++)
        time_min = (std::min)(time_min, result[i]);

    thread_prm->megabytes_per_sec = (check_size_bytes * (double)count_n / (1024.0 * 1024.0)) / (time_min * 0.000001);
}

double ram_speed_mt(int check_size_kilobytes, int mode, int thread_n) {
    std::vector<std::thread> threads(thread_n);
    std::vector<RAM_SPEED_THREAD> thread_prm(thread_n);
    RAM_SPEED_THREAD_WAKE thread_wake;
    cpu_info_t cpu_info;
    get_cpu_info(&cpu_info);
    for (uint32_t i = 0; i < threads.size(); i++) {
        thread_prm[i].physical_cores = cpu_info.physical_cores;
        thread_prm[i].mode = (mode == RAM_SPEED_MODE_RW) ? (i & 1) : mode;
        thread_prm[i].check_size_bytes = (check_size_kilobytes * 1024 / thread_n + 255) & ~255;
        thread_prm[i].thread_id = (i % cpu_info.physical_cores) * (cpu_info.logical_cores / cpu_info.physical_cores) + (int)(i / cpu_info.physical_cores);
        threads[i] = std::thread(ram_speed_func, &thread_prm[i], &thread_wake);
        //渡されたスレッドIDからスレッドAffinityを決定
        //特定のコアにスレッドを縛り付ける
        SetThreadAffinityMask(threads[i].native_handle(), 1 << (int)thread_prm[i].thread_id);
        //高優先度で実行
        SetThreadPriority(threads[i].native_handle(), THREAD_PRIORITY_HIGHEST);
    }
    
    { //スレッドを起動
        std::unique_lock<std::mutex> lock(thread_wake.mtx);
        thread_wake.ready = true;
        thread_wake.cv.notify_all();
    }
    for (uint32_t i = 0; i < threads.size(); i++) {
        threads[i].join();
    }

    double sum = 0.0;
    for (const auto& prm : thread_prm) {
        sum += prm.megabytes_per_sec;
    }
    return sum;
}

std::vector<double> ram_speed_mt_list(int check_size_kilobytes, int mode, bool logical_core) {
    cpu_info_t cpu_info;
    get_cpu_info(&cpu_info);

    std::vector<double> results;
    for (uint32_t ith = 1; ith <= cpu_info.physical_cores; ith++) {
        results.push_back(ram_speed_mt(check_size_kilobytes, mode, ith));
    }
    if (logical_core && cpu_info.logical_cores != cpu_info.physical_cores) {
        int smt = cpu_info.logical_cores / cpu_info.physical_cores;
        for (int i_smt = 2; i_smt <= smt; i_smt++) {
            results.push_back(ram_speed_mt(check_size_kilobytes, mode, cpu_info.physical_cores * i_smt));
        }
    }
    return results;
}
