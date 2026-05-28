// -----------------------------------------------------------------------------------------
// QSVEncKFM by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2024 rigaya
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
#ifndef __RGY_OPENCL_PERF_H__
#define __RGY_OPENCL_PERF_H__

#include "rgy_version.h"

#if ENABLE_OPENCL

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <tuple>
#include <functional>
#include "rgy_tchar.h"
#include "rgy_opencl.h"

// ヒストグラム段数: 100ns 起点 × 1.4 倍 × 64 段
static constexpr int RGY_CL_PERF_HIST_BUCKETS = 64;

// FNV-1a 64bit ハッシュ → 16hex 文字列で返す
std::string rgy_cl_perf_fnv1a_hex(const std::string& s);

// ヒストグラムのバケットインデックスを計算 (100ns 起点, 1.4 倍ピッチ)
int rgy_cl_perf_hist_bucket(uint64_t time_ns);

// ----------------------------------------
// Data structures
// ----------------------------------------

struct KernelInfo {
    std::string name;
    cl_uint  num_args              = 0;
    size_t   work_group_size       = 0;     // CL_KERNEL_WORK_GROUP_SIZE
    size_t   preferred_wg_multiple = 0;     // CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
    cl_ulong local_mem_size        = 0;     // CL_KERNEL_LOCAL_MEM_SIZE
    cl_ulong private_mem_size      = 0;     // CL_KERNEL_PRIVATE_MEM_SIZE
    cl_ulong spill_mem_size        = UINT64_MAX; // vendor-specific spill size; UINT64_MAX = unknown
    std::string spill_mem_source;
    std::string attributes;                 // CL_KERNEL_ATTRIBUTES
    // 取得失敗フィールドを null 出力する用
    bool work_group_size_valid       = false;
    bool preferred_wg_multiple_valid = false;
    bool local_mem_size_valid        = false;
    bool private_mem_size_valid      = false;
    bool spill_mem_size_valid        = false;
    bool attributes_valid            = false;
};

struct RGYOpenCLPerfProgramKey {
    std::string resource_name;  // "RGY_FILTER_RTGMC_BOB_CL" 等
    std::string options_hash;   // FNV-1a 先頭 16 hex
    bool operator==(const RGYOpenCLPerfProgramKey& o) const {
        return resource_name == o.resource_name && options_hash == o.options_hash;
    }
};

struct RGYOpenCLPerfProgramRecord {
    uint64_t    program_id      = 0;
    RGYOpenCLPerfProgramKey key;
    std::string build_options;
    std::string device_name;
    std::string driver_version;
    int         device_ip_version = 0; // -1 = unknown
    std::string build_log;
    uint64_t    build_time_ns = 0;
    std::vector<KernelInfo> kernels;
    std::string binary_filename;        // 相対パス: "binaries/<resource>__<hash>.bin"
    std::string build_log_filename;     // 相対パス: "build_logs/<resource>__<hash>.log"
};

struct RGYOpenCLPerfLaunchKey {
    uint64_t    program_id = 0;
    std::string kernel_name;
    std::array<size_t, 3> local      = {};
    std::array<size_t, 3> global_ceil = {};
    bool operator==(const RGYOpenCLPerfLaunchKey& o) const {
        return program_id  == o.program_id
            && kernel_name == o.kernel_name
            && local       == o.local
            && global_ceil == o.global_ceil;
    }
};

struct RGYOpenCLPerfLaunchKeyHash {
    size_t operator()(const RGYOpenCLPerfLaunchKey& k) const {
        size_t h = std::hash<uint64_t>{}(k.program_id);
        for (auto c : k.kernel_name) h ^= std::hash<char>{}(c) + 0x9e3779b9 + (h << 6) + (h >> 2);
        for (int i = 0; i < 3; i++) {
            h ^= std::hash<size_t>{}(k.local[i])       + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<size_t>{}(k.global_ceil[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct RGYOpenCLPerfLaunchAgg {
    uint64_t count        = 0;
    uint64_t time_sum_ns  = 0;
    uint64_t time_min_ns  = UINT64_MAX;
    uint64_t time_max_ns  = 0;
    std::array<uint64_t, RGY_CL_PERF_HIST_BUCKETS> hist_log = {};
    size_t   subgroup_size  = 0;
    size_t   subgroup_count = 0;
    bool     subgroup_init  = false;
};

struct RGYOpenCLPerfCommandKey {
    std::string command_name;
    bool operator==(const RGYOpenCLPerfCommandKey& o) const {
        return command_name == o.command_name;
    }
};

struct RGYOpenCLPerfCommandKeyHash {
    size_t operator()(const RGYOpenCLPerfCommandKey& k) const {
        size_t h = 0;
        for (auto c : k.command_name) h ^= std::hash<char>{}(c) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct RGYOpenCLPerfCommandAgg {
    uint64_t count = 0;
    uint64_t bytes_sum = 0;
    uint64_t time_sum_ns = 0;
    uint64_t time_min_ns = UINT64_MAX;
    uint64_t time_max_ns = 0;
    uint64_t queued_to_submit_sum_ns = 0;
    uint64_t submit_to_start_sum_ns = 0;
    uint64_t queued_to_start_sum_ns = 0;
    uint64_t host_time_sum_ns = 0;
    uint64_t host_time_min_ns = UINT64_MAX;
    uint64_t host_time_max_ns = 0;
    std::array<uint64_t, RGY_CL_PERF_HIST_BUCKETS> hist_log = {};
};

struct RGYOpenCLPerfAllocationKey {
    std::string operation_name;
    bool operator==(const RGYOpenCLPerfAllocationKey& o) const {
        return operation_name == o.operation_name;
    }
};

struct RGYOpenCLPerfAllocationKeyHash {
    size_t operator()(const RGYOpenCLPerfAllocationKey& k) const {
        size_t h = 0;
        for (auto c : k.operation_name) h ^= std::hash<char>{}(c) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct RGYOpenCLPerfAllocationAgg {
    uint64_t count = 0;
    uint64_t success_count = 0;
    uint64_t fail_count = 0;
    uint64_t bytes_sum = 0;
    uint64_t host_time_sum_ns = 0;
    uint64_t host_time_min_ns = UINT64_MAX;
    uint64_t host_time_max_ns = 0;
};

// ----------------------------------------
// Collector singleton
// ----------------------------------------

class RGYOpenCLPerfCollector {
public:
    static RGYOpenCLPerfCollector& instance();

    // 有効化。dump_dir 以下にファイルを書き出す
    void enable(const tstring& dump_dir);
    bool isEnabled() const { return m_enabled.load(std::memory_order_acquire); }

    // ビルド完了フック。program_id を返す
    uint64_t recordProgramBuild(const std::string& resource_name,
                                const std::string& build_options,
                                cl_program         program,
                                cl_device_id       devid,
                                const std::string& build_log,
                                uint64_t           build_time_ns);

    // launch フック (event は pending キューに積むだけ)
    void recordLaunch(uint64_t                    program_id,
                      const std::string&           kernel_name,
                      const RGYWorkSize&           local,
                      const RGYWorkSize&           global_ceil,
                      cl_kernel                    kernel,
                      cl_device_id                 devid,
                      RGYOpenCLEvent&              event);

    void recordCommand(const std::string& command_name,
                       uint64_t           bytes,
                       uint64_t           host_time_ns,
                       RGYOpenCLEvent&    event);

    void recordAllocation(const std::string& operation_name,
                          uint64_t           bytes,
                          uint64_t           host_time_ns,
                          bool               success);

    // 終了時: pending event を回収して JSONL を書き出す (best-effort)
    void flush();

private:
    RGYOpenCLPerfCollector();
    ~RGYOpenCLPerfCollector() = default;
    RGYOpenCLPerfCollector(const RGYOpenCLPerfCollector&) = delete;
    void operator=(const RGYOpenCLPerfCollector&) = delete;

    // pending を消化して agg に加算 (mutex 保持下で呼ぶこと)
    void drainPending();

    // JSONL 書き出し
    void writeProgramsJsonl(const std::string& path);
    void writeLaunchesJsonl(const std::string& path);
    void writeCommandsJsonl(const std::string& path);
    void writeAllocationsJsonl(const std::string& path);
    void writeMetaJson(const std::string& path);

    std::atomic<bool>     m_enabled;
    tstring               m_dump_dir;
    std::mutex            m_mtx;
    std::atomic<uint64_t> m_next_program_id;
    std::vector<RGYOpenCLPerfProgramRecord> m_programs;
    std::unordered_map<RGYOpenCLPerfLaunchKey,
                       RGYOpenCLPerfLaunchAgg,
                       RGYOpenCLPerfLaunchKeyHash> m_aggs;
    // pending: (key, event, kernel)
    std::vector<std::tuple<RGYOpenCLPerfLaunchKey, RGYOpenCLEvent, cl_kernel>> m_pending;
    std::unordered_map<RGYOpenCLPerfCommandKey,
                       RGYOpenCLPerfCommandAgg,
                       RGYOpenCLPerfCommandKeyHash> m_command_aggs;
    std::vector<std::tuple<RGYOpenCLPerfCommandKey, RGYOpenCLEvent, uint64_t, uint64_t>> m_command_pending;
    std::unordered_map<RGYOpenCLPerfAllocationKey,
                       RGYOpenCLPerfAllocationAgg,
                       RGYOpenCLPerfAllocationKeyHash> m_allocation_aggs;
};

#endif // ENABLE_OPENCL
#endif // __RGY_OPENCL_PERF_H__
