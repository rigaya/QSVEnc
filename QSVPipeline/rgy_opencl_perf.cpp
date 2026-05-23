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

#include "rgy_opencl_perf.h"

#if ENABLE_OPENCL

#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <ctime>

#include "rgy_filesystem.h"
#include "rgy_util.h"

// CL_KERNEL_SPILL_MEM_SIZE_INTEL は公式 SDK に入っていない場合があるため自前定義
#ifndef CL_KERNEL_SPILL_MEM_SIZE_INTEL
#define CL_KERNEL_SPILL_MEM_SIZE_INTEL 0x4109
#endif

// ----------------------------------------
// FNV-1a 64bit ハッシュ
// ----------------------------------------
std::string rgy_cl_perf_fnv1a_hex(const std::string& s) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : s) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ULL;
    }
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return std::string(buf);
}

// ----------------------------------------
// log-spaced histogram バケット計算
// 100ns 起点、1.4 倍ずつ、64 段
// ----------------------------------------
int rgy_cl_perf_hist_bucket(uint64_t time_ns) {
    if (time_ns < 100) return 0;
    const double ratio = std::log(static_cast<double>(time_ns) / 100.0) / std::log(1.4);
    int idx = static_cast<int>(ratio);
    if (idx < 0) idx = 0;
    if (idx >= RGY_CL_PERF_HIST_BUCKETS) idx = RGY_CL_PERF_HIST_BUCKETS - 1;
    return idx;
}

// ----------------------------------------
// シンプルな JSON 文字列エスケープ
// ----------------------------------------
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        case '\t': out += "\\t";  break;
        default:
            if (c < 0x20) {
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
                out += buf;
            } else {
                out += (char)c;
            }
            break;
        }
    }
    return out;
}

// ISO8601 タイムスタンプ (ローカルタイム)
static std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf;
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char buf[64];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d",
        tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
        tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
    return std::string(buf);
}

// ----------------------------------------
// RGYOpenCLPerfCollector 実装
// ----------------------------------------

RGYOpenCLPerfCollector::RGYOpenCLPerfCollector()
    : m_enabled(false), m_dump_dir(), m_mtx(),
      m_next_program_id(1), m_programs(), m_aggs(), m_pending(),
      m_command_aggs(), m_command_pending(), m_allocation_aggs() {
}

RGYOpenCLPerfCollector& RGYOpenCLPerfCollector::instance() {
    static RGYOpenCLPerfCollector s_instance;
    return s_instance;
}

void RGYOpenCLPerfCollector::enable(const tstring& dump_dir) {
    std::lock_guard<std::mutex> lock(m_mtx);
    m_dump_dir = dump_dir;
    m_enabled.store(true, std::memory_order_release);
}

uint64_t RGYOpenCLPerfCollector::recordProgramBuild(
    const std::string& resource_name,
    const std::string& build_options,
    cl_program         program,
    cl_device_id       devid,
    const std::string& build_log,
    uint64_t           build_time_ns)
{
    if (!isEnabled()) return 0;

    const std::string options_hash = rgy_cl_perf_fnv1a_hex(build_options);
    const uint64_t prog_id = m_next_program_id.fetch_add(1, std::memory_order_relaxed);

    RGYOpenCLPerfProgramRecord rec;
    rec.program_id    = prog_id;
    rec.key.resource_name = resource_name;
    rec.key.options_hash  = options_hash;
    rec.build_options = build_options;
    rec.build_log     = build_log;
    rec.build_time_ns = build_time_ns;

    // デバイス情報取得
    {
        auto devInfo = RGYOpenCLDevice(devid).info();
        rec.device_name    = devInfo.name;
        rec.driver_version = devInfo.driver_version;
#if ENCODER_QSV || CLFILTERS_AUF
        rec.device_ip_version = devInfo.ip_version_intel;
#else
        rec.device_ip_version = -1;
#endif
    }

    // kernel 列挙と情報取得
    {
        cl_uint num_kernels = 0;
        cl_int  err = clCreateKernelsInProgram(program, 0, nullptr, &num_kernels);
        if (err == CL_SUCCESS && num_kernels > 0) {
            std::vector<cl_kernel> kernels(num_kernels);
            err = clCreateKernelsInProgram(program, num_kernels, kernels.data(), nullptr);
            if (err == CL_SUCCESS) {
                for (auto& kobj : kernels) {
                    KernelInfo ki;
                    // 名前
                    size_t name_sz = 0;
                    if (clGetKernelInfo(kobj, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &name_sz) == CL_SUCCESS) {
                        std::vector<char> name_buf(name_sz);
                        clGetKernelInfo(kobj, CL_KERNEL_FUNCTION_NAME, name_sz, name_buf.data(), nullptr);
                        ki.name = std::string(name_buf.data());
                    }
                    // num_args
                    if (clGetKernelInfo(kobj, CL_KERNEL_NUM_ARGS, sizeof(ki.num_args), &ki.num_args, nullptr) == CL_SUCCESS) {}
                    // attributes
                    size_t attr_sz = 0;
                    if (clGetKernelInfo(kobj, CL_KERNEL_ATTRIBUTES, 0, nullptr, &attr_sz) == CL_SUCCESS) {
                        std::vector<char> attr_buf(attr_sz);
                        if (clGetKernelInfo(kobj, CL_KERNEL_ATTRIBUTES, attr_sz, attr_buf.data(), nullptr) == CL_SUCCESS) {
                            ki.attributes = std::string(attr_buf.data());
                            ki.attributes_valid = true;
                        }
                    }
                    // work group info
                    if (clGetKernelWorkGroupInfo(kobj, devid, CL_KERNEL_WORK_GROUP_SIZE,
                            sizeof(ki.work_group_size), &ki.work_group_size, nullptr) == CL_SUCCESS) {
                        ki.work_group_size_valid = true;
                    }
                    if (clGetKernelWorkGroupInfo(kobj, devid, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                            sizeof(ki.preferred_wg_multiple), &ki.preferred_wg_multiple, nullptr) == CL_SUCCESS) {
                        ki.preferred_wg_multiple_valid = true;
                    }
                    if (clGetKernelWorkGroupInfo(kobj, devid, CL_KERNEL_LOCAL_MEM_SIZE,
                            sizeof(ki.local_mem_size), &ki.local_mem_size, nullptr) == CL_SUCCESS) {
                        ki.local_mem_size_valid = true;
                    }
                    if (clGetKernelWorkGroupInfo(kobj, devid, CL_KERNEL_PRIVATE_MEM_SIZE,
                            sizeof(ki.private_mem_size), &ki.private_mem_size, nullptr) == CL_SUCCESS) {
                        ki.private_mem_size_valid = true;
                    }
                    // spill mem (Intel 拡張: CL_KERNEL_SPILL_MEM_SIZE_INTEL)
                    cl_ulong spill = 0;
                    if (clGetKernelWorkGroupInfo(kobj, devid, CL_KERNEL_SPILL_MEM_SIZE_INTEL,
                            sizeof(spill), &spill, nullptr) == CL_SUCCESS) {
                        ki.spill_mem_size_intel = spill;
                        ki.spill_mem_size_valid = true;
                    }

                    rec.kernels.push_back(std::move(ki));
                    clReleaseKernel(kobj);
                }
            }
        }
    }

    // バイナリ / ビルドログを保存 (dump_dir が有効な場合のみ)
    {
        const std::string safe_name = [&]() {
            // resource_name の非ASCII / 記号を _ に置換してファイル名に使う
            std::string s = resource_name;
            for (auto& c : s) {
                if (!isalnum((unsigned char)c) && c != '_' && c != '-') c = '_';
            }
            return s;
        }();
        const std::string stem = safe_name + "__" + options_hash;

        const std::string bin_relpath = std::string("binaries/") + stem + ".bin";
        const std::string log_relpath = std::string("build_logs/") + stem + ".log";
        rec.binary_filename   = bin_relpath;
        rec.build_log_filename = log_relpath;

        // dump_dir が空でなければ書き出す
        const std::string dump_dir_str = tchar_to_string(m_dump_dir);
        if (!dump_dir_str.empty()) {
            // binaries/ ディレクトリ作成
            const std::string bin_dir = dump_dir_str + "/binaries";
            CreateDirectoryRecursive(bin_dir.c_str());

            // バイナリ取得と保存
            {
                size_t binary_size = 0;
                cl_int err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, nullptr);
                if (err == CL_SUCCESS && binary_size > 0) {
                    std::vector<uint8_t> binary(binary_size, 0);
                    uint8_t* bin_ptr = binary.data();
                    err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, binary_size, &bin_ptr, nullptr);
                    if (err == CL_SUCCESS) {
                        const std::string bin_path = dump_dir_str + "/" + bin_relpath;
                        std::ofstream ofs(bin_path, std::ios::binary);
                        if (ofs) {
                            ofs.write(reinterpret_cast<const char*>(binary.data()), (std::streamsize)binary_size);
                        }
                    }
                }
            }

            // ビルドログ保存
            {
                const std::string log_dir = dump_dir_str + "/build_logs";
                CreateDirectoryRecursive(log_dir.c_str());
                const std::string log_path = dump_dir_str + "/" + log_relpath;
                std::ofstream ofs(log_path);
                if (ofs) {
                    ofs << build_log;
                }
            }
        }
    }

    std::lock_guard<std::mutex> lock(m_mtx);
    m_programs.push_back(std::move(rec));
    return prog_id;
}

void RGYOpenCLPerfCollector::recordLaunch(
    uint64_t           program_id,
    const std::string& kernel_name,
    const RGYWorkSize& local,
    const RGYWorkSize& global_ceil,
    cl_kernel          kernel,
    RGYOpenCLEvent&    event)
{
    if (!isEnabled()) return;

    RGYOpenCLPerfLaunchKey key;
    key.program_id  = program_id;
    key.kernel_name = kernel_name;
    for (int i = 0; i < 3; i++) {
        key.local[i]       = local.w[i];
        key.global_ceil[i] = global_ceil.w[i];
    }

    std::lock_guard<std::mutex> lock(m_mtx);

    // subgroup 情報: 初回のみ取得 (kernel ハンドルが必要)
    auto& agg = m_aggs[key];
    if (!agg.subgroup_init && kernel != nullptr) {
        auto clFunc = (clGetKernelSubGroupInfo) ? clGetKernelSubGroupInfo : clGetKernelSubGroupInfoKHR;
        if (clFunc) {
            size_t subGroupSize = 0;
            if (clFunc(kernel, nullptr, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
                    sizeof(RGYWorkSize::w), local(), sizeof(subGroupSize), &subGroupSize, nullptr) == CL_SUCCESS) {
                agg.subgroup_size = subGroupSize;
            }
            size_t subGroupCount = 0;
            if (clFunc(kernel, nullptr, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE,
                    sizeof(RGYWorkSize::w), local(), sizeof(subGroupCount), &subGroupCount, nullptr) == CL_SUCCESS) {
                agg.subgroup_count = subGroupCount;
            }
        }
        agg.subgroup_init = true;
    }

    // event を pending キューに積む (コピーを shared_ptr で保持)
    m_pending.emplace_back(key, event, kernel);
}

void RGYOpenCLPerfCollector::recordCommand(
    const std::string& command_name,
    uint64_t           bytes,
    uint64_t           host_time_ns,
    RGYOpenCLEvent&    event)
{
    if (!isEnabled()) return;

    RGYOpenCLPerfCommandKey key;
    key.command_name = command_name;

    std::lock_guard<std::mutex> lock(m_mtx);
    if (event() != nullptr) {
        m_command_pending.emplace_back(key, event, bytes, host_time_ns);
    } else {
        auto& agg = m_command_aggs[key];
        agg.count++;
        agg.bytes_sum += bytes;
        agg.host_time_sum_ns += host_time_ns;
        if (host_time_ns < agg.host_time_min_ns) agg.host_time_min_ns = host_time_ns;
        if (host_time_ns > agg.host_time_max_ns) agg.host_time_max_ns = host_time_ns;
    }
}

void RGYOpenCLPerfCollector::recordAllocation(
    const std::string& operation_name,
    uint64_t           bytes,
    uint64_t           host_time_ns,
    bool               success)
{
    if (!isEnabled()) return;

    RGYOpenCLPerfAllocationKey key;
    key.operation_name = operation_name;

    std::lock_guard<std::mutex> lock(m_mtx);
    auto& agg = m_allocation_aggs[key];
    agg.count++;
    agg.success_count += success ? 1 : 0;
    agg.fail_count += success ? 0 : 1;
    agg.bytes_sum += bytes;
    agg.host_time_sum_ns += host_time_ns;
    if (host_time_ns < agg.host_time_min_ns) agg.host_time_min_ns = host_time_ns;
    if (host_time_ns > agg.host_time_max_ns) agg.host_time_max_ns = host_time_ns;
}

void RGYOpenCLPerfCollector::drainPending() {
    // m_pending を走査: 完了済み event からプロファイリング情報を回収
    for (auto& [key, event, kobj] : m_pending) {
        if (event() == nullptr) continue;

        // clWaitForEvents は呼ばない (flush でまとめて待つ)
        uint64_t t_start = 0, t_end = 0;
        RGY_ERR err1 = event.getProfilingTimeStart(t_start);
        RGY_ERR err2 = event.getProfilingTimeEnd(t_end);

        if (err1 == RGY_ERR_NONE && err2 == RGY_ERR_NONE && t_end >= t_start) {
            const uint64_t dt = t_end - t_start;
            auto& agg = m_aggs[key];
            agg.count++;
            agg.time_sum_ns += dt;
            if (dt < agg.time_min_ns) agg.time_min_ns = dt;
            if (dt > agg.time_max_ns) agg.time_max_ns = dt;
            agg.hist_log[rgy_cl_perf_hist_bucket(dt)]++;
        }
        // (void)kobj; already used above
    }
    m_pending.clear();

    for (auto& [key, event, bytes, host_time_ns] : m_command_pending) {
        auto& agg = m_command_aggs[key];
        agg.count++;
        agg.bytes_sum += bytes;
        agg.host_time_sum_ns += host_time_ns;
        if (host_time_ns < agg.host_time_min_ns) agg.host_time_min_ns = host_time_ns;
        if (host_time_ns > agg.host_time_max_ns) agg.host_time_max_ns = host_time_ns;

        if (event() == nullptr) continue;

        uint64_t t_queued = 0, t_submit = 0, t_start = 0, t_end = 0;
        const auto err_queued = event.getProfilingTimeQueued(t_queued);
        const auto err_submit = event.getProfilingTimeSubmit(t_submit);
        const auto err_start = event.getProfilingTimeStart(t_start);
        const auto err_end = event.getProfilingTimeEnd(t_end);

        if (err_start == RGY_ERR_NONE && err_end == RGY_ERR_NONE && t_end >= t_start) {
            const uint64_t dt = t_end - t_start;
            agg.time_sum_ns += dt;
            if (dt < agg.time_min_ns) agg.time_min_ns = dt;
            if (dt > agg.time_max_ns) agg.time_max_ns = dt;
            agg.hist_log[rgy_cl_perf_hist_bucket(dt)]++;
        }
        if (err_queued == RGY_ERR_NONE && err_submit == RGY_ERR_NONE && t_submit >= t_queued) {
            agg.queued_to_submit_sum_ns += t_submit - t_queued;
        }
        if (err_submit == RGY_ERR_NONE && err_start == RGY_ERR_NONE && t_start >= t_submit) {
            agg.submit_to_start_sum_ns += t_start - t_submit;
        }
        if (err_queued == RGY_ERR_NONE && err_start == RGY_ERR_NONE && t_start >= t_queued) {
            agg.queued_to_start_sum_ns += t_start - t_queued;
        }
    }
    m_command_pending.clear();
}

void RGYOpenCLPerfCollector::flush() {
    try {
        std::lock_guard<std::mutex> lock(m_mtx);

        // pending event を一括 wait してから drainPending
        if (!m_pending.empty() || !m_command_pending.empty()) {
            std::vector<cl_event> cl_events;
            for (auto& [key, event, kobj] : m_pending) {
                if (event() != nullptr) cl_events.push_back(event());
            }
            for (auto& [key, event, bytes, host_time_ns] : m_command_pending) {
                if (event() != nullptr) cl_events.push_back(event());
            }
            if (!cl_events.empty()) {
                clWaitForEvents((cl_uint)cl_events.size(), cl_events.data());
            }
            drainPending();
        }

        if (m_programs.empty() && m_aggs.empty() && m_command_aggs.empty() && m_allocation_aggs.empty()) return;

        const std::string dump_dir_str = tchar_to_string(m_dump_dir);
        if (dump_dir_str.empty()) return;

        // dump_dir 作成
        CreateDirectoryRecursive(dump_dir_str.c_str());

        writeProgramsJsonl(dump_dir_str + "/programs.jsonl");
        writeLaunchesJsonl(dump_dir_str + "/launches.jsonl");
        writeCommandsJsonl(dump_dir_str + "/commands.jsonl");
        writeAllocationsJsonl(dump_dir_str + "/allocations.jsonl");
        writeMetaJson(dump_dir_str + "/meta.json");
    } catch (...) {
        // best-effort: 例外を飲む
    }
}

// ----------------------------------------
// JSONL writer helpers
// ----------------------------------------

static std::string size_arr_to_json(const std::array<size_t, 3>& a) {
    return "[" + std::to_string(a[0]) + ", " + std::to_string(a[1]) + ", " + std::to_string(a[2]) + "]";
}

void RGYOpenCLPerfCollector::writeProgramsJsonl(const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs) return;

    for (const auto& rec : m_programs) {
        ofs << "{";
        ofs << "\"program_id\":" << rec.program_id;
        ofs << ",\"resource_name\":\"" << json_escape(rec.key.resource_name) << "\"";
        ofs << ",\"options_hash\":\"" << json_escape(rec.key.options_hash) << "\"";
        ofs << ",\"build_options\":\"" << json_escape(rec.build_options) << "\"";
        ofs << ",\"device_name\":\"" << json_escape(rec.device_name) << "\"";
        if (rec.device_ip_version >= 0) {
            ofs << ",\"device_ip_version\":" << rec.device_ip_version;
        } else {
            ofs << ",\"device_ip_version\":null";
        }
        ofs << ",\"driver_version\":\"" << json_escape(rec.driver_version) << "\"";
        ofs << ",\"build_time_ns\":" << rec.build_time_ns;
        ofs << ",\"binary_path\":\"" << json_escape(rec.binary_filename) << "\"";
        ofs << ",\"build_log_path\":\"" << json_escape(rec.build_log_filename) << "\"";
        ofs << ",\"kernels\":[";
        for (size_t i = 0; i < rec.kernels.size(); i++) {
            if (i > 0) ofs << ",";
            const auto& ki = rec.kernels[i];
            ofs << "{";
            ofs << "\"name\":\"" << json_escape(ki.name) << "\"";
            ofs << ",\"num_args\":" << ki.num_args;
            if (ki.work_group_size_valid)       ofs << ",\"work_group_size\":"       << ki.work_group_size;
            else                                ofs << ",\"work_group_size\":null";
            if (ki.preferred_wg_multiple_valid) ofs << ",\"preferred_wg_multiple\":" << ki.preferred_wg_multiple;
            else                                ofs << ",\"preferred_wg_multiple\":null";
            if (ki.local_mem_size_valid)        ofs << ",\"local_mem_size\":"        << ki.local_mem_size;
            else                                ofs << ",\"local_mem_size\":null";
            if (ki.private_mem_size_valid)      ofs << ",\"private_mem_size\":"      << ki.private_mem_size;
            else                                ofs << ",\"private_mem_size\":null";
            if (ki.spill_mem_size_valid)        ofs << ",\"spill_mem_size_intel\":"  << ki.spill_mem_size_intel;
            else                                ofs << ",\"spill_mem_size_intel\":null";
            if (ki.attributes_valid)            ofs << ",\"attributes\":\"" << json_escape(ki.attributes) << "\"";
            else                                ofs << ",\"attributes\":null";
            ofs << "}";
        }
        ofs << "]";
        ofs << "}\n";
    }
}

void RGYOpenCLPerfCollector::writeLaunchesJsonl(const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs) return;

    for (const auto& [key, agg] : m_aggs) {
        ofs << "{";
        ofs << "\"program_id\":" << key.program_id;
        ofs << ",\"kernel_name\":\"" << json_escape(key.kernel_name) << "\"";
        ofs << ",\"local\":" << size_arr_to_json(key.local);
        ofs << ",\"global_ceil\":" << size_arr_to_json(key.global_ceil);
        ofs << ",\"count\":" << agg.count;
        ofs << ",\"time_sum_ns\":" << agg.time_sum_ns;
        if (agg.count > 0 && agg.time_min_ns != UINT64_MAX)
            ofs << ",\"time_min_ns\":" << agg.time_min_ns;
        else
            ofs << ",\"time_min_ns\":null";
        ofs << ",\"time_max_ns\":" << agg.time_max_ns;
        ofs << ",\"subgroup_size\":" << agg.subgroup_size;
        ofs << ",\"subgroup_count\":" << agg.subgroup_count;
        ofs << ",\"hist_log_base_ns\":100";
        ofs << ",\"hist_log_factor\":1.4";
        ofs << ",\"hist_buckets\":[";
        for (int i = 0; i < RGY_CL_PERF_HIST_BUCKETS; i++) {
            if (i > 0) ofs << ",";
            ofs << agg.hist_log[i];
        }
        ofs << "]";
        ofs << "}\n";
    }
}

void RGYOpenCLPerfCollector::writeCommandsJsonl(const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs) return;

    for (const auto& [key, agg] : m_command_aggs) {
        ofs << "{";
        ofs << "\"command_name\":\"" << json_escape(key.command_name) << "\"";
        ofs << ",\"count\":" << agg.count;
        ofs << ",\"bytes_sum\":" << agg.bytes_sum;
        ofs << ",\"time_sum_ns\":" << agg.time_sum_ns;
        if (agg.count > 0 && agg.time_min_ns != UINT64_MAX)
            ofs << ",\"time_min_ns\":" << agg.time_min_ns;
        else
            ofs << ",\"time_min_ns\":null";
        ofs << ",\"time_max_ns\":" << agg.time_max_ns;
        ofs << ",\"queued_to_submit_sum_ns\":" << agg.queued_to_submit_sum_ns;
        ofs << ",\"submit_to_start_sum_ns\":" << agg.submit_to_start_sum_ns;
        ofs << ",\"queued_to_start_sum_ns\":" << agg.queued_to_start_sum_ns;
        ofs << ",\"host_time_sum_ns\":" << agg.host_time_sum_ns;
        if (agg.count > 0 && agg.host_time_min_ns != UINT64_MAX)
            ofs << ",\"host_time_min_ns\":" << agg.host_time_min_ns;
        else
            ofs << ",\"host_time_min_ns\":null";
        ofs << ",\"host_time_max_ns\":" << agg.host_time_max_ns;
        ofs << ",\"hist_log_base_ns\":100";
        ofs << ",\"hist_log_factor\":1.4";
        ofs << ",\"hist_buckets\":[";
        for (int i = 0; i < RGY_CL_PERF_HIST_BUCKETS; i++) {
            if (i > 0) ofs << ",";
            ofs << agg.hist_log[i];
        }
        ofs << "]";
        ofs << "}\n";
    }
}

void RGYOpenCLPerfCollector::writeAllocationsJsonl(const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs) return;

    for (const auto& [key, agg] : m_allocation_aggs) {
        ofs << "{";
        ofs << "\"operation_name\":\"" << json_escape(key.operation_name) << "\"";
        ofs << ",\"count\":" << agg.count;
        ofs << ",\"success_count\":" << agg.success_count;
        ofs << ",\"fail_count\":" << agg.fail_count;
        ofs << ",\"bytes_sum\":" << agg.bytes_sum;
        ofs << ",\"host_time_sum_ns\":" << agg.host_time_sum_ns;
        if (agg.count > 0 && agg.host_time_min_ns != UINT64_MAX)
            ofs << ",\"host_time_min_ns\":" << agg.host_time_min_ns;
        else
            ofs << ",\"host_time_min_ns\":null";
        ofs << ",\"host_time_max_ns\":" << agg.host_time_max_ns;
        ofs << "}\n";
    }
}

void RGYOpenCLPerfCollector::writeMetaJson(const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs) return;

    // デバイス名・ドライバは programs の最初のものから取る
    std::string device_name, driver_version;
    int ip_version = -1;
    if (!m_programs.empty()) {
        device_name    = m_programs[0].device_name;
        driver_version = m_programs[0].driver_version;
        ip_version     = m_programs[0].device_ip_version;
    }

    // device_arch_hint / ocloc_device_hint: ip_version から簡易マップ
    // (詳細は arch_table.json に委ねる。ここでは代表的な値のみ)
    std::string arch_hint = "unknown";
    std::string ocloc_hint = "unknown";
    if (ip_version > 0) {
        // Xe2-HPG (Battlemage G21): 0x12F0xxxx
        if ((ip_version >> 16) == 0x12F0) {
            arch_hint  = "xe2-hpg";
            ocloc_hint = "bmg-g21";
        // Xe-HPG (Alchemist): 0x0C74xxxx
        } else if ((ip_version >> 16) == 0x0C74) {
            arch_hint  = "xe-hpg";
            ocloc_hint = "acm-g10";
        // Xe-LPG (Meteor Lake): 0x0308xxxx
        } else if ((ip_version >> 16) == 0x0308) {
            arch_hint  = "xe-lpg";
            ocloc_hint = "mtl-h-gt2";
        }
    }

    ofs << "{\n";
    ofs << "  \"encoder\":\""
#if ENCODER_QSV
        "QSVEncC"
#elif ENCODER_VCEENC
        "VCEEncC"
#elif ENCODER_MPP
        "rkmppenc"
#else
        "unknown"
#endif
        "\",\n";
    ofs << "  \"timestamp\":\"" << get_timestamp() << "\",\n";
    ofs << "  \"device_name\":\"" << json_escape(device_name) << "\",\n";
    ofs << "  \"driver_version\":\"" << json_escape(driver_version) << "\",\n";
    if (ip_version >= 0)
        ofs << "  \"device_ip_version\":" << ip_version << ",\n";
    else
        ofs << "  \"device_ip_version\":null,\n";
    ofs << "  \"device_arch_hint\":\"" << json_escape(arch_hint) << "\",\n";
    ofs << "  \"ocloc_device_hint\":\"" << json_escape(ocloc_hint) << "\"\n";
    ofs << "}\n";
}

#endif // ENABLE_OPENCL
