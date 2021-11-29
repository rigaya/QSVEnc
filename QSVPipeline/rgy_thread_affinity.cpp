// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2021 rigaya
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

#include <sstream>
#include <vector>
#include "rgy_thread_affinity.h"
#include "rgy_osdep.h"
#if defined(_WIN32) || defined(_WIN64)
#include <tlhelp32.h>
#endif //#if defined(_WIN32) || defined(_WIN64)
#include "cpu_info.h"

const TCHAR* rgy_thread_priority_mode_to_str(RGYThreadPriority mode) {
    for (const auto& p : RGY_THREAD_PRIORITY_STR) {
        if (p.first == mode) return p.second;
    }
    return nullptr;
}
RGYThreadPriority rgy_str_to_thread_priority_mode(const TCHAR* str) {
    tstring target(str);
    for (const auto& p : RGY_THREAD_PRIORITY_STR) {
        if (target == p.second) return p.first;
    }
    return RGYThreadPriority::Unknwon;
}

const TCHAR* rgy_thread_power_throttoling_mode_to_str(RGYThreadPowerThrottolingMode mode) {
    for (const auto& p : RGY_THREAD_POWER_THROTTOLING_MODE_STR) {
        if (p.first == mode) return p.second;
    }
    return nullptr;
}

RGYThreadPowerThrottolingMode rgy_str_to_thread_power_throttoling_mode(const TCHAR* str) {
    tstring target(str);
    for (const auto& p : RGY_THREAD_POWER_THROTTOLING_MODE_STR) {
        if (target == p.second) return p.first;
    }
    return RGYThreadPowerThrottolingMode::END;
}

RGYThreadAffinity::RGYThreadAffinity() : mode(), custom(std::numeric_limits<decltype(custom)>::max()) {};

RGYThreadAffinity::RGYThreadAffinity(RGYThreadAffinityMode affinityMode) : mode(affinityMode), custom(std::numeric_limits<decltype(custom)>::max()) {};

RGYThreadAffinity::RGYThreadAffinity(RGYThreadAffinityMode m, uint64_t customAffinity) : mode(m), custom(customAffinity) {};

tstring RGYThreadAffinity::to_string() const {
    if (mode == RGYThreadAffinityMode::CUSTOM) {
        TCHAR buf[64];
        _stprintf_s(buf, _T("0x%llx"), custom);
        return buf;
    }
    auto modeStr = rgy_thread_affnity_mode_to_str(mode);
    if (   mode == RGYThreadAffinityMode::LOGICAL
        || mode == RGYThreadAffinityMode::PHYSICAL
        || mode == RGYThreadAffinityMode::CACHEL2
        || mode == RGYThreadAffinityMode::CACHEL3
    ) {
        const auto cpu_info = get_cpu_info();
        int targetCount = 0;
        if (mode == RGYThreadAffinityMode::LOGICAL) {
            targetCount = cpu_info.logical_cores;
        } else if (mode == RGYThreadAffinityMode::PHYSICAL) {
            targetCount = cpu_info.physical_cores;
        } else if (mode == RGYThreadAffinityMode::CACHEL2) {
            targetCount = cpu_info.cache_count[1];
        } else if (mode == RGYThreadAffinityMode::CACHEL3) {
            targetCount = cpu_info.cache_count[2];
        }
        std::basic_stringstream<TCHAR> tmp;
        for (int id = 0; id < targetCount; id++) {
            const auto target = 1llu << id;
            if (target & custom) {
                tmp << _T(":") << id;
            }
        }
        if (!tmp.str().empty()) {
            return modeStr + tstring(_T("#")) + tmp.str().substr(1);
        } else {
            return modeStr;
        }
    }
    return modeStr;
}

const TCHAR *rgy_thread_affnity_mode_to_str(RGYThreadAffinityMode mode) {
    for (const auto& p : RGY_THREAD_AFFINITY_MODE_STR) {
        if (p.second == mode) return p.first;
    }
    return nullptr;
}

RGYThreadAffinityMode rgy_str_to_thread_affnity_mode(const TCHAR *str) {
    tstring target(str);
    for (const auto& p : RGY_THREAD_AFFINITY_MODE_STR) {
        if (target == p.first) return p.second;
    }
    return RGYThreadAffinityMode::END;
}

bool RGYThreadAffinity::operator==(const RGYThreadAffinity &x) const {
    return mode == x.mode
        && custom == x.custom;
}
bool RGYThreadAffinity::operator!=(const RGYThreadAffinity &x) const {
    return !(*this == x);
}

uint64_t RGYThreadAffinity::getMask(int idx) const {
    return selectMaskFromLowerBit(getMask(), idx);
}

uint64_t RGYThreadAffinity::getMask() const {
    uint64_t mask = 0;
    const auto cpu_info = get_cpu_info();
    switch (mode) {
    case RGYThreadAffinityMode::PCORE:
    case RGYThreadAffinityMode::ECORE: {
        auto maskSelected = cpu_info.maskSystem;
        if (mode == RGYThreadAffinityMode::PCORE && cpu_info.maskCoreP) maskSelected = cpu_info.maskCoreP;
        if (mode == RGYThreadAffinityMode::ECORE && cpu_info.maskCoreE) maskSelected = cpu_info.maskCoreE;
        int targetCore = 0;
        for (int i = 0; i < cpu_info.physical_cores; i++) {
            const auto target_i = get_mask(&cpu_info, RGYUnitType::Core, (int)RGYCoreType::Physical, i);
            if (maskSelected & target_i) { // PCoreであるか?
                const auto target_core_mask = 1llu << targetCore;
                if (target_core_mask & custom) { // customで指定のコアであるか?
                    mask |= target_i;
                }
                targetCore++;
            }
        }
    } break;
    case RGYThreadAffinityMode::LOGICAL:
        for (int i = 0; i < cpu_info.logical_cores; i++) {
            const auto target = 1llu << i;
            if (target & custom) {
                mask |= get_mask(&cpu_info, RGYUnitType::Core, (int)RGYCoreType::Logical, i);
            }
        }
        break;
    case RGYThreadAffinityMode::PHYSICAL:
        for (int i = 0; i < cpu_info.physical_cores; i++) {
            const auto target = 1llu << i;
            if (target & custom) {
                mask |= get_mask(&cpu_info, RGYUnitType::Core, (int)RGYCoreType::Physical, i);
            }
        }
        break;
    case RGYThreadAffinityMode::CACHEL2:
        for (int i = 0; i < cpu_info.cache_count[1]; i++) {
            const auto target = 1llu << i;
            if (target & custom) {
                mask |= get_mask(&cpu_info, RGYUnitType::Cache, (int)RGYCacheLevel::L2, i);
            }
        }
        break;
    case RGYThreadAffinityMode::CACHEL3:
        for (int i = 0; i < cpu_info.cache_count[2]; i++) {
            const auto target = 1llu << i;
            if (target & custom) {
                mask |= get_mask(&cpu_info, RGYUnitType::Cache, (int)RGYCacheLevel::L3, i);
            }
        }
        break;
    case RGYThreadAffinityMode::CUSTOM: mask = (custom) ? custom & cpu_info.maskSystem : cpu_info.maskSystem; break;
    case RGYThreadAffinityMode::ALL:
    default: mask = cpu_info.maskSystem; break;
    }
    return (mask) ? mask : std::numeric_limits<decltype(mask)>::max();
}

RGYParamThread::RGYParamThread() :
    affinity(),
    priority(RGYThreadPriority::Normal),
    throttling(RGYThreadPowerThrottolingMode::Auto) {

}

uint32_t RGYParamThread::getPriorityCalss() {
    static const std::array<std::pair<RGYThreadPriority, uint32_t>, RGY_THREAD_PRIORITY_STR.size()> RGY_THREAD_PRIORITY_CLASS = {
        std::pair<RGYThreadPriority, int>{ RGYThreadPriority::BackgroundBeign, PROCESS_MODE_BACKGROUND_BEGIN},
        std::pair<RGYThreadPriority, int>{ RGYThreadPriority::Idle,            IDLE_PRIORITY_CLASS},
        std::pair<RGYThreadPriority, int>{ RGYThreadPriority::Lowest,          IDLE_PRIORITY_CLASS},
        std::pair<RGYThreadPriority, int>{ RGYThreadPriority::BelowNormal,     BELOW_NORMAL_PRIORITY_CLASS},
        std::pair<RGYThreadPriority, int>{ RGYThreadPriority::Normal,          NORMAL_PRIORITY_CLASS},
        std::pair<RGYThreadPriority, int>{ RGYThreadPriority::AboveNormal,     ABOVE_NORMAL_PRIORITY_CLASS},
        std::pair<RGYThreadPriority, int>{ RGYThreadPriority::Highest,         HIGH_PRIORITY_CLASS}
    };
    for (const auto& p : RGY_THREAD_PRIORITY_CLASS) {
        if (p.first == priority) return p.second;
    }
    return 0u;
}

tstring RGYParamThread::to_string(RGYParamThreadType type) const {
    switch (type) {
    case RGYParamThreadType::affinity: return affinity.to_string();
    case RGYParamThreadType::priority: return rgy_thread_priority_mode_to_str(priority);
    case RGYParamThreadType::throttoling: return rgy_thread_power_throttoling_mode_to_str(throttling);
    case RGYParamThreadType::all:
    default: {
        tstring str = _T("affinity=");
        str += affinity.to_string();
        str += _T(",priority=");
        str += rgy_thread_priority_mode_to_str(priority);
        str += _T(",throttoling=");
        str += rgy_thread_power_throttoling_mode_to_str(throttling);
        return str;
    }
    }
}

tstring RGYParamThread::desc() const {
    tstring str;
    str += affinity.to_string();
    str += _T(" (0x");
    TCHAR buf[64];
    _stprintf_s(buf, _T("0x%llx"), affinity.getMask());
    str += buf;
    str += _T("), priority=");
    str += rgy_thread_priority_mode_to_str(priority);
    str += _T(", throttoling=");
    str += rgy_thread_power_throttoling_mode_to_str(throttling);
    return str;
}

bool RGYParamThread::operator==(const RGYParamThread& x) const {
    return affinity == x.affinity
        && priority == x.priority
        && throttling == x.throttling;
}
bool RGYParamThread::operator!=(const RGYParamThread& x) const {
    return !(*this == x);
}

void RGYParamThread::set(RGYThreadAffinity affinity_, RGYThreadPriority priority_, RGYThreadPowerThrottolingMode throttling_) {
    affinity = affinity_;
    priority = priority_;
    throttling = throttling_;
}

bool RGYParamThread::apply(void *threadHandle) const {
    bool ret = true;
    if (affinity.mode != RGYThreadAffinityMode::ALL) {
        SetThreadAffinityMask(threadHandle, affinity.getMask());
    }
    if (priority != RGYThreadPriority::Normal) {
        ret &= !!SetThreadPriority(threadHandle, (int)priority);
    }
    if (throttling != RGYThreadPowerThrottolingMode::Auto) {
        ret &= SetThreadPowerThrottolingMode(threadHandle, throttling);
    }
    return ret;
}

RGYParamThreads::RGYParamThreads() :
    process(),
    main(),
    dec(),
    enc(),
    csp(),
    input(),
    output(),
    audio(),
    perfmonitor(),
    videoquality() {
    apply_auto();
}

void RGYParamThreads::apply_auto() {
    perfmonitor.priority = RGYThreadPriority::BackgroundBeign;

    auto set_throttoling_auto = [](RGYParamThread& type, RGYThreadPowerThrottolingMode mode) { if (type.throttling == RGYThreadPowerThrottolingMode::Auto) { type.throttling = mode; }};
    set_throttoling_auto(dec, RGYThreadPowerThrottolingMode::Enabled);
    set_throttoling_auto(enc, RGYThreadPowerThrottolingMode::Enabled);
    set_throttoling_auto(output, RGYThreadPowerThrottolingMode::Enabled);
    set_throttoling_auto(perfmonitor, RGYThreadPowerThrottolingMode::Enabled);
    set_throttoling_auto(videoquality, RGYThreadPowerThrottolingMode::Enabled);
    // そのほかはUnsetにする
    for (int i = (int)RGYThreadType::ALL + 1; i < (int)RGYThreadType::END; i++) {
        set_throttoling_auto(get((RGYThreadType)i), RGYThreadPowerThrottolingMode::Unset);
    }
}

const TCHAR *rgy_thread_type_to_str(RGYThreadType type) {
    for (const auto& p : RGY_THREAD_TYPE_STR) {
        if (p.first == type) return p.second;
    }
    return nullptr;
}

RGYParamThread& RGYParamThreads::get(RGYThreadType type) {
    switch (type) {
    case RGYThreadType::MAIN: return main;
    case RGYThreadType::DEC:  return dec;
    case RGYThreadType::ENC:  return enc;
    case RGYThreadType::CSP:  return csp;
    case RGYThreadType::INPUT: return input;
    case RGYThreadType::OUTUT: return output;
    case RGYThreadType::AUDIO: return audio;
    case RGYThreadType::PERF_MONITOR: return perfmonitor;
    case RGYThreadType::VIDEO_QUALITY: return videoquality;
    case RGYThreadType::PROCESS: return process;
    case RGYThreadType::ALL:
    default: return process;
    }
}

const RGYParamThread& RGYParamThreads::get(RGYThreadType type) const {
    switch (type) {
    case RGYThreadType::MAIN: return main;
    case RGYThreadType::DEC:  return dec;
    case RGYThreadType::ENC:  return enc;
    case RGYThreadType::CSP:  return csp;
    case RGYThreadType::INPUT: return input;
    case RGYThreadType::OUTUT: return output;
    case RGYThreadType::AUDIO: return audio;
    case RGYThreadType::PERF_MONITOR: return perfmonitor;
    case RGYThreadType::VIDEO_QUALITY: return videoquality;
    case RGYThreadType::PROCESS: return process;
    case RGYThreadType::ALL:
    default: return process;
    }
}

void RGYParamThreads::set(const RGYThreadAffinity affinity, RGYThreadType type) {
    if (type == RGYThreadType::ALL) {
        for (int i = (int)RGYThreadType::ALL + 1; i < (int)RGYThreadType::END; i++) {
            get((RGYThreadType)i).affinity = affinity;
        }
    } else {
        get(type).affinity = affinity;
    }
}

void RGYParamThreads::set(const RGYThreadPriority priority, RGYThreadType type) {
    if (type == RGYThreadType::ALL) {
        for (int i = (int)RGYThreadType::ALL + 1; i < (int)RGYThreadType::END; i++) {
            get((RGYThreadType)i).priority = priority;
        }
    } else {
        get(type).priority = priority;
    }
}

void RGYParamThreads::set(const RGYThreadPowerThrottolingMode mode, RGYThreadType type) {
    if (type == RGYThreadType::ALL) {
        for (int i = (int)RGYThreadType::ALL + 1; i < (int)RGYThreadType::END; i++) {
            get((RGYThreadType)i).throttling = mode;
        }
    } else {
        get(type).throttling = mode;
    }
}

tstring RGYParamThreads::to_string(RGYParamThreadType type) const {
    std::basic_stringstream<TCHAR> tmp;
#define RGY_THREAD_AFF_ADD_TYPE(TYPE, VAR) { tmp << _T(",") << rgy_thread_type_to_str(TYPE) << _T("=") << VAR.to_string(type); }
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::PROCESS, process);
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::MAIN, main);
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::DEC, dec);
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::ENC, enc);
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::INPUT, input);
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::OUTUT, output);
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::AUDIO, audio);
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::PERF_MONITOR, perfmonitor);
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::VIDEO_QUALITY, videoquality);
#undef LOG_LEVEL_ADD_TYPE
    return tmp.str().substr(1);
}

bool RGYParamThreads::operator==(const RGYParamThreads&x) const {
    return process == x.process
        && main == x.main
        && dec == x.dec
        && enc == x.enc
        && csp == x.csp
        && input == x.input
        && output == x.output
        && audio == x.audio
        && perfmonitor == x.perfmonitor
        && videoquality == x.videoquality;
}
bool RGYParamThreads::operator!=(const RGYParamThreads&x) const {
    return !(*this == x);
}

#pragma warning(push)
#pragma warning(disable: 4146) //warning C4146: 符号付きの値を代入する変数は、符号付き型にキャストしなければなりません。
uint64_t selectMaskFromLowerBit(uint64_t mask, const int idx) {
    int count = 0;
    uint64_t ret = 0;
    do {
        mask &= (~ret);
        ret = (uint64_t)(mask & (-mask)); // select lowest bit
        count++;
    } while (count <= idx);
    return ret;
}
#pragma warning(pop)

#if defined(_WIN32) || defined(_WIN64)
static inline bool check_ptr_range(void *value, void *min, void *max) {
    return (min <= value && value <= max);
}

static const int ThreadQuerySetWin32StartAddress = 9;
typedef int (WINAPI* typeNtQueryInformationThread)(HANDLE, int, PVOID, ULONG, PULONG);

static void* GetThreadBeginAddress(const uint32_t TargetProcessId) {
    HMODULE hNtDll = NULL;
    typeNtQueryInformationThread NtQueryInformationThread = NULL;
    HANDLE hThread = NULL;
    ULONG length = 0;
    void* BeginAddress = NULL;

    if (   NULL != (hNtDll = LoadLibrary(_T("ntdll.dll")))
        && NULL != (NtQueryInformationThread = (typeNtQueryInformationThread)GetProcAddress(hNtDll, "NtQueryInformationThread"))
        && NULL != (hThread = OpenThread(THREAD_QUERY_INFORMATION, FALSE, TargetProcessId))) {
        NtQueryInformationThread(hThread, ThreadQuerySetWin32StartAddress, &BeginAddress, sizeof(BeginAddress), &length);
    }
    if (hNtDll)
        FreeLibrary(hNtDll);
    if (hThread)
        CloseHandle(hThread);
    return BeginAddress;
}

static inline std::vector<uint32_t> GetThreadList(const uint32_t TargetProcessId) {
    std::vector<uint32_t> ThreadList;
    HANDLE hSnapshot;

    if (INVALID_HANDLE_VALUE != (hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0x00))) {
        THREADENTRY32 te32 = { 0 };
        te32.dwSize = sizeof(THREADENTRY32);

        if (Thread32First(hSnapshot, &te32)) {
            do {
                if (te32.th32OwnerProcessID == TargetProcessId)
                    ThreadList.push_back(te32.th32ThreadID);
            } while (Thread32Next(hSnapshot, &te32));
        }
        CloseHandle(hSnapshot);
    }
    return ThreadList;
}

static inline std::vector<MODULEENTRY32> GetModuleList(const uint32_t TargetProcessId) {
    std::vector<MODULEENTRY32> ModuleList;
    HANDLE hSnapshot;

    if (INVALID_HANDLE_VALUE != (hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, TargetProcessId))) {
        MODULEENTRY32 me32 = { 0 };
        me32.dwSize = sizeof(MODULEENTRY32);

        if (Module32First(hSnapshot, &me32)) {
            do {
                ModuleList.push_back(me32);
            } while (Module32Next(hSnapshot, &me32));
        }
        CloseHandle(hSnapshot);
    }
    return ModuleList;
}

static bool SetThreadPriorityFromThreadId(const uint32_t TargetThreadId, const RGYThreadPriority ThreadPriority) {
    HANDLE hThread = OpenThread(THREAD_SET_INFORMATION, FALSE, TargetThreadId);
    if (hThread == NULL)
        return FALSE;
    BOOL ret = SetThreadPriority(hThread, (int)ThreadPriority);
    CloseHandle(hThread);
    return ret != 0;
}

bool SetThreadPriorityForModule(const uint32_t TargetProcessId, const TCHAR *TargetModule, const RGYThreadPriority ThreadPriority) {
    bool ret = true;
    const auto thread_list = GetThreadList(TargetProcessId);
    const auto module_list = GetModuleList(TargetProcessId);
    for (const auto thread_id : thread_list) {
        void* thread_address = GetThreadBeginAddress(thread_id);
        if (!thread_address) {
            ret = FALSE;
        } else {
            for (const auto& i_module : module_list) {
                if (check_ptr_range(thread_address, i_module.modBaseAddr, i_module.modBaseAddr + i_module.modBaseSize - 1)
                    && (TargetModule == nullptr || _tcsncicmp(TargetModule, i_module.szModule, _tcslen(TargetModule)) == 0)) {
                    ret &= !!SetThreadPriorityFromThreadId(thread_id, ThreadPriority);
                    break;
                }
            }
        }
    }
    return ret;
}

static bool SetThreadAffinityFromThreadId(const uint32_t TargetThreadId, const uint64_t ThreadAffinityMask) {
    HANDLE hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, TargetThreadId);
    if (hThread == NULL)
        return FALSE;
    auto ret = SetThreadAffinityMask(hThread, ThreadAffinityMask);
    CloseHandle(hThread);
    return (ret != 0);
}

bool SetThreadAffinityForModule(const uint32_t TargetProcessId, const TCHAR *TargetModule, const uint64_t ThreadAffinityMask) {
    bool ret = TRUE;
    const auto thread_list = GetThreadList(TargetProcessId);
    const auto module_list = GetModuleList(TargetProcessId);
    for (const auto thread_id : thread_list) {
        void* thread_address = GetThreadBeginAddress(thread_id);
        if (!thread_address) {
            ret = FALSE;
        } else {
            for (const auto& i_module : module_list) {
                if (check_ptr_range(thread_address, i_module.modBaseAddr, i_module.modBaseAddr + i_module.modBaseSize - 1)
                    && (TargetModule == nullptr || _tcsncicmp(TargetModule, i_module.szModule, _tcslen(TargetModule)) == 0)) {
                    ret &= !!SetThreadAffinityFromThreadId(thread_id, ThreadAffinityMask);
                    break;
                }
            }
        }
    }
    return ret;
}

bool SetThreadPowerThrottolingMode(void *threadHandle, const RGYThreadPowerThrottolingMode mode) {
    THREAD_POWER_THROTTLING_STATE throttlingState;
    RtlZeroMemory(&throttlingState, sizeof(throttlingState));
    throttlingState.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;

    switch (mode) {
    case RGYThreadPowerThrottolingMode::Enabled:
        throttlingState.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        throttlingState.StateMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        break;
    case RGYThreadPowerThrottolingMode::Disabled:
        throttlingState.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        throttlingState.StateMask = 0;
        break;
    case RGYThreadPowerThrottolingMode::Unset:
    case RGYThreadPowerThrottolingMode::Auto:
    default:
        throttlingState.ControlMask = 0;
        throttlingState.StateMask = 0;
        break;
    }
    return SetThreadInformation(threadHandle, ThreadPowerThrottling, &throttlingState, sizeof(throttlingState));
}

bool SetThreadPowerThrottolingModeForModule(const uint32_t TargetProcessId, const TCHAR* TargetModule, const RGYThreadPowerThrottolingMode mode) {
    bool ret = TRUE;
    const auto thread_list = GetThreadList(TargetProcessId);
    const auto module_list = GetModuleList(TargetProcessId);
    for (const auto thread_id : thread_list) {
        void* thread_address = GetThreadBeginAddress(thread_id);
        if (!thread_address) {
            ret = FALSE;
        }
        else {
            for (const auto& i_module : module_list) {
                if (check_ptr_range(thread_address, i_module.modBaseAddr, i_module.modBaseAddr + i_module.modBaseSize - 1)
                    && (TargetModule == nullptr || _tcsncicmp(TargetModule, i_module.szModule, _tcslen(TargetModule)) == 0)) {
                    HANDLE hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, thread_id);
                    if (hThread) {
                        ret &= !!SetThreadPowerThrottolingMode(hThread, mode);
                        CloseHandle(hThread);
                    }
                    break;
                }
            }
        }
    }
    return ret;
}
#else
bool SetThreadPriorityForModule(const uint32_t TargetProcessId, const TCHAR* TargetModule, const RGYThreadPriority ThreadPriority) {
    return false;
}
bool SetThreadAffinityForModule(const uint32_t TargetProcessId, const TCHAR* TargetModule, const uint64_t ThreadAffinityMask) {
    return false;
}
bool SetThreadPowerThrottolingMode(HANDLE threadHandle, const RGYThreadPowerThrottolingMode mode) {
    return false;
}
bool SetThreadPowerThrottolingModeForModule(const uint32_t TargetProcessId, const TCHAR* TargetModule, const RGYThreadPowerThrottolingMode mode) {
    return false;
}
#endif // #if defined(_WIN32) || defined(_WIN64)
