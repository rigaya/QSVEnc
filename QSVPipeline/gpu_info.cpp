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

#include "rgy_tchar.h"
#include <string>
#include <vector>
#include <random>
#include <future>
#include <algorithm>
#include "rgy_opencl.h"
#include "DeviceId.h"
#include "rgy_osdep.h"
#include "rgy_util.h"

typedef struct IntelDeviceInfo {
    unsigned int GPUMemoryBytes;
    unsigned int GPUMaxFreqMHz;
    unsigned int GPUMinFreqMHz;
    unsigned int GTGeneration;
    unsigned int EUCount;
    unsigned int PackageTDP;
    unsigned int MaxFillRate;
} IntelDeviceInfo;

#if ENABLE_OPENCL

static std::basic_string<TCHAR> to_tchar(const char *string) {
#if UNICODE
    int required_length = MultiByteToWideChar(CP_ACP, 0, string, -1, NULL, 0);
    std::basic_string<TCHAR> str(1+required_length, _T('\0'));
    MultiByteToWideChar(CP_ACP, 0, string, -1, &str[0], (int)str.size());
#else
    std::basic_string<char> str = string;
#endif
    return str;
};


static cl_int cl_create_info_string(const RGYOpenCLDeviceInfo *clinfo, const IntelDeviceInfo *info, TCHAR *buffer, unsigned int buffer_size) {
    cl_int ret = CL_SUCCESS;

    std::string str = (clinfo) ? clinfo->name : "";

    int numEU = (info) ? info->EUCount : 0;
    if (numEU == 0 && clinfo) {
        numEU = clinfo->max_compute_units;
    }
    if (numEU > 0) {
        str += strsprintf(" (%dEU)", numEU);
    }

    int MaxFreqMHz = (info) ? info->GPUMaxFreqMHz : 0;
    int MinFreqMHz = (info) ? info->GPUMinFreqMHz : 0;
    if (MaxFreqMHz == 0 && clinfo) {
        MaxFreqMHz = clinfo->max_clock_frequency;
    }
    if (MaxFreqMHz && MinFreqMHz) {
        str += strsprintf(" %d-%dMHz", MinFreqMHz, MaxFreqMHz);
    } else if (MaxFreqMHz) {
        str += strsprintf(" %dMHz", MaxFreqMHz);
    }
    if (info && info->PackageTDP > 0) {
        str += strsprintf(" [%dW]", info->PackageTDP);
    }
    if (clinfo && clinfo->driver_version.length() > 0) {
        str += " (" + clinfo->driver_version + ")";
    }
    _tcscpy_s(buffer, buffer_size, to_tchar(str.c_str()).c_str());

    auto remove_string = [](TCHAR *target_str, const TCHAR *remove_str) {
        TCHAR *ptr = _tcsstr(target_str, remove_str);
        if (nullptr != ptr) {
            memmove(ptr, ptr + _tcslen(remove_str), (_tcslen(ptr) - _tcslen(remove_str) + 1) *  sizeof(target_str[0]));
        }
    };
    remove_string(buffer, _T("(R)"));
    remove_string(buffer, _T("(TM)"));
    return ret;
}

#endif //ENABLE_OPENCL

#if (defined(_WIN32) || defined(_WIN64)) && !FOR_AUO
int getIntelGPUInfo(IntelDeviceInfo *info) {
    memset(info, 0, sizeof(info[0]));

    unsigned int VendorId, DeviceId, VideoMemory;
    if (!getGraphicsDeviceInfo(&VendorId, &DeviceId, &VideoMemory)) {
        return 1;
    }
    info->GPUMemoryBytes = VideoMemory;

    IntelDeviceInfoHeader intelDeviceInfoHeader = { 0 };
    char intelDeviceInfoBuffer[1024];
    if (GGF_SUCCESS != getIntelDeviceInfo(VendorId, &intelDeviceInfoHeader, &intelDeviceInfoBuffer)) {
        return 1;
    }

    IntelDeviceInfoV2 intelDeviceInfo = { 0 };
    memcpy(&intelDeviceInfo, intelDeviceInfoBuffer, intelDeviceInfoHeader.Size);
    info->GPUMaxFreqMHz = intelDeviceInfo.GPUMaxFreq;
    info->GPUMinFreqMHz = intelDeviceInfo.GPUMinFreq;
    if (intelDeviceInfoHeader.Version == 2) {
        info->EUCount      = intelDeviceInfo.EUCount;
        info->GTGeneration = intelDeviceInfo.GTGeneration;
        info->MaxFillRate  = intelDeviceInfo.MaxFillRate;
        info->PackageTDP   = intelDeviceInfo.PackageTDP;
    }
    return 0;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

#if LIBVA_SUPPORT
#include "qsv_hw_va.h"

tstring getGPUInfoVA() {
    std::unique_ptr<CLibVA> va(CreateLibVA());
    return char_to_tstring(vaQueryVendorString(va->GetVADisplay()));
}
#endif

#pragma warning (push)
#pragma warning (disable: 4100)
int getGPUInfo(const char *VendorName, TCHAR *buffer, unsigned int buffer_size, bool driver_version_only) {
#if LIBVA_SUPPORT
    _stprintf_s(buffer, buffer_size, _T("Intel Graphics / Driver : %s"), getGPUInfoVA().c_str());
    return 0;
#elif ENABLE_OPENCL
    int ret = CL_SUCCESS;
    RGYOpenCL cl;
    std::shared_ptr<RGYOpenCLPlatform> platform;
    auto platforms = cl.getPlatforms(VendorName);
    for (auto& p : platforms) {
        if (p->createDeviceList(CL_DEVICE_TYPE_GPU) == RGY_ERR_NONE && p->devs().size() > 0) {
            platform = p;
            break;
        }
    }
    if (driver_version_only) {
        if (platform) {
            _tcscpy_s(buffer, buffer_size, to_tchar(platform->dev(0).info().driver_version.c_str()).c_str());
            return 0;
        } else {
            _tcscpy_s(buffer, buffer_size, _T("Unknown"));
            return 1;
        }
    }

#if !FOR_AUO
    IntelDeviceInfo info = { 0 };
    const auto intelInfoRet = getIntelGPUInfo(&info);
    IntelDeviceInfo* intelinfoptr = (intelInfoRet == 0) ? &info : nullptr;
#else
    IntelDeviceInfo* intelinfoptr = nullptr;
#endif

    RGYOpenCLDeviceInfo clinfo;
    if (platform) {
        clinfo = platform->dev(0).info();
    }
    cl_create_info_string((platform) ? &clinfo : nullptr, intelinfoptr, buffer, buffer_size);
    return ret;
#else
    buffer[0] = _T('\0');
    return 1;
#endif // !ENABLE_OPENCL
}
#pragma warning (pop)
