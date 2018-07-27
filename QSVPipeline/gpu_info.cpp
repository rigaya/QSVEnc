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
#include "cl_func.h"
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

static cl_int cl_create_info_string(cl_data_t *cl_data, const cl_func_t *cl, const IntelDeviceInfo *info, TCHAR *buffer, unsigned int buffer_size) {
    cl_int ret = CL_SUCCESS;

    if (cl_data) {
        ret = cl_get_device_name(cl_data, cl, buffer, buffer_size);
    }

    int numEU = (info) ? info->EUCount : 0;
    if (numEU == 0 && cl_data) {
        numEU = cl_get_device_max_compute_units(cl_data, cl);
    }
    if (numEU) {
        _stprintf_s(buffer + _tcslen(buffer), buffer_size - _tcslen(buffer), _T(" (%dEU)"), numEU);
    }

    int MaxFreqMHz = (info) ? info->GPUMaxFreqMHz : 0;
    int MinFreqMHz = (info) ? info->GPUMinFreqMHz : 0;
    if (MaxFreqMHz == 0 && cl_data) {
        MaxFreqMHz = cl_get_device_max_clock_frequency_mhz(cl_data, cl);
    }
    if (MaxFreqMHz && MinFreqMHz) {
        _stprintf_s(buffer + _tcslen(buffer), buffer_size - _tcslen(buffer), _T(" %d-%dMHz"), MinFreqMHz, MaxFreqMHz);
    } else if (MaxFreqMHz) {
        _stprintf_s(buffer + _tcslen(buffer), buffer_size - _tcslen(buffer), _T(" %dMHz"), MaxFreqMHz);
    }
    if (info && info->PackageTDP) {
        _stprintf_s(buffer + _tcslen(buffer), buffer_size - _tcslen(buffer), _T(" [%dW]"), info->PackageTDP);
    }
    TCHAR driver_ver[256] = { 0 };
    if (cl_data && CL_SUCCESS == cl_get_driver_version(cl_data, cl, driver_ver, _countof(driver_ver))) {
        _stprintf_s(buffer + _tcslen(buffer), buffer_size - _tcslen(buffer), _T(" (%s)"), driver_ver);
    }
    auto remove_string =[](TCHAR *target_str, const TCHAR *remove_str) {
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

#if defined(_WIN32) || defined(_WIN64)
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

#ifdef LIBVA_SUPPORT
#include "qsv_hw_va.h"

tstring getGPUInfoVA() {
    std::unique_ptr<CLibVA> va(CreateLibVA());
    return char_to_tstring(vaQueryVendorString(va->GetVADisplay()));
}
#endif

#pragma warning (push)
#pragma warning (disable: 4100)
int getGPUInfo(const char *VendorName, TCHAR *buffer, unsigned int buffer_size, bool driver_version_only) {
#if !ENABLE_OPENCL
#ifdef LIBVA_SUPPORT
    _stprintf_s(buffer, buffer_size, _T("Intel Graphics / Driver : %s"), getGPUInfoVA().c_str());
#else
    _stprintf_s(buffer, buffer_size, _T("Unknown (not compiled with OpenCL support)"));
#endif
    return 0;
#else
    int ret = CL_SUCCESS;
    cl_func_t cl = { 0 };
    cl_data_t data = { 0 };
    IntelDeviceInfo info = { 0 };

    bool opencl_error = false;
    bool intel_error = false;
    if (CL_SUCCESS != (ret = cl_get_func(&cl))) {
        _tcscpy_s(buffer, buffer_size, _T("Intel HD Graphics"));
        opencl_error = true;
    } else if (CL_SUCCESS != (ret = cl_get_platform_and_device(VendorName, CL_DEVICE_TYPE_GPU, &data, &cl))) {
        _tcscpy_s(buffer, buffer_size, _T("Intel HD Graphics"));
        opencl_error = true;
    }

    if (!driver_version_only && 0 != getIntelGPUInfo(&info)) {
        _tcscpy_s(buffer, buffer_size, _T("Failed to get GPU Info."));
        intel_error = true;
    }


    if (driver_version_only) {
        if (!opencl_error) {
            cl_get_driver_version(&data, &cl, buffer, buffer_size);
        }
    } else {
        if (!(opencl_error && intel_error)) {
            cl_create_info_string((opencl_error) ? NULL : &data, &cl, (intel_error) ? NULL : &info, buffer, buffer_size);
        }
    }
    cl_release(&data, &cl);
    return ret;
#endif // !ENABLE_OPENCL
}
#pragma warning (pop)
