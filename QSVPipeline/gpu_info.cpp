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

static cl_int cl_create_kernel(cl_data_t *cl_data, const cl_func_t *cl) {
    cl_int ret = CL_SUCCESS;
    cl_data->contextCL = cl->createContext(0, 1, &cl_data->deviceID, NULL, NULL, &ret);
    if (CL_SUCCESS != ret)
        return ret;

    cl_data->commands = cl->createCommandQueue(cl_data->contextCL, cl_data->deviceID, NULL, &ret);
    if (CL_SUCCESS != ret)
        return ret;

    //OpenCLのカーネル用のコードはリソース埋め込みにしているので、それを呼び出し
    HRSRC hResource = NULL;
    HGLOBAL hResourceData = NULL;
    const char *clSourceFile = NULL;
    if (   NULL == (hResource = FindResource(NULL, _T("CLDATA"), _T("KERNEL_DATA")))
        || NULL == (hResourceData = LoadResource(NULL, hResource))
        || NULL == (clSourceFile = (const char *)LockResource(hResourceData))) {
        return 1;
    }
    size_t programLength = strlen(clSourceFile);
    cl_data->program = cl->createProgramWithSource(cl_data->contextCL, 1, (const char**)&clSourceFile, &programLength, &ret);
    if (CL_SUCCESS != ret)
        return ret;

    if (CL_SUCCESS != (ret = cl->buildProgram(cl_data->program, 1, &cl_data->deviceID, NULL, NULL, NULL))) {
        char buffer[2048];
        size_t length = 0;
        cl->getProgramBuildInfo(cl_data->program, cl_data->deviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
        fprintf(stderr, "%s\n", buffer);
        return ret;
    }
    cl_data->kernel = cl->createKernel(cl_data->program, "dummy_calc", &ret);
    if (CL_SUCCESS != ret)
        return ret;

    return ret;
}

static cl_int cl_calc(const cl_data_t *cl_data, const cl_func_t *cl) {
    using namespace std;
    const int LOOKAROUND = 10;
    const int BUFFER_X = 1024 * 8;
    const int BUFFER_Y = 1024;
    const size_t BUFFER_BYTE_SIZE = BUFFER_X * BUFFER_Y * sizeof(float);
    cl_int ret = CL_SUCCESS;
    cl_mem bufA = cl->createBuffer(cl_data->contextCL, CL_MEM_READ_ONLY,  BUFFER_BYTE_SIZE, NULL, &ret);
    cl_mem bufB = cl->createBuffer(cl_data->contextCL, CL_MEM_READ_ONLY,  BUFFER_BYTE_SIZE, NULL, &ret);
    cl_mem bufC = cl->createBuffer(cl_data->contextCL, CL_MEM_WRITE_ONLY, BUFFER_BYTE_SIZE, NULL, &ret);

    vector<float> arrayA(BUFFER_BYTE_SIZE);
    vector<float> arrayB(BUFFER_BYTE_SIZE);
    vector<float> arrayC(BUFFER_BYTE_SIZE, 0.0);

    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<float> random(0.0f, 10.0f);
    generate(arrayA.begin(), arrayA.end(), [&random, &mt]() { return random(mt); });
    generate(arrayB.begin(), arrayB.end(), [&random, &mt]() { return random(mt); });

    cl->enqueueWriteBuffer(cl_data->commands, bufA, CL_FALSE, 0, BUFFER_BYTE_SIZE, &arrayA[0], 0, NULL, NULL);
    cl->enqueueWriteBuffer(cl_data->commands, bufB, CL_FALSE, 0, BUFFER_BYTE_SIZE, &arrayB[0], 0, NULL, NULL);
    cl->setKernelArg(cl_data->kernel, 0, sizeof(cl_mem), &bufA);
    cl->setKernelArg(cl_data->kernel, 1, sizeof(cl_mem), &bufB);
    cl->setKernelArg(cl_data->kernel, 2, sizeof(cl_mem), &bufC);
    cl->setKernelArg(cl_data->kernel, 3, sizeof(cl_int), &BUFFER_X);
    cl->setKernelArg(cl_data->kernel, 4, sizeof(cl_int), &BUFFER_Y);
    cl->setKernelArg(cl_data->kernel, 5, sizeof(cl_int), &LOOKAROUND);

    size_t data_size = BUFFER_X * BUFFER_Y;
    cl->enqueueNDRangeKernel(cl_data->commands, cl_data->kernel, 1, 0, &data_size, NULL, 0, NULL, NULL);
    cl->enqueueReadBuffer(cl_data->commands, bufC, CL_TRUE, 0, BUFFER_BYTE_SIZE, &arrayC[0], 0, NULL, NULL);
    cl->finish(cl_data->commands);

    cl->releaseMemObject(bufA);
    cl->releaseMemObject(bufB);
    cl->releaseMemObject(bufC);

    return ret;
}

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

cl_int cl_get_driver_version(const cl_data_t *cl_data, const cl_func_t *cl, TCHAR *buffer, unsigned int buffer_size) {
    cl_int ret = CL_SUCCESS;
    char cl_info_buffer[1024] = { 0 };
    if (CL_SUCCESS == (ret = cl->getDeviceInfo(cl_data->deviceID, CL_DRIVER_VERSION, _countof(cl_info_buffer), cl_info_buffer, NULL))) {
        _tcscpy_s(buffer, buffer_size, to_tchar(cl_info_buffer).c_str());
    } else {
        _tcscpy_s(buffer, buffer_size, _T("Unknown"));
    }
    return ret;
}

static cl_int cl_create_info_string(cl_data_t *cl_data, const cl_func_t *cl, const IntelDeviceInfo *info, TCHAR *buffer, unsigned int buffer_size) {
    cl_int ret = CL_SUCCESS;

    char cl_info_buffer[1024] = { 0 };
    if (cl_data && CL_SUCCESS == (ret = cl->getDeviceInfo(cl_data->deviceID, CL_DEVICE_NAME, _countof(cl_info_buffer), cl_info_buffer, NULL))) {
        _tcscpy_s(buffer, buffer_size, to_tchar(cl_info_buffer).c_str());
    }

    int numEU = (info) ? info->EUCount : 0;
    if (numEU == 0 && cl_data && CL_SUCCESS == cl->getDeviceInfo(cl_data->deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, _countof(cl_info_buffer), cl_info_buffer, NULL)) {
        numEU = *(cl_uint *)cl_info_buffer;
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
    if (cl_data && CL_SUCCESS == cl->getDeviceInfo(cl_data->deviceID, CL_DRIVER_VERSION, _countof(cl_info_buffer), cl_info_buffer, NULL)) {
        _stprintf_s(buffer + _tcslen(buffer), buffer_size - _tcslen(buffer), _T(" (%s)"), to_tchar(cl_info_buffer).c_str());
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
