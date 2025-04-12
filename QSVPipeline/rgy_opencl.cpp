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

#include "rgy_tchar.h"
#include <vector>
#include <atomic>
#include <fstream>
#include "rgy_osdep.h"
#define CL_EXTERN
#include "rgy_opencl.h"
#include "rgy_resource.h"
#include "rgy_filesystem.h"

#if ENABLE_OPENCL

#ifndef CL_DEVICE_IP_VERSION_INTEL
#define CL_DEVICE_IP_VERSION_INTEL                          0x4250
#define CL_DEVICE_ID_INTEL                                  0x4251
#define CL_DEVICE_NUM_SLICES_INTEL                          0x4252
#define CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL            0x4253
#define CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL               0x4254
#define CL_DEVICE_NUM_THREADS_PER_EU_INTEL                  0x4255
#define CL_DEVICE_FEATURE_CAPABILITIES_INTEL                0x4256
#endif //#ifndef CL_DEVICE_IP_VERSION_INTEL

#if ENCODER_VCEENC
#define CL_DEVICE_PROFILING_TIMER_OFFSET_AMD            0x4036
#define CL_DEVICE_TOPOLOGY_AMD                          0x4037
#define CL_DEVICE_BOARD_NAME_AMD                        0x4038
#define CL_DEVICE_GLOBAL_FREE_MEMORY_AMD                0x4039
#define CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD             0x4040
#define CL_DEVICE_SIMD_WIDTH_AMD                        0x4041
#define CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD            0x4042
#define CL_DEVICE_WAVEFRONT_WIDTH_AMD                   0x4043
#define CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD               0x4044
#define CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD          0x4045
#define CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD     0x4046
#define CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD   0x4047
#define CL_DEVICE_LOCAL_MEM_BANKS_AMD                   0x4048
#define CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD            0x4049
#define CL_DEVICE_GFXIP_MAJOR_AMD                       0x404A
#define CL_DEVICE_GFXIP_MINOR_AMD                       0x404B
#define CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD            0x404C
#define CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_AMD         0x4030
#define CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD               0x4031
#define CL_DEVICE_PREFERRED_CONSTANT_BUFFER_SIZE_AMD    0x4033
#define CL_DEVICE_PCIE_ID_AMD                           0x4034
#endif


#define CL_LOG(level, ...)  { if (m_log) { m_log->write(level, RGY_LOGT_OPENCL, __VA_ARGS__); } }

HMODULE RGYOpenCL::openCLHandle = nullptr;

//OpenCLのドライバは場合によってはクラッシュする可能性がある
//クラッシュしたことがあれば、このフラグを立て、以降OpenCLを使用しないようにする
bool RGYOpenCL::openCLCrush = false;

static void to_tchar(TCHAR *buf, uint32_t buf_size, const char *string) {
#if UNICODE
    MultiByteToWideChar(CP_ACP, 0, string, -1, buf, buf_size);
#else
    strcpy_s(buf, buf_size, string);
#endif
};

static inline const char *strichr(const char *str, int c) {
    c = tolower(c);
    for (; *str; str++)
        if (c == tolower(*str))
            return str;
    return NULL;
}
static inline const char *stristr(const char *str, const char *substr) {
    size_t len = 0;
    if (substr && (len = strlen(substr)) != 0)
        for (; (str = strichr(str, substr[0])) != NULL; str++)
            if (_strnicmp(str, substr, len) == 0)
                return str;
    return NULL;
}

static std::string uuidToString(const void *uuid) {
    std::string str;
    const uint8_t *buf = (const uint8_t *)uuid;
    for (size_t i = 0; i < CL_UUID_SIZE_KHR; ++i) {
        char tmp[4];
        sprintf_s(tmp, "%02x", buf[i]);
        str += tmp;
    }
    return str;
};

static std::string luidToString(const void *uuid) {
    std::string str;
    const uint8_t *buf = (const uint8_t *)uuid;
    for (size_t i = 0; i < CL_LUID_SIZE_KHR; ++i) {
        char tmp[4];
        sprintf_s(tmp, "%02x", buf[i]);
        str += tmp;
    }
    return str;
};

static bool checkVendor(const char *str, const char *VendorName) {
    if (VendorName == nullptr) {
        return true;
    }
    if (stristr(str, VendorName) != nullptr)
        return true;
    if (stristr(VendorName, "AMD") != nullptr)
        return stristr(str, "Advanced Micro Devices") != nullptr;
    return false;
}


template<typename Functor, typename Target, typename T>
inline cl_int clGetInfo(Functor f, Target target, cl_uint name, T *param) {
    return f(target, name, sizeof(T), param, NULL);
}

template <typename Func, typename Target, typename T>
inline cl_int clGetInfo(Func f, Target target, cl_uint name, vector<T> *param) {
    size_t required;
    cl_int err = f(target, name, 0, NULL, &required);
    if (err != CL_SUCCESS) {
        return err;
    }
    const size_t elements = required / sizeof(T);

    // Temporary to avoid changing param on an error
    vector<T> localData(elements);
    err = f(target, name, required, localData.data(), NULL);
    if (err != CL_SUCCESS) {
        return err;
    }
    if (param) {
        *param = std::move(localData);
    }

    return CL_SUCCESS;
}

// Specialized GetInfoHelper for string params
template <typename Func, typename Target>
inline cl_int clGetInfo(Func f, Target target, cl_uint name, std::string *param) {
    size_t required;
    cl_int err = f(target, name, 0, NULL, &required);
    if (err != CL_SUCCESS) {
        return err;
    }

    // std::string has a constant data member
    // a char vector does not
    if (required > 0) {
        vector<char> value(required+1, '\0');
        err = f(target, name, required, value.data(), NULL);
        if (err != CL_SUCCESS) {
            return err;
        }
        if (param) {
            *param = value.data();
        }
    } else if (param) {
        param->assign("");
    }
    return CL_SUCCESS;
}

template <typename Func, typename Target, size_t N>
inline cl_int clGetInfo(Func f, Target target, cl_uint name, std::array<size_t, N> *param) {
    size_t required;
    cl_int err = f(target, name, 0, NULL, &required);
    if (err != CL_SUCCESS) {
        return err;
    }

    size_t elements = required / sizeof(size_t);
    vector<size_t> value(elements, 0);

    err = f(target, name, required, value.data(), NULL);
    if (err != CL_SUCCESS) {
        return err;
    }

    if (elements > N) {
        elements = N;
    }
    for (size_t i = 0; i < elements; ++i) {
        (*param)[i] = value[i];
    }

    return CL_SUCCESS;
}

static const auto RGY_ERR_TO_OPENCL = make_array<std::pair<RGY_ERR, cl_int>>(
    std::make_pair(RGY_ERR_NONE, CL_SUCCESS),
    std::make_pair(RGY_ERR_DEVICE_NOT_FOUND, CL_DEVICE_NOT_FOUND),
    std::make_pair(RGY_ERR_DEVICE_NOT_AVAILABLE, CL_DEVICE_NOT_AVAILABLE),
    std::make_pair(RGY_ERR_COMPILER_NOT_AVAILABLE, CL_COMPILER_NOT_AVAILABLE),
    std::make_pair(RGY_ERR_MEM_OBJECT_ALLOCATION_FAILURE, CL_MEM_OBJECT_ALLOCATION_FAILURE),
    std::make_pair(RGY_ERR_OUT_OF_RESOURCES, CL_OUT_OF_RESOURCES),
    std::make_pair(RGY_ERR_OUT_OF_HOST_MEMORY, CL_OUT_OF_HOST_MEMORY),
    std::make_pair(RGY_ERR_PROFILING_INFO_NOT_AVAILABLE, CL_PROFILING_INFO_NOT_AVAILABLE),
    std::make_pair(RGY_ERR_MEM_COPY_OVERLAP, CL_MEM_COPY_OVERLAP),
    std::make_pair(RGY_ERR_IMAGE_FORMAT_MISMATCH, CL_IMAGE_FORMAT_MISMATCH),
    std::make_pair(RGY_ERR_IMAGE_FORMAT_NOT_SUPPORTED, CL_IMAGE_FORMAT_NOT_SUPPORTED),
    std::make_pair(RGY_ERR_BUILD_PROGRAM_FAILURE, CL_BUILD_PROGRAM_FAILURE),
    std::make_pair(RGY_ERR_MAP_FAILURE, CL_MAP_FAILURE),
    std::make_pair(RGY_ERR_COMPILE_PROGRAM_FAILURE, CL_COMPILE_PROGRAM_FAILURE),
    std::make_pair(RGY_ERR_INVALID_CALL, CL_INVALID_VALUE),
    std::make_pair(RGY_ERR_INVALID_DEVICE_TYPE, CL_INVALID_DEVICE_TYPE),
    std::make_pair(RGY_ERR_INVALID_PLATFORM, CL_INVALID_PLATFORM),
    std::make_pair(RGY_ERR_INVALID_DEVICE, CL_INVALID_DEVICE),
    std::make_pair(RGY_ERR_INVALID_CONTEXT, CL_INVALID_CONTEXT),
    std::make_pair(RGY_ERR_INVALID_QUEUE_PROPERTIES, CL_INVALID_QUEUE_PROPERTIES),
    std::make_pair(RGY_ERR_INVALID_COMMAND_QUEUE, CL_INVALID_COMMAND_QUEUE),
    std::make_pair(RGY_ERR_INVALID_HOST_PTR, CL_INVALID_HOST_PTR),
    std::make_pair(RGY_ERR_INVALID_MEM_OBJECT, CL_INVALID_MEM_OBJECT),
    std::make_pair(RGY_ERR_INVALID_IMAGE_FORMAT_DESCRIPTOR, CL_INVALID_IMAGE_FORMAT_DESCRIPTOR),
    std::make_pair(RGY_ERR_INVALID_RESOLUTION, CL_INVALID_IMAGE_SIZE),
    std::make_pair(RGY_ERR_INVALID_SAMPLER, CL_INVALID_SAMPLER),
    std::make_pair(RGY_ERR_INVALID_BINARY, CL_INVALID_BINARY),
    std::make_pair(RGY_ERR_INVALID_BUILD_OPTIONS, CL_INVALID_BUILD_OPTIONS),
    std::make_pair(RGY_ERR_INVALID_PROGRAM, CL_INVALID_PROGRAM),
    std::make_pair(RGY_ERR_INVALID_PROGRAM_EXECUTABLE, CL_INVALID_PROGRAM_EXECUTABLE),
    std::make_pair(RGY_ERR_INVALID_KERNEL_NAME, CL_INVALID_KERNEL_NAME),
    std::make_pair(RGY_ERR_INVALID_KERNEL_DEFINITION, CL_INVALID_KERNEL_DEFINITION),
    std::make_pair(RGY_ERR_INVALID_KERNEL, CL_INVALID_KERNEL),
    std::make_pair(RGY_ERR_INVALID_ARG_INDEX, CL_INVALID_ARG_INDEX),
    std::make_pair(RGY_ERR_INVALID_ARG_VALUE, CL_INVALID_ARG_VALUE),
    std::make_pair(RGY_ERR_INVALID_ARG_SIZE, CL_INVALID_ARG_SIZE),
    std::make_pair(RGY_ERR_INVALID_KERNEL_ARGS, CL_INVALID_KERNEL_ARGS),
    std::make_pair(RGY_ERR_INVALID_WORK_DIMENSION, CL_INVALID_WORK_DIMENSION),
    std::make_pair(RGY_ERR_INVALID_WORK_GROUP_SIZE, CL_INVALID_WORK_GROUP_SIZE),
    std::make_pair(RGY_ERR_INVALID_WORK_ITEM_SIZE, CL_INVALID_WORK_ITEM_SIZE),
    std::make_pair(RGY_ERR_INVALID_GLOBAL_OFFSET, CL_INVALID_GLOBAL_OFFSET),
    std::make_pair(RGY_ERR_INVALID_EVENT_WAIT_LIST, CL_INVALID_EVENT_WAIT_LIST),
    std::make_pair(RGY_ERR_INVALID_EVENT, CL_INVALID_EVENT),
    std::make_pair(RGY_ERR_INVALID_OPERATION, CL_INVALID_OPERATION),
    std::make_pair(RGY_ERR_INVALID_GL_OBJECT, CL_INVALID_GL_OBJECT),
    std::make_pair(RGY_ERR_INVALID_BUFFER_SIZE, CL_INVALID_BUFFER_SIZE),
    std::make_pair(RGY_ERR_INVALID_MIP_LEVEL, CL_INVALID_MIP_LEVEL),
    std::make_pair(RGY_ERR_INVALID_GLOBAL_WORK_SIZE, CL_INVALID_GLOBAL_WORK_SIZE)
);
MAP_PAIR_0_1(err, rgy, RGY_ERR, cl, cl_int, RGY_ERR_TO_OPENCL, RGY_ERR_UNKNOWN, CL_INVALID_VALUE);

std::vector<cl_event> toVec(const std::vector<RGYOpenCLEvent>& wait_list) {
    std::vector<cl_event> events;
    if (wait_list.size() > 0) {
        for (auto &event : wait_list) {
            events.push_back(event());
        }
    }
    return events;
}

#if defined(_WIN32) || defined(_WIN64)
static const std::array<const TCHAR *, 1> opencl_dll_names = { _T("OpenCL.dll") };
#else
static const std::array<const TCHAR *, 2> opencl_dll_names = { _T("libOpenCL.so"), _T("libOpenCL.so.1") };
#endif

int initOpenCLGlobal() {
    if (RGYOpenCL::openCLHandle != nullptr) {
        return 0;
    }
    for (const auto dll_name : opencl_dll_names) {
        if ((RGYOpenCL::openCLHandle = RGY_LOAD_LIBRARY(dll_name)) != nullptr) {
            break;
        }
    }
    if (RGYOpenCL::openCLHandle == nullptr) {
        return 1;
    }

#define LOAD(name) \
    f_##name = (decltype(f_##name)) RGY_GET_PROC_ADDRESS(RGYOpenCL::openCLHandle, #name); \
    if (f_##name == nullptr) { \
        RGY_FREE_LIBRARY(RGYOpenCL::openCLHandle); \
        RGYOpenCL::openCLHandle = nullptr; \
        return 1; \
    }
#define LOAD_NO_CHECK(name) \
    f_##name = (decltype(f_##name)) RGY_GET_PROC_ADDRESS(RGYOpenCL::openCLHandle, #name);

    LOAD(clGetExtensionFunctionAddressForPlatform);
    LOAD(clGetDeviceInfo);
    LOAD(clGetPlatformIDs);
    LOAD(clGetDeviceIDs);
    LOAD(clGetPlatformInfo);

    LOAD(clCreateCommandQueue);
    LOAD(clReleaseCommandQueue);
    LOAD(clCreateContext);
    LOAD(clGetCommandQueueInfo);
    LOAD(clReleaseContext);
    LOAD(clGetSupportedImageFormats);

    LOAD(clCreateProgramWithSource);
    LOAD(clBuildProgram);
    LOAD(clGetProgramBuildInfo);
    LOAD(clGetProgramInfo);
    LOAD(clReleaseProgram);

    LOAD(clCreateBuffer);
    LOAD(clCreateImage);
    LOAD_NO_CHECK(clCreateImageWithProperties);
    LOAD(clReleaseMemObject);
    LOAD(clGetMemObjectInfo);
    LOAD(clGetImageInfo);
    LOAD(clCreateKernel);
    LOAD(clReleaseKernel);
    LOAD(clSetKernelArg);
    LOAD(clEnqueueNDRangeKernel);
    LOAD(clEnqueueTask);

    LOAD(clEnqueueReadBuffer);
    LOAD(clEnqueueReadBufferRect);
    LOAD(clEnqueueWriteBuffer);
    LOAD(clEnqueueWriteBufferRect);
    LOAD(clEnqueueCopyBuffer);
    LOAD(clEnqueueCopyBufferRect);
    LOAD(clEnqueueFillBuffer);

    LOAD(clEnqueueReadImage);
    LOAD(clEnqueueWriteImage);
    LOAD(clEnqueueCopyImage);
    LOAD(clEnqueueCopyImageToBuffer);
    LOAD(clEnqueueCopyBufferToImage);
    LOAD(clEnqueueMapBuffer);
    LOAD(clEnqueueMapImage);
    LOAD(clEnqueueUnmapMemObject);

    LOAD(clWaitForEvents);
    LOAD(clGetEventInfo);
    LOAD(clCreateUserEvent);
    LOAD(clRetainEvent);
    LOAD(clReleaseEvent);
    LOAD(clSetUserEventStatus);
    LOAD(clGetEventProfilingInfo);
    LOAD(clEnqueueWaitForEvents);
    LOAD(clEnqueueMarker);

    LOAD_NO_CHECK(clCreateSemaphoreWithPropertiesKHR);
    LOAD_NO_CHECK(clEnqueueWaitSemaphoresKHR);
    LOAD_NO_CHECK(clEnqueueSignalSemaphoresKHR);
    LOAD_NO_CHECK(clGetSemaphoreInfoKHR);
    LOAD_NO_CHECK(clReleaseSemaphoreKHR);

    LOAD(clFlush);
    LOAD(clFinish);

    LOAD_NO_CHECK(clGetKernelSubGroupInfo);
    LOAD_NO_CHECK(clGetKernelSubGroupInfoKHR);
    return 0;
}

#if defined(_WIN32) || defined(_WIN64)
tstring vendorOpenCLDLL() {
#if ENCODER_NVENC
#if _M_IX86
    return _T("nvopencl32.dll");
#else
    return _T("nvopencl64.dll");
#endif
#elif ENCODER_QSV
#if _M_IX86
    return _T("igdrcl32.dll");
#else
    return _T("igdrcl64.dll");
#endif
#elif ENCODER_VCEENC
#if _M_IX86
    return _T("amdocl.dll");
#else
    return _T("amdocl64.dll");
#endif
#else
    return _T("");
#endif
}
#endif

tstring checkOpenCLDLL() {
    tstring str;
    std::unique_ptr<std::remove_pointer_t<HMODULE>, module_deleter> handle;
    for (const auto dll_name : opencl_dll_names) {
        handle = std::unique_ptr<std::remove_pointer_t<HMODULE>, module_deleter>(RGY_LOAD_LIBRARY(dll_name), module_deleter());
        if (handle) {
            str += tstring(_T("Load success: ")) + dll_name;
#if defined(_WIN32) || defined(_WIN64)
            str += _T(" (") + getModulePath(handle.get()) + _T(")");
#endif
            str += _T("\n");
            break;
        }
    }
    handle.reset();
#if defined(_WIN32) || defined(_WIN64)
    if (vendorOpenCLDLL().length() > 0) {
        const auto filelist = get_file_list_with_filter(_T(R"(C:\Windows\System32\DriverStore\FileRepository)"), vendorOpenCLDLL());
        str += _T("\n");
        if (filelist.size()) {
            str += vendorOpenCLDLL() + _T(" found on following path...\n");
            for (const auto& file : filelist) {
                str += _T("  ") + file + _T("\n");
            }
            str += _T("\n");
        } else {
            str += vendorOpenCLDLL() + _T(" not found.\n");
        }
    }
#endif
    return str;
}

static const auto RGY_CLCOMMANDTYPE_TO_STR = make_array<std::pair<cl_mem_object_type, const TCHAR *>>(
#define COMMAND_TYPE(x) std::make_pair(CL_COMMAND_##x, _T(#x))
    COMMAND_TYPE(NDRANGE_KERNEL),
    COMMAND_TYPE(TASK),
    COMMAND_TYPE(NATIVE_KERNEL),
    COMMAND_TYPE(READ_BUFFER),
    COMMAND_TYPE(WRITE_BUFFER),
    COMMAND_TYPE(COPY_BUFFER),
    COMMAND_TYPE(READ_IMAGE),
    COMMAND_TYPE(WRITE_IMAGE),
    COMMAND_TYPE(COPY_IMAGE),
    COMMAND_TYPE(COPY_IMAGE_TO_BUFFER),
    COMMAND_TYPE(COPY_BUFFER_TO_IMAGE),
    COMMAND_TYPE(MAP_BUFFER),
    COMMAND_TYPE(MAP_IMAGE),
    COMMAND_TYPE(UNMAP_MEM_OBJECT),
    COMMAND_TYPE(MARKER),
    COMMAND_TYPE(ACQUIRE_GL_OBJECTS),
    COMMAND_TYPE(RELEASE_GL_OBJECTS)
#ifdef CL_VERSION_1_1
    ,
    COMMAND_TYPE(READ_BUFFER_RECT),
    COMMAND_TYPE(WRITE_BUFFER_RECT),
    COMMAND_TYPE(COPY_BUFFER_RECT),
    COMMAND_TYPE(USER)
#endif
#ifdef CL_VERSION_1_2
    ,
    COMMAND_TYPE(BARRIER),
    COMMAND_TYPE(MIGRATE_MEM_OBJECTS),
    COMMAND_TYPE(FILL_BUFFER),
    COMMAND_TYPE(FILL_IMAGE)
#endif
#ifdef CL_VERSION_2_0
    ,
    COMMAND_TYPE(SVM_FREE),
    COMMAND_TYPE(SVM_MEMCPY),
    COMMAND_TYPE(SVM_MEMFILL),
    COMMAND_TYPE(SVM_MAP),
    COMMAND_TYPE(SVM_UNMAP)
#endif
#ifdef CL_VERSION_3_0
    ,
    COMMAND_TYPE(SVM_MIGRATE_MEM),
#endif
#undef COMMAND_TYPE
);

MAP_PAIR_0_1(clcommandtype, cl, cl_command_type, str, const TCHAR *, RGY_CLCOMMANDTYPE_TO_STR, 0, _T("unknown"));

static const auto RGY_CLSTATUS_TO_STR = make_array<std::pair<cl_mem_object_type, const TCHAR *>>(
    std::make_pair(CL_QUEUED,    _T("queued")),
    std::make_pair(CL_SUBMITTED, _T("submitted")),
    std::make_pair(CL_RUNNING,   _T("running")),
    std::make_pair(CL_COMPLETE,  _T("complete"))
    );

MAP_PAIR_0_1(clstatus, cl, cl_int, str, const TCHAR *, RGY_CLSTATUS_TO_STR, 0, _T("unknown"));


tstring cldevice_cl_to_str(const cl_device_type type) {
    tstring str;
    if (type & CL_DEVICE_TYPE_DEFAULT)     str += _T(", default");
    if (type & CL_DEVICE_TYPE_CPU)         str += _T(", cpu");
    if (type & CL_DEVICE_TYPE_GPU)         str += _T(", gpu");
    if (type & CL_DEVICE_TYPE_ACCELERATOR) str += _T(", accelerator");
    if (type & CL_DEVICE_TYPE_CUSTOM)      str += _T(", custom");
    if (str.length() == 0)                 str += _T(", unknown");
    return str.substr(2);
}

tstring RGYOpenCLEventInfo::print() const {
    tstring str;
    str += strsprintf(_T("context:     0x%p\n"), context);
    str += strsprintf(_T("queue:       0x%p\n"), queue);
    str += strsprintf(_T("commandtype: %s\n"),   clcommandtype_cl_to_str(command_type));
    str += strsprintf(_T("status:      %s\n"),   clstatus_cl_to_str(status));
    str += strsprintf(_T("ref count:   %d\n"),   ref_count);
    return str;
}

RGY_ERR RGYOpenCLEvent::getProfilingTime(uint64_t& time, const cl_profiling_info info) {
    if (*event_ == nullptr) {
        return RGY_ERR_NULL_PTR;
    }
    return err_cl_to_rgy(clGetEventProfilingInfo(*event_, info, sizeof(time), &time, NULL));
}

RGY_ERR RGYOpenCLEvent::getProfilingTimeStart(uint64_t& time) {
    return getProfilingTime(time, CL_PROFILING_COMMAND_START);
}

RGY_ERR RGYOpenCLEvent::getProfilingTimeEnd(uint64_t& time) {
    return getProfilingTime(time, CL_PROFILING_COMMAND_END);
}

RGY_ERR RGYOpenCLEvent::getProfilingTimeSubmit(uint64_t& time) {
    return getProfilingTime(time, CL_PROFILING_COMMAND_SUBMIT);
}

RGY_ERR RGYOpenCLEvent::getProfilingTimeQueued(uint64_t& time) {
    return getProfilingTime(time, CL_PROFILING_COMMAND_QUEUED);
}

RGY_ERR RGYOpenCLEvent::getProfilingTimeComplete(uint64_t& time) {
    return getProfilingTime(time, CL_PROFILING_COMMAND_COMPLETE);
}

RGYOpenCLEventInfo RGYOpenCLEvent::getInfo() const {
    RGYOpenCLEventInfo info;
    try {
        clGetInfo(clGetEventInfo, *event_.get(), CL_EVENT_COMMAND_QUEUE, &info.queue);
        clGetInfo(clGetEventInfo, *event_.get(), CL_EVENT_COMMAND_TYPE, &info.command_type);
        clGetInfo(clGetEventInfo, *event_.get(), CL_EVENT_CONTEXT, &info.context);
        clGetInfo(clGetEventInfo, *event_.get(), CL_EVENT_COMMAND_EXECUTION_STATUS, &info.status);
        clGetInfo(clGetEventInfo, *event_.get(), CL_EVENT_REFERENCE_COUNT, &info.ref_count);
    } catch (...) {
        return RGYOpenCLEventInfo();
    }
    return info;
}

RGY_ERR RGYOpenCLSemaphore::wait(RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!semaphore_ || !*semaphore_) {
        return RGY_ERR_NULL_PTR;
    }
    std::vector<cl_event> cl_wait_events(wait_events.size());
    for (size_t i = 0; i < wait_events.size(); i++) {
        cl_wait_events[i] = wait_events[i]();
    }
    cl_event *event_ptr = (event) ? event->reset_ptr() : nullptr;
    cl_int err = clEnqueueWaitSemaphoresKHR(queue(), 1, semaphore_.get(), nullptr, (cl_uint)cl_wait_events.size(), cl_wait_events.data(), event_ptr);
    return err_cl_to_rgy(err);
}

RGY_ERR RGYOpenCLSemaphore::signal(RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!semaphore_ || !*semaphore_) {
        return RGY_ERR_NULL_PTR;
    }
    std::vector<cl_event> cl_wait_events(wait_events.size());
    for (size_t i = 0; i < wait_events.size(); i++) {
        cl_wait_events[i] = wait_events[i]();
    }
    cl_event *event_ptr = (event) ? event->reset_ptr() : nullptr;
    cl_int err = clEnqueueSignalSemaphoresKHR(queue(), 1, semaphore_.get(), nullptr, (cl_uint)cl_wait_events.size(), cl_wait_events.data(), event_ptr);
    return err_cl_to_rgy(err);
}

void RGYOpenCLSemaphore::release() {
    if (semaphore_ && *semaphore_) {
        clReleaseSemaphoreKHR(*semaphore_);
    }
    semaphore_.reset();
}

RGYOpenCLDeviceInfoVecWidth::RGYOpenCLDeviceInfoVecWidth() :
    w_char(std::make_pair(0,0)),
    w_short(std::make_pair(0,0)),
    w_int(std::make_pair(0,0)),
    w_long(std::make_pair(0,0)),
    w_half(std::make_pair(0,0)),
    w_float(std::make_pair(0,0)),
    w_double(std::make_pair(0,0)) {

}

std::string RGYOpenCLDeviceInfoVecWidth::print() const {
    std::stringstream ts;
    ts << "  vec width char:              " << w_char.first << "/" << w_char.second << std::endl;
    ts << "            short:             " << w_short.first << "/" << w_short.second << std::endl;
    ts << "            int:               " << w_int.first << "/" << w_int.second << std::endl;
    ts << "            long:              " << w_long.first << "/" << w_long.second << std::endl;
    ts << "            half:              " << w_half.first << "/" << w_half.second << std::endl;
    ts << "            float:             " << w_float.first << "/" << w_float.second << std::endl;
    ts << "            double:            " << w_double.first << "/" << w_double.second << std::endl;
    return ts.str();
}

RGYOpenCLDeviceInfo::RGYOpenCLDeviceInfo() :
    type(CL_DEVICE_TYPE_DEFAULT),
    vendor_id(0),
    max_compute_units(0),
    max_clock_frequency(0),
    max_samplers(0),
    global_mem_size(0),
    global_mem_cache_size(0),
    global_mem_cacheline_size(0),
    local_mem_size(0),
    image_support(0),
    image_2d_max_width(0),
    image_2d_max_height(0),
    image_3d_max_width(0),
    image_3d_max_height(0),
    image_3d_max_depth(0),
    image_pitch_alignment(0),
    profiling_timer_resolution(0),
    max_const_args(0),
    max_const_buffer_size(0),
    max_mem_alloc_size(0),
    max_parameter_size(0),
    max_read_image_args(0),
    max_work_group_size(0),
    max_work_item_dims(0),
    max_write_image_args(0),
    mem_base_addr_align(0),
    min_data_type_align_size(0),
    vecwidth(),
    name(),
    vendor(),
    driver_version(),
    profile(),
    version(),
    extensions(),
    uuid(),
    luid()
#if ENCODER_QSV || CLFILTERS_AUF
    ,
    ip_version_intel(0),
    id_intel(0),
    num_slices_intel(0),
    num_subslices_intel(0),
    num_eus_per_subslice_intel(0),
    num_threads_per_eu_intel(0),
    feature_capabilities_intel(0)
#endif
#if ENCODER_NVENC || CLFILTERS_AUF
    ,
    cc_major_nv(0),
    cc_minor_nv(0),
    regs_per_block_nv(0),
    warp_size_nv(0),
    gpu_overlap_nv(0),
    kernel_exec_timeout_nv(0),
    integrated_mem_nv(0)
#endif
#if ENCODER_VCEENC || CLFILTERS_AUF
    ,
    topology_amd(),
    board_name_amd(),
    global_free_mem_size_amd(0),
    simd_per_cu_amd(0),
    simd_width_amd(0),
    simd_instruction_width_amd(0),
    wavefront_width_amd(0),
    global_mem_channels_amd(0),
    global_mem_channel_banks_amd(0),
    global_mem_channel_bank_width_amd(0),
    local_mem_size_per_cu_amd(0),
    local_mem_banks_amd(0),
    thread_trace_supported_amd(0),
    async_queue_support_amd(0),
    max_work_group_size_amd(0),
    preferred_const_buffer_size_amd(0),
    pcie_id_amd(0)
#endif
{};

std::pair<int, int> RGYOpenCLDeviceInfo::clversion() const {
    int major, minor;
    if (sscanf_s(version.c_str(), "OpenCL %d.%d", &major, &minor) == 2) {
        return std::make_pair(major, minor);
    }
    return std::make_pair(0, 0);
}
bool RGYOpenCLDeviceInfo::checkVersion(int major, int minor) const {
    const auto clverpair = clversion();
    if (major < clverpair.first) return true;
    if (major == clverpair.first) return minor <= clverpair.second;
    return false;
}

bool RGYOpenCLDeviceInfo::checkExtension(const char* extension) const {
    return strstr(extensions.c_str(), extension) != 0;
}

RGYOpenCLDevice::RGYOpenCLDevice(cl_device_id device) : m_device(device) {

}

RGYOpenCLDeviceInfo RGYOpenCLDevice::info() const {
    RGYOpenCLDeviceInfo info;
    try {
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_TYPE, &info.type);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_VENDOR_ID, &info.vendor_id);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_MAX_COMPUTE_UNITS, &info.max_compute_units);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_MAX_CLOCK_FREQUENCY, &info.max_clock_frequency);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_MAX_SAMPLERS, &info.max_samplers);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_GLOBAL_MEM_SIZE, &info.global_mem_size);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &info.global_mem_cache_size);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, &info.global_mem_cacheline_size);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_LOCAL_MEM_SIZE, &info.local_mem_size);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_IMAGE_SUPPORT, &info.image_support);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_IMAGE2D_MAX_WIDTH, &info.image_2d_max_width);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, &info.image_2d_max_height);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_IMAGE3D_MAX_WIDTH, &info.image_3d_max_width);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, &info.image_3d_max_height);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_IMAGE3D_MAX_DEPTH, &info.image_3d_max_depth);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_IMAGE_PITCH_ALIGNMENT, &info.image_pitch_alignment);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, &info.profiling_timer_resolution);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_MAX_CONSTANT_ARGS, &info.max_const_args);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, &info.max_const_buffer_size);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, &info.max_mem_alloc_size);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_MAX_PARAMETER_SIZE, &info.max_parameter_size);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_MAX_READ_IMAGE_ARGS, &info.max_read_image_args);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, &info.max_work_group_size);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_MAX_WORK_ITEM_SIZES, &info.max_work_item_dims);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, &info.max_write_image_args);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, &info.mem_base_addr_align);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, &info.min_data_type_align_size);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,   &info.vecwidth.w_char.first);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,  &info.vecwidth.w_short.first);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,    &info.vecwidth.w_int.first);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,   &info.vecwidth.w_long.first);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,   &info.vecwidth.w_half.first);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,  &info.vecwidth.w_float.first);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, &info.vecwidth.w_double.first);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,      &info.vecwidth.w_char.second);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,     &info.vecwidth.w_short.second);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,       &info.vecwidth.w_int.second);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,      &info.vecwidth.w_long.second);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,      &info.vecwidth.w_half.second);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,     &info.vecwidth.w_float.second);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,    &info.vecwidth.w_double.second);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_NAME, &info.name);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_VENDOR, &info.vendor);
        clGetInfo(clGetDeviceInfo, m_device, CL_DRIVER_VERSION, &info.driver_version);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_PROFILE, &info.profile);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_VERSION, &info.version);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_EXTENSIONS, &info.extensions);
        clGetDeviceInfo(m_device, CL_DEVICE_UUID_KHR, sizeof(info.uuid), info.uuid, nullptr);
        clGetDeviceInfo(m_device, CL_DEVICE_LUID_KHR, sizeof(info.luid), info.luid, nullptr);
#if ENCODER_QSV || CLFILTERS_AUF
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_IP_VERSION_INTEL, &info.ip_version_intel);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_ID_INTEL, &info.id_intel);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_NUM_SLICES_INTEL, &info.num_slices_intel);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL, &info.num_subslices_intel);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL, &info.num_eus_per_subslice_intel);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_NUM_THREADS_PER_EU_INTEL, &info.num_threads_per_eu_intel);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_FEATURE_CAPABILITIES_INTEL, &info.feature_capabilities_intel);
#endif
#if ENCODER_NVENC || CLFILTERS_AUF
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, &info.cc_major_nv);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, &info.cc_minor_nv);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_REGISTERS_PER_BLOCK_NV, &info.regs_per_block_nv);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_WARP_SIZE_NV, &info.warp_size_nv);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_GPU_OVERLAP_NV, &info.gpu_overlap_nv);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, &info.kernel_exec_timeout_nv);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_INTEGRATED_MEMORY_NV, &info.integrated_mem_nv);
#endif
#if ENCODER_VCEENC || CLFILTERS_AUF
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_TOPOLOGY_AMD, &info.topology_amd);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_BOARD_NAME_AMD, &info.board_name_amd);
        info.board_name_amd = str_replace(info.board_name_amd, "(TM)", "");
        info.board_name_amd = str_replace(info.board_name_amd, "(R)", "");
        info.board_name_amd = str_replace(info.board_name_amd, "  ", " ");
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_GLOBAL_FREE_MEMORY_AMD, &info.global_free_mem_size_amd);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD, &info.simd_per_cu_amd);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_SIMD_WIDTH_AMD, &info.simd_width_amd);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD, &info.simd_instruction_width_amd);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_WAVEFRONT_WIDTH_AMD, &info.wavefront_width_amd);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD, &info.global_mem_channels_amd);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD, &info.global_mem_channel_banks_amd);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD, &info.global_mem_channel_bank_width_amd);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD, &info.local_mem_size_per_cu_amd);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_LOCAL_MEM_BANKS_AMD, &info.local_mem_banks_amd);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD, &info.thread_trace_supported_amd);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD, &info.async_queue_support_amd);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_AMD, &info.max_work_group_size_amd);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD, &info.preferred_const_buffer_size_amd);
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_PCIE_ID_AMD, &info.pcie_id_amd);
#endif // #if ENCODER_VCEENC
    } catch (...) {
        return RGYOpenCLDeviceInfo();
    }
    return info;
}

bool RGYOpenCLDevice::checkExtension(const char* extension) const {
    std::string extensions;
    try {
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_EXTENSIONS, &extensions);
    } catch (...) {
        return false;
    }
    return strstr(extensions.c_str(), extension) != 0;
}

bool RGYOpenCLDevice::checkVersion(int major, int minor) const {
    std::string version;
    try {
        clGetInfo(clGetDeviceInfo, m_device, CL_DEVICE_VERSION, &version);
    } catch (...) {
        return false;
    }
    int a, b;
    if (sscanf_s(version.c_str(), "OpenCL %d.%d", &a, &b) == 2) {
        if (major < a) return true;
        if (major == a) return minor <= b;
    }
    return false;
}

tstring RGYOpenCLDevice::infostr(bool full) const {
    const auto dev = info();
    std::stringstream ts;
#if ENCODER_VCEENC || CLFILTERS_AUF
    if (dev.board_name_amd.length() > 0) {
        ts << dev.board_name_amd;
    } else {
        ts << dev.name;
    }
#else
    ts << dev.name;
#endif
    if (dev.max_compute_units > 0) {
        ts << " (" << dev.max_compute_units << " CU)";
    }
    if (dev.max_clock_frequency > 0) {
        ts << " @ " << dev.max_clock_frequency << " MHz";
    }
    if (dev.driver_version.length() > 0) {
        ts << " (" << dev.driver_version << ")";
    }
    if (full) {
        [[maybe_unused]] const bool is_intel = checkVendor(dev.vendor.c_str(), "Intel");
        [[maybe_unused]] const bool is_nv    = checkVendor(dev.vendor.c_str(), "NVIDIA");
        [[maybe_unused]] const bool is_amd   = checkVendor(dev.vendor.c_str(), "AMD");
        ts << std::endl;
#if ENCODER_VCEENC || CLFILTERS_AUF
        if (is_amd) {
        ts << "  name :                       " << dev.name << std::endl;
        }
#endif
        ts << "  device type :                " << tchar_to_string(cldevice_cl_to_str(dev.type)) << std::endl;
        ts << "  vendor :                     " << dev.vendor_id << " (" << dev.vendor << ")" << std::endl;
        ts << "  profile :                    " << dev.profile << std::endl;
        ts << "  version :                    " << dev.version << std::endl;
        ts << "  extensions :                 " << dev.extensions << std::endl;
#if ENCODER_QSV || CLFILTERS_AUF
        if (is_intel) {
        ts << "  ip_version_intel :           " << dev.ip_version_intel << std::endl;
        ts << "  id_intel :                   " << dev.id_intel << std::endl;
        }
#endif
#if ENCODER_NVENC || CLFILTERS_AUF
        if (is_nv) {
        ts << "  cc :                         " << dev.cc_major_nv << "." << dev.cc_minor_nv << std::endl;
        }
#endif
#if ENCODER_VCEENC || CLFILTERS_AUF
        if (is_amd) {
        ts << "  pcie_id_amd :                " << dev.pcie_id_amd << std::endl;
        ts << "  topology :                   " << dev.topology_amd << std::endl;
        ts << "  board_name :                 " << dev.board_name_amd << std::endl;
        }
#endif
        ts << "  global_mem_size :            " << dev.global_mem_size / (1024 * 1024) << " MB" << std::endl;
#if ENCODER_VCEENC || CLFILTERS_AUF
        if (is_amd) {
        ts << "  global_free_mem_size_amd :   " << dev.global_free_mem_size_amd / (1024 * 1024) << " MB" << std::endl;
        ts << "  global_mem_channels_amd :    " << dev.global_mem_channels_amd << std::endl;
        ts << "  global_mem_banks_amd :       " << dev.global_mem_channel_banks_amd << std::endl;
        ts << "  global_mem_bank_width_amd :  " << dev.global_mem_channel_bank_width_amd << std::endl;
        }
#endif
        ts << "  global_mem_cache_size :      " << dev.global_mem_cache_size / 1024 << " KB" << std::endl;
        ts << "  global_mem_cacheline_size :  " << dev.global_mem_cacheline_size << " B" << std::endl;
        ts << "  max_mem_alloc_size :         " << dev.max_mem_alloc_size / (1024 * 1024) << " MB" << std::endl;
        ts << "  mem_base_addr_align :        " << dev.mem_base_addr_align << std::endl;
        ts << "  min_data_type_align_size :   " << dev.min_data_type_align_size << std::endl;
        ts << "  local_mem_size :             " << dev.local_mem_size / 1024 << " KB" << std::endl;
#if ENCODER_VCEENC || CLFILTERS_AUF
        if (is_amd) {
        ts << "  local_mem_size_per_cu_amd :  " << dev.local_mem_size_per_cu_amd / 1024 << " KB" << std::endl;
        ts << "  local_mem_banks_amd :        " << dev.local_mem_banks_amd << std::endl;
        }
#endif
        ts << "  max_const_args :             " << dev.max_const_args << std::endl;
        ts << "  max_const_buffer_size :      " << dev.max_const_buffer_size / 1024 << " KB";
#if ENCODER_VCEENC || CLFILTERS_AUF
        if (is_amd) {
        ts                                      << ", preferred " << dev.preferred_const_buffer_size_amd / 1024 << " KB";
        }
#endif
        ts << std::endl;
        ts << "  image support :              " << (dev.image_support ? "yes" : "no") << std::endl;
        ts << "  image2d max size :           " << dev.image_2d_max_width << " x " << dev.image_2d_max_height << std::endl;
        ts << "  image3d max size :           " << dev.image_3d_max_width << " x " << dev.image_3d_max_height << " x " << dev.image_3d_max_depth << std::endl;
        ts << "  image_pitch_alignment :      " << dev.image_pitch_alignment << std::endl;
        ts << "  max_image_args :             read " << dev.max_read_image_args << ", write " << dev.max_write_image_args << std::endl;
        ts << "  profiling_timer_resolution : " << dev.profiling_timer_resolution << " ns" << std::endl;
        ts << "  max_parameter_size :         " << dev.max_parameter_size << std::endl;
        ts << "  max_work_group_size :        " << dev.max_work_group_size << std::endl;
        ts << "  max_work_item_dims :         " << dev.max_work_item_dims << std::endl;
#if ENCODER_QSV || CLFILTERS_AUF
        if (is_intel) {
        ts << "  num_slices_intel :           " << dev.num_slices_intel << std::endl;
        ts << "  num_subslices_intel :        " << dev.num_subslices_intel << std::endl;
        ts << "  num_eus_per_subslice_intel : " << dev.num_eus_per_subslice_intel << std::endl;
        ts << "  num_threads_per_eu_intel :   " << dev.num_threads_per_eu_intel << std::endl;
        ts << "  feature_capabilities_intel : " << dev.feature_capabilities_intel << std::endl;
        }
#endif
#if ENCODER_NVENC || CLFILTERS_AUF
        if (is_nv) {
        ts << "  regs_per_block_nv :          " << dev.regs_per_block_nv << std::endl;
        ts << "  warp_size_nv :               " << dev.warp_size_nv << std::endl;
        ts << "  gpu_overlap_nv :             " << (dev.gpu_overlap_nv ? "yes" : "no") << std::endl;
        ts << "  kernel_exec_timeout_nv :     " << (dev.kernel_exec_timeout_nv ? "yes" : "no") << std::endl;
        ts << "  integrated_mem_nv :          " << (dev.integrated_mem_nv ? "yes" : "no") << std::endl;
        }
#endif
#if ENCODER_VCEENC || CLFILTERS_AUF
        if (is_amd) {
        ts << "  simd_per_cu_amd :            " << dev.simd_per_cu_amd << std::endl;
        ts << "  simd_width_amd :             " << dev.simd_width_amd << std::endl;
        ts << "  simd_instruction_width_amd : " << dev.simd_instruction_width_amd << std::endl;
        ts << "  wavefront_width_amd :        " << dev.wavefront_width_amd << std::endl;
        ts << "  thread_trace_supported_amd : " << dev.thread_trace_supported_amd << std::endl;
        ts << "  async_queue_support_amd :    " << dev.async_queue_support_amd << std::endl;
        }
#endif
        ts << dev.vecwidth.print();

    }
    return char_to_tstring(ts.str());
}

RGYOpenCLPlatform::RGYOpenCLPlatform(cl_platform_id platform, shared_ptr<RGYLog> pLog) : m_platform(platform), m_d3d9dev(nullptr), m_d3d11dev(nullptr), m_vadev(nullptr), m_devices(), m_log(pLog) {
}

#define LOAD_KHR(name) \
    if ((name) == nullptr) { \
        try { \
            f_##name = (decltype(f_##name))clGetExtensionFunctionAddressForPlatform(m_platform, #name); \
            if ((name) == nullptr) { \
                CL_LOG(RGY_LOG_ERROR, _T("Failed to load function %s\n"), char_to_tstring(#name).c_str()); \
                return RGY_ERR_NOT_FOUND; \
            } \
        }  catch (...) { \
            CL_LOG(RGY_LOG_ERROR, _T("Crush (clGetExtensionFunctionAddressForPlatform)\n")); \
            RGYOpenCL::openCLCrush = true; \
            return RGY_ERR_OPENCL_CRUSH; \
        } \
    }


RGY_ERR RGYOpenCLPlatform::createDeviceListD3D11(cl_device_type device_type, void *d3d11dev, const bool tryMode) {
#if !ENABLE_RGY_OPENCL_D3D11
    return RGY_ERR_UNSUPPORTED;
#else
    if (RGYOpenCL::openCLCrush) {
        return RGY_ERR_OPENCL_CRUSH;
    }
    CL_LOG(RGY_LOG_DEBUG, _T("createDeviceListD3D11(d3d11dev = %p)\n"), d3d11dev);

    auto ret = RGY_ERR_NONE;
    cl_uint device_count = 0;
    if (d3d11dev && checkExtension("cl_khr_d3d11_sharing")) {
        LOAD_KHR(clGetDeviceIDsFromD3D11KHR);
        LOAD_KHR(clCreateFromD3D11BufferKHR);
        LOAD_KHR(clCreateFromD3D11Texture2DKHR);
        LOAD_KHR(clCreateFromD3D11Texture3DKHR);
        LOAD_KHR(clEnqueueAcquireD3D11ObjectsKHR);
        LOAD_KHR(clEnqueueReleaseD3D11ObjectsKHR);
    }
    if (d3d11dev && clGetDeviceIDsFromD3D11KHR) {
        m_d3d11dev = d3d11dev;
        int select_dev_type = CL_PREFERRED_DEVICES_FOR_D3D11_KHR;
        try {
            if ((ret = err_cl_to_rgy(clGetDeviceIDsFromD3D11KHR(m_platform, CL_D3D11_DEVICE_KHR, d3d11dev, select_dev_type, 0, NULL, &device_count))) != RGY_ERR_NONE) {
                select_dev_type = CL_ALL_DEVICES_FOR_D3D11_KHR;
                if ((ret = err_cl_to_rgy(clGetDeviceIDsFromD3D11KHR(m_platform, CL_D3D11_DEVICE_KHR, d3d11dev, select_dev_type, 0, NULL, &device_count))) != RGY_ERR_NONE) {
                    CL_LOG((tryMode) ? RGY_LOG_DEBUG : RGY_LOG_ERROR, _T("Error (clGetDeviceIDsFromD3D11KHR): %s\n"), get_err_mes(ret));
                    return ret;
                }
            }
            CL_LOG(RGY_LOG_DEBUG, _T("D3D11 device count = %d\n"), device_count);
        } catch (...) {
            CL_LOG(RGY_LOG_ERROR, _T("Crush (clGetDeviceIDsFromD3D11KHR)\n"));
            return RGY_ERR_OPENCL_CRUSH;
        }
        if (device_count > 0) {
            std::vector<cl_device_id> devs(device_count, 0);
            try {
                ret = err_cl_to_rgy(clGetDeviceIDsFromD3D11KHR(m_platform, CL_D3D11_DEVICE_KHR, d3d11dev, select_dev_type, device_count, devs.data(), &device_count));
            } catch (...) {
                CL_LOG(RGY_LOG_ERROR, _T("Crush (clGetDeviceIDsFromD3D11KHR)\n"));
                return RGY_ERR_OPENCL_CRUSH;
            }
            if (ret == RGY_ERR_NONE) {
                m_devices = devs;
                CL_LOG(RGY_LOG_DEBUG, _T("clGetDeviceIDsFromD3D11KHR: Success\n"));
                return ret;
            }
        }
    } else {
        ret = createDeviceList(device_type);
    }
    return RGY_ERR_NONE;
#endif
}

RGY_ERR RGYOpenCLPlatform::createDeviceListD3D9(cl_device_type device_type, void *d3d9dev, const bool tryMode) {
#if !ENABLE_RGY_OPENCL_D3D9
    return RGY_ERR_UNSUPPORTED;
#else
    if (RGYOpenCL::openCLCrush) {
        return RGY_ERR_OPENCL_CRUSH;
    }
    CL_LOG(RGY_LOG_DEBUG, _T("createDeviceListD3D9(d3d9dev = %p)\n"), d3d9dev);

    auto ret = RGY_ERR_NONE;
    cl_uint device_count = 1;
    if (d3d9dev && checkExtension("cl_khr_dx9_media_sharing")) {
        LOAD_KHR(clGetDeviceIDsFromDX9MediaAdapterKHR);
        LOAD_KHR(clCreateFromDX9MediaSurfaceKHR);
        LOAD_KHR(clEnqueueAcquireDX9MediaSurfacesKHR);
        LOAD_KHR(clEnqueueReleaseDX9MediaSurfacesKHR);
    }
    if (d3d9dev) {
        if (clGetDeviceIDsFromDX9MediaAdapterKHR) {
            CL_LOG(RGY_LOG_DEBUG, _T("clGetDeviceIDsFromDX9MediaAdapterKHR(d3d9dev = %p)\n"), d3d9dev);
            m_d3d9dev = d3d9dev;
            std::vector<cl_device_id> devs(device_count, 0);
            try {
                cl_dx9_media_adapter_type_khr type = CL_ADAPTER_D3D9EX_KHR;
                ret = err_cl_to_rgy(clGetDeviceIDsFromDX9MediaAdapterKHR(m_platform, 1, &type, &d3d9dev, CL_PREFERRED_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR, device_count, devs.data(), &device_count));
                if (ret != RGY_ERR_NONE || device_count == 0) {
                    device_count = 1;
                    if ((ret = err_cl_to_rgy(clGetDeviceIDsFromDX9MediaAdapterKHR(m_platform, 1, &type, &d3d9dev, CL_ALL_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR, device_count, devs.data(), &device_count))) != RGY_ERR_NONE) {
                        CL_LOG((tryMode) ? RGY_LOG_DEBUG : RGY_LOG_ERROR, _T("Error (clGetDeviceIDsFromD3D11KHR): %s\n"), get_err_mes(ret));
                        return ret;
                    }
                }
            }
            catch (...) {
                CL_LOG(RGY_LOG_ERROR, _T("Crush (clGetDeviceIDsFromDX9MediaAdapterKHR)\n"));
                return RGY_ERR_OPENCL_CRUSH;
            }
            if (ret == RGY_ERR_NONE) {
                m_devices = devs;
                CL_LOG(RGY_LOG_DEBUG, _T("clGetDeviceIDsFromDX9MediaAdapterKHR: Success\n"));
                return ret;
            }
        }
#if 0
        if (ret != RGY_ERR_NONE || device_count == 0) {
            clGetDeviceIDsFromDX9MediaAdapterKHR = nullptr;
            clCreateFromDX9MediaSurfaceKHR = nullptr;
            clEnqueueAcquireDX9MediaSurfacesKHR = nullptr;
            clEnqueueReleaseDX9MediaSurfacesKHR = nullptr;
            if (checkExtension("cl_intel_dx9_media_sharing")) {
                LOAD_KHR(clGetDeviceIDsFromDX9INTEL);
                LOAD_KHR(clCreateFromDX9MediaSurfaceINTEL);
                LOAD_KHR(clEnqueueAcquireDX9ObjectsINTEL);
                LOAD_KHR(clEnqueueReleaseDX9ObjectsINTEL);
                if (clGetDeviceIDsFromDX9INTEL) {
                    CL_LOG(RGY_LOG_DEBUG, _T("clGetDeviceIDsFromDX9INTEL(d3d9dev = %p)\n"), d3d9dev);
                    device_count = 1;
                    std::vector<cl_device_id> devs(device_count, 0);
                    try {
                        cl_dx9_media_adapter_type_khr type = CL_ADAPTER_D3D9EX_KHR;
                        ret = err_cl_to_rgy(clGetDeviceIDsFromDX9INTEL(m_platform, CL_D3D9EX_DEVICE_INTEL, d3d9dev, CL_PREFERRED_DEVICES_FOR_DX9_INTEL, device_count, devs.data(), &device_count));
                    }
                    catch (...) {
                        CL_LOG(RGY_LOG_ERROR, _T("Crush (clGetDeviceIDsFromDX9INTEL)\n"));
                        RGYOpenCL::openCLCrush = true; //クラッシュフラグを立てる
                        return RGY_ERR_OPENCL_CRUSH;
                    }
                    if (ret == RGY_ERR_NONE) {
                        m_devices = devs;
                        CL_LOG(RGY_LOG_DEBUG, _T("clGetDeviceIDsFromDX9INTEL: Success\n"));
                        return ret;
                    }
                }
            }
        }
#endif
    } else {
        ret = createDeviceList(device_type);
    }
    return RGY_ERR_NONE;
#endif
}

RGY_ERR RGYOpenCLPlatform::createDeviceListVA(cl_device_type device_type, void *vadev, [[maybe_unused]] const bool tryMode) {
#if !ENABLE_RGY_OPENCL_VA
    UNREFERENCED_PARAMETER(device_type);
    UNREFERENCED_PARAMETER(vadev);
    return RGY_ERR_UNSUPPORTED;
#else
    if (RGYOpenCL::openCLCrush) {
        return RGY_ERR_OPENCL_CRUSH;
    }
    CL_LOG(RGY_LOG_DEBUG, _T("f_clGetDeviceIDsFromVA_APIMediaAdapterINTEL(vadev = %p)\n"), vadev);

    auto ret = RGY_ERR_NONE;
    cl_uint device_count = 0;
    if (vadev && checkExtension("cl_intel_va_api_media_sharing")) {
        LOAD_KHR(clGetDeviceIDsFromVA_APIMediaAdapterINTEL);
        LOAD_KHR(clCreateFromVA_APIMediaSurfaceINTEL);
        LOAD_KHR(clEnqueueAcquireVA_APIMediaSurfacesINTEL);
        LOAD_KHR(clEnqueueReleaseVA_APIMediaSurfacesINTEL);
    }
    if (vadev && clGetDeviceIDsFromVA_APIMediaAdapterINTEL) {
        m_vadev = vadev;
        int select_dev_type = CL_PREFERRED_DEVICES_FOR_VA_API_INTEL;
        try {
            if ((ret = err_cl_to_rgy(clGetDeviceIDsFromVA_APIMediaAdapterINTEL(m_platform, CL_VA_API_DISPLAY_INTEL, vadev, select_dev_type, 0, NULL, &device_count))) != RGY_ERR_NONE) {
                select_dev_type = CL_ALL_DEVICES_FOR_VA_API_INTEL;
                if ((ret = err_cl_to_rgy(clGetDeviceIDsFromVA_APIMediaAdapterINTEL(m_platform, CL_VA_API_DISPLAY_INTEL, vadev, select_dev_type, 0, NULL, &device_count))) != RGY_ERR_NONE) {
                    CL_LOG((tryMode) ? RGY_LOG_DEBUG : RGY_LOG_ERROR, _T("Error (clGetDeviceIDsFromVA_APIMediaAdapterINTEL): %s\n"), get_err_mes(ret));
                    return ret;
                }
            }
            CL_LOG(RGY_LOG_DEBUG, _T("VA device count = %d\n"), device_count);
        } catch (...) {
            CL_LOG(RGY_LOG_ERROR, _T("Crush (clGetDeviceIDsFromVA_APIMediaAdapterINTEL)\n"));
            RGYOpenCL::openCLCrush = true; //クラッシュフラグを立てる
            return RGY_ERR_OPENCL_CRUSH;
        }
        if (device_count > 0) {
            std::vector<cl_device_id> devs(device_count, 0);
            try {
                ret = err_cl_to_rgy(clGetDeviceIDsFromVA_APIMediaAdapterINTEL(m_platform, CL_VA_API_DISPLAY_INTEL, vadev, select_dev_type, device_count, devs.data(), &device_count));
            } catch (...) {
                CL_LOG(RGY_LOG_ERROR, _T("Crush (clGetDeviceIDsFromVA_APIMediaAdapterINTEL)\n"));
                RGYOpenCL::openCLCrush = true; //クラッシュフラグを立てる
                return RGY_ERR_OPENCL_CRUSH;
            }
            if (ret == RGY_ERR_NONE) {
                m_devices = devs;
                CL_LOG(RGY_LOG_DEBUG, _T("clGetDeviceIDsFromVA_APIMediaAdapterINTEL: Success\n"));
                return ret;
            }
        }
    } else {
        ret = createDeviceList(device_type);
    }
    return RGY_ERR_NONE;
#endif
}

RGY_ERR RGYOpenCLPlatform::loadSubGroupKHR() {
    if (clGetKernelSubGroupInfoKHR == nullptr) {
        LOAD_KHR(clGetKernelSubGroupInfoKHR);
    }
    return RGY_ERR_NONE;
}

RGYOpenCLSubGroupSupport RGYOpenCLPlatform::checkSubGroupSupport(const cl_device_id devid) {
    if (RGYOpenCL::openCLCrush) {
        return RGYOpenCLSubGroupSupport::NONE;
    }
    RGYOpenCLDevice device(devid);
    CL_LOG(RGY_LOG_DEBUG, _T("checkSubGroupSupport\n"));
    if (checkVersion(3, 0) && device.checkVersion(3, 0)) {
        if (device.checkExtension("cl_khr_subgroups"))   return loadSubGroupKHR() == RGY_ERR_NONE ? RGYOpenCLSubGroupSupport::STD20KHR  : RGYOpenCLSubGroupSupport::NONE;
        if (device.checkExtension("cl_intel_subgroups")) return loadSubGroupKHR() == RGY_ERR_NONE ? RGYOpenCLSubGroupSupport::INTEL_EXT : RGYOpenCLSubGroupSupport::NONE;
        return RGYOpenCLSubGroupSupport::NONE;
    }
    if (checkVersion(2, 2) && device.checkVersion(2, 2) && clGetKernelSubGroupInfo) {
        return RGYOpenCLSubGroupSupport::STD22;
    }

    if (checkVersion(2, 0) && device.checkVersion(2, 0) && device.checkExtension("cl_khr_subgroups")) {
        return loadSubGroupKHR() == RGY_ERR_NONE ? RGYOpenCLSubGroupSupport::STD20KHR : RGYOpenCLSubGroupSupport::NONE;
    }
    if (checkVersion(1, 2) && device.checkVersion(1, 2) && device.checkExtension("cl_intel_subgroups")) {
        return loadSubGroupKHR() == RGY_ERR_NONE ? RGYOpenCLSubGroupSupport::INTEL_EXT : RGYOpenCLSubGroupSupport::NONE;
    }
    return RGYOpenCLSubGroupSupport::NONE;
}

RGY_ERR RGYOpenCLPlatform::createDeviceList(cl_device_type device_type) {
    if (RGYOpenCL::openCLCrush) {
        return RGY_ERR_OPENCL_CRUSH;
    }
    auto ret = RGY_ERR_NONE;
    cl_uint device_count = 0;
    try {
        if ((ret = err_cl_to_rgy(clGetDeviceIDs(m_platform, device_type, 0, NULL, &device_count))) != RGY_ERR_NONE) {
            CL_LOG(RGY_LOG_ERROR, _T("Error (clGetDeviceIDs): %s\n"), get_err_mes(ret));
            return ret;
        }
        CL_LOG(RGY_LOG_DEBUG, _T("OpenCL device count = %d\n"), device_count);
    } catch (...) {
        CL_LOG(RGY_LOG_ERROR, _T("Crush (clGetDeviceIDs)\n"));
        RGYOpenCL::openCLCrush = true; //クラッシュフラグを立てる
        return RGY_ERR_OPENCL_CRUSH;
    }
    if (device_count > 0) {
        std::vector<cl_device_id> devs(device_count, 0);
        try {
            ret = err_cl_to_rgy(clGetDeviceIDs(m_platform, device_type, device_count, devs.data(), &device_count));
        } catch (...) {
            CL_LOG(RGY_LOG_ERROR, _T("Crush (clGetDeviceIDs)\n"));
            RGYOpenCL::openCLCrush = true; //クラッシュフラグを立てる
            return RGY_ERR_OPENCL_CRUSH;
        }
        if (ret == RGY_ERR_NONE) {
            m_devices = devs;
            CL_LOG(RGY_LOG_DEBUG, _T("clGetDeviceIDs: Success\n"));
            return ret;
        }
    }
    return RGY_ERR_NONE;
}

RGYOpenCLPlatformInfo::RGYOpenCLPlatformInfo() :
    profile(),
    version(),
    name(),
    vendor(),
    extensions() {
}

tstring RGYOpenCLPlatformInfo::print() const {
    std::string str = name + " " + vendor + " " + version + "[" + profile + "]\n  extensions:" + extensions;
    return char_to_tstring(str);
}

std::pair<int, int> RGYOpenCLPlatformInfo::clversion() const {
    int major, minor;
    if (sscanf_s(version.c_str(), "OpenCL %d.%d", &major, &minor) == 2) {
        return std::make_pair(major, minor);
    }
    return std::make_pair(0, 0);
}
bool RGYOpenCLPlatformInfo::checkVersion(int major, int minor) const {
    const auto clverpair = clversion();
    if (major < clverpair.first) return true;
    if (major == clverpair.first) return minor <= clverpair.second;
    return false;
}
bool RGYOpenCLPlatformInfo::checkExtension(const char* extension) const {
    return strstr(extensions.c_str(), extension) != 0;
}

RGYOpenCLPlatformInfo RGYOpenCLPlatform::info() const {
    RGYOpenCLPlatformInfo info;
    try {
        clGetInfo(clGetPlatformInfo, m_platform, CL_PLATFORM_PROFILE, &info.profile);
        clGetInfo(clGetPlatformInfo, m_platform, CL_PLATFORM_VERSION, &info.version);
        clGetInfo(clGetPlatformInfo, m_platform, CL_PLATFORM_NAME, &info.name);
        clGetInfo(clGetPlatformInfo, m_platform, CL_PLATFORM_VENDOR, &info.vendor);
        clGetInfo(clGetPlatformInfo, m_platform, CL_PLATFORM_EXTENSIONS, &info.extensions);
    } catch (...) {
        return RGYOpenCLPlatformInfo();
    }
    return info;
}

RGY_ERR RGYOpenCLPlatform::setDev(cl_device_id dev, void *d3d9dev, void *d3d11dev) {
    m_devices.clear(); m_devices.push_back(dev);
    if (d3d9dev) {
        m_d3d9dev = d3d9dev;
#if ENABLE_RGY_OPENCL_D3D9
        if (checkExtension("cl_khr_dx9_media_sharing")) {
            LOAD_KHR(clGetDeviceIDsFromDX9MediaAdapterKHR);
            LOAD_KHR(clCreateFromDX9MediaSurfaceKHR);
            LOAD_KHR(clEnqueueAcquireDX9MediaSurfacesKHR);
            LOAD_KHR(clEnqueueReleaseDX9MediaSurfacesKHR);
        }
#endif
    }
    if (d3d11dev) {
        m_d3d11dev = d3d11dev;
#if ENABLE_RGY_OPENCL_D3D11
        if (checkExtension("cl_khr_d3d11_sharing")) {
            LOAD_KHR(clGetDeviceIDsFromD3D11KHR);
            LOAD_KHR(clCreateFromD3D11BufferKHR);
            LOAD_KHR(clCreateFromD3D11Texture2DKHR);
            LOAD_KHR(clCreateFromD3D11Texture3DKHR);
            LOAD_KHR(clEnqueueAcquireD3D11ObjectsKHR);
            LOAD_KHR(clEnqueueReleaseD3D11ObjectsKHR);
        }
#endif
    }
    return RGY_ERR_NONE;
};

bool RGYOpenCLPlatform::isVendor(const char *vendor) const {
    return checkVendor(info().vendor.c_str(), vendor);
}

bool RGYOpenCLPlatform::checkExtension(const char* extension) const {
    return info().checkExtension(extension);
}

bool RGYOpenCLPlatform::checkVersion(int major, int minor) const {
    return info().checkVersion(major, minor);
}

RGYOpenCLContext::RGYOpenCLContext(shared_ptr<RGYOpenCLPlatform> platform, int buildThreads, shared_ptr<RGYLog> pLog) :
    m_platform(std::move(platform)),
    m_context(nullptr, clReleaseContext),
    m_queue(),
    m_log(pLog),
    m_copy(),
    m_threadPool(),
    m_buildThreads(buildThreads > 0 ? buildThreads : std::min(RGY_OPENCL_BUILD_THREAD_DEFAULT_MAX, (int)std::thread::hardware_concurrency())),
    m_hmodule(NULL) {

}

RGYOpenCLContext::~RGYOpenCLContext() {
    m_threadPool.reset();
    CL_LOG(RGY_LOG_DEBUG, _T("Closing CL Context...\n"));
    m_copy.clear();     CL_LOG(RGY_LOG_DEBUG, _T("Closed CL m_copy program.\n"));
    m_queue.clear();    CL_LOG(RGY_LOG_DEBUG, _T("Closed CL Queue.\n"));
    m_context.reset();  CL_LOG(RGY_LOG_DEBUG, _T("Closed CL Context.\n"));
    m_platform.reset(); CL_LOG(RGY_LOG_DEBUG, _T("Closed CL Platform.\n"));
    m_log.reset();
}

RGY_ERR RGYOpenCLContext::createContext(const cl_command_queue_properties queue_properties) {
    if (RGYOpenCL::openCLCrush) {
        return RGY_ERR_OPENCL_CRUSH;
    }
    {
        tstring devstr = _T("[");
        for (const auto dev : m_platform->devs()) {
            devstr += strsprintf(_T("%p,"), dev);
        }
        devstr = devstr.substr(0, devstr.length() - 1) + _T("]");
        CL_LOG(RGY_LOG_DEBUG, _T("create OpenCL Context for %s\n"), devstr.c_str());
    }

    cl_int err = RGY_ERR_NONE;
    std::vector<cl_context_properties> props = { CL_CONTEXT_PLATFORM, (cl_context_properties)(m_platform->get()) };
    bool enableInterop = false;
    #if ENABLE_RGY_OPENCL_D3D9
    if (m_platform->d3d9dev()) {
        props.push_back(CL_CONTEXT_ADAPTER_D3D9EX_KHR);
        props.push_back((cl_context_properties)m_platform->d3d9dev());
        CL_LOG(RGY_LOG_DEBUG, _T("Enable d3d9 interop for %p\n"), m_platform->d3d9dev());
        enableInterop = true;
    }
    #endif //ENABLE_RGY_OPENCL_D3D9
    #if ENABLE_RGY_OPENCL_D3D11
    if (m_platform->d3d11dev()) {
        props.push_back(CL_CONTEXT_D3D11_DEVICE_KHR);
        props.push_back((cl_context_properties)m_platform->d3d11dev());
        CL_LOG(RGY_LOG_DEBUG, _T("Enable d3d11 interop for %p\n"), m_platform->d3d11dev());
        enableInterop = true;
    }
    #endif //#if ENABLE_RGY_OPENCL_D3D11
    #if ENABLE_RGY_OPENCL_VA
    if (m_platform->vadev()) {
        props.push_back(CL_CONTEXT_VA_API_DISPLAY_INTEL);
        props.push_back((cl_context_properties)m_platform->vadev());
        CL_LOG(RGY_LOG_DEBUG, _T("Enable va interop for %p\n"), m_platform->d3d11dev());
        enableInterop = true;
    }
    #endif
    if (enableInterop) {
        props.push_back(CL_CONTEXT_INTEROP_USER_SYNC);
        props.push_back((ENCODER_QSV) ? CL_TRUE : CL_FALSE);
    }
    props.push_back(0);
    try {
        m_context = unique_context(clCreateContext(props.data(), (cl_uint)m_platform->devs().size(), m_platform->devs().data(), nullptr, nullptr, &err), clReleaseContext);
    } catch (...) {
        CL_LOG(RGY_LOG_ERROR, _T("Crush (clCreateContext)\n"));
        RGYOpenCL::openCLCrush = true; //クラッシュフラグを立てる
        return RGY_ERR_OPENCL_CRUSH;
    }
    if (err != CL_SUCCESS) {
        CL_LOG(RGY_LOG_ERROR, _T("Error (clCreateContext): %s\n"), cl_errmes(err));
        return err_cl_to_rgy(err);
    }
    for (int idev = 0; idev < (int)m_platform->devs().size(); idev++) {
        m_queue.push_back(createQueue(m_platform->dev(idev).id(), queue_properties));
    }
    return RGY_ERR_NONE;
}

RGYOpenCLQueue RGYOpenCLContext::createQueue(const cl_device_id devid, const cl_command_queue_properties properties) {
    RGYOpenCLQueue queue;
    cl_int err = RGY_ERR_NONE;
    CL_LOG(RGY_LOG_DEBUG, _T("createQueue for device : %p\n"), devid);
    try {
        queue = RGYOpenCLQueue(clCreateCommandQueue(m_context.get(), devid, properties, &err), devid);
        if (err != RGY_ERR_NONE) {
            CL_LOG(RGY_LOG_ERROR, _T("Error (clCreateCommandQueue): %s\n"), cl_errmes(err));
        }
    } catch (...) {
        CL_LOG(RGY_LOG_ERROR, _T("Crush (clCreateCommandQueue)\n"));
        RGYOpenCL::openCLCrush = true; //クラッシュフラグを立てる
    }
    return queue;
}

RGYOpenCLKernelLauncher::RGYOpenCLKernelLauncher(cl_kernel kernel, std::string kernelName, RGYOpenCLQueue &queue, const RGYWorkSize &local, const RGYWorkSize &global, shared_ptr<RGYLog> pLog, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event) :
    m_kernel(kernel), m_kernelName(kernelName), m_queue(queue), m_local(local), m_global(global), m_log(pLog), m_wait_events(toVec(wait_events)), m_event(event) {
}

size_t RGYOpenCLKernelLauncher::subGroupSize() const {
    auto clFunc = (clGetKernelSubGroupInfo) ? clGetKernelSubGroupInfo : clGetKernelSubGroupInfoKHR;
    if (clFunc == nullptr) return 0;
    size_t subGroupSize = 0;
    auto err = clFunc(m_kernel, m_queue.devid(), CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE, sizeof(RGYWorkSize::w), m_local(), sizeof(subGroupSize), &subGroupSize, nullptr);
    if (err != CL_SUCCESS) {
        CL_LOG(RGY_LOG_ERROR, _T("Error: Failed to get subGroupSize of kernel \"%s\": %s\n"), char_to_tstring(m_kernelName).c_str(), cl_errmes(err));
        return 0;
    }
    return subGroupSize;
}
size_t RGYOpenCLKernelLauncher::subGroupCount() const {
    auto clFunc = (clGetKernelSubGroupInfo) ? clGetKernelSubGroupInfo : clGetKernelSubGroupInfoKHR;
    if (clFunc == nullptr) return 0;
    size_t subGroupCount = 0;
    auto err = clFunc(m_kernel, m_queue.devid(), CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE, sizeof(RGYWorkSize::w), m_local(), sizeof(subGroupCount), &subGroupCount, nullptr);
    if (err != CL_SUCCESS) {
        CL_LOG(RGY_LOG_ERROR, _T("Error: Failed to get subGroupCount of kernel \"%s\": %s\n"), char_to_tstring(m_kernelName).c_str(), cl_errmes(err));
        return 0;
    }
    return subGroupCount;
}

RGY_ERR RGYOpenCLKernelLauncher::launch(std::vector<void *> arg_ptrs, std::vector<size_t> arg_size, std::vector<std::type_index> arg_type) {
    assert(arg_ptrs.size() == arg_size.size());
    assert(arg_ptrs.size() == arg_type.size());
    for (int i = 0; i < (int)arg_ptrs.size(); i++) {
        if (arg_type[i] == typeid(RGYOpenCLKernelDynamicLocal)) {
            auto ptr = reinterpret_cast<RGYOpenCLKernelDynamicLocal *>(arg_ptrs[i]);
            auto err = err_cl_to_rgy(clSetKernelArg(m_kernel, i, ptr->size(), nullptr));
            if (err != RGY_ERR_NONE) {
                CL_LOG(RGY_LOG_ERROR, _T("Error: Failed to set #%d arg (local array size: %d) to kernel \"%s\": %s\n"), i, ptr->size(), char_to_tstring(m_kernelName).c_str(), get_err_mes(err));
                return err;
            }
        } else {
            auto err = err_cl_to_rgy(clSetKernelArg(m_kernel, i, arg_size[i], arg_ptrs[i]));
            if (err != RGY_ERR_NONE) {
                uint64_t argvalue = *(uint64_t *)arg_ptrs[i];
                argvalue &= std::numeric_limits<uint64_t>::max() >> ((8 - arg_size[i]) * 8);
                CL_LOG(RGY_LOG_ERROR, _T("Error: Failed to set #%d arg to kernel \"%s\": %s, size: %d, ptr 0x%p, ptrvalue 0x%p\n"),
                    i, char_to_tstring(m_kernelName).c_str(), get_err_mes(err), arg_size[i], arg_ptrs[i], argvalue);
                return err;
            }
        }
    }
    auto globalCeiled = m_global.ceilGlobal(m_local);
    auto err = err_cl_to_rgy(clEnqueueNDRangeKernel(m_queue.get(), m_kernel, 3, NULL, globalCeiled(), m_local(),
        (int)m_wait_events.size(),
        (m_wait_events.size() > 0) ? m_wait_events.data() : nullptr,
        (m_event) ? m_event->reset_ptr() : nullptr));
    if (err != RGY_ERR_NONE) {
        CL_LOG(RGY_LOG_ERROR, _T("Error: Failed to run kernel \"%s\": %s\n"), char_to_tstring(m_kernelName).c_str(), get_err_mes(err));
        return err;
    }
    return err;
}

RGYOpenCLKernel::RGYOpenCLKernel(cl_kernel kernel, std::string kernelName, shared_ptr<RGYLog> pLog) : m_kernel(kernel), m_kernelName(kernelName), m_log(pLog) {

}

RGYOpenCLKernel::~RGYOpenCLKernel() {
    if (m_kernel) {
        clReleaseKernel(m_kernel);
        m_kernel = nullptr;
    }
    m_kernelName.clear();
    m_log.reset();
};

RGYOpenCLProgram::RGYOpenCLProgram(cl_program program, shared_ptr<RGYLog> pLog) : m_program(program), m_log(pLog), m_kernels() {
};

RGYOpenCLProgram::~RGYOpenCLProgram() {
    if (m_program) {
        CL_LOG(RGY_LOG_DEBUG, _T("clReleaseProgram...\n"));
        clReleaseProgram(m_program);
        m_program = nullptr;
        CL_LOG(RGY_LOG_DEBUG, _T("clReleaseProgram: fin.\n"));
    }
};

RGYOpenCLKernelLauncher RGYOpenCLKernel::config(RGYOpenCLQueue &queue, const RGYWorkSize &local, const RGYWorkSize &global, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    return RGYOpenCLKernelLauncher(m_kernel, m_kernelName, queue, local, global, m_log, wait_events, event);
}

RGYOpenCLKernelLauncher RGYOpenCLKernelHolder::config(RGYOpenCLQueue &queue, const RGYWorkSize &local, const RGYWorkSize &global) {
    return RGYOpenCLKernelLauncher(m_kernel->get(), m_kernel->name(), queue, local, global, m_log, {}, nullptr);
}

RGYOpenCLKernelLauncher RGYOpenCLKernelHolder::config(RGYOpenCLQueue &queue, const RGYWorkSize &local, const RGYWorkSize &global, RGYOpenCLEvent *event) {
    return RGYOpenCLKernelLauncher(m_kernel->get(), m_kernel->name(), queue, local, global, m_log, {}, event);
}

RGYOpenCLKernelLauncher RGYOpenCLKernelHolder::config(RGYOpenCLQueue &queue, const RGYWorkSize &local, const RGYWorkSize &global, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    return RGYOpenCLKernelLauncher(m_kernel->get(), m_kernel->name(), queue, local, global, m_log, wait_events, event);
}

RGYOpenCLKernelHolder::RGYOpenCLKernelHolder(RGYOpenCLKernel *kernel, shared_ptr<RGYLog> pLog) : m_kernel(kernel), m_log(pLog) {};

RGYOpenCLKernelHolder RGYOpenCLProgram::kernel(const char *kernelName) {
    for (auto& kernel : m_kernels) {
        if (strcmp(kernel->name().c_str(), kernelName) == 0) {
            return RGYOpenCLKernelHolder(kernel.get(), m_log);
        }
    }
    cl_int err = CL_SUCCESS;
    auto kernel = clCreateKernel(m_program, kernelName, &err);
    if (err != CL_SUCCESS) {
        CL_LOG(RGY_LOG_ERROR, _T("Failed to get kernel %s: %s\n"), char_to_tstring(kernelName).c_str(), cl_errmes(err));
    }
    m_kernels.push_back(std::make_unique<RGYOpenCLKernel>(kernel, kernelName, m_log));
    return RGYOpenCLKernelHolder(m_kernels.back().get(), m_log);
}

std::vector<uint8_t> RGYOpenCLProgram::getBinary() {
    std::vector<uint8_t> binary;
    if (!m_program) return binary;

    size_t binary_size = 0;
    cl_int err = clGetProgramInfo(m_program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, nullptr);
    if (err != CL_SUCCESS) {
        CL_LOG(RGY_LOG_ERROR, _T("Failed to get program binary size: %s\n"), cl_errmes(err));
        return binary;
    }

    binary.resize(binary_size + 1, 0);
    err = clGetProgramInfo(m_program, CL_PROGRAM_BINARIES, binary_size, binary.data(), &binary_size);
    if (err != CL_SUCCESS) {
        CL_LOG(RGY_LOG_ERROR, _T("Failed to get program binary: %s\n"), cl_errmes(err));
        binary.clear();
    }
    binary.resize(binary_size);
    return binary;
}


static const auto RGY_CLMEMOBJ_TO_STR = make_array<std::pair<cl_mem_object_type, const TCHAR *>>(
    std::make_pair(CL_MEM_OBJECT_BUFFER,         _T("buffer")),
    std::make_pair(CL_MEM_OBJECT_IMAGE2D,        _T("image2d")),
    std::make_pair(CL_MEM_OBJECT_IMAGE3D,        _T("image3d")),
    std::make_pair(CL_MEM_OBJECT_IMAGE2D_ARRAY,  _T("image2darray")),
    std::make_pair(CL_MEM_OBJECT_IMAGE1D,        _T("image1d")),
    std::make_pair(CL_MEM_OBJECT_IMAGE1D_ARRAY,  _T("image1darray")),
    std::make_pair(CL_MEM_OBJECT_IMAGE1D_BUFFER, _T("image1dbuffer")),
    std::make_pair(CL_MEM_OBJECT_PIPE,           _T("pipe"))
);

MAP_PAIR_0_1(clmemobj, cl, cl_mem_object_type, str, const TCHAR *, RGY_CLMEMOBJ_TO_STR, 0, _T("unknown"));

static const tstring getRGYCLMemFlagsStr(cl_mem_flags mem_flags) {
    tstring str;
    if (mem_flags & CL_MEM_READ_WRITE)             str += _T(", rw");
    if (mem_flags & CL_MEM_WRITE_ONLY)             str += _T(", r");
    if (mem_flags & CL_MEM_READ_ONLY)              str += _T(", w");
    if (mem_flags & CL_MEM_USE_HOST_PTR)           str += _T(", use host ptr");
    if (mem_flags & CL_MEM_ALLOC_HOST_PTR)         str += _T(", alloc host ptr");
    if (mem_flags & CL_MEM_COPY_HOST_PTR)          str += _T(", copy host ptr");
    if (mem_flags & CL_MEM_HOST_WRITE_ONLY)        str += _T(", host write only");
    if (mem_flags & CL_MEM_HOST_READ_ONLY)         str += _T(", host read only");
    if (mem_flags & CL_MEM_HOST_NO_ACCESS)         str += _T(", host no access");
    if (mem_flags & CL_MEM_SVM_FINE_GRAIN_BUFFER)  str += _T(", svm fine grain buf");
    if (mem_flags & CL_MEM_SVM_ATOMICS)            str += _T(", svm atomics");
    if (mem_flags & CL_MEM_KERNEL_READ_AND_WRITE)  str += _T(", kernel rw");
    return (str.length() > 0) ? str.substr(2) : _T("");
}

static const auto RGY_CHANNELORDER_TO_STR = make_array<std::pair<cl_channel_order, const TCHAR *>>(
    std::make_pair(CL_R,         _T("R")),
    std::make_pair(CL_A,         _T("A")),
    std::make_pair(CL_RA,        _T("RA")),
    std::make_pair(CL_RGB,       _T("RGB")),
    std::make_pair(CL_RGBA,      _T("RGBA")),
    std::make_pair(CL_BGRA,      _T("BGRA")),
    std::make_pair(CL_ARGB,      _T("ARGB")),
    std::make_pair(CL_INTENSITY, _T("INTENSITY")),
    std::make_pair(CL_Rx,         _T("Rx")),
    std::make_pair(CL_RGx,        _T("RGx")),
    std::make_pair(CL_RGBx,       _T("RGBx")),
    std::make_pair(CL_DEPTH,      _T("DEPTH")),
    std::make_pair(CL_DEPTH_STENCIL, _T("DEPTH_STENCIL")),
    std::make_pair(CL_sRGB,       _T("sRGB")),
    std::make_pair(CL_sRGBx,      _T("sRGBx")),
    std::make_pair(CL_sRGBA,      _T("sRGBA")),
    std::make_pair(CL_sBGRA,      _T("sBGRA")),
    std::make_pair(CL_ABGR,       _T("ABGR"))
);

MAP_PAIR_0_1(clchannelorder, cl, cl_channel_order, str, const TCHAR *, RGY_CHANNELORDER_TO_STR, 0, _T("unknown"));


static const auto RGY_CHANNELTYPE_TO_STR = make_array<std::pair<cl_channel_type, const TCHAR *>>(
    std::make_pair(CL_SNORM_INT8,         _T("int8n")),
    std::make_pair(CL_SNORM_INT16,        _T("int16n")),
    std::make_pair(CL_UNORM_INT8,         _T("uint8n")),
    std::make_pair(CL_UNORM_INT16,        _T("uint16n")),
    std::make_pair(CL_UNORM_SHORT_565,    _T("ushort565n")),
    std::make_pair(CL_UNORM_SHORT_555,    _T("ushort555n")),
    std::make_pair(CL_UNORM_INT_101010,   _T("uint101010n")),
    std::make_pair(CL_SIGNED_INT8,        _T("int8")),
    std::make_pair(CL_SIGNED_INT16,       _T("int16")),
    std::make_pair(CL_SIGNED_INT32,       _T("int32")),
    std::make_pair(CL_UNSIGNED_INT8,      _T("uint8")),
    std::make_pair(CL_UNSIGNED_INT16,     _T("uint16")),
    std::make_pair(CL_UNSIGNED_INT32,     _T("uint32")),
    std::make_pair(CL_HALF_FLOAT,         _T("fp16")),
    std::make_pair(CL_FLOAT,              _T("fp32")),
    std::make_pair(CL_UNORM_INT24,        _T("uint24")),
    std::make_pair(CL_UNORM_INT_101010_2, _T("uint101010"))
);

MAP_PAIR_0_1(clchanneltype, cl, cl_channel_type, str, const TCHAR *, RGY_CHANNELTYPE_TO_STR, 0, _T("unknown"));

static bool clchannel_type_is_normalized_type(cl_channel_type type) {
    static const auto RGY_CHANNELTYPE_NORMALIZED_TYPE = make_array<cl_channel_type>(
        (cl_channel_type)CL_SNORM_INT8, (cl_channel_type)CL_SNORM_INT16,
        (cl_channel_type)CL_UNORM_INT8, (cl_channel_type)CL_UNORM_INT16,
        (cl_channel_type)CL_UNORM_SHORT_565, (cl_channel_type)CL_UNORM_SHORT_555,
        (cl_channel_type)CL_UNORM_INT_101010, (cl_channel_type)CL_UNORM_INT24, (cl_channel_type)CL_UNORM_INT_101010_2);
    return std::find(RGY_CHANNELTYPE_NORMALIZED_TYPE.begin(), RGY_CHANNELTYPE_NORMALIZED_TYPE.end(), type) != RGY_CHANNELTYPE_NORMALIZED_TYPE.end();
}


static const auto RGY_DX9_ADAPTER_TYPE_TO_STR = make_array<std::pair<cl_dx9_media_adapter_type_khr, const TCHAR *>>(
    std::make_pair(0,                      _T("none"))
#if ENABLE_RGY_OPENCL_D3D9
    ,
    std::make_pair(CL_ADAPTER_D3D9_KHR,    _T("d3d9")),
    std::make_pair(CL_ADAPTER_D3D9EX_KHR,  _T("d3d9ex")),
    std::make_pair(CL_ADAPTER_DXVA_KHR,    _T("dxva"))
#endif
);

MAP_PAIR_0_1(cldx9adaptertype, cl, cl_dx9_media_adapter_type_khr, str, const TCHAR *, RGY_DX9_ADAPTER_TYPE_TO_STR, 0, _T("unknown"));

RGYCLMemObjInfo getRGYCLMemObjectInfo(cl_mem mem) {
    if (mem == 0) {
        return RGYCLMemObjInfo();
    }
    RGYCLMemObjInfo info;
    clGetMemObjectInfo(mem, CL_MEM_TYPE, sizeof(info.memtype), &info.memtype, nullptr);
    clGetMemObjectInfo(mem, CL_MEM_FLAGS, sizeof(info.memflags), &info.memflags, nullptr);
    clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(info.size), &info.size, nullptr);
    clGetMemObjectInfo(mem, CL_MEM_HOST_PTR, sizeof(info.host_ptr), &info.host_ptr, nullptr);
    clGetMemObjectInfo(mem, CL_MEM_MAP_COUNT, sizeof(info.map_count), &info.map_count, nullptr);
    clGetMemObjectInfo(mem, CL_MEM_REFERENCE_COUNT, sizeof(info.ref_count), &info.ref_count, nullptr);
    clGetMemObjectInfo(mem, CL_MEM_OFFSET, sizeof(info.mem_offset), &info.mem_offset, nullptr);
    clGetMemObjectInfo(mem, CL_MEM_CONTEXT, sizeof(info.context), &info.context, nullptr);
    clGetMemObjectInfo(mem, CL_MEM_ASSOCIATED_MEMOBJECT, sizeof(info.associated_mem), &info.associated_mem, nullptr);
    //clGetMemObjectInfo(mem, CL_​MEM_​USES_​SVM_​POINTER, sizeof(info.is_svm_ptr), &info.is_svm_ptr, nullptr);
    #if ENABLE_RGY_OPENCL_D3D9
    clGetMemObjectInfo(mem, CL_MEM_DX9_MEDIA_ADAPTER_TYPE_KHR, sizeof(info.d3d9_adapter_type), &info.d3d9_adapter_type, nullptr);
    clGetMemObjectInfo(mem, CL_MEM_DX9_MEDIA_SURFACE_INFO_KHR, sizeof(info.d3d9_surf_type), &info.d3d9_surf_type, nullptr);
    #endif
    #if ENABLE_RGY_OPENCL_D3D11
    clGetMemObjectInfo(mem, CL_MEM_D3D11_RESOURCE_KHR, sizeof(info.d3d11resource), &info.d3d11resource, nullptr);
    #endif
    #if ENABLE_RGY_OPENCL_VA
    clGetMemObjectInfo(mem, CL_MEM_VA_API_MEDIA_SURFACE_INTEL, sizeof(info.va_surfaceId), &info.va_surfaceId, nullptr);
    #endif

    switch (info.memtype) {
    case CL_MEM_OBJECT_IMAGE2D:
    case CL_MEM_OBJECT_IMAGE3D:
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
    case CL_MEM_OBJECT_IMAGE1D:
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
    case CL_MEM_OBJECT_IMAGE1D_BUFFER:
        clGetImageInfo(mem, CL_IMAGE_FORMAT, sizeof(info.image_format), &info.image_format, nullptr);
        clGetImageInfo(mem, CL_IMAGE_ELEMENT_SIZE, sizeof(info.image_elem_size), &info.image_elem_size, nullptr);
        clGetImageInfo(mem, CL_IMAGE_ROW_PITCH, sizeof(info.image.image_row_pitch), &info.image.image_row_pitch, nullptr);
        clGetImageInfo(mem, CL_IMAGE_SLICE_PITCH, sizeof(info.image.image_slice_pitch), &info.image.image_slice_pitch, nullptr);
        clGetImageInfo(mem, CL_IMAGE_WIDTH, sizeof(info.image.image_width), &info.image.image_width, nullptr);
        clGetImageInfo(mem, CL_IMAGE_HEIGHT, sizeof(info.image.image_height), &info.image.image_height, nullptr);
        clGetImageInfo(mem, CL_IMAGE_DEPTH, sizeof(info.image.image_depth), &info.image.image_depth, nullptr);
        clGetImageInfo(mem, CL_IMAGE_ARRAY_SIZE, sizeof(info.image.image_array_size), &info.image.image_array_size, nullptr);
        clGetImageInfo(mem, CL_IMAGE_BUFFER, sizeof(info.image.buffer), &info.image.buffer, nullptr);
        clGetImageInfo(mem, CL_IMAGE_NUM_MIP_LEVELS, sizeof(info.image.num_mip_levels), &info.image.num_mip_levels, nullptr);
        clGetImageInfo(mem, CL_IMAGE_NUM_SAMPLES, sizeof(info.image.num_samples), &info.image.num_samples, nullptr);
        //clGetImageInfo(mem, CL_IMAGE_D3D10_SUBRESOURCE_KHR, sizeof(info.d3d11resource), &info.d3d11resource, nullptr);
        #if ENABLE_RGY_OPENCL_D3D9
        clGetImageInfo(mem, CL_IMAGE_DX9_MEDIA_PLANE_KHR, sizeof(info.d3d9_media_plane), &info.d3d9_media_plane, nullptr);
        //clGetImageInfo(mem, CL_IMAGE_DX9_MEDIA_SURFACE_PLANE_KHR, sizeof(info.d3d11resource), &info.d3d11resource, nullptr);
        #endif //ENABLE_RGY_OPENCL_D3D9
        #if ENABLE_RGY_OPENCL_D3D11
        clGetImageInfo(mem, CL_IMAGE_D3D11_SUBRESOURCE_KHR, sizeof(info.d3d11subresource), &info.d3d11subresource, nullptr);
        #endif //#if ENABLE_RGY_OPENCL_D3D11
        #if ENABLE_RGY_OPENCL_VA
        clGetImageInfo(mem, CL_IMAGE_VA_API_PLANE_INTEL, sizeof(info.va_plane), &info.va_plane, nullptr);
        #endif
        break;
    default:
        break;
    }
    return info;
}

tstring RGYCLMemObjInfo::print() const {
    tstring str;
    str += strsprintf(_T("memtype:          %s\n"), clmemobj_cl_to_str(memtype));
    str += strsprintf(_T("flags:            %s\n"), getRGYCLMemFlagsStr(memtype).c_str());
    str += strsprintf(_T("size:             %zu\n"), size);
    str += strsprintf(_T("map count:        %d\n"), map_count);
    str += strsprintf(_T("ref count:        %d\n"), ref_count);
    str += strsprintf(_T("offset:           %zu\n"), mem_offset);
    str += strsprintf(_T("host ptr:         0x%p\n"), host_ptr);
    str += strsprintf(_T("context:          0x%p\n"), context);
    str += strsprintf(_T("associated mem:   0x%p\n"), associated_mem);
    str += strsprintf(_T("is_svm_ptr:       %s\n"), is_svm_ptr ? _T("yes") : _T("no"));
    str += strsprintf(_T("dx9 adapter type: %s\n"), cldx9adaptertype_cl_to_str(d3d9_adapter_type));
    str += strsprintf(_T("dx9 resource:     resource: %p, handle: %p\n"), d3d9_surf_type.resource, d3d9_surf_type.shared_handle);
    str += strsprintf(_T("dx11 resource:    %p\n"), d3d11resource);
    str += strsprintf(_T("va surfaceId:     %p\n"), va_surfaceId);
    if (image_format.image_channel_order != 0) {
        str += strsprintf(_T("image\n"));
        str += strsprintf(_T("data type:        %s\n"), clchanneltype_cl_to_str(image_format.image_channel_data_type));
        str += strsprintf(_T("channel order:    %s\n"), clchannelorder_cl_to_str(image_format.image_channel_order));
        str += strsprintf(_T("elem size:        %zu\n"), image_elem_size);
        str += strsprintf(_T("width:            %zu\n"), image.image_width);
        str += strsprintf(_T("height:           %zu\n"), image.image_height);
        str += strsprintf(_T("depth:            %zu\n"), image.image_depth);
        str += strsprintf(_T("row pitch:        %zu\n"), image.image_row_pitch);
        str += strsprintf(_T("slice pitch:      %zu\n"), image.image_slice_pitch);
        str += strsprintf(_T("array size:       %zu\n"), image.image_array_size);
        str += strsprintf(_T("buffer:           0x%p\n"), image.buffer);
        str += strsprintf(_T("num mip levels:   %zu\n"), image.num_mip_levels);
        str += strsprintf(_T("num samples:      %zu\n"), image.num_samples);
        str += strsprintf(_T("dx9 plane:        %d\n"), d3d9_media_plane);
        str += strsprintf(_T("dx11 subresource: 0x%p\n"), d3d11subresource);
        str += strsprintf(_T("va plane:         0x%p\n"), va_plane);
    }
    return str;
}

bool RGYCLMemObjInfo::isImageNormalizedType() const {
    if (image_format.image_channel_order == 0) return false;
    return clchannel_type_is_normalized_type(image_format.image_channel_data_type);
}

RGY_ERR RGYCLBufMap::map(cl_map_flags map_flags, size_t size, RGYOpenCLQueue &queue) {
    return map(map_flags, size, queue, {}, RGY_CL_MAP_BLOCK_NONE);
}

RGY_ERR RGYCLBufMap::map(cl_map_flags map_flags, size_t size, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, const RGYCLMapBlock block_map) {
    m_queue = queue.get();
    const std::vector<cl_event> v_wait_list = toVec(wait_events);
    const cl_event *wait_list = (v_wait_list.size() > 0) ? v_wait_list.data() : nullptr;
    cl_int err = 0;
    m_hostPtr = clEnqueueMapBuffer(m_queue, m_mem, (block_map != RGY_CL_MAP_BLOCK_NONE) ? CL_TRUE : CL_FALSE, map_flags, 0, size, (int)v_wait_list.size(), wait_list, m_eventMap.reset_ptr(), &err);
    return err_cl_to_rgy(err);
}

RGY_ERR RGYCLBufMap::unmap() {
    return unmap(m_queue, {});
}
RGY_ERR RGYCLBufMap::unmap(RGYOpenCLQueue &queue) {
    return unmap(queue, {});
}
RGY_ERR RGYCLBufMap::unmap(RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    return unmap(queue.get(), wait_events);
}
RGY_ERR RGYCLBufMap::unmap(cl_command_queue queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    if (!m_hostPtr) return RGY_ERR_NONE;
    m_queue = queue;
    const std::vector<cl_event> v_wait_list = toVec(wait_events);
    const cl_event *wait_list = (v_wait_list.size() > 0) ? v_wait_list.data() : nullptr;
    auto err = err_cl_to_rgy(clEnqueueUnmapMemObject(m_queue, m_mem, m_hostPtr, (int)v_wait_list.size(), wait_list, m_eventMap.reset_ptr()));
    m_hostPtr = nullptr;
    return err;
}

RGY_ERR RGYCLBuf::queueMapBuffer(RGYOpenCLQueue &queue, cl_map_flags map_flags, const std::vector<RGYOpenCLEvent> &wait_events, const RGYCLMapBlock block_map) {
    m_mapped = std::make_unique<RGYCLBufMap>(m_mem);
    return m_mapped->map(map_flags, m_size, queue, wait_events, block_map);
}

RGY_ERR RGYCLBuf::unmapBuffer() {
    auto err = m_mapped->unmap();
    m_mapped.reset();
    return err;
}
RGY_ERR RGYCLBuf::unmapBuffer(RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    auto err = m_mapped->unmap(queue, wait_events);
    m_mapped.reset();
    return err;
}

RGYCLMemObjInfo RGYCLBuf::getMemObjectInfo() const {
    return getRGYCLMemObjectInfo(m_mem);
}

RGYCLFrameMap::RGYCLFrameMap(RGYCLFrame *dev, RGYOpenCLQueue &queue) : m_dev(dev), m_queue(queue.get()), m_eventMap() {};

RGY_ERR RGYCLFrameMap::map(cl_map_flags map_flags, RGYOpenCLQueue& queue) {
    return map(map_flags, queue, {}, RGY_CL_MAP_BLOCK_NONE);
}

RGY_ERR RGYCLFrameMap::map(cl_map_flags map_flags, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, const RGYCLMapBlock block_map) {
    std::vector<cl_event> v_wait_list = toVec(wait_events);
    cl_event *wait_list = (v_wait_list.size() > 0) ? v_wait_list.data() : nullptr;
    frame = m_dev->frameInfo();
    m_queue = queue.get();
    for (int i = 0; i < _countof(frame.ptr); i++) {
        frame.ptr[i] = nullptr;
    }
    if (m_eventMap.size() != RGY_CSP_PLANES[m_dev->frame.csp]) {
        m_eventMap.resize(RGY_CSP_PLANES[m_dev->frame.csp]);
    }
    for (int i = 0; i < RGY_CSP_PLANES[frame.csp]; i++) {
        const auto plane = getPlane(&m_dev->frame, (RGY_PLANE)i);
        cl_int err = 0;
        cl_bool block = CL_FALSE;
        switch (block_map) {
            case RGY_CL_MAP_BLOCK_ALL: block = CL_TRUE; break;
            case RGY_CL_MAP_BLOCK_LAST: { if (i == (RGY_CSP_PLANES[m_dev->frame.csp]-1)) block = CL_TRUE; } break;
            case RGY_CL_MAP_BLOCK_NONE:
            default: break;
        }
        size_t size = (size_t)plane.pitch[0] * plane.height;
        frame.ptr[i] = (uint8_t *)clEnqueueMapBuffer(m_queue, (cl_mem)plane.ptr[0], block, map_flags, 0, size, (int)v_wait_list.size(), wait_list, m_eventMap[i].reset_ptr(), &err);
        if (err != 0) {
            return err_cl_to_rgy(err);
        }
        v_wait_list.clear();
        wait_list = nullptr;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYCLFrameMap::unmap() {
    return unmap(m_queue, {});
}
RGY_ERR RGYCLFrameMap::unmap(RGYOpenCLQueue &queue) {
    return unmap(queue, {});
}
RGY_ERR RGYCLFrameMap::unmap(RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    return unmap(queue.get(), wait_events);
}
RGY_ERR RGYCLFrameMap::unmap(cl_command_queue queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    std::vector<cl_event> v_wait_list = toVec(wait_events);
    cl_event *wait_list = (v_wait_list.size() > 0) ? v_wait_list.data() : nullptr;
    m_queue = queue;
    for (int i = 0; i < _countof(frame.ptr); i++) {
        if (frame.ptr[i]) {
            auto err = err_cl_to_rgy(clEnqueueUnmapMemObject(m_queue, (cl_mem)m_dev->frame.ptr[i], frame.ptr[i], (int)v_wait_list.size(), wait_list, m_eventMap[i].reset_ptr()));
            v_wait_list.clear();
            wait_list = nullptr;
            frame.ptr[i] = nullptr;
            if (err != RGY_ERR_NONE) {
                return err_cl_to_rgy(err);
            }
        }
    }
    return RGY_ERR_NONE;
}

void RGYCLFrameMap::setTimestamp(uint64_t timestamp) { frame.timestamp = timestamp; m_dev->setTimestamp(timestamp); }
void RGYCLFrameMap::setDuration(uint64_t duration) { frame.duration = duration; m_dev->setDuration(duration); }
void RGYCLFrameMap::setPicstruct(RGY_PICSTRUCT picstruct) { frame.picstruct = picstruct; m_dev->setPicstruct(picstruct); }
void RGYCLFrameMap::setInputFrameId(int id) { frame.inputFrameId = id; m_dev->setInputFrameId(id);}
void RGYCLFrameMap::setFlags(RGY_FRAME_FLAGS frameflags) { frame.flags = frameflags; m_dev->setTimestamp(frameflags); }
void RGYCLFrameMap::clearDataList() { frame.dataList.clear(); m_dev->clearDataList(); }
const std::vector<std::shared_ptr<RGYFrameData>>& RGYCLFrameMap::dataList() const { return m_dev->dataList(); }
std::vector<std::shared_ptr<RGYFrameData>>& RGYCLFrameMap::dataList() { return m_dev->dataList(); }
void RGYCLFrameMap::setDataList(const std::vector<std::shared_ptr<RGYFrameData>>& dataList) { m_dev->setDataList(dataList); }

RGY_ERR RGYCLFrame::queueMapBuffer(RGYOpenCLQueue &queue, cl_map_flags map_flags, const std::vector<RGYOpenCLEvent> &wait_events, const RGYCLMapBlock block_map) {
    m_mapped = std::make_unique<RGYCLFrameMap>(this, queue);
    return m_mapped->map(map_flags, queue, wait_events, block_map);
}

RGY_ERR RGYCLFrame::unmapBuffer() {
    return (m_mapped) ? m_mapped->unmap() : RGY_ERR_NONE;
}
RGY_ERR RGYCLFrame::unmapBuffer(RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    return (m_mapped) ? m_mapped->unmap(queue, wait_events) : RGY_ERR_NONE;
}

RGY_ERR RGYCLFrame::mapWait() const { return m_mapped->map_wait(); }

bool RGYCLFrame::isMapped() const { return m_mapped != nullptr;  }

RGYCLFrameMap *RGYCLFrame::mappedHost() { return m_mapped.get(); }

const RGYCLFrameMap *RGYCLFrame::mappedHost() const { return m_mapped.get(); }

std::vector<RGYOpenCLEvent>& RGYCLFrame::mapEvents() { return m_mapped->mapEvents(); }

void RGYCLFrame::clear() {
    m_mapped.reset();
    for (int i = 0; i < _countof(frame.ptr); i++) {
        if (mem(i)) {
            clReleaseMemObject(mem(i));
        }
        frame.ptr[i] = nullptr;
        frame.pitch[i] = 0;
    }
}

RGYCLMemObjInfo RGYCLFrame::getMemObjectInfo() const {
    return getRGYCLMemObjectInfo(mem(0));
}

void RGYCLFrame::resetMappedFrame() { m_mapped.reset(); }

RGY_ERR RGYCLFrameInterop::acquire(RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    cl_event *event_ptr = (event) ? event->reset_ptr() : nullptr;
    cl_int err = CL_SUCCESS;
#if ENABLE_RGY_OPENCL_D3D9
    if (m_interop == RGY_INTEROP_DX9) {
        err = clEnqueueAcquireDX9MediaSurfacesKHR(queue.get(), RGY_CSP_PLANES[frame.csp], (cl_mem *)frame.ptr, 0, nullptr, event_ptr);
    } else
#endif
#if ENABLE_RGY_OPENCL_D3D11
    if (m_interop == RGY_INTEROP_DX11) {
        err = clEnqueueAcquireD3D11ObjectsKHR(queue.get(), RGY_CSP_PLANES[frame.csp], (cl_mem *)frame.ptr, 0, nullptr, event_ptr);
    } else
#endif
#if ENABLE_RGY_OPENCL_VA
    if (m_interop == RGY_INTEROP_VA) {
        err = clEnqueueAcquireVA_APIMediaSurfacesINTEL(queue.get(), RGY_CSP_PLANES[frame.csp], (cl_mem *)frame.ptr, 0, nullptr, event_ptr);
    } else
#endif
    {
        CL_LOG(RGY_LOG_ERROR, _T("RGYCLFrameInterop::acquire: Unknown interop type!\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (err != 0) {
        CL_LOG(RGY_LOG_ERROR, _T("RGYCLFrameInterop::acquire: Failed to acquire object: %s!\n"), cl_errmes(err));
        return err_cl_to_rgy(err);
    }
    m_acquired = true;
    return RGY_ERR_NONE;
}

RGY_ERR RGYCLFrameInterop::release(RGYOpenCLEvent *event) {
    if (m_acquired) {
        cl_event *event_ptr = (event) ? event->reset_ptr() : nullptr;
        cl_int err = CL_SUCCESS;
#if ENABLE_RGY_OPENCL_D3D9
        if (m_interop == RGY_INTEROP_DX9) {
            err = clEnqueueReleaseDX9MediaSurfacesKHR(m_interop_queue.get(), RGY_CSP_PLANES[frame.csp], (cl_mem *)frame.ptr, 0, nullptr, event_ptr);
        } else
#endif
#if ENABLE_RGY_OPENCL_D3D11
        if (m_interop == RGY_INTEROP_DX11) {
            err = clEnqueueReleaseD3D11ObjectsKHR(m_interop_queue.get(), RGY_CSP_PLANES[frame.csp], (cl_mem *)frame.ptr, 0, nullptr, event_ptr);
        } else
#endif
#if ENABLE_RGY_OPENCL_VA
        if (m_interop == RGY_INTEROP_VA) {
            err = clEnqueueReleaseVA_APIMediaSurfacesINTEL(m_interop_queue.get(), RGY_CSP_PLANES[frame.csp], (cl_mem *)frame.ptr, 0, nullptr, event_ptr);
        } else
#endif
        {
            CL_LOG(RGY_LOG_ERROR, _T("RGYCLFrameInterop::release: Unknown interop type!\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        if (err != 0) {
            CL_LOG(RGY_LOG_ERROR, _T("RGYCLFrameInterop::acquire: Failed to acquire object: %s!\n"), cl_errmes(err));
            return err_cl_to_rgy(err);
        }
        m_acquired = false;
    }
    return RGY_ERR_NONE;
}

RGYCLImageFromBufferDeleter::RGYCLImageFromBufferDeleter() : m_pool(nullptr) {};
RGYCLImageFromBufferDeleter::RGYCLImageFromBufferDeleter(RGYCLFramePool *pool) : m_pool(pool) {};

void RGYCLImageFromBufferDeleter::operator()(RGYCLFrame *frame) {
    if (frame) {
        if (m_pool) {
            m_pool->add(frame);
        } else {
            delete frame;
        }
        frame = nullptr;
    }
}

RGYCLFramePool::RGYCLFramePool() : m_pool() {};
RGYCLFramePool::~RGYCLFramePool() {
    clear();
};
void RGYCLFramePool::clear() {
    m_pool.clear();
};

void RGYCLFramePool::add(RGYCLFrame *frame) {
    if (frame) {
        m_pool.push_back(std::unique_ptr<RGYCLFrame>(frame));
    }
}

std::unique_ptr<RGYCLFrame, RGYCLImageFromBufferDeleter> RGYCLFramePool::get(const RGYFrameInfo &frame, const bool normalized, const cl_mem_flags clflags) {
    const auto target_mem_type = (normalized) ? RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED : RGY_MEM_TYPE_GPU_IMAGE;
    for (auto it = m_pool.begin(); it != m_pool.end(); it++) {
        auto& poolFrame = (*it);
        if (!cmpFrameInfoCspResolution(&poolFrame->frame, &frame)
            && poolFrame->frame.mem_type == target_mem_type
            && poolFrame->clflags == clflags) {
            auto f = std::move(*it);
            m_pool.erase(it);
            return std::unique_ptr<RGYCLFrame, RGYCLImageFromBufferDeleter>(f.release(), RGYCLImageFromBufferDeleter(this));
        }
    }
    return nullptr;
}


tstring clcommandqueueproperties_cl_to_str(const cl_command_queue_properties prop) {
    tstring str;
    if (prop & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) str += _T(", OoO exec");
    if (prop & CL_QUEUE_PROFILING_ENABLE)              str += _T(", profiling enable");
    if (prop & CL_QUEUE_ON_DEVICE)                     str += _T(", on device");
    if (prop & CL_QUEUE_ON_DEVICE_DEFAULT)             str += _T(", on device default");
    return str.substr(2);
}

RGYOpenCLQueueInfo::RGYOpenCLQueueInfo() : context(0), devid(0), refcount(0), properties(0) {};

tstring RGYOpenCLQueueInfo::print() const {
    tstring str;
    str += strsprintf(_T("context:          0x%p\n"), context);
    str += strsprintf(_T("devid:            0x%p\n"), devid);
    str += strsprintf(_T("refcount:         %d\n"), refcount);
    str += strsprintf(_T("properties:       %s\n"), clcommandqueueproperties_cl_to_str(properties).c_str());
    return str;
}

RGYOpenCLQueue::RGYOpenCLQueue() : m_queue(nullptr, clReleaseCommandQueue), m_devid(0) {};

RGYOpenCLQueue::RGYOpenCLQueue(cl_command_queue queue, cl_device_id devid) : m_queue(queue, clReleaseCommandQueue), m_devid(devid) {};

RGYOpenCLQueue::~RGYOpenCLQueue() {
    m_queue.reset();
}

RGYOpenCLQueueInfo RGYOpenCLQueue::getInfo() const {
    RGYOpenCLQueueInfo info;
    try {
        clGetInfo(clGetCommandQueueInfo, get(), CL_QUEUE_CONTEXT, &info.context);
        clGetInfo(clGetCommandQueueInfo, get(), CL_QUEUE_DEVICE, &info.devid);
        clGetInfo(clGetCommandQueueInfo, get(), CL_QUEUE_REFERENCE_COUNT, &info.refcount);
        clGetInfo(clGetCommandQueueInfo, get(), CL_QUEUE_PROPERTIES, &info.properties);
    } catch (...) {
        return RGYOpenCLQueueInfo();
    }
    return info;
}

cl_command_queue_properties RGYOpenCLQueue::getProperties() const {
    return getInfo().properties;
}

RGY_ERR RGYOpenCLQueue::wait(const RGYOpenCLEvent& event) const {
    return err_cl_to_rgy(clEnqueueWaitForEvents(m_queue.get(), 1, event.ptr()));
}

RGY_ERR RGYOpenCLQueue::getmarker(RGYOpenCLEvent& event) const {
    return err_cl_to_rgy(clEnqueueMarker(m_queue.get(), event.reset_ptr()));
}

RGY_ERR RGYOpenCLQueue::flush() const {
    if (!m_queue) {
        return RGY_ERR_NULL_PTR;
    }
    return err_cl_to_rgy(clFlush(m_queue.get()));
}

RGY_ERR RGYOpenCLQueue::finish() const {
    if (!m_queue) {
        return RGY_ERR_NULL_PTR;
    }
    return err_cl_to_rgy(clFinish(m_queue.get()));
}

void RGYOpenCLQueue::clear() {
    m_queue.reset();
}

std::vector<cl_image_format> RGYOpenCLContext::getSupportedImageFormats(const cl_mem_object_type image_type) const {
    std::vector<cl_image_format> result;
    cl_uint num_formats = 0;
    cl_int err = CL_SUCCESS;
    if ((err = clGetSupportedImageFormats(m_context.get(), CL_MEM_READ_WRITE, image_type, 0, nullptr, &num_formats)) != CL_SUCCESS) {
        return result;
    }
    result.resize(num_formats);
    if ((err = clGetSupportedImageFormats(m_context.get(), CL_MEM_READ_WRITE, image_type, (cl_uint)result.size(), result.data(), &num_formats)) != CL_SUCCESS) {
        result.clear();
    }
    return result;
}

tstring RGYOpenCLContext::getSupportedImageFormatsStr(const cl_mem_object_type image_type) const {
    const auto formatList = getSupportedImageFormats(image_type);
    tstring str;
    for (auto& format : formatList) {
        str += strsprintf(_T("%s: %s\n"), clchannelorder_cl_to_str(format.image_channel_order), clchanneltype_cl_to_str(format.image_channel_data_type));
    }
    return str;
}

std::string RGYOpenCLContext::cspCopyOptions(const RGYFrameInfo& dst, const RGYFrameInfo& src) const {
    const auto options = strsprintf("-D MEM_TYPE_SRC=%d -D MEM_TYPE_DST=%d -D in_bit_depth=%d -D out_bit_depth=%d"
        " -D RGY_MATRIX_ST170_M=%d"
        " -D RGY_MATRIX_ST240_M=%d"
        " -D RGY_MATRIX_BT2020_NCL=%d"
        " -D RGY_MATRIX_BT2020_CL=%d"
        " -D RGY_MATRIX_BT709=%d",
        src.mem_type,
        dst.mem_type,
        RGY_CSP_BIT_DEPTH[src.csp],
        RGY_CSP_BIT_DEPTH[dst.csp],
        RGY_MATRIX_ST170_M,
        RGY_MATRIX_ST240_M,
        RGY_MATRIX_BT2020_NCL,
        RGY_MATRIX_BT2020_CL,
        RGY_MATRIX_BT709);
    return options;
}

void RGYOpenCLContext::requestCSPCopy(const RGYFrameInfo& dst, const RGYFrameInfo& src) {
    const auto options = cspCopyOptions(dst, src);
    if (m_copy.count(options) == 0) {
        m_copy[options].set(buildResourceAsync(_T("RGY_FILTER_CL"), _T("EXE_DATA"), options.c_str()));
    }
}

RGYOpenCLProgram *RGYOpenCLContext::getCspCopyProgram(const RGYFrameInfo& dst, const RGYFrameInfo& src) {
    const auto options = cspCopyOptions(dst, src);
    if (m_copy.count(options) == 0) {
        requestCSPCopy(dst, src);
    }
    return (m_copy.count(options) != 0) ? m_copy[options].get() : nullptr;
}

RGY_ERR RGYOpenCLContext::copyPlane(RGYFrameInfo *dst, const RGYFrameInfo *src) {
    return copyPlane(dst, src, nullptr);
}
RGY_ERR RGYOpenCLContext::copyPlane(RGYFrameInfo *dst, const RGYFrameInfo *src, const sInputCrop *srcCrop) {
    return copyPlane(dst, src, srcCrop, m_queue[0]);
}
RGY_ERR RGYOpenCLContext::copyPlane(RGYFrameInfo *dst, const RGYFrameInfo *src, const sInputCrop *srcCrop, RGYOpenCLQueue &queue) {
    return copyPlane(dst, src, srcCrop, queue, {}, nullptr);
}
RGY_ERR RGYOpenCLContext::copyPlane(RGYFrameInfo *dst, const RGYFrameInfo *src, const sInputCrop *srcCrop, RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    return copyPlane(dst, src, srcCrop, queue, {}, event);
}

RGY_ERR RGYOpenCLContext::copyPlane(RGYFrameInfo *planeDstOrg, const RGYFrameInfo *planeSrcOrg, const sInputCrop *planeCrop, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event, RGYFrameCopyMode copyMode) {
    cl_int err = CL_SUCCESS;
    const std::vector<cl_event> v_wait_list = toVec(wait_events);
    const int wait_count = (int)v_wait_list.size();
    const cl_event *wait_list = (wait_count > 0) ? v_wait_list.data() : nullptr;
    cl_event *event_ptr = (event) ? event->reset_ptr() : nullptr;

    const int pixel_size = RGY_CSP_BIT_DEPTH[planeDstOrg->csp] > 8 ? 2 : 1;
    RGYFrameInfo planeDst = *planeDstOrg;
    RGYFrameInfo planeSrc = *planeSrcOrg;
    if (copyMode != RGYFrameCopyMode::FRAME) {
        planeDst.pitch[0] <<= 1;
        planeDst.height >>= 1;
        planeSrc.pitch[0] <<= 1;
        planeSrc.height >>= 1;
    }
    size_t dst_origin[3] = { (copyMode == RGYFrameCopyMode::FIELD_BOTTOM) ? (size_t)planeDstOrg->pitch[0] : 0, 0, 0 };
    size_t src_origin[3] = { (copyMode == RGYFrameCopyMode::FIELD_BOTTOM) ? (size_t)planeSrcOrg->pitch[0] : 0, 0, 0 };
    if (planeCrop) {
        src_origin[0] += planeCrop->e.left * pixel_size;
        src_origin[1] += planeCrop->e.up;
    }
    size_t region[3] = { (size_t)planeSrc.width * pixel_size, (size_t)planeSrc.height, 1 };
    if (planeSrc.mem_type == RGY_MEM_TYPE_GPU) {
        if (planeDst.mem_type == RGY_MEM_TYPE_GPU) {
            if (planeDst.csp == planeSrc.csp) {
                err = clEnqueueCopyBufferRect(queue.get(), (cl_mem)planeSrc.ptr[0], (cl_mem)planeDst.ptr[0], src_origin, dst_origin,
                    region, planeSrc.pitch[0], 0, planeDst.pitch[0], 0, wait_count, wait_list, event_ptr);
            } else {
                auto copyProgram = getCspCopyProgram(planeDst, planeSrc);
                if (!copyProgram) {
                    CL_LOG(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_CL(B2B)\n"));
                    return RGY_ERR_OPENCL_CRUSH;
                }
                RGYWorkSize local(32, 8);
                RGYWorkSize global(planeDst.width, planeDst.height);
                auto rgy_err = copyProgram->kernel("kernel_copy_plane").config(queue, local, global, wait_events, event).launch(
                    (cl_mem)planeDst.ptr[0], planeDst.pitch[0], (int)dst_origin[0] / pixel_size, (int)dst_origin[1],
                    (cl_mem)planeSrc.ptr[0], planeSrc.pitch[0], (int)src_origin[0] / pixel_size, (int)src_origin[1],
                    planeSrc.width, planeSrc.height);
                err = err_rgy_to_cl(rgy_err);
            }
        } else if (planeDst.mem_type == RGY_MEM_TYPE_GPU_IMAGE || planeDst.mem_type == RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED) {
            auto copyProgram = getCspCopyProgram(planeDst, planeSrc);
            if (!copyProgram) {
                CL_LOG(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_CL(B2I)\n"));
                return RGY_ERR_OPENCL_CRUSH;
            }
            RGYWorkSize local(32, 8);
            RGYWorkSize global(planeDst.width, planeDst.height);
            auto rgy_err = copyProgram->kernel("kernel_copy_plane").config(queue, local, global, wait_events, event).launch(
                (cl_mem)planeDst.ptr[0], planeDst.pitch[0], (int)dst_origin[0] / pixel_size, (int)dst_origin[1],
                (cl_mem)planeSrc.ptr[0], planeSrc.pitch[0], (int)src_origin[0] / pixel_size, (int)src_origin[1],
                planeSrc.width, planeSrc.height);
            err = err_rgy_to_cl(rgy_err);
        } else if (planeDst.mem_type == RGY_MEM_TYPE_CPU
#if ENCODER_MPP
                || planeDst.mem_type == RGY_MEM_TYPE_MPP
#endif
        ) {
            err = clEnqueueReadBufferRect(queue.get(), (cl_mem)planeSrc.ptr[0], false, src_origin, dst_origin,
                region, planeSrc.pitch[0], 0, planeDst.pitch[0], 0, planeDst.ptr[0], wait_count, wait_list, event_ptr);
        } else {
            return RGY_ERR_UNSUPPORTED;
        }
    } else if (planeSrc.mem_type == RGY_MEM_TYPE_GPU_IMAGE || planeSrc.mem_type == RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED) {
        if (planeDst.mem_type == RGY_MEM_TYPE_GPU) {
            auto copyProgram = getCspCopyProgram(planeDst, planeSrc);
            if (!copyProgram) {
                CL_LOG(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_CL(I2B)\n"));
                return RGY_ERR_OPENCL_CRUSH;
            }
            RGYWorkSize local(32, 8);
            RGYWorkSize global(planeDst.width, planeDst.height);
            auto rgy_err = copyProgram->kernel("kernel_copy_plane").config(queue, local, global, wait_events, event).launch(
                (cl_mem)planeDst.ptr[0], planeDst.pitch[0], (int)dst_origin[0] / pixel_size, (int)dst_origin[1],
                (cl_mem)planeSrc.ptr[0], planeSrc.pitch[0], (int)src_origin[0] / pixel_size, (int)src_origin[1],
                planeSrc.width, planeSrc.height);
            err = err_rgy_to_cl(rgy_err);
        } else if (planeDst.mem_type == RGY_MEM_TYPE_GPU_IMAGE || planeDst.mem_type == RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED) {
            if (planeDst.csp == planeSrc.csp) {
                clGetImageInfo((cl_mem)planeDst.ptr[0], CL_IMAGE_WIDTH, sizeof(region[0]), &region[0], nullptr);
                err = clEnqueueCopyImage(queue.get(), (cl_mem)planeSrc.ptr[0], (cl_mem)planeDst.ptr[0], src_origin, dst_origin, region, wait_count, wait_list, event_ptr);
            } else {
                auto copyProgram = getCspCopyProgram(planeDst, planeSrc);
                if (!copyProgram) {
                    CL_LOG(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_CL(I2I)\n"));
                    return RGY_ERR_OPENCL_CRUSH;
                }
                RGYWorkSize local(32, 8);
                RGYWorkSize global(planeDst.width, planeDst.height);
                auto rgy_err = copyProgram->kernel("kernel_copy_plane").config(queue, local, global, wait_events, event).launch(
                    (cl_mem)planeDst.ptr[0], planeDst.pitch[0], (int)dst_origin[0] / pixel_size, (int)dst_origin[1],
                    (cl_mem)planeSrc.ptr[0], planeSrc.pitch[0], (int)src_origin[0] / pixel_size, (int)src_origin[1],
                    planeSrc.width, planeSrc.height);
                err = err_rgy_to_cl(rgy_err);
            }
        } else if (planeDst.mem_type == RGY_MEM_TYPE_CPU
#if ENCODER_MPP
                || planeDst.mem_type == RGY_MEM_TYPE_MPP
#endif
        ) {
            clGetImageInfo((cl_mem)planeSrc.ptr[0], CL_IMAGE_WIDTH, sizeof(region[0]), &region[0], nullptr);
            err = clEnqueueReadImage(queue.get(), (cl_mem)planeSrc.ptr[0], false, dst_origin,
                region, planeDst.pitch[0], 0, planeDst.ptr[0], wait_count, wait_list, event_ptr);
        } else {
            return RGY_ERR_UNSUPPORTED;
        }
    } else if (planeSrc.mem_type == RGY_MEM_TYPE_CPU
#if ENCODER_MPP
            || planeSrc.mem_type == RGY_MEM_TYPE_MPP
#endif
    ) {
        if (planeDst.mem_type == RGY_MEM_TYPE_GPU) {
            err = clEnqueueWriteBufferRect(queue.get(), (cl_mem)planeDst.ptr[0], false, dst_origin, src_origin,
                region, planeDst.pitch[0], 0, planeSrc.pitch[0], 0, planeSrc.ptr[0], wait_count, wait_list, event_ptr);
        } else if (planeDst.mem_type == RGY_MEM_TYPE_GPU_IMAGE) {
            clGetImageInfo((cl_mem)planeDst.ptr[0], CL_IMAGE_WIDTH, sizeof(region[0]), &region[0], nullptr);
            err = clEnqueueWriteImage(queue.get(), (cl_mem)planeDst.ptr[0], false, src_origin,
                region, planeSrc.pitch[0], 0, (void *)planeSrc.ptr[0], wait_count, wait_list, event_ptr);
        } else if (planeDst.mem_type == RGY_MEM_TYPE_CPU
#if ENCODER_MPP
                || planeDst.mem_type == RGY_MEM_TYPE_MPP
#endif
        ) {
            for (int y = 0; y < planeDst.height; y++) {
                memcpy(planeDst.ptr[0] + (y + dst_origin[1]) * planeDst.pitch[0] + dst_origin[0] * pixel_size,
                        planeSrc.ptr[0] + (y + src_origin[1]) * planeSrc.pitch[0] + src_origin[0] * pixel_size,
                        planeDst.width * pixel_size);
            }
        } else {
            return RGY_ERR_UNSUPPORTED;
        }
    } else {
        return RGY_ERR_UNSUPPORTED;
    }
    return err_cl_to_rgy(err);
}
RGY_ERR RGYOpenCLContext::copyFrame(RGYFrameInfo *dst, const RGYFrameInfo *src) {
    return copyFrame(dst, src, nullptr);
}
RGY_ERR RGYOpenCLContext::copyFrame(RGYFrameInfo *dst, const RGYFrameInfo *src, const sInputCrop *srcCrop) {
    return copyFrame(dst, src, srcCrop, m_queue[0]);
}
RGY_ERR RGYOpenCLContext::copyFrame(RGYFrameInfo *dst, const RGYFrameInfo *src, const sInputCrop *srcCrop, RGYOpenCLQueue &queue) {
    return copyFrame(dst, src, srcCrop, queue, {}, nullptr);
}
RGY_ERR RGYOpenCLContext::copyFrame(RGYFrameInfo *dst, const RGYFrameInfo *src, const sInputCrop *srcCrop, RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    return copyFrame(dst, src, srcCrop, queue, {}, event);
}

RGY_ERR RGYOpenCLContext::copyFrame(RGYFrameInfo *dst, const RGYFrameInfo *src, const sInputCrop *srcCrop, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event, RGYFrameCopyMode copyMode) {
    if (dst->csp != src->csp) {
        CL_LOG(RGY_LOG_ERROR, _T("in/out csp should be same in copyFrame.\n"));
        return RGY_ERR_INVALID_CALL;
    }

    RGY_ERR err = RGY_ERR_NONE;
    for (int i = 0; i < RGY_CSP_PLANES[dst->csp]; i++) {
        auto planeDst = getPlane(dst, (RGY_PLANE)i);
        auto planeSrc = getPlane(src, (RGY_PLANE)i);
        sInputCrop planeCrop = initCrop();
        if (srcCrop != nullptr) {
            planeCrop = getPlane(srcCrop, src->csp, (RGY_PLANE)i);
        }
        //nv12/p010のimage typeの色差成分の場合、専用のコードが必要
        if ((RGY_PLANE)i == RGY_PLANE_C
            && (planeSrc.csp == RGY_CSP_NV12 || planeSrc.csp == RGY_CSP_P010)
            && (planeSrc.mem_type == RGY_MEM_TYPE_GPU_IMAGE || planeSrc.mem_type == RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED
                || planeDst.mem_type == RGY_MEM_TYPE_GPU_IMAGE || planeDst.mem_type == RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED)) {

            auto copyProgram = getCspCopyProgram(planeDst, planeSrc);
            if (!copyProgram) {
                CL_LOG(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_CL(copyNV12)\n"));
                return RGY_ERR_OPENCL_CRUSH;
            }
            RGYWorkSize local(32, 8);
            RGYWorkSize global(planeDst.width >> 1, planeDst.height);
            err = copyProgram->kernel("kernel_copy_plane_nv12").config(queue, local, global, wait_events, event).launch(
                (cl_mem)planeDst.ptr[0], planeDst.pitch[0], (cl_mem)planeSrc.ptr[0], planeSrc.pitch[0], planeSrc.width >> 1, planeSrc.height,
                planeCrop.e.left, planeCrop.e.up);
            if (err != RGY_ERR_NONE) {
                CL_LOG(RGY_LOG_ERROR, _T("error at kernel_copy_plane (convertCspFromNV12(C)(%s -> %s)): %s.\n"),
                    RGY_CSP_NAMES[planeSrc.csp], RGY_CSP_NAMES[planeDst.csp], get_err_mes(err));
                return err;
            }
        } else {
            err = copyPlane(&planeDst, &planeSrc, &planeCrop, queue,
                (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>(),
                (i + 1 == RGY_CSP_PLANES[dst->csp]) ? event : nullptr,
                copyMode);
            if (err != RGY_ERR_NONE) {
                CL_LOG(RGY_LOG_ERROR, _T("Failed to copy frame(%d): %s\n"), i, cl_errmes(err));
                return err_cl_to_rgy(err);
            }
        }
    }
    dst->picstruct = src->picstruct;
    dst->duration = src->duration;
    dst->timestamp = src->timestamp;
    dst->flags = src->flags;
    dst->inputFrameId = src->inputFrameId;
    return err;
}

RGY_ERR RGYOpenCLContext::setPlane(int value, RGYFrameInfo *dst) {
    return setPlane(value, dst, nullptr);
}
RGY_ERR RGYOpenCLContext::setPlane(int value, RGYFrameInfo *dst, const sInputCrop *dstOffset) {
    return setPlane(value, dst, dstOffset, m_queue[0]);
}
RGY_ERR RGYOpenCLContext::setPlane(int value, RGYFrameInfo *dst, const sInputCrop *dstOffset, RGYOpenCLQueue &queue) {
    return setPlane(value, dst, dstOffset, queue, {}, nullptr);
}
RGY_ERR RGYOpenCLContext::setPlane(int value, RGYFrameInfo *dst, const sInputCrop *dstOffset, RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    return setPlane(value, dst, dstOffset, queue, {}, event);
}
RGY_ERR RGYOpenCLContext::setPlane(int value, RGYFrameInfo *planeDst, const sInputCrop *dstOffset, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    sInputCrop planeCropNone = { 0 };
    if (dstOffset == nullptr) {
        dstOffset = &planeCropNone;
    }
    if (planeDst->mem_type == RGY_MEM_TYPE_CPU) {
        if (RGY_CSP_BIT_DEPTH[planeDst->csp] > 8) {
            for (int y = dstOffset->e.up; y < planeDst->height - dstOffset->e.bottom; y++) {
                uint16_t *ptr = (uint16_t *)(planeDst->ptr[0] + y * planeDst->pitch[0]);
                ptr += dstOffset->e.left;
                const int length = planeDst->height - dstOffset->e.right - dstOffset->e.left;
                for (int x = 0; x < length; x++, ptr++) {
                    *ptr = (uint16_t)value;
                }
            }
        } else {
            for (int y = dstOffset->e.up; y < planeDst->height - dstOffset->e.bottom; y++) {
                uint8_t *ptr = (uint8_t *)(planeDst->ptr[0] + y * planeDst->pitch[0]);
                ptr += dstOffset->e.left;
                const int length = planeDst->height - dstOffset->e.right - dstOffset->e.left;
                for (int x = 0; x < length; x++, ptr++) {
                    *ptr = (uint8_t)value;
                }
            }
        }
        return RGY_ERR_NONE;
    }
    //set関数では、dstのみが必要(srcは任意)
    //m_copyに登録済みのもののうち、dstが一致するものがあれば、それを使う
    const auto optDstMemType = strsprintf("-D MEM_TYPE_DST=%d", planeDst->mem_type);
    const auto optDstBitdepth = strsprintf("-D out_bit_depth=%d", RGY_CSP_BIT_DEPTH[planeDst->csp]);
    RGYOpenCLProgram *setProgram = nullptr;
    for (auto& [opt, progam] : m_copy) {
        if (opt.find(optDstMemType) != std::string::npos && opt.find(optDstBitdepth) != std::string::npos) {
            setProgram = progam.get();
            break;
        }
    }
    if (!setProgram) {
        setProgram = getCspCopyProgram(*planeDst, *planeDst);
        if (!setProgram) {
            CL_LOG(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_CL(set)\n"));
            return RGY_ERR_OPENCL_CRUSH;
        }
    }
    RGYWorkSize local(32, 8);
    RGYWorkSize global(planeDst->width, planeDst->height);
    auto rgy_err = setProgram->kernel("kernel_set_plane").config(queue, local, global, wait_events, event).launch(
        (cl_mem)planeDst->ptr[0], planeDst->pitch[0], planeDst->width, planeDst->height,
        dstOffset->e.left, dstOffset->e.up,
        value);
    return rgy_err;
}
RGY_ERR RGYOpenCLContext::setFrame(int value, RGYFrameInfo *dst) {
    return setFrame(value, dst, nullptr);
}
RGY_ERR RGYOpenCLContext::setFrame(int value, RGYFrameInfo *dst, const sInputCrop *dstOffset) {
    return setFrame(value, dst, dstOffset, m_queue[0]);
}
RGY_ERR RGYOpenCLContext::setFrame(int value, RGYFrameInfo *dst, const sInputCrop *dstOffset, RGYOpenCLQueue &queue) {
    return setFrame(value, dst, dstOffset, queue, {}, nullptr);
}
RGY_ERR RGYOpenCLContext::setFrame(int value, RGYFrameInfo *dst, const sInputCrop *dstOffset, RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    return setFrame(value, dst, dstOffset, queue, {}, event);
}
RGY_ERR RGYOpenCLContext::setFrame(int value, RGYFrameInfo *dst, const sInputCrop *dstOffset, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR err = RGY_ERR_NONE;
    for (int i = 0; i < RGY_CSP_PLANES[dst->csp]; i++) {
        auto planeDst = getPlane(dst, (RGY_PLANE)i);
        sInputCrop planeCrop = { 0 };
        if (dstOffset != nullptr) {
            planeCrop = getPlane(dstOffset, dst->csp, (RGY_PLANE)i);
        }
        err = setPlane(value, &planeDst, &planeCrop, queue,
            (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>(),
            (i + 1 == RGY_CSP_PLANES[dst->csp]) ? event : nullptr);
        if (err != RGY_ERR_NONE) {
            CL_LOG(RGY_LOG_ERROR, _T("Failed to set frame(%d): %s\n"), i, cl_errmes(err));
            return err;
        }
    }
    return err;
}

RGY_ERR RGYOpenCLContext::setBuf(const void *pattern, size_t pattern_size, size_t fill_size_byte, RGYCLBuf *buf) {
    return setBuf(pattern, pattern_size, fill_size_byte, buf, m_queue[0]);
}
RGY_ERR RGYOpenCLContext::setBuf(const void *pattern, size_t pattern_size, size_t fill_size_byte, RGYCLBuf *buf, RGYOpenCLQueue &queue) {
    return setBuf(pattern, pattern_size, fill_size_byte, buf, queue, nullptr);
}
RGY_ERR RGYOpenCLContext::setBuf(const void *pattern, size_t pattern_size, size_t fill_size_byte, RGYCLBuf *buf, RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    return setBuf(pattern, pattern_size, fill_size_byte, buf, queue, {}, event);
}
RGY_ERR RGYOpenCLContext::setBuf(const void *pattern, size_t pattern_size, size_t fill_size_byte, RGYCLBuf *buf, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (fill_size_byte % pattern_size != 0) {
        CL_LOG(RGY_LOG_ERROR, _T("fill_size_byte  %z cannot be divided by pattern_size %z\n"), fill_size_byte, pattern_size);
        return RGY_ERR_INVALID_CALL;
    }
    if (fill_size_byte > buf->size()) {
        CL_LOG(RGY_LOG_ERROR, _T("fill_size_byte %z bigger than buffer size %z.\n"), fill_size_byte, buf->size());
        return RGY_ERR_INVALID_CALL;
    }
    const std::vector<cl_event> v_wait_list = toVec(wait_events);
    const int wait_count = (int)v_wait_list.size();
    const cl_event *wait_list = (wait_count > 0) ? v_wait_list.data() : nullptr;
    cl_event *event_ptr = (event) ? event->reset_ptr() : nullptr;
    auto err = err_cl_to_rgy(clEnqueueFillBuffer(queue.get(), buf->mem(), pattern, pattern_size, 0, fill_size_byte, wait_count, wait_list, event_ptr));
    if (err != RGY_ERR_NONE) {
        CL_LOG(RGY_LOG_ERROR, _T("Failed to set buf size: %u: %s\n"), buf->size(), cl_errmes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

std::unique_ptr<RGYOpenCLProgram> RGYOpenCLContext::buildProgram(const std::string datacopy, const std::string options) {
    auto datalen = datacopy.length();
    if (datacopy.size() == 0) {
        return nullptr;
    }
    const char *data = datacopy.data();
    {
        const uint8_t *ptr = (const uint8_t *)data;
        if (ptr[0] == 0xEF && ptr[1] == 0xBB && ptr[2] == 0xBF) { //skip UTF-8 BOM
            data += 3;
            datalen -= 3;
        }
    }
    CL_LOG(RGY_LOG_DEBUG, _T("building OpenCL source: size %u.\n"), datalen);

    bool buildCrush = false;
    cl_int err = CL_SUCCESS;
    cl_program program = nullptr;
    try {
        program = clCreateProgramWithSource(m_context.get(), 1, &data, &datalen, &err);
        if (err != CL_SUCCESS) {
            CL_LOG(RGY_LOG_ERROR, _T("Error (clCreateProgramWithSource): %s\n"), cl_errmes(err));
        }
    } catch (...) {
        CL_LOG(RGY_LOG_ERROR, _T("Error (clCreateProgramWithSource): Crush!\n"));
        return nullptr;
    }

    try {
        err = clBuildProgram(program, (cl_uint)m_platform->devs().size(), m_platform->devs().data(), options.c_str(), NULL, NULL);
    } catch (...) {
        err = CL_BUILD_PROGRAM_FAILURE;
        buildCrush = true;
    }
    if (err != CL_SUCCESS || m_log->getLogLevel(RGY_LOGT_VPP_BUILD) <= RGY_LOG_DEBUG) {
        const auto loglevel = (err != CL_SUCCESS) ? RGY_LOG_ERROR : RGY_LOG_DEBUG;

        const auto sep = _T("--------------------------------------------------------------------------\n");
        CL_LOG(loglevel, _T("%sbuilding OpenCL source: size %u.\n"), sep, datalen);
        CL_LOG(loglevel, _T("options: %s\nsource\n"), char_to_tstring(options).c_str());
        m_log->write_log(RGY_LOG_DEBUG, RGY_LOGT_VPP_BUILD, (char_to_tstring(str_replace(std::string(data, datalen), "\r\n", "\n"), CP_UTF8) + _T("\n") + sep).c_str(), true);

        for (const auto &device : m_platform->devs()) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

            std::vector<char> build_log(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);

            CL_LOG(loglevel, _T("%sbuild log of %s...\n"), sep, char_to_tstring(RGYOpenCLDevice(device).info().name).c_str());
            auto log = char_to_tstring(build_log.data()) + _T("\n") + sep;
            m_log->write_log(loglevel, RGY_LOGT_VPP_BUILD, log.c_str());
        }
        if (err != CL_SUCCESS) {
            CL_LOG(loglevel, _T("Error (clBuildProgram): %s\n"), cl_errmes(err));
            return nullptr;
        }
    }
    CL_LOG(RGY_LOG_DEBUG, _T("clBuildProgram success!\n"));
    return std::make_unique<RGYOpenCLProgram>(program, m_log);
}

std::unique_ptr<RGYOpenCLProgram> RGYOpenCLContext::build(const std::string &source, const char *options) {
    return buildProgram(source, options);
}

std::future<std::unique_ptr<RGYOpenCLProgram>> RGYOpenCLContext::buildAsync(const std::string &source, const char *options) {
    return threadPool()->enqueue([this, src = std::string(source), opt = std::string(options)]() {
        return buildProgram(src, opt);
    });
}

std::unique_ptr<RGYOpenCLProgram> RGYOpenCLContext::buildFile(const tstring filename, const std::string options) {
    std::ifstream inputFile(filename);
    if (inputFile.bad()) {
        CL_LOG(RGY_LOG_ERROR, _T("Failed to open source file \"%s\".\n"), filename.c_str());
        return nullptr;
    }
    CL_LOG(RGY_LOG_DEBUG, _T("Opened file \"%s\""), filename.c_str());
    std::istreambuf_iterator<char> data_begin(inputFile);
    std::istreambuf_iterator<char> data_end;
    std::string source = std::string(data_begin, data_end);
    inputFile.close();
    return buildProgram(source, options);
}

std::future<std::unique_ptr<RGYOpenCLProgram>> RGYOpenCLContext::buildFileAsync(const tstring& filename, const char *options) {
    return threadPool()->enqueue([this, file = tstring(filename), opt = std::string(options)]() {
        return buildFile(file, opt);
    });
}

std::unique_ptr<RGYOpenCLProgram> RGYOpenCLContext::buildResource(const tstring name, const tstring type, const std::string options) {
    void *data = nullptr;
    CL_LOG(RGY_LOG_DEBUG, _T("Load resource type: %s, name: %s\n"), type.c_str(), name.c_str());
    int size = getEmbeddedResource(&data, name.c_str(), type.c_str(), m_hmodule);
    if (data == nullptr || size == 0) {
        CL_LOG(RGY_LOG_ERROR, _T("Failed to load resource [%s] %s\n"), type.c_str(), name.c_str());
        return nullptr;
    }
    CL_LOG(RGY_LOG_DEBUG, _T("Loaded resource type: %s, name: %s, size = %d\n"), type.c_str(), name.c_str(), size);
    return buildProgram(std::string((const char *)data, size), std::string(options));
}

std::future<std::unique_ptr<RGYOpenCLProgram>> RGYOpenCLContext::buildResourceAsync(const TCHAR *name, const TCHAR *type, const char *options) {
    return threadPool()->enqueue([this, resource_name = tstring(name), resource_type = tstring(type), opt = std::string(options)]() {
        return buildResource(resource_name, resource_type, opt);
    });
}

std::unique_ptr<RGYCLBuf> RGYOpenCLContext::createBuffer(size_t size, cl_mem_flags flags, void *host_ptr) {
    cl_int err = CL_SUCCESS;
    cl_mem mem = clCreateBuffer(m_context.get(), flags, size, host_ptr, &err);
    if (err != CL_SUCCESS) {
        CL_LOG(RGY_LOG_ERROR, _T("Failed to allocate memory: %s\n"), cl_errmes(err));
    }
    return std::make_unique<RGYCLBuf>(mem, flags, size);
}

std::unique_ptr<RGYCLBuf> RGYOpenCLContext::copyDataToBuffer(const void *host_ptr, size_t size, cl_mem_flags flags, cl_command_queue queue) {
    auto buffer = createBuffer(size, flags);
    if (buffer != nullptr) {
        cl_int err = clEnqueueWriteBuffer((queue != RGYDefaultQueue) ? queue : m_queue[0].get(), buffer->mem(), true, 0, size, host_ptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            CL_LOG(RGY_LOG_ERROR, _T("Failed to copy data to buffer: %s\n"), cl_errmes(err));
        }
    }
    return buffer;
}

RGY_ERR RGYOpenCLContext::createImageFromPlane(cl_mem &image, const cl_mem buffer, const int bit_depth, const int channel_order, const bool normalized, const int pitch, const int width, const int height, const cl_mem_flags flags) {
    cl_image_format format;
    format.image_channel_order = channel_order; //チャンネル数
    format.image_channel_data_type =  //データ型
        (normalized) ? ((bit_depth > 8) ? CL_UNORM_INT16 : CL_UNORM_INT8)
                     : ((bit_depth > 8) ? CL_UNSIGNED_INT16 : CL_UNSIGNED_INT8);

    cl_image_desc img_desc;
    img_desc.image_type = CL_MEM_OBJECT_IMAGE2D; //2D
    img_desc.image_width = width;   //サイズ
    img_desc.image_height = height; //サイズ
    img_desc.image_depth = 0;
    img_desc.image_array_size = 0;
    img_desc.image_row_pitch = pitch;
    img_desc.image_slice_pitch = 0;
    img_desc.num_mip_levels = 0;
    img_desc.num_samples = 0;
    img_desc.buffer = 0;
    img_desc.mem_object = buffer;

    cl_int err = CL_SUCCESS;
    image = (cl_mem)clCreateImage(m_context.get(),
        flags,
        &format, &img_desc,
        nullptr,
        &err);
    return err_cl_to_rgy(err);
}

RGY_ERR RGYOpenCLContext::createImageFromFrame(RGYFrameInfo& frameImage, const RGYFrameInfo& frame, const bool normalized, const bool cl_image2d_from_buffer_support, const cl_mem_flags flags) {
    frameImage = frame;
    frameImage.mem_type = (normalized) ? RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED : RGY_MEM_TYPE_GPU_IMAGE;

    for (int i = 0; i < RGY_CSP_PLANES[frame.csp]; i++) {
        const auto plane = getPlane(&frame, (RGY_PLANE)i);
        cl_mem image;
        auto err = createImageFromPlane(image,
            (cl_image2d_from_buffer_support) ? (cl_mem)plane.ptr[0] : nullptr,
            RGY_CSP_BIT_DEPTH[frame.csp], CL_R, normalized,
            (cl_image2d_from_buffer_support) ? plane.pitch[0] : 0,
            plane.width, plane.height,
            (cl_image2d_from_buffer_support) ? flags : CL_MEM_READ_WRITE);
        if (err != CL_SUCCESS) {
            CL_LOG(RGY_LOG_ERROR, _T("Failed to create image for plane %d%s: %s\n"),
                i,
                cl_image2d_from_buffer_support ? _T(" from buffer memory") : _T(""),
                cl_errmes(err));
            for (int j = i - 1; j >= 0; j--) {
                if (frameImage.ptr[j] != nullptr) {
                    clReleaseMemObject((cl_mem)frameImage.ptr[j]);
                    frameImage.ptr[j] = nullptr;
                }
            }
            return err_cl_to_rgy(err);
        }
        frameImage.ptr[i] = (uint8_t *)image;
    }
    return RGY_ERR_NONE;
}

std::unique_ptr<RGYCLFrame, RGYCLImageFromBufferDeleter> RGYOpenCLContext::createImageFromFrameBuffer(const RGYFrameInfo &frame, const bool normalized, const cl_mem_flags flags, RGYCLFramePool *imgpool) {
    const auto device = RGYOpenCLDevice(queue().devid());
    // cl_khr_image2d_from_buffer は OpenCL 3.0 / 1.2 ではオプション、2.0 では必須
    // cl_khr_image2d_from_buffer のサポートがない場合は新しいimageをつくり、コピーする必要がある
    const bool cl_not_version_2_0 = device.checkVersion(3, 0) || !device.checkVersion(2, 0);
    const bool cl_image2d_from_buffer_support = (cl_not_version_2_0) ? device.checkExtension("cl_khr_image2d_from_buffer") : true;

    if (cl_image2d_from_buffer_support) {
        RGYFrameInfo frameImage;
        auto err = createImageFromFrame(frameImage, frame, normalized, cl_image2d_from_buffer_support, flags);
        if (err != RGY_ERR_NONE) {
            return nullptr;
        }
        return std::unique_ptr<RGYCLFrame, RGYCLImageFromBufferDeleter>(new RGYCLFrame(frameImage, flags), RGYCLImageFromBufferDeleter(nullptr));
    }

    std::unique_ptr<RGYCLFrame, RGYCLImageFromBufferDeleter> imgFrame;
    if (imgpool) {
        imgFrame = imgpool->get(frame, normalized, flags);
    }
    if (!imgFrame) {
        RGYFrameInfo frameImage;
        auto err = createImageFromFrame(frameImage, frame, normalized, cl_image2d_from_buffer_support, flags);
        if (err != RGY_ERR_NONE) {
            return nullptr;
        }
        imgFrame = std::unique_ptr<RGYCLFrame, RGYCLImageFromBufferDeleter>(new RGYCLFrame(frameImage, flags), RGYCLImageFromBufferDeleter(imgpool));
    }
    // メモリコピーが必要
    auto err = copyFrame(&imgFrame->frame, &frame);
    if (err != RGY_ERR_NONE) {
        imgFrame.reset();
        return nullptr;
    }
    copyFrameProp(&imgFrame->frame, &frame);
    return imgFrame;
}

std::unique_ptr<RGYCLFrame> RGYOpenCLContext::createFrameBuffer(const int width, const int height, const RGY_CSP csp, const int bitdepth, const cl_mem_flags flags) {
    RGYFrameInfo info(width, height, csp, bitdepth);
    return createFrameBuffer(info, flags);
}

std::unique_ptr<RGYCLFrame> RGYOpenCLContext::createFrameBuffer(const RGYFrameInfo& frame, cl_mem_flags flags) {
    cl_int err = CL_SUCCESS;
    int pixsize = (RGY_CSP_BIT_DEPTH[frame.csp] + 7) / 8;
    switch (frame.csp) {
    case RGY_CSP_BGR24R:
    case RGY_CSP_RGB24:
    case RGY_CSP_BGR24:
    case RGY_CSP_YC48:
        pixsize *= 3;
        break;
    case RGY_CSP_BGR32R:
    case RGY_CSP_RGB32:
    case RGY_CSP_BGR32:
    case RGY_CSP_ARGB32:
    case RGY_CSP_ABGR32:
    case RGY_CSP_RBGA32:
        pixsize *= 4;
        break;
    case RGY_CSP_RBGA64:
    case RGY_CSP_RGBA_FP16_P:
        pixsize *= 8;
        break;
    case RGY_CSP_VUYA:
    case RGY_CSP_VUYA_16:
        pixsize *= 4;
        break;
    case RGY_CSP_YUY2:
    case RGY_CSP_Y210:
    case RGY_CSP_Y216:
    case RGY_CSP_Y410:
        pixsize *= 2;
        break;
    case RGY_CSP_Y416:
        pixsize *= 4;
        break;
    default:
        break;
    }

    // convert系の関数でalignmentを前提としている箇所があるので、最低でも64にするようにする
    // またQSVEncのfixed-funcに渡すとき、256でないと異常が生じる場合がある
    const int image_pitch_alignment = std::max(m_platform->dev(0).info().image_pitch_alignment, 256);

    RGYFrameInfo clframe = frame;
    clframe.mem_type = RGY_MEM_TYPE_GPU;
    for (int i = 0; i < _countof(clframe.ptr); i++) {
        clframe.ptr[i] = nullptr;
        clframe.pitch[i] = 0;
    }
    for (int i = 0; i < RGY_CSP_PLANES[frame.csp]; i++) {
        const auto plane = getPlane(&clframe, (RGY_PLANE)i);
        const int widthByte = plane.width * pixsize;
        const int memPitch = ALIGN(widthByte, image_pitch_alignment);
        const int size = memPitch * plane.height;
        cl_mem mem = clCreateBuffer(m_context.get(), flags, size, nullptr, &err);
        if (err != CL_SUCCESS) {
            CL_LOG(RGY_LOG_ERROR, _T("Failed to allocate memory: %s\n"), cl_errmes(err));
            for (int j = i-1; j >= 0; j--) {
                if (clframe.ptr[j] != nullptr) {
                    clReleaseMemObject((cl_mem)clframe.ptr[j]);
                    clframe.ptr[j] = nullptr;
                }
            }
            return std::unique_ptr<RGYCLFrame>();
        }
        clframe.pitch[i] = memPitch;
        clframe.ptr[i] = (uint8_t *)mem;
    }
    return std::make_unique<RGYCLFrame>(clframe, flags);
}

std::unique_ptr<RGYCLFrameInterop> RGYOpenCLContext::createFrameFromD3D9Surface(void *surf, HANDLE shared_handle, const RGYFrameInfo &frame, RGYOpenCLQueue& queue, cl_mem_flags flags) {
#if !ENABLE_RGY_OPENCL_D3D9
    CL_LOG(RGY_LOG_ERROR, _T("OpenCL d3d9 interop not supported in this build.\n"));
    return std::unique_ptr<RGYCLFrameInterop>();
#else
    if (m_platform->d3d9dev() == nullptr) {
        CL_LOG(RGY_LOG_ERROR, _T("OpenCL platform not associated with d3d9 device.\n"));
        return std::unique_ptr<RGYCLFrameInterop>();
    }
    RGYFrameInfo clframe = frame;
    clframe.mem_type = RGY_MEM_TYPE_GPU_IMAGE;
    for (int i = 0; i < _countof(clframe.ptr); i++) {
        clframe.ptr[i] = nullptr;
        clframe.pitch[i] = 0;
    }
    cl_dx9_surface_info_khr surfInfo = { (IDirect3DSurface9 *)surf, shared_handle };
    for (int i = 0; i < RGY_CSP_PLANES[frame.csp]; i++) {
        cl_int err = 0;
        clframe.ptr[i] = (uint8_t *)clCreateFromDX9MediaSurfaceKHR(m_context.get(), flags, CL_ADAPTER_D3D9EX_KHR, &surfInfo, i, &err);
        if (err != CL_SUCCESS) {
            CL_LOG(RGY_LOG_ERROR, _T("Failed to create image from DX9 memory: %s\n"), cl_errmes(err));
            for (int j = i - 1; j >= 0; j--) {
                if (clframe.ptr[j] != nullptr) {
                    clReleaseMemObject((cl_mem)clframe.ptr[j]);
                    clframe.ptr[j] = nullptr;
                }
            }
            return std::unique_ptr<RGYCLFrameInterop>();
        }
    }
    auto meminfo = getRGYCLMemObjectInfo((cl_mem)clframe.ptr[0]);
    clframe.mem_type = (meminfo.isImageNormalizedType()) ? RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED : RGY_MEM_TYPE_GPU_IMAGE;
    return std::unique_ptr<RGYCLFrameInterop>(new RGYCLFrameInterop(clframe, flags, RGY_INTEROP_DX9, queue, m_log));
#endif
}

std::unique_ptr<RGYCLFrameInterop> RGYOpenCLContext::createFrameFromD3D11Surface(void *surf, const RGYFrameInfo &frame, RGYOpenCLQueue& queue, cl_mem_flags flags) {
#if !ENABLE_RGY_OPENCL_D3D11
    CL_LOG(RGY_LOG_ERROR, _T("OpenCL d3d11 interop not supported in this build.\n"));
    return std::unique_ptr<RGYCLFrameInterop>();
#else
    if (m_platform->d3d11dev() == nullptr) {
        CL_LOG(RGY_LOG_ERROR, _T("OpenCL platform not associated with d3d11 device.\n"));
        return std::unique_ptr<RGYCLFrameInterop>();
    }
    RGYFrameInfo clframe = frame;
    for (int i = 0; i < _countof(clframe.ptr); i++) {
        clframe.ptr[i] = nullptr;
        clframe.pitch[i] = 0;
    }
    for (int i = 0; i < RGY_CSP_PLANES[frame.csp]; i++) {
        cl_int err = CL_SUCCESS;
        try {
            clframe.ptr[i] = (uint8_t *)clCreateFromD3D11Texture2DKHR(m_context.get(), flags, (ID3D11Texture2D *)surf, i, &err);
        } catch (...) {
            CL_LOG(RGY_LOG_ERROR, _T("Failed to create image from DX11 texture 2D: crushed when calling clCreateFromD3D11Texture2DKHR: 0x%p[%d].\n"), cl_errmes(err), surf, i);
            err = CL_INVALID_MEM_OBJECT;
        }
        if (err != CL_SUCCESS) {
            CL_LOG(RGY_LOG_ERROR, _T("Failed to create image from DX11 texture 2D: %s\n"), cl_errmes(err));
            for (int j = i - 1; j >= 0; j--) {
                if (clframe.ptr[j] != nullptr) {
                    clReleaseMemObject((cl_mem)clframe.ptr[j]);
                    clframe.ptr[j] = nullptr;
                }
            }
            return std::unique_ptr<RGYCLFrameInterop>();
        }
    }
    auto meminfo = getRGYCLMemObjectInfo((cl_mem)clframe.ptr[0]);
    clframe.mem_type = (meminfo.isImageNormalizedType()) ? RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED : RGY_MEM_TYPE_GPU_IMAGE;
    return std::make_unique<RGYCLFrameInterop>(clframe, flags, RGY_INTEROP_DX11, queue, m_log);
#endif
}

std::unique_ptr<RGYCLFrameInterop> RGYOpenCLContext::createFrameFromD3D11SurfacePlanar(const RGYFrameInfo &frame, RGYOpenCLQueue& queue, cl_mem_flags flags) {
#if !ENABLE_RGY_OPENCL_D3D11
    CL_LOG(RGY_LOG_ERROR, _T("OpenCL d3d11 interop not supported in this build.\n"));
    return std::unique_ptr<RGYCLFrameInterop>();
#else
    if (m_platform->d3d11dev() == nullptr) {
        CL_LOG(RGY_LOG_ERROR, _T("OpenCL platform not associated with d3d11 device.\n"));
        return std::unique_ptr<RGYCLFrameInterop>();
    }
    RGYFrameInfo clframe = frame;
    for (int i = 0; i < _countof(clframe.ptr); i++) {
        clframe.ptr[i] = nullptr;
        clframe.pitch[i] = 0;
    }
    for (int i = 0; i < RGY_CSP_PLANES[frame.csp]; i++) {
        cl_int err = CL_SUCCESS;
        clframe.ptr[i] = (uint8_t *)clCreateFromD3D11Texture2DKHR(m_context.get(), flags, (ID3D11Texture2D *)frame.ptr[i], 0, &err);
        if (err != CL_SUCCESS) {
            CL_LOG(RGY_LOG_ERROR, _T("Failed to create image from DX11 texture 2D (planar %d): %s\n"), i, cl_errmes(err));
            for (int j = i - 1; j >= 0; j--) {
                if (clframe.ptr[j] != nullptr) {
                    clReleaseMemObject((cl_mem)clframe.ptr[j]);
                    clframe.ptr[j] = nullptr;
                }
            }
            return std::unique_ptr<RGYCLFrameInterop>();
        }
    }
    auto meminfo = getRGYCLMemObjectInfo((cl_mem)clframe.ptr[0]);
    clframe.mem_type = (meminfo.isImageNormalizedType()) ? RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED : RGY_MEM_TYPE_GPU_IMAGE;
    return std::make_unique<RGYCLFrameInterop>(clframe, flags, RGY_INTEROP_DX11, queue, m_log);
#endif
}

std::unique_ptr<RGYCLFrameInterop> RGYOpenCLContext::createFrameFromVASurface(void *surf, const RGYFrameInfo &frame, RGYOpenCLQueue& queue, cl_mem_flags flags) {
#if !ENABLE_RGY_OPENCL_VA
    UNREFERENCED_PARAMETER(surf);
    UNREFERENCED_PARAMETER(frame);
    UNREFERENCED_PARAMETER(queue);
    UNREFERENCED_PARAMETER(flags);
    CL_LOG(RGY_LOG_ERROR, _T("OpenCL VA interop not supported in this build.\n"));
    return std::unique_ptr<RGYCLFrameInterop>();
#else
    if (m_platform->vadev() == nullptr) {
        CL_LOG(RGY_LOG_ERROR, _T("OpenCL platform not associated with va device.\n"));
        return std::unique_ptr<RGYCLFrameInterop>();
    }
    RGYFrameInfo clframe = frame;
    for (int i = 0; i < _countof(clframe.ptr); i++) {
        clframe.ptr[i] = nullptr;
        clframe.pitch[i] = 0;
    }
    for (int i = 0; i < RGY_CSP_PLANES[frame.csp]; i++) {
        cl_int err = CL_SUCCESS;
        clframe.ptr[i] = (uint8_t *)clCreateFromVA_APIMediaSurfaceINTEL(m_context.get(), flags, (VASurfaceID *)surf, i, &err);
        if (err != CL_SUCCESS) {
            CL_LOG(RGY_LOG_ERROR, _T("Failed to create image from va surface: %s\n"), cl_errmes(err));
            for (int j = i - 1; j >= 0; j--) {
                if (clframe.ptr[j] != nullptr) {
                    clReleaseMemObject((cl_mem)clframe.ptr[j]);
                    clframe.ptr[j] = nullptr;
                }
            }
            return std::unique_ptr<RGYCLFrameInterop>();
        }
    }
    auto meminfo = getRGYCLMemObjectInfo((cl_mem)clframe.ptr[0]);
    clframe.mem_type = (meminfo.isImageNormalizedType()) ? RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED : RGY_MEM_TYPE_GPU_IMAGE;
    return std::make_unique<RGYCLFrameInterop>(clframe, flags, RGY_INTEROP_VA, queue, m_log);
#endif
}

RGYOpenCL::RGYOpenCL() : m_log(std::make_shared<RGYLog>(nullptr, RGY_LOG_ERROR)) {
    if (initOpenCLGlobal()) {
        CL_LOG(RGY_LOG_ERROR, _T("Failed to load OpenCL.\n"));
    } else {
        CL_LOG(RGY_LOG_DEBUG, _T("loaded OpenCL.\n"));
    }
}

RGYOpenCL::RGYOpenCL(shared_ptr<RGYLog> pLog) : m_log(pLog) {
    if (initOpenCLGlobal()) {
        CL_LOG(RGY_LOG_ERROR, _T("Failed to load OpenCL.\n"));
    } else {
        CL_LOG(RGY_LOG_DEBUG, _T("loaded OpenCL.\n"));
    }
}

RGYOpenCL::~RGYOpenCL() {

}

std::vector<shared_ptr<RGYOpenCLPlatform>> RGYOpenCL::getPlatforms(const char *vendor) {
    std::vector<shared_ptr<RGYOpenCLPlatform>> platform_list;
    if (RGYOpenCL::openCLCrush) {
        return platform_list;
    }

    cl_uint platform_count = 0;
    cl_int ret = CL_SUCCESS;

    //OpenCLのドライバは場合によってはクラッシュする可能性がある
    try {
        if (CL_SUCCESS != (ret = clGetPlatformIDs(0, NULL, &platform_count))) {
            CL_LOG(RGY_LOG_ERROR, _T("Error (clGetPlatformIDs): %s\n"), cl_errmes(ret));
            return platform_list;
        }
    } catch (...) {
        CL_LOG(RGY_LOG_ERROR, _T("Crush (clGetPlatformIDs)\n"));
        RGYOpenCL::openCLCrush = true; //クラッシュフラグを立てる
        return platform_list;
    }
    CL_LOG(RGY_LOG_DEBUG, _T("OpenCL platform count: %d\n"), platform_count);

    if (platform_count > 0) {
        std::vector<cl_platform_id> platforms(platform_count, 0);
        try {
            if (CL_SUCCESS != (ret = clGetPlatformIDs(platform_count, platforms.data(), &platform_count))) {
                CL_LOG(RGY_LOG_ERROR, _T("Error (clGetPlatformIDs): %s\n"), cl_errmes(ret));
                return platform_list;
            }
        } catch (...) {
            CL_LOG(RGY_LOG_ERROR, _T("Crush (clGetPlatformIDs)\n"));
            RGYOpenCL::openCLCrush = true; //クラッシュフラグを立てる
            return platform_list;
        }

        for (int i = 0; i < (int)platform_count; i++) {
            auto platform = std::make_shared<RGYOpenCLPlatform>(platforms[i], m_log);
            if (m_log->getLogLevel(RGY_LOGT_OPENCL) <= RGY_LOG_DEBUG) {
                CL_LOG(RGY_LOG_DEBUG, _T("OpenCL platform #%d[%p]: %s\n"), i, platforms[i], platform->info().print().c_str());
            }
            if (vendor == nullptr || strlen(vendor) == 0 || platform->isVendor(vendor)) {
                CL_LOG(RGY_LOG_DEBUG, _T("Add platform #%d[%p] to list.\n"), i, platforms[i]);
                platform_list.push_back(std::move(platform));
            }
        }
    }
    CL_LOG(RGY_LOG_DEBUG, _T("Created OpenCL platform list: %d\n"), (int)platform_list.size());
    return platform_list;
}

#endif

tstring getOpenCLInfo(const cl_device_type device_type) {
    auto log = std::make_shared<RGYLog>(nullptr, RGY_LOG_ERROR);
    RGYOpenCL cl(log);
    auto platforms = cl.getPlatforms(nullptr);
    if (platforms.size() == 0) {
        tstring str = _T("No OpenCL Platform found on this system.\n\n");
        str += checkOpenCLDLL();
        return str;
    }

    tstring str;
    for (int ip = 0; ip < (int)platforms.size(); ip++) {
        str += strsprintf(_T("OpenCL platform #%d [0x%p]\n%s\n"), ip, platforms[ip].get(), platforms[ip]->info().print().c_str());
        auto err = platforms[ip]->createDeviceList(device_type);
        if (err != RGY_ERR_NONE) {
            str += strsprintf(_T("    device: %s\n"), get_err_mes(err));
        } else {
            auto devices = platforms[ip]->devs();
            for (int idev = 0; idev < (int)devices.size(); idev++) {
                tstring devInfo = strsprintf(_T("device #%d [0x%p]\n%s\n"), idev, devices[idev], RGYOpenCLDevice(devices[idev]).infostr(true).c_str());
                str += add_indent(devInfo, 4);
                str += _T("\n");
            }
        }
        str += _T("\n");
    }
    return str;
}
