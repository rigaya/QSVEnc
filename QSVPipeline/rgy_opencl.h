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

#ifndef __RGY_OPENCL_H__
#define __RGY_OPENCL_H__

#include "rgy_version.h"

#if ENABLE_OPENCL

#if ENCODER_QSV
#define ENABLE_RGY_OPENCL_D3D9  D3D_SURFACES_SUPPORT
#define ENABLE_RGY_OPENCL_D3D11 (D3D_SURFACES_SUPPORT && MFX_D3D11_SUPPORT)
#define ENABLE_RGY_OPENCL_VA    LIBVA_SUPPORT
#elif ENCODER_VCEENC
#define ENABLE_RGY_OPENCL_D3D9  ENABLE_D3D9
#define ENABLE_RGY_OPENCL_D3D11 ENABLE_D3D11
#define ENABLE_RGY_OPENCL_VA    0
#elif ENCODER_MPP
#define ENABLE_RGY_OPENCL_D3D9  0
#define ENABLE_RGY_OPENCL_D3D11 0
#define ENABLE_RGY_OPENCL_VA    0
#else
#define ENABLE_RGY_OPENCL_D3D9  1
#define ENABLE_RGY_OPENCL_D3D11 1
#define ENABLE_RGY_OPENCL_VA    0
#endif

#include "rgy_osdep.h"
#define CL_TARGET_OPENCL_VERSION 210
#include <CL/opencl.h>
#if ENABLE_RGY_OPENCL_D3D9
#include <CL/cl_dx9_media_sharing.h>
#endif //#if ENABLE_RGY_OPENCL_D3D9
#if ENABLE_RGY_OPENCL_D3D11
#include <CL/cl_d3d11.h>
#endif //#if ENABLE_RGY_OPENCL_D3D11
#if ENABLE_RGY_OPENCL_VA
#include <va/va.h>
#endif //ENABLE_RGY_OPENCL_VA
#include <unordered_map>
#include <vector>
#include <array>
#include <deque>
#include <memory>
#include <future>
#include <typeindex>
#include "rgy_err.h"
#include "rgy_def.h"
#include "rgy_log.h"
#include "rgy_util.h"
#include "rgy_frame.h"
#include "convert_csp.h"
#include "rgy_frame_info.h"
#include "rgy_thread_pool.h"

#ifndef CL_EXTERN
#define CL_EXTERN extern
#endif

#define RGYDefaultQueue 0

#ifndef cl_device_feature_capabilities_intel
typedef cl_bitfield cl_device_feature_capabilities_intel;
#endif
#ifndef cl_mem_properties
typedef cl_bitfield cl_mem_properties;
#endif

#if !defined(cl_khr_external_semaphore)
typedef void* cl_semaphore_khr;
typedef cl_ulong cl_semaphore_properties_khr;
typedef cl_uint cl_semaphore_info_khr;
typedef cl_uint cl_semaphore_type_khr;
typedef cl_ulong cl_semaphore_payload_khr;
#endif

#ifndef CL_UUID_SIZE_KHR
#define CL_UUID_SIZE_KHR 16
#endif
#ifndef CL_DEVICE_UUID_KHR
#define CL_DEVICE_UUID_KHR          0x106A
#endif
#ifndef CL_DEVICE_LUID_KHR
#define CL_DEVICE_LUID_KHR          0x106B
#endif

#ifndef CL_LUID_SIZE_KHR
#define CL_LUID_SIZE_KHR 8
#endif
#ifndef CL_DEVICE_LUID_VALID_KHR
#define CL_DEVICE_LUID_VALID_KHR    0x106C
#endif
#ifndef CL_DRIVER_LUID_KHR
#define CL_DRIVER_LUID_KHR          0x106D
#endif

#if ENABLE_RGY_OPENCL_D3D9
// ---cl_dx9_media_sharing_intel ---
#define cl_intel_dx9_media_sharing 1

typedef cl_uint cl_dx9_device_source_intel;
typedef cl_uint cl_dx9_device_set_intel;

/* error codes */
#define CL_INVALID_DX9_DEVICE_INTEL                   -1010
#define CL_INVALID_DX9_RESOURCE_INTEL                 -1011
#define CL_DX9_RESOURCE_ALREADY_ACQUIRED_INTEL        -1012
#define CL_DX9_RESOURCE_NOT_ACQUIRED_INTEL            -1013

/* cl_dx9_device_source_intel */
#define CL_D3D9_DEVICE_INTEL                          0x4022
#define CL_D3D9EX_DEVICE_INTEL                        0x4070
#define CL_DXVA_DEVICE_INTEL                          0x4071

/* cl_dx9_device_set_intel */
#define CL_PREFERRED_DEVICES_FOR_DX9_INTEL            0x4024
#define CL_ALL_DEVICES_FOR_DX9_INTEL                  0x4025

/* cl_context_info */
#define CL_CONTEXT_D3D9_DEVICE_INTEL                  0x4026
#define CL_CONTEXT_D3D9EX_DEVICE_INTEL                0x4072
#define CL_CONTEXT_DXVA_DEVICE_INTEL                  0x4073

/* cl_mem_info */
#define CL_MEM_DX9_RESOURCE_INTEL                     0x4027
#define CL_MEM_DX9_SHARED_HANDLE_INTEL                0x4074

/* cl_image_info */
#define CL_IMAGE_DX9_PLANE_INTEL                      0x4075

/* cl_command_type */
#define CL_COMMAND_ACQUIRE_DX9_OBJECTS_INTEL          0x402A
#define CL_COMMAND_RELEASE_DX9_OBJECTS_INTEL          0x402B
#endif //#if ENABLE_OPENCL_D3D9

#if ENABLE_RGY_OPENCL_VA
// ---cl_intel_va_api_media_sharing  ---
/* error codes */
#define CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL               -1098
#define CL_INVALID_VA_API_MEDIA_SURFACE_INTEL               -1099
#define CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL      -1100
#define CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL          -1101

/* cl_va_api_device_source_intel */
#define CL_VA_API_DISPLAY_INTEL                             0x4094

/* cl_va_api_device_set_intel */
#define CL_PREFERRED_DEVICES_FOR_VA_API_INTEL               0x4095
#define CL_ALL_DEVICES_FOR_VA_API_INTEL                     0x4096

/* cl_context_info */
#define CL_CONTEXT_VA_API_DISPLAY_INTEL                     0x4097

/* cl_mem_info */
#define CL_MEM_VA_API_MEDIA_SURFACE_INTEL                   0x4098

/* cl_image_info */
#define CL_IMAGE_VA_API_PLANE_INTEL                         0x4099

/* cl_command_type */
#define CL_COMMAND_ACQUIRE_VA_API_MEDIA_SURFACES_INTEL      0x409A
#define CL_COMMAND_RELEASE_VA_API_MEDIA_SURFACES_INTEL      0x409B

typedef cl_uint cl_va_api_device_source_intel;
typedef cl_uint cl_va_api_device_set_intel;
// -------------------------------------------------------------
#endif //#ifdef ENABLE_OPENCL_VA

CL_EXTERN void *(CL_API_CALL *f_clGetExtensionFunctionAddressForPlatform)(cl_platform_id  platform, const char *funcname);

CL_EXTERN cl_int (CL_API_CALL* f_clGetPlatformIDs)(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms);
CL_EXTERN cl_int (CL_API_CALL* f_clGetPlatformInfo) (cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
CL_EXTERN cl_int (CL_API_CALL* f_clGetDeviceIDs) (cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices);
CL_EXTERN cl_int (CL_API_CALL* f_clGetDeviceInfo) (cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);

CL_EXTERN cl_context (CL_API_CALL* f_clCreateContext) (const cl_context_properties * properties, cl_uint num_devices, const cl_device_id * devices, void (CL_CALLBACK * pfn_notify)(const char *, const void *, size_t, void *), void * user_data, cl_int * errcode_ret);
CL_EXTERN cl_int (CL_API_CALL* f_clReleaseContext) (cl_context context);
CL_EXTERN cl_command_queue (CL_API_CALL* f_clCreateCommandQueue)(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int * errcode_ret);
CL_EXTERN cl_int (CL_API_CALL* f_clGetCommandQueueInfo)(cl_command_queue command_queue, cl_command_queue_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
CL_EXTERN cl_int (CL_API_CALL* f_clReleaseCommandQueue) (cl_command_queue command_queue);
CL_EXTERN cl_int (CL_API_CALL* f_clGetSupportedImageFormats)(cl_context context, cl_mem_flags flags, cl_mem_object_type image_type, cl_uint num_entries, cl_image_format * image_formats, cl_uint * num_image_formats);

CL_EXTERN cl_program(CL_API_CALL* f_clCreateProgramWithSource) (cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret);
CL_EXTERN cl_int (CL_API_CALL* f_clBuildProgram) (cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options, void (CL_CALLBACK *pfn_notify)(cl_program program, void *user_data), void* user_data);
CL_EXTERN cl_int (CL_API_CALL* f_clGetProgramBuildInfo) (cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
CL_EXTERN cl_int (CL_API_CALL* f_clGetProgramInfo)(cl_program program, cl_program_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
CL_EXTERN cl_int (CL_API_CALL* f_clReleaseProgram) (cl_program program);

CL_EXTERN cl_mem (CL_API_CALL* f_clCreateBuffer) (cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret);
CL_EXTERN cl_mem (CL_API_CALL* f_clCreateImage)(cl_context context, cl_mem_flags flags, const cl_image_format *image_format, const cl_image_desc *image_desc, void *host_ptr, cl_int *errcode_ret);
CL_EXTERN cl_mem (CL_API_CALL* f_clCreateImageWithProperties)(cl_context context, const cl_mem_properties *properties, cl_mem_flags flags, const cl_image_format *image_format, const cl_image_desc *image_desc, void *host_ptr, cl_int *errcode_ret);
CL_EXTERN cl_int (CL_API_CALL* f_clReleaseMemObject) (cl_mem memobj);
CL_EXTERN cl_int (CL_API_CALL* f_clGetMemObjectInfo)(cl_mem memobj, cl_mem_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
CL_EXTERN cl_int (CL_API_CALL* f_clGetImageInfo)(cl_mem memobj, cl_mem_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
CL_EXTERN cl_kernel (CL_API_CALL* f_clCreateKernel) (cl_program program, const char *kernel_name, cl_int *errcode_ret);
CL_EXTERN cl_int (CL_API_CALL* f_clReleaseKernel) (cl_kernel kernel);
CL_EXTERN cl_int (CL_API_CALL* f_clSetKernelArg) (cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value);
CL_EXTERN cl_int (CL_API_CALL* f_clEnqueueNDRangeKernel)(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event * event);
CL_EXTERN cl_int(CL_API_CALL* f_clEnqueueTask) (cl_command_queue command_queue, cl_kernel kernel, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);

CL_EXTERN cl_int(CL_API_CALL* f_clEnqueueReadBuffer) (cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
CL_EXTERN cl_int(CL_API_CALL* f_clEnqueueReadBufferRect)(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, const size_t *buffer_offset, const size_t *host_offset, const size_t *region, size_t buffer_row_pitch, size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
CL_EXTERN cl_int(CL_API_CALL* f_clEnqueueWriteBuffer) (cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueWriteBufferRect)(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, const size_t *buffer_offset, const size_t *host_offset, const size_t *region, size_t buffer_row_pitch, size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueCopyBuffer)(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset, size_t dst_offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueCopyBufferRect)(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, const size_t *src_origin, const size_t *dst_origin, const size_t *region, size_t src_row_pitch, size_t src_slice_pitch, size_t dst_row_pitch, size_t dst_slice_pitch, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueFillBuffer)(cl_command_queue command_queue, cl_mem buffer, const void *pattern, size_t pattern_size, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);

CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueReadImage)(cl_command_queue command_queue, cl_mem image, cl_bool blocking_read, const size_t origin[3], const size_t region[3], size_t row_pitch, size_t slice_pitch, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueWriteImage)(cl_command_queue command_queue, cl_mem image, cl_bool blocking_write, const size_t origin[3], const size_t region[3], size_t input_row_pitch, size_t input_slice_pitch, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueCopyImage)(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_image, const size_t src_origin[3], const size_t dst_origin[3], const size_t region[3], cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueCopyImageToBuffer)(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_buffer, const size_t src_origin[3], const size_t region[3], size_t dst_offset, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event *event);
CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueCopyBufferToImage)(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_image, size_t src_offset, const size_t dst_origin[3], const size_t region[3], cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event *event);
CL_EXTERN void *(CL_API_CALL *f_clEnqueueMapBuffer)(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_map, cl_map_flags map_flags, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event, cl_int *errcode_ret);
CL_EXTERN void *(CL_API_CALL *f_clEnqueueMapImage)(cl_command_queue  command_queue, cl_mem image, cl_bool blocking_map, cl_map_flags map_flags, const size_t origin[3], const size_t region[3], size_t *image_row_pitch, size_t *image_slice_pitch, cl_uint  num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event, cl_int *errcode_ret);
CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueUnmapMemObject)(cl_command_queue command_queue, cl_mem memobj, void *mapped_ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);

CL_EXTERN cl_int(CL_API_CALL *f_clWaitForEvents)(cl_uint num_events, const cl_event *event_list);
CL_EXTERN cl_int(CL_API_CALL *f_clGetEventInfo)(cl_event event, cl_event_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
CL_EXTERN cl_event(CL_API_CALL *f_clCreateUserEvent)(cl_context context, cl_int *errcode_ret);
CL_EXTERN cl_int(CL_API_CALL *f_clRetainEvent)(cl_event event);
CL_EXTERN cl_int(CL_API_CALL *f_clReleaseEvent)(cl_event event);
CL_EXTERN cl_int(CL_API_CALL *f_clSetUserEventStatus)(cl_event event, cl_int execution_status);
CL_EXTERN cl_int(CL_API_CALL *f_clGetEventProfilingInfo)(cl_event event, cl_profiling_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueWaitForEvents)(cl_command_queue command_queue, cl_uint num_events, const cl_event *event_list);
CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueMarker)(cl_command_queue command_queue, cl_event *event);

CL_EXTERN cl_semaphore_khr (CL_API_CALL *f_clCreateSemaphoreWithPropertiesKHR)(cl_context context, const cl_semaphore_properties_khr *sema_props, cl_int *errcode_ret);
CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueWaitSemaphoresKHR)(cl_command_queue command_queue,cl_uint num_sema_objects,const cl_semaphore_khr* sema_objects,const cl_semaphore_payload_khr* sema_payload_list,cl_uint num_events_in_wait_list,const cl_event* event_wait_list,cl_event* event);
CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueSignalSemaphoresKHR)(cl_command_queue command_queue,cl_uint num_sema_objects,const cl_semaphore_khr* sema_objects,const cl_semaphore_payload_khr* sema_payload_list,cl_uint num_events_in_wait_list,const cl_event* event_wait_list,cl_event* event);
CL_EXTERN cl_int(CL_API_CALL *f_clGetSemaphoreInfoKHR)(cl_semaphore_khr sema_object,cl_semaphore_info_khr param_name,size_t param_value_size,void* param_value,size_t* param_value_size_ret);
CL_EXTERN cl_int(CL_API_CALL *f_clReleaseSemaphoreKHR)(cl_semaphore_khr sema_object);

CL_EXTERN cl_int(CL_API_CALL *f_clFlush)(cl_command_queue command_queue);
CL_EXTERN cl_int(CL_API_CALL *f_clFinish)(cl_command_queue command_queue);

CL_EXTERN cl_int(CL_API_CALL *f_clGetKernelSubGroupInfo)(cl_kernel kernel, cl_device_id device, cl_kernel_sub_group_info param_name, size_t input_value_size, const void *input_value, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
CL_EXTERN cl_int(CL_API_CALL *f_clGetKernelSubGroupInfoKHR)(cl_kernel kernel, cl_device_id device, cl_kernel_sub_group_info param_name, size_t input_value_size, const void *input_value, size_t param_value_size, void *param_value, size_t *param_value_size_ret);

#if ENABLE_RGY_OPENCL_D3D9
CL_EXTERN cl_int (CL_API_CALL *f_clGetDeviceIDsFromDX9MediaAdapterKHR)(cl_platform_id platform, cl_uint num_media_adapters, cl_dx9_media_adapter_type_khr *media_adapter_type, void *media_adapters, cl_dx9_media_adapter_set_khr media_adapter_set, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices);
CL_EXTERN cl_mem(CL_API_CALL *f_clCreateFromDX9MediaSurfaceKHR)(cl_context context, cl_mem_flags flags, cl_dx9_media_adapter_type_khr adapter_type, void *surface_info, cl_uint plane, cl_int *errcode_ret);
CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueAcquireDX9MediaSurfacesKHR)(cl_command_queue command_queue, cl_uint num_objects, const cl_mem *mem_objects, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueReleaseDX9MediaSurfacesKHR)(cl_command_queue command_queue, cl_uint num_objects, const cl_mem *mem_objects, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);

CL_EXTERN cl_int(CL_API_CALL* f_clGetDeviceIDsFromDX9INTEL)(cl_platform_id platform, cl_dx9_device_source_intel dx9_device_source, void* dx9_object, cl_dx9_device_set_intel dx9_device_set, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices);
CL_EXTERN cl_mem(CL_API_CALL* f_clCreateFromDX9MediaSurfaceINTEL)(cl_context context, cl_mem_flags flags, IDirect3DSurface9* resource, HANDLE sharedHandle, UINT plane, cl_int* errcode_ret);
CL_EXTERN cl_int(CL_API_CALL* f_clEnqueueAcquireDX9ObjectsINTEL)(cl_command_queue command_queue, cl_uint  num_objects, const cl_mem* mem_objects, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
CL_EXTERN cl_int(CL_API_CALL* f_clEnqueueReleaseDX9ObjectsINTEL)(cl_command_queue command_queue, cl_uint num_objects, cl_mem* mem_objects, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
#endif //ENABLE_RGY_OPENCL_D3D9

#if ENABLE_RGY_OPENCL_D3D11
CL_EXTERN cl_int (CL_API_CALL *f_clGetDeviceIDsFromD3D11KHR)(cl_platform_id platform, cl_d3d11_device_source_khr d3d_device_source, void *d3d_object, cl_d3d11_device_set_khr d3d_device_set, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices);
CL_EXTERN cl_mem(CL_API_CALL *f_clCreateFromD3D11BufferKHR)(cl_context context, cl_mem_flags flags, ID3D11Buffer *resource, cl_int *errcode_ret);
CL_EXTERN cl_mem(CL_API_CALL *f_clCreateFromD3D11Texture2DKHR)(cl_context context, cl_mem_flags flags, ID3D11Texture2D *resource, UINT subresource, cl_int *errcode_ret);
CL_EXTERN cl_mem(CL_API_CALL *f_clCreateFromD3D11Texture3DKHR)(cl_context context, cl_mem_flags flags, ID3D11Texture3D *resource, UINT subresource, cl_int *errcode_ret);
CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueAcquireD3D11ObjectsKHR)(cl_command_queue command_queue, cl_uint num_objects, const cl_mem *mem_objects, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
CL_EXTERN cl_int(CL_API_CALL *f_clEnqueueReleaseD3D11ObjectsKHR)(cl_command_queue command_queue, cl_uint num_objects, const cl_mem *mem_objects, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
#endif //#if ENABLE_RGY_OPENCL_D3D11

#if ENABLE_RGY_OPENCL_VA
CL_EXTERN cl_int(CL_API_CALL* f_clGetDeviceIDsFromVA_APIMediaAdapterINTEL)(cl_platform_id platform, cl_va_api_device_source_intel media_adapter_type, void* media_adapter, cl_va_api_device_set_intel media_adapter_set, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices);
CL_EXTERN cl_mem(CL_API_CALL* f_clCreateFromVA_APIMediaSurfaceINTEL)(cl_context context, cl_mem_flags flags, VASurfaceID* surface, cl_uint plane, cl_int* errcode_ret);
CL_EXTERN cl_int(CL_API_CALL* f_clEnqueueAcquireVA_APIMediaSurfacesINTEL)(cl_command_queue command_queue, cl_uint num_objects, const cl_mem* mem_objects, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
CL_EXTERN cl_int(CL_API_CALL* f_clEnqueueReleaseVA_APIMediaSurfacesINTEL)(cl_command_queue command_queue, cl_uint num_objects, const cl_mem* mem_objects, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
#endif //#ifdef ENABLE_RGY_OPENCL_VA

#define clGetExtensionFunctionAddressForPlatform f_clGetExtensionFunctionAddressForPlatform

#define clGetPlatformIDs f_clGetPlatformIDs
#define clGetPlatformInfo f_clGetPlatformInfo
#define clGetDeviceIDs f_clGetDeviceIDs
#define clGetDeviceInfo f_clGetDeviceInfo

#define clCreateContext f_clCreateContext
#define clReleaseContext f_clReleaseContext
#define clCreateCommandQueue f_clCreateCommandQueue
#define clGetCommandQueueInfo f_clGetCommandQueueInfo
#define clReleaseCommandQueue f_clReleaseCommandQueue
#define clGetSupportedImageFormats f_clGetSupportedImageFormats

#define clCreateProgramWithSource f_clCreateProgramWithSource
#define clBuildProgram f_clBuildProgram
#define clGetProgramBuildInfo f_clGetProgramBuildInfo
#define clGetProgramInfo f_clGetProgramInfo
#define clReleaseProgram f_clReleaseProgram

#define clCreateBuffer f_clCreateBuffer
#define clCreateImage f_clCreateImage
#define clCreateImageWithProperties f_clCreateImageWithProperties
#define clReleaseMemObject f_clReleaseMemObject
#define clGetMemObjectInfo f_clGetMemObjectInfo
#define clGetImageInfo f_clGetImageInfo
#define clCreateKernel f_clCreateKernel
#define clReleaseKernel f_clReleaseKernel
#define clSetKernelArg f_clSetKernelArg
#define clEnqueueNDRangeKernel f_clEnqueueNDRangeKernel
#define clEnqueueTask f_clEnqueueTask

#define clEnqueueReadBuffer f_clEnqueueReadBuffer
#define clEnqueueReadBufferRect f_clEnqueueReadBufferRect
#define clEnqueueWriteBuffer f_clEnqueueWriteBuffer
#define clEnqueueWriteBufferRect f_clEnqueueWriteBufferRect
#define clEnqueueCopyBuffer f_clEnqueueCopyBuffer
#define clEnqueueCopyBufferRect f_clEnqueueCopyBufferRect
#define clEnqueueFillBuffer f_clEnqueueFillBuffer

#define clEnqueueReadImage f_clEnqueueReadImage
#define clEnqueueWriteImage f_clEnqueueWriteImage
#define clEnqueueCopyImage f_clEnqueueCopyImage
#define clEnqueueCopyImageToBuffer f_clEnqueueCopyImageToBuffer
#define clEnqueueCopyBufferToImage f_clEnqueueCopyBufferToImage
#define clEnqueueMapBuffer f_clEnqueueMapBuffer
#define clEnqueueMapImage f_clEnqueueMapImage
#define clEnqueueUnmapMemObject f_clEnqueueUnmapMemObject

#define clWaitForEvents f_clWaitForEvents
#define clGetEventInfo f_clGetEventInfo
#define clCreateUserEvent f_clCreateUserEvent
#define clRetainEvent f_clRetainEvent
#define clReleaseEvent f_clReleaseEvent
#define clSetUserEventStatus f_clSetUserEventStatus
#define clGetEventProfilingInfo f_clGetEventProfilingInfo
#define clEnqueueWaitForEvents f_clEnqueueWaitForEvents
#define clEnqueueMarker f_clEnqueueMarker

#define clCreateSemaphoreWithPropertiesKHR f_clCreateSemaphoreWithPropertiesKHR
#define clEnqueueWaitSemaphoresKHR f_clEnqueueWaitSemaphoresKHR
#define clEnqueueSignalSemaphoresKHR f_clEnqueueSignalSemaphoresKHR
#define clGetSemaphoreInfoKHR f_clGetSemaphoreInfoKHR
#define clReleaseSemaphoreKHR f_clReleaseSemaphoreKHR

#define clFlush f_clFlush
#define clFinish f_clFinish

#define clGetKernelSubGroupInfo f_clGetKernelSubGroupInfo
#define clGetKernelSubGroupInfoKHR f_clGetKernelSubGroupInfoKHR

#if ENABLE_RGY_OPENCL_D3D9
#define clGetDeviceIDsFromDX9MediaAdapterKHR f_clGetDeviceIDsFromDX9MediaAdapterKHR
#define clCreateFromDX9MediaSurfaceKHR f_clCreateFromDX9MediaSurfaceKHR
#define clEnqueueAcquireDX9MediaSurfacesKHR f_clEnqueueAcquireDX9MediaSurfacesKHR
#define clEnqueueReleaseDX9MediaSurfacesKHR f_clEnqueueReleaseDX9MediaSurfacesKHR

#define clGetDeviceIDsFromDX9INTEL f_clGetDeviceIDsFromDX9INTEL
#define clCreateFromDX9MediaSurfaceINTEL f_clCreateFromDX9MediaSurfaceINTEL
#define clEnqueueAcquireDX9ObjectsINTEL f_clEnqueueAcquireDX9ObjectsINTEL
#define clEnqueueReleaseDX9ObjectsINTEL f_clEnqueueReleaseDX9ObjectsINTEL
#endif //#if ENABLE_RGY_OPENCL_D3D9

#if ENABLE_RGY_OPENCL_D3D11
#define clGetDeviceIDsFromD3D11KHR f_clGetDeviceIDsFromD3D11KHR
#define clCreateFromD3D11BufferKHR f_clCreateFromD3D11BufferKHR
#define clCreateFromD3D11Texture2DKHR f_clCreateFromD3D11Texture2DKHR
#define clCreateFromD3D11Texture3DKHR f_clCreateFromD3D11Texture3DKHR
#define clEnqueueAcquireD3D11ObjectsKHR f_clEnqueueAcquireD3D11ObjectsKHR
#define clEnqueueReleaseD3D11ObjectsKHR f_clEnqueueReleaseD3D11ObjectsKHR
#endif //#if ENABLE_RGY_OPENCL_D3D11

#if ENABLE_RGY_OPENCL_VA
#define clGetDeviceIDsFromVA_APIMediaAdapterINTEL f_clGetDeviceIDsFromVA_APIMediaAdapterINTEL
#define clCreateFromVA_APIMediaSurfaceINTEL f_clCreateFromVA_APIMediaSurfaceINTEL
#define clEnqueueAcquireVA_APIMediaSurfacesINTEL f_clEnqueueAcquireVA_APIMediaSurfacesINTEL
#define clEnqueueReleaseVA_APIMediaSurfacesINTEL f_clEnqueueReleaseVA_APIMediaSurfacesINTEL
#endif //ENABLE_RGY_OPENCL_VA

tstring checkOpenCLDLL();

MAP_PAIR_0_1_PROTO(err, rgy, RGY_ERR, cl, cl_int);

static const int RGY_OPENCL_BUILD_THREAD_DEFAULT_MAX = 8;

class RGYOpenCLQueue;

typedef std::unique_ptr<std::remove_pointer<cl_context>::type, decltype(clReleaseContext)> unique_context;
typedef std::unique_ptr<std::remove_pointer<cl_command_queue>::type, decltype(clReleaseCommandQueue)> unique_queue;

static const TCHAR *cl_errmes(cl_int err) {
    return get_err_mes(err_cl_to_rgy(err));
}

enum RGYCLMemcpyKind {
    RGYCLMemcpyD2D,
    RGYCLMemcpyD2H,
    RGYCLMemcpyH2D,
    RGYCLMemcpyH2H,
};

static inline RGYCLMemcpyKind getMemcpyKind(RGY_MEM_TYPE inputDevice, RGY_MEM_TYPE outputDevice) {
    if (inputDevice != RGY_MEM_TYPE_CPU) {
        return (outputDevice != RGY_MEM_TYPE_CPU) ? RGYCLMemcpyD2D : RGYCLMemcpyD2H;
    } else {
        return (outputDevice != RGY_MEM_TYPE_CPU) ? RGYCLMemcpyH2D : RGYCLMemcpyH2H;
    }
}

static const TCHAR *getMemcpyKindStr(RGYCLMemcpyKind kind) {
    switch (kind) {
    case RGYCLMemcpyD2D:
        return _T("copyDtoD");
    case RGYCLMemcpyD2H:
        return _T("copyDtoH");
    case RGYCLMemcpyH2D:
        return _T("copyHtoD");
    case RGYCLMemcpyH2H:
        return _T("copyHtoH");
    default:
        return _T("copyUnknown");
    }
}

static const TCHAR *getMemcpyKindStr(RGY_MEM_TYPE inputDevice, RGY_MEM_TYPE outputDevice) {
    return getMemcpyKindStr(getMemcpyKind(inputDevice, outputDevice));
}

struct RGYOpenCLEventInfo {
    cl_command_queue queue;
    cl_command_type command_type;
    cl_context context;
    cl_int status;
    cl_uint ref_count;

    RGYOpenCLEventInfo() : queue(0), command_type(0), context(0), status(0), ref_count(0) {};
    ~RGYOpenCLEventInfo() {};

    tstring print() const;
};

struct cl_event_deleter {
    void operator()(cl_event *e) const {
        if (*e) {
            clReleaseEvent(*e);
        }
        delete e;
    }
};

class RGYOpenCLEvent {
public:
    RGYOpenCLEvent(const cl_event event) : event_(new cl_event, cl_event_deleter()) {
        *event_ = event;
    }

    RGYOpenCLEvent() : event_(new cl_event, cl_event_deleter()) {
        *event_ = nullptr;
    }

    RGY_ERR wait() const {
        return err_cl_to_rgy(clWaitForEvents(1, event_.get()));
    }
    void reset() {
        if (*event_ != nullptr) {
            event_ = std::shared_ptr<cl_event>(new cl_event, cl_event_deleter());
        }
        *event_ = nullptr;
    }
    cl_event *reset_ptr() {
        reset();
        return event_.get();
    }
    RGY_ERR getProfilingTimeStart(uint64_t& time);
    RGY_ERR getProfilingTimeEnd(uint64_t& time);
    RGY_ERR getProfilingTimeSubmit(uint64_t& time);
    RGY_ERR getProfilingTimeQueued(uint64_t& time);
    RGY_ERR getProfilingTimeComplete(uint64_t& time);
    cl_event &operator()() { return *event_; }
    const cl_event &operator()() const { return *event_; }
    const cl_event *ptr() const { return event_.get(); }
    static RGY_ERR wait(std::vector<RGYOpenCLEvent>& events) {
        auto err = CL_SUCCESS;
        if (events.size() > 0) {
            std::vector<cl_event> clevents(events.size());
            for (size_t i = 0; i < events.size(); i++)
                clevents[i] = events[i]();
            err = clWaitForEvents((int)events.size(), clevents.data());
        }
        return err_cl_to_rgy(err);
    }
    RGYOpenCLEventInfo getInfo() const;
private:
    RGY_ERR getProfilingTime(uint64_t& time, const cl_profiling_info info);
    std::shared_ptr<cl_event> event_;
};

 class RGYOpenCLSemaphore {
public:
    RGYOpenCLSemaphore() : semaphore_(std::make_unique<cl_semaphore_khr>(nullptr)) { }
    RGYOpenCLSemaphore(const cl_semaphore_khr semaphore) : semaphore_(std::make_unique<cl_semaphore_khr>(semaphore)) { }
    ~RGYOpenCLSemaphore() { release(); }
    RGY_ERR wait(RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events = {}, RGYOpenCLEvent *event = nullptr);
    RGY_ERR signal(RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events = {}, RGYOpenCLEvent *event = nullptr);
    void release();
private:
    std::unique_ptr<cl_semaphore_khr> semaphore_;
 };

#if !ENABLE_RGY_OPENCL_D3D9
typedef int cl_dx9_media_adapter_type_khr;
typedef struct _cl_dx9_surface_info_khr {
    void *resource; HANDLE shared_handle;
} cl_dx9_surface_info_khr;
#endif
#if !ENABLE_RGY_OPENCL_D3D11
typedef void ID3D11Resource;
#endif
struct RGYCLMemObjInfo {
    cl_mem_object_type memtype;
    cl_mem_flags memflags;
    size_t size;
    void *host_ptr;
    cl_uint map_count;
    cl_uint ref_count;
    size_t  mem_offset;
    cl_context context;
    cl_mem associated_mem;
    cl_bool is_svm_ptr;
    cl_dx9_media_adapter_type_khr d3d9_adapter_type;
    cl_dx9_surface_info_khr d3d9_surf_type;
    ID3D11Resource *d3d11resource;
    ID3D11Resource *d3d11subresource;
    void *va_surfaceId;
    cl_uint va_plane;
    cl_image_desc image;
    cl_image_format image_format;
    size_t image_elem_size;
    cl_uint d3d9_media_plane;

    RGYCLMemObjInfo() : memtype(0), memflags(0), size(0), host_ptr(nullptr), map_count(0), ref_count(0),
        mem_offset(0), context(nullptr), associated_mem(nullptr), is_svm_ptr(false),
        d3d9_adapter_type(0), d3d9_surf_type({ 0 }), d3d11resource(nullptr), d3d11subresource(nullptr), va_surfaceId(nullptr), va_plane(0),
        image(), image_elem_size(0), d3d9_media_plane(0) {
        memset(&image, 0, sizeof(image));
    };
    tstring print() const;
    bool isImageNormalizedType() const;
};

RGYCLMemObjInfo getRGYCLMemObjectInfo(cl_mem mem);

enum RGYCLMapBlock {
    RGY_CL_MAP_BLOCK_NONE,
    RGY_CL_MAP_BLOCK_ALL,
    RGY_CL_MAP_BLOCK_LAST
};

class RGYCLBufMap {
public:
    RGYCLBufMap(cl_mem mem) : m_mem(mem), m_queue(RGYDefaultQueue), m_hostPtr(nullptr), m_eventMap() {};
    ~RGYCLBufMap() {
        unmap();
    }
    RGY_ERR map(cl_map_flags map_flags, size_t size, RGYOpenCLQueue &queue);
    RGY_ERR map(cl_map_flags map_flags, size_t size, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, const RGYCLMapBlock block_map);
    RGY_ERR unmap();
    RGY_ERR unmap(RGYOpenCLQueue &queue);
    RGY_ERR unmap(RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);

    const RGYOpenCLEvent &event() const { return m_eventMap; }
    RGYOpenCLEvent &event() { return m_eventMap; }
    const void *ptr() const { return m_hostPtr; }
    void *ptr() { return m_hostPtr; }
protected:
    RGY_ERR unmap(cl_command_queue queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGYCLBufMap(const RGYCLBufMap &) = delete;
    void operator =(const RGYCLBufMap &) = delete;
    cl_mem m_mem;
    cl_command_queue m_queue;
    void *m_hostPtr;
    RGYOpenCLEvent m_eventMap;
};

class RGYCLBuf {
public:
    RGYCLBuf(cl_mem mem, cl_mem_flags flags, size_t size) : m_mem(mem), m_flags(flags), m_size(size), m_mapped() {
    };
    ~RGYCLBuf() {
        clear();
    }
    void clear() {
        m_mapped.reset();
        if (m_mem) {
            clReleaseMemObject(m_mem);
            m_mem = nullptr;
        }
    }
    cl_mem &mem() { return m_mem; }
    const cl_mem &mem() const { return m_mem; }
    size_t size() const { return m_size; }
    cl_mem_flags flags() const { return m_flags; }

    RGY_ERR queueMapBuffer(RGYOpenCLQueue &queue, cl_map_flags map_flags, const std::vector<RGYOpenCLEvent> &wait_events = {}, const RGYCLMapBlock block_map = RGY_CL_MAP_BLOCK_NONE);
    const RGYOpenCLEvent &mapEvent() const { return m_mapped->event(); }
    const void *mappedPtr() const { return m_mapped->ptr(); }
    void *mappedPtr() { return m_mapped->ptr(); }
    RGY_ERR unmapBuffer();
    RGY_ERR unmapBuffer(RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events = {});
    RGYCLMemObjInfo getMemObjectInfo() const;
protected:
    RGYCLBuf(const RGYCLBuf &) = delete;
    void operator =(const RGYCLBuf &) = delete;

    cl_mem m_mem;
    cl_mem_flags m_flags;
    size_t m_size;
    std::unique_ptr<RGYCLBufMap> m_mapped;
};

class RGYCLFrameMap;

struct RGYCLFrame : public RGYFrame {
public:
    RGYFrameInfo frame;
    cl_mem_flags clflags;
    std::unique_ptr<RGYCLFrameMap> m_mapped;
    RGYCLFrame()
        : frame(), clflags(0), m_mapped() {
    };
    RGYCLFrame(const RGYFrameInfo &info_, cl_mem_flags flags_ = CL_MEM_READ_WRITE)
        : frame(info_), clflags(flags_), m_mapped() {
    };
    RGY_ERR queueMapBuffer(RGYOpenCLQueue &queue, cl_map_flags map_flags, const std::vector<RGYOpenCLEvent> &wait_events = {}, const RGYCLMapBlock block_map = RGY_CL_MAP_BLOCK_NONE);
    RGY_ERR unmapBuffer();
    RGY_ERR unmapBuffer(RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events = {});
    RGY_ERR mapWait() const;
    bool isMapped() const;
    RGYCLFrameMap *mappedHost();
    const RGYCLFrameMap *mappedHost() const;
    std::vector<RGYOpenCLEvent>& mapEvents();
    RGYCLMemObjInfo getMemObjectInfo() const;
    void resetMappedFrame();
protected:
    RGYCLFrame(const RGYCLFrame &) = delete;
    void operator =(const RGYCLFrame &) = delete;
    virtual RGYFrameInfo getInfo() const override { return frameInfo(); };
public:
    virtual const RGYFrameInfo& frameInfo() const { return frame; }
    virtual bool isempty() const { return frame.ptr[0] == nullptr; }
    virtual void setTimestamp(uint64_t timestamp) override { frame.timestamp = timestamp; }
    virtual void setDuration(uint64_t duration) override { frame.duration = duration; }
    virtual void setPicstruct(RGY_PICSTRUCT picstruct) override { frame.picstruct = picstruct; }
    virtual void setInputFrameId(int id) override { frame.inputFrameId = id; }
    virtual void setFlags(RGY_FRAME_FLAGS frameflags) override { frame.flags = frameflags; }
    virtual void clearDataList() override { frame.dataList.clear(); }
    virtual const std::vector<std::shared_ptr<RGYFrameData>>& dataList() const override { return frame.dataList; }
    virtual std::vector<std::shared_ptr<RGYFrameData>>& dataList() override { return frame.dataList; }
    virtual void setDataList(const std::vector<std::shared_ptr<RGYFrameData>>& dataList) override { frame.dataList = dataList; }
    cl_mem mem(int i) const {
        return (cl_mem)frame.ptr[i];
    }
    void clear();
    virtual ~RGYCLFrame() {
        m_mapped.reset();
        clear();
    }
};

class RGYCLFrameMap : public RGYCLFrame {
public:
    RGYCLFrameMap(RGYCLFrame *dev, RGYOpenCLQueue &queue);
    virtual ~RGYCLFrameMap() {
        unmap();
    }
    RGY_ERR map(cl_map_flags map_flags, RGYOpenCLQueue &queue);
    RGY_ERR map(cl_map_flags map_flags, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, const RGYCLMapBlock block_map);
    RGY_ERR unmap();
    RGY_ERR unmap(RGYOpenCLQueue &queue);
    RGY_ERR unmap(RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);

    RGY_ERR map_wait() { return RGYOpenCLEvent::wait(m_eventMap); };
    const RGYFrameInfo& host() const { return frame; }
    std::vector<RGYOpenCLEvent>& mapEvents() { return m_eventMap; }
public:
    virtual bool isempty() const { return frame.ptr[0] == nullptr; }
    virtual void setTimestamp(uint64_t timestamp) override;
    virtual void setDuration(uint64_t duration) override;
    virtual void setPicstruct(RGY_PICSTRUCT picstruct) override;
    virtual void setInputFrameId(int id) override;
    virtual void setFlags(RGY_FRAME_FLAGS frameflags) override;
    virtual void clearDataList() override;
    virtual const std::vector<std::shared_ptr<RGYFrameData>>& dataList() const override;
    virtual std::vector<std::shared_ptr<RGYFrameData>>& dataList() override;
    virtual void setDataList(const std::vector<std::shared_ptr<RGYFrameData>>& dataList) override;
protected:
    RGY_ERR unmap(cl_command_queue queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGYCLFrameMap(const RGYCLFrameMap &) = delete;
    void operator =(const RGYCLFrameMap &) = delete;
    RGYCLFrame *m_dev;
    cl_command_queue m_queue;
    std::vector<RGYOpenCLEvent> m_eventMap;
};

enum RGYCLFrameInteropType {
    RGY_INTEROP_NONE,
    RGY_INTEROP_DX9,
    RGY_INTEROP_DX11,
    RGY_INTEROP_VA,
    RGY_INTEROP_VULKAN,
};

struct RGYCLFrameInterop : public RGYCLFrame {
protected:
    RGYCLFrameInteropType m_interop;
    RGYOpenCLQueue& m_interop_queue;
    std::shared_ptr<RGYLog> m_log;
    bool m_acquired;
public:
    RGYCLFrameInterop(const RGYFrameInfo &info, cl_mem_flags flags, RGYCLFrameInteropType interop, RGYOpenCLQueue& interop_queue, shared_ptr<RGYLog> log)
        : RGYCLFrame(info, flags), m_interop(interop), m_interop_queue(interop_queue), m_log(log), m_acquired(false) {
    };
    RGY_ERR acquire(RGYOpenCLQueue &queue, RGYOpenCLEvent *event = nullptr);
protected:
    RGYCLFrameInterop(const RGYCLFrameInterop &) = delete;
    void operator =(const RGYCLFrameInterop &) = delete;
public:
    const RGYCLFrameInteropType interop() const { return m_interop; }
    RGY_ERR release(RGYOpenCLEvent *event = nullptr);
    virtual ~RGYCLFrameInterop() {
        release();
    }
};

struct RGYOpenCLDeviceInfoVecWidth {
    std::pair<int, int> w_char, w_short, w_int, w_long, w_half, w_float, w_double;

    RGYOpenCLDeviceInfoVecWidth();
    std::string print() const;
};

struct RGYOpenCLDeviceInfo {
    cl_device_type type;
    int vendor_id;
    int max_compute_units;
    int max_clock_frequency;
    int max_samplers;
    uint64_t global_mem_size;
    uint64_t global_mem_cache_size;
    uint32_t global_mem_cacheline_size;
    uint64_t local_mem_size;
    uint32_t image_support;
    size_t image_2d_max_width;
    size_t image_2d_max_height;
    size_t image_3d_max_width;
    size_t image_3d_max_height;
    uint32_t image_3d_max_depth;
    int image_pitch_alignment;
    size_t profiling_timer_resolution;
    int max_const_args;
    uint64_t max_const_buffer_size;
    uint64_t max_mem_alloc_size;
    size_t max_parameter_size;
    int max_read_image_args;
    size_t max_work_group_size;
    int max_work_item_dims;
    int max_write_image_args;
    int mem_base_addr_align;
    int min_data_type_align_size;

    RGYOpenCLDeviceInfoVecWidth vecwidth;

    std::string name;
    std::string vendor;
    std::string driver_version;
    std::string profile;
    std::string version;
    std::string extensions;
    uint8_t uuid[CL_UUID_SIZE_KHR];
    uint8_t luid[CL_LUID_SIZE_KHR];

#if ENCODER_QSV || CLFILTERS_AUF
    int ip_version_intel;
    uint32_t id_intel;
    uint32_t num_slices_intel;
    uint32_t num_subslices_intel;
    uint32_t num_eus_per_subslice_intel;
    uint32_t num_threads_per_eu_intel;
    cl_device_feature_capabilities_intel feature_capabilities_intel;
#endif
#if ENCODER_NVENC || CLFILTERS_AUF
    uint32_t cc_major_nv;
    uint32_t cc_minor_nv;
    uint32_t regs_per_block_nv;
    uint32_t warp_size_nv;
    int gpu_overlap_nv;
    int kernel_exec_timeout_nv;
    int integrated_mem_nv;
#endif
#if ENCODER_VCEENC || CLFILTERS_AUF
    std::string topology_amd;
    std::string board_name_amd;
    uint64_t global_free_mem_size_amd;
    int simd_per_cu_amd;
    int simd_width_amd;
    int simd_instruction_width_amd;
    int wavefront_width_amd;
    int global_mem_channels_amd;
    int global_mem_channel_banks_amd;
    int global_mem_channel_bank_width_amd;
    int local_mem_size_per_cu_amd;
    int local_mem_banks_amd;
    int thread_trace_supported_amd;
    int async_queue_support_amd;
    int max_work_group_size_amd;
    int preferred_const_buffer_size_amd;
    int pcie_id_amd;
#endif

    RGYOpenCLDeviceInfo();
    std::pair<int, int> clversion() const;
    bool checkVersion(int major, int minor) const;
    bool checkExtension(const char* extension) const;
};

class RGYOpenCLDevice {
public:
    RGYOpenCLDevice(cl_device_id device);
    virtual ~RGYOpenCLDevice() {};
    RGYOpenCLDeviceInfo info() const;
    bool checkVersion(int major, int minor) const;
    bool checkExtension(const char* extension) const;
    tstring infostr(bool full = false) const;
    cl_device_id id() const { return m_device; }
protected:
    cl_device_id m_device;
};

struct RGYOpenCLPlatformInfo {
    std::string profile;
    std::string version;
    std::string name;
    std::string vendor;
    std::string extensions;

    RGYOpenCLPlatformInfo();
    tstring print() const;
    std::pair<int, int> clversion() const;
    bool checkVersion(int major, int minor) const;
    bool checkExtension(const char* extension) const;
};

enum class RGYOpenCLSubGroupSupport {
    NONE,      // unsupported
    INTEL_EXT, // use cl_intel_subgroups extension
    STD20KHR,  // OpenCL 2.0 extension
    STD22,     // OpenCL 2.2 core
};

class RGYOpenCLPlatform {
public:
    RGYOpenCLPlatform(cl_platform_id platform, shared_ptr<RGYLog> pLog);
    virtual ~RGYOpenCLPlatform() {};
    RGY_ERR createDeviceList(cl_device_type device_type);
    RGY_ERR createDeviceListD3D9(cl_device_type device_type, void *d3d9dev, const bool tryMode = false);
    RGY_ERR createDeviceListD3D11(cl_device_type device_type, void *d3d11dev, const bool tryMode = false);
    RGY_ERR createDeviceListVA(cl_device_type device_type, void *devVA, const bool tryMode = false);
    RGY_ERR loadSubGroupKHR();
    RGYOpenCLSubGroupSupport checkSubGroupSupport(const cl_device_id devid);
    cl_platform_id get() const { return m_platform; };
    const void *d3d9dev() const { return m_d3d9dev; };
    const void *d3d11dev() const { return m_d3d11dev; };
    const void *vadev() const { return m_vadev; };
    std::vector<cl_device_id>& devs() { return m_devices; };
    RGYOpenCLDevice dev(int idx) { return RGYOpenCLDevice(m_devices[idx]); };
    const std::vector<cl_device_id>& devs() const { return m_devices; };
    void setDev(cl_device_id dev) { m_devices.clear(); m_devices.push_back(dev); };
    RGY_ERR setDev(cl_device_id dev, void *d3d9dev, void *d3d11dev);
    void setDevs(std::vector<cl_device_id> &devs) { m_devices = devs; };
    bool isVendor(const char *vendor) const;
    bool checkExtension(const char* extension) const;
    bool checkVersion(int major, int minor) const;
    RGYOpenCLPlatformInfo info() const;
protected:

    cl_platform_id m_platform;
    void *m_d3d9dev;
    void *m_d3d11dev;
    void *m_vadev;
    std::vector<cl_device_id> m_devices;
    shared_ptr<RGYLog> m_log;
};

template<typename T>
T divCeil(T i, T div) {
    return (i + div - 1) / div;
}

struct RGYWorkSize {
    size_t w[3];
    RGYWorkSize() {
        w[0] = w[1] = w[2] = 1;
    }
    RGYWorkSize(size_t x) {
        w[0] = x;
        w[1] = w[2] = 1;
    };
    RGYWorkSize(size_t x, size_t y) {
        w[0] = x;
        w[1] = y;
        w[2] = 1;
    }
    RGYWorkSize(size_t x, size_t y, size_t z) {
        w[0] = x;
        w[1] = y;
        w[2] = z;
    }
    size_t total() const {
        return w[0] * w[1] * w[2];
    }
    const size_t *operator()() const {
        return &w[0];
    }
    const size_t operator()(int i) const {
        return w[i];
    }
    RGYWorkSize groups(const RGYWorkSize &local) const {
        RGYWorkSize group = *this;
        for (int i = 0; i < 3; i++) {
            if (local.w[i] > 0) {
                group.w[i] = divCeil(w[i], local.w[i]);
            }
        }
        return group;
    }
    RGYWorkSize ceilGlobal(const RGYWorkSize& local) const {
        const RGYWorkSize group = groups(local);
        RGYWorkSize global = *this;
        for (int i = 0; i < 3; i++) {
            if (local.w[i] > 0) {
                global.w[i] = group.w[i] * local.w[i];
            }
        }
        return global;
    }
};

class RGYOpenCLKernelDynamicLocal {
protected:
    size_t size_;
public:
    RGYOpenCLKernelDynamicLocal(size_t size) : size_(size) {};
    ~RGYOpenCLKernelDynamicLocal() {};
    size_t size() const { return size_; }
};

class RGYOpenCLKernelLauncher {
public:
    RGYOpenCLKernelLauncher(cl_kernel kernel, std::string kernelName, RGYOpenCLQueue &queue, const RGYWorkSize &local, const RGYWorkSize &global, shared_ptr<RGYLog> pLog, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    virtual ~RGYOpenCLKernelLauncher() {};

    size_t subGroupSize() const;
    size_t subGroupCount() const;
    RGY_ERR launch(std::vector<void *> arg_ptrs = std::vector<void *>(), std::vector<size_t> arg_size = std::vector<size_t>(), std::vector<std::type_index> = std::vector<std::type_index>());

    template <typename... ArgTypes>
    RGY_ERR operator()(ArgTypes... args) {
        return launch(args...);
    }
    template <typename... ArgTypes>
    RGY_ERR launch(ArgTypes... args) {
        return this->launch(
            std::vector<void *>({ (void *)&args... }),
            std::vector<size_t>({ sizeof(args)... }),
            std::vector<std::type_index>({ typeid(args)... })
        );
    }
protected:
    cl_kernel m_kernel;
    std::string m_kernelName;
    RGYOpenCLQueue &m_queue;
    RGYWorkSize m_local;
    RGYWorkSize m_global;
    shared_ptr<RGYLog> m_log;
    std::vector<cl_event> m_wait_events;
    RGYOpenCLEvent *m_event;
};

class RGYOpenCLKernel {
public:
    RGYOpenCLKernel() : m_kernel(), m_kernelName(), m_log() {};
    RGYOpenCLKernel(cl_kernel kernel, std::string kernelName, shared_ptr<RGYLog> pLog);
    cl_kernel get() const { return m_kernel; }
    const std::string& name() const { return m_kernelName; }
    virtual ~RGYOpenCLKernel();
    RGYOpenCLKernelLauncher config(RGYOpenCLQueue &queue, const RGYWorkSize &local, const RGYWorkSize &global, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event = nullptr);
protected:
    cl_kernel m_kernel;
    std::string m_kernelName;
    shared_ptr<RGYLog> m_log;
};

class RGYOpenCLKernelHolder {
public:
    RGYOpenCLKernelHolder(RGYOpenCLKernel *kernel, shared_ptr<RGYLog> pLog);
    ~RGYOpenCLKernelHolder() {};
    RGYOpenCLKernel *get() const { return m_kernel; }
    RGYOpenCLKernelLauncher config(RGYOpenCLQueue &queue, const RGYWorkSize &local, const RGYWorkSize &global);
    RGYOpenCLKernelLauncher config(RGYOpenCLQueue &queue, const RGYWorkSize &local, const RGYWorkSize &global, RGYOpenCLEvent *event);
    RGYOpenCLKernelLauncher config(RGYOpenCLQueue &queue, const RGYWorkSize &local, const RGYWorkSize &global, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event = nullptr);
protected:
    RGYOpenCLKernel *m_kernel;
    shared_ptr<RGYLog> m_log;
};

class RGYOpenCLProgram {
public:
    RGYOpenCLProgram(cl_program program, shared_ptr<RGYLog> pLog);
    virtual ~RGYOpenCLProgram();

    RGYOpenCLKernelHolder kernel(const char *kernelName);
    std::vector<uint8_t> getBinary();
protected:
    cl_program m_program;
    shared_ptr<RGYLog> m_log;
    std::vector<std::unique_ptr<RGYOpenCLKernel>> m_kernels;
};

class RGYOpenCLProgramAsync {
public:
    RGYOpenCLProgramAsync() : m_future(), m_program() {};
    RGYOpenCLProgramAsync(std::future<std::unique_ptr<RGYOpenCLProgram>>& future) : m_future(std::move(future)) {};
    virtual ~RGYOpenCLProgramAsync() { clear(); }
    void set(std::future<std::unique_ptr<RGYOpenCLProgram>> future) {
        m_future = std::move(future);
    }
    RGYOpenCLProgram *get() {
        if (m_future.valid()) {
            m_program = m_future.get();
        }
        return m_program.get();
    }
    void wait() const {
        return m_future.wait();
    }
    template <class Rep, class Period>
    std::future_status wait_for(const std::chrono::duration<Rep, Period>& rel_time) const {
        return m_future.wait_for(rel_time);
    }
    template <class Rep, class Period>
    std::future_status wait_until(const std::chrono::duration<Rep, Period>& abs_time) const {
        return m_future.wait_until(abs_time);
    }
    void clear() {
        if (m_future.valid()) {
            m_program = m_future.get();
        }
        m_program.reset();
    }
protected:
    std::future<std::unique_ptr<RGYOpenCLProgram>> m_future;
    std::unique_ptr<RGYOpenCLProgram> m_program;
};

struct RGYOpenCLQueueInfo {
    cl_context context;
    cl_device_id devid;
    cl_uint refcount;
    cl_command_queue_properties properties;

    RGYOpenCLQueueInfo();
    tstring print() const;
};

class RGYOpenCLQueue {
public:
    RGYOpenCLQueue();
    RGYOpenCLQueue(cl_command_queue queue, cl_device_id devid);
    RGYOpenCLQueue(RGYOpenCLQueue &&) = default;
    RGYOpenCLQueue &operator=(RGYOpenCLQueue &&rhs) {
        if (this != &rhs) {
            m_queue = std::move(rhs.m_queue);
            m_devid = rhs.m_devid;
        }
        return *this;
    }
    virtual ~RGYOpenCLQueue();

    cl_command_queue operator()() { return m_queue.get(); }
    const cl_command_queue operator()() const { return m_queue.get(); }
    const cl_command_queue get() const {
        return m_queue.get();
    }
    cl_device_id devid() const {
        return m_devid;
    }
    RGYOpenCLQueueInfo getInfo() const;
    cl_command_queue_properties getProperties() const;
    RGY_ERR wait(const RGYOpenCLEvent& event) const;
    RGY_ERR getmarker(RGYOpenCLEvent& event) const;
    RGY_ERR flush() const;
    RGY_ERR finish() const;
    void clear();
protected:
    RGYOpenCLQueue(const RGYOpenCLQueue &) = delete;
    void operator =(const RGYOpenCLQueue &) = delete;
    unique_queue m_queue;
    cl_device_id m_devid;
};

enum class RGYFrameCopyMode {
    FRAME,
    FIELD_TOP,
    FIELD_BOTTOM
};

class RGYCLFramePool;

struct RGYCLImageFromBufferDeleter {
    RGYCLImageFromBufferDeleter();
    RGYCLImageFromBufferDeleter(RGYCLFramePool *pool);
    void operator()(RGYCLFrame *frame);
private:
    RGYCLFramePool *m_pool;
};

class RGYCLFramePool {
public:
    RGYCLFramePool();
    ~RGYCLFramePool();
    void clear();
    void add(RGYCLFrame *frame);
    std::unique_ptr<RGYCLFrame, RGYCLImageFromBufferDeleter> get(const RGYFrameInfo &frame, const bool normalized, const cl_mem_flags flags);
private:
    std::deque<std::unique_ptr<RGYCLFrame>> m_pool;
};

class RGYOpenCLContext {
public:
    RGYOpenCLContext(shared_ptr<RGYOpenCLPlatform> platform, int buildThreads, shared_ptr<RGYLog> pLog);
    virtual ~RGYOpenCLContext();

    RGY_ERR createContext(const cl_command_queue_properties queue_properties);
    cl_context context() const { return m_context.get(); };
    const RGYOpenCLQueue& queue(int idx=0) const { return m_queue[idx]; };
    RGYOpenCLQueue& queue(int idx=0) { return m_queue[idx]; };
    RGYOpenCLPlatform *platform() const { return m_platform.get(); };

    RGYThreadPool *threadPool() {
        if (!m_threadPool) { m_threadPool = std::make_unique<RGYThreadPool>(m_buildThreads); }
        return m_threadPool.get();
    }

    void setModuleHandle(const HMODULE hmodule) { m_hmodule = hmodule; }
    HMODULE getModuleHandle() const { return m_hmodule; }
    std::unique_ptr<RGYOpenCLProgram> build(const std::string& source, const char *options);
    std::unique_ptr<RGYOpenCLProgram> buildFile(const tstring filename, const std::string options);
    std::unique_ptr<RGYOpenCLProgram> buildResource(const tstring name, const tstring type, const std::string options);

    std::future<std::unique_ptr<RGYOpenCLProgram>> buildAsync(const std::string& source, const char *options);
    std::future<std::unique_ptr<RGYOpenCLProgram>> buildFileAsync(const tstring &filename, const char *options);
    std::future<std::unique_ptr<RGYOpenCLProgram>> buildResourceAsync(const TCHAR *name, const TCHAR *type, const char *options);

    RGYOpenCLQueue createQueue(const cl_device_id devid, const cl_command_queue_properties properties);
    std::unique_ptr<RGYCLBuf> createBuffer(size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE, void *host_ptr = nullptr);
    std::unique_ptr<RGYCLBuf> copyDataToBuffer(const void *host_ptr, size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE, cl_command_queue queue = 0);
    RGY_ERR createImageFromPlane(cl_mem& image, const cl_mem buffer, const int bit_depth, const int channel_order, const bool normalized, const int pitch, const int width, const int height, const cl_mem_flags flags);
    RGY_ERR createImageFromFrame(RGYFrameInfo& frameImage, const RGYFrameInfo& frame, const bool normalized, const bool cl_image2d_from_buffer_support, const cl_mem_flags flags);
    std::unique_ptr<RGYCLFrame, RGYCLImageFromBufferDeleter> createImageFromFrameBuffer(const RGYFrameInfo &frame, const bool normalized, const cl_mem_flags flags, RGYCLFramePool *imgpool);
    std::unique_ptr<RGYCLFrame> createFrameBuffer(const int width, const int height, const RGY_CSP csp, const int bitdepth, const cl_mem_flags flags = CL_MEM_READ_WRITE);
    std::unique_ptr<RGYCLFrame> createFrameBuffer(const RGYFrameInfo &frame, cl_mem_flags flags = CL_MEM_READ_WRITE);
    std::unique_ptr<RGYCLFrameInterop> createFrameFromD3D9Surface(void *surf, HANDLE shared_handle, const RGYFrameInfo &frame, RGYOpenCLQueue& queue, cl_mem_flags flags = CL_MEM_READ_WRITE);
    std::unique_ptr<RGYCLFrameInterop> createFrameFromD3D11Surface(void *surf, const RGYFrameInfo &frame, RGYOpenCLQueue& queue, cl_mem_flags flags = CL_MEM_READ_WRITE);
    std::unique_ptr<RGYCLFrameInterop> createFrameFromD3D11SurfacePlanar(const RGYFrameInfo &frame, RGYOpenCLQueue& queue, cl_mem_flags flags = CL_MEM_READ_WRITE);
    std::unique_ptr<RGYCLFrameInterop> createFrameFromVASurface(void *surf, const RGYFrameInfo &frame, RGYOpenCLQueue& queue, cl_mem_flags flags = CL_MEM_READ_WRITE);
    RGY_ERR copyFrame(RGYFrameInfo *dst, const RGYFrameInfo *src);
    RGY_ERR copyFrame(RGYFrameInfo *dst, const RGYFrameInfo *src, const sInputCrop *srcCrop);
    RGY_ERR copyFrame(RGYFrameInfo *dst, const RGYFrameInfo *src, const sInputCrop *srcCrop, RGYOpenCLQueue &queue);
    RGY_ERR copyFrame(RGYFrameInfo *dst, const RGYFrameInfo *src, const sInputCrop *srcCrop, RGYOpenCLQueue &queue, RGYOpenCLEvent *event);
    RGY_ERR copyFrame(RGYFrameInfo *dst, const RGYFrameInfo *src, const sInputCrop *srcCrop, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event = nullptr, RGYFrameCopyMode copyMode = RGYFrameCopyMode::FRAME);
    RGY_ERR copyPlane(RGYFrameInfo *dst, const RGYFrameInfo *src);
    RGY_ERR copyPlane(RGYFrameInfo *dst, const RGYFrameInfo *src, const sInputCrop *srcCrop);
    RGY_ERR copyPlane(RGYFrameInfo *dst, const RGYFrameInfo *src, const sInputCrop *srcCrop, RGYOpenCLQueue &queue);
    RGY_ERR copyPlane(RGYFrameInfo *dst, const RGYFrameInfo *src, const sInputCrop *srcCrop, RGYOpenCLQueue &queue, RGYOpenCLEvent *event);
    RGY_ERR copyPlane(RGYFrameInfo *dst, const RGYFrameInfo *src, const sInputCrop *srcCrop, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event = nullptr, RGYFrameCopyMode copyMode = RGYFrameCopyMode::FRAME);
    RGY_ERR setPlane(int value, RGYFrameInfo *dst);
    RGY_ERR setPlane(int value, RGYFrameInfo *dst, const sInputCrop *srcCrop);
    RGY_ERR setPlane(int value, RGYFrameInfo *dst, const sInputCrop *srcCrop, RGYOpenCLQueue &queue);
    RGY_ERR setPlane(int value, RGYFrameInfo *dst, const sInputCrop *srcCrop, RGYOpenCLQueue &queue, RGYOpenCLEvent *event);
    RGY_ERR setPlane(int value, RGYFrameInfo *dst, const sInputCrop *srcCrop, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event = nullptr);
    RGY_ERR setFrame(int value, RGYFrameInfo *dst);
    RGY_ERR setFrame(int value, RGYFrameInfo *dst, const sInputCrop *srcCrop);
    RGY_ERR setFrame(int value, RGYFrameInfo *dst, const sInputCrop *srcCrop, RGYOpenCLQueue &queue);
    RGY_ERR setFrame(int value, RGYFrameInfo *dst, const sInputCrop *srcCrop, RGYOpenCLQueue &queue, RGYOpenCLEvent *event);
    RGY_ERR setFrame(int value, RGYFrameInfo *dst, const sInputCrop *srcCrop, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event = nullptr);
    RGY_ERR setBuf(const void *pattern, size_t pattern_size, size_t fill_size_byte, RGYCLBuf *buf);
    RGY_ERR setBuf(const void *pattern, size_t pattern_size, size_t fill_size_byte, RGYCLBuf *buf, RGYOpenCLQueue &queue);
    RGY_ERR setBuf(const void *pattern, size_t pattern_size, size_t fill_size_byte, RGYCLBuf *buf, RGYOpenCLQueue &queue, RGYOpenCLEvent *event);
    RGY_ERR setBuf(const void *pattern, size_t pattern_size, size_t fill_size_byte, RGYCLBuf *buf, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event = nullptr);
    std::string cspCopyOptions(const RGYFrameInfo& dst, const RGYFrameInfo& src) const;
    void requestCSPCopy(const RGYFrameInfo& dst, const RGYFrameInfo& src);
    RGYOpenCLProgram *getCspCopyProgram(const RGYFrameInfo& dst, const RGYFrameInfo& src);

    std::vector<cl_image_format> getSupportedImageFormats(const cl_mem_object_type image_type = CL_MEM_OBJECT_IMAGE2D) const;
    tstring getSupportedImageFormatsStr(const cl_mem_object_type image_type = CL_MEM_OBJECT_IMAGE2D) const;
protected:
    std::unique_ptr<RGYOpenCLProgram> buildProgram(std::string datacopy, const std::string options);

    shared_ptr<RGYOpenCLPlatform> m_platform;
    unique_context m_context;
    std::vector<RGYOpenCLQueue> m_queue;
    std::shared_ptr<RGYLog> m_log;
    std::unordered_map<std::string, RGYOpenCLProgramAsync> m_copy;
    std::unique_ptr<RGYThreadPool> m_threadPool;
    int m_buildThreads;
    HMODULE m_hmodule;
};

class RGYOpenCL {
public:
    static HMODULE openCLHandle;
    static bool openCLCrush;

    RGYOpenCL();
    RGYOpenCL(shared_ptr<RGYLog> pLog);
    virtual ~RGYOpenCL();

    std::vector<shared_ptr<RGYOpenCLPlatform>> getPlatforms(const char *vendor = nullptr);
    static bool openCLloaded() { return openCLHandle != nullptr; };
protected:
    shared_ptr<RGYLog> m_log;
};

int initOpenCLGlobal();
tstring getOpenCLInfo(const cl_device_type device_type);

#endif //ENABLE_OPENCL

#endif //__RGY_OPENCL_H__
