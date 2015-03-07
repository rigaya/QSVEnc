/*********************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2014 Intel Corporation. All Rights Reserved.

**********************************************************************************/

#ifndef __PLUGIN_UTILS_H__
#define __PLUGIN_UTILS_H__

#include "sample_defs.h"
#include "sample_types.h"

#if defined(_WIN32) || defined(_WIN64)
    #define MSDK_CPU_ROTATE_PLUGIN  MSDK_STRING("sample_rotate_plugin.dll")
    #define MSDK_OCL_ROTATE_PLUGIN  MSDK_STRING("sample_plugin_opencl.dll")
#else
    #define MSDK_CPU_ROTATE_PLUGIN  MSDK_STRING("libsample_rotate_plugin.so")
    #define MSDK_OCL_ROTATE_PLUGIN  MSDK_STRING("libsample_plugin_opencl.so")
#endif

typedef mfxI32 msdkComponentType;
enum
{
    MSDK_VDECODE = 0x0001,
    MSDK_VENCODE = 0x0002,
    MSDK_VPP = 0x0004,
    MSDK_VENC = 0x0008,
};

typedef enum {
    MFX_PLUGINLOAD_TYPE_GUID = 1,
    MFX_PLUGINLOAD_TYPE_FILE = 2
} MfxPluginLoadType;

static const mfxPluginUID MSDK_PLUGINGUID_NULL = {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}};

bool AreGuidsEqual(const mfxPluginUID& guid1, const mfxPluginUID& guid2);

mfxStatus ConvertStringToGuid(const msdk_string & strGuid, mfxPluginUID & mfxGuid);

const mfxPluginUID & msdkGetPluginUID(mfxIMPL impl, msdkComponentType type, mfxU32 uCodecid);

#endif //__PLUGIN_UTILS_H__
