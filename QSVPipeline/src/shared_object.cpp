/*********************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2012-2014 Intel Corporation. All Rights Reserved.

**********************************************************************************/

#include "mfx_samples_config.h"

#if defined(_WIN32) || defined(_WIN64)

#include "vm/so_defs.h"

#include <windows.h>

msdk_so_handle msdk_so_load(const msdk_char *file_name)
{
    if (!file_name) return NULL;
    return (msdk_so_handle) LoadLibrary((LPCTSTR)file_name);
}

msdk_func_pointer msdk_so_get_addr(msdk_so_handle handle, const char *func_name)
{
    if (!handle) return NULL;
    return (msdk_func_pointer)GetProcAddress((HMODULE)handle, /*(LPCSTR)*/func_name);
}

void msdk_so_free(msdk_so_handle handle)
{
    if (!handle) return;
    FreeLibrary((HMODULE)handle);
}

#endif // #if defined(_WIN32) || defined(_WIN64)
