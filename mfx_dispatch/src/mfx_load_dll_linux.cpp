/* ****************************************************************************** *\

Copyright (C) 2012-2014 Intel Corporation.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
- Neither the name of Intel Corporation nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY INTEL CORPORATION "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL INTEL CORPORATION BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

File Name: mfx_load_dll_linux.cpp

\* ****************************************************************************** */

#if !defined(_WIN32) && !defined(_WIN64)

#include "mfx_dispatcher.h"
#include <dlfcn.h>
#include <string.h>

#if !defined(_DEBUG)

#if defined(LINUX64)
const msdk_disp_char * defaultDLLName[2] = {"libmfxhw64.so",
                                            "libmfxsw64.so"};
const msdk_disp_char * defaultAudioDLLName[2] = {"libmfxaudiosw64.so",
                                                 "libmfxaudiosw64.so"};

const msdk_disp_char * defaultPluginDLLName[2] = {"libmfxplugin64_hw.so",
                                                 "libmfxplugin64_sw.so"};


#elif defined(__APPLE__)
#ifdef __i386__
const msdk_disp_char * defaultDLLName[2] = {"libmfxhw32.dylib",
                                            "libmfxsw32.dylib"};
const msdk_disp_char * defaultAudioDLLName[2] = {"libmfxaudiosw32.dylib",
                                            "libmfxaudiosw32.dylib"};

const msdk_disp_char * defaultPluginDLLName[2] = {"libmfxplugin32_hw.dylib",
                                                  "libmfxplugin32_sw.dylib"};
#else
const msdk_disp_char * defaultDLLName[2] = {"libmfxhw64.dylib",
                                            "libmfxsw64.dylib"};
const msdk_disp_char * defaultAudioDLLName[2] = {"libmfxaudiosw64.dylib",
                                            "libmfxaudiosw64.dylib"};

const msdk_disp_char * defaultPluginDLLName[2] = {"libmfxplugin64_hw.dylib",
                                                  "libmfxplugin64_sw.dylib"};

#endif // #ifdef __i386__ for __APPLE__

#else // for Linux32 and Android
const msdk_disp_char * defaultDLLName[2] = {"libmfxhw32.so",
                                            "libmfxsw32.so"};
const msdk_disp_char * defaultAudioDLLName[2] = {"libmfxaudiosw32.so",
                                            "libmfxaudiosw32.so"};

const msdk_disp_char * defaultPluginDLLName[2] = {"libmfxplugin32_hw.so",
                                                  "libmfxplugin32_sw.so"};
#endif // (defined(WIN64))

#else // defined(_DEBUG)

#if defined(LINUX64)
const msdk_disp_char * defaultDLLName[2] = {"libmfxhw64_d.so",
                                            "libmfxsw64_d.so"};
const msdk_disp_char * defaultAudioDLLName[2] = {"libmfxaudiosw64_d.so",
                                            "libmfxaudiosw64_d.so"};

const msdk_disp_char * defaultPluginDLLName[2] = {"libmfxplugin64_hw_d.so",
                                                  "libmfxplugin64_sw_d.so"};
#elif defined(__APPLE__)
#ifdef __i386__
const msdk_disp_char * defaultDLLName[2] = {"libmfxhw32_d.dylib",
                                            "libmfxsw32_d.dylib"};
const msdk_disp_char * defaultAudioDLLName[2] = {"libmfxaudiosw32_d.dylib",
                                            "libmfxaudiosw32_d.dylib"};

const msdk_disp_char * defaultPluginDLLName[2] = {"libmfxplugin32_hw_d.dylib",
                                                  "libmfxplugin32_sw_d.dylib"};

#else
const msdk_disp_char * defaultDLLName[2] = {"libmfxhw64_d.dylib",
                                            "libmfxsw64_d.dylib"};
const msdk_disp_char * defaultAudioDLLName[2] = {"libmfxaudiosw64_d.dylib",
                                            "libmfxaudiosw64_d.dylib"};

const msdk_disp_char * defaultPluginDLLName[2] = {"libmfxplugin64_hw_d.dylib",
                                                  "libmfxplugin64_sw_d.dylib"};
#endif // #ifdef __i386__ for __APPLE__

#else // for Linux32 and Android
const msdk_disp_char * defaultDLLName[2] = {"libmfxhw32_d.so",
                                            "libmfxsw32_d.so"};
const msdk_disp_char * defaultAudioDLLName[2] = {"libmfxaudiosw32_d.so",
                                            "libmfxaudiosw32_d.so"};

const msdk_disp_char * defaultPluginDLLName[2] = {"libmfxplugin32_hw_d.so",
                                                  "libmfxplugin32_sw_d.so"};
#endif // (defined(WIN64))

#endif // !defined(_DEBUG)

namespace MFX
{

mfxStatus mfx_get_default_dll_name(msdk_disp_char *pPath, size_t /*pathSize*/, eMfxImplType implType)
{
    strcpy(pPath, defaultDLLName[implType & 1]);

    return MFX_ERR_NONE;

} // mfxStatus GetDefaultDLLName(wchar_t *pPath, size_t pathSize, eMfxImplType implType)


mfxStatus mfx_get_default_plugin_name(msdk_disp_char *pPath, size_t pathSize, eMfxImplType implType)
{
    strcpy(pPath, defaultPluginDLLName[implType & 1]);

    return MFX_ERR_NONE;
}


mfxStatus mfx_get_default_audio_dll_name(msdk_disp_char *pPath, size_t /*pathSize*/, eMfxImplType implType)
{
    strcpy(pPath, defaultAudioDLLName[implType & 1]);

    return MFX_ERR_NONE;

} // mfxStatus GetDefaultAudioDLLName(wchar_t *pPath, size_t pathSize, eMfxImplType implType)

mfxModuleHandle mfx_dll_load(const msdk_disp_char *pFileName)
{
    mfxModuleHandle hModule = (mfxModuleHandle) 0;

    // check error(s)
    if (NULL == pFileName)
    {
        return NULL;
    }
    // load the module
    hModule = dlopen(pFileName, RTLD_LOCAL|RTLD_NOW);

    return hModule;
} // mfxModuleHandle mfx_dll_load(const wchar_t *pFileName)

mfxFunctionPointer mfx_dll_get_addr(mfxModuleHandle handle, const char *pFunctionName)
{
    if (NULL == handle)
    {
        return NULL;
    }

    mfxFunctionPointer addr = (mfxFunctionPointer) dlsym(handle, pFunctionName);
    if (!addr)
    {
        return NULL;
    }

    return addr;
} // mfxFunctionPointer mfx_dll_get_addr(mfxModuleHandle handle, const char *pFunctionName)

bool mfx_dll_free(mfxModuleHandle handle)
{
    if (NULL == handle)
    {
        return true;
    }
    dlclose(handle);

    return true;
} // bool mfx_dll_free(mfxModuleHandle handle)

mfxModuleHandle mfx_get_dll_handle(const msdk_disp_char *pFileName) {
    return mfx_dll_load(pFileName);
}

} // namespace MFX

#endif // #if !defined(_WIN32) && !defined(_WIN64)
