/*********************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2014 Intel Corporation. All Rights Reserved.

**********************************************************************************/

#include "mfx_samples_config.h"

#include <stdio.h>
#include <string.h>

#include "sample_defs.h"
#include "vm/strings_defs.h"
#include "mfx_plugin_uids.h"

static msdkPluginDesc g_msdk_supported_plugins[] = {
    //Supported Decoder Plugins
    { MSDK_VDEC | MSDK_IMPL_SW, MFX_CODEC_HEVC,  { g_msdk_hevcd_sw_uid, {0} }, MSDK_STRING("Intel (R) Media SDK plugin for HEVC DECODE") },
    { MSDK_VDEC | MSDK_IMPL_USR, MFX_CODEC_HEVC, { MSDK_NULL_GUID, {0} },   MSDK_STRING("User defined HEVC Plug-in") },
    { MSDK_VDEC | MSDK_IMPL_USR, MFX_CODEC_AVC,  { MSDK_NULL_GUID, {0} },   MSDK_STRING("User defined AVC Plug-in") },
    { MSDK_VDEC | MSDK_IMPL_USR, MFX_CODEC_MPEG2,{ MSDK_NULL_GUID, {0} },   MSDK_STRING("User defined MPEG2 Plug-in") },
    { MSDK_VDEC | MSDK_IMPL_USR, MFX_CODEC_VC1,  { MSDK_NULL_GUID, {0} },   MSDK_STRING("User defined VC1 Plug-in") },
    //Supported Encoder Plugins
    { MSDK_VENC | MSDK_IMPL_SW, MFX_CODEC_HEVC,  { g_msdk_hevce_uid, {0} }, MSDK_STRING("Intel (R) Media SDK plugin for HEVC ENCODE") },
    { MSDK_VENC | MSDK_IMPL_USR, MFX_CODEC_HEVC, { MSDK_NULL_GUID, {0} },   MSDK_STRING("User defined HEVC Plug-in") },
    { MSDK_VENC | MSDK_IMPL_USR, MFX_CODEC_AVC,  { MSDK_NULL_GUID, {0} },   MSDK_STRING("User defined AVC Plug-in") },
    { MSDK_VENC | MSDK_IMPL_USR, MFX_CODEC_MPEG2,{ MSDK_NULL_GUID, {0} },   MSDK_STRING("User defined MPEG2 Plug-in") },
    { MSDK_VENC | MSDK_IMPL_USR, MFX_CODEC_VC1,  { MSDK_NULL_GUID, {0} },   MSDK_STRING("User defined VC1 Plug-in") },
    //{ MSDK_VDEC | MSDK_IMPL_USR, MFX_CODEC_VP8,  { MSDK_NULL_GUID, {0} }, "User defined VP8 Plug-in" },
};

const msdkPluginUID* msdkGetPluginUID(mfxU32 type, mfxU32 codecid)
{
    mfxU32 i;

    for (i = 0; i < sizeof(g_msdk_supported_plugins)/sizeof(g_msdk_supported_plugins[0]); ++i) {
        if ((type == g_msdk_supported_plugins[i].type) && (codecid == g_msdk_supported_plugins[i].codecid)) {
            return (memcmp(&g_msdk_supported_plugins[i].uid, &MSDK_NULL_GUID, sizeof(MSDK_NULL_GUID)))?
                &(g_msdk_supported_plugins[i].uid):
                NULL;
        }
    }
    return NULL;
}

const msdk_char* msdkGetPluginName(const msdkPluginUID* uid)
{
    mfxU32 i;

    for (i = 0; i < sizeof(g_msdk_supported_plugins)/sizeof(g_msdk_supported_plugins[0]); ++i) {
        if (!memcmp(&g_msdk_supported_plugins[i].uid, uid, sizeof(MSDK_NULL_GUID))) {
            return g_msdk_supported_plugins[i].name;
        }
    }
    return NULL;
}

mfxStatus msdkSetPluginPath(mfxU32 type, mfxU32 codecid, msdk_char path[MSDK_MAX_FILENAME_LEN])
{
    mfxU32 i;

    for (i = 0; i < sizeof(g_msdk_supported_plugins)/sizeof(g_msdk_supported_plugins[0]); ++i) {
        if (g_msdk_supported_plugins[i].type == type && g_msdk_supported_plugins[i].codecid == codecid) {
            msdk_strcopy(g_msdk_supported_plugins[i].path, path);
            return MFX_ERR_NONE;
        }
    }
    return MFX_ERR_NOT_FOUND;
}

const msdk_char* msdkGetPluginPath(mfxU32 type, mfxU32 codecid)
{
    mfxU32 i;

    for (i = 0; i < sizeof(g_msdk_supported_plugins)/sizeof(g_msdk_supported_plugins[0]); ++i) {
        if (g_msdk_supported_plugins[i].type == type && g_msdk_supported_plugins[i].codecid == codecid) {
            return (g_msdk_supported_plugins[i].path[0])? g_msdk_supported_plugins[i].path: NULL;
        }
    }
    return NULL;
}

mfxStatus LoadPluginByUID(mfxSession* session, const msdkPluginUID* uid)
{
    mfxStatus sts = MFX_ERR_NONE;
    const msdk_char *pluginName = NULL;

    if (!session || !uid) return MFX_ERR_NULL_PTR;

    sts = MFXVideoUSER_Load(*session, &(uid->mfx), 1);

    pluginName = msdkGetPluginName(uid);
    if (MFX_ERR_NONE != sts) {
        msdk_printf(MSDK_STRING("error: failed to load Media SDK plugin:\n"));
        msdk_printf(MSDK_STRING("error:   GUID = { 0x%08x, 0x%04x, 0x%04x, { 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x } }\n"),
                uid->guid.data1, uid->guid.data2, uid->guid.data3,
                uid->guid.data4[0], uid->guid.data4[1], uid->guid.data4[2], uid->guid.data4[3],
                uid->guid.data4[4], uid->guid.data4[5], uid->guid.data4[6], uid->guid.data4[7]);
        msdk_printf(MSDK_STRING("error:   UID (mfx raw) = %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x\n"),
                uid->raw[0],  uid->raw[1],  uid->raw[2],  uid->raw[3],
                uid->raw[4],  uid->raw[5],  uid->raw[6],  uid->raw[7],
                uid->raw[8],  uid->raw[9],  uid->raw[10], uid->raw[11],
                uid->raw[12], uid->raw[13], uid->raw[14], uid->raw[15]);
        if (pluginName)
            msdk_printf(MSDK_STRING("error:   name = %s\n"), pluginName);
        msdk_printf(MSDK_STRING("error:   You may need to install this plugin separately!\n"));
    }
    else {
        if (pluginName)
            msdk_printf(MSDK_STRING("info: plugin '%s' loaded successfully\n"), pluginName);
    }
    return sts;
}
