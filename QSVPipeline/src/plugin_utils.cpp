/*********************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2014 Intel Corporation. All Rights Reserved.

**********************************************************************************/

#include "mfx_samples_config.h"

#include "plugin_utils.h"

bool AreGuidsEqual(const mfxPluginUID& guid1, const mfxPluginUID& guid2)
{
    for(size_t i = 0; i != sizeof(mfxPluginUID); i++)
    {
        if (guid1.Data[i] != guid2.Data[i])
            return false;
    }
    return true;
}

mfxStatus ConvertStringToGuid(const msdk_string & strGuid, mfxPluginUID & mfxGuid)
{
    mfxStatus sts = MFX_ERR_NONE;
    mfxU32 hex = 0;
    for(size_t i = 0; i != sizeof(mfxPluginUID); i++)
    {
        hex = 0;

#if defined(_WIN32) || defined(_WIN64)
        if (1 != _stscanf_s(strGuid.c_str() + 2*i, MSDK_STRING("%2x"), &hex))
#else
        if (1 != sscanf(strGuid.c_str() + 2*i, MSDK_STRING("%2x"), &hex))
#endif
        {
            sts = MFX_ERR_UNKNOWN;
            break;
        }
        if (hex == 0 && (const msdk_char *)strGuid.c_str() + 2*i != msdk_strstr((const msdk_char *)strGuid.c_str() + 2*i,  MSDK_STRING("00")))
        {
            sts = MFX_ERR_UNKNOWN;
            break;
        }
        mfxGuid.Data[i] = (mfxU8)hex;
    }
    if (sts != MFX_ERR_NONE)
        MSDK_ZERO_MEMORY(mfxGuid);
    return sts;
}

const mfxPluginUID & msdkGetPluginUID(mfxIMPL impl, msdkComponentType type, mfxU32 uCodecid)
{
    if (impl == MFX_IMPL_SOFTWARE)
    {
        switch(type)
        {
        case MSDK_VDECODE:
            switch(uCodecid)
            {
            case MFX_CODEC_HEVC:
                return MFX_PLUGINID_HEVCD_SW;
            }
            break;
        case MSDK_VENCODE:
            switch(uCodecid)
            {
            case MFX_CODEC_HEVC:
                return MFX_PLUGINID_HEVCE_SW;
            }
            break;
        }
    }
    else if (impl |= MFX_IMPL_HARDWARE)
    {
        switch(type)
        {
        case MSDK_VDECODE:
            switch(uCodecid)
            {
            case MFX_CODEC_HEVC:
                return MFX_PLUGINID_HEVCD_SW; // MFX_PLUGINID_HEVCD_SW for now
            }
            break;
        case MSDK_VENCODE:
            switch(uCodecid)
            {
            case MFX_CODEC_HEVC:
                return MFX_PLUGINID_HEVCE_SW; // MFX_PLUGINID_HEVCD_SW for now
            }
            break;
        case MSDK_VENC:
            switch(uCodecid)
            {
            case MFX_CODEC_HEVC:
                return MFX_PLUGINID_HEVCE_FEI_HW;   // HEVC FEI uses ENC interface
            }
            break;
        }
    }

    return MSDK_PLUGINGUID_NULL;
}
