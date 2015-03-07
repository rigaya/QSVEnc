/*********************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2014 Intel Corporation. All Rights Reserved.

**********************************************************************************/

#ifndef __SAMPLE_PARAMS_H__
#define __SAMPLE_PARAMS_H__

#include "sample_defs.h"
#include "plugin_utils.h"

struct sPluginParams
{
    mfxPluginUID      pluginGuid;
    mfxChar           strPluginPath[MSDK_MAX_FILENAME_LEN];
    MfxPluginLoadType type;
    sPluginParams()
    {
        MSDK_ZERO_MEMORY(*this);
    }
};

#endif //__SAMPLE_PARAMS_H__
