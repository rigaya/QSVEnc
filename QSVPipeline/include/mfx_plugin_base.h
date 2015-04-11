/* ****************************************************************************** *\

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2013 Intel Corporation. All Rights Reserved.

\* ****************************************************************************** */

#pragma once


#include <mfxplugin++.h>

typedef MFXDecoderPlugin* (*mfxCreateDecoderPlugin)();
typedef MFXEncoderPlugin* (*mfxCreateEncoderPlugin)();
typedef MFXGenericPlugin* (*mfxCreateGenericPlugin)();

class PluginFactory
{
    static MFXDecoderPlugin* CreateDecoderPlugin() {
        return PluginFactory::CreateDecoderPlugin();
    }
    static  MFXEncoderPlugin* CreateEncoderPlugin() {
        return PluginFactory::CreateEncoderPlugin();
    }
    static  MFXGenericPlugin* CreateGenericPlugin() {
        return PluginFactory::CreateGenericPlugin();
    }
};