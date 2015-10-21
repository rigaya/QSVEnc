/* ****************************************************************************** *\

Copyright (C) 2013-2014 Intel Corporation.  All rights reserved.

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

File Name: mfx_plugin_cfg_parser.h

\* ****************************************************************************** */

#if !defined(__MFX_PLUGIN_CFG_PARSER_H)
#define __MFX_PLUGIN_CFG_PARSER_H

#include "mfx_dispatcher_defs.h"
#include "mfxplugin.h"
#include "mfx_vector.h"
#include "mfx_plugin_hive.h"
#include <string.h>
#include <memory>
#include <stdio.h>

#pragma once

namespace MFX
{
    class PluginConfigParser
    {
    public:

        enum 
        {
            PARSED_TYPE        = 1,
            PARSED_CODEC_ID    = 2,
            PARSED_UID         = 4,
            PARSED_PATH        = 8,
            PARSED_DEFAULT     = 16,
            PARSED_VERSION     = 32,
            PARSED_API_VERSION = 64,
            PARSED_NAME        = 128,
        };

        explicit PluginConfigParser(const char * name);
        ~PluginConfigParser();

        // Returns current section name if any
        bool GetCurrentPluginName(char * pluginName, int nChars);

        template <size_t N>
        bool GetCurrentPluginName(char (& pluginName)[N])
        {
            return this->GetCurrentPluginName(pluginName, N);
        }

        // Tries to advance to the next section in config file
        bool AdvanceToNextPlugin();
        // Return to first line of the file
        bool Rewind();
        // Enumerates sections in currect file (no section headers - 1 section)
        int GetPluginCount();
        // Parses plugin parameters from current section
        bool ParsePluginParams(PluginDescriptionRecord & dst, mfxU32 & parsedFields);

    private:
        FILE * cfgFile;
        fpos_t sectionStart;

        bool ParseSingleParameter(const char * name, char * value, PluginDescriptionRecord & dst, mfxU32 & parsedFields);
    };

    bool parseGUID(const char* src, mfxU8* guid);
}

#endif // __MFX_PLUGIN_CFG_PARSER_H