// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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
// --------------------------------------------------------------------------------------------

#ifndef __QSV_PLUGIN_H__
#define __QSV_PLUGIN_H__

#include <memory>
#include <vector>
#include "mfxplugin++.h"

enum class MFXComponentType {
    UNKNOWN = 0x0000,
    DECODE = 0x0001,
    ENCODE = 0x0002,
    VPP = 0x0004,
    ENC = 0x0008,
    FEI = 0x1000,
};
const mfxPluginUID *getMFXPluginUID(MFXComponentType type, uint32_t codecID, const bool software);

class CMFXPlugin {
public:
    CMFXPlugin(mfxSession session);
    ~CMFXPlugin();
    mfxStatus LoadPlugin(mfxPluginType type, const mfxPluginUID &uid, mfxU32 version);
    void Unload();
protected:
    mfxPluginType m_type;
    mfxPluginUID m_uid;
    mfxSession m_session;
    mfxStatus m_status;
};

class CSessionPlugins {
public:
    CSessionPlugins(mfxSession session);
    ~CSessionPlugins();
    mfxStatus LoadPlugin(mfxPluginType type, const mfxPluginUID &uid, mfxU32 version);
    mfxStatus LoadPlugin(MFXComponentType type, uint32_t codecID, const bool software);
    void UnloadPlugins();
protected:
    mfxSession m_session;
    std::vector<std::unique_ptr<CMFXPlugin>> m_plugins;
};

#endif //__QSV_PLUGIN_H__

