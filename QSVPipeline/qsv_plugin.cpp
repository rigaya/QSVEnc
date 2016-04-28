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

#include "qsv_plugin.h"

CMFXPlugin::CMFXPlugin(mfxSession session) :
    m_type(MFX_PLUGINTYPE_VIDEO_GENERAL),
    m_uid(),
    m_session(session),
    m_status(MFX_ERR_NONE) {

};

CMFXPlugin::~CMFXPlugin() {
    Unload();
}

mfxStatus CMFXPlugin::LoadPlugin(mfxPluginType type, const mfxPluginUID &uid, mfxU32 version) {
    m_type = type;
    m_uid = uid;
    if (   m_type == MFX_PLUGINTYPE_AUDIO_DECODE
        || m_type == MFX_PLUGINTYPE_AUDIO_ENCODE) {
        m_status = MFXAudioUSER_Load(m_session, &m_uid, version);
    } else {
        m_status = MFXVideoUSER_Load(m_session, &m_uid, version);
    }
    return m_status;
}

void CMFXPlugin::Unload() {
    if (m_status == MFX_ERR_NONE) {
        if (   m_type == MFX_PLUGINTYPE_AUDIO_DECODE
            || m_type == MFX_PLUGINTYPE_AUDIO_ENCODE) {
            m_status = MFXAudioUSER_UnLoad(m_session, &m_uid);
        } else {
            m_status = MFXAudioUSER_UnLoad(m_session, &m_uid);
        }
    }
}


CSessionPlugins::CSessionPlugins(mfxSession session) : m_session(session), m_plugins() {
}
CSessionPlugins::~CSessionPlugins() {
    UnloadPlugins();
}
mfxStatus CSessionPlugins::LoadPlugin(mfxPluginType type, const mfxPluginUID &uid, mfxU32 version) {
    auto plugin = std::unique_ptr<CMFXPlugin>(new CMFXPlugin(m_session));
    mfxStatus sts = plugin->LoadPlugin(type, uid, version);
    if (sts == MFX_ERR_NONE) {
        m_plugins.push_back(std::move(plugin));
    }
    return sts;
}
void CSessionPlugins::UnloadPlugins() {
    m_plugins.clear();
}
