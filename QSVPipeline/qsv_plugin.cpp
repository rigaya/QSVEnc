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
#include "rgy_util.h"
#include <mfxvp8.h>

static const auto MFX_COMPONENT_TYPE_TO_PLUGIN_TYPE = make_array<std::pair<MFXComponentType, mfxPluginType>>(
    std::make_pair(MFXComponentType::DECODE, MFX_PLUGINTYPE_VIDEO_DECODE),
    std::make_pair(MFXComponentType::ENCODE, MFX_PLUGINTYPE_VIDEO_ENCODE),
    std::make_pair(MFXComponentType::VPP, MFX_PLUGINTYPE_VIDEO_VPP),
    std::make_pair(MFXComponentType::ENC, MFX_PLUGINTYPE_VIDEO_ENC)
    //std::make_pair(MFXComponentType::FEI, )
);

MAP_PAIR_0_1(component, type, MFXComponentType, plugin, mfxPluginType, MFX_COMPONENT_TYPE_TO_PLUGIN_TYPE, MFXComponentType::UNKNOWN, (mfxPluginType)-1);

const mfxPluginUID *getMFXPluginUID(MFXComponentType type, uint32_t codecID, const bool software) {
    switch (type) {
    case MFXComponentType::DECODE:
        switch (codecID) {
        case MFX_CODEC_HEVC:
            return (software) ? &MFX_PLUGINID_HEVCD_SW : &MFX_PLUGINID_HEVCD_HW;
        case MFX_CODEC_VP8:
            return (software) ? nullptr : &MFX_PLUGINID_VP8D_HW;
        case MFX_CODEC_VP9:
            return (software) ? nullptr : &MFX_PLUGINID_VP9D_HW;
        }
        break;
    case MFXComponentType::ENCODE:
        switch (codecID) {
        case MFX_CODEC_HEVC:
            return (software) ? &MFX_PLUGINID_HEVCE_SW : &MFX_PLUGINID_HEVCE_HW;
        case MFX_CODEC_VP8:
            return (software) ? nullptr : &MFX_PLUGINID_VP8E_HW;
        case MFX_CODEC_VP9:
            return (software) ? nullptr : &MFX_PLUGINID_VP9E_HW;
        }
        break;
        //case (MFXComponentType::ENCODE | MFXComponentType::FEI):
        //    switch (codecID) {
        //    case MFX_CODEC_HEVC:
        //        return MFX_PLUGINID_HEVC_FEI_ENCODE;
        //    }
        //    break;
    case MFXComponentType::ENC:
        switch (codecID) {
        case MFX_CODEC_HEVC:
            return &MFX_PLUGINID_HEVCE_FEI_HW; // HEVC FEI uses ENC interface
        }
        break;
    }
    return nullptr;
}

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
mfxStatus CSessionPlugins::LoadPlugin(MFXComponentType type, uint32_t codecID, const bool software) {
    auto plugin = getMFXPluginUID(type, codecID, software);
    auto plugintype = component_type_to_plugin(type);
    if (plugin == nullptr || plugintype < 0) {
        return MFX_ERR_NONE;
    }
    return LoadPlugin(plugintype, *plugin, 1);
}
void CSessionPlugins::UnloadPlugins() {
    m_plugins.clear();
}
