//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

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
