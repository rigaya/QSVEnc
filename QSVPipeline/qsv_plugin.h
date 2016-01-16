//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef __QSV_PLUGIN_H__
#define __QSV_PLUGIN_H__

#include <memory>
#include <vector>
#include "mfxplugin++.h"

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
    void UnloadPlugins();
protected:
    mfxSession m_session;
    std::vector<std::unique_ptr<CMFXPlugin>> m_plugins;
};

#endif //__QSV_PLUGIN_H__

