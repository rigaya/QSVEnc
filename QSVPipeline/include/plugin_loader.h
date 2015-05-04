/*********************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2013-2014 Intel Corporation. All Rights Reserved.

**********************************************************************************/

#pragma once

#ifndef __PLUGIN_LOADER_H__
#define __PLUGIN_LOADER_H__

#include "vm/so_defs.h"
#include "sample_utils.h"
#include "plugin_utils.h"
//#include "mfx_plugin_module.h"
#include <iostream>
#include <iomanip> // for std::setfill, std::setw
#include <memory> // for std::auto_ptr

class MsdkSoModule
{
protected:
    msdk_so_handle m_module;
public:
    MsdkSoModule()
        : m_module(NULL)
    {
    }
    MsdkSoModule(const msdk_string & pluginName)
        : m_module(NULL)
    {
        m_module = msdk_so_load(pluginName.c_str());
        if (NULL == m_module)
        {
            MSDK_TRACE_ERROR(msdk_tstring(MSDK_CHAR("Failed to load shared module: ")) + pluginName);
        }
    }
    template <class T>
    T GetAddr(const std::string & fncName)
    {
        T pCreateFunc = reinterpret_cast<T>(msdk_so_get_addr(m_module, fncName.c_str()));
        if (NULL == pCreateFunc) {
            MSDK_TRACE_ERROR(msdk_tstring("Failed to get function addres: ") + fncName.c_str());
        }
        return pCreateFunc;
    }

    virtual ~MsdkSoModule()
    {
        if (m_module)
        {
            msdk_so_free(m_module);
            m_module = NULL;
        }
    }
};

/*
* Rationale: class to load+register any mediasdk plugin decoder/encoder/generic by given name
*/
class PluginLoader : public MFXPlugin
{
protected:
    mfxPluginType     ePluginType;

    mfxSession        m_session;
    mfxPluginUID      m_uid;

private:
    const msdk_char* msdkGetPluginName(const mfxPluginUID& guid)
    {
        if (AreGuidsEqual(guid, MFX_PLUGINID_HEVCD_SW))
            return MSDK_STRING("Intel (R) Media SDK plugin for HEVC DECODE");
        else if(AreGuidsEqual(guid, MFX_PLUGINID_HEVCD_HW))
            return MSDK_STRING("Intel (R) Media SDK HW plugin for HEVC DECODE");
        else if(AreGuidsEqual(guid, MFX_PLUGINID_HEVCE_SW))
            return MSDK_STRING("Intel (R) Media SDK plugin for HEVC ENCODE");
        else if(AreGuidsEqual(guid, MFX_PLUGINID_H264LA_HW))
            return MSDK_STRING("Intel (R) Media SDK plugin for LA ENC");
        else
            return MSDK_STRING("Unknown plugin");
    }

public:
    PluginLoader(mfxPluginType type, mfxSession session, const mfxPluginUID & uid, mfxU32 version, const mfxChar *pluginName, mfxU32 len)
        : m_session()
        , m_uid()
        , ePluginType(type)
    {
        len; pluginName;
        mfxStatus sts = MFX_ERR_NONE;
        msdk_stringstream strStream;

        MSDK_MEMCPY(&m_uid, &uid, sizeof(mfxPluginUID));
        for (size_t i = 0; i != sizeof(mfxPluginUID); i++)
        {
            strStream << MSDK_STRING("0x") << std::setfill(MSDK_CHAR('0')) << std::setw(2) << std::hex << (int)m_uid.Data[i];
            if (i != (sizeof(mfxPluginUID)-1)) strStream << MSDK_STRING(", ");
        }

        if ((ePluginType == MFX_PLUGINTYPE_AUDIO_DECODE) ||
            (ePluginType == MFX_PLUGINTYPE_AUDIO_ENCODE))
        {
            // Audio plugins are not loaded by path
            sts = MFX_ERR_UNSUPPORTED;
        }
        else
        {
            sts = MFXVideoUSER_LoadByPath(session, &m_uid, version, pluginName, len);
        }

        if (MFX_ERR_NONE != sts)
        {
            MSDK_TRACE_ERROR(MSDK_STRING("Failed to load plugin from GUID, sts=") << sts << MSDK_STRING(": { ") << strStream.str().c_str() << MSDK_STRING(" } (") << msdkGetPluginName(m_uid) << MSDK_STRING(")"));
        }
        else
        {
            MSDK_TRACE_INFO(MSDK_STRING("Plugin was loaded from GUID"));
            m_session = session;
        }
    }

    PluginLoader(mfxPluginType type, mfxSession session, const mfxPluginUID & uid, mfxU32 version)
        : m_session()
        , m_uid()
        , ePluginType(type)
    {
        mfxStatus sts = MFX_ERR_NONE;
        msdk_stringstream strStream;

        MSDK_MEMCPY(&m_uid, &uid, sizeof(mfxPluginUID));
        for (size_t i = 0; i != sizeof(mfxPluginUID); i++)
        {
            strStream << MSDK_STRING("0x") << std::setfill(MSDK_CHAR('0')) << std::setw(2) << std::hex << (int)m_uid.Data[i];
            if (i != (sizeof(mfxPluginUID)-1)) strStream << MSDK_STRING(", ");
        }

        if ((ePluginType == MFX_PLUGINTYPE_AUDIO_DECODE) ||
            (ePluginType == MFX_PLUGINTYPE_AUDIO_ENCODE))
        {
            sts = MFXAudioUSER_Load(session, &m_uid, version);
        }
        else
        {
            sts = MFXVideoUSER_Load(session, &m_uid, version);
        }

        if (MFX_ERR_NONE != sts)
        {
            MSDK_TRACE_ERROR(MSDK_STRING("Failed to load plugin from GUID, sts=") << sts << MSDK_STRING(": { ") << strStream.str().c_str() << MSDK_STRING(" } (") << msdkGetPluginName(m_uid) << MSDK_STRING(")"));
        }
        else
        {
            MSDK_TRACE_INFO(MSDK_STRING("Plugin was loaded from GUID"));
            m_session = session;
        }
    }

    virtual ~PluginLoader()
    {
        mfxStatus sts = MFX_ERR_NONE;
        if (m_session)
        {
            if ((ePluginType == MFX_PLUGINTYPE_AUDIO_DECODE) ||
                (ePluginType == MFX_PLUGINTYPE_AUDIO_ENCODE))
            {
                sts = MFXAudioUSER_UnLoad(m_session, &m_uid);
            }
            else
            {
                sts = MFXVideoUSER_UnLoad(m_session, &m_uid);
            }

            if (sts != MFX_ERR_NONE)
            {
                  MSDK_TRACE_ERROR(MSDK_STRING("Failed to unload plugin from GUID, sts=") << sts);
            }
            else
            {
                MSDK_TRACE_INFO(MSDK_STRING("MFXBaseUSER_UnLoad(session=0x") << m_session << MSDK_STRING("), sts=") << sts);
            }
        }
    }

    bool IsOk() {
        return m_session != 0;
    }
    virtual mfxStatus PluginInit( mfxCoreInterface *core ) {
        core;
        return MFX_ERR_NULL_PTR;
    }
    virtual mfxStatus PluginClose() {
        return MFX_ERR_NULL_PTR;
    }
    virtual mfxStatus GetPluginParam( mfxPluginParam *par ) {
        par;
        return MFX_ERR_NULL_PTR;
    }
    virtual mfxStatus Execute( mfxThreadTask task, mfxU32 uid_p, mfxU32 uid_a ) {
        task; uid_p; uid_a;
        return MFX_ERR_NULL_PTR;
    }
    virtual mfxStatus FreeResources( mfxThreadTask task, mfxStatus sts ) {
        task; sts;
        return MFX_ERR_NULL_PTR;
    }
    virtual void Release() {
    }
    virtual mfxStatus Close() {
        return MFX_ERR_NULL_PTR;
    }
    virtual mfxStatus SetAuxParams( void* auxParam, int auxParamSize ) {
        auxParam; auxParamSize;
        return MFX_ERR_NULL_PTR;
    }
};

inline MFXPlugin * LoadPluginByType(mfxPluginType type, mfxSession session, const mfxPluginUID & uid, mfxU32 version, const mfxChar *pluginName, mfxU32 len) {
    std::auto_ptr<PluginLoader> plg(new PluginLoader (type, session, uid, version, pluginName, len));
    return plg->IsOk() ? plg.release() : NULL;
}

inline MFXPlugin * LoadPluginByGUID(mfxPluginType type, mfxSession session, const mfxPluginUID & uid, mfxU32 version) {
    std::auto_ptr<PluginLoader> plg(new PluginLoader (type, session, uid, version));
    return plg->IsOk() ? plg.release() : NULL;
}

inline MFXPlugin * LoadPlugin(mfxPluginType type, mfxSession session, const mfxPluginUID & uid, mfxU32 version, const mfxChar *pluginName, mfxU32 len) {
    return LoadPluginByType(type, session, uid, version, pluginName, len);
}

inline MFXPlugin * LoadPlugin(mfxPluginType type, mfxSession session, const mfxPluginUID & uid, mfxU32 version) {
    return LoadPluginByGUID(type, session, uid, version);
}
#endif // PLUGIN_LOADER