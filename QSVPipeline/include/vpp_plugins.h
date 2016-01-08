//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------
#pragma once
#ifndef __VPP_PLUGINS_H__
#define __VPP_PLUGINS_H__

#include <memory>
#include "sample_utils.h"
#include "vm/so_defs.h"
#include "mfx_plugin_base.h"
#include "mfx_plugin_module.h"
#include "sysmem_allocator.h"
#if D3D_SURFACES_SUPPORT
#include "d3d_device.h"
#include "d3d_allocator.h"
#if MFX_D3D11_SUPPORT
#include "d3d11_device.h"
#include "d3d11_allocator.h"
#endif
#endif

#include "rotate/plugin_rotate.h"
#include "delogo/plugin_delogo.h"

#define GPU_FILTER 0

using std::unique_ptr;

class CVPPPlugin
{
public:
    CVPPPlugin() {
        m_pPluginSurfaces = nullptr;
        m_nSurfNum = 0;
        m_memType = SYSTEM_MEMORY;
        m_bPluginFlushed = false;
        m_pMFXAllocator = nullptr;
        m_pmfxAllocatorParams = nullptr;
        m_pPluginSurfaces = nullptr;
#if GPU_FILTER
        m_hwdev.reset();
#endif
        QSV_MEMSET_ZERO(m_PluginResponse);
        QSV_MEMSET_ZERO(m_pluginVideoParams);
    };
public:
    ~CVPPPlugin() {
        Close();
    };
public:
    virtual void Close() {
        tstring pluginName = _T("");
        if (m_pUsrPlugin.get() != nullptr) {
            pluginName = m_pUsrPlugin->GetPluginName().c_str();
        }
        if (m_mfxSession) {
            MFXDisjoinSession(m_mfxSession);
            MFXVideoUSER_Unregister(m_mfxSession, 0);
            (*m_pQSVLog)(QSV_LOG_DEBUG, _T("CVPPPluginClose[%s]: unregistered plugin.\n"), pluginName.c_str());
            m_mfxSession.Close();
            (*m_pQSVLog)(QSV_LOG_DEBUG, _T("CVPPPluginClose[%s]: closed session.\n"), pluginName.c_str());
        }

        m_pUsrPlugin.reset();
        m_pPluginSurfaces.reset();
        //qsv_delete(m_pMFXAllocator);
        //qsv_delete(m_pmfxAllocatorParams);
        m_bPluginFlushed = false;
        m_nSurfNum = 0;
        m_memType = SYSTEM_MEMORY;
#if GPU_FILTER
        m_hwdev.reset();
#endif
        m_message.clear();
        (*m_pQSVLog)(QSV_LOG_DEBUG, _T("CVPPPluginClose[%s]: closed.\n"), pluginName.c_str());
        m_pQSVLog.reset();
    };

public:
    virtual mfxStatus AllocSurfaces(MFXFrameAllocator *pMFXAllocator, bool m_bExternalAlloc) {
        //pMFXAllocator = m_pMFXAllocator;
        if (m_PluginRequest.NumFrameSuggested == 0) {
            return MFX_ERR_NOT_INITIALIZED;
        }

        mfxStatus sts = pMFXAllocator->Alloc(pMFXAllocator->pthis, &m_PluginRequest, &m_PluginResponse);
        if (MFX_ERR_NONE != sts) {
            (*m_pQSVLog)(QSV_LOG_ERROR, _T("CVPPPluginAlloc[%s]: failed to allocate surface.\n"), m_pUsrPlugin->GetPluginName().c_str());
            return sts;
        }
        (*m_pQSVLog)(QSV_LOG_DEBUG, _T("CVPPPluginAlloc[%s]: allocated %d surfaces\n"), m_pUsrPlugin->GetPluginName().c_str(), m_PluginResponse.NumFrameActual);

        m_pPluginSurfaces.reset(new mfxFrameSurface1 [m_PluginResponse.NumFrameActual]);
        MSDK_CHECK_POINTER(m_pPluginSurfaces, MFX_ERR_MEMORY_ALLOC);

        for (int i = 0; i < m_PluginResponse.NumFrameActual; i++) {
            QSV_MEMSET_ZERO(m_pPluginSurfaces[i]);
            MSDK_MEMCPY_VAR(m_pPluginSurfaces[i].Info, &(m_pluginVideoParams.mfx.FrameInfo), sizeof(mfxFrameInfo));

            if (m_bExternalAlloc) {
                m_pPluginSurfaces[i].Data.MemId = m_PluginResponse.mids[i];
            } else {
                sts = pMFXAllocator->Lock(pMFXAllocator->pthis, m_PluginResponse.mids[i], &(m_pPluginSurfaces[i].Data));
                if (MFX_ERR_NONE != sts)
                    return sts;
            }
        }
        return MFX_ERR_NONE;
    }

public:
    virtual mfxStatus Init(mfxVersion ver, const TCHAR *pluginName, void *pPluginParam, mfxU32 nPluginParamSize,
        bool useHWLib, MemType memType, shared_ptr<CHWDevice> phwdev, MFXFrameAllocator* pAllocator,
        mfxU16 nAsyncDepth, const mfxFrameInfo& frameIn, mfxU16 IOPattern, shared_ptr<CQSVLog> pQSVLog) {

        MSDK_CHECK_POINTER(pluginName, MFX_ERR_NULL_PTR);
        MSDK_CHECK_POINTER(pPluginParam, MFX_ERR_NULL_PTR);
        
#if GPU_FILTER
        m_hwdev = phwdev;
#endif
        m_pQSVLog = pQSVLog;
        mfxStatus sts = InitSession(useHWLib, memType);
        if (sts != MFX_ERR_NONE) {
            (*m_pQSVLog)(QSV_LOG_ERROR, _T("CVPPPluginInit: failed to init session for plugin.\n"));
            return sts;
        }
        (*m_pQSVLog)(QSV_LOG_DEBUG, _T("CVPPPluginInit: initialized session for plugin.\n"));

        m_mfxSession.SetFrameAllocator(pAllocator);
        
        if (0 == _tcsicmp(pluginName, _T("rotate"))) {
            m_pUsrPlugin.reset(new Rotate());
        } else if (0 == _tcsicmp(pluginName, _T("delogo"))) {
            m_pUsrPlugin.reset(new Delogo());
        }
        if (m_pUsrPlugin.get() == nullptr) {
            (*m_pQSVLog)(QSV_LOG_ERROR, _T("CVPPPluginInit: plugin name \"%s\" could not be found."), pluginName);
            return MFX_ERR_NOT_FOUND;
        }

        InitMfxPluginParam(nAsyncDepth, frameIn, IOPattern);

        m_pUsrPlugin->SetMfxVer(ver);

        // register plugin callbacks in Media SDK
        mfxPlugin plg = make_mfx_plugin_adapter((MFXGenericPlugin*)m_pUsrPlugin.get());
        if (MFX_ERR_NONE != (sts = MFXVideoUSER_Register(m_mfxSession, 0, &plg))) {
            (*m_pQSVLog)(QSV_LOG_ERROR, _T("CVPPPluginInit: %s: failed to register plugin.\n"), m_pUsrPlugin->GetPluginName().c_str());
            return sts;
        }
        (*m_pQSVLog)(QSV_LOG_DEBUG, _T("CVPPPluginInit: registered plugin to plugin session.\n"));

        if (sts == MFX_ERR_NONE) sts = m_pUsrPlugin->Init(&m_pluginVideoParams);
        if (sts == MFX_ERR_NONE) sts = m_pUsrPlugin->SetAuxParams(pPluginParam, nPluginParamSize);
        m_message = strsprintf(_T("%s, %s\n"), m_pUsrPlugin->GetPluginName().c_str(), m_pUsrPlugin->GetPluginMessage().c_str());
        (*m_pQSVLog)(QSV_LOG_DEBUG, _T("CVPPPluginInit: %s\n"), m_message.c_str());
        return sts;
    }
public:
    virtual tstring getMessage() {
        return m_message;
    }
public:
    virtual tstring getFilterName() {
        if (m_pUsrPlugin.get() == nullptr) {
            return _T("");
        }
        return m_pUsrPlugin->GetPluginName();
    }
public:
    virtual mfxSession getSession() {
        return m_mfxSession;
    }

private:
    virtual mfxStatus InitSession(bool useHWLib, MemType memType) {
        mfxStatus sts = MFX_ERR_NONE;
        // init session, and set memory type
        mfxIMPL impl = 0;
        mfxVersion verRequired = MFX_LIB_VERSION_1_1;
        m_mfxSession.Close();
        if (useHWLib) {
            // try searching on all display adapters
            impl = MFX_IMPL_HARDWARE_ANY;
            m_memType = memType;
            if (memType & D3D9_MEMORY)
                impl |= MFX_IMPL_VIA_D3D9;
#if MFX_D3D11_SUPPORT
            else if (memType & D3D11_MEMORY)
                impl |= MFX_IMPL_VIA_D3D11;
#endif
            sts = m_mfxSession.Init(impl, &verRequired);

            // MSDK API version may not support multiple adapters - then try initialize on the default
            if (MFX_ERR_NONE != sts)
                sts = m_mfxSession.Init((impl & (~MFX_IMPL_HARDWARE_ANY)) | MFX_IMPL_HARDWARE, &verRequired);

            if (MFX_ERR_NONE == sts)
                return sts;
        } else {
            impl = MFX_IMPL_SOFTWARE;
            sts = m_mfxSession.Init(impl, &verRequired);
            m_memType = SYSTEM_MEMORY;
        }
        //使用できる最大のversionをチェック
        m_mfxVer = get_mfx_lib_version(impl);
        return sts;
    }
    
private:
    virtual mfxStatus InitMfxPluginParam(mfxU16 nAsyncDepth, const mfxFrameInfo& frameIn, mfxU16 IOPattern) {
        QSV_MEMSET_ZERO(m_pluginVideoParams);

        m_pluginVideoParams.AsyncDepth = nAsyncDepth;
        memcpy(&m_pluginVideoParams.vpp.In,  &frameIn, sizeof(frameIn));
        memcpy(&m_pluginVideoParams.vpp.Out, &frameIn, sizeof(frameIn));
        m_pluginVideoParams.IOPattern = IOPattern;
        return MFX_ERR_NONE;
    }
public:
    mfxVideoParam                  m_pluginVideoParams;   //カスタムVPP用の入出力パラメータ
    mfxU16                         m_nSurfNum;            //保持しているSurfaceの枚数
    mfxFrameAllocRequest           m_PluginRequest;       //AllocatorへのRequest
    mfxFrameAllocResponse          m_PluginResponse;      //AllocatorからのResponse
    unique_ptr<mfxFrameSurface1[]> m_pPluginSurfaces;     //保持しているSurface配列へのポインタ
    bool                           m_bPluginFlushed;      //使用していない
    MFXFrameAllocator             *m_pMFXAllocator;       //使用していない
private:
    mfxAllocatorParams            *m_pmfxAllocatorParams; //使用していない
    MFXVideoSession                m_mfxSession;          //カスタムVPP用のSession メインSessionにJoinして使用する
    MemType                        m_memType;             //パイプラインのSurfaceのメモリType
    mfxVersion                     m_mfxVer;              //使用しているMediaSDKのバージョン
    unique_ptr<QSVEncPlugin>       m_pUsrPlugin;          //カスタムプラグインのインスタンス
#if GPU_FILTER
    shared_ptr<CHWDevice>          m_hwdev;               //使用しているデバイス
#endif
    tstring                        m_message;             //このカスタムVPPからのメッセージ
    shared_ptr<CQSVLog>            m_pQSVLog;            //ログ出力用関数オブジェクト
};

#endif //__VPP_PLUGINS_H__

