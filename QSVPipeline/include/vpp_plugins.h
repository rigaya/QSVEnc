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
#include "plugin_rotate.h"

using std::unique_ptr;

class CVPPPlugin
{
public:
	CVPPPlugin() {
		m_PluginModule = NULL;
		m_pPluginSurfaces = NULL;
		m_nSurfNum = 0;
		m_bPluginFlushed = false;
		m_pMFXAllocator = NULL;
		m_pmfxAllocatorParams = NULL;
		m_hwdev = NULL;
		MSDK_ZERO_MEMORY(m_PluginResponse);
		MSDK_ZERO_MEMORY(m_pluginVideoParams);
	};
public:
	~CVPPPlugin() {
		Close();
	};
public:
	virtual void Close() {
		MFXDisjoinSession(m_mfxSession);
		MFXVideoUSER_Unregister(m_mfxSession, 0);

		m_pUsrPlugin.reset();
		MSDK_SAFE_DELETE_ARRAY(m_pPluginSurfaces);
		m_mfxSession.Close();
		MSDK_SAFE_DELETE(m_pMFXAllocator);
		MSDK_SAFE_DELETE(m_pmfxAllocatorParams);
		m_bPluginFlushed = false;
		m_nSurfNum = 0;
	};
	
private:
	virtual mfxStatus CreateHWDevice() {
		mfxStatus sts = MFX_ERR_NONE;
#if D3D_SURFACES_SUPPORT
		POINT point = { 0, 0 };
		HWND window = WindowFromPoint(point);

#if MFX_D3D11_SUPPORT
		if (D3D11_MEMORY == m_memType)
			m_hwdev = new CD3D11Device();
		else
#endif // #if MFX_D3D11_SUPPORT
			m_hwdev = new CD3D9Device();

		if (NULL == m_hwdev)
			return MFX_ERR_MEMORY_ALLOC;

		sts = m_hwdev->Init(
			window,
			0,
			MSDKAdapter::GetNumber(m_mfxSession));
		MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
#endif //D3D_SURFACES_SUPPORT
		return MFX_ERR_NONE;
	}

private:
	virtual mfxStatus CreateAllocator() {
		mfxStatus sts = MFX_ERR_NONE;

		if (D3D9_MEMORY == m_memType || D3D11_MEMORY == m_memType) {
#if D3D_SURFACES_SUPPORT
			//sts = CreateHWDevice();
			MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

			mfxHDL hdl = NULL;
			mfxHandleType hdl_t =
#if MFX_D3D11_SUPPORT
				D3D11_MEMORY == m_memType ? MFX_HANDLE_D3D11_DEVICE :
#endif // #if MFX_D3D11_SUPPORT
				MFX_HANDLE_D3D9_DEVICE_MANAGER;

			sts = m_hwdev->GetHandle(hdl_t, &hdl);
			MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

			// handle is needed for HW library only
			mfxIMPL impl = 0;
			m_mfxSession.QueryIMPL(&impl);
			if (impl != MFX_IMPL_SOFTWARE) {
				sts = m_mfxSession.SetHandle(hdl_t, hdl);
				MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
			}
#endif //D3D_SURFACES_SUPPORT

			// create D3D allocator
#if MFX_D3D11_SUPPORT
			if (D3D11_MEMORY == m_memType) {
				m_pMFXAllocator = new D3D11FrameAllocator;
				MSDK_CHECK_POINTER(m_pMFXAllocator, MFX_ERR_MEMORY_ALLOC);

				D3D11AllocatorParams *pd3dAllocParams = new D3D11AllocatorParams;
				MSDK_CHECK_POINTER(pd3dAllocParams, MFX_ERR_MEMORY_ALLOC);
				pd3dAllocParams->pDevice = reinterpret_cast<ID3D11Device *>(hdl);

				m_pmfxAllocatorParams = pd3dAllocParams;
			} else
#endif // #if MFX_D3D11_SUPPORT
			{
				m_pMFXAllocator = new D3DFrameAllocator;
				MSDK_CHECK_POINTER(m_pMFXAllocator, MFX_ERR_MEMORY_ALLOC);

				D3DAllocatorParams *pd3dAllocParams = new D3DAllocatorParams;
				MSDK_CHECK_POINTER(pd3dAllocParams, MFX_ERR_MEMORY_ALLOC);
				pd3dAllocParams->pManager = reinterpret_cast<IDirect3DDeviceManager9 *>(hdl);

				m_pmfxAllocatorParams = pd3dAllocParams;
			}

			/* In case of video memory we must provide MediaSDK with external allocator
			thus we demonstrate "external allocator" usage model.
			Call SetAllocator to pass allocator to Media SDK */
			sts = m_mfxSession.SetFrameAllocator(m_pMFXAllocator);
			MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
		} else {
			// create system memory allocator
			m_pMFXAllocator = new SysMemFrameAllocator;
			MSDK_CHECK_POINTER(m_pMFXAllocator, MFX_ERR_MEMORY_ALLOC);
		}
		sts = m_pMFXAllocator->Init(m_pmfxAllocatorParams);
		MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
		return sts;
	}

public:
	virtual mfxStatus AllocSurfaces(MFXFrameAllocator *pMFXAllocator, bool m_bExternalAlloc) {
		//pMFXAllocator = m_pMFXAllocator;
		if (m_PluginRequest.NumFrameSuggested == 0) {
			return MFX_ERR_NOT_INITIALIZED;
		}

		mfxStatus sts = pMFXAllocator->Alloc(pMFXAllocator->pthis, &m_PluginRequest, &m_PluginResponse);
		if (MFX_ERR_NONE != sts)
			return sts;

		m_pPluginSurfaces = new mfxFrameSurface1 [m_PluginResponse.NumFrameActual];
		MSDK_CHECK_POINTER(m_pPluginSurfaces, MFX_ERR_MEMORY_ALLOC);

		for (int i = 0; i < m_PluginResponse.NumFrameActual; i++) {
			MSDK_ZERO_MEMORY(m_pPluginSurfaces[i]);
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
	virtual mfxStatus Init(const TCHAR *pluginName, void *pPluginParam, mfxU32 nPluginParamSize,
		bool useHWLib, MemType memType, CHWDevice *phwdev, MFXFrameAllocator* pAllocator,
		mfxU16 nAsyncDepth, const mfxFrameInfo& frameIn, mfxU16 IOPattern) {

		MSDK_CHECK_POINTER(pluginName, MFX_ERR_NULL_PTR);
		MSDK_CHECK_POINTER(pPluginParam, MFX_ERR_NULL_PTR);
		MSDK_CHECK_POINTER(phwdev, MFX_ERR_NULL_PTR);

		m_hwdev = phwdev;

		mfxStatus sts = InitSession(useHWLib, memType);
		MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

		m_mfxSession.SetFrameAllocator(pAllocator);
		
		m_pUsrPlugin.reset(new Rotate());
		MSDK_CHECK_POINTER(m_pUsrPlugin.get(), MFX_ERR_NOT_FOUND);

		InitMfxPluginParam(nAsyncDepth, frameIn, IOPattern);

		// register plugin callbacks in Media SDK
		mfxPlugin plg = make_mfx_plugin_adapter((MFXGenericPlugin*)m_pUsrPlugin.get());
		sts = MFXVideoUSER_Register(m_mfxSession, 0, &plg);
		MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

		// need to call Init after registration because mfxCore interface is needed
		sts = m_pUsrPlugin->Init(&m_pluginVideoParams);
		MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

		sts = m_pUsrPlugin->SetAuxParams(pPluginParam, nPluginParamSize);
		MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

		return MFX_ERR_NONE;
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
		MSDK_ZERO_MEMORY(m_pluginVideoParams);

		m_pluginVideoParams.AsyncDepth = nAsyncDepth;
		memcpy(&m_pluginVideoParams.vpp.In,  &frameIn, sizeof(frameIn));
		memcpy(&m_pluginVideoParams.vpp.Out, &frameIn, sizeof(frameIn));
		m_pluginVideoParams.IOPattern = IOPattern;
		return MFX_ERR_NONE;
	}
public:
	mfxU16                  m_nSurfNum;
	mfxFrameAllocRequest    m_PluginRequest;
	mfxVideoParam           m_pluginVideoParams;
	mfxFrameAllocResponse   m_PluginResponse;
	mfxFrameSurface1       *m_pPluginSurfaces;
	bool                    m_bPluginFlushed;
	MFXFrameAllocator      *m_pMFXAllocator;
private:
	mfxAllocatorParams     *m_pmfxAllocatorParams;
	MFXVideoSession         m_mfxSession;
	MemType                 m_memType;
	mfxVersion              m_mfxVer;
	msdk_so_handle          m_PluginModule;
	unique_ptr<Rotate>      m_pUsrPlugin;
	CHWDevice              *m_hwdev;
};
