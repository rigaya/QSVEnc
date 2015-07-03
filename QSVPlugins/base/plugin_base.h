//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef __PLUGIN_BASE_H__
#define __PLUGIN_BASE_H__

#include <memory>
#include <vector>

#include "mfx_plugin_base.h"
#include "sample_defs.h"
#include "qsv_util.h"

using std::vector;
using std::unique_ptr;

typedef struct {
	mfxU32 StartLine;
	mfxU32 EndLine;
} DataChunk;

struct aligned_malloc_deleter {
	void operator()(void* ptr) const {
		_aligned_free(ptr);
	}
};

class Processor
{
public:
	Processor()
		: m_pIn(NULL)
		, m_pOut(NULL)
		, m_pAlloc(NULL) {
	}
	virtual ~Processor() {

	}
	virtual mfxStatus SetAllocator(mfxFrameAllocator *pAlloc) {
		m_pAlloc = pAlloc;
		return MFX_ERR_NONE;
	}
	virtual mfxStatus Init(mfxFrameSurface1 *frame_in, mfxFrameSurface1 *frame_out, const void *data) = 0;
	virtual mfxStatus Process(DataChunk *chunk, mfxU8 *pBuffer) = 0;
protected:
	//locks frame or report of an error
	mfxStatus LockFrame(mfxFrameSurface1 *frame) {
		MSDK_CHECK_POINTER(frame, MFX_ERR_NULL_PTR);
		//double lock impossible
		if (frame->Data.Y != 0 && frame->Data.MemId !=0)
			return MFX_ERR_UNSUPPORTED;
		//no allocator used, no need to do lock
		if (frame->Data.Y != 0)
			return MFX_ERR_NONE;
		//lock required
		mfxStatus sts = m_pAlloc->Lock(m_pAlloc->pthis, frame->Data.MemId, &frame->Data);
		MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
		return sts;
	}
	mfxStatus UnlockFrame(mfxFrameSurface1 *frame) {
		MSDK_CHECK_POINTER(frame, MFX_ERR_NULL_PTR);
		//unlock not possible, no allocator used
		if (frame->Data.Y != 0 && frame->Data.MemId ==0)
			return MFX_ERR_NONE;
		//already unlocked
		if (frame->Data.Y == 0)
			return MFX_ERR_NONE;
		//unlock required
		mfxStatus sts = m_pAlloc->Unlock(m_pAlloc->pthis, frame->Data.MemId, &frame->Data);
		MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
		return sts;
	}

	mfxFrameSurface1  *m_pIn;
	mfxFrameSurface1  *m_pOut;
	mfxFrameAllocator *m_pAlloc;

	vector<mfxU8>      m_YIn, m_UVIn;
	vector<mfxU8>      m_YOut, m_UVOut;
};

#pragma warning (push)
#pragma warning (disable: 4100)
typedef struct PluginTask {
	mfxFrameSurface1 *In = nullptr;
	mfxFrameSurface1 *Out = nullptr;
	bool bBusy = false;
	unique_ptr<Processor> pProcessor;
	unique_ptr<mfxU8, aligned_malloc_deleter> pBuffer;
	PluginTask() {};
	PluginTask(const PluginTask& o) {
	}
} DelogoTask;

class QSVEncPlugin : public MFXGenericPlugin
{
public:
	QSVEncPlugin() :
		m_bIsInOpaque(false),
		m_bIsOutOpaque(false),
		m_pAllocator(nullptr) {
		memset(&m_VideoParam, 0, sizeof(m_VideoParam));

		memset(&m_PluginParam, 0, sizeof(m_PluginParam));
		m_PluginParam.MaxThreadNum = 1;
		m_PluginParam.ThreadPolicy = MFX_THREADPOLICY_SERIAL;
	}

	virtual ~QSVEncPlugin() {

	}

	// methods to be called by Media SDK
	virtual mfxStatus PluginInit(mfxCoreInterface *core) {
		MSDK_CHECK_POINTER(core, MFX_ERR_NULL_PTR);
		m_mfxCore = MFXCoreInterface(*core);
		return MFX_ERR_NONE;
	}
	virtual mfxStatus PluginClose() {
		return MFX_ERR_NONE;
	}
	virtual mfxStatus GetPluginParam(mfxPluginParam *par) {
		MSDK_CHECK_POINTER(par, MFX_ERR_NULL_PTR);

		*par = m_PluginParam;

		return MFX_ERR_NONE;
	}
	virtual mfxStatus Execute(mfxThreadTask task, mfxU32 uid_p, mfxU32 uid_a) {
		MSDK_CHECK_ERROR(m_bInited, false, MFX_ERR_NOT_INITIALIZED);

		mfxStatus sts = MFX_ERR_NONE;
		PluginTask *current_task = (PluginTask *)task;

		// 0,...,NumChunks - 2 calls return TASK_WORKING, NumChunks - 1,.. return TASK_DONE
		if (uid_a < m_sChunks.size()) {
			// there's data to process
			sts = current_task->pProcessor->Process(&m_sChunks[uid_a], current_task->pBuffer.get());
			MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
			// last call?
			sts = ((m_sChunks.size() - 1) == uid_a) ? MFX_TASK_DONE : MFX_TASK_WORKING;
		} else {
			// no data to process
			sts = MFX_TASK_DONE;
		}

		return sts;
	}
	virtual mfxStatus FreeResources(mfxThreadTask task, mfxStatus sts) {
		MSDK_CHECK_ERROR(m_bInited, false, MFX_ERR_NOT_INITIALIZED);

		PluginTask *current_task = (PluginTask *)task;

		m_mfxCore.DecreaseReference(&(current_task->In->Data));
		m_mfxCore.DecreaseReference(&(current_task->Out->Data));

		current_task->bBusy = false;

		return MFX_ERR_NONE;
	}
	virtual void Release(){}
	// methods to be called by application
	virtual mfxStatus QueryIOSurf(mfxVideoParam *par, mfxFrameAllocRequest *in, mfxFrameAllocRequest *out) {
		MSDK_CHECK_POINTER(par, MFX_ERR_NULL_PTR);
		MSDK_CHECK_POINTER(in, MFX_ERR_NULL_PTR);
		MSDK_CHECK_POINTER(out, MFX_ERR_NULL_PTR);

		in->Info = par->vpp.In;
		in->NumFrameSuggested = in->NumFrameMin = par->AsyncDepth + 1;

		out->Info = par->vpp.Out;
		out->NumFrameSuggested = out->NumFrameMin = par->AsyncDepth + 1;

		return MFX_ERR_NONE;
	}

	virtual tstring GetPluginName() {
		return m_pluginName;
	}

	virtual tstring GetPluginMessage() {
		return m_message;
	}
	virtual void SetMfxVer(mfxVersion ver) {
		m_PluginParam.APIVersion = ver;
	}

protected:
	mfxU32 FindFreeTaskIdx() {
		mfxU32 i;
		const mfxU32 maxTasks = (mfxU32)m_sTasks.size();
		for (i = 0; i < maxTasks; i++) {
			if (false == m_sTasks[i].bBusy) {
				break;
			}
		}
		return i;
	}
	bool m_bInited;

	MFXCoreInterface m_mfxCore;

	mfxVideoParam   m_VideoParam;
	mfxPluginParam  m_PluginParam;

	vector<DataChunk> m_sChunks;
	vector<PluginTask> m_sTasks;

	mfxStatus CheckParam(mfxVideoParam *mfxParam) {
		MSDK_CHECK_POINTER(mfxParam, MFX_ERR_NULL_PTR);

		mfxInfoVPP *pParam = &mfxParam->vpp;

		// only NV12 color format is supported
		if (MFX_FOURCC_NV12 != pParam->In.FourCC || MFX_FOURCC_NV12 != pParam->Out.FourCC) {
			return MFX_ERR_UNSUPPORTED;
		}

		return MFX_ERR_NONE;
	}
	mfxStatus CheckInOutFrameInfo(mfxFrameInfo *pIn, mfxFrameInfo *pOut) {
		MSDK_CHECK_POINTER(pIn, MFX_ERR_NULL_PTR);
		MSDK_CHECK_POINTER(pOut, MFX_ERR_NULL_PTR);

		if (pIn->CropW != m_VideoParam.vpp.In.CropW   || pIn->CropH != m_VideoParam.vpp.In.CropH ||
			pIn->FourCC != m_VideoParam.vpp.In.FourCC ||
			pOut->CropW != m_VideoParam.vpp.Out.CropW || pOut->CropH != m_VideoParam.vpp.Out.CropH ||
			pOut->FourCC != m_VideoParam.vpp.Out.FourCC) {
			m_message += _T("In-Out Param is invalid.\n");
			return MFX_ERR_INVALID_VIDEO_PARAM;
		}

		return MFX_ERR_NONE;
	}

	bool m_bIsInOpaque;
	bool m_bIsOutOpaque;
	
	tstring m_pluginName;
	tstring m_message;
};
#pragma warning (pop)


#endif // __PLUGIN_BASE_H__
