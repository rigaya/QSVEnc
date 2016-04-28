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

#ifndef __PLUGIN_BASE_H__
#define __PLUGIN_BASE_H__

#include <memory>
#include <vector>
#include <mfxplugin++.h>
#include "qsv_version.h"
#include "qsv_util.h"
#include "qsv_log.h"
#include "qsv_allocator_d3d9.h"

#if D3D_SURFACES_SUPPORT
#define D3D_CALL(x) { HRESULT hr = (x); if( FAILED(hr) ) { return MFX_ERR_UNDEFINED_BEHAVIOR; } }
#endif

using std::vector;
using std::unique_ptr;

typedef struct {
    uint32_t StartLine;
    uint32_t EndLine;
} DataChunk;

class Processor
{
public:
    Processor()
        : m_ProcessorName()
        , m_pIn(nullptr)
        , m_pOut(nullptr)
        , m_pAlloc(nullptr)
        , m_hDevice(nullptr)
#if D3D_SURFACES_SUPPORT
        , m_pD3DDeviceManager(nullptr)
#endif
        , m_pPrintMes()
    {
    }
    virtual ~Processor() {
#if D3D_SURFACES_SUPPORT
        if (m_pD3DDeviceManager && m_hDevice) {
            m_pD3DDeviceManager->CloseDeviceHandle(m_hDevice);
            m_hDevice = nullptr;
        }
#endif
        m_pPrintMes.reset();
    }
    virtual mfxStatus SetAllocator(mfxFrameAllocator *pAlloc) {
        m_pAlloc = pAlloc;
        return MFX_ERR_NONE;
    }
    virtual mfxStatus Init(mfxFrameSurface1 *frame_in, mfxFrameSurface1 *frame_out, const void *data) = 0;
    virtual mfxStatus Process(DataChunk *chunk, uint8_t *pBuffer) = 0;
    void SetLog(shared_ptr<CQSVLog> pLog) {
        m_pPrintMes = pLog;
    }
protected:
    //D3DバッファをGPUでコピーする
    //CPUでmovntdqa(_mm_stream_load_si128)を駆使するよりは高速
    //正常に実行するためには、m_pAllocは、
    //PluginのmfxCoreから取得したAllocatorではなく、
    //メインパイプラインから直接受け取ったAllocatorでなければならない
    mfxStatus CopyD3DFrameGPU(mfxFrameSurface1 *pFrameIn, mfxFrameSurface1 *pFrameOut) {
#if D3D_SURFACES_SUPPORT
        if (m_pD3DDeviceManager == nullptr) {
            m_pD3DDeviceManager = ((QSVAllocatorD3D9*)m_pAlloc)->GetDeviceManager();
        }
        if (m_hDevice == NULL) {
            D3D_CALL(m_pD3DDeviceManager->OpenDeviceHandle(&m_hDevice));
        }
        IDirect3DDevice9 *pd3dDevice = nullptr;
        D3D_CALL(m_pD3DDeviceManager->LockDevice(m_hDevice, &pd3dDevice, false));
        D3D_CALL(pd3dDevice->StretchRect(
            static_cast<directxMemId *>(pFrameIn->Data.MemId)->m_surface, NULL,
            static_cast<directxMemId *>(pFrameOut->Data.MemId)->m_surface, NULL, D3DTEXF_NONE));
        D3D_CALL(pd3dDevice->Release());
        D3D_CALL(m_pD3DDeviceManager->UnlockDevice(m_hDevice, false));
#endif //#if D3D_SURFACES_SUPPORT
        return MFX_ERR_NONE;
    }

    //locks frame or report of an error
    mfxStatus LockFrame(mfxFrameSurface1 *frame) {
        //double lock impossible
        if (frame->Data.Y != 0 && frame->Data.MemId !=0)
            return MFX_ERR_UNSUPPORTED;
        //no allocator used, no need to do lock
        if (frame->Data.Y != 0)
            return MFX_ERR_NONE;
        //lock required
        return m_pAlloc->Lock(m_pAlloc->pthis, frame->Data.MemId, &frame->Data);
    }
    mfxStatus UnlockFrame(mfxFrameSurface1 *frame) {
        //unlock not possible, no allocator used
        if (frame->Data.Y != 0 && frame->Data.MemId ==0)
            return MFX_ERR_NONE;
        //already unlocked
        if (frame->Data.Y == 0)
            return MFX_ERR_NONE;
        //unlock required
        return m_pAlloc->Unlock(m_pAlloc->pthis, frame->Data.MemId, &frame->Data);
    }
    void AddMessage(int log_level, const tstring& str) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel()) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto& line : lines) {
            if (line[0] != _T('\0')) {
                m_pPrintMes->write(log_level, (m_ProcessorName + _T(": ") + line + _T("\n")).c_str());
            }
        }
    }
    void AddMessage(int log_level, const TCHAR *format, ... ) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel()) {
            return;
        }

        va_list args;
        va_start(args, format);
        int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
        tstring buffer;
        buffer.resize(len, _T('\0'));
        _vstprintf_s(&buffer[0], len, format, args);
        va_end(args);
        AddMessage(log_level, buffer);
    }

    tstring m_ProcessorName;
    mfxFrameSurface1  *m_pIn;
    mfxFrameSurface1  *m_pOut;
    mfxFrameAllocator *m_pAlloc;

    vector<uint8_t>      m_YIn, m_UVIn;
    vector<uint8_t>      m_YOut, m_UVOut;
#if D3D_SURFACES_SUPPORT
    IDirect3DDeviceManager9 *m_pD3DDeviceManager;
#endif //#if D3D_SURFACES_SUPPORT
    HANDLE                   m_hDevice;
    shared_ptr<CQSVLog>      m_pPrintMes; //ログ出力用オブジェクト
};

#pragma warning (push)
#pragma warning (disable: 4100)
typedef struct PluginTask {
    mfxFrameSurface1 *In = nullptr;
    mfxFrameSurface1 *Out = nullptr;
    bool bBusy = false;
    unique_ptr<Processor> pProcessor;
    unique_ptr<uint8_t, aligned_malloc_deleter> pBuffer;
    PluginTask() {};
    PluginTask(const PluginTask& o) {
    }
} DelogoTask;

enum {
    PLUGIN_SEND_DATA_UNKNOWN = 0,
    PLUGIN_SEND_DATA_AVPACKET,
};

class QSVEncPlugin : public MFXGenericPlugin
{
public:
    QSVEncPlugin() :
        m_bInited(false),
        m_bIsInOpaque(false),
        m_bIsOutOpaque(false),
        m_pPrintMes() {
        memset(&m_VideoParam, 0, sizeof(m_VideoParam));

        memset(&m_PluginParam, 0, sizeof(m_PluginParam));
        m_PluginParam.MaxThreadNum = 1;
        m_PluginParam.ThreadPolicy = MFX_THREADPOLICY_SERIAL;
    }

    virtual ~QSVEncPlugin() {
        m_pPrintMes.reset();
    }

    virtual void SetLog(shared_ptr<CQSVLog> pPrintMes) {
        m_pPrintMes = pPrintMes;
    }

    // methods to be called by Media SDK
    virtual mfxStatus PluginInit(mfxCoreInterface *core) {
        m_mfxCore = MFXCoreInterface(*core);
        return MFX_ERR_NONE;
    }
    virtual mfxStatus PluginClose() {
        m_pPrintMes.reset();
        return MFX_ERR_NONE;
    }
    virtual mfxStatus GetPluginParam(mfxPluginParam *par) override {
        *par = m_PluginParam;

        return MFX_ERR_NONE;
    }
    virtual mfxStatus Execute(mfxThreadTask task, mfxU32 uid_p, mfxU32 uid_a) {
        if (!m_bInited) return MFX_ERR_NOT_INITIALIZED;

        mfxStatus sts = MFX_ERR_NONE;
        PluginTask *current_task = (PluginTask *)task;

        // 0,...,NumChunks - 2 calls return TASK_WORKING, NumChunks - 1,.. return TASK_DONE
        if (uid_a < m_sChunks.size()) {
            // there's data to process
            sts = current_task->pProcessor->Process(&m_sChunks[uid_a], current_task->pBuffer.get());
            if (sts < MFX_ERR_NONE) return sts;
            // last call?
            sts = ((m_sChunks.size() - 1) == uid_a) ? MFX_TASK_DONE : MFX_TASK_WORKING;
        } else {
            return MFX_TASK_DONE;
        }

        return sts;
    }
    virtual mfxStatus FreeResources(mfxThreadTask task, mfxStatus sts) {
        if (!m_bInited) return MFX_ERR_NOT_INITIALIZED;

        PluginTask *current_task = (PluginTask *)task;

        m_mfxCore.DecreaseReference(&(current_task->In->Data));
        m_mfxCore.DecreaseReference(&(current_task->Out->Data));

        current_task->bBusy = false;

        return MFX_ERR_NONE;
    }
    virtual void Release(){}
    // methods to be called by application
    virtual mfxStatus QueryIOSurf(mfxVideoParam *par, mfxFrameAllocRequest *in, mfxFrameAllocRequest *out) {
        in->Info = par->vpp.In;
        in->NumFrameSuggested = in->NumFrameMin = par->AsyncDepth + 1;

        out->Info = par->vpp.Out;
        out->NumFrameSuggested = out->NumFrameMin = par->AsyncDepth + 1;

        return MFX_ERR_NONE;
    }

    virtual mfxStatus SendData(int nType, void *pData) {
        return MFX_ERR_NONE;
    }

    virtual tstring GetPluginName() {
        return m_pluginName;
    }

    virtual tstring GetPluginMessage() {
        return m_message;
    }

    virtual int getTargetTrack() {
        return -1;
    }

    virtual void SetMfxVer(mfxVersion ver) {
        m_PluginParam.APIVersion = ver;
    }

protected:
    uint32_t FindFreeTaskIdx() {
        uint32_t i;
        const uint32_t maxTasks = (uint32_t)m_sTasks.size();
        for (i = 0; i < maxTasks; i++) {
            if (false == m_sTasks[i].bBusy) {
                break;
            }
        }
        return i;
    }
    void AddMessage(int log_level, const tstring& str) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel()) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto& line : lines) {
            if (line[0] != _T('\0')) {
                m_pPrintMes->write(log_level, (m_pluginName + _T(": ") + line + _T("\n")).c_str());
            }
        }
    }
    void AddMessage(int log_level, const TCHAR *format, ... ) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel()) {
            return;
        }

        va_list args;
        va_start(args, format);
        int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
        tstring buffer;
        buffer.resize(len, _T('\0'));
        _vstprintf_s(&buffer[0], len, format, args);
        va_end(args);
        AddMessage(log_level, buffer);
    }
    bool m_bInited;

    MFXCoreInterface m_mfxCore;

    mfxVideoParam   m_VideoParam;
    mfxPluginParam  m_PluginParam;

    vector<DataChunk> m_sChunks;
    vector<PluginTask> m_sTasks;

    mfxStatus CheckParam(mfxVideoParam *mfxParam) {
        mfxInfoVPP *pParam = &mfxParam->vpp;

        // only NV12 color format is supported
        if (MFX_FOURCC_NV12 != pParam->In.FourCC || MFX_FOURCC_NV12 != pParam->Out.FourCC) {
            return MFX_ERR_UNSUPPORTED;
        }

        return MFX_ERR_NONE;
    }
    mfxStatus CheckInOutFrameInfo(mfxFrameInfo *pIn, mfxFrameInfo *pOut) {
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
    shared_ptr<CQSVLog> m_pPrintMes; //ログ出力用オブジェクト
};
#pragma warning (pop)

#endif // __PLUGIN_BASE_H__
