//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef __SAMPLE_PLUGIN_H__
#define __SAMPLE_PLUGIN_H__

#include <stdlib.h>
#include <memory.h>
#include <vector>

#include "mfx_plugin_base.h"
#include "sample_defs.h"

typedef struct {
    mfxU32 StartLine;
    mfxU32 EndLine;
} DataChunk;

class Processor
{
public:
    Processor();
    virtual ~Processor();
    virtual mfxStatus SetAllocator(mfxFrameAllocator *pAlloc);
    virtual mfxStatus Init(mfxFrameSurface1 *frame_in, mfxFrameSurface1 *frame_out);
    virtual mfxStatus Process(DataChunk *chunk) = 0;

protected:
    //locks frame or report of an error
    mfxStatus LockFrame(mfxFrameSurface1 *frame);
    mfxStatus UnlockFrame(mfxFrameSurface1 *frame);

    mfxFrameSurface1  *m_pIn;
    mfxFrameSurface1  *m_pOut;
    mfxFrameAllocator *m_pAlloc;

    std::vector<mfxU8> m_YIn, m_UVIn;
    std::vector<mfxU8> m_YOut, m_UVOut;

};

struct RotateParam {
    mfxU16   Angle;  // rotation angle

    RotateParam() : Angle(180) {
    };

    RotateParam(mfxU16 angle) {
        Angle = angle;
    };
};

class Rotator180 : public Processor
{
public:
    Rotator180();
    virtual ~Rotator180();

    virtual mfxStatus Process(DataChunk *chunk);
};

typedef struct {
    mfxFrameSurface1 *In;
    mfxFrameSurface1 *Out;
    bool bBusy;
    Processor *pProcessor;
} RotateTask;

class Rotate : public MFXGenericPlugin
{
public:
    Rotate();
    virtual ~Rotate();

    // methods to be called by Media SDK
    virtual mfxStatus PluginInit(mfxCoreInterface *core);
    virtual mfxStatus Init(mfxVideoParam *mfxParam);
    virtual mfxStatus SetAuxParams(void* auxParam, int auxParamSize);
    virtual mfxStatus PluginClose();
    virtual mfxStatus GetPluginParam(mfxPluginParam *par);
    virtual mfxStatus Submit(const mfxHDL *in, mfxU32 in_num, const mfxHDL *out, mfxU32 out_num, mfxThreadTask *task);
    virtual mfxStatus Execute(mfxThreadTask task, mfxU32 uid_p, mfxU32 uid_a);
    virtual mfxStatus FreeResources(mfxThreadTask task, mfxStatus sts);
    virtual void Release(){}
    // methods to be called by application
    virtual mfxStatus QueryIOSurf(mfxVideoParam *par, mfxFrameAllocRequest *in, mfxFrameAllocRequest *out);
    static MFXGenericPlugin* CreateGenericPlugin() {
        return new Rotate();
    }

    virtual mfxStatus Close();

protected:
    bool m_bInited;

    MFXCoreInterface m_mfxCore;

    mfxVideoParam   m_VideoParam;
    mfxPluginParam  m_PluginParam;
    RotateParam     m_Param;

    RotateTask      *m_pTasks;
    mfxU32          m_MaxNumTasks;

    DataChunk *m_pChunks;

    mfxU32 m_NumChunks;

    mfxStatus CheckParam(mfxVideoParam *mfxParam, RotateParam *pRotatePar);
    mfxStatus CheckInOutFrameInfo(mfxFrameInfo *pIn, mfxFrameInfo *pOut);
    mfxU32 FindFreeTaskIdx();

    bool m_bIsInOpaque;
    bool m_bIsOutOpaque;
};

#endif // __SAMPLE_PLUGIN_H__
