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
#include "../base/plugin_base.h"

class ProcessorRotate : public Processor
{
public:
    virtual mfxStatus Init(mfxFrameSurface1 *frame_in, mfxFrameSurface1 *frame_out, const void *data) override;
};

struct RotateParam {
    mfxU16   Angle;  // rotation angle

    RotateParam() : Angle(180) {
    };

    RotateParam(mfxU16 angle) {
        Angle = angle;
    };
};

class Rotator180 : public ProcessorRotate
{
public:
    Rotator180();
    virtual ~Rotator180();

    virtual mfxStatus Process(DataChunk *chunk, mfxU8 *pBuffer) override;
};

typedef struct {
    mfxFrameSurface1 *In;
    mfxFrameSurface1 *Out;
    bool bBusy;
    ProcessorRotate *pProcessor;
} RotateTask;

class Rotate : public QSVEncPlugin
{
public:
    Rotate();
    virtual ~Rotate();

    // methods to be called by Media SDK
    virtual mfxStatus Init(mfxVideoParam *mfxParam);
    virtual mfxStatus SetAuxParams(void* auxParam, int auxParamSize);
    virtual mfxStatus Submit(const mfxHDL *in, mfxU32 in_num, const mfxHDL *out, mfxU32 out_num, mfxThreadTask *task);
    // methods to be called by application
    static MFXGenericPlugin* CreateGenericPlugin() {
        return new Rotate();
    }

    virtual mfxStatus Close();

protected:
    RotateParam     m_Param;
};

#endif // __SAMPLE_PLUGIN_H__
