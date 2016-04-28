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

#ifndef __SAMPLE_PLUGIN_H__
#define __SAMPLE_PLUGIN_H__

#include <stdlib.h>
#include <memory.h>
#include <vector>

#include <mfxplugin++.h>
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
