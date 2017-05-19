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
// ------------------------------------------------------------------------------------------

#ifndef __QSV_HW_DEVICE_H__
#define __QSV_HW_DEVICE_H__

#include <cstdint>
#include <memory>
#include "mfxvideo++.h"
#include "rgy_log.h"
#include "rgy_version.h"

#if MFX_D3D11_SUPPORT
#include <sdkddkver.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#endif //#if MFX_D3D11_SUPPORT

class CQSVHWDevice {
public:
    CQSVHWDevice() {};
    virtual ~CQSVHWDevice() { }
    virtual mfxStatus Init(mfxHDL hWindow, uint32_t nAdapterNum, shared_ptr<RGYLog> pQSVLog) = 0;
    virtual mfxStatus Reset() = 0;
    virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL *pHdl) = 0;
    virtual void      Close() = 0;
protected:
    std::shared_ptr<RGYLog> m_pQSVLog;
};

#endif //#ifndef __QSV_HW_DEVICE_H__
