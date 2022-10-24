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
#include <cstdarg>
#include <memory>
#include "rgy_version.h"
#include "rgy_util.h"
#pragma warning (push)
#pragma warning (disable: 4201) //C4201: 非標準の拡張機能が使用されています: 無名の構造体または共用体です。
#pragma warning (disable: 4996) //C4996: 'MFXInit': が古い形式として宣言されました。
#pragma warning (disable: 4819) //C4819: ファイルは、現在のコード ページ (932) で表示できない文字を含んでいます。データの損失を防ぐために、ファイルを Unicode 形式で保存してください。
RGY_DISABLE_WARNING_PUSH
RGY_DISABLE_WARNING_STR("-Wdeprecated-declarations")
#include "mfxvideo++.h"
RGY_DISABLE_WARNING_POP
#pragma warning (pop)
#include "rgy_log.h"

#if MFX_D3D11_SUPPORT
#include <sdkddkver.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#endif //#if MFX_D3D11_SUPPORT

class CQSVHWDevice {
public:
    CQSVHWDevice(std::shared_ptr<RGYLog> pQSVLog) : m_name(_T("hwdev")), m_pQSVLog(pQSVLog) {  };
    virtual ~CQSVHWDevice() { }
    virtual mfxStatus Init(mfxHDL hWindow, uint32_t nViews, uint32_t nAdapterNum) = 0;
    virtual mfxStatus Reset() = 0;
    virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL *pHdl) = 0;
    virtual void      Close() = 0;
    virtual LUID      GetLUID() { return LUID(); };
    virtual tstring   GetName() { return _T(""); };
protected:
    void AddMessage(RGYLogLevel log_level, const tstring &str);
    void AddMessage(RGYLogLevel log_level, const TCHAR *format, ...);
    tstring m_name;
    std::shared_ptr<RGYLog> m_pQSVLog;
};

#endif //#ifndef __QSV_HW_DEVICE_H__
