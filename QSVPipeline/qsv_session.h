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

#ifndef _QSV_SESSION_H_
#define _QSV_SESSION_H_

#include "rgy_version.h"
#include "qsv_util.h"
#include "qsv_hw_device.h"
#include "qsv_allocator.h"
#include "rgy_log.h"

struct MFXVideoSession2Params {
    int threads;
    int priority;

    MFXVideoSession2Params();
};

class MFXVideoSession2 : public MFXVideoSession {
public:
    MFXVideoSession2();
    virtual ~MFXVideoSession2() {};

    void setParams(std::shared_ptr<RGYLog>& log, const MFXVideoSession2Params& params);
    mfxIMPL devNumToImpl(const QSVDeviceNum dev);
    mfxStatus initD3D9(const QSVDeviceNum dev, const bool suppressErrorMessage);
    mfxStatus initD3D11(const QSVDeviceNum dev, const bool suppressErrorMessage);
    mfxStatus initVA(const QSVDeviceNum dev, const bool suppressErrorMessage);
    mfxStatus initSW(const bool suppressErrorMessage);
    std::vector<mfxImplDescription> getImplList();

    mfxSession get() { return m_session; }
protected:
    RGY_ERR InitSessionInitParam();
    mfxStatus initHW(mfxIMPL& impl, const QSVDeviceNum dev);
    virtual void PrintMes(RGYLogLevel log_level, const TCHAR *format, ...);

    std::shared_ptr<RGYLog> m_log;
    MFXVideoSession2Params m_prm;

    mfxInitParam m_InitParam;
    mfxExtBuffer *m_pInitParamExtBuf[1];
    mfxExtThreadsParam m_ThreadsParam;
};

mfxIMPL GetDefaultMFXImpl();

std::vector<mfxImplDescription> getVPLImplList(std::shared_ptr<RGYLog>& log);

RGY_ERR InitSession(MFXVideoSession2& mfxSession, const MFXVideoSession2Params& params, const mfxIMPL implAcceleration, const QSVDeviceNum dev, std::shared_ptr<RGYLog>& log, const bool suppressErrorMessage = false);

RGY_ERR InitSessionAndDevice(std::unique_ptr<CQSVHWDevice>& hwdev, MFXVideoSession2& mfxSession, MemType& memType, const QSVDeviceNum dev, const MFXVideoSession2Params& params, std::shared_ptr<RGYLog>& log, const bool suppressErrorMessage = false);

RGY_ERR CreateAllocator(
    std::unique_ptr<QSVAllocator>& allocator, bool& externalAlloc,
    const MemType memType, CQSVHWDevice *hwdev, MFXVideoSession2& session, std::shared_ptr<RGYLog>& log);

#endif //_QSV_SESSION_H_
