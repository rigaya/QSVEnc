// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2021 rigaya
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

#ifndef __QSV_MFX_DEC_H__
#define __QSV_MFX_DEC_H__

#include "qsv_query.h"
#include "qsv_session.h"

class QSVAllocator;
class CQSVHWDevice;
class RGYLog;

class QSVMfxDec {
public:
    QSVMfxDec(CQSVHWDevice *hwdev, QSVAllocator *allocator,
        mfxVersion mfxVer, mfxIMPL impl, MemType memType, const MFXVideoSession2Params& sessionParams, QSVDeviceNum deviceNum, std::shared_ptr<RGYLog> log);
    virtual ~QSVMfxDec();

    RGY_ERR InitMFXSession();
    RGY_ERR SetParam(const RGY_CODEC inputCodec,
        RGYBitstream& inputHeader,
        const VideoInfo& inputFrameInfo);
    RGY_ERR Init();
    RGY_ERR Close();

    void clear();

    RGYFrameInfo GetFrameOut() const;
    rgy_rational<int> GetOutFps() const;
    mfxSession GetSession() { return m_mfxSession; }
    MFXVideoSession *GetVideoSessionPtr() { return &m_mfxSession; }
    MFXVideoDECODE *mfxdec() { return m_mfxDec.get(); }
    mfxVideoParam& mfxparams() { return m_mfxDecParams; }
    mfxVersion mfxver() const { return m_mfxVer; }
    MemType memType() const { return m_memType; }
    QSVAllocator *allocator() { return m_allocator; }
    bool skipAV1C() const { return m_skipAV1C; }
protected:
    void PrintMes(RGYLogLevel log_level, const TCHAR *format, ...);
    int clamp_param_int(int value, int low, int high, const TCHAR *param_name);
    RGY_ERR CheckParamList(int value, const CX_DESC *list, const char *param_name);
    RGYParamLogLevel logTemporarilyIgnoreErrorMes();
protected:
    MFXVideoSession2 m_mfxSession;          //VPP用のSession メインSessionにJoinして使用する
    mfxVersion m_mfxVer;
    CQSVHWDevice *m_hwdev; //mainから渡されるdevice情報
    mfxIMPL m_impl;
    MemType m_memType;             //パイプラインのSurfaceのメモリType;
    MFXVideoSession2Params m_sessionParams;
    QSVDeviceNum m_deviceNum;
    QSVAllocator *m_allocator;             //mainから渡されるallocator
    std::unique_ptr<QSVAllocator> m_allocatorInternal;

    sInputCrop m_crop;
    std::unique_ptr<MFXVideoDECODE> m_mfxDec;
    std::vector<mfxExtBuffer*> m_DecExtParams;
    mfxExtDecVideoProcessing m_DecVidProc;
    mfxVideoParam m_mfxDecParams;
    bool m_skipAV1C;

    std::shared_ptr<RGYLog> m_log;
};

#endif // __QSV_MFX_DEC_H__
