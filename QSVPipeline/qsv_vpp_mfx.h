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

#ifndef __QSV_VPP_MFX_H__
#define __QSV_VPP_MFX_H__


#include <memory>
#include "rgy_log.h"
#include "qsv_query.h"
#include "qsv_allocator.h"
#include "qsv_hw_device.h"
#include "qsv_session.h"

static const auto MFX_EXTBUFF_VPP_DENOISE_OLD = MFX_MAKEFOURCC('D','N','I','S');

class QSVVppMfx {
public:
    QSVVppMfx(CQSVHWDevice *hwdev, QSVAllocator *allocator, mfxVersion mfxVer, mfxIMPL impl, MemType memType, const MFXVideoSession2Params& sessionParams, QSVDeviceNum deviceNum, int asyncDepth, std::shared_ptr<RGYLog> log);
    virtual ~QSVVppMfx();

    RGY_ERR SetParam(sVppParams& params,
        const RGYFrameInfo& frameOut,
        const RGYFrameInfo& frameIn,
        const sInputCrop *crop, const rgy_rational<int> infps, const rgy_rational<int> sar, const int blockSize);
    RGY_ERR Init();
    RGY_ERR Close();
    RGY_ERR Reset(const mfxFrameInfo& frameOut, const mfxFrameInfo& frameIn);

    void clear();

    std::vector<VppType> GetVppList() const;
    RGYFrameInfo GetFrameOut() const;
    rgy_rational<int> GetOutFps() const;
    mfxSession GetSession() { return m_mfxSession; }
    MFXVideoVPP *mfxvpp() { return m_mfxVPP.get(); }
    mfxVideoParam& mfxparams() { return m_mfxVppParams; }
    mfxVersion mfxver() const { return m_mfxVer; }
    int asyncDepth() const { return m_asyncDepth; }
    tstring print() const { return VppExtMes; }
    bool isDeinterlace() const { return m_ExtDeinterlacing.Mode != 0;  }
protected:
    void InitStructs();
    RGY_ERR InitMFXSession();
    mfxFrameInfo SetMFXFrameIn(const RGYFrameInfo& frameIn, const sInputCrop *crop, const rgy_rational<int> infps, const rgy_rational<int> sar, const int blockSize);
    RGY_ERR SetMFXFrameOut(mfxFrameInfo& mfxOut, const sVppParams& params, const RGYFrameInfo& frameOut, const mfxFrameInfo& frameIn, const int blockSize);
    RGY_ERR SetVppExtBuffers(sVppParams& params);
    RGY_ERR InitMfxVppParams(const sVppParams& params, const mfxFrameInfo& mfxOut, const mfxFrameInfo& mfxIn);
    RGY_ERR checkVppParams(sVppParams& params, const bool inputInterlaced);

    void PrintMes(RGYLogLevel log_level, const TCHAR *format, ...);
    int clamp_param_int(int value, int low, int high, const TCHAR *param_name);
    RGY_ERR CheckParamList(int value, const CX_DESC *list, const char *param_name);
    void vppExtAddMes(const tstring& str);
    RGYParamLogLevel logTemporarilyIgnoreErrorMes();
protected:
    MFXVideoSession2 m_mfxSession;          //VPP用のSession メインSessionにJoinして使用する
    mfxVersion m_mfxVer;
    CQSVHWDevice *m_hwdev; //mainから渡されるdevice情報
    QSVAllocator *m_allocator;             //mainから渡されるallocator
    mfxIMPL m_impl;
    MemType m_memType;             //パイプラインのSurfaceのメモリType;
    MFXVideoSession2Params m_sessionParams;
    QSVDeviceNum m_deviceNum;
    int m_asyncDepth;

    sInputCrop m_crop;
    std::unique_ptr<MFXVideoVPP> m_mfxVPP;
    mfxVideoParam m_mfxVppParams;
    mfxExtVPPDoNotUse m_VppDoNotUse;
    mfxExtVPPDoNotUse m_VppDoUse;
#pragma warning(push)
#pragma warning(disable:4996) // warning C4996: 古い形式として宣言されました。
RGY_DISABLE_WARNING_PUSH
RGY_DISABLE_WARNING_STR("-Wdeprecated-declarations")
    mfxExtVPPDenoise m_ExtDenoise; //互換性のため必要
RGY_DISABLE_WARNING_POP
#pragma warning(pop)
    mfxExtVPPDenoise2 m_ExtDenoise2;
    mfxExtVppMctf m_ExtMctf;
    mfxExtVPPDetail m_ExtDetail;
    mfxExtVPPDeinterlacing m_ExtDeinterlacing;
    mfxExtVPPFrameRateConversion m_ExtFrameRateConv;
    mfxExtVPPRotation m_ExtRotate;
    mfxExtVPPVideoSignalInfo m_ExtVppVSI;
    mfxExtVPPImageStab m_ExtImageStab;
    mfxExtVPPMirroring m_ExtMirror;
    mfxExtVPPScaling m_ExtScaling;
    mfxExtVPPAISuperResolution m_ExtAISuperRes;
    mfxExtVPPPercEncPrefilter m_ExtPercEncPrefilter;
    std::vector<mfxU32> m_VppDoNotUseList;
    std::vector<mfxU32> m_VppDoUseList;
    std::vector<mfxExtBuffer*> m_VppExtParams;
    tstring VppExtMes;

    std::shared_ptr<RGYLog> m_log;
};

#endif // __QSV_VPP_MFX_H__
