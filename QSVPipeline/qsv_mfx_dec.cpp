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

#include <set>
#include "rgy_log.h"
#include "qsv_mfx_dec.h"
#include "qsv_allocator.h"
#include "qsv_hw_device.h"

QSVMfxDec::QSVMfxDec(CQSVHWDevice *hwdev, QSVAllocator *allocator,
    mfxVersion mfxVer, mfxIMPL impl, MemType memType, const MFXVideoSession2Params& sessionParams, QSVDeviceNum deviceNum, std::shared_ptr<RGYLog> log) :
    m_mfxSession(),
    m_mfxVer(mfxVer),
    m_hwdev(hwdev),
    m_impl(impl),
    m_memType(memType),
    m_sessionParams(sessionParams),
    m_deviceNum(deviceNum),
    m_allocator(allocator),
    m_allocatorInternal(),
    m_crop(),
    m_mfxDec(),
    m_DecExtParams(),
    m_DecVidProc(),
    m_mfxDecParams(),
    m_skipAV1C(false),
    m_log(log) {
    RGY_MEMSET_ZERO(m_DecVidProc);
    RGY_MEMSET_ZERO(m_mfxDecParams);
};

QSVMfxDec::~QSVMfxDec() { clear(); };

void QSVMfxDec::clear() {
    if (m_mfxDec) {
        m_mfxDec->Close();
        m_mfxDec.reset();
    }
    if (m_mfxSession) {
        m_mfxSession.DisjoinSession();
        m_mfxSession.Close();
    }
    m_allocatorInternal.reset();
    m_allocator = nullptr;
    m_hwdev = nullptr;

    m_log.reset();
}

RGY_ERR QSVMfxDec::InitMFXSession() {
    // init session, and set memory type
    m_mfxSession.Close();
    auto err = InitSession(m_mfxSession, m_sessionParams, m_impl, m_deviceNum, m_log);
    if (err != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to Init session for DEC: %s.\n"), get_err_mes(err));
        return err;
    }

    //使用できる最大のversionをチェック
    auto mfxVer = m_mfxVer;
    m_mfxSession.QueryVersion(&mfxVer);
    if (!check_lib_version(mfxVer, m_mfxVer)) {
        PrintMes(RGY_LOG_ERROR, _T("Session mfxver for DEC does not match version of the base session.\n"));
        return RGY_ERR_UNDEFINED_BEHAVIOR;
    }
    mfxIMPL impl;
    m_mfxSession.QueryIMPL(&impl);
    PrintMes(RGY_LOG_DEBUG, _T("InitSession: mfx lib version: %d.%02d, impl %s\n"), m_mfxVer.Major, m_mfxVer.Minor, MFXImplToStr(impl).c_str());

    if (!m_allocator) { // 内部で独自のallocatorを作る必要がある
        bool externalAlloc = false;
        // SetFrameAllocator も内部で行われる
        err = CreateAllocator(m_allocatorInternal, externalAlloc, m_memType, m_hwdev, m_mfxSession, m_log);
        if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to create internal allocator: %s.\n"), get_err_mes(err));
            return err;
        }
        m_allocator = m_allocatorInternal.get();
        PrintMes(RGY_LOG_DEBUG, _T("Created internal allocator for decode.\n"));
    } else {
        if (impl != MFX_IMPL_SOFTWARE) {
            const auto hdl_t = mfxHandleTypeFromMemType(m_memType, false);
            if (hdl_t) {
                mfxHDL hdl = nullptr;
                err = err_to_rgy(m_hwdev->GetHandle(hdl_t, &hdl));
                if (err != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to get HW device handle: %s.\n"), get_err_mes(err));
                    return err;
                }
                PrintMes(RGY_LOG_DEBUG, _T("Got HW device handle: %p.\n"), hdl);
                // hwエンコード時のみハンドルを渡す
                err = err_to_rgy(m_mfxSession.SetHandle(hdl_t, hdl));
                if (err != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to set HW device handle to dec session: %s.\n"), get_err_mes(err));
                    return err;
                }
                PrintMes(RGY_LOG_DEBUG, _T("set HW device handle %p to encode session.\n"), hdl);
            }
        }
        if ((err = err_to_rgy(m_mfxSession.SetFrameAllocator(m_allocator))) != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to set frame allocator: %s.\n"), get_err_mes(err));
            return err;
        }
        PrintMes(RGY_LOG_DEBUG, _T("Set allocator for decode.\n"));
    }

    return RGY_ERR_NONE;
}

RGY_ERR QSVMfxDec::SetParam(
    const RGY_CODEC inputCodec,
    RGYBitstream& inputHeader,
    const VideoInfo& inputFrameInfo) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (inputCodec == RGY_CODEC_UNKNOWN) {
        PrintMes(RGY_LOG_ERROR, _T("Unknown codec %s for hw decoder.\n"), CodecToStr(inputCodec).c_str());
        return RGY_ERR_UNSUPPORTED;
    }
    //デコーダの作成
    m_mfxDec.reset(new MFXVideoDECODE(m_mfxSession));
    if (!m_mfxDec) {
        return RGY_ERR_MEMORY_ALLOC;
    }

    if (inputCodec == RGY_CODEC_H264 || inputCodec == RGY_CODEC_HEVC) {
        //これを付加しないとMFXVideoDECODE_DecodeHeaderが成功しない
        const uint32_t IDR = 0x65010000;
        inputHeader.append((uint8_t *)&IDR, sizeof(IDR));
    }
    memset(&m_mfxDecParams, 0, sizeof(m_mfxDecParams));
    m_mfxDecParams.mfx.CodecId = codec_rgy_to_enc(inputCodec);
    m_mfxDecParams.IOPattern = (uint16_t)((m_memType != SYSTEM_MEMORY) ? MFX_IOPATTERN_OUT_VIDEO_MEMORY : MFX_IOPATTERN_OUT_SYSTEM_MEMORY);
    // RFF使用時に、フィールドの情報を取得するために必要
    // RFFのときに MFX_PICSTRUCT_PROGRESSIVE に加え、MFX_PICSTRUCT_FIELD_TFFまたはMFX_PICSTRUCT_FIELD_BFF、MFX_PICSTRUCT_FIELD_REPEATEDが設定される
    m_mfxDecParams.mfx.ExtendedPicStruct = 1;
    m_skipAV1C = false;
    sts = err_to_rgy(m_mfxDec->DecodeHeader(&inputHeader.bitstream(), &m_mfxDecParams));
    if (sts != RGY_ERR_NONE && inputCodec == RGY_CODEC_AV1) {
        // AV1ではそのままのヘッダだと、DecodeHeaderに失敗する場合がある QSVEnc #122
        // その場合、4byte飛ばすと読めるかも? https://github.com/FFmpeg/FFmpeg/commit/ffd1316e441a8310cf1746d86fed165e17e10018
        inputHeader.addOffset(4);
        inputHeader.setSize(inputHeader.size() - 4);
        PrintMes(RGY_LOG_DEBUG, _T("Skip 4 bytes of header and retry DecodeHeader: %s.\n"), get_err_mes(sts));
        sts = err_to_rgy(m_mfxDec->DecodeHeader(&inputHeader.bitstream(), &m_mfxDecParams));
        m_skipAV1C = true;
    }
    if (sts != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to DecodeHeader: %s.\n"), get_err_mes(sts));
        return sts;
    }

    //DecodeHeaderした結果をreaderにも反映
    if (!check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_9)
        || (inputCodec != RGY_CODEC_VP8 && inputCodec != RGY_CODEC_VP9 && inputCodec != RGY_CODEC_AV1)) { // VP8/VP9ではこの処理は不要
        if (m_mfxDecParams.mfx.FrameInfo.BitDepthLuma == 8)   m_mfxDecParams.mfx.FrameInfo.BitDepthLuma = 0;
        if (m_mfxDecParams.mfx.FrameInfo.BitDepthChroma == 8) m_mfxDecParams.mfx.FrameInfo.BitDepthChroma = 0;
    }
    if (m_mfxDecParams.mfx.FrameInfo.Shift
        && m_mfxDecParams.mfx.FrameInfo.BitDepthLuma == 0
        && m_mfxDecParams.mfx.FrameInfo.BitDepthChroma == 0) {
        PrintMes(RGY_LOG_DEBUG, _T("InitMfxDecParams: Bit shift required but bitdepth not set.\n"));
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }
    if (m_mfxDecParams.mfx.FrameInfo.FrameRateExtN == 0
        && m_mfxDecParams.mfx.FrameInfo.FrameRateExtD == 0) {
        if (inputFrameInfo.fpsN > 0 && inputFrameInfo.fpsD > 0) {
            m_mfxDecParams.mfx.FrameInfo.FrameRateExtN = inputFrameInfo.fpsN;
            m_mfxDecParams.mfx.FrameInfo.FrameRateExtD = inputFrameInfo.fpsD;
        }
    }

    PrintMes(RGY_LOG_DEBUG, _T("")
        _T("InitMfxDecParams: QSVDec prm: %s, Level %d, Profile %d\n")
        _T("InitMfxDecParams: Frame: %s, %dx%d%s [%d,%d,%d,%d] %d:%d\n")
        _T("InitMfxDecParams: color format %s, chroma %s, bitdepth %d, shift %d, picstruct %s\n"),
        CodecToStr(codec_enc_to_rgy(m_mfxDecParams.mfx.CodecId)).c_str(), m_mfxDecParams.mfx.CodecLevel, m_mfxDecParams.mfx.CodecProfile,
        ColorFormatToStr(m_mfxDecParams.mfx.FrameInfo.FourCC), m_mfxDecParams.mfx.FrameInfo.Width, m_mfxDecParams.mfx.FrameInfo.Height,
        (m_mfxDecParams.mfx.FrameInfo.PicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF)) ? _T("i") : _T("p"),
        m_mfxDecParams.mfx.FrameInfo.CropX, m_mfxDecParams.mfx.FrameInfo.CropY, m_mfxDecParams.mfx.FrameInfo.CropW, m_mfxDecParams.mfx.FrameInfo.CropH,
        m_mfxDecParams.mfx.FrameInfo.AspectRatioW, m_mfxDecParams.mfx.FrameInfo.AspectRatioH,
        ColorFormatToStr(m_mfxDecParams.mfx.FrameInfo.FourCC), ChromaFormatToStr(m_mfxDecParams.mfx.FrameInfo.ChromaFormat),
        m_mfxDecParams.mfx.FrameInfo.BitDepthLuma, m_mfxDecParams.mfx.FrameInfo.Shift,
        MFXPicStructToStr(m_mfxDecParams.mfx.FrameInfo.PicStruct).c_str());

    memset(&m_DecVidProc, 0, sizeof(m_DecVidProc));
    m_DecExtParams.clear();
#if 0
    const auto enc_fourcc = csp_rgy_to_enc(getEncoderCsp(pInParams, nullptr));
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_23)
        && ( m_mfxDecParams.mfx.FrameInfo.CropW  != pInParams->input.dstWidth
            || m_mfxDecParams.mfx.FrameInfo.CropH  != pInParams->input.dstHeight
            || m_mfxDecParams.mfx.FrameInfo.FourCC != enc_fourcc)
        && pInParams->vpp.nScalingQuality == MFX_SCALING_MODE_LOWPOWER
        && enc_fourcc == MFX_FOURCC_NV12
        && m_mfxDecParams.mfx.FrameInfo.FourCC == MFX_FOURCC_NV12
        && m_mfxDecParams.mfx.FrameInfo.ChromaFormat == MFX_CHROMAFORMAT_YUV420
        && !cropEnabled(pInParams->sInCrop)) {
        m_DecVidProc.Header.BufferId = MFX_EXTBUFF_DEC_VIDEO_PROCESSING;
        m_DecVidProc.Header.BufferSz = sizeof(m_DecVidProc);
        m_DecVidProc.In.CropX = 0;
        m_DecVidProc.In.CropY = 0;
        m_DecVidProc.In.CropW = m_mfxDecParams.mfx.FrameInfo.CropW;
        m_DecVidProc.In.CropH = m_mfxDecParams.mfx.FrameInfo.CropH;

        m_DecVidProc.Out.FourCC = enc_fourcc;
        m_DecVidProc.Out.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
        m_DecVidProc.Out.Width  = std::max<mfxU16>(ALIGN16(pInParams->input.dstWidth), m_mfxDecParams.mfx.FrameInfo.Width);
        m_DecVidProc.Out.Height = std::max<mfxU16>(ALIGN16(pInParams->input.dstHeight), m_mfxDecParams.mfx.FrameInfo.Height);
        m_DecVidProc.Out.CropX = 0;
        m_DecVidProc.Out.CropY = 0;
        m_DecVidProc.Out.CropW = pInParams->input.dstWidth;
        m_DecVidProc.Out.CropH = pInParams->input.dstHeight;

        m_DecExtParams.push_back((mfxExtBuffer *)&m_DecVidProc);
        m_mfxDecParams.ExtParam = &m_DecExtParams[0];
        m_mfxDecParams.NumExtParam = (mfxU16)m_DecExtParams.size();

        pInParams->input.srcWidth = pInParams->input.dstWidth;
        pInParams->input.srcHeight = pInParams->input.dstHeight;
    }
#endif
    return RGY_ERR_NONE;
}

RGY_ERR QSVMfxDec::Init() {
    //ここでの内部エラーは最終的にはmfxライブラリ内部で解決される場合もあり、これをログ上は無視するようにする。
    //具体的にはSandybridgeでd3dメモリでVPPを使用する際、m_pmfxVPP->Init()実行時に
    //"QSVAllocator: Failed CheckRequestType: undeveloped feature"と表示されるが、
    //m_mfxDec->Initの戻り値自体はMFX_ERR_NONEであるので、内部で解決されたものと思われる。
    //もちろん、m_mfxDec->Init自体がエラーを返した時にはきちんとログに残す。
    const auto log_level = logTemporarilyIgnoreErrorMes();
    auto err = err_to_rgy(m_mfxDec->Init(&m_mfxDecParams));
    m_log->setLogLevelAll(log_level);
    if (err == RGY_WRN_PARTIAL_ACCELERATION) {
        PrintMes(RGY_LOG_WARN, _T("partial acceleration on dec.\n"));
        err = RGY_ERR_NONE;
    }
    if (err != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("Failed to initialize dec: %s.\n"), get_err_mes(err));
        return err;
    }
    PrintMes(RGY_LOG_DEBUG, _T("Dec initialized.\n"));
    return RGY_ERR_NONE;
}

RGY_ERR QSVMfxDec::Close() {
    if (m_mfxDec) {
        auto err = err_to_rgy(m_mfxDec->Close());
        RGY_IGNORE_STS(err, RGY_ERR_NOT_INITIALIZED);
        if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to reset encoder (fail on closing): %s."), get_err_mes(err));
            return err;
        }
        PrintMes(RGY_LOG_DEBUG, _T("Dec Closed.\n"));
    }
    return RGY_ERR_NONE;
}

RGYFrameInfo QSVMfxDec::GetFrameOut() const {
    const auto& mfxOut = m_mfxDecParams.mfx.FrameInfo;

    const RGYFrameInfo info(mfxOut.CropW,  mfxOut.CropH,
        csp_enc_to_rgy(mfxOut.FourCC), (mfxOut.BitDepthLuma > 0) ? mfxOut.BitDepthLuma : 8,
        picstruct_enc_to_rgy(mfxOut.PicStruct),
        (m_memType != SYSTEM_MEMORY) ? RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED : RGY_MEM_TYPE_CPU);
    return info;
}

rgy_rational<int> QSVMfxDec::GetOutFps() const {
    const auto& mfxOut = m_mfxDecParams.mfx.FrameInfo;
    return rgy_rational<int>(mfxOut.FrameRateExtN, mfxOut.FrameRateExtD);
}

void QSVMfxDec::PrintMes(RGYLogLevel log_level, const TCHAR *format, ...) {
    if (m_log.get() == nullptr) {
        if (log_level <= RGY_LOG_INFO) {
            return;
        }
    } else if (log_level < m_log->getLogLevel(RGY_LOGT_DEC)) {
        return;
    }

    va_list args;
    va_start(args, format);

    int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    vector<TCHAR> buffer(len, 0);
    _vstprintf_s(buffer.data(), len, format, args);
    va_end(args);

    tstring mes = tstring(_T("MFXDEC: ")) + buffer.data();

    if (m_log.get() != nullptr) {
        m_log->write(log_level, RGY_LOGT_DEC, mes.c_str());
    } else {
        _ftprintf(stderr, _T("%s"), mes.c_str());
    }
}

int QSVMfxDec::clamp_param_int(int value, int low, int high, const TCHAR *param_name) {
    auto value_old = value;
    value = clamp(value, low, high);
    if (value != value_old) {
        PrintMes(RGY_LOG_WARN, _T("%s value changed %d -> %d, must be in range of %d-%d\n"), param_name, value_old, value, low, high);
    }
    return value;
}

RGY_ERR QSVMfxDec::CheckParamList(int value, const CX_DESC *list, const char *param_name) {
    for (int i = 0; list[i].desc; i++)
        if (list[i].value == value)
            return RGY_ERR_NONE;
    PrintMes(RGY_LOG_ERROR, _T("%s=%d, is not valid param.\n"), param_name, value);
    return RGY_ERR_INVALID_VIDEO_PARAM;
}

RGYParamLogLevel QSVMfxDec::logTemporarilyIgnoreErrorMes() {
    //MediaSDK内のエラーをRGY_LOG_DEBUG以下の時以外には一時的に無視するようにする。
    //RGY_LOG_DEBUG以下の時にも、「無視できるエラーが発生するかもしれない」ことをログに残す。
    const auto log_level = m_log->getLogLevelAll();
    if (   log_level.get(RGY_LOGT_CORE) >= RGY_LOG_MORE
        || log_level.get(RGY_LOGT_DEC)  >= RGY_LOG_MORE
        || log_level.get(RGY_LOGT_DEV)  >= RGY_LOG_MORE) {
        m_log->setLogLevel(RGY_LOG_QUIET, RGY_LOGT_ALL); //一時的にエラーを無視
    } else {
        PrintMes(RGY_LOG_DEBUG, _T("There might be error below, but it might be internal error which could be ignored.\n"));
    }
    return log_level;
}
