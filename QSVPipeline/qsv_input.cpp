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

#include "qsv_input.h"

CQSVInput::CQSVInput() {
    m_pPrintMes.reset();
    m_bInited = false;
    m_fSource = NULL;
    m_ColorFormat = MFX_FOURCC_YV12;
    m_nInputCodec = 0;
    m_pEncSatusInfo.reset();
    m_pEncThread = nullptr;
    m_pBuffer.reset();
    m_nBufSize = 0;
    m_sConvert = nullptr;
    memset(&m_sInputCrop,     0, sizeof(m_sInputCrop));
    memset(&m_inputFrameInfo, 0, sizeof(m_inputFrameInfo));
    m_strInputInfo.empty();
    m_sTrimParam.offset = 0;
    m_strReaderName = _T("unknown");
}

CQSVInput::~CQSVInput() {
    Close();
}

void CQSVInput::Close() {
    AddMessage(QSV_LOG_DEBUG, _T("Closing...\n"));
    if (m_fSource) {
        fclose(m_fSource);
        m_fSource = NULL;
        AddMessage(QSV_LOG_DEBUG, _T("Closed file pointer.\n"));
    }

    m_pEncSatusInfo.reset();
    m_pEncThread = nullptr;
    m_pBuffer.reset();
    m_nBufSize = 0;
    m_bInited = false;
    m_sConvert = nullptr;

    m_strInputInfo.empty();

    m_sTrimParam.list.clear();
    m_sTrimParam.offset = 0;
    AddMessage(QSV_LOG_DEBUG, _T("Close...\n"));
    m_pPrintMes.reset();
}

CQSVInputRaw::CQSVInputRaw() {
    m_by4m = false;
    m_strReaderName = _T("yuv reader");
}

CQSVInputRaw::~CQSVInputRaw() {
    Close();
}

mfxStatus CQSVInputRaw::Init(const TCHAR *strFileName, uint32_t ColorFormat, const void *prm,
    CEncodingThread *pEncThread, shared_ptr<CEncodeStatusInfo> pEncSatusInfo, sInputCrop *pInputCrop) {
    Close();

    m_pEncThread = pEncThread;
    m_pEncSatusInfo = pEncSatusInfo;
    memcpy(&m_sInputCrop, pInputCrop, sizeof(m_sInputCrop));

    m_by4m = *(bool *)prm;
    m_strReaderName = (m_by4m) ? _T("y4m reader") : _T("yuv reader");

    bool use_stdin = _tcscmp(strFileName, _T("-")) == 0;
    if (use_stdin) {
        m_fSource = stdin;
        AddMessage(QSV_LOG_DEBUG, _T("output to stdout.\n"));
    } else {
        int error = 0;
        if (0 != (error = _tfopen_s(&m_fSource, strFileName, _T("rb"))) || m_fSource == nullptr) {
            AddMessage(QSV_LOG_ERROR, _T("Failed to open file \"%s\": %s.\n"), strFileName, _tcserror(error));
            return MFX_ERR_NULL_PTR;
        } else {
            AddMessage(QSV_LOG_DEBUG, _T("Opened file: \"%s\".\n"), strFileName);
        }
    }

    switch (ColorFormat) {
    case MFX_FOURCC_NV12:
    case MFX_FOURCC_YV12:
        m_ColorFormat = ColorFormat;
        break;
    default:
        AddMessage(QSV_LOG_ERROR, _T("invalid color format.\n"));
        return MFX_ERR_UNSUPPORTED;
    }

    m_sConvert = get_convert_csp_func(m_ColorFormat, MFX_FOURCC_NV12, true);

    tstring mes;
    if (m_by4m) {
        //read y4m header
        char buf[128] = { 0 };
        if (fread(buf, 1, strlen("YUV4MPEG2"), m_fSource) != strlen("YUV4MPEG2")
            || strcmp(buf, "YUV4MPEG2") != 0
            || !fgets(buf, sizeof(buf), m_fSource)
            || MFX_ERR_NONE != ParseY4MHeader(buf, &m_inputFrameInfo)) {
            return MFX_ERR_UNSUPPORTED;
        }
        m_inputFrameInfo.CropW = m_inputFrameInfo.Width - (pInputCrop->left + pInputCrop->right);
        m_inputFrameInfo.CropH = m_inputFrameInfo.Height - (pInputCrop->up + pInputCrop->bottom);
        const uint32_t fps_gcd = qsv_gcd(m_inputFrameInfo.FrameRateExtN, m_inputFrameInfo.FrameRateExtD);

        mes = strsprintf(_T("y4m: %s->%s[%s], %dx%d, %d/%d fps"), ColorFormatToStr(m_ColorFormat), ColorFormatToStr(MFX_FOURCC_NV12), get_simd_str(m_sConvert->simd),
            m_inputFrameInfo.Width, m_inputFrameInfo.Height, m_inputFrameInfo.FrameRateExtN / fps_gcd, m_inputFrameInfo.FrameRateExtD / fps_gcd);
    } else {
        mes = ColorFormatToStr(m_ColorFormat);
    }
    AddMessage(QSV_LOG_DEBUG, mes);
    m_strInputInfo += mes;
    m_bInited = true;

    return MFX_ERR_NONE;
}

//この関数がMFX_ERR_NONE以外を返すと、Runループが終了し、
//EncodingThreadのm_stsThreadに返した値がセットされる。
mfxStatus CQSVInputRaw::LoadNextFrame(mfxFrameSurface1* pSurface) {
    mfxFrameInfo *pInfo = &pSurface->Info;
    mfxFrameData *pData = &pSurface->Data;

    int CropLeft = m_sInputCrop.left;
    int CropUp = m_sInputCrop.up;
    int CropRight = m_sInputCrop.right;
    int CropBottom = m_sInputCrop.bottom;

    uint32_t FourCCRequired = pInfo->FourCC;
    if (MFX_FOURCC_NV12 != FourCCRequired && MFX_FOURCC_YV12 != FourCCRequired) {
        return MFX_ERR_UNSUPPORTED;
    }

    //m_pEncSatusInfo->m_nInputFramesがtrimの結果必要なフレーム数を大きく超えたら、エンコードを打ち切る
    //ちょうどのところで打ち切ると他のストリームに影響があるかもしれないので、余分に取得しておく
    if (getVideoTrimMaxFramIdx() < (int)m_pEncSatusInfo->m_nInputFrames - TRIM_OVERREAD_FRAMES) {
        return MFX_ERR_MORE_DATA;
    }

    uint32_t w = 0, h = 0;
    if (pInfo->CropH > 0 && pInfo->CropW > 0) {
        w = pInfo->CropW;
        h = pInfo->CropH;
    } else {
        w = pInfo->Width;
        h = pInfo->Height;
    }
    w += (CropLeft + CropRight);
    h += (CropUp + CropBottom);
    uint32_t required_bufsize = 0;
    uint32_t uv_frames_size = ALIGN16((ALIGN16(w) * h)>>2);
    required_bufsize += ALIGN16(w); //Yの1line分
    required_bufsize += uv_frames_size * 2; //UVの各1フレーム分
    if (m_nBufSize < required_bufsize) {
        m_nBufSize = required_bufsize;
        m_pBuffer.reset();
        m_pBuffer = std::shared_ptr<uint8_t>((uint8_t *)_aligned_malloc(required_bufsize, 16), aligned_malloc_deleter());
        if (!m_pBuffer) {
            return MFX_ERR_NULL_PTR;
        }
    }
    auto bufY = m_pBuffer.get();
    auto bufU = bufY + ALIGN16(w);
    auto bufV = bufU + uv_frames_size;

    uint32_t pitch = pData->Pitch;
    uint8_t *ptr = pData->Y + pInfo->CropX + pInfo->CropY * pData->Pitch;

    if (m_by4m) {
        uint8_t y4m_buf[8] = { 0 };
        if (fread(y4m_buf, 1, strlen("FRAME"), m_fSource) != strlen("FRAME")
            || memcmp(y4m_buf, "FRAME", strlen("FRAME")) != 0) {
            return MFX_ERR_MORE_DATA;
        }
        for (int i = 0; fgetc(m_fSource) != '\n'; i++) {
            if (i >= 64) {
                return MFX_ERR_MORE_DATA;
            }
        }
    }
    // 輝度成分を読み込み
    uint32_t nBytesRead = 0;
    for (int i = 0; i < CropUp; i++) {
        nBytesRead += (uint32_t)fread(bufY, 1, w, m_fSource);
    }
    uint32_t i = 0;
    uint32_t i_fin = 0;
    if (CropLeft == 0) {
        for (i_fin = h - CropBottom - CropUp - (int)(w / pitch); i < i_fin; i++) {
            nBytesRead += (uint32_t)fread(ptr + i * pitch, 1, w, m_fSource);
        }
    }
    for (i_fin = h - CropBottom - CropUp; i < i_fin; i++) {
        nBytesRead += (uint32_t)fread(bufY, 1, w, m_fSource);
        memcpy(ptr + i * pitch, bufY + CropLeft, w - CropLeft - CropRight);
    }
    for (i_fin = h - CropUp; i < i_fin; i++) {
        nBytesRead += (uint32_t)fread(bufY, 1, w, m_fSource);
    }
    if (nBytesRead != (uint32_t)(w * h)) {
        return MFX_ERR_MORE_DATA;
    }

    // 色差を読み込み
    nBytesRead  = (uint32_t)fread(bufU, 1, (w*h)>>2, m_fSource);
    nBytesRead += (uint32_t)fread(bufV, 1, (w*h)>>2, m_fSource);
    if (nBytesRead != (uint32_t)((w*h)>>1)) {
        return MFX_ERR_MORE_DATA;
    }

    BOOL interlaced = 0 != (pSurface->Info.PicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF));
    int crop[4] = { CropLeft, CropUp, CropRight, CropBottom };
    const void *dst_ptr[3] = { pData->Y, pData->UV, NULL };
    const void *src_ptr[3] = { bufY, bufU, bufV };
    m_sConvert->func[interlaced]((void **)dst_ptr, (void **)src_ptr, w, w, w/2, pData->Pitch, h, crop);

    //pSurface->Data.TimeStamp = m_pEncSatusInfo->m_nInputFrames * (mfxU64)m_pEncSatusInfo->m_nOutputFPSScale;
    m_pEncSatusInfo->m_nInputFrames++;

    return m_pEncSatusInfo->UpdateDisplay(0);
}
