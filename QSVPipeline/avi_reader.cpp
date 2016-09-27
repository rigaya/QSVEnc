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

#include "avi_reader.h"
#if ENABLE_AVI_READER

#pragma warning(disable:4312)
#pragma warning(disable:4838)

CAVIReader::CAVIReader() {
    m_pAviFile = NULL;
    m_pAviStream = NULL;
    m_pGetFrame = NULL;
    m_pBitmapInfoHeader = NULL;
    m_nYPitchMultiplizer = 1;
    m_strReaderName = _T("avi");
}

CAVIReader::~CAVIReader() {
    Close();
}

#pragma warning(push)
#pragma warning(disable:4100)
mfxStatus CAVIReader::Init(const TCHAR *strFileName, mfxU32 ColorFormat, const void *option, CEncodingThread *pEncThread, shared_ptr<CEncodeStatusInfo> pEncSatusInfo, sInputCrop *pInputCrop) {
    Close();

    m_pEncThread = pEncThread;
    m_pEncSatusInfo = pEncSatusInfo;
    memcpy(&m_sInputCrop, pInputCrop, sizeof(m_sInputCrop));
    
    AVIFileInit();

    if (0 != AVIFileOpen(&m_pAviFile, strFileName, OF_READ | OF_SHARE_DENY_NONE, NULL)) {
        AddMessage(QSV_LOG_ERROR, _T("failed to open avi file: \"%s\"\n"), strFileName);
        return MFX_ERR_INVALID_HANDLE;
    }
    AddMessage(QSV_LOG_DEBUG, _T("openend avi file: \"%s\"\n"), strFileName);

    AVIFILEINFO finfo = { 0 };
    if (0 != AVIFileInfo(m_pAviFile, &finfo, sizeof(AVIFILEINFO))) {
        AddMessage(QSV_LOG_ERROR, _T("failed to get avi file info.\n"));
        return MFX_ERR_INVALID_HANDLE;
    }
    tstring strFcc;
    for (DWORD i_stream = 0; i_stream < finfo.dwStreams; i_stream++) {
        if (0 != AVIFileGetStream(m_pAviFile, &m_pAviStream, 0, i_stream))
            return MFX_ERR_INVALID_HANDLE;
        AVISTREAMINFO sinfo = { 0 };
        if (0 == AVIStreamInfo(m_pAviStream, &sinfo, sizeof(AVISTREAMINFO)) && sinfo.fccType == streamtypeVIDEO) {
            memset(&m_inputFrameInfo, 0, sizeof(m_inputFrameInfo));
            const DWORD fps_gcd = qsv_gcd(sinfo.dwRate, sinfo.dwScale);
            m_inputFrameInfo.Width = (mfxU16)(sinfo.rcFrame.right - sinfo.rcFrame.left);
            m_inputFrameInfo.Height = (mfxU16)(sinfo.rcFrame.bottom - sinfo.rcFrame.top);
            m_inputFrameInfo.CropW = m_inputFrameInfo.Width - (pInputCrop->left + pInputCrop->right);
            m_inputFrameInfo.CropH = m_inputFrameInfo.Height - (pInputCrop->up + pInputCrop->bottom);
            m_inputFrameInfo.FrameRateExtN = sinfo.dwRate / fps_gcd;
            m_inputFrameInfo.FrameRateExtD = sinfo.dwScale / fps_gcd;
            DWORD frames = sinfo.dwLength - sinfo.dwStart;
            memcpy(&m_inputFrameInfo.FrameId, &frames, sizeof(frames));
            m_ColorFormat = sinfo.fccHandler;
            strFcc = char_to_tstring((char *)sinfo.fccHandler);
            break;
        }
        AVIStreamRelease(m_pAviStream);
        m_pAviStream = NULL;
    }
    if (m_pAviStream == NULL) {
        AddMessage(QSV_LOG_ERROR, _T("failed to get valid stream from avi file.\n"));
        return MFX_ERR_INVALID_HANDLE;
    }
    AddMessage(QSV_LOG_DEBUG, _T("found video stream from avi file.\n"));

    if (   m_ColorFormat == MFX_FOURCC_YUY2
        || m_ColorFormat == MFX_FOURCC_YV12) {
        //何もしない
    } else {
        BITMAPINFOHEADER bih[4] = {
            { sizeof(BITMAPINFOHEADER), 0, 0, 1, 12, QSV_ENC_CSP_YV12, m_inputFrameInfo.Width * m_inputFrameInfo.Height * 3/2, 0, 0, 0, 0 },
            { sizeof(BITMAPINFOHEADER), 0, 0, 1, 16, QSV_ENC_CSP_YUY2, m_inputFrameInfo.Width * m_inputFrameInfo.Height * 2,   0, 0, 0, 0 },
            { sizeof(BITMAPINFOHEADER), 0, 0, 1, 24, BI_RGB,           m_inputFrameInfo.Width * m_inputFrameInfo.Height * 3,   0, 0, 0, 0 },
            { sizeof(BITMAPINFOHEADER), 0, 0, 1, 32, BI_RGB,           m_inputFrameInfo.Width * m_inputFrameInfo.Height * 3,   0, 0, 0, 0 }
        };
        for (int i = 0; i < _countof(bih); i++) {
            if (NULL == (m_pGetFrame = AVIStreamGetFrameOpen(m_pAviStream, &bih[i]))) {
                continue;
            }
            m_ColorFormat = (bih[i].biCompression == BI_RGB) ? ((bih[i].biBitCount == 24) ? MFX_FOURCC_RGB3 : MFX_FOURCC_RGB4) : bih[i].biCompression;
            break;
        }

        if (m_pGetFrame == NULL) {
            if (   NULL == (m_pGetFrame = AVIStreamGetFrameOpen(m_pAviStream, NULL))
                && NULL == (m_pGetFrame = AVIStreamGetFrameOpen(m_pAviStream, (BITMAPINFOHEADER *)AVIGETFRAMEF_BESTDISPLAYFMT))) {
                AddMessage(QSV_LOG_ERROR, _T("\nfailed to decode avi file.\n"));
                return MFX_ERR_INVALID_HANDLE;
            }
            BITMAPINFOHEADER *bmpInfoHeader = (BITMAPINFOHEADER *)AVIStreamGetFrame(m_pGetFrame, 0);
            if (NULL == bmpInfoHeader || bmpInfoHeader->biCompression != 0) {
                AddMessage(QSV_LOG_ERROR, _T("\nfailed to decode avi file.\n"));
                return MFX_ERR_MORE_DATA;
            }

            m_ColorFormat = (bmpInfoHeader->biBitCount == 24) ? MFX_FOURCC_RGB3 : MFX_FOURCC_RGB4;
        }
    }

    switch (m_ColorFormat) {
    case MFX_FOURCC_YUY2: m_nYPitchMultiplizer = 2; break;
    case MFX_FOURCC_RGB3: m_nYPitchMultiplizer = 3; break;
    case MFX_FOURCC_RGB4: m_nYPitchMultiplizer = 4; break;
    case MFX_FOURCC_YV12:
    default: m_nYPitchMultiplizer = 1; break;
    }

    if (   MFX_FOURCC_RGB4 == m_ColorFormat
        || MFX_FOURCC_RGB3 == m_ColorFormat) {
        m_inputFrameInfo.FourCC = MFX_FOURCC_RGB4;
        m_inputFrameInfo.ChromaFormat = 0;
    } else {
        m_inputFrameInfo.FourCC = MFX_FOURCC_NV12;
        m_inputFrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
    }
    m_sConvert = get_convert_csp_func(mfx_fourcc_to_qsv_enc_csp(m_ColorFormat), mfx_fourcc_to_qsv_enc_csp(m_inputFrameInfo.FourCC), false);
    tstring mes = strsprintf(_T("avi: %s(%s)->%s[%s], %dx%d, %d/%d fps"), strFcc.c_str(),
        QSV_ENC_CSP_NAMES[m_sConvert->csp_from], QSV_ENC_CSP_NAMES[m_sConvert->csp_to], get_simd_str(m_sConvert->simd),
        m_inputFrameInfo.Width, m_inputFrameInfo.Height, m_inputFrameInfo.FrameRateExtN, m_inputFrameInfo.FrameRateExtD);
    AddMessage(QSV_LOG_DEBUG, mes);
    m_strInputInfo += mes;

    m_bInited = true;
    return MFX_ERR_NONE;
}
#pragma warning(pop)

void CAVIReader::Close() {
    AddMessage(QSV_LOG_DEBUG, _T("Closing...\n"));
    if (m_pGetFrame)
        AVIStreamGetFrameClose(m_pGetFrame);
    if (m_pAviStream)
        AVIStreamRelease(m_pAviStream);
    if (m_pAviFile)
        AVIFileRelease(m_pAviFile);
    AVIFileExit();
    m_pBuffer.reset();
    m_nBufSize = 0;

    m_pAviFile = NULL;
    m_pAviStream = NULL;
    m_pGetFrame = NULL;
    m_pBitmapInfoHeader = NULL;
    m_bInited = false;
    m_nYPitchMultiplizer = 1;
    m_pEncSatusInfo.reset();
    AddMessage(QSV_LOG_DEBUG, _T("Closed.\n"));
}

mfxStatus CAVIReader::LoadNextFrame(mfxFrameSurface1* pSurface) {
    mfxFrameInfo *pInfo = &pSurface->Info;
    mfxFrameData *pData = &pSurface->Data;

    mfxU16 CropLeft = m_sInputCrop.left;
    mfxU16 CropUp = m_sInputCrop.up;
    mfxU16 CropRight = m_sInputCrop.right;
    mfxU16 CropBottom = m_sInputCrop.bottom;

    if (m_pEncSatusInfo->m_nInputFrames >= *(DWORD*)&m_inputFrameInfo.FrameId
        //m_pEncSatusInfo->m_nInputFramesがtrimの結果必要なフレーム数を大きく超えたら、エンコードを打ち切る
        //ちょうどのところで打ち切ると他のストリームに影響があるかもしれないので、余分に取得しておく
        || getVideoTrimMaxFramIdx() < (int)m_pEncSatusInfo->m_nInputFrames - TRIM_OVERREAD_FRAMES) {
        return MFX_ERR_MORE_DATA;
    }

    int w = 0, h = 0;
    if (pInfo->CropH > 0 && pInfo->CropW > 0) {
        w = pInfo->CropW;
        h = pInfo->CropH;
    } else {
        w = pInfo->Width;
        h = pInfo->Height;
    }
    w += (CropLeft + CropRight);
    h += (CropUp + CropBottom);

    mfxU8 *ptr_src = nullptr;
    if (m_pGetFrame) {
        if (NULL == (ptr_src = (mfxU8 *)AVIStreamGetFrame(m_pGetFrame, m_pEncSatusInfo->m_nInputFrames)))
            return MFX_ERR_MORE_DATA;
        ptr_src += sizeof(BITMAPINFOHEADER);
    } else {
        mfxU32 required_bufsize = w * h * 3;
        if (m_nBufSize < required_bufsize) {
            m_pBuffer.reset();
            m_pBuffer = std::shared_ptr<uint8_t>((uint8_t *)_aligned_malloc(required_bufsize, 16), aligned_malloc_deleter());
            if (m_pBuffer.get())
                return MFX_ERR_MEMORY_ALLOC;
            m_nBufSize = required_bufsize;
        }
        LONG sizeRead = 0;
        if (0 != AVIStreamRead(m_pAviStream, m_pEncSatusInfo->m_nInputFrames, 1, m_pBuffer.get(), (LONG)m_nBufSize, &sizeRead, NULL))
            return MFX_ERR_MORE_DATA;
        ptr_src = m_pBuffer.get();
    }

    BOOL interlaced = 0 != (pSurface->Info.PicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF));
    int crop[4] = { CropLeft, CropUp, CropRight, CropBottom };
    const void *dst_ptr[3] = { pData->Y, pData->UV, NULL };
    const void *src_ptr[3] = { ptr_src, ptr_src + w * h * 5 / 4, ptr_src + w * h };
    if (MFX_FOURCC_RGB4 == m_sConvert->csp_to) {
        dst_ptr[0] = min(min(pData->R, pData->G), pData->B);
    }
    m_sConvert->func[interlaced]((void **)dst_ptr, (const void **)src_ptr, w, w * m_nYPitchMultiplizer, w/2, pData->Pitch, h, h, crop);

    m_pEncSatusInfo->m_nInputFrames++;
    // display update
    return m_pEncSatusInfo->UpdateDisplay(0);
}

#endif //ENABLE_AVI_READER
