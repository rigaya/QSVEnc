//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

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

    MSDK_CHECK_POINTER(strFileName, MFX_ERR_NULL_PTR);
    MSDK_CHECK_ERROR(_tclen(strFileName), 0, MFX_ERR_NULL_PTR);

    Close();

    MSDK_CHECK_POINTER(pEncThread, MFX_ERR_NULL_PTR);
    m_pEncThread = pEncThread;

    MSDK_CHECK_POINTER(pEncSatusInfo.get(), MFX_ERR_NULL_PTR);
    m_pEncSatusInfo = pEncSatusInfo;

    
    MSDK_CHECK_POINTER(pInputCrop, MFX_ERR_NULL_PTR);
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
            { sizeof(BITMAPINFOHEADER), 0, 0, 1, 12, MFX_FOURCC_YV12, m_inputFrameInfo.Width * m_inputFrameInfo.Height * 3/2, 0, 0, 0, 0 },
            { sizeof(BITMAPINFOHEADER), 0, 0, 1, 16, MFX_FOURCC_YUY2, m_inputFrameInfo.Width * m_inputFrameInfo.Height * 2,   0, 0, 0, 0 },
            { sizeof(BITMAPINFOHEADER), 0, 0, 1, 24, BI_RGB,          m_inputFrameInfo.Width * m_inputFrameInfo.Height * 3,   0, 0, 0, 0 },
            { sizeof(BITMAPINFOHEADER), 0, 0, 1, 32, BI_RGB,          m_inputFrameInfo.Width * m_inputFrameInfo.Height * 3,   0, 0, 0, 0 }
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
    m_sConvert = get_convert_csp_func(m_ColorFormat, m_inputFrameInfo.FourCC, false);
    tstring mes = strsprintf(_T("avi: %s(%s)->%s[%s], %dx%d, %d/%d fps"), strFcc.c_str(),
        ColorFormatToStr(m_ColorFormat), ColorFormatToStr(m_inputFrameInfo.FourCC), get_simd_str(m_sConvert->simd),
        m_inputFrameInfo.Width, m_inputFrameInfo.Height, m_inputFrameInfo.FrameRateExtN, m_inputFrameInfo.FrameRateExtD);
    AddMessage(QSV_LOG_DEBUG, mes);
    m_strInputInfo += mes;
    m_tmLastUpdate = std::chrono::system_clock::now();

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
#ifdef _DEBUG
    MSDK_CHECK_ERROR(m_bInited, false, MFX_ERR_NOT_INITIALIZED);
#endif
    int w, h;
    mfxU8 *ptr_src;
    mfxFrameInfo* pInfo = &pSurface->Info;
    mfxFrameData* pData = &pSurface->Data;

    mfxU16 CropLeft = m_sInputCrop.left;
    mfxU16 CropUp = m_sInputCrop.up;
    mfxU16 CropRight = m_sInputCrop.right;
    mfxU16 CropBottom = m_sInputCrop.bottom;

    if (m_pEncSatusInfo->m_nInputFrames >= *(DWORD*)&m_inputFrameInfo.FrameId)
        return MFX_ERR_MORE_DATA;

    if (pInfo->CropH > 0 && pInfo->CropW > 0) {
        w = pInfo->CropW;
        h = pInfo->CropH;
    } else {
        w = pInfo->Width;
        h = pInfo->Height;
    }
    w += (CropLeft + CropRight);
    h += (CropUp + CropBottom);

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
    m_sConvert->func[interlaced]((void **)dst_ptr, (void **)src_ptr, w, w * m_nYPitchMultiplizer, w/2, pData->Pitch, h, crop);

    m_pEncSatusInfo->m_nInputFrames++;
    // display update
    auto tm = std::chrono::system_clock::now();
    if (duration_cast<std::chrono::milliseconds>(tm - m_tmLastUpdate).count() > UPDATE_INTERVAL) {
        m_tmLastUpdate = tm;
        m_pEncSatusInfo->UpdateDisplay(tm, 0);
    }
    return MFX_ERR_NONE;
}

#endif //ENABLE_AVI_READER
