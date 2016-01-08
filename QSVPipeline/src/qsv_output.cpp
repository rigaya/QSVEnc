//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include "qsv_output.h"

#define WRITE_CHECK(writtenBytes, expected) { \
    if (writtenBytes != expected) { \
        AddMessage(QSV_LOG_ERROR, _T("Error writing file.\nNot enough disk space!\n")); \
        return MFX_ERR_UNDEFINED_BEHAVIOR; \
    } }

CQSVOut::CQSVOut() :
    m_pEncSatusInfo(),
    m_fDest(),
    m_bOutputIsStdout(false),
    m_bInited(false),
    m_bNoOutput(false),
    m_bSourceHWMem(false),
    m_bY4mHeaderWritten(false),
    m_strWriterName(),
    m_strOutputInfo(),
    m_pPrintMes(),
    m_pOutputBuffer(),
    m_pReadBuffer(),
    m_pUVBuffer() {
}

CQSVOut::~CQSVOut() {
    Close();
}

void CQSVOut::Close() {
    AddMessage(QSV_LOG_DEBUG, _T("Closing...\n"));
    if (m_fDest) {
        m_fDest.reset();
        AddMessage(QSV_LOG_DEBUG, _T("Closed file pointer.\n"));
    }
    m_pOutputBuffer.reset();
    m_pReadBuffer.reset();
    m_pUVBuffer.reset();

    m_pEncSatusInfo.reset();
    m_bNoOutput = false;
    m_bInited = false;
    m_bSourceHWMem = false;
    m_bY4mHeaderWritten = false;
    AddMessage(QSV_LOG_DEBUG, _T("Closed.\n"));
    m_pPrintMes.reset();
}

CQSVOutBitstream::CQSVOutBitstream() {
    m_strWriterName = _T("bitstream");
}

CQSVOutBitstream::~CQSVOutBitstream() {
}

#pragma warning(push)
#pragma warning(disable:4100)
mfxStatus CQSVOutBitstream::Init(const TCHAR *strFileName, const void *prm, shared_ptr<CEncodeStatusInfo> pEncSatusInfo) {
    CQSVOutRawPrm *rawPrm = (CQSVOutRawPrm *)prm;
    if (!rawPrm->bBenchmark && _tcslen(strFileName) == 0) {
        AddMessage(QSV_LOG_ERROR, _T("output filename not set.\n"));
        return MFX_ERR_NULL_PTR;
    }

    Close();

    m_pEncSatusInfo = pEncSatusInfo;

    if (rawPrm->bBenchmark) {
        m_bNoOutput = true;
        AddMessage(QSV_LOG_DEBUG, _T("no output for benchmark mode.\n"));
    } else {
        if (_tcscmp(strFileName, _T("-")) == 0) {
            m_fDest.reset(stdout);
            m_bOutputIsStdout = true;
            AddMessage(QSV_LOG_DEBUG, _T("using stdout\n"));
        } else {
            FILE *fp = NULL;
            int error = _tfopen_s(&fp, strFileName, _T("wb+"));
            if (error != 0 || fp == NULL) {
                AddMessage(QSV_LOG_ERROR, _T("failed to open output file \"%s\": %s\n"), strFileName, _tcserror(error));
                return MFX_ERR_NULL_PTR;
            }
            m_fDest.reset(fp);
            AddMessage(QSV_LOG_DEBUG, _T("Opened file \"%s\"\n"), strFileName);

            const int bufferSizeByte = clamp(rawPrm->nBufSizeMB, 0, MAX_BUF_SIZE_MB) * 1024 * 1024;
            if (bufferSizeByte) {
                m_pOutputBuffer.reset((char *)malloc(bufferSizeByte));
                if (m_pOutputBuffer) {
                    setvbuf(m_fDest.get(), m_pOutputBuffer.get(), _IOFBF, bufferSizeByte);
                    AddMessage(QSV_LOG_DEBUG, _T("Added %d MB output buffer.\n"), bufferSizeByte / (1024 * 1024));
                }
            }
        }
    }
    m_bInited = true;
    return MFX_ERR_NONE;
}
mfxStatus CQSVOutBitstream::SetVideoParam(const mfxVideoParam *pMfxVideoPrm, const mfxExtCodingOption2 *cop2) { return MFX_ERR_NONE; };
#pragma warning(pop)
mfxStatus CQSVOutBitstream::WriteNextFrame(mfxBitstream *pMfxBitstream) {
    if (pMfxBitstream == nullptr) {
        AddMessage(QSV_LOG_ERROR, _T("Invalid call: WriteNextFrame\n"));
        return MFX_ERR_NULL_PTR;
    }

    uint32_t nBytesWritten = 0;
    if (!m_bNoOutput) {
        nBytesWritten = (uint32_t)fwrite(pMfxBitstream->Data + pMfxBitstream->DataOffset, 1, pMfxBitstream->DataLength, m_fDest.get());
        WRITE_CHECK(nBytesWritten, pMfxBitstream->DataLength);
    }

    pMfxBitstream->DataLength = 0;
    m_pEncSatusInfo->SetOutputData(nBytesWritten, pMfxBitstream->FrameType);

    return MFX_ERR_NONE;
}

#pragma warning(push)
#pragma warning(disable: 4100)
mfxStatus CQSVOutBitstream::WriteNextFrame(mfxFrameSurface1 *pSurface) {
    return MFX_ERR_UNSUPPORTED;
}
#pragma warning(pop)

CQSVOutFrame::CQSVOutFrame() : m_bY4m(true) {
    m_strWriterName = _T("yuv writer");
};

CQSVOutFrame::~CQSVOutFrame() {
};

#pragma warning(push)
#pragma warning(disable: 4100)
mfxStatus CQSVOutFrame::Init(const msdk_char *strFileName, const void *prm, shared_ptr<CEncodeStatusInfo> pEncSatusInfo) {
    Close();

    if (_tcscmp(strFileName, _T("-")) == 0) {
        m_fDest.reset(stdout);
        m_bOutputIsStdout = true;
        AddMessage(QSV_LOG_DEBUG, _T("using stdout\n"));
    } else {
        FILE *fp = NULL;
        int error = _tfopen_s(&fp, strFileName, _T("wb"));
        if (0 != error || fp == NULL) {
            AddMessage(QSV_LOG_DEBUG, _T("failed to open file \"%s\": %s\n"), strFileName, _tcserror(error));
            return MFX_ERR_NULL_PTR;
        }
        m_fDest.reset(fp);
    }

    YUVWriterParam *writerParam = (YUVWriterParam *)prm;

    m_pEncSatusInfo = pEncSatusInfo;
    m_bY4m = writerParam->bY4m;
    m_bSourceHWMem = !!(writerParam->memType & (D3D11_MEMORY | D3D9_MEMORY));
    m_bInited = true;

    return MFX_ERR_NONE;
}

mfxStatus CQSVOutFrame::SetVideoParam(const mfxVideoParam *pMfxVideoPrm, const mfxExtCodingOption2 *cop2) {
    return MFX_ERR_UNSUPPORTED;
}

mfxStatus CQSVOutFrame::WriteNextFrame(mfxBitstream *pMfxBitstream) {
    return MFX_ERR_UNSUPPORTED;
}
#pragma warning(pop)

mfxStatus CQSVOutFrame::WriteNextFrame(mfxFrameSurface1 *pSurface) {
    mfxFrameInfo &pInfo = pSurface->Info;
    mfxFrameData &pData = pSurface->Data;

    MSDK_CHECK_POINTER(m_fDest.get(), MFX_ERR_NULL_PTR);

    if (m_bSourceHWMem) {
        if (m_pReadBuffer.get() == nullptr) {
            m_pReadBuffer.reset((uint8_t *)_aligned_malloc(pData.Pitch + 128, 16));
        }
    }

    if (m_bY4m) {
        if (!m_bY4mHeaderWritten) {
            WriteY4MHeader(m_fDest.get(), &pInfo);
            m_bY4mHeaderWritten = true;
        }
        WRITE_CHECK(fwrite("FRAME\n", 1, strlen("FRAME\n"), m_fDest.get()), strlen("FRAME\n"));
    }

    auto loadLineToBuffer = [](uint8_t *ptrBuf, uint8_t *ptrSrc, int pitch) {
        for (int i = 0; i < pitch; i += 128, ptrSrc += 128, ptrBuf += 128) {
            __m128i x0 = _mm_stream_load_si128((__m128i *)(ptrSrc +   0));
            __m128i x1 = _mm_stream_load_si128((__m128i *)(ptrSrc +  16));
            __m128i x2 = _mm_stream_load_si128((__m128i *)(ptrSrc +  32));
            __m128i x3 = _mm_stream_load_si128((__m128i *)(ptrSrc +  48));
            __m128i x4 = _mm_stream_load_si128((__m128i *)(ptrSrc +  64));
            __m128i x5 = _mm_stream_load_si128((__m128i *)(ptrSrc +  80));
            __m128i x6 = _mm_stream_load_si128((__m128i *)(ptrSrc +  96));
            __m128i x7 = _mm_stream_load_si128((__m128i *)(ptrSrc + 112));
            _mm_store_si128((__m128i *)(ptrBuf +   0), x0);
            _mm_store_si128((__m128i *)(ptrBuf +  16), x1);
            _mm_store_si128((__m128i *)(ptrBuf +  32), x2);
            _mm_store_si128((__m128i *)(ptrBuf +  48), x3);
            _mm_store_si128((__m128i *)(ptrBuf +  64), x4);
            _mm_store_si128((__m128i *)(ptrBuf +  80), x5);
            _mm_store_si128((__m128i *)(ptrBuf +  96), x6);
            _mm_store_si128((__m128i *)(ptrBuf + 112), x7);
        }
    };

    const uint32_t lumaWidthBytes = pInfo.CropW << ((pInfo.FourCC == MFX_FOURCC_P010) ? 1 : 0);
    if (   pInfo.FourCC == MFX_FOURCC_YV12
        || pInfo.FourCC == MFX_FOURCC_NV12
        || pInfo.FourCC == MFX_FOURCC_P010) {
        const uint32_t cropOffset = pInfo.CropY * pData.Pitch + pInfo.CropX;
        if (m_bSourceHWMem) {
            for (uint32_t j = 0; j < pInfo.CropH; j++) {
                uint8_t *ptrBuf = m_pReadBuffer.get();
                uint8_t *ptrSrc = pData.Y + (pInfo.CropY + j) * pData.Pitch;
                loadLineToBuffer(ptrBuf, ptrSrc, pData.Pitch);
                WRITE_CHECK(fwrite(ptrBuf + pInfo.CropX, 1, lumaWidthBytes, m_fDest.get()), lumaWidthBytes);
            }
        } else {
            for (uint32_t j = 0; j < pInfo.CropH; j++) {
                WRITE_CHECK(fwrite(pData.Y + cropOffset + j * pData.Pitch, 1, lumaWidthBytes, m_fDest.get()), lumaWidthBytes);
            }
        }
    }

    uint32_t frameSize = 0;
    if (pInfo.FourCC == MFX_FOURCC_YV12) {
        frameSize = lumaWidthBytes * pInfo.CropH * 3 / 2;

        uint32_t uvPitch = pData.Pitch >> 1;
        uint32_t uvWidth = pInfo.CropW >> 1;
        uint32_t uvHeight = pInfo.CropH >> 1;
        uint8_t *ptrBuf = m_pReadBuffer.get();
        for (uint32_t i = 0; i < uvHeight; i++) {
            loadLineToBuffer(ptrBuf, pData.U + (pInfo.CropY + i) * uvPitch, uvPitch);
            WRITE_CHECK(fwrite(ptrBuf + (pInfo.CropX >> 1), 1, uvWidth, m_fDest.get()), uvWidth);
        }
        for (uint32_t i = 0; i < uvHeight; i++) {
            loadLineToBuffer(ptrBuf, pData.V + (pInfo.CropY + i) * uvPitch, uvPitch);
            WRITE_CHECK(fwrite(ptrBuf + (pInfo.CropX >> 1), 1, uvWidth, m_fDest.get()), uvWidth);
        }
    } else if (pInfo.FourCC == MFX_FOURCC_NV12) {
        frameSize = lumaWidthBytes * pInfo.CropH * 3 / 2;
        uint32_t uvWidth = pInfo.CropW >> 1;
        //uint32_t nv12Width = pInfo.CropW;
        uint32_t uvHeight = pInfo.CropH >> 1;
        uint32_t uvFrameOffset = ALIGN16(uvWidth * uvHeight + 16);
        if (m_pUVBuffer.get() == nullptr) {
            m_pUVBuffer.reset((uint8_t *)_aligned_malloc(uvFrameOffset << 1, 32));
        }

        alignas(16) static const uint16_t MASK_LOW8[] = {
            0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff
        };
        const __m128i xMaskLow8 = _mm_load_si128((__m128i *)MASK_LOW8);

        for (uint32_t j = 0; j < uvHeight; j++) {
            uint8_t *ptrBuf = m_pReadBuffer.get();
            uint8_t *ptrSrc = pData.UV + (pInfo.CropY + j) * pData.Pitch;
            if (m_bSourceHWMem) {
                loadLineToBuffer(ptrBuf, ptrSrc, pData.Pitch);
            } else {
                ptrBuf = ptrSrc;
            }

            uint8_t *ptrUV = ptrBuf + pInfo.CropX;
            uint8_t *ptrU = m_pUVBuffer.get() + j * uvWidth;
            uint8_t *ptrV = ptrU + uvFrameOffset;
            for (uint32_t i = 0; i < uvWidth; i += 16, ptrUV += 32, ptrU += 16, ptrV += 16) {
                __m128i x0 = _mm_loadu_si128((__m128i *)(ptrUV +  0));
                __m128i x1 = _mm_loadu_si128((__m128i *)(ptrUV + 16));
                _mm_storeu_si128((__m128i *)ptrU, _mm_packus_epi16(_mm_and_si128(x0, xMaskLow8), _mm_and_si128(x1, xMaskLow8)));
                _mm_storeu_si128((__m128i *)ptrV, _mm_packus_epi16(_mm_srli_epi16(x0, 8), _mm_srli_epi16(x1, 8)));
            }
        }
        WRITE_CHECK(fwrite(m_pUVBuffer.get(), 1, uvWidth * uvHeight, m_fDest.get()), uvWidth * uvHeight);
        WRITE_CHECK(fwrite(m_pUVBuffer.get() + uvFrameOffset, 1, uvWidth * uvHeight, m_fDest.get()), uvWidth * uvHeight);
    } else if (pInfo.FourCC == MFX_FOURCC_P010) {
        frameSize = lumaWidthBytes * pInfo.CropH * 3 / 2;
        uint8_t *ptrBuf = m_pReadBuffer.get();
        for (uint32_t i = 0; i < (uint32_t)(pInfo.CropH >> 1); i++) {
            loadLineToBuffer(ptrBuf, pData.UV + pInfo.CropY * (pData.Pitch >> 1) + i * pData.Pitch, pData.Pitch);
            WRITE_CHECK(fwrite(ptrBuf + pInfo.CropX, 1, (uint32_t)pInfo.CropW << 1, m_fDest.get()), (uint32_t)pInfo.CropW << 1);
        }
    } else if (pInfo.FourCC == MFX_FOURCC_RGB4
        || pInfo.FourCC == 100 //DXGI_FORMAT_AYUV
        || pInfo.FourCC == MFX_FOURCC_A2RGB10) {
        frameSize = lumaWidthBytes * pInfo.CropH * 4;
        uint32_t w, h;
        if (pInfo.CropH > 0 && pInfo.CropW > 0) {
            w = pInfo.CropW;
            h = pInfo.CropH;
        } else {
            w = pInfo.Width;
            h = pInfo.Height;
        }

        uint8_t *ptr = (std::min)((std::min)(pData.R, pData.G), pData.B) + pInfo.CropX + pInfo.CropY * pData.Pitch;

        for (uint32_t i = 0; i < h; i++) {
            WRITE_CHECK(fwrite(ptr + i * pData.Pitch, 1, 4*w, m_fDest.get()), 4*w);
        }
    } else {
        return MFX_ERR_UNSUPPORTED;
    }

    m_pEncSatusInfo->SetOutputData(frameSize, (MFX_FRAMETYPE_IDR | MFX_FRAMETYPE_I));
    return MFX_ERR_NONE;
}
