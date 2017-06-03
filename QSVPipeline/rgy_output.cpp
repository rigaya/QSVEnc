// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
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

#include "rgy_output.h"
#include <smmintrin.h>

static RGY_ERR WriteY4MHeader(FILE *fp, const VideoInfo *info) {
    char buffer[256] = { 0 };
    char *ptr = buffer;
    uint32_t len = 0;
    memcpy(ptr, "YUV4MPEG2 ", 10);
    len += 10;

    len += sprintf_s(ptr+len, sizeof(buffer)-len, "W%d H%d ", info->dstWidth, info->dstHeight);
    len += sprintf_s(ptr+len, sizeof(buffer)-len, "F%d:%d ", info->fpsN, info->fpsD);

    const char *picstruct = "Ip ";
    if (info->picstruct & RGY_PICSTRUCT_TFF) {
        picstruct = "It ";
    } else if (info->picstruct & RGY_PICSTRUCT_BFF) {
        picstruct = "Ib ";
    }
    strcpy_s(ptr+len, sizeof(buffer)-len, picstruct); len += 3;
    len += sprintf_s(ptr+len, sizeof(buffer)-len, "A%d:%d ", info->sar[0], info->sar[1]);
    strcpy_s(ptr+len, sizeof(buffer)-len, "C420mpeg2\n"); len += (mfxU32)strlen("C420mpeg2\n");
    return (len == fwrite(buffer, 1, len, fp)) ? RGY_ERR_NONE : RGY_ERR_UNDEFINED_BEHAVIOR;
}

#define WRITE_CHECK(writtenBytes, expected) { \
    if (writtenBytes != expected) { \
        AddMessage(RGY_LOG_ERROR, _T("Error writing file.\nNot enough disk space!\n")); \
        return RGY_ERR_UNDEFINED_BEHAVIOR; \
    } }

RGYOutput::RGYOutput() :
    m_pEncSatusInfo(),
    m_fDest(),
    m_bOutputIsStdout(false),
    m_bInited(false),
    m_bNoOutput(false),
    m_OutType(OUT_TYPE_BITSTREAM),
    m_bSourceHWMem(false),
    m_bY4mHeaderWritten(false),
    m_strWriterName(),
    m_strOutputInfo(),
    m_VideoOutputInfo(),
    m_pPrintMes(),
    m_pOutputBuffer(),
    m_pReadBuffer(),
    m_pUVBuffer() {
    memset(&m_VideoOutputInfo, 0, sizeof(m_VideoOutputInfo));
}

RGYOutput::~RGYOutput() {
    m_pEncSatusInfo.reset();
    m_pPrintMes.reset();
    Close();
}

void RGYOutput::Close() {
    AddMessage(RGY_LOG_DEBUG, _T("Closing...\n"));
    if (m_fDest) {
        m_fDest.reset();
        AddMessage(RGY_LOG_DEBUG, _T("Closed file pointer.\n"));
    }
    m_pEncSatusInfo.reset();
    m_pOutputBuffer.reset();
    m_pReadBuffer.reset();
    m_pUVBuffer.reset();

    m_bNoOutput = false;
    m_bInited = false;
    m_bSourceHWMem = false;
    m_bY4mHeaderWritten = false;
    AddMessage(RGY_LOG_DEBUG, _T("Closed.\n"));
    m_pPrintMes.reset();
}

RGYOutputRaw::RGYOutputRaw() {
    m_strWriterName = _T("bitstream");
    m_OutType = OUT_TYPE_BITSTREAM;
}

RGYOutputRaw::~RGYOutputRaw() {
}

RGY_ERR RGYOutputRaw::Init(const TCHAR *strFileName, const VideoInfo *pVideoOutputInfo, const void *prm) {
    UNREFERENCED_PARAMETER(pVideoOutputInfo);
    RGYOutputRawPrm *rawPrm = (RGYOutputRawPrm *)prm;
    if (!rawPrm->bBenchmark && _tcslen(strFileName) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("output filename not set.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (rawPrm->bBenchmark) {
        m_bNoOutput = true;
        AddMessage(RGY_LOG_DEBUG, _T("no output for benchmark mode.\n"));
    } else {
        if (_tcscmp(strFileName, _T("-")) == 0) {
            m_fDest.reset(stdout);
            m_bOutputIsStdout = true;
            AddMessage(RGY_LOG_DEBUG, _T("using stdout\n"));
        } else {
            CreateDirectoryRecursive(PathRemoveFileSpecFixed(strFileName).second.c_str());
            FILE *fp = NULL;
            int error = _tfopen_s(&fp, strFileName, _T("wb+"));
            if (error != 0 || fp == NULL) {
                AddMessage(RGY_LOG_ERROR, _T("failed to open output file \"%s\": %s\n"), strFileName, _tcserror(error));
                return RGY_ERR_FILE_OPEN;
            }
            m_fDest.reset(fp);
            AddMessage(RGY_LOG_DEBUG, _T("Opened file \"%s\"\n"), strFileName);

            int bufferSizeByte = clamp(rawPrm->nBufSizeMB, 0, RGY_OUTPUT_BUF_MB_MAX) * 1024 * 1024;
            if (bufferSizeByte) {
                void *ptr = nullptr;
                bufferSizeByte = (int)malloc_degeneracy(&ptr, bufferSizeByte, 1024 * 1024);
                if (bufferSizeByte) {
                    m_pOutputBuffer.reset((char*)ptr);
                    setvbuf(m_fDest.get(), m_pOutputBuffer.get(), _IOFBF, bufferSizeByte);
                    AddMessage(RGY_LOG_DEBUG, _T("Added %d MB output buffer.\n"), bufferSizeByte / (1024 * 1024));
                }
            }
        }
    }
    m_bInited = true;
    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputRaw::WriteNextFrame(RGYBitstream *pBitstream) {
    if (pBitstream == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid call: WriteNextFrame\n"));
        return RGY_ERR_NULL_PTR;
    }

    uint32_t nBytesWritten = 0;
    if (!m_bNoOutput) {
        nBytesWritten = (uint32_t)fwrite(pBitstream->data(), 1, pBitstream->size(), m_fDest.get());
        WRITE_CHECK(nBytesWritten, pBitstream->size());
    }

    m_pEncSatusInfo->SetOutputData(pBitstream->frametype(), pBitstream->size(), 0);
    pBitstream->setSize(0);

    return RGY_ERR_NONE;
}

RGY_ERR RGYOutputRaw::WriteNextFrame(RGYFrame *pSurface) {
    UNREFERENCED_PARAMETER(pSurface);
    return RGY_ERR_UNSUPPORTED;
}

CQSVOutFrame::CQSVOutFrame() : m_bY4m(true) {
    m_strWriterName = _T("yuv writer");
    m_OutType = OUT_TYPE_SURFACE;
};

CQSVOutFrame::~CQSVOutFrame() {
};

RGY_ERR CQSVOutFrame::Init(const TCHAR *strFileName, const VideoInfo *pVideoOutputInfo, const void *prm) {
    UNREFERENCED_PARAMETER(pVideoOutputInfo);
    if (_tcscmp(strFileName, _T("-")) == 0) {
        m_fDest.reset(stdout);
        m_bOutputIsStdout = true;
        AddMessage(RGY_LOG_DEBUG, _T("using stdout\n"));
    } else {
        FILE *fp = NULL;
        int error = _tfopen_s(&fp, strFileName, _T("wb"));
        if (0 != error || fp == NULL) {
            AddMessage(RGY_LOG_DEBUG, _T("failed to open file \"%s\": %s\n"), strFileName, _tcserror(error));
            return RGY_ERR_NULL_PTR;
        }
        m_fDest.reset(fp);
    }

    YUVWriterParam *writerParam = (YUVWriterParam *)prm;

    m_bY4m = writerParam->bY4m;
    m_bSourceHWMem = !!(writerParam->memType & (D3D11_MEMORY | D3D9_MEMORY));
    m_bInited = true;

    return RGY_ERR_NONE;
}

RGY_ERR CQSVOutFrame::WriteNextFrame(RGYBitstream *pBitstream) {
    UNREFERENCED_PARAMETER(pBitstream);
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR CQSVOutFrame::WriteNextFrame(RGYFrame *pSurface) {
    if (!m_fDest) {
        return RGY_ERR_NULL_PTR;
    }

    if (m_bSourceHWMem) {
        if (m_pReadBuffer.get() == nullptr) {
            m_pReadBuffer.reset((uint8_t *)_aligned_malloc(pSurface->pitch() + 128, 16));
        }
    }

    if (m_bY4m) {
        if (!m_bY4mHeaderWritten) {
            WriteY4MHeader(m_fDest.get(), &m_VideoOutputInfo);
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

    const uint32_t lumaWidthBytes = pSurface->width() << ((pSurface->csp() == RGY_CSP_P010) ? 1 : 0);
    if (   pSurface->csp() == RGY_CSP_YV12
        || pSurface->csp() == RGY_CSP_NV12
        || pSurface->csp() == RGY_CSP_P010) {
        const uint32_t cropOffset = pSurface->crop().e.up * pSurface->pitch() + pSurface->crop().e.left;
        if (m_bSourceHWMem) {
            for (uint32_t j = 0; j < pSurface->height(); j++) {
                uint8_t *ptrBuf = m_pReadBuffer.get();
                uint8_t *ptrSrc = pSurface->ptrY() + (pSurface->crop().e.up + j) * pSurface->pitch();
                loadLineToBuffer(ptrBuf, ptrSrc, pSurface->pitch());
                WRITE_CHECK(fwrite(ptrBuf + pSurface->crop().e.left, 1, lumaWidthBytes, m_fDest.get()), lumaWidthBytes);
            }
        } else {
            for (uint32_t j = 0; j < pSurface->height(); j++) {
                WRITE_CHECK(fwrite(pSurface->ptrY() + cropOffset + j * pSurface->pitch(), 1, lumaWidthBytes, m_fDest.get()), lumaWidthBytes);
            }
        }
    }

    uint32_t frameSize = 0;
    if (pSurface->csp() == RGY_CSP_YV12) {
        frameSize = lumaWidthBytes * pSurface->height() * 3 / 2;

        uint32_t uvPitch = pSurface->pitch() >> 1;
        uint32_t uvWidth = pSurface->width() >> 1;
        uint32_t uvHeight = pSurface->height() >> 1;
        uint8_t *ptrBuf = m_pReadBuffer.get();
        for (uint32_t i = 0; i < uvHeight; i++) {
            loadLineToBuffer(ptrBuf, pSurface->ptrU() + (pSurface->crop().e.up + i) * uvPitch, uvPitch);
            WRITE_CHECK(fwrite(ptrBuf + (pSurface->crop().e.left >> 1), 1, uvWidth, m_fDest.get()), uvWidth);
        }
        for (uint32_t i = 0; i < uvHeight; i++) {
            loadLineToBuffer(ptrBuf, pSurface->ptrV() + (pSurface->crop().e.up + i) * uvPitch, uvPitch);
            WRITE_CHECK(fwrite(ptrBuf + (pSurface->crop().e.left >> 1), 1, uvWidth, m_fDest.get()), uvWidth);
        }
    } else if (pSurface->csp() == RGY_CSP_NV12) {
        frameSize = lumaWidthBytes * pSurface->height() * 3 / 2;
        uint32_t uvWidth = pSurface->width() >> 1;
        //uint32_t nv12Width = pSurface->width();
        uint32_t uvHeight = pSurface->height() >> 1;
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
            uint8_t *ptrSrc = pSurface->ptrUV() + (pSurface->crop().e.up + j) * pSurface->pitch();
            if (m_bSourceHWMem) {
                loadLineToBuffer(ptrBuf, ptrSrc, pSurface->pitch());
            } else {
                ptrBuf = ptrSrc;
            }

            uint8_t *ptrUV = ptrBuf + pSurface->crop().e.left;
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
    } else if (pSurface->csp() == RGY_CSP_P010) {
        frameSize = lumaWidthBytes * pSurface->height() * 3 / 2;
        uint8_t *ptrBuf = m_pReadBuffer.get();
        for (uint32_t i = 0; i < (uint32_t)(pSurface->height() >> 1); i++) {
            loadLineToBuffer(ptrBuf, pSurface->ptrUV() + pSurface->crop().e.up * (pSurface->pitch() >> 1) + i * pSurface->pitch(), pSurface->pitch());
            WRITE_CHECK(fwrite(ptrBuf + pSurface->crop().e.left, 1, (uint32_t)pSurface->width() << 1, m_fDest.get()), (uint32_t)pSurface->width() << 1);
        }
    } else if (pSurface->csp() == RGY_CSP_RGB4
        || pSurface->csp() == 100 //DXGI_FORMAT_AYUV
        /*|| pSurface->csp() == RGY_CSP_A2RGB10*/) {
        frameSize = lumaWidthBytes * pSurface->height() * 4;
        uint32_t w = pSurface->width();
        uint32_t h = pSurface->height();

        uint8_t *ptr = pSurface->ptrRGB() + pSurface->crop().e.left + pSurface->crop().e.up * pSurface->pitch();

        for (uint32_t i = 0; i < h; i++) {
            WRITE_CHECK(fwrite(ptr + i * pSurface->pitch(), 1, 4*w, m_fDest.get()), 4*w);
        }
    } else {
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }

    m_pEncSatusInfo->SetOutputData(frametype_enc_to_rgy(MFX_FRAMETYPE_IDR | MFX_FRAMETYPE_I), frameSize, 0);
    return RGY_ERR_NONE;
}
