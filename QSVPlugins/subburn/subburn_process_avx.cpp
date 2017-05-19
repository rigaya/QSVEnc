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

#define USE_SSE2    1
#define USE_SSSE3   1
#define PSHUFB_SLOW 0
#define USE_SSE41   1
#define USE_AVX     1
#define USE_AVX2    0
#define USE_FMA3    0
#define USE_POPCNT  0
#include "subburn_process.h"
#include "subburn_process_simd.h"
#if defined(_MSC_VER) || defined(__AVX__)
#if ENABLE_AVSW_READER && ENABLE_LIBASS_SUBBURN

#if _MSC_VER >= 1800 && !defined(__AVX__) && !defined(_DEBUG)
static_assert(false, "do not forget to set /arch:AVX or /arch:AVX2 for this file.");
#endif


ProcessorSubBurnAVX::ProcessorSubBurnAVX() : ProcessorSubBurn() {
}

ProcessorSubBurnAVX::~ProcessorSubBurnAVX() {
}

void ProcessorSubBurnAVX::CopyFrameY() {
    const uint8_t *pFrameSrc = m_pIn->Data.Y;
    uint8_t *pFrameOut = m_pOut->Data.Y;
    const int w = m_pIn->Info.CropW;
    const int h = m_pIn->Info.CropH;
    const int pitch = m_pIn->Data.Pitch;
    for (int y = 0; y < h; y++, pFrameSrc += pitch, pFrameOut += pitch) {
        sse_memcpy(pFrameOut, pFrameSrc, w);
    }
}

void ProcessorSubBurnAVX::CopyFrameUV() {
    const uint8_t *pFrameSrc = m_pIn->Data.UV;
    uint8_t *pFrameOut = m_pOut->Data.UV;
    const int w = m_pIn->Info.CropW;
    const int h = m_pIn->Info.CropH;
    const int pitch = m_pIn->Data.Pitch;
    for (int y = 0; y < h; y += 2, pFrameSrc += pitch, pFrameOut += pitch) {
        sse_memcpy(pFrameOut, pFrameSrc, w);
    }
}

#pragma warning(push)
#pragma warning(disable: 4100)
int ProcessorSubBurnAVX::BlendSubYBitmap(const uint8_t *pSubColorIdx, int nColorLUT, const uint8_t *pSubColor, const uint8_t *pAlpha, int subX, int subY, int subW, int subStride, int subH, uint8_t *pBuf) {
    uint8_t *pFrame = m_pOut->Data.Y;
    const int w = m_pOut->Info.CropW;
    const int h = m_pOut->Info.CropH;
    const int pitch = m_pOut->Data.Pitch;
    subW = (std::min)(w, subX + subW) - subX;
    subH = (std::min)(h, subY + subH) - subY;
    if (nColorLUT <= 16) {
        return blend_sub<false, false, 16>(pFrame, pitch, pSubColorIdx, pSubColor, pAlpha, subX, subY, subW, subStride, subH, nullptr);
    } else if (nColorLUT <= 32) {
        return blend_sub<false, false, 32>(pFrame, pitch, pSubColorIdx, pSubColor, pAlpha, subX, subY, subW, subStride, subH, nullptr);
    } else {
        return blend_sub<false, false, 64>(pFrame, pitch, pSubColorIdx, pSubColor, pAlpha, subX, subY, subW, subStride, subH, nullptr);
    }
}

int ProcessorSubBurnAVX::BlendSubUVBitmap(const uint8_t *pSubColorIdx, int nColorLUT, const uint8_t *pSubColor, const uint8_t *pAlpha, int subX, int subY, int subW, int subStride, int subH, uint8_t *pBuf) {
    uint8_t *pFrame = m_pOut->Data.UV;
    const int w = m_pOut->Info.CropW;
    const int h = m_pOut->Info.CropH;
    const int pitch = m_pOut->Data.Pitch;
    subW = (std::min)(w, subX + subW) - subX;
    subH = (std::min)(h, subY + subH) - subY;
    if (nColorLUT <= 8) {
        return blend_sub<true, false, 8>(pFrame, pitch, pSubColorIdx, pSubColor, pAlpha, subX, subY, subW, subStride, subH, nullptr);
    } else if (nColorLUT <= 16) {
        return blend_sub<true, false, 16>(pFrame, pitch, pSubColorIdx, pSubColor, pAlpha, subX, subY, subW, subStride, subH, nullptr);
    } else if (nColorLUT <= 32) {
        return blend_sub<true, false, 32>(pFrame, pitch, pSubColorIdx, pSubColor, pAlpha, subX, subY, subW, subStride, subH, nullptr);
    } else {
        return blend_sub<true, false, 64>(pFrame, pitch, pSubColorIdx, pSubColor, pAlpha, subX, subY, subW, subStride, subH, nullptr);
    }
}

void ProcessorSubBurnAVX::BlendSubY(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcolory, uint8_t subTransparency, uint8_t *pBuf) {
    uint8_t *pFrame = m_pOut->Data.Y;
    const int w = m_pOut->Info.CropW;
    const int h = m_pOut->Info.CropH;
    const int pitch = m_pOut->Data.Pitch;
    bufW = (std::min)(w, bufX + bufW) - bufX;
    bufH = (std::min)(h, bufY + bufH) - bufY;
    blend_sub<false, false>(pFrame, pitch, pAlpha, bufX, bufY, bufW, bufStride, bufH, subcolory, subcolory, subTransparency, nullptr);
}

void ProcessorSubBurnAVX::BlendSubUV(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcoloru, uint8_t subcolorv, uint8_t subTransparency, uint8_t *pBuf) {
    uint8_t *pFrame = m_pOut->Data.UV;
    const int w = m_pOut->Info.CropW;
    const int h = m_pOut->Info.CropH;
    const int pitch = m_pOut->Data.Pitch;
    bufW = (std::min)(w, bufX + bufW) - bufX;
    bufH = (std::min)(h, bufY + bufH) - bufY;
    blend_sub<true, false>(pFrame, pitch, pAlpha, bufX, bufY, bufW, bufStride, bufH, subcoloru, subcolorv, subTransparency, nullptr);
}
#pragma warning(pop)

ProcessorSubBurnD3DAVX::ProcessorSubBurnD3DAVX() : ProcessorSubBurn() {
}

ProcessorSubBurnD3DAVX::~ProcessorSubBurnD3DAVX() {
}

void ProcessorSubBurnD3DAVX::CopyFrameY() {
}

void ProcessorSubBurnD3DAVX::CopyFrameUV() {
}

int ProcessorSubBurnD3DAVX::BlendSubYBitmap(const uint8_t *pSubColorIdx, int nColorLUT, const uint8_t *pSubColor, const uint8_t *pAlpha, int subX, int subY, int subW, int subStride, int subH, uint8_t *pBuf) {
    uint8_t *pFrame = m_pOut->Data.Y;
    const int w = m_pOut->Info.CropW;
    const int h = m_pOut->Info.CropH;
    const int pitch = m_pOut->Data.Pitch;
    subW = (std::min)(w, subX + subW) - subX;
    subH = (std::min)(h, subY + subH) - subY;
    if (nColorLUT <= 16) {
        return blend_sub<false, true, 16>(pFrame, pitch, pSubColorIdx, pSubColor, pAlpha, subX, subY, subW, subStride, subH, pBuf);
    } else if (nColorLUT <= 32) {
        return blend_sub<false, true, 32>(pFrame, pitch, pSubColorIdx, pSubColor, pAlpha, subX, subY, subW, subStride, subH, pBuf);
    } else {
        return blend_sub<false, true, 64>(pFrame, pitch, pSubColorIdx, pSubColor, pAlpha, subX, subY, subW, subStride, subH, pBuf);
    }
}

int ProcessorSubBurnD3DAVX::BlendSubUVBitmap(const uint8_t *pSubColorIdx, int nColorLUT, const uint8_t *pSubColor, const uint8_t *pAlpha, int subX, int subY, int subW, int subStride, int subH, uint8_t *pBuf) {
    uint8_t *pFrame = m_pOut->Data.UV;
    const int w = m_pOut->Info.CropW;
    const int h = m_pOut->Info.CropH;
    const int pitch = m_pOut->Data.Pitch;
    subW = (std::min)(w, subX + subW) - subX;
    subH = (std::min)(h, subY + subH) - subY;
    if (nColorLUT <= 8) {
        return blend_sub<true, true, 8>(pFrame, pitch, pSubColorIdx, pSubColor, pAlpha, subX, subY, subW, subStride, subH, pBuf);
    } else if (nColorLUT <= 16) {
        return blend_sub<true, true, 16>(pFrame, pitch, pSubColorIdx, pSubColor, pAlpha, subX, subY, subW, subStride, subH, pBuf);
    } else if (nColorLUT <= 32) {
        return blend_sub<true, true, 32>(pFrame, pitch, pSubColorIdx, pSubColor, pAlpha, subX, subY, subW, subStride, subH, pBuf);
    } else {
        return blend_sub<true, true, 64>(pFrame, pitch, pSubColorIdx, pSubColor, pAlpha, subX, subY, subW, subStride, subH, pBuf);
    }
}

void ProcessorSubBurnD3DAVX::BlendSubY(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcolory, uint8_t subTransparency, uint8_t *pBuf) {
    uint8_t *pFrame = m_pOut->Data.Y;
    const int w = m_pOut->Info.CropW;
    const int h = m_pOut->Info.CropH;
    const int pitch = m_pOut->Data.Pitch;
    bufW = (std::min)(w, bufX + bufW) - bufX;
    bufH = (std::min)(h, bufY + bufH) - bufY;
    blend_sub<false, true>(pFrame, pitch, pAlpha, bufX, bufY, bufW, bufStride, bufH, subcolory, subcolory, subTransparency, pBuf);
}

void ProcessorSubBurnD3DAVX::BlendSubUV(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcoloru, uint8_t subcolorv, uint8_t subTransparency, uint8_t *pBuf) {
    uint8_t *pFrame = m_pOut->Data.UV;
    const int w = m_pOut->Info.CropW;
    const int h = m_pOut->Info.CropH;
    const int pitch = m_pOut->Data.Pitch;
    bufW = (std::min)(w, bufX + bufW) - bufX;
    bufH = (std::min)(h, bufY + bufH) - bufY;
    blend_sub<true, true>(pFrame, pitch, pAlpha, bufX, bufY, bufW, bufStride, bufH, subcoloru, subcolorv, subTransparency, pBuf);
}

#endif //#if ENABLE_AVSW_READER && ENABLE_LIBASS_SUBBURN
#endif //#if defined(_MSC_VER) || defined(__AVX__)

