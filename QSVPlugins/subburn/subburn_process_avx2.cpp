//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#define USE_SSE2    1
#define USE_SSSE3   1
#define PSHUFB_SLOW 0
#define USE_SSE41   1
#define USE_AVX     1
#define USE_AVX2    1
#define USE_FMA3    1
#define USE_POPCNT  0
#include "subburn_process.h"
#include "subburn_process_simd.h"
#if ENABLE_AVCODEC_QSV_READER && ENABLE_LIBASS_SUBBURN

#if _MSC_VER >= 1800 && !defined(__AVX__) && !defined(_DEBUG)
static_assert(false, "do not forget to set /arch:AVX or /arch:AVX2 for this file.");
#endif

ProcessorSubBurnAVX2::ProcessorSubBurnAVX2() : ProcessorSubBurn() {
}

ProcessorSubBurnAVX2::~ProcessorSubBurnAVX2() {
}

void ProcessorSubBurnAVX2::CopyFrameY() {
    const uint8_t *pFrameSrc = m_pIn->Data.Y;
    uint8_t *pFrameOut = m_pOut->Data.Y;
    const int w = m_pIn->Info.CropW;
    const int h = m_pIn->Info.CropH;
    const int pitch = m_pIn->Data.Pitch;
    for (int y = 0; y < h; y++, pFrameSrc += pitch, pFrameOut += pitch) {
        const uint8_t *ptr_src = pFrameSrc;
        uint8_t *ptr_dst     = pFrameOut;
        uint8_t *ptr_dst_fin = ptr_dst + (w & ~127);
        for (; ptr_dst < ptr_dst_fin; ptr_dst += 128, ptr_src += 128) {
            __m256i y0 = _mm256_loadu_si256((const __m256i *)(ptr_src +  0));
            __m256i y1 = _mm256_loadu_si256((const __m256i *)(ptr_src + 32));
            __m256i y2 = _mm256_loadu_si256((const __m256i *)(ptr_src + 64));
            __m256i y3 = _mm256_loadu_si256((const __m256i *)(ptr_src + 96));
            _mm256_storeu_si256((__m256i *)(ptr_dst +  0), y0);
            _mm256_storeu_si256((__m256i *)(ptr_dst + 32), y1);
            _mm256_storeu_si256((__m256i *)(ptr_dst + 64), y2);
            _mm256_storeu_si256((__m256i *)(ptr_dst + 96), y3);
        }
        ptr_dst_fin += (w & 127);
        for (; ptr_dst < ptr_dst_fin; ptr_dst += 16, ptr_src += 16) {
            __m128i x0 = _mm_loadu_si128((__m128i *)(ptr_src + 0));
            _mm_storeu_si128((__m128i *)(ptr_dst +  0), x0);
        }
    }
    _mm256_zeroupper();
}

void ProcessorSubBurnAVX2::CopyFrameUV() {
    const uint8_t *pFrameSrc = m_pIn->Data.UV;
    uint8_t *pFrameOut = m_pOut->Data.UV;
    const int w = m_pIn->Info.CropW;
    const int h = m_pIn->Info.CropH;
    const int pitch = m_pIn->Data.Pitch;
    for (int y = 0; y < h; y += 2, pFrameSrc += pitch, pFrameOut += pitch) {
        const uint8_t *ptr_src = pFrameSrc;
        uint8_t *ptr_dst     = pFrameOut;
        uint8_t *ptr_dst_fin = ptr_dst + (w & ~127);
        for (; ptr_dst < ptr_dst_fin; ptr_dst += 128, ptr_src += 128) {
            __m256i y0 = _mm256_loadu_si256((const __m256i *)(ptr_src +  0));
            __m256i y1 = _mm256_loadu_si256((const __m256i *)(ptr_src + 32));
            __m256i y2 = _mm256_loadu_si256((const __m256i *)(ptr_src + 64));
            __m256i y3 = _mm256_loadu_si256((const __m256i *)(ptr_src + 96));
            _mm256_storeu_si256((__m256i *)(ptr_dst +  0), y0);
            _mm256_storeu_si256((__m256i *)(ptr_dst + 32), y1);
            _mm256_storeu_si256((__m256i *)(ptr_dst + 64), y2);
            _mm256_storeu_si256((__m256i *)(ptr_dst + 96), y3);
        }
        ptr_dst_fin += (w & 127);
        for (; ptr_dst < ptr_dst_fin; ptr_dst += 16, ptr_src += 16) {
            __m128i x0 = _mm_loadu_si128((__m128i *)(ptr_src + 0));
            _mm_storeu_si128((__m128i *)(ptr_dst +  0), x0);
        }
    }
    _mm256_zeroupper();
}

void ProcessorSubBurnAVX2::BlendSubY(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcolory, uint8_t subTransparency) {
    uint8_t *pFrame = m_pOut->Data.Y;
    const int w = m_pOut->Info.CropW;
    const int h = m_pOut->Info.CropH;
    const int pitch = m_pOut->Data.Pitch;
    bufW = (std::min)(w, bufX + bufW) - bufX;
    bufH = (std::min)(h, bufY + bufH) - bufY;
    blend_sub<false>(pFrame, pitch, pAlpha, bufX, bufY, bufW, bufStride, bufH, subcolory, subcolory, subTransparency);
    _mm256_zeroupper();
}

void ProcessorSubBurnAVX2::BlendSubUV(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcoloru, uint8_t subcolorv, uint8_t subTransparency) {
    uint8_t *pFrame = m_pOut->Data.UV;
    const int w = m_pOut->Info.CropW;
    const int h = m_pOut->Info.CropH;
    const int pitch = m_pOut->Data.Pitch;
    bufW = (std::min)(w, bufX + bufW) - bufX;
    bufH = (std::min)(h, bufY + bufH) - bufY;
    blend_sub<true>(pFrame, pitch, pAlpha, bufX, bufY, bufW, bufStride, bufH, subcoloru, subcolorv, subTransparency);
    _mm256_zeroupper();
}

#endif //#if ENABLE_AVCODEC_QSV_READER && ENABLE_LIBASS_SUBBURN
