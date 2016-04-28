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
// -------------------------------------------------------------------------------------------

#include <immintrin.h>
#include "scene_change_detection.h"
#include "scene_change_detection_simd.h"

#if _MSC_VER >= 1800 && !defined(__AVX__) && !defined(_DEBUG)
static_assert(false, "do not forget to set /arch:AVX or /arch:AVX2 for this file.");
#endif

#if defined(_MSC_VER) || defined(__AVX__)

void make_hist_avx(const uint8_t *frame_Y, hist_t *hist_buf, int y_start, int y_end, int y_step, int x_skip, int width, int pitch) {
    make_hist_simd(frame_Y, hist_buf, y_start, y_end, y_step, x_skip, width, pitch, AVX|POPCNT|SSE42|SSE41|SSSE3|SSE3|SSE2);
}

static uint32_t __forceinline get_count(__m256i y0, __m256i y1, __m256i yComp) {
    __m256i y2 = _mm256_cmpeq_epi8(y0, yComp);
    __m256i y3 = _mm256_cmpeq_epi8(y1, yComp);
    uint32_t count0 = _mm256_movemask_epi8(y2);
    uint32_t count1 = _mm256_movemask_epi8(y3);
#if _M_X64
    return (uint32_t)_mm_popcnt_u64(((uint64_t)count1 << 32) | (uint64_t)count0);
#else
    return _mm_popcnt_u32(count1) + _mm_popcnt_u32(count0);
#endif
}

void make_hist_avx2(const uint8_t *frame_Y, hist_t *hist_buf, int y_start, int y_end, int y_step, int x_skip, int width, int pitch) {
    __m256i y0, y1;
    __m256i yOne = _mm256_set1_epi8(1);
    __m256i yMask = _mm256_set1_epi8(0xff>>HIST_LEN_2N);
    alignas(32) hist_t tmp;
    _mm256_store_si256((__m256i*)((uint8_t *)&tmp +  0), _mm256_setzero_si256());
    _mm256_store_si256((__m256i*)((uint8_t *)&tmp + 32), _mm256_setzero_si256());

    for (int y = y_start; y < y_end; y += y_step) {
        const uint8_t *ptr = (uint8_t *)((size_t)(frame_Y + y * pitch + x_skip + 63) & ~63);
        const uint8_t *ptr_fin = ptr + (width & ~63) - x_skip;
        __m256i yComp = _mm256_setzero_si256();
#if _M_X64
        __m128i xStepBuf = _mm_setzero_si128();
        struct alignas(32) hist16_t {
            USHORT u[16];
        } line_sum;
        _mm256_store_si256((__m256i*)((uint8_t *)&line_sum +  0), yComp);

        for ( ; ptr < ptr_fin; ptr += 32) {
            yComp = _mm256_xor_si256(yComp, yComp);
            y0 = _mm256_stream_load_si256((__m256i*)(ptr +  0));
            y1 = _mm256_stream_load_si256((__m256i*)(ptr + 32));

            y0 = _mm256_srli_epi64(y0, HIST_LEN_2N);
            y1 = _mm256_srli_epi64(y1, HIST_LEN_2N);

            y0 = _mm256_and_si256(y0, yMask);
            y1 = _mm256_and_si256(y1, yMask);

            xStepBuf = _mm_insert_epi16(xStepBuf, get_count(y0, y1, yComp), 0);
            yComp = _mm256_add_epi8(yComp, yOne);
            xStepBuf = _mm_insert_epi16(xStepBuf, get_count(y0, y1, yComp), 1);
            yComp = _mm256_add_epi8(yComp, yOne);
            xStepBuf = _mm_insert_epi16(xStepBuf, get_count(y0, y1, yComp), 2);
            yComp = _mm256_add_epi8(yComp, yOne);
            xStepBuf = _mm_insert_epi16(xStepBuf, get_count(y0, y1, yComp), 3);
            yComp = _mm256_add_epi8(yComp, yOne);
            xStepBuf = _mm_insert_epi16(xStepBuf, get_count(y0, y1, yComp), 4);
            yComp = _mm256_add_epi8(yComp, yOne);
            xStepBuf = _mm_insert_epi16(xStepBuf, get_count(y0, y1, yComp), 5);
            yComp = _mm256_add_epi8(yComp, yOne);
            xStepBuf = _mm_insert_epi16(xStepBuf, get_count(y0, y1, yComp), 6);
            yComp = _mm256_add_epi8(yComp, yOne);
            xStepBuf = _mm_insert_epi16(xStepBuf, get_count(y0, y1, yComp), 7);
            yComp = _mm256_add_epi8(yComp, yOne);

            xStepBuf = _mm_adds_epu16(xStepBuf, _mm_load_si128((__m128i*)((uint8_t *)&line_sum +  0)));
            _mm_store_si128((__m128i*)((uint8_t *)&line_sum +  0), xStepBuf);
            
            xStepBuf = _mm_insert_epi16(xStepBuf, get_count(y0, y1, yComp), 0);
            yComp = _mm256_add_epi8(yComp, yOne);
            xStepBuf = _mm_insert_epi16(xStepBuf, get_count(y0, y1, yComp), 1);
            yComp = _mm256_add_epi8(yComp, yOne);
            xStepBuf = _mm_insert_epi16(xStepBuf, get_count(y0, y1, yComp), 2);
            yComp = _mm256_add_epi8(yComp, yOne);
            xStepBuf = _mm_insert_epi16(xStepBuf, get_count(y0, y1, yComp), 3);
            yComp = _mm256_add_epi8(yComp, yOne);
            xStepBuf = _mm_insert_epi16(xStepBuf, get_count(y0, y1, yComp), 4);
            yComp = _mm256_add_epi8(yComp, yOne);
            xStepBuf = _mm_insert_epi16(xStepBuf, get_count(y0, y1, yComp), 5);
            yComp = _mm256_add_epi8(yComp, yOne);
            xStepBuf = _mm_insert_epi16(xStepBuf, get_count(y0, y1, yComp), 6);
            yComp = _mm256_add_epi8(yComp, yOne);
            xStepBuf = _mm_insert_epi16(xStepBuf, get_count(y0, y1, yComp), 7);
            yComp = _mm256_add_epi8(yComp, yOne);

            xStepBuf = _mm_adds_epu16(xStepBuf, _mm_load_si128((__m128i*)((uint8_t *)&line_sum + 16)));
            _mm_store_si128((__m128i*)((uint8_t *)&line_sum + 16), xStepBuf);
        }
        //末端はスキップ
        //ptr_fin = ptr + (width & 31);
        //for ( ; ptr < ptr_fin; ptr += 2) {
        //    line_sum.u[ptr[0]>>HIST_LEN_2N]++;
        //    line_sum.u[ptr[1]>>HIST_LEN_2N]++;
        //}
        y0 = _mm256_load_si256((__m256i*)((uint8_t *)&line_sum +  0));
#else
        __m64 m0 = _mm_setzero_si64();
        __m64 m1 = _mm_setzero_si64();
        __m64 m2 = _mm_setzero_si64();
        __m64 m3 = _mm_setzero_si64();
        __m64 m4 = _mm_setzero_si64();
        __m64 m5 = _mm_setzero_si64();
        __m64 m6 = _mm_setzero_si64();
        __m64 m7 = _mm_setzero_si64();

        for ( ; ptr < ptr_fin; ptr += 64) {

            yComp = _mm256_xor_si256(yComp, yComp);
            y0 = _mm256_stream_load_si256((__m256i*)(ptr +  0));
            y1 = _mm256_stream_load_si256((__m256i*)(ptr + 32));

            y0 = _mm256_srli_epi64(y0, HIST_LEN_2N);
            y1 = _mm256_srli_epi64(y1, HIST_LEN_2N);

            y0 = _mm256_and_si256(y0, yMask);
            y1 = _mm256_and_si256(y1, yMask);
            
            m0 = _mm_insert_pi16(m0, get_count(y0, y1, yComp), 0);
            yComp = _mm256_add_epi8(yComp, yOne);
            m0 = _mm_insert_pi16(m0, get_count(y0, y1, yComp), 1);
            yComp = _mm256_add_epi8(yComp, yOne);
            m0 = _mm_insert_pi16(m0, get_count(y0, y1, yComp), 2);
            yComp = _mm256_add_epi8(yComp, yOne);
            m0 = _mm_insert_pi16(m0, get_count(y0, y1, yComp), 3);
            yComp = _mm256_add_epi8(yComp, yOne);

            m1 = _mm_insert_pi16(m1, get_count(y0, y1, yComp), 0);
            yComp = _mm256_add_epi8(yComp, yOne);
            m1 = _mm_insert_pi16(m1, get_count(y0, y1, yComp), 1);
            yComp = _mm256_add_epi8(yComp, yOne);
            m1 = _mm_insert_pi16(m1, get_count(y0, y1, yComp), 2);
            yComp = _mm256_add_epi8(yComp, yOne);
            m1 = _mm_insert_pi16(m1, get_count(y0, y1, yComp), 3);
            yComp = _mm256_add_epi8(yComp, yOne);

            m2 = _mm_insert_pi16(m2, get_count(y0, y1, yComp), 0);
            yComp = _mm256_add_epi8(yComp, yOne);
            m2 = _mm_insert_pi16(m2, get_count(y0, y1, yComp), 1);
            yComp = _mm256_add_epi8(yComp, yOne);
            m2 = _mm_insert_pi16(m2, get_count(y0, y1, yComp), 2);
            yComp = _mm256_add_epi8(yComp, yOne);
            m2 = _mm_insert_pi16(m2, get_count(y0, y1, yComp), 3);
            yComp = _mm256_add_epi8(yComp, yOne);
            
            m3 = _mm_insert_pi16(m3, get_count(y0, y1, yComp), 0);
            yComp = _mm256_add_epi8(yComp, yOne);
            m3 = _mm_insert_pi16(m3, get_count(y0, y1, yComp), 1);
            yComp = _mm256_add_epi8(yComp, yOne);
            m3 = _mm_insert_pi16(m3, get_count(y0, y1, yComp), 2);
            yComp = _mm256_add_epi8(yComp, yOne);
            m3 = _mm_insert_pi16(m3, get_count(y0, y1, yComp), 3);

            m4 = _mm_adds_pi16(m4, m0);
            m5 = _mm_adds_pi16(m5, m1);
            m6 = _mm_adds_pi16(m6, m2);
            m7 = _mm_adds_pi16(m7, m3);
        }
        //末端はスキップ
        //ptr_fin = ptr + (width & 31);
        //for ( ; ptr < ptr_fin; ptr += 2) {
        //    line_sum.u[ptr[0]>>HIST_LEN_2N]++;
        //    line_sum.u[ptr[1]>>HIST_LEN_2N]++;
        //}
        __m128i x3 = _mm_movpi64_epi64(m7);
        __m128i x2 = _mm_movpi64_epi64(m6);
        __m128i x1 = _mm_movpi64_epi64(m5);
        __m128i x0 = _mm_movpi64_epi64(m4);
        x2 = _mm_or_si128(x2, _mm_slli_si128(x3, 8));
        x0 = _mm_or_si128(x0, _mm_slli_si128(x1, 8));
        y0 = _mm256_inserti128_si256(_mm256_castsi128_si256(x0), x2, 1);
#endif
        y1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(y0, 1));
        y0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y0));
        y0 = _mm256_add_epi32(y0, _mm256_load_si256((__m256i*)((uint8_t *)&tmp +  0)));
        y1 = _mm256_add_epi32(y1, _mm256_load_si256((__m256i*)((uint8_t *)&tmp + 32)));
        _mm256_store_si256((__m256i*)((uint8_t *)&tmp +  0), y0);
        _mm256_store_si256((__m256i*)((uint8_t *)&tmp + 32), y1);
    }
    _mm256_store_si256((__m256i*)((uint8_t *)hist_buf +  0), _mm256_load_si256((__m256i *)((uint8_t *)&tmp +  0)));
    _mm256_store_si256((__m256i*)((uint8_t *)hist_buf + 32), _mm256_load_si256((__m256i *)((uint8_t *)&tmp + 32)));
    _mm256_zeroupper();
#if !_M_X64
    _mm_empty();
#endif
}

#endif //#if defined(_MSC_VER) || defined(__AVX__)

