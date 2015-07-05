//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ----------------------------------------------------------------------------------------

#include <immintrin.h>
#include "scene_change_detection.h"
#include "scene_change_detection_simd.h"

void make_hist_avx(const BYTE *frame_Y, hist_t *hist_buf, int y_start, int y_end, int y_step, int x_skip, int width, int pitch) {
    make_hist_simd(frame_Y, hist_buf, y_start, y_end, y_step, x_skip, width, pitch, AVX|POPCNT|SSE42|SSE41|SSSE3|SSE3|SSE2);
}

static DWORD __forceinline get_count(__m256i y0, __m256i y1, __m256i yComp) {
    __m256i y2 = _mm256_cmpeq_epi8(y0, yComp);
    __m256i y3 = _mm256_cmpeq_epi8(y1, yComp);
    DWORD count0 = _mm256_movemask_epi8(y2);
    DWORD count1 = _mm256_movemask_epi8(y3);
#if _M_X64
    return (DWORD)_mm_popcnt_u64(((UINT64)count1 << 32) | (UINT64)count0);
#else
    return _mm_popcnt_u32(count1) + _mm_popcnt_u32(count0);
#endif
}

void make_hist_avx2(const BYTE *frame_Y, hist_t *hist_buf, int y_start, int y_end, int y_step, int x_skip, int width, int pitch) {
    __m256i y0, y1;
    __m256i yOne = _mm256_set1_epi8(1);
    __m256i yMask = _mm256_set1_epi8(0xff>>HIST_LEN_2N);
    hist_t _declspec(align(32)) tmp;
    _mm256_store_si256((__m256i*)((BYTE *)&tmp +  0), _mm256_setzero_si256());
    _mm256_store_si256((__m256i*)((BYTE *)&tmp + 32), _mm256_setzero_si256());

    for (int y = y_start; y < y_end; y += y_step) {
        const BYTE *ptr = (BYTE *)((size_t)(frame_Y + y * pitch + x_skip + 63) & ~63);
        const BYTE *ptr_fin = ptr + (width & ~63) - x_skip;
        __m256i yComp = _mm256_setzero_si256();
#if _M_X64
        __m128i xStepBuf = _mm_setzero_si128();
        struct hist16_t {
            USHORT u[16];
        } _declspec(align(32)) line_sum;
        _mm256_store_si256((__m256i*)((BYTE *)&line_sum +  0), yComp);

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

            xStepBuf = _mm_adds_epu16(xStepBuf, _mm_load_si128((__m128i*)((BYTE *)&line_sum +  0)));
            _mm_store_si128((__m128i*)((BYTE *)&line_sum +  0), xStepBuf);
            
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

            xStepBuf = _mm_adds_epu16(xStepBuf, _mm_load_si128((__m128i*)((BYTE *)&line_sum + 16)));
            _mm_store_si128((__m128i*)((BYTE *)&line_sum + 16), xStepBuf);
        }
        //末端はスキップ
        //ptr_fin = ptr + (width & 31);
        //for ( ; ptr < ptr_fin; ptr += 2) {
        //    line_sum.u[ptr[0]>>HIST_LEN_2N]++;
        //    line_sum.u[ptr[1]>>HIST_LEN_2N]++;
        //}
        y0 = _mm256_load_si256((__m256i*)((BYTE *)&line_sum +  0));
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
        y0 = _mm256_add_epi32(y0, _mm256_load_si256((__m256i*)((BYTE *)&tmp +  0)));
        y1 = _mm256_add_epi32(y1, _mm256_load_si256((__m256i*)((BYTE *)&tmp + 32)));
        _mm256_store_si256((__m256i*)((BYTE *)&tmp +  0), y0);
        _mm256_store_si256((__m256i*)((BYTE *)&tmp + 32), y1);
    }
    _mm256_store_si256((__m256i*)((BYTE *)hist_buf +  0), _mm256_load_si256((__m256i *)((BYTE *)&tmp +  0)));
    _mm256_store_si256((__m256i*)((BYTE *)hist_buf + 32), _mm256_load_si256((__m256i *)((BYTE *)&tmp + 32)));
    _mm256_zeroupper();
#if !_M_X64
    _mm_empty();
#endif
}
