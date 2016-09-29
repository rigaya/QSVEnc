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

#pragma once
#include <cstdint>
#include "mfxdefs.h"
#include "plugin_delogo.h"
#include <emmintrin.h> //SSE2
#if USE_SSSE3
#include <tmmintrin.h> //SSSE3
#endif
#if USE_SSE41
#include <smmintrin.h>
#endif
#if USE_AVX
#include <immintrin.h>
#endif
#include "qsv_simd.h"

#if USE_AVX2
#define MEM_ALIGN 32
#else
#define MEM_ALIGN 16
#endif

//SSSE3のpalignrもどき
#define palignr_sse2(a,b,i) _mm_or_si128( _mm_slli_si128(a, 16-i), _mm_srli_si128(b, i) )

#if USE_SSSE3
#define _mm_alignr_epi8_simd _mm_alignr_epi8
#else
#define _mm_alignr_epi8_simd palignr_sse2
#endif

#if USE_SSE41
#define _mm_blendv_epi8_simd _mm_blendv_epi8
#else
static inline __m128i select_by_mask(__m128i a, __m128i b, __m128i mask) {
    return _mm_or_si128( _mm_andnot_si128(mask,a), _mm_and_si128(b,mask) );
}
#define _mm_blendv_epi8_simd select_by_mask
#endif

alignas(MEM_ALIGN) static const uint16_t MASK_16BIT[] = {
    0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000,
#if USE_AVX2
    0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000
#endif
};

#if USE_AVX2
//本来の256bit alignr
#define MM_ABS(x) (((x) < 0) ? -(x) : (x))
#define _mm256_alignr256_epi8(a, b, i) ((i<=16) ? _mm256_alignr_epi8(_mm256_permute2x128_si256(a, b, (0x00<<4) + 0x03), b, i) : _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, b, (0x00<<4) + 0x03), MM_ABS(i-16)))

//_mm256_srli_si256, _mm256_slli_si256は
//単に128bitシフト×2をするだけの命令である
#ifndef _mm256_bsrli_epi128
#define _mm256_bsrli_epi128 _mm256_srli_si256
#endif
#ifndef _mm256_bslli_epi128
#define _mm256_bslli_epi128 _mm256_slli_si256
#endif
//本当の256bitシフト
#define _mm256_srli256_si256(a, i) ((i<=16) ? _mm256_alignr_epi8(_mm256_permute2x128_si256(a, a, (0x08<<4) + 0x03), a, i) : _mm256_bsrli_epi128(_mm256_permute2x128_si256(a, a, (0x08<<4) + 0x03), MM_ABS(i-16)))
#define _mm256_slli256_si256(a, i) ((i<=16) ? _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, a, (0x00<<4) + 0x08), MM_ABS(16-i)) : _mm256_bslli_epi128(_mm256_permute2x128_si256(a, a, (0x00<<4) + 0x08), MM_ABS(i-16)))


alignas(MEM_ALIGN) static const unsigned int ARRAY_0x00008000[8] = {
    0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000,
};
static __forceinline __m256i cvtlo256_epi16_epi32(__m256i y0) {
    __m256i yWordsHi = _mm256_cmpgt_epi16(_mm256_setzero_si256(), y0);
    return _mm256_unpacklo_epi16(y0, yWordsHi);
}

static __forceinline __m256i cvthi256_epi16_epi32(__m256i y0) {
    __m256i yWordsHi = _mm256_cmpgt_epi16(_mm256_setzero_si256(), y0);
    return _mm256_unpackhi_epi16(y0, yWordsHi);
}

static __forceinline __m256i _mm256_neg_epi32(__m256i y) {
    return _mm256_sub_epi32(_mm256_setzero_si256(), y);
}
static __forceinline __m256i _mm256_neg_epi16(__m256i y) {
    return _mm256_sub_epi16(_mm256_setzero_si256(), y);
}
static __forceinline __m256 _mm256_rcp_ps_hp(__m256 y0) {
    __m256 y1, y2;
    y1 = _mm256_rcp_ps(y0);
    y0 = _mm256_mul_ps(y0, y1);
    y2 = _mm256_add_ps(y1, y1);
#if USE_FMA3
    y2 = _mm256_fnmadd_ps(y0, y1, y2);
#else
    y0 = _mm256_mul_ps(y0, y1);
    y2 = _mm256_sub_ps(y2, y0);
#endif
    return y2;
}

#elif USE_SSE2
alignas(MEM_ALIGN) static const unsigned int ARRAY_0x00008000[4] = {
    0x00008000, 0x00008000, 0x00008000, 0x00008000,
};
static __forceinline __m128i _mm_neg_epi32(__m128i y) {
    return _mm_sub_epi32(_mm_setzero_si128(), y);
}
static __forceinline __m128i _mm_neg_epi16(__m128i y) {
    return _mm_sub_epi16(_mm_setzero_si128(), y);
}
static __forceinline __m128 _mm_rcp_ps_hp(__m128 x0) {
    __m128 x1, x2;
    x1 = _mm_rcp_ps(x0);
    x0 = _mm_mul_ps(x0, x1);
    x2 = _mm_add_ps(x1, x1);
    x0 = _mm_mul_ps(x0, x1);
    x2 = _mm_sub_ps(x2, x0);
    return x2;
}

static __forceinline __m128i _mm_packus_epi32_simd(__m128i a, __m128i b) {
#if USE_SSE41
    return _mm_packus_epi32(a, b);
#else
    alignas(64) static const DWORD VAL[2][4] = {
        { 0x00008000, 0x00008000, 0x00008000, 0x00008000 },
        { 0x80008000, 0x80008000, 0x80008000, 0x80008000 }
    };
#define LOAD_32BIT_0x8000 _mm_load_si128((__m128i *)VAL[0])
#define LOAD_16BIT_0x8000 _mm_load_si128((__m128i *)VAL[1])
    a = _mm_sub_epi32(a, LOAD_32BIT_0x8000);
    b = _mm_sub_epi32(b, LOAD_32BIT_0x8000);
    a = _mm_packs_epi32(a, b);
    return _mm_add_epi16(a, LOAD_16BIT_0x8000);
#undef LOAD_32BIT_0x8000
#undef LOAD_16BIT_0x8000
#endif
}


static __forceinline __m128i _mm_mullo_epi32_simd(__m128i x0, __m128i x1) {
#if USE_SSE41
    return _mm_mullo_epi32(x0, x1);
#else
    __m128i x2 = _mm_mul_epu32(x0, x1);
    __m128i x3 = _mm_mul_epu32(_mm_shuffle_epi32(x0, 0xB1), _mm_shuffle_epi32(x1, 0xB1));
    
    x2 = _mm_shuffle_epi32(x2, 0xD8);
    x3 = _mm_shuffle_epi32(x3, 0xD8);
    
    return _mm_unpacklo_epi32(x2, x3);
#endif
}

static __forceinline __m128i cvtlo_epi16_epi32(__m128i x0) {
#if USE_SSE41
    return _mm_cvtepi16_epi32(x0);
#else
    __m128i xWordsHi = _mm_cmpgt_epi16(_mm_setzero_si128(), x0);
    return _mm_unpacklo_epi16(x0, xWordsHi);
#endif
}

static __forceinline __m128i cvthi_epi16_epi32(__m128i x0) {
#if USE_SSE41
    return _mm_cvtepi16_epi32(_mm_srli_si128(x0, 8));
#else
    __m128i xWordsHi = _mm_cmpgt_epi16(_mm_setzero_si128(), x0);
    return _mm_unpackhi_epi16(x0, xWordsHi);
#endif
}

static __forceinline __m128i blendv_epi8_simd(__m128i a, __m128i b, __m128i mask) {
#if USE_SSE41
    return _mm_blendv_epi8(a, b, mask);
#else
    return _mm_or_si128( _mm_andnot_si128(mask,a), _mm_and_si128(b,mask) );
#endif
}
#endif

template<mfxU32 step, bool ignore_fraction>
static __forceinline void load_line_to_buffer(mfxU8 *buffer, mfxU8 *src, mfxU32 width) {
#if USE_AVX
    static_assert(step % 32 == 0, "step should be mod32.");
#else
    static_assert(step % 16 == 0, "step should be mod16.");
#endif
#ifdef _M_IX86
#define UNROLL_64BIT (0)
#else
#define UNROLL_64BIT (1)
#endif
    const bool use_avx2 = USE_AVX2 && (0 == ((size_t)src & 0x10));
    const mfxU32 align = ((use_avx2) ? 32 : 16);
    const mfxU32 increment = (std::min)(step, ((use_avx2 || UNROLL_64BIT) ? 256u : 128u));
    mfxU8 *src_fin = src + ((increment == align || ignore_fraction) ? width : (width & ~(increment-1)));
    mfxU8 *src_ptr = src, *buf_ptr = buffer;
#if USE_AVX2
    if (!use_avx2) {
#endif
        for (; src_ptr < src_fin; src_ptr += increment, buf_ptr += increment) {
            __m128i x0, x1, x2, x3, x4, x5, x6, x7;
            if (step >=  16) x0  = _mm_stream_load_si128((__m128i *)(src_ptr +   0));
            if (step >=  32) x1  = _mm_stream_load_si128((__m128i *)(src_ptr +  16));
            if (step >=  48) x2  = _mm_stream_load_si128((__m128i *)(src_ptr +  32));
            if (step >=  64) x3  = _mm_stream_load_si128((__m128i *)(src_ptr +  48));
            if (step >=  80) x4  = _mm_stream_load_si128((__m128i *)(src_ptr +  64));
            if (step >=  96) x5  = _mm_stream_load_si128((__m128i *)(src_ptr +  80));
            if (step >= 112) x6  = _mm_stream_load_si128((__m128i *)(src_ptr +  96));
            if (step >= 128) x7  = _mm_stream_load_si128((__m128i *)(src_ptr + 112));
    #if UNROLL_64BIT
            __m128i x8, x9, x10, x11, x12, x13, x14, x15;
            if (step >= 144) x8  = _mm_stream_load_si128((__m128i *)(src_ptr + 128));
            if (step >= 160) x9  = _mm_stream_load_si128((__m128i *)(src_ptr + 144));
            if (step >= 176) x10 = _mm_stream_load_si128((__m128i *)(src_ptr + 160));
            if (step >= 192) x11 = _mm_stream_load_si128((__m128i *)(src_ptr + 176));
            if (step >= 208) x12 = _mm_stream_load_si128((__m128i *)(src_ptr + 192));
            if (step >= 224) x13 = _mm_stream_load_si128((__m128i *)(src_ptr + 208));
            if (step >= 240) x14 = _mm_stream_load_si128((__m128i *)(src_ptr + 224));
            if (step >= 256) x15 = _mm_stream_load_si128((__m128i *)(src_ptr + 240));
    #endif //UNROLL_64BIT
            if (step >=  16) _mm_store_si128((__m128i *)(buf_ptr +   0), x0);
            if (step >=  32) _mm_store_si128((__m128i *)(buf_ptr +  16), x1);
            if (step >=  48) _mm_store_si128((__m128i *)(buf_ptr +  32), x2);
            if (step >=  64) _mm_store_si128((__m128i *)(buf_ptr +  48), x3);
            if (step >=  80) _mm_store_si128((__m128i *)(buf_ptr +  64), x4);
            if (step >=  96) _mm_store_si128((__m128i *)(buf_ptr +  80), x5);
            if (step >= 112) _mm_store_si128((__m128i *)(buf_ptr +  96), x6);
            if (step >= 128) _mm_store_si128((__m128i *)(buf_ptr + 112), x7);
    #if UNROLL_64BIT
            if (step >= 128) _mm_store_si128((__m128i *)(buf_ptr + 128), x8);
            if (step >= 144) _mm_store_si128((__m128i *)(buf_ptr + 144), x9);
            if (step >= 160) _mm_store_si128((__m128i *)(buf_ptr + 160), x10);
            if (step >= 176) _mm_store_si128((__m128i *)(buf_ptr + 176), x11);
            if (step >= 192) _mm_store_si128((__m128i *)(buf_ptr + 192), x12);
            if (step >= 208) _mm_store_si128((__m128i *)(buf_ptr + 208), x13);
            if (step >= 224) _mm_store_si128((__m128i *)(buf_ptr + 224), x14);
            if (step >= 240) _mm_store_si128((__m128i *)(buf_ptr + 240), x15);
    #endif //UNROLL_64BIT
        }
#if USE_AVX2
    } else {
        for (; src_ptr < src_fin; src_ptr += increment, buf_ptr += increment) {
            __m256i y0, y1, y2, y3, y4, y5, y6, y7;
            if (step >=  32) y0 = _mm256_stream_load_si256((__m256i *)(src_ptr +   0));
            if (step >=  64) y1 = _mm256_stream_load_si256((__m256i *)(src_ptr +  32));
            if (step >=  96) y2 = _mm256_stream_load_si256((__m256i *)(src_ptr +  64));
            if (step >= 128) y3 = _mm256_stream_load_si256((__m256i *)(src_ptr +  96));
            if (step >= 160) y4 = _mm256_stream_load_si256((__m256i *)(src_ptr + 128));
            if (step >= 192) y5 = _mm256_stream_load_si256((__m256i *)(src_ptr + 160));
            if (step >= 224) y6 = _mm256_stream_load_si256((__m256i *)(src_ptr + 192));
            if (step >= 256) y7 = _mm256_stream_load_si256((__m256i *)(src_ptr + 224));
            if (step >=  32) _mm256_store_si256((__m256i *)(buf_ptr +   0), y0);
            if (step >=  64) _mm256_store_si256((__m256i *)(buf_ptr +  32), y1);
            if (step >=  96) _mm256_store_si256((__m256i *)(buf_ptr +  64), y2);
            if (step >= 128) _mm256_store_si256((__m256i *)(buf_ptr +  96), y3);
            if (step >= 160) _mm256_store_si256((__m256i *)(buf_ptr + 128), y4);
            if (step >= 192) _mm256_store_si256((__m256i *)(buf_ptr + 160), y5);
            if (step >= 224) _mm256_store_si256((__m256i *)(buf_ptr + 192), y6);
            if (step >= 256) _mm256_store_si256((__m256i *)(buf_ptr + 224), y7);
        }
    }
#endif

    if (!(ignore_fraction || increment == align)) {
        src_fin += width & (increment-1);
        for (; src_ptr < src_fin; src_ptr += 16, buf_ptr += 16) {
            __m128i x0  = _mm_stream_load_si128((__m128i *)(src_ptr));
            _mm_store_si128((__m128i *)(buf_ptr), x0);
        }
    }
}

template<mfxU32 step, bool ignore_fraction>
static __forceinline void store_line_from_buffer(mfxU8 *dst, mfxU8 *buffer, mfxU32 width) {
#if USE_AVX
    static_assert(step % 32 == 0, "step should be mod32.");
#else
    static_assert(step % 16 == 0, "step should be mod16.");
#endif
    const bool use_avx = USE_AVX && (0 == ((size_t)dst & 0x10));
    const mfxU32 align = ((use_avx) ? 32 : 16);
    const mfxU32 increment = (std::min)(step, ((use_avx) ? 256u : 128u));
    mfxU8 *dst_fin = dst + ((increment == align || ignore_fraction) ? width : (width & ~(increment-1)));
    mfxU8 *dst_ptr = dst, *buf_ptr = buffer;
#if USE_AVX
    if (!use_avx) {
#endif
        for (; dst_ptr < dst_fin; dst_ptr += increment, buf_ptr += increment) {
            __m128i x0, x1, x2, x3, x4, x5, x6, x7;
            if (step >=  16) x0 = _mm_load_si128((__m128i *)(buf_ptr +   0));
            if (step >=  32) x1 = _mm_load_si128((__m128i *)(buf_ptr +  16));
            if (step >=  48) x2 = _mm_load_si128((__m128i *)(buf_ptr +  32));
            if (step >=  64) x3 = _mm_load_si128((__m128i *)(buf_ptr +  48));
            if (step >=  80) x4 = _mm_load_si128((__m128i *)(buf_ptr +  64));
            if (step >=  96) x5 = _mm_load_si128((__m128i *)(buf_ptr +  80));
            if (step >= 112) x6 = _mm_load_si128((__m128i *)(buf_ptr +  96));
            if (step >= 128) x7 = _mm_load_si128((__m128i *)(buf_ptr + 112));
            if (step >=  16) _mm_store_si128((__m128i *)(dst_ptr +   0), x0);
            if (step >=  32) _mm_store_si128((__m128i *)(dst_ptr +  16), x1);
            if (step >=  48) _mm_store_si128((__m128i *)(dst_ptr +  32), x2);
            if (step >=  64) _mm_store_si128((__m128i *)(dst_ptr +  48), x3);
            if (step >=  80) _mm_store_si128((__m128i *)(dst_ptr +  64), x4);
            if (step >=  96) _mm_store_si128((__m128i *)(dst_ptr +  80), x5);
            if (step >= 112) _mm_store_si128((__m128i *)(dst_ptr +  96), x6);
            if (step >= 128) _mm_store_si128((__m128i *)(dst_ptr + 112), x7);
        }
#if USE_AVX
    } else {
        for (; dst_ptr < dst_fin; dst_ptr += increment, buf_ptr += increment) {
            __m256 y0, y1, y2, y3, y4, y5, y6, y7;
            if (step >=  32) y0 = _mm256_loadu_ps((float *)(buf_ptr +   0));
            if (step >=  64) y1 = _mm256_loadu_ps((float *)(buf_ptr +  32));
            if (step >=  96) y2 = _mm256_loadu_ps((float *)(buf_ptr +  64));
            if (step >= 128) y3 = _mm256_loadu_ps((float *)(buf_ptr +  96));
            if (step >= 160) y4 = _mm256_loadu_ps((float *)(buf_ptr + 128));
            if (step >= 192) y5 = _mm256_loadu_ps((float *)(buf_ptr + 160));
            if (step >= 224) y6 = _mm256_loadu_ps((float *)(buf_ptr + 192));
            if (step >= 256) y7 = _mm256_loadu_ps((float *)(buf_ptr + 224));
            if (step >=  32) _mm256_store_ps((float *)(dst_ptr +   0), y0);
            if (step >=  64) _mm256_store_ps((float *)(dst_ptr +  32), y1);
            if (step >=  96) _mm256_store_ps((float *)(dst_ptr +  64), y2);
            if (step >= 128) _mm256_store_ps((float *)(dst_ptr +  96), y3);
            if (step >= 160) _mm256_store_ps((float *)(dst_ptr + 128), y4);
            if (step >= 192) _mm256_store_ps((float *)(dst_ptr + 160), y5);
            if (step >= 224) _mm256_store_ps((float *)(dst_ptr + 192), y6);
            if (step >= 256) _mm256_store_ps((float *)(dst_ptr + 224), y7);
        }
    }
#endif
    if (!(ignore_fraction || increment == align)) {
        dst_fin += width & (increment-1);
        for (; dst_ptr < dst_fin; dst_ptr += 16, buf_ptr += 16) {
            __m128i x0 = _mm_load_si128((__m128i *)(buf_ptr));
            _mm_store_si128((__m128i *)(dst_ptr), x0);
        }
    }
}


#if USE_AVX2
#define CONST_M const __m256i
#define CONST_MR(x) const __m256i& yC_ ## x
#define SET_EPI16 _mm256_set1_epi16
#define SET_EPI32 _mm256_set1_epi32
#else //#if USE_AVX2
#define CONST_MR(x) const __m128i& xC_ ## x
#define CONST_M const __m128i
#define SET_EPI16 _mm_set1_epi16
#define SET_EPI32 _mm_set1_epi32
#endif //#if USE_AVX2

#define DEPTH_MUL_OPTIM (1)
#define USE_SIMPLE_RCPPS (1)
#if USE_SIMPLE_RCPPS
#define delogo_rcpps256 _mm256_rcp_ps
#define delogo_rcpps    _mm_rcp_ps
#else
#define delogo_rcpps256 _mm256_rcp_ps_hp
#define delogo_rcpps    _mm_rcp_ps_hp
#endif

static __forceinline void delogo_line(mfxU8 *ptr_buf, short *ptr_logo, int logo_i_width,
    CONST_MR(nv12_2_yc48_mul), CONST_MR(nv12_2_yc48_sub), CONST_MR(yc48_2_nv12_mul), CONST_MR(yc48_2_nv12_add),
    CONST_MR(offset), CONST_MR(depth_mul_fade_slft_3)) {
    mfxU8 *ptr_buf_fin = ptr_buf + logo_i_width;
#if USE_AVX2
    for (; ptr_buf < ptr_buf_fin; ptr_buf += 32, ptr_logo += 64) {
        __m256i y0, y1, y2, y3;
        __m256i yDp0, yDp1, yDp2, yDp3;
        __m256i ySrc0, ySrc1;
        y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i *)(ptr_logo + 32)), _mm_loadu_si128((__m128i *)(ptr_logo +  0)));
        y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i *)(ptr_logo + 40)), _mm_loadu_si128((__m128i *)(ptr_logo +  8)));
        y2 = _mm256_set_m128i(_mm_loadu_si128((__m128i *)(ptr_logo + 48)), _mm_loadu_si128((__m128i *)(ptr_logo + 16)));
        y3 = _mm256_set_m128i(_mm_loadu_si128((__m128i *)(ptr_logo + 56)), _mm_loadu_si128((__m128i *)(ptr_logo + 24)));

        // 不透明度情報のみ取り出し
        yDp0 = _mm256_and_si256(y0, _mm256_load_si256((__m256i *)MASK_16BIT));
        yDp1 = _mm256_and_si256(y1, _mm256_load_si256((__m256i *)MASK_16BIT));
        yDp2 = _mm256_and_si256(y2, _mm256_load_si256((__m256i *)MASK_16BIT));
        yDp3 = _mm256_and_si256(y3, _mm256_load_si256((__m256i *)MASK_16BIT));
        //ロゴ色データの取り出し
        y0 = _mm256_packs_epi32(_mm256_srai_epi32(y0, 16), _mm256_srai_epi32(y1, 16)); // lgp->yの抽出
        y1 = _mm256_packs_epi32(_mm256_srai_epi32(y2, 16), _mm256_srai_epi32(y3, 16));

        y0 = _mm256_add_epi16(y0, yC_offset); //lgp->y + py_offset
        y1 = _mm256_add_epi16(y1, yC_offset); //lgp->y + py_offset
#if DEPTH_MUL_OPTIM
        yDp0 = _mm256_packus_epi32(yDp0, yDp1);
        yDp1 = _mm256_packus_epi32(yDp2, yDp3);

        yDp0 = _mm256_slli_epi16(yDp0, 4);
        yDp1 = _mm256_slli_epi16(yDp1, 4);

        yDp0 = _mm256_mulhi_epi16(yDp0, yC_depth_mul_fade_slft_3);
        yDp1 = _mm256_mulhi_epi16(yDp1, yC_depth_mul_fade_slft_3);
#else
        //16bit→32bit
        yDp0 = _mm256_sub_epi32(_mm256_add_epi16(yDp0, _mm256_load_si256((__m256i *)ARRAY_0x00008000)), _mm256_load_si256((__m256i *)ARRAY_0x00008000));
        yDp1 = _mm256_sub_epi32(_mm256_add_epi16(yDp1, _mm256_load_si256((__m256i *)ARRAY_0x00008000)), _mm256_load_si256((__m256i *)ARRAY_0x00008000));
        yDp2 = _mm256_sub_epi32(_mm256_add_epi16(yDp2, _mm256_load_si256((__m256i *)ARRAY_0x00008000)), _mm256_load_si256((__m256i *)ARRAY_0x00008000));
        yDp3 = _mm256_sub_epi32(_mm256_add_epi16(yDp3, _mm256_load_si256((__m256i *)ARRAY_0x00008000)), _mm256_load_si256((__m256i *)ARRAY_0x00008000));
            
        //lgp->dp_y * logo_depth_mul_fade)/128 /LOGO_FADE_MAX;
        yDp0 = _mm256_srai_epi32(_mm256_mullo_epi32(yDp0, yC_depth_mul_fade), 15);
        yDp1 = _mm256_srai_epi32(_mm256_mullo_epi32(yDp1, yC_depth_mul_fade), 15);
        yDp2 = _mm256_srai_epi32(_mm256_mullo_epi32(yDp2, yC_depth_mul_fade), 15);
        yDp3 = _mm256_srai_epi32(_mm256_mullo_epi32(yDp3, yC_depth_mul_fade), 15);

        yDp0 = _mm256_packs_epi32(yDp0, yDp1);
        yDp1 = _mm256_packs_epi32(yDp2, yDp3);
#endif
        yDp0 = _mm256_neg_epi16(_mm256_add_epi16(yDp0, _mm256_cmpeq_epi16(yDp0, _mm256_set1_epi16(LOGO_MAX_DP)))); // -dp
        yDp1 = _mm256_neg_epi16(_mm256_add_epi16(yDp1, _mm256_cmpeq_epi16(yDp1, _mm256_set1_epi16(LOGO_MAX_DP)))); // -dp

        //ソースをロードしてNV12->YC48
        ySrc0 = _mm256_load_si256((__m256i *)(ptr_buf));
        ySrc1 = _mm256_unpackhi_epi8(ySrc0, _mm256_setzero_si256());
        ySrc0 = _mm256_unpacklo_epi8(ySrc0, _mm256_setzero_si256());

        ySrc0 = _mm256_slli_epi16(ySrc0, 6);
        ySrc1 = _mm256_slli_epi16(ySrc1, 6);
        ySrc0 = _mm256_mulhi_epi16(ySrc0, yC_nv12_2_yc48_mul);
        ySrc1 = _mm256_mulhi_epi16(ySrc1, yC_nv12_2_yc48_mul);
        ySrc0 = _mm256_sub_epi16(ySrc0, yC_nv12_2_yc48_sub);
        ySrc1 = _mm256_sub_epi16(ySrc1, yC_nv12_2_yc48_sub);

        y3 = _mm256_madd_epi16(_mm256_unpackhi_epi16(ySrc1, y1), _mm256_unpackhi_epi16(_mm256_set1_epi16(LOGO_MAX_DP), yDp1));
        y2 = _mm256_madd_epi16(_mm256_unpacklo_epi16(ySrc1, y1), _mm256_unpacklo_epi16(_mm256_set1_epi16(LOGO_MAX_DP), yDp1));
        y1 = _mm256_madd_epi16(_mm256_unpackhi_epi16(ySrc0, y0), _mm256_unpackhi_epi16(_mm256_set1_epi16(LOGO_MAX_DP), yDp0)); //xSrc0 * LOGO_MAX_DP + x0 * xDp0(-dp)
        y0 = _mm256_madd_epi16(_mm256_unpacklo_epi16(ySrc0, y0), _mm256_unpacklo_epi16(_mm256_set1_epi16(LOGO_MAX_DP), yDp0)); //xSrc0 * LOGO_MAX_DP + x0 * xDp0(-dp)

        yDp0 = _mm256_adds_epi16(_mm256_set1_epi16(LOGO_MAX_DP), yDp0); // LOGO_MAX_DP + (-dp)
        yDp1 = _mm256_adds_epi16(_mm256_set1_epi16(LOGO_MAX_DP), yDp1); // LOGO_MAX_DP + (-dp)

        //(ycp->y * LOGO_MAX_DP + yc * (-dp)) / (LOGO_MAX_DP +(-dp));
        y0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(y0), delogo_rcpps256(_mm256_cvtepi32_ps(cvtlo256_epi16_epi32(yDp0)))));
        y1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(y1), delogo_rcpps256(_mm256_cvtepi32_ps(cvthi256_epi16_epi32(yDp0)))));
        y2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(y2), delogo_rcpps256(_mm256_cvtepi32_ps(cvtlo256_epi16_epi32(yDp1)))));
        y3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(y3), delogo_rcpps256(_mm256_cvtepi32_ps(cvthi256_epi16_epi32(yDp1)))));
        y0 = _mm256_packs_epi32(y0, y1);
        y1 = _mm256_packs_epi32(y2, y3);

        //YC48->NV12
        y0 = _mm256_add_epi16(y0, yC_yc48_2_nv12_add);
        y1 = _mm256_add_epi16(y1, yC_yc48_2_nv12_add);

        y0 = _mm256_mulhi_epi16(y0, yC_yc48_2_nv12_mul);
        y1 = _mm256_mulhi_epi16(y1, yC_yc48_2_nv12_mul);

        y0 = _mm256_packus_epi16(y0, y1);

        _mm256_store_si256((__m256i *)(ptr_buf), y0);
#else
    for (; ptr_buf < ptr_buf_fin; ptr_buf += 16, ptr_logo += 32) {
        __m128i x0, x1, x2, x3;
        __m128i xDp0, xDp1, xDp2, xDp3;
        __m128i xSrc0, xSrc1;

        x0   = _mm_load_si128((__m128i *)(ptr_logo +  0));
        x1   = _mm_load_si128((__m128i *)(ptr_logo +  8));
        x2   = _mm_load_si128((__m128i *)(ptr_logo + 16));
        x3   = _mm_load_si128((__m128i *)(ptr_logo + 24));
            
        // 不透明度情報のみ取り出し
        xDp0 = _mm_and_si128(x0, _mm_load_si128((__m128i *)MASK_16BIT));
        xDp1 = _mm_and_si128(x1, _mm_load_si128((__m128i *)MASK_16BIT));
        xDp2 = _mm_and_si128(x2, _mm_load_si128((__m128i *)MASK_16BIT));
        xDp3 = _mm_and_si128(x3, _mm_load_si128((__m128i *)MASK_16BIT));
            
        //ロゴ色データの取り出し
        x0   = _mm_packs_epi32(_mm_srai_epi32(x0, 16), _mm_srai_epi32(x1, 16));
        x1   = _mm_packs_epi32(_mm_srai_epi32(x2, 16), _mm_srai_epi32(x3, 16));

        x0   = _mm_add_epi16(x0, xC_offset);
        x1   = _mm_add_epi16(x1, xC_offset);
#if DEPTH_MUL_OPTIM
        xDp0 = _mm_packus_epi32(xDp0, xDp1);
        xDp1 = _mm_packus_epi32(xDp2, xDp3);

        xDp0 = _mm_slli_epi16(xDp0, 4);
        xDp1 = _mm_slli_epi16(xDp1, 4);

        xDp0 = _mm_mulhi_epi16(xDp0, xC_depth_mul_fade_slft_3);
        xDp1 = _mm_mulhi_epi16(xDp1, xC_depth_mul_fade_slft_3);
#else
        //16bit→32bit
        xDp0 = _mm_sub_epi32(_mm_add_epi16(xDp0, _mm_load_si128((__m128i *)ARRAY_0x00008000)), _mm_load_si128((__m128i *)ARRAY_0x00008000));
        xDp1 = _mm_sub_epi32(_mm_add_epi16(xDp1, _mm_load_si128((__m128i *)ARRAY_0x00008000)), _mm_load_si128((__m128i *)ARRAY_0x00008000));
        xDp2 = _mm_sub_epi32(_mm_add_epi16(xDp2, _mm_load_si128((__m128i *)ARRAY_0x00008000)), _mm_load_si128((__m128i *)ARRAY_0x00008000));
        xDp3 = _mm_sub_epi32(_mm_add_epi16(xDp3, _mm_load_si128((__m128i *)ARRAY_0x00008000)), _mm_load_si128((__m128i *)ARRAY_0x00008000));

        //lgp->dp_y * logo_depth_mul_fade)/128 /LOGO_FADE_MAX;
        xDp0 = _mm_srai_epi32(_mm_mullo_epi32_simd(xDp0, xC_depth_mul_fade), 15);
        xDp1 = _mm_srai_epi32(_mm_mullo_epi32_simd(xDp1, xC_depth_mul_fade), 15);
        xDp2 = _mm_srai_epi32(_mm_mullo_epi32_simd(xDp2, xC_depth_mul_fade), 15);
        xDp3 = _mm_srai_epi32(_mm_mullo_epi32_simd(xDp3, xC_depth_mul_fade), 15);

        xDp0 = _mm_packs_epi32(xDp0, xDp1);
        xDp1 = _mm_packs_epi32(xDp2, xDp3);
#endif
            
        //dp -= (dp==LOGO_MAX_DP)
        //dp = -dp
        xDp0 = _mm_neg_epi16(_mm_add_epi16(xDp0, _mm_cmpeq_epi16(xDp0, _mm_set1_epi16(LOGO_MAX_DP)))); // -dp
        xDp1 = _mm_neg_epi16(_mm_add_epi16(xDp1, _mm_cmpeq_epi16(xDp1, _mm_set1_epi16(LOGO_MAX_DP)))); // -dp

        //ソースをロードしてNV12->YC48
        xSrc0 = _mm_load_si128((__m128i *)(ptr_buf));
        xSrc1 = _mm_unpackhi_epi8(xSrc0, _mm_setzero_si128());
        xSrc0 = _mm_unpacklo_epi8(xSrc0, _mm_setzero_si128());

        xSrc0 = _mm_slli_epi16(xSrc0, 6);
        xSrc1 = _mm_slli_epi16(xSrc1, 6);
        xSrc0 = _mm_mulhi_epi16(xSrc0, xC_nv12_2_yc48_mul);
        xSrc1 = _mm_mulhi_epi16(xSrc1, xC_nv12_2_yc48_mul);
        xSrc0 = _mm_sub_epi16(xSrc0, xC_nv12_2_yc48_sub);
        xSrc1 = _mm_sub_epi16(xSrc1, xC_nv12_2_yc48_sub);

        x3 = _mm_madd_epi16(_mm_unpackhi_epi16(xSrc1, x1), _mm_unpackhi_epi16(_mm_set1_epi16(LOGO_MAX_DP), xDp1));
        x2 = _mm_madd_epi16(_mm_unpacklo_epi16(xSrc1, x1), _mm_unpacklo_epi16(_mm_set1_epi16(LOGO_MAX_DP), xDp1));
        x1 = _mm_madd_epi16(_mm_unpackhi_epi16(xSrc0, x0), _mm_unpackhi_epi16(_mm_set1_epi16(LOGO_MAX_DP), xDp0)); //xSrc0 * LOGO_MAX_DP + x0 * xDp0(-dp)
        x0 = _mm_madd_epi16(_mm_unpacklo_epi16(xSrc0, x0), _mm_unpacklo_epi16(_mm_set1_epi16(LOGO_MAX_DP), xDp0)); //xSrc0 * LOGO_MAX_DP + x0 * xDp0(-dp)

        xDp0 = _mm_adds_epi16(_mm_set1_epi16(LOGO_MAX_DP), xDp0); // LOGO_MAX_DP + (-dp)
        xDp1 = _mm_adds_epi16(_mm_set1_epi16(LOGO_MAX_DP), xDp1); // LOGO_MAX_DP + (-dp)
            
        //(ycp->y * LOGO_MAX_DP + yc * (-dp)) / (LOGO_MAX_DP +(-dp));
        x0 = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(x0), delogo_rcpps(_mm_cvtepi32_ps(cvtlo_epi16_epi32(xDp0)))));
        x1 = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(x1), delogo_rcpps(_mm_cvtepi32_ps(cvthi_epi16_epi32(xDp0)))));
        x2 = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(x2), delogo_rcpps(_mm_cvtepi32_ps(cvtlo_epi16_epi32(xDp1)))));
        x3 = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(x3), delogo_rcpps(_mm_cvtepi32_ps(cvthi_epi16_epi32(xDp1)))));

        x0 = _mm_packs_epi32(x0, x1);
        x1 = _mm_packs_epi32(x2, x3);

        //YC48->NV12
        x0 = _mm_add_epi16(x0, xC_yc48_2_nv12_add);
        x1 = _mm_add_epi16(x1, xC_yc48_2_nv12_add);

        x0 = _mm_mulhi_epi16(x0, xC_yc48_2_nv12_mul);
        x1 = _mm_mulhi_epi16(x1, xC_yc48_2_nv12_mul);

        x0 = _mm_packus_epi16(x0, x1);

        _mm_store_si128((__m128i *)(ptr_buf), x0);
#endif
    }
}

//dstで示される画像フレームをsrcにコピーしつつ、ロゴ部分を消去する
//height_start, height_finは処理する範囲(NV12なら、色差を処理するときは、高さは半分になることに注意する)
static __forceinline void process_delogo_frame(mfxU8 *dst, const mfxU32 dst_pitch, mfxU8 *buffer, 
    mfxU8 *src, const mfxU32 src_pitch, const mfxU32 width, const mfxU32 height_start, const mfxU32 height_fin, const ProcessDataDelogo *data) {
    mfxU8 *src_line = src;
    mfxU8 *dst_line = dst;
    short *data_ptr = data->pLogoPtr.get();
    const mfxU32 logo_j_start  = data->j_start;
    const mfxU32 logo_j_height = data->height;
    const mfxU32 logo_i_start  = data->i_start;
    const mfxU32 logo_i_width  = data->pitch;
    CONST_M c_nv12_2_yc48_mul  = SET_EPI16(data->nv12_2_yc48_mul);
    CONST_M c_nv12_2_yc48_sub  = SET_EPI16(data->nv12_2_yc48_sub);
    CONST_M c_yc48_2_nv12_mul  = SET_EPI16(data->yc48_2_nv12_mul);
    CONST_M c_yc48_2_nv12_add  = SET_EPI16(data->yc48_2_nv12_add);
    CONST_M c_offset           = SET_EPI32(data->offset[0] | (data->offset[1] << 16));
#if DEPTH_MUL_OPTIM
    CONST_M c_depth_mul_fade_slft_3 = SET_EPI16((short)((data->depth * data->fade) >> 3));
#else //#if DEPTH_MUL_OPTIM
    CONST_M c_depth_mul_fade        = SET_EPI32(data->depth * data->fade);
#endif //#if DEPTH_MUL_OPTIM

    for (mfxU32 j = height_start; j < height_fin; j++, dst_line += dst_pitch, src_line += src_pitch) {
        load_line_to_buffer<256, false>(buffer, src_line, width);
        //if (logo_j_start <= j && j < logo_j_start + logo_j_height) {
        if (j - logo_j_start < logo_j_height) {
            mfxU8 *ptr_buf = buffer + logo_i_start;
            short *ptr_logo = data_ptr + (j - logo_j_start) * (logo_i_width << 1);
            delogo_line(ptr_buf, ptr_logo, logo_i_width, c_nv12_2_yc48_mul, c_nv12_2_yc48_sub, c_yc48_2_nv12_mul, c_yc48_2_nv12_add, c_offset, c_depth_mul_fade_slft_3);
        }
        store_line_from_buffer<256, false>(dst_line, buffer, width);
    }
#if USE_AVX
    _mm256_zeroupper();
#endif
}

//ptrで示される画像フレーム内のロゴ部分を消去して上書きする
//template引数stepはロゴ部分を一時バッファにロードする単位
//height_start, height_finは処理する範囲(NV12なら、色差を処理するときは、高さは半分になることに注意する)
template<mfxU32 step>
static __forceinline void process_delogo(mfxU8 *ptr, const mfxU32 pitch, mfxU8 *buffer, mfxU32 height_start, mfxU32 height_fin, const ProcessDataDelogo *data) {
    mfxU8 *ptr_line = ptr;
    short *data_ptr = data->pLogoPtr.get();
    const mfxU32 logo_j_start  = data->j_start;
    const mfxU32 logo_j_height = data->height;
    const mfxU32 logo_i_start  = data->i_start;
    const mfxU32 logo_i_width  = data->pitch;
    CONST_M c_nv12_2_yc48_mul  = SET_EPI16(data->nv12_2_yc48_mul);
    CONST_M c_nv12_2_yc48_sub  = SET_EPI16(data->nv12_2_yc48_sub);
    CONST_M c_yc48_2_nv12_mul  = SET_EPI16(data->yc48_2_nv12_mul);
    CONST_M c_yc48_2_nv12_add  = SET_EPI16(data->yc48_2_nv12_add);
    CONST_M c_offset           = SET_EPI32(data->offset[0] | (data->offset[1] << 16));
#if DEPTH_MUL_OPTIM
    CONST_M c_depth_mul_fade_slft_3 = SET_EPI16((short)((data->depth * data->fade) >> 3));
#else //#if DEPTH_MUL_OPTIM
    CONST_M c_depth_mul_fade        = SET_EPI32(data->depth * data->fade);
#endif //#if DEPTH_MUL_OPTIM

    height_start = (std::max)(height_start, logo_j_start);
    height_fin   = (std::min)(height_fin, logo_j_start + logo_j_height);

    ptr_line += logo_j_start * pitch;

    for (mfxU32 j = height_start; j < height_fin; j++, ptr_line += pitch) {
        load_line_to_buffer<step, true>(buffer, ptr_line + logo_i_start, logo_i_width);

        short *ptr_logo = data_ptr + (j - logo_j_start) * (logo_i_width << 1);
        delogo_line(buffer, ptr_logo, logo_i_width, c_nv12_2_yc48_mul, c_nv12_2_yc48_sub, c_yc48_2_nv12_mul, c_yc48_2_nv12_add, c_offset, c_depth_mul_fade_slft_3);

        store_line_from_buffer<step, true>(ptr_line + logo_i_start, buffer, logo_i_width);
    }
#if USE_AVX
    _mm256_zeroupper();
#endif
}

#if !USE_AVX
static __forceinline void logo_add_line(mfxU8 *ptr_buf, short *ptr_logo, int logo_i_width,
    CONST_MR(nv12_2_yc48_mul), CONST_MR(nv12_2_yc48_sub), CONST_MR(yc48_2_nv12_mul), CONST_MR(yc48_2_nv12_add),
    CONST_MR(offset), CONST_MR(depth_mul_fade_slft_3)) {
    mfxU8 *ptr_buf_fin = ptr_buf + logo_i_width;
    for (; ptr_buf < ptr_buf_fin; ptr_buf += 16, ptr_logo += 32) {
        __m128i x0, x1, x2, x3;
        __m128i xDp0, xDp1, xDp2, xDp3;
        __m128i xSrc0, xSrc1;

        x0   = _mm_load_si128((__m128i *)(ptr_logo +  0));
        x1   = _mm_load_si128((__m128i *)(ptr_logo +  8));
        x2   = _mm_load_si128((__m128i *)(ptr_logo + 16));
        x3   = _mm_load_si128((__m128i *)(ptr_logo + 24));

        // 不透明度情報のみ取り出し
        xDp0 = _mm_and_si128(x0, _mm_load_si128((__m128i *)MASK_16BIT));
        xDp1 = _mm_and_si128(x1, _mm_load_si128((__m128i *)MASK_16BIT));
        xDp2 = _mm_and_si128(x2, _mm_load_si128((__m128i *)MASK_16BIT));
        xDp3 = _mm_and_si128(x3, _mm_load_si128((__m128i *)MASK_16BIT));

        //ロゴ色データの取り出し
        x0   = _mm_packs_epi32(_mm_srai_epi32(x0, 16), _mm_srai_epi32(x1, 16));
        x1   = _mm_packs_epi32(_mm_srai_epi32(x2, 16), _mm_srai_epi32(x3, 16));

        x0   = _mm_add_epi16(x0, xC_offset);
        x1   = _mm_add_epi16(x1, xC_offset);
#if DEPTH_MUL_OPTIM
        xDp0 = _mm_packus_epi32(xDp0, xDp1);
        xDp1 = _mm_packus_epi32(xDp2, xDp3);

        xDp0 = _mm_slli_epi16(xDp0, 4);
        xDp1 = _mm_slli_epi16(xDp1, 4);

        xDp0 = _mm_mulhi_epi16(xDp0, xC_depth_mul_fade_slft_3);
        xDp1 = _mm_mulhi_epi16(xDp1, xC_depth_mul_fade_slft_3);
#else
        //16bit→32bit
        xDp0 = _mm_sub_epi32(_mm_add_epi16(xDp0, _mm_load_si128((__m128i *)ARRAY_0x00008000)), _mm_load_si128((__m128i *)ARRAY_0x00008000));
        xDp1 = _mm_sub_epi32(_mm_add_epi16(xDp1, _mm_load_si128((__m128i *)ARRAY_0x00008000)), _mm_load_si128((__m128i *)ARRAY_0x00008000));
        xDp2 = _mm_sub_epi32(_mm_add_epi16(xDp2, _mm_load_si128((__m128i *)ARRAY_0x00008000)), _mm_load_si128((__m128i *)ARRAY_0x00008000));
        xDp3 = _mm_sub_epi32(_mm_add_epi16(xDp3, _mm_load_si128((__m128i *)ARRAY_0x00008000)), _mm_load_si128((__m128i *)ARRAY_0x00008000));

        //lgp->dp_y * logo_depth_mul_fade)/128 /LOGO_FADE_MAX;
        xDp0 = _mm_srai_epi32(_mm_mullo_epi32_simd(xDp0, xC_depth_mul_fade), 15);
        xDp1 = _mm_srai_epi32(_mm_mullo_epi32_simd(xDp1, xC_depth_mul_fade), 15);
        xDp2 = _mm_srai_epi32(_mm_mullo_epi32_simd(xDp2, xC_depth_mul_fade), 15);
        xDp3 = _mm_srai_epi32(_mm_mullo_epi32_simd(xDp3, xC_depth_mul_fade), 15);

        xDp0 = _mm_packs_epi32(xDp0, xDp1);
        xDp1 = _mm_packs_epi32(xDp2, xDp3);
#endif

        //ソースをロードしてNV12->YC48
        xSrc0 = _mm_load_si128((__m128i *)(ptr_buf));
        xSrc1 = _mm_unpackhi_epi8(xSrc0, _mm_setzero_si128());
        xSrc0 = _mm_unpacklo_epi8(xSrc0, _mm_setzero_si128());

        xSrc0 = _mm_slli_epi16(xSrc0, 6);
        xSrc1 = _mm_slli_epi16(xSrc1, 6);
        xSrc0 = _mm_mulhi_epi16(xSrc0, xC_nv12_2_yc48_mul);
        xSrc1 = _mm_mulhi_epi16(xSrc1, xC_nv12_2_yc48_mul);
        xSrc0 = _mm_sub_epi16(xSrc0, xC_nv12_2_yc48_sub);
        xSrc1 = _mm_sub_epi16(xSrc1, xC_nv12_2_yc48_sub);

        xDp2 = _mm_subs_epi16(_mm_set1_epi16(LOGO_MAX_DP), xDp0);
        xDp3 = _mm_subs_epi16(_mm_set1_epi16(LOGO_MAX_DP), xDp1);
        x3 = _mm_madd_epi16(_mm_unpackhi_epi16(xSrc1, x1), _mm_unpackhi_epi16(xDp3, xDp1));
        x2 = _mm_madd_epi16(_mm_unpacklo_epi16(xSrc1, x1), _mm_unpacklo_epi16(xDp3, xDp1));
        x1 = _mm_madd_epi16(_mm_unpackhi_epi16(xSrc0, x0), _mm_unpackhi_epi16(xDp2, xDp0)); //xSrc0 * (LOGO_MAX_DP-logo_depth[i]) + x0 * xDp0
        x0 = _mm_madd_epi16(_mm_unpacklo_epi16(xSrc0, x0), _mm_unpacklo_epi16(xDp2, xDp0)); //xSrc0 * (LOGO_MAX_DP-logo_depth[i]) + x0 * xDp0

        //(ycp->y * (LOGO_MAX_DP-logo_depth[i]) + yc * (-dp)) / (LOGO_MAX_DP);
        // 1 / LOGO_MAX_DP = 131 / 131072 = 131 / (1<<17) 
        x0 = _mm_srai_epi32(_mm_mullo_epi32_simd(x0, _mm_set1_epi32(131)), 17);
        x1 = _mm_srai_epi32(_mm_mullo_epi32_simd(x1, _mm_set1_epi32(131)), 17);
        x2 = _mm_srai_epi32(_mm_mullo_epi32_simd(x2, _mm_set1_epi32(131)), 17);
        x3 = _mm_srai_epi32(_mm_mullo_epi32_simd(x3, _mm_set1_epi32(131)), 17);

        x0 = _mm_packs_epi32(x0, x1);
        x1 = _mm_packs_epi32(x2, x3);

        //YC48->NV12
        x0 = _mm_add_epi16(x0, xC_yc48_2_nv12_add);
        x1 = _mm_add_epi16(x1, xC_yc48_2_nv12_add);

        x0 = _mm_mulhi_epi16(x0, xC_yc48_2_nv12_mul);
        x1 = _mm_mulhi_epi16(x1, xC_yc48_2_nv12_mul);

        x0 = _mm_packus_epi16(x0, x1);

        _mm_store_si128((__m128i *)(ptr_buf), x0);
    }
}

//dstで示される画像フレームをsrcにコピーしつつ、ロゴ部分を消去する
//height_start, height_finは処理する範囲(NV12なら、色差を処理するときは、高さは半分になることに注意する)
static __forceinline void process_logo_add_frame(mfxU8 *dst, const mfxU32 dst_pitch, mfxU8 *buffer,
    mfxU8 *src, const mfxU32 src_pitch, const mfxU32 width, const mfxU32 height_start, const mfxU32 height_fin, const ProcessDataDelogo *data) {
    mfxU8 *src_line = src;
    mfxU8 *dst_line = dst;
    short *data_ptr = data->pLogoPtr.get();
    const mfxU32 logo_j_start  = data->j_start;
    const mfxU32 logo_j_height = data->height;
    const mfxU32 logo_i_start  = data->i_start;
    const mfxU32 logo_i_width  = data->pitch;
    CONST_M c_nv12_2_yc48_mul  = SET_EPI16(data->nv12_2_yc48_mul);
    CONST_M c_nv12_2_yc48_sub  = SET_EPI16(data->nv12_2_yc48_sub);
    CONST_M c_yc48_2_nv12_mul  = SET_EPI16(data->yc48_2_nv12_mul);
    CONST_M c_yc48_2_nv12_add  = SET_EPI16(data->yc48_2_nv12_add);
    CONST_M c_offset           = SET_EPI32(data->offset[0] | (data->offset[1] << 16));
#if DEPTH_MUL_OPTIM
    CONST_M c_depth_mul_fade_slft_3 = SET_EPI16((short)((data->depth * data->fade) >> 3));
#else //#if DEPTH_MUL_OPTIM
    CONST_M c_depth_mul_fade        = SET_EPI32(data->depth * data->fade);
#endif //#if DEPTH_MUL_OPTIM

    for (mfxU32 j = height_start; j < height_fin; j++, dst_line += dst_pitch, src_line += src_pitch) {
        load_line_to_buffer<256, false>(buffer, src_line, width);
        //if (logo_j_start <= j && j < logo_j_start + logo_j_height) {
        if (j - logo_j_start < logo_j_height) {
            mfxU8 *ptr_buf = buffer + logo_i_start;
            short *ptr_logo = data_ptr + (j - logo_j_start) * (logo_i_width << 1);
            logo_add_line(ptr_buf, ptr_logo, logo_i_width, c_nv12_2_yc48_mul, c_nv12_2_yc48_sub, c_yc48_2_nv12_mul, c_yc48_2_nv12_add, c_offset, c_depth_mul_fade_slft_3);
        }
        store_line_from_buffer<256, false>(dst_line, buffer, width);
    }
#if USE_AVX
    _mm256_zeroupper();
#endif
}

//ptrで示される画像フレーム内のロゴを付加して上書きする
//template引数stepはロゴ部分を一時バッファにロードする単位
//height_start, height_finは処理する範囲(NV12なら、色差を処理するときは、高さは半分になることに注意する)
template<uint32_t step>
static __forceinline void process_logo_add(mfxU8 *ptr, const uint32_t pitch, mfxU8 *buffer, uint32_t height_start, uint32_t height_fin, const ProcessDataDelogo *data) {
    mfxU8 *ptr_line = ptr;
    short *data_ptr = data->pLogoPtr.get();
    const uint32_t logo_j_start  = data->j_start;
    const uint32_t logo_j_height = data->height;
    const uint32_t logo_i_start  = data->i_start;
    const uint32_t logo_i_width  = data->pitch;
    CONST_M c_nv12_2_yc48_mul  = SET_EPI16(data->nv12_2_yc48_mul);
    CONST_M c_nv12_2_yc48_sub  = SET_EPI16(data->nv12_2_yc48_sub);
    CONST_M c_yc48_2_nv12_mul  = SET_EPI16(data->yc48_2_nv12_mul);
    CONST_M c_yc48_2_nv12_add  = SET_EPI16(data->yc48_2_nv12_add);
    CONST_M c_offset           = SET_EPI32(data->offset[0] | (data->offset[1] << 16));
#if DEPTH_MUL_OPTIM
    CONST_M c_depth_mul_fade_slft_3 = SET_EPI16((short)((data->depth * data->fade) >> 3));
#else //#if DEPTH_MUL_OPTIM
    CONST_M c_depth_mul_fade        = SET_EPI32(data->depth * data->fade);
#endif //#if DEPTH_MUL_OPTIM

    height_start = (std::max)(height_start, logo_j_start);
    height_fin   = (std::min)(height_fin, logo_j_start + logo_j_height);

    ptr_line += logo_j_start * pitch;

    for (mfxU32 j = height_start; j < height_fin; j++, ptr_line += pitch) {
        load_line_to_buffer<step, true>(buffer, ptr_line + logo_i_start, logo_i_width);

        short *ptr_logo = data_ptr + (j - logo_j_start) * (logo_i_width << 1);
        logo_add_line(buffer, ptr_logo, logo_i_width, c_nv12_2_yc48_mul, c_nv12_2_yc48_sub, c_yc48_2_nv12_mul, c_yc48_2_nv12_add, c_offset, c_depth_mul_fade_slft_3);

        store_line_from_buffer<step, true>(ptr_line + logo_i_start, buffer, logo_i_width);
    }
}
#endif
