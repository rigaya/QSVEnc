// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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

#ifndef TypePixel
#define TypePixel uchar
#endif

#ifndef RTGMC_SEARCH_PREFILTER_PIXEL_MAX
#define RTGMC_SEARCH_PREFILTER_PIXEL_MAX 255
#endif

#ifndef RTGMC_SEARCH_PREFILTER_LIMITED_Y_MIN
#define RTGMC_SEARCH_PREFILTER_LIMITED_Y_MIN 16
#endif

#ifndef RTGMC_SEARCH_PREFILTER_LIMITED_Y_RANGE
#define RTGMC_SEARCH_PREFILTER_LIMITED_Y_RANGE 219
#endif

#ifndef RTGMC_SEARCH_PREFILTER_LIMITED_C_OFFSET
#define RTGMC_SEARCH_PREFILTER_LIMITED_C_OFFSET 128
#endif

#ifndef RTGMC_SEARCH_PREFILTER_LIMITED_C_RANGE
#define RTGMC_SEARCH_PREFILTER_LIMITED_C_RANGE 112
#endif

#ifndef RTGMC_SEARCH_REFINE2_GAUSS_W0
#define RTGMC_SEARCH_REFINE2_GAUSS_W0 0.227027029f
#endif
#ifndef RTGMC_SEARCH_REFINE2_GAUSS_W1
#define RTGMC_SEARCH_REFINE2_GAUSS_W1 0.197707996f
#endif
#ifndef RTGMC_SEARCH_REFINE2_GAUSS_W2
#define RTGMC_SEARCH_REFINE2_GAUSS_W2 0.130435750f
#endif
#ifndef RTGMC_SEARCH_REFINE2_GAUSS_W3
#define RTGMC_SEARCH_REFINE2_GAUSS_W3 0.065223776f
#endif
#ifndef RTGMC_SEARCH_REFINE2_GAUSS_W4
#define RTGMC_SEARCH_REFINE2_GAUSS_W4 0.024685025f
#endif

#define RTGMC_SEARCH_PREFILTER_SCENECHANGE 28
#define RTGMC_SEARCH_PREFILTER_BLOCK_PIXELS (rtgmc_search_prefilter_block_x * rtgmc_search_prefilter_block_y)
#define RTGMC_SEARCH_REPAIR_THIN_WIDE_CORE (1u << 0)
#define RTGMC_SEARCH_REPAIR_THIN_CORE_BLEND (1u << 1)
#define RTGMC_SEARCH_REPAIR_THIN_RANK_LIMIT (1u << 2)
#define RTGMC_SEARCH_REPAIR_RESTORE_WIDE_ENVELOPE (1u << 0)
#define RTGMC_SEARCH_REPAIR_RESTORE_LEVEL4_PATH (1u << 1)
#define RTGMC_SEARCH_REPAIR_RESTORE_ENABLED (1u << 2)

inline int rtgmc_search_repair_profile_thin_reject_level(const uint repairProfile) {
    return (int)(repairProfile & 0xffu);
}

inline int rtgmc_search_repair_profile_restore_padding_level(const uint repairProfile) {
    return (int)((repairProfile >> 8) & 0xffu);
}

inline uint rtgmc_search_repair_profile_thin_reject_flags(const uint repairProfile) {
    return (repairProfile >> 16) & 0xffu;
}

inline uint rtgmc_search_repair_profile_restore_flags(const uint repairProfile) {
    return (repairProfile >> 24) & 0xffu;
}

inline TypePixel rtgmc_search_prefilter_clamp_pixel(const int value) {
    return (TypePixel)clamp(value, 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
}

inline int rtgmc_search_prefilter_pixel_load(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int x,
    const int y) {
    const int px = clamp(x, 0, width - 1);
    const int py = clamp(y, 0, height - 1);
    return (int)(*(__global const TypePixel *)(src + py * pitch + px * (int)sizeof(TypePixel)));
}

inline void rtgmc_search_prefilter_pixel_store(
    __global uchar *dst,
    const int pitch,
    const int x,
    const int y,
    const int value) {
    *(__global TypePixel *)(dst + y * pitch + x * (int)sizeof(TypePixel)) = rtgmc_search_prefilter_clamp_pixel(value);
}

inline int rtgmc_search_prefilter_blur3x3_weighted(
    const int p00,
    const int p10,
    const int p20,
    const int p01,
    const int p11,
    const int p21,
    const int p02,
    const int p12,
    const int p22) {
    const int sum =
        p00 + 2 * p10 + p20 +
        2 * p01 + 4 * p11 + 2 * p21 +
        p02 + 2 * p12 + p22;
    return (sum + 8) >> 4;
}

inline int rtgmc_search_prefilter_edge_soften_cross(
    const int left,
    const int up,
    const int center,
    const int down,
    const int right) {
    return (left + up + 4 * center + down + right + 4) >> 3;
}

inline int rtgmc_search_prefilter_range_half(void) {
    return (RTGMC_SEARCH_PREFILTER_PIXEL_MAX + 1) >> 1;
}

inline int rtgmc_search_prefilter_range_scale(void) {
    return max((RTGMC_SEARCH_PREFILTER_PIXEL_MAX + 1) >> 8, 1);
}

inline int rtgmc_search_prefilter_extreme_seed(const int highSide) {
    return highSide ? 0 : RTGMC_SEARCH_PREFILTER_PIXEL_MAX;
}

inline int rtgmc_search_prefilter_extreme_merge(const int value, const int sample, const int highSide) {
    return highSide ? max(value, sample) : min(value, sample);
}

inline int rtgmc_search_prefilter_polarity_core_seed(const int positive) {
    return rtgmc_search_prefilter_extreme_seed(!positive);
}

inline int rtgmc_search_prefilter_polarity_core_merge(const int value, const int sample, const int positive) {
    return rtgmc_search_prefilter_extreme_merge(value, sample, !positive);
}

inline int rtgmc_search_prefilter_polarity_envelope_seed(const int positive) {
    return rtgmc_search_prefilter_extreme_seed(positive);
}

inline int rtgmc_search_prefilter_polarity_envelope_merge(const int value, const int sample, const int positive) {
    return rtgmc_search_prefilter_extreme_merge(value, sample, positive);
}

inline void rtgmc_search_prefilter_sort2(__private int *a, __private int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = lo;
    *b = hi;
}

inline void rtgmc_search_prefilter_sort2_desc(__private int *a, __private int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = hi;
    *b = lo;
}

// Batcher's Bitonic Sort (1968), 8 elements / 24 comparisons / depth 6.
inline void rtgmc_search_prefilter_sort8(__private int *v) {
    rtgmc_search_prefilter_sort2     (&v[0], &v[1]); rtgmc_search_prefilter_sort2_desc(&v[2], &v[3]); rtgmc_search_prefilter_sort2     (&v[4], &v[5]); rtgmc_search_prefilter_sort2_desc(&v[6], &v[7]);
    rtgmc_search_prefilter_sort2     (&v[0], &v[2]); rtgmc_search_prefilter_sort2     (&v[1], &v[3]); rtgmc_search_prefilter_sort2_desc(&v[4], &v[6]); rtgmc_search_prefilter_sort2_desc(&v[5], &v[7]);
    rtgmc_search_prefilter_sort2     (&v[0], &v[1]); rtgmc_search_prefilter_sort2     (&v[2], &v[3]); rtgmc_search_prefilter_sort2_desc(&v[4], &v[5]); rtgmc_search_prefilter_sort2_desc(&v[6], &v[7]);
    rtgmc_search_prefilter_sort2     (&v[0], &v[4]); rtgmc_search_prefilter_sort2     (&v[1], &v[5]); rtgmc_search_prefilter_sort2     (&v[2], &v[6]); rtgmc_search_prefilter_sort2     (&v[3], &v[7]);
    rtgmc_search_prefilter_sort2     (&v[0], &v[2]); rtgmc_search_prefilter_sort2     (&v[1], &v[3]); rtgmc_search_prefilter_sort2     (&v[4], &v[6]); rtgmc_search_prefilter_sort2     (&v[5], &v[7]);
    rtgmc_search_prefilter_sort2     (&v[0], &v[1]); rtgmc_search_prefilter_sort2     (&v[2], &v[3]); rtgmc_search_prefilter_sort2     (&v[4], &v[5]); rtgmc_search_prefilter_sort2     (&v[6], &v[7]);
}

inline int rtgmc_search_prefilter_temporal_sample(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int slot) {
    switch (slot) {
    case 0:
        return rtgmc_search_prefilter_pixel_load(srcPrev2, pitch, srcWidth, srcHeight, px, py);
    case 1:
        return rtgmc_search_prefilter_pixel_load(srcPrev,  pitch, srcWidth, srcHeight, px, py);
    case 3:
        return rtgmc_search_prefilter_pixel_load(srcNext,  pitch, srcWidth, srcHeight, px, py);
    case 4:
        return rtgmc_search_prefilter_pixel_load(srcNext2, pitch, srcWidth, srcHeight, px, py);
    default:
        return rtgmc_search_prefilter_pixel_load(srcCur,   pitch, srcWidth, srcHeight, px, py);
    }
}

inline int rtgmc_search_prefilter_temporal_weighted_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int tapCount) {
    int sum = 0;
    if (tapCount >= 5) {
        const int taps[5] = { 1, 4, 6, 4, 1 };
#pragma unroll
        for (int i = 0; i < 5; i++) {
            sum += taps[i] * rtgmc_search_prefilter_temporal_sample(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
                srcWidth, srcHeight, px, py, i);
        }
        return (sum + 4) >> 4;
    }
    const int taps[3] = { 1, 2, 1 };
#pragma unroll
    for (int i = 0; i < 3; i++) {
        sum += taps[i] * rtgmc_search_prefilter_temporal_sample(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, py, i + 1);
    }
    return (sum + 2) >> 2;
}

inline int rtgmc_search_prefilter_temporal_candidate_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int smoothRadius) {
    if (smoothRadius >= 2) {
        return rtgmc_search_prefilter_temporal_weighted_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, py, 5);
    }
    if (smoothRadius >= 1) {
        return rtgmc_search_prefilter_temporal_weighted_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, py, 3);
    }
    return rtgmc_search_prefilter_pixel_load(srcCur, pitch, srcWidth, srcHeight, px, py);
}

inline int rtgmc_search_prefilter_makediff_value(const int ref, const int src) {
    return clamp(ref - src + rtgmc_search_prefilter_range_half(), 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
}

inline int rtgmc_search_prefilter_adddiff_value(const int src, const int diff) {
    return clamp(src + diff - rtgmc_search_prefilter_range_half(), 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
}

inline int rtgmc_search_prefilter_select_signed_correction(
    const int proposedSigned,
    const int positiveMaskSigned,
    const int negativeMaskSigned,
    const int threshold) {
    if (proposedSigned >= threshold) {
        return (positiveMaskSigned > 0) ? positiveMaskSigned : 0;
    }
    if (proposedSigned <= -threshold) {
        return (negativeMaskSigned < 0) ? negativeMaskSigned : 0;
    }
    return 0;
}

inline int rtgmc_search_prefilter_apply_signed_correction(
    const int src,
    const int proposedSigned,
    const int positiveMaskSigned,
    const int negativeMaskSigned,
    const int threshold) {
    const int appliedSigned = rtgmc_search_prefilter_select_signed_correction(
        proposedSigned,
        positiveMaskSigned,
        negativeMaskSigned,
        threshold);
    return clamp(src + appliedSigned, 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
}

inline int rtgmc_search_prefilter_round_float_to_pixel(const float value) {
    return clamp((int)(value + 0.5f), 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
}

inline int rtgmc_search_prefilter_mean3x3_diff_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int smoothRadius) {
    if (px <= 0 || py <= 0 || px >= srcWidth - 1 || py >= srcHeight - 1) {
        const int src = rtgmc_search_prefilter_temporal_candidate_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight,
            px, py, smoothRadius);
        const int ref = rtgmc_search_prefilter_pixel_load(srcCur, pitch, srcWidth, srcHeight, px, py);
        return rtgmc_search_prefilter_makediff_value(ref, src);
    }
    int sum = 0;
    for (int iy = -1; iy <= 1; iy++) {
        for (int ix = -1; ix <= 1; ix++) {
            const int src = rtgmc_search_prefilter_temporal_candidate_value(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
                srcWidth, srcHeight,
                px + ix, py + iy, smoothRadius);
            const int ref = rtgmc_search_prefilter_pixel_load(srcCur, pitch, srcWidth, srcHeight, px + ix, py + iy);
            sum += rtgmc_search_prefilter_makediff_value(ref, src);
        }
    }
    return (sum + 4) / 9;
}

inline int rtgmc_search_prefilter_search_correction_delta_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int smoothRadius) {
    const int src = rtgmc_search_prefilter_temporal_candidate_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight,
        px, py, smoothRadius);
    const int ref = rtgmc_search_prefilter_pixel_load(srcCur, pitch, srcWidth, srcHeight, px, py);
    return rtgmc_search_prefilter_makediff_value(ref, src);
}

inline int rtgmc_search_prefilter_removegrain4_diff_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int smoothRadius) {
    const int s = rtgmc_search_prefilter_search_correction_delta_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight,
        px, py, smoothRadius);
    if (px <= 0 || py <= 0 || px >= srcWidth - 1 || py >= srcHeight - 1) {
        return s;
    }
    int v[8];
    int count = 0;
#pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
#pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            if (dx != 0 || dy != 0) {
                v[count++] = rtgmc_search_prefilter_search_correction_delta_value(
                    srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
                    srcWidth, srcHeight, px + dx, py + dy, smoothRadius);
            }
        }
    }
    rtgmc_search_prefilter_sort8(v);
    return clamp(s, v[3], v[4]);
}

inline int rtgmc_search_prefilter_vertical_thin_reject_diff_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int smoothRadius,
    const int radius,
    const int positive) {
    int value = rtgmc_search_prefilter_polarity_core_seed(positive);
    for (int iy = -radius; iy <= radius; iy++) {
        const int diff = rtgmc_search_prefilter_search_correction_delta_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight,
            px, py + iy, smoothRadius);
        value = rtgmc_search_prefilter_polarity_core_merge(value, diff, positive);
    }
    return value;
}

inline int rtgmc_search_prefilter_vertical_restore_diff_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int smoothRadius,
    const int thinRejectRadius,
    const int restorePaddingRadius,
    const int positive) {
    int value = rtgmc_search_prefilter_polarity_envelope_seed(positive);
    for (int iy = -restorePaddingRadius; iy <= restorePaddingRadius; iy++) {
        const int diff = rtgmc_search_prefilter_vertical_thin_reject_diff_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight,
            px, py + iy, smoothRadius, thinRejectRadius, positive);
        value = rtgmc_search_prefilter_polarity_envelope_merge(value, diff, positive);
    }
    return value;
}

inline int rtgmc_search_prefilter_area_envelope_diff_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int smoothRadius,
    const int positive) {
    int value = rtgmc_search_prefilter_polarity_envelope_seed(positive);
    for (int iy = -1; iy <= 1; iy++) {
        for (int ix = -1; ix <= 1; ix++) {
            const int diff = rtgmc_search_prefilter_search_correction_delta_value(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
                srcWidth, srcHeight,
                px + ix, py + iy, smoothRadius);
            value = rtgmc_search_prefilter_polarity_envelope_merge(value, diff, positive);
        }
    }
    return value;
}

inline int rtgmc_search_prefilter_correction_gate_thin_core_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int smoothRadius,
    const uint repairProfile,
    const int positive) {
    const int thinRejectRadius = 2 + ((rtgmc_search_repair_profile_thin_reject_flags(repairProfile) & RTGMC_SEARCH_REPAIR_THIN_WIDE_CORE) ? 1 : 0);
    return rtgmc_search_prefilter_vertical_thin_reject_diff_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight,
        px, py, smoothRadius, thinRejectRadius, positive);
}

inline int rtgmc_search_prefilter_mean3x3_correction_gate_thin_core_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius,
    const uint repairProfile, const int positive) {
    if (px <= 0 || py <= 0 || px >= srcWidth - 1 || py >= srcHeight - 1) {
        return rtgmc_search_prefilter_correction_gate_thin_core_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
    }
    int sum = 0;
    for (int iy = -1; iy <= 1; iy++) {
        for (int ix = -1; ix <= 1; ix++) {
            sum += rtgmc_search_prefilter_correction_gate_thin_core_value(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
                srcWidth, srcHeight, px + ix, py + iy, smoothRadius, repairProfile, positive);
        }
    }
    return (sum + 4) / 9;
}

inline int rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius,
    const uint repairProfile, const int positive) {
    int value = rtgmc_search_prefilter_correction_gate_thin_core_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
    if (rtgmc_search_repair_profile_thin_reject_flags(repairProfile) & RTGMC_SEARCH_REPAIR_THIN_CORE_BLEND) {
        const int mean3x3 = rtgmc_search_prefilter_mean3x3_correction_gate_thin_core_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
        value = rtgmc_search_prefilter_polarity_core_merge(value, mean3x3, positive);
    }
    return value;
}

inline int rtgmc_search_prefilter_rank_limit4_correction_gate_mid_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius,
    const uint repairProfile, const int positive) {
    const int s = rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
    if (px <= 0 || py <= 0 || px >= srcWidth - 1 || py >= srcHeight - 1) {
        return s;
    }
    int v[8] = {
        rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px - 1, py - 1, smoothRadius, repairProfile, positive),
        rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + 0, py - 1, smoothRadius, repairProfile, positive),
        rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + 1, py - 1, smoothRadius, repairProfile, positive),
        rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px - 1, py + 0, smoothRadius, repairProfile, positive),
        rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + 1, py + 0, smoothRadius, repairProfile, positive),
        rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px - 1, py + 1, smoothRadius, repairProfile, positive),
        rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + 0, py + 1, smoothRadius, repairProfile, positive),
        rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + 1, py + 1, smoothRadius, repairProfile, positive)
    };
    rtgmc_search_prefilter_sort8(v);
    return clamp(s, v[3], v[4]);
}

inline int rtgmc_search_prefilter_correction_gate_mid_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius,
    const uint repairProfile, const int positive) {
    // Keep the rank limiter before the restore envelope so isolated thin residuals do not expand first.
    if (rtgmc_search_repair_profile_thin_reject_flags(repairProfile) & RTGMC_SEARCH_REPAIR_THIN_RANK_LIMIT) {
        return rtgmc_search_prefilter_rank_limit4_correction_gate_mid_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
    }
    return rtgmc_search_prefilter_correction_gate_mid_before_rank_limit_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
}

inline int rtgmc_search_prefilter_correction_gate_base_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius,
    const uint repairProfile, const int positive) {
    // Build a restore envelope around the rejected core; higher levels widen it by one row.
    const int restorePaddingRadius = 2 + ((rtgmc_search_repair_profile_restore_flags(repairProfile) & RTGMC_SEARCH_REPAIR_RESTORE_WIDE_ENVELOPE) ? 1 : 0);
    int value = rtgmc_search_prefilter_polarity_envelope_seed(positive);
    for (int iy = -restorePaddingRadius; iy <= restorePaddingRadius; iy++) {
        const int cur = rtgmc_search_prefilter_correction_gate_mid_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, py + iy, smoothRadius, repairProfile, positive);
        value = rtgmc_search_prefilter_polarity_envelope_merge(value, cur, positive);
    }
    return value;
}

inline int rtgmc_search_prefilter_mean3x3_correction_gate_base_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius,
    const uint repairProfile, const int positive) {
    if (px <= 0 || py <= 0 || px >= srcWidth - 1 || py >= srcHeight - 1) {
        return rtgmc_search_prefilter_correction_gate_base_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
    }
    int sum = 0;
    for (int iy = -1; iy <= 1; iy++) {
        for (int ix = -1; ix <= 1; ix++) {
            sum += rtgmc_search_prefilter_correction_gate_base_value(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
                srcWidth, srcHeight, px + ix, py + iy, smoothRadius, repairProfile, positive);
        }
    }
    return (sum + 4) / 9;
}

inline int rtgmc_search_prefilter_correction_gate_rank_smooth1_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius,
    const uint repairProfile, const int positive, const int useMax) {
    const int s = rtgmc_search_prefilter_correction_gate_base_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
    const int mean3x3 = rtgmc_search_prefilter_mean3x3_correction_gate_base_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
    return rtgmc_search_prefilter_extreme_merge(s, mean3x3, useMax);
}

inline int rtgmc_search_prefilter_mean3x3_correction_gate_rank_smooth1_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius,
    const uint repairProfile, const int positive, const int useMax) {
    if (px <= 0 || py <= 0 || px >= srcWidth - 1 || py >= srcHeight - 1) {
        return rtgmc_search_prefilter_correction_gate_rank_smooth1_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive, useMax);
    }
    int sum = 0;
    for (int iy = -1; iy <= 1; iy++) {
        for (int ix = -1; ix <= 1; ix++) {
            sum += rtgmc_search_prefilter_correction_gate_rank_smooth1_value(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
                srcWidth, srcHeight, px + ix, py + iy, smoothRadius, repairProfile, positive, useMax);
        }
    }
    return (sum + 4) / 9;
}

inline int rtgmc_search_prefilter_correction_gate_rank_smooth2_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius,
    const uint repairProfile, const int positive, const int useMax) {
    const int s = rtgmc_search_prefilter_correction_gate_rank_smooth1_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive, useMax);
    const int mean3x3 = rtgmc_search_prefilter_mean3x3_correction_gate_rank_smooth1_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive, useMax);
    return rtgmc_search_prefilter_extreme_merge(s, mean3x3, useMax);
}

inline int rtgmc_search_prefilter_correction_gate_area_envelope_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius,
    const uint repairProfile, const int positive, const int useMax) {
    int value = rtgmc_search_prefilter_extreme_seed(useMax);
    for (int iy = -1; iy <= 1; iy++) {
        for (int ix = -1; ix <= 1; ix++) {
            const int cur = rtgmc_search_prefilter_correction_gate_base_value(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
                srcWidth, srcHeight, px + ix, py + iy, smoothRadius, repairProfile, positive);
            value = rtgmc_search_prefilter_extreme_merge(value, cur, useMax);
        }
    }
    return value;
}

inline int rtgmc_search_prefilter_correction_gate_level4_core_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius,
    const int positive) {
    int value = rtgmc_search_prefilter_polarity_core_seed(positive);
    for (int iy = -2; iy <= 2; iy++) {
        const int sampleY = ((py + iy) < 0 || (py + iy) >= srcHeight) ? py : (py + iy);
        const int diff = rtgmc_search_prefilter_search_correction_delta_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, sampleY, smoothRadius);
        value = rtgmc_search_prefilter_polarity_core_merge(value, diff, positive);
    }
    return value;
}

inline int rtgmc_search_prefilter_correction_gate_level4_mean3x3_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius,
    const int positive) {
    const int s = rtgmc_search_prefilter_correction_gate_level4_core_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight, px, py, smoothRadius, positive);
    if (px <= 0 || py <= 0 || px >= srcWidth - 1 || py >= srcHeight - 1) {
        return s;
    }
    int sum = 0;
    for (int iy = -1; iy <= 1; iy++) {
        for (int ix = -1; ix <= 1; ix++) {
            sum += rtgmc_search_prefilter_correction_gate_level4_core_value(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
                srcWidth, srcHeight, px + ix, py + iy, smoothRadius, positive);
        }
    }
    return (sum + 4) / 9;
}

inline int rtgmc_search_prefilter_correction_gate_level4_mid_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius,
    const int positive) {
    const int s = rtgmc_search_prefilter_correction_gate_level4_core_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight, px, py, smoothRadius, positive);
    const int mean3x3 = rtgmc_search_prefilter_correction_gate_level4_mean3x3_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight, px, py, smoothRadius, positive);
    return rtgmc_search_prefilter_polarity_core_merge(s, mean3x3, positive);
}

inline int rtgmc_search_prefilter_correction_gate_level4_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius,
    const int positive) {
    int value = rtgmc_search_prefilter_polarity_envelope_seed(positive);
    for (int iy = -2; iy <= 2; iy++) {
        const int sampleY = ((py + iy) < 0 || (py + iy) >= srcHeight) ? py : (py + iy);
        const int cur = rtgmc_search_prefilter_correction_gate_level4_mid_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, sampleY, smoothRadius, positive);
        value = rtgmc_search_prefilter_polarity_envelope_merge(value, cur, positive);
    }
    return value;
}

inline int rtgmc_search_prefilter_correction_gate_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth, const int srcHeight,
    const int px, const int py, const int smoothRadius,
    const uint repairProfile, const int positive) {
    const int restorePaddingLevel = rtgmc_search_repair_profile_restore_padding_level(repairProfile);
    if (rtgmc_search_repair_profile_restore_flags(repairProfile) & RTGMC_SEARCH_REPAIR_RESTORE_LEVEL4_PATH) {
        return rtgmc_search_prefilter_correction_gate_level4_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, py, smoothRadius, positive);
    }
    switch (restorePaddingLevel) {
    case 0:
        return rtgmc_search_prefilter_correction_gate_base_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive);
    case 1:
        return rtgmc_search_prefilter_correction_gate_rank_smooth1_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive, positive);
    case 2:
        return rtgmc_search_prefilter_correction_gate_rank_smooth2_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive, positive);
    default:
        return rtgmc_search_prefilter_correction_gate_area_envelope_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, py, smoothRadius, repairProfile, positive, positive);
    }
}

// Field-parity correction for the temporal-search prefilter.
inline int rtgmc_search_prefilter_apply_field_correction_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int smoothRadius,
    const uint repairProfile) {
    const int base = rtgmc_search_prefilter_temporal_candidate_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight,
        px, py, smoothRadius);
    const int diff = rtgmc_search_prefilter_search_correction_delta_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight,
        px, py, smoothRadius);
    const int positiveMask = rtgmc_search_prefilter_correction_gate_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight,
        px, py, smoothRadius, repairProfile, 1);
    const int negativeMask = rtgmc_search_prefilter_correction_gate_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight,
        px, py, smoothRadius, repairProfile, 0);
    const int rangeHalf = rtgmc_search_prefilter_range_half();
    return rtgmc_search_prefilter_apply_signed_correction(
        base,
        diff - rangeHalf,
        positiveMask - rangeHalf,
        negativeMask - rangeHalf,
        rtgmc_search_prefilter_range_scale());
}

inline int rtgmc_search_prefilter_field_corrected_search_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int smoothRadius,
    const uint repairProfile) {
    if (rtgmc_search_repair_profile_restore_flags(repairProfile) & RTGMC_SEARCH_REPAIR_RESTORE_ENABLED) {
        return rtgmc_search_prefilter_apply_field_correction_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight,
            px, py, smoothRadius, repairProfile);
    }
    return rtgmc_search_prefilter_temporal_candidate_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight,
        px, py, smoothRadius);
}

inline int rtgmc_search_prefilter_half_search_base_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth,
    const int srcHeight,
    const int hx,
    const int hy,
    const int smoothRadius,
    const uint repairProfile) {
    const int filterSize = 4;
    const float filterSupport = 2.0f;
    const float filterStep = 0.5f;
    const float posY = 0.5f + 2.0f * (float)hy;
    int endY = (int)(posY + filterSupport);
    endY = min(endY, srcHeight - 1);
    int startY = max(endY - filterSize + 1, 0);
    const float okPosY = clamp(posY, 0.0f, (float)(srcHeight - 1));

    float totalY = 0.0f;
    float coeffY[4];
    for (int iy = 0; iy < filterSize; iy++) {
        const float d = fabs(((float)(startY + iy) - okPosY) * filterStep);
        coeffY[iy] = (d < 1.0f) ? (1.0f - d) : 0.0f;
        totalY += coeffY[iy];
    }
    totalY = (totalY == 0.0f) ? 1.0f : totalY;

    const float posX = 0.5f + 2.0f * (float)hx;
    int endX = (int)(posX + filterSupport);
    endX = min(endX, srcWidth - 1);
    int startX = max(endX - filterSize + 1, 0);
    const float okPosX = clamp(posX, 0.0f, (float)(srcWidth - 1));

    float totalX = 0.0f;
    float coeffX[4];
    for (int ix = 0; ix < filterSize; ix++) {
        const float d = fabs(((float)(startX + ix) - okPosX) * filterStep);
        coeffX[ix] = (d < 1.0f) ? (1.0f - d) : 0.0f;
        totalX += coeffX[ix];
    }
    totalX = (totalX == 0.0f) ? 1.0f : totalX;

    float sumY = 0.5f;
    for (int iy = 0; iy < filterSize; iy++) {
        float sumX = 0.5f;
        for (int ix = 0; ix < filterSize; ix++) {
            const int sample = rtgmc_search_prefilter_field_corrected_search_value(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
                srcWidth, srcHeight,
                startX + ix, startY + iy, smoothRadius, repairProfile);
            sumX += (coeffX[ix] / totalX) * (float)sample;
        }
        const int rowValue = clamp((int)sumX, 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
        sumY += (coeffY[iy] / totalY) * (float)rowValue;
    }
    return clamp((int)sumY, 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
}

inline int rtgmc_search_prefilter_half_search_smoothed_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth,
    const int srcHeight,
    const int hx,
    const int hy,
    const int smoothRadius,
    const uint repairProfile) {
    const int halfWidth = max(srcWidth >> 1, 1);
    const int halfHeight = max(srcHeight >> 1, 1);
    if (hx <= 0 || hy <= 0 || hx >= halfWidth - 1 || hy >= halfHeight - 1) {
        return rtgmc_search_prefilter_half_search_base_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight,
            clamp(hx, 0, halfWidth - 1), clamp(hy, 0, halfHeight - 1), smoothRadius, repairProfile);
    }
    const int x0 = clamp(hx - 1, 0, halfWidth - 1);
    const int x1 = clamp(hx,     0, halfWidth - 1);
    const int x2 = clamp(hx + 1, 0, halfWidth - 1);
    const int y0 = clamp(hy - 1, 0, halfHeight - 1);
    const int y1 = clamp(hy,     0, halfHeight - 1);
    const int y2 = clamp(hy + 1, 0, halfHeight - 1);
    const int p00 = rtgmc_search_prefilter_half_search_base_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x0, y0, smoothRadius, repairProfile);
    const int p10 = rtgmc_search_prefilter_half_search_base_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x1, y0, smoothRadius, repairProfile);
    const int p20 = rtgmc_search_prefilter_half_search_base_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x2, y0, smoothRadius, repairProfile);
    const int p01 = rtgmc_search_prefilter_half_search_base_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x0, y1, smoothRadius, repairProfile);
    const int p11 = rtgmc_search_prefilter_half_search_base_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x1, y1, smoothRadius, repairProfile);
    const int p21 = rtgmc_search_prefilter_half_search_base_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x2, y1, smoothRadius, repairProfile);
    const int p02 = rtgmc_search_prefilter_half_search_base_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x0, y2, smoothRadius, repairProfile);
    const int p12 = rtgmc_search_prefilter_half_search_base_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x1, y2, smoothRadius, repairProfile);
    const int p22 = rtgmc_search_prefilter_half_search_base_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, x2, y2, smoothRadius, repairProfile);
    return rtgmc_search_prefilter_blur3x3_weighted(p00, p10, p20, p01, p11, p21, p02, p12, p22);
}

inline int rtgmc_search_prefilter_half_resolution_search_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int smoothRadius,
    const uint repairProfile) {
    const int halfWidth = max(srcWidth >> 1, 1);
    const int halfHeight = max(srcHeight >> 1, 1);
    const int filterSize = 2;
    const float filterSupport = 1.0f;
    const float posY = -0.25f + 0.5f * (float)py;
    int endY = (int)(posY + filterSupport);
    endY = min(endY, halfHeight - 1);
    int startY = max(endY - filterSize + 1, 0);
    const float okPosY = clamp(posY, 0.0f, (float)(halfHeight - 1));

    float totalY = 0.0f;
    float coeffY[2];
    for (int iy = 0; iy < filterSize; iy++) {
        const float d = fabs((float)(startY + iy) - okPosY);
        coeffY[iy] = (d < 1.0f) ? (1.0f - d) : 0.0f;
        totalY += coeffY[iy];
    }
    totalY = (totalY == 0.0f) ? 1.0f : totalY;

    const float posX = -0.25f + 0.5f * (float)px;
    int endX = (int)(posX + filterSupport);
    endX = min(endX, halfWidth - 1);
    int startX = max(endX - filterSize + 1, 0);
    const float okPosX = clamp(posX, 0.0f, (float)(halfWidth - 1));

    float totalX = 0.0f;
    float coeffX[2];
    for (int ix = 0; ix < filterSize; ix++) {
        const float d = fabs((float)(startX + ix) - okPosX);
        coeffX[ix] = (d < 1.0f) ? (1.0f - d) : 0.0f;
        totalX += coeffX[ix];
    }
    totalX = (totalX == 0.0f) ? 1.0f : totalX;

    float sumY = 0.5f;
    for (int iy = 0; iy < filterSize; iy++) {
        float sumX = 0.5f;
        for (int ix = 0; ix < filterSize; ix++) {
            const int sample = rtgmc_search_prefilter_half_search_smoothed_value(
                srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
                srcWidth, srcHeight,
                clamp(startX + ix, 0, halfWidth - 1),
                clamp(startY + iy, 0, halfHeight - 1),
                smoothRadius, repairProfile);
            sumX += (coeffX[ix] / totalX) * (float)sample;
        }
        const int rowValue = clamp((int)sumX, 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
        sumY += (coeffY[iy] / totalY) * (float)rowValue;
    }
    return clamp((int)sumY, 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
}

inline int rtgmc_search_prefilter_field_corrected_search_weighted3x3_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int smoothRadius,
    const uint repairProfile) {
    if (px <= 0 || py <= 0 || px >= srcWidth - 1 || py >= srcHeight - 1) {
        return rtgmc_search_prefilter_field_corrected_search_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight, px, py, smoothRadius, repairProfile);
    }
    const int p00 = rtgmc_search_prefilter_field_corrected_search_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px - 1, py - 1, smoothRadius, repairProfile);
    const int p10 = rtgmc_search_prefilter_field_corrected_search_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px,     py - 1, smoothRadius, repairProfile);
    const int p20 = rtgmc_search_prefilter_field_corrected_search_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + 1, py - 1, smoothRadius, repairProfile);
    const int p01 = rtgmc_search_prefilter_field_corrected_search_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px - 1, py,     smoothRadius, repairProfile);
    const int p11 = rtgmc_search_prefilter_field_corrected_search_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px,     py,     smoothRadius, repairProfile);
    const int p21 = rtgmc_search_prefilter_field_corrected_search_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + 1, py,     smoothRadius, repairProfile);
    const int p02 = rtgmc_search_prefilter_field_corrected_search_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px - 1, py + 1, smoothRadius, repairProfile);
    const int p12 = rtgmc_search_prefilter_field_corrected_search_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px,     py + 1, smoothRadius, repairProfile);
    const int p22 = rtgmc_search_prefilter_field_corrected_search_value(srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch, srcWidth, srcHeight, px + 1, py + 1, smoothRadius, repairProfile);
    return rtgmc_search_prefilter_blur3x3_weighted(p00, p10, p20, p01, p11, p21, p02, p12, p22);
}

inline int rtgmc_search_prefilter_motion_guide_blend_value(const int spatialGuide, const int motionGuide) {
    const float guideWeight = 0.10f;
    const float value = mix((float)spatialGuide, (float)motionGuide, guideWeight);
    return clamp(convert_int_rte(value), 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
}

inline int rtgmc_search_prefilter_search_smoothed3x3_value(
    __global const uchar *src,
    const int src_pitch,
    const int width,
    const int height,
    const int x,
    const int y) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return rtgmc_search_prefilter_pixel_load(src, src_pitch, width, height, x, y);
    }
    const int p00 = rtgmc_search_prefilter_pixel_load(src, src_pitch, width, height, x - 1, y - 1);
    const int p10 = rtgmc_search_prefilter_pixel_load(src, src_pitch, width, height, x,     y - 1);
    const int p20 = rtgmc_search_prefilter_pixel_load(src, src_pitch, width, height, x + 1, y - 1);
    const int p01 = rtgmc_search_prefilter_pixel_load(src, src_pitch, width, height, x - 1, y);
    const int p11 = rtgmc_search_prefilter_pixel_load(src, src_pitch, width, height, x,     y);
    const int p21 = rtgmc_search_prefilter_pixel_load(src, src_pitch, width, height, x + 1, y);
    const int p02 = rtgmc_search_prefilter_pixel_load(src, src_pitch, width, height, x - 1, y + 1);
    const int p12 = rtgmc_search_prefilter_pixel_load(src, src_pitch, width, height, x,     y + 1);
    const int p22 = rtgmc_search_prefilter_pixel_load(src, src_pitch, width, height, x + 1, y + 1);
    return rtgmc_search_prefilter_blur3x3_weighted(p00, p10, p20, p01, p11, p21, p02, p12, p22);
}

inline int rtgmc_search_prefilter_motion_guide_stabilize_value(
    const int motionGuide,
    const int fieldGuide,
    const int spatialGuide) {
    const float guideEnvelope = 4.0f;
    const float residualGain = 0.50f;
    const float residualLimit = 3.0f;
    const float scale = (float)rtgmc_search_prefilter_range_scale();
    const float invScale = 1.0f / scale;
    const float motionGuidef = motionGuide * invScale;
    const float fieldGuidef = fieldGuide * invScale;
    const float spatialGuidef = spatialGuide * invScale;
    const float candidate = clamp(fieldGuidef, motionGuidef - guideEnvelope, motionGuidef + guideEnvelope);

    // Smooth bounded residual correction around the spatial guide.
    const float residual = candidate - spatialGuidef;
    const float normalized = residual * (residualGain / residualLimit);
    const float correction = residualGain * residual * native_rsqrt(1.0f + normalized * normalized);
    const float ret = spatialGuidef + correction;
    return rtgmc_search_prefilter_round_float_to_pixel(ret * scale);
}

inline int rtgmc_search_prefilter_motion_guide_blend_stabilized_value(
    const int spatialGuide,
    const int motionGuide,
    const int fieldGuide) {
    const int blendedGuide = rtgmc_search_prefilter_motion_guide_blend_value(spatialGuide, motionGuide);
    return rtgmc_search_prefilter_motion_guide_stabilize_value(motionGuide, fieldGuide, blendedGuide);
}

inline int rtgmc_search_prefilter_value(
    __global const uchar *srcPrev2,
    __global const uchar *srcPrev,
    __global const uchar *srcCur,
    __global const uchar *srcNext,
    __global const uchar *srcNext2,
    const int pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int smoothRadius,
    const int search_refine,
    const uint repairProfile) {
    if (search_refine >= 1) {
        return rtgmc_search_prefilter_half_resolution_search_value(
            srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
            srcWidth, srcHeight,
            px, py, smoothRadius, repairProfile);
    }
    return rtgmc_search_prefilter_field_corrected_search_value(
        srcPrev2, srcPrev, srcCur, srcNext, srcNext2, pitch,
        srcWidth, srcHeight,
        px, py, smoothRadius, repairProfile);
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_scenechange(
    __global const uchar *prev2,
    __global const uchar *prev,
    __global const uchar *cur,
    __global const uchar *next,
    __global const uchar *next2,
    const int src_pitch,
    __global uint *partial,
    const int groupCount,
    const int width,
    const int height) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    const int lid = (int)get_local_id(1) * rtgmc_search_prefilter_block_x + (int)get_local_id(0);
    const int groupIndex = (int)get_group_id(1) * (int)get_num_groups(0) + (int)get_group_id(0);

    __local uint sadPrev[RTGMC_SEARCH_PREFILTER_BLOCK_PIXELS];
    __local uint sadNext[RTGMC_SEARCH_PREFILTER_BLOCK_PIXELS];
    __local uint sadPrev2[RTGMC_SEARCH_PREFILTER_BLOCK_PIXELS];
    __local uint sadNext2[RTGMC_SEARCH_PREFILTER_BLOCK_PIXELS];

    uint diffPrev = 0;
    uint diffNext = 0;
    uint diffPrev2 = 0;
    uint diffNext2 = 0;
    if (x < width && y < height) {
        const int value = rtgmc_search_prefilter_pixel_load(cur, src_pitch, width, height, x, y);
        diffPrev = (uint)abs(value - rtgmc_search_prefilter_pixel_load(prev, src_pitch, width, height, x, y));
        diffNext = (uint)abs(value - rtgmc_search_prefilter_pixel_load(next, src_pitch, width, height, x, y));
        diffPrev2 = (uint)abs(value - rtgmc_search_prefilter_pixel_load(prev2, src_pitch, width, height, x, y));
        diffNext2 = (uint)abs(value - rtgmc_search_prefilter_pixel_load(next2, src_pitch, width, height, x, y));
    }
    sadPrev[lid] = diffPrev;
    sadNext[lid] = diffNext;
    sadPrev2[lid] = diffPrev2;
    sadNext2[lid] = diffNext2;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = RTGMC_SEARCH_PREFILTER_BLOCK_PIXELS >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            sadPrev[lid] += sadPrev[lid + stride];
            sadNext[lid] += sadNext[lid + stride];
            sadPrev2[lid] += sadPrev2[lid + stride];
            sadNext2[lid] += sadNext2[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0 && groupIndex < groupCount) {
        partial[groupIndex + groupCount * 0] = sadPrev[0];
        partial[groupIndex + groupCount * 1] = sadNext[0];
        partial[groupIndex + groupCount * 2] = sadPrev2[0];
        partial[groupIndex + groupCount * 3] = sadNext2[0];
    }
}

inline int rtgmc_search_prefilter_to_full_range(
    const int value,
    const int planeMode) {
    if (planeMode == 1) {
        return ((value - RTGMC_SEARCH_PREFILTER_LIMITED_Y_MIN) * RTGMC_SEARCH_PREFILTER_PIXEL_MAX
            + (RTGMC_SEARCH_PREFILTER_LIMITED_Y_RANGE >> 1)) / RTGMC_SEARCH_PREFILTER_LIMITED_Y_RANGE;
    }
    if (planeMode == 2) {
        const float rangeHalfF = (float)((RTGMC_SEARCH_PREFILTER_PIXEL_MAX + 1) >> 1);
        const float converted = ((float)value - (float)RTGMC_SEARCH_PREFILTER_LIMITED_C_OFFSET)
            * (rangeHalfF / (float)RTGMC_SEARCH_PREFILTER_LIMITED_C_RANGE)
            + rangeHalfF;
        return clamp((int)(converted + 0.5f), 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
    }
    return value;
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_field_stable_search(
    __global const uchar *prev2,
    __global const uchar *prev,
    __global const uchar *cur,
    __global const uchar *next,
    __global const uchar *next2,
    const int pitch,
    __global uchar *dst,
    const int width,
    const int height,
    const int tr0,
    const uint repairProfile) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int value = rtgmc_search_prefilter_field_corrected_search_value(
        prev2, prev, cur, next, next2, pitch,
        width, height,
        x, y, tr0, repairProfile);
    rtgmc_search_prefilter_pixel_store(dst, pitch, x, y, value);
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_search_smoothed3x3(
    __global const uchar *src,
    const int pitch,
    __global uchar *dst,
    const int width,
    const int height) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    rtgmc_search_prefilter_pixel_store(dst, pitch, x, y,
        rtgmc_search_prefilter_search_smoothed3x3_value(src, pitch, width, height, x, y));
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_refine2_tile(
    __global const uchar *motionGuide,
    const int pitch,
    __global uchar *dst,
    const int width,
    const int height,
    const int fullRangeMode) {
    const int lx = (int)get_local_id(0);
    const int ly = (int)get_local_id(1);
    const int localIndex = ly * rtgmc_search_prefilter_block_x + lx;
    const int localCount = rtgmc_search_prefilter_block_x * rtgmc_search_prefilter_block_y;
    const int tileW = rtgmc_search_prefilter_block_x + 8;
    const int tileH = rtgmc_search_prefilter_block_y + 8;
    const int groupX = (int)get_group_id(0) * rtgmc_search_prefilter_block_x;
    const int groupY = (int)get_group_id(1) * rtgmc_search_prefilter_block_y;

    __local int smoothTile[(rtgmc_search_prefilter_block_x + 8) * (rtgmc_search_prefilter_block_y + 8)];
    __local float gaussHTile[(rtgmc_search_prefilter_block_y + 8) * rtgmc_search_prefilter_block_x];

    for (int i = localIndex; i < tileW * tileH; i += localCount) {
        const int tx = i % tileW;
        const int ty = i / tileW;
        const int sx = clamp(groupX + tx - 4, 0, width - 1);
        const int sy = clamp(groupY + ty - 4, 0, height - 1);
        smoothTile[i] = rtgmc_search_prefilter_search_smoothed3x3_value(
            motionGuide, pitch, width, height, sx, sy);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = localIndex; i < tileH * rtgmc_search_prefilter_block_x; i += localCount) {
        const int hx = i % rtgmc_search_prefilter_block_x;
        const int hy = i / rtgmc_search_prefilter_block_x;
        const int base = hy * tileW + hx;
        const float value =
            (float)smoothTile[base + 0] * RTGMC_SEARCH_REFINE2_GAUSS_W4 +
            (float)smoothTile[base + 1] * RTGMC_SEARCH_REFINE2_GAUSS_W3 +
            (float)smoothTile[base + 2] * RTGMC_SEARCH_REFINE2_GAUSS_W2 +
            (float)smoothTile[base + 3] * RTGMC_SEARCH_REFINE2_GAUSS_W1 +
            (float)smoothTile[base + 4] * RTGMC_SEARCH_REFINE2_GAUSS_W0 +
            (float)smoothTile[base + 5] * RTGMC_SEARCH_REFINE2_GAUSS_W1 +
            (float)smoothTile[base + 6] * RTGMC_SEARCH_REFINE2_GAUSS_W2 +
            (float)smoothTile[base + 7] * RTGMC_SEARCH_REFINE2_GAUSS_W3 +
            (float)smoothTile[base + 8] * RTGMC_SEARCH_REFINE2_GAUSS_W4;
        gaussHTile[i] = value;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const float blur =
        gaussHTile[(ly + 0) * rtgmc_search_prefilter_block_x + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W4 +
        gaussHTile[(ly + 1) * rtgmc_search_prefilter_block_x + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W3 +
        gaussHTile[(ly + 2) * rtgmc_search_prefilter_block_x + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W2 +
        gaussHTile[(ly + 3) * rtgmc_search_prefilter_block_x + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W1 +
        gaussHTile[(ly + 4) * rtgmc_search_prefilter_block_x + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W0 +
        gaussHTile[(ly + 5) * rtgmc_search_prefilter_block_x + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W1 +
        gaussHTile[(ly + 6) * rtgmc_search_prefilter_block_x + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W2 +
        gaussHTile[(ly + 7) * rtgmc_search_prefilter_block_x + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W3 +
        gaussHTile[(ly + 8) * rtgmc_search_prefilter_block_x + lx] * RTGMC_SEARCH_REFINE2_GAUSS_W4;
    const int spatialGuideValue = (int)(clamp(blur, 0.0f, (float)RTGMC_SEARCH_PREFILTER_PIXEL_MAX) + 0.5f);
    const int motionGuideValue = rtgmc_search_prefilter_pixel_load(motionGuide, pitch, width, height, x, y);
    int value = rtgmc_search_prefilter_motion_guide_blend_value(spatialGuideValue, motionGuideValue);
    value = rtgmc_search_prefilter_to_full_range(value, fullRangeMode);
    rtgmc_search_prefilter_pixel_store(dst, pitch, x, y, value);
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_softened_search_blend(
    __global const uchar *spatialGuide,
    __global const uchar *motionGuide,
    __global uchar *dst,
    const int pitch,
    const int width,
    const int height,
    const int fullRangeMode) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int spatialGuideValue = rtgmc_search_prefilter_pixel_load(spatialGuide, pitch, width, height, x, y);
    const int motionGuideValue = rtgmc_search_prefilter_pixel_load(motionGuide, pitch, width, height, x, y);
    int value = rtgmc_search_prefilter_motion_guide_blend_value(spatialGuideValue, motionGuideValue);
    value = rtgmc_search_prefilter_to_full_range(value, fullRangeMode);
    rtgmc_search_prefilter_pixel_store(dst, pitch, x, y, value);
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_softened_search_blend_stabilized(
    __global const uchar *spatialGuide,
    __global const uchar *motionGuide,
    __global const uchar *fieldGuide,
    __global uchar *dst,
    const int pitch,
    const int width,
    const int height,
    const int fullRangeMode) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int spatialGuideValue = rtgmc_search_prefilter_pixel_load(spatialGuide, pitch, width, height, x, y);
    const int motionGuideValue = rtgmc_search_prefilter_pixel_load(motionGuide, pitch, width, height, x, y);
    const int fieldGuideValue = rtgmc_search_prefilter_pixel_load(fieldGuide, pitch, width, height, x, y);
    int value = rtgmc_search_prefilter_motion_guide_blend_stabilized_value(spatialGuideValue, motionGuideValue, fieldGuideValue);
    value = rtgmc_search_prefilter_to_full_range(value, fullRangeMode);
    rtgmc_search_prefilter_pixel_store(dst, pitch, x, y, value);
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_stabilized_search(
    __global const uchar *motionGuide,
    __global const uchar *fieldGuide,
    __global const uchar *spatialGuide,
    __global uchar *dst,
    const int pitch,
    const int width,
    const int height,
    const int fullRangeMode) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int motionGuideValue = rtgmc_search_prefilter_pixel_load(motionGuide, pitch, width, height, x, y);
    const int fieldGuideValue = rtgmc_search_prefilter_pixel_load(fieldGuide, pitch, width, height, x, y);
    const int spatialGuideValue = rtgmc_search_prefilter_pixel_load(spatialGuide, pitch, width, height, x, y);
    int value = rtgmc_search_prefilter_motion_guide_stabilize_value(motionGuideValue, fieldGuideValue, spatialGuideValue);
    value = rtgmc_search_prefilter_to_full_range(value, fullRangeMode);
    rtgmc_search_prefilter_pixel_store(dst, pitch, x, y, value);
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_half_search_base(
    __global const uchar *prev2,
    __global const uchar *prev,
    __global const uchar *cur,
    __global const uchar *next,
    __global const uchar *next2,
    const int src_pitch,
    __global uchar *dst,
    const int dst_pitch,
    const int width,
    const int height,
    const int tr0,
    const uint repairProfile) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    const int halfWidth = max(width >> 1, 1);
    const int halfHeight = max(height >> 1, 1);
    if (x >= halfWidth || y >= halfHeight) {
        return;
    }
    const int value = rtgmc_search_prefilter_half_search_base_value(
        prev2, prev, cur, next, next2, src_pitch,
        width, height,
        x, y, tr0, repairProfile);
    rtgmc_search_prefilter_pixel_store(dst, dst_pitch, x, y, value);
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_half_search_smoothed(
    __global const uchar *prev2,
    __global const uchar *prev,
    __global const uchar *cur,
    __global const uchar *next,
    __global const uchar *next2,
    const int src_pitch,
    __global uchar *dst,
    const int dst_pitch,
    const int width,
    const int height,
    const int tr0,
    const uint repairProfile) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    const int halfWidth = max(width >> 1, 1);
    const int halfHeight = max(height >> 1, 1);
    if (x >= halfWidth || y >= halfHeight) {
        return;
    }
    const int value = rtgmc_search_prefilter_half_search_smoothed_value(
        prev2, prev, cur, next, next2, src_pitch,
        width, height,
        x, y, tr0, repairProfile);
    rtgmc_search_prefilter_pixel_store(dst, dst_pitch, x, y, value);
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_range_convert(
    __global uchar *dst,
    const int dst_pitch,
    const int width,
    const int height,
    const int fullRangeMode) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int value = rtgmc_search_prefilter_to_full_range(
        rtgmc_search_prefilter_pixel_load(dst, dst_pitch, width, height, x, y),
        fullRangeMode);
    rtgmc_search_prefilter_pixel_store(dst, dst_pitch, x, y, value);
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_debug_temporal_candidate(
    __global const uchar *prev2,
    __global const uchar *prev,
    __global const uchar *cur,
    __global const uchar *next,
    __global const uchar *next2,
    const int pitch,
    __global uchar *dst,
    const int width,
    const int height,
    const int tr0,
    const uint repairProfile) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int value = rtgmc_search_prefilter_temporal_candidate_value(
        prev2, prev, cur, next, next2, pitch,
        width, height,
        x, y, tr0);
    rtgmc_search_prefilter_pixel_store(dst, pitch, x, y, value);
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_debug_field_stable_search(
    __global const uchar *prev2,
    __global const uchar *prev,
    __global const uchar *cur,
    __global const uchar *next,
    __global const uchar *next2,
    const int pitch,
    __global uchar *dst,
    const int width,
    const int height,
    const int tr0,
    const uint repairProfile) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int value = rtgmc_search_prefilter_field_corrected_search_value(
        prev2, prev, cur, next, next2, pitch,
        width, height,
        x, y, tr0, repairProfile);
    rtgmc_search_prefilter_pixel_store(dst, pitch, x, y, value);
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_debug_search_correction_delta(
    __global const uchar *prev2,
    __global const uchar *prev,
    __global const uchar *cur,
    __global const uchar *next,
    __global const uchar *next2,
    const int pitch,
    __global uchar *dst,
    const int width,
    const int height,
    const int tr0,
    const uint repairProfile) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int value = rtgmc_search_prefilter_search_correction_delta_value(
        prev2, prev, cur, next, next2, pitch,
        width, height,
        x, y, tr0);
    rtgmc_search_prefilter_pixel_store(dst, pitch, x, y, value);
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_debug_positive_correction_gate(
    __global const uchar *prev2,
    __global const uchar *prev,
    __global const uchar *cur,
    __global const uchar *next,
    __global const uchar *next2,
    const int pitch,
    __global uchar *dst,
    const int width,
    const int height,
    const int tr0,
    const uint repairProfile) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int value = rtgmc_search_prefilter_correction_gate_value(
        prev2, prev, cur, next, next2, pitch,
        width, height,
        x, y, tr0, repairProfile, 1);
    rtgmc_search_prefilter_pixel_store(dst, pitch, x, y, value);
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_debug_negative_correction_gate(
    __global const uchar *prev2,
    __global const uchar *prev,
    __global const uchar *cur,
    __global const uchar *next,
    __global const uchar *next2,
    const int pitch,
    __global uchar *dst,
    const int width,
    const int height,
    const int tr0,
    const uint repairProfile) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int value = rtgmc_search_prefilter_correction_gate_value(
        prev2, prev, cur, next, next2, pitch,
        width, height,
        x, y, tr0, repairProfile, 0);
    rtgmc_search_prefilter_pixel_store(dst, pitch, x, y, value);
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_luma(
    __global const uchar *prev2,
    __global const uchar *prev,
    __global const uchar *cur,
    __global const uchar *next,
    __global const uchar *next2,
    const int pitch,
    __global uchar *dst,
    const int width,
    const int height,
    const int tr0,
    const int search_refine,
    const uint repairProfile,
    const int fullRangeMode) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }

    int value = rtgmc_search_prefilter_value(
        prev2, prev, cur, next, next2, pitch,
        width, height,
        x, y, tr0, search_refine, repairProfile);
    value = rtgmc_search_prefilter_to_full_range(value, fullRangeMode);
    rtgmc_search_prefilter_pixel_store(dst, pitch, x, y, value);
}
