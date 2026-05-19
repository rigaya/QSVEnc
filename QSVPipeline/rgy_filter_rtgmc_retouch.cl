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
#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

static inline int read_pix_clamped(
    const __global uchar *src, int x, int y,
    const int pitch, const int width, const int height
) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    return (int)(*(const __global Type *)(src + y * pitch + x * (int)sizeof(Type)));
}

static inline void write_pix(
    __global uchar *dst, int x, int y, const int pitch, const int value
) {
    *(__global Type *)(dst + y * pitch + x * (int)sizeof(Type)) = (Type)clamp(value, 0, max_val);
}

static inline int rtgmc_retouch_median3(const int a, const int b, const int c) {
    const int lo = min(a, b);
    const int hi = max(a, b);
    return max(lo, min(hi, c));
}

static inline void rtgmc_retouch_sort2(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = lo;
    *b = hi;
}

static inline void rtgmc_retouch_sort2_desc(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = hi;
    *b = lo;
}

// Batcher's Bitonic Sort (1968), 8 elements / 24 comparisons / depth 6.
static inline void rtgmc_retouch_sort8(__private int *v) {
    rtgmc_retouch_sort2     (&v[0], &v[1]); rtgmc_retouch_sort2_desc(&v[2], &v[3]); rtgmc_retouch_sort2     (&v[4], &v[5]); rtgmc_retouch_sort2_desc(&v[6], &v[7]);
    rtgmc_retouch_sort2     (&v[0], &v[2]); rtgmc_retouch_sort2     (&v[1], &v[3]); rtgmc_retouch_sort2_desc(&v[4], &v[6]); rtgmc_retouch_sort2_desc(&v[5], &v[7]);
    rtgmc_retouch_sort2     (&v[0], &v[1]); rtgmc_retouch_sort2     (&v[2], &v[3]); rtgmc_retouch_sort2_desc(&v[4], &v[5]); rtgmc_retouch_sort2_desc(&v[6], &v[7]);
    rtgmc_retouch_sort2     (&v[0], &v[4]); rtgmc_retouch_sort2     (&v[1], &v[5]); rtgmc_retouch_sort2     (&v[2], &v[6]); rtgmc_retouch_sort2     (&v[3], &v[7]);
    rtgmc_retouch_sort2     (&v[0], &v[2]); rtgmc_retouch_sort2     (&v[1], &v[3]); rtgmc_retouch_sort2     (&v[4], &v[6]); rtgmc_retouch_sort2     (&v[5], &v[7]);
    rtgmc_retouch_sort2     (&v[0], &v[1]); rtgmc_retouch_sort2     (&v[2], &v[3]); rtgmc_retouch_sort2     (&v[4], &v[5]); rtgmc_retouch_sort2     (&v[6], &v[7]);
}

static inline int rtgmc_retouch_detail_ref_vertical_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    const int pixCenter = read_pix_clamped(src, x, y, pitch, width, height);
    const int pixUpper = (y > 0) ? read_pix_clamped(src, x, y - 1, pitch, width, height) : pixCenter;
    const int pixLower = (y + 1 < height) ? read_pix_clamped(src, x, y + 1, pitch, width, height) : pixCenter;
    const int triadSum = pixUpper + pixCenter + pixLower;
    const int pairLowerMin = min(pixUpper, pixCenter);
    const int pairLowerMax = max(pixUpper, pixCenter);
    const int triadMedian = max(pairLowerMin, min(pairLowerMax, pixLower));
    return (triadSum - triadMedian + 1) >> 1;
}

static inline int rtgmc_retouch_removegrain12_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return read_pix_clamped(src, x, y, pitch, width, height);
    }
    const int p00 = read_pix_clamped(src, x - 1, y - 1, pitch, width, height);
    const int p10 = read_pix_clamped(src, x,     y - 1, pitch, width, height);
    const int p20 = read_pix_clamped(src, x + 1, y - 1, pitch, width, height);
    const int p01 = read_pix_clamped(src, x - 1, y,     pitch, width, height);
    const int p11 = read_pix_clamped(src, x,     y,     pitch, width, height);
    const int p21 = read_pix_clamped(src, x + 1, y,     pitch, width, height);
    const int p02 = read_pix_clamped(src, x - 1, y + 1, pitch, width, height);
    const int p12 = read_pix_clamped(src, x,     y + 1, pitch, width, height);
    const int p22 = read_pix_clamped(src, x + 1, y + 1, pitch, width, height);
    const int sum =
        p00 + 2 * p10 + p20 +
        2 * p01 + 4 * p11 + 2 * p21 +
        p02 + 2 * p12 + p22;
    return (sum + 8) >> 4;
}

static inline int rtgmc_retouch_removegrain_smooth_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    return rtgmc_retouch_removegrain12_value(src, x, y, pitch, width, height);
}

static inline int rtgmc_retouch_verticalcleaner1_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    if (y <= 0 || y >= height - 1) {
        return read_pix_clamped(src, x, y, pitch, width, height);
    }
    const int top = read_pix_clamped(src, x, y - 1, pitch, width, height);
    const int center = read_pix_clamped(src, x, y, pitch, width, height);
    const int bottom = read_pix_clamped(src, x, y + 1, pitch, width, height);
    return rtgmc_retouch_median3(top, center, bottom);
}

static inline int rtgmc_retouch_blur10h_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    const int center = read_pix_clamped(src, x, y, pitch, width, height);
    const int left = (x > 0) ? read_pix_clamped(src, x - 1, y, pitch, width, height) : center;
    const int right = (x + 1 < width) ? read_pix_clamped(src, x + 1, y, pitch, width, height) : center;
    return (left + 2 * center + right + 2) >> 2;
}

static inline int rtgmc_retouch_precise_clamp_value(const int src, const int ref) {
    if (src < ref) {
        return min(src + 1, max_val);
    }
    if (src > ref) {
        return max(src - 1, 0);
    }
    return src;
}

static inline int rtgmc_retouch_make_diff_value(const int a, const int b) {
    return clamp(a - b + range_half, 0, max_val);
}

static inline int rtgmc_retouch_add_diff_value(const int src, const int diff) {
    return clamp(src + diff - range_half, 0, max_val);
}

static inline int rtgmc_retouch_round_clamp(const float value) {
    return (int)(clamp(value, 0.0f, (float)max_val) + 0.5f);
}

static inline void rtgmc_retouch_ref_ring_minmax(
    const __global uchar *ref,
    const int x, const int y,
    const int refPitch,
    const int width, const int height,
    __private int *minv,
    __private int *maxv
) {
#pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
#pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            if (dx != 0 || dy != 0) {
                const int sample = read_pix_clamped(ref, x + dx, y + dy, refPitch, width, height);
                *minv = min(*minv, sample);
                *maxv = max(*maxv, sample);
            }
        }
    }
}

static inline void rtgmc_retouch_collect_ref_ring(
    __private int *dst,
    const __global uchar *ref,
    const int x, const int y,
    const int refPitch,
    const int width, const int height
) {
    int count = 0;
#pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
#pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            if (dx != 0 || dy != 0) {
                dst[count++] = read_pix_clamped(ref, x + dx, y + dy, refPitch, width, height);
            }
        }
    }
}

static inline int rtgmc_retouch_repair_mode1_value(
    const __global uchar *src, const __global uchar *ref,
    const int x, const int y,
    const int srcPitch, const int refPitch,
    const int width, const int height
) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return read_pix_clamped(src, x, y, srcPitch, width, height);
    }
    const int s = read_pix_clamped(src, x, y, srcPitch, width, height);
    int minv = s;
    int maxv = s;
    rtgmc_retouch_ref_ring_minmax(ref, x, y, refPitch, width, height, &minv, &maxv);
    return clamp(s, minv, maxv);
}

static inline int rtgmc_retouch_repair_mode12_value(
    const __global uchar *src, const __global uchar *ref,
    const int x, const int y,
    const int srcPitch, const int refPitch,
    const int width, const int height
) {
    const int s = read_pix_clamped(src, x, y, srcPitch, width, height);
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return s;
    }
    int v[8];
    rtgmc_retouch_collect_ref_ring(v, ref, x, y, refPitch, width, height);
    rtgmc_retouch_sort8(v);
    const int c = read_pix_clamped(ref, x, y, refPitch, width, height);
    const int lo = min(v[1], c);
    const int hi = max(v[6], c);
    return clamp(s, lo, hi);
}

static inline int rtgmc_retouch_removegrain12_diff_value(
    const __global uchar *src, const int srcPitch,
    const __global uchar *base, const int basePitch,
    const int x, const int y, const int width, const int height
) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return rtgmc_retouch_make_diff_value(
            read_pix_clamped(src, x, y, srcPitch, width, height),
            read_pix_clamped(base, x, y, basePitch, width, height));
    }
    const int p00 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x - 1, y - 1, srcPitch, width, height),
        read_pix_clamped(base, x - 1, y - 1, basePitch, width, height));
    const int p10 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x, y - 1, srcPitch, width, height),
        read_pix_clamped(base, x, y - 1, basePitch, width, height));
    const int p20 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x + 1, y - 1, srcPitch, width, height),
        read_pix_clamped(base, x + 1, y - 1, basePitch, width, height));
    const int p01 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x - 1, y, srcPitch, width, height),
        read_pix_clamped(base, x - 1, y, basePitch, width, height));
    const int p11 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x, y, srcPitch, width, height),
        read_pix_clamped(base, x, y, basePitch, width, height));
    const int p21 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x + 1, y, srcPitch, width, height),
        read_pix_clamped(base, x + 1, y, basePitch, width, height));
    const int p02 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x - 1, y + 1, srcPitch, width, height),
        read_pix_clamped(base, x - 1, y + 1, basePitch, width, height));
    const int p12 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x, y + 1, srcPitch, width, height),
        read_pix_clamped(base, x, y + 1, basePitch, width, height));
    const int p22 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x + 1, y + 1, srcPitch, width, height),
        read_pix_clamped(base, x + 1, y + 1, basePitch, width, height));
    const int sum =
        p00 + 2 * p10 + p20 +
        2 * p01 + 4 * p11 + 2 * p21 +
        p02 + 2 * p12 + p22;
    return (sum + 8) >> 4;
}

static inline int rtgmc_retouch_detail_ref_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int precise
) {
    const int detailRef = rtgmc_retouch_detail_ref_vertical_value(src, x, y, pitch, width, height);
    if (precise == 0) {
        return detailRef;
    }
    const int srcPix = read_pix_clamped(src, x, y, pitch, width, height);
    return rtgmc_retouch_precise_clamp_value(detailRef, srcPix);
}

static inline int rtgmc_retouch_detail_ref_blur_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int precise
) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return rtgmc_retouch_detail_ref_value(src, x, y, pitch, width, height, precise);
    }
    const int p00 = rtgmc_retouch_detail_ref_value(src, x - 1, y - 1, pitch, width, height, precise);
    const int p10 = rtgmc_retouch_detail_ref_value(src, x,     y - 1, pitch, width, height, precise);
    const int p20 = rtgmc_retouch_detail_ref_value(src, x + 1, y - 1, pitch, width, height, precise);
    const int p01 = rtgmc_retouch_detail_ref_value(src, x - 1, y,     pitch, width, height, precise);
    const int p11 = rtgmc_retouch_detail_ref_value(src, x,     y,     pitch, width, height, precise);
    const int p21 = rtgmc_retouch_detail_ref_value(src, x + 1, y,     pitch, width, height, precise);
    const int p02 = rtgmc_retouch_detail_ref_value(src, x - 1, y + 1, pitch, width, height, precise);
    const int p12 = rtgmc_retouch_detail_ref_value(src, x,     y + 1, pitch, width, height, precise);
    const int p22 = rtgmc_retouch_detail_ref_value(src, x + 1, y + 1, pitch, width, height, precise);
    const int sum =
        p00 + 2 * p10 + p20 +
        2 * p01 + 4 * p11 + 2 * p21 +
        p02 + 2 * p12 + p22;
    return (sum + 8) >> 4;
}

static inline int rtgmc_retouch_stronger_non_neutral(const int candidate, const int baseline) {
    const int candidateOffset = candidate - range_half;
    const int baselineOffset = baseline - range_half;
    return (abs(candidateOffset) > abs(baselineOffset)) ? candidate : range_half;
}

static inline int rtgmc_retouch_vertical_balance_delta_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height,
    const float edgeNarrowingGain
) {
    const int srcPix = read_pix_clamped(src, x, y, pitch, width, height);
    const int cleaned = rtgmc_retouch_verticalcleaner1_value(src, x, y, pitch, width, height);
    const float value = fma((float)(cleaned - srcPix), edgeNarrowingGain, (float)range_half);
    return rtgmc_retouch_round_clamp(value);
}

static inline int rtgmc_retouch_horizontal_balance_delta_value(
    const __global uchar *src, int x, int y,
    const int pitch, const int width, const int height,
    const float edgeNarrowingGain
) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    const int center = rtgmc_retouch_vertical_balance_delta_value(src, x, y, pitch, width, height, edgeNarrowingGain);
    const int left = (x > 0)
        ? rtgmc_retouch_vertical_balance_delta_value(src, x - 1, y, pitch, width, height, edgeNarrowingGain)
        : center;
    const int right = (x + 1 < width)
        ? rtgmc_retouch_vertical_balance_delta_value(src, x + 1, y, pitch, width, height, edgeNarrowingGain)
        : center;
    return (left + 2 * center + right + 2) >> 2;
}

static inline int rtgmc_retouch_area_balance_delta_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height,
    const float edgeNarrowingGain
) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return rtgmc_retouch_horizontal_balance_delta_value(src, x, y, pitch, width, height, edgeNarrowingGain);
    }
    const int p00 = rtgmc_retouch_horizontal_balance_delta_value(src, x - 1, y - 1, pitch, width, height, edgeNarrowingGain);
    const int p10 = rtgmc_retouch_horizontal_balance_delta_value(src, x,     y - 1, pitch, width, height, edgeNarrowingGain);
    const int p20 = rtgmc_retouch_horizontal_balance_delta_value(src, x + 1, y - 1, pitch, width, height, edgeNarrowingGain);
    const int p01 = rtgmc_retouch_horizontal_balance_delta_value(src, x - 1, y,     pitch, width, height, edgeNarrowingGain);
    const int p11 = rtgmc_retouch_horizontal_balance_delta_value(src, x,     y,     pitch, width, height, edgeNarrowingGain);
    const int p21 = rtgmc_retouch_horizontal_balance_delta_value(src, x + 1, y,     pitch, width, height, edgeNarrowingGain);
    const int p02 = rtgmc_retouch_horizontal_balance_delta_value(src, x - 1, y + 1, pitch, width, height, edgeNarrowingGain);
    const int p12 = rtgmc_retouch_horizontal_balance_delta_value(src, x,     y + 1, pitch, width, height, edgeNarrowingGain);
    const int p22 = rtgmc_retouch_horizontal_balance_delta_value(src, x + 1, y + 1, pitch, width, height, edgeNarrowingGain);
    const int sum =
        p00 + 2 * p10 + p20 +
        2 * p01 + 4 * p11 + 2 * p21 +
        p02 + 2 * p12 + p22;
    return (sum + 8) >> 4;
}

static inline int rtgmc_retouch_temporal_detail_guard_value(
    const int srcPix,
    const __global uchar *ref,
    const __global uchar *motionBack,
    const __global uchar *motionForw,
    const int x, const int y,
    const int refPitch, const int motionBackPitch, const int motionForwPitch,
    const int width, const int height,
    const int sovs
) {
    const int refPix = read_pix_clamped(ref, x, y, refPitch, width, height);
    const int motionBackPix = read_pix_clamped(motionBack, x, y, motionBackPitch, width, height);
    const int motionForwPix = read_pix_clamped(motionForw, x, y, motionForwPitch, width, height);
    const int lower = min(refPix, min(motionBackPix, motionForwPix)) - sovs;
    const int upper = max(refPix, max(motionBackPix, motionForwPix)) + sovs;
    return clamp(srcPix, max(0, lower), min(max_val, upper));
}

static inline int rtgmc_retouch_spatial_min(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int radius
) {
    int value = max_val;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            value = min(value, read_pix_clamped(src, x + dx, y + dy, pitch, width, height));
        }
    }
    return value;
}

static inline int rtgmc_retouch_spatial_max(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int radius
) {
    int value = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            value = max(value, read_pix_clamped(src, x + dx, y + dy, pitch, width, height));
        }
    }
    return value;
}

__kernel void kernel_rtgmc_retouch_copy(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height));
}

__kernel void kernel_rtgmc_retouch_repair1(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict ref, const int refPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        rtgmc_retouch_repair_mode1_value((const __global uchar *)src, (const __global uchar *)ref, ix, iy, srcPitch, refPitch, width, height));
}

__kernel void kernel_rtgmc_retouch_repair12(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict ref, const int refPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        rtgmc_retouch_repair_mode12_value((const __global uchar *)src, (const __global uchar *)ref, ix, iy, srcPitch, refPitch, width, height));
}

__kernel void kernel_rtgmc_retouch_removegrain12(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        rtgmc_retouch_removegrain12_value((const __global uchar *)src, ix, iy, srcPitch, width, height));
}

__kernel void kernel_rtgmc_retouch_removegrain11(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        rtgmc_retouch_removegrain12_value((const __global uchar *)src, ix, iy, srcPitch, width, height));
}

__kernel void kernel_rtgmc_retouch_detail_ref_vertical(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        rtgmc_retouch_detail_ref_vertical_value((const __global uchar *)src, ix, iy, srcPitch, width, height));
}

__kernel void kernel_rtgmc_retouch_precise_clamp(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict ref, const int refPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int refPix = read_pix_clamped((const __global uchar *)ref, ix, iy, refPitch, width, height);
    write_pix((__global uchar *)dst, ix, iy, dstPitch, rtgmc_retouch_precise_clamp_value(srcPix, refPix));
}

__kernel void kernel_rtgmc_retouch_detail_boost(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict blur, const int blurPitch,
    const int width, const int height,
    const float detailGain
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int blurPix = read_pix_clamped((const __global uchar *)blur, ix, iy, blurPitch, width, height);
    const float value = (float)srcPix + (float)(srcPix - blurPix) * detailGain;
    write_pix((__global uchar *)dst, ix, iy, dstPitch, rtgmc_retouch_round_clamp(value));
}

__kernel void kernel_rtgmc_retouch_detail_boost_fused(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height,
    const int smode,
    const int precise,
    const float detailGain
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int blurPix = (smode == 2)
        ? rtgmc_retouch_detail_ref_blur_value((const __global uchar *)src, ix, iy, srcPitch, width, height, precise)
        : rtgmc_retouch_removegrain12_value((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const float value = (float)srcPix + (float)(srcPix - blurPix) * detailGain;
    write_pix((__global uchar *)dst, ix, iy, dstPitch, rtgmc_retouch_round_clamp(value));
}

__kernel void kernel_rtgmc_retouch_detail_boost_edge_narrow_fused(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height,
    const int smode,
    const int precise,
    const float detailGain,
    const float edgeNarrowingGain
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int blurPix = (smode == 2)
        ? rtgmc_retouch_detail_ref_blur_value((const __global uchar *)src, ix, iy, srcPitch, width, height, precise)
        : rtgmc_retouch_removegrain12_value((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const float boosted = (float)srcPix + (float)(srcPix - blurPix) * detailGain;
    const int boostedPix = rtgmc_retouch_round_clamp(boosted);
    const int centerDiff = rtgmc_retouch_horizontal_balance_delta_value((const __global uchar *)src, ix, iy, srcPitch, width, height, edgeNarrowingGain);
    const int smoothDiff = rtgmc_retouch_area_balance_delta_value((const __global uchar *)src, ix, iy, srcPitch, width, height, edgeNarrowingGain);
    const int correction = rtgmc_retouch_stronger_non_neutral(smoothDiff, centerDiff);
    write_pix((__global uchar *)dst, ix, iy, dstPitch, rtgmc_retouch_add_diff_value(boostedPix, correction));
}

__kernel void kernel_rtgmc_retouch_edge_narrow_delta(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height,
    const float edgeNarrowingGain
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        rtgmc_retouch_vertical_balance_delta_value((const __global uchar *)src, ix, iy, srcPitch, width, height, edgeNarrowingGain));
}

__kernel void kernel_rtgmc_retouch_blur_h(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        rtgmc_retouch_blur10h_value((const __global uchar *)src, ix, iy, srcPitch, width, height));
}

__kernel void kernel_rtgmc_retouch_edge_narrow_guard_delta(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int rgPix = rtgmc_retouch_removegrain_smooth_value((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int value = rtgmc_retouch_stronger_non_neutral(rgPix, srcPix);
    write_pix((__global uchar *)dst, ix, iy, dstPitch, value);
}

__kernel void kernel_rtgmc_retouch_edge_narrow_guard_delta11(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int rgPix = rtgmc_retouch_removegrain_smooth_value((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int value = rtgmc_retouch_stronger_non_neutral(rgPix, srcPix);
    write_pix((__global uchar *)dst, ix, iy, dstPitch, value);
}

__kernel void kernel_rtgmc_retouch_adddiff(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict diff, const int diffPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int diffPix = read_pix_clamped((const __global uchar *)diff, ix, iy, diffPitch, width, height);
    write_pix((__global uchar *)dst, ix, iy, dstPitch, rtgmc_retouch_add_diff_value(srcPix, diffPix));
}

__kernel void kernel_rtgmc_retouch_edge_narrow_fused(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict base, const int basePitch,
    const int width, const int height,
    const float edgeNarrowingGain
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int centerDiff = rtgmc_retouch_horizontal_balance_delta_value((const __global uchar *)base, ix, iy, basePitch, width, height, edgeNarrowingGain);
    const int smoothDiff = rtgmc_retouch_area_balance_delta_value((const __global uchar *)base, ix, iy, basePitch, width, height, edgeNarrowingGain);
    const int correction = rtgmc_retouch_stronger_non_neutral(smoothDiff, centerDiff);
    write_pix((__global uchar *)dst, ix, iy, dstPitch, rtgmc_retouch_add_diff_value(srcPix, correction));
}

__kernel void kernel_rtgmc_retouch_make_delta(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict base, const int basePitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int basePix = read_pix_clamped((const __global uchar *)base, ix, iy, basePitch, width, height);
    write_pix((__global uchar *)dst, ix, iy, dstPitch, rtgmc_retouch_make_diff_value(srcPix, basePix));
}

__kernel void kernel_rtgmc_retouch_smooth_delta_fused(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict base, const int basePitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        rtgmc_retouch_removegrain12_diff_value((const __global uchar *)src, srcPitch,
            (const __global uchar *)base, basePitch, ix, iy, width, height));
}

__kernel void kernel_rtgmc_retouch_limit(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict base, const int basePitch,
    const __global Type *restrict ref, const int refPitch,
    const __global Type *restrict motionBack, const int motionBackPitch,
    const __global Type *restrict motionForw, const int motionForwPitch,
    const int width, const int height,
    const int slmode,
    const int slrad,
    const int sovs,
    const float limit_strength,
    const int use_temporal_limit
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    float value = (float)read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    if ((slmode == 2 || slmode == 4) && use_temporal_limit != 0) {
        value = (float)rtgmc_retouch_temporal_detail_guard_value(
            rtgmc_retouch_round_clamp(value),
            (const __global uchar *)ref,
            (const __global uchar *)motionBack,
            (const __global uchar *)motionForw,
            ix, iy,
            refPitch, motionBackPitch, motionForwPitch,
            width, height,
            sovs);
    } else if (slmode == 1 || slmode == 2 || slmode == 4 || limit_strength > 0.0f) {
        const int radius = clamp(slrad, 1, 3);
        const float localMin = (float)max(0, rtgmc_retouch_spatial_min((const __global uchar *)base, ix, iy, basePitch, width, height, radius) - sovs);
        const float localMax = (float)min(max_val, rtgmc_retouch_spatial_max((const __global uchar *)base, ix, iy, basePitch, width, height, radius) + sovs);
        const float limited = clamp(value, localMin, localMax);
        const float strength = (slmode == 1 || slmode == 2) ? 1.0f : clamp(limit_strength, 0.0f, 1.0f);
        value = value + (limited - value) * strength;
    }
    write_pix((__global uchar *)dst, ix, iy, dstPitch, rtgmc_retouch_round_clamp(value));
}
