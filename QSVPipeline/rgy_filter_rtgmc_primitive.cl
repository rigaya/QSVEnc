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

static inline int rtgmc_read_pix(
    const __global uchar *src, int x, int y,
    const int pitch, const int width, const int height
) {
    x = clamp(x, 0, width  - 1);
    y = clamp(y, 0, height - 1);
    return (int)(*(const __global Type *)(src + y * pitch + x * sizeof(Type)));
}
static inline void rtgmc_write_pix(
    __global uchar *dst, int x, int y, const int pitch, const int value
) {
    __global Type *dstPix = (__global Type *)(dst + y * pitch + x * sizeof(Type));
    dstPix[0] = (Type)clamp(value, 0, max_val);
}

static inline int rtgmc_make_diff(const int a, const int b) {
    return clamp(a - b + range_half, 0, max_val);
}

static inline int rtgmc_add_diff(const int src, const int diff) {
    return clamp(src + diff - range_half, 0, max_val);
}

static inline int rtgmc_add_weighted_diff(const int src, const int diff, const float weight) {
    const float value = fma((float)(diff - range_half), weight, (float)src);
    return clamp(convert_int_rte(value), 0, max_val);
}

static inline int rtgmc_merge_weighted(const int src0, const int src1, const float weight) {
    const float value = mix((float)src0, (float)src1, weight);
    return clamp(convert_int_rte(value), 0, max_val);
}

static inline void rtgmc_sort2(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = lo;
    *b = hi;
}

static inline void rtgmc_sort2_desc(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = hi;
    *b = lo;
}

// Batcher's Bitonic Sort (1968), 8 elements / 24 comparisons / depth 6.
static inline void rtgmc_sort8(__private int *v) {
    rtgmc_sort2     (&v[0], &v[1]); rtgmc_sort2_desc(&v[2], &v[3]); rtgmc_sort2     (&v[4], &v[5]); rtgmc_sort2_desc(&v[6], &v[7]);
    rtgmc_sort2     (&v[0], &v[2]); rtgmc_sort2     (&v[1], &v[3]); rtgmc_sort2_desc(&v[4], &v[6]); rtgmc_sort2_desc(&v[5], &v[7]);
    rtgmc_sort2     (&v[0], &v[1]); rtgmc_sort2     (&v[2], &v[3]); rtgmc_sort2_desc(&v[4], &v[5]); rtgmc_sort2_desc(&v[6], &v[7]);
    rtgmc_sort2     (&v[0], &v[4]); rtgmc_sort2     (&v[1], &v[5]); rtgmc_sort2     (&v[2], &v[6]); rtgmc_sort2     (&v[3], &v[7]);
    rtgmc_sort2     (&v[0], &v[2]); rtgmc_sort2     (&v[1], &v[3]); rtgmc_sort2     (&v[4], &v[6]); rtgmc_sort2     (&v[5], &v[7]);
    rtgmc_sort2     (&v[0], &v[1]); rtgmc_sort2     (&v[2], &v[3]); rtgmc_sort2     (&v[4], &v[5]); rtgmc_sort2     (&v[6], &v[7]);
}

// 9-element sort: Bitonic Sort on v[0..7] then linear insert v[8] (24 + 8 = 32 comparisons).
static inline void rtgmc_sort9(__private int *v) {
    rtgmc_sort8(v);
    rtgmc_sort2(&v[7], &v[8]);
    rtgmc_sort2(&v[6], &v[7]);
    rtgmc_sort2(&v[5], &v[6]);
    rtgmc_sort2(&v[4], &v[5]);
    rtgmc_sort2(&v[3], &v[4]);
    rtgmc_sort2(&v[2], &v[3]);
    rtgmc_sort2(&v[1], &v[2]);
    rtgmc_sort2(&v[0], &v[1]);
}

static inline int rtgmc_on_inner_pixel(const int x, const int y, const int width, const int height) {
    return x > 0 && x < width - 1 && y > 0 && y < height - 1;
}

static inline void rtgmc_gather_ring3(
    __private int *dst,
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    int count = 0;
#pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
#pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            if (dx != 0 || dy != 0) {
                dst[count++] = rtgmc_read_pix(src, x + dx, y + dy, pitch, width, height);
            }
        }
    }
}

static inline void rtgmc_gather_ref_window_with_center(
    __private int *dst,
    const __global uchar *ref, const int centerValue,
    const int x, const int y,
    const int refPitch, const int width, const int height
) {
    int count = 0;
#pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
#pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            dst[count++] = (dx == 0 && dy == 0)
                ? centerValue
                : rtgmc_read_pix(ref, x + dx, y + dy, refPitch, width, height);
        }
    }
}

static inline int rtgmc_box3_sum(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int weighted
) {
    int sum = 0;
#pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
        const int wy = weighted ? (2 - abs(dy)) : 1;
#pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            const int wx = weighted ? (2 - abs(dx)) : 1;
            sum += wx * wy * rtgmc_read_pix(src, x + dx, y + dy, pitch, width, height);
        }
    }
    return sum;
}

static inline int rtgmc_rank_clipped_center(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int mode
) {
    const int s = rtgmc_read_pix(src, x, y, pitch, width, height);
    if (!rtgmc_on_inner_pixel(x, y, width, height)) {
        return s;
    }
    int v[8];
    rtgmc_gather_ring3(v, src, x, y, pitch, width, height);
    rtgmc_sort8(v);
    const int n = clamp(mode, 1, 4);
    return clamp(s, v[n - 1], v[8 - n]);
}

static inline int rtgmc_weighted_box3_center(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    const int s = rtgmc_read_pix(src, x, y, pitch, width, height);
    if (!rtgmc_on_inner_pixel(x, y, width, height)) {
        return s;
    }
    const int sum = rtgmc_box3_sum(src, x, y, pitch, width, height, 1);
    return (sum + 8) >> 4;
}

static inline int rtgmc_average_box3_center(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    const int s = rtgmc_read_pix(src, x, y, pitch, width, height);
    if (!rtgmc_on_inner_pixel(x, y, width, height)) {
        return s;
    }
    const int sum = rtgmc_box3_sum(src, x, y, pitch, width, height, 0);
    return (sum + 4) / 9;
}

static inline int rtgmc_removegrain(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int mode
) {
    if (mode == 11 || mode == 12) {
        return rtgmc_weighted_box3_center(src, x, y, pitch, width, height);
    }
    if (mode == 20) {
        return rtgmc_average_box3_center(src, x, y, pitch, width, height);
    }
    return rtgmc_rank_clipped_center(src, x, y, pitch, width, height, mode);
}

static inline int rtgmc_ref_rank_clipped_center(
    const __global uchar *src, const __global uchar *ref, const int x, const int y,
    const int srcPitch, const int refPitch, const int width, const int height,
    const int mode
) {
    const int s = rtgmc_read_pix(src, x, y, srcPitch, width, height);
    if (!rtgmc_on_inner_pixel(x, y, width, height)) {
        return s;
    }
    int v[9];
    rtgmc_gather_ref_window_with_center(v, ref, s, x, y, refPitch, width, height);
    rtgmc_sort9(v);
    const int n = clamp(mode, 1, 4);
    return clamp(s, v[n - 1], v[9 - n]);
}

static inline int rtgmc_ref_inner_range_clipped_center(
    const __global uchar *src, const __global uchar *ref, const int x, const int y,
    const int srcPitch, const int refPitch, const int width, const int height
) {
    const int s = rtgmc_read_pix(src, x, y, srcPitch, width, height);
    if (!rtgmc_on_inner_pixel(x, y, width, height)) {
        return s;
    }
    int v[8];
    rtgmc_gather_ring3(v, ref, x, y, refPitch, width, height);
    rtgmc_sort8(v);
    const int c = rtgmc_read_pix(ref, x, y, refPitch, width, height);
    const int lo = min(v[1], c);
    const int hi = max(v[6], c);
    return clamp(s, lo, hi);
}

static inline int rtgmc_repair(
    const __global uchar *src, const __global uchar *ref, const int x, const int y,
    const int srcPitch, const int refPitch, const int width, const int height,
    const int mode
) {
    if (mode == 12) {
        return rtgmc_ref_inner_range_clipped_center(src, ref, x, y, srcPitch, refPitch, width, height);
    }
    return rtgmc_ref_rank_clipped_center(src, ref, x, y, srcPitch, refPitch, width, height, mode);
}

static inline int rtgmc_vertical_window5_extreme(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int takeMax
) {
    const int center = rtgmc_read_pix(src, x, y, pitch, width, height);
    int value = center;
    for (int offset = -2; offset <= 2; offset++) {
        const int yy = y + offset;
        const int sample = (yy >= 0 && yy < height) ? rtgmc_read_pix(src, x, yy, pitch, width, height) : center;
        value = takeMax ? max(value, sample) : min(value, sample);
    }
    return value;
}

__attribute__((reqd_work_group_size(rtgmc_primitive_block_x, rtgmc_primitive_block_y, 1)))
__kernel void kernel_rtgmc_primitive_copy(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pSrc, const int srcPitch,
    const int width,
    const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int value = rtgmc_read_pix((const __global uchar *)pSrc, ix, iy, srcPitch, width, height);
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch, value);
}

__attribute__((reqd_work_group_size(rtgmc_primitive_block_x, rtgmc_primitive_block_y, 1)))
__kernel void kernel_rtgmc_primitive_makediff(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pSrc, const int srcPitch,
    const __global Type *restrict pRef, const int refPitch,
    const int width,
    const int height,
    const int mode
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int src = rtgmc_read_pix((const __global uchar *)pSrc, ix, iy, srcPitch, width, height);
    const int ref = rtgmc_read_pix((const __global uchar *)pRef, ix, iy, refPitch, width, height);
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch, rtgmc_make_diff(src, ref));
}

static inline int rtgmc_make_diff_from_frames(
    const __global uchar *src, const __global uchar *ref,
    const int x, const int y,
    const int srcPitch, const int refPitch,
    const int width, const int height
) {
    const int srcValue = rtgmc_read_pix(src, x, y, srcPitch, width, height);
    const int refValue = rtgmc_read_pix(ref, x, y, refPitch, width, height);
    return rtgmc_make_diff(srcValue, refValue);
}

__attribute__((reqd_work_group_size(rtgmc_primitive_block_x, rtgmc_primitive_block_y, 1)))
__kernel void kernel_rtgmc_primitive_makediff_removegrain20(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pSrc, const int srcPitch,
    const __global Type *restrict pRef, const int refPitch,
    const int width,
    const int height,
    const int mode
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const __global uchar *src = (const __global uchar *)pSrc;
    const __global uchar *ref = (const __global uchar *)pRef;
    int value = rtgmc_make_diff_from_frames(src, ref, ix, iy, srcPitch, refPitch, width, height);
    if (ix > 0 && ix < width - 1 && iy > 0 && iy < height - 1) {
        int sum = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                sum += rtgmc_make_diff_from_frames(src, ref, ix + dx, iy + dy, srcPitch, refPitch, width, height);
            }
        }
        value = (sum + 4) / 9;
    }
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch, value);
}

__attribute__((reqd_work_group_size(rtgmc_primitive_block_x, rtgmc_primitive_block_y, 1)))
__kernel void kernel_rtgmc_primitive_makediff_removegrain20_adddiff(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pSrc, const int srcPitch,
    const __global Type *restrict pRef, const int refPitch,
    const int width,
    const int height,
    const int mode
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const __global uchar *src = (const __global uchar *)pSrc;
    const __global uchar *ref = (const __global uchar *)pRef;
    int diff = rtgmc_make_diff_from_frames(src, ref, ix, iy, srcPitch, refPitch, width, height);
    if (ix > 0 && ix < width - 1 && iy > 0 && iy < height - 1) {
        int sum = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                sum += rtgmc_make_diff_from_frames(src, ref, ix + dx, iy + dy, srcPitch, refPitch, width, height);
            }
        }
        diff = (sum + 4) / 9;
    }
    const int base = rtgmc_read_pix(ref, ix, iy, refPitch, width, height);
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch, rtgmc_add_diff(base, diff));
}

__attribute__((reqd_work_group_size(rtgmc_primitive_block_x, rtgmc_primitive_block_y, 1)))
__kernel void kernel_rtgmc_primitive_adddiff(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pSrc, const int srcPitch,
    const __global Type *restrict pRef, const int refPitch,
    const int width,
    const int height,
    const int mode
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int src = rtgmc_read_pix((const __global uchar *)pSrc, ix, iy, srcPitch, width, height);
    const int diff = rtgmc_read_pix((const __global uchar *)pRef, ix, iy, refPitch, width, height);
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch, rtgmc_add_diff(src, diff));
}

__attribute__((reqd_work_group_size(rtgmc_primitive_block_x, rtgmc_primitive_block_y, 1)))
__kernel void kernel_rtgmc_primitive_addweighteddiff(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pSrc, const int srcPitch,
    const __global Type *restrict pRef, const int refPitch,
    const int width,
    const int height,
    const float weight
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int src = rtgmc_read_pix((const __global uchar *)pSrc, ix, iy, srcPitch, width, height);
    const int diff = rtgmc_read_pix((const __global uchar *)pRef, ix, iy, refPitch, width, height);
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch, rtgmc_add_weighted_diff(src, diff, weight));
}

__attribute__((reqd_work_group_size(rtgmc_primitive_block_x, rtgmc_primitive_block_y, 1)))
__kernel void kernel_rtgmc_primitive_removegrain(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pSrc, const int srcPitch,
    const int width,
    const int height,
    const int mode
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int value = rtgmc_removegrain((const __global uchar *)pSrc, ix, iy, srcPitch, width, height, mode);
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch, value);
}

__attribute__((reqd_work_group_size(rtgmc_primitive_block_x, rtgmc_primitive_block_y, 1)))
__kernel void kernel_rtgmc_primitive_repair(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pSrc, const int srcPitch,
    const __global Type *restrict pRef, const int refPitch,
    const int width,
    const int height,
    const int mode
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int value = rtgmc_repair((const __global uchar *)pSrc, (const __global uchar *)pRef, ix, iy, srcPitch, refPitch, width, height, mode);
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch, value);
}

__attribute__((reqd_work_group_size(rtgmc_primitive_block_x, rtgmc_primitive_block_y, 1)))
__kernel void kernel_rtgmc_primitive_merge(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pSrc, const int srcPitch,
    const __global Type *restrict pRef, const int refPitch,
    const int width,
    const int height,
    const float weight
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int src = rtgmc_read_pix((const __global uchar *)pSrc, ix, iy, srcPitch, width, height);
    const int ref = rtgmc_read_pix((const __global uchar *)pRef, ix, iy, refPitch, width, height);
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch, rtgmc_merge_weighted(src, ref, weight));
}

__attribute__((reqd_work_group_size(rtgmc_primitive_block_x, rtgmc_primitive_block_y, 1)))
__kernel void kernel_rtgmc_primitive_vertical_min5(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pSrc, const int srcPitch,
    const int width,
    const int height,
    const int mode
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int value = rtgmc_vertical_window5_extreme((const __global uchar *)pSrc, ix, iy, srcPitch, width, height, 0);
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch, value);
}

__attribute__((reqd_work_group_size(rtgmc_primitive_block_x, rtgmc_primitive_block_y, 1)))
__kernel void kernel_rtgmc_primitive_vertical_max5(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pSrc, const int srcPitch,
    const int width,
    const int height,
    const int mode
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int value = rtgmc_vertical_window5_extreme((const __global uchar *)pSrc, ix, iy, srcPitch, width, height, 1);
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch, value);
}

__attribute__((reqd_work_group_size(rtgmc_primitive_block_x, rtgmc_primitive_block_y, 1)))
__kernel void kernel_rtgmc_primitive_logicmin(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pSrc, const int srcPitch,
    const __global Type *restrict pRef, const int refPitch,
    const int width,
    const int height,
    const int mode
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int src = rtgmc_read_pix((const __global uchar *)pSrc, ix, iy, srcPitch, width, height);
    const int ref = rtgmc_read_pix((const __global uchar *)pRef, ix, iy, refPitch, width, height);
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch, min(src, ref));
}

__attribute__((reqd_work_group_size(rtgmc_primitive_block_x, rtgmc_primitive_block_y, 1)))
__kernel void kernel_rtgmc_primitive_logicmax(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pSrc, const int srcPitch,
    const __global Type *restrict pRef, const int refPitch,
    const int width,
    const int height,
    const int mode
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int src = rtgmc_read_pix((const __global uchar *)pSrc, ix, iy, srcPitch, width, height);
    const int ref = rtgmc_read_pix((const __global uchar *)pRef, ix, iy, refPitch, width, height);
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch, max(src, ref));
}
