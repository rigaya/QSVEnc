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

static inline int rtgmc_lossless_read_pix(
    const __global uchar *src, const int x, const int y,
    const int pitch
) {
    return (int)(*(const __global Type *)(src + y * pitch + x * sizeof(Type)));
}

static inline void rtgmc_lossless_write_pix(
    __global uchar *dst, const int x, const int y, const int pitch, const int value
) {
    __global Type *dstPix = (__global Type *)(dst + y * pitch + x * sizeof(Type));
    dstPix[0] = (Type)clamp(value, 0, max_val);
}

static inline int rtgmc_lossless_read_clamped(
    const __global uchar *src, int x, int y,
    const int pitch, const int width, const int height
) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    return rtgmc_lossless_read_pix(src, x, y, pitch);
}

static inline void rtgmc_lossless_sort2(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = lo;
    *b = hi;
}

static inline void rtgmc_lossless_sort2_desc(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = hi;
    *b = lo;
}

// Batcher's Bitonic Sort (1968), 8 elements / 24 comparisons / depth 6.
static inline void rtgmc_lossless_sort8(__private int *v) {
    rtgmc_lossless_sort2     (&v[0], &v[1]); rtgmc_lossless_sort2_desc(&v[2], &v[3]); rtgmc_lossless_sort2     (&v[4], &v[5]); rtgmc_lossless_sort2_desc(&v[6], &v[7]);
    rtgmc_lossless_sort2     (&v[0], &v[2]); rtgmc_lossless_sort2     (&v[1], &v[3]); rtgmc_lossless_sort2_desc(&v[4], &v[6]); rtgmc_lossless_sort2_desc(&v[5], &v[7]);
    rtgmc_lossless_sort2     (&v[0], &v[1]); rtgmc_lossless_sort2     (&v[2], &v[3]); rtgmc_lossless_sort2_desc(&v[4], &v[5]); rtgmc_lossless_sort2_desc(&v[6], &v[7]);
    rtgmc_lossless_sort2     (&v[0], &v[4]); rtgmc_lossless_sort2     (&v[1], &v[5]); rtgmc_lossless_sort2     (&v[2], &v[6]); rtgmc_lossless_sort2     (&v[3], &v[7]);
    rtgmc_lossless_sort2     (&v[0], &v[2]); rtgmc_lossless_sort2     (&v[1], &v[3]); rtgmc_lossless_sort2     (&v[4], &v[6]); rtgmc_lossless_sort2     (&v[5], &v[7]);
    rtgmc_lossless_sort2     (&v[0], &v[1]); rtgmc_lossless_sort2     (&v[2], &v[3]); rtgmc_lossless_sort2     (&v[4], &v[5]); rtgmc_lossless_sort2     (&v[6], &v[7]);
}

// 9-element sort: Bitonic Sort on v[0..7] then linear insert v[8] (24 + 8 = 32 comparisons).
static inline void rtgmc_lossless_sort9(__private int *v) {
    rtgmc_lossless_sort8(v);
    rtgmc_lossless_sort2(&v[7], &v[8]);
    rtgmc_lossless_sort2(&v[6], &v[7]);
    rtgmc_lossless_sort2(&v[5], &v[6]);
    rtgmc_lossless_sort2(&v[4], &v[5]);
    rtgmc_lossless_sort2(&v[3], &v[4]);
    rtgmc_lossless_sort2(&v[2], &v[3]);
    rtgmc_lossless_sort2(&v[1], &v[2]);
    rtgmc_lossless_sort2(&v[0], &v[1]);
}

static inline int rtgmc_lossless_make_diff(const int a, const int b) {
    return clamp(a - b + range_half, 0, max_val);
}

static inline int rtgmc_lossless_median3(const int a, const int b, const int c) {
    const int lo = min(a, b);
    const int hi = max(a, b);
    return max(lo, min(hi, c));
}

static inline int rtgmc_field_restore_select_offset(const int offsetA, const int offsetB) {
    if (offsetA == 0 || offsetB == 0 || ((offsetA < 0) != (offsetB < 0))) {
        return 0;
    }
    return (abs(offsetA) <= abs(offsetB)) ? offsetA : offsetB;
}

static inline int rtgmc_field_restore_pick_consistent_delta(const int candA, const int candB) {
    const int neutral = range_half;
    const int selectedOffset = rtgmc_field_restore_select_offset(candA - neutral, candB - neutral);
    return neutral + selectedOffset;
}

static inline int rtgmc_field_restore_reference_sample(
    const __global uchar *processed, const int processedPitch,
    const __global uchar *source, const int sourcePitch,
    int x, int y, const int width, const int height,
    const int sourceField
) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    if ((y & 1) == sourceField) {
        return rtgmc_lossless_read_pix(source, x, y, sourcePitch);
    }
    return rtgmc_lossless_read_pix(processed, x, y, processedPitch);
}

static inline int rtgmc_field_restore_direct_vertical_delta(
    const __global uchar *processed, const int processedPitch,
    const __global uchar *source, const int sourcePitch,
    const int x, const int y, const int width, const int height,
    const int sourceField
) {
    const int b = rtgmc_field_restore_reference_sample(processed, processedPitch, source, sourcePitch, x, y, width, height, sourceField);
    if (y <= 0 || y >= height - 1) {
        return range_half;
    }
    const int a = rtgmc_field_restore_reference_sample(processed, processedPitch, source, sourcePitch, x, y - 1, width, height, sourceField);
    const int c = rtgmc_field_restore_reference_sample(processed, processedPitch, source, sourcePitch, x, y + 1, width, height, sourceField);
    const int verticalMedian = rtgmc_lossless_median3(a, b, c);
    return rtgmc_lossless_make_diff(b, verticalMedian);
}

static inline int rtgmc_field_restore_reference_vertical_median(
    const __global uchar *reference, const int referencePitch,
    const int x, const int y, const int width, const int height
) {
    const int b = rtgmc_lossless_read_clamped(reference, x, y, referencePitch, width, height);
    if (y <= 0 || y >= height - 1) {
        return b;
    }
    const int a = rtgmc_lossless_read_clamped(reference, x, y - 1, referencePitch, width, height);
    const int c = rtgmc_lossless_read_clamped(reference, x, y + 1, referencePitch, width, height);
    return rtgmc_lossless_median3(a, b, c);
}

static inline int rtgmc_field_restore_vertical_delta(
    const __global uchar *reference, const int referencePitch,
    const int x, const int y, const int width, const int height
) {
    const int p = rtgmc_lossless_read_clamped(reference, x, y, referencePitch, width, height);
    const int vm = rtgmc_field_restore_reference_vertical_median(reference, referencePitch, x, y, width, height);
    return rtgmc_lossless_make_diff(p, vm);
}

static inline int rtgmc_field_restore_stabilized_delta(
    const __global uchar *delta, const int deltaPitch,
    const int x, const int y, const int width, const int height
) {
    const int b = rtgmc_lossless_read_clamped(delta, x, y, deltaPitch, width, height);
    int cleaned = b;
    if (y - 2 >= 0 && y + 2 < height) {
        const int a = rtgmc_lossless_read_clamped(delta, x, y - 2, deltaPitch, width, height);
        const int c = rtgmc_lossless_read_clamped(delta, x, y + 2, deltaPitch, width, height);
        cleaned = rtgmc_lossless_median3(a, b, c);
    }
    return rtgmc_field_restore_pick_consistent_delta(cleaned, b);
}

static inline int rtgmc_field_restore_rank_smooth_delta(
    const __global uchar *delta, const int deltaPitch,
    const int x, const int y, const int width, const int height
) {
    const int s = rtgmc_field_restore_stabilized_delta(delta, deltaPitch, x, y, width, height);
    if (x <= 0 || x >= width - 1 || y - 2 < 0 || y + 2 >= height) {
        return s;
    }
    int v[8] = {
        rtgmc_field_restore_stabilized_delta(delta, deltaPitch, x - 1, y - 2, width, height),
        rtgmc_field_restore_stabilized_delta(delta, deltaPitch, x + 0, y - 2, width, height),
        rtgmc_field_restore_stabilized_delta(delta, deltaPitch, x + 1, y - 2, width, height),
        rtgmc_field_restore_stabilized_delta(delta, deltaPitch, x - 1, y + 0, width, height),
        rtgmc_field_restore_stabilized_delta(delta, deltaPitch, x + 1, y + 0, width, height),
        rtgmc_field_restore_stabilized_delta(delta, deltaPitch, x - 1, y + 2, width, height),
        rtgmc_field_restore_stabilized_delta(delta, deltaPitch, x + 0, y + 2, width, height),
        rtgmc_field_restore_stabilized_delta(delta, deltaPitch, x + 1, y + 2, width, height)
    };
    rtgmc_lossless_sort8(v);
    return clamp(s, v[1], v[6]);
}

static inline int rtgmc_field_restore_bounded_delta(
    const __global uchar *delta, const int deltaPitch,
    const int x, const int y, const int width, const int height
) {
    const int s = rtgmc_field_restore_stabilized_delta(delta, deltaPitch, x, y, width, height);
    if (x <= 0 || x >= width - 1 || y - 2 < 0 || y + 2 >= height) {
        return s;
    }
    int v[9] = {
        rtgmc_field_restore_rank_smooth_delta(delta, deltaPitch, x - 1, y - 2, width, height),
        rtgmc_field_restore_rank_smooth_delta(delta, deltaPitch, x + 0, y - 2, width, height),
        rtgmc_field_restore_rank_smooth_delta(delta, deltaPitch, x + 1, y - 2, width, height),
        rtgmc_field_restore_rank_smooth_delta(delta, deltaPitch, x - 1, y + 0, width, height),
        s,
        rtgmc_field_restore_rank_smooth_delta(delta, deltaPitch, x + 1, y + 0, width, height),
        rtgmc_field_restore_rank_smooth_delta(delta, deltaPitch, x - 1, y + 2, width, height),
        rtgmc_field_restore_rank_smooth_delta(delta, deltaPitch, x + 0, y + 2, width, height),
        rtgmc_field_restore_rank_smooth_delta(delta, deltaPitch, x + 1, y + 2, width, height)
    };
    rtgmc_lossless_sort9(v);
    return clamp(s, v[0], v[8]);
}

static inline int rtgmc_field_restore_direct_stabilized_delta(
    const __global uchar *processed, const int processedPitch,
    const __global uchar *source, const int sourcePitch,
    const int x, const int y, const int width, const int height,
    const int sourceField
) {
    const int b = rtgmc_field_restore_direct_vertical_delta(processed, processedPitch, source, sourcePitch, x, y, width, height, sourceField);
    int cleaned = b;
    if (y - 2 >= 0 && y + 2 < height) {
        const int a = rtgmc_field_restore_direct_vertical_delta(processed, processedPitch, source, sourcePitch, x, y - 2, width, height, sourceField);
        const int c = rtgmc_field_restore_direct_vertical_delta(processed, processedPitch, source, sourcePitch, x, y + 2, width, height, sourceField);
        cleaned = rtgmc_lossless_median3(a, b, c);
    }
    return rtgmc_field_restore_pick_consistent_delta(cleaned, b);
}

static inline int rtgmc_field_restore_direct_rank_smooth_delta(
    const __global uchar *processed, const int processedPitch,
    const __global uchar *source, const int sourcePitch,
    const int x, const int y, const int width, const int height,
    const int sourceField
) {
    const int s = rtgmc_field_restore_direct_stabilized_delta(processed, processedPitch, source, sourcePitch, x, y, width, height, sourceField);
    if (x <= 0 || x >= width - 1 || y - 2 < 0 || y + 2 >= height) {
        return s;
    }
    int v[8] = {
        rtgmc_field_restore_direct_stabilized_delta(processed, processedPitch, source, sourcePitch, x - 1, y - 2, width, height, sourceField),
        rtgmc_field_restore_direct_stabilized_delta(processed, processedPitch, source, sourcePitch, x + 0, y - 2, width, height, sourceField),
        rtgmc_field_restore_direct_stabilized_delta(processed, processedPitch, source, sourcePitch, x + 1, y - 2, width, height, sourceField),
        rtgmc_field_restore_direct_stabilized_delta(processed, processedPitch, source, sourcePitch, x - 1, y + 0, width, height, sourceField),
        rtgmc_field_restore_direct_stabilized_delta(processed, processedPitch, source, sourcePitch, x + 1, y + 0, width, height, sourceField),
        rtgmc_field_restore_direct_stabilized_delta(processed, processedPitch, source, sourcePitch, x - 1, y + 2, width, height, sourceField),
        rtgmc_field_restore_direct_stabilized_delta(processed, processedPitch, source, sourcePitch, x + 0, y + 2, width, height, sourceField),
        rtgmc_field_restore_direct_stabilized_delta(processed, processedPitch, source, sourcePitch, x + 1, y + 2, width, height, sourceField)
    };
    rtgmc_lossless_sort8(v);
    return clamp(s, v[1], v[6]);
}

static inline int rtgmc_field_restore_direct_bounded_delta(
    const __global uchar *processed, const int processedPitch,
    const __global uchar *source, const int sourcePitch,
    const int x, const int y, const int width, const int height,
    const int sourceField
) {
    const int s = rtgmc_field_restore_direct_stabilized_delta(processed, processedPitch, source, sourcePitch, x, y, width, height, sourceField);
    if (x <= 0 || x >= width - 1 || y - 2 < 0 || y + 2 >= height) {
        return s;
    }
    int v[9] = {
        rtgmc_field_restore_direct_rank_smooth_delta(processed, processedPitch, source, sourcePitch, x - 1, y - 2, width, height, sourceField),
        rtgmc_field_restore_direct_rank_smooth_delta(processed, processedPitch, source, sourcePitch, x + 0, y - 2, width, height, sourceField),
        rtgmc_field_restore_direct_rank_smooth_delta(processed, processedPitch, source, sourcePitch, x + 1, y - 2, width, height, sourceField),
        rtgmc_field_restore_direct_rank_smooth_delta(processed, processedPitch, source, sourcePitch, x - 1, y + 0, width, height, sourceField),
        s,
        rtgmc_field_restore_direct_rank_smooth_delta(processed, processedPitch, source, sourcePitch, x + 1, y + 0, width, height, sourceField),
        rtgmc_field_restore_direct_rank_smooth_delta(processed, processedPitch, source, sourcePitch, x - 1, y + 2, width, height, sourceField),
        rtgmc_field_restore_direct_rank_smooth_delta(processed, processedPitch, source, sourcePitch, x + 0, y + 2, width, height, sourceField),
        rtgmc_field_restore_direct_rank_smooth_delta(processed, processedPitch, source, sourcePitch, x + 1, y + 2, width, height, sourceField)
    };
    rtgmc_lossless_sort9(v);
    return clamp(s, v[0], v[8]);
}

__attribute__((reqd_work_group_size(rtgmc_lossless_block_x, rtgmc_lossless_block_y, 1)))
__kernel void kernel_rtgmc_lossless_build_reference_frame(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pProcessed, const int processedPitch,
    const __global Type *restrict pSource, const int sourcePitch,
    const int width,
    const int height,
    const int sourceField
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const __global uchar *processed = (const __global uchar *)pProcessed;
    const __global uchar *source = (const __global uchar *)pSource;
    const int value = ((iy & 1) == sourceField)
        ? rtgmc_lossless_read_pix(source, ix, iy, sourcePitch)
        : rtgmc_lossless_read_pix(processed, ix, iy, processedPitch);
    rtgmc_lossless_write_pix((__global uchar *)pDst, ix, iy, dstPitch, value);
}

__attribute__((reqd_work_group_size(rtgmc_lossless_block_x, rtgmc_lossless_block_y, 1)))
__kernel void kernel_rtgmc_lossless_build_delta_map(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pReference, const int referencePitch,
    const int width,
    const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const int value = rtgmc_field_restore_vertical_delta(
        (const __global uchar *)pReference, referencePitch, ix, iy, width, height);
    rtgmc_lossless_write_pix((__global uchar *)pDst, ix, iy, dstPitch, value);
}

__attribute__((reqd_work_group_size(rtgmc_lossless_block_x, rtgmc_lossless_block_y, 1)))
__kernel void kernel_rtgmc_lossless_stabilize_delta_map(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pDelta, const int deltaPitch,
    const int width,
    const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const int value = rtgmc_field_restore_bounded_delta(
        (const __global uchar *)pDelta, deltaPitch, ix, iy, width, height);
    rtgmc_lossless_write_pix((__global uchar *)pDst, ix, iy, dstPitch, value);
}

__attribute__((reqd_work_group_size(rtgmc_lossless_block_x, rtgmc_lossless_block_y, 1)))
__kernel void kernel_rtgmc_lossless_apply_delta(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pProcessed, const int processedPitch,
    const __global Type *restrict pSource, const int sourcePitch,
    const __global Type *restrict pDelta, const int deltaPitch,
    const int width,
    const int height,
    const int sourceField
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    int value = 0;
    if ((iy & 1) == sourceField) {
        value = rtgmc_lossless_read_pix((const __global uchar *)pSource, ix, iy, sourcePitch);
    } else {
        const int newField = rtgmc_lossless_read_pix((const __global uchar *)pProcessed, ix, iy, processedPitch);
        const int delta = rtgmc_lossless_read_pix((const __global uchar *)pDelta, ix, iy, deltaPitch);
        value = rtgmc_lossless_make_diff(newField, delta);
    }
    rtgmc_lossless_write_pix((__global uchar *)pDst, ix, iy, dstPitch, value);
}

__attribute__((reqd_work_group_size(rtgmc_lossless_block_x, rtgmc_lossless_block_y, 1)))
__kernel void kernel_rtgmc_lossless_apply_direct_delta(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pProcessed, const int processedPitch,
    const __global Type *restrict pSource, const int sourcePitch,
    const int width,
    const int height,
    const int sourceField
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const __global uchar *processed = (const __global uchar *)pProcessed;
    const __global uchar *source = (const __global uchar *)pSource;
    int value = 0;
    if ((iy & 1) == sourceField) {
        value = rtgmc_lossless_read_pix(source, ix, iy, sourcePitch);
    } else {
        const int newField = rtgmc_lossless_read_pix(processed, ix, iy, processedPitch);
        const int delta = rtgmc_field_restore_direct_bounded_delta(processed, processedPitch, source, sourcePitch, ix, iy, width, height, sourceField);
        value = rtgmc_lossless_make_diff(newField, delta);
    }
    rtgmc_lossless_write_pix((__global uchar *)pDst, ix, iy, dstPitch, value);
}
