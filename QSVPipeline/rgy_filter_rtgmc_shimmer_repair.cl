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

static inline int rtgmc_signed_to_diff(const int signedValue) {
    return clamp(signedValue + range_half, 0, max_val);
}

static inline int rtgmc_repair_delta_centered(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    return clamp(
        rtgmc_read_pix(reference, x, y, referencePitch, width, height)
            - rtgmc_read_pix(input, x, y, inputPitch, width, height)
            + range_half,
        0,
        max_val);
}

static inline int rtgmc_repair_vertical_window(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int useMax
) {
    int value = rtgmc_repair_delta_centered(input, inputPitch, reference, referencePitch, x, y, width, height);
    for (int dy = -2; dy <= 2; dy++) {
        const int sample = rtgmc_repair_delta_centered(input, inputPitch, reference, referencePitch, x, y + dy, width, height);
        value = useMax ? max(value, sample) : min(value, sample);
    }
    return value;
}

static inline int rtgmc_repair_pos_vertical_contract(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    int value = rtgmc_repair_vertical_window(input, inputPitch, reference, referencePitch, x, y, width, height, 0);
    if (RTGMC_SHIMMER_REPAIR_THIN_LEVEL > 5) {
        for (int dy = -1; dy <= 1; dy++) {
            value = min(value, rtgmc_repair_vertical_window(input, inputPitch, reference, referencePitch, x, y + dy, width, height, 0));
        }
    }
    return value;
}

static inline int rtgmc_repair_neg_vertical_expand(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    int value = rtgmc_repair_vertical_window(input, inputPitch, reference, referencePitch, x, y, width, height, 1);
    if (RTGMC_SHIMMER_REPAIR_THIN_LEVEL > 5) {
        for (int dy = -1; dy <= 1; dy++) {
            value = max(value, rtgmc_repair_vertical_window(input, inputPitch, reference, referencePitch, x, y + dy, width, height, 1));
        }
    }
    return value;
}

static inline int rtgmc_repair_pos_local_contract(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    const int center = rtgmc_repair_pos_vertical_contract(input, inputPitch, reference, referencePitch, x, y, width, height);
    if ((RTGMC_SHIMMER_REPAIR_THIN_LEVEL % 3) == 0) {
        return center;
    }
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return center;
    }
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += rtgmc_repair_pos_vertical_contract(input, inputPitch, reference, referencePitch, x + dx, y + dy, width, height);
        }
    }
    return min(center, (sum + 4) / 9);
}

static inline int rtgmc_repair_neg_local_expand(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    const int center = rtgmc_repair_neg_vertical_expand(input, inputPitch, reference, referencePitch, x, y, width, height);
    if ((RTGMC_SHIMMER_REPAIR_THIN_LEVEL % 3) == 0) {
        return center;
    }
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return center;
    }
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += rtgmc_repair_neg_vertical_expand(input, inputPitch, reference, referencePitch, x + dx, y + dy, width, height);
        }
    }
    return max(center, (sum + 4) / 9);
}

static inline void rtgmc_repair_sort2(__private int *a, __private int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = lo;
    *b = hi;
}

static inline void rtgmc_repair_sort2_desc(__private int *a, __private int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = hi;
    *b = lo;
}

static inline void rtgmc_repair_sort8(__private int *v) {
    rtgmc_repair_sort2     (&v[0], &v[1]); rtgmc_repair_sort2_desc(&v[2], &v[3]); rtgmc_repair_sort2     (&v[4], &v[5]); rtgmc_repair_sort2_desc(&v[6], &v[7]);
    rtgmc_repair_sort2     (&v[0], &v[2]); rtgmc_repair_sort2     (&v[1], &v[3]); rtgmc_repair_sort2_desc(&v[4], &v[6]); rtgmc_repair_sort2_desc(&v[5], &v[7]);
    rtgmc_repair_sort2     (&v[0], &v[1]); rtgmc_repair_sort2     (&v[2], &v[3]); rtgmc_repair_sort2_desc(&v[4], &v[5]); rtgmc_repair_sort2_desc(&v[6], &v[7]);
    rtgmc_repair_sort2     (&v[0], &v[4]); rtgmc_repair_sort2     (&v[1], &v[5]); rtgmc_repair_sort2     (&v[2], &v[6]); rtgmc_repair_sort2     (&v[3], &v[7]);
    rtgmc_repair_sort2     (&v[0], &v[2]); rtgmc_repair_sort2     (&v[1], &v[3]); rtgmc_repair_sort2     (&v[4], &v[6]); rtgmc_repair_sort2     (&v[5], &v[7]);
    rtgmc_repair_sort2     (&v[0], &v[1]); rtgmc_repair_sort2     (&v[2], &v[3]); rtgmc_repair_sort2     (&v[4], &v[5]); rtgmc_repair_sort2     (&v[6], &v[7]);
}

static inline int rtgmc_repair_pos_rank_limit(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    const int center = rtgmc_repair_pos_local_contract(input, inputPitch, reference, referencePitch, x, y, width, height);
    if (RTGMC_SHIMMER_REPAIR_THIN_LEVEL != 2 && RTGMC_SHIMMER_REPAIR_THIN_LEVEL != 5) {
        return center;
    }
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return center;
    }
    int v[8] = {
        rtgmc_repair_pos_local_contract(input, inputPitch, reference, referencePitch, x - 1, y - 1, width, height),
        rtgmc_repair_pos_local_contract(input, inputPitch, reference, referencePitch, x    , y - 1, width, height),
        rtgmc_repair_pos_local_contract(input, inputPitch, reference, referencePitch, x + 1, y - 1, width, height),
        rtgmc_repair_pos_local_contract(input, inputPitch, reference, referencePitch, x - 1, y    , width, height),
        rtgmc_repair_pos_local_contract(input, inputPitch, reference, referencePitch, x + 1, y    , width, height),
        rtgmc_repair_pos_local_contract(input, inputPitch, reference, referencePitch, x - 1, y + 1, width, height),
        rtgmc_repair_pos_local_contract(input, inputPitch, reference, referencePitch, x    , y + 1, width, height),
        rtgmc_repair_pos_local_contract(input, inputPitch, reference, referencePitch, x + 1, y + 1, width, height)
    };
    rtgmc_repair_sort8(v);
    return clamp(center, v[3], v[4]);
}

static inline int rtgmc_repair_neg_rank_limit(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    const int center = rtgmc_repair_neg_local_expand(input, inputPitch, reference, referencePitch, x, y, width, height);
    if (RTGMC_SHIMMER_REPAIR_THIN_LEVEL != 2 && RTGMC_SHIMMER_REPAIR_THIN_LEVEL != 5) {
        return center;
    }
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return center;
    }
    int v[8] = {
        rtgmc_repair_neg_local_expand(input, inputPitch, reference, referencePitch, x - 1, y - 1, width, height),
        rtgmc_repair_neg_local_expand(input, inputPitch, reference, referencePitch, x    , y - 1, width, height),
        rtgmc_repair_neg_local_expand(input, inputPitch, reference, referencePitch, x + 1, y - 1, width, height),
        rtgmc_repair_neg_local_expand(input, inputPitch, reference, referencePitch, x - 1, y    , width, height),
        rtgmc_repair_neg_local_expand(input, inputPitch, reference, referencePitch, x + 1, y    , width, height),
        rtgmc_repair_neg_local_expand(input, inputPitch, reference, referencePitch, x - 1, y + 1, width, height),
        rtgmc_repair_neg_local_expand(input, inputPitch, reference, referencePitch, x    , y + 1, width, height),
        rtgmc_repair_neg_local_expand(input, inputPitch, reference, referencePitch, x + 1, y + 1, width, height)
    };
    rtgmc_repair_sort8(v);
    return clamp(center, v[3], v[4]);
}

static inline int rtgmc_repair_pos_vertical_restore(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    int value = rtgmc_repair_pos_rank_limit(input, inputPitch, reference, referencePitch, x, y, width, height);
    for (int dy = -2; dy <= 2; dy++) {
        value = max(value, rtgmc_repair_pos_rank_limit(input, inputPitch, reference, referencePitch, x, y + dy, width, height));
    }
    return value;
}

static inline int rtgmc_repair_neg_vertical_restore(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    int value = rtgmc_repair_neg_rank_limit(input, inputPitch, reference, referencePitch, x, y, width, height);
    for (int dy = -2; dy <= 2; dy++) {
        value = min(value, rtgmc_repair_neg_rank_limit(input, inputPitch, reference, referencePitch, x, y + dy, width, height));
    }
    return value;
}

static inline int rtgmc_repair_pos_restore_wide(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    int value = rtgmc_repair_pos_vertical_restore(input, inputPitch, reference, referencePitch, x, y, width, height);
    if (RTGMC_SHIMMER_REPAIR_THIN_LEVEL > 4) {
        for (int dy = -1; dy <= 1; dy++) {
            value = max(value, rtgmc_repair_pos_vertical_restore(input, inputPitch, reference, referencePitch, x, y + dy, width, height));
        }
    }
    return value;
}

static inline int rtgmc_repair_neg_restore_wide(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    int value = rtgmc_repair_neg_vertical_restore(input, inputPitch, reference, referencePitch, x, y, width, height);
    if (RTGMC_SHIMMER_REPAIR_THIN_LEVEL > 4) {
        for (int dy = -1; dy <= 1; dy++) {
            value = min(value, rtgmc_repair_neg_vertical_restore(input, inputPitch, reference, referencePitch, x, y + dy, width, height));
        }
    }
    return value;
}

static inline int rtgmc_repair_pos_restore_soft_once(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    const int center = rtgmc_repair_pos_restore_wide(input, inputPitch, reference, referencePitch, x, y, width, height);
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return center;
    }
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += rtgmc_repair_pos_restore_wide(input, inputPitch, reference, referencePitch, x + dx, y + dy, width, height);
        }
    }
    return max(center, (sum + 4) / 9);
}

static inline int rtgmc_repair_neg_restore_soft_once(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    const int center = rtgmc_repair_neg_restore_wide(input, inputPitch, reference, referencePitch, x, y, width, height);
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return center;
    }
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += rtgmc_repair_neg_restore_wide(input, inputPitch, reference, referencePitch, x + dx, y + dy, width, height);
        }
    }
    return min(center, (sum + 4) / 9);
}

static inline int rtgmc_repair_pos_restore_soft_twice(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    const int center = rtgmc_repair_pos_restore_soft_once(input, inputPitch, reference, referencePitch, x, y, width, height);
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return center;
    }
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += rtgmc_repair_pos_restore_soft_once(input, inputPitch, reference, referencePitch, x + dx, y + dy, width, height);
        }
    }
    return max(center, (sum + 4) / 9);
}

static inline int rtgmc_repair_neg_restore_soft_twice(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    const int center = rtgmc_repair_neg_restore_soft_once(input, inputPitch, reference, referencePitch, x, y, width, height);
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return center;
    }
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += rtgmc_repair_neg_restore_soft_once(input, inputPitch, reference, referencePitch, x + dx, y + dy, width, height);
        }
    }
    return min(center, (sum + 4) / 9);
}

static inline int rtgmc_repair_pos_restore_area(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    int value = rtgmc_repair_pos_restore_wide(input, inputPitch, reference, referencePitch, x, y, width, height);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            value = max(value, rtgmc_repair_pos_restore_wide(input, inputPitch, reference, referencePitch, x + dx, y + dy, width, height));
        }
    }
    return value;
}

static inline int rtgmc_repair_neg_restore_area(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    int value = rtgmc_repair_neg_restore_wide(input, inputPitch, reference, referencePitch, x, y, width, height);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            value = min(value, rtgmc_repair_neg_restore_wide(input, inputPitch, reference, referencePitch, x + dx, y + dy, width, height));
        }
    }
    return value;
}

static inline int rtgmc_repair_pos_limit(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    int value = rtgmc_repair_pos_restore_wide(input, inputPitch, reference, referencePitch, x, y, width, height);
    if (RTGMC_SHIMMER_REPAIR_PAD_LEVEL == 1 || RTGMC_SHIMMER_REPAIR_PAD_LEVEL == 2) {
        value = (RTGMC_SHIMMER_REPAIR_PAD_LEVEL == 1)
            ? rtgmc_repair_pos_restore_soft_once(input, inputPitch, reference, referencePitch, x, y, width, height)
            : rtgmc_repair_pos_restore_soft_twice(input, inputPitch, reference, referencePitch, x, y, width, height);
    } else if (RTGMC_SHIMMER_REPAIR_PAD_LEVEL >= 3) {
        value = rtgmc_repair_pos_restore_area(input, inputPitch, reference, referencePitch, x, y, width, height);
    }
    return value;
}

static inline int rtgmc_repair_neg_limit(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    int value = rtgmc_repair_neg_restore_wide(input, inputPitch, reference, referencePitch, x, y, width, height);
    if (RTGMC_SHIMMER_REPAIR_PAD_LEVEL == 1 || RTGMC_SHIMMER_REPAIR_PAD_LEVEL == 2) {
        value = (RTGMC_SHIMMER_REPAIR_PAD_LEVEL == 1)
            ? rtgmc_repair_neg_restore_soft_once(input, inputPitch, reference, referencePitch, x, y, width, height)
            : rtgmc_repair_neg_restore_soft_twice(input, inputPitch, reference, referencePitch, x, y, width, height);
    } else if (RTGMC_SHIMMER_REPAIR_PAD_LEVEL >= 3) {
        value = rtgmc_repair_neg_restore_area(input, inputPitch, reference, referencePitch, x, y, width, height);
    }
    return value;
}

static inline int rtgmc_repair_limited_delta(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height
) {
    int diff = rtgmc_repair_delta_centered(input, inputPitch, reference, referencePitch, x, y, width, height);
    if (diff >= range_half + 1) {
        const int upperEnvelope = rtgmc_repair_pos_limit(input, inputPitch, reference, referencePitch, x, y, width, height);
        diff = max(upperEnvelope, range_half);
    } else if (diff <= range_half - 1) {
        const int lowerEnvelope = rtgmc_repair_neg_limit(input, inputPitch, reference, referencePitch, x, y, width, height);
        diff = min(lowerEnvelope, range_half);
    }
    return clamp(diff, 0, max_val);
}

__attribute__((reqd_work_group_size(rtgmc_shimmer_repair_block_x, rtgmc_shimmer_repair_block_y, 1)))
__kernel void kernel_rtgmc_shimmer_repair_copy(
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

__attribute__((reqd_work_group_size(rtgmc_shimmer_repair_block_x, rtgmc_shimmer_repair_block_y, 1)))
__kernel void kernel_rtgmc_shimmer_repair_apply(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *pInput, const int inputPitch,
    const __global Type *pReference, const int referencePitch,
    const int width,
    const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const __global uchar *input = (const __global uchar *)pInput;
    const __global uchar *reference = (const __global uchar *)pReference;
    const int inputValue = rtgmc_read_pix(input, ix, iy, inputPitch, width, height);
    const int mergedDiff = rtgmc_repair_limited_delta(input, inputPitch, reference, referencePitch, ix, iy, width, height);

    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch,
        clamp(inputValue + mergedDiff - range_half, 0, max_val));
}

__attribute__((reqd_work_group_size(rtgmc_shimmer_repair_block_x, rtgmc_shimmer_repair_block_y, 1)))
__kernel void kernel_rtgmc_shimmer_repair_apply_fused(
    __global Type *restrict pDst, const int dstPitch,
    __global Type *restrict pCorrectionDelta, const int correctionDeltaPitch,
    __global Type *restrict pPositiveCorrectionGate, const int positiveCorrectionGatePitch,
    __global Type *restrict pNegativeCorrectionGate, const int negativeCorrectionGatePitch,
    const __global Type *pInput, const int inputPitch,
    const __global Type *pReference, const int referencePitch,
    const int width,
    const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const __global uchar *input = (const __global uchar *)pInput;
    const __global uchar *reference = (const __global uchar *)pReference;
    const int inputValue = rtgmc_read_pix(input, ix, iy, inputPitch, width, height);
    const int referenceValue = rtgmc_read_pix(reference, ix, iy, referencePitch, width, height);
    const int signedDelta = referenceValue - inputValue;
    const int mergedDiff = rtgmc_repair_limited_delta(input, inputPitch, reference, referencePitch, ix, iy, width, height);
    const int selectedSigned = mergedDiff - range_half;
    const int positiveGateSigned = (signedDelta > 0 && selectedSigned > 0) ? selectedSigned : 0;
    const int negativeGateSigned = (signedDelta < 0 && selectedSigned < 0) ? selectedSigned : 0;

    rtgmc_write_pix((__global uchar *)pCorrectionDelta, ix, iy, correctionDeltaPitch, rtgmc_signed_to_diff(signedDelta));
    rtgmc_write_pix((__global uchar *)pPositiveCorrectionGate, ix, iy, positiveCorrectionGatePitch, rtgmc_signed_to_diff(positiveGateSigned));
    rtgmc_write_pix((__global uchar *)pNegativeCorrectionGate, ix, iy, negativeCorrectionGatePitch, rtgmc_signed_to_diff(negativeGateSigned));
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch,
        clamp(inputValue + selectedSigned, 0, max_val));
}
