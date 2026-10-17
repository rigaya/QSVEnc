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

static inline int rtgmc_repair_stage_read(
    const __global Type *src,
    const int pitch,
    const int x,
    const int bufferY
) {
    return (int)(*(const __global Type *)((const __global uchar *)src + bufferY * pitch + x * sizeof(Type)));
}

static inline void rtgmc_repair_stage_write(
    __global Type *dst,
    const int pitch,
    const int x,
    const int bufferY,
    const int value
) {
    *(__global Type *)((__global uchar *)dst + bufferY * pitch + x * sizeof(Type)) = (Type)value;
}

__attribute__((reqd_work_group_size(rtgmc_shimmer_repair_block_x, rtgmc_shimmer_repair_block_y, 1)))
__kernel void kernel_rtgmc_shimmer_repair_stage_vertical(
    __global Type *restrict pVerticalContractPositive,
    __global Type *restrict pVerticalExpandNegative,
    const int stagePitch,
    const __global Type *pInput, const int inputPitch,
    const __global Type *pReference, const int referencePitch,
    const int width,
    const int height,
    const int stageYOffset
) {
    const int ix = get_global_id(0);
    const int bufferY = get_global_id(1);
    const int stagedHeight = height + stageYOffset * 2;
    if (ix >= width || bufferY >= stagedHeight) return;

    const int logicalY = bufferY - stageYOffset;
    const __global uchar *input = (const __global uchar *)pInput;
    const __global uchar *reference = (const __global uchar *)pReference;
    int positive = max_val;
    int negative = 0;
    for (int dy = -2; dy <= 2; dy++) {
        const int sample = rtgmc_repair_delta_centered(
            input, inputPitch, reference, referencePitch, ix, logicalY + dy, width, height);
        positive = min(positive, sample);
        negative = max(negative, sample);
    }
    rtgmc_repair_stage_write(pVerticalContractPositive, stagePitch, ix, bufferY, positive);
    rtgmc_repair_stage_write(pVerticalExpandNegative, stagePitch, ix, bufferY, negative);
}

__attribute__((reqd_work_group_size(rtgmc_shimmer_repair_block_x, rtgmc_shimmer_repair_block_y, 1)))
__kernel void kernel_rtgmc_shimmer_repair_stage_local(
    __global Type *restrict pLocalContractPositive,
    __global Type *restrict pLocalExpandNegative,
    const __global Type *pVerticalContractPositive,
    const __global Type *pVerticalExpandNegative,
    const int stagePitch,
    const int width,
    const int height,
    const int stageYOffset
) {
    const int ix = get_global_id(0);
    const int bufferY = get_global_id(1);
    const int stagedHeight = height + stageYOffset * 2;
    if (ix >= width || bufferY >= stagedHeight) return;

    const int logicalY = bufferY - stageYOffset;
    const int centerPositive = rtgmc_repair_stage_read(pVerticalContractPositive, stagePitch, ix, bufferY);
    const int centerNegative = rtgmc_repair_stage_read(pVerticalExpandNegative, stagePitch, ix, bufferY);
    int positive = centerPositive;
    int negative = centerNegative;
    if (ix > 0 && ix < width - 1 && logicalY > 0 && logicalY < height - 1) {
        int sumPositive = 0;
        int sumNegative = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                sumPositive += rtgmc_repair_stage_read(
                    pVerticalContractPositive, stagePitch, ix + dx, bufferY + dy);
                sumNegative += rtgmc_repair_stage_read(
                    pVerticalExpandNegative, stagePitch, ix + dx, bufferY + dy);
            }
        }
        positive = min(centerPositive, (sumPositive + 4) / 9);
        negative = max(centerNegative, (sumNegative + 4) / 9);
    }
    rtgmc_repair_stage_write(pLocalContractPositive, stagePitch, ix, bufferY, positive);
    rtgmc_repair_stage_write(pLocalExpandNegative, stagePitch, ix, bufferY, negative);
}

static inline int rtgmc_repair_staged_limited_delta(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const __global Type *localContractPositive,
    const __global Type *localExpandNegative,
    const int stagePitch,
    const int x,
    const int y,
    const int width,
    const int height,
    const int stageYOffset
) {
    int diff = rtgmc_repair_delta_centered(
        input, inputPitch, reference, referencePitch, x, y, width, height);
    if (diff >= range_half + 1) {
        int upperEnvelope = rtgmc_repair_stage_read(
            localContractPositive, stagePitch, x, y - 2 + stageYOffset);
        for (int dy = -1; dy <= 2; dy++) {
            upperEnvelope = max(upperEnvelope, rtgmc_repair_stage_read(
                localContractPositive, stagePitch, x, y + dy + stageYOffset));
        }
        diff = max(upperEnvelope, range_half);
    } else if (diff <= range_half - 1) {
        int lowerEnvelope = rtgmc_repair_stage_read(
            localExpandNegative, stagePitch, x, y - 2 + stageYOffset);
        for (int dy = -1; dy <= 2; dy++) {
            lowerEnvelope = min(lowerEnvelope, rtgmc_repair_stage_read(
                localExpandNegative, stagePitch, x, y + dy + stageYOffset));
        }
        diff = min(lowerEnvelope, range_half);
    }
    return clamp(diff, 0, max_val);
}

__attribute__((reqd_work_group_size(rtgmc_shimmer_repair_block_x, rtgmc_shimmer_repair_block_y, 1)))
__kernel void kernel_rtgmc_shimmer_repair_apply_staged(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *pInput, const int inputPitch,
    const __global Type *pReference, const int referencePitch,
    const __global Type *pLocalContractPositive,
    const __global Type *pLocalExpandNegative,
    const int stagePitch,
    const int width,
    const int height,
    const int stageYOffset
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const __global uchar *input = (const __global uchar *)pInput;
    const __global uchar *reference = (const __global uchar *)pReference;
    const int inputValue = rtgmc_read_pix(input, ix, iy, inputPitch, width, height);
    const int mergedDiff = rtgmc_repair_staged_limited_delta(
        input, inputPitch, reference, referencePitch,
        pLocalContractPositive, pLocalExpandNegative, stagePitch,
        ix, iy, width, height, stageYOffset);
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch,
        clamp(inputValue + mergedDiff - range_half, 0, max_val));
}

// -----------------------------------------------------------------------------------------
// staged経路 (thin 4 / pad 0) の3段 (stage_vertical -> stage_local -> apply_staged) を1カーネルで行う版。
// 3段版は段ごとに全画面の中間バッファ (正/負の2プレーン) をglobalへ書いては読み戻しており、
// 1画素1 work-itemで3x3平均も18回のglobal読み出しになっていた。
// 3x3平均は横3和を縦に3行足す分離形で、直前2行の横3和をレジスタに持ち回す。
// ここではwork-groupが出力タイル (RTGMC_SHIMMER_REPAIR_TILE_W x RTGMC_SHIMMER_REPAIR_TILE_H) を担当し、
// halo (縦5行、横1列) を含めてlocal memory上で delta -> 縦5tap min/max -> 3x3平均 -> 縦5tap envelope を計算する。
// すべて整数演算で、端の扱い (画素読み出しの座標clamp、3x3はlogical行1..h-2かつ列1..w-2のみ) も3段版と同じなので、出力は一致する。
// -----------------------------------------------------------------------------------------
#define RTGMC_SHIMMER_REPAIR_TILE_W 64
#define RTGMC_SHIMMER_REPAIR_TILE_H 16
// work-groupは64x4。x方向はタイル幅と同じにして、各段のループで除算・剰余を使わないようにする。
#define RTGMC_SHIMMER_REPAIR_TILE_LOCAL_Y 4
// local memoryの中間値はintで持つ。uchar/ushortのままだと隣接laneが同じ32bit語へ書き込んで衝突し、
// 3段版より遅くなった (A310で uchar 1.68ms → int 0.88ms/回)。
#define RTGMC_SHIMMER_REPAIR_TILE_STORE int
// delta: 出力行の上下5行 (envelope 2 + 3x3 1 + 縦window 2)、左右1列
#define RTGMC_SHIMMER_REPAIR_TILE_DELTA_H (RTGMC_SHIMMER_REPAIR_TILE_H + 10)
#define RTGMC_SHIMMER_REPAIR_TILE_DELTA_W (RTGMC_SHIMMER_REPAIR_TILE_W + 2)
// 縦5tap min/max: 出力行の上下3行、左右1列
#define RTGMC_SHIMMER_REPAIR_TILE_V_H (RTGMC_SHIMMER_REPAIR_TILE_H + 6)
// 3x3平均: 出力行の上下2行
#define RTGMC_SHIMMER_REPAIR_TILE_L_H (RTGMC_SHIMMER_REPAIR_TILE_H + 4)

__attribute__((reqd_work_group_size(RTGMC_SHIMMER_REPAIR_TILE_W, RTGMC_SHIMMER_REPAIR_TILE_LOCAL_Y, 1)))
__kernel void kernel_rtgmc_shimmer_repair_staged_tile(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *pInput, const int inputPitch,
    const __global Type *pReference, const int referencePitch,
    const int width,
    const int height
) {
    __local RTGMC_SHIMMER_REPAIR_TILE_STORE delta[RTGMC_SHIMMER_REPAIR_TILE_DELTA_H][RTGMC_SHIMMER_REPAIR_TILE_DELTA_W];
    __local RTGMC_SHIMMER_REPAIR_TILE_STORE verticalPositive[RTGMC_SHIMMER_REPAIR_TILE_V_H][RTGMC_SHIMMER_REPAIR_TILE_DELTA_W];
    __local RTGMC_SHIMMER_REPAIR_TILE_STORE verticalNegative[RTGMC_SHIMMER_REPAIR_TILE_V_H][RTGMC_SHIMMER_REPAIR_TILE_DELTA_W];
    __local RTGMC_SHIMMER_REPAIR_TILE_STORE localPositive[RTGMC_SHIMMER_REPAIR_TILE_L_H][RTGMC_SHIMMER_REPAIR_TILE_W];
    __local RTGMC_SHIMMER_REPAIR_TILE_STORE localNegative[RTGMC_SHIMMER_REPAIR_TILE_L_H][RTGMC_SHIMMER_REPAIR_TILE_W];

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int x0 = get_group_id(0) * RTGMC_SHIMMER_REPAIR_TILE_W;
    const int y0 = get_group_id(1) * RTGMC_SHIMMER_REPAIR_TILE_H;
    const __global uchar *input = (const __global uchar *)pInput;
    const __global uchar *reference = (const __global uchar *)pReference;

    // delta: logical座標 (x0-1+ix, y0-5+iy)。座標のclampはrtgmc_read_pix内で行う (3段版と同じ)。
    for (int iy = ly; iy < RTGMC_SHIMMER_REPAIR_TILE_DELTA_H; iy += RTGMC_SHIMMER_REPAIR_TILE_LOCAL_Y) {
        for (int ix = lx; ix < RTGMC_SHIMMER_REPAIR_TILE_DELTA_W; ix += RTGMC_SHIMMER_REPAIR_TILE_W) {
            delta[iy][ix] = (RTGMC_SHIMMER_REPAIR_TILE_STORE)rtgmc_repair_delta_centered(input, inputPitch, reference, referencePitch, x0 - 1 + ix, y0 - 5 + iy, width, height);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 縦5tap min/max (stage_vertical): logical行 y0-3+iy、delta行 iy..iy+4
    for (int iy = ly; iy < RTGMC_SHIMMER_REPAIR_TILE_V_H; iy += RTGMC_SHIMMER_REPAIR_TILE_LOCAL_Y) {
        for (int ix = lx; ix < RTGMC_SHIMMER_REPAIR_TILE_DELTA_W; ix += RTGMC_SHIMMER_REPAIR_TILE_W) {
            int positive = max_val;
            int negative = 0;
            for (int dy = 0; dy < 5; dy++) {
                const int sample = delta[iy + dy][ix];
                positive = min(positive, sample);
                negative = max(negative, sample);
            }
            verticalPositive[iy][ix] = (RTGMC_SHIMMER_REPAIR_TILE_STORE)positive;
            verticalNegative[iy][ix] = (RTGMC_SHIMMER_REPAIR_TILE_STORE)negative;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3x3平均 (stage_local): logical (x0+lx, y0-2+iy)、vertical行 iy+1 / 列 lx+1 が中心。
    // 各work-itemは連続する行を担当し、横3和を直前2行ぶんレジスタに持ち回して縦に3行足す (整数和なので順序に依らず一致)。
    {
        const int x = x0 + lx;
        const int rowsPerItem = (RTGMC_SHIMMER_REPAIR_TILE_L_H + RTGMC_SHIMMER_REPAIR_TILE_LOCAL_Y - 1) / RTGMC_SHIMMER_REPAIR_TILE_LOCAL_Y;
        const int rowBegin = ly * rowsPerItem;
        const int rowEnd = min(rowBegin + rowsPerItem, RTGMC_SHIMMER_REPAIR_TILE_L_H);
        const int interiorX = x > 0 && x < width - 1;
        int hPositive0 = 0, hPositive1 = 0, hNegative0 = 0, hNegative1 = 0;
        if (rowBegin < rowEnd) {
            hPositive0 = verticalPositive[rowBegin][lx] + verticalPositive[rowBegin][lx + 1] + verticalPositive[rowBegin][lx + 2];
            hNegative0 = verticalNegative[rowBegin][lx] + verticalNegative[rowBegin][lx + 1] + verticalNegative[rowBegin][lx + 2];
            hPositive1 = verticalPositive[rowBegin + 1][lx] + verticalPositive[rowBegin + 1][lx + 1] + verticalPositive[rowBegin + 1][lx + 2];
            hNegative1 = verticalNegative[rowBegin + 1][lx] + verticalNegative[rowBegin + 1][lx + 1] + verticalNegative[rowBegin + 1][lx + 2];
        }
        for (int iy = rowBegin; iy < rowEnd; iy++) {
            const int hPositive2 = verticalPositive[iy + 2][lx] + verticalPositive[iy + 2][lx + 1] + verticalPositive[iy + 2][lx + 2];
            const int hNegative2 = verticalNegative[iy + 2][lx] + verticalNegative[iy + 2][lx + 1] + verticalNegative[iy + 2][lx + 2];
            const int logicalY = y0 - 2 + iy;
            const int centerPositive = verticalPositive[iy + 1][lx + 1];
            const int centerNegative = verticalNegative[iy + 1][lx + 1];
            int positive = centerPositive;
            int negative = centerNegative;
            if (interiorX && logicalY > 0 && logicalY < height - 1) {
                positive = min(centerPositive, (int)((uint)(hPositive0 + hPositive1 + hPositive2 + 4) / 9u)); // 和は非負なので符号無し除算でも同じ値
                negative = max(centerNegative, (int)((uint)(hNegative0 + hNegative1 + hNegative2 + 4) / 9u));
            }
            localPositive[iy][lx] = (RTGMC_SHIMMER_REPAIR_TILE_STORE)positive;
            localNegative[iy][lx] = (RTGMC_SHIMMER_REPAIR_TILE_STORE)negative;
            hPositive0 = hPositive1; hPositive1 = hPositive2;
            hNegative0 = hNegative1; hNegative1 = hNegative2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 縦5tap envelopeで制限して適用 (apply_staged): logical (x0+lx, y0+iy)、local行 iy..iy+4
    const int x = x0 + lx;
    if (x >= width) {
        return;
    }
    for (int iy = ly; iy < RTGMC_SHIMMER_REPAIR_TILE_H; iy += RTGMC_SHIMMER_REPAIR_TILE_LOCAL_Y) {
        const int y = y0 + iy;
        if (y >= height) {
            break;
        }
        int diff = delta[iy + 5][lx + 1];
        if (diff >= range_half + 1) {
            int upperEnvelope = localPositive[iy][lx];
            for (int dy = 1; dy < 5; dy++) {
                upperEnvelope = max(upperEnvelope, (int)localPositive[iy + dy][lx]);
            }
            diff = max(upperEnvelope, range_half);
        } else if (diff <= range_half - 1) {
            int lowerEnvelope = localNegative[iy][lx];
            for (int dy = 1; dy < 5; dy++) {
                lowerEnvelope = min(lowerEnvelope, (int)localNegative[iy + dy][lx]);
            }
            diff = min(lowerEnvelope, range_half);
        }
        diff = clamp(diff, 0, max_val);
        const int inputValue = rtgmc_read_pix(input, x, y, inputPitch, width, height);
        rtgmc_write_pix((__global uchar *)pDst, x, y, dstPitch, clamp(inputValue + diff - range_half, 0, max_val));
    }
}

__attribute__((reqd_work_group_size(rtgmc_shimmer_repair_block_x, rtgmc_shimmer_repair_block_y, 1)))
__kernel void kernel_rtgmc_shimmer_repair_apply_fused_staged(
    __global Type *restrict pDst, const int dstPitch,
    __global Type *restrict pCorrectionDelta, const int correctionDeltaPitch,
    __global Type *restrict pPositiveCorrectionGate, const int positiveCorrectionGatePitch,
    __global Type *restrict pNegativeCorrectionGate, const int negativeCorrectionGatePitch,
    const __global Type *pInput, const int inputPitch,
    const __global Type *pReference, const int referencePitch,
    const __global Type *pLocalContractPositive,
    const __global Type *pLocalExpandNegative,
    const int stagePitch,
    const int width,
    const int height,
    const int stageYOffset
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const __global uchar *input = (const __global uchar *)pInput;
    const __global uchar *reference = (const __global uchar *)pReference;
    const int inputValue = rtgmc_read_pix(input, ix, iy, inputPitch, width, height);
    const int referenceValue = rtgmc_read_pix(reference, ix, iy, referencePitch, width, height);
    const int signedDelta = referenceValue - inputValue;
    const int mergedDiff = rtgmc_repair_staged_limited_delta(
        input, inputPitch, reference, referencePitch,
        pLocalContractPositive, pLocalExpandNegative, stagePitch,
        ix, iy, width, height, stageYOffset);
    const int selectedSigned = mergedDiff - range_half;
    const int positiveGateSigned = (signedDelta > 0 && selectedSigned > 0) ? selectedSigned : 0;
    const int negativeGateSigned = (signedDelta < 0 && selectedSigned < 0) ? selectedSigned : 0;

    rtgmc_write_pix((__global uchar *)pCorrectionDelta, ix, iy, correctionDeltaPitch, rtgmc_signed_to_diff(signedDelta));
    rtgmc_write_pix((__global uchar *)pPositiveCorrectionGate, ix, iy, positiveCorrectionGatePitch, rtgmc_signed_to_diff(positiveGateSigned));
    rtgmc_write_pix((__global uchar *)pNegativeCorrectionGate, ix, iy, negativeCorrectionGatePitch, rtgmc_signed_to_diff(negativeGateSigned));
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch,
        clamp(inputValue + selectedSigned, 0, max_val));
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
