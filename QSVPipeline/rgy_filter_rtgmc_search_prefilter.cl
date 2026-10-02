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

static inline int rtgmc_search_repair_profile_restore_padding_level(const uint repairProfile) {
    return (int)(((repairProfile) >> 8) & 0xffu);
}

static inline uint rtgmc_search_repair_profile_thin_reject_flags(const uint repairProfile) {
    return ((repairProfile) >> 16) & 0xffu;
}

static inline uint rtgmc_search_repair_profile_restore_flags(const uint repairProfile) {
    return ((repairProfile) >> 24) & 0xffu;
}

static inline TypePixel rtgmc_search_prefilter_clamp_pixel(const int value) {
    return (TypePixel)clamp(value, 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
}

static inline int rtgmc_search_prefilter_pixel_load(
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

static inline void rtgmc_search_prefilter_pixel_store(
    __global uchar *dst,
    const int pitch,
    const int x,
    const int y,
    const int value) {
    *(__global TypePixel *)(dst + y * pitch + x * (int)sizeof(TypePixel)) = rtgmc_search_prefilter_clamp_pixel(value);
}

static inline int rtgmc_search_prefilter_blur3x3_weighted(
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

static inline int rtgmc_search_prefilter_range_half(void) {
    return (RTGMC_SEARCH_PREFILTER_PIXEL_MAX + 1) >> 1;
}

static inline int rtgmc_search_prefilter_range_scale(void) {
    return max((RTGMC_SEARCH_PREFILTER_PIXEL_MAX + 1) >> 8, 1);
}

static inline int rtgmc_search_prefilter_extreme_seed(const int highSide) {
    return highSide ? 0 : RTGMC_SEARCH_PREFILTER_PIXEL_MAX;
}

static inline int rtgmc_search_prefilter_extreme_merge(const int value, const int sample, const int highSide) {
    return highSide ? max(value, sample) : min(value, sample);
}

static inline int rtgmc_search_prefilter_polarity_core_seed(const int positive) {
    return rtgmc_search_prefilter_extreme_seed(!positive);
}

static inline int rtgmc_search_prefilter_polarity_core_merge(const int value, const int sample, const int positive) {
    return rtgmc_search_prefilter_extreme_merge(value, sample, !positive);
}

static inline int rtgmc_search_prefilter_polarity_envelope_seed(const int positive) {
    return rtgmc_search_prefilter_extreme_seed(positive);
}

static inline int rtgmc_search_prefilter_polarity_envelope_merge(const int value, const int sample, const int positive) {
    return rtgmc_search_prefilter_extreme_merge(value, sample, positive);
}

static inline void rtgmc_search_prefilter_sort2(__private int *a, __private int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = lo;
    *b = hi;
}

static inline void rtgmc_search_prefilter_sort2_desc(__private int *a, __private int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = hi;
    *b = lo;
}

// Batcher's Bitonic Sort (1968), 8 elements / 24 comparisons / depth 6.
static inline void rtgmc_search_prefilter_sort8(__private int *v) {
    rtgmc_search_prefilter_sort2     (&v[0], &v[1]); rtgmc_search_prefilter_sort2_desc(&v[2], &v[3]); rtgmc_search_prefilter_sort2     (&v[4], &v[5]); rtgmc_search_prefilter_sort2_desc(&v[6], &v[7]);
    rtgmc_search_prefilter_sort2     (&v[0], &v[2]); rtgmc_search_prefilter_sort2     (&v[1], &v[3]); rtgmc_search_prefilter_sort2_desc(&v[4], &v[6]); rtgmc_search_prefilter_sort2_desc(&v[5], &v[7]);
    rtgmc_search_prefilter_sort2     (&v[0], &v[1]); rtgmc_search_prefilter_sort2     (&v[2], &v[3]); rtgmc_search_prefilter_sort2_desc(&v[4], &v[5]); rtgmc_search_prefilter_sort2_desc(&v[6], &v[7]);
    rtgmc_search_prefilter_sort2     (&v[0], &v[4]); rtgmc_search_prefilter_sort2     (&v[1], &v[5]); rtgmc_search_prefilter_sort2     (&v[2], &v[6]); rtgmc_search_prefilter_sort2     (&v[3], &v[7]);
    rtgmc_search_prefilter_sort2     (&v[0], &v[2]); rtgmc_search_prefilter_sort2     (&v[1], &v[3]); rtgmc_search_prefilter_sort2     (&v[4], &v[6]); rtgmc_search_prefilter_sort2     (&v[5], &v[7]);
    rtgmc_search_prefilter_sort2     (&v[0], &v[1]); rtgmc_search_prefilter_sort2     (&v[2], &v[3]); rtgmc_search_prefilter_sort2     (&v[4], &v[5]); rtgmc_search_prefilter_sort2     (&v[6], &v[7]);
}

static inline int rtgmc_search_prefilter_temporal_sample(
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

static inline int rtgmc_search_prefilter_temporal_weighted_value(
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

static inline int rtgmc_search_prefilter_temporal_candidate_value(
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

static inline int rtgmc_search_prefilter_makediff_value(const int ref, const int src) {
    return clamp(ref - src + rtgmc_search_prefilter_range_half(), 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
}

static inline int rtgmc_search_prefilter_select_signed_correction(
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

static inline int rtgmc_search_prefilter_apply_signed_correction(
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

static inline int rtgmc_search_prefilter_round_float_to_pixel(const float value) {
    return clamp((int)(value + 0.5f), 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
}

static inline int rtgmc_search_prefilter_motion_guide_blend_value(const int spatialGuide, const int motionGuide) {
    const float guideWeight = 0.10f;
    const float value = mix((float)spatialGuide, (float)motionGuide, guideWeight);
    return clamp(convert_int_rte(value), 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
}

static inline int rtgmc_search_prefilter_search_smoothed3x3_value(
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

static inline int rtgmc_search_prefilter_motion_guide_stabilize_value(
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

static inline int rtgmc_search_prefilter_motion_guide_blend_stabilized_value(
    const int spatialGuide,
    const int motionGuide,
    const int fieldGuide) {
    const int blendedGuide = rtgmc_search_prefilter_motion_guide_blend_value(spatialGuide, motionGuide);
    return rtgmc_search_prefilter_motion_guide_stabilize_value(motionGuide, fieldGuide, blendedGuide);
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

static inline int rtgmc_search_prefilter_to_full_range(
    const int value,
    const int planeMode) {
    if (planeMode == 1) {
        const float lumaScale = (float)RTGMC_SEARCH_PREFILTER_PIXEL_MAX / (float)RTGMC_SEARCH_PREFILTER_LIMITED_Y_RANGE;
        const float lumaOffset = -(float)RTGMC_SEARCH_PREFILTER_LIMITED_Y_MIN * lumaScale;
        return clamp(convert_int_rte(fma((float)value, lumaScale, lumaOffset)), 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
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

// 中間プレーンは仮想座標をそのまま格納する。画面内へのクランプはTC/Dの参照だけで行う。
static inline int rtgmc_search_prefilter_repair_plane_load(
    __global const uchar *src, const int pitch,
    const int hx, const int hy, const int x, const int y) {
    return (int)(*(__global const TypePixel *)(src + (y + hy) * pitch + (x + hx) * (int)sizeof(TypePixel)));
}

// 端の判定にはハロー内の格納座標ではなく、旧実装と同じ仮想的な評価座標を使う。
static inline int rtgmc_search_prefilter_repair_plane_mean3x3(
    __global const uchar *src, const int pitch, const int hx, const int hy,
    const int width, const int height, const int x, const int y) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return rtgmc_search_prefilter_repair_plane_load(src, pitch, hx, hy, x, y);
    }
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += rtgmc_search_prefilter_repair_plane_load(src, pitch, hx, hy, x + dx, y + dy);
        }
    }
    return (sum + 4) / 9;
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_repair_temporal(
    __global const uchar *prev2, __global const uchar *prev, __global const uchar *cur,
    __global const uchar *next, __global const uchar *next2, const int srcPitch,
    __global uchar *tc, __global uchar *delta, const int planePitch,
    const int width, const int height, const int tr0) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int value = rtgmc_search_prefilter_temporal_candidate_value(
        prev2, prev, cur, next, next2, srcPitch, width, height, x, y, tr0);
    const int ref = rtgmc_search_prefilter_pixel_load(cur, srcPitch, width, height, x, y);
    rtgmc_search_prefilter_pixel_store(tc, planePitch, x, y, value);
    rtgmc_search_prefilter_pixel_store(delta, planePitch, x, y, rtgmc_search_prefilter_makediff_value(ref, value));
}

static inline int rtgmc_search_prefilter_repair_stage_value(
    __global const uchar *src, const int pitch, const int hx, const int hy,
    const int width, const int height, const int x, const int y,
    const uint repairProfile, const int stage, const int positive,
    __global const uchar *core, const int corePitch, const int coreHx, const int coreHy) {
    const uint thinFlags = rtgmc_search_repair_profile_thin_reject_flags(repairProfile);
    const uint restoreFlags = rtgmc_search_repair_profile_restore_flags(repairProfile);
    if (restoreFlags & RTGMC_SEARCH_REPAIR_RESTORE_LEVEL4_PATH) {
        if (stage == 0 || stage == 3) {
            int value = (stage == 0) ? rtgmc_search_prefilter_polarity_core_seed(positive)
                : rtgmc_search_prefilter_polarity_envelope_seed(positive);
            for (int dy = -2; dy <= 2; dy++) {
                // はみ出した行は評価行へ戻す。例えばy=1,dy=-2では0ではなく1を読む。
                // Dの画面クランプとは別の置換であり、仮想座標での評価でもこの順序を保つ。
                const int sampleY = (y + dy < 0 || y + dy >= height) ? y : y + dy;
                const int sample = (stage == 0)
                    ? rtgmc_search_prefilter_pixel_load(src, pitch, width, height, x, sampleY)
                    : rtgmc_search_prefilter_repair_plane_load(src, pitch, hx, hy, x, sampleY);
                value = (stage == 0) ? rtgmc_search_prefilter_polarity_core_merge(value, sample, positive)
                    : rtgmc_search_prefilter_polarity_envelope_merge(value, sample, positive);
            }
            return value;
        }
        if (stage == 1) {
            return rtgmc_search_prefilter_repair_plane_mean3x3(src, pitch, hx, hy, width, height, x, y);
        }
        // level4のmidは前段の平均とstage0のcoreを合成する。平均だけをcoreと再合成してはいけない。
        const int mean = rtgmc_search_prefilter_repair_plane_load(src, pitch, hx, hy, x, y);
        const int center = rtgmc_search_prefilter_repair_plane_load(core, corePitch, coreHx, coreHy, x, y);
        return rtgmc_search_prefilter_polarity_core_merge(center, mean, positive);
    }
    if (stage == 0) {
        const int radius = 2 + ((thinFlags & RTGMC_SEARCH_REPAIR_THIN_WIDE_CORE) ? 1 : 0);
        int value = rtgmc_search_prefilter_polarity_core_seed(positive);
        for (int dy = -radius; dy <= radius; dy++) {
            // Dだけは画面サイズ。旧実装の入力画素読み込みと同じクランプを保つ。
            const int sample = rtgmc_search_prefilter_pixel_load(src, pitch, width, height, x, y + dy);
            value = rtgmc_search_prefilter_polarity_core_merge(value, sample, positive);
        }
        return value;
    }
    if (stage == 3) {
        const int radius = 2 + ((restoreFlags & RTGMC_SEARCH_REPAIR_RESTORE_WIDE_ENVELOPE) ? 1 : 0);
        int value = rtgmc_search_prefilter_polarity_envelope_seed(positive);
        for (int dy = -radius; dy <= radius; dy++) {
            const int sample = rtgmc_search_prefilter_repair_plane_load(src, pitch, hx, hy, x, y + dy);
            value = rtgmc_search_prefilter_polarity_envelope_merge(value, sample, positive);
        }
        return value;
    }
    const int center = rtgmc_search_prefilter_repair_plane_load(src, pitch, hx, hy, x, y);
    if (stage == 1 && (thinFlags & RTGMC_SEARCH_REPAIR_THIN_CORE_BLEND)) {
        const int mean = rtgmc_search_prefilter_repair_plane_mean3x3(src, pitch, hx, hy, width, height, x, y);
        return rtgmc_search_prefilter_polarity_core_merge(center, mean, positive);
    }
    if (stage == 2 && (thinFlags & RTGMC_SEARCH_REPAIR_THIN_RANK_LIMIT)) {
        if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
            return center;
        }
        int v[8];
        int count = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx != 0 || dy != 0) {
                    v[count++] = rtgmc_search_prefilter_repair_plane_load(src, pitch, hx, hy, x + dx, y + dy);
                }
            }
        }
        rtgmc_search_prefilter_sort8(v);
        return clamp(center, v[3], v[4]);
    }
    const int padding = rtgmc_search_repair_profile_restore_padding_level(repairProfile);
    if ((stage == 4 && (padding == 1 || padding == 2)) || stage == 5) {
        // pad=2ではR1を保持した後、同じ端分岐と整数丸めでR2を別段として評価する。
        const int mean = rtgmc_search_prefilter_repair_plane_mean3x3(src, pitch, hx, hy, width, height, x, y);
        return rtgmc_search_prefilter_extreme_merge(center, mean, positive);
    }
    if (stage == 4 && padding >= 3) {
        // area envelopeには平均の端分岐がない。画面外のBも仮想座標のまま参照する。
        int value = rtgmc_search_prefilter_extreme_seed(positive);
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                const int sample = rtgmc_search_prefilter_repair_plane_load(src, pitch, hx, hy, x + dx, y + dy);
                value = rtgmc_search_prefilter_extreme_merge(value, sample, positive);
            }
        }
        return value;
    }
    return center;
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_repair_stage(
    __global const uchar *srcPos, __global const uchar *srcNeg,
    const int srcPitch, const int srcHx, const int srcHy,
    __global uchar *dstPos, __global uchar *dstNeg,
    const int dstPitch, const int dstHx, const int dstHy,
    const int width, const int height, const uint repairProfile, const int stage,
    __global const uchar *corePos, __global const uchar *coreNeg,
    const int corePitch, const int coreHx, const int coreHy) {
    const int sx = (int)get_global_id(0);
    const int sy = (int)get_global_id(1);
    if (sx >= width + 2 * dstHx || sy >= height + 2 * dstHy) {
        return;
    }
    const int x = sx - dstHx;
    const int y = sy - dstHy;
    const int positive = rtgmc_search_prefilter_repair_stage_value(
        srcPos, srcPitch, srcHx, srcHy, width, height, x, y, repairProfile, stage, 1, corePos, corePitch, coreHx, coreHy);
    const int negative = rtgmc_search_prefilter_repair_stage_value(
        srcNeg, srcPitch, srcHx, srcHy, width, height, x, y, repairProfile, stage, 0, coreNeg, corePitch, coreHx, coreHy);
    rtgmc_search_prefilter_pixel_store(dstPos, dstPitch, sx, sy, positive);
    rtgmc_search_prefilter_pixel_store(dstNeg, dstPitch, sx, sy, negative);
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_repair_apply(
    __global const uchar *tc, __global const uchar *delta, const int temporalPitch,
    __global const uchar *gatePos, __global const uchar *gateNeg,
    const int gatePitch, const int gateHx, const int gateHy,
    __global uchar *dst, const int dstPitch, const int dstHx, const int dstHy,
    const int width, const int height, const uint repairProfile) {
    const int sx = (int)get_global_id(0);
    const int sy = (int)get_global_id(1);
    if (sx >= width + 2 * dstHx || sy >= height + 2 * dstHy) {
        return;
    }
    const int x = sx - dstHx;
    const int y = sy - dstHy;
    const int base = rtgmc_search_prefilter_pixel_load(tc, temporalPitch, width, height, x, y);
    int value = base;
    if (rtgmc_search_repair_profile_restore_flags(repairProfile) & RTGMC_SEARCH_REPAIR_RESTORE_ENABLED) {
        const int rangeHalf = rtgmc_search_prefilter_range_half();
        const int diff = rtgmc_search_prefilter_pixel_load(delta, temporalPitch, width, height, x, y);
        const int positive = rtgmc_search_prefilter_repair_plane_load(gatePos, gatePitch, gateHx, gateHy, x, y);
        const int negative = rtgmc_search_prefilter_repair_plane_load(gateNeg, gatePitch, gateHx, gateHy, x, y);
        value = rtgmc_search_prefilter_apply_signed_correction(
            base, diff - rangeHalf, positive - rangeHalf, negative - rangeHalf, rtgmc_search_prefilter_range_scale());
    }
    rtgmc_search_prefilter_pixel_store(dst, dstPitch, sx, sy, value);
}

static inline int rtgmc_search_prefilter_half_search_base_from_fc_value(
    __global const uchar *fc, const int fcPitch, const int fcHx, const int fcHy,
    const int srcWidth, const int srcHeight, const int hx, const int hy) {
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
            // 小寸法では4タップが画面外へ出る。FCハローの仮想座標をクランプせず読む。
            const int sample = rtgmc_search_prefilter_repair_plane_load(
                fc, fcPitch, fcHx, fcHy, startX + ix, startY + iy);
            sumX += (coeffX[ix] / totalX) * (float)sample;
        }
        const int rowValue = clamp((int)sumX, 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
        sumY += (coeffY[iy] / totalY) * (float)rowValue;
    }
    return clamp((int)sumY, 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
}

static inline int rtgmc_search_prefilter_half_search_smoothed_from_fc_value(
    __global const uchar *fc, const int fcPitch, const int fcHx, const int fcHy,
    const int srcWidth, const int srcHeight, const int hx, const int hy) {
    const int halfWidth = max(srcWidth >> 1, 1);
    const int halfHeight = max(srcHeight >> 1, 1);
    if (hx <= 0 || hy <= 0 || hx >= halfWidth - 1 || hy >= halfHeight - 1) {
        return rtgmc_search_prefilter_half_search_base_from_fc_value(
            fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight,
            clamp(hx, 0, halfWidth - 1), clamp(hy, 0, halfHeight - 1));
    }
    const int x0 = clamp(hx - 1, 0, halfWidth - 1);
    const int x1 = clamp(hx,     0, halfWidth - 1);
    const int x2 = clamp(hx + 1, 0, halfWidth - 1);
    const int y0 = clamp(hy - 1, 0, halfHeight - 1);
    const int y1 = clamp(hy,     0, halfHeight - 1);
    const int y2 = clamp(hy + 1, 0, halfHeight - 1);
    const int p00 = rtgmc_search_prefilter_half_search_base_from_fc_value(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x0, y0);
    const int p10 = rtgmc_search_prefilter_half_search_base_from_fc_value(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x1, y0);
    const int p20 = rtgmc_search_prefilter_half_search_base_from_fc_value(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x2, y0);
    const int p01 = rtgmc_search_prefilter_half_search_base_from_fc_value(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x0, y1);
    const int p11 = rtgmc_search_prefilter_half_search_base_from_fc_value(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x1, y1);
    const int p21 = rtgmc_search_prefilter_half_search_base_from_fc_value(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x2, y1);
    const int p02 = rtgmc_search_prefilter_half_search_base_from_fc_value(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x0, y2);
    const int p12 = rtgmc_search_prefilter_half_search_base_from_fc_value(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x1, y2);
    const int p22 = rtgmc_search_prefilter_half_search_base_from_fc_value(fc, fcPitch, fcHx, fcHy, srcWidth, srcHeight, x2, y2);
    return rtgmc_search_prefilter_blur3x3_weighted(p00, p10, p20, p01, p11, p21, p02, p12, p22);
}

static inline int rtgmc_search_prefilter_half_resolution_search_from_fc_value(
    __global const uchar *fc, const int fcPitch, const int fcHx, const int fcHy,
    const int srcWidth, const int srcHeight, const int px, const int py) {
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
            const int sample = rtgmc_search_prefilter_half_search_smoothed_from_fc_value(
                fc, fcPitch, fcHx, fcHy,
                srcWidth, srcHeight,
                clamp(startX + ix, 0, halfWidth - 1),
                clamp(startY + iy, 0, halfHeight - 1));
            sumX += (coeffX[ix] / totalX) * (float)sample;
        }
        const int rowValue = clamp((int)sumX, 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
        sumY += (coeffY[iy] / totalY) * (float)rowValue;
    }
    return clamp((int)sumY, 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
}

// FCを読む経路は入力画像群を参照しない。丸めと係数の加算順序は旧経路に合わせる。
__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_half_search_base_from_fc(
    __global const uchar *fc, const int fcPitch, const int fcHx, const int fcHy,
    __global uchar *dst, const int dstPitch, const int width, const int height) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= max(width >> 1, 1) || y >= max(height >> 1, 1)) return;
    const int value = rtgmc_search_prefilter_half_search_base_from_fc_value(
        fc, fcPitch, fcHx, fcHy, width, height, x, y);
    rtgmc_search_prefilter_pixel_store(dst, dstPitch, x, y, value);
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_half_search_smoothed_from_fc(
    __global const uchar *fc, const int fcPitch, const int fcHx, const int fcHy,
    __global uchar *dst, const int dstPitch, const int width, const int height) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= max(width >> 1, 1) || y >= max(height >> 1, 1)) return;
    const int value = rtgmc_search_prefilter_half_search_smoothed_from_fc_value(
        fc, fcPitch, fcHx, fcHy, width, height, x, y);
    rtgmc_search_prefilter_pixel_store(dst, dstPitch, x, y, value);
}

__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_luma_from_fc(
    __global const uchar *fc, const int fcPitch, const int fcHx, const int fcHy,
    __global uchar *dst, const int dstPitch, const int width, const int height,
    const int search_refine, const int fullRangeMode) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    const int value = (search_refine >= 1)
        ? rtgmc_search_prefilter_half_resolution_search_from_fc_value(fc, fcPitch, fcHx, fcHy, width, height, x, y)
        : rtgmc_search_prefilter_repair_plane_load(fc, fcPitch, fcHx, fcHy, x, y);
    rtgmc_search_prefilter_pixel_store(dst, dstPitch, x, y,
        rtgmc_search_prefilter_to_full_range(value, fullRangeMode));
}

// TC/D/G/FCのデバッグ表示と画面内への出力を共通化する。無修復のゲートだけ中立値にする。
__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_copy_from_plane(
    __global const uchar *src, const int srcPitch, const int srcHx, const int srcHy,
    __global uchar *dst, const int dstPitch, const int width, const int height, const int neutral) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    const int value = neutral ? rtgmc_search_prefilter_range_half()
        : rtgmc_search_prefilter_repair_plane_load(src, srcPitch, srcHx, srcHy, x, y);
    rtgmc_search_prefilter_pixel_store(dst, dstPitch, x, y, value);
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

// search_refine=1 段階実行用: half_resolution_search_value と同一の係数・丸め手順で
// 半解像度smoothed planeからフル解像度へ戻し、fullRangeModeも同時に適用する。
__attribute__((reqd_work_group_size(rtgmc_search_prefilter_block_x, rtgmc_search_prefilter_block_y, 1)))
__kernel void kernel_rtgmc_search_prefilter_half_resolution_upsample(
    __global const uchar *smoothed,
    const int smoothedPitch,
    __global uchar *dst,
    const int dstPitch,
    const int width,
    const int height,
    const int fullRangeMode) {
    const int px = (int)get_global_id(0);
    const int py = (int)get_global_id(1);
    if (px >= width || py >= height) {
        return;
    }
    const int halfWidth = max(width >> 1, 1);
    const int halfHeight = max(height >> 1, 1);
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
            const int sample = rtgmc_search_prefilter_pixel_load(
                smoothed, smoothedPitch, halfWidth, halfHeight,
                clamp(startX + ix, 0, halfWidth - 1),
                clamp(startY + iy, 0, halfHeight - 1));
            sumX += (coeffX[ix] / totalX) * (float)sample;
        }
        const int rowValue = clamp((int)sumX, 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX);
        sumY += (coeffY[iy] / totalY) * (float)rowValue;
    }
    const int value = rtgmc_search_prefilter_to_full_range(
        clamp((int)sumY, 0, RTGMC_SEARCH_PREFILTER_PIXEL_MAX),
        fullRangeMode);
    rtgmc_search_prefilter_pixel_store(dst, dstPitch, px, py, value);
}
