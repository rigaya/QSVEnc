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

#ifndef DEGRAIN_TV_RANGE
#define DEGRAIN_TV_RANGE 0
#endif

static inline int degrain_analysis_luma_to_full_range(const int value) {
#if DEGRAIN_TV_RANGE
#if DEGRAIN_PIXEL_MAX <= 255
    const int converted = ((value - 16) * DEGRAIN_PIXEL_MAX + (219 >> 1)) / 219;
#else
    const int limitedScale = max((DEGRAIN_PIXEL_MAX + 1) >> 8, 1);
    const int limitedOffset = 16 * limitedScale;
    const int limitedRange = 219 * limitedScale;
    const int delta = value - limitedOffset;
    const int converted = delta + (delta * (DEGRAIN_PIXEL_MAX - limitedRange) + (limitedRange >> 1)) / limitedRange;
#endif
    return clamp(converted, 0, DEGRAIN_PIXEL_MAX);
#else
    return value;
#endif
}

__kernel void kernel_degrain_downsample_luma2x(
    __global const uchar *src,
    const int src_pitch,
    __global uchar *dst,
    const int dst_pitch,
    const int src_width,
    const int src_height,
    const int dst_width,
    const int dst_height) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= dst_width || y >= dst_height) {
        return;
    }

    // 4-tap 対称フィルタによる 2x ダウンサンプル。重み (1,3,3,1) を縦横独立に適用し、
    // 合計 (1+3+3+1)^2 = 64 で正規化する。境界外ピクセルは degrain_pixel_load 側で
    // 端値 clamp されるため、係数は端でも固定のまま。
    const int sx = x * 2;
    const int sy = y * 2;
    const int wx[4] = { 1, 3, 3, 1 };
    const int wy[4] = { 1, 3, 3, 1 };

    int sum = 0;
    for (int ky = 0; ky < 4; ++ky) {
        const int py = sy + ky - 1;
        for (int kx = 0; kx < 4; ++kx) {
            const int px = sx + kx - 1;
            const int pix = degrain_pixel_load(src, src_pitch, src_width, src_height, px, py);
            sum += pix * wy[ky] * wx[kx];
        }
    }
    *(__global TypePixel *)(dst + y * dst_pitch + x * (int)sizeof(TypePixel)) = degrain_clamp_pixel((sum + 32) >> 6);
}

#ifndef DEGRAIN_TR0
#define DEGRAIN_TR0 -1
#endif

#ifndef DEGRAIN_SEARCH_REFINE
#define DEGRAIN_SEARCH_REFINE -1
#endif

#ifndef DEGRAIN_REP0
#define DEGRAIN_REP0 -1
#endif

static inline int degrain_temporal_smooth_value(
    __global const uchar *srcPrev2,
    const int srcPrev2Pitch,
    __global const uchar *srcPrev,
    const int srcPrevPitch,
    __global const uchar *srcCur,
    const int srcCurPitch,
    __global const uchar *srcNext,
    const int srcNextPitch,
    __global const uchar *srcNext2,
    const int srcNext2Pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int smoothRadius) {
    int value = degrain_pixel_load(srcCur, srcCurPitch, srcWidth, srcHeight, px, py);
#if DEGRAIN_TR0 >= 2
    const int sum =
        degrain_pixel_load(srcPrev2, srcPrev2Pitch, srcWidth, srcHeight, px, py)
      + 4 * degrain_pixel_load(srcPrev, srcPrevPitch, srcWidth, srcHeight, px, py)
      + 6 * value
      + 4 * degrain_pixel_load(srcNext, srcNextPitch, srcWidth, srcHeight, px, py)
      + degrain_pixel_load(srcNext2, srcNext2Pitch, srcWidth, srcHeight, px, py);
    value = (sum + 8) >> 4;
#elif DEGRAIN_TR0 >= 1
    const int sum =
        degrain_pixel_load(srcPrev, srcPrevPitch, srcWidth, srcHeight, px, py)
      + 2 * value
      + degrain_pixel_load(srcNext, srcNextPitch, srcWidth, srcHeight, px, py);
    value = (sum + 2) >> 2;
#elif DEGRAIN_TR0 == 0
    (void)smoothRadius;
#else
    if (smoothRadius >= 2) {
        const int sum =
            degrain_pixel_load(srcPrev2, srcPrev2Pitch, srcWidth, srcHeight, px, py)
          + 4 * degrain_pixel_load(srcPrev, srcPrevPitch, srcWidth, srcHeight, px, py)
          + 6 * value
          + 4 * degrain_pixel_load(srcNext, srcNextPitch, srcWidth, srcHeight, px, py)
          + degrain_pixel_load(srcNext2, srcNext2Pitch, srcWidth, srcHeight, px, py);
        value = (sum + 8) >> 4;
    } else if (smoothRadius >= 1) {
        const int sum =
            degrain_pixel_load(srcPrev, srcPrevPitch, srcWidth, srcHeight, px, py)
          + 2 * value
          + degrain_pixel_load(srcNext, srcNextPitch, srcWidth, srcHeight, px, py);
        value = (sum + 2) >> 2;
    }
#endif
    return value;
}

static inline int degrain_search_refine1_value(
    __global const uchar *srcPrev2,
    const int srcPrev2Pitch,
    __global const uchar *srcPrev,
    const int srcPrevPitch,
    __global const uchar *srcCur,
    const int srcCurPitch,
    __global const uchar *srcNext,
    const int srcNextPitch,
    __global const uchar *srcNext2,
    const int srcNext2Pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int smoothRadius) {
    const int p00 = degrain_temporal_smooth_value(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px - 1, py - 1, smoothRadius);
    const int p10 = degrain_temporal_smooth_value(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px,     py - 1, smoothRadius);
    const int p20 = degrain_temporal_smooth_value(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px + 1, py - 1, smoothRadius);
    const int p01 = degrain_temporal_smooth_value(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px - 1, py,     smoothRadius);
    const int p11 = degrain_temporal_smooth_value(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px,     py,     smoothRadius);
    const int p21 = degrain_temporal_smooth_value(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px + 1, py,     smoothRadius);
    const int p02 = degrain_temporal_smooth_value(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px - 1, py + 1, smoothRadius);
    const int p12 = degrain_temporal_smooth_value(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px,     py + 1, smoothRadius);
    const int p22 = degrain_temporal_smooth_value(srcPrev2, srcPrev2Pitch, srcPrev, srcPrevPitch, srcCur, srcCurPitch, srcNext, srcNextPitch, srcNext2, srcNext2Pitch, srcWidth, srcHeight, px + 1, py + 1, smoothRadius);
    const int blur = degrain_blur3x3_weighted(p00, p10, p20, p01, p11, p21, p02, p12, p22);
    const int edgeSoft = degrain_edge_soften_cross(p01, p10, p11, p12, p21);
    return degrain_search_refine1_blend(p11, blur, edgeSoft, p01, p10, p21, p12);
}

static inline int degrain_analysis_prefilter_value(
    __global const uchar *srcPrev2,
    const int srcPrev2Pitch,
    __global const uchar *srcPrev,
    const int srcPrevPitch,
    __global const uchar *srcCur,
    const int srcCurPitch,
    __global const uchar *srcNext,
    const int srcNextPitch,
    __global const uchar *srcNext2,
    const int srcNext2Pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int smoothRadius,
    const int search_refine) {
#if DEGRAIN_SEARCH_REFINE >= 1
    return degrain_search_refine1_value(
        srcPrev2, srcPrev2Pitch,
        srcPrev, srcPrevPitch,
        srcCur, srcCurPitch,
        srcNext, srcNextPitch,
        srcNext2, srcNext2Pitch,
        srcWidth, srcHeight,
        px, py, smoothRadius);
#elif DEGRAIN_SEARCH_REFINE == 0
    (void)search_refine;
    return degrain_temporal_smooth_value(
        srcPrev2, srcPrev2Pitch,
        srcPrev, srcPrevPitch,
        srcCur, srcCurPitch,
        srcNext, srcNextPitch,
        srcNext2, srcNext2Pitch,
        srcWidth, srcHeight,
        px, py, smoothRadius);
#else
    int value = degrain_temporal_smooth_value(
        srcPrev2, srcPrev2Pitch,
        srcPrev, srcPrevPitch,
        srcCur, srcCurPitch,
        srcNext, srcNextPitch,
        srcNext2, srcNext2Pitch,
        srcWidth, srcHeight,
        px, py, smoothRadius);
    if (search_refine >= 1) {
        value = degrain_search_refine1_value(
            srcPrev2, srcPrev2Pitch,
            srcPrev, srcPrevPitch,
            srcCur, srcCurPitch,
            srcNext, srcNextPitch,
            srcNext2, srcNext2Pitch,
            srcWidth, srcHeight,
            px, py, smoothRadius);
    }
    return value;
#endif
}

static inline int degrain_rep0_repair_value(
    __global const uchar *srcPrev2,
    const int srcPrev2Pitch,
    __global const uchar *srcPrev,
    const int srcPrevPitch,
    __global const uchar *srcCur,
    const int srcCurPitch,
    __global const uchar *srcNext,
    const int srcNextPitch,
    __global const uchar *srcNext2,
    const int srcNext2Pitch,
    const int srcWidth,
    const int srcHeight,
    const int px,
    const int py,
    const int smoothRadius,
    const int search_refine) {
    const int p0 = degrain_analysis_prefilter_value(
        srcPrev2, srcPrev2Pitch,
        srcPrev, srcPrevPitch,
        srcCur, srcCurPitch,
        srcNext, srcNextPitch,
        srcNext2, srcNext2Pitch,
        srcWidth, srcHeight,
        px, py, smoothRadius, search_refine);
    const int p1u = degrain_analysis_prefilter_value(
        srcPrev2, srcPrev2Pitch,
        srcPrev, srcPrevPitch,
        srcCur, srcCurPitch,
        srcNext, srcNextPitch,
        srcNext2, srcNext2Pitch,
        srcWidth, srcHeight,
        px, py - 1, smoothRadius, search_refine);
    const int p1d = degrain_analysis_prefilter_value(
        srcPrev2, srcPrev2Pitch,
        srcPrev, srcPrevPitch,
        srcCur, srcCurPitch,
        srcNext, srcNextPitch,
        srcNext2, srcNext2Pitch,
        srcWidth, srcHeight,
        px, py + 1, smoothRadius, search_refine);
    const int p2u = degrain_analysis_prefilter_value(
        srcPrev2, srcPrev2Pitch,
        srcPrev, srcPrevPitch,
        srcCur, srcCurPitch,
        srcNext, srcNextPitch,
        srcNext2, srcNext2Pitch,
        srcWidth, srcHeight,
        px, py - 2, smoothRadius, search_refine);
    const int p2d = degrain_analysis_prefilter_value(
        srcPrev2, srcPrev2Pitch,
        srcPrev, srcPrevPitch,
        srcCur, srcCurPitch,
        srcNext, srcNextPitch,
        srcNext2, srcNext2Pitch,
        srcWidth, srcHeight,
        px, py + 2, smoothRadius, search_refine);

    const int vertAvg = (p1u + p1d + 1) >> 1;
    const int nearDiff = abs(p1u - p1d);
    const int farDiff = abs(p2u - p2d);
    const int centerDiff = abs(p0 - vertAvg);
    const int flatness = nearDiff + (farDiff >> 1);
    const int threshold = max(flatness + max(DEGRAIN_PIXEL_MAX / 64, 1), max(DEGRAIN_PIXEL_MAX / 32, 1));
    if (centerDiff <= threshold) {
        return p0;
    }

    const int repair = (p0 + 3 * vertAvg + 2) >> 2;
    const int lo = min(min(p1u, p1d), min(p2u, p2d));
    const int hi = max(max(p1u, p1d), max(p2u, p2d));
    return clamp(repair, lo, hi);
}

__kernel void kernel_degrain_temporal_smooth_luma(
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
    const int search_refine,
    const int rep0) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }

    const int analysisValue =
#if DEGRAIN_REP0 >= 1
        degrain_rep0_repair_value(
            prev2, src_pitch,
            prev, src_pitch,
            cur, src_pitch,
            next, src_pitch,
            next2, src_pitch,
            width, height,
            x, y, tr0, search_refine);
#elif DEGRAIN_REP0 == 0
        degrain_analysis_prefilter_value(
            prev2, src_pitch,
            prev, src_pitch,
            cur, src_pitch,
            next, src_pitch,
            next2, src_pitch,
            width, height,
            x, y, tr0, search_refine);
#else
        (rep0 >= 1)
            ? degrain_rep0_repair_value(
            prev2, src_pitch,
            prev, src_pitch,
            cur, src_pitch,
            next, src_pitch,
            next2, src_pitch,
            width, height,
            x, y, tr0, search_refine)
        : degrain_analysis_prefilter_value(
            prev2, src_pitch,
            prev, src_pitch,
            cur, src_pitch,
            next, src_pitch,
            next2, src_pitch,
            width, height,
            x, y, tr0, search_refine);
#endif

    *(__global TypePixel *)(dst + y * dst_pitch + x * (int)sizeof(TypePixel)) =
        degrain_clamp_pixel(degrain_analysis_luma_to_full_range(analysisValue));
}

__kernel void kernel_degrain_clear_analysis(
    __global degrain_mv_t *mv,
    __global degrain_sad_t *sad,
    const int count) {
    const int idx = (int)get_global_id(0);
    if (idx >= count) {
        return;
    }

    mv[idx].dx = 0;
    mv[idx].dy = 0;
    mv[idx].sad = 0;
    mv[idx].refdir = 0;
    mv[idx].flags = 0;
    mv[idx].reserved = 0;

    sad[idx].sad = 0;
    sad[idx].srcAvg = 0;
    sad[idx].refAvg = 0;
    sad[idx].reserved = 0;
}
