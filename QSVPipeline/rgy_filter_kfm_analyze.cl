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

#ifndef Type
#define Type uchar
#endif

typedef struct {
    int move;
    int shima;
    int lshima;
} FMCount;

static inline int kfm_absdiff(const Type a, const Type b) {
    return abs((int)a - (int)b);
}

static inline int kfm_calc_combe(
    const Type L0, const Type L1, const Type L2, const Type L3,
    const Type L4, const Type L5, const Type L6, const Type L7) {
    const int diff8 = kfm_absdiff(L0, L7);
    const int diffT =
        kfm_absdiff(L0, L1) + kfm_absdiff(L1, L2) + kfm_absdiff(L2, L3) + kfm_absdiff(L3, L4) +
        kfm_absdiff(L4, L5) + kfm_absdiff(L5, L6) + kfm_absdiff(L6, L7) - diff8;
    const int diffE =
        kfm_absdiff(L0, L2) + kfm_absdiff(L2, L4) + kfm_absdiff(L4, L6) + kfm_absdiff(L6, L7) - diff8;
    const int diffO =
        kfm_absdiff(L0, L1) + kfm_absdiff(L1, L3) + kfm_absdiff(L3, L5) + kfm_absdiff(L5, L7) - diff8;
    return diffT - diffE - diffO;
}

static inline int kfm_calc_diff(
    const Type L00, const Type L10, const Type L01, const Type L11,
    const Type L02, const Type L12, const Type L03, const Type L13) {
    return kfm_absdiff(L00, L10) + kfm_absdiff(L01, L11) + kfm_absdiff(L02, L12) + kfm_absdiff(L03, L13);
}

static inline int kfm_clamp_u8(const int v) {
    return clamp(v, 0, 255);
}

static inline Type kfm_load_src(
    const __global Type *src,
    const int pitch,
    const int x,
    const int y,
    const int pixelStep,
    const int pixelOffset) {
    return src[x * pixelStep + pixelOffset + y * pitch];
}

static inline uchar4 kfm_analyze_block(
    const __global uchar *src0,
    const __global uchar *src1,
    const int srcPitch,
    const int parity,
    const int pixelStep,
    const int pixelOffset,
    const int bx,
    const int by) {
    const int shift = bit_depth - 8 + 4;
    const int srcPitchT = srcPitch / (int)sizeof(Type);
    const __global Type *f0 = (const __global Type *)src0;
    const __global Type *f1 = (const __global Type *)src1;

    int sum0 = 0;
    int sum1 = 0;
    int sum2 = 0;
    int sum3 = 0;
    const int xBase = bx * 4;
    const int yBase = by * 4;

    for (int tx = 0; tx < 8; ++tx) {
        const int x = xBase + tx;
        const int y = yBase;

        {
            const Type T00 = kfm_load_src(f0, srcPitchT, x, y + 0, pixelStep, pixelOffset);
            const Type B00 = kfm_load_src(f0, srcPitchT, x, y + 1, pixelStep, pixelOffset);
            const Type T01 = kfm_load_src(f0, srcPitchT, x, y + 2, pixelStep, pixelOffset);
            const Type B01 = kfm_load_src(f0, srcPitchT, x, y + 3, pixelStep, pixelOffset);
            const Type T02 = kfm_load_src(f0, srcPitchT, x, y + 4, pixelStep, pixelOffset);
            const Type B02 = kfm_load_src(f0, srcPitchT, x, y + 5, pixelStep, pixelOffset);
            const Type T03 = kfm_load_src(f0, srcPitchT, x, y + 6, pixelStep, pixelOffset);
            const Type B03 = kfm_load_src(f0, srcPitchT, x, y + 7, pixelStep, pixelOffset);
            const int tmp = kfm_calc_combe(T00, B00, T01, B01, T02, B02, T03, B03);
            if (parity) {
                sum0 += tmp;
            } else {
                sum2 += tmp;
            }
        }

        if (parity) {
            const Type T10 = kfm_load_src(f1, srcPitchT, x, y + 0, pixelStep, pixelOffset);
            const Type B00 = kfm_load_src(f0, srcPitchT, x, y + 1, pixelStep, pixelOffset);
            const Type T11 = kfm_load_src(f1, srcPitchT, x, y + 2, pixelStep, pixelOffset);
            const Type B01 = kfm_load_src(f0, srcPitchT, x, y + 3, pixelStep, pixelOffset);
            const Type T12 = kfm_load_src(f1, srcPitchT, x, y + 4, pixelStep, pixelOffset);
            const Type B02 = kfm_load_src(f0, srcPitchT, x, y + 5, pixelStep, pixelOffset);
            const Type T13 = kfm_load_src(f1, srcPitchT, x, y + 6, pixelStep, pixelOffset);
            const Type B03 = kfm_load_src(f0, srcPitchT, x, y + 7, pixelStep, pixelOffset);
            sum2 += kfm_calc_combe(T10, B00, T11, B01, T12, B02, T13, B03);
        } else {
            const Type T00 = kfm_load_src(f0, srcPitchT, x, y + 0, pixelStep, pixelOffset);
            const Type B10 = kfm_load_src(f1, srcPitchT, x, y + 1, pixelStep, pixelOffset);
            const Type T01 = kfm_load_src(f0, srcPitchT, x, y + 2, pixelStep, pixelOffset);
            const Type B11 = kfm_load_src(f1, srcPitchT, x, y + 3, pixelStep, pixelOffset);
            const Type T02 = kfm_load_src(f0, srcPitchT, x, y + 4, pixelStep, pixelOffset);
            const Type B12 = kfm_load_src(f1, srcPitchT, x, y + 5, pixelStep, pixelOffset);
            const Type T03 = kfm_load_src(f0, srcPitchT, x, y + 6, pixelStep, pixelOffset);
            const Type B13 = kfm_load_src(f1, srcPitchT, x, y + 7, pixelStep, pixelOffset);
            sum0 += kfm_calc_combe(T00, B10, T01, B11, T02, B12, T03, B13);
        }

        {
            const Type T00 = kfm_load_src(f0, srcPitchT, x, y + 0, pixelStep, pixelOffset);
            const Type T10 = kfm_load_src(f1, srcPitchT, x, y + 0, pixelStep, pixelOffset);
            const Type T01 = kfm_load_src(f0, srcPitchT, x, y + 2, pixelStep, pixelOffset);
            const Type T11 = kfm_load_src(f1, srcPitchT, x, y + 2, pixelStep, pixelOffset);
            const Type T02 = kfm_load_src(f0, srcPitchT, x, y + 4, pixelStep, pixelOffset);
            const Type T12 = kfm_load_src(f1, srcPitchT, x, y + 4, pixelStep, pixelOffset);
            const Type T03 = kfm_load_src(f0, srcPitchT, x, y + 6, pixelStep, pixelOffset);
            const Type T13 = kfm_load_src(f1, srcPitchT, x, y + 6, pixelStep, pixelOffset);
            sum1 += kfm_calc_diff(T00, T10, T01, T11, T02, T12, T03, T13);
        }

        {
            const Type B00 = kfm_load_src(f0, srcPitchT, x, y + 1, pixelStep, pixelOffset);
            const Type B10 = kfm_load_src(f1, srcPitchT, x, y + 1, pixelStep, pixelOffset);
            const Type B01 = kfm_load_src(f0, srcPitchT, x, y + 3, pixelStep, pixelOffset);
            const Type B11 = kfm_load_src(f1, srcPitchT, x, y + 3, pixelStep, pixelOffset);
            const Type B02 = kfm_load_src(f0, srcPitchT, x, y + 5, pixelStep, pixelOffset);
            const Type B12 = kfm_load_src(f1, srcPitchT, x, y + 5, pixelStep, pixelOffset);
            const Type B03 = kfm_load_src(f0, srcPitchT, x, y + 7, pixelStep, pixelOffset);
            const Type B13 = kfm_load_src(f1, srcPitchT, x, y + 7, pixelStep, pixelOffset);
            sum3 += kfm_calc_diff(B00, B10, B01, B11, B02, B12, B03, B13);
        }
    }

    return (uchar4)(
        kfm_clamp_u8(sum0 >> shift),
        kfm_clamp_u8(sum1 >> shift),
        kfm_clamp_u8(sum2 >> shift),
        kfm_clamp_u8(sum3 >> shift));
}

__kernel void kernel_kfm_analyze(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src0,
    const __global uchar *src1,
    const int srcPitch,
    const int width,
    const int height,
    const int parity,
    const int pixelStep,
    const int pixelOffset) {
    // width/height are the block-grid dimensions.
    const int bx = get_global_id(0);
    const int by = get_global_id(1);
    if (bx >= width - 1 || by >= height - 1) return;

    const int dstPitchT = dstPitch / (int)sizeof(uchar2);
    __global uchar2 *flag = (__global uchar2 *)dst;
    const uchar4 v = kfm_analyze_block(src0, src1, srcPitch, parity, pixelStep, pixelOffset, bx, by);

    const int dstX = bx + 1;
    const int dstY = (by + 1) * 2;
    flag[dstX + (dstY + 0) * dstPitchT] = (uchar2)(v.x, v.y);
    flag[dstX + (dstY + 1) * dstPitchT] = (uchar2)(v.z, v.w);
}

__kernel void kernel_kfm_clean_super(
    __global uchar *dst0,
    __global uchar *dst1,
    const __global uchar *prev0,
    const __global uchar *prev1,
    const __global uchar *cur0,
    const __global uchar *cur1,
    const int pitch,
    const int width,
    const int height,
    const int thresh) {
    const int plane = get_global_id(2);
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (plane >= 2 || x >= width || y >= height) return;

    __global uchar *dst = (plane == 0) ? dst0 : dst1;
    const __global uchar *prev = (plane == 0) ? prev0 : prev1;
    const __global uchar *cur = (plane == 0) ? cur0 : cur1;
    if (dst == 0 || prev == 0 || cur == 0) return;

    const int pitchT = pitch / (int)sizeof(uchar2);
    __global uchar2 *dst2 = (__global uchar2 *)dst;
    const __global uchar2 *prev2 = (const __global uchar2 *)prev;
    const __global uchar2 *cur2 = (const __global uchar2 *)cur;

    uchar2 v = cur2[x + y * pitchT];
    if (prev2[x + y * pitchT].y <= thresh && v.y <= thresh) {
        v.x = 0;
    }
    dst2[x + y * pitchT] = v;
}

__kernel void kernel_kfm_init_fmcount(
    __global FMCount *dst) {
    const int idx = get_global_id(0);
    if (idx >= 2) return;
    dst[idx].move = 0;
    dst[idx].shima = 0;
    dst[idx].lshima = 0;
}

__kernel void kernel_kfm_count_cmflags(
    __global FMCount *dst,
    const __global uchar *combe0,
    const __global uchar *combe1,
    const int pitch,
    const int width,
    const int height,
    const int parity,
    const int threshM,
    const int threshS,
    const int threshLS) {
    // The caller is expected to pass the inner region, matching the crop.
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int pitchT = pitch / (int)sizeof(uchar2);
    const __global uchar2 *c0 = (const __global uchar2 *)combe0;
    const __global uchar2 *c1 = (const __global uchar2 *)combe1;

    for (int i = 0; i < 2; ++i) {
        const uchar2 v = (i == 0 ? c0 : c1)[x + y * pitchT];
        const int dstIdx = i ^ (parity == 0);
        if (v.y >= threshM) atomic_add((volatile __global int *)&dst[dstIdx].move, 1);
        if (v.x >= threshS) atomic_add((volatile __global int *)&dst[dstIdx].shima, 1);
        if (v.x >= threshLS) atomic_add((volatile __global int *)&dst[dstIdx].lshima, 1);
    }
}

__kernel void kernel_kfm_count_cmflags_clean(
    __global FMCount *dst,
    const int dstOffset,
    const __global uchar *prevSuper,
    const __global uchar *curSuper,
    const int pitch,
    const int width,
    const int height,
    const int parity,
    const int threshM,
    const int threshS,
    const int threshLS,
    const int cleanThresh) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    dst += dstOffset;

    const int pitchT = pitch / (int)sizeof(uchar2);
    const int sx = x + 1;
    const int sy = y + 1;
    const __global uchar2 *prev = (const __global uchar2 *)prevSuper;
    const __global uchar2 *cur = (const __global uchar2 *)curSuper;

    const uchar2 prevField1 = prev[sx + (sy * 2 + 1) * pitchT];
    const uchar2 curField0 = cur[sx + (sy * 2 + 0) * pitchT];
    const uchar2 curField1 = cur[sx + (sy * 2 + 1) * pitchT];

    uchar2 v0 = curField0;
    if (prevField1.y <= cleanThresh && v0.y <= cleanThresh) {
        v0.x = 0;
    }
    uchar2 v1 = curField1;
    if (curField0.y <= cleanThresh && v1.y <= cleanThresh) {
        v1.x = 0;
    }

    const uchar2 vals[2] = { v0, v1 };
    for (int i = 0; i < 2; ++i) {
        const uchar2 v = vals[i];
        const int dstIdx = i ^ (parity == 0);
        if (v.y >= threshM) atomic_add((volatile __global int *)&dst[dstIdx].move, 1);
        if (v.x >= threshS) atomic_add((volatile __global int *)&dst[dstIdx].shima, 1);
        if (v.x >= threshLS) atomic_add((volatile __global int *)&dst[dstIdx].lshima, 1);
    }
}

__kernel void kernel_kfm_analyze_count_cmflags_clean(
    __global FMCount *dst,
    const int dstOffset,
    const __global uchar *prevSrc0,
    const __global uchar *prevSrc1,
    const __global uchar *curSrc0,
    const __global uchar *curSrc1,
    const int prevSrcPitch,
    const int curSrcPitch,
    const int width,
    const int height,
    const int prevParity,
    const int curParity,
    const int countParity,
    const int pixelStep,
    const int pixelOffset,
    const int threshM,
    const int threshS,
    const int threshLS,
    const int cleanThresh) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    dst += dstOffset;

    const uchar4 prevBlock = kfm_analyze_block(prevSrc0, prevSrc1, prevSrcPitch, prevParity, pixelStep, pixelOffset, x, y);
    const uchar4 curBlock = kfm_analyze_block(curSrc0, curSrc1, curSrcPitch, curParity, pixelStep, pixelOffset, x, y);

    const uchar2 prevField1 = (uchar2)(prevBlock.z, prevBlock.w);
    const uchar2 curField0 = (uchar2)(curBlock.x, curBlock.y);
    const uchar2 curField1 = (uchar2)(curBlock.z, curBlock.w);

    uchar2 v0 = curField0;
    if (prevField1.y <= cleanThresh && v0.y <= cleanThresh) {
        v0.x = 0;
    }
    uchar2 v1 = curField1;
    if (curField0.y <= cleanThresh && v1.y <= cleanThresh) {
        v1.x = 0;
    }

    const uchar2 vals[2] = { v0, v1 };
    for (int i = 0; i < 2; ++i) {
        const uchar2 v = vals[i];
        const int dstIdx = i ^ (countParity == 0);
        if (v.y >= threshM) atomic_add((volatile __global int *)&dst[dstIdx].move, 1);
        if (v.x >= threshS) atomic_add((volatile __global int *)&dst[dstIdx].shima, 1);
        if (v.x >= threshLS) atomic_add((volatile __global int *)&dst[dstIdx].lshima, 1);
    }
}

__kernel void kernel_kfm_count_cmflags_2planes(
    __global FMCount *dst,
    const __global uchar *combe0U,
    const __global uchar *combe1U,
    const __global uchar *combe0V,
    const __global uchar *combe1V,
    const int pitch,
    const int width,
    const int height,
    const int parity,
    const int threshM,
    const int threshS,
    const int threshLS) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int pitchT = pitch / (int)sizeof(uchar2);
    const __global uchar2 *c0U = (const __global uchar2 *)combe0U;
    const __global uchar2 *c1U = (const __global uchar2 *)combe1U;
    const __global uchar2 *c0V = (const __global uchar2 *)combe0V;
    const __global uchar2 *c1V = (const __global uchar2 *)combe1V;

    for (int i = 0; i < 2; ++i) {
        int cntM = 0;
        int cntS = 0;
        int cntLS = 0;

        uchar2 v = (i == 0 ? c0U : c1U)[x + y * pitchT];
        if (v.y >= threshM) cntM++;
        if (v.x >= threshS) cntS++;
        if (v.x >= threshLS) cntLS++;

        v = (i == 0 ? c0V : c1V)[x + y * pitchT];
        if (v.y >= threshM) cntM++;
        if (v.x >= threshS) cntS++;
        if (v.x >= threshLS) cntLS++;

        const int dstIdx = i ^ (parity == 0);
        if (cntM > 0) atomic_add((volatile __global int *)&dst[dstIdx].move, cntM);
        if (cntS > 0) atomic_add((volatile __global int *)&dst[dstIdx].shima, cntS);
        if (cntLS > 0) atomic_add((volatile __global int *)&dst[dstIdx].lshima, cntLS);
    }
}
