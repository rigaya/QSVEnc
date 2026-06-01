// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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

// Residual-combing cleanup applied after a deinterlace pass.
//
//   pb3  = vblur3(src)           [vertical 1-2-1, mirror BC]
//   pb6  = vblur5(pb3)           [vertical 1-4-6-4-1, mirror BC]
//
//   d1   = src - pb3
//   if (|d1| < thr) out = src
//   else:
//     d2  = pb3 - pb6
//     t   = d2 * sstr
//     da  = (|d1| < |t|) ? d1 : t                   [sign-preserving min]
//     add = ((d1 * t) < 0) ? (da * scl) : da
//     df  = pb3 + add
//     out = clamp(df, src - amnt, src + amnt)
//
// Type
// bit_depth

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

#define PIXEL_MAX ((1 << (bit_depth)) - 1)

// Mirror-mode vertical boundary handling matches the reference exactly:
// row 0 borrows from row 1 (not row 0); the last row borrows from
// height-2; the 5-tap kernel uses ±2 with the same +/-2 mirroring.
static inline int vmirror_p1(int y, int height) {
    if (height <= 1) return 0;
    return (y == 0) ? 1 : (y - 1);
}
static inline int vmirror_n1(int y, int height) {
    if (height <= 1) return 0;
    return (y == height - 1) ? (height - 2) : (y + 1);
}
static inline int vmirror_p2(int y, int height) {
    if (height <= 1) return 0;
    const int yy = (y < 2) ? (y + 2) : (y - 2);
    return (height < 4) ? clamp(yy, 0, height - 1) : yy;
}
static inline int vmirror_n2(int y, int height) {
    if (height <= 1) return 0;
    const int yy = (y > height - 3) ? (y - 2) : (y + 2);
    return (height < 4) ? clamp(yy, 0, height - 1) : yy;
}

static inline int read_pixel_i(const __global uchar *pSrc, int srcPitch, int x, int y) {
    return (int)(*(const __global Type *)(pSrc + y * srcPitch + x * sizeof(Type)));
}

// pb3 = vblur3(src) :  (a + 2b + c + 2) >> 2
__kernel void kernel_vinverse_vblur3(
    __global uchar *restrict pDst, const int dstPitch,
    const int dstWidth, const int dstHeight,
    const __global uchar *pSrc, const int srcPitch) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < dstWidth && iy < dstHeight) {
        const int yp = vmirror_p1(iy, dstHeight);
        const int yn = vmirror_n1(iy, dstHeight);
        const int a = read_pixel_i(pSrc, srcPitch, ix, yp);
        const int b = read_pixel_i(pSrc, srcPitch, ix, iy);
        const int c = read_pixel_i(pSrc, srcPitch, ix, yn);
        const int v = (a + (b << 1) + c + 2) >> 2;
        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(v, 0, PIXEL_MAX);
    }
}

// pb6 = vblur5(pb3) :  (a + 4(b+d) + 6c + e + 8) >> 4
__kernel void kernel_vinverse_vblur5(
    __global uchar *restrict pDst, const int dstPitch,
    const int dstWidth, const int dstHeight,
    const __global uchar *pSrc, const int srcPitch) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < dstWidth && iy < dstHeight) {
        const int ypp = vmirror_p2(iy, dstHeight);
        const int yp  = vmirror_p1(iy, dstHeight);
        const int yn  = vmirror_n1(iy, dstHeight);
        const int ynn = vmirror_n2(iy, dstHeight);
        const int a = read_pixel_i(pSrc, srcPitch, ix, ypp);
        const int b = read_pixel_i(pSrc, srcPitch, ix, yp);
        const int c = read_pixel_i(pSrc, srcPitch, ix, iy);
        const int d = read_pixel_i(pSrc, srcPitch, ix, yn);
        const int e = read_pixel_i(pSrc, srcPitch, ix, ynn);
        const int v = (a + ((b + d) << 2) + c * 6 + e + 8) >> 4;
        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(v, 0, PIXEL_MAX);
    }
}

// Helper: evaluate vblur3 ((a + 2b + c + 2) >> 2) at an arbitrary
// (already-mirrored) row index. The caller passes a row in [0, height)
// and we mirror only the ±1 taps; nesting with vmirror_p2/n2 from above
// stays in bounds because both helpers return values in [0, height-1]
// for any valid input.
static inline int vinverse_vblur3_at(const __global uchar *pSrc, int srcPitch,
                                      int ix, int row, int height) {
    const int yp = vmirror_p1(row, height);
    const int yn = vmirror_n1(row, height);
    const int a = read_pixel_i(pSrc, srcPitch, ix, yp);
    const int b = read_pixel_i(pSrc, srcPitch, ix, row);
    const int c = read_pixel_i(pSrc, srcPitch, ix, yn);
    return (a + (b << 1) + c + 2) >> 2;
}

// Fused vblur3 + vblur5 for Vinverse mode.
//
// Computes the pb3 (vblur3 of src) and pb6 (vblur5 of pb3) at the same
// (ix, iy) in a single kernel, eliminating the intermediate pb3 R+W
// in DRAM that the two-pass chain incurs. The finalize kernel reads
// pb3 at the centre row only and pb6 at the centre row only, so we
// emit one value per output buffer.
//
// Per-output-pixel work: 5 vblur3 evaluations against the source plane,
// each pulling 3 source rows. The L1/L2 caches absorb the column
// re-reads across the 5 vblur3 calls; with `local(8, 32)` work-group
// shape, 32 consecutive output rows in the same column share most of
// their input rows, keeping cache pressure low.
//
// Output is byte-identical to the two-pass chain on valid input: both
// vblur3 and vblur5 are guaranteed to land in [0, PIXEL_MAX] before
// clamp on saturated input (max + 0.5 rounded down).
__kernel void kernel_vinverse_vblur35(
    __global uchar *restrict pPb3, const int pb3Pitch,
    __global uchar *restrict pPb6, const int pb6Pitch,
    const int dstWidth, const int dstHeight,
    const __global uchar *pSrc, const int srcPitch) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= dstWidth || iy >= dstHeight) return;

    // Resolve the 4 mirrored row indices for the 5-tap vblur5; the
    // centre row is iy.
    const int r_pp = vmirror_p2(iy, dstHeight);
    const int r_p  = vmirror_p1(iy, dstHeight);
    const int r_n  = vmirror_n1(iy, dstHeight);
    const int r_nn = vmirror_n2(iy, dstHeight);

    // vblur3 at each of the 5 rows. The centre row's value is what we
    // write to pPb3.
    const int b3_pp = vinverse_vblur3_at(pSrc, srcPitch, ix, r_pp, dstHeight);
    const int b3_p  = vinverse_vblur3_at(pSrc, srcPitch, ix, r_p,  dstHeight);
    const int b3_c  = vinverse_vblur3_at(pSrc, srcPitch, ix, iy,   dstHeight);
    const int b3_n  = vinverse_vblur3_at(pSrc, srcPitch, ix, r_n,  dstHeight);
    const int b3_nn = vinverse_vblur3_at(pSrc, srcPitch, ix, r_nn, dstHeight);

    // vblur5 (1-4-6-4-1 / 16) over the five vblur3 results.
    const int v5 = (b3_pp + ((b3_p + b3_n) << 2) + b3_c * 6 + b3_nn + 8) >> 4;

    __global Type *p3 = (__global Type *)(pPb3 + iy * pb3Pitch + ix * sizeof(Type));
    __global Type *p6 = (__global Type *)(pPb6 + iy * pb6Pitch + ix * sizeof(Type));
    p3[0] = (Type)clamp(b3_c, 0, PIXEL_MAX);
    p6[0] = (Type)clamp(v5,   0, PIXEL_MAX);
}

// makediff : dst = clamp(c1 - c2 + h, 0, peak)
//   h is the half-peak shift centring the diff at peak/2 + 1; the
//   sbr_combine kernel reads the offset back out as (dst - h).
__kernel void kernel_vinverse_makediff(
    __global uchar *restrict pDst, const int dstPitch,
    const int dstWidth, const int dstHeight,
    const __global uchar *pC1, const int c1Pitch,
    const __global uchar *pC2, const int c2Pitch,
    const int h_offset) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < dstWidth && iy < dstHeight) {
        const int c1 = read_pixel_i(pC1, c1Pitch, ix, iy);
        const int c2 = read_pixel_i(pC2, c2Pitch, ix, iy);
        const int v  = clamp(c1 - c2 + h_offset, 0, PIXEL_MAX);
        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)v;
    }
}

// sbr_combine : Vinverse2's "soft bilateral reduction" final step.
//   diff = (src - vblur3(src) + h_offset)        [in pDiff]
//   blur = vblur3(diff)                          [in pBlur]
//   t  = diff - blur ;  t2 = diff - h_offset
//   if t*t2 < 0           : dst = src
//   else if |t| < |t2|    : dst = src - t
//   else                  : dst = src - (diff - h_offset)
__kernel void kernel_vinverse_sbr_combine(
    __global uchar *restrict pDst, const int dstPitch,
    const int dstWidth, const int dstHeight,
    const __global uchar *pSrc, const int srcPitch,
    const __global uchar *pDiff, const int diffPitch,
    const __global uchar *pBlur, const int blurPitch,
    const int h_offset) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < dstWidth && iy < dstHeight) {
        const int s    = read_pixel_i(pSrc,  srcPitch,  ix, iy);
        const int diff = read_pixel_i(pDiff, diffPitch, ix, iy);
        const int blur = read_pixel_i(pBlur, blurPitch, ix, iy);
        const int t   = diff - blur;
        const int t2  = diff - h_offset;

        int v;
        if ((t < 0 && t2 > 0) || (t > 0 && t2 < 0)) {
            v = s;
        } else if (abs(t) < abs(t2)) {
            v = s - t;
        } else {
            v = s - (diff - h_offset);
        }
        v = clamp(v, 0, PIXEL_MAX);
        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)v;
    }
}

// Sign-aware combine.
// thr_hbd / amnt_hbd are already scaled to the working bit depth host-side.
// amnt_hbd <= 0 disables the per-pixel cap (matches the source's "amnt = -1
// means peak" sentinel; the caller passes peak in that case so the clamp
// becomes a no-op against the [0, peak] range).
__kernel void kernel_vinverse_finalize(
    __global uchar *restrict pDst, const int dstPitch,
    const int dstWidth, const int dstHeight,
    const __global uchar *pSrc, const int srcPitch,
    const __global uchar *pPb3, const int pb3Pitch,
    const __global uchar *pPb6, const int pb6Pitch,
    const float sstr, const float scl,
    const int thr_hbd, const int amnt_hbd) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < dstWidth && iy < dstHeight) {
        const int s   = read_pixel_i(pSrc, srcPitch, ix, iy);
        const int pb3 = read_pixel_i(pPb3, pb3Pitch, ix, iy);
        const int d1  = s - pb3;

        int result;
        if (thr_hbd > 0 && abs(d1) < thr_hbd) {
            result = s;
        } else {
            const int pb6 = read_pixel_i(pPb6, pb6Pitch, ix, iy);
            const float d1f = (float)d1;
            const float t   = (float)(pb3 - pb6) * sstr;
            const float da  = (fabs(d1f) < fabs(t)) ? d1f : t;
            const float add = ((d1f * t) < 0.0f) ? (da * scl) : da;
            const int   df  = pb3 + (int)add;
            const int   lo  = s - amnt_hbd;
            const int   hi  = s + amnt_hbd;
            result = clamp(df, lo, hi);
        }

        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(result, 0, PIXEL_MAX);
    }
}
