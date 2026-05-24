// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
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
//
// AURORA deflicker algorithm. Reference: van Roosmalen 1999 PhD thesis,
// "Restoration of archived film and video", section on inter-frame
// brightness restoration.
//
// Two kernels are exposed:
//   deflicker_reduce -- workgroup-level parallel reduction. Each work-
//     group computes (sum, sum_of_squares) over its slice of the plane
//     using long accumulators (needed to avoid overflow at 16-bit
//     depths and large frame sizes) and writes the result to a pair of
//     global buffers indexed by workgroup id. The host sums these
//     partial results into the final (mean, sigma).
//
//   deflicker_apply  -- per-pixel correction. For luma:
//       out = clamp(strength * (in * mult + add) + (1-strength) * in, 0, max_val)
//     For chroma (is_chroma != 0) the centred-multiplicative form is
//     used to preserve neutral grey:
//       out = clamp(strength * (mult * (in - mid) + mid) + (1-strength) * in, 0, max_val)
//
// Build-time defines from rgy_filter_deflicker.cpp:
//   Type        : uchar (8-bit) or ushort (>8-bit)
//   bit_depth   : source bit depth (8, 10, 12, 14, 16)
//   max_val     : (1 << bit_depth) - 1
//   DEFLICKER_REDUCE_X / _Y : workgroup dimensions for the stats kernel

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

#ifndef DEFLICKER_REDUCE_X
#define DEFLICKER_REDUCE_X 32
#endif
#ifndef DEFLICKER_REDUCE_Y
#define DEFLICKER_REDUCE_Y 8
#endif

// Workgroup-level parallel reduction. Each thread reads up to one pixel
// (range-checked) and accumulates into private long sums; tree-reduces
// via local memory; first thread of the workgroup writes the workgroup
// totals to (pSum[wg_id], pSumSq[wg_id]).
//
// pSum and pSumSq are long arrays sized num_workgroups by the host.
__kernel
__attribute__((reqd_work_group_size(DEFLICKER_REDUCE_X, DEFLICKER_REDUCE_Y, 1)))
void deflicker_reduce(
    const __global uchar *pSrc, const int srcPitch,
    const int width, const int height,
    __global long *pSum,
    __global long *pSumSq
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int wgx = get_group_id(0);
    const int wgy = get_group_id(1);
    const int num_wg_x = get_num_groups(0);

    long val = 0;
    long val_sq = 0;
    if (ix < width && iy < height) {
        const int p = (int)(*(const __global Type *)(pSrc + iy * srcPitch + ix * sizeof(Type)));
        val    = (long)p;
        val_sq = (long)p * (long)p;
    }

    __local long sSum  [DEFLICKER_REDUCE_X * DEFLICKER_REDUCE_Y];
    __local long sSumSq[DEFLICKER_REDUCE_X * DEFLICKER_REDUCE_Y];
    const int lidx = ly * DEFLICKER_REDUCE_X + lx;
    sSum  [lidx] = val;
    sSumSq[lidx] = val_sq;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction. WG size is DEFLICKER_REDUCE_X*DEFLICKER_REDUCE_Y;
    // for the default 32x8 that's 256.
    const int wgsize = DEFLICKER_REDUCE_X * DEFLICKER_REDUCE_Y;
    for (int stride = wgsize / 2; stride > 0; stride >>= 1) {
        if (lidx < stride) {
            sSum  [lidx] += sSum  [lidx + stride];
            sSumSq[lidx] += sSumSq[lidx + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lidx == 0) {
        const int wg_index = wgy * num_wg_x + wgx;
        pSum  [wg_index] = sSum  [0];
        pSumSq[wg_index] = sSumSq[0];
    }
}

// Per-pixel correction. is_chroma flag selects the centred form for U/V.
// strength: blend between corrected and original.
//   luma: out = clamp(strength * (in * mult + add) + (1 - strength) * in, 0, max_val)
//   chroma: out = clamp(strength * (mult * (in - mid) + mid) + (1 - strength) * in, 0, max_val)
// mult, add, strength are floats; mid = 1 << (bit_depth - 1) for chroma.
__kernel void deflicker_apply(
    const __global uchar *pSrc, const int srcPitch,
    __global       uchar *pDst, const int dstPitch,
    const int width, const int height,
    const float mult, const float add,
    const float strength,
    const int is_chroma
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const float src_f = (float)(*(const __global Type *)(pSrc + iy * srcPitch + ix * sizeof(Type)));
    float corrected;
    if (is_chroma != 0) {
        const float mid = (float)(1 << (bit_depth - 1));
        corrected = mult * (src_f - mid) + mid;
    } else {
        corrected = mult * src_f + add;
    }
    float result = strength * corrected + (1.0f - strength) * src_f;
    if (result < 0.0f)            result = 0.0f;
    if (result > (float)max_val)  result = (float)max_val;

    __global Type *dst = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    dst[0] = (Type)(result + 0.5f);
}
