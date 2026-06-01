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

// Type
// bit_depth

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

#define PIXEL_MAX ((1 << (bit_depth)) - 1)

// Horizontal pass: solves the per-row inverse system. One work-item
// per row of the source image. The inner loop is sequential and
// runs three sub-loops per row:
//   1. A' b (matrix-vector product against the sparse upscale weights)
//   2. forward substitution    LD y = A' b
//   3. back substitution       L' x = y
// The per-row state lives entirely in this work-item's slice of the
// global pDst buffer (one float row at offset iy * dstPitchFloats);
// the read-after-write within a single work-item is well-defined.
__kernel void kernel_descale_h(
    __global float *restrict pDst, const int dstPitchFloats,
    const __global uchar *pSrc, const int srcPitch,
    const int src_h,
    const int dst_w,
    const int c_band,
    const int weights_columns,
    const __global float *weights,
    const __global int   *left_idx,
    const __global int   *right_idx,
    const __global float *lower,
    const __global float *upper,
    const __global float *diagonal) {
    const int iy = get_global_id(0);
    if (iy >= src_h) return;

    __global float *dstRow = pDst + iy * dstPitchFloats;
    const __global Type *srcRow = (const __global Type *)(pSrc + iy * srcPitch);

    // Forward sweep: A' b followed immediately by LD y = A' b.
    for (int j = 0; j < dst_w; ++j) {
        const int lj = left_idx[j];
        const int rj = right_idx[j];
        float sum = 0.0f;
        for (int k = lj; k < rj; ++k) {
            const float src_f = (float)srcRow[k] * (1.0f / (float)PIXEL_MAX);
            sum += weights[j * weights_columns + (k - lj)] * src_f;
        }
        int start = j - c_band;
        if (start < 0) start = 0;
        for (int k = start; k < j; ++k) {
            sum -= lower[(k - j + c_band) * dst_w + j] * dstRow[k];
        }
        dstRow[j] = sum * diagonal[j];
    }

    // Back sweep: L' x = y.
    for (int j = dst_w - 2; j >= 0; --j) {
        int end = j + c_band;
        if (end > dst_w - 1) end = dst_w - 1;
        float sum = 0.0f;
        for (int k = end; k > j; --k) {
            sum += upper[(k - j - 1) * dst_w + j] * dstRow[k];
        }
        dstRow[j] -= sum;
    }
}

// Vertical pass: solves the per-column inverse system on the H-pass
// float intermediate. One work-item per column. Internally writes
// the column's full dst_h-row settled values to the float scratch
// buffer pVScratch, then converts to Type at the end to populate
// the final output buffer.
//
// pVScratch is sized dst_w * dst_h floats. Each work-item touches
// only its own column (column ix), so the RAW reads from this
// buffer during the inverse sweeps are within-work-item and need
// no synchronization.
__kernel void kernel_descale_v(
    __global uchar *restrict pDst, const int dstPitch,
    __global float *pVScratch, const int scratchPitchFloats,
    const __global float *pSrc, const int srcPitchFloats,
    const int src_h,
    const int dst_w,
    const int dst_h,
    const int c_band,
    const int weights_columns,
    const __global float *weights,
    const __global int   *left_idx,
    const __global int   *right_idx,
    const __global float *lower,
    const __global float *upper,
    const __global float *diagonal,
    // 1 = quantise the settled float column to the output integer
    //     bit-depth and write it to pDst (used by the live filter path
    //     where pDst feeds the next filter stage).
    // 0 = skip the integer write entirely (used by the auto-detect
    //     probe path which only consumes pVScratch downstream and
    //     never reads pDst -- avoids ~110 MPix of wasted memory
    //     traffic per probe on typical 1080p / 150-candidate / 8-frame
    //     workloads).
    // pDst is still required to be a valid cl_mem because OpenCL
    // doesn't permit null buffer args; the probe passes a small
    // sentinel that the kernel never touches when writeIntegerOutput=0.
    const int writeIntegerOutput) {
    const int ix = get_global_id(0);
    if (ix >= dst_w) return;

    // Forward sweep on the column.
    for (int j = 0; j < dst_h; ++j) {
        const int lj = left_idx[j];
        const int rj = right_idx[j];
        float sum = 0.0f;
        for (int k = lj; k < rj; ++k) {
            sum += weights[j * weights_columns + (k - lj)] * pSrc[k * srcPitchFloats + ix];
        }
        int start = j - c_band;
        if (start < 0) start = 0;
        for (int k = start; k < j; ++k) {
            sum -= lower[(k - j + c_band) * dst_h + j] * pVScratch[k * scratchPitchFloats + ix];
        }
        pVScratch[j * scratchPitchFloats + ix] = sum * diagonal[j];
    }

    // Back sweep.
    for (int j = dst_h - 2; j >= 0; --j) {
        int end = j + c_band;
        if (end > dst_h - 1) end = dst_h - 1;
        float sum = 0.0f;
        for (int k = end; k > j; --k) {
            sum += upper[(k - j - 1) * dst_h + j] * pVScratch[k * scratchPitchFloats + ix];
        }
        pVScratch[j * scratchPitchFloats + ix] -= sum;
    }

    // Convert settled column from float to the output bit depth.
    // Skipped on the probe path (writeIntegerOutput == 0) because
    // pDst is never read by the downstream re-upscale kernels.
    if (writeIntegerOutput != 0) {
        for (int j = 0; j < dst_h; ++j) {
            float v = pVScratch[j * scratchPitchFloats + ix];
            v = clamp(v, 0.0f, 1.0f);
            __global Type *outPtr = (__global Type *)(pDst + j * dstPitch);
            outPtr[ix] = (Type)(v * (float)PIXEL_MAX + 0.5f);
        }
    }
}

// --- auto-detect probe kernels -------------------------------------------
//
// Forward re-upscale (`y = A x` where A is the upscale weights matrix
// built during LDLT init). One output pixel per work-item, fully
// parallel - no sequential dependency. Used by the probe path to
// reconstruct an upscaled image from a descaled candidate, against
// which the original input is compared.

// Horizontal re-upscale. pSrc is the descaled candidate (dst_w x src_h
// floats). pDst is the reconstruction at intermediate width = src_w
// (the original input width), still src_h tall. weights / left_idx /
// right_idx describe how each OUTPUT (source-width) pixel is composed
// from the DESCALED (dst_w-width) pixels - i.e., the same A matrix
// that produced the input, in its original (non-transposed) form.
__kernel void kernel_rescale_h(
    __global float *restrict pDst, const int dstPitchFloats,
    const __global float *pSrc, const int srcPitchFloats,
    const int src_w_recon, const int src_h,
    const int dst_w_descaled,
    const int weights_columns,
    const __global float *weights,
    const __global int   *left_idx,
    const __global int   *right_idx) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= src_w_recon || iy >= src_h) return;
    const int lj = left_idx[ix];
    const int rj = right_idx[ix];
    float sum = 0.0f;
    const __global float *srcRow = pSrc + iy * srcPitchFloats;
    for (int k = lj; k < rj; ++k) {
        sum += weights[ix * weights_columns + (k - lj)] * srcRow[k];
    }
    pDst[iy * dstPitchFloats + ix] = sum;
}

// Vertical re-upscale. Same as above but for the V direction.
__kernel void kernel_rescale_v(
    __global float *restrict pDst, const int dstPitchFloats,
    const __global float *pSrc, const int srcPitchFloats,
    const int src_w, const int src_h_recon,
    const int dst_h_descaled,
    const int weights_columns,
    const __global float *weights,
    const __global int   *left_idx,
    const __global int   *right_idx) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= src_w || iy >= src_h_recon) return;
    const int lj = left_idx[iy];
    const int rj = right_idx[iy];
    float sum = 0.0f;
    for (int k = lj; k < rj; ++k) {
        sum += weights[iy * weights_columns + (k - lj)] * pSrc[k * srcPitchFloats + ix];
    }
    pDst[iy * dstPitchFloats + ix] = sum;
}

// Per-pixel Sobel gradient magnitude, computed once per uploaded
// probe frame and reused by every candidate's MSE evaluation against
// that frame. The probe-time MSE is then weighted by this magnitude
// (with a small constant floor) so high-detail regions dominate the
// score; this is what gives cross-resolution discrimination on
// otherwise flat content where the unweighted reconstruction residual
// collapses to noise.
__kernel void kernel_compute_edge_weight(
    const __global uchar *pSrc, const int srcPitch,
    __global float *restrict pWeights, const int weightsPitchFloats,
    const int src_w, const int src_h) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= src_w || iy >= src_h) return;
    if (ix == 0 || ix == src_w - 1 || iy == 0 || iy == src_h - 1) {
        pWeights[iy * weightsPitchFloats + ix] = 0.0f;
        return;
    }
    const __global Type *row_m = (const __global Type *)(pSrc + (iy - 1) * srcPitch);
    const __global Type *row_0 = (const __global Type *)(pSrc + (iy    ) * srcPitch);
    const __global Type *row_p = (const __global Type *)(pSrc + (iy + 1) * srcPitch);
    const float inv_max = 1.0f / (float)PIXEL_MAX;
    // Cast to int before subtraction so uchar wraparound doesn't bite.
    const float gx =
          (float)((int)row_m[ix + 1] - (int)row_m[ix - 1]) * inv_max
        + 2.0f * (float)((int)row_0[ix + 1] - (int)row_0[ix - 1]) * inv_max
        + (float)((int)row_p[ix + 1] - (int)row_p[ix - 1]) * inv_max;
    const float gy =
          (float)((int)row_p[ix - 1] - (int)row_m[ix - 1]) * inv_max
        + 2.0f * (float)((int)row_p[ix    ] - (int)row_m[ix    ]) * inv_max
        + (float)((int)row_p[ix + 1] - (int)row_m[ix + 1]) * inv_max;
    pWeights[iy * weightsPitchFloats + ix] = native_sqrt(gx * gx + gy * gy);
}

// Per-row Huber loss summed across the row using Kahan compensated
// summation in float. cl_khr_fp64 isn't available on the Arc A770
// OpenCL driver, so native double accumulation isn't available; the
// compensation accumulator is marked volatile so the optimiser cannot
// algebraically cancel the round-off term. Loss at each pixel is
// `weight * huber(orig - recon)` where weight comes from
// kernel_compute_edge_weight (plus a small constant floor so flat
// regions still contribute) and the Huber threshold is 4/255 ~ 0.0157
// in the normalized luma space. Huber clips outlier contributions to
// linear, which suppresses the per-frame noise floor on compressed
// sources while still penalising real reconstruction error.
__kernel void kernel_descale_mse(
    __global float *restrict pRowSums,
    const __global uchar *pOrig, const int origPitch,
    const __global float *pRecon, const int reconPitchFloats,
    const __global float *pWeights, const int weightsPitchFloats,
    const int width, const int height) {
    const int iy = get_global_id(0);
    if (iy >= height) return;
    const __global Type *origRow = (const __global Type *)(pOrig + iy * origPitch);
    const __global float *reconRow = pRecon + iy * reconPitchFloats;
    const __global float *wRow    = pWeights + iy * weightsPitchFloats;
    const float inv_max = 1.0f / (float)PIXEL_MAX;
    const float delta = 4.0f / 255.0f;
    const float delta_sq = delta * delta;
    float sum = 0.0f;
    volatile float comp = 0.0f;
    for (int x = 0; x < width; ++x) {
        const float o = (float)origRow[x] * inv_max;
        const float r = reconRow[x];
        const float d = o - r;
        const float abs_d = fabs(d);
        // Huber loss: quadratic below delta, linear above.
        const float loss = (abs_d < delta) ? (d * d)
                                            : (2.0f * delta * abs_d - delta_sq);
        // Edge-weighted contribution. The +0.1f floor keeps flat
        // regions weighted nonzero so they don't disappear from the
        // score entirely on completely smooth content.
        const float w = wRow[x] + 0.1f;
        const float term = w * loss;
        const float y = term - comp;
        const volatile float t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    pRowSums[iy] = sum;
}
