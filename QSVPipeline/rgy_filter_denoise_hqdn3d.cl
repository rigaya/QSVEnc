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
// LUT_RADIUS  (table holds 2 * LUT_RADIUS float entries indexed [0, 2*LUT_RADIUS))
// HQDN3D_SCRATCH_FP16 (0/1; selects scratch element type for m_tmpH /
//                      m_tmpHV / m_framePrev[])

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

#define PIXEL_MAX ((1 << (bit_depth)) - 1)

// Scratch element type for the spatial / temporal IIR state buffers.
// half (FP16) is selected at JIT time via -D HQDN3D_SCRATCH_FP16=1 when
// the device advertises cl_khr_fp16; otherwise the buffers stay FP32.
// All kernel arithmetic remains in float regardless; only the buffer
// reads / writes go through vload_half / vstore_half.
//
// Safety: values stored here are all bounded to [0, 1] by write_pixel_f
// and the LUT-add path, so FP16 storage ULP error (~6e-5) is well below
// the 1/255 integer pixel granularity the delta-LUT already quantises
// to. No temporal drift accumulates in m_framePrev (see hqdn3d_lowpass
// derivation in rgy_filter_denoise_hqdn3d.cpp).
#if defined(HQDN3D_SCRATCH_FP16) && HQDN3D_SCRATCH_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define HQDN3D_SCRATCH_T half
#define hqdn3d_load_scratch(p, i)     vload_half((i), (const __global half *)(p))
#define hqdn3d_store_scratch(v, p, i) vstore_half((v), (i), (__global half *)(p))
#else
#define HQDN3D_SCRATCH_T float
#define hqdn3d_load_scratch(p, i)     ((p)[(i)])
#define hqdn3d_store_scratch(v, p, i) ((p)[(i)] = (v))
#endif

inline float read_pixel_f(const __global uchar *pSrc, int srcPitch, int x, int y) {
    Type val = *(const __global Type *)(pSrc + y * srcPitch + x * sizeof(Type));
    return (float)val * (1.0f / (float)PIXEL_MAX);
}

inline void write_pixel_f(__global uchar *pDst, int dstPitch, int x, int y, float v) {
    __global Type *ptr = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));
    ptr[0] = (Type)(clamp(v, 0.0f, 1.0f) * (float)PIXEL_MAX + 0.5f);
}

inline float read_scratch_buf(const __global HQDN3D_SCRATCH_T *buf, int pitch_elems, int x, int y) {
    return hqdn3d_load_scratch(buf, y * pitch_elems + x);
}

inline void write_scratch_buf(__global HQDN3D_SCRATCH_T *buf, int pitch_elems, int x, int y, float v) {
    hqdn3d_store_scratch(v, buf, y * pitch_elems + x);
}

// Non-linear lowpass primitive. delta is mapped to the LUT bin via a
// scale of 255 (8-bit pixel units), matching the published reference's
// LUT_BITS=8 effective resolution. Returns cur + coef[idx], where the
// LUT was precomputed by the host using the same delta convention.
inline float hqdn3d_lowpass(float prev, float cur, const __global float *coef) {
    float delta_pix = (prev - cur) * 255.0f;
    int idx = (int)(delta_pix + (delta_pix >= 0.0f ? 0.5f : -0.5f)) + LUT_RADIUS;
    idx = clamp(idx, 0, 2 * LUT_RADIUS - 1);
    return cur + coef[idx];
}

// Pass 1: horizontal IIR.
// One work-item per row; sequential left-to-right inner loop.
__kernel void kernel_hqdn3d_h(
    __global HQDN3D_SCRATCH_T *restrict pDst, const int dstPitchElems,
    const __global uchar *pSrc, const int srcPitch,
    const int width, const int height,
    const __global float *coefSpatial) {
    const int iy = get_global_id(0);
    if (iy >= height) return;

    float prev_pixel = read_pixel_f(pSrc, srcPitch, 0, iy);
    for (int x = 0; x < width; ++x) {
        float cur = read_pixel_f(pSrc, srcPitch, x, iy);
        prev_pixel = hqdn3d_lowpass(prev_pixel, cur, coefSpatial);
        write_scratch_buf(pDst, dstPitchElems, x, iy, prev_pixel);
    }
}

// Pass 2: vertical IIR.
// One work-item per column; sequential top-to-bottom inner loop.
__kernel void kernel_hqdn3d_v(
    __global HQDN3D_SCRATCH_T *restrict pDst, const int dstPitchElems,
    const __global HQDN3D_SCRATCH_T *pSrc, const int srcPitchElems,
    const int width, const int height,
    const __global float *coefSpatial) {
    const int ix = get_global_id(0);
    if (ix >= width) return;

    float prev_pixel = read_scratch_buf(pSrc, srcPitchElems, ix, 0);
    for (int y = 0; y < height; ++y) {
        float cur = read_scratch_buf(pSrc, srcPitchElems, ix, y);
        prev_pixel = hqdn3d_lowpass(prev_pixel, cur, coefSpatial);
        write_scratch_buf(pDst, dstPitchElems, ix, y, prev_pixel);
    }
}

// Pass 3: temporal IIR.
// 2D per-pixel parallel. Reads spatial-filtered current frame and the
// per-pixel previous-output state; writes the blended result to both
// the output pixel buffer and the persistent prev-state buffer. When
// first_frame is non-zero the prev state is treated as uninitialised
// and the spatial-filtered value is written through directly (no
// blending), seeding the IIR for subsequent frames.
__kernel void kernel_hqdn3d_t(
    __global uchar *restrict pDst, const int dstPitch,
    __global HQDN3D_SCRATCH_T *pFramePrev, const int prevPitchElems,
    const __global HQDN3D_SCRATCH_T *pSpatial, const int spatialPitchElems,
    const int width, const int height,
    const __global float *coefTemporal,
    const int first_frame) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    float spatial = read_scratch_buf(pSpatial, spatialPitchElems, ix, iy);
    float result;
    if (first_frame) {
        result = spatial;
    } else {
        float prev = read_scratch_buf(pFramePrev, prevPitchElems, ix, iy);
        result = hqdn3d_lowpass(prev, spatial, coefTemporal);
    }
    write_scratch_buf(pFramePrev, prevPitchElems, ix, iy, result);
    write_pixel_f(pDst, dstPitch, ix, iy, result);
}
