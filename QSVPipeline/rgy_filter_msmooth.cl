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

#define RGY_FLT_EPS (1e-6f)
#define PIXEL_MAX ((1 << (bit_depth)) - 1)

// Helper: read a pixel value from raw buffer with boundary clamping, returns float [0,1]
float read_pixel_f(const __global uchar *pSrc, int srcPitch, int x, int y, int width, int height) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    Type val = *(const __global Type *)(pSrc + y * srcPitch + x * sizeof(Type));
    return (float)val * (1.0f / (float)PIXEL_MAX);
}

// Helper: read a pixel as int from raw buffer with boundary clamping
int read_pixel_i(const __global uchar *pSrc, int srcPitch, int x, int y, int width, int height) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    return (int)(*(const __global Type *)(pSrc + y * srcPitch + x * sizeof(Type)));
}

// Fused 3x3 box blur + edge-mask threshold.
//
// Replaces the prior two-kernel chain (kernel_msmooth_blur ->
// kernel_msmooth_edge_mask) and eliminates the intermediate blur frame
// buffer. The edge-mask step needs blur values at five offsets around
// the centre: (0,0), (+1,+1), (-1,+1), and (highq) (0,+1), (+1,0).
// Each of those blurs is the average of a 3x3 source neighbourhood, so
// the union of source samples required is a 5-wide (ix-2..ix+2) by
// 4-tall (iy-1..iy+2) window -- the same shape as the fused msharpen
// kernel. We load that window into registers once and form every
// needed blur value from it.
//
// All arithmetic stays FP32; the mask write is binary (0 or PIXEL_MAX),
// matching the unfused chain byte-for-byte.
__kernel void kernel_msmooth_blur_mask(
    __global uchar *restrict pDst, const int dstPitch,
    const int dstWidth, const int dstHeight,
    const __global uchar *pSrc, const int srcPitch,
    const float threshold, const int highq) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= dstWidth || iy >= dstHeight) return;

    // Forced-edge border (matches the original edge_mask boundary case
    // at lines that previously emitted edge=1 when ix/iy hit the frame
    // edge). Skip the blur work entirely in this case.
    if (ix == 0 || ix >= dstWidth - 1 || iy == 0 || iy >= dstHeight - 1) {
        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)PIXEL_MAX;
        return;
    }

    // 5x4 source neighbourhood: s[dx+2][dy+1] = pSrc(ix+dx, iy+dy).
    float s[5][4];
    for (int dy = -1; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            s[dx + 2][dy + 1] = read_pixel_f(pSrc, srcPitch, ix + dx, iy + dy, dstWidth, dstHeight);
        }
    }

    // 3x3 box-blur means at the centre and its 4 neighbours.
    // b_cc centred at (ix, iy): cols {-1, 0, +1} x rows {-1, 0, +1}
    const float b_cc = (
          s[1][0] + s[2][0] + s[3][0]
        + s[1][1] + s[2][1] + s[3][1]
        + s[1][2] + s[2][2] + s[3][2]) * (1.0f / 9.0f);
    // b_br centred at (ix+1, iy+1): cols { 0, +1, +2} x rows {0, +1, +2}
    const float b_br = (
          s[2][1] + s[3][1] + s[4][1]
        + s[2][2] + s[3][2] + s[4][2]
        + s[2][3] + s[3][3] + s[4][3]) * (1.0f / 9.0f);
    // b_bl centred at (ix-1, iy+1): cols {-2, -1,  0} x rows {0, +1, +2}
    const float b_bl = (
          s[0][1] + s[1][1] + s[2][1]
        + s[0][2] + s[1][2] + s[2][2]
        + s[0][3] + s[1][3] + s[2][3]) * (1.0f / 9.0f);

    int edge = (fabs(b_cc - b_br) >= threshold) || (fabs(b_cc - b_bl) >= threshold);

    if (highq) {
        // b_bc centred at (ix, iy+1): cols {-1, 0, +1} x rows {0, +1, +2}
        const float b_bc = (
              s[1][1] + s[2][1] + s[3][1]
            + s[1][2] + s[2][2] + s[3][2]
            + s[1][3] + s[2][3] + s[3][3]) * (1.0f / 9.0f);
        // b_cr centred at (ix+1, iy): cols { 0, +1, +2} x rows {-1, 0, +1}
        const float b_cr = (
              s[2][0] + s[3][0] + s[4][0]
            + s[2][1] + s[3][1] + s[4][1]
            + s[2][2] + s[3][2] + s[4][2]) * (1.0f / 9.0f);
        edge = edge || (fabs(b_cc - b_bc) >= threshold) || (fabs(b_cc - b_cr) >= threshold);
    }

    __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = edge ? (Type)PIXEL_MAX : (Type)0;
}

// Kernel 3: Masked Smoothing Pass (single iteration)
__kernel void kernel_msmooth_smooth(
    __global uchar *restrict pDst, const int dstPitch,
    const __global uchar *pSrc, const int srcPitch,
    const __global uchar *pMask, const int maskPitch,
    const int width, const int height) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < width && iy < height) {
        const int m_cc = read_pixel_i(pMask, maskPitch, ix, iy, width, height);

        int result;
        if (m_cc != 0) {
            result = read_pixel_i(pSrc, srcPitch, ix, iy, width, height);
        } else {
            int center = read_pixel_i(pSrc, srcPitch, ix, iy, width, height);
            int sum = center;
            int count = 1;

            if (iy > 0 && read_pixel_i(pMask, maskPitch, ix, iy - 1, width, height) == 0) {
                sum += read_pixel_i(pSrc, srcPitch, ix, iy - 1, width, height);
                count++;
            }
            if (iy < height - 1 && read_pixel_i(pMask, maskPitch, ix, iy + 1, width, height) == 0) {
                sum += read_pixel_i(pSrc, srcPitch, ix, iy + 1, width, height);
                count++;
            }
            if (ix > 0 && read_pixel_i(pMask, maskPitch, ix - 1, iy, width, height) == 0) {
                sum += read_pixel_i(pSrc, srcPitch, ix - 1, iy, width, height);
                count++;
            }
            if (ix < width - 1 && read_pixel_i(pMask, maskPitch, ix + 1, iy, width, height) == 0) {
                sum += read_pixel_i(pSrc, srcPitch, ix + 1, iy, width, height);
                count++;
            }
            result = (sum + count / 2) / count;
        }

        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(result, 0, PIXEL_MAX);
    }
}
