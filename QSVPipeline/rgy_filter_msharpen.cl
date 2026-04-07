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

// Helper: read a pixel value from raw buffer with boundary clamping
float read_pixel_f(const __global uchar *pSrc, int srcPitch, int x, int y, int width, int height) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    Type val = *(const __global Type *)(pSrc + y * srcPitch + x * sizeof(Type));
    return (float)val * (1.0f / (float)PIXEL_MAX);
}

// Kernel 1: 3x3 Box Blur
__kernel void kernel_msharpen_blur(
    __global uchar *restrict pDst, const int dstPitch,
    const int dstWidth, const int dstHeight,
    const __global uchar *pSrc, const int srcPitch) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < dstWidth && iy < dstHeight) {
        float sum = 0.0f;
        sum += read_pixel_f(pSrc, srcPitch, ix - 1, iy - 1, dstWidth, dstHeight);
        sum += read_pixel_f(pSrc, srcPitch, ix,     iy - 1, dstWidth, dstHeight);
        sum += read_pixel_f(pSrc, srcPitch, ix + 1, iy - 1, dstWidth, dstHeight);
        sum += read_pixel_f(pSrc, srcPitch, ix - 1, iy,     dstWidth, dstHeight);
        sum += read_pixel_f(pSrc, srcPitch, ix,     iy,     dstWidth, dstHeight);
        sum += read_pixel_f(pSrc, srcPitch, ix + 1, iy,     dstWidth, dstHeight);
        sum += read_pixel_f(pSrc, srcPitch, ix - 1, iy + 1, dstWidth, dstHeight);
        sum += read_pixel_f(pSrc, srcPitch, ix,     iy + 1, dstWidth, dstHeight);
        sum += read_pixel_f(pSrc, srcPitch, ix + 1, iy + 1, dstWidth, dstHeight);

        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(clamp(sum * (1.0f / 9.0f), 0.0f, 1.0f - RGY_FLT_EPS) * PIXEL_MAX);
    }
}

// Kernel 2: Edge-Selective Sharpening
__kernel void kernel_msharpen_sharpen(
    __global uchar *restrict pDst, const int dstPitch,
    const int dstWidth, const int dstHeight,
    const __global uchar *pSrc, const int srcPitch,
    const __global uchar *pBlur, const int blurPitch,
    const float strength, const float threshold,
    const int highq, const int mask) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < dstWidth && iy < dstHeight) {
        float src = read_pixel_f(pSrc, srcPitch, ix, iy, dstWidth, dstHeight);

        float b_cc = read_pixel_f(pBlur, blurPitch, ix,     iy,     dstWidth, dstHeight);
        float b_br = read_pixel_f(pBlur, blurPitch, ix + 1, iy + 1, dstWidth, dstHeight);
        float b_bl = read_pixel_f(pBlur, blurPitch, ix - 1, iy + 1, dstWidth, dstHeight);

        // Diagonal edge detection
        int edge = (fabs(b_cc - b_br) >= threshold) || (fabs(b_cc - b_bl) >= threshold);

        // High quality: add vertical and horizontal
        if (highq) {
            float b_bc = read_pixel_f(pBlur, blurPitch, ix,     iy + 1, dstWidth, dstHeight);
            float b_cr = read_pixel_f(pBlur, blurPitch, ix + 1, iy,     dstWidth, dstHeight);
            edge = edge || (fabs(b_cc - b_bc) >= threshold) || (fabs(b_cc - b_cr) >= threshold);
        }

        float result;
        if (mask) {
            result = edge ? 1.0f : 0.0f;
        } else if (edge) {
            float sharpened = 4.0f * src - 3.0f * b_cc;
            sharpened = clamp(sharpened, 0.0f, 1.0f);
            result = strength * sharpened + (1.0f - strength) * src;
        } else {
            result = src;
        }

        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(clamp(result, 0.0f, 1.0f - RGY_FLT_EPS) * PIXEL_MAX);
    }
}
