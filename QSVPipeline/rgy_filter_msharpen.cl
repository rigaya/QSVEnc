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

// Sigmoid soft-mask helper.
// Returns a smooth weight in [0, 1] for the given max gradient. With
// slope <= 0 the caller falls back to the binary gate; this helper is
// only invoked when slope > 0.
//
// The math is the standard logistic: w = 1 / (1 + exp(-(g - thr) * slope)).
// The exponent argument is clamped to +/- 32 because exp() loses precision
// (and on some implementations diverges) well before that, and any pixel
// past +/-32 maps to a saturated 0 or 1 anyway.
inline float msharpen_sigmoid_weight(float g, float threshold, float slope) {
    float arg = (g - threshold) * slope;
    arg = clamp(arg, -32.0f, 32.0f);
    return 1.0f / (1.0f + exp(-arg));
}

// Kernel 2: Edge-Selective Sharpening
__kernel void kernel_msharpen_sharpen(
    __global uchar *restrict pDst, const int dstPitch,
    const int dstWidth, const int dstHeight,
    const __global uchar *pSrc, const int srcPitch,
    const __global uchar *pBlur, const int blurPitch,
    const float strength, const float threshold,
    const float slope, const float luma_limit_norm,
    const int highq, const int mask) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < dstWidth && iy < dstHeight) {
        float src = read_pixel_f(pSrc, srcPitch, ix, iy, dstWidth, dstHeight);

        float b_cc = read_pixel_f(pBlur, blurPitch, ix,     iy,     dstWidth, dstHeight);
        float b_br = read_pixel_f(pBlur, blurPitch, ix + 1, iy + 1, dstWidth, dstHeight);
        float b_bl = read_pixel_f(pBlur, blurPitch, ix - 1, iy + 1, dstWidth, dstHeight);

        // Diagonal gradient magnitudes.
        float g_br = fabs(b_cc - b_br);
        float g_bl = fabs(b_cc - b_bl);
        float g_max = fmax(g_br, g_bl);

        // Binary gate (for back-compat and mask= mode).
        int edge = (g_br >= threshold) || (g_bl >= threshold);

        // High quality: add vertical and horizontal samples.
        if (highq) {
            float b_bc = read_pixel_f(pBlur, blurPitch, ix,     iy + 1, dstWidth, dstHeight);
            float b_cr = read_pixel_f(pBlur, blurPitch, ix + 1, iy,     dstWidth, dstHeight);
            float g_bc = fabs(b_cc - b_bc);
            float g_cr = fabs(b_cc - b_cr);
            edge = edge || (g_bc >= threshold) || (g_cr >= threshold);
            g_max = fmax(g_max, fmax(g_bc, g_cr));
        }

        // Soft-mask weight via sigmoid when slope > 0; otherwise the
        // binary edge flag is used (back-compat path).
        const float soft_w = (slope > 0.0f) ? msharpen_sigmoid_weight(g_max, threshold, slope) : (edge ? 1.0f : 0.0f);

        // Luma-adaptive attenuation. luma_limit_norm <= 0 disables the
        // feature; positive values scale by min(src / luma_limit_norm, 1)
        // so dark regions get less sharpening (where noise is most visible).
        const float luma_w = (luma_limit_norm > 0.0f) ? fmin(src / luma_limit_norm, 1.0f) : 1.0f;

        float result;
        if (mask) {
            // In mask mode we visualise the gating weight directly.
            result = soft_w;
        } else {
            const float sharpened = clamp(4.0f * src - 3.0f * b_cc, 0.0f, 1.0f);
            // Effective mix weight: per-pixel mask * global strength * luma_weight.
            const float w = soft_w * strength * luma_w;
            result = w * sharpened + (1.0f - w) * src;
        }

        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(clamp(result, 0.0f, 1.0f - RGY_FLT_EPS) * PIXEL_MAX);
    }
}
