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

// Fused 3x3 box blur + edge-selective sharpen.
//
// Replaces the prior two-pass chain (kernel_msharpen_blur ->
// kernel_msharpen_sharpen) and eliminates the intermediate blur frame
// buffer. The sharpen step needs blur values at five offsets around
// the centre: (0,0), (+1,+1), (-1,+1), (0,+1), (+1,0). Each of those
// is a 3x3 average of the source plane, so the union of source samples
// required is a 5-wide (ix-2..ix+2) by 4-tall (iy-1..iy+2) window. We
// load all 20 of those samples into registers once and form every
// needed blur value from them, then run the existing sharpen / mask /
// block-protect / luma-limit logic against the in-register blurs.
//
// Output is mathematically identical to the two-pass version: all
// arithmetic is FP32, the box-blur average is computed with the same
// 1.0/9.0 factor, and the sharpen branch reads the same blur offsets.
__kernel void kernel_msharpen(
    __global uchar *restrict pDst, const int dstPitch,
    const int dstWidth, const int dstHeight,
    const __global uchar *pSrc, const int srcPitch,
    const float strength, const float threshold,
    const float slope, const float luma_limit_norm,
    const int highq, const int mask,
    const float block_protect) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix >= dstWidth || iy >= dstHeight) return;

    // Load the 5x4 source neighbourhood once. Layout:
    //   s[col + 2][row + 1] = pSrc(ix + col, iy + row)
    // with col in [-2, +2] and row in [-1, +2]. Rows / cols outside
    // the active set for the current `highq` branch will be DCE'd
    // by the compiler when their consumers are folded.
    float s[5][4];
    for (int dy = -1; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            s[dx + 2][dy + 1] = read_pixel_f(pSrc, srcPitch, ix + dx, iy + dy, dstWidth, dstHeight);
        }
    }

    // 3x3 box blur sums.
    // b_cc centred at (ix, iy) uses cols {-1, 0, +1} x rows {-1, 0, +1}
    const float b_cc = (
          s[1][0] + s[2][0] + s[3][0]
        + s[1][1] + s[2][1] + s[3][1]
        + s[1][2] + s[2][2] + s[3][2]) * (1.0f / 9.0f);
    // b_br centred at (ix+1, iy+1) uses cols { 0, +1, +2} x rows {0, +1, +2}
    const float b_br = (
          s[2][1] + s[3][1] + s[4][1]
        + s[2][2] + s[3][2] + s[4][2]
        + s[2][3] + s[3][3] + s[4][3]) * (1.0f / 9.0f);
    // b_bl centred at (ix-1, iy+1) uses cols {-2, -1,  0} x rows {0, +1, +2}
    const float b_bl = (
          s[0][1] + s[1][1] + s[2][1]
        + s[0][2] + s[1][2] + s[2][2]
        + s[0][3] + s[1][3] + s[2][3]) * (1.0f / 9.0f);

    const float src = s[2][1];  // pSrc(ix, iy)

    // Diagonal gradient magnitudes.
    float g_br = fabs(b_cc - b_br);
    float g_bl = fabs(b_cc - b_bl);
    float g_max = fmax(g_br, g_bl);

    // Binary gate (for back-compat and mask= mode).
    int edge = (g_br >= threshold) || (g_bl >= threshold);

    // High quality: add vertical and horizontal samples.
    if (highq) {
        // b_bc centred at (ix, iy+1) uses cols {-1, 0, +1} x rows {0, +1, +2}
        const float b_bc = (
              s[1][1] + s[2][1] + s[3][1]
            + s[1][2] + s[2][2] + s[3][2]
            + s[1][3] + s[2][3] + s[3][3]) * (1.0f / 9.0f);
        // b_cr centred at (ix+1, iy) uses cols { 0, +1, +2} x rows {-1, 0, +1}
        const float b_cr = (
              s[2][0] + s[3][0] + s[4][0]
            + s[2][1] + s[3][1] + s[4][1]
            + s[2][2] + s[3][2] + s[4][2]) * (1.0f / 9.0f);
        float g_bc = fabs(b_cc - b_bc);
        float g_cr = fabs(b_cc - b_cr);
        edge = edge || (g_bc >= threshold) || (g_cr >= threshold);
        g_max = fmax(g_max, fmax(g_bc, g_cr));
    }

    // Soft-mask weight via sigmoid when slope > 0; otherwise the
    // binary edge flag is used (back-compat path).
    const float soft_w = (slope > 0.0f) ? msharpen_sigmoid_weight(g_max, threshold, slope) : (edge ? 1.0f : 0.0f);

    // DCT block-boundary protection. When block_protect > 0, detect
    // edges aligned to the 8-pixel block grid using the existing
    // threshold= parameter. Pixels within 1 px of a detected boundary
    // get the edge mask attenuated by (1 - block_protect * block_w).
    // block_protect == 0 skips the entire branch and effective_mask
    // == soft_w, preserving byte-identity with prior builds.
    float effective_mask = soft_w;
    if (block_protect > 0.0f) {
        // Nearest vertical block line (multiple of 8) to ix.
        const int nearest_v = ((ix + 4) >> 3) << 3;
        const int dx = ix - nearest_v;
        const int abs_dx = (dx < 0) ? -dx : dx;
        float v_block_w = 0.0f;
        if (abs_dx <= 1) {
            // Compare pixels across the boundary line.
            const float pl = read_pixel_f(pSrc, srcPitch, nearest_v - 1, iy, dstWidth, dstHeight);
            const float pr = read_pixel_f(pSrc, srcPitch, nearest_v,     iy, dstWidth, dstHeight);
            if (fabs(pl - pr) > threshold) {
                v_block_w = (abs_dx == 0) ? 1.0f : 0.5f;
            }
        }
        // Nearest horizontal block line.
        const int nearest_h = ((iy + 4) >> 3) << 3;
        const int dy = iy - nearest_h;
        const int abs_dy = (dy < 0) ? -dy : dy;
        float h_block_w = 0.0f;
        if (abs_dy <= 1) {
            const float pu = read_pixel_f(pSrc, srcPitch, ix, nearest_h - 1, dstWidth, dstHeight);
            const float pd = read_pixel_f(pSrc, srcPitch, ix, nearest_h,     dstWidth, dstHeight);
            if (fabs(pu - pd) > threshold) {
                h_block_w = (abs_dy == 0) ? 1.0f : 0.5f;
            }
        }
        const float block_w = fmax(v_block_w, h_block_w);
        effective_mask = soft_w * (1.0f - block_protect * block_w);
    }

    // Luma-adaptive attenuation. luma_limit_norm <= 0 disables the
    // feature; positive values scale by min(src / luma_limit_norm, 1)
    // so dark regions get less sharpening (where noise is most visible).
    const float luma_w = (luma_limit_norm > 0.0f) ? fmin(src / luma_limit_norm, 1.0f) : 1.0f;

    float result;
    if (mask) {
        // In mask mode we visualise the gating weight directly,
        // reflecting any block-protect attenuation.
        result = effective_mask;
    } else {
        const float sharpened = clamp(4.0f * src - 3.0f * b_cc, 0.0f, 1.0f);
        // Effective mix weight: per-pixel mask * global strength * luma_weight.
        const float w = effective_mask * strength * luma_w;
        result = w * sharpened + (1.0f - w) * src;
    }

    __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(clamp(result, 0.0f, 1.0f - RGY_FLT_EPS) * PIXEL_MAX);
}
