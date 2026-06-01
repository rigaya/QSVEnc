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

// Contrast Adaptive Sharpening (CAS) algorithm:
//   Single-pass spatial sharpener using a 5-tap cross neighbourhood
//   {b, d, e, f, h} around centre e. Per-pixel adaptive coefficient
//   derived from local contrast keeps flat regions untouched while
//   sharpening edges.
//
//   peak  = -1 / mix(8.0, 5.0, sharpness)        (host-side, negative)
//   amp   = sqrt( saturate( min(mn, 1 - mx) / mx ) )
//   w     = amp * peak
//   out   = saturate( ((b + d + f + h) * w + e) / (1 + 4*w) )
//
// Type
// bit_depth

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

#define RGY_FLT_EPS (1e-6f)
#define PIXEL_MAX  ((1 << (bit_depth)) - 1)

// Read a sample as normalised float [0, 1] with boundary clamp.
static inline float cas_read(const __global uchar *pSrc, int srcPitch,
                             int x, int y, int width, int height) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    Type val = *(const __global Type *)(pSrc + y * srcPitch + x * sizeof(Type));
    return (float)val * (1.0f / (float)PIXEL_MAX);
}

// Gamma 2.0 approximation: cheap encode/decode of perceptual luma.
// linearise: x = x * x ; delinearise: x = sqrt(x).
// Used on the luma plane only when the caller did not request HDR
// bypass; chroma planes always skip this step (bipolar samples).
static inline float cas_linearise(float x, int apply) {
    return apply ? (x * x) : x;
}
static inline float cas_delinearise(float x, int apply) {
    return apply ? native_sqrt(x) : x;
}

__kernel void kernel_cas(
    __global uchar *restrict pDst, const int dstPitch,
    const int dstWidth, const int dstHeight,
    const __global uchar *pSrc, const int srcPitch,
    const float peak,
    const int apply_gamma2) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < dstWidth && iy < dstHeight) {
        //  a b c
        //  d e f
        //  g h i
        // Only the 5-tap cross {b, d, e, f, h} contributes to the filter;
        // the corner samples are unused in the default no-scaling path.
        float b = cas_read(pSrc, srcPitch, ix,     iy - 1, dstWidth, dstHeight);
        float d = cas_read(pSrc, srcPitch, ix - 1, iy,     dstWidth, dstHeight);
        float e = cas_read(pSrc, srcPitch, ix,     iy,     dstWidth, dstHeight);
        float f = cas_read(pSrc, srcPitch, ix + 1, iy,     dstWidth, dstHeight);
        float h = cas_read(pSrc, srcPitch, ix,     iy + 1, dstWidth, dstHeight);

        // Optional gamma 2.0 linearisation (luma only, non-HDR).
        b = cas_linearise(b, apply_gamma2);
        d = cas_linearise(d, apply_gamma2);
        e = cas_linearise(e, apply_gamma2);
        f = cas_linearise(f, apply_gamma2);
        h = cas_linearise(h, apply_gamma2);

        const float mn = fmin(fmin(fmin(fmin(b, d), e), f), h);
        const float mx = fmax(fmax(fmax(fmax(b, d), e), f), h);

        // Adaptive amplitude. saturate( min(mn, 1 - mx) / mx ), then sqrt
        // to shape the response. native_recip / native_sqrt match the
        // reference's APrxLoRcp / APrxLoSqrt approximations.
        const float rcpM = native_recip(fmax(mx, RGY_FLT_EPS));
        float amp = clamp(fmin(mn, 1.0f - mx) * rcpM, 0.0f, 1.0f);
        amp = native_sqrt(amp);

        // peak < 0, so w is a negative lobe weight.
        const float w = amp * peak;
        const float rcpW = native_recip(1.0f + 4.0f * w);
        float result = ((b + d + f + h) * w + e) * rcpW;
        result = clamp(result, 0.0f, 1.0f);

        result = cas_delinearise(result, apply_gamma2);
        result = clamp(result, 0.0f, 1.0f - RGY_FLT_EPS);

        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(result * PIXEL_MAX);
    }
}
