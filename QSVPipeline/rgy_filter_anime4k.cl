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

// Algorithm: Anime4K v3.2 by bloc97 (MIT, 2019).
//   * sobel_x / sobel_y      cite Anime4K_Upscale_Original_x2.glsl v3.2 (passes 2-3)
//   * refine_x / refine_y    cite Anime4K_Upscale_Original_x2.glsl v3.2 (passes 4-5)
//   * apply                  cite Anime4K_Upscale_Original_x2.glsl v3.2 (pass 6)
//   * polynomial P5..P0      cite same file, "Polynomial fit obtained by
//                            minimizing MSE error on image"
// No shader source was copied verbatim. See ACKNOWLEDGMENTS.md.
//
// Build-time defines:
//   Type        = uchar / ushort   (host plane data type)
//   bit_depth   = 8 / 10 / 12 / 16
//   SCALE       = 1 or 2           (output luma scale factor)
//   ANIME4K_REFINE_STRENGTH = float (0.2 .. 4.0; passed at build time)
//
// Scratch layout: two float4 buffers, outW * outH pixels each.
//   scratchA / scratchB are written by the chain in ping-pong order.
//   .xy carries the two-component Sobel partial / direction
//   .z  carries dval (edge-refinement weight) propagated forward
//        from sobel_y so the apply pass has it without needing a
//        separate LUMAD buffer.

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

#define PIXEL_MAX  ((1 << (bit_depth)) - 1)
#define RGY_FLT_EPS (1e-6f)
#define ANIME4K_DVAL_THRESHOLD 0.1f

// Polynomial-fit coefficients minimising MSE on the gradient-magnitude
// curve. Cited from Anime4K_Upscale_Original_x2.glsl v3.2 lines 97-102.
#define ANIME4K_P5 ( 11.68129591f)
#define ANIME4K_P4 (-42.46906057f)
#define ANIME4K_P3 ( 60.28286266f)
#define ANIME4K_P2 (-41.84451327f)
#define ANIME4K_P1 ( 14.05517353f)
#define ANIME4K_P0 (-1.081521930f)

// Normalised-coordinate sampler used to read the source luma plane.
// Hardware bilinear gives the free 2x upscale when reading at the
// 2x target grid; CLAMP_TO_EDGE matches the reference shader's
// implicit edge behaviour at the source frame boundary.
constant sampler_t anime4k_src_sampler =
    CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

static inline float anime4k_poly5(float x) {
    float x2 = x * x;
    float x3 = x2 * x;
    float x4 = x2 * x2;
    float x5 = x2 * x3;
    return ANIME4K_P5 * x5 + ANIME4K_P4 * x4 + ANIME4K_P3 * x3
         + ANIME4K_P2 * x2 + ANIME4K_P1 * x + ANIME4K_P0;
}

// Pixel center in normalised source coords. ix, iy are the output
// grid integer coordinates; out_w, out_h are output dims. For SCALE=2
// the implicit step is half a source pixel, which is exactly what
// hardware bilinear interpolates between source samples for.
static inline float2 anime4k_src_coord(int ix, int iy, int out_w, int out_h) {
    return (float2)(((float)ix + 0.5f) / (float)out_w,
                    ((float)iy + 0.5f) / (float)out_h);
}

// Forward declaration. Defined further down (next to the darken/thin
// kernels that were its original first users); the Tier 1 chroma
// kernels above the definition also need it, so declare it up-front.
static inline float anime4k_read_y_norm(__global const uchar *pY, int pitch, int x, int y);

// Pass 2 (cite Anime4K_Upscale_Original_x2.glsl v3.2): horizontal Sobel
// pre-pass. Reads three luma samples on a horizontal line via the
// source sampler (hardware bilinear handles the SCALE=2 upscale
// implicitly) and writes the partial Sobel (-l+r, l+2c+r) into the
// LUMAD-partial scratch's .xy. .z and .w are zeroed.
__kernel void kernel_anime4k_sobel_x(
    __global float4 *restrict pDstA, const int dstPitchFloats,
    __read_only image2d_t srcImage,
    const int srcW, const int srcH,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float dx = 1.0f / (float)outW;
    const float2 p = anime4k_src_coord(ix, iy, outW, outH);

    const float l = read_imagef(srcImage, anime4k_src_sampler, (float2)(p.x - dx, p.y)).x;
    const float c = read_imagef(srcImage, anime4k_src_sampler, p).x;
    const float r = read_imagef(srcImage, anime4k_src_sampler, (float2)(p.x + dx, p.y)).x;

    const float xgrad = -l + r;
    const float ygrad = l + c + c + r;

    pDstA[iy * dstPitchFloats + ix] = (float4)(xgrad, ygrad, 0.0f, 0.0f);
}

// Pass 3 (cite Anime4K_Upscale_Original_x2.glsl v3.2): vertical Sobel
// completion + polynomial refinement. Reads the horizontal partial
// produced by pass 2 at three vertical coords, completes the 2D
// Sobel norm, then maps sobel_norm through the published P5..P0
// polynomial scaled by REFINE_STRENGTH to produce per-pixel dval.
// Output scratchB.xy = (sobel_norm, dval); dval also lives in .y
// for clarity. .z and .w zeroed.
__kernel void kernel_anime4k_sobel_y(
    __global float4 *restrict pDstB, const int dstPitchFloats,
    __global const float4 *pSrcA, const int srcPitchFloats,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const int iy_t = max(iy - 1, 0);
    const int iy_b = min(iy + 1, outH - 1);

    const float4 t = pSrcA[iy_t * srcPitchFloats + ix];
    const float4 c = pSrcA[iy   * srcPitchFloats + ix];
    const float4 b = pSrcA[iy_b * srcPitchFloats + ix];

    const float xgrad = t.x + c.x + c.x + b.x;
    const float ygrad = -t.y + b.y;

    const float sobel_norm = clamp(native_sqrt(xgrad * xgrad + ygrad * ygrad), 0.0f, 1.0f);
    const float dval = clamp(anime4k_poly5(sobel_norm) * ANIME4K_REFINE_STRENGTH, 0.0f, 1.0f);

    pDstB[iy * dstPitchFloats + ix] = (float4)(sobel_norm, dval, 0.0f, 0.0f);
}

// Pass 4 (cite Anime4K_Upscale_Original_x2.glsl v3.2): horizontal
// min-max refinement. Reads sobel_norm (.x) from the LUMAD scratch
// at three horizontal coords and produces the LUMAMM horizontal
// partial. Propagates dval (the .y component at the centre coord)
// forward into the output .z so refine_y and apply have it without
// rereading the LUMAD buffer.
__kernel void kernel_anime4k_refine_x(
    __global float4 *restrict pDstA, const int dstPitchFloats,
    __global const float4 *pSrcB, const int srcPitchFloats,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float4 cval = pSrcB[iy * srcPitchFloats + ix];
    const float dval = cval.y;
    if (dval < ANIME4K_DVAL_THRESHOLD) {
        pDstA[iy * dstPitchFloats + ix] = (float4)(0.0f, 0.0f, dval, 0.0f);
        return;
    }

    const int ix_l = max(ix - 1, 0);
    const int ix_r = min(ix + 1, outW - 1);

    const float l = pSrcB[iy * srcPitchFloats + ix_l].x;
    const float c = cval.x;
    const float r = pSrcB[iy * srcPitchFloats + ix_r].x;

    const float xgrad = -l + r;
    const float ygrad = l + c + c + r;

    pDstA[iy * dstPitchFloats + ix] = (float4)(xgrad, ygrad, dval, 0.0f);
}

// Pass 5 (cite Anime4K_Upscale_Original_x2.glsl v3.2): vertical
// completion of LUMAMM, with norm normalisation. Reads the LUMAMM
// partial at three vertical coords; output .xy is the normalised
// gradient direction. dval is forwarded from the centre coord's .z.
__kernel void kernel_anime4k_refine_y(
    __global float4 *restrict pDstB, const int dstPitchFloats,
    __global const float4 *pSrcA, const int srcPitchFloats,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float4 cval = pSrcA[iy * srcPitchFloats + ix];
    const float dval = cval.z;
    if (dval < ANIME4K_DVAL_THRESHOLD) {
        pDstB[iy * dstPitchFloats + ix] = (float4)(0.0f, 0.0f, dval, 0.0f);
        return;
    }

    const int iy_t = max(iy - 1, 0);
    const int iy_b = min(iy + 1, outH - 1);

    const float4 t = pSrcA[iy_t * srcPitchFloats + ix];
    const float4 b = pSrcA[iy_b * srcPitchFloats + ix];

    const float xgrad = t.x + cval.x + cval.x + b.x;
    const float ygrad = -t.y + b.y;

    float norm = native_sqrt(xgrad * xgrad + ygrad * ygrad);
    float ndx = 0.0f;
    float ndy = 0.0f;
    if (norm > 0.001f) {
        ndx = xgrad / norm;
        ndy = ygrad / norm;
    }
    pDstB[iy * dstPitchFloats + ix] = (float4)(ndx, ndy, dval, 0.0f);
}

// Pass 6 (cite Anime4K_Upscale_Original_x2.glsl v3.2): apply.
// For each output pixel: if dval below threshold or direction is
// near-zero, pass through the bilinear-upscaled source unchanged.
// Otherwise, blend the centre source sample with one neighbour
// sample along x and one along y, weighted by the gradient direction
// magnitude, then mix with the centre by dval. Final value is
// denormalised to the output type / bit depth and stored in the
// output luma plane.
__kernel void kernel_anime4k_apply(
    __global uchar *restrict pDstY, const int dstPitch,
    __read_only image2d_t srcImage,
    __global const float4 *pSrcB, const int srcPitchFloats,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float dx = 1.0f / (float)outW;
    const float dy = 1.0f / (float)outH;
    const float2 p = anime4k_src_coord(ix, iy, outW, outH);

    const float center = read_imagef(srcImage, anime4k_src_sampler, p).x;

    const float4 dc = pSrcB[iy * srcPitchFloats + ix];
    const float dval = dc.z;

    float result;
    if (dval < ANIME4K_DVAL_THRESHOLD || fabs(dc.x + dc.y) <= 0.0001f) {
        result = center;
    } else {
        const float xstep = -sign(dc.x) * dx;
        const float ystep = -sign(dc.y) * dy;
        const float xval = read_imagef(srcImage, anime4k_src_sampler, (float2)(p.x + xstep, p.y)).x;
        const float yval = read_imagef(srcImage, anime4k_src_sampler, (float2)(p.x, p.y + ystep)).x;
        const float adx = fabs(dc.x);
        const float ady = fabs(dc.y);
        const float xyratio = adx / (adx + ady + RGY_FLT_EPS);
        const float avg = xyratio * xval + (1.0f - xyratio) * yval;
        result = avg * dval + center * (1.0f - dval);
    }

    result = clamp(result, 0.0f, 1.0f);

    __global Type *ptr = (__global Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)PIXEL_MAX + 0.5f);
}

// Chroma plane resize used only when SCALE=2 and the filter is
// configured to handle U/V (the default). chromaMode selects the
// resampling kernel: 0=spline36, 1=bilinear, 2=bicubic (Mitchell-
// Netravali B=1/3, C=1/3), 3=lanczos3. All three published kernels
// are public mathematical formulae and appear in every independent
// resampler; coefficients here are derived from the published
// formulae rather than copied from any specific implementation.
// This is plain geometric resize, not luma-guided.

// 3.14159265358979323846 is mathematical fact, not copyrightable.
#define ANIME4K_PI_F 3.14159265358979323846f

static inline float anime4k_sinc(float x) {
    if (fabs(x) < 1e-9f) return 1.0f;
    const float px = ANIME4K_PI_F * x;
    return native_sin(px) / px;
}

static inline float anime4k_chroma_weight(int chromaMode, float x) {
    const float ax = fabs(x);
    if (chromaMode == 1) {                       // bilinear (2-tap)
        return (ax < 1.0f) ? (1.0f - ax) : 0.0f;
    } else if (chromaMode == 2) {                // bicubic, Mitchell-Netravali B=C=1/3
        const float x2 = ax * ax;
        const float x3 = x2 * ax;
        if (ax < 1.0f) {
            return ((12.0f - 9.0f * (1.0f / 3.0f) - 6.0f * (1.0f / 3.0f)) * x3
                  + (-18.0f + 12.0f * (1.0f / 3.0f) + 6.0f * (1.0f / 3.0f)) * x2
                  + (6.0f - 2.0f * (1.0f / 3.0f))) / 6.0f;
        } else if (ax < 2.0f) {
            return ((-(1.0f / 3.0f) - 6.0f * (1.0f / 3.0f)) * x3
                  + (6.0f * (1.0f / 3.0f) + 30.0f * (1.0f / 3.0f)) * x2
                  + (-12.0f * (1.0f / 3.0f) - 48.0f * (1.0f / 3.0f)) * ax
                  + (8.0f * (1.0f / 3.0f) + 24.0f * (1.0f / 3.0f))) / 6.0f;
        }
        return 0.0f;
    } else if (chromaMode == 3) {                // lanczos3 (6-tap)
        if (ax < 3.0f) {
            return anime4k_sinc(x) * anime4k_sinc(x / 3.0f);
        }
        return 0.0f;
    } else {                                     // spline36 (6-tap)
        if (ax < 1.0f) {
            return ((13.0f / 11.0f) * ax - (453.0f / 209.0f)) * ax * ax + (1.0f - 3.0f / 209.0f);
        } else if (ax < 2.0f) {
            return ((-(6.0f / 11.0f) * ax + (612.0f / 209.0f)) * ax + (-(1038.0f / 209.0f))) * ax + (540.0f / 209.0f);
        } else if (ax < 3.0f) {
            return (((1.0f / 11.0f) * ax + (-(159.0f / 209.0f))) * ax + (434.0f / 209.0f)) * ax + (-(384.0f / 209.0f));
        }
        return 0.0f;
    }
}

#define ANIME4K_CHROMA_TAPS 6

__kernel void kernel_anime4k_chroma_resize(
    __global uchar *restrict pDstC, const int dstPitch,
    const int dstW, const int dstH,
    __global const uchar *pSrcC, const int srcPitch,
    const int srcW, const int srcH,
    const int chromaMode) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= dstW || iy >= dstH) return;

    const float sx = ((float)ix + 0.5f) * (float)srcW / (float)dstW - 0.5f;
    const float sy = ((float)iy + 0.5f) * (float)srcH / (float)dstH - 0.5f;
    const int sxi = (int)floor(sx);
    const int syi = (int)floor(sy);

    // `half` is an OpenCL reserved type name (FP16), so the kernel
    // half-width is named kernel_half to avoid shadowing the type.
    const int taps = (chromaMode == 1) ? 2 : ((chromaMode == 2) ? 4 : ANIME4K_CHROMA_TAPS);
    const int kernel_half = taps / 2;
    const int t0 = 1 - kernel_half;
    const int t1 = kernel_half;

    float acc = 0.0f;
    float wsum = 0.0f;
    for (int dy = t0; dy <= t1; ++dy) {
        const int sy_c = clamp(syi + dy, 0, srcH - 1);
        const float wy = anime4k_chroma_weight(chromaMode, (float)dy - (sy - (float)syi));
        for (int dx = t0; dx <= t1; ++dx) {
            const int sx_c = clamp(sxi + dx, 0, srcW - 1);
            const float wx = anime4k_chroma_weight(chromaMode, (float)dx - (sx - (float)sxi));
            const float w = wx * wy;
            const Type s = *(const __global Type *)(pSrcC + sy_c * srcPitch + sx_c * sizeof(Type));
            const float v = (float)s * (1.0f / (float)PIXEL_MAX);
            acc += v * w;
            wsum += w;
        }
    }
    float out = (wsum > 1e-6f) ? (acc / wsum) : 0.0f;
    out = clamp(out, 0.0f, 1.0f);

    __global Type *ptr = (__global Type *)(pDstC + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(out * (float)PIXEL_MAX + 0.5f);
}

// -------------------------------------------------------------------------
// Joint-bilateral chroma reconstruction (chroma_resize=joint). Ported from
// Artoriuz / Joao Chrisostomo FastBilateral.glsl (MIT). Upscales the source
// chroma plane to the output chroma resolution using the LUMA plane as a
// guide, so chroma edges follow luma edges (sharper, no bleeding) -- a real
// gain over a plain geometric resize on soft 4:2:0 sources.
// -------------------------------------------------------------------------

// Pass 1: box-downscale the source luma (lumaW x lumaH) to chroma resolution
// (lowW x lowH) for the bilateral intensity-similarity term.
__kernel void kernel_anime4k_chroma_luma_lowres(
    __global uchar *restrict pLow, const int lowPitch,
    const int lowW, const int lowH,
    __global const uchar *pLuma,   const int lumaPitch,
    const int lumaW, const int lumaH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= lowW || iy >= lowH) return;
    const int lx  = min(ix * 2,     lumaW - 1);
    const int ly  = min(iy * 2,     lumaH - 1);
    const int lx1 = min(lx + 1,     lumaW - 1);
    const int ly1 = min(ly + 1,     lumaH - 1);
    const float avg = 0.25f * (anime4k_read_y_norm(pLuma, lumaPitch, lx,  ly)
                             + anime4k_read_y_norm(pLuma, lumaPitch, lx1, ly)
                             + anime4k_read_y_norm(pLuma, lumaPitch, lx,  ly1)
                             + anime4k_read_y_norm(pLuma, lumaPitch, lx1, ly1));
    __global Type *ptr = (__global Type *)(pLow + iy * lowPitch + ix * sizeof(Type));
    ptr[0] = (Type)(avg * (float)PIXEL_MAX + 0.5f);
}

// Pass 2: for each output chroma pixel, blend its 2x2 source-chroma neighbours
// weighted by spatial distance and luma-intensity similarity (high-res luma at
// the output position vs the low-res luma at each neighbour). dist_coeff /
// int_coeff are FastBilateral's defaults (2.0 / 128.0). Runs once per chroma
// plane (U, V); the weights depend only on luma so both planes get the same
// envelope.
__kernel void kernel_anime4k_chroma_joint_bilateral(
    __global uchar *restrict pDstC, const int dstPitch,
    const int dstW, const int dstH,
    __global const uchar *pSrcC,  const int srcPitch,
    const int srcW, const int srcH,
    __global const uchar *pLuma,  const int lumaPitch, const int lumaW, const int lumaH,
    __global const uchar *pLow,   const int lowPitch,
    const float dist_coeff, const float int_coeff) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= dstW || iy >= dstH) return;
    // 出力chroma座標をソースluma空間にマッピング (YUV420では等倍、YUV444では縮小)
    const int lumaX = clamp((int)(((float)ix + 0.5f) * (float)lumaW / (float)dstW), 0, lumaW - 1);
    const int lumaY = clamp((int)(((float)iy + 0.5f) * (float)lumaH / (float)dstH), 0, lumaH - 1);
    const float luma_zero = anime4k_read_y_norm(pLuma, lumaPitch, lumaX, lumaY);
    const float px = ((float)ix + 0.5f) * (float)srcW / (float)dstW - 0.5f;
    const float py = ((float)iy + 0.5f) * (float)srcH / (float)dstH - 0.5f;
    const int fx = (int)floor(px);
    const int fy = (int)floor(py);
    const float frx = px - (float)fx;
    const float fry = py - (float)fy;
    float wt = 0.0f, ct = 0.0f;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            const int cx = clamp(fx + i, 0, srcW - 1);
            const int cy = clamp(fy + j, 0, srcH - 1);
            const float chroma = anime4k_read_y_norm(pSrcC, srcPitch, cx, cy);
            const float lowl   = anime4k_read_y_norm(pLow,  lowPitch, cx, cy);
            const float sdx = (float)i - frx;
            const float sdy = (float)j - fry;
            const float idiff = luma_zero - lowl;
            const float w = fmax(100.0f * native_exp(-dist_coeff * (sdx*sdx + sdy*sdy)
                                                     - int_coeff  * (idiff*idiff)), 1e-32f);
            wt += w;
            ct += w * chroma;
        }
    }
    const float outv = clamp(ct / wt, 0.0f, 1.0f);
    __global Type *ptr = (__global Type *)(pDstC + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(outv * (float)PIXEL_MAX + 0.5f);
}

// =========================================================================
// Darken_HQ + Thin_HQ post-process flags
// =========================================================================
//
// Both chains operate on the Y plane in place after the base apply pass.
// They share the same float4 scratchA / scratchB ping-pong allocated for
// the base chain; only the .x and .xy lanes are used by these kernels.
// When darken=false and thin=false, none of these kernels are dispatched
// and the output is byte-identical to the base chain.
//
// Algorithm citations:
//   Darken: Anime4K_Darken_HQ.glsl v3.2 (Experimental-Effects)
//   Thin:   Anime4K_Thin_HQ.glsl   v3.2 (Experimental-Effects)
// Both: MIT-licensed Anime4K v3.2 by bloc97 (2019). No GLSL was copied
// verbatim; the OpenCL is from the published algorithm description.

// Sigma scaling matches the reference shaders:
//   Darken: SPATIAL_SIGMA = 1.0 * h / 1080
//   Thin:   SPATIAL_SIGMA = 2.0 * h / 1080
// The 1D Gaussian kernel is built inline per pass (radius derived from
// sigma), so the same kernel source supports any output resolution.
// STRENGTH constants are the published values: darken=1.5, thin=0.6.

// Darken / Thin published strength constants. Made -D-overridable so
// the composite mode=dtd path can bake its tuned-down values (1.8 for
// darken, 0.4 for thin) without disturbing the standalone darken= /
// thin= post-process flags (which use 1.5 / 0.6).
#ifndef ANIME4K_DARKEN_STRENGTH
#define ANIME4K_DARKEN_STRENGTH 1.5f
#endif
#ifndef ANIME4K_THIN_STRENGTH
#define ANIME4K_THIN_STRENGTH   0.6f
#endif
// Sigma reference coefficients. The full per-frame sigma is
// (coef * outH / 1080); the coef itself selects the tier and is
// supplied by the host build at JIT time:
//   darken=hq       -> 1.0          thin=hq       -> 2.0
//   darken=fast     -> 0.5          thin=fast     -> 1.0
//   darken=veryfast -> 0.25         thin=veryfast -> 0.5
// The default values here match the HQ tier so the source can compile
// stand-alone for inspection.
#ifndef ANIME4K_DARKEN_SIGMA_REF
#define ANIME4K_DARKEN_SIGMA_REF 1.0f
#endif
#ifndef ANIME4K_THIN_SIGMA_REF
#define ANIME4K_THIN_SIGMA_REF   2.0f
#endif
// DoG-family (Deblur_DoG / Upscale_DoG / DTD) constants.
//   ANIME4K_DOG_STRENGTH          mode=dog_sharpen 0.6, mode=dog 0.8,
//                                 mode=dtd stage C 0.5
//   ANIME4K_DOG_BLUR_CURVE        soft-threshold power curve. 0.6 for
//                                 dog_sharpen, 0.8 for dtd; unused by
//                                 dog (the upscale apply skips soft
//                                 threshold and uses a plain unsharp).
//   ANIME4K_DOG_BLUR_THRESHOLD    upper edge of the soft-threshold
//                                 zone. Constant 0.1 across all DoG
//                                 variants in the upstream shaders.
//   ANIME4K_DOG_NOISE_THRESHOLD   lower edge. 0.001 for dog_sharpen,
//                                 0.004 for dtd.
#ifndef ANIME4K_DOG_STRENGTH
#define ANIME4K_DOG_STRENGTH         0.6f
#endif
#ifndef ANIME4K_DOG_BLUR_CURVE
#define ANIME4K_DOG_BLUR_CURVE       0.6f
#endif
#ifndef ANIME4K_DOG_BLUR_THRESHOLD
#define ANIME4K_DOG_BLUR_THRESHOLD   0.1f
#endif
#ifndef ANIME4K_DOG_NOISE_THRESHOLD
#define ANIME4K_DOG_NOISE_THRESHOLD  0.001f
#endif
#define ANIME4K_REF_HEIGHT       1080.0f
#define ANIME4K_GAUSS_MAX_RADIUS 12      // safety cap; covers sigma <= 4 (~4K)

// Scratch element type for the darken / thin chains. half (FP16) is
// selected at JIT time via -D ANIME4K_SCRATCH_FP16=1 when the OpenCL
// device advertises cl_khr_fp16; the kernel signatures use SCRATCH_BUF_T
// as the buffer's element type so the same kernel source compiles
// against either FP32 or FP16 scratch.
//
// SCRATCH_LOAD4 / SCRATCH_STORE4 read / write 4 packed elements (float4
// or half4) at the float4-or-half4 index `idx`. Both vload_half4 and
// vstore_half4 are OpenCL 1.2 core builtins, so no extension pragma
// is required for FP16 storage; arithmetic stays FP32 throughout
// (precision in [0, 1] = ~5e-4, well below the 1/255 output step).
//
// The base chain's m_scratchA / m_scratchB are not affected by this
// macro -- those buffers are float4 always and the polynomial P5..P0
// in `anime4k_poly5` needs full FP32 precision (worst-case
// intermediate magnitude ~170; FP16 absolute precision at that
// magnitude is ~0.17, ~40x the 1/255 quantisation step).
#if defined(ANIME4K_SCRATCH_FP16) && ANIME4K_SCRATCH_FP16
#define SCRATCH_ELEM_T half
#define SCRATCH_LOAD4(pBase, idx) vload_half4((idx), (__global const half *)(pBase))
#define SCRATCH_STORE4(val, pBase, idx) vstore_half4((val), (idx), (__global half *)(pBase))
#else
#define SCRATCH_ELEM_T float
#define SCRATCH_LOAD4(pBase, idx) (((__global const float4 *)(pBase))[(idx)])
#define SCRATCH_STORE4(val, pBase, idx) (((__global float4 *)(pBase))[(idx)] = (val))
#endif

// Pixel-domain Gaussian weight: w(d) = exp(-0.5 * (d/sigma)^2). Caller
// normalises by the running sum so the weights need not be pre-normalised.
static inline float anime4k_gauss_w(float d, float sigma) {
    const float s = d / sigma;
    return native_exp(-0.5f * s * s);
}

// Read pDstY[x,y] as a float in [0, 1]. Used by every darken/thin kernel
// that touches the output luma plane.
static inline float anime4k_read_y_norm(__global const uchar *pY, int pitch, int x, int y) {
    const Type v = *(const __global Type *)(pY + y * pitch + x * sizeof(Type));
    return (float)v * (1.0f / (float)PIXEL_MAX);
}

// -----------------------------------------------------------------------
// Darken_HQ -- four kernels
// -----------------------------------------------------------------------

// Darken pass 1 (cite Anime4K_Darken_HQ.glsl v3.2): separable horizontal
// Gaussian blur of the post-apply luma plane. Writes float to scratchA.x.
__kernel void kernel_anime4k_darken_gauss1_x(
    __global SCRATCH_ELEM_T *restrict pDstA, const int dstPitchFloats,
    __global const uchar *pSrcY, const int srcPitch,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float sigma = ANIME4K_DARKEN_SIGMA_REF * (float)outH / ANIME4K_REF_HEIGHT;
    const int radius = max((int)ceil(sigma * 2.0f), 1);
    const int r = min(radius, ANIME4K_GAUSS_MAX_RADIUS);

    float acc = 0.0f;
    float wsum = 0.0f;
    for (int dx = -r; dx <= r; ++dx) {
        const int xx = clamp(ix + dx, 0, outW - 1);
        const float w = anime4k_gauss_w((float)dx, sigma);
        acc  += anime4k_read_y_norm(pSrcY, srcPitch, xx, iy) * w;
        wsum += w;
    }
    SCRATCH_STORE4((float4)(acc / wsum, 0.0f, 0.0f, 0.0f), pDstA, iy * dstPitchFloats + ix);
}

// Darken pass 2 (cite Anime4K_Darken_HQ.glsl v3.2): vertical Gaussian
// completion against the post-apply luma plane, then DoG dark-half:
//   dark = min(luma_center - blur_y, 0)
// Only the negative half of the difference is kept, so subsequent
// passes act on dark lines only (the negative offset, added back later
// with positive STRENGTH, darkens line interiors).
__kernel void kernel_anime4k_darken_dog_y(
    __global SCRATCH_ELEM_T *restrict pDstB, const int dstPitchFloats,
    __global const SCRATCH_ELEM_T *pSrcA, const int srcPitchFloats,
    __global const uchar *pSrcY, const int srcPitch,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float sigma = ANIME4K_DARKEN_SIGMA_REF * (float)outH / ANIME4K_REF_HEIGHT;
    const int radius = max((int)ceil(sigma * 2.0f), 1);
    const int r = min(radius, ANIME4K_GAUSS_MAX_RADIUS);

    float acc = 0.0f;
    float wsum = 0.0f;
    for (int dy = -r; dy <= r; ++dy) {
        const int yy = clamp(iy + dy, 0, outH - 1);
        const float w = anime4k_gauss_w((float)dy, sigma);
        acc  += SCRATCH_LOAD4(pSrcA, yy * srcPitchFloats + ix).x * w;
        wsum += w;
    }
    const float blur = acc / wsum;
    const float luma = anime4k_read_y_norm(pSrcY, srcPitch, ix, iy);
    const float dog_dark = fmin(luma - blur, 0.0f);
    SCRATCH_STORE4((float4)(dog_dark, 0.0f, 0.0f, 0.0f), pDstB, iy * dstPitchFloats + ix);
}

// Darken pass 3 (cite Anime4K_Darken_HQ.glsl v3.2): horizontal Gaussian
// smoothing of the DoG dark-edge mask. Writes float to scratchA.x.
__kernel void kernel_anime4k_darken_gauss2_x(
    __global SCRATCH_ELEM_T *restrict pDstA, const int dstPitchFloats,
    __global const SCRATCH_ELEM_T *pSrcB, const int srcPitchFloats,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float sigma = ANIME4K_DARKEN_SIGMA_REF * (float)outH / ANIME4K_REF_HEIGHT;
    const int radius = max((int)ceil(sigma * 2.0f), 1);
    const int r = min(radius, ANIME4K_GAUSS_MAX_RADIUS);

    float acc = 0.0f;
    float wsum = 0.0f;
    for (int dx = -r; dx <= r; ++dx) {
        const int xx = clamp(ix + dx, 0, outW - 1);
        const float w = anime4k_gauss_w((float)dx, sigma);
        acc  += SCRATCH_LOAD4(pSrcB, iy * srcPitchFloats + xx).x * w;
        wsum += w;
    }
    SCRATCH_STORE4((float4)(acc / wsum, 0.0f, 0.0f, 0.0f), pDstA, iy * dstPitchFloats + ix);
}

// Darken pass 4 (cite Anime4K_Darken_HQ.glsl v3.2): vertical Gaussian
// completion + apply. Reads the smoothed DoG mask (vertical-blur-of-
// horizontal-blur from passes 2-3) and the current Y plane; writes back
// to Y with `Y + smoothed * STRENGTH`. STRENGTH=1.5 matches reference.
// Because dog_dark is non-positive, the addition darkens line interiors.
__kernel void kernel_anime4k_darken_apply_y(
    __global uchar *restrict pDstY, const int dstPitch,
    __global const SCRATCH_ELEM_T *pSrcA, const int srcPitchFloats,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float sigma = ANIME4K_DARKEN_SIGMA_REF * (float)outH / ANIME4K_REF_HEIGHT;
    const int radius = max((int)ceil(sigma * 2.0f), 1);
    const int r = min(radius, ANIME4K_GAUSS_MAX_RADIUS);

    float acc = 0.0f;
    float wsum = 0.0f;
    for (int dy = -r; dy <= r; ++dy) {
        const int yy = clamp(iy + dy, 0, outH - 1);
        const float w = anime4k_gauss_w((float)dy, sigma);
        acc  += SCRATCH_LOAD4(pSrcA, yy * srcPitchFloats + ix).x * w;
        wsum += w;
    }
    const float smoothed = acc / wsum;
    const float luma = anime4k_read_y_norm(pDstY, dstPitch, ix, iy);
    float result = luma + smoothed * ANIME4K_DARKEN_STRENGTH;
    result = clamp(result, 0.0f, 1.0f);

    __global Type *ptr = (__global Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)PIXEL_MAX + 0.5f);
}

// -----------------------------------------------------------------------
// Thin_HQ -- eight kernels
// -----------------------------------------------------------------------

// Thin passes 1+2 fused (cite Anime4K_Thin_HQ.glsl v3.2 "Sobel-X" +
// "Sobel-Y"): a register-level fusion of the horizontal and vertical
// Sobel passes. The original two-kernel form wrote a (xgrad, ygrad)
// partial to scratchA and then read scratchA at three vertical
// positions to complete the 2D Sobel; the fused kernel reads pSrcY
// at the 3x3 neighbourhood directly and folds both passes into one
// kernel without an intermediate buffer round-trip. Same math as
// original two-pass chain. Writes the shaped magnitude `pow(norm, 0.7)`
// to .x of the destination scratch -- equivalent to original sobel_y
// output. No SLM used.
__kernel void kernel_anime4k_thin_sobel_xy(
    __global SCRATCH_ELEM_T *restrict pDstB, const int dstPitchFloats,
    __global const uchar *pSrcY, const int srcPitch,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const int ix_l = max(ix - 1, 0);
    const int ix_r = min(ix + 1, outW - 1);
    const int iy_t = max(iy - 1, 0);
    const int iy_b = min(iy + 1, outH - 1);

    // 9 luma reads in a 3x3 stencil. Each row's horizontal Sobel
    // partial is (xgrad_row, ygrad_row); the vertical Sobel completion
    // then combines the three rows' partials with the canonical
    // (1, 2, 1) / (-1, 0, 1) weights divided by 8.
    const float l_t = anime4k_read_y_norm(pSrcY, srcPitch, ix_l, iy_t);
    const float c_t = anime4k_read_y_norm(pSrcY, srcPitch, ix,   iy_t);
    const float r_t = anime4k_read_y_norm(pSrcY, srcPitch, ix_r, iy_t);
    const float l_c = anime4k_read_y_norm(pSrcY, srcPitch, ix_l, iy);
    const float c_c = anime4k_read_y_norm(pSrcY, srcPitch, ix,   iy);
    const float r_c = anime4k_read_y_norm(pSrcY, srcPitch, ix_r, iy);
    const float l_b = anime4k_read_y_norm(pSrcY, srcPitch, ix_l, iy_b);
    const float c_b = anime4k_read_y_norm(pSrcY, srcPitch, ix,   iy_b);
    const float r_b = anime4k_read_y_norm(pSrcY, srcPitch, ix_r, iy_b);

    // Horizontal partials per row: xgrad = -l + r; ygrad = l + 2c + r.
    const float xg_t = -l_t + r_t;
    const float yg_t =  l_t + c_t + c_t + r_t;
    const float xg_c = -l_c + r_c;
    const float yg_c =  l_c + c_c + c_c + r_c;
    const float xg_b = -l_b + r_b;
    const float yg_b =  l_b + c_b + c_b + r_b;

    // Vertical completion + L2 norm + response curve.
    const float xgrad = (xg_t + xg_c + xg_c + xg_b) * (1.0f / 8.0f);
    const float ygrad = (-yg_t + yg_b)              * (1.0f / 8.0f);
    const float norm  = native_sqrt(xgrad * xgrad + ygrad * ygrad);
    const float resp  = pow(norm, 0.7f);

    SCRATCH_STORE4((float4)(resp, 0.0f, 0.0f, 0.0f), pDstB, iy * dstPitchFloats + ix);
}

// Thin pass 3 (cite Anime4K_Thin_HQ.glsl v3.2, "Gaussian-X"): horizontal
// Gaussian smoothing of the shaped Sobel magnitude. sigma = 2 * h / 1080.
__kernel void kernel_anime4k_thin_gauss_x(
    __global SCRATCH_ELEM_T *restrict pDstA, const int dstPitchFloats,
    __global const SCRATCH_ELEM_T *pSrcB, const int srcPitchFloats,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float sigma = ANIME4K_THIN_SIGMA_REF * (float)outH / ANIME4K_REF_HEIGHT;
    const int radius = max((int)ceil(sigma * 2.0f), 1);
    const int r = min(radius, ANIME4K_GAUSS_MAX_RADIUS);

    float acc = 0.0f;
    float wsum = 0.0f;
    for (int dx = -r; dx <= r; ++dx) {
        const int xx = clamp(ix + dx, 0, outW - 1);
        const float w = anime4k_gauss_w((float)dx, sigma);
        acc  += SCRATCH_LOAD4(pSrcB, iy * srcPitchFloats + xx).x * w;
        wsum += w;
    }
    SCRATCH_STORE4((float4)(acc / wsum, 0.0f, 0.0f, 0.0f), pDstA, iy * dstPitchFloats + ix);
}

// Thin pass 4 (cite Anime4K_Thin_HQ.glsl v3.2, "Gaussian-Y"): vertical
// Gaussian completion of the shaped Sobel magnitude.
__kernel void kernel_anime4k_thin_gauss_y(
    __global SCRATCH_ELEM_T *restrict pDstB, const int dstPitchFloats,
    __global const SCRATCH_ELEM_T *pSrcA, const int srcPitchFloats,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float sigma = ANIME4K_THIN_SIGMA_REF * (float)outH / ANIME4K_REF_HEIGHT;
    const int radius = max((int)ceil(sigma * 2.0f), 1);
    const int r = min(radius, ANIME4K_GAUSS_MAX_RADIUS);

    float acc = 0.0f;
    float wsum = 0.0f;
    for (int dy = -r; dy <= r; ++dy) {
        const int yy = clamp(iy + dy, 0, outH - 1);
        const float w = anime4k_gauss_w((float)dy, sigma);
        acc  += SCRATCH_LOAD4(pSrcA, yy * srcPitchFloats + ix).x * w;
        wsum += w;
    }
    SCRATCH_STORE4((float4)(acc / wsum, 0.0f, 0.0f, 0.0f), pDstB, iy * dstPitchFloats + ix);
}

// Thin passes 5+6 fused (cite Anime4K_Thin_HQ.glsl v3.2 "Kernel-X" +
// "Kernel-Y"): a register-level fusion of the second-round Sobel
// passes over the smoothed magnitude. Reads pSrcB at the 3x3
// neighbourhood and folds both passes into one kernel. Writes the
// signed (xgrad, ygrad) flow field to .xy of the destination scratch.
// No norm / pow on this pass -- the reference keeps the signed
// gradient direction for the subsequent warp. No SLM used.
__kernel void kernel_anime4k_thin_kernel_xy(
    __global SCRATCH_ELEM_T *restrict pDstA, const int dstPitchFloats,
    __global const SCRATCH_ELEM_T *pSrcB, const int srcPitchFloats,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const int ix_l = max(ix - 1, 0);
    const int ix_r = min(ix + 1, outW - 1);
    const int iy_t = max(iy - 1, 0);
    const int iy_b = min(iy + 1, outH - 1);

    // 9 scratch reads of .x (shaped magnitude). pSrcB.x carries the
    // gauss-Y output value at each position; .yzw are unused here.
    const float l_t = SCRATCH_LOAD4(pSrcB, iy_t * srcPitchFloats + ix_l).x;
    const float c_t = SCRATCH_LOAD4(pSrcB, iy_t * srcPitchFloats + ix  ).x;
    const float r_t = SCRATCH_LOAD4(pSrcB, iy_t * srcPitchFloats + ix_r).x;
    const float l_c = SCRATCH_LOAD4(pSrcB, iy   * srcPitchFloats + ix_l).x;
    const float c_c = SCRATCH_LOAD4(pSrcB, iy   * srcPitchFloats + ix  ).x;
    const float r_c = SCRATCH_LOAD4(pSrcB, iy   * srcPitchFloats + ix_r).x;
    const float l_b = SCRATCH_LOAD4(pSrcB, iy_b * srcPitchFloats + ix_l).x;
    const float c_b = SCRATCH_LOAD4(pSrcB, iy_b * srcPitchFloats + ix  ).x;
    const float r_b = SCRATCH_LOAD4(pSrcB, iy_b * srcPitchFloats + ix_r).x;

    // Horizontal partials per row.
    const float xg_t = -l_t + r_t;
    const float yg_t =  l_t + c_t + c_t + r_t;
    const float xg_c = -l_c + r_c;
    const float yg_c =  l_c + c_c + c_c + r_c;
    const float xg_b = -l_b + r_b;
    const float yg_b =  l_b + c_b + c_b + r_b;

    // Vertical completion of the second-round Sobel (no norm/pow here).
    const float xgrad = (xg_t + xg_c + xg_c + xg_b) * (1.0f / 8.0f);
    const float ygrad = (-yg_t + yg_b)              * (1.0f / 8.0f);

    SCRATCH_STORE4((float4)(xgrad, ygrad, 0.0f, 0.0f), pDstA, iy * dstPitchFloats + ix);
}

// Thin pass 7 (preparation for warp): copy the post-apply luma plane
// into scratchA.x (or scratchB, depending on which buffer the host
// has free at this point) as float [0, 1]. The warp pass reads this
// scratch via manual bilinear interpolation and writes pDstY in place;
// we must not read pDstY at sub-pixel positions while concurrently
// writing it.
__kernel void kernel_anime4k_thin_copy_y_to_ref(
    __global SCRATCH_ELEM_T *restrict pDstA, const int dstPitchFloats,
    __global const uchar *pSrcY, const int srcPitch,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float v = anime4k_read_y_norm(pSrcY, srcPitch, ix, iy);
    SCRATCH_STORE4((float4)(v, 0.0f, 0.0f, 0.0f), pDstA, iy * dstPitchFloats + ix);
}

// Manual bilinear sample of the .x component of a float4 buffer at
// fractional pixel coordinates (fx, fy). Used by the warp pass for the
// Y reference and (when the flow buffer is at downsampled resolution
// for the Fast / VeryFast tiers) by anime4k_bilinear_xy below.
// Equivalent to a hardware CLK_FILTER_LINEAR + CLK_ADDRESS_CLAMP_TO_EDGE
// sampler on a buffer.
static inline float anime4k_bilinear_x(__global const SCRATCH_ELEM_T *buf, int pitchFloats,
                                       int w, int h, float fx, float fy) {
    const int x0 = clamp((int)floor(fx), 0, w - 1);
    const int y0 = clamp((int)floor(fy), 0, h - 1);
    const int x1 = clamp(x0 + 1, 0, w - 1);
    const int y1 = clamp(y0 + 1, 0, h - 1);
    const float dx = fx - (float)x0;
    const float dy = fy - (float)y0;
    const float v00 = SCRATCH_LOAD4(buf, y0 * pitchFloats + x0).x;
    const float v01 = SCRATCH_LOAD4(buf, y0 * pitchFloats + x1).x;
    const float v10 = SCRATCH_LOAD4(buf, y1 * pitchFloats + x0).x;
    const float v11 = SCRATCH_LOAD4(buf, y1 * pitchFloats + x1).x;
    return (1.0f - dx) * (1.0f - dy) * v00
         +         dx  * (1.0f - dy) * v01
         + (1.0f - dx) *         dy  * v10
         +         dx  *         dy  * v11;
}

// Bilinear sample of the .xy components of a float4 buffer at
// fractional pixel coordinates. Used by the warp pass to read the flow
// field when it lives at a lower resolution than the output (Fast and
// VeryFast tiers); for the HQ tier the caller passes integer coords
// and this collapses to a single texel read.
static inline float2 anime4k_bilinear_xy(__global const SCRATCH_ELEM_T *buf, int pitchFloats,
                                         int w, int h, float fx, float fy) {
    const int x0 = clamp((int)floor(fx), 0, w - 1);
    const int y0 = clamp((int)floor(fy), 0, h - 1);
    const int x1 = clamp(x0 + 1, 0, w - 1);
    const int y1 = clamp(y0 + 1, 0, h - 1);
    const float dx = fx - (float)x0;
    const float dy = fy - (float)y0;
    const float4 v00 = SCRATCH_LOAD4(buf, y0 * pitchFloats + x0);
    const float4 v01 = SCRATCH_LOAD4(buf, y0 * pitchFloats + x1);
    const float4 v10 = SCRATCH_LOAD4(buf, y1 * pitchFloats + x0);
    const float4 v11 = SCRATCH_LOAD4(buf, y1 * pitchFloats + x1);
    const float w00 = (1.0f - dx) * (1.0f - dy);
    const float w01 =         dx  * (1.0f - dy);
    const float w10 = (1.0f - dx) *         dy;
    const float w11 =         dx  *         dy;
    return (float2)(
        w00 * v00.x + w01 * v01.x + w10 * v10.x + w11 * v11.x,
        w00 * v00.y + w01 * v01.y + w10 * v10.y + w11 * v11.y);
}

// Thin pass 8 (cite Anime4K_Thin_HQ.glsl v3.2, "Warp"): for each output
// pixel, read the local flow direction from scratchB.xy, quasi-normalise
// (div by length + 0.01 to avoid div-by-zero), scale by per-pixel step
// times relstr = h/1080 * STRENGTH, subtract from the centre coord, and
// bilinear-sample the Y reference at the shifted position. The reference
// uses a single iteration (ITERATIONS=1); we follow the same.
//
// pSrcA holds the Y reference scratch at full output resolution
// (outW x outH). pSrcB holds the flow field at (flowW x flowH); for
// the HQ tier flowW=outW and flowH=outH so the flow read is at integer
// coords. For Fast / VeryFast the flow buffer is half / quarter sized
// and the flow read bilinear-interpolates.
__kernel void kernel_anime4k_thin_warp(
    __global uchar *restrict pDstY, const int dstPitch,
    __global const SCRATCH_ELEM_T *pSrcA, const int srcAPitchFloats,
    __global const SCRATCH_ELEM_T *pSrcB, const int srcBPitchFloats,
    const int outW, const int outH,
    const int flowW, const int flowH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float relstr = (float)outH / ANIME4K_REF_HEIGHT * ANIME4K_THIN_STRENGTH;

    // Map output-grid coord into flow-buffer coord space; collapses to
    // (ix, iy) when flowW=outW. The -0.5 / +0.5 adjustment puts both
    // grids on a half-integer texel-centre convention.
    const float fx_flow = ((float)ix + 0.5f) * (float)flowW / (float)outW - 0.5f;
    const float fy_flow = ((float)iy + 0.5f) * (float)flowH / (float)outH - 0.5f;
    const float2 flow = anime4k_bilinear_xy(pSrcB, srcBPitchFloats, flowW, flowH, fx_flow, fy_flow);
    const float dnx = flow.x;
    const float dny = flow.y;
    const float invlen = 1.0f / (native_sqrt(dnx * dnx + dny * dny) + 0.01f);
    const float ddx = dnx * invlen * relstr;
    const float ddy = dny * invlen * relstr;

    const float fx = (float)ix - ddx;
    const float fy = (float)iy - ddy;
    float result = anime4k_bilinear_x(pSrcA, srcAPitchFloats, outW, outH, fx, fy);
    result = clamp(result, 0.0f, 1.0f);

    __global Type *ptr = (__global Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)PIXEL_MAX + 0.5f);
}

// =========================================================================
// Darken / Thin speed-tier helpers (Fast, VeryFast)
// =========================================================================
//
// The Fast and VeryFast tiers run the same DoG / Sobel-warp algorithm
// as the HQ tier, but at half / quarter output resolution with sigma
// scaled accordingly (1.0 -> 0.5 -> 0.25 for darken, 2.0 -> 1.0 -> 0.5
// for thin). Three additional kernels handle the resolution boundary:
//
//   downsample_y           full-res Y -> work-res Y (uchar / ushort)
//                          via box average; box=2 for Fast, box=4
//                          for VeryFast. Used at the start of any
//                          non-HQ chain to produce the work-res input.
//   darken_smooth_y        like darken_apply_y but writes the smoothed
//                          DoG mask to a float4 scratch instead of
//                          fusing the apply step. Used at the end of
//                          the Fast / VeryFast darken chain.
//   darken_upsample_apply  reads the smoothed DoG mask at work res via
//                          bilinear interpolation and adds the scaled
//                          contribution to the full-res Y plane. The
//                          replacement for the HQ tier's fused apply
//                          when the scratch is at lower resolution.
//
// The thin chain doesn't need a smooth_y / upsample_apply pair because
// its final pass is already a warp at full res that reads the flow
// scratch via bilinear interpolation (see kernel_anime4k_thin_warp).

// Box-average downsample of the Y plane to a work-resolution copy.
// box=2 produces half-res (Fast), box=4 produces quarter-res (VeryFast).
// Output type matches the source (uchar / ushort, selected via the
// Type / bit_depth defines) so the downstream kernels can read it the
// same way they read the full-res Y plane.
__kernel void kernel_anime4k_downsample_y(
    __global uchar *restrict pDstY, const int dstPitch,
    __global const uchar *pSrcY, const int srcPitch,
    const int dstW, const int dstH,
    const int srcW, const int srcH,
    const int box) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= dstW || iy >= dstH) return;

    const int sx0 = ix * box;
    const int sy0 = iy * box;
    int acc = 0;
    int n   = 0;
    for (int dy = 0; dy < box; ++dy) {
        const int yy = sy0 + dy;
        if (yy >= srcH) break;
        for (int dx = 0; dx < box; ++dx) {
            const int xx = sx0 + dx;
            if (xx >= srcW) break;
            acc += (int)*(const __global Type *)(pSrcY + yy * srcPitch + xx * sizeof(Type));
            n   += 1;
        }
    }
    const Type out = (n > 0) ? (Type)((acc + n / 2) / n) : (Type)0;
    *(__global Type *)(pDstY + iy * dstPitch + ix * sizeof(Type)) = out;
}

// Darken Fast / VeryFast pass 4: vertical Gaussian smoothing of the
// DoG mask, with NO apply step (the apply happens later at full res
// via darken_upsample_apply). Equivalent to kernel_anime4k_darken_apply_y
// minus the final pDstY write; writes the smoothed mask back to a
// float4 scratch instead.
__kernel void kernel_anime4k_darken_smooth_y(
    __global SCRATCH_ELEM_T *restrict pDstB, const int dstPitchFloats,
    __global const SCRATCH_ELEM_T *pSrcA, const int srcPitchFloats,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float sigma = ANIME4K_DARKEN_SIGMA_REF * (float)outH / ANIME4K_REF_HEIGHT;
    const int radius = max((int)ceil(sigma * 2.0f), 1);
    const int r = min(radius, ANIME4K_GAUSS_MAX_RADIUS);

    float acc = 0.0f;
    float wsum = 0.0f;
    for (int dy = -r; dy <= r; ++dy) {
        const int yy = clamp(iy + dy, 0, outH - 1);
        const float w = anime4k_gauss_w((float)dy, sigma);
        acc  += SCRATCH_LOAD4(pSrcA, yy * srcPitchFloats + ix).x * w;
        wsum += w;
    }
    SCRATCH_STORE4((float4)(acc / wsum, 0.0f, 0.0f, 0.0f), pDstB, iy * dstPitchFloats + ix);
}

// Darken Fast / VeryFast pass 5: read the smoothed DoG mask at work
// resolution (workW x workH) via bilinear interpolation, then add
// STRENGTH * smoothed to the full-resolution Y plane. dog_dark is
// non-positive throughout, so STRENGTH > 0 darkens line interiors.
__kernel void kernel_anime4k_darken_upsample_apply(
    __global uchar *restrict pDstY, const int dstPitch,
    __global const SCRATCH_ELEM_T *pSrcMask, const int srcPitchFloats,
    const int outW, const int outH,
    const int workW, const int workH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    // Map output grid to work-resolution coord space (half-integer
    // texel-centre convention on both grids).
    const float fx = ((float)ix + 0.5f) * (float)workW / (float)outW - 0.5f;
    const float fy = ((float)iy + 0.5f) * (float)workH / (float)outH - 0.5f;
    const float smoothed = anime4k_bilinear_x(pSrcMask, srcPitchFloats, workW, workH, fx, fy);

    const float luma = anime4k_read_y_norm(pDstY, dstPitch, ix, iy);
    float result = luma + smoothed * ANIME4K_DARKEN_STRENGTH;
    result = clamp(result, 0.0f, 1.0f);

    __global Type *ptr = (__global Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)PIXEL_MAX + 0.5f);
}

// =========================================================================
// Bilateral denoise (Mean / Median / Mode), Y-only single-pass kernels.
// =========================================================================
//
// Three kernels implementing the algorithms from
//   Anime4K_Denoise_Bilateral_{Mean, Median, Mode}.glsl v3.2.
// All three share the same per-tap bilateral weight: spatial Gaussian
// over pixel distance, intensity Gaussian over luma difference with
// the intensity sigma modulated by the centre luma via a power curve.
// The variants differ only in the reduction step:
//   Mean   -- weighted average
//   Median -- weighted median by luma ordering
//   Mode   -- Parzen-smoothed weighted mode
//
// The single-pass form (Y-only, no separate luma-extraction pass) is
// possible because the input is already a single-channel luma plane.
// The kernels read luma values via a float scratch buffer holding the
// post-darken / post-thin Y plane in [0, 1] -- typically m_scratchA.x
// populated by kernel_anime4k_thin_copy_y_to_ref before dispatch.
//
// Build-time defines (default to reference shader's published values
// so the source compiles stand-alone for inspection):
//   ANIME4K_DENOISE_SPATIAL_SIGMA    spatial sigma in pixel units (1.0)
//   ANIME4K_DENOISE_INTENSITY_SIGMA  range sigma base scale     (0.1)
//   ANIME4K_DENOISE_INTENSITY_CURVE  power curve on centre luma (1.0)
//   ANIME4K_DENOISE_HIST_REG         Parzen-window width for mode (0.0)
//                                    Median uses it when > 0, ignored
//                                    when 0 (the compiler eliminates
//                                    the regularisation block).
//   ANIME4K_DENOISE_MEAN_KHALF       half-kernel for mean       (2)
//   ANIME4K_DENOISE_MMM_KHALF        half-kernel for median/mode (1)
//   ANIME4K_DENOISE_MMM_KLEN         (2*MMM_KHALF+1)^2          (9)

#ifndef ANIME4K_DENOISE_SPATIAL_SIGMA
#define ANIME4K_DENOISE_SPATIAL_SIGMA   1.0f
#endif
#ifndef ANIME4K_DENOISE_INTENSITY_SIGMA
#define ANIME4K_DENOISE_INTENSITY_SIGMA 0.1f
#endif
#ifndef ANIME4K_DENOISE_INTENSITY_CURVE
#define ANIME4K_DENOISE_INTENSITY_CURVE 1.0f
#endif
#ifndef ANIME4K_DENOISE_HIST_REG
#define ANIME4K_DENOISE_HIST_REG        0.0f
#endif
#ifndef ANIME4K_DENOISE_MEAN_KHALF
#define ANIME4K_DENOISE_MEAN_KHALF      2
#endif
#ifndef ANIME4K_DENOISE_MMM_KHALF
#define ANIME4K_DENOISE_MMM_KHALF       1
#endif
#ifndef ANIME4K_DENOISE_MMM_KLEN
#define ANIME4K_DENOISE_MMM_KLEN        9
#endif

// Shared bilateral weight: combined spatial Gaussian (on integer
// pixel offset distance) and range Gaussian (on luma difference).
// Matches gaussian(length(ipos), ss, 0) * gaussian(v_n, is, vc) from
// the reference shader. sigma_i must be > 0; the caller is responsible
// for clamping when the power curve could collapse it to zero.
static inline float anime4k_bilateral_weight(
    int dx, int dy,
    float vc, float vn,
    float sigma_s, float sigma_i) {
    const float ds = (float)(dx * dx + dy * dy);
    const float dr = (vn - vc) / sigma_i;
    return native_exp(-0.5f * ds / (sigma_s * sigma_s))
         * native_exp(-0.5f * dr * dr);
}

// Reads the per-pixel centre luma and derives the brightness-modulated
// intensity sigma the same way the reference shader does.
static inline float anime4k_denoise_sigma_i(float vc) {
    const float base = pow(vc + 0.0001f, ANIME4K_DENOISE_INTENSITY_CURVE)
                     * ANIME4K_DENOISE_INTENSITY_SIGMA;
    return max(base, 1e-6f);
}

// Bilateral-mean denoise: classic Tomasi-Manduchi weighted average.
// Cite Anime4K_Denoise_Bilateral_Mean.glsl v3.2.
__kernel void kernel_anime4k_denoise_mean(
    __global uchar *restrict pDstY, const int dstPitch,
    __global const SCRATCH_ELEM_T *pSrcRef, const int srcPitchFloats,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float vc = SCRATCH_LOAD4(pSrcRef, iy * srcPitchFloats + ix).x;
    const float sigma_s = ANIME4K_DENOISE_SPATIAL_SIGMA;
    const float sigma_i = anime4k_denoise_sigma_i(vc);

    float sum = 0.0f;
    float n   = 0.0f;
    // `half` is an OpenCL reserved type name (FP16); use `khalf` instead.
    const int khalf = ANIME4K_DENOISE_MEAN_KHALF;
    for (int dy = -khalf; dy <= khalf; ++dy) {
        const int yy = clamp(iy + dy, 0, outH - 1);
        for (int dx = -khalf; dx <= khalf; ++dx) {
            const int xx = clamp(ix + dx, 0, outW - 1);
            const float v = SCRATCH_LOAD4(pSrcRef, yy * srcPitchFloats + xx).x;
            const float w = anime4k_bilateral_weight(dx, dy, vc, v, sigma_s, sigma_i);
            sum += w * v;
            n   += w;
        }
    }

    float result = (n > 1e-9f) ? (sum / n) : vc;
    result = clamp(result, 0.0f, 1.0f);
    __global Type *ptr = (__global Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)PIXEL_MAX + 0.5f);
}

// Bilateral-median denoise: weighted median by luma ordering with
// optional Parzen-smoothing of weights when ANIME4K_DENOISE_HIST_REG > 0.
// Cite Anime4K_Denoise_Bilateral_Median.glsl v3.2.
__kernel void kernel_anime4k_denoise_median(
    __global uchar *restrict pDstY, const int dstPitch,
    __global const SCRATCH_ELEM_T *pSrcRef, const int srcPitchFloats,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float vc = SCRATCH_LOAD4(pSrcRef, iy * srcPitchFloats + ix).x;
    const float sigma_s = ANIME4K_DENOISE_SPATIAL_SIGMA;
    const float sigma_i = anime4k_denoise_sigma_i(vc);

    float vs[ANIME4K_DENOISE_MMM_KLEN];
    float ws[ANIME4K_DENOISE_MMM_KLEN];
    float total_w = 0.0f;

    // `half` is an OpenCL reserved type name (FP16); use `khalf` instead.
    const int khalf = ANIME4K_DENOISE_MMM_KHALF;
    int idx = 0;
    for (int dy = -khalf; dy <= khalf; ++dy) {
        const int yy = clamp(iy + dy, 0, outH - 1);
        for (int dx = -khalf; dx <= khalf; ++dx) {
            const int xx = clamp(ix + dx, 0, outW - 1);
            const float v = SCRATCH_LOAD4(pSrcRef, yy * srcPitchFloats + xx).x;
            const float w = anime4k_bilateral_weight(dx, dy, vc, v, sigma_s, sigma_i);
            vs[idx] = v;
            ws[idx] = w;
            total_w += w;
            idx += 1;
        }
    }

    // Optional Parzen-smoothing pass over the per-sample weights.
    // ANIME4K_DENOISE_HIST_REG is a JIT-time constant; the compiler
    // dead-code-eliminates this whole block when the value is 0.0f
    // (the median default), matching the reference shader's
    // "if (HISTOGRAM_REGULARIZATION > 0.0) { ... }" gate.
    const float reg = ANIME4K_DENOISE_HIST_REG;
    if (reg > 0.0f) {
        float ws_reg[ANIME4K_DENOISE_MMM_KLEN];
        for (int i = 0; i < ANIME4K_DENOISE_MMM_KLEN; ++i) ws_reg[i] = 0.0f;
        total_w = 0.0f;
        const float inv_reg = 1.0f / reg;
        for (int i = 0; i < ANIME4K_DENOISE_MMM_KLEN; ++i) {
            ws_reg[i] += ws[i];   // d=0 case: gaussian(0, reg, 0) = 1
            for (int j = i + 1; j < ANIME4K_DENOISE_MMM_KLEN; ++j) {
                const float d = (vs[j] - vs[i]) * inv_reg;
                const float g = native_exp(-0.5f * d * d);
                ws_reg[j] += g * ws[i];
                ws_reg[i] += g * ws[j];
            }
            total_w += ws_reg[i];
        }
        for (int i = 0; i < ANIME4K_DENOISE_MMM_KLEN; ++i) ws[i] = ws_reg[i];
    }

    // Find the weighted median: the sample where cumulative weight
    // below is <= 50% AND cumulative weight above is also <= 50%.
    // Iterate in storage order; first match wins (matches reference).
    float median = vc;
    if (total_w > 1e-9f) {
        const float inv_tot = 1.0f / total_w;
        for (int i = 0; i < ANIME4K_DENOISE_MMM_KLEN; ++i) {
            float w_above = 0.0f;
            float w_below = 0.0f;
            for (int j = 0; j < ANIME4K_DENOISE_MMM_KLEN; ++j) {
                if (vs[j] > vs[i]) w_above += ws[j];
                else if (vs[j] < vs[i]) w_below += ws[j];
            }
            if ((total_w - w_above) * inv_tot >= 0.5f
             && w_below * inv_tot <= 0.5f) {
                median = vs[i];
                break;
            }
        }
    }

    float result = clamp(median, 0.0f, 1.0f);
    __global Type *ptr = (__global Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)PIXEL_MAX + 0.5f);
}

// Bilateral-mode denoise: Parzen-smoothed weighted mode over the luma
// dimension. Unconditional regularisation (matches the reference,
// which does not gate the smoothing pass). ANIME4K_DENOISE_HIST_REG
// is clamped to a small positive minimum to avoid divide-by-zero when
// the user passes 0; the reference's published default is 0.2.
// Cite Anime4K_Denoise_Bilateral_Mode.glsl v3.1.
__kernel void kernel_anime4k_denoise_mode(
    __global uchar *restrict pDstY, const int dstPitch,
    __global const SCRATCH_ELEM_T *pSrcRef, const int srcPitchFloats,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float vc = SCRATCH_LOAD4(pSrcRef, iy * srcPitchFloats + ix).x;
    const float sigma_s = ANIME4K_DENOISE_SPATIAL_SIGMA;
    const float sigma_i = anime4k_denoise_sigma_i(vc);
    const float reg     = max(ANIME4K_DENOISE_HIST_REG, 1e-6f);
    const float inv_reg = 1.0f / reg;

    float vs[ANIME4K_DENOISE_MMM_KLEN];
    float ws[ANIME4K_DENOISE_MMM_KLEN];
    float ws_reg[ANIME4K_DENOISE_MMM_KLEN];

    // `half` is an OpenCL reserved type name (FP16); use `khalf` instead.
    const int khalf = ANIME4K_DENOISE_MMM_KHALF;
    int idx = 0;
    for (int dy = -khalf; dy <= khalf; ++dy) {
        const int yy = clamp(iy + dy, 0, outH - 1);
        for (int dx = -khalf; dx <= khalf; ++dx) {
            const int xx = clamp(ix + dx, 0, outW - 1);
            const float v = SCRATCH_LOAD4(pSrcRef, yy * srcPitchFloats + xx).x;
            const float w = anime4k_bilateral_weight(dx, dy, vc, v, sigma_s, sigma_i);
            vs[idx] = v;
            ws[idx] = w;
            ws_reg[idx] = 0.0f;
            idx += 1;
        }
    }

    // Spread each sample's weight across nearby samples in the luma
    // dimension via a Gaussian of width reg. The off-diagonal terms
    // are symmetric so we accumulate both halves in one inner loop.
    for (int i = 0; i < ANIME4K_DENOISE_MMM_KLEN; ++i) {
        ws_reg[i] += ws[i];   // d=0 case
        for (int j = i + 1; j < ANIME4K_DENOISE_MMM_KLEN; ++j) {
            const float d = (vs[j] - vs[i]) * inv_reg;
            const float g = native_exp(-0.5f * d * d);
            ws_reg[j] += g * ws[i];
            ws_reg[i] += g * ws[j];
        }
    }

    // Argmax over the smoothed weights.
    float best_v = vc;
    float best_w = 0.0f;
    for (int i = 0; i < ANIME4K_DENOISE_MMM_KLEN; ++i) {
        if (ws_reg[i] > best_w) {
            best_w = ws_reg[i];
            best_v = vs[i];
        }
    }

    float result = clamp(best_v, 0.0f, 1.0f);
    __global Type *ptr = (__global Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)PIXEL_MAX + 0.5f);
}

// =========================================================================
// DoG-family modes (mode=dog_sharpen / dog / dtd)
// =========================================================================
//
// Three new upscale/sharpen variants based on Difference-of-Gaussians
// with anti-ringing via 3-tap min/max clamp:
//   mode=dog_sharpen  (1x unsharp + minmax + soft threshold)
//   mode=dog          (2x DoG upscale + minmax, no soft threshold)
//   mode=dtd          (2x Darken -> Thin -> Deblur fused chain)
//
// All three share two intermediate kernels that compute a 2D Gaussian
// (7-tap binomial, Pascal's-triangle weights) and a 3x3 local min/max
// in a single separable horizontal-then-vertical pass:
//
//   dog_kernel_x   horizontal 7-tap Gauss + 3-tap horizontal minmax,
//                  Type input -> float4 .xyz output (.x=Gauss, .y=min,
//                  .z=max). Operates at the resolution of the source
//                  Y-plane the caller passes in -- 1x for dog_sharpen
//                  and dog, 2x for the dtd stage-C deblur pass.
//   dog_kernel_y   vertical completion of both the Gauss and the
//                  minmax (min of horizontal mins, max of horizontal
//                  maxes => true 2D min/max over a 3x3 window).
//
// The 7-tap binomial weights [0.06136, 0.24477, 0.38774, 0.24477,
// 0.06136] are a fixed Pascal's-triangle approximation -- they do not
// scale with output resolution like the darken/thin sigmas do. This
// matches the reference shaders' design.
//
// Apply kernels diverge:
//   dog_apply_soft     soft-threshold sharpen + minmax clamp; reads
//                      a separate luma source (pSrcLuma), writes
//                      pDstY at the same resolution. Used by
//                      mode=dog_sharpen (pSrcLuma=pInputPlaneY) and
//                      by mode=dtd stage C (pSrcLuma=pDstY for an
//                      in-place 2x sharpen).
//   dog_apply_upscale  simple unsharp + minmax clamp; reads source
//                      luma at 2x output position via the hardware
//                      bilinear sampler on srcImage, and the 1x
//                      Gauss+minmax scratch via manual bilinear.
//                      Used only by mode=dog.

// Trivial Type->Type copy of a Y plane region. Used by mode=dtd to
// stage pInputPlaneY into a writable scratch before stage A modifies
// it in place. Iterates at the caller-supplied work dimensions.
__kernel void kernel_anime4k_copy_y_to_y(
    __global uchar *restrict pDstY, const int dstPitch,
    __global const uchar *pSrcY, const int srcPitch,
    const int width, const int height) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const Type v = *(const __global Type *)(pSrcY + iy * srcPitch + ix * sizeof(Type));
    *(__global Type *)(pDstY + iy * dstPitch + ix * sizeof(Type)) = v;
}

// Horizontal 7-tap binomial Gauss + 3-tap minmax. Output .x = Gauss,
// .y = local min, .z = local max, .w = unused.
// Cite Anime4K_Deblur_DoG.glsl v3.2 (Kernel-X). Same pattern is used by
// Anime4K_Upscale_DoG_x2.glsl and the stage-C pass of DTD-x2.
__kernel void kernel_anime4k_dog_kernel_x(
    __global float4 *restrict pDstA, const int dstPitchFloats,
    __global const uchar *pSrcY, const int srcPitch,
    const int width, const int height) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const int x_m2 = clamp(ix - 2, 0, width - 1);
    const int x_m1 = clamp(ix - 1, 0, width - 1);
    const int x_p1 = clamp(ix + 1, 0, width - 1);
    const int x_p2 = clamp(ix + 2, 0, width - 1);

    const float v_m2 = anime4k_read_y_norm(pSrcY, srcPitch, x_m2, iy);
    const float v_m1 = anime4k_read_y_norm(pSrcY, srcPitch, x_m1, iy);
    const float v_c  = anime4k_read_y_norm(pSrcY, srcPitch, ix,   iy);
    const float v_p1 = anime4k_read_y_norm(pSrcY, srcPitch, x_p1, iy);
    const float v_p2 = anime4k_read_y_norm(pSrcY, srcPitch, x_p2, iy);

    const float g  = (v_m2 + v_p2) * 0.06136f
                   + (v_m1 + v_p1) * 0.24477f
                   +  v_c          * 0.38774f;
    const float lo = fmin(fmin(v_m1, v_c), v_p1);
    const float hi = fmax(fmax(v_m1, v_c), v_p1);

    pDstA[iy * dstPitchFloats + ix] = (float4)(g, lo, hi, 0.0f);
}

// Vertical completion of the DoG kernel. Reads the horizontal Gauss
// in .x, horizontal mins in .y, horizontal maxes in .z; emits the
// fully-2D Gauss in .x, 2D min in .y, 2D max in .z. Source and
// destination must be distinct buffers -- this kernel reads multiple
// rows above and below its own row.
__kernel void kernel_anime4k_dog_kernel_y(
    __global float4 *restrict pDstB, const int dstPitchFloats,
    __global const float4 *pSrcA, const int srcPitchFloats,
    const int width, const int height) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const int y_m2 = clamp(iy - 2, 0, height - 1);
    const int y_m1 = clamp(iy - 1, 0, height - 1);
    const int y_p1 = clamp(iy + 1, 0, height - 1);
    const int y_p2 = clamp(iy + 2, 0, height - 1);

    const float4 v_m2 = pSrcA[y_m2 * srcPitchFloats + ix];
    const float4 v_m1 = pSrcA[y_m1 * srcPitchFloats + ix];
    const float4 v_c  = pSrcA[iy   * srcPitchFloats + ix];
    const float4 v_p1 = pSrcA[y_p1 * srcPitchFloats + ix];
    const float4 v_p2 = pSrcA[y_p2 * srcPitchFloats + ix];

    const float g  = (v_m2.x + v_p2.x) * 0.06136f
                   + (v_m1.x + v_p1.x) * 0.24477f
                   +  v_c.x            * 0.38774f;
    // 2D min = vertical min of horizontal mins; 2D max similarly.
    const float lo = fmin(fmin(v_m1.y, v_c.y), v_p1.y);
    const float hi = fmax(fmax(v_m1.z, v_c.z), v_p1.z);

    pDstB[iy * dstPitchFloats + ix] = (float4)(g, lo, hi, 0.0f);
}

// Apply: soft-threshold unsharp + minmax clamp at the source
// resolution. Used by mode=dog_sharpen (1x in-place via pSrcLuma =
// pInputPlaneY and pDstY = output) and by mode=dtd stage C (2x
// in-place via pSrcLuma == pDstY).
__kernel void kernel_anime4k_dog_apply_soft(
    __global uchar *restrict pDstY, const int dstPitch,
    __global const uchar *pSrcLuma, const int srcLumaPitch,
    __global const float4 *pSrcMM, const int srcMMPitchFloats,
    const int width, const int height) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const float luma = anime4k_read_y_norm(pSrcLuma, srcLumaPitch, ix, iy);
    const float4 mm  = pSrcMM[iy * srcMMPitchFloats + ix];
    const float blur = mm.x;
    const float lo   = mm.y;
    const float hi   = mm.z;

    const float c = (luma - blur) * ANIME4K_DOG_STRENGTH;
    const float t_range = ANIME4K_DOG_BLUR_THRESHOLD - ANIME4K_DOG_NOISE_THRESHOLD;

    // Soft-threshold: only sharpen the mid-frequency band, suppress
    // both noise (|c| <= NOISE_THRESHOLD) and already-sharp edges
    // (|c| above BLUR_THRESHOLD goes through a power-curve roll-off).
    float c_t;
    const float c_abs = fabs(c);
    if (c_abs > ANIME4K_DOG_NOISE_THRESHOLD) {
        float t = (c_abs - ANIME4K_DOG_NOISE_THRESHOLD) / t_range;
        t = pow(t, ANIME4K_DOG_BLUR_CURVE);
        t = t * t_range + ANIME4K_DOG_NOISE_THRESHOLD;
        c_t = t * sign(c);
    } else {
        c_t = c;
    }

    // Minmax clamp: bounds the sharpened value by the local 3x3 luma
    // range, which prevents the unsharp from creating ringing halos.
    float result = clamp(luma + c_t, lo, hi);
    result = clamp(result, 0.0f, 1.0f);

    __global Type *ptr = (__global Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)PIXEL_MAX + 0.5f);
}

// Apply: simple unsharp + minmax clamp with 2x upscale. Reads source
// luma via the hardware bilinear sampler on srcImage (same convention
// as kernel_anime4k_apply), and the 1x Gauss+minmax scratch via
// manual bilinear. No soft threshold -- the upstream Upscale_DoG_x2
// shader's apply pass deliberately skips it. Used by mode=dog.
__kernel void kernel_anime4k_dog_apply_upscale(
    __global uchar *restrict pDstY, const int dstPitch,
    __read_only image2d_t srcImage,
    __global const float4 *pSrcMM, const int srcMMPitchFloats,
    const int srcW, const int srcH,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float2 p = anime4k_src_coord(ix, iy, outW, outH);
    const float luma = read_imagef(srcImage, anime4k_src_sampler, p).x;

    // Map 2x output coord to 1x scratch pixel coord, then bilinear
    // sample .xyz from the float4 scratch.
    const float fx_1x = ((float)ix + 0.5f) * (float)srcW / (float)outW - 0.5f;
    const float fy_1x = ((float)iy + 0.5f) * (float)srcH / (float)outH - 0.5f;

    const int x0 = clamp((int)floor(fx_1x), 0, srcW - 1);
    const int y0 = clamp((int)floor(fy_1x), 0, srcH - 1);
    const int x1 = clamp(x0 + 1, 0, srcW - 1);
    const int y1 = clamp(y0 + 1, 0, srcH - 1);
    const float dx = fx_1x - (float)x0;
    const float dy = fy_1x - (float)y0;
    const float w00 = (1.0f - dx) * (1.0f - dy);
    const float w01 =         dx  * (1.0f - dy);
    const float w10 = (1.0f - dx) *         dy;
    const float w11 =         dx  *         dy;
    const float4 m00 = pSrcMM[y0 * srcMMPitchFloats + x0];
    const float4 m01 = pSrcMM[y0 * srcMMPitchFloats + x1];
    const float4 m10 = pSrcMM[y1 * srcMMPitchFloats + x0];
    const float4 m11 = pSrcMM[y1 * srcMMPitchFloats + x1];
    const float blur = w00 * m00.x + w01 * m01.x + w10 * m10.x + w11 * m11.x;
    const float lo   = w00 * m00.y + w01 * m01.y + w10 * m10.y + w11 * m11.y;
    const float hi   = w00 * m00.z + w01 * m01.z + w10 * m10.z + w11 * m11.z;

    const float c = (luma - blur) * ANIME4K_DOG_STRENGTH;
    float result = clamp(luma + c, lo, hi);
    result = clamp(result, 0.0f, 1.0f);

    __global Type *ptr = (__global Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)PIXEL_MAX + 0.5f);
}

// DTD stage-B warp: per-output-pixel warped lookup into the 1x luma
// source, with the flow field also at 1x. Unlike kernel_anime4k_thin_warp
// (which assumes yRef and the output are at the same resolution),
// the DTD warp upscales 1x -> 2x by sampling the 1x source at the
// position corresponding to the 2x output pixel, displaced by the
// flow vector. relstr uses the SOURCE height (matching the shader's
// HOOKED_size.y = MAIN at 1x), so the per-pixel displacement is in
// 1x-source-pixel units.
__kernel void kernel_anime4k_dtd_warp(
    __global uchar *restrict pDstY, const int dstPitch,
    __global const uchar *pSrcLuma, const int srcLumaPitch,
    __global const SCRATCH_ELEM_T *pSrcFlow, const int srcFlowPitchFloats,
    const int srcW, const int srcH,
    const int outW, const int outH) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= outW || iy >= outH) return;

    const float fx_1x = ((float)ix + 0.5f) * (float)srcW / (float)outW - 0.5f;
    const float fy_1x = ((float)iy + 0.5f) * (float)srcH / (float)outH - 0.5f;

    const float2 flow = anime4k_bilinear_xy(pSrcFlow, srcFlowPitchFloats, srcW, srcH, fx_1x, fy_1x);
    const float dnx = flow.x;
    const float dny = flow.y;
    const float invlen = 1.0f / (native_sqrt(dnx * dnx + dny * dny) + 0.01f);

    // DTD's thin uses the source height for relstr because the warp's
    // HOOKED in the shader is MAIN at 1x.
    const float relstr = (float)srcH / ANIME4K_REF_HEIGHT * ANIME4K_THIN_STRENGTH;
    const float ddx = dnx * invlen * relstr;
    const float ddy = dny * invlen * relstr;

    const float fx = fx_1x - ddx;
    const float fy = fy_1x - ddy;

    // Manual bilinear sample of the Type-format 1x luma scratch.
    const int x0 = clamp((int)floor(fx), 0, srcW - 1);
    const int y0 = clamp((int)floor(fy), 0, srcH - 1);
    const int x1 = clamp(x0 + 1, 0, srcW - 1);
    const int y1 = clamp(y0 + 1, 0, srcH - 1);
    const float dx = fx - (float)x0;
    const float dy = fy - (float)y0;
    const float v00 = anime4k_read_y_norm(pSrcLuma, srcLumaPitch, x0, y0);
    const float v01 = anime4k_read_y_norm(pSrcLuma, srcLumaPitch, x1, y0);
    const float v10 = anime4k_read_y_norm(pSrcLuma, srcLumaPitch, x0, y1);
    const float v11 = anime4k_read_y_norm(pSrcLuma, srcLumaPitch, x1, y1);
    float result = (1.0f - dx) * (1.0f - dy) * v00
                 +         dx  * (1.0f - dy) * v01
                 + (1.0f - dx) *         dy  * v10
                 +         dx  *         dy  * v11;
    result = clamp(result, 0.0f, 1.0f);

    __global Type *ptr = (__global Type *)(pDstY + iy * dstPitch + ix * sizeof(Type));
    ptr[0] = (Type)(result * (float)PIXEL_MAX + 0.5f);
}

// Clamp_Highlights post-process (bloc97 Anime4K_Clamp_Highlights, MIT).
// 3 kernels: vertical-max (shared), then horizontal-max + apply for the
// Y-only pipeline. STATSMAX is computed at source res from the anime4k input;
// the apply pass bilinear-upsamples STATSMAX to output res, then clamps each
// output pixel's luma at the source's local 5x5 max.
//
// Shared by all five: separable max-dilation over a 5x5 neighbourhood
// (radius=2). Edge handling: clamp-to-edge.

// Sample STATSMAX (1-ch fp16) at fractional position via bilinear.
static inline float anime4k_clamp_bilinear_stats(
    __global const half *pStatsMax, const int statsStride,
    const int srcW, const int srcH,
    const float sx_f, const float sy_f) {
    const int x0 = clamp((int)floor(sx_f), 0, srcW - 1);
    const int y0 = clamp((int)floor(sy_f), 0, srcH - 1);
    const int x1 = min(x0 + 1, srcW - 1);
    const int y1 = min(y0 + 1, srcH - 1);
    const float fx = sx_f - (float)x0;
    const float fy = sy_f - (float)y0;
    const float v00 = vload_half(y0 * statsStride + x0, pStatsMax);
    const float v01 = vload_half(y0 * statsStride + x1, pStatsMax);
    const float v10 = vload_half(y1 * statsStride + x0, pStatsMax);
    const float v11 = vload_half(y1 * statsStride + x1, pStatsMax);
    return (1.0f - fx) * (1.0f - fy) * v00 + fx * (1.0f - fy) * v01
         + (1.0f - fx) * fy         * v10 + fx * fy         * v11;
}

// Shared vertical-max pass: read 1-ch STATSMAX_h, write 1-ch STATSMAX.
// Combined with either horizontal-max pass it yields a separable 5x5 max
// dilation (kernel radius = 2, 5 taps per axis).
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void kernel_anime4k_clamp_v_max(
    __global half *restrict pDst, const int dstStride,
    __global const half *pSrc,    const int srcStride,
    const int W, const int H) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W || y >= H) return;
    float m = -1.0e30f;
    for (int i = -2; i <= 2; ++i) {
        const int yy = clamp(y + i, 0, H - 1);
        m = max(m, vload_half(yy * srcStride + x, pSrc));
    }
    vstore_half(m, y * dstStride + x, pDst);
}

// Y-only horizontal-max: read native pixel-format Y plane (Type is uchar
// or ushort depending on output bit depth; the macro is set at OpenCL
// compile time via the build options). Normalises Y to [0, 1] via the
// existing anime4k_read_y_norm helper so STATSMAX storage is always fp16
// regardless of source bit depth. Output is 1-ch fp16, source spatial.
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void kernel_anime4k_clamp_h_max_y(
    __global half *restrict pDst,    const int dstStride,
    __global const uchar *pSrcY,     const int srcPitch,
    const int W, const int H) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W || y >= H) return;
    float m = -1.0e30f;
    for (int i = -2; i <= 2; ++i) {
        const int xx = clamp(x + i, 0, W - 1);
        m = max(m, anime4k_read_y_norm(pSrcY, srcPitch, xx, y));
    }
    vstore_half(m, y * dstStride + x, pDst);
}

// Y-only apply: in-place clamp on native pixel-format Y plane. Reads Y as
// Type, normalises via anime4k_read_y_norm, bilinear-samples STATSMAX,
// min-caps, denormalises, writes back as Type. No broadcast (single
// channel). Only writes when curY > statsMax (skip the round-trip on the
// no-op path to preserve byte-exactness on unaffected pixels).
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void kernel_anime4k_clamp_apply_y(
    __global uchar *restrict pYOut,  const int dstPitch,
    __global const half *pStatsMax,  const int statsStride,
    const int dstW, const int dstH,
    const int srcW, const int srcH) {
    const int dx = get_global_id(0);
    const int dy = get_global_id(1);
    if (dx >= dstW || dy >= dstH) return;
    const float sx_f = ((float)dx + 0.5f) * (float)srcW / (float)dstW - 0.5f;
    const float sy_f = ((float)dy + 0.5f) * (float)srcH / (float)dstH - 0.5f;
    const float statsMax = anime4k_clamp_bilinear_stats(
        pStatsMax, statsStride, srcW, srcH, sx_f, sy_f);
    const float curY = anime4k_read_y_norm(pYOut, dstPitch, dx, dy);
    if (curY > statsMax) {
        __global Type *ptr = (__global Type *)(pYOut + dy * dstPitch + dx * sizeof(Type));
        ptr[0] = (Type)(statsMax * (float)PIXEL_MAX + 0.5f);
    }
}

// PixelClipper anti-ringing (Joao Chrisostomo / Artoriuz, MIT). Post-process:
// for each output luma pixel, clamp it to the [min, max] of the 2x2 SOURCE
// pixels surrounding its source position, then mix by `strength`. Removes the
// overshoot / undershoot ringing a sharp upscale introduces, on BOTH sides
// (unlike clamp_highlights, which clamps the high side only). antiring=<0..1>;
// 0 disables. Reads source + output Y directly (no scratch). Cite
// github.com/Artoriuz/glsl-pixel-clipper.
__kernel void kernel_anime4k_antiring_y(
    __global uchar *restrict pYOut, const int dstPitch,
    __global const uchar *pSrcY,    const int srcPitch,
    const int dstW, const int dstH,
    const int srcW, const int srcH,
    const float strength) {
    const int dx = get_global_id(0);
    const int dy = get_global_id(1);
    if (dx >= dstW || dy >= dstH) return;
    // output pixel -> source position (pixel centres), same mapping as clamp_apply_y
    const float sx_f = ((float)dx + 0.5f) * (float)srcW / (float)dstW - 0.5f;
    const float sy_f = ((float)dy + 0.5f) * (float)srcH / (float)dstH - 0.5f;
    const int fx = (int)floor(sx_f);
    const int fy = (int)floor(sy_f);
    const int x0 = clamp(fx,     0, srcW - 1);
    const int x1 = clamp(fx + 1, 0, srcW - 1);
    const int y0 = clamp(fy,     0, srcH - 1);
    const int y1 = clamp(fy + 1, 0, srcH - 1);
    const float a = anime4k_read_y_norm(pSrcY, srcPitch, x0, y0);
    const float b = anime4k_read_y_norm(pSrcY, srcPitch, x1, y0);
    const float c = anime4k_read_y_norm(pSrcY, srcPitch, x0, y1);
    const float d = anime4k_read_y_norm(pSrcY, srcPitch, x1, y1);
    const float lo = fmin(fmin(a, b), fmin(c, d));
    const float hi = fmax(fmax(a, b), fmax(c, d));
    const float cur = anime4k_read_y_norm(pYOut, dstPitch, dx, dy);
    const float outv = cur + (clamp(cur, lo, hi) - cur) * strength;
    __global Type *ptr = (__global Type *)(pYOut + dy * dstPitch + dx * sizeof(Type));
    ptr[0] = (Type)(clamp(outv, 0.0f, 1.0f) * (float)PIXEL_MAX + 0.5f);
}

