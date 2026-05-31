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
// Camera-shake stabilisation -- Phase Correlation
// (Kuglin & Hines 1975, "The phase correlation image alignment method",
//  Proc. Int. Conf. on Cybernetics and Society).
//
// Hand-rolled radix-2 FFT in OpenCL, following the same shape as
// rigaya's rgy_filter_denoise_fft3d.cl but as a whole-frame 2D FFT
// (FFT_N x FFT_N) instead of small tiled blocks. Algorithm: 2D FFT
// done as row-FFT then column-FFT via the same kernel with a stride
// argument. The detected inter-frame translation is recovered from
// the peak of the inverse FFT of the normalised cross-spectrum.
//
// Build-time defines (set via -D from rgy_filter_stab.cpp):
//   Type        : uchar (8-bit) or ushort (>8-bit)
//   bit_depth   : source bit depth (8, 10, 12, 14, 16)
//   max_val     : (1 << bit_depth) - 1
//   FFT_N       : the FFT size; must be 256 for the bitrev trick below
//   FFT_LOG2_N  : log2(FFT_N) -- must be 8 to match the 8-bit bitrev

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// 8-bit bit-reversal (matches FFT_N = 256). Three nibble-swap / pair-
// swap / bit-swap masks: same trick used in many DSP libraries.
inline int stab_bitrev8(int x) {
    x = ((x & 0xF0) >> 4) | ((x & 0x0F) << 4);
    x = ((x & 0xCC) >> 2) | ((x & 0x33) << 2);
    x = ((x & 0xAA) >> 1) | ((x & 0x55) << 1);
    return x;
}

// ---------------------------------------------------------------------------
// stab_luma_downsample: source luma plane -> FFT_N x FFT_N complex buffer,
// with imag = 0. Each output pixel is the box-filter average of the source
// rectangle it covers. Pixels outside the source clamp to the nearest edge
// implicitly (we just stop scanning at srcW / srcH).
//
// Why box-filter average instead of nearest-neighbour decimation: aliasing
// from coherent line structures in animation cels would otherwise create
// false correlation peaks at fractional pixel offsets.
// ---------------------------------------------------------------------------
__kernel void stab_luma_downsample(
    const __global uchar *pSrc, const int srcPitch,
    const int srcW, const int srcH,
    __global float2 *pDst
) {
    const int ox = get_global_id(0);
    const int oy = get_global_id(1);
    if (ox >= FFT_N || oy >= FFT_N) return;

    const float sx0_f = (float)ox       * (float)srcW * (1.0f / (float)FFT_N);
    const float sx1_f = (float)(ox + 1) * (float)srcW * (1.0f / (float)FFT_N);
    const float sy0_f = (float)oy       * (float)srcH * (1.0f / (float)FFT_N);
    const float sy1_f = (float)(oy + 1) * (float)srcH * (1.0f / (float)FFT_N);

    const int sx0 = (int)sx0_f;
    const int sy0 = (int)sy0_f;
    int sx1 = (int)sx1_f; if (sx1 <= sx0) sx1 = sx0 + 1;
    int sy1 = (int)sy1_f; if (sy1 <= sy0) sy1 = sy0 + 1;

    float sum = 0.0f;
    int   n   = 0;
    for (int sy = sy0; sy < sy1 && sy < srcH; sy++) {
        for (int sx = sx0; sx < sx1 && sx < srcW; sx++) {
            const Type p = *(const __global Type *)(pSrc + sy * srcPitch + sx * sizeof(Type));
            sum += (float)p;
            n++;
        }
    }

    const float v = (n > 0) ? (sum * (1.0f / (float)n) * (1.0f / (float)max_val)) : 0.0f;
    pDst[oy * FFT_N + ox] = (float2)(v, 0.0f);
}

// ---------------------------------------------------------------------------
// stab_fft_1d: one 2D FFT pass. Each workgroup handles one 1D FFT of size
// FFT_N. The same kernel does rows and columns; the caller picks the axis
// by setting `stride`:
//
//   stride = 1      : workgroup `wg` operates on row `wg`.
//                     Reads input[wg*FFT_N + i] for i = 0 .. FFT_N-1.
//                     Memory accesses are coalesced (consecutive threads
//                     -> consecutive addresses).
//
//   stride = FFT_N  : workgroup `wg` operates on column `wg`.
//                     Reads input[wg + i*FFT_N]. Strided global access,
//                     uncoalesced; the L2 picks up the slack on Arc A770.
//
// direction = +1.0f : forward FFT (no normalisation, twiddle sign -2pi)
// direction = -1.0f : inverse FFT (per-axis 1/FFT_N normalisation,
//                                  twiddle sign +2pi)
//
// Algorithm: Cooley-Tukey radix-2 decimation-in-time with bit-reversal at
// the load step. log2(FFT_N) = FFT_LOG2_N stages, each thread handles one
// butterfly per stage. Twiddles via inline native_cos / native_sin -- one
// trig pair per butterfly is cheap (~1024 evals per 1D FFT), and it keeps
// the kernel self-contained with no host-side twiddle table to manage.
// ---------------------------------------------------------------------------
__kernel
__attribute__((reqd_work_group_size(FFT_N / 2, 1, 1)))
void stab_fft_1d(
    const __global float2 *input,
    __global       float2 *output,
    const int   stride,
    const float direction
) {
    __local float2 sdata[FFT_N];

    const int tid  = get_local_id(0);
    const int wg   = get_group_id(0);
    const int base = (stride == 1) ? (wg * FFT_N) : wg;

    // Bit-reverse load. Each thread loads two elements; the load
    // positions are tid and tid + FFT_N/2 in the input, written into
    // sdata at the bit-reversed indices.
    {
        const int i0 = tid;
        const int i1 = tid + FFT_N / 2;
        sdata[stab_bitrev8(i0)] = input[base + i0 * stride];
        sdata[stab_bitrev8(i1)] = input[base + i1 * stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Butterfly stages. At stage s the butterfly group size is m = 2^(s+1)
    // and there are FFT_N/2 butterflies total -- one per thread.
    for (int s = 0; s < FFT_LOG2_N; s++) {
        const int bfly_half = 1 << s;
        const int m         = bfly_half << 1;
        const int k         = tid & (bfly_half - 1);
        const int group     = tid >> s;
        const int idx0      = (group << (s + 1)) + k;
        const int idx1      = idx0 + bfly_half;

        const float2 c0 = sdata[idx0];
        const float2 c1 = sdata[idx1];

        // W_m^k = exp(direction * -2pi*j * k / m)
        const float angle = direction * (-2.0f * M_PI_F) * (float)k * (1.0f / (float)m);
        float wc, ws;
        ws = sincos(angle, &wc);
        const float2 t = (float2)(wc * c1.x - ws * c1.y,
                                  wc * c1.y + ws * c1.x);

        barrier(CLK_LOCAL_MEM_FENCE);
        sdata[idx0] = c0 + t;
        sdata[idx1] = c0 - t;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store, with 1/FFT_N normalisation on the inverse pass. The 2D
    // inverse total of 1/(FFT_N*FFT_N) emerges naturally from doing 1/FFT_N
    // on each of the two passes.
    const float norm = (direction < 0.0f) ? (1.0f / (float)FFT_N) : 1.0f;
    const int o0 = tid;
    const int o1 = tid + FFT_N / 2;
    output[base + o0 * stride] = sdata[o0] * norm;
    output[base + o1 * stride] = sdata[o1] * norm;
}

// ---------------------------------------------------------------------------
// stab_cross_spectrum: pointwise G = F_cur * conj(F_prev), normalised to
// unit magnitude. This is the defining step of phase correlation -- it
// discards spectral magnitude so only the phase difference (which encodes
// translation) survives in the inverse FFT.
//
// (a + bi) * conj(c + di) = (a + bi) * (c - di)
//                         = (a*c + b*d) + (b*c - a*d)*i
// ---------------------------------------------------------------------------
__kernel void stab_cross_spectrum(
    const __global float2 *cur,
    const __global float2 *prev,
    __global       float2 *out,
    const int total_elems
) {
    const int gid = get_global_id(0);
    if (gid >= total_elems) return;

    const float2 a = cur[gid];
    const float2 b = prev[gid];
    const float2 g = (float2)(a.x * b.x + a.y * b.y,
                              a.y * b.x - a.x * b.y);
    const float mag = sqrt(g.x * g.x + g.y * g.y);
    if (mag > 1e-12f) {
        out[gid] = g * (1.0f / mag);
    } else {
        out[gid] = (float2)(0.0f, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// stab_sample: read one source pixel with border-mode handling.
// border_mode: 0 = BLACK (return fill_value on out-of-bounds)
//              1 = CLAMP (extend the edge pixel)
//              2 = MIRROR (reflect across the edge)
//
// One conditional reflection is enough for shifts within ~one frame; the
// final clamp() is belt-and-suspenders for pathological cases.
// ---------------------------------------------------------------------------
inline float stab_sample(
    const __global uchar *pSrc, int srcPitch,
    int x, int y, int w, int h,
    int border_mode, int fill_value
) {
    if (x < 0 || x >= w || y < 0 || y >= h) {
        if (border_mode == 0) {
            return (float)fill_value;
        } else if (border_mode == 1) {
            if (x < 0) x = 0; else if (x >= w) x = w - 1;
            if (y < 0) y = 0; else if (y >= h) y = h - 1;
        } else {
            if (x < 0)  x = -x - 1;
            if (x >= w) x = 2 * w - x - 1;
            if (y < 0)  y = -y - 1;
            if (y >= h) y = 2 * h - y - 1;
            x = clamp(x, 0, w - 1);
            y = clamp(y, 0, h - 1);
        }
    }
    const Type px = *(const __global Type *)(pSrc + y * srcPitch + x * sizeof(Type));
    return (float)px;
}

// ---------------------------------------------------------------------------
// stab_warp: bilinear-sample the input plane at (ox + 0.5 - shiftX,
// oy + 0.5 - shiftY), writing one output pixel per work-item.
//
// shiftX, shiftY are in plane-pixel units. The host divides the luma
// shift by the chroma subsampling factor before launching on chroma
// planes, so this kernel does not need to know which plane it is on.
//
// fill_value: pixel value used for out-of-bounds samples in BLACK mode.
// Luma plane should pass 0; chroma planes should pass the neutral mid-
// grey value (1 << (bit_depth - 1)) so a black border looks actually
// black rather than green-tinted.
// ---------------------------------------------------------------------------
__kernel void stab_warp(
    const __global uchar *pSrc, const int srcPitch,
    __global       uchar *pDst, const int dstPitch,
    const int width, const int height,
    const float shiftX, const float shiftY,
    const int border_mode,
    const int fill_value
) {
    const int ox = get_global_id(0);
    const int oy = get_global_id(1);
    if (ox >= width || oy >= height) return;

    // Pixel-centre coordinate in the source after the inverse-of-camera
    // shift. The 2x2 patch we sample has its top-left corner at
    // (sxc, syc) = (sx_f - 0.5, sy_f - 0.5).
    const float sx_f = (float)ox + 0.5f - shiftX;
    const float sy_f = (float)oy + 0.5f - shiftY;
    const float sxc  = sx_f - 0.5f;
    const float syc  = sy_f - 0.5f;
    const int sx0 = (int)floor(sxc);
    const int sy0 = (int)floor(syc);
    const float fx = sxc - (float)sx0;
    const float fy = syc - (float)sy0;

    const float v00 = stab_sample(pSrc, srcPitch, sx0,     sy0,     width, height, border_mode, fill_value);
    const float v10 = stab_sample(pSrc, srcPitch, sx0 + 1, sy0,     width, height, border_mode, fill_value);
    const float v01 = stab_sample(pSrc, srcPitch, sx0,     sy0 + 1, width, height, border_mode, fill_value);
    const float v11 = stab_sample(pSrc, srcPitch, sx0 + 1, sy0 + 1, width, height, border_mode, fill_value);

    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w10 = fx          * (1.0f - fy);
    const float w01 = (1.0f - fx) * fy;
    const float w11 = fx          * fy;
    const float v = v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11;

    const float vc = (v < 0.0f) ? 0.0f : ((v > (float)max_val) ? (float)max_val : v);
    *(__global Type *)(pDst + oy * dstPitch + ox * sizeof(Type)) = (Type)(vc + 0.5f);
}
