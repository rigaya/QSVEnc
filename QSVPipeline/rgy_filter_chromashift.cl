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
// ChromaShift -- sub-pixel chroma alignment correction.
//
// Reads from a normalised image2d_t source via CLK_FILTER_LINEAR so any
// fractional shift is resolved by hardware bilinear sampling at no extra
// cost compared to nearest-neighbour. Writes back to the chroma-plane
// raw global buffer.
//
// Build-time defines (set via -D from rgy_filter_chromashift.cpp):
//   Type        : uchar (8-bit) or ushort (>8-bit)
//   bit_depth   : source bit depth (8, 10, 12, 14, 16)
//   max_val     : (1 << bit_depth) - 1
//
// Both kernels assume the host has already converted the user-facing
// luma-pixel shift to chroma-plane units (i.e. multiplied by the
// CSP subsampling factor on each axis).

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

// ---------------------------------------------------------------------------
// chromashift_shift: bilinear sub-pixel shift of one chroma plane.
//
// shift_x / shift_y are in chroma-plane pixel units. With LINEAR filtering
// the sample at (x + 0.5 + shift_x, y + 0.5 + shift_y) gives the correct
// pixel-centre interpolation; integer shifts produce exact results, and
// fractional shifts use hardware bilinear with no extra arithmetic here.
// ---------------------------------------------------------------------------
__kernel void chromashift_shift(
    __read_only image2d_t srcImg,
    __global   uchar *pDst, const int dstPitch,
    const int width, const int height,
    const float shift_x_chroma,
    const float shift_y_chroma
) {
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE
                            | CLK_ADDRESS_CLAMP_TO_EDGE
                            | CLK_FILTER_LINEAR;

    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const float2 src_pos = (float2)(ix + 0.5f + shift_x_chroma,
                                    iy + 0.5f + shift_y_chroma);
    const float v = read_imagef(srcImg, sampler, src_pos).x;

    __global Type *dst = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    dst[0] = (Type)(clamp(v, 0.0f, 1.0f - 1e-6f) * (float)max_val);
}

// ---------------------------------------------------------------------------
// chromashift_laplacian: |Laplacian(U)| -> Y plane, diagnostic mode.
//
// Computes the 4-connected discrete Laplacian on the (already-shifted)
// chroma plane and writes the absolute response to the luma destination
// at full luma resolution. Result is upsampled by simple repeat (nearest
// neighbour) so the diagnostic does not introduce its own sub-pixel error.
//
//   L[x,y] = 4*c - up - down - left - right
//   out    = clamp(|L| * LAP_GAIN, 0, max_val)
//
// The theoretical max |L| is 4*max_val (centre vs. all-zero neighbours,
// or vice versa). In real content though, chroma edges are subtle:
// typical |L| values land in the 5..100 range at 8-bit, so a /4 or even
// /1 normalisation reads as near-black. LAP_GAIN=16 lifts those values
// into the visible band -- strong edges saturate to max_val (white),
// which is exactly what makes misalignment easy to spot at a glance.
//
// Chroma planes are filled with the neutral mid-grey value (1<<(bit_depth-1))
// so the diagnostic appears grayscale, not tinted.
// ---------------------------------------------------------------------------
#define CHROMASHIFT_LAP_GAIN 16
__kernel void chromashift_laplacian(
    const __global uchar *pSrcC, const int srcCPitch,
    const int chromaW, const int chromaH,
    const int subX, const int subY,
    __global uchar *pDstY, const int dstYPitch,
    const int lumaW, const int lumaH
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= lumaW || iy >= lumaH) return;

    // Project the luma coordinate into the chroma grid using the source
    // CSP's subsampling factors. 4:2:0: cx = ix/2, cy = iy/2. 4:2:2: cx
    // = ix/2, cy = iy. 4:4:4: cx = ix, cy = iy.
    const int cx = (subX > 0) ? (ix / subX) : ix;
    const int cy = (subY > 0) ? (iy / subY) : iy;

    const int xm = max(cx - 1, 0);
    const int xp = min(cx + 1, chromaW - 1);
    const int ym = max(cy - 1, 0);
    const int yp = min(cy + 1, chromaH - 1);

    #define SRC(x, y) (int)(*(const __global Type *)(pSrcC + (y) * srcCPitch + (x) * sizeof(Type)))
    const int c  = SRC(cx, cy);
    const int up = SRC(cx, ym);
    const int dn = SRC(cx, yp);
    const int le = SRC(xm, cy);
    const int ri = SRC(xp, cy);
    #undef SRC

    const int lap = (4 * c - up - dn - le - ri);
    const int absLap = (lap < 0 ? -lap : lap);
    const int scaled = absLap * CHROMASHIFT_LAP_GAIN;
    const int out = (scaled > max_val) ? max_val : scaled;

    __global Type *dst = (__global Type *)(pDstY + iy * dstYPitch + ix * sizeof(Type));
    dst[0] = (Type)out;
}

// ---------------------------------------------------------------------------
// chromashift_fill_neutral: write the neutral mid-grey value across one
// chroma plane. Used in show=laplacian mode so the diagnostic image is
// grayscale (chroma planes are zeroed/centred) instead of tinted.
// ---------------------------------------------------------------------------
__kernel void chromashift_fill_neutral(
    __global uchar *pDst, const int dstPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    // Neutral chroma = 1 << (bit_depth - 1). For 8-bit that's 128.
    const Type neutral = (Type)(1 << (bit_depth - 1));
    __global Type *dst = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    dst[0] = neutral;
}

// ---------------------------------------------------------------------------
// Auto-detection kernels.
//
// The auto path measures chroma/luma misalignment by comparing the zero
// crossings of the Y plane's Laplacian with the zero crossings of the
// upsampled-and-averaged U+V plane's Laplacian. If Y and UV are aligned
// the zero crossings co-locate; if they are offset, the displacement
// between matched zero crossings is the shift to correct.
//
// Three passes per analysis frame:
//   1. chromashift_lapsign_y       Y -> int8 Laplacian sign map.
//   2. chromashift_lapsign_uv      U+V -> int8 Laplacian sign map at
//                                  luma resolution (bilinear upsampling
//                                  of U and V to luma grid, average,
//                                  then 4-conn Laplacian).
//   3. chromashift_correlate       For each Y zero crossing pixel,
//                                  scan +/- R of the UV sign map.
//                                  Atomic-add the nearest matched
//                                  displacement (dx, dy) to global
//                                  counters [sum_dx, sum_dy, count].
//
// Sign maps are stored as signed bytes (char): -1 for negative, 0 for
// strict zero, +1 for positive. A pixel is a "zero crossing" when its
// sign differs from the right OR bottom neighbour (one direction is
// enough since the offset we are measuring is small).
// ---------------------------------------------------------------------------

#define CHROMASHIFT_AUTO_SEARCH_R 4

inline int chromashift_lap_clampX(int x, int w) { return (x < 0) ? 0 : (x >= w) ? (w - 1) : x; }
inline int chromashift_lap_clampY(int y, int h) { return (y < 0) ? 0 : (y >= h) ? (h - 1) : y; }

inline int chromashift_sign(int v) { return (v > 0) ? 1 : (v < 0) ? -1 : 0; }

// Bilinear sample of one chroma plane at fractional chroma-plane
// coordinates (fx, fy). Returns the interpolated value as a float in
// the original pixel domain (no normalisation).
inline float chromashift_sample_chroma_bilinear(
    const __global uchar *pSrc, const int srcPitch,
    const int chromaW, const int chromaH,
    float fx, float fy
) {
    if (fx < 0.0f)               fx = 0.0f;
    if (fx > (float)(chromaW-1)) fx = (float)(chromaW-1);
    if (fy < 0.0f)               fy = 0.0f;
    if (fy > (float)(chromaH-1)) fy = (float)(chromaH-1);
    const int x0 = (int)fx;
    const int y0 = (int)fy;
    const int x1 = (x0 + 1 < chromaW) ? (x0 + 1) : (chromaW - 1);
    const int y1 = (y0 + 1 < chromaH) ? (y0 + 1) : (chromaH - 1);
    const float fxF = fx - (float)x0;
    const float fyF = fy - (float)y0;
    const float w00 = (1.0f - fxF) * (1.0f - fyF);
    const float w10 =         fxF  * (1.0f - fyF);
    const float w01 = (1.0f - fxF) *         fyF;
    const float w11 =         fxF  *         fyF;
    return (float)(*(const __global Type *)(pSrc + y0 * srcPitch + x0 * sizeof(Type))) * w00
         + (float)(*(const __global Type *)(pSrc + y0 * srcPitch + x1 * sizeof(Type))) * w10
         + (float)(*(const __global Type *)(pSrc + y1 * srcPitch + x0 * sizeof(Type))) * w01
         + (float)(*(const __global Type *)(pSrc + y1 * srcPitch + x1 * sizeof(Type))) * w11;
}

// Y Laplacian sign map (luma resolution).
__kernel void chromashift_lapsign_y(
    const __global uchar *pSrcY, const int srcYPitch,
    __global       char  *pSign, const int signPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const int xm = chromashift_lap_clampX(ix - 1, width);
    const int xp = chromashift_lap_clampX(ix + 1, width);
    const int ym = chromashift_lap_clampY(iy - 1, height);
    const int yp = chromashift_lap_clampY(iy + 1, height);

    #define SRCY(x, y) (int)(*(const __global Type *)(pSrcY + (y) * srcYPitch + (x) * sizeof(Type)))
    const int c  = SRCY(ix, iy);
    const int up = SRCY(ix, ym);
    const int dn = SRCY(ix, yp);
    const int le = SRCY(xm, iy);
    const int ri = SRCY(xp, iy);
    #undef SRCY

    const int lap = 4 * c - up - dn - le - ri;
    pSign[iy * signPitch + ix] = (char)chromashift_sign(lap);
}

// UV-combined Laplacian sign map at luma resolution. The U and V planes
// are bilinearly upsampled to the luma grid, averaged, and the 4-conn
// Laplacian is taken at the luma resolution. Sign is written to pSign.
//
// subX/subY are the source CSP's chroma subsampling factors (e.g. 2/2
// for 4:2:0, 2/1 for 4:2:2, 1/1 for 4:4:4). chromaW/chromaH are the
// chroma plane dimensions.
__kernel void chromashift_lapsign_uv(
    const __global uchar *pSrcU, const int srcUPitch,
    const __global uchar *pSrcV, const int srcVPitch,
    const int chromaW, const int chromaH,
    const int subX, const int subY,
    __global       char  *pSign, const int signPitch,
    const int lumaW, const int lumaH
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= lumaW || iy >= lumaH) return;

    // Bilinear upsampler in chroma-plane coordinates. (cx_f, cy_f) is
    // the float position inside the chroma plane corresponding to luma
    // pixel (ix, iy). Neighbour luma offsets map to (1/subX, 1/subY)
    // step in chroma units.
    const float cx_f = ((float)ix + 0.5f) / (float)subX - 0.5f;
    const float cy_f = ((float)iy + 0.5f) / (float)subY - 0.5f;
    const float dxc  = 1.0f / (float)subX;
    const float dyc  = 1.0f / (float)subY;

    const float uC  = chromashift_sample_chroma_bilinear(pSrcU, srcUPitch, chromaW, chromaH, cx_f,        cy_f       );
    const float vC  = chromashift_sample_chroma_bilinear(pSrcV, srcVPitch, chromaW, chromaH, cx_f,        cy_f       );
    const float uU  = chromashift_sample_chroma_bilinear(pSrcU, srcUPitch, chromaW, chromaH, cx_f,        cy_f - dyc );
    const float vU  = chromashift_sample_chroma_bilinear(pSrcV, srcVPitch, chromaW, chromaH, cx_f,        cy_f - dyc );
    const float uD  = chromashift_sample_chroma_bilinear(pSrcU, srcUPitch, chromaW, chromaH, cx_f,        cy_f + dyc );
    const float vD  = chromashift_sample_chroma_bilinear(pSrcV, srcVPitch, chromaW, chromaH, cx_f,        cy_f + dyc );
    const float uL  = chromashift_sample_chroma_bilinear(pSrcU, srcUPitch, chromaW, chromaH, cx_f - dxc,  cy_f       );
    const float vL  = chromashift_sample_chroma_bilinear(pSrcV, srcVPitch, chromaW, chromaH, cx_f - dxc,  cy_f       );
    const float uR  = chromashift_sample_chroma_bilinear(pSrcU, srcUPitch, chromaW, chromaH, cx_f + dxc,  cy_f       );
    const float vR  = chromashift_sample_chroma_bilinear(pSrcV, srcVPitch, chromaW, chromaH, cx_f + dxc,  cy_f       );

    // Combined U+V signal at luma resolution.
    const int c  = (int)(uC + vC);
    const int up = (int)(uU + vU);
    const int dn = (int)(uD + vD);
    const int le = (int)(uL + vL);
    const int ri = (int)(uR + vR);

    const int lap = 4 * c - up - dn - le - ri;
    pSign[iy * signPitch + ix] = (char)chromashift_sign(lap);
}

// Cross-correlation pass: for each pixel that is a zero crossing of the
// Y Laplacian (sign differs from right OR down neighbour), scan a +/- R
// window of the UV sign map looking for any UV zero crossing. The
// nearest matched displacement (dx, dy) is added to three global counters
// [sum_dx, sum_dy, count] via atomic ops.
//
// pStats[0] = sum_dx, pStats[1] = sum_dy, pStats[2] = count.
__kernel void chromashift_correlate(
    const __global char *pSignY,  const int signYPitch,
    const __global char *pSignUV, const int signUVPitch,
    const int width, const int height,
    __global int *pStats
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width - 1 || iy >= height - 1) return;

    const int s_here  = (int)pSignY[iy * signYPitch + ix];
    const int s_right = (int)pSignY[iy * signYPitch + (ix + 1)];
    const int s_down  = (int)pSignY[(iy + 1) * signYPitch + ix];
    const bool isYZc  = (s_here != s_right) || (s_here != s_down);
    if (!isYZc) return;

    // Scan the UV sign map for any zero crossing within the search window.
    // Track the nearest by squared-distance and record its (dx, dy).
    int best_dx = 0;
    int best_dy = 0;
    int best_dist_sq = (CHROMASHIFT_AUTO_SEARCH_R + 1) * (CHROMASHIFT_AUTO_SEARCH_R + 1) * 2;
    bool found = false;

    for (int dy = -CHROMASHIFT_AUTO_SEARCH_R; dy <= CHROMASHIFT_AUTO_SEARCH_R; dy++) {
        const int ny = iy + dy;
        if (ny < 0 || ny >= height - 1) continue;
        for (int dx = -CHROMASHIFT_AUTO_SEARCH_R; dx <= CHROMASHIFT_AUTO_SEARCH_R; dx++) {
            const int nx = ix + dx;
            if (nx < 0 || nx >= width - 1) continue;
            const int u_here  = (int)pSignUV[ny * signUVPitch + nx];
            const int u_right = (int)pSignUV[ny * signUVPitch + (nx + 1)];
            const int u_down  = (int)pSignUV[(ny + 1) * signUVPitch + nx];
            const bool isUZc  = (u_here != u_right) || (u_here != u_down);
            if (!isUZc) continue;
            const int dist_sq = dx * dx + dy * dy;
            if (dist_sq < best_dist_sq) {
                best_dist_sq = dist_sq;
                best_dx = dx;
                best_dy = dy;
                found = true;
            }
        }
    }

    if (found) {
        atomic_add(&pStats[0], best_dx);
        atomic_add(&pStats[1], best_dy);
        atomic_inc(&pStats[2]);
    }
}
