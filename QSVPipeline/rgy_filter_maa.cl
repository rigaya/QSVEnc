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
// MAA (Masked Anti-Aliasing) OpenCL kernels
// From-scratch implementation for QSVEncC
// Algorithm: 9-direction SangNom2-style directional AA + FTurn rotation
// Reference projects: SangNom2, FTurn, MaskTools2
//
// Build-time defines (set via -D from rgy_filter_maa.cpp):
//   Type         : uchar (8-bit) or ushort (>8-bit)
//   bit_depth    : source bit depth
//   max_val      : (1 << bit_depth) - 1
//   maa_block_x  : work-group X dimension
//   maa_block_y  : work-group Y dimension
//
// HBD (high bit depth) safety:
//   * Pixel storage uses Type (uchar / ushort).
//   * Intermediate arithmetic uses int (32-bit). Worst-case:
//       - SangNom 3-tap (4*p1 + 5*p2 - p3) for 16-bit max_val=65535:
//         |4*65535 + 5*65535 - 0| = 589 815 — fits int32 (max ~2.1e9). ✓
//       - 3×7 smoothing 21-tap sum: 21 * 65535 = 1.4 M — fits int32. ✓
//       - Sobel sum-of-4 abs: 2 * (2 * 65535) = 262 140 — fits int32. ✓
//       - Inflate 8-tap mean: 8 * 65535 = 524 280 — fits int32. ✓
//   * Merge requires 64-bit arithmetic for 16-bit safety:
//         (max_val + 1) * max_val for 16-bit ≈ 4.29e9 — exceeds int32.
//         Mitigated by `long` casts in maa_merge below.
//   * All pitches are in BYTES (matches QSVEnc convention). Pixel
//     access is `(__global Type *)(plane + y * pitch + x * sizeof(Type))`.
//   * All clamps use `max_val` (no hardcoded 255).
//   * All `>>` divisors are either algorithmic constants (>>3 = /8 for
//     8-neighbor mean; >>4 = /16 for 21-tap normalization; >>1 = /2 for
//     sobel halving) or `>> bit_depth` for the merge alpha-blend.
//   * Build options at host: bit_depth=8/10/12/14/16, max_val derived,
//     Type=uchar for 8-bit and ushort for 9..16-bit (single-cl_short load).


// Edge-clamped 1-D pixel access. Used by kernels that walk a pixel row.
inline int loadPixClamp(const __global Type *row, int x, int width) {
    if (x < 0) x = 0;
    if (x >= width) x = width - 1;
    return (int)row[x];
}

// Edge-clamped read from a 2-D plane stored at `pitch` bytes per row.
// `plane` is a byte pointer; we re-interpret as Type using the offset.
inline int readPixClamp(const __global uchar *plane,
                        int x, int y,
                        int pitch, int width, int height) {
    if (x < 0) x = 0;
    if (x >= width)  x = width  - 1;
    if (y < 0) y = 0;
    if (y >= height) y = height - 1;
    return (int)(*(const __global Type *)(plane + y * pitch + x * sizeof(Type)));
}

// =============================================================================
// FTurn rotation kernels
// =============================================================================
//
// Pure transpose with an axis flip — zero arithmetic on pixel values.
//
// Input dimensions:  (srcW × srcH)
// Output dimensions: (srcH × srcW)   — rotation swaps the axes.
//
// FTurnLeft (counter-clockwise 90°):
//     dst[y_new, x_new] = src[x_new, srcW - 1 - y_new]
// FTurnRight (clockwise 90°):
//     dst[y_new, x_new] = src[srcH - 1 - x_new, y_new]
//
// Work-item layout: get_global_id(0) walks the destination column index
// (range 0..srcH-1 = dstW-1); get_global_id(1) walks the destination row
// (range 0..srcW-1 = dstH-1).

__kernel void maa_fturn_left(
    const __global uchar *pSrc,  int srcPitch,
    int srcW, int srcH,
    __global uchar       *pDst,  int dstPitch
) {
    const int x_new = get_global_id(0);   // dstW = srcH
    const int y_new = get_global_id(1);   // dstH = srcW
    if (x_new >= srcH || y_new >= srcW) return;

    const int x_old = srcW - 1 - y_new;
    const int y_old = x_new;

    const __global Type *srcPix = (const __global Type *)(pSrc + y_old * srcPitch + x_old * sizeof(Type));
    __global       Type *dstPix = (__global       Type *)(pDst + y_new * dstPitch + x_new * sizeof(Type));
    dstPix[0] = srcPix[0];
}

__kernel void maa_fturn_right(
    const __global uchar *pSrc,  int srcPitch,
    int srcW, int srcH,
    __global uchar       *pDst,  int dstPitch
) {
    const int x_new = get_global_id(0);   // dstW = srcH
    const int y_new = get_global_id(1);   // dstH = srcW
    if (x_new >= srcH || y_new >= srcW) return;

    const int x_old = y_new;
    const int y_old = srcH - 1 - x_new;

    const __global Type *srcPix = (const __global Type *)(pSrc + y_old * srcPitch + x_old * sizeof(Type));
    __global       Type *dstPix = (__global       Type *)(pDst + y_new * dstPitch + x_new * sizeof(Type));
    dstPix[0] = srcPix[0];
}

// =============================================================================
// SangNom2 — 9-direction edge-directed interpolation
// =============================================================================
//
// Input convention: keep-top mode. Even rows in the source plane are
// preserved, odd rows are interpolated. The cost buffers are stored
// compactly at HALF height (one row per missing row); ybuf in cost-buffer
// coordinates corresponds to source row 2*ybuf + 1.
//
// Cost-buffer dimensions:  bufW = src.width, bufH = src.height / 2.
// Cost-buffer pitch is independent of bufW (allocated once at host init
// and reused across the two SangNom passes; each pass passes its own
// width/height/pitch as parameters).

// 3-tap SangNom kernel: sn(p1, p2, p3) = (4*p1 + 5*p2 - p3) / 8
// Returns the result clamped to the valid pixel range so the abs-diff
// downstream stays in [0, max_val] and fits the cost-buffer's Type.
inline int sn3(int p1, int p2, int p3) {
    int v = (4 * p1 + 5 * p2 - p3) >> 3;
    if (v < 0) v = 0;
    if (v > max_val) v = max_val;
    return v;
}

// Stage 1 — prepare: compute the 9 raw cost values for each missing row.
//
// Work-item layout: (x, ybuf) over (bufW, bufH). Each work-item reads from
// the two preserved rows that bracket missing source row 2*ybuf+1, computes
// 7 direct |cur[x±k] - next[x∓k]| differences and 2 SangNom-kernel-based
// differences, and stores the 9 results at position (x, ybuf) inside the
// PACKED cost buffer at sub-buffer offsets 0..8 × bufSliceBytes.
__kernel void maa_sangnom_prepare(
    const __global uchar *pSrc, int srcPitch,
    int srcW, int srcH,
    __global uchar *pCostPacked,
    int bufPitch, int bufSliceBytes,
    int bufW, int bufH
) {
    const int x    = get_global_id(0);
    const int ybuf = get_global_id(1);
    if (x >= bufW || ybuf >= bufH) return;

    const int yCur  = 2 * ybuf;         // preserved row above (source coords)
    const int yNext = 2 * ybuf + 2;     // preserved row below (source coords; clamp at last buf row)

    const __global Type *cur  = (const __global Type *)(pSrc + (yCur  < srcH ? yCur  : srcH - 1) * srcPitch);
    const __global Type *next = (const __global Type *)(pSrc + (yNext < srcH ? yNext : srcH - 1) * srcPitch);

    // Direct differences along 7 of the 9 directions. ADIFF_M3_P3 .. ADIFF_P3_M3
    // and ADIFF_P0_M0 (vertical).
    const int cM3 = loadPixClamp(cur,  x - 3, srcW);
    const int cM2 = loadPixClamp(cur,  x - 2, srcW);
    const int cM1 = loadPixClamp(cur,  x - 1, srcW);
    const int cP0 = loadPixClamp(cur,  x    , srcW);
    const int cP1 = loadPixClamp(cur,  x + 1, srcW);
    const int cP2 = loadPixClamp(cur,  x + 2, srcW);
    const int cP3 = loadPixClamp(cur,  x + 3, srcW);

    const int nM3 = loadPixClamp(next, x - 3, srcW);
    const int nM2 = loadPixClamp(next, x - 2, srcW);
    const int nM1 = loadPixClamp(next, x - 1, srcW);
    const int nP0 = loadPixClamp(next, x    , srcW);
    const int nP1 = loadPixClamp(next, x + 1, srcW);
    const int nP2 = loadPixClamp(next, x + 2, srcW);
    const int nP3 = loadPixClamp(next, x + 3, srcW);

    const int costM3P3 = abs(cM3 - nP3);   // buf0: steep /
    const int costM2P2 = abs(cM2 - nP2);   // buf1: mid /
    const int costM1P1 = abs(cM1 - nP1);   // buf2: shallow /
    const int costP0M0 = abs(cP0 - nP0);   // buf4: vertical
    const int costP1M1 = abs(cP1 - nM1);   // buf6: shallow \ (note: stored as "P1_M1", direction is x+1 → x-1)
    const int costP2M2 = abs(cP2 - nM2);   // buf7: mid (\)
    const int costP3M3 = abs(cP3 - nM3);   // buf8: steep (\)

    // SangNom 3-tap kernel difference for the / and \ directions. The kernel
    // arguments use cur[x-1]/cur[x]/cur[x+1] for the "forward" direction and
    // their mirrored forms for the "reverse" direction, paired against the
    // mirrored next-row neighborhood
    const int fwdCur  = sn3(cM1, cP0, cP1);    // forward / kernel at cur (x-1, x, x+1)
    const int fwdNext = sn3(nP1, nP0, nM1);    // forward / kernel at next mirrored (x+1, x, x-1)
    const int costSgFwd = abs(fwdCur - fwdNext);  // buf3: SG_FORWARD

    const int bwdCur  = sn3(cP1, cP0, cM1);    // reverse \ kernel at cur (x+1, x, x-1)
    const int bwdNext = sn3(nM1, nP0, nP1);    // reverse \ kernel at next mirrored (x-1, x, x+1)
    const int costSgRev = abs(bwdCur - bwdNext);  // buf5: SG_REVERSE

    // Write all 9 cost values. Each cost value fits in `Type` because abs-diff
    // of two clamped pixel values is ≤ max_val. Slice `i` lives at byte
    // offset `i * bufSliceBytes` within the packed buffer.
    const int rowOffset = ybuf * bufPitch;
    __global Type *outRow0 = (__global Type *)(pCostPacked + 0 * bufSliceBytes + rowOffset);
    __global Type *outRow1 = (__global Type *)(pCostPacked + 1 * bufSliceBytes + rowOffset);
    __global Type *outRow2 = (__global Type *)(pCostPacked + 2 * bufSliceBytes + rowOffset);
    __global Type *outRow3 = (__global Type *)(pCostPacked + 3 * bufSliceBytes + rowOffset);
    __global Type *outRow4 = (__global Type *)(pCostPacked + 4 * bufSliceBytes + rowOffset);
    __global Type *outRow5 = (__global Type *)(pCostPacked + 5 * bufSliceBytes + rowOffset);
    __global Type *outRow6 = (__global Type *)(pCostPacked + 6 * bufSliceBytes + rowOffset);
    __global Type *outRow7 = (__global Type *)(pCostPacked + 7 * bufSliceBytes + rowOffset);
    __global Type *outRow8 = (__global Type *)(pCostPacked + 8 * bufSliceBytes + rowOffset);

    outRow0[x] = (Type)costM3P3;
    outRow1[x] = (Type)costM2P2;
    outRow2[x] = (Type)costM1P1;
    outRow3[x] = (Type)costSgFwd;
    outRow4[x] = (Type)costP0M0;
    outRow5[x] = (Type)costSgRev;
    outRow6[x] = (Type)costP1M1;
    outRow7[x] = (Type)costP2M2;
    outRow8[x] = (Type)costP3M3;
}

// Stage 2 — smooth: 3×7 spatial smoothing of one cost buffer with /16 divisor.
//
// Vertical: sum of 3 consecutive cost-buffer rows (ybuf-1, ybuf, ybuf+1),
// edge-clamped at the top/bottom of the cost buffer.
// Horizontal: 7-tap sum across (x-3..x+3), edge-clamped at the left/right.
// Final divisor is 16 (NOT 21)
//
// Sums fit in int32 comfortably: 21 × max_val ≤ 21 × 65535 ≈ 1.4 M.
// The output is clamped to [0, max_val] so it fits back into Type.
//
// Caller dispatches this kernel 9 times — once per cost slice — with the
// matching `bufIndex` (0..8). Both input and output use the SAME packed
// layout, so a single `pitch` and `sliceBytes` describe both.
__kernel void maa_sangnom_smooth(
    const __global uchar *pCostPacked,
    __global       uchar *pSmoothPacked,
    int bufPitch, int bufSliceBytes,
    int bufIndex,
    int bufW, int bufH
) {
    const int x    = get_global_id(0);
    const int ybuf = get_global_id(1);
    if (x >= bufW || ybuf >= bufH) return;

    const __global uchar *pBufIn  = pCostPacked   + bufIndex * bufSliceBytes;
    __global       uchar *pBufOut = pSmoothPacked + bufIndex * bufSliceBytes;

    int hsum = 0;
    for (int dx = -3; dx <= 3; dx++) {
        int xc = x + dx;
        if (xc < 0)     xc = 0;
        if (xc >= bufW) xc = bufW - 1;
        int vsum = 0;
        for (int dy = -1; dy <= 1; dy++) {
            int yc = ybuf + dy;
            if (yc < 0)     yc = 0;
            if (yc >= bufH) yc = bufH - 1;
            vsum += (int)(*(const __global Type *)(pBufIn + yc * bufPitch + xc * sizeof(Type)));
        }
        hsum += vsum;
    }

    int out = hsum >> 4;          // /16 (asymmetric normalization)
    if (out < 0)        out = 0;
    if (out > max_val)  out = max_val;

    __global Type *outRow = (__global Type *)(pBufOut + ybuf * bufPitch);
    outRow[x] = (Type)out;
}

// [MAA-3D-SMOOTH] 3-D dispatch variant: same math as maa_sangnom_smooth,
// but the bufIndex parameter is replaced by `get_global_id(2)`. The host
// dispatches a single (bufW, bufH, 9) grid instead of 9 separate
// (bufW, bufH) grids, dropping 8 of the 9 per-pass enqueue calls.
//
// The old maa_sangnom_smooth kernel is kept for fallback / reference
// (build option, debug toggle, future work) but is not currently called.
__kernel void maa_sangnom_smooth_3d(
    const __global uchar *pCostPacked,
    __global       uchar *pSmoothPacked,
    int bufPitch, int bufSliceBytes,
    int bufW, int bufH
) {
    const int x        = get_global_id(0);
    const int ybuf     = get_global_id(1);
    const int bufIndex = get_global_id(2);
    if (x >= bufW || ybuf >= bufH || bufIndex >= 9) return;

    const __global uchar *pBufIn  = pCostPacked   + bufIndex * bufSliceBytes;
    __global       uchar *pBufOut = pSmoothPacked + bufIndex * bufSliceBytes;

    int hsum = 0;
    for (int dx = -3; dx <= 3; dx++) {
        int xc = x + dx;
        if (xc < 0)     xc = 0;
        if (xc >= bufW) xc = bufW - 1;
        int vsum = 0;
        for (int dy = -1; dy <= 1; dy++) {
            int yc = ybuf + dy;
            if (yc < 0)     yc = 0;
            if (yc >= bufH) yc = bufH - 1;
            vsum += (int)(*(const __global Type *)(pBufIn + yc * bufPitch + xc * sizeof(Type)));
        }
        hsum += vsum;
    }

    int out = hsum >> 4;          // /16 (asymmetric normalization)
    if (out < 0)        out = 0;
    if (out > max_val)  out = max_val;

    __global Type *outRow = (__global Type *)(pBufOut + ybuf * bufPitch);
    outRow[x] = (Type)out;
}

// [MAA-LOCAL-SMOOTH] Prototype: smooth kernel with __local-memory tile.
//
// Status: NOT WIRED. Built into the program but not dispatched. Kept as a
// candidate optimisation behind profiler data.
//
// Design rationale: the existing maa_sangnom_smooth_3d issues 21 global-
// memory loads per work-item (3-row × 7-col stencil) for the read side.
// Within a workgroup, neighbouring work-items overlap by 6 columns and 2
// rows, so the unique pixel footprint of a (BX × BY) workgroup is only
// (BX + 6) × (BY + 2). For (32, 8) that is 38 × 10 = 380 unique reads
// vs 256 × 21 = 5376 redundant reads — about 14× more global traffic
// than necessary. Loading once into __local then computing from there
// should significantly reduce DRAM bandwidth.
//
// Local-memory cost per workgroup (one slice at a time, since z is
// fixed per workgroup in the 3-D dispatch):
//   8-bit:  380 bytes  (well under 64 KiB on Arc / Iris)
//   16-bit: 760 bytes
//
// This prototype uses fixed compile-time tile dimensions for clarity:
// MAA_LOCAL_BX = 32, MAA_LOCAL_BY = 8 (matching the default workgroup
// size). If the workgroup-size experiment (RGY_MAA_WG_X/Y) settles on
// a different size, the tile sizes here must be updated to match before
// wiring. A more flexible version would use dynamic local memory, but
// the static form is easier to reason about for the prototype.
#define MAA_LOCAL_BX 32
#define MAA_LOCAL_BY  8
#define MAA_LOCAL_TILE_W (MAA_LOCAL_BX + 6)   // 32 + 2*3 (horizontal halo)
#define MAA_LOCAL_TILE_H (MAA_LOCAL_BY + 2)   // 8  + 2*1 (vertical halo)

__kernel
__attribute__((reqd_work_group_size(MAA_LOCAL_BX, MAA_LOCAL_BY, 1)))
void maa_sangnom_smooth_local(
    const __global uchar *pCostPacked,
    __global       uchar *pSmoothPacked,
    int bufPitch, int bufSliceBytes,
    int bufW, int bufH
) {
    const int gx       = get_global_id(0);
    const int gy       = get_global_id(1);
    const int bufIndex = get_global_id(2);
    const int lx       = get_local_id(0);
    const int ly       = get_local_id(1);
    const int wgBaseX  = gx - lx;
    const int wgBaseY  = gy - ly;

    if (bufIndex >= 9) return;
    const __global uchar *pBufIn  = pCostPacked   + bufIndex * bufSliceBytes;
    __global       uchar *pBufOut = pSmoothPacked + bufIndex * bufSliceBytes;

    __local Type tile[MAA_LOCAL_TILE_H][MAA_LOCAL_TILE_W];

    // Cooperative tile load: each work-item loads one or more tile slots.
    // The tile is (BX+6) × (BY+2) entries — slightly larger than the WG.
    // We use a 2-D loop where each work-item strides by (BX, BY) until
    // the whole tile is covered. For BX=32, BY=8: TILE_W=38, TILE_H=10
    // → 380/(32*8)=380/256≈1.48 loads/work-item on average; some
    // work-items do 2 loads, most do 1.
    for (int dy = ly; dy < MAA_LOCAL_TILE_H; dy += MAA_LOCAL_BY) {
        const int srcY = wgBaseY + dy - 1;          // -1 for top halo
        const int yc   = (srcY < 0)        ? 0
                       : (srcY >= bufH)    ? bufH - 1
                       :                     srcY;
        const __global Type *row = (const __global Type *)(pBufIn + yc * bufPitch);
        for (int dx = lx; dx < MAA_LOCAL_TILE_W; dx += MAA_LOCAL_BX) {
            const int srcX = wgBaseX + dx - 3;      // -3 for left halo
            const int xc   = (srcX < 0)        ? 0
                           : (srcX >= bufW)    ? bufW - 1
                           :                     srcX;
            tile[dy][dx] = row[xc];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Out-of-bounds work-items still helped fill the tile above; only
    // skip the write here. (This guard is after the tile load so other
    // work-items that needed our halo cooperation have completed.)
    if (gx >= bufW || gy >= bufH) return;

    // 3×7 sum using __local reads. Tile coordinates: my pixel sits at
    // (lx + 3, ly + 1) — the +3/+1 accounts for the halo offset.
    const int tx = lx + 3;
    const int ty = ly + 1;
    int hsum = 0;
    for (int dx = -3; dx <= 3; dx++) {
        int vsum = 0;
        for (int dy = -1; dy <= 1; dy++) {
            vsum += (int)tile[ty + dy][tx + dx];
        }
        hsum += vsum;
    }

    int out = hsum >> 4;          // /16 (asymmetric normalization)
    if (out < 0)        out = 0;
    if (out > max_val)  out = max_val;

    __global Type *outRow = (__global Type *)(pBufOut + gy * bufPitch);
    outRow[gx] = (Type)out;
}

// Stage 3 — finalize: pick the min-cost direction (with vertical bail-out
// above aaf) and emit the corresponding average.
//
// Work-item layout: (x, y) over (srcW, srcH). For preserved rows
// (y even) the source pixel is copied verbatim. For missing rows
// (y odd) the cost buffers at (x, ybuf=y/2) are read for all 9 directions,
// the minimum cost is found, and the decision cascade picks the corresponding average.
//
// Decision cascade ORDER MATTERS — the reference comments warn explicitly
// "the order of following code is important, don't change them". Priority:
//   vertical > SG_REVERSE > SG_FORWARD > shallow > mid > steep
//   (with vertical also chosen when minBuf > aaf — the safety bail-out).
__kernel void maa_sangnom_finalize(
    const __global uchar *pSrc, int srcPitch,
    int srcW, int srcH,
    const __global uchar *pSmoothPacked,
    int bufPitch, int bufSliceBytes,
    int bufW, int bufH,
    __global uchar       *pDst, int dstPitch,
    float aaf
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= srcW || y >= srcH) return;

    __global Type *dstPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));

    // Preserved row → copy verbatim from source.
    if ((y & 1) == 0) {
        dstPix[0] = (Type)readPixClamp(pSrc, x, y, srcPitch, srcW, srcH);
        return;
    }

    // Missing row. Map to cost-buffer row.
    const int ybuf = y >> 1;

    // Read the 9 smoothed cost values at (x, ybuf). Slice `i` lives at
    // byte offset `i * bufSliceBytes` within the packed buffer.
    const __global uchar *pixBase = pSmoothPacked + ybuf * bufPitch + x * sizeof(Type);
    const int b0 = (int)(*(const __global Type *)(pixBase + 0 * bufSliceBytes));
    const int b1 = (int)(*(const __global Type *)(pixBase + 1 * bufSliceBytes));
    const int b2 = (int)(*(const __global Type *)(pixBase + 2 * bufSliceBytes));
    const int b3 = (int)(*(const __global Type *)(pixBase + 3 * bufSliceBytes));
    const int b4 = (int)(*(const __global Type *)(pixBase + 4 * bufSliceBytes));
    const int b5 = (int)(*(const __global Type *)(pixBase + 5 * bufSliceBytes));
    const int b6 = (int)(*(const __global Type *)(pixBase + 6 * bufSliceBytes));
    const int b7 = (int)(*(const __global Type *)(pixBase + 7 * bufSliceBytes));
    const int b8 = (int)(*(const __global Type *)(pixBase + 8 * bufSliceBytes));

    int minCost = b0;
    if (b1 < minCost) minCost = b1;
    if (b2 < minCost) minCost = b2;
    if (b3 < minCost) minCost = b3;
    if (b4 < minCost) minCost = b4;
    if (b5 < minCost) minCost = b5;
    if (b6 < minCost) minCost = b6;
    if (b7 < minCost) minCost = b7;
    if (b8 < minCost) minCost = b8;

    // Read the cur and next preserved-row pixels needed by the decision
    // cascade (cur = row y-1, next = row y+1, both clamped).
    const int yCur  = (y - 1 < 0)        ? 0          : y - 1;
    const int yNext = (y + 1 >= srcH)    ? srcH - 1   : y + 1;

    const __global Type *cur  = (const __global Type *)(pSrc + yCur  * srcPitch);
    const __global Type *next = (const __global Type *)(pSrc + yNext * srcPitch);

    const int cM3 = loadPixClamp(cur,  x - 3, srcW);
    const int cM2 = loadPixClamp(cur,  x - 2, srcW);
    const int cM1 = loadPixClamp(cur,  x - 1, srcW);
    const int cP0 = loadPixClamp(cur,  x    , srcW);
    const int cP1 = loadPixClamp(cur,  x + 1, srcW);
    const int cP2 = loadPixClamp(cur,  x + 2, srcW);
    const int cP3 = loadPixClamp(cur,  x + 3, srcW);

    const int nM3 = loadPixClamp(next, x - 3, srcW);
    const int nM2 = loadPixClamp(next, x - 2, srcW);
    const int nM1 = loadPixClamp(next, x - 1, srcW);
    const int nP0 = loadPixClamp(next, x    , srcW);
    const int nP1 = loadPixClamp(next, x + 1, srcW);
    const int nP2 = loadPixClamp(next, x + 2, srcW);
    const int nP3 = loadPixClamp(next, x + 3, srcW);

    int result;
    // Decision cascade — order matches the reference exactly. The first
    // matching test wins.
    if (b4 == minCost || (float)minCost > aaf) {
        // Vertical (or bail-out: no clear directional fit anywhere).
        result = (cP0 + nP0 + 1) >> 1;
    } else if (b5 == minCost) {
        // SG_REVERSE — average of the two reverse-kernel predictions.
        const int bwdCur  = sn3(cP1, cP0, cM1);
        const int bwdNext = sn3(nM1, nP0, nP1);
        result = (bwdCur + bwdNext + 1) >> 1;
    } else if (b3 == minCost) {
        // SG_FORWARD — average of the two forward-kernel predictions.
        const int fwdCur  = sn3(cM1, cP0, cP1);
        const int fwdNext = sn3(nP1, nP0, nM1);
        result = (fwdCur + fwdNext + 1) >> 1;
    } else if (b6 == minCost) {
        result = (cP1 + nM1 + 1) >> 1;          // shallow (\)
    } else if (b2 == minCost) {
        result = (cM1 + nP1 + 1) >> 1;          // shallow /
    } else if (b7 == minCost) {
        result = (cP2 + nM2 + 1) >> 1;          // mid (\)
    } else if (b1 == minCost) {
        result = (cM2 + nP2 + 1) >> 1;          // mid /
    } else if (b8 == minCost) {
        result = (cP3 + nM3 + 1) >> 1;          // steep (\)
    } else {
        result = (cM3 + nP3 + 1) >> 1;          // steep /  (b0)
    }

    if (result < 0)        result = 0;
    if (result > max_val)  result = max_val;
    dstPix[0] = (Type)result;
}

// =============================================================================
// MaskTools2-style edge-mask path
// =============================================================================
//
//   maa_edge_sobel  — simplified 4-tap orthogonal-difference edge detector,
//                     with a hard threshold (matches mt_edge "sobel" with
//                     low == high == mthresh, which is what MAA2 calls).
//   maa_inflate     — `max(8-neighbor mean, center)` 1-pixel mask dilation.
//   maa_merge       — mask-weighted blend `((maxv+1 - m)*A + m*B + half) >> bd`.
//   maa_mask_subsample — average-downsample helper for chroma=true with
//                        sub-sampled chroma (YV12, YV16). For YV24 the host
//                        skips this kernel and reuses the luma mask buffer.
//
// All kernels operate on a single plane. Border pixels are passed through
// (return 0 for edge / source for inflate / source for merge with mask=0)
// to match MaskTools2 semantics exactly.

// Stage A — simplified Sobel: 4-tap orthogonal-difference + hard threshold.
//
// Per-pixel formula (interior only):
//   edge = abs((right + below) - (left + above)) >> 1
//   out  = (edge >= mthresh) ? max_val : 0
//
// `mthresh` is bit-depth-scaled at the host (peak/255 factor). The hard
// threshold collapses MaskTools2's `threshold(value, low, high)` for the
// special case `low == high == mthresh`, which is the only form MAA2 uses.
__kernel void maa_edge_sobel(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height,
    int mthresh
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    __global Type *dstPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));

    // Border pixels: pass through 0 (no edge) — matches MaskTools2's behavior
    // of skipping the 1-pixel frame around the image.
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) {
        dstPix[0] = (Type)0;
        return;
    }

    const __global Type *rowAbove = (const __global Type *)(pSrc + (y - 1) * srcPitch);
    const __global Type *rowMid   = (const __global Type *)(pSrc +  y      * srcPitch);
    const __global Type *rowBelow = (const __global Type *)(pSrc + (y + 1) * srcPitch);

    const int above = (int)rowAbove[x];      // a21
    const int left  = (int)rowMid  [x - 1];  // a12
    const int right = (int)rowMid  [x + 1];  // a32
    const int below = (int)rowBelow[x];      // a23

    const int edge = abs((right + below) - (left + above)) >> 1;
    dstPix[0] = (Type)((edge >= mthresh) ? max_val : 0);
}

// Stage B — mt_inflate: max of 8-neighbor mean and center.
//
// Per-pixel formula (interior only):
//   mean8 = (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8) >> 3
//   out   = max(mean8, center)
//
// Border pixels: pass-through (preserve the 1-pixel border of the input
// mask, which from maa_edge_sobel is always 0 there).
__kernel void maa_inflate(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    __global Type *dstPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));

    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) {
        dstPix[0] = *(const __global Type *)(pSrc + y * srcPitch + x * sizeof(Type));
        return;
    }

    const __global Type *rowAbove = (const __global Type *)(pSrc + (y - 1) * srcPitch);
    const __global Type *rowMid   = (const __global Type *)(pSrc +  y      * srcPitch);
    const __global Type *rowBelow = (const __global Type *)(pSrc + (y + 1) * srcPitch);

    const int a1 = (int)rowAbove[x - 1];
    const int a2 = (int)rowAbove[x    ];
    const int a3 = (int)rowAbove[x + 1];
    const int a4 = (int)rowMid  [x - 1];
    const int c  = (int)rowMid  [x    ];
    const int a5 = (int)rowMid  [x + 1];
    const int a6 = (int)rowBelow[x - 1];
    const int a7 = (int)rowBelow[x    ];
    const int a8 = (int)rowBelow[x + 1];

    const int mean8 = (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8) >> 3;
    const int out   = (mean8 > c) ? mean8 : c;
    dstPix[0] = (Type)out;
}

// Stage C — mt_merge: mask-weighted alpha blend.
//
// Per-pixel formula:
//   if mask == 0       :  out = A           (untouched source)
//   if mask == max_val :  out = B           (full AA result)
//   else               :  out = (((maxv+1 - m) * A + m * B + (maxv+1)/2) >> bd)
//
// `maxv+1` and `>> bd` give a divisor of exactly `1 << bit_depth`, so the
// shift is exact (no rounding tricks needed; the `+ (maxv+1)/2` provides
// round-to-nearest). Matches MaskTools2's `((256-m)*A + m*B + 128) >> 8`
// at 8-bit and the equivalent generalization at higher bit depths.
//
// Use 64-bit arithmetic for the products to stay safe at 16-bit, where
// (maxv+1) * maxv ≈ 4.3e9 exceeds int32 range. For 8/10/12/14-bit the
// products fit in int32 but using long is harmless.
__kernel void maa_merge(
    const __global uchar *pSrcA, int srcAPitch,    // original (A)
    const __global uchar *pSrcB, int srcBPitch,    // AA result (B)
    const __global uchar *pMask, int maskPitch,    // edge mask
    __global       uchar *pDst,  int dstPitch,
    int width, int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const __global Type *aPix = (const __global Type *)(pSrcA + y * srcAPitch + x * sizeof(Type));
    const __global Type *bPix = (const __global Type *)(pSrcB + y * srcBPitch + x * sizeof(Type));
    const __global Type *mPix = (const __global Type *)(pMask + y * maskPitch + x * sizeof(Type));
    __global       Type *dPix = (__global       Type *)(pDst  + y * dstPitch  + x * sizeof(Type));

    const int a = (int)aPix[0];
    const int b = (int)bPix[0];
    const int m = (int)mPix[0];

    if (m == 0) {
        dPix[0] = (Type)a;
        return;
    }
    if (m == max_val) {
        dPix[0] = (Type)b;
        return;
    }

    const long maxvP1 = (long)max_val + 1L;
    const long blended = ( (maxvP1 - (long)m) * (long)a
                         + (long)m            * (long)b
                         + (maxvP1 >> 1) ) >> bit_depth;
    int out = (int)blended;
    if (out < 0)        out = 0;
    if (out > max_val)  out = max_val;
    dPix[0] = (Type)out;
}

// Stage D — mask sub-sampling helper for chroma=true with 4:2:0 / 4:2:2.
//
// The luma mask has dimensions (lumaW × lumaH). The chroma plane mask must
// match the chroma plane dimensions (chromaW × chromaH). This kernel reads
// a (subSampleX × subSampleY) block of luma mask values and writes the
// integer average to the chroma mask. For 4:4:4 (sub = 1×1) the host
// skips this kernel and reuses the luma mask directly.
__kernel void maa_mask_subsample(
    const __global uchar *pLumaMask, int lumaPitch,
    int lumaW, int lumaH,
    __global       uchar *pChromaMask, int chromaPitch,
    int chromaW, int chromaH,
    int subSampleX, int subSampleY
) {
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);
    if (cx >= chromaW || cy >= chromaH) return;

    const int xBase = cx * subSampleX;
    const int yBase = cy * subSampleY;

    int sum = 0;
    int count = 0;
    for (int dy = 0; dy < subSampleY; dy++) {
        const int yi = yBase + dy;
        if (yi >= lumaH) continue;
        const __global Type *row = (const __global Type *)(pLumaMask + yi * lumaPitch);
        for (int dx = 0; dx < subSampleX; dx++) {
            const int xi = xBase + dx;
            if (xi >= lumaW) continue;
            sum += (int)row[xi];
            count++;
        }
    }
    const int avg = (count > 0) ? (sum / count) : 0;
    __global Type *dst = (__global Type *)(pChromaMask + cy * chromaPitch + cx * sizeof(Type));
    dst[0] = (Type)avg;
}

// =============================================================================
// Show / debug overlay kernels (used when --vpp-maa show=1 or show=2)
// =============================================================================
//
// Reference: AviSynth MAA2 "show" path uses mt_lut("x 2 /") to halve all
// planes, then overlays the inflated edge mask on top. We split that into
// two small kernels:
//   maa_show_overlay — luma plane: dst = halve(src) + halve(mask).
//                      Mask values brighten the dimmed source so the
//                      edge-classified pixels are visible against the
//                      gray underlay.
//   maa_show_darken  — chroma plane (or luma when no mask is wanted):
//                      dst = halve(src). Halving chroma shifts it toward
//                      the lower channel range; visually this is a
//                      colour-tinted dim, intentionally distinct from
//                      a normal output to flag debug mode.
//
// HBD-safe: both kernels use `>> 1` (halving), which preserves precision
// for all bit depths. Output sum (halve_src + halve_mask) ≤ max_val.

// Stage E — show overlay: half-darken `pSrc` luma and overlay the inflated
// mask values (also halved). For show=1 the host calls this with
// pSrc = input luma; for show=2 with pSrc = AA-result luma. In both cases
// pMask is the inflated edge mask. The output luma plane shows the AA
// region as a brighter strip over the dimmed underlay.
__kernel void maa_show_overlay(
    const __global uchar *pSrc,  int srcPitch,
    const __global uchar *pMask, int maskPitch,
    __global       uchar *pDst,  int dstPitch,
    int width, int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const __global Type *sPix = (const __global Type *)(pSrc  + y * srcPitch  + x * sizeof(Type));
    const __global Type *mPix = (const __global Type *)(pMask + y * maskPitch + x * sizeof(Type));
    __global       Type *dPix = (__global       Type *)(pDst  + y * dstPitch  + x * sizeof(Type));

    const int s = (int)sPix[0];
    const int m = (int)mPix[0];

    int result = (s >> 1) + (m >> 1);
    if (result > max_val) result = max_val;
    dPix[0] = (Type)result;
}

// Stage F — show darken: simple `dst = src >> 1`. Used for chroma planes
// in show mode (and any plane where we want a dimmed underlay without a
// mask overlay).
__kernel void maa_show_darken(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const __global Type *sPix = (const __global Type *)(pSrc + y * srcPitch + x * sizeof(Type));
    __global       Type *dPix = (__global       Type *)(pDst + y * dstPitch + x * sizeof(Type));

    dPix[0] = (Type)((int)sPix[0] >> 1);
}
