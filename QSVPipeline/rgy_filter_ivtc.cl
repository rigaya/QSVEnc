// Type
// bit_depth
// ivtc_block_x
// ivtc_block_y

// ----- EXTERNAL-ALGORITHM NOTICE -----
// This file contains independently-written implementations of
// algorithms first described in:
//   - Telecide/Decomb (GPL-2.0) by Donald A. Graft / "tritical"
//   - BBC PH-2071 Weston 3-Field Deinterlacer (published spec)
//   - ffmpeg vf_bwdif (motion-adaptive extension of PH-2071)
// The filter coefficients in kernel_ivtc_bwdif_deint (5077, 981,
// 4309, 213, 5570, 3801, 1016) are the published PH-2071 values
// and are shared across all independent BWDIF implementations.
// See ACKNOWLEDGMENTS.md at the repository root.
// --------------------------------------

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

// SUB-PHASE 1 (2026-04-24): per-WG block "combed?" binary classifier.
// A 16x16 block is considered combed when its per-WG cX pixel sum meets
// or exceeds this threshold. After the c2 removal (rgy_filter_ivtc.cl
// :186), the maximum per-WG cX sum is 128 (8 first-parity rows * 16
// columns). 8 combed pixels = ~6% of the tested sample in the block,
// roughly matching TFM's default sensitivity (~6 pixels in a 24x24
// block). Diagnostic-only in SUB-PHASE 1 — not consumed by any gate.
#ifndef BLOCK_COMB_THRESH
#define BLOCK_COMB_THRESH 8
#endif

// Field overlay for RFF expansion (DGDecode CopyBot/CopyTop analogue).
// tff=1: overlay BOT field (odd rows) from src onto dst — used when the
//        synth frame should inherit cur's TOP and prev's BOT (TFF stream).
// tff=0: overlay TOP field (even rows) from src onto dst — BFF stream.
// Non-overlay rows are left untouched (dst already contains cur's data).
// Zero interpolation; matches DGDecode vfapidec.cpp:1027-1059 stride*2 blit.
__attribute__((reqd_work_group_size(ivtc_block_x, ivtc_block_y, 1)))
__kernel void kernel_ivtc_field_overlay(
    __global uchar *restrict pDst, const int dstPitch,
    const __global uchar *restrict pSrc, const int srcPitch,
    const int width, const int height, const int tff
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int targetParity = tff ? 1 : 0;  // 1 = overwrite odd rows; 0 = overwrite even rows
    if ((iy & 1) == targetParity) {
        __global       Type *dstPix = (__global       Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        const __global Type *srcPix = (const __global Type *)(pSrc + iy * srcPitch + ix * sizeof(Type));
        dstPix[0] = srcPix[0];
    }
}

// Clamp-read a single pixel from a merged-frame plane.
// Shared between all kernels in this file.
static inline int ivtc_readPix(
    const __global uchar *plane, int x, int y,
    const int pitch, const int width, const int height
) {
    x = clamp(x, 0, width  - 1);
    y = clamp(y, 0, height - 1);
    return (int)(*(const __global Type *)(plane + y * pitch + x * sizeof(Type)));
}

// Same-parity temporal read on a merged-frame reference.
// y in the reference frame is opposite-parity (odd for TFF missing rows);
// average y-1 and y+1 which ARE preserved-parity rows. Estimates the value
// the preserved field would take at spatial row y in that reference frame.
static inline int ivtc_readPixSameParity(
    const __global uchar *plane, int x, int y,
    const int pitch, const int width, const int height
) {
    const int up = ivtc_readPix(plane, x, y - 1, pitch, width, height);
    const int dn = ivtc_readPix(plane, x, y + 1, pitch, width, height);
    return (up + dn + 1) >> 1;
}

// Fetch a reconstructed pixel per candidate / row-parity.
// Candidate geometry (TFF, first-field = top = even rows):
//   C: [cur.top, cur.bot]                   — current frame as-is
//   P: [cur.top, prev.bot]                  — match-with-prev (borrow bot from previous)
//   N: [next.top, cur.bot]                  — match-with-next (borrow top from next)
// For BFF (tff=0) the first-field rows are odd-parity instead.
static int pix_match(
    const __global uchar *pPrev, const __global uchar *pCur, const __global uchar *pNext,
    const int pitch, int ix, int iy, const int width, const int height,
    const int tff, const int match) {
    ix = clamp(ix, 0, width - 1);
    iy = clamp(iy, 0, height - 1);
    const int is_first_field_row = (iy & 1) == (tff ? 0 : 1);
    const __global uchar *src;
    if (match == 1) {              // P: first-field from cur, other from prev
        src = is_first_field_row ? pCur : pPrev;
    } else if (match == 2) {       // N: first-field from next, other from cur
        src = is_first_field_row ? pNext : pCur;
    } else {                       // C: cur only
        src = pCur;
    }
    return (int)(*(const __global Type *)(src + iy * pitch + ix * sizeof(Type)));
}

// Score three candidates (C, P, N) on two INDEPENDENT per-pixel metrics:
//
//   match_quality  : interpolation-error metric. For each second-field row,
//                    compute what a linear vertical interpolation from its
//                    same-parity neighbors would predict, and take the L1
//                    error to the actual row value. Summed over both second-
//                    field rows (v1 between v0/v2, v3 between v2/v4).
//                    A well-reconstructed progressive frame has second-field
//                    rows close to the neighbor average; a combed reconstruction
//                    has them far from it. This is the interpolation-error
//                    metric used by yadif / ffmpeg fieldmatch and others.
//                    Lower => better field match.
//
//   combing_count  : count of other-field-row pixels that are local vertical
//                    extrema between their preserved-parity neighbors (v1 between
//                    v0/v2; v3 between v2/v4), where the extremum deviation from
//                    BOTH neighbors exceeds the combing-tolerance T in the SAME
//                    direction. Same-sign detection uses (d_up * d_dn > 0).
//                    Measures residual localized combing; drives post=2 trigger.
//
// Only threads at "first-field-parity" rows with valid ±4 row context contribute
// non-zero values. WG-internal reduction = SUM within WG = per-block block-sum.
// The host-side reduction takes MAX across WGs to produce block-max metrics.
__attribute__((reqd_work_group_size(ivtc_block_x, ivtc_block_y, 1)))
__kernel void kernel_ivtc_score_candidates(
    const __global uchar *restrict pPrev,
    const __global uchar *restrict pCur,
    const __global uchar *restrict pNext,
    const int srcPitch, const int width, const int height,
    const int tff,
    const int nt,       // noise threshold (scaled to bit depth; ~10 on 8-bit)
    const int T,        // combing-tolerance (scaled to bit depth; ~4 on 8-bit)
    const int y0,       // exclusion band: inclusive top row; contributes only iy>=y0
    const int y1,       // exclusion band: inclusive bottom row; contributes only iy<=y1
                        //                 (y0==0 && y1==0 = band disabled, full frame used)
    __global uint *restrict scores        // 9 uints per WG: [mC, mP, mN, cC, cP, cN, bC, bP, bN]
                                          //   mX = match-quality sum (WG-block sum)
                                          //   cX = combed-pixel sum   (WG-block sum)
                                          //   bX = combed-block flag  (1 iff cX >= BLOCK_COMB_THRESH, else 0)
) {
    const int thx = get_local_id(0);
    const int thy = get_local_id(1);
    const int ix  = get_global_id(0);
    const int iy  = get_global_id(1);
    const int tid = thy * ivtc_block_x + thx;
    const int WG_SIZE = ivtc_block_x * ivtc_block_y;

    uint mC = 0u, mP = 0u, mN = 0u;  // match-quality diffs
    uint cC = 0u, cP = 0u, cN = 0u;  // combing counts

    const int first_parity = tff ? 0 : 1;
    // Exclusion band: when y0==y1==0, use the whole frame (standard IVTC convention).
    // Otherwise only contribute rows in [y0, y1]. Applies to BOTH match-quality and combing.
    const int band_on  = (y0 != 0 || y1 != 0);
    const int in_band  = !band_on || (iy >= y0 && iy <= y1);
    if (in_band && ix < width && (iy & 1) == first_parity && iy + 4 < height) {
        // Unroll over candidates to inline pix_match without per-iteration branching penalty.
        for (int m = 0; m < 3; m++) {
            const int v0 = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy    , width, height, tff, m);
            const int v1 = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy + 1, width, height, tff, m);
            const int v2 = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy + 2, width, height, tff, m);
            const int v3 = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy + 3, width, height, tff, m);
            const int v4 = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy + 4, width, height, tff, m);

            // Match-quality: interpolation-error metric. For each second-field
            // row, compute the vertical average of its two preserved-parity
            // neighbors and take the L1 error. A progressive reconstruction
            // has both errors near zero; a combed reconstruction has them high.
            // nt is the noise-floor below which we treat the residual as zero.
            const int interp1 = (v0 + v2 + 1) >> 1;
            const int interp2 = (v2 + v4 + 1) >> 1;
            const int diff1   = abs(v1 - interp1);
            const int diff2   = abs(v3 - interp2);
            const int diff    = diff1 + diff2;
            const uint diff_u = (diff > nt) ? (uint)diff : 0u;

            // Combing-count at the other-field row v1. A row is "combed"
            // iff it is a local vertical extremum between the two
            // preserved-parity neighbors (same-sign deviation from both) AND
            // each deviation exceeds the tolerance T. Sign equality is
            // tested via (a<0)==(b<0) rather than (a*b>0) — the product
            // form overflows int32 for 16-bit pixel deltas (65535^2 ≈ 4.29e9,
            // int32 max ≈ 2.15e9); the boolean sign-equality form is exact
            // at every bit depth and is guaranteed non-zero here because
            // |d| > T ≥ 1 implies d != 0.
            //
            // 2026-04-24: ONE second-field row (v1) per thread. The previous
            // implementation also tested v3 via c2 — but v3 at iy is the v1
            // of the thread at iy+2, and that thread computes the identical
            // predicate on the same (row, row-1, row+1) triplet. Both fires
            // accumulated into the same WG sum, double-counting every
            // interior bot row. See analysis_pixel.txt for the full trace.
            // Keep v2/v3/v4 reads: they are still required by interp2 and
            // diff2 in the match_quality path above.
            const int d10 = v1 - v0, d12 = v1 - v2;
            const int c1 = (abs(d10) > T) && (abs(d12) > T) && ((d10 < 0) == (d12 < 0));
            const uint comb_u = (uint)c1;

            if (m == 0)      { mC = diff_u; cC = comb_u; }
            else if (m == 1) { mP = diff_u; cP = comb_u; }
            else             { mN = diff_u; cN = comb_u; }
        }
    }

    // WG block-sum reduction (6 lanes in parallel).
    __local uint lred[ivtc_block_x * ivtc_block_y * 6];
    lred[tid + 0 * WG_SIZE] = mC;
    lred[tid + 1 * WG_SIZE] = mP;
    lred[tid + 2 * WG_SIZE] = mN;
    lred[tid + 3 * WG_SIZE] = cC;
    lred[tid + 4 * WG_SIZE] = cP;
    lred[tid + 5 * WG_SIZE] = cN;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int k = 0; k < 6; k++) {
                lred[tid + k * WG_SIZE] += lred[tid + s + k * WG_SIZE];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        const int wg_idx = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        for (int k = 0; k < 6; k++) {
            scores[wg_idx * 9 + k] = lred[k * WG_SIZE];
        }
        // SUB-PHASE 1: emit 3 binary "block combed?" flags per candidate.
        // Each flag is 1 iff the block's combed-pixel sum for that candidate
        // meets BLOCK_COMB_THRESH, else 0. Host-side aggregation SUMS these
        // across WGs to yield a per-frame combed-block count (not a pixel
        // count) — TFM-style primary selection signal.
        const uint blockSumC = lred[3 * WG_SIZE];
        const uint blockSumP = lred[4 * WG_SIZE];
        const uint blockSumN = lred[5 * WG_SIZE];
        scores[wg_idx * 9 + 6] = (blockSumC >= (uint)BLOCK_COMB_THRESH) ? 1u : 0u;
        scores[wg_idx * 9 + 7] = (blockSumP >= (uint)BLOCK_COMB_THRESH) ? 1u : 0u;
        scores[wg_idx * 9 + 8] = (blockSumN >= (uint)BLOCK_COMB_THRESH) ? 1u : 0u;
    }
}

// SAD between two frames (for decimation). Per-WG sum reduced to a single uint per WG.
// Host takes MAX across WGs for block-max similarity.
__attribute__((reqd_work_group_size(ivtc_block_x, ivtc_block_y, 1)))
__kernel void kernel_ivtc_frame_diff(
    const __global uchar *restrict pA,
    const __global uchar *restrict pB,
    const int srcPitch,
    const int width, const int height,
    __global uint *restrict diffOut
) {
    const int thx = get_local_id(0);
    const int thy = get_local_id(1);
    const int ix  = get_global_id(0);
    const int iy  = get_global_id(1);
    const int tid = thy * ivtc_block_x + thx;
    const int WG_SIZE = ivtc_block_x * ivtc_block_y;

    uint d = 0u;
    if (ix < width && iy < height) {
        const int va = (int)(*(const __global Type *)(pA + iy * srcPitch + ix * sizeof(Type)));
        const int vb = (int)(*(const __global Type *)(pB + iy * srcPitch + ix * sizeof(Type)));
        d = (uint)abs(va - vb);
    }

    __local uint lred[ivtc_block_x * ivtc_block_y];
    lred[tid] = d;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) lred[tid] += lred[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        const int wg_idx = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        diffOut[wg_idx] = lred[0];
    }
}

// Synthesize one plane of output using the chosen match's row-sourcing scheme.
// apply_blend: when non-zero, the SECOND-field rows (odd rows for TFF, even for BFF)
// are replaced with BWDIF-style spatial cubic interpolation from the adjacent
// first-field rows — a significant quality upgrade over the simple vertical
// average used historically. Uses the BBC PH-2071 spatial-only (SP) coefficients
// (see kernel_ivtc_bwdif_deint below for the HF/LF variants; these are the
// published PH-2071 values, shared across BWDIF implementations):
//     sp = (5077*(rowU+rowL) - 981*(rowU3+rowL3) + 4096) >> 13
// with a clamp to the corridor [min(rowU,rowL), max(rowU,rowL)] to suppress
// overshoots on high-contrast edges. Falls back to the simple average on rows
// where the ±3 context is unavailable (top/bottom 3 rows).
#ifndef IVTC_W3F_SP0
#define IVTC_W3F_SP0   5077
#define IVTC_W3F_SP1    981
#define IVTC_W3F_SHIFT   13
#define IVTC_W3F_ROUND (1 << (IVTC_W3F_SHIFT - 1))  // = 4096
#endif

__attribute__((reqd_work_group_size(ivtc_block_x, ivtc_block_y, 1)))
__kernel void kernel_ivtc_synthesize(
    __global uchar *restrict pDst, const int dstPitch,
    const int dstWidth, const int dstHeight,
    const __global uchar *restrict pPrev,
    const __global uchar *restrict pCur,
    const __global uchar *restrict pNext,
    const int srcPitch,
    const int tff, const int match,
    const int apply_blend,
    const int dthresh       // per-pixel deinterlace gate; 0 disables (legacy whole-row replacement)
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix < dstWidth && iy < dstHeight) {
        int out_val;
        const int is_first_field_row = (iy & 1) == (tff ? 0 : 1);
        if (apply_blend && !is_first_field_row) {
            // ±1 rows are always safe to read (second-field rows are never at
            // iy=0 or iy=height-1 for even-height streams with bob semantics).
            const int rowU = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy - 1, dstWidth, dstHeight, tff, match);
            const int rowL = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy + 1, dstWidth, dstHeight, tff, match);

            // Per-pixel gate: only modify pixels whose actual value diverges
            // from the simple two-tap vertical interpolation by more than
            // dthresh. Pixels within the dthresh band are left as the
            // match-chosen value (already computed via pix_match at iy).
            // dthresh==0 disables the gate (legacy unconditional replacement).
            const int original    = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy, dstWidth, dstHeight, tff, match);
            const int interpTwoTap = (rowU + rowL + 1) >> 1;
            const int residual    = abs(original - interpTwoTap);
            if (dthresh > 0 && residual <= dthresh) {
                // Pixel is NOT combed — pass the match-chosen value through.
                out_val = original;
            } else {
                // Combed pixel. Smooth-blend between temporal (stable on
                // static / low-motion content) and spatial SP cubic
                // (ghost-free on motion) via a linear interpolation weighted
                // by motion level. Motion metric = max(|prev-cur|, |next-cur|)
                // at (ix, iy), which catches motion in EITHER direction
                // around cur — more responsive than the prev-vs-next
                // symmetric difference used previously. Reads at iy are
                // opposite-parity in merged frames; acceptable because the
                // temporal branch collapses to ~original on low motion
                // (prev ≈ cur ≈ next) and is blended away on high motion.
                //
                // motion_thresh = 2 * dthresh decouples the temporal motion
                // response from the spatial combing tolerance: the two
                // measure different phenomena and work on different scales.
                // dthresh=0 disables the gate and falls through to pure
                // spatial for legacy whole-row behaviour.
                const int prev_val      = ivtc_readPix(pPrev, ix, iy, srcPitch, dstWidth, dstHeight);
                const int next_val      = ivtc_readPix(pNext, ix, iy, srcPitch, dstWidth, dstHeight);
                const int motion_raw    = max(abs(prev_val - original), abs(next_val - original));
                // Motion noise floor: subtract ~1/8 of dthresh so camera-sensor
                // noise / compression noise doesn't push motion above zero on
                // genuinely-static content. dthresh is already bit-depth-scaled
                // at the kernel param boundary. Clamped to ≥ 0.
                const int noise_floor   = max(1, dthresh >> 3);
                const int motion        = max(0, motion_raw - noise_floor);
                const int motion_thresh = dthresh * 2;

                // Spatial SP cubic (used at high motion and as edge fallback).
                const int hasFullCtx = (iy >= 3) && (iy < dstHeight - 3);
                int spatial;
                if (hasFullCtx) {
                    const int rowU3 = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy - 3, dstWidth, dstHeight, tff, match);
                    const int rowL3 = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy + 3, dstWidth, dstHeight, tff, match);
                    spatial = (IVTC_W3F_SP0 * (rowU + rowL) - IVTC_W3F_SP1 * (rowU3 + rowL3) + IVTC_W3F_ROUND) >> IVTC_W3F_SHIFT;
                } else {
                    spatial = interpTwoTap;
                }

                // Temporal 4-way blend (dominant on low motion).
                const int temporal = (prev_val + next_val + 2 * original + 2) >> 2;

                // Smooth adaptive mix with QUADRATIC weighting:
                // w (linear)   : clamp(motion, 0, motion_thresh)
                // w2 (quadratic): w² / motion_thresh
                // Both endpoints stay pinned (w=0 → w2=0 → 100% temporal;
                // w=motion_thresh → w2=motion_thresh → 100% spatial), but
                // the curve stays closer to 0 for small motion — biases
                // toward temporal stability on low / slowly-accelerating
                // motion, then ramps up faster near the threshold. One
                // extra multiply per combed pixel (w*w fits int32 comfortably:
                // motion_thresh ≤ 510 → w² ≤ 260100 << INT_MAX).
                int blend_result;
                if (dthresh > 0 && motion_thresh > 0) {
                    const int w  = clamp(motion, 0, motion_thresh);
                    const int w2 = (w * w + (motion_thresh >> 1)) / motion_thresh;
                    blend_result = (temporal * (motion_thresh - w2)
                                  + spatial  * w2
                                  + (motion_thresh >> 1))      // round-to-nearest
                                  / motion_thresh;
                } else {
                    blend_result = spatial;
                }

                // Softened corridor clamp. The classic clamp to
                // [min(rowU,rowL), max(rowU,rowL)] is too aggressive on
                // smooth gradients: a blend that slightly overshoots past
                // the strictly-bounded corridor (within sensor noise) gets
                // snapped back, visible as subtle quantisation banding.
                // Widen the corridor by a dthresh-proportional epsilon:
                //   epsilon = max(1, dthresh/2), capped at max_eps
                //   max_eps = max(8, 8 << (bit_depth - 8))
                // The max() floor in max_eps guards against underflow when
                // bit_depth < 8 (not expected, but defensive).
                const int rowMin   = (rowU < rowL) ? rowU : rowL;
                const int rowMax   = (rowU > rowL) ? rowU : rowL;
                const int max_eps  = max(8, 8 << (bit_depth - 8));
                int epsilon        = max(1, dthresh >> 1);
                if (epsilon > max_eps) epsilon = max_eps;
                const int pixMax   = (1 << bit_depth) - 1;
                const int lo       = max(0,      rowMin - epsilon);
                const int hi       = min(pixMax, rowMax + epsilon);
                out_val = clamp(blend_result, lo, hi);
            }
        } else {
            out_val = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy, dstWidth, dstHeight, tff, match);
        }
        // No final saturation needed: the corridor clamp bounds the SP result
        // inside [min(rowU,rowL), max(rowU,rowL)] which is already a subset
        // of the source pixel range; the avg and pass-through branches are
        // similarly bounded by their inputs.

        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)out_val;
    }
}


// --- Full BWDIF deinterlacer for IVTC ------------------------------------
// Used only when combing remains after field-matching and the IVTC
// post-processing gate decides reconstruction is needed. IVTC owns a 5-frame
// ring (prev2/prev/cur/next/next2) so the full BBC PH-2071 temporal window
// is available. During ring startup (first 2 frames) or drain (last 2)
// prev2/next2 alias to prev/next — the caller in rgy_filter_ivtc.cpp
// handles the aliasing; kernel math degrades to a 3-frame approximation
// just like BWDIF does before its ring fills.
//
// Preserved-field rows are sourced from pCur.
// Missing-field rows run the full motion-adaptive w3fdif reconstruction.
//
// FIELD-PARITY CORRECTION (2026-04-21):
// Input frames are MERGED (ffmpeg decoder emits frames with both fields
// interleaved row-by-row). For TFF, even rows = top field at time T,
// odd rows = bot field at time T + field_period. The missing-row iy of
// cur is at the opposite parity (bot at odd iy for TFF); rowU / rowL
// (cur[iy±1]) happen to fall on preserved-parity rows and are fine.
// BUT: reading prev/next/prev2/next2 DIRECTLY at iy (or iy±2, iy±4)
// lands on OPPOSITE-parity rows in those frames — i.e. bot-field data
// from a different temporal moment, not the top-field temporal reference
// BWDIF's math was derived for (the BBC PH-2071 paper assumes separated
// fields). Using those values as p2_0 / n2_0 / p2_m2 / ... systematically
// injects cross-field, cross-time data into the reconstruction, visible
// as residual ghosting and row-alternating brightness shifts.
//
// Fix: for every reference-frame read at an EVEN offset from iy (iy, iy±2,
// iy±4 — all opposite-parity positions), estimate the same-parity value
// via vertical average of the adjacent iy±(1) preserved-parity rows.
// Reference-frame reads at ODD offsets (iy±1, iy±3) already land on
// preserved-parity rows and stay as direct reads. See merged-frame field
// deinterlace literature for the analogous same-parity convention (the
// preserved field is at every other row; temporal references at the
// missing-parity row index are estimated by vertical averaging).
//
// Kernel matches the 13-bit fixed-point coefficients defined above
// (IVTC_W3F_SP0/SP1/SHIFT), extended with the HF and LF coefficients from
// BWDIF proper.
#ifndef IVTC_W3F_LF0
#define IVTC_W3F_LF0   4309
#define IVTC_W3F_LF1    213
#define IVTC_W3F_HF0   5570
#define IVTC_W3F_HF1   3801
#define IVTC_W3F_HF2   1016
#endif

__attribute__((reqd_work_group_size(ivtc_block_x, ivtc_block_y, 1)))
__kernel void kernel_ivtc_bwdif_deint(
    __global uchar *restrict pDst, const int dstPitch,
    const int dstWidth, const int dstHeight,
    const __global uchar *restrict pPrev2,
    const __global uchar *restrict pPrev,
    const __global uchar *restrict pCur,
    const __global uchar *restrict pNext,
    const __global uchar *restrict pNext2,
    const int srcPitch,
    const int tff,
    const int scene_change,  // 1 = scene-change frame or immediate-next frame;
                             //     skip all temporal reads, use spatial-only SP cubic
    const int dthresh        // per-pixel deinterlace gate; 0 disables
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= dstWidth || iy >= dstHeight) return;

    // Preserved-field parity: TFF → even rows preserved, BFF → odd rows preserved.
    const int preservedParity = tff ? 0 : 1;
    const int needsInterp     = ((iy & 1) != preservedParity);

    __global Type *dstPix = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));

    if (!needsInterp) {
        // First-field row (preserved): straight from the current frame.
        dstPix[0] = (Type)ivtc_readPix(pCur, ix, iy, srcPitch, dstWidth, dstHeight);
        return;
    }

    // --- Missing-field row: reconstruction ---
    // rowU / rowL land on preserved-parity rows in cur → direct read is correct.
    const int rowU = ivtc_readPix(pCur,   ix, iy - 1, srcPitch, dstWidth, dstHeight);
    const int rowL = ivtc_readPix(pCur,   ix, iy + 1, srcPitch, dstWidth, dstHeight);

    // Per-pixel deinterlace gate: clean pixels (small residual vs simple
    // two-tap interpolation) pass through unchanged. dthresh=0 disables.
    const int originalPix  = ivtc_readPix(pCur, ix, iy, srcPitch, dstWidth, dstHeight);
    const int interpTwoTap = (rowU + rowL + 1) >> 1;
    if (dthresh > 0 && abs(originalPix - interpTwoTap) <= dthresh) {
        dstPix[0] = (Type)originalPix;
        return;
    }

    // Scene-change path: prev/next (and prev2/next2) belong to a different
    // scene, so any temporal reference would produce ghosting overlaying two
    // scenes. Use spatial-only SP cubic (BBC PH-2071 SP branch) with corridor
    // clamp — identical output to the apply_blend path in kernel_ivtc_synthesize.
    if (scene_change) {
        int sp;
        if ((iy >= 3) && (iy < dstHeight - 3)) {
            const int rowU3 = ivtc_readPix(pCur, ix, iy - 3, srcPitch, dstWidth, dstHeight);
            const int rowL3 = ivtc_readPix(pCur, ix, iy + 3, srcPitch, dstWidth, dstHeight);
            sp = (IVTC_W3F_SP0 * (rowU + rowL) - IVTC_W3F_SP1 * (rowU3 + rowL3) + IVTC_W3F_ROUND) >> IVTC_W3F_SHIFT;
            // Softened corridor clamp (same form as kernel_ivtc_synthesize).
            // Widens [min(rowU,rowL), max(rowU,rowL)] by a dthresh-proportional
            // epsilon so sub-noise overshoots aren't snapped back to the
            // strict corridor, which would cause subtle banding on gradients.
            const int rowMin  = (rowU < rowL) ? rowU : rowL;
            const int rowMax  = (rowU > rowL) ? rowU : rowL;
            const int max_eps = max(8, 8 << (bit_depth - 8));
            int epsilon       = max(1, dthresh >> 1);
            if (epsilon > max_eps) epsilon = max_eps;
            const int pixMax  = (1 << bit_depth) - 1;
            const int lo      = max(0,      rowMin - epsilon);
            const int hi      = min(pixMax, rowMax + epsilon);
            sp = clamp(sp, lo, hi);
        } else {
            sp = interpTwoTap;
        }
        dstPix[0] = (Type)sp;
        return;
    }

    // --- Full motion-adaptive BWDIF reconstruction (non-scene-change) ---
    // p2_0 / n2_0: same-parity (preserved-field) temporal references from
    // prev2 / next2 at spatial row iy. iy is opposite-parity in those frames,
    // so interpolate from iy±1 (their preserved-parity rows).
    const int p2_0 = ivtc_readPixSameParity(pPrev2, ix, iy, srcPitch, dstWidth, dstHeight);
    const int n2_0 = ivtc_readPixSameParity(pNext2, ix, iy, srcPitch, dstWidth, dstHeight);
    const int tAvg = (p2_0 + n2_0) >> 1;

    // prev/next at iy±1 land on preserved-parity rows → direct reads.
    const int pUp  = ivtc_readPix(pPrev, ix, iy - 1, srcPitch, dstWidth, dstHeight);
    const int pDn  = ivtc_readPix(pPrev, ix, iy + 1, srcPitch, dstWidth, dstHeight);
    const int nUp  = ivtc_readPix(pNext, ix, iy - 1, srcPitch, dstWidth, dstHeight);
    const int nDn  = ivtc_readPix(pNext, ix, iy + 1, srcPitch, dstWidth, dstHeight);

    const int motA = abs(p2_0 - n2_0);
    const int motB = (abs(pUp - rowU) + abs(pDn - rowL)) >> 1;
    const int motC = (abs(nUp - rowU) + abs(nDn - rowL)) >> 1;
    int motion = max(motA >> 1, max(motB, motC));

    // Motion threshold is 0 for IVTC's post-path: we only reach this kernel
    // when post=2 has already decided reconstruction is warranted. For
    // effectively-static content (motion==0) the tAvg fallback is correct.
    if (motion == 0) {
        dstPix[0] = (Type)tAvg;
        return;
    }

    const int hasSpatBounds = (iy >= 2) && (iy < dstHeight - 2);
    const int hasFullCtx    = (iy >= 4) && (iy < dstHeight - 4);

    // ±2 prev2/next2 samples shared between bound-tightening and HF coeff.
    // iy±2 is also opposite-parity → same-parity average.
    const int p2_m2 = ivtc_readPixSameParity(pPrev2, ix, iy - 2, srcPitch, dstWidth, dstHeight);
    const int p2_p2 = ivtc_readPixSameParity(pPrev2, ix, iy + 2, srcPitch, dstWidth, dstHeight);
    const int n2_m2 = ivtc_readPixSameParity(pNext2, ix, iy - 2, srcPitch, dstWidth, dstHeight);
    const int n2_p2 = ivtc_readPixSameParity(pNext2, ix, iy + 2, srcPitch, dstWidth, dstHeight);

    int localMotion = motion;
    if (hasSpatBounds) {
        const int spreadU = ((p2_m2 + n2_m2) >> 1) - rowU;
        const int spreadL = ((p2_p2 + n2_p2) >> 1) - rowL;
        const int dU      = tAvg - rowU;
        const int dL      = tAvg - rowL;
        const int hiSet   = max(dL, max(dU, min(spreadU, spreadL)));
        const int loSet   = min(dL, min(dU, max(spreadU, spreadL)));
        localMotion = max(localMotion, max(loSet, -hiSet));
    }

    int spatial;
    if (hasFullCtx) {
        // iy±4 still opposite-parity → same-parity average.
        const int p2_m4 = ivtc_readPixSameParity(pPrev2, ix, iy - 4, srcPitch, dstWidth, dstHeight);
        const int p2_p4 = ivtc_readPixSameParity(pPrev2, ix, iy + 4, srcPitch, dstWidth, dstHeight);
        const int n2_m4 = ivtc_readPixSameParity(pNext2, ix, iy - 4, srcPitch, dstWidth, dstHeight);
        const int n2_p4 = ivtc_readPixSameParity(pNext2, ix, iy + 4, srcPitch, dstWidth, dstHeight);
        // cur at iy±3 IS preserved-parity (iy odd → iy±3 even) → direct read.
        const int curU3 = ivtc_readPix(pCur,  ix, iy - 3, srcPitch, dstWidth, dstHeight);
        const int curD3 = ivtc_readPix(pCur,  ix, iy + 3, srcPitch, dstWidth, dstHeight);

        const int verticalEdge = abs(rowU - rowL);
        if (verticalEdge > motA) {
            const int hf = ( IVTC_W3F_HF0 * (p2_0 + n2_0)
                           - IVTC_W3F_HF1 * (p2_m2 + n2_m2 + p2_p2 + n2_p2)
                           + IVTC_W3F_HF2 * (p2_m4 + n2_m4 + p2_p4 + n2_p4)) >> 2;
            spatial = (hf + IVTC_W3F_LF0 * (rowU + rowL) - IVTC_W3F_LF1 * (curU3 + curD3)) >> IVTC_W3F_SHIFT;
        } else {
            spatial = (IVTC_W3F_SP0 * (rowU + rowL) - IVTC_W3F_SP1 * (curU3 + curD3)) >> IVTC_W3F_SHIFT;
        }
    } else {
        // Edge rows: simple vertical average (rounded to match the scene-
        // change and synthesize edge paths so all three stay byte-identical).
        spatial = (rowU + rowL + 1) >> 1;
    }

    // Smooth adaptive temporal/spatial blend (replaces the early-return
    // hard-switch that used motA < dthresh). Motion metric:
    //   motHybrid = max(|p2_0 - original|, |n2_0 - original|)
    // where p2_0 and n2_0 are same-parity averaged reads from prev2/next2
    // and original = cur[ix,iy]. Catches motion in EITHER direction
    // around cur — more responsive than the |p2_0 - n2_0| symmetric
    // difference (motA) used by the BWDIF motion-corridor logic above.
    //
    // motion_thresh = 2 * dthresh decouples the temporal-motion scale
    // from the spatial-combing tolerance.
    //
    // QUADRATIC weighting (w2 = w²/motion_thresh) biases toward the
    // temporal branch on low / slowly-changing motion: the curve stays
    // close to 0 for small w, then ramps up fast near the threshold.
    // Endpoints are pinned at 0 and motion_thresh, so full-static is
    // still 100% temporal and full-threshold is still 100% spatial —
    // just with a slower-at-start transition in between.
    //
    // dthresh=0 preserves the original full-BWDIF behaviour for bit-exact
    // back-compat (pure spatial w3fdif with motion corridor below).
    int interp;
    if (dthresh > 0) {
        const int temporal      = (p2_0 + n2_0 + 2 * originalPix + 2) >> 2;
        const int motHybrid_raw = max(abs(p2_0 - originalPix), abs(n2_0 - originalPix));
        // Motion noise floor (see kernel_ivtc_synthesize for rationale).
        const int noise_floor_b = max(1, dthresh >> 3);
        const int motHybrid     = max(0, motHybrid_raw - noise_floor_b);
        const int motion_thresh = dthresh * 2;
        const int w             = clamp(motHybrid, 0, motion_thresh);
        const int w2            = (w * w + (motion_thresh >> 1)) / motion_thresh;
        interp = (temporal * (motion_thresh - w2)
                + spatial  * w2
                + (motion_thresh >> 1))     // round-to-nearest
                / motion_thresh;
    } else {
        interp = spatial;
    }

    // Motion corridor around the temporal average (existing BWDIF step).
    interp = clamp(interp, tAvg - localMotion, tAvg + localMotion);
    // Final safety clamp to pixel range. The motion corridor can dip below
    // 0 or above max_val when tAvg and localMotion straddle the edge (e.g.
    // tAvg=3, localMotion=20 → corridor [-17, 23], an interp value of -10
    // would cast to uchar as 246). Mirrors the final clamp in bwdif.cl:185.
    const int ivtcMaxVal = (1 << bit_depth) - 1;
    interp = clamp(interp, 0, ivtcMaxVal);
    dstPix[0] = (Type)interp;
}
