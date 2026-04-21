// Type
// bit_depth
// ivtc_block_x
// ivtc_block_y

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

// Fetch a reconstructed pixel per candidate / row-parity.
// The "first-field" source is always from the primary frame, "second-field" from the borrowed frame.
// TFF (tff=1): first-field = top = even rows.
//   C: [cur.top, cur.bot]     — current frame as-is
//   P: [cur.top, prev.bot]    — first-field(top) from cur, second-field(bot) from prev
//   N: [next.top, cur.bot]    — first-field(top) from next, second-field(bot) from cur
// BFF (tff=0): first-field = bottom = odd rows.
//   C: [cur.top, cur.bot]     — same
//   P: [prev.top, cur.bot]    — first-field(bot) from cur, second-field(top) from prev
//   N: [cur.top, next.bot]    — first-field(bot) from next, second-field(top) from cur
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
//   match_quality  : sum of |top_sum3 - 1.5*bot_sum2| where diff > nt
//                    top_sum3 = v[iy] + v[iy+2] + v[iy+4] (three first-field rows)
//                    bot_sum2 = v[iy+1] + v[iy+3]         (two other-field rows)
//                    Measures how "non-progressive" the hypothetical reconstruction looks.
//                    Lower => better field match.
//
//   combing_count  : count of other-field-row pixels where the value sticks out ±T
//                    from BOTH adjacent first-field rows (zigzag pattern).
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
    __global uint *restrict scores        // 6 uints per WG: [mC, mP, mN, cC, cP, cN]
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
    if (ix < width && (iy & 1) == first_parity && iy + 4 < height) {
        // Unroll over candidates to inline pix_match without per-iteration branching penalty.
        for (int m = 0; m < 3; m++) {
            const int v0 = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy    , width, height, tff, m);
            const int v1 = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy + 1, width, height, tff, m);
            const int v2 = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy + 2, width, height, tff, m);
            const int v3 = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy + 3, width, height, tff, m);
            const int v4 = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy + 4, width, height, tff, m);

            // Match-quality: 3-top-sum vs 2-bot-sum (bot counted as 1.5x to balance row counts).
            const int bot_sum2 = v1 + v3;
            const int top_sum3 = v0 + v2 + v4;
            const int diff = abs(top_sum3 - (bot_sum2 >> 1) - bot_sum2);
            const uint diff_u = (diff > nt) ? (uint)diff : 0u;

            // Combing-count at the two other-field rows v1 and v3.
            const int hi1 = v1 + T, lo1 = v1 - T;
            const int hi3 = v3 + T, lo3 = v3 - T;
            const int c1 = (hi1 < v0 && hi1 < v2) || (lo1 > v0 && lo1 > v2);
            const int c2 = (hi3 < v2 && hi3 < v4) || (lo3 > v2 && lo3 > v4);
            const uint comb_u = (uint)(c1 + c2);

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
            scores[wg_idx * 6 + k] = lred[k * WG_SIZE];
        }
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
// are checked for residual combing and only combed pixels are replaced with the
// vertical average of the two adjacent first-field rows.
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
    const int T
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix < dstWidth && iy < dstHeight) {
        int out_val = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy, dstWidth, dstHeight, tff, match);
        const int is_first_field_row = (iy & 1) == (tff ? 0 : 1);
        if (apply_blend && !is_first_field_row) {
            const int hi = out_val + T;
            const int lo = out_val - T;
            const int v_m = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy - 1, dstWidth, dstHeight, tff, match);
            const int v_p = pix_match(pPrev, pCur, pNext, srcPitch, ix, iy + 1, dstWidth, dstHeight, tff, match);
            const int combed = (hi < v_m && hi < v_p) || (lo > v_m && lo > v_p);
            if (combed) {
                out_val = (v_m + v_p + 1) >> 1;
            }
        }
        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)out_val;
    }
}
