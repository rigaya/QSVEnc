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
// Motion-adaptive deinterlacer (same-rate / frame mode).
//
// Algorithm: based on the BBC Research Paper PH-2071 (Weston 3-Field Deinterlacer, "w3fdif")
// combined with a temporal motion metric derived from ±1 / ±2 row diffs across prev/cur/next.
// The 13-bit fixed-point coefficients below are the published BBC w3fdif weights; numeric
// constants from a published research paper are not protectable. Implementation is an
// independent OpenCL rewrite.
//
// Per output pixel (missing-field row):
//   tAvg     = (prev2 + next2) / 2                            -- same-parity temporal reference
//   motion   = max(|prev2-next2|/2, motPC, motCN)             -- motion metric
//   if motion <= thr        -> out = tAvg                     -- still region, trust temporal
//   else                    -> spatial reconstruction, bounded by ±motion around tAvg
//
// Row dispatch (based on vertical context availability):
//   interior (y in [4, h-5]) : full HF/LF or SP weighted filter, spat-bounds tightening
//   edge_spat (y in [2, h-3]) : simple vertical avg, spat-bounds tightening
//   edge_flat (else)          : simple vertical avg, motion bounds only
//
// Build-time defines (set via -D by the host):
//   Type          : uchar (8-bit) or ushort (>8-bit)
//   bit_depth     : source bit depth
//   max_val       : (1 << bit_depth) - 1
//   bwdif_block_x : WG X dim
//   bwdif_block_y : WG Y dim

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

// --- BBC PH-2071 w3fdif coefficients (13-bit fixed point; published algorithm constants) ---
#define W3F_LF0   4309
#define W3F_LF1    213
#define W3F_HF0   5570
#define W3F_HF1   3801
#define W3F_HF2   1016
#define W3F_SP0   5077
#define W3F_SP1    981
#define W3F_SHIFT   13

// Clamp-read a single pixel from a plane at (x, y), interpreted as Type-sized samples.
static inline int readPix(
    const __global uchar *plane, int x, int y,
    const int pitch, const int width, const int height
) {
    x = clamp(x, 0, width  - 1);
    y = clamp(y, 0, height - 1);
    return (int)(*(const __global Type *)(plane + y * pitch + x * sizeof(Type)));
}

// Same-rate BWDIF kernel.
//   tff = 1 : TFF input (top field preserved = even-index rows).
//   tff = 0 : BFF input (bottom field preserved = odd-index rows).
//   thr     : noise threshold in pixel-value units; motion <= thr → pure temporal average.
__attribute__((reqd_work_group_size(bwdif_block_x, bwdif_block_y, 1)))
__kernel void kernel_bwdif_frame(
    __global uchar *restrict pDst,           const int dstPitch,
    const __global uchar *restrict pPrev2,
    const __global uchar *restrict pPrev,
    const __global uchar *restrict pCur,
    const __global uchar *restrict pNext,
    const __global uchar *restrict pNext2,
    const int srcPitch,
    const int width,
    const int height,
    const int tff,
    const int thr
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    // Preserved-field parity: TFF → even rows preserved, BFF → odd rows preserved.
    const int preservedParity = tff ? 0 : 1;
    const int needsInterp     = ((iy & 1) != preservedParity);

    __global Type *dstPix = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));

    if (!needsInterp) {
        // Preserved field: copy pixel directly from current frame.
        dstPix[0] = (Type)readPix(pCur, ix, iy, srcPitch, width, height);
        return;
    }

    // ----- Missing-field row: reconstruct via motion-adaptive w3fdif -----
    const int rowU = readPix(pCur,   ix, iy - 1, srcPitch, width, height); // row above target
    const int rowL = readPix(pCur,   ix, iy + 1, srcPitch, width, height); // row below target
    const int p2_0 = readPix(pPrev2, ix, iy,     srcPitch, width, height);
    const int n2_0 = readPix(pNext2, ix, iy,     srcPitch, width, height);
    const int tAvg = (p2_0 + n2_0) >> 1;

    const int pUp  = readPix(pPrev,  ix, iy - 1, srcPitch, width, height);
    const int pDn  = readPix(pPrev,  ix, iy + 1, srcPitch, width, height);
    const int nUp  = readPix(pNext,  ix, iy - 1, srcPitch, width, height);
    const int nDn  = readPix(pNext,  ix, iy + 1, srcPitch, width, height);

    // YADIF temporal motion metric (Michael Niedermayer, 2006). Three
    // independent estimates of motion at the missing-row position,
    // reduced to their peak. Independent re-implementation; the published
    // metric is widely shared across deinterlacers.
    //   - crossTimeFull spans 2*T (prev2 to next2 is one full input-frame
    //     interval), so it is halved before reduction so it is comparable
    //     with the same-time-step pair averages below.
    //   - prev / next pair averages are the mean across the two
    //     preserved-parity rows that bracket the target row.
    const int crossTimeFull    = abs(p2_0 - n2_0);              // reused for the verticalEdge gate below
    const int prevPairDeltaSum = abs(pUp - rowU) + abs(pDn - rowL);
    const int nextPairDeltaSum = abs(nUp - rowU) + abs(nDn - rowL);
    const int prevPairAvg      = prevPairDeltaSum >> 1;
    const int nextPairAvg      = nextPairDeltaSum >> 1;
    int motion = crossTimeFull >> 1;
    if (prevPairAvg > motion) motion = prevPairAvg;
    if (nextPairAvg > motion) motion = nextPairAvg;

    if (motion <= thr) {
        dstPix[0] = (Type)clamp(tAvg, 0, max_val);
        return;
    }

    const int hasSpatBounds = (iy >= 2) && (iy < height - 2);
    const int hasFullCtx    = (iy >= 4) && (iy < height - 4);

    // ±2 prev2/next2 samples: shared between the hasSpatBounds bound-tightening and
    // the hasFullCtx HF coefficient sum. Load once; readPix clamps x/y so ±2 indices
    // are safe for all iy even when hasSpatBounds is false (edge rows <2 rows from border).
    const int p2_m2 = readPix(pPrev2, ix, iy - 2, srcPitch, width, height);
    const int p2_p2 = readPix(pPrev2, ix, iy + 2, srcPitch, width, height);
    const int n2_m2 = readPix(pNext2, ix, iy - 2, srcPitch, width, height);
    const int n2_p2 = readPix(pNext2, ix, iy + 2, srcPitch, width, height);

    // Tighten the motion bound using the ±2 vertical spread of same-parity temporal refs.
    int localMotion = motion;
    if (hasSpatBounds) {
        // Spatial-spread motion-corridor refinement (algorithm from
        // YADIF / BWDIF lineage; published, independently re-implemented
        // here in OpenCL).
        //
        // Idea: the ±2 same-parity neighbors give two predicted drifts
        // for how the missing-row pixel could deviate from tAvg. Combine
        // those with the per-side tAvg-vs-cur drifts to derive an upper
        // and lower deviation bound, then widen localMotion to absorb
        // whichever bound is the larger absolute corridor edge.
        const int predDriftAboveCur = ((p2_m2 + n2_m2) >> 1) - rowU;
        const int predDriftBelowCur = ((p2_p2 + n2_p2) >> 1) - rowL;
        const int avgDriftAboveCur  = tAvg - rowU;
        const int avgDriftBelowCur  = tAvg - rowL;
        // Inner-spread bracket: the smaller of the two pred drifts (the
        // narrower predicted edge) and the larger.
        const int innerSpreadHi = (predDriftAboveCur < predDriftBelowCur) ? predDriftAboveCur : predDriftBelowCur;
        const int innerSpreadLo = (predDriftAboveCur > predDriftBelowCur) ? predDriftAboveCur : predDriftBelowCur;
        // Sequential reduction over the three deviation candidates.
        int upperBound = avgDriftAboveCur;
        if (avgDriftBelowCur > upperBound) upperBound = avgDriftBelowCur;
        if (innerSpreadHi    > upperBound) upperBound = innerSpreadHi;
        int lowerBound = avgDriftAboveCur;
        if (avgDriftBelowCur < lowerBound) lowerBound = avgDriftBelowCur;
        if (innerSpreadLo    < lowerBound) lowerBound = innerSpreadLo;
        // Absorb whichever side produces the larger corridor edge.
        const int spreadMargin = (lowerBound > -upperBound) ? lowerBound : -upperBound;
        if (spreadMargin > localMotion) localMotion = spreadMargin;
    }

    int interp;
    if (hasFullCtx) {
        // Full interior reconstruction: HF+LF weighted sum when vertical feature dominates,
        // else pure SP (spatial-only cubic).
        const int p2_m4 = readPix(pPrev2, ix, iy - 4, srcPitch, width, height);
        const int p2_p4 = readPix(pPrev2, ix, iy + 4, srcPitch, width, height);
        const int n2_m4 = readPix(pNext2, ix, iy - 4, srcPitch, width, height);
        const int n2_p4 = readPix(pNext2, ix, iy + 4, srcPitch, width, height);
        const int curU3 = readPix(pCur,   ix, iy - 3, srcPitch, width, height);
        const int curD3 = readPix(pCur,   ix, iy + 3, srcPitch, width, height);

        const int verticalEdge = abs(rowU - rowL);
        if (verticalEdge > crossTimeFull) {
            const int hf = ( W3F_HF0 * (p2_0 + n2_0)
                           - W3F_HF1 * (p2_m2 + n2_m2 + p2_p2 + n2_p2)
                           + W3F_HF2 * (p2_m4 + n2_m4 + p2_p4 + n2_p4)) >> 2;
            interp = (hf + W3F_LF0 * (rowU + rowL) - W3F_LF1 * (curU3 + curD3)) >> W3F_SHIFT;
        } else {
            interp = (W3F_SP0 * (rowU + rowL) - W3F_SP1 * (curU3 + curD3)) >> W3F_SHIFT;
        }
    } else {
        // Edge row (no ±4 context): simple vertical average.
        interp = (rowU + rowL) >> 1;
    }

    // Clamp interpolation to motion-derived corridor around the temporal average.
    interp = clamp(interp, tAvg - localMotion, tAvg + localMotion);
    interp = clamp(interp, 0, max_val);
    dstPix[0] = (Type)interp;
}
