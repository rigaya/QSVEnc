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
// H.264 deblocking filter -- ITU-T Rec. H.264 (V14, 2022-08) section 8.7,
// non-strong filter (boundary strength = 1). Applied as a spatial post-
// process across every 4x4 transform-block edge. Two passes, in-place
// on a destination buffer that the host pre-copies from source:
//
//   Pass 1 -- deblock_vertical:   filters across vertical edges
//                                 (one thread per (edge_index, row))
//   Pass 2 -- deblock_horizontal: filters across horizontal edges
//                                 (one thread per (column, edge_index))
//
// Adjacent edges are spaced 4 pixels apart and the filter only modifies
// pixels within 2 of the edge, so neighbouring edges' write ranges do
// not overlap -- in-place updates are race-free within a pass.
//
// Build-time defines (from rgy_filter_deblock.cpp):
//   Type        : uchar (8-bit) or ushort (>8-bit)
//   bit_depth   : source bit depth (8, 10, 12, 14, 16)
//   max_val     : (1 << bit_depth) - 1
//
// Kernel-arg thresholds (alpha, beta, tc0) are already host-scaled to
// the working bit depth.

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

// Helper: clip a signed delta to [-tc, +tc].
inline int clip3(int v, int lo, int hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// =============================================================================
// deblock_vertical -- filter across vertical edges at columns 4, 8, 12, ...
// =============================================================================
//
// Each thread handles one (edge, row) pair:
//   edge_index in [0, num_vert_edges - 1] -> boundary at column (edge_index+1)*4
//   row        in [0, height - 1]
//
// For interior edges (boundary column in [3, width-3]) all six neighbour
// reads land inside the frame, so no clamping is required for valid
// sources. We still range-check defensively.
//
// is_chroma == 0 -> luma path: tc = tc0 + 1 if (ap < beta) or (aq < beta),
//                              p1/q1 update conditional on ap/aq < beta.
// is_chroma != 0 -> chroma path: tc = tc0 + 1, p1/q1 left untouched.
__kernel void deblock_vertical(
    __global uchar *pBuf, const int bufPitch,
    const int width, const int height,
    const int alpha, const int beta, const int tc0,
    const int is_chroma
) {
    const int edge_index = get_global_id(0);
    const int iy         = get_global_id(1);
    if (iy >= height) return;

    const int boundary_x = (edge_index + 1) * 4;
    if (boundary_x < 3 || boundary_x > width - 3) return;

    __global Type *row = (__global Type *)(pBuf + iy * bufPitch);

    const int p2 = (int)row[boundary_x - 3];
    const int p1 = (int)row[boundary_x - 2];
    const int p0 = (int)row[boundary_x - 1];
    const int q0 = (int)row[boundary_x    ];
    const int q1 = (int)row[boundary_x + 1];
    const int q2 = (int)row[boundary_x + 2];

    // §8.7.2.1 boundary filtering strength conditions.
    const int abs_p0q0 = abs(p0 - q0);
    const int abs_p1p0 = abs(p1 - p0);
    const int abs_q1q0 = abs(q1 - q0);
    if (abs_p0q0 >= alpha || abs_p1p0 >= beta || abs_q1q0 >= beta) {
        return; // boundary detection failed; leave row untouched
    }

    // §8.7.2.3 non-strong (bS in 1..3) filter -- central p0/q0 update.
    // delta = clip3(-tc, +tc, ((q0 - p0) << 2 + (p1 - q1) + 4) >> 3)
    // tc = tc0 for luma when ap<beta, aq<beta both fail; tc = tc0+ap_flag+aq_flag otherwise.
    // For chroma, tc = tc0 + 1 unconditionally (chromaEdgeFlag path).
    const int ap = abs(p2 - p0);
    const int aq = abs(q2 - q0);
    int tc;
    if (is_chroma != 0) {
        tc = tc0 + 1;
    } else {
        tc = tc0 + ((ap < beta) ? 1 : 0) + ((aq < beta) ? 1 : 0);
    }

    const int delta_raw = ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3;
    const int delta     = clip3(delta_raw, -tc, tc);
    const int p0_new    = clamp(p0 + delta, 0, max_val);
    const int q0_new    = clamp(q0 - delta, 0, max_val);

    row[boundary_x - 1] = (Type)p0_new;
    row[boundary_x    ] = (Type)q0_new;

    // §8.7.2.3 -- inner-pixel updates p1' and q1'. Luma only.
    if (is_chroma == 0) {
        const int pq_avg = (p0 + q0 + 1) >> 1;
        if (ap < beta) {
            const int p1_delta = clip3((p2 + pq_avg - (p1 << 1)) >> 1, -tc0, tc0);
            row[boundary_x - 2] = (Type)clamp(p1 + p1_delta, 0, max_val);
        }
        if (aq < beta) {
            const int q1_delta = clip3((q2 + pq_avg - (q1 << 1)) >> 1, -tc0, tc0);
            row[boundary_x + 1] = (Type)clamp(q1 + q1_delta, 0, max_val);
        }
    }
}

// =============================================================================
// deblock_horizontal -- filter across horizontal edges at rows 4, 8, 12, ...
// =============================================================================
// Same algorithm as deblock_vertical, transposed: thread layout is
// (column, edge_index) and the six samples are read vertically across
// the edge.
__kernel void deblock_horizontal(
    __global uchar *pBuf, const int bufPitch,
    const int width, const int height,
    const int alpha, const int beta, const int tc0,
    const int is_chroma
) {
    const int ix         = get_global_id(0);
    const int edge_index = get_global_id(1);
    if (ix >= width) return;

    const int boundary_y = (edge_index + 1) * 4;
    if (boundary_y < 3 || boundary_y > height - 3) return;

    #define PIX(y) (*(__global Type *)(pBuf + (y) * bufPitch + ix * sizeof(Type)))
    const int p2 = (int)PIX(boundary_y - 3);
    const int p1 = (int)PIX(boundary_y - 2);
    const int p0 = (int)PIX(boundary_y - 1);
    const int q0 = (int)PIX(boundary_y    );
    const int q1 = (int)PIX(boundary_y + 1);
    const int q2 = (int)PIX(boundary_y + 2);

    const int abs_p0q0 = abs(p0 - q0);
    const int abs_p1p0 = abs(p1 - p0);
    const int abs_q1q0 = abs(q1 - q0);
    if (abs_p0q0 >= alpha || abs_p1p0 >= beta || abs_q1q0 >= beta) {
        return;
    }

    const int ap = abs(p2 - p0);
    const int aq = abs(q2 - q0);
    int tc;
    if (is_chroma != 0) {
        tc = tc0 + 1;
    } else {
        tc = tc0 + ((ap < beta) ? 1 : 0) + ((aq < beta) ? 1 : 0);
    }

    const int delta_raw = ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3;
    const int delta     = clip3(delta_raw, -tc, tc);
    PIX(boundary_y - 1) = (Type)clamp(p0 + delta, 0, max_val);
    PIX(boundary_y    ) = (Type)clamp(q0 - delta, 0, max_val);

    if (is_chroma == 0) {
        const int pq_avg = (p0 + q0 + 1) >> 1;
        if (ap < beta) {
            const int p1_delta = clip3((p2 + pq_avg - (p1 << 1)) >> 1, -tc0, tc0);
            PIX(boundary_y - 2) = (Type)clamp(p1 + p1_delta, 0, max_val);
        }
        if (aq < beta) {
            const int q1_delta = clip3((q2 + pq_avg - (q1 << 1)) >> 1, -tc0, tc0);
            PIX(boundary_y + 1) = (Type)clamp(q1 + q1_delta, 0, max_val);
        }
    }
    #undef PIX
}
