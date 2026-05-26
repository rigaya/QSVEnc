// FineDehalo — fine-edge-aware halo removal kernels.
//
// Pipeline (luma plane only; chroma passes through unchanged):
//   1. DeHalo_alpha sub-filter           — produces `dehaloed`
//   2. finedehalo_prewitt                — Prewitt gradient + threshold ramp
//   3. dehalo_expand / dehalo_inpand     — 2 passes of 3×3 morph (r=1)
//   4. finedehalo_limitmask              — diff(src, dehaloed) ramp
//   5. finedehalo_combine                — invert em, AND with linemask, merge
//
// Build-time defines (set via -D from rgy_filter_finedehalo.cpp):
//   Type             : uchar (8-bit) or ushort (>8-bit)
//   bit_depth        : source bit depth (8, 10, 12, 14, 16)
//   max_val          : (1 << bit_depth) - 1
//   finedehalo_block_x / finedehalo_block_y : work-group dims
//
// HBD safety:
//   * Pixel storage uses Type. Intermediate arithmetic uses int / float.
//   * thmi/thma/thlimi/thlima are pre-scaled to working bit depth at the
//     host (8-bit → max_val).
//   * Mask is stored as Type in [0, max_val].
//   * Combine uses float arithmetic with a final clamp to [0, max_val].
//
// =============================================================================

inline int fdh_readPixClamp(const __global uchar *plane,
                             int x, int y,
                             int pitch, int width, int height) {
    if (x < 0)        x = 0;
    if (x >= width)   x = width  - 1;
    if (y < 0)        y = 0;
    if (y >= height)  y = height - 1;
    return (int)(*(const __global Type *)(plane + y * pitch + x * sizeof(Type)));
}

// =============================================================================
// SLM tile-load was tested for these 3x3 edge kernels on Arc A770 at
// 480p / 720p / 1080p / 4K (--vpp-perf-monitor, --trim 0:500). Result:
// SLM was indistinguishable from the global-memory path at every
// resolution. The 9x-redundant-read bandwidth model overestimates the
// real DRAM cost because Arc A770's 16 MB L2 covers the per-frame
// working set even at 4K, so the cooperative-load + workgroup-barrier
// overhead added by SLM is not offset by any bandwidth savings.
//
// See rgy_filter_maa.cl above maa_sangnom_smooth_local for the full
// measurement table (MAA was tested with the same sweep and showed a
// consistent ~75% regression; HQDering edge kernels showed +12% at
// 480p growing to +31% at 4K). Float4 vectorisation was skipped on
// the same reasoning -- L2-resident reads make load width irrelevant.
//
// Revisit only on hardware with a substantially smaller L2, or for
// operators with neighbourhoods large enough that the unique pixel
// footprint of a workgroup actually exceeds the L2 budget.
// =============================================================================

// =============================================================================
// finedehalo_prewitt — Prewitt gradient magnitude + sensitivity ramp.
//
// Per-pixel:
//   Gx = -p[-1,-1] - 2 p[-1,0] - p[-1,+1]  +  p[+1,-1] + 2 p[+1,0] + p[+1,+1]
//   Gy = -p[-1,-1] - 2 p[ 0,-1] - p[+1,-1]  +  p[-1,+1] + 2 p[ 0,+1] + p[+1,+1]
//   G  = |Gx| + |Gy|                     (cheap L1 approximation; the ramp
//                                          normalises the magnitude anyway)
//
//   if hi > lo: edges = clamp((G - lo) * max_val / (hi - lo), 0, max_val)
//   else      : edges = (G >= lo) ? max_val : 0
__kernel void finedehalo_prewitt(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height,
    int loScaled, int hiScaled
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int p00 = fdh_readPixClamp(pSrc, x - 1, y - 1, srcPitch, width, height);
    const int p10 = fdh_readPixClamp(pSrc, x    , y - 1, srcPitch, width, height);
    const int p20 = fdh_readPixClamp(pSrc, x + 1, y - 1, srcPitch, width, height);
    const int p01 = fdh_readPixClamp(pSrc, x - 1, y    , srcPitch, width, height);
    const int p21 = fdh_readPixClamp(pSrc, x + 1, y    , srcPitch, width, height);
    const int p02 = fdh_readPixClamp(pSrc, x - 1, y + 1, srcPitch, width, height);
    const int p12 = fdh_readPixClamp(pSrc, x    , y + 1, srcPitch, width, height);
    const int p22 = fdh_readPixClamp(pSrc, x + 1, y + 1, srcPitch, width, height);

    // Prewitt-style 3x3 convolution; the centre column / row are weighted 2
    // (matches the spec in rgy_filter_finedehalo.h — same as a Sobel kernel
    // in practice, but the names "Prewitt" and "Sobel" are interchangeable
    // in the AviSynth mt_edge "prewitt" mode used by the reference script).
    const int gx = -p00 - 2 * p01 - p02  +  p20 + 2 * p21 + p22;
    const int gy = -p00 - 2 * p10 - p20  +  p02 + 2 * p12 + p22;

    const int g = abs(gx) + abs(gy);

    int edges;
    if (hiScaled > loScaled) {
        long num = (long)(g - loScaled) * (long)max_val;
        int v = (int)(num / (long)(hiScaled - loScaled));
        if (v < 0)        v = 0;
        if (v > max_val)  v = max_val;
        edges = v;
    } else {
        edges = (g >= loScaled) ? max_val : 0;
    }

    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));
    dPix[0] = (Type)edges;
}

// =============================================================================
// Alternative edge operators (opt-in via --vpp-finedehalo edge=...).
//
// All operators normalise their L1 gradient magnitude to the SAME domain as
// the default finedehalo_prewitt above (peak ≈ 8 * max_val), so the user's
// lo/hi thresholds keep their meaning regardless of operator choice. The
// post-magnitude ramp is identical across operators (extracted into the
// FDH_EDGE_RAMP macro below).
//
// `sobel` shares the prewitt kernel — the existing finedehalo_prewitt is
// already centre-weighted (textbook Sobel math), so a separate sobel kernel
// would be byte-identical. Dispatch on `edge=sobel` therefore resolves to
// `finedehalo_prewitt` (see rgy_filter_finedehalo.cpp).
// =============================================================================

#define FDH_EDGE_RAMP(g_in_8maxval_domain)                                   \
    int edges;                                                               \
    if (hiScaled > loScaled) {                                               \
        long num = (long)((g_in_8maxval_domain) - loScaled) * (long)max_val; \
        int v = (int)(num / (long)(hiScaled - loScaled));                    \
        if (v < 0)        v = 0;                                             \
        if (v > max_val)  v = max_val;                                       \
        edges = v;                                                           \
    } else {                                                                 \
        edges = ((g_in_8maxval_domain) >= loScaled) ? max_val : 0;           \
    }                                                                        \
    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type)); \
    dPix[0] = (Type)edges

#define FDH_LOAD_3X3                                                         \
    const int tl = fdh_readPixClamp(pSrc, x - 1, y - 1, srcPitch, width, height); \
    const int tc = fdh_readPixClamp(pSrc, x    , y - 1, srcPitch, width, height); \
    const int tr = fdh_readPixClamp(pSrc, x + 1, y - 1, srcPitch, width, height); \
    const int cl = fdh_readPixClamp(pSrc, x - 1, y    , srcPitch, width, height); \
    const int cr = fdh_readPixClamp(pSrc, x + 1, y    , srcPitch, width, height); \
    const int bl = fdh_readPixClamp(pSrc, x - 1, y + 1, srcPitch, width, height); \
    const int bc = fdh_readPixClamp(pSrc, x    , y + 1, srcPitch, width, height); \
    const int br = fdh_readPixClamp(pSrc, x + 1, y + 1, srcPitch, width, height)

// Scharr — [3,10,3] weighting. Raw peak = 32*max_val; divide by 4 → 8*max_val.
__kernel void finedehalo_scharr(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height,
    int loScaled, int hiScaled
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    FDH_LOAD_3X3;
    const int gx = -3 * tl + 3 * tr - 10 * cl + 10 * cr - 3 * bl + 3 * br;
    const int gy = -3 * tl - 10 * tc - 3 * tr + 3 * bl + 10 * bc + 3 * br;
    const int g  = (abs(gx) + abs(gy)) / 4;
    FDH_EDGE_RAMP(g);
}

// Kirsch — max of 8 directional responses. Raw peak = 15*max_val;
// scale by 8/15 (integer) to land in 8*max_val domain.
__kernel void finedehalo_kirsch(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height,
    int loScaled, int hiScaled
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    FDH_LOAD_3X3;
    const int cc = fdh_readPixClamp(pSrc, x, y, srcPitch, width, height);
    (void)cc;
    const int n  =  5 * (tl + tc + tr) - 3 * (cl + cr + bl + bc + br);
    const int ne =  5 * (tc + tr + cr) - 3 * (tl + cl + bl + bc + br);
    const int e  =  5 * (tr + cr + br) - 3 * (tl + tc + cl + bl + bc);
    const int se =  5 * (cr + br + bc) - 3 * (tl + tc + tr + cl + bl);
    const int s  =  5 * (bl + bc + br) - 3 * (tl + tc + tr + cl + cr);
    const int sw =  5 * (cl + bl + bc) - 3 * (tl + tc + tr + cr + br);
    const int w  =  5 * (tl + cl + bl) - 3 * (tc + tr + cr + bc + br);
    const int nw =  5 * (tl + tc + cl) - 3 * (tr + cr + bl + bc + br);
    int m = max(max(max(n, ne), max(e, se)), max(max(s, sw), max(w, nw)));
    if (m < 0) m = 0;
    const int g = (m * 8) / 15;
    FDH_EDGE_RAMP(g);
}

// Laplacian (4-connected). Raw peak |L| = 4*max_val; multiply by 2 → 8*max_val.
__kernel void finedehalo_laplacian(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height,
    int loScaled, int hiScaled
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    const int cc = fdh_readPixClamp(pSrc, x    , y    , srcPitch, width, height);
    const int tc = fdh_readPixClamp(pSrc, x    , y - 1, srcPitch, width, height);
    const int cl = fdh_readPixClamp(pSrc, x - 1, y    , srcPitch, width, height);
    const int cr = fdh_readPixClamp(pSrc, x + 1, y    , srcPitch, width, height);
    const int bc = fdh_readPixClamp(pSrc, x    , y + 1, srcPitch, width, height);
    const int g  = abs(4 * cc - tc - cl - cr - bc) * 2;
    FDH_EDGE_RAMP(g);
}

// =============================================================================
// finedehalo_limitmask — only allow corrections that are large enough.
//
// Per-pixel:
//   diff      = max(src - dehaloed, 0)        // how much darker dehaloed is
//   linemask  = ramp(diff; thlimi, thlima)    // 0 below thlimi, max above thlima
//
// Reading both src and dehaloed at the same coordinate, so no neighbour
// loads are needed.
__kernel void finedehalo_limitmask(
    const __global uchar *pSrc,      int srcPitch,
    const __global uchar *pDehaloed, int dehPitch,
    __global       uchar *pDst,      int dstPitch,
    int width, int height,
    int thlimiScaled, int thlimaScaled
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int s = (int)(*(const __global Type *)(pSrc      + y * srcPitch + x * sizeof(Type)));
    const int d = (int)(*(const __global Type *)(pDehaloed + y * dehPitch + x * sizeof(Type)));

    int diff = s - d;
    if (diff < 0) diff = 0;

    int mask;
    if (thlimaScaled > thlimiScaled) {
        long num = (long)(diff - thlimiScaled) * (long)max_val;
        int v = (int)(num / (long)(thlimaScaled - thlimiScaled));
        if (v < 0)        v = 0;
        if (v > max_val)  v = max_val;
        mask = v;
    } else {
        mask = (diff >= thlimiScaled) ? max_val : 0;
    }

    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));
    dPix[0] = (Type)mask;
}

// =============================================================================
// finedehalo_combine — fused mask-AND + alpha-blend (steps 5+6).
//
// Per-pixel:
//   inv_em      = max_val - em
//   final_mask  = min(inv_em, linemask)
//   m           = final_mask / max_val           // 0..1
//   out         = src + (dehaloed - src) * m     (then clamp)
//
// `showmask` (0..4) selects the output:
//   0 → merged result     (the normal output)
//   1 → return src; host has copied raw `edges` to dst already (no-op here)
//   2 → return src; host has copied `em` to dst already
//   3 → return src; host has copied `linemask` to dst already
//   4 → return final_mask itself (so we can compute it in one pass)
__kernel void finedehalo_combine(
    const __global uchar *pSrc,      int srcPitch,
    const __global uchar *pDehaloed, int dehPitch,
    const __global uchar *pEm,       int emPitch,
    const __global uchar *pLineMask, int lmPitch,
    __global       uchar *pDst,      int dstPitch,
    int width, int height,
    int showmask
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int s  = (int)(*(const __global Type *)(pSrc      + y * srcPitch + x * sizeof(Type)));
    const int d  = (int)(*(const __global Type *)(pDehaloed + y * dehPitch + x * sizeof(Type)));
    const int em = (int)(*(const __global Type *)(pEm       + y * emPitch  + x * sizeof(Type)));
    const int lm = (int)(*(const __global Type *)(pLineMask + y * lmPitch  + x * sizeof(Type)));

    const int invEm = max_val - em;
    int finalMask = invEm < lm ? invEm : lm;

    int outVal;
    if (showmask == 4) {
        outVal = finalMask;
    } else {
        const float m = (float)finalMask / (float)max_val;
        const float blended = (float)s + ((float)d - (float)s) * m;
        int v = convert_int_rte(blended);
        if (v < 0)        v = 0;
        if (v > max_val)  v = max_val;
        outVal = v;
    }

    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));
    dPix[0] = (Type)outVal;
}
