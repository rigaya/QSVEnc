// HQDering — DCT-ringing reduction kernels.
//
// Pipeline (luma plane only; chroma passes through):
//   1. hqdering_edge       — Sobel + Levels-style threshold ramp
//   2. dehalo_expand × mrad — 3×3 morphological dilation (ping-pong)
//      (opt) hqdering_inpand3x3 × minp   — edge core, excluded in combine
//      (opt) hqdering_mean3x3 × msmooth  — feather the ring mask
//   3. hqdering_blur_h     — horizontal 1-D Gaussian
//   4. hqdering_blur_v     — vertical 1-D Gaussian
//      (opt) hqdering_rg11 [+ hqdering_mean3x3 ...] + hqdering_contra — contra-sharpen (sharp)
//      (opt) hqdering_repair3x3 — clamp blurred to src 3×3 min/max (drrep)
//   5. hqdering_combine    — alpha-blend src ↔ blurred via ring mask
//                            (+ opt LimitFilter-style thr/darkthr/elast limit)
//
// Build-time defines (set via -D from rgy_filter_hqdering.cpp):
//   Type           : uchar (8-bit) or ushort (>8-bit)
//   bit_depth      : source bit depth
//   max_val        : (1 << bit_depth) - 1
//   dering_block_x / dering_block_y : work-group dims
//   DERING_KERNEL_RADIUS_MAX : compile-time upper bound on Gaussian radius
//                              (used to size the static loop bound; the
//                              actual radius is passed as a kernel arg
//                              and may be smaller)
//
// HBD safety:
//   * Pixel storage uses Type. Edge/ring-mask are Type, in [0, max_val].
//   * mthr is pre-scaled to working bit depth at the host.
//   * Gaussian arithmetic is float; final clamp to [0, max_val].
//
// =============================================================================

inline int hqd_readPixClamp(const __global uchar *plane,
                             int x, int y,
                             int pitch, int width, int height) {
    if (x < 0)        x = 0;
    if (x >= width)   x = width  - 1;
    if (y < 0)        y = 0;
    if (y >= height)  y = height - 1;
    return (int)(*(const __global Type *)(plane + y * pitch + x * sizeof(Type)));
}

// =============================================================================
// hqdering_edge — Sobel gradient magnitude + Levels-style threshold.
//
// Per-pixel:
//   Gx = -p[-1,-1] - 2 p[-1,0] - p[-1,+1]  +  p[+1,-1] + 2 p[+1,0] + p[+1,+1]
//   Gy = -p[-1,-1] - 2 p[ 0,-1] - p[+1,-1]  +  p[-1,+1] + 2 p[ 0,+1] + p[+1,+1]
//   G  = |Gx| + |Gy|     (L1 approximation; the threshold ramp normalises it)
//
// Levels ramp:
//   if max_val > mthr: edge = clamp((G - mthr) * max_val / (max_val - mthr), 0, max_val)
//   else             : edge = (G >= mthr) ? max_val : 0
//
// Same shape as finedehalo_prewitt; reproduced here so the dering filter
// has no cross-program dependency for edge detection.
__kernel void hqdering_edge(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height,
    int mthrHbd
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int p00 = hqd_readPixClamp(pSrc, x - 1, y - 1, srcPitch, width, height);
    const int p10 = hqd_readPixClamp(pSrc, x    , y - 1, srcPitch, width, height);
    const int p20 = hqd_readPixClamp(pSrc, x + 1, y - 1, srcPitch, width, height);
    const int p01 = hqd_readPixClamp(pSrc, x - 1, y    , srcPitch, width, height);
    const int p21 = hqd_readPixClamp(pSrc, x + 1, y    , srcPitch, width, height);
    const int p02 = hqd_readPixClamp(pSrc, x - 1, y + 1, srcPitch, width, height);
    const int p12 = hqd_readPixClamp(pSrc, x    , y + 1, srcPitch, width, height);
    const int p22 = hqd_readPixClamp(pSrc, x + 1, y + 1, srcPitch, width, height);

    const int gx = -p00 - 2 * p01 - p02  +  p20 + 2 * p21 + p22;
    const int gy = -p00 - 2 * p10 - p20  +  p02 + 2 * p12 + p22;
    const int g  = abs(gx) + abs(gy);

    int edge;
    if (max_val > mthrHbd) {
        long num = (long)(g - mthrHbd) * (long)max_val;
        int v = (int)(num / (long)(max_val - mthrHbd));
        if (v < 0)        v = 0;
        if (v > max_val)  v = max_val;
        edge = v;
    } else {
        edge = (g >= mthrHbd) ? max_val : 0;
    }

    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));
    dPix[0] = (Type)edge;
}

// =============================================================================
// Alternative edge operators (opt-in via --vpp-hqdering edge=...).
//
// All operators normalise their L1 gradient magnitude to the SAME domain as
// the default hqdering_edge above (peak ≈ 8 * max_val), so the user's `mthr`
// threshold keeps its meaning regardless of operator choice. The post-
// magnitude Levels ramp is identical and extracted into the HQD_LEVELS_RAMP
// macro below.
//
// Default operator is `edge=log` (5×5 Laplacian of Gaussian): it has the
// best ring-detection sensitivity in flat regions, which is what HQDering
// most cares about. `edge=sobel` resolves to the original hqdering_edge.
// =============================================================================

#define HQD_LEVELS_RAMP(g_in_8maxval_domain)                                 \
    int edge;                                                                \
    if (max_val > mthrHbd) {                                                 \
        long num = (long)((g_in_8maxval_domain) - mthrHbd) * (long)max_val;  \
        int v = (int)(num / (long)(max_val - mthrHbd));                      \
        if (v < 0)        v = 0;                                             \
        if (v > max_val)  v = max_val;                                       \
        edge = v;                                                            \
    } else {                                                                 \
        edge = ((g_in_8maxval_domain) >= mthrHbd) ? max_val : 0;             \
    }                                                                        \
    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type)); \
    dPix[0] = (Type)edge

#define HQD_LOAD_3X3                                                         \
    const int tl = hqd_readPixClamp(pSrc, x - 1, y - 1, srcPitch, width, height); \
    const int tc = hqd_readPixClamp(pSrc, x    , y - 1, srcPitch, width, height); \
    const int tr = hqd_readPixClamp(pSrc, x + 1, y - 1, srcPitch, width, height); \
    const int cl = hqd_readPixClamp(pSrc, x - 1, y    , srcPitch, width, height); \
    const int cc = hqd_readPixClamp(pSrc, x    , y    , srcPitch, width, height); \
    const int cr = hqd_readPixClamp(pSrc, x + 1, y    , srcPitch, width, height); \
    const int bl = hqd_readPixClamp(pSrc, x - 1, y + 1, srcPitch, width, height); \
    const int bc = hqd_readPixClamp(pSrc, x    , y + 1, srcPitch, width, height); \
    const int br = hqd_readPixClamp(pSrc, x + 1, y + 1, srcPitch, width, height)

// Prewitt (flat 8-tap). Raw peak = 6*max_val; scale (g * 4) / 3 → 8*max_val.
__kernel void hqdering_edge_prewitt(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height,
    int mthrHbd
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    HQD_LOAD_3X3;
    (void)cc;
    const int gx = (-tl + tr) + (-cl + cr) + (-bl + br);
    const int gy = (-tl - tc - tr) + (bl + bc + br);
    const int g  = ((abs(gx) + abs(gy)) * 4) / 3;
    HQD_LEVELS_RAMP(g);
}

// Scharr — [3,10,3]. Raw peak = 32*max_val; /4 → 8*max_val.
__kernel void hqdering_edge_scharr(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height,
    int mthrHbd
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    HQD_LOAD_3X3;
    (void)cc;
    const int gx = -3 * tl + 3 * tr - 10 * cl + 10 * cr - 3 * bl + 3 * br;
    const int gy = -3 * tl - 10 * tc - 3 * tr + 3 * bl + 10 * bc + 3 * br;
    const int g  = (abs(gx) + abs(gy)) / 4;
    HQD_LEVELS_RAMP(g);
}

// Kirsch — 8-direction max. Raw peak = 15*max_val; *8/15 → 8*max_val.
__kernel void hqdering_edge_kirsch(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height,
    int mthrHbd
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    HQD_LOAD_3X3;
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
    HQD_LEVELS_RAMP(g);
}

// Laplacian (4-connected). Raw peak = 4*max_val; *2 → 8*max_val.
__kernel void hqdering_edge_laplacian(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height,
    int mthrHbd
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    const int cc = hqd_readPixClamp(pSrc, x    , y    , srcPitch, width, height);
    const int tc = hqd_readPixClamp(pSrc, x    , y - 1, srcPitch, width, height);
    const int cl = hqd_readPixClamp(pSrc, x - 1, y    , srcPitch, width, height);
    const int cr = hqd_readPixClamp(pSrc, x + 1, y    , srcPitch, width, height);
    const int bc = hqd_readPixClamp(pSrc, x    , y + 1, srcPitch, width, height);
    const int g  = abs(4 * cc - tc - cl - cr - bc) * 2;
    HQD_LEVELS_RAMP(g);
}

// Laplacian of Gaussian (5x5, isotropic). Default operator.
//
// Kernel layout (sums to 0):
//    0  0 -1  0  0
//    0 -1 -2 -1  0
//   -1 -2 16 -2 -1
//    0 -1 -2 -1  0
//    0  0 -1  0  0
//
// Raw peak |response| = 16*max_val (centre saturated, neighbours zero — or
// vice versa). Scale by /2 to land in 8*max_val domain.
__kernel void hqdering_edge_log(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height,
    int mthrHbd
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    // 5x5 neighbourhood loads (only the non-zero-weight positions).
    const int n2c = hqd_readPixClamp(pSrc, x    , y - 2, srcPitch, width, height);
    const int n1l = hqd_readPixClamp(pSrc, x - 1, y - 1, srcPitch, width, height);
    const int n1c = hqd_readPixClamp(pSrc, x    , y - 1, srcPitch, width, height);
    const int n1r = hqd_readPixClamp(pSrc, x + 1, y - 1, srcPitch, width, height);
    const int c2l = hqd_readPixClamp(pSrc, x - 2, y    , srcPitch, width, height);
    const int c1l = hqd_readPixClamp(pSrc, x - 1, y    , srcPitch, width, height);
    const int ccc = hqd_readPixClamp(pSrc, x    , y    , srcPitch, width, height);
    const int c1r = hqd_readPixClamp(pSrc, x + 1, y    , srcPitch, width, height);
    const int c2r = hqd_readPixClamp(pSrc, x + 2, y    , srcPitch, width, height);
    const int s1l = hqd_readPixClamp(pSrc, x - 1, y + 1, srcPitch, width, height);
    const int s1c = hqd_readPixClamp(pSrc, x    , y + 1, srcPitch, width, height);
    const int s1r = hqd_readPixClamp(pSrc, x + 1, y + 1, srcPitch, width, height);
    const int s2c = hqd_readPixClamp(pSrc, x    , y + 2, srcPitch, width, height);

    const int log_resp = 16 * ccc
                       -      (n2c + c2l + c2r + s2c)
                       - 2 *  (n1c + c1l + c1r + s1c)
                       -      (n1l + n1r + s1l + s1r);
    const int g = abs(log_resp) / 2;
    HQD_LEVELS_RAMP(g);
}

// =============================================================================
// The mask/limit extensions below (inpand, 3x3 mean, min/max repair and the
// LimitFilter-style change limit in the combine kernel) follow the reference
// HQDering parameter set. Their building blocks are the same primitives that
// the --vpp-finedehalo rework (#777) introduced in rgy_filter_dehalo.cl /
// rgy_filter_finedehalo.cl (square min/max morphology, RemoveGrain-mode-20
// style 3x3 mean, min/max range clamp). They are implemented here as
// hqdering-local kernels on purpose, so that this patch does not touch the
// freshly reworked dehalo sources - if a shared header for these morphology
// primitives is preferred, they can be unified later.
// =============================================================================

//expand3x3の対 (min morph) - #777のsquare morph(min)と同じプリミティブ
__kernel void hqdering_inpand3x3(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int m = INT_MAX;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            m = min(m, hqd_readPixClamp(pSrc, x + dx, y + dy, srcPitch, width, height));
        }
    }

    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));
    dPix[0] = (Type)m;
}

//3x3平均 (RemoveGrain mode20相当) - #777のfdh_removegrain20_approxと同じプリミティブ
__kernel void hqdering_mean3x3(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += hqd_readPixClamp(pSrc, x + dx, y + dy, srcPitch, width, height);
        }
    }
    int v = (sum + 4) / 9;
    if (v < 0)        v = 0;
    if (v > max_val)  v = max_val;

    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));
    dPix[0] = (Type)v;
}

//RemoveGrain mode11相当: 3x3二項フィルタ(1,2,1;2,4,2;1,2,1)/16 - コントラシャープのぼかし段
__kernel void hqdering_rg11(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int w[3][3] = { { 1, 2, 1 }, { 2, 4, 2 }, { 1, 2, 1 } };
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += w[dy + 1][dx + 1] * hqd_readPixClamp(pSrc, x + dx, y + dy, srcPitch, width, height);
        }
    }
    int v = (sum + 8) >> 4;
    if (v < 0)        v = 0;
    if (v > max_val)  v = max_val;

    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));
    dPix[0] = (Type)v;
}

//コントラシャープの合成段 (リファレンスのcontra-sharpening):
//  sharpDiff = smoothed - method(=RG11[->RG20...])
//  ssDD = Repair(sharpDiff, src - smoothed, mode1) 相当 = 3x3近傍の(src-smoothed)のmin/maxへクランプ
//  |ssDD| > |sharpDiff| なら sharpDiff を採用 (小さい方の変化のみ戻す)
//  out = smoothed + ssDD
//ブラーで失われた線をsrcに実在した変化の範囲でのみ戻すため、リンギングを再生成しない
__kernel void hqdering_contra(
    const __global uchar *pSrc,      int srcPitch,
    const __global uchar *pSmoothed, int smoothedPitch,
    const __global uchar *pMethod,   int methodPitch,
    __global       uchar *pDst,      int dstPitch,
    int width, int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int sm = hqd_readPixClamp(pSmoothed, x, y, smoothedPitch, width, height);
    const int mt = hqd_readPixClamp(pMethod,   x, y, methodPitch,   width, height);
    const int sharpDiff = sm - mt;
    int mn = INT_MAX, mx = INT_MIN;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            const int d = hqd_readPixClamp(pSrc,      x + dx, y + dy, srcPitch,      width, height)
                        - hqd_readPixClamp(pSmoothed, x + dx, y + dy, smoothedPitch, width, height);
            mn = min(mn, d);
            mx = max(mx, d);
        }
    }
    int ssDD = sharpDiff;
    if (ssDD < mn) ssDD = mn;
    if (ssDD > mx) ssDD = mx;
    if (abs(ssDD) > abs(sharpDiff)) {
        ssDD = sharpDiff;
    }
    int v = sm + ssDD;
    if (v < 0)        v = 0;
    if (v > max_val)  v = max_val;

    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));
    dPix[0] = (Type)v;
}

//repair mode1相当: ブラー結果をsrcの3x3 min/maxへクランプ - #777のsquare rangeと同じプリミティブ
__kernel void hqdering_repair3x3(
    const __global uchar *pSrc,     int srcPitch,
    const __global uchar *pBlurred, int blurredPitch,
    __global       uchar *pDst,     int dstPitch,
    int width, int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int mn = INT_MAX, mx = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            const int v = hqd_readPixClamp(pSrc, x + dx, y + dy, srcPitch, width, height);
            mn = min(mn, v);
            mx = max(mx, v);
        }
    }
    int b = hqd_readPixClamp(pBlurred, x, y, blurredPitch, width, height);
    if (b < mn) b = mn;
    if (b > mx) b = mx;

    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));
    dPix[0] = (Type)b;
}

// =============================================================================
// hqdering_blur_h / hqdering_blur_v — 1-D Gaussian, separable.
//
// `radius` and `sigma` are passed as kernel args. We compute weights
// inline per work-item: `w[i] = exp(-i*i / (2 sigma^2))` for
// i ∈ [-radius, +radius], accumulating value*w and w in one pass and
// dividing at the end.
//
// On Arc A770 the per-work-item exp() cost (≤ 21 calls/pass at
// sigma=5) is dwarfed by the global memory traffic, so this is fine
// without a `__constant float *` weights array. The static loop bound
// `DERING_KERNEL_RADIUS_MAX` lets the compiler unroll if it chooses;
// the runtime `radius` short-circuits the actual work.
__kernel void hqdering_blur_h(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height,
    int radius, float sigma
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const float invTwoSigmaSq = 1.0f / (2.0f * sigma * sigma);

    float vSum = 0.0f;
    float wSum = 0.0f;
    for (int i = -DERING_KERNEL_RADIUS_MAX; i <= DERING_KERNEL_RADIUS_MAX; i++) {
        if (i < -radius || i > radius) continue;
        const float wi = native_exp(-(float)(i * i) * invTwoSigmaSq);
        const int   pv = hqd_readPixClamp(pSrc, x + i, y, srcPitch, width, height);
        vSum += (float)pv * wi;
        wSum += wi;
    }

    int v = convert_int_rte(vSum / wSum);
    if (v < 0)        v = 0;
    if (v > max_val)  v = max_val;

    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));
    dPix[0] = (Type)v;
}

__kernel void hqdering_blur_v(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height,
    int radius, float sigma
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const float invTwoSigmaSq = 1.0f / (2.0f * sigma * sigma);

    float vSum = 0.0f;
    float wSum = 0.0f;
    for (int i = -DERING_KERNEL_RADIUS_MAX; i <= DERING_KERNEL_RADIUS_MAX; i++) {
        if (i < -radius || i > radius) continue;
        const float wi = native_exp(-(float)(i * i) * invTwoSigmaSq);
        const int   pv = hqd_readPixClamp(pSrc, x, y + i, srcPitch, width, height);
        vSum += (float)pv * wi;
        wSum += wi;
    }

    int v = convert_int_rte(vSum / wSum);
    if (v < 0)        v = 0;
    if (v > max_val)  v = max_val;

    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));
    dPix[0] = (Type)v;
}

// =============================================================================
// hqdering_combine — fused mask + alpha-blend (steps 4+5).
//
// Per-pixel:
//   if (protect): mask_used = max(ring_mask - edge_mask, 0)
//   else        : mask_used = ring_mask
//   m   = mask_used / max_val                  (normalised, 0..1)
//   out = src + (blurred - src) * m            (then clamp)
//
// `protect=1` subtracts the original (pre-dilation) edge mask from the
// dilated ring mask, so the blend skips the actual edge pixels (text
// strokes, fine lines stay sharp) and only smooths the annular ringing
// zone around them.
//
// `showmask != 0` writes the effective mask (after the protect /
// core-mask subtractions, if any) instead of the merged output — useful
// for confirming the mask toggles visually.
//
// `useCoreMask != 0` additionally subtracts the inpanded edge core
// (minp) from the ring mask. OpenCL cannot rely on a NULL cl_mem the
// way CUDA checks a null pointer, so the host passes a dummy buffer and
// this explicit flag when minp=0 — pCoreMask is then never read.
//
// `thrHbd > 0` enables a LimitFilter-style per-pixel change limit
// applied after the blend (thr/darkthr/elast). thrHbd=0 (the default)
// leaves the blend result untouched — bit-identical to the previous
// kernel.
__kernel void hqdering_combine(
    const __global uchar *pSrc,      int srcPitch,
    const __global uchar *pBlurred,  int blPitch,
    const __global uchar *pMask,     int mskPitch,
    const __global uchar *pEdgeMask, int edgePitch,
    const __global uchar *pCoreMask, int corePitch,
    __global       uchar *pDst,      int dstPitch,
    int width, int height,
    int showmask, int protect, int useCoreMask,
    int thrHbd, int darkthrHbd, float elast
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int s  = (int)(*(const __global Type *)(pSrc     + y * srcPitch + x * sizeof(Type)));
    const int b  = (int)(*(const __global Type *)(pBlurred + y * blPitch  + x * sizeof(Type)));
    const int mk = (int)(*(const __global Type *)(pMask    + y * mskPitch + x * sizeof(Type)));

    int effectiveMask = mk;
    if (protect != 0) {
        const int em = (int)(*(const __global Type *)(pEdgeMask + y * edgePitch + x * sizeof(Type)));
        int diff = mk - em;
        if (diff < 0) diff = 0;
        effectiveMask = diff;
    }
    if (useCoreMask != 0) {
        //minp: inpandしたエッジ芯をリングマスクから除外し、線そのものは処理しない (リファレンスのminp)
        const int cm = (int)(*(const __global Type *)(pCoreMask + y * corePitch + x * sizeof(Type)));
        int diff = effectiveMask - cm;
        if (diff < 0) diff = 0;
        effectiveMask = diff;
    }

    int outVal;
    if (showmask != 0) {
        outVal = effectiveMask;
    } else {
        const float m = (float)effectiveMask / (float)max_val;
        const float blended = (float)s + ((float)b - (float)s) * m;
        int v = convert_int_rte(blended);
        if (v < 0)        v = 0;
        if (v > max_val)  v = max_val;
        outVal = v;
        if (thrHbd > 0) {
            //LimitFilter形式の変化量制限 (リファレンスのthr/elast/darkthr):
            //|変化量|<=lim は全適用、lim*elast以上は不適用、間は線形減衰
            const int diff = outVal - s;
            const int lim = (diff < 0 && darkthrHbd >= 0) ? darkthrHbd : thrHbd;
            const int adiff = abs(diff);
            float ramp = 1.0f;
            if (adiff > lim) {
                const float elastRange = (float)lim * elast;
                ramp = (elast > 1.0f && (float)adiff < elastRange)
                    ? (elastRange - (float)adiff) / ((float)lim * (elast - 1.0f))
                    : 0.0f;
            }
            int vLim = s + (int)((float)diff * ramp + ((diff >= 0) ? 0.5f : -0.5f));
            if (vLim < 0)        vLim = 0;
            if (vLim > max_val)  vLim = max_val;
            outVal = vLim;
        }
    }

    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));
    dPix[0] = (Type)outVal;
}
