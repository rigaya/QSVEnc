// DeHalo_alpha — halo-removal OpenCL kernels.
// From-scratch implementation for QSVEncC.
//
// Pipeline (luma plane only; chroma passes through unchanged):
//   1. Spline36 resize-up (host-side, RGYFilterResize sub-filter)
//   2. dehalo_expand   — elliptic 3D local maximum over (rx, ry)
//   3. dehalo_inpand   — elliptic 3D local minimum over (rx, ry)
//   4. dehalo_mask     — sensitivity-ramp mask from (src, expand, inpand)
//   5. dehalo_apply    — alpha-blend correction using darkstr/brightstr
//   6. Spline36 resize-down (host-side)
//
// The resize stages re-use QSVEnc's existing RGYFilterResize sub-filter
// at host level — no spline36 kernels are duplicated here.
//
// Build-time defines (set via -D from rgy_filter_dehalo.cpp):
//   Type             : uchar (8-bit) or ushort (>8-bit)
//   bit_depth        : source bit depth (8, 10, 12, 14, 16)
//   max_val          : (1 << bit_depth) - 1
//   dehalo_block_x   : work-group X dimension
//   dehalo_block_y   : work-group Y dimension
//
// HBD safety:
//   * Pixel storage uses Type. Intermediate arithmetic uses int / float.
//   * lowsens / highsens come in pre-scaled to the working bit depth at the
//     host (host computes `lo = lowsens * 255 / 100` then scales by max_val/255).
//   * rx / ry are float; passed unchanged to the kernel (geometric, no
//     bit-depth scaling).
//   * Mask is stored as Type in [0, max_val].
//   * Apply uses float arithmetic with a final clamp to [0, max_val].
//
// =============================================================================

// Edge-clamped read from a 2-D plane stored at `pitch` bytes per row.
inline int dh_readPixClamp(const __global uchar *plane,
                            int x, int y,
                            int pitch, int width, int height) {
    if (x < 0)        x = 0;
    if (x >= width)   x = width  - 1;
    if (y < 0)        y = 0;
    if (y >= height)  y = height - 1;
    return (int)(*(const __global Type *)(plane + y * pitch + x * sizeof(Type)));
}

// =============================================================================
// dehalo_expand — elliptic local maximum.
//
// For each output pixel, iterate (dx, dy) over the bounding rectangle
// [-irx, irx] x [-iry, iry] and accept a sample only when it lies inside
// the ellipse (dx/rx)^2 + (dy/ry)^2 <= 1.0. The maximum across accepted
// samples is the output value. Edge pixels read with clamp-to-edge.
//
// SLM tile-load: WG 32x8 = 256 threads, cooperatively loads the
// (32 + 2*irx) x (8 + 2*iry) source window into __local int[] once per
// WG, then the per-pixel inner loop reads from local memory only. Tile
// is sized for max radius (DEHALO_RMAX = 10) so __local is a compile-
// time constant; cooperative load touches only the actually-needed
// (32 + 2*irx) x (8 + 2*iry) region.
//
// At rx=ry=2 (default): tile 36x12 = 432 ints (1728 B/WG); SLM gain is
// noise but does not regress. At rx=ry=10 (max): tile 52x28 = 1456 ints
// (5824 B/WG); -20% kernel time at 4K on Arc A770. Both well under the
// 64 KB per-WG SLM ceiling on Xe-HPG so occupancy is unconstrained.
#define DEHALO_RMAX     10
#define DEHALO_TILE_W_MAX (32 + 2 * DEHALO_RMAX)
#define DEHALO_TILE_H_MAX (8  + 2 * DEHALO_RMAX)

__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void dehalo_expand(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height,
    float rx, float ry
) {
    const int irx = (int)ceil(rx);
    const int iry = (int)ceil(ry);
    const int tile_w = 32 + 2 * irx;
    const int tile_h = 8  + 2 * iry;
    const float invRx2 = 1.0f / (rx * rx);
    const float invRy2 = 1.0f / (ry * ry);

    __local int tile[DEHALO_TILE_H_MAX * DEHALO_TILE_W_MAX];

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int gx0 = get_group_id(0) * 32;
    const int gy0 = get_group_id(1) * 8;
    const int tid = ly * 32 + lx;
    const int wg_size = 32 * 8;
    const int tile_total = tile_w * tile_h;

    for (int t = tid; t < tile_total; t += wg_size) {
        const int tx = t % tile_w;
        const int ty = t / tile_w;
        const int sx = clamp(gx0 + tx - irx, 0, width  - 1);
        const int sy = clamp(gy0 + ty - iry, 0, height - 1);
        tile[t] = (int)(*(const __global Type *)(pSrc + sy * srcPitch + sx * sizeof(Type)));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int x = gx0 + lx;
    const int y = gy0 + ly;
    if (x >= width || y >= height) return;

    const int tcx = lx + irx;
    const int tcy = ly + iry;
    int m = tile[tcy * tile_w + tcx];
    for (int dy = -iry; dy <= iry; dy++) {
        const float dyF = (float)dy;
        const float yTerm = dyF * dyF * invRy2;
        if (yTerm > 1.0f) continue;
        const float xLimitSq = 1.0f - yTerm;
        for (int dx = -irx; dx <= irx; dx++) {
            const float dxF = (float)dx;
            if (dxF * dxF * invRx2 > xLimitSq) continue;
            const int v = tile[(tcy + dy) * tile_w + (tcx + dx)];
            if (v > m) m = v;
        }
    }

    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));
    dPix[0] = (Type)m;
}

// =============================================================================
// dehalo_inpand — elliptic local minimum. Same loop body as dehalo_expand
// but using fmin instead of fmax. Same SLM tile pattern.
__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void dehalo_inpand(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height,
    float rx, float ry
) {
    const int irx = (int)ceil(rx);
    const int iry = (int)ceil(ry);
    const int tile_w = 32 + 2 * irx;
    const int tile_h = 8  + 2 * iry;
    const float invRx2 = 1.0f / (rx * rx);
    const float invRy2 = 1.0f / (ry * ry);

    __local int tile[DEHALO_TILE_H_MAX * DEHALO_TILE_W_MAX];

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int gx0 = get_group_id(0) * 32;
    const int gy0 = get_group_id(1) * 8;
    const int tid = ly * 32 + lx;
    const int wg_size = 32 * 8;
    const int tile_total = tile_w * tile_h;

    for (int t = tid; t < tile_total; t += wg_size) {
        const int tx = t % tile_w;
        const int ty = t / tile_w;
        const int sx = clamp(gx0 + tx - irx, 0, width  - 1);
        const int sy = clamp(gy0 + ty - iry, 0, height - 1);
        tile[t] = (int)(*(const __global Type *)(pSrc + sy * srcPitch + sx * sizeof(Type)));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int x = gx0 + lx;
    const int y = gy0 + ly;
    if (x >= width || y >= height) return;

    const int tcx = lx + irx;
    const int tcy = ly + iry;
    int m = tile[tcy * tile_w + tcx];
    for (int dy = -iry; dy <= iry; dy++) {
        const float dyF = (float)dy;
        const float yTerm = dyF * dyF * invRy2;
        if (yTerm > 1.0f) continue;
        const float xLimitSq = 1.0f - yTerm;
        for (int dx = -irx; dx <= irx; dx++) {
            const float dxF = (float)dx;
            if (dxF * dxF * invRx2 > xLimitSq) continue;
            const int v = tile[(tcy + dy) * tile_w + (tcx + dx)];
            if (v < m) m = v;
        }
    }

    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));
    dPix[0] = (Type)m;
}

// =============================================================================
// dehalo_mask — sensitivity-ramp halo-detection mask.
//
// Per-pixel:
//   range    = expand - inpand
//   abs_diff = (range == 0) ? 0
//            : clamp((src - inpand) * max_val / range, 0, max_val)
//   if (hi > lo): mask = clamp((abs_diff - lo) * max_val / (hi - lo), 0, max_val)
//   else        : mask = (abs_diff >= lo) ? max_val : 0     (step function)
//
// `lo` and `hi` are pre-scaled at the host (lowsens/highsens × max_val/100).
// The lo == hi edge case avoids divide-by-zero by falling through to a
// hard step.
__kernel void dehalo_mask(
    const __global uchar *pSrc,    int srcPitch,
    const __global uchar *pExpand, int expPitch,
    const __global uchar *pInpand, int inpPitch,
    __global       uchar *pMask,   int maskPitch,
    int width, int height,
    int loScaled, int hiScaled
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int s = (int)(*(const __global Type *)(pSrc    + y * srcPitch  + x * sizeof(Type)));
    const int e = (int)(*(const __global Type *)(pExpand + y * expPitch  + x * sizeof(Type)));
    const int i = (int)(*(const __global Type *)(pInpand + y * inpPitch  + x * sizeof(Type)));

    const int range = e - i;

    int abs_diff;
    if (range <= 0) {
        abs_diff = 0;
    } else {
        // (s - i) * max_val / range, clamped.
        long num = (long)(s - i) * (long)max_val;
        int v   = (int)(num / (long)range);
        if (v < 0)        v = 0;
        if (v > max_val)  v = max_val;
        abs_diff = v;
    }

    int mask;
    if (hiScaled > loScaled) {
        long num = (long)(abs_diff - loScaled) * (long)max_val;
        int v   = (int)(num / (long)(hiScaled - loScaled));
        if (v < 0)        v = 0;
        if (v > max_val)  v = max_val;
        mask = v;
    } else {
        // Degenerate case (lo == hi): hard step at lo.
        mask = (abs_diff >= loScaled) ? max_val : 0;
    }

    __global Type *mPix = (__global Type *)(pMask + y * maskPitch + x * sizeof(Type));
    mPix[0] = (Type)mask;
}

// =============================================================================
// dehalo_apply — alpha-blend correction.
//
// Per-pixel:
//   m  = mask / max_val           (normalised mask, 0..1)
//   correction = src
//              - m * darkstr   * (src - inpand)
//              + m * brightstr * (expand - src)
//   out = clamp(correction, 0, max_val)
//
// With the default brightstr = 0.0 the second blend term contributes
// nothing; we always read expand because the cost is negligible (one
// extra global load per work-item) and the alternative — compiling two
// kernel variants — adds maintenance complexity.
__kernel void dehalo_apply(
    const __global uchar *pSrc,    int srcPitch,
    const __global uchar *pExpand, int expPitch,
    const __global uchar *pInpand, int inpPitch,
    const __global uchar *pMask,   int maskPitch,
    __global       uchar *pDst,    int dstPitch,
    int width, int height,
    float darkstr, float brightstr
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const float s = (float)(int)(*(const __global Type *)(pSrc    + y * srcPitch  + x * sizeof(Type)));
    const float e = (float)(int)(*(const __global Type *)(pExpand + y * expPitch  + x * sizeof(Type)));
    const float i = (float)(int)(*(const __global Type *)(pInpand + y * inpPitch  + x * sizeof(Type)));
    const float m = (float)(int)(*(const __global Type *)(pMask   + y * maskPitch + x * sizeof(Type)));

    const float mn  = m / (float)max_val;
    const float darken_term  = mn * darkstr   * (s - i);
    const float brighten_term = mn * brightstr * (e - s);

    float r = s - darken_term + brighten_term;
    if (r < 0.0f)              r = 0.0f;
    if (r > (float)max_val)    r = (float)max_val;

    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));
    dPix[0] = (Type)convert_int_rte(r);
}
