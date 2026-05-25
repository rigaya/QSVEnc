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
// For each output pixel (x, y), iterate (dx, dy) over the bounding rectangle
// [-irx, irx] × [-iry, iry] and accept a sample only when it lies inside the
// ellipse `(dx/rx)^2 + (dy/ry)^2 <= 1.0`. The maximum across accepted samples
// is the output value. Edge pixels are read with clamp-to-edge.
//
// rx / ry are floats. The ellipse test uses float division so non-integer
// radii (e.g. rx=2.5) are handled correctly.
__kernel void dehalo_expand(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height,
    float rx, float ry
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int irx = (int)ceil(rx);
    const int iry = (int)ceil(ry);
    const float invRx2 = 1.0f / (rx * rx);
    const float invRy2 = 1.0f / (ry * ry);

    int m = (int)(*(const __global Type *)(pSrc + y * srcPitch + x * sizeof(Type)));
    for (int dy = -iry; dy <= iry; dy++) {
        const float dyF = (float)dy;
        const float yTerm = dyF * dyF * invRy2;
        if (yTerm > 1.0f) continue;
        const float xLimitSq = 1.0f - yTerm;     // dx^2 * invRx2 must be <= this
        for (int dx = -irx; dx <= irx; dx++) {
            const float dxF = (float)dx;
            if (dxF * dxF * invRx2 > xLimitSq) continue;
            const int v = dh_readPixClamp(pSrc, x + dx, y + dy, srcPitch, width, height);
            if (v > m) m = v;
        }
    }

    __global Type *dPix = (__global Type *)(pDst + y * dstPitch + x * sizeof(Type));
    dPix[0] = (Type)m;
}

// =============================================================================
// dehalo_inpand — elliptic local minimum. Same loop body as dehalo_expand
// but using fmin instead of fmax. Kept as a separate kernel for clarity
// (the two are dispatched independently and the OpenCL compiler inlines
// each cleanly).
__kernel void dehalo_inpand(
    const __global uchar *pSrc, int srcPitch,
    __global       uchar *pDst, int dstPitch,
    int width, int height,
    float rx, float ry
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int irx = (int)ceil(rx);
    const int iry = (int)ceil(ry);
    const float invRx2 = 1.0f / (rx * rx);
    const float invRy2 = 1.0f / (ry * ry);

    int m = (int)(*(const __global Type *)(pSrc + y * srcPitch + x * sizeof(Type)));
    for (int dy = -iry; dy <= iry; dy++) {
        const float dyF = (float)dy;
        const float yTerm = dyF * dyF * invRy2;
        if (yTerm > 1.0f) continue;
        const float xLimitSq = 1.0f - yTerm;
        for (int dx = -irx; dx <= irx; dx++) {
            const float dxF = (float)dx;
            if (dxF * dxF * invRx2 > xLimitSq) continue;
            const int v = dh_readPixClamp(pSrc, x + dx, y + dy, srcPitch, width, height);
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
