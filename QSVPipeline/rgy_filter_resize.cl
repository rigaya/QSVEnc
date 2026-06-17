
// Type
// bit_depth
// radius
// block_x
// block_y
// algo
// WEIGHT_BILINEAR
// WEIGHT_BICUBIC
// WEIGHT_SPLINE
// WEIGHT_LANCZOS
// WEIGHT_GAUSS
// gauss_p
// shared_weightXdim
// shared_weightYdim
// USE_LOCAL

#ifndef MIN3
#define MIN3(a,b,c) (min((a), min((b), (c))))
#endif
#ifndef MAX3
#define MAX3(a,b,c) (max((a), max((b), (c))))
#endif

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

__kernel void kernel_resize_texture_bilinear(
    __global uchar *restrict pDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
    __read_only image2d_t src,
    const float ratioInvX, const float ratioInvY) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    if (ix < dstWidth && iy < dstHeight) {
        const float x = (float)ix + 0.5f;
        const float y = (float)iy + 0.5f;

        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(read_imagef(src, sampler, (int2)(x * ratioInvX, y * ratioInvY)).x * (float)((1<<bit_depth)-1));
    }
}

float sinc(float x) {
    const float pi = (float)3.14159265358979323846f;
    const float pi_x = pi * x;
    return native_sin(pi_x) * native_recip(pi_x);
}

float factor_lanczos(const float x) {
    if (fabs(x) >= (float)radius) return 0.0f;
    if (x == 0.0f) return 1.0f;
    return sinc(x) * sinc(x * (1.0f / radius));
}

float factor_bilinear(const float x) {
    if (fabs(x) >= (float)radius) return 0.0f;
    return 1.0f - x * (1.0f / radius);
}

float factor_bicubic(float x, float B, float C) {
    x = fabs(x);
    if (x >= (float)radius) return 0.0f;
    const float x2 = x*x;
    const float x3 = x2*x;
    if (x <= 1.0f) {
        return ( 2.0f -  1.5f * B - 1.0f * C) * x3 +
               (-3.0f +  2.0f * B + 1.0f * C) * x2 +
               ( 1.0f -  (2.0f/6.0f) * B);
    } else {
        return (-(1.0f/6.0f) * B - 1.0f * C) * x3 +
               (        1.0f * B + 5.0f * C) * x2 +
               (       -2.0f * B - 8.0f * C) * x  +
               ( (8.0f/6.0f) * B + 4.0f * C);
    }
}

float factor_gauss(float x) {
    x = fabs(x);
    if (x > (float)radius) return 0.0f;
    return native_exp2(-(gauss_p * 0.1f) * x * x);
}

#if USE_LOCAL
#define SPLINE_FACTOR_MEM_TYPE __local
#else
#define SPLINE_FACTOR_MEM_TYPE __global
#endif

float factor_spline(const float x_raw, SPLINE_FACTOR_MEM_TYPE const float4 *restrict psFactor) {
    const float x = fabs(x_raw);
    if (x >= (float)radius) return 0.0f;

    const float4 weight = psFactor[min((int)x, radius - 1)];
    //重みを計算
    float w = weight.w;
    w += x * weight.z;
    const float x2 = x * x;
    w += x2 * weight.y;
    w += x2 * x * weight.x;
    return w;
}

float calc_weight(
    const int targetPos, const float srcPos,
    const float ratioClamped, SPLINE_FACTOR_MEM_TYPE const float4 *psCopyFactor) {
    const float delta = (algo == WEIGHT_GAUSS)
        ? ((float)targetPos - (srcPos - 0.5f)) * ratioClamped
        : (((float)targetPos + 0.5f) - srcPos) * ratioClamped;
    float weight = 0.0f;
    switch (algo) {
    case WEIGHT_LANCZOS:  weight = factor_lanczos(delta); break;
    case WEIGHT_SPLINE:   weight = factor_spline(delta, psCopyFactor); break;
    case WEIGHT_BICUBIC:  weight = factor_bicubic(delta, 0.0f, 0.6f); break;
    case WEIGHT_BILINEAR: weight = factor_bilinear(delta); break;
    case WEIGHT_GAUSS:    weight = factor_gauss(delta); break;
    default:
        break;
    }
    return weight;
}

#if USE_LOCAL
void calc_weight_to_local(
    __local float *pWeight, const float srcPos, const int srcFirst, const int srcEnd,
    const float ratioClamped, __local const float4 *psCopyFactor) {
    __local float *pW = pWeight;
    for (int i = srcFirst; i <= srcEnd; i++, pW++) {
        pW[0] = calc_weight(i, srcPos, ratioClamped, psCopyFactor);
    }
}
#endif

// =========================================================================
// FSR 1.0 path (added as algo=fsr1)
//
//   kernel_easu : edge-adaptive spatial upsample applied per plane.
//                 The reference algorithm computes a per-pixel pseudo-luma
//                 from RGB to drive edge detection; on YUV planes each
//                 plane is already scalar, so we substitute the plane
//                 value directly.
//   kernel_rcas : non-scaling sharpen with FSR_RCAS_LIMIT cap and per-pixel
//                 hitMin/hitMax saturation prevention.
//
// =========================================================================

#define FSR_RCAS_LIMIT (0.25f - (1.0f / 16.0f))

// Read a plane value as a normalised float [0, 1] with edge clamping.
// Used by EASU on the source plane and (FP32 path only) by RCAS on the
// integer-typed EASU intermediate.
static inline float fsr_load_norm(const __global uchar *pSrc, int srcPitch, int x, int y, int w, int h) {
    x = clamp(x, 0, w - 1);
    y = clamp(y, 0, h - 1);
    Type val = *(const __global Type *)(pSrc + y * srcPitch + x * sizeof(Type));
    return (float)val * (1.0f / (float)((1 << bit_depth) - 1));
}

// Read the EASU-to-RCAS intermediate at (x, y) as a normalised float
// [0, 1] with edge clamping. FSR1_FP16_SCRATCH selects between FP16
// storage (vload_half from a half-pitch row) and the legacy
// integer-pixel path (fsr_load_norm against Type). srcPitch is in
// BYTES in both modes; FP16 mode divides by 2 internally to recover
// the half-element offset.
//
// On the FP16 path the round-trip EASU(float) -> quantise(Type) ->
// RCAS dequantise(float) is replaced with EASU(float) -> vstore_half
// -> vload_half(float), eliminating the integer-pixel precision loss
// without changing bandwidth at HBD (the only mode the host enables
// FP16 in -- see m_fp16Easu init guard).
static inline float fsr_load_mid(const __global uchar *pSrc, int srcPitch, int x, int y, int w, int h) {
#if defined(FSR1_FP16_SCRATCH) && FSR1_FP16_SCRATCH
    x = clamp(x, 0, w - 1);
    y = clamp(y, 0, h - 1);
    const int half_pitch_elems = srcPitch >> 1;
    return vload_half(y * half_pitch_elems + x, (const __global half *)pSrc);
#else
    return fsr_load_norm(pSrc, srcPitch, x, y, w, h);
#endif
}

// FSR EASU's per-direction accumulator. Operates on the 5-tap "+" pattern
//     a
//   b c d
//     e
// `pp` is the fractional offset of the sample point within the inner 2x2;
// the (biS, biT, biU, biV) flags select which of the four bilinear corners
// this call corresponds to.
static inline void fsr_easu_set(
    float *dirX, float *dirY, float *lenAcc,
    float2 pp,
    int biS, int biT, int biU, int biV,
    float lA, float lB, float lC, float lD, float lE) {
    float w = 0.0f;
    if (biS) w = (1.0f - pp.x) * (1.0f - pp.y);
    if (biT) w =          pp.x * (1.0f - pp.y);
    if (biU) w = (1.0f - pp.x) *          pp.y;
    if (biV) w =          pp.x *          pp.y;
    // X axis
    float dc = lD - lC;
    float cb = lC - lB;
    float lenX_inv = fmax(fabs(dc), fabs(cb));
    float rcpX = (lenX_inv > 0.0f) ? native_recip(lenX_inv) : 0.0f;
    float dirXv = lD - lB;
    *dirX += dirXv * w;
    float lenX = clamp(fabs(dirXv) * rcpX, 0.0f, 1.0f);
    lenX *= lenX;
    *lenAcc += lenX * w;
    // Y axis
    float ec = lE - lC;
    float ca = lC - lA;
    float lenY_inv = fmax(fabs(ec), fabs(ca));
    float rcpY = (lenY_inv > 0.0f) ? native_recip(lenY_inv) : 0.0f;
    float dirYv = lE - lA;
    *dirY += dirYv * w;
    float lenY = clamp(fabs(dirYv) * rcpY, 0.0f, 1.0f);
    lenY *= lenY;
    *lenAcc += lenY * w;
}

// Apply one Lanczos-2-approximation tap with rotation by `dir`, anisotropy
// `len2`, negative-lobe strength `lob`, and squared-distance cutoff `clp`.
static inline void fsr_easu_tap(
    float *aC, float *aW,
    float2 off, float2 dir, float2 len2,
    float lob, float clp, float c) {
    float vx = off.x * ( dir.x) + off.y * dir.y;
    float vy = off.x * (-dir.y) + off.y * dir.x;
    vx *= len2.x; vy *= len2.y;
    float d2 = vx * vx + vy * vy;
    d2 = fmin(d2, clp);
    // Approximation of lanczos2: (25/16 * (2/5*d2 - 1)^2 - 9/16) * (lob*d2 - 1)^2
    float wB = 0.4f * d2 - 1.0f;
    float wA = lob   * d2 - 1.0f;
    wB *= wB;
    wA *= wA;
    wB = (25.0f / 16.0f) * wB - (25.0f / 16.0f - 1.0f);
    float w = wB * wA;
    *aC += c * w;
    *aW += w;
}

__kernel void kernel_easu(
    __global uchar *restrict pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    __global const uchar *pSrc, const int srcPitch, const int srcWidth, const int srcHeight,
    const float ratioInvX, const float ratioInvY,
    const float offsetX, const float offsetY) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < dstWidth && iy < dstHeight) {
        // Source-space position. offset = 0.5 * ratioInv - 0.5 pre-computed host-side.
        const float ppx = (float)ix * ratioInvX + offsetX;
        const float ppy = (float)iy * ratioInvY + offsetY;
        const int fpx = (int)floor(ppx);
        const int fpy = (int)floor(ppy);
        const float2 pp = (float2)(ppx - (float)fpx, ppy - (float)fpy);

        // 12-tap neighbourhood relative to F = (fpx, fpy):
        //         b c
        //       e f g h
        //       i j k l
        //         n o
        const float b = fsr_load_norm(pSrc, srcPitch, fpx + 0, fpy - 1, srcWidth, srcHeight);
        const float c = fsr_load_norm(pSrc, srcPitch, fpx + 1, fpy - 1, srcWidth, srcHeight);
        const float e = fsr_load_norm(pSrc, srcPitch, fpx - 1, fpy + 0, srcWidth, srcHeight);
        const float f = fsr_load_norm(pSrc, srcPitch, fpx + 0, fpy + 0, srcWidth, srcHeight);
        const float g = fsr_load_norm(pSrc, srcPitch, fpx + 1, fpy + 0, srcWidth, srcHeight);
        const float hh= fsr_load_norm(pSrc, srcPitch, fpx + 2, fpy + 0, srcWidth, srcHeight);
        const float i = fsr_load_norm(pSrc, srcPitch, fpx - 1, fpy + 1, srcWidth, srcHeight);
        const float j = fsr_load_norm(pSrc, srcPitch, fpx + 0, fpy + 1, srcWidth, srcHeight);
        const float k = fsr_load_norm(pSrc, srcPitch, fpx + 1, fpy + 1, srcWidth, srcHeight);
        const float l = fsr_load_norm(pSrc, srcPitch, fpx + 2, fpy + 1, srcWidth, srcHeight);
        const float n = fsr_load_norm(pSrc, srcPitch, fpx + 0, fpy + 2, srcWidth, srcHeight);
        const float o = fsr_load_norm(pSrc, srcPitch, fpx + 1, fpy + 2, srcWidth, srcHeight);

        // Per-plane scalar: each tap value is itself the "luma".
        float dirX = 0.0f, dirY = 0.0f, lenAcc = 0.0f;
        fsr_easu_set(&dirX, &dirY, &lenAcc, pp, 1, 0, 0, 0, b, e, f, g, j);
        fsr_easu_set(&dirX, &dirY, &lenAcc, pp, 0, 1, 0, 0, c, f, g, hh, k);
        fsr_easu_set(&dirX, &dirY, &lenAcc, pp, 0, 0, 1, 0, f, i, j, k, n);
        fsr_easu_set(&dirX, &dirY, &lenAcc, pp, 0, 0, 0, 1, g, j, k, l, o);

        // Normalise direction; collapse near-zero to (1, 0) to avoid NaN.
        const float dir2x = dirX * dirX;
        const float dir2y = dirY * dirY;
        const float dirR = dir2x + dir2y;
        const int zro = (dirR < (1.0f / 32768.0f));
        const float invR = zro ? 1.0f : native_rsqrt(dirR);
        const float ndx  = zro ? 1.0f : (dirX * invR);
        const float ndy  = zro ? 0.0f : (dirY * invR);

        // Shape len: was in [0, 2]; rescale to [0, 1] and square.
        float len = lenAcc * 0.5f;
        len *= len;
        // Stretch / lob / clp parameters from the reference.
        const float stretch_num = ndx * ndx + ndy * ndy;
        const float stretch_den = fmax(fabs(ndx), fabs(ndy));
        const float stretch = stretch_num * ((stretch_den > 0.0f) ? native_recip(stretch_den) : 0.0f);
        const float2 len2 = (float2)(1.0f + (stretch - 1.0f) * len, 1.0f - 0.5f * len);
        const float lob = 0.5f + ((1.0f / 4.0f) - 0.04f - 0.5f) * len;
        const float clp = (lob > 0.0f) ? native_recip(lob) : 0.0f;

        // Inner-2x2 min/max for dering.
        const float mn4 = fmin(fmin(fmin(f, g), j), k);
        const float mx4 = fmax(fmax(fmax(f, g), j), k);

        // 12-tap weighted accumulation.
        const float2 dir = (float2)(ndx, ndy);
        float aC = 0.0f, aW = 0.0f;
        fsr_easu_tap(&aC, &aW, (float2)( 0.0f - pp.x, -1.0f - pp.y), dir, len2, lob, clp, b);
        fsr_easu_tap(&aC, &aW, (float2)( 1.0f - pp.x, -1.0f - pp.y), dir, len2, lob, clp, c);
        fsr_easu_tap(&aC, &aW, (float2)(-1.0f - pp.x,  1.0f - pp.y), dir, len2, lob, clp, i);
        fsr_easu_tap(&aC, &aW, (float2)( 0.0f - pp.x,  1.0f - pp.y), dir, len2, lob, clp, j);
        fsr_easu_tap(&aC, &aW, (float2)( 0.0f - pp.x,  0.0f - pp.y), dir, len2, lob, clp, f);
        fsr_easu_tap(&aC, &aW, (float2)(-1.0f - pp.x,  0.0f - pp.y), dir, len2, lob, clp, e);
        fsr_easu_tap(&aC, &aW, (float2)( 1.0f - pp.x,  1.0f - pp.y), dir, len2, lob, clp, k);
        fsr_easu_tap(&aC, &aW, (float2)( 2.0f - pp.x,  1.0f - pp.y), dir, len2, lob, clp, l);
        fsr_easu_tap(&aC, &aW, (float2)( 2.0f - pp.x,  0.0f - pp.y), dir, len2, lob, clp, hh);
        fsr_easu_tap(&aC, &aW, (float2)( 1.0f - pp.x,  0.0f - pp.y), dir, len2, lob, clp, g);
        fsr_easu_tap(&aC, &aW, (float2)( 1.0f - pp.x,  2.0f - pp.y), dir, len2, lob, clp, o);
        fsr_easu_tap(&aC, &aW, (float2)( 0.0f - pp.x,  2.0f - pp.y), dir, len2, lob, clp, n);

        const float pix = (aW > 0.0f) ? (aC * native_recip(aW)) : f;
        const float dered = fmin(mx4, fmax(mn4, pix));
        const float clamped = clamp(dered, 0.0f, 1.0f);
#if defined(FSR1_FP16_SCRATCH) && FSR1_FP16_SCRATCH
        // FP16 path: store the [0, 1] float directly via vstore_half;
        // RCAS reads it back via fsr_load_mid -> vload_half with no
        // integer-pixel quantise round-trip. dstPitch is in BYTES (the
        // host passes width * sizeof(cl_half) = width * 2); we recover
        // the half-element row pitch by shifting right by 1.
        const int half_pitch_elems = dstPitch >> 1;
        vstore_half(clamped, iy * half_pitch_elems + ix, (__global half *)pDst);
#else
        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(clamped * (float)((1 << bit_depth) - 1));
#endif
    }
}

// RCAS: 5-tap cross sharpener with FSR_RCAS_LIMIT cap and per-pixel
// hitMin/hitMax saturation prevention. `con0_sharp` is exp2(-sharpness_stops)
// pre-computed host-side.
__kernel void kernel_rcas(
    __global uchar *restrict pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    __global const uchar *pSrc, const int srcPitch,
    const float con0_sharp) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < dstWidth && iy < dstHeight) {
        // RCAS source is the EASU intermediate -- fsr_load_mid routes
        // between FP16 (vload_half) and integer-pixel (fsr_load_norm)
        // based on the build-time FSR1_FP16_SCRATCH flag.
        const float bV = fsr_load_mid(pSrc, srcPitch, ix,     iy - 1, dstWidth, dstHeight);
        const float dV = fsr_load_mid(pSrc, srcPitch, ix - 1, iy,     dstWidth, dstHeight);
        const float eV = fsr_load_mid(pSrc, srcPitch, ix,     iy,     dstWidth, dstHeight);
        const float fV = fsr_load_mid(pSrc, srcPitch, ix + 1, iy,     dstWidth, dstHeight);
        const float hV = fsr_load_mid(pSrc, srcPitch, ix,     iy + 1, dstWidth, dstHeight);

        // Ring min/max (4-tap, no centre).
        const float mn4 = fmin(fmin(fmin(bV, dV), fV), hV);
        const float mx4 = fmax(fmax(fmax(bV, dV), fV), hV);

        // hitMin / hitMax: solve for the maximum w that keeps the output
        // inside [0, 1]. The reference uses `min(ring) * rcp(4*max(ring))`
        // for the dark side and a symmetric form for the bright side.
        const float rcpMx4 = (mx4 > 0.0f) ? native_recip(4.0f * mx4) : 0.0f;
        const float rcpMn4 = native_recip(4.0f * mn4 - 4.0f);
        const float hitMin = fmin(mn4, eV) * rcpMx4;
        const float hitMax = (1.0f - fmax(mx4, eV)) * rcpMn4;
        float lobe = fmax(-hitMin, hitMax);
        lobe = fmax(-FSR_RCAS_LIMIT, fmin(lobe, 0.0f)) * con0_sharp;

        const float rcpL = native_recip(4.0f * lobe + 1.0f);
        const float pix  = (lobe * (bV + dV + fV + hV) + eV) * rcpL;

        const float result = clamp(pix, 0.0f, 1.0f);
        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(result * (float)((1 << bit_depth) - 1));
    }
}

__kernel void kernel_resize(
    __global uchar *restrict pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    __global const uchar *restrict pSrc, const int srcPitch, const int srcWidth, const int srcHeight,
    const float ratioX, const float ratioY, __global const float4 *restrict pgFactor
) {
#if USE_LOCAL
    __local float weightXshared[shared_weightXdim * block_x];
    __local float weightYshared[shared_weightYdim * block_y];
    __local float4 psCopyFactor[radius];
#endif
    const int threadIdX = get_local_id(0);
    const int threadIdY = get_local_id(1);

    const float ratioInvX = 1.0f / ratioX;
    const float ratioClampedX = min(ratioX, 1.0f);
    const float srcWindowX = radius / ratioClampedX;

    const float ratioInvY = 1.0f / ratioY;
    const float ratioClampedY = min(ratioY, 1.0f);
    const float srcWindowY = radius / ratioClampedY;

#if USE_LOCAL
    if (algo == WEIGHT_SPLINE) {
        if (threadIdY == 0) {
            if (threadIdX < radius) {
                psCopyFactor[threadIdX] = pgFactor[threadIdX];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (threadIdY == 0) {
        // threadIdY==0のスレッドが、x方向の重みをそれぞれ計算してsharedメモリに書き込み
        const int dstX = get_group_id(0) * block_x + threadIdX;
        const float srcX = ((float)(dstX + 0.5f)) * ratioInvX;
        const int srcFirstX = max(0, (int)floor(srcX - srcWindowX));
        const int srcEndX = min(srcWidth - 1, (int)ceil(srcX + srcWindowX));
        calc_weight_to_local(weightXshared + threadIdX * shared_weightXdim, srcX, srcFirstX, srcEndX, ratioClampedX, psCopyFactor);

        if (threadIdX < block_y) {
            // threadIdY==0のスレッドが、y方向の重みをそれぞれ計算してsharedメモリに書き込み
            const int thready = threadIdX;
            const int dstY = get_group_id(1) * block_y + thready;
            const float srcY = ((float)(dstY + 0.5f)) * ratioInvY;
            const int srcFirstY = max(0, (int)floor(srcY - srcWindowY));
            const int srcEndY = min(srcHeight - 1, (int)ceil(srcY + srcWindowY));
            calc_weight_to_local(weightYshared + thready * shared_weightYdim, srcY, srcFirstY, srcEndY, ratioClampedY, psCopyFactor);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    const int ix = get_group_id(0) * block_x + threadIdX;
    const int iy = get_group_id(1) * block_y + threadIdY;

    if (ix < dstWidth && iy < dstHeight) {
        //ピクセルの中心を算出してからスケール
        const float x = ((float)ix + 0.5f) * ratioX;
        const float y = ((float)iy + 0.5f) * ratioY;

        const float srcX = ((float)(ix + 0.5f)) * ratioInvX;
        const int srcFirstX = max(0, (int)floor(srcX - srcWindowX));
        const int srcEndX = min(srcWidth - 1, (int)ceil(srcX + srcWindowX));

        const float srcY = ((float)(iy + 0.5f)) * ratioInvY;
        const int srcFirstY = max(0, (int)floor(srcY - srcWindowY));
        const int srcEndY = min(srcHeight - 1, (int)ceil(srcY + srcWindowY));
#if USE_LOCAL
        __local const float *weightX = weightXshared + threadIdX * shared_weightXdim;
        __local const float *weightY = weightYshared + threadIdY * shared_weightYdim;
#endif

        const __global uchar *srcLine = pSrc + srcFirstY * srcPitch + srcFirstX * sizeof(Type);
        float clr = 0.0f;
        float sumWeight = 0.0f;
        for (int j = srcFirstY; j <= srcEndY; j++, srcLine += srcPitch
#if USE_LOCAL
            , weightY++
#endif
        ) {
#if USE_LOCAL
            const float wy = weightY[0];
            __local const float *pwx = weightX;
#else
            const float wy = calc_weight(j, srcY, ratioClampedY, pgFactor);
#endif
            if (wy != 0.0f) {
                __global const Type *srcPtr = (__global const Type*)srcLine;
                for (int i = srcFirstX; i <= srcEndX; i++, srcPtr++
#if USE_LOCAL
                    , pwx++
#endif
                ) {
#if USE_LOCAL
                    const float wx = pwx[0];
#else
                    const float wx = calc_weight(i, srcX, ratioClampedX, pgFactor);
#endif
                    clr += srcPtr[0] * wx * wy;
                    sumWeight += wx * wy;
                }
            }
        }
        clr /= sumWeight;

        __global Type* ptr = (__global Type*)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(clr, 0.0f, (1 << bit_depth) - 0.1f);
    }
}

float calc_gauss_weight(const int targetPos, const float srcPos, const float ratioClamped) {
    const float delta = ((float)targetPos - (srcPos - 0.5f)) * ratioClamped;
    return factor_gauss(delta);
}

#if USE_LOCAL
void calc_gauss_weight_to_local_normalized(
    __local float *pWeight, const float srcPos, const int srcFirst, const int srcEnd,
    const float ratioClamped, const int weightDim) {
    float sumWeight = 0.0f;
    const int count = srcEnd - srcFirst + 1;
    for (int i = 0; i < weightDim; i++) {
        const float w = (i < count) ? calc_gauss_weight(srcFirst + i, srcPos, ratioClamped) : 0.0f;
        pWeight[i] = w;
        sumWeight += w;
    }
    if (sumWeight > 0.0f) {
        const float invSumWeight = 1.0f / sumWeight;
        for (int i = 0; i < weightDim; i++) {
            pWeight[i] *= invSumWeight;
        }
    }
}
#endif

__kernel void kernel_resize_gauss_h(
    __global uchar *restrict pTmp, const int tmpPitch, const int tmpWidth, const int tmpHeight,
    __global const uchar *restrict pSrc, const int srcPitch, const int srcWidth, const int srcHeight,
    const float ratioX
) {
#if USE_LOCAL
    __local float weightXshared[shared_weightXdim * block_x];
#endif
    const int threadIdX = get_local_id(0);
    const int threadIdY = get_local_id(1);

    const float ratioInvX = 1.0f / ratioX;
    const float ratioClampedX = min(ratioX, 1.0f);
    const float srcWindowX = radius / ratioClampedX;

#if USE_LOCAL
    if (threadIdY == 0) {
        const int dstX = get_group_id(0) * block_x + threadIdX;
        const float srcX = ((float)dstX + 0.5f) * ratioInvX;
        const int srcFirstX = max(0, (int)floor(srcX - srcWindowX));
        const int srcEndX = min(srcWidth - 1, (int)ceil(srcX + srcWindowX));
        calc_gauss_weight_to_local_normalized(
            weightXshared + threadIdX * shared_weightXdim,
            srcX, srcFirstX, srcEndX, ratioClampedX, shared_weightXdim);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    const int ix = get_group_id(0) * block_x + threadIdX;
    const int iy = get_group_id(1) * block_y + threadIdY;

    if (ix < tmpWidth && iy < tmpHeight) {
        const float srcX = ((float)ix + 0.5f) * ratioInvX;
        const int srcFirstX = max(0, (int)floor(srcX - srcWindowX));
        const int srcEndX = min(srcWidth - 1, (int)ceil(srcX + srcWindowX));
#if USE_LOCAL
        __local const float *weightX = weightXshared + threadIdX * shared_weightXdim;
#endif

        __global const Type *srcPtr = (__global const Type *)(pSrc + iy * srcPitch + srcFirstX * sizeof(Type));
        float clr = 0.0f;
#if !USE_LOCAL
        float sumWeight = 0.0f;
#endif
        for (int i = srcFirstX; i <= srcEndX; i++, srcPtr++
#if USE_LOCAL
            , weightX++
#endif
        ) {
#if USE_LOCAL
            const float wx = weightX[0];
#else
            const float wx = calc_gauss_weight(i, srcX, ratioClampedX);
            sumWeight += wx;
#endif
            clr += srcPtr[0] * wx;
        }
#if !USE_LOCAL
        if (sumWeight > 0.0f) {
            clr /= sumWeight;
        }
#endif
        ((__global float *)(pTmp + iy * tmpPitch) + ix)[0] = clr;
    }
}

__kernel void kernel_resize_gauss_v(
    __global uchar *restrict pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    __global const uchar *restrict pTmp, const int tmpPitch, const int tmpWidth, const int tmpHeight,
    const float ratioY
) {
#if USE_LOCAL
    __local float weightYshared[shared_weightYdim * block_y];
#endif
    const int threadIdX = get_local_id(0);
    const int threadIdY = get_local_id(1);

    const float ratioInvY = 1.0f / ratioY;
    const float ratioClampedY = min(ratioY, 1.0f);
    const float srcWindowY = radius / ratioClampedY;

#if USE_LOCAL
    if (threadIdX == 0) {
        const int dstY = get_group_id(1) * block_y + threadIdY;
        const float srcY = ((float)dstY + 0.5f) * ratioInvY;
        const int srcFirstY = max(0, (int)floor(srcY - srcWindowY));
        const int srcEndY = min(tmpHeight - 1, (int)ceil(srcY + srcWindowY));
        calc_gauss_weight_to_local_normalized(
            weightYshared + threadIdY * shared_weightYdim,
            srcY, srcFirstY, srcEndY, ratioClampedY, shared_weightYdim);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    const int ix = get_group_id(0) * block_x + threadIdX;
    const int iy = get_group_id(1) * block_y + threadIdY;

    if (ix < dstWidth && iy < dstHeight) {
        const float srcY = ((float)iy + 0.5f) * ratioInvY;
        const int srcFirstY = max(0, (int)floor(srcY - srcWindowY));
        const int srcEndY = min(tmpHeight - 1, (int)ceil(srcY + srcWindowY));
#if USE_LOCAL
        __local const float *weightY = weightYshared + threadIdY * shared_weightYdim;
#endif

        float clr = 0.0f;
#if !USE_LOCAL
        float sumWeight = 0.0f;
#endif
        for (int j = srcFirstY; j <= srcEndY; j++
#if USE_LOCAL
            , weightY++
#endif
        ) {
#if USE_LOCAL
            const float wy = weightY[0];
#else
            const float wy = calc_gauss_weight(j, srcY, ratioClampedY);
            sumWeight += wy;
#endif
            clr += ((__global const float *)(pTmp + j * tmpPitch) + ix)[0] * wy;
        }
#if !USE_LOCAL
        if (sumWeight > 0.0f) {
            clr /= sumWeight;
        }
#endif

        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(clamp(clr, 0.0f, (float)((1 << bit_depth) - 1)) + 0.5f);
    }
}

// ====================================================================
// NIS (NVIDIA Image Scaling)
// ====================================================================
// Algorithm reference: NIS v1.0.3 NIS_Scaler.h / NIS_Main.glsl by NVIDIA
// (MIT licence; coefficient tables sit in nis_coef_tables.h on the host
// side and are uploaded as constant cl_mem). This block lives in the
// shared resize .cl because the existing build pipeline compiles one
// program per filter, and NIS shares its host driver with the other
// resize algos.
//
// Build-time defines expected from the host:
//   NIS_KERNEL_ENABLED   1 to compile this block, 0/undefined to skip
//   NIS_BLOCK_WIDTH      pixels per work-group, X axis
//   NIS_BLOCK_HEIGHT     pixels per work-group, Y axis
//   NIS_HDR_MODE         0=None, 1=Linear, 2=PQ (matches NIS_Config.h enum)
// ====================================================================

#ifdef NIS_KERNEL_ENABLED

#ifndef NIS_BLOCK_WIDTH
#define NIS_BLOCK_WIDTH  32
#endif
#ifndef NIS_BLOCK_HEIGHT
#define NIS_BLOCK_HEIGHT 8
#endif
#ifndef NIS_HDR_MODE
#define NIS_HDR_MODE     0
#endif

// Mirror of host-side NISConfig (NIS_Config.h:45). Field order, types,
// and 256-byte alignment match the host struct so the cl_mem can be
// uploaded as a single byte-block. Keep this in lock-step with the
// host-side NISConfigHost definition in rgy_filter_resize.cpp.
typedef struct __attribute__((aligned(256))) NISConfigCL {
    float kDetectRatio;
    float kDetectThres;
    float kMinContrastRatio;
    float kRatioNorm;

    float kContrastBoost;
    float kEps;
    float kSharpStartY;
    float kSharpScaleY;

    float kSharpStrengthMin;
    float kSharpStrengthScale;
    float kSharpLimitMin;
    float kSharpLimitScale;

    float kScaleX;
    float kScaleY;
    float kDstNormX;
    float kDstNormY;

    float kSrcNormX;
    float kSrcNormY;

    uint  kInputViewportOriginX;
    uint  kInputViewportOriginY;
    uint  kInputViewportWidth;
    uint  kInputViewportHeight;

    uint  kOutputViewportOriginX;
    uint  kOutputViewportOriginY;
    uint  kOutputViewportWidth;
    uint  kOutputViewportHeight;

    float reserved0;
    float reserved1;
} NISConfigCL;

// K1: integer sample with edge clamp, normalised to [0,1] in SDR / kept
// raw in HDR linear / kept PQ-coded in HDR PQ. The NIS reference only
// reads luminance for the edge map + USM gates; the carry channels go
// through unmodified. QSVEnc routes one plane at a time so this is just
// a single-channel scalar.
static inline float nis_sample(__global const Type *src, int srcPitch, int srcW, int srcH, int x, int y) {
    x = (x < 0) ? 0 : ((x >= srcW) ? (srcW - 1) : x);
    y = (y < 0) ? 0 : ((y >= srcH) ? (srcH - 1) : y);
    const __global Type *row = (const __global Type *)((const __global uchar *)src + y * srcPitch);
    const float v = (float)row[x];
    const float maxv = (float)((1 << bit_depth) - 1);
    return v / maxv;  // [0,1] -- HDR PQ inputs are already encoded in this range
}

// K2: edge-direction estimator. Returns the dominant gradient bucket
// (0..3 = horizontal / vertical / diag45 / diag135) given 4 luminance
// samples around (x,y).
static inline int nis_get_edge_dir(__global const Type *src, int srcPitch, int srcW, int srcH, int x, int y) {
    const float c  = nis_sample(src, srcPitch, srcW, srcH, x,     y);
    const float r  = nis_sample(src, srcPitch, srcW, srcH, x + 1, y);
    const float d  = nis_sample(src, srcPitch, srcW, srcH, x,     y + 1);
    const float rd = nis_sample(src, srcPitch, srcW, srcH, x + 1, y + 1);
    const float gx = fabs(r - c);
    const float gy = fabs(d - c);
    const float g45 = fabs(rd - c);
    const float g135 = fabs(r - d);
    int best = 0;
    float gmax = gx;
    if (gy   > gmax) { gmax = gy;   best = 1; }
    if (g45  > gmax) { gmax = g45;  best = 2; }
    if (g135 > gmax) { gmax = g135; best = 3; }
    return best;
}

// K3: coefficient lookup. Returns 8-tap weights for the given phase
// (0..63). NIS organises taps as float8s in row-major (phase, tap). The
// uploaded buffers are `kPhaseCount * kFilterSize` floats wide.
static inline float nis_get_coef_scale(__constant const float *coefScale, int phase, int tap) {
    return coefScale[phase * 8 + tap];
}
static inline float nis_get_coef_usm(__constant const float *coefUsm, int phase, int tap) {
    return coefUsm[phase * 8 + tap];
}

// 6-tap separable polyphase apply. NIS's
// coef_scale and coef_usm share the same 64-phase x 8-tap shape (only
// taps 0..5 carry weight; taps 6,7 are 0 by data so we skip them).
// Same convolution kernel structure -- pass coef_scale for K4 (base
// resampled luma) or coef_usm for K5 (high-pass detail, sum=0 by
// construction).
// Math:
//   sx = (dx + 0.5) * kScaleX - 0.5     (half-pixel centred mapping)
//   isx = floor(sx);  frac = sx - isx
//   phase_x = clamp((int)(frac * 64), 0, 63)
// Tap i in [0..5] reads source pixel at (isx + i - 2), so phase 0 ->
// coef_scale[0] = [0,0,1,0,0,0,..] is exactly the identity at 1x scale
// (T1 stays bit-exact).
//
// Separability: compute 6 horizontal partial sums (one per source row in
// the 6-row neighbourhood), then combine via the vertical weights. 6+6
// LUT lookups + 36 sample-and-multiply per output pixel.
static inline float nis_polyphase_apply(
    __global const Type *src, int srcPitch, int srcW, int srcH,
    __constant const NISConfigCL *cfg,
    __constant const float *coefTable,
    int dx, int dy)
{
    const float sx = ((float)dx + 0.5f) * cfg->kScaleX - 0.5f;
    const float sy = ((float)dy + 0.5f) * cfg->kScaleY - 0.5f;
    const int isx_base = (int)floor(sx);
    const int isy_base = (int)floor(sy);
    const float fx = sx - (float)isx_base;
    const float fy = sy - (float)isy_base;

    int phase_x = (int)(fx * 64.0f);
    int phase_y = (int)(fy * 64.0f);
    if (phase_x > 63) phase_x = 63;
    if (phase_y > 63) phase_y = 63;
    if (phase_x < 0)  phase_x = 0;
    if (phase_y < 0)  phase_y = 0;

    // Pre-fetch the 6 active weights per axis (saves repeated LUT reads
    // inside the inner loop).
    const float wx0 = nis_get_coef_scale(coefTable, phase_x, 0);
    const float wx1 = nis_get_coef_scale(coefTable, phase_x, 1);
    const float wx2 = nis_get_coef_scale(coefTable, phase_x, 2);
    const float wx3 = nis_get_coef_scale(coefTable, phase_x, 3);
    const float wx4 = nis_get_coef_scale(coefTable, phase_x, 4);
    const float wx5 = nis_get_coef_scale(coefTable, phase_x, 5);
    const float wy0 = nis_get_coef_scale(coefTable, phase_y, 0);
    const float wy1 = nis_get_coef_scale(coefTable, phase_y, 1);
    const float wy2 = nis_get_coef_scale(coefTable, phase_y, 2);
    const float wy3 = nis_get_coef_scale(coefTable, phase_y, 3);
    const float wy4 = nis_get_coef_scale(coefTable, phase_y, 4);
    const float wy5 = nis_get_coef_scale(coefTable, phase_y, 5);

    float result = 0.0f;
    #pragma unroll
    for (int j = 0; j < 6; j++) {
        const int sy_int = isy_base + (j - 2);
        const float p0 = nis_sample(src, srcPitch, srcW, srcH, isx_base + (0 - 2), sy_int);
        const float p1 = nis_sample(src, srcPitch, srcW, srcH, isx_base + (1 - 2), sy_int);
        const float p2 = nis_sample(src, srcPitch, srcW, srcH, isx_base + (2 - 2), sy_int);
        const float p3 = nis_sample(src, srcPitch, srcW, srcH, isx_base + (3 - 2), sy_int);
        const float p4 = nis_sample(src, srcPitch, srcW, srcH, isx_base + (4 - 2), sy_int);
        const float p5 = nis_sample(src, srcPitch, srcW, srcH, isx_base + (5 - 2), sy_int);
        const float rowsum = p0 * wx0 + p1 * wx1 + p2 * wx2 + p3 * wx3 + p4 * wx4 + p5 * wx5;
        const float wy = (j == 0) ? wy0 : (j == 1) ? wy1 : (j == 2) ? wy2
                       : (j == 3) ? wy3 : (j == 4) ? wy4 : wy5;
        result += rowsum * wy;
    }
    return result;
}

__attribute__((reqd_work_group_size(NIS_BLOCK_WIDTH, NIS_BLOCK_HEIGHT, 1)))
__kernel void kernel_nis_scaler(
    __global       uchar *pDst,
    const int            dstPitch,
    const int            dstWidth,
    const int            dstHeight,
    __global const Type  *pSrc,
    const int            srcPitch,
    const int            srcWidth,
    const int            srcHeight,
    __constant const NISConfigCL *cfg,
    __constant const float       *coefScale,
    __constant const float       *coefUsm)
{
    const int dx = get_global_id(0);
    const int dy = get_global_id(1);
    if (dx >= dstWidth || dy >= dstHeight) return;

    // Base resampled luma via polyphase scale LUT.
    const float y_base = nis_polyphase_apply(pSrc, srcPitch, srcWidth, srcHeight,
                                              cfg, coefScale, dx, dy);

    // Unsharp-mask high-pass via polyphase USM LUT. coef_usm
    // sums to 0 by construction so this is the local detail signal --
    // positive where the pixel is brighter than its neighbourhood,
    // negative where darker.
    const float y_usm = nis_polyphase_apply(pSrc, srcPitch, srcWidth, srcHeight,
                                             cfg, coefUsm, dx, dy);

    // Luminance band gate. NIS only sharpens within [kSharpStartY,
    // kSharpEndY], scaled by kSharpScaleY = 1/(end-start). For HDR PQ
    // mode the host builds this as 0.35..0.55, which is the mid-tone
    // band where edge enhancement is visible but specular highlights
    // (>=1000 nit on a 4000-nit master) are excluded -- the headline
    // protection over ewa-lanczossharp on PQ content.
    float t = (y_base - cfg->kSharpStartY) * cfg->kSharpScaleY;
    t = clamp(t, 0.0f, 1.0f);

    // Strength + limit interpolate linearly across the band. The result
    // is the per-pixel USM amplitude; cap the contribution to +/- limit
    // so a single high-contrast neighbour cannot blow out the pixel.
    const float strength = cfg->kSharpStrengthMin + t * cfg->kSharpStrengthScale;
    const float limit    = cfg->kSharpLimitMin    + t * cfg->kSharpLimitScale;
    const float usm_clamped = clamp(y_usm, -limit, limit);

    // Keep the edge-direction helper alive for future directional weighting.
    (void)nis_get_edge_dir(pSrc, srcPitch, srcWidth, srcHeight, 0, 0);

    const float result = y_base + strength * usm_clamped;

    const float maxv = (float)((1 << bit_depth) - 1);
    __global Type *ptr = (__global Type *)(pDst + dy * dstPitch + dx * sizeof(Type));
    ptr[0] = (Type)(clamp(result * maxv, 0.0f, maxv) + 0.5f);
}

// === Perf 3 opt=gather kernel removed 2026-06-11 after A/B audit ===
// Shared 6x6 source gather between K4 and K5 measured -1.1% LOSS at
// 1500 frames on Arc A770 (private 6x6 float array adds register
// pressure; the L1 cache already keeps the K4 neighbourhood hot
// for K5's re-fetch). Implementation in git history.
//
// === Perf 4 opt=separable kernels removed 2026-06-11 ===
// 2-pass H+V scaler with float intermediate measured -0.01% WASH at
// 1500 frames on Arc A770 (the math reduction from 36 to 12 taps was
// wiped out by the intermediate buffer roundtrip cost). Output was
// not bit-equivalent (intermediate rounding). Implementation in git
// history.
//
// Below: only the winning K4-only + SLM kernels remain.


// Cooperative SLM tile load for opt=fast.
// Each work-group loads its source neighbourhood + 6-tap halo into
// __local memory, then all threads in the group compute their
// polyphase output from the cached tile. Saves the per-thread global
// memory load count if the source tile would otherwise miss L1
// (typical for large work-group counts).
//
// Tile size: NIS_BLOCK_W=32, NIS_BLOCK_H=8, kScale in [0.5, 1.0]
//   max source span = 32 * 1.0 + 6 = 38 (width), 8 * 1.0 + 6 = 14 (height)
// Round up to 40 x 32 for safety. As float = 5 KB SLM per work-group;
// Arc A770 has 64 KB SLM total so plenty of concurrent groups fit.
//
// BM3D precedent (opt=slm LOST): Arc A770 L1 already caches small
// windows. Likely loss expected here too; testing for data.
#define NIS_SLM_TILE_W 40
#define NIS_SLM_TILE_H 32

__attribute__((reqd_work_group_size(NIS_BLOCK_WIDTH, NIS_BLOCK_HEIGHT, 1)))
__kernel void kernel_nis_scaler_slm(
    __global       uchar *pDst,
    const int            dstPitch,
    const int            dstWidth,
    const int            dstHeight,
    __global const Type  *pSrc,
    const int            srcPitch,
    const int            srcWidth,
    const int            srcHeight,
    __constant const NISConfigCL *cfg,
    __constant const float       *coefScale,
    __constant const float       *coefUsm)
{
    __local float src_tile[NIS_SLM_TILE_H][NIS_SLM_TILE_W];

    const int wg_dx0 = get_group_id(0) * NIS_BLOCK_WIDTH;
    const int wg_dy0 = get_group_id(1) * NIS_BLOCK_HEIGHT;

    // Source-space anchor for this work-group's left/top output pixel.
    const float sx0_f = ((float)wg_dx0 + 0.5f) * cfg->kScaleX - 0.5f;
    const float sy0_f = ((float)wg_dy0 + 0.5f) * cfg->kScaleY - 0.5f;
    const int sbb_x0 = (int)floor(sx0_f) - 2;
    const int sbb_y0 = (int)floor(sy0_f) - 2;

    // Cooperatively load NIS_SLM_TILE_W x NIS_SLM_TILE_H source pixels.
    // Each thread loads ~ceil(TILE_W*TILE_H / wg_size) pixels.
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lid = ly * NIS_BLOCK_WIDTH + lx;
    const int wg_size = NIS_BLOCK_WIDTH * NIS_BLOCK_HEIGHT;
    const int total_load = NIS_SLM_TILE_W * NIS_SLM_TILE_H;
    for (int idx = lid; idx < total_load; idx += wg_size) {
        const int tlx = idx % NIS_SLM_TILE_W;
        const int tly = idx / NIS_SLM_TILE_W;
        src_tile[tly][tlx] = nis_sample(pSrc, srcPitch, srcWidth, srcHeight,
                                         sbb_x0 + tlx, sbb_y0 + tly);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int dx = wg_dx0 + lx;
    const int dy = wg_dy0 + ly;
    if (dx >= dstWidth || dy >= dstHeight) return;

    const float sx = ((float)dx + 0.5f) * cfg->kScaleX - 0.5f;
    const float sy = ((float)dy + 0.5f) * cfg->kScaleY - 0.5f;
    const int isx_base = (int)floor(sx);
    const int isy_base = (int)floor(sy);
    const float fx = sx - (float)isx_base;
    const float fy = sy - (float)isy_base;

    int phase_x = (int)(fx * 64.0f);
    int phase_y = (int)(fy * 64.0f);
    if (phase_x > 63) phase_x = 63;
    if (phase_y > 63) phase_y = 63;
    if (phase_x < 0)  phase_x = 0;
    if (phase_y < 0)  phase_y = 0;

    // SLM-relative source indices.
    const int slm_x0 = isx_base - sbb_x0;
    const int slm_y0 = isy_base - sbb_y0;

    const float sxw0 = nis_get_coef_scale(coefScale, phase_x, 0);
    const float sxw1 = nis_get_coef_scale(coefScale, phase_x, 1);
    const float sxw2 = nis_get_coef_scale(coefScale, phase_x, 2);
    const float sxw3 = nis_get_coef_scale(coefScale, phase_x, 3);
    const float sxw4 = nis_get_coef_scale(coefScale, phase_x, 4);
    const float sxw5 = nis_get_coef_scale(coefScale, phase_x, 5);
    const float syw0 = nis_get_coef_scale(coefScale, phase_y, 0);
    const float syw1 = nis_get_coef_scale(coefScale, phase_y, 1);
    const float syw2 = nis_get_coef_scale(coefScale, phase_y, 2);
    const float syw3 = nis_get_coef_scale(coefScale, phase_y, 3);
    const float syw4 = nis_get_coef_scale(coefScale, phase_y, 4);
    const float syw5 = nis_get_coef_scale(coefScale, phase_y, 5);

    const float uxw0 = nis_get_coef_scale(coefUsm, phase_x, 0);
    const float uxw1 = nis_get_coef_scale(coefUsm, phase_x, 1);
    const float uxw2 = nis_get_coef_scale(coefUsm, phase_x, 2);
    const float uxw3 = nis_get_coef_scale(coefUsm, phase_x, 3);
    const float uxw4 = nis_get_coef_scale(coefUsm, phase_x, 4);
    const float uxw5 = nis_get_coef_scale(coefUsm, phase_x, 5);
    const float uyw0 = nis_get_coef_scale(coefUsm, phase_y, 0);
    const float uyw1 = nis_get_coef_scale(coefUsm, phase_y, 1);
    const float uyw2 = nis_get_coef_scale(coefUsm, phase_y, 2);
    const float uyw3 = nis_get_coef_scale(coefUsm, phase_y, 3);
    const float uyw4 = nis_get_coef_scale(coefUsm, phase_y, 4);
    const float uyw5 = nis_get_coef_scale(coefUsm, phase_y, 5);

    float y_base = 0.0f;
    float y_usm  = 0.0f;
    #pragma unroll
    for (int j = 0; j < 6; j++) {
        const int sly = slm_y0 + (j - 2);
        const float p0 = src_tile[sly][slm_x0 - 2];
        const float p1 = src_tile[sly][slm_x0 - 1];
        const float p2 = src_tile[sly][slm_x0    ];
        const float p3 = src_tile[sly][slm_x0 + 1];
        const float p4 = src_tile[sly][slm_x0 + 2];
        const float p5 = src_tile[sly][slm_x0 + 3];
        const float scale_row = p0*sxw0 + p1*sxw1 + p2*sxw2 + p3*sxw3 + p4*sxw4 + p5*sxw5;
        const float usm_row   = p0*uxw0 + p1*uxw1 + p2*uxw2 + p3*uxw3 + p4*uxw4 + p5*uxw5;
        const float wys = (j == 0) ? syw0 : (j == 1) ? syw1 : (j == 2) ? syw2
                       : (j == 3) ? syw3 : (j == 4) ? syw4 : syw5;
        const float wyu = (j == 0) ? uyw0 : (j == 1) ? uyw1 : (j == 2) ? uyw2
                       : (j == 3) ? uyw3 : (j == 4) ? uyw4 : uyw5;
        y_base += scale_row * wys;
        y_usm  += usm_row   * wyu;
    }

    float t = (y_base - cfg->kSharpStartY) * cfg->kSharpScaleY;
    t = clamp(t, 0.0f, 1.0f);
    const float strength = cfg->kSharpStrengthMin + t * cfg->kSharpStrengthScale;
    const float limit    = cfg->kSharpLimitMin    + t * cfg->kSharpLimitScale;
    const float usm_clamped = clamp(y_usm, -limit, limit);
    const float result = y_base + strength * usm_clamped;

    const float maxv = (float)((1 << bit_depth) - 1);
    __global Type *ptr = (__global Type *)(pDst + dy * dstPitch + dx * sizeof(Type));
    ptr[0] = (Type)(clamp(result * maxv, 0.0f, maxv) + 0.5f);
}

// SLM + no-USM variant for cascade intermediate stages with opt=fast
// or opt=fastest (composes opt=slm + opt=skipusm). Same cooperative
// tile load as kernel_nis_scaler_slm, but no K5 polyphase math.
__attribute__((reqd_work_group_size(NIS_BLOCK_WIDTH, NIS_BLOCK_HEIGHT, 1)))
__kernel void kernel_nis_scaler_slm_no_usm(
    __global       uchar *pDst,
    const int            dstPitch,
    const int            dstWidth,
    const int            dstHeight,
    __global const Type  *pSrc,
    const int            srcPitch,
    const int            srcWidth,
    const int            srcHeight,
    __constant const NISConfigCL *cfg,
    __constant const float       *coefScale)
{
    __local float src_tile[NIS_SLM_TILE_H][NIS_SLM_TILE_W];

    const int wg_dx0 = get_group_id(0) * NIS_BLOCK_WIDTH;
    const int wg_dy0 = get_group_id(1) * NIS_BLOCK_HEIGHT;

    const float sx0_f = ((float)wg_dx0 + 0.5f) * cfg->kScaleX - 0.5f;
    const float sy0_f = ((float)wg_dy0 + 0.5f) * cfg->kScaleY - 0.5f;
    const int sbb_x0 = (int)floor(sx0_f) - 2;
    const int sbb_y0 = (int)floor(sy0_f) - 2;

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lid = ly * NIS_BLOCK_WIDTH + lx;
    const int wg_size = NIS_BLOCK_WIDTH * NIS_BLOCK_HEIGHT;
    const int total_load = NIS_SLM_TILE_W * NIS_SLM_TILE_H;
    for (int idx = lid; idx < total_load; idx += wg_size) {
        const int tlx = idx % NIS_SLM_TILE_W;
        const int tly = idx / NIS_SLM_TILE_W;
        src_tile[tly][tlx] = nis_sample(pSrc, srcPitch, srcWidth, srcHeight,
                                         sbb_x0 + tlx, sbb_y0 + tly);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int dx = wg_dx0 + lx;
    const int dy = wg_dy0 + ly;
    if (dx >= dstWidth || dy >= dstHeight) return;

    const float sx = ((float)dx + 0.5f) * cfg->kScaleX - 0.5f;
    const float sy = ((float)dy + 0.5f) * cfg->kScaleY - 0.5f;
    const int isx_base = (int)floor(sx);
    const int isy_base = (int)floor(sy);
    const float fx = sx - (float)isx_base;
    const float fy = sy - (float)isy_base;

    int phase_x = (int)(fx * 64.0f);
    int phase_y = (int)(fy * 64.0f);
    if (phase_x > 63) phase_x = 63;
    if (phase_y > 63) phase_y = 63;
    if (phase_x < 0)  phase_x = 0;
    if (phase_y < 0)  phase_y = 0;

    const int slm_x0 = isx_base - sbb_x0;
    const int slm_y0 = isy_base - sbb_y0;

    const float sxw0 = nis_get_coef_scale(coefScale, phase_x, 0);
    const float sxw1 = nis_get_coef_scale(coefScale, phase_x, 1);
    const float sxw2 = nis_get_coef_scale(coefScale, phase_x, 2);
    const float sxw3 = nis_get_coef_scale(coefScale, phase_x, 3);
    const float sxw4 = nis_get_coef_scale(coefScale, phase_x, 4);
    const float sxw5 = nis_get_coef_scale(coefScale, phase_x, 5);
    const float syw0 = nis_get_coef_scale(coefScale, phase_y, 0);
    const float syw1 = nis_get_coef_scale(coefScale, phase_y, 1);
    const float syw2 = nis_get_coef_scale(coefScale, phase_y, 2);
    const float syw3 = nis_get_coef_scale(coefScale, phase_y, 3);
    const float syw4 = nis_get_coef_scale(coefScale, phase_y, 4);
    const float syw5 = nis_get_coef_scale(coefScale, phase_y, 5);

    float y_base = 0.0f;
    #pragma unroll
    for (int j = 0; j < 6; j++) {
        const int sly = slm_y0 + (j - 2);
        const float p0 = src_tile[sly][slm_x0 - 2];
        const float p1 = src_tile[sly][slm_x0 - 1];
        const float p2 = src_tile[sly][slm_x0    ];
        const float p3 = src_tile[sly][slm_x0 + 1];
        const float p4 = src_tile[sly][slm_x0 + 2];
        const float p5 = src_tile[sly][slm_x0 + 3];
        const float scale_row = p0*sxw0 + p1*sxw1 + p2*sxw2 + p3*sxw3 + p4*sxw4 + p5*sxw5;
        const float wys = (j == 0) ? syw0 : (j == 1) ? syw1 : (j == 2) ? syw2
                       : (j == 3) ? syw3 : (j == 4) ? syw4 : syw5;
        y_base += scale_row * wys;
    }

    const float maxv = (float)((1 << bit_depth) - 1);
    __global Type *ptr = (__global Type *)(pDst + dy * dstPitch + dx * sizeof(Type));
    ptr[0] = (Type)(clamp(y_base * maxv, 0.0f, maxv) + 0.5f);
}

// K4-only kernel for cascade intermediate stages. Same scaler math as kernel_nis_scaler with the
// USM polyphase completely removed -- intermediate stages have their
// USM contribution multiplied by 0 in the cascade math anyway, so this
// path saves the ~36 sample-and-multiply per pixel that was wasted.
// The two helpers (nis_get_edge_dir, nis_get_coef_usm) are NOT touched
// here so the compiler can DCE their bodies if no caller references
// them through this kernel.
__attribute__((reqd_work_group_size(NIS_BLOCK_WIDTH, NIS_BLOCK_HEIGHT, 1)))
__kernel void kernel_nis_scaler_no_usm(
    __global       uchar *pDst,
    const int            dstPitch,
    const int            dstWidth,
    const int            dstHeight,
    __global const Type  *pSrc,
    const int            srcPitch,
    const int            srcWidth,
    const int            srcHeight,
    __constant const NISConfigCL *cfg,
    __constant const float       *coefScale)
{
    const int dx = get_global_id(0);
    const int dy = get_global_id(1);
    if (dx >= dstWidth || dy >= dstHeight) return;

    const float y_base = nis_polyphase_apply(pSrc, srcPitch, srcWidth, srcHeight,
                                              cfg, coefScale, dx, dy);

    const float maxv = (float)((1 << bit_depth) - 1);
    __global Type *ptr = (__global Type *)(pDst + dy * dstPitch + dx * sizeof(Type));
    ptr[0] = (Type)(clamp(y_base * maxv, 0.0f, maxv) + 0.5f);
}

#endif // NIS_KERNEL_ENABLED
