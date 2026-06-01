
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

#if defined(FSR1_FP16_SCRATCH) && FSR1_FP16_SCRATCH
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

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
