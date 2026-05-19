// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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

#ifndef Type
#define Type uchar
#endif

#define KFM_CAT_(a, b) a##b
#define KFM_CAT(a, b) KFM_CAT_(a, b)
#define Type4 KFM_CAT(Type, 4)

static inline int4 kfm_to_int4(const Type4 v) {
#if bit_depth > 8
    return (int4)(v.x, v.y, v.z, v.w);
#else
    return convert_int4(v);
#endif
}

static inline Type4 kfm_to_type4(const int4 v) {
#if bit_depth > 8
    return convert_ushort4_sat(v);
#else
    return convert_uchar4_sat(v);
#endif
}

static inline Type4 kfm_to_type4_float(const float4 v) {
#if bit_depth > 8
    return convert_ushort4_sat(v);
#else
    return convert_uchar4_sat(v);
#endif
}

static inline int4 kfm_calc_combe4(const Type4 a, const Type4 b, const Type4 c, const Type4 d, const Type4 e) {
    const int4 diff = kfm_to_int4(a) + kfm_to_int4(c) * 4 + kfm_to_int4(e) - (kfm_to_int4(b) + kfm_to_int4(d)) * 3;
    return max(diff, -diff);
}

__kernel void kernel_kfm_static(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width,
    const int height) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    const __global Type *pSrc = (const __global Type *)(src + y * srcPitch + x * (int)sizeof(Type));
    __global Type *pDst = (__global Type *)(dst + y * dstPitch + x * (int)sizeof(Type));
    pDst[0] = pSrc[0];
}

__kernel void kernel_kfm_static_from_diff5(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src0,
    const __global uchar *src1,
    const __global uchar *src2,
    const __global uchar *src3,
    const __global uchar *src4,
    const int srcPitch,
    const int width,
    const int height,
    const int threshStatic,
    const int threshMotion) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int srcPitchT = srcPitch / (int)sizeof(Type);
    const int dstPitchT = dstPitch / (int)sizeof(Type);
    const int off = x + y * srcPitchT;
    const int v0 = ((const __global Type *)src0)[off];
    const int v1 = ((const __global Type *)src1)[off];
    const int v2 = ((const __global Type *)src2)[off];
    const int v3 = ((const __global Type *)src3)[off];
    const int v4 = ((const __global Type *)src4)[off];
    const int minv = min(min(v0, v1), min(v2, min(v3, v4)));
    const int maxv = max(max(v0, v1), max(v2, max(v3, v4)));
    const int diff = maxv - minv;
    int coef = 0;
    if (diff <= threshStatic) {
        coef = 128;
    } else if (diff < threshMotion) {
        coef = ((threshMotion - diff) * 128) / max(threshMotion - threshStatic, 1);
    }
    ((__global Type *)dst)[x + y * dstPitchT] = (Type)clamp(coef, 0, 128);
}

__kernel void kernel_kfm_zero(
    __global uchar *dst,
    const int dstPitch,
    const int width,
    const int height) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    __global Type *pDst = (__global Type *)(dst + y * dstPitch + x * (int)sizeof(Type));
    pDst[0] = (Type)0;
}

__kernel void kernel_kfm_temporal_diff5(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src0,
    const __global uchar *src1,
    const __global uchar *src2,
    const __global uchar *src3,
    const __global uchar *src4,
    const int srcPitch,
    const int width4,
    const int height) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width4 || y >= height) return;

    const int off = y * (srcPitch / (int)sizeof(Type4)) + x;
    const Type4 a = ((const __global Type4 *)src0)[off];
    const Type4 b = ((const __global Type4 *)src1)[off];
    const Type4 c = ((const __global Type4 *)src2)[off];
    const Type4 d = ((const __global Type4 *)src3)[off];
    const Type4 e = ((const __global Type4 *)src4)[off];
    const int4 minv = min(min(kfm_to_int4(a), kfm_to_int4(b)), min(kfm_to_int4(c), min(kfm_to_int4(d), kfm_to_int4(e))));
    const int4 maxv = max(max(kfm_to_int4(a), kfm_to_int4(b)), max(kfm_to_int4(c), max(kfm_to_int4(d), kfm_to_int4(e))));
    ((__global Type4 *)dst)[y * (dstPitch / (int)sizeof(Type4)) + x] = kfm_to_type4(maxv - minv);
}

__kernel void kernel_kfm_temporal_min_diff5_3(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src0,
    const __global uchar *src1,
    const __global uchar *src2,
    const __global uchar *src3,
    const __global uchar *src4,
    const __global uchar *src5,
    const __global uchar *src6,
    const int srcPitch,
    const int width4,
    const int height) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width4 || y >= height) return;

    const int off = y * (srcPitch / (int)sizeof(Type4)) + x;
    const int4 v0 = kfm_to_int4(((const __global Type4 *)src0)[off]);
    const int4 v1 = kfm_to_int4(((const __global Type4 *)src1)[off]);
    const int4 v2 = kfm_to_int4(((const __global Type4 *)src2)[off]);
    const int4 v3 = kfm_to_int4(((const __global Type4 *)src3)[off]);
    const int4 v4 = kfm_to_int4(((const __global Type4 *)src4)[off]);
    const int4 v5 = kfm_to_int4(((const __global Type4 *)src5)[off]);
    const int4 v6 = kfm_to_int4(((const __global Type4 *)src6)[off]);

    const int4 min0 = min(min(v0, v1), min(v2, min(v3, v4)));
    const int4 max0 = max(max(v0, v1), max(v2, max(v3, v4)));
    const int4 diff0 = max0 - min0;

    const int4 min1 = min(min(v1, v2), min(v3, min(v4, v5)));
    const int4 max1 = max(max(v1, v2), max(v3, max(v4, v5)));
    const int4 diff1 = max1 - min1;

    const int4 min2 = min(min(v2, v3), min(v4, min(v5, v6)));
    const int4 max2 = max(max(v2, v3), max(v4, max(v5, v6)));
    const int4 diff2 = max2 - min2;

    ((__global Type4 *)dst)[y * (dstPitch / (int)sizeof(Type4)) + x] = kfm_to_type4(min(diff0, min(diff1, diff2)));
}

__kernel void kernel_kfm_min_frames3(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src0,
    const __global uchar *src1,
    const __global uchar *src2,
    const int srcPitch,
    const int width4,
    const int height) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width4 || y >= height) return;

    const int off = y * (srcPitch / (int)sizeof(Type4)) + x;
    const int4 minv = min(kfm_to_int4(((const __global Type4 *)src0)[off]),
                          min(kfm_to_int4(((const __global Type4 *)src1)[off]),
                              kfm_to_int4(((const __global Type4 *)src2)[off])));
    ((__global Type4 *)dst)[y * (dstPitch / (int)sizeof(Type4)) + x] = kfm_to_type4(minv);
}

__kernel void kernel_kfm_calc_combe(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width4,
    const int height,
    const int srcYOffset) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width4 || y >= height) return;

    const int srcPitch4 = srcPitch / (int)sizeof(Type4);
    const int dstPitch4 = dstPitch / (int)sizeof(Type4);
    const __global Type4 *s = (const __global Type4 *)src;
    const int yy = y + srcYOffset;
    const int yy0 = yy - 2;
    const int yy1 = yy - 1;
    const int yy3 = yy + 1;
    const int yy4 = yy + 2;
    const int4 combe = kfm_calc_combe4(s[x + yy0 * srcPitch4], s[x + yy1 * srcPitch4],
                                       s[x + yy * srcPitch4],
                                       s[x + yy3 * srcPitch4], s[x + yy4 * srcPitch4]);
    ((__global Type4 *)dst)[x + y * dstPitch4] = kfm_to_type4(clamp(combe >> 2, (int4)0, (int4)((1 << bit_depth) - 1)));
}

__kernel void kernel_kfm_merge_uv_coefs(
    __global uchar *flagY,
    const int pitchY,
    const __global uchar *flagU,
    const __global uchar *flagV,
    const int pitchUV,
    const int width,
    const int height,
    const int logUVx,
    const int logUVy) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    __global Type *fy = (__global Type *)(flagY + y * pitchY + x * (int)sizeof(Type));
    const __global Type *fu = (const __global Type *)(flagU + ((y >> logUVy) * pitchUV + (x >> logUVx) * (int)sizeof(Type)));
    const __global Type *fv = (const __global Type *)(flagV + ((y >> logUVy) * pitchUV + (x >> logUVx) * (int)sizeof(Type)));
    fy[0] = max(fy[0], max(fu[0], fv[0]));
}

__kernel void kernel_kfm_extend_coefs(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width4,
    const int height) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width4 || y >= height) return;

    const int srcPitch4 = srcPitch / (int)sizeof(Type4);
    const int dstPitch4 = dstPitch / (int)sizeof(Type4);
    const int y0 = max(y - 1, 0);
    const int y1 = min(y + 1, height - 1);
    const __global Type4 *s = (const __global Type4 *)src;
    const int4 v = max(kfm_to_int4(s[x + y0 * srcPitch4]),
                       max(kfm_to_int4(s[x + y * srcPitch4]), kfm_to_int4(s[x + y1 * srcPitch4])));
    ((__global Type4 *)dst)[x + y * dstPitch4] = kfm_to_type4(v);
}

__kernel void kernel_kfm_and_coefs(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *diff,
    const int diffPitch,
    const int width4,
    const int height,
    const float invcombe,
    const float invdiff) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width4 || y >= height) return;

    const int dstPitch4 = dstPitch / (int)sizeof(Type4);
    const int diffPitch4 = diffPitch / (int)sizeof(Type4);
    const float4 combe = clamp(convert_float4(kfm_to_int4(((__global Type4 *)dst)[x + y * dstPitch4])) * invcombe - 1.0f, -0.5f, 0.5f);
    const float4 df = clamp(convert_float4(kfm_to_int4(((const __global Type4 *)diff)[x + y * diffPitch4])) * (-invdiff) + 1.0f, -0.5f, 0.5f);
    ((__global Type4 *)dst)[x + y * dstPitch4] = kfm_to_type4_float(max(combe + df, 0.0f) * 128.0f + 0.5f);
}

__kernel void kernel_kfm_apply_uv_coefs_420(
    const __global uchar *flagY,
    const int pitchY,
    __global uchar *flagU,
    __global uchar *flagV,
    const int pitchUV,
    const int widthUV,
    const int heightUV) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= widthUV || y >= heightUV) return;

    const __global Type *fy = (const __global Type *)flagY;
    const int pitchYt = pitchY / (int)sizeof(Type);
    const int pitchUVt = pitchUV / (int)sizeof(Type);
    const int v = fy[(x * 2 + 0) + (y * 2 + 0) * pitchYt]
                + fy[(x * 2 + 1) + (y * 2 + 0) * pitchYt]
                + fy[(x * 2 + 0) + (y * 2 + 1) * pitchYt]
                + fy[(x * 2 + 1) + (y * 2 + 1) * pitchYt];
    const Type outv = (Type)((v + 2) >> 2);
    ((__global Type *)flagU)[x + y * pitchUVt] = outv;
    ((__global Type *)flagV)[x + y * pitchUVt] = outv;
}

__kernel void kernel_kfm_merge_static(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src60,
    const __global uchar *src30,
    const int srcPitch,
    const __global uchar *flag,
    const int flagPitch,
    const int width4,
    const int height) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width4 || y >= height) return;

    const int dstPitch4 = dstPitch / (int)sizeof(Type4);
    const int srcPitch4 = srcPitch / (int)sizeof(Type4);
    const int flagPitch4 = flagPitch / (int)sizeof(Type4);
    const int4 coef = kfm_to_int4(((const __global Type4 *)flag)[x + y * flagPitch4]);
    const int4 v30 = kfm_to_int4(((const __global Type4 *)src30)[x + y * srcPitch4]);
    const int4 v60 = kfm_to_int4(((const __global Type4 *)src60)[x + y * srcPitch4]);
    ((__global Type4 *)dst)[x + y * dstPitch4] = kfm_to_type4((coef * v30 + ((int4)128 - coef) * v60 + (int4)64) >> 7);
}
