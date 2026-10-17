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
// KFM phase4 placeholder resource: UCF stage.
// These kernels intentionally keep the frame unchanged so host-side wiring can
// be staged without embedding OpenCL C in .cpp.
// -----------------------------------------------------------------------------------------

#ifndef Type
#define Type uchar
#endif

#ifndef bit_depth
#define bit_depth 8
#endif

#define KFM_CAT_(a, b) a##b
#define KFM_CAT(a, b) KFM_CAT_(a, b)
#define Type4 KFM_CAT(Type, 4)

#define KFM_UCF_COPY_KERNEL(kernel_name) \
__kernel void kernel_name( \
    __global uchar *dst, \
    const int dstPitch, \
    const __global uchar *src, \
    const int srcPitch, \
    const int width, \
    const int height) { \
    const int x = get_global_id(0); \
    const int y = get_global_id(1); \
    if (x >= width || y >= height) return; \
    const __global Type *pSrc = (const __global Type *)(src + y * srcPitch + x * (int)sizeof(Type)); \
    __global Type *pDst = (__global Type *)(dst + y * dstPitch + x * (int)sizeof(Type)); \
    pDst[0] = pSrc[0]; \
}

KFM_UCF_COPY_KERNEL(kernel_kfm_ucf)
KFM_UCF_COPY_KERNEL(kernel_kfm_ucf_noise)
KFM_UCF_COPY_KERNEL(kernel_kfm_ucf_param)
KFM_UCF_COPY_KERNEL(kernel_kfm_ucf_30)
KFM_UCF_COPY_KERNEL(kernel_kfm_ucf_24)
KFM_UCF_COPY_KERNEL(kernel_kfm_ucf_60_flag)
KFM_UCF_COPY_KERNEL(kernel_kfm_ucf_60)

#undef KFM_UCF_COPY_KERNEL

__kernel void kernel_kfm_ucf_field_crop(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width,
    const int height,
    const int srcXOffset,
    const int srcYOffset,
    const int srcYStep) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int srcX = x + srcXOffset;
    const int srcY = y * srcYStep + srcYOffset;
    const __global Type *pSrc = (const __global Type *)(src + srcY * srcPitch + srcX * (int)sizeof(Type));
    __global Type *pDst = (__global Type *)(dst + y * dstPitch + x * (int)sizeof(Type));
    pDst[0] = pSrc[0];
}

static inline int kfm_ucf_read_pix(const __global uchar *ptr, const int x, const int y, const int pitch, const int width, const int height) {
    const int ix = clamp(x, 0, width - 1);
    const int iy = clamp(y, 0, height - 1);
    const __global Type *p = (const __global Type *)(ptr + iy * pitch + ix * (int)sizeof(Type));
    return convert_int(p[0]);
}

static inline void kfm_ucf_write_pix(__global uchar *ptr, const int x, const int y, const int pitch, const int value) {
    __global Type *p = (__global Type *)(ptr + y * pitch + x * (int)sizeof(Type));
    p[0] = (Type)value;
}

__kernel void kernel_kfm_ucf_gaussresize_v(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width,
    const int height,
    const __global int *offset,
    const __global float *coeff,
    const int filterSize) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int begin = offset[y];
    const int maxValue = (1 << bit_depth) - 1;
    float result = 0.0f;
    for (int i = 0; i < filterSize; i++) {
        result += (float)kfm_ucf_read_pix(src, x, begin + i, srcPitch, width, height) * coeff[y * filterSize + i];
    }
    kfm_ucf_write_pix(dst, x, y, dstPitch, (int)(clamp(result, 0.0f, (float)maxValue) + 0.5f));
}

static inline int kfm_ucf_read_field_crop_pix(
    const __global uchar *src,
    const int x,
    const int y,
    const int srcPitch,
    const int width,
    const int height,
    const int srcXOffset,
    const int srcYOffset,
    const int srcYStep) {
    const int ix = clamp(x, 0, width - 1);
    const int iy = clamp(y, 0, height - 1);
    const int sx = ix + srcXOffset;
    const int sy = iy * srcYStep + srcYOffset;
    const __global Type *p = (const __global Type *)(src + sy * srcPitch + sx * (int)sizeof(Type));
    return convert_int(p[0]);
}

__kernel void kernel_kfm_ucf_field_crop_gaussresize_v(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width,
    const int height,
    const int srcXOffset,
    const int srcYOffset,
    const int srcYStep,
    const __global int *offset,
    const __global float *coeff,
    const int filterSize) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int begin = offset[y];
    const int maxValue = (1 << bit_depth) - 1;
    float result = 0.0f;
    for (int i = 0; i < filterSize; i++) {
        result += (float)kfm_ucf_read_field_crop_pix(src, x, begin + i, srcPitch, width, height, srcXOffset, srcYOffset, srcYStep) * coeff[y * filterSize + i];
    }
    kfm_ucf_write_pix(dst, x, y, dstPitch, (int)(clamp(result, 0.0f, (float)maxValue) + 0.5f));
}

__kernel void kernel_kfm_ucf_gaussresize_h(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width,
    const int height,
    const __global int *offset,
    const __global float *coeff,
    const int filterSize) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int begin = offset[x];
    const int maxValue = (1 << bit_depth) - 1;
    float result = 0.0f;
    for (int i = 0; i < filterSize; i++) {
        result += (float)kfm_ucf_read_pix(src, begin + i, y, srcPitch, width, height) * coeff[x * filterSize + i];
    }
    kfm_ucf_write_pix(dst, x, y, dstPitch, (int)(clamp(result, 0.0f, (float)maxValue) + 0.5f));
}

static inline int kfm_ucf_read_pix_uv_interleaved(
    const __global uchar *ptr,
    const int chromaX,
    const int channel,
    const int y,
    const int pitch,
    const int chromaWidth,
    const int height) {
    const int ix = clamp(chromaX, 0, chromaWidth - 1);
    const int iy = clamp(y, 0, height - 1);
    const __global Type *p = (const __global Type *)(ptr + iy * pitch + ((ix << 1) + channel) * (int)sizeof(Type));
    return convert_int(p[0]);
}

__kernel void kernel_kfm_ucf_gaussresize_h_uv_interleaved(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width,
    const int height,
    const int chromaWidth,
    const __global int *offset,
    const __global float *coeff,
    const int filterSize) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int chromaX = x >> 1;
    const int channel = x & 1;
    const int begin = offset[chromaX];
    const int maxValue = (1 << bit_depth) - 1;
    float result = 0.0f;
    for (int i = 0; i < filterSize; i++) {
        result += (float)kfm_ucf_read_pix_uv_interleaved(src, begin + i, channel, y, srcPitch, chromaWidth, height) * coeff[chromaX * filterSize + i];
    }
    kfm_ucf_write_pix(dst, x, y, dstPitch, (int)(clamp(result, 0.0f, (float)maxValue) + 0.5f));
}

typedef struct {
    ulong noise0;
    ulong noise1;
    ulong noiseR0;
    ulong noiseR1;
    ulong diff0;
    ulong diff1;
} KFMUcfNoiseResult;

static inline int4 kfm_ucf_to_int4(const Type4 v) {
#if bit_depth > 8
    return (int4)(v.x, v.y, v.z, v.w);
#else
    return convert_int4(v);
#endif
}

static inline ulong kfm_ucf_hsum4(const int4 v) {
    return (ulong)(v.x + v.y + v.z + v.w);
}

static inline ulong kfm_ucf_hsum4u(const uint4 v) {
    return (ulong)v.x + (ulong)v.y + (ulong)v.z + (ulong)v.w;
}

static inline int4 kfm_ucf_calc_combe4(const Type4 a, const Type4 b, const Type4 c, const Type4 d, const Type4 e) {
    const int4 diff = kfm_ucf_to_int4(a) + kfm_ucf_to_int4(c) * 4 + kfm_ucf_to_int4(e) - (kfm_ucf_to_int4(b) + kfm_ucf_to_int4(d)) * 3;
    return max(diff, -diff);
}

__kernel void kernel_kfm_ucf_analyze_noise_partial(
    __global KFMUcfNoiseResult *dst,
    const int dstOffset,
    const __global uchar *src0,
    const __global uchar *src1,
    const __global uchar *src2,
    const int srcPitch,
    const int width4,
    const int height) {
    const int groupLinear = get_group_id(0) + get_num_groups(0) * get_group_id(1);
    const int localLinear = get_local_id(0) + get_local_size(0) * get_local_id(1);
    const int localSize = get_local_size(0) * get_local_size(1);
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    __local ulong lsum0[512];
    __local ulong lsum1[512];
    __local ulong lsumR0[512];
    __local ulong lsumR1[512];

    ulong sum0 = 0;
    ulong sum1 = 0;
    ulong sumR0 = 0;
    ulong sumR1 = 0;
    if (x < width4 && y < height) {
        const int pitch4 = srcPitch / (int)sizeof(Type4);
        const int off = x + y * pitch4;
        const int4 s0 = kfm_ucf_to_int4(((const __global Type4 *)src0)[off]);
        const int4 s1 = kfm_ucf_to_int4(((const __global Type4 *)src1)[off]);
        const int4 s2 = kfm_ucf_to_int4(((const __global Type4 *)src2)[off]);
        const int neutral = 1 << max(bit_depth - 1, 0);
        sum0 = kfm_ucf_hsum4u(abs(s0 - neutral));
        sum1 = kfm_ucf_hsum4u(abs(s1 - neutral));
        sumR0 = kfm_ucf_hsum4u(abs(s1 - s0));
        sumR1 = kfm_ucf_hsum4u(abs(s2 - s1));
    }

    lsum0[localLinear] = sum0;
    lsum1[localLinear] = sum1;
    lsumR0[localLinear] = sumR0;
    lsumR1[localLinear] = sumR1;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int step = localSize >> 1; step > 0; step >>= 1) {
        if (localLinear < step) {
            lsum0[localLinear] += lsum0[localLinear + step];
            lsum1[localLinear] += lsum1[localLinear + step];
            lsumR0[localLinear] += lsumR0[localLinear + step];
            lsumR1[localLinear] += lsumR1[localLinear + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localLinear == 0) {
        KFMUcfNoiseResult result;
        result.noise0 = lsum0[0];
        result.noise1 = lsum1[0];
        result.noiseR0 = lsumR0[0];
        result.noiseR1 = lsumR1[0];
        result.diff0 = 0;
        result.diff1 = 0;
        dst[dstOffset + groupLinear] = result;
    }
}

__kernel void kernel_kfm_ucf_analyze_diff_partial(
    __global KFMUcfNoiseResult *dst,
    const int dstOffset,
    const __global uchar *src0,
    const __global uchar *src1,
    const int srcPitch,
    const int width4,
    const int height,
    const int srcYOffset) {
    const int groupLinear = get_group_id(0) + get_num_groups(0) * get_group_id(1);
    const int localLinear = get_local_id(0) + get_local_size(0) * get_local_id(1);
    const int localSize = get_local_size(0) * get_local_size(1);
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    __local ulong ldiff0[512];
    __local ulong ldiff1[512];

    ulong sum0 = 0;
    ulong sum1 = 0;
    if (x < width4 && y < height) {
        const int pitch4 = srcPitch / (int)sizeof(Type4);
        const __global Type4 *f0 = (const __global Type4 *)src0;
        const __global Type4 *f1 = (const __global Type4 *)src1;
        const int yy = y + srcYOffset;
        const Type4 a0 = f0[x + (yy - 2) * pitch4];
        const Type4 b0 = f0[x + (yy - 1) * pitch4];
        const Type4 c0 = f0[x + yy * pitch4];
        const Type4 d0 = f0[x + (yy + 1) * pitch4];
        const Type4 e0 = f0[x + (yy + 2) * pitch4];
        sum0 = kfm_ucf_hsum4(kfm_ucf_calc_combe4(a0, b0, c0, d0, e0));

        if (y & 1) {
            const Type4 a = f0[x + (yy - 2) * pitch4];
            const Type4 b = f1[x + (yy - 1) * pitch4];
            const Type4 c = f0[x + yy * pitch4];
            const Type4 d = f1[x + (yy + 1) * pitch4];
            const Type4 e = f0[x + (yy + 2) * pitch4];
            sum1 = kfm_ucf_hsum4(kfm_ucf_calc_combe4(a, b, c, d, e));
        } else {
            const Type4 a = f1[x + (yy - 2) * pitch4];
            const Type4 b = f0[x + (yy - 1) * pitch4];
            const Type4 c = f1[x + yy * pitch4];
            const Type4 d = f0[x + (yy + 1) * pitch4];
            const Type4 e = f1[x + (yy + 2) * pitch4];
            sum1 = kfm_ucf_hsum4(kfm_ucf_calc_combe4(a, b, c, d, e));
        }
    }

    ldiff0[localLinear] = sum0;
    ldiff1[localLinear] = sum1;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int step = localSize >> 1; step > 0; step >>= 1) {
        if (localLinear < step) {
            ldiff0[localLinear] += ldiff0[localLinear + step];
            ldiff1[localLinear] += ldiff1[localLinear + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localLinear == 0) {
        KFMUcfNoiseResult result;
        result.noise0 = 0;
        result.noise1 = 0;
        result.noiseR0 = 0;
        result.noiseR1 = 0;
        result.diff0 = ldiff0[0];
        result.diff1 = ldiff1[0];
        dst[dstOffset + groupLinear] = result;
    }
}

static inline int kfm_ucf_limiter(const int x, const int neutral, const int maxValue, const int nmin, const int range) {
    if (x == neutral) {
        return neutral;
    }
    if (x < neutral) {
        return (((neutral - 1 - range) < x) & (x < (neutral - nmin))) ? 0 : ((56 * neutral) >> 7);
    }
    return (((neutral + nmin) < x) & (x < (neutral + 1 + range))) ? maxValue : ((199 * neutral) >> 7);
}

__kernel void kernel_kfm_ucf_noise_limit(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const __global uchar *noise,
    const int noisePitch,
    const int width,
    const int height,
    const int nmin,
    const int range) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const __global Type *pSrc = (const __global Type *)(src + y * srcPitch + x * (int)sizeof(Type));
    const __global Type *pNoise = (const __global Type *)(noise + y * noisePitch + x * (int)sizeof(Type));
    __global Type *pDst = (__global Type *)(dst + y * dstPitch + x * (int)sizeof(Type));

    const int neutral = 1 << max(bit_depth - 1, 0);
    const int maxValue = (1 << bit_depth) - 1;
    const int v = (convert_int(pSrc[0]) - convert_int(pNoise[0]) + neutral * 2) >> 1;
    const int scaledNmin = nmin << max(bit_depth - 8, 0);
    const int scaledRange = range << max(bit_depth - 8, 0);
    pDst[0] = (Type)clamp(kfm_ucf_limiter(v, neutral, maxValue, scaledNmin, scaledRange), 0, maxValue);
}

__kernel void kernel_kfm_ucf_source_crop_noise_limit(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const __global uchar *noise,
    const int noisePitch,
    const int width,
    const int height,
    const int srcXOffset,
    const int srcYOffset,
    const int srcYStep,
    const int nmin,
    const int range) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int sx = x + srcXOffset;
    const int sy = y * srcYStep + srcYOffset;
    const __global Type *pSrc = (const __global Type *)(src + sy * srcPitch + sx * (int)sizeof(Type));
    const __global Type *pNoise = (const __global Type *)(noise + y * noisePitch + x * (int)sizeof(Type));
    __global Type *pDst = (__global Type *)(dst + y * dstPitch + x * (int)sizeof(Type));

    const int neutral = 1 << max(bit_depth - 1, 0);
    const int maxValue = (1 << bit_depth) - 1;
    const int v = (convert_int(pSrc[0]) - convert_int(pNoise[0]) + neutral * 2) >> 1;
    const int scaledNmin = nmin << max(bit_depth - 8, 0);
    const int scaledRange = range << max(bit_depth - 8, 0);
    pDst[0] = (Type)clamp(kfm_ucf_limiter(v, neutral, maxValue, scaledNmin, scaledRange), 0, maxValue);
}

// V段の整数丸めをタイル内で済ませ、H段とnoise limitまで中間画像を作らず実行する。
// 左右8画素の余裕は端で片側へ寄る9 tapとUVインターリーブにも対応する。
#if bit_depth > 8
typedef ulong KFMUcfPacked4;
#define KFM_UCF_PACK4(v) as_ulong(v)
#define KFM_UCF_UNPACK4(v) as_ushort4(v)
#else
typedef uint KFMUcfPacked4;
#define KFM_UCF_PACK4(v) as_uint(v)
#define KFM_UCF_UNPACK4(v) as_uchar4(v)
#endif

#define KFM_TO_TYPE4(v) KFM_CAT(convert_, Type4)(v)

static inline int4 kfm_ucf_limiter4(const int4 x, const int neutral, const int maxValue, const int nmin, const int range) {
    const int4 low = select((int4)((56 * neutral) >> 7), (int4)0,
        ((neutral - 1 - range) < x) & (x < (neutral - nmin)));
    const int4 high = select((int4)((199 * neutral) >> 7), (int4)maxValue,
        ((neutral + nmin) < x) & (x < (neutral + 1 + range)));
    return select(select(high, low, x < neutral), (int4)neutral, x == neutral);
}

// 1 work-itemで横4画素を扱い、V段と入出力のアドレス計算をまとめる。
__attribute__((reqd_work_group_size(32, 4, 1)))
__kernel void kernel_kfm_ucf_preprocess(
    __global uchar *dst, const int dstPitch,
    const __global uchar *src, const int srcPitch,
    const int width, const int height,
    const int srcXOffset, const int srcYOffset, const int srcYStep, const int channels,
    const __global int *offsetV, const __global float *coeffV,
    const __global int *offsetH, const __global float *coeffH,
    const int nmin, const int range) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int x = get_global_id(0) * 4;
    const int y = get_global_id(1);
    const int baseX = get_group_id(0) * 128;
    const int channelShift = channels == 2 ? 1 : 0;
    const int planeWidth = width >> channelShift;
    const int tileBegin = offsetH[baseX >> channelShift];
    const int tileEnd = min(offsetH[min((baseX + 127) >> channelShift, planeWidth - 1)] + 9, planeWidth);
    // source上のpackedロードを揃える。余分な左端画素はH段から参照しない。
    const int tileOrigin = ((tileBegin * channels + srcXOffset) & ~3) - srcXOffset;
    const int tileWidth = tileEnd * channels - tileOrigin;
    const int maxValue = (1 << bit_depth) - 1;
    // UVのhaloとアラインメント調整分（最大147）に余裕を持たせ、packed単位でV結果を保存する。
    __local Type vertical[4][152] __attribute__((aligned(16)));
    const int4 hx = min((int4)(x, x + 1, x + 2, x + 3) >> channelShift, (int4)(planeWidth - 1));
    if (y < height) {
        const int begin = offsetV[y];
        for (int tx = lx * 4; tx < tileWidth; tx += 128) {
            const int vx = tileOrigin + tx;
            float4 result = 0.0f;
            #pragma unroll
            for (int i = 0; i < 9; i++) {
                const int iy = (height >= 9) ? begin + i : min(begin + i, height - 1);
                const int sy = iy * srcYStep + srcYOffset;
                const __global Type *p = (const __global Type *)(src + sy * srcPitch) + srcXOffset;
                Type4 pixels;
                if ((srcPitch & ((int)sizeof(KFMUcfPacked4) - 1)) == 0) {
                    const __global KFMUcfPacked4 *packed = (const __global KFMUcfPacked4 *)(src + sy * srcPitch);
                    pixels = KFM_UCF_UNPACK4(packed[(vx + srcXOffset) >> 2]);
                } else {
                    pixels = vx + 3 < width ? vload4(0, p + vx)
                        : (Type4)(p[min(vx, width - 1)], p[min(vx + 1, width - 1)],
                            p[min(vx + 2, width - 1)], p[min(vx + 3, width - 1)]);
                }
                result += convert_float4(pixels) * coeffV[y * 9 + i];
            }
            ((__local KFMUcfPacked4 *)vertical[ly])[tx >> 2] = KFM_UCF_PACK4(KFM_TO_TYPE4(convert_int4(clamp(result, (float4)0.0f, (float4)maxValue) + 0.5f)));
        }
    }
    // V結果をここで同期する。
    barrier(CLK_LOCAL_MEM_FENCE);
    if (x >= width || y >= height) return;
    const int4 begin = (int4)(offsetH[hx.s0], offsetH[hx.s1], offsetH[hx.s2], offsetH[hx.s3]);
    const int4 channel = (int4)(x, x + 1, x + 2, x + 3) & (channels - 1);
    const bool contiguous = x + 3 < width && begin.s3 == begin.s0 + (3 >> channelShift);
    float4 result = 0.0f;
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        Type4 pixels;
        if (contiguous) {
            const int ix = (begin.s0 + i) * channels - tileOrigin;
            const __local KFMUcfPacked4 *packed = (const __local KFMUcfPacked4 *)vertical[ly];
            const int shift = (ix & 3) * (int)sizeof(Type) * 8;
            const KFMUcfPacked4 lo = packed[ix >> 2];
#ifdef __ENDIAN_LITTLE__
            const KFMUcfPacked4 value = shift == 0 ? lo
                : (lo >> shift) | (packed[(ix >> 2) + 1] << ((int)sizeof(KFMUcfPacked4) * 8 - shift));
#else
            const KFMUcfPacked4 value = shift == 0 ? lo
                : (lo << shift) | (packed[(ix >> 2) + 1] >> ((int)sizeof(KFMUcfPacked4) * 8 - shift));
#endif
            pixels = KFM_UCF_UNPACK4(value);
        } else {
            const int4 ix = min(begin + i, (int4)(planeWidth - 1)) * channels + channel - tileOrigin;
            pixels = (Type4)(vertical[ly][ix.s0], vertical[ly][ix.s1], vertical[ly][ix.s2], vertical[ly][ix.s3]);
        }
        const int coeffPitch = (planeWidth + 3) & ~3;
        const float4 coeff = channels == 1 ? vload4(0, coeffH + i * coeffPitch + x)
            : vload2(0, coeffH + i * coeffPitch + (x >> 1)).s0011;
        result += convert_float4(pixels) * coeff;
    }
    const int4 noise = convert_int4(clamp(result, (float4)0.0f, (float4)maxValue) + 0.5f);
    const __global Type *original = (const __global Type *)(src + (y * srcYStep + srcYOffset) * srcPitch) + srcXOffset;
    const Type4 original4 = x + 3 < width ? vload4(0, original + x)
        : (Type4)(original[min(x, width - 1)], original[min(x + 1, width - 1)],
            original[min(x + 2, width - 1)], original[min(x + 3, width - 1)]);
    const int neutral = 1 << (bit_depth - 1);
    const int4 value = (convert_int4(original4) - noise + neutral * 2) >> 1;
    const int scaledNmin = nmin << max(bit_depth - 8, 0);
    const int scaledRange = range << max(bit_depth - 8, 0);
    const Type4 limited = KFM_TO_TYPE4(clamp(kfm_ucf_limiter4(value, neutral, maxValue, scaledNmin, scaledRange), (int4)0, (int4)maxValue));
    __global Type *out = (__global Type *)(dst + y * dstPitch);
    if (x + 3 < width) {
        if ((dstPitch & ((int)sizeof(KFMUcfPacked4) - 1)) == 0) {
            ((__global KFMUcfPacked4 *)out)[x >> 2] = KFM_UCF_PACK4(limited);
        } else {
            vstore4(limited, 0, out + x);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (x + i < width) out[x + i] = limited[i];
        }
    }
}

#ifndef KFM_UCF_SUBGROUP
#define KFM_UCF_SUBGROUP 0
#endif

#if KFM_UCF_SUBGROUP && !defined(KFM_UCF_DISABLE_SUBGROUPS)
#if KFM_UCF_SUBGROUP == 2
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif
#endif

__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void kernel_kfm_ucf_analyze_partial(
    __global KFMUcfNoiseResult *dst, const int dstOffset,
    const __global uchar *noise0, const __global uchar *noise1, const __global uchar *noise2,
    const int noisePitch, const int noiseWidth4, const int noiseHeight,
    const __global uchar *src0, const __global uchar *src1,
    const int diffPitch, const int diffWidth4, const int diffHeight, const int srcYOffset) {
    const int baseX = get_global_id(0) * 4;
    const int y = get_global_id(1);
    const int localLinear = get_local_id(0) + 32 * get_local_id(1);
    const int groupLinear = get_group_id(0) + get_num_groups(0) * get_group_id(1);
    // 横16画素を担当しても部分和は256 * 16 * 6 * 65535でuintに収まる。
    __local uint partial[6][256];
    uint sums[6] = { 0, 0, 0, 0, 0, 0 };
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int x = baseX + j;
        if (x < noiseWidth4 && y < noiseHeight) {
            const int pitch4 = noisePitch / (int)sizeof(Type4);
            const int off = x + y * pitch4;
            const int4 s0 = kfm_ucf_to_int4(((const __global Type4 *)noise0)[off]);
            const int4 s1 = kfm_ucf_to_int4(((const __global Type4 *)noise1)[off]);
            const int4 s2 = kfm_ucf_to_int4(((const __global Type4 *)noise2)[off]);
            const int neutral = 1 << max(bit_depth - 1, 0);
            sums[0] += kfm_ucf_hsum4u(abs(s0 - neutral));
            sums[1] += kfm_ucf_hsum4u(abs(s1 - neutral));
            sums[2] += kfm_ucf_hsum4u(abs(s1 - s0));
            sums[3] += kfm_ucf_hsum4u(abs(s2 - s1));
        }

        if (x < diffWidth4 && y < diffHeight) {
            const int pitch4 = diffPitch / (int)sizeof(Type4);
            const __global Type4 *f0 = (const __global Type4 *)src0;
            const __global Type4 *f1 = (const __global Type4 *)src1;
            const int yy = y + srcYOffset;
            const Type4 a0 = f0[x + (yy - 2) * pitch4];
            const Type4 b0 = f0[x + (yy - 1) * pitch4];
            const Type4 c0 = f0[x + yy * pitch4];
            const Type4 d0 = f0[x + (yy + 1) * pitch4];
            const Type4 e0 = f0[x + (yy + 2) * pitch4];
            sums[4] += kfm_ucf_hsum4(kfm_ucf_calc_combe4(a0, b0, c0, d0, e0));

            if (y & 1) {
                const Type4 a = f0[x + (yy - 2) * pitch4];
                const Type4 b = f1[x + (yy - 1) * pitch4];
                const Type4 c = f0[x + yy * pitch4];
                const Type4 d = f1[x + (yy + 1) * pitch4];
                const Type4 e = f0[x + (yy + 2) * pitch4];
                sums[5] += kfm_ucf_hsum4(kfm_ucf_calc_combe4(a, b, c, d, e));
            } else {
                const Type4 a = f1[x + (yy - 2) * pitch4];
                const Type4 b = f0[x + (yy - 1) * pitch4];
                const Type4 c = f1[x + yy * pitch4];
                const Type4 d = f0[x + (yy + 1) * pitch4];
                const Type4 e = f1[x + (yy + 2) * pitch4];
                sums[5] += kfm_ucf_hsum4(kfm_ucf_calc_combe4(a, b, c, d, e));
            }
        }

    }

#if KFM_UCF_SUBGROUP && !defined(KFM_UCF_DISABLE_SUBGROUPS)
    const int subgroup = get_sub_group_id();
    const int lane = get_sub_group_local_id();
    for (int i = 0; i < 6; i++) {
        const uint sum = sub_group_reduce_add(sums[i]);
        if (lane == 0) partial[i][subgroup] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (subgroup == 0) {
        for (int i = 0; i < 6; i++) {
            uint sum = 0;
            for (int j = lane; j < get_num_sub_groups(); j += get_sub_group_size()) sum += partial[i][j];
            sums[i] = sub_group_reduce_add(sum);
        }
    }
    const bool leader = subgroup == 0 && lane == 0;
#else
    for (int i = 0; i < 6; i++) partial[i][localLinear] = sums[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = 128; step > 0; step >>= 1) {
        if (localLinear < step) {
            for (int i = 0; i < 6; i++) partial[i][localLinear] += partial[i][localLinear + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = 0; i < 6; i++) sums[i] = partial[i][0];
    const bool leader = localLinear == 0;
#endif
    if (leader) {
        KFMUcfNoiseResult result;
        result.noise0 = sums[0];
        result.noise1 = sums[1];
        result.noiseR0 = sums[2];
        result.noiseR1 = sums[3];
        result.diff0 = sums[4];
        result.diff1 = sums[5];
        dst[dstOffset + groupLinear] = result;
    }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void kernel_kfm_ucf_analyze_reduce(
    __global KFMUcfNoiseResult *dst, const __global KFMUcfNoiseResult *src,
    const int lumaCount, const int chromaCount) {
    const int plane = get_group_id(0);
    const int lid = get_local_id(0);
    const int begin = plane == 0 ? 0 : lumaCount;
    const int count = plane == 0 ? lumaCount : chromaCount;
    __local ulong partial[6][256];
    ulong sums[6] = { 0, 0, 0, 0, 0, 0 };
    for (int i = lid; i < count; i += 256) {
        const KFMUcfNoiseResult v = src[begin + i];
        sums[0] += v.noise0;
        sums[1] += v.noise1;
        sums[2] += v.noiseR0;
        sums[3] += v.noiseR1;
        sums[4] += v.diff0;
        sums[5] += v.diff1;
    }
    for (int i = 0; i < 6; i++) partial[i][lid] = sums[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = 128; step > 0; step >>= 1) {
        if (lid < step) {
            for (int i = 0; i < 6; i++) partial[i][lid] += partial[i][lid + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        KFMUcfNoiseResult result;
        result.noise0 = partial[0][0];
        result.noise1 = partial[1][0];
        result.noiseR0 = partial[2][0];
        result.noiseR1 = partial[3][0];
        result.diff0 = partial[4][0];
        result.diff1 = partial[5][0];
        dst[plane] = result;
    }
}
