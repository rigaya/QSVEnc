
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
    const float delta = ((targetPos + 0.5f) - srcPos) * ratioClamped;
    float weight = 0.0f;
    switch (algo) {
    case WEIGHT_LANCZOS:  weight = factor_lanczos(delta); break;
    case WEIGHT_SPLINE:   weight = factor_spline(delta, psCopyFactor); break;
    case WEIGHT_BICUBIC:  weight = factor_bicubic(delta, 0.0f, 0.6f); break;
    case WEIGHT_BILINEAR: weight = factor_bilinear(delta); break;
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
