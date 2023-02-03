
// Type
// bit_depth
// radius
// block_x
// block_y
// algo
// WEIGHT_SPLINE
// WEIGHT_LANCZOS
// shared_weightXdim
// shared_weightYdim

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

float factor_spline(const float x, __local const float *psCopyFactor) {
    __local const float *psWeight = psCopyFactor + min((int)x, radius - 1) * 4;
    //重みを計算
    float w = psWeight[3];
    w += x * psWeight[2];
    const float x2 = x * x;
    w += x2 * psWeight[1];
    w += x2 * x * psWeight[0];
    return w;
}

void calc_weight(
    __local float *pWeight, const float srcPos, const int srcFirst, const int srcEnd,
    const float ratioClamped, __local const float *psCopyFactor) {
    __local float *pW = pWeight;
    for (int i = srcFirst; i <= srcEnd; i++, pW++) {
        const float delta = ((i + 0.5f) - srcPos) * ratioClamped;
        float weight = 0.0f;
        switch (algo) {
        case WEIGHT_LANCZOS: weight = factor_lanczos(delta); break;
        case WEIGHT_SPLINE:  weight = factor_spline(fabs(delta), psCopyFactor);
        default:
            break;
        }
        pW[0] = weight;
    }
}

__kernel void kernel_resize(
    __global uchar *restrict pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    __global const uchar *restrict pSrc, const int srcPitch, const int srcWidth, const int srcHeight,
    const float ratioX, const float ratioY, __global const float *restrict pgFactor
) {
    __local float weightXshared[shared_weightXdim * block_x];
    __local float weightYshared[shared_weightYdim * block_y];
    __local float psCopyFactor[radius * 4];
    
    const int threadIdX = get_local_id(0);
    const int threadIdY = get_local_id(1);

    const float ratioInvX = 1.0f / ratioX;
    const float ratioClampedX = min(ratioX, 1.0f);
    const float srcWindowX = radius / ratioClampedX;

    const float ratioInvY = 1.0f / ratioY;
    const float ratioClampedY = min(ratioY, 1.0f);
    const float srcWindowY = radius / ratioClampedY;

    if (algo == WEIGHT_SPLINE) {
        if (threadIdY == 0) {
            if (threadIdX < radius * 4) {
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
        calc_weight(weightXshared + threadIdX * shared_weightXdim, srcX, srcFirstX, srcEndX, ratioClampedX, psCopyFactor);

        if (threadIdX < block_y) {
            // threadIdY==0のスレッドが、y方向の重みをそれぞれ計算してsharedメモリに書き込み
            const int thready = threadIdX;
            const int dstY = get_group_id(1) * block_y + thready;
            const float srcY = ((float)(dstY + 0.5f)) * ratioInvY;
            const int srcFirstY = max(0, (int)floor(srcY - srcWindowY));
            const int srcEndY = min(srcHeight - 1, (int)ceil(srcY + srcWindowY));
            calc_weight(weightYshared + thready * shared_weightYdim, srcY, srcFirstY, srcEndY, ratioClampedY, psCopyFactor);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int ix = get_group_id(0) * block_x + threadIdX;
    const int iy = get_group_id(1) * block_y + threadIdY;

    if (ix < dstWidth && iy < dstHeight) {
        //ピクセルの中心を算出してからスケール
        const float x = ((float)ix + 0.5f) * ratioX;
        const float y = ((float)iy + 0.5f) * ratioY;

        const float srcX = ((float)(ix + 0.5f)) * ratioInvX;
        const int srcFirstX = max(0, (int)floor(srcX - srcWindowX));
        const int srcEndX = min(srcWidth - 1, (int)ceil(srcX + srcWindowX));
        __local const float *weightX = weightXshared + threadIdX * shared_weightXdim;

        const float srcY = ((float)(iy + 0.5f)) * ratioInvY;
        const int srcFirstY = max(0, (int)floor(srcY - srcWindowY));
        const int srcEndY = min(srcHeight - 1, (int)ceil(srcY + srcWindowY));
        __local const float *weightY = weightYshared + threadIdY * shared_weightYdim;

        const __global uchar *srcLine = pSrc + srcFirstY * srcPitch + srcFirstX * sizeof(Type);
        float clr = 0.0f;
        float sumWeight = 0.0f;
        for (int j = srcFirstY; j <= srcEndY; j++, weightY++, srcLine += srcPitch) {
            const float wy = weightY[0];
            __local const float *pwx = weightX;
            __global const Type *srcPtr = (__global const Type*)srcLine;
            for (int i = srcFirstX; i <= srcEndX; i++, pwx++, srcPtr++) {
                const float wx = pwx[0];
                clr += srcPtr[0] * wx * wy;
                sumWeight += wx * wy;
            }
        }
        clr /= sumWeight;

        __global Type* ptr = (__global Type*)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(clr, 0.0f, (1 << bit_depth) - 0.1f);
    }
}
