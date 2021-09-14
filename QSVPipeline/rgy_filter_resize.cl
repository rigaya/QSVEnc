
// Type
// bit_depth
// radius

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
    const float ratioX, const float ratioY) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    if (ix < dstWidth && iy < dstHeight) {
        const float x = (float)ix + 0.5f;
        const float y = (float)iy + 0.5f;

        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(read_imagef(src, sampler, (int2)(x * ratioX, y * ratioY)).x * (float)((1<<bit_depth)-1));
    }
}

__kernel void kernel_resize_spline(
    __global uchar *restrict pDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
    __read_only image2d_t src,
    const float ratioX, const float ratioY,
    const float ratioDistX, const float ratioDistY,
    __global const float *restrict pgFactor) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int threadIdX = get_local_id(0);
    const int threadIdY = get_local_id(1);

    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    //重みをsharedメモリにコピー
    __local float psCopyFactor[radius][4];
    if (threadIdY == 0 && threadIdX < radius * 4) {
        ((__local float *)psCopyFactor[0])[threadIdX] = pgFactor[threadIdX];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (ix < dstWidth && iy < dstHeight) {
        //ピクセルの中心を算出してからスケール
        const float x = ((float)ix + 0.5f) * ratioX;
        const float y = ((float)iy + 0.5f) * ratioY;

        float pWeightX[radius * 2];
        float pWeightY[radius * 2];

        #pragma unroll
        for (int i = 0; i < radius * 2; i++) {
            //+0.5fはピクセル中心とするため
            const float sx = floor(x) + i - radius + 1.0f + 0.5f;
            const float sy = floor(y) + i - radius + 1.0f + 0.5f;
            //拡大ならratioDistXは1.0f、縮小ならratioの逆数(縮小側の距離に変換)
            const float dx = fabs(sx - x) * ratioDistX;
            const float dy = fabs(sy - y) * ratioDistY;
            __local float *psWeightX = psCopyFactor[min((int)dx, radius-1)];
            __local float *psWeightY = psCopyFactor[min((int)dy, radius-1)];
            //重みを計算
            float wx = psWeightX[3];
            float wy = psWeightY[3];
            wx += dx * psWeightX[2];
            wy += dy * psWeightY[2];
            const float dx2 = dx * dx;
            const float dy2 = dy * dy;
            wx += dx2 * psWeightX[1];
            wy += dy2 * psWeightY[1];
            wx += dx2 * dx * psWeightX[0];
            wy += dy2 * dy * psWeightY[0];
            pWeightX[i] = wx;
            pWeightY[i] = wy;
        }

        float weightSum = 0.0f;
        float clr = 0.0f;
        for (int j = 0; j < radius * 2; j++) {
            const float sy = floor(y) + j - radius + 1.0f + 0.5f;
            const float weightY = pWeightY[j];
            #pragma unroll
            for (int i = 0; i < radius * 2; i++) {
                const float sx = floor(x) + i - radius + 1.0f + 0.5f;
                const float weightXY = pWeightX[i] * weightY;
                clr += read_imagef(src, sampler, (int2)(sx, sy)).x * weightXY;
                weightSum += weightXY;
            }
        }

        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(clr * (float)((1<<bit_depth)-1) * native_recip(weightSum), 0.0f, (1<<bit_depth) - 0.1f);
    }
}

float lanczos_factor(float x) {
    const float pi = 3.14159265358979323846f; //M_PI;
    if (x == 0.0f) return 1.0f;
    if (x >= (float)radius) return 0.0f;
    const float pi_x = pi * x;
    return (float)radius * native_sin(pi_x) * native_sin(pi_x * (1.0f / (float)radius)) *  native_recip(pi_x * pi_x);
}

__kernel void kernel_resize_lanczos(
    __global uchar* restrict pDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
    __read_only image2d_t src,
    const float ratioX, const float ratioY, const float ratioDistX, const float ratioDistY) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    if (ix < dstWidth && iy < dstHeight) {
        //ピクセルの中心を算出してからスケール
        const float x = ((float)ix + 0.5f) * ratioX;
        const float y = ((float)iy + 0.5f) * ratioY;

        float pWeightX[radius * 2];
        float pWeightY[radius * 2];

        #pragma unroll
        for (int i = 0; i < radius * 2; i++) {
            //+0.5fはピクセル中心とするため
            const float sx = floor(x) + i - radius + 1.0f + 0.5f;
            const float sy = floor(y) + i - radius + 1.0f + 0.5f;
            //拡大ならratioDistXは1.0f、縮小ならratioの逆数(縮小側の距離に変換)
            const float dx = fabs(sx - x) * ratioDistX;
            const float dy = fabs(sy - y) * ratioDistY;
            pWeightX[i] = lanczos_factor(dx);
            pWeightY[i] = lanczos_factor(dy);
        }

        float weightSum = 0.0f;
        float clr = 0.0f;
        for (int j = 0; j < radius * 2; j++) {
            const float sy = floor(y) + j - radius + 1.0f + 0.5f;
            const float weightY = pWeightY[j];
            #pragma unroll
            for (int i = 0; i < radius * 2; i++) {
                const float sx = floor(x) + i - radius + 1.0f + 0.5f;
                const float weightXY = pWeightX[i] * weightY;
                clr += read_imagef(src, sampler, (int2)(sx, sy)).x * weightXY;
                weightSum += weightXY;
            }
        }

        __global Type* ptr = (__global Type*)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(clr * (float)((1<<bit_depth)-1) * native_recip(weightSum), 0.0f, (1<<bit_depth) - 0.1f);
    }
}