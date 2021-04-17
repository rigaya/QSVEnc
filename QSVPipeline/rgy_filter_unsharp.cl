// Type
// radius
// bit_depth

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

#define SHARED_SIZE ((2 * radius + 1) * (2 * radius + 1))

#define RGY_FLT_EPS (1e-6f)

__kernel void kernel_unsharp(
    __global uchar *__restrict__ pDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
    __read_only image2d_t texSrc,
    const __global float *__restrict__ pGaussWeight,
    const float weight, const float threshold) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    const int local_id = get_local_id(1) * get_local_size(0) + get_local_id(0);
    __local float shared[SHARED_SIZE];
    if (local_id < SHARED_SIZE) {
        shared[local_id] = pGaussWeight[local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (ix < dstWidth && iy < dstHeight) {
        float sum = 0.0f;
        float center = (float)read_imagef(texSrc, sampler, (int2)(ix, iy)).x;
        __local float *ptr_weight = shared;

        for (int j = -radius; j <= radius; j++) {
            #pragma unroll
            for (int i = -radius; i <= radius; i++) {
                sum += (float)read_imagef(texSrc, sampler, (int2)(ix+i, iy+j)).x * ptr_weight[0];
                ptr_weight++;
            }
        }

        const float diff = center - sum;
        if (fabs(diff) >= threshold) {
            center += weight * diff;
        }

        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(clamp(center, 0.0f, 1.0f - RGY_FLT_EPS) * (1 << (bit_depth)));
    }
}
