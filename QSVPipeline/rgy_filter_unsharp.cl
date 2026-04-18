// Type
// radius
// bit_depth

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif
#define SHARED_SIZE ((2 * radius + 1))
#define GAUSSH_BUFF_MAX ((float)(1<<16))

#define RGY_FLT_EPS (1e-6f)

// Pass 1: horizontal 1-D Gaussian blur → ushort intermediate buffer.
__kernel void kernel_unsharp_h(
    __global ushort *__restrict__ pDst, const int dstPitch,
    const int width, const int height,
    __read_only image2d_t texSrc,
    const __global float *__restrict__ pGaussWeight
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    __local float shared[SHARED_SIZE];
    const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    if (lid < SHARED_SIZE) shared[lid] = pGaussWeight[lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (ix < width && iy < height) {
        float sum = 0.0f;
        #pragma unroll
        for (int i = -radius; i <= radius; i++) {
            sum += read_imagef(texSrc, sampler, (int2)(ix + i, iy)).x * shared[i + radius];
        }
        __global ushort *ptr = (__global ushort *)((__global uchar *)pDst + iy * dstPitch + ix * sizeof(ushort));
        ptr[0] = (ushort)(clamp(sum, 0.0f, 1.0f - RGY_FLT_EPS) * GAUSSH_BUFF_MAX);
    }
}

// Pass 2: vertical 1-D Gaussian blur from ushort intermediate + unsharp application.
__kernel void kernel_unsharp(
    __global uchar *__restrict__ pDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
    __read_only image2d_t texSrc,
    const __global ushort *__restrict__ pGaussBufH, const int gaussBufPitch,
    const __global float *__restrict__ pGaussWeight,
    const float weight, const float threshold
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    __local float shared[SHARED_SIZE];
    const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    if (lid < SHARED_SIZE) shared[lid] = pGaussWeight[lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (ix < dstWidth && iy < dstHeight) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = -radius; j <= radius; j++) {
            const int srcy = clamp(iy + j, 0, dstHeight - 1);
            const ushort hval = *(const __global ushort *)((const __global uchar *)pGaussBufH + srcy * gaussBufPitch + ix * sizeof(ushort));
            sum += hval * (1.0f/GAUSSH_BUFF_MAX) * shared[j + radius];
        }
        float center = read_imagef(texSrc, sampler, (int2)(ix, iy)).x;
        const float diff = center - sum;
        if (fabs(diff) >= threshold) center += weight * diff;
        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(clamp(center, 0.0f, 1.0f - RGY_FLT_EPS) * (1 << (bit_depth)));
    }
}
