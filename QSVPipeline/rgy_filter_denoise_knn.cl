
// Type
// bit_depth
// knn_radius
// knn_block_x
// knn_block_y
// SRC_IMAGE

#define knn_local_x (knn_block_x + knn_radius*2)
#define knn_local_y (knn_block_y + knn_radius*2)

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

float lerpf(float v0, float v1, float t) {
    float tmp = (1.0f-t)*v0;
    return tmp + t*v1;
}

__kernel void kernel_denoise_knn(
    __global uchar *restrict pDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
#if SRC_IMAGE
    __read_only image2d_t src,
#else
    const __global uchar *restrict pSrc,
#endif
    const int srcPitch,
    const float strength, const float lerpC, const float weight_threshold, const float lerp_threshold) {
    const float knn_window_area = (float)((2 * knn_radius + 1) * (2 * knn_radius + 1));
    const float inv_knn_window_area = 1.0f / knn_window_area;
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    
#if SRC_IMAGE
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    #define LOAD(offset_x, offset_y) (float)read_imagef(src, sampler, (int2)(ix+(offset_x)+0.5f, iy+(offset_y)+0.5f)).x
    const float denorm = (float)((1<<bit_depth)-1);
#else
    const int thx = get_local_id(0);
    const int thy = get_local_id(1);
    __local Type local_buf[knn_local_y][knn_local_x];
    #pragma unroll
    for (int y = thy; y < knn_local_y; y += knn_block_y) {
        const __global uchar *ptr_src = pSrc + y * srcPitch;
        #pragma unroll
        for (int x = thx; x < knn_local_x; x += knn_block_x, ptr_src += knn_block_x * sizeof(Type)) {
            local_buf[y][x] = *(const __global Type *)ptr_src;
        }
    }
    #define LOAD(offset_x, offset_y) (float)local_buf[thy+knn_radius+(offset_y)][thx+knn_radius+(offset_x)]
    const float denorm = 1.0f;
#endif

    if (ix < dstWidth && iy < dstHeight) {
        float fCount = 0.0f;
        float sumWeights = 0.0f;
        float sum = 0.0f;
        float center = LOAD(0,0);

        #pragma unroll
        for (int i = -knn_radius; i <= knn_radius; i++) {
            #pragma unroll
            for (int j = -knn_radius; j <= knn_radius; j++) {
                float clrIJ = LOAD(i,j);
                float distanceIJ = (center - clrIJ) * (center - clrIJ);

                float weightIJ = native_exp(-(distanceIJ * strength + (i * i + j * j) * inv_knn_window_area));

                sum += clrIJ * weightIJ;

                sumWeights += weightIJ;

                fCount += (weightIJ > weight_threshold) ? inv_knn_window_area : 0;
            }
        }
        float lerpQ = (fCount > lerp_threshold) ? lerpC : 1.0f - lerpC;

        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(lerpf(sum * native_recip(sumWeights), center, lerpQ) * denorm, 0.0f, (1<<bit_depth) - 0.1f);
    }
}
