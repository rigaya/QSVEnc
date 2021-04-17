
// Type
// bit_depth
// useExp

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

float pmd_exp(float x, float strength2, float inv_threshold2) {
    return strength2 * native_exp(-x*x * inv_threshold2);
}

float pmd(float x, float strength2, float inv_threshold2) {
    return strength2 * native_recip(1.0f + (x*x * inv_threshold2));
}

__kernel void kernel_denoise_pmd_gauss(
    __global uchar *restrict pDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
    __read_only image2d_t tSrc) {
    const float weight[5] = { 1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f };
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    if (ix < dstWidth && iy < dstHeight) {
        float sum = 0.0f;
        for (int j = 0; j < 5; j++) {
            float sum_line = 0.0f;
            #pragma unroll
            for (int i = 0; i < 5; i++) {
                sum_line += (float)read_imagef(tSrc, sampler, (int2)(ix-2+i, iy-2+j)).x * weight[i];
            }
            sum += sum_line * weight[j];
        }
        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(sum * (float)((1<<bit_depth)-1) + 0.5f);
    }
}

__kernel void kernel_denoise_pmd(
    __global uchar *restrict pDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
    __read_only image2d_t tSrc,
    __read_only image2d_t tGrf,
    const float strength2, const float inv_threshold2) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    const float denorm = (float)((1<<bit_depth)-1);
    if (ix < dstWidth && iy < dstHeight) {
        float clr   = (float)read_imagef(tSrc, sampler, (int2)(ix+0, iy+0)).x * denorm;
        float clrym = (float)read_imagef(tSrc, sampler, (int2)(ix+0, iy-1)).x * denorm;
        float clryp = (float)read_imagef(tSrc, sampler, (int2)(ix+0, iy+1)).x * denorm;
        float clrxm = (float)read_imagef(tSrc, sampler, (int2)(ix-1, iy+0)).x * denorm;
        float clrxp = (float)read_imagef(tSrc, sampler, (int2)(ix+1, iy+0)).x * denorm;
        float grf   = (float)read_imagef(tGrf, sampler, (int2)(ix+0, iy+0)).x * denorm;
        float grfym = (float)read_imagef(tGrf, sampler, (int2)(ix+0, iy-1)).x * denorm;
        float grfyp = (float)read_imagef(tGrf, sampler, (int2)(ix+0, iy+1)).x * denorm;
        float grfxm = (float)read_imagef(tGrf, sampler, (int2)(ix-1, iy+0)).x * denorm;
        float grfxp = (float)read_imagef(tGrf, sampler, (int2)(ix+1, iy+0)).x * denorm;
        clr += (useExp)
            ? (clrym - clr) * pmd_exp(grfym - grf, strength2, inv_threshold2)
            + (clryp - clr) * pmd_exp(grfyp - grf, strength2, inv_threshold2)
            + (clrxm - clr) * pmd_exp(grfxm - grf, strength2, inv_threshold2)
            + (clrxp - clr) * pmd_exp(grfxp - grf, strength2, inv_threshold2)
            : (clrym - clr) * pmd(grfym - grf, strength2, inv_threshold2)
            + (clryp - clr) * pmd(grfyp - grf, strength2, inv_threshold2)
            + (clrxm - clr) * pmd(grfxm - grf, strength2, inv_threshold2)
            + (clrxp - clr) * pmd(grfxp - grf, strength2, inv_threshold2);

        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(clamp(clr + 0.5f, 0.0f, (float)(1<<bit_depth)-0.1f));
    }
}
