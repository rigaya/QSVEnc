
// Type
// bit_depth
// useExp
// pmd_block_x
// pmd_block_y

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

float pmd_exp(float x, float strength2, float inv_threshold2) {
    return strength2 * native_exp(-x*x * inv_threshold2);
}

float pmd(float x, float strength2, float inv_threshold2) {
    return strength2 * native_recip(1.0f + (x*x * inv_threshold2));
}

// Gaussian is separable: G2D(x,y) = G1D(x) * G1D(y). This fused kernel does both
// 1D passes in SLM — one cooperative global load, then two tight SLM loops —
// dropping 25 sampler fetches/pixel to ~1-2 loads/thread + 10 SLM taps/pixel.
__attribute__((reqd_work_group_size(pmd_block_x, pmd_block_y, 1)))
__kernel void kernel_denoise_pmd_gauss(
    __global uchar *restrict pDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
    __read_only image2d_t tSrc) {
    const float weight[5] = { 1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f };
    const int thx = get_local_id(0);
    const int thy = get_local_id(1);
    const int bx  = get_group_id(0) * pmd_block_x;
    const int by  = get_group_id(1) * pmd_block_y;
    const int ix  = bx + thx;
    const int iy  = by + thy;
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    // Halo-padded source tile.
    __local float tile[pmd_block_y + 4][pmd_block_x + 4];
    for (int j = thy; j < pmd_block_y + 4; j += pmd_block_y) {
        for (int i = thx; i < pmd_block_x + 4; i += pmd_block_x) {
            const int srcx = clamp(bx + i - 2, 0, dstWidth - 1);
            const int srcy = clamp(by + j - 2, 0, dstHeight - 1);
            tile[j][i] = (float)read_imagef(tSrc, sampler, (int2)(srcx, srcy)).x;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Horizontal pass into a halo-height intermediate, each column owned by thx.
    __local float hpass[pmd_block_y + 4][pmd_block_x];
    for (int j = thy; j < pmd_block_y + 4; j += pmd_block_y) {
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            s += tile[j][thx + i] * weight[i];
        }
        hpass[j][thx] = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Vertical pass: 5-tap weighted sum from hpass → output.
    if (ix < dstWidth && iy < dstHeight) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 5; j++) {
            sum += hpass[thy + j][thx] * weight[j];
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
