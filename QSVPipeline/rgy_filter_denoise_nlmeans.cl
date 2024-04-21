
// Type
// TmpVTypeFP16
// TmpVType2
// TmpWPType
// TmpWPType2
// bit_depth

// support_radius
// template_radius
// shared_radius = max(support_radius, template_radius)

// NLEANS_BLOCK_X
// NLEANS_BLOCK_Y

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

Type get_xyoffset_pix(
    const __global uchar *restrict pSrc, const int srcPitch,
    const int ix, const int iy, const int xoffset, const int yoffset, const int width, const int height) {
    const int jx = clamp(ix + xoffset, 0, width - 1);
    const int jy = clamp(iy + yoffset, 0, height - 1);
    const __global uchar *ptr1 = pSrc + jy * srcPitch + jx * sizeof(Type);
    return *(const __global Type *)ptr1;
}

__kernel void kernel_calc_diff_square(
    __global uchar *restrict pDst, const int dstPitch,
    const __global uchar *restrict pSrc, const int srcPitch,
    const int width, const int height, int2 xoffset, int2 yoffset
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < width && iy < height) {
        const __global uchar *ptr0 = pSrc + iy * srcPitch + ix * sizeof(Type);
        const Type val0 = *(const __global Type *)ptr0;

        TmpVType2 val1;
        val1.x = get_xyoffset_pix(pSrc, srcPitch, ix, iy, xoffset.x, yoffset.x, width, height);
        val1.y = get_xyoffset_pix(pSrc, srcPitch, ix, iy, xoffset.y, yoffset.y, width, height);
        //val1.z = get_xyoffset_pix(pSrc, srcPitch, ix, iy, xoffset.z, yoffset.z, width, height);
        //val1.w = get_xyoffset_pix(pSrc, srcPitch, ix, iy, xoffset.w, yoffset.w, width, height);

        __global TmpVType2 *ptrDst = (__global TmpVType2 *)(pDst + iy * dstPitch + ix * sizeof(TmpVType2));
        const TmpVType2 fdiff = (((TmpVType2)val0) - val1) * (TmpVType2)(1.0f / ((1<<bit_depth) - 1));
        ptrDst[0] = fdiff * fdiff;
    }
}

__kernel void kernel_denoise_nlmeans_calc_v(
    __global uchar *restrict pDst, const int dstPitch,
    const __global uchar *restrict pSrc, const int srcPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < width && iy < height) {
        TmpVType2 sum = (TmpVType2)0.0f;
        for (int j = - template_radius; j <= template_radius; j++) {
            const int srcy = clamp(iy + j, 0, height - 1);
            for (int i = - template_radius; i <= template_radius; i++) {
                const int srcx = clamp(ix + i, 0, width - 1);
                const __global TmpVType2 *ptr = (__global TmpVType2 *)(pSrc + srcy * srcPitch + srcx * sizeof(TmpVType2));
                sum += ptr[0];
            }
        }
        __global TmpVType2 *ptr = (__global TmpVType2 *)(pDst + iy * dstPitch + ix * sizeof(TmpVType2));
        ptr[0] = sum;
    }
}

TmpWPType2 tmpv2_2_tmpwp2(TmpVType2 v) {
#if TmpVTypeFP16
    return convert_float2(v);
#else
    return v;
#endif 
}

__kernel void kernel_denoise_nlmeans_calc_weight(
    __global uchar *restrict pImgW0, __global uchar *restrict pImgW1, __global uchar *restrict pImgW2,
    __global uchar *restrict pWeight0, __global uchar *restrict pWeight1, __global uchar *restrict pWeight2, const int tmpPitch,
    const __global uchar *restrict pV, const int vPitch,
    const __global uchar *restrict pSrc, const int srcPitch,
    const int width, const int height, const float sigma, const float inv_param_h_h,
    const int2 xoffset, const int2 yoffset
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < width && iy < height) {
        const TmpVType2 v = *(const __global TmpVType2 *)(pV + iy * vPitch + ix * sizeof(TmpVType2));
        const TmpVType2 weightvt2 = native_exp(-max(v - (TmpVType2)(2.0f * sigma), (TmpVType2)0.0f) * (TmpVType2)inv_param_h_h);
        const TmpWPType2 weight = tmpv2_2_tmpwp2(weightvt2);

        {
            __global TmpWPType *ptrImgW0   = (__global TmpWPType *)(pImgW0   + iy * tmpPitch + ix * sizeof(TmpWPType));
            __global TmpWPType *ptrWeight0 = (__global TmpWPType *)(pWeight0 + iy * tmpPitch + ix * sizeof(TmpWPType));
            const Type pix = *(const __global Type *)(pSrc + iy * srcPitch + ix * sizeof(Type));
            const TmpWPType2 weight_pix = (TmpWPType2)weight * (TmpWPType2)(pix * (1.0f / ((1<<bit_depth) - 1)));
            ptrImgW0[0] += weight_pix.x + weight_pix.y;
            ptrWeight0[0] += weight.x + weight.y;
        }
        const int jx1 = ix + xoffset.x;
        const int jy1 = iy + yoffset.x;
        const int jx2 = ix + xoffset.y;
        const int jy2 = iy + yoffset.y;
        if (0 <= jx1 && jx1 < width && 0 <= jy1 && jy1 < height) {
            __global TmpWPType *ptrImgW1   = (__global TmpWPType *)(pImgW1   + jy1 * tmpPitch + jx1 * sizeof(TmpWPType));
            __global TmpWPType *ptrWeight1 = (__global TmpWPType *)(pWeight1 + jy1 * tmpPitch + jx1 * sizeof(TmpWPType));
            const Type pix = *(const __global Type *)(pSrc + jy1 * srcPitch + jx1 * sizeof(Type));
            ptrImgW1[0] += weight.x * (TmpWPType)(pix * (1.0f / ((1<<bit_depth) - 1)));
            ptrWeight1[0] += weight.x;
        }
        if (0 <= jx2 && jx2 < width && 0 <= jy2 && jy2 < height) {
            __global TmpWPType *ptrImgW2   = (__global TmpWPType *)(pImgW2   + jy2 * tmpPitch + jx2 * sizeof(TmpWPType));
            __global TmpWPType *ptrWeight2 = (__global TmpWPType *)(pWeight2 + jy2 * tmpPitch + jx2 * sizeof(TmpWPType));
            const Type pix = *(const __global Type *)(pSrc + jy2 * srcPitch + jx2 * sizeof(Type));
            ptrImgW2[0] += weight.y * (TmpWPType)(pix * (1.0f / ((1<<bit_depth) - 1)));
            ptrWeight2[0] += weight.y;
        }
    }
}

__kernel void kernel_denoise_nlmeans_normalize(
    __global uchar *restrict pDst, const int dstPitch,
    const __global uchar *restrict pImgW0, const __global uchar *restrict pImgW1, const __global uchar *restrict pImgW2,
    const __global uchar *restrict pWeight0, const __global uchar *restrict pWeight1, const __global uchar *restrict pWeight2, const int tmpPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < width && iy < height) {
        const __global TmpWPType *ptrImgW0   = (const __global TmpWPType *)(pImgW0   + iy * tmpPitch + ix * sizeof(TmpWPType));
        const __global TmpWPType *ptrWeight0 = (const __global TmpWPType *)(pWeight0 + iy * tmpPitch + ix * sizeof(TmpWPType));
        const __global TmpWPType *ptrImgW1   = (const __global TmpWPType *)(pImgW1   + iy * tmpPitch + ix * sizeof(TmpWPType));
        const __global TmpWPType *ptrWeight1 = (const __global TmpWPType *)(pWeight1 + iy * tmpPitch + ix * sizeof(TmpWPType));
        const __global TmpWPType *ptrImgW2   = (const __global TmpWPType *)(pImgW2   + iy * tmpPitch + ix * sizeof(TmpWPType));
        const __global TmpWPType *ptrWeight2 = (const __global TmpWPType *)(pWeight2 + iy * tmpPitch + ix * sizeof(TmpWPType));
        const float imgW = ptrImgW0[0] + ptrImgW1[0] + ptrImgW2[0];
        const float weight = ptrWeight0[0] + ptrWeight1[0] + ptrWeight2[0];
        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(imgW * native_recip(weight) * ((1<<bit_depth) - 1), 0.0f, (1<<bit_depth) - 0.1f);
    }
}

__kernel void kernel_denoise_nlmeans(
    __global uchar *restrict pDst, const int dstPitch,
    const __global uchar *restrict pSrc, const int srcPitch,
    const int width, const int height,
    const float sigma, const float inv_param_h_h) {
    const int thx = get_local_id(0);
    const int thy = get_local_id(1);
    const int global_bx = get_group_id(0) * NLEANS_BLOCK_X;
    const int global_by = get_group_id(1) * NLEANS_BLOCK_Y;
    const int ix = global_bx + thx;
    const int iy = global_by + thy;

    __local Type shared[NLEANS_BLOCK_Y + 2 * shared_radius][NLEANS_BLOCK_X + 2 * shared_radius];

    for (int j = thy; j < NLEANS_BLOCK_Y + 2 * shared_radius; j += NLEANS_BLOCK_Y) {
        for (int i = thx; i < NLEANS_BLOCK_X + 2 * shared_radius; i += NLEANS_BLOCK_X) {
            int srcx = clamp(global_bx + i - shared_radius, 0, width - 1);
            int srcy = clamp(global_by + j - shared_radius, 0, height - 1);
            Type val =  *(__global Type*)(pSrc + srcy * srcPitch + srcx * sizeof(Type));
            shared[j][i] = val;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (ix < width && iy < height) {

        const int shared_x = shared_radius + thx;
        const int shared_y = shared_radius + thx;
        
        float sumWeights = 0.0f;
        float sum = 0.0f;
        for (int j = - support_radius; j <= support_radius; j++) {
            for (int i = - support_radius; i <= support_radius; i++) {
                const int tx = shared_x + i;
                const int ty = shared_y + j;
                float dist = 0.0f;
                for (int tj = - template_radius; tj <= template_radius; tj++) {
                    for (int ti = - template_radius; ti <= template_radius; ti++) {
                        const Type diff = shared[ty + tj][tx + ti] - shared[shared_y + tj][shared_x + ti];
                        const float fdiff = (float)diff * (1.0f / ((1<<bit_depth) - 1));
                        dist += fdiff * fdiff;
                    }
                }
                const float arg = -max(dist - 2.0f * sigma * sigma, 0.0f) * inv_param_h_h;
                const float weight = native_exp(arg);

                sumWeights += weight;
                sum += shared[shared_y][shared_x] * weight;
            }
        }
        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(sum * native_recip(sumWeights) * ((1<<bit_depth) - 1), 0.0f, (1<<bit_depth) - 0.1f);
    }
}
