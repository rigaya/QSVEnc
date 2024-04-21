
// Type
// TmpVTypeFP16
// TmpVType4
// TmpWPType
// TmpWPType2
// TmpWPType4
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
    const int width, const int height, int4 xoffset, int4 yoffset
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < width && iy < height) {
        const __global uchar *ptr0 = pSrc + iy * srcPitch + ix * sizeof(Type);
        const Type val0 = *(const __global Type *)ptr0;

        TmpVType4 val1;
        val1.x = get_xyoffset_pix(pSrc, srcPitch, ix, iy, xoffset.x, yoffset.x, width, height);
        val1.y = get_xyoffset_pix(pSrc, srcPitch, ix, iy, xoffset.y, yoffset.y, width, height);
        val1.z = get_xyoffset_pix(pSrc, srcPitch, ix, iy, xoffset.z, yoffset.z, width, height);
        val1.w = get_xyoffset_pix(pSrc, srcPitch, ix, iy, xoffset.w, yoffset.w, width, height);

        __global TmpVType4 *ptrDst = (__global TmpVType4 *)(pDst + iy * dstPitch + ix * sizeof(TmpVType4));
        const TmpVType4 fdiff = (((TmpVType4)val0) - val1) * (TmpVType4)(1.0f / ((1<<bit_depth) - 1));
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
        TmpVType4 sum = (TmpVType4)0.0f;
        for (int j = - template_radius; j <= template_radius; j++) {
            const int srcy = clamp(iy + j, 0, height - 1);
            for (int i = - template_radius; i <= template_radius; i++) {
                const int srcx = clamp(ix + i, 0, width - 1);
                const __global TmpVType4 *ptr = (__global TmpVType4 *)(pSrc + srcy * srcPitch + srcx * sizeof(TmpVType4));
                sum += ptr[0];
            }
        }
        __global TmpVType4 *ptr = (__global TmpVType4 *)(pDst + iy * dstPitch + ix * sizeof(TmpVType4));
        ptr[0] = sum;
    }
}

TmpWPType4 tmpv4_2_tmpwp4(TmpVType4 v) {
#if TmpVTypeFP16
    return convert_float4(v);
#else
    return v;
#endif 
}

void add_reverse_side_offset(__global uchar *restrict pImgW, const int tmpPitch, const __global uchar *restrict pSrc, const int srcPitch, const int width, const int height, const int jx, const int jy, const TmpWPType weight) {
    if (0 <= jx && jx < width && 0 <= jy && jy < height) {
        __global TmpWPType2 *ptrImgW = (__global TmpWPType2 *)(pImgW + jy * tmpPitch + jx * sizeof(TmpWPType2));
        const Type pix = *(const __global Type *)(pSrc + jy * srcPitch + jx * sizeof(Type));
        TmpWPType2 weight_pix_2 = { weight * (TmpWPType)(pix * (1.0f / ((1<<bit_depth) - 1))), weight };
        ptrImgW[0] += weight_pix_2;
    }
}

__kernel void kernel_denoise_nlmeans_calc_weight(
    __global uchar *restrict pImgW0, __global uchar *restrict pImgW1, __global uchar *restrict pImgW2, __global uchar *restrict pImgW3, __global uchar *restrict pImgW4, const int tmpPitch,
    const __global uchar *restrict pV, const int vPitch,
    const __global uchar *restrict pSrc, const int srcPitch,
    const int width, const int height, const float sigma, const float inv_param_h_h,
    const int4 xoffset, const int4 yoffset
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < width && iy < height) {
        const TmpVType4 v_vt4 = *(const __global TmpVType4 *)(pV + iy * vPitch + ix * sizeof(TmpVType4));
        const TmpWPType4 v_tmpv4 = tmpv4_2_tmpwp4(v_vt4);
        const TmpWPType4 weight = native_exp(-max(v_tmpv4 - (TmpWPType4)(2.0f * sigma), (TmpWPType4)0.0f) * (TmpWPType4)inv_param_h_h);

        {
            __global TmpWPType2 *ptrImgW0 = (__global TmpWPType2 *)(pImgW0 + iy * tmpPitch + ix * sizeof(TmpWPType2));
            const Type pix = *(const __global Type *)(pSrc + iy * srcPitch + ix * sizeof(Type));
            const TmpWPType4 weight_pix = (TmpWPType4)weight * (TmpWPType4)(pix * (1.0f / ((1<<bit_depth) - 1)));
            TmpWPType2 weight_pix_2 = { weight_pix.x + weight_pix.y + weight_pix.z + weight_pix.w, weight.x + weight.y + weight.z + weight.w };
            ptrImgW0[0] += weight_pix_2;
        }
        add_reverse_side_offset(pImgW1, tmpPitch, pSrc, srcPitch, width, height, ix + xoffset.x, iy + yoffset.x, weight.x);
        add_reverse_side_offset(pImgW2, tmpPitch, pSrc, srcPitch, width, height, ix + xoffset.y, iy + yoffset.y, weight.y);
        add_reverse_side_offset(pImgW3, tmpPitch, pSrc, srcPitch, width, height, ix + xoffset.z, iy + yoffset.z, weight.y);
        add_reverse_side_offset(pImgW4, tmpPitch, pSrc, srcPitch, width, height, ix + xoffset.w, iy + yoffset.w, weight.y);
    }
}

__kernel void kernel_denoise_nlmeans_normalize(
    __global uchar *restrict pDst, const int dstPitch,
    const __global uchar *restrict pImgW0, const __global uchar *restrict pImgW1, const __global uchar *restrict pImgW2, const __global uchar *restrict pImgW3, const __global uchar *restrict pImgW4,
    const int tmpPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < width && iy < height) {
        const TmpWPType2 imgW0 = *(const __global TmpWPType2 *)(pImgW0 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW1 = *(const __global TmpWPType2 *)(pImgW1 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW2 = *(const __global TmpWPType2 *)(pImgW2 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW3 = *(const __global TmpWPType2 *)(pImgW3 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW4 = *(const __global TmpWPType2 *)(pImgW4 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const float imgW = imgW0.x + imgW1.x + imgW2.x + imgW3.x + imgW4.x;
        const float weight = imgW0.y + imgW1.y + imgW2.y + imgW3.y + imgW4.y;
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
