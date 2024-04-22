
// Type
// TmpVTypeFP16
// TmpVType8
// TmpWPType
// TmpWPType2
// TmpWPType8
// bit_depth

// search_radius
// template_radius
// shared_radius = max(search_radius, template_radius)

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
    const int width, const int height, int8 xoffset, int8 yoffset
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < width && iy < height) {
        const __global uchar *ptr0 = pSrc + iy * srcPitch + ix * sizeof(Type);
        const Type val0 = *(const __global Type *)ptr0;

        TmpVType8 val1;
        val1.s0 = get_xyoffset_pix(pSrc, srcPitch, ix, iy, xoffset.s0, yoffset.s0, width, height);
        val1.s1 = get_xyoffset_pix(pSrc, srcPitch, ix, iy, xoffset.s1, yoffset.s1, width, height);
        val1.s2 = get_xyoffset_pix(pSrc, srcPitch, ix, iy, xoffset.s2, yoffset.s2, width, height);
        val1.s3 = get_xyoffset_pix(pSrc, srcPitch, ix, iy, xoffset.s3, yoffset.s3, width, height);
        val1.s4 = get_xyoffset_pix(pSrc, srcPitch, ix, iy, xoffset.s4, yoffset.s4, width, height);
        val1.s5 = get_xyoffset_pix(pSrc, srcPitch, ix, iy, xoffset.s5, yoffset.s5, width, height);
        val1.s6 = get_xyoffset_pix(pSrc, srcPitch, ix, iy, xoffset.s6, yoffset.s6, width, height);
        val1.s7 = get_xyoffset_pix(pSrc, srcPitch, ix, iy, xoffset.s7, yoffset.s7, width, height);

        __global TmpVType8 *ptrDst = (__global TmpVType8 *)(pDst + iy * dstPitch + ix * sizeof(TmpVType8));
        const TmpVType8 fdiff = (((TmpVType8)val0) - val1) * (TmpVType8)(1.0f / ((1<<bit_depth) - 1));
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
        TmpVType8 sum = (TmpVType8)0.0f;
        for (int j = - template_radius; j <= template_radius; j++) {
            const int srcy = clamp(iy + j, 0, height - 1);
            for (int i = - template_radius; i <= template_radius; i++) {
                const int srcx = clamp(ix + i, 0, width - 1);
                const __global TmpVType8 *ptr = (__global TmpVType8 *)(pSrc + srcy * srcPitch + srcx * sizeof(TmpVType8));
                sum += ptr[0];
            }
        }
        __global TmpVType8 *ptr = (__global TmpVType8 *)(pDst + iy * dstPitch + ix * sizeof(TmpVType8));
        ptr[0] = sum;
    }
}

TmpWPType8 tmpv8_2_tmpwp8(TmpVType8 v) {
#if TmpVTypeFP16
    return convert_float8(v);
#else
    return v;
#endif 
}


void add_reverse_side_offset(__global uchar *restrict pImgW, const int tmpPitch, const int width, const int height, const int jx, const int jy, const TmpWPType pixNormalized, const TmpWPType weight) {
    if (0 <= jx && jx < width && 0 <= jy && jy < height) {
        __global TmpWPType2 *ptrImgW = (__global TmpWPType2 *)(pImgW + jy * tmpPitch + jx * sizeof(TmpWPType2));
        TmpWPType2 weight_pix_2 = { weight * pixNormalized, weight };
        ptrImgW[0] += weight_pix_2;
    }
}

TmpWPType getSrcPixXYOffset(const __global uchar *restrict pSrc, const int srcPitch, const int width, const int height, const int ix, const int iy, const int xoffset, const int yoffset) {
    const Type pix = *(const __global Type *)(pSrc + clamp(iy+yoffset, 0, height-1) * srcPitch + clamp(ix+xoffset,0,width-1) * sizeof(Type));
    return pix * (1.0f / ((1<<bit_depth) - 1));
}

TmpWPType8 getSrcPixXYOffset8(const __global uchar *restrict pSrc, const int srcPitch, const int width, const int height, const int ix, const int iy, const int8 xoffset, const int8 yoffset) {
    TmpWPType8 pix8;
    pix8.s0 = getSrcPixXYOffset(pSrc, srcPitch, width, height, ix, iy, xoffset.s0, yoffset.s0);
    pix8.s1 = getSrcPixXYOffset(pSrc, srcPitch, width, height, ix, iy, xoffset.s1, yoffset.s1);
    pix8.s2 = getSrcPixXYOffset(pSrc, srcPitch, width, height, ix, iy, xoffset.s2, yoffset.s2);
    pix8.s3 = getSrcPixXYOffset(pSrc, srcPitch, width, height, ix, iy, xoffset.s3, yoffset.s3);
    pix8.s4 = getSrcPixXYOffset(pSrc, srcPitch, width, height, ix, iy, xoffset.s4, yoffset.s4);
    pix8.s5 = getSrcPixXYOffset(pSrc, srcPitch, width, height, ix, iy, xoffset.s5, yoffset.s5);
    pix8.s6 = getSrcPixXYOffset(pSrc, srcPitch, width, height, ix, iy, xoffset.s6, yoffset.s6);
    pix8.s7 = getSrcPixXYOffset(pSrc, srcPitch, width, height, ix, iy, xoffset.s7, yoffset.s7);
    return pix8;
}

__kernel void kernel_denoise_nlmeans_calc_weight(
    __global uchar *restrict pImgW0,
    __global uchar *restrict pImgW1, __global uchar *restrict pImgW2, __global uchar *restrict pImgW3, __global uchar *restrict pImgW4,
    __global uchar *restrict pImgW5, __global uchar *restrict pImgW6, __global uchar *restrict pImgW7, __global uchar *restrict pImgW8,
    const int tmpPitch,
    const __global uchar *restrict pV, const int vPitch,
    const __global uchar *restrict pSrc, const int srcPitch,
    const int width, const int height, const float sigma, const float inv_param_h_h,
    const int8 xoffset, const int8 yoffset
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < width && iy < height) {
        const TmpVType8 v_vt8 = *(const __global TmpVType8 *)(pV + iy * vPitch + ix * sizeof(TmpVType8));
        const TmpWPType8 v_tmpv8 = tmpv8_2_tmpwp8(v_vt8); // expを使う前にfp32に変換
        const TmpWPType8 weight = native_exp(-max(v_tmpv8 - (TmpWPType8)(2.0f * sigma), (TmpWPType8)0.0f) * (TmpWPType8)inv_param_h_h);

        // 自分のほうはここですべて同じバッファ(ptrImgW0)に足し込んでしまう
        {
            __global TmpWPType2 *ptrImgW0 = (__global TmpWPType2 *)(pImgW0 + iy * tmpPitch + ix * sizeof(TmpWPType2));
            TmpWPType8 pix8 = getSrcPixXYOffset8(pSrc, srcPitch, width, height, ix, iy, xoffset, yoffset);
            TmpWPType8 weight_pix8 = weight * pix8;
            TmpWPType2 weight_pix_2 = {
                weight_pix8.s0 + weight_pix8.s1 + weight_pix8.s2 + weight_pix8.s3 + weight_pix8.s4 + weight_pix8.s5 + weight_pix8.s6 + weight_pix8.s7,
                weight.s0 + weight.s1 + weight.s2 + weight.s3 + weight.s4 + weight.s5 + weight.s6 + weight.s7
            };
            ptrImgW0[0] += weight_pix_2;
        }
        // 反対側は衝突を避けるため、別々に足し込む
        const Type pix = *(const __global Type *)(pSrc + iy * srcPitch + ix * sizeof(Type));
        const TmpWPType pixNormalized = pix * (1.0f / ((1<<bit_depth) - 1));
        add_reverse_side_offset(pImgW1, tmpPitch, width, height, ix + xoffset.s0, iy + yoffset.s0, pixNormalized, weight.s0);
        add_reverse_side_offset(pImgW2, tmpPitch, width, height, ix + xoffset.s1, iy + yoffset.s1, pixNormalized, weight.s1);
        add_reverse_side_offset(pImgW3, tmpPitch, width, height, ix + xoffset.s2, iy + yoffset.s2, pixNormalized, weight.s2);
        add_reverse_side_offset(pImgW4, tmpPitch, width, height, ix + xoffset.s3, iy + yoffset.s3, pixNormalized, weight.s3);
        add_reverse_side_offset(pImgW5, tmpPitch, width, height, ix + xoffset.s4, iy + yoffset.s4, pixNormalized, weight.s4);
        add_reverse_side_offset(pImgW6, tmpPitch, width, height, ix + xoffset.s5, iy + yoffset.s5, pixNormalized, weight.s5);
        add_reverse_side_offset(pImgW7, tmpPitch, width, height, ix + xoffset.s6, iy + yoffset.s6, pixNormalized, weight.s6);
        add_reverse_side_offset(pImgW8, tmpPitch, width, height, ix + xoffset.s7, iy + yoffset.s7, pixNormalized, weight.s7);
    }
}

__kernel void kernel_denoise_nlmeans_normalize(
    __global uchar *restrict pDst, const int dstPitch,
    const __global uchar *restrict pImgW0,
    const __global uchar *restrict pImgW1, const __global uchar *restrict pImgW2, const __global uchar *restrict pImgW3, const __global uchar *restrict pImgW4,
    const __global uchar *restrict pImgW5, const __global uchar *restrict pImgW6, const __global uchar *restrict pImgW7, const __global uchar *restrict pImgW8,
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
        const TmpWPType2 imgW5 = *(const __global TmpWPType2 *)(pImgW5 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW6 = *(const __global TmpWPType2 *)(pImgW6 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW7 = *(const __global TmpWPType2 *)(pImgW7 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const TmpWPType2 imgW8 = *(const __global TmpWPType2 *)(pImgW8 + iy * tmpPitch + ix * sizeof(TmpWPType2));
        const float imgW = imgW0.x + imgW1.x + imgW2.x + imgW3.x + imgW4.x + imgW5.x + imgW6.x + imgW7.x + imgW8.x;
        const float weight = imgW0.y + imgW1.y + imgW2.y + imgW3.y + imgW4.y + imgW5.y + imgW6.y + imgW7.y + imgW8.y;
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
        for (int j = - search_radius; j <= search_radius; j++) {
            for (int i = - search_radius; i <= search_radius; i++) {
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
