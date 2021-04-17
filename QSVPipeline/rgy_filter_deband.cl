// Type
// bit_depth
// sample_mode
// yuv420
// mode_yuv
// blur_first
// block_loop_x_inner
// block_loop_y_inner
// block_loop_x_outer
// block_loop_y_outer

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

int random_range(int random, int range) {
    return ((((range << 1) + 1) * random) >> 8) - range;
}
float random_range_float(int random, float range) {
    return (range * random) * (2.0f / 256.0f) - range;
}

float get_ref(int random, int range, bool interlaced, bool ref_y) {
    if (interlaced && ref_y) {
        return (float)(random_range(random, range) & -2);
    } else {
        return random_range_float(random, range + 0.5f);
    }
}

float get_diff_abs(float a, float b) {
    return fabs(a - b);
}
float get_avg(float a, float b) {
    return (a + b) * 0.5f;
}
float get_avg4(float a, float b, float c, float d) {
    return (a + b + c + d) * 0.25f;
}
int min4(int a, int b, int c, int d) {
    return min(min(a, b), min(c, d));
}
float get_max(float a, float b) {
    return max(a, b);
}
float get_max4(float a, float b, float c, float d) {
    return max(max(a, b), max(c, d));
}

#define MODE_Y (0)
#define MODE_U (1)
#define MODE_V (2)

//threshold = (fp->track[1,2,3] << (!(sample_mode && blur_first) + 1)) * (1.0f / (1 << 10));
//range = (yuv420 && target_is_uv) ? fp->track[0] >> 1 : fp->track[0];
//dither_range = (float)dither * pow(2.0f, bit_depth-10) + 0.5
//field_mask = fp->check[2] ? -2 : -1;
__kernel void kernel_deband(
    __global uchar * __restrict__ pDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
    __global uchar4 *__restrict__ pRand, int pitchRand,
    __read_only image2d_t texSrc,
    const int range, const float dither_range, const float threshold, const int field_mask, const int mode_yuv) {
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    const int itx = get_group_id(0) * get_local_size(0) * block_loop_x_inner * block_loop_x_outer + get_local_id(0);
    const int ity = get_group_id(1) * get_local_size(1) * block_loop_y_inner * block_loop_y_outer + get_local_id(1);
    const float denormalizer = (float)(1<<bit_depth)-1.0f;
    #pragma unroll
    for (int jby = 0; jby < block_loop_y_outer; jby++) {
        #pragma unroll
        for (int jbx = 0; jbx < block_loop_x_outer; jbx++) {
            #pragma unroll
            for (int iby = 0; iby < block_loop_y_inner; iby++) {
                const int iy = ity + (jby * block_loop_y_inner + iby) * get_local_size(1);
                if (iy < dstHeight) {
                    #pragma unroll
                    for (int ibx = 0; ibx < block_loop_x_inner; ibx++) {
                        const int ix = itx + (jbx * block_loop_x_inner + ibx) * get_local_size(0);
                        if (ix < dstWidth) {
                            const int gid = iy * (pitchRand >> 2 /* pitchRand / slzeof(uchar4)*/) + ix;

                            const int y_limit = min(iy, dstHeight - iy - 1);
                            const int range_limited = min4(range, y_limit, ix, dstWidth - ix - 1);
                            const uchar4 rand = pRand[gid];
                            const int refA = random_range(rand.x, range_limited);
                            const int refB = random_range(rand.y, range_limited);

                            const float clr_center = (float)read_imagef(texSrc, sampler, (int2)(ix, iy)).x;
                            float clr_avg, clr_diff;
                            if (sample_mode == 0) {
                                const float clr_ref0 = (float)read_imagef(texSrc, sampler, (int2)(ix + refB, iy + (refA & field_mask))).x;
                                clr_avg = clr_ref0;
                                clr_diff = get_diff_abs(clr_center, clr_ref0);
                            } else if (sample_mode == 1) {
                                const float clr_ref0 = (float)read_imagef(texSrc, sampler, (int2)(ix + refB, iy + (refA & field_mask))).x;
                                const float clr_ref1 = (float)read_imagef(texSrc, sampler, (int2)(ix - refB, iy - (refA & field_mask))).x;
                                clr_avg = get_avg(clr_ref0, clr_ref1);
                                clr_diff = (blur_first) ? get_diff_abs(clr_center, clr_avg)
                                                        : get_max(get_diff_abs(clr_center, clr_ref0),
                                                                  get_diff_abs(clr_center, clr_ref1));
                            } else {
                                const float clr_ref00 = (float)read_imagef(texSrc, sampler, (int2)(ix + refB, iy + (refA & field_mask))).x;
                                const float clr_ref01 = (float)read_imagef(texSrc, sampler, (int2)(ix - refB, iy - (refA & field_mask))).x;
                                const float clr_ref10 = (float)read_imagef(texSrc, sampler, (int2)(ix + refA, iy + (refB & field_mask))).x;
                                const float clr_ref11 = (float)read_imagef(texSrc, sampler, (int2)(ix - refA, iy - (refB & field_mask))).x;
                                clr_avg = get_avg4(clr_ref00, clr_ref01, clr_ref10, clr_ref11);
                                clr_diff = (blur_first) ? get_diff_abs(clr_center, clr_avg)
                                                        : get_max4(get_diff_abs(clr_center, clr_ref00),
                                                                   get_diff_abs(clr_center, clr_ref01),
                                                                   get_diff_abs(clr_center, clr_ref10),
                                                                   get_diff_abs(clr_center, clr_ref11));
                            }
                            const float clr_out = (clr_diff < threshold) ? clr_avg : clr_center;
                            float pix_out = clr_out * denormalizer;
                            if (sample_mode != 0) {
                                const uchar randu8 = ((mode_yuv == MODE_V) ? rand.w : rand.z);
                                pix_out += random_range_float((int)(randu8), dither_range);
                            }
                            __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
                            ptr[0] = (Type)clamp(pix_out + 0.5f, 0.0f, (float)(1<<bit_depth)-1.0f);
                        }
                    }
                }
            }
        }
    }
}
