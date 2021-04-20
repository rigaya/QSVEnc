// Type
// Type4

__kernel void kernel_mpdecimate_block_diff(
    const __global uint8_t *restrict p0, const int p0_pitch,
    const __global uint8_t *restrict p1, const int p1_pitch,
    const int width, const int height,
    __global uint8_t *restrict pDst, const int dst_pitch) {
    const int lx = get_local_id(0); //スレッド数=MPDECIMATE_BLOCK_X
    const int ly = get_local_id(1); //スレッド数=MPDECIMATE_BLOCK_Y
    const int blockoffset_x = get_group_id(0) * get_local_size(0);
    const int blockoffset_y = get_group_id(1) * get_local_size(1);
    const int imgx = (blockoffset_x + lx) * 8;
    const int imgy = (blockoffset_y + ly);

    int diff = 0;
    if (imgx < width && imgy < height) {
        p0 += imgy * p0_pitch + imgx * sizeof(Type);
        p1 += imgy * p1_pitch + imgx * sizeof(Type);
        __global Type4 *ptrp0 = (__global Type4 *)p0;
        __global Type4 *ptrp1 = (__global Type4 *)p1;
        {
            Type4 pix0 = ptrp0[0];
            Type4 pix1 = ptrp1[0];
            diff += abs_diff(pix0.x, pix1.x);
            if (imgx + 1 < width) diff += abs_diff(pix0.y, pix1.y);
            if (imgx + 2 < width) diff += abs_diff(pix0.z, pix1.z);
            if (imgx + 3 < width) diff += abs_diff(pix0.w, pix1.w);
        }
        if (imgx + 4 < width) {
            Type4 pix0 = ptrp0[1];
            Type4 pix1 = ptrp1[1];
            diff += abs_diff(pix0.x, pix1.x);
            if (imgx + 5 < width) diff += abs_diff(pix0.y, pix1.y);
            if (imgx + 6 < width) diff += abs_diff(pix0.z, pix1.z);
            if (imgx + 7 < width) diff += abs_diff(pix0.w, pix1.w);
        }
    }

    __local int tmp[MPDECIMATE_BLOCK_Y][MPDECIMATE_BLOCK_X +1];
    tmp[ly][lx] = diff;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ly == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            diff += tmp[i][lx];
        }
        const int block8x8X = blockoffset_x + lx;
        const int block8x8Y = get_group_id(1);
        pDst += block8x8Y * dst_pitch + block8x8X * sizeof(diff);
        *(__global int *)pDst = diff;
    }
}
