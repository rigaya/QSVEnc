// Type
// Type2
// Type4
// DTB_X
// DTB_Y
// DECIMATE_BLOCK_MAX (32)
// SUB_GROUP_SIZE

#ifndef SUB_GROUP_SIZE
#define SUB_GROUP_SIZE (0)
#endif

#if SUB_GROUP_SIZE > 0
#define DTB_BLOCK_SUM_SHARED_SIZE (DTB_X * DTB_Y / SUB_GROUP_SIZE)
#else
#define DTB_BLOCK_SUM_SHARED_SIZE (DTB_X * DTB_Y)
#endif

typedef int BOOL;

int block_sum_int(int val, __local int *shared) {
    const int lid = get_local_id(1) * DTB_X + get_local_id(0);
#if SUB_GROUP_SIZE > 0
    const int lane    = get_sub_group_local_id();
    const int warp_id = get_sub_group_id();
    
	int value_count = DTB_X * DTB_Y;
	for (;;) {
		val = sub_group_reduce_add(val);
		if (lane == 0) shared[warp_id] = val;
		barrier(CLK_LOCAL_MEM_FENCE);
		value_count /= SUB_GROUP_SIZE;
		if (value_count <= 1) break;
        val = (lid < value_count) ? shared[lane] : 0;
	}
#else
    shared[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = DTB_Y * DTB_X >> 1; offset > 0; offset >>= 1) {
        if (lid < offset) {
            shared[lid] += shared[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif
    int ret = 0;
    if (lid == 0) {
        ret = shared[lid];
    }
    return ret;
}

int block_max_int(int val, __local int *shared) {
    const int lid = get_local_id(1) * DTB_X + get_local_id(0);
#if SUB_GROUP_SIZE > 0
    const int lane    = get_sub_group_local_id();
    const int warp_id = get_sub_group_id();
    
	int value_count = DTB_X * DTB_Y;
	for (;;) {
		val = sub_group_reduce_max(val);
		if (lane == 0) shared[warp_id] = val;
		barrier(CLK_LOCAL_MEM_FENCE);
		value_count /= SUB_GROUP_SIZE;
		if (value_count <= 1) break;
        val = (lid < value_count) ? shared[lane] : 0;
	}
#else
    shared[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = DTB_Y * DTB_X >> 1; offset > 0; offset >>= 1) {
        if (lid < offset) {
            shared[lid] += shared[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif
    int ret = 0;
    if (lid == 0) {
        ret = shared[lid];
    }
    return ret;
}


int func_diff_block1(
    const __global uchar *restrict p0, const int p0_pitch,
    const __global uchar *restrict p1, const int p1_pitch,
    const int block_half_y,
    const int width, const int height,
    const int imgx, const int imgy) {
    int diff = 0;
    for (int y = 0; y < block_half_y; y++) {
        if (imgx < width && imgy + y < height) {
            Type pix0 = *(__global Type *)(p0 + (imgy + y) * p0_pitch + imgx * sizeof(Type));
            Type pix1 = *(__global Type *)(p1 + (imgy + y) * p1_pitch + imgx * sizeof(Type));
            diff += abs_diff(pix0, pix1);
        }
    }
    return diff;
}

int func_diff_block2(
    const __global uchar *restrict p0, const int p0_pitch,
    const __global uchar *restrict p1, const int p1_pitch,
    const int block_half_y,
    const int width, const int height,
    const int imgx, const int imgy) {
    int diff = 0;
    for (int y = 0; y < block_half_y; y++) {
        if (imgx < width && imgy + y < height) {
            Type2 pix0 = *(__global Type2 *)(p0 + (imgy + y) * p0_pitch + imgx * sizeof(Type));
            Type2 pix1 = *(__global Type2 *)(p1 + (imgy + y) * p1_pitch + imgx * sizeof(Type));
            diff += abs_diff(pix0.x, pix1.x);
            if (imgx + 1 < width) diff += abs_diff(pix0.y, pix1.y);
        }
    }
    return diff;
}

int func_diff_block4(
    const __global uchar *restrict p0, const int p0_pitch,
    const __global uchar *restrict p1, const int p1_pitch,
    const int block_half_y,
    const int width, const int height,
    const int imgx, const int imgy,
    const int block_half_x) {
    int diff = 0;
    for (int y = 0; y < block_half_y; y++) {
        if (imgx < width && imgy + y < height) {
            Type4 pix0 = *(Type4 *)(p0 + (imgy + y)* p0_pitch + imgx * sizeof(Type));
            Type4 pix1 = *(Type4 *)(p1 + (imgy + y)* p1_pitch + imgx * sizeof(Type));
            diff += abs_diff(pix0.x, pix1.x);
            if (imgx + 1 < width && block_half_x >= 2) diff += abs_diff(pix0.y, pix1.y);
            if (imgx + 2 < width && block_half_x >= 3) diff += abs_diff(pix0.z, pix1.z);
            if (imgx + 3 < width && block_half_x >= 4) diff += abs_diff(pix0.w, pix1.w);
            if (block_half_x > 4) {
                pix0 = *(__global Type4 *)(p0 + (imgy + y) * p0_pitch + (imgx + 4) * sizeof(Type));
                pix1 = *(__global Type4 *)(p1 + (imgy + y) * p1_pitch + (imgx + 4) * sizeof(Type));
                if (imgx + 4 < width) diff += abs_diff(pix0.x, pix1.x);
                if (imgx + 5 < width) diff += abs_diff(pix0.y, pix1.y);
                if (imgx + 6 < width) diff += abs_diff(pix0.z, pix1.z);
                if (imgx + 7 < width) diff += abs_diff(pix0.w, pix1.w);
            }
            if (block_half_x > 8) {
                pix0 = *(__global Type4 *)(p0 + (imgy + y) * p0_pitch + (imgx + 8) * sizeof(Type));
                pix1 = *(__global Type4 *)(p1 + (imgy + y) * p1_pitch + (imgx + 8) * sizeof(Type));
                if (imgx +  8 < width) diff += abs_diff(pix0.x, pix1.x);
                if (imgx +  9 < width) diff += abs_diff(pix0.y, pix1.y);
                if (imgx + 10 < width) diff += abs_diff(pix0.z, pix1.z);
                if (imgx + 11 < width) diff += abs_diff(pix0.w, pix1.w);

                pix0 = *(__global Type4 *)(p0 + (imgy + y) * p0_pitch + (imgx + 12) * sizeof(Type));
                pix1 = *(__global Type4 *)(p1 + (imgy + y) * p1_pitch + (imgx + 12) * sizeof(Type));
                if (imgx + 12 < width) diff += abs_diff(pix0.x, pix1.x);
                if (imgx + 13 < width) diff += abs_diff(pix0.y, pix1.y);
                if (imgx + 14 < width) diff += abs_diff(pix0.z, pix1.z);
                if (imgx + 15 < width) diff += abs_diff(pix0.w, pix1.w);
            }
        }
    }
    return diff;
}

void func_calc_sum_max(__local int diff[DTB_Y+1][DTB_X+1], __global int2 *restrict pDst, const BOOL firstPlane, __local int *tmp) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    int sum = diff[ly][lx];
    int b2x2 = diff[ly+0][lx+0]
             + diff[ly+0][lx+1]
             + diff[ly+1][lx+0]
             + diff[ly+1][lx+1];
    sum  = block_sum_int(sum, tmp);
    b2x2 = block_max_int(b2x2, tmp);
    const int lid = ly * DTB_X + lx;
    if (lid == 0) {
        const int gid = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        int2 ret = pDst[gid];
        ret.x = sum;
        ret.y = b2x2;
        if (firstPlane) {
            int2 dst = pDst[gid];
            ret.x += dst.x;
            ret.y = max(ret.y, dst.y);
        }
        pDst[gid] = ret;
    }
}

//block_half_x = 1の実装
//集計までをGPUで行う
__kernel void kernel_block_diff2_1(
    const __global uchar *restrict p0, const int p0_pitch,
    const __global uchar *restrict p1, const int p1_pitch,
    const int width, const int height,
    const int block_half_y, const BOOL firstPlane,
    __global int2 *restrict pDst) {
    const int block_half_x = 1;
    const int lx = get_local_id(0); //スレッド数=DTB_X
    const int ly = get_local_id(1); //スレッド数=DTB_Y
    const int imgx = (get_group_id(0) * DTB_X /* get_local_size(0)*/ + lx) * block_half_x;
    const int imgy = (get_group_id(1) * DTB_Y /* get_local_size(1)*/ + ly) * block_half_y;

    __local int diff[DTB_Y + 1][DTB_X + 1];
    diff[ly][lx] = func_diff_block1(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, imgx, imgy);
    if (ly == 0) {
        int loady = (get_group_id(1) + 1) * DTB_Y * block_half_y;
        diff[DTB_Y][lx] = func_diff_block1(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, imgx, loady);
    }
    {
        const int targety = ly * DTB_X + lx;
        if (targety <= DTB_Y) {
            const int loadx = (get_group_id(0) + 1) * DTB_X * block_half_x;
            const int loady = (get_group_id(1) * DTB_Y /* get_local_size(1)*/ + targety) * block_half_y;
            diff[targety][DTB_X] = func_diff_block1(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, loadx, loady);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __local int tmp[DTB_BLOCK_SUM_SHARED_SIZE];
    func_calc_sum_max(diff, pDst, firstPlane, tmp);
}

//block_half_x = 2の実装
//集計までをGPUで行う
__kernel void kernel_block_diff2_2(
    const __global uchar *restrict p0, const int p0_pitch,
    const __global uchar *restrict p1, const int p1_pitch,
    const int width, const int height,
    const int block_half_y, const BOOL firstPlane,
    __global int2 *restrict pDst) {
    const int block_half_x = 2;
    const int lx = get_local_id(0); //スレッド数=DTB_X
    const int ly = get_local_id(1); //スレッド数=DTB_Y
    const int imgx = (get_group_id(0) * DTB_X /* get_local_size(0)*/ + lx) * block_half_x;
    const int imgy = (get_group_id(1) * DTB_Y /* get_local_size(1)*/ + ly) * block_half_y;

    __local int diff[DTB_Y + 1][DTB_X + 1];
    diff[ly][lx] = func_diff_block2(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, imgx, imgy);
    if (ly == 0) {
        int loady = (get_group_id(1) + 1) * DTB_Y * block_half_y;
        diff[DTB_Y][lx] = func_diff_block2(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, imgx, loady);
    }
    {
        const int targety = ly * DTB_X + lx;
        if (targety <= DTB_Y) {
            const int loadx = (get_group_id(0) + 1) * DTB_X * block_half_x;
            const int loady = (get_group_id(1) * DTB_Y /* get_local_size(1)*/ + targety) * block_half_y;
            diff[targety][DTB_X] = func_diff_block2(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, loadx, loady);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __local int tmp[DTB_BLOCK_SUM_SHARED_SIZE];
    func_calc_sum_max(diff, pDst, firstPlane, tmp);
}

//block_half_x = 4, 8, 16の実装
//集計までをGPUで行う
void kernel_block_diff2_4_8_16(
    const __global uchar *restrict p0, const int p0_pitch,
    const __global uchar *restrict p1, const int p1_pitch,
    const int width, const int height,
    const int block_half_x, const int block_half_y, const BOOL firstPlane,
    __global int2 *restrict pDst,
     __local int diff[DTB_Y+1][DTB_X+1],
     __local int tmp[DTB_BLOCK_SUM_SHARED_SIZE]) {
    const int lx = get_local_id(0); //スレッド数=DTB_X
    const int ly = get_local_id(1); //スレッド数=DTB_Y
    const int imgx = (get_group_id(0) * DTB_X /* get_local_size(0)*/ + lx) * block_half_x;
    const int imgy = (get_group_id(1) * DTB_Y /* get_local_size(1)*/ + ly) * block_half_y;

    diff[ly][lx] = func_diff_block4(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, imgx, imgy, block_half_x);
    if (ly == 0) {
        int loady = (get_group_id(1) + 1) * DTB_Y * block_half_y;
        diff[DTB_Y][lx] = func_diff_block4(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, imgx, loady, block_half_x);
    }
    {
        const int targety = ly * DTB_X + lx;
        if (targety <= DTB_Y) {
            const int loadx = (get_group_id(0) + 1) * DTB_X * block_half_x;
            const int loady = (get_group_id(1) * DTB_Y /* get_local_size(1)*/ + targety) * block_half_y;
            diff[targety][DTB_X] = func_diff_block4(p0, p0_pitch, p1, p1_pitch, block_half_y, width, height, loadx, loady, block_half_x);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    func_calc_sum_max(diff, pDst, firstPlane, tmp);
}

__kernel void kernel_block_diff2_4(
    const __global uchar *restrict p0, const int p0_pitch,
    const __global uchar *restrict p1, const int p1_pitch,
    const int width, const int height,
    const int block_half_y, const BOOL firstPlane,
    __global int2 *restrict pDst) {
    const int block_half_x = 4;
    __local int diff[DTB_Y+1][DTB_X+1];
    __local int tmp[DTB_BLOCK_SUM_SHARED_SIZE];
    kernel_block_diff2_4_8_16(p0, p0_pitch, p1, p1_pitch, width, height, block_half_x, block_half_y, firstPlane, pDst, diff, tmp);
}

__kernel void kernel_block_diff2_8(
    const __global uchar *restrict p0, const int p0_pitch,
    const __global uchar *restrict p1, const int p1_pitch,
    const int width, const int height,
    const int block_half_y, const BOOL firstPlane,
    __global int2 *restrict pDst) {
    const int block_half_x = 8;
    __local int diff[DTB_Y+1][DTB_X+1];
    __local int tmp[DTB_BLOCK_SUM_SHARED_SIZE];
    kernel_block_diff2_4_8_16(p0, p0_pitch, p1, p1_pitch, width, height, block_half_x, block_half_y, firstPlane, pDst, diff, tmp);
}

__kernel void kernel_block_diff2_16(
    const __global uchar *restrict p0, const int p0_pitch,
    const __global uchar *restrict p1, const int p1_pitch,
    const int width, const int height,
    const int block_half_y, const BOOL firstPlane,
    __global int2 *restrict pDst) {
    const int block_half_x = 16;
    __local int diff[DTB_Y+1][DTB_X+1];
    __local int tmp[DTB_BLOCK_SUM_SHARED_SIZE];
    kernel_block_diff2_4_8_16(p0, p0_pitch, p1, p1_pitch, width, height, block_half_x, block_half_y, firstPlane, pDst, diff, tmp);
}


__kernel void kernel_block_diff(
    const __global uchar *restrict p0, const int p0_pitch,
    const __global uchar *restrict p1, const int p1_pitch,
    const int width, const int height,
    const int dummy, const BOOL firstPlane,
    __global int *restrict pDst) {
    const int lx = get_local_id(0); //スレッド数=SSIM_BLOCK_X
    const int ly = get_local_id(1); //スレッド数=SSIM_BLOCK_Y
    const int blockoffset_x = get_group_id(0) * get_local_size(0);
    const int blockoffset_y = get_group_id(1) * get_local_size(1);
    const int imgx = (blockoffset_x + lx) * 4;
    const int imgy = (blockoffset_y + ly);

    int diff = 0;
    if (imgx < width && imgy < height) {
        p0 += imgy * p0_pitch + imgx * sizeof(Type);
        p1 += imgy * p1_pitch + imgx * sizeof(Type);
        Type4 pix0 = *(__global Type4 *)p0;
        Type4 pix1 = *(__global Type4 *)p1;
        diff += abs_diff(pix0.x, pix1.x);
        if (imgx + 1 < width) diff += abs_diff(pix0.y, pix1.y);
        if (imgx + 2 < width) diff += abs_diff(pix0.z, pix1.z);
        if (imgx + 3 < width) diff += abs_diff(pix0.w, pix1.w);
    }

    __local int tmp[DECIMATE_BLOCK_MAX * DECIMATE_BLOCK_MAX];
    diff = block_sum_int(diff, tmp);

    const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    if (lid == 0) {
        const int gid = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        if (firstPlane) {
            diff += pDst[gid];
        }
        pDst[gid] = diff;
    }
}
