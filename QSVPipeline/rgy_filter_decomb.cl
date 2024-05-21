// TypePixel
// TypeMask
// TypeMaskVec
// BOX_X_LOG2
// BOX_Y_LOG2
// full
// blend
// DECOMB_BLOCK_IS_COMBED_X
// DECOMB_BLOCK_IS_COMBED_Y

__kernel void kernel_motion_map(
    __global uchar *__restrict__ dmaskp, __global uchar *__restrict__ fmaskp, const int dpitch,
    const __global uchar *__restrict__ srcp, const  int pitch,
    const int w, const int h,
    const float threshold, const float dthreshold) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < w && y < h) {
        TypeMask fmask = 0, dmask = 0;
        if (y > 0 && y < h - 1) {
            const float srcp0 = (float)((const __global TypePixel*)(srcp + max(y - 1, 0) * pitch + x * sizeof(TypePixel)))[0];
            const float srcp1 = (float)((const __global TypePixel*)(srcp + y * pitch + x * sizeof(TypePixel)))[0];
            const float srcp2 = (float)((const __global TypePixel*)(srcp + min(y + 1, h - 1) * pitch + x * sizeof(TypePixel)))[0];
            const float val = (float)(srcp2 - srcp1) * (float)(srcp0 - srcp1);
            if (full == 0 && val > threshold) {
                fmask = (TypeMask)0xff;
            }
            if (val > dthreshold) {
                dmask = (TypeMask)0xff;
            }
        } else {
            dmask = (TypeMask)0xff;
        }
        *(__global TypeMask *)(dmaskp + y * dpitch + x * sizeof(TypeMask)) = dmask;
        *(__global TypeMask *)(fmaskp + y * dpitch + x * sizeof(TypeMask)) = fmask;
    }
}

int is_combed_count(const TypeMask fm0, const TypeMask fm1, const TypeMask fm2) {
    return (fm0 == 0xff && fm1 == 0xff && fm2 == 0xff) ? 1 : 0;
}

int is_combed_box_x_count(const uchar4 fm0, const uchar4 fm1, const uchar4 fm2) {
    return is_combed_count(fm0.x, fm1.x, fm2.x)
         + is_combed_count(fm0.y, fm1.y, fm2.y)
         + is_combed_count(fm0.z, fm1.z, fm2.z)
         + is_combed_count(fm0.w, fm1.w, fm2.w);
}

__kernel void kernel_is_combed(
    __global int *__restrict__ isCombed,
    const __global uchar *__restrict__ fmaskp, const int dpitch,
    const int w, const int h,
    const int CT
) {
    // 1threadはx方向にBOX_X_LOG2 pixelを処理、さらにthread.xでy方向にBOX_Y_LOG2pixelを処理
    // thread.yはx方向にBOX_X_LOG2 pixelずつ処理
    const int x = (get_group_id(0) * get_local_size(0) + get_local_id(1)) << BOX_X_LOG2;
    const int y =  get_group_id(1) * get_local_size(1) + get_local_id(0);

    __local int block_result;
    if (get_local_id(0) == 0 && get_local_id(1) == 0) {
        block_result = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int count = 0;
    if (x < w && y < h) {

        const __global TypeMaskVec *fmask0 = (const __global TypeMaskVec *)(fmaskp + max(y - 1, 0) * dpitch + x * sizeof(TypeMask));
        const __global TypeMaskVec *fmask1 = (const __global TypeMaskVec *)(fmaskp + y * dpitch + x * sizeof(TypeMask));
        const __global TypeMaskVec *fmask2 = (const __global TypeMaskVec *)(fmaskp + min(y + 1, h - 1) * dpitch + x * sizeof(TypeMask));

        const TypeMaskVec fm0 = fmask0[0];
        const TypeMaskVec fm1 = fmask1[0];
        const TypeMaskVec fm2 = fmask2[0];

        count = is_combed_box_x_count(fm0, fm1, fm2);
    }

    __local int tmp[DECOMB_BLOCK_IS_COMBED_X * DECOMB_BLOCK_IS_COMBED_Y];
    const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    tmp[lid] = count;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = get_local_size(0) >> 1; offset > 0; offset >>= 1) {
        if (get_local_id(0) < offset) {
            tmp[lid] += tmp[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (get_local_id(0) == 0) {
        count = tmp[lid];
        // get_local_id(0) == 0のcountは各boxの判定結果
        // const int box = (y >> BOX_Y_LOG2) * (x >> BOX_X_LOG2);
        // まずsharedメモリ内でこのブロックの判定結果を作る
        // とりあえず1になればよいので、atomicAddで書き込む必要はなさそう
        if (count > CT) {
            block_result = 1;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // このブロックの判定結果をグローバルに書き出し
    // とりあえず1になればよいので、atomicAddで書き込む必要はなさそう
    if (get_local_id(0) == 0 && get_local_id(1) == 0 && block_result > 0) {
        isCombed[0] = 1;
    }
}

__kernel void kernel_deinterlace(
    __global char *__restrict__ ptr_dst,
    const __global char *__restrict__ ptr_src, const int pitch,
    const int w, const int h,
    const __global char *dmaskp, const int dpitch,
    const __global int *__restrict__ isCombed,
    const int uv420
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < w && y < h) {
        __global TypePixel       *ptrDst = (__global TypePixel *)(ptr_dst + y * pitch + x * sizeof(TypePixel));
        const __global TypePixel *ptrSrc0 = (const __global TypePixel *)(ptr_src + max(y - 1, 0) * pitch + x * sizeof(TypePixel));
        const __global TypePixel *ptrSrc1 = (const __global TypePixel *)(ptr_src + y * pitch + x * sizeof(TypePixel));
        const __global TypePixel *ptrSrc2 = (const __global TypePixel *)(ptr_src + min(y + 1, h - 1) * pitch + x * sizeof(TypePixel));
        const __global TypeMask  *ptrMask = (const __global TypeMask *)(dmaskp + ((uv420) ? y * 2 : y) * dpitch + ((uv420) ? x * 2 : x) * sizeof(TypeMask));

        if (blend) {
            TypePixel pix = ptrSrc1[0];
            if (full || isCombed[0]) {
                if (y == 0) {
                    pix = (TypePixel)(((int)pix + (int)ptrSrc2[0] + 1) >> 1);
                } else if (y == h - 1) {
                    pix = (TypePixel)(((int)pix + (int)ptrSrc0[0] + 1) >> 1);
                } else if (ptrMask[0]) {
                    pix = (TypePixel)((((int)pix * 2) + (int)ptrSrc0[0] + (int)ptrSrc2[0] + 3) >> 2);
                }
            }
            ptrDst[0] = pix;
        } else {
            TypePixel pix;
            if ((y & 1) && y < h - 1 && (full || isCombed[0]) && ptrMask[0]) {
                pix = (TypePixel)(((int)ptrSrc0[0] + (int)ptrSrc2[0] + 1) >> 1);
            } else {
                pix = ptrSrc1[0];
            }
            ptrDst[0] = pix;
        }
    }
}