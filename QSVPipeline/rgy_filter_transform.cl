// TypePixel
// TypePixel4
// flipX
// flipY
// TRASNPOSE_TILE_DIM
// TRASNPOSE_BLOCK_DIM
// FLIP_BLOCK_DIM

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

#define ALIGN(x,align) (((x)+((align)-1))&(~((align)-1)))

__kernel void kernel_transpose_plane(
    __global uchar *__restrict__ pDst,
    const int dstPitch,
    const int dstWidth,  // = srcHeight
    const int dstHeight, // = srcWidth
    const __global uchar *__restrict__ pSrc,
    const int srcPitch
) {
    __local TypePixel stemp[TRASNPOSE_TILE_DIM][TRASNPOSE_TILE_DIM + 4];
    const int srcHeight = dstWidth;
    const int srcWidth = dstHeight;
    const int dstBlockX = get_group_id(0);
    const int dstBlockY = get_group_id(1);
    const int srcBlockX = (flipX) ? get_num_groups(1) - 1 - get_group_id(1) : get_group_id(1);
    const int srcBlockY = (flipY) ? get_num_groups(0) - 1 - get_group_id(0) : get_group_id(0);
    const int offsetX = (flipX) ? srcWidth - ALIGN(srcWidth, TRASNPOSE_TILE_DIM) : 0;
    const int offsetY = (flipY) ? srcHeight - ALIGN(srcHeight, TRASNPOSE_TILE_DIM) : 0;
    {
        for (int j = get_local_id(1); j < TRASNPOSE_TILE_DIM; j += TRASNPOSE_BLOCK_DIM) {
            const int srcX = srcBlockX * TRASNPOSE_TILE_DIM + get_local_id(0) * 4 + offsetX;
            const int srcY = srcBlockY * TRASNPOSE_TILE_DIM + j + offsetY;
            TypePixel4 val = { 128, 128, 128, 128 };
            if (0 <= srcX && srcX < srcWidth && 0 <= srcY && srcY < srcHeight) {
                const __global TypePixel4 *ptr_src = (const __global TypePixel4 *)(pSrc + srcY * srcPitch + srcX * sizeof(TypePixel));
                if ((offsetX & 3) == 0) {
                    val = ptr_src[0];
                } else {
                    const __global TypePixel *ptr_src_elem = (const __global TypePixel *)ptr_src;
                    val.x = ptr_src_elem[0];
                    val.y = ptr_src_elem[1];
                    val.z = ptr_src_elem[2];
                    val.w = ptr_src_elem[3];
                }
            }
            *(__local TypePixel4 *)&stemp[j][get_local_id(0) * 4] = val;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        for (int j = get_local_id(1); j < TRASNPOSE_TILE_DIM; j += TRASNPOSE_BLOCK_DIM) {
            const int dstX = dstBlockX * TRASNPOSE_TILE_DIM + get_local_id(0) * 4;
            const int dstY = dstBlockY * TRASNPOSE_TILE_DIM + j;
            const int tmpY = (flipX) ? TRASNPOSE_TILE_DIM - 1 - j : j;
            if (dstX < dstWidth && dstY < dstHeight) {
                TypePixel4 val = { 0, 0, 0, 0 };
                if (flipY) {
                    val.x = stemp[TRASNPOSE_TILE_DIM - (get_local_id(0) + 1) * 4 + 3][tmpY];
                    val.y = stemp[TRASNPOSE_TILE_DIM - (get_local_id(0) + 1) * 4 + 2][tmpY];
                    val.z = stemp[TRASNPOSE_TILE_DIM - (get_local_id(0) + 1) * 4 + 1][tmpY];
                    val.w = stemp[TRASNPOSE_TILE_DIM - (get_local_id(0) + 1) * 4 + 0][tmpY];
                } else {
                    val.x = stemp[get_local_id(0) * 4 + 0][tmpY];
                    val.y = stemp[get_local_id(0) * 4 + 1][tmpY];
                    val.z = stemp[get_local_id(0) * 4 + 2][tmpY];
                    val.w = stemp[get_local_id(0) * 4 + 3][tmpY];
                }
                __global TypePixel4 *ptr_dst = (__global TypePixel4 *)(pDst + dstY * dstPitch + dstX * sizeof(TypePixel));
                *ptr_dst = val;
            }
        }
    }
};

__kernel void kernel_flip_plane(
    __global uchar *__restrict__ pDst,
    const int dstPitch,
    const int dstWidth,
    const int dstHeight,
    const __global uchar *__restrict__ pSrc,
    const int srcPitch
) {
    __local TypePixel stemp[FLIP_BLOCK_DIM][FLIP_BLOCK_DIM * 4];
    const int dstBlockX = get_group_id(0);
    const int dstBlockY = get_group_id(1);
    const int srcBlockX = (flipX) ? get_num_groups(0) - 1 - get_group_id(0) : get_group_id(0);
    const int srcBlockY = (flipY) ? get_num_groups(1) - 1 - get_group_id(1) : get_group_id(1);
    const int offsetX = (flipX) ? dstWidth - ALIGN(dstWidth, FLIP_BLOCK_DIM * 4) : 0;
    const int offsetY = (flipY) ? dstHeight - ALIGN(dstHeight, FLIP_BLOCK_DIM) : 0;
    const int srcX = (srcBlockX * FLIP_BLOCK_DIM + get_local_id(0)) * 4 + offsetX;
    const int srcY = srcBlockY * FLIP_BLOCK_DIM + get_local_id(1) + offsetY;

    TypePixel4 val = { 128, 128, 128, 128 };
    if (0 <= srcX && srcX < dstWidth && 0 <= srcY && srcY < dstHeight) {
        const __global TypePixel4 *ptr_src = (const __global TypePixel4 *)(pSrc + srcY * srcPitch + srcX * sizeof(TypePixel));
        if ((offsetX & 3) == 0) {
            val = ptr_src[0];
        } else {
            const __global TypePixel *ptr_src_elem = (const __global TypePixel *)ptr_src;
            val.x = ptr_src_elem[0];
            val.y = ptr_src_elem[1];
            val.z = ptr_src_elem[2];
            val.w = ptr_src_elem[3];
        }
    }
    *(__local TypePixel4 *)&stemp[get_local_id(1)][get_local_id(0) * 4] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    const int dstX = (dstBlockX * FLIP_BLOCK_DIM + get_local_id(0)) * 4;
    const int dstY = dstBlockY * FLIP_BLOCK_DIM + get_local_id(1);
    const int tmpY = (flipY) ? FLIP_BLOCK_DIM - 1 - get_local_id(1) : get_local_id(1);
    val = *(__local TypePixel4 *)&stemp[tmpY][get_local_id(0) * 4];
    if (flipX) {
        TypePixel4 val2 = *(__local TypePixel4 *)&stemp[tmpY][FLIP_BLOCK_DIM * 4 - (get_local_id(0) + 1) * 4];
        val.x = val2.w;
        val.y = val2.z;
        val.z = val2.y;
        val.w = val2.x;
    } else {
        val = *(__local TypePixel4 *)&stemp[tmpY][get_local_id(0) * 4];
    }
    if (dstX < dstWidth && dstY < dstHeight) {
        __global TypePixel4 *ptr_dst = (__global TypePixel4 *)(pDst + dstY * dstPitch + dstX * sizeof(TypePixel));
        *ptr_dst = val;
    }
};
