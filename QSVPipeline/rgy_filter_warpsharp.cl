// Type
// bit_depth
// blur_range

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

__kernel void kernel_warpsharp_sobel(
    __global uchar *pDst, const int dstPitch,
    const __global uchar *pSrc, const int srcPitch,
    const int width, const int height,
    const int threshold) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int pixel_max = (1 << (bit_depth)) - 1;

    if (ix < width && iy < height) {
        const int x0 = max(ix - 1, 0);
        const int x1 = ix;
        const int x2 = min(ix + 1, width - 1);
        const int y0 = max(iy - 1, 0);
        const int y1 = iy;
        const int y2 = min(iy + 1, height - 1);

        // p00 01 02
        //  10 11 12
        //  20 21 22
        #define SRC(x, y) *(__global Type *)(pSrc + (y) * srcPitch + (x) * sizeof(Type))
        const int p00 = SRC(x0, y0);
        const int p01 = SRC(x1, y0);
        const int p02 = SRC(x2, y0);
        const int p10 = SRC(x0, y1);
        const int p11 = SRC(x1, y1);
        const int p12 = SRC(x2, y1);
        const int p20 = SRC(x0, y2);
        const int p21 = SRC(x1, y2);
        const int p22 = SRC(x2, y2);
        #undef SRC

        const int avg_u = (p01 + ((p00 + p02 + 1) >> 1) + 1) >> 1;
        const int avg_d = (p21 + ((p20 + p22 + 1) >> 1) + 1) >> 1;
        const int avg_l = (p10 + ((p00 + p20 + 1) >> 1) + 1) >> 1;
        const int avg_r = (p12 + ((p02 + p22 + 1) >> 1) + 1) >> 1;
        const int abs_v = abs(avg_u - avg_d);
        const int abs_h = abs(avg_l - avg_r);
        const int abs_max = max(abs_v, abs_h);

        int absolute = min(abs_v + abs_h, pixel_max);
        absolute = min(absolute + abs_max, pixel_max);
        absolute = min(min(absolute * 2, pixel_max) + absolute, pixel_max);
        absolute = min(absolute * 2, pixel_max);
        absolute = min(absolute, threshold);

        __global Type* ptr = (__global Type*)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)absolute;
    }
}

float calc_blur(float center, float avg_range[blur_range]) {
    if (blur_range == 6) {
        float avg012 = (avg_range[0] + avg_range[1]) * 0.25f + center * 0.5f;
        float avg3456 = (avg_range[2] + avg_range[3] + avg_range[4] + avg_range[5]) * 0.25f;
        float avg0123456 = (avg012 + avg3456) * 0.5f;
        return (avg012 + avg0123456) * 0.5f;
    } else if (blur_range == 2) {
        return center * 0.5f + avg_range[0] * 0.375f + avg_range[1] * 0.125f;
    }
    return 0.0f;
}


__kernel void kernel_warpsharp_blur(
    __global uchar *pDst, const int dstPitch,
    const __global uchar *pSrc, const int srcPitch,
    const int width, const int height) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int imgx = get_global_id(0);
    const int imgy = get_global_id(1);
    #define SY_SIZE (WARPSHARP_BLOCK_Y + blur_range * 2)
    #define SX_SIZE (WARPSHARP_BLOCK_X + blur_range * 2)
    __local float stmp0[SY_SIZE][SX_SIZE];
    __local float stmp1[SY_SIZE][WARPSHARP_BLOCK_X];
    for (int sy = ly, loady = imgy - blur_range; sy < SY_SIZE; sy += WARPSHARP_BLOCK_Y, loady += WARPSHARP_BLOCK_Y) {
        for (int sx = lx, loadx = imgx - blur_range; sx < SX_SIZE; sx += WARPSHARP_BLOCK_X, loadx += WARPSHARP_BLOCK_X) {
            const int y = clamp(loady, 0, height - 1);
            const int x = clamp(loadx, 0, width - 1);
            Type value = *(const __global Type*)(pSrc + y * srcPitch + x * sizeof(Type));
            stmp0[sy][sx] = (float)value;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 横方向
    for (int sy = ly; sy < SY_SIZE; sy += WARPSHARP_BLOCK_Y) {
        float avg_range[blur_range];
        #pragma unroll
        for (int i = 1; i <= blur_range; i++) {
            avg_range[i-1] = (stmp0[sy][lx + blur_range - i] + stmp0[sy][lx + blur_range + i]) * 0.5f;
        }
        stmp1[sy][lx] = calc_blur(stmp0[sy][lx + blur_range], avg_range);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 縦方向
    if (imgx < width && imgy < height) {
        float avg_range[blur_range];
        #pragma unroll
        for (int i = 1; i <= blur_range; i++) {
            avg_range[i-1] = (stmp1[ly + blur_range - i][lx] + stmp1[ly + blur_range + i][lx]) * 0.5f;
        }
        float avg = calc_blur(stmp1[ly + blur_range][lx], avg_range);

        __global Type* ptr = (__global Type*)(pDst + imgy * dstPitch + imgx * sizeof(Type));
        ptr[0] = (Type)clamp((int)(avg + 0.5f), 0, (1<<bit_depth)-1);
    }
    #undef SY_SIZE
    #undef SX_SIZE
}

__kernel void kernel_warpsharp_warp(
    __global uchar *pDst, const int dstPitch,
    __read_only image2d_t texSrc,
    const __global uchar *pEdge, const int edgePitch,
    const int width, const int height,
    const float depth) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int imgx = get_global_id(0);
    const int imgy = get_global_id(1);
    __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    if (imgx < width && imgy < height) {
        pDst  += imgy * dstPitch  + imgx * sizeof(Type);
        pEdge += imgy * edgePitch + imgx * sizeof(Type);

        const int above = *(__global Type *)((imgy == 0)          ? pEdge : pEdge - edgePitch);
        const int below = *(__global Type *)((imgy == height - 1) ? pEdge : pEdge + edgePitch);
        const int left  = *(__global Type *)((imgx == 0)          ? pEdge : pEdge - sizeof(Type));
        const int right = *(__global Type *)((imgx == width - 1)  ? pEdge : pEdge + sizeof(Type));

        float h = (float)(left - right);
        float v = (float)(above - below);

        h *= depth * ((1.0f / 256.0f) / (float)(1 << (bit_depth - 8)));
        v *= depth * ((1.0f / 256.0f) / (float)(1 << (bit_depth - 8)));

        float val = read_imagef(texSrc, sampler, (float2)(imgx + 0.5f + h, imgy + 0.5f + v)).x;
        //float val = tex2D<float>(texSrc, imgx + 0.5f + h, imgy + 0.5f + v);

        *(__global Type *)pDst = (Type)(clamp(val, 0.0f, 1.0f - 1e-6f) * ((1 << bit_depth)-1));
    }
}

__kernel void kernel_warpsharp_downscale(
    __global uchar *pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const __global uchar *pSrc, const int srcPitch) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int idstx = get_global_id(0);
    const int idsty = get_global_id(1);

    if (idstx < dstWidth && idsty < dstHeight) {
        const int isrcx = idstx << 1;
        const int isrcy = idsty << 1;
        pSrc += isrcy * srcPitch + isrcx * sizeof(Type);

        int srcY0 = *(__global Type*)(pSrc);
        int srcY1 = *(__global Type*)(pSrc + srcPitch);

        pDst += idsty * dstPitch + idstx * sizeof(Type);
        *(__global Type*)pDst = (Type)((srcY0 + srcY1 + 1) >> 1);
    }
}
