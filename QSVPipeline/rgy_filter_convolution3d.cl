
// Type
// bit_depth
// C3D_FAST
// C3D_S0
// C3D_S1
// C3D_S2
// C3D_T0
// C3D_T1
// C3D_T2

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

void convolution3d_load(
    __local float temp[C3D_BLOCK_Y + 2][C3D_BLOCK_X + 2],
    const __global uchar *restrict pFrame, const int srcPitch,
    const int lx, const int ly, const int blockimgx, const int blockimgy,
    const int width, const int height) {
#define SRCPTR(ptr, pitch, ix, iy) (const __global Type *)((ptr) + clamp((iy), 0, height-1) * (pitch) + clamp((ix), 0, width-1) * sizeof(Type))
 
    if (true)       temp[            ly][            lx] = (float)(*SRCPTR(pFrame, srcPitch, blockimgx               + lx - 1, blockimgy               + ly - 1));
    if (lx < 2)     temp[            ly][C3D_BLOCK_X+lx] = (float)(*SRCPTR(pFrame, srcPitch, blockimgx + C3D_BLOCK_X + lx - 1, blockimgy               + ly - 1));
    if (ly < 2) {   temp[C3D_BLOCK_Y+ly][            lx] = (float)(*SRCPTR(pFrame, srcPitch, blockimgx               + lx - 1, blockimgy + C3D_BLOCK_Y + ly - 1));
        if (lx < 2) temp[C3D_BLOCK_Y+ly][C3D_BLOCK_X+lx] = (float)(*SRCPTR(pFrame, srcPitch, blockimgx + C3D_BLOCK_X + lx - 1, blockimgy + C3D_BLOCK_Y + ly - 1));
    }
#undef SRCPTR
}

float convolution3d_check_threshold(
    const float orig, const float pixel, const float thresh
) {
    return (fabs(orig - pixel) <= thresh) ? orig : pixel;
}

float convolution3d_spatial(
    const __local float temp[C3D_BLOCK_Y + 2][C3D_BLOCK_X + 2],
    const int lx, const int ly,
    const float src,
    const float threshold_spatial
) {
    float val0 = 0.0f;
    float val1 = 0.0f;
    float val2 = 0.0f;
    val0 += convolution3d_check_threshold(temp[ly+0][lx+0], src, threshold_spatial) * (float)(C3D_S0 * C3D_S0);
    val0 += convolution3d_check_threshold(temp[ly+0][lx+1], src, threshold_spatial) * (float)(C3D_S0 * C3D_S1);
    val0 += convolution3d_check_threshold(temp[ly+0][lx+2], src, threshold_spatial) * (float)(C3D_S0 * C3D_S2);
    val1 += convolution3d_check_threshold(temp[ly+1][lx+0], src, threshold_spatial) * (float)(C3D_S1 * C3D_S0);
    val1 += convolution3d_check_threshold(temp[ly+1][lx+1], src, threshold_spatial) * (float)(C3D_S1 * C3D_S1);
    val1 += convolution3d_check_threshold(temp[ly+1][lx+2], src, threshold_spatial) * (float)(C3D_S1 * C3D_S2);
    val2 += convolution3d_check_threshold(temp[ly+2][lx+0], src, threshold_spatial) * (float)(C3D_S2 * C3D_S0);
    val2 += convolution3d_check_threshold(temp[ly+2][lx+1], src, threshold_spatial) * (float)(C3D_S2 * C3D_S1);
    val2 += convolution3d_check_threshold(temp[ly+2][lx+2], src, threshold_spatial) * (float)(C3D_S2 * C3D_S2);
    int stotal = C3D_S0 + C3D_S1 + C3D_S2;
    return (val0 + val1 + val2) * (1.0f / (float)(stotal * stotal));
}

__kernel void kernel_convolution3d(
    __global uchar *restrict pDst, const int dstPitch,
    const __global uchar *restrict pPrev, const __global uchar *restrict pCur, const __global uchar *restrict pNext,
    const int srcPitch, const int width, const int height,
    const float threshold_spatial, const float threshold_temporal) {
    const int lx = get_local_id(0); //スレッド数=C3D_BLOCK_X
    const int ly = get_local_id(1); //スレッド数=C3D_BLOCK_Y
    const int blockimgx = get_group_id(0) * C3D_BLOCK_X;
    const int blockimgy = get_group_id(1) * C3D_BLOCK_Y;
    const int imgx = blockimgx + lx;
    const int imgy = blockimgy + ly;
    __local float temp_src[(C3D_FAST) ? 1 : 3][C3D_BLOCK_Y + 2][C3D_BLOCK_X + 2];
#define GETPTR(ptr, pitch, ix, iy) (__global Type *)((ptr) + (iy) * (pitch) + (ix) * sizeof(Type))

    const int SRC_CUR = (C3D_FAST) ? 0 : 1;

    convolution3d_load(temp_src[SRC_CUR], pCur, srcPitch, lx, ly, blockimgx, blockimgy, width, height);
    if (!C3D_FAST) {
        convolution3d_load(temp_src[0], pPrev, srcPitch, lx, ly, blockimgx, blockimgy, width, height);
        convolution3d_load(temp_src[2], pNext, srcPitch, lx, ly, blockimgx, blockimgy, width, height);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (imgx < width && imgy < height) {
        const float src = temp_src[SRC_CUR][ly + 1][lx + 1];
        const float cur = convolution3d_spatial(temp_src[SRC_CUR], lx, ly, src, threshold_spatial);

        float prev = 0.0f;
        float next = 0.0f;
        if (C3D_FAST) {
            prev = convolution3d_check_threshold((float)(*GETPTR(pPrev, srcPitch, imgx, imgy)), src, threshold_temporal);
            next = convolution3d_check_threshold((float)(*GETPTR(pNext, srcPitch, imgx, imgy)), src, threshold_temporal);
        } else {
            prev = convolution3d_spatial(temp_src[0], lx, ly, src, threshold_temporal);
            next = convolution3d_spatial(temp_src[2], lx, ly, src, threshold_temporal);
        }
        float result = 0.0f;
        result += prev * (float)C3D_T0;
        result += cur  * (float)C3D_T1;
        result += next * (float)C3D_T2;
        result *= (1.0f / (float)(C3D_T0 + C3D_T1 + C3D_T2));

        *GETPTR(pDst, dstPitch, imgx, imgy) = (Type)clamp(result + 0.5f, 0.0f, (float)((1 << bit_depth) - 1) + 1e-6f);
    }
#undef GETPTR
}
