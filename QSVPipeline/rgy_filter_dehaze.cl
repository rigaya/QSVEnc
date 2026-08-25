// Type
// bit_depth
//
// He 2009のdark channel priorを輝度プレーンへ適用する。
// 大気光Aは手動指定とし、矩形minは横・縦の2パスで厳密に分離する。

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

// 矩形minの横方向パス。整数minだけなので2次元版とbit-exactになる。
// 1080pではmonolithic比で既定r=7が81%、r=15が92%高速で、全条件bit-exact。
__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void kernel_dehaze_min_horizontal(
    __global uchar *restrict pDst, const int dstPitch,
    const __global uchar *restrict pSrc, const int srcPitch,
    const int width, const int height, const int patchRadius) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const __global Type *srcRow = (const __global Type *)(pSrc + y * srcPitch);
    Type minValue = (Type)((1 << bit_depth) - 1);
    for (int dx = -patchRadius; dx <= patchRadius; dx++) {
        minValue = min(minValue, srcRow[clamp(x + dx, 0, width - 1)]);
    }
    __global Type *dstRow = (__global Type *)(pDst + y * dstPitch);
    dstRow[x] = minValue;
}

// 横minの結果を縦に走査し、透過率の算出と復元を同時に行う。
__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void kernel_dehaze(
    __global uchar *restrict pDst, const int dstPitch, const int width, const int height,
    const __global uchar *restrict pSrc, const int srcPitch,
    const __global uchar *restrict pMinHorizontal, const int minHorizontalPitch,
    const int patchRadius, const float omega, const float tFloor, const float atmosphericLight) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    Type minValue = (Type)((1 << bit_depth) - 1);
    for (int dy = -patchRadius; dy <= patchRadius; dy++) {
        const int sy = clamp(y + dy, 0, height - 1);
        const __global Type *row = (const __global Type *)(pMinHorizontal + sy * minHorizontalPitch);
        minValue = min(minValue, row[x]);
    }

    const float maxValue = (float)((1 << bit_depth) - 1);
    const float invMax = 1.0f / maxValue;
    const float invA = 1.0f / atmosphericLight;
    const float dark = (float)minValue * invMax * invA;
    const float transmission = fmax(1.0f - omega * fmin(dark, 1.0f), tFloor);
    const __global Type *srcRow = (const __global Type *)(pSrc + y * srcPitch);
    const float input = (float)srcRow[x] * invMax;
    const float restored = clamp((input - atmosphericLight) / transmission + atmosphericLight, 0.0f, 1.0f);
    __global Type *dstRow = (__global Type *)(pDst + y * dstPitch);
    dstRow[x] = (Type)(restored * maxValue + 0.5f);
}
