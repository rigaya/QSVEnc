
// Type
// bit_depth
// knn_radius

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

float lerpf(float v0, float v1, float t) {
    float tmp = (1.0f-t)*v0;
    return tmp + t*v1;
}

// SLM tile-load. WG fixed at 32x8 = 256 threads. Per WG a tile of
// (32+2R) x (8+2R) source samples is cooperatively loaded into __local
// float[] before the per-pixel inner loop reads from local memory only.
//
// At knn_radius=3 the tile is 38x14 = 532 floats (2128 B SLM / WG); at
// knn_radius=5 it is 42x18 = 756 floats (3024 B). Well under the 64 KB
// per-WG SLM budget on Xe-HPG, so occupancy is unconstrained.
//
// The cooperative-load step issues only tile_w*tile_h image reads per WG
// (~3 per thread at R=5) instead of (2R+1)^2 per thread (121 at R=5),
// trading per-thread sampler issue cost for one workgroup barrier.
#define KNN_LX 32
#define KNN_LY 8
#define KNN_TILE_W (KNN_LX + 2 * knn_radius)
#define KNN_TILE_H (KNN_LY + 2 * knn_radius)

__attribute__((reqd_work_group_size(KNN_LX, KNN_LY, 1)))
__kernel void kernel_denoise_knn(
    __global uchar *restrict pDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
    __read_only image2d_t src,
    const float strength, const float lerpC, const float weight_threshold, const float lerp_threshold) {
    const float knn_window_area = (float)((2 * knn_radius + 1) * (2 * knn_radius + 1));
    const float inv_knn_window_area = 1.0f / knn_window_area;
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

    __local float tile[KNN_TILE_H * KNN_TILE_W];

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int gx0 = get_group_id(0) * KNN_LX;
    const int gy0 = get_group_id(1) * KNN_LY;
    const int tid = ly * KNN_LX + lx;
    const int wg_size = KNN_LX * KNN_LY;
    const int tile_total = KNN_TILE_W * KNN_TILE_H;

    for (int t = tid; t < tile_total; t += wg_size) {
        const int tx = t % KNN_TILE_W;
        const int ty = t / KNN_TILE_W;
        const int sx = clamp(gx0 + tx - knn_radius, 0, dstWidth  - 1);
        const int sy = clamp(gy0 + ty - knn_radius, 0, dstHeight - 1);
        tile[t] = (float)read_imagef(src, sampler, (int2)(sx, sy)).x;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int ix = gx0 + lx;
    const int iy = gy0 + ly;
    if (ix < dstWidth && iy < dstHeight) {
        const int tcx = lx + knn_radius;
        const int tcy = ly + knn_radius;
        const float center = tile[tcy * KNN_TILE_W + tcx];

        float fCount = 0.0f;
        float sumWeights = 0.0f;
        float sum = 0.0f;

        #pragma unroll
        for (int i = -knn_radius; i <= knn_radius; i++) {
            #pragma unroll
            for (int j = -knn_radius; j <= knn_radius; j++) {
                const float clrIJ = tile[(tcy + j) * KNN_TILE_W + (tcx + i)];
                const float distanceIJ = (center - clrIJ) * (center - clrIJ);
                const float weightIJ = native_exp(-(distanceIJ * strength + (i * i + j * j) * inv_knn_window_area));
                sum += clrIJ * weightIJ;
                sumWeights += weightIJ;
                fCount += (weightIJ > weight_threshold) ? inv_knn_window_area : 0.0f;
            }
        }
        const float lerpQ = (fCount > lerp_threshold) ? lerpC : 1.0f - lerpC;

        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(lerpf(sum * native_recip(sumWeights), center, lerpQ) * (float)((1 << bit_depth) - 1), 0.0f, (1 << bit_depth) - 0.1f);
    }
}
