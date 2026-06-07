// Type
// bit_depth

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

void check_min_max(float *vmin, float *vmax, float value) {
    *vmax = max(*vmax, value);
    *vmin = min(*vmin, value);
}

// SLM tile-load. Window: ±2 horizontal + ±2 vertical (8 image reads per
// pixel in the original). WG 32x8, tile 36x12 floats (1728 B/WG); well
// under the 64 KB per-WG SLM budget on Xe-HPG.
#define EDGELEVEL_LX 32
#define EDGELEVEL_LY 8
#define EDGELEVEL_R  2
#define EDGELEVEL_TILE_W (EDGELEVEL_LX + 2 * EDGELEVEL_R)
#define EDGELEVEL_TILE_H (EDGELEVEL_LY + 2 * EDGELEVEL_R)

__attribute__((reqd_work_group_size(EDGELEVEL_LX, EDGELEVEL_LY, 1)))
__kernel void kernel_edgelevel(
    __global uchar *restrict pDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
    __read_only image2d_t src,
    const float strength, const float threshold, const float black, const float white) {
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    __local float tile[EDGELEVEL_TILE_H * EDGELEVEL_TILE_W];

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int gx0 = get_group_id(0) * EDGELEVEL_LX;
    const int gy0 = get_group_id(1) * EDGELEVEL_LY;
    const int tid = ly * EDGELEVEL_LX + lx;
    const int wg_size = EDGELEVEL_LX * EDGELEVEL_LY;
    const int tile_total = EDGELEVEL_TILE_W * EDGELEVEL_TILE_H;

    for (int t = tid; t < tile_total; t += wg_size) {
        const int tx = t % EDGELEVEL_TILE_W;
        const int ty = t / EDGELEVEL_TILE_W;
        const int sx = gx0 + tx - EDGELEVEL_R;
        const int sy = gy0 + ty - EDGELEVEL_R;
        tile[t] = (float)read_imagef(src, sampler, (int2)(sx, sy)).x;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int ix = gx0 + lx;
    const int iy = gy0 + ly;
    if (ix < dstWidth && iy < dstHeight) {
        const int tcx = lx + EDGELEVEL_R;
        const int tcy = ly + EDGELEVEL_R;
        float center = tile[tcy * EDGELEVEL_TILE_W + tcx];
        float hmin = center;
        float vmin = center;
        float hmax = center;
        float vmax = center;

        check_min_max(&hmin, &hmax, tile[tcy * EDGELEVEL_TILE_W + (tcx - 2)]);
        check_min_max(&vmin, &vmax, tile[(tcy - 2) * EDGELEVEL_TILE_W + tcx]);
        check_min_max(&hmin, &hmax, tile[tcy * EDGELEVEL_TILE_W + (tcx - 1)]);
        check_min_max(&vmin, &vmax, tile[(tcy - 1) * EDGELEVEL_TILE_W + tcx]);
        check_min_max(&hmin, &hmax, tile[tcy * EDGELEVEL_TILE_W + (tcx + 1)]);
        check_min_max(&vmin, &vmax, tile[(tcy + 1) * EDGELEVEL_TILE_W + tcx]);
        check_min_max(&hmin, &hmax, tile[tcy * EDGELEVEL_TILE_W + (tcx + 2)]);
        check_min_max(&vmin, &vmax, tile[(tcy + 2) * EDGELEVEL_TILE_W + tcx]);

        if (hmax - hmin < vmax - vmin) {
            hmax = vmax, hmin = vmin;
        }

        if (hmax - hmin > threshold) {
            float avg = (hmin + hmax) * 0.5f;
            if (center == hmin)
                hmin -= black;
            hmin -= black;
            if (center == hmax)
                hmax += white;
            hmax += white;

            center = min(max((center + ((center - avg) * strength)), hmin), hmax);
        }

        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(clamp(center, 0.0f, 1.0f - 1e-6f) * ((1 << (bit_depth)) - 1));
    }
}
