// Type
// bit_depth

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

void check_min_max(float *vmin, float *vmax, float value) {
    *vmax = max(*vmax, value);
    *vmin = min(*vmin, value);
}

__kernel void kernel_edgelevel(
    __global uchar *restrict pDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
    __read_only image2d_t src,
    const float strength, const float threshold, const float black, const float white) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    if (ix < dstWidth && iy < dstHeight) {
        float center = (float)read_imagef(src, sampler, (int2)(ix, iy)).x;
        float hmin = center;
        float vmin = center;
        float hmax = center;
        float vmax = center;

        check_min_max(&hmin, &hmax, (float)read_imagef(src, sampler, (int2)(ix - 2, iy)).x);
        check_min_max(&vmin, &vmax, (float)read_imagef(src, sampler, (int2)(ix, iy - 2)).x);
        check_min_max(&hmin, &hmax, (float)read_imagef(src, sampler, (int2)(ix - 1, iy)).x);
        check_min_max(&vmin, &vmax, (float)read_imagef(src, sampler, (int2)(ix, iy - 1)).x);
        check_min_max(&hmin, &hmax, (float)read_imagef(src, sampler, (int2)(ix + 1, iy)).x);
        check_min_max(&vmin, &vmax, (float)read_imagef(src, sampler, (int2)(ix, iy + 1)).x);
        check_min_max(&hmin, &hmax, (float)read_imagef(src, sampler, (int2)(ix + 2, iy)).x);
        check_min_max(&vmin, &vmax, (float)read_imagef(src, sampler, (int2)(ix, iy + 2)).x);

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
        ptr[0] = (Type)(clamp(center, 0.0f, 1.0f - 1e-6f) * ((1 << (bit_depth))-1));
    }
}
