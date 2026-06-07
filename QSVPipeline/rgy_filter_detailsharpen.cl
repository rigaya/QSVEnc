// Type
// bit_depth
// detailsharpen_mode
// detailsharpen_med

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

#define PIXEL_MAX  ((1 << (bit_depth)) - 1)

static inline float detailsharpen_read(const __global uchar *pSrc, const int pitch, int x, int y, const int width, const int height) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    const __global Type *ptr = (const __global Type *)(pSrc + y * pitch + x * sizeof(Type));
    return (float)ptr[0];
}

static inline float detailsharpen_median9(float v[9]) {
    for (int i = 1; i < 9; i++) {
        const float key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > key) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = key;
    }
    return v[4];
}

__kernel void kernel_detailsharpen_blur(
    __global uchar *restrict pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const __global uchar *restrict pSrc, const int srcPitch) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < dstWidth && iy < dstHeight) {
        const float p00 = detailsharpen_read(pSrc, srcPitch, ix - 1, iy - 1, dstWidth, dstHeight);
        const float p01 = detailsharpen_read(pSrc, srcPitch, ix,     iy - 1, dstWidth, dstHeight);
        const float p02 = detailsharpen_read(pSrc, srcPitch, ix + 1, iy - 1, dstWidth, dstHeight);
        const float p10 = detailsharpen_read(pSrc, srcPitch, ix - 1, iy,     dstWidth, dstHeight);
        const float p11 = detailsharpen_read(pSrc, srcPitch, ix,     iy,     dstWidth, dstHeight);
        const float p12 = detailsharpen_read(pSrc, srcPitch, ix + 1, iy,     dstWidth, dstHeight);
        const float p20 = detailsharpen_read(pSrc, srcPitch, ix - 1, iy + 1, dstWidth, dstHeight);
        const float p21 = detailsharpen_read(pSrc, srcPitch, ix,     iy + 1, dstWidth, dstHeight);
        const float p22 = detailsharpen_read(pSrc, srcPitch, ix + 1, iy + 1, dstWidth, dstHeight);

        float blur = 0.0f;
        if (detailsharpen_mode == 0) {
            blur = (p00 + p02 + p20 + p22 + 2.0f * (p01 + p10 + p12 + p21) + 4.0f * p11) * (1.0f / 16.0f);
        } else {
            blur = (p00 + p01 + p02 + p10 + p11 + p12 + p20 + p21 + p22) * (1.0f / 9.0f);
        }
        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(blur + 0.5f);
    }
}

__kernel void kernel_detailsharpen_apply(
    __global uchar *restrict pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    const __global uchar *restrict pSrc, const int srcPitch,
    const __global uchar *restrict pBlur, const int blurPitch,
    const float z, const float invPower, const float ldmp, const float strengthScaled, const float invI) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < dstWidth && iy < dstHeight) {
        const float xv = detailsharpen_read(pSrc, srcPitch, ix, iy, dstWidth, dstHeight);
        float yv = 0.0f;
        if (detailsharpen_med) {
            float v[9];
            v[0] = detailsharpen_read(pBlur, blurPitch, ix - 1, iy - 1, dstWidth, dstHeight);
            v[1] = detailsharpen_read(pBlur, blurPitch, ix,     iy - 1, dstWidth, dstHeight);
            v[2] = detailsharpen_read(pBlur, blurPitch, ix + 1, iy - 1, dstWidth, dstHeight);
            v[3] = detailsharpen_read(pBlur, blurPitch, ix - 1, iy,     dstWidth, dstHeight);
            v[4] = detailsharpen_read(pBlur, blurPitch, ix,     iy,     dstWidth, dstHeight);
            v[5] = detailsharpen_read(pBlur, blurPitch, ix + 1, iy,     dstWidth, dstHeight);
            v[6] = detailsharpen_read(pBlur, blurPitch, ix - 1, iy + 1, dstWidth, dstHeight);
            v[7] = detailsharpen_read(pBlur, blurPitch, ix,     iy + 1, dstWidth, dstHeight);
            v[8] = detailsharpen_read(pBlur, blurPitch, ix + 1, iy + 1, dstWidth, dstHeight);
            yv = detailsharpen_median9(v);
        } else {
            yv = detailsharpen_read(pBlur, blurPitch, ix, iy, dstWidth, dstHeight);
        }

        float out = xv;
        if (xv != yv) {
            const float diff = (xv - yv) * invI;
            const float absd = fabs(diff);
            const float amp = pow(absd / z, invPower);
            out = xv + strengthScaled * amp * diff / (absd + ldmp);
        }
        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)(clamp(out, 0.0f, (float)PIXEL_MAX) + 0.5f);
    }
}
