// Type
// Type4
// bit_depth

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

Type apply_basic_tweak_y(Type y, const float contrast, const float brightness, const float gamma_inv) {
    float pixel = (float)y * (1.0f / (1 << bit_depth));
    pixel = contrast * (pixel - 0.5f) + 0.5f + brightness;
    pixel = pow(pixel, gamma_inv);
    return (Type)clamp((int)(pixel * (1 << (bit_depth))), 0, (1 << (bit_depth)) - 1);
}

__kernel void kernel_tweak_y(
    __global uchar *restrict pFrame,
    const int pitch, const int width, const int height,
    const float contrast, const float brightness, const float gamma_inv) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < width && iy < height) {
        __global Type4 *ptr = (__global Type4 *)(pFrame + iy * pitch + ix * sizeof(Type4));
        Type4 src = ptr[0];

        Type4 ret;
        ret.x = apply_basic_tweak_y(src.x, contrast, brightness, gamma_inv);
        ret.y = apply_basic_tweak_y(src.y, contrast, brightness, gamma_inv);
        ret.z = apply_basic_tweak_y(src.z, contrast, brightness, gamma_inv);
        ret.w = apply_basic_tweak_y(src.w, contrast, brightness, gamma_inv);

        ptr[0] = ret;
    }
}

void apply_basic_tweak_uv(Type *u, Type *v, const float saturation, const float hue_sin, const float hue_cos) {
    float u0 = (float)u[0] * (1.0f / (1 << bit_depth));
    float v0 = (float)v[0] * (1.0f / (1 << bit_depth));
    u0 = saturation * (u0 - 0.5f) + 0.5f;
    v0 = saturation * (v0 - 0.5f) + 0.5f;

    float u1 = ((hue_cos * (u0 - 0.5f)) - (hue_sin * (v0 - 0.5f))) + 0.5f;
    float v1 = ((hue_sin * (u0 - 0.5f)) + (hue_cos * (v0 - 0.5f))) + 0.5f;

    u[0] = (Type)clamp((int)(u1 * (1 << (bit_depth))), 0, (1 << (bit_depth)) - 1);
    v[0] = (Type)clamp((int)(v1 * (1 << (bit_depth))), 0, (1 << (bit_depth)) - 1);
}

__kernel void kernel_tweak_uv(
    __global uchar *restrict pFrameU,
    __global uchar *restrict pFrameV,
    const int pitch, const int width, const int height,
    const float saturation, const float hue_sin, const float hue_cos) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < width && iy < height) {
        __global Type4 *ptrU = (__global Type4 *)(pFrameU + iy * pitch + ix * sizeof(Type4));
        __global Type4 *ptrV = (__global Type4 *)(pFrameV + iy * pitch + ix * sizeof(Type4));

        Type4 pixelU = ptrU[0];
        Type4 pixelV = ptrV[0];
        
        Type u0 = pixelU.x, u1 = pixelU.y, u2 = pixelU.z, u3 = pixelU.w;
        Type v0 = pixelV.x, v1 = pixelV.y, v2 = pixelV.z, v3 = pixelV.w;

        apply_basic_tweak_uv(&u0, &v0, saturation, hue_sin, hue_cos);
        apply_basic_tweak_uv(&u1, &v1, saturation, hue_sin, hue_cos);
        apply_basic_tweak_uv(&u2, &v2, saturation, hue_sin, hue_cos);
        apply_basic_tweak_uv(&u3, &v3, saturation, hue_sin, hue_cos);

        pixelU.x = u0, pixelU.y = u1, pixelU.z = u2, pixelU.w = u3;
        pixelV.x = v0, pixelV.y = v1, pixelV.z = v2, pixelV.w = v3;

        ptrU[0] = pixelU;
        ptrV[0] = pixelV;
    }
}