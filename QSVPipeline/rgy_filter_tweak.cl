// Type
// Type4
// bit_depth
// TWEAK_Y
// TWEAK_CB
// TWEAK_CR

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

Type apply_basic_tweak_y(Type y, const float contrast, const float brightness, const float gamma_inv, const int clamp_min, const int clamp_max) {
    float pixel = (float)y * (1.0f / (1 << bit_depth));
    pixel = contrast * (pixel - 0.5f) + 0.5f + brightness;
    if (gamma_inv != 1.0f) pixel = pow(pixel, gamma_inv);
    return (Type)clamp((int)(pixel * (1 << (bit_depth))), clamp_min, clamp_max);
}

Type apply_basic_tweak_y_without_gamma(Type y, const float contrast, const float brightness, const int clamp_min, const int clamp_max) {
    float pixel = (float)y * (1.0f / (1 << bit_depth));
    pixel = contrast * (pixel - 0.5f) + 0.5f + brightness;
    return (Type)clamp((int)(pixel * (1 << (bit_depth))), clamp_min, clamp_max);
}

Type apply_basic_tweak_cbcr(Type y, const float contrast, const float brightness, const int clamp_min, const int clamp_max) {
    float pixel = (float)y * (1.0f / (1 << bit_depth));
    pixel = contrast * pixel + brightness;
    return (Type)clamp((int)(pixel * (1 << (bit_depth))), clamp_min, clamp_max);
}

__kernel void kernel_tweak_y(
    __global uchar *restrict pFrame,
    const int pitch, const int width, const int height,
    const float contrast, const float brightness, const float gamma_inv,
    const float y_gain, const float y_offset,
    const int clamp_min, const int clamp_max) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < width && iy < height) {
        __global Type4 *ptr = (__global Type4 *)(pFrame + iy * pitch + ix * sizeof(Type4));
        Type4 src = ptr[0];

        Type4 ret;
        ret.x = apply_basic_tweak_y(src.x, contrast, brightness, gamma_inv, clamp_min, clamp_max);
        ret.y = apply_basic_tweak_y(src.y, contrast, brightness, gamma_inv, clamp_min, clamp_max);
        ret.z = apply_basic_tweak_y(src.z, contrast, brightness, gamma_inv, clamp_min, clamp_max);
        ret.w = apply_basic_tweak_y(src.w, contrast, brightness, gamma_inv, clamp_min, clamp_max);

        if (TWEAK_Y) {
            ret.x = apply_basic_tweak_y_without_gamma(ret.x, y_gain, y_offset, clamp_min, clamp_max);
            ret.y = apply_basic_tweak_y_without_gamma(ret.y, y_gain, y_offset, clamp_min, clamp_max);
            ret.z = apply_basic_tweak_y_without_gamma(ret.z, y_gain, y_offset, clamp_min, clamp_max);
            ret.w = apply_basic_tweak_y_without_gamma(ret.w, y_gain, y_offset, clamp_min, clamp_max);
        }
        ptr[0] = ret;
    }
}

void apply_basic_tweak_uv(Type *u, Type *v, const float saturation, const float vibrance,
    const float hue_sin, const float hue_cos,
    const int hue_limit, const float hue_min, const float hue_max, const int clamp_min, const int clamp_max) {
    float u0 = (float)u[0] * (1.0f / (1 << bit_depth));
    float v0 = (float)v[0] * (1.0f / (1 << bit_depth));
    if (hue_limit) {
        //元画素の色相(atan2(Cr,Cb), 0-360度)が指定範囲外なら変更しない
        float deg = atan2(v0 - 0.5f, u0 - 0.5f) * (180.0f / M_PI_F);
        if (deg < 0.0f) deg += 360.0f;
        const int in_range = (hue_min <= hue_max) ? (deg >= hue_min && deg <= hue_max) : (deg >= hue_min || deg <= hue_max);
        if (!in_range) return;
    }
    float sat = saturation;
    if (vibrance != 0.0f) {
        // 現在の彩度を0（無彩色）から1（色差平面の端）へ正規化し、
        // 彩度が低い画素ほどvibranceの効果を強くする。
        const float du = u0 - 0.5f;
        const float dv = v0 - 0.5f;
        const float already = fmin(2.0f * native_sqrt(du * du + dv * dv), 1.0f);
        sat = saturation * (1.0f + vibrance * (1.0f - already));
        sat = fmax(sat, 0.0f);
    }
    u0 = sat * (u0 - 0.5f) + 0.5f;
    v0 = sat * (v0 - 0.5f) + 0.5f;

    float u1 = ((hue_cos * (u0 - 0.5f)) - (hue_sin * (v0 - 0.5f))) + 0.5f;
    float v1 = ((hue_sin * (u0 - 0.5f)) + (hue_cos * (v0 - 0.5f))) + 0.5f;

    u[0] = (Type)clamp((int)(u1 * (1 << (bit_depth))), clamp_min, clamp_max);
    v[0] = (Type)clamp((int)(v1 * (1 << (bit_depth))), clamp_min, clamp_max);
}

__kernel void kernel_tweak_uv(
    __global uchar *restrict pFrameU,
    __global uchar *restrict pFrameV,
    const int pitch, const int width, const int height,
    const float saturation, const float vibrance, const float hue_sin, const float hue_cos, const int swapuv,
    const float cb_gain, const float cb_offset,
    const float cr_gain, const float cr_offset,
    const int hue_limit, const float hue_min, const float hue_max, const int clamp_min, const int clamp_max) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if (ix < width && iy < height) {
        __global Type4 *ptrU = (__global Type4 *)(pFrameU + iy * pitch + ix * sizeof(Type4));
        __global Type4 *ptrV = (__global Type4 *)(pFrameV + iy * pitch + ix * sizeof(Type4));

        Type4 pixelU = ptrU[0];
        Type4 pixelV = ptrV[0];

        Type u0 = pixelU.x, u1 = pixelU.y, u2 = pixelU.z, u3 = pixelU.w;
        Type v0 = pixelV.x, v1 = pixelV.y, v2 = pixelV.z, v3 = pixelV.w;

        apply_basic_tweak_uv(&u0, &v0, saturation, vibrance, hue_sin, hue_cos, hue_limit, hue_min, hue_max, clamp_min, clamp_max);
        apply_basic_tweak_uv(&u1, &v1, saturation, vibrance, hue_sin, hue_cos, hue_limit, hue_min, hue_max, clamp_min, clamp_max);
        apply_basic_tweak_uv(&u2, &v2, saturation, vibrance, hue_sin, hue_cos, hue_limit, hue_min, hue_max, clamp_min, clamp_max);
        apply_basic_tweak_uv(&u3, &v3, saturation, vibrance, hue_sin, hue_cos, hue_limit, hue_min, hue_max, clamp_min, clamp_max);

        pixelU.x = u0, pixelU.y = u1, pixelU.z = u2, pixelU.w = u3;
        pixelV.x = v0, pixelV.y = v1, pixelV.z = v2, pixelV.w = v3;

        if (TWEAK_CB) {
            pixelU.x = apply_basic_tweak_cbcr(pixelU.x, cb_gain, cb_offset, clamp_min, clamp_max);
            pixelU.y = apply_basic_tweak_cbcr(pixelU.y, cb_gain, cb_offset, clamp_min, clamp_max);
            pixelU.z = apply_basic_tweak_cbcr(pixelU.z, cb_gain, cb_offset, clamp_min, clamp_max);
            pixelU.w = apply_basic_tweak_cbcr(pixelU.w, cb_gain, cb_offset, clamp_min, clamp_max);
        }
        if (TWEAK_CR) {
            pixelV.x = apply_basic_tweak_cbcr(pixelV.x, cr_gain, cr_offset, clamp_min, clamp_max);
            pixelV.y = apply_basic_tweak_cbcr(pixelV.y, cr_gain, cr_offset, clamp_min, clamp_max);
            pixelV.z = apply_basic_tweak_cbcr(pixelV.z, cr_gain, cr_offset, clamp_min, clamp_max);
            pixelV.w = apply_basic_tweak_cbcr(pixelV.w, cr_gain, cr_offset, clamp_min, clamp_max);
        }

        ptrU[0] = (swapuv) ? pixelV : pixelU;
        ptrV[0] = (swapuv) ? pixelU : pixelV;
    }
}