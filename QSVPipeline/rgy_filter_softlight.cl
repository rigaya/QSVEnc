static inline ushort softlight_to_u16(const float v) {
    const float x = fmin(fmax(v, 0.0f), 1.0f);
    return (ushort)(x * 65535.0f + 0.5f);
}

static inline float softlight_func(const float a, const float b, const int formula) {
    if (formula == 1) {
        return pow(a, pow(2.0f, 1.0f - 2.0f * b));
    }
    if (formula == 2) {
        if (b <= 0.5f) {
            return a - (1.0f - 2.0f * b) * a * (1.0f - a);
        }
        const float g = (a <= 0.25f) ? (((16.0f * a - 12.0f) * a + 4.0f) * a) : sqrt(a);
        return a + (2.0f * b - 1.0f) * (g - a);
    }
    return (1.0f - 2.0f * b) * a * a + 2.0f * b * a;
}

static inline void rgb_to_hsv_value(const float r, const float g, const float b, float *h, float *s, float *v) {
    const float mx = fmax(r, fmax(g, b));
    const float mn = fmin(r, fmin(g, b));
    const float d = mx - mn;
    *v = mx;
    *s = (mx <= 0.0f) ? 0.0f : d / mx;
    if (d <= 0.0f) {
        *h = 0.0f;
    } else if (mx == r) {
        *h = fmod((g - b) / d, 6.0f);
    } else if (mx == g) {
        *h = (b - r) / d + 2.0f;
    } else {
        *h = (r - g) / d + 4.0f;
    }
    if (*h < 0.0f) *h += 6.0f;
}

static inline void hsv_to_rgb_value(float h, const float s, const float v, float *r, float *g, float *b) {
    if (s <= 0.0f) {
        *r = *g = *b = v;
        return;
    }
    h = fmod(h, 6.0f);
    if (h < 0.0f) h += 6.0f;
    const float c = v * s;
    const float x = c * (1.0f - fabs(fmod(h, 2.0f) - 1.0f));
    const float m = v - c;
    if (h < 1.0f) {
        *r = c; *g = x; *b = 0.0f;
    } else if (h < 2.0f) {
        *r = x; *g = c; *b = 0.0f;
    } else if (h < 3.0f) {
        *r = 0.0f; *g = c; *b = x;
    } else if (h < 4.0f) {
        *r = 0.0f; *g = x; *b = c;
    } else if (h < 5.0f) {
        *r = x; *g = 0.0f; *b = c;
    } else {
        *r = c; *g = 0.0f; *b = x;
    }
    *r += m;
    *g += m;
    *b += m;
}

__kernel void kernel_reduce_rgb_u16(
    const __global uchar *pR, const int pitchR,
    const __global uchar *pG, const int pitchG,
    const __global uchar *pB, const int pitchB,
    const int width, const int height,
    __global long *out_partials
) {
    __local long sh0[softlight_block_x * softlight_block_y];
    __local long sh1[softlight_block_x * softlight_block_y];
    __local long sh2[softlight_block_x * softlight_block_y];
    __local long sh3[softlight_block_x * softlight_block_y];
    __local long sh4[softlight_block_x * softlight_block_y];
    __local long sh5[softlight_block_x * softlight_block_y];

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int lid = get_local_id(1) * softlight_block_x + get_local_id(0);
    long sumR = 0, sumG = 0, sumB = 0;
    long blackR = 0, blackG = 0, blackB = 0;
    if (x < width && y < height) {
        const ushort r = *((const __global ushort *)(pR + y * pitchR + x * sizeof(ushort)));
        const ushort g = *((const __global ushort *)(pG + y * pitchG + x * sizeof(ushort)));
        const ushort b = *((const __global ushort *)(pB + y * pitchB + x * sizeof(ushort)));
        sumR = (long)r; sumG = (long)g; sumB = (long)b;
        blackR = (r == 0); blackG = (g == 0); blackB = (b == 0);
    }

    sh0[lid] = sumR; sh1[lid] = sumG; sh2[lid] = sumB;
    sh3[lid] = blackR; sh4[lid] = blackG; sh5[lid] = blackB;
    barrier(CLK_LOCAL_MEM_FENCE);

    const int wgSize = softlight_block_x * softlight_block_y;
    for (int offset = wgSize >> 1; offset > 0; offset >>= 1) {
        if (lid < offset) {
            sh0[lid] += sh0[lid + offset];
            sh1[lid] += sh1[lid + offset];
            sh2[lid] += sh2[lid + offset];
            sh3[lid] += sh3[lid + offset];
            sh4[lid] += sh4[lid + offset];
            sh5[lid] += sh5[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        const int groupIdx = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        out_partials[groupIdx * 6 + 0] = sh0[0];
        out_partials[groupIdx * 6 + 1] = sh1[0];
        out_partials[groupIdx * 6 + 2] = sh2[0];
        out_partials[groupIdx * 6 + 3] = sh3[0];
        out_partials[groupIdx * 6 + 4] = sh4[0];
        out_partials[groupIdx * 6 + 5] = sh5[0];
    }
}

__kernel void kernel_softlight_scalar_u16(
    __global uchar *pPlane, const int pitch, const int width, const int height,
    const float b, const int formula
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < width && y < height) {
        __global ushort *ptr = (__global ushort *)(pPlane + y * pitch + x * sizeof(ushort));
        const float a = (float)ptr[0] * (1.0f / 65535.0f);
        ptr[0] = softlight_to_u16(softlight_func(a, b, formula));
    }
}

__kernel void kernel_softlight_self_u16(
    __global uchar *pPlane, const int pitch, const int width, const int height,
    const int formula
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < width && y < height) {
        __global ushort *ptr = (__global ushort *)(pPlane + y * pitch + x * sizeof(ushort));
        const float a = (float)ptr[0] * (1.0f / 65535.0f);
        ptr[0] = softlight_to_u16(softlight_func(a, a, formula));
    }
}

__kernel void kernel_softlight_self_f32(__global float *pPlane, const int width, const int height, const int formula) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < width && y < height) {
        const int idx = y * width + x;
        const float a = pPlane[idx];
        pPlane[idx] = fmin(fmax(softlight_func(a, a, formula), 0.0f), 1.0f);
    }
}

__kernel void kernel_rgb_to_v_u16(
    const __global uchar *pR, const int pitchR,
    const __global uchar *pG, const int pitchG,
    const __global uchar *pB, const int pitchB,
    const int width, const int height,
    __global float *pV
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < width && y < height) {
        const float r = (float)(*((const __global ushort *)(pR + y * pitchR + x * sizeof(ushort)))) * (1.0f / 65535.0f);
        const float g = (float)(*((const __global ushort *)(pG + y * pitchG + x * sizeof(ushort)))) * (1.0f / 65535.0f);
        const float b = (float)(*((const __global ushort *)(pB + y * pitchB + x * sizeof(ushort)))) * (1.0f / 65535.0f);
        pV[y * width + x] = fmax(r, fmax(g, b));
    }
}

__kernel void kernel_rgb_to_hs_u16(
    const __global uchar *pR, const int pitchR,
    const __global uchar *pG, const int pitchG,
    const __global uchar *pB, const int pitchB,
    const int width, const int height,
    __global float *pH, __global float *pS
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < width && y < height) {
        float h, s, v;
        const float r = (float)(*((const __global ushort *)(pR + y * pitchR + x * sizeof(ushort)))) * (1.0f / 65535.0f);
        const float g = (float)(*((const __global ushort *)(pG + y * pitchG + x * sizeof(ushort)))) * (1.0f / 65535.0f);
        const float b = (float)(*((const __global ushort *)(pB + y * pitchB + x * sizeof(ushort)))) * (1.0f / 65535.0f);
        rgb_to_hsv_value(r, g, b, &h, &s, &v);
        const int idx = y * width + x;
        pH[idx] = h;
        pS[idx] = s;
    }
}

__kernel void kernel_rgb_to_hsv_u16(
    const __global uchar *pR, const int pitchR,
    const __global uchar *pG, const int pitchG,
    const __global uchar *pB, const int pitchB,
    const int width, const int height,
    __global float *pH, __global float *pS, __global float *pV
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < width && y < height) {
        float h, s, v;
        const float r = (float)(*((const __global ushort *)(pR + y * pitchR + x * sizeof(ushort)))) * (1.0f / 65535.0f);
        const float g = (float)(*((const __global ushort *)(pG + y * pitchG + x * sizeof(ushort)))) * (1.0f / 65535.0f);
        const float b = (float)(*((const __global ushort *)(pB + y * pitchB + x * sizeof(ushort)))) * (1.0f / 65535.0f);
        rgb_to_hsv_value(r, g, b, &h, &s, &v);
        const int idx = y * width + x;
        pH[idx] = h;
        pS[idx] = s;
        pV[idx] = v;
    }
}

__kernel void kernel_hsv_to_rgb_u16(
    __global uchar *pR, const int pitchR,
    __global uchar *pG, const int pitchG,
    __global uchar *pB, const int pitchB,
    const int width, const int height,
    const __global float *pH, const __global float *pS, const __global float *pV
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < width && y < height) {
        const int idx = y * width + x;
        float r, g, b;
        hsv_to_rgb_value(pH[idx], pS[idx], pV[idx], &r, &g, &b);
        *((__global ushort *)(pR + y * pitchR + x * sizeof(ushort))) = softlight_to_u16(r);
        *((__global ushort *)(pG + y * pitchG + x * sizeof(ushort))) = softlight_to_u16(g);
        *((__global ushort *)(pB + y * pitchB + x * sizeof(ushort))) = softlight_to_u16(b);
    }
}
