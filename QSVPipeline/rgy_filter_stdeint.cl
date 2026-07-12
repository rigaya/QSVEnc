// ST-DeInt zero-copy 経路用の色変換・weaveカーネル。

inline uchar stdeint_to_u8(const float value) {
    return convert_uchar_sat(convert_int_rtz(value + 0.5f));
}

inline float3 stdeint_load_rgb(
    __global const float *input,
    __global const float *restorations,
    const int x,
    const int y,
    const int width,
    const int height,
    const int frameA) {
    const int plane = width * height;
    const int halfPlane = plane / 2;
    const int useInput = ((y & 1) == (frameA ? 0 : 1));
    const int inputIndex = y * width + x;
    const int restoreBase = (frameA ? 0 : 3 * halfPlane) + (y / 2) * width + x;
    return useInput
        ? (float3)(input[inputIndex], input[plane + inputIndex], input[2 * plane + inputIndex])
        : (float3)(restorations[restoreBase], restorations[halfPlane + restoreBase], restorations[2 * halfPlane + restoreBase]);
}

__kernel void stdeint_pack_rgb(
    __global const uchar *srcY, const int pitchY,
    __global const uchar *srcU, const int pitchU,
    __global const uchar *srcV, const int pitchV,
    const int nv12,
    __global float *dst,
    const int width, const int height,
    const float yOff, const float yScale,
    const float cOff, const float cScale,
    const float matVR, const float matUG, const float matVG, const float matUB) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int cx = x / 2;
    const int cy = y / 2;
    const int cindexU = cy * pitchU + cx * (nv12 ? 2 : 1);
    const int cindexV = cy * pitchV + cx * (nv12 ? 2 : 1) + (nv12 ? 1 : 0);
    const float yn = ((float)srcY[y * pitchY + x] - yOff) * yScale;
    const float un = ((float)srcU[cindexU] - cOff) * cScale;
    const float vn = ((float)srcV[cindexV] - cOff) * cScale;
    const int index = y * width + x;
    const int plane = width * height;
    dst[index] = clamp(yn + matVR * vn, 0.0f, 1.0f);
    dst[plane + index] = clamp(yn + matUG * un + matVG * vn, 0.0f, 1.0f);
    dst[2 * plane + index] = clamp(yn + matUB * un, 0.0f, 1.0f);
}

__kernel void stdeint_weave_yuv(
    __global const float *input,
    __global const float *restorations,
    __global uchar *dstY, const int pitchY,
    __global uchar *dstU, const int pitchU,
    __global uchar *dstV, const int pitchV,
    const int nv12,
    const int width, const int height,
    const int frameA,
    const float yOff, const float yRange,
    const float cOff, const float cRange,
    const float matRY, const float matGY, const float matBY,
    const float matRU, const float matGU, const float matBU,
    const float matRV, const float matGV, const float matBV) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }

    const float3 rgb = stdeint_load_rgb(input, restorations, x, y, width, height, frameA);
    const float luma = matRY * rgb.x + matGY * rgb.y + matBY * rgb.z;
    dstY[y * pitchY + x] = stdeint_to_u8(luma * yRange + yOff);

    if ((x & 1) == 0 && (y & 1) == 0) {
        const float3 rgb01 = stdeint_load_rgb(input, restorations, x + 1, y, width, height, frameA);
        const float3 rgb10 = stdeint_load_rgb(input, restorations, x, y + 1, width, height, frameA);
        const float3 rgb11 = stdeint_load_rgb(input, restorations, x + 1, y + 1, width, height, frameA);
        const float3 sum = rgb + rgb01 + rgb10 + rgb11;
        const float u = 0.25f * (matRU * sum.x + matGU * sum.y + matBU * sum.z);
        const float v = 0.25f * (matRV * sum.x + matGV * sum.y + matBV * sum.z);
        const int cx = x / 2;
        const int cy = y / 2;
        dstU[cy * pitchU + cx * (nv12 ? 2 : 1)] = stdeint_to_u8(u * cRange + cOff);
        dstV[cy * pitchV + cx * (nv12 ? 2 : 1) + (nv12 ? 1 : 0)] = stdeint_to_u8(v * cRange + cOff);
    }
}
