__kernel void pack_norm_y(__global const Type *srcY, int srcPitch,
                          __global float *dst, int srcW, int srcH,
                          int W, int H, int padL, int padT, float maxval) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    int sx = clamp(x - padL, 0, srcW - 1);
    int sy = clamp(y - padT, 0, srcH - 1);
    dst[y * W + x] = (float)srcY[sy * srcPitch + sx] / maxval;
}

__kernel void unpack_denorm_y(__global const float *src,
                              __global Type *dstY, int dstPitch, int W, int H, float maxval) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    int v = (int)(src[y * W + x] * maxval + 0.5f);
    dstY[y * dstPitch + x] = (Type)clamp(v, 0, (1 << bit_depth) - 1);
}

__kernel void pad_float_planes(__global const float *src0, __global const float *src1,
                               __global const float *src2, __global float *dst,
                               int srcW, int srcH, int W, int H, int padL, int padT) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    int sx = clamp(x - padL, 0, srcW - 1);
    int sy = clamp(y - padT, 0, srcH - 1);
    int srcIndex = sy * srcW + sx;
    int dstIndex = y * W + x;
    int planeSize = W * H;
    dst[dstIndex] = src0[srcIndex];
    dst[planeSize + dstIndex] = src1[srcIndex];
    dst[2 * planeSize + dstIndex] = src2[srcIndex];
}

__kernel void chroma_bilinear(__global const Type *src, int srcPitch, int srcStride, int srcOffset,
                              __global Type *dst, int dstPitch, int dstStride, int dstOffset,
                              int sw, int sh, int scale) {
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    int dw = sw * scale;
    int dh = sh * scale;
    if (dx >= dw || dy >= dh) return;
    float inv = 1.0f / (float)scale;
    float sx = (dx + 0.5f) * inv - 0.5f;
    float sy = (dy + 0.5f) * inv - 0.5f;
    int x0 = (int)floor(sx); float fx = sx - (float)x0;
    int y0 = (int)floor(sy); float fy = sy - (float)y0;
    int x0c = clamp(x0,     0, sw - 1);
    int x1c = clamp(x0 + 1, 0, sw - 1);
    int y0c = clamp(y0,     0, sh - 1);
    int y1c = clamp(y0 + 1, 0, sh - 1);
    float a = (float)src[y0c * srcPitch + x0c * srcStride + srcOffset];
    float b = (float)src[y0c * srcPitch + x1c * srcStride + srcOffset];
    float c = (float)src[y1c * srcPitch + x0c * srcStride + srcOffset];
    float d = (float)src[y1c * srcPitch + x1c * srcStride + srcOffset];
    float top = a + (b - a) * fx;
    float bot = c + (d - c) * fx;
    int v = (int)(top + (bot - top) * fy + 0.5f);
    dst[dy * dstPitch + dx * dstStride + dstOffset] = (Type)clamp(v, 0, (1 << bit_depth) - 1);
}
