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

float onnx_sample_chroma(__global const Type *src, int pitch, int stride, int offset,
                         int cw, int ch, int x, int y) {
    float fxpos = (x + 0.5f) * 0.5f - 0.5f;
    float fypos = (y + 0.5f) * 0.5f - 0.5f;
    int x0 = (int)floor(fxpos), y0 = (int)floor(fypos);
    float fx = fxpos - x0, fy = fypos - y0;
    int xa = clamp(x0, 0, cw - 1) * stride + offset;
    int xb = clamp(x0 + 1, 0, cw - 1) * stride + offset;
    int ya = clamp(y0, 0, ch - 1) * pitch;
    int yb = clamp(y0 + 1, 0, ch - 1) * pitch;
    float a = src[ya + xa], b = src[ya + xb], c = src[yb + xa], d = src[yb + xb];
    float top = a + (b - a) * fx, bottom = c + (d - c) * fx;
    return top + (bottom - top) * fy;
}

__kernel void pack_rgb(__global const Type *yplane, int ypitch,
                       __global const Type *uplane, int upitch,
                       __global const Type *vplane, int vpitch, int stride, int voffset,
                       __global float *dst, int srcW, int srcH, int W, int H, int padL, int padT,
                       float yoff, float yscale, float coff, float cscale,
                       float vr, float ug, float vg, float ub) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    int sx = clamp(x - padL, 0, srcW - 1), sy = clamp(y - padT, 0, srcH - 1);
    float yn = ((float)yplane[sy * ypitch + sx] - yoff) * yscale;
    float un = (onnx_sample_chroma(uplane, upitch, stride, 0, srcW / 2, srcH / 2, sx, sy) - coff) * cscale;
    float vn = (onnx_sample_chroma(vplane, vpitch, stride, voffset, srcW / 2, srcH / 2, sx, sy) - coff) * cscale;
    int i = y * W + x, plane = W * H;
    dst[i] = clamp(yn + vr * vn, 0.0f, 1.0f);
    dst[plane + i] = clamp(yn + ug * un + vg * vn, 0.0f, 1.0f);
    dst[2 * plane + i] = clamp(yn + ub * un, 0.0f, 1.0f);
}

__kernel void unpack_rgb_y(__global const float *src, __global Type *dst, int pitch, int W, int H,
                           float ry, float gy, float by, float yrange, float yoff) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    int i = y * W + x, plane = W * H;
    float yn = ry * src[i] + gy * src[plane + i] + by * src[2 * plane + i];
    dst[y * pitch + x] = (Type)clamp((int)(yn * yrange + yoff + 0.5f), 0, (1 << bit_depth) - 1);
}

__kernel void unpack_rgb_uv(__global const float *src, __global Type *dstU, int pitchU,
                            __global Type *dstV, int pitchV, int stride, int voffset, int W, int H,
                            float ru, float gu, float bu, float rv, float gv, float bv,
                            float crange, float coff) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W / 2 || y >= H / 2) return;
    int plane = W * H, i = 2 * y * W + 2 * x;
    float u[4], v[4];
    for (int k = 0; k < 4; k++) {
        int j = i + (k / 2) * W + (k % 2);
        float r = src[j], g = src[plane + j], b = src[2 * plane + j];
        u[k] = ru * r + gu * g + bu * b;
        v[k] = rv * r + gv * g + bv * b;
    }
    float ua = (u[0] + u[1] + u[2] + u[3]) * 0.25f;
    float va = (v[0] + v[1] + v[2] + v[3]) * 0.25f;
    dstU[y * pitchU + x * stride] = (Type)clamp((int)(ua * crange + coff + 0.5f), 0, (1 << bit_depth) - 1);
    dstV[y * pitchV + x * stride + voffset] = (Type)clamp((int)(va * crange + coff + 0.5f), 0, (1 << bit_depth) - 1);
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
