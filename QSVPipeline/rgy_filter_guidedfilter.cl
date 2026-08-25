// Type
// bit_depth
//
// Guided image filter kernel (--vpp-guidedfilter).
// Clean-room port of the He/Sun/Tang TPAMI'12 algorithm via the MIT
// Lisabug Python reference. Self-guided gray variant (I = p): use the
// source plane as both guide AND filtering input, giving an edge-aware
// smoothing pass. Useful as a denoise/detail-enhancement primitive and
// as a building block for the CLAHE/dehaze companion filter.
//
// Algorithm (self-guided simplification of the paper's 5 steps):
//   meanI   = box(I)
//   meanII  = box(I * I)
//   varI    = meanII - meanI^2
//   a       = varI / (varI + eps)
//   b       = (1 - a) * meanI
//   meana   = box(a)
//   meanb   = box(b)
//   q       = meana * I + meanb
//
// Implemented as two kernel passes with float intermediate (a, b)
// buffers:
//   pass 1 (kernel_guidedfilter_calc_ab): I -> (a, b)
//   pass 2 (kernel_guidedfilter_calc_q):  I + (a, b) -> q
// Border behaviour: clamp-to-edge (matches the paper's natural choice).

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void kernel_guidedfilter_calc_ab(
    __global float *restrict pA,
    __global float *restrict pB,
    const int abPitch,
    const __global uchar *restrict pSrc, const int srcPitch,
    const int width, const int height,
    const int radius, const float eps) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const float max_val = (float)((1 << bit_depth) - 1);
    const float inv_max = 1.0f / max_val;

    // Box sum of I and I^2 over the (2r+1)^2 neighbourhood, working in
    // normalised [0, 1] luminance to keep meanII numerically stable
    // regardless of bit depth.
    float sum_I  = 0.0f;
    float sum_II = 0.0f;
    int   count  = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        const int sy = clamp(y + dy, 0, height - 1);
        const __global Type *row = (const __global Type *)(pSrc + sy * srcPitch);
        for (int dx = -radius; dx <= radius; dx++) {
            const int sx = clamp(x + dx, 0, width - 1);
            const float v = (float)row[sx] * inv_max;
            sum_I  += v;
            sum_II += v * v;
            count++;
        }
    }
    const float inv_count = 1.0f / (float)count;
    const float meanI  = sum_I  * inv_count;
    const float meanII = sum_II * inv_count;
    const float varI   = meanII - meanI * meanI;
    const float a      = varI / (varI + eps);
    const float b      = (1.0f - a) * meanI;

    pA[y * abPitch + x] = a;
    pB[y * abPitch + x] = b;
}

__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void kernel_guidedfilter_calc_q(
    __global uchar *restrict pDst, const int dstPitch,
    const __global uchar *restrict pSrc, const int srcPitch,
    const int width, const int height,
    const __global float *restrict pA,
    const __global float *restrict pB,
    const int abPitch,
    const int radius) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const float max_val = (float)((1 << bit_depth) - 1);
    const float inv_max = 1.0f / max_val;

    // Box sum of a, b over the (2r+1)^2 neighbourhood.
    float sum_a = 0.0f;
    float sum_b = 0.0f;
    int   count = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        const int sy = clamp(y + dy, 0, height - 1);
        for (int dx = -radius; dx <= radius; dx++) {
            const int sx = clamp(x + dx, 0, width - 1);
            sum_a += pA[sy * abPitch + sx];
            sum_b += pB[sy * abPitch + sx];
            count++;
        }
    }
    const float inv_count = 1.0f / (float)count;
    const float meana = sum_a * inv_count;
    const float meanb = sum_b * inv_count;

    const __global Type *srcRow = (const __global Type *)(pSrc + y * srcPitch);
    const float I = (float)srcRow[x] * inv_max;
    const float q = meana * I + meanb;

    __global Type *dstRow = (__global Type *)(pDst + y * dstPitch);
    const int out_val = (int)clamp(q * max_val + 0.5f, 0.0f, max_val);
    dstRow[x] = (Type)out_val;
}

