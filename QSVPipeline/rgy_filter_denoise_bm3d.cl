// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2026 rigaya
// Copyright (c) 2015 Sampsa Sarjanoja
// Copyright (c) 2015-2016 mawen1250
// Copyright (c) 2021 HuangZhangming
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// -----------------------------------------------------------------------------------------

// Type
// bit_depth
//
// BM3D denoise (--vpp-bm3d), Step 1 Basic estimate. Clean-room from
// Dabov 2007 ("Image denoising by sparse 3D transform-domain
// collaborative filtering"). Port methodology mirrors the
// MIT-licensed Sampas/bm3dcl reference (Sarjanoja / Boutellier /
// Hannuksela, DASIP 2015): three-kernel pipeline.
//
//   1. kernel_bm3d_match  - per reference patch, find up to
//      group_size most-similar blocks within +/- bm_range; output
//      sorted top-K positions + block_count.
//   2. kernel_bm3d_basic  - per reference patch, stack the matched
//      blocks; apply 2D DCT per block; apply 1D Haar across the
//      group dimension; hard-threshold at tau_1d; inverse 1D Haar;
//      inverse 2D DCT; scatter the filtered blocks into a global
//      float accumulator + weight map (Kaiser-weighted aggregation,
//      atomic float add via CAS).
//   3. kernel_bm3d_normalize - per output pixel, out = sat(acc / w).
//
// BLOCK_SIZE is compile-time 8 (the DCT-8 butterfly math is fixed
// at this width per the spec). block_step, group_size, bm_range,
// and sigma flow as runtime kernel args.


// The fixed-count inner loops below are unrolled. Their trip counts are
// compile-time constants (BLOCK_SIZE, and 8 for the Haar stage), and leaving
// them rolled costs more than the loop bodies: measured on an Arc A770 this is
// worth -18.7% to -54.4% depending on profile. Unrolling does not reorder any
// accumulation, so the result is unchanged.
#define BLOCK_SIZE 8
#define BLOCK_SIZE_SQ 64
#define MAX_GROUP_SIZE_BASIC 16
// The Wiener step admits a larger group than the basic step because
// the basic-estimate-driven block-matching produces cleaner matches
// (Sampas reference: MAX_BLOCK_COUNT_1 = 16, MAX_BLOCK_COUNT_2 = 32).
#define MAX_GROUP_SIZE_WIENER 32

// V-BM3D temporal mode (radius > 0) caps group_size at 16. The temporal
// kernels build the noise + basic stacks in per-WI private memory just
// like the spatial Wiener kernel; capping the group at 16 keeps the
// per-WI private memory at 16*8*8*4 = 4 KB per stack (8 KB total for
// the Wiener step), the same envelope as the spatial basic step which
// is proven safe on Arc A770. A previous V-BM3D attempt that left the
// cap at 32 + scattered to per-frame accumulators crashed the Arc
// graphics driver via cumulative LSC scratch exhaustion.
#define MAX_GROUP_SIZE_TEMPORAL 16

// 2D DCT-8 butterfly constants (Sampas-port, all spec-derived).
// cos(3*PI/16)
#define DCT_C3A 0.83146961230254523707878837761791f
// sin(3*PI/16)
#define DCT_C3B 0.55557023301960222474283081394853f
// cos(PI/16)
#define DCT_C1A 0.98078528040323044912618223613424f
// sin(PI/16)
#define DCT_C1B 0.19509032201612826784828486847702f
// sqrt(2) * cos(3*PI/8)
#define DCT_S2C3A 0.54119610014619698439972320536639f
// sqrt(2) * sin(3*PI/8)
#define DCT_S2C3B 1.3065629648763765278566431734272f
#define DCT_NORM_2D 0.125f
#define DCT_SQRT2 1.4142135623730950488016887242097f

// 1D Haar normalization (1/sqrt(2)).
#define HAAR_INV_SQRT2 0.70710678118654752440084436210485f

// 8x8 Kaiser window (beta=2) for output aggregation; Sampas-port.
__constant float c_kaiser[BLOCK_SIZE_SQ] = {
    0.1924f, 0.2989f, 0.3846f, 0.4325f, 0.4325f, 0.3846f, 0.2989f, 0.1924f,
    0.2989f, 0.4642f, 0.5974f, 0.6717f, 0.6717f, 0.5974f, 0.4642f, 0.2989f,
    0.3846f, 0.5974f, 0.7688f, 0.8644f, 0.8644f, 0.7688f, 0.5974f, 0.3846f,
    0.4325f, 0.6717f, 0.8644f, 0.9718f, 0.9718f, 0.8644f, 0.6717f, 0.4325f,
    0.4325f, 0.6717f, 0.8644f, 0.9718f, 0.9718f, 0.8644f, 0.6717f, 0.4325f,
    0.3846f, 0.5974f, 0.7688f, 0.8644f, 0.8644f, 0.7688f, 0.5974f, 0.3846f,
    0.2989f, 0.4642f, 0.5974f, 0.6717f, 0.6717f, 0.5974f, 0.4642f, 0.2989f,
    0.1924f, 0.2989f, 0.3846f, 0.4325f, 0.4325f, 0.3846f, 0.2989f, 0.1924f
};

inline float read_pixel(const __global uchar *src, int srcPitch, int x, int y, int W, int H) {
    x = clamp(x, 0, W - 1);
    y = clamp(y, 0, H - 1);
    const __global Type *row = (const __global Type *)(src + y * srcPitch);
    return (float)row[x];
}

inline void atomic_add_global_float(volatile __global float *addr, float val) {
    union { uint u; float f; } prev, next;
    do {
        prev.f = *addr;
        next.f = prev.f + val;
    } while (atomic_cmpxchg((volatile __global uint *)addr, prev.u, next.u) != prev.u);
}

// 8-point DCT-II (in-place via temporaries). Normalised at the end.
inline void dct8_row(float v[8]) {
    float s[8], t[2], tmp;
    s[0] = v[0] + v[7]; s[1] = v[1] + v[6]; s[2] = v[2] + v[5]; s[3] = v[3] + v[4];
    s[4] = v[3] - v[4]; s[5] = v[2] - v[5]; s[6] = v[1] - v[6]; s[7] = v[0] - v[7];
    float u0 = s[0] + s[3];
    float u1 = s[1] + s[2];
    float u2 = s[1] - s[2];
    float u3 = s[0] - s[3];
    tmp = DCT_C3A * (s[4] + s[7]);
    float u4 = tmp + (DCT_C3B - DCT_C3A) * s[7];
    float u7 = tmp - (DCT_C3A + DCT_C3B) * s[4];
    tmp = DCT_C1A * (s[5] + s[6]);
    float u5 = tmp + (DCT_C1B - DCT_C1A) * s[6];
    float u6 = tmp - (DCT_C1A + DCT_C1B) * s[5];
    v[0] = u0 + u1;
    v[4] = u0 - u1;
    tmp = DCT_S2C3A * (u2 + u3);
    v[2] = tmp + (DCT_S2C3B - DCT_S2C3A) * u3;
    v[6] = tmp - (DCT_S2C3A + DCT_S2C3B) * u2;
    t[0] = u4 + u6;
    t[1] = u5 + u7;
    v[3] = (u7 - u5) * DCT_SQRT2;
    v[5] = (u4 - u6) * DCT_SQRT2;
    v[1] = t[0] + t[1];
    v[7] = t[1] - t[0];
}

// 8-point IDCT-II (in-place via temporaries). Normalised at the end.
inline void idct8_row(float v[8]) {
    float r[8], s4[2], tmp;
    s4[0] = v[1] - v[7];
    float st5 = v[3] * DCT_SQRT2;
    float st6 = v[5] * DCT_SQRT2;
    s4[1] = v[1] + v[7];
    r[0] = v[0] + v[4];
    r[1] = v[0] - v[4];
    tmp = DCT_S2C3A * (v[2] + v[6]);
    r[2] = tmp - (DCT_S2C3A + DCT_S2C3B) * v[6];
    r[3] = tmp + (DCT_S2C3B - DCT_S2C3A) * v[2];
    r[4] = s4[0] + st6;
    r[5] = s4[1] - st5;
    r[6] = s4[0] - st6;
    r[7] = st5 + s4[1];
    float w[8];
    w[0] = r[0] + r[3];
    w[1] = r[1] + r[2];
    w[2] = r[1] - r[2];
    w[3] = r[0] - r[3];
    tmp = DCT_C3A * (r[4] + r[7]);
    w[4] = tmp - (DCT_C3A + DCT_C3B) * r[7];
    w[7] = tmp + (DCT_C3B - DCT_C3A) * r[4];
    tmp = DCT_C1A * (r[5] + r[6]);
    w[5] = tmp - (DCT_C1A + DCT_C1B) * r[6];
    w[6] = tmp + (DCT_C1B - DCT_C1A) * r[5];
    v[0] = w[0] + w[7];
    v[1] = w[1] + w[6];
    v[2] = w[2] + w[5];
    v[3] = w[3] + w[4];
    v[4] = w[3] - w[4];
    v[5] = w[2] - w[5];
    v[6] = w[1] - w[6];
    v[7] = w[0] - w[7];
}

// 2D DCT-8 on an 8x8 block: rows then columns, then normalise.
inline void dct2d(float blk[BLOCK_SIZE][BLOCK_SIZE]) {
    #pragma unroll 8
    for (int j = 0; j < BLOCK_SIZE; j++) dct8_row(blk[j]);
    #pragma unroll 8
    for (int i = 0; i < BLOCK_SIZE; i++) {
        float col[8];
        #pragma unroll 8
        for (int j = 0; j < BLOCK_SIZE; j++) col[j] = blk[j][i];
        dct8_row(col);
        #pragma unroll 8
        for (int j = 0; j < BLOCK_SIZE; j++) blk[j][i] = col[j] * DCT_NORM_2D;
    }
}

inline void idct2d(float blk[BLOCK_SIZE][BLOCK_SIZE]) {
    #pragma unroll 8
    for (int j = 0; j < BLOCK_SIZE; j++) idct8_row(blk[j]);
    #pragma unroll 8
    for (int i = 0; i < BLOCK_SIZE; i++) {
        float col[8];
        #pragma unroll 8
        for (int j = 0; j < BLOCK_SIZE; j++) col[j] = blk[j][i];
        idct8_row(col);
        #pragma unroll 8
        for (int j = 0; j < BLOCK_SIZE; j++) blk[j][i] = col[j] * DCT_NORM_2D;
    }
}

// 8-point Haar transform (in/out separate, Sampas-style).
inline void haar8(float x[8], float y[8]) {
    int k = 8;
    for (int j = 0; j < 3; j++) {
        int k2 = k;
        k >>= 1;
        for (int i = 0; i < k; i++) {
            int i2 = i << 1;
            int i21 = i2 + 1;
            y[i]     = (x[i2] + x[i21]) * HAAR_INV_SQRT2;
            y[i + k] = (x[i2] - x[i21]) * HAAR_INV_SQRT2;
        }
        for (int i = 0; i < k2; i++) x[i] = y[i];
    }
}

inline void ihaar8(float x[8], float y[8]) {
    int k = 1;
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < k; i++) {
            int i2 = i << 1;
            int ik = i + k;
            y[i2]     = (x[i] + x[ik]) * HAAR_INV_SQRT2;
            y[i2 + 1] = (x[i] - x[ik]) * HAAR_INV_SQRT2;
        }
        k <<= 1;
        for (int i = 0; i < k; i++) x[i] = y[i];
    }
}

// kernel_bm3d_match: for each reference patch, find up to group_size
// most-similar blocks within a (2*bm_range+1)^2 window. SSD distance
// metric. Sorted insertion. Emits similar_coords[ref * G * 2] + block
// count. Global work = (ref_count_x, ref_count_y).
__kernel void kernel_bm3d_match(
    const __global uchar *src, const int srcPitch,
#if TEMPORAL
    const __global uchar *noisy_ring, const int noisy_ring_pitch,
    const int noisy_ring_slot_stride,
    const int ring_cursor, const int ring_radius, const int ring_filled,
#endif
    const int W, const int H,
    const int ref_count_x, const int ref_count_y,
    __global short *similar_coords,
#if TEMPORAL
    __global uchar *similar_frame_idx,
#endif
    __global uchar *block_counts,
    const int block_step, const int bm_range,
    const int group_size, const int dist_threshold) {

    const int rgx = get_global_id(0);
    const int rgy = get_global_id(1);
    if (rgx >= ref_count_x || rgy >= ref_count_y) return;

    const int rx = rgx * block_step;
    const int ry = rgy * block_step;
    const int ref_id = rgy * ref_count_x + rgx;

    // Read reference patch.
    float ref[BLOCK_SIZE][BLOCK_SIZE];
    #pragma unroll 8
    for (int j = 0; j < BLOCK_SIZE; j++) {
        #pragma unroll 8
        for (int i = 0; i < BLOCK_SIZE; i++) {
            ref[j][i] = read_pixel(src, srcPitch, rx + i, ry + j, W, H);
        }
    }

    int distances[MAX_GROUP_SIZE_BASIC];
    short positions_x[MAX_GROUP_SIZE_BASIC];
    short positions_y[MAX_GROUP_SIZE_BASIC];
#if TEMPORAL
    uchar positions_f[MAX_GROUP_SIZE_BASIC];
#endif
    for (int n = 0; n < MAX_GROUP_SIZE_BASIC; n++) {
        distances[n] = 0x7FFFFFFF;
        positions_x[n] = 0;
        positions_y[n] = 0;
#if TEMPORAL
        positions_f[n] = 0;
#endif
    }
    int count = 0;

    const int gcap = min(group_size, MAX_GROUP_SIZE_BASIC);

    // 時間版では現在フレームと利用可能な過去フレームを同じ探索経路へ流す。
#if TEMPORAL
    const int temporal_count = ring_filled;
#else
    const int temporal_count = 0;
#endif
    for (int k_back = 0; k_back <= temporal_count; k_back++) {
        const __global uchar *frame_src = src;
        int frame_pitch = srcPitch;
#if TEMPORAL
        if (k_back > 0) {
            const int slot = (ring_cursor + ring_radius - k_back) % ring_radius;
            frame_src = noisy_ring + slot * noisy_ring_slot_stride;
            frame_pitch = noisy_ring_pitch;
        }
#endif
        for (int wy = -bm_range; wy <= bm_range; wy++) {
            for (int wx = -bm_range; wx <= bm_range; wx++) {
            int d = 0;
            #pragma unroll 8
            for (int j = 0; j < BLOCK_SIZE; j++) {
                #pragma unroll 8
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    const float p = read_pixel(frame_src, frame_pitch, rx + wx + i, ry + wy + j, W, H);
                    const float diff = ref[j][i] - p;
                    d += (int)(diff * diff);
                }
            }

            if (d > dist_threshold) continue;

            // Sorted insert.
            for (int n = 0; n < gcap; n++) {
                if (d < distances[n]) {
                    for (int k = gcap - 1; k > n; k--) {
                        distances[k] = distances[k - 1];
                        positions_x[k] = positions_x[k - 1];
                        positions_y[k] = positions_y[k - 1];
#if TEMPORAL
                        positions_f[k] = positions_f[k - 1];
#endif
                    }
                    distances[n] = d;
                    positions_x[n] = (short)wx;
                    positions_y[n] = (short)wy;
#if TEMPORAL
                    positions_f[n] = (uchar)k_back;
#endif
                    if (count < gcap) count++;
                    break;
                }
            }
        }
    }
    }

    block_counts[ref_id] = (uchar)count;
    const int base = ref_id * gcap * 2;
    for (int n = 0; n < count; n++) {
        similar_coords[base + n * 2]     = positions_x[n];
        similar_coords[base + n * 2 + 1] = positions_y[n];
#if TEMPORAL
        similar_frame_idx[ref_id * gcap + n] = positions_f[n];
#endif
    }
}

// kernel_bm3d_basic: collaborative hard-threshold filter. Per reference
// patch, build stack from input + matched-block coords; 2D DCT per
// block; 1D Haar across the stack (depth); hard-threshold at tau_1d;
// inverse Haar; inverse 2D DCT; scatter to global accumulator + weight
// map with Kaiser weights and atomic float add.
__kernel void kernel_bm3d_basic(
    const __global uchar *src, const int srcPitch,
#if TEMPORAL
    const __global uchar *noisy_ring, const int noisy_ring_pitch,
    const int noisy_ring_slot_stride,
    const int ring_cursor, const int ring_radius,
#endif
    const int W, const int H,
    const int ref_count_x, const int ref_count_y,
    const __global short *similar_coords,
#if TEMPORAL
    const __global uchar *similar_frame_idx,
#endif
    const __global uchar *block_counts,
    __global float *accumulator, const int accPitch,
    __global float *weight_map,  const int wmapPitch,
    const int block_step, const int group_size,
    const float sigma_scaled, const float tau_1d) {

    const int rgx = get_global_id(0);
    const int rgy = get_global_id(1);
    if (rgx >= ref_count_x || rgy >= ref_count_y) return;

    const int rx = rgx * block_step;
    const int ry = rgy * block_step;
    const int ref_id = rgy * ref_count_x + rgx;
    const int gcap = min(group_size, MAX_GROUP_SIZE_BASIC);

    const int block_count = (int)block_counts[ref_id];
    if (block_count == 0) return;

    // Build the stack and DCT each block in-place.
    float stack[MAX_GROUP_SIZE_BASIC][BLOCK_SIZE][BLOCK_SIZE];
    const int base = ref_id * gcap * 2;
    for (int n = 0; n < block_count; n++) {
        const int sx = (int)similar_coords[base + n * 2];
        const int sy = (int)similar_coords[base + n * 2 + 1];
        const __global uchar *frame_src = src;
        int frame_pitch = srcPitch;
#if TEMPORAL
        const int sf = (int)similar_frame_idx[ref_id * gcap + n];
        if (sf > 0) {
            const int slot = (ring_cursor + ring_radius - sf) % ring_radius;
            frame_src = noisy_ring + slot * noisy_ring_slot_stride;
            frame_pitch = noisy_ring_pitch;
        }
#endif
        #pragma unroll 8
        for (int j = 0; j < BLOCK_SIZE; j++) {
            #pragma unroll 8
            for (int i = 0; i < BLOCK_SIZE; i++) {
                stack[n][j][i] = read_pixel(frame_src, frame_pitch, rx + sx + i, ry + sy + j, W, H);
            }
        }
        dct2d(stack[n]);
    }

    // 1D Haar across the stack + hard-threshold + inverse Haar.
    // Process in slabs of 8 (Haar-8). For block_count < 8, the upper
    // entries stay zero (and are filtered out as below-threshold).
    int retained = 0;
    #pragma unroll 8
    for (int j = 0; j < BLOCK_SIZE; j++) {
        #pragma unroll 8
        for (int i = 0; i < BLOCK_SIZE; i++) {
            int left = block_count;
            int k = 0;
            while (left > 0) {
                // 'slab' carries an 8-element vertical slice of the
                // matched-block stack at fixed (j, i). Cannot be named
                // 'pipe' because that identifier is reserved in OpenCL
                // C 2.0+ (built-in pipe type qualifier).
                float slab[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                float tpipe[8];
                const int take = min(left, 8);
                for (int n = 0; n < take; n++) slab[n] = stack[k * 8 + n][j][i];
                haar8(slab, tpipe);
                #pragma unroll 8
                for (int n = 0; n < 8; n++) {
                    if (fabs(tpipe[n]) <= tau_1d) {
                        tpipe[n] = 0.0f;
                    } else {
                        retained++;
                    }
                }
                ihaar8(tpipe, slab);
                for (int n = 0; n < take; n++) stack[k * 8 + n][j][i] = slab[n];
                k++;
                left -= 8;
            }
        }
    }

    // Group weight: inverse of (sigma^2 * retained_coefs). Falls back
    // to 1.0 when nothing was retained (rare with sensible sigma).
    const float wx = (retained >= 1) ? (1.0f / (sigma_scaled * sigma_scaled * (float)retained)) : 1.0f;

    // Inverse 2D DCT + scatter to accumulator. Each pixel contributes
    // block_value * wx * kaiser to the accumulator; weight_map gets
    // wx * kaiser. Out-of-bounds pixels are skipped via clamp guard.
    for (int n = 0; n < block_count; n++) {
        idct2d(stack[n]);
#if TEMPORAL
        if (similar_frame_idx[ref_id * gcap + n] != 0) continue;
#endif
        const int sx = (int)similar_coords[base + n * 2];
        const int sy = (int)similar_coords[base + n * 2 + 1];
        #pragma unroll 8
        for (int j = 0; j < BLOCK_SIZE; j++) {
            const int py = ry + sy + j;
            if (py < 0 || py >= H) continue;
            __global float *acc_row  = (__global float *)((__global char *)accumulator + py * accPitch);
            __global float *wmap_row = (__global float *)((__global char *)weight_map  + py * wmapPitch);
            #pragma unroll 8
            for (int i = 0; i < BLOCK_SIZE; i++) {
                const int px = rx + sx + i;
                if (px < 0 || px >= W) continue;
                const float k_w = c_kaiser[j * BLOCK_SIZE + i];
                const float pix_w = wx * k_w;
                atomic_add_global_float(&acc_row[px],  stack[n][j][i] * pix_w);
                atomic_add_global_float(&wmap_row[px], pix_w);
            }
        }
    }
}

// kernel_bm3d_normalize: per output pixel, write sat(acc / w). When
// weight_map is exactly zero (a pixel no reference patch contributed
// to - only possible at extreme image edges with tight bm_range), the
// source pixel passes through.
__kernel void kernel_bm3d_normalize(
    __global uchar *dst, const int dstPitch,
    const __global uchar *src, const int srcPitch,
    const int W, const int H,
    const __global float *accumulator, const int accPitch,
    const __global float *weight_map,  const int wmapPitch,
    const float pixel_max) {

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W || y >= H) return;

    const __global float *acc_row  = (const __global float *)((const __global char *)accumulator + y * accPitch);
    const __global float *wmap_row = (const __global float *)((const __global char *)weight_map  + y * wmapPitch);
    const float w = wmap_row[x];

    __global Type *dst_row = (__global Type *)(dst + y * dstPitch);
    if (w <= 0.0f) {
        const __global Type *src_row = (const __global Type *)(src + y * srcPitch);
        dst_row[x] = src_row[x];
    } else {
        const float v = acc_row[x] / w;
        const float vc = clamp(v, 0.0f, pixel_max);
        dst_row[x] = (Type)(vc + 0.5f);
    }
}

// kernel_bm3d_normalize_f32: same as kernel_bm3d_normalize but writes
// the basic estimate into a float buffer (no clamp, retains full
// precision so the Wiener step's block-matching and shrinkage have
// the cleanest possible reference).
__kernel void kernel_bm3d_normalize_f32(
    __global float *dst, const int dstPitch,
    const __global uchar *src, const int srcPitch,
    const int W, const int H,
    const __global float *accumulator, const int accPitch,
    const __global float *weight_map,  const int wmapPitch) {

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W || y >= H) return;

    const __global float *acc_row  = (const __global float *)((const __global char *)accumulator + y * accPitch);
    const __global float *wmap_row = (const __global float *)((const __global char *)weight_map  + y * wmapPitch);
    __global float *dst_row = (__global float *)((__global char *)dst + y * dstPitch);
    const float w = wmap_row[x];
    if (w <= 0.0f) {
        const __global Type *src_row = (const __global Type *)(src + y * srcPitch);
        dst_row[x] = (float)src_row[x];
    } else {
        dst_row[x] = acc_row[x] / w;
    }
}

// Helper: read a clamped float pixel from a strided float buffer.
inline float read_pixel_f32(const __global float *src, int srcPitch, int x, int y, int W, int H) {
    x = clamp(x, 0, W - 1);
    y = clamp(y, 0, H - 1);
    const __global float *row = (const __global float *)((const __global char *)src + y * srcPitch);
    return row[x];
}

// kernel_bm3d_match_basic: second-pass block-matching that uses the
// basic estimate (float) as the reference clip. Same sorted-top-K
// strategy as kernel_bm3d_match, but the SSD distance metric runs on
// float values and the threshold is float. The basic estimate has
// lower noise than the noisy original, so the matches are cleaner -
// the standard BM3D rationale for the two-step pipeline.
//
// Output: overwrites the same similar_coords / block_counts buffers
// that the Step 1 match populated; the Wiener step reads them.
__kernel void kernel_bm3d_match_basic(
    const __global float *basic, const int basicPitch,
#if TEMPORAL
    const __global float *basic_ring, const int basic_ring_pitch,
    const int basic_ring_slot_stride,
    const int ring_cursor, const int ring_radius, const int ring_filled,
#endif
    const int W, const int H,
    const int ref_count_x, const int ref_count_y,
    __global short *similar_coords,
#if TEMPORAL
    __global uchar *similar_frame_idx,
#endif
    __global uchar *block_counts,
    const int block_step, const int bm_range,
    const int group_size, const float dist_threshold) {

    const int rgx = get_global_id(0);
    const int rgy = get_global_id(1);
    if (rgx >= ref_count_x || rgy >= ref_count_y) return;

    const int rx = rgx * block_step;
    const int ry = rgy * block_step;
    const int ref_id = rgy * ref_count_x + rgx;

    float ref[BLOCK_SIZE][BLOCK_SIZE];
    #pragma unroll 8
    for (int j = 0; j < BLOCK_SIZE; j++) {
        #pragma unroll 8
        for (int i = 0; i < BLOCK_SIZE; i++) {
            ref[j][i] = read_pixel_f32(basic, basicPitch, rx + i, ry + j, W, H);
        }
    }

    float distances[MAX_GROUP_SIZE_WIENER];
    short positions_x[MAX_GROUP_SIZE_WIENER];
    short positions_y[MAX_GROUP_SIZE_WIENER];
#if TEMPORAL
    uchar positions_f[MAX_GROUP_SIZE_WIENER];
#endif
    for (int n = 0; n < MAX_GROUP_SIZE_WIENER; n++) {
        distances[n] = 1e30f;
        positions_x[n] = 0;
        positions_y[n] = 0;
#if TEMPORAL
        positions_f[n] = 0;
#endif
    }
    int count = 0;

    const int gcap = min(group_size, MAX_GROUP_SIZE_WIENER);

#if TEMPORAL
    const int temporal_count = ring_filled;
#else
    const int temporal_count = 0;
#endif
    for (int k_back = 0; k_back <= temporal_count; k_back++) {
        const __global float *frame_src = basic;
        int frame_pitch = basicPitch;
#if TEMPORAL
        if (k_back > 0) {
            const int slot = (ring_cursor + ring_radius - k_back) % ring_radius;
            frame_src = (const __global float *)((const __global char *)basic_ring + slot * basic_ring_slot_stride);
            frame_pitch = basic_ring_pitch;
        }
#endif
        for (int wy = -bm_range; wy <= bm_range; wy++) {
            for (int wx = -bm_range; wx <= bm_range; wx++) {
            float d = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < BLOCK_SIZE; j++) {
                #pragma unroll 8
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    const float p = read_pixel_f32(frame_src, frame_pitch, rx + wx + i, ry + wy + j, W, H);
                    const float diff = ref[j][i] - p;
                    d += diff * diff;
                }
            }

            if (d > dist_threshold) continue;

            for (int n = 0; n < gcap; n++) {
                if (d < distances[n]) {
                    for (int k = gcap - 1; k > n; k--) {
                        distances[k] = distances[k - 1];
                        positions_x[k] = positions_x[k - 1];
                        positions_y[k] = positions_y[k - 1];
#if TEMPORAL
                        positions_f[k] = positions_f[k - 1];
#endif
                    }
                    distances[n] = d;
                    positions_x[n] = (short)wx;
                    positions_y[n] = (short)wy;
#if TEMPORAL
                    positions_f[n] = (uchar)k_back;
#endif
                    if (count < gcap) count++;
                    break;
                }
            }
        }
    }
    }

    block_counts[ref_id] = (uchar)count;
    const int base = ref_id * gcap * 2;
    for (int n = 0; n < count; n++) {
        similar_coords[base + n * 2]     = positions_x[n];
        similar_coords[base + n * 2 + 1] = positions_y[n];
#if TEMPORAL
        similar_frame_idx[ref_id * gcap + n] = positions_f[n];
#endif
    }
}

// kernel_bm3d_wiener: Final estimate (Step 2). Build TWO stacks per
// reference patch - basic_stack from the basic estimate, noise_stack
// from the noisy original - using the basic-step-refined positions.
// 2D DCT both stacks; 1D Haar across the stack on both; compute
// Wiener shrinkage W = |B|^2 / (|B|^2 + sigma^2) from the basic
// coefficients; multiply the noise coefficients by W; inverse Haar
// + iDCT the filtered noise stack; scatter to accumulator + weight
// map. The group weight is wx = 1 / (sigma^2 * sum(W^2)).
__kernel void kernel_bm3d_wiener(
    const __global uchar *src, const int srcPitch,
#if TEMPORAL
    const __global uchar *noisy_ring, const int noisy_ring_pitch,
    const int noisy_ring_slot_stride,
#endif
    const __global float *basic, const int basicPitch,
#if TEMPORAL
    const __global float *basic_ring, const int basic_ring_pitch,
    const int basic_ring_slot_stride,
    const int ring_cursor, const int ring_radius,
#endif
    const int W, const int H,
    const int ref_count_x, const int ref_count_y,
    const __global short *similar_coords,
#if TEMPORAL
    const __global uchar *similar_frame_idx,
#endif
    const __global uchar *block_counts,
    __global float *accumulator, const int accPitch,
    __global float *weight_map,  const int wmapPitch,
    const int block_step, const int group_size,
    const float sigma_scaled) {

    const int rgx = get_global_id(0);
    const int rgy = get_global_id(1);
    if (rgx >= ref_count_x || rgy >= ref_count_y) return;

    const int rx = rgx * block_step;
    const int ry = rgy * block_step;
    const int ref_id = rgy * ref_count_x + rgx;
    const int gcap = min(group_size, MAX_GROUP_SIZE_WIENER);

    const int block_count = (int)block_counts[ref_id];
    if (block_count == 0) return;

    float noise_stack[MAX_GROUP_SIZE_WIENER][BLOCK_SIZE][BLOCK_SIZE];
    float basic_stack[MAX_GROUP_SIZE_WIENER][BLOCK_SIZE][BLOCK_SIZE];

    const int base = ref_id * gcap * 2;
    for (int n = 0; n < block_count; n++) {
        const int sx = (int)similar_coords[base + n * 2];
        const int sy = (int)similar_coords[base + n * 2 + 1];
        const __global uchar *noise_src = src;
        int noise_pitch = srcPitch;
        const __global float *basic_src = basic;
        int basic_src_pitch = basicPitch;
#if TEMPORAL
        const int sf = (int)similar_frame_idx[ref_id * gcap + n];
        if (sf > 0) {
            const int slot = (ring_cursor + ring_radius - sf) % ring_radius;
            noise_src = noisy_ring + slot * noisy_ring_slot_stride;
            noise_pitch = noisy_ring_pitch;
            basic_src = (const __global float *)((const __global char *)basic_ring + slot * basic_ring_slot_stride);
            basic_src_pitch = basic_ring_pitch;
        }
#endif
        #pragma unroll 8
        for (int j = 0; j < BLOCK_SIZE; j++) {
            #pragma unroll 8
            for (int i = 0; i < BLOCK_SIZE; i++) {
                noise_stack[n][j][i] = read_pixel(noise_src, noise_pitch, rx + sx + i, ry + sy + j, W, H);
                basic_stack[n][j][i] = read_pixel_f32(basic_src, basic_src_pitch, rx + sx + i, ry + sy + j, W, H);
            }
        }
        dct2d(noise_stack[n]);
        dct2d(basic_stack[n]);
    }

    const float sigma_sq = sigma_scaled * sigma_scaled;
    float sumsqr_weights = 0.0f;

    #pragma unroll 8
    for (int j = 0; j < BLOCK_SIZE; j++) {
        #pragma unroll 8
        for (int i = 0; i < BLOCK_SIZE; i++) {
            int left = block_count;
            int k = 0;
            while (left > 0) {
                float noise_pipe[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                float basic_pipe[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                float tr_noise[8];
                float tr_basic[8];
                const int take = min(left, 8);
                for (int n = 0; n < take; n++) {
                    noise_pipe[n] = noise_stack[k * 8 + n][j][i];
                    basic_pipe[n] = basic_stack[k * 8 + n][j][i];
                }
                haar8(noise_pipe, tr_noise);
                haar8(basic_pipe, tr_basic);

                float filt[8];
                #pragma unroll 8
                for (int n = 0; n < 8; n++) {
                    const float b2 = tr_basic[n] * tr_basic[n];
                    const float w  = b2 / (b2 + sigma_sq);
                    sumsqr_weights += w * w;
                    filt[n] = w * tr_noise[n];
                }

                float out_pipe[8];
                ihaar8(filt, out_pipe);
                for (int n = 0; n < take; n++) noise_stack[k * 8 + n][j][i] = out_pipe[n];
                k++;
                left -= 8;
            }
        }
    }

    // Group weight. Sumsqr_weights of zero would mean every Wiener
    // coefficient was exactly zero - essentially no signal at all in
    // any matched block. Fall back to 1.0 (preserves what little
    // contribution this group makes).
    const float wx = (sumsqr_weights > 1e-12f) ? (1.0f / (sigma_sq * sumsqr_weights)) : 1.0f;

    for (int n = 0; n < block_count; n++) {
        idct2d(noise_stack[n]);
#if TEMPORAL
        if (similar_frame_idx[ref_id * gcap + n] != 0) continue;
#endif
        const int sx = (int)similar_coords[base + n * 2];
        const int sy = (int)similar_coords[base + n * 2 + 1];
        #pragma unroll 8
        for (int j = 0; j < BLOCK_SIZE; j++) {
            const int py = ry + sy + j;
            if (py < 0 || py >= H) continue;
            __global float *acc_row  = (__global float *)((__global char *)accumulator + py * accPitch);
            __global float *wmap_row = (__global float *)((__global char *)weight_map  + py * wmapPitch);
            #pragma unroll 8
            for (int i = 0; i < BLOCK_SIZE; i++) {
                const int px = rx + sx + i;
                if (px < 0 || px >= W) continue;
                const float k_w = c_kaiser[j * BLOCK_SIZE + i];
                const float pix_w = wx * k_w;
                atomic_add_global_float(&acc_row[px],  noise_stack[n][j][i] * pix_w);
                atomic_add_global_float(&wmap_row[px], pix_w);
            }
        }
    }
}

