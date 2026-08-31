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
    __global uint *out_partials
) {
    // 1 work group内の各チャンネル合計は32*8*65535で約1.7e7に留まり、
    // uintの上限を超えないため、部分和を32bit化しても厳密性は失われない。
    __local uint sh0[softlight_block_x * softlight_block_y];
    __local uint sh1[softlight_block_x * softlight_block_y];
    __local uint sh2[softlight_block_x * softlight_block_y];
    __local uint sh3[softlight_block_x * softlight_block_y];
    __local uint sh4[softlight_block_x * softlight_block_y];
    __local uint sh5[softlight_block_x * softlight_block_y];

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int lid = get_local_id(1) * softlight_block_x + get_local_id(0);
    uint sumR = 0, sumG = 0, sumB = 0;
    uint blackR = 0, blackG = 0, blackB = 0;
    if (x < width && y < height) {
        const ushort r = *((const __global ushort *)(pR + y * pitchR + x * sizeof(ushort)));
        const ushort g = *((const __global ushort *)(pG + y * pitchG + x * sizeof(ushort)));
        const ushort b = *((const __global ushort *)(pB + y * pitchB + x * sizeof(ushort)));
        sumR = (uint)r; sumG = (uint)g; sumB = (uint)b;
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

// boost以外の各モードを、フル解像度の中間バッファを介さずレジスタ内で処理する。
// 旧処理が16bitプレーンへ中間値を書いていた地点では、同じsoftlight_to_u16()で
// 再量子化してから後段へ渡す。丸め位置を旧処理と合わせることでビット一致を保つ。
__kernel void kernel_softlight_fused_u16(
    __global uchar *pR, const int pitchR,
    __global uchar *pG, const int pitchG,
    __global uchar *pB, const int pitchB,
    const int width, const int height,
    const int mode,
    const __global float *bVals,
    const int formula
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < width && y < height) {
        // saturationは強度を参照しないが、分岐を単純に保つため先に読み込む。
        const float bR = bVals[0];
        const float bG = bVals[1];
        const float bB = bVals[2];
        __global ushort *ptrR = (__global ushort *)(pR + y * pitchR + x * sizeof(ushort));
        __global ushort *ptrG = (__global ushort *)(pG + y * pitchG + x * sizeof(ushort));
        __global ushort *ptrB = (__global ushort *)(pB + y * pitchB + x * sizeof(ushort));
        const float r = (float)ptrR[0] * (1.0f / 65535.0f);
        const float g = (float)ptrG[0] * (1.0f / 65535.0f);
        const float b = (float)ptrB[0] * (1.0f / 65535.0f);
        float ro = r, go = g, bo = b;
        if (mode == SOFTLIGHT_MODE_NEUTRALIZE) {
            const float vOrig = fmax(r, fmax(g, b));
            const float rm = (float)softlight_to_u16(softlight_func(r, bR, formula)) * (1.0f / 65535.0f);
            const float gm = (float)softlight_to_u16(softlight_func(g, bG, formula)) * (1.0f / 65535.0f);
            const float bm = (float)softlight_to_u16(softlight_func(b, bB, formula)) * (1.0f / 65535.0f);
            float h, sVal, vMod;
            rgb_to_hsv_value(rm, gm, bm, &h, &sVal, &vMod);
            hsv_to_rgb_value(h, sVal, vOrig, &ro, &go, &bo);
        } else if (mode == SOFTLIGHT_MODE_LIGHTNESS) {
            float h, sVal, vOrig;
            rgb_to_hsv_value(r, g, b, &h, &sVal, &vOrig);
            const float rm = (float)softlight_to_u16(softlight_func(r, bR, formula)) * (1.0f / 65535.0f);
            const float gm = (float)softlight_to_u16(softlight_func(g, bG, formula)) * (1.0f / 65535.0f);
            const float bm = (float)softlight_to_u16(softlight_func(b, bB, formula)) * (1.0f / 65535.0f);
            const float vMod = fmax(rm, fmax(gm, bm));
            hsv_to_rgb_value(h, sVal, vMod, &ro, &go, &bo);
        } else if (mode == SOFTLIGHT_MODE_NEUTRALIZE_BOOST_SAT) {
            const float vOrig = fmax(r, fmax(g, b));
            const float rm = (float)softlight_to_u16(softlight_func(r, bR, formula)) * (1.0f / 65535.0f);
            const float gm = (float)softlight_to_u16(softlight_func(g, bG, formula)) * (1.0f / 65535.0f);
            const float bm = (float)softlight_to_u16(softlight_func(b, bB, formula)) * (1.0f / 65535.0f);
            float h, sVal, vMod;
            rgb_to_hsv_value(rm, gm, bm, &h, &sVal, &vMod);
            const float sB = fmin(fmax(softlight_func(sVal, sVal, formula), 0.0f), 1.0f);
            hsv_to_rgb_value(h, sB, vOrig, &ro, &go, &bo);
        } else if (mode == SOFTLIGHT_MODE_NEUTRALIZE_FULL) {
            ro = softlight_func(r, bR, formula);
            go = softlight_func(g, bG, formula);
            bo = softlight_func(b, bB, formula);
        } else if (mode == SOFTLIGHT_MODE_NEUTRALIZE_BOOST) {
            const float rm = (float)softlight_to_u16(softlight_func(r, bR, formula)) * (1.0f / 65535.0f);
            const float gm = (float)softlight_to_u16(softlight_func(g, bG, formula)) * (1.0f / 65535.0f);
            const float bm = (float)softlight_to_u16(softlight_func(b, bB, formula)) * (1.0f / 65535.0f);
            ro = softlight_func(rm, rm, formula);
            go = softlight_func(gm, gm, formula);
            bo = softlight_func(bm, bm, formula);
        } else { // SOFTLIGHT_MODE_SATURATION
            float h, sVal, vOrig;
            rgb_to_hsv_value(r, g, b, &h, &sVal, &vOrig);
            const float sB = fmin(fmax(softlight_func(sVal, sVal, formula), 0.0f), 1.0f);
            hsv_to_rgb_value(h, sB, vOrig, &ro, &go, &bo);
        }
        ptrR[0] = softlight_to_u16(ro);
        ptrG[0] = softlight_to_u16(go);
        ptrB[0] = softlight_to_u16(bo);
    }
}

// work groupごとの部分和から強度をデバイス上で算出し、フレームごとの
// readbackとwaitによるOpenCLキューのドレインを除去する。
__kernel void kernel_softlight_finalize_b(
    const __global uint *partials, const int numGroups,
    const long totalPx, const int skipblack,
    __global float *bVals
) {
    __local long shSum[3][softlight_finalize_threads];
    // 黒画素数は全画素を合計してもuintに収まるため、SLM使用量を抑える。
    __local uint shBlack[3][softlight_finalize_threads];
    const int tid = get_local_id(0);
    long sum[3] = { 0, 0, 0 };
    uint black[3] = { 0, 0, 0 };
    for (int g = tid; g < numGroups; g += softlight_finalize_threads) {
        for (int i = 0; i < 3; i++) {
            sum[i] += (long)partials[g * 6 + i];
            black[i] += partials[g * 6 + 3 + i];
        }
    }
    for (int i = 0; i < 3; i++) {
        shSum[i][tid] = sum[i];
        shBlack[i][tid] = black[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = softlight_finalize_threads >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            for (int i = 0; i < 3; i++) {
                shSum[i][tid] += shSum[i][tid + offset];
                shBlack[i][tid] += shBlack[i][tid + offset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        for (int i = 0; i < 3; i++) {
            const long denom = totalPx - (skipblack ? (long)shBlack[i][0] : 0L);
            const float mean = (denom > 0) ? ((float)shSum[i][0] / (float)denom) * (1.0f / 65535.0f) : 0.0f;
            bVals[i] = 1.0f - mean;
        }
    }
}
