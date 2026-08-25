// Type
// hist_bit_depth
//
// CLAHE (Contrast Limited Adaptive Histogram Equalization) kernel
// (--vpp-clahe). Clean-room implementation of Zuiderveld 1994
// ('Contrast Limited Adaptive Histogram Equalization', Graphics Gems IV).
//
// Algorithm:
//   1. Divide image into tiles_x x tiles_y rectangular tiles.
//   2. 各タイルについて、bit_depth に応じたヒストグラムを作成する。
//   3. Contrast-limit: clip histogram bins at clipLimit = slope * mean,
//      then redistribute the clipped excess uniformly across all bins.
//   4. Compute the per-tile CDF as the equalisation transform.
//   5. Per output pixel, bilinearly interpolate the 4 nearest tile
//      transforms to suppress block-boundary seams.
//
// Three kernel passes:
//   pass 1 (kernel_clahe_hist):  build per-tile histograms
//   pass 2 (kernel_clahe_cdf):   contrast-limit + CDF -> transform table
//   pass 3 (kernel_clahe_apply): bilinear apply transform to output

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

#if hist_bit_depth > 10
#define CLAHE_BIN_BIT_DEPTH 10
#else
#define CLAHE_BIN_BIT_DEPTH hist_bit_depth
#endif
#define CLAHE_BINS (1 << CLAHE_BIN_BIT_DEPTH)
#define CLAHE_MAX_VALUE ((1 << storage_bit_depth) - 1)

// Pass 2: serial per-tile contrast-limit clip + CDF -> transform table.
// One work-item per tile.
__kernel void kernel_clahe_cdf(
    __global ushort *restrict pTransform,
    __global uint  *restrict pHist,
    const int width, const int height,
    const int tilesX, const int tilesY,
    const float slope) {
    const int tx = get_global_id(0);
    const int ty = get_global_id(1);
    if (tx >= tilesX || ty >= tilesY) return;

    __global uint  *hist = pHist      + (ty * tilesX + tx) * CLAHE_BINS;
    __global ushort *xfrm = pTransform + (ty * tilesX + tx) * CLAHE_BINS;

    const int tile_x0 = (tx       * width)  / tilesX;
    const int tile_x1 = ((tx + 1) * width)  / tilesX;
    const int tile_y0 = (ty       * height) / tilesY;
    const int tile_y1 = ((ty + 1) * height) / tilesY;
    const int tile_pixels = (tile_x1 - tile_x0) * (tile_y1 - tile_y0);

    if (tile_pixels == 0) {
        const uint maxValue = (uint)CLAHE_MAX_VALUE;
        for (int i = 0; i < CLAHE_BINS; i++) {
            xfrm[i] = (ushort)(((uint)i * maxValue + (CLAHE_BINS - 1) / 2) / (CLAHE_BINS - 1));
        }
        return;
    }

    // clipLimit = slope * mean_bin_count = slope * (tile_pixels / N_bins).
    int clipLimit = (int)(slope * (float)tile_pixels * (1.0f / (float)CLAHE_BINS));
    if (clipLimit < 1) clipLimit = 1;

    // Clip + accumulate excess.
    uint excess = 0u;
    for (int i = 0; i < CLAHE_BINS; i++) {
        if (hist[i] > (uint)clipLimit) {
            excess  += hist[i] - (uint)clipLimit;
            hist[i]  = (uint)clipLimit;
        }
    }
    // Redistribute uniformly. Residual (modulo CLAHE_BINS) is small and
    // ignored -- the paper's iterative redistribution adds minimal
    // visual quality for substantial extra cost.
    const uint redist = excess / CLAHE_BINS;
    for (int i = 0; i < CLAHE_BINS; i++) hist[i] += redist;

    // CDFから格納形式の最大値までの変換テーブルを作る。
    uint cum = 0u;
    const float max_val = (float)CLAHE_MAX_VALUE;
    const float scale = max_val / (float)tile_pixels;
    for (int i = 0; i < CLAHE_BINS; i++) {
        cum += hist[i];
        float v = (float)cum * scale + 0.5f;
        if (v < 0.0f)   v = 0.0f;
        if (v > max_val) v = max_val;
        xfrm[i] = (ushort)v;
    }
}

// ヒストグラムを複製してSLM上のatomic競合を減らし、最後に合算する。
// 10bitでは複製数4としてSLMを16KBに抑え、work-groupの同時実行数低下を避ける。
// 1080pの実測では単一ヒストグラム比で9%～48%高速かつbit-exact。

#if CLAHE_BIN_BIT_DEPTH > 8
#define CLAHE_NUM_SUBHIST 4
#else
#define CLAHE_NUM_SUBHIST 8
#endif

__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void kernel_clahe_hist(
    __global uint *restrict pHist,
    const __global uchar *restrict pSrc, const int srcPitch,
    const int width, const int height,
    const int tilesX, const int tilesY) {
    const int tx = get_group_id(0);
    const int ty = get_group_id(1);
    if (tx >= tilesX || ty >= tilesY) return;

    __local uint subHist[CLAHE_NUM_SUBHIST][CLAHE_BINS];
    const int lid    = get_local_id(1) * get_local_size(0) + get_local_id(0);
    const int wgSize = get_local_size(0) * get_local_size(1);
    const int subId  = lid & (CLAHE_NUM_SUBHIST - 1);   // lid % N (N is pow2)

    // Zero all sub-histograms.
    for (int i = lid; i < CLAHE_NUM_SUBHIST * CLAHE_BINS; i += wgSize) {
        ((__local uint *)subHist)[i] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int tile_x0 = (tx       * width)  / tilesX;
    const int tile_x1 = ((tx + 1) * width)  / tilesX;
    const int tile_y0 = (ty       * height) / tilesY;
    const int tile_y1 = ((ty + 1) * height) / tilesY;

    // 格納ビット深度とヒストグラムの実効ビット深度が異なる場合だけ、
    // 下位側を落としてbin番号へ変換する。
    for (int py = tile_y0 + get_local_id(1); py < tile_y1; py += get_local_size(1)) {
        const __global Type *row = (const __global Type *)(pSrc + py * srcPitch);
        for (int px = tile_x0 + get_local_id(0); px < tile_x1; px += get_local_size(0)) {
            int v = (int)row[px];
#if storage_bit_depth > CLAHE_BIN_BIT_DEPTH
            v >>= (storage_bit_depth - CLAHE_BIN_BIT_DEPTH);
#endif
            if (v > CLAHE_BINS - 1) v = CLAHE_BINS - 1;
            atomic_inc(&subHist[subId][v]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // bin数がwork-groupサイズを超える場合も全binを回収する。
    __global uint *gHist = pHist + (ty * tilesX + tx) * CLAHE_BINS;
    for (int b = lid; b < CLAHE_BINS; b += wgSize) {
        uint sum = 0u;
        for (int s = 0; s < CLAHE_NUM_SUBHIST; s++) {
            sum += subHist[s][b];
        }
        gHist[b] = sum;
    }
}

// Pass 3: bilinear interpolation of 4 nearest tile transforms.
__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void kernel_clahe_apply(
    __global uchar *restrict pDst, const int dstPitch,
    const __global uchar *restrict pSrc, const int srcPitch,
    const int width, const int height,
    const __global ushort *restrict pTransform,
    const int tilesX, const int tilesY) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const __global Type *srcRow = (const __global Type *)(pSrc + y * srcPitch);
    __global       Type *dstRow = (__global       Type *)(pDst + y * dstPitch);

    int bin = (int)srcRow[x];
#if storage_bit_depth > CLAHE_BIN_BIT_DEPTH
    bin >>= (storage_bit_depth - CLAHE_BIN_BIT_DEPTH);
#endif
    if (bin > CLAHE_BINS - 1) bin = CLAHE_BINS - 1;

    // Tile-center grid coordinates: pixel (x, y) sits in cell (fx, fy)
    // of the tilesX-1 by tilesY-1 grid of tile-center quads.
    const float tileW = (float)width  / (float)tilesX;
    const float tileH = (float)height / (float)tilesY;
    const float fx = ((float)x + 0.5f) / tileW - 0.5f;
    const float fy = ((float)y + 0.5f) / tileH - 0.5f;
    int   tx0 = (int)floor(fx);
    int   ty0 = (int)floor(fy);
    tx0 = clamp(tx0, 0, tilesX - 1);
    ty0 = clamp(ty0, 0, tilesY - 1);
    const int tx1 = clamp(tx0 + 1, 0, tilesX - 1);
    const int ty1 = clamp(ty0 + 1, 0, tilesY - 1);
    float u = fx - (float)tx0;
    float v = fy - (float)ty0;
    u = clamp(u, 0.0f, 1.0f);
    v = clamp(v, 0.0f, 1.0f);

    const float t00 = (float)pTransform[(ty0 * tilesX + tx0) * CLAHE_BINS + bin];
    const float t10 = (float)pTransform[(ty0 * tilesX + tx1) * CLAHE_BINS + bin];
    const float t01 = (float)pTransform[(ty1 * tilesX + tx0) * CLAHE_BINS + bin];
    const float t11 = (float)pTransform[(ty1 * tilesX + tx1) * CLAHE_BINS + bin];

    const float t = mix(mix(t00, t10, u), mix(t01, t11, u), v);

    const float max_val = (float)CLAHE_MAX_VALUE;
    float out_val = t;
    if (out_val < 0.0f)     out_val = 0.0f;
    if (out_val > max_val)  out_val = max_val;
    dstRow[x] = (Type)(out_val + 0.5f);
}
