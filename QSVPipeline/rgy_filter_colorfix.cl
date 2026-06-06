// ColorFix — colour correction kernels.
//
// Pipeline (orchestrated by rgy_filter_colorfix.cpp):
//   mode=manual: input YUV → CspCrop → RGB → colorfix_apply_rgb → CspCrop → YUV
//   mode=auto:   YUV in-place; reduce_uv during analysis; apply_uv afterwards
//   mode=gray:   input YUV → CspCrop → RGB → reduce_rgb during analysis;
//                colorfix_apply_rgb afterwards; CspCrop → YUV
//
// The reduction kernels write per-work-group partial sums as `long`
// values into a packed buffer. The host reduces these to global totals
// and computes the correction parameters.
//
// Build-time defines (set via -D from rgy_filter_colorfix.cpp):
//   Type           : uchar (8-bit) or ushort (>8-bit)         [YUV plane element]
//   RGB processing uses planar R/G/B planes.
//   bit_depth      : source bit depth
//   max_val        : (1 << bit_depth) - 1
//   colorfix_block_x / colorfix_block_y : work-group dims
//
// =============================================================================

// ---------------------------------------------------------------------------
// 1. colorfix_apply_rgb — per-channel lift/gain in RGB space.
//
// scaleR/G/B and offsetR/G/B are pre-computed on the host such that
//   out_channel = clamp(in_channel * scale + offset, 0, max_val)
// works directly in the target bit depth. Operates on planar RGB planes.
//
// For mode=manual:
//   scale_R  = max_val / (whiteR_hbd - blackR_hbd)
//   offset_R = -blackR_hbd * scale_R
// For mode=gray:
//   scale_R  = (meanAll / meanR) * strength + (1.0 - strength)
//   offset_R = 0
// ---------------------------------------------------------------------------
__kernel void colorfix_apply_rgb(
    __global uchar *pR,
    __global uchar *pG,
    __global uchar *pB, int pitch,
    int width, int height,
    float scaleR, float scaleG, float scaleB,
    float offsetR, float offsetG, float offsetB
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    __global Type *rPix = (__global Type *)(pR + y * pitch + x * sizeof(Type));
    __global Type *gPix = (__global Type *)(pG + y * pitch + x * sizeof(Type));
    __global Type *bPix = (__global Type *)(pB + y * pitch + x * sizeof(Type));

    // CRITICAL: do the multiply in float, not in the narrow integer type.
    // (white-black) can be small on heavily tinted source → 8-bit
    // intermediates would crush shadows / blow highlights before clamping.
    float r = (float)rPix[0] * scaleR + offsetR;
    float g = (float)gPix[0] * scaleG + offsetG;
    float b = (float)bPix[0] * scaleB + offsetB;

    int ir = convert_int_rte(r);
    int ig = convert_int_rte(g);
    int ib = convert_int_rte(b);

    if (ir < 0) ir = 0; if (ir > max_val) ir = max_val;
    if (ig < 0) ig = 0; if (ig > max_val) ig = max_val;
    if (ib < 0) ib = 0; if (ib > max_val) ib = max_val;

    rPix[0] = (Type)ir;
    gPix[0] = (Type)ig;
    bPix[0] = (Type)ib;
}

// ---------------------------------------------------------------------------
// 2. colorfix_reduce_uv — for mode=auto.
//
// Emits 4 longs per work-group: sum_U, sum_V, sum_Y, sum_Y².
// Host divides by pixel count to recover means + variance, then computes
// chroma offsets and checks the variance guard.
//
// Operates on the U, V, and Y planes separately. Dispatch shape is
// (chroma_width × chroma_height) — each work-item reads one U + one V
// sample, plus the four luma samples covered by this 2×2 chroma cell
// (so the Y reduction is over the full luma resolution).
//
// For NV12 input the chroma planes are interleaved; we still pass them
// as separate plane pointers because QSVEnc's getPlane() abstracts that
// (each "plane" is a 1-channel view). For YV12 / YUV420 planar the
// same code works without modification.
// ---------------------------------------------------------------------------
__kernel void colorfix_reduce_uv(
    const __global uchar *pY, int pitchY, int widthY, int heightY,
    const __global uchar *pU, int pitchU, int widthU, int heightU,
    const __global uchar *pV, int pitchV,
    int subX, int subY,           // chroma:luma sample ratio (typically 2,2 for 4:2:0)
    __global long *out_partials   // 4 longs per work-group
) {
    __local long sU[colorfix_block_x * colorfix_block_y];
    __local long sV[colorfix_block_x * colorfix_block_y];
    __local long sY[colorfix_block_x * colorfix_block_y];
    __local long sYsq[colorfix_block_x * colorfix_block_y];

    const int cx = get_global_id(0);
    const int cy = get_global_id(1);
    const int lid = get_local_id(1) * colorfix_block_x + get_local_id(0);

    long uVal = 0, vVal = 0, yAcc = 0, ysqAcc = 0;

    if (cx < widthU && cy < heightU) {
        uVal = (long)(*(const __global Type *)(pU + cy * pitchU + cx * sizeof(Type)));
        vVal = (long)(*(const __global Type *)(pV + cy * pitchV + cx * sizeof(Type)));

        // Accumulate the (subX × subY) luma samples covered by this chroma cell.
        for (int dy = 0; dy < subY; dy++) {
            const int ly = cy * subY + dy;
            if (ly >= heightY) break;
            for (int dx = 0; dx < subX; dx++) {
                const int lx = cx * subX + dx;
                if (lx >= widthY) break;
                const long yv = (long)(*(const __global Type *)(pY + ly * pitchY + lx * sizeof(Type)));
                yAcc   += yv;
                ysqAcc += yv * yv;
            }
        }
    }

    sU[lid]   = uVal;
    sV[lid]   = vVal;
    sY[lid]   = yAcc;
    sYsq[lid] = ysqAcc;

    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree-reduce in local memory.
    const int wgSize = colorfix_block_x * colorfix_block_y;
    for (int s = wgSize >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            sU[lid]   += sU[lid + s];
            sV[lid]   += sV[lid + s];
            sY[lid]   += sY[lid + s];
            sYsq[lid] += sYsq[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        const int groupIdx = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        out_partials[groupIdx * 4 + 0] = sU[0];
        out_partials[groupIdx * 4 + 1] = sV[0];
        out_partials[groupIdx * 4 + 2] = sY[0];
        out_partials[groupIdx * 4 + 3] = sYsq[0];
    }
}

// ---------------------------------------------------------------------------
// 3a. colorfix_apply_luma — for YUV-space manual mode.
//
// Per-pixel:
//   y_out = clamp(y_in * scaleY + offsetY, 0, max_val)
//
// Used by mode=manual when space=yuv to apply the luma component of the
// RGB->YUV-converted white/black point correction directly on the Y
// plane, skipping the YUV<->RGB CspCrop sub-filters entirely.
// ---------------------------------------------------------------------------
__kernel void colorfix_apply_luma(
    __global uchar *pY, int pitchY, int widthY, int heightY,
    float scaleY, float offsetY
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= widthY || y >= heightY) return;

    __global Type *yPix = (__global Type *)(pY + y * pitchY + x * sizeof(Type));
    const float v = (float)yPix[0] * scaleY + offsetY;
    int iv = convert_int_rte(v);
    if (iv < 0)        iv = 0;
    if (iv > max_val)  iv = max_val;
    yPix[0] = (Type)iv;
}

// ---------------------------------------------------------------------------
// 3. colorfix_apply_uv — for mode=auto (post-analysis frames).
//
// Subtract host-computed chroma offsets from U and V planes; luma
// passes through unchanged (we don't even touch the Y plane here —
// it's expected to be left untouched at the framework level, with the
// caller copying the source luma to the output buffer).
// ---------------------------------------------------------------------------
__kernel void colorfix_apply_uv(
    __global uchar *pU, int pitchU,
    __global uchar *pV, int pitchV,
    int widthU, int heightU,
    int offsetU, int offsetV
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= widthU || y >= heightU) return;

    __global Type *uPix = (__global Type *)(pU + y * pitchU + x * sizeof(Type));
    __global Type *vPix = (__global Type *)(pV + y * pitchV + x * sizeof(Type));

    int u = (int)uPix[0] + offsetU;
    int v = (int)vPix[0] + offsetV;

    if (u < 0) u = 0; if (u > max_val) u = max_val;
    if (v < 0) v = 0; if (v > max_val) v = max_val;

    uPix[0] = (Type)u;
    vPix[0] = (Type)v;
}

// ---------------------------------------------------------------------------
// 4. colorfix_reduce_rgb — for mode=gray.
//
// Emits 5 longs per work-group: sum_R, sum_G, sum_B, sum_Y, sum_Y².
// `pR/pG/pB` are planar RGB planes at full source resolution.
// Y is approximated using BT.601 luma weights for the variance guard
// only; the actual luma stays in the original YUV plane and isn't
// touched here.
// ---------------------------------------------------------------------------
__kernel void colorfix_reduce_rgb(
    const __global uchar *pR,
    const __global uchar *pG,
    const __global uchar *pB, int pitch,
    int width, int height,
    __global long *out_partials       // 5 longs per work-group
) {
    __local long sR[colorfix_block_x * colorfix_block_y];
    __local long sG[colorfix_block_x * colorfix_block_y];
    __local long sB[colorfix_block_x * colorfix_block_y];
    __local long sY[colorfix_block_x * colorfix_block_y];
    __local long sYsq[colorfix_block_x * colorfix_block_y];

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int lid = get_local_id(1) * colorfix_block_x + get_local_id(0);

    long rv = 0, gv = 0, bv = 0, yv = 0, ysq = 0;

    if (x < width && y < height) {
        rv = (long)(*(const __global Type *)(pR + y * pitch + x * sizeof(Type)));
        gv = (long)(*(const __global Type *)(pG + y * pitch + x * sizeof(Type)));
        bv = (long)(*(const __global Type *)(pB + y * pitch + x * sizeof(Type)));
        // Y approximation for variance guard (BT.601 weights × 65536):
        //   Y ≈ (19595 R + 38470 G +  7471 B + 32768) >> 16
        // The exact matrix doesn't matter — this is only used to decide
        // whether to skip a frame, not to write to the output.
        const long yLong = (19595L * rv + 38470L * gv + 7471L * bv + 32768L) >> 16;
        yv  = yLong;
        ysq = yLong * yLong;
    }

    sR[lid]   = rv;
    sG[lid]   = gv;
    sB[lid]   = bv;
    sY[lid]   = yv;
    sYsq[lid] = ysq;
    barrier(CLK_LOCAL_MEM_FENCE);

    const int wgSize = colorfix_block_x * colorfix_block_y;
    for (int s = wgSize >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            sR[lid]   += sR[lid + s];
            sG[lid]   += sG[lid + s];
            sB[lid]   += sB[lid + s];
            sY[lid]   += sY[lid + s];
            sYsq[lid] += sYsq[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        const int groupIdx = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        out_partials[groupIdx * 5 + 0] = sR[0];
        out_partials[groupIdx * 5 + 1] = sG[0];
        out_partials[groupIdx * 5 + 2] = sB[0];
        out_partials[groupIdx * 5 + 3] = sY[0];
        out_partials[groupIdx * 5 + 4] = sYsq[0];
    }
}
