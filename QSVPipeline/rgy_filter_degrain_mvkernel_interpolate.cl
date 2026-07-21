// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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
// ------------------------------------------------------------------------------------------

#ifndef RGY_FILTER_DEGRAIN_MOTION_SEARCH_INTERPOLATE_CL
#define RGY_FILTER_DEGRAIN_MOTION_SEARCH_INTERPOLATE_CL

// pel=2用: 4位相のサブペルプレーン (0=整数, 1=H半ペル, 2=V半ペル, 3=HV半ペル) を
// 事前計算する。各値は degrain_pixel_load_pel_mirror がオンザフライで計算する値と
// 同一 (同じ補間関数を同じ座標で呼ぶ) なので、SAD側の参照はビット一致のまま。
__kernel void kernel_degrain_mv_build_subpel_planes(
    __global const uchar *src,
    const int pitch,
    __global uchar *dst,
    const int planeStride,
    const int width,
    const int height) {
#if DEGRAIN_PEL == 2 && DEGRAIN_PIXEL_BYTES == 1
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int p0 = degrain_pixel_load(src, pitch, width, height, x, y);
    const int p1 = degrain_interp_halfpel_weighted_mirror(src, pitch, width, height, x, y, 1, 0, DEGRAIN_SUBPEL_INTERP);
    const int p2 = degrain_interp_halfpel_weighted_mirror(src, pitch, width, height, x, y, 0, 1, DEGRAIN_SUBPEL_INTERP);
    const int p3 = degrain_interp_halfpel_weighted_mirror(src, pitch, width, height, x, y, 1, 1, DEGRAIN_SUBPEL_INTERP);
    const int idx = y * pitch + x;
    dst[idx] = (uchar)clamp(p0, 0, 255);
    dst[(size_t)planeStride + idx] = (uchar)clamp(p1, 0, 255);
    dst[(size_t)planeStride * 2 + idx] = (uchar)clamp(p2, 0, 255);
    dst[(size_t)planeStride * 3 + idx] = (uchar)clamp(p3, 0, 255);
#else
    (void)src; (void)pitch; (void)dst; (void)planeStride; (void)width; (void)height;
#endif
}

__kernel void kernel_degrain_mv_expand_coarse_vectors(
    __global const degrain_mv_internal_t *srcVectorsFinal,
    __global degrain_mv_internal_t *dstVectors,
    __global degrain_mv_internal_t *dstVectorsPrev,
    __global uint *dstSads,
    const int srcFinalBase,
    const int dstPlaneBase,
    const int dstSadBase,
    const int srcBlockCount,
    const int dstBlockCount,
    const int srcBlocksX,
    const int srcBlocksY,
    const int dstBlocksX) {
    const int block = get_global_id(0);
    if (block >= dstBlockCount) {
        return;
    }
    const int dstX = block % dstBlocksX;
    const int dstY = block / dstBlocksX;
    const int srcX = min(dstX >> 1, srcBlocksX - 1);
    const int srcY = min(dstY >> 1, srcBlocksY - 1);
    const int srcBlock = srcY * srcBlocksX + srcX;
    degrain_mv_internal_t vec = srcVectorsFinal[degrain_motion_search_vec_final_index(srcFinalBase, srcBlockCount, srcBlock)];
    vec.pos_x <<= 1;
    vec.pos_y <<= 1;
    dstVectors[degrain_motion_search_vec_current_index(dstPlaneBase, dstBlockCount, block)] = vec;
    dstVectorsPrev[degrain_motion_search_vec_prev_index(dstPlaneBase, dstBlockCount, block)] = vec;
    dstSads[dstSadBase + block] = vec.sad_metric;
}

#endif
