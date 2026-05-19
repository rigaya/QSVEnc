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
