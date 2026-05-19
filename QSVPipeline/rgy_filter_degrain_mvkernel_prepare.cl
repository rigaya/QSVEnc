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

#ifndef RGY_FILTER_DEGRAIN_MOTION_SEARCH_PREPARE_CL
#define RGY_FILTER_DEGRAIN_MOTION_SEARCH_PREPARE_CL

__kernel void kernel_degrain_mv_seed_anchor_vectors(
    __global degrain_mv_internal_t *vectors,
    __global const int2 *frameAverageMV,
    const int planeBase,
    const int planeStride,
    const int planeCount) {
    const int plane = get_global_id(0);
    if (plane >= planeCount) {
        return;
    }
    const int base = planeBase + plane * planeStride;
    vectors[degrain_motion_search_vec_zero_index(base)] = degrain_motion_search_make_vector(0, 0, 0u, 0u);
    const int2 frameAverageVec = (frameAverageMV) ? frameAverageMV[plane] : (int2)(0, 0);
    vectors[degrain_motion_search_vec_global_index(base)] = degrain_motion_search_make_vector(
        frameAverageVec.x * DEGRAIN_NPEL,
        frameAverageVec.y * DEGRAIN_NPEL,
        0u,
        0u);
}

__kernel void kernel_degrain_mv_seed_zero_vectors(
    __global degrain_mv_internal_t *vectors,
    __global degrain_mv_internal_t *vectorsPrev,
    __global uint *sads,
    const int planeBase,
    const int sadBase,
    const int blockCount) {
    const int block = get_global_id(0);
    if (block >= blockCount) {
        return;
    }
    const degrain_mv_internal_t zero = vectors[degrain_motion_search_vec_zero_index(planeBase)];
    vectors[degrain_motion_search_vec_current_index(planeBase, blockCount, block)] = zero;
    vectorsPrev[degrain_motion_search_vec_prev_index(planeBase, blockCount, block)] = zero;
    sads[sadBase + block] = zero.sad_metric;
}

#endif
