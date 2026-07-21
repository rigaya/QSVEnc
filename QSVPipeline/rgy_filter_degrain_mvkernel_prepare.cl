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

#ifndef DEGRAIN_MOTION_SEARCH_GLOBAL_REDUCE_SIZE
#define DEGRAIN_MOTION_SEARCH_GLOBAL_REDUCE_SIZE 256
#endif

// level1(coarse)の最終ベクトルの平均をlevel0のGLOBALアンカーに書き込む。
// 平均はcoarse→fineのスケール(x2)を適用してlevel0の内部単位に揃える。
__attribute__((reqd_work_group_size(DEGRAIN_MOTION_SEARCH_GLOBAL_REDUCE_SIZE, 1, 1)))
__kernel void kernel_degrain_mv_seed_global_from_coarse(
    __global degrain_mv_internal_t *dstVectors,
    __global const degrain_mv_internal_t *srcVectorsFinal,
    const int dstPlaneBase,
    const int srcFinalBase,
    const int srcBlockCount) {
    __local long sumX[DEGRAIN_MOTION_SEARCH_GLOBAL_REDUCE_SIZE];
    __local long sumY[DEGRAIN_MOTION_SEARCH_GLOBAL_REDUCE_SIZE];
    const int tid = (int)get_local_id(0);
    long sx = 0, sy = 0;
    for (int i = tid; i < srcBlockCount; i += DEGRAIN_MOTION_SEARCH_GLOBAL_REDUCE_SIZE) {
        const degrain_mv_internal_t vec = srcVectorsFinal[degrain_motion_search_vec_final_index(srcFinalBase, srcBlockCount, i)];
        sx += vec.pos_x;
        sy += vec.pos_y;
    }
    sumX[tid] = sx;
    sumY[tid] = sy;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = DEGRAIN_MOTION_SEARCH_GLOBAL_REDUCE_SIZE >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sumX[tid] += sumX[tid + stride];
            sumY[tid] += sumY[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0 && srcBlockCount > 0) {
        // coarse→fineの2倍を除算前に適用し、0.5 coarse pixel相当の精度を保持する。
        const long roundHalf = (long)srcBlockCount >> 1;
        const long scaledSumX = sumX[0] * 2;
        const long scaledSumY = sumY[0] * 2;
        const long avgX = (scaledSumX >= 0) ? (scaledSumX + roundHalf) / srcBlockCount : -((-scaledSumX + roundHalf) / srcBlockCount);
        const long avgY = (scaledSumY >= 0) ? (scaledSumY + roundHalf) / srcBlockCount : -((-scaledSumY + roundHalf) / srcBlockCount);
        const int globalX = (int)clamp(avgX, (long)-32768, (long)32767);
        const int globalY = (int)clamp(avgY, (long)-32768, (long)32767);
        dstVectors[degrain_motion_search_vec_global_index(dstPlaneBase)] = degrain_motion_search_make_vector(
            globalX,
            globalY,
            0u,
            0u);
    }
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
