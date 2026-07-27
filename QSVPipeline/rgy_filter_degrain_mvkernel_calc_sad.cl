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

#ifndef RGY_FILTER_DEGRAIN_MOTION_SEARCH_CALC_SAD_CL
#define RGY_FILTER_DEGRAIN_MOTION_SEARCH_CALC_SAD_CL

__kernel void kernel_degrain_mv_export_sad(
    __global degrain_mv_internal_t *vectorsFinal,
    __global uint *sadsInternal,
    __global degrain_mv_t *outputMotion,
    __global degrain_sad_t *outputSad,
    const int finalBase,
    const int sadBase,
    const int blockCount,
    const int outOffset,
    const int referenceDirection) {
    const int block = get_global_id(0);
    if (block >= blockCount) {
        return;
    }

    degrain_mv_internal_t finalVector = vectorsFinal[degrain_motion_search_vec_final_index(finalBase, blockCount, block)];
    const uint finalSad = finalVector.sad_metric;
    finalVector.sad_metric = finalSad;
    finalVector.score_primary = finalSad;
    vectorsFinal[degrain_motion_search_vec_final_index(finalBase, blockCount, block)] = finalVector;
    sadsInternal[sadBase + block] = finalSad;

    const int outputIndex = outOffset + degrain_ref_index(block, referenceDirection);
    if (outputMotion) {
        degrain_mv_t exportedMotion;
        exportedMotion.dx = finalVector.pos_x;
        exportedMotion.dy = finalVector.pos_y;
        exportedMotion.sad = degrain_clamp_u16(finalSad);
        exportedMotion.refdir = (ushort)referenceDirection;
        exportedMotion.flags = 0u;
        exportedMotion.reserved = finalSad;
        outputMotion[outputIndex] = exportedMotion;
    }
    if (outputSad) {
        degrain_sad_t exportedSad;
        exportedSad.sad = finalSad;
        exportedSad.srcAvg = 0u;
        exportedSad.refAvg = 0u;
        exportedSad.reserved = finalSad;
        outputSad[outputIndex] = exportedSad;
    }
}

__kernel void kernel_degrain_mv_add_chroma_sad(
    __global const uchar *curU,
    __global const uchar *curV,
    const int curPitch,
    __global const uchar *refU,
    __global const uchar *refV,
    const int refPitch,
    const int width,
    const int height,
    __global degrain_mv_t *mv,
    __global degrain_sad_t *sad,
    const int blocksX,
    const int blockCount,
    const int blockSize,
    const int step,
    const int planeScaleX,
    const int planeScaleY,
    const int referenceDirection) {
    const int block = get_global_id(0);
    if (block >= blockCount) {
        return;
    }
    const int blockX = block % blocksX;
    const int blockY = block / blocksX;
    const int scaleX = max(planeScaleX, 1);
    const int scaleY = max(planeScaleY, 1);
    const int originX = (blockX * step) / scaleX;
    const int originY = (blockY * step) / scaleY;
    const int sizeX = max(blockSize / scaleX, 1);
    const int sizeY = max(blockSize / scaleY, 1);
    uint chromaSad = 0u;
    for (int y = originY; y < min(originY + sizeY, height); y++) {
        for (int x = originX; x < min(originX + sizeX, width); x++) {
            const int curSampleU = degrain_pixel_load(curU, curPitch, width, height, x, y);
            const int curSampleV = degrain_pixel_load(curV, curPitch, width, height, x, y);
            const int refSampleU = degrain_compensated_sample(
                refU, refPitch, width, height, mv, block, referenceDirection,
                planeScaleX, planeScaleY, x, y);
            const int refSampleV = degrain_compensated_sample(
                refV, refPitch, width, height, mv, block, referenceDirection,
                planeScaleX, planeScaleY, x, y);
            chromaSad += (uint)(abs(curSampleU - refSampleU) + abs(curSampleV - refSampleV));
        }
    }
    const int index = degrain_ref_index(block, referenceDirection);
    const uint combinedSad = sad[index].sad + chromaSad;
    sad[index].sad = combinedSad;
    sad[index].reserved = combinedSad;
    mv[index].sad = degrain_clamp_u16(combinedSad);
    mv[index].reserved = combinedSad;
}

#endif
