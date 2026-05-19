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

#ifndef RGY_FILTER_DEGRAIN_MOTION_SEARCH_SAD_CL
#define RGY_FILTER_DEGRAIN_MOTION_SEARCH_SAD_CL

inline int degrain_motion_search_ref_x(const int blockX, const int step, const int dx) {
    return blockX * step + degrain_floor_div_pel(dx);
}

inline int degrain_motion_search_ref_y(const int blockY, const int step, const int dy) {
    return blockY * step + degrain_floor_div_pel(dy);
}

inline int degrain_motion_search_ref_frac_x(const int dx) {
    const int base = degrain_floor_div_pel(dx);
    return degrain_floor_mod_pel(dx, base);
}

inline int degrain_motion_search_ref_frac_y(const int dy) {
    const int base = degrain_floor_div_pel(dy);
    return degrain_floor_mod_pel(dy, base);
}

inline int degrain_motion_search_ref_sample(
    __global const uchar *ref,
    const int refPitch,
    const int width,
    const int height,
    const int blockX,
    const int blockY,
    const int step,
    const int dx,
    const int dy,
    const int x,
    const int y) {
#if DEGRAIN_PEL <= 1
    return degrain_pixel_load_mirror(
        ref,
        refPitch,
        width,
        height,
        degrain_motion_search_ref_x(blockX, step, dx) + x,
        degrain_motion_search_ref_y(blockY, step, dy) + y);
#else
    return degrain_pixel_load_pel_mirror(
        ref,
        refPitch,
        width,
        height,
        (blockX * step + x) * DEGRAIN_PEL + dx,
        (blockY * step + y) * DEGRAIN_PEL + dy);
#endif
}

inline int degrain_motion_search_ref_is_integer_pel(const int dx, const int dy) {
#if DEGRAIN_PEL <= 1
    (void)dx;
    (void)dy;
    return 1;
#else
    return degrain_motion_search_ref_frac_x(dx) == 0
        && degrain_motion_search_ref_frac_y(dy) == 0;
#endif
}

inline uint degrain_motion_search_calc_sad_luma(
    __global const uchar *cur,
    __global const uchar *ref,
    const int curPitch,
    const int refPitch,
    const int width,
    const int height,
    const int blockX,
    const int blockY,
    const int step,
    const int dx,
    const int dy) {
    const int srcX = blockX * step;
    const int srcY = blockY * step;
    const int refX = degrain_motion_search_ref_x(blockX, step, dx);
    const int refY = degrain_motion_search_ref_y(blockY, step, dy);
    uint sad = 0u;
    for (int y = 0; y < DEGRAIN_BLK_SIZE; y++) {
        for (int x = 0; x < DEGRAIN_BLK_SIZE; x++) {
            const int srcValue = degrain_pixel_load(cur, curPitch, width, height, srcX + x, srcY + y);
            const int refValue = degrain_motion_search_ref_sample(ref, refPitch, width, height, blockX, blockY, step, dx, dy, x, y);
            sad += (uint)abs(srcValue - refValue);
        }
    }
    return sad;
}

inline uint degrain_motion_search_calc_sad_luma_part(
    __global const uchar *cur,
    __global const uchar *ref,
    const int curPitch,
    const int refPitch,
    const int width,
    const int height,
    const int blockX,
    const int blockY,
    const int step,
    const int dx,
    const int dy,
    const int tx) {
    const int srcX = blockX * step;
    const int srcY = blockY * step;
    const int refX = degrain_motion_search_ref_x(blockX, step, dx);
    const int refY = degrain_motion_search_ref_y(blockY, step, dy);
    const int x = tx % DEGRAIN_BLK_SIZE;
    uint sad = 0u;
    for (int y = tx / DEGRAIN_BLK_SIZE; y < DEGRAIN_BLK_SIZE; y += 8) {
        const int srcValue = degrain_pixel_load(cur, curPitch, width, height, srcX + x, srcY + y);
        const int refValue = degrain_motion_search_ref_sample(ref, refPitch, width, height, blockX, blockY, step, dx, dy, x, y);
        sad += (uint)abs(srcValue - refValue);
    }
    return sad;
}

inline uint degrain_motion_search_reduce_group(const uint value) {
    return value;
}

inline uint degrain_motion_search_reduce_candidates(const uint value) {
    return value;
}

#endif
