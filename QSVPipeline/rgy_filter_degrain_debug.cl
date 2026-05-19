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

__kernel void kernel_degrain_debug_mv(
    __global TypePixel *dst,
    const int dst_pitch,
    const int width,
    const int height,
    __global const degrain_mv_t *mv,
    __global const degrain_sad_t *sad,
    const int blocksX,
    const int blocksY,
    const int blockSize,
    const int overlap,
    const int step,
    const int coveredWidth,
    const int coveredHeight) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }

    if (!degrain_is_covered_pixel(x, y, coveredWidth, coveredHeight)) {
        return;
    }

    const int pitch = dst_pitch / (int)sizeof(TypePixel);
    const int block = degrain_primary_block_index(x, y, blocksX, blocksY, step);
    const int blockX = block % blocksX;
    const int blockY = block / blocksX;
    const int localX = clamp(x - degrain_block_origin(blockX, step), 0, blockSize - 1);
    const int localY = clamp(y - degrain_block_origin(blockY, step), 0, blockSize - 1);
    int refDirection = 0;
    int showDy = 0;
    if (DEGRAIN_REFS <= 2) {
        refDirection = ((localY * 2) >= blockSize) ? min(1, DEGRAIN_REFS - 1) : 0;
        showDy = (localX * 2) >= blockSize;
    } else {
        const int halfX = max(blockSize / 2, 1);
        const int halfY = max(blockSize / 2, 1);
        const int quadrantX = (localX >= halfX);
        const int quadrantY = (localY >= halfY);
        const int quadrantWidth = max(quadrantX ? (blockSize - halfX) : halfX, 1);
        const int localQuadrantX = quadrantX ? (localX - halfX) : localX;
        refDirection = clamp(quadrantY * 2 + quadrantX, 0, DEGRAIN_REFS - 1);
        showDy = (localQuadrantX * 2) >= quadrantWidth;
    }
    const degrain_mv_t motion = mv[degrain_ref_index(block, refDirection)];
    const int signedComponent = showDy ? (int)motion.dy : (int)motion.dx;
    const int value = degrain_debug_border(x, y, step)
        ? DEGRAIN_PIXEL_MAX
        : degrain_centered_signed_value(signedComponent);
    dst[y * pitch + x] = degrain_clamp_pixel(value);
}

__kernel void kernel_degrain_debug_sad(
    __global TypePixel *dst,
    const int dst_pitch,
    const int width,
    const int height,
    __global const degrain_mv_t *mv,
    __global const degrain_sad_t *sad,
    const int blocksX,
    const int blocksY,
    const int blockSize,
    const int overlap,
    const int step,
    const int coveredWidth,
    const int coveredHeight) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }

    if (!degrain_is_covered_pixel(x, y, coveredWidth, coveredHeight)) {
        return;
    }

    const int pitch = dst_pitch / (int)sizeof(TypePixel);
    const int block = degrain_primary_block_index(x, y, blocksX, blocksY, step);
    const int blockX = block % blocksX;
    const int blockY = block / blocksX;
    const int localX = clamp(x - degrain_block_origin(blockX, step), 0, blockSize - 1);
    const int localY = clamp(y - degrain_block_origin(blockY, step), 0, blockSize - 1);
    int refDirection = 0;
    if (DEGRAIN_REFS <= 2) {
        refDirection = ((localY * 2) >= blockSize) ? min(1, DEGRAIN_REFS - 1) : 0;
    } else {
        const int halfX = max(blockSize / 2, 1);
        const int halfY = max(blockSize / 2, 1);
        refDirection = clamp((localY >= halfY) * 2 + (localX >= halfX), 0, DEGRAIN_REFS - 1);
    }
    const int sadIndex = degrain_ref_index(block, refDirection);
    const uint sadMix = sad[sadIndex].sad + mv[sadIndex].sad;
    const int value = degrain_debug_border(x, y, step)
        ? DEGRAIN_PIXEL_MAX
        : min(DEGRAIN_PIXEL_MAX, (int)(sadMix >> 4));
    dst[y * pitch + x] = degrain_clamp_pixel(value);
}
