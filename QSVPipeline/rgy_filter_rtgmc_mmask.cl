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

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

typedef struct {
    uint sad;
    uint srcAvg;
    uint refAvg;
    uint reserved;
} rtgmc_mmask_sad_t;

static inline int rtgmc_mmask_read_pix(
    const __global uchar *src, int x, int y,
    const int pitch, const int width, const int height
) {
    x = clamp(x, 0, width  - 1);
    y = clamp(y, 0, height - 1);
    return (int)(*(const __global Type *)(src + y * pitch + x * sizeof(Type)));
}

static inline void rtgmc_mmask_write_pix(
    __global uchar *dst, int x, int y, const int pitch, const int value
) {
    __global Type *dstPix = (__global Type *)(dst + y * pitch + x * sizeof(Type));
    dstPix[0] = (Type)clamp(value, 0, max_val);
}

static inline int rtgmc_mmask_block_index(
    const int x,
    const int y,
    const int blocksX,
    const int blocksY,
    const int step
) {
    const int clampedStep = max(step, 1);
    const int bx = min(x / clampedStep, blocksX - 1);
    const int by = min(y / clampedStep, blocksY - 1);
    return by * blocksX + bx;
}

static inline float rtgmc_mmask_sad_weight(
    __global const rtgmc_mmask_sad_t *sad,
    const int block,
    const int blockSize,
    const int temporalDirections,
    const float ml,
    const float gamma
) {
    const uint sadValue = sad[block * temporalDirections].sad;
    const float bitScale = (float)(1 << max(bit_depth - 8, 0));
    const float denom = max(ml * (float)(blockSize * blockSize) * bitScale, 1.0f);
    const float normalized = clamp(((float)sadValue * 4.0f) / denom, 0.0f, 1.0f);
#if rtgmc_mmask_use_pow
    return pow(normalized, gamma);
#else
    return normalized;
#endif
}

__attribute__((reqd_work_group_size(rtgmc_mmask_block_x, rtgmc_mmask_block_y, 1)))
__kernel void kernel_rtgmc_mmask_copy(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pSrc, const int srcPitch,
    const int width,
    const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int value = rtgmc_mmask_read_pix((const __global uchar *)pSrc, ix, iy, srcPitch, width, height);
    rtgmc_mmask_write_pix((__global uchar *)pDst, ix, iy, dstPitch, value);
}

__attribute__((reqd_work_group_size(rtgmc_mmask_block_x, rtgmc_mmask_block_y, 1)))
__kernel void kernel_rtgmc_mmask_blend_y(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pSource, const int sourcePitch,
    const __global Type *restrict pEdi, const int ediPitch,
    const int width,
    const int height,
    __global const rtgmc_mmask_sad_t *sad,
    const int blocksX,
    const int blocksY,
    const int coveredWidth,
    const int coveredHeight,
    const int step,
    const int blockSize,
    const int temporalDirections,
    const float ml,
    const float gamma
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const int sourcePix = rtgmc_mmask_read_pix((const __global uchar *)pSource, ix, iy, sourcePitch, width, height);
    const int ediPix = rtgmc_mmask_read_pix((const __global uchar *)pEdi, ix, iy, ediPitch, width, height);
    if (ix >= coveredWidth || iy >= coveredHeight || blocksX <= 0 || blocksY <= 0 || temporalDirections <= 0) {
        rtgmc_mmask_write_pix((__global uchar *)pDst, ix, iy, dstPitch, sourcePix);
        return;
    }

    const int block = rtgmc_mmask_block_index(ix, iy, blocksX, blocksY, step);
    const float weight = rtgmc_mmask_sad_weight(sad, block, blockSize, temporalDirections, ml, gamma);
    const int value = (int)rint((float)sourcePix + ((float)ediPix - (float)sourcePix) * weight);
    rtgmc_mmask_write_pix((__global uchar *)pDst, ix, iy, dstPitch, value);
}
