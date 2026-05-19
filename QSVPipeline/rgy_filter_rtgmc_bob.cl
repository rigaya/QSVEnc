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

static inline int readPix(
    const __global uchar *plane, int x, int y,
    const int pitch, const int width, const int height
) {
    x = clamp(x, 0, width  - 1);
    y = clamp(y, 0, height - 1);
    return (int)(*(const __global Type *)(plane + y * pitch + x * sizeof(Type)));
}

// Mitchell & Netravali, "Reconstruction Filters in Computer Graphics", SIGGRAPH 1988.
static inline float mitchell_netravali_weight(const float x, const float b, const float c) {
    const float ax = fabs(x);
    if (ax < 1.0f) {
        return ((12.0f - 9.0f * b - 6.0f * c) * ax * ax * ax
              + (-18.0f + 12.0f * b + 6.0f * c) * ax * ax
              + (6.0f - 2.0f * b)) / 6.0f;
    }
    if (ax < 2.0f) {
        return ((-b - 6.0f * c) * ax * ax * ax
              + (6.0f * b + 30.0f * c) * ax * ax
              + (-12.0f * b - 48.0f * c) * ax
              + (8.0f * b + 24.0f * c)) / 6.0f;
    }
    return 0.0f;
}

static inline int bobInterpolate(
    const __global uchar *src, const int ix, const int iy,
    const int pitch, const int width, const int height,
    const int preservedParity, const int phaseQuarter
) {
    const int sourceHeight = (height + 1 - preservedParity) >> 1;
    const float cropStart = 0.25f * (float)phaseQuarter;
    const float fieldPosRaw = cropStart - 0.25f + 0.5f * (float)iy;
    const float fieldPos = clamp(fieldPosRaw, 0.0f, (float)(sourceHeight - 1));
    int endTapField = (int)(fieldPosRaw + 2.0f);
    if (endTapField > sourceHeight - 1) {
        endTapField = sourceHeight - 1;
    }
    int firstTapField = endTapField - 3;
    if (firstTapField < 0) {
        firstTapField = 0;
    }
    float weightSum = 0.0f;
    float interp = 0.0f;

    for (int tap = 0; tap < 4; tap++) {
        const int tapField = firstTapField + tap;
        const int tapY = preservedParity + tapField * 2;
        const float tapWeight = mitchell_netravali_weight((float)tapField - fieldPos, 0.0f, 0.5f);
        interp += (float)readPix(src, ix, tapY, pitch, width, height) * tapWeight;
        weightSum += tapWeight;
    }
    return (int)(interp / weightSum + 0.5f);
}

__attribute__((reqd_work_group_size(rtgmc_bob_block_x, rtgmc_bob_block_y, 1)))
__kernel void kernel_rtgmc_bob(
    __global uchar *restrict pDst, const int dstPitch,
    const __global uchar *restrict pSrc, const int srcPitch,
    const int width,
    const int height,
    const int preservedParity,
    const int phaseQuarter
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const int copySource = ((iy & 1) == preservedParity);
    const int value = copySource
        ? readPix(pSrc, ix, iy, srcPitch, width, height)
        : bobInterpolate(pSrc, ix, iy, srcPitch, width, height, preservedParity, phaseQuarter);

    __global Type *dstPix = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    dstPix[0] = (Type)clamp(value, 0, max_val);
}
