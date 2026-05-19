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

#if RTGMC_BORDER_BIT_DEPTH <= 8
#define Type uchar
#elif RTGMC_BORDER_BIT_DEPTH <= 16
#define Type ushort
#else
#define Type float
#endif

__kernel void kernel_rtgmc_border_edge(
    __global uchar *__restrict__ ptrDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const __global uchar *__restrict__ ptrSrc,
    const int srcPitch, const int srcWidth, const int srcHeight,
    const int borderRows) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < dstWidth && y < dstHeight) {
        int sy = y - borderRows;
        sy = min(max(sy, 0), srcHeight - 1);
        const __global Type *srcPix = (const __global Type *)(ptrSrc + sy * srcPitch + x * sizeof(Type));
        __global Type *dstPix = (__global Type *)(ptrDst + y * dstPitch + x * sizeof(Type));
        dstPix[0] = srcPix[0];
    }
}

__kernel void kernel_rtgmc_border_crop(
    __global uchar *__restrict__ ptrDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const __global uchar *__restrict__ ptrSrc,
    const int srcPitch, const int srcWidth, const int srcHeight,
    const int cropRows) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < dstWidth && y < dstHeight) {
        const int sy = y + cropRows;
        const __global Type *srcPix = (const __global Type *)(ptrSrc + sy * srcPitch + x * sizeof(Type));
        __global Type *dstPix = (__global Type *)(ptrDst + y * dstPitch + x * sizeof(Type));
        dstPix[0] = srcPix[0];
    }
}
