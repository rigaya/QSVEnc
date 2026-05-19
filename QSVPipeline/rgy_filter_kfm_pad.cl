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

#ifndef Type
#define Type uchar
#endif

static inline int kfm_mirror_index(const int pos, const int size) {
    if (pos < 0) {
        return -pos - 1;
    }
    if (pos >= size) {
        return size - (pos - size) - 1;
    }
    return pos;
}

__kernel void kernel_kfm_pad(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width,
    const int height,
    const int vpad) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int paddedHeight = height + vpad * 2;
    if (x >= width || y >= paddedHeight) return;

    const int srcY = kfm_mirror_index(y - vpad, height);
    const __global Type *pSrc = (const __global Type *)(src + srcY * srcPitch + x * (int)sizeof(Type));
    __global Type *pDst = (__global Type *)(dst + y * dstPitch + x * (int)sizeof(Type));
    pDst[0] = pSrc[0];
}

__kernel void kernel_kfm_padv_inplace(
    __global uchar *dst,
    const int dstPitch,
    const int width,
    const int height,
    const int vpad) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= vpad) return;

    __global Type *topDst = (__global Type *)(dst + (vpad - y - 1) * dstPitch + x * (int)sizeof(Type));
    __global Type *bottomDst = (__global Type *)(dst + (vpad + height + y) * dstPitch + x * (int)sizeof(Type));
    const __global Type *topSrc = (const __global Type *)(dst + (vpad + y) * dstPitch + x * (int)sizeof(Type));
    const __global Type *bottomSrc = (const __global Type *)(dst + (vpad + height - y - 1) * dstPitch + x * (int)sizeof(Type));
    topDst[0] = topSrc[0];
    bottomDst[0] = bottomSrc[0];
}
