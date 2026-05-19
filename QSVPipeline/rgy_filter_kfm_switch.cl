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

#ifndef bit_depth
#define bit_depth 8
#endif

static inline Type kfm_load_pixel(
    const __global uchar *src,
    const int pitch,
    const int x,
    const int y) {
    return ((const __global Type *)(src + y * pitch))[x];
}

static inline void kfm_store_pixel(
    __global uchar *dst,
    const int pitch,
    const int x,
    const int y,
    const Type v) {
    ((__global Type *)(dst + y * pitch))[x] = v;
}

static inline Type kfm_blend_patch(
    const Type base,
    const Type patch,
    const Type mask) {
    const int coef = clamp((int)mask, 0, 128);
    const int invcoef = 128 - coef;
    return (Type)((coef * (int)patch + invcoef * (int)base + 64) >> 7);
}

__kernel void kernel_kfm_switch(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width,
    const int height) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    kfm_store_pixel(dst, dstPitch, x, y, kfm_load_pixel(src, srcPitch, x, y));
}

__kernel void kernel_kfm_patch_combe(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *base,
    const int basePitch,
    const __global uchar *patch,
    const int patchPitch,
    const __global uchar *mask,
    const int maskPitch,
    const int width,
    const int height,
    const int threshold) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const Type m = kfm_load_pixel(mask, maskPitch, x, y);
    const Type b = kfm_load_pixel(base, basePitch, x, y);
    const Type p = kfm_load_pixel(patch, patchPitch, x, y);
    (void)threshold;
    const Type v = kfm_blend_patch(b, p, m);
    kfm_store_pixel(dst, dstPitch, x, y, v);
}

__kernel void kernel_kfm_copy_plane(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width,
    const int height) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    kfm_store_pixel(dst, dstPitch, x, y, kfm_load_pixel(src, srcPitch, x, y));
}
