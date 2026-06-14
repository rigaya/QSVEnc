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

static inline Type kfm_max_value(void) {
#if bit_depth > 8
    return (Type)((1 << bit_depth) - 1);
#else
    return (Type)255;
#endif
}

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

static inline int kfm_absdiff_render(const Type a, const Type b) {
    return abs((int)a - (int)b);
}

static inline int kfm_calc_combe_render(
    const Type L0, const Type L1, const Type L2, const Type L3,
    const Type L4, const Type L5, const Type L6, const Type L7) {
    const int diff8 = kfm_absdiff_render(L0, L7);
    const int diffT =
        kfm_absdiff_render(L0, L1) + kfm_absdiff_render(L1, L2) + kfm_absdiff_render(L2, L3) + kfm_absdiff_render(L3, L4) +
        kfm_absdiff_render(L4, L5) + kfm_absdiff_render(L5, L6) + kfm_absdiff_render(L6, L7) - diff8;
    const int diffE =
        kfm_absdiff_render(L0, L2) + kfm_absdiff_render(L2, L4) + kfm_absdiff_render(L4, L6) + kfm_absdiff_render(L6, L7) - diff8;
    const int diffO =
        kfm_absdiff_render(L0, L1) + kfm_absdiff_render(L1, L3) + kfm_absdiff_render(L3, L5) + kfm_absdiff_render(L5, L7) - diff8;
    return diffT - diffE - diffO;
}

static inline int kfm_calc_diff_render(
    const Type L00, const Type L10, const Type L01, const Type L11,
    const Type L02, const Type L12, const Type L03, const Type L13) {
    return kfm_absdiff_render(L00, L10) + kfm_absdiff_render(L01, L11) + kfm_absdiff_render(L02, L12) + kfm_absdiff_render(L03, L13);
}

static inline Type kfm_load_src_render(
    const __global Type *src,
    const int pitch,
    const int x,
    const int y,
    const int pixelStep,
    const int pixelOffset) {
    return src[x * pixelStep + pixelOffset + y * pitch];
}

static inline uchar4 kfm_analyze_block_render(
    const __global uchar *src0,
    const __global uchar *src1,
    const int srcPitch,
    const int parity,
    const int pixelStep,
    const int pixelOffset,
    const int bx,
    const int by) {
    const int shift = bit_depth - 8 + 4;
    const int srcPitchT = srcPitch / (int)sizeof(Type);
    const __global Type *f0 = (const __global Type *)src0;
    const __global Type *f1 = (const __global Type *)src1;

    int sum0 = 0;
    int sum1 = 0;
    int sum2 = 0;
    int sum3 = 0;
    const int xBase = bx * 4;
    const int yBase = by * 4;

    for (int tx = 0; tx < 8; ++tx) {
        const int x = xBase + tx;
        const int srcIdx0 = x * pixelStep + pixelOffset + yBase * srcPitchT;
        const int srcIdx1 = srcIdx0 + srcPitchT;
        const int srcIdx2 = srcIdx1 + srcPitchT;
        const int srcIdx3 = srcIdx2 + srcPitchT;
        const int srcIdx4 = srcIdx3 + srcPitchT;
        const int srcIdx5 = srcIdx4 + srcPitchT;
        const int srcIdx6 = srcIdx5 + srcPitchT;
        const int srcIdx7 = srcIdx6 + srcPitchT;

        {
            const Type T00 = f0[srcIdx0];
            const Type B00 = f0[srcIdx1];
            const Type T01 = f0[srcIdx2];
            const Type B01 = f0[srcIdx3];
            const Type T02 = f0[srcIdx4];
            const Type B02 = f0[srcIdx5];
            const Type T03 = f0[srcIdx6];
            const Type B03 = f0[srcIdx7];
            const int tmp = kfm_calc_combe_render(T00, B00, T01, B01, T02, B02, T03, B03);
            if (parity) {
                sum0 += tmp;
            } else {
                sum2 += tmp;
            }
        }

        if (parity) {
            const Type T10 = f1[srcIdx0];
            const Type B00 = f0[srcIdx1];
            const Type T11 = f1[srcIdx2];
            const Type B01 = f0[srcIdx3];
            const Type T12 = f1[srcIdx4];
            const Type B02 = f0[srcIdx5];
            const Type T13 = f1[srcIdx6];
            const Type B03 = f0[srcIdx7];
            sum2 += kfm_calc_combe_render(T10, B00, T11, B01, T12, B02, T13, B03);
        } else {
            const Type T00 = f0[srcIdx0];
            const Type B10 = f1[srcIdx1];
            const Type T01 = f0[srcIdx2];
            const Type B11 = f1[srcIdx3];
            const Type T02 = f0[srcIdx4];
            const Type B12 = f1[srcIdx5];
            const Type T03 = f0[srcIdx6];
            const Type B13 = f1[srcIdx7];
            sum0 += kfm_calc_combe_render(T00, B10, T01, B11, T02, B12, T03, B13);
        }

        {
            const Type T00 = f0[srcIdx0];
            const Type T10 = f1[srcIdx0];
            const Type T01 = f0[srcIdx2];
            const Type T11 = f1[srcIdx2];
            const Type T02 = f0[srcIdx4];
            const Type T12 = f1[srcIdx4];
            const Type T03 = f0[srcIdx6];
            const Type T13 = f1[srcIdx6];
            sum1 += kfm_calc_diff_render(T00, T10, T01, T11, T02, T12, T03, T13);
        }

        {
            const Type B00 = f0[srcIdx1];
            const Type B10 = f1[srcIdx1];
            const Type B01 = f0[srcIdx3];
            const Type B11 = f1[srcIdx3];
            const Type B02 = f0[srcIdx5];
            const Type B12 = f1[srcIdx5];
            const Type B03 = f0[srcIdx7];
            const Type B13 = f1[srcIdx7];
            sum3 += kfm_calc_diff_render(B00, B10, B01, B11, B02, B12, B03, B13);
        }
    }

    return (uchar4)(
        (uchar)clamp(sum0 >> shift, 0, 255),
        (uchar)clamp(sum1 >> shift, 0, 255),
        (uchar)clamp(sum2 >> shift, 0, 255),
        (uchar)clamp(sum3 >> shift, 0, 255));
}

static inline uchar2 kfm_analyze_super_pair_render(
    const __global uchar *src0,
    const __global uchar *src1,
    const int srcPitch,
    const int widthPairs,
    const int height,
    const int parity,
    const int pixelStep,
    const int pixelOffset,
    const int x,
    const int row) {
    if (x <= 0 || x >= widthPairs || row < 2 || row >= height * 2) {
        return (uchar2)(0, 0);
    }
    const int bx = x - 1;
    const int by = (row >> 1) - 1;
    if (bx >= widthPairs - 1 || by < 0 || by >= height - 1) {
        return (uchar2)(0, 0);
    }
    const uchar4 v = kfm_analyze_block_render(src0, src1, srcPitch, parity, pixelStep, pixelOffset, bx, by);
    return (row & 1) ? (uchar2)(v.z, v.w) : (uchar2)(v.x, v.y);
}

static inline Type kfm_telecine_weave_pixel(
    const __global uchar *src0,
    const int src0Pitch,
    const __global uchar *src1,
    const int src1Pitch,
    const __global uchar *src2,
    const int src2Pitch,
    const int x,
    const int y,
    const int srcYOffset,
    const int fieldStart,
    const int fieldCount,
    const int parity) {
    const int srcOutY = y + srcYOffset;
    const int outField = ((srcOutY & 1) == (parity & 1)) ? 1 : 0;
    const int fieldBase = fieldStart & ~1;
    const int fieldEnd = fieldStart + fieldCount;
    Type sum = (Type)0;
    int count = 0;

    for (int field = fieldStart; field < fieldEnd; field++) {
        if ((field & 1) != outField) {
            continue;
        }
        const int frameOffset = (field - fieldBase) >> 1;
        const int srcY = (field & 1) + ((srcOutY >> 1) << 1);
        Type v = (Type)0;
        if (frameOffset == 0) {
            v = kfm_load_pixel(src0, src0Pitch, x, srcY);
        } else if (frameOffset == 1) {
            v = kfm_load_pixel(src1, src1Pitch, x, srcY);
        } else {
            v = kfm_load_pixel(src2, src2Pitch, x, srcY);
        }
        if (count == 0) {
            sum = v;
        } else {
            sum = (Type)(((int)sum + (int)v) >> 1);
        }
        count++;
    }
    return sum;
}

__kernel void kernel_kfm_render(
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

__kernel void kernel_kfm_select_field(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width,
    const int height,
    const int field) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    kfm_store_pixel(dst, dstPitch, x, y, kfm_load_pixel(src, srcPitch, x, y * 2 + (field & 1)));
}

__kernel void kernel_kfm_weave_fields(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *field0,
    const int field0Pitch,
    const __global uchar *field1,
    const int field1Pitch,
    const int width,
    const int height,
    const int parity) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int fy = y >> 1;
    const int useField0 = ((y & 1) == (parity & 1));
    const Type v = useField0
        ? kfm_load_pixel(field0, field0Pitch, x, fy)
        : kfm_load_pixel(field1, field1Pitch, x, fy);
    kfm_store_pixel(dst, dstPitch, x, y, v);
}

__kernel void kernel_kfm_telecine_weave(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src0,
    const int src0Pitch,
    const __global uchar *src1,
    const int src1Pitch,
    const __global uchar *src2,
    const int src2Pitch,
    const int width,
    const int height,
    const int srcYOffset,
    const int fieldStart,
    const int fieldCount,
    const int parity) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const Type v = kfm_telecine_weave_pixel(
        src0, src0Pitch, src1, src1Pitch, src2, src2Pitch,
        x, y, srcYOffset, fieldStart, fieldCount, parity);
    kfm_store_pixel(dst, dstPitch, x, y, v);
}

__kernel void kernel_kfm_telecine_super_max(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src0,
    const int src0Pitch,
    const __global uchar *src1,
    const int src1Pitch,
    const __global uchar *src2,
    const int src2Pitch,
    const int width,
    const int height,
    const int frameCount) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    Type v = kfm_load_pixel(src0, src0Pitch, x, y);
    if (frameCount > 1) {
        v = max(v, kfm_load_pixel(src1, src1Pitch, x, y));
    }
    if (frameCount > 2) {
        v = max(v, kfm_load_pixel(src2, src2Pitch, x, y));
    }
    kfm_store_pixel(dst, dstPitch, x, y, v);
}

__kernel void kernel_kfm_clean_separated_super_max(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *prevSuper,
    const __global uchar *curSuper,
    const int superPitch,
    const int widthPairs,
    const int height,
    const int field,
    const int cleanThresh,
    const int maxMode,
    const int dstStep,
    const int dstOffset) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= widthPairs || y >= height) return;

    const int pitchT = superPitch / (int)sizeof(uchar2);
    const int srcField = field & 1;
    const int curRow = y * 2 + srcField;
    const int prevRow = (srcField == 0) ? (y * 2 + 1) : (y * 2);
    const __global uchar2 *prev = (const __global uchar2 *)((srcField == 0) ? prevSuper : curSuper);
    const __global uchar2 *cur = (const __global uchar2 *)curSuper;

    uchar2 v = cur[x + curRow * pitchT];
    const uchar2 pv = prev[x + prevRow * pitchT];
    if (pv.y <= cleanThresh && v.y <= cleanThresh) {
        v.x = 0;
    }

    __global uchar *p0 = dst + y * dstPitch + (x * 2 + 0) * dstStep + dstOffset;
    __global uchar *p1 = dst + y * dstPitch + (x * 2 + 1) * dstStep + dstOffset;
    if (maxMode) {
        p0[0] = max(p0[0], v.x);
        p1[0] = max(p1[0], v.y);
    } else {
        p0[0] = v.x;
        p1[0] = v.y;
    }
}

__kernel void kernel_kfm_clean_super_direct_max(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *prevSrc0,
    const __global uchar *prevSrc1,
    const int prevSrcPitch,
    const int prevParity,
    const __global uchar *curSrc0,
    const __global uchar *curSrc1,
    const int curSrcPitch,
    const int curParity,
    const int widthPairs,
    const int height,
    const int field,
    const int cleanThresh,
    const int maxMode,
    const int dstStep,
    const int dstOffset,
    const int pixelStep,
    const int pixelOffset) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= widthPairs || y >= height) return;

    const int srcField = field & 1;
    const int curRow = y * 2 + srcField;
    const int prevRow = (srcField == 0) ? (y * 2 + 1) : (y * 2);
    const uchar2 vcur = kfm_analyze_super_pair_render(
        curSrc0, curSrc1, curSrcPitch, widthPairs, height, curParity,
        pixelStep, pixelOffset, x, curRow);
    const uchar2 vprev = (srcField == 0)
        ? kfm_analyze_super_pair_render(prevSrc0, prevSrc1, prevSrcPitch, widthPairs, height, prevParity, pixelStep, pixelOffset, x, prevRow)
        : kfm_analyze_super_pair_render(curSrc0, curSrc1, curSrcPitch, widthPairs, height, curParity, pixelStep, pixelOffset, x, prevRow);

    uchar2 v = vcur;
    if (vprev.y <= cleanThresh && v.y <= cleanThresh) {
        v.x = 0;
    }

    __global uchar *p0 = dst + y * dstPitch + (x * 2 + 0) * dstStep + dstOffset;
    __global uchar *p1 = dst + y * dstPitch + (x * 2 + 1) * dstStep + dstOffset;
    if (maxMode) {
        p0[0] = max(p0[0], v.x);
        p1[0] = max(p1[0], v.y);
    } else {
        p0[0] = v.x;
        p1[0] = v.y;
    }
}

__kernel void kernel_kfm_remove_combe_copy(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const __global uchar *clean,
    const int cleanPitch,
    const __global uchar *mask,
    const int maskPitch,
    const int width,
    const int height,
    const int threshold) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int m = (int)kfm_load_pixel(mask, maskPitch, x, y);
    const Type v = (m >= threshold)
        ? kfm_load_pixel(clean, cleanPitch, x, y)
        : kfm_load_pixel(src, srcPitch, x, y);
    kfm_store_pixel(dst, dstPitch, x, y, v);
}

__kernel void kernel_kfm_remove_combe_binomial(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const __global uchar *combe,
    const int combePitch,
    const __global uchar *teleSrc0,
    const int teleSrc0Pitch,
    const __global uchar *teleSrc1,
    const int teleSrc1Pitch,
    const __global uchar *teleSrc2,
    const int teleSrc2Pitch,
    const int width,
    const int height,
    const int threshold,
    const int srcStep,
    const int srcOffset,
    const int combeStep,
    const int combeOffset,
    const int teleSrcYOffset,
    const int teleFieldStart,
    const int teleFieldCount,
    const int teleParity) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int sx = x * srcStep + srcOffset;
    const int cx = (x >> 2) * 2 * combeStep + combeOffset;
    const int cy = y >> 2;
    const int score = (int)combe[cy * combePitch + cx];
    Type v = kfm_load_pixel(src, srcPitch, sx, y);

    if (score >= threshold) {
        const int prevY = max(y - 1, 0);
        const int nextY = min(y + 1, height - 1);
        const int prev = (int)((y > 0)
            ? kfm_load_pixel(src, srcPitch, sx, prevY)
            : kfm_telecine_weave_pixel(teleSrc0, teleSrc0Pitch, teleSrc1, teleSrc1Pitch, teleSrc2, teleSrc2Pitch, sx, y - 1, teleSrcYOffset, teleFieldStart, teleFieldCount, teleParity));
        const int cur = (int)v;
        const int next = (int)((y + 1 < height)
            ? kfm_load_pixel(src, srcPitch, sx, nextY)
            : kfm_telecine_weave_pixel(teleSrc0, teleSrc0Pitch, teleSrc1, teleSrc1Pitch, teleSrc2, teleSrc2Pitch, sx, y + 1, teleSrcYOffset, teleFieldStart, teleFieldCount, teleParity));
        v = (Type)((prev + 2 * cur + next + 2) >> 2);
    }
    kfm_store_pixel(dst, dstPitch, sx, y, v);
}
