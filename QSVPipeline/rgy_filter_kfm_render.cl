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

#define KFM_CAT_(a, b) a##b
#define KFM_CAT(a, b) KFM_CAT_(a, b)
#define Type4 KFM_CAT(Type, 4)
#define Type16 KFM_CAT(Type, 16)

#ifndef bit_depth
#define bit_depth 8
#endif

static inline int4 kfm_to_int4(const Type4 v) {
#if bit_depth > 8
    return (int4)(v.x, v.y, v.z, v.w);
#else
    return convert_int4(v);
#endif
}

static inline Type4 kfm_to_type4(const int4 v) {
#if bit_depth > 8
    return convert_ushort4_sat(v);
#else
    return convert_uchar4_sat(v);
#endif
}

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

static inline Type4 kfm_load_pixel4(
    const __global uchar *src,
    const int pitch,
    const int x,
    const int y) {
    return vload4(0, ((const __global Type *)(src + y * pitch)) + x);
}

static inline void kfm_store_pixel4(
    __global uchar *dst,
    const int pitch,
    const int x,
    const int y,
    const Type4 v) {
    vstore4(v, 0, ((__global Type *)(dst + y * pitch)) + x);
}

static inline int8 kfm_absdiff_render(const int8 a, const int8 b) {
    return convert_int8(abs(a - b));
}

static inline int8 kfm_calc_combe_render(
    const int8 L0, const int8 L1, const int8 L2, const int8 L3,
    const int8 L4, const int8 L5, const int8 L6, const int8 L7) {
    const int8 diff8 = kfm_absdiff_render(L0, L7);
    const int8 diffT =
        kfm_absdiff_render(L0, L1) + kfm_absdiff_render(L1, L2) + kfm_absdiff_render(L2, L3) + kfm_absdiff_render(L3, L4) +
        kfm_absdiff_render(L4, L5) + kfm_absdiff_render(L5, L6) + kfm_absdiff_render(L6, L7) - diff8;
    const int8 diffE =
        kfm_absdiff_render(L0, L2) + kfm_absdiff_render(L2, L4) + kfm_absdiff_render(L4, L6) + kfm_absdiff_render(L6, L7) - diff8;
    const int8 diffO =
        kfm_absdiff_render(L0, L1) + kfm_absdiff_render(L1, L3) + kfm_absdiff_render(L3, L5) + kfm_absdiff_render(L5, L7) - diff8;
    return diffT - diffE - diffO;
}

static inline int8 kfm_calc_diff_render(
    const int8 L00, const int8 L10, const int8 L01, const int8 L11,
    const int8 L02, const int8 L12, const int8 L03, const int8 L13) {
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
    const int pitch = srcPitch / (int)sizeof(Type);
    const __global Type *f0 = (const __global Type *)src0;
    const __global Type *f1 = (const __global Type *)src1;
    const int xBase = bx * 4;
    const int yBase = by * 4;
    int8 row0[8];
    int8 row1[8];
    // 行単位で8列を読み、各列の解析はレジスタ上で同じ整数演算を行う。
    if (pixelStep == 1) {
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int offset = xBase + pixelOffset + (yBase + r) * pitch;
            row0[r] = convert_int8(vload8(0, f0 + offset));
            row1[r] = convert_int8(vload8(0, f1 + offset));
        }
    } else {
        // UVのV成分もペアの先頭から読み、最終列の外へ読み越さないようoddを選ぶ。
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int offset = xBase * 2 + (yBase + r) * pitch;
            const Type16 v0 = vload16(0, f0 + offset);
            const Type16 v1 = vload16(0, f1 + offset);
            row0[r] = convert_int8(pixelOffset == 0 ? v0.even : v0.odd);
            row1[r] = convert_int8(pixelOffset == 0 ? v1.even : v1.odd);
        }
    }
    const int8 same = kfm_calc_combe_render(row0[0], row0[1], row0[2], row0[3], row0[4], row0[5], row0[6], row0[7]);
    const int8 mixed = parity
        ? kfm_calc_combe_render(row1[0], row0[1], row1[2], row0[3], row1[4], row0[5], row1[6], row0[7])
        : kfm_calc_combe_render(row0[0], row1[1], row0[2], row1[3], row0[4], row1[5], row0[6], row1[7]);
    const int8 move0 = kfm_calc_diff_render(row0[0], row1[0], row0[2], row1[2], row0[4], row1[4], row0[6], row1[6]);
    const int8 move1 = kfm_calc_diff_render(row0[1], row1[1], row0[3], row1[3], row0[5], row1[5], row0[7], row1[7]);
    const int8 combe0 = parity ? same : mixed;
    const int8 combe1 = parity ? mixed : same;
    const int sum0 = combe0.s0 + combe0.s1 + combe0.s2 + combe0.s3 + combe0.s4 + combe0.s5 + combe0.s6 + combe0.s7;
    const int sum1 = move0.s0 + move0.s1 + move0.s2 + move0.s3 + move0.s4 + move0.s5 + move0.s6 + move0.s7;
    const int sum2 = combe1.s0 + combe1.s1 + combe1.s2 + combe1.s3 + combe1.s4 + combe1.s5 + combe1.s6 + combe1.s7;
    const int sum3 = move1.s0 + move1.s1 + move1.s2 + move1.s3 + move1.s4 + move1.s5 + move1.s6 + move1.s7;
    return (uchar4)(
        (uchar)clamp(sum0 >> shift, 0, 255),
        (uchar)clamp(sum1 >> shift, 0, 255),
        (uchar)clamp(sum2 >> shift, 0, 255),
        (uchar)clamp(sum3 >> shift, 0, 255));
}

// blockを丸ごと(uchar4)返す版。1つのblockからfield 0/1の両方を取り出す場合に使う
static inline uchar4 kfm_analyze_super_pair_render4(
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
        return (uchar4)(0, 0, 0, 0);
    }
    const int bx = x - 1;
    const int by = (row >> 1) - 1;
    if (bx >= widthPairs - 1 || by < 0 || by >= height - 1) {
        return (uchar4)(0, 0, 0, 0);
    }
    return kfm_analyze_block_render(src0, src1, srcPitch, parity, pixelStep, pixelOffset, bx, by);
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
    const uchar4 v = kfm_analyze_super_pair_render4(
        src0, src1, srcPitch, widthPairs, height, parity, pixelStep, pixelOffset, x, row);
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

static inline Type4 kfm_telecine_weave_pixel4(
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
    Type4 sum = (Type4)0;
    int count = 0;

    for (int field = fieldStart; field < fieldEnd; field++) {
        if ((field & 1) != outField) {
            continue;
        }
        const int frameOffset = (field - fieldBase) >> 1;
        const int srcY = (field & 1) + ((srcOutY >> 1) << 1);
        Type4 v = (Type4)0;
        if (frameOffset == 0) {
            v = kfm_load_pixel4(src0, src0Pitch, x, srcY);
        } else if (frameOffset == 1) {
            v = kfm_load_pixel4(src1, src1Pitch, x, srcY);
        } else {
            v = kfm_load_pixel4(src2, src2Pitch, x, srcY);
        }
        if (count == 0) {
            sum = v;
        } else {
            sum = kfm_to_type4((kfm_to_int4(sum) + kfm_to_int4(v)) >> 1);
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
    const int x = get_global_id(0) * 4;
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    if (x + 3 < width) {
        const Type4 v = kfm_telecine_weave_pixel4(
            src0, src0Pitch, src1, src1Pitch, src2, src2Pitch,
            x, y, srcYOffset, fieldStart, fieldCount, parity);
        kfm_store_pixel4(dst, dstPitch, x, y, v);
    } else {
        for (int ix = x; ix < width; ix++) {
            const Type v = kfm_telecine_weave_pixel(
                src0, src0Pitch, src1, src1Pitch, src2, src2Pitch,
                ix, y, srcYOffset, fieldStart, fieldCount, parity);
            kfm_store_pixel(dst, dstPitch, ix, y, v);
        }
    }
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
    uchar2 vcur;
    uchar2 vprev;
    if (srcField == 0) {
        vcur = kfm_analyze_super_pair_render(curSrc0, curSrc1, curSrcPitch, widthPairs, height, curParity,
            pixelStep, pixelOffset, x, y * 2 + 0);
        vprev = kfm_analyze_super_pair_render(prevSrc0, prevSrc1, prevSrcPitch, widthPairs, height, prevParity,
            pixelStep, pixelOffset, x, y * 2 + 1);
    } else {
        // odd fieldではvcur(row=y*2+1)とvprev(row=y*2)が同じblockを指すため、解析は1回で済む。
        // row依存の境界判定もy==0のときに両者とも0を返すため、結果は変わらない。
        const uchar4 v4 = kfm_analyze_super_pair_render4(curSrc0, curSrc1, curSrcPitch, widthPairs, height, curParity,
            pixelStep, pixelOffset, x, y * 2 + 1);
        vcur = (uchar2)(v4.z, v4.w);
        vprev = (uchar2)(v4.x, v4.y);
    }

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
    const int x = get_global_id(0) * 4;
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int cy = y >> 2;
    if (srcStep == 1 && x + 3 < width) {
        const int sx = x + srcOffset;
        const int cx = (x >> 2) * 2 * combeStep + combeOffset;
        const int score = (int)combe[cy * combePitch + cx];
        Type4 v = kfm_load_pixel4(src, srcPitch, sx, y);
        if (score >= threshold) {
            const int prevY = max(y - 1, 0);
            const int nextY = min(y + 1, height - 1);
            const Type4 prev = (y > 0)
                ? kfm_load_pixel4(src, srcPitch, sx, prevY)
                : kfm_telecine_weave_pixel4(teleSrc0, teleSrc0Pitch, teleSrc1, teleSrc1Pitch, teleSrc2, teleSrc2Pitch, sx, y - 1, teleSrcYOffset, teleFieldStart, teleFieldCount, teleParity);
            const Type4 next = (y + 1 < height)
                ? kfm_load_pixel4(src, srcPitch, sx, nextY)
                : kfm_telecine_weave_pixel4(teleSrc0, teleSrc0Pitch, teleSrc1, teleSrc1Pitch, teleSrc2, teleSrc2Pitch, sx, y + 1, teleSrcYOffset, teleFieldStart, teleFieldCount, teleParity);
            v = kfm_to_type4((kfm_to_int4(prev) + (int4)2 * kfm_to_int4(v) + kfm_to_int4(next) + (int4)2) >> 2);
        }
        kfm_store_pixel4(dst, dstPitch, sx, y, v);
        return;
    }

    const int xEnd = min(x + 4, width);
    for (int ix = x; ix < xEnd; ix++) {
        const int sx = ix * srcStep + srcOffset;
        const int cx = (ix >> 2) * 2 * combeStep + combeOffset;
        const int score = (int)combe[cy * combePitch + cx];
        Type v = kfm_load_pixel(src, srcPitch, sx, y);
        if (score < threshold) {
            kfm_store_pixel(dst, dstPitch, sx, y, v);
            continue;
        }
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
        kfm_store_pixel(dst, dstPitch, sx, y, v);
    }
}
