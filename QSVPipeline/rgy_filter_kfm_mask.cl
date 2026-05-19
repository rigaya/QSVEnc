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

static inline Type kfm_to_type_sat(const int v) {
#if bit_depth > 8
    return convert_ushort_sat(v);
#else
    return convert_uchar_sat(v);
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

static inline int kfm_div_floor_pow2(const int v, const int shift) {
    const int div = 1 << shift;
    const int q = v / div;
    return q - ((v < 0 && q * div != v) ? 1 : 0);
}

static inline int kfm_temporal_avg3(const int a, const int b, const int c) {
    return (a + b + c) / 3;
}

static inline uchar kfm_load_u8_zero_low(
    const __global uchar *src,
    const int pitch,
    const int x,
    const int y) {
    return (x < 0 || y < 0) ? (uchar)0 : src[y * pitch + x];
}

static inline uchar kfm_load_u8_max_extend_rb(
    const __global uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int x,
    const int y) {
    const int hx0 = (x == 0 && width > 1) ? 1 : x;
    const int hx1 = (x > 0 && x < width - 1) ? x + 1 : hx0;
    const int hy0 = (y == 0 && height > 1) ? 1 : y;
    const int hy1 = (y > 0 && y < height - 1) ? y + 1 : hy0;
    const uchar v00 = src[hy0 * pitch + hx0];
    const uchar v10 = src[hy0 * pitch + hx1];
    const uchar v01 = src[hy1 * pitch + hx0];
    const uchar v11 = src[hy1 * pitch + hx1];
    return max(max(v00, v10), max(v01, v11));
}

__kernel void kernel_kfm_mask(
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

__kernel void kernel_kfm_switch_flag(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width,
    const int height,
    const int threshold,
    const int setValue) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int v = (int)kfm_load_pixel(src, srcPitch, x, y);
    kfm_store_pixel(dst, dstPitch, x, y, (v >= threshold) ? kfm_to_type_sat(setValue) : (Type)0);
}

__kernel void kernel_kfm_contains_combe_init(
    __global uint *count) {
    if (get_global_id(0) == 0) {
        count[0] = 0;
    }
}

__kernel void kernel_kfm_contains_combe_count(
    const __global uchar *mask,
    const int maskPitch,
    __global uint *count,
    const int width,
    const int height,
    const int threshold) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    // Applied after Crop(0, 6, 0, -6)
    if (y >= 6 && y < height - 6 && (int)kfm_load_pixel(mask, maskPitch, x, y) >= threshold) {
        atomic_inc((volatile __global unsigned int *)count);
    }
}

__kernel void kernel_kfm_combe_mask(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width,
    const int height,
    const int threshold) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int v = (int)kfm_load_pixel(src, srcPitch, x, y);
    kfm_store_pixel(dst, dstPitch, x, y, (v >= threshold) ? kfm_max_value() : (Type)0);
}

__kernel void kernel_kfm_switch_flag_min(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *superY,
    const int superYPitch,
    const __global uchar *superUV,
    const int superUVPitch,
    const int flagWidth,
    const int flagHeight,
    const int innerWidth,
    const int innerHeight,
    const int thY,
    const int thC,
    const int hasUV) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= flagWidth || y >= flagHeight) return;

    if (x < 4 || y < 2 || x >= innerWidth + 4 || y >= innerHeight + 2) {
        dst[y * dstPitch + x] = (uchar)0;
        return;
    }

    const int ix = x - 4;
    const int iy = y - 2;
    const int sx0 = (ix << 2) - 2;
    const int sx1 = ix << 2;
    const int sy0 = (iy << 1) - 1;
    const int sy1 = iy << 1;
    const int ysum =
        (int)kfm_load_u8_zero_low(superY, superYPitch, sx0, sy0)
      + (int)kfm_load_u8_zero_low(superY, superYPitch, sx1, sy0)
      + (int)kfm_load_u8_zero_low(superY, superYPitch, sx0, sy1)
      + (int)kfm_load_u8_zero_low(superY, superYPitch, sx1, sy1);
    int cmax = 0;
    if (hasUV && ix > 0 && iy > 0) {
        const int ux = ix << 2;
        const int uy = iy;
        cmax = max((int)superUV[uy * superUVPitch + ux + 0], (int)superUV[uy * superUVPitch + ux + 1]);
    }
    dst[y * dstPitch + x] = (((ysum + 2) >> 2) >= thY || cmax >= thC) ? (uchar)128 : (uchar)0;
}

__kernel void kernel_kfm_switch_flag_extract_min(
    __global uchar *dstY,
    const int dstYPitch,
    __global uchar *dstC,
    const int dstCPitch,
    const __global uchar *superPrevY,
    const __global uchar *superY,
    const __global uchar *superNextY,
    const int superYPitch,
    const __global uchar *superPrevUV,
    const __global uchar *superUV,
    const __global uchar *superNextUV,
    const int superUVPitch,
    const int flagWidth,
    const int flagHeight,
    const int innerWidth,
    const int innerHeight,
    const int hasUV) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= flagWidth || y >= flagHeight) return;

    if (x < 4 || y < 2 || x >= innerWidth + 4 || y >= innerHeight + 2) {
        dstY[y * dstYPitch + x] = (uchar)0;
        dstC[y * dstCPitch + x] = (uchar)0;
        return;
    }

    const int ix = x - 4;
    const int iy = y - 2;
    const int sx0 = (ix << 2) - 2;
    const int sx1 = ix << 2;
    const int sy0 = (iy << 1) - 1;
    const int sy1 = iy << 1;
    const int ysum =
        kfm_temporal_avg3((int)kfm_load_u8_zero_low(superPrevY, superYPitch, sx0, sy0), (int)kfm_load_u8_zero_low(superY, superYPitch, sx0, sy0), (int)kfm_load_u8_zero_low(superNextY, superYPitch, sx0, sy0))
      + kfm_temporal_avg3((int)kfm_load_u8_zero_low(superPrevY, superYPitch, sx1, sy0), (int)kfm_load_u8_zero_low(superY, superYPitch, sx1, sy0), (int)kfm_load_u8_zero_low(superNextY, superYPitch, sx1, sy0))
      + kfm_temporal_avg3((int)kfm_load_u8_zero_low(superPrevY, superYPitch, sx0, sy1), (int)kfm_load_u8_zero_low(superY, superYPitch, sx0, sy1), (int)kfm_load_u8_zero_low(superNextY, superYPitch, sx0, sy1))
      + kfm_temporal_avg3((int)kfm_load_u8_zero_low(superPrevY, superYPitch, sx1, sy1), (int)kfm_load_u8_zero_low(superY, superYPitch, sx1, sy1), (int)kfm_load_u8_zero_low(superNextY, superYPitch, sx1, sy1));
    int cmax = 0;
    if (hasUV && ix > 0 && iy > 0) {
        const int ux = ix << 2;
        const int uy = iy;
        cmax = max(
            kfm_temporal_avg3((int)superPrevUV[uy * superUVPitch + ux + 0], (int)superUV[uy * superUVPitch + ux + 0], (int)superNextUV[uy * superUVPitch + ux + 0]),
            kfm_temporal_avg3((int)superPrevUV[uy * superUVPitch + ux + 1], (int)superUV[uy * superUVPitch + ux + 1], (int)superNextUV[uy * superUVPitch + ux + 1]));
    }
    dstY[y * dstYPitch + x] = (uchar)((ysum + 2) >> 2);
    dstC[y * dstCPitch + x] = (uchar)cmax;
}

__kernel void kernel_kfm_switch_flag_combe_min(
    __global uchar *dstY,
    const int dstYPitch,
    __global uchar *dstC,
    const int dstCPitch,
    const __global uchar *superPrevY,
    const __global uchar *superY,
    const __global uchar *superNextY,
    const int superYPitch,
    const __global uchar *superPrevUV,
    const __global uchar *superUV,
    const __global uchar *superNextUV,
    const int superUVPitch,
    const __global uchar *superPrevV,
    const __global uchar *superV,
    const __global uchar *superNextV,
    const int superVPitch,
    const int combeWidth,
    const int combeHeight,
    const int combeCWidth,
    const int combeCHeight,
    const int hasUV,
    const int interleavedUV) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < combeWidth && y < combeHeight) {
        const int sx = x << 1;
        dstY[y * dstYPitch + x] = (uchar)kfm_temporal_avg3(
            (int)superPrevY[y * superYPitch + sx],
            (int)superY[y * superYPitch + sx],
            (int)superNextY[y * superYPitch + sx]);
    }

    if (x < combeCWidth && y < combeCHeight) {
        int cmax = 0;
        if (hasUV) {
            const int ux = interleavedUV ? (x << 2) : (x << 1);
            const int uy = y;
            const int u = kfm_temporal_avg3(
                (int)superPrevUV[uy * superUVPitch + ux],
                (int)superUV[uy * superUVPitch + ux],
                (int)superNextUV[uy * superUVPitch + ux]);
            const int v = kfm_temporal_avg3(
                (int)superPrevV[uy * superVPitch + (interleavedUV ? (ux + 1) : ux)],
                (int)superV[uy * superVPitch + (interleavedUV ? (ux + 1) : ux)],
                (int)superNextV[uy * superVPitch + (interleavedUV ? (ux + 1) : ux)]);
            cmax = max(u, v);
        }
        dstC[y * dstCPitch + x] = (uchar)cmax;
    }
}

__kernel void kernel_kfm_switch_flag_extend_h_min(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width,
    const int height,
    const int offsetX,
    const int offsetY) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int px = x + offsetX;
    const int py = y + offsetY;
    uchar v;
    if (x == width - 1) {
        v = src[py * srcPitch + px];
    } else if (x == 0) {
        v = src[py * srcPitch + px + 1];
    } else {
        v = max(src[py * srcPitch + px], src[py * srcPitch + px + 1]);
    }
    dst[py * dstPitch + px] = v;
}

__kernel void kernel_kfm_switch_flag_extend_v_min(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width,
    const int height,
    const int offsetX,
    const int offsetY) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int px = x + offsetX;
    const int py = y + offsetY;
    uchar v;
    if (y == height - 1) {
        v = src[py * srcPitch + px];
    } else if (y == 0) {
        v = src[(py + 1) * srcPitch + px];
    } else {
        v = max(src[py * srcPitch + px], src[(py + 1) * srcPitch + px]);
    }
    dst[py * dstPitch + px] = v;
}

__kernel void kernel_kfm_switch_flag_from_combe_min(
    __global uchar *dstY,
    const int dstYPitch,
    __global uchar *dstC,
    const int dstCPitch,
    const __global uchar *combeY,
    const int combeYPitch,
    const __global uchar *combeC,
    const int combeCPitch,
    const int flagWidth,
    const int flagHeight,
    const int innerWidth,
    const int innerHeight,
    const int combeWidth,
    const int combeHeight,
    const int combeCWidth,
    const int combeCHeight) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= flagWidth || y >= flagHeight) return;

    const int ix = x - 4;
    const int iy = y - 2;
    if (x < 4 || y < 2 || x >= innerWidth + 4 || y >= innerHeight + 2 || ix == 0 || iy == 0) {
        dstY[y * dstYPitch + x] = (uchar)0;
        dstC[y * dstCPitch + x] = (uchar)0;
        return;
    }

    const int cx0 = (ix << 1) - 1;
    const int cx1 = ix << 1;
    const int cy0 = (iy << 1) - 1;
    const int cy1 = iy << 1;
    const int ysum =
        ((cx0 < 0 || cy0 < 0 || cx0 >= combeWidth || cy0 >= combeHeight) ? 0 : (int)kfm_load_u8_max_extend_rb(combeY, combeYPitch, combeWidth, combeHeight, cx0, cy0))
      + ((cx1 < 0 || cy0 < 0 || cx1 >= combeWidth || cy0 >= combeHeight) ? 0 : (int)kfm_load_u8_max_extend_rb(combeY, combeYPitch, combeWidth, combeHeight, cx1, cy0))
      + ((cx0 < 0 || cy1 < 0 || cx0 >= combeWidth || cy1 >= combeHeight) ? 0 : (int)kfm_load_u8_max_extend_rb(combeY, combeYPitch, combeWidth, combeHeight, cx0, cy1))
      + ((cx1 < 0 || cy1 < 0 || cx1 >= combeWidth || cy1 >= combeHeight) ? 0 : (int)kfm_load_u8_max_extend_rb(combeY, combeYPitch, combeWidth, combeHeight, cx1, cy1));
    dstY[y * dstYPitch + x] = (uchar)((ysum + 2) >> 2);
    dstC[y * dstCPitch + x] = (ix < combeCWidth && iy < combeCHeight) ? combeC[iy * combeCPitch + ix] : (uchar)0;
}

__kernel void kernel_kfm_switch_flag_box3x3_min(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width,
    const int height,
    const int innerWidth,
    const int innerHeight) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    if (x < 4 || y < 2 || x >= innerWidth + 4 || y >= innerHeight + 2) {
        dst[y * dstPitch + x] = (uchar)0;
        return;
    }

    int sum = 0;
    for (int dy = -1; dy <= 1; dy++) {
        const int sy = y + dy;
        for (int dx = -1; dx <= 1; dx++) {
            const int sx = x + dx;
            sum += (sx < 0 || sy < 0 || sx >= width || sy >= height) ? 0 : (int)src[sy * srcPitch + sx];
        }
    }
    dst[y * dstPitch + x] = (uchar)min(sum >> 2, 255);
}

__kernel void kernel_kfm_switch_flag_binary_min(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *srcY,
    const int srcYPitch,
    const __global uchar *srcC,
    const int srcCPitch,
    const int width,
    const int height,
    const int innerWidth,
    const int innerHeight,
    const int thY,
    const int thC) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    if (x < 4 || y < 2 || x >= innerWidth + 4 || y >= innerHeight + 2) {
        dst[y * dstPitch + x] = (uchar)0;
        return;
    }

    const int yv = (int)srcY[y * srcYPitch + x];
    const int cv = (int)srcC[y * srcCPitch + x];
    dst[y * dstPitch + x] = (yv >= thY || cv >= thC) ? (uchar)128 : (uchar)0;
}

__kernel void kernel_kfm_switch_flag_binary_extend_hv_min(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *srcY,
    const int srcYPitch,
    const __global uchar *srcC,
    const int srcCPitch,
    const int width,
    const int height,
    const int innerWidth,
    const int innerHeight,
    const int thY,
    const int thC) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    if (x < 4 || y < 2 || x >= innerWidth + 4 || y >= innerHeight + 2) {
        dst[y * dstPitch + x] = (uchar)0;
        return;
    }

    const int ix = x - 4;
    const int iy = y - 2;
    const int hx0 = (ix == 0 && innerWidth > 1) ? 1 : ix;
    const int hx1 = (ix > 0 && ix < innerWidth - 1) ? ix + 1 : hx0;
    const int hy0 = (iy == 0 && innerHeight > 1) ? 1 : iy;
    const int hy1 = (iy > 0 && iy < innerHeight - 1) ? iy + 1 : hy0;
    const int px0 = hx0 + 4;
    const int px1 = hx1 + 4;
    const int py0 = hy0 + 2;
    const int py1 = hy1 + 2;

    const int ymax = max(
        max((int)srcY[py0 * srcYPitch + px0], (int)srcY[py0 * srcYPitch + px1]),
        max((int)srcY[py1 * srcYPitch + px0], (int)srcY[py1 * srcYPitch + px1]));
    const int cmax = max(
        max((int)srcC[py0 * srcCPitch + px0], (int)srcC[py0 * srcCPitch + px1]),
        max((int)srcC[py1 * srcCPitch + px0], (int)srcC[py1 * srcCPitch + px1]));
    dst[y * dstPitch + x] = (ymax >= thY || cmax >= thC) ? (uchar)128 : (uchar)0;
}

__kernel void kernel_kfm_copy_u8_buffer_to_plane(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *src,
    const int srcPitch,
    const int width,
    const int height) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    dst[y * dstPitch + x] = src[y * srcPitch + x];
}

__kernel void kernel_kfm_contains_combe_mark(
    __global uchar *dst,
    const int dstPitch,
    const __global uint *count) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= 4 || y >= 1) return;
    dst[y * dstPitch + x] = (count[0] != 0) ? (uchar)255 : (uchar)0;
}

__kernel void kernel_kfm_combe_mask_resize_min(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *flag,
    const int flagPitch,
    const int width,
    const int height,
    const int srcStep,
    const int srcOffset,
    const int xShift,
    const int yShift) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int sx = x * srcStep + srcOffset;
    const int fx = (x >> xShift) + 4;
    const int fy = (y >> yShift) + 2;
    const Type v = (flag[fy * flagPitch + fx] != 0) ? kfm_max_value() : (Type)0;
    kfm_store_pixel(dst, dstPitch, sx, y, v);
}

__kernel void kernel_kfm_combe_mask_resize_bilinear_min(
    __global uchar *dst,
    const int dstPitch,
    const __global uchar *flag,
    const int flagPitch,
    const int width,
    const int height,
    const int srcStep,
    const int srcOffset,
    const int scaleX,
    const int shiftX,
    const int scaleY,
    const int shiftY,
    const int innerWidth,
    const int innerHeight) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int halfX = scaleX >> 1;
    const int halfY = scaleY >> 1;
    const int x0 = kfm_div_floor_pow2(x - halfX, shiftX);
    const int y0 = kfm_div_floor_pow2(y - halfY, shiftY);
    const int c0x = ((x0 + 1) << shiftX) - (x - halfX);
    const int c1x = scaleX - c0x;
    const int c0y = ((y0 + 1) << shiftY) - (y - halfY);
    const int c1y = scaleY - c0y;
    const int fx0 = clamp(x0, 0, innerWidth - 1) + 4;
    const int fx1 = clamp(x0 + 1, 0, innerWidth - 1) + 4;
    const int fy0 = clamp(y0, 0, innerHeight - 1) + 2;
    const int fy1 = clamp(y0 + 1, 0, innerHeight - 1) + 2;

    const int h0 = ((int)flag[fy0 * flagPitch + fx0] * c0x + (int)flag[fy0 * flagPitch + fx1] * c1x + halfX) >> shiftX;
    const int h1 = ((int)flag[fy1 * flagPitch + fx0] * c0x + (int)flag[fy1 * flagPitch + fx1] * c1x + halfX) >> shiftX;
    const int v = (h0 * c0y + h1 * c1y + halfY) >> shiftY;
    kfm_store_pixel(dst, dstPitch, x * srcStep + srcOffset, y, kfm_to_type_sat(v));
}
