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

#ifndef TypePixel
#define TypePixel uchar
#endif

#ifndef DEGRAIN_REFS
#define DEGRAIN_REFS 2
#endif

#ifndef DEGRAIN_PIXEL_MAX
#define DEGRAIN_PIXEL_MAX 255
#endif

#ifndef DEGRAIN_PEL
#define DEGRAIN_PEL 1
#endif

#if DEGRAIN_PEL <= 1
#define DEGRAIN_PEL_RSHIFT 0
#elif DEGRAIN_PEL == 2
#define DEGRAIN_PEL_RSHIFT 1
#elif DEGRAIN_PEL == 4
#define DEGRAIN_PEL_RSHIFT 2
#else
#define DEGRAIN_PEL_RSHIFT 0
#endif

#ifndef DEGRAIN_SUBPEL_INTERP
#define DEGRAIN_SUBPEL_INTERP 0
#endif

#ifndef DEGRAIN_BINOMIAL
#define DEGRAIN_BINOMIAL 0
#endif

typedef struct {
    short dx;
    short dy;
    ushort sad;
    ushort refdir;
    uint flags;
    uint reserved;
} degrain_mv_t;

typedef struct {
    uint sad;
    uint srcAvg;
    uint refAvg;
    uint reserved;
} degrain_sad_t;

#define DEGRAIN_MODE_COMPB 0
#define DEGRAIN_MODE_COMPF 1
#define DEGRAIN_MODE_DEGRAIN 2
#define DEGRAIN_PI_F 3.14159265358979323846f

static inline TypePixel degrain_clamp_pixel(const int value) {
    return (TypePixel)clamp(value, 0, DEGRAIN_PIXEL_MAX);
}

static inline ushort degrain_clamp_u16(const uint value) {
    return (ushort)min(value, 65535u);
}

static inline int degrain_centered_signed_value(const int value) {
    const int search = max(DEGRAIN_SEARCH * DEGRAIN_PEL, 1);
    const int clampedValue = clamp(value, -search, search);
    const int center = (DEGRAIN_PIXEL_MAX + 1) >> 1;
    const int range = max(center - 1, 1);
    return clamp(center + (clampedValue * range) / search, 0, DEGRAIN_PIXEL_MAX);
}

static inline int degrain_primary_block_index(const int x, const int y, const int blocksX, const int blocksY, const int step) {
    const int clampedStep = max(step, 1);
    const int blockX = min(x / clampedStep, blocksX - 1);
    const int blockY = min(y / clampedStep, blocksY - 1);
    return blockY * blocksX + blockX;
}

static inline int degrain_debug_border(const int x, const int y, const int step) {
    const int clampedStep = max(step, 1);
    return (x % clampedStep) == 0 || (y % clampedStep) == 0;
}

static inline int degrain_block_origin(const int block, const int step) {
    return block * max(step, 1);
}

static inline int degrain_is_covered_pixel(
    const int x,
    const int y,
    const int coveredWidth,
    const int coveredHeight) {
    return x < coveredWidth && y < coveredHeight;
}

static inline int degrain_ref_index(const int block, const int refDirection) {
    const int clampedRefDirection = clamp(refDirection, 0, DEGRAIN_REFS - 1);
    return block * DEGRAIN_REFS + clampedRefDirection;
}

static inline int degrain_pixel_load(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int x,
    const int y) {
    const int px = clamp(x, 0, width - 1);
    const int py = clamp(y, 0, height - 1);
    return (int)(*(__global const TypePixel *)(src + py * pitch + px * (int)sizeof(TypePixel)));
}

static inline int degrain_pixel_load_unchecked(
    __global const uchar *src,
    const int pitch,
    const int x,
    const int y) {
    return (int)(*(__global const TypePixel *)(src + y * pitch + x * (int)sizeof(TypePixel)));
}

static inline int degrain_mirror_coord(const int value, const int size) {
    // 想定入力域 [-size, 2*size) で半開区間 [0, size) への mirror reflection を行う。
    // 左端側の反射候補は (-value - 1)、右端側の反射候補は (2*size - 1 - value) に相当する。
    // 入力域内の value はそれら反射候補のいずれよりも内側に位置するので、
    // max(value, 左反射) で左側を、min(..., 右反射) で右側を抑える形に纏める。
    const int reflected_low  = max(value, -value - 1);
    const int reflected_high = min(reflected_low, 2 * size - 1 - value);
    return clamp(reflected_high, 0, size - 1);
}

static inline int degrain_pixel_load_mirror(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int x,
    const int y) {
    const int px = degrain_mirror_coord(x, width);
    const int py = degrain_mirror_coord(y, height);
    return (int)(*(__global const TypePixel *)(src + py * pitch + px * (int)sizeof(TypePixel)));
}

static inline int degrain_blur3x3_weighted(
    const int p00,
    const int p10,
    const int p20,
    const int p01,
    const int p11,
    const int p21,
    const int p02,
    const int p12,
    const int p22) {
    const int sum =
        p00 + 2 * p10 + p20 +
        2 * p01 + 4 * p11 + 2 * p21 +
        p02 + 2 * p12 + p22;
    return (sum + 8) >> 4;
}

static inline int degrain_edge_soften_cross(
    const int left,
    const int up,
    const int center,
    const int down,
    const int right) {
    return (left + up + 4 * center + down + right + 4) >> 3;
}

static inline int degrain_search_refine1_blend(
    const int center,
    const int blur,
    const int edgeSoft,
    const int left,
    const int up,
    const int right,
    const int down) {
    const int edgeScale = max((DEGRAIN_PIXEL_MAX + 31) / 32, 1);
    const int edgeStrength = abs(left - right) + abs(up - down) + abs(center - blur);
    const int edgeWeight = clamp((edgeStrength + (edgeScale >> 1)) / edgeScale, 0, 4);
    return (blur * (4 - edgeWeight) + edgeSoft * edgeWeight + 2) >> 2;
}

static inline int degrain_floor_rshift_signed(const int value, const int rshift) {
    if (rshift <= 0) {
        return value;
    }
    return value >= 0
        ? value >> rshift
        : -(((-value) + (1 << rshift) - 1) >> rshift);
}

static inline int degrain_round_rshift_signed(const int value, const int rshift) {
    if (rshift <= 0) {
        return value;
    }
    return value >= 0
        ? (value + (1 << (rshift - 1))) >> rshift
        : -(((-value) + (1 << (rshift - 1))) >> rshift);
}

static inline int degrain_floor_div_pel(const int value) {
#if DEGRAIN_PEL <= 1
    return value;
#else
    return degrain_floor_rshift_signed(value, DEGRAIN_PEL_RSHIFT);
#endif
}

static inline int degrain_floor_mod_pel(const int value, const int base) {
    return value - (base << DEGRAIN_PEL_RSHIFT);
}

static inline int degrain_plane_scale_rshift(const int planeScale) {
    return planeScale > 1 ? 1 : 0;
}

static inline int degrain_plane_scale_x(const int planeScaleX) {
#if defined(DEGRAIN_PLANE_SCALE_X)
    return DEGRAIN_PLANE_SCALE_X;
#else
    return max(planeScaleX, 1);
#endif
}

static inline int degrain_plane_scale_y(const int planeScaleY) {
#if defined(DEGRAIN_PLANE_SCALE_Y)
    return DEGRAIN_PLANE_SCALE_Y;
#else
    return max(planeScaleY, 1);
#endif
}

static inline int degrain_plane_scale_rshift_x(const int planeScaleX) {
    return degrain_plane_scale_rshift(degrain_plane_scale_x(planeScaleX));
}

static inline int degrain_plane_scale_rshift_y(const int planeScaleY) {
    return degrain_plane_scale_rshift(degrain_plane_scale_y(planeScaleY));
}

static inline int degrain_interp_halfpel_wiener_v(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int baseY) {
    const int s0 = degrain_pixel_load(src, pitch, width, height, baseX, baseY - 2);
    const int s1 = degrain_pixel_load(src, pitch, width, height, baseX, baseY - 1);
    const int s2 = degrain_pixel_load(src, pitch, width, height, baseX, baseY);
    const int s3 = degrain_pixel_load(src, pitch, width, height, baseX, baseY + 1);
    const int s4 = degrain_pixel_load(src, pitch, width, height, baseX, baseY + 2);
    const int s5 = degrain_pixel_load(src, pitch, width, height, baseX, baseY + 3);
    const int sum = s0 + 5 * (-s1 + (s2 << 2) + (s3 << 2) - s4) + s5;
    return degrain_clamp_pixel((sum + 16) >> 5);
}

static inline int degrain_interp_halfpel_wiener_h_from_samples(
    const int s0,
    const int s1,
    const int s2,
    const int s3,
    const int s4,
    const int s5) {
    const int sum = s0 + 5 * (-s1 + (s2 << 2) + (s3 << 2) - s4) + s5;
    return degrain_clamp_pixel((sum + 16) >> 5);
}

static inline int degrain_interp_halfpel_wiener_h(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int baseY) {
    return degrain_interp_halfpel_wiener_h_from_samples(
        degrain_pixel_load(src, pitch, width, height, baseX - 2, baseY),
        degrain_pixel_load(src, pitch, width, height, baseX - 1, baseY),
        degrain_pixel_load(src, pitch, width, height, baseX, baseY),
        degrain_pixel_load(src, pitch, width, height, baseX + 1, baseY),
        degrain_pixel_load(src, pitch, width, height, baseX + 2, baseY),
        degrain_pixel_load(src, pitch, width, height, baseX + 3, baseY));
}

static inline int degrain_interp_halfpel_wiener_hv(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int baseY) {
    return degrain_interp_halfpel_wiener_h_from_samples(
        degrain_interp_halfpel_wiener_v(src, pitch, width, height, baseX - 2, baseY),
        degrain_interp_halfpel_wiener_v(src, pitch, width, height, baseX - 1, baseY),
        degrain_interp_halfpel_wiener_v(src, pitch, width, height, baseX, baseY),
        degrain_interp_halfpel_wiener_v(src, pitch, width, height, baseX + 1, baseY),
        degrain_interp_halfpel_wiener_v(src, pitch, width, height, baseX + 2, baseY),
        degrain_interp_halfpel_wiener_v(src, pitch, width, height, baseX + 3, baseY));
}

static inline int degrain_interp_halfpel_weighted(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int baseY,
    const int fracX,
    const int fracY,
    const int interpMode) {
    if (interpMode == 2) {
        if (fracX != 0 && fracY != 0) {
            return degrain_interp_halfpel_wiener_hv(src, pitch, width, height, baseX, baseY);
        }
        if (fracX != 0) {
            return degrain_interp_halfpel_wiener_h(src, pitch, width, height, baseX, baseY);
        }
        if (fracY != 0) {
            return degrain_interp_halfpel_wiener_v(src, pitch, width, height, baseX, baseY);
        }
    }

    const int offsets[4] = { -1, 0, 1, 2 };
    int weightsX[4] = { 0, 1, 0, 0 };
    int weightsY[4] = { 0, 1, 0, 0 };
    int denomXShift = 0;
    int denomYShift = 0;

    if (fracX != 0) {
        if (interpMode == 2) {
            weightsX[0] = -1; weightsX[1] = 9; weightsX[2] = 9; weightsX[3] = -1;
            denomXShift = 4;
        } else {
            weightsX[0] = 1; weightsX[1] = 3; weightsX[2] = 3; weightsX[3] = 1;
            denomXShift = 3;
        }
    }
    if (fracY != 0) {
        if (interpMode == 2) {
            weightsY[0] = -1; weightsY[1] = 9; weightsY[2] = 9; weightsY[3] = -1;
            denomYShift = 4;
        } else {
            weightsY[0] = 1; weightsY[1] = 3; weightsY[2] = 3; weightsY[3] = 1;
            denomYShift = 3;
        }
    }

    int sum = 0;
    for (int iy = 0; iy < 4; iy++) {
        if (weightsY[iy] == 0) {
            continue;
        }
        for (int ix = 0; ix < 4; ix++) {
            if (weightsX[ix] == 0) {
                continue;
            }
            const int sample = degrain_pixel_load(src, pitch, width, height, baseX + offsets[ix], baseY + offsets[iy]);
            sum += sample * weightsX[ix] * weightsY[iy];
        }
    }
    return degrain_clamp_pixel(degrain_round_rshift_signed(sum, denomXShift + denomYShift));
}

static inline int degrain_interp_halfpel_from_samples(
    const int s0,
    const int s1,
    const int s2,
    const int s3,
    const int s4,
    const int s5,
    const int interpMode) {
    if (interpMode == 2) {
        return degrain_interp_halfpel_wiener_h_from_samples(s0, s1, s2, s3, s4, s5);
    }
    const int sum = s1 + 3 * s2 + 3 * s3 + s4;
    return degrain_clamp_pixel((sum + 4) >> 3);
}

static inline int degrain_interp_halfpel_wiener_v_mirror(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int baseY) {
    const int sum =
        degrain_pixel_load_mirror(src, pitch, width, height, baseX, baseY - 2)
        - 5 * 
            (degrain_pixel_load_mirror(src, pitch, width, height, baseX, baseY - 1)
            + 4 * degrain_pixel_load_mirror(src, pitch, width, height, baseX, baseY)
            + 4 * degrain_pixel_load_mirror(src, pitch, width, height, baseX, baseY + 1)
            - degrain_pixel_load_mirror(src, pitch, width, height, baseX, baseY + 2))
        + degrain_pixel_load_mirror(src, pitch, width, height, baseX, baseY + 3);
    return degrain_clamp_pixel((sum + 16) >> 5);
}

static inline int degrain_interp_halfpel_wiener_h_mirror(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int baseY) {
    return degrain_interp_halfpel_wiener_h_from_samples(
        degrain_pixel_load_mirror(src, pitch, width, height, baseX - 2, baseY),
        degrain_pixel_load_mirror(src, pitch, width, height, baseX - 1, baseY),
        degrain_pixel_load_mirror(src, pitch, width, height, baseX, baseY),
        degrain_pixel_load_mirror(src, pitch, width, height, baseX + 1, baseY),
        degrain_pixel_load_mirror(src, pitch, width, height, baseX + 2, baseY),
        degrain_pixel_load_mirror(src, pitch, width, height, baseX + 3, baseY));
}

static inline int degrain_interp_halfpel_wiener_hv_mirror(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int baseY) {
    return degrain_interp_halfpel_wiener_h_from_samples(
        degrain_interp_halfpel_wiener_v_mirror(src, pitch, width, height, baseX - 2, baseY),
        degrain_interp_halfpel_wiener_v_mirror(src, pitch, width, height, baseX - 1, baseY),
        degrain_interp_halfpel_wiener_v_mirror(src, pitch, width, height, baseX, baseY),
        degrain_interp_halfpel_wiener_v_mirror(src, pitch, width, height, baseX + 1, baseY),
        degrain_interp_halfpel_wiener_v_mirror(src, pitch, width, height, baseX + 2, baseY),
        degrain_interp_halfpel_wiener_v_mirror(src, pitch, width, height, baseX + 3, baseY));
}

static inline int degrain_interp_halfpel_weighted_mirror(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int baseY,
    const int fracX,
    const int fracY,
    const int interpMode) {
    if (interpMode == 2) {
        if (fracX != 0 && fracY != 0) {
            return degrain_interp_halfpel_wiener_hv_mirror(src, pitch, width, height, baseX, baseY);
        }
        if (fracX != 0) {
            return degrain_interp_halfpel_wiener_h_mirror(src, pitch, width, height, baseX, baseY);
        }
        if (fracY != 0) {
            return degrain_interp_halfpel_wiener_v_mirror(src, pitch, width, height, baseX, baseY);
        }
    }

    const int offsets[4] = { -1, 0, 1, 2 };
    int weightsX[4] = { 0, 1, 0, 0 };
    int weightsY[4] = { 0, 1, 0, 0 };
    int denomXShift = 0;
    int denomYShift = 0;

    if (fracX != 0) {
        if (interpMode == 2) {
            weightsX[0] = -1; weightsX[1] = 9; weightsX[2] = 9; weightsX[3] = -1;
            denomXShift = 4;
        } else {
            weightsX[0] = 1; weightsX[1] = 3; weightsX[2] = 3; weightsX[3] = 1;
            denomXShift = 3;
        }
    }
    if (fracY != 0) {
        if (interpMode == 2) {
            weightsY[0] = -1; weightsY[1] = 9; weightsY[2] = 9; weightsY[3] = -1;
            denomYShift = 4;
        } else {
            weightsY[0] = 1; weightsY[1] = 3; weightsY[2] = 3; weightsY[3] = 1;
            denomYShift = 3;
        }
    }

    int sum = 0;
    for (int iy = 0; iy < 4; iy++) {
        if (weightsY[iy] == 0) {
            continue;
        }
        for (int ix = 0; ix < 4; ix++) {
            if (weightsX[ix] == 0) {
                continue;
            }
            const int sample = degrain_pixel_load_mirror(src, pitch, width, height, baseX + offsets[ix], baseY + offsets[iy]);
            sum += sample * weightsX[ix] * weightsY[iy];
        }
    }
    return degrain_clamp_pixel(degrain_round_rshift_signed(sum, denomXShift + denomYShift));
}

static inline int degrain_interp_pel4_h(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int y,
    const int fracX,
    const int interpMode) {
    if (fracX == 0) {
        return degrain_pixel_load(src, pitch, width, height, baseX, y);
    }

    const int halfPix = (interpMode == 2)
        ? degrain_interp_halfpel_wiener_h(src, pitch, width, height, baseX, y)
        : degrain_interp_halfpel_from_samples(
            degrain_pixel_load(src, pitch, width, height, baseX - 2, y),
            degrain_pixel_load(src, pitch, width, height, baseX - 1, y),
            degrain_pixel_load(src, pitch, width, height, baseX, y),
            degrain_pixel_load(src, pitch, width, height, baseX + 1, y),
            degrain_pixel_load(src, pitch, width, height, baseX + 2, y),
            degrain_pixel_load(src, pitch, width, height, baseX + 3, y),
            interpMode);
    if (fracX == 2) {
        return halfPix;
    }

    const int side = degrain_pixel_load(src, pitch, width, height, baseX + (fracX > 2 ? 1 : 0), y);
    return (side + halfPix + 1) >> 1;
}

static inline int degrain_interp_pel4(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int baseY,
    const int fracX,
    const int fracY,
    const int interpMode) {
    if ((fracX & 1) == 0 && (fracY & 1) == 0) {
        return degrain_interp_halfpel_weighted(src, pitch, width, height, baseX, baseY, fracX >> 1, fracY >> 1, interpMode);
    }
    if (fracY == 0) {
        return degrain_interp_pel4_h(src, pitch, width, height, baseX, baseY, fracX, interpMode);
    }

    const int halfPix = degrain_interp_halfpel_from_samples(
        degrain_interp_pel4_h(src, pitch, width, height, baseX, baseY - 2, fracX, interpMode),
        degrain_interp_pel4_h(src, pitch, width, height, baseX, baseY - 1, fracX, interpMode),
        degrain_interp_pel4_h(src, pitch, width, height, baseX, baseY,     fracX, interpMode),
        degrain_interp_pel4_h(src, pitch, width, height, baseX, baseY + 1, fracX, interpMode),
        degrain_interp_pel4_h(src, pitch, width, height, baseX, baseY + 2, fracX, interpMode),
        degrain_interp_pel4_h(src, pitch, width, height, baseX, baseY + 3, fracX, interpMode),
        interpMode);
    if (fracY == 2) {
        return halfPix;
    }

    const int side = degrain_interp_pel4_h(src, pitch, width, height, baseX, baseY + (fracY > 2 ? 1 : 0), fracX, interpMode);
    return (side + halfPix + 1) >> 1;
}

static inline int degrain_pixel_load_pel(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int xPel,
    const int yPel) {
    if (DEGRAIN_PEL <= 1) {
        return degrain_pixel_load(src, pitch, width, height, xPel, yPel);
    }

    const int baseX = degrain_floor_div_pel(xPel);
    const int baseY = degrain_floor_div_pel(yPel);
    const int fracX = degrain_floor_mod_pel(xPel, baseX);
    const int fracY = degrain_floor_mod_pel(yPel, baseY);
    if (fracX == 0 && fracY == 0) {
        return degrain_pixel_load(src, pitch, width, height, baseX, baseY);
    }

    if (DEGRAIN_PEL == 2 && DEGRAIN_SUBPEL_INTERP >= 1) {
        return degrain_interp_halfpel_weighted(src, pitch, width, height, baseX, baseY, fracX, fracY, DEGRAIN_SUBPEL_INTERP);
    }
    if (DEGRAIN_PEL == 4 && DEGRAIN_SUBPEL_INTERP >= 1) {
        return degrain_interp_pel4(src, pitch, width, height, baseX, baseY, fracX, fracY, DEGRAIN_SUBPEL_INTERP);
    }

    const int p00 = degrain_pixel_load(src, pitch, width, height, baseX,     baseY);
    const int p10 = degrain_pixel_load(src, pitch, width, height, baseX + 1, baseY);
    const int p01 = degrain_pixel_load(src, pitch, width, height, baseX,     baseY + 1);
    const int p11 = degrain_pixel_load(src, pitch, width, height, baseX + 1, baseY + 1);
    const int invX = DEGRAIN_PEL - fracX;
    const int invY = DEGRAIN_PEL - fracY;
    const int value = p00 * invX * invY
        + p10 * fracX * invY
        + p01 * invX * fracY
        + p11 * fracX * fracY;
    return degrain_round_rshift_signed(value, DEGRAIN_PEL_RSHIFT << 1);
}

static inline int degrain_pixel_load_pel_mirror(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int xPel,
    const int yPel) {
    if (DEGRAIN_PEL <= 1) {
        return degrain_pixel_load_mirror(src, pitch, width, height, xPel, yPel);
    }

    const int baseX = degrain_floor_div_pel(xPel);
    const int baseY = degrain_floor_div_pel(yPel);
    const int fracX = degrain_floor_mod_pel(xPel, baseX);
    const int fracY = degrain_floor_mod_pel(yPel, baseY);
    if (fracX == 0 && fracY == 0) {
        return degrain_pixel_load_mirror(src, pitch, width, height, baseX, baseY);
    }

    if (DEGRAIN_PEL == 2 && DEGRAIN_SUBPEL_INTERP >= 1) {
        return degrain_interp_halfpel_weighted_mirror(src, pitch, width, height, baseX, baseY, fracX, fracY, DEGRAIN_SUBPEL_INTERP);
    }

    const int p00 = degrain_pixel_load_mirror(src, pitch, width, height, baseX,     baseY);
    const int p10 = degrain_pixel_load_mirror(src, pitch, width, height, baseX + 1, baseY);
    const int p01 = degrain_pixel_load_mirror(src, pitch, width, height, baseX,     baseY + 1);
    const int p11 = degrain_pixel_load_mirror(src, pitch, width, height, baseX + 1, baseY + 1);
    const int invX = DEGRAIN_PEL - fracX;
    const int invY = DEGRAIN_PEL - fracY;
    const int value = p00 * invX * invY
        + p10 * fracX * invY
        + p01 * invX * fracY
        + p11 * fracX * fracY;
    return degrain_round_rshift_signed(value, DEGRAIN_PEL_RSHIFT << 1);
}

static inline int degrain_reference_is_valid(
    __global const degrain_mv_t *mv,
    __global const degrain_sad_t *sad,
    const int block,
    const int refDirection,
    const uint thsad,
    const int directionDisabled) {
    if (directionDisabled) {
        return 0;
    }
    const int clampedRefDirection = clamp(refDirection, 0, DEGRAIN_REFS - 1);
    const int index = degrain_ref_index(block, clampedRefDirection);
    return ((int)mv[index].refdir == clampedRefDirection) && (sad[index].sad < thsad);
}

static inline float degrain_overlap_blend_curve(const float phase) {
#if defined(DEGRAIN_FAST_OVERLAP_TRIG) && DEGRAIN_FAST_OVERLAP_TRIG
    const float c = native_cos(DEGRAIN_PI_F * phase);
#else
    const float c = cos(DEGRAIN_PI_F * phase);
#endif
    return 0.5f + 0.5f * c;
}

static inline float degrain_overlap_axis_gain(
    const int pos,
    const int blockSize,
    const int overlap,
    const int isFirst,
    const int isLast) {
    if (pos < 0 || pos >= blockSize) {
        return 0.0f;
    }
    if (overlap <= 0) {
        return 1.0f;
    }
    if (pos < overlap) {
        if (isFirst) {
            return 1.0f;
        }
        const float phase = ((float)pos + 0.5f) / (float)overlap;
        return degrain_overlap_blend_curve(phase);
    }
    if (pos >= blockSize - overlap) {
        if (isLast) {
            return 1.0f;
        }
        const float phase = ((float)(blockSize - pos) - 0.5f) / (float)overlap;
        return degrain_overlap_blend_curve(phase);
    }
    return 1.0f;
}

static inline float degrain_window_factor_rect_2d(
    const int x,
    const int y,
    const int baseX,
    const int baseY,
    const int blockSizeX,
    const int blockSizeY,
    const int overlapX,
    const int overlapY,
    const int blockX,
    const int blockY,
    const int blocksX,
    const int blocksY) {
    const int localX = x - baseX;
    const int localY = y - baseY;
    if (localX < 0 || localX >= blockSizeX || localY < 0 || localY >= blockSizeY) {
        return 0.0f;
    }
    const float wx = degrain_overlap_axis_gain(localX, blockSizeX, overlapX, blockX == 0, blockX == blocksX - 1);
    const float wy = degrain_overlap_axis_gain(localY, blockSizeY, overlapY, blockY == 0, blockY == blocksY - 1);
    return wx * wy;
}

static inline float degrain_reference_affinity_from_sad(
    const int sadLimit,
    const int blockSad) {
    if (sadLimit <= blockSad) {
        return 0.0f;
    }
    const float sadRatio = (float)blockSad / (float)sadLimit;
    const float sadRatio2 = sadRatio * sadRatio;
    return (1.0f - sadRatio2) / (1.0f + sadRatio2);
}

static inline float degrain_reference_mix_affinity(
    __global const degrain_mv_t *mv,
    __global const degrain_sad_t *sad,
    const int block,
    const int refDirection,
    const uint thsad,
    const int directionDisabled) {
    if (!degrain_reference_is_valid(mv, sad, block, refDirection, thsad, directionDisabled)) {
        return 0.0f;
    }
    return degrain_reference_affinity_from_sad((int)thsad, (int)sad[degrain_ref_index(block, refDirection)].sad);
}

static inline float degrain_temporal_mix_prior_center(__global const float *temporalMixPrior) {
    return temporalMixPrior[0];
}

static inline float degrain_temporal_mix_prior_ref(
    __global const float *temporalMixPrior,
    const int refDirection) {
    return temporalMixPrior[1 + refDirection];
}

static inline int degrain_trace_float_to_q8(const float value) {
    return convert_int_rte(value * 256.0f);
}

static inline int degrain_ref_direction_disabled(const uint disableMask, const int refDirection) {
    return ((disableMask >> refDirection) & 1u) != 0u;
}

static inline int degrain_compensated_sample(
    __global const uchar *ref,
    const int refPitch,
    const int width,
    const int height,
    __global const degrain_mv_t *mv,
    const int block,
    const int refDirection,
    const int planeScaleX,
    const int planeScaleY,
    const int x,
    const int y) {
    const int index = degrain_ref_index(block, refDirection);
    const degrain_mv_t motion = mv[index];
    const int scaledDx = degrain_floor_rshift_signed((int)motion.dx, degrain_plane_scale_rshift_x(planeScaleX));
    const int scaledDy = degrain_floor_rshift_signed((int)motion.dy, degrain_plane_scale_rshift_y(planeScaleY));
    if (DEGRAIN_PEL <= 1) {
        const int sampleX = x + scaledDx;
        const int sampleY = y + scaledDy;
        if ((uint)sampleX < (uint)width && (uint)sampleY < (uint)height) {
            return degrain_pixel_load_unchecked(ref, refPitch, sampleX, sampleY);
        }
        return degrain_pixel_load_mirror(ref, refPitch, width, height, sampleX, sampleY);
    }
    return degrain_pixel_load_pel_mirror(
        ref, refPitch, width, height,
        x * DEGRAIN_PEL + scaledDx,
        y * DEGRAIN_PEL + scaledDy);
}

static inline __global const uchar *degrain_ref_plane_ptr_same_pitch(
    __global const uchar *refBackward1,
    __global const uchar *refForward1,
    __global const uchar *refBackward2,
    __global const uchar *refForward2,
    __global const uchar *refBackward3,
    __global const uchar *refForward3,
    __global const uchar *refBackward4,
    __global const uchar *refForward4,
    __global const uchar *refBackward5,
    __global const uchar *refForward5,
    const int refDirection) {
#if DEGRAIN_REFS <= 2
    switch (refDirection) {
    case 0: return refBackward1;
    default: return refForward1;
    }
#elif DEGRAIN_REFS <= 4
    switch (refDirection) {
    case 0: return refBackward1;
    case 1: return refForward1;
    case 2: return refBackward2;
    default: return refForward2;
    }
#elif DEGRAIN_REFS <= 6
    switch (refDirection) {
    case 0: return refBackward1;
    case 1: return refForward1;
    case 2: return refBackward2;
    case 3: return refForward2;
    case 4: return refBackward3;
    default: return refForward3;
    }
#elif DEGRAIN_REFS <= 8
    switch (refDirection) {
    case 0: return refBackward1;
    case 1: return refForward1;
    case 2: return refBackward2;
    case 3: return refForward2;
    case 4: return refBackward3;
    case 5: return refForward3;
    case 6: return refBackward4;
    default: return refForward4;
    }
#else
    switch (refDirection) {
    case 0: return refBackward1;
    case 1: return refForward1;
    case 2: return refBackward2;
    case 3: return refForward2;
    case 4: return refBackward3;
    case 5: return refForward3;
    case 6: return refBackward4;
    case 7: return refForward4;
    case 8: return refBackward5;
    default: return refForward5;
    }
#endif
}

static inline int degrain_compensate_block_sample(
    __global const uchar *ref0,
    __global const uchar *ref,
    const int pitch,
    const int width,
    const int height,
    __global const degrain_mv_t *mv,
    __global const degrain_sad_t *sad,
    const int block,
    const int refDirection,
    const uint thsad,
    const int directionDisabled,
    const int planeScaleX,
    const int planeScaleY,
    const int x,
    const int y) {
    const int useReference = degrain_reference_is_valid(mv, sad, block, refDirection, thsad, directionDisabled);
    return useReference
        ? degrain_compensated_sample(ref, pitch, width, height, mv, block, refDirection, planeScaleX, planeScaleY, x, y)
        : degrain_pixel_load(ref0, pitch, width, height, x, y);
}

static inline int degrain_degrain_block_sample(
    __global const uchar *cur,
    const int pitch,
    __global const uchar *refBackward1,
    __global const uchar *refForward1,
    __global const uchar *refBackward2,
    __global const uchar *refForward2,
    __global const uchar *refBackward3,
    __global const uchar *refForward3,
    __global const uchar *refBackward4,
    __global const uchar *refForward4,
    __global const uchar *refBackward5,
    __global const uchar *refForward5,
    const int width,
    const int height,
    __global const degrain_mv_t *mv,
    __global const degrain_sad_t *sad,
    const int block,
    const uint thsad,
    const uint disableMask,
    __global const float *temporalMixPrior,
    const int planeScaleX,
    const int planeScaleY,
    const int x,
    const int y) {
    const int currentSample = degrain_pixel_load(cur, pitch, width, height, x, y);
    const float sourceConfidenceRaw = degrain_temporal_mix_prior_center(temporalMixPrior);
    float referenceConfidenceRaw[DEGRAIN_REFS];
    float confidenceTotal = sourceConfidenceRaw;
    for (int referenceDirection = 0; referenceDirection < DEGRAIN_REFS; referenceDirection++) {
        const float temporalMixPriorRef = degrain_temporal_mix_prior_ref(temporalMixPrior, referenceDirection);
        referenceConfidenceRaw[referenceDirection] = degrain_reference_mix_affinity(mv, sad, block, referenceDirection, thsad, degrain_ref_direction_disabled(disableMask, referenceDirection)) * temporalMixPriorRef;
        confidenceTotal += referenceConfidenceRaw[referenceDirection];
    }
    const float invTotal = (confidenceTotal > 0.0f) ? (1.0f / confidenceTotal) : 0.0f;
    float mixedValue = (float)currentSample * (sourceConfidenceRaw * invTotal);
    for (int referenceDirection = 0; referenceDirection < DEGRAIN_REFS; referenceDirection++) {
        if (referenceConfidenceRaw[referenceDirection] <= 0.0f) {
            continue;
        }
        const float referenceMixNorm = referenceConfidenceRaw[referenceDirection] * invTotal;
        __global const uchar *referencePlane = degrain_ref_plane_ptr_same_pitch(
            refBackward1, refForward1,
            refBackward2, refForward2,
            refBackward3, refForward3,
            refBackward4, refForward4,
            refBackward5, refForward5,
            referenceDirection);
        const float referenceSample = (float)degrain_compensated_sample(referencePlane, pitch, width, height, mv, block, referenceDirection, planeScaleX, planeScaleY, x, y);
        mixedValue = fma(referenceSample, referenceMixNorm, mixedValue);
    }
    return degrain_clamp_pixel(convert_int_rte(mixedValue));
}

static inline int degrain_temporal_mix_plan_offset(const int block) {
    return block * (DEGRAIN_REFS + 1);
}

static inline float degrain_temporal_mix_plan_src(
    __global const float *temporalMixPlan,
    const int planOffset) {
    return temporalMixPlan[planOffset];
}

static inline float degrain_temporal_mix_plan_ref(
    __global const float *temporalMixPlan,
    const int planOffset,
    const int refDirection) {
    return temporalMixPlan[planOffset + 1 + refDirection];
}

static inline int degrain_apply_temporal_mix_plan_same_pitch(
    __global const uchar *cur,
    const int pitch,
    __global const uchar *refBackward1,
    __global const uchar *refForward1,
    __global const uchar *refBackward2,
    __global const uchar *refForward2,
    __global const uchar *refBackward3,
    __global const uchar *refForward3,
    __global const uchar *refBackward4,
    __global const uchar *refForward4,
    __global const uchar *refBackward5,
    __global const uchar *refForward5,
    const int width,
    const int height,
    __global const degrain_mv_t *mv,
    const int block,
    __global const float *temporalMixPlan,
    const int planeScaleX,
    const int planeScaleY,
    const int x,
    const int y) {
    const int srcSample = degrain_pixel_load(cur, pitch, width, height, x, y);
    const int planOffset = degrain_temporal_mix_plan_offset(block);
    float value = (float)srcSample * degrain_temporal_mix_plan_src(temporalMixPlan, planOffset);
    for (int refDirection = 0; refDirection < DEGRAIN_REFS; refDirection++) {
        const float referenceMixNorm = degrain_temporal_mix_plan_ref(temporalMixPlan, planOffset, refDirection);
        if (referenceMixNorm <= 0.0f) {
            continue;
        }
        __global const uchar *ref = degrain_ref_plane_ptr_same_pitch(
            refBackward1, refForward1,
            refBackward2, refForward2,
            refBackward3, refForward3,
            refBackward4, refForward4,
            refBackward5, refForward5,
            refDirection);
        const float refSample = (float)degrain_compensated_sample(ref, pitch, width, height, mv, block, refDirection, planeScaleX, planeScaleY, x, y);
        value = fma(refSample, referenceMixNorm, value);
    }
    return degrain_clamp_pixel(convert_int_rte(value));
}

static inline int degrain_windowed_sample_contribution(
    const int sample,
    const float windowWeight) {
    return convert_int_rte((float)sample * windowWeight);
}

typedef float degrain_window_accum_t;

static inline degrain_window_accum_t degrain_window_accum_zero() {
    return 0.0f;
}

static inline void degrain_accumulate_windowed_sample(
    __private degrain_window_accum_t *sampleSum,
    __private degrain_window_accum_t *weightSum,
    const int sample,
    const float windowWeight) {
    if (windowWeight > 0.0f) {
        *sampleSum = fma((float)sample, windowWeight, *sampleSum);
        *weightSum += windowWeight;
    }
}

static inline int degrain_finalize_windowed_sample(
    const degrain_window_accum_t sampleSum,
    const degrain_window_accum_t weightSum,
    const int fallback) {
    return (weightSum > 0.0f) ? convert_int_rte(sampleSum / weightSum) : fallback;
}

static inline int degrain_trace_window_accum(const degrain_window_accum_t sampleSum) {
    return convert_int_rte(sampleSum);
}

static inline void degrain_accumulate_weighted_sample_fp32(
    __private float *sampleSum,
    __private float *weightSum,
    const int sample,
    const float weight) {
    if (weight > 0.0f) {
        *sampleSum = fma((float)sample, weight, *sampleSum);
        *weightSum += weight;
    }
}

static inline int degrain_finalize_weighted_sample_fp32(
    const float sampleSum,
    const float weightSum,
    const int fallback) {
    return (weightSum > 0.0f) ? convert_int_rte(sampleSum / weightSum) : fallback;
}
