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

static inline int read_pix_clamped(
    const __global uchar *src, int x, int y,
    const int pitch, const int width, const int height
) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    return (int)(*(const __global Type *)(src + y * pitch + x * (int)sizeof(Type)));
}

static inline void write_pix(
    __global uchar *dst, int x, int y, const int pitch, const int value
) {
    *(__global Type *)(dst + y * pitch + x * (int)sizeof(Type)) = (Type)clamp(value, 0, max_val);
}

typedef struct {
    short dx;
    short dy;
    ushort sad;
    ushort refdir;
    uint flags;
    uint reserved;
} rtgmc_degrain_mv_t;

typedef struct {
    uint sad;
    uint srcAvg;
    uint refAvg;
    uint reserved;
} rtgmc_degrain_sad_t;

static inline Type rtgmc_degrain_clamp_pixel(const int value) {
    return (Type)clamp(value, 0, max_val);
}

static inline int rtgmc_degrain_floor_rshift_signed(const int value, const int rshift) {
    if (rshift <= 0) {
        return value;
    }
    return value >= 0
        ? value >> rshift
        : -(((-value) + (1 << rshift) - 1) >> rshift);
}

static inline int rtgmc_degrain_round_rshift_signed(const int value, const int rshift) {
    if (rshift <= 0) {
        return value;
    }
    return value >= 0
        ? (value + (1 << (rshift - 1))) >> rshift
        : -(((-value) + (1 << (rshift - 1))) >> rshift);
}

static inline int rtgmc_degrain_pel_rshift(const int pel) {
    return pel == 4 ? 2 : (pel == 2 ? 1 : 0);
}

static inline int rtgmc_degrain_floor_div_pel(const int value, const int pel) {
    return pel <= 1 ? value : rtgmc_degrain_floor_rshift_signed(value, rtgmc_degrain_pel_rshift(pel));
}

static inline int rtgmc_degrain_floor_mod_pel(const int value, const int base, const int pel) {
    return value - (base << rtgmc_degrain_pel_rshift(pel));
}

static inline int rtgmc_degrain_plane_scale_rshift(const int planeScale) {
    return planeScale > 1 ? 1 : 0;
}

static inline int rtgmc_degrain_mirror_coord(const int value, const int size) {
    const int reflectedLow = max(value, -value - 1);
    const int reflectedHigh = min(reflectedLow, 2 * size - 1 - value);
    return clamp(reflectedHigh, 0, size - 1);
}

static inline int rtgmc_degrain_pixel_load(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int x,
    const int y) {
    const int px = clamp(x, 0, width - 1);
    const int py = clamp(y, 0, height - 1);
    return (int)(*(__global const Type *)(src + py * pitch + px * (int)sizeof(Type)));
}

static inline int rtgmc_degrain_pixel_load_unchecked(
    __global const uchar *src,
    const int pitch,
    const int x,
    const int y) {
    return (int)(*(__global const Type *)(src + y * pitch + x * (int)sizeof(Type)));
}

static inline int rtgmc_degrain_pixel_load_mirror(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int x,
    const int y) {
    const int px = rtgmc_degrain_mirror_coord(x, width);
    const int py = rtgmc_degrain_mirror_coord(y, height);
    return (int)(*(__global const Type *)(src + py * pitch + px * (int)sizeof(Type)));
}

static inline int rtgmc_degrain_interp_halfpel_wiener_v(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int baseY) {
    const int s0 = rtgmc_degrain_pixel_load(src, pitch, width, height, baseX, baseY - 2);
    const int s1 = rtgmc_degrain_pixel_load(src, pitch, width, height, baseX, baseY - 1);
    const int s2 = rtgmc_degrain_pixel_load(src, pitch, width, height, baseX, baseY);
    const int s3 = rtgmc_degrain_pixel_load(src, pitch, width, height, baseX, baseY + 1);
    const int s4 = rtgmc_degrain_pixel_load(src, pitch, width, height, baseX, baseY + 2);
    const int s5 = rtgmc_degrain_pixel_load(src, pitch, width, height, baseX, baseY + 3);
    const int sum = s0 + 5 * (-s1 + (s2 << 2) + (s3 << 2) - s4) + s5;
    return rtgmc_degrain_clamp_pixel((sum + 16) >> 5);
}

static inline int rtgmc_degrain_interp_halfpel_wiener_h_from_samples(
    const int s0,
    const int s1,
    const int s2,
    const int s3,
    const int s4,
    const int s5) {
    const int sum = s0 + 5 * (-s1 + (s2 << 2) + (s3 << 2) - s4) + s5;
    return rtgmc_degrain_clamp_pixel((sum + 16) >> 5);
}

static inline int rtgmc_degrain_interp_halfpel_wiener_h(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int baseY) {
    return rtgmc_degrain_interp_halfpel_wiener_h_from_samples(
        rtgmc_degrain_pixel_load(src, pitch, width, height, baseX - 2, baseY),
        rtgmc_degrain_pixel_load(src, pitch, width, height, baseX - 1, baseY),
        rtgmc_degrain_pixel_load(src, pitch, width, height, baseX, baseY),
        rtgmc_degrain_pixel_load(src, pitch, width, height, baseX + 1, baseY),
        rtgmc_degrain_pixel_load(src, pitch, width, height, baseX + 2, baseY),
        rtgmc_degrain_pixel_load(src, pitch, width, height, baseX + 3, baseY));
}

static inline int rtgmc_degrain_interp_halfpel_wiener_hv(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int baseY) {
    return rtgmc_degrain_interp_halfpel_wiener_h_from_samples(
        rtgmc_degrain_interp_halfpel_wiener_v(src, pitch, width, height, baseX - 2, baseY),
        rtgmc_degrain_interp_halfpel_wiener_v(src, pitch, width, height, baseX - 1, baseY),
        rtgmc_degrain_interp_halfpel_wiener_v(src, pitch, width, height, baseX, baseY),
        rtgmc_degrain_interp_halfpel_wiener_v(src, pitch, width, height, baseX + 1, baseY),
        rtgmc_degrain_interp_halfpel_wiener_v(src, pitch, width, height, baseX + 2, baseY),
        rtgmc_degrain_interp_halfpel_wiener_v(src, pitch, width, height, baseX + 3, baseY));
}

static inline int rtgmc_degrain_interp_halfpel_wiener_v_mirror(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int baseY) {
    const int sum =
        rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX, baseY - 2)
        - 5 *
            (rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX, baseY - 1)
            + 4 * rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX, baseY)
            + 4 * rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX, baseY + 1)
            - rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX, baseY + 2))
        + rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX, baseY + 3);
    return rtgmc_degrain_clamp_pixel((sum + 16) >> 5);
}

static inline int rtgmc_degrain_interp_halfpel_wiener_h_mirror(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int baseY) {
    return rtgmc_degrain_interp_halfpel_wiener_h_from_samples(
        rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX - 2, baseY),
        rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX - 1, baseY),
        rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX, baseY),
        rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX + 1, baseY),
        rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX + 2, baseY),
        rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX + 3, baseY));
}

static inline int rtgmc_degrain_interp_halfpel_wiener_hv_mirror(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int baseY) {
    return rtgmc_degrain_interp_halfpel_wiener_h_from_samples(
        rtgmc_degrain_interp_halfpel_wiener_v_mirror(src, pitch, width, height, baseX - 2, baseY),
        rtgmc_degrain_interp_halfpel_wiener_v_mirror(src, pitch, width, height, baseX - 1, baseY),
        rtgmc_degrain_interp_halfpel_wiener_v_mirror(src, pitch, width, height, baseX, baseY),
        rtgmc_degrain_interp_halfpel_wiener_v_mirror(src, pitch, width, height, baseX + 1, baseY),
        rtgmc_degrain_interp_halfpel_wiener_v_mirror(src, pitch, width, height, baseX + 2, baseY),
        rtgmc_degrain_interp_halfpel_wiener_v_mirror(src, pitch, width, height, baseX + 3, baseY));
}

static inline int rtgmc_degrain_interp_halfpel_weighted(
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
            return rtgmc_degrain_interp_halfpel_wiener_hv(src, pitch, width, height, baseX, baseY);
        }
        if (fracX != 0) {
            return rtgmc_degrain_interp_halfpel_wiener_h(src, pitch, width, height, baseX, baseY);
        }
        if (fracY != 0) {
            return rtgmc_degrain_interp_halfpel_wiener_v(src, pitch, width, height, baseX, baseY);
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
            const int sample = rtgmc_degrain_pixel_load(src, pitch, width, height, baseX + offsets[ix], baseY + offsets[iy]);
            sum += sample * weightsX[ix] * weightsY[iy];
        }
    }
    return rtgmc_degrain_clamp_pixel(rtgmc_degrain_round_rshift_signed(sum, denomXShift + denomYShift));
}

static inline int rtgmc_degrain_interp_halfpel_weighted_mirror(
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
            return rtgmc_degrain_interp_halfpel_wiener_hv_mirror(src, pitch, width, height, baseX, baseY);
        }
        if (fracX != 0) {
            return rtgmc_degrain_interp_halfpel_wiener_h_mirror(src, pitch, width, height, baseX, baseY);
        }
        if (fracY != 0) {
            return rtgmc_degrain_interp_halfpel_wiener_v_mirror(src, pitch, width, height, baseX, baseY);
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
            const int sample = rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX + offsets[ix], baseY + offsets[iy]);
            sum += sample * weightsX[ix] * weightsY[iy];
        }
    }
    return rtgmc_degrain_clamp_pixel(rtgmc_degrain_round_rshift_signed(sum, denomXShift + denomYShift));
}

static inline int rtgmc_degrain_interp_halfpel_from_samples(
    const int s0,
    const int s1,
    const int s2,
    const int s3,
    const int s4,
    const int s5,
    const int interpMode) {
    if (interpMode == 2) {
        return rtgmc_degrain_interp_halfpel_wiener_h_from_samples(s0, s1, s2, s3, s4, s5);
    }
    const int sum = s1 + 3 * s2 + 3 * s3 + s4;
    return rtgmc_degrain_clamp_pixel((sum + 4) >> 3);
}

static inline int rtgmc_degrain_interp_pel4_h(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int y,
    const int fracX,
    const int interpMode) {
    if (fracX == 0) {
        return rtgmc_degrain_pixel_load(src, pitch, width, height, baseX, y);
    }

    const int halfPix = (interpMode == 2)
        ? rtgmc_degrain_interp_halfpel_wiener_h(src, pitch, width, height, baseX, y)
        : rtgmc_degrain_interp_halfpel_from_samples(
            rtgmc_degrain_pixel_load(src, pitch, width, height, baseX - 2, y),
            rtgmc_degrain_pixel_load(src, pitch, width, height, baseX - 1, y),
            rtgmc_degrain_pixel_load(src, pitch, width, height, baseX, y),
            rtgmc_degrain_pixel_load(src, pitch, width, height, baseX + 1, y),
            rtgmc_degrain_pixel_load(src, pitch, width, height, baseX + 2, y),
            rtgmc_degrain_pixel_load(src, pitch, width, height, baseX + 3, y),
            interpMode);
    if (fracX == 2) {
        return halfPix;
    }

    const int side = rtgmc_degrain_pixel_load(src, pitch, width, height, baseX + (fracX > 2 ? 1 : 0), y);
    return (side + halfPix + 1) >> 1;
}

static inline int rtgmc_degrain_interp_pel4(
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
        return rtgmc_degrain_interp_halfpel_weighted(src, pitch, width, height, baseX, baseY, fracX >> 1, fracY >> 1, interpMode);
    }
    if (fracY == 0) {
        return rtgmc_degrain_interp_pel4_h(src, pitch, width, height, baseX, baseY, fracX, interpMode);
    }

    const int halfPix = rtgmc_degrain_interp_halfpel_from_samples(
        rtgmc_degrain_interp_pel4_h(src, pitch, width, height, baseX, baseY - 2, fracX, interpMode),
        rtgmc_degrain_interp_pel4_h(src, pitch, width, height, baseX, baseY - 1, fracX, interpMode),
        rtgmc_degrain_interp_pel4_h(src, pitch, width, height, baseX, baseY,     fracX, interpMode),
        rtgmc_degrain_interp_pel4_h(src, pitch, width, height, baseX, baseY + 1, fracX, interpMode),
        rtgmc_degrain_interp_pel4_h(src, pitch, width, height, baseX, baseY + 2, fracX, interpMode),
        rtgmc_degrain_interp_pel4_h(src, pitch, width, height, baseX, baseY + 3, fracX, interpMode),
        interpMode);
    if (fracY == 2) {
        return halfPix;
    }

    const int side = rtgmc_degrain_interp_pel4_h(src, pitch, width, height, baseX, baseY + (fracY > 2 ? 1 : 0), fracX, interpMode);
    return (side + halfPix + 1) >> 1;
}

static inline int rtgmc_degrain_interp_pel4_h_mirror(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int baseX,
    const int y,
    const int fracX,
    const int interpMode) {
    if (fracX == 0) {
        return rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX, y);
    }

    const int halfPix = (interpMode == 2)
        ? rtgmc_degrain_interp_halfpel_wiener_h_mirror(src, pitch, width, height, baseX, y)
        : rtgmc_degrain_interp_halfpel_from_samples(
            rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX - 2, y),
            rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX - 1, y),
            rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX, y),
            rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX + 1, y),
            rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX + 2, y),
            rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX + 3, y),
            interpMode);
    if (fracX == 2) {
        return halfPix;
    }

    const int side = rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX + (fracX > 2 ? 1 : 0), y);
    return (side + halfPix + 1) >> 1;
}

static inline int rtgmc_degrain_interp_pel4_mirror(
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
        return rtgmc_degrain_interp_halfpel_weighted_mirror(src, pitch, width, height, baseX, baseY, fracX >> 1, fracY >> 1, interpMode);
    }
    if (fracY == 0) {
        return rtgmc_degrain_interp_pel4_h_mirror(src, pitch, width, height, baseX, baseY, fracX, interpMode);
    }

    const int halfPix = rtgmc_degrain_interp_halfpel_from_samples(
        rtgmc_degrain_interp_pel4_h_mirror(src, pitch, width, height, baseX, baseY - 2, fracX, interpMode),
        rtgmc_degrain_interp_pel4_h_mirror(src, pitch, width, height, baseX, baseY - 1, fracX, interpMode),
        rtgmc_degrain_interp_pel4_h_mirror(src, pitch, width, height, baseX, baseY,     fracX, interpMode),
        rtgmc_degrain_interp_pel4_h_mirror(src, pitch, width, height, baseX, baseY + 1, fracX, interpMode),
        rtgmc_degrain_interp_pel4_h_mirror(src, pitch, width, height, baseX, baseY + 2, fracX, interpMode),
        rtgmc_degrain_interp_pel4_h_mirror(src, pitch, width, height, baseX, baseY + 3, fracX, interpMode),
        interpMode);
    if (fracY == 2) {
        return halfPix;
    }

    const int side = rtgmc_degrain_interp_pel4_h_mirror(src, pitch, width, height, baseX, baseY + (fracY > 2 ? 1 : 0), fracX, interpMode);
    return (side + halfPix + 1) >> 1;
}

static inline int rtgmc_degrain_pixel_load_pel_mirror(
    __global const uchar *src,
    const int pitch,
    const int width,
    const int height,
    const int xPel,
    const int yPel,
    const int pel,
    const int subpelInterp) {
    if (pel <= 1) {
        return rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, xPel, yPel);
    }

    const int baseX = rtgmc_degrain_floor_div_pel(xPel, pel);
    const int baseY = rtgmc_degrain_floor_div_pel(yPel, pel);
    const int fracX = rtgmc_degrain_floor_mod_pel(xPel, baseX, pel);
    const int fracY = rtgmc_degrain_floor_mod_pel(yPel, baseY, pel);
    if (fracX == 0 && fracY == 0) {
        return rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX, baseY);
    }

    if (pel == 2 && subpelInterp >= 1) {
        return rtgmc_degrain_interp_halfpel_weighted_mirror(src, pitch, width, height, baseX, baseY, fracX, fracY, subpelInterp);
    }
    if (pel == 4 && subpelInterp >= 1) {
        return rtgmc_degrain_interp_pel4_mirror(src, pitch, width, height, baseX, baseY, fracX, fracY, subpelInterp);
    }

    const int p00 = rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX,     baseY);
    const int p10 = rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX + 1, baseY);
    const int p01 = rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX,     baseY + 1);
    const int p11 = rtgmc_degrain_pixel_load_mirror(src, pitch, width, height, baseX + 1, baseY + 1);
    const int invX = pel - fracX;
    const int invY = pel - fracY;
    const int value = p00 * invX * invY
        + p10 * fracX * invY
        + p01 * invX * fracY
        + p11 * fracX * fracY;
    return rtgmc_degrain_round_rshift_signed(value, rtgmc_degrain_pel_rshift(pel) << 1);
}

static inline int rtgmc_degrain_compensated_sample(
    __global const uchar *ref,
    const int refPitch,
    const int width,
    const int height,
    __global const rtgmc_degrain_mv_t *mv,
    const int block,
    const int refDirection,
    const int refs,
    const int planeScaleX,
    const int planeScaleY,
    const int x,
    const int y,
    const int pel,
    const int subpelInterp) {
    const int clampedRefDirection = clamp(refDirection, 0, refs - 1);
    const int index = block * refs + clampedRefDirection;
    const rtgmc_degrain_mv_t motion = mv[index];
    const int scaledDx = rtgmc_degrain_floor_rshift_signed((int)motion.dx, rtgmc_degrain_plane_scale_rshift(max(planeScaleX, 1)));
    const int scaledDy = rtgmc_degrain_floor_rshift_signed((int)motion.dy, rtgmc_degrain_plane_scale_rshift(max(planeScaleY, 1)));
    if (pel <= 1) {
        const int sampleX = x + scaledDx;
        const int sampleY = y + scaledDy;
        if ((uint)sampleX < (uint)width && (uint)sampleY < (uint)height) {
            return rtgmc_degrain_pixel_load_unchecked(ref, refPitch, sampleX, sampleY);
        }
        return rtgmc_degrain_pixel_load_mirror(ref, refPitch, width, height, sampleX, sampleY);
    }
    return rtgmc_degrain_pixel_load_pel_mirror(
        ref, refPitch, width, height,
        x * pel + scaledDx,
        y * pel + scaledDy,
        pel, subpelInterp);
}

static inline int rtgmc_degrain_ref_direction_disabled(const uint disableMask, const int refDirection) {
    return ((disableMask >> refDirection) & 1u) != 0u;
}

static inline int rtgmc_degrain_reference_is_valid(
    __global const rtgmc_degrain_mv_t *mv,
    __global const rtgmc_degrain_sad_t *sad,
    const int block,
    const int refDirection,
    const int refs,
    const uint thsad,
    const int directionDisabled) {
    if (directionDisabled) {
        return 0;
    }
    const int clampedRefDirection = clamp(refDirection, 0, refs - 1);
    const int index = block * refs + clampedRefDirection;
    return ((int)mv[index].refdir == clampedRefDirection) && (sad[index].sad < thsad);
}

static inline int rtgmc_degrain_compensate_block_sample(
    __global const uchar *ref0,
    __global const uchar *ref,
    const int pitch,
    const int width,
    const int height,
    __global const rtgmc_degrain_mv_t *mv,
    __global const rtgmc_degrain_sad_t *sad,
    const int block,
    const int refDirection,
    const int refs,
    const uint thsad,
    const int directionDisabled,
    const int planeScaleX,
    const int planeScaleY,
    const int x,
    const int y,
    const int pel,
    const int subpelInterp) {
    const int useReference = rtgmc_degrain_reference_is_valid(mv, sad, block, refDirection, refs, thsad, directionDisabled);
    return useReference
        ? rtgmc_degrain_compensated_sample(ref, pitch, width, height, mv, block, refDirection, refs, planeScaleX, planeScaleY, x, y, pel, subpelInterp)
        : rtgmc_degrain_pixel_load(ref0, pitch, width, height, x, y);
}

static inline int rtgmc_degrain_scale_floor(const int value, const int scale) {
    return value / max(scale, 1);
}

static inline int rtgmc_degrain_block_origin(const int block, const int step) {
    return block * max(step, 1);
}

static inline int rtgmc_degrain_is_covered_pixel(
    const int x,
    const int y,
    const int coveredWidth,
    const int coveredHeight) {
    return x < coveredWidth && y < coveredHeight;
}

static inline void rtgmc_degrain_accumulate_weighted_sample_fp32(
    __private float *sampleSum,
    __private float *weightSum,
    const int sample,
    const float weight) {
    if (weight > 0.0f) {
        *sampleSum = fma((float)sample, weight, *sampleSum);
        *weightSum += weight;
    }
}

static inline int rtgmc_degrain_finalize_weighted_sample_fp32(
    const float sampleSum,
    const float weightSum,
    const int fallback) {
    return (weightSum > 0.0f) ? convert_int_rte(sampleSum / weightSum) : fallback;
}

static inline int rtgmc_degrain_compensate_overlap_pixel_value(
    __global const uchar *cur,
    const int curPitch,
    __global const uchar *ref0,
    __global const uchar *ref,
    const int refDirection,
    const int width,
    const int height,
    __global const rtgmc_degrain_mv_t *mv,
    __global const rtgmc_degrain_sad_t *sad,
    const int blocksX,
    const int blocksY,
    const int blockSize,
    const int overlap,
    const int step,
    const int coveredWidth,
    const int coveredHeight,
    const int planeScaleX,
    const int planeScaleY,
    const uint thsad,
    const uint disableMask,
    __global const float *windowRamp,
    const int x,
    const int y,
    const int refs,
    const int pel,
    const int subpelInterp) {
    const int fallback = rtgmc_degrain_pixel_load(cur, curPitch, width, height, x, y);
    const int scaleX = max(planeScaleX, 1);
    const int scaleY = max(planeScaleY, 1);
    const int renderBlockSize = blockSize;
    const int renderOverlap = overlap;
    const int renderStep = step;
    const int renderBlocksX = blocksX;
    const int renderBlocksY = blocksY;
    const int renderCoveredWidth = coveredWidth;
    const int renderCoveredHeight = coveredHeight;
    if (!rtgmc_degrain_is_covered_pixel(x, y, renderCoveredWidth, renderCoveredHeight)) {
        return fallback;
    }

    const int planeBlockSizeX = max(rtgmc_degrain_scale_floor(renderBlockSize, scaleX), 1);
    const int planeBlockSizeY = max(rtgmc_degrain_scale_floor(renderBlockSize, scaleY), 1);
    const int planeOverlapX = max(rtgmc_degrain_scale_floor(renderOverlap, scaleX), 0);
    const int planeOverlapY = max(rtgmc_degrain_scale_floor(renderOverlap, scaleY), 0);
    const int planeStepX = max(rtgmc_degrain_scale_floor(renderStep, scaleX), 1);
    const int planeStepY = max(rtgmc_degrain_scale_floor(renderStep, scaleY), 1);
    const int primaryBlockX = min(x / planeStepX, renderBlocksX - 1);
    const int primaryBlockY = min(y / planeStepY, renderBlocksY - 1);
    const int primaryBaseX = rtgmc_degrain_block_origin(primaryBlockX, planeStepX);
    const int primaryBaseY = rtgmc_degrain_block_origin(primaryBlockY, planeStepY);
    const int primaryLocalX = x - primaryBaseX;
    const int primaryLocalY = y - primaryBaseY;
    const int primaryBlock = primaryBlockY * renderBlocksX + primaryBlockX;
    const int usePrevBlockX = planeOverlapX > 0 && primaryBlockX > 0 && primaryLocalX < planeOverlapX;
    const int usePrevBlockY = planeOverlapY > 0 && primaryBlockY > 0 && primaryLocalY < planeOverlapY;
    const float wxPrev = usePrevBlockX ? windowRamp[primaryLocalX] : 0.0f;
    const float wyPrev = usePrevBlockY ? windowRamp[planeOverlapX + primaryLocalY] : 0.0f;
    const float wx[2] = { 1.0f - wxPrev, wxPrev };
    const float wy[2] = { 1.0f - wyPrev, wyPrev };

    const int blockXs[2] = { primaryBlockX, primaryBlockX - 1 };
    const int blockYs[2] = { primaryBlockY, primaryBlockY - 1 };
    const int localXs[2] = { primaryLocalX, primaryLocalX + planeStepX };
    const int localYs[2] = { primaryLocalY, primaryLocalY + planeStepY };
    const int blockRows[2] = { primaryBlock, primaryBlock - renderBlocksX };
    const int blockCountX = usePrevBlockX ? 2 : 1;
    const int blockCountY = usePrevBlockY ? 2 : 1;
    const int directionDisabled = rtgmc_degrain_ref_direction_disabled(disableMask, refDirection);

    float sampleSum = 0.0f;
    float weightSum = 0.0f;
    for (int byIndex = 0; byIndex < blockCountY; byIndex++) {
        const int blockY = blockYs[byIndex];
        const int localY = localYs[byIndex];
        const int blockRow = blockRows[byIndex];
        for (int bxIndex = 0; bxIndex < blockCountX; bxIndex++) {
            const int blockX = blockXs[bxIndex];
            const int localX = localXs[bxIndex];
            if (localX < 0 || localX >= planeBlockSizeX || localY < 0 || localY >= planeBlockSizeY
                || blockX < 0 || blockX >= renderBlocksX || blockY < 0 || blockY >= renderBlocksY) {
                continue;
            }
            const int block = blockRow - bxIndex;
            const int sample = rtgmc_degrain_compensate_block_sample(
                ref0, ref, curPitch,
                width, height,
                mv, sad,
                block, refDirection, refs, thsad, directionDisabled,
                planeScaleX, planeScaleY,
                x, y,
                pel, subpelInterp);
            rtgmc_degrain_accumulate_weighted_sample_fp32(&sampleSum, &weightSum, sample, wx[bxIndex] * wy[byIndex]);
        }
    }

    return rtgmc_degrain_finalize_weighted_sample_fp32(sampleSum, weightSum, fallback);
}

static inline int rtgmc_retouch_median3(const int a, const int b, const int c) {
    const int lo = min(a, b);
    const int hi = max(a, b);
    return max(lo, min(hi, c));
}

static inline void rtgmc_retouch_sort2(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = lo;
    *b = hi;
}

static inline void rtgmc_retouch_sort2_desc(int *a, int *b) {
    const int lo = min(*a, *b);
    const int hi = max(*a, *b);
    *a = hi;
    *b = lo;
}

// Batcher's Bitonic Sort (1968), 8 elements / 24 comparisons / depth 6.
static inline void rtgmc_retouch_sort8(__private int *v) {
    rtgmc_retouch_sort2     (&v[0], &v[1]); rtgmc_retouch_sort2_desc(&v[2], &v[3]); rtgmc_retouch_sort2     (&v[4], &v[5]); rtgmc_retouch_sort2_desc(&v[6], &v[7]);
    rtgmc_retouch_sort2     (&v[0], &v[2]); rtgmc_retouch_sort2     (&v[1], &v[3]); rtgmc_retouch_sort2_desc(&v[4], &v[6]); rtgmc_retouch_sort2_desc(&v[5], &v[7]);
    rtgmc_retouch_sort2     (&v[0], &v[1]); rtgmc_retouch_sort2     (&v[2], &v[3]); rtgmc_retouch_sort2_desc(&v[4], &v[5]); rtgmc_retouch_sort2_desc(&v[6], &v[7]);
    rtgmc_retouch_sort2     (&v[0], &v[4]); rtgmc_retouch_sort2     (&v[1], &v[5]); rtgmc_retouch_sort2     (&v[2], &v[6]); rtgmc_retouch_sort2     (&v[3], &v[7]);
    rtgmc_retouch_sort2     (&v[0], &v[2]); rtgmc_retouch_sort2     (&v[1], &v[3]); rtgmc_retouch_sort2     (&v[4], &v[6]); rtgmc_retouch_sort2     (&v[5], &v[7]);
    rtgmc_retouch_sort2     (&v[0], &v[1]); rtgmc_retouch_sort2     (&v[2], &v[3]); rtgmc_retouch_sort2     (&v[4], &v[5]); rtgmc_retouch_sort2     (&v[6], &v[7]);
}

static inline int rtgmc_retouch_detail_ref_vertical_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    const int pixCenter = read_pix_clamped(src, x, y, pitch, width, height);
    const int pixUpper = (y > 0) ? read_pix_clamped(src, x, y - 1, pitch, width, height) : pixCenter;
    const int pixLower = (y + 1 < height) ? read_pix_clamped(src, x, y + 1, pitch, width, height) : pixCenter;
    const int triadSum = pixUpper + pixCenter + pixLower;
    const int pairLowerMin = min(pixUpper, pixCenter);
    const int pairLowerMax = max(pixUpper, pixCenter);
    const int triadMedian = max(pairLowerMin, min(pairLowerMax, pixLower));
    return (triadSum - triadMedian + 1) >> 1;
}

static inline int rtgmc_retouch_removegrain12_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return read_pix_clamped(src, x, y, pitch, width, height);
    }
    const int p00 = read_pix_clamped(src, x - 1, y - 1, pitch, width, height);
    const int p10 = read_pix_clamped(src, x,     y - 1, pitch, width, height);
    const int p20 = read_pix_clamped(src, x + 1, y - 1, pitch, width, height);
    const int p01 = read_pix_clamped(src, x - 1, y,     pitch, width, height);
    const int p11 = read_pix_clamped(src, x,     y,     pitch, width, height);
    const int p21 = read_pix_clamped(src, x + 1, y,     pitch, width, height);
    const int p02 = read_pix_clamped(src, x - 1, y + 1, pitch, width, height);
    const int p12 = read_pix_clamped(src, x,     y + 1, pitch, width, height);
    const int p22 = read_pix_clamped(src, x + 1, y + 1, pitch, width, height);
    const int sum =
        p00 + 2 * p10 + p20 +
        2 * p01 + 4 * p11 + 2 * p21 +
        p02 + 2 * p12 + p22;
    return (sum + 8) >> 4;
}

static inline int rtgmc_retouch_removegrain_smooth_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    return rtgmc_retouch_removegrain12_value(src, x, y, pitch, width, height);
}

static inline int rtgmc_retouch_verticalcleaner1_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    if (y <= 0 || y >= height - 1) {
        return read_pix_clamped(src, x, y, pitch, width, height);
    }
    const int top = read_pix_clamped(src, x, y - 1, pitch, width, height);
    const int center = read_pix_clamped(src, x, y, pitch, width, height);
    const int bottom = read_pix_clamped(src, x, y + 1, pitch, width, height);
    return rtgmc_retouch_median3(top, center, bottom);
}

static inline int rtgmc_retouch_blur10h_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height
) {
    const int center = read_pix_clamped(src, x, y, pitch, width, height);
    const int left = (x > 0) ? read_pix_clamped(src, x - 1, y, pitch, width, height) : center;
    const int right = (x + 1 < width) ? read_pix_clamped(src, x + 1, y, pitch, width, height) : center;
    return (left + 2 * center + right + 2) >> 2;
}

static inline int rtgmc_retouch_precise_clamp_value(const int src, const int ref) {
    if (src < ref) {
        return min(src + 1, max_val);
    }
    if (src > ref) {
        return max(src - 1, 0);
    }
    return src;
}

static inline int rtgmc_retouch_make_diff_value(const int a, const int b) {
    return clamp(a - b + range_half, 0, max_val);
}

static inline int rtgmc_retouch_add_diff_value(const int src, const int diff) {
    return clamp(src + diff - range_half, 0, max_val);
}

static inline int rtgmc_retouch_round_clamp(const float value) {
    return (int)(clamp(value, 0.0f, (float)max_val) + 0.5f);
}

static inline void rtgmc_retouch_ref_ring_minmax(
    const __global uchar *ref,
    const int x, const int y,
    const int refPitch,
    const int width, const int height,
    __private int *minv,
    __private int *maxv
) {
#pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
#pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            if (dx != 0 || dy != 0) {
                const int sample = read_pix_clamped(ref, x + dx, y + dy, refPitch, width, height);
                *minv = min(*minv, sample);
                *maxv = max(*maxv, sample);
            }
        }
    }
}

static inline void rtgmc_retouch_collect_ref_ring(
    __private int *dst,
    const __global uchar *ref,
    const int x, const int y,
    const int refPitch,
    const int width, const int height
) {
    int count = 0;
#pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
#pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            if (dx != 0 || dy != 0) {
                dst[count++] = read_pix_clamped(ref, x + dx, y + dy, refPitch, width, height);
            }
        }
    }
}

static inline int rtgmc_retouch_repair_mode1_value(
    const __global uchar *src, const __global uchar *ref,
    const int x, const int y,
    const int srcPitch, const int refPitch,
    const int width, const int height
) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return read_pix_clamped(src, x, y, srcPitch, width, height);
    }
    const int s = read_pix_clamped(src, x, y, srcPitch, width, height);
    int minv = s;
    int maxv = s;
    rtgmc_retouch_ref_ring_minmax(ref, x, y, refPitch, width, height, &minv, &maxv);
    return clamp(s, minv, maxv);
}

static inline int rtgmc_retouch_repair_mode12_value(
    const __global uchar *src, const __global uchar *ref,
    const int x, const int y,
    const int srcPitch, const int refPitch,
    const int width, const int height
) {
    const int s = read_pix_clamped(src, x, y, srcPitch, width, height);
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        return s;
    }
    int v[8];
    rtgmc_retouch_collect_ref_ring(v, ref, x, y, refPitch, width, height);
    rtgmc_retouch_sort8(v);
    const int c = read_pix_clamped(ref, x, y, refPitch, width, height);
    const int lo = min(v[1], c);
    const int hi = max(v[6], c);
    return clamp(s, lo, hi);
}

static inline int rtgmc_retouch_removegrain12_diff_value(
    const __global uchar *src, const int srcPitch,
    const __global uchar *base, const int basePitch,
    const int x, const int y, const int width, const int height
) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return rtgmc_retouch_make_diff_value(
            read_pix_clamped(src, x, y, srcPitch, width, height),
            read_pix_clamped(base, x, y, basePitch, width, height));
    }
    const int p00 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x - 1, y - 1, srcPitch, width, height),
        read_pix_clamped(base, x - 1, y - 1, basePitch, width, height));
    const int p10 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x, y - 1, srcPitch, width, height),
        read_pix_clamped(base, x, y - 1, basePitch, width, height));
    const int p20 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x + 1, y - 1, srcPitch, width, height),
        read_pix_clamped(base, x + 1, y - 1, basePitch, width, height));
    const int p01 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x - 1, y, srcPitch, width, height),
        read_pix_clamped(base, x - 1, y, basePitch, width, height));
    const int p11 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x, y, srcPitch, width, height),
        read_pix_clamped(base, x, y, basePitch, width, height));
    const int p21 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x + 1, y, srcPitch, width, height),
        read_pix_clamped(base, x + 1, y, basePitch, width, height));
    const int p02 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x - 1, y + 1, srcPitch, width, height),
        read_pix_clamped(base, x - 1, y + 1, basePitch, width, height));
    const int p12 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x, y + 1, srcPitch, width, height),
        read_pix_clamped(base, x, y + 1, basePitch, width, height));
    const int p22 = rtgmc_retouch_make_diff_value(
        read_pix_clamped(src, x + 1, y + 1, srcPitch, width, height),
        read_pix_clamped(base, x + 1, y + 1, basePitch, width, height));
    const int sum =
        p00 + 2 * p10 + p20 +
        2 * p01 + 4 * p11 + 2 * p21 +
        p02 + 2 * p12 + p22;
    return (sum + 8) >> 4;
}

static inline int rtgmc_retouch_detail_ref_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int precise
) {
    const int detailRef = rtgmc_retouch_detail_ref_vertical_value(src, x, y, pitch, width, height);
    if (precise == 0) {
        return detailRef;
    }
    const int srcPix = read_pix_clamped(src, x, y, pitch, width, height);
    return rtgmc_retouch_precise_clamp_value(detailRef, srcPix);
}

static inline int rtgmc_retouch_detail_ref_blur_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int precise
) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return rtgmc_retouch_detail_ref_value(src, x, y, pitch, width, height, precise);
    }
    const int p00 = rtgmc_retouch_detail_ref_value(src, x - 1, y - 1, pitch, width, height, precise);
    const int p10 = rtgmc_retouch_detail_ref_value(src, x,     y - 1, pitch, width, height, precise);
    const int p20 = rtgmc_retouch_detail_ref_value(src, x + 1, y - 1, pitch, width, height, precise);
    const int p01 = rtgmc_retouch_detail_ref_value(src, x - 1, y,     pitch, width, height, precise);
    const int p11 = rtgmc_retouch_detail_ref_value(src, x,     y,     pitch, width, height, precise);
    const int p21 = rtgmc_retouch_detail_ref_value(src, x + 1, y,     pitch, width, height, precise);
    const int p02 = rtgmc_retouch_detail_ref_value(src, x - 1, y + 1, pitch, width, height, precise);
    const int p12 = rtgmc_retouch_detail_ref_value(src, x,     y + 1, pitch, width, height, precise);
    const int p22 = rtgmc_retouch_detail_ref_value(src, x + 1, y + 1, pitch, width, height, precise);
    const int sum =
        p00 + 2 * p10 + p20 +
        2 * p01 + 4 * p11 + 2 * p21 +
        p02 + 2 * p12 + p22;
    return (sum + 8) >> 4;
}

static inline int rtgmc_retouch_stronger_non_neutral(const int candidate, const int baseline) {
    const int candidateOffset = candidate - range_half;
    const int baselineOffset = baseline - range_half;
    return (abs(candidateOffset) > abs(baselineOffset)) ? candidate : range_half;
}

static inline int rtgmc_retouch_vertical_balance_delta_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height,
    const float edgeNarrowingGain
) {
    const int srcPix = read_pix_clamped(src, x, y, pitch, width, height);
    const int cleaned = rtgmc_retouch_verticalcleaner1_value(src, x, y, pitch, width, height);
    const float value = fma((float)(cleaned - srcPix), edgeNarrowingGain, (float)range_half);
    return rtgmc_retouch_round_clamp(value);
}

static inline int rtgmc_retouch_horizontal_balance_delta_value(
    const __global uchar *src, int x, int y,
    const int pitch, const int width, const int height,
    const float edgeNarrowingGain
) {
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    const int center = rtgmc_retouch_vertical_balance_delta_value(src, x, y, pitch, width, height, edgeNarrowingGain);
    const int left = (x > 0)
        ? rtgmc_retouch_vertical_balance_delta_value(src, x - 1, y, pitch, width, height, edgeNarrowingGain)
        : center;
    const int right = (x + 1 < width)
        ? rtgmc_retouch_vertical_balance_delta_value(src, x + 1, y, pitch, width, height, edgeNarrowingGain)
        : center;
    return (left + 2 * center + right + 2) >> 2;
}

static inline int rtgmc_retouch_area_balance_delta_value(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height,
    const float edgeNarrowingGain
) {
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return rtgmc_retouch_horizontal_balance_delta_value(src, x, y, pitch, width, height, edgeNarrowingGain);
    }
    const int p00 = rtgmc_retouch_horizontal_balance_delta_value(src, x - 1, y - 1, pitch, width, height, edgeNarrowingGain);
    const int p10 = rtgmc_retouch_horizontal_balance_delta_value(src, x,     y - 1, pitch, width, height, edgeNarrowingGain);
    const int p20 = rtgmc_retouch_horizontal_balance_delta_value(src, x + 1, y - 1, pitch, width, height, edgeNarrowingGain);
    const int p01 = rtgmc_retouch_horizontal_balance_delta_value(src, x - 1, y,     pitch, width, height, edgeNarrowingGain);
    const int p11 = rtgmc_retouch_horizontal_balance_delta_value(src, x,     y,     pitch, width, height, edgeNarrowingGain);
    const int p21 = rtgmc_retouch_horizontal_balance_delta_value(src, x + 1, y,     pitch, width, height, edgeNarrowingGain);
    const int p02 = rtgmc_retouch_horizontal_balance_delta_value(src, x - 1, y + 1, pitch, width, height, edgeNarrowingGain);
    const int p12 = rtgmc_retouch_horizontal_balance_delta_value(src, x,     y + 1, pitch, width, height, edgeNarrowingGain);
    const int p22 = rtgmc_retouch_horizontal_balance_delta_value(src, x + 1, y + 1, pitch, width, height, edgeNarrowingGain);
    const int sum =
        p00 + 2 * p10 + p20 +
        2 * p01 + 4 * p11 + 2 * p21 +
        p02 + 2 * p12 + p22;
    return (sum + 8) >> 4;
}

static inline int rtgmc_retouch_temporal_detail_guard_value(
    const int srcPix,
    const __global uchar *ref,
    const __global uchar *motionBack,
    const __global uchar *motionForw,
    const int x, const int y,
    const int refPitch, const int motionBackPitch, const int motionForwPitch,
    const int width, const int height,
    const int sovs
) {
    const int refPix = read_pix_clamped(ref, x, y, refPitch, width, height);
    const int motionBackPix = read_pix_clamped(motionBack, x, y, motionBackPitch, width, height);
    const int motionForwPix = read_pix_clamped(motionForw, x, y, motionForwPitch, width, height);
    const int lower = min(refPix, min(motionBackPix, motionForwPix)) - sovs;
    const int upper = max(refPix, max(motionBackPix, motionForwPix)) + sovs;
    return clamp(srcPix, max(0, lower), min(max_val, upper));
}

static inline int rtgmc_retouch_temporal_detail_guard_value_inline_comp(
    const int srcPix,
    const __global uchar *ref,
    const int x, const int y,
    const int refPitch,
    const int width, const int height,
    const __global uchar *compCur,
    const int compCurPitch,
    const __global uchar *compRefBack,
    const __global uchar *compRefForw,
    const int compRefDirBack,
    const int compRefDirForw,
    const __global rtgmc_degrain_mv_t *compMv,
    const __global rtgmc_degrain_sad_t *compSad,
    const int compBlocksX,
    const int compBlocksY,
    const int compBlockSize,
    const int compOverlap,
    const int compStep,
    const int compCoveredWidth,
    const int compCoveredHeight,
    const int compPlaneScaleX,
    const int compPlaneScaleY,
    const uint compThsad,
    const uint compDisableMask,
    const __global float *compWindowRamp,
    const int compWidth,
    const int compHeight,
    const int compRefs,
    const int compPel,
    const int compSubpelInterp,
    const int sovs
) {
    const int refPix = read_pix_clamped(ref, x, y, refPitch, width, height);
    const int motionBackPix = rtgmc_degrain_compensate_overlap_pixel_value(
        compCur, compCurPitch,
        compCur, compRefBack,
        compRefDirBack,
        compWidth, compHeight,
        compMv, compSad,
        compBlocksX, compBlocksY,
        compBlockSize, compOverlap, compStep,
        compCoveredWidth, compCoveredHeight,
        compPlaneScaleX, compPlaneScaleY,
        compThsad, compDisableMask,
        compWindowRamp,
        x, y,
        compRefs, compPel, compSubpelInterp);
    const int motionForwPix = rtgmc_degrain_compensate_overlap_pixel_value(
        compCur, compCurPitch,
        compCur, compRefForw,
        compRefDirForw,
        compWidth, compHeight,
        compMv, compSad,
        compBlocksX, compBlocksY,
        compBlockSize, compOverlap, compStep,
        compCoveredWidth, compCoveredHeight,
        compPlaneScaleX, compPlaneScaleY,
        compThsad, compDisableMask,
        compWindowRamp,
        x, y,
        compRefs, compPel, compSubpelInterp);
    const int lower = min(refPix, min(motionBackPix, motionForwPix)) - sovs;
    const int upper = max(refPix, max(motionBackPix, motionForwPix)) + sovs;
    return clamp(srcPix, max(0, lower), min(max_val, upper));
}

static inline int rtgmc_retouch_spatial_min(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int radius
) {
    int value = max_val;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            value = min(value, read_pix_clamped(src, x + dx, y + dy, pitch, width, height));
        }
    }
    return value;
}

static inline int rtgmc_retouch_spatial_max(
    const __global uchar *src, const int x, const int y,
    const int pitch, const int width, const int height,
    const int radius
) {
    int value = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            value = max(value, read_pix_clamped(src, x + dx, y + dy, pitch, width, height));
        }
    }
    return value;
}

__kernel void kernel_rtgmc_retouch_copy(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height));
}

__kernel void kernel_rtgmc_retouch_repair1(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict ref, const int refPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        rtgmc_retouch_repair_mode1_value((const __global uchar *)src, (const __global uchar *)ref, ix, iy, srcPitch, refPitch, width, height));
}

__kernel void kernel_rtgmc_retouch_repair12(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict ref, const int refPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        rtgmc_retouch_repair_mode12_value((const __global uchar *)src, (const __global uchar *)ref, ix, iy, srcPitch, refPitch, width, height));
}

__kernel void kernel_rtgmc_retouch_removegrain12(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        rtgmc_retouch_removegrain12_value((const __global uchar *)src, ix, iy, srcPitch, width, height));
}

__kernel void kernel_rtgmc_retouch_removegrain11(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        rtgmc_retouch_removegrain12_value((const __global uchar *)src, ix, iy, srcPitch, width, height));
}

__kernel void kernel_rtgmc_retouch_detail_ref_vertical(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        rtgmc_retouch_detail_ref_vertical_value((const __global uchar *)src, ix, iy, srcPitch, width, height));
}

__kernel void kernel_rtgmc_retouch_precise_clamp(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict ref, const int refPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int refPix = read_pix_clamped((const __global uchar *)ref, ix, iy, refPitch, width, height);
    write_pix((__global uchar *)dst, ix, iy, dstPitch, rtgmc_retouch_precise_clamp_value(srcPix, refPix));
}

__kernel void kernel_rtgmc_retouch_detail_boost(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict blur, const int blurPitch,
    const int width, const int height,
    const float detailGain
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int blurPix = read_pix_clamped((const __global uchar *)blur, ix, iy, blurPitch, width, height);
    const float value = (float)srcPix + (float)(srcPix - blurPix) * detailGain;
    write_pix((__global uchar *)dst, ix, iy, dstPitch, rtgmc_retouch_round_clamp(value));
}

__kernel void kernel_rtgmc_retouch_detail_boost_fused(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height,
    const int smode,
    const int precise,
    const float detailGain
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int blurPix = (smode == 2)
        ? rtgmc_retouch_detail_ref_blur_value((const __global uchar *)src, ix, iy, srcPitch, width, height, precise)
        : rtgmc_retouch_removegrain12_value((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const float value = (float)srcPix + (float)(srcPix - blurPix) * detailGain;
    write_pix((__global uchar *)dst, ix, iy, dstPitch, rtgmc_retouch_round_clamp(value));
}

__kernel void kernel_rtgmc_retouch_detail_boost_edge_narrow_fused(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height,
    const int smode,
    const int precise,
    const float detailGain,
    const float edgeNarrowingGain
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int blurPix = (smode == 2)
        ? rtgmc_retouch_detail_ref_blur_value((const __global uchar *)src, ix, iy, srcPitch, width, height, precise)
        : rtgmc_retouch_removegrain12_value((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const float boosted = (float)srcPix + (float)(srcPix - blurPix) * detailGain;
    const int boostedPix = rtgmc_retouch_round_clamp(boosted);
    const int centerDiff = rtgmc_retouch_horizontal_balance_delta_value((const __global uchar *)src, ix, iy, srcPitch, width, height, edgeNarrowingGain);
    const int smoothDiff = rtgmc_retouch_area_balance_delta_value((const __global uchar *)src, ix, iy, srcPitch, width, height, edgeNarrowingGain);
    const int correction = rtgmc_retouch_stronger_non_neutral(smoothDiff, centerDiff);
    write_pix((__global uchar *)dst, ix, iy, dstPitch, rtgmc_retouch_add_diff_value(boostedPix, correction));
}

__kernel void kernel_rtgmc_retouch_edge_narrow_delta(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height,
    const float edgeNarrowingGain
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        rtgmc_retouch_vertical_balance_delta_value((const __global uchar *)src, ix, iy, srcPitch, width, height, edgeNarrowingGain));
}

__kernel void kernel_rtgmc_retouch_blur_h(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        rtgmc_retouch_blur10h_value((const __global uchar *)src, ix, iy, srcPitch, width, height));
}

__kernel void kernel_rtgmc_retouch_edge_narrow_guard_delta(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int rgPix = rtgmc_retouch_removegrain_smooth_value((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int value = rtgmc_retouch_stronger_non_neutral(rgPix, srcPix);
    write_pix((__global uchar *)dst, ix, iy, dstPitch, value);
}

__kernel void kernel_rtgmc_retouch_edge_narrow_guard_delta11(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int rgPix = rtgmc_retouch_removegrain_smooth_value((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int value = rtgmc_retouch_stronger_non_neutral(rgPix, srcPix);
    write_pix((__global uchar *)dst, ix, iy, dstPitch, value);
}

__kernel void kernel_rtgmc_retouch_adddiff(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict diff, const int diffPitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int diffPix = read_pix_clamped((const __global uchar *)diff, ix, iy, diffPitch, width, height);
    write_pix((__global uchar *)dst, ix, iy, dstPitch, rtgmc_retouch_add_diff_value(srcPix, diffPix));
}

__kernel void kernel_rtgmc_retouch_edge_narrow_fused(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict base, const int basePitch,
    const int width, const int height,
    const float edgeNarrowingGain
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int centerDiff = rtgmc_retouch_horizontal_balance_delta_value((const __global uchar *)base, ix, iy, basePitch, width, height, edgeNarrowingGain);
    const int smoothDiff = rtgmc_retouch_area_balance_delta_value((const __global uchar *)base, ix, iy, basePitch, width, height, edgeNarrowingGain);
    const int correction = rtgmc_retouch_stronger_non_neutral(smoothDiff, centerDiff);
    write_pix((__global uchar *)dst, ix, iy, dstPitch, rtgmc_retouch_add_diff_value(srcPix, correction));
}

__kernel void kernel_rtgmc_retouch_make_delta(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict base, const int basePitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int srcPix = read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int basePix = read_pix_clamped((const __global uchar *)base, ix, iy, basePitch, width, height);
    write_pix((__global uchar *)dst, ix, iy, dstPitch, rtgmc_retouch_make_diff_value(srcPix, basePix));
}

__kernel void kernel_rtgmc_retouch_smooth_delta_fused(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict base, const int basePitch,
    const int width, const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    write_pix((__global uchar *)dst, ix, iy, dstPitch,
        rtgmc_retouch_removegrain12_diff_value((const __global uchar *)src, srcPitch,
            (const __global uchar *)base, basePitch, ix, iy, width, height));
}

__kernel void kernel_rtgmc_retouch_limit(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict base, const int basePitch,
    const __global Type *restrict ref, const int refPitch,
    const __global Type *restrict motionBack, const int motionBackPitch,
    const __global Type *restrict motionForw, const int motionForwPitch,
    const int width, const int height,
    const int slmode,
    const int slrad,
    const int sovs,
    const float limit_strength,
    const int use_temporal_limit
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    float value = (float)read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    if ((slmode == 2 || slmode == 4) && use_temporal_limit != 0) {
        value = (float)rtgmc_retouch_temporal_detail_guard_value(
            rtgmc_retouch_round_clamp(value),
            (const __global uchar *)ref,
            (const __global uchar *)motionBack,
            (const __global uchar *)motionForw,
            ix, iy,
            refPitch, motionBackPitch, motionForwPitch,
            width, height,
            sovs);
    } else if (slmode == 1 || slmode == 2 || slmode == 4 || limit_strength > 0.0f) {
        const int radius = clamp(slrad, 1, 3);
        const float localMin = (float)max(0, rtgmc_retouch_spatial_min((const __global uchar *)base, ix, iy, basePitch, width, height, radius) - sovs);
        const float localMax = (float)min(max_val, rtgmc_retouch_spatial_max((const __global uchar *)base, ix, iy, basePitch, width, height, radius) + sovs);
        const float limited = clamp(value, localMin, localMax);
        const float strength = (slmode == 1 || slmode == 2) ? 1.0f : clamp(limit_strength, 0.0f, 1.0f);
        value = value + (limited - value) * strength;
    }
    write_pix((__global uchar *)dst, ix, iy, dstPitch, rtgmc_retouch_round_clamp(value));
}

__kernel void kernel_rtgmc_retouch_limit_inline_comp(
    __global Type *restrict dst, const int dstPitch,
    const __global Type *restrict src, const int srcPitch,
    const __global Type *restrict ref, const int refPitch,
    const __global uchar *compCur,
    const int compCurPitch,
    const __global uchar *compRefBack,
    const __global uchar *compRefForw,
    const int compRefDirBack,
    const int compRefDirForw,
    const __global rtgmc_degrain_mv_t *compMv,
    const __global rtgmc_degrain_sad_t *compSad,
    const int compBlocksX,
    const int compBlocksY,
    const int compBlockSize,
    const int compOverlap,
    const int compStep,
    const int compCoveredWidth,
    const int compCoveredHeight,
    const int compPlaneScaleX,
    const int compPlaneScaleY,
    const uint compThsad,
    const uint compDisableMask,
    const __global float *compWindowRamp,
    const int compWidth,
    const int compHeight,
    const int compRefs,
    const int compPel,
    const int compSubpelInterp,
    const int width, const int height,
    const int sovs
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const float value = (float)read_pix_clamped((const __global uchar *)src, ix, iy, srcPitch, width, height);
    const int result = rtgmc_retouch_temporal_detail_guard_value_inline_comp(
        rtgmc_retouch_round_clamp(value),
        (const __global uchar *)ref,
        ix, iy,
        refPitch,
        width, height,
        compCur, compCurPitch,
        compRefBack, compRefForw,
        compRefDirBack, compRefDirForw,
        compMv, compSad,
        compBlocksX, compBlocksY,
        compBlockSize, compOverlap, compStep,
        compCoveredWidth, compCoveredHeight,
        compPlaneScaleX, compPlaneScaleY,
        compThsad, compDisableMask,
        compWindowRamp,
        compWidth, compHeight,
        compRefs, compPel, compSubpelInterp,
        sovs);
    write_pix((__global uchar *)dst, ix, iy, dstPitch, result);
}
