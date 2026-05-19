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

#ifndef max3
#define max3(a, b, c) (max(max((a), (b)), (c)))
#endif

#ifndef min3
#define min3(a, b, c) (min(min((a), (b)), (c)))
#endif

static inline int read_pix(
    const __global uchar *src, int x, int y,
    const int pitch, const int width, const int height
) {
    x = clamp(x, 0, width  - 1);
    y = clamp(y, 0, height - 1);
    return (int)(*(const __global Type *)(src + y * pitch + x * sizeof(Type)));
}

static inline int rtgmc_edi_yadif_spatial(
    const __global uchar *src, const int ix, const int iy,
    const int pitch, const int width, const int height
) {
    int ym1[7], yp1[7];
    #pragma unroll
    for (int dx = -3; dx <= 3; dx++) {
        ym1[dx + 3] = read_pix(src, ix + dx, iy - 1, pitch, width, height);
        yp1[dx + 3] = read_pix(src, ix + dx, iy + 1, pitch, width, height);
    }

    const int score[5] = {
        abs(ym1[2] - yp1[2]) + abs(ym1[3] - yp1[3]) + abs(ym1[4] - yp1[4]) - 1,
        abs(ym1[1] - yp1[3]) + abs(ym1[2] - yp1[4]) + abs(ym1[3] - yp1[5]),
        abs(ym1[0] - yp1[4]) + abs(ym1[1] - yp1[5]) + abs(ym1[2] - yp1[6]),
        abs(ym1[3] - yp1[1]) + abs(ym1[4] - yp1[2]) + abs(ym1[5] - yp1[3]),
        abs(ym1[4] - yp1[0]) + abs(ym1[5] - yp1[1]) + abs(ym1[6] - yp1[2])
    };
    int minscore = score[0];
    int minidx = 0;
    if (score[1] < minscore) {
        minscore = score[1];
        minidx = 1;
    }
    if (score[2] < minscore) {
        minscore = score[2];
        minidx = 2;
    }
    if (score[3] < minscore) {
        minscore = score[3];
        minidx = 3;
    }
    if (score[4] < minscore) {
        minscore = score[4];
        minidx = 4;
    }

    switch (minidx) {
    case 0: return (ym1[3] + yp1[3]) >> 1;
    case 1: return (ym1[2] + yp1[4]) >> 1;
    case 2: return (ym1[1] + yp1[5]) >> 1;
    case 3: return (ym1[4] + yp1[2]) >> 1;
    case 4:
    default:return (ym1[5] + yp1[1]) >> 1;
    }
}

static inline int rtgmc_edi_tdeint_spatial(
    const __global uchar *src, const int ix, const int iy,
    const int pitch, const int width, const int height
) {
    const int up2 = read_pix(src, ix, iy - 3, pitch, width, height);
    const int up1 = read_pix(src, ix, iy - 1, pitch, width, height);
    const int dn1 = read_pix(src, ix, iy + 1, pitch, width, height);
    const int dn2 = read_pix(src, ix, iy + 3, pitch, width, height);
    const int cubic = (-up2 + 9 * up1 + 9 * dn1 - dn2 + 8) >> 4;
    return clamp(cubic, min(up1, dn1), max(up1, dn1));
}

static inline int rtgmc_edi_yadif_temporal(
    const __global uchar *prevSrc,
    const __global uchar *curSrc,
    const __global uchar *nextSrc,
    const int ix, const int iy,
    const int prevPitch, const int curPitch, const int nextPitch,
    const int width, const int height,
    const int valSpatial,
    const int fieldSecond
) {
    const int t00m1 = read_pix(prevSrc, ix, iy - 1, prevPitch, width, height);
    const int t00p1 = read_pix(prevSrc, ix, iy + 1, prevPitch, width, height);
    const int t10m1 = read_pix(curSrc,  ix, iy - 1, curPitch,  width, height);
    const int t10p1 = read_pix(curSrc,  ix, iy + 1, curPitch,  width, height);
    const int t20m1 = read_pix(nextSrc, ix, iy - 1, nextPitch, width, height);
    const int t20p1 = read_pix(nextSrc, ix, iy + 1, nextPitch, width, height);

    const __global uchar *src01 = fieldSecond ? curSrc : prevSrc;
    const __global uchar *src12 = fieldSecond ? nextSrc : curSrc;
    const int src01Pitch = fieldSecond ? curPitch : prevPitch;
    const int src12Pitch = fieldSecond ? nextPitch : curPitch;

    const int t01m2 = read_pix(src01, ix, iy - 2, src01Pitch, width, height);
    const int t01_0 = read_pix(src01, ix, iy + 0, src01Pitch, width, height);
    const int t01p2 = read_pix(src01, ix, iy + 2, src01Pitch, width, height);
    const int t12m2 = read_pix(src12, ix, iy - 2, src12Pitch, width, height);
    const int t12_0 = read_pix(src12, ix, iy + 0, src12Pitch, width, height);
    const int t12p2 = read_pix(src12, ix, iy + 2, src12Pitch, width, height);

    const int tm2 = (t01m2 + t12m2) >> 1;
    const int t_0 = (t01_0 + t12_0) >> 1;
    const int tp2 = (t01p2 + t12p2) >> 1;

    const int diff = max3(
        abs(t01_0 - t12_0) >> 1,
        (abs(t00m1 - t10m1) + abs(t00p1 - t10p1)) >> 1,
        (abs(t20m1 - t10m1) + abs(t10p1 - t20p1)) >> 1);
    return max(min(valSpatial, t_0 + diff), t_0 - diff);
}

static inline int rtgmc_edi_yadif_edge(
    const __global uchar *src, const int ix, const int iy,
    const int pitch, const int width, const int height,
    const int targetField
) {
    if (targetField == 0) {
        if (iy == 0) {
            return read_pix(src, ix, 1, pitch, width, height);
        }
        if (iy == height - 2) {
            return (read_pix(src, ix, height - 3, pitch, width, height)
                + read_pix(src, ix, height - 1, pitch, width, height) + 1) >> 1;
        }
    } else {
        if (iy == 1) {
            return (read_pix(src, ix, 0, pitch, width, height)
                + read_pix(src, ix, 2, pitch, width, height) + 1) >> 1;
        }
        if (iy == height - 1) {
            return read_pix(src, ix, height - 2, pitch, width, height);
        }
    }
    return -1;
}

static inline int rtgmc_edi_repair_mode2(
    const int edi,
    const __global uchar *src, const int ix, const int iy,
    const int pitch, const int width, const int height
) {
    if (ix <= 0 || iy <= 0 || ix >= width - 1 || iy >= height - 1) {
        return edi;
    }

    int a[9] = {
        read_pix(src, ix - 1, iy - 1, pitch, width, height),
        read_pix(src, ix + 0, iy - 1, pitch, width, height),
        read_pix(src, ix + 1, iy - 1, pitch, width, height),
        read_pix(src, ix - 1, iy + 0, pitch, width, height),
        read_pix(src, ix + 0, iy + 0, pitch, width, height),
        read_pix(src, ix + 1, iy + 0, pitch, width, height),
        read_pix(src, ix - 1, iy + 1, pitch, width, height),
        read_pix(src, ix + 0, iy + 1, pitch, width, height),
        read_pix(src, ix + 1, iy + 1, pitch, width, height)
    };
    for (int i = 1; i < 9; i++) {
        int value = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > value) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = value;
    }

    return clamp(edi, a[1], a[7]);
}

static inline int rtgmc_edi_luma(
    const __global uchar *bobSrc,
    const __global uchar *ediPrevSrc,
    const __global uchar *ediSrc,
    const __global uchar *ediNextSrc,
    const int ix, const int iy,
    const int bobPitch, const int ediPrevPitch, const int ediPitch, const int ediNextPitch,
    const int width, const int height,
    const int targetField,
    const int fieldSecond,
    const int mode
) {
    const int bobValue = read_pix(bobSrc, ix, iy, bobPitch, width, height);
    const int ediValue = read_pix(ediSrc, ix, iy, ediPitch, width, height);
    if (mode == 6) {
        return ediValue;
    }
    if ((iy & 1) != targetField) {
        return ediValue;
    }
    if (mode == 0) {
        return bobValue;
    }
    if (mode == 1 || mode == 4) {
        const int edge = rtgmc_edi_yadif_edge(ediSrc, ix, iy, ediPitch, width, height, targetField);
        if (edge >= 0) {
            return edge;
        }
        const int spatial = rtgmc_edi_yadif_spatial(ediSrc, ix, iy, ediPitch, width, height);
        return rtgmc_edi_yadif_temporal(ediPrevSrc, ediSrc, ediNextSrc, ix, iy,
            ediPrevPitch, ediPitch, ediNextPitch, width, height, spatial, fieldSecond);
    }
    if (mode == 2) {
        const int edge = rtgmc_edi_yadif_edge(ediSrc, ix, iy, ediPitch, width, height, targetField);
        if (edge >= 0) {
            return edge;
        }
        const int spatial = rtgmc_edi_yadif_spatial(ediSrc, ix, iy, ediPitch, width, height);
        return rtgmc_edi_yadif_temporal(ediPrevSrc, ediSrc, ediNextSrc, ix, iy,
            ediPrevPitch, ediPitch, ediNextPitch, width, height, spatial, fieldSecond);
    }
    if (mode == 3) {
        return rtgmc_edi_tdeint_spatial(ediSrc, ix, iy, ediPitch, width, height);
    }
    if (mode == 5) {
        const int edge = rtgmc_edi_yadif_edge(ediSrc, ix, iy, ediPitch, width, height, targetField);
        const int yadifValue = (edge >= 0)
            ? edge
            : rtgmc_edi_yadif_temporal(ediPrevSrc, ediSrc, ediNextSrc, ix, iy,
                ediPrevPitch, ediPitch, ediNextPitch, width, height,
                rtgmc_edi_yadif_spatial(ediSrc, ix, iy, ediPitch, width, height), fieldSecond);
        return rtgmc_edi_repair_mode2(yadifValue, bobSrc, ix, iy, bobPitch, width, height);
    }
    return bobValue;
}

static inline void rtgmc_edi_store(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pBobSrc, const int bobPitch,
    const __global Type *restrict pEdiPrevSrc, const int ediPrevPitch,
    const __global Type *restrict pEdiSrc, const int ediPitch,
    const __global Type *restrict pEdiNextSrc, const int ediNextPitch,
    const int width,
    const int height,
    const int planeIndex,
    const int targetField,
    const int fieldSecond,
    const int mode
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const int value = (mode == 6)
        ? read_pix((const __global uchar *)pEdiSrc, ix, iy, ediPitch, width, height)
        : rtgmc_edi_luma(
            (const __global uchar *)pBobSrc,
            (const __global uchar *)pEdiPrevSrc,
            (const __global uchar *)pEdiSrc,
            (const __global uchar *)pEdiNextSrc,
            ix, iy, bobPitch, ediPrevPitch, ediPitch, ediNextPitch, width, height, targetField, fieldSecond, mode);
    __global Type *dstPix = (__global Type *)((__global uchar *)pDst + iy * dstPitch + ix * sizeof(Type));
    dstPix[0] = (Type)clamp(value, 0, max_val);
}

#define RTGMC_EDI_KERNEL(name, mode_value) \
__attribute__((reqd_work_group_size(rtgmc_edi_block_x, rtgmc_edi_block_y, 1))) \
__kernel void name( \
    __global Type *restrict pDst, const int dstPitch, \
    const __global Type *restrict pBobSrc, const int bobPitch, \
    const __global Type *restrict pEdiPrevSrc, const int ediPrevPitch, \
    const __global Type *restrict pEdiSrc, const int ediPitch, \
    const __global Type *restrict pEdiNextSrc, const int ediNextPitch, \
    const int width, \
    const int height, \
    const int planeIndex, \
    const int targetField, \
    const int fieldSecond \
) { \
    rtgmc_edi_store(pDst, dstPitch, pBobSrc, bobPitch, pEdiPrevSrc, ediPrevPitch, pEdiSrc, ediPitch, pEdiNextSrc, ediNextPitch, width, height, planeIndex, targetField, fieldSecond, mode_value); \
}

RTGMC_EDI_KERNEL(kernel_rtgmc_edi_bob, 0)
RTGMC_EDI_KERNEL(kernel_rtgmc_edi_yadif, 1)
RTGMC_EDI_KERNEL(kernel_rtgmc_edi_cyadif, 2)
RTGMC_EDI_KERNEL(kernel_rtgmc_edi_tdeint, 3)
RTGMC_EDI_KERNEL(kernel_rtgmc_edi_rep_yadif, 4)
RTGMC_EDI_KERNEL(kernel_rtgmc_edi_rep_cyadif, 5)
RTGMC_EDI_KERNEL(kernel_rtgmc_edi_passthrough, 6)

