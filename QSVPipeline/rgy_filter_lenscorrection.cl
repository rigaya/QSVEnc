// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
// Copyright (c) 2014-2016 rigaya
// (see rgy_filter_lenscorrection.cpp for full licence text)
// -----------------------------------------------------------------------------------------
//
// Build-time defines:
//   Type       : uchar (8bit) / ushort (>8bit)
//
// Geometric lens correction by the standard radial polynomial (Brown-Conrady) model:
//   for each OUTPUT pixel, the source coordinate is
//       rn    = |p - centre| / (0.5 * hypot(W,H))
//       scale = 1 + k1*rn^2 + k2*rn^4
//       src   = centre + (p - centre) * scale
//   then the input is sampled there with bilinear interpolation (black outside the frame).
// This is a pure backward map + gather, one output pixel per work-item.

#ifndef Type
#define Type uchar
#endif

#define SAMPLE(xx, yy) \
    (((xx) >= 0 && (xx) < srcWidth && (yy) >= 0 && (yy) < srcHeight) \
        ? (float)(*(__global Type *)(pSrc + (yy) * srcPitch + (xx) * sizeof(Type))) \
        : fillValue)

// fillValue is the out-of-frame border value for this plane (0 for luma, neutral for chroma).
__kernel void kernel_lenscorrection(
    __global uchar *restrict pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    __global uchar *restrict pSrc, const int srcPitch, const int srcWidth, const int srcHeight,
    const float k1, const float k2, const float cx, const float cy, const float fillValue) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= dstWidth || iy >= dstHeight) {
        return;
    }
    const float dx = (float)ix - cx * (float)dstWidth;
    const float dy = (float)iy - cy * (float)dstHeight;
    const float r0 = 0.5f * sqrt((float)dstWidth * (float)dstWidth + (float)dstHeight * (float)dstHeight);
    const float rn = sqrt(dx * dx + dy * dy) / r0;
    const float scale = 1.0f + k1 * rn * rn + k2 * rn * rn * rn * rn;
    const float sx = cx * (float)srcWidth  + dx * scale;
    const float sy = cy * (float)srcHeight + dy * scale;

    const int x0 = (int)floor(sx);
    const int y0 = (int)floor(sy);
    const float fx = sx - (float)x0;
    const float fy = sy - (float)y0;
    const float v00 = SAMPLE(x0,     y0);
    const float v10 = SAMPLE(x0 + 1, y0);
    const float v01 = SAMPLE(x0,     y0 + 1);
    const float v11 = SAMPLE(x0 + 1, y0 + 1);
    const float v = v00 * (1.0f - fx) * (1.0f - fy) + v10 * fx * (1.0f - fy)
                  + v01 * (1.0f - fx) * fy          + v11 * fx * fy;

    __global Type *ptrDst = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    ptrDst[0] = (Type)(v + 0.5f);
}
