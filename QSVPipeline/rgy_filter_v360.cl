// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
// Copyright (c) 2014-2016 rigaya
// (full licence text in rgy_filter_v360.cpp)
// -----------------------------------------------------------------------------------------
//
// Projection conversion (v360): for each OUTPUT pixel, map its position through the OUTPUT
// projection to a 3D ray, rotate by the view matrix, then map that ray through the INPUT
// projection to a source coordinate, and bilinearly sample the input there.
// This is the standard cartography ray core (equirectangular / rectilinear / cubemap), an
// independent implementation from that published maths.
//
// Build-time defines:
//   Type     : uchar / ushort
//   IN_PROJ  : 0 equirect, 1 flat(rectilinear), 2 cubemap 3x2
//   OUT_PROJ : 0 equirect, 1 flat(rectilinear), 2 cubemap 3x2

#ifndef Type
#define Type uchar
#endif
#define PROJ_EQUIRECT 0
#define PROJ_FLAT     1
#define PROJ_CUBE     2
#define PI_F 3.14159265358979323846f

// cubemap 3x2 layout, faces order r l u d f b ; row/col of each face
// r=(0,0) l=(0,1) u=(0,2) d=(1,0) f=(1,1) b=(1,2)

// output pixel -> ray (in the output projection's frame). sets *valid.
static float3 out_to_ray(const float fx, const float fy, const int W, const int H, const float hfov, int *valid) {
    *valid = 1;
#if OUT_PROJ == PROJ_FLAT
    const float f = (W * 0.5f) / tan(hfov * 0.5f);
    return (float3)(fx - W * 0.5f + 0.5f, fy - H * 0.5f + 0.5f, f);
#elif OUT_PROJ == PROJ_EQUIRECT
    const float lon = ((fx + 0.5f) / W - 0.5f) * 2.0f * PI_F;
    const float lat = ((fy + 0.5f) / H - 0.5f) * PI_F;
    return (float3)(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
#elif OUT_PROJ == PROJ_CUBE
    const float cw = W / 3.0f, ch = H / 2.0f;
    const int col = (int)(fx / cw), row = (int)(fy / ch);
    const float a = (fx - col * cw) / cw * 2.0f - 1.0f;
    const float b = (fy - row * ch) / ch * 2.0f - 1.0f;
    const int face = row * 3 + col; // 0 r,1 l,2 u,3 d,4 f,5 b
    if (face == 0) return (float3)( 1.0f,  b, -a);
    if (face == 1) return (float3)(-1.0f,  b,  a);
    if (face == 2) return (float3)( a,  1.0f, -b);
    if (face == 3) return (float3)( a, -1.0f,  b);
    if (face == 4) return (float3)( a,  b,  1.0f);
    return                (float3)(-a,  b, -1.0f);
#endif
}

// world ray -> input coordinate (pixel index). sets *valid.
static float2 ray_to_in(const float3 d, const int W, const int H, const float hfov, int *valid) {
    *valid = 1;
#if IN_PROJ == PROJ_EQUIRECT
    const float lon = atan2(d.x, d.z);
    const float lat = atan2(d.y, sqrt(d.x * d.x + d.z * d.z));
    float u = (lon / (2.0f * PI_F) + 0.5f) * W;
    float v = (lat / PI_F + 0.5f) * H;
    u = fmod(u + W, (float)W);                 // wrap longitude
    v = clamp(v, 0.0f, H - 1.0f);              // clamp latitude
    return (float2)(u, v);
#elif IN_PROJ == PROJ_FLAT
    if (d.z <= 0.0f) { *valid = 0; return (float2)(0.0f, 0.0f); }
    const float f = (W * 0.5f) / tan(hfov * 0.5f);
    const float u = d.x / d.z * f + W * 0.5f;
    const float v = d.y / d.z * f + H * 0.5f;
    if (u < 0.0f || u >= W || v < 0.0f || v >= H) *valid = 0;
    return (float2)(u, v);
#elif IN_PROJ == PROJ_CUBE
    const float ax = fabs(d.x), ay = fabs(d.y), az = fabs(d.z);
    const float cw = W / 3.0f, ch = H / 2.0f;
    float a, b, dom; int row, col;
    if (ax >= ay && ax >= az) {
        if (d.x > 0.0f) { a = -d.z; b = d.y; dom = ax; row = 0; col = 0; } // r
        else            { a =  d.z; b = d.y; dom = ax; row = 0; col = 1; } // l
    } else if (ay >= ax && ay >= az) {
        if (d.y > 0.0f) { a = d.x; b = -d.z; dom = ay; row = 0; col = 2; } // u
        else            { a = d.x; b =  d.z; dom = ay; row = 1; col = 0; } // d
    } else {
        if (d.z > 0.0f) { a =  d.x; b = d.y; dom = az; row = 1; col = 1; } // f
        else            { a = -d.x; b = d.y; dom = az; row = 1; col = 2; } // b
    }
    const float uc = a / dom, vc = b / dom;
    return (float2)((col + (uc + 1.0f) * 0.5f) * cw, (row + (vc + 1.0f) * 0.5f) * ch);
#endif
}

#define SAMPLE(xx, yy) \
    (((xx) >= 0 && (xx) < srcWidth && (yy) >= 0 && (yy) < srcHeight) \
        ? (float)(*(__global Type *)(pSrc + (yy) * srcPitch + (xx) * sizeof(Type))) : fillValue)

__kernel void kernel_v360(
    __global uchar *restrict pDst, const int dstPitch, const int dstWidth, const int dstHeight,
    __global uchar *restrict pSrc, const int srcPitch, const int srcWidth, const int srcHeight,
    const float m00, const float m01, const float m02,
    const float m10, const float m11, const float m12,
    const float m20, const float m21, const float m22,
    const float out_hfov, const float in_hfov, const float fillValue) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= dstWidth || iy >= dstHeight) {
        return;
    }
    int valid;
    float3 d = out_to_ray((float)ix, (float)iy, dstWidth, dstHeight, out_hfov, &valid);
    // rotate output ray into world: w = M * d
    float3 w = (float3)(m00 * d.x + m01 * d.y + m02 * d.z,
                        m10 * d.x + m11 * d.y + m12 * d.z,
                        m20 * d.x + m21 * d.y + m22 * d.z);
    int valid2;
    float2 s = ray_to_in(w, srcWidth, srcHeight, in_hfov, &valid2);

    __global Type *ptrDst = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
    if (!valid || !valid2) { ptrDst[0] = (Type)(fillValue + 0.5f); return; }

    const float sx = s.x, sy = s.y;
    const int x0 = (int)floor(sx), y0 = (int)floor(sy);
    const float fx = sx - x0, fy = sy - y0;
    const float v00 = SAMPLE(x0,     y0);
    const float v10 = SAMPLE(x0 + 1, y0);
    const float v01 = SAMPLE(x0,     y0 + 1);
    const float v11 = SAMPLE(x0 + 1, y0 + 1);
    const float val = v00 * (1.0f - fx) * (1.0f - fy) + v10 * fx * (1.0f - fy)
                    + v01 * (1.0f - fx) * fy          + v11 * fx * fy;
    ptrDst[0] = (Type)(val + 0.5f);
}
