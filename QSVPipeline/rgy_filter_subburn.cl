// TypePixel
// TypePixel2
// bit_depth
// yuv420

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

float lerpf(float a, float b, float c) {
    return a + (b - a) * c;
}

TypePixel blend(TypePixel pix, uchar alpha, uchar val, float transparency_offset, float pix_offset, float contrast) {
    //alpha値は 0が透明, 255が不透明
    float subval = val * (1.0f / (float)(1 << 8));
    subval = contrast * (subval - 0.5f) + 0.5f + pix_offset;
    float ret = lerpf((float)pix, subval * (float)(1 << bit_depth), alpha * (1.0f / 255.0f) * (1.0f - transparency_offset));
    return (TypePixel)clamp(ret, 0.0f, (1 << bit_depth) - 0.5f);
}

void blend2(__global void *pix, const __global void *alpha, const __global void *val, float transparency_offset, float pix_offset, float contrast) {
    uchar2 a = *(__global uchar2 *)alpha;
    uchar2 v = *(__global uchar2 *)val;
    TypePixel2 p = *(__global TypePixel2 *)pix;
    p.x = blend(p.x, a.x, v.x, transparency_offset, pix_offset, contrast);
    p.y = blend(p.y, a.y, v.y, transparency_offset, pix_offset, contrast);
    *(__global TypePixel2 *)pix = p;
}

__kernel void kernel_subburn(
    __global uchar *pPlaneY,
    __global uchar *pPlaneU,
    __global uchar *pPlaneV,
    const int pitchFrameY,
    const int pitchFrameU,
    const int pitchFrameV,
    const int frameOffsetByteY,
    const int frameOffsetByteU,
    const int frameOffsetByteV,
    const __global uchar *pSubY,
    const __global uchar *pSubU,
    const __global uchar *pSubV,
    const __global uchar *pSubA,
    const int pitchSub,
    const int width, const int height, const int interlaced,
    const float transparency_offset, const float brightness, const float contrast) {
    //縦横2x2pixelを1スレッドで処理する
    const int ix = get_global_id(0) * 2;
    const int iy = get_global_id(1) * 2;

    pPlaneY += frameOffsetByteY;
    pPlaneU += frameOffsetByteU;
    pPlaneV += frameOffsetByteV;

    if (ix < width && iy < height) {
        pPlaneY += iy * pitchFrameY + ix * sizeof(TypePixel);
        pSubY += iy * pitchSub + ix;
        pSubU += iy * pitchSub + ix;
        pSubV += iy * pitchSub + ix;
        pSubA += iy * pitchSub + ix;

        blend2(pPlaneY, pSubA, pSubY, transparency_offset, brightness, contrast);
        blend2(pPlaneY + pitchFrameY, pSubA + pitchSub, pSubY + pitchSub, transparency_offset, brightness, contrast);

        if (yuv420) {
            pPlaneU += (iy >> 1) * pitchFrameU + (ix >> 1) * sizeof(TypePixel);
            pPlaneV += (iy >> 1) * pitchFrameV + (ix >> 1) * sizeof(TypePixel);
            uchar subU, subV, subA;
            if (interlaced) {
                if (((iy >> 1) & 1) == 0) {
                    const int offset_y1 = (iy + 2 < height) ? pitchSub * 2 : 0;
                    subU = (pSubU[0] * 3 + pSubU[offset_y1] + 2) >> 2;
                    subV = (pSubV[0] * 3 + pSubV[offset_y1] + 2) >> 2;
                    subA = (pSubA[0] * 3 + pSubA[offset_y1] + 2) >> 2;
                } else {
                    subU = (pSubU[-pitchSub] + pSubU[pitchSub] * 3 + 2) >> 2;
                    subV = (pSubV[-pitchSub] + pSubV[pitchSub] * 3 + 2) >> 2;
                    subA = (pSubA[-pitchSub] + pSubA[pitchSub] * 3 + 2) >> 2;
                }
            } else {
                subU = (pSubU[0] + pSubU[pitchSub] + 1) >> 1;
                subV = (pSubV[0] + pSubV[pitchSub] + 1) >> 1;
                subA = (pSubA[0] + pSubA[pitchSub] + 1) >> 1;
            }
            *(__global TypePixel *)pPlaneU = blend(*(__global TypePixel *)pPlaneU, subA, subU, transparency_offset, 0.0f, 1.0f);
            *(__global TypePixel *)pPlaneV = blend(*(__global TypePixel *)pPlaneV, subA, subV, transparency_offset, 0.0f, 1.0f);
        } else {
            pPlaneU += iy * pitchFrameU + ix * sizeof(TypePixel);
            pPlaneV += iy * pitchFrameV + ix * sizeof(TypePixel);
            blend2(pPlaneU, pSubA, pSubU, transparency_offset, 0.0f, 1.0f);
            blend2(pPlaneU + pitchFrameU, pSubA + pitchSub, pSubU + pitchSub, transparency_offset, 0.0f, 1.0f);
            blend2(pPlaneV, pSubA, pSubV, transparency_offset, 0.0f, 1.0f);
            blend2(pPlaneV + pitchFrameV, pSubA + pitchSub, pSubV + pitchSub, transparency_offset, 0.0f, 1.0f);
        }
    }
}
