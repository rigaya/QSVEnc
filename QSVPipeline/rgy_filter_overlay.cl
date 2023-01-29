//Type
//bit_depth

__kernel void kernel_overlay(
    __global uchar *restrict pDst, const int dstPitch,
    const __global uchar *restrict pSrc, const int srcPitch, const int width, const int height,
    const __global uchar *restrict pOverlay, const int overlayPitch,
    const __global uchar *restrict pAlpha, const int alphaPitch, const int overlayWidth, const int overlayHeight,
    const int overlayPosX, const int overlayPosY) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix < width && iy < height) {
        int ret = *(const __global Type *)(pSrc + iy * srcPitch + ix * sizeof(Type));
        if (overlayPosX <= ix && ix < overlayPosX + overlayWidth
            && overlayPosY <= iy && iy < overlayPosY + overlayHeight) {
            const int overlaySrc   = *(const __global Type  *)(pOverlay + (iy - overlayPosY) * overlayPitch + (ix - overlayPosX) * sizeof(Type));
            const int overlayAlpha = *(const __global uchar *)(pAlpha   + (iy - overlayPosY) * alphaPitch   + (ix - overlayPosX) * sizeof(uchar));
            const float overlayAlphaF = overlayAlpha / 255.0f;
            float blend = overlaySrc * overlayAlphaF + ret * (1.0f - overlayAlphaF);
            ret = (int)(blend + 0.5f);
        }

        __global Type *ptr = (__global Type *)(pDst + iy * dstPitch + ix * sizeof(Type));
        ptr[0] = (Type)clamp(ret, 0, (1 << bit_depth) - 1);
    }
}
