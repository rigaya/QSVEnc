// Type

__kernel void kernel_pad(
    __global uchar *__restrict__ ptrDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const __global uchar *__restrict__ ptrSrc,
    const int srcPitch, const int srcWidth, const int srcHeight,
    const int pad_left, const int pad_up,
    const int pad_color) {
    const int ox = get_global_id(0);
    const int oy = get_global_id(1);

    if (ox < dstWidth && oy < dstHeight) {
        const int ix = ox - pad_left;
        const int iy = oy - pad_up;

        Type out_color = (Type)pad_color;
        if (0 <= ix && ix < srcWidth && 0 <= iy && iy < srcHeight) {
            const __global Type *ptrSrcPix = (const __global Type *)(ptrSrc + iy * srcPitch + ix * sizeof(Type));
            out_color = ptrSrcPix[0];
        }

        __global Type *ptrDstPix = (__global Type *)(ptrDst + oy * dstPitch + ox * sizeof(Type));
        ptrDstPix[0] = out_color;
    }
}
