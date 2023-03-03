// Type
// Type4

__kernel void kernel_curves(
    __global uchar *restrict pFrame, const int pitch, const int width, const int height,
    const __global Type *restrict pLut) {
    const int PIX_PER_THREAD = 4;
    const int ix = get_global_id(0) * PIX_PER_THREAD;
    const int iy = get_global_id(1);
    if (ix < width && iy < height) {
        __global Type4 *ptr = (__global Type4 *)(pFrame + iy * pitch + ix * sizeof(Type));

        Type4 pix4 = ptr[0];
        pix4.x = pLut[pix4.x];
        pix4.y = pLut[pix4.y];
        pix4.z = pLut[pix4.z];
        pix4.w = pLut[pix4.w];
        ptr[0] = pix4;
    }
}
