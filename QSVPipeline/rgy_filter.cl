// MEM_TYPE_SRC
// MEM_TYPE_DST
// in_bit_depth
// out_bit_depth

// RGY_MATRIX_ST170_M
// RGY_MATRIX_ST240_M
// RGY_MATRIX_BT2020_NCL
// RGY_MATRIX_BT2020_CL
// RGY_MATRIX_BT709

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

#define RGB_ORDER_RGB (0)
#define RGB_ORDER_BGR (1)
#define RGB_ORDER_GBR (2)
#define RGB_ORDER_RBG (3)

#if in_bit_depth <= 8
#define TypeIn  uchar
#define TypeIn4 uchar4
#define convert_TypeIn4 convert_uchar4
#elif in_bit_depth <= 16
#define TypeIn  ushort
#define TypeIn4 ushort4
#define convert_TypeIn4 convert_ushort4
#elif in_bit_depth == 32
#define TypeIn  float
#define TypeIn4 float4
#define convert_TypeIn4 convert_float4
#endif

#if out_bit_depth <= 8
#define TypeOut  uchar
#define TypeOut4 uchar4
#define convert_TypeOut4 convert_uchar4
#elif out_bit_depth <= 16
#define TypeOut  ushort
#define TypeOut4 ushort4
#define convert_TypeOut4 convert_ushort4
#elif out_bit_depth == 32
#define TypeOut  float
#define TypeOut4 float4
#define convert_TypeOut4 convert_float4
#endif

// samplerでnormalizeした場合、0 -> 0.0f, 255 -> 1.0f

#define RGY_MEM_TYPE_CPU                    (0)
#define RGY_MEM_TYPE_GPU                    (1)
#define RGY_MEM_TYPE_GPU_IMAGE              (2)
#define RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED   (3)

#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define __read_only
#define __write_only
#define image2d_t void*
#define uchar unsigned char
#endif

#define IntegerIsSigned(intType)    ((intType)(-1) < 0)

inline int conv_bit_depth_lsft(const int bit_depth_in, const int bit_depth_out, const int shift_offset) {
    const int lsft = bit_depth_out - (bit_depth_in + shift_offset);
    return lsft < 0 ? 0 : lsft;
}

inline int conv_bit_depth_rsft(const int bit_depth_in, const int bit_depth_out, const int shift_offset) {
    const int rsft = bit_depth_in + shift_offset - bit_depth_out;
    return rsft < 0 ? 0 : rsft;
}

inline int conv_bit_depth_rsft_add(const int bit_depth_in, const int bit_depth_out, const int shift_offset) {
    const int rsft = conv_bit_depth_rsft(bit_depth_in, bit_depth_out, shift_offset);
    return (rsft - 1 >= 0) ? 1 << (rsft - 1) : 0;
}

inline int conv_bit_depth(const int c, const int bit_depth_in, const int bit_depth_out, const int shift_offset) {
    if (bit_depth_out > bit_depth_in + shift_offset) {
        return c << conv_bit_depth_lsft(bit_depth_in, bit_depth_out, shift_offset);
    } else if (bit_depth_out < bit_depth_in + shift_offset) {
        const int x = (c + conv_bit_depth_rsft_add(bit_depth_in, bit_depth_out, shift_offset)) >> conv_bit_depth_rsft(bit_depth_in, bit_depth_out, shift_offset);
        const int low = 0;
        const int high = (1 << bit_depth_out) - 1;
        return (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high));
    } else {
        return c;
    }
}

#define NORM_SCALE_IN  (float)((1<<(sizeof(TypeIn)*8))-1)
#define NORM_SCALE_OUT (1.0f/(float)((1<<(sizeof(TypeOut)*8))-1))

#define BIT_DEPTH_CONV(x) (TypeOut)conv_bit_depth((x), in_bit_depth, out_bit_depth, 0)

#define BIT_DEPTH_CONV_FLOAT(x) (TypeOut)((out_bit_depth == in_bit_depth) \
    ? (x) \
    : ((out_bit_depth > in_bit_depth) \
        ? ((x) * (float)(1 << (out_bit_depth - in_bit_depth))) \
        : ((x) * (float)(1.0f / (1 << (in_bit_depth - out_bit_depth))))))

#define BIT_DEPTH_CONV_AVG(a, b) (TypeOut)conv_bit_depth((a)+(b), in_bit_depth, out_bit_depth, 1)

#define BIT_DEPTH_CONV_3x1_AVG(a, b) (TypeOut)conv_bit_depth(((a)<<1)+(a)+(b), in_bit_depth, out_bit_depth, 2)

#define BIT_DEPTH_CONV_7x1_AVG(a, b) (TypeOut)conv_bit_depth(((a)<<3)-(a)+(b), in_bit_depth, out_bit_depth, 3)

#define AVG3x1(a, b) ((((a)<<1)+(a)+(b)+2)>>2)
#define AVG7x1(a, b) ((((a)<<3)-(a)+(b)+4)>>3)

#define LOAD_IMG(src_img, ix, iy) (TypeIn)(read_imageui((src_img), sampler, (int2)((ix), (iy))).x)
#define LOAD_IMG_AYUV(src_img, ix, iy) convert_TypeIn4(read_imageui((src_img), sampler, (int2)((ix), (iy))))
#define LOAD_IMG_NV12_UV(src_img, src_u, src_v, ix, iy, cropX, cropY) { \
    uint4 ret = read_imageui((src_img), sampler, (int2)((ix) + ((cropX)>>1), (iy) + ((cropY)>>1))); \
    (src_u) = (TypeIn)ret.x; \
    (src_v) = (TypeIn)ret.y; \
}
#define LOAD_BUF(src_buf, ix, iy) *(__global TypeIn *)(&(src_buf)[(iy) * srcPitch + (ix) * sizeof(TypeIn)])
#define LOAD_BUF_AYUV(src_buf, ix, iy) *(__global TypeIn4 *)(&(src_buf)[(iy) * srcPitch + (ix) * sizeof(TypeIn4)])
#define LOAD_BUF_NV12_UV(src_buf, src_u, src_v, ix, iy, cropX, cropY) { \
    (src_u) = LOAD((src_buf), ((ix)<<1) + 0 + (cropX), (iy) + ((cropY)>>1)); \
    (src_v) = LOAD((src_buf), ((ix)<<1) + 1 + (cropX), (iy) + ((cropY)>>1)); \
}

#define LOAD_IMG_NORM(src_img, ix, iy) (TypeIn)(read_imagef((src_img), sampler, (int2)((ix), (iy))).x * NORM_SCALE_IN + 0.5f)
#define LOAD_IMG_NORM_AYUV(src_img, ix, iy) convert_TypeIn4(read_imagef((src_img), sampler, (int2)((ix), (iy))) * (float4)NORM_SCALE_IN + (float4)0.5f)
#define LOAD_IMG_NORM_NV12_UV(src_img, src_u, src_v, ix, iy, cropX, cropY) { \
    float4 ret = read_imagef((src_img), sampler, (int2)((ix) + ((cropX)>>1), (iy) + ((cropY)>>1))); \
    (src_u) = (TypeIn)(ret.x * NORM_SCALE_IN + 0.5f); \
    (src_v) = (TypeIn)(ret.y * NORM_SCALE_IN + 0.5f); \
}

#if MEM_TYPE_SRC == RGY_MEM_TYPE_GPU_IMAGE
#define IMAGE_SRC    1
#define LOAD         LOAD_IMG
#define LOAD_AYUV    LOAD_IMG_AYUV
#define LOAD_NV12_UV LOAD_IMG_NV12_UV
#elif MEM_TYPE_SRC == RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED
#define IMAGE_SRC    1
#define LOAD         LOAD_IMG_NORM
#define LOAD_AYUV    LOAD_IMG_NORM_AYUV
#define LOAD_NV12_UV LOAD_IMG_NORM_NV12_UV
#else
#define IMAGE_SRC    0
#define LOAD         LOAD_BUF
#define LOAD_AYUV    LOAD_BUF_AYUV
#define LOAD_NV12_UV LOAD_BUF_NV12_UV
#endif

#define STORE_IMG(dst_img, ix, iy, val) write_imageui((dst_img), (int2)((ix), (iy)), (val))
#define STORE_IMG_AYUV(dst_img, ix, iy, val) write_imageui((dst_img), (int2)((ix), (iy)), convert_uint4(val))
#define STORE_IMG_NV12_UV(dst_img, ix, iy, val_u, val_v) { \
    uint4 val = (uint4)(val_u, val_v, val_v, val_v); \
    write_imageui((dst_img), (int2)((ix), (iy)), (val)); \
}

#define STORE_IMG_NORM(dst_img, ix, iy, val) write_imagef(dst_img, (int2)((ix), (iy)), (val * NORM_SCALE_OUT))
#define STORE_IMG_NORM_AYUV(dst_img, ix, iy, val) write_imagef(dst_img, (int2)((ix), (iy)), (convert_float4(val) * (float4)NORM_SCALE_OUT))
#define STORE_IMG_NORM_NV12_UV(dst_img, ix, iy, val_u, val_v) { \
    float4 val = (float4)(val_u * NORM_SCALE_OUT, val_v * NORM_SCALE_OUT, val_v * NORM_SCALE_OUT, val_v * NORM_SCALE_OUT); \
    write_imagef(dst_img, (int2)((ix), (iy)), (val)); \
}
#define STORE_BUF(dst_buf, ix, iy, val)  { \
    __global TypeOut *ptr = (__global TypeOut *)(&(dst_buf)[(iy) * dstPitch + (ix) * sizeof(TypeOut)]); \
    ptr[0] = (TypeOut)(val); \
}
#define STORE_BUF_AYUV(dst_buf, ix, iy, val)  { \
    __global TypeOut4 *ptr = (__global TypeOut4 *)(&(dst_buf)[(iy) * dstPitch + (ix) * sizeof(TypeOut4)]); \
    ptr[0] = (TypeOut4)(val); \
}
#define STORE_BUF_NV12_UV(dst_buf, ix, iy, val_u, val_v) { \
    STORE(dst_buf, ((ix) << 1) + 0, (iy), val_u); \
    STORE(dst_buf, ((ix) << 1) + 1, (iy), val_v); \
}
#if MEM_TYPE_DST == RGY_MEM_TYPE_GPU_IMAGE
#define IMAGE_DST     1
#define STORE         STORE_IMG
#define STORE_AYUV    STORE_IMG_AYUV
#define STORE_NV12_UV STORE_IMG_NV12_UV
#elif MEM_TYPE_DST == RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED
#define IMAGE_DST     1
#define STORE         STORE_IMG_NORM
#define STORE_AYUV    STORE_IMG_NORM_AYUV
#define STORE_NV12_UV STORE_IMG_NORM_NV12_UV
#else
#define IMAGE_DST     0
#define STORE         STORE_BUF
#define STORE_AYUV    STORE_BUF_AYUV
#define STORE_NV12_UV STORE_BUF_NV12_UV
#endif

void conv_c_yuv420_yuv444_internal(
    int *pixDst11, int *pixDst12,
    int *pixDst21, int *pixDst22,
    int pixSrc01, int pixSrc02,
    int pixSrc11, int pixSrc12,
    int pixSrc21, int pixSrc22
) {
    pixSrc02 = (pixSrc01 + pixSrc02 + 1) >> 1;
    pixSrc12 = (pixSrc11 + pixSrc12 + 1) >> 1;
    pixSrc22 = (pixSrc21 + pixSrc22 + 1) >> 1;

    *pixDst11 = BIT_DEPTH_CONV_3x1_AVG(pixSrc11, pixSrc01);
    *pixDst12 = BIT_DEPTH_CONV_3x1_AVG(pixSrc12, pixSrc02);
    *pixDst21 = BIT_DEPTH_CONV_7x1_AVG(pixSrc11, pixSrc21);
    *pixDst22 = BIT_DEPTH_CONV_7x1_AVG(pixSrc12, pixSrc22);
}
void yuv420_yuv444_no_bitdepth_change(
    int *pixDst11, int *pixDst12,
    int *pixDst21, int *pixDst22,
    int pixSrc01, int pixSrc02,
    int pixSrc11, int pixSrc12,
    int pixSrc21, int pixSrc22
) {
    pixSrc02 = (pixSrc01 + pixSrc02 + 1) >> 1;
    pixSrc12 = (pixSrc11 + pixSrc12 + 1) >> 1;
    pixSrc22 = (pixSrc21 + pixSrc22 + 1) >> 1;

    *pixDst11 = AVG3x1(pixSrc11, pixSrc01);
    *pixDst12 = AVG3x1(pixSrc12, pixSrc02);
    *pixDst21 = AVG7x1(pixSrc11, pixSrc21);
    *pixDst22 = AVG7x1(pixSrc12, pixSrc22);
}

void conv_c_yuv420_yuv444(
#if IMAGE_DST
    __write_only image2d_t dst,
#else
    __global uchar *dst,
#endif
    const int dstPitch,
    const int dst_x, const int dst_y,
    int pixSrc01, int pixSrc02,
    int pixSrc11, int pixSrc12,
    int pixSrc21, int pixSrc22
) {
    int pixDst11, pixDst12, pixDst21, pixDst22;
    conv_c_yuv420_yuv444_internal(
        &pixDst11, &pixDst12, &pixDst21, &pixDst22,
        pixSrc01, pixSrc02, pixSrc11, pixSrc12, pixSrc21, pixSrc22
    );

    STORE(dst, dst_x+0, dst_y+0, (TypeOut)pixDst11);
    STORE(dst, dst_x+1, dst_y+0, (TypeOut)pixDst12);
    STORE(dst, dst_x+0, dst_y+1, (TypeOut)pixDst21);
    STORE(dst, dst_x+1, dst_y+1, (TypeOut)pixDst22);
}

__kernel void kernel_copy_plane(
#if IMAGE_DST
    __write_only image2d_t dst,
#else
    __global uchar *dst,
#endif
    int dstPitch,
    int dstOffsetX,
    int dstOffsetY,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcOffsetX,
    int srcOffsetY,
    int width,
    int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < width && y < height) {
        TypeIn pixSrc = LOAD(src, x + srcOffsetX, y + srcOffsetY);
        TypeOut out = BIT_DEPTH_CONV(pixSrc);
        STORE(dst, x + dstOffsetX, y + dstOffsetY, out);
    }
}

__kernel void kernel_copy_plane_nv12(
#if IMAGE_DST
    __write_only image2d_t dst,
#else
    __global uchar *dst,
#endif
    int dstPitch,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int uvWidth,
    int uvHeight,
    int cropX,     // 輝度と同じcropを想定
    int cropY      // 輝度と同じcropを想定
) {
    const int uv_x = get_global_id(0);
    const int uv_y = get_global_id(1);
    if (uv_x < uvWidth && uv_y < uvHeight) {
        TypeIn pixSrcU, pixSrcV;
        LOAD_NV12_UV(src, pixSrcU, pixSrcV, uv_x, uv_y, cropX, cropY);
        TypeOut pixDstU = BIT_DEPTH_CONV(pixSrcU);
        TypeOut pixDstV = BIT_DEPTH_CONV(pixSrcV);
        STORE_NV12_UV(dst, uv_x, uv_y, pixDstU, pixDstV);
    }
}

__kernel void kernel_crop_nv12_yv12(
#if IMAGE_DST
    __write_only image2d_t dstU,
    __write_only image2d_t dstV,
#else
    __global uchar *dstU,
    __global uchar *dstV,
#endif
    int dstPitch,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int uvWidth,
    int uvHeight,
    int cropX,     // 輝度と同じcropを想定
    int cropY      // 輝度と同じcropを想定
) {
    const int uv_x = get_global_id(0);
    const int uv_y = get_global_id(1);
    if (uv_x < uvWidth && uv_y < uvHeight) {
        TypeIn pixSrcU, pixSrcV;
        LOAD_NV12_UV(src, pixSrcU, pixSrcV, uv_x, uv_y, cropX, cropY);
        TypeOut pixDstU = BIT_DEPTH_CONV(pixSrcU);
        TypeOut pixDstV = BIT_DEPTH_CONV(pixSrcV);
        STORE(dstU, uv_x, uv_y, pixDstU);
        STORE(dstV, uv_x, uv_y, pixDstV);
    }
}

__kernel void kernel_crop_nv12_yuv444(
#if IMAGE_DST
    __write_only image2d_t dstU,
    __write_only image2d_t dstV,
#else
    __global uchar *dstU,
    __global uchar *dstV,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,     // 輝度と同じcropを想定
    int cropY      // 輝度と同じcropを想定
) {
    const int src_x = get_global_id(0);
    const int src_y = get_global_id(1);
    const int dst_x = src_x << 1;
    const int dst_y = src_y << 1;

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int loadx = src_x + (cropX>>1);
        const int loady = src_y + (cropY>>1);

        TypeIn pixSrcU01, pixSrcV01, pixSrcU02, pixSrcV02;
        TypeIn pixSrcU11, pixSrcV11, pixSrcU12, pixSrcV12;
        TypeIn pixSrcU21, pixSrcV21, pixSrcU22, pixSrcV22;
        LOAD_NV12_UV(src, pixSrcU01, pixSrcV01,     loadx,                max(loady-1, 0),           0, 0);
        LOAD_NV12_UV(src, pixSrcU02, pixSrcV02, min(loadx+1, srcWidth-1), max(loady-1, 0),           0, 0);
        LOAD_NV12_UV(src, pixSrcU11, pixSrcV11,     loadx,                    loady,                 0, 0);
        LOAD_NV12_UV(src, pixSrcU12, pixSrcV12, min(loadx+1, srcWidth-1),     loady,                 0, 0);
        LOAD_NV12_UV(src, pixSrcU21, pixSrcV21,     loadx,                min(loady+1, srcHeight-1), 0, 0);
        LOAD_NV12_UV(src, pixSrcU22, pixSrcV22, min(loadx+1, srcWidth-1), min(loady+1, srcHeight-1), 0, 0);

        conv_c_yuv420_yuv444(dstU, dstPitch, dst_x, dst_y, pixSrcU01, pixSrcU02, pixSrcU11, pixSrcU12, pixSrcU21, pixSrcU22);
        conv_c_yuv420_yuv444(dstV, dstPitch, dst_x, dst_y, pixSrcV01, pixSrcV02, pixSrcV11, pixSrcV12, pixSrcV21, pixSrcV22);
    }
}

__kernel void kernel_crop_c_yuv444_nv12(
#if IMAGE_DST
    __write_only image2d_t dst,
#else
    __global uchar *dst,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t srcU,
    __read_only image2d_t srcV,
#else
    __global uchar *srcU,
    __global uchar *srcV,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,     // 輝度と同じcropを想定
    int cropY      // 輝度と同じcropを想定
) {
    const int dst_x = get_global_id(0);
    const int dst_y = get_global_id(1);

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int src_x = dst_x << 1;
        const int src_y = dst_y << 1;
        const int loadx = src_x + cropX;
        const int loady = src_y + cropY;
        const TypeIn pixSrcU00 = LOAD(srcU, loadx+0, loady+0);
        const TypeIn pixSrcU10 = LOAD(srcU, loadx+0, loady+1);
        const TypeIn pixSrcV00 = LOAD(srcV, loadx+0, loady+0);
        const TypeIn pixSrcV10 = LOAD(srcV, loadx+0, loady+1);
        TypeOut pixDstU = BIT_DEPTH_CONV_AVG(pixSrcU00, pixSrcU10);
        TypeOut pixDstV = BIT_DEPTH_CONV_AVG(pixSrcV00, pixSrcV10);
        STORE_NV12_UV(dst, dst_x, dst_y, pixDstU, pixDstV);
    }
}

__kernel void kernel_crop_yv12_nv12(
#if IMAGE_DST
    __write_only image2d_t dst,
#else
    __global uchar *dst,
#endif
    int dstPitch,
#if IMAGE_SRC
    __read_only image2d_t srcU,
    __read_only image2d_t srcV,
#else
    __global uchar *srcU,
    __global uchar *srcV,
#endif
    int srcPitch,
    int uvWidth,
    int uvHeight,
    int cropX,     // 輝度と同じcropを想定
    int cropY      // 輝度と同じcropを想定
) {
    const int uv_x = get_global_id(0);
    const int uv_y = get_global_id(1);

    if (uv_x < uvWidth && uv_y < uvHeight) {
        const TypeIn pixSrcU = LOAD(srcU, uv_x + (cropX>>1), uv_y + (cropY>>1));
        const TypeIn pixSrcV = LOAD(srcV, uv_x + (cropX>>1), uv_y + (cropY>>1));
        const TypeOut pixDstU = BIT_DEPTH_CONV(pixSrcU);
        const TypeOut pixDstV = BIT_DEPTH_CONV(pixSrcV);
        STORE_NV12_UV(dst, uv_x, uv_y, pixDstU, pixDstV);
    }
}

__kernel void kernel_crop_c_yv12_yuv444(
#if IMAGE_DST
    __write_only image2d_t dst,
#else
    __global uchar *dst,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,     // 輝度と同じcropを想定
    int cropY      // 輝度と同じcropを想定
) {
    const int src_x = get_global_id(0);
    const int src_y = get_global_id(1);
    const int dst_x = src_x << 1;
    const int dst_y = src_y << 1;

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int loadx = src_x + (cropX>>1);
        const int loady = src_y + (cropY>>1);
        const int pixSrc01 = LOAD(src,     loadx,                max(loady-1, 0)          );
        const int pixSrc02 = LOAD(src, min(loadx+1, srcWidth-1), max(loady-1, 0)          );
        const int pixSrc11 = LOAD(src,     loadx,                    loady                );
        const int pixSrc12 = LOAD(src, min(loadx+1, srcWidth-1),     loady                );
        const int pixSrc21 = LOAD(src,     loadx,                min(loady+1, srcHeight-1));
        const int pixSrc22 = LOAD(src, min(loadx+1, srcWidth-1), min(loady+1, srcHeight-1));

        conv_c_yuv420_yuv444(dst, dstPitch, dst_x, dst_y, pixSrc01, pixSrc02, pixSrc11, pixSrc12, pixSrc21, pixSrc22);
    }
}

__kernel void kernel_crop_c_yuv444_yv12(
#if IMAGE_DST
    __write_only image2d_t dst,
#else
    __global uchar *dst,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,     // 輝度と同じcropを想定
    int cropY      // 輝度と同じcropを想定
) {
    const int dst_x = get_global_id(0);
    const int dst_y = get_global_id(1);

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int src_x = dst_x << 1;
        const int src_y = dst_y << 1;
        const int loadx = src_x + cropX;
        const int loady = src_y + cropY;
        const int pixSrc00 = LOAD(src, loadx+0, loady+0);
        const int pixSrc10 = LOAD(src, loadx+0, loady+1);
        const TypeOut pixDst = BIT_DEPTH_CONV_AVG(pixSrc00, pixSrc10);
        STORE(dst, dst_x, dst_y, pixDst);
    }
}

#define mat(i,j) (pmat[i*3+j])

float3 conv_rgb_yuv(const float3 rgb, const int matrix) {
    const float mat_bt601[9] = {
        0.299f, 0.587f, 0.114f,
        -0.168735892f, -0.331264108f, 0.5f,
        0.5f, -0.418687589f, -0.081312411f
    };
    const float mat_bt709[9] = {
        0.2126f, 0.7152f, 0.0722f,
        -0.13500127f, -0.454152908f, 0.589154178f,
        0.424337142f, -0.385427894f, -0.038909248f
    };
    const float mat_bt2020[9] = {
        0.2627f, 0.678f, 0.0593f,
        -0.178150007f, -0.459785705f, 0.637935711f,
        0.391889019f, -0.360369937f, -0.031519082f
    };
    const float mat_st240m[9] = {
        0.212f, 0.701f, 0.087f,
        -0.134517766f, -0.444796954f, 0.579314721f,
        0.431544359f, -0.383899233f, -0.047645126f
    };
    const float *pmat = mat_bt601;
    switch (matrix) {
        case RGY_MATRIX_BT709: pmat = mat_bt709; break;
        case RGY_MATRIX_BT2020_NCL:
        case RGY_MATRIX_BT2020_CL: pmat = mat_bt2020; break;
        case RGY_MATRIX_ST240_M: pmat = mat_st240m; break;
        case RGY_MATRIX_ST170_M:
        default: break;
    };

    float3 yuv;
    yuv.x = mat(0,0) * rgb.x + mat(0,1) * rgb.y + mat(0,2) * rgb.z;
    yuv.y = mat(1,0) * rgb.x + mat(1,1) * rgb.y + mat(1,2) * rgb.z;
    yuv.z = mat(2,0) * rgb.x + mat(2,1) * rgb.y + mat(2,2) * rgb.z;
    return yuv;
}

float3 conv_yuv_rgb(const float3 yuv, const int matrix) {
    const float mat_bt601[9] = {
        1.0f, 0.0f, 1.402f,
        1.0f, -0.344136286f, -0.714136286f,
        1.0f, 1.772f, 0.0f
    };
    const float mat_bt709[9] = {
        1.0f, 0.0f, 1.8556f,
        1.0f, -0.158977293f, -0.551594743f,
        1.0f, 1.5748f, 0.0f
    };
    const float mat_bt2020[9] = {
        1.0f, 0.0f, 1.8814f,
        1.0f, -0.128973127f, -0.728973127f,
        1.0f, 1.4746f, 0.0f
    };
    const float mat_st240m[9] = {
        1.0f, 0.0f, 1.826f,
        1.0f, -0.195594864f, -0.552228245f,
        1.0f, 1.576f, 0.0f
    };
    const float *pmat = mat_bt601;
    switch (matrix) {
        case RGY_MATRIX_BT709: pmat = mat_bt709; break;
        case RGY_MATRIX_BT2020_NCL:
        case RGY_MATRIX_BT2020_CL: pmat = mat_bt2020; break;
        case RGY_MATRIX_ST240_M: pmat = mat_st240m; break;
        case RGY_MATRIX_ST170_M:
        default: break;
    };

    float3 rgb;
    rgb.x = mat(0,0) * yuv.x + mat(0,1) * yuv.y + mat(0,2) * yuv.z;
    rgb.y = mat(1,0) * yuv.x + mat(1,1) * yuv.y + mat(1,2) * yuv.z;
    rgb.z = mat(2,0) * yuv.x + mat(2,1) * yuv.y + mat(2,2) * yuv.z;
    return rgb;
}

#undef mat

TypeOut scaleRGBFloatToPix(float x) {
    if (out_bit_depth == 32) {
        return x;
    }
    const float range = (float)((1ll << out_bit_depth) - 1);
    return (TypeOut)clamp(x * range + 0.5f, 0.0f, (float)(1ll << (out_bit_depth)) - 0.5f);
}

TypeOut scaleYFloatToPix(float x) {
    if (out_bit_depth == 32) {
        return x;
    }
    const float range = (float)(219 << (out_bit_depth - 8));
    const float offset = (float)(16 << (out_bit_depth - 8));
    return (TypeOut)clamp(x * range + offset + 0.5f, 0.0f, (float)(1ll << (out_bit_depth)) - 0.5f);
}

TypeOut scaleUVFloatToPix(float x) {
    if (out_bit_depth == 32) {
        return x;
    }
    const float range = (float)(224 << (out_bit_depth - 8));
    const float offset = (float)(1 << (out_bit_depth - 1));
    return (TypeOut)clamp(x * range + offset + 0.5f, 0.0f, (float)(1ll << (out_bit_depth)) - 0.5f);
}

float scaleRGBPixToFloat(TypeIn x) {
    if (in_bit_depth == 32) {
        return x;
    }
    const float range = (float)((1ll << in_bit_depth) - 1);
    const float range_inv = 1.0f / range;
    return clamp((float)x * range_inv, 0.0f, 1.0f);
}

float scaleYPixToFloat(TypeIn x) {
    if (in_bit_depth == 32) {
        return x;
    }
    const float range = (float)(219 << (in_bit_depth - 8));
    const float offset = (float)(16 << (in_bit_depth - 8));
    const float range_inv = 1.0f / range;
    const float offset_inv = -offset * (1.0f / range);
    return clamp((float)x * range_inv + offset_inv, 0.0f, 1.0f);
}

float scaleUVPixToFloat(TypeIn x) {
    if (in_bit_depth == 32) {
        return x;
    }
    const float range = (float)(224 << (in_bit_depth - 8));
    const float offset = (float)(1 << (in_bit_depth - 1));
    const float range_inv = 1.0f / range;
    const float offset_inv = -offset * (1.0f / range);
    return clamp((float)x * range_inv + offset_inv, -0.5f, 0.5f);
}

float3 make_float_yuv3(TypeIn y, TypeIn u, TypeIn v) {
    if (in_bit_depth == 32) {
        return (float3)(y, u, v);
    }
    return (float3)(
        scaleYPixToFloat(y),
        scaleUVPixToFloat(u),
        scaleUVPixToFloat(v));
}

float3 make_float_rgb3(TypeIn r, TypeIn g, TypeIn b) {
    if (in_bit_depth == 32) {
        return (float3)(r, g, b);
    }
    return (float3)(
        scaleRGBPixToFloat(r),
        scaleRGBPixToFloat(g),
        scaleRGBPixToFloat(b));
}

float3 make_float_rgb3_from_TypeIn4(TypeIn4 rgb) {
    if (in_bit_depth == 32) {
        return (float3)(rgb.x, rgb.y, rgb.z);
    }
    return (float3)(
        scaleRGBPixToFloat(rgb.x),
        scaleRGBPixToFloat(rgb.y),
        scaleRGBPixToFloat(rgb.z));
}

TypeIn4 bgr3_to_rgb3(TypeIn4 bgr) {
    return (TypeIn4)(bgr.z, bgr.y, bgr.x, bgr.w);
}

TypeIn4 gbr3_to_rgb3(TypeIn4 gbr) {
    return (TypeIn4)(gbr.y, gbr.z, gbr.x, gbr.w);
}

TypeIn4 rbg3_to_rgb3(TypeIn4 rbg) {
    return (TypeIn4)(rbg.x, rbg.z, rbg.y, rbg.w);
}

TypeOut4 rgb3_to_bgr3(TypeOut4 rgb) {
    return (TypeOut4)(rgb.z, rgb.y, rgb.x, rgb.w);
}

TypeOut4 rgb3_to_gbr3(TypeOut4 rgb) {
    return (TypeOut4)(rgb.y, rgb.z, rgb.x, rgb.w);
}

TypeOut4 rgb3_to_rbg3(TypeOut4 rgb) {
    return (TypeOut4)(rgb.x, rgb.z, rgb.y, rgb.w);
}

TypeIn4 rgb_packed_to_rgb3(TypeIn4 rgb_packed, int rgb_order) {
    switch (rgb_order) {
        case RGB_ORDER_BGR: return bgr3_to_rgb3(rgb_packed);
        case RGB_ORDER_GBR: return gbr3_to_rgb3(rgb_packed);
        case RGB_ORDER_RBG: return rbg3_to_rgb3(rgb_packed);
        case RGB_ORDER_RGB:
        default: return rgb_packed;
    }
}

TypeOut4 rgb3_to_rgb_packed(TypeOut4 rgb3, int rgb_order) {
    switch (rgb_order) {
        case RGB_ORDER_BGR: return rgb3_to_bgr3(rgb3);
        case RGB_ORDER_GBR: return rgb3_to_gbr3(rgb3);
        case RGB_ORDER_RBG: return rgb3_to_rbg3(rgb3);
        case RGB_ORDER_RGB: 
        default: return rgb3;
    }
}

void crop_rgb_packed_yuv444(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstU,
    __write_only image2d_t dstV,
#else
    __global uchar *dstY,
    __global uchar *dstU,
    __global uchar *dstV,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix,
    int rgb_order
) {
    const int dst_x = get_global_id(0);
    const int dst_y = get_global_id(1);

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int loadx = dst_x + cropX;
        const int loady = dst_y + cropY;
        const TypeIn4 pixSrcRGB = rgb_packed_to_rgb3(LOAD_AYUV(src, loadx, loady), rgb_order);
        float3 yuv = conv_rgb_yuv(make_float_rgb3_from_TypeIn4(pixSrcRGB), matrix);
        STORE(dstY, dst_x, dst_y, scaleYFloatToPix(yuv.x));
        STORE(dstU, dst_x, dst_y, scaleUVFloatToPix(yuv.y));
        STORE(dstV, dst_x, dst_y, scaleUVFloatToPix(yuv.z));
    }
}

__kernel void kernel_crop_rgb32_yuv444(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstU,
    __write_only image2d_t dstV,
#else
    __global uchar *dstY,
    __global uchar *dstU,
    __global uchar *dstV,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    crop_rgb_packed_yuv444(dstY, dstU, dstV, dstPitch, dstWidth, dstHeight, src, srcPitch, srcWidth, srcHeight, cropX, cropY, matrix, RGB_ORDER_RGB);
}

__kernel void kernel_crop_bgr32_yuv444(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstU,
    __write_only image2d_t dstV,
#else
    __global uchar *dstY,
    __global uchar *dstU,
    __global uchar *dstV,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    crop_rgb_packed_yuv444(dstY, dstU, dstV, dstPitch, dstWidth, dstHeight, src, srcPitch, srcWidth, srcHeight, cropX, cropY, matrix, RGB_ORDER_BGR);
}

__kernel void kernel_crop_gbr32_yuv444(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstU,
    __write_only image2d_t dstV,
#else
    __global uchar *dstY,
    __global uchar *dstU,
    __global uchar *dstV,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    crop_rgb_packed_yuv444(dstY, dstU, dstV, dstPitch, dstWidth, dstHeight, src, srcPitch, srcWidth, srcHeight, cropX, cropY, matrix, RGB_ORDER_GBR);
}

__kernel void kernel_crop_rbg32_yuv444(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstU,
    __write_only image2d_t dstV,
#else
    __global uchar *dstY,
    __global uchar *dstU,
    __global uchar *dstV,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    crop_rgb_packed_yuv444(dstY, dstU, dstV, dstPitch, dstWidth, dstHeight, src, srcPitch, srcWidth, srcHeight, cropX, cropY, matrix, RGB_ORDER_RBG);
}

void crop_rgb_packed_yv12(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstU,
    __write_only image2d_t dstV,
#else
    __global uchar *dstY,
    __global uchar *dstU,
    __global uchar *dstV,
#endif
    int dstPitchY,
    int dstPitchU,
    int dstPitchV,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix,
    int rgb_order
) {
    const int dstC_x = get_global_id(0);
    const int dstC_y = get_global_id(1);
    const int dstY_x = dstC_x << 1;
    const int dstY_y = dstC_y << 1;

    if (dstY_x + 1 < dstWidth && dstY_y + 1 < dstHeight) {
        const int loadx = dstY_x + cropX;
        const int loady = dstY_y + cropY;
        const TypeIn4 pixSrcRGB00 = rgb_packed_to_rgb3(LOAD_AYUV(src, loadx+0, loady+0), rgb_order) ;
        const TypeIn4 pixSrcRGB01 = rgb_packed_to_rgb3(LOAD_AYUV(src, loadx+1, loady+0), rgb_order);
        const TypeIn4 pixSrcRGB10 = rgb_packed_to_rgb3(LOAD_AYUV(src, loadx+0, loady+1), rgb_order);
        const TypeIn4 pixSrcRGB11 = rgb_packed_to_rgb3(LOAD_AYUV(src, loadx+1, loady+1), rgb_order);
        const float3 yuv00 = conv_rgb_yuv(make_float_rgb3_from_TypeIn4(pixSrcRGB00), matrix);
        const float3 yuv01 = conv_rgb_yuv(make_float_rgb3_from_TypeIn4(pixSrcRGB01), matrix);
        const float3 yuv10 = conv_rgb_yuv(make_float_rgb3_from_TypeIn4(pixSrcRGB10), matrix);
        const float3 yuv11 = conv_rgb_yuv(make_float_rgb3_from_TypeIn4(pixSrcRGB11), matrix);
        int dstPitch = dstPitchY;
        STORE(dstY, dstY_x+0, dstY_y+0, scaleYFloatToPix(yuv00.x));
        STORE(dstY, dstY_x+1, dstY_y+0, scaleYFloatToPix(yuv01.x));
        STORE(dstY, dstY_x+0, dstY_y+1, scaleYFloatToPix(yuv10.x));
        STORE(dstY, dstY_x+1, dstY_y+1, scaleYFloatToPix(yuv11.x));
        const TypeOut pixU = scaleUVFloatToPix((yuv00.y + yuv10.y) * 0.5f);
        const TypeOut pixV = scaleUVFloatToPix((yuv00.z + yuv10.z) * 0.5f);
        dstPitch = dstPitchU;
        STORE(dstU, dstC_x, dstC_y, pixU);
        dstPitch = dstPitchV;
        STORE(dstV, dstC_x, dstC_y, pixV);
    }
}

__kernel void kernel_crop_rgb32_yv12(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstU,
    __write_only image2d_t dstV,
#else
    __global uchar *dstY,
    __global uchar *dstU,
    __global uchar *dstV,
#endif
    int dstPitchY,
    int dstPitchU,
    int dstPitchV,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    crop_rgb_packed_yv12(dstY, dstU, dstV, dstPitchY, dstPitchU, dstPitchV, dstWidth, dstHeight, src, srcPitch, srcWidth, srcHeight, cropX, cropY, matrix, RGB_ORDER_RGB);
}

__kernel void kernel_crop_bgr32_yv12(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstU,
    __write_only image2d_t dstV,
#else
    __global uchar *dstY,
    __global uchar *dstU,
    __global uchar *dstV,
#endif
    int dstPitchY,
    int dstPitchU,
    int dstPitchV,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    crop_rgb_packed_yv12(dstY, dstU, dstV, dstPitchY, dstPitchU, dstPitchV, dstWidth, dstHeight, src, srcPitch, srcWidth, srcHeight, cropX, cropY, matrix, RGB_ORDER_BGR);
}

__kernel void kernel_crop_gbr32_yv12(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstU,
    __write_only image2d_t dstV,
#else
    __global uchar *dstY,
    __global uchar *dstU,
    __global uchar *dstV,
#endif
    int dstPitchY,
    int dstPitchU,
    int dstPitchV,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    crop_rgb_packed_yv12(dstY, dstU, dstV, dstPitchY, dstPitchU, dstPitchV, dstWidth, dstHeight, src, srcPitch, srcWidth, srcHeight, cropX, cropY, matrix, RGB_ORDER_GBR);
}

__kernel void kernel_crop_rbg32_yv12(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstU,
    __write_only image2d_t dstV,
#else
    __global uchar *dstY,
    __global uchar *dstU,
    __global uchar *dstV,
#endif
    int dstPitchY,
    int dstPitchU,
    int dstPitchV,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    crop_rgb_packed_yv12(dstY, dstU, dstV, dstPitchY, dstPitchU, dstPitchV, dstWidth, dstHeight, src, srcPitch, srcWidth, srcHeight, cropX, cropY, matrix, RGB_ORDER_RBG);
}

void crop_rgb_packed_nv12(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstC,
#else
    __global uchar *dstY,
    __global uchar *dstC,
#endif
    int dstPitchY,
    int dstPitchC,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix,
    int rgb_order
) {
    const int dstC_x = get_global_id(0);
    const int dstC_y = get_global_id(1);
    const int dstY_x = dstC_x << 1;
    const int dstY_y = dstC_y << 1;

    if (dstY_x < dstWidth && dstY_y < dstHeight) {
        const int loadx = dstY_x + cropX;
        const int loady = dstY_y + cropY;
        const TypeIn4 pixSrcRGB00 = rgb_packed_to_rgb3(LOAD_AYUV(src, loadx+0, loady+0), rgb_order);
        const TypeIn4 pixSrcRGB01 = rgb_packed_to_rgb3(LOAD_AYUV(src, loadx+1, loady+0), rgb_order);
        const TypeIn4 pixSrcRGB10 = rgb_packed_to_rgb3(LOAD_AYUV(src, loadx+0, loady+1), rgb_order);
        const TypeIn4 pixSrcRGB11 = rgb_packed_to_rgb3(LOAD_AYUV(src, loadx+1, loady+1), rgb_order);
        const float3 yuv00 = conv_rgb_yuv(make_float_rgb3_from_TypeIn4(pixSrcRGB00), matrix);
        const float3 yuv01 = conv_rgb_yuv(make_float_rgb3_from_TypeIn4(pixSrcRGB01), matrix);
        const float3 yuv10 = conv_rgb_yuv(make_float_rgb3_from_TypeIn4(pixSrcRGB10), matrix);
        const float3 yuv11 = conv_rgb_yuv(make_float_rgb3_from_TypeIn4(pixSrcRGB11), matrix);
        int dstPitch = dstPitchY;
        STORE(dstY, dstY_x+0, dstY_y+0, scaleYFloatToPix(yuv00.x));
        STORE(dstY, dstY_x+1, dstY_y+0, scaleYFloatToPix(yuv01.x));
        STORE(dstY, dstY_x+0, dstY_y+1, scaleYFloatToPix(yuv10.x));
        STORE(dstY, dstY_x+1, dstY_y+1, scaleYFloatToPix(yuv11.x));
        const TypeOut pixU = scaleUVFloatToPix((yuv00.y + yuv10.y) * 0.5f);
        const TypeOut pixV = scaleUVFloatToPix((yuv00.z + yuv10.z) * 0.5f);
        dstPitch = dstPitchC;
        STORE_NV12_UV(dstC, dstC_x, dstC_y, pixU, pixV);
    }
}

void crop_rgb32_nv12(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstC,
#else
    __global uchar *dstY,
    __global uchar *dstC,
#endif
    int dstPitchY,
    int dstPitchC,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    crop_rgb_packed_nv12(dstY, dstC, dstPitchY, dstPitchC, dstWidth, dstHeight, src, srcPitch, srcWidth, srcHeight, cropX, cropY, matrix, RGB_ORDER_RGB);
}

void crop_bgr32_nv12(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstC,
#else
    __global uchar *dstY,
    __global uchar *dstC,
#endif
    int dstPitchY,
    int dstPitchC,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    crop_rgb_packed_nv12(dstY, dstC, dstPitchY, dstPitchC, dstWidth, dstHeight, src, srcPitch, srcWidth, srcHeight, cropX, cropY, matrix, RGB_ORDER_BGR);
}

void crop_gbr32_nv12(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstC,
#else
    __global uchar *dstY,
    __global uchar *dstC,
#endif
    int dstPitchY,
    int dstPitchC,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    crop_rgb_packed_nv12(dstY, dstC, dstPitchY, dstPitchC, dstWidth, dstHeight, src, srcPitch, srcWidth, srcHeight, cropX, cropY, matrix, RGB_ORDER_GBR);
}

void crop_rbg32_nv12(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstC,
#else
    __global uchar *dstY,
    __global uchar *dstC,
#endif
    int dstPitchY,
    int dstPitchC,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    crop_rgb_packed_nv12(dstY, dstC, dstPitchY, dstPitchC, dstWidth, dstHeight, src, srcPitch, srcWidth, srcHeight, cropX, cropY, matrix, RGB_ORDER_RBG);
}

__kernel void kernel_crop_rgb_yv12(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstU,
    __write_only image2d_t dstV,
#else
    __global uchar *dstY,
    __global uchar *dstU,
    __global uchar *dstV,
#endif
    int dstPitchY,
    int dstPitchU,
    int dstPitchV,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t srcR,
    __read_only image2d_t srcG,
    __read_only image2d_t srcB,
#else
    __global uchar *srcR,
    __global uchar *srcG,
    __global uchar *srcB,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    const int dstC_x = get_global_id(0);
    const int dstC_y = get_global_id(1);
    const int dstY_x = dstC_x << 1;
    const int dstY_y = dstC_y << 1;

    if (dstY_x+1 < dstWidth && dstY_y+1 < dstHeight) {
        const int loadx = dstY_x + cropX;
        const int loady = dstY_y + cropY;
        const TypeIn pixSrcR00 = LOAD(srcR, loadx+0, loady+0);
        const TypeIn pixSrcR01 = LOAD(srcR, loadx+1, loady+0);
        const TypeIn pixSrcR10 = LOAD(srcR, loadx+0, loady+1);
        const TypeIn pixSrcR11 = LOAD(srcR, loadx+1, loady+1);
        const TypeIn pixSrcG00 = LOAD(srcG, loadx+0, loady+0);
        const TypeIn pixSrcG01 = LOAD(srcG, loadx+1, loady+0);
        const TypeIn pixSrcG10 = LOAD(srcG, loadx+0, loady+1);
        const TypeIn pixSrcG11 = LOAD(srcG, loadx+1, loady+1);
        const TypeIn pixSrcB00 = LOAD(srcB, loadx+0, loady+0);
        const TypeIn pixSrcB01 = LOAD(srcB, loadx+1, loady+0);
        const TypeIn pixSrcB10 = LOAD(srcB, loadx+0, loady+1);
        const TypeIn pixSrcB11 = LOAD(srcB, loadx+1, loady+1);
        const float3 rgb00 = make_float_rgb3(pixSrcR00, pixSrcG00, pixSrcB00);
        const float3 rgb01 = make_float_rgb3(pixSrcR01, pixSrcG01, pixSrcB01);
        const float3 rgb10 = make_float_rgb3(pixSrcR10, pixSrcG10, pixSrcB10);
        const float3 rgb11 = make_float_rgb3(pixSrcR11, pixSrcG11, pixSrcB11);
        const float3 yuv00 = conv_rgb_yuv(rgb00, matrix);
        const float3 yuv01 = conv_rgb_yuv(rgb01, matrix);
        const float3 yuv10 = conv_rgb_yuv(rgb10, matrix);
        const float3 yuv11 = conv_rgb_yuv(rgb11, matrix);
        int dstPitch = dstPitchY;
        STORE(dstY, dstY_x+0, dstY_y+0, scaleYFloatToPix(yuv00.x));
        STORE(dstY, dstY_x+1, dstY_y+0, scaleYFloatToPix(yuv01.x));
        STORE(dstY, dstY_x+0, dstY_y+1, scaleYFloatToPix(yuv10.x));
        STORE(dstY, dstY_x+1, dstY_y+1, scaleYFloatToPix(yuv11.x));
        dstPitch = dstPitchU;
        STORE(dstU, dstC_x, dstC_y, scaleUVFloatToPix((yuv00.y + yuv10.y) * 0.5f));
        dstPitch = dstPitchV;
        STORE(dstV, dstC_x, dstC_y, scaleUVFloatToPix((yuv00.z + yuv10.z) * 0.5f));
    }
}

__kernel void kernel_crop_rgb_nv12(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstC,
#else
    __global uchar *dstY,
    __global uchar *dstC,
#endif
    int dstPitchY,
    int dstPitchC,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t srcR,
    __read_only image2d_t srcG,
    __read_only image2d_t srcB,
#else
    __global uchar *srcR,
    __global uchar *srcG,
    __global uchar *srcB,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    const int dstC_x = get_global_id(0);
    const int dstC_y = get_global_id(1);
    const int dstY_x = dstC_x << 1;
    const int dstY_y = dstC_y << 1;

    if (dstY_x+1 < dstWidth && dstY_y+1 < dstHeight) {
        const int loadx = dstY_x + cropX;
        const int loady = dstY_y + cropY;
        const TypeIn pixSrcR00 = LOAD(srcR, loadx+0, loady+0);
        const TypeIn pixSrcR01 = LOAD(srcR, loadx+1, loady+0);
        const TypeIn pixSrcR10 = LOAD(srcR, loadx+0, loady+1);
        const TypeIn pixSrcR11 = LOAD(srcR, loadx+1, loady+1);
        const TypeIn pixSrcG00 = LOAD(srcG, loadx+0, loady+0);
        const TypeIn pixSrcG01 = LOAD(srcG, loadx+1, loady+0);
        const TypeIn pixSrcG10 = LOAD(srcG, loadx+0, loady+1);
        const TypeIn pixSrcG11 = LOAD(srcG, loadx+1, loady+1);
        const TypeIn pixSrcB00 = LOAD(srcB, loadx+0, loady+0);
        const TypeIn pixSrcB01 = LOAD(srcB, loadx+1, loady+0);
        const TypeIn pixSrcB10 = LOAD(srcB, loadx+0, loady+1);
        const TypeIn pixSrcB11 = LOAD(srcB, loadx+1, loady+1);
        const float3 rgb00 = make_float_rgb3(pixSrcR00, pixSrcG00, pixSrcB00);
        const float3 rgb01 = make_float_rgb3(pixSrcR01, pixSrcG01, pixSrcB01);
        const float3 rgb10 = make_float_rgb3(pixSrcR10, pixSrcG10, pixSrcB10);
        const float3 rgb11 = make_float_rgb3(pixSrcR11, pixSrcG11, pixSrcB11);
        const float3 yuv00 = conv_rgb_yuv(rgb00, matrix);
        const float3 yuv01 = conv_rgb_yuv(rgb01, matrix);
        const float3 yuv10 = conv_rgb_yuv(rgb10, matrix);
        const float3 yuv11 = conv_rgb_yuv(rgb11, matrix);
        int dstPitch = dstPitchY;
        STORE(dstY, dstY_x+0, dstY_y+0, scaleYFloatToPix(yuv00.x));
        STORE(dstY, dstY_x+1, dstY_y+0, scaleYFloatToPix(yuv01.x));
        STORE(dstY, dstY_x+0, dstY_y+1, scaleYFloatToPix(yuv10.x));
        STORE(dstY, dstY_x+1, dstY_y+1, scaleYFloatToPix(yuv11.x));
        dstPitch = dstPitchC;
        const TypeOut pixU = scaleUVFloatToPix((yuv00.y + yuv10.y) * 0.5f);
        const TypeOut pixV = scaleUVFloatToPix((yuv00.z + yuv10.z) * 0.5f);
        STORE_NV12_UV(dstC, dstC_x, dstC_y, pixU, pixV);
    }
}

__kernel void kernel_crop_rgb_yuv444(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstU,
    __write_only image2d_t dstV,
#else
    __global uchar *dstY,
    __global uchar *dstU,
    __global uchar *dstV,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t srcR,
    __read_only image2d_t srcG,
    __read_only image2d_t srcB,
#else
    __global uchar *srcR,
    __global uchar *srcG,
    __global uchar *srcB,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    const int dst_x = get_global_id(0);
    const int dst_y = get_global_id(1);

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int loadx = dst_x + cropX;
        const int loady = dst_y + cropY;
        const TypeIn pixSrcR = LOAD(srcR, loadx, loady);
        const TypeIn pixSrcG = LOAD(srcG, loadx, loady);
        const TypeIn pixSrcB = LOAD(srcB, loadx, loady);
        const float3 rgb = make_float_rgb3(pixSrcR, pixSrcG, pixSrcB);
        const float3 yuv = conv_rgb_yuv(rgb, matrix);
        STORE(dstY, dst_x, dst_y, scaleYFloatToPix(yuv.x));
        STORE(dstU, dst_x, dst_y, scaleUVFloatToPix(yuv.y));
        STORE(dstV, dst_x, dst_y, scaleUVFloatToPix(yuv.z));
    }
}

__kernel void kernel_crop_yv12_rgb(
#if IMAGE_DST
    __write_only image2d_t dstR,
    __write_only image2d_t dstG,
    __write_only image2d_t dstB,
#else
    __global uchar *dstR,
    __global uchar *dstG,
    __global uchar *dstB,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t srcY,
    __read_only image2d_t srcU,
    __read_only image2d_t srcV,
#else
    __global uchar *srcY,
    __global uchar *srcU,
    __global uchar *srcV,
#endif
    int srcPitchY,
    int srcPitchU,
    int srcPitchV,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    const int dstC_x = get_global_id(0);
    const int dstC_y = get_global_id(1);
    const int dstY_x = dstC_x << 1;
    const int dstY_y = dstC_y << 1;

    if (dstY_x < dstWidth && dstY_y < dstHeight) {
        int loadx = dstY_x + cropX;
        int loady = dstY_y + cropY;
        int srcPitch = srcPitchY;
        const TypeIn pixSrcY11 = LOAD(srcY, loadx+0, loady+0);
        const TypeIn pixSrcY12 = LOAD(srcY, loadx+1, loady+0);
        const TypeIn pixSrcY21 = LOAD(srcY, loadx+0, loady+1);
        const TypeIn pixSrcY22 = LOAD(srcY, loadx+1, loady+1);

        const int srcWidthUV  = srcWidth  >> 1;
        const int srcHeightUV = srcHeight >> 1;
        loadx = dstC_x + (cropX>>1);
        loady = dstC_y + (cropY>>1);
        srcPitch = srcPitchU;
        const int pixSrcU01 = LOAD(srcU,     loadx,                  max(loady-1, 0)          );
        const int pixSrcU02 = LOAD(srcU, min(loadx+1, srcWidthUV-1), max(loady-1, 0)          );
        const int pixSrcU11 = LOAD(srcU,     loadx,                      loady                );
        const int pixSrcU12 = LOAD(srcU, min(loadx+1, srcWidthUV-1),     loady                );
        const int pixSrcU21 = LOAD(srcU,     loadx,                  min(loady+1, srcHeightUV-1));
        const int pixSrcU22 = LOAD(srcU, min(loadx+1, srcWidthUV-1), min(loady+1, srcHeightUV-1));

        int pixTmpU11, pixTmpU12, pixTmpU21, pixTmpU22;
        yuv420_yuv444_no_bitdepth_change(
            &pixTmpU11, &pixTmpU12, &pixTmpU21, &pixTmpU22,
            pixSrcU01, pixSrcU02, pixSrcU11, pixSrcU12, pixSrcU21, pixSrcU22
        );

        srcPitch = srcPitchV;
        const int pixSrcV01 = LOAD(srcV,     loadx,                  max(loady-1, 0)          );
        const int pixSrcV02 = LOAD(srcV, min(loadx+1, srcWidthUV-1), max(loady-1, 0)          );
        const int pixSrcV11 = LOAD(srcV,     loadx,                      loady                );
        const int pixSrcV12 = LOAD(srcV, min(loadx+1, srcWidthUV-1),     loady                );
        const int pixSrcV21 = LOAD(srcV,     loadx,                  min(loady+1, srcHeightUV-1));
        const int pixSrcV22 = LOAD(srcV, min(loadx+1, srcWidthUV-1), min(loady+1, srcHeightUV-1));

        int pixTmpV11, pixTmpV12, pixTmpV21, pixTmpV22;
        yuv420_yuv444_no_bitdepth_change(
            &pixTmpV11, &pixTmpV12, &pixTmpV21, &pixTmpV22,
            pixSrcV01, pixSrcV02, pixSrcV11, pixSrcV12, pixSrcV21, pixSrcV22
        );

        const float3 yuv11 = make_float_yuv3(pixSrcY11, pixTmpU11, pixTmpV11);
        const float3 yuv12 = make_float_yuv3(pixSrcY12, pixTmpU12, pixTmpV12);
        const float3 yuv21 = make_float_yuv3(pixSrcY21, pixTmpU21, pixTmpV21);
        const float3 yuv22 = make_float_yuv3(pixSrcY22, pixTmpU22, pixTmpV22);

        const float3 rgb11 = conv_yuv_rgb(yuv11, matrix);
        const float3 rgb12 = conv_yuv_rgb(yuv12, matrix);
        const float3 rgb21 = conv_yuv_rgb(yuv21, matrix);
        const float3 rgb22 = conv_yuv_rgb(yuv22, matrix);

        STORE(dstR, dstY_x+0, dstY_y+0, scaleRGBFloatToPix(rgb11.x));
        STORE(dstR, dstY_x+1, dstY_y+0, scaleRGBFloatToPix(rgb12.x));
        STORE(dstR, dstY_x+0, dstY_y+1, scaleRGBFloatToPix(rgb21.x));
        STORE(dstR, dstY_x+1, dstY_y+1, scaleRGBFloatToPix(rgb22.x));

        STORE(dstG, dstY_x+0, dstY_y+0, scaleRGBFloatToPix(rgb11.y));
        STORE(dstG, dstY_x+1, dstY_y+0, scaleRGBFloatToPix(rgb12.y));
        STORE(dstG, dstY_x+0, dstY_y+1, scaleRGBFloatToPix(rgb21.y));
        STORE(dstG, dstY_x+1, dstY_y+1, scaleRGBFloatToPix(rgb22.y));

        STORE(dstB, dstY_x+0, dstY_y+0, scaleRGBFloatToPix(rgb11.z));
        STORE(dstB, dstY_x+1, dstY_y+0, scaleRGBFloatToPix(rgb12.z));
        STORE(dstB, dstY_x+0, dstY_y+1, scaleRGBFloatToPix(rgb21.z));
        STORE(dstB, dstY_x+1, dstY_y+1, scaleRGBFloatToPix(rgb22.z));
    }
}

__kernel void kernel_crop_nv12_rgb(
#if IMAGE_DST
    __write_only image2d_t dstR,
    __write_only image2d_t dstG,
    __write_only image2d_t dstB,
#else
    __global uchar *dstR,
    __global uchar *dstG,
    __global uchar *dstB,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t srcY,
    __read_only image2d_t srcC,
#else
    __global uchar *srcY,
    __global uchar *srcC,
#endif
    int srcPitchY,
    int srcPitchC,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    const int dstC_x = get_global_id(0);
    const int dstC_y = get_global_id(1);
    const int dstY_x = dstC_x << 1;
    const int dstY_y = dstC_y << 1;

    if (dstY_x < dstWidth && dstY_y < dstHeight) {
        int loadx = dstY_x + cropX;
        int loady = dstY_y + cropY;
        int srcPitch = srcPitchY;
        const TypeIn pixSrcY11 = LOAD(srcY, loadx+0, loady+0);
        const TypeIn pixSrcY12 = LOAD(srcY, loadx+1, loady+0);
        const TypeIn pixSrcY21 = LOAD(srcY, loadx+0, loady+1);
        const TypeIn pixSrcY22 = LOAD(srcY, loadx+1, loady+1);

        const int srcWidthUV  = srcWidth  >> 1;
        const int srcHeightUV = srcHeight >> 1;
        loadx = dstC_x + (cropX>>1);
        loady = dstC_y + (cropY>>1);

        TypeIn pixSrcU01, pixSrcV01, pixSrcU02, pixSrcV02;
        TypeIn pixSrcU11, pixSrcV11, pixSrcU12, pixSrcV12;
        TypeIn pixSrcU21, pixSrcV21, pixSrcU22, pixSrcV22;
        LOAD_NV12_UV(srcC, pixSrcU01, pixSrcV01,     loadx,                  max(loady-1, 0),             0, 0);
        LOAD_NV12_UV(srcC, pixSrcU02, pixSrcV02, min(loadx+1, srcWidthUV-1), max(loady-1, 0),             0, 0);
        LOAD_NV12_UV(srcC, pixSrcU11, pixSrcV11,     loadx,                      loady,                   0, 0);
        LOAD_NV12_UV(srcC, pixSrcU12, pixSrcV12, min(loadx+1, srcWidthUV-1),     loady,                   0, 0);
        LOAD_NV12_UV(srcC, pixSrcU21, pixSrcV21,     loadx,                  min(loady+1, srcHeightUV-1), 0, 0);
        LOAD_NV12_UV(srcC, pixSrcU22, pixSrcV22, min(loadx+1, srcWidthUV-1), min(loady+1, srcHeightUV-1), 0, 0);

        int pixTmpU11, pixTmpU12, pixTmpU21, pixTmpU22;
        yuv420_yuv444_no_bitdepth_change(
            &pixTmpU11, &pixTmpU12, &pixTmpU21, &pixTmpU22,
            pixSrcU01, pixSrcU02, pixSrcU11, pixSrcU12, pixSrcU21, pixSrcU22
        );
        
        int pixTmpV11, pixTmpV12, pixTmpV21, pixTmpV22;
        yuv420_yuv444_no_bitdepth_change(
            &pixTmpV11, &pixTmpV12, &pixTmpV21, &pixTmpV22,
            pixSrcV01, pixSrcV02, pixSrcV11, pixSrcV12, pixSrcV21, pixSrcV22
        );

        const float3 yuv11 = make_float_yuv3(pixSrcY11, pixTmpU11, pixTmpV11);
        const float3 yuv12 = make_float_yuv3(pixSrcY12, pixTmpU12, pixTmpV12);
        const float3 yuv21 = make_float_yuv3(pixSrcY21, pixTmpU21, pixTmpV21);
        const float3 yuv22 = make_float_yuv3(pixSrcY22, pixTmpU22, pixTmpV22);

        const float3 rgb11 = conv_yuv_rgb(yuv11, matrix);
        const float3 rgb12 = conv_yuv_rgb(yuv12, matrix);
        const float3 rgb21 = conv_yuv_rgb(yuv21, matrix);
        const float3 rgb22 = conv_yuv_rgb(yuv22, matrix);

        STORE(dstR, dstY_x+0, dstY_y+0, scaleRGBFloatToPix(rgb11.x));
        STORE(dstR, dstY_x+1, dstY_y+0, scaleRGBFloatToPix(rgb12.x));
        STORE(dstR, dstY_x+0, dstY_y+1, scaleRGBFloatToPix(rgb21.x));
        STORE(dstR, dstY_x+1, dstY_y+1, scaleRGBFloatToPix(rgb22.x));

        STORE(dstG, dstY_x+0, dstY_y+0, scaleRGBFloatToPix(rgb11.y));
        STORE(dstG, dstY_x+1, dstY_y+0, scaleRGBFloatToPix(rgb12.y));
        STORE(dstG, dstY_x+0, dstY_y+1, scaleRGBFloatToPix(rgb21.y));
        STORE(dstG, dstY_x+1, dstY_y+1, scaleRGBFloatToPix(rgb22.y));

        STORE(dstB, dstY_x+0, dstY_y+0, scaleRGBFloatToPix(rgb11.z));
        STORE(dstB, dstY_x+1, dstY_y+0, scaleRGBFloatToPix(rgb12.z));
        STORE(dstB, dstY_x+0, dstY_y+1, scaleRGBFloatToPix(rgb21.z));
        STORE(dstB, dstY_x+1, dstY_y+1, scaleRGBFloatToPix(rgb22.z));
    }
}

__kernel void kernel_crop_yuv444_rgb(
#if IMAGE_DST
    __write_only image2d_t dstR,
    __write_only image2d_t dstG,
    __write_only image2d_t dstB,
#else
    __global uchar *dstR,
    __global uchar *dstG,
    __global uchar *dstB,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t srcY,
    __read_only image2d_t srcU,
    __read_only image2d_t srcV,
#else
    __global uchar *srcY,
    __global uchar *srcU,
    __global uchar *srcV,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY,
    int matrix
) {
    const int dst_x = get_global_id(0);
    const int dst_y = get_global_id(1);

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int loadx = dst_x + cropX;
        const int loady = dst_y + cropY;
        const TypeIn pixSrcY = LOAD(srcY, loadx, loady);
        const TypeIn pixSrcU = LOAD(srcU, loadx, loady);
        const TypeIn pixSrcV = LOAD(srcV, loadx, loady);
        const float3 yuv = make_float_yuv3(pixSrcY, pixSrcU, pixSrcV);
        const float3 rgb = conv_yuv_rgb(yuv, matrix);
        STORE(dstR, dst_x, dst_y, scaleRGBFloatToPix(rgb.x));
        STORE(dstG, dst_x, dst_y, scaleRGBFloatToPix(rgb.y));
        STORE(dstB, dst_x, dst_y, scaleRGBFloatToPix(rgb.z));
    }
}

__kernel void kernel_crop_ayuv_yuv444(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstU,
    __write_only image2d_t dstV,
#else
    __global uchar *dstY,
    __global uchar *dstU,
    __global uchar *dstV,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY
) {
    const int dst_x = get_global_id(0);
    const int dst_y = get_global_id(1);

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int loadx = dst_x + cropX;
        const int loady = dst_y + cropY;
        TypeIn4 pix = LOAD_AYUV(src, loadx, loady); //RGBA = VUYA
        TypeOut pixY = (TypeOut)BIT_DEPTH_CONV(pix.z);
        TypeOut pixU = (TypeOut)BIT_DEPTH_CONV(pix.y);
        TypeOut pixV = (TypeOut)BIT_DEPTH_CONV(pix.x);
        STORE(dstY, dst_x, dst_y, pixY);
        STORE(dstU, dst_x, dst_y, pixU);
        STORE(dstV, dst_x, dst_y, pixV);
    }
}

__kernel void kernel_crop_rgb32_rgb(
#if IMAGE_DST
    __write_only image2d_t dstR,
    __write_only image2d_t dstG,
    __write_only image2d_t dstB,
#else
    __global uchar *dstR,
    __global uchar *dstG,
    __global uchar *dstB,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY
) {
    const int dst_x = get_global_id(0);
    const int dst_y = get_global_id(1);

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int loadx = dst_x + cropX;
        const int loady = dst_y + cropY;
        TypeIn4 pix = rgb_packed_to_rgb3(LOAD_AYUV(src, loadx, loady), RGB_ORDER_RGB);
        TypeOut pixR = (TypeOut)BIT_DEPTH_CONV(pix.x);
        TypeOut pixG = (TypeOut)BIT_DEPTH_CONV(pix.y);
        TypeOut pixB = (TypeOut)BIT_DEPTH_CONV(pix.z);
        STORE(dstR, dst_x, dst_y, pixR);
        STORE(dstG, dst_x, dst_y, pixG);
        STORE(dstB, dst_x, dst_y, pixB);
    }
}

__kernel void kernel_crop_bgr32_rgb(
#if IMAGE_DST
    __write_only image2d_t dstR,
    __write_only image2d_t dstG,
    __write_only image2d_t dstB,
#else
    __global uchar *dstR,
    __global uchar *dstG,
    __global uchar *dstB,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY
) {
    const int dst_x = get_global_id(0);
    const int dst_y = get_global_id(1);

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int loadx = dst_x + cropX;
        const int loady = dst_y + cropY;
        TypeIn4 pix = rgb_packed_to_rgb3(LOAD_AYUV(src, loadx, loady), RGB_ORDER_BGR);
        TypeOut pixR = (TypeOut)BIT_DEPTH_CONV(pix.x);
        TypeOut pixG = (TypeOut)BIT_DEPTH_CONV(pix.y);
        TypeOut pixB = (TypeOut)BIT_DEPTH_CONV(pix.z);
        STORE(dstR, dst_x, dst_y, pixR);
        STORE(dstG, dst_x, dst_y, pixG);
        STORE(dstB, dst_x, dst_y, pixB);
    }
}

__kernel void kernel_crop_gbr32_rgb(
#if IMAGE_DST
    __write_only image2d_t dstR,
    __write_only image2d_t dstG,
    __write_only image2d_t dstB,
#else
    __global uchar *dstR,
    __global uchar *dstG,
    __global uchar *dstB,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY
) {
    const int dst_x = get_global_id(0);
    const int dst_y = get_global_id(1);

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int loadx = dst_x + cropX;
        const int loady = dst_y + cropY;
        TypeIn4 pix = rgb_packed_to_rgb3(LOAD_AYUV(src, loadx, loady), RGB_ORDER_GBR);
        TypeOut pixR = (TypeOut)BIT_DEPTH_CONV(pix.x);
        TypeOut pixG = (TypeOut)BIT_DEPTH_CONV(pix.y);
        TypeOut pixB = (TypeOut)BIT_DEPTH_CONV(pix.z);
        STORE(dstR, dst_x, dst_y, pixR);
        STORE(dstG, dst_x, dst_y, pixG);
        STORE(dstB, dst_x, dst_y, pixB);
    }
}

__kernel void kernel_crop_rbg32_rgb(
#if IMAGE_DST
    __write_only image2d_t dstR,
    __write_only image2d_t dstG,
    __write_only image2d_t dstB,
#else
    __global uchar *dstR,
    __global uchar *dstG,
    __global uchar *dstB,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY
) {
    const int dst_x = get_global_id(0);
    const int dst_y = get_global_id(1);

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int loadx = dst_x + cropX;
        const int loady = dst_y + cropY;
        TypeIn4 pix = rgb_packed_to_rgb3(LOAD_AYUV(src, loadx, loady), RGB_ORDER_RBG);
        TypeOut pixR = (TypeOut)BIT_DEPTH_CONV(pix.x);
        TypeOut pixG = (TypeOut)BIT_DEPTH_CONV(pix.y);
        TypeOut pixB = (TypeOut)BIT_DEPTH_CONV(pix.z);
        STORE(dstR, dst_x, dst_y, pixR);
        STORE(dstG, dst_x, dst_y, pixG);
        STORE(dstB, dst_x, dst_y, pixB);
    }
}

__kernel void kernel_crop_ayuv_yv12(
#if IMAGE_DST
    __write_only image2d_t dstY,
    __write_only image2d_t dstU,
    __write_only image2d_t dstV,
#else
    __global uchar *dstY,
    __global uchar *dstU,
    __global uchar *dstV,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t src,
#else
    __global uchar *src,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY
) {
    const int dst_x_C = get_global_id(0);
    const int dst_y_C = get_global_id(1);
    const int dst_x_Y = dst_x_C << 1;
    const int dst_y_Y = dst_y_C << 1;

    if (dst_x_Y < dstWidth && dst_y_Y < dstHeight) {
        const int src_x = dst_x_Y;
        const int src_y = dst_y_Y;
        const int loadx = src_x + cropX;
        const int loady = src_y + cropY;
        const TypeIn4 pixSrc00 = LOAD_AYUV(src, loadx+0, loady+0);
        const TypeIn4 pixSrc01 = LOAD_AYUV(src, loadx+1, loady+0);
        const TypeIn4 pixSrc10 = LOAD_AYUV(src, loadx+0, loady+1);
        const TypeIn4 pixSrc11 = LOAD_AYUV(src, loadx+1, loady+1);
        TypeOut pixY00 = (TypeOut)BIT_DEPTH_CONV(pixSrc00.z);
        TypeOut pixY01 = (TypeOut)BIT_DEPTH_CONV(pixSrc01.z);
        TypeOut pixY10 = (TypeOut)BIT_DEPTH_CONV(pixSrc10.z);
        TypeOut pixY11 = (TypeOut)BIT_DEPTH_CONV(pixSrc11.z);
        TypeOut pixU   = (TypeOut)BIT_DEPTH_CONV_AVG(pixSrc00.y, pixSrc10.y);
        TypeOut pixV   = (TypeOut)BIT_DEPTH_CONV_AVG(pixSrc00.x, pixSrc10.x);
        STORE(dstY, dst_x_Y+0, dst_y_Y+0, pixY00);
        STORE(dstY, dst_x_Y+1, dst_y_Y+0, pixY01);
        STORE(dstY, dst_x_Y+0, dst_y_Y+1, pixY10);
        STORE(dstY, dst_x_Y+1, dst_y_Y+1, pixY11);
        STORE(dstU, dst_x_C,   dst_y_C,   pixU);
        STORE(dstV, dst_x_C,   dst_y_C,   pixV);
    }
}

__kernel void kernel_crop_yuv444_ayuv(
#if IMAGE_DST
    __write_only image2d_t dst,
#else
    __global uchar *dst,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t srcY,
    __read_only image2d_t srcU,
    __read_only image2d_t srcV,
#else
    __global uchar *srcY,
    __global uchar *srcU,
    __global uchar *srcV,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY
) {
    const int dst_x = get_global_id(0);
    const int dst_y = get_global_id(1);

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int loadx = dst_x + cropX;
        const int loady = dst_y + cropY;
        const int pixY = LOAD(srcY, loadx, loady);
        const int pixU = LOAD(srcU, loadx, loady);
        const int pixV = LOAD(srcV, loadx, loady);
        TypeOut4 pix;
        pix.w = 0;
        pix.z = BIT_DEPTH_CONV(pixY);
        pix.y = BIT_DEPTH_CONV(pixU);
        pix.x = BIT_DEPTH_CONV(pixV);
        STORE_AYUV(dst, dst_x, dst_y, pix);
    }
}

__kernel void kernel_crop_rgb_rgb32(
#if IMAGE_DST
    __write_only image2d_t dst,
#else
    __global uchar *dst,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t srcR,
    __read_only image2d_t srcG,
    __read_only image2d_t srcB,
#else
    __global uchar *srcR,
    __global uchar *srcG,
    __global uchar *srcB,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY
) {
    const int dst_x = get_global_id(0);
    const int dst_y = get_global_id(1);

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int loadx = dst_x + cropX;
        const int loady = dst_y + cropY;
        const int pixR = LOAD(srcR, loadx, loady);
        const int pixG = LOAD(srcG, loadx, loady);
        const int pixB = LOAD(srcB, loadx, loady);
        TypeOut4 pixRGB;
        pixRGB.x = BIT_DEPTH_CONV(pixR);
        pixRGB.y = BIT_DEPTH_CONV(pixG);
        pixRGB.z = BIT_DEPTH_CONV(pixB);
        pixRGB.w = 0;
        TypeOut4 pix = rgb3_to_rgb_packed(pixRGB, RGB_ORDER_RGB);
        STORE_AYUV(dst, dst_x, dst_y, pix);
    }
}

__kernel void kernel_crop_rgb_bgr32(
#if IMAGE_DST
    __write_only image2d_t dst,
#else
    __global uchar *dst,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t srcR,
    __read_only image2d_t srcG,
    __read_only image2d_t srcB,
#else
    __global uchar *srcR,
    __global uchar *srcG,
    __global uchar *srcB,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY
) {
    const int dst_x = get_global_id(0);
    const int dst_y = get_global_id(1);

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int loadx = dst_x + cropX;
        const int loady = dst_y + cropY;
        const int pixR = LOAD(srcR, loadx, loady);
        const int pixG = LOAD(srcG, loadx, loady);
        const int pixB = LOAD(srcB, loadx, loady);
        TypeOut4 pixRGB;
        pixRGB.x = BIT_DEPTH_CONV(pixR);
        pixRGB.y = BIT_DEPTH_CONV(pixG);
        pixRGB.z = BIT_DEPTH_CONV(pixB);
        pixRGB.w = 0;
        TypeOut4 pix = rgb3_to_rgb_packed(pixRGB, RGB_ORDER_BGR);
        STORE_AYUV(dst, dst_x, dst_y, pix);
    }
}

__kernel void kernel_crop_rgb_gbr32(
#if IMAGE_DST
    __write_only image2d_t dst,
#else
    __global uchar *dst,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t srcR,
    __read_only image2d_t srcG,
    __read_only image2d_t srcB,
#else
    __global uchar *srcR,
    __global uchar *srcG,
    __global uchar *srcB,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY
) {
    const int dst_x = get_global_id(0);
    const int dst_y = get_global_id(1);

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int loadx = dst_x + cropX;
        const int loady = dst_y + cropY;
        const int pixR = LOAD(srcR, loadx, loady);
        const int pixG = LOAD(srcG, loadx, loady);
        const int pixB = LOAD(srcB, loadx, loady);
        TypeOut4 pixRGB;
        pixRGB.x = BIT_DEPTH_CONV(pixR);
        pixRGB.y = BIT_DEPTH_CONV(pixG);
        pixRGB.z = BIT_DEPTH_CONV(pixB);
        pixRGB.w = 0;
        TypeOut4 pix = rgb3_to_rgb_packed(pixRGB, RGB_ORDER_GBR);
        STORE_AYUV(dst, dst_x, dst_y, pix);
    }
}

__kernel void kernel_crop_rgb_rbg32(
#if IMAGE_DST
    __write_only image2d_t dst,
#else
    __global uchar *dst,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t srcR,
    __read_only image2d_t srcG,
    __read_only image2d_t srcB,
#else
    __global uchar *srcR,
    __global uchar *srcG,
    __global uchar *srcB,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY
) {
    const int dst_x = get_global_id(0);
    const int dst_y = get_global_id(1);

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int loadx = dst_x + cropX;
        const int loady = dst_y + cropY;
        const int pixR = LOAD(srcR, loadx, loady);
        const int pixG = LOAD(srcG, loadx, loady);
        const int pixB = LOAD(srcB, loadx, loady);
        TypeOut4 pixRGB;
        pixRGB.x = BIT_DEPTH_CONV(pixR);
        pixRGB.y = BIT_DEPTH_CONV(pixG);
        pixRGB.z = BIT_DEPTH_CONV(pixB);
        pixRGB.w = 0;
        TypeOut4 pix = rgb3_to_rgb_packed(pixRGB, RGB_ORDER_RBG);
        STORE_AYUV(dst, dst_x, dst_y, pix);
    }
}

__kernel void kernel_crop_yv12_ayuv(
#if IMAGE_DST
    __write_only image2d_t dst,
#else
    __global uchar *dst,
#endif
    int dstPitch,
    int dstWidth,
    int dstHeight,
#if IMAGE_SRC
    __read_only image2d_t srcY,
    __read_only image2d_t srcU,
    __read_only image2d_t srcV,
#else
    __global uchar *srcY,
    __global uchar *srcU,
    __global uchar *srcV,
#endif
    int srcPitch,
    int srcWidth,
    int srcHeight,
    int cropX,
    int cropY
) {
    const int src_C_x = get_global_id(0);
    const int src_C_y = get_global_id(1);
    const int dst_x = src_C_x << 1;
    const int dst_y = src_C_y << 1;

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int load_C_x = src_C_x + (cropX>>1);
        const int load_C_y = src_C_y + (cropY>>1);

        const int pixSrcU01 = LOAD(srcU,     load_C_x,                max(load_C_y-1, 0)          );
        const int pixSrcU02 = LOAD(srcU, min(load_C_x+1, srcWidth-1), max(load_C_y-1, 0)          );
        const int pixSrcU11 = LOAD(srcU,     load_C_x,                    load_C_y                );
        const int pixSrcU12 = LOAD(srcU, min(load_C_x+1, srcWidth-1),     load_C_y                );
        const int pixSrcU21 = LOAD(srcU,     load_C_x,                min(load_C_y+1, srcHeight-1));
        const int pixSrcU22 = LOAD(srcU, min(load_C_x+1, srcWidth-1), min(load_C_y+1, srcHeight-1));

        const int pixSrcV01 = LOAD(srcV,     load_C_x,                max(load_C_y-1, 0)          );
        const int pixSrcV02 = LOAD(srcV, min(load_C_x+1, srcWidth-1), max(load_C_y-1, 0)          );
        const int pixSrcV11 = LOAD(srcV,     load_C_x,                    load_C_y                );
        const int pixSrcV12 = LOAD(srcV, min(load_C_x+1, srcWidth-1),     load_C_y                );
        const int pixSrcV21 = LOAD(srcV,     load_C_x,                min(load_C_y+1, srcHeight-1));
        const int pixSrcV22 = LOAD(srcV, min(load_C_x+1, srcWidth-1), min(load_C_y+1, srcHeight-1));

        int pixDstU11, pixDstU12, pixDstU21, pixDstU22;
        conv_c_yuv420_yuv444_internal(&pixDstU11, &pixDstU12, &pixDstU21, &pixDstU22, pixSrcU01, pixSrcU02, pixSrcU11, pixSrcU12, pixSrcU21, pixSrcU22);
        
        int pixDstV11, pixDstV12, pixDstV21, pixDstV22;
        conv_c_yuv420_yuv444_internal(&pixDstU11, &pixDstU12, &pixDstU21, &pixDstU22, pixSrcU01, pixSrcU02, pixSrcU11, pixSrcU12, pixSrcU21, pixSrcU22);
        
        const int load_Y_x = load_C_x << 1;
        const int load_Y_y = load_C_y << 1;

        const int pixSrcY11 = LOAD(srcY, load_Y_x+0, load_Y_y+0);
        const int pixSrcY12 = LOAD(srcY, load_Y_x+1, load_Y_y+0);
        const int pixSrcY21 = LOAD(srcY, load_Y_x+0, load_Y_y+1);
        const int pixSrcY22 = LOAD(srcY, load_Y_x+1, load_Y_y+1);

        TypeOut4 pix11, pix12, pix21, pix22;

        pix11.w = 0;
        pix11.z = BIT_DEPTH_CONV(pixSrcY11);
        pix11.y = pixDstU11;
        pix11.x = pixDstV11;

        pix12.w = 0;
        pix12.z = BIT_DEPTH_CONV(pixSrcY12);
        pix12.y = pixDstU12;
        pix12.x = pixDstV12;

        pix21.w = 0;
        pix21.z = BIT_DEPTH_CONV(pixSrcY21);
        pix21.y = pixDstU21;
        pix21.x = pixDstV21;

        pix22.w = 0;
        pix22.z = BIT_DEPTH_CONV(pixSrcY22);
        pix22.y = pixDstU22;
        pix22.x = pixDstV22;

        STORE_AYUV(dst, dst_x+0, dst_y+0, pix11);
        STORE_AYUV(dst, dst_x+1, dst_y+0, pix12);
        STORE_AYUV(dst, dst_x+0, dst_y+1, pix21);
        STORE_AYUV(dst, dst_x+1, dst_y+1, pix22);
    }
}

__kernel void kernel_separate_fields(
    __global uchar *dst0,
    __global uchar *dst1,
    int dstPitch,
    __global uchar *src,
    int srcPitch,
    int width,
    int height_field
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < width && y < height_field) {
        TypeIn pixSrc0 = LOAD_BUF(src, x, y*2+0);
        TypeIn pixSrc1 = LOAD_BUF(src, x, y*2+1);
        STORE_BUF(dst0, x, y, pixSrc0);
        STORE_BUF(dst1, x, y, pixSrc1);
    }
}

__kernel void kernel_merge_fields(
    __global uchar *dst,
    int dstPitch,
    __global uchar *src0,
    __global uchar *src1,
    int srcPitch,
    int width,
    int height_field
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < width && y < height_field) {
        TypeIn pixSrc0 = LOAD_BUF(src0, x, y);
        TypeIn pixSrc1 = LOAD_BUF(src1, x, y);
        STORE_BUF(dst, x, y*2+0, pixSrc0);
        STORE_BUF(dst, x, y*2+1, pixSrc1);
    }
}

__kernel void kernel_set_plane(
#if IMAGE_DST
    __write_only image2d_t dst,
#else
    __global uchar *dst,
#endif
    int dstPitch,
    int width,
    int height,
    int cropX,
    int cropY,
    int value
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < width && y < height) {
        STORE(dst, x + cropX, y + cropY, value);
    }
}

__kernel void kernel_set_plane_nv12(
#if IMAGE_DST
    __write_only image2d_t dst,
#else
    __global uchar *dst,
#endif
    int dstPitch,
    int uvWidth,
    int uvHeight,
    int cropX,
    int cropY,
    int valueU,
    int valueV
) {
    const int uv_x = get_global_id(0);
    const int uv_y = get_global_id(1);
    if (uv_x < uvWidth && uv_y < uvHeight) {
        TypeOut pixDstU = valueU;
        TypeOut pixDstV = valueV;
        STORE_NV12_UV(dst, uv_x + cropX, uv_y + cropY, pixDstU, pixDstV);
    }
}

