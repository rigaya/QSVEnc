// TypeIn
// TypeOut
// MEM_TYPE_SRC
// MEM_TYPE_DST
// in_bit_depth
// out_bit_depth

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

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

#define LOAD_IMG(src, ix, iy) (TypeIn)(read_imageui((src), sampler, (int2)((ix), (iy))).x)
#define LOAD_IMG_NV12_UV(src, src_u, src_v, ix, iy, cropX, cropY) { \
    uint4 ret = read_imageui((src), sampler, (int2)((ix) + ((cropX)>>1), (iy) + ((cropY)>>1))); \
    (src_u) = (TypeIn)ret.x; \
    (src_v) = (TypeIn)ret.y; \
}
#define LOAD_BUF(src, ix, iy) *(__global TypeIn *)(&(src)[(iy) * srcPitch + (ix) * sizeof(TypeIn)])
#define LOAD_BUF_NV12_UV(src, src_u, src_v, ix, iy, cropX, cropY) { \
    (src_u) = LOAD((src), ((ix)<<1) + 0 + (cropX), (iy) + ((cropY)>>1)); \
    (src_v) = LOAD((src), ((ix)<<1) + 1 + (cropX), (iy) + ((cropY)>>1)); \
}

#define LOAD_IMG_NORM(src, ix, iy) (TypeIn)(read_imagef((src), sampler, (int2)((ix), (iy))).x * NORM_SCALE_IN + 0.5f)
#define LOAD_IMG_NORM_NV12_UV(src, src_u, src_v, ix, iy, cropX, cropY) { \
    float4 ret = read_imagef((src), sampler, (int2)((ix) + ((cropX)>>1), (iy) + ((cropY)>>1))); \
    (src_u) = (TypeIn)(ret.x * NORM_SCALE_IN + 0.5f); \
    (src_v) = (TypeIn)(ret.y * NORM_SCALE_IN + 0.5f); \
}

#if MEM_TYPE_SRC == RGY_MEM_TYPE_GPU_IMAGE
#define IMAGE_SRC    1
#define LOAD         LOAD_IMG
#define LOAD_NV12_UV LOAD_IMG_NV12_UV
#elif MEM_TYPE_SRC == RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED
#define IMAGE_SRC    1
#define LOAD         LOAD_IMG_NORM
#define LOAD_NV12_UV LOAD_IMG_NORM_NV12_UV
#else
#define IMAGE_SRC    0
#define LOAD         LOAD_BUF
#define LOAD_NV12_UV LOAD_BUF_NV12_UV
#endif

#define STORE_IMG(dst, ix, iy, val) write_imageui((dst), (int2)((ix), (iy)), (val))
#define STORE_IMG_NV12_UV(dst, ix, iy, val_u, val_v) { \
    uint4 val = (uint4)(val_u, val_v, val_v, val_v); \
    write_imageui((dst), (int2)((ix), (iy)), (val)); \
}

#define STORE_IMG_NORM(dst, ix, iy, val) write_imagef((dst), (int2)((ix), (iy)), (val * NORM_SCALE_OUT))
#define STORE_IMG_NORM_NV12_UV(dst, ix, iy, val_u, val_v) { \
    float4 val = (float4)(val_u * NORM_SCALE_OUT, val_v * NORM_SCALE_OUT, val_v * NORM_SCALE_OUT, val_v * NORM_SCALE_OUT); \
    write_imagef((dst), (int2)((ix), (iy)), (val)); \
}
#define STORE_BUF(dst, ix, iy, val)  { \
    __global TypeOut *ptr = (__global TypeOut *)(&(dst)[(iy) * dstPitch + (ix) * sizeof(TypeOut)]); \
    ptr[0] = (TypeOut)(val); \
}
#define STORE_BUF_NV12_UV(dst, ix, iy, val_u, val_v) { \
    STORE(dst, ((ix) << 1) + 0, (iy), val_u); \
    STORE(dst, ((ix) << 1) + 1, (iy), val_v); \
}
#if MEM_TYPE_DST == RGY_MEM_TYPE_GPU_IMAGE
#define IMAGE_DST     1
#define STORE         STORE_IMG
#define STORE_NV12_UV STORE_IMG_NV12_UV
#elif MEM_TYPE_DST == RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED
#define IMAGE_DST     1
#define STORE         STORE_IMG_NORM
#define STORE_NV12_UV STORE_IMG_NORM_NV12_UV
#else
#define IMAGE_DST     0
#define STORE         STORE_BUF
#define STORE_NV12_UV STORE_BUF_NV12_UV
#endif


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
    pixSrc02 = (pixSrc01 + pixSrc02 + 1) >> 1;
    pixSrc12 = (pixSrc11 + pixSrc12 + 1) >> 1;
    pixSrc22 = (pixSrc21 + pixSrc22 + 1) >> 1;

    TypeOut pixDst11 = BIT_DEPTH_CONV_3x1_AVG(pixSrc11, pixSrc01);
    TypeOut pixDst12 = BIT_DEPTH_CONV_3x1_AVG(pixSrc12, pixSrc02);
    TypeOut pixDst21 = BIT_DEPTH_CONV_7x1_AVG(pixSrc11, pixSrc21);
    TypeOut pixDst22 = BIT_DEPTH_CONV_7x1_AVG(pixSrc12, pixSrc22);

    STORE(dst, dst_x+0, dst_y+0, pixDst11);
    STORE(dst, dst_x+1, dst_y+0, pixDst12);
    STORE(dst, dst_x+0, dst_y+1, pixDst21);
    STORE(dst, dst_x+1, dst_y+1, pixDst22);
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
    int cropX,
    int cropY
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
    int cropX,
    int cropY
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
    int cropX,
    int cropY
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
        LOAD_NV12_UV(src, pixSrcU01, pixSrcV01,     loadx,              max(loady-1, 0),         0, 0);
        LOAD_NV12_UV(src, pixSrcU02, pixSrcV02, min(loadx+1, srcWidth), max(loady-1, 0),         0, 0);
        LOAD_NV12_UV(src, pixSrcU11, pixSrcV11,     loadx,                  loady,               0, 0);
        LOAD_NV12_UV(src, pixSrcU12, pixSrcV12, min(loadx+1, srcWidth),     loady,               0, 0);
        LOAD_NV12_UV(src, pixSrcU21, pixSrcV21,     loadx,              min(loady+1, srcHeight), 0, 0);
        LOAD_NV12_UV(src, pixSrcU22, pixSrcV22, min(loadx+1, srcWidth), min(loady+1, srcHeight), 0, 0);

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
    int cropX,
    int cropY
) {
    const int dst_x = get_global_id(0);
    const int dst_y = get_global_id(1);

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int src_x = dst_x << 1;
        const int src_y = dst_y << 1;
        const int loadx = src_x + cropX;
        const int loady = src_y + cropY;
        const int pixSrcU00 = LOAD(srcU, loadx+0, loady+0);
        const int pixSrcU10 = LOAD(srcU, loadx+0, loady+1);
        const int pixSrcV00 = LOAD(srcV, loadx+0, loady+0);
        const int pixSrcV10 = LOAD(srcV, loadx+0, loady+1);
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
    int cropX,
    int cropY
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
    int cropX,
    int cropY
) {
    const int src_x = get_global_id(0);
    const int src_y = get_global_id(1);
    const int dst_x = src_x << 1;
    const int dst_y = src_y << 1;

    if (dst_x < dstWidth && dst_y < dstHeight) {
        const int loadx = src_x + (cropX>>1);
        const int loady = src_y + (cropY>>1);
        const int pixSrc01 = LOAD(src,     loadx,              max(loady-1, 0)        );
        const int pixSrc02 = LOAD(src, min(loadx+1, srcWidth), max(loady-1, 0)        );
        const int pixSrc11 = LOAD(src,     loadx,                  loady              );
        const int pixSrc12 = LOAD(src, min(loadx+1, srcWidth),     loady              );
        const int pixSrc21 = LOAD(src,     loadx,              min(loady+1, srcHeight));
        const int pixSrc22 = LOAD(src, min(loadx+1, srcWidth), min(loady+1, srcHeight));

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
    int cropX,
    int cropY
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