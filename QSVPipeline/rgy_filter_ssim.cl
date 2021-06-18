// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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

//BIT_DEPTH
//SSIM_BLOCK_X
//SSIM_BLOCK_Y

#if BIT_DEPTH > 8
typedef ushort         DATA;
typedef ushort4        DATA4;
#else
typedef uchar         DATA;
typedef uchar4        DATA4;
#endif

float ssim_end1x(long s1, long s2, long ss, long s12) {
    const long max = ((1 << BIT_DEPTH) - 1);
    const long ssim_c1 = (long)(0.01 * 0.01 * max * max * 64.0 + 0.5);
    const long ssim_c2 = (long)(0.03 * 0.03 * max * max * 64.0 * 63.0 + 0.5);

    const long vars = ss * 64 - s1 * s1 - s2 * s2;
    const long covar = s12 * 64 - s1 * s2;

    float f = ((float)(2 * s1 * s2 + ssim_c1) * (float)(2 * covar + ssim_c2))
        / ((float)(s1 * s1 + s2 * s2 + ssim_c1) * (float)(vars + ssim_c2));
    return f;
}

void func_ssim_pix(long4 *ss, int a, int b) {
    ss->x += a;
    ss->y += b;
    ss->z += a * a;
    ss->z += b * b;
    ss->w += a * b;
}

long4 func_ssim_block(
    __global const uchar *p0, const int p0_pitch,
    __global const uchar *p1, const int p1_pitch) {
    long4 ss = (long4)0;
    #pragma unroll
    for (int y = 0; y < 4; y++, p0 += p0_pitch, p1 += p1_pitch) {
        DATA4 pix0 = *(__global const DATA4 *)p0;
        DATA4 pix1 = *(__global const DATA4 *)p1;
        func_ssim_pix(&ss, pix0.x, pix1.x);
        func_ssim_pix(&ss, pix0.y, pix1.y);
        func_ssim_pix(&ss, pix0.z, pix1.z);
        func_ssim_pix(&ss, pix0.w, pix1.w);
    }
    return ss;
}

__kernel void kernel_ssim(
    __global const uchar *p0, const int p0_pitch,
    __global const uchar *p1, const int p1_pitch,
    const int width, const int height,
    __global float *restrict pDst) {
    const int lx = get_local_id(0); //スレッド数=SSIM_BLOCK_X
    int ly = get_local_id(1);       //スレッド数=SSIM_BLOCK_Y
    const int blockoffset_x = get_group_id(0) * SSIM_BLOCK_X;
    const int blockoffset_y = get_group_id(1) * SSIM_BLOCK_Y;
    const int imgx = (blockoffset_x + lx) * 4;
    int imgy = (blockoffset_y + ly) * 4;

    __local long4 stmp[SSIM_BLOCK_Y + 1][SSIM_BLOCK_X + 1];
#define STMP(x, y) ((stmp)[(y)][x])
    float ssim = 0.0f;
#if 1
    if (ly == 0) {
        if (imgx < width) {
            STMP(lx, ly) = func_ssim_block(
                p0 + imgy * p0_pitch + imgx * sizeof(DATA), p0_pitch,
                p1 + imgy * p1_pitch + imgx * sizeof(DATA), p1_pitch);
        } else {
            STMP(lx, ly) = (long4)0;
        }
        if (lx == 0) {
            const int sx = SSIM_BLOCK_X;
            const int sy = 0;
            const int gx = (blockoffset_x + sx) * 4;
            const int gy = (blockoffset_y + sy) * 4;
            if (gx < width && gy < height) {
                STMP(sx, sy) = func_ssim_block(
                    p0 + gy * p0_pitch + gx * sizeof(DATA), p0_pitch,
                    p1 + gy * p1_pitch + gx * sizeof(DATA), p1_pitch);
            } else {
                STMP(sx, sy) = (long4)0;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    imgy += 4;

    if (imgx < width && imgy < height) {
        STMP(lx, ly + 1) = func_ssim_block(
            p0 + imgy * p0_pitch + imgx * sizeof(DATA), p0_pitch,
            p1 + imgy * p1_pitch + imgx * sizeof(DATA), p1_pitch);
    } else {
        STMP(lx, ly + 1) = (long4)0;
    }
    if (ly == 0 && lx < SSIM_BLOCK_Y) {
        const int sx = SSIM_BLOCK_X;
        const int sy = lx + 1;
        const int gx = (blockoffset_x + sx) * 4;
        const int gy = (blockoffset_y + sy) * 4;
        if (gx < width && gy < height) {
            STMP(sx, sy) = func_ssim_block(
                p0 + gy * p0_pitch + gx * sizeof(DATA), p0_pitch,
                p1 + gy * p1_pitch + gx * sizeof(DATA), p1_pitch);
        } else {
            STMP(sx, sy) = (long4)0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (imgx < (width - 4) && imgy < height) {
        long4 sx0y0 = STMP(lx + 0, ly + 0);
        long4 sx1y0 = STMP(lx + 1, ly + 0);
        long4 sx0y1 = STMP(lx + 0, ly + 1);
        long4 sx1y1 = STMP(lx + 1, ly + 1);
        ssim += ssim_end1x(
            sx0y0.x + sx1y0.x + sx0y1.x + sx1y1.x,
            sx0y0.y + sx1y0.y + sx0y1.y + sx1y1.y,
            sx0y0.z + sx1y0.z + sx0y1.z + sx1y1.z,
            sx0y0.w + sx1y0.w + sx0y1.w + sx1y1.w);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    if (imgx < (width - 4) && imgy < (height - 4)) {
        long4 sx0y0 = func_ssim_block(
            p0 + (imgy + 0) * p0_pitch + (imgx + 0) * sizeof(DATA), p0_pitch,
            p1 + (imgy + 0) * p1_pitch + (imgx + 0) * sizeof(DATA), p1_pitch);
        long4 sx1y0 = func_ssim_block(
            p0 + (imgy + 0) * p0_pitch + (imgx + 4) * sizeof(DATA), p0_pitch,
            p1 + (imgy + 0) * p1_pitch + (imgx + 4) * sizeof(DATA), p1_pitch);
        long4 sx0y1 = func_ssim_block(
            p0 + (imgy + 4) * p0_pitch + (imgx + 0) * sizeof(DATA), p0_pitch,
            p1 + (imgy + 4) * p1_pitch + (imgx + 0) * sizeof(DATA), p1_pitch);
        long4 sx1y1 = func_ssim_block(
            p0 + (imgy + 4) * p0_pitch + (imgx + 4) * sizeof(DATA), p0_pitch,
            p1 + (imgy + 4) * p1_pitch + (imgx + 4) * sizeof(DATA), p1_pitch);
        ssim += ssim_end1x(
            sx0y0.x + sx1y0.x + sx0y1.x + sx1y1.x,
            sx0y0.y + sx1y0.y + sx0y1.y + sx1y1.y,
            sx0y0.z + sx1y0.z + sx0y1.z + sx1y1.z,
            sx0y0.w + sx1y0.w + sx0y1.w + sx1y1.w);
    }
#endif

    __local float *ptr_reduction = (__local float *)stmp;
    const int lid = get_local_id(1) * SSIM_BLOCK_X + get_local_id(0);
    ptr_reduction[lid] = ssim;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = SSIM_BLOCK_Y * SSIM_BLOCK_X >> 1; offset > 0; offset >>= 1) {
        if (lid < offset) {
            ptr_reduction[lid] += ptr_reduction[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        const int gid = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        pDst[gid] = ptr_reduction[0];
    }
}

int func_psnr_pix(int a, int b) {
    int i = a - b;
    return i * i;
}

__kernel void kernel_psnr(
    __global const uchar *p0, const int p0_pitch,
    __global const uchar *p1, const int p1_pitch,
    const int width, const int height,
    __global int *restrict pDst) {
    const int lx = get_local_id(0); //スレッド数=SSIM_BLOCK_X
    const int ly = get_local_id(1); //スレッド数=SSIM_BLOCK_Y
    const int blockoffset_x = get_group_id(0) * SSIM_BLOCK_X;
    const int blockoffset_y = get_group_id(1) * SSIM_BLOCK_Y;
    const int imgx = (blockoffset_x + lx) * 4;
    const int imgy = (blockoffset_y + ly);

    int psnr = 0;
    if (imgx < width && imgy < height) {
        p0 += imgy * p0_pitch + imgx * sizeof(DATA);
        p1 += imgy * p1_pitch + imgx * sizeof(DATA);
        DATA4 pix0 = *(__global const DATA4 *)p0;
        DATA4 pix1 = *(__global const DATA4 *)p1;
        psnr += func_psnr_pix(pix0.x, pix1.x);
        if (imgx + 1 < width) psnr += func_psnr_pix(pix0.y, pix1.y);
        if (imgx + 2 < width) psnr += func_psnr_pix(pix0.z, pix1.z);
        if (imgx + 3 < width) psnr += func_psnr_pix(pix0.w, pix1.w);
    }

    __local int tmp[SSIM_BLOCK_X * SSIM_BLOCK_Y / 2];

    __local int *ptr_reduction = tmp;
    const int lid = get_local_id(1) * SSIM_BLOCK_X + get_local_id(0);
    ptr_reduction[lid] = psnr;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = SSIM_BLOCK_Y * SSIM_BLOCK_X >> 1; offset > 0; offset >>= 1) {
        if (lid < offset) {
            ptr_reduction[lid] += ptr_reduction[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        const int gid = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        pDst[gid] = ptr_reduction[0];
    }
}
