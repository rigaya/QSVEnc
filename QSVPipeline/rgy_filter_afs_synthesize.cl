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


// SYN_BLOCK_INT_X  (32) //work groupサイズ(x) = スレッド数/work group
// SYN_BLOCK_Y       (8) //work groupサイズ(y) = スレッド数/work group
// SYN_BLOCK_LOOP_Y  (1) //work groupのy方向反復数
// TUNE_SELECT       (0) //0: デフォルト, 1: SIP, 2: SP_SHIFT, 3: SP_NONSHIFT
// TUNE_SIP
// TUNE_SP_SHIFT
// TUNE_SP_NONSHIFT

#define u8x4(x)  (((uint)x) | (((uint)x) <<  8) | (((uint)x) << 16) | (((uint)x) << 24))

#define AFS_FLAG_SHIFT0      0x01
#define AFS_FLAG_SHIFT1      0x02
#define AFS_FLAG_SHIFT2      0x04
#define AFS_FLAG_SHIFT3      0x08
#define AFS_FLAG_FRAME_DROP  0x10
#define AFS_FLAG_SMOOTHING   0x20
#define AFS_FLAG_FORCE24     0x40

// mode
// BIT_DEPTH

__constant sampler_t sampler_y = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
#if YUV420
__constant sampler_t sampler_c = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
#else
__constant sampler_t sampler_c = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
#endif

#if BIT_DEPTH > 8
typedef ushort        DATA;
typedef ushort2       DATA2;
typedef ushort4       DATA4;
typedef ushort8       DATA8;
#define CONVERT_DATA8 convert_ushort8
#define CONVERT_DATA4 convert_ushort4
#define CONVERT_FLAG4_SAT convert_uchar4_sat
#define AS_DATA4      as_ushort4
#else
typedef uchar         DATA;
typedef uchar2        DATA2;
typedef uchar4        DATA4;
typedef uchar8        DATA8;
#define CONVERT_DATA8 convert_uchar8
#define CONVERT_DATA4 convert_uchar4
#define CONVERT_FLAG4_SAT
#define AS_DATA4      as_uchar4
#endif

typedef uchar         Flag;
typedef uchar4        Flag4;
typedef uchar8        Flag8;

float lerp(float v0, float v1, float t) {
    return (1.0f - t) * v0 + t * v1;
}

uint blend_mask(uint a_if_0, uint b_if_1, uint mask) {
    return (a_if_0 & (~mask)) | (b_if_1 & (mask));
}

// 後方フィールド判定
int is_latter_field(int pos_y, int tb_order) {
    return (((pos_y + tb_order + 1) & 1));
}

DATA8 deint8(DATA8 src1, DATA8 src3, DATA8 src4, DATA8 src5, DATA8 src7, Flag8 flag, Flag mask) {
    const int8 tmp2 = convert_int8(src1) + convert_int8(src7);
    const int8 tmp3 = convert_int8(src3) + convert_int8(src5);
    const DATA8 tmp = CONVERT_DATA8((tmp3 - ((tmp2 - tmp3) >> 3) + (int8)1) >> 1);
    return (CONVERT_DATA8((flag & (Flag8)mask) == (Flag8)0) ? tmp : src4);
}

float deintf(float src1, float src3, float src4, float src5, float src7, Flag flag, Flag mask) {
    //const float tmp2 = src1 + src7;
    //const float tmp3 = src3 + src5;
    //const float tmp = (tmp3 - ((tmp2 - tmp3) * 0.125f)) * 0.5f;
    const float tmp = (src3 + src5) * 0.5625f - (src1 + src7) * 0.0625f;
    return (((flag & mask) == 0) ? tmp : src4);
}

DATA8 blend8(DATA8 src1, DATA8 src2, DATA8 src3, Flag8 flag, Flag mask) {
    DATA8 tmp = CONVERT_DATA8((convert_int8(src1) + convert_int8(src3) + convert_int8(src2) + convert_int8(src2) + (int8)2) >> 2);
    return (CONVERT_DATA8((flag & (Flag8)mask) == (Flag8)0) ? tmp : src2);
}

float blendf(float src1, float src2, float src3, Flag flag, Flag mask) {
    float tmp = (src1 + src3 + 2.0f * src2) * 0.25f;
    return ((flag & mask) == 0) ? tmp : src2;
}

DATA8 mie_inter8(DATA8 src1, DATA8 src2, DATA8 src3, DATA8 src4) {
    return CONVERT_DATA8((convert_int8(src1) + convert_int8(src2) + convert_int8(src3) + convert_int8(src4) + (int8)2) >> 2);
}

float mie_interf(float src1, float src2, float src3, float src4) {
    return (src1 + src2 + src3 + src4) * 0.25f;
}

DATA8 mie_spot8(DATA8 src1, DATA8 src2, DATA8 src3, DATA8 src4, DATA8 src_spot) {
    return CONVERT_DATA8((convert_int8(mie_inter8(src1, src2, src3, src4)) + convert_int8(src_spot) + (int8)1) >> 1);
}

float mie_spotf(float src1, float src2, float src3, float src4, float src_spot) {
    return (mie_interf(src1, src2, src3, src4) + src_spot) * 0.5f;
}

DATA8 piny(
    __global const uchar *__restrict__ p0,
    __global const uchar *__restrict__ p1,
    int y_h1_pos, int y_h2_pos, int y_h3_pos,
    int y_h4_pos, int y_h5_pos, int y_h6_pos, int y_h7_pos,
    int plane, int line) {
    __global const uchar *ptr = (plane) ? p1 : p0;
    switch (line) {
    case 1: ptr += y_h1_pos; break;
    case 2: ptr += y_h2_pos; break;
    case 3: ptr += y_h3_pos; break;
    case 4: ptr += y_h4_pos; break;
    case 5: ptr += y_h5_pos; break;
    case 6: ptr += y_h6_pos; break;
    case 7: ptr += y_h7_pos; break;
    default: break;
    }
    return *(__global DATA8 *)ptr;
}

void proc_y(
    __global uchar *__restrict__ dst,
    __global const uchar *__restrict__ p0,
    __global const uchar *__restrict__ p1,
    __global const uchar *__restrict__ sip,
    const int tb_order, const uchar status,
    int y_h1_pos, int y_h2_pos, int y_h3_pos,
    int y_h4_pos, int y_h5_pos, int y_h6_pos, int y_h7_pos
) {
    //static_assert(-1 <= mode && mode <= 4, "mode should be -1 - 4");
#define pin(plane, line) piny(p0, p1, y_h1_pos, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos, plane, line)

    DATA8 pout;
    if (mode == 1) {
        if (status & AFS_FLAG_SHIFT0) {
            if (!is_latter_field(0, tb_order)) {
                pout = mie_inter8(pin(0, 2), pin(1, 1), pin(1, 2), pin(1, 3));
            } else {
                pout = mie_spot8(pin(0, 1), pin(0, 3), pin(1, 1), pin(1, 3), pin(1, 2));
            }
        } else {
            if (is_latter_field(0, tb_order)) {
                pout = mie_inter8(pin(0, 1), pin(0, 2), pin(0, 3), pin(1, 2));
            } else {
                pout = mie_spot8(pin(0, 1), pin(0, 3), pin(1, 1), pin(1, 3), pin(0, 2));
            }
        }
    } else if (mode == 2) {
        if (status & AFS_FLAG_SHIFT0) {
            if (!is_latter_field(0, tb_order)) {
                pout = blend8(pin(1, 1), pin(0, 2), pin(1, 3), *(__global Flag8 *)sip, 0x02);
            } else {
                pout = blend8(pin(0, 1), pin(1, 2), pin(0, 3), *(__global Flag8 *)sip, 0x02);
            }
        } else {
            pout = blend8(pin(0, 1), pin(0, 2), pin(0, 3), *(__global Flag8 *)sip, 0x01);
        }
    } else if (mode == 3) {
        if (status & AFS_FLAG_SHIFT0) {
            if (!is_latter_field(0, tb_order)) {
                pout = blend8(pin(1, 1), pin(0, 2), pin(1, 3), *(__global Flag8 *)sip, 0x06);
            } else {
                pout = blend8(pin(0, 1), pin(1, 2), pin(0, 3), *(__global Flag8 *)sip, 0x06);
            }
        } else {
            pout = blend8(pin(0, 1), pin(0, 2), pin(0, 3), *(__global Flag8 *)sip, 0x05);
        }
    } else if (mode == 4) {
        if (status & AFS_FLAG_SHIFT0) {
            if (!is_latter_field(0, tb_order)) {
                pout = deint8(pin(1, 1), pin(1, 3), pin(0, 4), pin(1, 5), pin(1, 7), *(__global Flag8 *)sip, 0x06);
            } else {
                pout = pin(1, 4);
            }
        } else {
            if (is_latter_field(0, tb_order)) {
                pout = deint8(pin(0, 1), pin(0, 3), pin(0, 4), pin(0, 5), pin(0, 7), *(__global Flag8 *)sip, 0x05);
            } else {
                pout = pin(0, 4);
            }
        }
    }
    *(__global DATA8 *)dst = pout;
#undef pin
}

float get_uv(__read_only image2d_t src_p0_0, __read_only image2d_t src_p0_1, float ifx, int iy) {
    //static const float WEIGHT[4] = {
    //    7.0f / 8.0f,
    //    5.0f / 8.0f,
    //    3.0f / 8.0f,
    //    1.0f / 8.0f
    //};
    //const float ifytex = ify + WEIGHT[iy & 3];
    const float ify = ((iy - 2) >> 2) + 0.5f;
    const float ifytex = ify + (3.5f - (float)(iy & 3)) * 0.25f;
    if (iy & 1) {
        return read_imagef(src_p0_1, sampler_c, (float2)(ifx, ifytex)).x;
    } else {
        return read_imagef(src_p0_0, sampler_c, (float2)(ifx, ifytex)).x;
    }
}


#define PREREAD_Y ((mode == 4) ? 6 : 2) //必要になる前後の行のぶん
#define SHARED_C_Y (SYN_BLOCK_Y * 2 + PREREAD_Y)
#define SHARED_C_XY (SHARED_C_Y * SYN_BLOCK_INT_X)

void proc_uv(
    __global uchar *__restrict__ dst,
    __read_only image2d_t src_p0_0,
    __read_only image2d_t src_p0_1,
    __read_only image2d_t src_p1_0,
    __read_only image2d_t src_p1_1,
    __global const uchar *__restrict__ sip,
    __local float s_tmp[3][SHARED_C_Y][SYN_BLOCK_INT_X],
    __local DATA s_out[SYN_BLOCK_Y][SYN_BLOCK_INT_X * 4],
    const int sip_pitch,
    const int src_width, const int src_height,
    const int imgx,  //グローバルスレッドID x
    const int imgy,  //グローバルスレッドID y
    const int lx, const int ly, //ローカルスレッドID
    const int tb_order, const uchar status
) {
    //static_assert(-1 <= mode && mode <= 4, "mode should be -1 - 4");
#define SOFFSET(x,y,depth) ((depth) * SHARED_C_XY + (y) * SYN_BLOCK_INT_X + (x))
    __local float *pSharedX = (__local float *)&s_tmp[0][0][0] + SOFFSET(lx, 0, 0);
    __local DATA *psOut = &s_out[ly][lx];
    int ix = get_group_id(0) * SYN_BLOCK_INT_X * 4 + get_local_id(0); //YUV422ベースの色差インデックス 1スレッドは横方向に4つの色差pixelを担当
    int iy = get_group_id(1) * SYN_BLOCK_Y * 2 + get_local_id(1);     //YUV422ベースの色差インデックス 1スレッドは縦方向に2つの色差pixelを担当 (出力時はYUV420なので1つの色差pixelを担当)
    float ifx = (float)ix + 0.5f;

    //この関数内でsipだけはYUV444のデータであることに注意
    sip += iy * sip_pitch + ix * 2/*YUV444->YUV420*/ * sizeof(uchar);

    //sharedメモリ上に、YUV422相当のデータ(32x(16+PREREAD))を縦方向のテクスチャ補間で作ってから、
    //blendを実行して、YUV422相当の合成データ(32x16)を作り、
    //その後YUV420相当のデータ(32x8)をs_outに出力する
    //横方向に4回ループを回して、32pixel x4の出力結果をs_out(横:128pixel)に格納する
    for (int i = 0; i < 4; i++, ifx += SYN_BLOCK_INT_X, psOut += SYN_BLOCK_INT_X, sip += SYN_BLOCK_INT_X * 2/*YUV444->YUV420*/) {
        //shredメモリに値をロード
        //縦方向のテクスチャ補間を使って、YUV422相当のデータとしてロード
        //横方向には補間しない
        if (ly < PREREAD_Y) {
            __local float *pShared = pSharedX + SOFFSET(0, ly, 0);
            pShared[SOFFSET(0, 0, 0)] = get_uv(src_p0_0, src_p0_1, ifx, iy - (PREREAD_Y >> 1));
            pShared[SOFFSET(0, 0, 1)] = get_uv(src_p1_0, src_p1_1, ifx, iy - (PREREAD_Y >> 1));
        }
        __local float *pShared = pSharedX + SOFFSET(0, ly + PREREAD_Y, 0);
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            pShared[SOFFSET(0, j*SYN_BLOCK_Y, 0)] = get_uv(src_p0_0, src_p0_1, ifx, iy + (PREREAD_Y >> 1) + j * SYN_BLOCK_Y);
            pShared[SOFFSET(0, j*SYN_BLOCK_Y, 1)] = get_uv(src_p1_0, src_p1_1, ifx, iy + (PREREAD_Y >> 1) + j * SYN_BLOCK_Y);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int j = 0; j < 2; j++) {
            //sipのy (境界チェックに必要)
            const int iy_sip = iy + j * SYN_BLOCK_Y;
            __global const Flag *psip = sip + j * SYN_BLOCK_Y * sip_pitch;

            // -1するのは、pinのlineは最小値が1だから
            __local float *pShared = pSharedX + SOFFSET(0, ly-1+j*SYN_BLOCK_Y, 0);
#define pin(plane, line) (pShared[SOFFSET(0,line,plane)])

            float pout;
            if (mode == 1) {
                if (status & AFS_FLAG_SHIFT0) {
                    if (!is_latter_field(ly, tb_order)) {
                        pout = mie_interf(pin(0, 2), pin(1, 1), pin(1, 2), pin(1, 3));
                    } else {
                        pout = mie_spotf(pin(0, 1), pin(0, 3), pin(1, 1), pin(1, 3), pin(1, 2));
                    }
                } else {
                    if (is_latter_field(ly, tb_order)) {
                        pout = mie_interf(pin(0, 1), pin(0, 2), pin(0, 3), pin(1, 2));
                    } else {
                        pout = mie_spotf(pin(0, 1), pin(0, 3), pin(1, 1), pin(1, 3), pin(0, 2));
                    }
                }
            } else if (mode == 2) {
                const Flag sip0 = (iy_sip < src_height) ? *psip : 0;
                if (status & AFS_FLAG_SHIFT0) {
                    if (!is_latter_field(ly, tb_order)) {
                        pout = blendf(pin(1, 1), pin(0, 2), pin(1, 3), sip0, 0x02);
                    } else {
                        pout = blendf(pin(0, 1), pin(1, 2), pin(0, 3), sip0, 0x02);
                    }
                } else {
                    pout = blendf(pin(0, 1), pin(0, 2), pin(0, 3), sip0, 0x01);
                }
            } else if (mode == 3) {
                const Flag sip0 = (iy_sip < src_height) ? *psip : 0;
                if (status & AFS_FLAG_SHIFT0) {
                    if (!is_latter_field(ly, tb_order)) {
                        pout = blendf(pin(1, 1), pin(0, 2), pin(1, 3), sip0, 0x06);
                    } else {
                        pout = blendf(pin(0, 1), pin(1, 2), pin(0, 3), sip0, 0x06);
                    }
                } else {
                    pout = blendf(pin(0, 1), pin(0, 2), pin(0, 3), sip0, 0x05);
                }
            } else if (mode == 4) {
                const Flag sip0 = (iy_sip < src_height) ? *psip : 0;
                if (status & AFS_FLAG_SHIFT0) {
                    if (!is_latter_field(ly, tb_order)) {
                        pout = deintf(pin(1, 1), pin(1, 3), pin(0, 4), pin(1, 5), pin(1, 7), sip0, 0x06);
                    } else {
                        pout = pin(1, 4);
                    }
                } else {
                    if (is_latter_field(ly, tb_order)) {
                        pout = deintf(pin(0, 1), pin(0, 3), pin(0, 4), pin(0, 5), pin(0, 7), sip0, 0x05);
                    } else {
                        pout = pin(0, 4);
                    }
                }
            }
            pSharedX[SOFFSET(0, ly+j*SYN_BLOCK_Y, 2)] = pout;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        //sharedメモリ内でYUV422->YUV420
        const int sy = (ly << 1) - (ly & 1);
        pShared = pSharedX + SOFFSET(0, sy, 2);
        psOut[0] = (DATA)(lerp(pShared[SOFFSET(0, 0, 0)], pShared[SOFFSET(0, 2, 0)], (ly & 1) ? 0.75f : 0.25f) * (float)(1<<(8*sizeof(DATA))) + 0.5f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //s_outに出力したものをメモリに書き出す
    if (imgx < (src_width >> 1) && imgy < (src_height >> 1)) {
        *(__global DATA4 *)dst = *(__local DATA4 *)(&s_out[ly][lx << 2]);
    }
#undef SOFFSET
}

void set_y_h_pos(const int imgx, const int y_h_center, int height, const int src_pitch, int *y_h1_pos, int *y_h2_pos, int *y_h3_pos, int *y_h4_pos, int *y_h5_pos, int *y_h6_pos, int *y_h7_pos, int *y_h8_pos) {
    if (mode == 4) {
        *y_h4_pos = y_h_center * src_pitch + imgx * sizeof(DATA8);
        *y_h3_pos = *y_h4_pos + ((y_h_center - 1 >= 0) ? -src_pitch : src_pitch);
        *y_h2_pos = *y_h3_pos + ((y_h_center - 2 >= 0) ? -src_pitch : src_pitch);
        *y_h1_pos = *y_h2_pos + ((y_h_center - 3 >= 0) ? -src_pitch : src_pitch);
        *y_h5_pos = *y_h4_pos + ((y_h_center < height - 1) ? src_pitch : -src_pitch);
        *y_h6_pos = *y_h5_pos + ((y_h_center < height - 2) ? src_pitch : -src_pitch);
        *y_h7_pos = *y_h6_pos + ((y_h_center < height - 3) ? src_pitch : -src_pitch);
        *y_h8_pos = *y_h7_pos + ((y_h_center < height - 4) ? src_pitch : -src_pitch);
    } else {
        *y_h2_pos = y_h_center * src_pitch + imgx * sizeof(DATA8);
        *y_h1_pos = *y_h2_pos + ((y_h_center - 1 >= 0) ? -src_pitch : src_pitch);
        *y_h3_pos = *y_h2_pos + ((y_h_center < height - 1) ? src_pitch : -src_pitch);
        *y_h4_pos = *y_h3_pos + ((y_h_center < height - 2) ? src_pitch : -src_pitch);
    }
}

__kernel void kernel_synthesize_mode_1234_yuv420(
    __global uchar *__restrict__ dst_y,
    __global uchar *__restrict__ dst_u,
    __global uchar *__restrict__ dst_v,
    __global const uchar *__restrict__ p0,
    __global const uchar *__restrict__ p1,
    __global const uchar *__restrict__ sip,
    __read_only image2d_t src_u0_0,
    __read_only image2d_t src_u0_1,
    __read_only image2d_t src_u1_0,
    __read_only image2d_t src_u1_1,
    __read_only image2d_t src_v0_0,
    __read_only image2d_t src_v0_1,
    __read_only image2d_t src_v1_0,
    __read_only image2d_t src_v1_1,
    const int width, const int height,
    const int src_y_pitch, const int dst_y_pitch, const int dst_c_pitch, const int sip_pitch,
    const int tb_order, const uchar status) {
    const int lx = get_local_id(0); //スレッド数=SYN_BLOCK_INT_X
    const int ly = get_local_id(1); //スレッド数=SYN_BLOCK_Y
    const int imgx = get_group_id(0) * SYN_BLOCK_INT_X /*blockDim.x*/ + lx; //グローバルスレッドID x
    const int imgy = get_group_id(1) * SYN_BLOCK_Y     /*blockDim.y*/ + ly; //グローバルスレッドID y
    __local float s_tmp[3][SHARED_C_Y][SYN_BLOCK_INT_X];
    __local DATA s_out[SYN_BLOCK_Y][SYN_BLOCK_INT_X * 4];

    if (imgx * 8 < width && imgy < (height >> 1)) {
        //y
        const int y_h_center = imgy << 1;

        int y_h1_pos, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos, y_h8_pos;
        set_y_h_pos(imgx, y_h_center, height, src_y_pitch, &y_h1_pos, &y_h2_pos, &y_h3_pos, &y_h4_pos, &y_h5_pos, &y_h6_pos, &y_h7_pos, &y_h8_pos);

        dst_y += y_h_center * dst_y_pitch + imgx * sizeof(DATA8);
        __global const Flag *sip_y = sip  + (y_h_center * sip_pitch + imgx * sizeof(Flag) * 8);
        proc_y(dst_y +           0, p0, p1, sip_y +         0, tb_order + 0, status, y_h1_pos, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos);
        proc_y(dst_y + dst_y_pitch, p0, p1, sip_y + sip_pitch, tb_order + 1, status, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos, y_h8_pos);
    }
    {
        const int uv_pos_dst = imgy * dst_c_pitch + imgx * sizeof(DATA4);
        dst_u += uv_pos_dst;
        dst_v += uv_pos_dst;

        //u
        proc_uv(dst_u, src_u0_0, src_u0_1, src_u1_0, src_u1_1, sip, s_tmp, s_out, sip_pitch, width, height, imgx, imgy,
            lx, ly, tb_order, status);

        //v
        proc_uv(dst_v, src_v0_0, src_v0_1, src_v1_0, src_v1_1, sip, s_tmp, s_out, sip_pitch, width, height, imgx, imgy,
            lx, ly, tb_order, status);
    }
}

__kernel void kernel_synthesize_mode_1234_yuv444(
    __global uchar *__restrict__ dst_y,
    __global uchar *__restrict__ dst_u,
    __global uchar *__restrict__ dst_v,
    __global const uchar *__restrict__ p0Y,
    __global const uchar *__restrict__ p0U,
    __global const uchar *__restrict__ p0V,
    __global const uchar *__restrict__ p1Y,
    __global const uchar *__restrict__ p1U,
    __global const uchar *__restrict__ p1V,
    __global const uchar *__restrict__ sip,
    const int width, const int height,
    const int src_pitch, const int dst_pitch, //Y/U/Vのpitchはすべて共通であることを前提とする
    const int sip_pitch,
    const int tb_order, const uchar status) {
    const int lx = get_local_id(0); //スレッド数=SYN_BLOCK_INT_X
    const int ly = get_local_id(1); //スレッド数=SYN_BLOCK_Y
    const int imgx = get_group_id(0) * SYN_BLOCK_INT_X /*blockDim.x*/ + lx; //グローバルスレッドID x
    const int imgy = get_group_id(1) * SYN_BLOCK_Y     /*blockDim.y*/ + ly; //グローバルスレッドID y

    if (imgx * 8 < width && imgy < (height >> 1)) {
        //y
        const int y_h_center = imgy << 1;

        int y_h1_pos, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos, y_h8_pos;
        set_y_h_pos(imgx, y_h_center, height, src_pitch, &y_h1_pos, &y_h2_pos, &y_h3_pos, &y_h4_pos, &y_h5_pos, &y_h6_pos, &y_h7_pos, &y_h8_pos);

        dst_y += y_h_center * dst_pitch + imgx * sizeof(DATA8);
        dst_u += y_h_center * dst_pitch + imgx * sizeof(DATA8);
        dst_v += y_h_center * dst_pitch + imgx * sizeof(DATA8);
        __global const uchar *sip_y = sip  + (y_h_center * sip_pitch + imgx * sizeof(Flag) * 8);

        proc_y(dst_y +         0, p0Y, p1Y, sip_y +         0, tb_order + 0, status, y_h1_pos, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos);
        proc_y(dst_y + dst_pitch, p0Y, p1Y, sip_y + sip_pitch, tb_order + 1, status, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos, y_h8_pos);

        proc_y(dst_u +         0, p0U, p1U, sip_y +         0, tb_order + 0, status, y_h1_pos, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos);
        proc_y(dst_u + dst_pitch, p0U, p1U, sip_y + sip_pitch, tb_order + 1, status, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos, y_h8_pos);

        proc_y(dst_v +         0, p0V, p1V, sip_y +         0, tb_order + 0, status, y_h1_pos, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos);
        proc_y(dst_v + dst_pitch, p0V, p1V, sip_y + sip_pitch, tb_order + 1, status, y_h2_pos, y_h3_pos, y_h4_pos, y_h5_pos, y_h6_pos, y_h7_pos, y_h8_pos);
    }
}

__kernel void kernel_synthesize_mode_0(
    __global uchar *__restrict__ dst_y,
    __global uchar *__restrict__ dst_u,
    __global uchar *__restrict__ dst_v,
    __global const uchar *__restrict__ p0Y,
    __global const uchar *__restrict__ p0U,
    __global const uchar *__restrict__ p0V,
    __global const uchar *__restrict__ p1Y,
    __global const uchar *__restrict__ p1U,
    __global const uchar *__restrict__ p1V,
    const int width, const int height,
    const int src_y_pitch, const int src_c_pitch,
    const int dst_y_pitch, const int dst_c_pitch,
    const int tb_order, const uchar status) {
    const int lx = get_local_id(0); //スレッド数=SYN_BLOCK_INT_X
    const int ly = get_local_id(1); //スレッド数=SYN_BLOCK_Y
    const int imgx = get_group_id(0) * SYN_BLOCK_INT_X /*blockDim.x*/ + lx;
    const int imgy = get_group_id(1) * SYN_BLOCK_Y     /*blockDim.y*/ + ly;

    if (imgx * 8 < width) {
        __global const uchar *src_y, *src_u, *src_v;
        if (is_latter_field(ly, tb_order) & (status & AFS_FLAG_SHIFT0)) {
            src_y = p1Y;
            src_u = p1U;
            src_v = p1V;
        } else {
            src_y = p0Y;
            src_u = p0U;
            src_v = p0V;
        }
        {
            //y
            const int y_line = (get_group_id(1) * SYN_BLOCK_Y * 2) + ly;
            dst_y += y_line * dst_y_pitch + imgx * sizeof(DATA8);
            src_y += y_line * src_y_pitch + imgx * sizeof(DATA8);
            if (y_line < height) {
                *(__global DATA8 *)dst_y = *(__global DATA8 *)src_y;
                if (!YUV420) {
                    dst_u += y_line * dst_c_pitch + imgx * sizeof(DATA8);
                    dst_v += y_line * dst_c_pitch + imgx * sizeof(DATA8);
                    src_u += y_line * src_c_pitch + imgx * sizeof(DATA8);
                    src_v += y_line * src_c_pitch + imgx * sizeof(DATA8);
                    *(__global DATA8 *)dst_u = *(__global DATA8 *)src_u;
                    *(__global DATA8 *)dst_v = *(__global DATA8 *)src_v;
                }
            }
            if (y_line + SYN_BLOCK_Y < height) {
                dst_y += dst_y_pitch * SYN_BLOCK_Y;
                src_y += src_y_pitch * SYN_BLOCK_Y;
                *(__global DATA8 *)dst_y = *(__global DATA8 *)src_y;
                if (!YUV420) {
                    dst_u += dst_c_pitch * SYN_BLOCK_Y;
                    dst_v += dst_c_pitch * SYN_BLOCK_Y;
                    src_u += src_c_pitch * SYN_BLOCK_Y;
                    src_v += src_c_pitch * SYN_BLOCK_Y;
                    *(__global DATA8 *)dst_u = *(__global DATA8 *)src_u;
                    *(__global DATA8 *)dst_v = *(__global DATA8 *)src_v;
                }
            }
        }

        if (YUV420 && ((imgy << 1) < height)) {
            //u
            const int uv_pos_dst = imgy        * dst_c_pitch + imgx * sizeof(DATA4);
            const int uv_pos_src = (imgy >> 1) * src_c_pitch + imgx * sizeof(DATA4);
            dst_u += uv_pos_dst;
            dst_v += uv_pos_dst;
            src_u += uv_pos_src;
            src_v += uv_pos_src;
            *(__global DATA4 *)dst_u = *(__global DATA4 *)src_u;
            *(__global DATA4 *)dst_v = *(__global DATA4 *)src_v;
        }
    }
}

enum {
    TUNE_COLOR_BLACK = 0,
    TUNE_COLOR_GREY,
    TUNE_COLOR_BLUE,
    TUNE_COLOR_LIGHT_BLUE,
    TUNE_COLOR_RED,
    TUNE_COLOR_LIGHT_RED,
    TUNE_COLOR_PURPLE,
    TUNE_COLOR_LIGHT_PURPLE,
    TUNE_COLOR_MAX,
};

int synthesize_mode_tune_select_color(const uchar sip, const uchar status) {
    int ret = 0;
    if (TUNE_SELECT == TUNE_SP_SHIFT) {
        const uchar tmp = (sip & 0x03) | ((~sip) & 0x40); // 動き(0x40)はflagが逆転している
        if (tmp == 0x01) {
            ret = TUNE_COLOR_GREY;
        } else if (tmp == 0x02) {
            ret = TUNE_COLOR_RED;
        } else if (tmp == 0x40) {
            ret = TUNE_COLOR_BLUE;
        } else if (tmp == (0x01 + 0x02)) {
            ret = TUNE_COLOR_LIGHT_RED;
        } else if (tmp == (0x01 + 0x40)) {
            ret = TUNE_COLOR_LIGHT_BLUE;
        } else if (tmp == (0x02 + 0x40)) {
            ret = TUNE_COLOR_PURPLE;
        } else if (tmp == (0x01 + 0x02 + 0x40)) {
            ret = TUNE_COLOR_LIGHT_PURPLE;
        }
    } else if (TUNE_SELECT == TUNE_SP_NONSHIFT) {
        const uchar tmp = (sip & 0x30) | ((~sip) & 0x04); // 動き(0x04)はflagが逆転している
        if (tmp == 0x10) {
            ret = TUNE_COLOR_GREY;
        } else if (tmp == 0x20) {
            ret = TUNE_COLOR_RED;
        } else if (tmp == 0x04) {
            ret = TUNE_COLOR_BLUE;
        } else if (tmp == (0x10+0x20)) {
            ret = TUNE_COLOR_LIGHT_RED;
        } else if (tmp == (0x10+0x04)) {
            ret = TUNE_COLOR_LIGHT_BLUE;
        } else if (tmp == (0x20+0x04)) {
            ret = TUNE_COLOR_PURPLE;
        } else if (tmp == (0x10+0x20+0x04)) {
            ret = TUNE_COLOR_LIGHT_PURPLE;
        }
    } else { // TUNE_SELECT == TUNE_SIP
        if (status & AFS_FLAG_SHIFT0) {
            if (!(sip & 0x06))
                ret = TUNE_COLOR_LIGHT_BLUE;
            else if (~sip & 0x02)
                ret = TUNE_COLOR_GREY;
            else if (~sip & 0x04)
                ret = TUNE_COLOR_BLUE;
            else
                ret = TUNE_COLOR_BLACK;
        } else {
            if (!(sip & 0x05))
                ret = TUNE_COLOR_LIGHT_BLUE;
            else if (~sip & 0x01)
                ret = TUNE_COLOR_GREY;
            else if (~sip & 0x04)
                ret = TUNE_COLOR_BLUE;
            else
                ret = TUNE_COLOR_BLACK;
        }
    }
    return ret;
}

__kernel void kernel_synthesize_mode_tune(
    __global uchar *__restrict__ dst_y,
    __global uchar *__restrict__ dst_u,
    __global uchar *__restrict__ dst_v,
    __global const uchar *__restrict__ sip,
    const int width, const int height,
    const int dst_y_pitch, const int dst_c_pitch, const int sip_pitch,
    const int tb_order, const uchar status) {
    const int lx = get_local_id(0); //スレッド数=SYN_BLOCK_INT_X
    const int ly = get_local_id(1); //スレッド数=SYN_BLOCK_Y
    const int imgc_x = get_group_id(0) * get_local_size(0) + lx;
    const int imgc_y = get_group_id(1) * get_local_size(1) + ly;
    const int imgy_x = imgc_x << 1;
    const int imgy_y = imgc_y << 1;
    const int YUY2_COLOR[TUNE_COLOR_MAX][3] = {
        {  16,  128, 128 }, // TUNE_COLOR_BLACK
        {  98,  128, 128 }, // TUNE_COLOR_GREY
        {  41,  240, 110 }, // TUNE_COLOR_BLUE
        { 169,  166,  16 }, // TUNE_COLOR_LIGHT_BLUE
        {  95,   98, 216 }, // TUNE_COLOR_RED
        { 173,  115, 167 }, // TUNE_COLOR_LIGHT_RED
        {  83,  167, 142 }, // TUNE_COLOR_PURPLE
        { 167,  152, 136 }  // TUNE_COLOR_LIGHT_PURPLE
    };

    if (imgy_x < width && imgy_y < height) {
        sip   += imgy_y * sip_pitch + imgy_x * sizeof(uchar);
        dst_y += imgy_y * dst_y_pitch + imgy_x * sizeof(DATA);

        uchar2 sip2 = *(__global uchar2 *)sip;
        const int c00 = synthesize_mode_tune_select_color(sip2.x, status);
        const int c01 = synthesize_mode_tune_select_color(sip2.y, status);
        sip2 = *(__global uchar2 *)(sip + sip_pitch);
        const int c10 = synthesize_mode_tune_select_color(sip2.x, status);
        const int c11 = synthesize_mode_tune_select_color(sip2.y, status);

        DATA2 dst_y2;
        dst_y2.x = (DATA)(YUY2_COLOR[c00][0] << (BIT_DEPTH - 8));
        dst_y2.y = (DATA)(YUY2_COLOR[c01][0] << (BIT_DEPTH - 8));
        *(__global DATA2 *)dst_y = dst_y2;
        dst_y2.x = (DATA)(YUY2_COLOR[c10][0] << (BIT_DEPTH - 8));
        dst_y2.y = (DATA)(YUY2_COLOR[c11][0] << (BIT_DEPTH - 8));
        *(__global DATA2 *)(dst_y + dst_y_pitch) = dst_y2;

        if (YUV420) {
            dst_u += imgc_y * dst_c_pitch + imgc_x * sizeof(DATA);
            dst_v += imgc_y * dst_c_pitch + imgc_x * sizeof(DATA);
            *(__global DATA *)dst_u = (DATA)(((YUY2_COLOR[c00][1] + YUY2_COLOR[c01][1] + YUY2_COLOR[c10][1] + YUY2_COLOR[c11][1] + 2) << (BIT_DEPTH - 8)) >> 2);
            *(__global DATA *)dst_v = (DATA)(((YUY2_COLOR[c00][2] + YUY2_COLOR[c01][2] + YUY2_COLOR[c10][2] + YUY2_COLOR[c11][2] + 2) << (BIT_DEPTH - 8)) >> 2);
        } else {
            dst_u += imgy_y * dst_c_pitch + imgy_x * sizeof(DATA);
            dst_v += imgy_y * dst_c_pitch + imgy_x * sizeof(DATA);

            DATA2 dst_u2;
            dst_u2.x = (DATA)(YUY2_COLOR[c00][1] << (BIT_DEPTH - 8));
            dst_u2.y = (DATA)(YUY2_COLOR[c01][1] << (BIT_DEPTH - 8));
            *(__global DATA2 *)dst_u = dst_u2;
            dst_u2.x = (DATA)(YUY2_COLOR[c10][1] << (BIT_DEPTH - 8));
            dst_u2.y = (DATA)(YUY2_COLOR[c11][1] << (BIT_DEPTH - 8));
            *(__global DATA2 *)(dst_u + dst_c_pitch) = dst_u2;

            DATA2 dst_v2;
            dst_v2.x = (DATA)(YUY2_COLOR[c00][2] << (BIT_DEPTH - 8));
            dst_v2.y = (DATA)(YUY2_COLOR[c01][2] << (BIT_DEPTH - 8));
            *(__global DATA2 *)dst_v = dst_v2;
            dst_v2.x = (DATA)(YUY2_COLOR[c10][2] << (BIT_DEPTH - 8));
            dst_v2.y = (DATA)(YUY2_COLOR[c11][2] << (BIT_DEPTH - 8));
            *(__global DATA2 *)(dst_v + dst_c_pitch) = dst_v2;
        }
    }
}
