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

DATA8 blend8(DATA8 src1, DATA8 src2, DATA8 src3, Flag8 flag, Flag mask) {
    DATA8 tmp = CONVERT_DATA8((convert_int8(src1) + convert_int8(src3) + convert_int8(src2) + convert_int8(src2) + (int8)2) >> 2);
    return (CONVERT_DATA8((flag & (Flag8)mask) == (Flag8)0) ? tmp : src2);
}

DATA8 mie_inter8(DATA8 src1, DATA8 src2, DATA8 src3, DATA8 src4) {
    return CONVERT_DATA8((convert_int8(src1) + convert_int8(src2) + convert_int8(src3) + convert_int8(src4) + (int8)2) >> 2);
}

DATA8 mie_spot8(DATA8 src1, DATA8 src2, DATA8 src3, DATA8 src4, DATA8 src_spot) {
    return CONVERT_DATA8((convert_int8(mie_inter8(src1, src2, src3, src4)) + convert_int8(src_spot) + (int8)1) >> 1);
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

// YUV444のビルドでもkernel_synthesize_mode_1234_yuv420から参照されるため、#if YUV420では囲まないこと
// 色差の合成は、420→422の補間値を8倍の整数で持ち回り、テクスチャ(image)、localメモリ、barrierを使わずに行う。
// (以前はread_imagefでの補間結果をlocalメモリに置き、work group内で18回同期しながら処理していたため、
//  localメモリの読み書きと同期待ちが律速になっていた)
// 色差はソースキャッシュでフィールド分離済み (src0=偶数行, src1=奇数行、高さはlastFieldRow+1)

// 422相当の行rowの補間値 (8倍) を、色差4画素(x*4～x*4+3)分求める
// 縦方向の重みは従来の線形補間と同じ (同じフィールドの色差の行を 1/8単位で補間する)
int4 load_uv_422(__global const uchar *src0, __global const uchar *src1, const int pitch, const int lastFieldRow, const int x, const int row) {
    __global const uchar *src = (row & 1) ? src1 : src0;
    const int base = (row - 2) >> 2;
    const int weight = 7 - ((row & 3) << 1);
    const int row0 = clamp(base,     0, lastFieldRow);
    const int row1 = clamp(base + 1, 0, lastFieldRow);
    const int4 a = convert_int4(vload4(x, (__global const DATA *)(src + row0 * pitch)));
    const int4 b = convert_int4(vload4(x, (__global const DATA *)(src + row1 * pitch)));
    return (int4)(8 - weight) * a + (int4)weight * b;
}

int4 mie_inter4(int4 src1, int4 src2, int4 src3, int4 src4) {
    return (src1 + src2 + src3 + src4 + (int4)2) >> 2;
}

int4 mie_spot4(int4 src1, int4 src2, int4 src3, int4 src4, int4 src_spot) {
    return (mie_inter4(src1, src2, src3, src4) + src_spot + (int4)1) >> 1;
}

int4 blend4(int4 src1, int4 src2, int4 src3, int4 flag, int mask) {
    const int4 tmp = (src1 + src3 + src2 + src2 + (int4)2) >> 2;
    return ((flag & (int4)mask) == (int4)0) ? tmp : src2;
}

// 422相当の行rowの合成結果 (8倍) を、色差4画素分求める
// p0x/p1x: p0/p1のフィールド分離された色差 (x=0:偶数行, 1:奇数行)
int4 synth_uv_422(
    __global const uchar *p0_0, __global const uchar *p0_1,
    __global const uchar *p1_0, __global const uchar *p1_1,
    const int pitch, const int lastFieldRow, const int x, const int row,
    const int4 flags, const int latter, const int shift) {
    __global const uchar *center0   = (shift && latter)  ? p1_0 : p0_0;
    __global const uchar *center1   = (shift && latter)  ? p1_1 : p0_1;
    __global const uchar *neighbor0 = (shift && !latter) ? p1_0 : p0_0;
    __global const uchar *neighbor1 = (shift && !latter) ? p1_1 : p0_1;
    const int4 middle = load_uv_422(center0, center1, pitch, lastFieldRow, x, row);
    if (mode == 1) {
        const int4 top    = load_uv_422(neighbor0, neighbor1, pitch, lastFieldRow, x, row - 1);
        const int4 bottom = load_uv_422(neighbor0, neighbor1, pitch, lastFieldRow, x, row + 1);
        if (shift != latter) {
            const int4 other = load_uv_422(p1_0, p1_1, pitch, lastFieldRow, x, row);
            return mie_inter4(top, middle, bottom, other);
        } else {
            const int4 otherTop    = load_uv_422(p1_0, p1_1, pitch, lastFieldRow, x, row - 1);
            const int4 otherBottom = load_uv_422(p1_0, p1_1, pitch, lastFieldRow, x, row + 1);
            return mie_spot4(top, bottom, otherTop, otherBottom, middle);
        }
    } else if (mode == 4) {
        if (shift != latter) {
            const int4 top       = load_uv_422(neighbor0, neighbor1, pitch, lastFieldRow, x, row - 1);
            const int4 bottom    = load_uv_422(neighbor0, neighbor1, pitch, lastFieldRow, x, row + 1);
            const int4 farTop    = load_uv_422(neighbor0, neighbor1, pitch, lastFieldRow, x, row - 3);
            const int4 farBottom = load_uv_422(neighbor0, neighbor1, pitch, lastFieldRow, x, row + 3);
            const int mask = (shift) ? 0x06 : 0x05;
            const int maxScaled = ((1 << (8 * sizeof(DATA))) - 1) * 8;
            const int4 filtered = ((int4)9 * (top + bottom) - (farTop + farBottom) + (int4)8) >> 4;
            return ((flags & (int4)mask) != (int4)0) ? middle : clamp(filtered, (int4)0, (int4)maxScaled);
        } else {
            return middle;
        }
    } else {
        const int4 top    = load_uv_422(neighbor0, neighbor1, pitch, lastFieldRow, x, row - 1);
        const int4 bottom = load_uv_422(neighbor0, neighbor1, pitch, lastFieldRow, x, row + 1);
        const int mask = (mode == 2) ? ((shift) ? 0x02 : 0x01) : ((shift) ? 0x06 : 0x05);
        return blend4(top, middle, bottom, flags, mask);
    }
}

// 422相当の行rowの、色差4画素に対応するsipのフラグ (sipは輝度解像度なので偶数画素のもの) を読む
int4 load_uv_flags(__global const uchar *sip, const int sip_pitch, const int x, const int row, const int height) {
    if (row >= height) {
        return (int4)0;
    }
    const uchar8 flags = vload8(x, sip + row * sip_pitch);
    return convert_int4(flags.s0246);
}

// 出力(420)の行imgyの色差4画素を求める
// 422相当の2行 (row, row+2) を合成してから、420へ縦に補間する
DATA4 proc_uv(
    __global const uchar *p0_0, __global const uchar *p0_1,
    __global const uchar *p1_0, __global const uchar *p1_1,
    const int pitch, const int lastFieldRow, const int x, const int row, const int odd,
    const int4 flags0, const int4 flags1, const int latter, const int shift) {
    const int4 s0 = synth_uv_422(p0_0, p0_1, p1_0, p1_1, pitch, lastFieldRow, x, row,     flags0, latter, shift);
    const int4 s1 = synth_uv_422(p0_0, p0_1, p1_0, p1_1, pitch, lastFieldRow, x, row + 2, flags1, latter, shift);
    const int4 v = (odd) ? (s0 + (int4)3 * s1) : ((int4)3 * s0 + s1);
    // 整数の画素値に戻すため、旧経路の正規化分母と復元倍率の差による偏りがない。
    return CONVERT_DATA4((v + (int4)16) >> 5);
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
    __global const uchar *__restrict__ p0U0, //色差はフィールド分離済み (x0:偶数行, x1:奇数行)
    __global const uchar *__restrict__ p0U1,
    __global const uchar *__restrict__ p1U0,
    __global const uchar *__restrict__ p1U1,
    __global const uchar *__restrict__ p0V0,
    __global const uchar *__restrict__ p0V1,
    __global const uchar *__restrict__ p1V0,
    __global const uchar *__restrict__ p1V1,
    const int width, const int height,
    const int src_y_pitch, const int src_c_pitch, const int dst_y_pitch, const int dst_c_pitch, const int sip_pitch,
    const int tb_order, const uchar status) {
    const int lx = get_local_id(0); //スレッド数=SYN_BLOCK_INT_X
    const int ly = get_local_id(1); //スレッド数=SYN_BLOCK_Y
    const int imgx = get_group_id(0) * SYN_BLOCK_INT_X /*blockDim.x*/ + lx; //グローバルスレッドID x
    const int imgy = get_group_id(1) * SYN_BLOCK_Y     /*blockDim.y*/ + ly; //グローバルスレッドID y

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
    const int uv_height = height >> 1;
    if (imgx * 8 < width && imgy < uv_height) {
        //imgyは出力(420)の色差の行、rowは422相当の行 (420の行imgyは422の行 row と row+2 から補間する)
        const int odd = imgy & 1;
        const int row = 2 * imgy - odd;
        const int lastFieldRow = (uv_height >> 1) - 1;
        const int4 flags0 = (mode >= 2) ? load_uv_flags(sip, sip_pitch, imgx, row,     height) : (int4)0;
        const int4 flags1 = (mode >= 2) ? load_uv_flags(sip, sip_pitch, imgx, row + 2, height) : (int4)0;
        const int latter = is_latter_field(row, tb_order);
        const int shift = (status & AFS_FLAG_SHIFT0) != 0;
        const int uv_pos_dst = imgy * dst_c_pitch + imgx * sizeof(DATA4);
        *(__global DATA4 *)(dst_u + uv_pos_dst) = proc_uv(p0U0, p0U1, p1U0, p1U1, src_c_pitch, lastFieldRow, imgx, row, odd, flags0, flags1, latter, shift);
        *(__global DATA4 *)(dst_v + uv_pos_dst) = proc_uv(p0V0, p0V1, p1V0, p1V1, src_c_pitch, lastFieldRow, imgx, row, odd, flags0, flags1, latter, shift);
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
    //YUV420の場合、色差はフィールド分離済みで、p0U/p0V/p1U/p1Vが偶数行、以下が奇数行 (YUV444では使用しない)
    __global const uchar *__restrict__ p0U1,
    __global const uchar *__restrict__ p0V1,
    __global const uchar *__restrict__ p1U1,
    __global const uchar *__restrict__ p1V1,
    const int width, const int height,
    const int src_y_pitch, const int src_c_pitch,
    const int dst_y_pitch, const int dst_c_pitch,
    const int tb_order, const uchar status) {
    const int lx = get_local_id(0); //スレッド数=SYN_BLOCK_INT_X
    const int ly = get_local_id(1); //スレッド数=SYN_BLOCK_Y
    const int imgx = get_group_id(0) * SYN_BLOCK_INT_X /*blockDim.x*/ + lx;
    const int imgy = get_group_id(1) * SYN_BLOCK_Y     /*blockDim.y*/ + ly;

    if (imgx * 8 < width) {
        //YUV420の色差の行imgyは、imgyの偶奇と同じフィールドのバッファの imgy>>1 行目にある
        //(以前は偶数行のバッファだけを参照していたため、奇数行に偶数行の色差が入っていた)
        const int field_c = imgy & 1;
        __global const uchar *src_y, *src_u, *src_v;
        if (is_latter_field(ly, tb_order) & (status & AFS_FLAG_SHIFT0)) {
            src_y = p1Y;
            src_u = (YUV420 && field_c) ? p1U1 : p1U;
            src_v = (YUV420 && field_c) ? p1V1 : p1V;
        } else {
            src_y = p0Y;
            src_u = (YUV420 && field_c) ? p0U1 : p0U;
            src_v = (YUV420 && field_c) ? p0V1 : p0V;
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
