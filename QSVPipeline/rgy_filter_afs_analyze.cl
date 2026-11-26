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


// BLOCK_INT_X  (32) //blockDim(x) = スレッド数/ブロック
// BLOCK_Y       (8) //blockDim(y) = スレッド数/ブロック
// BLOCK_LOOP_Y (16) //ブロックのy方向反復数

#define SHARED_INT_X (BLOCK_INT_X) //sharedメモリの幅
#define SHARED_Y     (16) //sharedメモリの縦

// TB_ORDER 0
// YUV420 0
// BIT_DEPTH
// SUB_GROUP_SIZE
// SELECT_PLANE
// SELECT_PLANE_Y
// SELECT_PLANE_U
// SELECT_PLANE_V
// LOAD_IMAGEF    // LinuxのIntel iGPUドライバでは、read_imageuiを使うとエラーになることへの回避策
// error: undefined reference to _Z37__spirv_ImageSampleExplicitLod_Ruint4とか出てしまう
// ビルドエラーになったらLOAD_IMAGEF=1にして、無理やり回避する

__constant sampler_t sampler_y = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#ifndef SUB_GROUP_SIZE
#define SUB_GROUP_SIZE (0)
#endif

int block_sum_int(int val, __local int *shared) {
    const int lid = get_local_id(1) * BLOCK_INT_X + get_local_id(0);
#if SUB_GROUP_SIZE > 0
    const int lane    = get_sub_group_local_id();
    const int warp_id = get_sub_group_id();
    
	int value_count = BLOCK_INT_X * BLOCK_Y;
	for (;;) {
		val = sub_group_reduce_add(val);
		if (lane == 0) shared[warp_id] = val;
		barrier(CLK_LOCAL_MEM_FENCE);
		value_count /= SUB_GROUP_SIZE;
		if (value_count <= 1) break;
        val = (lid < value_count) ? shared[lane] : 0;
        // 全subgroupの読み取りが終わるまで、次の反復で共有領域を上書きしない。
        barrier(CLK_LOCAL_MEM_FENCE);
	}
#else
    shared[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = BLOCK_Y * BLOCK_INT_X >> 1; offset > 0; offset >>= 1) {
        if (lid < offset) {
            shared[lid] += shared[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif
    int ret = 0;
    if (lid == 0) {
        ret = shared[lid];
    }
    return ret;
}

#define u8x4(x)  (((uint)x) | (((uint)x) <<  8) | (((uint)x) << 16) | (((uint)x) << 24))

#if BIT_DEPTH > 8
typedef ushort         DATA;
typedef ushort4        DATA4;
typedef ushort8        DATA8;
#define CONVERT_DATA4 convert_ushort4
#define CONVERT_DATA8 convert_ushort8
#define CONVERT_FLAG4_SAT convert_uchar4_sat
#define AS_DATA4      as_ushort4
#else
typedef uchar         DATA;
typedef uchar4        DATA4;
typedef uchar8        DATA8;
#define CONVERT_DATA4 convert_uchar4
#define CONVERT_DATA8 convert_uchar8
#define CONVERT_FLAG4_SAT
#define AS_DATA4      as_uchar4
#endif

typedef uchar         Flag;
typedef uchar4        Flag4;
typedef uint          Flag4U;
#define CONVERT_FLAG4 convert_uchar4
#define AS_FLAG4      as_uchar4
#define AS_FLAG4U     as_uint

//      7       6         5        4        3        2        1       0
// | motion  |         non-shift        | motion  |          shift          |
// |  shift  |  sign  |  shift |  deint |  flag   | sign  |  shift |  deint |
#define motion_flag   (0x08u)
#define motion_shift  (0x80u)

#define non_shift_sign  (0x40u)
#define non_shift_shift (0x20u)
#define non_shift_deint (0x10u)

#define shift_sign  (0x04u)
#define shift_shift (0x02u)
#define shift_deint (0x01u)

Flag4U analyze_motion(DATA4 p0, DATA4 p1, DATA thre_motion, DATA thre_shift) {
    DATA4 absdata = abs_diff(p0, p1);
    Flag4 mask_motion = CONVERT_FLAG4_SAT(((DATA4)thre_motion > absdata) ? (DATA4)motion_flag  : (DATA4)0);
    Flag4 mask_shift  = CONVERT_FLAG4_SAT(((DATA4)thre_shift  > absdata) ? (DATA4)motion_shift : (DATA4)0);
    return AS_FLAG4U(mask_motion) | AS_FLAG4U(mask_shift);
}

Flag4U analyze_stripe(DATA4 p0, DATA4 p1, Flag flag_sign, Flag flag_deint, Flag flag_shift, const DATA thre_deint, const DATA thre_shift) {
    DATA4 absdata = abs_diff(p0, p1);
    Flag4 new_sign   = CONVERT_FLAG4_SAT((p0 >= p1)                    ? (DATA4)flag_sign  : (DATA4)0);
    Flag4 mask_deint = CONVERT_FLAG4_SAT((absdata > (DATA4)thre_deint) ? (DATA4)flag_deint : (DATA4)0);
    Flag4 mask_shift = CONVERT_FLAG4_SAT((absdata > (DATA4)thre_shift) ? (DATA4)flag_shift : (DATA4)0);
    return AS_FLAG4U(new_sign) | AS_FLAG4U(mask_deint) | AS_FLAG4U(mask_shift);
}

DATA4 load_y(__read_only image2d_t src, int ix, int iy) {
    #if LOAD_IMAGEF
        return CONVERT_DATA4(read_imagef(src, sampler_y, (int2)(ix, iy)) * (float4)((1<<BIT_DEPTH)-1) + (float4)0.5f);
    #else
        return CONVERT_DATA4(read_imageui(src, sampler_y, (int2)(ix, iy)));
    #endif
}

Flag4U analyze_y(
    __read_only image2d_t src_p0,
    __read_only image2d_t src_p1,
    int ix, int iy,
    DATA thre_motion, DATA thre_deint, DATA thre_shift) {

    //motion
    DATA4 p0 = load_y(src_p0, ix, iy);
    DATA4 p1 = load_y(src_p1, ix, iy);
    DATA4 p2 = p1;
    Flag4U flag = analyze_motion(p0, p1, thre_motion, thre_shift);

    if (iy >= 1) {
        //non-shift
        p1 = load_y(src_p0, ix, iy-1);
        flag |= analyze_stripe(p0, p1, non_shift_sign, non_shift_deint, non_shift_shift, thre_deint, thre_shift);

        //shift
        if (TB_ORDER) {
            if (iy & 1) {
                p0 = p2;
            } else {
                p1 = load_y(src_p1, ix, iy-1);
            }
        } else {
            if (iy & 1) {
                p1 = load_y(src_p1, ix, iy-1);
            } else {
                p0 = p2;
            }
        }
        flag |= analyze_stripe(p1, p0, shift_sign, shift_deint, shift_shift, thre_deint, thre_shift);
    }
    return flag;
}

#if YUV420
// 4:2:0の色差を、インターレースを考慮して輝度解像度(4:4:4相当)へ整数演算で拡大する。
// 以前はanalyzeカーネル内で色差をread_imagefの線形補間で毎回求めていたが、
// その結果をfloatのスカラーで判定するため、輝度のような4画素同時処理ができず、
// 読み込み数と命令数がanalyzeの大半を占めていた。
// ここで1フレームにつき1回だけ拡大しておき、analyzeでは輝度と同じ4画素単位の処理に乗せる。
// 重みは従来の線形補間と同じ:
//   縦: 輝度の行iy=4m+r に対し、同じフィールド(iy&1)の色差の行Fを使う (重みは1/8単位)
//       r=0: F0[m-1]*1 + F0[m]*7, r=1: F1[m-1]*3 + F1[m]*5,
//       r=2: F0[m]*5 + F0[m+1]*3, r=3: F1[m]*7 + F1[m+1]*1   (F0=偶数行, F1=奇数行, 端はclamp)
//   横: 輝度の偶数画素は色差の画素そのまま、奇数画素は左右の色差の平均
// 丸めは1回だけ行う (floatの値との差は許容する)。
// 入力の色差はソースキャッシュでフィールド分離済み (src_x0=偶数行, src_x1=奇数行、高さはfield_height)
// 1work itemで色差4画素(輝度8画素)を処理する。get_global_id(2)で 0:U, 1:V を選ぶ
__kernel void kernel_afs_upsample_uv420(
    __global uchar *restrict dst_u,
    __global uchar *restrict dst_v,
    const int dst_pitch,
    __global const uchar *restrict src_u0,
    __global const uchar *restrict src_u1,
    __global const uchar *restrict src_v0,
    __global const uchar *restrict src_v1,
    const int src_pitch,
    const int uv_width, const int field_height, const int height) {
    const int cx = get_global_id(0) * 4; //色差のx
    const int iy = get_global_id(1);     //輝度のy
    const int plane = get_global_id(2);  //0:U, 1:V
    if (cx >= uv_width || iy >= height) return;

    const int field = iy & 1;
    __global const uchar *src = (plane) ? ((field) ? src_v1 : src_v0) : ((field) ? src_u1 : src_u0);
    __global uchar *dst = (plane) ? dst_v : dst_u;
    const int m = iy >> 2;
    const int r = iy & 3;
    const int fa = (r < 2) ? m - 1 : m;
    const int wb = 7 - 2 * r; // r=0:7, 1:5, 2:3, 3:1
    const int wa = 8 - wb;
    const int ya = clamp(fa,     0, field_height - 1);
    const int yb = clamp(fa + 1, 0, field_height - 1);
    __global const DATA *line_a = (__global const DATA *)(src + ya * src_pitch);
    __global const DATA *line_b = (__global const DATA *)(src + yb * src_pitch);

    //縦補間した値 (8倍) を5画素分求める。右端より先はpitch内のゴミなので、最後の有効画素で置き換える
    const int4 v4 = (int4)wa * convert_int4(vload4(0, line_a + cx)) + (int4)wb * convert_int4(vload4(0, line_b + cx));
    const int cx4 = min(cx + 4, uv_width - 1);
    int v[5];
    v[0] = v4.s0;
    v[1] = (cx + 1 < uv_width) ? v4.s1 : v[0];
    v[2] = (cx + 2 < uv_width) ? v4.s2 : v[1];
    v[3] = (cx + 3 < uv_width) ? v4.s3 : v[2];
    v[4] = (cx + 4 < uv_width) ? (wa * (int)line_a[cx4] + wb * (int)line_b[cx4]) : v[3];

    int8 out;
    out.s0 = (v[0] + 4) >> 3; out.s1 = (v[0] + v[1] + 8) >> 4;
    out.s2 = (v[1] + 4) >> 3; out.s3 = (v[1] + v[2] + 8) >> 4;
    out.s4 = (v[2] + 4) >> 3; out.s5 = (v[2] + v[3] + 8) >> 4;
    out.s6 = (v[3] + 4) >> 3; out.s7 = (v[3] + v[4] + 8) >> 4;
    vstore8(CONVERT_DATA8(out), 0, (__global DATA *)(dst + iy * dst_pitch) + cx * 2);
}
#endif //#if YUV420

int shared_int_idx(int x, int y, int dep) {
    return dep * SHARED_INT_X * SHARED_Y + (y&15) * SHARED_INT_X + x;
}

void count_flags_skip(Flag4U dat0, Flag4U dat1, Flag4U *restrict count_deint, Flag4U *restrict count_shift) {
    Flag4U deint, shift, mask;
    Flag4U new_count_deint = AS_FLAG4U(*count_deint);
    Flag4U new_count_shift = AS_FLAG4U(*count_shift);
    mask = as_uint((dat0 ^ dat1) & u8x4(non_shift_sign  | shift_sign));
    deint = as_uint(dat0         & u8x4(non_shift_deint | shift_deint));
    shift = as_uint(dat0         & u8x4(non_shift_shift | shift_shift));
    mask >>= 1;
    //最初はshiftの位置にしかビットはたっていない
    //deintにマスクはいらない
    //new_count_deint &= mask;
    new_count_shift &= mask;
    new_count_deint  = deint; //deintに値は入っていないので代入でよい
    new_count_shift += shift;
    *count_deint = AS_FLAG4U(new_count_deint);
    *count_shift = AS_FLAG4U(new_count_shift);
}

void count_flags(Flag4U dat0, Flag4U dat1, Flag4U *restrict count_deint, Flag4U *restrict count_shift) {
    Flag4U deint, shift, mask;
    Flag4U new_count_deint = AS_FLAG4U(*count_deint);
    Flag4U new_count_shift = AS_FLAG4U(*count_shift);
    mask = as_uint((dat0 ^ dat1) & u8x4(non_shift_sign  | shift_sign));
    deint = as_uint(dat0         & u8x4(non_shift_deint | shift_deint));
    shift = as_uint(dat0         & u8x4(non_shift_shift | shift_shift));
    mask |= (mask << 1);
    mask |= (mask >> 2);
    new_count_deint &= mask;
    new_count_shift &= mask;
    new_count_deint += deint;
    new_count_shift += shift;
    *count_deint = AS_FLAG4U(new_count_deint);
    *count_shift = AS_FLAG4U(new_count_shift);
}

Flag4U generate_flags(int ly, int idepth, __local Flag4U *restrict ptr_shared) {
    Flag4U count_deint = 0;
    Flag4U dat0, dat1;

    //sharedメモリはあらかじめ-4もデータを作ってあるので、問題なく使用可能
    dat1 = ptr_shared[shared_int_idx(0, ly-3, idepth)];
    Flag4U count_shift = dat1 & u8x4(non_shift_shift | shift_shift);

    dat0 = ptr_shared[shared_int_idx(0, ly-2, idepth)];
    count_flags_skip(dat0, dat1, &count_deint, &count_shift);

    dat1 = ptr_shared[shared_int_idx(0, ly-1, idepth)];
    count_flags(dat1, dat0, &count_deint, &count_shift);

    dat0 = ptr_shared[shared_int_idx(0, ly+0, idepth)];
    count_flags(dat0, dat1, &count_deint, &count_shift);

    //      7       6         5        4        3        2        1       0
    // | motion  |         non-shift        | motion  |          shift          |
    // |  shift  |  sign  |  shift |  deint |  flag   | sign  |  shift |  deint |
    //motion 0x8888 -> 0x4444 とするため右シフト
    Flag4 flag0 = AS_FLAG4((dat0 & u8x4(motion_flag | motion_shift)) >> 1); //motion flag / motion shift

    //nonshift deint - countbit:654 / setbit 0x01
    //if ((count_deint & (0x70u << 0)) > (2u<<(4+ 0))) flag0 |= 0x01u<< 0; //nonshift deint(0)
    //if ((count_deint & (0x70u << 8)) > (2u<<(4+ 8))) flag1 |= 0x01u<< 8; //nonshift deint(1)
    //if ((count_deint & (0x70u <<16)) > (2u<<(4+16))) flag0 |= 0x01u<<16; //nonshift deint(2)
    //if ((count_deint & (0x70u <<24)) > (2u<<(4+24))) flag1 |= 0x01u<<24; //nonshift deint(3)
    Flag4 flag1 = (AS_FLAG4(count_deint & u8x4(0x70u)) > (Flag4)(2u << 4)) ? (Flag4)(0x01u) : (Flag4)0; //nonshift deint
    //nonshift shift - countbit:765 / setbit 0x10
    //if ((count_shift & (0xE0u << 0)) > (3u<<(5+ 0))) flag0 |= 0x01u<< 4; //nonshift shift(0)
    //if ((count_shift & (0xE0u << 8)) > (3u<<(5+ 8))) flag1 |= 0x01u<<12; //nonshift shift(1)
    //if ((count_shift & (0xE0u <<16)) > (3u<<(5+16))) flag0 |= 0x01u<<20; //nonshift shift(2)
    //if ((count_shift & (0xE0u <<24)) > (3u<<(5+24))) flag1 |= 0x01u<<28; //nonshift shift(3)
    Flag4 flag2 = (AS_FLAG4(count_shift & u8x4(0xE0u)) > (Flag4)(3u << 5)) ? (Flag4)(0x10u) : (Flag4)0; //nonshift shift
    //shift deint - countbit:210 / setbit 0x02
    //if ((count_deint & (0x07u << 0)) > (2u<<(0+ 0))) flag0 |= 0x01u<< 1; //shift deint(0)
    //if ((count_deint & (0x07u << 8)) > (2u<<(0+ 8))) flag1 |= 0x01u<< 9; //shift deint(1)
    //if ((count_deint & (0x07u <<16)) > (2u<<(0+16))) flag0 |= 0x01u<<17; //shift deint(2)
    //if ((count_deint & (0x07u <<24)) > (2u<<(0+24))) flag1 |= 0x01u<<25; //shift deint(3)
    Flag4 flag3 = (AS_FLAG4(count_deint & u8x4(0x07u)) > (Flag4)(2u << 0)) ? (Flag4)(0x02u) : (Flag4)0; //shift deint
    //shift shift - countbit:321 / setbit 0x20
    //if ((count_shift & (0x0Eu << 0)) > (3u<<(1+ 0))) flag0 |= 0x01u<< 5; //shift shift(0)
    //if ((count_shift & (0x0Eu << 8)) > (3u<<(1+ 8))) flag1 |= 0x01u<<13; //shift shift(1)
    //if ((count_shift & (0x0Eu <<16)) > (3u<<(1+16))) flag0 |= 0x01u<<21; //shift shift(2)
    //if ((count_shift & (0x0Eu <<24)) > (3u<<(1+24))) flag1 |= 0x01u<<29; //shift shift(3)
    Flag4 flag4 = (AS_FLAG4(count_shift & u8x4(0x0Eu)) > (Flag4)(3u << 1)) ? (Flag4)(0x20u) : (Flag4)0; //shift shift

    return AS_FLAG4U(flag0) | AS_FLAG4U(flag1) | AS_FLAG4U(flag2) | AS_FLAG4U(flag3) | AS_FLAG4U(flag4);
}

void merge_mask(Flag4U masky, Flag4U masku, Flag4U maskv, Flag4U *restrict mask0, Flag4U *restrict mask1) {
    // tune_modeの値に基づいてビット演算でマスクを選択
    // 各ビット位置で選択するマスクを決定
    if (SELECT_PLANE != 0) {
        const Flag4U select_y = (SELECT_PLANE & SELECT_PLANE_Y) ? 0xffffffff : 0x00;
        const Flag4U select_u = (SELECT_PLANE & SELECT_PLANE_U) ? 0xffffffff : 0x00;
        const Flag4U select_v = (SELECT_PLANE & SELECT_PLANE_V) ? 0xffffffff : 0x00;

        *mask0 = (masky | (~select_y)) & (masku | (~select_u)) & (maskv | (~select_v));
        *mask1 = (masky & select_y)    | (masku & select_u)    | (maskv & select_v);
    } else {
        *mask0 = masky & masku & maskv;
        *mask1 = masky | masku | maskv;
    }

    *mask0 &= u8x4(0xcc); //motion
    *mask1 &= u8x4(0x33); //shift/deint

    *mask0 |= *mask1;
}

//色差はyuv420の場合もkernel_afs_upsample_uv420で輝度解像度に拡大したものを受け取り、輝度と同じ4画素単位の処理で判定する
__kernel void kernel_afs_analyze_12(
    __global Flag4 *restrict ptr_dst,
    __global int *restrict ptr_count,
    __read_only image2d_t src_p0y,
    __read_only image2d_t src_p0u,
    __read_only image2d_t src_p0v,
    __read_only image2d_t src_p1y,
    __read_only image2d_t src_p1u,
    __read_only image2d_t src_p1v,
    const int width_int, const int si_pitch_int, const int h,
    const DATA thre_Ymotion, const DATA thre_deint, const DATA thre_shift,
    const DATA thre_Cmotion, const DATA thre_Cdeint, const DATA thre_Cshift,
    const uint scan_left, const uint scan_top, const uint scan_width, const uint scan_height) {

    __local int shared[SHARED_INT_X * SHARED_Y * 5]; //int単位でアクセスする
    const int lx = get_local_id(0); //スレッド数=BLOCK_INT_X
    int ly = get_local_id(1); //スレッド数=BLOCK_Y
    const int gidy = get_group_id(1); //グループID
    const int imgx = get_group_id(0) * BLOCK_INT_X /*blockDim.x*/ + lx;
    int imgy = (gidy * BLOCK_LOOP_Y * BLOCK_Y + ly);
    const int imgy_block_fin = min(h, ((gidy + 1) * BLOCK_LOOP_Y) * BLOCK_Y);
    uint motion_count = 0;

#define CALL_ANALYZE_Y(p0, p1, y_offset) analyze_y((p0), (p1), (imgx), (imgy+(y_offset)), thre_Ymotion,  thre_deint,  thre_shift)
#define CALL_ANALYZE_C(p0, p1, y_offset) analyze_y((p0), (p1), (imgx), (imgy+(y_offset)), thre_Cmotion, thre_Cdeint, thre_Cshift)

    __local Flag4U *ptr_shared = (__local Flag4U *)shared + shared_int_idx(lx,0,0);
    ptr_dst += (imgy-4) * si_pitch_int + imgx;

    //前の4ライン分、計算しておく
    //sharedの SHARED_Y-4 ～ SHARED_Y-1 (ブロック直前の4行) と、0 ～ 3 (ブロック先頭の4行) を埋める
    //generate_flags(ly) は ly-3 ～ ly を参照するため、ブロック直前の4行が未設定だと、
    //未初期化のlocalメモリを読んでブロック境界 (128行ごと) の判定が不定になる
    for (int iy = ly; iy < 8; iy += BLOCK_Y) {
        const int y_offset = iy - ly - 4; //ブロック先頭からの相対位置 -4 ～ 3
        const bool in_frame = imgy + y_offset >= 0; //フレーム上端より上は判定なし(0)とする
        ptr_shared[shared_int_idx(0, iy-4, 0)] = (in_frame) ? CALL_ANALYZE_Y(src_p0y, src_p1y, y_offset) : 0;
        ptr_shared[shared_int_idx(0, iy-4, 1)] = (in_frame) ? CALL_ANALYZE_C(src_p0u, src_p1u, y_offset) : 0;
        ptr_shared[shared_int_idx(0, iy-4, 2)] = (in_frame) ? CALL_ANALYZE_C(src_p0v, src_p1v, y_offset) : 0;
    }

    for (int iloop = 0; iloop <= BLOCK_LOOP_Y; iloop++,
        ptr_dst += BLOCK_Y * si_pitch_int, imgy += BLOCK_Y,
        ly += BLOCK_Y
    ) {
        { //差分情報を計算
            ptr_shared[shared_int_idx(0, ly+4, 0)] = CALL_ANALYZE_Y(src_p0y, src_p1y, 4);
            ptr_shared[shared_int_idx(0, ly+4, 1)] = CALL_ANALYZE_C(src_p0u, src_p1u, 4);
            ptr_shared[shared_int_idx(0, ly+4, 2)] = CALL_ANALYZE_C(src_p0v, src_p1v, 4);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        Flag4U mask1;
        { //マスク生成
            Flag4U masky = generate_flags(ly, 0, ptr_shared);
            Flag4U masku = generate_flags(ly, 1, ptr_shared);
            Flag4U maskv = generate_flags(ly, 2, ptr_shared);
            Flag4U mask0;
            merge_mask(masky, masku, maskv, &mask0, &mask1);
            ptr_shared[shared_int_idx(0, ly, 3)] = mask0;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        { //最終出力
            //ly+4とか使っているので準備ができてないうちから、次の列のデータを使うことになってまずい
            Flag4U mask4, mask5, mask6, mask7;
            mask4 = ptr_shared[shared_int_idx(0, ly-1, 3)];
            mask5 = ptr_shared[shared_int_idx(0, ly-2, 3)];
            mask6 = ptr_shared[shared_int_idx(0, ly-3, 3)];
            mask7 = ptr_shared[shared_int_idx(0, ly-4, 3)];
            mask1 &= u8x4(0x30);
            mask4 |= mask5 | mask6;
            mask4 &= u8x4(0x33);
            mask1 |= mask4 | mask7;
            if (imgx < width_int && (imgy - 4) < imgy_block_fin && ly - 4 >= 0) {
                //motion_countの実行
                if ((((uint)imgx - scan_left) < scan_width) && (((uint)(imgy - 4) - scan_top) < scan_height)) {
                    motion_count += popcount((~mask1) & u8x4(0x40)); //opencl版を変更、xorしてからマスク
                }
                //判定結果の出力
                ptr_dst[0] = AS_FLAG4(mask1);
            }
            //次に書き換えるのは(x,y,0)(x,y,1)(x,y,2)なので、(x,y,3)の読み込みと同時に行うことができる
            //ここでの同期は不要
            //__syncthreads()
        }
    }


    //motion countの総和演算
    // 32               16              0
    //  |  count_latter ||  count_first |
    int motion_count_01;
    //static_assert(BLOCK_INT_X * sizeof(int) * BLOCK_Y * BLOCK_LOOP_Y < (1<<(sizeof(short)*8-1)), "reduce block size for proper reduction in 16bit.");
    if (TB_ORDER) {
        motion_count_01 = (int)(( ly      & 1) ? (uint)motion_count << 16 : (uint)motion_count);
    } else {
        motion_count_01 = (int)(((ly + 1) & 1) ? (uint)motion_count << 16 : (uint)motion_count);
    }
    __local int *ptr_reduction = (__local int *)shared;
	motion_count_01 = block_sum_int(motion_count_01, ptr_reduction);
    const int lid = get_local_id(1) * BLOCK_INT_X + get_local_id(0);
    if (lid == 0) {
        const int gid = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        ptr_count[gid] = motion_count_01;
    }
}
