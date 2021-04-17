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

//Type
//MERGE_BLOCK_INT_X  (32) //work groupサイズ(x) = スレッド数/work group
//MERGE_BLOCK_Y       (8) //work groupサイズ(y) = スレッド数/work group
//MERGE_BLOCK_LOOP_Y  (1) //work groupのy方向反復数

#define u8x4(x)  (uint)(((uint)(x)) | (((uint)(x)) <<  8) | (((uint)(x)) << 16) | (((uint)(x)) << 24))

__kernel void kernel_afs_merge_scan(
    __global Type *__restrict__ ptr_dst,
    __global int *__restrict__ ptr_count,
    __global const Type *__restrict__ src_p0,
    __global const Type *__restrict__ src_p1,
    const int si_w_type, const int pitch_type, const int height,
    const int tb_order,
    const uint scan_left, const uint scan_top, const uint scan_width, const uint scan_height) {
    //int lx = get_local_id(0); //スレッド数=MERGE_BLOCK_INT_X
    int ly = get_local_id(1); //スレッド数=MERGE_BLOCK_Y
    int imgx = get_group_id(0) * MERGE_BLOCK_INT_X /*blockDim.x*/ + get_local_id(0);

    int stripe_count = 0;
    const int field_select = ((ly + tb_order) & 1);

    if (imgx < si_w_type) {
        const int mask_lsft = (field_select) ? 1 : 2;
        int imgy = get_group_id(1) * MERGE_BLOCK_LOOP_Y * MERGE_BLOCK_Y + ly;
        src_p0  += imgx + pitch_type * imgy;
        src_p1  += imgx + pitch_type * imgy;
        ptr_dst += imgx + pitch_type * imgy;
        #pragma unroll
        for (int iloop = 0; iloop < MERGE_BLOCK_LOOP_Y; iloop++,
            imgy    += MERGE_BLOCK_Y,
            ptr_dst += MERGE_BLOCK_Y * pitch_type,
            src_p0  += MERGE_BLOCK_Y * pitch_type,
            src_p1  += MERGE_BLOCK_Y * pitch_type
            ) {
            if (imgy < height) {
                const int offsetm = (imgy == 0       ) ? 0 : -pitch_type;
                const int offsetp = (imgy >= height-1) ? 0 :  pitch_type;
                Type p0m = src_p0[offsetm];
                Type p0c = src_p0[0];
                Type p0p = src_p0[offsetp];

                Type p1m = src_p1[offsetm];
                Type p1c = src_p1[0];
                Type p1p = src_p1[offsetp];

                Type m4 = (p0m | p0p | u8x4(0xf3)) & p0c;
                Type m5 = (p1m | p1p | u8x4(0xf3)) & p1c;

                Type m6 = (m4 & m5 & u8x4(0x44)) | (~(p0c) & u8x4(0x33));

                ptr_dst[0] = m6;

                if ((((uint)imgx - scan_left) < scan_width) && (((uint)imgy - scan_top) < scan_height)) {
                    //(m6 & MASK) == 0 を各バイトについて加算する
                    //MASK = 0x50 or 0x60 (ly, tb_order依存)
                    //まず、0x10 or 0x20 のビットを 0x40に集約する
                    Type check = m6 | ((m6) << mask_lsft);
                    //0x40のビットが1なら0、0なら1としたいので、xorでチェック
                    //他のビットはいらないので、自分とxorして消す
                    stripe_count += popcount(check ^ (check | u8x4(0x40)));
                }
            }
        }
    }
    //motion countの総和演算
    // 32               16              0
    //  |  count_latter ||  count_first |
    //static_assert(MERGE_BLOCK_INT_X * sizeof(int) * MERGE_BLOCK_Y * MERGE_BLOCK_LOOP_Y < (1<<(sizeof(short)*8-1)), "reduce block size for proper reduction in 16bit.");
    int stripe_count_01 = (int)(field_select ? (uint)stripe_count << 16 : (uint)stripe_count);

    __local int shared[MERGE_BLOCK_INT_X * MERGE_BLOCK_Y]; //int単位でアクセスする
    const int lid = get_local_id(1) * MERGE_BLOCK_INT_X + get_local_id(0);
    shared[lid] = stripe_count_01;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = MERGE_BLOCK_Y * MERGE_BLOCK_INT_X >> 1; offset > 0; offset >>= 1) {
        if (lid < offset) {
            shared[lid] += shared[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        const int gid = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        ptr_count[gid] = shared[0];
    }
}
