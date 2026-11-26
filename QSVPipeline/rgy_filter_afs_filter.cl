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

// MAPF_BLOCK_X (64) //work groupサイズ(x) = スレッド数/work group
// MAPF_ROWS    (16) //1スレッドが縦方向に処理する行数

#define u8x4(x)  (((uint)x) | (((uint)x) <<  8) | (((uint)x) << 16) | (((uint)x) << 24))

// ストライプマップに 横(1) -> 縦(1) -> 横(2) -> 縦(2) の順に3タップのフィルタをかける。
// 以前はlocalメモリに4段の中間結果を書き、barrierで同期しながら処理していたが、
// 1スレッドが1語(4画素)の列を担当し、縦方向はMAPF_ROWS行を直近の行だけレジスタに保持して順に処理するようにした。
// localメモリもbarrierも使わない。
// 横の隣接語は、各スレッドがグローバルメモリから直接読む (隣のスレッドと重複するがキャッシュに乗る)。
// 横のフィルタが隣の語から参照するのは、隣接する端の1バイトだけなので、
//   中央の語 : 左右の語の端のバイトを使う
//   左の語   : 右端のバイト(中央の語との境界)だけが必要 -> さらに左の語は0とする
//   右の語   : 左端のバイト(中央の語との境界)だけが必要 -> さらに右の語は0とする
// として、3語の読み込みで横(2)に必要な左右の端のバイトまで求められる。
// (旧実装はlocalメモリのタイルの右ハローを1語先から読んでいたため、タイル右端の語の結果が誤っていた。本実装で解消している)

// 横(1): 左右の画素のどちらかが立っていればshift/deint(0x03)を、両方立っていればsign(0x04)を立てる
uint filter_h_1(uint x0, uint x1, uint x2) {
    const uint x3 = x0 >> 24 | x1 <<  8;
    const uint x4 = x1 >>  8 | x2 << 24;
    return x1 | ((x3 | x4) & u8x4(0x03)) | ((x3 & x4) & u8x4(0x04));
}

// 縦(1): 上下の画素の両方が立っていれば0x07を立てる
uint filter_v_1(uint x0, uint x1, uint x2) {
    return x1 | (x0 & x2 & u8x4(0x03 | 0x04));
}

// 横(2): 左右の画素の両方が立っていなければ下位3bit(0x07)を落とす
uint filter_h_2(uint x0, uint x1, uint x2) {
    const uint x3 = x0 >> 24 | x1 <<  8;
    const uint x4 = x1 >>  8 | x2 << 24;
    return x1 & ((x3 & x4) | u8x4(0xf8));
}

// 縦(2): 上下の画素の両方が立っていなければ下位3bit(0x07)を落とす
uint filter_v_2(uint x0, uint x1, uint x2) {
    return x1 & ((x0 & x2) | u8x4(0xf8));
}

__kernel void kernel_afs_analyze_map_filter(
    __global uint *restrict ptr_dst,
    __global const uint *restrict ptr_src,
    const int si_w_type, const int pitch_type, const int height) {
    const int ix = get_global_id(0);
    const int y0 = get_global_id(1) * MAPF_ROWS;
    if (ix >= si_w_type) return;
    //左右の語は画面端でclampする (旧実装と同じ)
    const int ixl = max(ix - 1, 0);
    const int ixr = min(ix + 1, si_w_type - 1);

    //h1x: 横(1)の結果、v1x: 縦(1)の結果 (L:左の語, C:中央の語, R:右の語)、h2: 横(2)の結果
    //Prev2: 2行前, Prev: 1行前
    uint h1LPrev2 = 0, h1LPrev = 0, h1CPrev2 = 0, h1CPrev = 0, h1RPrev2 = 0, h1RPrev = 0;
    uint h2Prev2 = 0, h2Prev = 0;
    #pragma unroll
    for (int row = -2; row < MAPF_ROWS + 2; row++) {
        //上下の行は画面端でclampする (旧実装と同じ)
        __global const uint *src = ptr_src + clamp(y0 + row, 0, height - 1) * pitch_type;
        const uint rawL = src[ixl];
        const uint rawC = src[ix];
        const uint rawR = src[ixr];
        const uint h1L = filter_h_1(0,    rawL, rawC);
        const uint h1C = filter_h_1(rawL, rawC, rawR);
        const uint h1R = filter_h_1(rawC, rawR, 0);
        if (row >= 0) {
            //縦(1)・横(2)は1行前(row-1)の結果
            const uint v1L = filter_v_1(h1LPrev2, h1LPrev, h1L);
            const uint v1C = filter_v_1(h1CPrev2, h1CPrev, h1C);
            const uint v1R = filter_v_1(h1RPrev2, h1RPrev, h1R);
            const uint h2 = filter_h_2(v1L, v1C, v1R);
            //縦(2)は2行前(row-2)の結果
            const int y = y0 + row - 2;
            if (row >= 2 && y < height) {
                ptr_dst[y * pitch_type + ix] = filter_v_2(h2Prev2, h2Prev, h2);
            }
            h2Prev2 = h2Prev;
            h2Prev  = h2;
        }
        h1LPrev2 = h1LPrev; h1LPrev = h1L;
        h1CPrev2 = h1CPrev; h1CPrev = h1C;
        h1RPrev2 = h1RPrev; h1RPrev = h1R;
    }
}
