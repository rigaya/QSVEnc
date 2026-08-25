// -----------------------------------------------------------------------------------------
//     QSVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2019-2021 rigaya
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

// 縦2倍では元画素iを出力2iへ配置するため、プログレッシブ画像では半画素の
// 位置ずれが残る。転置は既存transformを使い、ここでは両軸処理後の位置を補正する。

#define NU_SRC(x, y) ((float)(*(__global const Type *)(pSrc \
    + (size_t)min(max((y), 0), srcH - 1) * srcPitch \
    + (size_t)min(max((x), 0), srcW - 1) * sizeof(Type))))

// 4タップcubic (-1/16, 9/16, 9/16, -1/16) で両軸を半画素補正する。
__kernel void kernel_nnedi_upscale_shift(
    __global uchar *restrict pDst, const int dstPitch,
    __global const uchar *restrict pSrc, const int srcPitch, const int srcW, const int srcH) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= srcW || y >= srcH) {
        return;
    }
    float col[4];
    for (int k = 0; k < 4; k++) {
        const int xx = x - 2 + k;
        col[k] = -0.0625f * NU_SRC(xx, y - 2)
               +  0.5625f * NU_SRC(xx, y - 1)
               +  0.5625f * NU_SRC(xx, y    )
               -  0.0625f * NU_SRC(xx, y + 1);
    }
    const float v = -0.0625f * col[0] + 0.5625f * col[1] + 0.5625f * col[2] - 0.0625f * col[3];
    const float maxval = (float)((1 << bit_depth) - 1);
    *(__global Type *)(pDst + (size_t)y * dstPitch + (size_t)x * sizeof(Type))
        = (Type)clamp(v + 0.5f, 0.0f, maxval);
}

// 2タップ平均による半画素補正。負のローブによるエッジ周辺のリンギングを避ける。
__kernel void kernel_nnedi_upscale_shift_linear(
    __global uchar *restrict pDst, const int dstPitch,
    __global const uchar *restrict pSrc, const int srcPitch, const int srcW, const int srcH) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= srcW || y >= srcH) {
        return;
    }
    const float v = 0.25f * (NU_SRC(x, y) + NU_SRC(x - 1, y) + NU_SRC(x, y - 1) + NU_SRC(x - 1, y - 1));
    const float maxval = (float)((1 << bit_depth) - 1);
    *(__global Type *)(pDst + (size_t)y * dstPitch + (size_t)x * sizeof(Type))
        = (Type)clamp(v + 0.5f, 0.0f, maxval);
}

