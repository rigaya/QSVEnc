// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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
// -------------------------------------------------------------------------------------------

#include <cstdint>
#include "rgy_simd.h"
#include "scene_change_detection.h"
#include "scene_change_detection_simd.h"

#if defined(_MSC_VER) || defined(__AVX2__)
#define FUNC_AVX2(func) func
#else
#define FUNC_AVX2(func)
#endif

#if defined(_MSC_VER) || defined(__AVX__)
#define FUNC_AVX(func) func,
#else
#define FUNC_AVX(func)
#endif

static const func_make_hist_simd FUNC_MAKE_HIST_LIST[] = {
    make_hist_sse2, make_hist_sse41_popcnt, FUNC_AVX(make_hist_avx) FUNC_AVX2(make_hist_avx2)
};
func_make_hist_simd get_make_hist_func() {
    const uint32_t simd = get_availableSIMD();
    int index = ((simd & (SSE41|POPCNT)) == (SSE41|POPCNT)) + ((simd & (AVX|POPCNT)) == (AVX|POPCNT)) + ((simd & (AVX2|AVX|POPCNT)) == (AVX2|AVX|POPCNT));
    return FUNC_MAKE_HIST_LIST[index];
}

void make_hist_sse2(const uint8_t *frame_Y, hist_t *hist_buf, int y_start, int y_end, int y_step, int x_skip, int width, int pitch) {
    make_hist_simd(frame_Y, hist_buf, y_start, y_end, y_step, x_skip, width, pitch, SSE2);
}
void make_hist_sse41_popcnt(const uint8_t *frame_Y, hist_t *hist_buf, int y_start, int y_end, int y_step, int x_skip, int width, int pitch) {
    make_hist_simd(frame_Y, hist_buf, y_start, y_end, y_step, x_skip, width, pitch, POPCNT|SSE41|SSSE3|SSE3|SSE2);
}
