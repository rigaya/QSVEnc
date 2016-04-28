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
// --------------------------------------------------------------------------------------------

#ifndef _CONVERT_H_
#define _CONVERT_H_

#include <Windows.h>

//音声16bit->8bit変換
typedef void (*func_audio_16to8) (BYTE *dst, short *src, int n);

func_audio_16to8 get_audio_16to8_func(BOOL split); //使用する音声16bit->8bit関数の選択

void convert_audio_16to8(BYTE *dst, short *src, int n);
void convert_audio_16to8_sse2(BYTE *dst, short *src, int n);

void split_audio_16to8x2(BYTE *dst, short *src, int n);
void split_audio_16to8x2_sse2(BYTE *dst, short *src, int n);

#if (_MSC_VER >= 1700)
void convert_audio_16to8_avx2(BYTE *dst, short *src, int n);
void split_audio_16to8x2_avx2(BYTE *dst, short *src, int n);
#endif

#endif //_CONVERT_H_
