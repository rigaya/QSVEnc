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

#pragma once

#include "rgy_version.h"

#if ENABLE_VAAPI
#include "qsv_prm.h"
#include "rgy_libavcodec_vaapi.h"

// 入力の初期化後に、VA 非対応の指定と読み取った映像形式をまとめて検証する。
RGY_ERR qsvVACheckParam(sInputParams& prm, std::shared_ptr<RGYLog> log);

// 入力の初期化後に確定した出力情報を使い、QSV の指定を共通 VA パラメータへ変換する。
RGY_ERR qsvVAEncParam(RGYVAEncParam& dst, const sInputParams& prm, RGYDeviceVA *dev,
    int width, int height, rgy_rational<int> fps, rgy_rational<int> sar,
    rgy_rational<int> timebase, const VideoVUIInfo& vui, std::shared_ptr<RGYLog> log);
// 初期化済みのエンコーダ設定をQSVのログ書式に整形する。
tstring qsvVAEncInfo(const RGYEncoderVA& enc);
tstring qsvVAProfileString(const RGYEncoderVA& enc);
#endif
