// -----------------------------------------------------------------------------------------
//     QSVEnc/VCEEnc/rkmppenc by rigaya
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
// IABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_OPENVINO_H__
#define __RGY_OPENVINO_H__

#include <memory>
#include <cstddef>
#include "rgy_err.h"
#include "rgy_version.h"

// Thin wrapper over the OpenVINO C API. The entire per-network "graph build" is a single
// read_model() of an ONNX (or OpenVINO IR) file; one inference is set_tensor +
// infer. This is the whole point of the onnx experiment: this one class
// replaces the per-family build*()/forward*() oneDNN graphs in the native
// anime4k filter. OpenVINO symbols are resolved from the OpenVINO C runtime
// library at runtime, so the executable does not import openvino_c.lib directly.
class RGYOpenVINO {
public:
    RGYOpenVINO();
    ~RGYOpenVINO();

    // Parse a model (no compile) and report its input/output channel counts, so
    // the caller can choose a backend before committing to a compile. A channel
    // count that is not statically known is returned as -1. Cheap (parse only).
    RGY_ERR peekChannels(const tstring &modelPath, int &inChannels, int &outChannels,
                         tstring &errMessage);

    // Load an ONNX/IR model, reshape its input to [1, channels, height, width]
    // (channels taken from the model itself), and compile it for the given
    // device ("GPU.0", "GPU", "CPU", "AUTO", ...). On failure, errMessage is
    // filled with the OpenVINO exception text.
    // OpenVINOのCACHE_DIRを設定 (コンパイル済みモデルをキャッシュし、次回以降の起動を高速化)
    // 空文字 = キャッシュ無効 (従来動作)。init系の呼び出し前に設定すること。
    void setCacheDir(const tstring &dir);

    RGY_ERR init(const tstring &modelPath, const tstring &device,
                 const int height, const int width, tstring &errMessage,
                 const tstring &precision = _T("auto"));

    // Compile the model inside an OpenVINO GPU remote context created from the
    // OpenCL queue/context selected by QSVEnc. This pins OpenVINO GPU execution
    // to the same physical GPU without relying on GPU.N enumeration order.
    RGY_ERR initFromOpenCLQueue(const tstring &modelPath, void *clQueue, void *clContext,
                                const int height, const int width, tstring &errMessage,
                                const tstring &precision = _T("auto"));

    // Fallback helper for environments where remote context creation is not
    // available. Returns "GPU.N" when an OpenVINO GPU with matching UUID/LUID is
    // found, or an empty string otherwise.
    tstring findDeviceByUuidLuid(const void *uuid, const size_t uuidSize,
                                 const void *luid, const size_t luidSize,
                                 tstring &errMessage);

    // Synchronous inference. in points to inChannels()*inHeight()*inWidth()
    // floats (CHW); out receives outChannels()*outHeight()*outWidth() floats
    // (CHW). in must stay valid for the duration of the call (it does: infer()
    // is blocking).
    RGY_ERR infer(const float *in, float *out);

    // --- OpenCL共有キューを使うzero-copy経路 ---
    // 指定されたcontext/queueでモデルをコンパイルし、同じキュー上のOpenCLカーネルと
    // OpenVINO推論をホスト転送なしで直列化する。
    RGY_ERR initShared(const tstring &modelPath, void *clQueue, void *clContext,
                       const int height, const int width, tstring &errMessage,
                       const tstring &precision = _T("auto"));
    // 入出力shapeと同じ大きさのf32 cl_memをremote tensorとして一度だけバインドする。
    RGY_ERR setSharedIO(void *inClMem, void *outClMem);
    // バインド済みremote tensorを使って推論する。
    RGY_ERR inferShared();
    bool usingSharedContext() const;

    int inChannels()  const;
    int inHeight()    const;
    int inWidth()     const;
    int outChannels() const;
    int outHeight()   const;
    int outWidth()    const;
    size_t outElemCount() const; // outChannels()*outHeight()*outWidth()

    tstring deviceFullName() const;     // e.g. "Intel(R) Arc(TM) A770 Graphics (dGPU)"
    tstring inferencePrecision() const; // e.g. "f16"

    static bool available();
    static tstring availabilityStatus();
    static tstring runtimeVersion(); // OpenVINOランタイムのビルド番号 (例: "2025.2.0-XXXXX-abcdef"); 取得不可なら空文字

private:
    RGYOpenVINO(const RGYOpenVINO &) = delete;
    void operator=(const RGYOpenVINO &) = delete;

    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif //__RGY_OPENVINO_H__
