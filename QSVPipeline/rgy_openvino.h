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

#include <string>
#include <memory>
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
    RGY_ERR peekChannels(const std::string &modelPath, int &inChannels, int &outChannels,
                         std::string &errMessage);

    // Load an ONNX/IR model, reshape its input to [1, channels, height, width]
    // (channels taken from the model itself), and compile it for the given
    // device ("GPU.0", "GPU", "CPU", "AUTO", ...). On failure, errMessage is
    // filled with the OpenVINO exception text.
    RGY_ERR init(const std::string &modelPath, const std::string &device,
                 const int height, const int width, std::string &errMessage);

    // Synchronous inference. in points to inChannels()*inHeight()*inWidth()
    // floats (CHW); out receives outChannels()*outHeight()*outWidth() floats
    // (CHW). in must stay valid for the duration of the call (it does: infer()
    // is blocking).
    RGY_ERR infer(const float *in, float *out);

    // --- zero-copy path (shared OpenCL queue) ---
    // Shared OpenCL zero-copy path. This public API is kept for callers, but
    // the dynamic C API implementation currently returns RGY_ERR_UNSUPPORTED
    // until the C remote-context varargs path is completed.
    RGY_ERR initShared(const std::string &modelPath, void *clQueue,
                       const int height, const int width, std::string &errMessage);
    // Bind the input and output cl_mem buffers (f32, sized to the in/out
    // shapes) as the inference request's remote tensors. Call once; the buffers
    // are reused every frame.
    RGY_ERR setSharedIO(void *inClMem, void *outClMem);
    // Run inference against the bound remote tensors (no host data moves).
    RGY_ERR inferShared();
    bool usingSharedContext() const;

    int inChannels()  const;
    int inHeight()    const;
    int inWidth()     const;
    int outChannels() const;
    int outHeight()   const;
    int outWidth()    const;
    size_t outElemCount() const; // outChannels()*outHeight()*outWidth()

    std::string deviceFullName() const;     // e.g. "Intel(R) Arc(TM) A770 Graphics (dGPU)"
    std::string inferencePrecision() const; // e.g. "f16"

    static bool available();
    static std::string availabilityStatus();

private:
    RGYOpenVINO(const RGYOpenVINO &) = delete;
    void operator=(const RGYOpenVINO &) = delete;

    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif //__RGY_OPENVINO_H__
