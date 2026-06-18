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

#include "rgy_openvino.h"

#if ENABLE_OPENVINO

#include <cstring>
#include <vector>

// OpenVINO's GPU interop header (ocl.hpp -> ocl_wrapper.hpp) pulls in the
// Khronos OpenCL C++ bindings. Use the modern single header CL/opencl.hpp
// (vendored alongside the OpenCL C headers in $(OPENCL_HEADERS)) and pin the
// API versions so the C headers do not warn and the wrapper accepts them.
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 300
#endif
#ifndef CL_HPP_MINIMUM_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#endif
#define OV_GPU_USE_OPENCL_HPP

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4244 4267 4127 4100)
#endif
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

class RGYOpenVINO::Impl {
public:
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled;
    ov::InferRequest req;
    ov::Shape inShape;
    ov::Shape outShape;
    std::string devName;
    std::string precision;
    std::unique_ptr<ov::intel_gpu::ocl::ClContext> remoteCtx; // zero-copy path
    ov::Tensor inRemote;
    ov::Tensor outRemote;
    bool shared = false;
};

// the model channel count (static dim 1) if present, else 1
static int64_t modelInputChannels(const std::shared_ptr<ov::Model> &model) {
    const auto inPart = model->input().get_partial_shape();
    return (inPart.rank().is_static() && inPart.size() >= 2 && inPart[1].is_static())
        ? inPart[1].get_length() : 1;
}

RGYOpenVINO::RGYOpenVINO() : m_impl(std::make_unique<Impl>()) {}
RGYOpenVINO::~RGYOpenVINO() {}

static int staticChannels(const ov::PartialShape &p) {
    return (p.rank().is_static() && p.size() >= 2 && p[1].is_static()) ? (int)p[1].get_length() : -1;
}

RGY_ERR RGYOpenVINO::peekChannels(const std::string &modelPath, int &inChannels, int &outChannels,
                                  std::string &errMessage) {
    try {
        auto tmp = m_impl->core.read_model(modelPath);
        inChannels  = staticChannels(tmp->input().get_partial_shape());
        outChannels = staticChannels(tmp->output().get_partial_shape());
    } catch (const std::exception &e) {
        errMessage = e.what();
        return RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOpenVINO::init(const std::string &modelPath, const std::string &device,
                          const int height, const int width, std::string &errMessage) {
    try {
        auto &I = *m_impl;

        // The entire "graph build" for ANY architecture: read the ONNX/IR file.
        I.model = I.core.read_model(modelPath);

        // Channel count comes from the model; only N/H/W are pinned. Mirrors the
        // python POC (ov_poc.py) so behaviour is identical.
        const auto inPart = I.model->input().get_partial_shape();
        const int64_t ch = (inPart.rank().is_static() && inPart.size() >= 2 && inPart[1].is_static())
            ? inPart[1].get_length() : 1;
        I.model->reshape(ov::PartialShape{ 1, ch, height, width });

        I.compiled = I.core.compile_model(I.model, device);
        I.req      = I.compiled.create_infer_request();
        I.inShape  = I.compiled.input().get_shape();
        I.outShape = I.compiled.output().get_shape();

        try { I.devName = I.core.get_property(device, ov::device::full_name); } catch (...) {}
        try { I.precision = I.compiled.get_property(ov::hint::inference_precision).to_string(); } catch (...) {}
    } catch (const std::exception &e) {
        errMessage = e.what();
        return RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOpenVINO::infer(const float *in, float *out) {
    try {
        auto &I = *m_impl;
        // Wrap the caller's host buffer directly as the input tensor. infer() is
        // synchronous, so the pointer stays valid for the whole call.
        ov::Tensor inT(ov::element::f32, I.inShape, const_cast<float *>(in));
        I.req.set_input_tensor(inT);
        I.req.infer();
        const ov::Tensor outT = I.req.get_output_tensor();
        std::memcpy(out, outT.data(), outT.get_byte_size());
    } catch (const std::exception &) {
        return RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOpenVINO::initShared(const std::string &modelPath, void *clQueue,
                                const int height, const int width, std::string &errMessage) {
    try {
        auto &I = *m_impl;
        I.model = I.core.read_model(modelPath);
        const int64_t ch = modelInputChannels(I.model);
        I.model->reshape(ov::PartialShape{ 1, ch, height, width });
        // Build a remote context from the caller's cl_command_queue and compile
        // the model on it. OpenVINO enqueues inference on that same queue, so on
        // an in-order queue it slots between the caller's surrounding kernels
        // with no host synchronisation and no host roundtrip.
        I.remoteCtx = std::make_unique<ov::intel_gpu::ocl::ClContext>(I.core, static_cast<cl_command_queue>(clQueue));
        I.compiled  = I.core.compile_model(I.model, *I.remoteCtx);
        I.req       = I.compiled.create_infer_request();
        I.inShape   = I.compiled.input().get_shape();
        I.outShape  = I.compiled.output().get_shape();
        I.shared    = true;
        try { I.devName = I.compiled.get_property(ov::device::full_name); } catch (...) {}
        try { I.precision = I.compiled.get_property(ov::hint::inference_precision).to_string(); } catch (...) {}
    } catch (const std::exception &e) {
        errMessage = e.what();
        return RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOpenVINO::setSharedIO(void *inClMem, void *outClMem) {
    try {
        auto &I = *m_impl;
        if (!I.remoteCtx) return RGY_ERR_NULL_PTR;
        I.inRemote  = I.remoteCtx->create_tensor(ov::element::f32, I.inShape,  static_cast<cl_mem>(inClMem));
        I.outRemote = I.remoteCtx->create_tensor(ov::element::f32, I.outShape, static_cast<cl_mem>(outClMem));
        I.req.set_input_tensor(I.inRemote);
        I.req.set_output_tensor(I.outRemote);
    } catch (const std::exception &) {
        return RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOpenVINO::inferShared() {
    try {
        m_impl->req.infer();
    } catch (const std::exception &) {
        return RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

bool RGYOpenVINO::usingSharedContext() const { return m_impl->shared; }

int RGYOpenVINO::inChannels()  const { return m_impl->inShape.size()  >= 4 ? (int)m_impl->inShape[1]  : 1; }
int RGYOpenVINO::inHeight()    const { return m_impl->inShape.size()  >= 4 ? (int)m_impl->inShape[2]  : 0; }
int RGYOpenVINO::inWidth()     const { return m_impl->inShape.size()  >= 4 ? (int)m_impl->inShape[3]  : 0; }
int RGYOpenVINO::outChannels() const { return m_impl->outShape.size() >= 4 ? (int)m_impl->outShape[1] : 1; }
int RGYOpenVINO::outHeight()   const { return m_impl->outShape.size() >= 4 ? (int)m_impl->outShape[2] : 0; }
int RGYOpenVINO::outWidth()    const { return m_impl->outShape.size() >= 4 ? (int)m_impl->outShape[3] : 0; }
size_t RGYOpenVINO::outElemCount() const {
    size_t n = 1;
    for (auto d : m_impl->outShape) n *= d;
    return n;
}
std::string RGYOpenVINO::deviceFullName()     const { return m_impl->devName; }
std::string RGYOpenVINO::inferencePrecision() const { return m_impl->precision; }

#else // !ENABLE_OPENVINO

class RGYOpenVINO::Impl {};
RGYOpenVINO::RGYOpenVINO() : m_impl(nullptr) {}
RGYOpenVINO::~RGYOpenVINO() {}
RGY_ERR RGYOpenVINO::peekChannels(const std::string &, int &inChannels, int &outChannels, std::string &errMessage) {
    inChannels = outChannels = 0;
    errMessage = "this build of QSVEnc does not include OpenVINO support";
    return RGY_ERR_UNSUPPORTED;
}
RGY_ERR RGYOpenVINO::init(const std::string &, const std::string &, const int, const int, std::string &errMessage) {
    errMessage = "this build of QSVEnc does not include OpenVINO support";
    return RGY_ERR_UNSUPPORTED;
}
RGY_ERR RGYOpenVINO::infer(const float *, float *) { return RGY_ERR_UNSUPPORTED; }
RGY_ERR RGYOpenVINO::initShared(const std::string &, void *, const int, const int, std::string &errMessage) {
    errMessage = "this build of QSVEnc does not include OpenVINO support";
    return RGY_ERR_UNSUPPORTED;
}
RGY_ERR RGYOpenVINO::setSharedIO(void *, void *) { return RGY_ERR_UNSUPPORTED; }
RGY_ERR RGYOpenVINO::inferShared() { return RGY_ERR_UNSUPPORTED; }
bool RGYOpenVINO::usingSharedContext() const { return false; }
int RGYOpenVINO::inChannels()  const { return 0; }
int RGYOpenVINO::inHeight()    const { return 0; }
int RGYOpenVINO::inWidth()     const { return 0; }
int RGYOpenVINO::outChannels() const { return 0; }
int RGYOpenVINO::outHeight()   const { return 0; }
int RGYOpenVINO::outWidth()    const { return 0; }
size_t RGYOpenVINO::outElemCount() const { return 0; }
std::string RGYOpenVINO::deviceFullName()     const { return std::string(); }
std::string RGYOpenVINO::inferencePrecision() const { return std::string(); }

#endif // ENABLE_OPENVINO
