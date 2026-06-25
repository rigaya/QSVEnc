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

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rgy_tchar.h"
#include "rgy_util.h"
#include "rgy_osdep.h"
#include <openvino/c/openvino.h>

using ov_core_create_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(ov_core_t **);
using ov_core_free_t = void(OPENVINO_C_API_CALLBACK *)(ov_core_t *);
using ov_core_read_model_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_core_t *, const char *, const char *, ov_model_t **);
using ov_core_compile_model_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_core_t *, const ov_model_t *, const char *, const size_t, ov_compiled_model_t **, ...);
using ov_core_get_property_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_core_t *, const char *, const char *, char **);
using ov_core_create_context_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_core_t *, const char *, const size_t, ov_remote_context_t **, ...);
using ov_core_compile_model_with_context_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_core_t *, const ov_model_t *, const ov_remote_context_t *, const size_t, ov_compiled_model_t **, ...);
using ov_remote_context_free_t = void(OPENVINO_C_API_CALLBACK *)(ov_remote_context_t *);
using ov_remote_context_get_device_name_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_remote_context_t *, char **);
using ov_remote_context_create_host_tensor_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_remote_context_t *, const ov_element_type_e, const ov_shape_t, ov_tensor_t **);
using ov_model_free_t = void(OPENVINO_C_API_CALLBACK *)(ov_model_t *);
using ov_model_const_input_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_model_t *, ov_output_const_port_t **);
using ov_model_const_output_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_model_t *, ov_output_const_port_t **);
using ov_model_reshape_single_input_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_model_t *, const ov_partial_shape_t);
using ov_compiled_model_free_t = void(OPENVINO_C_API_CALLBACK *)(ov_compiled_model_t *);
using ov_compiled_model_create_infer_request_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_compiled_model_t *, ov_infer_request_t **);
using ov_compiled_model_input_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_compiled_model_t *, ov_output_const_port_t **);
using ov_compiled_model_output_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_compiled_model_t *, ov_output_const_port_t **);
using ov_compiled_model_get_property_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_compiled_model_t *, const char *, char **);
using ov_infer_request_free_t = void(OPENVINO_C_API_CALLBACK *)(ov_infer_request_t *);
using ov_infer_request_set_input_tensor_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(ov_infer_request_t *, const ov_tensor_t *);
using ov_infer_request_set_output_tensor_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(ov_infer_request_t *, const ov_tensor_t *);
using ov_infer_request_infer_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(ov_infer_request_t *);
using ov_infer_request_get_output_tensor_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_infer_request_t *, ov_tensor_t **);
using ov_tensor_create_from_host_ptr_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_element_type_e, const ov_shape_t, void *, ov_tensor_t **);
using ov_tensor_data_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_tensor_t *, void **);
using ov_tensor_get_byte_size_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_tensor_t *, size_t *);
using ov_tensor_free_t = void(OPENVINO_C_API_CALLBACK *)(ov_tensor_t *);
using ov_port_get_partial_shape_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_output_const_port_t *, ov_partial_shape_t *);
using ov_const_port_get_shape_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const ov_output_const_port_t *, ov_shape_t *);
using ov_output_const_port_free_t = void(OPENVINO_C_API_CALLBACK *)(ov_output_const_port_t *);
using ov_partial_shape_create_static_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(const int64_t, const int64_t *, ov_partial_shape_t *);
using ov_partial_shape_free_t = void(OPENVINO_C_API_CALLBACK *)(ov_partial_shape_t *);
using ov_shape_free_t = ov_status_e(OPENVINO_C_API_CALLBACK *)(ov_shape_t *);
using ov_get_error_info_t = const char *(OPENVINO_C_API_CALLBACK *)(ov_status_e);
using ov_get_last_err_msg_t = const char *(OPENVINO_C_API_CALLBACK *)();
using ov_free_t = void(OPENVINO_C_API_CALLBACK *)(const char *);

struct OpenVINOLoader {
    HMODULE module = nullptr;
    bool tried = false;
    bool ready = false;
    tstring error;
    std::mutex mtx;

    ov_core_create_t core_create = nullptr;
    ov_core_free_t core_free = nullptr;
    ov_core_read_model_t core_read_model = nullptr;
    ov_core_compile_model_t core_compile_model = nullptr;
    ov_core_get_property_t core_get_property = nullptr;
    ov_core_create_context_t core_create_context = nullptr;
    ov_core_compile_model_with_context_t core_compile_model_with_context = nullptr;
    ov_remote_context_free_t remote_context_free = nullptr;
    ov_remote_context_get_device_name_t remote_context_get_device_name = nullptr;
    ov_remote_context_create_host_tensor_t remote_context_create_host_tensor = nullptr;
    ov_model_free_t model_free = nullptr;
    ov_model_const_input_t model_const_input = nullptr;
    ov_model_const_output_t model_const_output = nullptr;
    ov_model_reshape_single_input_t model_reshape_single_input = nullptr;
    ov_compiled_model_free_t compiled_model_free = nullptr;
    ov_compiled_model_create_infer_request_t compiled_model_create_infer_request = nullptr;
    ov_compiled_model_input_t compiled_model_input = nullptr;
    ov_compiled_model_output_t compiled_model_output = nullptr;
    ov_compiled_model_get_property_t compiled_model_get_property = nullptr;
    ov_infer_request_free_t infer_request_free = nullptr;
    ov_infer_request_set_input_tensor_t infer_request_set_input_tensor = nullptr;
    ov_infer_request_set_output_tensor_t infer_request_set_output_tensor = nullptr;
    ov_infer_request_infer_t infer_request_infer = nullptr;
    ov_infer_request_get_output_tensor_t infer_request_get_output_tensor = nullptr;
    ov_tensor_create_from_host_ptr_t tensor_create_from_host_ptr = nullptr;
    ov_tensor_data_t tensor_data = nullptr;
    ov_tensor_get_byte_size_t tensor_get_byte_size = nullptr;
    ov_tensor_free_t tensor_free = nullptr;
    ov_port_get_partial_shape_t port_get_partial_shape = nullptr;
    ov_const_port_get_shape_t const_port_get_shape = nullptr;
    ov_output_const_port_free_t output_const_port_free = nullptr;
    ov_partial_shape_create_static_t partial_shape_create_static = nullptr;
    ov_partial_shape_free_t partial_shape_free = nullptr;
    ov_shape_free_t shape_free = nullptr;
    ov_get_error_info_t get_error_info = nullptr;
    ov_get_last_err_msg_t get_last_err_msg = nullptr;
    ov_free_t free_string = nullptr;

    ~OpenVINOLoader() {
        if (module) {
            RGY_FREE_LIBRARY(module);
            module = nullptr;
        }
    }

    template<typename Func>
    bool load(Func &func, const char *name) {
        func = reinterpret_cast<Func>(RGY_GET_PROC_ADDRESS(module, name));
        if (func == nullptr) {
            error = tstring(_T("OpenVINO runtime is missing required symbol: ")) + char_to_tstring(name);
            return false;
        }
        return true;
    }

    template<typename Func>
    void loadOptional(Func &func, const char *name) {
        func = reinterpret_cast<Func>(RGY_GET_PROC_ADDRESS(module, name));
    }

    bool load() {
        std::lock_guard<std::mutex> lock(mtx);
        if (tried) {
            return ready;
        }
        tried = true;

#if defined(_WIN32) || defined(_WIN64)
        static const std::array<const TCHAR *, 2> dllNames = {
            _T("openvino_c.dll"),
            _T("openvino_cd.dll")
        };
#else
        static const std::array<const TCHAR *, 5> dllNames = {
            _T("libopenvino_c.so"),
            _T("libopenvino_c.so.2600"),
            _T("libopenvino_c.so.2500"),
            _T("libopenvino_c.so.2025"),
            _T("libopenvino_c.so.2024")
        };
#endif
        for (const auto dllName : dllNames) {
            module = RGY_LOAD_LIBRARY(dllName);
            if (module != nullptr) {
                break;
            }
        }
        if (module == nullptr) {
            error = _T("OpenVINO C runtime library could not be loaded (openvino_c.dll/libopenvino_c.so)");
            return false;
        }

#define LOAD_OV(name) if (!load(name, "ov_" #name)) { RGY_FREE_LIBRARY(module); module = nullptr; return false; }
        LOAD_OV(core_create);
        LOAD_OV(core_free);
        LOAD_OV(core_read_model);
        LOAD_OV(core_compile_model);
        LOAD_OV(core_get_property);
        loadOptional(core_create_context, "ov_core_create_context");
        loadOptional(core_compile_model_with_context, "ov_core_compile_model_with_context");
        loadOptional(remote_context_free, "ov_remote_context_free");
        loadOptional(remote_context_get_device_name, "ov_remote_context_get_device_name");
        loadOptional(remote_context_create_host_tensor, "ov_remote_context_create_host_tensor");
        LOAD_OV(model_free);
        LOAD_OV(model_const_input);
        LOAD_OV(model_const_output);
        LOAD_OV(model_reshape_single_input);
        LOAD_OV(compiled_model_free);
        LOAD_OV(compiled_model_create_infer_request);
        LOAD_OV(compiled_model_input);
        LOAD_OV(compiled_model_output);
        LOAD_OV(compiled_model_get_property);
        LOAD_OV(infer_request_free);
        LOAD_OV(infer_request_set_input_tensor);
        loadOptional(infer_request_set_output_tensor, "ov_infer_request_set_output_tensor");
        LOAD_OV(infer_request_infer);
        LOAD_OV(infer_request_get_output_tensor);
        LOAD_OV(tensor_create_from_host_ptr);
        LOAD_OV(tensor_data);
        LOAD_OV(tensor_get_byte_size);
        LOAD_OV(tensor_free);
        LOAD_OV(port_get_partial_shape);
        LOAD_OV(const_port_get_shape);
        LOAD_OV(output_const_port_free);
        LOAD_OV(partial_shape_create_static);
        LOAD_OV(partial_shape_free);
        LOAD_OV(shape_free);
        LOAD_OV(get_error_info);
        LOAD_OV(get_last_err_msg);
        if (!load(free_string, "ov_free")) { RGY_FREE_LIBRARY(module); module = nullptr; return false; }
#undef LOAD_OV

        ready = true;
        return true;
    }

    tstring statusText(ov_status_e status) const {
        tstring text;
        if (get_last_err_msg) {
            if (const auto last = get_last_err_msg(); last && last[0]) {
                text = char_to_tstring(last);
            }
        }
        if (text.empty() && get_error_info) {
            if (const auto info = get_error_info(status); info && info[0]) {
                text = char_to_tstring(info);
            }
        }
        if (text.empty()) {
            text = _T("OpenVINO C API error: ") + char_to_tstring(std::to_string((int)status));
        }
        return text;
    }
};

static OpenVINOLoader &ovLoader() {
    static OpenVINOLoader loader;
    return loader;
}

template<typename T, typename FreeFunc>
using ov_unique_ptr = std::unique_ptr<T, FreeFunc>;

static RGY_ERR ensureOpenVINO(tstring &errMessage) {
    auto &ov = ovLoader();
    if (!ov.load()) {
        errMessage = ov.error;
        return RGY_ERR_UNSUPPORTED;
    }
    return RGY_ERR_NONE;
}

static RGY_ERR ovCheck(const ov_status_e status, tstring &errMessage) {
    if (status == OK) {
        return RGY_ERR_NONE;
    }
    errMessage = ovLoader().statusText(status);
    return RGY_ERR_UNKNOWN;
}

static int staticChannels(const ov_partial_shape_t &shape) {
    if (shape.rank.min == shape.rank.max && shape.rank.min >= 2 && shape.dims != nullptr) {
        const auto &dim = shape.dims[1];
        if (dim.min == dim.max) {
            return (int)dim.min;
        }
    }
    return -1;
}

static int64_t modelInputChannels(const ov_model_t *model, tstring &errMessage) {
    auto &ov = ovLoader();
    ov_output_const_port_t *portRaw = nullptr;
    if (ovCheck(ov.model_const_input(model, &portRaw), errMessage) != RGY_ERR_NONE) {
        return 1;
    }
    ov_unique_ptr<ov_output_const_port_t, ov_output_const_port_free_t> port(portRaw, ov.output_const_port_free);
    ov_partial_shape_t partialShape = {};
    if (ovCheck(ov.port_get_partial_shape(port.get(), &partialShape), errMessage) != RGY_ERR_NONE) {
        return 1;
    }
    const auto ch = staticChannels(partialShape);
    ov.partial_shape_free(&partialShape);
    return ch > 0 ? ch : 1;
}

static ov_shape_t makeShape(const std::vector<int64_t> &dims) {
    ov_shape_t shape = {};
    shape.rank = (int64_t)dims.size();
    shape.dims = const_cast<int64_t *>(dims.data());
    return shape;
}

static RGY_ERR getPortShape(ov_output_const_port_t *port, std::vector<int64_t> &shape, tstring &errMessage);

static std::string bytesToHex(const void *data, const size_t size) {
    if (data == nullptr || size == 0) {
        return std::string();
    }
    std::string str;
    const uint8_t *ptr = reinterpret_cast<const uint8_t *>(data);
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < size; i++) {
        str += hex[(ptr[i] >> 4) & 0x0f];
        str += hex[ptr[i] & 0x0f];
    }
    return str;
}

static std::string normalizeHexString(const std::string &str) {
    std::string out;
    for (auto c : str) {
        if (std::isxdigit((unsigned char)c)) {
            out += (char)std::tolower((unsigned char)c);
        }
    }
    return out;
}

static bool isZeroBytes(const void *data, const size_t size) {
    if (data == nullptr || size == 0) {
        return true;
    }
    const uint8_t *ptr = reinterpret_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; i++) {
        if (ptr[i] != 0) {
            return false;
        }
    }
    return true;
}

static std::vector<std::string> splitDeviceIds(const std::string &str) {
    std::vector<std::string> ids;
    std::string token;
    for (auto c : str) {
        if (std::isalnum((unsigned char)c) || c == '.' || c == '_' || c == '-') {
            token += c;
        } else if (!token.empty()) {
            ids.push_back(token);
            token.clear();
        }
    }
    if (!token.empty()) {
        ids.push_back(token);
    }
    return ids;
}

class RGYOpenVINO::Impl {
public:
    Impl() {
        tstring err;
        if (ensureOpenVINO(err) != RGY_ERR_NONE) {
            loadErr = err;
            return;
        }
        auto &ov = ovLoader();
        ov_core_t *coreRaw = nullptr;
        if (ov.core_create(&coreRaw) != OK) {
            loadErr = ov.statusText(UNKNOWN_C_ERROR);
            return;
        }
        core.reset(coreRaw);
    }

    RGY_ERR updateCompiledShapes(tstring &errMessage) {
        auto &ov = ovLoader();
        ov_output_const_port_t *inPortRaw = nullptr;
        auto ret = ovCheck(ov.compiled_model_input(compiled.get(), &inPortRaw), errMessage);
        if (ret != RGY_ERR_NONE) {
            return ret;
        }
        ov_unique_ptr<ov_output_const_port_t, ov_output_const_port_free_t> inPort(inPortRaw, ov.output_const_port_free);
        ret = getPortShape(inPort.get(), inShape, errMessage);
        if (ret != RGY_ERR_NONE) {
            return ret;
        }

        ov_output_const_port_t *outPortRaw = nullptr;
        ret = ovCheck(ov.compiled_model_output(compiled.get(), &outPortRaw), errMessage);
        if (ret != RGY_ERR_NONE) {
            return ret;
        }
        ov_unique_ptr<ov_output_const_port_t, ov_output_const_port_free_t> outPort(outPortRaw, ov.output_const_port_free);
        return getPortShape(outPort.get(), outShape, errMessage);
    }

    void resetRuntimeObjects() {
        req.reset();
        remoteInTensor.reset();
        remoteOutTensor.reset();
        compiled.reset();
        remote.reset();
        model.reset();
        inShape.clear();
        outShape.clear();
        devName.clear();
        devId.clear();
        precision.clear();
        shared = false;
    }

    struct CoreDeleter { void operator()(ov_core_t *p) const { if (p) ovLoader().core_free(p); } };
    struct ModelDeleter { void operator()(ov_model_t *p) const { if (p) ovLoader().model_free(p); } };
    struct CompiledDeleter { void operator()(ov_compiled_model_t *p) const { if (p) ovLoader().compiled_model_free(p); } };
    struct RequestDeleter { void operator()(ov_infer_request_t *p) const { if (p) ovLoader().infer_request_free(p); } };
    struct RemoteContextDeleter { void operator()(ov_remote_context_t *p) const { if (p && ovLoader().remote_context_free) ovLoader().remote_context_free(p); } };
    struct TensorDeleter { void operator()(ov_tensor_t *p) const { if (p) ovLoader().tensor_free(p); } };

    std::unique_ptr<ov_core_t, CoreDeleter> core;
    std::unique_ptr<ov_model_t, ModelDeleter> model;
    std::unique_ptr<ov_compiled_model_t, CompiledDeleter> compiled;
    std::unique_ptr<ov_infer_request_t, RequestDeleter> req;
    std::unique_ptr<ov_remote_context_t, RemoteContextDeleter> remote;
    std::unique_ptr<ov_tensor_t, TensorDeleter> remoteInTensor;
    std::unique_ptr<ov_tensor_t, TensorDeleter> remoteOutTensor;
    std::vector<int64_t> inShape;
    std::vector<int64_t> outShape;
    tstring devName;
    std::string devId;
    tstring precision;
    tstring loadErr;
    bool shared = false;
};

RGYOpenVINO::RGYOpenVINO() : m_impl(std::make_unique<Impl>()) {}
RGYOpenVINO::~RGYOpenVINO() {}

static RGY_ERR getPortShape(ov_output_const_port_t *port, std::vector<int64_t> &shape, tstring &errMessage) {
    auto &ov = ovLoader();
    ov_shape_t ovShape = {};
    auto ret = ovCheck(ov.const_port_get_shape(port, &ovShape), errMessage);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }
    shape.assign(ovShape.dims, ovShape.dims + ovShape.rank);
    ov.shape_free(&ovShape);
    return RGY_ERR_NONE;
}

static tstring getAndFreeProperty(char *value) {
    tstring str;
    if (value != nullptr) {
        str = char_to_tstring(value);
        ovLoader().free_string(value);
    }
    return str;
}

RGY_ERR RGYOpenVINO::peekChannels(const tstring &modelPath, int &inChannels, int &outChannels,
                                  tstring &errMessage) {
    auto ret = ensureOpenVINO(errMessage);
    if (ret != RGY_ERR_NONE) {
        inChannels = outChannels = 0;
        return ret;
    }
    if (!m_impl || !m_impl->core) {
        inChannels = outChannels = 0;
        errMessage = m_impl ? m_impl->loadErr : _T("OpenVINO runtime is not initialized");
        return RGY_ERR_UNSUPPORTED;
    }

    auto &ov = ovLoader();
    ov_model_t *modelRaw = nullptr;
    const auto modelPathA = tchar_to_string(modelPath, CP_UTF8);
    ret = ovCheck(ov.core_read_model(m_impl->core.get(), modelPathA.c_str(), nullptr, &modelRaw), errMessage);
    if (ret != RGY_ERR_NONE) {
        inChannels = outChannels = 0;
        return ret;
    }
    ov_unique_ptr<ov_model_t, ov_model_free_t> tmp(modelRaw, ov.model_free);

    auto getChannels = [&](bool input) {
        ov_output_const_port_t *portRaw = nullptr;
        auto status = input ? ov.model_const_input(tmp.get(), &portRaw) : ov.model_const_output(tmp.get(), &portRaw);
        if (status != OK) {
            errMessage = ov.statusText(status);
            return 0;
        }
        ov_unique_ptr<ov_output_const_port_t, ov_output_const_port_free_t> port(portRaw, ov.output_const_port_free);
        ov_partial_shape_t partialShape = {};
        status = ov.port_get_partial_shape(port.get(), &partialShape);
        if (status != OK) {
            errMessage = ov.statusText(status);
            return 0;
        }
        const auto ch = staticChannels(partialShape);
        ov.partial_shape_free(&partialShape);
        return ch;
    };
    inChannels = getChannels(true);
    outChannels = getChannels(false);
    return errMessage.empty() ? RGY_ERR_NONE : RGY_ERR_UNKNOWN;
}

RGY_ERR RGYOpenVINO::init(const tstring &modelPath, const tstring &device,
                          const int height, const int width, tstring &errMessage) {
    auto ret = ensureOpenVINO(errMessage);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }
    auto &I = *m_impl;
    if (!I.core) {
        errMessage = I.loadErr;
        return RGY_ERR_UNSUPPORTED;
    }
    auto &ov = ovLoader();
    I.resetRuntimeObjects();
    const auto modelPathA = tchar_to_string(modelPath, CP_UTF8);
    const auto deviceA = tchar_to_string(device, CP_UTF8);

    ov_model_t *modelRaw = nullptr;
    ret = ovCheck(ov.core_read_model(I.core.get(), modelPathA.c_str(), nullptr, &modelRaw), errMessage);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }
    I.model.reset(modelRaw);

    const int64_t ch = modelInputChannels(I.model.get(), errMessage);
    const int64_t dims[4] = { 1, ch, height, width };
    ov_partial_shape_t reshape = {};
    ret = ovCheck(ov.partial_shape_create_static(4, dims, &reshape), errMessage);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }
    ret = ovCheck(ov.model_reshape_single_input(I.model.get(), reshape), errMessage);
    ov.partial_shape_free(&reshape);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }

    ov_compiled_model_t *compiledRaw = nullptr;
    ret = ovCheck(ov.core_compile_model(I.core.get(), I.model.get(), deviceA.c_str(), 0, &compiledRaw), errMessage);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }
    I.compiled.reset(compiledRaw);

    ov_infer_request_t *reqRaw = nullptr;
    ret = ovCheck(ov.compiled_model_create_infer_request(I.compiled.get(), &reqRaw), errMessage);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }
    I.req.reset(reqRaw);

    ret = I.updateCompiledShapes(errMessage);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }

    I.devId = deviceA;
    char *property = nullptr;
    if (ov.core_get_property(I.core.get(), deviceA.c_str(), "FULL_DEVICE_NAME", &property) == OK) {
        I.devName = getAndFreeProperty(property);
    }
    property = nullptr;
    if (ov.compiled_model_get_property(I.compiled.get(), "INFERENCE_PRECISION_HINT", &property) == OK) {
        I.precision = getAndFreeProperty(property);
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYOpenVINO::initFromOpenCLQueue(const tstring &modelPath, void *clQueue, void *clContext,
                                         const int height, const int width, tstring &errMessage) {
    auto ret = ensureOpenVINO(errMessage);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }
    auto &I = *m_impl;
    if (!I.core) {
        errMessage = I.loadErr;
        return RGY_ERR_UNSUPPORTED;
    }
    auto &ov = ovLoader();
    if (!ov.core_create_context || !ov.core_compile_model_with_context || !ov.remote_context_free
        || !ov.remote_context_create_host_tensor || !ov.infer_request_set_output_tensor) {
        errMessage = _T("OpenVINO C runtime does not provide remote context API required for host tensor binding");
        return RGY_ERR_UNSUPPORTED;
    }
    if (clQueue == nullptr && clContext == nullptr) {
        errMessage = _T("OpenCL queue/context is not available");
        return RGY_ERR_INVALID_PARAM;
    }
    I.resetRuntimeObjects();
    const auto modelPathA = tchar_to_string(modelPath, CP_UTF8);

    ov_model_t *modelRaw = nullptr;
    ret = ovCheck(ov.core_read_model(I.core.get(), modelPathA.c_str(), nullptr, &modelRaw), errMessage);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }
    I.model.reset(modelRaw);

    const int64_t ch = modelInputChannels(I.model.get(), errMessage);
    const int64_t dims[4] = { 1, ch, height, width };
    ov_partial_shape_t reshape = {};
    ret = ovCheck(ov.partial_shape_create_static(4, dims, &reshape), errMessage);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }
    ret = ovCheck(ov.model_reshape_single_input(I.model.get(), reshape), errMessage);
    ov.partial_shape_free(&reshape);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }

    ov_remote_context_t *remoteRaw = nullptr;
    tstring queueErr;
    if (clQueue != nullptr) {
        ret = ovCheck(ov.core_create_context(I.core.get(), "GPU", 4, &remoteRaw,
            const_cast<char *>("CONTEXT_TYPE"), const_cast<char *>("OCL"),
            const_cast<char *>("OCL_QUEUE"), clQueue), errMessage);
        if (ret != RGY_ERR_NONE) {
            queueErr = errMessage;
            errMessage.clear();
        }
    }
    if (remoteRaw == nullptr && clQueue == nullptr && clContext != nullptr) {
        ret = ovCheck(ov.core_create_context(I.core.get(), "GPU", 6, &remoteRaw,
            const_cast<char *>("CONTEXT_TYPE"), const_cast<char *>("OCL"),
            const_cast<char *>("OCL_CONTEXT"), clContext,
            const_cast<char *>("OCL_CONTEXT_DEVICE_ID"), const_cast<char *>("0")), errMessage);
    }
    if (remoteRaw == nullptr) {
        if (!queueErr.empty() && !errMessage.empty()) {
            errMessage = queueErr + _T(" / ") + errMessage;
        } else if (!queueErr.empty()) {
            errMessage = queueErr;
        }
        return RGY_ERR_UNSUPPORTED;
    }
    I.remote.reset(remoteRaw);

    ov_compiled_model_t *compiledRaw = nullptr;
    ret = ovCheck(ov.core_compile_model_with_context(I.core.get(), I.model.get(), I.remote.get(), 0, &compiledRaw), errMessage);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }
    I.compiled.reset(compiledRaw);

    ov_infer_request_t *reqRaw = nullptr;
    ret = ovCheck(ov.compiled_model_create_infer_request(I.compiled.get(), &reqRaw), errMessage);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }
    I.req.reset(reqRaw);

    ret = I.updateCompiledShapes(errMessage);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }

    auto inShape = makeShape(I.inShape);
    ov_tensor_t *inHostTensorRaw = nullptr;
    ret = ovCheck(ov.remote_context_create_host_tensor(I.remote.get(), F32, inShape, &inHostTensorRaw), errMessage);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }
    I.remoteInTensor.reset(inHostTensorRaw);

    auto outShape = makeShape(I.outShape);
    ov_tensor_t *outHostTensorRaw = nullptr;
    ret = ovCheck(ov.remote_context_create_host_tensor(I.remote.get(), F32, outShape, &outHostTensorRaw), errMessage);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }
    I.remoteOutTensor.reset(outHostTensorRaw);
    ret = ovCheck(ov.infer_request_set_input_tensor(I.req.get(), I.remoteInTensor.get()), errMessage);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }
    ret = ovCheck(ov.infer_request_set_output_tensor(I.req.get(), I.remoteOutTensor.get()), errMessage);
    if (ret != RGY_ERR_NONE) {
        return ret;
    }

    if (ov.remote_context_get_device_name) {
        char *deviceName = nullptr;
        if (ov.remote_context_get_device_name(I.remote.get(), &deviceName) == OK) {
            I.devId = getAndFreeProperty(deviceName);
        }
    }
    if (I.devId.empty()) {
        I.devId = "GPU";
    }

    char *property = nullptr;
    if (ov.core_get_property(I.core.get(), I.devId.c_str(), "FULL_DEVICE_NAME", &property) == OK) {
        I.devName = getAndFreeProperty(property);
    } else {
        I.devName = char_to_tstring(I.devId);
    }
    property = nullptr;
    if (ov.compiled_model_get_property(I.compiled.get(), "INFERENCE_PRECISION_HINT", &property) == OK) {
        I.precision = getAndFreeProperty(property);
    }
    I.shared = true;
    return RGY_ERR_NONE;
}

tstring RGYOpenVINO::findDeviceByUuidLuid(const void *uuid, const size_t uuidSize,
                                          const void *luid, const size_t luidSize,
                                          tstring &errMessage) {
    auto ret = ensureOpenVINO(errMessage);
    if (ret != RGY_ERR_NONE) {
        return tstring();
    }
    if (!m_impl || !m_impl->core) {
        errMessage = m_impl ? m_impl->loadErr : _T("OpenVINO runtime is not initialized");
        return tstring();
    }
    auto &ov = ovLoader();
    const auto targetUuid = isZeroBytes(uuid, uuidSize) ? std::string() : bytesToHex(uuid, uuidSize);
    const auto targetLuid = isZeroBytes(luid, luidSize) ? std::string() : bytesToHex(luid, luidSize);
    if (targetUuid.empty() && targetLuid.empty()) {
        errMessage = _T("OpenCL device UUID/LUID is not available");
        return tstring();
    }

    std::vector<std::string> candidates;
    char *property = nullptr;
    if (ov.core_get_property(m_impl->core.get(), "GPU", "AVAILABLE_DEVICES", &property) == OK) {
        for (const auto &id : splitDeviceIds(tchar_to_string(getAndFreeProperty(property)))) {
            candidates.push_back(id.rfind("GPU", 0) == 0 ? id : ("GPU." + id));
        }
    }
    if (candidates.empty()) {
        candidates.push_back("GPU.0");
        candidates.push_back("GPU");
    } else {
        candidates.push_back("GPU");
    }

    for (const auto &device : candidates) {
        bool uuidQueried = false;
        bool uuidMatch = false;
        bool luidMatch = false;
        property = nullptr;
        if (!targetUuid.empty() && ov.core_get_property(m_impl->core.get(), device.c_str(), "DEVICE_UUID", &property) == OK) {
            uuidQueried = true;
            uuidMatch = normalizeHexString(tchar_to_string(getAndFreeProperty(property))) == targetUuid;
        }
        property = nullptr;
        if (!targetLuid.empty() && ov.core_get_property(m_impl->core.get(), device.c_str(), "DEVICE_LUID", &property) == OK) {
            luidMatch = normalizeHexString(tchar_to_string(getAndFreeProperty(property))) == targetLuid;
        }
        if (uuidMatch || ((!uuidQueried || targetUuid.empty()) && luidMatch)) {
            return char_to_tstring(device);
        }
    }

    errMessage = _T("OpenVINO GPU matching the selected OpenCL device was not found");
    return tstring();
}

RGY_ERR RGYOpenVINO::infer(const float *in, float *out) {
    tstring errMessage;
    if (ensureOpenVINO(errMessage) != RGY_ERR_NONE || !m_impl || !m_impl->req) {
        return RGY_ERR_UNSUPPORTED;
    }
    auto &ov = ovLoader();
    if (m_impl->remote) {
        if (!m_impl->remoteInTensor || !m_impl->remoteOutTensor) {
            return RGY_ERR_UNSUPPORTED;
        }
        void *inData = nullptr;
        void *outData = nullptr;
        size_t inBytes = 0;
        size_t outBytes = 0;
        if (ov.tensor_data(m_impl->remoteInTensor.get(), &inData) != OK
            || ov.tensor_get_byte_size(m_impl->remoteInTensor.get(), &inBytes) != OK
            || ov.tensor_data(m_impl->remoteOutTensor.get(), &outData) != OK
            || ov.tensor_get_byte_size(m_impl->remoteOutTensor.get(), &outBytes) != OK) {
            return RGY_ERR_UNKNOWN;
        }
        if (inData == nullptr || outData == nullptr) {
            return RGY_ERR_UNKNOWN;
        }
        std::memcpy(inData, in, inBytes);
        if (ov.infer_request_infer(m_impl->req.get()) != OK) {
            return RGY_ERR_UNKNOWN;
        }
        std::memcpy(out, outData, outBytes);
        return RGY_ERR_NONE;
    }
    auto inShape = makeShape(m_impl->inShape);
    ov_tensor_t *inTensorRaw = nullptr;
    if (ov.tensor_create_from_host_ptr(F32, inShape, const_cast<float *>(in), &inTensorRaw) != OK) {
        return RGY_ERR_UNKNOWN;
    }
    ov_unique_ptr<ov_tensor_t, ov_tensor_free_t> inTensor(inTensorRaw, ov.tensor_free);
    if (ov.infer_request_set_input_tensor(m_impl->req.get(), inTensor.get()) != OK) {
        return RGY_ERR_UNKNOWN;
    }
    if (ov.infer_request_infer(m_impl->req.get()) != OK) {
        return RGY_ERR_UNKNOWN;
    }
    ov_tensor_t *outTensorRaw = nullptr;
    if (ov.infer_request_get_output_tensor(m_impl->req.get(), &outTensorRaw) != OK) {
        return RGY_ERR_UNKNOWN;
    }
    ov_unique_ptr<ov_tensor_t, ov_tensor_free_t> outTensor(outTensorRaw, ov.tensor_free);
    void *outData = nullptr;
    size_t outBytes = 0;
    if (ov.tensor_data(outTensor.get(), &outData) != OK || ov.tensor_get_byte_size(outTensor.get(), &outBytes) != OK) {
        return RGY_ERR_UNKNOWN;
    }
    std::memcpy(out, outData, outBytes);
    return RGY_ERR_NONE;
}

RGY_ERR RGYOpenVINO::initShared(const tstring &, void *, const int, const int, tstring &errMessage) {
    errMessage = _T("OpenVINO shared OpenCL path is not available in the C API dynamic loader implementation yet");
    return RGY_ERR_UNSUPPORTED;
}
RGY_ERR RGYOpenVINO::setSharedIO(void *, void *) { return RGY_ERR_UNSUPPORTED; }
RGY_ERR RGYOpenVINO::inferShared() { return RGY_ERR_UNSUPPORTED; }
bool RGYOpenVINO::usingSharedContext() const { return m_impl && m_impl->shared; }

int RGYOpenVINO::inChannels()  const { return m_impl && m_impl->inShape.size()  >= 4 ? (int)m_impl->inShape[1]  : 1; }
int RGYOpenVINO::inHeight()    const { return m_impl && m_impl->inShape.size()  >= 4 ? (int)m_impl->inShape[2]  : 0; }
int RGYOpenVINO::inWidth()     const { return m_impl && m_impl->inShape.size()  >= 4 ? (int)m_impl->inShape[3]  : 0; }
int RGYOpenVINO::outChannels() const { return m_impl && m_impl->outShape.size() >= 4 ? (int)m_impl->outShape[1] : 1; }
int RGYOpenVINO::outHeight()   const { return m_impl && m_impl->outShape.size() >= 4 ? (int)m_impl->outShape[2] : 0; }
int RGYOpenVINO::outWidth()    const { return m_impl && m_impl->outShape.size() >= 4 ? (int)m_impl->outShape[3] : 0; }
size_t RGYOpenVINO::outElemCount() const {
    if (!m_impl || m_impl->outShape.empty()) {
        return 0;
    }
    size_t n = 1;
    for (auto d : m_impl->outShape) n *= d;
    return n;
}
tstring RGYOpenVINO::deviceFullName()     const { return m_impl ? m_impl->devName : tstring(); }
tstring RGYOpenVINO::inferencePrecision() const { return m_impl ? m_impl->precision : tstring(); }
bool RGYOpenVINO::available() { return ovLoader().load(); }
tstring RGYOpenVINO::availabilityStatus() {
    auto &ov = ovLoader();
    if (ov.load()) {
        return tstring();
    }
    return ov.error;
}

#else // !ENABLE_OPENVINO

class RGYOpenVINO::Impl {};
RGYOpenVINO::RGYOpenVINO() : m_impl(nullptr) {}
RGYOpenVINO::~RGYOpenVINO() {}
RGY_ERR RGYOpenVINO::peekChannels(const tstring &, int &inChannels, int &outChannels, tstring &errMessage) {
    inChannels = outChannels = 0;
    errMessage = _T("this build of QSVEnc does not include OpenVINO support");
    return RGY_ERR_UNSUPPORTED;
}
RGY_ERR RGYOpenVINO::init(const tstring &, const tstring &, const int, const int, tstring &errMessage) {
    errMessage = _T("this build of QSVEnc does not include OpenVINO support");
    return RGY_ERR_UNSUPPORTED;
}
RGY_ERR RGYOpenVINO::initFromOpenCLQueue(const tstring &, void *, void *, const int, const int, tstring &errMessage) {
    errMessage = _T("this build of QSVEnc does not include OpenVINO support");
    return RGY_ERR_UNSUPPORTED;
}
tstring RGYOpenVINO::findDeviceByUuidLuid(const void *, const size_t, const void *, const size_t, tstring &errMessage) {
    errMessage = _T("this build of QSVEnc does not include OpenVINO support");
    return tstring();
}
RGY_ERR RGYOpenVINO::infer(const float *, float *) { return RGY_ERR_UNSUPPORTED; }
RGY_ERR RGYOpenVINO::initShared(const tstring &, void *, const int, const int, tstring &errMessage) {
    errMessage = _T("this build of QSVEnc does not include OpenVINO support");
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
tstring RGYOpenVINO::deviceFullName()     const { return tstring(); }
tstring RGYOpenVINO::inferencePrecision() const { return tstring(); }
bool RGYOpenVINO::available() { return false; }
tstring RGYOpenVINO::availabilityStatus() { return _T("this build of QSVEnc does not include OpenVINO support"); }

#endif // ENABLE_OPENVINO
