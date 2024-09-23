// -----------------------------------------------------------------------------------------
// RGY by rigaya
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

#include "rgy_filter_libplacebo.h"

tstring RGYFilterParamLibplaceboResample::print() const {
    return resample.print();
}

#if ENABLE_LIBPLACEBO

#pragma comment(lib, "libplacebo-349.lib")

static const TCHAR *RGY_LIBPLACEBO_DLL_NAME = _T("libplacebo-349.dll");

static const RGYLogType RGY_LOGT_LIBPLACEBO = RGY_LOGT_VPP;

static const auto RGY_LOG_LEVEL_TO_LIBPLACEBO = make_array<std::pair<RGYLogLevel, pl_log_level>>(
    std::make_pair(RGYLogLevel::RGY_LOG_QUIET, PL_LOG_NONE),
    std::make_pair(RGYLogLevel::RGY_LOG_ERROR, PL_LOG_ERR),
    std::make_pair(RGYLogLevel::RGY_LOG_WARN,  PL_LOG_WARN),
    std::make_pair(RGYLogLevel::RGY_LOG_INFO,  PL_LOG_INFO),
    std::make_pair(RGYLogLevel::RGY_LOG_DEBUG, PL_LOG_DEBUG),
    std::make_pair(RGYLogLevel::RGY_LOG_TRACE, PL_LOG_TRACE)
);

MAP_PAIR_0_1(loglevel, rgy, RGYLogLevel, libplacebo, pl_log_level, RGY_LOG_LEVEL_TO_LIBPLACEBO, RGYLogLevel::RGY_LOG_INFO, PL_LOG_INFO);

static const auto RGY_RESIZE_ALGO_TO_LIBPLACEBO = make_array<std::pair<RGY_VPP_RESIZE_ALGO, const char*>>(
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_SPLINE16, "spline16"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_SPLINE36, "spline36"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_SPLINE64, "spline64"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_NEAREST, "nearest"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_BILINEAR, "bilinear"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_GAUSSIAN, "gaussian"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_SINC, "sinc"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_LANCZOS, "lanczos"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_GINSENG, "ginseng"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_JINC, "ewa_jinc"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_LANCZOS, "ewa_lanczos"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_LANCZOSSHARP, "ewa_lanczossharp"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_LANCZOS4SHARPEST, "ewa_lanczos4sharpest"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_GINSENG, "ewa_ginseng"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_HANN, "ewa_hann"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_HANNING, "ewa_hanning"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_BICUBIC, "bicubic"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_TRIANGLE, "triangle"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_HERMITE, "hermite"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_CATMULL_ROM, "catmull_rom"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_MITCHELL, "mitchell"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_MITCHELL_CLAMP, "mitchell_clamp"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_ROBIDOUX, "robidoux"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_ROBIDOUXSHARP, "robidouxsharp"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_ROBIDOUX, "ewa_robidoux"),
    std::make_pair(RGY_VPP_RESIZE_LIBPLACEBO_EWA_ROBIDOUXSHARP, "ewa_robidouxsharp")
);

MAP_PAIR_0_1(resize_algo, rgy, RGY_VPP_RESIZE_ALGO, libplacebo, const char*, RGY_RESIZE_ALGO_TO_LIBPLACEBO, RGY_VPP_RESIZE_UNKNOWN, nullptr);

std::unique_ptr<std::remove_pointer<pl_tex>::type, RGYLibplaceboTexDeleter> rgy_pl_tex_recreate(pl_gpu gpu, const pl_tex_params& tex_params) {
    pl_tex tex_tmp = { 0 };
    if (!pl_tex_recreate(gpu, &tex_tmp, &tex_params)) {
        return std::unique_ptr<std::remove_pointer<pl_tex>::type, RGYLibplaceboTexDeleter>();
    }
    return std::unique_ptr<std::remove_pointer<pl_tex>::type, RGYLibplaceboTexDeleter>(
        tex_tmp, RGYLibplaceboTexDeleter(gpu));
}

static void libplacebo_log_func(void *private_data, pl_log_level level, const char* msg) {
    auto log = static_cast<RGYLog*>(private_data);
    auto log_level = loglevel_libplacebo_to_rgy(level);
    if (log == nullptr || log_level < log->getLogLevel(RGY_LOGT_LIBPLACEBO)) {
        return;
    }
    log->write_log(log_level, RGY_LOGT_LIBPLACEBO, (tstring(_T("libplacebo: ")) + char_to_tstring(msg) + _T("\n")).c_str());
}

RGYFrameD3D11::RGYFrameD3D11() : frame(), clframe() {}

RGYFrameD3D11::~RGYFrameD3D11() { deallocate(); };

RGY_ERR RGYFrameD3D11::allocate(ID3D11Device *device, const int width, const int height, const RGY_CSP csp, const int bitdepth) {
    if (!device) {
        return RGY_ERR_NULL_PTR;
    }
    if (frame.ptr[0]) {
        deallocate();
    }
    const auto dxgi_format = (RGY_CSP_DATA_TYPE[csp] != RGY_DATA_TYPE_U8) ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R8_UNORM;
    frame = RGYFrameInfo(width, height, csp, bitdepth);
    for (int iplane = 0; iplane < RGY_CSP_PLANES[csp]; iplane++) {
        auto plane = getPlane(&frame, (RGY_PLANE)iplane);
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = plane.width;
        desc.Height = plane.height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = dxgi_format;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_RENDER_TARGET;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

        ID3D11Texture2D *pTexture = nullptr;
        HRESULT hr = device->CreateTexture2D(&desc, nullptr, &pTexture);
        if (FAILED(hr)) {
            return RGY_ERR_MEMORY_ALLOC;
        }
        frame.ptr[iplane] = (uint8_t*)pTexture;
    }
    return RGY_ERR_NONE;
}

void RGYFrameD3D11::deallocate() {
    if (clframe) {
        clframe.reset();
    }
    for (int iplane = 0; iplane < RGY_CSP_PLANES[frame.csp]; iplane++) {
        if (frame.ptr[iplane]) {
            auto pTexture = (ID3D11Texture2D*)frame.ptr[iplane];
            pTexture->Release();
            frame.ptr[iplane] = nullptr;
        }
    }
}

RGYCLFrameInterop *RGYFrameD3D11::getCLFrame(RGYOpenCLContext *clctx, RGYOpenCLQueue& queue) {
    if (!clframe) {
        clframe = clctx->createFrameFromD3D11SurfacePlanar(frame, queue, CL_MEM_READ_WRITE);
    }
    return clframe.get();
}

RGYFilterLibplacebo::RGYFilterLibplacebo(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_textCspIn(RGY_CSP_NA),
    m_textCspOut(RGY_CSP_NA),
    m_dxgiformatIn(DXGI_FORMAT_UNKNOWN),
    m_dxgiformatOut(DXGI_FORMAT_UNKNOWN),
    m_log(),
    m_d3d11(),
    m_dispatch(),
    m_renderer(),
    m_dither_state(),
    m_textIn(),
    m_textOut() {
    m_name = _T("libplacebo");
}
RGYFilterLibplacebo::~RGYFilterLibplacebo() {
    close();
}

RGY_ERR RGYFilterLibplacebo::initLibplacebo(const RGYFilterParam *param) {
    auto prm = dynamic_cast<const RGYFilterParamLibplacebo*>(param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (!m_cl->platform()->d3d11dev()) {
        AddMessage(RGY_LOG_ERROR, _T("DX11 device not set\n"));
        return RGY_ERR_NULL_PTR;
    }
    if (auto hModule = RGY_LOAD_LIBRARY(RGY_LIBPLACEBO_DLL_NAME); hModule == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("%s is required but not found.\n"), RGY_LIBPLACEBO_DLL_NAME);
        return RGY_ERR_UNKNOWN;
    } else {
        RGY_FREE_LIBRARY(hModule);
    }
    const pl_log_params log_params = {libplacebo_log_func, m_pLog.get(), loglevel_rgy_to_libplacebo(m_pLog->getLogLevel(RGY_LOGT_LIBPLACEBO))};
    m_log = std::unique_ptr<std::remove_pointer<pl_log>::type, RGYLibplaceboDeleter<pl_log>>(pl_log_create(0, &log_params), RGYLibplaceboDeleter<pl_log>(pl_log_destroy));
    if (!m_log) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to create libplacebo log.\n"));
        return RGY_ERR_UNKNOWN;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Created libplacebo log.\n"));

    pl_d3d11_params gpu_params = { 0 };
    gpu_params.device = (ID3D11Device*)m_cl->platform()->d3d11dev();

    m_d3d11 = std::unique_ptr<std::remove_pointer<pl_d3d11>::type, RGYLibplaceboDeleter<pl_d3d11>>(
        pl_d3d11_create(m_log.get(), &gpu_params), RGYLibplaceboDeleter<pl_d3d11>(pl_d3d11_destroy));
    if (!m_d3d11) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to create libplacebo D3D11 device.\n"));
        return RGY_ERR_UNKNOWN;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Created libplacebo D3D11 device.\n"));

    m_dispatch = std::unique_ptr<std::remove_pointer<pl_dispatch>::type, RGYLibplaceboDeleter<pl_dispatch>>(
        pl_dispatch_create(m_log.get(), m_d3d11->gpu), RGYLibplaceboDeleter<pl_dispatch>(pl_dispatch_destroy));
    if (!m_dispatch) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to create libplacebo dispatch.\n"));
        return RGY_ERR_UNKNOWN;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Created libplacebo dispatch.\n"));

    m_renderer = std::unique_ptr<std::remove_pointer<pl_renderer>::type, RGYLibplaceboDeleter<pl_renderer>>(
        pl_renderer_create(m_log.get(), m_d3d11->gpu), RGYLibplaceboDeleter<pl_renderer>(pl_renderer_destroy));
    if (!m_renderer) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to create libplacebo renderer.\n"));
        return RGY_ERR_UNKNOWN;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Created libplacebo renderer.\n"));
    return RGY_ERR_NONE;
}

RGY_CSP RGYFilterLibplacebo::getTextureCsp(const RGY_CSP csp) {
    const auto inChromaFmt = RGY_CSP_CHROMA_FORMAT[csp];
    if (inChromaFmt == RGY_CHROMAFMT_RGB) {
        return (RGY_CSP_DATA_TYPE[csp] != RGY_DATA_TYPE_U8) ? RGY_CSP_RGB_16 : RGY_CSP_RGB;
    } else if (inChromaFmt == RGY_CHROMAFMT_YUV420) {
        return (RGY_CSP_DATA_TYPE[csp] != RGY_DATA_TYPE_U8) ? RGY_CSP_YV12_16 : RGY_CSP_YV12;
    } else if (inChromaFmt == RGY_CHROMAFMT_YUV444) {
        return (RGY_CSP_DATA_TYPE[csp] != RGY_DATA_TYPE_U8) ? RGY_CSP_YUV444_16 : RGY_CSP_YUV444;
    }
    return RGY_CSP_NA;
}

DXGI_FORMAT RGYFilterLibplacebo::getTextureDXGIFormat(const RGY_CSP csp) {
    return (RGY_CSP_DATA_TYPE[csp] != RGY_DATA_TYPE_U8) ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R8_UNORM;
}

RGY_ERR RGYFilterLibplacebo::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;

    RGY_ERR sts = RGY_ERR_NONE;
    sts = checkParam(pParam.get());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    if (rgy_csp_has_alpha(pParam->frameIn.csp)) {
        AddMessage(RGY_LOG_ERROR, _T("nfx filters does not support alpha channel.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    sts = initLibplacebo(pParam.get());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    sts = setLibplaceboParam(pParam.get());
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    m_textCspIn = getTextureCsp(pParam->frameIn.csp);
    m_textCspOut = getTextureCsp(pParam->frameOut.csp);
    m_dxgiformatIn = getTextureDXGIFormat(pParam->frameIn.csp);
    m_dxgiformatOut = getTextureDXGIFormat(pParam->frameOut.csp);

    sts = initCommon(pParam);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return sts;
}

RGY_ERR RGYFilterLibplacebo::initCommon(shared_ptr<RGYFilterParam> pParam) {
    RGY_ERR sts = RGY_ERR_NONE;
    auto prm = dynamic_cast<RGYFilterParamLibplacebo*>(pParam.get());
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const auto inChromaFmt = RGY_CSP_CHROMA_FORMAT[pParam->frameIn.csp];
    VideoVUIInfo vui = prm->vui;
    if (inChromaFmt == RGY_CHROMAFMT_RGB || inChromaFmt == RGY_CHROMAFMT_RGB_PACKED) {
        vui.setIfUnset(VideoVUIInfo().to(RGY_MATRIX_RGB).to(RGY_PRIM_BT709).to(RGY_TRANSFER_IEC61966_2_1));
    } else {
        vui.setIfUnset(VideoVUIInfo().to((CspMatrix)COLOR_VALUE_AUTO_RESOLUTION).to((CspColorprim)COLOR_VALUE_AUTO_RESOLUTION).to((CspTransfer)COLOR_VALUE_AUTO_RESOLUTION));
    }
    vui.apply_auto(VideoVUIInfo(), pParam->frameIn.height);

    if (!m_srcCrop
        || m_srcCrop->GetFilterParam()->frameIn.width != pParam->frameIn.width
        || m_srcCrop->GetFilterParam()->frameIn.height != pParam->frameIn.height) {
        AddMessage(RGY_LOG_DEBUG, _T("Create input csp conversion filter.\n"));
        unique_ptr<RGYFilterCspCrop> filter(new RGYFilterCspCrop(m_cl));
        shared_ptr<RGYFilterParamCrop> paramCrop(new RGYFilterParamCrop());
        paramCrop->frameIn = pParam->frameIn;
        paramCrop->frameOut = paramCrop->frameIn;
        paramCrop->frameOut.csp = m_textCspIn;
        paramCrop->baseFps = pParam->baseFps;
        paramCrop->frameIn.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->bOutOverwrite = false;
        sts = filter->init(paramCrop, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_srcCrop = std::move(filter);
        AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_srcCrop->GetInputMessage().c_str());
    }
    const int numPlanes = RGY_CSP_PLANES[pParam->frameIn.csp];
    if (numPlanes != RGY_CSP_PLANES[pParam->frameOut.csp]) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp, int out plane count does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (!m_textIn
    || m_textIn->width() != pParam->frameIn.width
    || m_textIn->height() != pParam->frameIn.height
    || m_textIn->csp() != pParam->frameIn.csp) {
        m_textIn = std::make_unique<RGYFrameD3D11>();
        sts = m_textIn->allocate((ID3D11Device*)m_cl->platform()->d3d11dev(), pParam->frameIn.width, pParam->frameIn.height, m_textCspIn, RGY_CSP_BIT_DEPTH[m_textCspIn]);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_DEBUG, _T("failed to create input texture: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }
    if (!m_textOut
    || m_textOut->width() != pParam->frameIn.width
    || m_textOut->height() != pParam->frameIn.height
    || m_textOut->csp() != pParam->frameIn.csp) {
        m_textOut = std::make_unique<RGYFrameD3D11>();
        sts = m_textOut->allocate((ID3D11Device*)m_cl->platform()->d3d11dev(), pParam->frameOut.width, pParam->frameOut.height, m_textCspOut, RGY_CSP_BIT_DEPTH[m_textCspOut]);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_DEBUG, _T("failed to create output texture: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    if (!m_dstCrop
        || m_dstCrop->GetFilterParam()->frameOut.width != pParam->frameOut.width
        || m_dstCrop->GetFilterParam()->frameOut.height != pParam->frameOut.height) {
        AddMessage(RGY_LOG_DEBUG, _T("Create output csp conversion filter.\n"));
        unique_ptr<RGYFilterCspCrop> filter(new RGYFilterCspCrop(m_cl));
        shared_ptr<RGYFilterParamCrop> paramCrop(new RGYFilterParamCrop());
        paramCrop->frameIn = pParam->frameOut;
        paramCrop->frameIn.csp = m_textCspOut;
        paramCrop->frameOut = pParam->frameOut;
        paramCrop->baseFps = pParam->baseFps;
        paramCrop->frameIn.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->bOutOverwrite = false;
        sts = filter->init(paramCrop, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_dstCrop = std::move(filter);
        AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_dstCrop->GetInputMessage().c_str());

        m_textFrameBufOut = m_cl->createFrameBuffer(m_dstCrop->GetFilterParam()->frameIn.width, m_dstCrop->GetFilterParam()->frameIn.height, m_textCspOut, RGY_CSP_BIT_DEPTH[m_textCspOut]);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_DEBUG, _T("failed to allocate memory for libplacebo output: %s.\n"), get_err_mes(sts));
            return sts;
        }
    }

    if (m_frameBuf.size() == 0
        || !cmpFrameInfoCspResolution(&m_frameBuf[0]->frame, &pParam->frameOut)) {
        sts = AllocFrameBuf(pParam->frameOut, 2);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    for (int i = 0; i < RGY_CSP_PLANES[pParam->frameOut.csp]; i++) {
        pParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    const tstring nameBlank(m_name.length() + _tcslen(_T(": ")), _T(' '));
    tstring info = m_name + _T(": ");
    tstring INFO_INDENT = _T("    ");
    if (m_srcCrop) {
        info += tstring(INFO_INDENT) + nameBlank + m_srcCrop->GetInputMessage() + _T("\n");
    }
    info += tstring(INFO_INDENT) + nameBlank + pParam->print() + _T("\n");
    if (m_dstCrop) {
        info += tstring(INFO_INDENT) + nameBlank + m_dstCrop->GetInputMessage() + _T("\n");
    }
    setFilterInfo(info);
    m_param = pParam;
    return sts;
}

RGY_ERR RGYFilterLibplacebo::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }
    auto prm = dynamic_cast<RGYFilterParamLibplacebo*>(m_param.get());
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    //const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    //if (memcpyKind != cudaMemcpyDeviceToDevice) {
    //    AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
    //    return RGY_ERR_INVALID_PARAM;
    //}
    // pInputFrame -> txtFrameBufIn
    const RGYFrameInfo *txtFrameBufIn = nullptr;
    if (!m_srcCrop) {
        AddMessage(RGY_LOG_ERROR, _T("srcCrop is not set.\n"));
        return RGY_ERR_NULL_PTR;
    }
#define COPY_DEBUG 0
    {
        auto textInCL = m_textIn->getCLFrame(m_cl.get(), queue);
        auto err = textInCL->acquire(queue);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to acquire CL frame: %s.\n"), get_err_mes(err));
            return err;
        }
        auto textInCLInfo = textInCL->frameInfo();
        int filterOutputNum = 0;
        RGYFrameInfo *outInfo[1] = { &textInCLInfo };
        RGYFrameInfo cropInput = *pInputFrame;
        auto sts_filter = m_srcCrop->filter(&cropInput, (RGYFrameInfo **)&outInfo, &filterOutputNum, queue, wait_events, event);
        txtFrameBufIn = outInfo[0];
        if (txtFrameBufIn == nullptr || filterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_srcCrop->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || filterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_srcCrop->name().c_str());
            return sts_filter;
        }
        copyFramePropWithoutRes(outInfo[0], pInputFrame);
#if COPY_DEBUG
        auto textOutCL = m_textOut->getCLFrame(m_cl.get(), queue);
        err = textOutCL->acquire(queue);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to acquire CL frame: %s.\n"), get_err_mes(err));
            return err;
        }
        auto textOutCLInfo = textOutCL->frameInfo();
        for (int iplane = 0; iplane < RGY_CSP_PLANES[m_textIn->csp()]; iplane++) {
            auto planeIn = getPlane(&textInCLInfo, (RGY_PLANE)iplane);
            auto planeOut = getPlane(&textOutCLInfo, (RGY_PLANE)iplane);
            size_t origin[3] = { 0, 0, 0 };
            size_t region[3] = { std::min<size_t>(planeIn.width, planeOut.width), std::min<size_t>(planeIn.height, planeOut.height), 1};
            err = err_cl_to_rgy(clEnqueueCopyImage(queue.get(), (cl_mem)planeIn.ptr[0], (cl_mem)planeOut.ptr[0],
                origin, origin, region, 0, nullptr, nullptr));
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to copy CL iamge: %s.\n"), get_err_mes(err));
                return err;
            }
        }
        textInCL->release();
    }
#else
        textInCL->release();
        queue.finish();
    }

    // フィルタを適用
    for (int iplane = 0; iplane < RGY_CSP_PLANES[m_textIn->csp()]; iplane++) {
        auto textInFrameInfo = m_textIn->frameInfo();
        auto planeIn = getPlane(&textInFrameInfo, (RGY_PLANE)iplane);
        pl_d3d11_wrap_params d3d11_wrap_in = { 0 };
        d3d11_wrap_in.tex = (ID3D11Texture2D*)planeIn.ptr[0];
        d3d11_wrap_in.array_slice = 0;
        d3d11_wrap_in.fmt = m_dxgiformatIn;
        d3d11_wrap_in.w = planeIn.width;
        d3d11_wrap_in.h = planeIn.height;
        auto pl_tex_in = std::unique_ptr<std::remove_pointer<pl_tex>::type, RGYLibplaceboTexDeleter>(
            pl_d3d11_wrap(m_d3d11->gpu, &d3d11_wrap_in), RGYLibplaceboTexDeleter(m_d3d11->gpu));
        if (!pl_tex_in) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to wrap input d3d11 plane(%d) to pl_tex.\n"), iplane);
            return RGY_ERR_NULL_PTR;
        }

        auto textOutFrameInfo = m_textOut->frameInfo();
        auto planeOut = getPlane(&textOutFrameInfo, (RGY_PLANE)iplane);
        pl_d3d11_wrap_params d3d11_wrap_out = { 0 };
        d3d11_wrap_out.tex = (ID3D11Texture2D*)planeOut.ptr[0];
        d3d11_wrap_out.array_slice = 0;
        d3d11_wrap_out.fmt = m_dxgiformatOut;
        d3d11_wrap_out.w = planeOut.width;
        d3d11_wrap_out.h = planeOut.height;
        auto pl_tex_out = std::unique_ptr<std::remove_pointer<pl_tex>::type, RGYLibplaceboTexDeleter>(
            pl_d3d11_wrap(m_d3d11->gpu, &d3d11_wrap_out), RGYLibplaceboTexDeleter(m_d3d11->gpu));
        if (!pl_tex_out) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to wrap output d3d11 plane(%d) to pl_tex.\n"), iplane);
            return RGY_ERR_NULL_PTR;
        }

        sts = procPlane(pl_tex_out.get(), &planeOut, pl_tex_in.get(), &planeIn);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to process plane(%d): %s.\n"), iplane, get_err_mes(sts));
            return sts;
        }
    }
    // CL_CONTEXT_INTEROP_USER_SYNC=trueの場合、ここでlibplaceboの処理の終了を待つ必要がある
    pl_gpu_flush(m_d3d11->gpu);
#endif
    if (!ppOutputFrames[0]) {
        ppOutputFrames[0] = &m_frameBuf[0]->frame;
        *pOutputFrameNum = 1;
    }
    // m_ngxFrameBufOut -> ppOutputFrames
    auto textOutCL = m_textOut->getCLFrame(m_cl.get(), queue);
#if !COPY_DEBUG
    auto err = textOutCL->acquire(queue);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to acquire CL frame: %s.\n"), get_err_mes(err));
        return err;
    }
#endif
    auto textOutCLInfo = textOutCL->frameInfo();
    if (m_dstCrop) {
        auto sts_filter = m_dstCrop->filter(&textOutCLInfo, ppOutputFrames, pOutputFrameNum, queue);
        if (ppOutputFrames[0] == nullptr || *pOutputFrameNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_dstCrop->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || *pOutputFrameNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_dstCrop->name().c_str());
            return sts_filter;
        }
        copyFramePropWithoutRes(ppOutputFrames[0], pInputFrame);
        textOutCL->release();
    }
    return RGY_ERR_NONE;
}

int RGYFilterLibplacebo::getTextureBytePerPix(const DXGI_FORMAT format) const {
    switch (format) {
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_UNORM:
        return 1;
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16_UNORM:
        return 2;
    default:
        return 0;
    }
}

void RGYFilterLibplacebo::close() {
    m_textIn.reset();
    m_textOut.reset();
    m_textFrameBufOut.reset();
    m_srcCrop.reset();
    m_dstCrop.reset();
    
    m_renderer.reset();
    m_dispatch.reset();
    m_d3d11.reset();
    m_log.reset();

    m_frameBuf.clear();
}

RGYFilterLibplaceboResample::RGYFilterLibplaceboResample(shared_ptr<RGYOpenCLContext> context) :
    RGYFilterLibplacebo(context),
    m_filter_params() {
    m_name = _T("libplacebo-resample");
}

RGYFilterLibplaceboResample::~RGYFilterLibplaceboResample() {
}

RGY_ERR RGYFilterLibplaceboResample::checkParam(const RGYFilterParam *param) {
    auto prm = dynamic_cast<const RGYFilterParamLibplaceboResample*>(param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    // prm->resampleの各値の範囲をチェック
    if (prm->resample.radius > 16.0f) {
        AddMessage(RGY_LOG_ERROR, _T("radius must be between 0.0f and 16.0f.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->resample.blur < 0.0f || prm->resample.blur > 100.0f) {
        AddMessage(RGY_LOG_ERROR, _T("blur must be between 0.0f and 100.0f.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->resample.taper < 0.0f || prm->resample.taper > 1.0f) {
        AddMessage(RGY_LOG_ERROR, _T("taper must be between 0.0f and 1.0f.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->resample.clamp_ < 0.0f || prm->resample.clamp_ > 1.0f) {
        AddMessage(RGY_LOG_ERROR, _T("clamp must be between 0.0f and 1.0f.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->resample.antiring < 0.0f || prm->resample.antiring > 1.0f) {
        AddMessage(RGY_LOG_ERROR, _T("antiring must be between 0.0f and 1.0f.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->resample.cplace < 0 || prm->resample.cplace > 2) {
        AddMessage(RGY_LOG_ERROR, _T("cplace must be between 0 and 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterLibplaceboResample::setLibplaceboParam(const RGYFilterParam *param) {
    auto prm = dynamic_cast<const RGYFilterParamLibplaceboResample*>(param);

    m_filter_params = std::make_unique<pl_sample_filter_params>();
    m_filter_params->no_widening = false;
    m_filter_params->no_compute = false;
    m_filter_params->antiring = prm->resample.antiring;

    auto resample_filter_name = resize_algo_rgy_to_libplacebo(prm->resize_algo);
    if (resample_filter_name == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported resize algorithm.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    auto filter_config = pl_find_filter_config(resample_filter_name, PL_FILTER_UPSCALING);
    if (!filter_config) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported filter type.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    m_filter_params->filter = *filter_config;
    m_filter_params->filter.clamp = prm->resample.clamp_;
    m_filter_params->filter.blur = prm->resample.blur;
    m_filter_params->filter.taper = prm->resample.taper;
    if (prm->resample.radius >= 0.0) {
        if (!m_filter_params->filter.kernel->resizable) {
            AddMessage(RGY_LOG_WARN, _T("radius %.1f ignored for non-resizable filter: %s.\n"), char_to_tstring(resample_filter_name).c_str());
        } else {
            m_filter_params->filter.radius = prm->resample.radius;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterLibplaceboResample::procPlane(pl_tex texOut, const RGYFrameInfo *pDstPlane, pl_tex texIn, const RGYFrameInfo *pSrcPlane) {
    auto prm = dynamic_cast<RGYFilterParamLibplaceboResample*>(m_param.get());
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    pl_shader_obj lut = { 0 };
    auto filter_params = m_filter_params.get();
    filter_params->lut = &lut;

    std::unique_ptr<std::remove_pointer<pl_tex>::type, RGYLibplaceboTexDeleter> tex_tmp1;

    pl_sample_src src = { 0 };
    src.tex = texIn;
    {
        pl_shader shader1 = pl_dispatch_begin(m_dispatch.get());

        pl_tex_params tex_params = { 0 };
        tex_params.w = src.tex->params.w;
        tex_params.h = src.tex->params.h;
        tex_params.renderable = true;
        tex_params.sampleable = true;
        tex_params.format = src.tex->params.format;

        tex_tmp1 = rgy_pl_tex_recreate(m_d3d11->gpu, tex_params);
        if (!tex_tmp1) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to recreate texture.\n"));
            return RGY_ERR_UNKNOWN;
        }

        pl_shader_sample_direct(shader1, &src);

        //if (d->linear) {
        //    pl_color_space colorspace;
        //    colorspace.transfer = d->trc;
        //    pl_shader_linearize(shader1, &colorspace);
        //}
//
        //if (d->sigmoid_params.get()) {
        //    pl_shader_sigmoidize(shader1, d->sigmoid_params.get());
        //}

        pl_dispatch_params dispatch_params = { 0 };
        dispatch_params.target = tex_tmp1.get();
        dispatch_params.shader = &shader1;

        if (!pl_dispatch_finish(m_dispatch.get(), &dispatch_params)) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to dispatch (1).\n"));
            return RGY_ERR_UNKNOWN;
        }
    }

    src.tex = tex_tmp1.get();
    src.rect = pl_rect2df{ 0.0f, 0.0f, (float)pSrcPlane->width, (float)pSrcPlane->height };
    src.new_h = pDstPlane->height;
    src.new_w = pDstPlane->width;

    pl_shader shader2 = pl_dispatch_begin(m_dispatch.get());
    std::unique_ptr<std::remove_pointer<pl_tex>::type, RGYLibplaceboTexDeleter> tex_tmp2;
    if (filter_params->filter.polar) {
        if (!pl_shader_sample_polar(shader2, &src, filter_params)) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to sample polar.\n"));
            return RGY_ERR_UNKNOWN;
        }
    } else {
        pl_sample_src src1 = src;
        src.new_w = src.tex->params.w;
        src.rect.x0 = 0.0f;
        src.rect.x1 = (float)src.new_w;
        src1.rect.y0 = 0.0f;
        src1.rect.y1 = (float)src.new_h;
        {
            pl_shader shader3 = pl_dispatch_begin(m_dispatch.get());
            if (!pl_shader_sample_ortho2(shader3, &src, filter_params)) {
                pl_dispatch_abort(m_dispatch.get(), &shader3);
                AddMessage(RGY_LOG_ERROR, _T("Failed to sample ortho2(1).\n"));
                return RGY_ERR_UNKNOWN;
            }

            pl_tex_params tex_params = { 0 };
            tex_params.w = src.new_w;
            tex_params.h = src.new_h;
            tex_params.renderable = true;
            tex_params.sampleable = true;
            tex_params.format = src.tex->params.format;
            tex_tmp2 = rgy_pl_tex_recreate(m_d3d11->gpu, tex_params);
            if (!tex_tmp2) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to recreate temp texture.\n"));
                return RGY_ERR_UNKNOWN;
            }

            pl_dispatch_params dispatch_params = { 0 };
            dispatch_params.target = tex_tmp2.get();
            dispatch_params.shader = &shader3;

            if (!pl_dispatch_finish(m_dispatch.get(), &dispatch_params)) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to sample polar.\n"));
                return RGY_ERR_UNKNOWN;
            }
        }

        src1.tex = tex_tmp2.get();
        src1.scale = 1.0f;

        if (!pl_shader_sample_ortho2(shader2, &src1, filter_params)) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to sample ortho2(2).\n"));
            return RGY_ERR_UNKNOWN;
        }
    }

    //if (d->sigmoid_params.get()) {
    //    pl_shader_unsigmoidize(shader2, d->sigmoid_params.get());
    //}

    //if (d->linear) {
    //    pl_color_space colorspace;
    //    colorspace.transfer = d->trc;
    //    pl_shader_delinearize(shader2, &colorspace);
    //}

    pl_dispatch_params dispatch_params = { 0 };
    dispatch_params.target = texOut;
    dispatch_params.shader = &shader2;

    if (!pl_dispatch_finish(m_dispatch.get(), &dispatch_params)) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to dispatch (2).\n"));
        return RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

#else

RGYFilterLibplaceboResample::RGYFilterLibplaceboResample(shared_ptr<RGYOpenCLContext> context) : RGYFilterDisabled(context) { m_name = _T("libplacebo-resample"); }
RGYFilterLibplaceboResample::~RGYFilterLibplaceboResample() {};

#endif // ENABLE_LIBPLACEBO
