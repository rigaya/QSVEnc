// -----------------------------------------------------------------------------------------
// QSVEnc/VCEEnc/rkmppenc by rigaya
// -----------------------------------------------------------------------------------------
//
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

#include <map>
#include <array>
#include <cstdint>
#include "rgy_filter_cl.h"

RGY_ERR RGYFilterCspCrop::convertCspFromNV12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto pCropParam = std::dynamic_pointer_cast<RGYFilterParamCrop>(m_param);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    static const auto supportedCspNV12 = make_array<RGY_CSP>(RGY_CSP_NV12, RGY_CSP_P010);
    if (pOutputFrame->csp == pInputFrame->csp
        && std::find(supportedCspNV12.begin(), supportedCspNV12.end(), pCropParam->frameOut.csp) != supportedCspNV12.end()) {
        auto err = m_cl->copyFrame(pOutputFrame, pInputFrame, &pCropParam->crop, queue, wait_events, event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at copyFrame (convertCspFromNV12(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
    }

    auto copyProgram = m_cl->getCspCopyProgram(*pOutputFrame, *pInputFrame);
    if (!copyProgram) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_CL\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    if (RGY_CSP_CHROMA_FORMAT[pOutputFrame->csp] == RGY_CHROMAFMT_RGB) {
        auto planeDstR = getPlane(pOutputFrame, RGY_PLANE_R);
        auto planeDstG = getPlane(pOutputFrame, RGY_PLANE_G);
        auto planeDstB = getPlane(pOutputFrame, RGY_PLANE_B);
        auto planeSrcY = getPlane(pInputFrame, RGY_PLANE_Y);
        auto planeSrcC = getPlane(pInputFrame, RGY_PLANE_C);
        RGYWorkSize local(64, 4);
        RGYWorkSize global(pOutputFrame->width >> 1, pOutputFrame->height >> 1);
        auto err = copyProgram->kernel("kernel_crop_nv12_rgb").config(queue, local, global, wait_events, event).launch(
            (cl_mem)planeDstR.ptr[0], (cl_mem)planeDstG.ptr[0], (cl_mem)planeDstB.ptr[0],
            planeDstR.pitch[0], planeDstR.width, planeDstR.height,
            (cl_mem)planeSrcY.ptr[0], (cl_mem)planeSrcC.ptr[0],
            planeSrcY.pitch[0], planeSrcC.pitch[0], pInputFrame->width, pInputFrame->height, pCropParam->crop.e.left, pCropParam->crop.e.up,
            pCropParam->matrix);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_crop_nv12_rgb (convertCspFromNV12(RGB)(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
        return RGY_ERR_NONE;
    }
    //Y
    if (pOutputFrame->csp == pInputFrame->csp) {
        auto planeDst = getPlane(pOutputFrame, RGY_PLANE_Y);
        auto planeSrc = getPlane(pInputFrame, RGY_PLANE_Y);
        auto err = m_cl->copyPlane(&planeDst, &planeSrc, &pCropParam->crop, queue, wait_events);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at copyPlane(Y) (convertCspFromNV12(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
    } else {
        auto planeDst = getPlane(pOutputFrame, RGY_PLANE_Y);
        auto planeSrc = getPlane(pInputFrame,  RGY_PLANE_Y);
        RGYWorkSize local(32, 8);
        RGYWorkSize global(planeDst.width, planeDst.height);
        auto err = copyProgram->kernel("kernel_copy_plane").config(queue, local, global, wait_events, event).launch(
            (cl_mem)planeDst.ptr[0], planeDst.pitch[0], 0, 0,
            (cl_mem)planeSrc.ptr[0], planeSrc.pitch[0], pCropParam->crop.e.left, pCropParam->crop.e.up,
            planeSrc.width, planeSrc.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_copy_plane (convertCspFromNV12(Y)(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
    }

    static const auto supportedCspYV12 = make_array<RGY_CSP>(RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
    static const auto supportedCspYUV444 = make_array<RGY_CSP>(RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);
    if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), pCropParam->frameOut.csp) != supportedCspYV12.end()) {
        //UV: nv12 -> yv12
        auto planeDstU = getPlane(pOutputFrame, RGY_PLANE_U);
        auto planeDstV = getPlane(pOutputFrame, RGY_PLANE_V);
        auto planeSrc = getPlane(pInputFrame, RGY_PLANE_C);
        //cl_image_format val;
        //clGetImageInfo((cl_mem)planeSrc.ptr[0], CL_IMAGE_FORMAT, sizeof(val), &val, nullptr);
        RGYWorkSize local(32, 8);
        RGYWorkSize global(planeDstU.width, planeDstU.height);
        auto err = copyProgram->kernel("kernel_crop_nv12_yv12").config(queue, local, global, event).launch(
            (cl_mem)planeDstU.ptr[0], (cl_mem)planeDstV.ptr[0], planeDstU.pitch[0], (cl_mem)planeSrc.ptr[0], planeSrc.pitch[0], planeDstU.width, planeDstU.height,
            pCropParam->crop.e.left, pCropParam->crop.e.up);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_crop_nv12_yv12 (convertCspFromNV12(UV)(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
    } else if (std::find(supportedCspNV12.begin(), supportedCspNV12.end(), pCropParam->frameOut.csp) != supportedCspNV12.end()) {
        auto planeDst = getPlane(pOutputFrame, RGY_PLANE_C);
        auto planeSrc = getPlane(pInputFrame, RGY_PLANE_C);
        RGYWorkSize local(32, 8);
        RGYWorkSize global(planeDst.width >> 1, planeDst.height);
        auto err = copyProgram->kernel("kernel_copy_plane_nv12").config(queue, local, global, wait_events, event).launch(
            (cl_mem)planeDst.ptr[0], planeDst.pitch[0], (cl_mem)planeSrc.ptr[0], planeSrc.pitch[0], planeSrc.width >> 1, planeSrc.height,
            pCropParam->crop.e.left, pCropParam->crop.e.up);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_copy_plane (convertCspFromNV12(C)(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
    } else if (std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), pCropParam->frameOut.csp) != supportedCspYUV444.end()) {
        auto planeDstU = getPlane(pOutputFrame, RGY_PLANE_U);
        auto planeDstV = getPlane(pOutputFrame, RGY_PLANE_V);
        auto planeSrc = getPlane(pInputFrame, RGY_PLANE_C);
        RGYWorkSize local(32, 8);
        RGYWorkSize global(planeSrc.width >> 1, planeSrc.height);
        auto err = copyProgram->kernel("kernel_crop_nv12_yuv444").config(queue, local, global, wait_events, event).launch(
            (cl_mem)planeDstU.ptr[0], (cl_mem)planeDstV.ptr[0], planeDstU.pitch[0], planeDstU.width, planeDstU.height,
            (cl_mem)planeSrc.ptr[0], planeSrc.pitch[0], planeSrc.width >> 1, planeSrc.height,
            pCropParam->crop.e.left, pCropParam->crop.e.up);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_copy_plane (convertCspFromNV12(U)(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
    } else {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterCspCrop::convertCspFromYV12(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto pCropParam = std::dynamic_pointer_cast<RGYFilterParamCrop>(m_param);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    static const auto supportedCspYV12 = make_array<RGY_CSP>(RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
    if (pOutputFrame->csp == pInputFrame->csp
        && std::find(supportedCspYV12.begin(), supportedCspYV12.end(), pCropParam->frameOut.csp) != supportedCspYV12.end()) {
        auto err = m_cl->copyFrame(pOutputFrame, pInputFrame, &pCropParam->crop, queue, wait_events, event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at copyFrame (convertCspFromYV12(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
    }

    auto copyProgram = m_cl->getCspCopyProgram(*pOutputFrame, *pInputFrame);
    if (!copyProgram) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_CL\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    if (RGY_CSP_CHROMA_FORMAT[pOutputFrame->csp] == RGY_CHROMAFMT_RGB) {
        auto planeDstR = getPlane(pOutputFrame, RGY_PLANE_R);
        auto planeDstG = getPlane(pOutputFrame, RGY_PLANE_G);
        auto planeDstB = getPlane(pOutputFrame, RGY_PLANE_B);
        auto planeSrcY = getPlane(pInputFrame, RGY_PLANE_Y);
        auto planeSrcU = getPlane(pInputFrame, RGY_PLANE_U);
        auto planeSrcV = getPlane(pInputFrame, RGY_PLANE_V);
        RGYWorkSize local(64, 4);
        RGYWorkSize global(pOutputFrame->width >> 1, pOutputFrame->height >> 1);
        auto err = copyProgram->kernel("kernel_crop_yv12_rgb").config(queue, local, global, wait_events, event).launch(
            (cl_mem)planeDstR.ptr[0], (cl_mem)planeDstG.ptr[0], (cl_mem)planeDstB.ptr[0],
            planeDstR.pitch[0], planeDstR.width, planeDstR.height,
            (cl_mem)planeSrcY.ptr[0], (cl_mem)planeSrcU.ptr[0], (cl_mem)planeSrcV.ptr[0],
            planeSrcY.pitch[0], planeSrcU.pitch[0], planeSrcV.pitch[0], planeSrcY.width, planeSrcY.height, pCropParam->crop.e.left, pCropParam->crop.e.up,
            pCropParam->matrix);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_crop_yv12_rgb (convertCspFromYV12(RGB)(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
        return RGY_ERR_NONE;
    }
    static const auto supportedCspAYUV444 = make_array<RGY_CSP>(RGY_CSP_VUYA, RGY_CSP_VUYA_16);
    if (std::find(supportedCspAYUV444.begin(), supportedCspAYUV444.end(), pCropParam->frameOut.csp) != supportedCspAYUV444.end()) {
        auto planeSrcY = getPlane(pInputFrame, RGY_PLANE_Y);
        auto planeSrcU = getPlane(pInputFrame, RGY_PLANE_U);
        auto planeSrcV = getPlane(pInputFrame, RGY_PLANE_V);
        RGYWorkSize local(32, 8);
        RGYWorkSize global(planeSrcU.width, planeSrcU.height);
        auto err = copyProgram->kernel("kernel_crop_yv12_ayuv").config(queue, local, global, wait_events, event).launch(
            (cl_mem)pOutputFrame->ptr[0],
            pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
            (cl_mem)planeSrcY.ptr[0], (cl_mem)planeSrcU.ptr[0], (cl_mem)planeSrcV.ptr[0],
            pInputFrame->pitch[0], pCropParam->crop.e.left, pCropParam->crop.e.up,
            pInputFrame->width, pInputFrame->height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_copy_plane (convertCspFromYV12(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
        return RGY_ERR_NONE;
    }
    //Y
    if (pOutputFrame->csp == pInputFrame->csp) {
        auto planeDst = getPlane(pOutputFrame, RGY_PLANE_Y);
        auto planeSrc = getPlane(pInputFrame, RGY_PLANE_Y);
        auto err = m_cl->copyPlane(&planeDst, &planeSrc, &pCropParam->crop, queue, wait_events);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at copyPlane(Y) (convertCspFromYV12(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
    } else {
        auto planeDst = getPlane(pOutputFrame, RGY_PLANE_Y);
        auto planeSrc = getPlane(pInputFrame, RGY_PLANE_Y);
        RGYWorkSize local(32, 8);
        RGYWorkSize global(planeDst.width, planeDst.height);
        auto err = copyProgram->kernel("kernel_copy_plane").config(queue, local, global, wait_events, event).launch(
            (cl_mem)planeDst.ptr[0], planeDst.pitch[0], 0, 0,
            (cl_mem)planeSrc.ptr[0], planeSrc.pitch[0], pCropParam->crop.e.left, pCropParam->crop.e.up,
            planeSrc.width, planeSrc.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_copy_plane (convertCspFromYV12(Y)(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
    }
    static const auto supportedCspNV12 = make_array<RGY_CSP>(RGY_CSP_NV12, RGY_CSP_P010);
    static const auto supportedCspYUV444 = make_array<RGY_CSP>(RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);
    if (std::find(supportedCspNV12.begin(), supportedCspNV12.end(), pCropParam->frameOut.csp) != supportedCspNV12.end()) {
        //UV: yv12 -> nv12
        auto planeDstC = getPlane(pOutputFrame, RGY_PLANE_C);
        auto planeSrcU = getPlane(pInputFrame, RGY_PLANE_U);
        auto planeSrcV = getPlane(pInputFrame, RGY_PLANE_V);
        //cl_image_format val;
        //clGetImageInfo((cl_mem)planeDstC.ptr[0], CL_IMAGE_FORMAT, sizeof(val), &val, nullptr);
        RGYWorkSize local(32, 8);
        RGYWorkSize global(planeSrcU.width, planeSrcU.height);
        auto err = copyProgram->kernel("kernel_crop_yv12_nv12").config(queue, local, global, event).launch(
            (cl_mem)planeDstC.ptr[0], planeDstC.pitch[0],
            (cl_mem)planeSrcU.ptr[0], (cl_mem)planeSrcV.ptr[0], planeSrcU.pitch[0], planeSrcU.width, planeSrcU.height,
            pCropParam->crop.e.left, pCropParam->crop.e.up);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_crop_nv12_yv12 (convertCspFromYV12(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
    } else if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), pCropParam->frameOut.csp) != supportedCspYV12.end()) {
        for (int i = 1; i < 3; i++) {
            auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
            auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
            RGYWorkSize local(32, 8);
            RGYWorkSize global(planeDst.width, planeDst.height);
            auto err = copyProgram->kernel("kernel_copy_plane").config(queue, local, global, wait_events, event).launch(
                (cl_mem)planeDst.ptr[0], planeDst.pitch[0], 0, 0,
                (cl_mem)planeSrc.ptr[0], planeSrc.pitch[0], pCropParam->crop.e.left >> 1, pCropParam->crop.e.up >> 1,
                planeSrc.width, planeSrc.height);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at kernel_copy_plane (convertCspFromYV12(U)(%s -> %s)): %s.\n"),
                    RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
                return err;
            }
        }
    } else if (std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), pCropParam->frameOut.csp) != supportedCspYUV444.end()) {
        for (int i = 1; i < 3; i++) {
            auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
            auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
            RGYWorkSize local(32, 8);
            RGYWorkSize global(planeSrc.width, planeSrc.height);
            auto err = copyProgram->kernel("kernel_crop_c_yv12_yuv444").config(queue, local, global, wait_events, event).launch(
                (cl_mem)planeDst.ptr[0], planeDst.pitch[0], planeDst.width, planeDst.height,
                (cl_mem)planeSrc.ptr[0], planeSrc.pitch[0], planeSrc.width, planeSrc.height, pCropParam->crop.e.left, pCropParam->crop.e.up);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at kernel_copy_plane (convertCspFromYV12(U)(%s -> %s)): %s.\n"),
                    RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
                return err;
            }
        }
    } else {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterCspCrop::convertCspFromYUV444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto pCropParam = std::dynamic_pointer_cast<RGYFilterParamCrop>(m_param);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    static const auto supportedCspYUV444 = make_array<RGY_CSP>(RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);
    if (pOutputFrame->csp == pInputFrame->csp
        && std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), pCropParam->frameOut.csp) != supportedCspYUV444.end()) {
        auto err = m_cl->copyFrame(pOutputFrame, pInputFrame, &pCropParam->crop, queue, wait_events, event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at copyFrame (convertCspFromYUV444(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
    }

    auto copyProgram = m_cl->getCspCopyProgram(*pOutputFrame, *pInputFrame);
    if (!copyProgram) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_CL\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    if (RGY_CSP_CHROMA_FORMAT[pOutputFrame->csp] == RGY_CHROMAFMT_RGB) {
        auto planeDstR = getPlane(pOutputFrame, RGY_PLANE_R);
        auto planeDstG = getPlane(pOutputFrame, RGY_PLANE_G);
        auto planeDstB = getPlane(pOutputFrame, RGY_PLANE_B);
        auto planeSrcY = getPlane(pInputFrame, RGY_PLANE_Y);
        auto planeSrcU = getPlane(pInputFrame, RGY_PLANE_U);
        auto planeSrcV = getPlane(pInputFrame, RGY_PLANE_V);
        RGYWorkSize local(64, 4);
        RGYWorkSize global(pOutputFrame->width, pOutputFrame->height);
        auto err = copyProgram->kernel("kernel_crop_yuv444_rgb").config(queue, local, global, wait_events, event).launch(
            (cl_mem)planeDstR.ptr[0], (cl_mem)planeDstG.ptr[0], (cl_mem)planeDstB.ptr[0],
            planeDstR.pitch[0], planeDstR.width, planeDstR.height,
            (cl_mem)planeSrcY.ptr[0], (cl_mem)planeSrcU.ptr[0], (cl_mem)planeSrcV.ptr[0],
            planeSrcY.pitch[0], planeSrcY.width, planeSrcY.height, pCropParam->crop.e.left, pCropParam->crop.e.up,
            pCropParam->matrix);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_crop_yuv444_rgb (convertCspFromYUV444(RGB)(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
        return RGY_ERR_NONE;
    }
    static const auto supportedCspAYUV444 = make_array<RGY_CSP>(RGY_CSP_VUYA, RGY_CSP_VUYA_16);
    if (std::find(supportedCspAYUV444.begin(), supportedCspAYUV444.end(), pCropParam->frameOut.csp) != supportedCspAYUV444.end()) {
        auto planeSrcY = getPlane(pInputFrame, RGY_PLANE_Y);
        auto planeSrcU = getPlane(pInputFrame, RGY_PLANE_U);
        auto planeSrcV = getPlane(pInputFrame, RGY_PLANE_V);
        RGYWorkSize local(32, 8);
        RGYWorkSize global(pOutputFrame->width, pOutputFrame->height);
        auto err = copyProgram->kernel("kernel_crop_yuv444_ayuv").config(queue, local, global, wait_events, event).launch(
            (cl_mem)pOutputFrame->ptr[0],
            pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
            (cl_mem)planeSrcY.ptr[0], (cl_mem)planeSrcU.ptr[0], (cl_mem)planeSrcV.ptr[0],
            pInputFrame->pitch[0], pInputFrame->width, pInputFrame->height,
            pCropParam->crop.e.left, pCropParam->crop.e.up);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_copy_plane (convertCspFromYUV444(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
        return RGY_ERR_NONE;
    }
    //Y
    if (pOutputFrame->csp == pInputFrame->csp) {
        auto planeDst = getPlane(pOutputFrame, RGY_PLANE_Y);
        auto planeSrc = getPlane(pInputFrame, RGY_PLANE_Y);
        auto err = m_cl->copyPlane(&planeDst, &planeSrc, &pCropParam->crop, queue, wait_events);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at copyPlane(Y) (convertCspFromYUV444(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
    } else {
        auto planeDst = getPlane(pOutputFrame, RGY_PLANE_Y);
        auto planeSrc = getPlane(pInputFrame, RGY_PLANE_Y);
        RGYWorkSize local(32, 8);
        RGYWorkSize global(planeDst.width, planeDst.height);
        auto err = copyProgram->kernel("kernel_copy_plane").config(queue, local, global, wait_events, event).launch(
            (cl_mem)planeDst.ptr[0], planeDst.pitch[0], 0, 0,
            (cl_mem)planeSrc.ptr[0], planeSrc.pitch[0], pCropParam->crop.e.left, pCropParam->crop.e.up,
            planeSrc.width, planeSrc.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_copy_plane (convertCspFromYUV444(Y)(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
    }

    static const auto supportedCspNV12 = make_array<RGY_CSP>(RGY_CSP_NV12, RGY_CSP_P010);
    static const auto supportedCspYV12 = make_array<RGY_CSP>(RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
    if (std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), pCropParam->frameOut.csp) != supportedCspYUV444.end()) {
        for (int i = 1; i < 3; i++) {
            auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
            auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
            RGYWorkSize local(32, 8);
            RGYWorkSize global(planeSrc.width, planeSrc.height);
            auto err = copyProgram->kernel("kernel_copy_plane").config(queue, local, global, wait_events, event).launch(
                (cl_mem)planeDst.ptr[0], planeDst.pitch[0], planeDst.width, planeDst.height,
                (cl_mem)planeSrc.ptr[0], planeSrc.pitch[0], planeSrc.width, planeSrc.height, pCropParam->crop.e.left, pCropParam->crop.e.up);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at kernel_copy_plane (convertCspFromYUV444(U)(%s -> %s)): %s.\n"),
                    RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
                return err;
            }
        }
    } else if (std::find(supportedCspNV12.begin(), supportedCspNV12.end(), pCropParam->frameOut.csp) != supportedCspNV12.end()) {
        //UV: yuv444 -> nv12
        auto planeDstC = getPlane(pOutputFrame, RGY_PLANE_C);
        auto planeSrcU = getPlane(pInputFrame, RGY_PLANE_U);
        auto planeSrcV = getPlane(pInputFrame, RGY_PLANE_V);
        RGYWorkSize local(32, 8);
        RGYWorkSize global(planeDstC.width >> 1, planeDstC.height);
        auto err = copyProgram->kernel("kernel_crop_c_yuv444_nv12").config(queue, local, global, event).launch(
            (cl_mem)planeDstC.ptr[0], planeDstC.pitch[0], planeDstC.width >> 1, planeDstC.height,
            (cl_mem)planeSrcU.ptr[0], (cl_mem)planeSrcV.ptr[0], planeSrcU.pitch[0], planeSrcU.width, planeSrcU.height,
            pCropParam->crop.e.left, pCropParam->crop.e.up);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_crop_c_yuv444_nv12 (convertCspFromYUV444(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
    } else if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), pCropParam->frameOut.csp) != supportedCspYV12.end()) {
        auto planeDstU = getPlane(pOutputFrame, RGY_PLANE_U);
        auto planeDstV = getPlane(pOutputFrame, RGY_PLANE_V);
        auto planeSrcU = getPlane(pInputFrame, RGY_PLANE_U);
        auto planeSrcV = getPlane(pInputFrame, RGY_PLANE_V);
        for (int i = 1; i < 3; i++) {
            auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
            auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
            RGYWorkSize local(32, 8);
            RGYWorkSize global(planeDst.width, planeDst.height);
            auto err = copyProgram->kernel("kernel_crop_c_yuv444_yv12").config(queue, local, global, wait_events, event).launch(
                (cl_mem)planeDst.ptr[0], planeDst.pitch[0], planeDst.width, planeDst.height,
                (cl_mem)planeSrc.ptr[0], planeSrc.pitch[0], planeSrc.width, planeSrc.height, pCropParam->crop.e.left, pCropParam->crop.e.up);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at kernel_copy_plane (convertCspFromYUV444(U)(%s -> %s)): %s.\n"),
                    RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
                return err;
            }
        }
    } else {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterCspCrop::convertCspFromAYUVPacked444(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto pCropParam = std::dynamic_pointer_cast<RGYFilterParamCrop>(m_param);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    static const auto supportedCspAYUV444 = make_array<RGY_CSP>(RGY_CSP_VUYA, RGY_CSP_VUYA_16);
    if (pOutputFrame->csp == pInputFrame->csp
        && std::find(supportedCspAYUV444.begin(), supportedCspAYUV444.end(), pCropParam->frameOut.csp) != supportedCspAYUV444.end()) {
        auto err = m_cl->copyFrame(pOutputFrame, pInputFrame, &pCropParam->crop, queue, wait_events, event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at copyFrame (convertCspFromAYUVPacked444(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
    }

    auto copyProgram = m_cl->getCspCopyProgram(*pOutputFrame, *pInputFrame);
    if (!copyProgram) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_CL\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    static const auto supportedCspYUV444 = make_array<RGY_CSP>(RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);
    static const auto supportedCspYV12 = make_array<RGY_CSP>(RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
    auto planeDstY = getPlane(pOutputFrame, RGY_PLANE_Y);
    auto planeDstU = getPlane(pOutputFrame, RGY_PLANE_U);
    auto planeDstV = getPlane(pOutputFrame, RGY_PLANE_V);
    if (std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), pCropParam->frameOut.csp) != supportedCspYUV444.end()) {
        RGYWorkSize local(32, 8);
        RGYWorkSize global(pInputFrame->width, pInputFrame->height);
        auto err = copyProgram->kernel("kernel_crop_ayuv_yuv444").config(queue, local, global, wait_events, event).launch(
            (cl_mem)planeDstY.ptr[0], (cl_mem)planeDstU.ptr[0], (cl_mem)planeDstV.ptr[0],
            planeDstY.pitch[0], planeDstY.width, planeDstY.height,
            (cl_mem)pInputFrame->ptr[0], pInputFrame->pitch[0], pCropParam->crop.e.left, pCropParam->crop.e.up,
            pInputFrame->width, pInputFrame->height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_crop_ayuv_yuv444 (convertCspFromAYUVPacked444(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
    } else if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), pCropParam->frameOut.csp) != supportedCspYV12.end()) {
        RGYWorkSize local(32, 8);
        RGYWorkSize global(planeDstU.width, planeDstU.height);
        auto err = copyProgram->kernel("kernel_crop_ayuv_yv12").config(queue, local, global, wait_events, event).launch(
            (cl_mem)planeDstY.ptr[0], (cl_mem)planeDstU.ptr[0], (cl_mem)planeDstV.ptr[0],
            planeDstY.pitch[0], planeDstY.width, planeDstY.height,
            (cl_mem)pInputFrame->ptr[0], pInputFrame->pitch[0], pCropParam->crop.e.left, pCropParam->crop.e.up,
            pInputFrame->width, pInputFrame->height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_crop_ayuv_yv12 (convertCspFromAYUVPacked444(%s -> %s)): %s.\n"),
                RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
            return err;
        }
    } else {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterCspCrop::convertCspFromRGB(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto pCropParam = std::dynamic_pointer_cast<RGYFilterParamCrop>(m_param);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (RGY_CSP_CHROMA_FORMAT[pInputFrame->csp] != RGY_CHROMAFMT_RGB_PACKED
        && RGY_CSP_CHROMA_FORMAT[pInputFrame->csp] != RGY_CHROMAFMT_RGB) {
        AddMessage(RGY_LOG_ERROR, _T("Conversion from %s not supported.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_INVALID_PARAM;
    }

    auto copyProgram = m_cl->getCspCopyProgram(*pOutputFrame, *pInputFrame);
    if (!copyProgram) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_CL\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    static const auto supportedCspYUV444 = make_array<RGY_CSP>(RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);
    static const auto supportedCspYV12 = make_array<RGY_CSP>(RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
    static const auto supportedCspNV12 = make_array<RGY_CSP>(RGY_CSP_NV12, RGY_CSP_P010);
    if (RGY_CSP_CHROMA_FORMAT[pInputFrame->csp] == RGY_CHROMAFMT_RGB_PACKED) {
        auto inputRGBOrder = rgb_order_str(RGY_CSP_RGB_ORDER[pInputFrame->csp], false);
        if (!inputRGBOrder) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported rgb order: %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
            return RGY_ERR_UNSUPPORTED;
        }
        if (std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), pCropParam->frameOut.csp) != supportedCspYUV444.end()) {
            auto kernelName = tstring(_T("kernel_crop_")) + inputRGBOrder + _T("32_yuv444");
            auto planeDstY = getPlane(pOutputFrame, RGY_PLANE_Y);
            auto planeDstU = getPlane(pOutputFrame, RGY_PLANE_U);
            auto planeDstV = getPlane(pOutputFrame, RGY_PLANE_V);
            RGYWorkSize local(32, 8);
            RGYWorkSize global(pInputFrame->width, pInputFrame->height);
            auto err = copyProgram->kernel(tchar_to_string(kernelName).c_str()).config(queue, local, global, wait_events, event).launch(
                (cl_mem)planeDstY.ptr[0], (cl_mem)planeDstU.ptr[0], (cl_mem)planeDstV.ptr[0],
                planeDstY.pitch[0], planeDstY.width, planeDstY.height,
                (cl_mem)pInputFrame->ptr[0], pInputFrame->pitch[0],
                pInputFrame->width, pInputFrame->height,
                pCropParam->crop.e.left, pCropParam->crop.e.up, pCropParam->matrix);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at %s (convertCspFromRGB(%s -> %s)): %s.\n"),
                    kernelName.c_str(), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
                return err;
            }
        } else if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), pCropParam->frameOut.csp) != supportedCspYV12.end()) {
            auto kernelName = tstring(_T("kernel_crop_")) + inputRGBOrder + _T("32_yv12");
            auto planeDstY = getPlane(pOutputFrame, RGY_PLANE_Y);
            auto planeDstU = getPlane(pOutputFrame, RGY_PLANE_U);
            auto planeDstV = getPlane(pOutputFrame, RGY_PLANE_V);
            RGYWorkSize local(32, 8);
            RGYWorkSize global(planeDstY.width >> 1, planeDstY.height >> 1);
            auto err = copyProgram->kernel(tchar_to_string(kernelName).c_str()).config(queue, local, global, wait_events, event).launch(
                (cl_mem)planeDstY.ptr[0], (cl_mem)planeDstU.ptr[0], (cl_mem)planeDstV.ptr[0],
                planeDstY.pitch[0], planeDstU.pitch[0], planeDstV.pitch[0],
                planeDstY.width, planeDstY.height,
                (cl_mem)pInputFrame->ptr[0], pInputFrame->pitch[0],
                pInputFrame->width, pInputFrame->height,
                pCropParam->crop.e.left, pCropParam->crop.e.up, pCropParam->matrix);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at %s (convertCspFromRGB(%s -> %s)): %s.\n"),
                    kernelName.c_str(), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
                return err;
            }
        } else if (std::find(supportedCspNV12.begin(), supportedCspNV12.end(), pCropParam->frameOut.csp) != supportedCspNV12.end()) {
            auto kernelName = tstring(_T("kernel_crop_")) + inputRGBOrder + _T("32_nv12");
            auto planeDstY = getPlane(pOutputFrame, RGY_PLANE_Y);
            auto planeDstC = getPlane(pOutputFrame, RGY_PLANE_C);
            RGYWorkSize local(32, 8);
            RGYWorkSize global(planeDstY.width >> 1, planeDstY.height >> 1);
            auto err = copyProgram->kernel(tchar_to_string(kernelName).c_str()).config(queue, local, global, wait_events, event).launch(
                (cl_mem)planeDstY.ptr[0], (cl_mem)planeDstC.ptr[0],
                planeDstY.pitch[0], planeDstC.pitch[0], planeDstY.width, planeDstY.height,
                (cl_mem)pInputFrame->ptr[0], pInputFrame->pitch[0],
                pInputFrame->width, pInputFrame->height,
                pCropParam->crop.e.left, pCropParam->crop.e.up, pCropParam->matrix);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at %s (convertCspFromRGB(%s -> %s)): %s.\n"),
                    kernelName.c_str(), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
                return err;
            }
        } else if (RGY_CSP_CHROMA_FORMAT[pOutputFrame->csp] == RGY_CHROMAFMT_RGB) {
            auto kernelName = tstring(_T("kernel_crop_")) + inputRGBOrder + _T("32_rgb");
            auto planeDstR = getPlane(pOutputFrame, RGY_PLANE_R);
            auto planeDstG = getPlane(pOutputFrame, RGY_PLANE_G);
            auto planeDstB = getPlane(pOutputFrame, RGY_PLANE_B);
            RGYWorkSize local(32, 8);
            RGYWorkSize global(planeDstR.width, planeDstR.height);
            auto err = copyProgram->kernel(tchar_to_string(kernelName).c_str()).config(queue, local, global, wait_events, event).launch(
                (cl_mem)planeDstR.ptr[0], (cl_mem)planeDstG.ptr[0], (cl_mem)planeDstB.ptr[0],
                planeDstR.pitch[0], planeDstR.width, planeDstR.height,
                (cl_mem)pInputFrame->ptr[0], pInputFrame->pitch[0],
                pInputFrame->width, pInputFrame->height,
                pCropParam->crop.e.left, pCropParam->crop.e.up);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at %s (convertCspFromRGB(%s -> %s)): %s.\n"),
                    kernelName.c_str(), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
                return err;
            }
        } else {
            AddMessage(RGY_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
            return RGY_ERR_UNSUPPORTED;
        }
    } else { //RGY_CSP_CHROMA_FORMAT[pInputFrame->csp] == RGY_CHROMAFMT_RGB
        if (std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), pCropParam->frameOut.csp) != supportedCspYUV444.end()) {
            auto planeDstY = getPlane(pOutputFrame, RGY_PLANE_Y);
            auto planeDstU = getPlane(pOutputFrame, RGY_PLANE_U);
            auto planeDstV = getPlane(pOutputFrame, RGY_PLANE_V);
            auto planeSrcR = getPlane(pInputFrame, RGY_PLANE_R);
            auto planeSrcG = getPlane(pInputFrame, RGY_PLANE_G);
            auto planeSrcB = getPlane(pInputFrame, RGY_PLANE_B);
            RGYWorkSize local(64, 4);
            RGYWorkSize global(pInputFrame->width, pInputFrame->height);
            auto err = copyProgram->kernel("kernel_crop_rgb_yuv444").config(queue, local, global, wait_events, event).launch(
                (cl_mem)planeDstY.ptr[0], (cl_mem)planeDstU.ptr[0], (cl_mem)planeDstV.ptr[0],
                planeDstY.pitch[0], planeDstY.width, planeDstY.height,
                (cl_mem)planeSrcR.ptr[0], (cl_mem)planeSrcG.ptr[0], (cl_mem)planeSrcB.ptr[0],
                planeSrcR.pitch[0], planeSrcR.width, planeSrcR.height,
                pCropParam->crop.e.left, pCropParam->crop.e.up, pCropParam->matrix);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at kernel_crop_rgb_yuv444 (convertCspFromRGB(%s -> %s)): %s.\n"),
                    RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
                return err;
            }
        } else if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), pCropParam->frameOut.csp) != supportedCspYV12.end()) {
            auto planeDstY = getPlane(pOutputFrame, RGY_PLANE_Y);
            auto planeDstU = getPlane(pOutputFrame, RGY_PLANE_U);
            auto planeDstV = getPlane(pOutputFrame, RGY_PLANE_V);
            auto planeSrcR = getPlane(pInputFrame, RGY_PLANE_R);
            auto planeSrcG = getPlane(pInputFrame, RGY_PLANE_G);
            auto planeSrcB = getPlane(pInputFrame, RGY_PLANE_B);
            RGYWorkSize local(64, 4);
            RGYWorkSize global(planeDstY.width >> 1, planeDstY.height >> 1);
            auto err = copyProgram->kernel("kernel_crop_rgb_yv12").config(queue, local, global, wait_events, event).launch(
                (cl_mem)planeDstY.ptr[0], (cl_mem)planeDstU.ptr[0], (cl_mem)planeDstV.ptr[0],
                planeDstY.pitch[0], planeDstU.pitch[0], planeDstV.pitch[0],
                planeDstY.width, planeDstY.height,
                (cl_mem)planeSrcR.ptr[0], (cl_mem)planeSrcG.ptr[0], (cl_mem)planeSrcB.ptr[0],
                planeSrcR.pitch[0], planeSrcR.width, planeSrcR.height,
                pCropParam->crop.e.left, pCropParam->crop.e.up, pCropParam->matrix);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at kernel_crop_rgb_yv12 (convertCspFromRGB(%s -> %s)): %s.\n"),
                    RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
                return err;
            }
        } else if (std::find(supportedCspNV12.begin(), supportedCspNV12.end(), pCropParam->frameOut.csp) != supportedCspNV12.end()) {
            auto planeDstY = getPlane(pOutputFrame, RGY_PLANE_Y);
            auto planeDstC = getPlane(pOutputFrame, RGY_PLANE_C);
            auto planeSrcR = getPlane(pInputFrame, RGY_PLANE_R);
            auto planeSrcG = getPlane(pInputFrame, RGY_PLANE_G);
            auto planeSrcB = getPlane(pInputFrame, RGY_PLANE_B);
            RGYWorkSize local(64, 4);
            RGYWorkSize global(planeDstY.width >> 1, planeDstY.height >> 1);
            auto err = copyProgram->kernel("kernel_crop_rgb_nv12").config(queue, local, global, wait_events, event).launch(
                (cl_mem)planeDstY.ptr[0], (cl_mem)planeDstC.ptr[0],
                planeDstY.pitch[0], planeDstC.pitch[0], planeDstY.width, planeDstY.height,
                (cl_mem)planeSrcR.ptr[0], (cl_mem)planeSrcG.ptr[0], (cl_mem)planeSrcB.ptr[0],
                planeSrcR.pitch[0], planeSrcR.width, planeSrcR.height,
                pCropParam->crop.e.left, pCropParam->crop.e.up, pCropParam->matrix);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at kernel_crop_rgb_nv12 (convertCspFromRGB(%s -> %s)): %s.\n"),
                    RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
                return err;
            }
        } else if (RGY_CSP_CHROMA_FORMAT[pOutputFrame->csp] == RGY_CHROMAFMT_RGB_PACKED) {
            auto outputRGBOrder = rgb_order_str(RGY_CSP_RGB_ORDER[pOutputFrame->csp], false);
            auto kernelName = tstring(_T("kernel_crop_rgb_")) + outputRGBOrder + _T("32");
            auto planeSrcR = getPlane(pInputFrame, RGY_PLANE_R);
            auto planeSrcG = getPlane(pInputFrame, RGY_PLANE_G);
            auto planeSrcB = getPlane(pInputFrame, RGY_PLANE_B);
            RGYWorkSize local(64, 4);
            RGYWorkSize global(pOutputFrame->width, pOutputFrame->height);
            auto err = copyProgram->kernel(tchar_to_string(kernelName).c_str()).config(queue, local, global, wait_events, event).launch(
                (cl_mem)pOutputFrame->ptr[0], pOutputFrame->pitch[0], pOutputFrame->width, pOutputFrame->height,
                (cl_mem)planeSrcR.ptr[0], (cl_mem)planeSrcG.ptr[0], (cl_mem)planeSrcB.ptr[0],
                planeSrcR.pitch[0], planeSrcR.width, planeSrcR.height,
                pCropParam->crop.e.left, pCropParam->crop.e.up);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at %s (convertCspFromRGB(%s -> %s)): %s.\n"),
                    kernelName.c_str(), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp], get_err_mes(err));
                return err;
            }
        } else {
            AddMessage(RGY_LOG_ERROR, _T("unsupported csp conversion: %s -> %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], RGY_CSP_NAMES[pOutputFrame->csp]);
            return RGY_ERR_UNSUPPORTED;
        }
    }
    return RGY_ERR_NONE;
}


RGYFilterCspCrop::RGYFilterCspCrop(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context) {
    m_name = _T("copy/cspconv/crop");
}

RGYFilterCspCrop::~RGYFilterCspCrop() {
    close();
}

RGY_ERR RGYFilterCspCrop::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto pCropParam = std::dynamic_pointer_cast<RGYFilterParamCrop>(pParam);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //フィルタ名の調整
    m_name = _T("");
    if (cropEnabled(pCropParam->crop)) {
        m_name += _T("crop");
    }
    if (pCropParam->frameOut.csp != pCropParam->frameIn.csp) {
        m_name += (m_name.length()) ? _T("/cspconv") : _T("cspconv");
    }
    if (m_name.length() == 0) {
        const auto memcpyKind = getMemcpyKind(pParam->frameIn.mem_type, pParam->frameOut.mem_type);
        m_name += getMemcpyKindStr(memcpyKind);
    }
    //パラメータチェック
    for (int i = 0; i < _countof(pCropParam->crop.c); i++) {
        if ((pCropParam->crop.c[i] & 1) != 0) {
            AddMessage(RGY_LOG_ERROR, _T("crop should be divided by 2.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
    }
    pCropParam->frameOut.height = pCropParam->frameIn.height - pCropParam->crop.e.bottom - pCropParam->crop.e.up;
    pCropParam->frameOut.width = pCropParam->frameIn.width - pCropParam->crop.e.left - pCropParam->crop.e.right;
    if (pCropParam->frameOut.height <= 0 || pCropParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("crop size is too big.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    m_cl->requestCSPCopy(pCropParam->frameOut, pCropParam->frameIn);

    auto err = AllocFrameBuf(pCropParam->frameOut, 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
        return RGY_ERR_MEMORY_ALLOC;
    }
    memcpy(pCropParam->frameOut.pitch, m_frameBuf[0]->frame.pitch, sizeof(m_frameBuf[0]->frame.pitch));

    //フィルタ情報の調整
    m_infoStr = _T("");
    if (cropEnabled(pCropParam->crop)) {
        m_infoStr += strsprintf(_T("crop: %d,%d,%d,%d"), pCropParam->crop.e.left, pCropParam->crop.e.up, pCropParam->crop.e.right, pCropParam->crop.e.bottom);
    }
    if (pCropParam->frameOut.csp != pCropParam->frameIn.csp) {
        m_infoStr += (m_infoStr.length()) ? _T("/cspconv") : _T("cspconv");
        m_infoStr += strsprintf(_T("(%s -> %s)"), RGY_CSP_NAMES[pCropParam->frameIn.csp], RGY_CSP_NAMES[pCropParam->frameOut.csp]);
        if (   RGY_CSP_CHROMA_FORMAT[pCropParam->frameIn.csp]  == RGY_CHROMAFMT_RGB
            || RGY_CSP_CHROMA_FORMAT[pCropParam->frameOut.csp] == RGY_CHROMAFMT_RGB) {
            if (   pCropParam->matrix == RGY_MATRIX_BT709
                || pCropParam->matrix == RGY_MATRIX_BT470_BG
                || pCropParam->matrix == RGY_MATRIX_ST170_M
                || pCropParam->matrix == RGY_MATRIX_BT2020_NCL
                || pCropParam->matrix == RGY_MATRIX_BT2020_CL) {
                //filterInfo += strsprintf(_T(" [%s]"), get_cx_desc(list_colormatrix, matrix));
            }
        }
    }
    if (m_infoStr.length() == 0) {
        m_infoStr += getMemcpyKindStr(pCropParam->frameIn.mem_type, pCropParam->frameOut.mem_type);
    }

    m_param = pCropParam;
    return sts;
}

RGY_ERR RGYFilterCspCrop::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;

    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
    }
    auto pCropParam = std::dynamic_pointer_cast<RGYFilterParamCrop>(m_param);
    if (!pCropParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (m_param->frameOut.csp == m_param->frameIn.csp) {
        //cropがなければ、一度に転送可能
        auto err = m_cl->copyFrame(ppOutputFrames[0], pInputFrame, &pCropParam->crop, queue, wait_events, event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to copy frame: %s.\n"), get_err_mes(err));
            return RGY_ERR_INVALID_PARAM;
        }
    } else if (memcpyKind != RGYCLMemcpyD2D) {
        AddMessage(RGY_LOG_ERROR, _T("converting csp while copying from host to device is not supported.\n"));
        return RGY_ERR_UNSUPPORTED;
    } else {
        //色空間変換
        static const auto supportedCspNV12   = make_array<RGY_CSP>(RGY_CSP_NV12, RGY_CSP_P010);
        static const auto supportedCspYV12   = make_array<RGY_CSP>(RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
        static const auto supportedCspYUV444 = make_array<RGY_CSP>(RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);
        static const auto supportedCspAYUV444 = make_array<RGY_CSP>(RGY_CSP_VUYA, RGY_CSP_VUYA_16);
#if 0
        static const auto supportedCspNV16   = make_array<RGY_CSP>(RGY_CSP_NV16, RGY_CSP_P210);
#endif
        static const auto supportedCspRGB    = make_array<RGY_CSP>(
            RGY_CSP_RGB32,
            RGY_CSP_RGB, RGY_CSP_RGBA, RGY_CSP_GBR, RGY_CSP_GBRA,
            RGY_CSP_RGB_16, RGY_CSP_RGBA_16, RGY_CSP_BGR_16, RGY_CSP_BGRA_16,
            RGY_CSP_RGB_F32, RGY_CSP_RGBA_F32, RGY_CSP_BGR_F32, RGY_CSP_BGRA_F32);
        if (std::find(supportedCspNV12.begin(), supportedCspNV12.end(), pCropParam->frameIn.csp) != supportedCspNV12.end()) {
            sts = convertCspFromNV12(ppOutputFrames[0], pInputFrame, queue, wait_events, event);
        } else if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), pCropParam->frameIn.csp) != supportedCspYV12.end()) {
            sts = convertCspFromYV12(ppOutputFrames[0], pInputFrame, queue, wait_events, event);
        } else if (std::find(supportedCspYUV444.begin(), supportedCspYUV444.end(), pCropParam->frameIn.csp) != supportedCspYUV444.end()) {
            sts = convertCspFromYUV444(ppOutputFrames[0], pInputFrame, queue, wait_events, event);
        } else if (std::find(supportedCspAYUV444.begin(), supportedCspAYUV444.end(), pCropParam->frameIn.csp) != supportedCspAYUV444.end()) {
            sts = convertCspFromAYUVPacked444(ppOutputFrames[0], pInputFrame, queue, wait_events, event);
#if 0
        } else if (std::find(supportedCspNV16.begin(), supportedCspNV16.end(), pCropParam->frameIn.csp) != supportedCspNV16.end()) {
            sts = convertCspFromNV16(ppOutputFrames[0], pInputFrame, queue, wait_events, event);
#endif
        } else if (RGY_CSP_CHROMA_FORMAT[pInputFrame->csp] == RGY_CHROMAFMT_RGB || RGY_CSP_CHROMA_FORMAT[pInputFrame->csp] == RGY_CHROMAFMT_RGB_PACKED) {
            sts = convertCspFromRGB(ppOutputFrames[0], pInputFrame, queue, wait_events, event);
        } else {
            AddMessage(RGY_LOG_ERROR, _T("converting csp from %s is not supported.\n"), RGY_CSP_NAMES[pCropParam->frameIn.csp]);
            sts = RGY_ERR_UNSUPPORTED;
        }
    }
    return sts;
}

void RGYFilterCspCrop::close() {
    m_frameBuf.clear();
    m_cl.reset();
}
