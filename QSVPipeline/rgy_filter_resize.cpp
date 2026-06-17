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

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>
#include <map>
#include <array>
#include "convert_csp.h"
#include "rgy_filter_resize.h"
#include "rgy_filter_libplacebo.h"
#include "rgy_prm.h"
#include "nis_coef_tables.h"

static const int RESIZE_BLOCK_X = 32;
static const int RESIZE_BLOCK_Y = 8;
static_assert(RESIZE_BLOCK_Y <= RESIZE_BLOCK_X, "RESIZE_BLOCK_Y <= RESIZE_BLOCK_X");

// NIS thread-group geometry. Matches NIS_Config.h
// NISOptimizer recommendations for Intel/AMD/NVIDIA generic: 32x24 for
// upscale, 128 threads per group. Compile-time so the kernel can use
// reqd_work_group_size for the launcher to plan around.
static const int NIS_BLOCK_WIDTH       = 32;
static const int NIS_BLOCK_HEIGHT      = 24;
static const int NIS_THREAD_GROUP_SIZE = 128;

// Host mirror of the .cl NISConfigCL. Field order, types, and 256-byte
// alignment MUST stay in lock-step with the kernel-side struct.
struct alignas(256) NISConfigHost {
    float kDetectRatio;
    float kDetectThres;
    float kMinContrastRatio;
    float kRatioNorm;

    float kContrastBoost;
    float kEps;
    float kSharpStartY;
    float kSharpScaleY;

    float kSharpStrengthMin;
    float kSharpStrengthScale;
    float kSharpLimitMin;
    float kSharpLimitScale;

    float kScaleX;
    float kScaleY;
    float kDstNormX;
    float kDstNormY;

    float kSrcNormX;
    float kSrcNormY;

    uint32_t kInputViewportOriginX;
    uint32_t kInputViewportOriginY;
    uint32_t kInputViewportWidth;
    uint32_t kInputViewportHeight;

    uint32_t kOutputViewportOriginX;
    uint32_t kOutputViewportOriginY;
    uint32_t kOutputViewportWidth;
    uint32_t kOutputViewportHeight;

    float reserved0;
    float reserved1;
};
static_assert(sizeof(NISConfigHost) == 256, "NISConfigHost must be 256 bytes to match aligned NIS_Config.h NISConfig.");

// Port of NIS_Config.h NVScalerUpdateConfig (lines 156..254). Pure math,
// no NIS headers pulled in (the originals depend on <algorithm> which
// is fine; the port keeps the same numerics). Returns true if the
// requested scale lies in NIS's supported 0.5..1.0 source-step range
// (i.e. 1.0x..2.0x upscale). >2x cases are handled by the cascade
// orchestrator.
static bool nisBuildConfig(NISConfigHost &config, float sharpness, int hdrMode,
        uint32_t inputViewportWidth, uint32_t inputViewportHeight,
        uint32_t inputTextureWidth, uint32_t inputTextureHeight,
        uint32_t outputViewportWidth, uint32_t outputViewportHeight,
        uint32_t outputTextureWidth, uint32_t outputTextureHeight) {
    sharpness = std::max(0.0f, std::min(1.0f, sharpness));
    const float sharpen_slider = sharpness - 0.5f;
    const float MaxScale   = (sharpen_slider >= 0.0f) ? 1.25f : 1.75f;
    const float MinScale   = (sharpen_slider >= 0.0f) ? 1.25f : 1.0f;
    const float LimitScale = (sharpen_slider >= 0.0f) ? 1.25f : 1.0f;

    float kDetectRatio = 2 * 1127.f / 1024.f;
    float kDetectThres = 64.0f / 1024.0f;
    float kMinContrastRatio = 2.0f;
    float kMaxContrastRatio = 10.0f;

    float kSharpStartY = 0.45f;
    float kSharpEndY   = 0.9f;
    float kSharpStrengthMin = std::max(0.0f, 0.4f + sharpen_slider * MinScale * 1.2f);
    float kSharpStrengthMax = 1.6f + sharpen_slider * MaxScale * 1.8f;
    float kSharpLimitMin    = std::max(0.1f,  0.14f + sharpen_slider * LimitScale * 0.32f);
    float kSharpLimitMax    = 0.5f + sharpen_slider * LimitScale * 0.6f;

    // hdrMode: 0=None, 1=Linear, 2=PQ (matches the .cl NIS_HDR_MODE define).
    if (hdrMode == 1 || hdrMode == 2) {
        kDetectThres = 32.0f / 1024.0f;
        kMinContrastRatio = 1.5f;
        kMaxContrastRatio = 5.0f;
        kSharpStrengthMin = std::max(0.0f,  0.4f + sharpen_slider * MinScale * 1.1f);
        kSharpStrengthMax = 2.2f + sharpen_slider * MaxScale * 1.8f;
        kSharpLimitMin    = std::max(0.06f, 0.10f + sharpen_slider * LimitScale * 0.28f);
        kSharpLimitMax    = 0.6f + sharpen_slider * LimitScale * 0.6f;
        if (hdrMode == 2) {
            kSharpStartY = 0.35f;  // PQ specular-protect band
            kSharpEndY   = 0.55f;
        } else {
            kSharpStartY = 0.3f;
            kSharpEndY   = 0.5f;
        }
    }

    const float kRatioNorm = 1.0f / (kMaxContrastRatio - kMinContrastRatio);
    const float kSharpScaleY = 1.0f / (kSharpEndY - kSharpStartY);
    const float kSharpStrengthScale = kSharpStrengthMax - kSharpStrengthMin;
    const float kSharpLimitScale    = kSharpLimitMax - kSharpLimitMin;

    config.kInputViewportWidth   = inputViewportWidth  == 0 ? inputTextureWidth  : inputViewportWidth;
    config.kInputViewportHeight  = inputViewportHeight == 0 ? inputTextureHeight : inputViewportHeight;
    config.kOutputViewportWidth  = outputViewportWidth  == 0 ? outputTextureWidth  : outputViewportWidth;
    config.kOutputViewportHeight = outputViewportHeight == 0 ? outputTextureHeight : outputViewportHeight;
    if (config.kInputViewportWidth == 0 || config.kInputViewportHeight == 0
        || config.kOutputViewportWidth == 0 || config.kOutputViewportHeight == 0) return false;

    config.kInputViewportOriginX  = 0;
    config.kInputViewportOriginY  = 0;
    config.kOutputViewportOriginX = 0;
    config.kOutputViewportOriginY = 0;

    config.kSrcNormX = 1.f / inputTextureWidth;
    config.kSrcNormY = 1.f / inputTextureHeight;
    config.kDstNormX = 1.f / outputTextureWidth;
    config.kDstNormY = 1.f / outputTextureHeight;
    config.kScaleX = config.kInputViewportWidth  / (float)config.kOutputViewportWidth;
    config.kScaleY = config.kInputViewportHeight / (float)config.kOutputViewportHeight;
    config.kDetectRatio = kDetectRatio;
    config.kDetectThres = kDetectThres;
    config.kMinContrastRatio = kMinContrastRatio;
    config.kRatioNorm = kRatioNorm;
    config.kContrastBoost = 1.0f;
    config.kEps = 1.0f / 255.0f;
    config.kSharpStartY = kSharpStartY;
    config.kSharpScaleY = kSharpScaleY;
    config.kSharpStrengthMin   = kSharpStrengthMin;
    config.kSharpStrengthScale = kSharpStrengthScale;
    config.kSharpLimitMin      = kSharpLimitMin;
    config.kSharpLimitScale    = kSharpLimitScale;
    config.reserved0 = 0.0f;
    config.reserved1 = 0.0f;

    if (config.kScaleX < 0.5f || config.kScaleX > 1.f
        || config.kScaleY < 0.5f || config.kScaleY > 1.f) return false;
    return true;
}

// hdr=auto resolution. Maps the input's VUI transfer characteristic to
// NIS's HDR mode enum (0=None, 1=Linear, 2=PQ -- matches the .cl
// NIS_HDR_MODE define and NIS_Config.h NISHDRMode).
//
//   ST2084  (BT.2020 PQ)   -> 2  (PQ band 0.35..0.55, specular-protect)
//   ARIB_B67 (BT.2020 HLG) -> 1  (linear HDR band 0.30..0.50)
//   anything else           -> 0  (SDR band 0.45..0.90)
//
// Explicit hdr=sdr / hdr=pq from CLI always wins over auto.
static int nisResolveHdrMode(const RGYFilterParamResize *param) {
    switch (param->nis.hdrMode) {
    case RGY_NIS_HDR_SDR: return 0;
    case RGY_NIS_HDR_PQ:  return 2;
    case RGY_NIS_HDR_AUTO:
    default:
        if (param->vui.transfer == RGY_TRANSFER_ST2084)   return 2;
        if (param->vui.transfer == RGY_TRANSFER_ARIB_B67) return 1;
        return 0;
    }
}

static inline int get_radius(const RGY_VPP_RESIZE_ALGO interp) {
    int radius = 1;
    switch (interp) {
    case RGY_VPP_RESIZE_BICUBIC:
    case RGY_VPP_RESIZE_LANCZOS2:
    case RGY_VPP_RESIZE_SPLINE16:
        radius = 2;
        break;
    case RGY_VPP_RESIZE_SPLINE36:
    case RGY_VPP_RESIZE_LANCZOS3:
        radius = 3;
        break;
    case RGY_VPP_RESIZE_LANCZOS4:
    case RGY_VPP_RESIZE_SPLINE64:
    case RGY_VPP_RESIZE_GAUSS:
        radius = 4;
        break;
    case RGY_VPP_RESIZE_BILINEAR:
    default:
        break;
    }
    return radius;
}

enum RESIZE_WEIGHT_TYPE {
    WEIGHT_UNKNOWN,
    WEIGHT_BILINEAR,
    WEIGHT_BICUBIC,
    WEIGHT_LANCZOS,
    WEIGHT_SPLINE,
    WEIGHT_GAUSS,
};

static inline RESIZE_WEIGHT_TYPE get_weight_type(const RGY_VPP_RESIZE_ALGO interp) {
    auto type = WEIGHT_UNKNOWN;
    switch (interp) {
    case RGY_VPP_RESIZE_BILINEAR:
        type = WEIGHT_BILINEAR;
        break;
    case RGY_VPP_RESIZE_BICUBIC:
        type = WEIGHT_BICUBIC;
        break;
    case RGY_VPP_RESIZE_LANCZOS2:
    case RGY_VPP_RESIZE_LANCZOS3:
    case RGY_VPP_RESIZE_LANCZOS4:
        type = WEIGHT_LANCZOS;
        break;
    case RGY_VPP_RESIZE_SPLINE16:
    case RGY_VPP_RESIZE_SPLINE36:
    case RGY_VPP_RESIZE_SPLINE64:
        type = WEIGHT_SPLINE;
        break;
    case RGY_VPP_RESIZE_GAUSS:
        type = WEIGHT_GAUSS;
        break;
    default:
        break;
    }
    return type;
}

static float getSrcWindow(const int radius, const int dst_size, const int src_size) {
    const float ratio = (float)(dst_size) / src_size;
    const float ratioClamped = std::min(ratio, 1.0f);
    const float srcWindow = radius / ratioClamped;
    return srcWindow;
}

static bool useTextureBilinear(const RGYFilterParamResize *param) {
    return param->interp == RGY_VPP_RESIZE_BILINEAR
        && param->frameOut.width > param->frameIn.width
        && param->frameOut.height > param->frameIn.height;
}

RGY_ERR RGYFilterResize::resizePlaneFsr(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
    cl_mem midMem, int midPitchBytes, int midWidth, int midHeight,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto pResizeParam = std::dynamic_pointer_cast<RGYFilterParamResize>(m_param);
    if (!pResizeParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const float ratioInvX = (float)pInputPlane->width / (float)pOutputPlane->width;
    const float ratioInvY = (float)pInputPlane->height / (float)pOutputPlane->height;
    const float offsetX = 0.5f * ratioInvX - 0.5f;
    const float offsetY = 0.5f * ratioInvY - 0.5f;
    const float sharpness_user = clamp(pResizeParam->fsr1.sharpness, 0.0f, 1.0f);
    const float stops = (1.0f - sharpness_user) * 4.0f;
    const float con0_sharp = (float)std::exp2(-stops);

    RGYOpenCLEvent eventEasu;
    {
        const char *kernel_name = "kernel_easu";
        RGYWorkSize local(RESIZE_BLOCK_X, RESIZE_BLOCK_Y);
        RGYWorkSize global(midWidth, midHeight);
        auto err = m_resize.get()->kernel(kernel_name).config(queue, local, global, wait_events, &eventEasu).launch(
            midMem, midPitchBytes, midWidth, midHeight,
            (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0], pInputPlane->width, pInputPlane->height,
            ratioInvX, ratioInvY, offsetX, offsetY);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (resizePlaneFsr(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    {
        const char *kernel_name = "kernel_rcas";
        RGYWorkSize local(RESIZE_BLOCK_X, RESIZE_BLOCK_Y);
        RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
        const std::vector<RGYOpenCLEvent> waitRcas{ eventEasu };
        auto err = m_resize.get()->kernel(kernel_name).config(queue, local, global, waitRcas, event).launch(
            (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
            midMem, midPitchBytes,
            con0_sharp);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (resizePlaneFsr(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterResize::resizePlaneNis(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
                                         cl_mem cfgMem, bool applyUsm, bool useSlm,
                                         RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (!cfgMem || !m_nisCoefScale || !m_nisCoefUsm) {
        AddMessage(RGY_LOG_ERROR, _T("NIS buffers not initialised.\n"));
        return RGY_ERR_NULL_PTR;
    }
    RGYWorkSize local(NIS_BLOCK_WIDTH, NIS_BLOCK_HEIGHT);
    RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
    // 4 kernel variants: { SLM, non-SLM } x { with USM, no USM }.
    const char *kernel_name =
        useSlm ? (applyUsm ? "kernel_nis_scaler_slm"  : "kernel_nis_scaler_slm_no_usm")
               : (applyUsm ? "kernel_nis_scaler"      : "kernel_nis_scaler_no_usm");
    RGY_ERR err = RGY_ERR_NONE;
    if (applyUsm) {
        err = m_resize.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
            (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
            (cl_mem)pInputPlane->ptr[0],  pInputPlane->pitch[0],  pInputPlane->width,  pInputPlane->height,
            cfgMem, m_nisCoefScale->mem(), m_nisCoefUsm->mem());
    } else {
        // K4-only variants drop the coefUsm arg.
        err = m_resize.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
            (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
            (cl_mem)pInputPlane->ptr[0],  pInputPlane->pitch[0],  pInputPlane->width,  pInputPlane->height,
            cfgMem, m_nisCoefScale->mem());
    }
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (resizePlaneNis(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

// NIS cascade orchestrator. For 1-stage (ratio <=2x or
// auto disabled cascade) this collapses to one resizePlaneNis per
// plane. For N>=2 stages, the orchestrator loops:
//   stage 0:        original input -> intermediates[0]
//   stage k (0<k<N-1): intermediates[k-1] -> intermediates[k]
//   stage N-1:      intermediates[N-2] -> final output
// Each stage uses its own m_nisCascadeCfgs[k] (different kScale; for
// intermediates, USM strength is zeroed). Plane loop happens inside
// each stage so the staging buffer can be re-used.
RGY_ERR RGYFilterResize::resizeFrameNisCascade(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
                                                RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const int N = (int)m_nisCascadeCfgs.size();
    if (N <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("NIS cascade not initialised (stage count=0).\n"));
        return RGY_ERR_NULL_PTR;
    }
    auto pResizeParam = std::dynamic_pointer_cast<RGYFilterParamResize>(m_param);
    const int optMode = pResizeParam ? pResizeParam->nis.opt : RGY_NIS_OPT_DEFAULT;
    // opt=skipusm (perf 1): on intermediate cascade stages, dispatch the
    // K4-only kernel so the (zero-weighted) USM polyphase is not computed.
    // opt=fast composes the two A/B winners: skipusm (skip K5 on cascade
    // intermediates) + slm (cooperative SLM tile load). See
    const bool optSkipUsm = (optMode == RGY_NIS_OPT_FAST);
    const bool optSlm     = (optMode == RGY_NIS_OPT_FAST);
    const int numPlanes = RGY_CSP_PLANES[pOutputFrame->csp];
    const RGYFrameInfo *srcFrame = pInputFrame;
    for (int stage = 0; stage < N; stage++) {
        const bool finalStage = (stage == N - 1);
        // USM applies on the final stage always. On intermediates, the
        // default opt still runs the full kernel (zero-weighted USM,
        // wasted work) so the A/B baseline is the conservative path.
        // opt=skipusm switches intermediates to the no-USM kernel.
        const bool applyUsm = finalStage || !optSkipUsm;
        RGYFrameInfo *dstFrame = finalStage ? pOutputFrame : &m_nisCascadeIntermediates[stage]->frame;
        cl_mem cfgMem = m_nisCascadeCfgs[stage]->mem();
        for (int p = 0; p < numPlanes; p++) {
            auto planeDst = getPlane(dstFrame, (RGY_PLANE)p);
            auto planeSrc = getPlane(srcFrame, (RGY_PLANE)p);
            const bool firstPlaneFirstStage = (stage == 0 && p == 0);
            const bool lastPlaneLastStage   = (finalStage && p == numPlanes - 1);
            const std::vector<RGYOpenCLEvent> &plane_wait = firstPlaneFirstStage ? wait_events : std::vector<RGYOpenCLEvent>();
            RGYOpenCLEvent *plane_event = lastPlaneLastStage ? event : nullptr;
            auto err = resizePlaneNis(&planeDst, &planeSrc, cfgMem, applyUsm, optSlm, queue, plane_wait, plane_event);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error in NIS cascade stage %d plane %d: %s.\n"), stage, p, get_err_mes(err));
                return err;
            }
        }
        srcFrame = dstFrame;  // chain into next stage
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterResize::resizePlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    return resizePlane(pOutputPlane, pInputPlane, 0, queue, wait_events, event);
}

RGY_ERR RGYFilterResize::resizePlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const int plane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto pResizeParam = std::dynamic_pointer_cast<RGYFilterParamResize>(m_param);
    if (!pResizeParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (pResizeParam->interp == RGY_VPP_RESIZE_GAUSS) {
        return resizePlaneGauss2Pass(pOutputPlane, pInputPlane, plane, queue, wait_events, event);
    }

    const float ratioX = (float)(pOutputPlane->width) / pInputPlane->width;
    const float ratioY = (float)(pOutputPlane->height) / pInputPlane->height;

    {
        const char *kernel_name = nullptr;
        RGY_ERR err = RGY_ERR_NONE;
        RGYWorkSize local(RESIZE_BLOCK_X, RESIZE_BLOCK_Y);
        RGYWorkSize global(pOutputPlane->width, pOutputPlane->height);
        if (useTextureBilinear(pResizeParam.get())) {
            kernel_name = "kernel_resize_texture_bilinear";
            err = m_resize.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
                (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
                (cl_mem)pInputPlane->ptr[0],
                1.0f / ratioX, 1.0f / ratioY
            );
        } else {
            kernel_name = "kernel_resize";
            err = m_resize.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
                (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
                (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0], pInputPlane->width, pInputPlane->height,
                ratioX, ratioY,
                (m_weightSpline) ? (cl_mem)m_weightSpline->mem() : nullptr);
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (resizePlane(%s)): %s.\n"),
                char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterResize::resizePlaneGauss2Pass(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const int plane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (plane < 0 || plane >= (int)m_gauss2pass.size()) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid resize plane index: %d.\n"), plane);
        return RGY_ERR_INVALID_PARAM;
    }

    auto& gauss2pass = m_gauss2pass[plane];
    auto err = createGaussTmp(gauss2pass, *pOutputPlane, *pInputPlane);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    auto pTmpPlane = &gauss2pass.tmp->frame;

    const float ratioX = (float)(pOutputPlane->width) / pInputPlane->width;
    const float ratioY = (float)(pOutputPlane->height) / pInputPlane->height;

    RGYOpenCLEvent eventH;
    err = m_resize.get()->kernel("kernel_resize_gauss_h").config(queue,
        RGYWorkSize(RESIZE_BLOCK_X, RESIZE_BLOCK_Y),
        RGYWorkSize(pTmpPlane->width, pTmpPlane->height),
        wait_events, &eventH).launch(
            (cl_mem)pTmpPlane->ptr[0], pTmpPlane->pitch[0], pTmpPlane->width, pTmpPlane->height,
            (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0], pInputPlane->width, pInputPlane->height,
            ratioX);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (resizePlane(%s)): %s.\n"),
            _T("kernel_resize_gauss_h"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }

    const std::vector<RGYOpenCLEvent> waitV{ eventH };
    err = m_resize.get()->kernel("kernel_resize_gauss_v").config(queue,
        RGYWorkSize(RESIZE_BLOCK_X, RESIZE_BLOCK_Y),
        RGYWorkSize(pOutputPlane->width, pOutputPlane->height),
        waitV, event).launch(
            (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
            (cl_mem)pTmpPlane->ptr[0], pTmpPlane->pitch[0], pTmpPlane->width, pTmpPlane->height,
            ratioY);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (resizePlane(%s)): %s.\n"),
            _T("kernel_resize_gauss_v"), RGY_CSP_NAMES[pInputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterResize::resizeFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto pResizeParam = std::dynamic_pointer_cast<RGYFilterParamResize>(m_param);
    if (!pResizeParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const RGYFrameInfo *pInputPtr = pInputFrame;
    std::unique_ptr<RGYCLFrame, RGYCLImageFromBufferDeleter> srcImage;
    if (useTextureBilinear(pResizeParam.get())) {
        srcImage = m_cl->createImageFromFrameBuffer(*pInputFrame, true, CL_MEM_READ_ONLY, &m_srcImagePool);
        if (!srcImage) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to create image for input frame.\n"));
            return RGY_ERR_MEM_OBJECT_ALLOCATION_FAILURE;
        }
        pInputPtr = &srcImage->frame;
    }
    const bool useFsrInt = (pResizeParam->interp == RGY_VPP_RESIZE_FSR1) && !m_fp16Easu && m_easuOutput;
    const bool useFsrF16 = (pResizeParam->interp == RGY_VPP_RESIZE_FSR1) && m_fp16Easu && m_easuOutputF16[0];
    const bool useFsr = useFsrInt || useFsrF16;
    const bool useNis = (pResizeParam->interp == RGY_VPP_RESIZE_NIS) && !m_nisCascadeCfgs.empty() && m_nisCoefScale && m_nisCoefUsm;
    if (useNis) {
        // Cascade orchestrator handles both single-stage (N=1) and
        // multi-stage (N>=2) paths; plane loop happens inside.
        return resizeFrameNisCascade(pOutputFrame, pInputPtr, queue, wait_events, event);
    }
    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputPtr,    (RGY_PLANE)i);
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[pOutputFrame->csp] - 1) ? event : nullptr;
        RGY_ERR err = RGY_ERR_NONE;
        if (useFsr) {
            cl_mem midMem;
            int midPitchBytes;
            int midWidth;
            int midHeight;
            if (useFsrF16) {
                midMem = m_easuOutputF16[i]->mem();
                midWidth = m_easuOutputF16Width[i];
                midHeight = m_easuOutputF16Height[i];
                midPitchBytes = midWidth * (int)sizeof(uint16_t);
            } else {
                auto planeMid = getPlane(&m_easuOutput->frame, (RGY_PLANE)i);
                midMem = (cl_mem)planeMid.ptr[0];
                midPitchBytes = planeMid.pitch[0];
                midWidth = planeMid.width;
                midHeight = planeMid.height;
            }
            err = resizePlaneFsr(&planeDst, &planeSrc, midMem, midPitchBytes, midWidth, midHeight, queue, plane_wait_event, plane_event);
        } else {
            err = resizePlane(&planeDst, &planeSrc, i, queue, plane_wait_event, plane_event);
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to resize frame(%d) %s: %s\n"), i, cl_errmes(err));
            return err_cl_to_rgy(err);
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterResize::RGYResizeGaussPlane::RGYResizeGaussPlane() :
    tmp() {
}

RGYFilterResize::RGYFilterResize(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_bInterlacedWarn(false), m_weightSpline(), m_gauss2pass(), m_libplaceboResample(), m_easuOutput(), m_easuOutputF16(), m_easuOutputF16Width{}, m_easuOutputF16Height{}, m_fp16Easu(false), m_nisConfigBuf(), m_nisCoefScale(), m_nisCoefUsm(), m_resize(), m_srcImagePool() {
    m_name = _T("resize");
}

RGYFilterResize::~RGYFilterResize() {
    close();
}

RGY_ERR RGYFilterResize::createGaussTmp(RGYResizeGaussPlane& planeTmp, const RGYFrameInfo& planeOut, const RGYFrameInfo& planeIn) {
    RGYFrameInfo tmpInfo(planeOut.width, planeIn.height, RGY_CSP_Y_F32, 32);
    if (planeTmp.tmp
        && planeTmp.tmp->frameInfo().width == tmpInfo.width
        && planeTmp.tmp->frameInfo().height == tmpInfo.height
        && planeTmp.tmp->frameInfo().csp == tmpInfo.csp
        && planeTmp.tmp->frameInfo().bitdepth == tmpInfo.bitdepth) {
        return RGY_ERR_NONE;
    }
    planeTmp.tmp = m_cl->createFrameBuffer(tmpInfo);
    if (!planeTmp.tmp) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate gauss resize tmp frame.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    return RGY_ERR_NONE;
}

void RGYFilterResize::clearGaussTmp() {
    for (auto& tmp : m_gauss2pass) {
        tmp = RGYResizeGaussPlane();
    }
}

RGY_ERR RGYFilterResize::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto pResizeParam = std::dynamic_pointer_cast<RGYFilterParamResize>(pParam);
    if (!pResizeParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (pResizeParam->frameOut.height <= 0 || pResizeParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (isLibplaceboResizeFiter(pResizeParam->interp)) {
        if (!m_libplaceboResample) {
            m_libplaceboResample = std::make_unique<RGYFilterLibplaceboResample>(m_cl);
        }
        pResizeParam->libplaceboResample->frameIn = pResizeParam->frameIn;
        pResizeParam->libplaceboResample->frameOut = pResizeParam->frameOut;
        sts = m_libplaceboResample->init(pResizeParam->libplaceboResample, m_pLog);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to init libplacebo resample filter: %s.\n"), get_err_mes(sts));
            return sts;
        }
        clearGaussTmp();
    } else {
        m_libplaceboResample.reset(); // 不要になったら解放
        pResizeParam->libplaceboResample.reset();

        auto err = AllocFrameBuf(pResizeParam->frameOut, 1);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
            return RGY_ERR_MEMORY_ALLOC;
        }
        for (int i = 0; i < 4; i++) {
            pResizeParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
        }
        // Allocate / drop the EASU intermediate buffer based on algo.
        // The buffer is output-sized (EASU writes the upscaled result, RCAS
        // reads it at the same resolution). For HBD sources we prefer
        // FP16 storage (eliminates the EASU's quantise-to-integer /
        // RCAS dequantise round-trip; same per-pixel cost as the
        // existing ushort path); for 8-bit sources we keep the uchar
        // integer intermediate (FP16 would double storage and
        // bandwidth without a precision win that matters at 8-bit).
        const int bitDepthFsr = RGY_CSP_BIT_DEPTH[pResizeParam->frameOut.csp];
        m_fp16Easu = false;
        if (pResizeParam->interp == RGY_VPP_RESIZE_FSR1) {
            if (bitDepthFsr > 8
                && RGYOpenCLDevice(m_cl->queue().devid()).checkExtension("cl_khr_fp16")) {
                m_fp16Easu = true;
            }
            if (m_fp16Easu) {
                m_easuOutput.reset();
                // Per-plane FP16 buffers; pitch is implicit (width *
                // sizeof(cl_half) bytes, no padding). Plane dimensions
                // follow the output csp's per-plane subsampling.
                RGYFrameInfo outRefForPlane = pResizeParam->frameOut;
                for (int i = 0; i < RGY_CSP_PLANES[pResizeParam->frameOut.csp]; i++) {
                    const auto planeInfo = getPlane(&outRefForPlane, (RGY_PLANE)i);
                    const int pw = planeInfo.width;
                    const int ph = planeInfo.height;
                    if (!m_easuOutputF16[i]
                        || m_easuOutputF16Width[i]  != pw
                        || m_easuOutputF16Height[i] != ph) {
                        const size_t bytes = (size_t)pw * (size_t)ph * sizeof(uint16_t);
                        m_easuOutputF16[i] = m_cl->createBuffer(bytes, CL_MEM_READ_WRITE);
                        if (!m_easuOutputF16[i] || !m_easuOutputF16[i]->mem()) {
                            AddMessage(RGY_LOG_ERROR, _T("failed to allocate FSR intermediate FP16 buffer (plane %d).\n"), i);
                            return RGY_ERR_MEMORY_ALLOC;
                        }
                        m_easuOutputF16Width[i]  = pw;
                        m_easuOutputF16Height[i] = ph;
                    }
                }
                AddMessage(RGY_LOG_DEBUG, _T("FSR EASU intermediate: FP16 enabled (cl_khr_fp16, bit_depth=%d).\n"), bitDepthFsr);
            } else {
                for (auto &b : m_easuOutputF16) b.reset();
                if (!m_easuOutput || cmpFrameInfoCspResolution(&m_easuOutput->frame, &pResizeParam->frameOut)) {
                    m_easuOutput = m_cl->createFrameBuffer(pResizeParam->frameOut, CL_MEM_READ_WRITE);
                    if (!m_easuOutput) {
                        AddMessage(RGY_LOG_ERROR, _T("failed to allocate FSR intermediate buffer.\n"));
                        return RGY_ERR_MEMORY_ALLOC;
                    }
                }
                AddMessage(RGY_LOG_DEBUG, _T("FSR EASU intermediate: integer pixels (bit_depth=%d).\n"), bitDepthFsr);
            }
        } else {
            m_easuOutput.reset();
            for (auto &b : m_easuOutputF16) b.reset();
        }

        // NIS resource setup. Coefficient LUTs are
        // small (2 KB each) and constant; upload once and re-use.
        //
        // Cascade: for ratios outside NIS's 1x..2x range
        // (kScale < 0.5), split the work across N stages of <=2x each.
        // cascade=auto picks N from log2(ratio); cascade=on forces
        // N>=2 (test path); cascade=off rejects ratios > 2x.
        if (pResizeParam->interp == RGY_VPP_RESIZE_NIS) {
            if (!m_nisCoefScale) {
                m_nisCoefScale = m_cl->copyDataToBuffer(
                    nis::coef_scale, sizeof(nis::coef_scale), CL_MEM_READ_ONLY);
                if (!m_nisCoefScale) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to upload NIS coef_scale LUT.\n"));
                    return RGY_ERR_NULL_PTR;
                }
            }
            if (!m_nisCoefUsm) {
                m_nisCoefUsm = m_cl->copyDataToBuffer(
                    nis::coef_usm, sizeof(nis::coef_usm), CL_MEM_READ_ONLY);
                if (!m_nisCoefUsm) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to upload NIS coef_usm LUT.\n"));
                    return RGY_ERR_NULL_PTR;
                }
            }

            // Compute cascade stage count from the requested ratio.
            const float inW  = (float)pResizeParam->frameIn.width;
            const float inH  = (float)pResizeParam->frameIn.height;
            const float outW = (float)pResizeParam->frameOut.width;
            const float outH = (float)pResizeParam->frameOut.height;
            const float ratioX = outW / inW;
            const float ratioY = outH / inH;
            const float maxRatio = std::max(ratioX, ratioY);

            int stages = 1;
            if (maxRatio > 2.0f) {
                stages = (int)std::ceil(std::log2(maxRatio) - 1e-4f);
                if (stages < 2) stages = 2;
            }
            if (pResizeParam->nis.cascade == RGY_NIS_CASCADE_OFF && stages > 1) {
                AddMessage(RGY_LOG_ERROR, _T("NIS cascade=off cannot handle %.2fx > 2x. Use cascade=auto or pick a smaller output size.\n"), maxRatio);
                return RGY_ERR_UNSUPPORTED;
            }
            if (pResizeParam->nis.cascade == RGY_NIS_CASCADE_ON && stages < 2 && maxRatio > 1.0f) {
                stages = 2;  // unusual test path: force 2-stage even at <=2x
            }

            const int hdrMode = nisResolveHdrMode(pResizeParam.get());
            // Per-stage scale factor (geometric distribution). For N=1
            // this is just the full ratio; for N>=2 each stage scales
            // by maxRatio^(1/N), so the cumulative product hits the
            // requested output dims at the final stage.
            const float perStageRatio = std::pow(maxRatio, 1.0f / (float)stages);

            // (Re-)allocate cfg buffers and intermediate frames.
            m_nisCascadeCfgs.clear();
            m_nisCascadeIntermediates.clear();
            m_nisCascadeCfgs.resize(stages);
            if (stages > 1) m_nisCascadeIntermediates.resize(stages - 1);

            int curW = pResizeParam->frameIn.width;
            int curH = pResizeParam->frameIn.height;
            for (int k = 0; k < stages; k++) {
                const bool finalStage = (k == stages - 1);
                int nextW, nextH;
                if (finalStage) {
                    nextW = pResizeParam->frameOut.width;
                    nextH = pResizeParam->frameOut.height;
                } else {
                    nextW = (int)std::round((float)pResizeParam->frameIn.width  * std::pow(perStageRatio, (float)(k + 1)));
                    nextH = (int)std::round((float)pResizeParam->frameIn.height * std::pow(perStageRatio, (float)(k + 1)));
                }
                NISConfigHost cfg{};
                const bool ok = nisBuildConfig(cfg,
                    pResizeParam->nis.sharpness, hdrMode,
                    0, 0, (uint32_t)curW, (uint32_t)curH,
                    0, 0, (uint32_t)nextW, (uint32_t)nextH);
                if (!ok) {
                    AddMessage(RGY_LOG_DEBUG, _T("NIS stage %d: scale outside 1x-2x (cfg accepted, kernel uses kScale as-is).\n"), k);
                }
                if (!finalStage) {
                    // Intermediate stages: zero USM so only the final
                    // pass applies sharpening (avoids chained
                    // over-sharpen).
                    cfg.kSharpStrengthMin   = 0.0f;
                    cfg.kSharpStrengthScale = 0.0f;
                    cfg.kSharpLimitMin      = 0.0f;
                    cfg.kSharpLimitScale    = 0.0f;
                }
                m_nisCascadeCfgs[k] = m_cl->copyDataToBuffer(&cfg, sizeof(cfg), CL_MEM_READ_ONLY);
                if (!m_nisCascadeCfgs[k]) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to upload NIS stage %d config.\n"), k);
                    return RGY_ERR_NULL_PTR;
                }
                // Allocate intermediate frame buffer for non-final
                // stages. Same csp + bitdepth as the user's output.
                if (!finalStage) {
                    RGYFrameInfo midInfo = pResizeParam->frameOut;
                    midInfo.width  = nextW;
                    midInfo.height = nextH;
                    m_nisCascadeIntermediates[k] = m_cl->createFrameBuffer(midInfo, CL_MEM_READ_WRITE);
                    if (!m_nisCascadeIntermediates[k]) {
                        AddMessage(RGY_LOG_ERROR, _T("failed to allocate NIS stage %d intermediate frame %dx%d.\n"),
                            k, nextW, nextH);
                        return RGY_ERR_MEMORY_ALLOC;
                    }
                }
                AddMessage(RGY_LOG_DEBUG, _T("NIS stage %d/%d: %dx%d -> %dx%d, kScale=%.4fx%.4f, USM=%s.\n"),
                    k + 1, stages, curW, curH, nextW, nextH, cfg.kScaleX, cfg.kScaleY,
                    finalStage ? _T("on") : _T("off"));
                curW = nextW;
                curH = nextH;
            }
            // Keep m_nisConfigBuf pointing at the final stage's config
            // for any single-config diagnostic paths.
            m_nisConfigBuf = nullptr;  // explicit: cascade owns the cfgs now
            AddMessage(RGY_LOG_DEBUG, _T("NIS init: %d stage(s), sharpness=%.2f, hdrMode=%d, cascade=%s.\n"),
                stages, pResizeParam->nis.sharpness, hdrMode,
                pResizeParam->nis.cascade == RGY_NIS_CASCADE_AUTO ? _T("auto") :
                pResizeParam->nis.cascade == RGY_NIS_CASCADE_ON   ? _T("on")   : _T("off"));
        } else {
            m_nisConfigBuf.reset();
            m_nisCoefScale.reset();
            m_nisCoefUsm.reset();
            m_nisCascadeCfgs.clear();
            m_nisCascadeIntermediates.clear();
        }
        auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamResize>(m_param);
        if (!m_resize.get()
            || !prmPrev
            || prmPrev->frameIn.csp != pResizeParam->frameIn.csp
            || prmPrev->frameOut.csp != pResizeParam->frameOut.csp
            || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
            || prmPrev->interp != pResizeParam->interp
            || prmPrev->frameIn.width != pResizeParam->frameIn.width
            || prmPrev->frameIn.height != pResizeParam->frameIn.height
            || prmPrev->frameOut.width != pResizeParam->frameOut.width
            || prmPrev->frameOut.height != pResizeParam->frameOut.height
            || (pResizeParam->interp == RGY_VPP_RESIZE_GAUSS && prmPrev->gaussP != pResizeParam->gaussP)
            || (pResizeParam->interp == RGY_VPP_RESIZE_NIS && prmPrev->nis != pResizeParam->nis)) {
            const int radius = get_radius(pResizeParam->interp);
            const auto algo = get_weight_type(pResizeParam->interp);

            int shared_weightXdim = 0;
            int shared_weightYdim = 0;
            for (int i = 0; i < RGY_CSP_PLANES[pResizeParam->frameOut.csp]; i++) {
                const auto planeOut = getPlane(&pResizeParam->frameOut, (RGY_PLANE)i);
                const auto planeIn = getPlane(&pResizeParam->frameIn, (RGY_PLANE)i);
                const float srcWindowX = getSrcWindow(radius, planeOut.width, planeIn.width);
                shared_weightXdim = std::max(shared_weightXdim, (((int)ceil(srcWindowX) + 1) * 2));
                const float srcWindowY = getSrcWindow(radius, planeOut.height, planeIn.height);
                shared_weightYdim = std::max(shared_weightYdim, (((int)ceil(srcWindowY) + 1) * 2));
            }

            const int use_local = (ENCODER_MPP) ? 0 : 1;

            const int nisEnabled = (pResizeParam->interp == RGY_VPP_RESIZE_NIS) ? 1 : 0;
            const int nisHdrMode = (pResizeParam->interp == RGY_VPP_RESIZE_NIS) ? nisResolveHdrMode(pResizeParam.get()) : 0;
            const auto options = strsprintf("-D Type=%s -D bit_depth=%d -D radius=%d -D algo=%d"
                " -D block_x=%d -D block_y=%d -D shared_weightXdim=%d -D shared_weightYdim=%d"
                " -D WEIGHT_BILINEAR=%d -D WEIGHT_BICUBIC=%d -D WEIGHT_SPLINE=%d -D WEIGHT_LANCZOS=%d -D WEIGHT_GAUSS=%d"
                " -D gauss_p=%.9ff -D USE_LOCAL=%d -D FSR1_FP16_SCRATCH=%d"
                "%s -D NIS_BLOCK_WIDTH=%d -D NIS_BLOCK_HEIGHT=%d -D NIS_HDR_MODE=%d",
                RGY_CSP_BIT_DEPTH[pResizeParam->frameOut.csp] > 8 ? "ushort" : "uchar",
                RGY_CSP_BIT_DEPTH[pResizeParam->frameOut.csp],
                radius, algo,
                RESIZE_BLOCK_X, RESIZE_BLOCK_Y, shared_weightXdim, shared_weightYdim,
                WEIGHT_BILINEAR, WEIGHT_BICUBIC, WEIGHT_SPLINE, WEIGHT_LANCZOS, WEIGHT_GAUSS,
                pResizeParam->gaussP, use_local, m_fp16Easu ? 1 : 0,
                nisEnabled ? " -D NIS_KERNEL_ENABLED=1" : "",
                NIS_BLOCK_WIDTH, NIS_BLOCK_HEIGHT, nisHdrMode);
            m_resize.set(m_cl->buildResourceAsync(_T("RGY_FILTER_RESIZE_CL"), _T("EXE_DATA"), options.c_str()));
            if (algo != WEIGHT_SPLINE) {
                m_weightSpline.reset();
            }
            if ((!m_weightSpline || !prmPrev || prmPrev->interp != pResizeParam->interp)
                && algo == WEIGHT_SPLINE) {
                static const auto SPLINE16_WEIGHT = std::vector<float>{
                    1.0f,       -9.0f/5.0f,  -1.0f/5.0f, 1.0f,
                    -1.0f/3.0f,  9.0f/5.0f, -46.0f/15.0f, 8.0f/5.0f
                };
                static const auto SPLINE36_WEIGHT = std::vector<float>{
                    13.0f/11.0f, -453.0f/209.0f,    -3.0f/209.0f,  1.0f,
                    -6.0f/11.0f,  612.0f/209.0f, -1038.0f/209.0f,  540.0f/209.0f,
                    1.0f/11.0f, -159.0f/209.0f,   434.0f/209.0f, -384.0f/209.0f
                };
                static const auto SPLINE64_WEIGHT = std::vector<float>{
                    49.0f/41.0f, -6387.0f/2911.0f,     -3.0f/2911.0f,  1.0f,
                    -24.0f/41.0f,  9144.0f/2911.0f, -15504.0f/2911.0f,  8064.0f/2911.0f,
                    6.0f/41.0f, -3564.0f/2911.0f,   9726.0f/2911.0f, -8604.0f/2911.0f,
                    -1.0f/41.0f,   807.0f/2911.0f,  -3022.0f/2911.0f,  3720.0f/2911.0f
                };
                const std::vector<float> *weight = nullptr;
                switch (pResizeParam->interp) {
                case RGY_VPP_RESIZE_SPLINE16: weight = &SPLINE16_WEIGHT; break;
                case RGY_VPP_RESIZE_SPLINE36: weight = &SPLINE36_WEIGHT; break;
                case RGY_VPP_RESIZE_SPLINE64: weight = &SPLINE64_WEIGHT; break;
                default: {
                    AddMessage(RGY_LOG_ERROR, _T("unknown interpolation type: %d.\n"), pResizeParam->interp);
                    return RGY_ERR_INVALID_PARAM;
                }
                }

                m_weightSpline = m_cl->copyDataToBuffer(weight->data(), sizeof((*weight)[0]) * weight->size(), CL_MEM_READ_ONLY);
                if (!m_weightSpline) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to send weight to gpu memory.\n"));
                    return RGY_ERR_NULL_PTR;
                }
            }
        }
    }

    if (pResizeParam->interp == RGY_VPP_RESIZE_GAUSS) {
        const int planeCount = RGY_CSP_PLANES[pResizeParam->frameOut.csp];
        for (int i = 0; i < planeCount; i++) {
            const auto planeOut = getPlane(&pResizeParam->frameOut, (RGY_PLANE)i);
            const auto planeIn = getPlane(&pResizeParam->frameIn, (RGY_PLANE)i);
            sts = createGaussTmp(m_gauss2pass[i], planeOut, planeIn);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        for (int i = planeCount; i < RGY_MAX_PLANES; i++) {
            m_gauss2pass[i].tmp.reset();
        }
    } else {
        clearGaussTmp();
    }

    auto str = strsprintf(_T("resize(%s): %dx%d -> %dx%d"),
        get_chr_from_value(list_vpp_resize, pResizeParam->interp),
        pResizeParam->frameIn.width, pResizeParam->frameIn.height,
        pResizeParam->frameOut.width, pResizeParam->frameOut.height);
    if (m_libplaceboResample) {
        str += _T("\n                 ");
        str += pResizeParam->libplaceboResample->print();
    }
    if (pResizeParam->interp == RGY_VPP_RESIZE_FSR1) {
        str += _T(", ") + pResizeParam->fsr1.print();
    }
    if (pResizeParam->interp == RGY_VPP_RESIZE_NIS) {
        str += _T(", ") + pResizeParam->nis.print();
        if (m_nisCascadeCfgs.size() > 1) {
            str += strsprintf(_T(", stages=%d"), (int)m_nisCascadeCfgs.size());
        }
    }
    setFilterInfo(str);

    //コピーを保存
    m_param = pResizeParam;
    return sts;
}

RGY_ERR RGYFilterResize::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }

    if (m_libplaceboResample) {
        RGYFrameInfo inputFrame = *pInputFrame;
        auto sts_filter = m_libplaceboResample->filter(&inputFrame, ppOutputFrames, pOutputFrameNum, queue, wait_events, event);
        if (ppOutputFrames[0] == nullptr || *pOutputFrameNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_libplaceboResample->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || *pOutputFrameNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_libplaceboResample->name().c_str());
            return sts_filter;
        }
        return RGY_ERR_NONE;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;

    if (!m_resize.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_RESIZE_CL(resize)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }

    //if (interlaced(*pInputFrame)) {
    //    return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
    //}
    const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != RGYCLMemcpyD2D) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    static const auto supportedCspYV12   = make_array<RGY_CSP>(RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
    static const auto supportedCspYUV444 = make_array<RGY_CSP>(RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);

    auto pResizeParam = std::dynamic_pointer_cast<RGYFilterParamResize>(m_param);
    if (!pResizeParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    sts = resizeFrame(ppOutputFrames[0], pInputFrame, queue, wait_events, event);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at resizeFrame (%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }

    return sts;
}

void RGYFilterResize::close() {
    m_srcImagePool.clear();
    m_frameBuf.clear();
    m_resize.clear();
    m_weightSpline.reset();
    clearGaussTmp();
    m_easuOutput.reset();
    for (auto &b : m_easuOutputF16) b.reset();
    m_easuOutputF16Width.fill(0);
    m_easuOutputF16Height.fill(0);
    m_fp16Easu = false;
    m_nisConfigBuf.reset();
    m_nisCoefScale.reset();
    m_nisCoefUsm.reset();
    m_nisCascadeCfgs.clear();
    m_nisCascadeIntermediates.clear();
    m_cl.reset();
    m_bInterlacedWarn = false;
}
