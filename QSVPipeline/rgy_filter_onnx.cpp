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

#include "rgy_filter_onnx.h"
#include "rgy_filter_resize.h" // opt-in end-of-chain resize (out_res=/resize=)
#include "rgy_aspect_ratio.h"  // set_auto_resolution() for out_res= negative auto-aspect
#include "rgy_filesystem.h"
#include "rgy_model_registry.h"
#include "rgy_avutil.h"   // mask= reads the user mask image via avformat/avcodec
#include "rgy_version.h"
#include <cmath>
#include <cstring>
#include <algorithm>

tstring RGYFilterParamOnnx::print() const {
    return onnx.print();
}

RGYFilterOnnx::RGYFilterOnnx(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context), m_ov(), m_io(OnnxIO::LumaSR), m_inC(1), m_outC(1),
    m_scale(1), m_modelInW(0), m_modelInH(0), m_padL(0), m_padT(0),
    m_maxval(255.0f), m_bitdepth(8), m_useOcl(false), m_ycbcr(false), m_sigmaNorm(0.0f),
    m_yOff(0.0f), m_yScale(1.0f), m_yRange(255.0f), m_cOff(128.0f), m_cScale(1.0f), m_cRange(255.0f),
    m_matVR(0), m_matUG(0), m_matVG(0), m_matUB(0),
    m_matRY(0), m_matGY(0), m_matBY(0), m_matRU(0), m_matGU(0), m_matBU(0), m_matRV(0), m_matGV(0), m_matBV(0),
    m_inStaging(), m_outStaging(), m_inBuf(), m_outBuf(), m_u444(), m_v444(),
    m_program(), m_inBufCL(), m_outBufCL(), m_inRgbPlanes(), m_outRgbPlanes(), m_inYuvPlanes(), m_outYuvPlanes(), m_outChromaPlanes(),
    m_cropToRgb(), m_cropFromRgb(), m_cropToYuv444(), m_cropFromYuv444(),
    m_temporalT(1), m_ringBaseIdx(0), m_recvCount(0), m_emitCount(0),
    m_ovm(), m_maskModelW(0), m_maskModelH(0), m_imgPortIdx(0), m_mskPortIdx(1), m_outScale(0.0f) {
    m_name = _T("onnx");
}

RGYFilterOnnx::~RGYFilterOnnx() {
    close();
}

namespace {
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline int sample_x(const int x, const int padL, const int width) {
    return clampi(x - padL, 0, width - 1);
}
static inline int sample_y(const int y, const int padT, const int height) {
    return clampi(y - padT, 0, height - 1);
}

static const TCHAR *cx_desc_or_unknown(const CX_DESC *list, int value) {
    const auto desc = get_cx_desc(list, value);
    return (desc != nullptr) ? desc : _T("unknown");
}

static bool onnx_matrix_to_coeff_id(CspMatrix matrix, int inputHeight, int& matrixSel) {
    if (matrix == RGY_MATRIX_AUTO || (int)matrix == COLOR_VALUE_AUTO_RESOLUTION) {
        matrixSel = (inputHeight <= 576) ? 601 : 709;
        return true;
    }
    switch (matrix) {
    case RGY_MATRIX_ST170_M:
    case RGY_MATRIX_BT470_BG:
        matrixSel = 601;
        return true;
    case RGY_MATRIX_BT709:
        matrixSel = 709;
        return true;
    case RGY_MATRIX_BT2020_NCL:
        matrixSel = 2020;
        return true;
    default:
        return false;
    }
}

static bool onnx_supported_colorrange(CspColorRange range) {
    return range == RGY_COLORRANGE_AUTO
        || range == RGY_COLORRANGE_LIMITED
        || range == RGY_COLORRANGE_FULL;
}

static CspMatrix onnx_coeff_id_to_matrix(const int matrixSel) {
    switch (matrixSel) {
    case 601:  return RGY_MATRIX_ST170_M;
    case 2020: return RGY_MATRIX_BT2020_NCL;
    case 709:
    default:   return RGY_MATRIX_BT709;
    }
}

// Bilinear upscale of one channel from (sw x sh) to (sw*scale x sh*scale)
// on the CPU (host path). Mirrors the chroma_bilinear kernel above.
// Pitches in bytes, strides in samples (2 for an nv12/p010-interleaved channel).
template<typename TPix>
static void upscale_bilinear(uint8_t *dst, const int dstPitch, const int dstStride,
                             const uint8_t *src, const int srcPitch, const int srcStride,
                             const int sw, const int sh, const int scale, const int pixMax) {
    const int dw = sw * scale;
    const int dh = sh * scale;
    const float inv = 1.0f / (float)scale;
    for (int dy = 0; dy < dh; dy++) {
        float sy = (dy + 0.5f) * inv - 0.5f;
        int y0 = (int)std::floor(sy);
        float fy = sy - (float)y0;
        const TPix *row0 = (const TPix *)(src + (size_t)clampi(y0,     0, sh - 1) * srcPitch);
        const TPix *row1 = (const TPix *)(src + (size_t)clampi(y0 + 1, 0, sh - 1) * srcPitch);
        TPix *drow = (TPix *)(dst + (size_t)dy * dstPitch);
        for (int dx = 0; dx < dw; dx++) {
            float sx = (dx + 0.5f) * inv - 0.5f;
            int x0 = (int)std::floor(sx);
            float fx = sx - (float)x0;
            const int x0c = clampi(x0,     0, sw - 1) * srcStride;
            const int x1c = clampi(x0 + 1, 0, sw - 1) * srcStride;
            const float a = row0[x0c], b = row0[x1c];
            const float c = row1[x0c], d = row1[x1c];
            const float top = a + (b - a) * fx;
            const float bot = c + (d - c) * fx;
            const int v = (int)(top + (bot - top) * fy + 0.5f);
            drow[dx * dstStride] = (TPix)clampi(v, 0, pixMax);
        }
    }
}

// Bilinearly sample one chroma channel (half-res, 4:2:0) at the location
// of luma pixel (lx, ly), upsampling x2. Returns the raw value (0..pixmax) as a
// float. plane/pitch/stride address the channel (pitch in bytes; stride 2 for
// nv12/p010-interleaved, 1 for planar). Matches the (d+0.5)/scale-0.5
// convention of the OCL chroma_bilinear kernel.
template<typename TPix>
static inline float sample_chroma_up2(const uint8_t *plane, const int pitch, const int stride,
                                      const int cw, const int ch, const int lx, const int ly) {
    const float cx = (lx + 0.5f) * 0.5f - 0.5f;
    const float cy = (ly + 0.5f) * 0.5f - 0.5f;
    const int x0 = (int)std::floor(cx); const float fx = cx - (float)x0;
    const int y0 = (int)std::floor(cy); const float fy = cy - (float)y0;
    const int x0c = clampi(x0,     0, cw - 1) * stride;
    const int x1c = clampi(x0 + 1, 0, cw - 1) * stride;
    const TPix *r0 = (const TPix *)(plane + (size_t)clampi(y0,     0, ch - 1) * pitch);
    const TPix *r1 = (const TPix *)(plane + (size_t)clampi(y0 + 1, 0, ch - 1) * pitch);
    const float a = r0[x0c], b = r0[x1c];
    const float c = r1[x0c], d = r1[x1c];
    const float top = a + (b - a) * fx;
    const float bot = c + (d - c) * fx;
    return top + (bot - top) * fy;
}

// 2x2 box-downsample a full-res normalised channel to a half-res chroma
// plane, encoding each averaged value as v*encScale + encOff (rounded, clamped).
// dstPitch in bytes, dstStride in samples.
template<typename TPix>
static void downsample420_encode(uint8_t *dst, const int dstPitch, const int dstStride,
                                 const float *srcFull, const int fullW, const int fullH,
                                 const float encScale, const float encOff, const int pixMax) {
    const int cw = fullW / 2;
    const int ch = fullH / 2;
    for (int cy = 0; cy < ch; cy++) {
        const float *s0 = srcFull + (size_t)(2 * cy)     * fullW;
        const float *s1 = srcFull + (size_t)(2 * cy + 1) * fullW;
        TPix *drow = (TPix *)(dst + (size_t)cy * dstPitch);
        for (int cx = 0; cx < cw; cx++) {
            const int x0 = 2 * cx;
            const float avg = (s0[x0] + s0[x0 + 1] + s1[x0] + s1[x0 + 1]) * 0.25f;
            const int v = (int)(avg * encScale + encOff + 0.5f);
            drow[cx * dstStride] = (TPix)clampi(v, 0, pixMax);
        }
    }
}

// Copy one plane (row-by-row, honouring pitches). width is in samples,
// pitches in bytes, srcStride/dstStride 1 for planar, 2 for nv12/p010-interleaved.
template<typename TPix>
static void copy_plane(uint8_t *dst, const int dstPitch, const int dstStride,
                       const uint8_t *src, const int srcPitch, const int srcStride,
                       const int width, const int height) {
    for (int y = 0; y < height; y++) {
        const TPix *srow = (const TPix *)(src + (size_t)y * srcPitch);
        TPix *drow = (TPix *)(dst + (size_t)y * dstPitch);
        if (srcStride == 1 && dstStride == 1) {
            memcpy(drow, srow, (size_t)width * sizeof(TPix));
        } else {
            for (int x = 0; x < width; x++) drow[x * dstStride] = srow[x * srcStride];
        }
    }
}
} // namespace

RGY_ERR RGYFilterOnnx::checkParam(const std::shared_ptr<RGYFilterParamOnnx> prm) {
    int matrixSel = 0;
    if (!onnx_matrix_to_coeff_id(prm->onnx.colormatrix, prm->frameIn.height, matrixSel)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: unsupported colormatrix %s.\n"),
            cx_desc_or_unknown(list_colormatrix, prm->onnx.colormatrix));
        return RGY_ERR_UNSUPPORTED;
    }
    if (!onnx_matrix_to_coeff_id(prm->onnx.colormatrixOut, prm->frameIn.height, matrixSel)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: unsupported colormatrix_out %s.\n"),
            cx_desc_or_unknown(list_colormatrix, prm->onnx.colormatrixOut));
        return RGY_ERR_UNSUPPORTED;
    }
    if (!onnx_supported_colorrange(prm->onnx.colorrange)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: unsupported colorrange %s.\n"),
            cx_desc_or_unknown(list_colorrange, prm->onnx.colorrange));
        return RGY_ERR_UNSUPPORTED;
    }
    return RGY_ERR_NONE;
}

void RGYFilterOnnx::setupColorCoeffs(int matrixSelIn, int matrixSelOut, bool rangeTV, int pixMax) {
    // Forward YUV->RGB matrix (Kr, Kb per matrix; Kg = 1 - Kr - Kb). Identical
    // to the native anime4k RGB bookend so the OpenVINO path reproduces it.
    float Kr = 0.2126f, Kb = 0.0722f;        // BT.709 default
    if (matrixSelIn == 601)  { Kr = 0.299f;  Kb = 0.114f; }
    if (matrixSelIn == 2020) { Kr = 0.2627f; Kb = 0.0593f; }
    const float Kg = 1.0f - Kr - Kb;
    m_matVR = 2.0f * (1.0f - Kr);
    m_matUG = -2.0f * Kb * (1.0f - Kb) / Kg;
    m_matVG = -2.0f * Kr * (1.0f - Kr) / Kg;
    m_matUB = 2.0f * (1.0f - Kb);
    // Inverse RGB->YUV matrix - its own matrix selection: SDR->HDR系モデルは
    // 入力(bt709)と異なる色空間(bt2020)で出力するため、出力側は別係数を使える。
    // matrixSelOut == matrixSelIn なら従来と完全に同じ係数になる。
    float Kr2 = 0.2126f, Kb2 = 0.0722f;      // BT.709 default
    if (matrixSelOut == 601)  { Kr2 = 0.299f;  Kb2 = 0.114f; }
    if (matrixSelOut == 2020) { Kr2 = 0.2627f; Kb2 = 0.0593f; }
    const float Kg2 = 1.0f - Kr2 - Kb2;
    m_matRY = Kr2;                             m_matGY = Kg2;                             m_matBY = Kb2;
    m_matRU = -Kr2 / (2.0f * (1.0f - Kb2));    m_matGU = -Kg2 / (2.0f * (1.0f - Kb2));    m_matBU = 0.5f;
    m_matRV = 0.5f;                            m_matGV = -Kg2 / (2.0f * (1.0f - Kr2));    m_matBV = -Kb2 / (2.0f * (1.0f - Kr2));
    // Range normalisation (forward and inverse). TV-range offsets scale with
    // bit depth by 2^(n-8) (16/235 -> 4096/60160 at 16-bit), matching the
    // pipeline's left-shift bit depth promotion exactly. 8-bit: scale = 1.
    const float depthScale = (float)(pixMax + 1) / 256.0f;
    m_yOff   = rangeTV ? (16.0f  * depthScale) : 0.0f;
    m_yRange = rangeTV ? (219.0f * depthScale) : (float)pixMax;
    m_yScale = 1.0f / m_yRange;
    m_cOff   = rangeTV ? (128.0f * depthScale) : ((float)pixMax / 2.0f);
    m_cRange = rangeTV ? (224.0f * depthScale) : (float)pixMax;
    m_cScale = 1.0f / m_cRange;
}

// OpenVINOのCACHE_DIRはコンパイル済みバイナリをそのまま保存するだけで、実行環境が変わっても
// 古いキャッシュを消したり検証したりはしない (QSVEncビルド・OpenVINOランタイム・GPUドライバの
// いずれかが変わっても同じファイルを読みに行く)。QSVEncのdevice_info_cache
// (rgy_device_info_cache.cpp) が「期待するバージョン文字列と不一致なら丸ごと無効化する」のと
// 同じ考え方で、cache_dir配下にこれらを埋め込んだサブフォルダを切ることで対応する。
// 一致しなければ単に別のサブフォルダを使うだけなので、壊れたキャッシュを読む心配がない。
static tstring sanitizeForPath(const tstring &str) {
    tstring ret = str;
    for (auto &c : ret) {
        if (!((c >= _T('0') && c <= _T('9')) || (c >= _T('A') && c <= _T('Z')) || (c >= _T('a') && c <= _T('z')) || c == _T('.') || c == _T('-') || c == _T('_'))) {
            c = _T('_');
        }
    }
    return ret;
}

static tstring onnxCacheDirFingerprint(const std::shared_ptr<RGYOpenCLContext> &cl) {
    tstring fingerprint = char_to_tstring(ENCODER_NAME) + _T("_") + VER_STR_FILEVERSION_TCHAR + _T("_rev") + char_to_tstring(ENCODER_REV);
    if (const auto ovVer = RGYOpenVINO::runtimeVersion(); !ovVer.empty()) {
        fingerprint += _T("_ov") + ovVer;
    }
    if (cl) {
        if (const auto driverVer = RGYOpenCLDevice(cl->queue().devid()).info().driver_version; !driverVer.empty()) {
            fingerprint += _T("_drv") + char_to_tstring(driverVer);
        }
    }
    return sanitizeForPath(fingerprint);
}

RGY_ERR RGYFilterOnnx::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamOnnx>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!RGYOpenVINO::available()) {
        const auto status = RGYOpenVINO::availabilityStatus();
        AddMessage(RGY_LOG_ERROR, _T("onnx: OpenVINO runtime is not available: %s.\n"), status.c_str());
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->onnx.modelFile.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: model= (path to an .onnx / .xml model) is required.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->onnx.modelFile.find_first_of(_T("/\\.")) == tstring::npos && !prm->modelDir.empty()) {
        RGYModelRegistry registry;
        auto jsonPath = PathCombineS(prm->modelDir, _T("models.json"));
        auto err = registry.load(jsonPath, m_pLog);
        if (err != RGY_ERR_NONE) return err;
        auto entry = registry.find(prm->onnx.modelFile);
        if (!entry) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: model \"%s\" not found in models.json\n"), prm->onnx.modelFile.c_str());
            return RGY_ERR_NOT_FOUND;
        }
        prm->onnx.modelFile = registry.resolveModelPath(prm->onnx.modelFile);
        if (prm->onnx.colorspace.empty() || prm->onnx.colorspace == _T("auto")) {
            prm->onnx.colorspace = entry->colorspace;
        }
        if (prm->onnx.noise == 15) {
            prm->onnx.noise = entry->noise;
        }
        if (prm->onnx.frames == 1 && entry->frames > 1) {
            prm->onnx.frames = entry->frames;
        }
        if (prm->onnx.precision == _T("auto") && entry->fp32) {
            prm->onnx.precision = _T("fp32");
        }
        if (prm->onnx.colormatrixOut == RGY_MATRIX_AUTO && entry->colormatrixOut != RGY_MATRIX_UNSPECIFIED) {
            prm->onnx.colormatrixOut = entry->colormatrixOut;
        }
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (!rgy_file_exists(prm->onnx.modelFile)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: model file not found: %s\n"), prm->onnx.modelFile.c_str());
        return RGY_ERR_FILE_OPEN;
    }

    const auto inCsp = prm->frameIn.csp;
    if ((inCsp != RGY_CSP_YV12 && inCsp != RGY_CSP_NV12 && inCsp != RGY_CSP_YV12_16 && inCsp != RGY_CSP_P010)
        || (prm->frameIn.bitdepth != 8 && prm->frameIn.bitdepth != 16)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: supports yuv420 8-bit (yv12/nv12) or 16-bit (yv12(16bit)/p010) only; got %s %dbit.\n"),
            RGY_CSP_NAMES[inCsp], prm->frameIn.bitdepth);
        return RGY_ERR_UNSUPPORTED;
    }

    const int inW = prm->frameIn.width;
    const int inH = prm->frameIn.height;

    // Two-input inpainting mode: mask= switches the filter onto the multi-port
    // backend; everything below this point is the single-tensor path.
    if (!prm->onnx.maskFile.empty()) {
        if (prm->onnx.frames > 1) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: mask= and frames= cannot be combined.\n"));
            return RGY_ERR_UNSUPPORTED;
        }
        return initMask(prm, inW, inH, inCsp);
    }

    const tstring dev = prm->onnx.device;
    const bool deviceWantsGpu = (dev.substr(0, 3) == _T("GPU") || dev == _T("AUTO"));
    bool preferRemoteContext = deviceWantsGpu && m_cl;

    m_ov = std::make_unique<RGYOpenVINO>();
    if (!prm->onnx.cacheDir.empty()) {
        //コンパイル済みモデルのキャッシュ: 初回コンパイル後、同一モデル+設定なら次回以降ほぼ即時ロード
        //QSVEncビルド/OpenVINOランタイム/GPUドライバのいずれかが変わったら別サブフォルダになるようにし、
        //古いキャッシュを誤って読むことがないようにする (詳細はonnxCacheDirFingerprint()のコメント参照)
        const tstring effectiveCacheDir = PathCombineS(prm->onnx.cacheDir, onnxCacheDirFingerprint(m_cl));
        if (CreateDirectoryRecursive(effectiveCacheDir.c_str())) {
            m_ov->setCacheDir(effectiveCacheDir);
            AddMessage(RGY_LOG_DEBUG, _T("onnx: OpenVINO CACHE_DIR = %s\n"), effectiveCacheDir.c_str());
        } else {
            AddMessage(RGY_LOG_WARN, _T("onnx: failed to create cache_dir %s, cache disabled.\n"), effectiveCacheDir.c_str());
        }
    }
    tstring errMsg;
    tstring effectiveDevice = prm->onnx.device;
    bool usingOpenCLRemoteContext = false;

    // モデルのチャンネル数から共有経路を自動選択する。
    // 1ch輝度、2ch Gray+Noise、3ch Chroma/RGB、4ch RGB+Noiseモデル以外はホスト経路を使用する。
    int peekIn = 0, peekOut = 0;
    RGY_ERR err = m_ov->peekChannels(prm->onnx.modelFile, peekIn, peekOut, errMsg);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: failed to read model %s: %s\n"),
            prm->onnx.modelFile.c_str(), errMsg.c_str());
        return err;
    }
    bool fastOcl = deviceWantsGpu && m_cl
        && ((peekIn == 1 && peekOut == 1) || (peekIn == 2 && peekOut == 1)
            || (peekIn == 3 && peekOut == 2) || (peekIn == 3 && peekOut == 3)
            || (peekIn == 4 && peekOut == 3));

    auto initModel = [&](const int modelInH, const int modelInW) {
        if (fastOcl) {
            return m_ov->initShared(prm->onnx.modelFile, (void *)m_cl->queue().get(), (void *)m_cl->context(), modelInH, modelInW, errMsg, prm->onnx.precision);
        }
        if (preferRemoteContext) {
            auto remoteErr = m_ov->initFromOpenCLQueue(prm->onnx.modelFile, (void *)m_cl->queue().get(), (void *)m_cl->context(), modelInH, modelInW, errMsg, prm->onnx.precision);
            if (remoteErr == RGY_ERR_NONE) {
                usingOpenCLRemoteContext = true;
                effectiveDevice = _T("GPU(OpenCL context)");
                return remoteErr;
            }
            AddMessage(RGY_LOG_DEBUG, _T("onnx: OpenVINO remote OpenCL context compile is unavailable, trying device match fallback: %s\n"),
                errMsg.c_str());
            preferRemoteContext = false;
            usingOpenCLRemoteContext = false;

            tstring matchErr;
            const auto clInfo = RGYOpenCLDevice(m_cl->queue().devid()).info();
            const auto matchedDevice = m_ov->findDeviceByUuidLuid(clInfo.uuid, sizeof(clInfo.uuid), clInfo.luid, sizeof(clInfo.luid), matchErr);
            if (!matchedDevice.empty()) {
                effectiveDevice = matchedDevice;
                AddMessage(RGY_LOG_DEBUG, _T("onnx: selected OpenVINO device %s by matching OpenCL UUID/LUID.\n"),
                    effectiveDevice.c_str());
            } else {
                AddMessage(RGY_LOG_WARN, _T("onnx: failed to match OpenVINO GPU to selected OpenCL device, falling back to device=%s: %s\n"),
                    prm->onnx.device.c_str(), matchErr.c_str());
                effectiveDevice = prm->onnx.device;
            }
            errMsg.clear();
        }
        return m_ov->init(prm->onnx.modelFile, effectiveDevice, modelInH, modelInW, errMsg, prm->onnx.precision);
    };

    m_modelInW = inW;
    m_modelInH = inH;
    m_padL = 0;
    m_padT = 0;

    if (fastOcl) {
        // share QSVEnc's in-order command queue so OpenVINO inference enqueues
        // between this filter's kernels with no host synchronisation.
        err = initModel(m_modelInH, m_modelInW);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_WARN, _T("onnx: shared OpenCL context is unavailable, falling back to host path: %s\n"),
                errMsg.c_str());
            fastOcl = false;
            errMsg.clear();
            err = initModel(m_modelInH, m_modelInW);
        }
    } else {
        // device-string compile; the host readback path uses this for every
        // multi-channel mode (the model still runs on the GPU).
        err = initModel(m_modelInH, m_modelInW);
    }
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: failed to load/compile model (%s): %s\n"),
            fastOcl ? _T("shared OpenCL context") : effectiveDevice.c_str(),
            errMsg.c_str());
        return err;
    }

    // Infer the I/O convention from the compiled model's channel counts.
    m_inC  = m_ov->inChannels();
    m_outC = m_ov->outChannels();
    // Multi-frame temporal window (frames= > 1): the model takes T frames' worth
    // of RGB stacked on the channel axis (T*3 in) and returns one RGB frame (3 out).
    // Everything downstream reuses the RGB bookend; only the input packing differs,
    // so the single-frame I/O inference below is skipped for this mode.
    m_temporalT = std::max(1, prm->onnx.frames);
    if (m_temporalT > 1) {
        if (m_inC != m_temporalT * 3 || m_outC != 3) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: frames=%d needs a model with %dch in (T*3 RGB) and 3ch out, but this model is %dch in / %dch out.\n"),
                m_temporalT, m_temporalT * 3, m_inC, m_outC);
            return RGY_ERR_UNSUPPORTED;
        }
        m_io = OnnxIO::RGB;
        // 1-in/1-out with a (T-1)/2 delay (and multi-out at flush): the framework's
        // auto path-through would stamp each output with the current input's
        // timestamp/id (wrong during the delay) and with -1 at flush. Clear those
        // bits; run_filter stamps timestamp/duration/picstruct/inputFrameId per output.
        m_pathThrough = (FILTER_PATHTHROUGH_FRAMEINFO)(m_pathThrough &
            (~(uint32_t)(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_FLAGS)));
    }
    else if (m_inC == 1 && m_outC == 1) m_io = OnnxIO::LumaSR;
    else if (m_inC == 2 && m_outC == 1) m_io = OnnxIO::GrayNoise;
    else if (m_inC == 3 && m_outC == 2) m_io = OnnxIO::Chroma;
    else if (m_inC == 3 && m_outC == 3) m_io = OnnxIO::RGB;
    else if (m_inC == 4 && m_outC == 3) m_io = OnnxIO::RGBNoise;
    else {
        AddMessage(RGY_LOG_ERROR, _T("onnx: unsupported model I/O: %dch in / %dch out.\n"), m_inC, m_outC);
        return RGY_ERR_UNSUPPORTED;
    }

    int outW = m_ov->outWidth();
    int outH = m_ov->outHeight();
    auto exactScale = [&](int &scale) {
        if (outW <= 0 || outH <= 0 || (outW % inW) != 0 || (outH % inH) != 0 || (outW / inW) != (outH / inH)) {
            return false;
        }
        scale = outW / inW;
        return scale > 0;
    };
    if (!exactScale(m_scale)) {
        // Some models converted from NCNN pipelines (RealCUGAN dynamic,
        // waifu2x CUNet, etc.) use valid convolutions and intentionally trim
        // border pixels: e.g. 640x480 input may produce 1208x888 (2x minus 72)
        // or 584x424 (1x minus 56). If the trim is symmetric, feed the model
        // an edge-replicated padded tensor instead of rejecting it, so the
        // model's trimmed output lands exactly on the original frame's integer
        // upscale size. This keeps the pipeline output size predictable without
        // cropping away real source pixels.
        const int guessedScale = std::max(1, (int)(std::round((double)outW / (double)inW)));
        const int cropW = guessedScale * inW - outW;
        const int cropH = guessedScale * inH - outH;
        // 共有経路はエッジパディングに対応していないため、既存のホスト経路へ切り替える。
        // m_useOcl はこのブロックより後で設定されるので、ここでは fastOcl を判定する。
        if (fastOcl) {
            AddMessage(RGY_LOG_WARN, _T("onnx: model output %dx%d needs edge padding; falling back to host path.\n"),
                outW, outH);
            fastOcl = false;
        }
        if (guessedScale <= 0 || cropW <= 0 || cropH <= 0 || cropW != cropH || (cropW % guessedScale) != 0) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: model output %dx%d is not an integer upscale of input %dx%d.\n"),
                outW, outH, inW, inH);
            return RGY_ERR_UNSUPPORTED;
        }
        const int padTotal = cropW / guessedScale;
        m_padL = padTotal / 2;
        m_padT = padTotal / 2;
        m_modelInW = inW + padTotal;
        m_modelInH = inH + padTotal;
        errMsg.clear();
        err = initModel(m_modelInH, m_modelInW);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: failed to load/compile padded model (%s, input %dx%d): %s\n"),
                effectiveDevice.c_str(), m_modelInW, m_modelInH, errMsg.c_str());
            return err;
        }
        outW = m_ov->outWidth();
        outH = m_ov->outHeight();
        if (!exactScale(m_scale)) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: model output %dx%d is not an integer upscale of original input %dx%d after input padding %d.\n"),
                outW, outH, inW, inH, padTotal);
            return RGY_ERR_UNSUPPORTED;
        }
    }
    if ((m_io == OnnxIO::GrayNoise || m_io == OnnxIO::Chroma) && m_scale != 1) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: %s model must be scale=1 (got x%d).\n"),
            (m_io == OnnxIO::Chroma) ? _T("chroma") : _T("gray+noise"), m_scale);
        return RGY_ERR_UNSUPPORTED;
    }
    m_maxval = (float)((1 << prm->frameIn.bitdepth) - 1);
    m_bitdepth = prm->frameIn.bitdepth;

    // colourspace handling: Chroma is inherently planar YCbCr; a 3ch RGB model
    // can be told to treat its planes as YCbCr via colorspace=ycbcr.
    m_ycbcr = (m_io == OnnxIO::Chroma) || (m_io == OnnxIO::RGB && prm->onnx.colorspace == _T("ycbcr"));

    // noise sigma for the conditioning channel of the 2ch/4ch noise models.
    const int noiseClamped = std::max(0, std::min(255, prm->onnx.noise));
    m_sigmaNorm = (float)noiseClamped / 255.0f;

    int matrixSel = 0;
    int matrixSelOut = 0;
    onnx_matrix_to_coeff_id(prm->onnx.colormatrix, inH, matrixSel);
    if (prm->onnx.colormatrixOut == RGY_MATRIX_AUTO) {
        matrixSelOut = matrixSel;
    } else {
        onnx_matrix_to_coeff_id(prm->onnx.colormatrixOut, inH, matrixSelOut);
    }
    const bool rangeTV = (prm->onnx.colorrange != RGY_COLORRANGE_FULL);
    setupColorCoeffs(matrixSel, matrixSelOut, rangeTV, (int)m_maxval);

    if (fastOcl && (m_io == OnnxIO::Chroma || m_io == OnnxIO::RGB || m_io == OnnxIO::RGBNoise)) {
        const auto alignBits = std::max(1, RGYOpenCLDevice(m_cl->queue().devid()).info().mem_base_addr_align);
        const size_t alignBytes = std::max<size_t>(1, (alignBits + 7) / 8);
        const size_t inPlaneBytes = (size_t)inW * inH * sizeof(float);
        const size_t outPlaneBytes = (size_t)outW * outH * sizeof(float);
        if ((inPlaneBytes % alignBytes) != 0 || (outPlaneBytes % alignBytes) != 0) {
            AddMessage(RGY_LOG_WARN, _T("onnx: tensor plane offsetがOpenCLのアラインメントを満たさないため、ホスト経路へ切り替えます。\n"));
            fastOcl = false;
        }
    }
    m_useOcl = fastOcl && (m_io == OnnxIO::LumaSR || m_io == OnnxIO::GrayNoise
        || m_io == OnnxIO::Chroma
        || m_io == OnnxIO::RGB || m_io == OnnxIO::RGBNoise);

    // Output frame buffer at the (possibly upscaled) resolution.
    auto frameOut = prm->frameOut;
    frameOut.csp    = inCsp;
    frameOut.width  = outW;
    frameOut.height = outH;
    prm->frameOut   = frameOut;
    err = AllocFrameBuf(prm->frameOut, 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: failed to allocate output frame buffer: %s.\n"), get_err_mes(err));
        return err;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    if (m_useOcl) {
        m_inBufCL  = m_cl->createBuffer((size_t)m_inC  * inW  * inH  * sizeof(float));
        m_outBufCL = m_cl->createBuffer((size_t)m_outC * outW * outH * sizeof(float));
        if (!m_inBufCL || !m_outBufCL) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: failed to allocate network buffers.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        err = m_ov->setSharedIO((void *)m_inBufCL->mem(), (void *)m_outBufCL->mem());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: failed to bind shared GPU tensors.\n"));
            return err;
        }
        if (m_io == OnnxIO::LumaSR || m_io == OnnxIO::GrayNoise) {
            const auto clBuildOptions = strsprintf("-D Type=%s -D bit_depth=%d",
                (prm->frameIn.bitdepth > 8) ? "ushort" : "uchar", prm->frameIn.bitdepth);
            m_program = m_cl->buildResource(_T("RGY_FILTER_ONNX_CL"), _T("EXE_DATA"), clBuildOptions);
            if (!m_program) {
                AddMessage(RGY_LOG_ERROR, _T("onnx: failed to build RGY_FILTER_ONNX_CL.\n"));
                return RGY_ERR_OPENCL_CRUSH;
            }
        } else if (m_io == OnnxIO::Chroma || m_ycbcr) {
            err = createFloatPlanes(m_inBufCL.get(), inW, inH, 3, m_inYuvPlanes);
            if (err == RGY_ERR_NONE) {
                err = (m_io == OnnxIO::Chroma)
                    ? createFloatPlanes(m_outBufCL.get(), outW, outH, 2, m_outChromaPlanes)
                    : createFloatPlanes(m_outBufCL.get(), outW, outH, 3, m_outYuvPlanes);
            }
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("onnx: YUV tensor plane viewの作成に失敗しました。\n"));
                return err;
            }

            auto cropToYuv444Param = std::make_shared<RGYFilterParamCrop>();
            cropToYuv444Param->frameIn = prm->frameIn;
            cropToYuv444Param->frameOut = yuv444Frame(m_inYuvPlanes, inW, inH);
            cropToYuv444Param->baseFps = prm->baseFps;
            m_cropToYuv444 = std::make_unique<RGYFilterCspCrop>(m_cl);
            err = m_cropToYuv444->init(cropToYuv444Param, m_pLog);
            if (err != RGY_ERR_NONE) {
                return err;
            }

            auto outputYuv444 = (m_io == OnnxIO::Chroma)
                ? yuv444Frame(m_inYuvPlanes, outW, outH)
                : yuv444Frame(m_outYuvPlanes, outW, outH);
            if (m_io == OnnxIO::Chroma) {
                outputYuv444.ptr[1] = (uint8_t *)m_outChromaPlanes[0]->mem();
                outputYuv444.ptr[2] = (uint8_t *)m_outChromaPlanes[1]->mem();
            }
            auto cropFromYuv444Param = std::make_shared<RGYFilterParamCrop>();
            cropFromYuv444Param->frameIn = outputYuv444;
            cropFromYuv444Param->frameOut = frameOut;
            cropFromYuv444Param->baseFps = prm->baseFps;
            m_cropFromYuv444 = std::make_unique<RGYFilterCspCrop>(m_cl);
            err = m_cropFromYuv444->init(cropFromYuv444Param, m_pLog);
            if (err != RGY_ERR_NONE) {
                return err;
            }
        } else if (m_io == OnnxIO::RGB || m_io == OnnxIO::RGBNoise) {
            err = createFloatPlanes(m_inBufCL.get(), inW, inH, 3, m_inRgbPlanes);
            if (err == RGY_ERR_NONE) {
                err = createFloatPlanes(m_outBufCL.get(), outW, outH, 3, m_outRgbPlanes);
            }
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("onnx: RGB tensor plane viewの作成に失敗しました。\n"));
                return err;
            }

            auto cropToRgbParam = std::make_shared<RGYFilterParamCrop>();
            cropToRgbParam->frameIn = prm->frameIn;
            cropToRgbParam->frameOut = rgbFrame(m_inRgbPlanes, inW, inH);
            cropToRgbParam->baseFps = prm->baseFps;
            cropToRgbParam->matrix = onnx_coeff_id_to_matrix(matrixSel);
            cropToRgbParam->colorrange = rangeTV ? RGY_COLORRANGE_LIMITED : RGY_COLORRANGE_FULL;
            cropToRgbParam->chroma420Interpolate = true;
            m_cropToRgb = std::make_unique<RGYFilterCspCrop>(m_cl);
            err = m_cropToRgb->init(cropToRgbParam, m_pLog);
            if (err != RGY_ERR_NONE) {
                return err;
            }

            auto cropFromRgbParam = std::make_shared<RGYFilterParamCrop>();
            cropFromRgbParam->frameIn = rgbFrame(m_outRgbPlanes, outW, outH);
            cropFromRgbParam->frameOut = frameOut;
            cropFromRgbParam->baseFps = prm->baseFps;
            cropFromRgbParam->matrix = onnx_coeff_id_to_matrix(matrixSelOut);
            cropFromRgbParam->colorrange = cropToRgbParam->colorrange;
            m_cropFromRgb = std::make_unique<RGYFilterCspCrop>(m_cl);
            err = m_cropFromRgb->init(cropFromRgbParam, m_pLog);
            if (err != RGY_ERR_NONE) {
                return err;
            }
        }
    } else {
        // host-readback scratch
        m_inBuf.resize((size_t)m_inC * m_modelInW * m_modelInH);
        m_outBuf.resize(m_ov->outElemCount());
        // RGB->YUV post needs normalised chroma at output luma res before the
        // 4:2:0 downsample (only when we synthesise chroma from RGB).
        if ((m_io == OnnxIO::RGB || m_io == OnnxIO::RGBNoise) && !m_ycbcr) {
            m_u444.resize((size_t)outW * outH);
            m_v444.resize((size_t)outW * outH);
        }
        m_inStaging  = m_cl->createFrameBuffer(inW,  inH,  inCsp, prm->frameIn.bitdepth, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
        m_outStaging = m_cl->createFrameBuffer(outW, outH, inCsp, prm->frameIn.bitdepth, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
        if (!m_inStaging || !m_outStaging) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: failed to allocate staging frame buffers.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    // Opt-in end-of-chain resize. The network produces output at (outW x outH);
    // if out_res= was given, run an internal RGYFilterResize AFTER the core to
    // land an arbitrary final resolution in one pass (CNN THEN resize, the
    // correct order, which the global --vpp-resize stage cannot do because it
    // runs before this filter). Reuses the shared resampler family.
    m_postResize.reset();
    if (prm->onnx.postResizeW != 0 && prm->onnx.postResizeH != 0) {
        int tgtW = prm->onnx.postResizeW;
        int tgtH = prm->onnx.postResizeH;
        // A negative value on one axis keeps the source aspect (magnitude =
        // rounding step), matching --output-res. The integer CNN scale preserves
        // the source aspect, so resolving against (outW x outH) + the input SAR
        // gives the DAR-correct result.
        if (tgtW < 0 || tgtH < 0) {
            sInputCrop nocrop;
            memset(&nocrop, 0, sizeof(nocrop));
            set_auto_resolution(tgtW, tgtH, 1, 1, outW, outH, prm->sar[0], prm->sar[1],
                2, 2, RGYResizeResMode::Normal, false, nocrop);
        }
        if (tgtW > 0 && tgtH > 0 && (tgtW != outW || tgtH != outH)) {
            auto resizeParam = std::make_shared<RGYFilterParamResize>();
            resizeParam->interp = (prm->onnx.postResizeAlgo == RGY_VPP_RESIZE_AUTO)
                                  ? RGY_VPP_RESIZE_LANCZOS4 : prm->onnx.postResizeAlgo;
            resizeParam->frameIn  = prm->frameOut;             // network output: outW x outH, csp/pitch set above
            resizeParam->frameOut = prm->frameOut;
            resizeParam->frameOut.width  = tgtW;
            resizeParam->frameOut.height = tgtH;
            resizeParam->baseFps       = prm->baseFps;
            resizeParam->bOutOverwrite = false;
            m_postResize = std::make_unique<RGYFilterResize>(m_cl);
            auto rsts = m_postResize->init(resizeParam, m_pLog);
            if (rsts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("onnx: failed to init end-of-chain resize: %s.\n"), get_err_mes(rsts));
                return rsts;
            }
            // Report the FINAL (resized) frame info to the pipeline; m_frameBuf
            // stays at outW x outH as the core intermediate.
            prm->frameOut = resizeParam->frameOut;
        }
    }

    static const TCHAR *ioName[] = { _T("luma-sr"), _T("gray+noise"), _T("chroma"), _T("rgb"), _T("rgb+noise") };
    tstring info = strsprintf(_T("onnx: %s  %dx%d -> %dx%d (x%d)  io=%s%s  path=%s"),
        PathGetFilename(prm->onnx.modelFile).c_str(), inW, inH, outW, outH, m_scale,
        ioName[(int)m_io], (m_ycbcr && m_io == OnnxIO::RGB) ? _T("(ycbcr)") : _T(""),
        m_useOcl ? _T("ocl") : _T("host"));
    if (prm->frameIn.bitdepth > 8) {
        info += strsprintf(_T(" %dbit"), prm->frameIn.bitdepth);
    }
    if (m_io == OnnxIO::RGB || m_io == OnnxIO::RGBNoise || m_io == OnnxIO::Chroma) {
        info += strsprintf(_T(" matrix=bt%d range=%s"), matrixSel, rangeTV ? _T("tv") : _T("pc"));
        if (matrixSelOut != matrixSel) {
            info += strsprintf(_T(" matrix_out=bt%d"), matrixSelOut);
        }
    }
    if (m_io == OnnxIO::GrayNoise || m_io == OnnxIO::RGBNoise) {
        info += strsprintf(_T(" noise=%d"), noiseClamped);
    }
    if (m_modelInW != inW || m_modelInH != inH) {
        info += strsprintf(_T(" pad-input=%dx%d"), m_modelInW, m_modelInH);
    }
    if (!m_useOcl) {
        info += strsprintf(_T(" device=%s"), effectiveDevice.c_str());
        if (usingOpenCLRemoteContext) {
            info += _T(" remote-ocl");
        }
    }
    if (!m_ov->deviceFullName().empty()) {
        info += strsprintf(_T(" [%s]"), m_ov->deviceFullName().c_str());
    }
    if (!m_ov->inferencePrecision().empty()) {
        info += strsprintf(_T(" prec=%s"), m_ov->inferencePrecision().c_str());
    }
    if (m_postResize) {
        info += strsprintf(_T(" -> out_res %dx%d (%s)"), prm->frameOut.width, prm->frameOut.height,
            get_cx_desc(list_vpp_resize, (prm->onnx.postResizeAlgo == RGY_VPP_RESIZE_AUTO)
                ? RGY_VPP_RESIZE_LANCZOS4 : prm->onnx.postResizeAlgo));
    }
    setFilterInfo(info);
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterOnnx::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (m_temporalT > 1) {
        return runTemporal(pInputFrame, ppOutputFrames, pOutputFrameNum, queue, wait_events, event);
    }
    if (m_ovm) {
        return runMask(pInputFrame, ppOutputFrames, pOutputFrameNum, queue, wait_events, event);
    }
    if (pInputFrame->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        return RGY_ERR_NONE;
    }
    // The CNN core writes its (outW x outH) result into m_frameBuf.
    auto pOutFrame = m_frameBuf[0].get();
    RGYFrameInfo *coreFrame = &pOutFrame->frame;
    coreFrame->picstruct = pInputFrame->picstruct;
    coreFrame->timestamp = pInputFrame->timestamp;
    coreFrame->duration  = pInputFrame->duration;
    coreFrame->flags     = pInputFrame->flags;

    // When an end-of-chain resize is active, the core signals no completion event
    // (the in-order queue serialises the resize after it, and the resize signals
    // the real event); otherwise the core signals the event directly.
    RGYOpenCLEvent *coreEvent = m_postResize ? nullptr : event;
    auto cerr = !m_useOcl
        ? runHost(pInputFrame, coreFrame, queue, wait_events, coreEvent)
        : (m_io == OnnxIO::Chroma || m_ycbcr
            ? runOclYuv444(pInputFrame, coreFrame, queue, wait_events, coreEvent)
            : (m_io == OnnxIO::RGB || m_io == OnnxIO::RGBNoise)
            ? runOclRGB(pInputFrame, coreFrame, queue, wait_events, coreEvent)
            : runOcl(pInputFrame, coreFrame, queue, wait_events, coreEvent));
    if (cerr != RGY_ERR_NONE) {
        return cerr;
    }

    if (!m_postResize) {
        ppOutputFrames[0] = coreFrame;
        *pOutputFrameNum = 1;
        return RGY_ERR_NONE;
    }
    // Resize the core output to the requested resolution. bOutOverwrite=false =>
    // the sub-filter writes into its own buffer and returns it in resizeOut[0].
    RGYFrameInfo *resizeOut[1] = { nullptr };
    int resizeNum = 0;
    auto rerr = m_postResize->filter(coreFrame, resizeOut, &resizeNum, queue, {}, event);
    if (rerr != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: end-of-chain resize failed: %s.\n"), get_err_mes(rerr));
        return rerr;
    }
    ppOutputFrames[0] = resizeOut[0];
    *pOutputFrameNum = 1;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterOnnx::runOcl(const RGYFrameInfo *in, RGYFrameInfo *out,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const int inW  = in->width;
    const int inH  = in->height;
    const int outW = out->width;
    const int outH = out->height;
    const int cInW = inW / 2;
    const int cInH = inH / 2;
    const int cOutW = cInW * m_scale;
    const int cOutH = cInH * m_scale;
    const int pixSize = (m_bitdepth > 8) ? 2 : 1; // kernel pitches are in samples

    // 1. pack + normalise the input luma into the network input buffer.
    {
        RGYWorkSize local(32, 8);
        RGYWorkSize global(inW, inH);
        auto err = m_program->kernel("pack_norm_y").config(queue, local, global, wait_events, nullptr).launch(
            (cl_mem)in->ptr[0], in->pitch[0] / pixSize, m_inBufCL->mem(), inW, inH, m_maxval);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: pack_norm_y failed: %s.\n"), get_err_mes(err)); return err; }
    }

    if (m_io == OnnxIO::GrayNoise) {
        const size_t planeBytes = (size_t)inW * inH * sizeof(float);
        const auto clerr = clEnqueueFillBuffer(queue.get(), m_inBufCL->mem(), &m_sigmaNorm, sizeof(m_sigmaNorm),
            planeBytes, planeBytes, 0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: ノイズ強度平面の設定に失敗しました: %s。\n"), cl_errmes(clerr));
            return err_cl_to_rgy(clerr);
        }
    }

    // 2. inference, enqueued on the SAME in-order queue right after the pack
    //    kernel, so it reads the freshly packed input with no host finish().
    auto err = m_ov->inferShared();
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: inference failed.\n")); return err; }

    // 3a. unpack + denormalise the network output into the output luma plane.
    {
        RGYWorkSize local(32, 8);
        RGYWorkSize global(outW, outH);
        err = m_program->kernel("unpack_denorm_y").config(queue, local, global, {}, nullptr).launch(
            m_outBufCL->mem(), (cl_mem)out->ptr[0], out->pitch[0] / pixSize, outW, outH, m_maxval);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: unpack_denorm_y failed: %s.\n"), get_err_mes(err)); return err; }
    }

    // 3b. resample chroma at the same integer scale. The last kernel carries the
    //     completion event.
    RGYWorkSize clocal(32, 8);
    RGYWorkSize cglobal(cOutW, cOutH);
    if (in->csp == RGY_CSP_YV12 || in->csp == RGY_CSP_YV12_16) {
        err = m_program->kernel("chroma_bilinear").config(queue, clocal, cglobal, {}, nullptr).launch(
            (cl_mem)in->ptr[1], in->pitch[1] / pixSize, 1, 0, (cl_mem)out->ptr[1], out->pitch[1] / pixSize, 1, 0, cInW, cInH, m_scale);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: chroma(U) failed: %s.\n"), get_err_mes(err)); return err; }
        err = m_program->kernel("chroma_bilinear").config(queue, clocal, cglobal, {}, event).launch(
            (cl_mem)in->ptr[2], in->pitch[2] / pixSize, 1, 0, (cl_mem)out->ptr[2], out->pitch[2] / pixSize, 1, 0, cInW, cInH, m_scale);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: chroma(V) failed: %s.\n"), get_err_mes(err)); return err; }
    } else { // RGY_CSP_NV12 / RGY_CSP_P010: plane 1 holds interleaved U,V
        err = m_program->kernel("chroma_bilinear").config(queue, clocal, cglobal, {}, nullptr).launch(
            (cl_mem)in->ptr[1], in->pitch[1] / pixSize, 2, 0, (cl_mem)out->ptr[1], out->pitch[1] / pixSize, 2, 0, cInW, cInH, m_scale);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: chroma(U) failed: %s.\n"), get_err_mes(err)); return err; }
        err = m_program->kernel("chroma_bilinear").config(queue, clocal, cglobal, {}, event).launch(
            (cl_mem)in->ptr[1], in->pitch[1] / pixSize, 2, 1, (cl_mem)out->ptr[1], out->pitch[1] / pixSize, 2, 1, cInW, cInH, m_scale);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: chroma(V) failed: %s.\n"), get_err_mes(err)); return err; }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterOnnx::createFloatPlanes(RGYCLBuf *parent, const int width, const int height, const int planeCount,
    std::array<std::unique_ptr<RGYCLBuf>, 3>& planes) {
    if (planeCount < 1 || planeCount > (int)planes.size()) {
        return RGY_ERR_INVALID_PARAM;
    }
    const size_t planeBytes = (size_t)width * height * sizeof(float);
    for (int i = 0; i < planeCount; i++) {
        cl_buffer_region region = { (size_t)i * planeBytes, planeBytes };
        cl_int clerr = CL_SUCCESS;
        auto subbuf = clCreateSubBuffer(parent->mem(), CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &clerr);
        if (clerr != CL_SUCCESS || subbuf == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: tensor plane %dのsub-buffer作成に失敗しました: %s。\n"), i, cl_errmes(clerr));
            return (clerr == CL_SUCCESS) ? RGY_ERR_MEMORY_ALLOC : err_cl_to_rgy(clerr);
        }
        planes[i] = std::make_unique<RGYCLBuf>(subbuf, CL_MEM_READ_WRITE, planeBytes);
    }
    return RGY_ERR_NONE;
}

RGYFrameInfo RGYFilterOnnx::rgbFrame(const std::array<std::unique_ptr<RGYCLBuf>, 3>& planes,
    const int width, const int height) const {
    RGYFrameInfo frame;
    frame.width = width;
    frame.height = height;
    frame.csp = RGY_CSP_RGB_F32;
    frame.bitdepth = 32;
    frame.mem_type = RGY_MEM_TYPE_GPU;
    frame.picstruct = RGY_PICSTRUCT_FRAME;
    for (int i = 0; i < 3; i++) {
        frame.ptr[i] = (uint8_t *)planes[i]->mem();
        frame.pitch[i] = width * sizeof(float);
    }
    return frame;
}

RGYFrameInfo RGYFilterOnnx::yuv444Frame(const std::array<std::unique_ptr<RGYCLBuf>, 3>& planes,
    const int width, const int height) const {
    RGYFrameInfo frame;
    frame.width = width;
    frame.height = height;
    frame.csp = RGY_CSP_YUV444_F32;
    frame.bitdepth = 32;
    frame.mem_type = RGY_MEM_TYPE_GPU;
    frame.picstruct = RGY_PICSTRUCT_FRAME;
    for (int i = 0; i < 3; i++) {
        frame.ptr[i] = (uint8_t *)planes[i]->mem();
        frame.pitch[i] = width * sizeof(float);
    }
    return frame;
}

RGY_ERR RGYFilterOnnx::runOclRGB(const RGYFrameInfo *in, RGYFrameInfo *out,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto inputYuv = *in;
    inputYuv.picstruct = RGY_PICSTRUCT_FRAME;
    auto inputRgb = rgbFrame(m_inRgbPlanes, in->width, in->height);
    RGYFrameInfo *inputRgbOut[1] = { &inputRgb };
    int outputCount = 0;
    auto err = m_cropToRgb->filter(&inputYuv, inputRgbOut, &outputCount, queue, wait_events, nullptr);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: YUV to RGB conversion failed: %s.\n"), get_err_mes(err));
        return err;
    }

    if (m_io == OnnxIO::RGBNoise) {
        const size_t planeBytes = (size_t)in->width * in->height * sizeof(float);
        const auto clerr = clEnqueueFillBuffer(queue.get(), m_inBufCL->mem(), &m_sigmaNorm, sizeof(m_sigmaNorm),
            3 * planeBytes, planeBytes, 0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("onnx: ノイズ強度平面の設定に失敗しました: %s。\n"), cl_errmes(clerr));
            return err_cl_to_rgy(clerr);
        }
    }

    err = m_ov->inferShared();
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: inference failed.\n"));
        return err;
    }

    auto outputRgb = rgbFrame(m_outRgbPlanes, out->width, out->height);
    RGYFrameInfo *outputYuv[1] = { out };
    outputCount = 0;
    err = m_cropFromRgb->filter(&outputRgb, outputYuv, &outputCount, queue, {}, event);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: RGB to YUV conversion failed: %s.\n"), get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterOnnx::runOclYuv444(const RGYFrameInfo *in, RGYFrameInfo *out,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto inputYuv = *in;
    inputYuv.picstruct = RGY_PICSTRUCT_FRAME;
    auto inputYuv444 = yuv444Frame(m_inYuvPlanes, in->width, in->height);
    RGYFrameInfo *inputYuv444Out[1] = { &inputYuv444 };
    int outputCount = 0;
    auto err = m_cropToYuv444->filter(&inputYuv, inputYuv444Out, &outputCount, queue, wait_events, nullptr);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: YUV420からYUV444_F32への変換に失敗しました: %s。\n"), get_err_mes(err));
        return err;
    }

    err = m_ov->inferShared();
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: 推論に失敗しました。\n"));
        return err;
    }

    auto outputYuv444 = (m_io == OnnxIO::Chroma)
        ? yuv444Frame(m_inYuvPlanes, out->width, out->height)
        : yuv444Frame(m_outYuvPlanes, out->width, out->height);
    if (m_io == OnnxIO::Chroma) {
        outputYuv444.ptr[1] = (uint8_t *)m_outChromaPlanes[0]->mem();
        outputYuv444.ptr[2] = (uint8_t *)m_outChromaPlanes[1]->mem();
    }
    RGYFrameInfo *outputYuv[1] = { out };
    outputCount = 0;
    err = m_cropFromYuv444->filter(&outputYuv444, outputYuv, &outputCount, queue, {}, event);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: YUV444_F32からYUV420への変換に失敗しました: %s。\n"), get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

// Pack the mapped input frame into m_inBuf (inC*inW*inH floats, CHW), applying
// the normalisation / colour conversion / conditioning each I/O mode needs.
void RGYFilterOnnx::fillInputHost(const RGYFrameInfo &hin) {
    if (m_bitdepth > 8) {
        fillInputHostT<uint16_t>(hin);
    } else {
        fillInputHostT<uint8_t>(hin);
    }
}

template<typename TPix>
void RGYFilterOnnx::fillInputHostT(const RGYFrameInfo &hin) {
    const int srcW = hin.width;
    const int srcH = hin.height;
    const int inW = m_modelInW;
    const int inH = m_modelInH;
    const size_t chSize = (size_t)inW * inH;
    const bool nv12 = (hin.csp == RGY_CSP_NV12 || hin.csp == RGY_CSP_P010);
    const int cw = srcW / 2, ch = srcH / 2;
    const uint8_t *pU = hin.ptr[1];
    const uint8_t *pV = nv12 ? (hin.ptr[1] + sizeof(TPix)) : hin.ptr[2];
    const int cStride = nv12 ? 2 : 1;
    const int cPitchU = hin.pitch[1];
    const int cPitchV = nv12 ? hin.pitch[1] : hin.pitch[2];
    float *base = m_inBuf.data();

    switch (m_io) {
    case OnnxIO::LumaSR:
    case OnnxIO::GrayNoise:
        // channel 0 = luma / maxval (models operate on [0,1] luma)
        for (int y = 0; y < inH; y++) {
            const int sy = sample_y(y, m_padT, srcH);
            const TPix *srow = (const TPix *)(hin.ptr[0] + (size_t)sy * hin.pitch[0]);
            float *drow = base + (size_t)y * inW;
            for (int x = 0; x < inW; x++) drow[x] = (float)srow[sample_x(x, m_padL, srcW)] / m_maxval;
        }
        if (m_io == OnnxIO::GrayNoise) {
            std::fill(base + chSize, base + 2 * chSize, m_sigmaNorm); // channel 1 = sigma
        }
        break;
    case OnnxIO::Chroma:
        // channels [Y, Cb, Cr] = plane/maxval, chroma bilinear-upsampled to luma res.
        for (int y = 0; y < inH; y++) {
            const int sy = sample_y(y, m_padT, srcH);
            const TPix *yrow = (const TPix *)(hin.ptr[0] + (size_t)sy * hin.pitch[0]);
            float *yd = base + (size_t)y * inW;
            float *ud = base + chSize + (size_t)y * inW;
            float *vd = base + 2 * chSize + (size_t)y * inW;
            for (int x = 0; x < inW; x++) {
                const int sx = sample_x(x, m_padL, srcW);
                yd[x] = (float)yrow[sx] / m_maxval;
                ud[x] = sample_chroma_up2<TPix>(pU, cPitchU, cStride, cw, ch, sx, sy) / m_maxval;
                vd[x] = sample_chroma_up2<TPix>(pV, cPitchV, cStride, cw, ch, sx, sy) / m_maxval;
            }
        }
        break;
    case OnnxIO::RGB:
    case OnnxIO::RGBNoise:
        if (m_ycbcr) {
            // planar YCbCr 0..1 (Y, Cb, Cr)
            for (int y = 0; y < inH; y++) {
                const int sy = sample_y(y, m_padT, srcH);
                const TPix *yrow = (const TPix *)(hin.ptr[0] + (size_t)sy * hin.pitch[0]);
                float *c0 = base + (size_t)y * inW;
                float *c1 = base + chSize + (size_t)y * inW;
                float *c2 = base + 2 * chSize + (size_t)y * inW;
                for (int x = 0; x < inW; x++) {
                    const int sx = sample_x(x, m_padL, srcW);
                    c0[x] = (float)yrow[sx] / m_maxval;
                    c1[x] = sample_chroma_up2<TPix>(pU, cPitchU, cStride, cw, ch, sx, sy) / m_maxval;
                    c2[x] = sample_chroma_up2<TPix>(pV, cPitchV, cStride, cw, ch, sx, sy) / m_maxval;
                }
            }
        } else {
            // YUV -> RGB bookend (same math as the native anime4k RGB pipeline).
            for (int y = 0; y < inH; y++) {
                const int sy = sample_y(y, m_padT, srcH);
                const TPix *yrow = (const TPix *)(hin.ptr[0] + (size_t)sy * hin.pitch[0]);
                float *rd = base + (size_t)y * inW;
                float *gd = base + chSize + (size_t)y * inW;
                float *bd = base + 2 * chSize + (size_t)y * inW;
                for (int x = 0; x < inW; x++) {
                    const int sx = sample_x(x, m_padL, srcW);
                    const float yn = ((float)yrow[sx] - m_yOff) * m_yScale;
                    const float un = (sample_chroma_up2<TPix>(pU, cPitchU, cStride, cw, ch, sx, sy) - m_cOff) * m_cScale;
                    const float vn = (sample_chroma_up2<TPix>(pV, cPitchV, cStride, cw, ch, sx, sy) - m_cOff) * m_cScale;
                    rd[x] = clampf(yn + m_matVR * vn, 0.0f, 1.0f);
                    gd[x] = clampf(yn + m_matUG * un + m_matVG * vn, 0.0f, 1.0f);
                    bd[x] = clampf(yn + m_matUB * un, 0.0f, 1.0f);
                }
            }
        }
        if (m_io == OnnxIO::RGBNoise) {
            std::fill(base + 3 * chSize, base + 4 * chSize, m_sigmaNorm); // channel 3 = sigma
        }
        break;
    }
}

// Unpack m_outBuf (outC*outW*outH floats, CHW) into the mapped output frame,
// inverting the colour conversion and resampling chroma back to 4:2:0.
void RGYFilterOnnx::writeOutputHost(const RGYFrameInfo &hout, const RGYFrameInfo &hin) {
    if (m_bitdepth > 8) {
        writeOutputHostT<uint16_t>(hout, hin);
    } else {
        writeOutputHostT<uint8_t>(hout, hin);
    }
}

template<typename TPix>
void RGYFilterOnnx::writeOutputHostT(const RGYFrameInfo &hout, const RGYFrameInfo &hin) {
    const int outW = hout.width;
    const int outH = hout.height;
    const size_t chSize = (size_t)outW * outH;
    const bool nv12 = (hout.csp == RGY_CSP_NV12 || hout.csp == RGY_CSP_P010);
    const int pixMax = (int)m_maxval;
    const float *ob = m_outBuf.data();
    uint8_t *oU = hout.ptr[1];
    uint8_t *oV = nv12 ? (hout.ptr[1] + sizeof(TPix)) : hout.ptr[2];
    const int oStride = nv12 ? 2 : 1;
    const int oPitchU = hout.pitch[1];
    const int oPitchV = nv12 ? hout.pitch[1] : hout.pitch[2];

    switch (m_io) {
    case OnnxIO::LumaSR: {
        // out 1ch luma; chroma bilinear-upscaled at the model's integer scale.
        for (int y = 0; y < outH; y++) {
            const float *srow = ob + (size_t)y * outW;
            TPix *drow = (TPix *)(hout.ptr[0] + (size_t)y * hout.pitch[0]);
            for (int x = 0; x < outW; x++) { int v = (int)(srow[x] * m_maxval + 0.5f); drow[x] = (TPix)clampi(v, 0, pixMax); }
        }
        const int cInW = hin.width / 2, cInH = hin.height / 2;
        if (!nv12) {
            upscale_bilinear<TPix>(hout.ptr[1], hout.pitch[1], 1, hin.ptr[1], hin.pitch[1], 1, cInW, cInH, m_scale, pixMax);
            upscale_bilinear<TPix>(hout.ptr[2], hout.pitch[2], 1, hin.ptr[2], hin.pitch[2], 1, cInW, cInH, m_scale, pixMax);
        } else {
            upscale_bilinear<TPix>(hout.ptr[1], hout.pitch[1], 2, hin.ptr[1], hin.pitch[1], 2, cInW, cInH, m_scale, pixMax);
            upscale_bilinear<TPix>(hout.ptr[1] + sizeof(TPix), hout.pitch[1], 2, hin.ptr[1] + sizeof(TPix), hin.pitch[1], 2, cInW, cInH, m_scale, pixMax);
        }
        break;
    }
    case OnnxIO::GrayNoise: {
        // out 1ch luma (scale=1); chroma copied straight through.
        for (int y = 0; y < outH; y++) {
            const float *srow = ob + (size_t)y * outW;
            TPix *drow = (TPix *)(hout.ptr[0] + (size_t)y * hout.pitch[0]);
            for (int x = 0; x < outW; x++) { int v = (int)(srow[x] * m_maxval + 0.5f); drow[x] = (TPix)clampi(v, 0, pixMax); }
        }
        const int cw = hin.width / 2, chh = hin.height / 2;
        const uint8_t *iU = hin.ptr[1];
        const uint8_t *iV = nv12 ? (hin.ptr[1] + sizeof(TPix)) : hin.ptr[2];
        const int iStride = nv12 ? 2 : 1;
        const int iPitchU = hin.pitch[1], iPitchV = nv12 ? hin.pitch[1] : hin.pitch[2];
        copy_plane<TPix>(oU, oPitchU, oStride, iU, iPitchU, iStride, cw, chh);
        copy_plane<TPix>(oV, oPitchV, oStride, iV, iPitchV, iStride, cw, chh);
        break;
    }
    case OnnxIO::Chroma:
        // out 2ch [Cb,Cr] at full res (scale=1); luma passes through; chroma -> 4:2:0.
        copy_plane<TPix>(hout.ptr[0], hout.pitch[0], 1, hin.ptr[0], hin.pitch[0], 1, outW, outH);
        downsample420_encode<TPix>(oU, oPitchU, oStride, ob + 0 * chSize, outW, outH, m_maxval, 0.0f, pixMax);
        downsample420_encode<TPix>(oV, oPitchV, oStride, ob + 1 * chSize, outW, outH, m_maxval, 0.0f, pixMax);
        break;
    case OnnxIO::RGB:
    case OnnxIO::RGBNoise:
        if (m_ycbcr) {
            // out planar YCbCr 0..1 -> Y plane, chroma -> 4:2:0.
            for (int y = 0; y < outH; y++) {
                const float *srow = ob + (size_t)y * outW;
                TPix *drow = (TPix *)(hout.ptr[0] + (size_t)y * hout.pitch[0]);
                for (int x = 0; x < outW; x++) { int v = (int)(srow[x] * m_maxval + 0.5f); drow[x] = (TPix)clampi(v, 0, pixMax); }
            }
            downsample420_encode<TPix>(oU, oPitchU, oStride, ob + 1 * chSize, outW, outH, m_maxval, 0.0f, pixMax);
            downsample420_encode<TPix>(oV, oPitchV, oStride, ob + 2 * chSize, outW, outH, m_maxval, 0.0f, pixMax);
        } else {
            // RGB -> YUV bookend. Y written directly; U,V stored normalised then 4:2:0-downsampled.
            for (int y = 0; y < outH; y++) {
                const float *rr = ob + 0 * chSize + (size_t)y * outW;
                const float *gg = ob + 1 * chSize + (size_t)y * outW;
                const float *bb = ob + 2 * chSize + (size_t)y * outW;
                TPix *yd = (TPix *)(hout.ptr[0] + (size_t)y * hout.pitch[0]);
                float *un = m_u444.data() + (size_t)y * outW;
                float *vn = m_v444.data() + (size_t)y * outW;
                for (int x = 0; x < outW; x++) {
                    const float R = rr[x], G = gg[x], B = bb[x];
                    const float Yn = m_matRY * R + m_matGY * G + m_matBY * B;
                    un[x] = m_matRU * R + m_matGU * G + m_matBU * B;
                    vn[x] = m_matRV * R + m_matGV * G + m_matBV * B;
                    const int v = (int)(Yn * m_yRange + m_yOff + 0.5f);
                    yd[x] = (TPix)clampi(v, 0, pixMax);
                }
            }
            downsample420_encode<TPix>(oU, oPitchU, oStride, m_u444.data(), outW, outH, m_cRange, m_cOff, pixMax);
            downsample420_encode<TPix>(oV, oPitchV, oStride, m_v444.data(), outW, outH, m_cRange, m_cOff, pixMax);
        }
        break;
    }
}

// -------------------- two-input inpainting mode (mask=) --------------------
#if ENABLE_AVSW_READER
// Decode the first video frame of an image file (png/bmp/jpg and friends) with the
// ffmpeg libraries QSVEnc already links, returned as grey [0,1] floats. Masks are
// effectively binary, so the exact grey weighting of colour inputs does not matter.
static RGY_ERR loadImageGray(const tstring &path, std::vector<float> &gray, int &width, int &height, tstring &errMessage) {
    const auto pathA = tchar_to_string(path, CP_UTF8);
    AVFormatContext *fmt = nullptr;
    if (avformat_open_input(&fmt, pathA.c_str(), nullptr, nullptr) != 0) {
        errMessage = _T("could not open the file");
        return RGY_ERR_FILE_OPEN;
    }
    RGY_ERR ret = RGY_ERR_INVALID_DATA_TYPE;
    errMessage = _T("could not decode an image frame");
    AVCodecContext *dec = nullptr;
    AVFrame *frame = av_frame_alloc();
    AVPacket *pkt = av_packet_alloc();
    do {
        if (avformat_find_stream_info(fmt, nullptr) < 0) break;
        int vidIdx = -1;
        for (unsigned int i = 0; i < fmt->nb_streams; i++) {
            if (fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) { vidIdx = (int)i; break; }
        }
        if (vidIdx < 0) break;
        const AVCodec *codec = avcodec_find_decoder(fmt->streams[vidIdx]->codecpar->codec_id);
        if (!codec) break;
        dec = avcodec_alloc_context3(codec);
        if (!dec || avcodec_parameters_to_context(dec, fmt->streams[vidIdx]->codecpar) < 0) break;
        if (avcodec_open2(dec, codec, nullptr) < 0) break;
        bool got = false;
        while (!got && av_read_frame(fmt, pkt) >= 0) {
            if (pkt->stream_index == vidIdx) {
                if (avcodec_send_packet(dec, pkt) == 0 && avcodec_receive_frame(dec, frame) == 0) got = true;
            }
            av_packet_unref(pkt);
        }
        if (!got) { // single-image decoders may hold the frame until flushed
            avcodec_send_packet(dec, nullptr);
            got = (avcodec_receive_frame(dec, frame) == 0);
        }
        if (!got) break;
        width = frame->width;
        height = frame->height;
        gray.resize((size_t)width * height);
        const auto pixfmt = (AVPixelFormat)frame->format;
        // packed RGB layouts: (bytes per pixel, offset of the first colour byte)
        int bpp = 0, coff = 0;
        switch (pixfmt) {
        case AV_PIX_FMT_RGB24: case AV_PIX_FMT_BGR24: bpp = 3; coff = 0; break;
        case AV_PIX_FMT_RGBA:  case AV_PIX_FMT_BGRA:  bpp = 4; coff = 0; break;
        case AV_PIX_FMT_ARGB:  case AV_PIX_FMT_ABGR:  bpp = 4; coff = 1; break;
        default: break;
        }
        if (bpp > 0) {
            for (int y = 0; y < height; y++) {
                const uint8_t *src = frame->data[0] + (size_t)y * frame->linesize[0];
                float *dst = gray.data() + (size_t)y * width;
                for (int x = 0; x < width; x++) {
                    const uint8_t *px = src + (size_t)x * bpp + coff;
                    dst[x] = (px[0] + px[1] + px[2]) / (3.0f * 255.0f);
                }
            }
            ret = RGY_ERR_NONE;
        } else if (pixfmt == AV_PIX_FMT_GRAY8
                || pixfmt == AV_PIX_FMT_YUV420P  || pixfmt == AV_PIX_FMT_YUVJ420P
                || pixfmt == AV_PIX_FMT_YUV422P  || pixfmt == AV_PIX_FMT_YUVJ422P
                || pixfmt == AV_PIX_FMT_YUV444P  || pixfmt == AV_PIX_FMT_YUVJ444P
                || pixfmt == AV_PIX_FMT_NV12) {
            // plane 0 is 8-bit luma in all of these
            for (int y = 0; y < height; y++) {
                const uint8_t *src = frame->data[0] + (size_t)y * frame->linesize[0];
                float *dst = gray.data() + (size_t)y * width;
                for (int x = 0; x < width; x++) dst[x] = src[x] / 255.0f;
            }
            ret = RGY_ERR_NONE;
        } else {
            errMessage = strsprintf(_T("unsupported mask pixel format: %s"),
                char_to_tstring(av_get_pix_fmt_name(pixfmt) ? av_get_pix_fmt_name(pixfmt) : "unknown").c_str());
        }
    } while (false);
    av_packet_free(&pkt);
    av_frame_free(&frame);
    if (dec) avcodec_free_context(&dec);
    avformat_close_input(&fmt);
    return ret;
}
#else
static RGY_ERR loadImageGray(const tstring &, std::vector<float> &, int &, int &, tstring &errMessage) {
    errMessage = _T("this build lacks the avcodec image reader needed for mask=");
    return RGY_ERR_UNSUPPORTED;
}
#endif //#if ENABLE_AVSW_READER

// bilinear resample of a single float plane (half-pixel-centre convention)
static void resizeBilinearPlaneF(const float *src, const int srcW, const int srcH, float *dst, const int dstW, const int dstH) {
    for (int y = 0; y < dstH; y++) {
        const float fy = (dstH > 1) ? ((y + 0.5f) * srcH / dstH - 0.5f) : 0.0f;
        int y0 = (int)std::floor(fy);
        const float wy = fy - y0;
        const int y1 = std::min(y0 + 1, srcH - 1);
        y0 = std::max(y0, 0);
        for (int x = 0; x < dstW; x++) {
            const float fx = (dstW > 1) ? ((x + 0.5f) * srcW / dstW - 0.5f) : 0.0f;
            int x0 = (int)std::floor(fx);
            const float wx = fx - x0;
            const int x1 = std::min(x0 + 1, srcW - 1);
            x0 = std::max(x0, 0);
            const float a = src[(size_t)y0 * srcW + x0] * (1.0f - wx) + src[(size_t)y0 * srcW + x1] * wx;
            const float b = src[(size_t)y1 * srcW + x0] * (1.0f - wx) + src[(size_t)y1 * srcW + x1] * wx;
            dst[(size_t)y * dstW + x] = a * (1.0f - wy) + b * wy;
        }
    }
}

// bilinear point sample of a float plane at (fx, fy) in source pixel coordinates
static inline float sampleBilinearF(const float *src, const int srcW, const int srcH, const float fx, const float fy) {
    int x0 = (int)std::floor(fx);
    int y0 = (int)std::floor(fy);
    const float wx = fx - x0;
    const float wy = fy - y0;
    const int x1 = std::min(x0 + 1, srcW - 1);
    const int y1 = std::min(y0 + 1, srcH - 1);
    x0 = std::max(x0, 0);
    y0 = std::max(y0, 0);
    const float a = src[(size_t)y0 * srcW + x0] * (1.0f - wx) + src[(size_t)y0 * srcW + x1] * wx;
    const float b = src[(size_t)y1 * srcW + x0] * (1.0f - wx) + src[(size_t)y1 * srcW + x1] * wx;
    return a * (1.0f - wy) + b * wy;
}

RGY_ERR RGYFilterOnnx::initMask(std::shared_ptr<RGYFilterParamOnnx> prm, const int inW, const int inH, const RGY_CSP inCsp) {
    if (!RGYOpenVINOMultiIO::available()) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: this OpenVINO runtime lacks the multi-tensor C API needed for mask=.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->onnx.postResizeW != 0 || prm->onnx.postResizeH != 0) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: out_res= is not supported together with mask=.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    tstring errMsg;
    m_ovm = std::make_unique<RGYOpenVINOMultiIO>();
    auto err = m_ovm->init(prm->onnx.modelFile, prm->onnx.device, errMsg, prm->onnx.precision);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: failed to load/compile model (%s): %s\n"),
            prm->onnx.device.c_str(), errMsg.c_str());
        return err;
    }
    // expected: one 3-channel image input + one 1-channel mask input (port order free),
    // one 3-channel output, all at the model's own static spatial size.
    if (m_ovm->inputNames().size() != 2 || m_ovm->outputNames().size() != 1) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: mask= needs a model with 2 inputs (image + mask) and 1 output, but this model has %d in / %d out.\n"),
            (int)m_ovm->inputNames().size(), (int)m_ovm->outputNames().size());
        return RGY_ERR_UNSUPPORTED;
    }
    auto chOf = [](const std::vector<int64_t> &s) { return (s.size() == 4) ? s[1] : (int64_t)-1; };
    if      (chOf(m_ovm->inputShape(0)) == 3 && chOf(m_ovm->inputShape(1)) == 1) { m_imgPortIdx = 0; m_mskPortIdx = 1; }
    else if (chOf(m_ovm->inputShape(0)) == 1 && chOf(m_ovm->inputShape(1)) == 3) { m_imgPortIdx = 1; m_mskPortIdx = 0; }
    else {
        AddMessage(RGY_LOG_ERROR, _T("onnx: mask= expects one 3-channel image input and one 1-channel mask input.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    const auto &simg = m_ovm->inputShape(m_imgPortIdx);
    const auto &smsk = m_ovm->inputShape(m_mskPortIdx);
    const auto &sout = m_ovm->outputShape(0);
    m_maskModelW = (int)simg[3];
    m_maskModelH = (int)simg[2];
    if (m_maskModelW <= 0 || m_maskModelH <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: mask= needs a model with a static input size.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if ((int)smsk[3] != m_maskModelW || (int)smsk[2] != m_maskModelH
        || chOf(sout) != 3 || (int)sout[3] != m_maskModelW || (int)sout[2] != m_maskModelH) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: mask= expects the mask input and the output at the image input's size (%dx%d).\n"),
            m_maskModelW, m_maskModelH);
        return RGY_ERR_UNSUPPORTED;
    }

    // decode the user mask once, then keep a 0/1 map at frame resolution (the
    // composite selector) and at model resolution (the network's mask input).
    if (!rgy_file_exists(prm->onnx.maskFile)) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: mask file not found: %s\n"), prm->onnx.maskFile.c_str());
        return RGY_ERR_FILE_OPEN;
    }
    std::vector<float> maskNative;
    int maskW = 0, maskH = 0;
    err = loadImageGray(prm->onnx.maskFile, maskNative, maskW, maskH, errMsg);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: failed to read mask %s: %s\n"), prm->onnx.maskFile.c_str(), errMsg.c_str());
        return err;
    }
    auto threshold01 = [](std::vector<float> &v) { for (auto &f : v) f = (f >= 0.5f) ? 1.0f : 0.0f; };
    m_maskFrame.resize((size_t)inW * inH);
    resizeBilinearPlaneF(maskNative.data(), maskW, maskH, m_maskFrame.data(), inW, inH);
    threshold01(m_maskFrame);
    m_maskModel.resize((size_t)m_maskModelW * m_maskModelH);
    resizeBilinearPlaneF(maskNative.data(), maskW, maskH, m_maskModel.data(), m_maskModelW, m_maskModelH);
    threshold01(m_maskModel);

    // the RGB<->YUV bookend reuses the single-tensor helpers at frame resolution
    // (scale = 1); the model's own size only matters between resample steps.
    m_io = OnnxIO::RGB;
    m_ycbcr = false;
    m_scale = 1;
    m_modelInW = inW;
    m_modelInH = inH;
    m_padL = 0;
    m_padT = 0;
    m_maxval = (float)((1 << prm->frameIn.bitdepth) - 1);
    m_bitdepth = prm->frameIn.bitdepth;
    m_useOcl = false;
    m_outScale = 0.0f;
    int matrixSel = 0;
    int matrixSelOut = 0;
    onnx_matrix_to_coeff_id(prm->onnx.colormatrix, inH, matrixSel);
    if (prm->onnx.colormatrixOut == RGY_MATRIX_AUTO) {
        matrixSelOut = matrixSel;
    } else {
        onnx_matrix_to_coeff_id(prm->onnx.colormatrixOut, inH, matrixSelOut);
    }
    const bool rangeTV = (prm->onnx.colorrange != RGY_COLORRANGE_FULL);
    setupColorCoeffs(matrixSel, matrixSelOut, rangeTV, (int)m_maxval);

    m_frameRGB.resize((size_t)3 * inW * inH);
    m_modelIn.resize((size_t)3 * m_maskModelW * m_maskModelH);
    m_modelOut.resize((size_t)3 * m_maskModelW * m_maskModelH);
    m_outBuf.resize((size_t)3 * inW * inH);
    m_u444.resize((size_t)inW * inH);
    m_v444.resize((size_t)inW * inH);

    auto frameOut = prm->frameOut;
    frameOut.csp    = inCsp;
    frameOut.width  = inW;
    frameOut.height = inH;
    prm->frameOut   = frameOut;
    err = AllocFrameBuf(prm->frameOut, 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: failed to allocate output frame buffer: %s.\n"), get_err_mes(err));
        return err;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    m_inStaging  = m_cl->createFrameBuffer(inW, inH, inCsp, prm->frameIn.bitdepth, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
    m_outStaging = m_cl->createFrameBuffer(inW, inH, inCsp, prm->frameIn.bitdepth, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
    if (!m_inStaging || !m_outStaging) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: failed to allocate staging frame buffers.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }

    tstring info = strsprintf(_T("onnx: %s  %dx%d  io=rgb+mask (model %dx%d)  device=%s"),
        PathGetFilename(prm->onnx.modelFile).c_str(), inW, inH, m_maskModelW, m_maskModelH, prm->onnx.device.c_str());
    info += strsprintf(_T(" matrix=bt%d range=%s"), matrixSel, rangeTV ? _T("tv") : _T("pc"));
    if (prm->frameIn.bitdepth > 8) {
        info += strsprintf(_T(" %dbit"), prm->frameIn.bitdepth);
    }
    if (!m_ovm->deviceFullName().empty()) {
        info += strsprintf(_T(" [%s]"), m_ovm->deviceFullName().c_str());
    }
    if (!m_ovm->inferencePrecision().empty()) {
        info += strsprintf(_T(" prec=%s"), m_ovm->inferencePrecision().c_str());
    }
    setFilterInfo(info);
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterOnnx::runMask(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (pInputFrame->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        return RGY_ERR_NONE;
    }
    const int fw = m_modelInW; // frame resolution (the mask path packs at frame size)
    const int fh = m_modelInH;
    const size_t chFrame = (size_t)fw * fh;
    const size_t chModel = (size_t)m_maskModelW * m_maskModelH;

    auto err = m_cl->copyFrame(&m_inStaging->frame, pInputFrame, nullptr, queue, wait_events, nullptr);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: copy input to staging failed: %s.\n"), get_err_mes(err)); return err; }
    err = m_inStaging->queueMapBuffer(queue, CL_MAP_READ, {}, RGY_CL_MAP_BLOCK_ALL);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: map input staging failed: %s.\n"), get_err_mes(err)); return err; }
    err = m_outStaging->queueMapBuffer(queue, CL_MAP_WRITE, {}, RGY_CL_MAP_BLOCK_ALL);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: map output staging failed: %s.\n"), get_err_mes(err)); return err; }
    const RGYFrameInfo &hin  = m_inStaging->mappedHost()->host();
    const RGYFrameInfo &hout = m_outStaging->mappedHost()->host();

    // 1. frame -> planar RGB [0,1] at frame resolution, then resample each channel
    //    to the model's own input size.
    packFrameRGB(hin, m_frameRGB.data());
    for (int c = 0; c < 3; c++) {
        resizeBilinearPlaneF(m_frameRGB.data() + c * chFrame, fw, fh,
                             m_modelIn.data() + c * chModel, m_maskModelW, m_maskModelH);
    }

    // 2. inference: frame + mask, bound to their ports by index.
    std::vector<const float *> ins(2);
    ins[m_imgPortIdx] = m_modelIn.data();
    ins[m_mskPortIdx] = m_maskModel.data();
    std::vector<float *> outs = { m_modelOut.data() };
    tstring errMsg;
    err = m_ovm->infer(ins, outs, errMsg);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: inference failed: %s\n"), errMsg.c_str());
        m_inStaging->unmapBuffer(queue);
        m_outStaging->unmapBuffer(queue);
        return err;
    }

    // 3. exported inpainting models disagree on output range ([0,1] vs [0,255]);
    //    settle it once from the first frame's maximum.
    if (m_outScale == 0.0f) {
        float maxv = 0.0f;
        for (auto v : m_modelOut) maxv = std::max(maxv, v);
        m_outScale = (maxv > 2.0f) ? (1.0f / 255.0f) : 1.0f;
        AddMessage(RGY_LOG_DEBUG, _T("onnx: model output range detected as [0,%d].\n"), (maxv > 2.0f) ? 255 : 1);
    }

    // 4. composite: masked pixels take the (resampled) model result; everything
    //    else keeps the frame's own RGB, so unmasked content never passes through
    //    the model or its resampling.
    for (int c = 0; c < 3; c++) {
        const float *modelOut = m_modelOut.data() + c * chModel;
        const float *frameRGB = m_frameRGB.data() + c * chFrame;
        float *dst = m_outBuf.data() + c * chFrame;
        for (int y = 0; y < fh; y++) {
            const float my = ((y + 0.5f) * m_maskModelH) / fh - 0.5f;
            for (int x = 0; x < fw; x++) {
                const size_t i = (size_t)y * fw + x;
                if (m_maskFrame[i] > 0.5f) {
                    const float mx = ((x + 0.5f) * m_maskModelW) / fw - 0.5f;
                    dst[i] = clampf(sampleBilinearF(modelOut, m_maskModelW, m_maskModelH, mx, my) * m_outScale, 0.0f, 1.0f);
                } else {
                    dst[i] = frameRGB[i];
                }
            }
        }
    }

    // 5. RGB -> YUV into the mapped output frame (the standard RGB unpack).
    writeOutputHost(hout, hout);

    err = m_inStaging->unmapBuffer(queue);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: unmap input staging failed: %s.\n"), get_err_mes(err)); return err; }
    err = m_outStaging->unmapBuffer(queue);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: unmap output staging failed: %s.\n"), get_err_mes(err)); return err; }

    auto pOutFrame = m_frameBuf[0].get();
    RGYFrameInfo *coreFrame = &pOutFrame->frame;
    err = m_cl->copyFrame(coreFrame, &m_outStaging->frame, nullptr, queue, {}, event);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: copy staging to output failed: %s.\n"), get_err_mes(err)); return err; }
    // metadata AFTER the copy: copyFrame overwrites frame props from its source
    coreFrame->picstruct    = pInputFrame->picstruct;
    coreFrame->timestamp    = pInputFrame->timestamp;
    coreFrame->duration     = pInputFrame->duration;
    coreFrame->flags        = pInputFrame->flags;
    coreFrame->inputFrameId = pInputFrame->inputFrameId;

    ppOutputFrames[0] = coreFrame;
    *pOutputFrameNum = 1;
    return RGY_ERR_NONE;
}

// -------------------- multi-frame temporal window (frames= > 1) --------------------
// Pack one mapped input frame's YUV as planar RGB [0,1] CHW into dst
// (3*modelInW*modelInH floats). This is the RGB case of fillInputHost lifted out for
// a single frame so the temporal path can stack several frames on the channel axis;
// the single-frame fillInputHost path is left exactly as it was.
void RGYFilterOnnx::packFrameRGB(const RGYFrameInfo &hin, float *dst) {
    if (m_bitdepth > 8) {
        packFrameRGBT<uint16_t>(hin, dst);
    } else {
        packFrameRGBT<uint8_t>(hin, dst);
    }
}

template<typename TPix>
void RGYFilterOnnx::packFrameRGBT(const RGYFrameInfo &hin, float *dst) {
    const int srcW = hin.width;
    const int srcH = hin.height;
    const int inW = m_modelInW;
    const int inH = m_modelInH;
    const size_t chSize = (size_t)inW * inH;
    const bool nv12 = (hin.csp == RGY_CSP_NV12 || hin.csp == RGY_CSP_P010);
    const int cw = srcW / 2, ch = srcH / 2;
    const uint8_t *pU = hin.ptr[1];
    const uint8_t *pV = nv12 ? (hin.ptr[1] + sizeof(TPix)) : hin.ptr[2];
    const int cStride = nv12 ? 2 : 1;
    const int cPitchU = hin.pitch[1];
    const int cPitchV = nv12 ? hin.pitch[1] : hin.pitch[2];
    if (m_ycbcr) {
        // planar YCbCr 0..1 (Y, Cb, Cr)
        for (int y = 0; y < inH; y++) {
            const int sy = sample_y(y, m_padT, srcH);
            const TPix *yrow = (const TPix *)(hin.ptr[0] + (size_t)sy * hin.pitch[0]);
            float *c0 = dst + (size_t)y * inW;
            float *c1 = dst + chSize + (size_t)y * inW;
            float *c2 = dst + 2 * chSize + (size_t)y * inW;
            for (int x = 0; x < inW; x++) {
                const int sx = sample_x(x, m_padL, srcW);
                c0[x] = (float)yrow[sx] / m_maxval;
                c1[x] = sample_chroma_up2<TPix>(pU, cPitchU, cStride, cw, ch, sx, sy) / m_maxval;
                c2[x] = sample_chroma_up2<TPix>(pV, cPitchV, cStride, cw, ch, sx, sy) / m_maxval;
            }
        }
    } else {
        // YUV -> RGB bookend (same maths as the native anime4k RGB pipeline).
        for (int y = 0; y < inH; y++) {
            const int sy = sample_y(y, m_padT, srcH);
            const TPix *yrow = (const TPix *)(hin.ptr[0] + (size_t)sy * hin.pitch[0]);
            float *rd = dst + (size_t)y * inW;
            float *gd = dst + chSize + (size_t)y * inW;
            float *bd = dst + 2 * chSize + (size_t)y * inW;
            for (int x = 0; x < inW; x++) {
                const int sx = sample_x(x, m_padL, srcW);
                const float yn = ((float)yrow[sx] - m_yOff) * m_yScale;
                const float un = (sample_chroma_up2<TPix>(pU, cPitchU, cStride, cw, ch, sx, sy) - m_cOff) * m_cScale;
                const float vn = (sample_chroma_up2<TPix>(pV, cPitchV, cStride, cw, ch, sx, sy) - m_cOff) * m_cScale;
                rd[x] = clampf(yn + m_matVR * vn, 0.0f, 1.0f);
                gd[x] = clampf(yn + m_matUG * un + m_matVG * vn, 0.0f, 1.0f);
                bd[x] = clampf(yn + m_matUB * un, 0.0f, 1.0f);
            }
        }
    }
}

// Buffer the last T RGB frames and emit a centred temporal window (k=(T-1)/2 frames
// on each side, edges replicated). 1-in-1-out with a k-frame latency: output i is
// produced when input i+k arrives, and the trailing k outputs are drained at flush,
// one per null-input call (the pipeline re-invokes flush until the filter returns 0).
RGY_ERR RGYFilterOnnx::runTemporal(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const int T = m_temporalT;
    const int k = (T - 1) / 2;
    const size_t chSize = (size_t)m_modelInW * m_modelInH;

    if (pInputFrame->ptr[0] != nullptr) {
        // real frame: copy to host-mappable staging, map, pack as RGB, buffer it.
        auto err = m_cl->copyFrame(&m_inStaging->frame, pInputFrame, nullptr, queue, wait_events, nullptr);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: copy input to staging failed: %s.\n"), get_err_mes(err)); return err; }
        err = m_inStaging->queueMapBuffer(queue, CL_MAP_READ, {}, RGY_CL_MAP_BLOCK_ALL);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: map input staging failed: %s.\n"), get_err_mes(err)); return err; }
        const RGYFrameInfo &hin = m_inStaging->mappedHost()->host();

        RingFrame rf;
        rf.rgb.resize(3 * chSize);
        packFrameRGB(hin, rf.rgb.data());
        rf.timestamp    = pInputFrame->timestamp;
        rf.duration     = pInputFrame->duration;
        rf.picstruct    = pInputFrame->picstruct;
        rf.flags        = pInputFrame->flags;
        rf.inputFrameId = pInputFrame->inputFrameId;

        err = m_inStaging->unmapBuffer(queue);
        if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: unmap input staging failed: %s.\n"), get_err_mes(err)); return err; }

        m_ring.push_back(std::move(rf));
        m_recvCount++;
        while ((int)m_ring.size() > T) { m_ring.pop_front(); m_ringBaseIdx++; }

        // frame (m_emitCount + k) has now arrived => output m_emitCount is complete.
        if (m_recvCount - 1 >= m_emitCount + (int64_t)k) {
            return emitTemporalOutput(m_emitCount++, ppOutputFrames, pOutputFrameNum, queue, event);
        }
        *pOutputFrameNum = 0;
        return RGY_ERR_NONE;
    }

    // flush: drain the trailing outputs, one per null-input call, then report empty.
    if (m_emitCount < m_recvCount) {
        return emitTemporalOutput(m_emitCount++, ppOutputFrames, pOutputFrameNum, queue, event);
    }
    *pOutputFrameNum = 0;
    return RGY_ERR_NONE;
}

// Assemble the T-frame window centred on global index outIdx (edges replicated) into
// the network input tensor, run inference, write the RGB result into m_frameBuf[0]
// (optionally post-resized) and stamp it with the centre frame's metadata.
RGY_ERR RGYFilterOnnx::emitTemporalOutput(int64_t outIdx, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    const int T = m_temporalT;
    const int k = (T - 1) / 2;
    const size_t chSize = (size_t)m_modelInW * m_modelInH;
    const int ringN = (int)m_ring.size();
    if (ringN <= 0) { *pOutputFrameNum = 0; return RGY_ERR_NONE; }

    // 1. gather the window [outIdx-k .. outIdx+k] from the ring, replicating at the
    //    sequence edges, into m_inBuf [T*3, H, W] (frame-major on the channel axis).
    for (int j = 0; j < T; j++) {
        int64_t g = outIdx - k + j;
        int64_t r = g - m_ringBaseIdx;
        if (r < 0) r = 0; else if (r >= ringN) r = ringN - 1;
        const std::vector<float> &src = m_ring[(size_t)r].rgb;
        std::copy(src.begin(), src.end(), m_inBuf.begin() + (size_t)j * 3 * chSize);
    }

    // 2. map output staging, run inference, unpack RGB -> YUV into it.
    auto pOutFrame = m_frameBuf[0].get();
    RGYFrameInfo *coreFrame = &pOutFrame->frame;
    auto err = m_outStaging->queueMapBuffer(queue, CL_MAP_WRITE, {}, RGY_CL_MAP_BLOCK_ALL);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: map output staging failed: %s.\n"), get_err_mes(err)); return err; }
    const RGYFrameInfo &hout = m_outStaging->mappedHost()->host();
    err = m_ov->infer(m_inBuf.data(), m_outBuf.data());
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: inference failed.\n")); m_outStaging->unmapBuffer(queue); return err; }
    writeOutputHost(hout, hout); // RGB mode synthesises all planes from the output; the 2nd arg is unused
    err = m_outStaging->unmapBuffer(queue);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: unmap output staging failed: %s.\n"), get_err_mes(err)); return err; }

    // 3. staging -> device output; when a post-resize follows, it carries the event.
    RGYOpenCLEvent *coreEvent = m_postResize ? nullptr : event;
    err = m_cl->copyFrame(coreFrame, &m_outStaging->frame, nullptr, queue, {}, coreEvent);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: copy staging to output failed: %s.\n"), get_err_mes(err)); return err; }

    // 4. centre-frame metadata for the emitted output. Stamp AFTER copyFrame: copyFrame
    //    overwrites every frame property (incl. inputFrameId) from the staging buffer,
    //    so any stamping before it would be clobbered.
    int64_t cr = outIdx - m_ringBaseIdx;
    if (cr < 0) cr = 0; else if (cr >= ringN) cr = ringN - 1;
    const RingFrame &centre = m_ring[(size_t)cr];
    coreFrame->picstruct    = centre.picstruct;
    coreFrame->timestamp    = centre.timestamp;
    coreFrame->duration     = centre.duration;
    coreFrame->flags        = centre.flags;
    coreFrame->inputFrameId = centre.inputFrameId;

    if (!m_postResize) {
        ppOutputFrames[0] = coreFrame;
        *pOutputFrameNum = 1;
        return RGY_ERR_NONE;
    }
    RGYFrameInfo *resizeOut[1] = { nullptr };
    int resizeNum = 0;
    auto rerr = m_postResize->filter(coreFrame, resizeOut, &resizeNum, queue, {}, event);
    if (rerr != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: end-of-chain resize failed: %s.\n"), get_err_mes(rerr)); return rerr; }
    ppOutputFrames[0] = resizeOut[0];
    *pOutputFrameNum = 1;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterOnnx::runHost(const RGYFrameInfo *in, RGYFrameInfo *out,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    // 1. device input -> host-mappable staging, then map for read.
    auto err = m_cl->copyFrame(&m_inStaging->frame, in, nullptr, queue, wait_events, nullptr);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: copy input to staging failed: %s.\n"), get_err_mes(err)); return err; }
    err = m_inStaging->queueMapBuffer(queue, CL_MAP_READ, {}, RGY_CL_MAP_BLOCK_ALL);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: map input staging failed: %s.\n"), get_err_mes(err)); return err; }
    err = m_outStaging->queueMapBuffer(queue, CL_MAP_WRITE, {}, RGY_CL_MAP_BLOCK_ALL);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: map output staging failed: %s.\n"), get_err_mes(err)); return err; }
    const RGYFrameInfo &hin  = m_inStaging->mappedHost()->host();
    const RGYFrameInfo &hout = m_outStaging->mappedHost()->host();

    // 2. pack the input frame into the network tensor (per I/O mode).
    fillInputHost(hin);

    // 3. inference.
    err = m_ov->infer(m_inBuf.data(), m_outBuf.data());
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("onnx: inference failed.\n"));
        m_inStaging->unmapBuffer(queue);
        m_outStaging->unmapBuffer(queue);
        return err;
    }

    // 4. unpack the network output into the output frame (per I/O mode).
    writeOutputHost(hout, hin);

    // 5. unmap and copy staging -> device output, signalling the event.
    err = m_inStaging->unmapBuffer(queue);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: unmap input staging failed: %s.\n"), get_err_mes(err)); return err; }
    err = m_outStaging->unmapBuffer(queue);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: unmap output staging failed: %s.\n"), get_err_mes(err)); return err; }
    err = m_cl->copyFrame(out, &m_outStaging->frame, nullptr, queue, {}, event);
    if (err != RGY_ERR_NONE) { AddMessage(RGY_LOG_ERROR, _T("onnx: copy staging to output failed: %s.\n"), get_err_mes(err)); return err; }
    return RGY_ERR_NONE;
}

void RGYFilterOnnx::close() {
    m_postResize.reset();
    m_cropToRgb.reset();
    m_cropFromRgb.reset();
    m_cropToYuv444.reset();
    m_cropFromYuv444.reset();
    for (auto& plane : m_inRgbPlanes) plane.reset();
    for (auto& plane : m_outRgbPlanes) plane.reset();
    for (auto& plane : m_inYuvPlanes) plane.reset();
    for (auto& plane : m_outYuvPlanes) plane.reset();
    for (auto& plane : m_outChromaPlanes) plane.reset();
    m_inStaging.reset();
    m_outStaging.reset();
    m_inBufCL.reset();
    m_outBufCL.reset();
    m_program.reset();
    m_ov.reset();
    m_inBuf.clear();
    m_outBuf.clear();
    m_u444.clear();
    m_v444.clear();
    m_ovm.reset();
    m_maskFrame.clear();
    m_maskModel.clear();
    m_frameRGB.clear();
    m_modelIn.clear();
    m_modelOut.clear();
    m_outScale = 0.0f;
    m_ring.clear();
    m_ringBaseIdx = 0;
    m_recvCount = 0;
    m_emitCount = 0;
    m_temporalT = 1;
    m_frameBuf.clear();
}
