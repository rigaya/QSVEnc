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

#include <map>
#include <array>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#define _USE_MATH_DEFINES
#include <cmath>
#include "rgy_filter_nnedi.h"
#include "rgy_filesystem.h"
#include "rgy_resource.h"

//dot_product1で重み(nns)方向のループアンロールを行う
//これにより、一度sharedメモリからレジスタにのせたpixel情報を使いまわすことができる
#define ENABLE_DP1_WEIGHT_LOOP_UNROLL 1

//ENABLE_DP1_WEIGHT_LOOP_UNROLLに対応して通常の重みの並び [nns*2][nnxy]を変更する
//並びは[nns/WEIGHT_LOOP][nnxy][WEIGHT_LOOP][2]
#define ENABLE_DP1_WEIGHT_ARRAY_OPT (1 && ENABLE_DP1_WEIGHT_LOOP_UNROLL)

//shuffle命令を使ったweight係数の分配により高速化する
//現状、OpenCLでは正しく動作させられていないので、無効化
#define ENABLE_DP1_SHUFFLE_OPT 0

static const int THREAD_Y_LOOP_K0 = 2;
static const int THREAD_Y_LOOP_K1 = 4;

static const int NNEDI_BLOCK_X = 32;
static const int NNEDI_BLOCK_Y = 8;

static const int weight0size = 49 * 4 + 5 * 4 + 9 * 4;
static const int weight0sizenew = 4 * 65 + 4 * 5;

RGY_ERR nnedi_compute_network_0(RGYFrameInfo *pOutputPlane,
    const RGYFrameInfo *pInputPlane,
    const RGYCLBuf *weight0,
    const VppNnediPreScreen pre_screen,
    const NnediTargetField targetField,
    RGYOpenCLQueue &queue,
    RGYOpenCLProgram *nnedi_k0,
    const std::vector<RGYOpenCLEvent> &wait_events,
    RGYOpenCLEvent *event
) {
    RGYWorkSize local(NNEDI_BLOCK_X, NNEDI_BLOCK_Y);
    const char *kernel_name = "kernel_compute_network0";

    auto err = RGY_ERR_NONE;
    if ((pre_screen & VPP_NNEDI_PRE_SCREEN_MODE) == VPP_NNEDI_PRE_SCREEN_ORIGINAL) {
        RGYWorkSize global(
            pOutputPlane->width,
            divCeil(pOutputPlane->height >> 1, THREAD_Y_LOOP_K0));

        err = nnedi_k0->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
            (cl_mem)pOutputPlane->ptr[0],
            pOutputPlane->pitch[0] * (targetField == NNEDI_GEN_FIELD_TOP ? 0 : 1), //生成するほうのフィールドを選択
            pOutputPlane->pitch[0] * 2,  //1行おきなので通常の2倍
            pOutputPlane->width,
            pOutputPlane->height,
            (cl_mem)pInputPlane->ptr[0],  //有効フィールド
            pInputPlane->pitch[0] * (targetField == NNEDI_GEN_FIELD_TOP ? 1 : 0), //元となるほうのフィールドを選択
            pInputPlane->pitch[0] * 2,  //1行おきなので通常の2倍
            pInputPlane->width,
            pInputPlane->height,
            weight0->mem(),
            (int)targetField);
    } else if ((pre_screen & VPP_NNEDI_PRE_SCREEN_MODE) >= VPP_NNEDI_PRE_SCREEN_NEW) {
        RGYWorkSize global(
            divCeil(pOutputPlane->width, 4 /*4ピクセル分一度に処理する*/),
            divCeil(pOutputPlane->height >> 1, THREAD_Y_LOOP_K0));

        err = nnedi_k0->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
            (cl_mem)pOutputPlane->ptr[0],
            pOutputPlane->pitch[0] * (targetField == NNEDI_GEN_FIELD_TOP ? 0 : 1), //生成するほうのフィールドを選択
            pOutputPlane->pitch[0] * 2,  //1行おきなので通常の2倍
            pOutputPlane->width,
            pOutputPlane->height,
            (cl_mem)pInputPlane->ptr[0],  //有効フィールド
            pInputPlane->pitch[0] * (targetField == NNEDI_GEN_FIELD_TOP ? 1 : 0), //元となるほうのフィールドを選択
            pInputPlane->pitch[0] * 2,  //1行おきなので通常の2倍
            pInputPlane->width,
            pInputPlane->height,
            weight0->mem(),
            (int)targetField);
    } else {
        RGYWorkSize global(
            divCeil(pOutputPlane->width, 4 /*4ピクセル分一度に処理する*/),
            pOutputPlane->height >> 1);
        err = nnedi_k0->kernel("kernel_set_field_value").config(queue, local, global, wait_events, event).launch(
            (cl_mem)pOutputPlane->ptr[0],
            pOutputPlane->pitch[0] * (targetField == NNEDI_GEN_FIELD_TOP ? 0 : 1), //生成するほうのフィールドを選択
            pOutputPlane->pitch[0] * 2,  //1行おきなので通常の2倍
            pOutputPlane->width,
            pOutputPlane->height,
            -1);
    }
    return err;
}

RGY_ERR nnedi_compute_network_1(
    RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pInputPlane,
    const RGYCLBuf *weight10,
    const RGYCLBuf *weight11,
    const NnediTargetField targetField,
    const VppNnediQuality quality,
    const VppNnediPreScreen pre_screen,
    RGYOpenCLQueue &queue,
    RGYOpenCLProgram *nnedi_k1,
    const std::vector<RGYOpenCLEvent> &wait_events,
    RGYOpenCLEvent *event
) {
    //重み(nns)方向のループアンロール数
    //やりすぎると使用レジスタ数が増え、かえって遅くなる
    //static_assert(WEIGHT_LOOP_1 <= WARP_SIZE, "WEIGHT_LOOP < WARP_SIZE");

    RGYWorkSize local(NNEDI_BLOCK_X, NNEDI_BLOCK_Y);
    RGYWorkSize global(
        pOutputFrame->width,
        divCeil(pOutputFrame->height >> 1, THREAD_Y_LOOP_K1));

    const char *kernel_name = "kernel_compute_network1";
    auto err = nnedi_k1->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputFrame->ptr[0],
        pOutputFrame->pitch[0] * ((targetField == NNEDI_GEN_FIELD_TOP) ? 0 : 1), //生成するほうのフィールドを選択
        pOutputFrame->pitch[0] * 2, //1行おきなので通常の2倍
        pOutputFrame->width,
        pOutputFrame->height,
        (cl_mem)pInputPlane->ptr[0],  //有効フィールド
        pInputPlane->pitch[0] * (targetField == NNEDI_GEN_FIELD_TOP ? 1 : 0), //元となるほうのフィールドを選択
        pInputPlane->pitch[0] * 2,  //1行おきなので通常の2倍
        pInputPlane->width,
        pInputPlane->height,
        weight10->mem(), weight11->mem(),
        (int)quality, (int)targetField, (int)pre_screen);
    return err;
}

RGY_ERR RGYFilterNnedi::procPlane(
    RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, const NnediTargetField targetField,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event
) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamNnedi>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    {
        sInputCrop planeCrop = initCrop();
        auto err = m_cl->copyPlane(pOutputPlane, pInputPlane, &planeCrop, queue, wait_events, nullptr, targetField == NNEDI_GEN_FIELD_TOP ? RGYFrameCopyMode::FIELD_BOTTOM : RGYFrameCopyMode::FIELD_TOP);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to copyPlaneField: %s\n"), get_err_mes(err));
            return err;
        }
    }

    auto err = nnedi_compute_network_0(pOutputPlane,
        pInputPlane,
        m_weight0.get(),
        (prm->nnedi.pre_screen & VPP_NNEDI_PRE_SCREEN_MODE),
        targetField,
        queue, m_nnedi_k0.get(), {}, nullptr);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    if (!(prm->nnedi.pre_screen & VPP_NNEDI_PRE_SCREEN_ONLY)) {
        err = nnedi_compute_network_1(
            pOutputPlane,
            pInputPlane,
            m_weight1[0].get(),
            m_weight1[1].get(),
            targetField,
            prm->nnedi.quality,
            (prm->nnedi.pre_screen & (VPP_NNEDI_PRE_SCREEN_MODE | VPP_NNEDI_PRE_SCREEN_BLOCK)),
            queue, m_nnedi_k1.get(), {}, event);
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    if (err != RGY_ERR_NONE) {
        return err;
    }
    return err;
}


RGY_ERR RGYFilterNnedi::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, const NnediTargetField targetField, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        const std::vector<RGYOpenCLEvent> &plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent *plane_event = (i == RGY_CSP_PLANES[pOutputFrame->csp] - 1) ? event : nullptr;
        auto err = procPlane(&planeDst, &planeSrc, targetField, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to nnedi frame(%d) %s: %s\n"), i, cl_errmes(err));
            return err_cl_to_rgy(err);
        }
    }
    return RGY_ERR_NONE;
}

const int RGYFilterNnedi::weight_loop_0 = 2;
const int RGYFilterNnedi::weight_loop_1 = 4;
const int RGYFilterNnedi::sizeNX[] = { 8, 16, 32, 48, 8, 16, 32 };
const int RGYFilterNnedi::sizeNY[] = { 6, 6, 6, 6, 4, 4, 4 };
const int RGYFilterNnedi::sizeNN[] = { 16, 32, 64, 128, 256 };

RGYFilterNnedi::RGYFilterNnedi(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context), m_clfp16support(), m_nnedi_k0(), m_nnedi_k1(), m_weight0(), m_weight1() {
    m_name = _T("nnedi");
}

RGYFilterNnedi::~RGYFilterNnedi() {
    close();
}
RGY_ERR RGYFilterNnedi::checkParam(const std::shared_ptr<RGYFilterParamNnedi> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int hight_mul = (RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp] == RGY_CHROMAFMT_YUV420) ? 4 : 2;
    if ((prm->frameOut.height % hight_mul) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Height must be multiple of %d.\n"), hight_mul);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nnedi.field <= VPP_NNEDI_FIELD_UNKNOWN || VPP_NNEDI_FIELD_MAX <= prm->nnedi.field) {
        AddMessage(RGY_LOG_ERROR, _T("invalid value for param \"field\": %d\n"), prm->nnedi.field);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nnedi.nns < 16 || 256 < prm->nnedi.nns) {
        prm->nnedi.nns = clamp(prm->nnedi.nns, 16, 256);
        AddMessage(RGY_LOG_WARN, _T("nns should be in range of %d - %d.\n"), 16, 256);
    }
    if (prm->nnedi.nsize <= VPP_NNEDI_NSIZE_UNKNOWN || VPP_NNEDI_NSIZE_MAX <= prm->nnedi.nsize) {
        AddMessage(RGY_LOG_ERROR, _T("invalid value for param \"nsize\": %d\n"), prm->nnedi.nsize);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nnedi.quality <= VPP_NNEDI_QUALITY_UNKNOWN || VPP_NNEDI_QUALITY_MAX <= prm->nnedi.quality) {
        AddMessage(RGY_LOG_ERROR, _T("invalid value for param \"quality\": %d\n"), prm->nnedi.quality);
        return RGY_ERR_INVALID_PARAM;
    }
    if (VPP_NNEDI_PRE_SCREEN_MAX <= prm->nnedi.pre_screen) {
        AddMessage(RGY_LOG_ERROR, _T("invalid value for param \"pre_screen\": %d\n"), prm->nnedi.pre_screen);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nnedi.precision < VPP_FP_PRECISION_UNKNOWN || VPP_FP_PRECISION_MAX <= prm->nnedi.precision) {
        AddMessage(RGY_LOG_ERROR, _T("invalid value for param \"prec\": %d\n"), prm->nnedi.precision);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

shared_ptr<const float> RGYFilterNnedi::readWeights(const tstring& weightFile, HMODULE hModule) {
    shared_ptr<const float> weights;
    const uint32_t expectedFileSize = 13574928u;
    uint64_t weightFileSize = 0;
    if (weightFile.length() == 0) {
        //埋め込みデータを使用する
        void *pDataPtr = nullptr;
        weightFileSize = getEmbeddedResource(&pDataPtr, _T("NNEDI_WEIGHTBIN"), _T("EXE_DATA"), hModule);
        if (pDataPtr == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get Weights data.\n"));
        } else if (expectedFileSize != weightFileSize) {
            AddMessage(RGY_LOG_ERROR, _T("Weights data has unexpected size %lld [expected: %u].\n"),
                (long long int)weightFileSize, expectedFileSize);
        } else {
            weights = shared_ptr<const float>((const float *)pDataPtr, [](const float *x) { UNREFERENCED_PARAMETER(x); return; /*何もしない*/ });
        }
    } else {
        if (!rgy_file_exists(weightFile.c_str())) {
            AddMessage(RGY_LOG_ERROR, _T("weight file \"%s\" does not exist.\n"), weightFile.c_str());
        } else if (!rgy_get_filesize(weightFile.c_str(), &weightFileSize)) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get filesize of weight file \"%s\".\n"), weightFile.c_str());
        } else if (weightFileSize != expectedFileSize) {
            AddMessage(RGY_LOG_ERROR, _T("Weights file \"%s\" has unexpected file size %lld [expected: %u].\n"),
                weightFile.c_str(), (long long int)weightFileSize, expectedFileSize);
        } else {
            std::ifstream fin(weightFile, std::ios::in | std::ios::binary);
            if (!fin.good()) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to open weights file \"%s\".\n"), weightFile.c_str());
            } else {
                float *buffer = new float[weightFileSize / sizeof(float)];
                if (!buffer) {
                    AddMessage(RGY_LOG_ERROR, _T("Failed to allocate buffer memory for \"%s\".\n"), weightFile.c_str());
                } else {
                    weights = shared_ptr<float>(buffer, std::default_delete<float[]>());
                    if (fin.read((char *)weights.get(), weightFileSize).gcount() != (int64_t)weightFileSize) {
                        AddMessage(RGY_LOG_ERROR, _T("Failed to read weights file \"%s\".\n"), weightFile.c_str());
                        weights.reset();
                    }
                }
                fin.close();
            }
        }
    }
    return weights;
}

RGY_ERR RGYFilterNnedi::initParams(const std::shared_ptr<RGYFilterParamNnedi> prm) {
    auto weights = readWeights(prm->nnedi.weightfile, prm->hModule);
    if (!weights) {
        return RGY_ERR_INVALID_PARAM;
    }

    const int weight1size = prm->nnedi.nns * 2 * (sizeNX[prm->nnedi.nsize] * sizeNY[prm->nnedi.nsize] + 1);
    const int sizeofweight = (prm->nnedi.precision == VPP_FP_PRECISION_FP32) ? 4 : 2;
    int weight1size_tsize = 0;
    int weight1size_offset = 0;
    for (int j = 0; j < (int)_countof(sizeNN); j++) {
        for (int i = 0; i < (int)_countof(sizeNX); i++) {
            if (i == prm->nnedi.nsize
                && j == get_cx_index(list_vpp_nnedi_nns, prm->nnedi.nns)) {
                weight1size_offset = weight1size_tsize;
            }
            weight1size_tsize += sizeNN[j] * (sizeNX[i] * sizeNY[i] + 1) * 4;
        }
    }

    std::vector<char> weight0f;
    weight0f.resize((((prm->nnedi.pre_screen & VPP_NNEDI_PRE_SCREEN_MODE) >= VPP_NNEDI_PRE_SCREEN_NEW) ? weight0sizenew : weight0size) * sizeofweight);
    if (prm->nnedi.precision == VPP_FP_PRECISION_FP32) {
        setWeight0<float>((float *)weight0f.data(), weights.get(), prm);
    } else {
        setWeight0<cl_half>((cl_half *)weight0f.data(), weights.get(), prm);
    }

    std::array<std::vector<char>, 2> weight1;
    for (int i = 0; i < 2; i++) {
        weight1[i].resize(weight1size * sizeofweight, 0);
        const float *ptrW = weights.get() + weight0size + weight0sizenew * 3 + weight1size_tsize * prm->nnedi.errortype + weight1size_offset + i * weight1size;
        if (prm->nnedi.precision == VPP_FP_PRECISION_FP32) {
            setWeight1<float>((float *)weight1[i].data(), ptrW, prm);
        } else {
            setWeight1<cl_half>((cl_half *)weight1[i].data(), ptrW, prm);
        }
    }
    m_weight0 = m_cl->copyDataToBuffer(weight0f.data(), weight0f.size());
    for (size_t i = 0; i < weight1.size(); i++) {
        m_weight1[i] = m_cl->copyDataToBuffer(weight1[i].data(), weight1[i].size());
    }
    return RGY_ERR_NONE;
}

template<typename TypeCalc> TypeCalc toWeight(float f);
template<> float toWeight<float>(float f) { return f; }
template<> cl_half toWeight<cl_half>(float f) { return (cl_half)float2half(f); }

template<typename TypeCalc>
void RGYFilterNnedi::setWeight0(TypeCalc *ptrDst, const float *ptrW, const std::shared_ptr<RGYFilterParamNnedi> prm) {
    if ((prm->nnedi.pre_screen & VPP_NNEDI_PRE_SCREEN_MODE) >= VPP_NNEDI_PRE_SCREEN_NEW) {
        auto index = [](int j, int k) {
            return ((k >> 3) << 5) + ((j & 3) << 3) + (k & 7);
        };

        const auto ptr_w = ptrW + weight0size + weight0sizenew * ((prm->nnedi.pre_screen & VPP_NNEDI_PRE_SCREEN_MODE) - VPP_NNEDI_PRE_SCREEN_NEW);
        double avg[4] = { 0.0, 0.0, 0.0, 0.0 };
        for (int j = 0; j < 4; j++) {
            double sum = 0.0;
            for (int k = 0; k < 64; k++) {
                sum += ptr_w[index(j, k)];
            }
            avg[j] = sum * (1.0 / 64.0);
        }
        const double halfinv = 1.0 / (((1 << 8) - 1) * 0.5);
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 64; k++) {
                //ptrDst[index(j, k)] = (float)((ptr_w[index(j, k)] - avg[j]) * halfinv);
                ptrDst[j*64+k] = toWeight<TypeCalc>((float)((ptr_w[index(j, k)] - avg[j]) * halfinv));
            }
        }
        for (int i = 0; i < 4; i++) {
            ptrDst[4*64+i] = toWeight<TypeCalc>(ptr_w[4*64+i]);
        }
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                ptrDst[4*65+j*4+k] = toWeight<TypeCalc>(ptr_w[4*65+ j + k*4]); //転置
            }
        }
        for (int i = 0; i < 4; i++) {
            ptrDst[4*65+4*4+i] = toWeight<TypeCalc>(ptr_w[4*65+4*4+i]);
        }
        //<<<<<< ここまでで通常(CPU版)の並びのデータが作成できた

        if (prm->nnedi.precision == VPP_FP_PRECISION_FP16) {
            //並べ替え
            std::vector<TypeCalc> tmp(ptrDst, ptrDst + weight0sizenew);
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 64; k++) {
                    int j2 = j / 4;
                    int j3 = j % 4;
                    ptrDst[(j2 * 64 + k) * 4 + j3] = tmp[j * 64 + k];
                }
            }
            for (int j = 0; j < 4; j++) {
                ptrDst[64*4 + j] = tmp[64*4 + j];
            }
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    int j2 = j / 4;
                    int j3 = j % 4;
                    ptrDst[65*4 + (j2 * 4 + k) * 4 + j3] = tmp[65*4 + j * 4 + k];
                }
            }
            for (int j = 0; j < 4; j++) {
                ptrDst[65*4+4*4 + j] = tmp[65*4+4*4 + j];
            }
        }
    } else {
        const auto ptr_w = ptrW;
        double avg[4] = { 0.0, 0.0, 0.0, 0.0 };
        for (int j = 0; j < 4; j++) {
            double sum = 0.0;
            for (int k = 0; k < 48; k++) {
                sum += ptr_w[j * 48 + k];
            }
            avg[j] = sum * (1.0 / 48.0);
        }
        const double halfinv = 1.0 / (((1 << 8) - 1) * 0.5);
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 48; k++) {
                ptrDst[j * 48 + k] = toWeight<TypeCalc>((float)((ptr_w[j * 48 + k] - avg[j]) * halfinv));
            }
        }
        for (int i = 4 * 48; i < 4*49; i++) {
            ptrDst[i] = toWeight<TypeCalc>(ptr_w[i]);
        }
        for (int i = 4 * 49; i < 4*49 + 4*4; i++) {
            ptrDst[i] = toWeight<TypeCalc>(ptr_w[i]);
        }
        for (int i = 4 * 49 + 4*4; i < 4*49 + 4*5; i++) {
            ptrDst[i] = toWeight<TypeCalc>(ptr_w[i]);
        }
        for (int i = 4*49 + 4*5; i < 4*49 + 4*5+ 4*8; i++) {
            ptrDst[i] = toWeight<TypeCalc>(ptr_w[i]);
        }
        for (int i = 4*49 + 4*5+ 4*8; i < 4*49 + 4*5+ 4*9; i++) {
            ptrDst[i] = toWeight<TypeCalc>(ptr_w[i]);
        }
        //<<<<<< ここまでで通常(CPU版)の並びのデータが作成できた

        if (prm->nnedi.precision == VPP_FP_PRECISION_FP16) {
            //並べ替え
            std::vector<TypeCalc> tmp(ptrDst, ptrDst + weight0size);
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 48; k++) {
                    int j2 = j / 4;
                    int j3 = j % 4;
                    ptrDst[(j2 * 48 + k) * 4 + j3] = tmp[j * 48 + k];
                }
            }
            for (int j = 0; j < 4; j++) {
                ptrDst[48*4 + j] = tmp[48*4 + j];
            }
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    int j2 = j / 4;
                    int j3 = j % 4;
                    ptrDst[49*4+(j2 * 4 + k) * 4 + j3] = tmp[49*4+j * 4 + k];
                }
            }
            for (int j = 0; j < 4; j++) {
                ptrDst[49*4+4*4 + j] = tmp[49*4+4*4 + j];
            }
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 8; k++) {
                    int j2 = j / 4;
                    int j3 = j % 4;
                    ptrDst[49*4+5*4 + (j2 * 8 + k) * 4 + j3] = tmp[49*4+5*4 + j * 8 + k];
                }
            }
            for (int j = 0; j < 4; j++) {
                ptrDst[49*4+5*4+8*4 + j] = tmp[49*4+5*4+8*4 + j];
            }
        }
    }
}

template<typename TypeCalc>
void RGYFilterNnedi::setWeight1(TypeCalc *ptrDst, const float *ptrW, const std::shared_ptr<RGYFilterParamNnedi> prm) {
    const int sizeNXY = sizeNX[prm->nnedi.nsize] * sizeNY[prm->nnedi.nsize];

    std::vector<double> mean0(prm->nnedi.nns * 2, 0.0);
    for (int j = 0; j < prm->nnedi.nns * 2; j++) {
        const float *ptr = ptrW + j * sizeNXY;
        mean0[j] = std::accumulate(ptr, ptr + sizeNXY, 0.0) / (double)sizeNXY;
    }

    const double inv_nns = 1.0 / (double)prm->nnedi.nns;
    std::vector<double> mean1(sizeNXY, 0.0);
    for (int j = 0; j < prm->nnedi.nns; j++) {
        for (int k = 0; k < sizeNXY; k++) {
            mean1[k] += (ptrW[j * sizeNXY + k] - mean0[j]) * inv_nns;
        }
    }

    const float *ptr = ptrW + prm->nnedi.nns * 2 * sizeNXY;
    const double mean2 = std::accumulate(ptr, ptr + prm->nnedi.nns, 0.0) * inv_nns;

    vector<float> buf(prm->nnedi.nns * 2 * sizeNXY);
    float max0 = 0.0f, max1 = 0.0f;
    for (int j = 0; j < prm->nnedi.nns * 2; j++) {
        for (int k = 0; k < sizeNXY; k++) {
            buf[j * sizeNXY + k] = (float)(ptrW[j * sizeNXY + k] - mean0[j] - (j < prm->nnedi.nns ? mean1[k] : 0.0));
            if (j < prm->nnedi.nns) {
                max0 = std::max(max0, buf[j * sizeNXY + k]);
            } else {
                max1 = std::max(max1, buf[j * sizeNXY + k]);
            }
        }
        //fp16の場合、オーバーフローを避けるため途中まで0～1の範囲で計算するので、offsetの部分には1/256が必要
        float scale = (prm->nnedi.precision == VPP_FP_PRECISION_FP16) ? 1.0f / 256.0f : 1.0f;
        ptrDst[prm->nnedi.nns * 2 * sizeNXY + j] = toWeight<TypeCalc>((ptrW[prm->nnedi.nns * 2 * sizeNXY + j] - (float)(j < prm->nnedi.nns ? mean2 : 0.0)) * scale);
    }
    for (int j = 0; j < prm->nnedi.nns * 2; j++) {
        for (int k = 0; k < sizeNXY; k++) {
            ptrDst[j * sizeNXY + k] = toWeight<TypeCalc>(buf[j * sizeNXY + k]);
        }
    }
    //<<<<<< ここまでで通常(CPU版)の並びのデータが作成できた

#if ENABLE_DP1_WEIGHT_ARRAY_OPT
    //最適化のため、本来の並びを変更する
    //[2][nns][nnxy] -> [nns/weight_loop_1][nnxy][weight_loop_1][2]
    vector<TypeCalc> tmp(prm->nnedi.nns * 2 * (sizeNXY + 1));
    memcpy(tmp.data(), ptrDst, sizeof(tmp[0]) * tmp.size());
    for (int j = 0; j < prm->nnedi.nns * 2; j++) {
        for (int k = 0; k < sizeNXY; k++) {
            const int j1 = j  / prm->nnedi.nns;
            const int j2 = j  % prm->nnedi.nns;
            const int j3 = j2 / weight_loop_1;
            const int w  = j2 % weight_loop_1;
            ptrDst[((j3 * sizeNXY + k) * weight_loop_1 + w) * 2 + j1] = tmp[j * sizeNXY + k];
        }
    }
    ptrDst += prm->nnedi.nns * 2 * sizeNXY;
    auto tmp2 = tmp.data() + prm->nnedi.nns * 2 * sizeNXY;
    for (int j = 0; j < prm->nnedi.nns; j++) {
        ptrDst[j * 2 + 0] = tmp2[j];
        ptrDst[j * 2 + 1] = tmp2[prm->nnedi.nns + j];
    }
#endif
}

RGY_ERR RGYFilterNnedi::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamNnedi>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->nnedi.precision == VPP_FP_PRECISION_AUTO) {
        prm->nnedi.precision = VPP_FP_PRECISION_FP32;
    }
    if (!m_clfp16support.has_value()) {
        m_clfp16support = RGYOpenCLDevice(m_cl->queue().devid()).checkExtension("cl_khr_fp16");
    }
    if (prm->nnedi.precision == VPP_FP_PRECISION_FP16 && !m_clfp16support.value_or(false)) {
        AddMessage(RGY_LOG_WARN, _T("fp16 not supported on this device, switching to fp32 mode.\n"));
        prm->nnedi.precision = VPP_FP_PRECISION_FP32;
    }
    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamNnedi>(m_param);
    if (!prmPrev
        || !m_nnedi_k0.get()
        || !m_nnedi_k1.get()
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
        || prmPrev->nnedi != prm->nnedi
        ) {
        if ((sts = checkParam(prm)) != RGY_ERR_NONE) {
            return sts;
        }
        if ((sts = initParams(prm)) != RGY_ERR_NONE) {
            return sts;
        }
        const auto sub_group_ext_avail = m_cl->platform()->checkSubGroupSupport(m_cl->queue().devid());
        std::string clversionRequired;
        switch (sub_group_ext_avail) {
        case RGYOpenCLSubGroupSupport::STD22:
        case RGYOpenCLSubGroupSupport::STD20KHR:
            clversionRequired = "-cl-std=CL2.0 "; break;
        case RGYOpenCLSubGroupSupport::INTEL_EXT:
        case RGYOpenCLSubGroupSupport::NONE:
        default:
            break;
        }
        enum class NNediCollectFlagMode {
            SubGroupAny = 0,
            LocalAtomicOr = 1,
            NoOptimization = 2
        };
        auto collect_flag_mode = NNediCollectFlagMode::NoOptimization;
        if (sub_group_ext_avail != RGYOpenCLSubGroupSupport::NONE) {
            collect_flag_mode = NNediCollectFlagMode::SubGroupAny;
        } else if (RGYOpenCLDevice(m_cl->queue().devid()).checkExtension("cl_khr_local_int32_extended_atomics")) { // atomic_or
            collect_flag_mode = NNediCollectFlagMode::LocalAtomicOr;
        }
        const int prescreen_new = ((prm->nnedi.pre_screen & VPP_NNEDI_PRE_SCREEN_MODE) == VPP_NNEDI_PRE_SCREEN_ORIGINAL) ? 0 : 1;
        const auto fields = make_array<NnediTargetField>(NNEDI_GEN_FIELD_TOP, NNEDI_GEN_FIELD_BOTTOM);
        m_nnedi_k0.set(m_cl->threadPool()->enqueue([cl = m_cl, log = m_pLog, prescreen_new, clversionRequired, prm]() {
            const auto nnedi_common_cl = getEmbeddedResourceStr(_T("RGY_FILTER_NNEDI_COMMON_CL"), _T("EXE_DATA"), cl->getModuleHandle());
            if (nnedi_common_cl.length() == 0) {
                log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("Failed to load RGY_FILTER_NNEDI_COMMON_CL."));
                return std::unique_ptr<RGYOpenCLProgram>();
            }
            auto nnedi_k0_cl = getEmbeddedResourceStr(_T("RGY_FILTER_NNEDI_K0_CL"), _T("EXE_DATA"), cl->getModuleHandle());
            if (nnedi_k0_cl.length() == 0) {
                log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("Failed to load RGY_FILTER_NNEDI_K1_CL."));
                return std::unique_ptr<RGYOpenCLProgram>();
            }
            auto pos = nnedi_k0_cl.find("#include \"rgy_filter_nnedi_common.cl\"");
            if (pos == std::string::npos) {
                log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to search #include \"rgy_filter_nnedi_common.cl\"\n"));
                return std::unique_ptr<RGYOpenCLProgram>();
            }
            const int wstep = prm->nnedi.precision == VPP_FP_PRECISION_FP16 ? 2 : 1; //half2なら2, floatなら1
            const int nnx = (prescreen_new) ? 16 : 12;
            const int nny = 4;
            const int nns = 4 / wstep; //half2の場合、nns方向を2つ格納できる
            nnedi_k0_cl = str_replace(nnedi_k0_cl, "#include \"rgy_filter_nnedi_common.cl\"", nnedi_common_cl);
            const auto options = clversionRequired + strsprintf(" "
                "-D TypePixel=%s -D TypePixel2=%s -D TypePixel4=%s -D bit_depth=%d -D TypeCalc=%s -D USE_FP16=%d "
                "-D nnx=%d -D nny=%d -D nnxy=%d -D nns=%d "
                "-D thread_y_loop=%d -D weight_loop=%d -D prescreen_new=%d "
                "-D ENABLE_DP1_WEIGHT_LOOP_UNROLL=%d -D ENABLE_DP1_WEIGHT_ARRAY_OPT=%d -D ENABLE_DP1_SHUFFLE_OPT=%d",
                RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
                RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort2" : "uchar2",
                RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort4" : "uchar4",
                RGY_CSP_BIT_DEPTH[prm->frameOut.csp],
                prm->nnedi.precision == VPP_FP_PRECISION_FP16 ? "half2" : "float",
                prm->nnedi.precision == VPP_FP_PRECISION_FP16 ? 1 : 0,
                nnx, nny, nnx * nny, nns,
                THREAD_Y_LOOP_K0,
                weight_loop_0,
                prescreen_new,
                ENABLE_DP1_WEIGHT_LOOP_UNROLL ? 1 : 0,
                ENABLE_DP1_WEIGHT_ARRAY_OPT ? 1 : 0,
                ENABLE_DP1_SHUFFLE_OPT ? 1 : 0
            );
            auto nnedi_k0 = cl->build(nnedi_k0_cl, options.c_str());
            if (!nnedi_k0) {
                log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to build RGY_FILTER_NNEDI_K0_CL(m_nnedi_k0)\n"));
                return std::unique_ptr<RGYOpenCLProgram>();
            }
            return nnedi_k0;
        }));
        m_nnedi_k1.set(m_cl->threadPool()->enqueue([cl = m_cl, log = m_pLog, clversionRequired, collect_flag_mode, prescreen_new, prm]() {
            const auto nnedi_common_cl = getEmbeddedResourceStr(_T("RGY_FILTER_NNEDI_COMMON_CL"), _T("EXE_DATA"), cl->getModuleHandle());
            if (nnedi_common_cl.length() == 0) {
                log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("Failed to load RGY_FILTER_NNEDI_COMMON_CL."));
                return std::unique_ptr<RGYOpenCLProgram>();
            }
            auto nnedi_k1_cl = getEmbeddedResourceStr(_T("RGY_FILTER_NNEDI_K1_CL"), _T("EXE_DATA"), cl->getModuleHandle());
            if (nnedi_k1_cl.length() == 0) {
                log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("Failed to load RGY_FILTER_NNEDI_K1_CL."));
                return std::unique_ptr<RGYOpenCLProgram>();
            }
            auto pos = nnedi_k1_cl.find("#include \"rgy_filter_nnedi_common.cl\"");
            if (pos == std::string::npos) {
                log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to search #include \"rgy_filter_nnedi_common.cl\"\n"));
                return std::unique_ptr<RGYOpenCLProgram>();
            }
            nnedi_k1_cl = str_replace(nnedi_k1_cl, "#include \"rgy_filter_nnedi_common.cl\"", nnedi_common_cl);
            const auto options = clversionRequired + strsprintf(" "
                "-D TypePixel=%s -D TypePixel2=%s -D TypePixel4=%s -D bit_depth=%d -D TypeCalc=%s -D USE_FP16=%d "
                "-D nnx=%d -D nny=%d -D nnxy=%d -D nns=%d "
                "-D thread_y_loop=%d -D weight_loop=%d -D prescreen_new=%d "
                "-D ENABLE_DP1_WEIGHT_LOOP_UNROLL=%d -D ENABLE_DP1_WEIGHT_ARRAY_OPT=%d -D ENABLE_DP1_SHUFFLE_OPT=%d "
                "-D COLLECT_FLAG_MODE=%d",
                RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort"  : "uchar",
                RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort2" : "uchar2",
                RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort4" : "uchar4",
                RGY_CSP_BIT_DEPTH[prm->frameOut.csp],
                prm->nnedi.precision == VPP_FP_PRECISION_FP16 ? "half2" : "float",
                prm->nnedi.precision == VPP_FP_PRECISION_FP16 ? 1 : 0,
                sizeNX[prm->nnedi.nsize], sizeNY[prm->nnedi.nsize], sizeNX[prm->nnedi.nsize] * sizeNY[prm->nnedi.nsize], prm->nnedi.nns,
                THREAD_Y_LOOP_K1,
                weight_loop_1,
                prescreen_new,
                ENABLE_DP1_WEIGHT_LOOP_UNROLL ? 1 : 0,
                ENABLE_DP1_WEIGHT_ARRAY_OPT ? 1 : 0,
                ENABLE_DP1_SHUFFLE_OPT ? 1 : 0,
                (int)collect_flag_mode
                );
            //options += "-fbin-exe -save-temps=F:\\temp\\nnedi_";
            auto nnedi_k1 = cl->build(nnedi_k1_cl, options.c_str());
            if (!nnedi_k1) {
                log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to build RGY_FILTER_NNEDI_K1_CL(m_nnedi_k1)\n"));
                return std::unique_ptr<RGYOpenCLProgram>();
            };
#if 0
            if (sub_group_ext_avail != RGYOpenCLSubGroupSupport::NONE) {
                auto getKernelSubGroupInfo = clGetKernelSubGroupInfo != nullptr ? clGetKernelSubGroupInfo : clGetKernelSubGroupInfoKHR;
                RGYWorkSize local(NNEDI_BLOCK_X, NNEDI_BLOCK_Y);
                const char *kernel_name = "kernel_compute_network1";
                size_t result;
                auto err = getKernelSubGroupInfo(m_nnedi_k1->kernel(kernel_name).get(), m_cl->platform()->dev(0), CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR,
                    sizeof(local.w[0]) * 2, &local.w[0], sizeof(result), &result, nullptr);
                if (err_cl_to_rgy(err) != RGY_ERR_NONE) {
                    ///////////<<<<< 未実装 >>>>>>>>>>>;
                }
            }
#endif
            return nnedi_k1;
        }));
        m_cl->requestCSPCopy(prm->frameOut, prm->frameIn);
    }
    if (prm->nnedi.isbob()) {
        pParam->baseFps *= 2;
        m_pathThrough &= (~(FILTER_PATHTHROUGH_TIMESTAMP));
    }

    auto err = AllocFrameBuf(prm->frameOut, prm->nnedi.isbob() ? 2 : 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    m_pathThrough &= (~(FILTER_PATHTHROUGH_PICSTRUCT));

    //コピーを保存
    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterNnedi::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }
    auto prm = std::dynamic_pointer_cast<RGYFilterParamNnedi>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
        if (prm->nnedi.isbob()) {
            pOutFrame = m_frameBuf[1].get();
            ppOutputFrames[1] = &pOutFrame->frame;
            ppOutputFrames[1]->picstruct = pInputFrame->picstruct;
            *pOutputFrameNum = 2;
        }
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    //if (interlaced(*pInputFrame)) {
    //    return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
    //}
    if (!m_nnedi_k0.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build RGY_FILTER_NNEDI_K0_CL(m_nnedi_k0)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    if (!m_nnedi_k1.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build RGY_FILTER_NNEDI_K1_CL(m_nnedi_k1)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != RGYCLMemcpyD2D) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    NnediTargetField targetField = NNEDI_GEN_FIELD_UNKNOWN;
    if (   prm->nnedi.field == VPP_NNEDI_FIELD_USE_AUTO
        || prm->nnedi.field == VPP_NNEDI_FIELD_BOB_AUTO) {
        if ((pInputFrame->picstruct & RGY_PICSTRUCT_INTERLACED) == 0) {
            m_cl->copyFrame(ppOutputFrames[0], pInputFrame);
            if (prm->nnedi.isbob()) {
                m_cl->copyFrame(ppOutputFrames[1], pInputFrame);
                setBobTimestamp(pInputFrame, ppOutputFrames);
            }
            return RGY_ERR_NONE;
        } else if ((pInputFrame->picstruct & RGY_PICSTRUCT_FRAME_TFF) == RGY_PICSTRUCT_FRAME_TFF) {
            targetField = NNEDI_GEN_FIELD_BOTTOM;
        } else if ((pInputFrame->picstruct & RGY_PICSTRUCT_FRAME_BFF) == RGY_PICSTRUCT_FRAME_BFF) {
            targetField = NNEDI_GEN_FIELD_TOP;
        }
    } else if (prm->nnedi.field == VPP_NNEDI_FIELD_USE_TOP
        || prm->nnedi.field == VPP_NNEDI_FIELD_BOB_TOP_BOTTOM) {
        targetField = NNEDI_GEN_FIELD_BOTTOM;
    } else if (prm->nnedi.field == VPP_NNEDI_FIELD_USE_BOTTOM
        || prm->nnedi.field == VPP_NNEDI_FIELD_BOB_BOTTOM_TOP) {
        targetField = NNEDI_GEN_FIELD_TOP;
    } else {
        AddMessage(RGY_LOG_ERROR, _T("Not implemented yet.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto err = procFrame(ppOutputFrames[0], pInputFrame, targetField, queue, wait_events, event);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at procFrame(0): %s.\n"), get_err_mes(err));
        return err;
    }
    ppOutputFrames[0]->picstruct = RGY_PICSTRUCT_FRAME;

    if (prm->nnedi.isbob()) {
        targetField = (targetField == NNEDI_GEN_FIELD_BOTTOM) ? NNEDI_GEN_FIELD_TOP : NNEDI_GEN_FIELD_BOTTOM;
        err = procFrame(ppOutputFrames[1], pInputFrame, targetField, queue, wait_events, event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at procFrame(1): %s.\n"), get_err_mes(err));
            return err;
        }
        setBobTimestamp(pInputFrame, ppOutputFrames);
    }

    return sts;
}

void RGYFilterNnedi::setBobTimestamp(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamNnedi>(m_param);

    auto frameDuration = pInputFrame->duration;
    if (frameDuration == 0) {
        frameDuration = (decltype(frameDuration))((prm->timebase.inv() / prm->baseFps * 2).qdouble() + 0.5);
    }

    ppOutputFrames[1]->picstruct = RGY_PICSTRUCT_FRAME;
    ppOutputFrames[0]->timestamp = pInputFrame->timestamp;
    ppOutputFrames[0]->duration = (frameDuration + 1) / 2;
    ppOutputFrames[1]->timestamp = ppOutputFrames[0]->timestamp + ppOutputFrames[0]->duration;
    ppOutputFrames[1]->duration = frameDuration - ppOutputFrames[0]->duration;
    ppOutputFrames[0]->inputFrameId = pInputFrame->inputFrameId;
    ppOutputFrames[1]->inputFrameId = pInputFrame->inputFrameId;
}

void RGYFilterNnedi::close() {
    m_frameBuf.clear();
    m_nnedi_k0.clear();
    m_nnedi_k1.clear();
    m_cl.reset();
}
