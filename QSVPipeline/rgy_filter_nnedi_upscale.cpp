// -----------------------------------------------------------------------------------------
//     QSVEnc/VCEEnc by rigaya
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
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#include "rgy_filter_nnedi_upscale.h"
#include "rgy_filter_transform.h"
#include <climits>

RGYFilterNnediUpscale::RGYFilterNnediUpscale(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context), m_shift(), m_passV(), m_passH(), m_transposeToH(), m_transposeBack() {
    m_name = _T("nnedi-upscale");
}

RGYFilterNnediUpscale::~RGYFilterNnediUpscale() {
    close();
}

RGY_ERR RGYFilterNnediUpscale::makeNnedi(std::unique_ptr<RGYFilterNnedi>& target,
    const RGYFrameInfo& frameIn, const RGYFilterParamNnediUpscale *prm, const TCHAR *label) {
    auto param = std::make_shared<RGYFilterParamNnedi>();
    param->frameIn = frameIn;
    param->frameOut = frameIn;
    param->baseFps = prm->baseFps;
    param->timebase = prm->timebase;
    param->bOutOverwrite = false;
    param->nnedi.enable = true;
    // プログレッシブ画像の既存行を上フィールドとして扱い、空いた行を補間する。
    param->nnedi.field = VPP_NNEDI_FIELD_TOP;
    param->nnedi.doubleHeight = true;
    // doubleHeightでは全プレーンを同じ倍率で処理する。
    param->nnedi.processPlane = { true, true, true, true };
    param->nnedi.nsize = prm->nnediUpscale.nnedi.nsize;
    param->nnedi.nns = prm->nnediUpscale.nnedi.nns;
    param->nnedi.quality = prm->nnediUpscale.nnedi.quality;
    param->nnedi.prescreen = prm->nnediUpscale.nnedi.prescreen;
    param->nnedi.errortype = prm->nnediUpscale.nnedi.errortype;
    param->nnedi.clamp = prm->nnediUpscale.nnedi.clamp;
    param->nnedi.weightfile = prm->nnediUpscale.nnedi.weightfile;
    target = std::make_unique<RGYFilterNnedi>(m_cl);
    auto err = target->init(param, m_pLog);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialise the %s doubling pass: %s.\n"),
            label, get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterNnediUpscale::makeTranspose(std::unique_ptr<RGYFilterTransform>& target,
    const RGYFrameInfo& frameIn, const RGYFilterParamNnediUpscale *prm, const TCHAR *label) {
    auto param = std::make_shared<RGYFilterParamTransform>();
    param->frameIn = frameIn;
    param->frameOut = frameIn;
    param->baseFps = prm->baseFps;
    param->bOutOverwrite = false;
    param->trans.transpose = true;
    target = std::make_unique<RGYFilterTransform>(m_cl);
    const auto err = target->init(param, m_pLog);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialise the %s transpose: %s.\n"),
            label, get_err_mes(err));
    }
    return err;
}

RGY_ERR RGYFilterNnediUpscale::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamNnediUpscale>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameIn.height <= 0 || prm->frameIn.width <= 0
        || prm->frameIn.height > INT_MAX / 2 || prm->frameIn.width > INT_MAX / 2) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    // 4:2:2は転置すると4:4:0相当になり、同じCSPの中間フレームでは表現できない。
    if (RGY_CSP_CHROMA_FORMAT[prm->frameIn.csp] == RGY_CHROMAFMT_YUV422) {
        AddMessage(RGY_LOG_ERROR, _T("nnedi-upscale does not support 4:2:2 input.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->frameIn.picstruct & RGY_PICSTRUCT_INTERLACED) {
        AddMessage(RGY_LOG_ERROR, _T("nnedi-upscale requires progressive input; deinterlace first.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    // 各軸をフィールドとして2分するため、縦横とも偶数を要求する。
    if ((prm->frameIn.width & 1) || (prm->frameIn.height & 1)) {
        AddMessage(RGY_LOG_ERROR, _T("nnedi-upscale needs even width and height (got %dx%d).\n"),
            prm->frameIn.width, prm->frameIn.height);
        return RGY_ERR_INVALID_PARAM;
    }

    const auto W = prm->frameIn.width;
    const auto H = prm->frameIn.height;

    auto mid = prm->frameIn;
    mid.width = W;
    mid.height = H * 2;

    auto midT = prm->frameIn;
    midT.width = H * 2;
    midT.height = W;

    auto midH = prm->frameIn;
    midH.width = H * 2;
    midH.height = W * 2;

    sts = makeNnedi(m_passV, prm->frameIn, prm.get(), _T("vertical"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = makeTranspose(m_transposeToH, mid, prm.get(), _T("first"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = makeNnedi(m_passH, midT, prm.get(), _T("horizontal"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = makeTranspose(m_transposeBack, midH, prm.get(), _T("second"));
    if (sts != RGY_ERR_NONE) return sts;

    const auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamNnediUpscale>(m_param);
    const bool rebuild = !m_shift.get() || !prmPrev
        || prmPrev->frameIn.csp != prm->frameIn.csp;
    if (rebuild) {
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d",
            RGY_CSP_BIT_DEPTH[prm->frameIn.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameIn.csp]);
        m_shift.set(m_cl->buildResourceAsync(_T("RGY_FILTER_NNEDI_UPSCALE_CL"), _T("EXE_DATA"), options.c_str()));
    }

    prm->frameOut = prm->frameIn;
    prm->frameOut.width = W * 2;
    prm->frameOut.height = H * 2;

    auto err = AllocFrameBuf(prm->frameOut, 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterNnediUpscale::halfPixel(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
    RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamNnediUpscale>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int nPlanes = RGY_CSP_PLANES[pSrc->csp];
    for (int i = 0; i < nPlanes; i++) {
        auto planeDst = getPlane(pDst, (RGY_PLANE)i);
        auto planeSrc = getPlane(pSrc, (RGY_PLANE)i);
        const char *kernel_name = prm->nnediUpscale.shiftCubic
            ? "kernel_nnedi_upscale_shift" : "kernel_nnedi_upscale_shift_linear";
        RGYWorkSize local(32, 8);
        RGYWorkSize global(planeSrc.width, planeSrc.height);
        auto err = m_shift.get()->kernel(kernel_name).config(queue, local, global,
            std::vector<RGYOpenCLEvent>(), (i == nPlanes - 1) ? event : nullptr).launch(
            (cl_mem)planeDst.ptr[0], planeDst.pitch[0],
            (cl_mem)planeSrc.ptr[0], planeSrc.pitch[0], planeSrc.width, planeSrc.height);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"), char_to_tstring(kernel_name).c_str(), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterNnediUpscale::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }
    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
    }
    ppOutputFrames[0]->picstruct = RGY_PICSTRUCT_FRAME;
    if (!m_shift.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build RGY_FILTER_NNEDI_UPSCALE_CL\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != RGYCLMemcpyD2D) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    // 拡大処理では入力のフィールド情報に関係なく各行をプログレッシブとして扱う。
    auto srcFrame = *pInputFrame;
    srcFrame.picstruct = RGY_PICSTRUCT_FRAME;

    int count = 0;
    RGYFrameInfo *outV[2] = { nullptr, nullptr };
    sts = m_passV->filter(&srcFrame, outV, &count, queue, wait_events, nullptr);
    if (sts != RGY_ERR_NONE || count != 1 || outV[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("the vertical doubling failed: %s.\n"), get_err_mes(sts));
        return sts != RGY_ERR_NONE ? sts : RGY_ERR_UNKNOWN;
    }
    count = 0;
    RGYFrameInfo *outTransposedV[1] = { nullptr };
    sts = m_transposeToH->filter(outV[0], outTransposedV, &count, queue, {}, nullptr);
    if (sts != RGY_ERR_NONE || count != 1 || outTransposedV[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("the first transpose failed: %s.\n"), get_err_mes(sts));
        return sts != RGY_ERR_NONE ? sts : RGY_ERR_UNKNOWN;
    }

    auto midFrame = *outTransposedV[0];
    midFrame.picstruct = RGY_PICSTRUCT_FRAME;
    count = 0;
    RGYFrameInfo *outH[2] = { nullptr, nullptr };
    sts = m_passH->filter(&midFrame, outH, &count, queue, {}, nullptr);
    if (sts != RGY_ERR_NONE || count != 1 || outH[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("the horizontal doubling failed: %s.\n"), get_err_mes(sts));
        return sts != RGY_ERR_NONE ? sts : RGY_ERR_UNKNOWN;
    }
    count = 0;
    RGYFrameInfo *outTransposedH[1] = { nullptr };
    sts = m_transposeBack->filter(outH[0], outTransposedH, &count, queue, {}, nullptr);
    if (sts != RGY_ERR_NONE || count != 1 || outTransposedH[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("the second transpose failed: %s.\n"), get_err_mes(sts));
        return sts != RGY_ERR_NONE ? sts : RGY_ERR_UNKNOWN;
    }

    // 各パスが各軸に残す半画素ずれを最後にまとめて補正する。
    sts = halfPixel(ppOutputFrames[0], outTransposedH[0], queue, event);
    if (sts != RGY_ERR_NONE) return sts;

    copyFramePropWithoutRes(ppOutputFrames[0], pInputFrame);
    ppOutputFrames[0]->picstruct = RGY_PICSTRUCT_FRAME;
    return sts;
}

void RGYFilterNnediUpscale::close() {
    m_passV.reset();
    m_passH.reset();
    m_transposeToH.reset();
    m_transposeBack.reset();
    m_frameBuf.clear();
    m_shift.clear();
    m_cl.reset();
}

