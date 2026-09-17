// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2019 rigaya
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
#include <cmath>
#include <numeric>
#include "rgy_filesystem.h"
#include "cpu_info.h"
#include "rgy_avutil.h"
#include "rgy_filter_ssim.h"
#if ENCODER_QSV
#include "qsv_mfx_dec.h"
#include "qsv_pipeline_ctrl.h"
#endif
#if ENCODER_VCEENC
#include "vce_util.h"
#include "VideoDecoderUVD.h"

const TCHAR *AMFRetString(AMF_RESULT ret);

#define VCEAMF(x) x
#else
#define VCEAMF(x)
#endif

static const int SSIM_BLOCK_X = 32;
static const int SSIM_BLOCK_Y = 8;

static double ssim_db(double ssim, double weight) {
    return 10.0 * log10(weight / (weight - ssim));
}

static double get_psnr(double mse, uint64_t nb_frames, int max) {
    return 10.0 * log10((max * max) / (mse / nb_frames));
}

RGYFilterParamSsim::RGYFilterParamSsim() : metric(), deviceId(0), bitDepth(8), input(), streamtimebase(), threadParam()
#if ENCODER_VCEENC
, factory(nullptr), trace(nullptr), context()
#endif
#if ENCODER_QSV
, mfxDEC()
#endif
{

};
RGYFilterParamSsim::~RGYFilterParamSsim() {};

tstring RGYFilterParamSsim::print() const {
    tstring str;
    if (metric.ssim) str += _T("ssim ");
    if (metric.psnr) str += _T("psnr ");
    if (metric.vmaf.enable) str += _T("vmaf ");
    if (metric.vshipSsimu2.enable) str += _T("vship-ssimulacra2 ");
    if (metric.vshipButteraugli.enable) str += _T("vship-butteraugli ");
    if (metric.vshipCvvdp.enable) str += _T("vship-cvvdp ");
    return str;
}

RGYFilterSsim::RGYFilterSsim(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_decodeStarted(false),
    m_deviceId(0),
    m_thread(),
    m_mtx(),
    m_abort(false),
    m_dec_flush(false),
    m_inputOriginal(0),
    m_inputEnc(0),
    m_input(),
    m_unused(),
    m_inputReady(),
#if ENCODER_VCEENC
    m_trace(nullptr),
    m_factory(nullptr),
    m_context(),
    m_decoder(),
#endif
#if ENCODER_QSV
    m_encBitstream(),
    m_encBitstreamUnused(),
    m_mfxDEC(),
    m_taskDec(),
#if ENABLE_QSV_OPENCL_INPUT_COPY
    m_inputCopy(),
#endif
    m_surfVppInInterop(),
#endif
    m_cropOrg(),
    m_cropDec(),
    m_decFrameCopy(),
    m_tmpSsim(),
    m_tmpPsnr(),
    m_cropEvent(),
    m_queueCrop(),
    m_queueCalcSsim(),
    m_queueCalcPsnr(),
    m_planeCoef(),
    m_ssimTotalPlane(),
    m_ssimTotal(0.0),
    m_psnrTotalPlane(),
    m_psnrTotal(0.0),
    m_frames(0),
    m_finishCalled(false),
    m_finishResult(RGY_ERR_NONE),
#if ENABLE_VMAF || ENABLE_LIBVSHIP
    m_metricSlots(),
    m_metricReady(),
    m_metricFree(),
    m_metricThread(),
    m_metricMutex(),
    m_metricReadyCv(),
    m_metricFreeCv(),
    m_metricInitCv(),
    m_metricInputFin(false),
    m_metricStop(false),
    m_metricWorkerReady(false),
    m_metricWorkerDone(false),
    m_metricFinishCalled(false),
    m_metricNextIndex(0),
    m_metricError(RGY_ERR_NONE),
    m_metricFinishResult(RGY_ERR_NONE),
#if ENABLE_VMAF
    m_libvmaf(),
    m_vmafContext(nullptr),
    m_vmafModel(nullptr),
    m_vmafModelCollection(nullptr),
    m_vmafScore(0.0),
    m_vmafFrames(0),
#endif
#if ENABLE_LIBVSHIP
    m_libvship(),
    m_vshipSsimu2(),
    m_vshipButteraugli(),
    m_vshipCvvdp(),
    m_vshipSsimu2Initialized(false),
    m_vshipButteraugliInitialized(false),
    m_vshipCvvdpInitialized(false),
    m_vshipSsimu2Total(0.0),
    m_vshipSsimu2Frames(0),
    m_vshipSsimu2Scores(),
    m_vshipButteraugliNormQ(0.0),
    m_vshipButteraugliNorm3(0.0),
    m_vshipButteraugliNormInf(0.0),
    m_vshipButteraugliFrames(0),
    m_vshipCvvdpScore(0.0),
    m_vshipCvvdpFrames(0),
#endif
#endif
    m_kernel() {
    m_name = _T("ssim/psnr");
}

RGYFilterSsim::~RGYFilterSsim() {
    close();
}

RGY_ERR RGYFilterSsim::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;

    auto prm = std::dynamic_pointer_cast<RGYFilterParamSsim>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (RGY_CSP_CHROMA_FORMAT[pParam->frameIn.csp] != RGY_CHROMAFMT_YUV420 && RGY_CSP_CHROMA_FORMAT[pParam->frameIn.csp] != RGY_CHROMAFMT_YUV444) {
        AddMessage(RGY_LOG_ERROR, _T("this filter does not support csp %s.\n"), RGY_CSP_NAMES[pParam->frameIn.csp]);
        return RGY_ERR_UNSUPPORTED;
    }

    m_deviceId = prm->deviceId;
    m_finishCalled = false;
    m_finishResult = RGY_ERR_NONE;
    m_cropOrg.reset();
    m_cropDec.reset();
    if (prm->metric.vmaf.enable) {
        if (prm->metric.vmaf.model.empty() || prm->metric.vmaf.threads < 0 || prm->metric.vmaf.subsample < 1) {
            AddMessage(RGY_LOG_ERROR, _T("Invalid VMAF parameters.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
    }
    if (prm->metric.vshipButteraugli.enable
        && (prm->metric.vshipButteraugli.Qnorm <= 0 || !std::isfinite(prm->metric.vshipButteraugli.intensity_multiplier) || prm->metric.vshipButteraugli.intensity_multiplier <= 0.0f)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid Butteraugli parameters.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->metric.vshipCvvdp.enable
        && (prm->metric.vshipCvvdp.model.empty() || prm->baseFps.n() <= 0 || prm->baseFps.d() <= 0)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid CVVDP parameters.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->frameOut.csp == RGY_CSP_NV12) {
        pParam->frameOut.csp = RGY_CSP_YV12;
    } else if (pParam->frameOut.csp == RGY_CSP_P010) {
        if (prm->bitDepth <= 8) {
            AddMessage(RGY_LOG_ERROR, _T("Invalid bit depth.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        switch (prm->bitDepth) {
        case 10: pParam->frameOut.csp = RGY_CSP_YV12_10; break;
        case 12: pParam->frameOut.csp = RGY_CSP_YV12_12; break;
        case 14: pParam->frameOut.csp = RGY_CSP_YV12_14; break;
        case 16: pParam->frameOut.csp = RGY_CSP_YV12_16; break;
        default:
            AddMessage(RGY_LOG_ERROR, _T("Invalid bit depth.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
    }
    {
        unique_ptr<RGYFilterCspCrop> filterCrop(new RGYFilterCspCrop(m_cl));
        shared_ptr<RGYFilterParamCrop> paramCrop(new RGYFilterParamCrop());
        paramCrop->frameIn = pParam->frameIn;
        paramCrop->frameOut = pParam->frameOut;
        paramCrop->baseFps = pParam->baseFps;
#if ENCODER_QSV
        paramCrop->frameIn.mem_type = RGY_MEM_TYPE_GPU_IMAGE_NORMALIZED;
#elif ENCODER_VCEENC
        paramCrop->frameIn.mem_type = RGY_MEM_TYPE_GPU_IMAGE;
#endif
        paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
        paramCrop->bOutOverwrite = false;
        sts = filterCrop->init(paramCrop, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_cropDec = std::move(filterCrop);
        AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_cropDec->GetInputMessage().c_str());
        pParam->frameOut = paramCrop->frameOut;
    }
    AddMessage(RGY_LOG_DEBUG, _T("ssim original format %s -> %s.\n"), RGY_CSP_NAMES[pParam->frameIn.csp], RGY_CSP_NAMES[pParam->frameOut.csp]);

    {
        int elemSum = 0;
        for (size_t i = 0; i < m_ssimTotalPlane.size(); i++) {
            const auto plane = getPlane(&pParam->frameOut, (RGY_PLANE)i);
            elemSum += plane.width * plane.height;
        }
        for (size_t i = 0; i < m_ssimTotalPlane.size(); i++) {
            const auto plane = getPlane(&pParam->frameOut, (RGY_PLANE)i);
            m_planeCoef[i] = (double)(plane.width * plane.height) / elemSum;
            AddMessage(RGY_LOG_DEBUG, _T("Plane coef : %f\n"), m_planeCoef[i]);
        }
    }
    //SSIM
    for (size_t i = 0; i < m_ssimTotalPlane.size(); i++) {
        m_ssimTotalPlane[i] = 0.0;
    }
    m_ssimTotal = 0.0;
    //PSNR
    for (size_t i = 0; i < m_psnrTotalPlane.size(); i++) {
        m_psnrTotalPlane[i] = 0.0;
    }
    m_psnrTotal = 0.0;
#if ENCODER_VCEENC
    m_context = prm->context;
    m_factory = prm->factory;
    m_trace = prm->trace;
#endif //#if ENCODER_VCEENC
#if ENCODER_QSV
    m_mfxDEC = std::move(prm->mfxDEC);
    if ((sts = m_mfxDEC->InitMFXSession()) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed init session for hw decoder.\n"));
        return sts;
    }
    m_encBitstream.init(256, 30, 0);
    m_encBitstreamUnused.init(256);
#endif //#if ENCODER_QSV

    setFilterInfo(pParam->print() + _T("(") + RGY_CSP_NAMES[pParam->frameOut.csp] + _T(")"));
    m_param = pParam;
    return sts;
}

RGY_ERR RGYFilterSsim::initDecode(const RGYBitstream *bitstream) {
    AddMessage(RGY_LOG_DEBUG, _T("initDecode() with bitstream size: %d.\n"), (int)bitstream->size());

    auto prm = std::dynamic_pointer_cast<RGYFilterParamSsim>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    int ret = 0;
    const auto avcodecID = getAVCodecId(prm->input.codec);
    const auto codec = avcodec_find_decoder(avcodecID);
    if (codec == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("failed to find decoder for codec %s.\n"), CodecToStr(prm->input.codec).c_str());
        return RGY_ERR_NULL_PTR;
    }
    auto codecCtx = std::unique_ptr<AVCodecContext, RGYAVDeleter<AVCodecContext>>(avcodec_alloc_context3(codec), RGYAVDeleter<AVCodecContext>(avcodec_free_context));
    if (0 > (ret = avcodec_open2(codecCtx.get(), codec, nullptr))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open codec %s: %s.\n"), char_to_tstring(avcodec_get_name(avcodecID)).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_NULL_PTR;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Opened decoder for codec %s\n"), char_to_tstring(avcodec_get_name(avcodecID)).c_str());

    const char *bsf_name = "extract_extradata";
    const auto bsf = av_bsf_get_by_name(bsf_name);
    if (bsf == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("failed to bsf %s.\n"), char_to_tstring(bsf_name).c_str());
        return RGY_ERR_NULL_PTR;
    }
    AVBSFContext *bsfctmp = nullptr;
    if (0 > (ret = av_bsf_alloc(bsf, &bsfctmp))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for %s: %s.\n"), bsf_name, qsv_av_err2str(ret).c_str());
        return RGY_ERR_NULL_PTR;
    }
    unique_ptr<AVBSFContext, RGYAVDeleter<AVBSFContext>> bsfc(bsfctmp, RGYAVDeleter<AVBSFContext>(av_bsf_free));
    bsfctmp = nullptr;

    unique_ptr<AVCodecParameters, RGYAVDeleter<AVCodecParameters>> codecpar(avcodec_parameters_alloc(), RGYAVDeleter<AVCodecParameters>(avcodec_parameters_free));
    if (0 > (ret = avcodec_parameters_from_context(codecpar.get(), codecCtx.get()))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to get codec parameter for %s: %s.\n"), bsf_name, qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    if (0 > (ret = avcodec_parameters_copy(bsfc->par_in, codecpar.get()))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy parameter for %s: %s.\n"), bsf_name, qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    if (0 > (ret = av_bsf_init(bsfc.get()))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to init %s: %s.\n"), bsf_name, qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Initialized bsf %s\n"), bsf_name);

    AVPacket pkt;
    av_new_packet(&pkt, (int)bitstream->size());
    memcpy(pkt.data, bitstream->data(), (int)bitstream->size());
    if (0 > (ret = av_bsf_send_packet(bsfc.get(), &pkt))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to send packet to %s bitstream filter: %s.\n"),
            char_to_tstring(bsfc->filter->name).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    ret = av_bsf_receive_packet(bsfc.get(), &pkt);
    if (ret == AVERROR(EAGAIN)) {
        return RGY_ERR_NONE;
    } else if ((ret < 0 && ret != AVERROR_EOF) || pkt.size < 0) {
        AddMessage(RGY_LOG_ERROR, _T("failed to run %s bitstream filter: %s.\n"),
            char_to_tstring(bsfc->filter->name).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_UNKNOWN;
    }
    std::remove_pointer<RGYArgN<2U, decltype(av_packet_get_side_data)>::type>::type side_data_size = 0;
    auto side_data = av_packet_get_side_data(&pkt, AV_PKT_DATA_NEW_EXTRADATA, &side_data_size);
    if (side_data) {
        prm->input.codecExtra = malloc(side_data_size);
        prm->input.codecExtraSize = (decltype(prm->input.codecExtraSize))side_data_size;
        memcpy(prm->input.codecExtra, side_data, side_data_size);
        AddMessage(RGY_LOG_DEBUG, _T("Found extradata of codec %s: size %d\n"), char_to_tstring(avcodec_get_name(avcodecID)).c_str(), side_data_size);
    }
    av_packet_unref(&pkt);

    // QSVでは別スレッドで行うと、デコードでエラーが発生したり、黙って異常終了したり、SSIMの計算結果が安定しない
    // そのため、同一スレッド内で処理するよう変更する
    if (false) {
        //比較用のスレッドの開始
        m_thread = std::thread(&RGYFilterSsim::thread_func, this, prm->threadParam);
        AddMessage(RGY_LOG_DEBUG, _T("Started ssim/psnr calculation thread.\n"));

        //デコードの開始を待つ必要がある
        while (m_thread.joinable() && !m_decodeStarted) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } else {
        //シングルスレッド動作時
        auto sts = init_cl_resources();
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        m_decodeStarted = true;
    }

    AddMessage(RGY_LOG_DEBUG, _T("initDecode(): fin.\n"));
    return (m_decodeStarted) ? RGY_ERR_NONE : RGY_ERR_UNKNOWN;
}

RGY_ERR RGYFilterSsim::init_cl_resources() {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSsim>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    VCEAMF(amf::AMFContext::AMFOpenCLLocker locker(m_context));
    m_queueCrop = m_cl->createQueue(m_cl->queue().devid(), m_cl->queue().getProperties());
    if (prm->metric.ssim) {
        for (auto& q : m_queueCalcSsim) {
            q = m_cl->createQueue(m_cl->queue().devid(), m_cl->queue().getProperties());
        }
    }
    if (prm->metric.psnr) {
        for (auto &q : m_queueCalcPsnr) {
            q = m_cl->createQueue(m_cl->queue().devid(), m_cl->queue().getProperties());
        }
    }
    if (prm->metric.ssim || prm->metric.psnr) {
        if (auto err = build_kernel(m_param->frameOut.csp); err != RGY_ERR_NONE) {
            return err;
        }
    }
#if ENCODER_VCEENC
    auto codec_uvd_name = codec_rgy_to_dec(prm->input.codec);
    if (codec_uvd_name == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("Input codec \"%s\" not supported.\n"), CodecToStr(prm->input.codec).c_str());
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->input.codec == RGY_CODEC_HEVC && prm->input.csp == RGY_CSP_P010) {
        codec_uvd_name = AMFVideoDecoderHW_H265_MAIN10;
    }
    AddMessage(RGY_LOG_DEBUG, _T("decoder: use codec \"%s\".\n"), wstring_to_tstring(codec_uvd_name).c_str());
    auto res = m_factory->CreateComponent(m_context, codec_uvd_name, &m_decoder);
    if (res != AMF_OK) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to create decoder context: %s\n"), AMFRetString(res));
        return err_to_rgy(res);
    }
    AddMessage(RGY_LOG_DEBUG, _T("created decoder context.\n"));

    if (AMF_OK != (res = m_decoder->SetProperty(AMF_TIMESTAMP_MODE, amf_int64(AMF_TS_PRESENTATION)))) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to set deocder: %s\n"), AMFRetString(res));
        return err_to_rgy(res);
    }

    //AMF_VIDEO_DECODER_SURFACE_COPYを使用すると、pre-analysis使用時などに発生するSubmitInput時のAMF_DECODER_NO_FREE_SURFACESを回避できる
    //しかし、メモリ確保エラーが発生することがある(AMF_DIRECTX_FAIL)
    //そこで、AMF_VIDEO_DECODER_SURFACE_COPYは使用せず、QueryOutput後、明示的にsurface->Duplicateを行って同様の挙動を再現する
    //AV1デコードでは、これを有効にしないとAMF_DECODER_NO_FREE_SURFACESで止まってしまうことがわかったので、再度有効にする
    m_decoder->SetProperty(AMF_VIDEO_DECODER_SURFACE_COPY, true);

    amf::AMFBufferPtr buffer;
    m_context->AllocBuffer(amf::AMF_MEMORY_HOST, prm->input.codecExtraSize, &buffer);

    memcpy(buffer->GetNative(), prm->input.codecExtra, prm->input.codecExtraSize);
    m_decoder->SetProperty(AMF_VIDEO_DECODER_EXTRADATA, amf::AMFVariant(buffer));

    AddMessage(RGY_LOG_DEBUG, _T("initialize decoder: %dx%d, %s.\n"),
        prm->input.srcWidth, prm->input.srcHeight,
        wstring_to_tstring(m_trace->SurfaceGetFormatName(csp_rgy_to_enc(prm->input.csp))).c_str());
    if (AMF_OK != (res = m_decoder->Init(csp_rgy_to_enc(prm->input.csp), prm->input.srcWidth, prm->input.srcHeight))) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to init decoder: %s\n"), AMFRetString(res));
        return err_to_rgy(res);
    }
#endif
#if ENCODER_QSV
    RGYBitstream header = RGYBitstreamInit();
    header.copy((const uint8_t *)prm->input.codecExtra, prm->input.codecExtraSize);
    auto sts = m_mfxDEC->SetParam(prm->input.codec, header, prm->input);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to set param to hw decoder.\n"));
        return sts;
    }

    m_taskDec = std::make_unique<PipelineTaskMFXDecode>(m_mfxDEC->GetVideoSessionPtr(), 1, m_mfxDEC->mfxdec(), m_mfxDEC->mfxparams(), m_mfxDEC->skipAV1C(), -1, nullptr, m_mfxDEC->mfxver(), m_pLog);
    auto allocRequest = m_taskDec->requiredSurfOut();
    if (!allocRequest.has_value()) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to get required surface num for hw decoder.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    allocRequest.value().AllocId            = m_mfxDEC->allocator()->getExtAllocCounts();
    allocRequest.value().NumFrameSuggested += (mfxU16)m_taskDec->outputMaxQueueSize();
    allocRequest.value().NumFrameMin       += (mfxU16)m_taskDec->outputMaxQueueSize();
    if ((sts = m_taskDec->workSurfacesAlloc(allocRequest.value(), true, m_mfxDEC->allocator())) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to allocate frames for hw decoder.\n"));
        return sts;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Allocated %d frames for decode [id=%d].\n"), allocRequest.value().NumFrameSuggested, allocRequest.value().AllocId);

    if ((sts = m_mfxDEC->Close()) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to reset hw decoder.\n"));
        return sts;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Closed decoder.\n"));

    if ((sts = m_mfxDEC->Init()) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to init hw decoder.\n"));
        return sts;
    }
#endif //#if ENCODER_QSV
    if (prm->metric.vmaf.enable || prm->metric.vshipEnabled()) {
        auto sts = init_metric_worker();
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    AddMessage(RGY_LOG_DEBUG, _T("Initialized decoder\n"));
    return RGY_ERR_NONE;
}

void RGYFilterSsim::close_cl_resources() {
#if ENCODER_QSV
#if ENABLE_QSV_OPENCL_INPUT_COPY
    // interopが参照するqueueとデコーダーを破棄する前に専用共有面を解放する。
    m_inputCopy.reset();
#endif
    m_surfVppInInterop.clear();
#endif
    m_queueCrop.clear();
    m_cropEvent.reset();
    for (auto &q : m_queueCalcSsim) {
        q.clear();
    }
    for (auto &q : m_queueCalcPsnr) {
        q.clear();
    }
    for (auto &buf : m_tmpSsim) {
        buf.reset();
    }
    for (auto &buf : m_tmpPsnr) {
        buf.reset();
    }
    m_decFrameCopy.reset();
    m_input.clear();
    m_unused.clear();
    m_inputReady.clear();
    m_kernel.clear();
#if ENCODER_VCEENC
    m_decoder.Release();
    m_context.Release();
    m_factory = nullptr;
    m_trace = nullptr;
#endif //#if ENCODER_VCEENC
#if ENCODER_QSV
    m_taskDec.reset();
    m_mfxDEC.reset();
#endif //#if ENCODER_QSV
}

RGY_ERR RGYFilterSsim::addBitstream(const RGYBitstream *bitstream) {
#if ENCODER_VCEENC
    if (bitstream == nullptr) {
        m_decoder->Drain();
        return RGY_ERR_NONE;
    }
    amf::AMFBufferPtr pictureBuffer;
    auto ar = m_context->AllocBuffer(amf::AMF_MEMORY_HOST, bitstream->size(), &pictureBuffer);
    if (ar != AMF_OK) {
        return err_to_rgy(ar);
    }
    memcpy(pictureBuffer->GetNative(), bitstream->data(), bitstream->size());

    //const auto duration = rgy_change_scale(bitstream.duration(), to_rgy(inTimebase), VCE_TIMEBASE);
    //const auto pts = rgy_change_scale(bitstream.pts(), to_rgy(inTimebase), VCE_TIMEBASE);
    pictureBuffer->SetDuration(bitstream->duration());
    pictureBuffer->SetPts(bitstream->pts());
    for (;;) {
        try {
            ar = m_decoder->SubmitInput(pictureBuffer);
        } catch (...) {
            AddMessage(RGY_LOG_ERROR, _T("ERROR: Unexpected error while submitting bitstream to decoder.\n"));
            ar = AMF_UNEXPECTED;
        }
        if (ar == AMF_NEED_MORE_INPUT) {
            break;
        } else if (ar == AMF_RESOLUTION_CHANGED || ar == AMF_RESOLUTION_UPDATED) {
            AddMessage(RGY_LOG_ERROR, _T("ERROR: Resolution changed during decoding.\n"));
            break;
        } else if (ar == AMF_INPUT_FULL || ar == AMF_DECODER_NO_FREE_SURFACES) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else if (ar == AMF_REPEAT) {
            continue; // 46ab4241 を反映、データはまだ使用されていないので、再度呼び出し
        } else {
            break;
        }
    }
    if (ar != AMF_OK) {
        return err_to_rgy(ar);
    }
#endif //#if ENCODER_VCEENC
#if ENCODER_QSV
    RGYBitstream bitstreamCopy;
    if (!m_encBitstreamUnused.front_copy_and_pop_no_lock(&bitstreamCopy)) {
        //なにも取得できなかった場合
        bitstreamCopy = RGYBitstreamInit();
    }
    if (bitstream) {
        bitstreamCopy.copy(bitstream);
    } else {
        //flushを意味する
        bitstreamCopy.setSize(0);
        bitstreamCopy.setOffset(0);
    }
    m_encBitstream.push(bitstreamCopy);
    AddMessage(RGY_LOG_TRACE, _T("m_inputEnc      = %d.\n"), m_inputEnc);
    m_inputEnc++;
#endif //#if ENCODER_QSV
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSsim::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    UNREFERENCED_PARAMETER(ppOutputFrames);
    UNREFERENCED_PARAMETER(pOutputFrameNum);
    RGY_ERR sts = RGY_ERR_NONE;
    {
        std::lock_guard<std::mutex> lock(m_mtx); //ロックを忘れないこと
        if (m_unused.size() == 0) {
            //待機中のフレームバッファがなければ新たに作成する
            m_unused.push_back(m_cl->createFrameBuffer(m_param->frameOut));
        }
        auto &copyFrame = m_unused.front();
        RGYOpenCLEvent inputReady;
        if (m_param->frameOut.csp == pInputFrame->csp) {
            sts = m_cl->copyFrame(&copyFrame->frame, pInputFrame, nullptr, queue, wait_events, &inputReady);
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to copy original frame: %s.\n"), get_err_mes(sts));
                return sts;
            }
        } else {
            if (!m_cropOrg) {
                unique_ptr<RGYFilterCspCrop> filterCrop(new RGYFilterCspCrop(m_cl));
                shared_ptr<RGYFilterParamCrop> paramCrop(new RGYFilterParamCrop());
                paramCrop->frameIn = *pInputFrame;
                paramCrop->frameOut = m_param->frameOut;
                paramCrop->frameOut.mem_type = RGY_MEM_TYPE_GPU;
                paramCrop->baseFps = m_param->baseFps;
                paramCrop->bOutOverwrite = false;
                sts = filterCrop->init(paramCrop, m_pLog);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                m_cropOrg = std::move(filterCrop);
                AddMessage(RGY_LOG_DEBUG, _T("created %s.\n"), m_cropOrg->GetInputMessage().c_str());
            }
            int cropFilterOutputNum = 0;
            RGYFrameInfo *outInfo[1] = { &copyFrame->frame };
            RGYFrameInfo cropInput = *pInputFrame;
            auto sts_filter = m_cropOrg->filter(&cropInput, (RGYFrameInfo **)&outInfo, &cropFilterOutputNum, queue, wait_events, &inputReady);
            if (outInfo[0] == nullptr || cropFilterOutputNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_cropOrg->name().c_str());
                return sts_filter;
            }
            if (sts_filter != RGY_ERR_NONE || cropFilterOutputNum != 1) {
                AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_cropOrg->name().c_str());
                return sts_filter;
            }
        }
        //フレームをm_unusedからm_inputに移す
        m_input.push_back(std::move(copyFrame));
        m_inputReady.push_back(inputReady);
        m_unused.pop_front();
        m_inputOriginal++;
        if (event) {
            *event = inputReady;
        }
    }

    if (m_decodeStarted) {
        if (!m_thread.joinable()) {
            while (sts == RGY_ERR_NONE) {
                sts = compare_frames();
            }
            if (sts == RGY_ERR_MORE_BITSTREAM) {
                sts = RGY_ERR_NONE;
            }
        }
    }

    AddMessage(RGY_LOG_TRACE, _T("m_inputOriginal = %d.\n"), m_inputOriginal);
    return sts;
}

void RGYFilterSsim::showResult() {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSsim>(m_param);
    if (!prm) {
        return;
    }
    const auto finishStatus = finish();
    if (finishStatus != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Video quality metric failed: %s.\n"), get_err_mes(finishStatus));
        return;
    }
    if (prm->metric.ssim && m_frames > 0) {
        auto str = strsprintf(_T("\nSSIM YUV:"));
        for (int i = 0; i < RGY_CSP_PLANES[m_param->frameOut.csp]; i++) {
            str += strsprintf(_T(" %f (%f),"), m_ssimTotalPlane[i] / m_frames, ssim_db(m_ssimTotalPlane[i], (double)m_frames));
        }
        str += strsprintf(_T(" All: %f (%f), (Frames: %d)\n"), m_ssimTotal / m_frames, ssim_db(m_ssimTotal, (double)m_frames), m_frames);
        AddMessage(RGY_LOG_INFO, _T("%s\n"), str.c_str());
    }
    if (prm->metric.psnr && m_frames > 0) {
        auto str = strsprintf(_T("\nPSNR YUV:"));
        for (int i = 0; i < RGY_CSP_PLANES[m_param->frameOut.csp]; i++) {
            str += strsprintf(_T(" %f,"), get_psnr(m_psnrTotalPlane[i], m_frames, (1 << RGY_CSP_BIT_DEPTH[prm->frameOut.csp]) - 1));
        }
        str += strsprintf(_T(" Avg: %f, (Frames: %d)\n"), get_psnr(m_psnrTotal, m_frames, (1 << RGY_CSP_BIT_DEPTH[prm->frameOut.csp]) - 1), m_frames);
        AddMessage(RGY_LOG_INFO, _T("%s\n"), str.c_str());
    }
#if ENABLE_VMAF
    if (prm->metric.vmaf.enable) {
        AddMessage(RGY_LOG_INFO, _T("VMAF Score %.6f (Frames: %d)\n"), m_vmafScore, m_vmafFrames);
    }
#endif
#if ENABLE_LIBVSHIP
    if (prm->metric.vshipSsimu2.enable) {
        auto scores = m_vshipSsimu2Scores;
        std::sort(scores.begin(), scores.end());
        const auto percentile = [&scores](const double p) {
            const auto position = (scores.size() - 1) * p;
            const auto lower = (size_t)std::floor(position);
            const auto upper = (size_t)std::ceil(position);
            return scores[lower] + (scores[upper] - scores[lower]) * (position - lower);
        };
        const auto average = m_vshipSsimu2Total / m_vshipSsimu2Frames;
        double variance = 0.0;
        for (const auto score : scores) variance += (score - average) * (score - average);
        AddMessage(RGY_LOG_INFO, _T("SSIMULACRA2: Avg %.6f, StdDev %.6f, Median %.6f, P5 %.6f, P95 %.6f, Min %.6f, Max %.6f (Frames: %d)\n"),
            average, std::sqrt(variance / scores.size()),
            percentile(0.5), percentile(0.05), percentile(0.95), scores.front(), scores.back(), m_vshipSsimu2Frames);
    }
    if (prm->metric.vshipButteraugli.enable) {
        AddMessage(RGY_LOG_INFO, _T("Butteraugli normQ: %.6f, norm3: %.6f, norminf: %.6f (Frames: %d)\n"),
            m_vshipButteraugliNormQ / m_vshipButteraugliFrames, m_vshipButteraugliNorm3 / m_vshipButteraugliFrames,
            m_vshipButteraugliNormInf / m_vshipButteraugliFrames, m_vshipButteraugliFrames);
    }
    if (prm->metric.vshipCvvdp.enable) {
        AddMessage(RGY_LOG_INFO, _T("CVVDP Score %.6f\n"), m_vshipCvvdpScore);
    }
#endif
}

RGY_ERR RGYFilterSsim::thread_func(RGYParamThread threadParam) {
    auto sts = init_cl_resources();
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    threadParam.apply(GetCurrentThread());
    AddMessage(RGY_LOG_DEBUG, _T("Set ssim/psnr calculation thread param: %s.\n"), threadParam.desc().c_str());
    m_decodeStarted = true;
    auto ret = thread_func_compare_frames();
    AddMessage(RGY_LOG_DEBUG, _T("Finishing ssim/psnr calculation thread: %s.\n"), get_err_mes(ret));
    close_cl_resources();
    return ret;
}

RGY_ERR RGYFilterSsim::thread_func_compare_frames() {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSsim>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto res = RGY_ERR_NONE;

    while (!m_abort) {
        res = compare_frames();
        if (res != RGY_ERR_NONE && res != RGY_ERR_MORE_BITSTREAM) {
            break;
        }
    }
    return res;
}

RGY_ERR RGYFilterSsim::compare_frames() {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSsim>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
#if ENCODER_VCEENC
    if (!m_decoder) {
        return RGY_ERR_MORE_DATA;
    }
    amf::AMFSurfacePtr surf;
    auto ar = AMF_REPEAT;
    //auto timeS = std::chrono::system_clock::now();
    amf::AMFDataPtr data;
    try {
        ar = m_decoder->QueryOutput(&data);
    } catch (...) {
        AddMessage(RGY_LOG_ERROR, _T("ERROR: Unexpected error while getting frame from decoder.\n"));
        ar = AMF_UNEXPECTED;
    }
    if (ar == AMF_EOF) {
        return RGY_ERR_MORE_DATA;
    }
    if (ar == AMF_REPEAT) {
        ar = AMF_OK; //これ重要...ここが欠けると最後の数フレームが欠落する
    }
    if (ar != AMF_OK) {
        auto res = err_to_rgy(ar);
        AddMessage(RGY_LOG_ERROR, _T("Failed to query output: %s.\n"), get_err_mes(res));
        return res;
    }
    if (m_abort) {
        return RGY_ERR_ABORTED;
    }
    if (data == nullptr) {
        if (m_thread.joinable()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return RGY_ERR_MORE_BITSTREAM;
    }
    // ar == AMF_OK && data != nullptr のケース: 取得できたフレームを処理
    surf = amf::AMFSurfacePtr(data);
    //if ((std::chrono::system_clock::now() - timeS) > std::chrono::seconds(10)) {
    //    PrintMes(RGY_LOG_ERROR, _T("10 sec has passed after getting last frame from decoder.\n"));
    //    PrintMes(RGY_LOG_ERROR, _T("Decoder seems to have crushed.\n"));
    //    ar = AMF_FAIL;
    //    break;
    //}
    // 取得したデコーダサーフェスを必ずOpenCLメモリへDuplicateしてから使用する
    amf::AMFDataPtr dataOCL;
    {
        VCEAMF(amf::AMFContext::AMFOpenCLLocker locker(m_context));
        ar = surf->Duplicate(amf::AMF_MEMORY_OPENCL, &dataOCL);
    }
    if (ar != AMF_OK) {
        auto res = err_to_rgy(ar);
        AddMessage(RGY_LOG_ERROR, _T("Failed to copy decoded frame to OpenCL: %s.\n"), get_err_mes(res));
        return res;
    }
    auto decFrame = std::make_unique<RGYFrameAMF>(amf::AMFSurfacePtr(dataOCL));
    {
        if (!m_cropDec) {
            AddMessage(RGY_LOG_ERROR, _T("m_cropDec not set.\n"));
            return RGY_ERR_UNKNOWN;
        }
        VCEAMF(amf::AMFContext::AMFOpenCLLocker locker(m_context));
        if (!m_decFrameCopy) {
            m_decFrameCopy = m_cl->createFrameBuffer(m_cropDec->GetFilterParam()->frameOut);
        }
        int cropFilterOutputNum = 0;
        RGYFrameInfo *outInfo[1] = { &m_decFrameCopy->frame };
        RGYFrameInfo decFrameInfo = decFrame->getInfoCopy();
        auto sts_filter = m_cropDec->filter(&decFrameInfo, (RGYFrameInfo **)&outInfo, &cropFilterOutputNum, m_queueCrop, &m_cropEvent);
        if (outInfo[0] == nullptr || cropFilterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_cropDec->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || cropFilterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_cropDec->name().c_str());
            return sts_filter;
        }

        //比較用のキューの先頭に積まれているものから順次比較していく
        RGYCLFrame *originalFrame = nullptr;
        RGYOpenCLEvent originalReady;
        {
            std::lock_guard<std::mutex> lock(m_mtx);
            if (m_input.empty() || m_inputReady.empty()) {
                AddMessage(RGY_LOG_ERROR, _T("Original frame to be compared is missing.\n"));
                return RGY_ERR_UNKNOWN;
            }
            originalFrame = m_input.front().get();
            originalReady = m_inputReady.front();
        }
        if (prm->metric.ssim || prm->metric.psnr) {
            if ((sts_filter = originalReady.wait()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to wait for original frame: %s.\n"), get_err_mes(sts_filter));
                return sts_filter;
            }
        }
        sts_filter = calc_ssim_psnr(&originalFrame->frame, &m_decFrameCopy->frame);
        if (sts_filter != RGY_ERR_NONE) {
            return sts_filter;
        }
        if (prm->metric.vmaf.enable || prm->metric.vshipEnabled()) {
            if ((sts_filter = submit_metric_frame(originalFrame, m_decFrameCopy.get(), { originalReady }, { m_cropEvent })) != RGY_ERR_NONE) {
                return sts_filter;
            }
        }
        //フレームをm_inputからm_unusedに移す
        {
            std::lock_guard<std::mutex> lock(m_mtx);
            m_unused.push_back(std::move(m_input.front()));
            m_input.pop_front();
            m_inputReady.pop_front();
        }
        m_frames++;
    }
#endif //#if ENCODER_VCEENC
#if ENCODER_QSV
    RGYBitstream bitstream = RGYBitstreamInit();
    if (!m_dec_flush // flushでなく、キューに何もない場合はsleep
        && !m_encBitstream.front_copy_no_lock(&bitstream)) { // ここではキューからpopしない(あとで行う)
        if (m_thread.joinable()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return RGY_ERR_MORE_BITSTREAM;
    }
    auto err = RGY_ERR_NONE;
    if (bitstream.size() > 0) {
        err = m_taskDec->sendFrame(&bitstream);
        if (err < RGY_ERR_NONE && err != RGY_ERR_MORE_DATA && err != RGY_ERR_MORE_SURFACE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to send frame to hw decoder.\n"));
            return err;
        }
        //sendFrameでbitstreamが消費されたかをチェックする
        //残っている場合は、キューに残したままにし、完全に消費された場合はキューから取り除く
        if (bitstream.size() == 0) {
            m_encBitstream.pop();
            m_encBitstreamUnused.push(bitstream);
        }
    } else {
        //flushのため、出力バッファを0に
        m_taskDec->setOutputMaxQueueSize(0);
        err = m_taskDec->sendFrame(nullptr); //flushのため。nullptrで呼ぶ
        if (err == RGY_ERR_MORE_DATA) {
            if (!m_dec_flush) { // 1回目は内部のバッファを消化する場合がある
                err = RGY_ERR_NONE;
            } else {
                return RGY_ERR_MORE_DATA; //flush完了、もう出てない
            }
        } else if (err < RGY_ERR_NONE && err != RGY_ERR_MORE_SURFACE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to flush hw decoder.\n"));
            return err;
        }
        m_dec_flush = true;
    }
    if (err != RGY_ERR_NONE) {
        return RGY_ERR_NONE;
    }
    if (!m_decFrameCopy) {
        m_decFrameCopy = m_cl->createFrameBuffer(m_cropDec->GetFilterParam()->frameOut);
    }
    auto outputFrames = m_taskDec->getOutput(true);
    for (auto& out : outputFrames) {
        PipelineTaskOutputSurf *taskSurf = dynamic_cast<PipelineTaskOutputSurf *>(out.get());
        if (taskSurf == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("Invalid task surface.\n"));
            return RGY_ERR_NULL_PTR;
        }
        RGYCLFrameInterop *clFrameInInterop = nullptr;
        mfxFrameSurface1 *surfVppIn = taskSurf->surf().mfx()->surf();
        if (surfVppIn == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get mfx surface pointer.\n"));
            return RGY_ERR_NULL_PTR;
        }
#if ENABLE_QSV_OPENCL_INPUT_COPY
        if (useQSVOpenCLInputCopy(m_mfxDEC->memType(), m_mfxDEC->allocator())) {
            if (!m_inputCopy) m_inputCopy = std::make_unique<QSVOpenCLInputCopy>();
            const auto copyErr = m_inputCopy->prepare(surfVppIn, m_mfxDEC->allocator(), m_cl.get(), m_queueCrop, m_cropDec->GetFilterParam()->frameIn);
            if (copyErr != RGY_ERR_NONE) {
                if (m_mfxDEC->memType() == VA_MEMORY && copyErr == RGY_ERR_UNSUPPORTED) {
                    m_inputCopy.reset();
                } else {
                    AddMessage(RGY_LOG_ERROR, _T("画質評価用デコード面のOpenCL共有用コピーに失敗しました: %s。\n"), get_err_mes(copyErr));
                    return copyErr;
                }
            } else {
                clFrameInInterop = m_inputCopy->interop();
            }
        }
#endif
        if (!clFrameInInterop) {
            if (m_surfVppInInterop.count(surfVppIn) == 0) {
                m_surfVppInInterop[surfVppIn] = getOpenCLFrameInterop(surfVppIn, m_mfxDEC->memType(), CL_MEM_READ_ONLY, m_mfxDEC->allocator(), m_cl.get(), m_queueCrop, m_cropDec->GetFilterParam()->frameIn);
            }
            clFrameInInterop = m_surfVppInInterop[surfVppIn].get();
        }
        if (!clFrameInInterop) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get OpenCL interop [in].\n"));
            return RGY_ERR_NULL_PTR;
        }
        err = clFrameInInterop->acquire(m_queueCrop);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to acquire OpenCL interop [in]: %s.\n"), get_err_mes(err));
            return RGY_ERR_NULL_PTR;
        }
        clFrameInInterop->frame.flags = taskSurf->surf().frame()->flags();
        clFrameInInterop->frame.timestamp = taskSurf->surf().frame()->timestamp();
        clFrameInInterop->frame.inputFrameId = taskSurf->surf().frame()->inputFrameId();
        clFrameInInterop->frame.picstruct = taskSurf->surf().frame()->picstruct();
        auto releaseInputInterop = [&]() {
            RGYOpenCLEvent event;
            const auto releaseErr = clFrameInInterop->release(&event);
            if (releaseErr != RGY_ERR_NONE) {
                return releaseErr;
            }
#if ENABLE_QSV_OPENCL_INPUT_COPY
            if (m_inputCopy && m_inputCopy->interop() == clFrameInInterop) {
                m_inputCopy->setReleaseEvent(event);
            }
#endif
            clFrameInInterop = nullptr;
            taskSurf->addClEvent(event);
            return RGY_ERR_NONE;
        };
        int cropFilterOutputNum = 0;
        RGYFrameInfo *outInfo[1] = { &m_decFrameCopy->frame };
        RGYFrameInfo decFrameInfo = clFrameInInterop->frameInfo();
        auto sts_filter = m_cropDec->filter(&decFrameInfo, (RGYFrameInfo **)&outInfo, &cropFilterOutputNum, m_queueCrop, &m_cropEvent);
        if (outInfo[0] == nullptr || cropFilterOutputNum != 1) {
            if ((err = releaseInputInterop()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to release OpenCL interop [in]: %s.\n"), get_err_mes(err));
                return err;
            }
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_cropDec->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || cropFilterOutputNum != 1) {
            if ((err = releaseInputInterop()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to release OpenCL interop [in]: %s.\n"), get_err_mes(err));
                return err;
            }
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_cropDec->name().c_str());
            return sts_filter;
        }
        if (clFrameInInterop) {
            if ((err = releaseInputInterop()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to release OpenCL interop [in]: %s.\n"), get_err_mes(err));
                return err;
            }
        }

        //比較用のキューの先頭に積まれているものから順次比較していく
        RGYCLFrame *originalFrame = nullptr;
        RGYOpenCLEvent originalReady;
        {
            std::lock_guard<std::mutex> lock(m_mtx);
            if (m_input.empty() || m_inputReady.empty()) {
                AddMessage(RGY_LOG_ERROR, _T("Original frame to be compared is missing.\n"));
                return RGY_ERR_UNKNOWN;
            }
            originalFrame = m_input.front().get();
            originalReady = m_inputReady.front();
        }
        if (prm->metric.ssim || prm->metric.psnr) {
            if ((sts_filter = originalReady.wait()) != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to wait for original frame: %s.\n"), get_err_mes(sts_filter));
                return sts_filter;
            }
        }
        sts_filter = calc_ssim_psnr(&originalFrame->frame, &m_decFrameCopy->frame);
        if (sts_filter != RGY_ERR_NONE) {
            return sts_filter;
        }
        if (prm->metric.vmaf.enable || prm->metric.vshipEnabled()) {
            if ((sts_filter = submit_metric_frame(originalFrame, m_decFrameCopy.get(), { originalReady }, { m_cropEvent })) != RGY_ERR_NONE) {
                return sts_filter;
            }
        }
        //フレームをm_inputからm_unusedに移す
        {
            std::lock_guard<std::mutex> lock(m_mtx);
            m_unused.push_back(std::move(m_input.front()));
            m_input.pop_front();
            m_inputReady.pop_front();
        }
        AddMessage(RGY_LOG_TRACE, _T("compared %d: 0x%p.\n"), m_frames, surfVppIn);
        m_frames++;
    }
    for (auto& out : outputFrames) {
        PipelineTaskOutputSurf *taskSurf = dynamic_cast<PipelineTaskOutputSurf *>(out.get());
        if (taskSurf == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("Invalid task surface.\n"));
            return RGY_ERR_NULL_PTR;
        }
        taskSurf->depend_clear();
    }
#endif //#if ENCODER_QSV
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSsim::build_kernel(const RGY_CSP csp) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSsim>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if ((prm->metric.enabled()) && !m_kernel.get()) {
        const auto options = strsprintf("-D BIT_DEPTH=%d -D SSIM_BLOCK_X=%d -D SSIM_BLOCK_Y=%d",
            RGY_CSP_BIT_DEPTH[csp],
            SSIM_BLOCK_X, SSIM_BLOCK_Y);
        m_kernel.set(m_cl->buildResourceAsync(_T("RGY_FILTER_SSIM_CL"), _T("EXE_DATA"), options.c_str()));
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSsim::calc_ssim_plane(const RGYFrameInfo *p0, const RGYFrameInfo *p1, std::unique_ptr<RGYCLBuf>& tmp, RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    RGYWorkSize local(SSIM_BLOCK_X, SSIM_BLOCK_Y);
    RGYWorkSize global(divCeil(p0->width, 4), divCeil(p0->height, 4));
    RGYWorkSize groups = global.groups(local);

    const auto grid_count = groups(0) * groups(1);
    if (!tmp || tmp->size() < grid_count * sizeof(float)) {
        tmp = m_cl->createBuffer(grid_count * sizeof(float));
    }
    auto err = m_kernel.get()->kernel("kernel_ssim").config(queue, local, global, wait_events).launch(
        (cl_mem)p0->ptr[0], p0->pitch[0], (cl_mem)p1->ptr[0], p1->pitch[0],
        p0->width, p0->height,
        tmp->mem()
    );
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_ssim (calc_ssim_plane(%s)): %s.\n"), RGY_CSP_NAMES[p0->csp], get_err_mes(err));
        return err;
    }
    err = tmp->queueMapBuffer(queue, CL_MAP_READ);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at queueMapBuffer (calc_ssim_plane(%s)): %s.\n"), RGY_CSP_NAMES[p0->csp], get_err_mes(err));
        return err;
    }
    return err;
}

RGY_ERR RGYFilterSsim::calc_ssim_frame(const RGYFrameInfo *p0, const RGYFrameInfo *p1) {
    for (int i = 0; i < RGY_CSP_PLANES[p0->csp]; i++) {
        const auto plane0 = getPlane(p0, (RGY_PLANE)i);
        const auto plane1 = getPlane(p1, (RGY_PLANE)i);
        const auto err = calc_ssim_plane(&plane0, &plane1, m_tmpSsim[i], m_queueCalcSsim[i], { m_cropEvent });
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSsim::calc_psnr_plane(const RGYFrameInfo *p0, const RGYFrameInfo *p1, std::unique_ptr<RGYCLBuf> &tmp, RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    RGYWorkSize local(SSIM_BLOCK_X, SSIM_BLOCK_Y);
    RGYWorkSize global(divCeil(p0->width, 4), p0->height);
    RGYWorkSize groups = global.groups(local);
    const auto grid_count = groups(0) * groups(1);
    if (!tmp || tmp->size() < grid_count * sizeof(float)) {
        tmp = m_cl->createBuffer(grid_count * sizeof(float));
    }
    auto err = m_kernel.get()->kernel("kernel_psnr").config(queue, local, global, wait_events).launch(
        (cl_mem)p0->ptr[0], p0->pitch[0], (cl_mem)p1->ptr[0], p1->pitch[0],
        p0->width, p0->height,
        tmp->mem()
    );
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at kernel_psnr (calc_psnr_plane(%s)): %s.\n"), RGY_CSP_NAMES[p0->csp], get_err_mes(err));
        return err;
    }
    err = tmp->queueMapBuffer(queue, CL_MAP_READ);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at queueMapBuffer (calc_psnr_plane(%s)): %s.\n"), RGY_CSP_NAMES[p0->csp], get_err_mes(err));
        return err;
    }
    return err;
}

RGY_ERR RGYFilterSsim::calc_psnr_frame(const RGYFrameInfo *p0, const RGYFrameInfo *p1) {
    for (int i = 0; i < RGY_CSP_PLANES[p0->csp]; i++) {
        const auto plane0 = getPlane(p0, (RGY_PLANE)i);
        const auto plane1 = getPlane(p1, (RGY_PLANE)i);
        const auto err = calc_psnr_plane(&plane0, &plane1, m_tmpPsnr[i], m_queueCalcPsnr[i], { m_cropEvent });
        if (err != RGY_ERR_NONE) {
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSsim::calc_ssim_psnr(const RGYFrameInfo *p0, const RGYFrameInfo *p1) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSsim>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto err = RGY_ERR_NONE;
    if ((prm->metric.ssim || prm->metric.psnr) && !m_kernel.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_SSIM_CL\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    if (prm->metric.ssim) {
        if ((err = calc_ssim_frame(p0, p1)) != RGY_ERR_NONE) {
            return err;
        }
    }

    if (prm->metric.psnr) {
        if ((err = calc_psnr_frame(p0, p1)) != RGY_ERR_NONE) {
            return err;
        }
    }

    if (prm->metric.ssim) {
        double ssimv = 0.0;
        for (int i = 0; i < RGY_CSP_PLANES[p0->csp]; i++) {
            VCEAMF(amf::AMFContext::AMFOpenCLLocker locker(m_context));
            m_tmpSsim[i]->mapEvent().wait();

            const int count = (int)m_tmpSsim[i]->size() / sizeof(float);
            float *ptrHost = (float *)m_tmpSsim[i]->mappedPtr();
            std::sort(ptrHost, ptrHost + count);
            double ssimPlane = 0.0;
            for (int j = 0; j < count; j++) {
                ssimPlane += (double)ptrHost[j];
            }
            const auto plane0 = getPlane(p0, (RGY_PLANE)i);
            ssimPlane /= (double)(((plane0.width >> 2) - 1) *((plane0.height >> 2) - 1));
            m_ssimTotalPlane[i] += ssimPlane;
            ssimv += ssimPlane * m_planeCoef[i];
            AddMessage(RGY_LOG_TRACE, _T("ssimPlane = %.16e, m_ssimTotalPlane[i] = %.16e"), ssimPlane, m_ssimTotalPlane[i]);
            m_tmpSsim[i]->unmapBuffer();
        }
        m_ssimTotal += ssimv;
    }

    if (prm->metric.psnr) {
        double psnrv = 0.0;
        for (int i = 0; i < RGY_CSP_PLANES[p0->csp]; i++) {
            VCEAMF(amf::AMFContext::AMFOpenCLLocker locker(m_context));
            m_tmpPsnr[i]->mapEvent().wait();

            const int count = (int)m_tmpPsnr[i]->size() / sizeof(int);
            int *ptrHost = (int *)m_tmpPsnr[i]->mappedPtr();
            int64_t psnrPlane = 0;
            for (int j = 0; j < count; j++) {
                psnrPlane += ptrHost[j];
            }
            const auto plane0 = getPlane(p0, (RGY_PLANE)i);
            double psnrPlaneF = psnrPlane / (double)(plane0.width * plane0.height);
            m_psnrTotalPlane[i] += psnrPlaneF;
            psnrv += psnrPlaneF * m_planeCoef[i];
            AddMessage(RGY_LOG_TRACE, _T("psnrPlane = %.16e, m_psnrTotalPlane[i] = %.16e"), psnrPlane, m_psnrTotalPlane[i]);
            m_tmpPsnr[i]->unmapBuffer();
        }
        m_psnrTotal += psnrv;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSsim::metricStatus() const {
#if ENABLE_VMAF || ENABLE_LIBVSHIP
    std::lock_guard<std::mutex> lock(m_metricMutex);
    return m_metricError;
#else
    return RGY_ERR_NONE;
#endif
}

RGY_ERR RGYFilterSsim::init_metric_worker() {
#if ENABLE_VMAF || ENABLE_LIBVSHIP
    std::unique_lock<std::mutex> lock(m_metricMutex);
    if (m_metricThread.joinable()) {
        return m_metricError;
    }
    m_metricReady.clear();
    m_metricFree = { 0, 1 };
    m_metricInputFin = false;
    m_metricStop = false;
    m_metricWorkerReady = false;
    m_metricWorkerDone = false;
    m_metricFinishCalled = false;
    m_metricNextIndex = 0;
    m_metricError = RGY_ERR_NONE;
    m_metricFinishResult = RGY_ERR_NONE;
    m_metricThread = std::thread(&RGYFilterSsim::metric_worker, this);
    m_metricInitCv.wait(lock, [this]() { return m_metricWorkerReady; });
    return m_metricError;
#else
    return RGY_ERR_NONE;
#endif
}

#if ENABLE_VMAF || ENABLE_LIBVSHIP
RGY_ERR RGYFilterSsim::copy_metric_frame(RGYCLFrame *source, const std::vector<RGYOpenCLEvent> &waitEvents, MetricHostFrame& destination) {
    if (!source) {
        return RGY_ERR_NULL_PTR;
    }
    auto sts = source->queueMapBuffer(m_queueCrop, CL_MAP_READ, waitEvents, RGY_CL_MAP_BLOCK_NONE);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to map metric frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
    sts = source->mapWait();
    if (sts != RGY_ERR_NONE) {
        source->unmapBuffer(m_queueCrop);
        AddMessage(RGY_LOG_ERROR, _T("Failed to wait for mapped metric frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
    const auto& mapped = source->mappedHost()->host();
    destination.csp = mapped.csp;
    destination.width = mapped.width;
    destination.height = mapped.height;
    const auto pixelSize = (RGY_CSP_BIT_DEPTH[mapped.csp] > 8) ? 2 : 1;
    for (int i = 0; i < RGY_CSP_PLANES[mapped.csp]; i++) {
        const auto plane = getPlane(&mapped, (RGY_PLANE)i);
        const auto rowBytes = (size_t)plane.width * pixelSize;
        destination.planeWidth[i] = plane.width;
        destination.planeHeight[i] = plane.height;
        destination.pitch[i] = (int64_t)rowBytes;
        destination.plane[i].resize(rowBytes * plane.height);
        for (int y = 0; y < plane.height; y++) {
            memcpy(destination.plane[i].data() + rowBytes * y, plane.ptr[0] + plane.pitch[0] * y, rowBytes);
        }
    }
    if ((sts = source->unmapBuffer(m_queueCrop)) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to unmap metric frame: %s.\n"), get_err_mes(sts));
        return sts;
    }
    return m_queueCrop.finish();
}
#endif

RGY_ERR RGYFilterSsim::submit_metric_frame(RGYCLFrame *reference, RGYCLFrame *distorted, const std::vector<RGYOpenCLEvent> &referenceWaitEvents, const std::vector<RGYOpenCLEvent> &distortedWaitEvents) {
#if ENABLE_VMAF || ENABLE_LIBVSHIP
    if (auto sts = metricStatus(); sts != RGY_ERR_NONE) {
        return sts;
    }
    int slotIndex = -1;
    {
        std::unique_lock<std::mutex> lock(m_metricMutex);
        m_metricFreeCv.wait(lock, [this]() { return !m_metricFree.empty() || m_metricStop || m_metricError != RGY_ERR_NONE; });
        if (m_metricError != RGY_ERR_NONE) {
            return m_metricError;
        }
        if (m_metricStop || m_metricFree.empty()) {
            return RGY_ERR_ABORTED;
        }
        slotIndex = m_metricFree.front();
        m_metricFree.pop_front();
    }
    auto& slot = m_metricSlots[slotIndex];
    auto restoreSlot = [&]() {
        std::lock_guard<std::mutex> lock(m_metricMutex);
        m_metricFree.push_back(slotIndex);
        m_metricFreeCv.notify_one();
    };
    if (auto sts = copy_metric_frame(reference, referenceWaitEvents, slot.reference); sts != RGY_ERR_NONE) {
        restoreSlot();
        return sts;
    }
    if (auto sts = copy_metric_frame(distorted, distortedWaitEvents, slot.distorted); sts != RGY_ERR_NONE) {
        restoreSlot();
        return sts;
    }
    RGY_ERR finalStatus = RGY_ERR_NONE;
    {
        std::lock_guard<std::mutex> lock(m_metricMutex);
        if (m_metricError != RGY_ERR_NONE) {
            finalStatus = m_metricError;
        } else if (m_metricStop) {
            finalStatus = RGY_ERR_ABORTED;
        } else {
            slot.index = m_metricNextIndex++;
            m_metricReady.push_back(slotIndex);
        }
    }
    if (finalStatus != RGY_ERR_NONE) {
        restoreSlot();
        return finalStatus;
    }
    m_metricReadyCv.notify_one();
    return RGY_ERR_NONE;
#else
    UNREFERENCED_PARAMETER(reference);
    UNREFERENCED_PARAMETER(distorted);
    UNREFERENCED_PARAMETER(referenceWaitEvents);
    UNREFERENCED_PARAMETER(distortedWaitEvents);
    return RGY_ERR_NONE;
#endif
}

RGY_ERR RGYFilterSsim::finish_metric_worker() {
#if ENABLE_VMAF || ENABLE_LIBVSHIP
    {
        std::lock_guard<std::mutex> lock(m_metricMutex);
        m_metricInputFin = true;
    }
    m_metricReadyCv.notify_all();
    m_metricFreeCv.notify_all();
    if (m_metricThread.joinable()) {
        m_metricThread.join();
    }
    std::lock_guard<std::mutex> lock(m_metricMutex);
    return m_metricError;
#else
    return RGY_ERR_NONE;
#endif
}

#if ENABLE_VMAF || ENABLE_LIBVSHIP
RGY_ERR RGYFilterSsim::metric_worker() {
    try {
    auto workerPrm = std::dynamic_pointer_cast<RGYFilterParamSsim>(m_param);
    if (!workerPrm) {
        {
            std::lock_guard<std::mutex> lock(m_metricMutex);
            m_metricError = RGY_ERR_INVALID_PARAM;
            m_metricStop = true;
            m_metricWorkerReady = true;
            m_metricWorkerDone = true;
        }
        m_metricInitCv.notify_all();
        m_metricReadyCv.notify_all();
        m_metricFreeCv.notify_all();
        return RGY_ERR_INVALID_PARAM;
    }
    workerPrm->threadParam.apply(GetCurrentThread());
    AddMessage(RGY_LOG_DEBUG, _T("Set video quality metric worker param: %s.\n"), workerPrm->threadParam.desc().c_str());
    RGY_ERR sts = RGY_ERR_NONE;
#if ENABLE_VMAF
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSsim>(m_param);
    if (prm && prm->metric.vmaf.enable) {
        sts = init_vmaf_metric();
    }
#endif
#if ENABLE_LIBVSHIP
    if (sts == RGY_ERR_NONE) {
        auto prm = std::dynamic_pointer_cast<RGYFilterParamSsim>(m_param);
        if (prm && prm->metric.vshipEnabled()) {
            sts = init_vship_metric();
        }
    }
#endif
    {
        std::lock_guard<std::mutex> lock(m_metricMutex);
        if (sts != RGY_ERR_NONE) {
            m_metricError = sts;
            m_metricStop = true;
        }
        m_metricWorkerReady = true;
    }
    m_metricInitCv.notify_all();
    m_metricFreeCv.notify_all();
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    for (;;) {
        int slotIndex = -1;
        {
            std::unique_lock<std::mutex> lock(m_metricMutex);
            m_metricReadyCv.wait(lock, [this]() { return m_metricStop || !m_metricReady.empty() || m_metricInputFin; });
            if (m_metricStop) {
                break;
            }
            if (m_metricReady.empty()) {
                if (m_metricInputFin) {
                    break;
                }
                continue;
            }
            slotIndex = m_metricReady.front();
            m_metricReady.pop_front();
        }
        sts = process_metric_pair(m_metricSlots[slotIndex]);
        {
            std::lock_guard<std::mutex> lock(m_metricMutex);
            m_metricFree.push_back(slotIndex);
            if (sts != RGY_ERR_NONE && m_metricError == RGY_ERR_NONE) {
                m_metricError = sts;
                m_metricStop = true;
            }
        }
        m_metricFreeCv.notify_all();
        m_metricReadyCv.notify_all();
        if (sts != RGY_ERR_NONE) {
            break;
        }
    }
    if (sts == RGY_ERR_NONE) {
#if ENABLE_VMAF
        auto prm = std::dynamic_pointer_cast<RGYFilterParamSsim>(m_param);
        if (prm && prm->metric.vmaf.enable) {
            sts = finish_vmaf_metric();
        }
#endif
#if ENABLE_LIBVSHIP
        if (sts == RGY_ERR_NONE) {
            auto prm = std::dynamic_pointer_cast<RGYFilterParamSsim>(m_param);
            if (prm && prm->metric.vshipEnabled()) {
                sts = finish_vship_metric();
            }
        }
#endif
    }
    {
        std::lock_guard<std::mutex> lock(m_metricMutex);
        if (sts != RGY_ERR_NONE && m_metricError == RGY_ERR_NONE) {
            m_metricError = sts;
        }
        m_metricWorkerDone = true;
    }
    m_metricFreeCv.notify_all();
    m_metricReadyCv.notify_all();
    return sts;
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(m_metricMutex);
            if (m_metricError == RGY_ERR_NONE) {
                m_metricError = RGY_ERR_MEMORY_ALLOC;
            }
            m_metricStop = true;
            m_metricWorkerReady = true;
            m_metricWorkerDone = true;
        }
        m_metricInitCv.notify_all();
        m_metricReadyCv.notify_all();
        m_metricFreeCv.notify_all();
        AddMessage(RGY_LOG_ERROR, _T("Video quality metric worker failed unexpectedly.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
}

RGY_ERR RGYFilterSsim::process_metric_pair(const MetricHostPair& pair) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSsim>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_PARAM;
    }
#if ENABLE_VMAF
    if (prm->metric.vmaf.enable) {
        if (auto sts = process_vmaf_metric(pair); sts != RGY_ERR_NONE) {
            return sts;
        }
    }
#endif
#if ENABLE_LIBVSHIP
    if (prm->metric.vshipEnabled()) {
        if (auto sts = process_vship_metric(pair); sts != RGY_ERR_NONE) {
            return sts;
        }
    }
#endif
    return RGY_ERR_NONE;
}
#endif

#if ENABLE_VMAF
static VmafPixelFormat metric_vmaf_pixfmt(const RGY_CSP csp) {
    switch (RGY_CSP_CHROMA_FORMAT[csp]) {
    case RGY_CHROMAFMT_YUV420: return VMAF_PIX_FMT_YUV420P;
    case RGY_CHROMAFMT_YUV422: return VMAF_PIX_FMT_YUV422P;
    case RGY_CHROMAFMT_YUV444: return VMAF_PIX_FMT_YUV444P;
    default: return VMAF_PIX_FMT_UNKNOWN;
    }
}

RGY_ERR RGYFilterSsim::init_vmaf_metric() {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSsim>(m_param);
    if (!prm || !m_libvmaf.load()) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to load %s.\n"), RGY_LIBVMAF_FILENAME);
        return RGY_ERR_FILE_OPEN;
    }
    VmafConfiguration config = {};
    config.log_level = (enum VmafLogLevel)VMAF_LOG_LEVEL_INFO;
    config.n_threads = (prm->metric.vmaf.threads == 0) ? get_cpu_info().physical_cores : prm->metric.vmaf.threads;
    config.n_subsample = prm->metric.vmaf.subsample;
    config.cpumask = 0;
    if (m_libvmaf.p_vmaf_init()(&m_vmafContext, config) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to initialize VMAF context.\n"));
        return RGY_ERR_UNKNOWN;
    }
    std::string model;
    if (tchar_to_string(prm->metric.vmaf.model.c_str(), model) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to convert VMAF model name.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    VmafModelConfig modelConfig = {};
    modelConfig.name = "vmaf";
    modelConfig.flags = (prm->metric.vmaf.enable_transform || prm->metric.vmaf.phone_model) ? VMAF_MODEL_FLAG_ENABLE_TRANSFORM : VMAF_MODEL_FLAGS_DEFAULT;
    const bool modelPath = rgy_file_exists(model);
    const auto lowerModel = [&model]() {
        auto value = model;
        std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) { return (char)std::tolower(c); });
        return value;
    }();
    if (!modelPath && lowerModel.size() >= 5 && lowerModel.substr(lowerModel.size() - 5) == ".json") {
        AddMessage(RGY_LOG_ERROR, _T("VMAF model file not found: %s.\n"), prm->metric.vmaf.model.c_str());
        return RGY_ERR_FILE_OPEN;
    }
    int err = modelPath
        ? m_libvmaf.p_vmaf_model_load_from_path()(&m_vmafModel, &modelConfig, model.c_str())
        : m_libvmaf.p_vmaf_model_load()(&m_vmafModel, &modelConfig, model.c_str());
    if (err != 0 && m_libvmaf.version_class() == RGYLibVMAFVersion::V3_OR_LATER) {
        err = modelPath
            ? m_libvmaf.p_vmaf_model_collection_load_from_path()(&m_vmafModel, &m_vmafModelCollection, &modelConfig, model.c_str())
            : m_libvmaf.p_vmaf_model_collection_load()(&m_vmafModel, &m_vmafModelCollection, &modelConfig, model.c_str());
    }
    if (err != 0 || !m_vmafModel) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to load VMAF model: %s.\n"), prm->metric.vmaf.model.c_str());
        return RGY_ERR_UNKNOWN;
    }
    err = m_vmafModelCollection
        ? m_libvmaf.p_vmaf_use_features_from_model_collection()(m_vmafContext, m_vmafModelCollection)
        : m_libvmaf.p_vmaf_use_features_from_model()(m_vmafContext, m_vmafModel);
    if (err != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to load VMAF model features.\n"));
        return RGY_ERR_UNKNOWN;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Loaded %s (%s), CPU feature extraction.\n"), RGY_LIBVMAF_FILENAME, char_to_tstring(m_libvmaf.version()).c_str());
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSsim::process_vmaf_metric(const MetricHostPair& pair) {
    const auto pixfmt = metric_vmaf_pixfmt(pair.reference.csp);
    if (pixfmt == VMAF_PIX_FMT_UNKNOWN || pair.reference.csp != pair.distorted.csp) {
        AddMessage(RGY_LOG_ERROR, _T("Unsupported format for VMAF.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    VmafPicture reference = {};
    VmafPicture distorted = {};
    const auto bitDepth = RGY_CSP_BIT_DEPTH[pair.reference.csp];
    const auto referenceAllocated = m_libvmaf.p_vmaf_picture_alloc()(&reference, pixfmt, bitDepth, pair.reference.width, pair.reference.height) == 0;
    const auto distortedAllocated = referenceAllocated && m_libvmaf.p_vmaf_picture_alloc()(&distorted, pixfmt, bitDepth, pair.distorted.width, pair.distorted.height) == 0;
    if (!referenceAllocated || !distortedAllocated) {
        if (referenceAllocated) m_libvmaf.p_vmaf_picture_unref()(&reference);
        if (distortedAllocated) m_libvmaf.p_vmaf_picture_unref()(&distorted);
        AddMessage(RGY_LOG_ERROR, _T("Failed to allocate VMAF pictures.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    const auto copyPicture = [](VmafPicture& picture, const MetricHostFrame& source) {
        const int pixelSize = (RGY_CSP_BIT_DEPTH[source.csp] > 8) ? 2 : 1;
        for (int i = 0; i < RGY_CSP_PLANES[source.csp]; i++) {
            const auto bytes = (size_t)source.planeWidth[i] * pixelSize;
            for (int y = 0; y < source.planeHeight[i]; y++) {
                memcpy((uint8_t *)picture.data[i] + picture.stride[i] * y, source.plane[i].data() + source.pitch[i] * y, bytes);
            }
        }
    };
    copyPicture(reference, pair.reference);
    copyPicture(distorted, pair.distorted);
    const auto err = m_libvmaf.p_vmaf_read_pictures()(m_vmafContext, &reference, &distorted, pair.index);
    if (err != 0) {
        m_libvmaf.p_vmaf_picture_unref()(&reference);
        m_libvmaf.p_vmaf_picture_unref()(&distorted);
        AddMessage(RGY_LOG_ERROR, _T("Failed to submit VMAF picture %d.\n"), pair.index);
        return RGY_ERR_UNKNOWN;
    }
    m_vmafFrames++;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSsim::finish_vmaf_metric() {
    if (m_vmafFrames == 0) {
        AddMessage(RGY_LOG_ERROR, _T("No frames were provided to VMAF.\n"));
        return RGY_ERR_UNKNOWN;
    }
    if (m_libvmaf.p_vmaf_read_pictures()(m_vmafContext, nullptr, nullptr, 0) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to finalize VMAF score.\n"));
        return RGY_ERR_UNKNOWN;
    }
    if (m_vmafModelCollection) {
        VmafModelCollectionScore collectionScore = {};
        if (m_libvmaf.p_vmaf_score_pooled_model_collection()(m_vmafContext, m_vmafModelCollection, VMAF_POOL_METHOD_MEAN, &collectionScore, 0, m_vmafFrames - 1) != 0) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to finalize VMAF model collection.\n"));
            return RGY_ERR_UNKNOWN;
        }
    }
    if (m_libvmaf.p_vmaf_score_pooled()(m_vmafContext, m_vmafModel, VMAF_POOL_METHOD_MEAN, &m_vmafScore, 0, m_vmafFrames - 1) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to finalize VMAF score.\n"));
        return RGY_ERR_UNKNOWN;
    }
    if (!std::isfinite(m_vmafScore)) {
        auto prm = std::dynamic_pointer_cast<RGYFilterParamSsim>(m_param);
        double scoreSum = 0.0;
        int scoreCount = 0;
        for (int index = 0; index < m_vmafFrames; index++) {
            if (prm->metric.vmaf.subsample > 1 && (index % prm->metric.vmaf.subsample) != 0) {
                continue;
            }
            double score = 0.0;
            if (m_libvmaf.p_vmaf_score_at_index()(m_vmafContext, m_vmafModel, &score, index) != 0) {
                AddMessage(RGY_LOG_ERROR, _T("Failed to get VMAF score at frame %d.\n"), index);
                return RGY_ERR_UNKNOWN;
            }
            if (std::isfinite(score)) {
                scoreSum += score;
                scoreCount++;
            }
        }
        if (scoreCount == 0) {
            AddMessage(RGY_LOG_ERROR, _T("VMAF returned no finite frame scores.\n"));
            return RGY_ERR_UNKNOWN;
        }
        m_vmafScore = scoreSum / scoreCount;
        AddMessage(RGY_LOG_WARN, _T("VMAF pooled score was non-finite; recalculated from %d frame scores.\n"), scoreCount);
    }
    return RGY_ERR_NONE;
}
#endif

#if ENABLE_LIBVSHIP
static RGY_ERR metric_vship_colorspace(Vship_Colorspace_t& colorspace, const RGYFrameInfo& frame, const VideoVUIInfo& vui) {
    colorspace = {};
    colorspace.width = frame.width;
    colorspace.height = frame.height;
    colorspace.target_width = -1;
    colorspace.target_height = -1;
    switch (RGY_CSP_BIT_DEPTH[frame.csp]) {
    case 8:  colorspace.sample = Vship_SampleUINT8; break;
    case 9:  colorspace.sample = Vship_SampleUINT9; break;
    case 10: colorspace.sample = Vship_SampleUINT10; break;
    case 12: colorspace.sample = Vship_SampleUINT12; break;
    case 14: colorspace.sample = Vship_SampleUINT14; break;
    case 16: colorspace.sample = Vship_SampleUINT16; break;
    default: return RGY_ERR_UNSUPPORTED;
    }
    switch (vui.colorrange) {
    case RGY_COLORRANGE_FULL: colorspace.range = Vship_RangeFull; break;
    case RGY_COLORRANGE_LIMITED:
    case RGY_COLORRANGE_UNSPECIFIED:
    case RGY_COLORRANGE_AUTO: colorspace.range = Vship_RangeLimited; break;
    default: return RGY_ERR_UNSUPPORTED;
    }
    switch (RGY_CSP_CHROMA_FORMAT[frame.csp]) {
    case RGY_CHROMAFMT_YUV420: colorspace.subsampling = { 1, 1 }; break;
    case RGY_CHROMAFMT_YUV422: colorspace.subsampling = { 1, 0 }; break;
    case RGY_CHROMAFMT_YUV444: colorspace.subsampling = { 0, 0 }; break;
    default: return RGY_ERR_UNSUPPORTED;
    }
    switch (vui.chromaloc) {
    case RGY_CHROMALOC_UNSPECIFIED:
    case RGY_CHROMALOC_AUTO:
    case RGY_CHROMALOC_LEFT: colorspace.chromaLocation = Vship_ChromaLoc_Left; break;
    case RGY_CHROMALOC_CENTER: colorspace.chromaLocation = Vship_ChromaLoc_Center; break;
    case RGY_CHROMALOC_TOPLEFT: colorspace.chromaLocation = Vship_ChromaLoc_TopLeft; break;
    case RGY_CHROMALOC_TOP: colorspace.chromaLocation = Vship_ChromaLoc_Top; break;
    default: return RGY_ERR_UNSUPPORTED;
    }
    colorspace.colorFamily = Vship_ColorYUV;
    switch (vui.matrix) {
    case RGY_MATRIX_AUTO:
    case RGY_MATRIX_UNSPECIFIED:
    case RGY_MATRIX_BT709: colorspace.YUVMatrix = Vship_MATRIX_BT709; break;
    case RGY_MATRIX_BT470_BG: colorspace.YUVMatrix = Vship_MATRIX_BT470_BG; break;
    case RGY_MATRIX_ST170_M: colorspace.YUVMatrix = Vship_MATRIX_ST170_M; break;
    case RGY_MATRIX_BT2020_NCL: colorspace.YUVMatrix = Vship_MATRIX_BT2020_NCL; break;
    case RGY_MATRIX_BT2020_CL: colorspace.YUVMatrix = Vship_MATRIX_BT2020_CL; break;
    case RGY_MATRIX_ICTCP: colorspace.YUVMatrix = Vship_MATRIX_BT2100_ICTCP; break;
    default: return RGY_ERR_UNSUPPORTED;
    }
    switch (vui.transfer) {
    case RGY_TRANSFER_AUTO:
    case RGY_TRANSFER_UNSPECIFIED:
    case RGY_TRANSFER_BT709: colorspace.transferFunction = Vship_TRC_BT709; break;
    case RGY_TRANSFER_BT470_M: colorspace.transferFunction = Vship_TRC_BT470_M; break;
    case RGY_TRANSFER_BT470_BG: colorspace.transferFunction = Vship_TRC_BT470_BG; break;
    case RGY_TRANSFER_BT601: colorspace.transferFunction = Vship_TRC_BT601; break;
    case RGY_TRANSFER_ST240_M: colorspace.transferFunction = Vship_TRC_ST240_M; break;
    case RGY_TRANSFER_LINEAR: colorspace.transferFunction = Vship_TRC_Linear; break;
    case RGY_TRANSFER_IEC61966_2_1: colorspace.transferFunction = Vship_TRC_sRGB; break;
    case RGY_TRANSFER_ST2084: colorspace.transferFunction = Vship_TRC_PQ; break;
    case RGY_TRANSFER_ARIB_B67: colorspace.transferFunction = Vship_TRC_HLG; break;
    default: return RGY_ERR_UNSUPPORTED;
    }
    switch (vui.colorprim) {
    case RGY_PRIM_AUTO:
    case RGY_PRIM_UNSPECIFIED:
    case RGY_PRIM_BT709: colorspace.primaries = Vship_PRIMARIES_BT709; break;
    case RGY_PRIM_BT470_M: colorspace.primaries = Vship_PRIMARIES_BT470_M; break;
    case RGY_PRIM_BT470_BG: colorspace.primaries = Vship_PRIMARIES_BT470_BG; break;
    case RGY_PRIM_ST170_M: colorspace.primaries = Vship_PRIMARIES_ST170_M; break;
    case RGY_PRIM_ST240_M: colorspace.primaries = Vship_PRIMARIES_ST240_M; break;
    case RGY_PRIM_BT2020: colorspace.primaries = Vship_PRIMARIES_BT2020; break;
    case RGY_PRIM_ST432_1: colorspace.primaries = Vship_PRIMARIES_DisplayP3; break;
    default: return RGY_ERR_UNSUPPORTED;
    }
    colorspace.crop = { 0, 0, 0, 0 };
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSsim::init_vship_metric() {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSsim>(m_param);
    if (!prm || !m_libvship.load()) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to load %s.\n"), RGY_LIBVSHIP_DLL_NAME);
        return RGY_ERR_FILE_OPEN;
    }
    int deviceCount = 0;
    if (m_libvship.p_Vship_GetDeviceCount()(&deviceCount) != Vship_NoError || deviceCount <= 0
        || m_libvship.p_Vship_GPUFullCheck()(0) != Vship_NoError
        || m_libvship.p_Vship_SetDevice()(0) != Vship_NoError) {
        AddMessage(RGY_LOG_ERROR, _T("libvship GPU 0 is unavailable.\n"));
        return RGY_ERR_DEVICE_NOT_FOUND;
    }
    Vship_DeviceInfo deviceInfo = {};
    if (m_libvship.p_Vship_GetDeviceInfo()(&deviceInfo, 0) == Vship_NoError) {
        AddMessage(RGY_LOG_DEBUG, _T("libvship uses GPU 0: %s.\n"), char_to_tstring(deviceInfo.name).c_str());
    }
    // 未指定値だけを解像度から既定化し、明示された未対応値は下の変換で拒否する。
    auto vui = prm->input.vui;
    const auto defaultVui = VideoVUIInfo()
        .to((CspMatrix)COLOR_VALUE_AUTO_RESOLUTION)
        .to((CspColorprim)COLOR_VALUE_AUTO_RESOLUTION)
        .to((CspTransfer)COLOR_VALUE_AUTO_RESOLUTION);
    vui.setIfUnsetUnknwonAuto(defaultVui);
    vui.apply_auto(VideoVUIInfo(), m_param->frameOut.height);
    Vship_Colorspace_t colorspace = {};
    if (auto sts = metric_vship_colorspace(colorspace, m_param->frameOut, vui); sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Unsupported colorspace for libvship: matrix %d, transfer %d, primaries %d, range %d, chroma location %d.\n"),
            vui.matrix, vui.transfer, vui.colorprim, vui.colorrange, vui.chromaloc);
        return sts;
    }
    if (prm->metric.vshipSsimu2.enable) {
        if (m_libvship.p_Vship_SSIMU2Init2()(&m_vshipSsimu2, colorspace, colorspace, 0) != Vship_NoError) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to initialize SSIMULACRA2.\n"));
            return RGY_ERR_UNKNOWN;
        }
        m_vshipSsimu2Initialized = true;
    }
    if (prm->metric.vshipButteraugli.enable) {
        if (m_libvship.p_Vship_ButteraugliInit2()(&m_vshipButteraugli, colorspace, colorspace, prm->metric.vshipButteraugli.Qnorm, prm->metric.vshipButteraugli.intensity_multiplier, 0) != Vship_NoError) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to initialize Butteraugli.\n"));
            return RGY_ERR_UNKNOWN;
        }
        m_vshipButteraugliInitialized = true;
    }
    if (prm->metric.vshipCvvdp.enable) {
        std::string model, config;
        tchar_to_string(prm->metric.vshipCvvdp.model.c_str(), model);
        tchar_to_string(prm->metric.vshipCvvdp.model_config_json.c_str(), config);
        const auto fps = (float)prm->baseFps.n() / prm->baseFps.d();
        if (m_libvship.p_Vship_CVVDPInit3()(&m_vshipCvvdp, colorspace, colorspace, fps, prm->metric.vshipCvvdp.resize, model.c_str(), config.empty() ? nullptr : config.c_str(), 0) != Vship_NoError) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to initialize CVVDP.\n"));
            return RGY_ERR_UNKNOWN;
        }
        m_vshipCvvdpInitialized = true;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSsim::process_vship_metric(const MetricHostPair& pair) {
    const uint8_t *reference[3] = { nullptr, nullptr, nullptr };
    const uint8_t *distorted[3] = { nullptr, nullptr, nullptr };
    int64_t referencePitch[3] = { 0, 0, 0 };
    int64_t distortedPitch[3] = { 0, 0, 0 };
    for (int i = 0; i < RGY_CSP_PLANES[pair.reference.csp]; i++) {
        reference[i] = pair.reference.plane[i].data();
        distorted[i] = pair.distorted.plane[i].data();
        referencePitch[i] = pair.reference.pitch[i];
        distortedPitch[i] = pair.distorted.pitch[i];
    }
    if (m_vshipSsimu2Initialized) {
        double score = 0.0;
        if (m_libvship.p_Vship_ComputeSSIMU2()(m_vshipSsimu2, &score, reference, distorted, referencePitch, distortedPitch) != Vship_NoError || !std::isfinite(score)) {
            AddMessage(RGY_LOG_ERROR, _T("SSIMULACRA2 failed at frame %d.\n"), pair.index);
            return RGY_ERR_UNKNOWN;
        }
        m_vshipSsimu2Total += score;
        m_vshipSsimu2Scores.push_back(score);
        m_vshipSsimu2Frames++;
    }
    if (m_vshipButteraugliInitialized) {
        Vship_ButteraugliScore score = {};
        if (m_libvship.p_Vship_ComputeButteraugli()(m_vshipButteraugli, &score, nullptr, 0, reference, distorted, referencePitch, distortedPitch) != Vship_NoError
            || !std::isfinite(score.normQ) || !std::isfinite(score.norm3) || !std::isfinite(score.norminf)) {
            AddMessage(RGY_LOG_ERROR, _T("Butteraugli failed at frame %d.\n"), pair.index);
            return RGY_ERR_UNKNOWN;
        }
        m_vshipButteraugliNormQ += score.normQ;
        m_vshipButteraugliNorm3 += score.norm3;
        m_vshipButteraugliNormInf += score.norminf;
        m_vshipButteraugliFrames++;
    }
    if (m_vshipCvvdpInitialized) {
        if (m_libvship.p_Vship_ComputeCVVDP()(m_vshipCvvdp, &m_vshipCvvdpScore, nullptr, 0, reference, distorted, referencePitch, distortedPitch) != Vship_NoError || !std::isfinite(m_vshipCvvdpScore)) {
            AddMessage(RGY_LOG_ERROR, _T("CVVDP failed at frame %d.\n"), pair.index);
            return RGY_ERR_UNKNOWN;
        }
        m_vshipCvvdpFrames++;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSsim::finish_vship_metric() {
    if ((m_vshipSsimu2Initialized && m_vshipSsimu2Frames == 0)
        || (m_vshipButteraugliInitialized && m_vshipButteraugliFrames == 0)
        || (m_vshipCvvdpInitialized && m_vshipCvvdpFrames == 0)) {
        AddMessage(RGY_LOG_ERROR, _T("No frames were provided to libvship.\n"));
        return RGY_ERR_UNKNOWN;
    }
    if (m_vshipCvvdpInitialized) m_libvship.p_Vship_CVVDPFree()(m_vshipCvvdp);
    if (m_vshipButteraugliInitialized) m_libvship.p_Vship_ButteraugliFree()(m_vshipButteraugli);
    if (m_vshipSsimu2Initialized) m_libvship.p_Vship_SSIMU2Free()(m_vshipSsimu2);
    m_vshipCvvdpInitialized = false;
    m_vshipButteraugliInitialized = false;
    m_vshipSsimu2Initialized = false;
    return RGY_ERR_NONE;
}
#endif

void RGYFilterSsim::close_metric_resources() {
#if ENABLE_VMAF || ENABLE_LIBVSHIP
    {
        std::lock_guard<std::mutex> lock(m_metricMutex);
        m_metricStop = true;
        m_metricInputFin = true;
    }
    m_metricReadyCv.notify_all();
    m_metricFreeCv.notify_all();
    if (m_metricThread.joinable()) {
        m_metricThread.join();
    }
#if ENABLE_VMAF
    if (m_vmafModelCollection) {
        m_libvmaf.p_vmaf_model_collection_destroy()(m_vmafModelCollection);
        m_vmafModelCollection = nullptr;
    }
    if (m_vmafModel) {
        m_libvmaf.p_vmaf_model_destroy()(m_vmafModel);
        m_vmafModel = nullptr;
    }
    if (m_vmafContext) {
        m_libvmaf.p_vmaf_close()(m_vmafContext);
        m_vmafContext = nullptr;
    }
    m_libvmaf.close();
#endif
#if ENABLE_LIBVSHIP
    if (m_libvship.loaded()) {
        if (m_vshipCvvdpInitialized) m_libvship.p_Vship_CVVDPFree()(m_vshipCvvdp);
        if (m_vshipButteraugliInitialized) m_libvship.p_Vship_ButteraugliFree()(m_vshipButteraugli);
        if (m_vshipSsimu2Initialized) m_libvship.p_Vship_SSIMU2Free()(m_vshipSsimu2);
    }
    m_vshipCvvdpInitialized = false;
    m_vshipButteraugliInitialized = false;
    m_vshipSsimu2Initialized = false;
    m_libvship.close();
#endif
#endif
}

RGY_ERR RGYFilterSsim::finish() {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSsim>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_finishCalled) {
        return m_finishResult;
    }
    RGY_ERR result = metricStatus();
    const bool compareThreadJoined = m_thread.joinable();
    if (compareThreadJoined) {
        m_thread.join();
    }
    if (result == RGY_ERR_NONE && m_decodeStarted && !compareThreadJoined) {
        for (;;) {
            const auto sts = compare_frames();
            if (sts == RGY_ERR_NONE) {
                continue;
            }
            if (sts == RGY_ERR_MORE_DATA) {
                break;
            }
            result = sts;
            break;
        }
    }
    if (result == RGY_ERR_NONE && !m_input.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("Decoded frame count does not match original frames.\n"));
        result = RGY_ERR_UNKNOWN;
    }
    if (result == RGY_ERR_NONE && m_frames == 0) {
        AddMessage(RGY_LOG_ERROR, _T("評価対象なし: video quality metric received no frame pairs.\n"));
        result = RGY_ERR_UNKNOWN;
    }
    if (result == RGY_ERR_NONE) {
        result = finish_metric_worker();
    } else {
#if ENABLE_VMAF || ENABLE_LIBVSHIP
        {
            std::lock_guard<std::mutex> lock(m_metricMutex);
            m_metricStop = true;
        }
        m_metricReadyCv.notify_all();
        m_metricFreeCv.notify_all();
        finish_metric_worker();
#endif
    }
#if ENABLE_VMAF || ENABLE_LIBVSHIP
    {
        std::lock_guard<std::mutex> lock(m_metricMutex);
        m_metricFinishCalled = true;
        m_metricFinishResult = result;
    }
#endif
    m_finishCalled = true;
    m_finishResult = result;
    return result;
}


void RGYFilterSsim::close() {
    close_metric_resources();
    if (m_thread.joinable()) {
        AddMessage(RGY_LOG_DEBUG, _T("Waiting for ssim/psnr calculation thread to finish.\n"));
        m_abort = true;
        m_thread.join();
    }
    // デコーダの出力を確実に回収してサーフェスを解放する
#if ENCODER_VCEENC
    if (m_decoder) {
        try {
            m_decoder->Drain();
        } catch (...) {
            AddMessage(RGY_LOG_ERROR, _T("ERROR: Unexpected error while draining decoder.\n"));
        }
        for (;;) {
            amf::AMFDataPtr data;
            AMF_RESULT ar = AMF_OK;
            try {
                ar = m_decoder->QueryOutput(&data);
            } catch (...) {
                AddMessage(RGY_LOG_ERROR, _T("ERROR: Unexpected error while getting frame from decoder on close.\n"));
                break;
            }
            if (ar == AMF_EOF) {
                break;
            }
            if (ar == AMF_REPEAT || (ar == AMF_OK && data == nullptr)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            if (ar != AMF_OK) {
                AddMessage(RGY_LOG_WARN, _T("Decoder QueryOutput returned %s on close.\n"), AMFRetString(ar));
                break;
            }
            // dataがある場合は、そのままスコープアウトで参照を落とす
        }
    }
#endif
    close_cl_resources();
    m_cropOrg.reset();
    m_cropDec.reset();
    AddMessage(RGY_LOG_DEBUG, _T("closed ssim/psnr filter.\n"));
}
