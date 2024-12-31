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
    m_cropOrg.reset();
    m_cropDec.reset();
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
    std::remove_pointer<RGYArgN<2U, decltype(av_stream_get_side_data)>::type>::type side_data_size = 0;
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
    if (auto err = build_kernel(m_param->frameOut.csp); err != RGY_ERR_NONE) {
        return err;
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
    AddMessage(RGY_LOG_DEBUG, _T("Initialized decoder\n"));
    return RGY_ERR_NONE;
}

void RGYFilterSsim::close_cl_resources() {
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
    m_surfVppInInterop.clear();
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
        ar = m_decoder->SubmitInput(pictureBuffer);
        if (ar == AMF_NEED_MORE_INPUT) {
            break;
        } else if (ar == AMF_RESOLUTION_CHANGED || ar == AMF_RESOLUTION_UPDATED) {
            AddMessage(RGY_LOG_ERROR, _T("ERROR: Resolution changed during decoding.\n"));
            break;
        } else if (ar == AMF_INPUT_FULL || ar == AMF_DECODER_NO_FREE_SURFACES) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else if (ar == AMF_REPEAT) {
            pictureBuffer = nullptr;
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
        if (m_param->frameOut.csp == pInputFrame->csp) {
            m_cl->copyFrame(&copyFrame->frame, pInputFrame, nullptr, queue, wait_events, event);
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
            auto sts_filter = m_cropOrg->filter(&cropInput, (RGYFrameInfo **)&outInfo, &cropFilterOutputNum, queue, wait_events, event);
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
        m_unused.pop_front();
        m_inputOriginal++;
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
    if (m_thread.joinable()) {
        AddMessage(RGY_LOG_DEBUG, _T("Waiting for ssim/psnr calculation thread to finish.\n"));
        m_thread.join();
    } else {
        //シングルスレッド動作時はここで最終処理を行う
        auto sts = RGY_ERR_NONE;
        while (sts == RGY_ERR_NONE) {
            sts = compare_frames();
        }
    }
    if (prm->metric.ssim) {
        auto str = strsprintf(_T("\nSSIM YUV:"));
        for (int i = 0; i < RGY_CSP_PLANES[m_param->frameOut.csp]; i++) {
            str += strsprintf(_T(" %f (%f),"), m_ssimTotalPlane[i] / m_frames, ssim_db(m_ssimTotalPlane[i], (double)m_frames));
        }
        str += strsprintf(_T(" All: %f (%f), (Frames: %d)\n"), m_ssimTotal / m_frames, ssim_db(m_ssimTotal, (double)m_frames), m_frames);
        AddMessage(RGY_LOG_INFO, _T("%s\n"), str.c_str());
    }
    if (prm->metric.psnr) {
        auto str = strsprintf(_T("\nPSNR YUV:"));
        for (int i = 0; i < RGY_CSP_PLANES[m_param->frameOut.csp]; i++) {
            str += strsprintf(_T(" %f,"), get_psnr(m_psnrTotalPlane[i], m_frames, (1 << RGY_CSP_BIT_DEPTH[prm->frameOut.csp]) - 1));
        }
        str += strsprintf(_T(" Avg: %f, (Frames: %d)\n"), get_psnr(m_psnrTotal, m_frames, (1 << RGY_CSP_BIT_DEPTH[prm->frameOut.csp]) - 1), m_frames);
        AddMessage(RGY_LOG_INFO, _T("%s\n"), str.c_str());
    }
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
#if ENCODER_VCEENC
    if (!m_decoder) {
        return RGY_ERR_MORE_DATA;
    }
    amf::AMFSurfacePtr surf;
    auto ar = AMF_REPEAT;
    //auto timeS = std::chrono::system_clock::now();
    amf::AMFDataPtr data;
    ar = m_decoder->QueryOutput(&data);
    if (ar == AMF_EOF) {
        return RGY_ERR_MORE_DATA;
    }
    if (ar == AMF_REPEAT) {
        ar = AMF_OK; //これ重要...ここが欠けると最後の数フレームが欠落する
    }
    if (ar == AMF_OK && data != nullptr) {
        surf = amf::AMFSurfacePtr(data);
    } else if (ar != AMF_OK) {
        auto res = err_to_rgy(ar);
        AddMessage(RGY_LOG_ERROR, _T("Failed to query output: %s.\n"), get_err_mes(res));
        return res;
    } else if (m_abort) {
        return RGY_ERR_ABORTED;
    } else {
        if (m_thread.joinable()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return RGY_ERR_MORE_BITSTREAM;
    }
    //if ((std::chrono::system_clock::now() - timeS) > std::chrono::seconds(10)) {
    //    PrintMes(RGY_LOG_ERROR, _T("10 sec has passed after getting last frame from decoder.\n"));
    //    PrintMes(RGY_LOG_ERROR, _T("Decoder seems to have crushed.\n"));
    //    ar = AMF_FAIL;
    //    break;
    //}
    auto decFrame = std::make_unique<RGYFrameAMF>(surf);
    const auto &decAmf = decFrame->amf();
    {
        VCEAMF(amf::AMFContext::AMFOpenCLLocker locker(m_context));
#if 1
        //dummyのCPUへのメモリコピーを行う
        //こうしないとデコーダからの出力をOpenCLに渡したときに、フレームが壊れる(フレーム順序が入れ替わってガクガクする)
        amf::AMFDataPtr data;
        decAmf->Duplicate(amf::AMF_MEMORY_HOST, &data);
#endif
        ar = decAmf->Convert(amf::AMF_MEMORY_OPENCL);
        if (ar != AMF_OK) {
            auto res = err_to_rgy(ar);
            AddMessage(RGY_LOG_ERROR, _T("Failed to load input frame: %s.\n"), get_err_mes(res));
            return res;
        }
    }
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
        std::lock_guard<std::mutex> lock(m_mtx); //ロックを忘れないこと
        auto &originalFrame = m_input.front();
        sts_filter = calc_ssim_psnr(&originalFrame->frame, &m_decFrameCopy->frame);
        if (sts_filter != RGY_ERR_NONE) {
            return sts_filter;
        }
        //フレームをm_inputからm_unusedに移す
        m_unused.push_back(std::move(originalFrame));
        m_input.pop_front();
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
        if (m_surfVppInInterop.count(surfVppIn) == 0) {
            m_surfVppInInterop[surfVppIn] = getOpenCLFrameInterop(surfVppIn, m_mfxDEC->memType(), CL_MEM_READ_ONLY, m_mfxDEC->allocator(), m_cl.get(), m_cl->queue(), m_cropDec->GetFilterParam()->frameIn);
        }
        clFrameInInterop = m_surfVppInInterop[surfVppIn].get();
        if (!clFrameInInterop) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to get OpenCL interop [in].\n"));
            return RGY_ERR_NULL_PTR;
        }
        err = clFrameInInterop->acquire(m_cl->queue());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to acquire OpenCL interop [in]: %s.\n"), get_err_mes(err));
            return RGY_ERR_NULL_PTR;
        }
        clFrameInInterop->frame.flags = taskSurf->surf().frame()->flags();
        clFrameInInterop->frame.timestamp = taskSurf->surf().frame()->timestamp();
        clFrameInInterop->frame.inputFrameId = taskSurf->surf().frame()->inputFrameId();
        clFrameInInterop->frame.picstruct = taskSurf->surf().frame()->picstruct();
        int cropFilterOutputNum = 0;
        RGYFrameInfo *outInfo[1] = { &m_decFrameCopy->frame };
        RGYFrameInfo decFrameInfo = clFrameInInterop->frameInfo();
        auto sts_filter = m_cropDec->filter(&decFrameInfo, (RGYFrameInfo **)&outInfo, &cropFilterOutputNum, m_queueCrop, &m_cropEvent);
        if (outInfo[0] == nullptr || cropFilterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Unknown behavior \"%s\".\n"), m_cropDec->name().c_str());
            return sts_filter;
        }
        if (sts_filter != RGY_ERR_NONE || cropFilterOutputNum != 1) {
            AddMessage(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_cropDec->name().c_str());
            return sts_filter;
        }
        if (clFrameInInterop) {
            RGYOpenCLEvent event;
            clFrameInInterop->release(&event);
            clFrameInInterop = nullptr;
            taskSurf->addClEvent(event);
        }

        //比較用のキューの先頭に積まれているものから順次比較していく
        std::lock_guard<std::mutex> lock(m_mtx); //ロックを忘れないこと
        auto &originalFrame = m_input.front();
        sts_filter = calc_ssim_psnr(&originalFrame->frame, &m_decFrameCopy->frame);
        if (sts_filter != RGY_ERR_NONE) {
            return sts_filter;
        }
        //フレームをm_inputからm_unusedに移す
        m_unused.push_back(std::move(originalFrame));
        m_input.pop_front();
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
    if (!m_kernel.get()) {
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


void RGYFilterSsim::close() {
    if (m_thread.joinable()) {
        AddMessage(RGY_LOG_DEBUG, _T("Waiting for ssim/psnr calculation thread to finish.\n"));
        m_abort = true;
        m_thread.join();
    }
    close_cl_resources();
    m_cropOrg.reset();
    m_cropDec.reset();
    AddMessage(RGY_LOG_DEBUG, _T("closed ssim/psnr filter.\n"));
}
