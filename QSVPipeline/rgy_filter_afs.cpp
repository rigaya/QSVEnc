// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
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
#if defined(_M_IX86) || defined(_M_X64)
#include <emmintrin.h>
#endif // #if defined(_M_IX86) || defined(_M_X64)
#include "convert_csp.h"
#include "rgy_filter_afs.h"
#include "afs_stg.h"
#include "rgy_avutil.h"
#include "rgy_filesystem.h"
#pragma warning (push)

static void afs_get_motion_count_simd(int *motion_count, const uint8_t *ptr, const AFS_SCAN_CLIP *clip, int pitch, int scan_w, int scan_h, int tb_order);
static void afs_get_stripe_count_simd(int *stripe_count, const uint8_t *ptr, const AFS_SCAN_CLIP *clip, int pitch, int scan_w, int scan_h, int tb_order);

template<typename T>
T max3(T a, T b, T c) {
    return std::max(std::max(a, b), c);
}
template<typename T>
T absdiff(T a, T b) {
    T a_b = a - b;
    T b_a = b - a;
    return (a >= b) ? a_b : b_a;
}

afsSourceCache::afsSourceCache(shared_ptr<RGYOpenCLContext> cl) :
    m_cl(cl),
    m_sourceArray(),
    m_csp(RGY_CSP_NA),
    m_nFramesInput(0) {
}

RGY_ERR afsSourceCache::alloc(const RGYFrameInfo& frameInfo) {
    m_csp = frameInfo.csp;
    if (RGY_CSP_CHROMA_FORMAT[m_csp] == RGY_CHROMAFMT_YUV444) {
        for (int i = 0; i < (int)m_sourceArray.size(); i++) {
            m_sourceArray[i].y = m_cl->createFrameBuffer(frameInfo);
        }
    } else if (RGY_CSP_CHROMA_FORMAT[m_csp] == RGY_CHROMAFMT_YUV420) {
        RGYFrameInfo frameY = getPlane(&frameInfo, RGY_PLANE_Y);
        RGYFrameInfo frameU = getPlane(&frameInfo, RGY_PLANE_U);
        RGYFrameInfo frameV = getPlane(&frameInfo, RGY_PLANE_V);
        frameY.csp = (RGY_CSP_BIT_DEPTH[frameInfo.csp] > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;
        frameU.csp = frameY.csp;
        frameV.csp = frameY.csp;
        frameU.height >>= 1; //フィールドを分離して保存するため
        frameV.height >>= 1; //フィールドを分離して保存するため
        for (int i = 0; i < (int)m_sourceArray.size(); i++) {
            m_sourceArray[i].y  = m_cl->createFrameBuffer(frameY);
            for (int j = 0; j < (int)m_sourceArray[i].cb.size(); j++) {
                m_sourceArray[i].cb[j] = m_cl->createFrameBuffer(frameU);
                m_sourceArray[i].cr[j] = m_cl->createFrameBuffer(frameV);
            }
        }
    } else {
        return RGY_ERR_UNSUPPORTED;
    }
    return RGY_ERR_NONE;
}

RGY_ERR afsSourceCache::build(const RGYFrameInfo& frameInfo) {
    m_cl->requestCSPCopy(frameInfo, frameInfo);
    return RGY_ERR_NONE;
}

RGY_ERR afsSourceCache::add(const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue_main, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent &event) {
    const int iframe = m_nFramesInput++;
    auto pDstFrame = get(iframe);
    pDstFrame->y->frame.flags        = pInputFrame->flags;
    pDstFrame->y->frame.picstruct    = pInputFrame->picstruct;
    pDstFrame->y->frame.timestamp    = pInputFrame->timestamp;
    pDstFrame->y->frame.duration     = pInputFrame->duration;
    pDstFrame->y->frame.inputFrameId = pInputFrame->inputFrameId;

    auto ret = RGY_ERR_NONE;
    if (RGY_CSP_CHROMA_FORMAT[m_csp] == RGY_CHROMAFMT_YUV444) {
        ret = m_cl->copyFrame(&pDstFrame->y->frame, pInputFrame, nullptr, queue_main, wait_events, &event);
    } else if (RGY_CSP_CHROMA_FORMAT[m_csp] == RGY_CHROMAFMT_YUV420) {
        RGYFrameInfo frameY = getPlane(pInputFrame, RGY_PLANE_Y);
        RGYFrameInfo frameU = getPlane(pInputFrame, RGY_PLANE_U);
        RGYFrameInfo frameV = getPlane(pInputFrame, RGY_PLANE_V);

        ret = m_cl->copyPlane(&pDstFrame->y->frame, &frameY, nullptr, queue_main, &event);
        if (ret != RGY_ERR_NONE) return ret;

        auto copyProgram = m_cl->getCspCopyProgram(pDstFrame->cb[0]->frame, frameU);
        if (!copyProgram) {
            return RGY_ERR_OPENCL_CRUSH;
        }
        RGYWorkSize local(32, 8);
        RGYWorkSize globalCb(pDstFrame->cb[0]->frame.width, pDstFrame->cb[0]->frame.height);
        ret = copyProgram->kernel("kernel_separate_fields").config(queue_main, local, globalCb, wait_events, &event).launch(
            pDstFrame->cb[0]->mem(0), pDstFrame->cb[1]->mem(0), pDstFrame->cb[0]->frame.pitch[0],
            (cl_mem)frameU.ptr[0], frameU.pitch[0], pDstFrame->cb[0]->frame.width, pDstFrame->cb[0]->frame.height);
        if (ret != RGY_ERR_NONE) return ret;

        RGYWorkSize globalCr(pDstFrame->cr[0]->frame.width, pDstFrame->cr[0]->frame.height);
        ret = copyProgram->kernel("kernel_separate_fields").config(queue_main, local, globalCr, wait_events, &event).launch(
            pDstFrame->cr[0]->mem(0), pDstFrame->cr[1]->mem(0), pDstFrame->cr[0]->frame.pitch[0],
            (cl_mem)frameV.ptr[0], frameV.pitch[0], pDstFrame->cr[0]->frame.width, pDstFrame->cr[0]->frame.height);
        if (ret != RGY_ERR_NONE) return ret;
    } else {
        return RGY_ERR_UNSUPPORTED;
    }
    return ret;
}

RGY_ERR afsSourceCache::copyFrame(RGYCLFrame *pOut, int srcFrame, RGYOpenCLQueue &queue, RGYOpenCLEvent *event) {
    afsSourceCacheFrame *pSrc = get(srcFrame);
    if (RGY_CSP_CHROMA_FORMAT[m_csp] == RGY_CHROMAFMT_YUV444) {
        return m_cl->copyFrame(&pOut->frame, &pSrc->y->frame, nullptr, queue, event);
    } else if (RGY_CSP_CHROMA_FORMAT[m_csp] == RGY_CHROMAFMT_YUV420) {
        RGYFrameInfo frameY = getPlane(&pOut->frame, RGY_PLANE_Y);
        RGYFrameInfo frameU = getPlane(&pOut->frame, RGY_PLANE_U);
        RGYFrameInfo frameV = getPlane(&pOut->frame, RGY_PLANE_V);

        auto ret = m_cl->copyPlane(&frameY, &pSrc->y->frame, nullptr, queue, event);
        if (ret != RGY_ERR_NONE) return ret;

        auto copyProgram = m_cl->getCspCopyProgram(pOut->frame, pSrc->cb[0]->frame);
        if (!copyProgram) {
            return RGY_ERR_OPENCL_CRUSH;
        }

        RGYWorkSize local(32, 8);
        RGYWorkSize globalCb(pSrc->cb[0]->frame.width, pSrc->cb[0]->frame.height);
        ret = copyProgram->kernel("kernel_merge_fields").config(queue, local, globalCb, event).launch(
            pOut->mem(1), pOut->frame.pitch[1],
            pSrc->cb[0]->mem(0), pSrc->cb[1]->mem(0), pSrc->cb[0]->frame.pitch[0],
            pSrc->cb[0]->frame.width, pSrc->cb[0]->frame.height);
        if (ret != RGY_ERR_NONE) return ret;

        RGYWorkSize globalCr(pSrc->cr[0]->frame.width, pSrc->cr[0]->frame.height);
        ret = copyProgram->kernel("kernel_merge_fields").config(queue, local, globalCr, event).launch(
            pOut->mem(2), pOut->frame.pitch[2],
            pSrc->cr[0]->mem(0), pSrc->cr[1]->mem(0), pSrc->cr[0]->frame.pitch[0],
            pSrc->cr[0]->frame.width, pSrc->cr[0]->frame.height);
        if (ret != RGY_ERR_NONE) return ret;
        return ret;
    } else {
        return RGY_ERR_UNSUPPORTED;
    }
}

void afsSourceCache::clear() {
    for (int i = 0; i < (int)m_sourceArray.size(); i++) {
        m_sourceArray[i].y->clear();
        for (const auto &cache : m_sourceArray[i].cb) {
            cache->clear();
        }
        for (const auto &cache : m_sourceArray[i].cr) {
            cache->clear();
        }
    }
    m_nFramesInput = 0;
}

afsSourceCache::~afsSourceCache() {
    clear();
}

afsScanCache::afsScanCache(shared_ptr<RGYOpenCLContext> cl) :
    m_cl(cl),
    m_scanArray() {
}

void afsScanCache::clearcache(int iframe) {
    auto data = get(iframe);
    data->status = 0;
    data->frame = 0;
    data->tb_order = 0;
    data->thre_shift = 0;
    data->thre_deint = 0;
    data->thre_Ymotion = 0;
    data->thre_Cmotion = 0;
    memset(&data->clip, 0, sizeof(data->clip));
    data->ff_motion = 0;
    data->lf_motion = 0;
    data->event.reset();
}

void afsScanCache::initcache(int iframe) {
    clearcache(iframe);
    auto data = get(iframe);
    data->event.reset();
}

RGY_ERR afsScanCache::alloc(const RGYFrameInfo& frameInfo) {
    RGYFrameInfo scanFrame = frameInfo;
    scanFrame.csp = RGY_CSP_NV12;
    for (int i = 0; i < (int)m_scanArray.size(); i++) {
        initcache(i);
        m_scanArray[i].map = m_cl->createFrameBuffer(scanFrame);
        if (!m_scanArray[i].map) {
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_scanArray[i].status = 0;
    }
    return RGY_ERR_NONE;
}

void afsScanCache::clear() {
    for (int i = 0; i < (int)m_scanArray.size(); i++) {
        m_scanArray[i].map.reset();
        m_scanArray[i].event.reset();
        clearcache(i);
    }
}

afsScanCache::~afsScanCache() {
    clear();
}

afsStripeCache::afsStripeCache(shared_ptr<RGYOpenCLContext> cl) :
    m_cl(cl),
    m_stripeArray(),
    m_analyzeMapFilter() {
}

void afsStripeCache::clearcache(int iframe) {
    auto data = get(iframe);
    data->status = 0;
    data->frame = 0;
    data->count0 = 0;
    data->count1 = 0;
    data->event.reset();
}

void afsStripeCache::initcache(int iframe) {
    auto data = get(iframe);
    data->event.reset();
}

void afsStripeCache::expire(int iframe) {
    auto stp = get(iframe);
    if (stp->frame == iframe && stp->status > 0) {
        stp->status = 0;
    }
}

RGY_ERR afsStripeCache::alloc(const RGYFrameInfo& frameInfo) {
    RGYFrameInfo stripeFrame = frameInfo;
    stripeFrame.csp = RGY_CSP_NV12;
    for (int i = 0; i < (int)m_stripeArray.size(); i++) {
        initcache(i);
        m_stripeArray[i].map = m_cl->createFrameBuffer(stripeFrame);
        if (!m_stripeArray[i].map) {
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    return RGY_ERR_NONE;
}

bool afsStripeCache::kernelBuildSuccess() {
    return m_analyzeMapFilter.get() != nullptr;
}

AFS_STRIPE_DATA *afsStripeCache::filter(int iframe, int analyze, RGYOpenCLQueue &queue, RGY_ERR *pErr) {
    auto sip = get(iframe);
    if (analyze > 1) {
        auto sip_dst = getFiltered();
        if ((*pErr = map_filter(sip_dst, sip, queue)) != RGY_ERR_NONE) {
            sip_dst = nullptr;
        }
        sip = sip_dst;
    }
    return sip;
}

void afsStripeCache::clear() {
    for (int i = 0; i < (int)m_stripeArray.size(); i++) {
        m_stripeArray[i].map.reset();
        m_stripeArray[i].buf_count_stripe.reset();
        m_stripeArray[i].event.reset();
        clearcache(i);
    }
}

afsStripeCache::~afsStripeCache() {
    clear();
}

afsStreamStatus::afsStreamStatus() :
    m_initialized(false),
    m_quarter_jitter(0),
    m_additional_jitter(0),
    m_phase24(0),
    m_position24(0),
    m_prev_jitter(0),
    m_prev_rff_smooth(0),
    m_prev_status(0),
    m_set_frame(-1),
    m_pos(),
    m_fpLog() {
};

afsStreamStatus::~afsStreamStatus() {
    m_fpLog.reset();
}

void afsStreamStatus::init(uint8_t status, int drop24) {
    m_prev_status = status;
    m_prev_jitter = 0;
    m_additional_jitter = 0;
    m_prev_rff_smooth = 0;
    m_phase24 = 4;
    m_position24 = 0;
    if (drop24 ||
        (!(status & AFS_FLAG_SHIFT0) &&
        (status & AFS_FLAG_SHIFT1) &&
            (status & AFS_FLAG_SHIFT2))) {
        m_phase24 = 0;
    }
    if (status & AFS_FLAG_FORCE24) {
        m_position24++;
    } else {
        m_phase24 -= m_position24 + 1;
        m_position24 = 0;
    }
    m_initialized = true;
}

int afsStreamStatus::open_log(const tstring& log_filename) {
    FILE *fp = NULL;
    if (_tfopen_s(&fp, log_filename.c_str(), _T("w"))) {
        return 1;
    }
    m_fpLog = unique_ptr<FILE, fp_deleter>(fp, fp_deleter());
    fprintf(m_fpLog.get(), " iframe,  sts,       ,        pos,   orig_pts, q_jit, prevjit, pos24, phase24, rff_smooth\n");
    return 0;
}

void afsStreamStatus::write_log(const afsFrameTs *const frameTs) {
    if (!m_fpLog) {
        return;
    }
    fprintf(m_fpLog.get(), "%7d, 0x%2x, %s%s%s%s%s%s, %10lld, %10lld, %3d, %3d, %3d, %3d, %3d\n",
        frameTs->iframe,
        m_prev_status,
        m_prev_status & AFS_FLAG_PROGRESSIVE ? "p" : "i",
        ISRFF(m_prev_status) ? "r" : "-",
        (((m_prev_status & AFS_FLAG_PROGRESSIVE) ? 0 : m_prev_status) & AFS_FLAG_SHIFT0) ? "0" : "-",
        (((m_prev_status & AFS_FLAG_PROGRESSIVE) ? 0 : m_prev_status) & AFS_FLAG_SHIFT1) ? "1" : "-",
        (((m_prev_status & AFS_FLAG_PROGRESSIVE) ? 0 : m_prev_status) & AFS_FLAG_SHIFT2) ? "2" : "-",
        (((m_prev_status & AFS_FLAG_PROGRESSIVE) ? 0 : m_prev_status) & AFS_FLAG_SHIFT3) ? "3" : "-",
        (long long int)frameTs->pos, (long long int)frameTs->orig_pts,
        m_quarter_jitter, m_prev_jitter, m_position24, m_phase24, m_prev_rff_smooth);
    return;
}

int afsStreamStatus::set_status(int iframe, uint8_t status, int drop24, int64_t orig_pts) {
    afsFrameTs *const frameTs = &m_pos[iframe & 15];
    frameTs->iframe = iframe;
    frameTs->orig_pts = orig_pts;
    if (!m_initialized) {
        init(status, 0);
        frameTs->pos = orig_pts;
        m_set_frame = iframe;
        write_log(frameTs);
        return 0;
    }
    if (iframe > m_set_frame + 1) {
        return 1;
    }
    m_set_frame = iframe;

    int pull_drop = 0;
    int quarter_jitter = 0;
    int rff_smooth = 0;
    if (status & AFS_FLAG_PROGRESSIVE) {
        if (status & (AFS_FLAG_FORCE24 | AFS_FLAG_SMOOTHING)) {
            if (!m_prev_rff_smooth) {
                if (ISRFF(m_prev_status)) rff_smooth = -1;
                else if ((m_prev_status & AFS_FLAG_PROGRESSIVE) && ISRFF(status)) rff_smooth = 1;
            }
            quarter_jitter = rff_smooth;
        }
        pull_drop = 0;
        m_additional_jitter = 0;
        drop24 = 0;
    } else {
        if (status & AFS_FLAG_SHIFT0) {
            quarter_jitter = -2;
        } else if (m_prev_status & AFS_FLAG_SHIFT0) {
            quarter_jitter = (status & AFS_FLAG_SMOOTHING) ? -1 : -2;
        } else {
            quarter_jitter = 0;
        }
        quarter_jitter += ((status & AFS_FLAG_SMOOTHING) || m_additional_jitter != -1) ? m_additional_jitter : -2;

        if (status & (AFS_FLAG_FORCE24 | AFS_FLAG_SMOOTHING)) {
            if (!m_prev_rff_smooth) {
                if (ISRFF(m_prev_status)) rff_smooth = -1;
                else if ((m_prev_status & AFS_FLAG_PROGRESSIVE) && ISRFF(status)) rff_smooth = 1;
            }
        }
        quarter_jitter += rff_smooth;
        m_position24 += rff_smooth;

        pull_drop = (status & AFS_FLAG_FRAME_DROP)
            && !((m_prev_status|status) & AFS_FLAG_SHIFT0)
            && (status & AFS_FLAG_SHIFT1);
        m_additional_jitter = pull_drop ? -1 : 0;

        drop24 = drop24 ||
            (!(status & AFS_FLAG_SHIFT0) &&
              (status & AFS_FLAG_SHIFT1) &&
              (status & AFS_FLAG_SHIFT2));
    }

    if (drop24) m_phase24 = (m_position24 + 100) % 5;
    drop24 = 0;
    if (m_position24 >= m_phase24 &&
        ((m_position24 + 100) % 5 == m_phase24 ||
         (m_position24 +  99) % 5 == m_phase24)) {
        m_position24 -= 5;
        drop24 = 1;
    }

    if (status & AFS_FLAG_FORCE24) {
        pull_drop = drop24;
        if (status & AFS_FLAG_PROGRESSIVE) {
            quarter_jitter += m_position24;
        } else {
            quarter_jitter = m_position24++;
        }
    } else if (!(status & AFS_FLAG_PROGRESSIVE)) {
        m_phase24 -= m_position24 + 1;
        m_position24 = 0;
    }
    int drop_thre = (status & AFS_FLAG_FRAME_DROP) ? 0 : -3;
    if (!(status & AFS_FLAG_PROGRESSIVE) && ISRFF(m_prev_status)) {
        //rffからの切替時はなるべくdropさせない
        drop_thre = -3;
    }
    int drop = (quarter_jitter - m_prev_jitter < drop_thre);

    m_quarter_jitter = quarter_jitter;
    m_prev_rff_smooth = rff_smooth;
    m_prev_status = status;

    drop |= pull_drop;
    if (drop) {
        m_prev_jitter -= 4;
        m_quarter_jitter = 0;
        frameTs->pos = AFS_SSTS_DROP; //drop
    } else {
        m_prev_jitter = m_quarter_jitter;
        frameTs->pos = frameTs->orig_pts + m_quarter_jitter;
    }
    write_log(frameTs);
    return 0;
}

int64_t afsStreamStatus::get_duration(int64_t iframe) {
    if (m_set_frame < iframe + 2) {
        return AFS_SSTS_ERROR;
    }
    auto iframe_pos = m_pos[(iframe + 0) & 15].pos;
    if (iframe_pos < 0) {
        return AFS_SSTS_DROP;
    }
    auto next_pos = m_pos[(iframe + 1) & 15].pos;
    if (next_pos < 0) {
        //iframe + 1がdropならその先のフレームを参照
        next_pos = m_pos[(iframe + 2) & 15].pos;
    }
    if (next_pos < 0) {
        //iframe + 1がdropならその先のフレームを参照
        next_pos = m_pos[(iframe + 3) & 15].pos;
    }
    const auto duration = next_pos - iframe_pos;
    return (duration > 0) ? duration : AFS_SSTS_DROP;
}

RGYFilterAfs::RGYFilterAfs(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_queueAnalyze(),
    m_queueCopy(),
    m_eventSrcAdd(),
    m_eventScanFrame(),
    m_eventMergeScan(),
    m_nFrame(0),
    m_nPts(0),
    m_source(context),
    m_scan(context),
    m_stripe(context),
    m_status(),
    m_streamsts(),
    m_count_motion(),
    m_fpTimecode(),
    m_mergeScan(),
    m_analyze(),
    m_synthesize() {
    m_name = _T("afs");
}

RGYFilterAfs::~RGYFilterAfs() {
    close();
}

RGY_ERR RGYFilterAfs::check_param(shared_ptr<RGYFilterParamAfs> pAfsParam) {
    if (pAfsParam->frameOut.height <= 0 || pAfsParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int hight_mul = (RGY_CSP_CHROMA_FORMAT[pAfsParam->frameOut.csp] == RGY_CHROMAFMT_YUV420) ? 4 : 2;
    if ((pAfsParam->frameOut.height % hight_mul) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("Height must be multiple of %d.\n"), hight_mul);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pAfsParam->afs.clip.top < 0 || pAfsParam->afs.clip.top >= pAfsParam->frameOut.height) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (clip.top).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pAfsParam->afs.clip.bottom < 0 || pAfsParam->afs.clip.bottom >= pAfsParam->frameOut.height) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (clip.bottom).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pAfsParam->afs.clip.top + pAfsParam->afs.clip.bottom >= pAfsParam->frameOut.height) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (clip.top + clip.bottom).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pAfsParam->afs.clip.left < 0 || pAfsParam->afs.clip.left >= pAfsParam->frameOut.width) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (clip.left).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pAfsParam->afs.clip.left % 4 != 0) {
        AddMessage(RGY_LOG_ERROR, _T("parameter \"left\" rounded to multiple of 4.\n"));
        pAfsParam->afs.clip.left = (pAfsParam->afs.clip.left + 2) & ~3;
    }
    if (pAfsParam->afs.clip.right < 0 || pAfsParam->afs.clip.right >= pAfsParam->frameOut.width) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (clip.right).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pAfsParam->afs.clip.right % 4 != 0) {
        AddMessage(RGY_LOG_ERROR, _T("parameter \"right\" rounded to multiple of 4.\n"));
        pAfsParam->afs.clip.right = (pAfsParam->afs.clip.right + 2) & ~3;
    }
    if (pAfsParam->afs.clip.left + pAfsParam->afs.clip.right >= pAfsParam->frameOut.width) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (clip.left + clip.right).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pAfsParam->afs.method_switch < 0 || pAfsParam->afs.method_switch > 256) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (method_switch).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pAfsParam->afs.coeff_shift < 0 || pAfsParam->afs.coeff_shift > 256) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (coeff_shift).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pAfsParam->afs.thre_shift < 0 || pAfsParam->afs.thre_shift > 1024) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (thre_shift).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pAfsParam->afs.thre_deint < 0 || pAfsParam->afs.thre_deint > 1024) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (thre_deint).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pAfsParam->afs.thre_Ymotion < 0 || pAfsParam->afs.thre_Ymotion > 1024) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (thre_Ymotion).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pAfsParam->afs.thre_Cmotion < 0 || pAfsParam->afs.thre_Cmotion > 1024) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (thre_Cmotion).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pAfsParam->afs.analyze < 0 || pAfsParam->afs.analyze > 5) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter (level).\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (!pAfsParam->afs.shift) {
        AddMessage(RGY_LOG_WARN, _T("shift was off, so drop and smooth will also be off.\n"));
        pAfsParam->afs.drop = false;
        pAfsParam->afs.smooth = false;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterAfs::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto pAfsParam = std::dynamic_pointer_cast<RGYFilterParamAfs>(pParam);
    if (!pAfsParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //パラメータチェック
    if (check_param(pAfsParam) != RGY_ERR_NONE) {
        return RGY_ERR_INVALID_PARAM;
    }

    auto err = AllocFrameBuf(pAfsParam->frameOut, 1);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
        return err;
    }
    for (int i = 0; i < _countof(pAfsParam->frameOut.pitch); i++) {
        pAfsParam->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    AddMessage(RGY_LOG_DEBUG, _T("allocated output buffer: %dx%d, pitch %d, %s.\n"),
        m_frameBuf[0]->frame.width, m_frameBuf[0]->frame.height, m_frameBuf[0]->frame.pitch[0], RGY_CSP_NAMES[m_frameBuf[0]->frame.csp]);

    if (RGY_ERR_NONE != (err = m_source.alloc(pAfsParam->frameOut))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
        return err;
    }
    AddMessage(RGY_LOG_DEBUG, _T("allocated source buffer: %dx%d, pitch %d, %s.\n"),
        m_source.get(0)->frameinfo().width, m_source.get(0)->frameinfo().height, m_source.get(0)->frameinfo().pitch[0], RGY_CSP_NAMES[m_source.get(0)->frameinfo().csp]);

    if (RGY_ERR_NONE != (err = m_source.build(pAfsParam->frameOut))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build kernel for source cache: %s.\n"), get_err_mes(err));
        return err;
    }

    if (RGY_ERR_NONE != (err = build_analyze(pAfsParam->frameOut.csp, pAfsParam->afs.tb_order))) {
        return err;
    }

    if (RGY_ERR_NONE != (err = build_merge_scan())) {
        return err;
    }

    if (RGY_ERR_NONE != (err = m_scan.alloc(pAfsParam->frameOut))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
        return err;
    }
    AddMessage(RGY_LOG_DEBUG, _T("allocated scan buffer: %dx%d, pitch %d, %s.\n"),
        m_scan.get(0)->map->frame.width, m_scan.get(0)->map->frame.height, m_scan.get(0)->map->frame.pitch[0], RGY_CSP_NAMES[m_scan.get(0)->map->frame.csp]);

    if (RGY_ERR_NONE != (err = m_stripe.init(m_pLog))) {
        return err;
    }
    if (RGY_ERR_NONE != (err = m_stripe.alloc(pAfsParam->frameOut))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(err));
        return err;
    }
    AddMessage(RGY_LOG_DEBUG, _T("allocated stripe buffer: %dx%d, pitch %d, %s.\n"),
        m_stripe.get(0)->map->frame.width, m_stripe.get(0)->map->frame.height, m_stripe.get(0)->map->frame.pitch[0], RGY_CSP_NAMES[m_stripe.get(0)->map->frame.csp]);

    if (RGY_ERR_NONE != (err = build_synthesize(pAfsParam->frameOut.csp, pAfsParam->afs.analyze))) {
        return err;
    }

    m_queueAnalyze = m_cl->createQueue(m_cl->queue().devid(), m_cl->queue().getProperties());
    if (!m_queueAnalyze.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to createQueue.\n"));
        return RGY_ERR_UNKNOWN;
    }

    m_queueCopy = m_cl->createQueue(m_cl->queue().devid(), m_cl->queue().getProperties());
    if (!m_queueCopy.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to createQueue.\n"));
        return RGY_ERR_UNKNOWN;
    }

    pAfsParam->frameOut.picstruct = RGY_PICSTRUCT_FRAME;
    m_nFrame = 0;
    m_nPts = 0;
    m_pathThrough &= (~(FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_FLAGS));
    if (pAfsParam->afs.force24) {
        pAfsParam->baseFps *= rgy_rational<int>(4, 5);
    }

    if (pAfsParam->afs.timecode != 0) {
        const tstring tc_filename = PathRemoveExtensionS(pAfsParam->outFilename) + ((pAfsParam->afs.timecode == 2) ? _T(".timecode.afs.txt") : _T(".timecode.txt"));
        if (open_timecode(tc_filename)) {
            errno_t error = errno;
            AddMessage(RGY_LOG_ERROR, _T("failed to open timecode file \"%s\": %s.\n"), tc_filename.c_str(), _tcserror(error));
            return RGY_ERR_FILE_OPEN; // Couldn't open file
        }
        AddMessage(RGY_LOG_DEBUG, _T("opened timecode file \"%s\".\n"), tc_filename.c_str());
    }

    if (pAfsParam->afs.log) {
        const tstring log_filename = PathRemoveExtensionS(pAfsParam->outFilename) + _T(".afslog.csv");
        if (m_streamsts.open_log(log_filename)) {
            errno_t error = errno;
            AddMessage(RGY_LOG_ERROR, _T("failed to open afs log file \"%s\": %s.\n"), log_filename.c_str(), _tcserror(error));
            return RGY_ERR_FILE_OPEN; // Couldn't open file
        }
        AddMessage(RGY_LOG_DEBUG, _T("opened afs log file \"%s\".\n"), log_filename.c_str());
    }

    setFilterInfo(pParam->print());
    m_param = pParam;
    return sts;
}

tstring RGYFilterParamAfs::print() const {
    return afs.print();
}

bool RGYFilterAfs::scan_frame_result_cached(int frame, const VppAfs *pAfsPrm) {
    auto sp = m_scan.get(frame);
    const int mode = pAfsPrm->analyze == 0 ? 0 : 1;
    return sp->status > 0 && sp->frame == frame && sp->tb_order == pAfsPrm->tb_order && sp->thre_shift == pAfsPrm->thre_shift &&
        ((mode == 0) ||
        (mode == 1 && sp->mode == 1 && sp->thre_deint == pAfsPrm->thre_deint && sp->thre_Ymotion == pAfsPrm->thre_Ymotion && sp->thre_Cmotion == pAfsPrm->thre_Cmotion));
}

RGY_ERR RGYFilterAfs::scan_frame(int iframe, int force, const RGYFilterParamAfs *pAfsPrm, RGYOpenCLQueue &queue, std::vector<RGYOpenCLEvent> wait_event) {
    if (!force && scan_frame_result_cached(iframe, &pAfsPrm->afs)) {
        return RGY_ERR_NONE;
    }
    auto p1 = m_source.get(iframe-1);
    auto p0 = m_source.get(iframe);
    auto sp = m_scan.get(iframe);

    const int mode = pAfsPrm->afs.analyze == 0 ? 0 : 1;
    m_stripe.expire(iframe - 1);
    m_stripe.expire(iframe);
    sp->status = 1;
    sp->frame = iframe, sp->mode = mode, sp->tb_order = pAfsPrm->afs.tb_order;
    sp->thre_shift = pAfsPrm->afs.thre_shift, sp->thre_deint = pAfsPrm->afs.thre_deint;
    sp->thre_Ymotion = pAfsPrm->afs.thre_Ymotion, sp->thre_Cmotion = pAfsPrm->afs.thre_Cmotion;
    sp->clip.top = sp->clip.bottom = sp->clip.left = sp->clip.right = -1;
    auto err = analyze_stripe(p0, p1, sp, m_count_motion, pAfsPrm, queue, wait_event, m_eventScanFrame);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed analyze_stripe: %s.\n"), get_err_mes(err));
        return err;
    }
    if (STREAM_OPT) {
        err = m_count_motion->queueMapBuffer(m_queueCopy, CL_MAP_READ, { m_eventScanFrame });
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed m_count_motion.queueMapBuffer: %s.\n"), get_err_mes(err));
            return err;
        }
        sp->event = m_count_motion->mapEvent();
        sp = m_scan.get(iframe-1);
    }

    err = count_motion(sp, &pAfsPrm->afs.clip, queue);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed analyze_stripe: %s.\n"), get_err_mes(err));
        return err;
    }
    return err;
}

RGY_ERR RGYFilterAfs::count_motion(AFS_SCAN_DATA *sp, const AFS_SCAN_CLIP *clip, RGYOpenCLQueue &queue) {
    sp->clip = *clip;

    auto err = RGY_ERR_NONE;
    if (STREAM_OPT) {
        sp->event.wait();
    } else {
        err = m_count_motion->queueMapBuffer(queue, CL_MAP_READ, {});
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed m_count_motion.queueMapBuffer: %s.\n"), get_err_mes(err));
            return err;
        }
        m_count_motion->mapEvent().wait();
    }

    const int nSize = (int)(m_count_motion->size() / sizeof(uint32_t));
    int count0 = 0;
    int count1 = 0;
    const uint32_t *ptrCount = (uint32_t *)m_count_motion->mappedPtr();
    for (int i = 0; i < nSize; i++) {
        uint32_t count = ptrCount[i];
        count0 += count & 0xffff;
        count1 += count >> 16;
    }
    sp->ff_motion = count0;
    sp->lf_motion = count1;
    m_count_motion->unmapBuffer();
    //AddMessage(RGY_LOG_INFO, _T("count_motion[%6d]: %6d - %6d (ff,lf)"), sp->frame, sp->ff_motion, sp->lf_motion);
#if 0
    uint8_t *ptr = nullptr;
    if (RGY_ERR_NONE != (err = cudaMallocHost(&ptr, sp->map.frame.pitch * sp->map.frame.height))) {
        AddMessage(RGY_LOG_ERROR, _T("failed cudaMallocHost: %s.\n"), get_err_mes(err));
        return err;
    }
    if (RGY_ERR_NONE != (err = cudaMemcpy2D(ptr, sp->map.frame.pitch, sp->map.frame.ptr, sp->map.frame.pitch, sp->map.frame.width, sp->map.frame.height, cudaMemcpyDeviceToHost))) {
        AddMessage(RGY_LOG_ERROR, _T("failed cudaMemcpy2D: %s.\n"), get_err_mes(err));
        return err;
    }

    int motion_count[2] = { 0, 0 };
    afs_get_motion_count_simd(motion_count, ptr, &sp->clip, sp->map.frame.pitch, sp->map.frame.width, sp->map.frame.height, sp->tb_order);
    AddMessage((count0 == motion_count[0] && count1 == motion_count[1]) ? RGY_LOG_INFO : RGY_LOG_ERROR, _T("count_motion(ret, debug) = (%6d, %6d) / (%6d, %6d)\n"), count0, motion_count[0], count1, motion_count[1]);
    if (RGY_ERR_NONE != (err = cudaFreeHost(ptr))) {
        AddMessage(RGY_LOG_ERROR, _T("failed cudaFreeHost: %s.\n"), get_err_mes(err));
        return err;
    }
#endif
    return err;
}

RGY_ERR RGYFilterAfs::get_stripe_info(RGYOpenCLQueue &queue, int iframe, int mode, const RGYFilterParamAfs *pAfsPrm) {
    AFS_STRIPE_DATA *sp = m_stripe.get(iframe);
    if (sp->status > mode && sp->status < 4 && sp->frame == iframe) {
        if (sp->status == 2) {
            auto err = count_stripe(queue, sp, &pAfsPrm->afs.clip, pAfsPrm->afs.tb_order);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed count_stripe: %s.\n"), get_err_mes(err));
                return err;
            }
            sp->status = 3;
        }
        return RGY_ERR_NONE;
    }

    AFS_SCAN_DATA *sp0 = m_scan.get(iframe);
    AFS_SCAN_DATA *sp1 = m_scan.get(iframe + 1);
    auto err = merge_scan(sp, sp0, sp1, sp->buf_count_stripe, pAfsPrm, (STREAM_OPT) ? m_queueAnalyze : queue, {}, m_eventMergeScan);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed merge_scan: %s.\n"), get_err_mes(err));
        return err;
    }
    sp->status = 2;
    sp->frame = iframe;

    if (STREAM_OPT) {
        err = sp->buf_count_stripe->queueMapBuffer(m_queueCopy, CL_MAP_READ, { m_eventMergeScan });
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed m_count_motion.copyDtoHAsync: %s.\n"), get_err_mes(err));
            return err;
        }
        sp->event = sp->buf_count_stripe->mapEvent();
    } else {
        if (RGY_ERR_NONE != (err = count_stripe(queue, sp, &pAfsPrm->afs.clip, pAfsPrm->afs.tb_order))) {
            AddMessage(RGY_LOG_ERROR, _T("failed count_stripe: %s.\n"), get_err_mes(err));
            return err;
        }
        sp->status = 3;
    }
    return err;
}

RGY_ERR RGYFilterAfs::count_stripe(RGYOpenCLQueue &queue, AFS_STRIPE_DATA *sp, const AFS_SCAN_CLIP *clip, int tb_order) {
    auto err = RGY_ERR_NONE;
    if (STREAM_OPT) {
        sp->event.wait();
    } else {
        err = sp->buf_count_stripe->queueMapBuffer(queue, CL_MAP_READ, {});
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed m_count_stripe.queueMapBuffer: %s.\n"), get_err_mes(err));
            return err;
        }
        sp->buf_count_stripe->mapEvent().wait();
    }

    const int nSize = (int)(sp->buf_count_stripe->size() / sizeof(uint32_t));
    int count0 = 0;
    int count1 = 0;
    const uint32_t *ptrCount = (uint32_t *)sp->buf_count_stripe->mappedPtr();
    for (int i = 0; i < nSize; i++) {
        uint32_t count = ptrCount[i];
        count0 += count & 0xffff;
        count1 += count >> 16;
    }
    sp->count0 = count0;
    sp->count1 = count1;
    sp->buf_count_stripe->unmapBuffer();
    //AddMessage(RGY_LOG_INFO, _T("count_stripe[%6d]: %6d - %6d"), sp->frame, count0, count1);
#if 0
    uint8_t *ptr = nullptr;
    if (RGY_ERR_NONE != (err = cudaMallocHost(&ptr, sp->map.frame.pitch * sp->map.frame.height))) {
        AddMessage(RGY_LOG_ERROR, _T("failed cudaMallocHost: %s.\n"), get_err_mes(err));
        return err;
    }
    if (RGY_ERR_NONE != (err = cudaMemcpy2D(ptr, sp->map.frame.pitch, sp->map.frame.ptr, sp->map.frame.pitch, sp->map.frame.width, sp->map.frame.height, cudaMemcpyDeviceToHost))) {
        AddMessage(RGY_LOG_ERROR, _T("failed cudaMemcpy2D: %s.\n"), get_err_mes(err));
        return err;
    }

    int stripe_count[2] = { 0, 0 };
    afs_get_stripe_count_simd(stripe_count, ptr, clip, sp->map.frame.pitch, sp->map.frame.width, sp->map.frame.height, tb_order);
    AddMessage((count0 == stripe_count[0] && count1 == stripe_count[1]) ? RGY_LOG_INFO : RGY_LOG_ERROR, _T("count_stripe(ret, debug) = (%6d, %6d) / (%6d, %6d)\n"), count0, stripe_count[0], count1, stripe_count[1]);
    if (RGY_ERR_NONE != (err = cudaFreeHost(ptr))) {
        AddMessage(RGY_LOG_ERROR, _T("failed cudaFreeHost: %s.\n"), get_err_mes(err));
        return err;
    }
#else
    UNREFERENCED_PARAMETER(tb_order);
    UNREFERENCED_PARAMETER(clip);
#endif
    return err;
}

int RGYFilterAfs::detect_telecine_cross(int iframe, int coeff_shift) {
    using std::max;
    const AFS_SCAN_DATA *sp1 = m_scan.get(iframe - 1);
    const AFS_SCAN_DATA *sp2 = m_scan.get(iframe + 0);
    const AFS_SCAN_DATA *sp3 = m_scan.get(iframe + 1);
    const AFS_SCAN_DATA *sp4 = m_scan.get(iframe + 2);
    int shift = 0;

    if (max(absdiff(sp1->lf_motion + sp2->lf_motion, sp2->ff_motion),
        absdiff(sp3->ff_motion + sp4->ff_motion, sp3->lf_motion)) * coeff_shift >
        max3(absdiff(sp1->ff_motion + sp2->ff_motion, sp1->lf_motion),
            absdiff(sp2->ff_motion + sp3->ff_motion, sp2->lf_motion),
            absdiff(sp3->lf_motion + sp4->lf_motion, sp4->ff_motion)) * 256)
        if (max(sp2->lf_motion, sp3->ff_motion) * coeff_shift > sp2->ff_motion * 256)
            shift = 1;

    if (max(absdiff(sp1->lf_motion + sp2->lf_motion, sp2->ff_motion),
        absdiff(sp3->ff_motion + sp4->ff_motion, sp3->lf_motion)) * coeff_shift >
        max3(absdiff(sp1->ff_motion + sp2->ff_motion, sp1->lf_motion),
            absdiff(sp2->lf_motion + sp3->lf_motion, sp3->ff_motion),
            absdiff(sp3->lf_motion + sp4->lf_motion, sp4->ff_motion)) * 256)
        if (max(sp2->lf_motion, sp3->ff_motion) * coeff_shift > sp3->lf_motion * 256)
            shift = 1;

    return shift;
}

RGY_ERR RGYFilterAfs::analyze_frame(RGYOpenCLQueue &queue, int iframe, const RGYFilterParamAfs *pAfsPrm, int reverse[4], int assume_shift[4], int result_stat[4]) {
    for (int i = 0; i < 4; i++) {
        assume_shift[i] = detect_telecine_cross(iframe + i, pAfsPrm->afs.coeff_shift);
    }

    AFS_SCAN_DATA *scp = m_scan.get(iframe);
    const int scan_w = m_param->frameIn.width;
    const int scan_h = m_param->frameIn.height;
    int total = 0;
    if (scan_h - scp->clip.bottom - ((scan_h - scp->clip.top - scp->clip.bottom) & 1) > scp->clip.top && scan_w - scp->clip.right > scp->clip.left)
        total = (scan_h - scp->clip.bottom - ((scan_h - scp->clip.top - scp->clip.bottom) & 1) - scp->clip.top) * (scan_w - scp->clip.right - scp->clip.left);
    const int threshold = (total * pAfsPrm->afs.method_switch) >> 12;

    for (int i = 0; i < 4; i++) {
        auto err = get_stripe_info(queue, iframe + i, 0, pAfsPrm);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed on get_stripe_info(iframe=%d): %s.\n"), iframe + i, get_err_mes(err));
            return err;
        }
        AFS_STRIPE_DATA *stp = m_stripe.get(iframe + i);
        result_stat[i] = (stp->count0 * pAfsPrm->afs.coeff_shift > stp->count1 * 256) ? 1 : 0;
        if (threshold > stp->count1 && threshold > stp->count0)
            result_stat[i] += 2;
    }

    uint8_t status = AFS_STATUS_DEFAULT;
    if (result_stat[0] & 2)
        status |= assume_shift[0] ? AFS_FLAG_SHIFT0 : 0;
    else
        status |= (result_stat[0] & 1) ? AFS_FLAG_SHIFT0 : 0;
    if (reverse[0]) status ^= AFS_FLAG_SHIFT0;

    if (result_stat[1] & 2)
        status |= assume_shift[1] ? AFS_FLAG_SHIFT1 : 0;
    else
        status |= (result_stat[1] & 1) ? AFS_FLAG_SHIFT1 : 0;
    if (reverse[1]) status ^= AFS_FLAG_SHIFT1;

    if (result_stat[2] & 2)
        status |= assume_shift[2] ? AFS_FLAG_SHIFT2 : 0;
    else
        status |= (result_stat[2] & 1) ? AFS_FLAG_SHIFT2 : 0;
    if (reverse[2]) status ^= AFS_FLAG_SHIFT2;

    if (result_stat[3] & 2)
        status |= assume_shift[3] ? AFS_FLAG_SHIFT3 : 0;
    else
        status |= (result_stat[3] & 1) ? AFS_FLAG_SHIFT3 : 0;
    if (reverse[3]) status ^= AFS_FLAG_SHIFT3;

    const auto& frameinfo = m_source.get(iframe)->frameinfo();
    if (!interlaced(frameinfo)) {
        status |= AFS_FLAG_PROGRESSIVE;
        if (frameinfo.flags & RGY_FRAME_FLAG_RFF) status |= AFS_FLAG_RFF;
    }
    if (pAfsPrm->afs.drop) {
        if (interlaced(frameinfo)) status |= AFS_FLAG_FRAME_DROP;
        if (pAfsPrm->afs.smooth) status |= AFS_FLAG_SMOOTHING;
    }
    if (pAfsPrm->afs.force24) status |= AFS_FLAG_FORCE24;
    if (iframe < 1) status &= AFS_MASK_SHIFT0;

    m_status[iframe] = status;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterAfs::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue_main, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;

    auto pAfsParam = std::dynamic_pointer_cast<RGYFilterParamAfs>(m_param);
    if (!pAfsParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const int iframe = m_source.inframe();
    if (pInputFrame->ptr[0] == nullptr && m_nFrame >= iframe) {
        //終了
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return sts;
    } else if (pInputFrame->ptr[0] != nullptr) {
        //エラーチェック
        if (!m_analyze.get()) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build RGY_FILTER_AFS_ANALYZE_CL\n"));
            return RGY_ERR_OPENCL_CRUSH;
        }
        if (!m_mergeScan.get()) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build RGY_FILTER_AFS_MERGE_CL\n"));
            return RGY_ERR_OPENCL_CRUSH;
        }
        if (!m_stripe.kernelBuildSuccess()) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build RGY_FILTER_AFS_FILTER_CL\n"));
            return RGY_ERR_OPENCL_CRUSH;
        }
        if (!m_synthesize.get()) {
            AddMessage(RGY_LOG_ERROR, _T("failed to build RGY_FILTER_AFS_SYNTHESIZE_CL\n"));
            return RGY_ERR_OPENCL_CRUSH;
        }
        const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, m_frameBuf[0]->frame.mem_type);
        if (memcpyKind != RGYCLMemcpyD2D) {
            AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
            return RGY_ERR_INVALID_CALL;
        }
        if (m_param->frameOut.csp != m_param->frameIn.csp) {
            AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        //sourceキャッシュにコピー
        auto err = m_source.add(pInputFrame, queue_main, wait_events, m_eventSrcAdd);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to add frame to sorce buffer: %s.\n"), get_err_mes(err));
            return RGY_ERR_CUDA;
        }
        if (iframe == 0) {
            // scan_frame(p1 = -2, p0 = -1)のscan_frameも必要
            if (RGY_ERR_NONE != (err = scan_frame(iframe-1, false, pAfsParam.get(), queue_main, {}))) {
                AddMessage(RGY_LOG_ERROR, _T("failed on scan_frame(iframe-1=%d): %s.\n"), iframe-1, get_err_mes(err));
                return RGY_ERR_CUDA;
            }
        }
        if (RGY_ERR_NONE != (err = scan_frame(iframe, false, pAfsParam.get(), (STREAM_OPT) ? m_queueAnalyze : queue_main, { m_eventSrcAdd }))) {
            AddMessage(RGY_LOG_ERROR, _T("failed on scan_frame(iframe=%d): %s.\n"), iframe, get_err_mes(err));
            return RGY_ERR_CUDA;
        }
    }

    if (iframe >= 5) {
        int reverse[4] = { 0 }, assume_shift[4] = { 0 }, result_stat[4] = { 0 };
        auto err = analyze_frame(queue_main, iframe - 5, pAfsParam.get(), reverse, assume_shift, result_stat);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed on scan_frame(iframe=%d): %s.\n"), iframe - 5, get_err_mes(err));
            return RGY_ERR_CUDA;
        }
    }
    static const int preread_len = 3;
    //十分な数のフレームがたまった、あるいはdrainモードならフレームを出力
    if (iframe >= (5+preread_len+STREAM_OPT) || pInputFrame->ptr[0] == nullptr) {
        int reverse[4] = { 0 }, assume_shift[4] = { 0 }, result_stat[4] = { 0 };

        //m_streamsts.get_durationを呼ぶには、3フレーム先までstatusをセットする必要がある
        //そのため、analyze_frameを使って、3フレーム先までstatusを計算しておく
        for (int i = preread_len; i >= 0; i--) {
            //ここでは、これまで発行したanalyze_frameの結果からstatusの更新を行う(analyze_frameの内部で行われる)
            auto err = analyze_frame(queue_main, m_nFrame + i, pAfsParam.get(), reverse, assume_shift, result_stat);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error on analyze_frame(m_nFrame=%d, iframe=%d): %s.\n"),
                    m_nFrame, m_nFrame + i, iframe, get_err_mes(err));
                return RGY_ERR_CUDA;
            }
        }

        if (m_nFrame == 0) {
            //m_nFrame == 0のときは、下記がセットされていない
            for (int i = 0; i < preread_len; i++) {
                if (m_streamsts.set_status(i, m_status[i], i, m_source.get(i)->frameinfo().timestamp) != 0) {
                    AddMessage(RGY_LOG_ERROR, _T("failed to set afs_status(%d).\n"), i);
                    return RGY_ERR_CUDA;
                }
            }
        }
        //m_streamsts.get_durationを呼ぶには、3フレーム先までstatusをセットする必要がある
        {
            auto timestamp = m_source.get(m_nFrame+preread_len)->frameinfo().timestamp;
            //読み込まれた範囲を超える部分のtimestampは外挿する
            //こうしないと最終フレームのdurationが正しく計算されない
            if (m_nFrame+preread_len >= m_source.inframe()) {
                //1フレームの平均時間
                auto inframe_avg_duration = (m_source.get(m_source.inframe()-1)->frameinfo().timestamp + m_source.inframe() / 2) / m_source.inframe();
                //外挿するフレーム数をかけて足し込む
                timestamp += (m_nFrame+preread_len - (m_source.inframe()-1)) * inframe_avg_duration;
            }
            if (m_streamsts.set_status(m_nFrame+preread_len, m_status[m_nFrame+preread_len], 0, timestamp) != 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to set afs_status(%d).\n"), m_nFrame+preread_len);
                return RGY_ERR_CUDA;
            }
        }
        const auto afs_duration = m_streamsts.get_duration(m_nFrame);
        if (afs_duration == afsStreamStatus::AFS_SSTS_DROP) {
            //出力フレームなし
            *pOutputFrameNum = 0;
            ppOutputFrames[0] = nullptr;
        } else if (afs_duration < 0) {
            AddMessage(RGY_LOG_ERROR, _T("invalid call for m_streamsts.get_duration(%d).\n"), m_nFrame);
            return RGY_ERR_INVALID_CALL;
        } else {
            //出力先のフレーム
            RGYCLFrame *pOutFrame = nullptr;
            *pOutputFrameNum = 1;
            if (ppOutputFrames[0] == nullptr) {
                pOutFrame = m_frameBuf[0].get();
                ppOutputFrames[0] = &pOutFrame->frame;
            }

            if (pAfsParam->afs.timecode) {
                write_timecode(m_nPts, pAfsParam->outTimebase);
            }

            pOutFrame->frame.flags = m_source.get(m_nFrame)->frameinfo().flags & (~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_BFF | RGY_FRAME_FLAG_RFF_TFF));
            pOutFrame->frame.inputFrameId = m_source.get(m_nFrame)->frameinfo().inputFrameId;
            pOutFrame->frame.picstruct = RGY_PICSTRUCT_FRAME;
            pOutFrame->frame.duration = rational_rescale(afs_duration, pAfsParam->inTimebase, pAfsParam->outTimebase);
            pOutFrame->frame.timestamp = m_nPts;
            m_nPts += pOutFrame->frame.duration;

            //出力するフレームを作成
            get_stripe_info(queue_main, m_nFrame, 1, pAfsParam.get());
            RGY_ERR err = RGY_ERR_NONE;
            auto sip_filtered = m_stripe.filter(m_nFrame, pAfsParam->afs.analyze, queue_main, &err);
            if (sip_filtered == nullptr || err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("failed m_stripe.filter(m_nFrame=%d, iframe=%d): %s.\n"), m_nFrame, iframe - (5+preread_len), get_err_mes(err));
                return RGY_ERR_INVALID_CALL;
            }

            if (interlaced(m_source.get(m_nFrame)->frameinfo()) || pAfsParam->afs.tune) {
                err = synthesize(m_nFrame, pOutFrame, m_source.get(m_nFrame), m_source.get(m_nFrame-1), sip_filtered, pAfsParam.get(), queue_main);
            } else {
                err = m_source.copyFrame(pOutFrame, m_nFrame, queue_main, event);
            }
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error on synthesize(m_nFrame=%d, iframe=%d): %s.\n"), m_nFrame, iframe - (5+preread_len), get_err_mes(err));
                return RGY_ERR_CUDA;
            }
        }

        m_nFrame++;

        // drain中にdropが発生した場合には、次のフレームを出力するようにする
        if (pInputFrame->ptr[0] == nullptr && afs_duration == afsStreamStatus::AFS_SSTS_DROP) {
            return run_filter(pInputFrame, ppOutputFrames, pOutputFrameNum, queue_main, {}, event);
        }
    } else {
        //出力フレームなし
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
    }
    return sts;
}

int RGYFilterAfs::open_timecode(tstring tc_filename) {
    FILE *fp = NULL;
    if (_tfopen_s(&fp, tc_filename.c_str(), _T("w"))) {
        return 1;
    }
    m_fpTimecode = unique_ptr<FILE, fp_deleter>(fp, fp_deleter());
    fprintf(m_fpTimecode.get(), "# timecode format v2\n");
    return 0;
}

void RGYFilterAfs::write_timecode(int64_t pts, const rgy_rational<int>& timebase) {
    if (pts >= 0) {
        fprintf(m_fpTimecode.get(), "%.6lf\n", pts * timebase.qdouble() * 1000.0);
    }
}

void RGYFilterAfs::close() {
    m_eventSrcAdd.reset();
    m_eventScanFrame.reset();
    m_eventMergeScan.reset();
    m_nFrame = 0;
    m_nPts = 0;
    m_frameBuf.clear();
    m_source.clear();
    m_scan.clear();
    m_stripe.clear();
    m_status.clear();
    m_count_motion.reset();
    m_fpTimecode.reset();
    AddMessage(RGY_LOG_DEBUG, _T("closed afs filter.\n"));
}

static inline BOOL is_latter_field(int pos_y, int tb_order) {
    return ((pos_y & 1) == tb_order);
}

static void afs_get_stripe_count_simd(int *stripe_count, const uint8_t *ptr, const AFS_SCAN_CLIP *clip, int pitch, int scan_w, int scan_h, int tb_order) {
    static const uint8_t STRIPE_COUNT_CHECK_MASK[][16] = {
        { 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50 },
        { 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60 },
    };
    const int y_fin = scan_h - clip->bottom - ((scan_h - clip->top - clip->bottom) & 1);
    const uint32_t check_mask[2] = { 0x50, 0x60 };
    for (int pos_y = clip->top; pos_y < y_fin; pos_y++) {
        const uint8_t *sip = ptr + pos_y * pitch + clip->left;
        const int first_field_flag = !is_latter_field(pos_y, tb_order);
        const int x_count = scan_w - clip->right - clip->left;
#if defined(_M_IX86) || defined(_M_X64)
        __m128i xZero = _mm_setzero_si128();
        __m128i xMask, x0, x1;
        xMask = _mm_loadu_si128((const __m128i*)STRIPE_COUNT_CHECK_MASK[first_field_flag]);
        const uint8_t *sip_fin = sip + (x_count & ~31);
        for (; sip < sip_fin; sip += 32) {
            x0 = _mm_loadu_si128((const __m128i*)(sip +  0));
            x1 = _mm_loadu_si128((const __m128i*)(sip + 16));
            x0 = _mm_and_si128(x0, xMask);
            x1 = _mm_and_si128(x1, xMask);
            x0 = _mm_cmpeq_epi8(x0, xZero);
            x1 = _mm_cmpeq_epi8(x1, xZero);
            uint32_t count0 = _mm_movemask_epi8(x0);
            uint32_t count1 = _mm_movemask_epi8(x1);
            stripe_count[first_field_flag] += popcnt32(((count1 << 16) | count0));
        }
        if (x_count & 16) {
            x0 = _mm_loadu_si128((const __m128i*)sip);
            x0 = _mm_and_si128(x0, xMask);
            x0 = _mm_cmpeq_epi8(x0, xZero);
            uint32_t count0 = _mm_movemask_epi8(x0);
            stripe_count[first_field_flag] += popcnt32(count0);
            sip += 16;
        }
        sip_fin = sip + (x_count & 15);
        for (; sip < sip_fin; sip++)
            stripe_count[first_field_flag] += (!(*sip & check_mask[first_field_flag]));
#else
        if (first_field_flag) {
            for (int pos_x = 0; pos_x < x_count; pos_x++) {
                if (!(*sip & 0x50)) stripe_count[0]++;
                sip++;
            }
        } else {
            for (int pos_x = 0; pos_x < x_count; pos_x++) {
                if (!(*sip & 0x60)) stripe_count[1]++;
                sip++;
            }
        }
#endif
    }
}

static void afs_get_motion_count_simd(int *motion_count, const uint8_t *ptr, const AFS_SCAN_CLIP *clip, int pitch, int scan_w, int scan_h, int tb_order) {
    const int y_fin = scan_h - clip->bottom - ((scan_h - clip->top - clip->bottom) & 1);
    for (int pos_y = clip->top; pos_y < y_fin; pos_y++) {
        const uint8_t *sip = ptr + pos_y * pitch + clip->left;
        const int x_count = scan_w - clip->right - clip->left;
        const int is_latter_feild = is_latter_field(pos_y, tb_order);
#if defined(_M_IX86) || defined(_M_X64)
        __m128i xMotion = _mm_set1_epi8(0x40);
        __m128i x0, x1;
        const uint8_t *sip_fin = sip + (x_count & ~31);
        for (; sip < sip_fin; sip += 32) {
            x0 = _mm_loadu_si128((const __m128i*)(sip +  0));
            x1 = _mm_loadu_si128((const __m128i*)(sip + 16));
            x0 = _mm_andnot_si128(x0, xMotion);
            x1 = _mm_andnot_si128(x1, xMotion);
            x0 = _mm_cmpeq_epi8(x0, xMotion);
            x1 = _mm_cmpeq_epi8(x1, xMotion);
            uint32_t count0 = _mm_movemask_epi8(x0);
            uint32_t count1 = _mm_movemask_epi8(x1);
            motion_count[is_latter_feild] += popcnt32(((count1 << 16) | count0));
        }
        if (x_count & 16) {
            x0 = _mm_loadu_si128((const __m128i*)sip);
            x0 = _mm_andnot_si128(x0, xMotion);
            x0 = _mm_cmpeq_epi8(x0, xMotion);
            uint32_t count0 = _mm_movemask_epi8(x0);
            motion_count[is_latter_feild] += popcnt32(count0);
            sip += 16;
        }
        sip_fin = sip + (x_count & 15);
        for (; sip < sip_fin; sip++)
            motion_count[is_latter_feild] += ((~*sip & 0x40) >> 6);
#else
        if (is_latter_feild) {
            for (int pos_x = 0; pos_x < x_count; pos_x++) {
                motion_count[1] += ((~*sip & 0x40) >> 6);
                sip++;
            }
        } else {
            for (int pos_x = 0; pos_x < x_count; pos_x++) {
                motion_count[0] += ((~*sip & 0x40) >> 6);
                sip++;
            }
        }
#endif
    }
}