// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
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

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include "convert_csp.h"
#include "rgy_avutil.h"
#include "rgy_filesystem.h"
#include "rgy_filter_ivtc.h"

static const int IVTC_BLOCK_X = 32;
static const int IVTC_BLOCK_Y = 8;
static const int IVTC_CACHE_SIZE = 3;
static_assert((IVTC_BLOCK_X * IVTC_BLOCK_Y & (IVTC_BLOCK_X * IVTC_BLOCK_Y - 1)) == 0,
    "IVTC_BLOCK_X * IVTC_BLOCK_Y must be a power of 2 for OpenCL WG reduction");

RGYFilterIvtc::RGYFilterIvtc(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context),
    m_ivtc(),
    m_cacheFrames(),
    m_scoreBuf(),
    m_scoreHost(),
    m_diffBuf(),
    m_diffHost(),
    m_cycleInPts(),
    m_cycleInDur(),
    m_cycleInputIds(),
    m_cycleMatchScore(),
    m_cycleCombScore(),
    m_cycleMatchType(),
    m_cycleApplyBlend(),
    m_cycleDiffPrev(),
    m_emitQueue(),
    m_stagingBase(0),
    m_nPts(0),
    m_nPtsInit(false),
    m_cfrBaseDur(0),
    m_hasSaveSlot(false),
    m_cycleFilled(0),
    m_outputFrameCount(0),
    m_inputCount(0),
    m_drained(false),
    m_tffDefault(1),
    m_fpLog() {
    m_name = _T("ivtc");
}

RGYFilterIvtc::~RGYFilterIvtc() {
    close();
}

RGY_ERR RGYFilterIvtc::checkParam(const std::shared_ptr<RGYFilterParamIvtc> pParam) {
    if (pParam->frameOut.height <= 0 || pParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->ivtc.combThresh < 0.0f || pParam->ivtc.combThresh > 1.0f) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid combthresh %.3f: must be in [0.0, 1.0].\n"), pParam->ivtc.combThresh);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->ivtc.cleanFrac < 0.0f || pParam->ivtc.cleanFrac > 1.0f) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid cleanfrac %.3f: must be in [0.0, 1.0].\n"), pParam->ivtc.cleanFrac);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->ivtc.guide != 0 && pParam->ivtc.guide != 1) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid guide=%d: only 0 and 1 are supported.\n"), pParam->ivtc.guide);
        return RGY_ERR_INVALID_PARAM;
    }
    // post=1 (metrics-only) not implemented; use post=0 or post=2
    if (pParam->ivtc.post != 0 && pParam->ivtc.post != 2) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid post=%d: post=1 (metrics-only) not implemented; use post=0 or post=2.\n"), pParam->ivtc.post);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->ivtc.cycle > 16 || (pParam->ivtc.cycle > 0 && pParam->ivtc.cycle < 2)) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid cycle=%d: must be -1 (auto), 0 (disabled) or in [2, 16].\n"), pParam->ivtc.cycle);
        return RGY_ERR_INVALID_PARAM;
    }
    if (pParam->ivtc.cycle > 0 && pParam->ivtc.drop != 1) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid drop=%d: only drop=1 is supported in this build.\n"), pParam->ivtc.drop);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterIvtc::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamIvtc>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    // 出力は progressive 扱い (field-matching の結果として)
    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;

    // cycle=-1 (auto) の場合、入力 fps を見てデシメーションの ON/OFF を決める。
    // 入力が既に 24fps 近辺なら (例: MPEG-2 DVD を --vpp-rff なしで読んだ場合や、
    // 元々 film-transferred な素材)、デシメーションすると duplicate pair が存在せず
    // 誤って正常フレームを落としてしまう。26fps を閾値に、~24/23.976 は auto-off とする。
    if (prm->ivtc.cycle < 0) {
        const double inputFps = (prm->baseFps.d() > 0) ? (double)prm->baseFps.n() / (double)prm->baseFps.d() : 0.0;
        if (inputFps > 28.0 && inputFps < 32.0) {
            AddMessage(RGY_LOG_DEBUG, _T("ivtc: input is %.3f fps (~30fps), enabling 3:2 pulldown decimation (cycle=5).\n"), inputFps);
            prm->ivtc.cycle = 5;
        } else {
            AddMessage(RGY_LOG_INFO, _T("ivtc: input is %.3f fps, decimation skipped (auto enables only for ~30fps).\n"), inputFps);
            prm->ivtc.cycle = 0;
        }
    }

    // IVTC は「現在入力」ではなく「中央フレーム」相当を出力するため、
    // frame metadata は常に手動管理する。
    m_pathThrough &= ~(FILTER_PATHTHROUGH_TIMESTAMP | FILTER_PATHTHROUGH_FLAGS | FILTER_PATHTHROUGH_PICSTRUCT | FILTER_PATHTHROUGH_DATA);

    // デシメーションが有効な場合は出力 baseFps を更新する。
    if (prm->ivtc.cycle > 0) {
        pParam->baseFps *= rgy_rational<int>(prm->ivtc.cycle - prm->ivtc.drop, prm->ivtc.cycle);
    }

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamIvtc>(m_param);
    if (!m_ivtc.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]) {
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d -D ivtc_block_x=%d -D ivtc_block_y=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp],
            IVTC_BLOCK_X, IVTC_BLOCK_Y);
        m_ivtc.set(m_cl->buildResourceAsync(_T("RGY_FILTER_IVTC_CL"), _T("EXE_DATA"), options.c_str()));
    }

    // m_frameBuf レイアウト (cycle>0 時):
    //   [0 .. cycleLen-1]            : サイクル蓄積バッファ
    //   [cycleLen]                   : 前サイクル末尾の保存スロット (SAD 比較用)
    //   [cycleLen+1 .. stagingEnd-1] : emit-staging (cycleLen - drop 枚)
    // AFS/Decimate と同じく run_filter は 1 call 1 emit を守るため、flushCycle が
    // ここで決まった emit 候補を staging にコピーして m_emitQueue に積み、後続 call で
    // 1 枚ずつ popEmit する。次サイクルは cycle[0..cycleLen-1] を安心して上書きできる。
    const int cycleLen = std::max(prm->ivtc.cycle, 0);
    const int stagingCount = (cycleLen > 0) ? (cycleLen - prm->ivtc.drop) : 0;
    const int bufCount = (cycleLen > 0) ? (cycleLen + 1 + stagingCount) : 1;
    m_stagingBase = (cycleLen > 0) ? (cycleLen + 1) : 0;
    sts = AllocFrameBuf(prm->frameOut, bufCount);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // prev/cur/next 用にリングバッファを確保 (元フレームはインタレ扱いのまま保持)
    if ((int)m_cacheFrames.size() != IVTC_CACHE_SIZE
        || !prmPrev
        || cmpFrameInfoCspResolution(&m_cacheFrames[0]->frame, &prm->frameIn)) {
        m_cacheFrames.clear();
        for (int i = 0; i < IVTC_CACHE_SIZE; i++) {
            auto clframe = m_cl->createFrameBuffer(prm->frameIn);
            if (!clframe) {
                AddMessage(RGY_LOG_ERROR, _T("failed to allocate cache frame %d.\n"), i);
                return RGY_ERR_MEMORY_ALLOC;
            }
            m_cacheFrames.push_back(std::move(clframe));
        }
    }

    // スコア集計バッファ: WG ごとに 6 uints = [mC, mP, mN, cC, cP, cN]
    //   mX = match-quality (3-top vs 2-bot pattern diff sum per block)
    //   cX = combing-count (zigzag pixels per block)
    // CPU 側で全 WG 値の MAX を取って block-max メトリックを得る
    // (based on published 3:2 pulldown detection algorithm).
    const int wg_count_x = (prm->frameIn.width  + IVTC_BLOCK_X - 1) / IVTC_BLOCK_X;
    const int wg_count_y = (prm->frameIn.height + IVTC_BLOCK_Y - 1) / IVTC_BLOCK_Y;
    const size_t wg_count = (size_t)wg_count_x * wg_count_y;
    const size_t score_count = wg_count * 6;
    if (!m_scoreBuf || m_scoreHost.size() != score_count) {
        m_scoreBuf = m_cl->createBuffer(score_count * sizeof(uint32_t), CL_MEM_READ_WRITE);
        if (!m_scoreBuf) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate score buffer.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_scoreHost.assign(score_count, 0u);
    }
    // SAD 集計用バッファ (WGごとに uint×1)
    if (cycleLen > 0 && (!m_diffBuf || m_diffHost.size() != wg_count)) {
        m_diffBuf = m_cl->createBuffer(wg_count * sizeof(uint32_t), CL_MEM_READ_WRITE);
        if (!m_diffBuf) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate diff buffer.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_diffHost.assign(wg_count, 0u);
    }
    // サイクルのメタデータ用ベクトル
    if (cycleLen > 0) {
        m_cycleInPts.assign(cycleLen, 0);
        m_cycleInDur.assign(cycleLen, 0);
        m_cycleInputIds.assign(cycleLen, 0);
        m_cycleMatchScore.assign(cycleLen, 0);
        m_cycleCombScore.assign(cycleLen, 0);
        m_cycleMatchType.assign(cycleLen, 0);
        m_cycleApplyBlend.assign(cycleLen, 0);
        m_cycleDiffPrev.assign(cycleLen, 0);
    }

    // auto 時に各フレームの picstruct から field order を参照し、
    // 情報がなければここで決めた既定値へフォールバックする。
    if (prm->frameIn.picstruct & RGY_PICSTRUCT_BFF) {
        m_tffDefault = 0;
    } else {
        m_tffDefault = 1;
    }

    if (!prmPrev
        || prmPrev->ivtc != prm->ivtc
        || prmPrev->outFilename != prm->outFilename) {
        m_fpLog.reset();
    }
    if (!prm->ivtc.log) {
        m_fpLog.reset();
    }
    if (prm->ivtc.log && !m_fpLog) {
        const auto logPath = (prm->ivtc.logPath.length() > 0)
            ? prm->ivtc.logPath
            : PathRemoveExtensionS(prm->outFilename) + _T(".ivtc.log.txt");
        prm->ivtc.logPath = logPath;
        m_fpLog = std::unique_ptr<FILE, fp_deleter>(_tfopen(logPath.c_str(), _T("w")), fp_deleter());
        if (m_fpLog) {
            fprintf(m_fpLog.get(), "#out_idx\tin_id\tmatch\tpost\tstatus\tmQ\tcComb\tdiff_to_prev\n");
            fflush(m_fpLog.get());
            AddMessage(RGY_LOG_DEBUG, _T("opened ivtc log file \"%s\".\n"), logPath.c_str());
        } else {
            AddMessage(RGY_LOG_ERROR, _T("failed to open ivtc log file \"%s\".\n"), logPath.c_str());
            return RGY_ERR_FILE_OPEN;
        }
    }

    // 状態リセット (ストリーム切り替え時など)
    if (!prmPrev || prmPrev->ivtc != prm->ivtc) {
        m_inputCount = 0;
        m_drained = false;
        m_cycleFilled = 0;
        m_hasSaveSlot = false;
        m_outputFrameCount = 0;
        m_emitQueue.clear();
        m_nPts = 0;
        m_nPtsInit = false;
        m_cfrBaseDur = 0;
    }

    setFilterInfo(prm->print() + _T("\n                         tff=") + ((prm->ivtc.tff >= 0) ? (prm->ivtc.tff ? _T("on") : _T("off")) : (m_tffDefault ? _T("auto(default=tff)") : _T("auto(default=bff)"))));
    m_param = prm;
    return sts;
}

int RGYFilterIvtc::getTff(const RGYFrameInfo *frame) const {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamIvtc>(m_param);
    if (prm && prm->ivtc.tff >= 0) {
        return prm->ivtc.tff;
    }
    if (frame) {
        if ((frame->picstruct & RGY_PICSTRUCT_FRAME_BFF) == RGY_PICSTRUCT_FRAME_BFF) {
            return 0;
        }
        if ((frame->picstruct & RGY_PICSTRUCT_FRAME_TFF) == RGY_PICSTRUCT_FRAME_TFF) {
            return 1;
        }
        if (frame->picstruct & RGY_PICSTRUCT_BFF) {
            return 0;
        }
        if (frame->picstruct & RGY_PICSTRUCT_TFF) {
            return 1;
        }
    }
    return m_tffDefault;
}

RGY_ERR RGYFilterIvtc::scoreCandidates(const RGYFrameInfo *prev, const RGYFrameInfo *cur, const RGYFrameInfo *next, int tff, uint64_t matchScoreOut[3], uint64_t combScoreOut[3], RGYOpenCLQueue &queue) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamIvtc>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    // 8bit 基準の閾値 (nt=10, T=4) を入力 bitdepth にスケール。
    const int bitDepth = RGY_CSP_BIT_DEPTH[cur->csp];
    const int maxVal = (1 << bitDepth) - 1;
    const int nt = std::max(1, (maxVal * 10) / 255);
    const int T  = std::max(1, (int)std::round(maxVal * prm->ivtc.combThresh));

    const char *kernel_name = "kernel_ivtc_score_candidates";
    RGYWorkSize local(IVTC_BLOCK_X, IVTC_BLOCK_Y);
    RGYWorkSize global(cur->width, cur->height);

    auto err = m_ivtc.get()->kernel(kernel_name).config(queue, local, global).launch(
        (cl_mem)prev->ptr[0], (cl_mem)cur->ptr[0], (cl_mem)next->ptr[0],
        cur->pitch[0], cur->width, cur->height,
        tff, nt, T,
        m_scoreBuf->mem());
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"), char_to_tstring(kernel_name).c_str(), get_err_mes(err));
        return err;
    }

    // 同期リードバック。block-max アグリゲーション (MAX across WGs) を CPU 側で実行。
    const size_t score_bytes = m_scoreHost.size() * sizeof(uint32_t);
    auto clerr = clEnqueueReadBuffer(queue.get(), m_scoreBuf->mem(), CL_TRUE, 0, score_bytes, m_scoreHost.data(), 0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("clEnqueueReadBuffer failed: %s.\n"), cl_errmes(clerr));
        return err_cl_to_rgy(clerr);
    }

    uint64_t max_mC = 0, max_mP = 0, max_mN = 0;
    uint64_t max_cC = 0, max_cP = 0, max_cN = 0;
    const size_t wg_count = m_scoreHost.size() / 6;
    for (size_t i = 0; i < wg_count; i++) {
        const uint32_t *e = &m_scoreHost[i * 6];
        if (e[0] > max_mC) max_mC = e[0];
        if (e[1] > max_mP) max_mP = e[1];
        if (e[2] > max_mN) max_mN = e[2];
        if (e[3] > max_cC) max_cC = e[3];
        if (e[4] > max_cP) max_cP = e[4];
        if (e[5] > max_cN) max_cN = e[5];
    }
    matchScoreOut[0] = max_mC; matchScoreOut[1] = max_mP; matchScoreOut[2] = max_mN;
    combScoreOut [0] = max_cC; combScoreOut [1] = max_cP; combScoreOut [2] = max_cN;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterIvtc::synthesizeToCycle(int cycleSlot, const RGYFrameInfo *prev, const RGYFrameInfo *cur, const RGYFrameInfo *next, int tff, const IvtcMatch match, const bool applyBlend, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamIvtc>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    RGYFrameInfo *pOutputFrame = &m_frameBuf[cycleSlot]->frame;
    const char *kernel_name = "kernel_ivtc_synthesize";
    const int bitDepth = RGY_CSP_BIT_DEPTH[cur->csp];
    const int maxVal = (1 << bitDepth) - 1;
    const int T  = std::max(1, (int)std::round(maxVal * prm->ivtc.combThresh));
    const int planes = RGY_CSP_PLANES[cur->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto planeOut = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const auto planePrev = getPlane(prev, (RGY_PLANE)iplane);
        const auto planeCur  = getPlane(cur,  (RGY_PLANE)iplane);
        const auto planeNext = getPlane(next, (RGY_PLANE)iplane);
        RGYWorkSize local(IVTC_BLOCK_X, IVTC_BLOCK_Y);
        RGYWorkSize global(planeOut.width, planeOut.height);
        const std::vector<RGYOpenCLEvent> &waitHere = (iplane == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        auto err = m_ivtc.get()->kernel(kernel_name).config(queue, local, global, waitHere, nullptr).launch(
            (cl_mem)planeOut.ptr[0], planeOut.pitch[0], planeOut.width, planeOut.height,
            (cl_mem)planePrev.ptr[0], (cl_mem)planeCur.ptr[0], (cl_mem)planeNext.ptr[0],
            planeCur.pitch[0],
            tff, (int)match,
            applyBlend ? 1 : 0, T);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s (plane %d): %s.\n"), char_to_tstring(kernel_name).c_str(), iplane, get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterIvtc::computePairDiff(const RGYFrameInfo *pA, const RGYFrameInfo *pB, uint64_t &diffOut, RGYOpenCLQueue &queue) {
    const char *kernel_name = "kernel_ivtc_frame_diff";
    RGYWorkSize local(IVTC_BLOCK_X, IVTC_BLOCK_Y);
    RGYWorkSize global(pA->width, pA->height);
    auto err = m_ivtc.get()->kernel(kernel_name).config(queue, local, global).launch(
        (cl_mem)pA->ptr[0], (cl_mem)pB->ptr[0],
        pA->pitch[0], pA->width, pA->height,
        m_diffBuf->mem());
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"), char_to_tstring(kernel_name).c_str(), get_err_mes(err));
        return err;
    }
    const size_t bytes = m_diffHost.size() * sizeof(uint32_t);
    auto clerr = clEnqueueReadBuffer(queue.get(), m_diffBuf->mem(), CL_TRUE, 0, bytes, m_diffHost.data(), 0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("clEnqueueReadBuffer (diff) failed: %s.\n"), cl_errmes(clerr));
        return err_cl_to_rgy(clerr);
    }
    uint64_t sum = 0;
    for (const uint32_t v : m_diffHost) sum += v;
    diffOut = sum;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterIvtc::processInputToCycle(int idx_prev, int idx_cur, int idx_next, RGYOpenCLQueue &queue_main, const std::vector<RGYOpenCLEvent> &wait_events) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamIvtc>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int cycleLen = std::max(prm->ivtc.cycle, 0);
    const int slot = (cycleLen > 0) ? m_cycleFilled : 0;

    // 1. 候補スコア計算 (match-quality と combing-count を独立に算出 = block-max メトリック)
    uint64_t matchScore[3] = { 0, 0, 0 };
    uint64_t combScore [3] = { 0, 0, 0 };
    const RGYFrameInfo &curInfo = m_cacheFrames[idx_cur]->frame;
    const int tff = getTff(&curInfo);
    auto err = scoreCandidates(
        &m_cacheFrames[idx_prev]->frame,
        &m_cacheFrames[idx_cur ]->frame,
        &m_cacheFrames[idx_next]->frame,
        tff,
        matchScore, combScore, queue_main);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    // block-max ベースの「クリーン閾値」: 1 ブロック内で何ピクセルが combing 判定されたら汚いとするか。
    // 旧実装は全フレーム総和ピクセル数に対して cleanFrac を掛けていたが、block-max 化に合わせて
    // 1 ブロックあたり IVTC_BLOCK_X*IVTC_BLOCK_Y * cleanFrac ピクセルを許容上限とする。
    const uint64_t blockPixels = (uint64_t)IVTC_BLOCK_X * (uint64_t)IVTC_BLOCK_Y;
    const uint64_t cleanBlockThresh = std::max<uint64_t>(1, (uint64_t)((double)blockPixels * prm->ivtc.cleanFrac));

    // 2. マッチ選択: match-quality の argmin で C / P / N を選ぶ
    //    (based on published 3:2 pulldown detection algorithm)。
    //    guide=1 のとき、C の combing-count が cleanBlockThresh 以下なら既にクリーンなので
    //    無駄な P/N への切り替えを避ける (pattern hint の簡易版)。
    //    さらに「C に 1 ピクセルの combing も検出されない (= 既に完全 progressive)」
    //    ケースは、ソフトテレシネ素材のデコーダ出力に当てはまる状況で、ここで P/N に
    //    切り替えると別時刻のフィールドを混ぜて逆に combing を生成してしまう。
    //    guide に依存しない hard-guard として C をロックする。
    IvtcMatch match = IvtcMatch::C;
    uint64_t chosenMatchScore = matchScore[(int)IvtcMatch::C];
    const bool cIsZeroComb = (combScore[(int)IvtcMatch::C] == 0);
    const bool cIsGuideClean = (prm->ivtc.guide == 1 && combScore[(int)IvtcMatch::C] <= cleanBlockThresh);
    if (cIsZeroComb || cIsGuideClean) {
        match = IvtcMatch::C;
        chosenMatchScore = matchScore[(int)IvtcMatch::C];
    } else if (prm->ivtc.guide == 1) {
        if (matchScore[(int)IvtcMatch::P] <= matchScore[(int)IvtcMatch::N]) {
            match = IvtcMatch::P;
            chosenMatchScore = matchScore[(int)IvtcMatch::P];
        } else {
            match = IvtcMatch::N;
            chosenMatchScore = matchScore[(int)IvtcMatch::N];
        }
    } else {
        if (matchScore[(int)IvtcMatch::P] < chosenMatchScore) {
            chosenMatchScore = matchScore[(int)IvtcMatch::P];
            match = IvtcMatch::P;
        }
        if (matchScore[(int)IvtcMatch::N] < chosenMatchScore) {
            chosenMatchScore = matchScore[(int)IvtcMatch::N];
            match = IvtcMatch::N;
        }
    }

    // 3. post=2 ブレンド発火判定は「選択マッチ後も残っている combing 量」で判定する。
    //    match-quality ではなく combing-count を使うのがポイント (post=2 はピクセル単位の
    //    凸凹を平滑するための後処理で、residual comb に反応すべきなので)。
    //    発火時は synthesize カーネル側で second-field 行について per-pixel のコーミング
    //    判定を行い、combed と検出されたピクセルのみ bob 補間 (上下平均) に置換する。
    //    first-field 行は常にそのまま保持され、解像度劣化がない。
    const uint64_t chosenCombScore = combScore[(int)match];
    const bool applyBlend = (prm->ivtc.post >= 2) && (chosenCombScore > cleanBlockThresh);

    // 4. マッチ結果を cycle スロット (= m_frameBuf[slot]) に合成
    err = synthesizeToCycle(slot,
        &m_cacheFrames[idx_prev]->frame,
        &m_cacheFrames[idx_cur ]->frame,
        &m_cacheFrames[idx_next]->frame,
        tff,
        match, applyBlend, queue_main, wait_events);
    if (err != RGY_ERR_NONE) {
        return err;
    }

    // 5. 入力メタデータを保持 (出力時に使うため)
    m_frameBuf[slot]->frame.picstruct = RGY_PICSTRUCT_FRAME;
    m_frameBuf[slot]->frame.flags     = curInfo.flags & (~(RGY_FRAME_FLAG_RFF | RGY_FRAME_FLAG_RFF_COPY | RGY_FRAME_FLAG_RFF_BFF | RGY_FRAME_FLAG_RFF_TFF));
    m_frameBuf[slot]->frame.dataList  = curInfo.dataList;
    if (cycleLen > 0) {
        m_cycleInPts[slot]      = curInfo.timestamp;
        m_cycleInDur[slot]      = curInfo.duration;
        m_cycleInputIds[slot]   = curInfo.inputFrameId;
        m_cycleMatchScore[slot] = chosenMatchScore;
        m_cycleCombScore[slot]  = chosenCombScore;
        m_cycleMatchType[slot]  = (int)match;
        m_cycleApplyBlend[slot] = applyBlend ? 1 : 0;

        // 6. 直前フレームとの SAD を計算。slot==0 の場合のみ前サイクル末尾 (保存スロット) と比較。
        uint64_t diff = 0;
        if (slot > 0) {
            err = computePairDiff(&m_frameBuf[slot]->frame, &m_frameBuf[slot - 1]->frame, diff, queue_main);
        } else if (m_hasSaveSlot) {
            err = computePairDiff(&m_frameBuf[slot]->frame, &m_frameBuf[cycleLen]->frame, diff, queue_main);
        } else {
            // 最初のサイクルの 1 フレーム目。比較対象がないので、decimate 候補から除外するため
            // 最大値を入れておく (最小値で選ばれる drop 候補にならない)。
            diff = std::numeric_limits<uint64_t>::max();
        }
        if (err != RGY_ERR_NONE) {
            return err;
        }
        m_cycleDiffPrev[slot] = diff;
        m_cycleFilled++;
    } else {
        // cycle=0 の場合は即座にこの1枚を出力するため、メタをそのまま残す。
        m_frameBuf[slot]->frame.timestamp    = curInfo.timestamp;
        m_frameBuf[slot]->frame.duration     = curInfo.duration;
        m_frameBuf[slot]->frame.inputFrameId = curInfo.inputFrameId;

        // cycle=0 パスでは flushCycle を通らないので、per-frame ログもここで書く。
        if (m_fpLog) {
            const char *matchStr = ((int)match == 0) ? "c" : ((int)match == 1) ? "p" : "n";
            fprintf(m_fpLog.get(), "%d\t%d\t%s\t%s\t%s\t%llu\t%llu\t%llu\n",
                m_outputFrameCount,
                curInfo.inputFrameId,
                matchStr,
                applyBlend ? "blend" : "     ",
                "emit ",
                (unsigned long long)chosenMatchScore,
                (unsigned long long)chosenCombScore,
                0ull);
            fflush(m_fpLog.get());
        }
        m_outputFrameCount++;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterIvtc::flushCycle(bool finalFlush, int64_t nextInputPts, RGYOpenCLQueue &queue_main) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamIvtc>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int cycleLen = std::max(prm->ivtc.cycle, 0);
    const int filled = m_cycleFilled;
    if (filled <= 0) {
        return RGY_ERR_NONE;
    }

    // Drop 対象の決定: SAD の最小を持つ frame を落とす (3:2 プルダウン由来の重複)。
    // 完全サイクル (filled == cycleLen) 時のみ drop する。部分サイクル (finalFlush) は
    // shippable な情報だけ出す。
    int dropIdx = -1;
    if (cycleLen > 0 && filled == cycleLen && prm->ivtc.drop >= 1) {
        uint64_t minDiff = std::numeric_limits<uint64_t>::max();
        for (int i = 0; i < filled; i++) {
            if (m_cycleDiffPrev[i] < minDiff) {
                minDiff = m_cycleDiffPrev[i];
                dropIdx = i;
            }
        }
    }
    const int dropCount = (dropIdx >= 0) ? 1 : 0;
    const int emitCount = filled - dropCount;

    // CFR 出力用の「均一 duration」を算出する。
    //   baseDur = 今サイクル totalCycleDur / emitCount (= 出力 emit あたり duration)
    //   最初の「完全サイクル」(filled == cycleLen かつ drop 有効) で確定させた baseDur を
    //   m_cfrBaseDur にキャッシュし、以降のサイクル (部分 flush / drain 含む) は常にこの値を使う。
    //   こうすることで drain 末尾の短いサイクルや入力 jitter があっても、
    //   全出力フレームの duration が完全に同一になり、MediaInfo は CFR と判定する。
    // PTS は m_nPts アキュムレータから発行 (単調増加・ユニーク保証)。初回のみ最初の入力 PTS を seed。
    bool ptsInvalid = false;
    for (int i = 0; i < filled; i++) {
        if (m_cycleInPts[i] == AV_NOPTS_VALUE) { ptsInvalid = true; break; }
    }
    int64_t baseDur = m_cfrBaseDur;
    if (baseDur <= 0) {
        int64_t totalCycleDur = 0;
        if (!ptsInvalid && nextInputPts != AV_NOPTS_VALUE && filled >= 1) {
            totalCycleDur = nextInputPts - m_cycleInPts[0];
        }
        if (totalCycleDur <= 0) {
            for (int i = 0; i < filled; i++) totalCycleDur += m_cycleInDur[i];
        }
        baseDur = (emitCount > 0 && totalCycleDur > 0) ? (totalCycleDur / emitCount) : 0;
        if (baseDur <= 0) baseDur = 1;
        // 最初の完全サイクルなら値を確定させる。
        if (filled == cycleLen && dropIdx >= 0) {
            m_cfrBaseDur = baseDur;
        }
    }

    if (!m_nPtsInit && emitCount > 0) {
        m_nPts = ptsInvalid ? 0 : m_cycleInPts[0];
        m_nPtsInit = true;
    }

    // emit されるフレームを staging にコピーし、IvtcEmitEntry を m_emitQueue に積む。
    // staging 領域は m_frameBuf[m_stagingBase .. m_stagingBase+stagingCount-1] (cycleLen-drop 枚)。
    int emitted = 0;
    for (int i = 0; i < filled; i++) {
        if (i == dropIdx) continue;
        const int stagingIdx = m_stagingBase + emitted;
        if (stagingIdx >= (int)m_frameBuf.size()) {
            AddMessage(RGY_LOG_ERROR, _T("ivtc staging overflow: idx=%d size=%zu.\n"), stagingIdx, m_frameBuf.size());
            return RGY_ERR_UNKNOWN;
        }
        auto cpErr = m_cl->copyFrame(&m_frameBuf[stagingIdx]->frame, &m_frameBuf[i]->frame, nullptr, queue_main);
        if (cpErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy emit frame to staging[%d]: %s.\n"), stagingIdx, get_err_mes(cpErr));
            return cpErr;
        }
        m_frameBuf[stagingIdx]->frame.picstruct = m_frameBuf[i]->frame.picstruct;
        m_frameBuf[stagingIdx]->frame.flags = m_frameBuf[i]->frame.flags;
        m_frameBuf[stagingIdx]->frame.dataList = m_frameBuf[i]->frame.dataList;
        IvtcEmitEntry e{};
        e.stagingIdx    = stagingIdx;
        e.inputFrameId  = m_cycleInputIds[i];
        e.timestamp     = m_nPts;
        e.duration      = baseDur;
        m_nPts += baseDur;
        m_emitQueue.push_back(e);
        AddMessage(RGY_LOG_DEBUG, _T("ivtc enqueue[%d]: cycleSlot=%d staging=%d pts=%lld dur=%lld inputId=%d match=%d blend=%d (drop=%d mQ=%llu cComb=%llu diff=%llu)\n"),
            emitted, i, stagingIdx,
            (long long)e.timestamp, (long long)e.duration,
            e.inputFrameId, m_cycleMatchType[i], m_cycleApplyBlend[i],
            dropIdx,
            (unsigned long long)m_cycleMatchScore[i],
            (unsigned long long)m_cycleCombScore[i],
            (unsigned long long)m_cycleDiffPrev[i]);
        emitted++;
    }

    // 次サイクルの SAD[0] 用に、今サイクル末尾を保存スロットにコピーしておく。
    if (cycleLen > 0 && filled >= 1 && !finalFlush) {
        const RGYFrameInfo *pLast = &m_frameBuf[filled - 1]->frame;
        RGYFrameInfo *pSave = &m_frameBuf[cycleLen]->frame;
        auto cpErr = m_cl->copyFrame(pSave, pLast, nullptr, queue_main);
        if (cpErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy cycle tail to save slot: %s.\n"), get_err_mes(cpErr));
            return cpErr;
        }
        m_hasSaveSlot = true;
    }

    // per-frame ログ (emit / DROP 両方)。out_idx は emit 順で付番。
    if (m_fpLog) {
        int emitOrdinal = m_outputFrameCount;
        for (int i = 0; i < filled; i++) {
            const char *matchStr = (m_cycleMatchType[i] == 0) ? "c" : (m_cycleMatchType[i] == 1) ? "p" : "n";
            const bool isDrop = (i == dropIdx);
            fprintf(m_fpLog.get(), "%d\t%d\t%s\t%s\t%s\t%llu\t%llu\t%llu\n",
                isDrop ? -1 : emitOrdinal,
                m_cycleInputIds[i],
                matchStr,
                m_cycleApplyBlend[i] ? "blend" : "     ",
                isDrop ? "DROP " : "emit ",
                (unsigned long long)m_cycleMatchScore[i],
                (unsigned long long)m_cycleCombScore[i],
                (unsigned long long)m_cycleDiffPrev[i]);
            if (!isDrop) emitOrdinal++;
        }
        fflush(m_fpLog.get());
    }
    m_cycleFilled = 0;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterIvtc::popEmit(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    if (m_emitQueue.empty()) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return RGY_ERR_NONE;
    }
    const IvtcEmitEntry e = m_emitQueue.front();
    m_emitQueue.pop_front();
    RGYFrameInfo *pOut = &m_frameBuf[e.stagingIdx]->frame;
    pOut->timestamp    = e.timestamp;
    pOut->duration     = e.duration;
    pOut->inputFrameId = e.inputFrameId;
    pOut->picstruct    = RGY_PICSTRUCT_FRAME;
    ppOutputFrames[0] = pOut;
    *pOutputFrameNum = 1;
    m_outputFrameCount++;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterIvtc::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue_main, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    if (!m_ivtc.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_IVTC_CL(m_ivtc).\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    auto prm = std::dynamic_pointer_cast<RGYFilterParamIvtc>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int cycleLen = std::max(prm->ivtc.cycle, 0);
    (void)event;

    const bool hasInput = (pInputFrame && pInputFrame->ptr[0]);

    // 1. 入力を消費する (ある場合)。cycle=0 なら即時 1 emit、cycle>0 なら cycle が満タン
    //    になったタイミングで flushCycle が emit キューを積む。
    if (hasInput) {
        const int slot = m_inputCount % IVTC_CACHE_SIZE;
        RGYFrameInfo *pSlot = &m_cacheFrames[slot]->frame;
        auto copyErr = m_cl->copyFrame(pSlot, pInputFrame, nullptr, queue_main, wait_events);
        if (copyErr != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copy input frame to cache slot %d: %s.\n"), slot, get_err_mes(copyErr));
            return copyErr;
        }
        pSlot->timestamp    = pInputFrame->timestamp;
        pSlot->duration     = pInputFrame->duration;
        pSlot->inputFrameId = pInputFrame->inputFrameId;
        pSlot->picstruct    = pInputFrame->picstruct;
        pSlot->flags        = pInputFrame->flags;
        pSlot->dataList     = pInputFrame->dataList;

        m_inputCount++;

        if (m_inputCount >= 2) {
            // 中央スロット (前の入力) に対応する field-match 出力をサイクルに書き込む。
            const int idx_cur  = (m_inputCount - 2) % IVTC_CACHE_SIZE;
            const int idx_next = (m_inputCount - 1) % IVTC_CACHE_SIZE;
            const int idx_prev = (m_inputCount >= 3) ? (m_inputCount - 3) % IVTC_CACHE_SIZE : idx_cur;

            auto err = processInputToCycle(idx_prev, idx_cur, idx_next, queue_main, wait_events);
            if (err != RGY_ERR_NONE) {
                return err;
            }

            if (cycleLen == 0) {
                // decimation オフ: 1 in 1 out。m_frameBuf[0] がこの call の出力。
                ppOutputFrames[0] = &m_frameBuf[0]->frame;
                *pOutputFrameNum = 1;
                return RGY_ERR_NONE;
            }

            // cycle が満タンになったら staging に移して emit キューへ積む。
            if (m_cycleFilled >= cycleLen) {
                auto ferr = flushCycle(false, pInputFrame->timestamp, queue_main);
                if (ferr != RGY_ERR_NONE) {
                    return ferr;
                }
            }
        }
        // cycleLen > 0 の通常入力時はここで 1 popEmit (キューにあれば) する。
        // 入力がキュー生成をトリガした (flushCycle) ケースも、1 call 1 emit なので 1 枚だけ取り出す。
        return popEmit(ppOutputFrames, pOutputFrameNum);
    }

    // 2. ドレイン: 最後の入力を 1 回だけ field-match (next を cur でデュープ)。
    if (!m_drained && m_inputCount >= 1) {
        const int idx_cur  = (m_inputCount - 1) % IVTC_CACHE_SIZE;
        const int idx_next = idx_cur;
        const int idx_prev = (m_inputCount >= 2) ? (m_inputCount - 2) % IVTC_CACHE_SIZE : idx_cur;
        m_drained = true;

        auto err = processInputToCycle(idx_prev, idx_cur, idx_next, queue_main, wait_events);
        if (err != RGY_ERR_NONE) {
            return err;
        }

        if (cycleLen == 0) {
            ppOutputFrames[0] = &m_frameBuf[0]->frame;
            *pOutputFrameNum = 1;
            return RGY_ERR_NONE;
        }

        // ドレイン時は nextInputPts なし。full サイクルなら drop あり flush、
        // 部分サイクルなら drop なし flush。どちらもキューに積み、popEmit で 1 枚返す。
        const bool finalFlush = (m_cycleFilled < cycleLen);
        auto ferr = flushCycle(finalFlush, AV_NOPTS_VALUE, queue_main);
        if (ferr != RGY_ERR_NONE) {
            return ferr;
        }
        return popEmit(ppOutputFrames, pOutputFrameNum);
    }

    // 3. drain 済み: キューに残りがあれば 1 枚返す、空なら 0 emit。
    return popEmit(ppOutputFrames, pOutputFrameNum);
}

void RGYFilterIvtc::close() {
    m_ivtc.clear();
    m_cacheFrames.clear();
    m_scoreBuf.reset();
    m_scoreHost.clear();
    m_diffBuf.reset();
    m_diffHost.clear();
    m_cycleInPts.clear();
    m_cycleInDur.clear();
    m_cycleInputIds.clear();
    m_cycleMatchScore.clear();
    m_cycleCombScore.clear();
    m_cycleMatchType.clear();
    m_cycleApplyBlend.clear();
    m_cycleDiffPrev.clear();
    m_emitQueue.clear();
    m_stagingBase = 0;
    m_nPts = 0;
    m_nPtsInit = false;
    m_cfrBaseDur = 0;
    m_hasSaveSlot = false;
    m_cycleFilled = 0;
    m_outputFrameCount = 0;
    m_inputCount = 0;
    m_drained = false;
    m_tffDefault = 1;
    m_fpLog.reset();
}
