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

#pragma once
#include <deque>
#include "rgy_filter_cl.h"
#include "rgy_prm.h"

class RGYFilterParamIvtc : public RGYFilterParam {
public:
    VppIvtc ivtc;
    tstring outFilename;

    RGYFilterParamIvtc() : ivtc(), outFilename() {};
    virtual ~RGYFilterParamIvtc() {};
    virtual tstring print() const override { return ivtc.print(); };
};

enum class IvtcMatch : int {
    C = 0, // 現在フレームをそのまま使用
    P = 1, // TFF: [cur.top, prev.bot]  match-with-previous (bot field borrowed from prev)
    N = 2, // TFF: [next.top, cur.bot]  match-with-next (top field borrowed from next)
};

// 出力キューに積む単位。m_frameBuf[stagingIdx] の CLFrame を指し、
// 下流に渡す際のメタデータを個別に保持する。
struct IvtcEmitEntry {
    int stagingIdx;
    int64_t timestamp;
    int64_t duration;
    int inputFrameId;
};

class RGYFilterIvtc : public RGYFilter {
public:
    RGYFilterIvtc(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterIvtc();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue_main, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamIvtc> pParam);
    int getTff(const RGYFrameInfo *frame) const;

    RGY_ERR scoreCandidates(const RGYFrameInfo *prev, const RGYFrameInfo *cur, const RGYFrameInfo *next, int tff, uint64_t matchScoreOut[3], uint64_t combScoreOut[3], RGYOpenCLQueue &queue);
    RGY_ERR synthesizeToCycle(int cycleSlot, const RGYFrameInfo *prev, const RGYFrameInfo *cur, const RGYFrameInfo *next, int tff, const IvtcMatch match, const bool applyBlend, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR computePairDiff(const RGYFrameInfo *pA, const RGYFrameInfo *pB, uint64_t &diffOut, RGYOpenCLQueue &queue);
    RGY_ERR processInputToCycle(int idx_prev, int idx_cur, int idx_next, RGYOpenCLQueue &queue_main, const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR flushCycle(bool finalFlush, int64_t nextInputPts, RGYOpenCLQueue &queue_main);
    RGY_ERR popEmit(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum);

    RGYOpenCLProgramAsync m_ivtc;
    std::vector<std::unique_ptr<RGYCLFrame>> m_cacheFrames; // リングバッファ: prev/cur/next の3枚
    std::unique_ptr<RGYCLBuf> m_scoreBuf;                   // WGごとのスコア集計バッファ
    std::vector<uint32_t> m_scoreHost;                      // ホスト側リードバック用
    std::unique_ptr<RGYCLBuf> m_diffBuf;                    // WGごとの SAD 集計バッファ (reused)
    std::vector<uint32_t> m_diffHost;                       // ホスト側リードバック用
    std::vector<int64_t> m_cycleInPts;                      // サイクル内フレームの入力タイムスタンプ
    std::vector<int64_t> m_cycleInDur;                      // サイクル内フレームの入力 duration
    std::vector<int> m_cycleInputIds;                       // サイクル内フレームの inputFrameId (ログ用)
    std::vector<uint64_t> m_cycleMatchScore;                // 選択マッチの match-quality (ログ用)
    std::vector<uint64_t> m_cycleCombScore;                 // 選択マッチの combing-count (ログ用)
    std::vector<int> m_cycleMatchType;                      // 選択マッチ種別 (0=C, 1=P, 2=N) ログ用
    std::vector<int> m_cycleApplyBlend;                     // post=2 の blend 適用フラグ (ログ用)
    std::vector<uint64_t> m_cycleDiffPrev;                  // SAD(cycle[i], cycle[i-1]) for i>=1; [0] uses m_saveSlot
    std::deque<IvtcEmitEntry> m_emitQueue;                  // 1 call 1 emit 用の出力キュー (AFS/Decimate 方式)
    int m_stagingBase;                                      // m_frameBuf のうち emit-staging 領域の先頭 index
    int64_t m_nPts;                                         // 出力 PTS アキュムレータ (CFR duration を積んで単調増加)
    bool m_nPtsInit;                                        // m_nPts を初回 emit で seed 済みか
    int64_t m_cfrBaseDur;                                   // 最初の完全サイクルで確定させる emit あたり duration (drain や部分 flush でも再利用)
    bool m_hasSaveSlot;                                     // 前サイクル末尾を m_frameBuf[cycle] に保存済みか
    int m_cycleFilled;                                      // 現サイクルに蓄積されたフレーム数
    int m_outputFrameCount;                                 // 出力済みフレーム数 (ログ用)
    int m_inputCount;                                       // 入力フレーム数の累計
    bool m_drained;                                         // ドレイン時に末尾フレームを1回だけ出力
    int m_tffDefault;                                       // auto 時に picstruct 情報がないフレームへ使う既定値
    unique_ptr<FILE, fp_deleter> m_fpLog;
};
