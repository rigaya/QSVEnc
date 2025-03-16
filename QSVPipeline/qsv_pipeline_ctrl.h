// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2021 rigaya
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

#ifndef __QSV_PIPELINE_CTRL_H__
#define __QSV_PIPELINE_CTRL_H__

#include "rgy_version.h"
#include "rgy_osdep.h"
#include "rgy_input.h"
#include "qsv_util.h"
#include "qsv_prm.h"
#include <deque>
#include <set>
#include <optional>
#include "qsv_hw_device.h"
#include "rgy_opencl.h"
#include "qsv_opencl.h"
#include "qsv_query.h"
#include "qsv_allocator.h"
#include "rgy_util.h"
#include "rgy_thread.h"
#include "rgy_timecode.h"
#include "rgy_input.h"
#include "rgy_input_sm.h"
#include "rgy_filter.h"
#include "rgy_filter_ssim.h"
#include "rgy_output.h"
#include "rgy_output_avcodec.h"
#include "qsv_util.h"
#include "qsv_mfx_dec.h"
#include "qsv_vpp_mfx.h"
#include "rgy_parallel_enc.h"

const uint32_t MSDK_DEC_WAIT_INTERVAL = 60000;
const uint32_t MSDK_ENC_WAIT_INTERVAL = 10000;
const uint32_t MSDK_VPP_WAIT_INTERVAL = 60000;
const uint32_t MSDK_WAIT_INTERVAL = MSDK_DEC_WAIT_INTERVAL + 3 * MSDK_VPP_WAIT_INTERVAL + MSDK_ENC_WAIT_INTERVAL; // an estimate for the longest pipeline we have in samples

const uint32_t MSDK_INVALID_SURF_IDX = 0xFFFF;

static void copy_crop_info(mfxFrameSurface1 *dst, const mfxFrameInfo *src) {
    if (dst != nullptr) {
        dst->Info.CropX = src->CropX;
        dst->Info.CropY = src->CropY;
        dst->Info.CropW = src->CropW;
        dst->Info.CropH = src->CropH;
    }
};

struct VppVilterBlock {
    VppFilterType type;
    std::unique_ptr<QSVVppMfx> vppmfx;
    std::vector<std::unique_ptr<RGYFilter>> vppcl;

    VppVilterBlock(std::unique_ptr<QSVVppMfx>& filter) : type(VppFilterType::FILTER_MFX), vppmfx(std::move(filter)), vppcl() {};
    VppVilterBlock(std::vector<std::unique_ptr<RGYFilter>>& filter) : type(VppFilterType::FILTER_OPENCL), vppmfx(), vppcl(std::move(filter)) {};
};


enum class PipelineTaskOutputType {
    UNKNOWN,
    SURFACE,
    BITSTREAM
};

enum class PipelineTaskSurfaceType {
    UNKNOWN,
    CL,
    MFX
};

class PipelineTaskStopWatch {
    std::array<std::vector<std::pair<tstring, int64_t>>, 2> m_ticks;
    std::array<std::chrono::high_resolution_clock::time_point, 2> m_prevTimepoints;
public:
    PipelineTaskStopWatch(const std::vector<tstring>& tickSend, const std::vector<tstring>& tickGet) : m_ticks(), m_prevTimepoints() {
        for (size_t i = 0; i < tickSend.size(); i++) {
            m_ticks[0].push_back({ tickSend[i], 0 });
        }
        for (size_t i = 0; i < tickGet.size(); i++) {
            m_ticks[1].push_back({ tickGet[i], 0 });
        }
    };
    void set(const int type) {
        m_prevTimepoints[type] = std::chrono::high_resolution_clock::now();
    }
    void add(const int type, const int idx) {
        auto now = std::chrono::high_resolution_clock::now();
        m_ticks[type][idx].second += std::chrono::duration_cast<std::chrono::nanoseconds>(now - m_prevTimepoints[type]).count();
        m_prevTimepoints[type] = now;
    }
    int64_t totalTicks() const {
        int64_t total = 0;
        for (int itype = 0; itype < 2; itype++) {
            for (int i = 0; i < (int)m_ticks[itype].size(); i++) {
                total += m_ticks[itype][i].second;
            }
        }
        return total;
    }
    size_t maxWorkStrLen() const {
        size_t maxLen = 0;
        for (size_t itype = 0; itype < m_ticks.size(); itype++) {
            for (int i = 0; i < (int)m_ticks[itype].size(); i++) {
                maxLen = (std::max)(maxLen, m_ticks[itype][i].first.length());
            }
        }
        return maxLen;
    }
    tstring print(const int64_t totalTicks, const size_t maxLen) {
        const TCHAR *type[] = {_T("send"), _T("get ")};
        tstring str;
        for (size_t itype = 0; itype < m_ticks.size(); itype++) {
            int64_t total = 0;
            for (int i = 0; i < (int)m_ticks[itype].size(); i++) {
                str += type[itype] + tstring(_T(":"));
                str += m_ticks[itype][i].first;
                str += tstring(maxLen - m_ticks[itype][i].first.length(), _T(' '));
                str += strsprintf(_T(" : %8d ms [%5.1f]\n"), ((m_ticks[itype][i].second + 500000) / 1000000), m_ticks[itype][i].second * 100.0 / totalTicks);
                total += m_ticks[itype][i].second;
            }
            if (m_ticks[itype].size() > 1) {
                str += type[itype] + tstring(_T(":"));
                str += _T("total");
                str += tstring(maxLen - _tcslen(_T("total")), _T(' '));
                str += strsprintf(_T(" : %8d ms [%5.1f]\n"), ((total + 500000) / 1000000), total * 100.0 / totalTicks);
            }
        }
        return str;
    }
};

class PipelineTaskSurface {
private:
    RGYFrame *surf;
    std::atomic<int> *ref;
public:
    PipelineTaskSurface() : surf(nullptr), ref(nullptr) {};
    PipelineTaskSurface(RGYFrame *surf_, std::atomic<int> *ref_) : surf(surf_), ref(ref_) { if (surf) (*ref)++; };
    PipelineTaskSurface(const PipelineTaskSurface& obj) : surf(obj.surf), ref(obj.ref) { if (surf) (*ref)++; }
    PipelineTaskSurface &operator=(const PipelineTaskSurface &obj) {
        if (this != &obj) { // 自身の代入チェック
            surf = obj.surf;
            ref = obj.ref;
            if (surf) (*ref)++;
        }
        return *this;
    }
    ~PipelineTaskSurface() { reset(); }
    void reset() { if (surf) (*ref)--; surf = nullptr; ref = nullptr; }
    bool operator !() const {
        return frame() == nullptr;
    }
    bool operator !=(const PipelineTaskSurface& obj) const { return frame() != obj.frame(); }
    bool operator ==(const PipelineTaskSurface& obj) const { return frame() == obj.frame(); }
    bool operator !=(std::nullptr_t) const { return frame() != nullptr; }
    bool operator ==(std::nullptr_t) const { return frame() == nullptr; }
    const RGYFrameMFXSurf *mfx() const { return dynamic_cast<const RGYFrameMFXSurf*>(surf); }
    RGYFrameMFXSurf *mfx() { return dynamic_cast<RGYFrameMFXSurf*>(surf); }
    const RGYCLFrame *cl() const { return dynamic_cast<const RGYCLFrame*>(surf); }
    RGYCLFrame *cl() { return dynamic_cast<RGYCLFrame*>(surf); }
    const RGYFrame *frame() const { return surf; }
    RGYFrame *frame() { return surf; }
};

// アプリ用の独自参照カウンタと組み合わせたクラス
class PipelineTaskSurfaces {
private:
    class PipelineTaskSurfacesPair {
    private:
        std::unique_ptr<RGYFrame> surf_;
        std::atomic<int> ref;
    public:
        PipelineTaskSurfacesPair(std::unique_ptr<RGYFrame> s) : surf_(std::move(s)), ref(0) {};

        // 使用されていないフレームかを返す
        // mfxの参照カウンタと独自参照カウンタの両方をチェック
        bool isFree() const {
            if (ref != 0) return false;
            if (auto mfxsurf = dynamic_cast<const RGYFrameMFXSurf*>(surf_.get()); mfxsurf != nullptr) {
                return mfxsurf->locked() == 0;
            }
            return true;
        }
        PipelineTaskSurface getRef() { return PipelineTaskSurface(surf_.get(), &ref); };
        const RGYFrame *surf() const { return surf_.get(); }
        RGYFrame *surf() { return surf_.get(); }
        PipelineTaskSurfaceType type() const {
            if (!surf_) return PipelineTaskSurfaceType::UNKNOWN;
            if (dynamic_cast<const RGYCLFrame*>(surf_.get())) return PipelineTaskSurfaceType::CL;
            if (dynamic_cast<const RGYFrameMFXSurf*>(surf_.get())) return PipelineTaskSurfaceType::MFX;
            return PipelineTaskSurfaceType::UNKNOWN;
        }
    };
    std::vector<std::unique_ptr<PipelineTaskSurfacesPair>> m_surfaces; // フレームと参照カウンタ
public:
    PipelineTaskSurfaces() : m_surfaces() {};
    ~PipelineTaskSurfaces() { }

    void clear() {
        m_surfaces.clear();
    }
    void setSurfaces(std::vector<mfxFrameSurface1>& surfs) {
        clear();
        m_surfaces.resize(surfs.size());
        for (size_t i = 0; i < m_surfaces.size(); i++) {
            m_surfaces[i] = std::make_unique<PipelineTaskSurfacesPair>(std::move(std::make_unique<RGYFrameMFXSurf>(surfs[i])));
        }
    }
    void setSurfaces(std::vector<std::unique_ptr<RGYCLFrame>>& surfs) {
        clear();
        m_surfaces.resize(surfs.size());
        for (size_t i = 0; i < m_surfaces.size(); i++) {
            m_surfaces[i] = std::make_unique<PipelineTaskSurfacesPair>(std::move(surfs[i]));
        }
    }

    PipelineTaskSurface getFreeSurf() {
        for (auto& s : m_surfaces) {
            if (s->isFree()) {
                return s->getRef();
            }
        }
        return PipelineTaskSurface();
    }
    PipelineTaskSurface get(mfxFrameSurface1 *surf) {
        for (auto& s : m_surfaces) {
            if (auto mfx = dynamic_cast<RGYFrameMFXSurf*>(s->surf()); mfx != nullptr) {
                if (mfx->surf() == surf) {
                    return s->getRef();
                }
            }
        }
        return PipelineTaskSurface();
    }
    size_t bufCount() const { return m_surfaces.size(); }

    bool isAllFree() const {
        for (const auto& s : m_surfaces) {
            if (!s->isFree()) {
                return false;
            }
        }
        return true;
    }
protected:
    PipelineTaskSurfacesPair *findSurf(RGYFrame *surf) {
        for (auto& s : m_surfaces) {
            if (s->surf() == surf) {
                return s.get();
            }
        }
        return nullptr;
    }
};

class PipelineTaskOutputDataCustom {
    int type;
public:
    PipelineTaskOutputDataCustom() {};
    virtual ~PipelineTaskOutputDataCustom() {};
};

class PipelineTaskOutputDataCheckPts : public PipelineTaskOutputDataCustom {
private:
    int64_t timestamp;
public:
    PipelineTaskOutputDataCheckPts() : PipelineTaskOutputDataCustom() {};
    PipelineTaskOutputDataCheckPts(int64_t timestampOverride) : PipelineTaskOutputDataCustom(), timestamp(timestampOverride) {};
    virtual ~PipelineTaskOutputDataCheckPts() {};
    int64_t timestampOverride() const { return timestamp; }
};

class PipelineTaskOutput {
protected:
    PipelineTaskOutputType m_type;
    MFXVideoSession *m_mfxSession;
    mfxSyncPoint m_syncpoint;
    std::unique_ptr<PipelineTaskOutputDataCustom> m_customData;
public:
    PipelineTaskOutput(MFXVideoSession *mfxSession) : m_type(PipelineTaskOutputType::UNKNOWN), m_mfxSession(mfxSession), m_syncpoint(nullptr), m_customData() {};
    PipelineTaskOutput(MFXVideoSession *mfxSession, PipelineTaskOutputType type, mfxSyncPoint syncpoint) : m_type(type), m_mfxSession(mfxSession), m_syncpoint(syncpoint), m_customData() {};
    PipelineTaskOutput(MFXVideoSession *mfxSession, PipelineTaskOutputType type, mfxSyncPoint syncpoint, std::unique_ptr<PipelineTaskOutputDataCustom>& customData) : m_type(type), m_mfxSession(mfxSession), m_syncpoint(syncpoint), m_customData(std::move(customData)) {};
    RGY_ERR waitsync(uint32_t wait = MSDK_WAIT_INTERVAL) {
        if (m_syncpoint == nullptr) {
            return RGY_ERR_NONE;
        }
        auto err = m_mfxSession->SyncOperation(m_syncpoint, wait);
        m_syncpoint = nullptr;
        return err_to_rgy(err);
    }
    virtual void depend_clear() {};
    mfxSyncPoint syncpoint() const { return m_syncpoint; }
    PipelineTaskOutputType type() const { return m_type; }
    const PipelineTaskOutputDataCustom *customdata() const { return m_customData.get(); }
    virtual RGY_ERR write([[maybe_unused]] RGYOutput *writer, [[maybe_unused]] QSVAllocator *allocator, [[maybe_unused]] RGYOpenCLQueue *clqueue, [[maybe_unused]] RGYFilterSsim *videoQualityMetric) {
        return RGY_ERR_UNSUPPORTED;
    }
    virtual ~PipelineTaskOutput() {};
};

class PipelineTaskOutputSurf : public PipelineTaskOutput {
protected:
    PipelineTaskSurface m_surf;
    std::unique_ptr<PipelineTaskOutput> m_dependencyFrame;
    std::vector<RGYOpenCLEvent> m_clevents;
public:
    PipelineTaskOutputSurf(MFXVideoSession *mfxSession, PipelineTaskSurface surf, mfxSyncPoint syncpoint) :
        PipelineTaskOutput(mfxSession, PipelineTaskOutputType::SURFACE, syncpoint), m_surf(surf), m_dependencyFrame(), m_clevents() { };
    PipelineTaskOutputSurf(MFXVideoSession *mfxSession, PipelineTaskSurface surf, mfxSyncPoint syncpoint, std::unique_ptr<PipelineTaskOutputDataCustom>& customData) :
        PipelineTaskOutput(mfxSession, PipelineTaskOutputType::SURFACE, syncpoint, customData), m_surf(surf), m_dependencyFrame(), m_clevents() { };
    PipelineTaskOutputSurf(MFXVideoSession *mfxSession, PipelineTaskSurface surf, std::unique_ptr<PipelineTaskOutput>& dependencyFrame, RGYOpenCLEvent& clevent) :
        PipelineTaskOutput(mfxSession, PipelineTaskOutputType::SURFACE, nullptr),
        m_surf(surf), m_dependencyFrame(std::move(dependencyFrame)), m_clevents() {
        m_clevents.push_back(clevent);
    };
    virtual ~PipelineTaskOutputSurf() {
        depend_clear();
        m_surf.reset();
    };

    PipelineTaskSurface& surf() { return m_surf; }

    void addClEvent(RGYOpenCLEvent& clevent) {
        m_clevents.push_back(clevent);
    }

    virtual void depend_clear() override {
        RGYOpenCLEvent::wait(m_clevents);
        m_clevents.clear();
        m_dependencyFrame.reset();
    }

    RGY_ERR writeMFX(RGYOutput *writer, QSVAllocator *allocator) {
        auto mfxSurf = m_surf.mfx()->surf();
        const bool allocatorD3D11 = IS_ALLOCATOR_D3D11(allocator);
        if (mfxSurf->Data.MemId) {
            // MFXReadWriteMidの使用はd3d11使用時のみにする必要がある
            // MFXReadWriteMidの寿命を考慮し、引数として渡す場所で三項演算子を使用する
            auto sts = allocator->Lock(allocator->pthis, (allocatorD3D11) ? (mfxMemId)MFXReadWriteMid(mfxSurf->Data.MemId, MFXReadWriteMid::read) : mfxSurf->Data.MemId, &(mfxSurf->Data));
            if (sts < MFX_ERR_NONE) {
                return err_to_rgy(sts);
            }
        }
        auto err = writer->WriteNextFrame(m_surf.frame());
        if (mfxSurf->Data.MemId) {
            // MFXReadWriteMidの使用はd3d11使用時のみにする必要がある
            // MFXReadWriteMidの寿命を考慮し、引数として渡す場所で三項演算子を使用する
            allocator->Unlock(allocator->pthis, (allocatorD3D11) ? (mfxMemId)MFXReadWriteMid(mfxSurf->Data.MemId, MFXReadWriteMid::read) : mfxSurf->Data.MemId, &(mfxSurf->Data));
        }
        return err;
    }

    RGY_ERR writeCL(RGYOutput *writer, RGYOpenCLQueue *clqueue) {
        if (clqueue == nullptr) {
            return RGY_ERR_NULL_PTR;
        }
        auto clframe = m_surf.cl();
        auto err = clframe->queueMapBuffer(*clqueue, CL_MAP_READ); // CPUが読み込むためにmapする
        if (err != RGY_ERR_NONE) {
            return err;
        }
        clframe->mapWait();
        auto mappedframe = clframe->mappedHost();
        err = writer->WriteNextFrame(mappedframe);
        clframe->unmapBuffer();
        return err;
    }

    virtual RGY_ERR write([[maybe_unused]] RGYOutput *writer, [[maybe_unused]] QSVAllocator *allocator, [[maybe_unused]] RGYOpenCLQueue *clqueue, [[maybe_unused]] RGYFilterSsim *videoQualityMetric) override {
        if (!writer || writer->getOutType() == OUT_TYPE_NONE) {
            return RGY_ERR_NOT_INITIALIZED;
        }
        if (writer->getOutType() != OUT_TYPE_SURFACE) {
            return RGY_ERR_INVALID_OPERATION;
        }
        auto err = (m_surf.mfx() != nullptr) ? writeMFX(writer, allocator) : writeCL(writer, clqueue);
        return err;
    }
};

class PipelineTaskOutputBitstream : public PipelineTaskOutput {
protected:
    std::shared_ptr<RGYBitstream> m_bs;
public:
    PipelineTaskOutputBitstream(MFXVideoSession *mfxSession, std::shared_ptr<RGYBitstream> bs, mfxSyncPoint syncpoint) : PipelineTaskOutput(mfxSession, PipelineTaskOutputType::BITSTREAM, syncpoint), m_bs(bs) {};
    virtual ~PipelineTaskOutputBitstream() { };

    std::shared_ptr<RGYBitstream>& bitstream() { return m_bs; }

    virtual RGY_ERR write([[maybe_unused]] RGYOutput *writer, [[maybe_unused]] QSVAllocator *allocator, [[maybe_unused]] RGYOpenCLQueue *clqueue, [[maybe_unused]] RGYFilterSsim *videoQualityMetric) override {
        if (!writer || writer->getOutType() == OUT_TYPE_NONE) {
            return RGY_ERR_NOT_INITIALIZED;
        }
        if (writer->getOutType() != OUT_TYPE_BITSTREAM) {
            return RGY_ERR_INVALID_OPERATION;
        }
        if (videoQualityMetric) {
            if (!videoQualityMetric->decodeStarted()) {
                videoQualityMetric->initDecode(m_bs.get());
            }
            videoQualityMetric->addBitstream(m_bs.get());
        }
        return writer->WriteNextFrame(m_bs.get());
    }
};

enum class PipelineTaskType {
    UNKNOWN,
    MFXVPP,
    MFXDEC,
    MFXENC,
    MFXENCODE,
    INPUT,
    INPUTCL,
    CHECKPTS,
    TRIM,
    AUDIO,
    OUTPUTRAW,
    OPENCL,
    VIDEOMETRIC,
    PECOLLECT,
};

static const TCHAR *getPipelineTaskTypeName(PipelineTaskType type) {
    switch (type) {
    case PipelineTaskType::MFXVPP:      return _T("MFXVPP");
    case PipelineTaskType::MFXDEC:      return _T("MFXDEC");
    case PipelineTaskType::MFXENC:      return _T("MFXENC");
    case PipelineTaskType::MFXENCODE:   return _T("MFXENCODE");
    case PipelineTaskType::INPUT:       return _T("INPUT");
    case PipelineTaskType::INPUTCL:     return _T("INPUTCL");
    case PipelineTaskType::CHECKPTS:    return _T("CHECKPTS");
    case PipelineTaskType::TRIM:        return _T("TRIM");
    case PipelineTaskType::OPENCL:      return _T("OPENCL");
    case PipelineTaskType::AUDIO:       return _T("AUDIO");
    case PipelineTaskType::VIDEOMETRIC: return _T("VIDEOMETRIC");
    case PipelineTaskType::OUTPUTRAW:   return _T("OUTRAW");
    case PipelineTaskType::PECOLLECT:   return _T("PECOLLECT");
    default: return _T("UNKNOWN");
    }
}

// Alllocするときの優先度 値が高い方が優先
static const int getPipelineTaskAllocPriority(PipelineTaskType type) {
    switch (type) {
    case PipelineTaskType::MFXENCODE: return 4;
    case PipelineTaskType::MFXENC:    return 3;
    case PipelineTaskType::MFXDEC:    return 2;
    case PipelineTaskType::MFXVPP:    return 1;
    case PipelineTaskType::INPUT:
    case PipelineTaskType::INPUTCL:
    case PipelineTaskType::CHECKPTS:
    case PipelineTaskType::TRIM:
    case PipelineTaskType::OPENCL:
    case PipelineTaskType::AUDIO:
    case PipelineTaskType::OUTPUTRAW:
    case PipelineTaskType::VIDEOMETRIC:
    case PipelineTaskType::PECOLLECT:
    default: return 0;
    }
}

class PipelineTask {
protected:
    PipelineTaskType m_type;
    std::deque<std::unique_ptr<PipelineTaskOutput>> m_outQeueue;
    PipelineTaskSurfaces m_workSurfs;
    MFXVideoSession *m_mfxSession;
    QSVAllocator *m_allocator;
    mfxFrameAllocResponse m_allocResponse;
    int m_inFrames;
    int m_outFrames;
    int m_outMaxQueueSize;
    mfxVersion m_mfxVer;
    std::shared_ptr<RGYLog> m_log;
    std::unique_ptr<PipelineTaskStopWatch> m_stopwatch;
public:
    PipelineTask() : m_type(PipelineTaskType::UNKNOWN), m_outQeueue(), m_workSurfs(), m_mfxSession(nullptr), m_allocator(nullptr), m_allocResponse({ 0 }), m_inFrames(0), m_outFrames(0), m_outMaxQueueSize(0), m_mfxVer({ 0 }), m_log(), m_stopwatch() {};
    PipelineTask(PipelineTaskType type, int outMaxQueueSize, MFXVideoSession *mfxSession, mfxVersion mfxVer, std::shared_ptr<RGYLog> log) :
        m_type(type), m_outQeueue(), m_workSurfs(), m_mfxSession(mfxSession), m_allocator(nullptr), m_allocResponse({ 0 }), m_inFrames(0), m_outFrames(0), m_outMaxQueueSize(outMaxQueueSize), m_mfxVer(mfxVer), m_log(log), m_stopwatch() {
    };
    virtual ~PipelineTask() {
        if (m_allocator) {
            m_allocator->Free(m_allocator->pthis, &m_allocResponse);
        }
        m_stopwatch.reset();
        m_workSurfs.clear();
    }
    virtual void setStopWatch() {};
    virtual void printStopWatch(const int64_t totalTicks, const size_t maxLen) {
        if (m_stopwatch) {
            const auto strlines = split(m_stopwatch->print(totalTicks, maxLen), _T("\n"));
            for (auto& str : strlines) {
                if (str.length() > 0) {
                    PrintMes(RGY_LOG_INFO, _T("%s\n"), str.c_str());
                }
            }
        }
    }
    virtual int64_t getStopWatchTotal() const {
        return (m_stopwatch) ? m_stopwatch->totalTicks() : 0ll;
    }
    virtual size_t getStopWatchMaxWorkStrLen() const {
        return (m_stopwatch) ? m_stopwatch->maxWorkStrLen() : 0u;
    }
    virtual bool isPassThrough() const { return false; }
    virtual tstring print() const { return getPipelineTaskTypeName(m_type); }
    virtual std::optional<mfxFrameAllocRequest> requiredSurfIn() = 0;
    virtual std::optional<mfxFrameAllocRequest> requiredSurfOut() = 0;
    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) = 0;
    virtual RGY_ERR getOutputFrameInfo(mfxFrameInfo& info) { info = { 0 }; return RGY_ERR_INVALID_CALL; }
    virtual std::vector<std::unique_ptr<PipelineTaskOutput>> getOutput(const bool sync) {
        if (m_stopwatch) m_stopwatch->set(1);
        std::vector<std::unique_ptr<PipelineTaskOutput>> output;
        while ((int)m_outQeueue.size() > m_outMaxQueueSize) {
            auto out = std::move(m_outQeueue.front());
            m_outQeueue.pop_front();
            if (sync) {
                out->waitsync();
            }
            out->depend_clear();
            m_outFrames++;
            output.push_back(std::move(out));
        }
        if (m_stopwatch) m_stopwatch->add(1, 0);
        return output;
    }
    bool isMFXTask(const PipelineTaskType task) const {
        return task == PipelineTaskType::MFXDEC
            || task == PipelineTaskType::MFXVPP
            || task == PipelineTaskType::MFXENC
            || task == PipelineTaskType::MFXENCODE;
    }
    // mfx関連とそうでないtaskのやり取りでロックが必要
    bool requireSync(const PipelineTaskType nextTaskType) const {
        return isMFXTask(m_type) != isMFXTask(nextTaskType);
    }
    int workSurfacesAllocPriority() const {
        return getPipelineTaskAllocPriority(m_type);
    }
    size_t workSurfacesCount() const {
        return m_workSurfs.bufCount();
    }

    void PrintMes(RGYLogLevel log_level, const TCHAR *format, ...) {
        if (m_log.get() == nullptr) {
            if (log_level <= RGY_LOG_INFO) {
                return;
            }
        } else if (log_level < m_log->getLogLevel(RGY_LOGT_CORE)) {
            return;
        }

        va_list args;
        va_start(args, format);

        int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
        vector<TCHAR> buffer(len, 0);
        _vstprintf_s(buffer.data(), len, format, args);
        va_end(args);

        tstring mes = getPipelineTaskTypeName(m_type) + tstring(_T(": ")) + buffer.data();

        if (m_log.get() != nullptr) {
            m_log->write(log_level, RGY_LOGT_CORE, mes.c_str());
        } else {
            _ftprintf(stderr, _T("%s"), mes.c_str());
        }
    }
protected:
    RGY_ERR workSurfacesClear() {
        if (m_outQeueue.size() != 0) {
            return RGY_ERR_UNSUPPORTED;
        }
        if (!m_workSurfs.isAllFree()) {
            return RGY_ERR_UNSUPPORTED;
        }
        if (m_allocator && m_workSurfs.bufCount() > 0) {
            auto err = err_to_rgy(m_allocator->Free(m_allocator->pthis, &m_allocResponse));
            if (err != RGY_ERR_NONE) {
                return err;
            }
            m_workSurfs.clear();
        }
        return RGY_ERR_NONE;
    }
public:
    RGY_ERR workSurfacesAlloc(mfxFrameAllocRequest& allocRequest, const bool externalAlloc, QSVAllocator *allocator) {
        auto sts = workSurfacesClear();
        if (sts != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("allocWorkSurfaces:   Failed to clear old surfaces: %s.\n"), get_err_mes(sts));
            return sts;
        }
        PrintMes(RGY_LOG_DEBUG, _T("allocWorkSurfaces:   cleared old surfaces: %s.\n"), get_err_mes(sts));

        m_allocator = allocator;
        sts = err_to_rgy(m_allocator->Alloc(m_allocator->pthis, &allocRequest, &m_allocResponse));
        if (sts != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("allocWorkSurfaces:   Failed to allocate frames: %s.\n"), get_err_mes(sts));
            return sts;
        }
        PrintMes(RGY_LOG_DEBUG, _T("allocWorkSurfaces:   allocated %d frames.\n"), m_allocResponse.NumFrameActual);

        std::vector<mfxFrameSurface1> workSurfs(m_allocResponse.NumFrameActual);
        for (size_t i = 0; i < workSurfs.size(); i++) {
            memset(&(workSurfs[i]), 0, sizeof(workSurfs[0]));
            memcpy(&workSurfs[i].Info, &(allocRequest.Info), sizeof(mfxFrameInfo));

            if (externalAlloc) {
                workSurfs[i].Data.MemId = m_allocResponse.mids[i];
            } else {
                sts = err_to_rgy(m_allocator->Lock(m_allocator->pthis, m_allocResponse.mids[i], &(workSurfs[i].Data)));
                if (sts != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("allocWorkSurfaces:   Failed to lock frame #%d: %s.\n"), i, get_err_mes(sts));
                    return sts;
                }
            }
        }
        m_workSurfs.setSurfaces(workSurfs);
        return RGY_ERR_NONE;
    }
    RGY_ERR workSurfacesAllocCL(const int numFrames, const RGYFrameInfo &frame, RGYOpenCLContext *cl) {
        auto sts = workSurfacesClear();
        if (sts != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("allocWorkSurfaces:   Failed to clear old surfaces: %s.\n"), get_err_mes(sts));
            return sts;
        }
        PrintMes(RGY_LOG_DEBUG, _T("allocWorkSurfaces:   cleared old surfaces: %s.\n"), get_err_mes(sts));

        // OpenCLフレームの確保
        std::vector<std::unique_ptr<RGYCLFrame>> frames(numFrames);
        for (size_t i = 0; i < frames.size(); i++) {
            //CPUとのやり取りが効率化できるよう、CL_MEM_ALLOC_HOST_PTR を指定する
            //これでmap/unmapで可能な場合コピーが発生しない
            frames[i] = cl->createFrameBuffer(frame, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
        }
        m_workSurfs.setSurfaces(frames);
        return RGY_ERR_NONE;
    }

    // surfの対応するPipelineTaskSurfaceを見つけ、これから使用するために参照を増やす
    // 破棄時にアプリ側の参照カウンタを減算するようにshared_ptrで設定してある
    PipelineTaskSurface useTaskSurf(mfxFrameSurface1 *surf) {
        return m_workSurfs.get(surf);
    }
    // 使用中でないフレームを探してきて、参照カウンタを加算したものを返す
    // 破棄時にアプリ側の参照カウンタを減算するようにshared_ptrで設定してある
    PipelineTaskSurface getWorkSurf() {
        if (m_workSurfs.bufCount() == 0) {
            PrintMes(RGY_LOG_ERROR, _T("getWorkSurf:   No buffer allocated!\n"));
            return PipelineTaskSurface();
        }
        for (uint32_t i = 0; i < MSDK_WAIT_INTERVAL; i++) {
            PipelineTaskSurface s = m_workSurfs.getFreeSurf();
            if (s != nullptr) {
                return s;
            }
            sleep_hybrid(i);
        }
        PrintMes(RGY_LOG_ERROR, _T("getWorkSurf:   Failed to get work surface, all %d frames used.\n"), m_workSurfs.bufCount());
        return PipelineTaskSurface();
    }

    void setOutputMaxQueueSize(int size) { m_outMaxQueueSize = size; }

    PipelineTaskType taskType() const { return m_type; }
    int inputFrames() const { return m_inFrames; }
    int outputFrames() const { return m_outFrames; }
    int outputMaxQueueSize() const { return m_outMaxQueueSize; }
};

class PipelineTaskInput : public PipelineTask {
    RGYInput *m_input;
    QSVAllocator *m_allocator;
    int64_t m_endPts; // 並列処理時用の終了時刻 (この時刻は含まないようにする) -1の場合は制限なし(最後まで)
    bool m_allocatorD3D11;
    std::shared_ptr<RGYOpenCLContext> m_cl;
public:
    PipelineTaskInput(MFXVideoSession *mfxSession, QSVAllocator *allocator, int64_t endPts, int outMaxQueueSize, RGYInput *input, mfxVersion mfxVer, std::shared_ptr<RGYOpenCLContext> cl, std::shared_ptr<RGYLog> log)
        : PipelineTask(PipelineTaskType::INPUT, outMaxQueueSize, mfxSession, mfxVer, log), m_input(input), m_allocator(allocator), m_endPts(endPts), m_allocatorD3D11(IS_ALLOCATOR_D3D11(allocator)), m_cl(cl) {

    };
    virtual ~PipelineTaskInput() {};
    virtual void setStopWatch() override {
        m_stopwatch = std::make_unique<PipelineTaskStopWatch>(
            std::vector<tstring>{ _T("getWorkSurf"), _T("allocatorLock"), _T("CLqueueMapBuffer"), _T("LoadNextFrame"), _T("allocatorUnLock"), _T("CLunmapBuffer") },
            std::vector<tstring>{_T("")}
        );
    }
    virtual std::optional<mfxFrameAllocRequest> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<mfxFrameAllocRequest> requiredSurfOut() override { return std::nullopt; };
    RGY_ERR loadNextFrameMFX(PipelineTaskSurface& surfWork) {
        if (m_stopwatch) m_stopwatch->set(0);
        auto mfxSurf = surfWork.mfx()->surf();
        if (mfxSurf->Data.MemId) {
            // MFXReadWriteMidの使用はd3d11使用時のみにする必要がある
            // MFXReadWriteMidの寿命を考慮し、引数として渡す場所で三項演算子を使用する
            auto sts = m_allocator->Lock(m_allocator->pthis, (m_allocatorD3D11) ? (mfxMemId)MFXReadWriteMid(mfxSurf->Data.MemId, MFXReadWriteMid::write) : mfxSurf->Data.MemId, &(mfxSurf->Data));
            if (sts < MFX_ERR_NONE) {
                return err_to_rgy(sts);
            }
        }
        if (m_stopwatch) m_stopwatch->add(0, 1);
        auto err = m_input->LoadNextFrame(surfWork.frame());
        if (err != RGY_ERR_NONE) {
            //Unlockする必要があるので、ここに入ってもすぐにreturnしてはいけない
            if (err == RGY_ERR_MORE_DATA) { // EOF
                err = RGY_ERR_MORE_BITSTREAM; // EOF を PipelineTaskMFXDecode のreturnコードに合わせる
            } else {
                PrintMes(RGY_LOG_ERROR, _T("Error in reader: %s.\n"), get_err_mes(err));
            }
        }
        if (m_stopwatch) m_stopwatch->add(0, 3);
        if (mfxSurf->Data.MemId) {
            // MFXReadWriteMid の使用はd3d11使用時のみにする必要がある
            // MFXReadWriteMidの寿命を考慮し、引数として渡す場所で三項演算子を使用する
            m_allocator->Unlock(m_allocator->pthis, (m_allocatorD3D11) ? (mfxMemId)MFXReadWriteMid(mfxSurf->Data.MemId, MFXReadWriteMid::write) : mfxSurf->Data.MemId, &(mfxSurf->Data));
        }
        if (m_stopwatch) m_stopwatch->add(0, 4);
        return err;
    }
    RGY_ERR loadNextFrameCL(PipelineTaskSurface& surfWork) {
        if (m_stopwatch) m_stopwatch->set(0);
        auto clframe = surfWork.cl();
        auto err = clframe->queueMapBuffer(m_cl->queue(), CL_MAP_WRITE); // CPUが書き込むためにMapする
        if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to map buffer: %s.\n"), get_err_mes(err));
            return err;
        }
        clframe->mapWait(); //すぐ終わるはず
        if (m_stopwatch) m_stopwatch->add(0, 2);
        auto mappedframe = clframe->mappedHost();
        err = m_input->LoadNextFrame(mappedframe);
        if (err != RGY_ERR_NONE) {
            //Unlockする必要があるので、ここに入ってもすぐにreturnしてはいけない
            if (err == RGY_ERR_MORE_DATA) { // EOF
                err = RGY_ERR_MORE_BITSTREAM; // EOF を PipelineTaskMFXDecode のreturnコードに合わせる
            } else {
                PrintMes(RGY_LOG_ERROR, _T("Error in reader: %s.\n"), get_err_mes(err));
            }
        }
        if (m_stopwatch) m_stopwatch->add(0, 3);
        clframe->setPropertyFrom(mappedframe);
        auto clerr = clframe->unmapBuffer();
        if (clerr != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to unmap buffer: %s.\n"), get_err_mes(err));
            if (err == RGY_ERR_NONE) {
                err = clerr;
            }
        }
        if (m_stopwatch) m_stopwatch->add(0, 5);
        return err;
    }
    virtual RGY_ERR sendFrame([[maybe_unused]] std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (m_stopwatch) m_stopwatch->set(0);
        auto surfWork = getWorkSurf();
        if (surfWork == nullptr) {
            PrintMes(RGY_LOG_ERROR, _T("failed to get work surface for input.\n"));
            return RGY_ERR_NOT_ENOUGH_BUFFER;
        }
        if (m_stopwatch) m_stopwatch->add(0, 0);
        auto err = (surfWork.mfx() != nullptr) ? loadNextFrameMFX(surfWork) : loadNextFrameCL(surfWork);
        if (err == RGY_ERR_NONE) {
            if (m_endPts >= 0
                && (int64_t)surfWork.frame()->timestamp() != AV_NOPTS_VALUE // timestampが設定されていない場合は無視
                && (int64_t)surfWork.frame()->timestamp() >= m_endPts) { // m_endPtsは含まないようにする(重要)
                return RGY_ERR_MORE_BITSTREAM; //入力ビットストリームは終了
            }
            surfWork.frame()->setInputFrameId(m_inFrames++);
            m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_mfxSession, surfWork, nullptr));
        }
        return err;
    }
    virtual RGY_ERR getOutputFrameInfo(mfxFrameInfo& info) override {
        auto frameInfo = m_input->GetInputFrameInfo();
        info = frameinfo_rgy_to_enc(frameInfo);
        return RGY_ERR_NONE;
    }
};

class PipelineTaskMFXDecode : public PipelineTask {
protected:
    struct FrameFlags {
        int64_t timestamp;
        RGY_FRAME_FLAGS flags;

        FrameFlags() : timestamp(AV_NOPTS_VALUE), flags(RGY_FRAME_FLAG_NONE) {};
        FrameFlags(int64_t pts, RGY_FRAME_FLAGS f) : timestamp(pts), flags(f) {};
    };
    MFXVideoDECODE *m_dec;
    mfxVideoParam& m_mfxDecParams;
    RGYInput *m_input;
    bool m_skipAV1C;
    bool m_getNextBitstream;
    int m_decFrameOutCount;
    int m_decRemoveRemainingBytesWarnCount; // removing %d bytes from input bitstream not read by decoder の表示回数
    int64_t m_firstPts;
    int64_t m_endPts; // 並列処理時用の終了時刻 (この時刻は含まないようにする) -1の場合は制限なし(最後まで)
    RGYBitstream m_decInputBitstream;
    RGYQueueMPMP<RGYFrameDataMetadata*> m_queueHDR10plusMetadata;
    RGYQueueMPMP<FrameFlags> m_dataFlag;
public:
    PipelineTaskMFXDecode(MFXVideoSession *mfxSession, int outMaxQueueSize, MFXVideoDECODE *mfxdec, mfxVideoParam& decParams, bool skipAV1C, int64_t endPts, RGYInput *input, mfxVersion mfxVer, std::shared_ptr<RGYLog> log)
        : PipelineTask(PipelineTaskType::MFXDEC, outMaxQueueSize, mfxSession, mfxVer, log), m_dec(mfxdec), m_mfxDecParams(decParams), m_input(input), m_skipAV1C(skipAV1C), m_getNextBitstream(true), m_decFrameOutCount(0), m_decRemoveRemainingBytesWarnCount(0), m_firstPts(-1), m_endPts(endPts), m_decInputBitstream(), m_queueHDR10plusMetadata(), m_dataFlag() {
        m_decInputBitstream.init(AVCODEC_READER_INPUT_BUF_SIZE);
        m_dataFlag.init();
        //TimeStampはQSVに自動的に計算させる
        m_decInputBitstream.setPts(MFX_TIMESTAMP_UNKNOWN);
        //ヘッダーがあれば読み込んでおく
        if (input) {
            input->GetHeader(&m_decInputBitstream);
        }
        m_queueHDR10plusMetadata.init(256);
    };
    virtual ~PipelineTaskMFXDecode() {
        m_queueHDR10plusMetadata.close([](RGYFrameDataMetadata **ptr) { if (*ptr) { delete *ptr; *ptr = nullptr; }; });
        m_decInputBitstream.clear();
    };
    void setDec(MFXVideoDECODE *mfxdec) { m_dec = mfxdec; };
    virtual void setStopWatch() override {
        m_stopwatch = std::make_unique<PipelineTaskStopWatch>(
            std::vector<tstring>{ _T("LoadNextFrame"), _T("getWorkSurf"), _T("DecodeFrameAsync"), _T("DecoderBusy"), _T("PushQueue") },
            std::vector<tstring>{_T("")}
        );
    }

    virtual std::optional<mfxFrameAllocRequest> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<mfxFrameAllocRequest> requiredSurfOut() override {
        mfxFrameAllocRequest allocRequest = { 0 };
        auto err = err_to_rgy(m_dec->QueryIOSurf(&m_mfxDecParams, &allocRequest));
        if (err < RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("  Failed to get required buffer size for %s: %s\n"), getPipelineTaskTypeName(m_type), get_err_mes(err));
            return std::nullopt;
        } else if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_WARN, _T("  surface alloc request for %s: %s\n"), getPipelineTaskTypeName(m_type), get_err_mes(err));
        }
        PrintMes(RGY_LOG_DEBUG, _T("  %s required buffer: %d [%s]\n"), getPipelineTaskTypeName(m_type), allocRequest.NumFrameSuggested, qsv_memtype_str(allocRequest.Type).c_str());
        return std::optional<mfxFrameAllocRequest>(allocRequest);
    }
    virtual RGY_ERR getOutputFrameInfo(mfxFrameInfo& info) override {
        info = m_mfxDecParams.mfx.FrameInfo;
        return RGY_ERR_NONE;
    }
    //データを使用すると、bitstreamのsizeを0にする
    //これを確認すること
    RGY_ERR sendFrame(RGYBitstream *bitstream) {
        if (bitstream) {
            //m_DecInputBitstream.size() > 0のときにbitstreamを連結してしまうと
            //環境によっては正常にフレームが取り出せなくなることがある
            if (m_getNextBitstream && m_decInputBitstream.size() <= 1) {
                m_decInputBitstream.append(bitstream);
                m_decInputBitstream.setPts(bitstream->pts());
                bitstream->setSize(0);
                bitstream->setOffset(0);
            }
        } else {
            //bitstream == nullptrの場合はflush
            //ただ、m_decInputBitstreamにデータが残っている場合はまずそちらを転送する
            if (m_decInputBitstream.size() == 0) {
                m_getNextBitstream = false; // m_decInputBitstreamのデータがなくなったらflushさせる
            }
        }
        return sendBitstream();
    }
    virtual RGY_ERR sendFrame([[maybe_unused]] std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (m_stopwatch) m_stopwatch->set(0);
        if (m_getNextBitstream
            //m_DecInputBitstream.size() > 0のときにbitstreamを連結してしまうと
            //環境によっては正常にフレームが取り出せなくなることがある
            //これを避けるため、m_DecInputBitstream.size() == 0のときのみbitstreamを取得する
            //これにより GetNextFrame / SetNextFrame の回数が異常となり、
            //GetNextFrameのロックが抜けれらなくなる場合がある。
            //HWデコード時、本来GetNextFrameのロックは必要ないので、
            //これを無視する実装も併せて行った。
            && (m_decInputBitstream.size() <= 1)) {
            auto ret = m_input->LoadNextFrame(nullptr);
            if (ret != RGY_ERR_NONE && ret != RGY_ERR_MORE_DATA && ret != RGY_ERR_MORE_BITSTREAM) {
                PrintMes(RGY_LOG_ERROR, _T("Error in reader: %s.\n"), get_err_mes(ret));
                return ret;
            }
            //この関数がMFX_ERR_NONE以外を返せば、入力ビットストリームは終了
            ret = m_input->GetNextBitstream(&m_decInputBitstream);
            if (ret == RGY_ERR_MORE_BITSTREAM) {
                m_getNextBitstream = false;
                return ret; //入力ビットストリームは終了
            }
            if (ret != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("Error on getting video bitstream: %s.\n"), get_err_mes(ret));
                return ret;
            }
            for (auto& frameData : m_decInputBitstream.getFrameDataList()) {
                if (frameData->dataType() == RGY_FRAME_DATA_HDR10PLUS) {
                    auto ptr = dynamic_cast<RGYFrameDataHDR10plus*>(frameData);
                    if (ptr) {
                        m_queueHDR10plusMetadata.push(new RGYFrameDataHDR10plus(*ptr));
                    }
                } else if (frameData->dataType() == RGY_FRAME_DATA_DOVIRPU) {
                    auto ptr = dynamic_cast<RGYFrameDataDOVIRpu*>(frameData);
                    if (ptr) {
                        m_queueHDR10plusMetadata.push(new RGYFrameDataDOVIRpu(*ptr));
                    }
                }
            }
            const auto flags = FrameFlags(m_decInputBitstream.pts(), (RGY_FRAME_FLAGS)m_decInputBitstream.dataflag());
            m_dataFlag.push(flags);
        }
        if (m_stopwatch) m_stopwatch->add(0, 0);
        return sendBitstream();
    }
protected:
    RGY_ERR sendBitstream() {
        m_getNextBitstream |= m_decInputBitstream.size() > 0;

        //デコードも行う場合は、デコード用のフレームをpSurfVppInかpSurfEncInから受け取る
        auto surfDecWork = getWorkSurf();
        if (surfDecWork == nullptr) {
            PrintMes(RGY_LOG_ERROR, _T("failed to get work surface for decoder.\n"));
            return RGY_ERR_NOT_ENOUGH_BUFFER;
        }
        surfDecWork.frame()->clearDataList();
        mfxBitstream *inputBitstream = (m_getNextBitstream) ? &m_decInputBitstream.bitstream() : nullptr;
        auto mfxSurfDecWork = surfDecWork.mfx()->surf();
        if (!m_mfxDecParams.mfx.FrameInfo.FourCC) {
            //デコード前には、デコード用のパラメータでFrameInfoを更新
            copy_crop_info(mfxSurfDecWork, &m_mfxDecParams.mfx.FrameInfo);
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_9)
            && (m_mfxDecParams.mfx.CodecId == MFX_CODEC_VP8 || m_mfxDecParams.mfx.CodecId == MFX_CODEC_VP9)) { // VP8/VP9ではこの処理が必要
            if (mfxSurfDecWork->Info.BitDepthLuma == 0 || mfxSurfDecWork->Info.BitDepthChroma == 0) {
                mfxSurfDecWork->Info.BitDepthLuma = m_mfxDecParams.mfx.FrameInfo.BitDepthLuma;
                mfxSurfDecWork->Info.BitDepthChroma = m_mfxDecParams.mfx.FrameInfo.BitDepthChroma;
            }
        }
        if (inputBitstream != nullptr) {
            if (m_skipAV1C && m_mfxDecParams.mfx.CodecId == MFX_CODEC_AV1 && inputBitstream->DataLength > 4 && (inputBitstream->Data[0] & 0x80)) {
                // AV1ではそのままのヘッダだと、Decodeに失敗する場合がある QSVEnc #122
                // その場合、4byte飛ばすと読めるかも?
                // https://github.com/FFmpeg/FFmpeg/commit/ffd1316e441a8310cf1746d86fed165e17e10018
                // https://aomediacodec.github.io/av1-isobmff/
                inputBitstream->DataOffset += 4;
                inputBitstream->DataLength -= 4;
            }
            if (inputBitstream->TimeStamp == (mfxU64)AV_NOPTS_VALUE) {
                inputBitstream->TimeStamp = (mfxU64)MFX_TIMESTAMP_UNKNOWN;
            } else if (m_firstPts < 0) {
                m_firstPts = inputBitstream->TimeStamp;
            }
            inputBitstream->DecodeTimeStamp = MFX_TIMESTAMP_UNKNOWN;
        }
        mfxSurfDecWork->Data.TimeStamp = (mfxU64)MFX_TIMESTAMP_UNKNOWN;
        mfxSurfDecWork->Data.DataFlag |= MFX_FRAMEDATA_ORIGINAL_TIMESTAMP;
        m_inFrames++;
        if (m_stopwatch) m_stopwatch->add(0, 1);

        mfxStatus dec_sts = MFX_ERR_NONE;
        mfxSyncPoint lastSyncP = nullptr;
        mfxFrameSurface1 *surfDecOut = nullptr;
        for (int i = 0; ; i++) {
            if (m_stopwatch) m_stopwatch->set(0);
            const auto inputDataLen = (inputBitstream) ? inputBitstream->DataLength : 0;
            mfxSyncPoint decSyncPoint = nullptr;
            dec_sts = m_dec->DecodeFrameAsync(inputBitstream, mfxSurfDecWork, &surfDecOut, &decSyncPoint);
            lastSyncP = decSyncPoint;
            if (m_stopwatch) m_stopwatch->add(0, 2);

            if (MFX_ERR_NONE < dec_sts && !decSyncPoint) {
                if (MFX_WRN_DEVICE_BUSY == dec_sts)
                    sleep_hybrid(i);
                if (i > 1024 * 1024 * 30) {
                    PrintMes(RGY_LOG_ERROR, _T("device kept on busy for 30s, unknown error occurred.\n"));
                    return RGY_ERR_GPU_HANG;
                }
            } else if (MFX_ERR_NONE < dec_sts && decSyncPoint) {
                dec_sts = MFX_ERR_NONE; //出力があれば、警告は無視する
                break;
            } else if (dec_sts < MFX_ERR_NONE && (dec_sts != MFX_ERR_MORE_DATA && dec_sts != MFX_ERR_MORE_SURFACE)) {
                PrintMes(RGY_LOG_ERROR, _T("DecodeFrameAsync error: %s.\n"), get_err_mes(dec_sts));
                break;
            } else {
                //pInputBitstreamの長さがDecodeFrameAsyncを経ても全く変わっていない場合は、そのデータは捨てる
                //これを行わないとデコードが止まってしまう
                if (dec_sts == MFX_ERR_MORE_DATA && inputBitstream && inputBitstream->DataLength == inputDataLen) {
                    PrintMes((inputDataLen >= 10) ? RGY_LOG_WARN : ((m_decRemoveRemainingBytesWarnCount >= 10) ? RGY_LOG_TRACE : RGY_LOG_DEBUG),
                        _T("DecodeFrameAsync: removing %d bytes from input bitstream not read by decoder.\n"), inputDataLen);
                    inputBitstream->DataLength = 0;
                    inputBitstream->DataOffset = 0;
                    if (m_decRemoveRemainingBytesWarnCount == 10) {
                        PrintMes(RGY_LOG_DEBUG, _T("DecodeFrameAsync:   This message was shown 10 times, will be only shown in trace log after this when removing bytes is samller than 10bytes.\n"));
                    }
                    m_decRemoveRemainingBytesWarnCount++;
                }
                break;
            }
            if (m_stopwatch) m_stopwatch->add(0, 3);
        }
        if (m_stopwatch) m_stopwatch->add(0, 3);
        if (m_endPts >= 0
            && surfDecOut != nullptr
            && surfDecOut->Data.TimeStamp >= (uint64_t)m_endPts) { // m_endPtsは含まないようにする(重要)
            m_getNextBitstream = false;
            return RGY_ERR_MORE_BITSTREAM; //入力ビットストリームは終了
        }
        if (surfDecOut != nullptr && lastSyncP != nullptr
            // 最初のフレームはOpenGOPのBフレームのために投入フレーム以前のデータの場合があるので、その場合はフレームを無視する
            && (m_firstPts <= (int64_t)surfDecOut->Data.TimeStamp || m_decFrameOutCount > 0)) {
            auto taskSurf = useTaskSurf(surfDecOut);
            const auto picstruct = taskSurf.mfx()->surf()->Info.PicStruct;
            auto flags = RGY_FRAME_FLAG_NONE;
            // RFFの場合、MFX_PICSTRUCT_PROGRESSIVEに加えて、MFX_PICSTRUCT_FIELD_TFFまたはMFX_PICSTRUCT_FIELD_BFF、MFX_PICSTRUCT_FIELD_REPEATEDが立っている
            // picstructにはprogressiveを設定し、flagsにRFF関係の情報を設定しなおす
            // この情報の取得には、m_mfxDecParams.mfx.ExtendedPicStruct = 1 としてデコーダを初期化する必要がある
            if ((picstruct & MFX_PICSTRUCT_PROGRESSIVE) && (picstruct & (MFX_PICSTRUCT_FIELD_TFF| MFX_PICSTRUCT_FIELD_BFF))) {
                taskSurf.frame()->setPicstruct(RGY_PICSTRUCT_FRAME);
                if (picstruct & MFX_PICSTRUCT_FIELD_REPEATED) {
                    flags |= RGY_FRAME_FLAG_RFF;
                }
            }
            if (picstruct & MFX_PICSTRUCT_FIELD_TFF) {
                flags |= RGY_FRAME_FLAG_RFF_TFF;
            }
            if (picstruct & MFX_PICSTRUCT_FIELD_BFF) {
                flags |= RGY_FRAME_FLAG_RFF_BFF;
            }
            taskSurf.frame()->setInputFrameId(m_decFrameOutCount++);

            if (getDataFlag(surfDecOut->Data.TimeStamp) & RGY_FRAME_FLAG_RFF) {
                flags |= RGY_FRAME_FLAG_RFF;
            }
            taskSurf.frame()->setFlags(flags);
            taskSurf.frame()->setDuration(0); // QSVはdurationを返さない

            taskSurf.frame()->clearDataList();
            if (auto data = getMetadata(RGY_FRAME_DATA_HDR10PLUS, surfDecOut->Data.TimeStamp); data) {
                taskSurf.frame()->dataList().push_back(data);
            }
            if (auto data = getMetadata(RGY_FRAME_DATA_DOVIRPU, surfDecOut->Data.TimeStamp); data) {
                taskSurf.frame()->dataList().push_back(data);
            }
            m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_mfxSession, taskSurf, lastSyncP));
        }
        if (m_stopwatch) m_stopwatch->add(0, 4);
        return err_to_rgy(dec_sts);
    }
    RGY_FRAME_FLAGS getDataFlag(const int64_t timestamp) {
        FrameFlags pts_flag;
        while (m_dataFlag.front_copy_no_lock(&pts_flag)) {
            if (pts_flag.timestamp < timestamp || pts_flag.timestamp == AV_NOPTS_VALUE) {
                m_dataFlag.pop();
            } else {
                break;
            }
        }
        size_t queueSize = m_dataFlag.size();
        for (uint32_t i = 0; i < queueSize; i++) {
            if (m_dataFlag.copy(&pts_flag, i, &queueSize)) {
                if (pts_flag.timestamp == timestamp) {
                    return pts_flag.flags;
                }
            }
        }
        return RGY_FRAME_FLAG_NONE;
    }
    std::shared_ptr<RGYFrameData> getMetadata(const RGYFrameDataType datatype, const int64_t timestamp) {
        std::shared_ptr<RGYFrameData> frameData;
        RGYFrameDataMetadata *frameDataPtr = nullptr;
        while (m_queueHDR10plusMetadata.front_copy_no_lock(&frameDataPtr)) {
            if (frameDataPtr->timestamp() < timestamp) {
                m_queueHDR10plusMetadata.pop();
                delete frameDataPtr;
            } else {
                break;
            }
        }
        size_t queueSize = m_queueHDR10plusMetadata.size();
        for (uint32_t i = 0; i < queueSize; i++) {
            if (m_queueHDR10plusMetadata.copy(&frameDataPtr, i, &queueSize)) {
                if (frameDataPtr->timestamp() == timestamp && frameDataPtr->dataType() == datatype) {
                    if (frameDataPtr->dataType() == RGY_FRAME_DATA_HDR10PLUS) {
                        auto ptr = dynamic_cast<RGYFrameDataHDR10plus*>(frameDataPtr);
                        if (ptr) {
                            frameData = std::make_shared<RGYFrameDataHDR10plus>(*ptr);
                        }
                    } else if (frameDataPtr->dataType() == RGY_FRAME_DATA_DOVIRPU) {
                        auto ptr = dynamic_cast<RGYFrameDataDOVIRpu*>(frameDataPtr);
                        if (ptr) {
                            frameData = std::make_shared<RGYFrameDataDOVIRpu>(*ptr);
                        }
                    }
                    break;
                }
            }
        }
        return frameData;
    };
};

class PipelineTaskCheckPTS : public PipelineTask {
protected:
    rgy_rational<int> m_srcTimebase;
    rgy_rational<int> m_outputTimebase;
    bool m_vpp_rff;
    bool m_vpp_afs_rff_aware;
    RGYAVSync m_avsync;
    bool m_timestampPassThrough;
    int64_t m_outFrameDuration; //(m_outputTimebase基準)
    int64_t m_tsOutFirst;     //(m_outputTimebase基準)
    int64_t m_tsOutEstimated; //(m_outputTimebase基準)
    int64_t m_tsPrev;         //(m_outputTimebase基準)
public:
    PipelineTaskCheckPTS(MFXVideoSession *mfxSession, rgy_rational<int> srcTimebase, rgy_rational<int> outputTimebase, int64_t outFrameDuration, RGYAVSync avsync, bool timestampPassThrough, bool vpp_afs_rff_aware, mfxVersion mfxVer, std::shared_ptr<RGYLog> log) :
        PipelineTask(PipelineTaskType::CHECKPTS, /*outMaxQueueSize = */ 0 /*常に0である必要がある*/, mfxSession, mfxVer, log),
        m_srcTimebase(srcTimebase), m_outputTimebase(outputTimebase), m_vpp_rff(false), m_vpp_afs_rff_aware(vpp_afs_rff_aware), m_avsync(avsync), m_timestampPassThrough(timestampPassThrough), m_outFrameDuration(outFrameDuration), m_tsOutFirst(-1), m_tsOutEstimated(0), m_tsPrev(-1) {
    };
    virtual ~PipelineTaskCheckPTS() {};

    virtual void setStopWatch() override {
        m_stopwatch = std::make_unique<PipelineTaskStopWatch>(
            std::vector<tstring>{ _T("") },
            std::vector<tstring>{_T("")}
        );
    }
    virtual bool isPassThrough() const override {
        // そのまま渡すのでPassThrough
        return true;
    }
    static const int MAX_FORCECFR_INSERT_FRAMES = 1024; //事実上の無制限
public:
    virtual std::optional<mfxFrameAllocRequest> requiredSurfIn() override {
        return std::nullopt;
    };
    virtual std::optional<mfxFrameAllocRequest> requiredSurfOut() override {
        return std::nullopt;
    };
    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (m_stopwatch) m_stopwatch->set(0);
        if (!frame) {
            //PipelineTaskCheckPTSは、getOutputで1フレームずつしか取り出さない
            //そのためm_outQeueueにまだフレームが残っている可能性がある
            return (m_outQeueue.size() > 0) ? RGY_ERR_MORE_SURFACE : RGY_ERR_MORE_DATA;
        }
        int64_t outPtsSource = m_tsOutEstimated; //(m_outputTimebase基準)
        int64_t outDuration = m_outFrameDuration; //入力fpsに従ったduration

        PipelineTaskOutputSurf *taskSurf = dynamic_cast<PipelineTaskOutputSurf *>(frame.get());
        if (taskSurf == nullptr) {
            PrintMes(RGY_LOG_ERROR, _T("Invalid frame type: failed to cast to PipelineTaskOutputSurf.\n"));
            return RGY_ERR_UNSUPPORTED;
        }

        if ((m_srcTimebase.n() > 0 && m_srcTimebase.is_valid())
            && ((m_avsync & (RGY_AVSYNC_VFR | RGY_AVSYNC_FORCE_CFR)) || m_vpp_rff || m_vpp_afs_rff_aware || m_timestampPassThrough)) {
            //CFR仮定ではなく、オリジナルの時間を見る
            const auto srcTimestamp = taskSurf->surf().frame()->timestamp();
            outPtsSource = rational_rescale(srcTimestamp, m_srcTimebase, m_outputTimebase);
            if (taskSurf->surf().frame()->duration() > 0 && (m_avsync | RGY_AVSYNC_FORCE_CFR) != RGY_AVSYNC_FORCE_CFR) {
                outDuration = rational_rescale(taskSurf->surf().frame()->duration(), m_srcTimebase, m_outputTimebase);
                taskSurf->surf().frame()->setDuration(outDuration);
            }
        }
        PrintMes(RGY_LOG_TRACE, _T("check_pts(%d/%d): nOutEstimatedPts %lld, outPtsSource %lld, outDuration %d\n"), taskSurf->surf().frame()->inputFrameId(), m_inFrames, m_tsOutEstimated, outPtsSource, outDuration);
        if (m_tsOutFirst < 0) {
            m_tsOutFirst = outPtsSource; //最初のpts
            PrintMes(RGY_LOG_TRACE, _T("check_pts: m_tsOutFirst %lld\n"), outPtsSource);
        }
        if (!m_timestampPassThrough) {
            //最初のptsを0に修正
            outPtsSource -= m_tsOutFirst;
        }

        if ((m_avsync & RGY_AVSYNC_VFR) || m_vpp_rff || m_vpp_afs_rff_aware) {
            if (m_vpp_rff || m_vpp_afs_rff_aware) {
                if (std::abs(outPtsSource - m_tsOutEstimated) >= 32 * m_outFrameDuration) {
                    PrintMes(RGY_LOG_TRACE, _T("check_pts: detected gap %lld, changing offset.\n"), outPtsSource, std::abs(outPtsSource - m_tsOutEstimated));
                    //timestampに一定以上の差があればそれを無視する
                    m_tsOutFirst += (outPtsSource - m_tsOutEstimated); //今後の位置合わせのための補正
                    outPtsSource = m_tsOutEstimated;
                    PrintMes(RGY_LOG_TRACE, _T("check_pts:   changed to m_tsOutFirst %lld, outPtsSource %lld.\n"), m_tsOutFirst, outPtsSource);
                }
                auto ptsDiff = outPtsSource - m_tsOutEstimated;
                if (ptsDiff <= std::min<int64_t>(-1, -1 * m_outFrameDuration * 7 / 8)) {
                    //間引きが必要
                    PrintMes(RGY_LOG_TRACE, _T("check_pts(%d):   skipping frame (vfr)\n"), taskSurf->surf().frame()->inputFrameId());
                    return RGY_ERR_MORE_SURFACE;
                }
                // 少しのずれはrffによるものとみなし、基準値を修正する
                m_tsOutEstimated = outPtsSource;
            }
#if 0
            if (streamIn) {
                //cuvidデコード時は、timebaseの分子はかならず1なので、streamIn->time_baseとズレているかもしれないのでオリジナルを計算
                const auto orig_pts = rational_rescale(taskSurf->surf()->Data.TimeStamp, m_srcTimebase, to_rgy(streamIn->time_base));
                //ptsからフレーム情報を取得する
                const auto framePos = pReader->GetFramePosList()->findpts(orig_pts, &nInputFramePosIdx);
                PrintMes(RGY_LOG_TRACE, _T("check_pts(%d):   estimetaed orig_pts %lld, framePos %d\n"), taskSurf->surf().frame()->inputFrameId(), orig_pts, framePos.poc);
                if (framePos.poc != FRAMEPOS_POC_INVALID && framePos.duration > 0) {
                    //有効な値ならオリジナルのdurationを使用する
                    outDuration = rational_rescale(framePos.duration, to_rgy(streamIn->time_base), m_outputTimebase);
                    PrintMes(RGY_LOG_TRACE, _T("check_pts(%d):   changing duration to original: %d\n"), taskSurf->surf().frame()->inputFrameId(), outDuration);
                }
            }
#endif
        }
        if (m_avsync & RGY_AVSYNC_FORCE_CFR) {
            if (std::abs(outPtsSource - m_tsOutEstimated) >= CHECK_PTS_MAX_INSERT_FRAMES * m_outFrameDuration) {
                //timestampに一定以上の差があればそれを無視する
                m_tsOutFirst += (outPtsSource - m_tsOutEstimated); //今後の位置合わせのための補正
                outPtsSource = m_tsOutEstimated;
                PrintMes(RGY_LOG_WARN, _T("Big Gap was found between 2 frames, avsync might be corrupted.\n"));
                PrintMes(RGY_LOG_TRACE, _T("check_pts:   changed to m_tsOutFirst %lld, outPtsSource %lld.\n"), m_tsOutFirst, outPtsSource);
            }
            auto ptsDiff = outPtsSource - m_tsOutEstimated;
            if (ptsDiff <= std::min<int64_t>(-1, -1 * m_outFrameDuration * 7 / 8)) {
                //間引きが必要
                PrintMes(RGY_LOG_DEBUG, _T("Drop frame: framepts %lld, estimated next %lld, diff %lld [%.1f]\n"), outPtsSource, m_tsOutEstimated, ptsDiff, ptsDiff / (double)m_outFrameDuration);
                return RGY_ERR_MORE_SURFACE;
            }
            while (ptsDiff >= std::max<int64_t>(1, m_outFrameDuration * 7 / 8)) {
                PrintMes(RGY_LOG_DEBUG, _T("Insert frame: framepts %lld, estimated next %lld, diff %lld [%.1f]\n"), outPtsSource, m_tsOutEstimated, ptsDiff, ptsDiff / (double)m_outFrameDuration);
                //水増しが必要
                PipelineTaskSurface surfVppOut = taskSurf->surf();
                mfxSyncPoint lastSyncPoint = taskSurf->syncpoint();
                surfVppOut.frame()->setInputFrameId(taskSurf->surf().frame()->inputFrameId());
                surfVppOut.frame()->setTimestamp(m_tsOutEstimated);
                surfVppOut.frame()->setDuration(m_outFrameDuration);
                //timestampの上書き情報
                //surfVppOut内部のmfxSurface1自体は同じデータを指すため、複数のタイムスタンプを持つことができない
                //この問題をm_outQeueueのPipelineTaskOutput(これは個別)に与えるPipelineTaskOutputDataCheckPtsの値で、
                //PipelineTaskCheckPTS::getOutput時にtimestampを変更するようにする
                //そのため、checkptsからgetOutputしたフレームは
                //(次にPipelineTaskCheckPTS::getOutputを呼ぶより前に)直ちに後続タスクに投入するよう制御する必要がある
                std::unique_ptr<PipelineTaskOutputDataCustom> timestampOverride(new PipelineTaskOutputDataCheckPts(m_tsOutEstimated));
                m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_mfxSession, surfVppOut, lastSyncPoint, timestampOverride));
                m_tsOutEstimated += m_outFrameDuration;
                ptsDiff = outPtsSource - m_tsOutEstimated;
            }
            outPtsSource = m_tsOutEstimated;
        }
        if (m_tsPrev >= outPtsSource) {
            if (m_tsPrev - outPtsSource >= MAX_FORCECFR_INSERT_FRAMES * m_outFrameDuration) {
                PrintMes(RGY_LOG_DEBUG, _T("check_pts: previous pts %lld, current pts %lld, estimated pts %lld, m_tsOutFirst %lld, changing offset.\n"), m_tsPrev, outPtsSource, m_tsOutEstimated, m_tsOutFirst);
                m_tsOutFirst += (outPtsSource - m_tsOutEstimated); //今後の位置合わせのための補正
                outPtsSource = m_tsOutEstimated;
                PrintMes(RGY_LOG_DEBUG, _T("check_pts:   changed to m_tsOutFirst %lld, outPtsSource %lld.\n"), m_tsOutFirst, outPtsSource);
            } else {
                if (m_avsync & RGY_AVSYNC_FORCE_CFR) {
                    //間引きが必要
                    PrintMes(RGY_LOG_WARN, _T("check_pts(%d/%d): timestamp of video frame is smaller than previous frame, skipping frame: previous pts %lld, current pts %lld.\n"),
                        taskSurf->surf().frame()->inputFrameId(), m_inFrames, m_tsPrev, outPtsSource);
                    return RGY_ERR_MORE_SURFACE;
                } else {
                    const auto origPts = outPtsSource;
                    outPtsSource = m_tsPrev + std::max<int64_t>(1, m_outFrameDuration / 4);
                    PrintMes(RGY_LOG_WARN, _T("check_pts(%d/%d): timestamp of video frame is smaller than previous frame, changing pts: %lld -> %lld (previous pts %lld).\n"),
                        taskSurf->surf().frame()->inputFrameId(), m_inFrames, origPts, outPtsSource, m_tsPrev);
                }
            }
        }

        //次のフレームのptsの予想
        m_inFrames++;
        m_tsOutEstimated += outDuration;
        m_tsPrev = outPtsSource;
        PipelineTaskSurface outSurf = taskSurf->surf();
        mfxSyncPoint lastSyncPoint = taskSurf->syncpoint();
        outSurf.frame()->setInputFrameId(taskSurf->surf().frame()->inputFrameId());
        outSurf.frame()->setTimestamp(outPtsSource);
        outSurf.frame()->setDuration(outDuration);
        std::unique_ptr<PipelineTaskOutputDataCustom> timestampOverride(new PipelineTaskOutputDataCheckPts(outPtsSource));
        m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_mfxSession, outSurf, lastSyncPoint, timestampOverride));
        if (m_stopwatch) m_stopwatch->add(0, 0);
        return RGY_ERR_NONE;
    }
    //checkptsではtimestampを上書きするため特別に常に1フレームしか取り出さない
    //これは--avsync frocecfrでフレームを参照コピーする際、
    //mfxSurface1自体は同じデータを指すため、複数のタイムスタンプを持つことができないため、
    //1フレームずつgetOutputし、都度タイムスタンプを上書きしてすぐに後続のタスクに投入してタイムスタンプを反映させる必要があるため
    virtual std::vector<std::unique_ptr<PipelineTaskOutput>> getOutput(const bool sync) override {
        if (m_stopwatch) m_stopwatch->set(1);
        std::vector<std::unique_ptr<PipelineTaskOutput>> output;
        if ((int)m_outQeueue.size() > m_outMaxQueueSize) {
            auto out = std::move(m_outQeueue.front());
            m_outQeueue.pop_front();
            if (sync) {
                out->waitsync();
            }
            out->depend_clear();
            if (out->customdata() != nullptr) {
                const auto dataCheckPts = dynamic_cast<const PipelineTaskOutputDataCheckPts *>(out->customdata());
                if (dataCheckPts == nullptr) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to get timestamp data, timestamp might be inaccurate!\n"));
                } else {
                    PipelineTaskOutputSurf *outSurf = dynamic_cast<PipelineTaskOutputSurf *>(out.get());
                    outSurf->surf().frame()->setTimestamp(dataCheckPts->timestampOverride());
                }
            }
            m_outFrames++;
            output.push_back(std::move(out));
        }
        if (output.size() > 1) {
            PrintMes(RGY_LOG_ERROR, _T("output queue more than 1, invalid!\n"));
        }
        if (m_stopwatch) m_stopwatch->add(1, 0);
        return output;
    }
};

class PipelineTaskAudio : public PipelineTask {
protected:
    RGYInput *m_input;
    std::map<int, std::shared_ptr<RGYOutputAvcodec>> m_pWriterForAudioStreams;
    std::map<int, RGYFilter *> m_filterForStreams;
    std::vector<std::shared_ptr<RGYInput>> m_audioReaders;
public:
    PipelineTaskAudio(RGYInput *input, std::vector<std::shared_ptr<RGYInput>>& audioReaders, std::vector<std::shared_ptr<RGYOutput>>& fileWriterListAudio, std::vector<VppVilterBlock>& vpFilters, int outMaxQueueSize, mfxVersion mfxVer, std::shared_ptr<RGYLog> log) :
        PipelineTask(PipelineTaskType::AUDIO, outMaxQueueSize, nullptr, mfxVer, log),
        m_input(input), m_audioReaders(audioReaders) {
        //streamのindexから必要なwriteへのポインタを返すテーブルを作成
        for (auto writer : fileWriterListAudio) {
            auto pAVCodecWriter = std::dynamic_pointer_cast<RGYOutputAvcodec>(writer);
            if (pAVCodecWriter) {
                auto trackIdList = pAVCodecWriter->GetStreamTrackIdList();
                for (auto trackID : trackIdList) {
                    m_pWriterForAudioStreams[trackID] = pAVCodecWriter;
                }
            }
        }
        //streamのtrackIdからパケットを送信するvppフィルタへのポインタを返すテーブルを作成
        for (auto& filterBlock : vpFilters) {
            if (filterBlock.type == VppFilterType::FILTER_OPENCL) {
                for (auto& filter : filterBlock.vppcl) {
                    const auto targetTrackId = filter->targetTrackIdx();
                    if (targetTrackId != 0) {
                        m_filterForStreams[targetTrackId] = filter.get();
                    }
                }
            }
        }
    };
    virtual ~PipelineTaskAudio() {};

    virtual bool isPassThrough() const override { return true; }

    virtual std::optional<mfxFrameAllocRequest> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<mfxFrameAllocRequest> requiredSurfOut() override { return std::nullopt; };


    void flushAudio() {
        PrintMes(RGY_LOG_DEBUG, _T("Clear packets in writer...\n"));
        std::set<RGYOutputAvcodec*> writers;
        for (const auto& [ streamid, writer ] : m_pWriterForAudioStreams) {
            auto pWriter = std::dynamic_pointer_cast<RGYOutputAvcodec>(writer);
            if (pWriter != nullptr) {
                writers.insert(pWriter.get());
            }
        }
        for (const auto& writer : writers) {
            //エンコーダなどにキャッシュされたパケットを書き出す
            writer->WriteNextPacket(nullptr);
        }
    }

    RGY_ERR extractAudio(int inputFrames) {
        RGY_ERR ret = RGY_ERR_NONE;
#if ENABLE_AVSW_READER
        if (m_pWriterForAudioStreams.size() > 0) {
#if ENABLE_SM_READER
            RGYInputSM *pReaderSM = dynamic_cast<RGYInputSM *>(m_input);
            const int droppedInAviutl = (pReaderSM != nullptr) ? pReaderSM->droppedFrames() : 0;
#else
            const int droppedInAviutl = 0;
#endif

            auto packetList = m_input->GetStreamDataPackets(inputFrames + droppedInAviutl);

            //音声ファイルリーダーからのトラックを結合する
            for (const auto& reader : m_audioReaders) {
                vector_cat(packetList, reader->GetStreamDataPackets(inputFrames + droppedInAviutl));
            }
            //パケットを各Writerに分配する
            for (uint32_t i = 0; i < packetList.size(); i++) {
                AVPacket *pkt = packetList[i];
                const int nTrackId = pktFlagGetTrackID(pkt);
                const bool sendToFilter = m_filterForStreams.count(nTrackId) > 0;
                const bool sendToWriter = m_pWriterForAudioStreams.count(nTrackId) > 0;
                if (sendToFilter) {
                    AVPacket *pktToFilter = nullptr;
                    if (sendToWriter) {
                        pktToFilter = av_packet_clone(pkt);
                    } else {
                        std::swap(pktToFilter, pkt);
                    }
                    auto err = m_filterForStreams[nTrackId]->addStreamPacket(pktToFilter);
                    if (err != RGY_ERR_NONE) {
                        return err;
                    }
                }
                if (sendToWriter) {
                    auto pWriter = m_pWriterForAudioStreams[nTrackId];
                    if (pWriter == nullptr) {
                        PrintMes(RGY_LOG_ERROR, _T("Invalid writer found for %s track #%d\n"), char_to_tstring(trackMediaTypeStr(nTrackId)).c_str(), trackID(nTrackId));
                        return RGY_ERR_NOT_FOUND;
                    }
                    auto err = pWriter->WriteNextPacket(pkt);
                    if (err != RGY_ERR_NONE) {
                        return err;
                    }
                    pkt = nullptr;
                }
                if (pkt != nullptr) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to find writer for %s track #%d\n"), char_to_tstring(trackMediaTypeStr(nTrackId)).c_str(), trackID(nTrackId));
                    return RGY_ERR_NOT_FOUND;
                }
            }
        }
#endif //ENABLE_AVSW_READER
        return ret;
    };

    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        m_inFrames++;
        auto err = extractAudio(m_inFrames);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        if (!frame) {
            flushAudio();
            return RGY_ERR_MORE_DATA;
        }
        PipelineTaskOutputSurf *taskSurf = dynamic_cast<PipelineTaskOutputSurf *>(frame.get());
        m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_mfxSession, taskSurf->surf(), taskSurf->syncpoint()));
        return RGY_ERR_NONE;
    }
};

class PipelineTaskParallelEncBitstream : public PipelineTask {
protected:
    RGYInput *m_input;
    int m_currentChunk; // いま並列処理の何番目を処理中か
    RGYTimestamp *m_encTimestamp;
    RGYTimecode *m_timecode;
    RGYParallelEnc *m_parallelEnc;
    EncodeStatus *m_encStatus;
    rgy_rational<int> m_encFps;
    rgy_rational<int> m_outputTimebase;
    std::unique_ptr<PipelineTaskAudio> m_taskAudio;
    std::unique_ptr<FILE, fp_deleter> m_fReader;
    int64_t m_firstPts; //最初のpts
    int64_t m_maxPts; // 最後のpts
    int64_t m_ptsOffset; // 分割出力間の(2分割目以降の)ptsのオフセット
    int64_t m_encFrameOffset; // 分割出力間の(2分割目以降の)エンコードフレームのオフセット
    int64_t m_inputFrameOffset; // 分割出力間の(2分割目以降の)エンコードフレームのオフセット
    int64_t m_maxEncFrameIdx; // 最後にエンコードしたフレームのindex
    int64_t m_maxInputFrameIdx; // 最後にエンコードしたフレームのindex
    RGYBitstream m_decInputBitstream; // 映像読み込み (ダミー)
    bool m_inputBitstreamEOF; // 映像側の読み込み終了フラグ (音声処理の終了も確認する必要があるため)
    RGYListRef<RGYBitstream> m_bitStreamOut;
    RGYDurationCheck m_durationCheck;
    bool m_tsDebug;
public:
    PipelineTaskParallelEncBitstream(RGYInput *input, RGYTimestamp *encTimestamp, RGYTimecode *timecode, RGYParallelEnc *parallelEnc, EncodeStatus *encStatus,
        rgy_rational<int> encFps, rgy_rational<int> outputTimebase,
        std::unique_ptr<PipelineTaskAudio>& taskAudio, int outMaxQueueSize, mfxVersion mfxVer, std::shared_ptr<RGYLog> log) :
        PipelineTask(PipelineTaskType::PECOLLECT, outMaxQueueSize, nullptr, mfxVer, log),
        m_input(input), m_currentChunk(-1), m_encTimestamp(encTimestamp), m_timecode(timecode),
        m_parallelEnc(parallelEnc), m_encStatus(encStatus), m_encFps(encFps), m_outputTimebase(outputTimebase),
        m_taskAudio(std::move(taskAudio)), m_fReader(std::unique_ptr<FILE, fp_deleter>(nullptr, fp_deleter())),
        m_firstPts(-1), m_maxPts(-1), m_ptsOffset(0), m_encFrameOffset(0), m_inputFrameOffset(0), m_maxEncFrameIdx(-1), m_maxInputFrameIdx(-1),
        m_decInputBitstream(), m_inputBitstreamEOF(false), m_bitStreamOut(), m_durationCheck(), m_tsDebug(false) {
        m_decInputBitstream.init(AVCODEC_READER_INPUT_BUF_SIZE);
        auto reader = dynamic_cast<RGYInputAvcodec*>(input);
        if (reader) {
            // 親側で不要なデコーダを終了させる、こうしないとavsw使用時に映像が無駄にデコードされてしまう
            reader->CloseVideoDecoder();
        }
    };
    virtual ~PipelineTaskParallelEncBitstream() {
        m_decInputBitstream.clear();
    };

    virtual bool isPassThrough() const override { return true; }

    virtual std::optional<mfxFrameAllocRequest> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<mfxFrameAllocRequest> requiredSurfOut() override { return std::nullopt; };
protected:
    RGY_ERR checkEncodeResult() {
        // まずそのエンコーダの終了を待機
        while (m_parallelEnc->waitProcessFinished(m_currentChunk, UPDATE_INTERVAL) != WAIT_OBJECT_0) {
            // 進捗表示の更新
            auto currentData = m_encStatus->GetEncodeData();
            m_encStatus->UpdateDisplay(currentData.progressPercent);
        }
        // 戻り値を確認
        auto procsts = m_parallelEnc->processReturnCode(m_currentChunk);
        if (!procsts.has_value()) { // そんなはずはないのだが、一応
            PrintMes(RGY_LOG_ERROR, _T("Unknown error in parallel enc: %d.\n"), m_currentChunk);
            return RGY_ERR_UNKNOWN;
        }
        if (procsts.value() != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("Error in parallel enc %d: %s\n"), m_currentChunk, get_err_mes(procsts.value()));
            return procsts.value();
        }
        return RGY_ERR_NONE;
    }

    RGY_ERR openNextFile() {
        if (m_currentChunk >= 0 && m_parallelEnc->cacheMode(m_currentChunk) == RGYParamParallelEncCache::Mem) {
            // メモリモードの場合は、まだそのエンコーダの戻り値をチェックしていないので、ここでチェック
            auto procsts = checkEncodeResult();
            if (procsts != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("Error in parallel enc %d: %s\n"), m_currentChunk, get_err_mes(procsts));
                return procsts;
            }
        }

        m_currentChunk++;
        if (m_currentChunk >= (int)m_parallelEnc->parallelCount()) {
            return RGY_ERR_MORE_BITSTREAM;
        }
        
        if (m_parallelEnc->cacheMode(m_currentChunk) == RGYParamParallelEncCache::File) {
            // 戻り値を確認
            auto procsts = checkEncodeResult();
            if (procsts != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("Error in parallel enc %d: %s\n"), m_currentChunk, get_err_mes(procsts));
                return procsts;
            }
            // ファイルを開く
            auto tmpPath = m_parallelEnc->tmpPath(m_currentChunk);
            if (tmpPath.empty()) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to get tmp path for parallel enc %d.\n"), m_currentChunk);
                return RGY_ERR_UNKNOWN;
            }
            m_fReader = std::unique_ptr<FILE, fp_deleter>(_tfopen(tmpPath.c_str(), _T("rb")), fp_deleter());
            if (m_fReader == nullptr) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to open file: %s\n"), tmpPath.c_str());
                return RGY_ERR_FILE_OPEN;
            }
        }
        //最初のファイルに対するptsの差を取り、それをtimebaseを変換して適用する
        const auto inputFrameInfo = m_input->GetInputFrameInfo();
        const auto inputFpsTimebase = rgy_rational<int>((int)inputFrameInfo.fpsD, (int)inputFrameInfo.fpsN);
        const auto srcTimebase = (m_input->getInputTimebase().n() > 0 && m_input->getInputTimebase().is_valid()) ? m_input->getInputTimebase() : inputFpsTimebase;
        // seek結果による入力ptsを用いて計算した本来のpts offset
        const auto ptsOffsetOrig = (m_firstPts < 0) ? 0 : rational_rescale(m_parallelEnc->getVideofirstKeyPts(m_currentChunk), srcTimebase, m_outputTimebase) - m_firstPts;
        // 直前のフレームから計算したpts offset(-1フレーム分) 最低でもこれ以上のoffsetがないといけない
        const auto ptsOffsetMax = (m_firstPts < 0) ? 0 : m_maxPts - m_firstPts;
        // フレームの長さを決める
        int64_t lastDuration = 0;
        const auto frameDuration = m_durationCheck.getDuration(lastDuration);
        // frameDuration のうち、登場回数が最も多いものを探す
        int mostFrequentDuration = 0;
        int64_t mostFrequentDurationCount = 0;
        int64_t totalFrameCount = 0;
        for (const auto& [duration, count] : frameDuration) {
            if (count > mostFrequentDurationCount) {
                mostFrequentDuration = duration;
                mostFrequentDurationCount = count;
            }
            totalFrameCount += count;
        }
        // フレーム長が1つしかない場合、あるいは登場頻度の高いフレーム長がある場合、そのフレーム長を採用する
        if (frameDuration.size() == 1 || ((totalFrameCount * 9 / 10) < mostFrequentDurationCount)) {
            m_ptsOffset = ptsOffsetMax + mostFrequentDuration;
        } else if (frameDuration.size() == 2) {
            if ((totalFrameCount * 7 / 10) < mostFrequentDurationCount || lastDuration != mostFrequentDuration) {
                m_ptsOffset = ptsOffsetMax + mostFrequentDuration;
            } else {
                int otherDuration = mostFrequentDuration;
                for (auto itr = frameDuration.begin(); itr != frameDuration.end(); itr++) {
                    if (itr->first != mostFrequentDuration) {
                        otherDuration = itr->first;
                        break;
                    }
                }
                m_ptsOffset = ptsOffsetMax + otherDuration;
            }
        } else {
            // ptsOffsetOrigが必要offsetの最小値(ptsOffsetMax)より大きく、そのずれが2フレーム以内ならそれを採用する
            // そうでなければ、ptsOffsetMaxに1フレーム分の時間を足した時刻にする
            m_ptsOffset = (m_firstPts < 0) ? 0 :
                ((ptsOffsetOrig - ptsOffsetMax > 0 && ptsOffsetOrig - ptsOffsetMax <= rational_rescale(2, m_encFps.inv(), m_outputTimebase))
                    ? ptsOffsetOrig : (ptsOffsetMax + rational_rescale(1, m_encFps.inv(), m_outputTimebase)));
        }
        m_encFrameOffset = (m_currentChunk > 0) ? m_maxEncFrameIdx + 1 : 0;
        m_inputFrameOffset = (m_currentChunk > 0) ? m_maxInputFrameIdx + 1 : 0;
        PrintMes(m_tsDebug ? RGY_LOG_ERROR : RGY_LOG_TRACE, _T("Switch to next file: pts offset %lld, frame offset %d.\n")
            _T("  firstKeyPts 0: % lld, %d : % lld.\n")
            _T("  ptsOffsetOrig: %lld, ptsOffsetMax: %lld, m_maxPts: %lld\n"),
            m_ptsOffset, m_encFrameOffset,
            m_firstPts, m_currentChunk, rational_rescale(m_parallelEnc->getVideofirstKeyPts(m_currentChunk), srcTimebase, m_outputTimebase),
            ptsOffsetOrig, ptsOffsetMax, m_maxPts);
        return RGY_ERR_NONE;
    }

    void updateAndSetHeaderProperties(RGYBitstream *bsOut, RGYOutputRawPEExtHeader *header) {
        header->pts += m_ptsOffset;
        header->dts += m_ptsOffset;
        header->encodeFrameIdx += m_encFrameOffset;
        header->inputFrameIdx += m_inputFrameOffset;
        bsOut->setPts(header->pts);
        bsOut->setDts(header->dts);
        bsOut->setDuration(header->duration);
        bsOut->setFrametype(header->frameType);
        bsOut->setPicstruct(header->picstruct);
        bsOut->setFrameIdx(header->encodeFrameIdx);
        bsOut->setDataflag((RGY_FRAME_FLAGS)header->flags);
    }

    RGY_ERR getBitstreamOneFrameFromQueue(RGYBitstream *bsOut, RGYOutputRawPEExtHeader& header) {
        RGYOutputRawPEExtHeader *packet = nullptr;
        auto err = m_parallelEnc->getNextPacket(m_currentChunk, &packet);
        if (err != RGY_ERR_NONE) {
            return err;
        }
        if (packet == nullptr) {
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        }
        updateAndSetHeaderProperties(bsOut, packet);
        if (packet->size <= 0) {
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        } else {
            bsOut->resize(packet->size);
            memcpy(&header, packet, sizeof(header));
            memcpy(bsOut->data(), (void *)(packet + 1), packet->size);
        }
        // メモリを使いまわすため、使い終わったパケットを回収する
        m_parallelEnc->putFreePacket(m_currentChunk, packet);
        PrintMes(RGY_LOG_TRACE, _T("Q: pts %08lld, dts %08lld, size %d.\n"), bsOut->pts(), bsOut->dts(), bsOut->size());
        return RGY_ERR_NONE;
    }

    RGY_ERR getBitstreamOneFrameFromFile(FILE *fp, RGYBitstream *bsOut, RGYOutputRawPEExtHeader& header) {
        if (fread(&header, 1, sizeof(header), fp) != sizeof(header)) {
            return RGY_ERR_MORE_BITSTREAM;
        }
        if (header.size <= 0) {
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        }
        updateAndSetHeaderProperties(bsOut, &header);
        bsOut->resize(header.size);
        PrintMes(RGY_LOG_TRACE, _T("F: pts %08lld, dts %08lld, size %d.\n"), bsOut->pts(), bsOut->dts(), bsOut->size());

        if (fread(bsOut->data(), 1, bsOut->size(), fp) != bsOut->size()) {
            return RGY_ERR_UNDEFINED_BEHAVIOR;
        }
        return RGY_ERR_NONE;
    }

    RGY_ERR getBitstreamOneFrame(RGYBitstream *bsOut, RGYOutputRawPEExtHeader& header) {
        return (m_parallelEnc->cacheMode(m_currentChunk) == RGYParamParallelEncCache::File)
            ? getBitstreamOneFrameFromFile(m_fReader.get(), bsOut, header)
            : getBitstreamOneFrameFromQueue(bsOut, header);
    }

    virtual RGY_ERR getBitstream(RGYBitstream *bsOut, RGYOutputRawPEExtHeader& header) {
        if (m_currentChunk < 0) {
            if (auto err = openNextFile(); err != RGY_ERR_NONE) {
                return err;
            }
        } else if (m_currentChunk >= (int)m_parallelEnc->parallelCount()) {
            return RGY_ERR_MORE_BITSTREAM;
        }
        auto err = getBitstreamOneFrame(bsOut, header);
        if (err == RGY_ERR_MORE_BITSTREAM) {
            if ((err = openNextFile()) != RGY_ERR_NONE) {
                return err;
            }
            err = getBitstreamOneFrame(bsOut, header);
        }
        return err;
    }
public:
    virtual RGY_ERR sendFrame([[maybe_unused]] std::unique_ptr<PipelineTaskOutput>& frame) override {
        m_inFrames++;
        auto ret = m_input->LoadNextFrame(nullptr); // 進捗表示用のダミー
        if (ret != RGY_ERR_NONE && ret != RGY_ERR_MORE_DATA && ret != RGY_ERR_MORE_BITSTREAM) {
            PrintMes(RGY_LOG_ERROR, _T("Error in reader: %s.\n"), get_err_mes(ret));
            return ret;
        }
        m_inputBitstreamEOF |= (ret == RGY_ERR_MORE_DATA || ret == RGY_ERR_MORE_BITSTREAM);

        // 音声等抽出のため、入力ファイルの読み込みを進める
        //この関数がMFX_ERR_NONE以外を返せば、入力ビットストリームは終了
        ret = m_input->GetNextBitstream(&m_decInputBitstream);
        m_inputBitstreamEOF |= (ret == RGY_ERR_MORE_DATA || ret == RGY_ERR_MORE_BITSTREAM);
        if (ret != RGY_ERR_NONE && ret != RGY_ERR_MORE_DATA && ret != RGY_ERR_MORE_BITSTREAM) {
            PrintMes(RGY_LOG_ERROR, _T("Error in reader: %s.\n"), get_err_mes(ret));
            return ret; //エラー
        }
        m_decInputBitstream.clear();

        if (m_taskAudio) {
            ret = m_taskAudio->extractAudio(m_inFrames);
            if (ret != RGY_ERR_NONE) {
                return ret;
            }
        }

        // 定期的に全スレッドでエラー終了したものがないかチェックする
        if ((m_inFrames & 15) == 0) {
            if ((ret = m_parallelEnc->checkAllProcessErrors()) != RGY_ERR_NONE) {
                return ret; //エラー
            }
        }

        auto bsOut = m_bitStreamOut.get([](RGYBitstream *bs) {
            auto sts = mfxBitstreamInit(bs->bsptr(), 1 * 1024 * 1024);
            if (sts != MFX_ERR_NONE) {
                return 1;
            }
            return 0;
        });
        RGYOutputRawPEExtHeader header;
        ret = getBitstream(bsOut.get(), header);
        if (ret != RGY_ERR_NONE && ret != RGY_ERR_MORE_BITSTREAM) {
            return ret;
        }
        if (ret == RGY_ERR_NONE && bsOut->size() > 0) {
            std::vector<std::shared_ptr<RGYFrameData>> metadatalist;
            const auto duration = (ENCODER_QSV) ? header.duration : bsOut->duration(); // QSVの場合、Bitstreamにdurationの値がないため、durationはheaderから取得する
            m_encTimestamp->add(bsOut->pts(), header.inputFrameIdx, header.encodeFrameIdx, duration, metadatalist);
            if (m_firstPts < 0) m_firstPts = bsOut->pts();
            m_maxPts = std::max(m_maxPts, bsOut->pts());
            m_maxEncFrameIdx = std::max(m_maxEncFrameIdx, header.encodeFrameIdx);
            m_maxInputFrameIdx = std::max(m_maxInputFrameIdx, header.inputFrameIdx);
            PrintMes(m_tsDebug ? RGY_LOG_WARN : RGY_LOG_TRACE, _T("Packet: pts %lld, dts: %lld, duration: %d, input idx: %lld, encode idx: %lld, size %lld.\n"), bsOut->pts(), bsOut->dts(), duration, header.inputFrameIdx, header.encodeFrameIdx, bsOut->size());
            if (m_timecode) {
                m_timecode->write(bsOut->pts(), m_outputTimebase);
            }
            m_durationCheck.add(bsOut->pts());
            m_outQeueue.push_back(std::make_unique<PipelineTaskOutputBitstream>(nullptr, bsOut, nullptr));
        }
        if (m_inputBitstreamEOF && ret == RGY_ERR_MORE_BITSTREAM && m_taskAudio) {
            m_taskAudio->flushAudio();
        }
        return (m_inputBitstreamEOF && ret == RGY_ERR_MORE_BITSTREAM) ? RGY_ERR_MORE_BITSTREAM : RGY_ERR_NONE;
    }
};

class PipelineTaskTrim : public PipelineTask {
protected:
    const sTrimParam &m_trimParam;
    RGYInput *m_input;
    RGYParallelEnc *m_parallelEnc;
    rgy_rational<int> m_srcTimebase;
public:
    PipelineTaskTrim(const sTrimParam &trimParam, RGYInput *input, RGYParallelEnc *parallelEnc, const rgy_rational<int>& srcTimebase, int outMaxQueueSize, mfxVersion mfxVer, std::shared_ptr<RGYLog> log) :
        PipelineTask(PipelineTaskType::TRIM, outMaxQueueSize, nullptr, mfxVer, log),
        m_trimParam(trimParam), m_input(input), m_parallelEnc(parallelEnc), m_srcTimebase(srcTimebase) {
    };
    virtual ~PipelineTaskTrim() {};

    virtual void setStopWatch() override {
        m_stopwatch = std::make_unique<PipelineTaskStopWatch>(
            std::vector<tstring>{ _T("") },
            std::vector<tstring>{_T("")}
        );
    }
    virtual bool isPassThrough() const override { return true; }
    virtual std::optional<mfxFrameAllocRequest> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<mfxFrameAllocRequest> requiredSurfOut() override { return std::nullopt; };

    virtual RGY_ERR sendFrame([[maybe_unused]] std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (m_stopwatch) m_stopwatch->set(0);
        if (!frame) {
            return RGY_ERR_MORE_DATA;
        }
        m_inFrames++;
        PipelineTaskOutputSurf *taskSurf = dynamic_cast<PipelineTaskOutputSurf *>(frame.get());
        if (!frame_inside_range(taskSurf->surf().frame()->inputFrameId(), m_trimParam.list).first) {
            return RGY_ERR_NONE;
        }
        const auto surfPts = (int64_t)taskSurf->surf().frame()->timestamp();
        if (m_parallelEnc) {
            auto finKeyPts = m_parallelEnc->getVideoEndKeyPts();
            if (finKeyPts >= 0 && surfPts >= finKeyPts) {
                m_parallelEnc->setVideoFinished();
                return RGY_ERR_NONE;
            }
        }
        if (!m_input->checkTimeSeekTo(surfPts, m_srcTimebase)) {
            return RGY_ERR_NONE; //seektoにより脱落させるフレーム
        }
        m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_mfxSession, taskSurf->surf(), taskSurf->syncpoint()));
        if (m_stopwatch) m_stopwatch->add(0, 0);
        return RGY_ERR_NONE;
    }
};

class PipelineTaskMFXVpp : public PipelineTask {
protected:
    QSVVppMfx *m_vpp;
    rgy_rational<int> m_outputTimebase;
    RGYTimestamp m_timestamp;
    mfxVideoParam& m_mfxVppParams;
    std::vector<std::shared_ptr<RGYFrameData>> m_lastFrameDataList;
public:
    PipelineTaskMFXVpp(MFXVideoSession *mfxSession, int outMaxQueueSize, QSVVppMfx *mfxvpp, mfxVideoParam& vppParams, mfxVersion mfxVer, rgy_rational<int> outputTimebase, bool timestampPassThrough, std::shared_ptr<RGYLog> log)
        : PipelineTask(PipelineTaskType::MFXVPP, outMaxQueueSize, mfxSession, mfxVer, log), m_vpp(mfxvpp), m_outputTimebase(outputTimebase), m_timestamp(RGYTimestamp(timestampPassThrough, false)), m_mfxVppParams(vppParams), m_lastFrameDataList() {
    };
    virtual ~PipelineTaskMFXVpp() {};
    void setVpp(QSVVppMfx *mfxvpp) { m_vpp = mfxvpp; };
    virtual void setStopWatch() override {
        m_stopwatch = std::make_unique<PipelineTaskStopWatch>(
            std::vector<tstring>{ _T("Reset"), _T("getWorkSurf"), _T("RunFrameVppAsync"), _T("DeviceBusy"), _T("PushQueue") },
            std::vector<tstring>{_T("")}
        );
    }
protected:
    RGY_ERR requiredSurfInOut(std::array<mfxFrameAllocRequest,2>& allocRequest) {
        for (auto& request : allocRequest) {
            memset(&request, 0, sizeof(request));
        }
        // allocRequest[0]はvppへの入力, allocRequest[1]はvppからの出力
        auto err = err_to_rgy(m_vpp->mfxvpp()->QueryIOSurf(&m_mfxVppParams, allocRequest.data()));
        if (err < RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("  Failed to get required buffer size for %s: %s\n"), getPipelineTaskTypeName(m_type), get_err_mes(err));
            return err;
        } else if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_WARN, _T("  surface alloc request for %s: %s\n"), getPipelineTaskTypeName(m_type), get_err_mes(err));
        }
        PrintMes(RGY_LOG_DEBUG, _T("  %s required buffer in: %d [%s], out %d [%s]\n"), getPipelineTaskTypeName(m_type),
            allocRequest[0].NumFrameSuggested, qsv_memtype_str(allocRequest[0].Type).c_str(),
            allocRequest[1].NumFrameSuggested, qsv_memtype_str(allocRequest[1].Type).c_str());
        return err;
    }
public:
    virtual std::optional<mfxFrameAllocRequest> requiredSurfIn() override {
        std::array<mfxFrameAllocRequest,2> allocRequest;
        if (requiredSurfInOut(allocRequest) < RGY_ERR_NONE) { //RGY_WRN_xxx ( > 0) は無視する
            return std::nullopt;
        }
        return std::optional<mfxFrameAllocRequest>(allocRequest[0]);
    };
    virtual std::optional<mfxFrameAllocRequest> requiredSurfOut() override {
        std::array<mfxFrameAllocRequest,2> allocRequest;
        if (requiredSurfInOut(allocRequest) < RGY_ERR_NONE) { //RGY_WRN_xxx ( > 0) は無視する
            return std::nullopt;
        }
        return std::optional<mfxFrameAllocRequest>(allocRequest[1]);
    };
    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (m_stopwatch) m_stopwatch->set(0);
        if (frame && frame->type() != PipelineTaskOutputType::SURFACE) {
            PrintMes(RGY_LOG_ERROR, _T("Invalid frame type.\n"));
            return RGY_ERR_UNSUPPORTED;
        }

        mfxStatus vpp_sts = MFX_ERR_NONE;

        if (frame) {
            m_lastFrameDataList = dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->surf().frame()->dataList();
        }

        const auto estDuration = av_rescale_q(1, av_make_q(m_outputTimebase.inv()), av_make_q(m_mfxVppParams.vpp.In.FrameRateExtN, m_mfxVppParams.vpp.In.FrameRateExtD));

        mfxFrameSurface1 *surfVppIn = (frame) ? dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->surf().mfx()->surf() : nullptr;
        //vpp前に、vpp用のパラメータでFrameInfoを更新
        copy_crop_info(surfVppIn, &m_mfxVppParams.mfx.FrameInfo);
        if (surfVppIn) {
            // durationは適用でもfpsから設定しておく --vpp-deinterlace bobでも入力がRFFやプログレッシブだと2フレーム目を投入する前にフレーム出力が出る場合があり、durationを設定しておかないとおかしくなる
            m_timestamp.add(surfVppIn->Data.TimeStamp, dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->surf().frame()->inputFrameId(), 0 /*dummy*/, estDuration, {});
            surfVppIn->Data.DataFlag |= MFX_FRAMEDATA_ORIGINAL_TIMESTAMP;
            m_inFrames++;

            // インタレ解除を使用中、入力フレームのインタレが変更になると、そのまま処理を継続すると device busyで処理がフリーズしてしまうことがある
            // そのため、インタレ解除設定が変更になった場合は、フィルタをリセットする
            if (m_vpp->isDeinterlace()
                && surfVppIn->Info.PicStruct != m_mfxVppParams.vpp.In.PicStruct) {
                const auto picStructPrev = m_mfxVppParams.vpp.In.PicStruct;
                PrintMes(RGY_LOG_DEBUG, _T("Change deinterlace settings input: %s -> %s.\n"), MFXPicStructToStr(picStructPrev).c_str(), MFXPicStructToStr(surfVppIn->Info.PicStruct).c_str());

                // まずflushする
                auto sts = RGY_ERR_NONE;
                while (sts == RGY_ERR_NONE) {
                    auto flushFrame = std::unique_ptr<PipelineTaskOutput>();
                    sts = sendFrame(flushFrame); // flush
                    if (sts == RGY_ERR_MORE_DATA) {
                        break; // flush 成功
                    } else if (sts != RGY_ERR_NONE) {
                        PrintMes(RGY_LOG_ERROR, _T("  Failed to flush filter to change interlace settings %s -> %s: %s.\n"), MFXPicStructToStr(picStructPrev).c_str(), MFXPicStructToStr(surfVppIn->Info.PicStruct).c_str(), get_err_mes(sts));
                        return sts;
                    }
                }

                // インタレ設定変更を反映してリセットする
                m_mfxVppParams.vpp.In.PicStruct = surfVppIn->Info.PicStruct;
                sts = m_vpp->Reset(m_mfxVppParams.vpp.Out, m_mfxVppParams.vpp.In);
                if (sts != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("  Failed to reset filter to change interlace settings %s -> %s: %s.\n"), MFXPicStructToStr(picStructPrev).c_str(), MFXPicStructToStr(surfVppIn->Info.PicStruct).c_str(), get_err_mes(sts));
                    return sts;
                }
            }
        }
        if (m_stopwatch) m_stopwatch->add(0, 0);


        bool vppMoreOutput = false;
        do {
            if (m_stopwatch) m_stopwatch->set(0);
            vppMoreOutput = false;
            auto surfVppOut = getWorkSurf();
            mfxSyncPoint lastSyncPoint = nullptr;
            if (m_stopwatch) m_stopwatch->add(0, 1);
            for (int i = 0; ; i++) {
                if (m_stopwatch) m_stopwatch->set(0);
                //bob化の際、pSurfVppInに連続で同じフレーム(同じtimestamp)を投入すると、
                //最初のフレームには設定したtimestamp、次のフレームにはMFX_TIMESTAMP_UNKNOWNが設定されて出てくる
                //特別pSurfVppOut側のTimestampを設定する必要はなさそう
                mfxSyncPoint VppSyncPoint = nullptr;
                vpp_sts = m_vpp->mfxvpp()->RunFrameVPPAsync(surfVppIn, surfVppOut.mfx()->surf(), nullptr, &VppSyncPoint);
                lastSyncPoint = VppSyncPoint;
                if (m_stopwatch) m_stopwatch->add(0, 2);

                if (MFX_ERR_NONE < vpp_sts && !VppSyncPoint) {
                    if (MFX_WRN_DEVICE_BUSY == vpp_sts)
                        sleep_hybrid(i);
                    if (i > 1024 * 1024 * 30) {
                        PrintMes(RGY_LOG_ERROR, _T("device kept on busy for 30s, unknown error occurred.\n"));
                        return RGY_ERR_GPU_HANG;
                    }
                } else if (MFX_ERR_NONE < vpp_sts && VppSyncPoint) {
                    vpp_sts = MFX_ERR_NONE;
                    break;
                } else {
                    break;
                }
                if (m_stopwatch) m_stopwatch->add(0, 3);
            }
            if (m_stopwatch) m_stopwatch->add(0, 3);

            if (surfVppIn && vpp_sts == MFX_ERR_MORE_DATA) {
                vpp_sts = MFX_ERR_NONE;
            } else if (vpp_sts == MFX_ERR_MORE_SURFACE) {
                vppMoreOutput = true;
                vpp_sts = MFX_ERR_NONE;
            } else if (vpp_sts != MFX_ERR_NONE) {
                return err_to_rgy(vpp_sts);
            }

            if (lastSyncPoint != nullptr) {
                //bob化の際に増えたフレームのTimeStampには、MFX_TIMESTAMP_UNKNOWNが設定されているのでこれを補間して修正する
                auto tsMap = m_timestamp.check(surfVppOut.frame()->timestamp());
                surfVppOut.frame()->setTimestamp(tsMap.timestamp);
                surfVppOut.frame()->setInputFrameId((int)tsMap.inputFrameId);
                surfVppOut.frame()->setDataList(m_lastFrameDataList);
                m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_mfxSession, surfVppOut, lastSyncPoint));
            }
            if (m_stopwatch) m_stopwatch->add(0, 4);
        } while (vppMoreOutput);
        return err_to_rgy(vpp_sts);
    }
};

class PipelineTaskVideoQualityMetric : public PipelineTask {
private:
    std::shared_ptr<RGYOpenCLContext> m_cl;
    RGYFilterSsim *m_videoMetric;
    std::unordered_map<mfxFrameSurface1 *, std::unique_ptr<RGYCLFrameInterop>> m_surfVppInInterop;
    MemType m_memType;
public:
    PipelineTaskVideoQualityMetric(RGYFilterSsim *videoMetric, std::shared_ptr<RGYOpenCLContext> cl, MemType memType, QSVAllocator *allocator, MFXVideoSession *mfxSession, int outMaxQueueSize, mfxVersion mfxVer, std::shared_ptr<RGYLog> log)
        : PipelineTask(PipelineTaskType::VIDEOMETRIC, outMaxQueueSize, mfxSession, mfxVer, log), m_cl(cl), m_videoMetric(videoMetric), m_surfVppInInterop(), m_memType(memType) {
        m_allocator = allocator;
    };

    virtual bool isPassThrough() const override { return true; }
    virtual std::optional<mfxFrameAllocRequest> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<mfxFrameAllocRequest> requiredSurfOut() override { return std::nullopt; };
    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (!frame) {
            return RGY_ERR_MORE_DATA;
        }
        //明示的に待機が必要
        frame->depend_clear();

        RGYCLFrameInterop *clFrameInInterop = nullptr;
        PipelineTaskOutputSurf *taskSurf = dynamic_cast<PipelineTaskOutputSurf *>(frame.get());
        if (taskSurf == nullptr) {
            PrintMes(RGY_LOG_ERROR, _T("Invalid task surface.\n"));
            return RGY_ERR_NULL_PTR;
        }
        RGYFrameInfo inputFrame;
        mfxFrameSurface1 *surfVppIn = taskSurf->surf().mfx()->surf();
        if (surfVppIn != nullptr) {
            if (m_surfVppInInterop.count(surfVppIn) == 0) {
                m_surfVppInInterop[surfVppIn] = getOpenCLFrameInterop(surfVppIn, m_memType, CL_MEM_READ_ONLY, m_allocator, m_cl.get(), m_cl->queue(), m_videoMetric->GetFilterParam()->frameIn);
            }
            clFrameInInterop = m_surfVppInInterop[surfVppIn].get();
            if (!clFrameInInterop) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to get OpenCL interop [in].\n"));
                return RGY_ERR_NULL_PTR;
            }
            auto err = clFrameInInterop->acquire(m_cl->queue());
            if (err != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("Failed to acquire OpenCL interop [in]: %s.\n"), get_err_mes(err));
                return RGY_ERR_NULL_PTR;
            }
            clFrameInInterop->frame.flags = taskSurf->surf().frame()->flags();
            clFrameInInterop->frame.timestamp = taskSurf->surf().frame()->timestamp();
            clFrameInInterop->frame.inputFrameId = taskSurf->surf().frame()->inputFrameId();
            clFrameInInterop->frame.picstruct = taskSurf->surf().frame()->picstruct();
            inputFrame = clFrameInInterop->frameInfo();
        } else if (auto clframe = taskSurf->surf().cl(); clframe != nullptr) {
            //OpenCLフレームが出てきた時の場合
            inputFrame = clframe->frameInfo();
        } else {
            PrintMes(RGY_LOG_ERROR, _T("Invalid input frame.\n"));
            return RGY_ERR_NULL_PTR;
        }
        //フレームを転送
        RGYOpenCLEvent inputReleaseEvent;
        int dummy = 0;
        auto err = m_videoMetric->filter(&inputFrame, nullptr, &dummy, m_cl->queue(), &inputReleaseEvent);
        if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to send frame for video metric calcualtion: %s.\n"), get_err_mes(err));
            return err;
        }
        if (clFrameInInterop) {
            clFrameInInterop->release(&inputReleaseEvent); // input frameの解放
            clFrameInInterop = nullptr;
        }
        //eventを入力フレームを使用し終わったことの合図として登録する
        taskSurf->addClEvent(inputReleaseEvent);
        m_outQeueue.push_back(std::move(frame));
        return RGY_ERR_NONE;
    }
};

class encCtrlData {
protected:
    mfxEncodeCtrl encCtrl;
    std::vector<uint8_t> dhdr10plus_sei;
    mfxPayload payLoad;
    mfxPayload *payLoads[1];
public:
    encCtrlData() : encCtrl({ 0 }), dhdr10plus_sei(), payLoad({ 0 }), payLoads() {
        payLoads[0] = &payLoad;
        encCtrl.NumPayload = 1;
        encCtrl.Payload = payLoads;
    }

    mfxEncodeCtrl *getCtrlPtr() {
        return &encCtrl;
    }

    bool hasData() const {
        return dhdr10plus_sei.size() > 0;
    }

    void setHDR10PlusPayload(const std::vector<uint8_t>& data) {
        dhdr10plus_sei.clear();
        dhdr10plus_sei.reserve(data.size() + 2);
        dhdr10plus_sei.push_back(USER_DATA_REGISTERED_ITU_T_T35);
        auto datasize = data.size();
        for (; datasize > 0xff; datasize -= 0xff)
            dhdr10plus_sei.push_back((uint8_t)0xff);
        dhdr10plus_sei.push_back((uint8_t)datasize);
        vector_cat(dhdr10plus_sei, data);

        payLoad.Type = USER_DATA_REGISTERED_ITU_T_T35;
        payLoad.Data = dhdr10plus_sei.data();
        payLoad.BufSize = (mfxU16)dhdr10plus_sei.size();
        payLoad.NumBit = (mfxU32)dhdr10plus_sei.size() * 8;
    }
};

class PipelineTaskMFXEncode : public PipelineTask {
protected:
    MFXVideoENCODE *m_encode;
    RGYTimecode *m_timecode;
    RGYTimestamp *m_encTimestamp;
    QSVVideoParam& m_encParams;
    rgy_rational<int> m_outputTimebase;
    RGYListRef<RGYBitstream> m_bitStreamOut;
    QSVRCParam m_baseRC;
    std::vector<QSVRCParam>& m_dynamicRC;
    int m_appliedDynamicRC;
    const RGYHDR10Plus *m_hdr10plus;
    const DOVIRpu *m_doviRpu;
    encCtrlData m_encCtrlData;
public:
    PipelineTaskMFXEncode(
        MFXVideoSession *mfxSession, int outMaxQueueSize, MFXVideoENCODE *mfxencode, mfxVersion mfxVer, QSVVideoParam& encParams,
        RGYTimecode *timecode, RGYTimestamp *encTimestamp, rgy_rational<int> outputTimebase, std::vector<QSVRCParam>& dynamicRC,
        const RGYHDR10Plus *hdr10plus, const DOVIRpu *doviRpu, std::shared_ptr<RGYLog> log)
        : PipelineTask(PipelineTaskType::MFXENCODE, outMaxQueueSize, mfxSession, mfxVer, log),
        m_encode(mfxencode), m_timecode(timecode), m_encTimestamp(encTimestamp), m_encParams(encParams), m_outputTimebase(outputTimebase), m_bitStreamOut(),
        m_baseRC(getRCParam(encParams)), m_dynamicRC(dynamicRC), m_appliedDynamicRC(-1),
        m_hdr10plus(hdr10plus), m_doviRpu(doviRpu),
        m_encCtrlData() {
    };
    virtual ~PipelineTaskMFXEncode() {
        m_outQeueue.clear(); // m_bitStreamOutが解放されるよう前にこちらを解放する
    };
    void setEnc(MFXVideoENCODE *mfxencode) { m_encode = mfxencode; };
    virtual void setStopWatch() override {
        m_stopwatch = std::make_unique<PipelineTaskStopWatch>(
            std::vector<tstring>{ _T("GetBitstream"), _T("Reset"), _T("EncodeFrameAsync"), _T("EncoderBusy") },
            std::vector<tstring>{_T("")}
        );
    }

    QSVRCParam getRCParam(const QSVVideoParam& encParams) {
        mfxExtCodingOption3 *cop3 = nullptr;
        for (size_t i = 0; i < encParams.videoPrm.NumExtParam; i++) {
            if (encParams.videoPrm.ExtParam[i]->BufferId == MFX_EXTBUFF_CODING_OPTION3) {
                cop3 = (mfxExtCodingOption3 *)encParams.videoPrm.ExtParam[i];
                break;
            }
        }
        return QSVRCParam(
            encParams.videoPrm.mfx.RateControlMethod, encParams.videoPrm.mfx.TargetKbps, encParams.videoPrm.mfx.MaxKbps, m_encParams.videoPrm.mfx.BufferSizeInKB*8,
            encParams.videoPrm.mfx.Accuracy, encParams.videoPrm.mfx.Convergence,
            { encParams.videoPrm.mfx.QPI, encParams.videoPrm.mfx.QPP, encParams.videoPrm.mfx.QPB },
            encParams.videoPrm.mfx.ICQQuality, encParams.cop3.QVBRQuality);
    }

    void setRCParam(mfxVideoParam& encParams, const QSVRCParam& rcParams) {
        // rcParams の値を encParams に反映する
        encParams.mfx.RateControlMethod = (decltype(encParams.mfx.RateControlMethod))rcParams.encMode;
        if (encParams.mfx.RateControlMethod == MFX_RATECONTROL_CQP) {
            encParams.mfx.QPI = (decltype(encParams.mfx.QPI))rcParams.qp.qpI;
            encParams.mfx.QPP = (decltype(encParams.mfx.QPP))rcParams.qp.qpP;
            encParams.mfx.QPB = (decltype(encParams.mfx.QPB))rcParams.qp.qpB;
        } else if (encParams.mfx.RateControlMethod == MFX_RATECONTROL_ICQ
                || encParams.mfx.RateControlMethod == MFX_RATECONTROL_LA_ICQ) {
            encParams.mfx.ICQQuality = (decltype(encParams.mfx.ICQQuality))rcParams.icqQuality;
            encParams.mfx.MaxKbps = 0;
        } else {
            const int maxBitrate = (encParams.mfx.RateControlMethod == MFX_RATECONTROL_CQP) ? rcParams.bitrate : ((rcParams.maxBitrate) ? rcParams.maxBitrate : QSV_DEFAULT_MAX_BITRATE);
            const auto maxRCRate = (std::max)((std::max)(rcParams.bitrate, maxBitrate),
                rcParams.vbvBufSize / 8 /*これはbyte単位の指定*/);
            m_encParams.videoPrm.mfx.BRCParamMultiplier = (maxRCRate > USHRT_MAX) ? (mfxU16)(maxRCRate / USHRT_MAX) + 1 : 1;
            encParams.mfx.TargetKbps = (decltype(encParams.mfx.TargetKbps))rcParams.bitrate / m_encParams.videoPrm.mfx.BRCParamMultiplier;
            if (encParams.mfx.RateControlMethod == MFX_RATECONTROL_AVBR) {
                if (rcParams.avbrAccuarcy > 0) {
                    encParams.mfx.Accuracy = (decltype(encParams.mfx.Accuracy))rcParams.avbrAccuarcy;
                }
                if (rcParams.avbrConvergence > 0) {
                    encParams.mfx.Convergence = (decltype(encParams.mfx.Convergence))rcParams.avbrConvergence;
                }
            } else {
                encParams.mfx.MaxKbps = (decltype(encParams.mfx.MaxKbps))maxBitrate / m_encParams.videoPrm.mfx.BRCParamMultiplier;
                if (rcParams.vbvBufSize > 0) {
                    encParams.mfx.BufferSizeInKB = (decltype(encParams.mfx.BufferSizeInKB))((rcParams.vbvBufSize / 8 /*これはbyte単位の指定*/) / m_encParams.videoPrm.mfx.BRCParamMultiplier);
                    encParams.mfx.InitialDelayInKB = encParams.mfx.BufferSizeInKB / 2;
                }
            }
            if (encParams.mfx.RateControlMethod == MFX_RATECONTROL_QVBR && rcParams.qvbrQuality > 0) {
                mfxExtCodingOption3 *cop3 = nullptr;
                for (size_t i = 0; i < encParams.NumExtParam; i++) {
                    if (encParams.ExtParam[i]->BufferId == MFX_EXTBUFF_CODING_OPTION3) {
                        cop3 = (mfxExtCodingOption3 *)encParams.ExtParam[i];
                        break;
                    }
                }
                if (cop3) {
                    cop3->QVBRQuality = (decltype(cop3->QVBRQuality))rcParams.qvbrQuality;
                }
            }
        }
    }

    virtual std::optional<mfxFrameAllocRequest> requiredSurfIn() override {
        mfxFrameAllocRequest allocRequest = { 0 };
        auto err = err_to_rgy(m_encode->QueryIOSurf(&m_encParams.videoPrm, &allocRequest));
        if (err < RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("  Failed to get required buffer size for %s: %s\n"), getPipelineTaskTypeName(m_type), get_err_mes(err));
            return std::nullopt;
        } else if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_WARN, _T("  surface alloc request for %s: %s\n"), getPipelineTaskTypeName(m_type), get_err_mes(err));
        }
        PrintMes(RGY_LOG_DEBUG, _T("  %s required buffer: %d [%s]\n"), getPipelineTaskTypeName(m_type), allocRequest.NumFrameSuggested, qsv_memtype_str(allocRequest.Type).c_str());
        const int blocksz = (m_encParams.videoPrm.mfx.CodecId == MFX_CODEC_HEVC) ? 32 : 16;
        allocRequest.Info.Width  = (mfxU16)ALIGN(allocRequest.Info.Width,  blocksz);
        allocRequest.Info.Height = (mfxU16)ALIGN(allocRequest.Info.Height, blocksz);
        return std::optional<mfxFrameAllocRequest>(allocRequest);
    };

    int getDynamicRCIndex(const int inputFrameId) {
        for (int i = 0; i < (int)m_dynamicRC.size(); i++) {
            if (m_dynamicRC[i].start <= inputFrameId && inputFrameId <= m_dynamicRC[i].end) {
                return i;
            }
        }
        return -1;
    }

    virtual std::optional<mfxFrameAllocRequest> requiredSurfOut() override { return std::nullopt; };
    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (m_stopwatch) m_stopwatch->set(0);
        if (frame && frame->type() != PipelineTaskOutputType::SURFACE) {
            PrintMes(RGY_LOG_ERROR, _T("Invalid frame type.\n"));
            return RGY_ERR_UNSUPPORTED;
        }

        auto bsOut = m_bitStreamOut.get([enc = m_encode, log = m_log](RGYBitstream *bs) {
            mfxVideoParam par = { 0 };
            mfxStatus sts = enc->GetVideoParam(&par);
            if (sts != MFX_ERR_NONE) {
                log->write(RGY_LOG_ERROR, RGY_LOGT_CORE, _T("Failed to get required output buffer size from encoder: %s\n"), get_err_mes(sts));
                mfxBitstreamClear(bs->bsptr());
                return 1;
            }

            sts = mfxBitstreamInit(bs->bsptr(), par.mfx.BufferSizeInKB * 1000 * (std::max)(1, (int)par.mfx.BRCParamMultiplier));
            if (sts != MFX_ERR_NONE) {
                log->write(RGY_LOG_ERROR, RGY_LOGT_CORE, _T("Failed to allocate memory for output bufffer: %s\n"), get_err_mes(sts));
                mfxBitstreamClear(bs->bsptr());
                return 1;
            }
            return 0;
        });
        if (!bsOut) {
            return RGY_ERR_NULL_PTR;
        }
        if (m_stopwatch) m_stopwatch->add(0, 0);

        std::vector<std::shared_ptr<RGYFrameData>> metadatalist;
        if (m_encParams.videoPrm.mfx.CodecId == MFX_CODEC_HEVC || m_encParams.videoPrm.mfx.CodecId == MFX_CODEC_AV1) {
            if (frame) {
                metadatalist = dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->surf().frame()->dataList();
            }
            if (m_hdr10plus) {
                // 外部からHDR10+を読み込む場合、metadatalist 内のHDR10+の削除
                for (auto it = metadatalist.begin(); it != metadatalist.end(); ) {
                    if ((*it)->dataType() == RGY_FRAME_DATA_HDR10PLUS) {
                        it = metadatalist.erase(it);
                    } else {
                        it++;
                    }
                }
            }
            if (m_doviRpu) {
                // 外部からdoviを読み込む場合、metadatalist 内のdovi rpuの削除
                for (auto it = metadatalist.begin(); it != metadatalist.end(); ) {
                    if ((*it)->dataType() == RGY_FRAME_DATA_DOVIRPU) {
                        it = metadatalist.erase(it);
                    } else {
                        it++;
                    }
                }
            }
        }

        //以下の処理は
        mfxFrameSurface1 *surfEncodeIn = (frame) ? dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->surf().mfx()->surf() : nullptr;
        if (surfEncodeIn) {
            //TimeStampをMFX_TIMESTAMP_UNKNOWNにしておくと、きちんと設定される
            bsOut->setPts((uint64_t)MFX_TIMESTAMP_UNKNOWN);
            bsOut->setDts((uint64_t)MFX_TIMESTAMP_UNKNOWN);
            if (m_timecode) {
                m_timecode->write(surfEncodeIn->Data.TimeStamp, m_outputTimebase);
            }
            // ここまではm_outputTimebase
            //最後にQSVのHW_TIMEBASEに変換する
            surfEncodeIn->Data.TimeStamp = rational_rescale(surfEncodeIn->Data.TimeStamp, m_outputTimebase, rgy_rational<int>(1, HW_TIMEBASE));
            surfEncodeIn->Data.DataFlag |= MFX_FRAMEDATA_ORIGINAL_TIMESTAMP;

            const auto inputFrameId = dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->surf().frame()->inputFrameId();
            if (inputFrameId < 0) {
                PrintMes(RGY_LOG_ERROR, _T("Invalid inputFrameId: %d.\n"), inputFrameId);
                return RGY_ERR_UNKNOWN;
            }

            const auto targetDynamicRC = getDynamicRCIndex(inputFrameId);
            if (targetDynamicRC != m_appliedDynamicRC) {
                // 指定にしたがってエンコーダのパラメータを変更する
                setRCParam(m_encParams.videoPrm, (targetDynamicRC >= 0) ? m_dynamicRC[targetDynamicRC] : m_baseRC);
                m_appliedDynamicRC = targetDynamicRC;

                auto sts = RGY_ERR_NONE;
                while (sts == RGY_ERR_NONE) {
                    auto flushFrame = std::unique_ptr<PipelineTaskOutput>();
                    sts = sendFrame(flushFrame); // flush
                    if (sts == RGY_ERR_MORE_DATA) {
                        break; // flush 成功
                    } else if (sts != RGY_ERR_NONE) {
                        PrintMes(RGY_LOG_ERROR, _T("Failed to flush encoder for dynamic rc(%d): %s.\n"), targetDynamicRC, get_err_mes(sts));
                        return sts;
                    }
                }

                // m_encode->Reset()では、エラーが返る場合があるので、m_encode->Close() -> m_encode->Init()の順で行う
                sts = err_to_rgy(m_encode->Close());
                if (sts != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to close encoder for dynamic rc(%d): %s.\n"), targetDynamicRC, get_err_mes(sts));
                    return sts;
                }

                sts = err_to_rgy(m_encode->Init(&m_encParams.videoPrm));
                if (sts != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to init encoder for dynamic rc(%d): %s.\n"), targetDynamicRC, get_err_mes(sts));
                    PrintMes(RGY_LOG_ERROR, _T("  parameter was %s.\n"), m_dynamicRC[targetDynamicRC].print().c_str());
                    return sts;
                }
            }
            m_encTimestamp->add(surfEncodeIn->Data.TimeStamp, inputFrameId, m_inFrames, 0, metadatalist);
            m_inFrames++;
            PrintMes(RGY_LOG_TRACE, _T("send encoder %6d/%6d/%10lld.\n"), m_inFrames, inputFrameId, surfEncodeIn->Data.TimeStamp);
        }
        //エンコーダまでたどり着いたフレームについてはdataListを解放
        if (frame) {
            dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->surf().frame()->clearDataList();
        }
        if (m_stopwatch) m_stopwatch->add(0, 1);

        std::unique_ptr<std::chrono::system_clock::time_point> device_busy;
        auto enc_sts = MFX_ERR_NONE;
        mfxSyncPoint lastSyncP = nullptr;
        bool bDeviceBusy = false;
        for (int i = 0; ; i++) {
            if (m_stopwatch) m_stopwatch->set(0);
            auto ctrlPtr = (m_encCtrlData.hasData()) ? m_encCtrlData.getCtrlPtr() : nullptr;
            enc_sts = m_encode->EncodeFrameAsync(ctrlPtr, surfEncodeIn, bsOut->bsptr(), &lastSyncP);
            bDeviceBusy = false;
            if (m_stopwatch) m_stopwatch->add(0, 2);

            if (MFX_ERR_NONE < enc_sts && lastSyncP == nullptr) {
                bDeviceBusy = true;
                if (enc_sts == MFX_WRN_DEVICE_BUSY) {
                    sleep_hybrid(i);
                    if (!device_busy) {
                        device_busy = std::make_unique<std::chrono::system_clock::time_point>(std::chrono::system_clock::now());
                    } else if ((i & 1023) == 0) {
                        // 15秒以上エンコーダがビジー状態が続いている場合、エンコーダが復帰しないと判断する
                        const auto now = std::chrono::system_clock::now();
                        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - *device_busy).count();
                        if (elapsed > 15) {
                            PrintMes(RGY_LOG_ERROR, _T("device kept on busy for 30s, unknown error occurred.\n"));
                            return RGY_ERR_GPU_HANG;
                        }
                    }
                }
            } else if (MFX_ERR_NONE < enc_sts && lastSyncP != nullptr) {
                enc_sts = MFX_ERR_NONE;
                break;
            } else if (enc_sts == MFX_ERR_NOT_ENOUGH_BUFFER) {
                enc_sts = mfxBitstreamExtend(bsOut->bsptr(), (uint32_t)bsOut->bufsize() * 3 / 2);
                if (enc_sts < MFX_ERR_NONE) return err_to_rgy(enc_sts);
            } else if (enc_sts < MFX_ERR_NONE && (enc_sts != MFX_ERR_MORE_DATA && enc_sts != MFX_ERR_MORE_SURFACE)) {
                PrintMes(RGY_LOG_ERROR, _T("EncodeFrameAsync error: %s.\n"), get_err_mes(enc_sts));
                break;
            } else {
                QSV_IGNORE_STS(enc_sts, MFX_ERR_MORE_BITSTREAM);
                break;
            }
            if (m_stopwatch) m_stopwatch->add(0, 3);
        }
        if (lastSyncP != nullptr) {
            m_outQeueue.push_back(std::make_unique<PipelineTaskOutputBitstream>(m_mfxSession, bsOut, lastSyncP));
        }
        return err_to_rgy(enc_sts);
    }

};

class PipelineTaskOpenCL : public PipelineTask {
protected:
    std::shared_ptr<RGYOpenCLContext> m_cl;
    std::vector<std::unique_ptr<RGYFilter>>& m_vpFilters;
    std::unordered_map<mfxFrameSurface1 *, std::unique_ptr<RGYCLFrameInterop>> m_surfVppInInterop;
    std::unordered_map<mfxFrameSurface1 *, std::unique_ptr<RGYCLFrameInterop>> m_surfVppOutInterop;
    std::deque<std::unique_ptr<PipelineTaskOutput>> m_prevInputFrame; //前回投入されたフレーム、完了通知を待ってから解放するため、参照を保持する
    RGYFilterSsim *m_videoMetric;
    MemType m_memType;
public:
    PipelineTaskOpenCL(std::vector<std::unique_ptr<RGYFilter>>& vppfilters, RGYFilterSsim *videoMetric, std::shared_ptr<RGYOpenCLContext> cl, MemType memType, QSVAllocator *allocator, MFXVideoSession *mfxSession, int outMaxQueueSize, std::shared_ptr<RGYLog> log) :
        PipelineTask(PipelineTaskType::OPENCL, outMaxQueueSize, mfxSession, MFX_LIB_VERSION_0_0, log), m_cl(cl), m_vpFilters(vppfilters), m_surfVppInInterop(), m_surfVppOutInterop(), m_prevInputFrame(), m_videoMetric(videoMetric), m_memType(memType) {
        m_allocator = allocator;
    };
    virtual ~PipelineTaskOpenCL() {
        m_prevInputFrame.clear();
        m_surfVppInInterop.clear();
        m_surfVppOutInterop.clear();
        m_cl.reset();
    };

    virtual void setStopWatch() override {
        m_stopwatch = std::make_unique<PipelineTaskStopWatch>(
            std::vector<tstring>{ _T("Sync"), _T("Interop"), _T("Filtering") },
            std::vector<tstring>{_T("")}
        );
    }
    void setVideoQualityMetricFilter(RGYFilterSsim *videoMetric) {
        m_videoMetric = videoMetric;
    }

    virtual RGY_ERR getOutputFrameInfo(mfxFrameInfo& info) override {
        if (m_vpFilters.size() == 0) {
            return RGY_ERR_UNKNOWN;
        }
        auto lastFilterOut = m_vpFilters.back()->GetFilterParam()->frameOut;
        auto fps = m_vpFilters.back()->GetFilterParam()->baseFps;
        info = frameinfo_rgy_to_enc(lastFilterOut, fps, rgy_rational<int>(1,1), 2);
        return RGY_ERR_NONE;
    }

    virtual std::optional<mfxFrameAllocRequest> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<mfxFrameAllocRequest> requiredSurfOut() override { return std::nullopt; };
    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (m_stopwatch) m_stopwatch->set(0);
        if (m_prevInputFrame.size() > 0) {
            //前回投入したフレームの処理が完了していることを確認したうえで参照を破棄することでロックを解放する
            auto prevframe = std::move(m_prevInputFrame.front());
            m_prevInputFrame.pop_front();
            prevframe->depend_clear();
        }
        if (m_stopwatch) m_stopwatch->add(0, 0);

        deque<std::pair<RGYFrameInfo, uint32_t>> filterframes;
        RGYCLFrameInterop *clFrameInInterop = nullptr;

        bool drain = !frame;
        if (!frame) {
            filterframes.push_back(std::make_pair(RGYFrameInfo(), 0u));
        } else {
            auto taskSurf = dynamic_cast<PipelineTaskOutputSurf *>(frame.get());
            if (taskSurf == nullptr) {
                PrintMes(RGY_LOG_ERROR, _T("Invalid task surface.\n"));
                return RGY_ERR_NULL_PTR;
            }
            if (taskSurf->surf().mfx()) {
                mfxFrameSurface1 *surfVppIn = taskSurf->surf().mfx()->surf();
                if (m_surfVppInInterop.count(surfVppIn) == 0) {
                    m_surfVppInInterop[surfVppIn] = getOpenCLFrameInterop(surfVppIn, m_memType, CL_MEM_READ_ONLY, m_allocator, m_cl.get(), m_cl->queue(), m_vpFilters.front()->GetFilterParam()->frameIn);
                }
                clFrameInInterop = m_surfVppInInterop[surfVppIn].get();
                if (!clFrameInInterop) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to get OpenCL interop [in].\n"));
                    return RGY_ERR_NULL_PTR;
                }
                auto err = clFrameInInterop->acquire(m_cl->queue());
                if (err != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to acquire OpenCL interop [in]: %s.\n"), get_err_mes(err));
                    return RGY_ERR_NULL_PTR;
                }
                clFrameInInterop->frame.flags = taskSurf->surf().frame()->flags();
                clFrameInInterop->frame.timestamp = taskSurf->surf().frame()->timestamp();
                clFrameInInterop->frame.inputFrameId = taskSurf->surf().frame()->inputFrameId();
                clFrameInInterop->frame.picstruct = taskSurf->surf().frame()->picstruct();
                clFrameInInterop->frame.dataList = taskSurf->surf().frame()->dataList();
                filterframes.push_back(std::make_pair(clFrameInInterop->frameInfo(), 0u));
            } else if (auto clframe = taskSurf->surf().cl(); clframe != nullptr) {
                //OpenCLフレームが出てきた時の場合
                filterframes.push_back(std::make_pair(clframe->frameInfo(), 0u));
            } else {
                PrintMes(RGY_LOG_ERROR, _T("Invalid input frame.\n"));
                return RGY_ERR_NULL_PTR;
            }
            //ここでinput frameの参照を m_prevInputFrame で保持するようにして、OpenCLによるフレームの処理が完了しているかを確認できるようにする
            //これを行わないとこのフレームが再度使われてしまうことになる
            m_prevInputFrame.push_back(std::move(frame));
        }
        if (m_stopwatch) m_stopwatch->add(0, 1);
#define FRAME_COPY_ONLY 0
#if !FRAME_COPY_ONLY
        std::vector<std::unique_ptr<PipelineTaskOutputSurf>> outputSurfs;
        while (filterframes.size() > 0 || drain) {
            auto surfVppOut = getWorkSurf();
            RGYCLFrameInterop *clFrameOutInterop = nullptr;
            if (auto mfxsurfOut = (surfVppOut.mfx()) ? surfVppOut.mfx()->surf() : nullptr; mfxsurfOut != nullptr) {
                // 通常のmfxフレームの場合
                if (m_surfVppOutInterop.count(mfxsurfOut) == 0) {
                    m_surfVppOutInterop[mfxsurfOut] = getOpenCLFrameInterop(mfxsurfOut, m_memType, CL_MEM_WRITE_ONLY, m_allocator, m_cl.get(), m_cl->queue(), m_vpFilters.back()->GetFilterParam()->frameOut);
                }
                clFrameOutInterop = m_surfVppOutInterop[mfxsurfOut].get();
                if (!clFrameOutInterop) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to get OpenCL interop [out].\n"));
                    return RGY_ERR_NULL_PTR;
                }
                auto err = clFrameOutInterop->acquire(m_cl->queue());
                if (err != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to acquire OpenCL interop [out]: %s.\n"), get_err_mes(err));
                    return RGY_ERR_NULL_PTR;
                }
            } else if (surfVppOut.cl() != nullptr) {
                //OpenCLフレームが出てきた時の場合...特にすることはない
            } else {
                PrintMes(RGY_LOG_ERROR, _T("Invalid work frame [out].\n"));
                return RGY_ERR_NULL_PTR;
            }
            #define clFrameOutInteropRelease { if (clFrameOutInterop) clFrameOutInterop->release(); }
            //フィルタリングするならここ
            for (uint32_t ifilter = filterframes.front().second; ifilter < m_vpFilters.size() - 1; ifilter++) {
                // コピーを作ってそれをfilter関数に渡す
                // vpp-rffなどoverwirteするフィルタのときに、filterframes.pop_front -> push がうまく動作しない
                RGYFrameInfo input = filterframes.front().first;

                int nOutFrames = 0;
                RGYFrameInfo *outInfo[16] = { 0 };
                auto sts_filter = m_vpFilters[ifilter]->filter(&input, (RGYFrameInfo **)&outInfo, &nOutFrames);
                if (sts_filter != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), m_vpFilters[ifilter]->name().c_str());
                    clFrameOutInteropRelease;
                    return sts_filter;
                }
                if (clFrameInInterop) {
                    RGYOpenCLEvent inputReleaseEvent;
                    clFrameInInterop->release(&inputReleaseEvent); // input frameの解放
                    clFrameInInterop = nullptr;
                    if (!m_prevInputFrame.empty() && m_prevInputFrame.back()) {
                        //解放処理のeventを入力フレームを使用し終わったことの合図として登録する
                        dynamic_cast<PipelineTaskOutputSurf *>(m_prevInputFrame.back().get())->addClEvent(inputReleaseEvent);
                    }
                }
                if (nOutFrames == 0) {
                    if (drain) {
                        filterframes.front().second++;
                        continue;
                    }
                    clFrameOutInteropRelease;
                    return RGY_ERR_NONE;
                }
                filterframes.pop_front();
                drain = false; //途中でフレームが出てきたら、drain完了していない

                //最初に出てきたフレームは先頭に追加する
                for (int jframe = nOutFrames - 1; jframe >= 0; jframe--) {
                    filterframes.push_front(std::make_pair(*outInfo[jframe], ifilter + 1));
                }
            }
            if (drain) {
                clFrameOutInteropRelease;
                return RGY_ERR_MORE_DATA; //最後までdrain = trueなら、drain完了
            }
            //エンコードバッファにコピー
            auto &lastFilter = m_vpFilters[m_vpFilters.size() - 1];
            //最後のフィルタはRGYFilterCspCropでなければならない
            if (typeid(*lastFilter.get()) != typeid(RGYFilterCspCrop)) {
                PrintMes(RGY_LOG_ERROR, _T("Last filter setting invalid.\n"));
                clFrameOutInteropRelease;
                return RGY_ERR_INVALID_PARAM;
            }
            //エンコードバッファのポインタを渡す
            int nOutFrames = 0;
            auto encSurfaceInfo = (clFrameOutInterop) ? clFrameOutInterop->frameInfo() : surfVppOut.cl()->frameInfo();
            RGYFrameInfo *outInfo[1];
            outInfo[0] = &encSurfaceInfo;
            RGYOpenCLEvent clevent; // 最終フィルタの処理完了を伝えるevent
            auto sts_filter = lastFilter->filter(&filterframes.front().first, (RGYFrameInfo **)&outInfo, &nOutFrames, m_cl->queue(), &clevent);
            if (sts_filter != RGY_ERR_NONE) {
                PrintMes(RGY_LOG_ERROR, _T("Error while running filter \"%s\".\n"), lastFilter->name().c_str());
                clFrameOutInteropRelease;
                return sts_filter;
            }
            if (m_videoMetric) {
                //フレームを転送
                int dummy = 0;
                auto err = m_videoMetric->filter(&filterframes.front().first, nullptr, &dummy, m_cl->queue(), &clevent);
                if (err != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to send frame for video metric calcualtion: %s.\n"), get_err_mes(err));
                    return err;
                }
            }
            filterframes.pop_front();

            if (clFrameOutInterop) {
                auto err = clFrameOutInterop->release(&clevent);
                if (err != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to release out frame interop after \"%s\".\n"), lastFilter->name().c_str());
                    return sts_filter;
                }
            }
            surfVppOut.frame()->setTimestamp(encSurfaceInfo.timestamp);
            surfVppOut.frame()->setInputFrameId(encSurfaceInfo.inputFrameId);
            surfVppOut.frame()->setPicstruct(encSurfaceInfo.picstruct);
            surfVppOut.frame()->setFlags(encSurfaceInfo.flags);
            surfVppOut.frame()->setDataList(encSurfaceInfo.dataList);

            outputSurfs.push_back(std::make_unique<PipelineTaskOutputSurf>(m_mfxSession, surfVppOut, frame, clevent));

            #undef clFrameOutInteropRelease
        }
        if (clFrameInInterop) {
            RGYOpenCLEvent clevent;
            clFrameInInterop->release(&clevent); // input frameの解放
            for (auto& surf : outputSurfs) {
                surf->addClEvent(clevent);
            }
            if (!m_prevInputFrame.empty() && m_prevInputFrame.back()) {
                //解放処理のeventを入力フレームを使用し終わったことの合図として登録する
                dynamic_cast<PipelineTaskOutputSurf *>(m_prevInputFrame.back().get())->addClEvent(clevent);
            }
        }
        m_outQeueue.insert(m_outQeueue.end(),
            std::make_move_iterator(outputSurfs.begin()),
            std::make_move_iterator(outputSurfs.end())
        );
        if (m_stopwatch) m_stopwatch->add(0, 2);
#else
        auto surfVppOut = getWorkSurf();
        if (m_surfVppOutInterop.count(surfVppOut.get()) == 0) {
            m_surfVppOutInterop[surfVppOut.get()] = getOpenCLFrameInterop(surfVppOut.get(), m_memType, CL_MEM_WRITE_ONLY, m_allocator, m_cl.get(), m_cl->queue(), m_vpFilters.front()->GetFilterParam()->frameIn);
        }
        auto clFrameOutInterop = m_surfVppOutInterop[surfVppOut.get()].get();
        if (!clFrameOutInterop) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to get OpenCL interop [out].\n"));
            return RGY_ERR_NULL_PTR;
        }
        auto err = clFrameOutInterop->acquire(m_cl->queue());
        if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to acquire OpenCL interop [out]: %s.\n"), get_err_mes(err));
            return RGY_ERR_NULL_PTR;
        }
        auto inputSurface = clFrameInInterop->frameInfo();
        surfVppOut->Data.TimeStamp = inputSurface.timestamp;
        surfVppOut->Data.FrameOrder = inputSurface.inputFrameId;
        surfVppOut->Info.PicStruct = picstruct_rgy_to_enc(inputSurface.picstruct);
        surfVppOut->Data.DataFlag = (mfxU16)inputSurface.flags;

        auto encSurfaceInfo = clFrameOutInterop->frameInfo();
        RGYOpenCLEvent clevent;
        m_cl->copyFrame(&encSurfaceInfo, &inputSurface, nullptr, m_cl->queue(), &clevent);
        if (clFrameInInterop) {
            clFrameInInterop->release(&clevent);
            if (!m_prevInputFrame.empty() && m_prevInputFrame.back()) {
                dynamic_cast<PipelineTaskOutputSurf *>(m_prevInputFrame.back().get())->addClEvent(clevent);
            }
        }
        clFrameOutInterop->release(&clevent);
        m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_mfxSession, surfVppOut, frame, clevent));
#endif
        return RGY_ERR_NONE;
    }
};

class PipelineTaskOutputRaw : public PipelineTask {
public:
    PipelineTaskOutputRaw(MFXVideoSession *mfxSession, int outMaxQueueSize, mfxVersion mfxVer, std::shared_ptr<RGYLog> log) :
        PipelineTask(PipelineTaskType::OUTPUTRAW, outMaxQueueSize, mfxSession, mfxVer, log) {
    };
    virtual ~PipelineTaskOutputRaw() {};

    virtual std::optional<mfxFrameAllocRequest> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<mfxFrameAllocRequest> requiredSurfOut() override { return std::nullopt; };

    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (!frame) {
            return RGY_ERR_MORE_DATA;
        }
        m_inFrames++;
        m_outQeueue.push_back(std::move(frame));
        return RGY_ERR_NONE;
    }
};
#endif // __QSV_PIPELINE_CTRL_H__
