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
        return mfxsurf() == nullptr && clframe() == nullptr;
    }
    bool operator !=(const PipelineTaskSurface& obj) const { return mfxsurf() != obj.mfxsurf() || clframe() != obj.clframe(); }
    bool operator ==(const PipelineTaskSurface& obj) const { return mfxsurf() == obj.mfxsurf() && clframe() == obj.clframe(); }
    bool operator !=(std::nullptr_t) const { return mfxsurf() != nullptr || clframe() != nullptr; }
    bool operator ==(std::nullptr_t) const { return mfxsurf() == nullptr && clframe() == nullptr; }
    const mfxFrameSurface1 *mfxsurf() const { return (surf) ? surf->surf() : nullptr; }
    mfxFrameSurface1 *mfxsurf() { return (surf) ? surf->surf() : nullptr; }
    const RGYCLFrame *clframe() const { return (surf) ? surf->clframe() : nullptr; }
    RGYCLFrame *clframe() { return (surf) ? surf->clframe() : nullptr; }
    const RGYFrame *frame() const { return surf; }
    RGYFrame *frame() { return surf; }
};

// アプリ用の独自参照カウンタと組み合わせたクラス
class PipelineTaskSurfaces {
private:
    std::deque<std::pair<std::unique_ptr<RGYFrame>, std::atomic<int>>> m_surfaces; // フレームと参照カウンタ
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
            m_surfaces[i].first = std::unique_ptr<RGYFrame>(new RGYFrameMFXSurf(surfs[i]));
            m_surfaces[i].second = 0;
        }
    }
    void setSurfaces(std::vector<std::unique_ptr<RGYCLFrame>>& frames) {
        clear();
        m_surfaces.resize(frames.size());
        for (size_t i = 0; i < m_surfaces.size(); i++) {
            m_surfaces[i].first = std::unique_ptr<RGYFrame>(new RGYFrameCL(frames[i]));
            m_surfaces[i].second = 0;
        }
    }

    PipelineTaskSurface getFreeSurf() {
        for (auto& s : m_surfaces) {
            if (isFree(&s)) {
                return PipelineTaskSurface(s.first.get(), &s.second);
            }
        }
        return PipelineTaskSurface();
    }
    PipelineTaskSurface get(mfxFrameSurface1 *surf) {
        auto s = findSurf(surf);
        if (s != nullptr) {
            return PipelineTaskSurface(s->first.get(), &s->second);
        }
        return PipelineTaskSurface();
    }
    PipelineTaskSurface get(RGYCLFrame *frame) {
        auto s = findSurf(frame);
        if (s != nullptr) {
            return PipelineTaskSurface(s->first.get(), &s->second);
        }
        return PipelineTaskSurface();
    }
    size_t bufCount() const { return m_surfaces.size(); }

    bool isAllFree() const {
        for (const auto& s : m_surfaces) {
            if (!isFree(&s)) {
                return false;
            }
        }
        return true;
    }

    // 使用されていないフレームかを返す
    // mfxの参照カウンタと独自参照カウンタの両方をチェック
    bool isFree(const std::pair<std::unique_ptr<RGYFrame>, std::atomic<int>> *s) const {
        if (s->second != 0) return false;
        auto mfxsurf = s->first->surf();
        if (mfxsurf) {
            return mfxsurf->Data.Locked == 0;
        }
        return true;
    }
protected:
    std::pair<std::unique_ptr<RGYFrame>, std::atomic<int>> *findSurf(mfxFrameSurface1 *surf) {
        for (auto& s : m_surfaces) {
            if (s.first->surf() == surf) {
                return &s;
            }
        }
        return nullptr;
    }
    std::pair<std::unique_ptr<RGYFrame>, std::atomic<int>> *findSurf(RGYCLFrame *frame) {
        for (auto& s : m_surfaces) {
            if (s.first->clframe() == frame) {
                return &s;
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
        auto mfxSurf = m_surf.mfxsurf();
        if (mfxSurf->Data.MemId) {
            auto sts = allocator->Lock(allocator->pthis, mfxSurf->Data.MemId, &(mfxSurf->Data));
            if (sts < MFX_ERR_NONE) {
                return err_to_rgy(sts);
            }
        }
        auto err = writer->WriteNextFrame(m_surf.frame());
        if (mfxSurf->Data.MemId) {
            allocator->Unlock(allocator->pthis, mfxSurf->Data.MemId, &(mfxSurf->Data));
        }
        return err;
    }

    RGY_ERR writeCL(RGYOutput *writer, RGYOpenCLQueue *clqueue) {
        if (clqueue == nullptr) {
            return RGY_ERR_NULL_PTR;
        }
        auto clframe = m_surf.clframe();
        auto err = clframe->queueMapBuffer(*clqueue, CL_MAP_READ); // CPUが読み込むためにmapする
        if (err != RGY_ERR_NONE) {
            return err;
        }
        clframe->mapWait();
        auto mappedframe = std::make_unique<RGYFrameRef>(clframe->mappedHost());
        err = writer->WriteNextFrame(mappedframe.get());
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
        auto err = (m_surf.mfxsurf() != nullptr) ? writeMFX(writer, allocator) : writeCL(writer, clqueue);
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
public:
    PipelineTask() : m_type(PipelineTaskType::UNKNOWN), m_outQeueue(), m_workSurfs(), m_mfxSession(nullptr), m_allocator(nullptr), m_allocResponse({ 0 }), m_inFrames(0), m_outFrames(0), m_outMaxQueueSize(0), m_mfxVer({ 0 }), m_log() {};
    PipelineTask(PipelineTaskType type, int outMaxQueueSize, MFXVideoSession *mfxSession, mfxVersion mfxVer, std::shared_ptr<RGYLog> log) :
        m_type(type), m_outQeueue(), m_workSurfs(), m_mfxSession(mfxSession), m_allocator(nullptr), m_allocResponse({ 0 }), m_inFrames(0), m_outFrames(0), m_outMaxQueueSize(outMaxQueueSize), m_mfxVer(mfxVer), m_log(log) {
    };
    virtual ~PipelineTask() {
        if (m_allocator) {
            m_allocator->Free(m_allocator->pthis, &m_allocResponse);
        }
        m_workSurfs.clear();
    }
    virtual bool isPassThrough() const { return false; }
    virtual tstring print() const { return getPipelineTaskTypeName(m_type); }
    virtual std::optional<mfxFrameAllocRequest> requiredSurfIn() = 0;
    virtual std::optional<mfxFrameAllocRequest> requiredSurfOut() = 0;
    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) = 0;
    virtual RGY_ERR getOutputFrameInfo(mfxFrameInfo& info) { info = { 0 }; return RGY_ERR_INVALID_CALL; }
    virtual std::vector<std::unique_ptr<PipelineTaskOutput>> getOutput(const bool sync) {
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
    std::shared_ptr<RGYOpenCLContext> m_cl;
public:
    PipelineTaskInput(MFXVideoSession *mfxSession, QSVAllocator *allocator, int outMaxQueueSize, RGYInput *input, mfxVersion mfxVer, std::shared_ptr<RGYOpenCLContext> cl, std::shared_ptr<RGYLog> log)
        : PipelineTask(PipelineTaskType::INPUT, outMaxQueueSize, mfxSession, mfxVer, log), m_input(input), m_allocator(allocator), m_cl(cl) {

    };
    virtual ~PipelineTaskInput() {};
    virtual std::optional<mfxFrameAllocRequest> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<mfxFrameAllocRequest> requiredSurfOut() override { return std::nullopt; };
    RGY_ERR loadNextFrameMFX(PipelineTaskSurface& surfWork) {
        auto mfxSurf = surfWork.mfxsurf();
        if (mfxSurf->Data.MemId) {
            auto sts = m_allocator->Lock(m_allocator->pthis, mfxSurf->Data.MemId, &(mfxSurf->Data));
            if (sts < MFX_ERR_NONE) {
                return err_to_rgy(sts);
            }
        }
        auto err = m_input->LoadNextFrame(surfWork.frame());
        if (err != RGY_ERR_NONE) {
            //Unlockする必要があるので、ここに入ってもすぐにreturnしてはいけない
            if (err == RGY_ERR_MORE_DATA) { // EOF
                err = RGY_ERR_MORE_BITSTREAM; // EOF を PipelineTaskMFXDecode のreturnコードに合わせる
            } else {
                PrintMes(RGY_LOG_ERROR, _T("Error in reader: %s.\n"), get_err_mes(err));
            }
        }
        if (mfxSurf->Data.MemId) {
            m_allocator->Unlock(m_allocator->pthis, mfxSurf->Data.MemId, &(mfxSurf->Data));
        }
        return err;
    }
    RGY_ERR loadNextFrameCL(PipelineTaskSurface& surfWork) {
        auto clframe = surfWork.clframe();
        auto err = clframe->queueMapBuffer(m_cl->queue(), CL_MAP_WRITE); // CPUが書き込むためにMapする
        if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to map buffer: %s.\n"), get_err_mes(err));
            return err;
        }
        clframe->mapWait(); //すぐ終わるはず
        auto mappedframe = std::make_unique<RGYFrameRef>(clframe->mappedHost());
        err = m_input->LoadNextFrame(mappedframe.get());
        if (err != RGY_ERR_NONE) {
            //Unlockする必要があるので、ここに入ってもすぐにreturnしてはいけない
            if (err == RGY_ERR_MORE_DATA) { // EOF
                err = RGY_ERR_MORE_BITSTREAM; // EOF を PipelineTaskMFXDecode のreturnコードに合わせる
            } else {
                PrintMes(RGY_LOG_ERROR, _T("Error in reader: %s.\n"), get_err_mes(err));
            }
        }
        copyFrameProp(&clframe->frame, mappedframe->info());
        auto clerr = clframe->unmapBuffer();
        if (clerr != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("Failed to unmap buffer: %s.\n"), get_err_mes(err));
            if (err == RGY_ERR_NONE) {
                err = clerr;
            }
        }
        return err;
    }
    virtual RGY_ERR sendFrame([[maybe_unused]] std::unique_ptr<PipelineTaskOutput>& frame) override {
        auto surfWork = getWorkSurf();
        if (surfWork == nullptr) {
            PrintMes(RGY_LOG_ERROR, _T("failed to get work surface for input.\n"));
            return RGY_ERR_NOT_ENOUGH_BUFFER;
        }
        auto err = (surfWork.mfxsurf() != nullptr) ? loadNextFrameMFX(surfWork) : loadNextFrameCL(surfWork);
        if (err == RGY_ERR_NONE) {
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
    bool m_getNextBitstream;
    int m_decFrameOutCount;
    RGYBitstream m_decInputBitstream;
    RGYQueueMPMP<RGYFrameDataMetadata*> m_queueHDR10plusMetadata;
    RGYQueueMPMP<FrameFlags> m_dataFlag;
public:
    PipelineTaskMFXDecode(MFXVideoSession *mfxSession, int outMaxQueueSize, MFXVideoDECODE *mfxdec, mfxVideoParam& decParams, RGYInput *input, mfxVersion mfxVer, std::shared_ptr<RGYLog> log)
        : PipelineTask(PipelineTaskType::MFXDEC, outMaxQueueSize, mfxSession, mfxVer, log), m_dec(mfxdec), m_mfxDecParams(decParams), m_input(input), m_getNextBitstream(true), m_decFrameOutCount(0), m_decInputBitstream(), m_queueHDR10plusMetadata(), m_dataFlag() {
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
    };
    void setDec(MFXVideoDECODE *mfxdec) { m_dec = mfxdec; };

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
        mfxBitstream *inputBitstream = (m_getNextBitstream) ? &m_decInputBitstream.bitstream() : nullptr;
        auto mfxSurfDecWork = surfDecWork.mfxsurf();
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
            if (inputBitstream->TimeStamp == (mfxU64)AV_NOPTS_VALUE) {
                inputBitstream->TimeStamp = (mfxU64)MFX_TIMESTAMP_UNKNOWN;
            }
            inputBitstream->DecodeTimeStamp = MFX_TIMESTAMP_UNKNOWN;
        }
        mfxSurfDecWork->Data.TimeStamp = (mfxU64)MFX_TIMESTAMP_UNKNOWN;
        mfxSurfDecWork->Data.DataFlag |= MFX_FRAMEDATA_ORIGINAL_TIMESTAMP;
        m_inFrames++;

        mfxStatus dec_sts = MFX_ERR_NONE;
        mfxSyncPoint lastSyncP = nullptr;
        mfxFrameSurface1 *surfDecOut = nullptr;
        for (int i = 0; ; i++) {
            const auto inputDataLen = (inputBitstream) ? inputBitstream->DataLength : 0;
            mfxSyncPoint decSyncPoint = nullptr;
            dec_sts = m_dec->DecodeFrameAsync(inputBitstream, mfxSurfDecWork, &surfDecOut, &decSyncPoint);
            lastSyncP = decSyncPoint;

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
                    PrintMes((inputDataLen >= 10) ? RGY_LOG_WARN : RGY_LOG_DEBUG,
                        _T("DecodeFrameAsync: removing %d bytes from input bitstream not read by decoder.\n"), inputDataLen);
                    inputBitstream->DataLength = 0;
                    inputBitstream->DataOffset = 0;
                }
                break;
            }
        }
        if (surfDecOut != nullptr && lastSyncP != nullptr) {
            auto taskSurf = useTaskSurf(surfDecOut);
            taskSurf.frame()->clearDataList();
            taskSurf.frame()->setInputFrameId(m_decFrameOutCount++);
            if (getDataFlag(surfDecOut->Data.TimeStamp) & RGY_FRAME_FLAG_RFF) {
                taskSurf.frame()->setFlags(RGY_FRAME_FLAG_RFF);
            }
            if (auto data = getMetadata(RGY_FRAME_DATA_HDR10PLUS, surfDecOut->Data.TimeStamp); data) {
                taskSurf.frame()->dataList().push_back(data);
            }
            if (auto data = getMetadata(RGY_FRAME_DATA_DOVIRPU, surfDecOut->Data.TimeStamp); data) {
                taskSurf.frame()->dataList().push_back(data);
            }
            m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_mfxSession, taskSurf, lastSyncP));
        }
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
                    } else if (frameData->dataType() == RGY_FRAME_DATA_DOVIRPU) {
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
    int64_t m_outFrameDuration; //(m_outputTimebase基準)
    int64_t m_tsOutFirst;     //(m_outputTimebase基準)
    int64_t m_tsOutEstimated; //(m_outputTimebase基準)
    int64_t m_tsPrev;         //(m_outputTimebase基準)
public:
    PipelineTaskCheckPTS(MFXVideoSession *mfxSession, rgy_rational<int> srcTimebase, rgy_rational<int> outputTimebase, int64_t outFrameDuration, RGYAVSync avsync, bool vpp_afs_rff_aware, mfxVersion mfxVer, std::shared_ptr<RGYLog> log) :
        PipelineTask(PipelineTaskType::CHECKPTS, /*outMaxQueueSize = */ 0 /*常に0である必要がある*/, mfxSession, mfxVer, log),
        m_srcTimebase(srcTimebase), m_outputTimebase(outputTimebase), m_avsync(avsync), m_vpp_rff(false), m_vpp_afs_rff_aware(vpp_afs_rff_aware), m_outFrameDuration(outFrameDuration), m_tsOutFirst(-1), m_tsOutEstimated(0), m_tsPrev(-1) {
    };
    virtual ~PipelineTaskCheckPTS() {};

    virtual bool isPassThrough() const override {
        // そのまま渡すのでpaththrough
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
            && ((m_avsync & (RGY_AVSYNC_VFR | RGY_AVSYNC_FORCE_CFR)) || m_vpp_rff || m_vpp_afs_rff_aware)) {
            //CFR仮定ではなく、オリジナルの時間を見る
            const auto srcTimestamp = taskSurf->surf().frame()->timestamp();
            outPtsSource = rational_rescale(srcTimestamp, m_srcTimebase, m_outputTimebase);
        }
        PrintMes(RGY_LOG_TRACE, _T("check_pts(%d/%d): nOutEstimatedPts %lld, outPtsSource %lld, outDuration %d\n"), taskSurf->surf().frame()->inputFrameId(), m_inFrames, m_tsOutEstimated, outPtsSource, outDuration);
        if (m_tsOutFirst < 0) {
            m_tsOutFirst = outPtsSource; //最初のpts
            PrintMes(RGY_LOG_TRACE, _T("check_pts: m_tsOutFirst %lld\n"), outPtsSource);
        }
        //最初のptsを0に修正
        outPtsSource -= m_tsOutFirst;

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
                PrintMes(RGY_LOG_TRACE, _T("check_pts(%d):   estimetaed orig_pts %lld, framePos %d\n"), taskSurf->surf()->Data.FrameOrder, orig_pts, framePos.poc);
                if (framePos.poc != FRAMEPOS_POC_INVALID && framePos.duration > 0) {
                    //有効な値ならオリジナルのdurationを使用する
                    outDuration = rational_rescale(framePos.duration, to_rgy(streamIn->time_base), m_outputTimebase);
                    PrintMes(RGY_LOG_TRACE, _T("check_pts(%d):   changing duration to original: %d\n"), taskSurf->surf()->Data.FrameOrder, outDuration);
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
        std::unique_ptr<PipelineTaskOutputDataCustom> timestampOverride(new PipelineTaskOutputDataCheckPts(outPtsSource));
        m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_mfxSession, outSurf, lastSyncPoint, timestampOverride));
        return RGY_ERR_NONE;
    }
    //checkptsではtimestampを上書きするため特別に常に1フレームしか取り出さない
    //これは--avsync frocecfrでフレームを参照コピーする際、
    //mfxSurface1自体は同じデータを指すため、複数のタイムスタンプを持つことができないため、
    //1フレームずつgetOutputし、都度タイムスタンプを上書きしてすぐに後続のタスクに投入してタイムスタンプを反映させる必要があるため
    virtual std::vector<std::unique_ptr<PipelineTaskOutput>> getOutput(const bool sync) override {
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
                const int nTrackId = (int)((uint32_t)pkt->flags >> 16);
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

class PipelineTaskTrim : public PipelineTask {
protected:
    const sTrimParam &m_trimParam;
    RGYInput *m_input;
    rgy_rational<int> m_srcTimebase;
public:
    PipelineTaskTrim(const sTrimParam &trimParam, RGYInput *input, const rgy_rational<int>& srcTimebase, int outMaxQueueSize, mfxVersion mfxVer, std::shared_ptr<RGYLog> log) :
        PipelineTask(PipelineTaskType::TRIM, outMaxQueueSize, nullptr, mfxVer, log),
        m_trimParam(trimParam), m_input(input), m_srcTimebase(srcTimebase) {
    };
    virtual ~PipelineTaskTrim() {};

    virtual bool isPassThrough() const override { return true; }
    virtual std::optional<mfxFrameAllocRequest> requiredSurfIn() override { return std::nullopt; };
    virtual std::optional<mfxFrameAllocRequest> requiredSurfOut() override { return std::nullopt; };

    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
        if (!frame) {
            return RGY_ERR_MORE_DATA;
        }
        m_inFrames++;
        PipelineTaskOutputSurf *taskSurf = dynamic_cast<PipelineTaskOutputSurf *>(frame.get());
        if (!frame_inside_range(taskSurf->surf().frame()->inputFrameId(), m_trimParam.list).first) {
            return RGY_ERR_NONE;
        }
        if (!m_input->checkTimeSeekTo(taskSurf->surf().frame()->timestamp(), m_srcTimebase)) {
            return RGY_ERR_NONE; //seektoにより脱落させるフレーム
        }
        m_outQeueue.push_back(std::make_unique<PipelineTaskOutputSurf>(m_mfxSession, taskSurf->surf(), taskSurf->syncpoint()));
        return RGY_ERR_NONE;
    }
};

class PipelineTaskMFXVpp : public PipelineTask {
protected:
    MFXVideoVPP *m_vpp;
    RGYTimestamp m_timestamp;
    mfxVideoParam& m_mfxVppParams;
    std::vector<std::shared_ptr<RGYFrameData>> m_lastFrameDataList;
public:
    PipelineTaskMFXVpp(MFXVideoSession *mfxSession, int outMaxQueueSize, MFXVideoVPP *mfxvpp, mfxVideoParam& vppParams, mfxVersion mfxVer, std::shared_ptr<RGYLog> log)
        : PipelineTask(PipelineTaskType::MFXVPP, outMaxQueueSize, mfxSession, mfxVer, log), m_vpp(mfxvpp), m_timestamp(), m_mfxVppParams(vppParams), m_lastFrameDataList() {};
    virtual ~PipelineTaskMFXVpp() {};
    void setVpp(MFXVideoVPP *mfxvpp) { m_vpp = mfxvpp; };
protected:
    RGY_ERR requiredSurfInOut(std::array<mfxFrameAllocRequest,2>& allocRequest) {
        for (auto& request : allocRequest) {
            memset(&request, 0, sizeof(request));
        }
        // allocRequest[0]はvppへの入力, allocRequest[1]はvppからの出力
        auto err = err_to_rgy(m_vpp->QueryIOSurf(&m_mfxVppParams, allocRequest.data()));
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
        if (frame && frame->type() != PipelineTaskOutputType::SURFACE) {
            PrintMes(RGY_LOG_ERROR, _T("Invalid frame type.\n"));
            return RGY_ERR_UNSUPPORTED;
        }

        mfxStatus vpp_sts = MFX_ERR_NONE;

        if (frame) {
            m_lastFrameDataList = dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->surf().frame()->dataList();
        }

        mfxFrameSurface1 *surfVppIn = (frame) ? dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->surf().mfxsurf() : nullptr;
        //vpp前に、vpp用のパラメータでFrameInfoを更新
        copy_crop_info(surfVppIn, &m_mfxVppParams.mfx.FrameInfo);
        if (surfVppIn) {
            m_timestamp.add(surfVppIn->Data.TimeStamp, dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->surf().frame()->inputFrameId(), 0 /*dummy*/, 0, {});
            surfVppIn->Data.DataFlag |= MFX_FRAMEDATA_ORIGINAL_TIMESTAMP;
            m_inFrames++;
        }

        bool vppMoreOutput = false;
        do {
            vppMoreOutput = false;
            auto surfVppOut = getWorkSurf();
            mfxSyncPoint lastSyncPoint = nullptr;
            for (int i = 0; ; i++) {
                //bob化の際、pSurfVppInに連続で同じフレーム(同じtimestamp)を投入すると、
                //最初のフレームには設定したtimestamp、次のフレームにはMFX_TIMESTAMP_UNKNOWNが設定されて出てくる
                //特別pSurfVppOut側のTimestampを設定する必要はなさそう
                mfxSyncPoint VppSyncPoint = nullptr;
                vpp_sts = m_vpp->RunFrameVPPAsync(surfVppIn, surfVppOut.mfxsurf(), nullptr, &VppSyncPoint);
                lastSyncPoint = VppSyncPoint;

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
            }

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
        mfxFrameSurface1 *surfVppIn = taskSurf->surf().mfxsurf();
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
            clFrameInInterop->frame.flags = (RGY_FRAME_FLAGS)surfVppIn->Data.DataFlag;
            clFrameInInterop->frame.timestamp = surfVppIn->Data.TimeStamp;
            clFrameInInterop->frame.inputFrameId = surfVppIn->Data.FrameOrder;
            clFrameInInterop->frame.picstruct = picstruct_enc_to_rgy(surfVppIn->Info.PicStruct);
            inputFrame = clFrameInInterop->frameInfo();
        } else if (taskSurf->surf().clframe() != nullptr) {
            //OpenCLフレームが出てきた時の場合
            auto clframe = taskSurf->surf().clframe();
            if (clframe == nullptr) {
                PrintMes(RGY_LOG_ERROR, _T("Invalid cl frame.\n"));
                return RGY_ERR_NULL_PTR;
            }
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
    mfxVideoParam& m_mfxEncParams;
    rgy_rational<int> m_outputTimebase;
    RGYListRef<RGYBitstream> m_bitStreamOut;
    RGYHDR10Plus *m_hdr10plus;
    bool m_hdr10plusMetadataCopy;
    encCtrlData m_encCtrlData;
public:
    PipelineTaskMFXEncode(
        MFXVideoSession *mfxSession, int outMaxQueueSize, MFXVideoENCODE *mfxencode, mfxVersion mfxVer, mfxVideoParam& encParams,
        RGYTimecode *timecode, RGYTimestamp *encTimestamp, rgy_rational<int> outputTimebase, RGYHDR10Plus *hdr10plus, bool hdr10plusMetadataCopy, std::shared_ptr<RGYLog> log)
        : PipelineTask(PipelineTaskType::MFXENCODE, outMaxQueueSize, mfxSession, mfxVer, log),
        m_encode(mfxencode), m_timecode(timecode), m_encTimestamp(encTimestamp), m_mfxEncParams(encParams), m_outputTimebase(outputTimebase), m_bitStreamOut(), m_hdr10plus(hdr10plus), m_hdr10plusMetadataCopy(hdr10plusMetadataCopy), m_encCtrlData() {};
    virtual ~PipelineTaskMFXEncode() {
        m_outQeueue.clear(); // m_bitStreamOutが解放されるよう前にこちらを解放する
    };
    void setEnc(MFXVideoENCODE *mfxencode) { m_encode = mfxencode; };

    virtual std::optional<mfxFrameAllocRequest> requiredSurfIn() override {
        mfxFrameAllocRequest allocRequest = { 0 };
        auto err = err_to_rgy(m_encode->QueryIOSurf(&m_mfxEncParams, &allocRequest));
        if (err < RGY_ERR_NONE) {
            PrintMes(RGY_LOG_ERROR, _T("  Failed to get required buffer size for %s: %s\n"), getPipelineTaskTypeName(m_type), get_err_mes(err));
            return std::nullopt;
        } else if (err != RGY_ERR_NONE) {
            PrintMes(RGY_LOG_WARN, _T("  surface alloc request for %s: %s\n"), getPipelineTaskTypeName(m_type), get_err_mes(err));
        }
        PrintMes(RGY_LOG_DEBUG, _T("  %s required buffer: %d [%s]\n"), getPipelineTaskTypeName(m_type), allocRequest.NumFrameSuggested, qsv_memtype_str(allocRequest.Type).c_str());
        const int blocksz = (m_mfxEncParams.mfx.CodecId == MFX_CODEC_HEVC) ? 32 : 16;
        allocRequest.Info.Width  = (mfxU16)ALIGN(allocRequest.Info.Width,  blocksz);
        allocRequest.Info.Height = (mfxU16)ALIGN(allocRequest.Info.Height, blocksz);
        return std::optional<mfxFrameAllocRequest>(allocRequest);
    };
    virtual std::optional<mfxFrameAllocRequest> requiredSurfOut() override { return std::nullopt; };
    virtual RGY_ERR sendFrame(std::unique_ptr<PipelineTaskOutput>& frame) override {
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

        std::vector<std::shared_ptr<RGYFrameData>> metadatalist;
        if (m_mfxEncParams.mfx.CodecId == MFX_CODEC_HEVC || m_mfxEncParams.mfx.CodecId == MFX_CODEC_AV1) {
            if (m_hdr10plus) {
                if (const auto data = m_hdr10plus->getData(m_inFrames); data) {
                    metadatalist.push_back(std::make_shared<RGYFrameDataHDR10plus>(data->data(), data->size(), dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->surf().frame()->timestamp()));
                }
            } else if (m_hdr10plusMetadataCopy && frame) {
                metadatalist = dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->surf().frame()->dataList();
            }
        }

        //以下の処理は
        mfxFrameSurface1 *surfEncodeIn = (frame) ? dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->surf().mfxsurf() : nullptr;
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
            m_encTimestamp->add(surfEncodeIn->Data.TimeStamp, inputFrameId, m_inFrames, 0, metadatalist);
            m_inFrames++;
            PrintMes(RGY_LOG_TRACE, _T("send encoder %6d/%6d.\n"), m_inFrames, inputFrameId);
        }
        //エンコーダまでたどり着いたフレームについてはdataListを解放
        if (frame) {
            dynamic_cast<PipelineTaskOutputSurf *>(frame.get())->surf().frame()->clearDataList();
        }

        auto enc_sts = MFX_ERR_NONE;
        mfxSyncPoint lastSyncP = nullptr;
        bool bDeviceBusy = false;
        for (int i = 0; ; i++) {
            auto ctrlPtr = (m_encCtrlData.hasData()) ? m_encCtrlData.getCtrlPtr() : nullptr;
            enc_sts = m_encode->EncodeFrameAsync(ctrlPtr, surfEncodeIn, bsOut->bsptr(), &lastSyncP);
            bDeviceBusy = false;

            if (MFX_ERR_NONE < enc_sts && lastSyncP == nullptr) {
                bDeviceBusy = true;
                if (enc_sts == MFX_WRN_DEVICE_BUSY) {
                    sleep_hybrid(i);
                }
                if (i > 65536 * 1024 * 30) {
                    PrintMes(RGY_LOG_ERROR, _T("device kept on busy for 30s, unknown error occurred.\n"));
                    return RGY_ERR_GPU_HANG;
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
        if (m_prevInputFrame.size() > 0) {
            //前回投入したフレームの処理が完了していることを確認したうえで参照を破棄することでロックを解放する
            auto prevframe = std::move(m_prevInputFrame.front());
            m_prevInputFrame.pop_front();
            prevframe->depend_clear();
        }

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
            mfxFrameSurface1 *surfVppIn = taskSurf->surf().mfxsurf();
            if (surfVppIn != nullptr) {
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
                clFrameInInterop->frame.flags = (RGY_FRAME_FLAGS)surfVppIn->Data.DataFlag;
                clFrameInInterop->frame.timestamp = surfVppIn->Data.TimeStamp;
                clFrameInInterop->frame.inputFrameId = surfVppIn->Data.FrameOrder;
                clFrameInInterop->frame.picstruct = picstruct_enc_to_rgy(surfVppIn->Info.PicStruct);
                clFrameInInterop->frame.dataList = taskSurf->surf().frame()->dataList();
                filterframes.push_back(std::make_pair(clFrameInInterop->frameInfo(), 0u));
            } else if (taskSurf->surf().clframe() != nullptr) {
                //OpenCLフレームが出てきた時の場合
                auto clframe = taskSurf->surf().clframe();
                if (clframe == nullptr) {
                    PrintMes(RGY_LOG_ERROR, _T("Invalid cl frame.\n"));
                    return RGY_ERR_NULL_PTR;
                }
                filterframes.push_back(std::make_pair(clframe->frameInfo(), 0u));
            } else {
                PrintMes(RGY_LOG_ERROR, _T("Invalid input frame.\n"));
                return RGY_ERR_NULL_PTR;
            }
            //ここでinput frameの参照を m_prevInputFrame で保持するようにして、OpenCLによるフレームの処理が完了しているかを確認できるようにする
            //これを行わないとこのフレームが再度使われてしまうことになる
            m_prevInputFrame.push_back(std::move(frame));
        }
#define FRAME_COPY_ONLY 0
#if !FRAME_COPY_ONLY
        std::vector<std::unique_ptr<PipelineTaskOutputSurf>> outputSurfs;
        while (filterframes.size() > 0 || drain) {
            auto surfVppOut = getWorkSurf();
            RGYCLFrameInterop *clFrameOutInterop = nullptr;
            if (surfVppOut.mfxsurf() != nullptr) {
                // 通常のmfxフレームの場合
                if (m_surfVppOutInterop.count(surfVppOut.mfxsurf()) == 0) {
                    m_surfVppOutInterop[surfVppOut.mfxsurf()] = getOpenCLFrameInterop(surfVppOut.mfxsurf(), m_memType, CL_MEM_WRITE_ONLY, m_allocator, m_cl.get(), m_cl->queue(), m_vpFilters.back()->GetFilterParam()->frameOut);
                }
                clFrameOutInterop = m_surfVppOutInterop[surfVppOut.mfxsurf()].get();
                if (!clFrameOutInterop) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to get OpenCL interop [out].\n"));
                    return RGY_ERR_NULL_PTR;
                }
                auto err = clFrameOutInterop->acquire(m_cl->queue());
                if (err != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_ERROR, _T("Failed to acquire OpenCL interop [out]: %s.\n"), get_err_mes(err));
                    return RGY_ERR_NULL_PTR;
                }
            } else if (surfVppOut.clframe() != nullptr) {
                //OpenCLフレームが出てきた時の場合...特にすることはない
            } else {
                PrintMes(RGY_LOG_ERROR, _T("Invalid work frame [out].\n"));
                return RGY_ERR_NULL_PTR;
            }
            #define clFrameOutInteropRelease { if (clFrameOutInterop) clFrameOutInterop->release(); }
            //フィルタリングするならここ
            for (uint32_t ifilter = filterframes.front().second; ifilter < m_vpFilters.size() - 1; ifilter++) {
                int nOutFrames = 0;
                RGYFrameInfo *outInfo[16] = { 0 };
                auto sts_filter = m_vpFilters[ifilter]->filter(&filterframes.front().first, (RGYFrameInfo **)&outInfo, &nOutFrames);
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
            auto encSurfaceInfo = (clFrameOutInterop) ? clFrameOutInterop->frameInfo() : surfVppOut.clframe()->frameInfo();
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
