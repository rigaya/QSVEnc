// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2026 rigaya
// Copyright (c) 2015 Sampsa Sarjanoja
// Copyright (c) 2015-2016 mawen1250
// Copyright (c) 2021 HuangZhangming
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

#include <array>
#include "rgy_filter_cl.h"
#include "rgy_prm.h"

class RGYFilterParamDenoiseBm3d : public RGYFilterParam {
public:
    VppDenoiseBm3d bm3d;
    RGYFilterParamDenoiseBm3d() : bm3d() {};
    virtual ~RGYFilterParamDenoiseBm3d() {};
    virtual tstring print() const override { return bm3d.print(); };
};

class RGYFilterDenoiseBm3d : public RGYFilter {
public:
    RGYFilterDenoiseBm3d(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterDenoiseBm3d();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    RGY_ERR procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    // radius > 0 のときに使うV-BM3D時間方向処理。
    // ring_filledをカーネルへ渡すため、履歴が埋まる前から同じ経路を使える。
    RGY_ERR procPlaneTemporal(int planeIdx, RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    // Per-plane sized scratch buffers for the 3-kernel pipeline. Allocated
    // lazily based on the largest plane dimensions seen so far.
    RGY_ERR ensureScratch(int width, int height);

    // Per-plane ring buffers for V-BM3D temporal (radius > 0). One past-
    // noisy slice + one past-basic-estimate slice per radius step. Allocated
    // when radius>0 and the spatial scratch sizing first lands. Push happens
    // AFTER procPlane succeeds so the just-processed frame becomes available
    // as past history for the next frame.
    RGY_ERR ensureRingBuffers(int planeIdx, int width, int height);
    RGY_ERR pushNoisyToRing(int planeIdx, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue);
    RGY_ERR pushBasicToRing(int planeIdx, RGYOpenCLQueue &queue);

    RGYOpenCLProgramAsync m_bm3d;
    std::unique_ptr<RGYCLBuf> m_bufSimilarCoords;
    std::unique_ptr<RGYCLBuf> m_bufBlockCounts;
    std::unique_ptr<RGYCLBuf> m_bufAccumulator;
    std::unique_ptr<RGYCLBuf> m_bufWeightMap;
    std::unique_ptr<RGYCLBuf> m_bufBasicEstimate;
    // Per-match frame index (uchar per entry, sized to ref_count * group_size_cap).
    // Only allocated when radius > 0. Spatial pipeline doesn't touch it.
    std::unique_ptr<RGYCLBuf> m_bufSimilarFrameIdx;
    // Ring buffer is per-plane (Y / U / V). Slot stride = (W * H * bytes_per_sample)
    // for noisy and (W * H * sizeof(float)) for basic. Single packed cl_mem per
    // plane: layout [slot][H][W] in row-major order.
    std::array<std::unique_ptr<RGYCLBuf>, 3> m_pastNoisyRing;
    std::array<std::unique_ptr<RGYCLBuf>, 3> m_pastBasicRing;
    std::array<int, 3> m_ringW;
    std::array<int, 3> m_ringH;
    std::array<int, 3> m_ringNoisyPitch;
    std::array<int, 3> m_ringBasicPitch;
    int m_ringRadius;        // configured radius (0 = spatial-only)
    int m_ringSlotCursor;    // next slot to fill (mod radius)
    int m_ringFilled;        // number of slots populated (0 .. radius)
    int m_scratchW;
    int m_scratchH;
    int m_scratchBlockStep;
    int m_scratchGroupSize;
    int m_accPitch;
    int m_wmapPitch;
    int m_basicPitch;
};
