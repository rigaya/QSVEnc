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

#include "rgy_filter_denoise_bm3d.h"
#include <cmath>

// Block size is fixed at compile time -- the DCT-8 butterfly math in
// the kernel only implements the 8-point variant. Group caps must
// match MAX_GROUP_SIZE_BASIC / MAX_GROUP_SIZE_WIENER in the kernel.
static const int BM3D_BLOCK_SIZE = 8;
static const int BM3D_MAX_GROUP_SIZE_BASIC = 16;
static const int BM3D_MAX_GROUP_SIZE_WIENER = 32;

RGY_ERR RGYFilterDenoiseBm3d::ensureScratch(int width, int height) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseBm3d>(m_param);
    if (!prm) {
        return RGY_ERR_INVALID_PARAM;
    }
    const int block_step = std::min(prm->bm3d.block_step, BM3D_BLOCK_SIZE);
    // similar_coords / block_counts are shared between the basic and
    // wiener steps; size for the wiener cap (the larger of the two)
    // so the second pass can overwrite without resizing.
    const int group_size = std::min(prm->bm3d.group_size, BM3D_MAX_GROUP_SIZE_WIENER);

    const bool sameLayout = block_step == m_scratchBlockStep && group_size == m_scratchGroupSize;
    if (m_bufSimilarCoords && m_bufBlockCounts && m_bufAccumulator && m_bufWeightMap && m_bufBasicEstimate
        && width <= m_scratchW && height <= m_scratchH && sameLayout) {
        m_accPitch = width * (int)sizeof(float);
        m_wmapPitch = width * (int)sizeof(float);
        m_basicPitch = width * (int)sizeof(float);
        return RGY_ERR_NONE;
    }

    // chroma処理で小さい平面へ移っても縮小再確保せず、最大容量を保持する。
    const int allocWidth = sameLayout ? std::max(width, m_scratchW) : width;
    const int allocHeight = sameLayout ? std::max(height, m_scratchH) : height;
    const int ref_count_x = (allocWidth  + block_step - 1) / block_step;
    const int ref_count_y = (allocHeight + block_step - 1) / block_step;
    const size_t ref_count = (size_t)ref_count_x * (size_t)ref_count_y;

    // similar_coords: [ref_count][group_size][2 shorts]
    const size_t coord_bytes = ref_count * (size_t)group_size * 2 * sizeof(int16_t);
    // block_counts: [ref_count] uchars
    const size_t count_bytes = ref_count * sizeof(uint8_t);
    // accumulator + weight_map + basic_estimate: float32 per pixel.
    const int allocPitch = allocWidth * (int)sizeof(float);
    const size_t acc_bytes   = (size_t)allocPitch * (size_t)allocHeight;
    const size_t wmap_bytes  = (size_t)allocPitch * (size_t)allocHeight;
    const size_t basic_bytes = (size_t)allocPitch * (size_t)allocHeight;

    m_bufSimilarCoords = m_cl->createBuffer(coord_bytes);
    m_bufBlockCounts   = m_cl->createBuffer(count_bytes);
    m_bufAccumulator   = m_cl->createBuffer(acc_bytes);
    m_bufWeightMap     = m_cl->createBuffer(wmap_bytes);
    m_bufBasicEstimate = m_cl->createBuffer(basic_bytes);
    if (!m_bufSimilarCoords || !m_bufBlockCounts || !m_bufAccumulator || !m_bufWeightMap || !m_bufBasicEstimate) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate scratch buffers.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }

    // Temporal-only buffer: similar_frame_idx is one uchar per match.
    // Sized to the temporal cap (16), independent of the spatial group_size
    // (which can go up to 32 for Wiener).
    if (m_ringRadius > 0) {
        const size_t fidx_bytes = ref_count * 16 * sizeof(uint8_t);
        m_bufSimilarFrameIdx = m_cl->createBuffer(fidx_bytes);
        if (!m_bufSimilarFrameIdx) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate similar_frame_idx buffer.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    } else {
        m_bufSimilarFrameIdx.reset();
    }

    m_scratchW = allocWidth;
    m_scratchH = allocHeight;
    m_scratchBlockStep = block_step;
    m_scratchGroupSize = group_size;
    m_accPitch = width * (int)sizeof(float);
    m_wmapPitch = width * (int)sizeof(float);
    m_basicPitch = width * (int)sizeof(float);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseBm3d::procPlaneTemporal(int planeIdx, RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseBm3d>(m_param);
    if (!prm) return RGY_ERR_INVALID_PARAM;
    const int W = pInputPlane->width;
    const int H = pInputPlane->height;

    auto sts = ensureScratch(W, H);
    if (sts != RGY_ERR_NONE) return sts;
    if (!m_bufSimilarFrameIdx) {
        AddMessage(RGY_LOG_ERROR, _T("similar_frame_idx buffer not allocated; cannot run temporal path.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    if (!m_pastNoisyRing[planeIdx] || !m_pastBasicRing[planeIdx]) {
        AddMessage(RGY_LOG_ERROR, _T("temporal ring not populated for plane %d.\n"), planeIdx);
        return RGY_ERR_INVALID_PARAM;
    }

    const int block_step = m_scratchBlockStep;
    // Temporal mode caps group_size at MAX_GROUP_SIZE_TEMPORAL (16).
    const int group_size = std::min(prm->bm3d.group_size, 16);
    const int bm_range   = prm->bm3d.bm_range;
    const int ref_count_x = (W + block_step - 1) / block_step;
    const int ref_count_y = (H + block_step - 1) / block_step;

    const int bit_depth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const float sigma_scaled = prm->bm3d.sigma * (float)(1 << (bit_depth - 8));
    const float tau_1d = 2.7f * sigma_scaled;
    const float pixel_scale_sq = (float)(1 << (bit_depth - 8));
    const int dist_threshold = (int)(2500.0f * pixel_scale_sq * pixel_scale_sq);

    const int noisy_pitch = m_ringNoisyPitch[planeIdx];
    const int basic_pitch = m_ringBasicPitch[planeIdx];
    const int noisy_slot_stride = noisy_pitch * H;
    const int basic_slot_stride = basic_pitch * H;
    const int ring_cursor = m_ringSlotCursor;
    const int ring_radius = m_ringRadius;

    auto zeroFloatBuffer = [&](RGYCLBuf *buf, size_t bytes, const TCHAR *name) -> RGY_ERR {
        const float zero = 0.0f;
        const cl_int clerr = clEnqueueFillBuffer(queue.get(), buf->mem(),
                                                  &zero, sizeof(float),
                                                  0, bytes,
                                                  0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("failed to zero %s: %d.\n"), name, clerr);
            return err_cl_to_rgy(clerr);
        }
        return RGY_ERR_NONE;
    };

    // === Step 1: Basic estimate (temporal hard-threshold). ===

    sts = zeroFloatBuffer(m_bufAccumulator.get(), (size_t)m_accPitch  * (size_t)H, _T("accumulator (basic t)"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = zeroFloatBuffer(m_bufWeightMap.get(),   (size_t)m_wmapPitch * (size_t)H, _T("weight_map (basic t)"));
    if (sts != RGY_ERR_NONE) return sts;

    // 1.1 Temporal block-match (noisy current + noisy ring).
    {
        RGYWorkSize local(8, 8);
        RGYWorkSize global(ALIGN(ref_count_x, 8), ALIGN(ref_count_y, 8));
        auto err = m_bm3d.get()->kernel("kernel_bm3d_match").config(queue, local, global, wait_events).launch(
            (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
            m_pastNoisyRing[planeIdx]->mem(), noisy_pitch, noisy_slot_stride,
            ring_cursor, ring_radius, m_ringFilled,
            W, H,
            ref_count_x, ref_count_y,
            m_bufSimilarCoords->mem(),
            m_bufSimilarFrameIdx->mem(),
            m_bufBlockCounts->mem(),
            block_step, bm_range, group_size, dist_threshold);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_bm3d_match: %s.\n"), get_err_mes(err));
            return err;
        }
    }

    // 1.2 Temporal basic collaborative filter.
    {
        RGYWorkSize local(8, 8);
        RGYWorkSize global(ALIGN(ref_count_x, 8), ALIGN(ref_count_y, 8));
        auto err = m_bm3d.get()->kernel("kernel_bm3d_basic").config(queue, local, global).launch(
            (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
            m_pastNoisyRing[planeIdx]->mem(), noisy_pitch, noisy_slot_stride,
            ring_cursor, ring_radius,
            W, H,
            ref_count_x, ref_count_y,
            m_bufSimilarCoords->mem(),
            m_bufSimilarFrameIdx->mem(),
            m_bufBlockCounts->mem(),
            m_bufAccumulator->mem(), m_accPitch,
            m_bufWeightMap->mem(),   m_wmapPitch,
            block_step, group_size,
            sigma_scaled, tau_1d);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_bm3d_basic: %s.\n"), get_err_mes(err));
            return err;
        }
    }

    // 1.3 Normalize -> basic estimate (float). Same kernel as spatial.
    {
        RGYWorkSize local(32, 8);
        RGYWorkSize global(ALIGN(W, 32), ALIGN(H, 8));
        auto err = m_bm3d.get()->kernel("kernel_bm3d_normalize_f32").config(queue, local, global).launch(
            m_bufBasicEstimate->mem(), m_basicPitch,
            (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
            W, H,
            m_bufAccumulator->mem(), m_accPitch,
            m_bufWeightMap->mem(),   m_wmapPitch);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_bm3d_normalize_f32 (t): %s.\n"), get_err_mes(err));
            return err;
        }
    }

    // === Step 2: Final estimate (temporal Wiener). ===

    sts = zeroFloatBuffer(m_bufAccumulator.get(), (size_t)m_accPitch  * (size_t)H, _T("accumulator (wiener t)"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = zeroFloatBuffer(m_bufWeightMap.get(),   (size_t)m_wmapPitch * (size_t)H, _T("weight_map (wiener t)"));
    if (sts != RGY_ERR_NONE) return sts;

    const float dist_threshold_basic = (float)dist_threshold / 6.25f;

    // 2.1 Temporal block-match using basic estimate as reference clip.
    {
        RGYWorkSize local(8, 8);
        RGYWorkSize global(ALIGN(ref_count_x, 8), ALIGN(ref_count_y, 8));
        auto err = m_bm3d.get()->kernel("kernel_bm3d_match_basic").config(queue, local, global).launch(
            m_bufBasicEstimate->mem(), m_basicPitch,
            m_pastBasicRing[planeIdx]->mem(), basic_pitch, basic_slot_stride,
            ring_cursor, ring_radius, m_ringFilled,
            W, H,
            ref_count_x, ref_count_y,
            m_bufSimilarCoords->mem(),
            m_bufSimilarFrameIdx->mem(),
            m_bufBlockCounts->mem(),
            block_step, bm_range, group_size, dist_threshold_basic);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_bm3d_match_basic: %s.\n"), get_err_mes(err));
            return err;
        }
    }

    // 2.2 Temporal Wiener collaborative filter.
    {
        RGYWorkSize local(8, 8);
        RGYWorkSize global(ALIGN(ref_count_x, 8), ALIGN(ref_count_y, 8));
        auto err = m_bm3d.get()->kernel("kernel_bm3d_wiener").config(queue, local, global).launch(
            (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
            m_pastNoisyRing[planeIdx]->mem(), noisy_pitch, noisy_slot_stride,
            m_bufBasicEstimate->mem(), m_basicPitch,
            m_pastBasicRing[planeIdx]->mem(), basic_pitch, basic_slot_stride,
            ring_cursor, ring_radius,
            W, H,
            ref_count_x, ref_count_y,
            m_bufSimilarCoords->mem(),
            m_bufSimilarFrameIdx->mem(),
            m_bufBlockCounts->mem(),
            m_bufAccumulator->mem(), m_accPitch,
            m_bufWeightMap->mem(),   m_wmapPitch,
            block_step, group_size,
            sigma_scaled);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_bm3d_wiener: %s.\n"), get_err_mes(err));
            return err;
        }
    }

    // 2.3 Normalize accumulator -> output plane. Same kernel as spatial.
    {
        const float pixel_max = (float)((1 << bit_depth) - 1);
        RGYWorkSize local(32, 8);
        RGYWorkSize global(ALIGN(W, 32), ALIGN(H, 8));
        auto err = m_bm3d.get()->kernel("kernel_bm3d_normalize").config(queue, local, global, {}, event).launch(
            (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
            (cl_mem)pInputPlane->ptr[0],  pInputPlane->pitch[0],
            W, H,
            m_bufAccumulator->mem(), m_accPitch,
            m_bufWeightMap->mem(),   m_wmapPitch,
            pixel_max);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_bm3d_normalize (t): %s.\n"), get_err_mes(err));
            return err;
        }
    }

    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseBm3d::procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseBm3d>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int W = pInputPlane->width;
    const int H = pInputPlane->height;

    auto sts = ensureScratch(W, H);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    const int block_step = m_scratchBlockStep;
    const int group_size = m_scratchGroupSize;
    const int bm_range   = prm->bm3d.bm_range;
    const int ref_count_x = (W + block_step - 1) / block_step;
    const int ref_count_y = (H + block_step - 1) / block_step;

    // Sigma scaling: user input is in 0..255 equivalent units; for 10-bit
    // and above, scale up to the actual pixel domain so distance metrics
    // and the hard-threshold land where the user expects.
    const int bit_depth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const float sigma_scaled = prm->bm3d.sigma * (float)(1 << (bit_depth - 8));
    const float tau_1d = 2.7f * sigma_scaled;
    // Distance threshold for block matching: same convention as the
    // reference port -- (3 * block_size^2 * tuning) scaled to the pixel
    // domain. 2500 is the SSD-per-pixel-pair admission limit on the
    // 0..255 scale, scaled by bit-depth^2.
    const float pixel_scale_sq = (float)(1 << (bit_depth - 8));
    const int dist_threshold = (int)(2500.0f * pixel_scale_sq * pixel_scale_sq);

    // Helper to zero a float buffer.
    auto zeroFloatBuffer = [&](RGYCLBuf *buf, size_t bytes, const TCHAR *name) -> RGY_ERR {
        const float zero = 0.0f;
        const cl_int clerr = clEnqueueFillBuffer(queue.get(), buf->mem(),
                                                  &zero, sizeof(float),
                                                  0, bytes,
                                                  0, nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("failed to zero %s: %d.\n"), name, clerr);
            return err_cl_to_rgy(clerr);
        }
        return RGY_ERR_NONE;
    };

    // === Step 1: Basic estimate (hard-threshold collaborative filter). ===

    sts = zeroFloatBuffer(m_bufAccumulator.get(), (size_t)m_accPitch  * (size_t)H, _T("accumulator (basic)"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = zeroFloatBuffer(m_bufWeightMap.get(),   (size_t)m_wmapPitch * (size_t)H, _T("weight_map (basic)"));
    if (sts != RGY_ERR_NONE) return sts;

    // Step 1.1: ノイズを含む入力に対するブロックマッチング。
    {
        RGYWorkSize local(8, 8);
        RGYWorkSize global(ALIGN(ref_count_x, 8), ALIGN(ref_count_y, 8));
        auto err = m_bm3d.get()->kernel("kernel_bm3d_match").config(queue, local, global, wait_events).launch(
            (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0], W, H,
            ref_count_x, ref_count_y,
            m_bufSimilarCoords->mem(),
            m_bufBlockCounts->mem(),
            block_step, bm_range, group_size, dist_threshold);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_bm3d_match: %s.\n"), get_err_mes(err));
            return err;
        }
    }

    // Step 1.2: hard-threshold collaborative filter + scatter.
    {
        const int group_basic = std::min(group_size, BM3D_MAX_GROUP_SIZE_BASIC);
        RGYWorkSize local(8, 8);
        RGYWorkSize global(ALIGN(ref_count_x, 8), ALIGN(ref_count_y, 8));
        auto err = m_bm3d.get()->kernel("kernel_bm3d_basic").config(queue, local, global).launch(
            (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0], W, H,
            ref_count_x, ref_count_y,
            m_bufSimilarCoords->mem(),
            m_bufBlockCounts->mem(),
            m_bufAccumulator->mem(), m_accPitch,
            m_bufWeightMap->mem(),   m_wmapPitch,
            block_step, group_basic,
            sigma_scaled, tau_1d);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_bm3d_basic: %s.\n"), get_err_mes(err));
            return err;
        }
    }

    // Step 1.3: normalize -> basic estimate (float buffer).
    {
        RGYWorkSize local(32, 8);
        RGYWorkSize global(ALIGN(W, 32), ALIGN(H, 8));
        auto err = m_bm3d.get()->kernel("kernel_bm3d_normalize_f32").config(queue, local, global).launch(
            m_bufBasicEstimate->mem(), m_basicPitch,
            (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
            W, H,
            m_bufAccumulator->mem(), m_accPitch,
            m_bufWeightMap->mem(),   m_wmapPitch);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_bm3d_normalize_f32: %s.\n"), get_err_mes(err));
            return err;
        }
    }

    // === Step 2: Final estimate (Wiener). ===

    sts = zeroFloatBuffer(m_bufAccumulator.get(), (size_t)m_accPitch  * (size_t)H, _T("accumulator (final)"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = zeroFloatBuffer(m_bufWeightMap.get(),   (size_t)m_wmapPitch * (size_t)H, _T("weight_map (final)"));
    if (sts != RGY_ERR_NONE) return sts;

    // Step 2.1: re-match using the basic estimate as the reference clip.
    // The basic-estimate distance threshold is tighter (the basic is
    // less noisy, so candidate blocks can be required to match more
    // closely). Sampas-port convention: threshold_2 = threshold_1 / ~6.
    const float dist_threshold_basic = (float)dist_threshold / 6.25f;
    {
        RGYWorkSize local(8, 8);
        RGYWorkSize global(ALIGN(ref_count_x, 8), ALIGN(ref_count_y, 8));
        auto err = m_bm3d.get()->kernel("kernel_bm3d_match_basic").config(queue, local, global).launch(
            m_bufBasicEstimate->mem(), m_basicPitch,
            W, H,
            ref_count_x, ref_count_y,
            m_bufSimilarCoords->mem(),
            m_bufBlockCounts->mem(),
            block_step, bm_range, group_size, dist_threshold_basic);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_bm3d_match_basic: %s.\n"), get_err_mes(err));
            return err;
        }
    }

    // Step 2.2: Wiener collaborative filter + scatter.
    {
        RGYWorkSize local(8, 8);
        RGYWorkSize global(ALIGN(ref_count_x, 8), ALIGN(ref_count_y, 8));
        auto err = m_bm3d.get()->kernel("kernel_bm3d_wiener").config(queue, local, global).launch(
            (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
            m_bufBasicEstimate->mem(),   m_basicPitch,
            W, H,
            ref_count_x, ref_count_y,
            m_bufSimilarCoords->mem(),
            m_bufBlockCounts->mem(),
            m_bufAccumulator->mem(), m_accPitch,
            m_bufWeightMap->mem(),   m_wmapPitch,
            block_step, group_size,
            sigma_scaled);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_bm3d_wiener: %s.\n"), get_err_mes(err));
            return err;
        }
    }

    // Step 2.3: normalize accumulator -> output plane.
    {
        const float pixel_max = (float)((1 << bit_depth) - 1);
        RGYWorkSize local(32, 8);
        RGYWorkSize global(ALIGN(W, 32), ALIGN(H, 8));
        auto err = m_bm3d.get()->kernel("kernel_bm3d_normalize").config(queue, local, global, {}, event).launch(
            (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
            (cl_mem)pInputPlane->ptr[0],  pInputPlane->pitch[0],
            W, H,
            m_bufAccumulator->mem(), m_accPitch,
            m_bufWeightMap->mem(),   m_wmapPitch,
            pixel_max);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_bm3d_normalize: %s.\n"), get_err_mes(err));
            return err;
        }
    }

    return RGY_ERR_NONE;
}

RGYFilterDenoiseBm3d::RGYFilterDenoiseBm3d(shared_ptr<RGYOpenCLContext> context) :
    RGYFilter(context), m_bm3d(),
    m_bufSimilarCoords(), m_bufBlockCounts(),
    m_bufAccumulator(), m_bufWeightMap(), m_bufBasicEstimate(),
    m_pastNoisyRing(), m_pastBasicRing(),
    m_ringW({0, 0, 0}), m_ringH({0, 0, 0}),
    m_ringNoisyPitch({0, 0, 0}), m_ringBasicPitch({0, 0, 0}),
    m_ringRadius(0), m_ringSlotCursor(0), m_ringFilled(0),
    m_scratchW(0), m_scratchH(0),
    m_scratchBlockStep(0), m_scratchGroupSize(0),
    m_accPitch(0), m_wmapPitch(0), m_basicPitch(0) {
    m_name = _T("bm3d");
}

RGY_ERR RGYFilterDenoiseBm3d::ensureRingBuffers(int planeIdx, int width, int height) {
    if (planeIdx < 0 || planeIdx >= 3) return RGY_ERR_INVALID_PARAM;
    if (m_ringRadius <= 0) return RGY_ERR_NONE;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseBm3d>(m_param);
    if (!prm) return RGY_ERR_INVALID_PARAM;

    const int bit_depth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int bytes_per_sample = (bit_depth > 8) ? 2 : 1;
    const int noisy_pitch = width * bytes_per_sample;
    const int basic_pitch = width * (int)sizeof(float);

    if (m_pastNoisyRing[planeIdx] && m_pastBasicRing[planeIdx]
        && m_ringW[planeIdx] == width && m_ringH[planeIdx] == height
        && m_ringNoisyPitch[planeIdx] == noisy_pitch
        && m_ringBasicPitch[planeIdx] == basic_pitch) {
        return RGY_ERR_NONE;
    }

    const size_t noisy_slot_bytes = (size_t)noisy_pitch * (size_t)height;
    const size_t basic_slot_bytes = (size_t)basic_pitch * (size_t)height;
    const size_t noisy_total = noisy_slot_bytes * (size_t)m_ringRadius;
    const size_t basic_total = basic_slot_bytes * (size_t)m_ringRadius;

    m_pastNoisyRing[planeIdx] = m_cl->createBuffer(noisy_total);
    m_pastBasicRing[planeIdx] = m_cl->createBuffer(basic_total);
    if (!m_pastNoisyRing[planeIdx] || !m_pastBasicRing[planeIdx]) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate temporal ring buffers (plane %d).\n"), planeIdx);
        return RGY_ERR_MEMORY_ALLOC;
    }
    m_ringW[planeIdx] = width;
    m_ringH[planeIdx] = height;
    m_ringNoisyPitch[planeIdx] = noisy_pitch;
    m_ringBasicPitch[planeIdx] = basic_pitch;
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseBm3d::pushNoisyToRing(int planeIdx, const RGYFrameInfo *pInputPlane, RGYOpenCLQueue &queue) {
    if (m_ringRadius <= 0 || !m_pastNoisyRing[planeIdx]) return RGY_ERR_NONE;
    const int W = pInputPlane->width;
    const int H = pInputPlane->height;
    const int slot = m_ringSlotCursor;
    const size_t slot_offset = (size_t)slot * (size_t)m_ringNoisyPitch[planeIdx] * (size_t)H;
    // Tight-pack copy from the (potentially padded) input plane into the
    // ring slot. Use clEnqueueCopyBufferRect for source-stride handling.
    const size_t src_row_pitch = (size_t)pInputPlane->pitch[0];
    const size_t dst_row_pitch = (size_t)m_ringNoisyPitch[planeIdx];
    const size_t row_bytes     = (size_t)W * (size_t)((RGY_CSP_BIT_DEPTH[m_param->frameOut.csp] > 8) ? 2 : 1);
    const size_t src_origin[3] = {0, 0, 0};
    const size_t dst_origin[3] = {slot_offset, 0, 0};
    const size_t region[3]     = {row_bytes, (size_t)H, 1};
    const cl_int clerr = clEnqueueCopyBufferRect(queue.get(),
        (cl_mem)pInputPlane->ptr[0], m_pastNoisyRing[planeIdx]->mem(),
        src_origin, dst_origin, region,
        src_row_pitch, 0, dst_row_pitch, 0,
        0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("failed to push noisy plane %d into ring slot %d: %d.\n"), planeIdx, slot, clerr);
        return err_cl_to_rgy(clerr);
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDenoiseBm3d::pushBasicToRing(int planeIdx, RGYOpenCLQueue &queue) {
    if (m_ringRadius <= 0 || !m_pastBasicRing[planeIdx] || !m_bufBasicEstimate) return RGY_ERR_NONE;
    const int H = m_ringH[planeIdx];
    const int slot = m_ringSlotCursor;
    const size_t slot_offset = (size_t)slot * (size_t)m_ringBasicPitch[planeIdx] * (size_t)H;
    const size_t bytes = (size_t)m_ringBasicPitch[planeIdx] * (size_t)H;
    const cl_int clerr = clEnqueueCopyBuffer(queue.get(),
        m_bufBasicEstimate->mem(), m_pastBasicRing[planeIdx]->mem(),
        0, slot_offset, bytes,
        0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("failed to push basic plane %d into ring slot %d: %d.\n"), planeIdx, slot, clerr);
        return err_cl_to_rgy(clerr);
    }
    return RGY_ERR_NONE;
}

RGYFilterDenoiseBm3d::~RGYFilterDenoiseBm3d() {
    close();
}

RGY_ERR RGYFilterDenoiseBm3d::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseBm3d>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int planeCount = RGY_CSP_PLANES[prm->frameOut.csp];
    if ((RGY_CSP_DATA_TYPE[prm->frameOut.csp] != RGY_DATA_TYPE_U8
            && RGY_CSP_DATA_TYPE[prm->frameOut.csp] != RGY_DATA_TYPE_U16)
        || bitDepth > 12 || (planeCount != 1 && planeCount != 3)
        || rgy_chromafmt_is_rgb(RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp])) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp for bm3d: %s (planar YUV up to 12-bit is required).\n"),
            RGY_CSP_NAMES[prm->frameOut.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (!std::isfinite(prm->bm3d.sigma)) {
        AddMessage(RGY_LOG_ERROR, _T("sigma must be finite.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->bm3d.sigma != 0.0f && (prm->bm3d.sigma < 0.5f || 100.0f < prm->bm3d.sigma)) {
        prm->bm3d.sigma = clamp(prm->bm3d.sigma, 0.5f, 100.0f);
        AddMessage(RGY_LOG_WARN, _T("sigma should be in range of %.2f - %.2f.\n"), 0.5f, 100.0f);
    }
    if (prm->bm3d.block_step < 1 || BM3D_BLOCK_SIZE < prm->bm3d.block_step) {
        prm->bm3d.block_step = clamp(prm->bm3d.block_step, 1, BM3D_BLOCK_SIZE);
        AddMessage(RGY_LOG_WARN, _T("block_step should be in range of %d - %d.\n"), 1, BM3D_BLOCK_SIZE);
    }
    // group_size cap is the Wiener-step ceiling; the basic step
    // internally caps at BM3D_MAX_GROUP_SIZE_BASIC.
    if (prm->bm3d.group_size < 1 || BM3D_MAX_GROUP_SIZE_WIENER < prm->bm3d.group_size) {
        prm->bm3d.group_size = clamp(prm->bm3d.group_size, 1, BM3D_MAX_GROUP_SIZE_WIENER);
        AddMessage(RGY_LOG_WARN, _T("group_size should be in range of %d - %d.\n"), 1, BM3D_MAX_GROUP_SIZE_WIENER);
    }
    if (prm->bm3d.bm_range < 1 || 32 < prm->bm3d.bm_range) {
        prm->bm3d.bm_range = clamp(prm->bm3d.bm_range, 1, 32);
        AddMessage(RGY_LOG_WARN, _T("bm_range should be in range of %d - %d.\n"), 1, 32);
    }
    // V-BM3D temporal radius cap. Per-WI noise + basic stacks are
    // capped at 16 entries in the temporal kernels to stay within the
    // spatial-basic safe envelope (8 KB total per WI). Radius cap 4 =
    // 4 past frames searched + current = 5-frame temporal window.
    if (prm->bm3d.radius < 0 || 4 < prm->bm3d.radius) {
        prm->bm3d.radius = clamp(prm->bm3d.radius, 0, 4);
        AddMessage(RGY_LOG_WARN, _T("radius should be in range of %d - %d.\n"), 0, 4);
    }
    if (prm->bm3d.radius > 0 && prm->bm3d.group_size > 16) {
        AddMessage(RGY_LOG_WARN, _T("V-BM3D temporal mode caps group_size at 16; clamping from %d.\n"), prm->bm3d.group_size);
        prm->bm3d.group_size = 16;
    }
    // Reset ring state when radius changes; release ring buffers that may
    // be stale-sized from a previous parameter set.
    auto prmRingPrev = std::dynamic_pointer_cast<RGYFilterParamDenoiseBm3d>(m_param);
    if (!prmRingPrev
        || prmRingPrev->bm3d.radius != prm->bm3d.radius
        || prmRingPrev->bm3d.chroma != prm->bm3d.chroma
        || cmpFrameInfoCspResolution(&prmRingPrev->frameOut, &prm->frameOut)) {
        for (int p = 0; p < 3; p++) {
            m_pastNoisyRing[p].reset();
            m_pastBasicRing[p].reset();
            m_ringW[p] = 0; m_ringH[p] = 0;
            m_ringNoisyPitch[p] = 0; m_ringBasicPitch[p] = 0;
        }
        m_ringSlotCursor = 0;
        m_ringFilled = 0;
    }
    m_ringRadius = prm->bm3d.radius;

    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamDenoiseBm3d>(m_param);
    const bool temporal_build = prm->bm3d.radius > 0;
    if (!m_bm3d.get()
        || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameOut.csp] != RGY_CSP_BIT_DEPTH[pParam->frameOut.csp]
        || (prmPrev->bm3d.radius > 0) != temporal_build) {
        std::string options = strsprintf("-D Type=%s -D bit_depth=%d -D TEMPORAL=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp], temporal_build ? 1 : 0);
        m_bm3d.set(m_cl->buildResourceAsync(_T("RGY_FILTER_DENOISE_BM3D_CL"), _T("EXE_DATA"), options.c_str()));
    }

    // Invalidate scratch when params that affect buffer sizing change.
    // Radius also matters: the m_bufSimilarFrameIdx buffer is only
    // allocated in temporal mode, and ensureScratch's early-return path
    // doesn't re-check radius transitions.
    if (prmPrev
        && (prmPrev->bm3d.block_step != prm->bm3d.block_step
         || prmPrev->bm3d.group_size != prm->bm3d.group_size
         || prmPrev->bm3d.radius     != prm->bm3d.radius)) {
        m_bufSimilarCoords.reset();
        m_bufBlockCounts.reset();
        m_bufAccumulator.reset();
        m_bufWeightMap.reset();
        m_bufBasicEstimate.reset();
        m_bufSimilarFrameIdx.reset();
        m_scratchW = 0;
        m_scratchH = 0;
    }

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterDenoiseBm3d::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[0].get();
        ppOutputFrames[0] = &pOutFrame->frame;
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != RGYCLMemcpyD2D) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    auto prm = std::dynamic_pointer_cast<RGYFilterParamDenoiseBm3d>(m_param);
    const int numPlanes = RGY_CSP_PLANES[ppOutputFrames[0]->csp];
    if (prm->bm3d.sigma == 0.0f) {
        for (int i = 0; i < numPlanes; i++) {
            auto planeDst = getPlane(ppOutputFrames[0], (RGY_PLANE)i);
            auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
            sts = m_cl->copyPlane(&planeDst, &planeSrc, nullptr, queue,
                (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>(),
                (i == numPlanes - 1) ? event : nullptr);
            if (sts != RGY_ERR_NONE) return sts;
        }
        return RGY_ERR_NONE;
    }
    if (!m_bm3d.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to build RGY_FILTER_DENOISE_BM3D_CL(m_bm3d)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    // Decide spatial vs temporal once per frame; chroma re-runs through
    // the same selector so all planes use a consistent path.
    const bool useTemporal = m_ringRadius > 0;
    for (int i = 0; i < numPlanes; i++) {
        auto planeDst = getPlane(ppOutputFrames[0], (RGY_PLANE)i);
        auto planeSrc = getPlane(pInputFrame,       (RGY_PLANE)i);
        const bool runFilter = (i == 0) || prm->bm3d.chroma;
        if (runFilter) {
            // Make sure ring buffers exist before procPlaneTemporal needs them.
            if (m_ringRadius > 0) {
                sts = ensureRingBuffers(i, planeSrc.width, planeSrc.height);
                if (sts != RGY_ERR_NONE) return sts;
            }
            if (useTemporal) {
                sts = procPlaneTemporal(i, &planeDst, &planeSrc, queue,
                                        (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>(),
                                        (m_ringRadius == 0 && i == numPlanes - 1) ? event : nullptr);
            } else {
                sts = procPlane(&planeDst, &planeSrc, queue,
                                (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>(),
                                (m_ringRadius == 0 && i == numPlanes - 1) ? event : nullptr);
            }
            if (sts != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at bm3dFrame (%s): %s.\n"),
                    RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
                return sts;
            }
            // After successful processing, push the just-seen NOISY plane
            // and the just-computed BASIC estimate into the ring so the
            // next frame's procPlane can search across past history. Both
            // the spatial path (warm-up frames) and the temporal path
            // populate the ring identically - the basic estimate is the
            // current frame's Step-1 output in either case.
            if (m_ringRadius > 0) {
                sts = pushNoisyToRing(i, &planeSrc, queue);
                if (sts != RGY_ERR_NONE) return sts;
                sts = pushBasicToRing(i, queue);
                if (sts != RGY_ERR_NONE) return sts;
            }
        } else {
            sts = m_cl->copyPlane(&planeDst, &planeSrc, nullptr, queue, {},
                (m_ringRadius == 0 && i == numPlanes - 1) ? event : nullptr);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
    }
    // Advance the ring cursor ONCE per frame (after all planes are pushed).
    // Cursor wraps modulo radius; fill counter saturates at radius.
    if (m_ringRadius > 0) {
        m_ringSlotCursor = (m_ringSlotCursor + 1) % m_ringRadius;
        if (m_ringFilled < m_ringRadius) m_ringFilled++;
        if (event) {
            sts = queue.getmarker(*event);
        }
    }
    return sts;
}

void RGYFilterDenoiseBm3d::close() {
    m_frameBuf.clear();
    m_bm3d.clear();
    m_bufSimilarCoords.reset();
    m_bufBlockCounts.reset();
    m_bufAccumulator.reset();
    m_bufWeightMap.reset();
    m_bufBasicEstimate.reset();
    m_bufSimilarFrameIdx.reset();
    for (int p = 0; p < 3; p++) {
        m_pastNoisyRing[p].reset();
        m_pastBasicRing[p].reset();
        m_ringW[p] = 0; m_ringH[p] = 0;
        m_ringNoisyPitch[p] = 0; m_ringBasicPitch[p] = 0;
    }
    m_ringRadius = 0;
    m_ringSlotCursor = 0;
    m_ringFilled = 0;
    m_cl.reset();
}
