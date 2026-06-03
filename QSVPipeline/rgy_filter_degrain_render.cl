// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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

static inline int degrain_render_const_block_size(const int blockSize) {
#if defined(DEGRAIN_BLOCK_SIZE)
    return DEGRAIN_BLOCK_SIZE;
#else
    return blockSize;
#endif
}

static inline int degrain_render_const_overlap(const int overlap) {
#if defined(DEGRAIN_OVERLAP)
    return DEGRAIN_OVERLAP;
#else
    return overlap;
#endif
}

static inline int degrain_render_const_step(const int step) {
#if defined(DEGRAIN_STEP)
    return DEGRAIN_STEP;
#else
    return step;
#endif
}

static inline int degrain_render_const_blocks_x(const int blocksX) {
#if defined(DEGRAIN_BLOCKS_X)
    return DEGRAIN_BLOCKS_X;
#else
    return blocksX;
#endif
}

static inline int degrain_render_const_blocks_y(const int blocksY) {
#if defined(DEGRAIN_BLOCKS_Y)
    return DEGRAIN_BLOCKS_Y;
#else
    return blocksY;
#endif
}

static inline int degrain_render_scale_covered(const int covered, const int scale) {
    const int rshift = degrain_plane_scale_rshift(scale);
    return (covered + ((1 << rshift) - 1)) >> rshift;
}

static inline int degrain_render_scale_floor(const int value, const int scale) {
    return value >> degrain_plane_scale_rshift(scale);
}

static inline int degrain_render_const_covered_width(const int coveredWidth, const int scaleX) {
#if defined(DEGRAIN_COVERED_WIDTH)
    return degrain_render_scale_covered(DEGRAIN_COVERED_WIDTH, scaleX);
#else
    return coveredWidth;
#endif
}

static inline int degrain_render_const_covered_height(const int coveredHeight, const int scaleY) {
#if defined(DEGRAIN_COVERED_HEIGHT)
    return degrain_render_scale_covered(DEGRAIN_COVERED_HEIGHT, scaleY);
#else
    return coveredHeight;
#endif
}

__kernel void kernel_degrain_build_temporal_mix_plan(
    __global float *temporalMixPlan,
    __global const degrain_mv_t *mv,
    __global const degrain_sad_t *sad,
    __global const float *temporalMixPrior,
    const int blockCount,
    const uint thsad,
    __global const uint *disableMaskPtr) {
    const int block = (int)get_global_id(0);
    if (block >= blockCount) {
        return;
    }
    const uint disableMask = disableMaskPtr[0];

    const float sourceConfidenceRaw = degrain_temporal_mix_prior_center(temporalMixPrior);
    float referenceConfidenceRaw[DEGRAIN_REFS];
    float confidenceTotal = sourceConfidenceRaw;
    for (int referenceDirection = 0; referenceDirection < DEGRAIN_REFS; referenceDirection++) {
        const float temporalMixPriorRef = degrain_temporal_mix_prior_ref(temporalMixPrior, referenceDirection);
        referenceConfidenceRaw[referenceDirection] = degrain_reference_mix_affinity(mv, sad, block, referenceDirection, thsad, degrain_ref_direction_disabled(disableMask, referenceDirection)) * temporalMixPriorRef;
        confidenceTotal += referenceConfidenceRaw[referenceDirection];
    }

    float referenceMixTotal = 0.0f;
    const float invTotal = (confidenceTotal > 0.0f) ? (1.0f / confidenceTotal) : 0.0f;
    const int planOffset = degrain_temporal_mix_plan_offset(block);
    for (int referenceDirection = 0; referenceDirection < DEGRAIN_REFS; referenceDirection++) {
        float referenceMixNorm = 0.0f;
        if (referenceConfidenceRaw[referenceDirection] > 0.0f) {
            referenceMixNorm = referenceConfidenceRaw[referenceDirection] * invTotal;
            referenceMixTotal += referenceMixNorm;
        }
        temporalMixPlan[planOffset + 1 + referenceDirection] = referenceMixNorm;
    }
    const float sourceMixNorm = max(1.0f - referenceMixTotal, 0.0f);
    temporalMixPlan[planOffset] = sourceMixNorm;
}

__kernel void kernel_degrain_scene_change_count(
    __global uint *sceneChangeCounts,
    __global const degrain_sad_t *sad,
    const int blockCount,
    const int temporalDirections,
    const uint thscd1,
    const uint baseDisableMask) {
    const int idx = (int)get_global_id(0);
    const int total = blockCount * temporalDirections;
    if (idx >= total) {
        return;
    }
    const int refDirection = idx % temporalDirections;
    if (((baseDisableMask >> refDirection) & 1u) != 0u) {
        return;
    }
    if (sad[idx].sad > thscd1) {
        atomic_inc((volatile __global unsigned int *)&sceneChangeCounts[refDirection]);
    }
}

__kernel void kernel_degrain_scene_change_mask(
    __global uint *disableMaskPtr,
    __global const uint *sceneChangeCounts,
    const int temporalDirections,
    const uint baseDisableMask,
    const uint thscd2) {
    uint disableMask = baseDisableMask;
    for (int refDirection = 0; refDirection < temporalDirections; refDirection++) {
        if (sceneChangeCounts[refDirection] > thscd2) {
            disableMask |= (1u << refDirection);
        }
    }
    disableMaskPtr[0] = disableMask;
}

__kernel void kernel_degrain_overlap_plane(
    __global TypePixel *dst,
    const int dst_pitch,
    __global const uchar *cur,
    const int cur_pitch,
    __global const uchar *ref0,
    __global const uchar *refBackward1,
    __global const uchar *refForward1,
    __global const uchar *refBackward2,
    __global const uchar *refForward2,
    __global const uchar *refBackward3,
    __global const uchar *refForward3,
    __global const uchar *refBackward4,
    __global const uchar *refForward4,
    __global const uchar *refBackward5,
    __global const uchar *refForward5,
    const int width,
    const int height,
    __global const degrain_mv_t *mv,
    __global const degrain_sad_t *sad,
    __global const float *temporalMixPrior,
    const int blocksX,
    const int blocksY,
    const int blockSize,
    const int overlap,
    const int step,
    const int coveredWidth,
    const int coveredHeight,
    const int planeScaleX,
    const int planeScaleY,
    const int modeType,
    const int refDirection,
    const uint thsad,
    __global const uint *disableMaskPtr) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const uint disableMask = disableMaskPtr[0];

    const int dstPitch = dst_pitch / (int)sizeof(TypePixel);
    const int fallback = degrain_pixel_load(cur, cur_pitch, width, height, x, y);
    const int scaleX = degrain_plane_scale_x(planeScaleX);
    const int scaleY = degrain_plane_scale_y(planeScaleY);
    const int renderBlockSize = degrain_render_const_block_size(blockSize);
    const int renderOverlap = degrain_render_const_overlap(overlap);
    const int renderStep = degrain_render_const_step(step);
    const int renderBlocksX = degrain_render_const_blocks_x(blocksX);
    const int renderBlocksY = degrain_render_const_blocks_y(blocksY);
    const int renderCoveredWidth = degrain_render_const_covered_width(coveredWidth, scaleX);
    const int renderCoveredHeight = degrain_render_const_covered_height(coveredHeight, scaleY);
    if (!degrain_is_covered_pixel(x, y, renderCoveredWidth, renderCoveredHeight)) {
        dst[y * dstPitch + x] = degrain_clamp_pixel(fallback);
        return;
    }

    const int planeBlockSizeX = max(degrain_render_scale_floor(renderBlockSize, scaleX), 1);
    const int planeBlockSizeY = max(degrain_render_scale_floor(renderBlockSize, scaleY), 1);
    const int planeOverlapX = max(degrain_render_scale_floor(renderOverlap, scaleX), 0);
    const int planeOverlapY = max(degrain_render_scale_floor(renderOverlap, scaleY), 0);
    const int planeStepX = max(degrain_render_scale_floor(renderStep, scaleX), 1);
    const int planeStepY = max(degrain_render_scale_floor(renderStep, scaleY), 1);
    const int primaryBlockX = min(x / planeStepX, renderBlocksX - 1);
    const int primaryBlockY = min(y / planeStepY, renderBlocksY - 1);
    const int usePrevBlockX = planeOverlapX > 0 && primaryBlockX > 0 && x < degrain_block_origin(primaryBlockX, planeStepX) + planeOverlapX;
    const int usePrevBlockY = planeOverlapY > 0 && primaryBlockY > 0 && y < degrain_block_origin(primaryBlockY, planeStepY) + planeOverlapY;
    const int blockXs[2] = { primaryBlockX, primaryBlockX - 1 };
    const int blockYs[2] = { primaryBlockY, primaryBlockY - 1 };
    const int blockCountX = usePrevBlockX ? 2 : 1;
    const int blockCountY = usePrevBlockY ? 2 : 1;

    degrain_window_accum_t sampleSum = degrain_window_accum_zero();
    degrain_window_accum_t weightSum = degrain_window_accum_zero();
    int sampleCount = 0;
    for (int byIndex = 0; byIndex < blockCountY; byIndex++) {
        const int blockY = blockYs[byIndex];
        const int baseY = degrain_block_origin(blockY, planeStepY);
        for (int bxIndex = 0; bxIndex < blockCountX; bxIndex++) {
            const int blockX = blockXs[bxIndex];
            const int baseX = degrain_block_origin(blockX, planeStepX);
            const int localX = x - baseX;
            const int localY = y - baseY;
            if (localX < 0 || localX >= planeBlockSizeX || localY < 0 || localY >= planeBlockSizeY) {
                continue;
            }
            const float windowWeight = degrain_window_factor_rect_2d(
                x, y,
                baseX, baseY,
                planeBlockSizeX, planeBlockSizeY,
                planeOverlapX, planeOverlapY,
                blockX, blockY,
                renderBlocksX, renderBlocksY);

            const int block = blockY * renderBlocksX + blockX;
            int sample = degrain_pixel_load(cur, cur_pitch, width, height, x, y);
            if (modeType == DEGRAIN_MODE_COMPB || modeType == DEGRAIN_MODE_COMPF) {
                if (refDirection < DEGRAIN_REFS
                    && (((modeType == DEGRAIN_MODE_COMPB) && ((refDirection & 1) == 0))
                        || ((modeType == DEGRAIN_MODE_COMPF) && ((refDirection & 1) == 1)))) {
                    __global const uchar *ref = degrain_ref_plane_ptr_same_pitch(
                        refBackward1, refForward1,
                        refBackward2, refForward2,
                        refBackward3, refForward3,
                        refBackward4, refForward4,
                        refBackward5, refForward5,
                        refDirection);
                    sample = degrain_compensate_block_sample(
                        ref0, ref, cur_pitch,
                        width, height,
                        mv, sad,
                        block, refDirection, thsad, degrain_ref_direction_disabled(disableMask, refDirection),
                        planeScaleX, planeScaleY,
                        x, y);
                }
            } else {
                sample = degrain_degrain_block_sample(
                    cur, cur_pitch,
                    refBackward1, refForward1,
                    refBackward2, refForward2,
                    refBackward3, refForward3,
                    refBackward4, refForward4,
                    refBackward5, refForward5,
                    width, height,
                    mv, sad,
                    block, thsad,
                    disableMask,
                    temporalMixPrior,
                    planeScaleX, planeScaleY,
                    x, y);
            }
            degrain_accumulate_windowed_sample(&sampleSum, &weightSum, sample, windowWeight);
            sampleCount++;
        }
    }

    const int result = (sampleCount > 0) ? degrain_finalize_windowed_sample(sampleSum, weightSum, fallback) : fallback;
    dst[y * dstPitch + x] = degrain_clamp_pixel(result);
}

static inline void degrain_compensate_overlap_plane_ramp_generic(
    __global TypePixel *dst,
    const int dst_pitch,
    __global const uchar *cur,
    const int cur_pitch,
    __global const uchar *ref0,
    __global const uchar *ref,
    const int refDirection,
    const int width,
    const int height,
    __global const degrain_mv_t *mv,
    __global const degrain_sad_t *sad,
    const int blocksX,
    const int blocksY,
    const int blockSize,
    const int overlap,
    const int step,
    const int coveredWidth,
    const int coveredHeight,
    const int planeScaleX,
    const int planeScaleY,
    const uint thsad,
    const uint disableMask,
    __global const float *windowRamp,
    const int originX,
    const int originY,
    const int compactTopLeftBorder) {
    const int globalX = (int)get_global_id(0);
    const int globalY = (int)get_global_id(1);
    int x = originX + globalX;
    int y = originY + globalY;
    if (compactTopLeftBorder) {
        if (originX <= 0 || originY <= 0) {
            return;
        }
        const int compactScaleX = degrain_plane_scale_x(planeScaleX);
        const int compactScaleY = degrain_plane_scale_y(planeScaleY);
        const int compactStep = degrain_render_const_step(step);
        const int compactBlocksX = degrain_render_const_blocks_x(blocksX);
        const int compactBlocksY = degrain_render_const_blocks_y(blocksY);
        const int compactPlaneStepX = max(degrain_render_scale_floor(compactStep, compactScaleX), 1);
        const int compactPlaneStepY = max(degrain_render_scale_floor(compactStep, compactScaleY), 1);
        const int interiorEndX = min(width, compactBlocksX * compactPlaneStepX);
        const int interiorEndY = min(height, compactBlocksY * compactPlaneStepY);
        const int lowerHeight = max(height - originY, 0);
        const int rightBorderWidth = max(width - interiorEndX, 0);
        const int bottomBorderWidth = max(interiorEndX - originX, 0);
        const int borderIndex = globalX;
        const int topBorderPixels = width * originY;
        const int leftBorderPixels = originX * lowerHeight;
        const int rightBorderPixels = rightBorderWidth * lowerHeight;
        if (borderIndex < topBorderPixels) {
            x = borderIndex % width;
            y = borderIndex / width;
        } else if (borderIndex < topBorderPixels + leftBorderPixels) {
            const int leftBorderIndex = borderIndex - topBorderPixels;
            x = leftBorderIndex % originX;
            y = originY + leftBorderIndex / originX;
        } else if (borderIndex < topBorderPixels + leftBorderPixels + rightBorderPixels) {
            const int rightBorderIndex = borderIndex - topBorderPixels - leftBorderPixels;
            x = interiorEndX + rightBorderIndex % rightBorderWidth;
            y = originY + rightBorderIndex / rightBorderWidth;
        } else {
            const int bottomBorderIndex = borderIndex - topBorderPixels - leftBorderPixels - rightBorderPixels;
            if (bottomBorderWidth <= 0) {
                return;
            }
            x = originX + bottomBorderIndex % bottomBorderWidth;
            y = interiorEndY + bottomBorderIndex / bottomBorderWidth;
        }
    }
    if (x >= width || y >= height) {
        return;
    }

    const int dstPitch = dst_pitch / (int)sizeof(TypePixel);
    const int fallback = degrain_pixel_load(cur, cur_pitch, width, height, x, y);
    const int scaleX = degrain_plane_scale_x(planeScaleX);
    const int scaleY = degrain_plane_scale_y(planeScaleY);
    const int renderBlockSize = degrain_render_const_block_size(blockSize);
    const int renderOverlap = degrain_render_const_overlap(overlap);
    const int renderStep = degrain_render_const_step(step);
    const int renderBlocksX = degrain_render_const_blocks_x(blocksX);
    const int renderBlocksY = degrain_render_const_blocks_y(blocksY);
    const int renderCoveredWidth = degrain_render_const_covered_width(coveredWidth, scaleX);
    const int renderCoveredHeight = degrain_render_const_covered_height(coveredHeight, scaleY);
    if (!degrain_is_covered_pixel(x, y, renderCoveredWidth, renderCoveredHeight)) {
        dst[y * dstPitch + x] = degrain_clamp_pixel(fallback);
        return;
    }

    const int planeBlockSizeX = max(degrain_render_scale_floor(renderBlockSize, scaleX), 1);
    const int planeBlockSizeY = max(degrain_render_scale_floor(renderBlockSize, scaleY), 1);
    const int planeOverlapX = max(degrain_render_scale_floor(renderOverlap, scaleX), 0);
    const int planeOverlapY = max(degrain_render_scale_floor(renderOverlap, scaleY), 0);
    const int planeStepX = max(degrain_render_scale_floor(renderStep, scaleX), 1);
    const int planeStepY = max(degrain_render_scale_floor(renderStep, scaleY), 1);
    const int primaryBlockX = min(x / planeStepX, renderBlocksX - 1);
    const int primaryBlockY = min(y / planeStepY, renderBlocksY - 1);
    const int primaryBaseX = degrain_block_origin(primaryBlockX, planeStepX);
    const int primaryBaseY = degrain_block_origin(primaryBlockY, planeStepY);
    const int primaryLocalX = x - primaryBaseX;
    const int primaryLocalY = y - primaryBaseY;
    const int primaryBlock = primaryBlockY * renderBlocksX + primaryBlockX;
    const int usePrevBlockX = planeOverlapX > 0 && primaryBlockX > 0 && primaryLocalX < planeOverlapX;
    const int usePrevBlockY = planeOverlapY > 0 && primaryBlockY > 0 && primaryLocalY < planeOverlapY;
    const float wxPrev = usePrevBlockX ? windowRamp[primaryLocalX] : 0.0f;
    const float wyPrev = usePrevBlockY ? windowRamp[planeOverlapX + primaryLocalY] : 0.0f;
    const float wx[2] = { 1.0f - wxPrev, wxPrev };
    const float wy[2] = { 1.0f - wyPrev, wyPrev };

    const int blockXs[2] = { primaryBlockX, primaryBlockX - 1 };
    const int blockYs[2] = { primaryBlockY, primaryBlockY - 1 };
    const int localXs[2] = { primaryLocalX, primaryLocalX + planeStepX };
    const int localYs[2] = { primaryLocalY, primaryLocalY + planeStepY };
    const int blockRows[2] = { primaryBlock, primaryBlock - renderBlocksX };
    const int blockCountX = usePrevBlockX ? 2 : 1;
    const int blockCountY = usePrevBlockY ? 2 : 1;
    const int directionDisabled = degrain_ref_direction_disabled(disableMask, refDirection);

    float sampleSum = 0.0f;
    float weightSum = 0.0f;
    for (int byIndex = 0; byIndex < blockCountY; byIndex++) {
        const int blockY = blockYs[byIndex];
        const int localY = localYs[byIndex];
        const int blockRow = blockRows[byIndex];
        for (int bxIndex = 0; bxIndex < blockCountX; bxIndex++) {
            const int blockX = blockXs[bxIndex];
            const int localX = localXs[bxIndex];
            if (localX < 0 || localX >= planeBlockSizeX || localY < 0 || localY >= planeBlockSizeY
                || blockX < 0 || blockX >= renderBlocksX || blockY < 0 || blockY >= renderBlocksY) {
                continue;
            }
            const int block = blockRow - bxIndex;
            const int sample = degrain_compensate_block_sample(
                ref0, ref, cur_pitch,
                width, height,
                mv, sad,
                block, refDirection, thsad, directionDisabled,
                planeScaleX, planeScaleY,
                x, y);
            degrain_accumulate_weighted_sample_fp32(&sampleSum, &weightSum, sample, wx[bxIndex] * wy[byIndex]);
        }
    }

    const int result = degrain_finalize_weighted_sample_fp32(sampleSum, weightSum, fallback);
    dst[y * dstPitch + x] = degrain_clamp_pixel(result);
}

__kernel void kernel_degrain_compensate_overlap_plane_ramp(
    __global TypePixel *dst,
    const int dst_pitch,
    __global const uchar *cur,
    const int cur_pitch,
    __global const uchar *ref0,
    __global const uchar *ref,
    const int refDirection,
    const int width,
    const int height,
    __global const degrain_mv_t *mv,
    __global const degrain_sad_t *sad,
    const int blocksX,
    const int blocksY,
    const int blockSize,
    const int overlap,
    const int step,
    const int coveredWidth,
    const int coveredHeight,
    const int planeScaleX,
    const int planeScaleY,
    const uint thsad,
    __global const uint *disableMaskPtr,
    __global const float *windowRamp) {
    const uint disableMask = disableMaskPtr[0];
    degrain_compensate_overlap_plane_ramp_generic(
        dst, dst_pitch,
        cur, cur_pitch,
        ref0, ref, refDirection,
        width, height,
        mv, sad,
        blocksX, blocksY,
        blockSize, overlap, step,
        coveredWidth, coveredHeight,
        planeScaleX, planeScaleY,
        thsad, disableMask,
        windowRamp,
        0, 0, 0);
}

__kernel void kernel_degrain_degrain_overlap_plane(
    __global TypePixel *dst,
    const int dst_pitch,
    __global const uchar *cur,
    const int cur_pitch,
    __global const uchar *refBackward1,
    __global const uchar *refForward1,
    __global const uchar *refBackward2,
    __global const uchar *refForward2,
    __global const uchar *refBackward3,
    __global const uchar *refForward3,
    __global const uchar *refBackward4,
    __global const uchar *refForward4,
    __global const uchar *refBackward5,
    __global const uchar *refForward5,
    const int width,
    const int height,
    __global const degrain_mv_t *mv,
    __global const degrain_sad_t *sad,
    __global const float *temporalMixPrior,
    const int blocksX,
    const int blocksY,
    const int blockSize,
    const int overlap,
    const int step,
    const int coveredWidth,
    const int coveredHeight,
    const int planeScaleX,
    const int planeScaleY,
    const uint thsad,
    __global const uint *disableMaskPtr) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const uint disableMask = disableMaskPtr[0];

    const int dstPitch = dst_pitch / (int)sizeof(TypePixel);
    const int fallback = degrain_pixel_load(cur, cur_pitch, width, height, x, y);
    const int scaleX = degrain_plane_scale_x(planeScaleX);
    const int scaleY = degrain_plane_scale_y(planeScaleY);
    const int renderBlockSize = degrain_render_const_block_size(blockSize);
    const int renderOverlap = degrain_render_const_overlap(overlap);
    const int renderStep = degrain_render_const_step(step);
    const int renderBlocksX = degrain_render_const_blocks_x(blocksX);
    const int renderBlocksY = degrain_render_const_blocks_y(blocksY);
    const int renderCoveredWidth = degrain_render_const_covered_width(coveredWidth, scaleX);
    const int renderCoveredHeight = degrain_render_const_covered_height(coveredHeight, scaleY);
    if (!degrain_is_covered_pixel(x, y, renderCoveredWidth, renderCoveredHeight)) {
        dst[y * dstPitch + x] = degrain_clamp_pixel(fallback);
        return;
    }

    const int planeBlockSizeX = max(degrain_render_scale_floor(renderBlockSize, scaleX), 1);
    const int planeBlockSizeY = max(degrain_render_scale_floor(renderBlockSize, scaleY), 1);
    const int planeOverlapX = max(degrain_render_scale_floor(renderOverlap, scaleX), 0);
    const int planeOverlapY = max(degrain_render_scale_floor(renderOverlap, scaleY), 0);
    const int planeStepX = max(degrain_render_scale_floor(renderStep, scaleX), 1);
    const int planeStepY = max(degrain_render_scale_floor(renderStep, scaleY), 1);
    const int primaryBlockX = min(x / planeStepX, renderBlocksX - 1);
    const int primaryBlockY = min(y / planeStepY, renderBlocksY - 1);
    const int usePrevBlockX = planeOverlapX > 0 && primaryBlockX > 0 && x < degrain_block_origin(primaryBlockX, planeStepX) + planeOverlapX;
    const int usePrevBlockY = planeOverlapY > 0 && primaryBlockY > 0 && y < degrain_block_origin(primaryBlockY, planeStepY) + planeOverlapY;
    const int blockXs[2] = { primaryBlockX, primaryBlockX - 1 };
    const int blockYs[2] = { primaryBlockY, primaryBlockY - 1 };
    const int blockCountX = usePrevBlockX ? 2 : 1;
    const int blockCountY = usePrevBlockY ? 2 : 1;

    degrain_window_accum_t sampleSum = degrain_window_accum_zero();
    degrain_window_accum_t weightSum = degrain_window_accum_zero();
    int sampleCount = 0;
    for (int byIndex = 0; byIndex < blockCountY; byIndex++) {
        const int blockY = blockYs[byIndex];
        const int baseY = degrain_block_origin(blockY, planeStepY);
        for (int bxIndex = 0; bxIndex < blockCountX; bxIndex++) {
            const int blockX = blockXs[bxIndex];
            const int baseX = degrain_block_origin(blockX, planeStepX);
            const int localX = x - baseX;
            const int localY = y - baseY;
            if (localX < 0 || localX >= planeBlockSizeX || localY < 0 || localY >= planeBlockSizeY) {
                continue;
            }
            const float windowWeight = degrain_window_factor_rect_2d(
                x, y,
                baseX, baseY,
                planeBlockSizeX, planeBlockSizeY,
                planeOverlapX, planeOverlapY,
                blockX, blockY,
                renderBlocksX, renderBlocksY);

            const int block = blockY * renderBlocksX + blockX;
            int sample = degrain_pixel_load(cur, cur_pitch, width, height, x, y);
            sample = degrain_degrain_block_sample(
                cur, cur_pitch,
                refBackward1, refForward1,
                refBackward2, refForward2,
                refBackward3, refForward3,
                refBackward4, refForward4,
                refBackward5, refForward5,
                width, height,
                mv, sad,
                block, thsad,
                disableMask,
                temporalMixPrior,
                planeScaleX, planeScaleY,
                x, y);
            degrain_accumulate_windowed_sample(&sampleSum, &weightSum, sample, windowWeight);
            sampleCount++;
        }
    }

    const int result = (sampleCount > 0) ? degrain_finalize_windowed_sample(sampleSum, weightSum, fallback) : fallback;
    dst[y * dstPitch + x] = degrain_clamp_pixel(result);
}

static inline void degrain_degrain_overlap_plane_preweighted_ramp_generic(
    __global TypePixel *dst,
    const int dst_pitch,
    __global const uchar *cur,
    const int cur_pitch,
    __global const uchar *refBackward1,
    __global const uchar *refForward1,
    __global const uchar *refBackward2,
    __global const uchar *refForward2,
    __global const uchar *refBackward3,
    __global const uchar *refForward3,
    __global const uchar *refBackward4,
    __global const uchar *refForward4,
    __global const uchar *refBackward5,
    __global const uchar *refForward5,
    const int width,
    const int height,
    __global const degrain_mv_t *mv,
    const int blocksX,
    const int blocksY,
    const int blockSize,
    const int overlap,
    const int step,
    const int coveredWidth,
    const int coveredHeight,
    const int planeScaleX,
    const int planeScaleY,
    __global const float *windowRamp,
    __global const float *temporalMixPlan,
    const int originX,
    const int originY,
    const int compactTopLeftBorder) {
    const int globalX = (int)get_global_id(0);
    const int globalY = (int)get_global_id(1);
    int x = originX + globalX;
    int y = originY + globalY;
    if (compactTopLeftBorder) {
        if (originX <= 0 || originY <= 0) {
            return;
        }
        const int compactScaleX = degrain_plane_scale_x(planeScaleX);
        const int compactScaleY = degrain_plane_scale_y(planeScaleY);
        const int compactStep = degrain_render_const_step(step);
        const int compactBlocksX = degrain_render_const_blocks_x(blocksX);
        const int compactBlocksY = degrain_render_const_blocks_y(blocksY);
        const int compactPlaneStepX = max(degrain_render_scale_floor(compactStep, compactScaleX), 1);
        const int compactPlaneStepY = max(degrain_render_scale_floor(compactStep, compactScaleY), 1);
        const int interiorEndX = min(width, compactBlocksX * compactPlaneStepX);
        const int interiorEndY = min(height, compactBlocksY * compactPlaneStepY);
        const int lowerHeight = max(height - originY, 0);
        const int rightBorderWidth = max(width - interiorEndX, 0);
        const int bottomBorderWidth = max(interiorEndX - originX, 0);
        const int borderIndex = globalX;
        const int topBorderPixels = width * originY;
        const int leftBorderPixels = originX * lowerHeight;
        const int rightBorderPixels = rightBorderWidth * lowerHeight;
        if (borderIndex < topBorderPixels) {
            x = borderIndex % width;
            y = borderIndex / width;
        } else if (borderIndex < topBorderPixels + leftBorderPixels) {
            const int leftBorderIndex = borderIndex - topBorderPixels;
            x = leftBorderIndex % originX;
            y = originY + leftBorderIndex / originX;
        } else if (borderIndex < topBorderPixels + leftBorderPixels + rightBorderPixels) {
            const int rightBorderIndex = borderIndex - topBorderPixels - leftBorderPixels;
            x = interiorEndX + rightBorderIndex % rightBorderWidth;
            y = originY + rightBorderIndex / rightBorderWidth;
        } else {
            const int bottomBorderIndex = borderIndex - topBorderPixels - leftBorderPixels - rightBorderPixels;
            if (bottomBorderWidth <= 0) {
                return;
            }
            x = originX + bottomBorderIndex % bottomBorderWidth;
            y = interiorEndY + bottomBorderIndex / bottomBorderWidth;
        }
    }
    if (x >= width || y >= height) {
        return;
    }

    const int dstPitch = dst_pitch / (int)sizeof(TypePixel);
    const int fallback = degrain_pixel_load(cur, cur_pitch, width, height, x, y);
    const int scaleX = degrain_plane_scale_x(planeScaleX);
    const int scaleY = degrain_plane_scale_y(planeScaleY);
    const int renderBlockSize = degrain_render_const_block_size(blockSize);
    const int renderOverlap = degrain_render_const_overlap(overlap);
    const int renderStep = degrain_render_const_step(step);
    const int renderBlocksX = degrain_render_const_blocks_x(blocksX);
    const int renderBlocksY = degrain_render_const_blocks_y(blocksY);
    const int renderCoveredWidth = degrain_render_const_covered_width(coveredWidth, scaleX);
    const int renderCoveredHeight = degrain_render_const_covered_height(coveredHeight, scaleY);
    if (!degrain_is_covered_pixel(x, y, renderCoveredWidth, renderCoveredHeight)) {
        dst[y * dstPitch + x] = degrain_clamp_pixel(fallback);
        return;
    }

    const int planeBlockSizeX = max(degrain_render_scale_floor(renderBlockSize, scaleX), 1);
    const int planeBlockSizeY = max(degrain_render_scale_floor(renderBlockSize, scaleY), 1);
    const int planeOverlapX = max(degrain_render_scale_floor(renderOverlap, scaleX), 0);
    const int planeOverlapY = max(degrain_render_scale_floor(renderOverlap, scaleY), 0);
    const int planeStepX = max(degrain_render_scale_floor(renderStep, scaleX), 1);
    const int planeStepY = max(degrain_render_scale_floor(renderStep, scaleY), 1);
    const int primaryBlockX = min(x / planeStepX, renderBlocksX - 1);
    const int primaryBlockY = min(y / planeStepY, renderBlocksY - 1);
    const int primaryBaseX = degrain_block_origin(primaryBlockX, planeStepX);
    const int primaryBaseY = degrain_block_origin(primaryBlockY, planeStepY);
    const int primaryLocalX = x - primaryBaseX;
    const int primaryLocalY = y - primaryBaseY;
    const int primaryBlock = primaryBlockY * renderBlocksX + primaryBlockX;
    const int usePrevBlockX = planeOverlapX > 0 && primaryBlockX > 0 && primaryLocalX < planeOverlapX;
    const int usePrevBlockY = planeOverlapY > 0 && primaryBlockY > 0 && primaryLocalY < planeOverlapY;
    const float wxPrev = usePrevBlockX ? windowRamp[primaryLocalX] : 0.0f;
    const float wyPrev = usePrevBlockY ? windowRamp[planeOverlapX + primaryLocalY] : 0.0f;
    const float wx[2] = { 1.0f - wxPrev, wxPrev };
    const float wy[2] = { 1.0f - wyPrev, wyPrev };

    const int blockXs[2] = { primaryBlockX, primaryBlockX - 1 };
    const int blockYs[2] = { primaryBlockY, primaryBlockY - 1 };
    const int localXs[2] = { primaryLocalX, primaryLocalX + planeStepX };
    const int localYs[2] = { primaryLocalY, primaryLocalY + planeStepY };
    const int blockRows[2] = { primaryBlock, primaryBlock - renderBlocksX };
    const int blockCountX = usePrevBlockX ? 2 : 1;
    const int blockCountY = usePrevBlockY ? 2 : 1;

    float sampleSum = 0.0f;
    float weightSum = 0.0f;
    for (int byIndex = 0; byIndex < blockCountY; byIndex++) {
        const int blockY = blockYs[byIndex];
        const int localY = localYs[byIndex];
        const int blockRow = blockRows[byIndex];
        for (int bxIndex = 0; bxIndex < blockCountX; bxIndex++) {
            const int blockX = blockXs[bxIndex];
            const int localX = localXs[bxIndex];
            if (localX < 0 || localX >= planeBlockSizeX || localY < 0 || localY >= planeBlockSizeY
                || blockX < 0 || blockX >= renderBlocksX || blockY < 0 || blockY >= renderBlocksY) {
                continue;
            }
            const int block = blockRow - bxIndex;
            const int sample = degrain_apply_temporal_mix_plan_same_pitch(
                cur, cur_pitch,
                refBackward1, refForward1,
                refBackward2, refForward2,
                refBackward3, refForward3,
                refBackward4, refForward4,
                refBackward5, refForward5,
                width, height,
                mv, block, temporalMixPlan,
                planeScaleX, planeScaleY,
                x, y);
            degrain_accumulate_weighted_sample_fp32(&sampleSum, &weightSum, sample, wx[bxIndex] * wy[byIndex]);
        }
    }

    const int result = degrain_finalize_weighted_sample_fp32(sampleSum, weightSum, fallback);
    dst[y * dstPitch + x] = degrain_clamp_pixel(result);
}

__kernel void kernel_degrain_degrain_overlap_plane_preweighted_ramp(
    __global TypePixel *dst,
    const int dst_pitch,
    __global const uchar *cur,
    const int cur_pitch,
    __global const uchar *refBackward1,
    __global const uchar *refForward1,
    __global const uchar *refBackward2,
    __global const uchar *refForward2,
    __global const uchar *refBackward3,
    __global const uchar *refForward3,
    __global const uchar *refBackward4,
    __global const uchar *refForward4,
    __global const uchar *refBackward5,
    __global const uchar *refForward5,
    const int width,
    const int height,
    __global const degrain_mv_t *mv,
    const int blocksX,
    const int blocksY,
    const int blockSize,
    const int overlap,
    const int step,
    const int coveredWidth,
    const int coveredHeight,
    const int planeScaleX,
    const int planeScaleY,
    __global const float *windowRamp,
    __global const float *temporalMixPlan) {
    degrain_degrain_overlap_plane_preweighted_ramp_generic(
        dst, dst_pitch,
        cur, cur_pitch,
        refBackward1, refForward1,
        refBackward2, refForward2,
        refBackward3, refForward3,
        refBackward4, refForward4,
        refBackward5, refForward5,
        width, height,
        mv,
        blocksX, blocksY,
        blockSize, overlap, step,
        coveredWidth, coveredHeight,
        planeScaleX, planeScaleY,
        windowRamp, temporalMixPlan,
        0, 0, 0);
}

__kernel void kernel_degrain_pixel_trace(
    __global const uchar *cur,
    const int cur_pitch,
    __global const uchar *refBackward1,
    __global const uchar *refForward1,
    __global const uchar *refBackward2,
    __global const uchar *refForward2,
    __global const uchar *refBackward3,
    __global const uchar *refForward3,
    __global const uchar *refBackward4,
    __global const uchar *refForward4,
    __global const uchar *refBackward5,
    __global const uchar *refForward5,
    const int width,
    const int height,
    __global const degrain_mv_t *mv,
    __global const degrain_sad_t *sad,
    __global const float *temporalMixPrior,
    const int blocksX,
    const int blocksY,
    const int blockSize,
    const int overlap,
    const int step,
    const int coveredWidth,
    const int coveredHeight,
    const int planeScaleX,
    const int planeScaleY,
    const uint thsad,
    __global const uint *disableMaskPtr,
    const int targetX,
    const int targetY,
    __global int *trace) {
    const int x = clamp(targetX, 0, max(width - 1, 0));
    const int y = clamp(targetY, 0, max(height - 1, 0));
    const uint disableMask = disableMaskPtr[0];
    const int fallback = degrain_pixel_load(cur, cur_pitch, width, height, x, y);
    const int scaleX = degrain_plane_scale_x(planeScaleX);
    const int scaleY = degrain_plane_scale_y(planeScaleY);
    const int renderBlockSize = degrain_render_const_block_size(blockSize);
    const int renderOverlap = degrain_render_const_overlap(overlap);
    const int renderStep = degrain_render_const_step(step);
    const int renderBlocksX = degrain_render_const_blocks_x(blocksX);
    const int renderBlocksY = degrain_render_const_blocks_y(blocksY);
    const int renderCoveredWidth = degrain_render_const_covered_width(coveredWidth, scaleX);
    const int renderCoveredHeight = degrain_render_const_covered_height(coveredHeight, scaleY);
    const int covered = degrain_is_covered_pixel(x, y, renderCoveredWidth, renderCoveredHeight);

    const int planeBlockSizeX = max(degrain_render_scale_floor(renderBlockSize, scaleX), 1);
    const int planeBlockSizeY = max(degrain_render_scale_floor(renderBlockSize, scaleY), 1);
    const int planeOverlapX = max(degrain_render_scale_floor(renderOverlap, scaleX), 0);
    const int planeOverlapY = max(degrain_render_scale_floor(renderOverlap, scaleY), 0);
    const int planeStepX = max(degrain_render_scale_floor(renderStep, scaleX), 1);
    const int planeStepY = max(degrain_render_scale_floor(renderStep, scaleY), 1);
    const int primaryBlockX = min(x / planeStepX, renderBlocksX - 1);
    const int primaryBlockY = min(y / planeStepY, renderBlocksY - 1);
    const int usePrevBlockX = planeOverlapX > 0 && primaryBlockX > 0 && x < degrain_block_origin(primaryBlockX, planeStepX) + planeOverlapX;
    const int usePrevBlockY = planeOverlapY > 0 && primaryBlockY > 0 && y < degrain_block_origin(primaryBlockY, planeStepY) + planeOverlapY;
    const int blockXs[2] = { primaryBlockX, primaryBlockX - 1 };
    const int blockYs[2] = { primaryBlockY, primaryBlockY - 1 };
    const int blockCountX = usePrevBlockX ? 2 : 1;
    const int blockCountY = usePrevBlockY ? 2 : 1;

    degrain_window_accum_t sampleSum = degrain_window_accum_zero();
    degrain_window_accum_t weightSum = degrain_window_accum_zero();
    int sampleCount = 0;
    int record = 0;
    for (int i = 0; i < 256; i++) {
        trace[i] = 0;
    }

    if (covered) {
        for (int byIndex = 0; byIndex < blockCountY; byIndex++) {
            const int blockY = blockYs[byIndex];
            const int baseY = degrain_block_origin(blockY, planeStepY);
            for (int bxIndex = 0; bxIndex < blockCountX; bxIndex++) {
                const int blockX = blockXs[bxIndex];
                const int baseX = degrain_block_origin(blockX, planeStepX);
                const int localX = x - baseX;
                const int localY = y - baseY;
                if (localX < 0 || localX >= planeBlockSizeX || localY < 0 || localY >= planeBlockSizeY) {
                    continue;
                }
                const float windowWeight = degrain_window_factor_rect_2d(
                    x, y,
                    baseX, baseY,
                    planeBlockSizeX, planeBlockSizeY,
                    planeOverlapX, planeOverlapY,
                    blockX, blockY,
                    renderBlocksX, renderBlocksY);

                const int block = blockY * renderBlocksX + blockX;
                const int srcSample = fallback;
                const float sourceConfidenceRaw = degrain_temporal_mix_prior_center(temporalMixPrior);
                float referenceConfidenceRaw[DEGRAIN_REFS];
                int referenceSample[DEGRAIN_REFS];
                int referenceValid[DEGRAIN_REFS];
                float confidenceTotal = sourceConfidenceRaw;
                for (int referenceDirection = 0; referenceDirection < DEGRAIN_REFS; referenceDirection++) {
                    const float temporalMixPriorRef = degrain_temporal_mix_prior_ref(temporalMixPrior, referenceDirection);
                    referenceValid[referenceDirection] = degrain_reference_is_valid(mv, sad, block, referenceDirection, thsad, degrain_ref_direction_disabled(disableMask, referenceDirection));
                    referenceConfidenceRaw[referenceDirection] = degrain_reference_mix_affinity(mv, sad, block, referenceDirection, thsad, degrain_ref_direction_disabled(disableMask, referenceDirection)) * temporalMixPriorRef;
                    confidenceTotal += referenceConfidenceRaw[referenceDirection];
                    referenceSample[referenceDirection] = 0;
                }
                const float invTotal = (confidenceTotal > 0.0f) ? (1.0f / confidenceTotal) : 0.0f;
                float mixedValue = (float)srcSample * (sourceConfidenceRaw * invTotal);
                float referenceMixTotal = 0.0f;
                float referenceMixNorm[DEGRAIN_REFS];
                for (int referenceDirection = 0; referenceDirection < DEGRAIN_REFS; referenceDirection++) {
                    referenceMixNorm[referenceDirection] = (referenceConfidenceRaw[referenceDirection] > 0.0f) ? (referenceConfidenceRaw[referenceDirection] * invTotal) : 0.0f;
                    referenceMixTotal += referenceMixNorm[referenceDirection];
                    if (referenceMixNorm[referenceDirection] <= 0.0f) {
                        continue;
                    }
                    __global const uchar *referencePlane = degrain_ref_plane_ptr_same_pitch(
                        refBackward1, refForward1,
                        refBackward2, refForward2,
                        refBackward3, refForward3,
                        refBackward4, refForward4,
                        refBackward5, refForward5,
                        referenceDirection);
                    referenceSample[referenceDirection] = degrain_compensated_sample(referencePlane, cur_pitch, width, height, mv, block, referenceDirection, planeScaleX, planeScaleY, x, y);
                    mixedValue = fma((float)referenceSample[referenceDirection], referenceMixNorm[referenceDirection], mixedValue);
                }
                const float sourceMixNorm = sourceConfidenceRaw * invTotal;
                const int sample = degrain_clamp_pixel(convert_int_rte(mixedValue));
                const int contribution = degrain_windowed_sample_contribution(sample, windowWeight);
                degrain_accumulate_windowed_sample(&sampleSum, &weightSum, sample, windowWeight);
                sampleCount++;

                if (record < 4) {
                    const int out = 32 + record * 48;
                    trace[out + 0] = blockX;
                    trace[out + 1] = blockY;
                    trace[out + 2] = block;
                    trace[out + 3] = baseX;
                    trace[out + 4] = baseY;
                    trace[out + 5] = x - baseX;
                    trace[out + 6] = y - baseY;
                    trace[out + 7] = degrain_trace_float_to_q8(windowWeight);
                    trace[out + 8] = srcSample;
                    trace[out + 9] = sample;
                    trace[out + 10] = contribution;
                    trace[out + 11] = degrain_trace_float_to_q8(sourceMixNorm);
                    trace[out + 12] = degrain_trace_float_to_q8(referenceMixTotal);
                    trace[out + 13] = degrain_trace_float_to_q8(confidenceTotal);
                    trace[out + 14] = degrain_trace_float_to_q8(degrain_temporal_mix_prior_center(temporalMixPrior));
                    for (int refDirection = 0; refDirection < min(DEGRAIN_REFS, 4); refDirection++) {
                        const int traceOffset = out + 15 + refDirection * 6;
                        const int motionIndex = degrain_ref_index(block, refDirection);
                        trace[traceOffset + 0] = degrain_trace_float_to_q8(referenceMixNorm[refDirection]);
                        trace[traceOffset + 1] = referenceSample[refDirection];
                        trace[traceOffset + 2] = (int)mv[motionIndex].dx;
                        trace[traceOffset + 3] = (int)mv[motionIndex].dy;
                        trace[traceOffset + 4] = (int)sad[motionIndex].sad;
                        trace[traceOffset + 5] = referenceValid[refDirection];
                    }
                    record++;
                }
            }
        }
    }

    const int result = (covered && sampleCount > 0) ? degrain_finalize_windowed_sample(sampleSum, weightSum, fallback) : fallback;
    trace[0] = 0x4d435054;
    trace[1] = x;
    trace[2] = y;
    trace[3] = width;
    trace[4] = height;
    trace[5] = fallback;
    trace[6] = covered;
    trace[7] = scaleX;
    trace[8] = scaleY;
    trace[9] = planeBlockSizeX;
    trace[10] = planeBlockSizeY;
    trace[11] = planeOverlapX;
    trace[12] = planeOverlapY;
    trace[13] = planeStepX;
    trace[14] = primaryBlockX;
    trace[15] = primaryBlockY;
    trace[16] = blockCountX;
    trace[17] = blockCountY;
    trace[18] = degrain_trace_window_accum(sampleSum);
    trace[19] = sampleCount;
    trace[20] = result;
    trace[21] = (int)thsad;
    trace[22] = (int)disableMask;
    trace[23] = renderBlocksX;
    trace[24] = renderBlocksY;
    trace[25] = record;
}
