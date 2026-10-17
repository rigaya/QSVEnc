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

#ifndef RGY_FILTER_DEGRAIN_MOTION_SEARCH_SEARCH_CL
#define RGY_FILTER_DEGRAIN_MOTION_SEARCH_SEARCH_CL

#define DEGRAIN_MOTION_SEARCH_SEARCH_CANDIDATES 3
#define DEGRAIN_MOTION_SEARCH_MAX_CANDIDATE_GROUPS 8
#define DEGRAIN_MOTION_SEARCH_SEARCH_LOCAL_SIZE (DEGRAIN_BLK_SIZE * 8)
#define DEGRAIN_MOTION_SEARCH_LARGE_COST 0xffffffffu

#ifndef DEGRAIN_MOTION_SEARCH_SUBGROUP
#define DEGRAIN_MOTION_SEARCH_SUBGROUP 0
#endif

#ifndef DEGRAIN_MOTION_SEARCH_SUBGROUP_SIZE
#define DEGRAIN_MOTION_SEARCH_SUBGROUP_SIZE 0
#endif

#ifndef DEGRAIN_MOTION_SEARCH_SUBGROUP_DIRECT_REDUCE
#define DEGRAIN_MOTION_SEARCH_SUBGROUP_DIRECT_REDUCE 1
#endif

#ifndef DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE
#define DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE 1
#endif

#ifndef DEGRAIN_MOTION_SEARCH_LAZY_REF_WINDOW
#define DEGRAIN_MOTION_SEARCH_LAZY_REF_WINDOW 1
#endif

#ifndef DEGRAIN_MOTION_SEARCH_SPATIAL_REUSE_PREVIOUS_SAD
#define DEGRAIN_MOTION_SEARCH_SPATIAL_REUSE_PREVIOUS_SAD 1
#endif

#ifndef DEGRAIN_MOTION_SEARCH_EARLY_SAD_THRESHOLD
#define DEGRAIN_MOTION_SEARCH_EARLY_SAD_THRESHOLD -1
#endif

#ifndef DEGRAIN_MOTION_SPATIAL_EARLY_SAD_THRESHOLD
#define DEGRAIN_MOTION_SPATIAL_EARLY_SAD_THRESHOLD -1
#endif

#ifndef DEGRAIN_MOTION_SEARCH_PASS_FLAT_FLAG
#define DEGRAIN_MOTION_SEARCH_PASS_FLAT_FLAG 0
#endif

// spatial refineへ平坦判定を渡す間だけscore_primaryの上位bitを使用する。
// export_sadが最終SADでscore_primaryを正規化するため外部表現には残らない。
#define DEGRAIN_MOTION_SEARCH_FLAT_FLAG 0x80000000u

#ifndef DEGRAIN_PIXEL_BYTES
#define DEGRAIN_PIXEL_BYTES 1
#endif

#ifndef DEGRAIN_PAD
#define DEGRAIN_PAD 0
#endif

#ifndef DEGRAIN_MOTION_COST_SCALE
#define DEGRAIN_MOTION_COST_SCALE 0
#endif

#ifndef DEGRAIN_LOW_SAD_WEIGHT_SCALE
#define DEGRAIN_LOW_SAD_WEIGHT_SCALE 0
#endif

#ifndef DEGRAIN_ZERO_CANDIDATE_COST_SCALE
#define DEGRAIN_ZERO_CANDIDATE_COST_SCALE 0
#endif

#ifndef DEGRAIN_FRAME_AVERAGE_CANDIDATE_COST_SCALE
#define DEGRAIN_FRAME_AVERAGE_CANDIDATE_COST_SCALE 0
#endif

#ifndef DEGRAIN_NEW_CANDIDATE_COST_SCALE
#define DEGRAIN_NEW_CANDIDATE_COST_SCALE 0
#endif

#ifndef DEGRAIN_LEVEL
#define DEGRAIN_LEVEL 0
#endif

static inline int degrain_motion_search_const_width(const int width) {
#if defined(DEGRAIN_WIDTH)
    return DEGRAIN_WIDTH;
#else
    return width;
#endif
}

static inline int degrain_motion_search_const_height(const int height) {
#if defined(DEGRAIN_HEIGHT)
    return DEGRAIN_HEIGHT;
#else
    return height;
#endif
}

static inline int degrain_motion_search_const_blocks_x(const int blocksX) {
#if defined(DEGRAIN_BLOCKS_X)
    return DEGRAIN_BLOCKS_X;
#else
    return blocksX;
#endif
}

static inline int degrain_motion_search_const_blocks_y(const int blocksY) {
#if defined(DEGRAIN_BLOCKS_Y)
    return DEGRAIN_BLOCKS_Y;
#else
    return blocksY;
#endif
}

static inline int degrain_motion_search_const_step(const int step) {
#if defined(DEGRAIN_STEP)
    return DEGRAIN_STEP;
#else
    return step;
#endif
}

#define DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_RADIUS 3
#define DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE (DEGRAIN_BLK_SIZE + DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_RADIUS * 2)

#if DEGRAIN_MOTION_SEARCH_SUBGROUP
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

typedef struct {
    uint score_primary;
    uint sad_metric;
    short pos_x;
    short pos_y;
} degrain_motion_search_candidate_cost_t;

static inline degrain_motion_search_saved_vector_t degrain_motion_search_candidate_cost_to_saved_vector(
    const degrain_motion_search_candidate_cost_t candidateCosts) {
    return degrain_motion_search_make_vector(candidateCosts.pos_x, candidateCosts.pos_y, candidateCosts.sad_metric, candidateCosts.score_primary);
}

static inline degrain_motion_search_candidate_cost_t degrain_motion_search_make_candidate_cost(
    const int posX,
    const int posY,
    const uint sadMetric,
    const uint scorePrimary) {
    degrain_motion_search_candidate_cost_t candidateCosts;
    candidateCosts.score_primary = scorePrimary;
    candidateCosts.sad_metric = sadMetric;
    candidateCosts.pos_x = (short)posX;
    candidateCosts.pos_y = (short)posY;
    return candidateCosts;
}

static inline degrain_motion_search_candidate_t degrain_motion_search_saved_vector_to_spatial_candidate(
    degrain_motion_search_saved_vector_t vec) {
    vec.score_primary &= ~DEGRAIN_MOTION_SEARCH_FLAT_FLAG;
    return degrain_motion_search_saved_vector_to_candidate(vec);
}

typedef struct {
    int minX;
    int minY;
    int maxX;
    int maxY;
    int motionCostWeight;
} degrain_motion_search_search_context_t;

typedef enum {
    DEGRAIN_MOTION_SEARCH_CANDIDATE_ZERO = 0,
    DEGRAIN_MOTION_SEARCH_CANDIDATE_FRAME_AVERAGE = 1,
    DEGRAIN_MOTION_SEARCH_CANDIDATE_SEED = 2
} degrain_motion_search_candidate_kind_t;

static inline int degrain_motion_search_squared_distance(
    const int ax,
    const int ay,
    const int bx,
    const int by) {
    const int motionOffsetX = ax - bx;
    const int motionOffsetY = ay - by;
    return motionOffsetX * motionOffsetX + motionOffsetY * motionOffsetY;
}

static inline int degrain_motion_search_median_of_three(
    const int a,
    const int b,
    const int c) {
    const int lo = min(a, b);
    const int hi = max(a, b);
    return max(lo, min(hi, c));
}

static inline degrain_motion_search_candidate_t degrain_motion_search_constrain_candidate(
    degrain_motion_search_candidate_t candidate,
    __local const degrain_motion_search_search_context_t *context) {
    const int maxX = context->maxX;
    const int maxY = context->maxY;
    const int minX = context->minX;
    const int minY = context->minY;
    if (maxX > minX && maxY > minY) {
        candidate.pos_x = (short)clamp((int)candidate.pos_x, minX, maxX - 1);
        candidate.pos_y = (short)clamp((int)candidate.pos_y, minY, maxY - 1);
    }
    return candidate;
}

static inline int degrain_motion_search_motion_inside_search_window(
    const int motionOffsetX,
    const int motionOffsetY,
    __local const degrain_motion_search_search_context_t *context) {
    const int maxX = context->maxX;
    const int maxY = context->maxY;
    const int minX = context->minX;
    const int minY = context->minY;
    return maxX > minX && maxY > minY
        && motionOffsetX >= minX && motionOffsetX < maxX
        && motionOffsetY >= minY && motionOffsetY < maxY;
}

#define DEGRAIN_MOTION_SEARCH_COST_SHIFT 10
#define DEGRAIN_MOTION_SEARCH_COST_INPUT_SCALE (1 << (DEGRAIN_MOTION_SEARCH_COST_SHIFT - 8))
#define DEGRAIN_MOTION_SEARCH_COST_ROUND ((ulong)1 << (DEGRAIN_MOTION_SEARCH_COST_SHIFT - 1))

static inline uint degrain_motion_search_calc_motion_cost(
    const degrain_motion_search_candidate_t candidate,
    const int seedDx,
    const int seedDy,
    __local const degrain_motion_search_search_context_t *context) {
    const ulong accum = (ulong)max(context->motionCostWeight, 0) * (ulong)degrain_motion_search_squared_distance(
        (int)candidate.pos_x,
        (int)candidate.pos_y,
        seedDx,
        seedDy);
    return (uint)((accum + DEGRAIN_MOTION_SEARCH_COST_ROUND) >> DEGRAIN_MOTION_SEARCH_COST_SHIFT);
}

static inline uint degrain_motion_search_scaled_sad_penalty(
    const uint sad,
    const int penalty) {
    const ulong accum = (ulong)sad * (ulong)max(penalty, 0);
    return (uint)((accum + DEGRAIN_MOTION_SEARCH_COST_ROUND) >> DEGRAIN_MOTION_SEARCH_COST_SHIFT);
}

#if DEGRAIN_LEVEL == 0
#define DEGRAIN_MOTION_SEARCH_ZERO_CANDIDATE_COST_SCALE_VALUE 0
#else
#define DEGRAIN_MOTION_SEARCH_ZERO_CANDIDATE_COST_SCALE_VALUE (DEGRAIN_ZERO_CANDIDATE_COST_SCALE * DEGRAIN_MOTION_SEARCH_COST_INPUT_SCALE)
#endif

#define degrain_motion_search_initial_cost_scale(candidateGroupIndex) ( \
    ((candidateGroupIndex) == DEGRAIN_MOTION_SEARCH_CANDIDATE_ZERO)   ? DEGRAIN_MOTION_SEARCH_ZERO_CANDIDATE_COST_SCALE_VALUE : \
    ((candidateGroupIndex) == DEGRAIN_MOTION_SEARCH_CANDIDATE_FRAME_AVERAGE) ? (DEGRAIN_FRAME_AVERAGE_CANDIDATE_COST_SCALE * DEGRAIN_MOTION_SEARCH_COST_INPUT_SCALE) : \
    ((candidateGroupIndex) == DEGRAIN_MOTION_SEARCH_CANDIDATE_SEED)   ? 0 : \
                                                              (DEGRAIN_NEW_CANDIDATE_COST_SCALE * DEGRAIN_MOTION_SEARCH_COST_INPUT_SCALE))

static inline degrain_motion_search_search_context_t degrain_motion_search_make_search_context(
    const degrain_mv_internal_t seed,
    const int width,
    const int height,
    const int blockGridX,
    const int blockGridY,
    const int step) {
    const int sourceBaseX = blockGridX * step;
    const int sourceBaseY = blockGridY * step;
    degrain_motion_search_search_context_t context;
    context.maxX = DEGRAIN_PEL * (width + DEGRAIN_PAD - sourceBaseX - DEGRAIN_BLK_SIZE);
    context.maxY = DEGRAIN_PEL * (height + DEGRAIN_PAD - sourceBaseY - DEGRAIN_BLK_SIZE);
    context.minX = -DEGRAIN_PEL * (sourceBaseX + DEGRAIN_PAD);
    context.minY = -DEGRAIN_PEL * (sourceBaseY + DEGRAIN_PAD);
    const int sadHalf = (int)(seed.sad_metric >> 1);
    context.motionCostWeight = 0;
    if (blockGridY > 0 && DEGRAIN_LOW_SAD_WEIGHT_SCALE > 0) {
        const long denomLL = (long)DEGRAIN_LOW_SAD_WEIGHT_SCALE + (long)sadHalf;
        const long denom2 = denomLL * denomLL;
        const int motionCostWeight = (denom2 > 0)
            ? (int)(((long)DEGRAIN_MOTION_COST_SCALE * (long)DEGRAIN_LOW_SAD_WEIGHT_SCALE * (long)DEGRAIN_LOW_SAD_WEIGHT_SCALE) / denom2)
            : DEGRAIN_MOTION_COST_SCALE;
        context.motionCostWeight = motionCostWeight * DEGRAIN_MOTION_SEARCH_COST_INPUT_SCALE;
    }
    return context;
}

static inline uint degrain_motion_search_accumulate_luma_sad_lane(
    __local const TypePixel *sourceBlockPixels,
    __global const uchar *referencePlane,
    __global const uchar *subpelPlanes,
    const int subpelPlaneStride,
    const int refPitch,
    const int width,
    const int height,
    const int blockX,
    const int blockY,
    const int step,
    const int motionOffsetX,
    const int motionOffsetY,
    const int sadLane) {
    const int referenceX = degrain_motion_search_ref_x(blockX, step, motionOffsetX);
    const int referenceY = degrain_motion_search_ref_y(blockY, step, motionOffsetY);
    // 4 pixel lanes map directly to vload4 and keep the minimum supported block/chroma
    // widths practical without extra tail handling.
    const int lanesPerRow = DEGRAIN_BLK_SIZE / 4;
    const int x = (sadLane % lanesPerRow) * 4;
    const int rowsPerLane = DEGRAIN_BLK_SIZE / lanesPerRow;
    const int firstLaneRow = sadLane / lanesPerRow;
    const int sourceStridePerLane = rowsPerLane * DEGRAIN_BLK_SIZE;
#if DEGRAIN_PEL == 2 && DEGRAIN_PIXEL_BYTES == 1
    // 事前計算済みの4位相サブペルプレーンがあれば、半ペル候補も整数座標のプレーン参照になる。
    // プレーン値は degrain_pixel_load_pel_mirror と同一の補間結果なのでビット一致。
    const int useSubpelPlanes = (subpelPlaneStride != 0);
    const int subpelPhase = (motionOffsetX & 1) + ((motionOffsetY & 1) << 1);
    __global const uchar *fastReferencePlane = useSubpelPlanes
        ? subpelPlanes + (size_t)subpelPhase * (size_t)subpelPlaneStride
        : referencePlane;
    const int useFastPath =
        (useSubpelPlanes || degrain_motion_search_ref_is_integer_pel(motionOffsetX, motionOffsetY))
        && referenceX >= 0 && referenceY >= 0
        && referenceX + DEGRAIN_BLK_SIZE <= width
        && referenceY + DEGRAIN_BLK_SIZE <= height;
#else
    __global const uchar *fastReferencePlane = referencePlane;
    const int useFastPath =
#if DEGRAIN_PIXEL_BYTES == 1
#if DEGRAIN_PEL > 1
        degrain_motion_search_ref_is_integer_pel(motionOffsetX, motionOffsetY) &&
#endif
        referenceX >= 0 && referenceY >= 0
        && referenceX + DEGRAIN_BLK_SIZE <= width
        && referenceY + DEGRAIN_BLK_SIZE <= height;
#else
        0;
#endif
#endif
    int sad = 0;
    if (useFastPath) {
#if DEGRAIN_PIXEL_BYTES == 1
        __local const uchar *sourcePtr = (__local const uchar *)(sourceBlockPixels + firstLaneRow * DEGRAIN_BLK_SIZE + x);
        const int referenceStridePerLane = rowsPerLane * refPitch;
        __global const uchar *referencePtr = fastReferencePlane + (referenceY + firstLaneRow) * refPitch + referenceX + x;
        for (int y = firstLaneRow; y < DEGRAIN_BLK_SIZE; y += rowsPerLane) {
            const uchar4 sourceValue = vload4(0, sourcePtr);
            const uchar4 referenceValue = vload4(0, referencePtr);
            const uchar4 diff = abs_diff(sourceValue, referenceValue);
            sad += diff.x + diff.y + diff.z + diff.w;
            sourcePtr += sourceStridePerLane;
            referencePtr += referenceStridePerLane;
        }
#endif
    } else {
#if DEGRAIN_PIXEL_BYTES == 1
        __local const uchar *sourcePtr = (__local const uchar *)(sourceBlockPixels + firstLaneRow * DEGRAIN_BLK_SIZE + x);
        for (int y = firstLaneRow; y < DEGRAIN_BLK_SIZE; y += rowsPerLane) {
            const uchar4 sourceValue = vload4(0, sourcePtr);
#if DEGRAIN_PEL <= 1 && DEGRAIN_PIXEL_BYTES == 1
            const int referenceSampleY = degrain_mirror_coord(referenceY + y, height);
            const int referenceSampleX0 = degrain_mirror_coord(referenceX + x + 0, width);
            const int referenceSampleX1 = degrain_mirror_coord(referenceX + x + 1, width);
            const int referenceSampleX2 = degrain_mirror_coord(referenceX + x + 2, width);
            const int referenceSampleX3 = degrain_mirror_coord(referenceX + x + 3, width);
            __global const uchar *referenceLine = referencePlane + referenceSampleY * refPitch;
            const int referenceValue0 = (int)referenceLine[referenceSampleX0];
            const int referenceValue1 = (int)referenceLine[referenceSampleX1];
            const int referenceValue2 = (int)referenceLine[referenceSampleX2];
            const int referenceValue3 = (int)referenceLine[referenceSampleX3];
#else
            const int referenceValue0 = degrain_motion_search_ref_sample(referencePlane, refPitch, width, height, blockX, blockY, step, motionOffsetX, motionOffsetY, x + 0, y);
            const int referenceValue1 = degrain_motion_search_ref_sample(referencePlane, refPitch, width, height, blockX, blockY, step, motionOffsetX, motionOffsetY, x + 1, y);
            const int referenceValue2 = degrain_motion_search_ref_sample(referencePlane, refPitch, width, height, blockX, blockY, step, motionOffsetX, motionOffsetY, x + 2, y);
            const int referenceValue3 = degrain_motion_search_ref_sample(referencePlane, refPitch, width, height, blockX, blockY, step, motionOffsetX, motionOffsetY, x + 3, y);
#endif
            sad += abs((int)sourceValue.x - referenceValue0);
            sad += abs((int)sourceValue.y - referenceValue1);
            sad += abs((int)sourceValue.z - referenceValue2);
            sad += abs((int)sourceValue.w - referenceValue3);
            sourcePtr += sourceStridePerLane;
        }
#else
        for (int y = firstLaneRow; y < DEGRAIN_BLK_SIZE; y += rowsPerLane) {
            const int sourceBase = y * DEGRAIN_BLK_SIZE + x;
            const int sourceValue0 = (int)sourceBlockPixels[sourceBase + 0];
            const int sourceValue1 = (int)sourceBlockPixels[sourceBase + 1];
            const int sourceValue2 = (int)sourceBlockPixels[sourceBase + 2];
            const int sourceValue3 = (int)sourceBlockPixels[sourceBase + 3];
            const int referenceValue0 = degrain_motion_search_ref_sample(referencePlane, refPitch, width, height, blockX, blockY, step, motionOffsetX, motionOffsetY, x + 0, y);
            const int referenceValue1 = degrain_motion_search_ref_sample(referencePlane, refPitch, width, height, blockX, blockY, step, motionOffsetX, motionOffsetY, x + 1, y);
            const int referenceValue2 = degrain_motion_search_ref_sample(referencePlane, refPitch, width, height, blockX, blockY, step, motionOffsetX, motionOffsetY, x + 2, y);
            const int referenceValue3 = degrain_motion_search_ref_sample(referencePlane, refPitch, width, height, blockX, blockY, step, motionOffsetX, motionOffsetY, x + 3, y);
            sad += abs(sourceValue0 - referenceValue0);
            sad += abs(sourceValue1 - referenceValue1);
            sad += abs(sourceValue2 - referenceValue2);
            sad += abs(sourceValue3 - referenceValue3);
        }
#endif
    }
    return (uint)sad;
}

#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
static inline void degrain_motion_search_load_reference_window(
    __local uchar *referenceWindowPixels,
    __global const uchar *referencePlane,
    const int refPitch,
    const int width,
    const int height,
    const int referenceWindowX,
    const int referenceWindowY,
    const int referenceWindowIsValid,
    const int localThreadId,
    const int localSize) {
    if (referenceWindowIsValid) {
        for (int i = localThreadId; i < DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE * DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE; i += localSize) {
            const int x = i % DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE;
            const int y = i / DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE;
            referenceWindowPixels[i] = referencePlane[(referenceWindowY + y) * refPitch + referenceWindowX + x];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

static inline uint degrain_motion_search_accumulate_luma_sad_lane_cached_ref(
    __local const TypePixel *sourceBlockPixels,
    __global const uchar *referencePlane,
    __global const uchar *subpelPlanes,
    const int subpelPlaneStride,
    __local const uchar *referenceWindowPixels,
    const int refPitch,
    const int width,
    const int height,
    const int blockX,
    const int blockY,
    const int step,
    const int motionOffsetX,
    const int motionOffsetY,
    const int sadLane,
    const int referenceWindowX,
    const int referenceWindowY,
    const int referenceWindowIsValid) {
#if DEGRAIN_PEL <= 1
    const int referenceX = degrain_motion_search_ref_x(blockX, step, motionOffsetX);
    const int referenceY = degrain_motion_search_ref_y(blockY, step, motionOffsetY);
    const int useLocalRef = referenceWindowIsValid
        && referenceX >= referenceWindowX && referenceY >= referenceWindowY
        && referenceX + DEGRAIN_BLK_SIZE <= referenceWindowX + DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE
        && referenceY + DEGRAIN_BLK_SIZE <= referenceWindowY + DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE;
    if (useLocalRef) {
        // Keep the same 4 pixel lane shape as the global-memory SAD path so the
        // cached reference window only changes the memory source, not scheduling.
        const int lanesPerRow = DEGRAIN_BLK_SIZE / 4;
        const int x = (sadLane % lanesPerRow) * 4;
        const int rowsPerLane = DEGRAIN_BLK_SIZE / lanesPerRow;
        const int firstLaneRow = sadLane / lanesPerRow;
        const int sourceStridePerLane = rowsPerLane * DEGRAIN_BLK_SIZE;
        const int referenceStridePerLane = rowsPerLane * DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE;
        const int localReferenceX = referenceX - referenceWindowX;
        const int localReferenceY = referenceY - referenceWindowY;
        int sad = 0u;
        __local const uchar *sourcePtr = (__local const uchar *)(sourceBlockPixels + firstLaneRow * DEGRAIN_BLK_SIZE + x);
        __local const uchar *referencePtr = referenceWindowPixels + (localReferenceY + firstLaneRow) * DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE + localReferenceX + x;
        for (int y = firstLaneRow; y < DEGRAIN_BLK_SIZE; y += rowsPerLane) {
            const uchar4 sourceValue = vload4(0, sourcePtr);
            const uchar4 referenceValue = vload4(0, referencePtr);
            const uchar4 diff = abs_diff(sourceValue, referenceValue);
            sad += diff.x + diff.y + diff.z + diff.w;
            sourcePtr += sourceStridePerLane;
            referencePtr += referenceStridePerLane;
        }
        return (uint)sad;
    }
#endif
    return degrain_motion_search_accumulate_luma_sad_lane(
        sourceBlockPixels,
        referencePlane,
        subpelPlanes,
        subpelPlaneStride,
        refPitch,
        width,
        height,
        blockX,
        blockY,
        step,
        motionOffsetX,
        motionOffsetY,
        sadLane);
}
#endif

static inline uint degrain_motion_search_sum_candidate_sad_lanes(
    __local uint *candidateLaneSums,
    const uint sad,
    const int candidateIsValid,
    const int sadLane,
    const int candidateGroupIndex) {
    const int partialBase = candidateGroupIndex * DEGRAIN_BLK_SIZE;
#if DEGRAIN_MOTION_SEARCH_SUBGROUP
#if DEGRAIN_MOTION_SEARCH_SUBGROUP_SIZE > 0
    const uint subgroupSize = (uint)DEGRAIN_MOTION_SEARCH_SUBGROUP_SIZE;
    const int useSubgroup = subgroupSize <= (uint)DEGRAIN_BLK_SIZE
        && (((uint)DEGRAIN_BLK_SIZE % subgroupSize) == 0u)
        && (((uint)DEGRAIN_MOTION_SEARCH_SEARCH_LOCAL_SIZE % subgroupSize) == 0u);
#else
    const uint subgroupSize = get_sub_group_size();
    const uint subgroupCountInWorkgroup = get_num_sub_groups();
    const uint localSize = get_local_size(0);
    const int useSubgroup = subgroupSize <= (uint)DEGRAIN_BLK_SIZE
        && (((uint)DEGRAIN_BLK_SIZE % subgroupSize) == 0u)
        && (subgroupCountInWorkgroup == localSize / subgroupSize);
#endif
    if (useSubgroup) {
        const uint subgroupSad = sub_group_reduce_add(candidateIsValid ? sad : 0u);
#if DEGRAIN_MOTION_SEARCH_SUBGROUP_DIRECT_REDUCE && DEGRAIN_MOTION_SEARCH_SUBGROUP_SIZE == DEGRAIN_BLK_SIZE
        return subgroupSad;
#else
        if (candidateIsValid && get_sub_group_local_id() == 0u) {
            candidateLaneSums[partialBase + sadLane / (int)subgroupSize] = subgroupSad;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        uint totalSad = 0u;
        if (candidateIsValid && sadLane == 0) {
            const int subgroupCount = DEGRAIN_BLK_SIZE / (int)subgroupSize;
            for (int i = 0; i < subgroupCount; i++) {
                totalSad += candidateLaneSums[partialBase + i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        return totalSad;
#endif
    }
#endif

    if (candidateIsValid) {
        candidateLaneSums[partialBase + sadLane] = sad;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 各候補グループの先頭レーンがDEGRAIN_BLK_SIZE個を逐次加算する。
    // 以前の木構造リダクションはlog2(DEGRAIN_BLK_SIZE)回のworkgroup barrierを要し、
    // 1ブロック=1 workgroupの探索ではbarrier待ちが支配的だった。整数加算なので結果は同一。
    // 呼び出し元はすべて、この後candidateLaneSumsへ再度書き込む前にbarrierを置いていること。
    uint totalSad = 0u;
    if (candidateIsValid && sadLane == 0) {
        for (int i = 0; i < DEGRAIN_BLK_SIZE; i++) {
            totalSad += candidateLaneSums[partialBase + i];
        }
    }
    return totalSad;
}

static inline int degrain_motion_search_candidate_cost_magnitude(
    const degrain_motion_search_candidate_cost_t candidateCost) {
    return abs((int)candidateCost.pos_x) + abs((int)candidateCost.pos_y);
}

static inline int degrain_motion_search_candidate_cost_prefers(
    const degrain_motion_search_candidate_cost_t lhs,
    const degrain_motion_search_candidate_cost_t rhs) {
    if (lhs.score_primary != rhs.score_primary) {
        return lhs.score_primary < rhs.score_primary;
    }
    return degrain_motion_search_candidate_cost_magnitude(lhs) < degrain_motion_search_candidate_cost_magnitude(rhs);
}

static inline void degrain_motion_search_select_lowest_candidate_cost(
    __local degrain_motion_search_candidate_cost_t *candidateCosts,
    const int localThreadId,
    const int candidateCount) {
    if (localThreadId < 8 && localThreadId >= candidateCount) {
        candidateCosts[localThreadId] = degrain_motion_search_make_candidate_cost(0, 0, 0u, DEGRAIN_MOTION_SEARCH_LARGE_COST);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = 1; stride < 8; stride <<= 1) {
        if (localThreadId < 8
            && (localThreadId + stride) < 8
            && (localThreadId & ((stride << 1) - 1)) == 0
            && degrain_motion_search_candidate_cost_prefers(candidateCosts[localThreadId + stride], candidateCosts[localThreadId])) {
            candidateCosts[localThreadId] = candidateCosts[localThreadId + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

static inline int degrain_motion_search_find_first_matching_candidate(
    __local const degrain_motion_search_candidate_t *candidate,
    const int candidateGroupIndex,
    const int candidateCount) {
    int canonical = candidateGroupIndex;
    if (candidateGroupIndex < candidateCount) {
        const degrain_motion_search_candidate_t motionVector = candidate[candidateGroupIndex];
        for (int i = 0; i < candidateGroupIndex; i++) {
            if (candidate[i].pos_x == motionVector.pos_x && candidate[i].pos_y == motionVector.pos_y) {
                canonical = i;
                break;
            }
        }
    }
    return canonical;
}

static inline int2 degrain_motion_search_refine_wide6_offset(const int offsetIndex) {
    const int row = offsetIndex >> 1;
    const int side = offsetIndex & 1;
    const int y = (row - 1) * 2;
    const int x = (row == 1) ? (side * 4 - 2) : (side * 2 - 1);
    return (int2)(x, y);
}

static inline int2 degrain_motion_search_refine_square_offset(const int offsetIndex) {
    const int gridIndex = offsetIndex + ((offsetIndex >= 4) ? 1 : 0);
    return (int2)(gridIndex % 3 - 1, gridIndex / 3 - 1);
}

static inline void degrain_motion_search_refine_clear_results(
    __local degrain_motion_search_candidate_cost_t *candidateCosts,
    const int localThreadId) {
    if (localThreadId < 8) {
        candidateCosts[localThreadId] = degrain_motion_search_make_candidate_cost(0, 0, 0u, DEGRAIN_MOTION_SEARCH_LARGE_COST);
    }
}

static inline void degrain_motion_search_refine_prepare_offset_candidate(
    __local const degrain_motion_search_search_context_t *context,
    __local degrain_motion_search_candidate_cost_t *candidateCosts,
    __local degrain_motion_search_candidate_cost_t *bestCandidateCost,
    const int localThreadId,
    const int candidateCount,
    const int baseX,
    const int baseY,
    const int2 offset) {
    if (localThreadId < candidateCount) {
        const int motionOffsetX = baseX + offset.x;
        const int motionOffsetY = baseY + offset.y;
        const degrain_motion_search_candidate_t candidate = degrain_motion_search_make_candidate(motionOffsetX, motionOffsetY, 0u, 0u);
        const uint motionCost = degrain_motion_search_calc_motion_cost(candidate, baseX, baseY, context);
        if (degrain_motion_search_motion_inside_search_window(motionOffsetX, motionOffsetY, context) && motionCost < bestCandidateCost->score_primary) {
            candidateCosts[localThreadId].pos_x = (short)motionOffsetX;
            candidateCosts[localThreadId].pos_y = (short)motionOffsetY;
            candidateCosts[localThreadId].score_primary = motionCost;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

static inline void degrain_motion_search_refine_prepare_hex_candidates(
    __local const degrain_motion_search_search_context_t *context,
    __local degrain_motion_search_candidate_cost_t *candidateCosts,
    __local degrain_motion_search_candidate_cost_t *bestCandidateCost,
    const int localThreadId,
    const int baseX,
    const int baseY) {
    degrain_motion_search_refine_clear_results(candidateCosts, localThreadId);
    degrain_motion_search_refine_prepare_offset_candidate(
        context, candidateCosts, bestCandidateCost, localThreadId, 6, baseX, baseY, degrain_motion_search_refine_wide6_offset(localThreadId));
}

static inline void degrain_motion_search_refine_prepare_square_candidates(
    __local const degrain_motion_search_search_context_t *context,
    __local degrain_motion_search_candidate_cost_t *candidateCosts,
    __local degrain_motion_search_candidate_cost_t *bestCandidateCost,
    const int localThreadId,
    const int baseX,
    const int baseY) {
    degrain_motion_search_refine_clear_results(candidateCosts, localThreadId);
    degrain_motion_search_refine_prepare_offset_candidate(
        context, candidateCosts, bestCandidateCost, localThreadId, 8, baseX, baseY, degrain_motion_search_refine_square_offset(localThreadId));
}

static inline int degrain_motion_search_refine_has_valid_candidates(
    __local const degrain_motion_search_candidate_cost_t *candidateCosts,
    const int candidateCount) {
    int hasCandidateToEvaluate = 0;
    for (int i = 0; i < candidateCount; i++) {
        hasCandidateToEvaluate |= candidateCosts[i].score_primary != DEGRAIN_MOTION_SEARCH_LARGE_COST;
    }
    return hasCandidateToEvaluate;
}

static inline void degrain_motion_search_refine_evaluate_candidates(
    __local const TypePixel *sourceBlockPixels,
    __global const uchar *referencePlane,
    __global const uchar *subpelPlanes,
    const int subpelPlaneStride,
    __local const degrain_motion_search_search_context_t *context,
    __local degrain_motion_search_candidate_cost_t *candidateCosts,
    __local uint *candidateLaneSums,
    __local degrain_motion_search_candidate_cost_t *bestCandidateCost,
    const int localThreadId,
    const int sadLane,
    const int candidateGroupIndex,
    const int candidateCount,
    const int blockX,
    const int blockY,
    const int step,
    const int refPitch,
    const int width,
    const int height
#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
    ,
    __local const uchar *referenceWindowPixels,
    const int referenceWindowX,
    const int referenceWindowY,
    const int referenceWindowIsValid
#endif
    ) {

    const int candidateIsValid = candidateGroupIndex < candidateCount
        && candidateCosts[candidateGroupIndex].score_primary != DEGRAIN_MOTION_SEARCH_LARGE_COST;
    uint sad = 0u;
    if (candidateIsValid) {
#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
        sad = degrain_motion_search_accumulate_luma_sad_lane_cached_ref(
            sourceBlockPixels,
            referencePlane,
            subpelPlanes,
            subpelPlaneStride,
            referenceWindowPixels,
            refPitch,
            width,
            height,
            blockX,
            blockY,
            step,
            candidateCosts[candidateGroupIndex].pos_x,
            candidateCosts[candidateGroupIndex].pos_y,
            sadLane,
            referenceWindowX,
            referenceWindowY,
            referenceWindowIsValid);
#else
        sad = degrain_motion_search_accumulate_luma_sad_lane(
            sourceBlockPixels,
            referencePlane,
            subpelPlanes,
            subpelPlaneStride,
            refPitch,
            width,
            height,
            blockX,
            blockY,
            step,
            candidateCosts[candidateGroupIndex].pos_x,
            candidateCosts[candidateGroupIndex].pos_y,
            sadLane);
#endif
    }
    sad = degrain_motion_search_sum_candidate_sad_lanes(candidateLaneSums, sad, candidateIsValid, sadLane, candidateGroupIndex);

    if (candidateIsValid && sadLane == 0) {
        candidateCosts[candidateGroupIndex].sad_metric = sad;
        candidateCosts[candidateGroupIndex].score_primary += sad + degrain_motion_search_scaled_sad_penalty(sad, DEGRAIN_NEW_CANDIDATE_COST_SCALE * DEGRAIN_MOTION_SEARCH_COST_INPUT_SCALE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    degrain_motion_search_select_lowest_candidate_cost(candidateCosts, localThreadId, candidateCount);

    if (localThreadId == 0 && degrain_motion_search_candidate_cost_prefers(candidateCosts[0], *bestCandidateCost)) {
        *bestCandidateCost = candidateCosts[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

static inline void degrain_motion_search_refine_prepared_candidates(
    __local const TypePixel *sourceBlockPixels,
    __global const uchar *referencePlane,
    __global const uchar *subpelPlanes,
    const int subpelPlaneStride,
    __local const degrain_motion_search_search_context_t *context,
    __local degrain_motion_search_candidate_cost_t *candidateCosts,
    __local uint *candidateLaneSums,
    __local degrain_motion_search_candidate_cost_t *bestCandidateCost,
    const int localThreadId,
    const int sadLane,
    const int candidateGroupIndex,
    const int blockX,
    const int blockY,
    const int step,
    const int refPitch,
    const int width,
    const int height,
    const int candidateCount
#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
    ,
    __local const uchar *referenceWindowPixels,
    const int referenceWindowX,
    const int referenceWindowY,
    const int referenceWindowIsValid
#endif
    ) {
    degrain_motion_search_refine_evaluate_candidates(
        sourceBlockPixels, referencePlane, subpelPlanes, subpelPlaneStride, context, candidateCosts, candidateLaneSums, bestCandidateCost,
        localThreadId, sadLane, candidateGroupIndex, candidateCount, blockX, blockY, step, refPitch, width, height
#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
        ,
        referenceWindowPixels,
        referenceWindowX,
        referenceWindowY,
        referenceWindowIsValid
#endif
        );
}

static inline void degrain_motion_search_refine_hex2(
    __local const TypePixel *sourceBlockPixels,
    __global const uchar *referencePlane,
    __global const uchar *subpelPlanes,
    const int subpelPlaneStride,
    __local const degrain_motion_search_search_context_t *context,
    __local degrain_motion_search_candidate_cost_t *candidateCosts,
    __local uint *candidateLaneSums,
    __local degrain_motion_search_candidate_cost_t *bestCandidateCost,
    const int localThreadId,
    const int sadLane,
    const int candidateGroupIndex,
    const int blockX,
    const int blockY,
    const int step,
    const int refPitch,
    const int width,
    const int height
#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
    ,
    __local const uchar *referenceWindowPixels,
    const int referenceWindowX,
    const int referenceWindowY,
    const int referenceWindowIsValid
#endif
    ) {
    degrain_motion_search_refine_prepare_hex_candidates(context, candidateCosts, bestCandidateCost, localThreadId, bestCandidateCost->pos_x, bestCandidateCost->pos_y);
    degrain_motion_search_refine_prepared_candidates(
        sourceBlockPixels, referencePlane, subpelPlanes, subpelPlaneStride, context, candidateCosts, candidateLaneSums, bestCandidateCost,
        localThreadId, sadLane, candidateGroupIndex, blockX, blockY, step, refPitch, width, height, 6
#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
        ,
        referenceWindowPixels,
        referenceWindowX,
        referenceWindowY,
        referenceWindowIsValid
#endif
        );
}

static inline void degrain_motion_search_refine_square8(
    __local const TypePixel *sourceBlockPixels,
    __global const uchar *referencePlane,
    __global const uchar *subpelPlanes,
    const int subpelPlaneStride,
    __local const degrain_motion_search_search_context_t *context,
    __local degrain_motion_search_candidate_cost_t *candidateCosts,
    __local uint *candidateLaneSums,
    __local degrain_motion_search_candidate_cost_t *bestCandidateCost,
    const int localThreadId,
    const int sadLane,
    const int candidateGroupIndex,
    const int blockX,
    const int blockY,
    const int step,
    const int refPitch,
    const int width,
    const int height
#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
    ,
    __local const uchar *referenceWindowPixels,
    const int referenceWindowX,
    const int referenceWindowY,
    const int referenceWindowIsValid
#endif
    ) {
    degrain_motion_search_refine_prepare_square_candidates(context, candidateCosts, bestCandidateCost, localThreadId, bestCandidateCost->pos_x, bestCandidateCost->pos_y);
    degrain_motion_search_refine_prepared_candidates(
        sourceBlockPixels, referencePlane, subpelPlanes, subpelPlaneStride, context, candidateCosts, candidateLaneSums, bestCandidateCost,
        localThreadId, sadLane, candidateGroupIndex, blockX, blockY, step, refPitch, width, height, 8
#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
        ,
        referenceWindowPixels,
        referenceWindowX,
        referenceWindowY,
        referenceWindowIsValid
#endif
        );
}

// workgroup全体でuintの部分和を総和し、全スレッドへ同一の合計値を返す。
// 整数加算のみのため加算順序に依らず結果は一意。
// 内部でbarrierを使うので、workgroup全スレッドから一様に呼び出すこと。
static inline uint degrain_motion_search_workgroup_reduce_add(
    __local uint *laneSums,
    const uint value,
    const int localThreadId) {
    laneSums[localThreadId] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = DEGRAIN_MOTION_SEARCH_SEARCH_LOCAL_SIZE >> 1; offset > 0; offset >>= 1) {
        if (localThreadId < offset) {
            laneSums[localThreadId] += laneSums[localThreadId + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const uint total = laneSums[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return total;
}

// source blockの分散をworkgroup並列で求める。
// 内部でbarrierを使うので、workgroup全スレッドから一様に呼び出すこと。
static inline uint degrain_motion_search_source_block_variance_parallel(
    __local const TypePixel *sourceBlockPixels,
    __local uint *laneSums,
    const int localThreadId) {
    const int count = DEGRAIN_BLK_SIZE * DEGRAIN_BLK_SIZE;
#if DEGRAIN_PIXEL_BYTES == 1
    // 8bit画素: 列ごとの和/2乗和をDEGRAIN_BLK_SIZEレーンで求め、1回のbarrierで集約する。
    // 以前はworkgroup全体(256レーン)の木構造リダクションを2回行っており、barrierが約20回発生して
    // L0探索カーネルの約1/4をこの判定だけで消費していた。整数加算なので集約順序を変えても結果は同一。
    // 列の2乗和は最大 DEGRAIN_BLK_SIZE * 255^2 で32bitに収まる。
    if (localThreadId < DEGRAIN_BLK_SIZE) {
        uint columnSum = 0u;
        uint columnSumSq = 0u;
        for (int y = 0; y < DEGRAIN_BLK_SIZE; y++) {
            const uint value = (uint)sourceBlockPixels[y * DEGRAIN_BLK_SIZE + localThreadId];
            columnSum += value;
            columnSumSq += value * value;
        }
        laneSums[localThreadId] = columnSum;
        laneSums[DEGRAIN_BLK_SIZE + localThreadId] = columnSumSq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    long sum = 0;
    long sumSq = 0;
    for (int i = 0; i < DEGRAIN_BLK_SIZE; i++) {
        sum += laneSums[i];
        sumSq += laneSums[DEGRAIN_BLK_SIZE + i];
    }
    // 呼び出し元がlaneSumsを再利用するため、全スレッドの読み出し完了を待つ
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    uint partialSum = 0u;
    // 16bit画素は2乗和の合計が32bitを超えるため、v^2 (32bitに収まる) を上位/下位に分けて総和する
    uint partialSumSqLo = 0u;
    uint partialSumSqHi = 0u;
    for (int i = localThreadId; i < count; i += DEGRAIN_MOTION_SEARCH_SEARCH_LOCAL_SIZE) {
        const uint value = (uint)sourceBlockPixels[i];
        partialSum += value;
        const uint valueSq = value * value;
        partialSumSqLo += valueSq & 0xffffu;
        partialSumSqHi += valueSq >> 16;
    }
    const long sum = (long)degrain_motion_search_workgroup_reduce_add(laneSums, partialSum, localThreadId);
    const long sumSqLo = (long)degrain_motion_search_workgroup_reduce_add(laneSums, partialSumSqLo, localThreadId);
    const long sumSqHi = (long)degrain_motion_search_workgroup_reduce_add(laneSums, partialSumSqHi, localThreadId);
    const long sumSq = (sumSqHi << 16) + sumSqLo;
#endif
    const long mean = sum / count;
    const long varianceNumer = sumSq - mean * sum;
    return (varianceNumer <= 0) ? 0u : (uint)(varianceNumer / count);
}

// full-block SADをworkgroup並列で求める。レーン分割は探索時のSAD計算と同一。
// 内部でbarrierを使うので、workgroup全スレッドから一様に呼び出すこと。
static inline uint degrain_motion_search_full_block_sad_parallel(
    __local const TypePixel *sourceBlockPixels,
    __global const uchar *referencePlane,
    __global const uchar *subpelPlanes,
    const int subpelPlaneStride,
    __local uint *laneSums,
    const int pitch,
    const int width,
    const int height,
    const int blockGridX,
    const int blockGridY,
    const int step,
    const int motionOffsetX,
    const int motionOffsetY,
    const int localThreadId) {
    uint partialSad = 0u;
    if (localThreadId < DEGRAIN_BLK_SIZE) {
        partialSad = degrain_motion_search_accumulate_luma_sad_lane(
            sourceBlockPixels,
            referencePlane,
            subpelPlanes,
            subpelPlaneStride,
            pitch,
            width,
            height,
            blockGridX,
            blockGridY,
            step,
            motionOffsetX,
            motionOffsetY,
            localThreadId);
    }
    return degrain_motion_search_workgroup_reduce_add(laneSums, partialSad, localThreadId);
}

// 探索勝者のraw SAD確定とflat領域補正 (分散0のブロックはMVの信頼性がないためzero MVへ寄せる)。
// 分散はworkgroup内で同一値になるため、flat分岐は全スレッド一様でbarrier安全。
// 内部でbarrierを使うので、workgroup全スレッドから同一のbestを渡して一様に呼び出すこと。
static inline degrain_motion_search_candidate_cost_t degrain_motion_search_finalize_candidate_cost_parallel(
    __local const TypePixel *sourceBlockPixels,
    __global const uchar *referencePlane,
    __global const uchar *subpelPlanes,
    const int subpelPlaneStride,
    __local uint *laneSums,
    const int pitch,
    const int width,
    const int height,
    const int blockGridX,
    const int blockGridY,
    const int step,
    degrain_motion_search_candidate_cost_t best,
    const int sourceBlockIsFlat,
    const int localThreadId) {
    // 探索時に保持したraw SADを再利用し、勝者位置の同一SAD計算を省略する。
    best.score_primary = best.sad_metric;

    const int isFlat = (sourceBlockIsFlat >= 0)
        ? sourceBlockIsFlat
        : (degrain_motion_search_source_block_variance_parallel(sourceBlockPixels, laneSums, localThreadId) == 0u);
    if (isFlat) {
        const uint sadZero = degrain_motion_search_full_block_sad_parallel(
            sourceBlockPixels,
            referencePlane,
            subpelPlanes,
            subpelPlaneStride,
            laneSums,
            pitch,
            width,
            height,
            blockGridX,
            blockGridY,
            step,
            0,
            0,
            localThreadId);
        best.pos_x = 0;
        best.pos_y = 0;
        best.sad_metric = sadZero;
        best.score_primary = sadZero;
    }
#if DEGRAIN_MOTION_SEARCH_PASS_FLAT_FLAG
    if (isFlat) {
        best.score_primary |= DEGRAIN_MOTION_SEARCH_FLAT_FLAG;
    }
#endif
    return best;
}

static inline degrain_motion_search_candidate_t degrain_motion_search_load_base_candidate(
    __global const degrain_mv_internal_t *vectors,
    __local const degrain_motion_search_search_context_t *context,
    const int candidateSlot,
    const int planeBase,
    const int blockCount,
    const int block) {
    degrain_motion_search_saved_vector_t motionVector = degrain_motion_search_make_vector(0, 0, 0u, 0u);
    switch (candidateSlot) {
    case DEGRAIN_MOTION_SEARCH_CANDIDATE_ZERO:
        motionVector = vectors[degrain_motion_search_vec_zero_index(planeBase)];
        break;
    case DEGRAIN_MOTION_SEARCH_CANDIDATE_FRAME_AVERAGE:
        motionVector = vectors[degrain_motion_search_vec_global_index(planeBase)];
        break;
    case DEGRAIN_MOTION_SEARCH_CANDIDATE_SEED:
        motionVector = vectors[degrain_motion_search_vec_current_index(planeBase, blockCount, block)];
        break;
    default:
        break;
    }
    return degrain_motion_search_constrain_candidate(degrain_motion_search_saved_vector_to_candidate(motionVector), context);
}

static inline void degrain_motion_search_search_one_block(
    __global const uchar *sourcePlane,
    __global const uchar *referencePlane,
    __global const uchar *subpelPlanes,
    const int subpelPlaneStride,
    __global degrain_mv_internal_t *vectors,
    const int pitch,
    const int width,
    const int height,
    const int contextWidth,
    const int contextHeight,
    const int planeBase,
    const int blockCount,
    const int step,
    const int block,
    const int blockGridX,
    const int blockGridY,
    const int localThreadId,
    const int sadLane,
    const int candidateGroupIndex,
    const int localSize,
    __local TypePixel *sourceBlockPixels,
    __local degrain_motion_search_search_context_t *context,
    __local degrain_motion_search_candidate_t *candidate,
    __local degrain_motion_search_candidate_cost_t *candidateCosts,
    __local degrain_motion_search_candidate_cost_t *bestCandidateCost,
    __local uint *candidateLaneSums
#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
    ,
    __local uchar *referenceWindowPixels,
    const int useRefLocalCache
#endif
    ) {
    const int sourceBaseX = blockGridX * step;
    const int sourceBaseY = blockGridY * step;
    for (int i = localThreadId; i < DEGRAIN_BLK_SIZE * DEGRAIN_BLK_SIZE; i += localSize) {
        const int x = i % DEGRAIN_BLK_SIZE;
        const int y = i / DEGRAIN_BLK_SIZE;
        sourceBlockPixels[i] = (TypePixel)degrain_pixel_load(sourcePlane, pitch, width, height, sourceBaseX + x, sourceBaseY + y);
    }

    const degrain_mv_internal_t initialSeed = vectors[degrain_motion_search_vec_current_index(planeBase, blockCount, block)];
    if (localThreadId == 0) {
        *context = degrain_motion_search_make_search_context(initialSeed, contextWidth, contextHeight, blockGridX, blockGridY, step);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (localThreadId < DEGRAIN_MOTION_SEARCH_SEARCH_CANDIDATES) {
        candidate[localThreadId] = degrain_motion_search_load_base_candidate(
            vectors,
            context,
            localThreadId,
            planeBase,
            blockCount,
            block);
    }
    if (localThreadId < DEGRAIN_MOTION_SEARCH_MAX_CANDIDATE_GROUPS) {
        candidateCosts[localThreadId] = degrain_motion_search_make_candidate_cost(0, 0, 0u, DEGRAIN_MOTION_SEARCH_LARGE_COST);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int candidateIsValid = candidateGroupIndex < DEGRAIN_MOTION_SEARCH_SEARCH_CANDIDATES;
    degrain_motion_search_candidate_t motionVector = degrain_motion_search_make_candidate(0, 0, 0u, 0u);
    if (candidateIsValid) {
        motionVector = candidate[candidateGroupIndex];
    }
    const int firstMatchingCandidateIndex = degrain_motion_search_find_first_matching_candidate(
        candidate, candidateGroupIndex, DEGRAIN_MOTION_SEARCH_SEARCH_CANDIDATES);
    const int candidateNeedsEvaluation = candidateIsValid && firstMatchingCandidateIndex == candidateGroupIndex;
    uint sad = 0u;
    if (candidateNeedsEvaluation) {
        sad = degrain_motion_search_accumulate_luma_sad_lane(
            sourceBlockPixels,
            referencePlane,
            subpelPlanes,
            subpelPlaneStride,
            pitch,
            width,
            height,
            blockGridX,
            blockGridY,
            step,
            motionVector.pos_x,
            motionVector.pos_y,
            sadLane);
    }
    sad = degrain_motion_search_sum_candidate_sad_lanes(candidateLaneSums, sad, candidateNeedsEvaluation, sadLane, candidateGroupIndex);

    if (candidateNeedsEvaluation && sadLane == 0) {
        uint cost = sad;
        if (candidateGroupIndex < 3) {
            cost += degrain_motion_search_scaled_sad_penalty(sad, degrain_motion_search_initial_cost_scale(candidateGroupIndex));
        } else {
            cost += degrain_motion_search_calc_motion_cost(motionVector, initialSeed.pos_x, initialSeed.pos_y, context);
        }
        candidateCosts[candidateGroupIndex].pos_x = motionVector.pos_x;
        candidateCosts[candidateGroupIndex].pos_y = motionVector.pos_y;
        candidateCosts[candidateGroupIndex].sad_metric = sad;
        candidateCosts[candidateGroupIndex].score_primary = cost;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (candidateIsValid && !candidateNeedsEvaluation && sadLane == 0) {
        sad = candidateCosts[firstMatchingCandidateIndex].sad_metric;
        uint cost = sad;
        if (candidateGroupIndex < 3) {
            cost += degrain_motion_search_scaled_sad_penalty(sad, degrain_motion_search_initial_cost_scale(candidateGroupIndex));
        } else {
            cost += degrain_motion_search_calc_motion_cost(motionVector, initialSeed.pos_x, initialSeed.pos_y, context);
        }
        candidateCosts[candidateGroupIndex].pos_x = motionVector.pos_x;
        candidateCosts[candidateGroupIndex].pos_y = motionVector.pos_y;
        candidateCosts[candidateGroupIndex].sad_metric = sad;
        candidateCosts[candidateGroupIndex].score_primary = cost;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    degrain_motion_search_select_lowest_candidate_cost(candidateCosts, localThreadId, DEGRAIN_MOTION_SEARCH_SEARCH_CANDIDATES);

    if (localThreadId == 0) {
        *bestCandidateCost = candidateCosts[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 1 work-groupが1 blockだけを担当するため、全work-itemで一様な分岐になる。
    // 閾値はホスト側でblock sizeとbit depthに合わせてスケール済み。
    // 0はSAD==0だけを対象とし、負値では探索を省略しない。
#if DEGRAIN_MOTION_SEARCH_EARLY_SAD_THRESHOLD >= 0
    const int skipFullSearch = (DEGRAIN_MOTION_SEARCH_EARLY_SAD_THRESHOLD == 0)
        ? (bestCandidateCost->sad_metric == 0u)
        : (bestCandidateCost->sad_metric < (uint)DEGRAIN_MOTION_SEARCH_EARLY_SAD_THRESHOLD);
#else
    const int skipFullSearch = 0;
#endif
    if (!skipFullSearch) {
// searchparam<=1はsquare8のみ、searchparam>=2はwide6+square8を実行する。
#if DEGRAIN_SEARCH_PARAM >= 2
#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
    degrain_motion_search_refine_prepare_hex_candidates(context, candidateCosts, bestCandidateCost, localThreadId, bestCandidateCost->pos_x, bestCandidateCost->pos_y);
    {
        const int hasCandidateToEvaluate = degrain_motion_search_refine_has_valid_candidates(candidateCosts, 6);
        const int refWinCenterX = degrain_motion_search_ref_x(blockGridX, step, bestCandidateCost->pos_x);
        const int refWinCenterY = degrain_motion_search_ref_y(blockGridY, step, bestCandidateCost->pos_y);
        const int referenceWindowX = refWinCenterX - DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_RADIUS;
        const int referenceWindowY = refWinCenterY - DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_RADIUS;
        const int referenceWindowIsValid = useRefLocalCache
#if DEGRAIN_PEL <= 1
            && referenceWindowX >= 0 && referenceWindowY >= 0
            && referenceWindowX + DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE <= width
            && referenceWindowY + DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE <= height
#else
            && 0
#endif
            ;
        if (useRefLocalCache && (!DEGRAIN_MOTION_SEARCH_LAZY_REF_WINDOW || hasCandidateToEvaluate)) {
            degrain_motion_search_load_reference_window(referenceWindowPixels, referencePlane, pitch, width, height, referenceWindowX, referenceWindowY, referenceWindowIsValid, localThreadId, localSize);
        }
        degrain_motion_search_refine_evaluate_candidates(
            sourceBlockPixels, referencePlane, subpelPlanes, subpelPlaneStride, context, candidateCosts, candidateLaneSums, bestCandidateCost,
            localThreadId, sadLane, candidateGroupIndex, 6, blockGridX, blockGridY, step, pitch, width, height,
            referenceWindowPixels, referenceWindowX, referenceWindowY, referenceWindowIsValid);
    }
#else
    degrain_motion_search_refine_hex2(
        sourceBlockPixels,
        referencePlane,
        subpelPlanes,
        subpelPlaneStride,
        context,
        candidateCosts,
        candidateLaneSums,
        bestCandidateCost,
        localThreadId,
        sadLane,
        candidateGroupIndex,
        blockGridX,
        blockGridY,
        step,
        pitch,
        width,
        height
#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
        ,
        referenceWindowPixels,
        referenceWindowX,
        referenceWindowY,
        referenceWindowIsValid
#endif
        );
#endif
#endif // DEGRAIN_SEARCH_PARAM >= 2

#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
    degrain_motion_search_refine_prepare_square_candidates(context, candidateCosts, bestCandidateCost, localThreadId, bestCandidateCost->pos_x, bestCandidateCost->pos_y);
    {
        const int hasCandidateToEvaluate = degrain_motion_search_refine_has_valid_candidates(candidateCosts, 8);
        const int refWinCenterX = degrain_motion_search_ref_x(blockGridX, step, bestCandidateCost->pos_x);
        const int refWinCenterY = degrain_motion_search_ref_y(blockGridY, step, bestCandidateCost->pos_y);
        const int referenceWindowX = refWinCenterX - DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_RADIUS;
        const int referenceWindowY = refWinCenterY - DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_RADIUS;
        const int referenceWindowIsValid = useRefLocalCache
#if DEGRAIN_PEL <= 1
            && referenceWindowX >= 0 && referenceWindowY >= 0
            && referenceWindowX + DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE <= width
            && referenceWindowY + DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE <= height
#else
            && 0
#endif
            ;
        if (useRefLocalCache && (!DEGRAIN_MOTION_SEARCH_LAZY_REF_WINDOW || hasCandidateToEvaluate)) {
            degrain_motion_search_load_reference_window(referenceWindowPixels, referencePlane, pitch, width, height, referenceWindowX, referenceWindowY, referenceWindowIsValid, localThreadId, localSize);
        }
        degrain_motion_search_refine_evaluate_candidates(
            sourceBlockPixels, referencePlane, subpelPlanes, subpelPlaneStride, context, candidateCosts, candidateLaneSums, bestCandidateCost,
            localThreadId, sadLane, candidateGroupIndex, 8, blockGridX, blockGridY, step, pitch, width, height,
            referenceWindowPixels, referenceWindowX, referenceWindowY, referenceWindowIsValid);
    }
#else
    degrain_motion_search_refine_square8(
        sourceBlockPixels,
        referencePlane,
        subpelPlanes,
        subpelPlaneStride,
        context,
        candidateCosts,
        candidateLaneSums,
        bestCandidateCost,
        localThreadId,
        sadLane,
        candidateGroupIndex,
        blockGridX,
        blockGridY,
        step,
        pitch,
        width,
        height
#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
        ,
        referenceWindowPixels,
        referenceWindowX,
        referenceWindowY,
        referenceWindowIsValid
#endif
        );
#endif

    }

    // bestCandidateCostは直前のsquare8 refine末尾のbarrierにより全スレッドから可視
    const degrain_motion_search_candidate_cost_t finalizedBest = degrain_motion_search_finalize_candidate_cost_parallel(
        sourceBlockPixels,
        referencePlane,
        subpelPlanes,
        subpelPlaneStride,
        candidateLaneSums,
        pitch,
        width,
        height,
        blockGridX,
        blockGridY,
        step,
        *bestCandidateCost,
        -1,
        localThreadId);
    if (localThreadId == 0) {
        vectors[degrain_motion_search_vec_current_index(planeBase, blockCount, block)] =
            degrain_motion_search_candidate_cost_to_saved_vector(finalizedBest);
    }
}

__kernel void kernel_degrain_mv_search_parallel(
    __global const uchar *sourcePlane,
    __global const uchar *referencePlane,
    __global const uchar *subpelPlanes,
    const int subpelPlaneStride,
    __global degrain_mv_internal_t *vectors,
    const int pitch,
    const int width,
    const int height,
    const int planeBase,
    const int blockCount,
    const int blocksX,
    const int blocksY,
    const int step) {
    const int localThreadId = get_local_id(0);
    // Split a work-group into candidate groups and SAD lanes. This keeps candidate
    // evaluation parallel, reuses the local source block for spatial locality,
    // minimizes global memory traffic, and keeps synchronization points limited to
    // the reductions and best-candidate updates.
    const int sadLane = localThreadId % DEGRAIN_BLK_SIZE;
    const int candidateGroupIndex = localThreadId / DEGRAIN_BLK_SIZE;
    const int localSize = get_local_size(0);
    const int block = (int)get_group_id(0);
    const int kernelWidth = degrain_motion_search_const_width(width);
    const int kernelHeight = degrain_motion_search_const_height(height);
    const int kernelBlocksX = degrain_motion_search_const_blocks_x(blocksX);
    const int kernelBlocksY = degrain_motion_search_const_blocks_y(blocksY);
    const int kernelStep = degrain_motion_search_const_step(step);

    __local TypePixel sourceBlockPixels[DEGRAIN_BLK_SIZE * DEGRAIN_BLK_SIZE];
    __local degrain_motion_search_search_context_t context;
    // Candidate/result/best stay in local memory because they are small, reused by
    // all lanes, and updated between initial search and refine without global trips.
    __local degrain_motion_search_candidate_t candidate[DEGRAIN_MOTION_SEARCH_SEARCH_CANDIDATES];
    __local degrain_motion_search_candidate_cost_t candidateCosts[8];
    __local degrain_motion_search_candidate_cost_t bestCandidateCost;
    __local uint candidateLaneSums[DEGRAIN_MOTION_SEARCH_SEARCH_LOCAL_SIZE];
#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
    __local uchar referenceWindowPixels[DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE * DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE];
#endif

    if (sourcePlane == 0 || referencePlane == 0 || kernelBlocksX <= 0 || kernelBlocksY <= 0 || block >= blockCount) {
        return;
    }

    const int blockGridX = block % kernelBlocksX;
    const int blockGridY = block / kernelBlocksX;
    if (blockGridY >= kernelBlocksY) {
        return;
    }

    degrain_motion_search_search_one_block(
        sourcePlane,
        referencePlane,
        subpelPlanes,
        subpelPlaneStride,
        vectors,
        pitch,
        kernelWidth,
        kernelHeight,
        width,
        height,
        planeBase,
        blockCount,
        kernelStep,
        block,
        blockGridX,
        blockGridY,
        localThreadId,
        sadLane,
        candidateGroupIndex,
        localSize,
        sourceBlockPixels,
        &context,
        candidate,
        candidateCosts,
        &bestCandidateCost,
        candidateLaneSums
#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
        ,
        referenceWindowPixels,
        1
#endif
        );
}

__kernel void kernel_degrain_mv_spatial_refine(
    __global const uchar *sourcePlane,
    __global const uchar *referencePlane,
    __global const uchar *subpelPlanes,
    const int subpelPlaneStride,
    __global degrain_mv_internal_t *vectors,
    __global const degrain_mv_internal_t *vectorsPrev,
    __global degrain_mv_internal_t *vectorsFinal,
    const int pitch,
    const int width,
    const int height,
    const int planeBase,
    const int finalBase,
    const int blockCount,
    const int blocksX,
    const int blocksY,
    const int step) {
    const int localThreadId = get_local_id(0);
    // Use the same lane layout as the initial search so this pass keeps candidate
    // evaluation parallel, preserves local source-block reuse, minimizes global
    // memory traffic, and needs only the reduction/update barriers.
    const int sadLane = localThreadId % DEGRAIN_BLK_SIZE;
    const int candidateGroupIndex = localThreadId / DEGRAIN_BLK_SIZE;
    const int localSize = get_local_size(0);
    const int block = get_group_id(0);
    const int kernelWidth = degrain_motion_search_const_width(width);
    const int kernelHeight = degrain_motion_search_const_height(height);
    const int kernelBlocksX = degrain_motion_search_const_blocks_x(blocksX);
    const int kernelBlocksY = degrain_motion_search_const_blocks_y(blocksY);
    const int kernelStep = degrain_motion_search_const_step(step);
    if (sourcePlane == 0 || referencePlane == 0 || block >= blockCount || kernelBlocksX <= 0 || kernelBlocksY <= 0) {
        return;
    }

    const int blockGridX = block % kernelBlocksX;
    const int blockGridY = block / kernelBlocksX;
    __local TypePixel sourceBlockPixels[DEGRAIN_BLK_SIZE * DEGRAIN_BLK_SIZE];
    __local degrain_motion_search_search_context_t context;
    // Keep the small candidate set in local memory so it can be scored and refined
    // together without serial candidate loops or extra global-memory round trips.
    __local degrain_motion_search_candidate_t candidate[5];
    __local degrain_motion_search_candidate_cost_t candidateCosts[8];
    __local degrain_motion_search_candidate_cost_t bestCandidateCost;
    __local int reusePreviousSad;
    __local int sourceBlockIsFlat;
    __local degrain_motion_search_saved_vector_t baseVector;
    __local uint candidateLaneSums[DEGRAIN_MOTION_SEARCH_SEARCH_LOCAL_SIZE];
#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
    __local uchar referenceWindowPixels[DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE * DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE];
#endif

    if (localThreadId == 0) {
        baseVector = vectors[degrain_motion_search_vec_current_index(planeBase, blockCount, block)];
#if DEGRAIN_MOTION_SEARCH_PASS_FLAT_FLAG
        sourceBlockIsFlat = (baseVector.score_primary & DEGRAIN_MOTION_SEARCH_FLAT_FLAG) != 0u;
#else
        sourceBlockIsFlat = -1;
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#if DEGRAIN_MOTION_SEARCH_PASS_FLAT_FLAG
    if (sourceBlockIsFlat) {
        if (localThreadId == 0) {
            vectorsFinal[degrain_motion_search_vec_final_index(finalBase, blockCount, block)] = baseVector;
        }
        return;
    }
#endif

    // level searchで十分小さいraw SADが得られたblockは、近傍MVを使った再探索を省略する。
    // 1 work-groupが1 blockだけを担当するため、全work-itemで一様な分岐になる。
#if DEGRAIN_MOTION_SPATIAL_EARLY_SAD_THRESHOLD >= 0
    const int skipSpatialRefine = (DEGRAIN_MOTION_SPATIAL_EARLY_SAD_THRESHOLD == 0)
        ? (baseVector.sad_metric == 0u)
        : (baseVector.sad_metric < (uint)DEGRAIN_MOTION_SPATIAL_EARLY_SAD_THRESHOLD);
    if (skipSpatialRefine) {
        if (localThreadId == 0) {
            vectorsFinal[degrain_motion_search_vec_final_index(finalBase, blockCount, block)] = baseVector;
        }
        return;
    }
#endif

    const int sourceBaseX = blockGridX * kernelStep;
    const int sourceBaseY = blockGridY * kernelStep;
    for (int i = localThreadId; i < DEGRAIN_BLK_SIZE * DEGRAIN_BLK_SIZE; i += localSize) {
        const int x = i % DEGRAIN_BLK_SIZE;
        const int y = i / DEGRAIN_BLK_SIZE;
        sourceBlockPixels[i] = (TypePixel)degrain_pixel_load(sourcePlane, pitch, kernelWidth, kernelHeight, sourceBaseX + x, sourceBaseY + y);
    }

    const degrain_mv_internal_t initialSeed =
        vectorsPrev[degrain_motion_search_vec_prev_index(planeBase, blockCount, block)];
    if (localThreadId == 0) {
        context = degrain_motion_search_make_search_context(initialSeed, width, height, blockGridX, blockGridY, kernelStep);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (localThreadId == 0) {
        degrain_motion_search_saved_vector_t baseWithoutFlatFlag = baseVector;
#if DEGRAIN_MOTION_SEARCH_PASS_FLAT_FLAG
        baseWithoutFlatFlag.score_primary &= ~DEGRAIN_MOTION_SEARCH_FLAT_FLAG;
#endif
        const degrain_motion_search_candidate_t baseCandidate =
            degrain_motion_search_saved_vector_to_spatial_candidate(baseWithoutFlatFlag);
        bestCandidateCost.pos_x = baseVector.pos_x;
        bestCandidateCost.pos_y = baseVector.pos_y;
        bestCandidateCost.sad_metric = baseVector.sad_metric;
        bestCandidateCost.score_primary = baseWithoutFlatFlag.score_primary;
        candidate[0] = degrain_motion_search_constrain_candidate(baseCandidate, &context);
        // candidate[0] is the constrained base vector; reuse its SAD only when bounds keep it unchanged.
        reusePreviousSad = DEGRAIN_MOTION_SEARCH_SPATIAL_REUSE_PREVIOUS_SAD
            && candidate[0].pos_x == baseVector.pos_x
            && candidate[0].pos_y == baseVector.pos_y;
        candidate[1] = (blockGridX > 0)
            ? degrain_motion_search_constrain_candidate(degrain_motion_search_saved_vector_to_spatial_candidate(vectors[degrain_motion_search_vec_current_index(planeBase, blockCount, block - 1)]), &context)
            : candidate[0];
        candidate[2] = (blockGridY > 0)
            ? degrain_motion_search_constrain_candidate(degrain_motion_search_saved_vector_to_spatial_candidate(vectors[degrain_motion_search_vec_current_index(planeBase, blockCount, block - kernelBlocksX)]), &context)
            : candidate[0];
        candidate[3] = (blockGridX + 1 < kernelBlocksX && blockGridY + 1 < kernelBlocksY)
            ? degrain_motion_search_constrain_candidate(degrain_motion_search_saved_vector_to_spatial_candidate(vectors[degrain_motion_search_vec_current_index(planeBase, blockCount, block + kernelBlocksX + 1)]), &context)
            : candidate[0];
        candidate[4].pos_x = (short)degrain_motion_search_median_of_three(candidate[1].pos_x, candidate[2].pos_x, candidate[3].pos_x);
        candidate[4].pos_y = (short)degrain_motion_search_median_of_three(candidate[1].pos_y, candidate[2].pos_y, candidate[3].pos_y);
        candidate[4].sad_metric = 0u;
        candidate[4].score_primary = 0u;
    }
    if (localThreadId < 5) {
        candidateCosts[localThreadId] = degrain_motion_search_make_candidate_cost(0, 0, 0u, DEGRAIN_MOTION_SEARCH_LARGE_COST);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int candidateCount = 5;
    const int candidateIsValid = candidateGroupIndex < candidateCount;
    const int firstMatchingCandidateIndex = degrain_motion_search_find_first_matching_candidate(
        candidate, candidateGroupIndex, candidateCount);
    const int candidateNeedsEvaluation = candidateIsValid && firstMatchingCandidateIndex == candidateGroupIndex;
    uint sad = 0u;
    if (candidateNeedsEvaluation) {
        if (candidateGroupIndex == 0 && reusePreviousSad) {
            sad = (sadLane == 0) ? bestCandidateCost.sad_metric : 0u;
        } else {
            const degrain_motion_search_candidate_t motionVector = candidate[candidateGroupIndex];
            sad = degrain_motion_search_accumulate_luma_sad_lane(
                sourceBlockPixels,
                referencePlane,
                subpelPlanes,
                subpelPlaneStride,
                pitch,
                kernelWidth,
                kernelHeight,
                blockGridX,
                blockGridY,
                kernelStep,
                motionVector.pos_x,
                motionVector.pos_y,
                sadLane);
        }
    }
    sad = degrain_motion_search_sum_candidate_sad_lanes(candidateLaneSums, sad, candidateNeedsEvaluation, sadLane, candidateGroupIndex);

    if (candidateNeedsEvaluation && sadLane == 0) {
        const degrain_motion_search_candidate_t motionVector = candidate[candidateGroupIndex];
        uint cost = sad;
        if (candidateGroupIndex > 0) {
            cost += degrain_motion_search_calc_motion_cost(motionVector, initialSeed.pos_x, initialSeed.pos_y, &context);
        } else {
            cost = min(bestCandidateCost.score_primary, sad);
        }
        candidateCosts[candidateGroupIndex].pos_x = motionVector.pos_x;
        candidateCosts[candidateGroupIndex].pos_y = motionVector.pos_y;
        candidateCosts[candidateGroupIndex].sad_metric = sad;
        candidateCosts[candidateGroupIndex].score_primary = cost;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (candidateIsValid && !candidateNeedsEvaluation && sadLane == 0) {
        const degrain_motion_search_candidate_t motionVector = candidate[candidateGroupIndex];
        sad = candidateCosts[firstMatchingCandidateIndex].sad_metric;
        uint cost = sad;
        if (candidateGroupIndex > 0) {
            cost += degrain_motion_search_calc_motion_cost(motionVector, initialSeed.pos_x, initialSeed.pos_y, &context);
        } else {
            cost = min(bestCandidateCost.score_primary, sad);
        }
        candidateCosts[candidateGroupIndex].pos_x = motionVector.pos_x;
        candidateCosts[candidateGroupIndex].pos_y = motionVector.pos_y;
        candidateCosts[candidateGroupIndex].sad_metric = sad;
        candidateCosts[candidateGroupIndex].score_primary = cost;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    degrain_motion_search_select_lowest_candidate_cost(candidateCosts, localThreadId, candidateCount);

    if (localThreadId == 0 && degrain_motion_search_candidate_cost_prefers(candidateCosts[0], bestCandidateCost)) {
        bestCandidateCost = candidateCosts[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
    degrain_motion_search_refine_prepare_square_candidates(&context, candidateCosts, &bestCandidateCost, localThreadId, bestCandidateCost.pos_x, bestCandidateCost.pos_y);
    {
        const int hasCandidateToEvaluate = degrain_motion_search_refine_has_valid_candidates(candidateCosts, 8);
        const int refWinCenterX = degrain_motion_search_ref_x(blockGridX, kernelStep, bestCandidateCost.pos_x);
        const int refWinCenterY = degrain_motion_search_ref_y(blockGridY, kernelStep, bestCandidateCost.pos_y);
        const int referenceWindowX = refWinCenterX - DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_RADIUS;
        const int referenceWindowY = refWinCenterY - DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_RADIUS;
        const int referenceWindowIsValid =
#if DEGRAIN_PEL <= 1
            referenceWindowX >= 0 && referenceWindowY >= 0
            && referenceWindowX + DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE <= kernelWidth
            && referenceWindowY + DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE <= kernelHeight
#else
            0
#endif
            ;
        if (!DEGRAIN_MOTION_SEARCH_LAZY_REF_WINDOW || hasCandidateToEvaluate) {
            degrain_motion_search_load_reference_window(referenceWindowPixels, referencePlane, pitch, kernelWidth, kernelHeight, referenceWindowX, referenceWindowY, referenceWindowIsValid, localThreadId, localSize);
        }
        degrain_motion_search_refine_evaluate_candidates(
            sourceBlockPixels, referencePlane, subpelPlanes, subpelPlaneStride, &context, candidateCosts, candidateLaneSums, &bestCandidateCost,
            localThreadId, sadLane, candidateGroupIndex, 8, blockGridX, blockGridY, kernelStep, pitch, kernelWidth, kernelHeight,
            referenceWindowPixels, referenceWindowX, referenceWindowY, referenceWindowIsValid);
    }
#else
    degrain_motion_search_refine_square8(
        sourceBlockPixels,
        referencePlane,
        subpelPlanes,
        subpelPlaneStride,
        &context,
        candidateCosts,
        candidateLaneSums,
        &bestCandidateCost,
        localThreadId,
        sadLane,
        candidateGroupIndex,
        blockGridX,
        blockGridY,
        kernelStep,
        pitch,
        kernelWidth,
        kernelHeight
#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
        ,
        referenceWindowPixels,
        referenceWindowX,
        referenceWindowY,
        referenceWindowIsValid
#endif
        );
#endif

    // bestCandidateCostは直前のrefine末尾のbarrierにより全スレッドから可視
    const degrain_motion_search_candidate_cost_t finalizedBest = degrain_motion_search_finalize_candidate_cost_parallel(
        sourceBlockPixels,
        referencePlane,
        subpelPlanes,
        subpelPlaneStride,
        candidateLaneSums,
        pitch,
        kernelWidth,
        kernelHeight,
        blockGridX,
        blockGridY,
        kernelStep,
        bestCandidateCost,
        sourceBlockIsFlat,
        localThreadId);
    if (localThreadId == 0) {
        vectorsFinal[degrain_motion_search_vec_final_index(finalBase, blockCount, block)] =
            degrain_motion_search_candidate_cost_to_saved_vector(finalizedBest);
    }
}

// -----------------------------------------------------------------------------------------
// 1 subgroup = 1 block で探索する版 (kernel_degrain_mv_search_sgblock)
//
// kernel_degrain_mv_search_parallel は1 work-group(256 work-item)で1 blockを担当し、候補評価・lane集約・
// 最良候補の選択を段ごとにbarrierで区切っている。1 blockあたりのSAD計算量は小さく(候補20程度 x 32x32画素)、
// 1 blockに16 HWスレッドを割り当てる割に、各段で動くlaneが少ない (初期候補は8グループ中3つだけ)。
// ここでは1 subgroupが1 blockを担当し、候補を逐次評価して、集約はsub_group_reduce_addだけで行う。
// work-group内の各subgroupは別々のblockを処理するため、work-groupのbarrierは使わない。
// Arc A310での実測 (1440x1080, blksize 32):
// - barrierを無くしただけ (4画素単位の読み出し) では、逆に遅くなった (level0 1.9ms -> 2.2ms)。
// - 16画素単位の読み出し + subgroup幅16の固定 + work-group 64で、level0 1.9ms -> 1.12ms、level1 0.81ms -> 0.54ms。
// - 行ループのunroll、複数候補をまとめて評価する形は、レジスタが溢れて遅くなったため採用していない。
//
// 出力はkernel_degrain_mv_search_parallelとビット一致させる。
// - 候補の順序、重複候補でのSAD再利用、コスト式は同じ。
// - 最良候補の選択: 元の実装は「右が厳密に優れる場合のみ置き換える」木構造の最小値選択で、
//   これは添字順の逐次走査で同じ関係(candidate_cost_prefers)を使った場合と同じ結果(最小のうち最も左)になる。
// - SADは整数の総和なので、画素のlaneへの割り当てや集約の順序を変えても一致する。
//   高速経路でない場合 (フレーム端など) は、元と同じaccumulate_luma_sad_laneをsadLaneごとに呼ぶ。
// - 参照窓のlocal cacheは値を変えない性能用の経路なので、ここではglobal参照のみを使う。
// -----------------------------------------------------------------------------------------
#if DEGRAIN_MOTION_SEARCH_SGBLOCK
#if DEGRAIN_MOTION_SEARCH_SGBLOCK == 2
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifndef DEGRAIN_MOTION_SEARCH_SGBLOCK_LOCAL_SIZE
#define DEGRAIN_MOTION_SEARCH_SGBLOCK_LOCAL_SIZE 128
#endif
// subgroup幅を固定できる場合 (cl_intel_required_subgroup_size) はその幅、できない場合は下限を8と見て、
// work-group内の最大block数ぶんのlocal memoryを確保する。
#ifdef DEGRAIN_MOTION_SEARCH_SGBLOCK_REQD_SG
#define DEGRAIN_MOTION_SEARCH_SGBLOCK_MAX_BLOCKS (DEGRAIN_MOTION_SEARCH_SGBLOCK_LOCAL_SIZE / DEGRAIN_MOTION_SEARCH_SGBLOCK_REQD_SG)
#else
#define DEGRAIN_MOTION_SEARCH_SGBLOCK_MAX_BLOCKS (DEGRAIN_MOTION_SEARCH_SGBLOCK_LOCAL_SIZE / 8)
#endif

#if DEGRAIN_PIXEL_BYTES == 1 && (DEGRAIN_BLK_SIZE % 16) == 0
#define DEGRAIN_MOTION_SEARCH_SGBLOCK_WIDE_LOAD 1
// 参照blockの左上を指すポインタを返す。高速経路 (参照がフレーム内に収まり、整数画素またはsubpelプレーンで読める) に
// 乗らない場合は0を返す。参照プレーンの選択と条件はaccumulate_luma_sad_laneと同じ。
static inline __global const uchar *degrain_motion_search_sgblock_fast_reference(
    __global const uchar *referencePlane,
    __global const uchar *subpelPlanes,
    const int subpelPlaneStride,
    const int pitch,
    const int width,
    const int height,
    const int blockGridX,
    const int blockGridY,
    const int step,
    const int motionOffsetX,
    const int motionOffsetY) {
    const int referenceX = degrain_motion_search_ref_x(blockGridX, step, motionOffsetX);
    const int referenceY = degrain_motion_search_ref_y(blockGridY, step, motionOffsetY);
#if DEGRAIN_PEL == 2
    const int useSubpelPlanes = (subpelPlaneStride != 0);
    const int subpelPhase = (motionOffsetX & 1) + ((motionOffsetY & 1) << 1);
    __global const uchar *fastReferencePlane = useSubpelPlanes
        ? subpelPlanes + (size_t)subpelPhase * (size_t)subpelPlaneStride
        : referencePlane;
    const int useFastPath = (useSubpelPlanes || degrain_motion_search_ref_is_integer_pel(motionOffsetX, motionOffsetY))
        && referenceX >= 0 && referenceY >= 0 && referenceX + DEGRAIN_BLK_SIZE <= width && referenceY + DEGRAIN_BLK_SIZE <= height;
#else
    __global const uchar *fastReferencePlane = referencePlane;
    const int useFastPath =
#if DEGRAIN_PEL > 1
        degrain_motion_search_ref_is_integer_pel(motionOffsetX, motionOffsetY) &&
#endif
        referenceX >= 0 && referenceY >= 0 && referenceX + DEGRAIN_BLK_SIZE <= width && referenceY + DEGRAIN_BLK_SIZE <= height;
#endif
    return useFastPath ? fastReferencePlane + referenceY * pitch + referenceX : (__global const uchar *)0;
}

static inline uint degrain_motion_search_sgblock_abs_diff_sum16(const uchar16 a, const uchar16 b) {
    const ushort16 diff = convert_ushort16(abs_diff(a, b));
    const ushort8 sum8 = diff.lo + diff.hi;
    const ushort4 sum4 = sum8.lo + sum8.hi;
    const ushort2 sum2 = sum4.lo + sum4.hi;
    return (uint)sum2.x + (uint)sum2.y;
}
#else
#define DEGRAIN_MOTION_SEARCH_SGBLOCK_WIDE_LOAD 0
#endif

static inline uint degrain_motion_search_sgblock_sad(
    __local const TypePixel *sourceBlockPixels,
    __global const uchar *referencePlane,
    __global const uchar *subpelPlanes,
    const int subpelPlaneStride,
    const int pitch,
    const int width,
    const int height,
    const int blockGridX,
    const int blockGridY,
    const int step,
    const int motionOffsetX,
    const int motionOffsetY,
    const int lane,
    const int subgroupSize) {
    uint sad = 0u;
#if DEGRAIN_MOTION_SEARCH_SGBLOCK_WIDE_LOAD
    // 高速経路では1 laneで16画素ずつ読む。accumulate_luma_sad_laneの4画素単位より読み出し命令数が少ない。
    // SADは整数和なので、画素の割り当てを変えても結果は同じ。
    __global const uchar *reference = degrain_motion_search_sgblock_fast_reference(
        referencePlane, subpelPlanes, subpelPlaneStride, pitch, width, height, blockGridX, blockGridY, step, motionOffsetX, motionOffsetY);
    const int loadsPerRow = DEGRAIN_BLK_SIZE / 16;
    if (reference && (subgroupSize % loadsPerRow) == 0) {
        const int rowsPerStep = subgroupSize / loadsPerRow;
        const int x = (lane % loadsPerRow) * 16;
        for (int y = lane / loadsPerRow; y < DEGRAIN_BLK_SIZE; y += rowsPerStep) {
            sad += degrain_motion_search_sgblock_abs_diff_sum16(
                vload16(0, (__local const uchar *)sourceBlockPixels + y * DEGRAIN_BLK_SIZE + x),
                vload16(0, reference + y * pitch + x));
        }
        return sub_group_reduce_add(sad);
    }
#endif
    for (int sadLane = lane; sadLane < DEGRAIN_BLK_SIZE; sadLane += subgroupSize) {
        sad += degrain_motion_search_accumulate_luma_sad_lane(
            sourceBlockPixels, referencePlane, subpelPlanes, subpelPlaneStride, pitch, width, height,
            blockGridX, blockGridY, step, motionOffsetX, motionOffsetY, sadLane);
    }
    return sub_group_reduce_add(sad);
}

// candidate_cost_prefersで添字順に最小を選ぶ (select_lowest_candidate_costと同じ結果)。
static inline degrain_motion_search_candidate_cost_t degrain_motion_search_sgblock_select_lowest(
    const degrain_motion_search_candidate_cost_t *candidateCosts) {
    degrain_motion_search_candidate_cost_t best = candidateCosts[0];
    for (int i = 1; i < DEGRAIN_MOTION_SEARCH_MAX_CANDIDATE_GROUPS; i++) {
        if (degrain_motion_search_candidate_cost_prefers(candidateCosts[i], best)) {
            best = candidateCosts[i];
        }
    }
    return best;
}

// wide6 (offsetCount=6) / square8 (offsetCount=8) のrefineを1回行う。
// refine_prepare_offset_candidate + refine_evaluate_candidates と同じ判定・コスト式。
static inline degrain_motion_search_candidate_cost_t degrain_motion_search_sgblock_refine(
    __local const TypePixel *sourceBlockPixels,
    __global const uchar *referencePlane,
    __global const uchar *subpelPlanes,
    const int subpelPlaneStride,
    __local const degrain_motion_search_search_context_t *context,
    degrain_motion_search_candidate_cost_t best,
    const int offsetCount,
    const int pitch,
    const int width,
    const int height,
    const int blockGridX,
    const int blockGridY,
    const int step,
    const int lane,
    const int subgroupSize) {
    degrain_motion_search_candidate_cost_t candidateCosts[DEGRAIN_MOTION_SEARCH_MAX_CANDIDATE_GROUPS];
    const int baseX = best.pos_x;
    const int baseY = best.pos_y;
    for (int i = 0; i < DEGRAIN_MOTION_SEARCH_MAX_CANDIDATE_GROUPS; i++) {
        candidateCosts[i] = degrain_motion_search_make_candidate_cost(0, 0, 0u, DEGRAIN_MOTION_SEARCH_LARGE_COST);
        if (i >= offsetCount) {
            continue;
        }
        const int2 offset = (offsetCount == 6) ? degrain_motion_search_refine_wide6_offset(i) : degrain_motion_search_refine_square_offset(i);
        const int motionOffsetX = baseX + offset.x;
        const int motionOffsetY = baseY + offset.y;
        const degrain_motion_search_candidate_t candidate = degrain_motion_search_make_candidate(motionOffsetX, motionOffsetY, 0u, 0u);
        const uint motionCost = degrain_motion_search_calc_motion_cost(candidate, baseX, baseY, context);
        // refine_prepare_offset_candidateと同じ条件で、評価する候補を決める。
        if (!degrain_motion_search_motion_inside_search_window(motionOffsetX, motionOffsetY, context) || motionCost >= best.score_primary) {
            continue;
        }
        // 候補をまとめて評価する形 (候補ごとの参照ポインタを配列に持つ) も試したが、レジスタが溢れて遅くなったため、1候補ずつ評価する。
        const uint sad = degrain_motion_search_sgblock_sad(
            sourceBlockPixels, referencePlane, subpelPlanes, subpelPlaneStride, pitch, width, height,
            blockGridX, blockGridY, step, motionOffsetX, motionOffsetY, lane, subgroupSize);
        candidateCosts[i].pos_x = (short)motionOffsetX;
        candidateCosts[i].pos_y = (short)motionOffsetY;
        candidateCosts[i].sad_metric = sad;
        candidateCosts[i].score_primary = motionCost + sad
            + degrain_motion_search_scaled_sad_penalty(sad, DEGRAIN_NEW_CANDIDATE_COST_SCALE * DEGRAIN_MOTION_SEARCH_COST_INPUT_SCALE);
    }
    const degrain_motion_search_candidate_cost_t lowest = degrain_motion_search_sgblock_select_lowest(candidateCosts);
    return degrain_motion_search_candidate_cost_prefers(lowest, best) ? lowest : best;
}

// finalize_candidate_cost_parallel と同じ。分散・SADはいずれも整数の総和なので集約順序に依らない。
static inline degrain_motion_search_candidate_cost_t degrain_motion_search_sgblock_finalize(
    __local const TypePixel *sourceBlockPixels,
    __global const uchar *referencePlane,
    __global const uchar *subpelPlanes,
    const int subpelPlaneStride,
    const int pitch,
    const int width,
    const int height,
    const int blockGridX,
    const int blockGridY,
    const int step,
    degrain_motion_search_candidate_cost_t best,
    const int lane,
    const int subgroupSize) {
    best.score_primary = best.sad_metric;
    const int count = DEGRAIN_BLK_SIZE * DEGRAIN_BLK_SIZE;
    ulong partialSum = 0;
    ulong partialSumSq = 0;
    for (int i = lane; i < count; i += subgroupSize) {
        const ulong value = (ulong)sourceBlockPixels[i];
        partialSum += value;
        partialSumSq += value * value;
    }
    const long sum = (long)sub_group_reduce_add(partialSum);
    const long sumSq = (long)sub_group_reduce_add(partialSumSq);
    const long mean = sum / count;
    const long varianceNumer = sumSq - mean * sum;
    const uint variance = (varianceNumer <= 0) ? 0u : (uint)(varianceNumer / count);
    const int isFlat = variance == 0u;
    if (isFlat) {
        const uint sadZero = degrain_motion_search_sgblock_sad(
            sourceBlockPixels, referencePlane, subpelPlanes, subpelPlaneStride, pitch, width, height,
            blockGridX, blockGridY, step, 0, 0, lane, subgroupSize);
        best.pos_x = 0;
        best.pos_y = 0;
        best.sad_metric = sadZero;
        best.score_primary = sadZero;
    }
#if DEGRAIN_MOTION_SEARCH_PASS_FLAT_FLAG
    if (isFlat) {
        best.score_primary |= DEGRAIN_MOTION_SEARCH_FLAT_FLAG;
    }
#endif
    return best;
}

// 引数はkernel_degrain_mv_search_parallelと同じ。global sizeは
// (blockCount / (local size / subgroup幅) の切り上げ) * local size とすること。
__attribute__((reqd_work_group_size(DEGRAIN_MOTION_SEARCH_SGBLOCK_LOCAL_SIZE, 1, 1)))
#ifdef DEGRAIN_MOTION_SEARCH_SGBLOCK_REQD_SG
__attribute__((intel_reqd_sub_group_size(DEGRAIN_MOTION_SEARCH_SGBLOCK_REQD_SG)))
#endif
__kernel void kernel_degrain_mv_search_sgblock(
    __global const uchar *sourcePlane,
    __global const uchar *referencePlane,
    __global const uchar *subpelPlanes,
    const int subpelPlaneStride,
    __global degrain_mv_internal_t *vectors,
    const int pitch,
    const int width,
    const int height,
    const int planeBase,
    const int blockCount,
    const int blocksX,
    const int blocksY,
    const int step) {
    __local TypePixel sourceBlockPixelsAll[DEGRAIN_MOTION_SEARCH_SGBLOCK_MAX_BLOCKS][DEGRAIN_BLK_SIZE * DEGRAIN_BLK_SIZE];
    __local degrain_motion_search_search_context_t contextAll[DEGRAIN_MOTION_SEARCH_SGBLOCK_MAX_BLOCKS];

    const int subgroupId = (int)get_sub_group_id();
    const int lane = (int)get_sub_group_local_id();
    const int subgroupSize = (int)get_sub_group_size();
    const int block = (int)get_group_id(0) * (int)get_num_sub_groups() + subgroupId;
    const int kernelWidth = degrain_motion_search_const_width(width);
    const int kernelHeight = degrain_motion_search_const_height(height);
    const int kernelBlocksX = degrain_motion_search_const_blocks_x(blocksX);
    const int kernelBlocksY = degrain_motion_search_const_blocks_y(blocksY);
    const int kernelStep = degrain_motion_search_const_step(step);
    // subgroup単位で一様な条件なので、work-groupのbarrierを使わない本カーネルでは早期returnしてよい。
    if (sourcePlane == 0 || referencePlane == 0 || kernelBlocksX <= 0 || kernelBlocksY <= 0 || block >= blockCount
        || subgroupId >= DEGRAIN_MOTION_SEARCH_SGBLOCK_MAX_BLOCKS) {
        return;
    }
    const int blockGridX = block % kernelBlocksX;
    const int blockGridY = block / kernelBlocksX;
    if (blockGridY >= kernelBlocksY) {
        return;
    }
    __local TypePixel *sourceBlockPixels = sourceBlockPixelsAll[subgroupId];
    __local degrain_motion_search_search_context_t *context = &contextAll[subgroupId];

    const int sourceBaseX = blockGridX * kernelStep;
    const int sourceBaseY = blockGridY * kernelStep;
    for (int i = lane; i < DEGRAIN_BLK_SIZE * DEGRAIN_BLK_SIZE; i += subgroupSize) {
        const int x = i % DEGRAIN_BLK_SIZE;
        const int y = i / DEGRAIN_BLK_SIZE;
        sourceBlockPixels[i] = (TypePixel)degrain_pixel_load(sourcePlane, pitch, kernelWidth, kernelHeight, sourceBaseX + x, sourceBaseY + y);
    }
    const degrain_mv_internal_t initialSeed = vectors[degrain_motion_search_vec_current_index(planeBase, blockCount, block)];
    if (lane == 0) {
        *context = degrain_motion_search_make_search_context(initialSeed, width, height, blockGridX, blockGridY, kernelStep);
    }
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);

    // 初期候補 (zero / frame average / seed)。重複する候補は先に評価した同一位置のSADを使う。
    degrain_motion_search_candidate_t candidate[DEGRAIN_MOTION_SEARCH_SEARCH_CANDIDATES];
    degrain_motion_search_candidate_cost_t candidateCosts[DEGRAIN_MOTION_SEARCH_MAX_CANDIDATE_GROUPS];
    for (int i = 0; i < DEGRAIN_MOTION_SEARCH_MAX_CANDIDATE_GROUPS; i++) {
        candidateCosts[i] = degrain_motion_search_make_candidate_cost(0, 0, 0u, DEGRAIN_MOTION_SEARCH_LARGE_COST);
    }
    for (int c = 0; c < DEGRAIN_MOTION_SEARCH_SEARCH_CANDIDATES; c++) {
        candidate[c] = degrain_motion_search_load_base_candidate(vectors, context, c, planeBase, blockCount, block);
        int canonical = c;
        for (int i = 0; i < c; i++) {
            if (candidate[i].pos_x == candidate[c].pos_x && candidate[i].pos_y == candidate[c].pos_y) {
                canonical = i;
                break;
            }
        }
        const uint sad = (canonical == c)
            ? degrain_motion_search_sgblock_sad(
                sourceBlockPixels, referencePlane, subpelPlanes, subpelPlaneStride, pitch, kernelWidth, kernelHeight,
                blockGridX, blockGridY, kernelStep, candidate[c].pos_x, candidate[c].pos_y, lane, subgroupSize)
            : candidateCosts[canonical].sad_metric;
        candidateCosts[c].pos_x = candidate[c].pos_x;
        candidateCosts[c].pos_y = candidate[c].pos_y;
        candidateCosts[c].sad_metric = sad;
        candidateCosts[c].score_primary = sad + degrain_motion_search_scaled_sad_penalty(sad, degrain_motion_search_initial_cost_scale(c));
    }
    degrain_motion_search_candidate_cost_t best = degrain_motion_search_sgblock_select_lowest(candidateCosts);

#if DEGRAIN_MOTION_SEARCH_EARLY_SAD_THRESHOLD >= 0
    const int skipFullSearch = (DEGRAIN_MOTION_SEARCH_EARLY_SAD_THRESHOLD == 0)
        ? (best.sad_metric == 0u)
        : (best.sad_metric < (uint)DEGRAIN_MOTION_SEARCH_EARLY_SAD_THRESHOLD);
#else
    const int skipFullSearch = 0;
#endif
    if (!skipFullSearch) {
#if DEGRAIN_SEARCH_PARAM >= 2
        best = degrain_motion_search_sgblock_refine(
            sourceBlockPixels, referencePlane, subpelPlanes, subpelPlaneStride, context, best, 6, pitch, kernelWidth, kernelHeight,
            blockGridX, blockGridY, kernelStep, lane, subgroupSize);
#endif
        best = degrain_motion_search_sgblock_refine(
            sourceBlockPixels, referencePlane, subpelPlanes, subpelPlaneStride, context, best, 8, pitch, kernelWidth, kernelHeight,
            blockGridX, blockGridY, kernelStep, lane, subgroupSize);
    }

    best = degrain_motion_search_sgblock_finalize(
        sourceBlockPixels, referencePlane, subpelPlanes, subpelPlaneStride, pitch, kernelWidth, kernelHeight,
        blockGridX, blockGridY, kernelStep, best, lane, subgroupSize);
    if (lane == 0) {
        vectors[degrain_motion_search_vec_current_index(planeBase, blockCount, block)] =
            degrain_motion_search_candidate_cost_to_saved_vector(best);
    }
}
#endif // DEGRAIN_MOTION_SEARCH_SGBLOCK


#endif
