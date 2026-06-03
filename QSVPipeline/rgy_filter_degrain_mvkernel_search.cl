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
    int sad = 0;
    if (useFastPath) {
#if DEGRAIN_PIXEL_BYTES == 1
        __local const uchar *sourcePtr = (__local const uchar *)(sourceBlockPixels + firstLaneRow * DEGRAIN_BLK_SIZE + x);
        const int referenceStridePerLane = rowsPerLane * refPitch;
        __global const uchar *referencePtr = referencePlane + (referenceY + firstLaneRow) * refPitch + referenceX + x;
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

    for (int offset = DEGRAIN_BLK_SIZE >> 1; offset > 0; offset >>= 1) {
        if (candidateIsValid && sadLane < offset) {
            candidateLaneSums[partialBase + sadLane] += candidateLaneSums[partialBase + sadLane + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return (candidateIsValid && sadLane == 0) ? candidateLaneSums[partialBase] : 0u;
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
            && candidateCosts[localThreadId + stride].score_primary < candidateCosts[localThreadId].score_primary) {
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

    if (localThreadId == 0 && candidateCosts[0].score_primary < bestCandidateCost->score_primary) {
        *bestCandidateCost = candidateCosts[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

static inline void degrain_motion_search_refine_prepared_candidates(
    __local const TypePixel *sourceBlockPixels,
    __global const uchar *referencePlane,
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
        sourceBlockPixels, referencePlane, context, candidateCosts, candidateLaneSums, bestCandidateCost,
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
        sourceBlockPixels, referencePlane, context, candidateCosts, candidateLaneSums, bestCandidateCost,
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
        sourceBlockPixels, referencePlane, context, candidateCosts, candidateLaneSums, bestCandidateCost,
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
            sourceBlockPixels, referencePlane, context, candidateCosts, candidateLaneSums, bestCandidateCost,
            localThreadId, sadLane, candidateGroupIndex, 6, blockGridX, blockGridY, step, pitch, width, height,
            referenceWindowPixels, referenceWindowX, referenceWindowY, referenceWindowIsValid);
    }
#else
    degrain_motion_search_refine_hex2(
        sourceBlockPixels,
        referencePlane,
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
            sourceBlockPixels, referencePlane, context, candidateCosts, candidateLaneSums, bestCandidateCost,
            localThreadId, sadLane, candidateGroupIndex, 8, blockGridX, blockGridY, step, pitch, width, height,
            referenceWindowPixels, referenceWindowX, referenceWindowY, referenceWindowIsValid);
    }
#else
    degrain_motion_search_refine_square8(
        sourceBlockPixels,
        referencePlane,
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

    if (localThreadId == 0) {
        vectors[degrain_motion_search_vec_current_index(planeBase, blockCount, block)] =
            degrain_motion_search_candidate_cost_to_saved_vector(*bestCandidateCost);
    }
}

__kernel void kernel_degrain_mv_search_parallel(
    __global const uchar *sourcePlane,
    __global const uchar *referencePlane,
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
    __local uint candidateLaneSums[DEGRAIN_MOTION_SEARCH_SEARCH_LOCAL_SIZE];
#if DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE && DEGRAIN_PIXEL_BYTES == 1
    __local uchar referenceWindowPixels[DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE * DEGRAIN_MOTION_SEARCH_REF_LOCAL_CACHE_SIZE];
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
        const degrain_motion_search_saved_vector_t base =
            vectors[degrain_motion_search_vec_current_index(planeBase, blockCount, block)];
        const degrain_motion_search_candidate_t baseCandidate =
            degrain_motion_search_saved_vector_to_candidate(base);
        bestCandidateCost.pos_x = base.pos_x;
        bestCandidateCost.pos_y = base.pos_y;
        bestCandidateCost.sad_metric = base.sad_metric;
        bestCandidateCost.score_primary = base.score_primary;
        candidate[0] = degrain_motion_search_constrain_candidate(baseCandidate, &context);
        // candidate[0] is the constrained base vector; reuse its SAD only when bounds keep it unchanged.
        reusePreviousSad = DEGRAIN_MOTION_SEARCH_SPATIAL_REUSE_PREVIOUS_SAD
            && candidate[0].pos_x == base.pos_x
            && candidate[0].pos_y == base.pos_y;
        candidate[1] = (blockGridX > 0)
            ? degrain_motion_search_constrain_candidate(degrain_motion_search_saved_vector_to_candidate(vectors[degrain_motion_search_vec_current_index(planeBase, blockCount, block - 1)]), &context)
            : candidate[0];
        candidate[2] = (blockGridY > 0)
            ? degrain_motion_search_constrain_candidate(degrain_motion_search_saved_vector_to_candidate(vectors[degrain_motion_search_vec_current_index(planeBase, blockCount, block - kernelBlocksX)]), &context)
            : candidate[0];
        candidate[3] = (blockGridX + 1 < kernelBlocksX && blockGridY + 1 < kernelBlocksY)
            ? degrain_motion_search_constrain_candidate(degrain_motion_search_saved_vector_to_candidate(vectors[degrain_motion_search_vec_current_index(planeBase, blockCount, block + kernelBlocksX + 1)]), &context)
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

    if (localThreadId == 0 && candidateCosts[0].score_primary < bestCandidateCost.score_primary) {
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
            sourceBlockPixels, referencePlane, &context, candidateCosts, candidateLaneSums, &bestCandidateCost,
            localThreadId, sadLane, candidateGroupIndex, 8, blockGridX, blockGridY, kernelStep, pitch, kernelWidth, kernelHeight,
            referenceWindowPixels, referenceWindowX, referenceWindowY, referenceWindowIsValid);
    }
#else
    degrain_motion_search_refine_square8(
        sourceBlockPixels,
        referencePlane,
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

    if (localThreadId == 0) {
        vectorsFinal[degrain_motion_search_vec_final_index(finalBase, blockCount, block)] =
            degrain_motion_search_candidate_cost_to_saved_vector(bestCandidateCost);
    }
}

#endif
