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

#define RGY_FLT_EPS (1.192092896e-07F) // from float.h

#ifndef Type
#define Type uchar
#endif
#ifndef Type2
#define Type2 uchar2
#endif
#ifndef Type4
#define Type4 uchar4
#endif
#ifndef Type8
#define Type8 uchar8
#endif
#ifndef RNNEDI_BIT_DEPTH
#define RNNEDI_BIT_DEPTH 8
#endif

#ifndef RNNEDI_PRED_XDIA
#error "RNNEDI_PRED_XDIA must be defined"
#endif
#ifndef RNNEDI_PRED_YDIA
#error "RNNEDI_PRED_YDIA must be defined"
#endif
#ifndef RNNEDI_PRED_K
#error "RNNEDI_PRED_K must be defined"
#endif
#ifndef RNNEDI_PRED_NNS
#error "RNNEDI_PRED_NNS must be defined"
#endif
#ifndef RNNEDI_PRED_QUAL
#error "RNNEDI_PRED_QUAL must be defined"
#endif
#ifndef RNNEDI_PRED_SUBGROUP_OPT
#define RNNEDI_PRED_SUBGROUP_OPT 0
#endif
#ifndef RNNEDI_PRED_SUBGROUP_SIZE
#define RNNEDI_PRED_SUBGROUP_SIZE 0
#endif
#if RNNEDI_PRED_K != (RNNEDI_PRED_XDIA * RNNEDI_PRED_YDIA)
#error "RNNEDI_PRED_K must match RNNEDI_PRED_XDIA * RNNEDI_PRED_YDIA"
#endif
#if !(RNNEDI_PRED_XDIA == 8 || RNNEDI_PRED_XDIA == 16 || RNNEDI_PRED_XDIA == 32 || RNNEDI_PRED_XDIA == 48)
#error "RNNEDI_PRED_XDIA must be one of 8, 16, 32, 48"
#endif
#if !(RNNEDI_PRED_YDIA == 4 || RNNEDI_PRED_YDIA == 6)
#error "RNNEDI_PRED_YDIA must be one of 4, 6"
#endif
#if !(RNNEDI_PRED_NNS == 16 || RNNEDI_PRED_NNS == 32 || RNNEDI_PRED_NNS == 64 || RNNEDI_PRED_NNS == 128 || RNNEDI_PRED_NNS == 256)
#error "RNNEDI_PRED_NNS must be one of 16, 32, 64, 128, 256"
#endif
#if !(RNNEDI_PRED_QUAL == 1 || RNNEDI_PRED_QUAL == 2)
#error "RNNEDI_PRED_QUAL must be one of 1, 2"
#endif

#define RNNEDI_PRED_LOCAL_X 16
#define RNNEDI_PRED_LOCAL_Y 32
#if (RNNEDI_PRED_NNS % RNNEDI_PRED_LOCAL_X) != 0
#error "RNNEDI_PRED_NNS must be divisible by RNNEDI_PRED_LOCAL_X"
#endif
#define RNNEDI_PRED_GROUPS (RNNEDI_PRED_NNS / RNNEDI_PRED_LOCAL_X)
#define RNNEDI_PRED_BLOCK_FLOAT2_COUNT (RNNEDI_PRED_K * RNNEDI_PRED_LOCAL_X)
#define RNNEDI_PRED_QUAL_BODY_FLOAT2_COUNT (RNNEDI_PRED_GROUPS * RNNEDI_PRED_BLOCK_FLOAT2_COUNT)
#define RNNEDI_PRED_BODY_FLOAT2_COUNT (RNNEDI_PRED_QUAL * RNNEDI_PRED_QUAL_BODY_FLOAT2_COUNT)
#define RNNEDI_TILE_GROUPS_X 32
#define RNNEDI_TILE_ROWS 16
#define RNNEDI_TILE_PIXELS_X (RNNEDI_TILE_GROUPS_X * 4)
#define RNNEDI_TILE_MASK_COUNT (RNNEDI_TILE_GROUPS_X * RNNEDI_TILE_ROWS)
#define RNNEDI_TILE_MAX_CANDIDATES (RNNEDI_TILE_MASK_COUNT * 4)
#define RNNEDI_PRE_HIDDEN_COUNT 4
#define RNNEDI_PRE_OUTPUT_LANES 4
#define RNNEDI_PRE_SAMPLE_ROWS 4
#define RNNEDI_PRE_ROW_TAPS 16
#define RNNEDI_PRE_CENTER_TAP_OFFSET 2
#define RNNEDI_PRE_SAMPLE_COUNT (RNNEDI_PRE_SAMPLE_ROWS * RNNEDI_PRE_ROW_TAPS)
#define RNNEDI_PRE_HIDDEN_WEIGHT4_OFFSET 0
#define RNNEDI_PRE_HIDDEN_SCALE4_INDEX (RNNEDI_PRE_HIDDEN_WEIGHT4_OFFSET + RNNEDI_PRE_SAMPLE_COUNT)
#define RNNEDI_PRE_HIDDEN_BIAS4_INDEX (RNNEDI_PRE_HIDDEN_SCALE4_INDEX + 1)
#define RNNEDI_PRE_OUTPUT_MIX4_OFFSET (RNNEDI_PRE_HIDDEN_BIAS4_INDEX + 1)
#define RNNEDI_PRE_OUTPUT_BIAS4_INDEX (RNNEDI_PRE_OUTPUT_MIX4_OFFSET + RNNEDI_PRE_OUTPUT_LANES)
#define RNNEDI_PRE_WEIGHT_FLOAT4_COUNT (RNNEDI_PRE_OUTPUT_BIAS4_INDEX + 1)
#define RNNEDI_PRED_PATCH_AT(row, col) ((row) * RNNEDI_PRED_K + (col))
#define RNNEDI_PRED_PART_AT(row, col)  ((row) * RNNEDI_PRED_LOCAL_X + (col))
#define RNNEDI_PRED_SOFTMAX_DENOM_EPS (1.0e-10f)
#define RNNEDI_PRED_ELLIOTT_OUTPUT_SCALE (5.0f)

#if RNNEDI_PRED_SUBGROUP_OPT
#if !(RNNEDI_PRED_SUBGROUP_SIZE == 16 || RNNEDI_PRED_SUBGROUP_SIZE == 32)
#error "RNNEDI_PRED_SUBGROUP_SIZE must be 16 or 32 when RNNEDI_PRED_SUBGROUP_OPT is enabled"
#endif
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

static inline int rnnedi_mirror_index(const int pos, const int length) {
    if (length <= 0) {
        return 0;
    }
    if (pos < 0) {
        return -pos - 1;
    }
    if (pos >= length) {
        return length - (pos - length) - 1;
    }
    return pos;
}
static inline void rnnedi_load_patch_distributed(
    __local Type *restrict patch,
    const __global uchar *restrict pRef, const int refPitch, const int refOffset,
    const int x, const int y, const bool active, const int tx
) {
#if RNNEDI_PRED_XDIA == 8
        for (int chunk = tx; chunk < (RNNEDI_PRED_YDIA << 2); chunk += 16) {
            const int row = chunk >> 2;
            const int col = (chunk & 3) << 1;
            __local Type *dst = patch + row * 8 + col;
            const __global Type *src = (const __global Type *)(pRef + refOffset + (y + row) * refPitch + (x + col) * (int)sizeof(Type));
            Type2 v = (Type2)(0);
            if (active) {
                v = vload2(0, src);
            }
            vstore2(v, 0, dst);
        }
#else
    const int chunksPerRow = RNNEDI_PRED_XDIA >> 3;
    const int chunks = chunksPerRow * RNNEDI_PRED_YDIA;
    for (int chunk = tx; chunk < chunks; chunk += 16) {
        const int row = chunk / chunksPerRow;
        const int col = (chunk - row * chunksPerRow) << 3;
        __local Type *dst = patch + row * RNNEDI_PRED_XDIA + col;
        const __global Type *src = (const __global Type *)(pRef + refOffset + (y + row) * refPitch + (x + col) * (int)sizeof(Type));
        Type8 v = (Type8)(0);
        if (active) {
            v = vload8(0, src);
        }
        vstore8(v, 0, dst);
    }
#endif
}

__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void kernel_rnnedi_pad_ref_and_copy_half_scalar(
    __global uchar *restrict pDst, const int dstPitch, const int dstOffset,
    __global uchar *restrict pRef, const int refPitch, const int refOffset,
    const __global uchar *restrict pSrc, const int srcPitch, const int srcOffset,
    const int width, const int height,
    const int hpad, const int vpad
) {
    const int xBase = (get_global_id(0) << 2) - hpad;
    const int y = get_global_id(1) - vpad;
    const int paddedWidth = ((width + hpad * 2 + 3) >> 2) << 2;

    if (xBase >= paddedWidth - hpad || y >= height + vpad) {
        return;
    }

    const bool pady = (y < 0 || y >= height);
    const int srcy = rnnedi_mirror_index(y, height);
    if (xBase >= 0 && xBase + 3 < width) {
        const __global Type *src = (const __global Type *)(pSrc + srcOffset + srcy * srcPitch + xBase * (int)sizeof(Type));
        const Type4 packed = vload4(0, src);
        __global Type *ref = (__global Type *)(pRef + refOffset + (y + vpad) * refPitch + (xBase + hpad) * (int)sizeof(Type));
        vstore4(packed, 0, ref);
        if (!pady) {
            __global Type *dst = (__global Type *)(pDst + dstOffset + y * dstPitch + xBase * (int)sizeof(Type));
            vstore4(packed, 0, dst);
        }
        return;
    }

    for (int lane = 0; lane < 4; lane++) {
        const int x = xBase + lane;
        if (x >= paddedWidth - hpad) {
            continue;
        }
        const bool padx = (x < 0 || x >= width);
        const int srcx = rnnedi_mirror_index(x, width);
        const Type v = *(const __global Type *)(pSrc + srcOffset + srcy * srcPitch + srcx * (int)sizeof(Type));

        *(__global Type *)(pRef + refOffset + (y + vpad) * refPitch + (x + hpad) * (int)sizeof(Type)) = v;

        if (!padx && !pady) {
            *(__global Type *)(pDst + dstOffset + y * dstPitch + x * (int)sizeof(Type)) = v;
        }
    }
}

static inline int4 rnnedi_to_int4(const Type4 v) {
    return (int4)(v.x, v.y, v.z, v.w);
}

// expf overflow guard: clamp the upper end below the float32 overflow point
// (~88.72). Underflow toward 0 is acceptable in the softmax-style ratio below.
static inline float rnnedi_expf(const float f) {
    return native_exp(min(f, 88.0f));
}

static inline float rnnedi_reduce16_sum(const __local float *v) {
    float s = 0.0f;
    for (int i = 0; i < 16; i++) {
        s += v[i];
    }
    return s;
}

#if RNNEDI_PRED_SUBGROUP_OPT
static inline float rnnedi_subgroup_reduce16_float(const float v) {
#if RNNEDI_PRED_SUBGROUP_SIZE == 16
    return sub_group_reduce_add(v);
#else
    const uint lane = get_sub_group_local_id();
    const float lo = sub_group_reduce_add((lane < 16u) ? v : 0.0f);
    const float hi = sub_group_reduce_add((lane >= 16u) ? v : 0.0f);
    return (lane < 16u) ? lo : hi;
#endif
}

static inline int rnnedi_subgroup_reduce16_int(const int v) {
#if RNNEDI_PRED_SUBGROUP_SIZE == 16
    return sub_group_reduce_add(v);
#else
    const uint lane = get_sub_group_local_id();
    const int lo = sub_group_reduce_add((lane < 16u) ? v : 0);
    const int hi = sub_group_reduce_add((lane >= 16u) ? v : 0);
    return (lane < 16u) ? lo : hi;
#endif
}

static inline float rnnedi_subgroup_broadcast16_float(const float v) {
#if RNNEDI_PRED_SUBGROUP_SIZE == 16
    return sub_group_broadcast(v, 0u);
#else
    const uint lane = get_sub_group_local_id();
    const float lo = sub_group_broadcast(v, 0u);
    const float hi = sub_group_broadcast(v, 16u);
    return (lane < 16u) ? lo : hi;
#endif
}
#endif

static inline int rnnedi_round_to_pixel_match(const float value) {
    return (int)(value + 0.5f);
}

#if RNNEDI_BIT_DEPTH <= 8
static inline void rnnedi_predictor_lane_patch_moments(
    const __local Type *restrict patch,
    __local int *restrict sumPart,
    __local int *restrict sumsqPart,
    const int row,
    const int lane
) {
    int sum = 0;
    int sumsq = 0;
    for (int k = lane; k < RNNEDI_PRED_K; k += RNNEDI_PRED_LOCAL_X) {
        const int v = (int)patch[k];
        sum += v;
        sumsq += v * v;
    }
    sumPart[RNNEDI_PRED_PART_AT(row, lane)] = sum;
    sumsqPart[RNNEDI_PRED_PART_AT(row, lane)] = sumsq;
}

static inline void rnnedi_predictor_finalize_patch_stats(
    __local int *restrict sumPart,
    __local int *restrict sumsqPart,
    __local float *restrict avg,
    __local float *restrict stddev,
    __local float *restrict invvar,
    const int row
) {
    int sumAll = 0;
    int sumsqAll = 0;
    __local int *sumPartRow = sumPart + row * RNNEDI_PRED_LOCAL_X;
    __local int *sumsqPartRow = sumsqPart + row * RNNEDI_PRED_LOCAL_X;
    for (int i = 0; i < RNNEDI_PRED_LOCAL_X; i++) {
        sumAll += sumPartRow[i];
        sumsqAll += sumsqPartRow[i];
    }
    const float scale = 1.0f / (float)RNNEDI_PRED_K;
    const float avg_ = (float)sumAll * scale;
    float stddev_ = (float)sumsqAll * scale - avg_ * avg_;
    float invvar_ = 0.0f;
    if (stddev_ <= RGY_FLT_EPS) {
        stddev_ = 0.0f;
    } else {
        stddev_ = sqrt(stddev_);
        invvar_ = 1.0f / stddev_;
    }
    avg[row] = avg_;
    stddev[row] = stddev_;
    invvar[row] = invvar_;
}
#else
static inline void rnnedi_predictor_lane_patch_sum(
    const __local Type *restrict patch,
    __local int *restrict sumPart,
    const int row,
    const int lane
) {
    int sum = 0;
    for (int k = lane; k < RNNEDI_PRED_K; k += RNNEDI_PRED_LOCAL_X) {
        sum += (int)patch[k];
    }
    sumPart[RNNEDI_PRED_PART_AT(row, lane)] = sum;
}

static inline void rnnedi_predictor_finalize_patch_avg(
    __local int *restrict sumPart,
    __local float *restrict avg,
    const int row
) {
    int sumAll = 0;
    __local int *sumPartRow = sumPart + row * RNNEDI_PRED_LOCAL_X;
    for (int i = 0; i < RNNEDI_PRED_LOCAL_X; i++) {
        sumAll += sumPartRow[i];
    }
    avg[row] = (float)sumAll * (1.0f / (float)RNNEDI_PRED_K);
}

static inline void rnnedi_predictor_lane_patch_variance(
    const __local Type *restrict patch,
    __local float *restrict varPart,
    const int row,
    const int lane,
    const float avg
) {
    float sumsq = 0.0f;
    for (int k = lane; k < RNNEDI_PRED_K; k += RNNEDI_PRED_LOCAL_X) {
        const float diff = (float)patch[k] - avg;
        sumsq = fma(diff, diff, sumsq);
    }
    varPart[RNNEDI_PRED_PART_AT(row, lane)] = sumsq;
}

static inline void rnnedi_predictor_finalize_patch_stddev(
    __local float *restrict varPart,
    __local float *restrict stddev,
    __local float *restrict invvar,
    const int row
) {
    const int partBase = row * RNNEDI_PRED_LOCAL_X;
    float stddev_ = rnnedi_reduce16_sum(&varPart[partBase]) * (1.0f / (float)RNNEDI_PRED_K);
    float invvar_ = 0.0f;
    if (stddev_ <= RGY_FLT_EPS) {
        stddev_ = 0.0f;
    } else {
        stddev_ = sqrt(stddev_);
        invvar_ = 1.0f / stddev_;
    }
    stddev[row] = stddev_;
    invvar[row] = invvar_;
}
#endif

static inline float2 rnnedi_predictor_lane_vote(
    const __local Type *restrict patch,
    const __global float2 *restrict weightsBody,
    const __global float2 *restrict weightsBias,
    const int tx,
    const int q,
    const float invvar
) {
    float weightedElliottVoteSum = 0.0f;
    float softmaxVoteWeightSum = 0.0f;
    const __global float2 *neuronBlockWeights = weightsBody + q * RNNEDI_PRED_QUAL_BODY_FLOAT2_COUNT;
    const __global float2 *neuronBiasPtr = weightsBias + q * RNNEDI_PRED_NNS + tx;
    for (int neuronGroup = 0; neuronGroup < RNNEDI_PRED_GROUPS; neuronGroup++) {
        // Weights are repacked into a contiguous fp32 layout for coalesced vector loads.
        // The predictor body is stored as [q][neuronBlock16][sample][lane], so
        // tx=0..15 read adjacent float2 values for each sample.
        const __global float2 *sampleWeights = neuronBlockWeights + tx;
        const __local Type *patchPtr = patch;
        float2 weightedPatchSums = (float2)(0.0f, 0.0f);
        for (int sampleIndex = 0; sampleIndex < RNNEDI_PRED_K; sampleIndex++) {
            const int patchPixelValue = (int)(*patchPtr);
            weightedPatchSums = fma((float)patchPixelValue, *sampleWeights, weightedPatchSums);
            patchPtr++;
            sampleWeights += RNNEDI_PRED_LOCAL_X;
        }

        const float2 neuronBias = *neuronBiasPtr;
        const float softmaxLogit = fma(weightedPatchSums.x, invvar, neuronBias.x);
        const float elliottInput = fma(weightedPatchSums.y, invvar, neuronBias.y);
        const float softmaxVoteWeight = rnnedi_expf(softmaxLogit);
        weightedElliottVoteSum += softmaxVoteWeight * (elliottInput / (1.0f + fabs(elliottInput)));
        softmaxVoteWeightSum += softmaxVoteWeight;
        neuronBlockWeights += RNNEDI_PRED_BLOCK_FLOAT2_COUNT;
        neuronBiasPtr += RNNEDI_PRED_LOCAL_X;
    }
    return (float2)(weightedElliottVoteSum, softmaxVoteWeightSum);
}

static inline float rnnedi_predictor_merge_votes(
    const __local float *restrict weightedElliottVoteParts,
    const __local float *restrict softmaxVoteWeightParts,
    const int row,
    const float avg,
    const float stddev,
    const float result
) {
    const int partBase = row * RNNEDI_PRED_LOCAL_X;
    const float weightedElliottVoteSum = rnnedi_reduce16_sum(&weightedElliottVoteParts[partBase]);
    const float softmaxVoteWeightSum = rnnedi_reduce16_sum(&softmaxVoteWeightParts[partBase]);
    if (softmaxVoteWeightSum > RNNEDI_PRED_SOFTMAX_DENOM_EPS) {
        return result + ((RNNEDI_PRED_ELLIOTT_OUTPUT_SCALE * weightedElliottVoteSum) / softmaxVoteWeightSum) * stddev + avg;
    }
    return result + avg;
}

#if RNNEDI_PRED_SUBGROUP_OPT
static inline float rnnedi_predictor_merge_vote_sums(
    const float weightedElliottVoteSum,
    const float softmaxVoteWeightSum,
    const float avg,
    const float stddev,
    const float result
) {
    if (softmaxVoteWeightSum > RNNEDI_PRED_SOFTMAX_DENOM_EPS) {
        return result + ((RNNEDI_PRED_ELLIOTT_OUTPUT_SCALE * weightedElliottVoteSum) / softmaxVoteWeightSum) * stddev + avg;
    }
    return result + avg;
}
#endif

static inline void rnnedi_predictor_write_result(
    __global uchar *restrict pDst,
    const int dstPitch,
    const int dstOffset,
    const int x,
    const int y,
    const float result,
    const int valMin,
    const int valMax
) {
    const float scale = 1.0f / (float)RNNEDI_PRED_QUAL;
    *(__global Type *)(pDst + dstOffset + y * dstPitch + x * (int)sizeof(Type)) = (Type)clamp(rnnedi_round_to_pixel_match(result * scale), valMin, valMax);
}

static inline void rnnedi_prescreen_cache_weights(
    __local float4 *restrict prescreenWeights,
    const __global float *restrict weights,
    const int tid
) {
    // Prescreener weights are a compact fp32 MLP block:
    // [sample][hidden4], hidden scale/bias, [outputLane][hidden4], output bias.
    // Caching it as one float4 array keeps the kernel layout tied to GPU vector
    // reads instead of the source weight-file prescreener packing.
    const __global float4 *weights4 = (const __global float4 *)weights;
    for (int i = tid; i < RNNEDI_PRE_WEIGHT_FLOAT4_COUNT; i += RNNEDI_TILE_MASK_COUNT) {
        prescreenWeights[i] = weights4[i];
    }
}

static inline float4 rnnedi_prescreen_accumulate_aligned_window4(
    const int4 v0,
    const int4 v1,
    const int4 v2,
    const int4 v3,
    const int4 v4,
    const __local float4 *restrict weights
) {
    float4 sum = (float4)(0.0f);
    sum = fma((float4)((float)v0.z), weights[ 0], sum);
    sum = fma((float4)((float)v0.w), weights[ 1], sum);
    sum = fma((float4)((float)v1.x), weights[ 2], sum);
    sum = fma((float4)((float)v1.y), weights[ 3], sum);
    sum = fma((float4)((float)v1.z), weights[ 4], sum);
    sum = fma((float4)((float)v1.w), weights[ 5], sum);
    sum = fma((float4)((float)v2.x), weights[ 6], sum);
    sum = fma((float4)((float)v2.y), weights[ 7], sum);
    sum = fma((float4)((float)v2.z), weights[ 8], sum);
    sum = fma((float4)((float)v2.w), weights[ 9], sum);
    sum = fma((float4)((float)v3.x), weights[10], sum);
    sum = fma((float4)((float)v3.y), weights[11], sum);
    sum = fma((float4)((float)v3.z), weights[12], sum);
    sum = fma((float4)((float)v3.w), weights[13], sum);
    sum = fma((float4)((float)v4.x), weights[14], sum);
    sum = fma((float4)((float)v4.y), weights[15], sum);
    return sum;
}

static inline float4 rnnedi_prescreen_hidden_response4(
    const __global Type4 *restrict ref4,
    const int refPitch4,
    const int xbase,
    const int ybase,
    const __local float4 *restrict prescreenWeights
) {
    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (int y = 0; y < RNNEDI_PRE_SAMPLE_ROWS; y++) {
        const int refRow = (y + ybase) * refPitch4;
        const int4 v0 = rnnedi_to_int4(ref4[xbase + 0 + refRow]);
        const int4 v1 = rnnedi_to_int4(ref4[xbase + 1 + refRow]);
        const int4 v2 = rnnedi_to_int4(ref4[xbase + 2 + refRow]);
        const int4 v3 = rnnedi_to_int4(ref4[xbase + 3 + refRow]);
        const int4 v4 = rnnedi_to_int4(ref4[xbase + 4 + refRow]);
        const __local float4 *rowWeights = prescreenWeights + RNNEDI_PRE_HIDDEN_WEIGHT4_OFFSET + y * RNNEDI_PRE_ROW_TAPS;
        sum += rnnedi_prescreen_accumulate_aligned_window4(v0, v1, v2, v3, v4, rowWeights);
    }
    return sum;
}

static inline float4 rnnedi_prescreen_hidden_activation4(
    const float4 hiddenDot,
    const __local float4 *restrict prescreenWeights
) {
    const float4 hiddenInput = fma(
        hiddenDot,
        prescreenWeights[RNNEDI_PRE_HIDDEN_SCALE4_INDEX],
        prescreenWeights[RNNEDI_PRE_HIDDEN_BIAS4_INDEX]);
    return hiddenInput / (fabs(hiddenInput) + 1.0f);
}

static inline float4 rnnedi_prescreen_output_logits4(
    const float4 hiddenActivation,
    const __local float4 *restrict prescreenWeights
) {
    const __local float4 *outputMix = prescreenWeights + RNNEDI_PRE_OUTPUT_MIX4_OFFSET;
    return (float4)(
        dot(hiddenActivation, outputMix[0]),
        dot(hiddenActivation, outputMix[1]),
        dot(hiddenActivation, outputMix[2]),
        dot(hiddenActivation, outputMix[3])) + prescreenWeights[RNNEDI_PRE_OUTPUT_BIAS4_INDEX];
}

static inline float4 rnnedi_prescreen_classify4(
    const __global uchar *restrict pRef,
    const int refPitch,
    const int refOffset,
    const int xbase,
    const int ybase,
    const __local float4 *restrict prescreenWeights
) {
    const __global Type4 *ref4 = (const __global Type4 *)(pRef + refOffset);
    const int refPitch4 = refPitch / (4 * (int)sizeof(Type));
    const float4 hiddenDot = rnnedi_prescreen_hidden_response4(ref4, refPitch4, xbase, ybase, prescreenWeights);
    const float4 hiddenActivation = rnnedi_prescreen_hidden_activation4(hiddenDot, prescreenWeights);
    return rnnedi_prescreen_output_logits4(hiddenActivation, prescreenWeights);
}

static inline int rnnedi_prescreen_valid_lane_mask(const int xpixel, const int width) {
    const int remain = width - xpixel;
    return (remain >= 4) ? 15 : ((remain <= 0) ? 0 : ((1 << remain) - 1));
}

static inline int rnnedi_prescreen_candidate_mask(const float4 result, const int validLaneMask) {
    const int4 bits = select((int4)(0), (int4)(1, 2, 4, 8), result <= (float4)(0.0f));
    return (bits.x | bits.y | bits.z | bits.w) & validLaneMask;
}

static inline int rnnedi_prescreen_lane_count(const int mask) {
    return popcount((uint)(mask & 15));
}

static inline void rnnedi_local_prefix512(__local int *restrict prefix, const int tid, const int count) {
    prefix[tid] = count;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = 1; offset < RNNEDI_TILE_MASK_COUNT; offset <<= 1) {
        const int addend = (tid >= offset) ? prefix[tid - offset] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        prefix[tid] += addend;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

static inline ushort rnnedi_pack_tile_xy(const int x, const int y) {
    return (ushort)((y << 7) | x);
}

static inline void rnnedi_unpack_tile_xy(const ushort packed, int *restrict x, int *restrict y) {
    *x = (int)(packed & 127);
    *y = (int)(packed >> 7);
}

static inline void rnnedi_predictor_expand_tile_masks(
    __local ushort *restrict candidateQueue,
    __local int *restrict candidatePrefix,
    const __global uchar *restrict candidateMask,
    const int maskOffset,
    const int tid
) {
    const int mask = (int)(candidateMask[maskOffset + tid] & 15);
    const int count = rnnedi_prescreen_lane_count(mask);
    rnnedi_local_prefix512(candidatePrefix, tid, count);

    int index = candidatePrefix[tid] - count;
    const int groupX = tid - (tid / RNNEDI_TILE_GROUPS_X) * RNNEDI_TILE_GROUPS_X;
    const int y = tid / RNNEDI_TILE_GROUPS_X;
    const int x = groupX << 2;
    if (mask & 1) candidateQueue[index++] = rnnedi_pack_tile_xy(x + 0, y);
    if (mask & 2) candidateQueue[index++] = rnnedi_pack_tile_xy(x + 1, y);
    if (mask & 4) candidateQueue[index++] = rnnedi_pack_tile_xy(x + 2, y);
    if (mask & 8) candidateQueue[index++] = rnnedi_pack_tile_xy(x + 3, y);
    barrier(CLK_LOCAL_MEM_FENCE);
}

static inline int4 rnnedi_keys_cubic_fallback4(
    const int4 upperOuterPixels,
    const int4 upperInnerPixels,
    const int4 lowerInnerPixels,
    const int4 lowerOuterPixels,
    const int valMin,
    const int valMax
) {
    return clamp(
        ((upperOuterPixels * -3 + (upperInnerPixels + lowerInnerPixels) * 19 + lowerOuterPixels * -3 + 16) >> 5),
        (int4)(valMin), (int4)(valMax));
}

static inline void rnnedi_prescreen_write_cubic4(
    __global uchar *restrict pDst,
    const int dstPitch,
    const int dstOffset,
    const __global uchar *restrict pRef,
    const int refPitch,
    const int refOffset,
    const int xbase,
    const int ybase,
    const int xpixel,
    const int valMin,
    const int valMax,
    const int laneMask
) {
    const __global Type4 *ref4 = (const __global Type4 *)(pRef + refOffset);
    const int refPitch4 = refPitch / (4 * (int)sizeof(Type));
    const int4 upperOuterPixels = rnnedi_to_int4(ref4[(xbase + 2) + (ybase + 0) * refPitch4]);
    const int4 upperInnerPixels = rnnedi_to_int4(ref4[(xbase + 2) + (ybase + 1) * refPitch4]);
    const int4 lowerInnerPixels = rnnedi_to_int4(ref4[(xbase + 2) + (ybase + 2) * refPitch4]);
    const int4 lowerOuterPixels = rnnedi_to_int4(ref4[(xbase + 2) + (ybase + 3) * refPitch4]);
    const int4 cubicInterpolatedPixels = rnnedi_keys_cubic_fallback4(
        upperOuterPixels, upperInnerPixels, lowerInnerPixels, lowerOuterPixels, valMin, valMax);
    __global Type *dstPixels = (__global Type *)(pDst + dstOffset + ybase * dstPitch + xpixel * (int)sizeof(Type));
    if (laneMask & 1) dstPixels[0] = (Type)cubicInterpolatedPixels.x;
    if (laneMask & 2) dstPixels[1] = (Type)cubicInterpolatedPixels.y;
    if (laneMask & 4) dstPixels[2] = (Type)cubicInterpolatedPixels.z;
    if (laneMask & 8) dstPixels[3] = (Type)cubicInterpolatedPixels.w;
}

__attribute__((reqd_work_group_size(32, 16, 1)))
__kernel void kernel_rnnedi_prescreen_cubic(
    __global uchar *restrict pDst, const int dstPitch, const int dstOffset,
    const __global uchar *restrict pRef, const int refPitch, const int refOffset,
    const __global float *restrict weights,
    __global uchar *restrict candidateMask, __global int *restrict numblocks,
    const int width4, const int width, const int height, const int valMin, const int valMax
) {
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tid = tx + ty * 32;
    const int xbase = tx + get_group_id(0) * 32;
    const int ybase = ty + get_group_id(1) * 16;

    __local float4 prescreenWeights[RNNEDI_PRE_WEIGHT_FLOAT4_COUNT];
    __local int candidateCount[RNNEDI_TILE_MASK_COUNT];

    rnnedi_prescreen_cache_weights(prescreenWeights, weights, tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 result = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
    const bool valid = xbase < width4 && ybase < height;
    const int xpixel = xbase << 2;
    const int validLaneMask = valid ? rnnedi_prescreen_valid_lane_mask(xpixel, width) : 0;
    if (valid) {
        result = rnnedi_prescreen_classify4(
            pRef, refPitch, refOffset, xbase, ybase,
            prescreenWeights);
    }

    const int mask = rnnedi_prescreen_candidate_mask(result, validLaneMask);
    const int num = rnnedi_prescreen_lane_count(mask);
    const int bid = get_group_id(0) + get_group_id(1) * get_num_groups(0);
    candidateMask[bid * RNNEDI_TILE_MASK_COUNT + tid] = (uchar)mask;

    candidateCount[tid] = num;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = RNNEDI_TILE_MASK_COUNT >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            candidateCount[tid] += candidateCount[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        numblocks[bid] = candidateCount[0];
    }

    if (num < 4 && valid) {
        rnnedi_prescreen_write_cubic4(
            pDst, dstPitch, dstOffset,
            pRef, refPitch, refOffset,
            xbase, ybase, xpixel,
            valMin, valMax,
            validLaneMask);
    }
}

__attribute__((reqd_work_group_size(RNNEDI_PRED_LOCAL_X, RNNEDI_PRED_LOCAL_Y, 1)))
__kernel void kernel_rnnedi_predictor_network(
    __global uchar *restrict pDst, const int dstPitch, const int dstOffset,
    const __global uchar *restrict pRef, const int refPitch, const int refOffset,
    const __global uchar *restrict candidateMask, const __global int *restrict numblocks,
    const __global float *restrict weights,
    const int width4, const int height, const int valMin, const int valMax
) {
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int blockX = get_group_id(0);
    const int blockY = get_group_id(1);
    const int blocksX = get_num_groups(0);
    const int bid = blockX + blockY * blocksX;
    const int maskOffset = bid * RNNEDI_TILE_MASK_COUNT;
    const int tid = ty * RNNEDI_PRED_LOCAL_X + tx;
    const int xbase = blockX * RNNEDI_TILE_PIXELS_X;
    const int ybase = blockY * RNNEDI_TILE_ROWS;

    __local Type patchPixels[RNNEDI_PRED_LOCAL_Y * RNNEDI_PRED_K];
    __local ushort candidateQueue[RNNEDI_TILE_MAX_CANDIDATES];
    __local int candidatePrefix[RNNEDI_TILE_MASK_COUNT];
#if !RNNEDI_PRED_SUBGROUP_OPT
    __local int sumPart[RNNEDI_PRED_LOCAL_Y * RNNEDI_PRED_LOCAL_X];
#if RNNEDI_BIT_DEPTH <= 8
    __local int sumsqPart[RNNEDI_PRED_LOCAL_Y * RNNEDI_PRED_LOCAL_X];
#else
    __local float varPart[RNNEDI_PRED_LOCAL_Y * RNNEDI_PRED_LOCAL_X];
#endif
    __local float avg[RNNEDI_PRED_LOCAL_Y];
    __local float stddev[RNNEDI_PRED_LOCAL_Y];
    __local float invvar[RNNEDI_PRED_LOCAL_Y];
    __local float weightedElliottVoteParts[RNNEDI_PRED_LOCAL_Y * RNNEDI_PRED_LOCAL_X];
    __local float softmaxVoteWeightParts[RNNEDI_PRED_LOCAL_Y * RNNEDI_PRED_LOCAL_X];
#endif

    const int nb = numblocks[bid];
    if (nb <= 0) {
        return;
    }
    const __global float2 *weightsBody = (const __global float2 *)weights;
    const __global float2 *weightsBias = weightsBody + RNNEDI_PRED_BODY_FLOAT2_COUNT;

    // OpenCL kernel uses tile-local candidate masks to avoid running the expensive
    // predictor on pixels rejected by prescreening. That keeps most lanes within
    // a subgroup on real network work instead of diverging on sparse per-pixel
    // branches after the inexpensive fallback pass.
    rnnedi_predictor_expand_tile_masks(candidateQueue, candidatePrefix, candidateMask, maskOffset, tid);

    for (int b = 0; b < nb; b += RNNEDI_PRED_LOCAL_Y) {
        int localX = 0;
        int localY = 0;
        const bool active = (b + ty < nb);
        if (active) {
            rnnedi_unpack_tile_xy(candidateQueue[b + ty], &localX, &localY);
        }
        const int x = xbase + localX;
        const int y = ybase + localY;

        __local Type *patch = &patchPixels[RNNEDI_PRED_PATCH_AT(ty, 0)];
        rnnedi_load_patch_distributed(patch, pRef, refPitch, refOffset, x, y, active, tx);
        barrier(CLK_LOCAL_MEM_FENCE);

        float avgValue = 0.0f;
        float stddevValue = 0.0f;
        float invvarValue = 0.0f;
#if RNNEDI_BIT_DEPTH <= 8
#if RNNEDI_PRED_SUBGROUP_OPT
        int sum = 0;
        int sumsq = 0;
        for (int k = tx; k < RNNEDI_PRED_K; k += RNNEDI_PRED_LOCAL_X) {
            const int v = (int)patch[k];
            sum += v;
            sumsq += v * v;
        }
        const int sumAll = rnnedi_subgroup_reduce16_int(sum);
        const int sumsqAll = rnnedi_subgroup_reduce16_int(sumsq);
        if (tx == 0) {
            const float scale = 1.0f / (float)RNNEDI_PRED_K;
            avgValue = (float)sumAll * scale;
            stddevValue = (float)sumsqAll * scale - avgValue * avgValue;
            if (stddevValue <= RGY_FLT_EPS) {
                stddevValue = 0.0f;
                invvarValue = 0.0f;
            } else {
                stddevValue = sqrt(stddevValue);
                invvarValue = 1.0f / stddevValue;
            }
        }
        avgValue = rnnedi_subgroup_broadcast16_float(avgValue);
        stddevValue = rnnedi_subgroup_broadcast16_float(stddevValue);
        invvarValue = rnnedi_subgroup_broadcast16_float(invvarValue);
#else
        rnnedi_predictor_lane_patch_moments(patch, sumPart, sumsqPart, ty, tx);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tx == 0) {
            rnnedi_predictor_finalize_patch_stats(sumPart, sumsqPart, avg, stddev, invvar, ty);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        avgValue = avg[ty];
        stddevValue = stddev[ty];
        invvarValue = invvar[ty];
#endif
#else
#if RNNEDI_PRED_SUBGROUP_OPT
        int sum = 0;
        for (int k = tx; k < RNNEDI_PRED_K; k += RNNEDI_PRED_LOCAL_X) {
            sum += (int)patch[k];
        }
        const int sumAll = rnnedi_subgroup_reduce16_int(sum);
        if (tx == 0) {
            avgValue = (float)sumAll * (1.0f / (float)RNNEDI_PRED_K);
        }
        avgValue = rnnedi_subgroup_broadcast16_float(avgValue);

        float sumsq = 0.0f;
        for (int k = tx; k < RNNEDI_PRED_K; k += RNNEDI_PRED_LOCAL_X) {
            const float diff = (float)patch[k] - avgValue;
            sumsq = fma(diff, diff, sumsq);
        }
        const float varAll = rnnedi_subgroup_reduce16_float(sumsq) * (1.0f / (float)RNNEDI_PRED_K);
        if (tx == 0) {
            if (varAll <= RGY_FLT_EPS) {
                stddevValue = 0.0f;
                invvarValue = 0.0f;
            } else {
                stddevValue = sqrt(varAll);
                invvarValue = 1.0f / stddevValue;
            }
        }
        stddevValue = rnnedi_subgroup_broadcast16_float(stddevValue);
        invvarValue = rnnedi_subgroup_broadcast16_float(invvarValue);
#else
        rnnedi_predictor_lane_patch_sum(patch, sumPart, ty, tx);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tx == 0) {
            rnnedi_predictor_finalize_patch_avg(sumPart, avg, ty);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        rnnedi_predictor_lane_patch_variance(patch, varPart, ty, tx, avg[ty]);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tx == 0) {
            rnnedi_predictor_finalize_patch_stddev(varPart, stddev, invvar, ty);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        avgValue = avg[ty];
        stddevValue = stddev[ty];
        invvarValue = invvar[ty];
#endif
#endif

        float result = 0.0f;
#if !RNNEDI_PRED_SUBGROUP_OPT
        const int partBase = ty * RNNEDI_PRED_LOCAL_X;
#endif
        for (int q = 0; q < RNNEDI_PRED_QUAL; q++) {
            const float2 vote = rnnedi_predictor_lane_vote(
                patch, weightsBody, weightsBias,
                tx, q, invvarValue);
#if RNNEDI_PRED_SUBGROUP_OPT
            const float weightedElliottVoteSum = rnnedi_subgroup_reduce16_float(vote.x);
            const float softmaxVoteWeightSum = rnnedi_subgroup_reduce16_float(vote.y);
            result = rnnedi_predictor_merge_vote_sums(
                weightedElliottVoteSum, softmaxVoteWeightSum,
                avgValue, stddevValue, result);
#else
            weightedElliottVoteParts[partBase + tx] = vote.x;
            softmaxVoteWeightParts[partBase + tx] = vote.y;
            barrier(CLK_LOCAL_MEM_FENCE);

            if (tx == 0) {
                result = rnnedi_predictor_merge_votes(
                    weightedElliottVoteParts, softmaxVoteWeightParts,
                    ty, avgValue, stddevValue, result);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
#endif
        }

        if (tx == 0 && active) {
            rnnedi_predictor_write_result(pDst, dstPitch, dstOffset, x, y, result, valMin, valMax);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
