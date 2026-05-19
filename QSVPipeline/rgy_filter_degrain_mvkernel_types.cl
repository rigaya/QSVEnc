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

#ifndef RGY_FILTER_DEGRAIN_MOTION_SEARCH_TYPES_CL
#define RGY_FILTER_DEGRAIN_MOTION_SEARCH_TYPES_CL

#ifndef DEGRAIN_BLK_SIZE
#define DEGRAIN_BLK_SIZE 32
#endif

#ifndef DEGRAIN_NPEL
#define DEGRAIN_NPEL DEGRAIN_PEL
#endif

#ifndef DEGRAIN_SEARCH_MODE
#define DEGRAIN_SEARCH_MODE 4
#endif

#ifndef DEGRAIN_SEARCH_PARAM
#define DEGRAIN_SEARCH_PARAM 2
#endif

#ifndef DEGRAIN_CHROMA
#define DEGRAIN_CHROMA 0
#endif

#ifndef DEGRAIN_TRUE_MOTION
#define DEGRAIN_TRUE_MOTION 0
#endif

#ifndef DEGRAIN_GLOBAL_MOTION
#define DEGRAIN_GLOBAL_MOTION 0
#endif

typedef enum {
    DEGRAIN_MOTION_SEARCH_VEC_KIND_ZERO   = 0,
    DEGRAIN_MOTION_SEARCH_VEC_KIND_GLOBAL = 1,
    // ブロックスロットの開始位置。sentinel スロット数も兼ねる。
    DEGRAIN_MOTION_SEARCH_VEC_KIND_BLOCK  = 2,
} degrain_motion_search_vec_kind_t;

typedef struct {
    uint score_primary;
    uint sad_metric;
    short pos_x;
    short pos_y;
} degrain_mv_internal_t;

typedef degrain_mv_internal_t degrain_motion_search_saved_vector_t;

typedef struct {
    uint score_primary;
    uint sad_metric;
    short pos_x;
    short pos_y;
} degrain_motion_search_candidate_t;

typedef struct {
    int frameNumber;
    int refdir;
    int blockOffset;
    int blocksX;
    int blocksY;
    int reserved0;
    int reserved1;
    int reserved2;
} degrain_search_batch_data_t;

inline int degrain_motion_search_vec_zero_index(const int planeBase) {
    return planeBase + (int)DEGRAIN_MOTION_SEARCH_VEC_KIND_ZERO;
}

inline int degrain_motion_search_vec_global_index(const int planeBase) {
    return planeBase + (int)DEGRAIN_MOTION_SEARCH_VEC_KIND_GLOBAL;
}

inline int degrain_motion_search_vec_block_index(const int planeBase, const int block) {
    return planeBase + (int)DEGRAIN_MOTION_SEARCH_VEC_KIND_BLOCK + block;
}

inline int degrain_motion_search_vec_current_index(
    const int planeBase,
    const int blockCount,
    const int block) {
    return planeBase + (int)DEGRAIN_MOTION_SEARCH_VEC_KIND_BLOCK + clamp(block, 0, max(blockCount - 1, 0));
}

// 前回スナップショット用バッファは vectors と独立した cl_mem として確保される。
// host 側は同レイアウト (sentinel 2 + blockCount 個) のバッファを vectorsPrev に渡す。
inline int degrain_motion_search_vec_prev_index(
    const int planeBase,
    const int blockCount,
    const int block) {
    return planeBase + (int)DEGRAIN_MOTION_SEARCH_VEC_KIND_BLOCK + clamp(block, 0, max(blockCount - 1, 0));
}

inline int degrain_motion_search_vec_final_index(
    const int finalBase,
    const int blockCount,
    const int block) {
    return finalBase + clamp(block, 0, max(blockCount - 1, 0));
}

inline degrain_mv_internal_t degrain_motion_search_make_vector(
    const int posX,
    const int posY,
    const uint sadMetric,
    const uint scorePrimary) {
    degrain_mv_internal_t vec;
    vec.score_primary = scorePrimary;
    vec.sad_metric = sadMetric;
    vec.pos_x = (short)posX;
    vec.pos_y = (short)posY;
    return vec;
}

inline degrain_motion_search_candidate_t degrain_motion_search_make_candidate(
    const int posX,
    const int posY,
    const uint sadMetric,
    const uint scorePrimary) {
    degrain_motion_search_candidate_t candidate;
    candidate.score_primary = scorePrimary;
    candidate.sad_metric = sadMetric;
    candidate.pos_x = (short)posX;
    candidate.pos_y = (short)posY;
    return candidate;
}

inline degrain_motion_search_candidate_t degrain_motion_search_saved_vector_to_candidate(
    const degrain_motion_search_saved_vector_t vec) {
    return degrain_motion_search_make_candidate(vec.pos_x, vec.pos_y, vec.sad_metric, vec.score_primary);
}

inline degrain_motion_search_saved_vector_t degrain_motion_search_candidate_to_saved_vector(
    const degrain_motion_search_candidate_t candidate) {
    return degrain_motion_search_make_vector(candidate.pos_x, candidate.pos_y, candidate.sad_metric, candidate.score_primary);
}

#endif
