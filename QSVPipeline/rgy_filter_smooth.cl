// TypePixel
// TypeQP
// TypeQP4
// usefp16Dct
// usefp16IO
// radius
// bit_depth
// SPP_THREAD_BLOCK_X
// SPP_THREAD_BLOCK_Y
// SPP_SHARED_BLOCK_NUM_X
// SPP_SHARED_BLOCK_NUM_Y
// SPP_LOOP_COUNT_BLOCK
// DCT_IDCT_BARRIER_MODE // 0... off, 1... barrier(), 2... sub_group_barrier

#if usefp16Dct || usefp16IO
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if usefp16Dct
#define TypeDct half2
#else
#define TypeDct float
#endif

#if usefp16IO
#define TypeIO half
#else
#define TypeIO float
#endif

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

#if DCT_IDCT_BARRIER_MODE == 1
#define DCT_IDCT_BARRIER(x) barrier(x)
#elif DCT_IDCT_BARRIER_MODE == 2
#define DCT_IDCT_BARRIER(x) sub_group_barrier(x)
#else
#define DCT_IDCT_BARRIER(x)
#endif

//CUDA Sampleより拝借
#define C_a (1.387039845322148f) //!< a = (2^0.5) * cos(    pi / 16);  Used in forward and inverse DCT.
#define C_b (1.306562964876377f) //!< b = (2^0.5) * cos(    pi /  8);  Used in forward and inverse DCT.
#define C_c (1.175875602419359f) //!< c = (2^0.5) * cos(3 * pi / 16);  Used in forward and inverse DCT.
#define C_d (0.785694958387102f) //!< d = (2^0.5) * cos(5 * pi / 16);  Used in forward and inverse DCT.
#define C_e (0.541196100146197f) //!< e = (2^0.5) * cos(3 * pi /  8);  Used in forward and inverse DCT.
#define C_f (0.275899379282943f) //!< f = (2^0.5) * cos(7 * pi / 16);  Used in forward and inverse DCT.

//Normalization constant that is used in forward and inverse DCT
#define C_norm (0.3535533905932737f) // 1 / (8^0.5)

void CUDAsubroutineInplaceDCTvector(__local TypeDct *Vect0, const int Step) {
    __local TypeDct *Vect1 = Vect0 + Step;
    __local TypeDct *Vect2 = Vect1 + Step;
    __local TypeDct *Vect3 = Vect2 + Step;
    __local TypeDct *Vect4 = Vect3 + Step;
    __local TypeDct *Vect5 = Vect4 + Step;
    __local TypeDct *Vect6 = Vect5 + Step;
    __local TypeDct *Vect7 = Vect6 + Step;

    TypeDct X07P = (*Vect0) + (*Vect7);
    TypeDct X16P = (*Vect1) + (*Vect6);
    TypeDct X25P = (*Vect2) + (*Vect5);
    TypeDct X34P = (*Vect3) + (*Vect4);

    TypeDct X07M = (*Vect0) - (*Vect7);
    TypeDct X61M = (*Vect6) - (*Vect1);
    TypeDct X25M = (*Vect2) - (*Vect5);
    TypeDct X43M = (*Vect4) - (*Vect3);

    TypeDct X07P34PP = X07P + X34P;
    TypeDct X07P34PM = X07P - X34P;
    TypeDct X16P25PP = X16P + X25P;
    TypeDct X16P25PM = X16P - X25P;

    (*Vect0) = (TypeDct)(C_norm) * (X07P34PP + X16P25PP);
    (*Vect2) = (TypeDct)(C_norm) * ((TypeDct)(C_b) * X07P34PM + (TypeDct)(C_e) * X16P25PM);
    (*Vect4) = (TypeDct)(C_norm) * (X07P34PP - X16P25PP);
    (*Vect6) = (TypeDct)(C_norm) * ((TypeDct)(C_e) * X07P34PM - (TypeDct)(C_b) * X16P25PM);

    (*Vect1) = (TypeDct)(C_norm) * ((TypeDct)(C_a) * X07M - (TypeDct)(C_c) * X61M + (TypeDct)(C_d) * X25M - (TypeDct)(C_f) * X43M);
    (*Vect3) = (TypeDct)(C_norm) * ((TypeDct)(C_c) * X07M + (TypeDct)(C_f) * X61M - (TypeDct)(C_a) * X25M + (TypeDct)(C_d) * X43M);
    (*Vect5) = (TypeDct)(C_norm) * ((TypeDct)(C_d) * X07M + (TypeDct)(C_a) * X61M + (TypeDct)(C_f) * X25M - (TypeDct)(C_c) * X43M);
    (*Vect7) = (TypeDct)(C_norm) * ((TypeDct)(C_f) * X07M + (TypeDct)(C_d) * X61M + (TypeDct)(C_c) * X25M + (TypeDct)(C_a) * X43M);
}

void CUDAsubroutineInplaceIDCTvector(__local TypeDct *Vect0, const int Step) {
    __local TypeDct *Vect1 = Vect0 + Step;
    __local TypeDct *Vect2 = Vect1 + Step;
    __local TypeDct *Vect3 = Vect2 + Step;
    __local TypeDct *Vect4 = Vect3 + Step;
    __local TypeDct *Vect5 = Vect4 + Step;
    __local TypeDct *Vect6 = Vect5 + Step;
    __local TypeDct *Vect7 = Vect6 + Step;

    TypeDct Y04P = (*Vect0) + (*Vect4);
    TypeDct Y2b6eP = (TypeDct)(C_b) * (*Vect2) + (TypeDct)(C_e) * (*Vect6);

    TypeDct Y04P2b6ePP = Y04P + Y2b6eP;
    TypeDct Y04P2b6ePM = Y04P - Y2b6eP;
    TypeDct Y7f1aP3c5dPP = (TypeDct)(C_f) * (*Vect7) + (TypeDct)(C_a) * (*Vect1) + (TypeDct)(C_c) * (*Vect3) + (TypeDct)(C_d) * (*Vect5);
    TypeDct Y7a1fM3d5cMP = (TypeDct)(C_a) * (*Vect7) - (TypeDct)(C_f) * (*Vect1) + (TypeDct)(C_d) * (*Vect3) - (TypeDct)(C_c) * (*Vect5);

    TypeDct Y04M = (*Vect0) - (*Vect4);
    TypeDct Y2e6bM = (TypeDct)(C_e) * (*Vect2) - (TypeDct)(C_b) * (*Vect6);

    TypeDct Y04M2e6bMP = Y04M + Y2e6bM;
    TypeDct Y04M2e6bMM = Y04M - Y2e6bM;
    TypeDct Y1c7dM3f5aPM = (TypeDct)(C_c) * (*Vect1) - (TypeDct)(C_d) * (*Vect7) - (TypeDct)(C_f) * (*Vect3) - (TypeDct)(C_a) * (*Vect5);
    TypeDct Y1d7cP3a5fMM = (TypeDct)(C_d) * (*Vect1) + (TypeDct)(C_c) * (*Vect7) - (TypeDct)(C_a) * (*Vect3) + (TypeDct)(C_f) * (*Vect5);

    (*Vect0) = (TypeDct)(C_norm) * (Y04P2b6ePP + Y7f1aP3c5dPP);
    (*Vect7) = (TypeDct)(C_norm) * (Y04P2b6ePP - Y7f1aP3c5dPP);
    (*Vect4) = (TypeDct)(C_norm) * (Y04P2b6ePM + Y7a1fM3d5cMP);
    (*Vect3) = (TypeDct)(C_norm) * (Y04P2b6ePM - Y7a1fM3d5cMP);

    (*Vect1) = (TypeDct)(C_norm) * (Y04M2e6bMP + Y1c7dM3f5aPM);
    (*Vect5) = (TypeDct)(C_norm) * (Y04M2e6bMM - Y1d7cP3a5fMM);
    (*Vect2) = (TypeDct)(C_norm) * (Y04M2e6bMM + Y1d7cP3a5fMM);
    (*Vect6) = (TypeDct)(C_norm) * (Y04M2e6bMP - Y1c7dM3f5aPM);
}

//こうしたバリアには全スレッドが通るようにしないとRX5500などでは正常に動作しない (他の箇所でbarrierしても意味がない)
//なので、計算の有無はenableフラグで切り替える
void dct8x8(bool enable, __local TypeDct shared_tmp[8][9], int thWorker) {
    DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
    if (enable) CUDAsubroutineInplaceDCTvector((__local TypeDct *)&shared_tmp[thWorker][0], 1); // row
    DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
    if (enable) CUDAsubroutineInplaceDCTvector((__local TypeDct *)&shared_tmp[0][thWorker], 9); // column
    DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
}

void idct8x8(bool enable, __local TypeDct shared_tmp[8][9], int thWorker) {
    DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
    if (enable) CUDAsubroutineInplaceIDCTvector((__local TypeDct *)&shared_tmp[0][thWorker], 9); // column
    DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
    if (enable) CUDAsubroutineInplaceIDCTvector((__local TypeDct *)&shared_tmp[thWorker][0], 1); // row
    DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
}
float calcThreshold(const float qp, const float threshA, const float threshB) {
    return clamp(threshA * qp + threshB, 0.0f, qp);
}

void threshold8x8(__local TypeDct shared_tmp[8][9], int thWorker, const TypeDct threshold) {
    #pragma unroll
    for (int y = 0; y < 8; y++) {
        if (y > 0 || thWorker > 0) {
            __local TypeDct *ptr = &shared_tmp[y][thWorker];
            TypeDct val = ptr[0];
            ptr[0] = (fabs(val) <= threshold) ? (TypeDct)0.0f : val;
        }
    }
}

__constant uchar2 SPP_DEBLOCK_OFFSET[127] = {
  { 0,0 },                                                         // quality = 0

  { 0,0 },{ 4,4 },                                                 // quality = 1

  { 0,0 },{ 2,2 },{ 6,4 },{ 4,6 },                                 // quality = 2

  { 0,0 },{ 5,1 },{ 2,2 },{ 7,3 },{ 4,4 },{ 1,5 },{ 6,6 },{ 3,7 }, // quality = 3

  { 0,0 },{ 4,0 },{ 1,1 },{ 5,1 },{ 3,2 },{ 7,2 },{ 2,3 },{ 6,3 }, // quality = 4
  { 0,4 },{ 4,4 },{ 1,5 },{ 5,5 },{ 3,6 },{ 7,6 },{ 2,7 },{ 6,7 },

  { 0,0 },{ 0,2 },{ 0,4 },{ 0,6 },{ 1,1 },{ 1,3 },{ 1,5 },{ 1,7 }, // quality = 5
  { 2,0 },{ 2,2 },{ 2,4 },{ 2,6 },{ 3,1 },{ 3,3 },{ 3,5 },{ 3,7 },
  { 4,0 },{ 4,2 },{ 4,4 },{ 4,6 },{ 5,1 },{ 5,3 },{ 5,5 },{ 5,7 },
  { 6,0 },{ 6,2 },{ 6,4 },{ 6,6 },{ 7,1 },{ 7,3 },{ 7,5 },{ 7,7 },

  { 0,0 },{ 4,4 },{ 0,4 },{ 4,0 },{ 2,2 },{ 6,6 },{ 2,6 },{ 6,2 }, // quality = 6
  { 0,2 },{ 4,6 },{ 0,6 },{ 4,2 },{ 2,0 },{ 6,4 },{ 2,4 },{ 6,0 },
  { 1,1 },{ 5,5 },{ 1,5 },{ 5,1 },{ 3,3 },{ 7,7 },{ 3,7 },{ 7,3 },
  { 1,3 },{ 5,7 },{ 1,7 },{ 5,3 },{ 3,1 },{ 7,5 },{ 3,5 },{ 7,1 },
  { 0,1 },{ 4,5 },{ 0,5 },{ 4,1 },{ 2,3 },{ 6,7 },{ 2,7 },{ 6,3 },
  { 0,3 },{ 4,7 },{ 0,7 },{ 4,3 },{ 2,1 },{ 6,5 },{ 2,5 },{ 6,1 },
  { 1,0 },{ 5,4 },{ 1,4 },{ 5,0 },{ 3,2 },{ 7,6 },{ 3,6 },{ 7,2 },
  { 1,2 },{ 5,6 },{ 1,6 },{ 5,2 },{ 3,0 },{ 7,4 },{ 3,4 },{ 7,0 },
};

#define STMP(x, y) (shared_tmp[(y)][(x)])
#define SIN(x, y)  (shared_in[(y) & (8 * SPP_SHARED_BLOCK_NUM_Y - 1)][(x)])
#define SOUT(x, y) (shared_out[(y) & (8 * SPP_SHARED_BLOCK_NUM_Y - 1)][(x)])

void load_8x8(__local TypeIO shared_in[8 * SPP_SHARED_BLOCK_NUM_Y][8 * SPP_SHARED_BLOCK_NUM_X], __read_only image2d_t texSrc, int thWorker, int shared_bx, int shared_by, int src_global_bx, int src_global_by) {
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    #pragma unroll
    for (int y = 0; y < 8; y++) {
        SIN(shared_bx * 8 + thWorker, shared_by * 8 + y) = (TypeIO)(read_imagef(texSrc, sampler, (int2)(src_global_bx * 8 + thWorker, src_global_by * 8 + y)).x);
    }
}
void zero_8x8(__local TypeIO shared_out[8 * SPP_SHARED_BLOCK_NUM_Y][8 * SPP_SHARED_BLOCK_NUM_X], int thWorker, int shared_bx, int shared_by) {
    #pragma unroll
    for (int y = 0; y < 8; y++) {
        SOUT(shared_bx * 8 + thWorker, shared_by * 8 + y) = (TypeIO)0.0f;
    }
}
void load_8x8tmp(__local TypeDct shared_tmp[8][9], __local TypeIO shared_in[8 * SPP_SHARED_BLOCK_NUM_Y][8 * SPP_SHARED_BLOCK_NUM_X], int thWorker, int shared_bx, int shared_by, int offset1_x, int offset1_y, int offset2_x, int offset2_y) {
    #pragma unroll
    for (int y = 0; y < 8; y++) {
        TypeIO v0 = SIN(shared_bx * 8 + offset1_x + thWorker, shared_by * 8 + offset1_y + y);
#if usefp16Dct
        TypeIO v1 = SIN(shared_bx * 8 + offset2_x + thWorker, shared_by * 8 + offset2_y + y);
        STMP(thWorker, y) = (half2)(v0, v1);
#else
        STMP(thWorker, y) = (TypeDct)v0;
#endif
    }
}
void add_8x8tmp(__local TypeIO shared_out[8 * SPP_SHARED_BLOCK_NUM_Y][8 * SPP_SHARED_BLOCK_NUM_X], __local TypeDct shared_tmp[8][9], int thWorker, int shared_bx, int shared_by, int offset1_x, int offset1_y, int offset2_x, int offset2_y) {
    #pragma unroll
    for (int y = 0; y < 8; y++) {
        TypeDct v = STMP(thWorker, y);
#if usefp16Dct
        SOUT(shared_bx * 8 + offset1_x + thWorker, shared_by * 8 + offset1_y + y) += (TypeIO)v.x;
        SOUT(shared_bx * 8 + offset2_x + thWorker, shared_by * 8 + offset2_y + y) += (TypeIO)v.y;
#else
        SOUT(shared_bx * 8 + offset1_x + thWorker, shared_by * 8 + offset1_y + y) += (TypeIO)v;
#endif
    }
}

void store_8x8(__global char *pDst, int dstPitch, int dstWidth, int dstHeight, __local TypeIO shared_out[8 * SPP_SHARED_BLOCK_NUM_Y][8 * SPP_SHARED_BLOCK_NUM_X], int thWorker, int shared_bx, int shared_by, int dst_global_bx, int dst_global_by, int quality) {
    const int dst_global_x = dst_global_bx * 8 + thWorker;
    if (dst_global_x < dstWidth) {
        const int dst_block_offset = (dst_global_by * 8) * dstPitch + dst_global_x * sizeof(TypePixel);
        __global char *ptrDst = pDst + dst_block_offset;

        const int y_max = dstHeight - dst_global_by * 8;
        #pragma unroll
        for (int y = 0; y < 8; y++, ptrDst += dstPitch) {
            if (y < y_max) {
                *(__global TypePixel *)ptrDst = (TypePixel)clamp((float)SOUT(shared_bx * 8 + thWorker, shared_by * 8 + y) * (float)(1 << (bit_depth - quality)), 0.0f, (float)((1 << bit_depth) - 0.5f));
            }
        }
    }
}

__kernel void kernel_smooth(
    __global char *ptrDst,
    __read_only image2d_t texSrc,
    const int dstPitch,
    const int dstWidth,
    const int dstHeight,
    const __global char *ptrQP,
    const int qpPitch,
    const int qpWidth,
    const int qpHeight,
    const int qpBlockShift,
    const float qpMul,
    const int quality,
    const float strength,
    const float threshA, const float threshB) {
    const int thWorker = get_local_id(0); // SPP_THREAD_BLOCK_X
    const int local_bx = get_local_id(1); // SPP_THREAD_BLOCK_Y
    const int global_bx = get_group_id(0) * SPP_BLOCK_SIZE_X + local_bx;
    int global_by = get_group_id(1) * SPP_LOOP_COUNT_BLOCK;
    const int count = 1 << quality;

    __local TypeDct shared_tmp[SPP_THREAD_BLOCK_Y][8][9];
    __local TypeIO shared_in[8 * SPP_SHARED_BLOCK_NUM_Y][8 * SPP_SHARED_BLOCK_NUM_X];
    __local TypeIO shared_out[8 * SPP_SHARED_BLOCK_NUM_Y][8 * SPP_SHARED_BLOCK_NUM_X];

    load_8x8(shared_in, texSrc, thWorker, local_bx, 0, global_bx - 1, global_by - 1);
    zero_8x8(shared_out, thWorker, local_bx, 0);
    if (local_bx < (SPP_SHARED_BLOCK_NUM_X - SPP_BLOCK_SIZE_X)) {
        load_8x8(shared_in, texSrc, thWorker, local_bx + SPP_BLOCK_SIZE_X, 0, global_bx + SPP_BLOCK_SIZE_X - 1, global_by - 1);
        zero_8x8(shared_out, thWorker, local_bx + SPP_BLOCK_SIZE_X, 0);
    }

    for (int local_by = 0; local_by <= SPP_LOOP_COUNT_BLOCK; local_by++, global_by++) {
        const TypeQP qp = *(__global TypeQP *)(ptrQP + min(global_by >> qpBlockShift, qpHeight) * qpPitch + min(global_bx >> qpBlockShift, qpWidth) * sizeof(TypeQP));
        const TypeDct threshold = (TypeDct)((1.0f / (8.0f * (float)(1<<8 /*閾値は8bitベースで対象は規格化済み*/))) * (calcThreshold((float)qp * qpMul, threshA, threshB) * ((float)(1 << 2) + strength) - 1.0f));

        load_8x8(shared_in, texSrc, thWorker, local_bx, local_by+1, global_bx - 1, global_by);
        zero_8x8(shared_out, thWorker, local_bx, local_by+1);
        if (local_bx < (SPP_SHARED_BLOCK_NUM_X - SPP_BLOCK_SIZE_X)) {
            load_8x8(shared_in, texSrc, thWorker, local_bx + SPP_BLOCK_SIZE_X, local_by+1, global_bx + SPP_BLOCK_SIZE_X - 1, global_by);
            zero_8x8(shared_out, thWorker, local_bx + SPP_BLOCK_SIZE_X, local_by+1);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //fp16では、icount2つ分をSIMD的に2並列で処理する
        for (int icount = 0; icount < count; icount += (usefp16Dct) ? 2 : 1) {
            const uchar2 offset = SPP_DEBLOCK_OFFSET[count - 1 + icount];
            const int offset1_x = offset.x;
            const int offset1_y = offset.y;
            int offset2_x = 0;
            int offset2_y = 0;
            if (usefp16Dct) {
                const uchar2 offset2 = SPP_DEBLOCK_OFFSET[count + icount];
                offset2_x = offset2.x;
                offset2_y = offset2.y;
            }

            //fp16では、icount2つ分をSIMD的に2並列で処理するが、
            //add_8x8tmpで衝突する可能性がある
            //衝突するのは、warp(subgroup)間の書き込み先がオーバーラップした場合なので、
            //そこで、warp(subgroup)間を1ブロック空けて処理することでオーバーラップが起こらないようにする
            //1warp(subgroup)=32threadの場合、SPP_THREAD_BLOCK_X(blockDim)=8なので、
            //warp1=local_bx[0-3], warp2=local_bx[4-7]
            //local_bx 3と4の間をひとつ開けるようにする
            //どのみち、1ブロックは別に処理する必要があるので、都合がよい
            //1warp(subgroup)=64threadの場合、特に気にしなくてよい
            //1warp(subgroup)=16threadの場合には対応できない
            int target_bx = (local_bx < 4) ? local_bx : local_bx + 1;
            load_8x8tmp(shared_tmp[local_bx], shared_in, thWorker, target_bx, local_by, offset1_x, offset1_y, offset2_x, offset2_y);
            dct8x8(true, shared_tmp[local_bx], thWorker);
            threshold8x8(shared_tmp[local_bx], thWorker, threshold);
            idct8x8(true, shared_tmp[local_bx], thWorker);
            add_8x8tmp(shared_out, shared_tmp[local_bx], thWorker, target_bx, local_by, offset1_x, offset1_y, offset2_x, offset2_y);
            if (usefp16Dct) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            { // あまったブロックの処理
                const bool enable = local_bx < 1;
                target_bx = 4;
                if (enable) load_8x8tmp(shared_tmp[local_bx], shared_in, thWorker, target_bx, local_by, offset1_x, offset1_y, offset2_x, offset2_y);
                dct8x8(enable, shared_tmp[local_bx], thWorker);
                if (enable) threshold8x8(shared_tmp[local_bx], thWorker, threshold);
                idct8x8(enable, shared_tmp[local_bx], thWorker);
                if (enable) add_8x8tmp(shared_out, shared_tmp[local_bx], thWorker, target_bx, local_by, offset1_x, offset1_y, offset2_x, offset2_y);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (local_by > 0) {
            store_8x8(ptrDst, dstPitch, dstWidth, dstHeight, shared_out, thWorker, local_bx+1, local_by, global_bx, global_by-1, quality);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void kernel_smooth_set_qp(
    __global uchar *ptrQP,
    const int qpPitch, const int qpWidth, const int qpHeight,
    const int qp) {
    const int qpx = get_global_id(0) * 4;
    const int qpy = get_global_id(1);

    if (qpx < qpWidth && qpy < qpHeight) {
        TypeQP4 qp4 = (TypeQP4)qp;
        ptrQP += (qpy * qpPitch + qpx * sizeof(qp4.x));
        *(__global TypeQP4 *)ptrQP = qp4;
    }
}
