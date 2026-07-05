// TypePixel
// bit_depth
// USE_FP16
// TypeComplex
// BLOCK_SIZE
// DENOISE_BLOCK_SIZE_X
// temporalCurrentIdx
// temporalCount
// FFT_BARRIER_MODE
// SUB_GROUP_SIZE
// filterMethod

//#define TypeComplex half2

#if USE_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if FFT_BARRIER_MODE == 2 && BLOCK_SIZE <= SUB_GROUP_SIZE
#define BLOCK_SYNC sub_group_barrier(CLK_LOCAL_MEM_FENCE)
#else
#define BLOCK_SYNC barrier(CLK_LOCAL_MEM_FENCE)
#endif

#define FFT_M_PI (3.14159265358979323846f)

/*
以下の定数テーブルをコンパイル時に結合して定数の最適化を行う
__constant int bitreverse_BLOCK_SIZE[BLOCK_SIZE]
__constant TypeComplex FW_BLOCK_K_Ntrue[]
__constant TypeComplex FW_BLOCK_K_Nfalse[]
__constant TypeComplex FW_TEMPORAL3[2][5]
*/

const __global char *selectptr(const __global char *ptr0, const __global char *ptr1, const int idx) {
    if (idx == 1) return ptr1;
    return ptr0;
}
__global char *selectptrdst(__global char *ptr0, __global char *ptr1, const int idx) {
    if (idx == 1) return ptr1;
    return ptr0;
}

int wrap_idx(const int idx, const int min, const int max) {
    if (idx < min) {
        return min - idx;
    }
    if (idx > max) {
        return max - (idx - max);
    }
    return idx;
}

TypeComplex cmul(const TypeComplex a, const TypeComplex b) {
//    result.v.x = (a.x * b.x) - (a.y * b.y);
//    result.v.y = (a.x * b.y) + (a.y * b.x);
    return a.xx * b + a.yy * b.yx * (TypeComplex)(-1.0f, 1.0f);
}

float csquare(const TypeComplex a) {
    float ax = a.x;
    float ay = a.y;
    return ax * ax + ay * ay;
}

//const TypeComplex fw(const bool forward, const int k, const int N) { \
//    const float theta = ((forward) ? -2.0f : +2.0f) * FFT_M_PI * k / (float)N; \
//    return (TypeComplex)(cos(theta), sin(theta)); \
//}

#define FW_N_FORWARD(N, forward) \
    const TypeComplex fw##N##forward(const int k) { \
        return FW_BLOCK_K_##N##forward[k]; \
    }

FW_N_FORWARD(2, true)
FW_N_FORWARD(2, false)
FW_N_FORWARD(4, true)
FW_N_FORWARD(4, false)
FW_N_FORWARD(8, true)
FW_N_FORWARD(8, false)
FW_N_FORWARD(16, true)
FW_N_FORWARD(16, false)
FW_N_FORWARD(32, true)
FW_N_FORWARD(32, false)
FW_N_FORWARD(64, true)
FW_N_FORWARD(64, false)

#define FFT_CALC0(N, forward) \
    TypeComplex fft_calc0_##N##forward(TypeComplex c0, TypeComplex c1, const int k) { \
        return c0 + cmul(fw##N##forward(k), c1); \
    }

#define FFT_CALC1(N, forward) \
    TypeComplex fft_calc1_##N##forward(TypeComplex c0, TypeComplex c1, const int k) { \
        return c0 - cmul(fw##N##forward(k), c1); \
    }

FFT_CALC0(2, true)
FFT_CALC0(2, false)
FFT_CALC0(4, true)
FFT_CALC0(4, false)
FFT_CALC0(8, true)
FFT_CALC0(8, false)
FFT_CALC0(16, true)
FFT_CALC0(16, false)
FFT_CALC0(32, true)
FFT_CALC0(32, false)
FFT_CALC0(64, true)
FFT_CALC0(64, false)

FFT_CALC1(2, true)
FFT_CALC1(2, false)
FFT_CALC1(4, true)
FFT_CALC1(4, false)
FFT_CALC1(8, true)
FFT_CALC1(8, false)
FFT_CALC1(16, true)
FFT_CALC1(16, false)
FFT_CALC1(32, true)
FFT_CALC1(32, false)
FFT_CALC1(64, true)
FFT_CALC1(64, false)


#define fftpermute(N) static void fftpermute##N(const int step, __local TypeComplex *data) { \
        TypeComplex work[N]; \
        for (int i = 0; i < N; i++) { \
            work[i] = data[i * step]; \
        } \
        BLOCK_SYNC; \
        for (int i = 0; i < N; i++) { \
            data[i * step] = work[bitreverse_BLOCK_SIZE[i]]; \
        } \
    }

fftpermute(2);
fftpermute(4);
fftpermute(8);
fftpermute(16);
fftpermute(32);
fftpermute(64);

void fft1true(const int step, __local TypeComplex *data) { }
void fft1false(const int step, __local TypeComplex *data) { }

#define FFT_N_FORWARD(N, Nhalf, forward) \
    void fft##N##forward(const int step, __local TypeComplex *data) { \
        if (N >= 4) { \
            fft##Nhalf##forward(step, data); \
            fft##Nhalf##forward(step, data + (N / 2) * step); \
        } \
        \
        for (int i = 0; i < N / 2; i++) { \
            TypeComplex c0 = data[(i        ) * step]; \
            TypeComplex c1 = data[(i + N / 2) * step]; \
            data[(i        ) * step] = fft_calc0_##N##forward(c0, c1, i); \
            data[(i + N / 2) * step] = fft_calc1_##N##forward(c0, c1, i); \
        } \
    }

FFT_N_FORWARD(2, 1, true)
FFT_N_FORWARD(2, 1, false)
FFT_N_FORWARD(4, 2, true)
FFT_N_FORWARD(4, 2, false)
FFT_N_FORWARD(8, 4, true)
FFT_N_FORWARD(8, 4, false)
FFT_N_FORWARD(16, 8, true)
FFT_N_FORWARD(16, 8, false)
FFT_N_FORWARD(32, 16, true)
FFT_N_FORWARD(32, 16, false)
FFT_N_FORWARD(64, 32, true)
FFT_N_FORWARD(64, 32, false)



#define IFFT_NORMALIZE(N) \
void ifft_normalize##N(const int step, __local TypeComplex *data) { \
    const TypeComplex invN = (TypeComplex)(1.0f / (float)N); \
    for (int i = 0; i < N; i++) { \
        data[i * step] *= invN; \
    } \
}

IFFT_NORMALIZE(2)
IFFT_NORMALIZE(4)
IFFT_NORMALIZE(8)
IFFT_NORMALIZE(16)
IFFT_NORMALIZE(32)
IFFT_NORMALIZE(64)

// select the twiddle table matching temporalCount (FW_TEMPORAL1..4, generated host-side)
#define FW_TEMPORAL_CAT2(a, b) a##b
#define FW_TEMPORAL_CAT(a, b) FW_TEMPORAL_CAT2(a, b)
#define FW_TEMPORAL_TBL FW_TEMPORAL_CAT(FW_TEMPORAL, temporalCount)

static void dft_tmprl(const bool forward, const int step, TypeComplex *data) {
    TypeComplex work[temporalCount];
    #pragma unroll
    for (int i = 0; i < temporalCount; i++) {
        work[i] = (TypeComplex)(0.0f);
    }
    #pragma unroll
    for (int i = 0; i < temporalCount; i++) {
        #pragma unroll
        for (int k = 0; k < temporalCount; k++) {
            work[k] += cmul(data[i * step], FW_TEMPORAL_TBL[forward ? 1 : 0][i*k]);
        }
    }
    const TypeComplex invN = (forward) ? (TypeComplex)1.0f : (TypeComplex)(1.0f / (float)temporalCount);
    #pragma unroll
    for (int i = 0; i < temporalCount; i++) {
        data[i * step] = work[i] * invN;
    }
}

#define FFT_BLOCK(BLOCK_N) \
    void fftBlock(__local TypeComplex shared_tmp[BLOCK_N][BLOCK_N + 1], const int thWorker) { \
        fftpermute##BLOCK_N(1, &shared_tmp[thWorker][0]); BLOCK_SYNC; \
        fft##BLOCK_N##true(1, &shared_tmp[thWorker][0]); BLOCK_SYNC; \
        fftpermute##BLOCK_N(BLOCK_N+1, &shared_tmp[0][thWorker]); BLOCK_SYNC; \
        fft##BLOCK_N##true(BLOCK_N+1, &shared_tmp[0][thWorker]); \
    }


#define IFFT_BLOCK(BLOCK_N) \
    void ifftBlock(__local TypeComplex shared_tmp[BLOCK_N][BLOCK_N + 1], const int thWorker) { \
        fftpermute##BLOCK_N(BLOCK_N + 1, &shared_tmp[0][thWorker]); BLOCK_SYNC; \
        fft##BLOCK_N##false(BLOCK_N + 1, &shared_tmp[0][thWorker]); BLOCK_SYNC; \
        ifft_normalize##BLOCK_N(BLOCK_N + 1, &shared_tmp[0][thWorker]); BLOCK_SYNC; \
        fftpermute##BLOCK_N(1, &shared_tmp[thWorker][0]); BLOCK_SYNC; \
        fft##BLOCK_N##false(1, &shared_tmp[thWorker][0]); BLOCK_SYNC; \
        ifft_normalize##BLOCK_N(1, &shared_tmp[thWorker][0]); \
    }

#if BLOCK_SIZE == 2
FFT_BLOCK(2);
IFFT_BLOCK(2);
#elif BLOCK_SIZE == 4 
FFT_BLOCK(4);
IFFT_BLOCK(4);
#elif BLOCK_SIZE == 8
FFT_BLOCK(8);
IFFT_BLOCK(8);
#elif BLOCK_SIZE == 16
FFT_BLOCK(16);
IFFT_BLOCK(16);
#elif BLOCK_SIZE == 32
FFT_BLOCK(32);
IFFT_BLOCK(32);
#elif BLOCK_SIZE == 64
FFT_BLOCK(64);
IFFT_BLOCK(64);
#endif

__kernel void kernel_fft(
    __global char *const __restrict__ ptrDst0,
    __global char *const __restrict__ ptrDst1,
    const int dstPitch,
    const __global char *const __restrict__ ptrSrc0,
    const __global char *const __restrict__ ptrSrc1,
    const int srcPitch,
    const int width, const int height,
    const int block_count_x,
    const __global float *const __restrict__ ptrBlockWindow,
    const int ov1, const int ov2
) {
    const int thWorker = get_local_id(0); // BLOCK_SIZE
    const int local_bx = get_local_id(1); // DENOISE_BLOCK_SIZE_X
    const int global_bx = get_group_id(0) * DENOISE_BLOCK_SIZE_X + local_bx;
    const int global_by = get_group_id(1);
    const int plane_idx = get_group_id(2);

    const int block_eff = BLOCK_SIZE - ov1 - ov1 - ov2;
    const int block_x = global_bx * block_eff - ov1 - ov2;
    const int block_y = global_by * block_eff - ov1 - ov2;

    __global char *const __restrict__ ptrDst = selectptrdst(ptrDst0, ptrDst1, plane_idx);
    const __global char *const __restrict__ ptrSrc = selectptr(ptrSrc0, ptrSrc1, plane_idx);

    __local TypeComplex stmp[DENOISE_BLOCK_SIZE_X][BLOCK_SIZE][BLOCK_SIZE + 1];

    // stmpにptrSrcの該当位置からデータを読み込む
    {
        const float winFuncX = ptrBlockWindow[thWorker];
        #pragma unroll
        for (int y = 0; y < BLOCK_SIZE; y++) {
            if (global_bx < block_count_x) {
                const int src_x = wrap_idx(block_x + thWorker, 0, width - 1);
                const int src_y = wrap_idx(block_y + y, 0, height - 1);
                const __global TypePixel *ptr_src = (const __global TypePixel *)(ptrSrc + src_y * srcPitch + src_x * sizeof(TypePixel));
                stmp[local_bx][y][thWorker] = (TypeComplex)((float)ptr_src[0] * winFuncX * ptrBlockWindow[y] * (1.0f / (float)((1 << bit_depth) - 1)), 0.0f);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    fftBlock(stmp[local_bx], thWorker);

    barrier(CLK_LOCAL_MEM_FENCE);

    // 計算内容をptrDstに出力
    #pragma unroll
    for (int y = 0; y < BLOCK_SIZE; y++) {
        if (global_bx < block_count_x) {
            const int dst_x = global_bx * BLOCK_SIZE + thWorker;
            const int dst_y = global_by * BLOCK_SIZE + y;
            __global TypeComplex *ptr_dst = (__global TypeComplex *)(ptrDst + dst_y * dstPitch + dst_x * sizeof(TypeComplex));
            ptr_dst[0] = stmp[local_bx][y][thWorker];
        }
    }
}

TypeComplex temporal_filter(
    const TypeComplex srcA,
    const TypeComplex srcB,
    const TypeComplex srcC,
    const TypeComplex srcD,
    const float sigma, const float limit,
    const float wsharpen, const float sminSq, const float smaxSq) {
    TypeComplex work[temporalCount];
    work[0] = srcA;
    if (temporalCount >= 2) { work[1] = srcB; }
    if (temporalCount >= 3) { work[2] = srcC; }
    if (temporalCount >= 4) { work[3] = srcD; }

    if (temporalCount >= 2) {
        dft_tmprl(true, 1, work);
    }

    // filterMethod < 0 -> no denoising (bt=-1: sharpen/degrid only)
    if (filterMethod >= 0) {
        #pragma unroll
        for (int z = 0; z < temporalCount; z++) {
            const float power = csquare(work[z]);

            float factor;
            if (filterMethod == 0) {
                factor = max(limit, (power - sigma) * native_recip(power + 1e-15f));
            } else {
                factor = power < sigma ? limit : 1.0f;
            }
            work[z] *= (TypeComplex)factor;
        }
    }

    if (temporalCount >= 2) {
        dft_tmprl(false, 1, work);
    }

    TypeComplex result = work[temporalCurrentIdx];
#if useSharpen
    // frequency-domain sharpening (after denoising): amplify mid amplitudes only.
    // weak bins (psd ~< sminSq) stay put to avoid boosting noise, strong bins
    // (psd ~> smaxSq) stay put to avoid oversharpening/halo; wsharpen carries
    // the sharpen strength and the per-bin gaussian high-pass frequency weight.
    {
        const float psd = csquare(result);
        const float sfact = 1.0f + wsharpen * native_sqrt(psd * smaxSq * native_recip((psd + sminSq) * (psd + smaxSq) + 1e-30f));
        result *= (TypeComplex)sfact;
    }
#endif
    return result;
}

__kernel void kernel_tfft_filter_ifft(
    __global char *const __restrict__ ptrDst0,
    __global char *const __restrict__ ptrDst1,
    const int dstPitch,
    const __global char *const __restrict__ ptrSrcA0,
    const __global char *const __restrict__ ptrSrcA1,
    const __global char *const __restrict__ ptrSrcB0,
    const __global char *const __restrict__ ptrSrcB1,
    const __global char *const __restrict__ ptrSrcC0,
    const __global char *const __restrict__ ptrSrcC1,
    const __global char *const __restrict__ ptrSrcD0,
    const __global char *const __restrict__ ptrSrcD1,
    const int srcPitch,
    const int block_count_x,
    const __global float *const __restrict__ ptrBlockWindowInverse,
    const int ov1, const int ov2,
    const __global float *const __restrict__ ptrSigma, const float limit,
    const __global float *const __restrict__ ptrWSharpen, const float sminSq, const float smaxSq,
    const __global float *const __restrict__ ptrGridSample, const float degridFactor
) {
    const int thWorker = get_local_id(0); // BLOCK_SIZE
    const int local_bx = get_local_id(1); // DENOISE_BLOCK_SIZE_X
    const int global_bx = get_group_id(0) * DENOISE_BLOCK_SIZE_X + local_bx;
    const int global_by = get_group_id(1);
    const int plane_idx = get_group_id(2);

    __global char *const __restrict__ ptrDst = selectptrdst(ptrDst0, ptrDst1, plane_idx);
    const __global char *const __restrict__ ptrSrcA = selectptr(ptrSrcA0, ptrSrcA1, plane_idx);
    const __global char *const __restrict__ ptrSrcB = (temporalCount >= 2) ? selectptr(ptrSrcB0, ptrSrcB1, plane_idx) : 0;
    const __global char *const __restrict__ ptrSrcC = (temporalCount >= 3) ? selectptr(ptrSrcC0, ptrSrcC1, plane_idx) : 0;
    const __global char *const __restrict__ ptrSrcD = (temporalCount >= 4) ? selectptr(ptrSrcD0, ptrSrcD1, plane_idx) : 0;

    __local TypeComplex stmp[DENOISE_BLOCK_SIZE_X][BLOCK_SIZE][BLOCK_SIZE + 1];

#if useDegrid
    // degrid: each frame's block DC scaled by degrid/gridDC. The grid correction
    // for a bin is gridsample[bin] * blockDC * degrid / gridDC - for a flat block
    // this equals the block's own (window-biased) spectrum, so subtracting it
    // before the wiener filter keeps the overlap-add window bias from being
    // modulated by the (amplitude-dependent) denoising.
    TypeComplex dcA = (TypeComplex)(0.0f), dcB = (TypeComplex)(0.0f), dcC = (TypeComplex)(0.0f), dcD = (TypeComplex)(0.0f);
    if (global_bx < block_count_x) {
        const int dc_idx = (global_by * BLOCK_SIZE) * srcPitch + (global_bx * BLOCK_SIZE) * sizeof(TypeComplex);
        dcA = ((const __global TypeComplex *)(ptrSrcA + dc_idx))[0] * (TypeComplex)degridFactor;
        if (temporalCount >= 2) { dcB = ((const __global TypeComplex *)(ptrSrcB + dc_idx))[0] * (TypeComplex)degridFactor; }
        if (temporalCount >= 3) { dcC = ((const __global TypeComplex *)(ptrSrcC + dc_idx))[0] * (TypeComplex)degridFactor; }
        if (temporalCount >= 4) { dcD = ((const __global TypeComplex *)(ptrSrcD + dc_idx))[0] * (TypeComplex)degridFactor; }
    }
#endif

    #pragma unroll
    for (int y = 0; y < BLOCK_SIZE; y++) {
        if (global_bx < block_count_x) {
            const int src_x = global_bx * BLOCK_SIZE + thWorker;
            const int src_y = global_by * BLOCK_SIZE + y;
            const int src_idx = src_y * srcPitch + src_x * sizeof(TypeComplex);
            // per-frequency-bin sigma (sigma/sigma2/sigma3/sigma4 interpolated on host)
            const float binSigma = ptrSigma[y * BLOCK_SIZE + thWorker];
            TypeComplex srcA = ((const __global TypeComplex *)(ptrSrcA + src_idx))[0];
            TypeComplex srcB = (TypeComplex)(0.0f), srcC = (TypeComplex)(0.0f), srcD = (TypeComplex)(0.0f);
            if (temporalCount >= 2) { srcB = ((const __global TypeComplex *)(ptrSrcB + src_idx))[0]; }
            if (temporalCount >= 3) { srcC = ((const __global TypeComplex *)(ptrSrcC + src_idx))[0]; }
            if (temporalCount >= 4) { srcD = ((const __global TypeComplex *)(ptrSrcD + src_idx))[0]; }
            TypeComplex corrCur = (TypeComplex)(0.0f);
#if useDegrid
            {
                const float2 gv = ((const __global float2 *)ptrGridSample)[y * BLOCK_SIZE + thWorker];
                const TypeComplex grid = (TypeComplex)(gv.x, gv.y);
                srcA -= cmul(grid, dcA);
                if (temporalCount >= 2) { srcB -= cmul(grid, dcB); }
                if (temporalCount >= 3) { srcC -= cmul(grid, dcC); }
                if (temporalCount >= 4) { srcD -= cmul(grid, dcD); }
                // current frame's correction, added back after filtering
                if (temporalCurrentIdx == 0) corrCur = cmul(grid, dcA);
                if (temporalCount >= 2 && temporalCurrentIdx == 1) corrCur = cmul(grid, dcB);
                if (temporalCount >= 3 && temporalCurrentIdx == 2) corrCur = cmul(grid, dcC);
                if (temporalCount >= 4 && temporalCurrentIdx == 3) corrCur = cmul(grid, dcD);
            }
#endif
#if useSharpen
            // sharpen applies to luma only: the host passes ptrWSharpen = null for
            // the chroma launch (same compiled kernel serves both launches)
            const float wsharpen = (ptrWSharpen) ? ptrWSharpen[y * BLOCK_SIZE + thWorker] : 0.0f;
#else
            const float wsharpen = 0.0f;
#endif
            TypeComplex result = temporal_filter(
                srcA, srcB, srcC, srcD,
                binSigma, limit,
                wsharpen, sminSq, smaxSq);
            result += corrCur;
            stmp[local_bx][y][thWorker] = result;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    ifftBlock(stmp[local_bx], thWorker);

    barrier(CLK_LOCAL_MEM_FENCE);
    {
        // 計算内容をptrDstに出力
        const float winFuncInvX = ptrBlockWindowInverse[thWorker];
        #pragma unroll
        for (int y = 0; y < BLOCK_SIZE; y++) {
            if (global_bx < block_count_x) {
                const int dst_x = global_bx * BLOCK_SIZE + thWorker;
                const int dst_y = global_by * BLOCK_SIZE + y;
                __global TypePixel *ptr_dst = (__global TypePixel *)(ptrDst + dst_y * dstPitch + dst_x * sizeof(TypePixel));
                ptr_dst[0] = (TypePixel)clamp((float)(stmp[local_bx][y][thWorker].x) * winFuncInvX * ptrBlockWindowInverse[y] * ((float)((1 << bit_depth) - 1)), 0.0f, (1 << bit_depth) - 1e-6f);
            }
        }
    }
}

__kernel void kernel_merge(
    __global char *const __restrict__ ptrDst0,
    __global char *const __restrict__ ptrDst1,
    const int dstPitch,
    const __global char *const __restrict__ ptrSrc0,
    const __global char *const __restrict__ ptrSrc1,
    const int srcPitch,
    const int width, const int height,
    const int block_count_x, const int block_count_y,
    const int ov1, const int ov2
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int plane_idx = get_group_id(2);
    const int block_eff = BLOCK_SIZE - ov1 - ov1 - ov2;

    __global char *const __restrict__ ptrDst = selectptrdst(ptrDst0, ptrDst1, plane_idx);
    const __global char *const __restrict__ ptrSrc = selectptr(ptrSrc0, ptrSrc1, plane_idx);

    if (x < width && y < height) {
        const int block_x = x / block_eff;
        const int block_y = y / block_eff;
        const int block_local_pos_x = x - block_x * block_eff + ov1 + ov2;
        const int block_local_pos_y = y - block_y * block_eff + ov1 + ov2;
        int shift = 0;
#define BLOCK_VAL(x, y) (((__global TypePixel *)(ptrSrc + (y) * srcPitch + (x) * sizeof(TypePixel)))[0])
        int pix = BLOCK_VAL(block_x * BLOCK_SIZE + block_local_pos_x, block_y * BLOCK_SIZE + block_local_pos_y);
        if (block_local_pos_x >= block_eff + ov1 && (block_x + 1) < block_count_x) {
            pix += BLOCK_VAL((block_x + 1) * BLOCK_SIZE + block_local_pos_x - block_eff, block_y * BLOCK_SIZE + block_local_pos_y);
            shift++;
        }
        if (block_local_pos_y >= block_eff + ov1 && (block_y + 1) < block_count_y) {
            pix += BLOCK_VAL(block_x * BLOCK_SIZE + block_local_pos_x, (block_y + 1) * BLOCK_SIZE + block_local_pos_y - block_eff);
            shift++;
            if (block_local_pos_x >= block_eff + ov1 && (block_x + 1) < block_count_x) {
                pix += BLOCK_VAL((block_x + 1) * BLOCK_SIZE + block_local_pos_x - block_eff, (block_y + 1) * BLOCK_SIZE + block_local_pos_y - block_eff);
            }
        }
#undef BLOCK_VAL
        ((__global TypePixel *)(ptrDst + y * dstPitch + x * sizeof(TypePixel)))[0] = (TypePixel)((pix + shift) >> shift);
    }
}
