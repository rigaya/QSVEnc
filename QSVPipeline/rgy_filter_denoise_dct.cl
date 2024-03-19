// bit_depth
// BLOCK_SIZE
// TypePixel
// TypeTmp
// STEP
// DENOISE_BLOCK_SIZE_X
// DENOISE_SHARED_BLOCK_NUM_X
// DENOISE_SHARED_BLOCK_NUM_Y
// DENOISE_LOOP_COUNT_BLOCK
// DCT_IDCT_BARRIER_MODE // 0... off, 1... barrier(), 2... sub_group_barrier

//#define DENOISE_BLOCK_SIZE_X (8) //ひとつのスレッドブロックの担当するx方向の8x8ブロックの数
//
//#define DENOISE_SHARED_BLOCK_NUM_X (DENOISE_BLOCK_SIZE_X+2) //sharedメモリ上のx方向の8x8ブロックの数
//#define DENOISE_SHARED_BLOCK_NUM_Y (2)                      //sharedメモリ上のy方向の8x8ブロックの数
//
//#define DENOISE_LOOP_COUNT_BLOCK (8)

#if DCT_IDCT_BARRIER_MODE == 1
#define DCT_IDCT_BARRIER(x) barrier(x)
#elif DCT_IDCT_BARRIER_MODE == 2
#define DCT_IDCT_BARRIER(x) sub_group_barrier(x)
#else
#define DCT_IDCT_BARRIER(x)
#endif

#define DCT3X3_0_0 ( 0.5773502691896258f) /*  1/sqrt(3) */
#define DCT3X3_0_1 ( 0.5773502691896258f) /*  1/sqrt(3) */
#define DCT3X3_0_2 ( 0.5773502691896258f) /*  1/sqrt(3) */
#define DCT3X3_1_0 ( 0.7071067811865475f) /*  1/sqrt(2) */
#define DCT3X3_1_2 (-0.7071067811865475f) /* -1/sqrt(2) */
#define DCT3X3_2_0 ( 0.4082482904638631f) /*  1/sqrt(6) */
#define DCT3X3_2_1 (-0.8164965809277261f) /* -2/sqrt(6) */
#define DCT3X3_2_2 ( 0.4082482904638631f) /*  1/sqrt(6) */

//CUDA Sampleより拝借
#define C_a 1.387039845322148f //!< a = (2^0.5) * cos(    pi / 16);  Used in forward and inverse DCT.
#define C_b 1.306562964876377f //!< b = (2^0.5) * cos(    pi /  8);  Used in forward and inverse DCT.
#define C_c 1.175875602419359f //!< c = (2^0.5) * cos(3 * pi / 16);  Used in forward and inverse DCT.
#define C_d 0.785694958387102f //!< d = (2^0.5) * cos(5 * pi / 16);  Used in forward and inverse DCT.
#define C_e 0.541196100146197f //!< e = (2^0.5) * cos(3 * pi /  8);  Used in forward and inverse DCT.
#define C_f 0.275899379282943f //!< f = (2^0.5) * cos(7 * pi / 16);  Used in forward and inverse DCT.

//Normalization constant that is used in forward and inverse DCT
#define C_norm 0.3535533905932737f // 1 / (8^0.5)

int wrap_idx(const int idx, const int min, const int max) {
    if (idx < min) {
        return min - idx;
    }
    if (idx > max) {
        return max - (idx - max);
    }
    return idx;
}

const __global char *selectptr(const __global char *ptr0, const __global char *ptr1, const __global char *ptr2, const int idx) {
    if (idx == 1) return ptr1;
    if (idx == 2) return ptr2;
    return ptr0;
}
__global char *selectptrdst(__global char *ptr0, __global char *ptr1, __global char *ptr2, const int idx) {
    if (idx == 1) return ptr1;
    if (idx == 2) return ptr2;
    return ptr0;
}


void CUDAsubroutineInplaceDCT8vector(__local TypeTmp *Vect0, const int Step) {
    __local TypeTmp *Vect1 = Vect0 + Step;
    __local TypeTmp *Vect2 = Vect1 + Step;
    __local TypeTmp *Vect3 = Vect2 + Step;
    __local TypeTmp *Vect4 = Vect3 + Step;
    __local TypeTmp *Vect5 = Vect4 + Step;
    __local TypeTmp *Vect6 = Vect5 + Step;
    __local TypeTmp *Vect7 = Vect6 + Step;

    TypeTmp X07P = (*Vect0) + (*Vect7);
    TypeTmp X16P = (*Vect1) + (*Vect6);
    TypeTmp X25P = (*Vect2) + (*Vect5);
    TypeTmp X34P = (*Vect3) + (*Vect4);

    TypeTmp X07M = (*Vect0) - (*Vect7);
    TypeTmp X61M = (*Vect6) - (*Vect1);
    TypeTmp X25M = (*Vect2) - (*Vect5);
    TypeTmp X43M = (*Vect4) - (*Vect3);

    TypeTmp X07P34PP = X07P + X34P;
    TypeTmp X07P34PM = X07P - X34P;
    TypeTmp X16P25PP = X16P + X25P;
    TypeTmp X16P25PM = X16P - X25P;

    (*Vect0) = (TypeTmp)(C_norm) * (X07P34PP + X16P25PP);
    (*Vect2) = (TypeTmp)(C_norm) * ((TypeTmp)(C_b) * X07P34PM + (TypeTmp)(C_e) * X16P25PM);
    (*Vect4) = (TypeTmp)(C_norm) * (X07P34PP - X16P25PP);
    (*Vect6) = (TypeTmp)(C_norm) * ((TypeTmp)(C_e) * X07P34PM - (TypeTmp)(C_b) * X16P25PM);

    (*Vect1) = (TypeTmp)(C_norm) * ((TypeTmp)(C_a) * X07M - (TypeTmp)(C_c) * X61M + (TypeTmp)(C_d) * X25M - (TypeTmp)(C_f) * X43M);
    (*Vect3) = (TypeTmp)(C_norm) * ((TypeTmp)(C_c) * X07M + (TypeTmp)(C_f) * X61M - (TypeTmp)(C_a) * X25M + (TypeTmp)(C_d) * X43M);
    (*Vect5) = (TypeTmp)(C_norm) * ((TypeTmp)(C_d) * X07M + (TypeTmp)(C_a) * X61M + (TypeTmp)(C_f) * X25M - (TypeTmp)(C_c) * X43M);
    (*Vect7) = (TypeTmp)(C_norm) * ((TypeTmp)(C_f) * X07M + (TypeTmp)(C_d) * X61M + (TypeTmp)(C_c) * X25M + (TypeTmp)(C_a) * X43M);
}

void CUDAsubroutineInplaceIDCT8vector(__local TypeTmp *Vect0, const int Step) {
    __local TypeTmp *Vect1 = Vect0 + Step;
    __local TypeTmp *Vect2 = Vect1 + Step;
    __local TypeTmp *Vect3 = Vect2 + Step;
    __local TypeTmp *Vect4 = Vect3 + Step;
    __local TypeTmp *Vect5 = Vect4 + Step;
    __local TypeTmp *Vect6 = Vect5 + Step;
    __local TypeTmp *Vect7 = Vect6 + Step;

    TypeTmp Y04P = (*Vect0) + (*Vect4);
    TypeTmp Y2b6eP = (TypeTmp)(C_b) * (*Vect2) + (TypeTmp)(C_e) * (*Vect6);

    TypeTmp Y04P2b6ePP = Y04P + Y2b6eP;
    TypeTmp Y04P2b6ePM = Y04P - Y2b6eP;
    TypeTmp Y7f1aP3c5dPP = (TypeTmp)(C_f) * (*Vect7) + (TypeTmp)(C_a) * (*Vect1) + (TypeTmp)(C_c) * (*Vect3) + (TypeTmp)(C_d) * (*Vect5);
    TypeTmp Y7a1fM3d5cMP = (TypeTmp)(C_a) * (*Vect7) - (TypeTmp)(C_f) * (*Vect1) + (TypeTmp)(C_d) * (*Vect3) - (TypeTmp)(C_c) * (*Vect5);

    TypeTmp Y04M = (*Vect0) - (*Vect4);
    TypeTmp Y2e6bM = (TypeTmp)(C_e) * (*Vect2) - (TypeTmp)(C_b) * (*Vect6);

    TypeTmp Y04M2e6bMP = Y04M + Y2e6bM;
    TypeTmp Y04M2e6bMM = Y04M - Y2e6bM;
    TypeTmp Y1c7dM3f5aPM = (TypeTmp)(C_c) * (*Vect1) - (TypeTmp)(C_d) * (*Vect7) - (TypeTmp)(C_f) * (*Vect3) - (TypeTmp)(C_a) * (*Vect5);
    TypeTmp Y1d7cP3a5fMM = (TypeTmp)(C_d) * (*Vect1) + (TypeTmp)(C_c) * (*Vect7) - (TypeTmp)(C_a) * (*Vect3) + (TypeTmp)(C_f) * (*Vect5);

    (*Vect0) = (TypeTmp)(C_norm) * (Y04P2b6ePP + Y7f1aP3c5dPP);
    (*Vect7) = (TypeTmp)(C_norm) * (Y04P2b6ePP - Y7f1aP3c5dPP);
    (*Vect4) = (TypeTmp)(C_norm) * (Y04P2b6ePM + Y7a1fM3d5cMP);
    (*Vect3) = (TypeTmp)(C_norm) * (Y04P2b6ePM - Y7a1fM3d5cMP);

    (*Vect1) = (TypeTmp)(C_norm) * (Y04M2e6bMP + Y1c7dM3f5aPM);
    (*Vect5) = (TypeTmp)(C_norm) * (Y04M2e6bMM - Y1d7cP3a5fMM);
    (*Vect2) = (TypeTmp)(C_norm) * (Y04M2e6bMM + Y1d7cP3a5fMM);
    (*Vect6) = (TypeTmp)(C_norm) * (Y04M2e6bMP - Y1c7dM3f5aPM);
}

void  CUDAsubroutineInplaceDCT16vector(__local TypeTmp *Vect00, const int Step) {
    __local TypeTmp *Vect01 = Vect00 + Step;
    __local TypeTmp *Vect02 = Vect01 + Step;
    __local TypeTmp *Vect03 = Vect02 + Step;
    __local TypeTmp *Vect04 = Vect03 + Step;
    __local TypeTmp *Vect05 = Vect04 + Step;
    __local TypeTmp *Vect06 = Vect05 + Step;
    __local TypeTmp *Vect07 = Vect06 + Step;

    __local TypeTmp *Vect08 = Vect00 + (Step << 3);
    __local TypeTmp *Vect09 = Vect08 + Step;
    __local TypeTmp *Vect10 = Vect09 + Step;
    __local TypeTmp *Vect11 = Vect10 + Step;
    __local TypeTmp *Vect12 = Vect11 + Step;
    __local TypeTmp *Vect13 = Vect12 + Step;
    __local TypeTmp *Vect14 = Vect13 + Step;
    __local TypeTmp *Vect15 = Vect14 + Step;

    const float x00 = (*Vect00) + (*Vect15);
    const float x01 = (*Vect01) + (*Vect14);
    const float x02 = (*Vect02) + (*Vect13);
    const float x03 = (*Vect03) + (*Vect12);
    const float x04 = (*Vect04) + (*Vect11);
    const float x05 = (*Vect05) + (*Vect10);
    const float x06 = (*Vect06) + (*Vect09);
    const float x07 = (*Vect07) + (*Vect08);
    const float x08 = (*Vect00) - (*Vect15);
    const float x09 = (*Vect01) - (*Vect14);
    const float x0a = (*Vect02) - (*Vect13);
    const float x0b = (*Vect03) - (*Vect12);
    const float x0c = (*Vect04) - (*Vect11);
    const float x0d = (*Vect05) - (*Vect10);
    const float x0e = (*Vect06) - (*Vect09);
    const float x0f = (*Vect07) - (*Vect08);
    const float x10 = x00 + x07;
    const float x11 = x01 + x06;
    const float x12 = x02 + x05;
    const float x13 = x03 + x04;
    const float x14 = x00 - x07;
    const float x15 = x01 - x06;
    const float x16 = x02 - x05;
    const float x17 = x03 - x04;
    const float x18 = x10 + x13;
    const float x19 = x11 + x12;
    const float x1a = x10 - x13;
    const float x1b = x11 - x12;
    const float x1c =   1.38703984532215f*x14 + 0.275899379282943f*x17;
    const float x1d =   1.17587560241936f*x15 + 0.785694958387102f*x16;
    const float x1e = -0.785694958387102f*x15 + 1.17587560241936f *x16;
    const float x1f =  0.275899379282943f*x14 - 1.38703984532215f *x17;
    const float x20 = 0.25f * (x1c - x1d);
    const float x21 = 0.25f * (x1e - x1f);
    const float x22 =  1.40740373752638f *x08 + 0.138617169199091f*x0f;
    const float x23 =  1.35331800117435f *x09 + 0.410524527522357f*x0e;
    const float x24 =  1.24722501298667f *x0a + 0.666655658477747f*x0d;
    const float x25 =  1.09320186700176f *x0b + 0.897167586342636f*x0c;
    const float x26 = -0.897167586342636f*x0b + 1.09320186700176f *x0c;
    const float x27 =  0.666655658477747f*x0a - 1.24722501298667f *x0d;
    const float x28 = -0.410524527522357f*x09 + 1.35331800117435f *x0e;
    const float x29 =  0.138617169199091f*x08 - 1.40740373752638f *x0f;
    const float x2a = x22 + x25;
    const float x2b = x23 + x24;
    const float x2c = x22 - x25;
    const float x2d = x23 - x24;
    const float x2e = 0.25f * (x2a - x2b);
    const float x2f = 0.326640741219094f*x2c + 0.135299025036549f*x2d;
    const float x30 = 0.135299025036549f*x2c - 0.326640741219094f*x2d;
    const float x31 = x26 + x29;
    const float x32 = x27 + x28;
    const float x33 = x26 - x29;
    const float x34 = x27 - x28;
    const float x35 = 0.25f * (x31 - x32);
    const float x36 = 0.326640741219094f*x33 + 0.135299025036549f*x34;
    const float x37 = 0.135299025036549f*x33 - 0.326640741219094f*x34;
    (*Vect00) = 0.25f * (x18 + x19);
    (*Vect01) = 0.25f * (x2a + x2b);
    (*Vect02) = 0.25f * (x1c + x1d);
    (*Vect03) = 0.707106781186547f * (x2f - x37);
    (*Vect04) = 0.326640741219094f * x1a + 0.135299025036549f * x1b;
    (*Vect05) = 0.707106781186547f * (x2f + x37);
    (*Vect06) = 0.707106781186547f * (x20 - x21);
    (*Vect07) = 0.707106781186547f * (x2e + x35);
    (*Vect08) = 0.25f * (x18 - x19);
    (*Vect09) = 0.707106781186547f * (x2e - x35);
    (*Vect10) = 0.707106781186547f * (x20 + x21);
    (*Vect11) = 0.707106781186547f * (x30 - x36);
    (*Vect12) = 0.135299025036549f*x1a - 0.326640741219094f*x1b;
    (*Vect13) = 0.707106781186547f * (x30 + x36);
    (*Vect14) = 0.25f * (x1e + x1f);
    (*Vect15) = 0.25f * (x31 + x32);
}

void  CUDAsubroutineInplaceIDCT16vector(__local TypeTmp *Vect00, const int Step) {
    __local TypeTmp *Vect01 = Vect00 + Step;
    __local TypeTmp *Vect02 = Vect01 + Step;
    __local TypeTmp *Vect03 = Vect02 + Step;
    __local TypeTmp *Vect04 = Vect03 + Step;
    __local TypeTmp *Vect05 = Vect04 + Step;
    __local TypeTmp *Vect06 = Vect05 + Step;
    __local TypeTmp *Vect07 = Vect06 + Step;

    __local TypeTmp *Vect08 = Vect00 + (Step << 3);
    __local TypeTmp *Vect09 = Vect08 + Step;
    __local TypeTmp *Vect10 = Vect09 + Step;
    __local TypeTmp *Vect11 = Vect10 + Step;
    __local TypeTmp *Vect12 = Vect11 + Step;
    __local TypeTmp *Vect13 = Vect12 + Step;
    __local TypeTmp *Vect14 = Vect13 + Step;
    __local TypeTmp *Vect15 = Vect14 + Step;

    const float x00 =  1.4142135623731f   * (*Vect00);
    const float x01 =  1.40740373752638f  * (*Vect01) + 0.138617169199091f * (*Vect15);
    const float x02 =  1.38703984532215f  * (*Vect02) + 0.275899379282943f * (*Vect14);
    const float x03 =  1.35331800117435f  * (*Vect03) + 0.410524527522357f * (*Vect13);
    const float x04 =  1.30656296487638f  * (*Vect04) + 0.541196100146197f * (*Vect12);
    const float x05 =  1.24722501298667f  * (*Vect05) + 0.666655658477747f * (*Vect11);
    const float x06 =  1.17587560241936f  * (*Vect06) + 0.785694958387102f * (*Vect10);
    const float x07 =  1.09320186700176f  * (*Vect07) + 0.897167586342636f * (*Vect09);
    const float x08 =  1.4142135623731f   * (*Vect08);
    const float x09 = -0.897167586342636f * (*Vect07) + 1.09320186700176f * (*Vect09);
    const float x0a =  0.785694958387102f * (*Vect06) - 1.17587560241936f * (*Vect10);
    const float x0b = -0.666655658477747f * (*Vect05) + 1.24722501298667f * (*Vect11);
    const float x0c =  0.541196100146197f * (*Vect04) - 1.30656296487638f * (*Vect12);
    const float x0d = -0.410524527522357f * (*Vect03) + 1.35331800117435f * (*Vect13);
    const float x0e =  0.275899379282943f * (*Vect02) - 1.38703984532215f * (*Vect14);
    const float x0f = -0.138617169199091f * (*Vect01) + 1.40740373752638f * (*Vect15);
    const float x12 = x00 + x08;
    const float x13 = x01 + x07;
    const float x14 = x02 + x06;
    const float x15 = x03 + x05;
    const float x16 = 1.4142135623731f*x04;
    const float x17 = x00 - x08;
    const float x18 = x01 - x07;
    const float x19 = x02 - x06;
    const float x1a = x03 - x05;
    const float x1d = x12 + x16;
    const float x1e = x13 + x15;
    const float x1f = 1.4142135623731f*x14;
    const float x20 = x12 - x16;
    const float x21 = x13 - x15;
    const float x22 = 0.25f * (x1d - x1f);
    const float x23 = 0.25f * (x20 + x21);
    const float x24 = 0.25f * (x20 - x21);
    const float x25 = 1.4142135623731f*x17;
    const float x26 = 1.30656296487638f*x18 + 0.541196100146197f*x1a;
    const float x27 = 1.4142135623731f*x19;
    const float x28 = -0.541196100146197f*x18 + 1.30656296487638f*x1a;
    const float x29 = 0.176776695296637f * (x25 + x27) + 0.25f*x26;
    const float x2a = 0.25f * (x25 - x27);
    const float x2b = 0.176776695296637f * (x25 + x27) - 0.25f*x26;
    const float x2c = 0.353553390593274f*x28;
    const float x1b = 0.707106781186547f * (x2a - x2c);
    const float x1c = 0.707106781186547f * (x2a + x2c);
    const float x2d = 1.4142135623731f*x0c;
    const float x2e = x0b + x0d;
    const float x2f = x0a + x0e;
    const float x30 = x09 + x0f;
    const float x31 = x09 - x0f;
    const float x32 = x0a - x0e;
    const float x33 = x0b - x0d;
    const float x37 = 1.4142135623731f*x2d;
    const float x38 = 1.30656296487638f*x2e + 0.541196100146197f*x30;
    const float x39 = 1.4142135623731f*x2f;
    const float x3a = -0.541196100146197f*x2e + 1.30656296487638f*x30;
    const float x3b = 0.176776695296637f * (x37 + x39) + 0.25f*x38;
    const float x3c = 0.25f * (x37 - x39);
    const float x3d = 0.176776695296637f * (x37 + x39) - 0.25f*x38;
    const float x3e = 0.353553390593274f*x3a;
    const float x34 = 0.707106781186547f * (x3c - x3e);
    const float x35 = 0.707106781186547f * (x3c + x3e);
    const float x3f = 1.4142135623731f*x32;
    const float x40 = x31 + x33;
    const float x41 = x31 - x33;
    const float x42 = 0.25f * (x3f + x40);
    const float x43 = 0.25f * (x3f - x40);
    const float x44 = 0.353553390593274f*x41;
    (*Vect00) = 0.176776695296637f * (x1d + x1f) + 0.25f * x1e;
    (*Vect01) = 0.707106781186547f * (x29 + x3d);
    (*Vect02) = 0.707106781186547f * (x29 - x3d);
    (*Vect03) = 0.707106781186547f * (x23 - x43);
    (*Vect04) = 0.707106781186547f * (x23 + x43);
    (*Vect05) = 0.707106781186547f * (x1b - x35);
    (*Vect06) = 0.707106781186547f * (x1b + x35);
    (*Vect07) = 0.707106781186547f * (x22 + x44);
    (*Vect08) = 0.707106781186547f * (x22 - x44);
    (*Vect09) = 0.707106781186547f * (x1c + x34);
    (*Vect10) = 0.707106781186547f * (x1c - x34);
    (*Vect11) = 0.707106781186547f * (x24 + x42);
    (*Vect12) = 0.707106781186547f * (x24 - x42);
    (*Vect13) = 0.707106781186547f * (x2b - x3b);
    (*Vect14) = 0.707106781186547f * (x2b + x3b);
    (*Vect15) = 0.176776695296637f * (x1d + x1f) - 0.25f*x1e;
}

//こうしたバリアには全スレッドが通るようにしないとRX5500などでは正常に動作しない (他の箇所でbarrierしても意味がない)
//なので、計算の有無はenableフラグで切り替える
void dctBlock(const bool enable, __local TypeTmp shared_tmp[BLOCK_SIZE][BLOCK_SIZE + 1], const int thWorker) {
    //static_assert(BLOCK_SIZE == 8 || BLOCK_SIZE == 16, "BLOCK_SIZE must be 8 or 16");
    if (BLOCK_SIZE == 8) {
        DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
        if (enable) CUDAsubroutineInplaceDCT8vector((__local TypeTmp *)&shared_tmp[thWorker][0], 1); // row
        DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
        if (enable) CUDAsubroutineInplaceDCT8vector((__local TypeTmp *)&shared_tmp[0][thWorker], BLOCK_SIZE + 1); // column
        DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
    } else if (BLOCK_SIZE == 16) {
        DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
        if (enable) CUDAsubroutineInplaceDCT16vector((__local TypeTmp *)&shared_tmp[thWorker][0], 1); // row
        DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
        if (enable) CUDAsubroutineInplaceDCT16vector((__local TypeTmp *)&shared_tmp[0][thWorker], BLOCK_SIZE + 1); // column
        DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
    }
}

void idctBlock(const bool enable, __local TypeTmp shared_tmp[BLOCK_SIZE][BLOCK_SIZE + 1], const int thWorker) {
    //static_assert(BLOCK_SIZE == 8 || BLOCK_SIZE == 16, "BLOCK_SIZE must be 8 or 16");
    if (BLOCK_SIZE == 8) {
        DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
        if (enable) CUDAsubroutineInplaceIDCT8vector((__local TypeTmp *)&shared_tmp[0][thWorker], BLOCK_SIZE + 1); // column
        DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
        if (enable) CUDAsubroutineInplaceIDCT8vector((__local TypeTmp *)&shared_tmp[thWorker][0], 1); // row
        DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
    } else if (BLOCK_SIZE == 16) {
        DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
        if (enable) CUDAsubroutineInplaceIDCT16vector((__local TypeTmp *)&shared_tmp[0][thWorker], BLOCK_SIZE + 1); // column
        DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
        if (enable) CUDAsubroutineInplaceIDCT16vector((__local TypeTmp *)&shared_tmp[thWorker][0], 1); // row
        DCT_IDCT_BARRIER(CLK_LOCAL_MEM_FENCE);
    }
}

void thresholdBlock(__local TypeTmp shared_tmp[BLOCK_SIZE][BLOCK_SIZE + 1], const int thWorker, const float threshold) {
    #pragma unroll
    for (int y = 0; y < BLOCK_SIZE; y++) {
        if (y > 0 || thWorker > 0) {
            __local TypeTmp *ptr = &shared_tmp[y][thWorker];
            const TypeTmp val = ptr[0];
            if (fabs(val) <= threshold) {
                ptr[0] = 0.0f;
            }
        }
    }
}

#define SHARED_TMP __local TypeTmp shared_tmp[DENOISE_BLOCK_SIZE_X][BLOCK_SIZE][BLOCK_SIZE + 1]
#define SHARED_OUT __local TypeTmp shared_out[BLOCK_SIZE * DENOISE_SHARED_BLOCK_NUM_Y][BLOCK_SIZE * DENOISE_SHARED_BLOCK_NUM_X]


void clearSharedOutLine(
    SHARED_OUT,
    const int local_bx,
    const int thWorker,
    const int sy
) {
    const int y = sy % (BLOCK_SIZE * DENOISE_SHARED_BLOCK_NUM_Y);
    shared_out[y][local_bx * BLOCK_SIZE + thWorker] = 0;
    if (local_bx < (DENOISE_SHARED_BLOCK_NUM_X - DENOISE_BLOCK_SIZE_X)) {
        shared_out[y][(local_bx + DENOISE_BLOCK_SIZE_X) * BLOCK_SIZE + thWorker] = 0;
    }
}


void clearSharedOut(
    SHARED_OUT,
    const int local_bx,
    const int thWorker
) {
    #pragma unroll
    for (int y = 0; y < BLOCK_SIZE * DENOISE_SHARED_BLOCK_NUM_Y; y++) {
        clearSharedOutLine(shared_out, local_bx, thWorker, y);
    }
}

void loadBlocktmp(
    SHARED_TMP,
    const int local_bx, const int thWorker,
    const __global char *const __restrict__ ptrSrc, const int srcPitch,
    const int block_x, const int block_y,
    const int width, const int height) {
    #pragma unroll
    for (int y = 0; y < BLOCK_SIZE; y++) {
        const int src_x = wrap_idx(block_x + thWorker, 0, width  - 1);
        const int src_y = wrap_idx(block_y + y,        0, height - 1);
        TypePixel pix = ((const __global TypePixel *)(ptrSrc + src_y * srcPitch + src_x * sizeof(TypePixel)))[0];
        shared_tmp[local_bx][y][thWorker] = (TypeTmp)pix;
    }
}

void addBlocktmp(
    SHARED_OUT,
    const int shared_block_x, const int shared_block_y,
    const SHARED_TMP,
    const int local_bx, const int thWorker) {
    #pragma unroll
    for (int y = 0; y < BLOCK_SIZE; y++) {
        shared_out[(shared_block_y + y) % (BLOCK_SIZE * DENOISE_SHARED_BLOCK_NUM_Y)][shared_block_x + thWorker]
            += shared_tmp[local_bx][y][thWorker];
    }
}

// デバッグ用
void directAddBlock(
    SHARED_OUT,
    const int shared_block_x, const int shared_block_y,
    const int thWorker,
    const __global char *const __restrict__ ptrSrc, const int srcPitch,
    const int block_x, const int block_y,
    const int width, const int height) {
    #pragma unroll
    for (int y = 0; y < BLOCK_SIZE; y++) {
        const int src_x = wrap_idx(block_x + thWorker, 0, width - 1);
        const int src_y = wrap_idx(block_y + y,        0, height - 1);
        TypePixel pix = ((const __global TypePixel *)(ptrSrc + src_y * srcPitch + src_x * sizeof(TypePixel)))[0];
        shared_out[(shared_block_y + y) % (BLOCK_SIZE * DENOISE_SHARED_BLOCK_NUM_Y)][shared_block_x + thWorker] += pix;
    }
}

void filter_block(
    const bool enable,
    const __global char *const __restrict__ ptrSrc, const int srcPitch,
    SHARED_TMP,
    SHARED_OUT,
    const int local_bx, const int thWorker,
    const int shared_block_x, const int shared_block_y,
    const int block_x, const int block_y,
    const int width, const int height,
    const float threshold) {
#if 1
    if (enable) loadBlocktmp(shared_tmp, local_bx, thWorker, ptrSrc, srcPitch, block_x, block_y, width, height);
    dctBlock(enable, shared_tmp[local_bx], thWorker);
    thresholdBlock(shared_tmp[local_bx], thWorker, threshold);
    idctBlock(enable, shared_tmp[local_bx], thWorker);
    if (enable) addBlocktmp(shared_out, shared_block_x, shared_block_y, shared_tmp, local_bx, thWorker);
#else
    if (enable) directAddBlock(shared_out, shared_block_x, shared_block_y, thWorker, ptrSrc, srcPitch, block_x, block_y, width, height);
#endif
}

void write_output(
    __global char *const __restrict__ ptrDst, const int dstPitch,
    SHARED_OUT,
    const int width, const int height,
    const int sx, const int sy, 
    const int x, const int y) {
    if (x < width && y < height) {
        __global TypePixel *dst = (__global TypePixel*)(ptrDst + y * dstPitch + x * sizeof(TypePixel));
        const __local TypeTmp *out = &shared_out[sy % (BLOCK_SIZE * DENOISE_SHARED_BLOCK_NUM_Y)][sx];
        const float weight = (1.0f / (float)(BLOCK_SIZE * BLOCK_SIZE / (STEP * STEP)));
        dst[0] = out[0] * weight;
    }
}

__kernel void kernel_denoise_dct(
    __global char *const __restrict__ ptrDst0,
    __global char *const __restrict__ ptrDst1,
    __global char *const __restrict__ ptrDst2,
    const int dstPitch,
    const __global char *const __restrict__ ptrSrc0,
    const __global char *const __restrict__ ptrSrc1,
    const __global char *const __restrict__ ptrSrc2,
    const int srcPitch,
    const int width, const int height,
    const float threshold) {
    const int thWorker = get_local_id(0); // BLOCK_SIZE
    const int local_bx = get_local_id(1); // DENOISE_BLOCK_SIZE_X
    const int global_bx = get_group_id(0) * DENOISE_BLOCK_SIZE_X + local_bx;
    const int global_by = get_group_id(1) * DENOISE_LOOP_COUNT_BLOCK;
    const int plane_idx = get_group_id(2);

    const int block_x = global_bx * BLOCK_SIZE;
    const int block_y = global_by * BLOCK_SIZE;

    __global char *const __restrict__ ptrDst = selectptrdst(ptrDst0, ptrDst1, ptrDst2, plane_idx);
    const __global char *const __restrict__ ptrSrc = selectptr(ptrSrc0, ptrSrc1, ptrSrc2, plane_idx);

    SHARED_TMP;
    SHARED_OUT;

    #define FILTER_BLOCK(enable, SHARED_X, SHARED_Y, X, Y) \
        { filter_block((enable), ptrSrc, srcPitch, shared_tmp, shared_out, local_bx, thWorker, (SHARED_X), (SHARED_Y), (X), (Y), width, height, threshold); }

    { // SHARED_OUTの初期化
        clearSharedOut(shared_out, local_bx, thWorker);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    { // y方向の事前計算
        const int block_y_start = (block_y - BLOCK_SIZE) + STEP;
        for (int y = block_y_start; y < block_y; y += STEP) {
            const int shared_y = y - (block_y - BLOCK_SIZE);
            for (int ix_loop = 0; ix_loop < BLOCK_SIZE; ix_loop += STEP) {
                const int x = block_x + ix_loop;
                const int shared_x = local_bx * BLOCK_SIZE + ix_loop;
                { // local_bx < 1 のときのみ実行、enable引数で切りかえる
                    FILTER_BLOCK(local_bx < 1, shared_x, shared_y, x - BLOCK_SIZE, y); // x方向の事前計算
                }
                FILTER_BLOCK(true, shared_x + BLOCK_SIZE, shared_y, x, y);
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
    }

    { // 本計算
        const int block_y_fin = min(height, block_y + DENOISE_LOOP_COUNT_BLOCK * BLOCK_SIZE);
        for (int y = block_y; y < block_y_fin; y += STEP) {
            const int shared_y = y - (block_y - BLOCK_SIZE);
            for (int ix_loop = 0; ix_loop < BLOCK_SIZE; ix_loop += STEP) {
                const int x = block_x + ix_loop;
                const int shared_x = local_bx * BLOCK_SIZE + ix_loop;
                { // local_bx < 1 のときのみ実行、enable引数で切りかえる
                    FILTER_BLOCK(local_bx < 1, shared_x, shared_y, x - BLOCK_SIZE, y);
                }
                FILTER_BLOCK(true, shared_x + BLOCK_SIZE, shared_y, x, y);
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            for (int iy = 0; iy < STEP; iy++) {
                write_output(ptrDst, dstPitch, shared_out, width, height,
                    (local_bx + 1 /*1ブロック分ずれている*/) * BLOCK_SIZE + thWorker, shared_y + iy, block_x + thWorker, y + iy);

                clearSharedOutLine(shared_out, local_bx, thWorker, shared_y + iy + BLOCK_SIZE /*1ブロック先をクリア*/);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    #undef FILTER_BLOCK
}

__kernel void kernel_color_decorrelation(
    __global uchar *__restrict__ dst0, __global uchar *__restrict__ dst1, __global uchar *__restrict__ dst2, const int dstPitch,
    const __global uchar *__restrict__ src0, const __global uchar *__restrict__ src1, const __global uchar *__restrict__ src2, const int srcPitch,
    const int width, const int height) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix < width && iy < height) {
        const float ptrSrc0 = (float)(((const __global TypePixel *)(src0 + iy * srcPitch + ix * sizeof(TypePixel)))[0]);
        const float ptrSrc1 = (float)(((const __global TypePixel *)(src1 + iy * srcPitch + ix * sizeof(TypePixel)))[0]);
        const float ptrSrc2 = (float)(((const __global TypePixel *)(src2 + iy * srcPitch + ix * sizeof(TypePixel)))[0]);

        const float d0 = ptrSrc0 * DCT3X3_0_0 + ptrSrc1 * DCT3X3_0_1 + ptrSrc2 * DCT3X3_0_2;
        const float d1 = ptrSrc0 * DCT3X3_1_0 +                        ptrSrc2 * DCT3X3_1_2;
        const float d2 = ptrSrc0 * DCT3X3_2_0 + ptrSrc1 * DCT3X3_2_1 + ptrSrc2 * DCT3X3_2_2;

        __global TypePixel *ptrDst0 = (__global TypePixel *)(dst0 + iy * dstPitch + ix * sizeof(TypePixel));
        __global TypePixel *ptrDst1 = (__global TypePixel *)(dst1 + iy * dstPitch + ix * sizeof(TypePixel));
        __global TypePixel *ptrDst2 = (__global TypePixel *)(dst2 + iy * dstPitch + ix * sizeof(TypePixel));
        ptrDst0[0] = d0;
        ptrDst1[0] = d1;
        ptrDst2[0] = d2;
    }
}

__kernel void kernel_color_correlation(
    __global uchar *__restrict__ dst0, __global uchar *__restrict__ dst1, __global uchar *__restrict__ dst2, const int dstPitch,
    const __global uchar *__restrict__ src0, const __global uchar *__restrict__ src1, const __global uchar *__restrict__ src2, const int srcPitch,
    const int width, const int height) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix < width && iy < height) {
        const float ptrSrc0 = (float)(((const __global TypePixel *)(src0 + iy * srcPitch + ix * sizeof(TypePixel)))[0]);
        const float ptrSrc1 = (float)(((const __global TypePixel *)(src1 + iy * srcPitch + ix * sizeof(TypePixel)))[0]);
        const float ptrSrc2 = (float)(((const __global TypePixel *)(src2 + iy * srcPitch + ix * sizeof(TypePixel)))[0]);

        const float d0 = ptrSrc0 * DCT3X3_0_0 + ptrSrc1 * DCT3X3_1_0 + ptrSrc2 * DCT3X3_2_0;
        const float d1 = ptrSrc0 * DCT3X3_0_1                        + ptrSrc2 * DCT3X3_2_1;
        const float d2 = ptrSrc0 * DCT3X3_0_2 + ptrSrc1 * DCT3X3_1_2 + ptrSrc2 * DCT3X3_2_2;

        __global TypePixel *ptrDst0 = (__global TypePixel *)(dst0 + iy * dstPitch + ix * sizeof(TypePixel));
        __global TypePixel *ptrDst1 = (__global TypePixel *)(dst1 + iy * dstPitch + ix * sizeof(TypePixel));
        __global TypePixel *ptrDst2 = (__global TypePixel *)(dst2 + iy * dstPitch + ix * sizeof(TypePixel));
        ptrDst0[0] = d0;
        ptrDst1[0] = d1;
        ptrDst2[0] = d2;
    }
}
