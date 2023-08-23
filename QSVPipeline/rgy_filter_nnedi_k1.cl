// TypePixel
// TypePixel2
// TypePixel4
// bit_depth
// TypeCalc
// USE_FP16
// nnx
// nny
// thread_y_loop
// weight_loop
// prescreen_new
// ENABLE_DP1_SHUFFLE_OPT
// COLLECT_FLAG_MODE 0...sub_group_any, 1...cl_khr_local_int32_base_atomics

#if USE_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#include "rgy_filter_nnedi_common.cl"

void kernel_comute_network1_calc_scale_get_sum_sumsq(float *sum, float *sumsq, TypeCalc tsum, TypeCalc tsumsq) {
#if USE_FP16
    //half2では、textureからのロード時に256倍していない
    //ここで、256倍して、本来の値に戻す(ここで256倍しないと、後段のelliottが本来の値を返さない)
    //なお、textureからのロード時に256倍してしまうとtsumsqの計算がオーバーフローしてしまう
    *sum = ((float)tsum.x + (float)tsum.y) * 256.0f;
    *sumsq = ((float)tsumsq.x + (float)tsumsq.y) * 256.0f * 256.0f;
#else
    *sum = tsum, *sumsq = tsumsq;
#endif
}

void kernel_comute_network1_calc_scale(
    float mstd[thread_y_loop][4],
    __local TypeCalc *__restrict__ const ptr_temp,
    const __local TypeCalc *__restrict__ const ptr_src, const int ssrc_dim,
    const int thIdX, const int thIdY) {
    const int step = kernel_comute_network1_calc_scale_step_TypeCalc();
#define TMP_IDX(x,y,i) ((((i)*(nny + NNEDI_BLOCK_Y * thread_y_loop)+(y))*NNEDI_BLOCK_X)+(x))
    for (int y = 0; y + thIdY < nny + NNEDI_BLOCK_Y * thread_y_loop; y += NNEDI_BLOCK_Y) {
        TypeCalc sum = (TypeCalc)(0.0f), sumsq = (TypeCalc)(0.0f);
        //まず各ピクセルごとに、x方向の総和をとる
        #pragma unroll (4)
        for (int x = 0; x < nnx; x += step) {
            const TypeCalc value = ptr_src[SSRC(x + thIdX, y + thIdY)];
            sum += value;
            sumsq += value * value;
        }
        //一度sharedメモリに格納
        ptr_temp[TMP_IDX(thIdX, thIdY+y, 0)] = sum;
        ptr_temp[TMP_IDX(thIdX, thIdY+y, 1)] = sumsq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float inv_nnxy = 1.0f / nnxy;

    //次にy方向の総和をとる
    #pragma unroll
    for (int ithy = 0; ithy < thread_y_loop; ithy++) {
        TypeCalc tsum = (TypeCalc)(0.0f), tsumsq = (TypeCalc)(0.0f);
        #pragma unroll
        for (int y = 0; y < nny; y++) {
            tsum   += ptr_temp[TMP_IDX(thIdX, thIdY*thread_y_loop+ithy+y, 0)];
            tsumsq += ptr_temp[TMP_IDX(thIdX, thIdY*thread_y_loop+ithy+y, 1)];
        }

        //half2使用時に並列で計算したものを集約するとともに、256倍の補正を適用する
        float sum, sumsq;
        kernel_comute_network1_calc_scale_get_sum_sumsq(&sum, &sumsq, tsum, tsumsq);

        mstd[ithy][3] = 0.0f;
        mstd[ithy][0] = sum * inv_nnxy;
        float tmp = sumsq * inv_nnxy - mstd[ithy][0] * mstd[ithy][0];
        if (tmp <= RGY_FLT_EPS) {
            mstd[ithy][1] = 0.0f;
            mstd[ithy][2] = 0.0f;
        } else {
            mstd[ithy][1] = native_sqrt(tmp);
            mstd[ithy][2] = native_recip(mstd[ithy][1]);
        }
    }
#undef TMP_IDX
}

#if USE_FP16
void dot_product_frame1_fp16(
    half2 sum[thread_y_loop][weight_loop],
    __local half2 *__restrict__ const ptr_src, const int ssrc_dim,
    const __global half2 *__restrict__ const ptr_weight, const int sweight_dim,
    const __global half2 *__restrict__ weight_offset,
    const int thIdX, const int thIdY,
    const half2 weight_scale[thread_y_loop]
) {
#if ENABLE_DP1_SHUFFLE_OPT
    const int laneid = get_sub_group_local_id();
#endif
    #pragma unroll
    for (int ithy = 0; ithy < thread_y_loop; ithy++) {
        #pragma unroll
        for (int i = 0; i < weight_loop; i++) {
            sum[ithy][i] = (half2)(0.0f);
        }
    }
    const __global half2 *ptr_w = ptr_weight;
    for (int y = 0; y < nny; y++) {
        const __local half2 *ptr_s = &ptr_src[SSRC(thIdX, thIdY * thread_y_loop + y)];

        //ptr_srcでは、重複配置をしているので、各スレッドは、2つおきに読む必要がある
        //  最初           次           その次
        //   ↓            ↓            ↓
        // | 0, 1 | 1, 2 | 2, 3 | 3, 4 | 4, 5 | ...
        for (int x = 0; x < nnx; x += 2, ptr_s += 2) {
            //このsharedメモリからロードしたpixelデータをレジスタ上で使いまわすのが重要
            half2 s0[thread_y_loop];
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                s0[ithy] = ptr_s[SSRC(0, ithy)];
            }
#if ENABLE_DP1_SHUFFLE_OPT
            //[nns/weight_loop][nnxy][weight_loop][2]
            //最後の2つには、nns方向の[i]と[i+nns]のものを配置しているので、これがセットでhalf2に乗る
            //   <---------------   nns  -------------------->
            //   <---  weight_loop  --->  (weight_loop = 2の場合)
            //   <-- half2-->
            //    [0]  [nns]  [1] [1+nns]
            //  |0----|---->|1----|---->|   <<< x0にかかる重み
            //  |2----|---->|3----|---->|   <<< x1にかかる重み
            //まず、各スレッドでweight_loop*2分だけ重みをwにロードし、
            //これをshuffleで全スレッドにbroadcastして使用するようにする
            half2 w;
            if (laneid < weight_loop*2) { w = ptr_w[laneid]; };
            ptr_w += weight_loop*2;
            #pragma unroll
            for (int i = 0; i < weight_loop; i++) {
                const half2 w0 = sub_group_broadcast(w,            +i); //x0にかかる重み
                const half2 w1 = sub_group_broadcast(w, weight_loop+i); //x1にかかる重み
                #pragma unroll
                for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                    sum[ithy][i] += (half2)(s0[ithy].x) * w0;  //x0 * w([i], [i+nns])
                    sum[ithy][i] += (half2)(s0[ithy].y) * w1;  //x1 * w([i], [i+nns])
                }
            }
#else
            #pragma unroll
            for (int i = 0; i < weight_loop; i++, ptr_w++) {
                const half2 w0 = ptr_w[0];
                const half2 w1 = ptr_w[weight_loop];
                #pragma unroll
                for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                    //nns方向の計算をhalf2内で同時に行っていくイメージ
                    sum[ithy][i] += (half2)(s0[ithy].x) * w0;  //x0 * (w0, w1)
                    sum[ithy][i] += (half2)(s0[ithy].y) * w1;  //x1 * (w4, w5)
                }
            }
            ptr_w += weight_loop;
#endif
        }
    }
    #pragma unroll
    for (int i = 0; i < weight_loop; i++, weight_offset++) {
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            //weight offsetもw([i], [i+nns])の並びになっている
            sum[ithy][i] = sum[ithy][i] * weight_scale[ithy] + weight_offset[0];
        }
    }
}

void kernel_comute_network1_dot_product(
    float wsum[thread_y_loop],
    float vsum[thread_y_loop],
    half2 *const ptr_src, const int ssrc_dim,
    const half2 *const weight,
    float mstd[thread_y_loop][4],
    const int thIdX, const int thIdY) {
    //未実装
}

void kernel_comute_network1_dot_product_opt(
    float wsum[thread_y_loop],
    float vsum[thread_y_loop],
    __local half2 *const ptr_src, const int ssrc_dim,
    const __global half2 *const weight,
    float mstd[thread_y_loop][4],
    const int thIdX, const int thIdY) {
    //[iw]と[iw+nns]の重みが隣り合って_half2に入るので、half2としてはnnxyのまま
    const int sweight_dim = nnxy;
    for (int iw = 0; iw < nns; iw += weight_loop) {
        half2 sum[thread_y_loop][weight_loop]; //レジスタにのることを期待する
        // 重み(nns)方向に、weight_loop分のdotproduct
        //ひとつの__half2に[iw, iw+nns]の両方の内積の結果が入っている
        // sum0[i](iw, iw+nns)
        half2 weight_scale[thread_y_loop];
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            weight_scale[ithy] = (half2)(mstd[ithy][2]);
        }
        dot_product_frame1_fp16(
            sum, ptr_src, ssrc_dim, weight+iw*sweight_dim, sweight_dim, weight + nns*nnxy + iw, thIdX, thIdY, weight_scale);
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            #pragma unroll
            for (int ithw = 0; ithw < weight_loop; ithw++) {
                //half2使用時には、オーバーフローを避けるため、textureからのロード時に256倍していないので、ここでfloatにしてから補正する
                float ret0 = exp_((float)(sum[ithy][ithw].x) * 256.0f);
                float ret1 = (float)(sum[ithy][ithw].y) * 256.0f;
                wsum[ithy] += ret0;
                vsum[ithy] += ret0 * (ret1 * native_recip(1.0f + fabs(ret1)));
            }
        }
    }
}
#else //#if USE_FP16
void dot_product_frame1_fp32(
    float sum0[thread_y_loop][weight_loop], //レジスタにのることを期待する
    float sum1[thread_y_loop][weight_loop], //レジスタにのることを期待する
    __local TypeCalc *__restrict__ const ptr_src, const int ssrc_dim,
    const __global TypeCalc *__restrict__ const ptr_weight, const int sweight_dim,
    const __global TypeCalc *__restrict__ weight_offset,
    const int thIdX, const int thIdY,
    const float mstd[thread_y_loop][4]
) {
#if ENABLE_DP1_SHUFFLE_OPT
    const int laneid = get_sub_group_local_id();
#endif //#if ENABLE_DP1_SHUFFLE_OPT
    #pragma unroll
    for (int ithy = 0; ithy < thread_y_loop; ithy++) {
        #pragma unroll
        for (int i = 0; i < weight_loop; i++) {
            sum0[ithy][i] = sum1[ithy][i] = 0.0f;
        }
    }
    const __global TypeCalc *ptr_w = ptr_weight;
    for (int y = 0; y < nny; y++) {
        const __local TypeCalc *ptr_s = &ptr_src[SSRC(thIdX, thIdY * thread_y_loop + y)];
#if ENABLE_DP1_WEIGHT_ARRAY_OPT
        //#pragma unroll (4)
        for (int x = 0; x < nnx; x++, ptr_s++) {
            //このsharedメモリからロードしたpixelデータをレジスタ上で使いまわすのが重要
            TypeCalc s0[thread_y_loop];
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                s0[ithy] = ptr_s[SSRC(0, ithy)];
            }
#if ENABLE_DP1_SHUFFLE_OPT
            //[nns/weight_loop][nnxy][weight_loop][2]
            //最後の2つには、nns方向の[i]と[i+nns]のものを配置している
            //   <---------------   nns  -------------------->
            //   <---  weight_loop  --->  (weight_loop = 2の場合)
            //    [0]  [nns]  [1] [1+nns]
            //  |0----|1--->|2----|3--->|
            //まず、各スレッドでweight_loop*2分だけ重みをwにロードし、
            //これをshuffleで全スレッドにbroadcastして使用するようにする
            TypeCalc w;
            if (laneid < weight_loop*2) { w = ptr_w[laneid]; };
            ptr_w += weight_loop*2;
            #pragma unroll
            for (int i = 0; i < weight_loop; i++) {
                const TypeCalc w0 = sub_group_broadcast(w, i*2+0); //[i]の重み
                const TypeCalc w1 = sub_group_broadcast(w, i*2+1); //[i+nns]の重み
                #pragma unroll
                for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                    sum0[ithy][i] += s0[ithy] * w0;
                    sum1[ithy][i] += s0[ithy] * w1;
                }
            }
#else
            #pragma unroll
            for (int i = 0; i < weight_loop; i++, ptr_w += 2) {
                const TypeCalc w0 = ptr_w[0];
                const TypeCalc w1 = ptr_w[1];
                #pragma unroll
                for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                    sum0[ithy][i] += s0[ithy] * w0;
                    sum1[ithy][i] += s0[ithy] * w1;
                }
            }
#endif
        }
    }
#else
    #pragma unroll (4)
    for (int x = 0; x < nnx; x++, ptr_w++, ptr_s++) {
        //このsharedメモリからロードしたpixelデータをレジスタ上で使いまわすのが重要
        TypePixel s0[thread_y_loop];
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            s0[ithy] = ptr_s[SSRC(0, ithy*NNEDI_BLOCK_Y)];
        }
        #pragma unroll
        for (int i = 0; i < weight_loop; i++) {
            TypeCalc w0 = ptr_w[SWHT_IDX(0, i)];
            TypeCalc w1 = ptr_w[SWHT_IDX(0, i+nns)];
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                sum0[i][ithy] += s0[ithy] * w0;
                sum1[i][ithy] += s0[ithy] * w1;
            }
        }
    }
#endif
#if ENABLE_DP1_WEIGHT_ARRAY_OPT
    #pragma unroll
    for (int i = 0; i < weight_loop; i++, weight_offset += 2) {
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            //weight offsetもw([i], [i+nns])の並びになっている
            sum0[ithy][i] = sum0[ithy][i] * mstd[ithy][2] + weight_offset[0]; //w[i]用のweight_offset
            sum1[ithy][i] = sum1[ithy][i] * mstd[ithy][2] + weight_offset[1]; //w[i+nns]用のweight_offset
        }
    }
#else
    #pragma unroll
    for (int i = 0; i < weight_loop; i++, weight_offset++) {
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            sum0[ithy][i] = sum0[ithy][i] * mstd[ithy][2] + weight_offset[0];
            sum1[ithy][i] = sum1[ithy][i] * mstd[ithy][2] + weight_offset[nns];
        }
    }
#endif
}

void kernel_comute_network1_dot_product(
    float wsum[thread_y_loop],
    float vsum[thread_y_loop],
    __local float *const ptr_src, const int ssrc_dim,
    const __global float *const weight,
    float mstd[thread_y_loop][4],
    const int thIdX, const int thIdY) {
    const int sweight_dim = (ENABLE_DP1_WEIGHT_ARRAY_OPT) ? 2 * nnxy : nnxy;
    for (int iw = 0; iw < nns; iw += weight_loop) {
        float sum0[thread_y_loop][weight_loop]; //レジスタにのることを期待する
        dot_product0(false, true, sum0, ptr_src, ssrc_dim, weight+ (iw)*nnxy, sweight_dim, weight + (nns*2)*nnxy + iw, nnx, nny, thIdX, thIdY, 1, mstd);

        float sum1[thread_y_loop][weight_loop]; //レジスタにのることを期待する
        dot_product0(false, true, sum1, ptr_src, ssrc_dim, weight+ (nns+iw)*nnxy, sweight_dim, weight + (nns*2)*nnxy+nns + iw, nnx, nny, thIdX, thIdY, 1, mstd);

        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            #pragma unroll
            for (int ithw = 0; ithw < weight_loop; ithw++) {
                float ret0 = exp_(sum0[ithy][ithw]);
                float ret1 = sum1[ithy][ithw];
                wsum[ithy] += ret0;
                vsum[ithy] += ret0 * (ret1 * native_recip(1.0f + fabs(ret1)));
            }
        }
    }
}

void kernel_comute_network1_dot_product_opt(
    float wsum[thread_y_loop],
    float vsum[thread_y_loop],
    __local float *const ptr_src, const int ssrc_dim,
    const __global float *const weight,
    float mstd[thread_y_loop][4],
    const int thIdX, const int thIdY) {
    //ENABLE_DP1_WEIGHT_ARRAY_OPTが有効の場合、
    //[iw]と[iw+nns]の重みが隣り合って並んでいるので、sweight_dimは2倍
    const int sweight_dim = (ENABLE_DP1_WEIGHT_ARRAY_OPT) ? 2 * nnxy : nnxy;
    for (int iw = 0; iw < nns; iw += weight_loop) {
        float sum0[thread_y_loop][weight_loop]; //レジスタにのることを期待する
        float sum1[thread_y_loop][weight_loop]; //レジスタにのることを期待する
        // 重み(nns)方向に、weight_loop分のdotproduct
        // sum0[i] <- iw     - iw+weight_loop
        // sum1[i] <- iw+nns - iw+weight_loop+nns
        dot_product_frame1_fp32(
            sum0, sum1, ptr_src, ssrc_dim, weight+iw*sweight_dim, sweight_dim, weight + (nns*2)*nnxy + iw*2, thIdX, thIdY, mstd);
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            #pragma unroll
            for (int ithw = 0; ithw < weight_loop; ithw++) {
                float ret0 = exp_(sum0[ithy][ithw]); // iw     - iw+weight_loop     の計算結果
                float ret1 = sum1[ithy][ithw];       // iw+nns - iw+weight_loop+nns の計算結果
                wsum[ithy] += ret0;
                vsum[ithy] += ret0 * (ret1 * native_recip(1.0f + fabs(ret1)));
            }
        }
    }
}
#endif //#if USE_FP16

__kernel void kernel_compute_network1(
    __global uchar *__restrict__ pDst, //top field / bottom field は考慮済みとする
    const int dstOffset,
    const int dstPitch, //1行おきなので通常の2倍の値が入っている
    const int dstWidth,
    const int dstHeight,
    __global uchar *__restrict__ pIn,
    const int inOffset, //top field / bottom field の考慮
    const int inPitch, //1行おきなので通常の2倍の値が入っている
    const int inWidth,
    const int inHeight,
    const __global TypeCalc *__restrict__ weight10,
    const __global TypeCalc *__restrict__ weight11,
    const int quals,
    const int targetField,
    const int prescreen
) {
    const int thIdX      = get_local_id(0); //(サイズ: NNEDI_BLOCK_X)
    const int thIdY      = get_local_id(1); //(サイズ: NNEDI_BLOCK_Y)
    const int gIdX       = get_group_id(0) * NNEDI_BLOCK_X /*blockDim.x*/ + thIdX;
    const int gIdY       =(get_group_id(1) * NNEDI_BLOCK_Y /*blockDim.y*/ + thIdY) * thread_y_loop; //フィールド単位

    //sharedメモリのサイズと使途
    //1.src: (NNEDI_BLOCK_X + nnx) * (NNEDI_BLOCK_Y * thread_y_loop + nny) * sizeof(ptr_src[0])
    //2.tmp: (nny + NNEDI_BLOCK_Y * thread_y_loop) * NNEDI_BLOCK_X * 2 * sizeof(ptr_temp[0])
    __local TypeCalc shared_src[(NNEDI_BLOCK_X + nnx) * (NNEDI_BLOCK_Y * thread_y_loop + nny)];////src 計算用
    __local TypeCalc shared_tmp[(nny + NNEDI_BLOCK_Y * thread_y_loop) * NNEDI_BLOCK_X * 2]; //tmp (計算結果の一時保管用)
#if COLLECT_FLAG_MODE == 1
    __local uint flag_collect[NNEDI_BLOCK_Y];
    if (thIdX == 0) {
        flag_collect[thIdY] = 0; // 初期化
    }
#endif
    const int ssrc_dim = NNEDI_BLOCK_X + nnx;

    pDst += dstOffset;
    pIn  += inOffset;

    //input(texture) -> shared
    //textureからpixel情報をsharedメモリにロードする
    //範囲外の折り返し等はtextureでやってくれるのでここでは無視
    const int nnx_2_m1 = nnx / 2 - 1;
    const int nny_2 = nny / 2 - (targetField == NNEDI_GEN_FIELD_BOTTOM ? 1 : 0);
    load_texSrc(1, false, shared_src, ssrc_dim, NULL, 0, pIn, inPitch, inWidth, inHeight, nnx, nny, nnx_2_m1, nny_2, thIdX, thIdY);
    barrier(CLK_LOCAL_MEM_FENCE);

    float mstd[thread_y_loop][4];
    kernel_comute_network1_calc_scale(mstd, shared_tmp, shared_src, ssrc_dim, thIdX, thIdY);

    __global uchar *const ptr_dst_base = (__global uchar *)pDst + gIdY * dstPitch + gIdX * sizeof(TypePixel);
    uint flag_sum = 0xffffffff; //処理するかどうかのフラグ
    if (((uint)prescreen & (uint)VPP_NNEDI_PRE_SCREEN_MODE) != 0) { //prescreenをやっていれば確認する
        flag_sum = 0x00;
        __global uchar *ptr_dst = ptr_dst_base;
        //自分のスレッドの担当するpixelについて調査する
        //処理対象となっていたらビットを立てる
        //thread_y_loopについて、下のビットから使っていく
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++, ptr_dst += dstPitch) {
            uint flag = 0x00;
            if ((gIdY + ithy) * 2 < dstHeight) { //縦方向は1行おきの処理となるので "*2"
                flag = (((__global TypePixel *)ptr_dst)[0] == prescreen_flag()) ? 0x01 << ithy : 0x00;
            }
            flag_sum |= flag;
            //ビットを使い切らないようにチェック
            //static_assert(thread_y_loop <= sizeof(flag_sum) * 8, "thread_y_loop <= sizeof(flag_sum) * 8");
        }
    }

#if 0
                                      |<-------- nns*2 --------->|
                                    WEIGHT_LOOP
                                      |<-->| ---> 繰り返し処理
                                 ---  |--------------------------|
                                      |                          |
                                      |                          |
                                      |                          |
                             nnxy     |                          |
                                      |                          |
                                      |                          |
                                      |                          |
                                 ---  |--------------------------|

                |<----   nnxy  --->|
            --- |------------------|  |----|
NNEDI_BLOCK_X   |                  |  |    | <-- 各スレッドはこの出力の1pixel分(縦方向)をそれぞれ担当
*NNEDI_BLOCK_Y  |                  |  |    |      横: WEIGHT_LOOP
            --- |                  |  |----|      縦: NNEDI_BLOCK_X * NNEDI_BLOCK_Y
                |                  |
                |                  |
        pixels  |                  |
           |    |                  |
           |    |                  |
        　↓    |                  |

#endif
    //weightの先頭のポインタ
#if COLLECT_FLAG_MODE == 0
    if (sub_group_any(flag_sum)) { //どのpixelも処理する必要がなければ、スキップする : cl_khr_subgroups
#elif COLLECT_FLAG_MODE == 1
    atom_or(&flag_collect[thIdY], flag_sum); // cl_khr_local_int32_extended_atomics
    barrier(CLK_LOCAL_MEM_FENCE);
    if (flag_collect[thIdY]) { //どのpixelも処理する必要がなければ、スキップする
#endif //#if COLLECT_FLAG_MODE == 0 || 1

        for (int iquality = 0; iquality < quals; iquality++) {
            const __global TypeCalc *const weight = (iquality) ? weight11 : weight10;
            float wsum[thread_y_loop], vsum[thread_y_loop];
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                wsum[ithy] = vsum[ithy] = 0.0f;
            }
            if (ENABLE_DP1_WEIGHT_LOOP_UNROLL) {
                kernel_comute_network1_dot_product_opt(
                    wsum, vsum, shared_src, ssrc_dim, weight, mstd, thIdX, thIdY);
            } else {
                kernel_comute_network1_dot_product(
                    wsum, vsum, shared_src, ssrc_dim, weight, mstd, thIdX, thIdY);
            }

            const float min_weight_sum = 1e-10f;
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                if (wsum[ithy] > min_weight_sum) {
                    mstd[ithy][3] += ((5.0f * vsum[ithy]) * native_recip(wsum[ithy])) * mstd[ithy][1];
                }
                mstd[ithy][3] += mstd[ithy][0];
            }
        }

        if (gIdX < dstWidth) {
            const float scale = (1<<bit_depth) / 256.0f * ((quals > 1) ? 0.5f : 1.0f);
            __global uchar *ptr_dst = (__global uchar *)ptr_dst_base;
            for (int ithy = 0; ithy < thread_y_loop; ithy++, ptr_dst += dstPitch) {
                if ((((uint)prescreen & (uint)VPP_NNEDI_PRE_SCREEN_BLOCK) || (flag_sum & (1<<ithy))) //処理対象かチェック、ブロックモードなら常に処理する
                    && (gIdY + ithy) * 2 < dstHeight) { //縦方向は1行おきの処理となるので "*2"
                    ((__global TypePixel *)ptr_dst)[0] = (TypePixel)clamp(mstd[ithy][3] * scale + 0.5f, 0.0f, (1<<bit_depth)-1.0f);
                }
            }
        }
#if COLLECT_FLAG_MODE == 0 || COLLECT_FLAG_MODE == 1 
    }
#endif
}