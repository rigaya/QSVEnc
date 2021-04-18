// Type
// TypePixel
// TypePixel2
// TypePixel4
// bit_depth
// TypeCalc
// USE_FP16
// nnx
// nny
// nns
// thread_y_loop
// weight_loop
// prescreen_new

#if USE_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#include "rgy_filter_nnedi_common.cl"

// prescreen_new では、4ピクセル分一度に処理する
#define  pix_x_per_thread ((prescreen_new) ? 4 : 1)

#if USE_FP16
 bool compute_kernel0_get_flag_original(const half2 ret[thread_y_loop][nns], int ithy) {
    //__hlaf2には重み方向に2つの値が入っている
    //やっていることはfloat版と同じ
    return (max(ret[ithy][1].x, ret[ithy][1].y) <= max(ret[ithy][0].x, ret[ithy][0].y));
}
 void compute_kernel0_get_flags_new(bool flags[4], const half2 ret[thread_y_loop][nns], int ithy) {
    flags[0] = ret[ithy][0].x > (half)(0.0f);
    flags[1] = ret[ithy][0].y > (half)(0.0f);
    flags[2] = ret[ithy][1].x > (half)(0.0f);
    flags[3] = ret[ithy][1].y > (half)(0.0f);
}
#else //#if USE_FP16
 bool compute_kernel0_get_flag_original(const float ret[thread_y_loop][nns], int ithy) {
    return (max(ret[ithy][2], ret[ithy][3]) <= max(ret[ithy][0], ret[ithy][1]));
}
 void compute_kernel0_get_flags_new(bool flags[4], const float ret[thread_y_loop][nns], int ithy) {
    flags[0] = ret[ithy][0] > 0.0f;
    flags[1] = ret[ithy][1] > 0.0f;
    flags[2] = ret[ithy][2] > 0.0f;
    flags[3] = ret[ithy][3] > 0.0f;
}
#endif //#if USE_FP16

TypePixel interp_ret(const __local TypePixel *const ptr_src, const int ssrc_dim,
    const bool flag, const int thIdX, const int thIdY, const int ithy, const int nnx_2_m1, const int nny_2) {
    TypePixel val = prescreen_flag();
    if (flag) {
        float tmp =
            (19.0f / 32.0f) * ((float)ptr_src[SSRC(thIdX + nnx_2_m1, thIdY * thread_y_loop + ithy + 1)] + (float)ptr_src[SSRC(thIdX + nnx_2_m1, thIdY * thread_y_loop + ithy + 2)])
            - (3.0f / 32.0f) * ((float)ptr_src[SSRC(thIdX + nnx_2_m1, thIdY * thread_y_loop + ithy + 0)] + (float)ptr_src[SSRC(thIdX + nnx_2_m1, thIdY * thread_y_loop + ithy + 3)]);
        val = (TypePixel)clamp(tmp + 0.5f, 0.0f, (1<<bit_depth)-1.0f);
    }
    return val;
}



//half2なら2, floatなら1
#define wstep (USE_FP16 ? 2 : 1)

//half2の場合、値を2つ格納できる
#define stmp_dim (((prescreen_new) ? 4 : 8) / wstep)

#if USE_FP16 && prescreen_new
//prescreen=new かつ __half2使用の時は、重複した配置を行わない
#define ssrc_dim ((NNEDI_BLOCK_X * pix_x_per_thread + nnx) / 2)
#else
//floatの時のサイズ　また、__half2でもprescreen=originalの時は重複配置するので、floatと同じサイズ
#define ssrc_dim (NNEDI_BLOCK_X * pix_x_per_thread + nnx)
#endif

__kernel void kernel_compute_network0(
    __global uchar *__restrict__ pDst,
    const int dstOffset, //top field / bottom field の考慮
    const int dstPitch, //1行おきなので通常の2倍の値が入っている
    const int dstWidth,
    const int dstHeight,
    __global uchar *__restrict__ pIn,
    const int inOffset, //top field / bottom field の考慮
    const int inPitch, //1行おきなので通常の2倍の値が入っている
    const int inWidth,
    const int inHeight,
    const __global TypeCalc *__restrict__ weight,
    const int targetField
  ) {
    const int thIdX      = get_local_id(0); //(サイズ: NNEDI_BLOCK_X)
    const int thIdY      = get_local_id(1); //(サイズ: NNEDI_BLOCK_Y)
    const int gIdX       =(get_group_id(0) * NNEDI_BLOCK_X /*blockDim.x*/ + thIdX) * pix_x_per_thread;
    const int gIdY       =(get_group_id(1) * NNEDI_BLOCK_Y /*blockDim.y*/ + thIdY) * thread_y_loop; //フィールド単位

    //sharedメモリのサイズと使途
    __local char shared[
        ssrc_dim * (NNEDI_BLOCK_Y * thread_y_loop + nny) * sizeof(TypeCalc) + //src 計算用
        NNEDI_BLOCK_X * NNEDI_BLOCK_Y * thread_y_loop * stmp_dim * sizeof(TypeCalc) + //tmp (計算結果の一時保管用)
        (NNEDI_BLOCK_X * pix_x_per_thread) * (NNEDI_BLOCK_Y * thread_y_loop + nny) * sizeof(TypePixel) //interp_retで補間に使うため
    ];
    __local TypeCalc *const ptr_src = (__local TypeCalc *)shared;

    __local TypeCalc *const ptr_temp = (__local TypeCalc *)((__local char *)ptr_src
        + (ssrc_dim * (NNEDI_BLOCK_Y * thread_y_loop + nny) * sizeof(ptr_src[0])));
#define STMP_IDX(i,x,y) ( ((y)*(NNEDI_BLOCK_X)+(x)) * stmp_dim + (i))

    //interp_ret()で補間を行う時に使用するデータ
    //16bit精度(int)の場合、fp16では精度が落ちる可能性があるため、ptr_srcとは別に保持することにした
    //interp_ret()では縦方向にしか補間しないので、ptr_srcのようにnnx分余分に読む必要はない
    //ここではsharedメモリ節約のため、floatではなく整数で保持する
    __local TypePixel *const ptr_pix = (__local TypePixel *)((__local char *)ptr_temp
        + NNEDI_BLOCK_X * NNEDI_BLOCK_Y * thread_y_loop * stmp_dim * sizeof(TypeCalc));
    const int spix_dim = NNEDI_BLOCK_X * pix_x_per_thread;

    pDst += dstOffset;
    pIn  += inOffset;

    //input(texture) -> shared, spix
    //textureからpixel情報をsharedメモリにロードする
    //範囲外の折り返し等はtextureでやってくれるのでここでは無視
    const int nnx_2_m1 = nnx / 2 - 1;
    const int nny_2 = nny / 2 - (targetField == NNEDI_GEN_FIELD_BOTTOM ? 1 : 0);
    load_texSrc(pix_x_per_thread, true, ptr_src, ssrc_dim, ptr_pix, spix_dim, pIn, inPitch, inWidth, inHeight, nnx, nny, nnx_2_m1, nny_2, thIdX, thIdY);
    barrier(CLK_LOCAL_MEM_FENCE);

    float dummy[thread_y_loop][4];
    const int sweight_dim = (wstep == 1) ? nnxy : nnxy * weight_loop;
    if (!prescreen_new) {
        #pragma unroll
        for (int iw = 0; iw < nns; iw += weight_loop) {
            TypeCalc sum[thread_y_loop][weight_loop]; //レジスタにのることを期待する
            dot_product0(true, true, sum, ptr_src, ssrc_dim, weight+iw*sweight_dim, /*sweight_dim=*/nnxy, weight+48*nns+iw, nnx, nny, thIdX, thIdY, pix_x_per_thread, dummy);
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                #pragma unroll
                for (int ithw = 0; ithw < weight_loop; ithw++) {
                    ptr_temp[STMP_IDX(iw+ithw, thIdX, thIdY * thread_y_loop + ithy)] = elliott(sum[ithy][ithw]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int iw = 0; iw < nns; iw += weight_loop) {
            TypeCalc sum[thread_y_loop][weight_loop]; //レジスタにのることを期待する
            dot_product0(true, false, sum, ptr_temp, stmp_dim, weight+49*nns+iw*nns, /*sweight_dim=nnxy=*/4, weight+49*nns + 4*nns+iw, /*nnx=*/4, /*nny=*/1, thIdX, thIdY, pix_x_per_thread, dummy);
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                #pragma unroll
                for (int ithw = 0; ithw < weight_loop; ithw++) {
                    //half2なら、値を2つ格納できることに注意して、4/wstepとする
                    ptr_temp[STMP_IDX(4/wstep+iw+ithw, thIdX, thIdY * thread_y_loop + ithy)] = elliott(sum[ithy][ithw]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        TypeCalc ret[thread_y_loop][nns]; //レジスタにのることを期待する
        #pragma unroll
        for (int iw = 0; iw < nns; iw += weight_loop) {
            TypeCalc sum[thread_y_loop][weight_loop]; //レジスタにのることを期待する
            dot_product0(true, false, sum, ptr_temp, stmp_dim, weight + nns*49 + nns*5+stmp_dim*iw, /*sweight_dim=nnxy=*/8, weight + nns*49 + nns*5 + nns*8+iw, /*nnx=*/8, /*nny=*/1, thIdX, thIdY, pix_x_per_thread, dummy);
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                #pragma unroll
                for (int ithw = 0; ithw < weight_loop; ithw++) {
                    ret[ithy][ithw+iw] = sum[ithy][ithw];
                }
            }
        }

        if (gIdX < dstWidth) {
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                if ((gIdY + ithy) * 2 < dstHeight) { //縦方向は1行おきの処理となるので "*2"
                    const bool flag = compute_kernel0_get_flag_original(ret, ithy);
                    __global TypePixel *const ptr_dst = (__global TypePixel *)((__global uchar *)pDst + (gIdY + ithy) * dstPitch + gIdX * sizeof(TypePixel));
                    ptr_dst[0] = interp_ret(ptr_pix, spix_dim, flag, thIdX, thIdY, ithy, 0, nny_2);
                }
            }
        }
    } else {
        #pragma unroll
        for (int iw = 0; iw < nns; iw += weight_loop) {
            TypeCalc sum[thread_y_loop][weight_loop]; //レジスタにのることを期待する
            dot_product0(true, true, sum, ptr_src, ssrc_dim, weight+iw*sweight_dim, /*sweight_dim=*/nnxy, weight+64*nns+iw, nnx, nny, thIdX, thIdY, pix_x_per_thread, dummy);
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                #pragma unroll
                for (int ithw = 0; ithw < weight_loop; ithw++) {
                    ptr_temp[STMP_IDX(iw+ithw, thIdX, thIdY * thread_y_loop + ithy)] = elliott(sum[ithy][ithw]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        TypeCalc ret[thread_y_loop][nns]; //レジスタにのることを期待する
        #pragma unroll
        for (int iw = 0; iw < nns; iw += weight_loop) {
            TypeCalc sum[thread_y_loop][weight_loop]; //レジスタにのることを期待する
            dot_product0(true, false, sum, ptr_temp, stmp_dim, weight+65*nns+iw*nns, /*sweight_dim=nnxy=*/4, weight+65*nns + 4*nns + iw, /*nnx=*/4, /*nny=*/1, thIdX, thIdY, pix_x_per_thread, dummy);
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                #pragma unroll
                for (int ithw = 0; ithw < weight_loop; ithw++) {
                    ret[ithy][ithw+iw] = sum[ithy][ithw];
                }
            }
        }

        if (gIdX < dstWidth) {
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                if ((gIdY + ithy) * 2 < dstHeight) { //縦方向は1行おきの処理となるので "*2"
                    __global TypePixel4 *const ptr_dst = (__global TypePixel4 *)((__global uchar *)pDst + (gIdY + ithy) * dstPitch + gIdX * sizeof(TypePixel));
                    //1スレッドで4pixel分出力する
                    bool flags[4];
                    compute_kernel0_get_flags_new(flags, ret, ithy);
                    TypePixel4 out;
                    out.x = interp_ret(ptr_pix+0, spix_dim, flags[0], thIdX * pix_x_per_thread, thIdY, ithy, 0, nny_2);
                    out.y = interp_ret(ptr_pix+1, spix_dim, flags[1], thIdX * pix_x_per_thread, thIdY, ithy, 0, nny_2);
                    out.z = interp_ret(ptr_pix+2, spix_dim, flags[2], thIdX * pix_x_per_thread, thIdY, ithy, 0, nny_2);
                    out.w = interp_ret(ptr_pix+3, spix_dim, flags[3], thIdX * pix_x_per_thread, thIdY, ithy, 0, nny_2);
                    ptr_dst[0] = out;
                }
            }
        }
    }
}
