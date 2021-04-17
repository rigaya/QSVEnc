#define NNEDI_BLOCK_X      (32)
#define NNEDI_BLOCK_Y      (8)

#define weight0size    (49 * 4 + 5 * 4 + 9 * 4)
#define weight0sizenew (4 * 65 + 4 * 5)

#define NNEDI_GEN_FIELD_TOP    (0)
#define NNEDI_GEN_FIELD_BOTTOM (1)

#define VPP_NNEDI_PRE_SCREEN_NONE            (0x00)
#define VPP_NNEDI_PRE_SCREEN_ORIGINAL        (0x01)
#define VPP_NNEDI_PRE_SCREEN_NEW             (0x02)
#define VPP_NNEDI_PRE_SCREEN_MODE            (0x07)
#define VPP_NNEDI_PRE_SCREEN_BLOCK           (0x10)
#define VPP_NNEDI_PRE_SCREEN_ONLY            (0x20)

#define RGY_FLT_EPS (1e-6f)

#define SSRC(x,y) ((y)*(ssrc_dim)+(x))
#define SPIX(x,y) ((y)*(spix_dim)+(x))
#define SWHT_IDX(i,thIdWeight) ((thIdWeight)*sweight_dim+(i))

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif
#ifndef wrap
#define wrap(x, low, high) (((x) < (low)) ? (((low)<<1)-(x)) : (((x) >= (high)) ? (((high)<<1) - (x)) : (x)))
#endif

float exp_(float val) {
    return native_exp(clamp(val, -80.0f, 80.0f));
}

#if USE_FP16
half2 elliott(half2 val) {
    return val / ((half2)(1.0f) + fabs(val));
}
#else
float elliott(float val) {
    return val *  native_recip(1.0f + fabs(val));
}
#endif

float load_pix(__global uchar *__restrict__ pIn,
    const int inPitch, //1行おきなので通常の2倍の値が入っている
    const int inWidth,
    const int inHeight,
    const int x, const int y) {
    const int ix = wrap(x, 0, inWidth-1);
    const int iy = wrap(y, 0, (inHeight>>1)-1); //高さは半分なので÷2
    TypePixel p = *(__global TypePixel*)(pIn + iy * inPitch + ix * sizeof(TypePixel));
    return (float)p * (1.0f / ((float)(1<<(sizeof(TypePixel)*8))));
}

#if USE_FP16
void load_texSrc(
    const int pix_x_per_thread, const bool load_for_interp,
    __local half2 *const ptr_src, const int ssrc_dim, __local TypePixel *const ptr_pix, const int spix_dim,
    __global uchar *__restrict__ pIn,
    const int inPitch, //1行おきなので通常の2倍の値が入っている
    const int inWidth,
    const int inHeight,
    const int nnx_load, const int nny_load, const int nnx_2_m1, const int nny_2,
    const int thIdX, const int thIdY) {
    if (pix_x_per_thread == 1) {
        //sharedメモリ上に、以下のように重複配置する
        // | 0, 1 | 1, 2 | 2, 3 | 3, 4 | 4, 5 | ...
        for (int y = 0; y + thIdY < NNEDI_BLOCK_Y * thread_y_loop + nny_load; y += NNEDI_BLOCK_Y) {
            for (int x = 0; x + thIdX < ssrc_dim; x += NNEDI_BLOCK_X) {
                const int px = get_group_id(0) * NNEDI_BLOCK_X /*blockDim.x*/ + thIdX + x - nnx_2_m1;
                const int py = get_group_id(1) * NNEDI_BLOCK_Y /*blockDim.y*/ * thread_y_loop + thIdY + y - nny_2;
                const float v0 = load_pix(pIn, inPitch, inWidth, inHeight, px+0, py);
                const float v1 = load_pix(pIn, inPitch, inWidth, inHeight, px+1, py);
                ptr_src[SSRC(x + thIdX, y + thIdY)] = (half2)(v0, v1); //half2のときはここでは256倍せず、0～1の範囲を使用する
                if (load_for_interp && 0 <= thIdX + x - nnx_2_m1 && thIdX + x - nnx_2_m1 < spix_dim) {
                    ptr_pix[SPIX(x + thIdX - nnx_2_m1, y + thIdY)] = (TypePixel)(v0 * (float)(1<<bit_depth) + 0.5f);
                }
            }
        }
    } else { //pix_x_per_thread == 4
        //sharedメモリ上に、以下のように配置する
        // | 0, 1 | 2, 3 | 4, 5 | ...
        for (int y = 0; y + thIdY < NNEDI_BLOCK_Y * thread_y_loop + nny_load; y += NNEDI_BLOCK_Y) {
            for (int x = 0; x + thIdX < ssrc_dim; x += NNEDI_BLOCK_X) {
                const int load_x = (thIdX + x) * 2 - nnx_2_m1;
                const int px = get_group_id(0) * NNEDI_BLOCK_X /*blockDim.x*/ * pix_x_per_thread + load_x;
                const int py = get_group_id(1) * NNEDI_BLOCK_Y /*blockDim.y*/ * thread_y_loop + thIdY + y - nny_2;
                const float v0 = load_pix(pIn, inPitch, inWidth, inHeight, px+0, py);
                const float v1 = load_pix(pIn, inPitch, inWidth, inHeight, px+1, py);
                ptr_src[SSRC(x + thIdX, y + thIdY)] = (half2)(v0, v1); //half2のときはここでは256倍せず、0～1の範囲を使用する
                if (load_for_interp && 0 <= load_x && load_x < spix_dim) {
                    TypePixel2 p;
                    p.x = (TypePixel)(v0 * (float)(1<<bit_depth) + 0.5f);
                    p.y = (TypePixel)(v1 * (float)(1<<bit_depth) + 0.5f);
                    *(__local TypePixel2 *)&ptr_pix[SPIX(load_x, y + thIdY)] = p;
                }
            }
        }
    }
}
#else //#if USE_FP16
void load_texSrc(
    const int pix_x_per_thread, const bool load_for_interp,
    __local float *const ptr_src, const int ssrc_dim, __local TypePixel *const ptr_pix, const int spix_dim, 
    __global uchar *__restrict__ pIn,
    const int inPitch, //1行おきなので通常の2倍の値が入っている
    const int inWidth,
    const int inHeight,
    const int nnx_load, const int nny_load, const int nnx_2_m1, const int nny_2,
    const int thIdX, const int thIdY) {
    for (int y = 0; y + thIdY < NNEDI_BLOCK_Y * thread_y_loop + nny_load; y += NNEDI_BLOCK_Y) {
        for (int x = 0; x + thIdX < ssrc_dim; x += NNEDI_BLOCK_X) {
            const int px = get_group_id(0) * NNEDI_BLOCK_X /*blockDim.x*/ * pix_x_per_thread + thIdX + x - nnx_2_m1;
            const int py = get_group_id(1) * NNEDI_BLOCK_Y /*blockDim.y*/ * thread_y_loop + thIdY + y - nny_2;
            const float value = load_pix(pIn, inPitch, inWidth, inHeight, px, py);
            ptr_src[SSRC(x + thIdX, y + thIdY)] = value * 256.0f; //floatのときはここで256倍して8bit相当に戻す
            if (load_for_interp && 0 <= thIdX + x - nnx_2_m1 && thIdX + x - nnx_2_m1 < spix_dim) {
                ptr_pix[SPIX(x + thIdX - nnx_2_m1, y + thIdY)] = (TypePixel)(value * (float)(1<<bit_depth) + 0.5f);
            }
        }
    }
}
#endif //#if USE_FP16

TypePixel prescreen_flag() {
    return (1<<bit_depth)-1;
}

int kernel_comute_network1_calc_scale_step_float() { return 1; };
int kernel_comute_network1_calc_scale_step_half2() { return 2; };
int kernel_comute_network1_calc_scale_step_TypeCalc() { return (USE_FP16) ? 2 : 1; };

#if USE_FP16
void dot_product0(
    const bool scale_dummy, const bool src_is_frame,
    half2 sum[thread_y_loop][weight_loop],
    const __local half2 *const ptr_src, const int ssrc_dim,
    const __global half2 *const ptr_weight, const int sweight_dim,
    const __global half2 *__restrict__ weight_offset,
    const int nnx_dp0, const int nny_dp0, const int thIdX, const int thIdY,
    const int pix_x_per_thread,
    const float mstd[thread_y_loop][4]
) {
    const int laneid = get_sub_group_local_id();
    #pragma unroll
    for (int ithy = 0; ithy < thread_y_loop; ithy++) {
        #pragma unroll
        for (int i = 0; i < weight_loop; i++) {
            sum[ithy][i] = (half2)(0.0f);
        }
    }
    const int pix_x_per_thread_for_half2 = (prescreen_new) ? 2 : 1;
    const int wstep = kernel_comute_network1_calc_scale_step_half2();
    const half2 *ptr_w = ptr_weight;
    for (int y = 0; y < nny_dp0; y++) {
        const int src_index = (src_is_frame)
            //srcがフレームのキャッシュを指しているとき
            //通常、pix_x_per_thread=1なので、thIdXによって各スレッドが担当するpixelをロードする
            //pre_screen=originalでは、重複配置をしているので、各スレッドは、__hlaf2ごとにロードすればよい
            //   th=0   th=1   th=2   th=3   th=4
            // | 0, 1 | 1, 2 | 2, 3 | 3, 4 | 4, 5 | ...
            //pre_screen=newの時には各スレッドが4pixel担当するので、とびとびの値をロードする。
            //このとき、half2に2pixel分収まり、pre_screen=originalのときのように重複配置はしていないので、
            //pix_x_per_thread_for_half2=2をthIdXに積算する
            //   th=0          th=1          th=2
            // | 0, 1 | 2, 3 | 3, 4 | 5, 6 | 7, 8 |
            ? SSRC(thIdX * pix_x_per_thread_for_half2, thIdY * thread_y_loop + y)
            //kernel_comute_network0で、srcがptr_tmpの計算結果の場合
            //担当pixelはstmp_dim(ssrc_dim)ごとに並んでいるので、x=0、y=担当行でロードする
            : SSRC(0, thIdY * thread_y_loop * NNEDI_BLOCK_X + thIdX);
        const __local half2 *ptr_s = &ptr_src[src_index];

        //src_is_frame = trueのとき
        //pre_screen=originalでは、重複配置をしているので、各スレッドは、2つおきに読む必要がある
        //  最初           次           その次
        //   ↓            ↓            ↓
        // | 0, 1 | 1, 2 | 2, 3 | 3, 4 | 4, 5 | ...
        //
        //pre_screen=newの時には重複配置ではないので、各スレッドはすぐ隣を読めばよい
        //  最初    次    その次
        //   ↓     ↓     ↓
        // | 0, 1 | 2, 3 | 3, 4 | 5, 6 | 7, 8 |
        const int sstep = ((src_is_frame && !prescreen_new) ? wstep : 1);

        for (int x = 0; x < nnx_dp0; x += wstep, ptr_s += sstep) {
            half2 s0[thread_y_loop];
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                s0[ithy] = ptr_s[(src_is_frame) ? (SSRC(0, ithy)) : (SSRC(0, ithy * NNEDI_BLOCK_X))];
            }
#if ENABLE_DP1_SHUFFLE_OPT
            //kernel_comute_network0ではhalf2の場合 nns_k0= 4 / 2
            //なので、weight_loopが2より大きいとおかしなことになる
            static_assert(weight_loop <= 2, "weight_loop <= 2");

            //wに連続するweight_loop*2の値を読み込み、shuffleによりbroadcastする
            //重みは、nns方向にまず並んでいる
            //基本的には下記のような感じでロード
            //   <------  nns_k0 ------>
            //   <-- half2-->
            //     w0    w1    w2   w3
            //  |0----|---->|1----|---->|   <<< x0にかかる重み
            //  |2----|---->|3----|---->|   <<< x1にかかる重み
            //     w4    w5    w6   w7
            half2 w;
            if (laneid < weight_loop*2) w = ptr_w[laneid];
            ptr_w += weight_loop*2;
            #pragma unroll
            for (int i = 0; i < weight_loop; i++) {
                half2 w0 = sub_group_broadcast(w, i+0);           //x0にかかる重み
                half2 w1 = sub_group_broadcast(w, i+weight_loop); //x1にかかる重み
                #pragma unroll
                for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                    //nns方向の計算をhalf2内で同時に行っていくイメージ
                    sum[ithy][i] += (half2)(s0[ithy].x) * w0;  //x0 * (w0, w1)
                    sum[ithy][i] += (half2)(s0[ithy].y) * w1; //x1 * (w4, w5)
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
        const half2 wo = weight_offset[0];
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            //srcがフレームのキャッシュを指しているときは、
            //half2の場合、ロード時に256倍していないので、ここで256倍する
            //kernel_comute_network0で、srcがptr_tmpの計算結果の場合は必要ない
            //なお、ここで256倍しないと、後段のelliottが本来の値を返さない
            const half2 scale = (half2)((src_is_frame) ? 256.0f : 1.0f);
            sum[ithy][i] = sum[ithy][i] * scale + wo;
        }
    }
}
#else //#if USE_FP16
void dot_product0(
    const bool scale_dummy, const bool src_is_frame,
    float sum[thread_y_loop][weight_loop],
    const __local float *const ptr_src, const int ssrc_dim,
    const __global float *const ptr_weight, const int sweight_dim,
    const __global float *__restrict__ weight_offset,
    const int nnx_dp0, const int nny_dp0, const int thIdX, const int thIdY,
    const int pix_x_per_thread,
    const float mstd[thread_y_loop][4]
) {
    #pragma unroll
    for (int ithy = 0; ithy < thread_y_loop; ithy++) {
        #pragma unroll
        for (int i = 0; i < weight_loop; i++) {
            sum[ithy][i] = 0.0f;
        }
    }
    const __global float *ptr_w = ptr_weight;
    for (int y = 0; y < nny_dp0; y++) {
        const int src_index = (src_is_frame)
            //srcがフレームのキャッシュを指しているとき
            //通常、pix_x_per_thread=1なので、thIdXによって各スレッドが担当するpixelをロードする
            //pre_screen=newの時には各スレッドが4pixel担当するので、pix_x_per_threadが4になり、とびとびの値をロードする
            ? SSRC(thIdX * pix_x_per_thread, thIdY * thread_y_loop + y)
            //kernel_comute_network0で、srcがptr_tmpの計算結果の場合
            //担当pixelはstmp_dim(ssrc_dim)ごとに並んでいるので、x=0、y=担当行でロードする
            : SSRC(0, thIdY * thread_y_loop * NNEDI_BLOCK_X + thIdX);
        const __local float *ptr_s = &ptr_src[src_index];

        for (int x = 0; x < nnx_dp0; x++, ptr_s++, ptr_w++) {
            float s0[thread_y_loop];
            #pragma unroll
            for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                s0[ithy] = ptr_s[(src_is_frame) ? (SSRC(0, ithy)) : (SSRC(0, ithy * NNEDI_BLOCK_X))];
            }
            #pragma unroll
            for (int i = 0; i < weight_loop; i++) {
                float w0 = ptr_w[SWHT_IDX(0, i)];
                #pragma unroll
                for (int ithy = 0; ithy < thread_y_loop; ithy++) {
                    sum[ithy][i] += s0[ithy] * w0;
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < weight_loop; i++, weight_offset++) {
        const float wo = weight_offset[0];
        #pragma unroll
        for (int ithy = 0; ithy < thread_y_loop; ithy++) {
            const float scale = (float)((scale_dummy) ? 1.0f : mstd[ithy][2]);
            sum[ithy][i] = sum[ithy][i] * scale + wo;
        }
    }
}

#endif //#if USE_FP16
