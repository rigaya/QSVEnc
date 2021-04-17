// yuv420
// gen_rand_block_loop_y

#include <clRNG/mrg31k3p.clh>

#define clrngHostStream                clrngMrg31k3pHostStream
#define clrngStream                    clrngMrg31k3pStream
#define clrngCopyOverStreamsFromGlobal clrngMrg31k3pCopyOverStreamsFromGlobal
#define clrngCopyOverStreamsToGlobal   clrngMrg31k3pCopyOverStreamsToGlobal
#define clrngRandomInteger             clrngMrg31k3pRandomInteger
#define clrngNextState                 clrngMrg31k3pNextState

uint gen_random_u32(clrngStream *stream) {
    int i = clrngNextState(&stream->current); // 1  -  2^31-1
    int j = clrngNextState(&stream->current); // 1  -  2^31-1
    return (uint)i + (uint)j;
}

//block size 32x8
//
//乱数の各バイトの中身
//1pixelあたり32bit
//pRandY           [ refA0, refB0, ditherV0, 0, refA1, refB1, ditherY1, 0, ... ]
//pRandUV (yuv420) [ refA0, refB0, ditherU0, ditherV0, refA2, refB2, ditherU2, ditherV2, ... ]
//pRandUV (yuv444) [ refA0, refB0, ditherU0, ditherV0, refA1, refB1, ditherU1, ditherV1, ... ]
__kernel void kernel_deband_gen_rand(
    __global char *__restrict__ pRandY,
    __global char *__restrict__ pRandUV,
    int pitchY, int pitchUV, int width, int height,
    __global clrngHostStream *streams) {
    const int gid_i_half = get_group_id(0) * get_local_size(0) /* 32 */ + get_local_id(0);
    int gid_j_half = get_group_id(1) * gen_rand_block_loop_y * get_local_size(1) /* 8 */ + get_local_id(1);
    if ((gid_i_half << 1) < width) {
        const int thread_x_num = get_num_groups(0) * get_local_size(0);
        const int gid = (get_group_id(1) * get_local_size(1) + get_local_id(1)) * thread_x_num + gid_i_half;
        clrngStream state;
        clrngCopyOverStreamsFromGlobal(1, &state, &streams[gid]);

        #pragma unroll
        for (int iyb_loop = 0; iyb_loop < gen_rand_block_loop_y; iyb_loop++, gid_j_half += get_local_size(1)) {
            if ((gid_j_half << 1) < height) {
                uint rand0 = (uint)gen_random_u32(&state);
                const uint refAB0 = rand0 & 0xffff;

                uint rand1 = (uint)gen_random_u32(&state);
                const uint refAB1 = rand1 & 0xffff;

                uint rand2 = (uint)gen_random_u32(&state);
                const uint refAB2 = rand2 & 0xffff;

                uint rand3 = (uint)gen_random_u32(&state);
                const uint refAB3 = rand3 & 0xffff;

                //const uchar dithY0 = (rand0 & 0x00ff0000) >> 16;
                //const uchar dithY1 = (rand1 & 0x00ff0000) >> 16;
                //const uchar dithY2 = (rand2 & 0x00ff0000) >> 16;
                //const uchar dithY3 = (rand3 & 0x00ff0000) >> 16;
                //const uchar dithU0 = (rand0 & 0xff000000) >> 24;
                //const uchar dithV0 = (rand1 & 0xff000000) >> 24;

                //y line0
                //char8 data_y0 = { refA0, refB0, dithY0, 0, refA1, refB1, dithY1, 0 };
                uint2 data_y0 = (uint2)(rand0, rand1);
                //y line1
                //charB data_y1 = { refA2, refB2, dithY2, 0, refA3, refB3, dithY3, 0 };
                uint2 data_y1 = (uint2)(rand2, rand3);

                *(__global uint2 *)(pRandY + (gid_j_half << 1) * pitchY + (gid_i_half << 1) * sizeof(uint) + 0)      = data_y0;
                *(__global uint2 *)(pRandY + (gid_j_half << 1) * pitchY + (gid_i_half << 1) * sizeof(uint) + pitchY) = data_y1;
                if (yuv420) {
                    // { refAB0, dithU0, dithV0 }
                    uint data_c0 = refAB0 | ((rand0 & 0xff000000) >> 8)| (rand1 & 0xff000000);
                    *(__global uint *)(pRandUV + gid_j_half * pitchUV + gid_i_half * sizeof(uint)) = data_c0;
                } else {
                    //const uchar dithU1 = (rand2 & 0xff000000) >> 24;
                    //const uchar dithV1 = (rand3 & 0xff000000) >> 24;
                    uint rand4 = (uint)gen_random_u32(&state);
                    //const uchar dithU2 = (rand4 & 0x000000ff);
                    //const uchar dithV2 = (rand4 & 0x0000ff00) >> 8;
                    //const uchar dithU3 = (rand4 & 0x00ff0000) >> 16;
                    //const uchar dithV3 = (rand4 & 0xff000000) >> 24;
                    //c line0
                    //{ refAB0, dithU0, dithV0, refAB1, dithU1, dithV1 }
                    uint2 data_c0 = (uint2)(refAB0 | ((rand0 & 0xff000000) >> 8) | (rand1 & 0xff000000),
                                            refAB1 | ((rand2 & 0xff000000) >> 8) | (rand3 & 0xff000000));
                    //c line1
                    //{ refAB2, rand4 & 0x0000ffff, refAB3, rand4 & 0xffff0000 };
                    uint2 data_c1 = (uint2)(refAB2 | ((rand4 & 0x0000ffff) << 16), refAB3 | (rand4 & 0xffff0000));
                    *(__global uint2 *)(pRandUV + (gid_j_half << 1) * pitchUV + (gid_i_half << 1) * sizeof(uint) +       0) = data_c0;
                    *(__global uint2 *)(pRandUV + (gid_j_half << 1) * pitchUV + (gid_i_half << 1) * sizeof(uint) + pitchUV) = data_c1;
                }
            }
        }
        clrngCopyOverStreamsToGlobal(1, &streams[gid], &state);
    }
}
