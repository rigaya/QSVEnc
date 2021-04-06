// Type
// bit_depth

#define LOGO_MAX_DP   (1000)
#define LOGO_FADE_MAX (256)

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

typedef int BOOL;

float delogo_yc48(Type pixel, const short2 logo_data, const float logo_depth_mul_fade, const BOOL target_y) {
    const float nv12_2_yc48_mul = (target_y) ? 1197.0f / (1<<(bit_depth-2)) : 4681.0f / (1<<bit_depth);
    const float nv12_2_yc48_sub = (target_y) ? 299.0f : 599332.0f / 256.0f;

    //ロゴ情報取り出し
    float logo_dp = (float)logo_data.x;
    float logo    = (float)logo_data.y;

    logo_dp = (logo_dp * logo_depth_mul_fade) * (1.0f / (float)(128 * LOGO_FADE_MAX));
    //0での除算回避
    if (logo_dp == LOGO_MAX_DP) {
        logo_dp -= 1.0f;
    }

    //nv12->yc48
    float pixel_yc48 = (float)pixel * nv12_2_yc48_mul - nv12_2_yc48_sub;

    //ロゴ除去
    return (pixel_yc48 * (float)LOGO_MAX_DP - logo * logo_dp + ((float)LOGO_MAX_DP - logo_dp) * 0.5f) / ((float)LOGO_MAX_DP - logo_dp);
}

Type delogo(Type pixel, const short2 logo_data, const float logo_depth_mul_fade, const BOOL target_y) {
    //ロゴ除去
    float yc = delogo_yc48(pixel, logo_data, logo_depth_mul_fade, target_y);

    const float yc48_2_nv12_mul = (target_y) ?   219.0f / (1<<(20-bit_depth)) :    14.0f / (1<<(16-bit_depth));
    const float yc48_2_nv12_add = (target_y) ? 65919.0f / (1<<(20-bit_depth)) : 32900.0f / (1<<(16-bit_depth));
    return (Type)clamp((yc * yc48_2_nv12_mul + yc48_2_nv12_add + 0.5f), 0.0f, (float)(1<<bit_depth)-0.1f);
}

Type logo_add(Type pixel, const short2 logo_data, const float logo_depth_mul_fade, const BOOL target_y) {
    const float nv12_2_yc48_mul = (target_y) ? 1197.0f / (1<<(bit_depth-2)) : 4681.0f / (1<<bit_depth);
    const float nv12_2_yc48_sub = (target_y) ? 299.0f : 599332.0f / 256.0f;
    const float yc48_2_nv12_mul = (target_y) ?   219.0f / (1<<(20-bit_depth)) :    14.0f / (1<<(16-bit_depth));
    const float yc48_2_nv12_add = (target_y) ? 65919.0f / (1<<(20-bit_depth)) : 32900.0f / (1<<(16-bit_depth));

    //ロゴ情報取り出し
    float logo_dp = (float)logo_data.x;
    float logo    = (float)logo_data.y;

    logo_dp = (logo_dp * logo_depth_mul_fade) * (1.0f / (float)(128 * LOGO_FADE_MAX));

    //nv12->yc48
    float pixel_yc48 = (float)pixel * nv12_2_yc48_mul - nv12_2_yc48_sub;

    //ロゴ付加
    float yc = (pixel_yc48 * ((float)LOGO_MAX_DP - logo_dp) + logo * logo_dp) * (1.0f / (float)LOGO_MAX_DP);

    return (Type)clamp((yc * yc48_2_nv12_mul + yc48_2_nv12_add + 0.5f), 0.0f, (float)(1<<bit_depth)-0.1f);
}

__kernel void kernel_delogo(
    __global uchar *restrict pFrame, const int frame_pitch, const int width, const int height,
    __global uchar *restrict pLogo, const int logo_pitch, const int logo_x, const int logo_y, const int logo_width, const int logo_height, const float logo_depth_mul_fade,
    const BOOL target_y) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < logo_width && y < logo_height && (x + logo_x) < width && (y + logo_y) < height) {
        //ロゴ情報取り出し
        const short2 logo_data = *(__global short2 *)(&pLogo[y * logo_pitch + x * sizeof(short2)]);

        //画素データ取り出し
        pFrame += (y + logo_y) * frame_pitch + (x + logo_x) * sizeof(Type);
        Type pixel_yuv = *(__global Type *)pFrame;
        Type ret = (target_y) ? delogo(pixel_yuv, logo_data, logo_depth_mul_fade, true)
                              : delogo(pixel_yuv, logo_data, logo_depth_mul_fade, false);

        *(__global Type *)pFrame = ret;
    }
}

__kernel void kernel_logo_add(
    __global uchar *restrict pFrame, const int frame_pitch, const int width, const int height,
    __global uchar *restrict pLogo, const int logo_pitch, const int logo_x, const int logo_y, const int logo_width, const int logo_height, const float logo_depth_mul_fade,
    const BOOL target_y) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < logo_width && y < logo_height && (x + logo_x) < width && (y + logo_y) < height) {
        //ロゴ情報取り出し
        const short2 logo_data = *(__global short2 *)(&pLogo[y * logo_pitch + x * sizeof(short2)]);

        //画素データ取り出し
        pFrame += (y + logo_y) * frame_pitch + (x + logo_x) * sizeof(Type);
        Type pixel_yuv = *(__global Type *)pFrame;
        Type ret = (target_y) ? logo_add(pixel_yuv, logo_data, logo_depth_mul_fade, true)
                              : logo_add(pixel_yuv, logo_data, logo_depth_mul_fade, false);

        *(__global Type *)pFrame = ret;
    }
}
