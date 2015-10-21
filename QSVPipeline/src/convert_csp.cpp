//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#include <cstdint>
#include "qsv_tchar.h"
#include <vector>
#include "mfxstructures.h"
#include "qsv_simd.h"
#include "qsv_version.h"
#include "convert_csp.h"

enum : uint32_t {
    _P_ = 0x1,
    _I_ = 0x2,
    ALL = _P_ | _I_,
};

void convert_yuy2_to_nv12_sse2(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);
void convert_yuy2_to_nv12_avx(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);
void convert_yuy2_to_nv12_avx2(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);

void convert_yuy2_to_nv12_i_sse2(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);
void convert_yuy2_to_nv12_i_ssse3(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);
void convert_yuy2_to_nv12_i_avx(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);
void convert_yuy2_to_nv12_i_avx2(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);

void convert_yv12_to_nv12_sse2(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);
void convert_yv12_to_nv12_avx(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);
void convert_yv12_to_nv12_avx2(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);

void convert_uv_yv12_to_nv12_sse2(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);
void convert_uv_yv12_to_nv12_avx(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);
void convert_uv_yv12_to_nv12_avx2(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);

void convert_rgb3_to_rgb4_ssse3(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);
void convert_rgb3_to_rgb4_avx(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);
void convert_rgb3_to_rgb4_avx2(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);

void convert_rgb4_to_rgb4_sse2(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);
void convert_rgb4_to_rgb4_avx(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);
void convert_rgb4_to_rgb4_avx2(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);

void convert_yuv42010_to_p101_avx2(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);
void convert_yuv42010_to_p101_avx(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);
void convert_yuv42010_to_p101_sse2(void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);

static const ConvertCSP funcList[] = {
    { MFX_FOURCC_YUY2, MFX_FOURCC_NV12, false, { convert_yuy2_to_nv12_avx2,     convert_yuy2_to_nv12_i_avx2   }, AVX2|AVX },
    { MFX_FOURCC_YUY2, MFX_FOURCC_NV12, false, { convert_yuy2_to_nv12_avx,      convert_yuy2_to_nv12_i_avx    }, AVX },
    { MFX_FOURCC_YUY2, MFX_FOURCC_NV12, false, { convert_yuy2_to_nv12_sse2,     convert_yuy2_to_nv12_i_ssse3  }, SSSE3|SSE2 },
    { MFX_FOURCC_YUY2, MFX_FOURCC_NV12, false, { convert_yuy2_to_nv12_sse2,     convert_yuy2_to_nv12_i_sse2   }, SSE2 },
#if !QSVENC_AUO
    { MFX_FOURCC_YV12, MFX_FOURCC_NV12, false, { convert_yv12_to_nv12_avx2,     convert_yv12_to_nv12_avx2     }, AVX2|AVX },
    { MFX_FOURCC_YV12, MFX_FOURCC_NV12, false, { convert_yv12_to_nv12_avx,      convert_yv12_to_nv12_avx      }, AVX },
    { MFX_FOURCC_YV12, MFX_FOURCC_NV12, false, { convert_yv12_to_nv12_sse2,     convert_yv12_to_nv12_sse2     }, SSE2 },
    { MFX_FOURCC_YV12, MFX_FOURCC_NV12, true,  { convert_uv_yv12_to_nv12_avx2,  convert_uv_yv12_to_nv12_avx2  }, AVX2|AVX },
    { MFX_FOURCC_YV12, MFX_FOURCC_NV12, true,  { convert_uv_yv12_to_nv12_avx,   convert_uv_yv12_to_nv12_avx   }, AVX },
    { MFX_FOURCC_YV12, MFX_FOURCC_NV12, true,  { convert_uv_yv12_to_nv12_sse2,  convert_uv_yv12_to_nv12_sse2  }, SSE2 },
    { MFX_FOURCC_RGB3, MFX_FOURCC_RGB4, false, { convert_rgb3_to_rgb4_avx2,     convert_rgb3_to_rgb4_avx2     }, AVX2|AVX },
    { MFX_FOURCC_RGB3, MFX_FOURCC_RGB4, false, { convert_rgb3_to_rgb4_avx,      convert_rgb3_to_rgb4_avx      }, AVX },
    { MFX_FOURCC_RGB3, MFX_FOURCC_RGB4, false, { convert_rgb3_to_rgb4_ssse3,    convert_rgb3_to_rgb4_ssse3    }, SSSE3|SSE2 },
    { MFX_FOURCC_RGB4, MFX_FOURCC_RGB4, false, { convert_rgb4_to_rgb4_avx2,     convert_rgb4_to_rgb4_avx2     }, AVX2|AVX },
    { MFX_FOURCC_RGB4, MFX_FOURCC_RGB4, false, { convert_rgb4_to_rgb4_avx,      convert_rgb4_to_rgb4_avx      }, AVX },
    { MFX_FOURCC_RGB4, MFX_FOURCC_RGB4, false, { convert_rgb4_to_rgb4_sse2,     convert_rgb4_to_rgb4_sse2     }, SSE2 },
    { MFX_FOURCC_YV12, MFX_FOURCC_P010, false, { convert_yuv42010_to_p101_avx2, convert_yuv42010_to_p101_avx2 }, AVX2|AVX },
    { MFX_FOURCC_YV12, MFX_FOURCC_P010, false, { convert_yuv42010_to_p101_avx,  convert_yuv42010_to_p101_avx  }, AVX },
    { MFX_FOURCC_YV12, MFX_FOURCC_P010, false, { convert_yuv42010_to_p101_sse2, convert_yuv42010_to_p101_sse2 }, SSE2 },
#endif
    { 0, 0, false, 0x0, 0 },
};

const ConvertCSP* get_convert_csp_func(unsigned int csp_from, unsigned int csp_to, bool uv_only) {
    unsigned int availableSIMD = get_availableSIMD();
    const ConvertCSP *convert = nullptr;
    for (int i = 0; funcList[i].func; i++) {
        if (csp_from != funcList[i].csp_from)
            continue;
        
        if (csp_to != funcList[i].csp_to)
            continue;
        
        if (uv_only != funcList[i].uv_only)
            continue;
        
        if (funcList[i].simd != (availableSIMD & funcList[i].simd))
            continue;

        convert = &funcList[i];
        break;
    }
    return convert;
}

const TCHAR *get_simd_str(unsigned int simd) {
    static std::vector<std::pair<uint32_t, const TCHAR*>> simd_str_list = {
        { AVX2,  _T("AVX2")   },
        { AVX,   _T("AVX")    },
        { SSE42, _T("SSE4.2") },
        { SSE41, _T("SSE4.2") },
        { SSSE3, _T("SSSE3")  },
        { SSE2,  _T("SSE2")   },
    };
    for (auto simd_str : simd_str_list) {
        if (simd_str.first & simd)
            return simd_str.second;
    }
    return _T("-");
}
