// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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
// --------------------------------------------------------------------------------------------

#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <fstream>
#include <iostream>
#include <filesystem>
#include "qsv_util.h"
#include "qsv_query.h"
#include "rgy_simd.h"
#include "rgy_osdep.h"
#include "rgy_ini.h"
#include "rgy_log.h"
#include "plugin_subburn.h"
#include "subburn_process.h"
#if ENABLE_AVSW_READER && ENABLE_LIBASS_SUBBURN
#if defined(_MSC_VER)
#pragma comment(lib, "libass-9.lib")

#pragma warning(disable : 4100)
#endif

//日本語環境の一般的なコードページ一覧
enum : uint32_t {
    CODE_PAGE_SJIS        = 932, //Shift-JIS
    CODE_PAGE_JIS         = 50220,
    CODE_PAGE_EUC_JP      = 51932,
    CODE_PAGE_UTF8        = 65001,
    CODE_PAGE_UTF16_LE    = 1200, //WindowsのUnicode WCHAR のコードページ
    CODE_PAGE_UTF16_BE    = 1201,
    CODE_PAGE_US_ASCII    = 20127,
    CODE_PAGE_WEST_EUROPE = 1252,  //厄介な西ヨーロッパ言語
    CODE_PAGE_UNSET       = 0xffffffff,
};

//BOM文字リスト
static const int MAX_UTF8_CHAR_LENGTH = 6;
static const uint8_t UTF8_BOM[]     = { 0xEF, 0xBB, 0xBF };
static const uint8_t UTF16_LE_BOM[] = { 0xFF, 0xFE };
static const uint8_t UTF16_BE_BOM[] = { 0xFE, 0xFF };

//ボム文字かどうか、コードページの判定
static uint32_t check_bom(const void* chr) {
    if (chr == nullptr) return CODE_PAGE_UNSET;
    if (memcmp(chr, UTF16_LE_BOM, sizeof(UTF16_LE_BOM)) == 0) return CODE_PAGE_UTF16_LE;
    if (memcmp(chr, UTF16_BE_BOM, sizeof(UTF16_BE_BOM)) == 0) return CODE_PAGE_UTF16_BE;
    if (memcmp(chr, UTF8_BOM,     sizeof(UTF8_BOM))     == 0) return CODE_PAGE_UTF8;
    return CODE_PAGE_UNSET;
}

static BOOL isJis(const void *str, uint32_t size_in_byte) {
    static const uint8_t ESCAPE[][7] = {
        //先頭に比較すべきバイト数
        { 3, 0x1B, 0x28, 0x42, 0x00, 0x00, 0x00 },
        { 3, 0x1B, 0x28, 0x4A, 0x00, 0x00, 0x00 },
        { 3, 0x1B, 0x28, 0x49, 0x00, 0x00, 0x00 },
        { 3, 0x1B, 0x24, 0x40, 0x00, 0x00, 0x00 },
        { 3, 0x1B, 0x24, 0x42, 0x00, 0x00, 0x00 },
        { 6, 0x1B, 0x26, 0x40, 0x1B, 0x24, 0x42 },
        { 4, 0x1B, 0x24, 0x28, 0x44, 0x00, 0x00 },
        { 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 } //終了
    };
    const uint8_t * const str_fin = (const uint8_t *)str + size_in_byte;
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin; chr++) {
        if (*chr > 0x7F)
            return FALSE;
        for (int i = 0; ESCAPE[i][0]; i++) {
            if (str_fin - chr > ESCAPE[i][0] &&
                memcmp(chr, &ESCAPE[i][1], ESCAPE[i][0]) == 0)
                return TRUE;
        }
    }
    return FALSE;
}

static uint32_t isUTF16(const void *str, uint32_t size_in_byte) {
    const uint8_t * const str_fin = (const uint8_t *)str + size_in_byte;
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin; chr++) {
        if (chr[0] == 0x00 && str_fin - chr > 1 && chr[1] <= 0x7F)
            return ((chr - (const uint8_t *)str) % 2 == 1) ? CODE_PAGE_UTF16_LE : CODE_PAGE_UTF16_BE;
    }
    return CODE_PAGE_UNSET;
}

static BOOL isASCII(const void *str, uint32_t size_in_byte) {
    const uint8_t * const str_fin = (const uint8_t *)str + size_in_byte;
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin; chr++) {
        if (*chr == 0x1B || *chr >= 0x80)
            return FALSE;
    }
    return TRUE;
}

static uint32_t jpn_check(const void *str, uint32_t size_in_byte) {
    int score_sjis = 0;
    int score_euc = 0;
    int score_utf8 = 0;
    const uint8_t * const str_fin = (const uint8_t *)str + size_in_byte;
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin - 1; chr++) {
        if ((0x81 <= chr[0] && chr[0] <= 0x9F) ||
            (0xE0 <= chr[0] && chr[0] <= 0xFC) ||
            (0x40 <= chr[1] && chr[1] <= 0x7E) ||
            (0x80 <= chr[1] && chr[1] <= 0xFC)) {
            score_sjis += 2; chr++;
        }
    }
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin - 1; chr++) {
        if ((0xC0 <= chr[0] && chr[0] <= 0xDF) &&
            (0x80 <= chr[1] && chr[1] <= 0xBF)) {
            score_utf8 += 2; chr++;
        } else if (
            str_fin - chr > 2 &&
            (0xE0 <= chr[0] && chr[0] <= 0xEF) &&
            (0x80 <= chr[1] && chr[1] <= 0xBF) &&
            (0x80 <= chr[2] && chr[2] <= 0xBF)) {
            score_utf8 += 3; chr++;
        }
    }
    for (const uint8_t *chr = (const uint8_t *)str; chr < str_fin - 1; chr++) {
        if (((0xA1 <= chr[0] && chr[0] <= 0xFE) && (0xA1 <= chr[1] && chr[1] <= 0xFE)) ||
            (chr[0] == 0x8E                     && (0xA1 <= chr[1] && chr[1] <= 0xDF))) {
            score_euc += 2; chr++;
        } else if (
            str_fin - chr > 2 &&
            chr[0] == 0x8F &&
            (0xA1 <= chr[1] && chr[1] <= 0xFE) &&
            (0xA1 <= chr[2] && chr[2] <= 0xFE)) {
            score_euc += 3; chr += 2;
        }
    }
    if (score_sjis > score_euc && score_sjis > score_utf8)
        return CODE_PAGE_SJIS;
    if (score_utf8 > score_euc && score_utf8 > score_sjis)
        return CODE_PAGE_UTF8;
    if (score_euc > score_sjis && score_euc > score_utf8)
        return CODE_PAGE_EUC_JP;
    return CODE_PAGE_UNSET;
}

static uint32_t get_code_page(const void *str, uint32_t size_in_byte) {
    uint32_t ret = CODE_PAGE_UNSET;
    if ((ret = check_bom(str)) != CODE_PAGE_UNSET)
        return ret;

    if (isJis(str, size_in_byte))
        return CODE_PAGE_JIS;

    if ((ret = isUTF16(str, size_in_byte)) != CODE_PAGE_UNSET)
        return ret;

    if (isASCII(str, size_in_byte))
        return CODE_PAGE_US_ASCII;

    return jpn_check(str, size_in_byte);
}


static bool check_libass_dll() {
#if defined(_WIN32) || defined(_WIN64)
    HMODULE hDll = LoadLibrary(_T("libass.dll"));
    if (hDll == NULL) {
        return false;
    }
    FreeLibrary(hDll);
    return true;
#else
    return true;
#endif //#if defined(_WIN32) || defined(_WIN64)
}

//MSGL_FATAL 0 - RGY_LOG_ERROR  2
//MSGL_ERR   1 - RGY_LOG_ERROR  2
//MSGL_WARN  2 - RGY_LOG_WARN   1
//           3 - RGY_LOG_WARN   1
//MSGL_INFO  4 - RGY_LOG_MORE  -1 (いろいろ情報が出すぎるので)
//           5 - RGY_LOG_MORE  -1
//MSGL_V     6 - RGY_LOG_DEBUG -2
//MSGL_DBG2  7 - RGY_LOG_TRACE -3
static inline int log_level_ass2qsv(int level) {
    static const int log_level_map[] = {
        RGY_LOG_ERROR,
        RGY_LOG_ERROR,
        RGY_LOG_WARN,
        RGY_LOG_WARN,
        RGY_LOG_MORE,
        RGY_LOG_MORE,
        RGY_LOG_DEBUG,
        RGY_LOG_TRACE
    };
    return log_level_map[clamp(level, 0, _countof(log_level_map) - 1)];
}

static void ass_log(int ass_level, const char *fmt, va_list args, void *ctx) {
    ((RGYLog *)ctx)->write_line(log_level_ass2qsv(ass_level), fmt, args, CP_UTF8);
}

static void ass_log_error_only(int ass_level, const char *fmt, va_list args, void *ctx) {
    auto qsv_level = log_level_ass2qsv(ass_level);
    if (qsv_level >= RGY_LOG_ERROR) {
        ((RGYLog *)ctx)->write_line(qsv_level, fmt, args, CP_UTF8);
    }
}

// SubBurn class implementation
SubBurn::SubBurn() :
    m_nCpuGen(getCPUGenCpuid()),
    m_nSimdAvail(get_availableSIMD()),
    m_SubBurnParam(),
    m_vProcessData() {
    m_pluginName = _T("subburn");
}

SubBurn::~SubBurn() {
    PluginClose();
    Close();
}

mfxStatus SubBurn::Submit(const mfxHDL *in, mfxU32 in_num, const mfxHDL *out, mfxU32 out_num, mfxThreadTask *task) {
    if (in == nullptr || out == nullptr || *in == nullptr || *out == nullptr || task == nullptr) {
        return MFX_ERR_NULL_PTR;
    }
    if (in_num != 1 || out_num != 1) {
        return MFX_ERR_UNSUPPORTED;
    }
    if (!m_bInited) return MFX_ERR_NOT_INITIALIZED;

    mfxFrameSurface1 *surface_in = (mfxFrameSurface1 *)in[0];
    mfxFrameSurface1 *surface_out = (mfxFrameSurface1 *)out[0];
    mfxFrameSurface1 *real_surface_in = surface_in;
    mfxFrameSurface1 *real_surface_out = surface_out;

    mfxStatus sts = MFX_ERR_NONE;

    if (m_bIsInOpaque) {
        sts = m_mfxCore.GetRealSurface(surface_in, &real_surface_in);
        if (sts < MFX_ERR_NONE) return sts;
    }

    if (m_bIsOutOpaque) {
        sts = m_mfxCore.GetRealSurface(surface_out, &real_surface_out);
        if (sts < MFX_ERR_NONE) return sts;
    }

    // check validity of parameters
    sts = CheckInOutFrameInfo(&real_surface_in->Info, &real_surface_out->Info);
    if (sts < MFX_ERR_NONE) return sts;

    uint32_t ind = FindFreeTaskIdx();

    if (ind >= m_sTasks.size()) {
        return MFX_WRN_DEVICE_BUSY; // currently there are no free tasks available
    }

    m_mfxCore.IncreaseReference(&(real_surface_in->Data));
    m_mfxCore.IncreaseReference(&(real_surface_out->Data));

    m_sTasks[ind].In = real_surface_in;
    m_sTasks[ind].Out = real_surface_out;
    m_sTasks[ind].bBusy = true;

    if (m_sTasks[ind].pProcessor.get() == nullptr) {
        const bool d3dSurface = !!(m_SubBurnParam.memType & D3D9_MEMORY);
#if defined(_MSC_VER) || defined(__AVX2__)
        if ((m_nSimdAvail & AVX2) == AVX2) {
            m_sTasks[ind].pProcessor.reset((d3dSurface) ? static_cast<Processor *>(new ProcessorSubBurnD3DAVX2) : static_cast<Processor *>(new ProcessorSubBurnAVX2));
        } else
#endif //#if defined(_MSC_VER) || defined(__AVX2__)
#if defined(_MSC_VER) || defined(__AVX__)
        if (m_nSimdAvail & AVX) {
            m_sTasks[ind].pProcessor.reset((d3dSurface) ? static_cast<Processor *>(new ProcessorSubBurnD3DAVX) : static_cast<Processor *>(new ProcessorSubBurnAVX));
        } else
#endif //#if defined(_MSC_VER) || defined(__AVX__)
        if (m_nSimdAvail & SSE41) {
            if (m_nCpuGen == CPU_GEN_AIRMONT || m_nCpuGen == CPU_GEN_SILVERMONT) {
                m_sTasks[ind].pProcessor.reset((d3dSurface) ? static_cast<Processor *>(new ProcessorSubBurnD3DSSE41PshufbSlow) : static_cast<Processor *>(new ProcessorSubBurnSSE41PshufbSlow));
            } else {
                m_sTasks[ind].pProcessor.reset((d3dSurface) ? static_cast<Processor *>(new ProcessorSubBurnD3DSSE41) : static_cast<Processor *>(new ProcessorSubBurnSSE41));
            }
        } else {
            m_sTasks[ind].pProcessor.reset((d3dSurface) ? nullptr : static_cast<Processor *>(new ProcessorSubBurn));
        }

        if (!m_sTasks[ind].pProcessor) {
            AddMessage(RGY_LOG_ERROR, _T("Unsupported.\n"));
            return MFX_ERR_UNSUPPORTED;
        }
        m_sTasks[ind].pProcessor->SetLog(m_pPrintMes);

        //GPUによるD3DSurfaceのコピーを正常に実行するためには、m_pAllocは、
        //PluginのmfxCoreから取得したAllocatorではなく、
        //メインパイプラインから直接受け取ったAllocatorでなければならない
        m_sTasks[ind].pProcessor->SetAllocator((m_SubBurnParam.pAllocator) ? m_SubBurnParam.pAllocator : &m_mfxCore.FrameAllocator());
    }
    m_sTasks[ind].pProcessor->Init(real_surface_in, real_surface_out, &m_vProcessData[ind]);

    *task = (mfxThreadTask)&m_sTasks[ind];

    return MFX_ERR_NONE;
}

mfxStatus SubBurn::Init(mfxVideoParam *mfxParam) {
    if (mfxParam == nullptr) {
        return MFX_ERR_NULL_PTR;
    }
    if (!check_libass_dll()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load libass dll.\n"));
        return MFX_ERR_NULL_PTR;
    }
    mfxStatus sts = MFX_ERR_NONE;
    m_VideoParam = *mfxParam;

    // map opaque surfaces array in case of opaque surfaces
    m_bIsInOpaque = (m_VideoParam.IOPattern & MFX_IOPATTERN_IN_OPAQUE_MEMORY) ? true : false;
    m_bIsOutOpaque = (m_VideoParam.IOPattern & MFX_IOPATTERN_OUT_OPAQUE_MEMORY) ? true : false;
    mfxExtOpaqueSurfaceAlloc *pluginOpaqueAlloc = NULL;

    if (m_bIsInOpaque || m_bIsOutOpaque) {
        pluginOpaqueAlloc = (mfxExtOpaqueSurfaceAlloc*)GetExtBuffer(m_VideoParam.ExtParam,
            m_VideoParam.NumExtParam, MFX_EXTBUFF_OPAQUE_SURFACE_ALLOCATION);
        if (sts != MFX_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed GetExtBuffer for OpaqueAlloc.\n"));
            return sts;
        }
    }

    // check existence of corresponding allocs
    if ((m_bIsInOpaque && !pluginOpaqueAlloc->In.Surfaces) || (m_bIsOutOpaque && !pluginOpaqueAlloc->Out.Surfaces))
        return MFX_ERR_INVALID_VIDEO_PARAM;

    if (m_bIsInOpaque) {
        sts = m_mfxCore.MapOpaqueSurface(pluginOpaqueAlloc->In.NumSurface,
            pluginOpaqueAlloc->In.Type, pluginOpaqueAlloc->In.Surfaces);
        if (sts != MFX_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed MapOpaqueSurface[In].\n"));
            return sts;
        }
    }

    if (m_bIsOutOpaque) {
        sts = m_mfxCore.MapOpaqueSurface(pluginOpaqueAlloc->Out.NumSurface,
            pluginOpaqueAlloc->Out.Type, pluginOpaqueAlloc->Out.Surfaces);
        if (sts != MFX_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed MapOpaqueSurface[Out].\n"));
            return sts;
        }
    }

    m_sTasks.resize((std::max)(1, (int)m_VideoParam.AsyncDepth));
    m_sChunks.resize(m_PluginParam.MaxThreadNum);

    // divide frame into data chunks
    const uint32_t num_lines_in_chunk = mfxParam->vpp.In.CropH / m_PluginParam.MaxThreadNum; // integer division
    const uint32_t remainder_lines = mfxParam->vpp.In.CropH % m_PluginParam.MaxThreadNum; // get remainder
    // remaining lines are distributed among first chunks (+ extra 1 line each)
    for (uint32_t i = 0; i < m_PluginParam.MaxThreadNum; i++) {
        m_sChunks[i].StartLine = (i == 0) ? 0 : m_sChunks[i-1].EndLine + 1;
        m_sChunks[i].EndLine = (i < remainder_lines) ? (i + 1) * num_lines_in_chunk : (i + 1) * num_lines_in_chunk - 1;
    }

    for (uint32_t i = 0; i < m_sTasks.size(); i++) {
        m_sTasks[i].pBuffer.reset((uint8_t *)_aligned_malloc(((std::max)(mfxParam->mfx.FrameInfo.Width, mfxParam->mfx.FrameInfo.CropW) + 255 + 16) & ~255, 32));
        if (m_sTasks[i].pBuffer.get() == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate buffer.\n"));
            return MFX_ERR_NULL_PTR;
        }
    }

    AddMessage(RGY_LOG_DEBUG, _T("init success.\n"));
    m_bInited = true;
    return MFX_ERR_NONE;
}

mfxStatus SubBurn::InitLibAss(ProcessDataSubBurn *pProcData) {
    //libassの初期化
    if (nullptr == (pProcData->pAssLibrary = ass_library_init())) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialize libass.\n"));
        return MFX_ERR_NULL_PTR;
    }
    ass_set_message_cb(pProcData->pAssLibrary, (pProcData->nTaskId == 0) ? ass_log : ass_log_error_only, m_pPrintMes.get());

    ass_set_extract_fonts(pProcData->pAssLibrary, 1);
    ass_set_style_overrides(pProcData->pAssLibrary, nullptr);

    if (nullptr == (pProcData->pAssRenderer = ass_renderer_init(pProcData->pAssLibrary))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialize libass renderer.\n"));
        return MFX_ERR_NULL_PTR;
    }

    ass_set_use_margins(pProcData->pAssRenderer, 0);
    ass_set_hinting(pProcData->pAssRenderer, ASS_HINTING_LIGHT);
    ass_set_font_scale(pProcData->pAssRenderer, 1.0);
    ass_set_line_spacing(pProcData->pAssRenderer, 1.0);
    ass_set_shaper(pProcData->pAssRenderer, pProcData->nAssShaping);

    const char *font = nullptr;
    const char *family = "Arial";
    ass_set_fonts(pProcData->pAssRenderer, font, family, 1, nullptr, 1);

    if (nullptr == (pProcData->pAssTrack = ass_new_track(pProcData->pAssLibrary))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialize libass track.\n"));
        return MFX_ERR_NULL_PTR;
    }

    const int width = pProcData->frameInfo.CropW - pProcData->sCrop.e.left - pProcData->sCrop.e.right;
    const int height = pProcData->frameInfo.CropH - pProcData->sCrop.e.up - pProcData->sCrop.e.bottom;
    ass_set_frame_size(pProcData->pAssRenderer, width, height);

    const AVRational sar = { pProcData->frameInfo.AspectRatioW, pProcData->frameInfo.AspectRatioH };
    double par = 1.0;
    if (sar.num * sar.den > 0) {
        par = (double)sar.num / sar.den;
    }
    ass_set_aspect_ratio(pProcData->pAssRenderer, 1, par);

    if (pProcData->pOutCodecDecodeCtx && pProcData->pOutCodecDecodeCtx->subtitle_header && pProcData->pOutCodecDecodeCtx->subtitle_header_size > 0) {
        ass_process_codec_private(pProcData->pAssTrack, (char *)pProcData->pOutCodecDecodeCtx->subtitle_header, pProcData->pOutCodecDecodeCtx->subtitle_header_size);
    }
    return MFX_ERR_NONE;
}

void SubBurn::SetExtraData(AVCodecContext *codecCtx, const uint8_t *data, uint32_t size) {
    if (data == nullptr || size == 0)
        return;
    if (codecCtx->extradata)
        av_free(codecCtx->extradata);
    codecCtx->extradata_size = size;
    codecCtx->extradata      = (uint8_t *)av_malloc(codecCtx->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
    memcpy(codecCtx->extradata, data, size);
};

tstring SubBurn::errorMesForCodec(const TCHAR *mes, AVCodecID targetCodec) {
    return mes + tstring(_T(" for ")) + char_to_tstring(avcodec_get_name(targetCodec)) + tstring(_T(".\n"));
};

mfxStatus SubBurn::InitAvcodec(ProcessDataSubBurn *pProcData) {
    AVCodecID inputCodecId = AV_CODEC_ID_NONE;
    if (pProcData->pFilePath) {
        //ファイル読み込みの場合
        AddMessage(RGY_LOG_DEBUG, _T("trying to open subtitle file \"%s\""), pProcData->pFilePath);

        std::string filename_char;
        if (0 == tchar_to_string(pProcData->pFilePath, filename_char, CP_UTF8)) {
            AddMessage(RGY_LOG_ERROR, _T("failed to convert filename to utf-8 characters.\n"));
            return MFX_ERR_INVALID_HANDLE;
        }
        int ret = avformat_open_input(&pProcData->pFormatCtx, filename_char.c_str(), nullptr, nullptr);
        if (ret < 0) {
            AddMessage(RGY_LOG_ERROR, _T("error opening file: \"%s\": %s\n"), char_to_tstring(filename_char, CP_UTF8).c_str(), qsv_av_err2str(ret).c_str());
            return MFX_ERR_NULL_PTR; // Couldn't open file
        }

        if (avformat_find_stream_info(pProcData->pFormatCtx, nullptr) < 0) {
            AddMessage(RGY_LOG_ERROR, _T("error finding stream information.\n"));
            return MFX_ERR_NULL_PTR; // Couldn't find stream information
        }
        AddMessage(RGY_LOG_DEBUG, _T("got stream information.\n"));
        av_dump_format(pProcData->pFormatCtx, 0, filename_char.c_str(), 0);

        if (0 > (pProcData->nSubtitleStreamIndex = av_find_best_stream(pProcData->pFormatCtx, AVMEDIA_TYPE_SUBTITLE, -1, -1, nullptr, 0))) {
            AddMessage(RGY_LOG_ERROR, _T("no subtitle stream found in \"%s\".\n"), char_to_tstring(filename_char, CP_UTF8).c_str());
            return MFX_ERR_NULL_PTR; // Couldn't open file
        }
        inputCodecId = pProcData->pFormatCtx->streams[pProcData->nSubtitleStreamIndex]->codecpar->codec_id;
        AddMessage(RGY_LOG_DEBUG, _T("found subtitle in stream #%d (%s).\n"), pProcData->nSubtitleStreamIndex, char_to_tstring(avcodec_get_name(inputCodecId)).c_str());
    } else {
        inputCodecId = pProcData->pStreamIn->codecpar->codec_id;
    }

    pProcData->nType = avcodec_descriptor_get(inputCodecId)->props;
    AddMessage(RGY_LOG_DEBUG, _T("sub type: %s\n"), (pProcData->nType & AV_CODEC_PROP_TEXT_SUB) ? _T("text") : _T("bitmap"));

    auto copy_subtitle_header = [](AVCodecContext *pDstCtx, const AVCodecContext *pSrcCtx) {
        if (pSrcCtx->subtitle_header_size) {
            pDstCtx->subtitle_header_size = pSrcCtx->subtitle_header_size;
            pDstCtx->subtitle_header = (uint8_t *)av_mallocz(pDstCtx->subtitle_header_size + AV_INPUT_BUFFER_PADDING_SIZE);
            memcpy(pDstCtx->subtitle_header, pSrcCtx->subtitle_header, pSrcCtx->subtitle_header_size);
        }
    };
    //decoderの初期化
    if (NULL == (pProcData->pOutCodecDecode = avcodec_find_decoder(inputCodecId))) {
        AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to find decoder"), inputCodecId));
        AddMessage(RGY_LOG_ERROR, _T("Please use --check-decoders to check available decoder.\n"));
        return MFX_ERR_NULL_PTR;
    }
    if (NULL == (pProcData->pOutCodecDecodeCtx = avcodec_alloc_context3(pProcData->pOutCodecDecode))) {
        AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to get decode codec context"), inputCodecId));
        return MFX_ERR_NULL_PTR;
    }
    if (pProcData->pStreamIn) {
        //設定されていない必須情報があれば設定する
#define COPY_IF_ZERO(dst, src) { if ((dst)==0) (dst)=(src); }
        COPY_IF_ZERO(pProcData->pOutCodecDecodeCtx->width, pProcData->pStreamIn->codecpar->width);
        COPY_IF_ZERO(pProcData->pOutCodecDecodeCtx->height, pProcData->pStreamIn->codecpar->height);
#undef COPY_IF_ZERO
        pProcData->pOutCodecDecodeCtx->pkt_timebase = pProcData->pStreamIn->time_base;
        SetExtraData(pProcData->pOutCodecDecodeCtx, pProcData->pStreamIn->codecpar->extradata, pProcData->pStreamIn->codecpar->extradata_size);
    } else {
        pProcData->pOutCodecDecodeCtx->pkt_timebase = pProcData->pFormatCtx->streams[pProcData->nSubtitleStreamIndex]->time_base;
        auto codecpar = pProcData->pFormatCtx->streams[pProcData->nSubtitleStreamIndex]->codecpar;
        SetExtraData(pProcData->pOutCodecDecodeCtx, codecpar->extradata, codecpar->extradata_size);
    }

    int ret;
    AVDictionary *pCodecOpts = nullptr;
    if (pProcData->nType & AV_CODEC_PROP_TEXT_SUB) {
        if (pProcData->pFilePath) {
            if (pProcData->sCharEnc.length() == 0) {
                FILE *fp = NULL;
                if (_tfopen_s(&fp, pProcData->pFilePath, _T("rb")) || fp == NULL) {
                    AddMessage(RGY_LOG_ERROR, _T("error opening file: \"%s\"\n"), pProcData->pFilePath);
                    return MFX_ERR_NULL_PTR; // Couldn't open file
                }

                std::vector<char> buffer(256 * 1024, 0);
                const auto readBytes = fread(buffer.data(), 1, sizeof(buffer[0]) * buffer.size(), fp);
                fclose(fp);

                const auto estCodePage = get_code_page(buffer.data(), (int)readBytes);
                std::map<uint32_t, std::string> codePageMap ={
                    { CODE_PAGE_SJIS,     "CP932"       },
                    { CODE_PAGE_JIS,      "ISO-2022-JP" },
                    { CODE_PAGE_EUC_JP,   "EUC-JP"      },
                    { CODE_PAGE_UTF8,     "UTF-8"       },
                    { CODE_PAGE_UTF16_LE, "UTF-16LE"    },
                    { CODE_PAGE_UTF16_BE, "UTF-16BE"    },
                    { CODE_PAGE_US_ASCII, "ASCII"       },
                    { CODE_PAGE_UNSET,    ""            },
                };
                if (codePageMap.find(estCodePage) != codePageMap.end()) {
                    pProcData->sCharEnc = codePageMap[estCodePage];
                }
            }
        }
        if (pProcData->sCharEnc.length() > 0) {
            if (0 > (ret = av_dict_set(&pCodecOpts, "sub_charenc", pProcData->sCharEnc.c_str(), 0))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to set \"sub_charenc\" option for subtitle decoder: %s\n"), qsv_av_err2str(ret).c_str());
                return MFX_ERR_NULL_PTR;
            }
        }
        if (0 > (ret = av_dict_set(&pCodecOpts, "sub_text_format", "ass", 0))) {
            AddMessage(RGY_LOG_ERROR, _T("failed to set \"sub_text_format\" option for subtitle decoder: %s\n"), qsv_av_err2str(ret).c_str());
            return MFX_ERR_NULL_PTR;
        }
    }
    if (0 > (ret = avcodec_open2(pProcData->pOutCodecDecodeCtx, pProcData->pOutCodecDecode, &pCodecOpts))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open decoder for %s: %s\n"),
            char_to_tstring(avcodec_get_name(pProcData->pStreamIn->codecpar->codec_id)).c_str(), qsv_av_err2str(ret).c_str());
        return MFX_ERR_NULL_PTR;
    }
    if (pProcData->nInTrackId == 0) {
        AddMessage(RGY_LOG_DEBUG, _T("Subtitle Decoder opened\n"));
        AddMessage(RGY_LOG_DEBUG, _T("Subtitle Decode Info: %s, %dx%d\n"), char_to_tstring(avcodec_get_name(inputCodecId)).c_str(),
            pProcData->pOutCodecDecodeCtx->width, pProcData->pOutCodecDecodeCtx->height);
    }
#if 0
    //エンコーダを探す
    const AVCodecID codecId = AV_CODEC_ID_ASS;
    if (NULL == (pProcData->pOutCodecEncode = avcodec_find_encoder(codecId))) {
        AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to find encoder"), codecId));
        AddMessage(RGY_LOG_ERROR, _T("Please use --check-encoders to find available encoder.\n"));
        return MFX_ERR_NULL_PTR;
    }
    AddMessage(RGY_LOG_DEBUG, _T("found encoder for codec %s for subtitle track %d\n"), char_to_tstring(pProcData->pOutCodecEncode->name).c_str(), pProcData->nInTrackId);

    if (NULL == (pProcData->pOutCodecEncodeCtx = avcodec_alloc_context3(pProcData->pOutCodecEncode))) {
        AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to get encode codec context"), codecId));
        return MFX_ERR_NULL_PTR;
    }
    pProcData->pOutCodecEncodeCtx->time_base = av_make_q(1, 1000);
    copy_subtitle_header(pProcData->pOutCodecEncodeCtx, pProcData->pCodecCtxIn);

    AddMessage(RGY_LOG_DEBUG, _T("Subtitle Encoder Param: %s, %dx%d\n"), char_to_tstring(pProcData->pOutCodecEncode->name).c_str(),
        pProcData->pOutCodecEncodeCtx->width, pProcData->pOutCodecEncodeCtx->height);
    if (pProcData->pOutCodecEncode->capabilities & CODEC_CAP_EXPERIMENTAL) {
        //問答無用で使うのだ
        av_opt_set(pProcData->pOutCodecEncodeCtx, "strict", "experimental", 0);
    }
    if (0 > (ret = avcodec_open2(pProcData->pOutCodecEncodeCtx, pProcData->pOutCodecEncode, NULL))) {
        AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to open encoder"), codecId));
        AddMessage(RGY_LOG_ERROR, _T("%s\n"), qsv_av_err2str(ret).c_str());
        return MFX_ERR_NULL_PTR;
    }
    AddMessage(RGY_LOG_DEBUG, _T("Opened Subtitle Encoder Param: %s\n"), char_to_tstring(pProcData->pOutCodecEncode->name).c_str());
    if (nullptr == (pProcData->pBuf = (uint8_t *)av_malloc(1024 * 1024))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate buffer memory for subtitle encoding.\n"));
        return MFX_ERR_NULL_PTR;
    }
#endif
    return MFX_ERR_NONE;
}

mfxStatus SubBurn::ProcSub(ProcessDataSubBurn *pProcData) {
    AVPacket pkt;
    av_init_packet(&pkt);
    while (av_read_frame(pProcData->pFormatCtx, &pkt) >= 0) {
        if (pkt.stream_index == pProcData->nSubtitleStreamIndex) {
            if (pProcData->nType & AV_CODEC_PROP_TEXT_SUB) {
                int got_sub = 0;
                AVSubtitle sub ={ 0 };
                if (0 > avcodec_decode_subtitle2(pProcData->pOutCodecDecodeCtx, &sub, &got_sub, &pkt)) {
                    AddMessage(RGY_LOG_ERROR, _T("Failed to decode subtitle.\n"));
                    return MFX_ERR_UNKNOWN;
                }
                if (got_sub) {
                    const int64_t nStartTime = av_rescale_q(sub.pts, av_make_q(1, AV_TIME_BASE), av_make_q(1, 1000));
                    const int64_t nDuration  = sub.end_display_time;
                    for (uint32_t i = 0; i < sub.num_rects; i++) {
                        auto *ass = sub.rects[i]->ass;
                        if (!ass) {
                            break;
                        }
                        ass_process_chunk(pProcData->pAssTrack, ass, (int)strlen(ass), nStartTime, nDuration);
                    }
                }
                avsubtitle_free(&sub);
            } else {
                pProcData->qSubPackets.push(pkt);
                continue;
            }
        }
        av_packet_unref(&pkt);
    }
    return MFX_ERR_NONE;
}

mfxStatus SubBurn::SetAuxParams(void *auxParam, int auxParamSize) {
    SubBurnParam *pSubBurnPar = (SubBurnParam *)auxParam;
    if (pSubBurnPar == nullptr) {
        return MFX_ERR_NULL_PTR;
    }

    // check validity of parameters
    mfxStatus sts = CheckParam(&m_VideoParam);
    if (sts < MFX_ERR_NONE) return sts;

    std::map<int, ASS_ShapingLevel> mShapingLevel = {
        { QSV_VPP_SUB_SIMPLE,  ASS_SHAPING_SIMPLE  },
        { QSV_VPP_SUB_COMPLEX, ASS_SHAPING_COMPLEX },
    };
    if (mShapingLevel.find(pSubBurnPar->nShaping) == mShapingLevel.end()) {
        AddMessage(RGY_LOG_ERROR, _T("unknown shaping mode for sub burning.\n"));
        return MFX_ERR_INVALID_VIDEO_PARAM;
    }

    if ((m_nSimdAvail & AVX2) == AVX2) {
        m_pluginName = _T("sub[avx2]");
    } else if (m_nSimdAvail & AVX) {
        m_pluginName = _T("sub[avx]");
    } else if (m_nSimdAvail & SSE41) {
        if (m_nCpuGen == CPU_GEN_AIRMONT || m_nCpuGen == CPU_GEN_SILVERMONT) {
            m_pluginName = _T("sub[sse4.1(pshufb slow)]");
        } else {
            m_pluginName = _T("sub[sse4.1]");
        }
    } else {
        m_pluginName = _T("sub[c]");
        return MFX_ERR_UNSUPPORTED;
    }
    m_pluginName += tstring(_T(" ")) + get_chr_from_value(list_vpp_sub_shaping, pSubBurnPar->nShaping);
    if (m_SubBurnParam.pCharEnc) {
        m_pluginName += tstring(_T(" ")) + tstring(m_SubBurnParam.pCharEnc);
    }

    memcpy(&m_SubBurnParam, pSubBurnPar, sizeof(m_SubBurnParam));
    if (m_SubBurnParam.src.trackId != 0) {
        m_pluginName += strsprintf(_T(" track #%d"), trackID(m_SubBurnParam.src.trackId));
    } else {
        AddMessage(RGY_LOG_DEBUG, _T("input file path \"%s\".\n"), m_SubBurnParam.pFilePath);
        std::wstring sFilename = std::filesystem::path(m_SubBurnParam.pFilePath).filename().wstring();
        if (sFilename.length() > 23) {
            sFilename = sFilename.substr(0, 20) + L"...";
        }
        m_pluginName += tstring(_T(" : ")) + wstring_to_tstring(sFilename);
    }

    m_vProcessData = std::vector<ProcessDataSubBurn>(m_sTasks.size());
    for (uint32_t i = 0; i < m_sTasks.size(); i++) {
        m_vProcessData[i].nTaskId = i;
        m_vProcessData[i].memType = m_SubBurnParam.memType;
        m_vProcessData[i].pFilePath = m_SubBurnParam.pFilePath;
        m_vProcessData[i].sCharEnc = tchar_to_string(m_SubBurnParam.pCharEnc);
        m_vProcessData[i].sCrop = m_SubBurnParam.sCrop;
        m_vProcessData[i].nAssShaping = mShapingLevel[m_SubBurnParam.nShaping];
        m_vProcessData[i].frameInfo = m_SubBurnParam.frameInfo;
        m_vProcessData[i].nInTrackId = m_SubBurnParam.src.trackId;
        m_vProcessData[i].nStreamIndexIn = (m_SubBurnParam.src.stream) ? m_SubBurnParam.src.stream->index : -1;
        m_vProcessData[i].nVideoInputFirstKeyPts = m_SubBurnParam.nVideoInputFirstKeyPts;
        m_vProcessData[i].pAssLibrary = nullptr;
        m_vProcessData[i].pAssRenderer = nullptr;
        m_vProcessData[i].pAssTrack = nullptr;
        m_vProcessData[i].pBuf = nullptr;
        m_vProcessData[i].pStreamIn = m_SubBurnParam.src.stream;
        m_vProcessData[i].pVideoInputStream = m_SubBurnParam.pVideoInputStream;
        m_vProcessData[i].nSimdAvail = m_nSimdAvail;
        m_vProcessData[i].qSubPackets.init();

        AddMessage(RGY_LOG_DEBUG, _T("initializing task %d/%d...\n"), i, (uint32_t)m_sTasks.size());

        if (MFX_ERR_NONE != (sts = InitAvcodec(&m_vProcessData[i]))) {
            return sts;
        }
        if (m_vProcessData[i].nType & AV_CODEC_PROP_TEXT_SUB) {
            //テキスト型の字幕ならlibassが必要
            if (MFX_ERR_NONE != (sts = InitLibAss(&m_vProcessData[i]))) {
                return sts;
            }
        }
        if (m_vProcessData[i].pFormatCtx) {
            //ファイルから読み込んでいる場合、初期化段階ですべて読み込んでしまう
            if (MFX_ERR_NONE != (sts = ProcSub(&m_vProcessData[i]))) {
                return sts;
            }
        }
    }
    return MFX_ERR_NONE;
}

mfxStatus SubBurn::SendData(int nType, void *pData) {
    if (nType == PLUGIN_SEND_DATA_AVPACKET) {
        for (uint32_t i = 1; i < m_sTasks.size(); i++) {
            if (m_vProcessData[i].nType & AV_CODEC_PROP_TEXT_SUB) {
                if (m_vProcessData[i].pAssTrack == nullptr) {
                    AddMessage(RGY_LOG_ERROR, _T("ass track not initialized.\n"));
                    return MFX_ERR_NULL_PTR;
                }
            }
            if (m_vProcessData[i].pOutCodecDecodeCtx == nullptr) {
                AddMessage(RGY_LOG_ERROR, _T("sub decoder not initialized.\n"));
                return MFX_ERR_NULL_PTR;
            }
            AVPacket *pktCopy = av_packet_clone((AVPacket *)pData);
            m_vProcessData[i].qSubPackets.push(*pktCopy);
        }
        m_vProcessData[0].qSubPackets.push(*(AVPacket *)pData);
        AddMessage(RGY_LOG_TRACE, _T("Add subtitle packet\n"));
        return MFX_ERR_NONE;
    } else {
        AddMessage(RGY_LOG_ERROR, _T("SendData: unknown data type.\n"));
        return MFX_ERR_UNSUPPORTED;
    }
}

mfxStatus SubBurn::Close() {

    if (!m_bInited)
        return MFX_ERR_NONE;

    mfxStatus sts = MFX_ERR_NONE;


    for (uint32_t i = 0; i < m_sTasks.size(); i++) {
        //close decoder
        if (m_vProcessData[i].pOutCodecDecodeCtx) {
            avcodec_close(m_vProcessData[i].pOutCodecDecodeCtx);
            av_free(m_vProcessData[i].pOutCodecDecodeCtx);
            AddMessage(RGY_LOG_DEBUG, _T("Closed pOutCodecDecodeCtx.\n"));
        }

        //close encoder
        if (m_vProcessData[i].pOutCodecEncodeCtx) {
            avcodec_close(m_vProcessData[i].pOutCodecEncodeCtx);
            av_free(m_vProcessData[i].pOutCodecEncodeCtx);
            AddMessage(RGY_LOG_DEBUG, _T("Closed pOutCodecEncodeCtx.\n"));
        }

        if (m_vProcessData[i].pBuf) {
            av_free(m_vProcessData[i].pBuf);
            m_vProcessData[i].pBuf = nullptr;
        }

        if (m_vProcessData[i].subtitle.num_rects) {
            avsubtitle_free(&m_vProcessData[i].subtitle);
            memset(&m_vProcessData[i].subtitle, 0, sizeof(m_vProcessData[i].subtitle));
        }

        //close format
        if (m_vProcessData[i].pFormatCtx) {
            avformat_close_input(&m_vProcessData[i].pFormatCtx);
            m_vProcessData[i].pFormatCtx = nullptr;
        }

        //libass関連の開放
        if (m_vProcessData[i].pAssTrack) {
            ass_free_track(m_vProcessData[i].pAssTrack);
            m_vProcessData[i].pAssTrack = nullptr;
        }
        if (m_vProcessData[i].pAssRenderer) {
            ass_renderer_done(m_vProcessData[i].pAssRenderer);
            m_vProcessData[i].pAssRenderer = nullptr;
        }
        if (m_vProcessData[i].pAssLibrary) {
            ass_library_done(m_vProcessData[i].pAssLibrary);
            m_vProcessData[i].pAssLibrary = nullptr;
        }
    }
    m_vProcessData.clear();

    m_sTasks.clear();
    m_sChunks.clear();

    mfxExtOpaqueSurfaceAlloc *pluginOpaqueAlloc = nullptr;

    if (m_bIsInOpaque || m_bIsOutOpaque) {
        if (nullptr == (pluginOpaqueAlloc = (mfxExtOpaqueSurfaceAlloc*)
            GetExtBuffer(m_VideoParam.ExtParam, m_VideoParam.NumExtParam, MFX_EXTBUFF_OPAQUE_SURFACE_ALLOCATION))) {
            return MFX_ERR_INVALID_VIDEO_PARAM;
        }
    }

    if ((m_bIsInOpaque && !pluginOpaqueAlloc->In.Surfaces) || (m_bIsOutOpaque && !pluginOpaqueAlloc->Out.Surfaces))
        return MFX_ERR_INVALID_VIDEO_PARAM;

    if (m_bIsInOpaque) {
        sts = m_mfxCore.UnmapOpaqueSurface(pluginOpaqueAlloc->In.NumSurface,
            pluginOpaqueAlloc->In.Type, pluginOpaqueAlloc->In.Surfaces);
        if (sts < MFX_ERR_NONE) return sts;
    }

    if (m_bIsOutOpaque) {
        sts = m_mfxCore.UnmapOpaqueSurface(pluginOpaqueAlloc->Out.NumSurface,
            pluginOpaqueAlloc->Out.Type, pluginOpaqueAlloc->Out.Surfaces);
        if (sts < MFX_ERR_NONE) return sts;
    }

    m_message.clear();
    m_bInited = false;

    return MFX_ERR_NONE;
}

mfxStatus ProcessorSubBurn::Init(mfxFrameSurface1 *frame_in, mfxFrameSurface1 *frame_out, const void *data) {
    if (frame_in == nullptr || frame_out == nullptr || data == nullptr) {
        return MFX_ERR_NULL_PTR;
    }

    m_pProcData = (ProcessDataSubBurn *)data;

    m_pIn = frame_in;
    m_pOut = frame_out;
    m_pOut->Data.TimeStamp = m_pIn->Data.TimeStamp;

    return MFX_ERR_NONE;
}

mfxStatus ProcessorSubBurn::ProcessSubText(uint8_t *pBuffer) {
    const uint32_t nSimdAvail = m_pProcData->nSimdAvail;

    const bool d3dSurface = !!(m_pProcData->memType & D3D9_MEMORY);
    mfxStatus sts = MFX_ERR_NONE;
    if (!d3dSurface) {
        if (MFX_ERR_NONE != (sts = LockFrame(m_pIn))) return sts;
        if (MFX_ERR_NONE != (sts = LockFrame(m_pOut))) {
            return UnlockFrame(m_pIn);
        }
    }

    AVPacket pkt;
    while (m_pProcData->qSubPackets.front_copy_and_pop_no_lock(&pkt)) {
        int got_sub = 0;
        AVSubtitle sub = { 0 };
        if (0 > avcodec_decode_subtitle2(m_pProcData->pOutCodecDecodeCtx, &sub, &got_sub, &pkt)) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to decode subtitle.\n"));
            return MFX_ERR_UNKNOWN;
        }

        if (got_sub) {
            const int64_t nStartTime = av_rescale_q(sub.pts, av_make_q(1, AV_TIME_BASE), av_make_q(1, 1000));
            const int64_t nDuration  = sub.end_display_time;
            for (uint32_t i = 0; i < sub.num_rects; i++) {
                auto *ass = sub.rects[i]->ass;
                if (!ass) {
                    break;
                }
                ass_process_chunk(m_pProcData->pAssTrack, ass, (int)strlen(ass), nStartTime, nDuration);
            }
        }
        avsubtitle_free(&sub);
        av_packet_unref(&pkt);
    }

    const auto frameTimebase = (m_pProcData->pVideoInputStream) ? m_pProcData->pVideoInputStream->time_base : HW_NATIVE_TIMEBASE;
    const double dTimeMs = (m_pIn->Data.TimeStamp - m_pProcData->nVideoInputFirstKeyPts) * av_q2d(frameTimebase) * 1000.0;

    int nDetectChange = 0;
    auto pFrameImages = ass_render_frame(m_pProcData->pAssRenderer, m_pProcData->pAssTrack, (int64_t)dTimeMs, &nDetectChange);

    if (d3dSurface) {
        if (MFX_ERR_NONE != (sts = CopyD3DFrameGPU(m_pIn, m_pOut))) {
            return sts;
        }
        if (pFrameImages) {
            if (MFX_ERR_NONE != (sts = LockFrame(m_pOut))) {
                return sts;
            }
        }
    }

    CopyFrameY(); //system memory mode の時のみ有効, d3d9 memoryの時はCopyD3DFrameGPUですでにコピーされている
    for (auto pImage = pFrameImages; pImage; pImage = pImage->next) {
        if (MFX_ERR_NONE != (sts = SubBurn<false>(pImage, pBuffer))) return sts;
    }
    CopyFrameUV(); //system memory mode の時のみ有効, d3d9 memoryの時はCopyD3DFrameGPUですでにコピーされている
    for (auto pImage = pFrameImages; pImage; pImage = pImage->next) {
        if (MFX_ERR_NONE != (sts = SubBurn<true>(pImage, pBuffer))) return sts;
    }

    if (!d3dSurface) {
        UnlockFrame(m_pIn);
    }
    if (!d3dSurface || pFrameImages) {
        UnlockFrame(m_pOut);
    }

    return sts;
}

mfxStatus ProcessorSubBurn::ProcessSubBitmap(uint8_t *pBuffer) {
    const uint32_t nSimdAvail = m_pProcData->nSimdAvail;

    const bool d3dSurface = !!(m_pProcData->memType & D3D9_MEMORY);
    mfxStatus sts = MFX_ERR_NONE;
    if (!d3dSurface) {
        if (MFX_ERR_NONE != (sts = LockFrame(m_pIn))) return sts;
        if (MFX_ERR_NONE != (sts = LockFrame(m_pOut))) {
            return UnlockFrame(m_pIn);
        }
    }

    const auto frameTimebase = (m_pProcData->pVideoInputStream) ? m_pProcData->pVideoInputStream->time_base : HW_NATIVE_TIMEBASE;
    const int64_t nFrameTimeMs = av_rescale_q((m_pIn->Data.TimeStamp - m_pProcData->nVideoInputFirstKeyPts), frameTimebase, { 1, 1000 });

    AVPacket pkt;
    while (m_pProcData->qSubPackets.front_copy_no_lock(&pkt)) {
        //字幕パケットのptsが、フレームのptsより古ければ、処理する必要がある
        if (nFrameTimeMs < av_rescale_q(pkt.pts, m_pProcData->pStreamIn->time_base, { 1, 1000 })) {
            //取得したパケットが未来のパケットなら無視
            break;
        }
        //字幕パケットをキューから取り除く
        m_pProcData->qSubPackets.pop();

        //現在蓄えているデコードされた字幕を開放
        if (m_pProcData->subtitle.num_rects) {
            avsubtitle_free(&m_pProcData->subtitle);
        }

        //字幕パケットをデコードする
        int got_sub = 0;
        if (0 > avcodec_decode_subtitle2(m_pProcData->pOutCodecDecodeCtx, &m_pProcData->subtitle, &got_sub, &pkt)) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to decode subtitle.\n"));
            return MFX_ERR_UNKNOWN;
        }
        av_packet_unref(&pkt);
    }

    //いまなんらかの字幕情報がデコード済みなら、その有効期限をチェックする
    if (m_pProcData->subtitle.num_rects) {
        const int64_t nStartTime = av_rescale_q(m_pProcData->subtitle.pts, av_make_q(1, AV_TIME_BASE), av_make_q(1, 1000));
        const int64_t nDuration  = m_pProcData->subtitle.end_display_time;
        if (nStartTime + nDuration < nFrameTimeMs) {
            //現在蓄えているデコードされた字幕を開放
            if (m_pProcData->subtitle.num_rects) {
                avsubtitle_free(&m_pProcData->subtitle);
                memset(&m_pProcData->subtitle, 0, sizeof(m_pProcData->subtitle));
            }
        }
    }

    if (d3dSurface) {
        if (MFX_ERR_NONE != (sts = CopyD3DFrameGPU(m_pIn, m_pOut))) {
            return sts;
        }
        if (m_pProcData->subtitle.num_rects) {
            if (MFX_ERR_NONE != (sts = LockFrame(m_pOut))) {
                return sts;
            }
        }
    }
    CopyFrameY(); //system memory mode の時のみ有効, d3d9 memoryの時はCopyD3DFrameGPUですでにコピーされている
    for (uint32_t i = 0; i < m_pProcData->subtitle.num_rects; i++) {
        SubBurn<false>(m_pProcData->subtitle.rects[i], pBuffer);
    }

    CopyFrameUV(); //system memory mode の時のみ有効, d3d9 memoryの時はCopyD3DFrameGPUですでにコピーされている
    for (uint32_t i = 0; i < m_pProcData->subtitle.num_rects; i++) {
        SubBurn<true>(m_pProcData->subtitle.rects[i], pBuffer);
    }

    if (!d3dSurface) {
        UnlockFrame(m_pIn);
    }
    if (!d3dSurface || m_pProcData->subtitle.num_rects) {
        UnlockFrame(m_pOut);
    }

    return sts;
}

mfxStatus ProcessorSubBurn::Process(DataChunk *chunk, uint8_t *pBuffer) {
    if (chunk == nullptr || pBuffer == nullptr) {
        return MFX_ERR_NULL_PTR;
    }

    if (m_pIn->Info.FourCC != MFX_FOURCC_NV12) {
        return MFX_ERR_UNSUPPORTED;
    }

    return (m_pProcData->nType & AV_CODEC_PROP_TEXT_SUB) ? ProcessSubText(pBuffer) : ProcessSubBitmap(pBuffer);
}

void ProcessorSubBurn::CopyFrameY() {
    const uint8_t *pFrameSrc = m_pIn->Data.Y;
    uint8_t *pFrameOut = m_pOut->Data.Y;
    const int w = m_pIn->Info.CropW;
    const int h = m_pIn->Info.CropH;
    const int pitch = m_pIn->Data.Pitch;
    for (int y = 0; y < h; y++, pFrameSrc += pitch, pFrameOut += pitch) {
        memcpy(pFrameOut, pFrameSrc, w);
    }
}

void ProcessorSubBurn::CopyFrameUV() {
    const uint8_t *pFrameSrc = m_pIn->Data.UV;
    uint8_t *pFrameOut = m_pOut->Data.UV;
    const int w = m_pIn->Info.CropW;
    const int h = m_pIn->Info.CropH;
    const int pitch = m_pIn->Data.Pitch;
    for (int y = 0; y < h; y += 2, pFrameSrc += pitch, pFrameOut += pitch) {
        memcpy(pFrameOut, pFrameSrc, w);
    }
}

void ProcessorSubBurn::BlendSubY(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcolory, uint8_t subTransparency, uint8_t *pBuf) {
    uint8_t *pFrame = m_pOut->Data.Y;
    const int w = m_pOut->Info.CropW;
    const int h = m_pOut->Info.CropH;
    const int pitch = m_pOut->Data.Pitch;
    const int subalpha = 255 - subTransparency;
    pFrame += bufY * pitch + bufX;
    bufW = (std::min)(w, bufX + bufW) - bufX;
    bufH = (std::min)(h, bufY + bufH) - bufY;
    for (int y = 0; y < bufH; y++, pFrame += pitch, pAlpha += bufStride) {
        uint8_t *ptr_dst = pFrame;
        uint8_t *ptr_dst_fin = ptr_dst + bufW;
        const uint8_t *ptr_alpha = pAlpha;
        for ( ; ptr_dst < ptr_dst_fin; ptr_dst++, ptr_alpha++) {
            int alpha = subalpha * ptr_alpha[0] >> 9;
            ptr_dst[0] = (uint8_t)((ptr_dst[0] * (127 - alpha) + subcolory * alpha + 256) >> 7);
        }
    }
}

void ProcessorSubBurn::BlendSubUV(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcoloru, uint8_t subcolorv, uint8_t subTransparency, uint8_t *pBuf) {
    uint8_t *pFrame = m_pOut->Data.UV;
    const int w = m_pOut->Info.CropW;
    const int h = m_pOut->Info.CropH >> 1;
    const int pitch = m_pOut->Data.Pitch;
    const int subalpha = 255 - subTransparency;
    pFrame += (bufY >> 1) * pitch + bufX;
    bufW = (std::min)(w, bufX + bufW) - bufX;
    bufH = (std::min)(h, bufY + bufH) - bufY;
    for (int y = 0; y < bufH; y += 2, pFrame += pitch, pAlpha += (bufStride << 1)) {
        uint8_t *ptr_dst = pFrame;
        uint8_t *ptr_dst_fin = ptr_dst + bufW;
        const uint8_t *ptr_alpha = pAlpha;
        for ( ; ptr_dst < ptr_dst_fin; ptr_dst += 2, ptr_alpha += 2) {
            int alpha = subalpha * ptr_alpha[0] >> 9;
            ptr_dst[0] = (uint8_t)((ptr_dst[0] * (127 - alpha) + subcoloru * alpha + 256) >> 7);
            ptr_dst[1] = (uint8_t)((ptr_dst[1] * (127 - alpha) + subcolorv * alpha + 256) >> 7);
        }
    }
}

template<bool forUV>
mfxStatus ProcessorSubBurn::SubBurn(ASS_Image *pImage, uint8_t *pBuffer) {
    const uint32_t nSubColor = pImage->color;
    const uint8_t subR = (uint8_t) (nSubColor >> 24);
    const uint8_t subG = (uint8_t)((nSubColor >> 16) & 0xff);
    const uint8_t subB = (uint8_t)((nSubColor >>  8) & 0xff);
    const uint8_t subA = (uint8_t) (nSubColor        & 0xff);

    const uint8_t subY = (uint8_t)clamp((( 66 * subR + 129 * subG +  25 * subB + 128) >> 8) +  16, 0, 255);
    const uint8_t subU = (uint8_t)clamp(((-38 * subR -  74 * subG + 112 * subB + 128) >> 8) + 128, 0, 255);
    const uint8_t subV = (uint8_t)clamp(((112 * subR -  94 * subG -  18 * subB + 128) >> 8) + 128, 0, 255);

    if (!forUV)
        BlendSubY( pImage->bitmap, pImage->dst_x + m_pProcData->sCrop.e.left, pImage->dst_y + m_pProcData->sCrop.e.up, pImage->w, pImage->stride, pImage->h, subY, subA, pBuffer);
    else
        BlendSubUV(pImage->bitmap, pImage->dst_x + m_pProcData->sCrop.e.left, pImage->dst_y + m_pProcData->sCrop.e.up, pImage->w, pImage->stride, pImage->h, subU, subV, subA, pBuffer);

    return MFX_ERR_NONE;
}

int ProcessorSubBurn::BlendSubYBitmap(const uint8_t *pSubColorIdx, int nColorLUT, const uint8_t *pSubColor, const uint8_t *pAlpha, int subX, int subY, int subW, int subStride, int subH, uint8_t *pBuf) {
    return 256;
}
int ProcessorSubBurn::BlendSubUVBitmap(const uint8_t *pSubColorIdx, int nColorLUT, const uint8_t *pSubColor, const uint8_t *pAlpha, int subX, int subY, int subW, int subStride, int subH, uint8_t *pBuf) {
    return 256;
}

template<bool forUV>
mfxStatus ProcessorSubBurn::SubBurn(AVSubtitleRect *pRect, uint8_t *pBuffer) {
    uint32_t nColorTableSize = pRect->nb_colors;
#define INT(b) ((b) ? 1 : 0)
    alignas(32) uint8_t pColor[256 << (INT(forUV))];
    alignas(32) uint8_t pAlpha[256];

    const uint32_t *pColorARGB = (uint32_t *)pRect->data[1];
    for (uint32_t ic = 0; ic < nColorTableSize; ic++) {
        const uint32_t nSubColor = pColorARGB[ic];
        const uint8_t subA = (uint8_t) (nSubColor >> 24);
        const uint8_t subR = (uint8_t)((nSubColor >> 16) & 0xff);
        const uint8_t subG = (uint8_t)((nSubColor >>  8) & 0xff);
        const uint8_t subB = (uint8_t) (nSubColor        & 0xff);

        const uint8_t subY = (uint8_t)clamp((( 66 * subR + 129 * subG +  25 * subB + 128) >> 8) +  16, 0, 255);
        const uint8_t subU = (uint8_t)clamp(((-38 * subR -  74 * subG + 112 * subB + 128) >> 8) + 128, 0, 255);
        const uint8_t subV = (uint8_t)clamp(((112 * subR -  94 * subG -  18 * subB + 128) >> 8) + 128, 0, 255);

        if (forUV) {
            pColor[2*ic + 0] = subU;
            pColor[2*ic + 1] = subV;
        } else {
            pColor[ic] = subY;
        }
        pAlpha[ic] = subA >> 1;
    }
    int nMaxIndex = (forUV)
        ? BlendSubUVBitmap(pRect->data[0], pRect->nb_colors, pColor, pAlpha, pRect->x + m_pProcData->sCrop.e.left, pRect->y + m_pProcData->sCrop.e.up, pRect->w, pRect->linesize[0], pRect->h, pBuffer)
        : BlendSubYBitmap(pRect->data[0], pRect->nb_colors, pColor, pAlpha, pRect->x + m_pProcData->sCrop.e.left, pRect->y + m_pProcData->sCrop.e.up, pRect->w, pRect->linesize[0], pRect->h, pBuffer);
    pRect->nb_colors = (std::min)(pRect->nb_colors, nMaxIndex);
    return MFX_ERR_NONE;
}


#endif //#if ENABLE_AVSW_READER && ENABLE_LIBASS_SUBBURN
