// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc/rkmppenc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2020-2021 rigaya
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
// ------------------------------------------------------------------------------------------

#define _USE_MATH_DEFINES
#include <cmath>
#include <map>
#include <array>
#include <filesystem>
#include "rgy_filter_subburn.h"
#include "rgy_filesystem.h"
#include "rgy_codepage.h"

#if ENABLE_AVSW_READER && ENABLE_LIBASS_SUBBURN

#pragma comment(lib, "libass-9.lib")

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
static inline RGYLogLevel log_level_ass2qsv(int level) {
    static const RGYLogLevel log_level_map[] = {
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
    const auto rgy_log_level = log_level_ass2qsv(ass_level);
    if (ctx == nullptr || rgy_log_level < ((RGYLog *)ctx)->getLogLevel(RGY_LOGT_LIBASS)) {
        return;
    }
    ((RGYLog *)ctx)->write_line(rgy_log_level, RGY_LOGT_LIBASS, fmt, args, CP_UTF8);
}

static void ass_log_error_only(int ass_level, const char *fmt, va_list args, void *ctx) {
    const auto rgy_log_level = log_level_ass2qsv(ass_level);
    if (rgy_log_level >= RGY_LOG_ERROR) {
        ((RGYLog *)ctx)->write_line(rgy_log_level, RGY_LOGT_LIBASS, fmt, args, CP_UTF8);
    }
}

static bool font_attached(const AVStream *stream) {
    const AVDictionaryEntry *tag = av_dict_get(stream->metadata, "mimetype", NULL, AV_DICT_MATCH_CASE);
    if (tag) {
        const auto font_mimetypes = make_array<std::string>(
            "font/ttf",
            "font/otf",
            "font/sfnt",
            "font/woff",
            "font/woff2",
            "application/font-sfnt",
            "application/font-woff",
            "application/x-font-ttf",
            "application/x-truetype-font",
            "application/vnd.ms-opentype");
        for (const auto &mime : font_mimetypes) {
            if (tolowercase(mime) == tolowercase(tag->value)) {
                return true;
            }
        }
    }
    return false;
}

RGY_ERR RGYFilterSubburn::procFrame(RGYFrameInfo *pFrame,
    const RGYFrameInfo *pSubImg,
    int pos_x, int pos_y,
    float transparency_offset, float brightness, float contrast,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    //焼きこみフレームの範囲内に収まるようチェック
    const int burnWidth = std::min((pos_x & ~1) + pSubImg->width, pFrame->width) - (pos_x & ~1);
    const int burnHeight = std::min((pos_y & ~1) + pSubImg->height, pFrame->height) - (pos_y & ~1);
    if (burnWidth <= 0 || burnHeight <= 0) {
        return RGY_ERR_NONE;
    }

    RGYWorkSize local(32, 8);
    RGYWorkSize global(divCeil(burnWidth, 2), divCeil(burnHeight, 2));
    auto planeFrameY = getPlane(pFrame, RGY_PLANE_Y);
    auto planeFrameU = getPlane(pFrame, RGY_PLANE_U);
    auto planeFrameV = getPlane(pFrame, RGY_PLANE_V);
    auto planeSubY = getPlane(pSubImg, RGY_PLANE_Y);
    auto planeSubU = getPlane(pSubImg, RGY_PLANE_U);
    auto planeSubV = getPlane(pSubImg, RGY_PLANE_V);
    auto planeSubA = getPlane(pSubImg, RGY_PLANE_A);

    const int pixSize = RGY_CSP_BIT_DEPTH[pFrame->csp] > 8 ? 2 : 1;
    const int subPosX_Y = (pos_x & ~1);
    const int subPosY_Y = (pos_y & ~1);
    const int subPosX_UV = (RGY_CSP_CHROMA_FORMAT[pFrame->csp] == RGY_CHROMAFMT_YUV420) ? (pos_x >> 1) : (pos_x & ~1);
    const int subPosY_UV = (RGY_CSP_CHROMA_FORMAT[pFrame->csp] == RGY_CHROMAFMT_YUV420) ? (pos_y >> 1) : (pos_y & ~1);
    const int frameOffsetByteY = subPosY_Y  * planeFrameY.pitch[0] + subPosX_Y  * pixSize;
    const int frameOffsetByteU = subPosY_UV * planeFrameU.pitch[0] + subPosX_UV * pixSize;
    const int frameOffsetByteV = subPosY_UV * planeFrameV.pitch[0] + subPosX_UV * pixSize;

    if (   planeSubY.pitch[0] != planeSubU.pitch[0]
        || planeSubY.pitch[0] != planeSubV.pitch[0]
        || planeSubY.pitch[0] != planeSubA.pitch[0]) {
        AddMessage(RGY_LOG_ERROR, _T("plane pitch error!\n"));
        return RGY_ERR_UNKNOWN;
    }

    const char *kernel_name = "kernel_subburn";
    auto err = m_subburn.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        planeFrameY.ptr[0],
        planeFrameU.ptr[0],
        planeFrameV.ptr[0],
        planeFrameY.pitch[0],
        planeFrameU.pitch[0],
        planeFrameV.pitch[0],
        frameOffsetByteY,
        frameOffsetByteU,
        frameOffsetByteV,
        planeSubY.ptr[0], planeSubU.ptr[0], planeSubV.ptr[0], planeSubA.ptr[0], planeSubY.pitch[0],
        burnWidth, burnHeight, interlaced(*pFrame) ? 1 : 0, transparency_offset, brightness, contrast);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procFrame(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pFrame->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

SubImageData RGYFilterSubburn::textRectToImage(const ASS_Image *image, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    //YUV420の関係で縦横2pixelずつ処理するので、2で割り切れている必要がある
    const int x_offset = ((image->dst_x % 2) != 0) ? 1 : 0;
    const int y_offset = ((image->dst_y % 2) != 0) ? 1 : 0;
    auto frameTemp = m_cl->createFrameBuffer(ALIGN(image->w + x_offset, 2), ALIGN(image->h + y_offset, 2), RGY_CSP_YUVA444, RGY_CSP_BIT_DEPTH[RGY_CSP_YUVA444]);
    frameTemp->queueMapBuffer(queue, CL_MAP_WRITE, wait_events);
    frameTemp->mapWait();
    auto img = frameTemp->mappedHost()->frameInfo();

    auto planeY = getPlane(&img, RGY_PLANE_Y);
    auto planeU = getPlane(&img, RGY_PLANE_U);
    auto planeV = getPlane(&img, RGY_PLANE_V);
    auto planeA = getPlane(&img, RGY_PLANE_A);

    //とりあえずすべて0で初期化しておく
    memset(planeY.ptr[0], 0, (size_t)planeY.pitch[0] * planeY.height);

    //とりあえずすべて0で初期化しておく
    //Alpha=0で透明なので都合がよい
    memset(planeA.ptr[0], 0, (size_t)planeA.pitch[0] * planeA.height);

    for (int j = 0; j < planeU.height; j++) {
        uint8_t *ptr = planeU.ptr[0] + j * planeU.pitch[0];
        for (size_t i = 0; i < planeU.pitch[0] / sizeof(ptr[0]); i++) {
            ptr[i] = 128;
        }
    }
    for (int j = 0; j < planeV.height; j++) {
        uint8_t *ptr = planeV.ptr[0] + j * planeV.pitch[0];
        for (size_t i = 0; i < planeV.pitch[0] / sizeof(ptr[0]); i++) {
            ptr[i] = 128;
        }
    }

    const uint32_t subColor = image->color;
    const uint8_t subR = (uint8_t) (subColor >> 24);
    const uint8_t subG = (uint8_t)((subColor >> 16) & 0xff);
    const uint8_t subB = (uint8_t)((subColor >>  8) & 0xff);
    const uint8_t subA = (uint8_t)(255 - (subColor        & 0xff));

    const uint8_t subY = (uint8_t)clamp((( 66 * subR + 129 * subG +  25 * subB + 128) >> 8) +  16, 0, 255);
    const uint8_t subU = (uint8_t)clamp(((-38 * subR -  74 * subG + 112 * subB + 128) >> 8) + 128, 0, 255);
    const uint8_t subV = (uint8_t)clamp(((112 * subR -  94 * subG -  18 * subB + 128) >> 8) + 128, 0, 255);

    //YUVで字幕の画像データを構築
    for (int j = 0; j < image->h; j++) {
        for (int i = 0; i < image->w; i++) {
            const int src_idx = j * image->stride + i;
            const uint8_t alpha = image->bitmap[src_idx];

            #define PLANE_DST(plane, x, y) (plane.ptr[0][(y) * plane.pitch[0] + (x)])
            PLANE_DST(planeY, i + x_offset, j + y_offset) = subY;
            PLANE_DST(planeU, i + x_offset, j + y_offset) = subU;
            PLANE_DST(planeV, i + x_offset, j + y_offset) = subV;
            PLANE_DST(planeA, i + x_offset, j + y_offset) = (uint8_t)clamp(((int)subA * alpha) >> 8, 0, 255);
            #undef PLANE_DST
        }
    }
    //GPUへ転送
    frameTemp->unmapBuffer(queue);
    return SubImageData(std::move(frameTemp), std::unique_ptr<RGYCLFrame>(), image->dst_x, image->dst_y);
}

RGY_ERR RGYFilterSubburn::procFrameText(RGYFrameInfo *pOutputFrame, int64_t frameTimeMs, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    int nDetectChange = 0;
    const auto frameImages = ass_render_frame(m_assRenderer.get(), m_assTrack.get(), frameTimeMs, &nDetectChange);

    if (!frameImages) {
        m_subImages.clear();
    } else if (nDetectChange) {
        m_subImages.clear();
        for (auto image = frameImages; image; image = image->next) {
            if (image->w > 0 && image->h > 0) {
                m_subImages.push_back(textRectToImage(image, queue, wait_events));
            }
        }
    }
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSubburn>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_subImages.size()) {
        for (uint32_t irect = 0; irect < m_subImages.size(); irect++) {
            const RGYFrameInfo *pSubImg = &m_subImages[irect].image->frame;
            auto err = procFrame(pOutputFrame, pSubImg, m_subImages[irect].x, m_subImages[irect].y,
                prm->subburn.transparency_offset, prm->subburn.brightness, prm->subburn.contrast, queue, wait_events, event);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at subburn(%s): %s.\n"),
                    RGY_CSP_NAMES[pOutputFrame->csp],
                    get_err_mes(err));
                return RGY_ERR_CUDA;
            }
        }
    }
    return RGY_ERR_NONE;
}

SubImageData RGYFilterSubburn::bitmapRectToImage(const AVSubtitleRect *rect, const RGYFrameInfo *outputFrame, const sInputCrop &crop, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events) {
    //YUV420の関係で縦横2pixelずつ処理するので、2で割り切れている必要がある
    const int x_offset = ((rect->x % 2) != 0) ? 1 : 0;
    const int y_offset = ((rect->y % 2) != 0) ? 1 : 0;
    auto frameTemp = m_cl->createFrameBuffer(ALIGN(rect->w + x_offset, 2), ALIGN(rect->h + y_offset, 2), RGY_CSP_YUVA444, RGY_CSP_BIT_DEPTH[RGY_CSP_YUVA444]);
    frameTemp->queueMapBuffer(queue, CL_MAP_WRITE, wait_events);
    frameTemp->mapWait();
    auto img = frameTemp->mappedHost()->frameInfo();

    auto planeY = getPlane(&img, RGY_PLANE_Y);
    auto planeU = getPlane(&img, RGY_PLANE_U);
    auto planeV = getPlane(&img, RGY_PLANE_V);
    auto planeA = getPlane(&img, RGY_PLANE_A);

    //とりあえずすべて0で初期化しておく
    memset(planeY.ptr[0], 0, (size_t)planeY.pitch[0] * planeY.height);

    //とりあえずすべて0で初期化しておく
    //Alpha=0で透明なので都合がよい
    memset(planeA.ptr[0], 0, (size_t)planeA.pitch[0] * planeA.height);

    for (int j = 0; j < planeU.height; j++) {
        uint8_t *ptr = planeU.ptr[0] + j * planeU.pitch[0];
        for (size_t i = 0; i < planeU.pitch[0] / sizeof(ptr[0]); i++) {
            ptr[i] = 128;
        }
    }
    for (int j = 0; j < planeV.height; j++) {
        uint8_t *ptr = planeV.ptr[0] + j * planeU.pitch[0];
        for (size_t i = 0; i < planeV.pitch[0] / sizeof(ptr[0]); i++) {
            ptr[i] = 128;
        }
    }

    //色テーブルをRGBA->YUVAに変換
    const uint32_t *pColorARGB = (uint32_t *)rect->data[1];
    alignas(32) uint32_t colorTableYUVA[256];
    memset(colorTableYUVA, 0, sizeof(colorTableYUVA));

    const uint32_t nColorTableSize = rect->nb_colors;
    assert(nColorTableSize <= _countof(colorTableYUVA));
    for (uint32_t ic = 0; ic < nColorTableSize; ic++) {
        const uint32_t subColor = pColorARGB[ic];
        const uint8_t subA = (uint8_t)(subColor >> 24);
        const uint8_t subR = (uint8_t)((subColor >> 16) & 0xff);
        const uint8_t subG = (uint8_t)((subColor >>  8) & 0xff);
        const uint8_t subB = (uint8_t)(subColor        & 0xff);

        const uint8_t subY = (uint8_t)clamp((( 66 * subR + 129 * subG +  25 * subB + 128) >> 8) +  16, 0, 255);
        const uint8_t subU = (uint8_t)clamp(((-38 * subR -  74 * subG + 112 * subB + 128) >> 8) + 128, 0, 255);
        const uint8_t subV = (uint8_t)clamp(((112 * subR -  94 * subG -  18 * subB + 128) >> 8) + 128, 0, 255);

        colorTableYUVA[ic] = ((subA << 24) | (subV << 16) | (subU << 8) | subY);
    }

    //YUVで字幕の画像データを構築
    for (int j = 0; j < rect->h; j++) {
        for (int i = 0; i < rect->w; i++) {
            const int src_idx = j * rect->linesize[0] + i;
            const int ic = rect->data[0][src_idx];

            const uint32_t subColor = colorTableYUVA[ic];
            const uint8_t subA = (uint8_t)(subColor >> 24);
            const uint8_t subV = (uint8_t)((subColor >> 16) & 0xff);
            const uint8_t subU = (uint8_t)((subColor >>  8) & 0xff);
            const uint8_t subY = (uint8_t)(subColor        & 0xff);

            #define PLANE_DST(plane, x, y) (plane.ptr[0][(y) * plane.pitch[0] + (x)])
            PLANE_DST(planeY, i + x_offset, j + y_offset) = subY;
            PLANE_DST(planeU, i + x_offset, j + y_offset) = subU;
            PLANE_DST(planeV, i + x_offset, j + y_offset) = subV;
            PLANE_DST(planeA, i + x_offset, j + y_offset) = subA;
            #undef PLANE_DST
        }
    }

    //GPUへ転送
    frameTemp->unmapBuffer(queue);
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSubburn>(m_param);

    decltype(frameTemp) frame;
    if (prm->subburn.scale == 1.0f) {
        frame = std::move(frameTemp);
    } else {

        frame = m_cl->createFrameBuffer(
            ALIGN((int)(img.width  * prm->subburn.scale + 0.5f), 4),
            ALIGN((int)(img.height * prm->subburn.scale + 0.5f), 4), img.csp, img.bitdepth);
        unique_ptr<RGYFilterResize> filterResize(new RGYFilterResize(m_cl));
        shared_ptr<RGYFilterParamResize> paramResize(new RGYFilterParamResize());
        paramResize->frameIn = frameTemp->frame;
        paramResize->frameOut = frame->frame;
        paramResize->baseFps = prm->baseFps;
        paramResize->bOutOverwrite = false;
        paramResize->interp = RGY_VPP_RESIZE_BILINEAR;
        filterResize->init(paramResize, m_pLog);
        m_resize = std::move(filterResize);

        int filterOutputNum = 0;
        RGYFrameInfo *filterOutput[1] = { &frame->frame };
        m_resize->filter(&frameTemp->frame, (RGYFrameInfo **)&filterOutput, &filterOutputNum, queue);
    }
    int x_pos = ALIGN((int)(prm->subburn.scale * rect->x + 0.5f) - ((crop.e.left + crop.e.right) / 2), 2);
    int y_pos = ALIGN((int)(prm->subburn.scale * rect->y + 0.5f) - crop.e.up - crop.e.bottom, 2);
    if (m_outCodecDecodeCtx->height > 0) {
        const double y_factor = rect->y / (double)m_outCodecDecodeCtx->height;
        y_pos = ALIGN((int)(outputFrame->height * y_factor + 0.5f), 2);
        y_pos = std::min(y_pos, outputFrame->height - rect->h);
    }
    return SubImageData(std::move(frame), std::move(frameTemp), x_pos, y_pos);
}


RGY_ERR RGYFilterSubburn::procFrameBitmap(RGYFrameInfo *pOutputFrame, const int64_t frameTimeMs, const sInputCrop &crop, const bool forced_subs_only, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (m_subData) {
        if (m_subData->num_rects != m_subImages.size()) {
            for (uint32_t irect = 0; irect < m_subData->num_rects; irect++) {
                const AVSubtitleRect *rect = m_subData->rects[irect];
                if (forced_subs_only && !(rect->flags & AV_SUBTITLE_FLAG_FORCED)) {
                    AddMessage(RGY_LOG_DEBUG, _T("skipping non-forced sub at %s\n"), getTimestampString(frameTimeMs, av_make_q(1, 1000)).c_str());
                    // 空の値をいれる
                    m_subImages.push_back(SubImageData(std::unique_ptr<RGYCLFrame>(), std::unique_ptr<RGYCLFrame>(), 0, 0));
                } else if (rect->w == 0 || rect->h == 0) {
                    // 空の値をいれる
                    m_subImages.push_back(SubImageData(std::unique_ptr<RGYCLFrame>(), std::unique_ptr<RGYCLFrame>(), 0, 0));
                } else {
                    m_subImages.push_back(bitmapRectToImage(rect, pOutputFrame, crop, queue, wait_events));
                }
            }
        }
        if ((m_subData->num_rects != m_subImages.size())) {
            AddMessage(RGY_LOG_ERROR, _T("unexpected error.\n"));
            return RGY_ERR_UNKNOWN;
        }
        auto prm = std::dynamic_pointer_cast<RGYFilterParamSubburn>(m_param);
        if (!prm) {
            AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        for (uint32_t irect = 0; irect < m_subImages.size(); irect++) {
            if (m_subImages[irect].image) {
                const RGYFrameInfo *pSubImg = &m_subImages[irect].image->frame;
                auto err = procFrame(pOutputFrame, pSubImg, m_subImages[irect].x, m_subImages[irect].y,
                    prm->subburn.transparency_offset, prm->subburn.brightness, prm->subburn.contrast, queue, wait_events, event);
                if (err != RGY_ERR_NONE) {
                    AddMessage(RGY_LOG_ERROR, _T("error at subburn(%s): %s.\n"),
                        RGY_CSP_NAMES[pOutputFrame->csp],
                        get_err_mes(err));
                    return RGY_ERR_CUDA;
                }
            }
        }
    }
    return RGY_ERR_NONE;
}


RGYFilterSubburn::RGYFilterSubburn(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context),
    m_subType(0),
    m_formatCtx(),
    m_subtitleStreamIndex(-1),
    m_outCodecDecode(nullptr),
    m_outCodecDecodeCtx(),
    m_subData(),
    m_subImages(),
    m_assLibrary(unique_ptr<ASS_Library, decltype(&ass_library_done)>(nullptr, ass_library_done)),
    m_assRenderer(unique_ptr<ASS_Renderer, decltype(&ass_renderer_done)>(nullptr, ass_renderer_done)),
    m_assTrack(unique_ptr<ASS_Track, decltype(&ass_free_track)>(nullptr, ass_free_track)),
    m_resize(),
    m_poolPkt(nullptr),
    m_queueSubPackets(),
    m_subburn() {
    m_name = _T("subburn");
}

RGYFilterSubburn::~RGYFilterSubburn() {
    close();
}

RGY_ERR RGYFilterSubburn::checkParam(const std::shared_ptr<RGYFilterParamSubburn> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid frame size.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->subburn.filename.length() > 0 && prm->subburn.trackId != 0) {
        AddMessage(RGY_LOG_ERROR, _T("track and filename should not be set at the same time.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->subburn.filename.length() > 0 && !rgy_file_exists(prm->subburn.filename.c_str())) {
        AddMessage(RGY_LOG_ERROR, _T("subtitle file \"%s\" does not exist\n"), prm->subburn.filename.c_str());
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->subburn.trackId != 0) {
        prm->subburn.vid_ts_offset = true;
    }
    if (prm->subburn.brightness < -1.0f || 1.0f < prm->subburn.brightness) {
        AddMessage(RGY_LOG_ERROR, _T("\"brightness\" must be in range of -1.0 - 1.0, but %.2f set.\n"), prm->subburn.brightness);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->subburn.contrast < -2.0f || 2.0f < prm->subburn.contrast) {
        AddMessage(RGY_LOG_ERROR, _T("\"contrast\" must be in range of -2.0 - 2.0, but %.2f set.\n"), prm->subburn.contrast);
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->subburn.transparency_offset < 0.0f || 1.0f < prm->subburn.transparency_offset) {
        AddMessage(RGY_LOG_ERROR, _T("\"transparency\" must be in range of 0.0 - 1.0, but %.2f set.\n"), prm->subburn.transparency_offset);
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}


void RGYFilterSubburn::SetExtraData(AVCodecContext *codecCtx, const uint8_t *data, uint32_t size) {
    if (data == nullptr || size == 0)
        return;
    if (codecCtx->extradata)
        av_free(codecCtx->extradata);
    codecCtx->extradata_size = size;
    codecCtx->extradata      = (uint8_t *)av_malloc(codecCtx->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
    memcpy(codecCtx->extradata, data, size);
};

RGY_ERR RGYFilterSubburn::initAVCodec(const std::shared_ptr<RGYFilterParamSubburn> prm) {
    auto inputCodecId = AV_CODEC_ID_NONE;
    if (prm->subburn.filename.length() > 0) {
        //ファイル読み込みの場合
        AddMessage(RGY_LOG_DEBUG, _T("trying to open subtitle file \"%s\""), prm->subburn.filename.c_str());

        std::string filename_char;
        if (0 == tchar_to_string(prm->subburn.filename.c_str(), filename_char, CP_UTF8)) {
            AddMessage(RGY_LOG_ERROR, _T("failed to convert filename to utf-8 characters.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        {
            AVFormatContext *tmpFormatCtx = nullptr;
            int ret = avformat_open_input(&tmpFormatCtx, filename_char.c_str(), nullptr, nullptr);
            if (ret < 0) {
                AddMessage(RGY_LOG_ERROR, _T("error opening file: \"%s\": %s\n"), char_to_tstring(filename_char, CP_UTF8).c_str(), qsv_av_err2str(ret).c_str());
                return RGY_ERR_FILE_OPEN; // Couldn't open file
            }
            m_formatCtx = unique_ptr<AVFormatContext, RGYAVDeleter<AVFormatContext>>(tmpFormatCtx, RGYAVDeleter<AVFormatContext>(avformat_close_input));
        }

        if (avformat_find_stream_info(m_formatCtx.get(), nullptr) < 0) {
            AddMessage(RGY_LOG_ERROR, _T("error finding stream information.\n"));
            return RGY_ERR_INVALID_FORMAT; // Couldn't find stream information
        }
        AddMessage(RGY_LOG_DEBUG, _T("got stream information.\n"));
        av_dump_format(m_formatCtx.get(), 0, filename_char.c_str(), 0);

        if (0 > (m_subtitleStreamIndex = av_find_best_stream(m_formatCtx.get(), AVMEDIA_TYPE_SUBTITLE, -1, -1, nullptr, 0))) {
            AddMessage(RGY_LOG_ERROR, _T("no subtitle stream found in \"%s\".\n"), char_to_tstring(filename_char, CP_UTF8).c_str());
            return RGY_ERR_INVALID_FORMAT; // Couldn't open file
        }
        const auto pstream = m_formatCtx->streams[m_subtitleStreamIndex];
        inputCodecId = pstream->codecpar->codec_id;
        AddMessage(RGY_LOG_DEBUG, _T("found subtitle in stream #%d (%s), timebase %d/%d.\n"),
            m_subtitleStreamIndex, char_to_tstring(avcodec_get_name(inputCodecId)).c_str(),
            pstream->time_base.num, pstream->time_base.den);
    } else {
        if (prm->streamIn.stream == nullptr) {
            AddMessage(RGY_LOG_ERROR, _T("internal error: stream info not provided.\n"));
            return RGY_ERR_UNKNOWN;
        }
        inputCodecId = prm->streamIn.stream->codecpar->codec_id;
        AddMessage(RGY_LOG_DEBUG, _T("using subtitle track #%d (%s), timebase %d/%d.\n"),
            prm->subburn.trackId, char_to_tstring(avcodec_get_name(inputCodecId)).c_str(),
            prm->streamIn.stream->time_base.num, prm->streamIn.stream->time_base.den);
    }

    m_subType = avcodec_descriptor_get(inputCodecId)->props;
    AddMessage(RGY_LOG_DEBUG, _T("sub type: %s\n"), (m_subType & AV_CODEC_PROP_TEXT_SUB) ? _T("text") : _T("bitmap"));

    auto copy_subtitle_header = [](AVCodecContext *pDstCtx, const AVCodecContext *pSrcCtx) {
        if (pSrcCtx->subtitle_header_size) {
            pDstCtx->subtitle_header_size = pSrcCtx->subtitle_header_size;
            pDstCtx->subtitle_header = (uint8_t *)av_mallocz(pDstCtx->subtitle_header_size + AV_INPUT_BUFFER_PADDING_SIZE);
            memcpy(pDstCtx->subtitle_header, pSrcCtx->subtitle_header, pSrcCtx->subtitle_header_size);
        }
    };
    //decoderの初期化
    if (NULL == (m_outCodecDecode = avcodec_find_decoder(inputCodecId))) {
        AddMessage(RGY_LOG_ERROR, errorMesForCodec(_T("failed to find decoder"), inputCodecId));
        AddMessage(RGY_LOG_ERROR, _T("Please use --check-decoders to check available decoder.\n"));
        return RGY_ERR_NULL_PTR;
    }
    m_outCodecDecodeCtx = unique_ptr<AVCodecContext, RGYAVDeleter<AVCodecContext>>(avcodec_alloc_context3(m_outCodecDecode), RGYAVDeleter<AVCodecContext>(avcodec_free_context));
    if (prm->streamIn.stream) {
        //設定されていない必須情報があれば設定する
#define COPY_IF_ZERO(dst, src) { if ((dst)==0) (dst)=(src); }
        COPY_IF_ZERO(m_outCodecDecodeCtx->width, prm->streamIn.stream->codecpar->width);
        COPY_IF_ZERO(m_outCodecDecodeCtx->height, prm->streamIn.stream->codecpar->height);
#undef COPY_IF_ZERO
        m_outCodecDecodeCtx->pkt_timebase = prm->streamIn.stream->time_base;
        SetExtraData(m_outCodecDecodeCtx.get(), prm->streamIn.stream->codecpar->extradata, prm->streamIn.stream->codecpar->extradata_size);
    } else {
        m_outCodecDecodeCtx->pkt_timebase = m_formatCtx->streams[m_subtitleStreamIndex]->time_base;
        auto *codecpar = m_formatCtx->streams[m_subtitleStreamIndex]->codecpar;
        SetExtraData(m_outCodecDecodeCtx.get(), codecpar->extradata, codecpar->extradata_size);
    }

    int ret;
    AVDictionary *pCodecOpts = nullptr;
    if (m_subType & AV_CODEC_PROP_TEXT_SUB) {
        if (prm->subburn.filename.length() > 0) {
            if (prm->subburn.charcode.length() == 0) {
                FILE *fp = NULL;
                if (_tfopen_s(&fp, prm->subburn.filename.c_str(), _T("rb")) || fp == NULL) {
                    AddMessage(RGY_LOG_ERROR, _T("error opening file: \"%s\"\n"), prm->subburn.filename.c_str());
                    return RGY_ERR_NULL_PTR; // Couldn't open file
                }

                std::vector<char> buffer(256 * 1024, 0);
                const auto readBytes = fread(buffer.data(), 1, sizeof(buffer[0]) * buffer.size(), fp);
                fclose(fp);

                const auto estCodePage = get_code_page(buffer.data(), (int)readBytes);
                std::map<uint32_t, std::string> codePageMap = {
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
                    prm->subburn.charcode = codePageMap[estCodePage];
                }
            }
        }
        if (prm->subburn.charcode.length() > 0) {
            if (0 > (ret = av_dict_set(&pCodecOpts, "sub_charenc", prm->subburn.charcode.c_str(), 0))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to set \"sub_charenc\" option for subtitle decoder: %s\n"), qsv_av_err2str(ret).c_str());
                return RGY_ERR_NULL_PTR;
            }
        }
        AddMessage(RGY_LOG_DEBUG, _T("set \"sub_charenc\" to \"%s\""), char_to_tstring(prm->subburn.charcode).c_str());

        const auto avcodec_ver = avcodec_version();
        const auto avcodec_ver_major = (avcodec_ver >> 16) & 0xff;
        const auto avcodec_ver_minor = (avcodec_ver >>  8) & 0xff;
        if (avcodec_ver_major < 59 || (avcodec_ver_major == 59 && avcodec_ver_minor <= 8)) {
            if (0 > (ret = av_dict_set(&pCodecOpts, "sub_text_format", "ass", 0))) {
                AddMessage(RGY_LOG_ERROR, _T("failed to set \"sub_text_format\" option for subtitle decoder: %s\n"), qsv_av_err2str(ret).c_str());
                return RGY_ERR_NULL_PTR;
            }
            AddMessage(RGY_LOG_DEBUG, _T("set \"sub_text_format\" to \"ass\""));
        }
    }
    if (0 > (ret = avcodec_open2(m_outCodecDecodeCtx.get(), m_outCodecDecode, &pCodecOpts))) {
        AddMessage(RGY_LOG_ERROR, _T("failed to open decoder for %s: %s\n"),
            char_to_tstring(avcodec_get_name(inputCodecId)).c_str(), qsv_av_err2str(ret).c_str());
        return RGY_ERR_NULL_PTR;
    }
    if (prm->subburn.trackId == 0) {
        AddMessage(RGY_LOG_DEBUG, _T("Subtitle Decoder opened\n"));
        AddMessage(RGY_LOG_DEBUG, _T("Subtitle Decode Info: %s, %dx%d\n"), char_to_tstring(avcodec_get_name(inputCodecId)).c_str(),
            m_outCodecDecodeCtx->width, m_outCodecDecodeCtx->height);
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSubburn::InitLibAss(const std::shared_ptr<RGYFilterParamSubburn> prm) {
    //libassの初期化
    m_assLibrary = unique_ptr<ASS_Library, decltype(&ass_library_done)>(ass_library_init(), ass_library_done);
    if (!m_assLibrary) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialize libass.\n"));
        return RGY_ERR_NULL_PTR;
    }
    ass_set_message_cb(m_assLibrary.get(), ass_log, m_pLog.get());

    if (prm->subburn.fontsdir.length() > 0) {
        if (!std::filesystem::exists(std::filesystem::path(prm->subburn.fontsdir))) {
            AddMessage(RGY_LOG_WARN, _T("fontsdir=\"%s\" does not exist, ignored.\n"), prm->subburn.fontsdir.c_str());
        } else {
            std::string fontsdir;
            if (tchar_to_string(prm->subburn.fontsdir.c_str(), fontsdir, CP_UTF8) == 0) {
                AddMessage(RGY_LOG_ERROR, _T("failed to convert fontsdir=\"%s\" to UTF8.\n"), prm->subburn.fontsdir.c_str());
                return RGY_ERR_NULL_PTR;
            }
            AddMessage(RGY_LOG_DEBUG, _T("Setting fontsdir \"%s\"\n"), prm->subburn.fontsdir.c_str());
            ass_set_fonts_dir(m_assLibrary.get(), fontsdir.c_str());
        }
    }
    for (const auto& s : prm->attachmentStreams) {
        if (font_attached(s)) {
            const AVDictionaryEntry *tag = av_dict_get(s->metadata, "filename", NULL, AV_DICT_MATCH_CASE);
            if (tag) {
                AddMessage(RGY_LOG_DEBUG, _T("Loading attached font: %s\n"), char_to_tstring(tag->value).c_str());
                ass_add_font(m_assLibrary.get(), tag->value, (char *)s->codecpar->extradata, s->codecpar->extradata_size);
            } else {
                AddMessage(RGY_LOG_WARN, _T("Font attachment has no filename, ignored.\n"));
            }
        }
    }


    ass_set_extract_fonts(m_assLibrary.get(), 1);
    ass_set_style_overrides(m_assLibrary.get(), nullptr);

    m_assRenderer = unique_ptr<ASS_Renderer, decltype(&ass_renderer_done)>(ass_renderer_init(m_assLibrary.get()), ass_renderer_done);
    if (!m_assRenderer) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialize libass renderer.\n"));
        return RGY_ERR_NULL_PTR;
    }

    ass_set_use_margins(m_assRenderer.get(), 0);
    ass_set_hinting(m_assRenderer.get(), ASS_HINTING_LIGHT);
    ass_set_font_scale(m_assRenderer.get(), 1.0);
    ass_set_line_spacing(m_assRenderer.get(), 1.0);
    ass_set_shaper(m_assRenderer.get(), (ASS_ShapingLevel)prm->subburn.assShaping);

    ass_set_fonts(m_assRenderer.get(), nullptr, nullptr, 1, nullptr, 1);

    m_assTrack = unique_ptr<ASS_Track, decltype(&ass_free_track)>(ass_new_track(m_assLibrary.get()), ass_free_track);
    if (!m_assTrack) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialize libass track.\n"));
        return RGY_ERR_NULL_PTR;
    }

    if (prm->videoInfo.srcWidth <= 0 || prm->videoInfo.srcHeight <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("failed to detect frame size: %dx%d.\n"), prm->videoInfo.srcWidth, prm->videoInfo.srcHeight);
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }
    const int width = prm->videoInfo.srcWidth - prm->videoInfo.crop.e.left - prm->videoInfo.crop.e.right;
    const int height = prm->videoInfo.srcHeight - prm->videoInfo.crop.e.up - prm->videoInfo.crop.e.bottom;
    ass_set_frame_size(m_assRenderer.get(), width, height);

    const AVRational sar = { prm->videoInfo.sar[0], prm->videoInfo.sar[1] };
    double par = 1.0;
    if (sar.num * sar.den > 0) {
        par = (double)sar.num / sar.den;
    }
    ass_set_pixel_aspect(m_assRenderer.get(), par);

    if (m_outCodecDecodeCtx && m_outCodecDecodeCtx->subtitle_header && m_outCodecDecodeCtx->subtitle_header_size > 0) {
        ass_process_codec_private(m_assTrack.get(), (char *)m_outCodecDecodeCtx->subtitle_header, m_outCodecDecodeCtx->subtitle_header_size);
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSubburn::readSubFile() {
    for (auto pkt = m_poolPkt->getFree();
        av_read_frame(m_formatCtx.get(), pkt.get()) >= 0;
        pkt = m_poolPkt->getFree()) {
        if (pkt->stream_index == m_subtitleStreamIndex) {
            addStreamPacket(pkt.release());
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSubburn::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSubburn>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    m_poolPkt = prm->poolPkt;
    //パラメータチェック
    if ((sts = checkParam(prm)) != RGY_ERR_NONE) {
        return sts;
    }
    //subburnは常に元のフレームを書き換え
    if (!prm->bOutOverwrite) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid param, subburn will overwrite input frame.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    prm->frameOut = prm->frameIn;
    m_queueSubPackets.init();

    if (!m_param
        || std::dynamic_pointer_cast<RGYFilterParamSubburn>(m_param)->subburn != prm->subburn) {
        const auto options = strsprintf("-D TypePixel=%s -D TypePixel2=%s -D bit_depth=%d -D yuv420=%d",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort"  : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8 ? "ushort2" : "uchar2",
            RGY_CSP_BIT_DEPTH[prm->frameOut.csp],
            RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp] == RGY_CHROMAFMT_YUV420 ? 1 : 0
        );

        m_subburn.set(m_cl->buildResourceAsync(_T("RGY_FILTER_SUBBURN_CL"), _T("EXE_DATA"), options.c_str()));

        //字幕読み込み・デコーダの初期化
        if ((sts = initAVCodec(prm)) != RGY_ERR_NONE) {
            return sts;
        }
        if (m_subType & AV_CODEC_PROP_TEXT_SUB) {
            if ((sts = InitLibAss(prm)) != RGY_ERR_NONE) {
                return sts;
            }
        }
        m_param = prm;
        if (prm->streamIn.stream == nullptr) {
            if ((sts = readSubFile()) != RGY_ERR_NONE) {
                return sts;
            }
        }
        if (prm->subburn.scale <= 0.0f) {
            if (m_outCodecDecodeCtx->width > 0 && m_outCodecDecodeCtx->height > 0) {
                double scaleX = prm->frameOut.width / m_outCodecDecodeCtx->width;
                double scaleY = prm->frameOut.height / m_outCodecDecodeCtx->height;
                prm->subburn.scale = (float)std::sqrt(scaleX * scaleY);
                if (std::abs(prm->subburn.scale - 1.0f) <= 0.1f) {
                    prm->subburn.scale = 1.0f;
                }
            } else {
                prm->subburn.scale = 1.0f;
            }
        } else if (m_subType & AV_CODEC_PROP_TEXT_SUB) {
            AddMessage(RGY_LOG_WARN, _T("manual scaling not available for text type fonts.\n"));
            prm->subburn.scale = 1.0f;
        }
    }

    //コピーを保存
    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

int RGYFilterSubburn::targetTrackIdx() {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSubburn>(m_param);
    if (!prm) {
        return 0;
    }
    return prm->streamIn.trackId;
}

RGY_ERR RGYFilterSubburn::addStreamPacket(AVPacket *pkt) {
    m_queueSubPackets.push(pkt);
    const auto log_level = RGY_LOG_TRACE;
    if (m_pLog != nullptr && log_level >= m_pLog->getLogLevel(RGY_LOGT_VPP)) {
        auto prm = std::dynamic_pointer_cast<RGYFilterParamSubburn>(m_param);
        if (!prm) {
            AddMessage(log_level, _T("Invalid parameter type.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        const auto inputSubStream = (prm->streamIn.stream) ? prm->streamIn.stream : m_formatCtx->streams[m_subtitleStreamIndex];
        const int64_t vidInputOffsetMs = (prm->videoInputStream && prm->subburn.vid_ts_offset) ? av_rescale_q(prm->videoInputFirstKeyPts, prm->videoInputStream->time_base, { 1, 1000 }) : 0;
        const int64_t tsOffsetMs = (int64_t)(prm->subburn.ts_offset * 1000.0 + 0.5);
        const auto pktTimeMs = av_rescale_q(pkt->pts, inputSubStream->time_base, { 1, 1000 }) - vidInputOffsetMs + tsOffsetMs;
        AddMessage(log_level, _T("Add subtitle packet: %s\n"), getTimestampString(pktTimeMs, av_make_q(1, 1000)).c_str());
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterSubburn::procFrame(RGYFrameInfo *pOutputFrame, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamSubburn>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const auto inputSubStream = (prm->streamIn.stream) ? prm->streamIn.stream : m_formatCtx->streams[m_subtitleStreamIndex];
    const int64_t nFrameTimeMs = av_rescale_q(pOutputFrame->timestamp, prm->videoOutTimebase, { 1, 1000 });
    const int64_t vidInputOffsetMs = (prm->videoInputStream && prm->subburn.vid_ts_offset) ? av_rescale_q(prm->videoInputFirstKeyPts, prm->videoInputStream->time_base, { 1, 1000 }) : 0;
    const int64_t tsOffsetMs = (int64_t)(prm->subburn.ts_offset * 1000.0 + 0.5);

    AVPacket *pkt = nullptr;
    while (m_queueSubPackets.front_copy_no_lock(&pkt)) {
        const auto pktTimeMs = av_rescale_q(pkt->pts, inputSubStream->time_base, { 1, 1000 }) - vidInputOffsetMs + tsOffsetMs;
        if (!(m_subType & AV_CODEC_PROP_TEXT_SUB)) {
            //字幕パケットのptsが、フレームのptsより古ければ、処理する必要がある
            if (nFrameTimeMs < pktTimeMs) {
                //取得したパケットが未来のパケットなら無視
                break;
            }
        }
        //字幕パケットをキューから取り除く
        m_queueSubPackets.pop();

        //新たに字幕構造体を確保(これまで構築していたデータは破棄される)
        m_subData = unique_ptr<AVSubtitle, subtitle_deleter>(new AVSubtitle(), subtitle_deleter());
        if (!(m_subType & AV_CODEC_PROP_TEXT_SUB)) {
            m_subImages.clear();
        }

        //字幕パケットをデコードする
        int got_sub = 0;
        if (0 > avcodec_decode_subtitle2(m_outCodecDecodeCtx.get(), m_subData.get(), &got_sub, pkt)) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to decode subtitle.\n"));
            return RGY_ERR_NONE;
        }
        if (got_sub) {
            const int64_t nStartTime = av_rescale_q(m_subData->pts, av_make_q(1, AV_TIME_BASE), av_make_q(1, 1000)) - vidInputOffsetMs + tsOffsetMs;
            AddMessage(RGY_LOG_TRACE, _T("decoded subtitle chunk (%s - %s), Video frame (%s)"),
                getTimestampString(nStartTime, av_make_q(1, 1000)).c_str(),
                getTimestampString(nStartTime + m_subData->end_display_time, av_make_q(1, 1000)).c_str(),
                getTimestampString(nFrameTimeMs, av_make_q(1, 1000)).c_str());
        }
        if (got_sub && (m_subType & AV_CODEC_PROP_TEXT_SUB)) {
            const int64_t nStartTime = av_rescale_q(m_subData->pts, av_make_q(1, AV_TIME_BASE), av_make_q(1, 1000)) - vidInputOffsetMs + tsOffsetMs;
            const int64_t nDuration  = m_subData->end_display_time;
            for (uint32_t i = 0; i < m_subData->num_rects; i++) {
                auto *ass = m_subData->rects[i]->ass;
                if (!ass) {
                    break;
                }
                ass_process_chunk(m_assTrack.get(), ass, (int)strlen(ass), nStartTime, nDuration);
            }
        }
        m_poolPkt->returnFree(&pkt);
    }

    if (m_subType & AV_CODEC_PROP_TEXT_SUB) {
        return procFrameText(pOutputFrame, nFrameTimeMs, queue, wait_events, event);
    } else {
        if (m_subData) {
            //いまなんらかの字幕情報がデコード済みなら、その有効期限をチェックする
            const int64_t nStartTime = av_rescale_q(m_subData->pts, av_make_q(1, AV_TIME_BASE), av_make_q(1, 1000)) - vidInputOffsetMs + tsOffsetMs;
            const int64_t nDuration  = m_subData->end_display_time;
            if (nStartTime + nDuration < nFrameTimeMs) {
                //現在蓄えている字幕データを開放
                AddMessage(RGY_LOG_TRACE, _T("release subtitle chunk (%s - %s) [video frame (%s)]"),
                    getTimestampString(nStartTime, av_make_q(1, 1000)).c_str(),
                    getTimestampString(nStartTime + nDuration, av_make_q(1, 1000)).c_str(),
                    getTimestampString(nFrameTimeMs, av_make_q(1, 1000)).c_str());
                m_subData.reset();
                m_subImages.clear();
                return RGY_ERR_NONE;
            }
            AddMessage(RGY_LOG_TRACE, _T("burn subtitle into video frame (%s)"),
                getTimestampString(nFrameTimeMs, av_make_q(1, 1000)).c_str());
            return procFrameBitmap(pOutputFrame, nFrameTimeMs, prm->crop, prm->subburn.forced_subs_only, queue, wait_events, event);
        }
    }
    return RGY_ERR_NONE;
}


RGY_ERR RGYFilterSubburn::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("ppOutputFrames[0] must be set.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //if (interlaced(*pInputFrame)) {
    //    return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], cudaStreamDefault);
    //}
    if (!m_subburn.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_SUBBURN_CL(m_subburn)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    const auto memcpyKind = getMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != RGYCLMemcpyD2D) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if ((sts = procFrame(ppOutputFrames[0], queue, wait_events, event)) != RGY_ERR_NONE) {
        return sts;
    }

    return sts;
}

void RGYFilterSubburn::close() {
    m_subburn.clear();
    m_assTrack.reset();
    m_assRenderer.reset();
    m_assLibrary.reset();
    m_queueSubPackets.clear();
    m_subData.reset();
    m_outCodecDecodeCtx.reset();
    m_formatCtx.reset();
    m_subType = 0;
    m_frameBuf.clear();
    m_cl.reset();
}

#endif //#if ENABLE_AVSW_READER
