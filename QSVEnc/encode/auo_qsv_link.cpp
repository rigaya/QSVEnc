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

#include <Windows.h>
#include <Process.h>
#include <Math.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")

#include "output.h"
#include "vphelp_client.h"

#pragma warning( push )
#pragma warning( disable: 4127 )
#include "afs_client.h"
#pragma warning( pop )

#include "auo_util.h"
#include "auo_qsv_link.h"
#include "auo_video.h"
#include "auo_encode.h"
#include "auo_audio_parallel.h"
#include "auo_frm.h"
#include "auo_error.h"
#include "convert.h"

AUO_RESULT aud_parallel_task(const OUTPUT_INFO *oip, PRM_ENC *pe);

static int calc_input_frame_size(int width, int height, int color_format) {
    width = (color_format == CF_RGB) ? (width+3) & ~3 : (width+1) & ~1;
    return width * height * COLORFORMATS[color_format].size;
}

BOOL setup_afsvideo(const OUTPUT_INFO *oip, const SYSTEM_DATA *sys_dat, CONF_GUIEX *conf, PRM_ENC *pe) {
    //すでに初期化してある または 必要ない
    if (pe->afs_init || pe->video_out_type == VIDEO_OUTPUT_DISABLED || !conf->vid.afs)
        return TRUE;

    const int color_format = CF_YUY2;
    const int frame_size = calc_input_frame_size(oip->w, oip->h, color_format);
    //Aviutl(自動フィールドシフト)からの映像入力
    if (afs_vbuf_setup((OUTPUT_INFO *)oip, conf->vid.afs, frame_size, COLORFORMATS[color_format].FOURCC)) {
        pe->afs_init = TRUE;
        return TRUE;
    } else if (conf->vid.afs && sys_dat->exstg->s_local.auto_afs_disable) {
        afs_vbuf_release(); //一度解放
        warning_auto_afs_disable();
        conf->vid.afs = FALSE;
        //再度使用するmuxerをチェックする
        pe->muxer_to_be_used = check_muxer_to_be_used(conf, sys_dat, pe->temp_filename, pe->video_out_type, (oip->flag & OUTPUT_INFO_FLAG_AUDIO) != 0);
        return TRUE;
    }
    //エラー
    error_afs_setup(conf->vid.afs, sys_dat->exstg->s_local.auto_afs_disable);
    return FALSE;
}

void close_afsvideo(PRM_ENC *pe) {
    if (!pe->afs_init || pe->video_out_type == VIDEO_OUTPUT_DISABLED)
        return;

    afs_vbuf_release();

    pe->afs_init = FALSE;
}

AUO_YUVReader::AUO_YUVReader() :
    oip(nullptr),
    conf(nullptr),
    pe(nullptr),
    jitter(nullptr),
    m_iFrame(0),
    m_pause(FALSE) {
}

#pragma warning(push)
#pragma warning(disable: 4100)
RGY_ERR AUO_YUVReader::Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const void *prm) {
    auto *info = (const InputInfoAuo *)(prm);
    memcpy(&m_inputVideoInfo, pInputInfo, sizeof(m_inputVideoInfo));

    oip = info->oip;
    conf = info->conf;
    pe = info->pe;
    jitter = info->jitter;

    m_inputVideoInfo.frames = oip->n;
    m_inputVideoInfo.srcWidth = oip->w;
    m_inputVideoInfo.srcHeight = oip->h;
    m_inputVideoInfo.fpsN = oip->rate;
    m_inputVideoInfo.fpsD = oip->scale;
    m_inputVideoInfo.srcPitch = oip->w;
    rgy_reduce(m_inputVideoInfo.fpsN, m_inputVideoInfo.fpsD);

    const RGY_CSP input_csp = (m_inputVideoInfo.csp == RGY_CSP_YUV444 || m_inputVideoInfo.csp == RGY_CSP_P010 || m_inputVideoInfo.csp == RGY_CSP_YUV444_16) ? RGY_CSP_YC48 : RGY_CSP_YUY2;
    m_sConvert = get_convert_csp_func(input_csp, m_inputVideoInfo.csp, false);
    m_inputVideoInfo.shift = (m_inputVideoInfo.csp == RGY_CSP_P010 && m_inputVideoInfo.shift) ? m_inputVideoInfo.shift : 0;

    if (m_sConvert == nullptr) {
        AddMessage(RGY_LOG_ERROR, "invalid colorformat.\n");
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }

    if (conf->vid.afs) {
        if (!setup_afsvideo(oip, info->sys_dat, conf, pe)) {
            AddMessage(RGY_LOG_ERROR, "自動フィールドシフトの初期化に失敗しました。\n");
            return RGY_ERR_UNKNOWN;
        }
    }
    CreateInputInfo(_T("auo"), RGY_CSP_NAMES[m_sConvert->csp_from], RGY_CSP_NAMES[m_sConvert->csp_to], get_simd_str(m_sConvert->simd), &m_inputVideoInfo);
    AddMessage(RGY_LOG_DEBUG, m_strInputInfo);
    *pInputInfo = m_inputVideoInfo;
    return RGY_ERR_NONE;
}
#pragma warning(pop)

AUO_YUVReader::~AUO_YUVReader() {
    Close();
}

void AUO_YUVReader::Close() {
    disable_enc_control();
    oip = nullptr;
    conf = nullptr;
    pe = nullptr;
    jitter = nullptr;
    m_iFrame = 0;
    m_pause = FALSE;
    m_pEncSatusInfo.reset();
}

RGY_ERR AUO_YUVReader::LoadNextFrame(RGYFrame *pSurface) {
    const int total_frames = oip->n;

    if (m_iFrame >= total_frames) {
        oip->func_rest_time_disp(m_iFrame-1, total_frames);
        release_audio_parallel_events(pe);
        return RGY_ERR_MORE_DATA;
    }
    
    void *frame = nullptr;
    if (pe->afs_init) {
        BOOL drop = FALSE;
        for ( ; ; ) {
            if ((frame = afs_get_video((OUTPUT_INFO *)oip, m_iFrame, &drop, &jitter[m_iFrame + 1])) == NULL) {
                error_afs_get_frame();
                return RGY_ERR_MORE_DATA;
            }
            if (!drop)
                break;
            jitter[m_iFrame] = DROP_FRAME_FLAG;
            pe->drop_count++;
            m_pEncSatusInfo->m_sData.frameDrop++;
            m_iFrame++;
            if (m_iFrame >= total_frames) {
                oip->func_rest_time_disp(m_iFrame, total_frames);
                release_audio_parallel_events(pe);
                return RGY_ERR_MORE_DATA;
            }
        }
    } else {
        if ((frame = oip->func_get_video_ex(m_iFrame, COLORFORMATS[m_sConvert->csp_from == RGY_CSP_YC48 ? CF_YC48 : CF_YUY2].FOURCC)) == NULL) {
            error_afs_get_frame();
            return RGY_ERR_MORE_DATA;
        }
    }

    void *dst_array[3];
    pSurface->ptrArray(dst_array, m_sConvert->csp_to == RGY_CSP_RGB24 || m_sConvert->csp_to == RGY_CSP_RGB32);
    int src_pitch = m_inputVideoInfo.srcPitch * ((m_sConvert->csp_from == RGY_CSP_YC48) ? 6 : 2); //high444出力ならAviutlからYC48をもらう
    m_sConvert->func[(m_inputVideoInfo.picstruct & RGY_PICSTRUCT_INTERLACED) ? 1 : 0](
        dst_array, (const void **)&frame, m_inputVideoInfo.srcWidth, src_pitch, 0,
        pSurface->pitch(), m_inputVideoInfo.srcHeight, m_inputVideoInfo.srcHeight, m_inputVideoInfo.crop.c);

    m_iFrame++;
    if (!(m_pEncSatusInfo->m_sData.frameIn & 7))
        aud_parallel_task(oip, pe);

    m_pEncSatusInfo->m_sData.frameIn++;
    return m_pEncSatusInfo->UpdateDisplay();    
}

AUO_EncodeStatusInfo::AUO_EncodeStatusInfo() {
    m_tmLastLogUpdate = std::chrono::system_clock::now();
    log_process_events();
}

AUO_EncodeStatusInfo::~AUO_EncodeStatusInfo() {     }

#pragma warning(push)
#pragma warning(disable: 4100)
void AUO_EncodeStatusInfo::SetPrivData(void *pPrivateData) {
    m_auoData = *(InputInfoAuo *)pPrivateData;
    enable_enc_control(&m_pause, m_auoData.pe->afs_init, FALSE, timeGetTime(), m_auoData.oip->n);
};
#pragma warning(pop)

void AUO_EncodeStatusInfo::WriteLine(const TCHAR *mes) {
    const char *HEADER = "qsv [info]: ";
    int buf_len = strlen(mes) + 1 + strlen(HEADER);
    char *buf = (char *)calloc(buf_len, sizeof(buf[0]));
    if (buf) {
        memcpy(buf, HEADER, strlen(HEADER));
        memcpy(buf + strlen(HEADER), mes, strlen(mes) + 1);
        write_log_line(LOG_INFO, buf);
        free(buf);
    }
}

#pragma warning(push)
#pragma warning(disable: 4100)
void AUO_EncodeStatusInfo::UpdateDisplay(const TCHAR *mes, double progressPercent) {
    set_log_title_and_progress(mes, progressPercent * 0.01);
    m_auoData.oip->func_rest_time_disp(m_sData.frameOut + m_sData.frameDrop, m_sData.frameTotal);
    m_auoData.oip->func_update_preview();
}
#pragma warning(pop)

RGY_ERR AUO_EncodeStatusInfo::UpdateDisplay(double progressPercent) {
    auto tm = std::chrono::system_clock::now();

    if (m_auoData.oip->func_is_abort())
        return RGY_ERR_ABORTED;

    if (duration_cast<std::chrono::milliseconds>(tm - m_tmLastLogUpdate).count() >= LOG_UPDATE_INTERVAL) {
        log_process_events();

        while (m_pause) {
            Sleep(LOG_UPDATE_INTERVAL);
            if (m_auoData.oip->func_is_abort())
                return RGY_ERR_ABORTED;
            log_process_events();
        }
        m_tmLastLogUpdate = tm;
    }
    return EncodeStatus::UpdateDisplay(progressPercent);
}
