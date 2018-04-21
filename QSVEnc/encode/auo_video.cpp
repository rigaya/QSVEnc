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
#pragma comment(lib, "user32.lib") //WaitforInputIdle
#include <process.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include <vector>

#include "output.h"
#include "vphelp_client.h"

#pragma warning( push )
#pragma warning( disable: 4127 )
#include "afs_client.h"
#pragma warning( pop )

#include "convert.h"

#include "auo.h"
#include "auo_frm.h"
#include "auo_pipe.h"
#include "auo_error.h"
#include "auo_conf.h"
#include "auo_util.h"
#include "auo_system.h"
#include "auo_version.h"
#include "qsv_cmd.h"
#include "rgy_avutil.h"

#include "auo_encode.h"
#include "auo_convert.h"
#include "auo_video.h"
#include "auo_audio_parallel.h"

typedef struct video_output_thread_t {
    CONVERT_CF_DATA *pixel_data;
    FILE *f_out;
    BOOL abort;
    HANDLE thread;
    HANDLE he_out_start;
    HANDLE he_out_fin;
    int repeat;
} video_output_thread_t;

static int getLwiRealPath(char *path, size_t size) {
    int ret = 1;
    FILE *fp = fopen(path, "rb");
    if (fp) {
        char buffer[2048] = { 0 };
        while (nullptr != fgets(buffer, _countof(buffer), fp)) {
            static const char *TARGET = "InputFilePath";
            auto ptr = strstr(buffer, TARGET);
            auto qtr = strrstr(buffer, TARGET);
            if (ptr != nullptr && qtr != nullptr) {
                ptr = strchr(ptr + strlen(TARGET), '>');
                while (*qtr != '<') {
                    qtr--;
                    if (ptr >= qtr) {
                        qtr = nullptr;
                        break;
                    }
                }
                if (ptr != nullptr && qtr != nullptr) {
                    ptr++;
                    *qtr = '\0';
                    strcpy_s(path, size, trim(ptr).c_str());
                    ret = 0;
                    break;
                }
            }
        }
        fclose(fp);
    }
    return ret;
}

int get_aviutl_color_format(int use_highbit, RGY_CSP csp) {
    //Aviutlからの入力に使用するフォーマット
    RGY_CHROMAFMT chromafmt = RGY_CSP_CHROMA_FORMAT[csp];

    if (use_highbit) {
        return CF_YC48;
    }

    switch (chromafmt) {
    case RGY_CHROMAFMT_YUV444:
        return CF_YC48;
    case RGY_CHROMAFMT_YUV420:
    case RGY_CHROMAFMT_YUV422:
    default:
        return CF_YUY2;
    }
}

void get_csp_and_bitdepth(bool& use_highbit, RGY_CSP& csp, const CONF_GUIEX *conf) {
    ParseCmdError err;
    sInputParams enc_prm;
    parse_cmd(&enc_prm, conf->qsv.cmd, err);

    use_highbit = enc_prm.CodecId == MFX_CODEC_HEVC && enc_prm.CodecProfile == MFX_PROFILE_HEVC_MAIN10;
    csp = (use_highbit) ? RGY_CSP_P010 : RGY_CSP_NV12;
}

static int calc_input_frame_size(int width, int height, int color_format) {
    width = (color_format == CF_RGB) ? (width+3) & ~3 : (width+1) & ~1;
    return width * height * COLORFORMATS[color_format].size;
}

BOOL setup_afsvideo(const OUTPUT_INFO *oip, const SYSTEM_DATA *sys_dat, CONF_GUIEX *conf, PRM_ENC *pe) {
    //すでに初期化してある または 必要ない
    if (pe->afs_init || pe->video_out_type == VIDEO_OUTPUT_DISABLED || !conf->vid.afs)
        return TRUE;

    bool use_highbit;
    RGY_CSP csp;
    get_csp_and_bitdepth(use_highbit, csp, conf);
    const int color_format = get_aviutl_color_format(use_highbit, csp);
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

DWORD tcfile_out(int *jitter, int frame_n, double fps, BOOL afs, const PRM_ENC *pe) {
    DWORD ret = AUO_RESULT_SUCCESS;
    char auotcfile[MAX_PATH_LEN];
    FILE *tcfile = NULL;

    if (afs)
        fps *= 4; //afsなら4倍精度
    double tm_multi = 1000.0 / fps;

    //ファイル名作成
    apply_appendix(auotcfile, sizeof(auotcfile), pe->temp_filename, pe->append.tc);

    if (NULL != fopen_s(&tcfile, auotcfile, "wb")) {
        ret |= AUO_RESULT_ERROR; warning_auo_tcfile_failed();
    } else {
        fprintf(tcfile, "# timecode format v2\r\n");
        if (afs) {
            int time_additional_frame = 0;
            //オーディオディレイカットのために映像フレームを追加したらその分を考慮したタイムコードを出力する
            if (pe->delay_cut_additional_vframe) {
                //24fpsと30fpsどちらに近いかを考慮する
                const int multi_for_additional_vframe = 4 + !!fps_after_afs_is_24fps(frame_n, pe);
                for (int i = 0; i < pe->delay_cut_additional_vframe; i++)
                    fprintf(tcfile, "%.6lf\r\n", i * multi_for_additional_vframe * tm_multi);

                time_additional_frame = pe->delay_cut_additional_vframe * multi_for_additional_vframe;
            }
            for (int i = 0; i < frame_n; i++)
                if (jitter[i] != DROP_FRAME_FLAG)
                    fprintf(tcfile, "%.6lf\r\n", (i * 4 + jitter[i] + time_additional_frame) * tm_multi);
        } else {
            frame_n += pe->delay_cut_additional_vframe;
            for (int i = 0; i < frame_n; i++)
                fprintf(tcfile, "%.6lf\r\n", i * tm_multi);
        }
        fclose(tcfile);
    }
    return ret;
}

static void build_full_cmd(char *cmd, size_t nSize, const CONF_GUIEX *conf, const sInputParams *encPrm, const OUTPUT_INFO *oip, const PRM_ENC *pe, const SYSTEM_DATA *sys_dat, const char *input) {
    //cliモードでない
    //自動設定の適用
    //apply_guiEx_auto_settings(&prm.x264, oip->w, oip->h, oip->rate, oip->scale, sys_dat->exstg->s_local.auto_ref_limit_by_level);
    //GUI部のコマンドライン生成
    strcpy_s(cmd, nSize, gen_cmd(encPrm, false).c_str());
    //キーフレーム検出を行い、そのQPファイルが存在し、かつ--qpfileの指定がなければ、それをqpfileで読み込む
    if (!conf->oth.link_prm.active) {
        //出力ファイル
        sprintf_s(cmd + strlen(cmd), nSize - strlen(cmd), " -o \"%s\"", pe->temp_filename);
        //入力
        sprintf_s(cmd + strlen(cmd), nSize - strlen(cmd), " --y4m -i -");
    }
}

static void set_pixel_data(CONVERT_CF_DATA *pixel_data, const CONF_GUIEX *conf, int w, int h, bool output_highbit_depth, RGY_CSP rgy_output_csp) {
    const int byte_per_pixel = (output_highbit_depth) ? sizeof(short) : sizeof(BYTE);
    ZeroMemory(pixel_data, sizeof(CONVERT_CF_DATA));
    switch (rgy_output_csp) {
    case OUT_CSP_YUY2: //yuy2 (YUV422)
        pixel_data->count = 1;
        pixel_data->size[0] = w * h * byte_per_pixel * 2;
        break;
    case RGY_CSP_YUV444:
    case RGY_CSP_YUV444_16: //i444 (YUV444 planar)
        pixel_data->count = 3;
        pixel_data->size[0] = w * h * byte_per_pixel;
        pixel_data->size[1] = pixel_data->size[0];
        pixel_data->size[2] = pixel_data->size[0];
        break;
    case RGY_CSP_NV12: //nv12 (YUV420)
    case RGY_CSP_P010: //nv12 (YUV420)
    default:
        pixel_data->count = 2;
        pixel_data->size[0] = w * h * byte_per_pixel;
        pixel_data->size[1] = pixel_data->size[0] / 2;
        break;
    }
    //サイズの総和計算
    for (int i = 0; i < pixel_data->count; i++)
        pixel_data->total_size += pixel_data->size[i];
}

//並列処理時に音声データを取得する
AUO_RESULT aud_parallel_task(const OUTPUT_INFO *oip, PRM_ENC *pe) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    AUD_PARALLEL_ENC *aud_p = &pe->aud_parallel; //長いんで省略したいだけ
    if (aud_p->th_aud) {
        //---   排他ブロック 開始  ---> 音声スレッドが止まっていなければならない
        if_valid_wait_for_single_object(aud_p->he_vid_start, INFINITE);
        if (aud_p->he_vid_start && aud_p->get_length) {
            DWORD required_buf_size = aud_p->get_length * (DWORD)oip->audio_size;
            if (aud_p->buf_max_size < required_buf_size) {
                //メモリ不足なら再確保
                if (aud_p->buffer) free(aud_p->buffer);
                aud_p->buf_max_size = required_buf_size;
                if (NULL == (aud_p->buffer = malloc(aud_p->buf_max_size)))
                    aud_p->buf_max_size = 0; //ここのmallocエラーは次の分岐でAUO_RESULT_ERRORに設定
            }
            void *data_ptr = NULL;
            if (NULL == aud_p->buffer || 
                NULL == (data_ptr = oip->func_get_audio(aud_p->start, aud_p->get_length, &aud_p->get_length))) {
                ret = AUO_RESULT_ERROR; //mallocエラーかget_audioのエラー
            } else {
                //自前のバッファにコピーしてdata_ptrが破棄されても良いようにする
                memcpy(aud_p->buffer, data_ptr, aud_p->get_length * oip->audio_size);
            }
            //すでにTRUEなら変更しないようにする
            aud_p->abort |= oip->func_is_abort();
        }
        flush_audio_log();
        if_valid_set_event(aud_p->he_aud_start);
        //---   排他ブロック 終了  ---> 音声スレッドを開始
    }
    return ret;
}

//音声処理をどんどん回して終了させる
static AUO_RESULT finish_aud_parallel_task(const OUTPUT_INFO *oip, PRM_ENC *pe, AUO_RESULT vid_ret) {
    //エラーが発生していたら音声出力ループをとめる
    pe->aud_parallel.abort |= (vid_ret != AUO_RESULT_SUCCESS);
    if (pe->aud_parallel.th_aud) {
        write_log_auo_line(LOG_INFO, "音声処理の終了を待機しています...");
        set_window_title("音声処理の終了を待機しています...", PROGRESSBAR_MARQUEE);
        while (pe->aud_parallel.he_vid_start)
            vid_ret |= aud_parallel_task(oip, pe);
        set_window_title(AUO_FULL_NAME, PROGRESSBAR_DISABLED);
    }
    return vid_ret;
}

//並列処理スレッドの終了を待ち、終了コードを回収する
static AUO_RESULT exit_audio_parallel_control(const OUTPUT_INFO *oip, PRM_ENC *pe, AUO_RESULT vid_ret) {
    vid_ret |= finish_aud_parallel_task(oip, pe, vid_ret); //wav出力を完了させる
    release_audio_parallel_events(pe);
    if (pe->aud_parallel.buffer) free(pe->aud_parallel.buffer);
    if (pe->aud_parallel.th_aud) {
        //音声エンコードを完了させる
        //2passエンコードとかだと音声エンコーダの終了を待機する必要あり
        BOOL wait_for_audio = FALSE;
        while (WaitForSingleObject(pe->aud_parallel.th_aud, LOG_UPDATE_INTERVAL) == WAIT_TIMEOUT) {
            if (!wait_for_audio) {
                set_window_title("音声処理の終了を待機しています...", PROGRESSBAR_MARQUEE);
                wait_for_audio = !wait_for_audio;
            }
            pe->aud_parallel.abort |= oip->func_is_abort();
            log_process_events();
        }
        flush_audio_log();
        if (wait_for_audio)
            set_window_title(AUO_FULL_NAME, PROGRESSBAR_DISABLED);

        DWORD exit_code = 0;
        //GetExitCodeThreadの返り値がNULLならエラー
        vid_ret |= (NULL == GetExitCodeThread(pe->aud_parallel.th_aud, &exit_code)) ? AUO_RESULT_ERROR : exit_code;
        CloseHandle(pe->aud_parallel.th_aud);
    }
    //初期化 (重要!!!)
    ZeroMemory(&pe->aud_parallel, sizeof(pe->aud_parallel));
    return vid_ret;
}

static void write_y4m_header(FILE *fp, const OUTPUT_INFO *oip, RGY_CSP rgy_output_csp) {
    const char *y4m_csp = nullptr;
    switch (rgy_output_csp) {
    case RGY_CSP_YUV444:    y4m_csp = "444"; break;
    case RGY_CSP_YUV444_16: y4m_csp = "444p16"; break;
    case RGY_CSP_P010:      y4m_csp = "p010"; break;
    case RGY_CSP_NV12:
    default:                y4m_csp = "nv12"; break;
    }
    fprintf(fp, "YUV4MPEG2 W%d H%d F%d:%d C%s\n", oip->w, oip->h, oip->rate, oip->scale, y4m_csp);
}

//auo_pipe.cppのread_from_pipeの特別版
static int ReadLogEnc(PIPE_SET *pipes, int total_drop, int current_frames) {
    DWORD pipe_read = 0;
    if (!PeekNamedPipe(pipes->stdErr.h_read, NULL, 0, NULL, &pipe_read, NULL))
        return -1;
    if (pipe_read) {
        ReadFile(pipes->stdErr.h_read, pipes->read_buf + pipes->buf_len, sizeof(pipes->read_buf) - pipes->buf_len - 1, &pipe_read, NULL);
        pipes->buf_len += pipe_read;
        write_log_enc_mes(pipes->read_buf, &pipes->buf_len, total_drop, current_frames, NULL);
    } else {
        log_process_events();
    }
    return pipe_read;
}

static unsigned __stdcall video_output_thread_func(void *prm) {
    video_output_thread_t *thread_data = reinterpret_cast<video_output_thread_t *>(prm);
    CONVERT_CF_DATA *pixel_data = thread_data->pixel_data;
    WaitForSingleObject(thread_data->he_out_start, INFINITE);
    while (false == thread_data->abort) {
        const char *FRAME_HEADER = "FRAME\n";
        _fwrite_nolock(FRAME_HEADER, 1, strlen(FRAME_HEADER), thread_data->f_out);
        //映像データをパイプに
        for (int i = 0; i < 1 + thread_data->repeat; i++)
            for (int j = 0; j < pixel_data->count; j++)
                _fwrite_nolock((void *)pixel_data->data[j], 1, pixel_data->size[j], thread_data->f_out);

        thread_data->repeat = 0;
        SetEvent(thread_data->he_out_fin);
        WaitForSingleObject(thread_data->he_out_start, INFINITE);
    }
    return 0;
}

static int video_output_create_thread(video_output_thread_t *thread_data, CONVERT_CF_DATA *pixel_data, FILE *pipe_stdin) {
    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    thread_data->abort = false;
    thread_data->pixel_data = pixel_data;
    thread_data->f_out = pipe_stdin;
    if (NULL == (thread_data->he_out_start = (HANDLE)CreateEvent(NULL, false, false, NULL))
        || NULL == (thread_data->he_out_fin   = (HANDLE)CreateEvent(NULL, false, true, NULL))
        || NULL == (thread_data->thread       = (HANDLE)_beginthreadex(NULL, 0, video_output_thread_func, thread_data, 0, NULL))) {
        ret = AUO_RESULT_ERROR;
    }
    return ret;
}

static void video_output_close_thread(video_output_thread_t *thread_data, AUO_RESULT ret) {
    if (thread_data->thread) {
        if (!ret)
            while (WAIT_TIMEOUT == WaitForSingleObject(thread_data->he_out_fin, LOG_UPDATE_INTERVAL))
                log_process_events();
        thread_data->abort = true;
        SetEvent(thread_data->he_out_start);
        WaitForSingleObject(thread_data->thread, INFINITE);
        CloseHandle(thread_data->thread);
        CloseHandle(thread_data->he_out_start);
        CloseHandle(thread_data->he_out_fin);
    }
    memset(thread_data, 0, sizeof(thread_data[0]));
}

struct AVQSV_PARM {
    int nSubtitleCopyAll;
    sAudioSelect audioSelect;
    char audioCodec[128];
    std::vector<sAudioSelect *> audioSelectList;
};

void init_avqsv_prm(AVQSV_PARM *avqsv_prm) {
    avqsv_prm->nSubtitleCopyAll = 0;
    memset(&avqsv_prm->audioSelect, 0, sizeof(avqsv_prm->audioSelect));
    memset(&avqsv_prm->audioCodec,  0, sizeof(avqsv_prm->audioCodec));
    avqsv_prm->audioSelectList.clear();
}


static void set_conf_qsvp_avqsv_prm(sInputParams *enc_prm, const CONF_GUIEX *conf, const PRM_ENC *pe, AVQSV_PARM *avqsv_prm) {
    init_avqsv_prm(avqsv_prm);
#if ENABLE_AVSW_READER
    enc_prm->nInputFmt = RGY_INPUT_FMT_AVHW;

    avqsv_prm->audioSelectList.push_back(&avqsv_prm->audioSelect);
    switch (conf->aud_avqsv.encoder) {
    case QSV_AUD_ENC_NONE:
        break;
    case QSV_AUD_ENC_COPY:
        enc_prm->nAVMux |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        avqsv_prm->audioSelect.nAudioSelect = 1;
        avqsv_prm->audioSelect.pAVAudioEncodeCodec = avqsv_prm->audioCodec;
        strcpy_s(avqsv_prm->audioCodec, RGY_AVCODEC_COPY);
        enc_prm->ppAudioSelectList = avqsv_prm->audioSelectList.data();
        enc_prm->nAudioSelectCount = 1;
        break;
    default:
        enc_prm->nAVMux |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        avqsv_prm->audioSelect.nAudioSelect = 1;
        avqsv_prm->audioSelect.pAVAudioEncodeCodec = avqsv_prm->audioCodec;
        strcpy_s(avqsv_prm->audioCodec, list_avqsv_aud_encoder[get_cx_index(list_avqsv_aud_encoder, conf->aud_avqsv.encoder)].desc);
        avqsv_prm->audioSelect.nAVAudioEncodeBitrate = conf->aud_avqsv.bitrate;
        enc_prm->ppAudioSelectList = avqsv_prm->audioSelectList.data();
        enc_prm->nAudioSelectCount = 1;
        break;
    }
    enc_prm->nTrimCount = (uint16_t)conf->oth.link_prm.trim_count;
    enc_prm->pTrimList = (enc_prm->nTrimCount) ? (sTrim *)conf->oth.link_prm.trim : nullptr;

    if (enc_prm->nSubtitleSelectCount) {
        enc_prm->nSubtitleSelectCount = 1;
        enc_prm->pSubtitleSelect = &avqsv_prm->nSubtitleCopyAll;
    }
    strcpy_s(enc_prm->strDstFile, pe->temp_filename);

    if ((enc_prm->nPicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF)) == FALSE) {
        enc_prm->vpp.nDeinterlace = MFX_DEINTERLACE_NONE;
    }
    if (check_ext(enc_prm->strSrcFile, { ".lwi" })) {
        getLwiRealPath(enc_prm->strSrcFile, sizeof(enc_prm->strSrcFile));
    }
#endif //#if ENABLE_AVSW_READER
}

#pragma warning( push )
#pragma warning( disable: 4100 )
static DWORD video_output_inside(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat) {
    //動画エンコードの必要がなければ終了
    if (pe->video_out_type == VIDEO_OUTPUT_DISABLED)
        return AUO_RESULT_SUCCESS;

    ParseCmdError err;
    sInputParams enc_prm;
    init_qsvp_prm(&enc_prm);
    parse_cmd(&enc_prm, conf->qsv.cmd, err);
    enc_prm.nBluray += (enc_prm.nBluray == 1 && sys_dat->exstg->s_local.force_bluray);
    enc_prm.bDisableTimerPeriodTuning = !sys_dat->exstg->s_local.timer_period_tuning;
    enc_prm.nLogLevel = (mfxI16)sys_dat->exstg->s_log.log_level;
    enc_prm.nHeight = 0;
    enc_prm.nWidth = 0;
    enc_prm.nFPSRate = 0;
    enc_prm.nFPSScale = 0;
    if (!enc_prm.vpp.bUseResize) {
        enc_prm.nDstWidth = 0;
        enc_prm.nDstHeight = 0;
    }

#if ENABLE_AUO_LINK
    AVQSV_PARM avqsv_prm = { 0 };
    if (conf->oth.link_prm.active) {
        set_conf_qsvp_avqsv_prm(&enc_prm, conf, pe, &avqsv_prm);
    }
#endif //#if ENABLE_AUO_LINK

    char exe_cmd[MAX_CMD_LEN] = { 0 };
    char exe_args[MAX_CMD_LEN] = { 0 };
    char exe_dir[MAX_PATH_LEN] = { 0 };

    AUO_RESULT ret = AUO_RESULT_SUCCESS;
    PIPE_SET pipes = { 0 };
    PROCESS_INFORMATION pi_enc = { 0 };
    const BOOL afs = conf->vid.afs != 0;

    //プロセス用情報準備
    if (!PathFileExists(sys_dat->exstg->s_vid.fullpath)) {
        ret |= AUO_RESULT_ERROR; error_no_exe_file(ENCODER_NAME, sys_dat->exstg->s_vid.fullpath);
        return ret;
    }
    PathGetDirectory(exe_dir, _countof(exe_dir), sys_dat->exstg->s_vid.fullpath);

    const int output_csp = OUT_CSP_NV12;
    RGY_CSP rgy_output_csp;
    bool output_highbit_depth;
    get_csp_and_bitdepth(output_highbit_depth, rgy_output_csp, conf);
    const bool interlaced = ((enc_prm.nPicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF))) != 0;

    CONVERT_CF_DATA pixel_data ={ 0 };
    set_pixel_data(&pixel_data, conf, oip->w, oip->h, output_highbit_depth, rgy_output_csp);

    //YUY2/YC48->NV12/YUV444, RGBコピー用関数
    const int input_csp_idx = get_aviutl_color_format(output_highbit_depth, rgy_output_csp);
    const func_convert_frame convert_frame = get_convert_func(oip->w, input_csp_idx, (output_highbit_depth) ? 16 : 8, interlaced, output_csp);
    if (convert_frame == NULL) {
        ret |= AUO_RESULT_ERROR; error_select_convert_func(oip->w, oip->h, output_highbit_depth, interlaced, output_csp);
        return ret;
    }
    //映像バッファ用メモリ確保
    if (!malloc_pixel_data(&pixel_data, oip->w, oip->h, output_csp, (output_highbit_depth) ? 16 : 8)) {
        ret |= AUO_RESULT_ERROR; error_malloc_pixel_data();
        return ret;
    }

    //コマンドライン生成
    build_full_cmd(exe_cmd, _countof(exe_cmd), conf, &enc_prm, oip, pe, sys_dat, PIPE_FN);
    write_log_auo_line(LOG_INFO, "QSVEncC options...");
    write_args(exe_cmd);
    sprintf_s(exe_args, _countof(exe_args), "\"%s\" %s", sys_dat->exstg->s_vid.fullpath, exe_cmd);
    remove(pe->temp_filename); //ファイルサイズチェックの時に旧ファイルを参照してしまうのを回避

    //パイプの設定
    pipes.stdIn.mode = AUO_PIPE_ENABLE;
    pipes.stdErr.mode = AUO_PIPE_ENABLE;
    pipes.stdIn.bufferSize = pixel_data.total_size * 2;

    set_window_title("QSVEnc エンコード", PROGRESSBAR_CONTINUOUS);
    log_process_events();

    video_output_thread_t thread_data = { 0 };

    int *jitter = NULL;
    int rp_ret = 0;

    if (conf->vid.afs && conf->oth.link_prm.active) {
        ret |= AUO_RESULT_ERROR; error_afs_auo_link();
    } else if (conf->vid.afs && interlaced) {
        ret |= AUO_RESULT_ERROR; error_afs_interlace_stg();
    } else if (conf->vid.afs && NULL == (jitter = (int *)calloc(oip->n + 1, sizeof(int)))) {
        ret |= AUO_RESULT_ERROR; error_malloc_tc();
    //Aviutl(afs)からのフレーム読み込み
    } else if (!setup_afsvideo(oip, sys_dat, conf, pe)) {
        ret |= AUO_RESULT_ERROR; //Aviutl(afs)からのフレーム読み込みに失敗
    //x264プロセス開始
    } else if ((rp_ret = RunProcess(exe_args, exe_dir, &pi_enc, &pipes, GetPriorityClass(pe->h_p_aviutl), TRUE, FALSE)) != RP_SUCCESS) {
        ret |= AUO_RESULT_ERROR; error_run_process("QSVEncC", rp_ret);
        //書き込みスレッドを開始
    } else if (!conf->oth.link_prm.active && video_output_create_thread(&thread_data, &pixel_data, pipes.f_stdin)) {
        ret |= AUO_RESULT_ERROR; error_video_output_thread_start();
    } else if (conf->oth.link_prm.active) {
        //auo linkモード

        //Aviutlの時間を取得
        PROCESS_TIME time_aviutl;
        GetProcessTime(pe->h_p_aviutl, &time_aviutl);

        DWORD tm_vid_enc_start = timeGetTime();

        while (ret == AUO_RESULT_SUCCESS) {
            if (read_from_pipe(&pipes, TRUE) < 0) {
                write_log_exe_mes(pipes.read_buf, &pipes.buf_len, "QSVEncC", NULL);
                break;
            }
            Sleep(LOG_UPDATE_INTERVAL);
            ret |= (oip->func_is_abort()) ? AUO_RESULT_ABORT : AUO_RESULT_SUCCESS;
            log_process_events();
        }

        DWORD tm_vid_enc_fin = timeGetTime();
        write_log_auo_line_fmt(LOG_INFO, "CPU使用率: Aviutl: %.2f%% / QSVEnc: %.2f%%", GetProcessAvgCPUUsage(pe->h_p_aviutl, &time_aviutl), GetProcessAvgCPUUsage(pi_enc.hProcess));
        write_log_auo_enc_time("QSVEncエンコード時間", tm_vid_enc_fin - tm_vid_enc_start);
    } else {
        //全て正常
        int i = 0;
        void *frame = NULL;
        int *next_jitter = NULL;
        bool enc_pause = false;
        BOOL copy_frame = false, drop = false;
        const DWORD aviutl_color_fmt = COLORFORMATS[get_aviutl_color_format(output_highbit_depth, rgy_output_csp)].FOURCC;

        //Aviutlの時間を取得
        PROCESS_TIME time_aviutl;
        GetProcessTime(pe->h_p_aviutl, &time_aviutl);

        //x264が待機に入るまでこちらも待機
        while (WaitForInputIdle(pi_enc.hProcess, LOG_UPDATE_INTERVAL) == WAIT_TIMEOUT) {
            ReadLogEnc(&pipes, pe->drop_count, i);
            log_process_events();
        }

        write_y4m_header(pipes.f_stdin, oip, rgy_output_csp);

        //ログウィンドウ側から制御を可能に
        DWORD tm_vid_enc_start = timeGetTime();
        enable_enc_control(&enc_pause, afs, TRUE, tm_vid_enc_start, oip->n);

        //------------メインループ------------
        for (i = 0, next_jitter = jitter + 1, pe->drop_count = 0; i < oip->n; i++, next_jitter++) {
            //中断を確認
            ret |= (oip->func_is_abort()) ? AUO_RESULT_ABORT : AUO_RESULT_SUCCESS;

            //x264が実行中なら、メッセージを取得・ログウィンドウに表示
            if (ReadLogEnc(&pipes, pe->drop_count, i) < 0) {
                //勝手に死んだ...
                ret |= AUO_RESULT_ERROR; error_x264_dead();
                break;
            }

            if (!(i & 7)) {
                //Aviutlの進捗表示を更新
                oip->func_rest_time_disp(i, oip->n);

                //音声同時処理
                ret |= aud_parallel_task(oip, pe);
            }

            //一時停止
            while (enc_pause & !ret) {
                Sleep(LOG_UPDATE_INTERVAL);
                ret |= (oip->func_is_abort()) ? AUO_RESULT_ABORT : AUO_RESULT_SUCCESS;
                ReadLogEnc(&pipes, pe->drop_count, i);
                log_process_events();
            }

            //標準入力への書き込み完了をチェック
            while (WAIT_TIMEOUT == WaitForSingleObject(thread_data.he_out_fin, LOG_UPDATE_INTERVAL)) {
                ret |= (oip->func_is_abort()) ? AUO_RESULT_ABORT : AUO_RESULT_SUCCESS;
                ReadLogEnc(&pipes, pe->drop_count, i);
                log_process_events();
            }

            //中断・エラー等をチェック
            if (AUO_RESULT_SUCCESS != ret)
                break;

            //コピーフレームフラグ処理
            copy_frame = (!!i & (oip->func_get_flag(i) & OUTPUT_INFO_FRAME_FLAG_COPYFRAME));

            //Aviutl(afs)からフレームをもらう
            if (NULL == (frame = ((afs) ? afs_get_video((OUTPUT_INFO *)oip, i, &drop, next_jitter) : oip->func_get_video_ex(i, aviutl_color_fmt)))) {
                ret |= AUO_RESULT_ERROR; error_afs_get_frame();
                break;
            }

            drop |= (afs & copy_frame);

            if (!drop) {
                //コピーフレームの場合は、映像バッファの中身を更新せず、そのままパイプに流す
                if (!copy_frame)
                    convert_frame(frame, &pixel_data, oip->w, oip->h);  /// YUY2/YC48->NV12/YUV444変換, RGBコピー
                //標準入力への書き込みを開始
                SetEvent(thread_data.he_out_start);
            } else {
                *(next_jitter - 1) = DROP_FRAME_FLAG;
                pe->drop_count++;
                //次のフレームの変換を許可
                SetEvent(thread_data.he_out_fin);
            }

            // 「表示 -> セーブ中もプレビュー表示」がチェックされていると
            // func_update_preview() の呼び出しによって func_get_video_ex() の
            // 取得したバッファが書き換えられてしまうので、呼び出し位置を移動 (拡張AVI出力 plus より)
            oip->func_update_preview();
        }
        //------------メインループここまで--------------

        //書き込みスレッドを終了
        video_output_close_thread(&thread_data, ret);

        //ログウィンドウからのx264制御を無効化
        disable_enc_control();

        //パイプを閉じる
        CloseStdIn(&pipes);

        if (!ret) oip->func_rest_time_disp(oip->n, oip->n);

        //音声の同時処理を終了させる
        ret |= finish_aud_parallel_task(oip, pe, ret);
        //音声との同時処理が終了
        release_audio_parallel_events(pe);

        //タイムコード出力
        if (!ret && (afs || conf->vid.auo_tcfile_out))
            tcfile_out(jitter, oip->n, (double)oip->rate / (double)oip->scale, afs, pe);

        //エンコーダ終了待機
        while (WaitForSingleObject(pi_enc.hProcess, LOG_UPDATE_INTERVAL) == WAIT_TIMEOUT)
            ReadLogEnc(&pipes, pe->drop_count, i);

        DWORD tm_vid_enc_fin = timeGetTime();

        //最後にメッセージを取得
        while (ReadLogEnc(&pipes, pe->drop_count, i) > 0);

        if (!(ret & AUO_RESULT_ERROR) && afs)
            write_log_auo_line_fmt(LOG_INFO, "drop %d / %d frames", pe->drop_count, i);
        
        write_log_auo_line_fmt(LOG_INFO, "CPU使用率: Aviutl: %.2f%% / QSVEnc: %.2f%%", GetProcessAvgCPUUsage(pe->h_p_aviutl, &time_aviutl), GetProcessAvgCPUUsage(pi_enc.hProcess));
        write_log_auo_enc_time("QSVEncエンコード時間", tm_vid_enc_fin - tm_vid_enc_start);
    }
    set_window_title(AUO_FULL_NAME, PROGRESSBAR_DISABLED);
    
    if (jitter) free(jitter);

    return ret;
#if 0
    {
#if ENABLE_AVSW_READER
        if (!check_avcodec_dll() || !conf->vid.afs) {

        }
#endif //ENABLE_AVSW_READER
        set_conf_qsvp_prm(&conf->qsv, oip, pe, sys_dat->exstg->s_local.force_bluray, sys_dat->exstg->s_local.timer_period_tuning, sys_dat->exstg->s_log.log_level);
    }
    conf->qsv.nPerfMonitorSelect        = (sys_dat->exstg->s_local.perf_monitor) ? PERF_MONITOR_ALL : 0;
    conf->qsv.nPerfMonitorSelectMatplot = (sys_dat->exstg->s_local.perf_monitor_plot) ?
        PERF_MONITOR_CPU | PERF_MONITOR_CPU_KERNEL
        | PERF_MONITOR_THREAD_MAIN | PERF_MONITOR_THREAD_ENC | PERF_MONITOR_THREAD_OUT
        | PERF_MONITOR_FPS
        : 0;
    std::auto_ptr<CQSVPipeline> pPipeline;

    mfxStatus sts = MFX_ERR_NONE;
    int *jitter = NULL;

    //sts = ParseInputString(argv, (mfxU8)argc, &Params);
    //MSDK_CHECK_PARSE_RESULT(sts, MFX_ERR_NONE, 1);

    //pPipeline.reset((Params.nRotationAngle) ? new CUserPipeline : new CQSVPipeline); 
    pPipeline.reset(new AuoPipeline);
    //MSDK_CHECK_POINTER(pPipeline.get(), MFX_ERR_MEMORY_ALLOC);
    if (!pPipeline.get()) {
        write_log_auo_line(LOG_ERROR, "メモリの確保に失敗しました。");
        return AUO_RESULT_ERROR;
    }

    if (conf->vid.afs && (conf->qsv.nPicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF))) {
        sts = MFX_ERR_INVALID_VIDEO_PARAM; error_afs_interlace_stg();
    } else if ((jitter = (int *)calloc(oip->n + 1, sizeof(int))) == NULL) {
        sts = MFX_ERR_MEMORY_ALLOC; error_malloc_tc();
    //Aviutl(afs)からのフレーム読み込み
    } else if (!setup_afsvideo(oip, sys_dat, conf, pe)) {
        sts = MFX_ERR_UNKNOWN; //Aviutl(afs)からのフレーム読み込みに失敗
    //QSVEncプロセス開始
    } else {
        InputInfoAuo inputInfo = { 0 };
        set_input_info_auo(&inputInfo, oip, conf, pe, sys_dat, jitter);
        conf->qsv.pPrivatePrm = &inputInfo;
        if (MFX_ERR_NONE != (sts = pPipeline->Init(&conf->qsv))) {
            write_mfx_message(sts);
        } else if (MFX_ERR_NONE == (sts = pPipeline->CheckCurrentVideoParam())) {
            if (conf->vid.afs) write_log_auo_line(LOG_INFO, _T("自動フィールドシフト    on"));

            DWORD tm_qsv = timeGetTime();
            const char * const encode_name = (conf->qsv.bUseHWLib) ? "QuickSyncVideoエンコード" : "IntelMediaSDKエンコード";
            set_window_title(encode_name, PROGRESSBAR_CONTINUOUS);
            log_process_events();

            DWORD_PTR subThreadAffinityMask = 0x00;
            DWORD_PTR oldThreadAffinity = 0x00;
            if (sys_dat->exstg->s_local.thread_tuning)
                oldThreadAffinity = setThreadAffinityMaskforQSVEnc(NULL, &subThreadAffinityMask);

            for (;;) {
                sts = pPipeline->Run(subThreadAffinityMask);

                if (MFX_ERR_DEVICE_LOST == sts || MFX_ERR_DEVICE_FAILED == sts) {
                    write_log_auo_line(LOG_WARNING, "Hardware device was lost or returned an unexpected error. Recovering...");
                    if (MFX_ERR_NONE != (sts = pPipeline->ResetDevice())) {
                        write_mfx_message(sts);
                        break;
                    }

                    if (MFX_ERR_NONE != (sts = pPipeline->ResetMFXComponents(&conf->qsv))) {
                        write_mfx_message(sts);
                        break;
                    }
                    continue;
                } else {
                    //if (sts != MFX_ERR_NONE)
                    //    write_mfx_message(sts);
                    break;
                }
            }

            pPipeline->Close();
            resetThreadAffinityMaskforQSVEnc(oldThreadAffinity);
            write_log_auo_enc_time(encode_name, timeGetTime() - tm_qsv);
        }
    }
    //タイムコード出力
    if (sts == MFX_ERR_NONE && (conf->vid.afs || conf->vid.auo_tcfile_out))
        tcfile_out(jitter, oip->n, (double)oip->rate / (double)oip->scale, conf->vid.afs, pe);
    if (sts == MFX_ERR_NONE && conf->vid.afs)
        write_log_auo_line_fmt(LOG_INFO, "drop %d / %d frames", pe->drop_count, oip->n);
    set_window_title(AUO_FULL_NAME, PROGRESSBAR_DISABLED);
    if (jitter) free(jitter);

    return (sts == MFX_ERR_NONE) ? AUO_RESULT_SUCCESS : AUO_RESULT_ERROR;
#endif
}
#pragma warning( pop )

AUO_RESULT video_output(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat) {
    return exit_audio_parallel_control(oip, pe, video_output_inside(conf, oip, pe, sys_dat));
}
