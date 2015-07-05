//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <Windows.h>
#pragma comment(lib, "user32.lib") //WaitforInputIdle
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include <vector>

#include "output.h"
#include "convert.h"

#include "auo.h"
#include "auo_frm.h"
#include "auo_pipe.h"
#include "auo_error.h"
#include "auo_conf.h"
#include "auo_util.h"
#include "auo_system.h"
#include "auo_version.h"
#include "auo_qsv_link.h"
#include "avcodec_qsv.h"

#include "auo_encode.h"
#include "auo_video.h"
#include "auo_audio_parallel.h"
#include "auo_pipeline.h"

DWORD set_auo_yuvreader_g_data(const OUTPUT_INFO *_oip, CONF_GUIEX *conf, PRM_ENC *_pe, int *jitter);
void clear_auo_yuvreader_g_data();

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
        for (int wait_for_audio_count = 0; pe->aud_parallel.he_vid_start; wait_for_audio_count++) {
            vid_ret |= aud_parallel_task(oip, pe);
            if (wait_for_audio_count == 5)
                write_log_auo_line(LOG_INFO, "音声処理の終了を待機しています...");
        }
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
        int wait_for_audio_count = 0;
        while (WaitForSingleObject(pe->aud_parallel.th_aud, LOG_UPDATE_INTERVAL) == WAIT_TIMEOUT) {
            if (wait_for_audio_count == 10)
                set_window_title("音声処理の終了を待機しています...", PROGRESSBAR_MARQUEE);
            pe->aud_parallel.abort |= oip->func_is_abort();
            log_process_events();
            wait_for_audio_count++;
        }
        flush_audio_log();
        if (wait_for_audio_count > 10)
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

void set_conf_qsvp_prm(sInputParams *prm, const OUTPUT_INFO *oip, const PRM_ENC *pe, BOOL force_bluray, BOOL timer_period_tuning) {
    prm->nHeight = (mfxU16)oip->h;
    prm->nWidth = (mfxU16)oip->w;
    prm->nFPSRate = oip->rate;
    prm->nFPSScale = oip->scale;
    if (!prm->vpp.bEnable) {
        //vppを無効化する
        ZeroMemory(&prm->vpp, sizeof(sVppParams));
    }
    if (!prm->vpp.bUseResize) {
        prm->nDstWidth = (mfxU16)oip->w;
        prm->nDstHeight = (mfxU16)oip->h;
    }
    if ((prm->nPicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF)) == FALSE) {
        prm->vpp.nDeinterlace = MFX_DEINTERLACE_NONE;
    }
    strcpy_s(prm->strDstFile, sizeof(prm->strDstFile), pe->temp_filename);
    prm->nInputBufSize = clamp(prm->nInputBufSize, QSV_INPUT_BUF_MIN, QSV_INPUT_BUF_MAX);
    
    prm->nBluray += (prm->nBluray == 1 && force_bluray);

    prm->bDisableTimerPeriodTuning = !timer_period_tuning;
}

static DWORD_PTR setThreadAffinityMaskforQSVEnc(DWORD_PTR *mainThreadAffinityMask, DWORD_PTR *subThreadAffinityMask) {
    DWORD_PTR dwProcessAffinityMask = 0, dwSystemAffinityMask = 0;
    if (mainThreadAffinityMask) *mainThreadAffinityMask = 0;
    if (subThreadAffinityMask)  *subThreadAffinityMask = 0;
    if (FALSE == GetProcessAffinityMask(GetCurrentProcess(), &dwProcessAffinityMask, &dwSystemAffinityMask))
        return NULL;
    
    cpu_info_t cpu_info;
    if (!get_cpu_info(&cpu_info)
        || sizeof(DWORD_PTR) * 8 < cpu_info.logical_cores
        || cpu_info.physical_cores <= 2
        || cpu_info.logical_cores <= 4)
        return NULL;

    DWORD_PTR newMainThreadAffinityMask = 0x00, tmpMask = 0x01;
    for (DWORD i = 0; i < cpu_info.logical_cores / cpu_info.physical_cores; i++, tmpMask <<= 1)
        newMainThreadAffinityMask |= tmpMask;

    DWORD_PTR otherThreadAffinityMask = dwProcessAffinityMask & (~newMainThreadAffinityMask);

    //dwProcessAffinityMask = SetThreadAffinityMask(GetCurrentThread(), newMainThreadAffinityMask);
    SetThreadIdealProcessor(GetCurrentThread(), 0);
    SetThreadAffinityForModule(GetCurrentProcessId(), "libmfxhw", otherThreadAffinityMask);

    if (mainThreadAffinityMask) *mainThreadAffinityMask = newMainThreadAffinityMask;
    if (subThreadAffinityMask)  *subThreadAffinityMask = otherThreadAffinityMask;
    return dwProcessAffinityMask;
}

static BOOL resetThreadAffinityMaskforQSVEnc(DWORD_PTR MaskBefore) {
    BOOL ret = FALSE;
    if (MaskBefore)
        if (SetThreadAffinityMask(GetCurrentThread(), MaskBefore))
            ret = TRUE;
    return ret;
}

#pragma warning( push )
#pragma warning( disable: 4100 )
static DWORD video_output_inside(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat) {
    //動画エンコードの必要がなければ終了
    if (pe->video_out_type == VIDEO_OUTPUT_DISABLED)
        return AUO_RESULT_SUCCESS;

#if ENABLE_AVCODEC_QSV_READER
    if (!check_avcodec_dll() || !conf->vid.afs)
#endif //ENABLE_AVCODEC_QSV_READER
        //H.264/ESで出力するので拡張子を変更
        change_ext(pe->temp_filename, _countof(pe->temp_filename), ".264");

    sInputParams *Params = &conf->qsv;
    set_conf_qsvp_prm(Params, oip, pe, sys_dat->exstg->s_local.force_bluray, sys_dat->exstg->s_local.timer_period_tuning);

    std::auto_ptr<CEncodingPipeline> pPipeline;

    mfxStatus sts = MFX_ERR_NONE;
    int *jitter = NULL;

    //sts = ParseInputString(argv, (mfxU8)argc, &Params);
    //MSDK_CHECK_PARSE_RESULT(sts, MFX_ERR_NONE, 1);

    //pPipeline.reset((Params.nRotationAngle) ? new CUserPipeline : new CEncodingPipeline); 
    pPipeline.reset(new AuoPipeline);
    //MSDK_CHECK_POINTER(pPipeline.get(), MFX_ERR_MEMORY_ALLOC);
    if (!pPipeline.get()) {
        write_log_auo_line(LOG_ERROR, "メモリの確保に失敗しました。");
        return AUO_RESULT_ERROR;
    }

    //if (Params.bIsMVC)
    //{
    //    pPipeline->SetMultiView();
    //    pPipeline->SetNumView(Params.numViews);
    //}
    if (conf->vid.afs && (conf->qsv.nPicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF))) {
        sts = MFX_ERR_INVALID_VIDEO_PARAM; error_afs_interlace_stg();
    } else if ((jitter = (int *)calloc(oip->n + 1, sizeof(int))) == NULL) {
        sts = MFX_ERR_MEMORY_ALLOC; error_malloc_tc();
    //Aviutl(afs)からのフレーム読み込み
    } else if (!setup_afsvideo(oip, sys_dat, conf, pe)) {
        sts = MFX_ERR_UNKNOWN; //Aviutl(afs)からのフレーム読み込みに失敗
    //QSVEncプロセス開始
    } else if (AUO_RESULT_SUCCESS != set_auo_yuvreader_g_data(oip, conf, pe, jitter)
            || MFX_ERR_NONE != (sts = pPipeline->Init(Params))) {
        write_mfx_message(sts);
    } else if (MFX_ERR_NONE == (sts = pPipeline->CheckCurrentVideoParam())) {
        if (conf->vid.afs) write_log_auo_line(LOG_INFO, _T("自動フィールドシフト    on"));

        //_ftprintf(stderr, _T("Processing started\n"));
        DWORD tm_qsv = timeGetTime();
        //write_log_auo_line(LOG_INFO, "Processing started...");
        const char * const encode_name = (conf->qsv.bUseHWLib) ? "QuickSyncVideoエンコード" : "IntelMediaSDKエンコード";
        set_window_title(encode_name, PROGRESSBAR_CONTINUOUS);

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

                if (MFX_ERR_NONE != (sts = pPipeline->ResetMFXComponents(Params))) {
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
    clear_auo_yuvreader_g_data();
    //タイムコード出力
    if (sts == MFX_ERR_NONE && (conf->vid.afs || conf->vid.auo_tcfile_out))
        tcfile_out(jitter, oip->n, (double)oip->rate / (double)oip->scale, conf->vid.afs, pe);
    if (sts == MFX_ERR_NONE && conf->vid.afs)
        write_log_auo_line_fmt(LOG_INFO, "drop %d / %d frames", pe->drop_count, oip->n);
    set_window_title(AUO_FULL_NAME, PROGRESSBAR_DISABLED);
    if (jitter) free(jitter);

    return (sts == MFX_ERR_NONE) ? AUO_RESULT_SUCCESS : AUO_RESULT_ERROR;
}
#pragma warning( pop )

AUO_RESULT video_output(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat) {
    return exit_audio_parallel_control(oip, pe, video_output_inside(conf, oip, pe, sys_dat));
}
