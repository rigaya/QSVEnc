//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <string.h>
#include <stdio.h>
#include <Windows.h>
#include "auo_util.h"
#include "auo_conf.h"

void guiEx_config::convert_qsvstgv1_to_stgv3(CONF_GUIEX *conf, int size) {
    strcpy_s(conf->conf_name, CONF_NAME);
    conf->qsv.nBitRate = conf->qsv.__nBitRate;
    conf->qsv.nMaxBitrate = conf->qsv.__nMaxBitrate;
    conf->qsv.__nBitRate = 0;
    conf->qsv.__nMaxBitrate = 0;
    strcpy_s(conf->conf_name, CONF_NAME_OLD_2);
    
    memset(((BYTE *)conf) + size - 2056, 0, 2056);
    strcpy_s(conf->conf_name, CONF_NAME_OLD_3);
}

void guiEx_config::convert_qsvstgv2_to_stgv3(CONF_GUIEX *conf) {
    static const DWORD OLD_FLAG_AFTER  = 0x01;
    static const DWORD OLD_FLAG_BEFORE = 0x02;

    char bat_path_before_process[1024];
    char bat_path_after_process[1024];
    strcpy_s(bat_path_after_process,  conf->oth.batfiles[0]);
    strcpy_s(bat_path_before_process, conf->oth.batfiles[2]);
    
    DWORD old_run_bat_flags = conf->oth.run_bat;
    conf->oth.run_bat  = 0x00;
    conf->oth.run_bat |= (old_run_bat_flags & OLD_FLAG_BEFORE) ? RUN_BAT_BEFORE_PROCESS : 0x00;
    conf->oth.run_bat |= (old_run_bat_flags & OLD_FLAG_AFTER)  ? RUN_BAT_AFTER_PROCESS  : 0x00;

    memset(&conf->oth.batfiles[0], 0, sizeof(conf->oth.batfiles));
    strcpy_s(conf->oth.batfile.before_process, bat_path_before_process);
    strcpy_s(conf->oth.batfile.after_process,  bat_path_after_process);
    strcpy_s(conf->conf_name, CONF_NAME_OLD_3);
}

void guiEx_config::convert_qsvstgv3_to_stgv4(CONF_GUIEX *conf) {
    if (conf->qsv.nOutputBufSizeMB == 0) {
        conf->qsv.nOutputBufSizeMB = QSV_DEFAULT_OUTPUT_BUF_MB;
    } else {
        conf->qsv.nOutputBufSizeMB = clamp(conf->qsv.nOutputBufSizeMB, 0, QSV_OUTPUT_BUF_MB_MAX);
    }
    strcpy_s(conf->conf_name, CONF_NAME_OLD_4);
}
