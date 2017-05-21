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
        conf->qsv.nOutputBufSizeMB = clamp(conf->qsv.nOutputBufSizeMB, 0, RGY_OUTPUT_BUF_MB_MAX);
    }
    strcpy_s(conf->conf_name, CONF_NAME_OLD_4);
}

void guiEx_config::convert_qsvstgv4_to_stgv5(CONF_GUIEX *conf) {
    if (conf->qsv.nOutputThread == 0) {
        conf->qsv.nOutputThread = RGY_OUTPUT_THREAD_AUTO;
    }
    if (conf->qsv.nAudioThread == 0) {
        conf->qsv.nAudioThread = RGY_AUDIO_THREAD_AUTO;
    }
    strcpy_s(conf->conf_name, CONF_NAME_OLD_5);
}
