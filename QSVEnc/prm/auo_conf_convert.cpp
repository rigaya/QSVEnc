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

void guiEx_config::convert_qsvstgv1_to_stgv2(CONF_GUIEX *conf) {
	strcpy_s(conf->conf_name, CONF_NAME);
	conf->qsv.nBitRate = conf->qsv.__nBitRate;
	conf->qsv.nMaxBitrate = conf->qsv.__nMaxBitrate;
	conf->qsv.__nBitRate = 0;
	conf->qsv.__nMaxBitrate = 0;
	conf->qsv.__nThreads = 0;
}
