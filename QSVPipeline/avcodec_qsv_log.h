//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------
#ifndef _AVCODEC_QSV_LOG_H_
#define _AVCODEC_QSV_LOG_H_

#include "qsv_version.h"

#if ENABLE_AVCODEC_QSV_READER

#include "avcodec_qsv.h"

void av_qsv_log_set(std::shared_ptr<CQSVLog>& pQSVLog);
void av_qsv_log_free();

#endif //ENABLE_AVCODEC_QSV_READER

#endif //_AVCODEC_QSV_LOG_H_
