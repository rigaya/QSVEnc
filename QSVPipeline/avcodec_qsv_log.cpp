//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#include <memory>
#include "qsv_version.h"

#if ENABLE_AVCODEC_QSV_READER

#include "qsv_log.h"
#include "avcodec_qsv_log.h"

static std::weak_ptr<CQSVLog> g_pQSVLog;
static int print_prefix = 1;

static void av_qsv_log_callback(void *ptr, int level, const char *fmt, va_list vl) {
    if (auto pQSVLog = g_pQSVLog.lock()) {
        const int qsv_log_level = log_level_av2qsv(level);
        if (qsv_log_level >= pQSVLog->getLogLevel()) {
            char mes[4096];
            av_log_format_line(ptr, level, fmt, vl, mes, sizeof(mes), &print_prefix);
            pQSVLog->write_log(qsv_log_level, char_to_tstring(mes).c_str(), true);
        }
    }
    av_log_default_callback(ptr, level, fmt, vl);
}

void av_qsv_log_set(std::shared_ptr<CQSVLog>& pQSVLog) {
    g_pQSVLog = pQSVLog;
    av_log_set_callback(av_qsv_log_callback);
}

void av_qsv_log_free() {
    av_log_set_callback(av_log_default_callback);
    g_pQSVLog.reset();
}

#endif //ENABLE_AVCODEC_QSV_READER
