//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef __QSV_LOG_H__
#define __QSV_LOG_H__

#include <cstdint>
#include <thread>
#include <mutex>
#include "qsv_tchar.h"
#include "qsv_prm.h"

class CQSVLog {
protected:
    int m_nLogLevel = QSV_LOG_INFO;
    const TCHAR *m_pStrLog = nullptr;
    bool m_bHtml = false;
    std::mutex m_mtx;
    static const char *HTML_FOOTER;
public:
    CQSVLog(const TCHAR *pLogFile, int log_level = QSV_LOG_INFO) {
        init(pLogFile, log_level);
    };
    virtual ~CQSVLog() {
    };
    void init(const TCHAR *pLogFile, int log_level = QSV_LOG_INFO);
    void writeHtmlHeader();
    void writeFileHeader(const TCHAR *pDstFilename);
    void writeFileFooter();
    int getLogLevel() {
        return m_nLogLevel;
    }
    virtual void write_log(int log_level, TCHAR *buffer);
    virtual void write(int log_level, const TCHAR *format, ...);
};


#endif //__QSV_LOG_H__
