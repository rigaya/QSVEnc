//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef __QSV_OUTPUT_H__
#define __QSV_OUTPUT_H__

#include <memory>
#include <vector>
#include "qsv_osdep.h"
#include "qsv_tchar.h"
#include "qsv_log.h"
#include "qsv_control.h"

using std::unique_ptr;
using std::shared_ptr;

static const int MAX_BUF_SIZE_MB = 128;

class CQSVOut {
public:
    CQSVOut();
    virtual ~CQSVOut();

    virtual void SetQSVLogPtr(shared_ptr<CQSVLog> pQSVLog) {
        m_pPrintMes = pQSVLog;
    }
    virtual mfxStatus Init(const TCHAR *strFileName, const void *prm, shared_ptr<CEncodeStatusInfo> pEncSatusInfo) = 0;

    virtual mfxStatus SetVideoParam(const mfxVideoParam *pMfxVideoPrm, const mfxExtCodingOption2 *cop2) = 0;

    virtual mfxStatus WriteNextFrame(mfxBitstream *pMfxBitstream) = 0;
    virtual mfxStatus WriteNextFrame(mfxFrameSurface1 *pSurface) = 0;
    virtual void Close();

    virtual bool outputStdout() {
        return m_bOutputIsStdout;
    }

    const TCHAR *GetOutputMessage() {
        const TCHAR *mes = m_strOutputInfo.c_str();
        return (mes) ? mes : _T("");
    }
    void AddMessage(int log_level, const tstring& str) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel()) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto& line : lines) {
            if (line[0] != _T('\0')) {
                (*m_pPrintMes)(log_level, (m_strWriterName + _T(": ") + line + _T("\n")).c_str());
            }
        }
    }
    void AddMessage(int log_level, const TCHAR *format, ... ) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel()) {
            return;
        }

        va_list args;
        va_start(args, format);
        int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
        tstring buffer;
        buffer.resize(len, _T('\0'));
        _vstprintf_s(&buffer[0], len, format, args);
        va_end(args);
        AddMessage(log_level, buffer);
    }
protected:
    shared_ptr<CEncodeStatusInfo> m_pEncSatusInfo;
    unique_ptr<FILE, fp_deleter>  m_fDest;
    bool        m_bOutputIsStdout;
    bool        m_bInited;
    bool        m_bNoOutput;
    bool        m_bSourceHWMem;
    bool        m_bY4mHeaderWritten;
    tstring     m_strWriterName;
    tstring     m_strOutputInfo;
    shared_ptr<CQSVLog> m_pPrintMes;  //ログ出力
    unique_ptr<char, malloc_deleter>            m_pOutputBuffer;
    unique_ptr<uint8_t, aligned_malloc_deleter> m_pReadBuffer;
    unique_ptr<mfxU8, aligned_malloc_deleter>   m_pUVBuffer;
};

struct CQSVOutRawPrm {
    bool bBenchmark;
    int nBufSizeMB;
};

class CQSVOutBitstream : public CQSVOut {
public:

    CQSVOutBitstream();
    virtual ~CQSVOutBitstream();

    virtual mfxStatus Init(const TCHAR *strFileName, const void *prm, shared_ptr<CEncodeStatusInfo> pEncSatusInfo) override;

    virtual mfxStatus SetVideoParam(const mfxVideoParam *pMfxVideoPrm, const mfxExtCodingOption2 *cop2) override;

    virtual mfxStatus WriteNextFrame(mfxBitstream *pMfxBitstream) override;
    virtual mfxStatus WriteNextFrame(mfxFrameSurface1 *pSurface) override;
};


struct YUVWriterParam {
    bool bY4m;
    MemType memType;
};

class CQSVOutFrame : public CQSVOut {
public:

    CQSVOutFrame();
    virtual ~CQSVOutFrame();

    virtual mfxStatus Init(const TCHAR *strFileName, const void *prm, shared_ptr<CEncodeStatusInfo> pEncSatusInfo) override;

    virtual mfxStatus SetVideoParam(const mfxVideoParam *pMfxVideoPrm, const mfxExtCodingOption2 *cop2) override;
    virtual mfxStatus WriteNextFrame(mfxBitstream *pMfxBitstream) override;
    virtual mfxStatus WriteNextFrame(mfxFrameSurface1 *pSurface) override;
protected:
    bool m_bY4m;
};



#endif //__QSV_OUTPUT_H__

