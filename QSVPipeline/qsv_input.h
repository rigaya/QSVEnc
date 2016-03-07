//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef __QSV_INPUT_H__
#define __QSV_INPUT_H__

#include <memory>
#include "qsv_osdep.h"
#include "qsv_tchar.h"
#include "qsv_log.h"
#include "qsv_event.h"
#include "qsv_control.h"
#include "convert_csp.h"

class CQSVInput
{
public:

    CQSVInput();
    virtual ~CQSVInput();

    virtual void SetQSVLogPtr(shared_ptr<CQSVLog> pQSVLog) {
        m_pPrintMes = pQSVLog;
    }
    virtual mfxStatus Init(const TCHAR *strFileName, mfxU32 ColorFormat, const void *prm, CEncodingThread *pEncThread, shared_ptr<CEncodeStatusInfo> pEncSatusInfo, sInputCrop *pInputCrop) = 0;

    //この関数がMFX_ERR_NONE以外を返すことでRunEncodeは終了処理に入る
    mfxStatus GetNextFrame(mfxFrameSurface1** pSurface) {
        const int inputBufIdx = m_pEncThread->m_nFrameGet % m_pEncThread->m_nFrameBuffer;
        sInputBufSys *pInputBuf = &m_pEncThread->m_InputBuf[inputBufIdx];

        //_ftprintf(stderr, "GetNextFrame: wait for %d\n", m_pEncThread->m_nFrameGet);
        //_ftprintf(stderr, "wait for heInputDone, %d\n", m_pEncThread->m_nFrameGet);
        AddMessage(QSV_LOG_TRACE, _T("Enc Thread: Wait Done %d.\n"), m_pEncThread->m_nFrameGet);
        WaitForSingleObject(pInputBuf->heInputDone, INFINITE);
        //エラー・中断要求などでの終了
        if (m_pEncThread->m_bthForceAbort) {
            AddMessage(QSV_LOG_DEBUG, _T("GetNextFrame: Encode Aborted...\n"));
            return m_pEncThread->m_stsThread;
        }
        //読み込み完了による終了
        if (m_pEncThread->m_stsThread == MFX_ERR_MORE_DATA && m_pEncThread->m_nFrameGet == m_pEncSatusInfo->m_nInputFrames) {
            AddMessage(QSV_LOG_DEBUG, _T("GetNextFrame: Frame read finished.\n"));
            return m_pEncThread->m_stsThread;
        }
        //フレーム読み込みでない場合は、フレーム関連の処理は行わない
        if (!getInputCodec()) {
            *pSurface = pInputBuf->pFrameSurface;
            (*pSurface)->Data.TimeStamp = inputBufIdx;
            (*pSurface)->Data.Locked = FALSE;
            m_pEncThread->m_nFrameGet++;
        }
        return MFX_ERR_NONE;
    }

#pragma warning (push)
#pragma warning (disable: 4100)
    virtual mfxStatus GetNextBitstream(mfxBitstream *bitstream) {
        return MFX_ERR_NONE;
    }
    virtual mfxStatus GetHeader(mfxBitstream *bitstream) {
        return MFX_ERR_NONE;
    }
#pragma warning (pop)

    mfxStatus SetNextSurface(mfxFrameSurface1 *pSurface) {
        const int inputBufIdx = m_pEncThread->m_nFrameSet % m_pEncThread->m_nFrameBuffer;
        sInputBufSys *pInputBuf = &m_pEncThread->m_InputBuf[inputBufIdx];
        //フレーム読み込みでない場合は、フレーム関連の処理は行わない
        if (!getInputCodec()) {
            //_ftprintf(stderr, "Set heInputStart: %d\n", m_pEncThread->m_nFrameSet);
            pSurface->Data.Locked = TRUE;
            //_ftprintf(stderr, "set surface %d, set event heInputStart %d\n", pSurface, m_pEncThread->m_nFrameSet);
            pInputBuf->pFrameSurface = pSurface;
        }
        SetEvent(pInputBuf->heInputStart);
        AddMessage(QSV_LOG_TRACE, _T("Enc Thread: Set Start %d.\n"), m_pEncThread->m_nFrameSet);
        m_pEncThread->m_nFrameSet++;
        return MFX_ERR_NONE;
    }

    virtual void Close();
    //virtual mfxStatus Init(const TCHAR *strFileName, const mfxU32 ColorFormat, const mfxU32 numViews, std::vector<TCHAR*> srcFileBuff);
    virtual mfxStatus LoadNextFrame(mfxFrameSurface1 *pSurface) = 0;

    void SetTrimParam(const sTrimParam& trim) {
        m_sTrimParam = trim;
    }

    const sTrimParam *GetTrimParam() {
        return &m_sTrimParam;
    }
    mfxU32 m_ColorFormat; // color format of input YUV data, YUV420 or NV12
    void GetInputCropInfo(sInputCrop *cropInfo) {
        memcpy(cropInfo, &m_sInputCrop, sizeof(m_sInputCrop));
    }
    void GetInputFrameInfo(mfxFrameInfo *inputFrameInfo) {
        memcpy(inputFrameInfo, &m_inputFrameInfo, sizeof(m_inputFrameInfo));
    }

    //入力ファイルに存在する音声のトラック数を返す
    virtual int GetAudioTrackCount() {
        return 0;
    }

    //入力ファイルに存在する字幕のトラック数を返す
    virtual int GetSubtitleTrackCount() {
        return 0;
    }
    const TCHAR *GetInputMessage() {
        const TCHAR *mes = m_strInputInfo.c_str();
        return (mes) ? mes : _T("");
    }
    void AddMessage(int log_level, const tstring& str) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel()) {
            return;
        }
        auto lines = split(str, _T("\n"));
        for (const auto& line : lines) {
            if (line[0] != _T('\0')) {
                m_pPrintMes->write(log_level, (m_strReaderName + _T(": ") + line + _T("\n")).c_str());
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
    //QSVデコードを行う場合のコーデックを返す
    //行わない場合は0を返す
    mfxU32 getInputCodec() {
        return m_nInputCodec;
    }
protected:
    //trim listを参照し、動画の最大フレームインデックスを取得する
    int getVideoTrimMaxFramIdx() {
        if (m_sTrimParam.list.size() == 0) {
            return INT_MAX;
        }
        return m_sTrimParam.list[m_sTrimParam.list.size()-1].fin;
    }

    FILE *m_fSource;
    CEncodingThread *m_pEncThread;
    shared_ptr<CEncodeStatusInfo> m_pEncSatusInfo;
    bool m_bInited;
    sInputCrop m_sInputCrop;

    mfxFrameInfo m_inputFrameInfo;

    const ConvertCSP *m_sConvert;

    mfxU32 m_nInputCodec;

    mfxU32 m_nBufSize;
    shared_ptr<uint8_t> m_pBuffer;

    tstring m_strReaderName;
    tstring m_strInputInfo;
    shared_ptr<CQSVLog> m_pPrintMes;  //ログ出力

    sTrimParam m_sTrimParam;
};

class CQSVInputRaw : public CQSVInput {
public:
    CQSVInputRaw();
    ~CQSVInputRaw();
protected:
    virtual mfxStatus Init(const TCHAR *strFileName, mfxU32 ColorFormat, const void *prm, CEncodingThread *pEncThread, shared_ptr<CEncodeStatusInfo> pEncSatusInfo, sInputCrop *pInputCrop) override;
    virtual mfxStatus LoadNextFrame(mfxFrameSurface1* pSurface) override;
    bool m_by4m;
};


#endif //__QSV_INPUT_H__

