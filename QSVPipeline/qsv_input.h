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

#ifndef __QSV_INPUT_H__
#define __QSV_INPUT_H__

#include <memory>
#include "qsv_osdep.h"
#include "qsv_tchar.h"
#include "qsv_log.h"
#include "qsv_event.h"
#include "qsv_control.h"
#include "convert_csp.h"
#include "rgy_err.h"

static_assert(std::is_pod<VideoInfo>::value == true, "VideoInfo is POD");

class CQSVInput
{
public:

    CQSVInput();
    virtual ~CQSVInput();

    virtual void SetQSVLogPtr(shared_ptr<CQSVLog> pQSVLog) {
        m_pPrintMes = pQSVLog;
    }
    virtual RGY_ERR Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const void *prm, CEncodingThread *pEncThread, shared_ptr<CEncodeStatusInfo> pEncSatusInfo) = 0;

    //この関数がRGY_ERR_NONE以外を返すことでRunEncodeは終了処理に入る
    RGY_ERR GetNextFrame(mfxFrameSurface1** pSurface) {
        const int inputBufIdx = m_pEncThread->m_nFrameGet % m_pEncThread->m_nFrameBuffer;
        sInputBufSys *pInputBuf = &m_pEncThread->m_InputBuf[inputBufIdx];

        //_ftprintf(stderr, "GetNextFrame: wait for %d\n", m_pEncThread->m_nFrameGet);
        //_ftprintf(stderr, "wait for heInputDone, %d\n", m_pEncThread->m_nFrameGet);
        AddMessage(RGY_LOG_TRACE, _T("Enc Thread: Wait Done %d.\n"), m_pEncThread->m_nFrameGet);
        WaitForSingleObject(pInputBuf->heInputDone, INFINITE);
        //エラー・中断要求などでの終了
        if (m_pEncThread->m_bthForceAbort) {
            AddMessage(RGY_LOG_DEBUG, _T("GetNextFrame: Encode Aborted...\n"));
            return m_pEncThread->m_stsThread;
        }
        //読み込み完了による終了
        if (m_pEncThread->m_stsThread == RGY_ERR_MORE_DATA && m_pEncThread->m_nFrameGet == m_pEncSatusInfo->m_nInputFrames) {
            AddMessage(RGY_LOG_DEBUG, _T("GetNextFrame: Frame read finished.\n"));
            return m_pEncThread->m_stsThread;
        }
        //フレーム読み込みでない場合は、フレーム関連の処理は行わない
        if (getInputCodec() == RGY_CODEC_UNKNOWN) {
            *pSurface = pInputBuf->pFrameSurface;
            (*pSurface)->Data.TimeStamp = inputBufIdx;
            (*pSurface)->Data.Locked = FALSE;
            m_pEncThread->m_nFrameGet++;
        }
        return RGY_ERR_NONE;
    }

#pragma warning (push)
#pragma warning (disable: 4100)
    //動画ストリームの1フレーム分のデータをbitstreamに追加する (リーダー側のデータは消す)
    virtual RGY_ERR GetNextBitstream(RGYBitstream *pBitstream) {
        return RGY_ERR_NONE;
    }

    //動画ストリームの1フレーム分のデータをbitstreamに追加する (リーダー側のデータは残す)
    virtual RGY_ERR GetNextBitstreamNoDelete(RGYBitstream *pBitstream) {
        return RGY_ERR_NONE;
    }

    //ストリームのヘッダ部分を取得する
    virtual RGY_ERR GetHeader(RGYBitstream *pBitstream) {
        return RGY_ERR_NONE;
    }
#pragma warning (pop)

    RGY_ERR SetNextSurface(mfxFrameSurface1 *pSurface) {
        const int inputBufIdx = m_pEncThread->m_nFrameSet % m_pEncThread->m_nFrameBuffer;
        sInputBufSys *pInputBuf = &m_pEncThread->m_InputBuf[inputBufIdx];
        //フレーム読み込みでない場合は、フレーム関連の処理は行わない
        if (getInputCodec() == RGY_CODEC_UNKNOWN) {
            //_ftprintf(stderr, "Set heInputStart: %d\n", m_pEncThread->m_nFrameSet);
            pSurface->Data.Locked = TRUE;
            //_ftprintf(stderr, "set surface %d, set event heInputStart %d\n", pSurface, m_pEncThread->m_nFrameSet);
            pInputBuf->pFrameSurface = pSurface;
        }
        SetEvent(pInputBuf->heInputStart);
        AddMessage(RGY_LOG_TRACE, _T("Enc Thread: Set Start %d.\n"), m_pEncThread->m_nFrameSet);
        m_pEncThread->m_nFrameSet++;
        return RGY_ERR_NONE;
    }

    virtual void Close();
    //virtual RGY_ERR Init(const TCHAR *strFileName, const mfxU32 ColorFormat, const mfxU32 numViews, std::vector<TCHAR*> srcFileBuff);
    virtual RGY_ERR LoadNextFrame(mfxFrameSurface1 *pSurface) = 0;

    void SetTrimParam(const sTrimParam& trim) {
        m_sTrimParam = trim;
    }

    sTrimParam *GetTrimParam() {
        return &m_sTrimParam;
    }

    sInputCrop GetInputCropInfo() {
        return m_inputVideoInfo.crop;
    }
    VideoInfo GetInputFrameInfo() {
        return m_inputVideoInfo;
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
    //HWデコードを行う場合のコーデックを返す
    //行わない場合はRGY_CODEC_UNKNOWNを返す
    RGY_CODEC getInputCodec() {
        return m_inputVideoInfo.codec;
    }
protected:
    virtual void CreateInputInfo(const TCHAR *inputTypeName, const TCHAR *inputCSpName, const TCHAR *outputCSpName, const TCHAR *convSIMD, const VideoInfo *inputPrm);

    //trim listを参照し、動画の最大フレームインデックスを取得する
    int getVideoTrimMaxFramIdx() {
        if (m_sTrimParam.list.size() == 0) {
            return INT_MAX;
        }
        return m_sTrimParam.list[m_sTrimParam.list.size()-1].fin;
    }

    CEncodingThread *m_pEncThread;
    shared_ptr<CEncodeStatusInfo> m_pEncSatusInfo;

    VideoInfo m_inputVideoInfo;

    RGY_CSP m_InputCsp;
    const ConvertCSP *m_sConvert;

    tstring m_strInputInfo;
    shared_ptr<CQSVLog> m_pPrintMes;  //ログ出力
    tstring m_strReaderName;

    sTrimParam m_sTrimParam;
};

class CQSVInputRaw : public CQSVInput {
public:
    CQSVInputRaw();
    ~CQSVInputRaw();
protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const void *prm, CEncodingThread *pEncThread, shared_ptr<CEncodeStatusInfo> pEncSatusInfo) override;
    virtual RGY_ERR LoadNextFrame(mfxFrameSurface1* pSurface) override;
    virtual void Close() override;

    RGY_ERR ParseY4MHeader(char *buf, VideoInfo *pInfo);

    FILE *m_fSource;

    uint32_t m_nBufSize;
    shared_ptr<uint8_t> m_pBuffer;
};


#endif //__QSV_INPUT_H__

