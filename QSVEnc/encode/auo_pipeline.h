//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------


#ifndef _AUO_PIPELINE_ENCODE_H_
#define _AUO_PIPELINE_ENCODE_H_

#include "pipeline_encode.h"

class CAuoLog : public CQSVLog {
public:
    CAuoLog(const TCHAR *pLogFile, int log_level) : CQSVLog(pLogFile, log_level) { };
    virtual void operator()(int log_level, const TCHAR *format, ...) override;
};

class AuoPipeline : public CEncodingPipeline
{
public:
    AuoPipeline();
    virtual ~AuoPipeline();
    virtual mfxStatus InitLog(sInputParams *pParams) override;
    virtual mfxStatus InitInput(sInputParams *pParams) override;
    virtual mfxStatus InitOutput(sInputParams *pParams) override;
};


#endif //_AUO_PIPELINE_ENCODE_H_
