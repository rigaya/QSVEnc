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


#ifndef _AUO_PIPELINE_ENCODE_H_
#define _AUO_PIPELINE_ENCODE_H_
#if 0
#include "qsv_pipeline.h"
#include "rgy_input.h"
#include "rgy_output.h"

class CAuoLog : public RGYLog {
public:
    CAuoLog(const TCHAR *pLogFile, int log_level) : RGYLog(pLogFile, log_level) { };

    virtual void write_log(int log_level, const TCHAR *buffer, bool file_only = false) override;
    virtual void write(int log_level, const TCHAR *format, ...) override;
};

class AuoPipeline : public CQSVPipeline
{
public:
    AuoPipeline();
    virtual ~AuoPipeline();
    virtual mfxStatus InitLog(sInputParams *pParams) override;
    virtual mfxStatus InitInput(sInputParams *pParams) override;
    virtual mfxStatus InitOutput(sInputParams *pParams) override;
};
#endif

#endif //_AUO_PIPELINE_ENCODE_H_
