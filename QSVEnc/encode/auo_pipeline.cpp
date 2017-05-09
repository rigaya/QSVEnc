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

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "auo_frm.h"
#include "auo_pipeline.h"
#include "auo_qsv_link.h"

AuoPipeline::AuoPipeline() {
}

AuoPipeline::~AuoPipeline() {
}

mfxStatus AuoPipeline::InitLog(sInputParams *pParams) {
    m_pQSVLog.reset(new CAuoLog(pParams->pStrLogFile, pParams->nLogLevel));
    return MFX_ERR_NONE;
}

mfxStatus AuoPipeline::InitInput(sInputParams *pParams) {
    m_pEncSatusInfo = std::make_shared<AUO_EncodeStatusInfo>();
    m_pEncSatusInfo->SetPrivData(nullptr);

    if (pParams->nInputFmt != INPUT_FMT_AUO) {
        return CQSVPipeline::InitInput(pParams);
    }

    // prepare input file reader
    m_pFileReader = std::make_shared<AUO_YUVReader>();
    m_pFileReader->SetQSVLogPtr(m_pQSVLog);
    auto ret = m_pFileReader->Init(NULL, NULL, false, &m_EncThread, m_pEncSatusInfo, NULL);
    if (ret != RGY_ERR_NONE) return err_to_mfx(ret);

    mfxFrameInfo inputFrameInfo = { 0 };
    m_pFileReader->GetInputFrameInfo(&inputFrameInfo);

    mfxU32 OutputFPSRate = pParams->nFPSRate;
    mfxU32 OutputFPSScale = pParams->nFPSScale;
    mfxU32 outputFrames = *(mfxU32 *)&inputFrameInfo.FrameId;
    if ((pParams->nPicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF))) {
        switch (pParams->vpp.nDeinterlace) {
            case MFX_DEINTERLACE_IT:
            case MFX_DEINTERLACE_IT_MANUAL:
                OutputFPSRate = OutputFPSRate * 4;
                OutputFPSScale = OutputFPSScale * 5;
                outputFrames = (outputFrames * 4) / 5;
                break;
            case MFX_DEINTERLACE_BOB:
            case MFX_DEINTERLACE_AUTO_DOUBLE:
                OutputFPSRate = OutputFPSRate * 2;
                outputFrames *= 2;
                break;
            default:
                break;
        }
    }
    switch (pParams->vpp.nFPSConversion) {
    case FPS_CONVERT_MUL2:
        OutputFPSRate = OutputFPSRate * 2;
        outputFrames *= 2;
        break;
    case FPS_CONVERT_MUL2_5:
        OutputFPSRate = OutputFPSRate * 5 / 2;
        outputFrames = outputFrames * 5 / 2;
        break;
    default:
        break;
    }
    mfxU32 gcd = qsv_gcd(OutputFPSRate, OutputFPSScale);
    OutputFPSRate /= gcd;
    OutputFPSScale /= gcd;

    m_pEncSatusInfo->Init(OutputFPSRate, OutputFPSScale, outputFrames, m_pQSVLog, nullptr);

    return MFX_ERR_NONE;
}

mfxStatus AuoPipeline::InitOutput(sInputParams *pParams) {
    if (pParams->nInputFmt != INPUT_FMT_AUO) {
        return CQSVPipeline::InitOutput(pParams);
    }

    m_pFileWriter = std::make_shared<CQSVOutBitstream>();
    m_pFileWriter->SetQSVLogPtr(m_pQSVLog);
    bool bDummy = false;
    auto ret = m_pFileWriter->Init(pParams->strDstFile, &bDummy, m_pEncSatusInfo);
    if (ret != RGY_ERR_NONE) return err_to_mfx(ret);

    return MFX_ERR_NONE;
}

#pragma warning(push)
#pragma warning(disable:4100)
void CAuoLog::write_log(int log_level, const TCHAR *mes, bool file_only) {
    int len = _tcslen(mes) + 1;
    std::vector<TCHAR> buffer(len * 2 + 64, 0);
    memcpy(buffer.data(), mes, sizeof(buffer[0]) * len);
    TCHAR *buffer_line = buffer.data() + len;
    TCHAR *q = NULL;
    for (TCHAR *p = buffer.data(); (p = _tcstok_s(p, _T("\n"), &q)) != NULL; ) {
        static const TCHAR *const LOG_STRING[] = { _T("trace"),  _T("debug"), _T("info"), _T("info"), _T("warn"), _T("error") };
        _stprintf_s(buffer_line, len + 64, "qsv [%s]: %s", LOG_STRING[clamp(log_level, RGY_LOG_TRACE, RGY_LOG_ERROR) - RGY_LOG_TRACE], p);
        write_log_line(log_level, buffer_line);
        p = NULL;
    }
}
#pragma warning(pop)

void CAuoLog::write(int log_level, const TCHAR *format, ... ) {
    if (log_level < m_nLogLevel) {
        return;
    }

    va_list args;
    va_start(args, format);

    int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    TCHAR *buffer = (TCHAR*)calloc((len * 2 + 64), sizeof(buffer[0]));

    _vstprintf_s(buffer, len, format, args);
    write_log(log_level, buffer);

    free(buffer);
}

