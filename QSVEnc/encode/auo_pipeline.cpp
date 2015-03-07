//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

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

mfxStatus AuoPipeline::InitInOut(sInputParams *pParams) {
	mfxStatus sts = MFX_ERR_NONE;


	m_pEncSatusInfo = new AUO_EncodeStatusInfo();

	// prepare input file reader
	m_pFileReader = new AUO_YUVReader();
	sts = m_pFileReader->Init(NULL, NULL, false, &m_EncThread, m_pEncSatusInfo, NULL);
	MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

	m_pFileWriter = new CSmplBitstreamWriter();
	sts = m_pFileWriter->Init(pParams->strDstFile, pParams, m_pEncSatusInfo);
	MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

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
	mfxU32 gcd = GCD(OutputFPSRate, OutputFPSScale);
	OutputFPSRate /= gcd;
	OutputFPSScale /= gcd;

	m_pEncSatusInfo->Init(OutputFPSRate, OutputFPSScale, outputFrames, NULL);

	return sts;
}
#pragma warning( push )
#pragma warning( disable: 4100 )
void AuoPipeline::PrintMes(const TCHAR *format, ... ) {
    va_list args;
    va_start(args, format);

    int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    TCHAR *buffer = (TCHAR*)calloc((len * 2 + 20), sizeof(buffer[0]));
	TCHAR *buffer_line = buffer + len;

    _vstprintf_s(buffer, len, format, args);
	//_ftprintf(fp, buffer);

	TCHAR *q = NULL;
	for (TCHAR *p = buffer; (p = _tcstok_s(p, _T("\n"), &q)) != NULL; ) {
		_stprintf_s(buffer_line, len + 20, "qsv [info]: %s", p);
		write_log_line(LOG_INFO, buffer_line);
		p = NULL;
	}

	free(buffer);
}
#pragma warning( pop )
