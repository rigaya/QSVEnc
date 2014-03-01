/* ////////////////////////////////////////////////////////////////////////////// */
/*
//
//              INTEL CORPORATION PROPRIETARY INFORMATION
//  This software is supplied under the terms of a license  agreement or
//  nondisclosure agreement with Intel Corporation and may not be copied
//  or disclosed except in  accordance  with the terms of that agreement.
//        Copyright (c) 2005-2011 Intel Corporation. All Rights Reserved.
//
//
*/

#ifndef __QSV_CONTROL_H__
#define __QSV_CONTROL_H__

#include <Windows.h>
#include <tchar.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <sstream>
#include <vector>
#include <intrin.h>
#include "mfxstructures.h"
#include "mfxvideo.h"
#include "mfxjpeg.h"
#include "sample_defs.h"
#include "qsv_prm.h"

typedef struct {
	mfxFrameSurface1* pFrameSurface;
	HANDLE heInputStart;
	HANDLE heSubStart;
	HANDLE heInputDone;
	mfxU32 frameFlag;
	int    AQP[2];
	mfxU8 reserved[64-(sizeof(mfxFrameSurface1*)+sizeof(HANDLE)*3+sizeof(mfxU32)+sizeof(int)*2)];
} sInputBufSys;

typedef struct {
	int frameCountI;
	int frameCountP;
	int frameCountB;
	int sumQPI;
	int sumQPP;
	int sumQPB;
} sFrameTypeInfo;

class CQSVFrameTypeSimulation
{
public:
	CQSVFrameTypeSimulation() {
		i_frame = 0;
		BFrames = 0;
		GOPSize = 1;
		openGOP = false;
		maxIdrInterval = 0;
	}
	void Init(int _GOPSize, int _BFrames, int _QPI, int _QPP, int _QPB, bool _openGOP, double frameRate) {
		GOPSize = max(_GOPSize, 1);
		BFrames = max(_BFrames, 0);
		QPI = _QPI;
		QPP = _QPP;
		QPB = _QPB;
		i_frame = 0;
		i_from_last_idr = 0;
		openGOP = _openGOP;
		maxIdrInterval = (int)(frameRate + 0.5) * 20;
		MSDK_ZERO_MEMORY(m_info);
	}
	~CQSVFrameTypeSimulation() {
	}
	mfxU32 GetFrameType(bool I_Insert) {
		mfxU32 ret;
		if (I_Insert || (GOPSize && i_frame % GOPSize == 0)) {
			i_frame = 0;
		}
		if (i_frame == 0) {
			ret = MFX_FRAMETYPE_I | MFX_FRAMETYPE_REF;
			if (!openGOP || i_from_last_idr >= maxIdrInterval || 0 == i_from_last_idr) {
				i_from_last_idr = 0;
				ret |= MFX_FRAMETYPE_IDR;
			}
		} else if ((i_frame - 1) % (BFrames + 1) == BFrames) {
			ret = MFX_FRAMETYPE_P | MFX_FRAMETYPE_REF;
		} else {
			ret = MFX_FRAMETYPE_B;
		}
		return ret;
	}
	void ToNextFrame() {
		i_frame++;
		i_from_last_idr++;
	}
	int CurrentQP(bool I_Insert, int qp_offset) {
		mfxU32 frameType = GetFrameType(I_Insert);
		int qp;
		if (frameType & MFX_FRAMETYPE_I) {
			qp = QPI;
			m_info.sumQPI += qp;
			m_info.frameCountI++;
		} else if (frameType & MFX_FRAMETYPE_P) {
			qp = clamp(QPP + qp_offset, 0, 51);
			m_info.sumQPP += qp;
			m_info.frameCountP++;
		} else {
			qp = clamp(QPB + qp_offset, 0, 51);
			m_info.sumQPB += qp;
			m_info.frameCountB++;
		}
		return qp;
	}
	void getFrameInfo(sFrameTypeInfo *info) {
		memcpy(info, &m_info, sizeof(info[0]));
	}
private:
	int i_frame;
	int i_from_last_idr;

	int GOPSize;
	int BFrames;

	int QPI;
	int QPP;
	int QPB;

	bool openGOP;
	int maxIdrInterval;

	sFrameTypeInfo m_info;
};

typedef struct sEncodeStatusData {
	mfxU32 nProcessedFramesNum;
	mfxU64 nWrittenBytes;
	mfxU32 nIDRCount;
	mfxU32 nICount;
	mfxU32 nPCount;
	mfxU32 nBCount;
	mfxU64 nIFrameSize;
	mfxU64 nPFrameSize;
	mfxU64 nBFrameSize;
	mfxU32 tmStart;
	mfxU32 nTotalOutFrames;
	mfxF64 fEncodeFps;
	mfxF64 fBitrateKbps;
} sEncodeStatusData;

class CEncodeStatusInfo
{
public:
	CEncodeStatusInfo();
	void Init(mfxU32 outputFPSRate, mfxU32 outputFPSScale, mfxU32 totalOutputFrames, TCHAR *pStrLog);
	void SetStart();
	void SetOutputData(mfxU64 nBytesWritten, mfxU32 frameType)
	{
		m_sData.nProcessedFramesNum++;
		m_sData.nWrittenBytes += nBytesWritten;
		m_sData.nIDRCount     += ((frameType & MFX_FRAMETYPE_IDR) >> 7);
		m_sData.nICount       +=  (frameType & MFX_FRAMETYPE_I);
		m_sData.nPCount       += ((frameType & MFX_FRAMETYPE_P) >> 1);
		m_sData.nBCount       += ((frameType & MFX_FRAMETYPE_B) >> 2);
		m_sData.nIFrameSize   += nBytesWritten *  (frameType & MFX_FRAMETYPE_I);
		m_sData.nPFrameSize   += nBytesWritten * ((frameType & MFX_FRAMETYPE_P) >> 1);
		m_sData.nBFrameSize   += nBytesWritten * ((frameType & MFX_FRAMETYPE_B) >> 2);
	}
#pragma warning(push)
#pragma warning(disable:4100)
	virtual void UpdateDisplay(const TCHAR *mes, int drop_frames)
	{
#if UNICODE
		char *mes_char = NULL;
		if (!m_bStdErrWriteToConsole) {
			//コンソールへの出力でなければ、ANSIに変換する
			const int buf_length = (int)(wcslen(mes) + 1) * 2;
			if (NULL != (mes_char = (char *)calloc(buf_length, 1))) {
				WideCharToMultiByte(CP_THREAD_ACP, 0, mes, -1, mes_char, buf_length, NULL, NULL);
				fprintf(stderr, "%s\r", mes_char);
				free(mes_char);
			}
		} else
#endif
			_ftprintf(stderr, _T("%s\r"), mes);

		fflush(stderr); //リダイレクトした場合でもすぐ読み取れるようflush
	}
#pragma warning(pop)
	virtual void UpdateDisplay(mfxU32 tm, int drop_frames)
	{
		if (m_sData.nProcessedFramesNum + drop_frames) {
			TCHAR mes[256];
			m_sData.fEncodeFps = (m_sData.nProcessedFramesNum + drop_frames) * 1000.0 / (double)(tm - m_sData.tmStart);
			m_sData.fBitrateKbps = (mfxF64)m_sData.nWrittenBytes * (m_nOutputFPSRate / (mfxF64)m_nOutputFPSScale) / ((1000 / 8) * (m_sData.nProcessedFramesNum + drop_frames));
			if (m_sData.nTotalOutFrames) {
				mfxU32 remaining_time = (mfxU32)((m_sData.nTotalOutFrames - (m_sData.nProcessedFramesNum + drop_frames)) * 1000.0 / ((m_sData.nProcessedFramesNum + drop_frames) * 1000.0 / (mfxF64)(tm - m_sData.tmStart)));
				int hh = remaining_time / (60*60*1000);
				remaining_time -= hh * (60*60*1000);
				int mm = remaining_time / (60*1000);
				remaining_time -= mm * (60*1000);
				int ss = (remaining_time + 500) / 1000;

				int len = _stprintf_s(mes, _countof(mes), _T("[%.1lf%%] %d frames: %.2lf fps, %0.2lf kb/s, remain %d:%02d:%02d  "),
					(m_sData.nProcessedFramesNum + drop_frames) * 100 / (mfxF64)m_sData.nTotalOutFrames,
					(m_sData.nProcessedFramesNum + drop_frames),
					m_sData.fEncodeFps,
					m_sData.fBitrateKbps,
					hh, mm, ss );
				if (drop_frames)
					_stprintf_s(mes + len - 2, _countof(mes) - len + 2, _T(", afs drop %d/%d  "), drop_frames, (m_sData.nProcessedFramesNum + drop_frames));
			} else {
				_stprintf_s(mes, _countof(mes), _T("%d frames: %0.2lf fps, %0.2lf kbps  "), 
					(m_sData.nProcessedFramesNum + drop_frames),
					m_sData.fEncodeFps,
					m_sData.fBitrateKbps
					);
			}
			UpdateDisplay(mes, drop_frames);
		}
	}
	virtual void WriteLine(const TCHAR *mes) {
#ifdef UNICODE
		char *mes_char = NULL;
		if (m_pStrLog || !m_bStdErrWriteToConsole) {
			int buf_len = (int)wcslen(mes) + 1;
			if (NULL != (mes_char = (char *)calloc(buf_len * 2, sizeof(mes_char[0]))))
				WideCharToMultiByte(CP_THREAD_ACP, WC_NO_BEST_FIT_CHARS, mes, -1, mes_char, buf_len * 2, NULL, NULL);
		}
		if (mes_char) {
#else
			const char *mes_char = mes;
#endif
			if (m_pStrLog) {
				FILE *fp_log = NULL;
				if (0 == _tfopen_s(&fp_log, m_pStrLog, _T("a")) && fp_log) {
					fprintf(fp_log, "%s\n", mes_char);
					fclose(fp_log);
				}
			}
#ifdef UNICODE
			if (m_bStdErrWriteToConsole)
				_ftprintf(stderr, _T("%s\n"), mes); //出力先がコンソールならWCHARで
			else
#endif
				fprintf(stderr, "%s\n", mes_char); //出力先がリダイレクトされるならANSIで
#ifdef UNICODE
			free(mes_char);
		}
#endif
	}
	virtual void WriteFrameTypeResult(const TCHAR *header, mfxU32 count, mfxU32 maxCount, mfxU64 frameSize, mfxU64 maxFrameSize, double avgQP) {
		if (count) {
			TCHAR mes[512] = { 0 };
			int mes_len = 0;
			const int header_len = (int)_tcslen(header);
			memcpy(mes, header, header_len * sizeof(mes[0]));
			mes_len += header_len;

			for (int i = max(0, (int)log10((double)count)); i < (int)log10((double)maxCount) && mes_len < _countof(mes); i++, mes_len++)
				mes[mes_len] = _T(' ');
			mes_len += _stprintf_s(mes + mes_len, _countof(mes) - mes_len, _T("%u"), count);

			if (avgQP >= 0.0) {
				mes_len += _stprintf_s(mes + mes_len, _countof(mes) - mes_len, _T(",  avgQP  %4.2f"), avgQP);
			}
			
			if (frameSize > 0) {
				const TCHAR *TOTAL_SIZE = _T(",  total size  ");
				memcpy(mes + mes_len, TOTAL_SIZE, _tcslen(TOTAL_SIZE) * sizeof(mes[0]));
				mes_len += (int)_tcslen(TOTAL_SIZE);

				for (int i = max(0, (int)log10((double)frameSize / (double)(1024 * 1024))); i < (int)log10((double)maxFrameSize / (double)(1024 * 1024)) && mes_len < _countof(mes); i++, mes_len++)
					mes[mes_len] = _T(' ');

				mes_len += _stprintf_s(mes + mes_len, _countof(mes) - mes_len, _T("%.2f MB"), (double)frameSize / (double)(1024 * 1024));
			}

			WriteLine(mes);
		}
	}
	virtual void WriteResults(sFrameTypeInfo *info)
	{
		mfxU32 tm_result = timeGetTime();
		mfxU32 time_elapsed = tm_result - m_sData.tmStart;
		m_sData.fEncodeFps = m_sData.nProcessedFramesNum * 1000.0 / (double)time_elapsed;
		m_sData.fBitrateKbps = (mfxF64)(m_sData.nWrittenBytes * 8) *  (m_nOutputFPSRate / (double)m_nOutputFPSScale) / (1000.0 * m_sData.nProcessedFramesNum);

		TCHAR mes[512] = { 0 };
		for (int i = 0; i < 79; i++)
			mes[i] = ' ';
		WriteLine(mes);

		_stprintf_s(mes, _countof(mes), _T("encoded %d frames, %.2f fps, %.2f kbps, %.2f MB"),
			m_sData.nProcessedFramesNum,
			m_sData.fEncodeFps,
			m_sData.fBitrateKbps,
			(double)m_sData.nWrittenBytes / (double)(1024 * 1024)
			);
		WriteLine(mes);

		int hh = time_elapsed / (60*60*1000);
		time_elapsed -= hh * (60*60*1000);
		int mm = time_elapsed / (60*1000);
		time_elapsed -= mm * (60*1000);
		int ss = (time_elapsed + 500) / 1000;
		_stprintf_s(mes, _countof(mes), _T("encode time %d:%02d:%02d\n"), hh, mm, ss);
		WriteLine(mes);

		mfxU32 maxCount = MAX3(m_sData.nICount, m_sData.nPCount, m_sData.nBCount);
		mfxU64 maxFrameSize = MAX3(m_sData.nIFrameSize, m_sData.nPFrameSize, m_sData.nBFrameSize);

		WriteFrameTypeResult(_T("frame type IDR "), m_sData.nIDRCount, maxCount,                   0, maxFrameSize, -1.0);
		WriteFrameTypeResult(_T("frame type I   "), m_sData.nICount,   maxCount, m_sData.nIFrameSize, maxFrameSize, (info) ? info->sumQPI / (double)info->frameCountI : -1);
		WriteFrameTypeResult(_T("frame type P   "), m_sData.nPCount,   maxCount, m_sData.nPFrameSize, maxFrameSize, (info) ? info->sumQPP / (double)info->frameCountP : -1);
		WriteFrameTypeResult(_T("frame type B   "), m_sData.nBCount,   maxCount, m_sData.nBFrameSize, maxFrameSize, (info) ? info->sumQPB / (double)info->frameCountB : -1);
	}
	mfxU32 m_nInputFrames;
	mfxU32 m_nTotalOutFrames;
	mfxU32 m_nOutputFPSRate;
	mfxU32 m_nOutputFPSScale;
protected:
	sEncodeStatusData m_sData;
	TCHAR *m_pStrLog;
	bool m_bStdErrWriteToConsole;
};

class CEncodingThread 
{
public:
	CEncodingThread();
	~CEncodingThread();

	mfxStatus Init(mfxU16 bufferSize);
	void Close();
	//終了を待機する
	mfxStatus WaitToFinish(mfxStatus sts);
	mfxU16    GetBufferSize();
	mfxStatus RunEncFuncbyThread(unsigned (__stdcall * func) (void *), void *pClass, DWORD_PTR threadAffinityMask);
	mfxStatus RunSubFuncbyThread(unsigned (__stdcall * func) (void *), void *pClass, DWORD_PTR threadAffinityMask);

	HANDLE GetHandleEncThread() {
		return m_thEncode;
	}
	HANDLE GetHandleSubThread() {
		return m_thSub;
	}

	BOOL m_bthForceAbort;
	BOOL m_bthSubAbort;
	sInputBufSys *m_InputBuf;
	mfxU32 m_nFrameSet;
	mfxU32 m_nFrameGet;
	mfxStatus m_stsThread;
	mfxU16  m_nFrameBuffer;
protected:
	HANDLE m_thEncode;
	HANDLE m_thSub;
	bool m_bInit;
};

#endif //__QSV_CONTROL_H__
