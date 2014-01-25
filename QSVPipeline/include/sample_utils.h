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

#ifndef __SAMPLE_UTILS_H__
#define __SAMPLE_UTILS_H__

#include <Windows.h>
#include <tchar.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <vector>
#include <intrin.h>
#include "mfxstructures.h"
#include "mfxvideo.h"
#include "mfxjpeg.h"
#include "sample_defs.h"
#include "qsv_prm.h"

#ifndef MIN3
#define MIN3(a,b,c) (min((a), min((b), (c))))
#endif
#ifndef MAX3
#define MAX3(a,b,c) (max((a), max((b), (c))))
#endif

// A macro to disallow the copy constructor and operator= functions
// This should be used in the private: declarations for a class
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName&);               \
    void operator=(const TypeName&)

//! Base class for types that should not be assigned.
class no_assign {
    // Deny assignment
    void operator=( const no_assign& );
public:
};

//! Base class for types that should not be copied or assigned.
class no_copy: no_assign {
    //! Deny copy construction
    no_copy( const no_copy& );
public:
    //! Allow default construction
    no_copy() {}
};

static void __forceinline sse_memcpy(BYTE *dst, const BYTE *src, int size) {
	BYTE *dst_fin = dst + size;
	BYTE *dst_aligned_fin = (BYTE *)(((size_t)dst_fin & ~15) - 64);
	__m128 x0, x1, x2, x3;
	const int start_align_diff = (int)((size_t)dst & 15);
	if (start_align_diff) {
		x0 = _mm_loadu_ps((float*)src);
		_mm_storeu_ps((float*)dst, x0);
		dst += start_align_diff;
		src += start_align_diff;
	}
	for ( ; dst < dst_aligned_fin; dst += 64, src += 64) {
		x0 = _mm_loadu_ps((float*)(src +  0));
		x1 = _mm_loadu_ps((float*)(src + 16));
		x2 = _mm_loadu_ps((float*)(src + 32));
		x3 = _mm_loadu_ps((float*)(src + 48));
		_mm_store_ps((float*)(dst +  0), x0);
		_mm_store_ps((float*)(dst + 16), x1);
		_mm_store_ps((float*)(dst + 32), x2);
		_mm_store_ps((float*)(dst + 48), x3);
	}
	BYTE *dst_tmp = dst_fin - 64;
	src -= (dst - dst_tmp);
	x0 = _mm_loadu_ps((float*)(src +  0));
	x1 = _mm_loadu_ps((float*)(src + 16));
	x2 = _mm_loadu_ps((float*)(src + 32));
	x3 = _mm_loadu_ps((float*)(src + 48));
	_mm_storeu_ps((float*)(dst_tmp +  0), x0);
	_mm_storeu_ps((float*)(dst_tmp + 16), x1);
	_mm_storeu_ps((float*)(dst_tmp + 32), x2);
	_mm_storeu_ps((float*)(dst_tmp + 48), x3);
}

static bool isHaswellOrLater() {
	//単純にAVX2フラグを見る
	int CPUInfo[4];
	__cpuid(CPUInfo, 7);
	return ((CPUInfo[1] & 0x00000020) == 0x00000020);
}

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
	}
	void Init(int _GOPSize, int _BFrames, int _QPI, int _QPP, int _QPB) {
		GOPSize = max(_GOPSize, 1);
		BFrames = max(_BFrames, 0);
		QPI = _QPI;
		QPP = _QPP;
		QPB = _QPB;
		i_frame = 0;
		MSDK_ZERO_MEMORY(m_info);
	}
	~CQSVFrameTypeSimulation() {
	}
	mfxU32 GetFrameType(bool IdrInsert) {
		mfxU32 ret;
		if (IdrInsert || (GOPSize && i_frame % GOPSize == 0))
			i_frame = 0;
		if (i_frame == 0)
			ret = MFX_FRAMETYPE_IDR | MFX_FRAMETYPE_I | MFX_FRAMETYPE_REF;
		else if ((i_frame - 1) % (BFrames + 1) == BFrames)
			ret = MFX_FRAMETYPE_P | MFX_FRAMETYPE_REF;
		else
			ret = MFX_FRAMETYPE_B;
		return ret;
	}
	void ToNextFrame() {
		i_frame++;
	}
	int CurrentQP(bool IdrInsert, int qp_offset) {
		mfxU32 frameType = GetFrameType(IdrInsert);
		int qp;
		if (frameType & (MFX_FRAMETYPE_IDR | MFX_FRAMETYPE_I)) {
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

	int GOPSize;
	int BFrames;

	int QPI;
	int QPP;
	int QPB;

	sFrameTypeInfo m_info;
};

class CEncodeStatusInfo
{
public:
	CEncodeStatusInfo();
	void Init(mfxU32 outputFPSRate, mfxU32 outputFPSScale, mfxU32 totalOutputFrames, TCHAR *pStrLog);
	void SetStart();
	void SetOutputData(mfxU64 nBytesWritten, mfxU32 frameType)
	{
		m_nWrittenBytes += nBytesWritten;
		m_nProcessedFramesNum++;
		m_nIDRCount += ((frameType & MFX_FRAMETYPE_IDR) >> 7);
		m_nICount   +=  (frameType & MFX_FRAMETYPE_I);
		m_nPCount   += ((frameType & MFX_FRAMETYPE_P) >> 1);
		m_nBCount   += ((frameType & MFX_FRAMETYPE_B) >> 2);
		m_nIFrameSize += nBytesWritten *  (frameType & MFX_FRAMETYPE_I);
		m_nPFrameSize += nBytesWritten * ((frameType & MFX_FRAMETYPE_P) >> 1);
		m_nBFrameSize += nBytesWritten * ((frameType & MFX_FRAMETYPE_B) >> 2);
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
	}
#pragma warning(pop)
	virtual void UpdateDisplay(mfxU32 tm, int drop_frames)
	{
		if (m_nProcessedFramesNum + drop_frames) {
			TCHAR mes[256];
			mfxF64 encode_fps = (m_nProcessedFramesNum + drop_frames) * 1000.0 / (double)(tm - m_tmStart);
			if (m_nTotalOutFrames) {
				mfxU32 remaining_time = (mfxU32)((m_nTotalOutFrames - (m_nProcessedFramesNum + drop_frames)) * 1000.0 / ((m_nProcessedFramesNum + drop_frames) * 1000.0 / (mfxF64)(tm - m_tmStart)));
				int hh = remaining_time / (60*60*1000);
				remaining_time -= hh * (60*60*1000);
				int mm = remaining_time / (60*1000);
				remaining_time -= mm * (60*1000);
				int ss = (remaining_time + 500) / 1000;

				int len = _stprintf_s(mes, _countof(mes), _T("[%.1lf%%] %d frames: %.2lf fps, %0.2lf kb/s, remain %d:%02d:%02d  "),
					(m_nProcessedFramesNum + drop_frames) * 100 / (mfxF64)m_nTotalOutFrames,
					(m_nProcessedFramesNum + drop_frames),
					encode_fps,
					(mfxF64)m_nWrittenBytes * (m_nOutputFPSRate / (mfxF64)m_nOutputFPSScale) / ((1000 / 8) * (m_nProcessedFramesNum + drop_frames)),
					hh, mm, ss );
				if (drop_frames)
					_stprintf_s(mes + len - 2, _countof(mes) - len + 2, _T(", afs drop %d/%d  "), drop_frames, (m_nProcessedFramesNum + drop_frames));
			} else {
				_stprintf_s(mes, _countof(mes), _T("%d frames: %0.2lf fps, %0.2lf kbps  "), 
					(m_nProcessedFramesNum + drop_frames),
					encode_fps,
					(mfxF64)(m_nWrittenBytes * 8) * (m_nOutputFPSRate / (mfxF64)m_nOutputFPSScale) / (1000.0 * (m_nProcessedFramesNum + drop_frames))
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
		mfxU32 time_elapsed = tm_result - m_tmStart;
		mfxF64 encode_fps = m_nProcessedFramesNum * 1000.0 / (double)time_elapsed;

		TCHAR mes[512] = { 0 };
		for (int i = 0; i < 79; i++)
			mes[i] = ' ';
		WriteLine(mes);

		_stprintf_s(mes, _countof(mes), _T("encoded %d frames, %.2f fps, %.2f kbps, %.2f MB"),
			m_nProcessedFramesNum,
			encode_fps,
			(mfxF64)(m_nWrittenBytes * 8) *  (m_nOutputFPSRate / (double)m_nOutputFPSScale) / (1000.0 * m_nProcessedFramesNum),
			(double)m_nWrittenBytes / (double)(1024 * 1024)
			);
		WriteLine(mes);

		int hh = time_elapsed / (60*60*1000);
		time_elapsed -= hh * (60*60*1000);
		int mm = time_elapsed / (60*1000);
		time_elapsed -= mm * (60*1000);
		int ss = (time_elapsed + 500) / 1000;
		_stprintf_s(mes, _countof(mes), _T("encode time %d:%02d:%02d\n"), hh, mm, ss);
		WriteLine(mes);

		mfxU32 maxCount = MAX3(m_nICount, m_nPCount, m_nBCount);
		mfxU64 maxFrameSize = MAX3(m_nIFrameSize, m_nPFrameSize, m_nBFrameSize);

		WriteFrameTypeResult(_T("frame type IDR "), m_nIDRCount, maxCount,             0, maxFrameSize, -1.0);
		WriteFrameTypeResult(_T("frame type I   "), m_nICount,   maxCount, m_nIFrameSize, maxFrameSize, (info) ? info->sumQPI / (double)info->frameCountI : -1);
		WriteFrameTypeResult(_T("frame type P   "), m_nPCount,   maxCount, m_nPFrameSize, maxFrameSize, (info) ? info->sumQPP / (double)info->frameCountP : -1);
		WriteFrameTypeResult(_T("frame type B   "), m_nBCount,   maxCount, m_nBFrameSize, maxFrameSize, (info) ? info->sumQPB / (double)info->frameCountB : -1);
	}
	mfxU32 m_nInputFrames;
	mfxU32 m_nOutputFPSRate;
	mfxU32 m_nOutputFPSScale;
protected:
	mfxU32 m_nProcessedFramesNum;
	mfxU64 m_nWrittenBytes;
	mfxU32 m_nIDRCount;
	mfxU32 m_nICount;
	mfxU32 m_nPCount;
	mfxU32 m_nBCount;
	mfxU64 m_nIFrameSize;
	mfxU64 m_nPFrameSize;
	mfxU64 m_nBFrameSize;
	mfxU32 m_tmStart;
	mfxU32 m_nTotalOutFrames;
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

class CSmplYUVReader
{
public :

	CSmplYUVReader();
	virtual ~CSmplYUVReader();

	virtual mfxStatus Init(const TCHAR *strFileName, mfxU32 ColorFormat, int option, CEncodingThread *pEncThread, CEncodeStatusInfo *pEncSatusInfo, sInputCrop *pInputCrop);

	//この関数がMFX_ERR_NONE以外を返すことでRunEncodeは終了処理に入る
	mfxStatus GetNextFrame(mfxFrameSurface1** pSurface)
	{
#ifdef _DEBUG
		MSDK_CHECK_POINTER(pSurface, MFX_ERR_NULL_PTR);
		MSDK_CHECK_POINTER(m_pEncThread, MFX_ERR_NULL_PTR);
#endif
		const int inputBufIdx = m_pEncThread->m_nFrameGet % m_pEncThread->m_nFrameBuffer;
		sInputBufSys *pInputBuf = &m_pEncThread->m_InputBuf[inputBufIdx];

		//_ftprintf(stderr, "GetNextFrame: wait for %d\n", m_pEncThread->m_nFrameGet);
		//_ftprintf(stderr, "wait for heInputDone, %d\n", m_pEncThread->m_nFrameGet);
		WaitForSingleObject(pInputBuf->heInputDone, INFINITE);
		//エラー・中断要求などでの終了
		if (m_pEncThread->m_bthForceAbort)
			return m_pEncThread->m_stsThread;
		//読み込み完了による終了
		if (m_pEncThread->m_stsThread == MFX_ERR_MORE_DATA && m_pEncThread->m_nFrameGet == m_pEncSatusInfo->m_nInputFrames)
			return m_pEncThread->m_stsThread;
		*pSurface = pInputBuf->pFrameSurface;
		(*pSurface)->Data.TimeStamp = inputBufIdx;
		(*pSurface)->Data.Locked = FALSE;
		m_pEncThread->m_nFrameGet++;
		return MFX_ERR_NONE;
	}


	mfxStatus SetNextSurface(mfxFrameSurface1* pSurface)
	{
#ifdef _DEBUG
		MSDK_CHECK_POINTER(pSurface, MFX_ERR_NULL_PTR);
		MSDK_CHECK_POINTER(m_pEncThread, MFX_ERR_NULL_PTR);
#endif
		const int inputBufIdx = m_pEncThread->m_nFrameSet % m_pEncThread->m_nFrameBuffer;
		sInputBufSys *pInputBuf = &m_pEncThread->m_InputBuf[inputBufIdx];
		//_ftprintf(stderr, "Set heInputStart: %d\n", m_pEncThread->m_nFrameSet);
		pSurface->Data.Locked = TRUE;
		//_ftprintf(stderr, "set surface %d, set event heInputStart %d\n", pSurface, m_pEncThread->m_nFrameSet);
		pInputBuf->pFrameSurface = pSurface;
		SetEvent(pInputBuf->heInputStart);
		m_pEncThread->m_nFrameSet++;
		return MFX_ERR_NONE;
	}

	virtual void Close();
	//virtual mfxStatus Init(const TCHAR *strFileName, const mfxU32 ColorFormat, const mfxU32 numViews, std::vector<TCHAR*> srcFileBuff);
	virtual mfxStatus LoadNextFrame(mfxFrameSurface1* pSurface);
	mfxU32 m_ColorFormat; // color format of input YUV data, YUV420 or NV12
	void GetInputCropInfo(sInputCrop *cropInfo) {
		memcpy(cropInfo, &m_sInputCrop, sizeof(m_sInputCrop));
	}
	void GetInputFrameInfo(mfxFrameInfo *inputFrameInfo) {
		memcpy(inputFrameInfo, &m_inputFrameInfo, sizeof(m_inputFrameInfo));
	}
	const TCHAR *GetInputMessage() {
		const TCHAR *mes = m_strInputInfo.c_str();
		return (mes) ? mes : _T("");
	}
#if ENABLE_MVC_ENCODING
	void SetMultiView() { m_bIsMultiView = true; }
#endif
protected:
	FILE *m_fSource;
#if ENABLE_MVC_ENCODING
	FILE **m_fSourceMVC;
	bool m_bIsMultiView;
	mfxU32 m_numLoadedFiles;
#endif
	CEncodingThread *m_pEncThread;
	CEncodeStatusInfo *m_pEncSatusInfo;
	bool m_by4m;
	bool m_bInited;
	mfxU32 m_tmLastUpdate;
	sInputCrop m_sInputCrop;

	mfxFrameInfo m_inputFrameInfo;

	mfxU32 bufSize;
	mfxU8 *buffer;

	std::basic_string<TCHAR> m_strInputInfo;
};

class CSmplBitstreamWriter
{
public :

	CSmplBitstreamWriter();
	virtual ~CSmplBitstreamWriter();

	virtual mfxStatus Init(const TCHAR *strFileName, sInputParams *prm, CEncodeStatusInfo *pEncSatusInfo);

	virtual mfxStatus SetVideoParam(mfxVideoParam *pMfxVideoPrm);

	virtual mfxStatus WriteNextFrame(mfxBitstream *pMfxBitstream);
	virtual void Close();

protected:
	CEncodeStatusInfo *m_pEncSatusInfo;
	FILE*       m_fSource;
	bool        m_bInited;
};

class CSmplYUVWriter
{
public :

	CSmplYUVWriter();
	virtual ~CSmplYUVWriter();

	virtual void      Close();
	virtual mfxStatus Init(const TCHAR *strFileName, const mfxU32 numViews);
	virtual mfxStatus WriteNextFrame(mfxFrameSurface1 *pSurface);

	void SetMultiView() { m_bIsMultiView = true; }

protected:
	FILE         *m_fDest, **m_fDestMVC;
	bool         m_bInited, m_bIsMultiView;
	mfxU32       m_numCreatedFiles;
};

class CSmplBitstreamReader
{
public :

	CSmplBitstreamReader();
	virtual ~CSmplBitstreamReader();

	//resets position to file begin
	virtual void      Reset();
	virtual void      Close();
	virtual mfxStatus Init(const TCHAR *strFileName);
	virtual mfxStatus ReadNextFrame(mfxBitstream *pBS);

protected:
	FILE*     m_fSource;
	bool      m_bInited;
};

//provides output bistream with at least 1 slice, reports about error
class CH264FrameReader : public CSmplBitstreamReader
{
public:
	CH264FrameReader();
	virtual mfxStatus ReadNextFrame(mfxBitstream *pBS);
protected:
	//1 - means slice start indicator present
	//2 - means slice start and backend startcode present
	int FindSlice(mfxBitstream *pBS, int & pos2ndnalu);


	mfxBitstream m_lastBs;
	std::vector<mfxU8> m_bsBuffer;
};

//provides output bistream with at least 1 frame, reports about error
class CJPEGFrameReader : public CSmplBitstreamReader
{
public:
	virtual mfxStatus ReadNextFrame(mfxBitstream *pBS);
protected:
	//1 - means SOI marker present
	//2 - means 2 SOI markers present
	int FindSOImarkers(mfxBitstream *pBS);
};

//timeinterval calculation helper
#ifdef WIN32
#include "windows.h"
#endif

template <int tag = 0>
class CTimeInterval : private no_copy
{
	static double g_Freq;
	double       &m_start;
	double        m_own;//reference to this if external counter not required
	//since QPC functions are quite slow it make sense to optionally enable them
	bool         m_bEnable;
#ifdef  WIN32
	LARGE_INTEGER m_liStart;
#endif

public:
	CTimeInterval(double &dRef , bool bEnable = true)
		: m_start(dRef)
		, m_bEnable(bEnable)
	{
		if (!m_bEnable)
			return;
		Initialize();
	}
	CTimeInterval(bool bEnable = true)
		: m_start(m_own)
		, m_bEnable(bEnable)
		, m_own()
	{
		if (!m_bEnable)
			return;
		Initialize();
	}

	//updates external value with current time
	double Commit()
	{
		if (!m_bEnable)
			return 0.0;

		if (0.0 != g_Freq)
		{
#ifdef  WIN32
			LARGE_INTEGER liEnd;
			QueryPerformanceCounter(&liEnd);
			m_start = ((double)liEnd.QuadPart - (double)m_liStart.QuadPart)  / g_Freq;
#endif
		}
		return m_start;
	}
	//lastcomitted value
	double Last()
	{
		return m_start;
	}
	~CTimeInterval()
	{
		Commit();
	}
private:
	void Initialize()
	{
#ifdef  WIN32
		if (0.0 == g_Freq)
		{
			QueryPerformanceFrequency(&m_liStart);
			g_Freq = (double)m_liStart.QuadPart;
		}
		QueryPerformanceCounter(&m_liStart);
#endif
	}
};

template <int tag>double CTimeInterval<tag>::g_Freq = 0.0f;


mfxStatus ConvertFrameRate(mfxF64 dFrameRate, mfxU32* pnFrameRateExtN, mfxU32* pnFrameRateExtD);
mfxF64 CalculateFrameRate(mfxU32 nFrameRateExtN, mfxU32 nFrameRateExtD);

static inline mfxU16 GetFreeSurface(mfxFrameSurface1* pSurfacesPool, mfxU16 nPoolSize) {
    static const int SleepInterval = 1; // milliseconds
    //wait if there's no free surface
    for (mfxU32 j = 0; j < MSDK_WAIT_INTERVAL; j += SleepInterval) {
		for (mfxU16 i = 0; i < nPoolSize; i++)
			if (0 == pSurfacesPool[i].Data.Locked)
                return i;
        MSDK_SLEEP(SleepInterval);
    }
    return MSDK_INVALID_SURF_IDX;
}

static inline mfxU16 GetFreeSurfaceIndex(mfxFrameSurface1* pSurfacesPool, mfxU16 nPoolSize, mfxU16 step)
{
	if (pSurfacesPool)
	{
		for (mfxU16 i = 0; i < nPoolSize; i = (mfxU16)(i + step), pSurfacesPool += step)
		{
			if (0 == pSurfacesPool[0].Data.Locked)
			{
				return i;
			}
		}
	}

	return MSDK_INVALID_SURF_IDX;
}
mfxStatus InitMfxBitstream(mfxBitstream* pBitstream, mfxU32 nSize);
//performs copy to end if possible also move data to buffer begin if necessary
//shifts offset pointer in source bitstream in success case
mfxStatus MoveMfxBitstream(mfxBitstream *pTarget, mfxBitstream *pSrc, mfxU32 nBytesToCopy);
mfxStatus ExtendMfxBitstream(mfxBitstream* pBitstream, mfxU32 nSize);
void WipeMfxBitstream(mfxBitstream* pBitstream);
const TCHAR* CodecIdToStr(mfxU32 nFourCC);
mfxU16 CalculateDefaultBitrate(mfxU32 nCodecId, mfxU32 nTargetUsage, mfxU32 nWidth, mfxU32 nHeight, mfxF64 dFrameRate);
mfxU16 StrToTargetUsage(TCHAR* strInput);
const TCHAR* TargetUsageToStr(mfxU16 tu);
const TCHAR* ColorFormatToStr(mfxU32 format);
const TCHAR* MfxStatusToStr(mfxStatus sts);
const TCHAR* EncmodeToStr(mfxU32 enc_mode);
const TCHAR* MemTypeToStr(mfxU32 memType);
mfxU32 GCD(mfxU32 a, mfxU32 b);
mfxI64 GCDI64(mfxI64 a, mfxI64 b);

mfxStatus ParseY4MHeader(char *buf, mfxFrameInfo *info);

// sets bitstream->PicStruct parsing first APP0 marker in bitstream
mfxStatus MJPEG_AVI_ParsePicStruct(mfxBitstream *bitstream);

// For MVC encoding/decoding purposes
std::basic_string<TCHAR> FormMVCFileName(const TCHAR *strFileName, const mfxU32 numView);

//piecewise linear function for bitrate approximation
class PartiallyLinearFNC
{
	mfxF64 *m_pX;
	mfxF64 *m_pY;
	mfxU32  m_nPoints;
	mfxU32  m_nAllocated;

public:
	PartiallyLinearFNC();
	~PartiallyLinearFNC();

	void AddPair(mfxF64 x, mfxF64 y);
	mfxF64 at(mfxF64);
private:
	DISALLOW_COPY_AND_ASSIGN(PartiallyLinearFNC);
};

// function for conversion of display aspect ratio to pixel aspect ratio
mfxStatus DARtoPAR(mfxU32 darw, mfxU32 darh, mfxU32 w, mfxU32 h, mfxU16 *pparw, mfxU16 *pparh);

// function for getting a pointer to a specific external buffer from the array
mfxExtBuffer* GetExtBuffer(mfxExtBuffer** ebuffers, mfxU32 nbuffers, mfxU32 BufferId);

//declare used extended buffers
template<class T>
struct mfx_ext_buffer_id{
};
template<>struct mfx_ext_buffer_id<mfxExtCodingOption>{
	enum {id = MFX_EXTBUFF_CODING_OPTION};
};
template<>struct mfx_ext_buffer_id<mfxExtCodingOption2>{
    enum {id = MFX_EXTBUFF_CODING_OPTION2};
};
template<>struct mfx_ext_buffer_id<mfxExtAvcTemporalLayers>{
	enum {id = MFX_EXTBUFF_AVC_TEMPORAL_LAYERS};
};
template<>struct mfx_ext_buffer_id<mfxExtAVCRefListCtrl>{
	enum {id = MFX_EXTBUFF_AVC_REFLIST_CTRL};
};

//helper function to initialize mfx ext buffer structure
template <class T>
void init_ext_buffer(T & ext_buffer)
{
	memset(&ext_buffer, 0, sizeof(ext_buffer));
	reinterpret_cast<mfxExtBuffer*>(&ext_buffer)->BufferId = mfx_ext_buffer_id<T>::id;
	reinterpret_cast<mfxExtBuffer*>(&ext_buffer)->BufferSz = sizeof(ext_buffer);
}

// returns false if buf length is insufficient, otherwise 
// skips step bytes in buf with specified length and returns true
template <typename Buf_t, typename Length_t>
bool skip(const Buf_t *&buf, Length_t &length, Length_t step)
{
	if (length < step)
		return false;

	buf    += step;
	length -= step;

	return true;
}

// returns the number of adapter associated with MSDK session, 0 for SW session
mfxU32 GetMSDKAdapterNumber(mfxSession session = 0);

struct APIChangeFeatures {
    bool JpegDecode;
    bool JpegEncode;
    bool MVCDecode;
    bool MVCEncode;
    bool IntraRefresh;
    bool LowLatency;
    bool ViewOutput;
    bool LookAheadBRC;
};

mfxVersion getMinimalRequiredVersion(const APIChangeFeatures &features);
void ConfigureAspectRatioConversion(mfxInfoVPP* pVppInfo);

#endif //__SAMPLE_UTILS_H__
