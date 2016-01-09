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

#include "qsv_tchar.h"
#include <stdio.h>
#include <math.h>
#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <atomic>
#include <thread>
#include "mfxstructures.h"
#include "mfxvideo.h"
#include "mfxjpeg.h"
#include "qsv_prm.h"
#include "qsv_util.h"
#include "qsv_log.h"
#include "cpu_info.h"
#include "gpuz_info.h"

static const int UPDATE_INTERVAL = 800;
const uint32_t MSDK_DEC_WAIT_INTERVAL = 60000;
const uint32_t MSDK_ENC_WAIT_INTERVAL = 10000;
const uint32_t MSDK_VPP_WAIT_INTERVAL = 60000;
const uint32_t MSDK_WAIT_INTERVAL = MSDK_DEC_WAIT_INTERVAL+3*MSDK_VPP_WAIT_INTERVAL+MSDK_ENC_WAIT_INTERVAL; // an estimate for the longest pipeline we have in samples

const uint32_t MSDK_INVALID_SURF_IDX = 0xFFFF;

using std::chrono::duration_cast;
using std::shared_ptr;
class CEncodingPipeline;

typedef struct {
    mfxFrameSurface1* pFrameSurface;
    HANDLE heInputStart;
    HANDLE heSubStart;
    HANDLE heInputDone;
    std::atomic<uint32_t> frameFlag;
    std::atomic_int AQP[2];
    mfxU8 reserved[64-(sizeof(mfxFrameSurface1*)+sizeof(HANDLE)*3+sizeof(std::atomic<uint32_t>)+sizeof(std::atomic_int)*2)];
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
        GOPSize = (std::max)(_GOPSize, 1);
        BFrames = (std::max)(_BFrames, 0);
        QPI = _QPI;
        QPP = _QPP;
        QPB = _QPB;
        i_frame = 0;
        i_from_last_idr = 0;
        openGOP = _openGOP;
        maxIdrInterval = (int)(frameRate + 0.5) * 20;
        QSV_MEMSET_ZERO(m_info);
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
    uint32_t nProcessedFramesNum;
    uint64_t nWrittenBytes;
    uint32_t nIDRCount;
    uint32_t nICount;
    uint32_t nPCount;
    uint32_t nBCount;
    uint64_t nIFrameSize;
    uint64_t nPFrameSize;
    uint64_t nBFrameSize;
    double   fEncodeFps;
    double   fBitrateKbps;
    double   fCPUUsagePercent;
    int      nGPUInfoCountSuccess;
    int      nGPUInfoCountFail;
    double   fGPULoadPercentTotal;
    double   fGPUClockTotal;
} sEncodeStatusData;

class CEncodeStatusInfo
{
public:
    CEncodeStatusInfo();
    virtual ~CEncodeStatusInfo();
    virtual void Init(mfxU32 outputFPSRate, mfxU32 outputFPSScale, mfxU32 totalOutputFrames, shared_ptr<CQSVLog> pQSVLog);
    void SetStart();
    void GetEncodeData(sEncodeStatusData *data) {
        if (NULL != data) {
            memcpy(data, &m_sData, sizeof(sEncodeStatusData));
        }
    }
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
    virtual void SetPrivData(void *pPrivateData) {};
    virtual void UpdateDisplay(const TCHAR *mes, int drop_frames, double progressPercent)
    {
        if (m_pQSVLog != nullptr && m_pQSVLog->getLogLevel() > QSV_LOG_INFO) {
            return;
        }
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
    virtual mfxStatus UpdateDisplay(int drop_frames, double progressPercent = 0.0)
    {
        if (m_pQSVLog != nullptr && m_pQSVLog->getLogLevel() > QSV_LOG_INFO) {
            return MFX_ERR_NONE;
        }
        if (m_sData.nProcessedFramesNum + drop_frames <= 0) {
            return MFX_ERR_NONE;
        }
        auto tm = std::chrono::system_clock::now();
        if (duration_cast<std::chrono::milliseconds>(tm - m_tmLastUpdate).count() < UPDATE_INTERVAL) {
            return MFX_ERR_NONE;
        }
        m_tmLastUpdate = tm;
        TCHAR mes[256];
        double elapsedTime = (double)duration_cast<std::chrono::milliseconds>(tm - m_tmStart).count();
        m_sData.fEncodeFps = (m_sData.nProcessedFramesNum + drop_frames) * 1000.0 / elapsedTime;
        m_sData.fBitrateKbps = (mfxF64)m_sData.nWrittenBytes * (m_nOutputFPSRate / (mfxF64)m_nOutputFPSScale) / ((1000 / 8) * (m_sData.nProcessedFramesNum + drop_frames));
        if (m_nTotalOutFrames || progressPercent > 0.0) {
            if (progressPercent == 0.0) {
                progressPercent = (m_sData.nProcessedFramesNum + drop_frames) * 100 / (mfxF64)m_nTotalOutFrames;
            }
            progressPercent = (std::min)(progressPercent, 100.0);
            mfxU32 remaining_time = (mfxU32)(elapsedTime * (100.0 - progressPercent) / progressPercent + 0.5);
            int hh = remaining_time / (60*60*1000);
            remaining_time -= hh * (60*60*1000);
            int mm = remaining_time / (60*1000);
            remaining_time -= mm * (60*1000);
            int ss = (remaining_time + 500) / 1000;

            int len = _stprintf_s(mes, _countof(mes), _T("[%.1lf%%] %d frames: %.2lf fps, %0.2lf kb/s, remain %d:%02d:%02d  "),
                progressPercent,
                m_sData.nProcessedFramesNum + drop_frames,
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
#if defined(_WIN32) || defined(_WIN64)
        GPUZ_SH_MEM gpu_info = { 0 };
        if (0 == get_gpuz_info(&gpu_info)) {
            m_sData.nGPUInfoCountSuccess++;
            m_sData.fGPULoadPercentTotal += gpu_load(&gpu_info);
            m_sData.fGPUClockTotal += gpu_core_clock(&gpu_info);
        } else {
            m_sData.nGPUInfoCountFail++;
        }
#endif //#if defined(_WIN32) || defined(_WIN64)
        UpdateDisplay(mes, drop_frames, progressPercent);
        return MFX_ERR_NONE;
    }
    virtual void WriteLine(const TCHAR *mes) {
        if (m_pQSVLog != nullptr && m_pQSVLog->getLogLevel() > QSV_LOG_INFO) {
            return;
        }
        (*m_pQSVLog)(QSV_LOG_INFO, _T("%s\n"), mes);
    }
    virtual void WriteLineDirect(TCHAR *mes) {
        if (m_pQSVLog != nullptr && m_pQSVLog->getLogLevel() > QSV_LOG_INFO) {
            return;
        }
        m_pQSVLog->write_log(QSV_LOG_INFO, mes);
    }
    virtual void WriteFrameTypeResult(const TCHAR *header, mfxU32 count, mfxU32 maxCount, mfxU64 frameSize, mfxU64 maxFrameSize, double avgQP) {
        if (count) {
            TCHAR mes[512] = { 0 };
            int mes_len = 0;
            const int header_len = (int)_tcslen(header);
            memcpy(mes, header, header_len * sizeof(mes[0]));
            mes_len += header_len;

            for (int i = (std::max)(0, (int)log10((double)count)); i < (int)log10((double)maxCount) && mes_len < _countof(mes); i++, mes_len++)
                mes[mes_len] = _T(' ');
            mes_len += _stprintf_s(mes + mes_len, _countof(mes) - mes_len, _T("%u"), count);

            if (avgQP >= 0.0) {
                mes_len += _stprintf_s(mes + mes_len, _countof(mes) - mes_len, _T(",  avgQP  %4.2f"), avgQP);
            }
            
            if (frameSize > 0) {
                const TCHAR *TOTAL_SIZE = _T(",  total size  ");
                memcpy(mes + mes_len, TOTAL_SIZE, _tcslen(TOTAL_SIZE) * sizeof(mes[0]));
                mes_len += (int)_tcslen(TOTAL_SIZE);

                for (int i = (std::max)(0, (int)log10((double)frameSize / (double)(1024 * 1024))); i < (int)log10((double)maxFrameSize / (double)(1024 * 1024)) && mes_len < _countof(mes); i++, mes_len++)
                    mes[mes_len] = _T(' ');

                mes_len += _stprintf_s(mes + mes_len, _countof(mes) - mes_len, _T("%.2f MB"), (double)frameSize / (double)(1024 * 1024));
            }

            WriteLine(mes);
        }
    }
    virtual void WriteResults(sFrameTypeInfo *info)
    {
        auto tm_result = std::chrono::system_clock::now();
        const auto time_elapsed64 = std::chrono::duration_cast<std::chrono::milliseconds>(tm_result - m_tmStart).count();
        m_sData.fEncodeFps = m_sData.nProcessedFramesNum * 1000.0 / (double)time_elapsed64;
        m_sData.fBitrateKbps = (mfxF64)(m_sData.nWrittenBytes * 8) *  (m_nOutputFPSRate / (double)m_nOutputFPSScale) / (1000.0 * m_sData.nProcessedFramesNum);

        TCHAR mes[512] = { 0 };
        for (int i = 0; i < 79; i++)
            mes[i] = _T(' ');
        WriteLine(mes);

        _stprintf_s(mes, _T("encoded %d frames, %.2f fps, %.2f kbps, %.2f MB"),
            m_sData.nProcessedFramesNum,
            m_sData.fEncodeFps,
            m_sData.fBitrateKbps,
            (double)m_sData.nWrittenBytes / (double)(1024 * 1024)
            );
        WriteLine(mes);

        int hh = (int)(time_elapsed64 / (60*60*1000));
        int time_elapsed = (int)(time_elapsed64 - hh * (60*60*1000));
        int mm = time_elapsed / (60*1000);
        time_elapsed -= mm * (60*1000);
        int ss = (time_elapsed + 500) / 1000;
#if defined(_WIN32) || defined(_WIN64)
        m_sData.fCPUUsagePercent = GetProcessAvgCPUUsage(&m_sStartTime);
        if (m_sData.nGPUInfoCountSuccess > m_sData.nGPUInfoCountFail) {
            double gpu_load = m_sData.fGPULoadPercentTotal / m_sData.nGPUInfoCountSuccess;
            int gpu_clock_avg = (int)(m_sData.fGPUClockTotal / m_sData.nGPUInfoCountSuccess + 0.5);
            _stprintf_s(mes, _T("encode time %d:%02d:%02d, CPULoad: %.2f%%, GPULoad: %.2f%%, GPUClockAvg: %dMHz\n"), hh, mm, ss, m_sData.fCPUUsagePercent, gpu_load, gpu_clock_avg);
        } else {
            _stprintf_s(mes, _T("encode time %d:%02d:%02d, CPULoad: %.2f%%\n"), hh, mm, ss, m_sData.fCPUUsagePercent);
        }
#else
        _stprintf_s(mes, _T("encode time %d:%02d:%02d\n"), hh, mm, ss);
#endif
        WriteLineDirect(mes);

        mfxU32 maxCount = (std::max)(m_sData.nICount, (std::max)(m_sData.nPCount, m_sData.nBCount));
        mfxU64 maxFrameSize = (std::max)(m_sData.nIFrameSize, (std::max)(m_sData.nPFrameSize, m_sData.nBFrameSize));

        WriteFrameTypeResult(_T("frame type IDR "), m_sData.nIDRCount, maxCount,                   0, maxFrameSize, -1.0);
        WriteFrameTypeResult(_T("frame type I   "), m_sData.nICount,   maxCount, m_sData.nIFrameSize, maxFrameSize, (info) ? info->sumQPI / (double)info->frameCountI : -1);
        WriteFrameTypeResult(_T("frame type P   "), m_sData.nPCount,   maxCount, m_sData.nPFrameSize, maxFrameSize, (info) ? info->sumQPP / (double)info->frameCountP : -1);
        WriteFrameTypeResult(_T("frame type B   "), m_sData.nBCount,   maxCount, m_sData.nBFrameSize, maxFrameSize, (info) ? info->sumQPB / (double)info->frameCountB : -1);
    }
    virtual int64_t getStartTimeMicroSec() {
#if defined(_WIN32) || defined(_WIN64)
        return m_sStartTime.creation / 10;
#else
        return (int)(m_sStartTime.creation * (double)(1e6 / CLOCKS_PER_SEC) + 0.5);
#endif
    }
    virtual bool getEncStarted() {
        return m_bEncStarted;
    }
    BOOL m_pause;
    mfxU32 m_nInputFrames;
    mfxU32 m_nTotalOutFrames;
    mfxU32 m_nOutputFPSRate;
    mfxU32 m_nOutputFPSScale;
protected:
    std::chrono::system_clock::time_point m_tmStart;
    std::chrono::system_clock::time_point m_tmLastUpdate;
    PROCESS_TIME m_sStartTime;
    sEncodeStatusData m_sData;
    shared_ptr<CQSVLog> m_pQSVLog;
    bool m_bStdErrWriteToConsole;
    bool m_bEncStarted;
};

class CEncodingThread 
{
public:
    CEncodingThread();
    ~CEncodingThread();

    mfxStatus Init(mfxU16 bufferSize);
    void Close();
    //終了を待機する
    mfxStatus WaitToFinish(mfxStatus sts, shared_ptr<CQSVLog> pQSVLog);
    mfxStatus RunEncFuncbyThread(void(*func)(void *prm), CEncodingPipeline *pipeline, size_t threadAffinityMask);
    mfxStatus RunSubFuncbyThread(void(*func)(void *prm), CEncodingPipeline *pipeline, size_t threadAffinityMask);

    std::thread& GetHandleEncThread() {
        return m_thEncode;
    }
    std::thread& GetHandleSubThread() {
        return m_thSub;
    }

    std::atomic_int m_bthForceAbort;
    std::atomic_int m_bthSubAbort;
    sInputBufSys *m_InputBuf;
    mfxU32 m_nFrameSet;
    mfxU32 m_nFrameGet;
    mfxStatus m_stsThread;
    mfxU16  m_nFrameBuffer;
protected:
    std::thread m_thEncode;
    std::thread m_thSub;
    bool m_bInit;
};

static inline int GetFreeSurface(mfxFrameSurface1* pSurfacesPool, int nPoolSize) {
    static const int SleepInterval = 1; // milliseconds
                                        //wait if there's no free surface
    for (mfxU32 j = 0; j < MSDK_WAIT_INTERVAL; j += SleepInterval) {
        for (mfxU16 i = 0; i < nPoolSize; i++) {
            if (0 == pSurfacesPool[i].Data.Locked)
                return i;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(SleepInterval));
    }
    return MSDK_INVALID_SURF_IDX;
}

static inline mfxU16 GetFreeSurfaceIndex(mfxFrameSurface1* pSurfacesPool, mfxU16 nPoolSize, mfxU16 step) {
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

#endif //__QSV_CONTROL_H__
