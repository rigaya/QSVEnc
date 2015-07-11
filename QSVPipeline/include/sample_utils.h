/* ////////////////////////////////////////////////////////////////////////////// */
/*
//
//              INTEL CORPORATION PROPRIETARY INFORMATION
//  This software is supplied under the terms of a license  agreement or
//  nondisclosure agreement with Intel Corporation and may not be copied
//  or disclosed except in  accordance  with the terms of that agreement.
//        Copyright (c) 2005-2014 Intel Corporation. All Rights Reserved.
//
//
*/

#ifndef __SAMPLE_UTILS_H__
#define __SAMPLE_UTILS_H__

#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>

#include "mfxstructures.h"
#include "mfxvideo.h"
#include "mfxjpeg.h"
#include "mfxplugin.h"

#include "vm/strings_defs.h"
#include "vm/file_defs.h"
#include "vm/time_defs.h"
#include "vm/atomic_defs.h"

#include "sample_types.h"
#include "sample_defs.h"
#include "qsv_prm.h"
#include "qsv_control.h"
#include "convert_csp.h"

using std::vector;

// A macro to disallow the copy constructor and operator= functions
// This should be used in the private: declarations for a class
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName&);               \
    void operator=(const TypeName&)

//! Base class for types that should not be assigned.
class no_assign {
    // Deny assignment
    void operator=(const no_assign&);
public:
#if __GNUC__
    //! Explicitly define default construction, because otherwise gcc issues gratuitous warning.
    no_assign() {}
#endif /* __GNUC__ */
};

//! Base class for types that should not be copied or assigned.
class no_copy : no_assign {
    //! Deny copy construction
    no_copy(const no_copy&);
public:
    //! Allow default construction
    no_copy() {}
};

typedef std::basic_string<msdk_char> msdk_string;
typedef std::basic_stringstream<msdk_char> msdk_stringstream;
typedef std::basic_ostream<msdk_char, std::char_traits<msdk_char> > msdk_ostream;
typedef std::basic_istream<msdk_char, std::char_traits<msdk_char> > msdk_istream;

#ifdef UNICODE 
#define msdk_cout std::wcout
#define msdk_err std::wcerr
#else
#define msdk_cout std::cout
#define msdk_err std::cerr
#endif

struct DeletePtr {
    template <class T> T* operator () (T* p) const {
        delete p;
        return 0;
    }
};

enum {
    CODEC_VP8 = MFX_MAKEFOURCC('V','P','8',' '),
    CODEC_MVC = MFX_MAKEFOURCC('M','V','C',' '),
};

bool IsDecodeCodecSupported(mfxU32 codecFormat);
bool IsEncodeCodecSupported(mfxU32 codecFormat);
bool IsPluginCodecSupported(mfxU32 codecFormat);

class CSmplYUVReader
{
public:

    CSmplYUVReader();
    virtual ~CSmplYUVReader();
    
    virtual void SetQSVLogPtr(CQSVLog *pQSVLog) {
        m_pPrintMes = pQSVLog;
    }
    virtual mfxStatus Init(const msdk_char *strFileName, mfxU32 ColorFormat, const void *prm, CEncodingThread *pEncThread, CEncodeStatusInfo *pEncSatusInfo, sInputCrop *pInputCrop);

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
        if (m_pEncThread->m_bthForceAbort) {
            AddMessage(QSV_LOG_DEBUG, _T("GetNextFrame: Encode Aborted...\n"));
            return m_pEncThread->m_stsThread;
        }
        //読み込み完了による終了
        if (m_pEncThread->m_stsThread == MFX_ERR_MORE_DATA && m_pEncThread->m_nFrameGet == m_pEncSatusInfo->m_nInputFrames) {
            AddMessage(QSV_LOG_DEBUG, _T("GetNextFrame: Frame read finished.\n"));
            return m_pEncThread->m_stsThread;
        }
        *pSurface = pInputBuf->pFrameSurface;
        (*pSurface)->Data.TimeStamp = inputBufIdx;
        (*pSurface)->Data.Locked = FALSE;
        m_pEncThread->m_nFrameGet++;
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
    //virtual mfxStatus Init(const msdk_char *strFileName, const mfxU32 ColorFormat, const mfxU32 numViews, std::vector<msdk_char*> srcFileBuff);
    virtual mfxStatus LoadNextFrame(mfxFrameSurface1* pSurface);

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
    void GetDecParam(mfxVideoParam *decParam) {
        memcpy(decParam, &m_sDecParam, sizeof(m_sDecParam));
    }
    const msdk_char *GetInputMessage() {
        const msdk_char *mes = m_strInputInfo.c_str();
        return (mes) ? mes : _T("");
    }
    void AddMessage(int log_level, const tstring& str) {
        if (m_pPrintMes == nullptr || log_level < m_pPrintMes->getLogLevel()) {
            return;
        }
         auto lines = split(str, _T("\n"));
         for (const auto& line : lines) {
             if (line[0] != _T('\0')) {
                 (*m_pPrintMes)(log_level, (m_strReaderName + _T(": ") + line + _T("\n")).c_str());
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
    mfxVideoParam m_sDecParam;

    const ConvertCSP *m_sConvert;

    mfxU32 m_nInputCodec;

    mfxU32 bufSize;
    mfxU8 *buffer;

    tstring m_strReaderName;
    tstring m_strInputInfo;
    CQSVLog *m_pPrintMes;  //ログ出力

    sTrimParam m_sTrimParam;
};

class CSmplBitstreamWriter
{
public:

    CSmplBitstreamWriter();
    virtual ~CSmplBitstreamWriter();

    virtual void SetQSVLogPtr(CQSVLog *pQSVLog) {
        m_pPrintMes = pQSVLog;
    }
    virtual mfxStatus Init(const msdk_char *strFileName, const void *prm, CEncodeStatusInfo *pEncSatusInfo);

    virtual mfxStatus SetVideoParam(const mfxVideoParam *pMfxVideoPrm, const mfxExtCodingOption2 *cop2);

    virtual mfxStatus WriteNextFrame(mfxBitstream *pMfxBitstream);
    virtual void Close();

    virtual bool outputStdout() {
        return m_bOutputIsStdout;
    }
    
    const msdk_char *GetOutputMessage() {
        const msdk_char *mes = m_strOutputInfo.c_str();
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
    CEncodeStatusInfo *m_pEncSatusInfo;
    FILE*       m_fSource;
    bool        m_bOutputIsStdout;
    bool        m_bInited;
    bool        m_bNoOutput;
    char*       m_pOutputBuffer;
    tstring     m_strWriterName;
    tstring     m_strOutputInfo;
    CQSVLog    *m_pPrintMes;  //ログ出力
};

class CSmplYUVWriter
{
public:

    CSmplYUVWriter();
    virtual ~CSmplYUVWriter();

    virtual void      Close();
    virtual mfxStatus Init(const msdk_char *strFileName, const mfxU32 numViews);
    virtual mfxStatus WriteNextFrame(mfxFrameSurface1 *pSurface);

    void SetMultiView() { m_bIsMultiView = true; }

protected:
    FILE         *m_fDest, **m_fDestMVC;
    bool         m_bInited, m_bIsMultiView;
    mfxU32       m_numCreatedFiles;
};

class CSmplBitstreamReader
{
public:

    CSmplBitstreamReader();
    virtual ~CSmplBitstreamReader();

    //resets position to file begin
    virtual void      Reset();
    virtual void      Close();
    virtual mfxStatus Init(const msdk_char *strFileName);
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
    bool SOImarkerIsFound(mfxBitstream *pBS);
    bool EOImarkerIsFound(mfxBitstream *pBS);
};

//appends output bistream with exactly 1 frame, reports about error
class CIVFFrameReader : public CSmplBitstreamReader
{
public:
    CIVFFrameReader();
    virtual mfxStatus Init(const msdk_char *strFileName);
    virtual mfxStatus ReadNextFrame(mfxBitstream *pBS);

protected:

    /*bytes 0-3    signature: 'DKIF'
  bytes 4-5    version (should be 0)
  bytes 6-7    length of header in bytes
  bytes 8-11   codec FourCC (e.g., 'VP80')
  bytes 12-13  width in pixels
  bytes 14-15  height in pixels
  bytes 16-19  frame rate
  bytes 20-23  time scale
  bytes 24-27  number of frames in file
  bytes 28-31  unused*/

    struct DKIFHrd
    {
        mfxU32 dkif;
        mfxU16 version;
        mfxU16 header_len;
        mfxU32 codec_FourCC;
        mfxU16 width;
        mfxU16 height;
        mfxU32 frame_rate;
        mfxU32 time_scale;
        mfxU32 num_frames;
        mfxU32 unused;
    }m_hdr;
};

// writes bitstream to duplicate-file & supports joining
// (for ViewOutput encoder mode)
class CSmplBitstreamDuplicateWriter : public CSmplBitstreamWriter
{
public:
    CSmplBitstreamDuplicateWriter();

    virtual mfxStatus InitDuplicate(const msdk_char *strFileName);
    virtual mfxStatus JoinDuplicate(CSmplBitstreamDuplicateWriter *pJoinee);
    virtual mfxStatus WriteNextFrame(mfxBitstream *pMfxBitstream, bool isPrint = true);
    virtual void Close();
protected:
    FILE*     m_fSourceDuplicate;
    bool      m_bJoined;
};

//timeinterval calculation helper

template <int tag = 0>
class CTimeInterval : private no_copy
{
    static double g_Freq;
    double       &m_start;
    double        m_own;//reference to this if external counter not required
    //since QPC functions are quite slow it makes sense to optionally enable them
    bool         m_bEnable;
    msdk_tick    m_StartTick;

public:
    CTimeInterval(double &dRef, bool bEnable = true)
        : m_start(dRef)
        , m_bEnable(bEnable)
    {
        if (!m_bEnable)
            return;
        Initialize();
    }
    CTimeInterval(bool bEnable = true)
        : m_start(m_own)
        , m_own()
        , m_bEnable(bEnable)
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
            m_start = MSDK_GET_TIME(msdk_time_get_tick(), m_StartTick, g_Freq);
        }
        return m_start;
    }
    //last comitted value
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
        if (0.0 == g_Freq)
        {
            g_Freq = (double)msdk_time_get_frequency();
        }
        m_StartTick = msdk_time_get_tick();
    }
};

template <int tag>double CTimeInterval<tag>::g_Freq = 0.0f;

/** Helper class to measure execution time of some code. Use this class
 * if you need manual measurements.
 *
 * Usage example:
 * {
 *   CTimer timer;
 *   msdk_tick summary_tick;
 *
 *   timer.Start()
 *   function_to_measure();
 *   summary_tick = timer.GetDelta();
 *   printf("Elapsed time 1: %f\n", timer.GetTime());
 *   ...
 *   if (condition) timer.Start();
     function_to_measure();
 *   if (condition) {
 *     summary_tick += timer.GetDelta();
 *     printf("Elapsed time 2: %f\n", timer.GetTime();
 *   }
 *   printf("Overall time: %f\n", CTimer::ConvertToSeconds(summary_tick);
 * }
 */
class CTimer
{
public:
    CTimer():
        start(0)
    {
    }
    static msdk_tick GetFrequency()
    {
        if (!frequency) frequency = msdk_time_get_frequency();
        return frequency;
    }
    static mfxF64 ConvertToSeconds(msdk_tick elapsed)
    {
        return MSDK_GET_TIME(elapsed, 0, GetFrequency());
    }

    inline void Start()
    {
        start = msdk_time_get_tick();
    }
    inline msdk_tick GetDelta()
    {
        return msdk_time_get_tick() - start;
    }
    inline mfxF64 GetTime()
    {
        return MSDK_GET_TIME(msdk_time_get_tick(), start, GetFrequency());
    }

protected:
    static msdk_tick frequency;
    msdk_tick start;
private:
    CTimer(const CTimer&);
    void operator=(const CTimer&);
};

/** Helper class to measure overall execution time of some code. Use this
 * class if you want to measure execution time of the repeatedly executed
 * code.
 *
 * Usage example 1:
 *
 * msdk_tick summary_tick = 0;
 *
 * void function() {
 *
 * {
 *   CAutoTimer timer(&summary_tick);
 *   ...
 * }
 *     ...
 * int main() {
 *   for (;condition;) {
 *     function();
 *   }
 *   printf("Elapsed time: %f\n", CTimer::ConvertToSeconds(summary_tick);
 *   return 0;
 * }
 *
 * Usage example 2:
 * {
 *   msdk_tick summary_tick = 0;
 *
 *   {
 *     CAutoTimer timer(&summary_tick);
 *
 *     for (;condition;) {
 *       ...
 *       {
 *         function_to_measure();
 *         timer.Sync();
 *         printf("Progress: %f\n", CTimer::ConvertToSeconds(summary_tick);
 *       }
 *       ...
 *     }
 *   }
 *   printf("Elapsed time: %f\n", CTimer::ConvertToSeconds(summary_tick);
 * }
 *
 */
class CAutoTimer
{
public:
    CAutoTimer(msdk_tick& _elapsed):
        elapsed(_elapsed),
        start(0)
    {
        elapsed = _elapsed;
        start = msdk_time_get_tick();
    }
    ~CAutoTimer()
    {
        elapsed += msdk_time_get_tick() - start;
    }
    msdk_tick Sync()
    {
        msdk_tick cur = msdk_time_get_tick();
        elapsed += cur - start;
        start = cur;
        return elapsed;
    }
protected:
    msdk_tick& elapsed;
    msdk_tick start;
private:
    CAutoTimer(const CAutoTimer&);
    void operator=(const CAutoTimer&);
};

mfxStatus ConvertFrameRate(mfxF64 dFrameRate, mfxU32* pnFrameRateExtN, mfxU32* pnFrameRateExtD);
mfxF64 CalculateFrameRate(mfxU32 nFrameRateExtN, mfxU32 nFrameRateExtD);

static inline int GetFreeSurface(mfxFrameSurface1* pSurfacesPool, int nPoolSize) {
    static const int SleepInterval = 1; // milliseconds
    //wait if there's no free surface
    for (mfxU32 j = 0; j < MSDK_WAIT_INTERVAL; j += SleepInterval) {
        for (mfxU16 i = 0; i < nPoolSize; i++) {
            if (0 == pSurfacesPool[i].Data.Locked)
                return i;
        }
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

//performs copy to end if possible, also move data to buffer begin if necessary
//shifts offset pointer in source bitstream in success case
mfxStatus MoveMfxBitstream(mfxBitstream *pTarget, mfxBitstream *pSrc, mfxU32 nBytesToCopy);
mfxStatus ExtendMfxBitstream(mfxBitstream* pBitstream, mfxU32 nSize);
void WipeMfxBitstream(mfxBitstream* pBitstream);

mfxU16 CalculateDefaultBitrate(mfxU32 nCodecId, mfxU32 nTargetUsage, mfxU32 nWidth, mfxU32 nHeight, mfxF64 dFrameRate);

//serialization fnc set
std::basic_string<msdk_char> CodecIdToStr(mfxU32 nFourCC);
mfxU16 StrToTargetUsage(msdk_char* strInput);
const msdk_char* TargetUsageToStr(mfxU16 tu);
const msdk_char* ColorFormatToStr(mfxU32 format);
const msdk_char* MfxStatusToStr(mfxStatus sts);
const msdk_char* EncmodeToStr(mfxU32 enc_mode);
const msdk_char* MemTypeToStr(mfxU32 memType);

// sets bitstream->PicStruct parsing first APP0 marker in bitstream
mfxStatus MJPEG_AVI_ParsePicStruct(mfxBitstream *bitstream);

// For MVC encoding/decoding purposes
std::basic_string<msdk_char> FormMVCFileName(const msdk_char *strFileName, const mfxU32 numView);

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
    enum { id = 0 };
};
template<>struct mfx_ext_buffer_id<mfxExtCodingOption>{
    enum { id = MFX_EXTBUFF_CODING_OPTION };
};
template<>struct mfx_ext_buffer_id<mfxExtCodingOption2>{
    enum { id = MFX_EXTBUFF_CODING_OPTION2 };
};
template<>struct mfx_ext_buffer_id<mfxExtAvcTemporalLayers>{
    enum { id = MFX_EXTBUFF_AVC_TEMPORAL_LAYERS };
};
template<>struct mfx_ext_buffer_id<mfxExtAVCRefListCtrl>{
    enum { id = MFX_EXTBUFF_AVC_REFLIST_CTRL };
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

//do not link MediaSDK dispatched if class not used
struct MSDKAdapter {
    // returns the number of adapter associated with MSDK session, 0 for SW session
    static mfxU32 GetNumber(mfxSession session = 0) {
        mfxU32 adapterNum = 0; // default
        mfxIMPL impl = MFX_IMPL_SOFTWARE; // default in case no HW IMPL is found

        // we don't care for error codes in further code; if something goes wrong we fall back to the default adapter
        if (session)
        {
            MFXQueryIMPL(session, &impl);
        }
        else
        {
            // an auxiliary session, internal for this function
            mfxSession auxSession;
            memset(&auxSession, 0, sizeof(auxSession));

            mfxVersion ver = { {1, 1 }}; // minimum API version which supports multiple devices
            MFXInit(MFX_IMPL_HARDWARE_ANY, &ver, &auxSession);
            MFXQueryIMPL(auxSession, &impl);
            MFXClose(auxSession);
        }

        // extract the base implementation type
        mfxIMPL baseImpl = MFX_IMPL_BASETYPE(impl);

        const struct
        {
            // actual implementation
            mfxIMPL impl;
            // adapter's number
            mfxU32 adapterID;

        } implTypes[] = {
            { MFX_IMPL_HARDWARE,  0 },
            { MFX_IMPL_SOFTWARE,  0 },
            { MFX_IMPL_HARDWARE2, 1 },
            { MFX_IMPL_HARDWARE3, 2 },
            { MFX_IMPL_HARDWARE4, 3 }
        };


        // get corresponding adapter number
        for (mfxU8 i = 0; i < sizeof(implTypes)/sizeof(*implTypes); i++)
        {
            if (implTypes[i].impl == baseImpl)
            {
                adapterNum = implTypes[i].adapterID;
                break;
            }
        }

        return adapterNum;
    }
};

struct APIChangeFeatures {
    bool JpegDecode;
    bool JpegEncode;
    bool MVCDecode;
    bool MVCEncode;
    bool IntraRefresh;
    bool LowLatency;
    bool ViewOutput;
    bool LookAheadBRC;
    bool AudioDecode;
    bool SupportCodecPluginAPI;
};

mfxVersion getMinimalRequiredVersion(const APIChangeFeatures &features);

enum msdkAPIFeature {
    MSDK_FEATURE_NONE,
    MSDK_FEATURE_MVC,
    MSDK_FEATURE_JPEG_DECODE,
    MSDK_FEATURE_LOW_LATENCY,
    MSDK_FEATURE_MVC_VIEWOUTPUT,
    MSDK_FEATURE_JPEG_ENCODE,
    MSDK_FEATURE_LOOK_AHEAD,
    MSDK_FEATURE_PLUGIN_API
};

/* Returns true if feature is supported in the given API version */
bool CheckVersion(mfxVersion* version, msdkAPIFeature feature);

void ConfigureAspectRatioConversion(mfxInfoVPP* pVppInfo);

enum MsdkTraceLevel {
    MSDK_TRACE_LEVEL_SILENT = -1,
    MSDK_TRACE_LEVEL_CRITICAL = 0,
    MSDK_TRACE_LEVEL_ERROR = 1,
    MSDK_TRACE_LEVEL_WARNING = 2,
    MSDK_TRACE_LEVEL_INFO = 3,
    MSDK_TRACE_LEVEL_DEBUG = 4,
};

msdk_string NoFullPath(const msdk_string &);
int  msdk_trace_get_level();
void msdk_trace_set_level(int);
bool msdk_trace_is_printable(int);

msdk_ostream & operator <<(msdk_ostream & os, MsdkTraceLevel tt);

template<typename T>
    mfxStatus msdk_opt_read(const msdk_char* string, T& value);

template<size_t S>
    mfxStatus msdk_opt_read(const msdk_char* string, msdk_char (&value)[S])
    {
    #if defined(_WIN32) || defined(_WIN64)
        return (0 == _tcscpy_s(value, string))? MFX_ERR_NONE: MFX_ERR_UNKNOWN;
    #else
        if (strlen(string) < S) {
            strncpy(value, string, S);
            return MFX_ERR_NONE;
        }
        return MFX_ERR_UNKNOWN;
    #endif
    }

template<typename T>
    inline mfxStatus msdk_opt_read(const msdk_string& string, T& value)
    {
        return msdk_opt_read(string.c_str(), value);
    }

mfxStatus StrFormatToCodecFormatFourCC(msdk_char* strInput, mfxU32 &codecFormat);

#endif //__SAMPLE_UTILS_H__
