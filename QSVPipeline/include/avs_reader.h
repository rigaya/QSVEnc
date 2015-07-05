//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#ifndef _AVS_READER_H_
#define _AVS_READER_H_

#include "qsv_version.h"
#if ENABLE_AVISYNTH_READER
#include <Windows.h>
#include "sample_utils.h"
#pragma warning(push)
#pragma warning(disable:4244)
#include "avisynth_c.h" //Avisynth ver 2.5.8 (2.6.0の機能等は不要)
#pragma warning(pop)

typedef AVS_Value (__stdcall *func_avs_invoke)(AVS_ScriptEnvironment *scriptEnv, const char *name, AVS_Value args, const char** arg_names);
typedef AVS_Clip * (__stdcall *func_avs_take_clip)(AVS_Value value, AVS_ScriptEnvironment *scriptEnv);
typedef void  (__stdcall *func_avs_release_value)(AVS_Value value);
typedef AVS_ScriptEnvironment * (__stdcall *func_avs_create_script_environment)(int version);
typedef const AVS_VideoInfo * (__stdcall *func_avs_get_video_info)(AVS_Clip *clip);
typedef AVS_VideoFrame * (__stdcall *func_avs_get_frame)(AVS_Clip *clip, int n);
typedef void (__stdcall *func_avs_release_video_frame)(AVS_VideoFrame * frame);
typedef void (__stdcall *func_avs_release_clip)(AVS_Clip *clip);
typedef void (__stdcall *func_avs_delete_script_environment) (AVS_ScriptEnvironment *scriptEnv);
typedef float (__stdcall *func_avs_get_version)(void);

typedef struct {
    HMODULE h_avisynth;
    func_avs_invoke invoke;
    func_avs_take_clip take_clip;
    func_avs_release_value release_value;
    func_avs_create_script_environment create_script_environment;
    func_avs_get_video_info get_video_info;
    func_avs_get_frame get_frame;
    func_avs_release_video_frame release_video_frame;
    func_avs_release_clip release_clip;
    func_avs_delete_script_environment delete_script_environment;
    func_avs_get_version get_version;
} avs_dll_t;

class CAVSReader : public CSmplYUVReader
{
public:
    CAVSReader();
    virtual ~CAVSReader();

    virtual mfxStatus Init(const TCHAR *strFileName, mfxU32 ColorFormat, const void *option, CEncodingThread *pEncThread, CEncodeStatusInfo *pEncSatusInfo, sInputCrop *pInputCrop);

    virtual void Close();
    virtual mfxStatus LoadNextFrame(mfxFrameSurface1* pSurface);
private:
    mfxStatus load_avisynth();
    void release_avisynth();

    AVS_ScriptEnvironment *m_sAVSenv;
    AVS_Clip *m_sAVSclip;
    const AVS_VideoInfo *m_sAVSinfo;

    avs_dll_t m_sAvisynth;
};

#endif //ENABLE_AVISYNTH_READER

#endif //_AVS_READER_H_
