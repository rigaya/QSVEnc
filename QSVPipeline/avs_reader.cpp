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
// ------------------------------------------------------------------------------------------

#include "avs_reader.h"
#if ENABLE_AVISYNTH_READER

#if defined(_WIN32) || defined(_WIN64)
static const TCHAR *avisynth_dll_name = _T("avisynth.dll");
#else
static const TCHAR *avisynth_dll_name = _T("libavxsynth.so");
#endif

CAVSReader::CAVSReader() {
    m_sAVSenv = NULL;
    m_sAVSclip = NULL;
    m_sAVSinfo = NULL;

    memset(&m_sAvisynth, 0, sizeof(m_sAvisynth));
    m_strReaderName = _T("avs");
}

CAVSReader::~CAVSReader() {
    Close();
}

void CAVSReader::release_avisynth() {
    if (m_sAvisynth.h_avisynth)
#if defined(_WIN32) || defined(_WIN64)
        FreeLibrary(m_sAvisynth.h_avisynth);
#else
        dlclose(m_sAvisynth.h_avisynth);
#endif

    memset(&m_sAvisynth, 0, sizeof(m_sAvisynth));
}

mfxStatus CAVSReader::load_avisynth() {
    release_avisynth();

#if defined(_WIN32) || defined(_WIN64)
    if (   NULL == (m_sAvisynth.h_avisynth = (HMODULE)LoadLibrary(avisynth_dll_name))
#else
    if (   NULL == (m_sAvisynth.h_avisynth = dlopen(avisynth_dll_name, RTLD_LAZY))
#endif
        || NULL == (m_sAvisynth.invoke = (func_avs_invoke)QSV_GET_PROC_ADDRESS(m_sAvisynth.h_avisynth, "avs_invoke"))
        || NULL == (m_sAvisynth.take_clip = (func_avs_take_clip)QSV_GET_PROC_ADDRESS(m_sAvisynth.h_avisynth, "avs_take_clip"))
        || NULL == (m_sAvisynth.create_script_environment = (func_avs_create_script_environment)QSV_GET_PROC_ADDRESS(m_sAvisynth.h_avisynth, "avs_create_script_environment"))
        || NULL == (m_sAvisynth.delete_script_environment = (func_avs_delete_script_environment)QSV_GET_PROC_ADDRESS(m_sAvisynth.h_avisynth, "avs_delete_script_environment"))
        || NULL == (m_sAvisynth.get_frame = (func_avs_get_frame)QSV_GET_PROC_ADDRESS(m_sAvisynth.h_avisynth, "avs_get_frame"))
        || NULL == (m_sAvisynth.get_version = (func_avs_get_version)QSV_GET_PROC_ADDRESS(m_sAvisynth.h_avisynth, "avs_get_version"))
        || NULL == (m_sAvisynth.get_video_info = (func_avs_get_video_info)QSV_GET_PROC_ADDRESS(m_sAvisynth.h_avisynth, "avs_get_video_info"))
        || NULL == (m_sAvisynth.release_clip = (func_avs_release_clip)QSV_GET_PROC_ADDRESS(m_sAvisynth.h_avisynth, "avs_release_clip"))
        || NULL == (m_sAvisynth.release_value = (func_avs_release_value)QSV_GET_PROC_ADDRESS(m_sAvisynth.h_avisynth, "avs_release_value"))
        || NULL == (m_sAvisynth.release_video_frame = (func_avs_release_video_frame)QSV_GET_PROC_ADDRESS(m_sAvisynth.h_avisynth, "avs_release_video_frame")))
        return MFX_ERR_INVALID_HANDLE;
    return MFX_ERR_NONE;
}

#pragma warning(push)
#pragma warning(disable:4100)
mfxStatus CAVSReader::Init(const TCHAR *strFileName, mfxU32 ColorFormat, const void *option, CEncodingThread *pEncThread, shared_ptr<CEncodeStatusInfo> pEncSatusInfo, sInputCrop *pInputCrop) {
    Close();
    m_pEncThread = pEncThread;
    m_pEncSatusInfo = pEncSatusInfo;
    memcpy(&m_sInputCrop, pInputCrop, sizeof(m_sInputCrop));

    if (MFX_ERR_NONE != load_avisynth()) {
        AddMessage(QSV_LOG_ERROR,  _T("failed to load %s.\n"), avisynth_dll_name);
        return MFX_ERR_INVALID_HANDLE;
    }

    if (NULL == (m_sAVSenv = m_sAvisynth.create_script_environment(AVISYNTH_INTERFACE_VERSION))) {
        AddMessage(QSV_LOG_ERROR,  _T("failed to init avisynth enviroment.\n"));
        return MFX_ERR_INVALID_HANDLE;
    }
    std::string filename_char;
    if (0 == tchar_to_string(strFileName, filename_char)) {
        AddMessage(QSV_LOG_ERROR,  _T("failed to convert to ansi characters.\n"));
        return MFX_ERR_INVALID_HANDLE;
    }
    AVS_Value val_filename = avs_new_value_string(filename_char.c_str());
    AVS_Value val_res = m_sAvisynth.invoke(m_sAVSenv, "Import", val_filename, NULL);
    m_sAvisynth.release_value(val_filename);
    AddMessage(QSV_LOG_DEBUG,  _T("opened avs file: \"%s\"\n"), char_to_tstring(filename_char).c_str());
    if (!avs_is_clip(val_res)) {
        AddMessage(QSV_LOG_ERROR,  _T("invalid clip.\n"));
        if (avs_is_error(val_res)) {
            AddMessage(QSV_LOG_ERROR, char_to_tstring(avs_as_string(val_res)) + _T("\n"));
        }
        m_sAvisynth.release_value(val_res);
        return MFX_ERR_INVALID_HANDLE;
    }
    m_sAVSclip = m_sAvisynth.take_clip(val_res, m_sAVSenv);
    m_sAvisynth.release_value(val_res);

    if (NULL == (m_sAVSinfo = m_sAvisynth.get_video_info(m_sAVSclip))) {
        AddMessage(QSV_LOG_ERROR,  _T("failed to get avs info.\n"));
        return MFX_ERR_INVALID_HANDLE;
    }

    if (!avs_has_video(m_sAVSinfo)) {
        AddMessage(QSV_LOG_ERROR,  _T("avs has no video.\n"));
        return MFX_ERR_INVALID_HANDLE;
    }
    AddMessage(QSV_LOG_DEBUG,  _T("found video from avs file, pixel type 0x%x.\n"), m_sAVSinfo->pixel_type);

    memset(&m_inputFrameInfo, 0, sizeof(m_inputFrameInfo));

    typedef struct CSPMap {
        int fmtID;
        QSV_ENC_CSP in, out;
    } CSPMap;

    static const std::vector<CSPMap> valid_csp_list = {
        { AVS_CS_YV12,  QSV_ENC_CSP_YV12, QSV_ENC_CSP_NV12},
        { AVS_CS_I420,  QSV_ENC_CSP_YV12, QSV_ENC_CSP_NV12},
        { AVS_CS_IYUV,  QSV_ENC_CSP_YV12, QSV_ENC_CSP_NV12},
        { AVS_CS_YUY2,  QSV_ENC_CSP_YUY2, QSV_ENC_CSP_NV12},
        { AVS_CS_BGR24, QSV_ENC_CSP_RGB3, QSV_ENC_CSP_RGB4},
        { AVS_CS_BGR32, QSV_ENC_CSP_RGB4, QSV_ENC_CSP_RGB4},
    };

    m_ColorFormat = 0x00;
    for (auto csp : valid_csp_list) {
        if (csp.fmtID == m_sAVSinfo->pixel_type) {
            m_ColorFormat = QSV_ENC_CSP_TO_MFX_FOURCC[csp.in];
            m_inputFrameInfo.FourCC = QSV_ENC_CSP_TO_MFX_FOURCC[csp.out];
            m_sConvert = get_convert_csp_func(csp.in, csp.out, false);
            break;
        }
    }

    if (0x00 == m_ColorFormat || nullptr == m_sConvert) {
        AddMessage(QSV_LOG_ERROR,  _T("invalid colorformat.\n"));
        return MFX_ERR_INVALID_COLOR_FORMAT;
    }

    int fps_gcd = qsv_gcd(m_sAVSinfo->fps_numerator, m_sAVSinfo->fps_denominator);
    m_inputFrameInfo.Width = (mfxU16)m_sAVSinfo->width;
    m_inputFrameInfo.Height = (mfxU16)m_sAVSinfo->height;
    m_inputFrameInfo.CropW = m_inputFrameInfo.Width - (pInputCrop->left + pInputCrop->right);
    m_inputFrameInfo.CropH = m_inputFrameInfo.Height - (pInputCrop->up + pInputCrop->bottom);
    m_inputFrameInfo.FrameRateExtN = m_sAVSinfo->fps_numerator / fps_gcd;
    m_inputFrameInfo.FrameRateExtD = m_sAVSinfo->fps_denominator / fps_gcd;
    memcpy(&m_inputFrameInfo.FrameId, &m_sAVSinfo->num_frames, sizeof(m_sAVSinfo->num_frames));
    
    tstring avisynth_version;
    AVS_Value val_version = m_sAvisynth.invoke(m_sAVSenv, "VersionNumber", avs_new_value_array(NULL, 0), NULL);
    if (avs_is_float(val_version)) {
        avisynth_version = strsprintf(_T("%.2f"), avs_as_float(val_version));
    }
    m_sAvisynth.release_value(val_version);
    
    tstring mes = strsprintf( _T("Avisynth %s (%s)->%s[%s], %dx%d, %d/%d fps"), avisynth_version.c_str(),
        QSV_ENC_CSP_NAMES[m_sConvert->csp_from], QSV_ENC_CSP_NAMES[m_sConvert->csp_to], get_simd_str(m_sConvert->simd),
        m_inputFrameInfo.Width, m_inputFrameInfo.Height, m_inputFrameInfo.FrameRateExtN, m_inputFrameInfo.FrameRateExtD);
    AddMessage(QSV_LOG_DEBUG, mes);
    m_strInputInfo += mes;

    m_bInited = true;
    return MFX_ERR_NONE;
}
#pragma warning(pop)

void CAVSReader::Close() {
    AddMessage(QSV_LOG_DEBUG, _T("Closing...\n"));
    if (m_sAVSclip)
        m_sAvisynth.release_clip(m_sAVSclip);
    if (m_sAVSenv)
        m_sAvisynth.delete_script_environment(m_sAVSenv);

    release_avisynth();

    m_sAVSenv = NULL;
    m_sAVSclip = NULL;
    m_sAVSinfo = NULL;
    m_bInited = false;
    m_pEncSatusInfo.reset();
    AddMessage(QSV_LOG_DEBUG, _T("Closed.\n"));
}

mfxStatus CAVSReader::LoadNextFrame(mfxFrameSurface1* pSurface) {
    mfxFrameInfo *pInfo = &pSurface->Info;
    mfxFrameData *pData = &pSurface->Data;

    mfxU16 CropLeft = m_sInputCrop.left;
    mfxU16 CropUp = m_sInputCrop.up;
    mfxU16 CropRight = m_sInputCrop.right;
    mfxU16 CropBottom = m_sInputCrop.bottom;

    uint32_t nTotalFrame = 0;
    memcpy(&nTotalFrame, &m_inputFrameInfo.FrameId, sizeof(nTotalFrame));
    if (m_pEncSatusInfo->m_nInputFrames >= nTotalFrame
        //m_pEncSatusInfo->m_nInputFramesがtrimの結果必要なフレーム数を大きく超えたら、エンコードを打ち切る
        //ちょうどのところで打ち切ると他のストリームに影響があるかもしれないので、余分に取得しておく
        || getVideoTrimMaxFramIdx() < (int)m_pEncSatusInfo->m_nInputFrames - TRIM_OVERREAD_FRAMES) {
        return MFX_ERR_MORE_DATA;
    }

    int w = 0, h = 0;
    if (pInfo->CropH > 0 && pInfo->CropW > 0) {
        w = pInfo->CropW;
        h = pInfo->CropH;
    } else {
        w = pInfo->Width;
        h = pInfo->Height;
    }
    w += (CropLeft + CropRight);
    h += (CropUp + CropBottom);
    
    AVS_VideoFrame *frame = m_sAvisynth.get_frame(m_sAVSclip, m_pEncSatusInfo->m_nInputFrames);
    if (frame == NULL) {
        return MFX_ERR_MORE_DATA;
    }
    
    BOOL interlaced = 0 != (pSurface->Info.PicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF));
    int crop[4] = { CropLeft, CropUp, CropRight, CropBottom };
    const void *dst_ptr[3] = { pData->Y, pData->UV, NULL };
    const void *src_ptr[3] = { avs_get_read_ptr_p(frame, AVS_PLANAR_Y), avs_get_read_ptr_p(frame, AVS_PLANAR_U), avs_get_read_ptr_p(frame, AVS_PLANAR_V) };
    if (MFX_FOURCC_RGB4 == m_sConvert->csp_to) {
        dst_ptr[0] = (std::min)((std::min)(pData->R, pData->G), pData->B);
    }
    m_sConvert->func[interlaced]((void **)dst_ptr, (const void **)src_ptr, w, avs_get_pitch_p(frame, AVS_PLANAR_Y), avs_get_pitch_p(frame, AVS_PLANAR_U), pData->Pitch, h, h, crop);
    
    m_sAvisynth.release_video_frame(frame);

    m_pEncSatusInfo->m_nInputFrames++;
    return m_pEncSatusInfo->UpdateDisplay(0);
}

#endif //ENABLE_AVISYNTH_READER
