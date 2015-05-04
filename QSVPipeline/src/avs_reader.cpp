//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#include "avs_reader.h"
#if ENABLE_AVISYNTH_READER

CAVSReader::CAVSReader() {
	m_sAVSenv = NULL;
	m_sAVSclip = NULL;
	m_sAVSinfo = NULL;

	memset(&m_sAvisynth, 0, sizeof(m_sAvisynth));
}

CAVSReader::~CAVSReader() {
	Close();
}

void CAVSReader::release_avisynth() {
	if (m_sAvisynth.h_avisynth)
		FreeLibrary(m_sAvisynth.h_avisynth);

	memset(&m_sAvisynth, 0, sizeof(m_sAvisynth));
}

mfxStatus CAVSReader::load_avisynth() {
	release_avisynth();

	if (   NULL == (m_sAvisynth.h_avisynth = (HMODULE)LoadLibrary(_T("avisynth.dll")))
		|| NULL == (m_sAvisynth.invoke = (func_avs_invoke)GetProcAddress(m_sAvisynth.h_avisynth, "avs_invoke"))
		|| NULL == (m_sAvisynth.take_clip = (func_avs_take_clip)GetProcAddress(m_sAvisynth.h_avisynth, "avs_take_clip"))
		|| NULL == (m_sAvisynth.create_script_environment = (func_avs_create_script_environment)GetProcAddress(m_sAvisynth.h_avisynth, "avs_create_script_environment"))
		|| NULL == (m_sAvisynth.delete_script_environment = (func_avs_delete_script_environment)GetProcAddress(m_sAvisynth.h_avisynth, "avs_delete_script_environment"))
		|| NULL == (m_sAvisynth.get_frame = (func_avs_get_frame)GetProcAddress(m_sAvisynth.h_avisynth, "avs_get_frame"))
		|| NULL == (m_sAvisynth.get_version = (func_avs_get_version)GetProcAddress(m_sAvisynth.h_avisynth, "avs_get_version"))
		|| NULL == (m_sAvisynth.get_video_info = (func_avs_get_video_info)GetProcAddress(m_sAvisynth.h_avisynth, "avs_get_video_info"))
		|| NULL == (m_sAvisynth.release_clip = (func_avs_release_clip)GetProcAddress(m_sAvisynth.h_avisynth, "avs_release_clip"))
		|| NULL == (m_sAvisynth.release_value = (func_avs_release_value)GetProcAddress(m_sAvisynth.h_avisynth, "avs_release_value"))
		|| NULL == (m_sAvisynth.release_video_frame = (func_avs_release_video_frame)GetProcAddress(m_sAvisynth.h_avisynth, "avs_release_video_frame")))
		return MFX_ERR_INVALID_HANDLE;
	return MFX_ERR_NONE;
}

#pragma warning(push)
#pragma warning(disable:4100)
mfxStatus CAVSReader::Init(const TCHAR *strFileName, mfxU32 ColorFormat, const void *option, CEncodingThread *pEncThread, CEncodeStatusInfo *pEncSatusInfo, sInputCrop *pInputCrop) {

	MSDK_CHECK_POINTER(strFileName, MFX_ERR_NULL_PTR);
	MSDK_CHECK_ERROR(_tclen(strFileName), 0, MFX_ERR_NULL_PTR);

	Close();

	MSDK_CHECK_POINTER(pEncThread, MFX_ERR_NULL_PTR);
	m_pEncThread = pEncThread;

	MSDK_CHECK_POINTER(pEncSatusInfo, MFX_ERR_NULL_PTR);
	m_pEncSatusInfo = pEncSatusInfo;

	MSDK_CHECK_POINTER(pInputCrop, MFX_ERR_NULL_PTR);
	memcpy(&m_sInputCrop, pInputCrop, sizeof(m_sInputCrop));

	if (MFX_ERR_NONE != load_avisynth()) {
		m_strInputInfo += _T("avisynth: failed to load avisynth.dll.\n");
		return MFX_ERR_INVALID_HANDLE;
	}

	if (NULL == (m_sAVSenv = m_sAvisynth.create_script_environment(AVISYNTH_INTERFACE_VERSION))) {
		m_strInputInfo += _T("avisynth: failed to init avisynth enviroment.\n");
		return MFX_ERR_INVALID_HANDLE;
	}
	std::string filename_char;
	if (0 == tchar_to_string(strFileName, filename_char)) {
		m_strInputInfo += _T("avcodec: failed to convert to ansi characters.\n");
		return MFX_ERR_INVALID_HANDLE;
	}
	AVS_Value val_filename = avs_new_value_string(filename_char.c_str());
	AVS_Value val_res = m_sAvisynth.invoke(m_sAVSenv, "Import", val_filename, NULL);
	m_sAvisynth.release_value(val_filename);
	if (!avs_is_clip(val_res)) {
		m_strInputInfo += _T("avisynth: invalid clip.\n");
		if (avs_is_error(val_res)) {
#if UNICODE
			WCHAR buf[1024];
			MultiByteToWideChar(CP_THREAD_ACP, MB_PRECOMPOSED, avs_as_string(val_res), -1, buf, _countof(buf));
			m_strInputInfo += buf;
#else
			m_strInputInfo += avs_as_string(val_res);
#endif
			m_strInputInfo += _T("\n");
		}
		m_sAvisynth.release_value(val_res);
		return MFX_ERR_INVALID_HANDLE;
	}
	m_sAVSclip = m_sAvisynth.take_clip(val_res, m_sAVSenv);
	m_sAvisynth.release_value(val_res);

	if (NULL == (m_sAVSinfo = m_sAvisynth.get_video_info(m_sAVSclip))) {
		m_strInputInfo += _T("avisynth: failed to get avs info.\n");
		return MFX_ERR_INVALID_HANDLE;
	}

	if (!avs_has_video(m_sAVSinfo)) {
		m_strInputInfo += _T("avisynth: avs has no video.\n");
		return MFX_ERR_INVALID_HANDLE;
	}

	memset(&m_inputFrameInfo, 0, sizeof(m_inputFrameInfo));

	typedef struct CSPMap {
		int fmtID;
		mfxU32 in, out;
	} CSPMap;

	static const std::vector<CSPMap> valid_csp_list = {
		{ AVS_CS_YV12,  MFX_FOURCC_YV12, MFX_FOURCC_NV12},
		{ AVS_CS_I420,  MFX_FOURCC_YV12, MFX_FOURCC_NV12},
		{ AVS_CS_IYUV,  MFX_FOURCC_YV12, MFX_FOURCC_NV12},
		{ AVS_CS_YUY2,  MFX_FOURCC_YUY2, MFX_FOURCC_NV12},
		{ AVS_CS_BGR24, MFX_FOURCC_RGB3, MFX_FOURCC_RGB4},
		{ AVS_CS_BGR32, MFX_FOURCC_RGB4, MFX_FOURCC_RGB4},
	};

	m_ColorFormat = 0x00;
	for (auto csp : valid_csp_list) {
		if (csp.fmtID == m_sAVSinfo->pixel_type) {
			m_ColorFormat = csp.in;
			m_inputFrameInfo.FourCC = csp.out;
			m_sConvert = get_convert_csp_func(csp.in, csp.out, false);
			break;
		}
	}

	if (0x00 == m_ColorFormat || nullptr == m_sConvert) {
		m_strInputInfo += _T("avisynth: invalid colorformat.\n");
		return MFX_ERR_INVALID_COLOR_FORMAT;
	}

	int fps_gcd = GCD(m_sAVSinfo->fps_numerator, m_sAVSinfo->fps_denominator);
	m_inputFrameInfo.Width = (mfxU16)m_sAVSinfo->width;
	m_inputFrameInfo.Height = (mfxU16)m_sAVSinfo->height;
	m_inputFrameInfo.CropW = m_inputFrameInfo.Width - (pInputCrop->left + pInputCrop->right);
	m_inputFrameInfo.CropH = m_inputFrameInfo.Height - (pInputCrop->up + pInputCrop->bottom);
	m_inputFrameInfo.FrameRateExtN = m_sAVSinfo->fps_numerator / fps_gcd;
	m_inputFrameInfo.FrameRateExtD = m_sAVSinfo->fps_denominator / fps_gcd;
	*(DWORD*)&m_inputFrameInfo.FrameId = m_sAVSinfo->num_frames;
	
	TCHAR avisynth_version[32] = { 0 };
	AVS_Value val_version = m_sAvisynth.invoke(m_sAVSenv, "VersionNumber", avs_new_value_array(NULL, 0), NULL);
	if (avs_is_float(val_version)) {
		_stprintf_s(avisynth_version, _countof(avisynth_version), _T("%.2f"), avs_as_float(val_version));
	}
	m_sAvisynth.release_value(val_version);
	
	TCHAR mes[256];
	_stprintf_s(mes, _countof(mes), _T("Avisynth %s (%s)->%s[%s], %dx%d, %d/%d fps"), avisynth_version, ColorFormatToStr(m_ColorFormat), ColorFormatToStr(m_inputFrameInfo.FourCC), get_simd_str(m_sConvert->simd),
		m_inputFrameInfo.Width, m_inputFrameInfo.Height, m_inputFrameInfo.FrameRateExtN, m_inputFrameInfo.FrameRateExtD);
	m_strInputInfo += mes;
	m_tmLastUpdate = timeGetTime();

	m_bInited = true;
	return MFX_ERR_NONE;
}
#pragma warning(pop)

void CAVSReader::Close() {
	if (m_sAVSclip)
		m_sAvisynth.release_clip(m_sAVSclip);
	if (m_sAVSenv)
		m_sAvisynth.delete_script_environment(m_sAVSenv);

	release_avisynth();

	m_sAVSenv = NULL;
	m_sAVSclip = NULL;
	m_sAVSinfo = NULL;
	m_bInited = false;
}

mfxStatus CAVSReader::LoadNextFrame(mfxFrameSurface1* pSurface) {
#ifdef _DEBUG
	MSDK_CHECK_ERROR(m_bInited, false, MFX_ERR_NOT_INITIALIZED);
#endif
	int w, h;
	mfxFrameInfo* pInfo = &pSurface->Info;
	mfxFrameData* pData = &pSurface->Data;

	mfxU16 CropLeft = m_sInputCrop.left;
	mfxU16 CropUp = m_sInputCrop.up;
	mfxU16 CropRight = m_sInputCrop.right;
	mfxU16 CropBottom = m_sInputCrop.bottom;

	if (m_pEncSatusInfo->m_nInputFrames >= *(DWORD*)&m_inputFrameInfo.FrameId)
		return MFX_ERR_MORE_DATA;

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
		dst_ptr[0] = min(min(pData->R, pData->G), pData->B);
	}
	m_sConvert->func[interlaced]((void **)dst_ptr, (void **)src_ptr, w, avs_get_pitch_p(frame, AVS_PLANAR_Y), avs_get_pitch_p(frame, AVS_PLANAR_U), pData->Pitch, h, crop);
	
	m_sAvisynth.release_video_frame(frame);

	m_pEncSatusInfo->m_nInputFrames++;
	// display update
	mfxU32 tm = timeGetTime();
	if (tm - m_tmLastUpdate > UPDATE_INTERVAL) {
		m_tmLastUpdate = tm;
		m_pEncSatusInfo->UpdateDisplay(tm, 0);
	}
	return MFX_ERR_NONE;
}

#endif //ENABLE_AVISYNTH_READER
