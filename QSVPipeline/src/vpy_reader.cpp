//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#include "vpy_reader.h"
#if ENABLE_VAPOURSYNTH_READER
#include <map>
#include <fstream>
#include <string>
#include <algorithm>

CVSReader::CVSReader() {
	m_sVSapi = NULL;
	m_sVSscript = NULL;
	m_sVSnode = NULL;
	m_nAsyncFrames = 0;
	memset(m_pAsyncBuffer, 0, sizeof(m_pAsyncBuffer));
	memset(m_hAsyncEvent, 0, sizeof(m_hAsyncEvent));

	hVSScriptDLL = NULL;
	vs_init = NULL;
	vs_finalize = NULL;
	vs_evaluateScript = NULL;
	vs_evaluateFile = NULL;
	vs_freeScript = NULL;
	vs_getError = NULL;
	vs_getOutput = NULL;
	vs_clearOutput = NULL;
	vs_getCore = NULL;
	vs_getVSApi = NULL;
}

CVSReader::~CVSReader() {
	Close();
}

int CVSReader::initAsyncEvents() {
	for (int i = 0; i < _countof(m_hAsyncEvent); i++) {
		if (NULL == (m_hAsyncEvent[i] = CreateEvent(NULL, FALSE, FALSE, NULL)))
			return 1;
	}
	return 0;
}
void CVSReader::closeAsyncEvents() {
	for (int i = 0; i < _countof(m_hAsyncEvent); i++)
		if (m_hAsyncEvent[i])
			CloseHandle(m_hAsyncEvent[i]);
	memset(m_hAsyncEvent, 0, sizeof(m_hAsyncEvent));
}

#pragma warning(push)
#pragma warning(disable:4100)
void __stdcall frameDoneCallback(void *userData, const VSFrameRef *f, int n, VSNodeRef *, const char *errorMsg) {
	reinterpret_cast<CVSReader*>(userData)->setFrameToAsyncBuffer(n, f);
}
#pragma warning(pop)

void CVSReader::setFrameToAsyncBuffer(int n, const VSFrameRef* f) {
	m_pAsyncBuffer[n & (ASYNC_BUFFER_SIZE-1)] = f;
	SetEvent(m_hAsyncEvent[n & (ASYNC_BUFFER_SIZE-1)]);

	if (m_nAsyncFrames < *(int*)&m_inputFrameInfo.FrameId) {
		m_sVSapi->getFrameAsync(m_nAsyncFrames, m_sVSnode, frameDoneCallback, this);
		m_nAsyncFrames++;
	}
}

int CVSReader::getRevInfo(const char *vsVersionString) {
	char *api_info = NULL;
	char buf[1024];
	strcpy_s(buf, _countof(buf), vsVersionString);
	for (char *p = buf, *q = NULL, *r = NULL; NULL != (q = strtok_s(p, "\n", &r)); ) {
		if (NULL != (api_info = strstr(q, "Core"))) {
			strcpy_s(buf, _countof(buf), api_info);
			int rev = 0;
			return (1 == sscanf_s(buf, "Core r%d", &rev)) ? rev : 0;
		}
		p = NULL;
	}
	return 0;
}

#pragma warning(push)
#pragma warning(disable:4100)
mfxStatus CVSReader::Init(const TCHAR *strFileName, mfxU32 ColorFormat, int option, CEncodingThread *pEncThread, CEncodeStatusInfo *pEncSatusInfo, sInputCrop *pInputCrop) {
	MSDK_CHECK_POINTER(strFileName, MFX_ERR_NULL_PTR);
	MSDK_CHECK_ERROR(_tclen(strFileName), 0, MFX_ERR_NULL_PTR);

	Close();

	MSDK_CHECK_POINTER(pEncThread, MFX_ERR_NULL_PTR);
	m_pEncThread = pEncThread;

	MSDK_CHECK_POINTER(pEncSatusInfo, MFX_ERR_NULL_PTR);
	m_pEncSatusInfo = pEncSatusInfo;

	MSDK_CHECK_POINTER(pInputCrop, MFX_ERR_NULL_PTR);
	memcpy(&m_sInputCrop, pInputCrop, sizeof(m_sInputCrop));

	const bool use_mt_mode = option != 0;

	if (NULL == (hVSScriptDLL = LoadLibrary(_T("vsscript.dll")))) {
		m_strInputInfo += _T("Failed to load vsscript.dll.\n");
		return MFX_ERR_INVALID_HANDLE;
	}

	std::map<void **, const char*> vs_func_list = {
		{ (void **)&vs_init,           (VPY_X64) ? "vsscript_init"           : "_vsscript_init@0"            },
		{ (void **)&vs_finalize,       (VPY_X64) ? "vsscript_finalize"       : "_vsscript_finalize@0",       },
		{ (void **)&vs_evaluateScript, (VPY_X64) ? "vsscript_evaluateScript" : "_vsscript_evaluateScript@16" },
		{ (void **)&vs_evaluateFile,   (VPY_X64) ? "vsscript_evaluateFile"   : "_vsscript_evaluateFile@12"   },
		{ (void **)&vs_freeScript,     (VPY_X64) ? "vsscript_freeScript"     : "_vsscript_freeScript@4"      },
		{ (void **)&vs_getError,       (VPY_X64) ? "vsscript_getError"       : "_vsscript_getError@4"        },
		{ (void **)&vs_getOutput,      (VPY_X64) ? "vsscript_getOutput"      : "_vsscript_getOutput@8"       },
		{ (void **)&vs_clearOutput,    (VPY_X64) ? "vsscript_clearOutput"    : "_vsscript_clearOutput@8"     },
		{ (void **)&vs_getCore,        (VPY_X64) ? "vsscript_getCore"        : "_vsscript_getCore@4"         },
		{ (void **)&vs_getVSApi,       (VPY_X64) ? "vsscript_getVSApi"       : "_vsscript_getVSApi@0"        },
	};

	for (auto vs_func : vs_func_list) {
		if (NULL == (*(vs_func.first) = GetProcAddress(hVSScriptDLL, vs_func.second))) {
			m_strInputInfo += _T("Failed to load vsscript functions.\n");
			return MFX_ERR_INVALID_HANDLE;
		}
	}

	//ファイルデータ読み込み
	std::ifstream inputFile(strFileName);
	if (inputFile.bad()) {
		m_strInputInfo += _T("Failed to open vpy file.\n");
		return MFX_ERR_INVALID_HANDLE;
	}
	std::istreambuf_iterator<char> data_begin(inputFile);
	std::istreambuf_iterator<char> data_end;
	std::string script_data = std::string(data_begin, data_end);
	inputFile.close();

	const VSVideoInfo *vsvideoinfo = NULL;
	const VSCoreInfo *vscoreinfo = NULL;
	if (   !vs_init()
		|| initAsyncEvents()
		|| NULL == (m_sVSapi = vs_getVSApi())
		|| vs_evaluateScript(&m_sVSscript, script_data.c_str(), NULL, efSetWorkingDir)
		|| NULL == (m_sVSnode = vs_getOutput(m_sVSscript, 0))
		|| NULL == (vsvideoinfo = m_sVSapi->getVideoInfo(m_sVSnode))
		|| NULL == (vscoreinfo = m_sVSapi->getCoreInfo(vs_getCore(m_sVSscript)))) {
		m_strInputInfo += _T("VapourSynth Initialize Error.\n");
		if (m_sVSscript) {
#if UNICODE
			WCHAR buf[1024];
			MultiByteToWideChar(CP_THREAD_ACP, MB_PRECOMPOSED, vs_getError(m_sVSscript), -1, buf, _countof(buf));
			m_strInputInfo += buf;
#else
			m_strInputInfo += vs_getError(m_sVSscript);
#endif
			m_strInputInfo += _T("\n");
		}
		return MFX_ERR_NULL_PTR;
	}
	if (vscoreinfo->api < 3) {
		m_strInputInfo += _T("VapourSynth API v3 or later is necessary.\n");
		return MFX_ERR_INCOMPATIBLE_VIDEO_PARAM;
	}

	if (vsvideoinfo->height <= 0 || vsvideoinfo->width <= 0) {
		m_strInputInfo += _T("Variable resolution is not supported.\n");
		return MFX_ERR_INCOMPATIBLE_VIDEO_PARAM;
	}

	if (vsvideoinfo->numFrames == 0) {
		m_strInputInfo += _T("Length of input video is unknown.\n");
		return MFX_ERR_INCOMPATIBLE_VIDEO_PARAM;
	}

	if (!vsvideoinfo->format) {
		m_strInputInfo += _T("Variable colorformat is not supported.\n");
		return MFX_ERR_INCOMPATIBLE_VIDEO_PARAM;
	}

	if (pfYUV420P8 != vsvideoinfo->format->id) {
		m_strInputInfo += _T("Invalid colorformat.\n");
		return MFX_ERR_INVALID_COLOR_FORMAT;
	}

	if (vsvideoinfo->fpsNum <= 0 || vsvideoinfo->fpsDen <= 0) {
		m_strInputInfo += _T("Invalid framerate.\n");
		return MFX_ERR_INCOMPATIBLE_VIDEO_PARAM;
	}
	const mfxI64 fps_gcd = GCDI64(vsvideoinfo->fpsNum, vsvideoinfo->fpsDen);

	m_inputFrameInfo.Width = (mfxU16)vsvideoinfo->width;
	m_inputFrameInfo.Height = (mfxU16)vsvideoinfo->height;
	m_inputFrameInfo.CropW = m_inputFrameInfo.Width - (pInputCrop->left + pInputCrop->right);
	m_inputFrameInfo.CropH = m_inputFrameInfo.Height - (pInputCrop->up + pInputCrop->bottom);
	m_inputFrameInfo.FrameRateExtN = (mfxU32)(vsvideoinfo->fpsNum / fps_gcd);
	m_inputFrameInfo.FrameRateExtD = (mfxU32)(vsvideoinfo->fpsDen / fps_gcd);
	*(DWORD*)&m_inputFrameInfo.FrameId = vsvideoinfo->numFrames;
	m_nAsyncFrames = vsvideoinfo->numFrames;
	m_nAsyncFrames = (std::min)(m_nAsyncFrames, vscoreinfo->numThreads);
	m_nAsyncFrames = (std::min)(m_nAsyncFrames, ASYNC_BUFFER_SIZE-1);
	if (!use_mt_mode)
		m_nAsyncFrames = (std::max)(m_nAsyncFrames, 1);

	m_ColorFormat = MFX_FOURCC_YV12;
	m_inputFrameInfo.FourCC = MFX_FOURCC_NV12;

	for (int i = 0; i < m_nAsyncFrames; i++)
		m_sVSapi->getFrameAsync(i, m_sVSnode, frameDoneCallback, this);

	TCHAR mes[256];
	TCHAR rev_info[128] = { 0 };
	int rev = getRevInfo(vscoreinfo->versionString);
	if (0 != rev)
		_stprintf_s(rev_info, _countof(rev_info), _T(" r%d"), rev);
	_stprintf_s(mes, _countof(mes), _T("VapourSynth%s%s (%s) -> %s, %dx%d, %d/%d fps"), (use_mt_mode) ? _T("MT") : _T(""), rev_info, ColorFormatToStr(m_ColorFormat), ColorFormatToStr(m_inputFrameInfo.FourCC),
		m_inputFrameInfo.Width, m_inputFrameInfo.Height, m_inputFrameInfo.FrameRateExtN, m_inputFrameInfo.FrameRateExtD);
	m_strInputInfo += mes;
	m_tmLastUpdate = timeGetTime();

	m_bInited = true;
	return MFX_ERR_NONE;
}

void CVSReader::Close() {
	if (m_sVSapi && m_sVSnode)
		m_sVSapi->freeNode(m_sVSnode);
	if (m_sVSscript)
		vs_freeScript(m_sVSscript);
	if (m_sVSapi)
		vs_finalize();
	closeAsyncEvents();
	if (hVSScriptDLL)
		FreeLibrary(hVSScriptDLL);

	hVSScriptDLL = NULL;
	vs_init = NULL;
	vs_finalize = NULL;
	vs_evaluateScript = NULL;
	vs_freeScript = NULL;
	vs_getError = NULL;
	vs_getOutput = NULL;
	vs_clearOutput = NULL;
	vs_getCore = NULL;
	vs_getVSApi = NULL;

	m_sVSapi = NULL;
	m_sVSscript = NULL;
	m_sVSnode = NULL;
	m_nAsyncFrames = 0;

	m_bInited = false;
	bufSize = 0;
	buffer = NULL;
}

mfxStatus CVSReader::LoadNextFrame(mfxFrameSurface1* pSurface) {
#ifdef _DEBUG
	MSDK_CHECK_ERROR(m_bInited, false, MFX_ERR_NOT_INITIALIZED);
#endif
	int w, h, pitch;
	mfxU8 *ptr_dst, *ptr_dst2, *ptr_dst_fin;
	mfxFrameInfo* pInfo = &pSurface->Info;
	mfxFrameData* pData = &pSurface->Data;

	mfxU16 CropLeft = m_sInputCrop.left;
	mfxU16 CropUp = m_sInputCrop.up;
	mfxU16 CropRight = m_sInputCrop.right;
	mfxU16 CropBottom = m_sInputCrop.bottom;

	if (m_pEncSatusInfo->m_nInputFrames >= *(DWORD*)&m_inputFrameInfo.FrameId)
		return MFX_ERR_MORE_DATA;

	//this reader supports only NV12 mfx surfaces for code transparency,
	//other formats may be added if application requires such functionality
	mfxU32 FourCCRequired = pInfo->FourCC;
	//if (MFX_FOURCC_NV12 != FourCCRequired && MFX_FOURCC_YV12 != FourCCRequired)
	//	return MFX_ERR_UNSUPPORTED;

	if (pInfo->CropH > 0 && pInfo->CropW > 0) {
		w = pInfo->CropW;
		h = pInfo->CropH;
	} else {
		w = pInfo->Width;
		h = pInfo->Height;
	}
	w += (CropLeft + CropRight);
	h += (CropUp + CropBottom);

	pitch = pData->Pitch;
	ptr_dst = pData->Y + pInfo->CropX + pInfo->CropY * pData->Pitch;

	//const VSFrameRef *src_frame = m_sVSapi->getFrame(m_pEncSatusInfo->m_nInputFrames, m_sVSnode, "failed to get frame", NULL);
	const VSFrameRef *src_frame = getFrameFromAsyncBuffer(m_pEncSatusInfo->m_nInputFrames);
	if (NULL == src_frame) {
		return MFX_ERR_MORE_DATA;
	}

	// color format of data in the input file
	if (m_ColorFormat == MFX_FOURCC_YV12) {
		int src_pitch_y = m_sVSapi->getStride(src_frame, 0);
		const mfxU8 *ptr_y = m_sVSapi->getReadPtr(src_frame, 0);
		//copy luma
		for (int y = 0, y_fin = h - CropUp - CropBottom; y < y_fin; y++)
			sse_memcpy(ptr_dst + y * pitch, ptr_y + (y + CropUp) * src_pitch_y + CropLeft, w - CropLeft - CropRight);

		//copy chroma
		int src_pitch_c = m_sVSapi->getStride(src_frame, 1);
		const mfxU8 *ptr_u = m_sVSapi->getReadPtr(src_frame, 1);
		const mfxU8 *ptr_v = m_sVSapi->getReadPtr(src_frame, 2);
		if (FourCCRequired == MFX_FOURCC_NV12) {
			ptr_dst = pData->UV + pInfo->CropX + (pInfo->CropY>>1) * pitch;

			const mfxU8 *bufV = ptr_v;
			const mfxU8 *bufU = ptr_u;

			h >>= 1;
			w >>= 1;
			CropBottom >>= 1;
			CropUp >>= 1;
			CropLeft >>= 1;
			CropRight >>= 1;

			if (((mfxU32)ptr_dst & 0x0F) == 0x00) {
				__m128i x0, x1, x2;
				for (int y = 0, y_fin = h - CropBottom - CropUp; y < y_fin; y++) {
					const mfxU8 *U = bufU + (y + CropUp) * src_pitch_c + CropLeft;
					const mfxU8 *V = bufV + (y + CropUp) * src_pitch_c + CropLeft;
					ptr_dst2 = ptr_dst + y * pitch;
					ptr_dst_fin = ptr_dst2 + ((w - CropRight - CropLeft)<<1);
					for (; ptr_dst2 < ptr_dst_fin; ptr_dst2 += 32, U += 16, V += 16) {
						x0 = _mm_loadu_si128((const __m128i*)U);
						x1 = _mm_loadu_si128((const __m128i*)V);

						x2 = _mm_unpackhi_epi8(x0, x1);
						x0 = _mm_unpacklo_epi8(x0, x1);

						_mm_store_si128((__m128i *)(ptr_dst2 +  0), x0);
						_mm_store_si128((__m128i *)(ptr_dst2 + 16), x2);
					}
				}

			} else {

				__m128i x0, x1, x2;
				for (int y = 0, y_fin = h - CropBottom - CropUp; y < y_fin; y++) {
					const mfxU8 *U = bufU + (y + CropUp) * src_pitch_c + CropLeft;
					const mfxU8 *V = bufV + (y + CropUp) * src_pitch_c + CropLeft;
					ptr_dst2 = ptr_dst + y * pitch;
					ptr_dst_fin = ptr_dst2 + ((w - CropRight - CropLeft)<<1);
					for (; ptr_dst2 < ptr_dst_fin; ptr_dst2 += 32, U += 16, V += 16) {
						x0 = _mm_loadu_si128((const __m128i*)U);
						x1 = _mm_loadu_si128((const __m128i*)V);

						x2 = _mm_unpackhi_epi8(x0, x1);
						x0 = _mm_unpacklo_epi8(x0, x1);

						_mm_storeu_si128((__m128i *)(ptr_dst2 +  0), x0);
						_mm_storeu_si128((__m128i *)(ptr_dst2 + 16), x2);
					}
				}

			}
		} else if (FourCCRequired == MFX_FOURCC_YV12) {
			ptr_dst = pData->V + (pInfo->CropX / 2) + (pInfo->CropY / 2) * pitch;

			h >>= 1;
			w >>= 1;
			CropUp >>= 1;
			CropBottom >>= 1;
			CropLeft >>= 1;
			CropRight >>= 1;
			for (int y = 0, y_fin = h - CropUp - CropBottom; y < y_fin; y++)
				sse_memcpy(ptr_dst + (y * pitch >> 1), ptr_v + (y + CropUp) * src_pitch_y + CropLeft, w - CropLeft - CropRight);

			ptr_dst = pData->U + (pInfo->CropX / 2) + (pInfo->CropY / 2) * pitch;
			for (int y = 0, y_fin = h - CropUp - CropBottom; y < y_fin; y++)
				sse_memcpy(ptr_dst + (y * pitch >> 1), ptr_u + (y + CropUp) * src_pitch_c + CropLeft, w - CropLeft - CropRight);
		} else {
			return MFX_ERR_UNSUPPORTED;
		}
	} else {
		return MFX_ERR_UNSUPPORTED;
	}

	m_sVSapi->freeFrame(src_frame);

	m_pEncSatusInfo->m_nInputFrames++;

	// display update
	mfxU32 tm = timeGetTime();
	if (tm - m_tmLastUpdate > UPDATE_INTERVAL) {
		m_tmLastUpdate = tm;
		m_pEncSatusInfo->UpdateDisplay(tm, 0);
	}
	return MFX_ERR_NONE;
}

#endif //ENABLE_VAPOURSYNTH_READER
