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
	memset(m_hAsyncEventFrameSetFin,   0, sizeof(m_hAsyncEventFrameSetFin));
	memset(m_hAsyncEventFrameSetStart, 0, sizeof(m_hAsyncEventFrameSetStart));
	
	m_bAbortAsync = false;
	m_nCopyOfInputFrames = 0;
	
	memset(&m_sVS, 0, sizeof(m_sVS));
}

CVSReader::~CVSReader() {
	Close();
}

void CVSReader::release_vapoursynth() {
	if (m_sVS.hVSScriptDLL)
		FreeLibrary(m_sVS.hVSScriptDLL);

	memset(&m_sVS, 0, sizeof(m_sVS));
}

int CVSReader::load_vapoursynth() {
	release_vapoursynth();
	
	if (NULL == (m_sVS.hVSScriptDLL = LoadLibrary(_T("vsscript.dll")))) {
		m_strInputInfo += _T("Failed to load vsscript.dll.\n");
		return 1;
	}

	std::map<void **, const char*> vs_func_list = {
		{ (void **)&m_sVS.init,           (VPY_X64) ? "vsscript_init"           : "_vsscript_init@0"            },
		{ (void **)&m_sVS.finalize,       (VPY_X64) ? "vsscript_finalize"       : "_vsscript_finalize@0",       },
		{ (void **)&m_sVS.evaluateScript, (VPY_X64) ? "vsscript_evaluateScript" : "_vsscript_evaluateScript@16" },
		{ (void **)&m_sVS.evaluateFile,   (VPY_X64) ? "vsscript_evaluateFile"   : "_vsscript_evaluateFile@12"   },
		{ (void **)&m_sVS.freeScript,     (VPY_X64) ? "vsscript_freeScript"     : "_vsscript_freeScript@4"      },
		{ (void **)&m_sVS.getError,       (VPY_X64) ? "vsscript_getError"       : "_vsscript_getError@4"        },
		{ (void **)&m_sVS.getOutput,      (VPY_X64) ? "vsscript_getOutput"      : "_vsscript_getOutput@8"       },
		{ (void **)&m_sVS.clearOutput,    (VPY_X64) ? "vsscript_clearOutput"    : "_vsscript_clearOutput@8"     },
		{ (void **)&m_sVS.getCore,        (VPY_X64) ? "vsscript_getCore"        : "_vsscript_getCore@4"         },
		{ (void **)&m_sVS.getVSApi,       (VPY_X64) ? "vsscript_getVSApi"       : "_vsscript_getVSApi@0"        },
	};

	for (auto vs_func : vs_func_list) {
		if (NULL == (*(vs_func.first) = GetProcAddress(m_sVS.hVSScriptDLL, vs_func.second))) {
			m_strInputInfo += _T("Failed to load vsscript functions.\n");
			return 1;
		}
	}
	return 0;
}

int CVSReader::initAsyncEvents() {
	for (int i = 0; i < _countof(m_hAsyncEventFrameSetFin); i++) {
		if (   NULL == (m_hAsyncEventFrameSetFin[i]   = CreateEvent(NULL, FALSE, FALSE, NULL))
			|| NULL == (m_hAsyncEventFrameSetStart[i] = CreateEvent(NULL, FALSE, TRUE,  NULL)))
			return 1;
	}
	return 0;
}
void CVSReader::closeAsyncEvents() {
	m_bAbortAsync = true;
	for (int i_frame = m_nCopyOfInputFrames; i_frame < m_nAsyncFrames; i_frame++) {
		if (m_hAsyncEventFrameSetFin[i_frame & (ASYNC_BUFFER_SIZE-1)])
			WaitForSingleObject(m_hAsyncEventFrameSetFin[i_frame & (ASYNC_BUFFER_SIZE-1)], INFINITE);
	}
	for (int i = 0; i < _countof(m_hAsyncEventFrameSetFin); i++) {
		if (m_hAsyncEventFrameSetFin[i])
			CloseHandle(m_hAsyncEventFrameSetFin[i]);
		if (m_hAsyncEventFrameSetStart[i])
			CloseHandle(m_hAsyncEventFrameSetStart[i]);
	}
	memset(m_hAsyncEventFrameSetFin,   0, sizeof(m_hAsyncEventFrameSetFin));
	memset(m_hAsyncEventFrameSetStart, 0, sizeof(m_hAsyncEventFrameSetStart));
	m_bAbortAsync = false;
}

#pragma warning(push)
#pragma warning(disable:4100)
void __stdcall frameDoneCallback(void *userData, const VSFrameRef *f, int n, VSNodeRef *, const char *errorMsg) {
	reinterpret_cast<CVSReader*>(userData)->setFrameToAsyncBuffer(n, f);
}
#pragma warning(pop)

void CVSReader::setFrameToAsyncBuffer(int n, const VSFrameRef* f) {
	WaitForSingleObject(m_hAsyncEventFrameSetStart[n & (ASYNC_BUFFER_SIZE-1)], INFINITE);
	m_pAsyncBuffer[n & (ASYNC_BUFFER_SIZE-1)] = f;
	SetEvent(m_hAsyncEventFrameSetFin[n & (ASYNC_BUFFER_SIZE-1)]);

	if (m_nAsyncFrames < *(int*)&m_inputFrameInfo.FrameId && !m_bAbortAsync) {
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
			for (char *s = buf; *s; s++)
				*s = (char)tolower(*s);
			int rev = 0;
			return (1 == sscanf_s(buf, "core r%d", &rev)) ? rev : 0;
		}
		p = NULL;
	}
	return 0;
}

#pragma warning(push)
#pragma warning(disable:4100)
mfxStatus CVSReader::Init(const TCHAR *strFileName, mfxU32 ColorFormat, const void *option, CEncodingThread *pEncThread, CEncodeStatusInfo *pEncSatusInfo, sInputCrop *pInputCrop) {
	MSDK_CHECK_POINTER(strFileName, MFX_ERR_NULL_PTR);
	MSDK_CHECK_ERROR(_tclen(strFileName), 0, MFX_ERR_NULL_PTR);

	Close();

	MSDK_CHECK_POINTER(pEncThread, MFX_ERR_NULL_PTR);
	m_pEncThread = pEncThread;

	MSDK_CHECK_POINTER(pEncSatusInfo, MFX_ERR_NULL_PTR);
	m_pEncSatusInfo = pEncSatusInfo;

	MSDK_CHECK_POINTER(pInputCrop, MFX_ERR_NULL_PTR);
	memcpy(&m_sInputCrop, pInputCrop, sizeof(m_sInputCrop));

	const bool use_mt_mode = ((VSReaderPrm *)option)->use_mt;
	
	if (load_vapoursynth()) {
		return MFX_ERR_NULL_PTR;
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
	if (   !m_sVS.init()
		|| initAsyncEvents()
		|| NULL == (m_sVSapi = m_sVS.getVSApi())
		|| m_sVS.evaluateScript(&m_sVSscript, script_data.c_str(), NULL, efSetWorkingDir)
		|| NULL == (m_sVSnode = m_sVS.getOutput(m_sVSscript, 0))
		|| NULL == (vsvideoinfo = m_sVSapi->getVideoInfo(m_sVSnode))
		|| NULL == (vscoreinfo = m_sVSapi->getCoreInfo(m_sVS.getCore(m_sVSscript)))) {
		m_strInputInfo += _T("VapourSynth Initialize Error.\n");
		if (m_sVSscript) {
#if UNICODE
			WCHAR buf[1024];
			MultiByteToWideChar(CP_THREAD_ACP, MB_PRECOMPOSED, m_sVS.getError(m_sVSscript), -1, buf, _countof(buf));
			m_strInputInfo += buf;
#else
			m_strInputInfo += m_sVS.getError(m_sVSscript);
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
	

	typedef struct CSPMap {
		int fmtID;
		mfxU32 in, out;
	} CSPMap;

	static const std::vector<CSPMap> valid_csp_list = {
		{ pfYUV420P8, MFX_FOURCC_YV12, MFX_FOURCC_NV12 },
	};

	m_ColorFormat = 0x00;
	for (auto csp : valid_csp_list) {
		if (csp.fmtID == vsvideoinfo->format->id) {
			m_ColorFormat = csp.in;
			m_inputFrameInfo.FourCC = csp.out;
			m_sConvert = get_convert_csp_func(csp.in, csp.out, false);
			break;
		}
	}

	if (0x00 == m_ColorFormat || nullptr == m_sConvert) {
		m_strInputInfo += _T("invalid colorformat.\n");
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
		m_nAsyncFrames = 1;

	for (int i = 0; i < m_nAsyncFrames; i++)
		m_sVSapi->getFrameAsync(i, m_sVSnode, frameDoneCallback, this);

	TCHAR mes[256];
	TCHAR rev_info[128] = { 0 };
	int rev = getRevInfo(vscoreinfo->versionString);
	if (0 != rev)
		_stprintf_s(rev_info, _countof(rev_info), _T(" r%d"), rev);
	_stprintf_s(mes, _countof(mes), _T("VapourSynth%s%s (%s)->%s[%s], %dx%d, %d/%d fps"), (use_mt_mode) ? _T("MT") : _T(""), rev_info, ColorFormatToStr(m_ColorFormat), ColorFormatToStr(m_inputFrameInfo.FourCC), get_simd_str(m_sConvert->simd),
		m_inputFrameInfo.Width, m_inputFrameInfo.Height, m_inputFrameInfo.FrameRateExtN, m_inputFrameInfo.FrameRateExtD);
	m_strInputInfo += mes;
	m_tmLastUpdate = timeGetTime();

	m_bInited = true;
	return MFX_ERR_NONE;
}

void CVSReader::Close() {
	closeAsyncEvents();
	if (m_sVSapi && m_sVSnode)
		m_sVSapi->freeNode(m_sVSnode);
	if (m_sVSscript)
		m_sVS.freeScript(m_sVSscript);
	if (m_sVSapi)
		m_sVS.finalize();

	release_vapoursynth();

	m_bAbortAsync = false;
	m_nCopyOfInputFrames = 0;

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

	const VSFrameRef *src_frame = getFrameFromAsyncBuffer(m_pEncSatusInfo->m_nInputFrames);
	if (NULL == src_frame) {
		return MFX_ERR_MORE_DATA;
	}

	BOOL interlaced = 0 != (pSurface->Info.PicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF));
	int crop[4] = { CropLeft, CropUp, CropRight, CropBottom };
	const void *dst_ptr[3] = { pData->Y, pData->UV, NULL };
	const void *src_ptr[3] = { m_sVSapi->getReadPtr(src_frame, 0), m_sVSapi->getReadPtr(src_frame, 1), m_sVSapi->getReadPtr(src_frame, 2) };
	m_sConvert->func[interlaced]((void **)dst_ptr, (void **)src_ptr, w, m_sVSapi->getStride(src_frame, 0), m_sVSapi->getStride(src_frame, 1), pData->Pitch, h, crop);

	m_sVSapi->freeFrame(src_frame);

	m_pEncSatusInfo->m_nInputFrames++;
	m_nCopyOfInputFrames = m_pEncSatusInfo->m_nInputFrames;

	// display update
	mfxU32 tm = timeGetTime();
	if (tm - m_tmLastUpdate > UPDATE_INTERVAL) {
		m_tmLastUpdate = tm;
		m_pEncSatusInfo->UpdateDisplay(tm, 0);
	}
	return MFX_ERR_NONE;
}

#endif //ENABLE_VAPOURSYNTH_READER
