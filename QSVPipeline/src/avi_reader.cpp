//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#include "avi_reader.h"
#if ENABLE_AVI_READER

#define ENABLE_AVI_DIRECT_MEMCPY 0

CAVIReader::CAVIReader() {
	m_pAviFile = NULL;
	m_pAviStream = NULL;
	m_pGetFrame = NULL;
	m_pBitmapInfoHeader = NULL;
}

CAVIReader::~CAVIReader() {
	Close();
}

#pragma warning(push)
#pragma warning(disable:4100)
mfxStatus CAVIReader::Init(const TCHAR *strFileName, mfxU32 ColorFormat, int option, CEncodingThread *pEncThread, CEncodeStatusInfo *pEncSatusInfo, sInputCrop *pInputCrop) {

	MSDK_CHECK_POINTER(strFileName, MFX_ERR_NULL_PTR);
	MSDK_CHECK_ERROR(_tclen(strFileName), 0, MFX_ERR_NULL_PTR);

	Close();

	MSDK_CHECK_POINTER(pEncThread, MFX_ERR_NULL_PTR);
	m_pEncThread = pEncThread;

	MSDK_CHECK_POINTER(pEncSatusInfo, MFX_ERR_NULL_PTR);
	m_pEncSatusInfo = pEncSatusInfo;

	
	MSDK_CHECK_POINTER(pInputCrop, MFX_ERR_NULL_PTR);
	memcpy(&m_sInputCrop, pInputCrop, sizeof(m_sInputCrop));
	
	AVIFileInit();

	if (0 != AVIFileOpen(&m_pAviFile, strFileName, OF_READ | OF_SHARE_DENY_NONE, NULL)) {
		m_strInputInfo += _T("avi: failed to open avi file.\n");
		return MFX_ERR_INVALID_HANDLE;
	}

	AVIFILEINFO finfo = { 0 };
	if (0 != AVIFileInfo(m_pAviFile, &finfo, sizeof(AVIFILEINFO))) {
		m_strInputInfo += _T("avi: failed to get avi file info.\n");
		return MFX_ERR_INVALID_HANDLE;
	}
	for (DWORD i_stream = 0; i_stream < finfo.dwStreams; i_stream++) {
		if (0 != AVIFileGetStream(m_pAviFile, &m_pAviStream, 0, i_stream))
			return MFX_ERR_INVALID_HANDLE;
		AVISTREAMINFO sinfo = { 0 };
		if (0 == AVIStreamInfo(m_pAviStream, &sinfo, sizeof(AVISTREAMINFO)) && sinfo.fccType == streamtypeVIDEO) {
			memset(&m_inputFrameInfo, 0, sizeof(m_inputFrameInfo));
			const DWORD fps_gcd = GCD(sinfo.dwRate, sinfo.dwScale);
			m_inputFrameInfo.Width = (mfxU16)(sinfo.rcFrame.right - sinfo.rcFrame.left);
			m_inputFrameInfo.Height = (mfxU16)(sinfo.rcFrame.bottom - sinfo.rcFrame.top);
			m_inputFrameInfo.CropW = m_inputFrameInfo.Width - (pInputCrop->left + pInputCrop->right);
			m_inputFrameInfo.CropH = m_inputFrameInfo.Height - (pInputCrop->up + pInputCrop->bottom);
			m_inputFrameInfo.FrameRateExtN = sinfo.dwRate / fps_gcd;
			m_inputFrameInfo.FrameRateExtD = sinfo.dwScale / fps_gcd;
			*(DWORD*)&m_inputFrameInfo.FrameId = sinfo.dwLength - sinfo.dwStart;
			m_ColorFormat = sinfo.fccHandler;
			TCHAR fcc[5] = { 0 };
#if defined(UNICODE) || defined(_UNICODE)
			MultiByteToWideChar(CP_THREAD_ACP, MB_PRECOMPOSED, (char *)&sinfo.fccHandler, 4, fcc, _countof(fcc));
#else
			*(DWORD *)fcc = sinfo.fccHandler;
#endif
			m_strInputInfo += _T("avi: ");
			m_strInputInfo += fcc;
			break;
		}
		AVIStreamRelease(m_pAviStream);
		m_pAviStream = NULL;
	}
	if (m_pAviStream == NULL) {
		m_strInputInfo += _T("\navi: failed to get valid stream from avi file.\n");
		return MFX_ERR_INVALID_HANDLE;
	}

	if (   m_ColorFormat == MFX_FOURCC_YUY2
		|| m_ColorFormat == MFX_FOURCC_YV12) {
		//何もしない
	} else {
		BITMAPINFOHEADER bih[4] = {
			{ sizeof(BITMAPINFOHEADER), 0, 0, 1, 12, MFX_FOURCC_YV12, m_inputFrameInfo.Width * m_inputFrameInfo.Height * 3/2, 0, 0, 0, 0 },
			{ sizeof(BITMAPINFOHEADER), 0, 0, 1, 16, MFX_FOURCC_YUY2, m_inputFrameInfo.Width * m_inputFrameInfo.Height * 2,   0, 0, 0, 0 },
			{ sizeof(BITMAPINFOHEADER), 0, 0, 1, 24, BI_RGB,          m_inputFrameInfo.Width * m_inputFrameInfo.Height * 3,   0, 0, 0, 0 },
			{ sizeof(BITMAPINFOHEADER), 0, 0, 1, 32, BI_RGB,          m_inputFrameInfo.Width * m_inputFrameInfo.Height * 3,   0, 0, 0, 0 }
		};
		for (int i = 0; i < _countof(bih); i++) {
			if (NULL == (m_pGetFrame = AVIStreamGetFrameOpen(m_pAviStream, &bih[i]))) {
				continue;
			}
			m_ColorFormat = (bih[i].biCompression == BI_RGB) ? ((bih[i].biBitCount == 24) ? MFX_FOURCC_RGB3 : MFX_FOURCC_RGB4) : bih[i].biCompression;
			break;
		}

		if (m_pGetFrame == NULL) {
			if (   NULL == (m_pGetFrame = AVIStreamGetFrameOpen(m_pAviStream, NULL))
				&& NULL == (m_pGetFrame = AVIStreamGetFrameOpen(m_pAviStream, (BITMAPINFOHEADER *)AVIGETFRAMEF_BESTDISPLAYFMT))) {
				m_strInputInfo += _T("\navi: failed to decode avi file.\n");
				return MFX_ERR_INVALID_HANDLE;
			}
			BITMAPINFOHEADER *bmpInfoHeader = (BITMAPINFOHEADER *)AVIStreamGetFrame(m_pGetFrame, 0);
			if (NULL == bmpInfoHeader || bmpInfoHeader->biCompression != 0) {
				m_strInputInfo += _T("\navi: failed to decode avi file.\n");
				return MFX_ERR_MORE_DATA;
			}

			m_ColorFormat = (bmpInfoHeader->biBitCount == 24) ? MFX_FOURCC_RGB3 : MFX_FOURCC_RGB4;
		}
	}

	if (   MFX_FOURCC_RGB4 == m_ColorFormat
		|| MFX_FOURCC_RGB3 == m_ColorFormat) {
		m_inputFrameInfo.FourCC = MFX_FOURCC_RGB4;
		m_inputFrameInfo.ChromaFormat = 0;
#if ENABLE_AVI_DIRECT_MEMCPY
	} else if (   MFX_FOURCC_YV12 == m_ColorFormat
		       && (m_inputFrameInfo.Width + pInputCrop->left + pInputCrop->right) % 256 == 0
			   && (m_inputFrameInfo.Height + pInputCrop->up + pInputCrop->bottom) %  32 == 0) {
		m_inputFrameInfo.CropW = m_inputFrameInfo.Width;
		m_inputFrameInfo.CropH = m_inputFrameInfo.Height;
		m_inputFrameInfo.CropX = pInputCrop->left;
		m_inputFrameInfo.CropY = pInputCrop->up;
		m_inputFrameInfo.FourCC = MFX_FOURCC_YV12;
		m_inputFrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
#endif //ENABLE_AVI_DIRECT_MEMCPY
	} else {
		m_inputFrameInfo.FourCC = MFX_FOURCC_NV12;
		m_inputFrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
	}
	TCHAR mes[256];
	_stprintf_s(mes, _countof(mes), _T("(%s) -> %s, %dx%d, %d/%d fps"), ColorFormatToStr(m_ColorFormat), ColorFormatToStr(m_inputFrameInfo.FourCC),
		m_inputFrameInfo.Width, m_inputFrameInfo.Height, m_inputFrameInfo.FrameRateExtN, m_inputFrameInfo.FrameRateExtD);
	m_strInputInfo += mes;
	m_tmLastUpdate = timeGetTime();

	m_bInited = true;
	return MFX_ERR_NONE;
}
#pragma warning(pop)

void CAVIReader::Close() {
	if (m_pGetFrame)
		AVIStreamGetFrameClose(m_pGetFrame);
	if (m_pAviStream)
		AVIStreamRelease(m_pAviStream);
	if (m_pAviFile)
		AVIFileRelease(m_pAviFile);
	if (buffer)
		_aligned_free(buffer);
    AVIFileExit();

	m_pAviFile = NULL;
	m_pAviStream = NULL;
	m_pGetFrame = NULL;
	m_pBitmapInfoHeader = NULL;
	m_bInited = false;
	bufSize = 0;
	buffer = NULL;
}

mfxStatus CAVIReader::LoadNextFrame(mfxFrameSurface1* pSurface) {
#ifdef _DEBUG
	MSDK_CHECK_ERROR(m_bInited, false, MFX_ERR_NOT_INITIALIZED);
#endif
	int w, h, pitch;
	mfxU8 *ptr_dst, *ptr_dst2, *ptr_dst_fin, *ptr_src;
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

#if ENABLE_AVI_DIRECT_MEMCPY
	if (   m_ColorFormat  == MFX_FOURCC_YV12
		&& FourCCRequired == MFX_FOURCC_YV12
		&& pitch == w
		&& pData->V - pData->Y == w * h
		&& m_pGetFrame == NULL) {
		//directly copy frame into buffer
		LONG sizeRead = 0;
		if (0 != AVIStreamRead(m_pAviStream, m_pEncSatusInfo->m_nInputFrames, 1, pData->Y, w * h * 3 / 2, &sizeRead, NULL))
			return MFX_ERR_MORE_DATA;
	} else {
#endif //#if ENABLE_AVI_DIRECT_MEMCPY
		if (m_pGetFrame) {
			if (NULL == (ptr_src = (mfxU8 *)AVIStreamGetFrame(m_pGetFrame, m_pEncSatusInfo->m_nInputFrames)))
				return MFX_ERR_MORE_DATA;
			ptr_src += sizeof(BITMAPINFOHEADER);
		} else {
			mfxU32 required_bufsize = w * h * 3;
			if (bufSize < required_bufsize) {
				if (buffer)
					_aligned_free(buffer);
				if (NULL == (buffer = (mfxU8 *)_aligned_malloc(sizeof(mfxU8) * required_bufsize, 16)))
					return MFX_ERR_MEMORY_ALLOC;
				bufSize = required_bufsize;
			}
			LONG sizeRead = 0;
			if (0 != AVIStreamRead(m_pAviStream, m_pEncSatusInfo->m_nInputFrames, 1, buffer, (LONG)bufSize, &sizeRead, NULL))
				return MFX_ERR_MORE_DATA;
			ptr_src = buffer;
		}

		switch (m_ColorFormat) // color format of data in the input file
		{
		case MFX_FOURCC_YV12: // YUV420 is implied
			//copy luma
			for (int y = 0, y_fin = h - CropUp - CropBottom; y < y_fin; y++)
				sse_memcpy(ptr_dst + y * pitch, ptr_src + (y + CropUp) * w + CropLeft, w - CropLeft - CropRight);
			//copy chroma
			switch (FourCCRequired)
			{
			case MFX_FOURCC_NV12:
				{
				ptr_dst = pData->UV + pInfo->CropX + (pInfo->CropY>>1) * pitch;

				mfxU8 *bufV = ptr_src + w * h;
				mfxU8 *bufU = bufV + ((w * h) >> 2);

				h >>= 1;
				w >>= 1;
				CropBottom >>= 1;
				CropUp >>= 1;
				CropLeft >>= 1;
				CropRight >>= 1;

				if (((mfxU32)ptr_dst & 0x0F) == 0x00) {
					__m128i x0, x1, x2;
					for (int y = 0, y_fin = h - CropBottom - CropUp; y < y_fin; y++) {
						mfxU8 *U = bufU + (y + CropUp) * w + CropLeft;
						mfxU8 *V = bufV + (y + CropUp) * w + CropLeft;
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
						mfxU8 *U = bufU + (y + CropUp) * w + CropLeft;
						mfxU8 *V = bufV + (y + CropUp) * w + CropLeft;
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
				}
				break;
			case MFX_FOURCC_YV12:
				ptr_src += w * h;
				ptr_dst = pData->V + (pInfo->CropX / 2) + (pInfo->CropY / 2) * pitch;

				h >>= 1;
				w >>= 1;
				CropUp >>= 1;
				CropBottom >>= 1;
				CropLeft >>= 1;
				CropRight >>= 1;
				for (int y = 0, y_fin = h - CropUp - CropBottom; y < y_fin; y++)
					sse_memcpy(ptr_dst + (y * pitch >> 1), ptr_src + (y + CropUp) * w + CropLeft, w - CropLeft - CropRight);

				ptr_src += (w * h);
				ptr_dst = pData->U + (pInfo->CropX / 2) + (pInfo->CropY / 2) * pitch;
				for (int y = 0, y_fin = h - CropUp - CropBottom; y < y_fin; y++)
					sse_memcpy(ptr_dst + (y * pitch >> 1), ptr_src + (y + CropUp) * w + CropLeft, w - CropLeft - CropRight);
				break;
			default:
				return MFX_ERR_UNSUPPORTED;
			}
			break;
		case MFX_FOURCC_YUY2:
			switch (FourCCRequired)
			{
			case MFX_FOURCC_NV12:
				ptr_dst2 = pData->UV + pInfo->CropX + (pInfo->CropY>>1) * pitch;
				if (pSurface->Info.PicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF)) {
					mfxU8 *p, *pw, *Y, *C;
					__m128i x0, x1, x2, x3;
					static const _declspec(align(16)) BYTE  Array_INTERLACE_WEIGHT[2][16] = { 
						{1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3},
						{3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1}
					};
					for (int y = 0, y_fin = h - CropUp - CropBottom; y < y_fin; y += 4) {
						for (int i = 0; i < 2; i++) {
							p = ptr_src + (((y + i + CropUp) * w + CropLeft) << 1);
							pw = p + (w << 2);
							Y = ptr_dst + ((y + i) * pitch);
							C = ptr_dst2 + (((y + i*2) * pitch) >> 1);
							for (int x = 0, x_fin = w - CropLeft - CropRight; x < x_fin; x += 16, p += 32, pw += 32) {
								//-----------    1行目   ---------------
								x0 = _mm_loadu_si128((__m128i *)(p+ 0));    // VYUYVYUYVYUYVYUY
								x1 = _mm_loadu_si128((__m128i *)(p+16));    // VYUYVYUYVYUYVYUY

								_mm_prefetch((const char *)pw, _MM_HINT_T1);

								x2 = _mm_unpacklo_epi8(x0, x1); //VVYYUUYYVVYYUUYY
								x1 = _mm_unpackhi_epi8(x0, x1); //VVYYUUYYVVYYUUYY

								x0 = _mm_unpacklo_epi8(x2, x1); //VVVVYYYYUUUUYYYY
								x1 = _mm_unpackhi_epi8(x2, x1); //VVVVYYYYUUUUYYYY

								x2 = _mm_unpacklo_epi8(x0, x1); //UUUUUUUUYYYYYYYY
								x1 = _mm_unpackhi_epi8(x0, x1); //VVVVVVVVYYYYYYYY

								x0 = _mm_unpacklo_epi8(x2, x1); //YYYYYYYYYYYYYYYY
								x3 = _mm_unpackhi_epi8(x2, x1); //VUVUVUVUVUVUVUVU

								_mm_storeu_si128((__m128i *)(Y + x), x0);
								//-----------1行目終了---------------

								//-----------3行目---------------
								x0 = _mm_loadu_si128((__m128i *)(pw+ 0));    // VYUYVYUYVYUYVYUY
								x1 = _mm_loadu_si128((__m128i *)(pw+16));    // VYUYVYUYVYUYVYUY

								x2 = _mm_unpacklo_epi8(x0, x1); //VVYYUUYYVVYYUUYY
								x1 = _mm_unpackhi_epi8(x0, x1); //VVYYUUYYVVYYUUYY

								x0 = _mm_unpacklo_epi8(x2, x1); //VVVVYYYYUUUUYYYY
								x1 = _mm_unpackhi_epi8(x2, x1); //VVVVYYYYUUUUYYYY

								x2 = _mm_unpacklo_epi8(x0, x1); //UUUUUUUUYYYYYYYY
								x1 = _mm_unpackhi_epi8(x0, x1); //VVVVVVVVYYYYYYYY

								x0 = _mm_unpacklo_epi8(x2, x1); //YYYYYYYYYYYYYYYY
								x1 = _mm_unpackhi_epi8(x2, x1); //VUVUVUVUVUVUVUVU

								_mm_storeu_si128((__m128i *)(Y + (pitch<<1) + x), x0);
								//-----------3行目終了---------------

								x0 = _mm_unpacklo_epi8(x1, x3);
								x1 = _mm_unpackhi_epi8(x1, x3);
								x0 = _mm_maddubs_epi16(x0, _mm_load_si128((__m128i*)Array_INTERLACE_WEIGHT[i]));
								x1 = _mm_maddubs_epi16(x1, _mm_load_si128((__m128i*)Array_INTERLACE_WEIGHT[i]));
								x0 = _mm_add_epi16(x0, _mm_set1_epi16(2));
								x1 = _mm_add_epi16(x1, _mm_set1_epi16(2));
								x0 = _mm_srai_epi16(x0, 2);
								x1 = _mm_srai_epi16(x1, 2);
								x0 = _mm_packus_epi16(x0, x1); //VUVUVUVUVUVUVUVU
								_mm_storeu_si128((__m128i *)(C + x), x0);
							}
						}
					}

				} else {

					mfxU8 *p, *pw, *Y, *C;
					__m128i x0, x1, x2, x3;
					for (int y = 0, y_fin = h - CropUp - CropBottom; y < y_fin; y += 2) {
						p = ptr_src + (((y + CropUp) * w + CropLeft) << 1);
						pw = p + (w << 1);
						Y = ptr_dst + (y * pitch);
						C = ptr_dst2 + ((y * pitch) >> 1);
						for (int x = 0, x_fin = w - CropLeft - CropRight; x < x_fin; x += 16, p += 32, pw += 32) {
							//-----------1行目---------------
							x0 = _mm_loadu_si128((const __m128i *)(p+ 0));    // VYUYVYUYVYUYVYUY
							x1 = _mm_loadu_si128((const __m128i *)(p+16));    // VYUYVYUYVYUYVYUY

							_mm_prefetch((const char *)pw, _MM_HINT_T1);

							x2 = _mm_unpacklo_epi8(x0, x1); //VVYYUUYYVVYYUUYY
							x1 = _mm_unpackhi_epi8(x0, x1); //VVYYUUYYVVYYUUYY

							x0 = _mm_unpacklo_epi8(x2, x1); //VVVVYYYYUUUUYYYY
							x1 = _mm_unpackhi_epi8(x2, x1); //VVVVYYYYUUUUYYYY

							x2 = _mm_unpacklo_epi8(x0, x1); //UUUUUUUUYYYYYYYY
							x1 = _mm_unpackhi_epi8(x0, x1); //VVVVVVVVYYYYYYYY

							x0 = _mm_unpacklo_epi8(x2, x1); //YYYYYYYYYYYYYYYY
							x3 = _mm_unpackhi_epi8(x2, x1); //VUVUVUVUVUVUVUVU

							_mm_storeu_si128((__m128i *)(Y + x), x0);
							//-----------1行目終了---------------

							//-----------2行目---------------
							x0 = _mm_loadu_si128((const __m128i *)(pw+ 0));    // VYUYVYUYVYUYVYUY
							x1 = _mm_loadu_si128((const __m128i *)(pw+16));    // VYUYVYUYVYUYVYUY

							x2 = _mm_unpacklo_epi8(x0, x1); //VVYYUUYYVVYYUUYY
							x1 = _mm_unpackhi_epi8(x0, x1); //VVYYUUYYVVYYUUYY

							x0 = _mm_unpacklo_epi8(x2, x1); //VVVVYYYYUUUUYYYY
							x1 = _mm_unpackhi_epi8(x2, x1); //VVVVYYYYUUUUYYYY

							x2 = _mm_unpacklo_epi8(x0, x1); //UUUUUUUUYYYYYYYY
							x1 = _mm_unpackhi_epi8(x0, x1); //VVVVVVVVYYYYYYYY

							x0 = _mm_unpacklo_epi8(x2, x1); //YYYYYYYYYYYYYYYY
							x1 = _mm_unpackhi_epi8(x2, x1); //VUVUVUVUVUVUVUVU

							_mm_storeu_si128((__m128i *)(Y + pitch + x), x0);
							//-----------2行目終了---------------

							x1 = _mm_avg_epu8(x1, x3);  //VUVUVUVUVUVUVUVU
							_mm_storeu_si128((__m128i *)(C + x), x1);
						}
					}
				}
				break;
			default:
				return MFX_ERR_UNSUPPORTED;
			}
			break;
		case MFX_FOURCC_RGB3:
			switch (FourCCRequired)
			{
			case MFX_FOURCC_RGB4:
				{
					mfxU8 *dstR = min( min(pData->R, pData->G), pData->B ) + pInfo->CropX + pInfo->CropY * pData->Pitch;
					mfxU8 *srcBGR = ptr_src;
					int src_pitch = (w*3+3) & ~3;
					const char __declspec(align(16)) MASK_RGB3_TO_RGB4[] = { 0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1 };
					__m128i xMask = _mm_load_si128((__m128i*)MASK_RGB3_TO_RGB4);
					for (int y = h - CropUp - 1; y >= CropBottom; y--, dstR += pitch) {
						srcBGR = ptr_src + (src_pitch * y) + CropLeft * 3;
						mfxU8 *dst = dstR;
						int x = 0, x_fin = w - CropLeft - CropRight - 16;
						for ( ; x < x_fin; x += 16, dst += 64, srcBGR += 48) {
							__m128i x0 = _mm_loadu_si128((__m128i*)(srcBGR +  0));
							__m128i x1 = _mm_loadu_si128((__m128i*)(srcBGR + 16));
							__m128i x2 = _mm_loadu_si128((__m128i*)(srcBGR + 32));
							__m128i x3 = _mm_srli_si128(x2, 4);
							x3 = _mm_shuffle_epi8(x3, xMask);
							x2 = _mm_alignr_epi8(x2, x1, 8);
							x2 = _mm_shuffle_epi8(x2, xMask);
							x1 = _mm_alignr_epi8(x1, x0, 12);
							x1 = _mm_shuffle_epi8(x1, xMask);
							x0 = _mm_shuffle_epi8(x0, xMask);
							_mm_storeu_si128((__m128i*)(dst + 48), x3);
							_mm_storeu_si128((__m128i*)(dst + 32), x2);
							_mm_storeu_si128((__m128i*)(dst + 16), x1);
							_mm_storeu_si128((__m128i*)(dst +  0), x0);
						}
						x_fin = w - CropLeft - CropRight;
						for ( ; x < x_fin; x++, dst += 4, srcBGR += 3) {
							dst[0] = srcBGR[0];
							dst[1] = srcBGR[1];
							dst[2] = srcBGR[2];
							dst[3] = 0;
						}
					}
				}
				break;
			default:
				return MFX_ERR_UNSUPPORTED;
			}
			break;
		case MFX_FOURCC_RGB4:
			switch (FourCCRequired)
			{
			case MFX_FOURCC_RGB4:
				{
					mfxU8 *dstR = min( min(pData->R, pData->G), pData->B ) + pInfo->CropX + pInfo->CropY * pData->Pitch;
					//mfxU8 *srcBGR = ptr_src;
					int src_pitch = w * 4;
					int copy_width = (w - CropLeft - CropRight) << 2;

					for (int y = h - CropUp - 1; y >= CropBottom; y--, dstR += pitch) {
						sse_memcpy(dstR, ptr_src + (src_pitch * y) + (CropLeft<<2), copy_width);
						//srcBGR = ptr_src + (src_pitch * y);
						//for (int x = 0, x_fin = w - CropLeft - CropRight; x < x_fin; x++) {
						//	dstR[(x<<2)+0] = srcBGR[((x+CropLeft)<<2)+0];
						//	dstR[(x<<2)+1] = srcBGR[((x+CropLeft)<<2)+1];
						//	dstR[(x<<2)+2] = srcBGR[((x+CropLeft)<<2)+2];
						//	dstR[(x<<2)+3] = srcBGR[((x+CropLeft)<<2)+3];
						//}
					}
				}
				break;
			default:
				return MFX_ERR_UNSUPPORTED;
			}
			break;
		default:
			return MFX_ERR_UNSUPPORTED;
		}
#if ENABLE_AVI_DIRECT_MEMCPY
	}
#endif //#if ENABLE_AVI_DIRECT_MEMCPY

	m_pEncSatusInfo->m_nInputFrames++;
	// display update
	mfxU32 tm = timeGetTime();
	if (tm - m_tmLastUpdate > UPDATE_INTERVAL) {
		m_tmLastUpdate = tm;
		m_pEncSatusInfo->UpdateDisplay(tm, 0);
	}
	return MFX_ERR_NONE;
}

#endif //ENABLE_AVI_READER
