//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <cstdio>
#include <cstring>
#include "qsv_util.h"
#include "plugin_delogo.h"
#include "delogo_process.h"
#include "qsv_simd.h"
#include "qsv_osdep.h"
#include "qsv_ini.h"

// disable "unreferenced formal parameter" warning -
// not all formal parameters of interface functions will be used by sample plugin
#pragma warning(disable : 4100)

/* Delogo class implementation */
Delogo::Delogo() :
    m_nSimdAvail(0x00),
    m_nLogoIdx(-1) {
    memset(&m_DelogoParam, 0, sizeof(m_DelogoParam));
    memset(&m_sProcessData, 0, sizeof(m_sProcessData));
    m_nSimdAvail = get_availableSIMD();
    m_pluginName = _T("delogo");
}

Delogo::~Delogo() {
    PluginClose();
    Close();
}

mfxStatus Delogo::Submit(const mfxHDL *in, mfxU32 in_num, const mfxHDL *out, mfxU32 out_num, mfxThreadTask *task) {
    if (in == nullptr || out == nullptr || *in == nullptr || *out == nullptr || task == nullptr) {
        return MFX_ERR_NULL_PTR;
    }
    if (in_num != 1 || out_num != 1) {
        return MFX_ERR_UNSUPPORTED;
    }
    if (!m_bInited) return MFX_ERR_NOT_INITIALIZED;

    mfxFrameSurface1 *surface_in = (mfxFrameSurface1 *)in[0];
    mfxFrameSurface1 *surface_out = (mfxFrameSurface1 *)out[0];
    mfxFrameSurface1 *real_surface_in = surface_in;
    mfxFrameSurface1 *real_surface_out = surface_out;

    mfxStatus sts = MFX_ERR_NONE;

    if (m_bIsInOpaque) {
        sts = m_mfxCore.GetRealSurface(surface_in, &real_surface_in);
        if (sts < MFX_ERR_NONE) return sts;
    }

    if (m_bIsOutOpaque) {
        sts = m_mfxCore.GetRealSurface(surface_out, &real_surface_out);
        if (sts < MFX_ERR_NONE) return sts;
    }

    // check validity of parameters
    sts = CheckInOutFrameInfo(&real_surface_in->Info, &real_surface_out->Info);
    if (sts < MFX_ERR_NONE) return sts;

    mfxU32 ind = FindFreeTaskIdx();

    if (ind >= m_sTasks.size()) {
        return MFX_WRN_DEVICE_BUSY; // currently there are no free tasks available
    }

    m_mfxCore.IncreaseReference(&(real_surface_in->Data));
    m_mfxCore.IncreaseReference(&(real_surface_out->Data));

    m_sTasks[ind].In = real_surface_in;
    m_sTasks[ind].Out = real_surface_out;
    m_sTasks[ind].bBusy = true;

    if (m_sTasks[ind].pProcessor.get() == nullptr) {
        bool d3dSurface = !!(m_DelogoParam.memType & D3D9_MEMORY);
#if defined(_MSC_VER) || defined(__AVX2__)
        if ((m_nSimdAvail & (AVX2 | FMA3)) == (AVX2 | FMA3)) {
            m_sTasks[ind].pProcessor.reset((d3dSurface) ? static_cast<Processor *>(new DelogoProcessD3DAVX2) : new DelogoProcessAVX2);
        } else
#endif //#if defined(_MSC_VER) || defined(__AVX2__)
#if defined(_MSC_VER) || defined(__AVX__)
        if (m_nSimdAvail & AVX) {
            m_sTasks[ind].pProcessor.reset((d3dSurface) ? static_cast<Processor *>(new DelogoProcessD3DAVX) : new DelogoProcessAVX);
        } else
#endif //#ifdefined(_MSC_VER) || defined(__AVX__)
        if (m_nSimdAvail & SSE41) {
            m_sTasks[ind].pProcessor.reset((d3dSurface) ? static_cast<Processor *>(new DelogoProcessD3DSSE41) : new DelogoProcessSSE41);
        } else {
            m_message += _T("vpp-delogo requires SSE4.1 support.\n");
            return MFX_ERR_UNSUPPORTED;
        }

        //GPUによるD3DSurfaceのコピーを正常に実行するためには、m_pAllocは、
        //PluginのmfxCoreから取得したAllocatorではなく、
        //メインパイプラインから直接受け取ったAllocatorでなければならない
        m_sTasks[ind].pProcessor->SetAllocator((m_DelogoParam.pAllocator) ? m_DelogoParam.pAllocator : &m_mfxCore.FrameAllocator());
    }
    m_sTasks[ind].pProcessor->Init(real_surface_in, real_surface_out, m_sProcessData);

    *task = (mfxThreadTask)&m_sTasks[ind];

    return MFX_ERR_NONE;
}

std::string Delogo::logoNameList() {
    std::string strlist;
    for (int i = 0; i < (int)m_sLogoDataList.size(); i++) {
        strlist += strsprintf("%3d: %s\n", i+1, m_sLogoDataList[i].header.name);
    }
    return strlist;
}

mfxStatus Delogo::readLogoFile() {
    mfxStatus sts = MFX_ERR_NONE;
    
    if (m_DelogoParam.logoFilePath == NULL) {
        return MFX_ERR_NULL_PTR;
    }
    auto file_deleter = [](FILE *fp) {
        fclose(fp);
    };
    unique_ptr<FILE, decltype(file_deleter)> fp(_tfopen(m_DelogoParam.logoFilePath, _T("rb")), file_deleter);
    if (fp.get() == NULL) {
        m_message += strsprintf(_T("could not open logo file \"%s\".\n"), m_DelogoParam.logoFilePath);
        return MFX_ERR_NULL_PTR;
    }
    // ファイルヘッダ取得
    int logo_header_ver = 0;
    LOGO_FILE_HEADER logo_file_header = { 0 };
    if (sizeof(logo_file_header) != fread(&logo_file_header, 1, sizeof(logo_file_header), fp.get())) {
        m_message += _T("invalid logo file.\n");
        sts = MFX_ERR_UNSUPPORTED;
    } else if (0 == (logo_header_ver = get_logo_file_header_ver(&logo_file_header))) {
        m_message += _T("invalid logo file.\n");
        sts = MFX_ERR_UNSUPPORTED;
    } else {
        const size_t logo_header_size = (logo_header_ver == 2) ? sizeof(LOGO_HEADER) : sizeof(LOGO_HEADER_OLD);
        const int logonum = SWAP_ENDIAN(logo_file_header.logonum.l);
        m_sLogoDataList.resize(logonum);

        for (int i = 0; i < logonum; i++) {
            memset(&m_sLogoDataList[i], 0, sizeof(m_sLogoDataList[i]));
            if (logo_header_size != fread(&m_sLogoDataList[i].header, 1, logo_header_size, fp.get())) {
                m_message += _T("invalid logo file.\n");
                sts = MFX_ERR_UNSUPPORTED;
                break;
            }
            if (logo_header_ver == 1) {
                convert_logo_header_v1_to_v2(&m_sLogoDataList[i].header);
            }

            const mfxU32 logoPixelBytes = logo_pixel_size(&m_sLogoDataList[i].header);

            // メモリ確保
            m_sLogoDataList[i].logoPixel.resize(logoPixelBytes / sizeof(m_sLogoDataList[i].logoPixel[0]), { 0 });

            if (logoPixelBytes != (mfxU32)fread(m_sLogoDataList[i].logoPixel.data(), 1, logoPixelBytes, fp.get())) {
                m_message += _T("invalid logo file.\n");
                sts = MFX_ERR_UNSUPPORTED;
                break;
            }
        }
    }
    return sts;
}

int Delogo::getLogoIdx(const std::string& logoName) {
    int idx = LOGO_AUTO_SELECT_INVALID;
    for (int i = 0; i < (int)m_sLogoDataList.size(); i++) {
        if (0 == strcmp(m_sLogoDataList[i].header.name, logoName.c_str())) {
            idx = i;
            break;
        }
    }
    return idx;
}

int Delogo::selectLogo(const TCHAR *selectStr) {
    if (selectStr == nullptr) {
        if (m_sLogoDataList.size() > 1) {
            m_message += _T("--vpp-delogo-select option is required to select logo from logo pack.\n");
            m_message += char_to_tstring(logoNameList());
            return LOGO_AUTO_SELECT_INVALID;
        }
        return 0;
    }

    //ロゴ名として扱い、インデックスを取得
    {
        int idx = getLogoIdx(tchar_to_string(selectStr));
        if (idx != LOGO_AUTO_SELECT_INVALID) {
            return idx;
        }
    }
    //数字として扱い、インデックスを取得
    {
        TCHAR *eptr = nullptr;
        long j = _tcstol(selectStr, &eptr, 10);
        if (j != 0
            && (eptr == nullptr || eptr == selectStr + _tcslen(selectStr))
            && 0 < j && j <= (int)m_sLogoDataList.size())
            return j-1;
    }

    //自動ロゴ選択ファイルか?
    std::string logoName = GetFullPath(tchar_to_string(selectStr).c_str());
    if (!PathFileExists(selectStr)) {
        m_message += _T("")
            _T("--vpp-delogo-select option has invalid param.\n")
            _T("Please set logo name or logo index (starting from 1),\n")
            _T("or auto select file.\n");
        return LOGO_AUTO_SELECT_INVALID;
    }
    //自動選択キー
    int count = 0;
    for (;; count++) {
        char buf[512] = { 0 };
        GetPrivateProfileStringA("LOGO_AUTO_SELECT", strsprintf("logo%d", count+1).c_str(), "", buf, sizeof(buf), logoName.c_str());
        if (strlen(buf) == 0)
            break;
    }
    if (count == 0) {
        m_message += strsprintf(_T("could not find any key to auto select from \"%s\".\n"), selectStr);
        return LOGO_AUTO_SELECT_INVALID;
    }
    std::vector<LOGO_SELECT_KEY> logoAutoSelectKeys;
    logoAutoSelectKeys.reserve(count);
    for (int i = 0; i < count; i++) {
        LOGO_SELECT_KEY selectKey;
        char buf[512] = { 0 };
        GetPrivateProfileStringA("LOGO_AUTO_SELECT", strsprintf("logo%d", i+1).c_str(), "", buf, sizeof(buf), logoName.c_str());
        char *ptr = strchr(buf, ',');
        if (ptr != NULL) {
            ptr[0] = '\0';
            selectKey.key = buf;
            strcpy_s(selectKey.logoname, ptr+1);
            logoAutoSelectKeys.push_back(std::move(selectKey));
        }
    }
    for (const auto& selectKey : logoAutoSelectKeys) {
        if (NULL != _tcsstr(m_DelogoParam.inputFileName, char_to_tstring(selectKey.key.c_str()).c_str())) {
            logoName = selectKey.logoname;
            return getLogoIdx(logoName);
        }
    }
    return LOGO_AUTO_SELECT_NOHIT;
}

mfxStatus Delogo::Init(mfxVideoParam *mfxParam) {
    if (mfxParam == nullptr) {
        return MFX_ERR_NULL_PTR;
    }
    mfxStatus sts = MFX_ERR_NONE;
    m_VideoParam = *mfxParam;

    // map opaque surfaces array in case of opaque surfaces
    m_bIsInOpaque = (m_VideoParam.IOPattern & MFX_IOPATTERN_IN_OPAQUE_MEMORY) ? true : false;
    m_bIsOutOpaque = (m_VideoParam.IOPattern & MFX_IOPATTERN_OUT_OPAQUE_MEMORY) ? true : false;
    mfxExtOpaqueSurfaceAlloc* pluginOpaqueAlloc = NULL;

    if (m_bIsInOpaque || m_bIsOutOpaque) {
        pluginOpaqueAlloc = (mfxExtOpaqueSurfaceAlloc*)GetExtBuffer(m_VideoParam.ExtParam,
            m_VideoParam.NumExtParam, MFX_EXTBUFF_OPAQUE_SURFACE_ALLOCATION);
        if (sts != MFX_ERR_NONE) {
            m_message += _T("failed GetExtBuffer for OpaqueAlloc.\n");
            return sts;
        }
    }

    // check existence of corresponding allocs
    if ((m_bIsInOpaque && !pluginOpaqueAlloc->In.Surfaces) || (m_bIsOutOpaque && !pluginOpaqueAlloc->Out.Surfaces))
        return MFX_ERR_INVALID_VIDEO_PARAM;

    if (m_bIsInOpaque) {
        sts = m_mfxCore.MapOpaqueSurface(pluginOpaqueAlloc->In.NumSurface,
            pluginOpaqueAlloc->In.Type, pluginOpaqueAlloc->In.Surfaces);
        if (sts != MFX_ERR_NONE) {
            m_message += _T("failed MapOpaqueSurface[In].\n");
            return sts;
        }
    }

    if (m_bIsOutOpaque) {
        sts = m_mfxCore.MapOpaqueSurface(pluginOpaqueAlloc->Out.NumSurface,
            pluginOpaqueAlloc->Out.Type, pluginOpaqueAlloc->Out.Surfaces);
        if (sts != MFX_ERR_NONE) {
            m_message += _T("failed MapOpaqueSurface[Out].\n");
            return sts;
        }
    }

    m_sTasks.resize((std::max)(1, (int)m_VideoParam.AsyncDepth));
    m_sChunks.resize(m_PluginParam.MaxThreadNum);

    // divide frame into data chunks
    const mfxU32 num_lines_in_chunk = mfxParam->vpp.In.CropH / m_PluginParam.MaxThreadNum; // integer division
    const mfxU32 remainder_lines = mfxParam->vpp.In.CropH % m_PluginParam.MaxThreadNum; // get remainder
    // remaining lines are distributed among first chunks (+ extra 1 line each)
    for (mfxU32 i = 0; i < m_PluginParam.MaxThreadNum; i++) {
        m_sChunks[i].StartLine = (i == 0) ? 0 : m_sChunks[i-1].EndLine + 1;
        m_sChunks[i].EndLine = (i < remainder_lines) ? (i + 1) * num_lines_in_chunk : (i + 1) * num_lines_in_chunk - 1;
    }

    for (mfxU32 i = 0; i < m_sTasks.size(); i++) {
        m_sTasks[i].pBuffer.reset((mfxU8 *)_aligned_malloc(((std::max)(mfxParam->mfx.FrameInfo.Width, mfxParam->mfx.FrameInfo.CropW) + 255 + 16) & ~255, 32));
        if (m_sTasks[i].pBuffer.get() == nullptr) {
            m_message += _T("failed to allocate buffer.\n");
            return MFX_ERR_NULL_PTR;
        }
    }

    m_bInited = true;

    return MFX_ERR_NONE;
}

mfxStatus Delogo::SetAuxParams(void *auxParam, int auxParamSize) {
    DelogoParam *pDelogoPar = (DelogoParam *)auxParam;
    if (pDelogoPar == nullptr) {
        return MFX_ERR_NULL_PTR;
    }

    // check validity of parameters
    mfxStatus sts = CheckParam(&m_VideoParam);
    if (sts < MFX_ERR_NONE) return sts;

    memcpy(&m_DelogoParam, pDelogoPar, sizeof(m_DelogoParam));

    if (MFX_ERR_NONE != (sts = readLogoFile())) {
        return sts;
    }
    if (0 > (m_nLogoIdx = selectLogo(m_DelogoParam.logoSelect))) {
        if (m_nLogoIdx == LOGO_AUTO_SELECT_NOHIT) {
            m_message += strsprintf(_T("no logo was selected by auto select \"%s\".\n"), m_DelogoParam.logoSelect);
            return MFX_ERR_ABORTED;
        } else {
            m_message += strsprintf(_T("could not select logo by \"%s\".\n"), m_DelogoParam.logoSelect);
            m_message += char_to_tstring(logoNameList());
            return MFX_ERR_UNKNOWN;
        }
    }

    auto& logoData = m_sLogoDataList[m_nLogoIdx];
    if (m_DelogoParam.posX || m_DelogoParam.posY) {
        LogoData origData;
        origData.header = logoData.header;
        origData.logoPixel = logoData.logoPixel;

        logoData.logoPixel = std::vector<LOGO_PIXEL>((logoData.header.w + 1) * (logoData.header.h + 1), { 0 });

        create_adj_exdata(logoData.logoPixel.data(), &logoData.header, origData.logoPixel.data(), &origData.header, m_DelogoParam.posX, m_DelogoParam.posY);
    }

    const int frameWidth  = m_VideoParam.mfx.FrameInfo.CropW;
    const int frameHeight = m_VideoParam.mfx.FrameInfo.CropH;

    m_sProcessData[0].offset[0] = pDelogoPar->Y  << 4;
    m_sProcessData[0].offset[1] = pDelogoPar->Y  << 4;
    m_sProcessData[1].offset[0] = pDelogoPar->Cb << 4;
    m_sProcessData[1].offset[1] = pDelogoPar->Cr << 4;

    m_sProcessData[0].fade = 256;
    m_sProcessData[1].fade = 256;

    m_sProcessData[0].depth = pDelogoPar->depth;
    m_sProcessData[1].depth = pDelogoPar->depth;

    //nv12->YC48パラメータ
    m_sProcessData[0].nv12_2_yc48_mul = 19152;
    m_sProcessData[0].nv12_2_yc48_sub = 299;
    m_sProcessData[1].nv12_2_yc48_mul = 18725;
    m_sProcessData[1].nv12_2_yc48_sub = 2340;
    
    //YC48->nv12パラメータ
    m_sProcessData[0].yc48_2_nv12_mul = 3504;
    m_sProcessData[0].yc48_2_nv12_add = 301;
    m_sProcessData[1].yc48_2_nv12_mul = 3584;
    m_sProcessData[1].yc48_2_nv12_add = 2350;

    m_sProcessData[0].i_start = (std::min)(logoData.header.x & ~63, frameWidth);
    m_sProcessData[0].pitch   = (((std::min)(logoData.header.x + logoData.header.w, frameWidth) + 63) & ~63) - m_sProcessData[0].i_start;
    m_sProcessData[1].i_start = m_sProcessData[0].i_start;
    m_sProcessData[1].pitch   = m_sProcessData[0].pitch;
    const int yWidthOffset = logoData.header.x - m_sProcessData[0].i_start;

    m_sProcessData[0].j_start = (std::min)((int)logoData.header.y, frameHeight);
    m_sProcessData[0].height  = (std::min)(logoData.header.y + logoData.header.h, frameHeight) - m_sProcessData[0].j_start;
    m_sProcessData[1].j_start = logoData.header.y >> 1;
    m_sProcessData[1].height  = (((logoData.header.y + logoData.header.h + 1) & ~1) - (m_sProcessData[1].j_start << 1)) >> 1;

    if (logoData.header.x >= frameWidth || logoData.header.y >= frameHeight) {
        m_message += strsprintf(_T("\"%s\" was not included in frame size %dx%d.\ndelogo disabled.\n"), m_DelogoParam.logoSelect, frameWidth, frameHeight);
        m_message += strsprintf(_T("logo pos x=%d, y=%d, including pos offset value %d:%d.\n"), logoData.header.x, logoData.header.y, m_DelogoParam.posX, m_DelogoParam.posY);
        return MFX_ERR_ABORTED;
    }

    m_sProcessData[0].pLogoPtr.reset((mfxI16 *)_aligned_malloc(sizeof(mfxI16) * 2 * m_sProcessData[0].pitch * m_sProcessData[0].height, 32));
    m_sProcessData[1].pLogoPtr.reset((mfxI16 *)_aligned_malloc(sizeof(mfxI16) * 2 * m_sProcessData[1].pitch * m_sProcessData[1].height, 32));

    memset(m_sProcessData[0].pLogoPtr.get(), 0, sizeof(mfxI16) * 2 * m_sProcessData[0].pitch * m_sProcessData[0].height);
    memset(m_sProcessData[1].pLogoPtr.get(), 0, sizeof(mfxI16) * 2 * m_sProcessData[1].pitch * m_sProcessData[1].height);

    //まず輝度成分をコピーしてしまう
    for (mfxU32 j = 0; j < m_sProcessData[0].height; j++) {
        //輝度成分はそのままコピーするだけ
        for (int i = 0; i < logoData.header.w; i++) {
            mfxI16Pair logoY = *(mfxI16Pair *)&logoData.logoPixel[j * logoData.header.w + i].dp_y;
            *(mfxI16Pair *)&m_sProcessData[0].pLogoPtr.get()[(j * m_sProcessData[0].pitch + i + yWidthOffset) * 2] = logoY;
        }
    }
    //まずは4:4:4->4:2:0処理時に端を気にしなくていいよう、縦横ともに2の倍数となるよう拡張する
    //CbCrの順番に並べていく
    //0で初期化しておく
    std::vector<mfxI16Pair> bufferCbCr444ForShrink(2 * m_sProcessData[1].height * 2 * m_sProcessData[0].pitch, { 0, 0 });
    int j_src = 0; //読み込み側の行
    int j_dst = 0; //書き込み側の行
    auto copyUVLineForShrink = [&]() {
        for (int i = 0; i < logoData.header.w; i++) {
            mfxI16Pair logoCb = *(mfxI16Pair *)&logoData.logoPixel[j_src * logoData.header.w + i].dp_cb;
            mfxI16Pair logoCr = *(mfxI16Pair *)&logoData.logoPixel[j_src * logoData.header.w + i].dp_cr;
            bufferCbCr444ForShrink[(j_dst * m_sProcessData[1].pitch + i + yWidthOffset) * 2 + 0] = logoCb;
            bufferCbCr444ForShrink[(j_dst * m_sProcessData[1].pitch + i + yWidthOffset) * 2 + 1] = logoCr;
        }
        if (yWidthOffset & 1) {
            //奇数列はじまりなら、それをその前の偶数列に拡張する
            mfxI16Pair logoCb = *(mfxI16Pair *)&bufferCbCr444ForShrink[(j_dst * m_sProcessData[1].pitch + 0 + yWidthOffset) * 2 + 0];
            mfxI16Pair logoCr = *(mfxI16Pair *)&bufferCbCr444ForShrink[(j_dst * m_sProcessData[1].pitch + 0 + yWidthOffset) * 2 + 1];
            bufferCbCr444ForShrink[(j_dst * m_sProcessData[1].pitch + 0 + yWidthOffset - 1) * 2 + 0] = logoCb;
            bufferCbCr444ForShrink[(j_dst * m_sProcessData[1].pitch + 0 + yWidthOffset - 1) * 2 + 1] = logoCr;
        }
        if ((yWidthOffset + logoData.header.w) & 1) {
            //偶数列おわりなら、それをその次の奇数列に拡張する
            mfxI16Pair logoCb = *(mfxI16Pair *)&bufferCbCr444ForShrink[(j_dst * m_sProcessData[1].pitch + logoData.header.w + yWidthOffset) * 2 + 0];
            mfxI16Pair logoCr = *(mfxI16Pair *)&bufferCbCr444ForShrink[(j_dst * m_sProcessData[1].pitch + logoData.header.w + yWidthOffset) * 2 + 1];
            bufferCbCr444ForShrink[(j_dst * m_sProcessData[1].pitch + logoData.header.w + yWidthOffset + 1) * 2 + 0] = logoCb;
            bufferCbCr444ForShrink[(j_dst * m_sProcessData[1].pitch + logoData.header.w + yWidthOffset + 1) * 2 + 1] = logoCr;
        }
    };
    if (logoData.header.y & 1) {
        copyUVLineForShrink();
        j_dst++; //書き込み側は1行進める
    }
    for (; j_src < logoData.header.h; j_src++, j_dst++) {
        copyUVLineForShrink();
    }
    if ((logoData.header.y + logoData.header.h) & 1) {
        j_src--; //読み込み側は1行戻る
        copyUVLineForShrink();
    }

    //実際に縮小処理を行う
    //2x2->1x1の処理なのでインクリメントはそれぞれ2ずつ
    for (mfxU32 j = 0; j < m_sProcessData[0].height; j += 2) {
        for (mfxU32 i = 0; i < m_sProcessData[1].pitch; i += 2) {
            mfxI16Pair logoCb0 = bufferCbCr444ForShrink[((j + 0) * m_sProcessData[1].pitch + i + 0) * 2 + 0];
            mfxI16Pair logoCr0 = bufferCbCr444ForShrink[((j + 0) * m_sProcessData[1].pitch + i + 0) * 2 + 1];
            mfxI16Pair logoCb1 = bufferCbCr444ForShrink[((j + 0) * m_sProcessData[1].pitch + i + 1) * 2 + 0];
            mfxI16Pair logoCr1 = bufferCbCr444ForShrink[((j + 0) * m_sProcessData[1].pitch + i + 1) * 2 + 1];
            mfxI16Pair logoCb2 = bufferCbCr444ForShrink[((j + 1) * m_sProcessData[1].pitch + i + 0) * 2 + 0];
            mfxI16Pair logoCr2 = bufferCbCr444ForShrink[((j + 1) * m_sProcessData[1].pitch + i + 0) * 2 + 1];
            mfxI16Pair logoCb3 = bufferCbCr444ForShrink[((j + 1) * m_sProcessData[1].pitch + i + 1) * 2 + 0];
            mfxI16Pair logoCr3 = bufferCbCr444ForShrink[((j + 1) * m_sProcessData[1].pitch + i + 1) * 2 + 1];

            mfxI16Pair logoCb, logoCr;
            logoCb.x = (logoCb0.x + logoCb1.x + logoCb2.x + logoCb3.x + 2) >> 2;
            logoCb.y = (logoCb0.y + logoCb1.y + logoCb2.y + logoCb3.y + 2) >> 2;
            logoCr.x = (logoCr0.x + logoCr1.x + logoCr2.x + logoCr3.x + 2) >> 2;
            logoCr.y = (logoCr0.y + logoCr1.y + logoCr2.y + logoCr3.y + 2) >> 2;

            //単純平均により4:4:4->4:2:0に
            *(mfxI16Pair *)&m_sProcessData[1].pLogoPtr.get()[(j >> 1) * m_sProcessData[1].pitch * 2 + (i >> 1) * 4 + 0] = logoCb;
            *(mfxI16Pair *)&m_sProcessData[1].pLogoPtr.get()[(j >> 1) * m_sProcessData[1].pitch * 2 + (i >> 1) * 4 + 2] = logoCr;
        }
    }

    if ((m_nSimdAvail & (AVX2 | FMA3)) == (AVX2 | FMA3)) {
        m_pluginName = _T("delogo[AVX2]");
    } else if (m_nSimdAvail & AVX) {
        m_pluginName = _T("delogo[AVX]");
    } else if (m_nSimdAvail & SSE41) {
        m_pluginName = _T("delogo[SSE4.1]");
    } else {
        m_message += _T("requires SSE4.1 or higher.\n");
        return MFX_ERR_UNSUPPORTED;
    }

    std::string str = "";
    if (pDelogoPar->posX || pDelogoPar->posY) {
        str += strsprintf(", pos=%d:%d", pDelogoPar->posX, pDelogoPar->posY);
    }
    if (pDelogoPar->depth != QSV_DEFAULT_VPP_DELOGO_DEPTH) {
        str += strsprintf(", dpth=%d", pDelogoPar->depth);
    }
    if (pDelogoPar->Y || pDelogoPar->Cb || pDelogoPar->Cr) {
        str += strsprintf(", YCbCr=%d:%d:%d", pDelogoPar->Y, pDelogoPar->Cb, pDelogoPar->Cr);
    }
    m_message += char_to_tstring(logoData.header.name + str);
    return MFX_ERR_NONE;
}

mfxStatus Delogo::Close() {

    if (!m_bInited)
        return MFX_ERR_NONE;

    memset(&m_DelogoParam, 0, sizeof(m_DelogoParam));

    m_sTasks.clear();
    m_sChunks.clear();

    mfxStatus sts = MFX_ERR_NONE;

    mfxExtOpaqueSurfaceAlloc *pluginOpaqueAlloc = nullptr;

    if (m_bIsInOpaque || m_bIsOutOpaque) {
        if (nullptr == (pluginOpaqueAlloc = (mfxExtOpaqueSurfaceAlloc*)
            GetExtBuffer(m_VideoParam.ExtParam, m_VideoParam.NumExtParam, MFX_EXTBUFF_OPAQUE_SURFACE_ALLOCATION))) {
            return MFX_ERR_INVALID_VIDEO_PARAM;
        }
    }

    if ((m_bIsInOpaque && !pluginOpaqueAlloc->In.Surfaces) || (m_bIsOutOpaque && !pluginOpaqueAlloc->Out.Surfaces))
        return MFX_ERR_INVALID_VIDEO_PARAM;

    if (m_bIsInOpaque) {
        sts = m_mfxCore.UnmapOpaqueSurface(pluginOpaqueAlloc->In.NumSurface,
            pluginOpaqueAlloc->In.Type, pluginOpaqueAlloc->In.Surfaces);
        if (sts < MFX_ERR_NONE) return sts;
    }

    if (m_bIsOutOpaque) {
        sts = m_mfxCore.UnmapOpaqueSurface(pluginOpaqueAlloc->Out.NumSurface,
            pluginOpaqueAlloc->Out.Type, pluginOpaqueAlloc->Out.Surfaces);
        if (sts < MFX_ERR_NONE) return sts;
    }

    m_sLogoDataList.clear();
    memset(&m_sProcessData, 0, sizeof(m_sProcessData));
    m_nLogoIdx = LOGO_AUTO_SELECT_INVALID;
    
    m_message.clear();
    m_bInited = false;

    return MFX_ERR_NONE;
}

mfxStatus ProcessorDelogo::Init(mfxFrameSurface1 *frame_in, mfxFrameSurface1 *frame_out, const void *data) {
    if (frame_in == nullptr || frame_out == nullptr || data == nullptr) {
        return MFX_ERR_NULL_PTR;
    }

    const ProcessDataDelogo *param = (const ProcessDataDelogo *)data;

    m_pIn = frame_in;
    m_pOut = frame_out;
    m_sData[0] = &param[0];
    m_sData[1] = &param[1];

    return MFX_ERR_NONE;
}
