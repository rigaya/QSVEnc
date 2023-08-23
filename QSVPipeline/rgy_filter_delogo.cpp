// -----------------------------------------------------------------------------------------
// QSVEnc/VCEEnc/rkmppenc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2021 rigaya
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

#include <map>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include "rgy_ini.h"
#include "rgy_filesystem.h"
#include "rgy_codepage.h"
#include "rgy_filter_delogo.h"

RGY_ERR RGYFilterDelogo::delogoPlane(RGYFrameInfo* pOutputPlane, const ProcessDataDelogo *pDelego, const float fade, const bool target_y, RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent* event) {
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDelogo>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    RGYWorkSize local(32, 4);
    RGYWorkSize global(pDelego->width, pDelego->height);
    const char* kernel_name = (prm->delogo.mode == DELOGO_MODE_ADD) ? "kernel_logo_add" : "kernel_delogo";
    auto err = m_delogo.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        (cl_mem)pDelego->pDevLogo->frame.ptr[0], pDelego->pDevLogo->frame.pitch[0],
        pDelego->i_start, pDelego->j_start, pDelego->width, pDelego->height, (float)pDelego->depth * fade, target_y ? 1 : 0);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s (procPlane(%s)): %s.\n"),
            char_to_tstring(kernel_name).c_str(), RGY_CSP_NAMES[pOutputPlane->csp], get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDelogo::delogoFrame(RGYFrameInfo* pOutputFrame, const float fade, RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent* event) {
    const auto supportedCspYV12 = make_array<RGY_CSP>(RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
    for (int i = 0; i < RGY_CSP_PLANES[pOutputFrame->csp]; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        const ProcessDataDelogo *pDelego = &m_sProcessData[i];
        if (i > RGY_PLANE_Y) {
            if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), pOutputFrame->csp) != supportedCspYV12.end()) {
                pDelego = &m_sProcessData[i+1];
            }
        }
        const std::vector<RGYOpenCLEvent>& plane_wait_event = (i == 0) ? wait_events : std::vector<RGYOpenCLEvent>();
        RGYOpenCLEvent* plane_event = (i == RGY_CSP_PLANES[pOutputFrame->csp] - 1) ? event : nullptr;
        auto err = delogoPlane(&planeDst, pDelego, fade, (RGY_PLANE)i == RGY_PLANE_Y, queue, plane_wait_event, plane_event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to delogo frame(%d) %s: %s\n"), i, cl_errmes(err));
            return err_cl_to_rgy(err);
        }
    }
    return RGY_ERR_NONE;
}

RGYFilterDelogo::RGYFilterDelogo(shared_ptr<RGYOpenCLContext> context) : RGYFilter(context),
    m_LogoFilePath(),
    m_nLogoIdx(-1),
    m_sLogoDataList(),
    m_sProcessData(),
    m_delogo() {
    m_name = _T("delogo");
}

RGYFilterDelogo::~RGYFilterDelogo() {
    close();
}

int RGYFilterDelogo::readLogoFile(const std::shared_ptr<RGYFilterParamDelogo> pDelogoParam) {
    int sts = 0;
    if (pDelogoParam->delogo.logoFilePath.length() == 0) {
        return 1;
    }
    if (m_LogoFilePath == pDelogoParam->delogo.logoFilePath) {
        return -1;
    }
    auto file_deleter = [](FILE *fp) {
        fclose(fp);
    };
    AddMessage(RGY_LOG_DEBUG, _T("Opening logo file: %s\n"), pDelogoParam->delogo.logoFilePath.c_str());
    unique_ptr<FILE, decltype(file_deleter)> fp(_tfopen(pDelogoParam->delogo.logoFilePath.c_str(), _T("rb")), file_deleter);
    if (fp.get() == NULL) {
        AddMessage(RGY_LOG_ERROR, _T("could not open logo file \"%s\".\n"), pDelogoParam->delogo.logoFilePath.c_str());
        return 1;
    }
    // ファイルヘッダ取得
    int logo_header_ver = 0;
    LOGO_FILE_HEADER logo_file_header = { 0 };
    if (sizeof(logo_file_header) != fread(&logo_file_header, 1, sizeof(logo_file_header), fp.get())) {
        AddMessage(RGY_LOG_ERROR, _T("invalid logo file.\n"));
        sts = 1;
    } else if (0 == (logo_header_ver = get_logo_file_header_ver(&logo_file_header))) {
        AddMessage(RGY_LOG_ERROR, _T("invalid logo file.\n"));
        sts = 1;
    } else {
        AddMessage(RGY_LOG_DEBUG, _T("logo_header_ver: %d\n"), logo_header_ver);
        const size_t logo_header_size = (logo_header_ver == 2) ? sizeof(LOGO_HEADER) : sizeof(LOGO_HEADER_OLD);
        const int logonum = SWAP_ENDIAN(logo_file_header.logonum.l);
        AddMessage(RGY_LOG_DEBUG, _T("logonum: %d\n"), logonum);
        m_sLogoDataList.resize(logonum);

        for (int i = 0; i < logonum; i++) {
            if (logo_header_size != fread(&m_sLogoDataList[i].header, 1, logo_header_size, fp.get())) {
                AddMessage(RGY_LOG_ERROR, _T("invalid logo file.\n"));
                sts = 1;
                break;
            }
            if (logo_header_ver == 1) {
                convert_logo_header_v1_to_v2(&m_sLogoDataList[i].header);
            }

            const auto logoPixelBytes = logo_pixel_size(&m_sLogoDataList[i].header);

            // メモリ確保
            m_sLogoDataList[i].logoPixel.resize(logoPixelBytes / sizeof(m_sLogoDataList[i].logoPixel[0]), { 0 });

            if (logoPixelBytes != (int)fread(m_sLogoDataList[i].logoPixel.data(), 1, logoPixelBytes, fp.get())) {
                AddMessage(RGY_LOG_ERROR, _T("invalid logo file.\n"));
                sts = 1;
                break;
            }
        }
    }
    m_LogoFilePath = pDelogoParam->delogo.logoFilePath;
    return sts;
}

std::string RGYFilterDelogo::logoNameList() {
    std::string strlist;
    for (int i = 0; i < (int)m_sLogoDataList.size(); i++) {
        const std::string str = char_to_string(CP_THREAD_ACP, m_sLogoDataList[i].header.name, CODE_PAGE_SJIS);
        strlist += strsprintf("%3d: %s\n", i+1, str.c_str());
    }
    return strlist;
}

int RGYFilterDelogo::getLogoIdx(const std::string& logoName) {
    int idx = LOGO_AUTO_SELECT_INVALID;
    AddMessage(RGY_LOG_DEBUG, _T("getLogoIdx: \"%s\"\n"), char_to_tstring(logoName).c_str());
    for (int i = 0; i < (int)m_sLogoDataList.size(); i++) {
        const std::string str = char_to_string(CP_THREAD_ACP, m_sLogoDataList[i].header.name, CODE_PAGE_SJIS);
        AddMessage(RGY_LOG_DEBUG, _T("  name: %s\n"), char_to_tstring(str).c_str());
        if (str == logoName) {
            idx = i;
            break;
        }
    }
    return idx;
}

int RGYFilterDelogo::selectLogo(const tstring& selectStr, const tstring& inputFilename) {
    if (selectStr.length() == 0) {
        if (m_sLogoDataList.size() > 1) {
            AddMessage(RGY_LOG_ERROR, _T("--vpp-delogo-select option is required to select logo from logo pack.\n"));
            AddMessage(RGY_LOG_ERROR, char_to_tstring(logoNameList()));
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
    try {
        int j = std::stoi(selectStr);
        if (0 < j && j <= (int)m_sLogoDataList.size()) {
            return j-1;
        }
    } catch (...) {
        ;//後続の処理へ
    }

    //自動ロゴ選択ファイルか?
    std::string logoName = GetFullPathFrom(tchar_to_string(selectStr).c_str());
    if (!rgy_file_exists(selectStr.c_str())) {
        AddMessage(RGY_LOG_ERROR,
            _T("--vpp-delogo-select option has invalid param.\n")
            _T("Please set logo name or logo index (starting from 1),\n")
            _T("or auto select file.\n"));
        return LOGO_AUTO_SELECT_INVALID;
    }
    //自動選択キー
#if (defined(_WIN32) || defined(_WIN64))
    [[maybe_unused]] uint32_t codepage = CP_THREAD_ACP;
#else
    uint32_t codepage = CODE_PAGE_UNSET;
#endif
    int count = 0;
    for (;; count++) {
        char buf[512] = { 0 };
        GetPrivateProfileStringCP("LOGO_AUTO_SELECT", strsprintf("logo%d", count+1).c_str(), "", buf, sizeof(buf), logoName.c_str(), codepage);
        if (strlen(buf) == 0)
            break;
    }
    if (count == 0) {
        AddMessage(RGY_LOG_ERROR, _T("could not find any key to auto select from \"%s\".\n"), selectStr.c_str());
        return LOGO_AUTO_SELECT_INVALID;
    }
    std::vector<LOGO_SELECT_KEY> logoAutoSelectKeys;
    logoAutoSelectKeys.reserve(count);
    for (int i = 0; i < count; i++) {
        char buf[512] = { 0 };
        GetPrivateProfileStringCP("LOGO_AUTO_SELECT", strsprintf("logo%d", i+1).c_str(), "", buf, sizeof(buf), logoName.c_str(), codepage);
        char *ptr = strchr(buf, ',');
        if (ptr != NULL) {
            LOGO_SELECT_KEY selectKey;
            ptr[0] = '\0';
            selectKey.key = buf;
            strcpy_s(selectKey.logoname, ptr+1);
            logoAutoSelectKeys.push_back(std::move(selectKey));
        }
    }
    for (const auto& selectKey : logoAutoSelectKeys) {
        if (NULL != _tcsstr(inputFilename.c_str(), char_to_tstring(selectKey.key.c_str()).c_str())) {
            logoName = selectKey.logoname;
            return getLogoIdx(logoName);
        }
    }
    return LOGO_AUTO_SELECT_NOHIT;
}

RGY_ERR RGYFilterDelogo::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pLog) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pLog;
    auto pDelogoParam = std::dynamic_pointer_cast<RGYFilterParamDelogo>(pParam);
    if (!pDelogoParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    //delogoは常に元のフレームを書き換え
    if (!pDelogoParam->bOutOverwrite) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid param, delogo will overwrite input frame.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    pDelogoParam->frameOut = pDelogoParam->frameIn;

    //パラメータチェック
    int ret_logofile = readLogoFile(pDelogoParam);
    if (ret_logofile > 0) {
        return RGY_ERR_INVALID_PARAM;
    }
    const int logoidx = selectLogo(pDelogoParam->delogo.logoSelect, pDelogoParam->inputFileName);
    if (logoidx < 0) {
        if (logoidx == LOGO_AUTO_SELECT_NOHIT) {
            AddMessage(RGY_LOG_ERROR, _T("no logo was selected by auto select \"%s\".\n"), pDelogoParam->delogo.logoSelect.c_str());
            return RGY_ERR_INVALID_PARAM;
        } else {
            AddMessage(RGY_LOG_ERROR, _T("could not select logo by \"%s\".\n"), pDelogoParam->delogo.logoSelect.c_str());
            AddMessage(RGY_LOG_ERROR, char_to_tstring(logoNameList()));
            return RGY_ERR_INVALID_PARAM;
        }
    }
    if (pDelogoParam->frameOut.height <= 0 || pDelogoParam->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pDelogoParam->delogo.NRArea < 0 || 3 < pDelogoParam->delogo.NRArea) {
        AddMessage(RGY_LOG_ERROR, _T("nr_area must be in range of 0 - 3.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (pDelogoParam->delogo.NRValue < 0 || 4 < pDelogoParam->delogo.NRValue) {
        AddMessage(RGY_LOG_ERROR, _T("nr_value must be in range of 0 - 4.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (ret_logofile == 0 || m_nLogoIdx != logoidx) {
        m_nLogoIdx = logoidx;
        m_param = pDelogoParam;

        auto& logoData = m_sLogoDataList[m_nLogoIdx];
        if (pDelogoParam->delogo.posX || pDelogoParam->delogo.posY) {
            LogoData origData;
            origData.header = logoData.header;
            origData.logoPixel = logoData.logoPixel;

            logoData.logoPixel = std::vector<LOGO_PIXEL>((logoData.header.w + 1) * (logoData.header.h + 1), { 0 });

            create_adj_exdata(logoData.logoPixel.data(), &logoData.header, origData.logoPixel.data(), &origData.header, pDelogoParam->delogo.posX, pDelogoParam->delogo.posY);
        }
        const int frameWidth  = pDelogoParam->frameIn.width;
        const int frameHeight = pDelogoParam->frameIn.height;

        m_sProcessData[LOGO__Y].offset[0] = (short)pDelogoParam->delogo.Y  << 4;
        m_sProcessData[LOGO__Y].offset[1] = (short)pDelogoParam->delogo.Y  << 4;
        m_sProcessData[LOGO_UV].offset[0] = (short)pDelogoParam->delogo.Cb << 4;
        m_sProcessData[LOGO_UV].offset[1] = (short)pDelogoParam->delogo.Cr << 4;
        m_sProcessData[LOGO__U].offset[0] = (short)pDelogoParam->delogo.Cb << 4;
        m_sProcessData[LOGO__U].offset[1] = (short)pDelogoParam->delogo.Cb << 4;
        m_sProcessData[LOGO__V].offset[0] = (short)pDelogoParam->delogo.Cr << 4;
        m_sProcessData[LOGO__V].offset[1] = (short)pDelogoParam->delogo.Cr << 4;

        m_sProcessData[LOGO__Y].fade = 256;
        m_sProcessData[LOGO_UV].fade = 256;
        m_sProcessData[LOGO__U].fade = 256;
        m_sProcessData[LOGO__V].fade = 256;

        m_sProcessData[LOGO__Y].depth = pDelogoParam->delogo.depth;
        m_sProcessData[LOGO_UV].depth = pDelogoParam->delogo.depth;
        m_sProcessData[LOGO__U].depth = pDelogoParam->delogo.depth;
        m_sProcessData[LOGO__V].depth = pDelogoParam->delogo.depth;

        m_sProcessData[LOGO__Y].i_start = (std::min)(logoData.header.x & ~63, frameWidth);
        m_sProcessData[LOGO__Y].width   = (((std::min)(logoData.header.x + logoData.header.w, frameWidth) + 63) & ~63) - m_sProcessData[LOGO__Y].i_start;
        m_sProcessData[LOGO_UV].i_start = m_sProcessData[LOGO__Y].i_start;
        m_sProcessData[LOGO_UV].width   = m_sProcessData[LOGO__Y].width;
        m_sProcessData[LOGO__U].i_start = m_sProcessData[LOGO__Y].i_start >> 1;
        m_sProcessData[LOGO__U].width   = m_sProcessData[LOGO__Y].width >> 1;
        m_sProcessData[LOGO__V].i_start = m_sProcessData[LOGO__U].i_start;
        m_sProcessData[LOGO__V].width   = m_sProcessData[LOGO__U].width;
        const int yWidthOffset = logoData.header.x - m_sProcessData[LOGO__Y].i_start;

        m_sProcessData[LOGO__Y].j_start = (std::min)((int)logoData.header.y, frameHeight);
        m_sProcessData[LOGO__Y].height  = (std::min)(logoData.header.y + logoData.header.h, frameHeight) - m_sProcessData[LOGO__Y].j_start;
        m_sProcessData[LOGO_UV].j_start = logoData.header.y >> 1;
        m_sProcessData[LOGO_UV].height  = (((logoData.header.y + logoData.header.h + 1) & ~1) - (m_sProcessData[LOGO_UV].j_start << 1)) >> 1;
        m_sProcessData[LOGO__U].j_start = m_sProcessData[LOGO_UV].j_start;
        m_sProcessData[LOGO__U].height  = m_sProcessData[LOGO_UV].height;
        m_sProcessData[LOGO__V].j_start = m_sProcessData[LOGO__U].j_start;
        m_sProcessData[LOGO__V].height  = m_sProcessData[LOGO__U].height;

        if (logoData.header.x >= frameWidth || logoData.header.y >= frameHeight) {
            AddMessage(RGY_LOG_ERROR, _T("\"%s\" was not included in frame size %dx%d.\ndelogo disabled.\n"), pDelogoParam->delogo.logoSelect.c_str(), frameWidth, frameHeight);
            AddMessage(RGY_LOG_ERROR, _T("logo pos x=%d, y=%d, including pos offset value %d:%d.\n"), logoData.header.x, logoData.header.y, pDelogoParam->delogo.posX, pDelogoParam->delogo.posY);
            return RGY_ERR_INVALID_PARAM;
        }

        m_sProcessData[LOGO__Y].pLogoPtr.reset((int16_t *)_aligned_malloc(sizeof(int16_t) * 2 * m_sProcessData[LOGO__Y].width * m_sProcessData[LOGO__Y].height, 32));
        m_sProcessData[LOGO_UV].pLogoPtr.reset((int16_t *)_aligned_malloc(sizeof(int16_t) * 2 * m_sProcessData[LOGO_UV].width * m_sProcessData[LOGO_UV].height, 32));
        m_sProcessData[LOGO__U].pLogoPtr.reset((int16_t *)_aligned_malloc(sizeof(int16_t) * 2 * m_sProcessData[LOGO__U].width * m_sProcessData[LOGO__U].height, 32));
        m_sProcessData[LOGO__V].pLogoPtr.reset((int16_t *)_aligned_malloc(sizeof(int16_t) * 2 * m_sProcessData[LOGO__V].width * m_sProcessData[LOGO__V].height, 32));

        memset(m_sProcessData[LOGO__Y].pLogoPtr.get(), 0, sizeof(int16_t) * 2 * m_sProcessData[LOGO__Y].width * m_sProcessData[LOGO__Y].height);
        memset(m_sProcessData[LOGO_UV].pLogoPtr.get(), 0, sizeof(int16_t) * 2 * m_sProcessData[LOGO_UV].width * m_sProcessData[LOGO_UV].height);
        memset(m_sProcessData[LOGO__U].pLogoPtr.get(), 0, sizeof(int16_t) * 2 * m_sProcessData[LOGO__U].width * m_sProcessData[LOGO__U].height);
        memset(m_sProcessData[LOGO__V].pLogoPtr.get(), 0, sizeof(int16_t) * 2 * m_sProcessData[LOGO__V].width * m_sProcessData[LOGO__V].height);

        //まず輝度成分をコピーしてしまう
        for (int j = 0; j < m_sProcessData[LOGO__Y].height; j++) {
            //輝度成分はそのままコピーするだけ
            for (int i = 0; i < logoData.header.w; i++) {
                int16x2_t logoY = *(int16x2_t *)&logoData.logoPixel[j * logoData.header.w + i].dp_y;
                ((int16x2_t *)m_sProcessData[LOGO__Y].pLogoPtr.get())[j * m_sProcessData[LOGO__Y].width + i + yWidthOffset] = logoY;
            }
        }
        //まずは4:4:4->4:2:0処理時に端を気にしなくていいよう、縦横ともに2の倍数となるよう拡張する
        //CbCrの順番に並べていく
        //0で初期化しておく
        std::vector<int16x2_t> bufferCbCr444ForShrink(2 * m_sProcessData[LOGO_UV].height * 2 * m_sProcessData[LOGO__Y].width, { 0, 0 });
        int j_src = 0; //読み込み側の行
        int j_dst = 0; //書き込み側の行
        auto copyUVLineForShrink = [&]() {
            for (int i = 0; i < logoData.header.w; i++) {
                int16x2_t logoCb = *(int16x2_t *)&logoData.logoPixel[j_src * logoData.header.w + i].dp_cb;
                int16x2_t logoCr = *(int16x2_t *)&logoData.logoPixel[j_src * logoData.header.w + i].dp_cr;
                bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + i + yWidthOffset) * 2 + 0] = logoCb;
                bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + i + yWidthOffset) * 2 + 1] = logoCr;
            }
            if (yWidthOffset & 1) {
                //奇数列はじまりなら、それをその前の偶数列に拡張する
                int16x2_t logoCb = *(int16x2_t *)&bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + 0 + yWidthOffset) * 2 + 0];
                int16x2_t logoCr = *(int16x2_t *)&bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + 0 + yWidthOffset) * 2 + 1];
                bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + 0 + yWidthOffset - 1) * 2 + 0] = logoCb;
                bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + 0 + yWidthOffset - 1) * 2 + 1] = logoCr;
            }
            if ((yWidthOffset + logoData.header.w) & 1) {
                //偶数列おわりなら、それをその次の奇数列に拡張する
                int16x2_t logoCb = *(int16x2_t *)&bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + logoData.header.w + yWidthOffset) * 2 + 0];
                int16x2_t logoCr = *(int16x2_t *)&bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + logoData.header.w + yWidthOffset) * 2 + 1];
                bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + logoData.header.w + yWidthOffset + 1) * 2 + 0] = logoCb;
                bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + logoData.header.w + yWidthOffset + 1) * 2 + 1] = logoCr;
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
        for (int j = 0; j < m_sProcessData[LOGO__Y].height; j += 2) {
            for (int i = 0; i < m_sProcessData[LOGO_UV].width; i += 2) {
                int16x2_t logoCb0 = bufferCbCr444ForShrink[((j + 0) * m_sProcessData[LOGO_UV].width + i + 0) * 2 + 0];
                int16x2_t logoCr0 = bufferCbCr444ForShrink[((j + 0) * m_sProcessData[LOGO_UV].width + i + 0) * 2 + 1];
                int16x2_t logoCb1 = bufferCbCr444ForShrink[((j + 0) * m_sProcessData[LOGO_UV].width + i + 1) * 2 + 0];
                int16x2_t logoCr1 = bufferCbCr444ForShrink[((j + 0) * m_sProcessData[LOGO_UV].width + i + 1) * 2 + 1];
                int16x2_t logoCb2 = bufferCbCr444ForShrink[((j + 1) * m_sProcessData[LOGO_UV].width + i + 0) * 2 + 0];
                int16x2_t logoCr2 = bufferCbCr444ForShrink[((j + 1) * m_sProcessData[LOGO_UV].width + i + 0) * 2 + 1];
                int16x2_t logoCb3 = bufferCbCr444ForShrink[((j + 1) * m_sProcessData[LOGO_UV].width + i + 1) * 2 + 0];
                int16x2_t logoCr3 = bufferCbCr444ForShrink[((j + 1) * m_sProcessData[LOGO_UV].width + i + 1) * 2 + 1];

                int16x2_t logoCb, logoCr;
                logoCb.x = (logoCb0.x + logoCb1.x + logoCb2.x + logoCb3.x + 2) >> 2;
                logoCb.y = (logoCb0.y + logoCb1.y + logoCb2.y + logoCb3.y + 2) >> 2;
                logoCr.x = (logoCr0.x + logoCr1.x + logoCr2.x + logoCr3.x + 2) >> 2;
                logoCr.y = (logoCr0.y + logoCr1.y + logoCr2.y + logoCr3.y + 2) >> 2;

                //単純平均により4:4:4->4:2:0に
                ((int16x2_t *)m_sProcessData[LOGO_UV].pLogoPtr.get())[(j >> 1) * m_sProcessData[LOGO_UV].width * 1 + (i >> 1) * 2 + 0] = logoCb;
                ((int16x2_t *)m_sProcessData[LOGO_UV].pLogoPtr.get())[(j >> 1) * m_sProcessData[LOGO_UV].width * 1 + (i >> 1) * 2 + 1] = logoCr;
                ((int16x2_t *)m_sProcessData[LOGO__U].pLogoPtr.get())[(j >> 1) * m_sProcessData[LOGO__U].width * 1 + (i >> 1) * 1] = logoCb;
                ((int16x2_t *)m_sProcessData[LOGO__V].pLogoPtr.get())[(j >> 1) * m_sProcessData[LOGO__V].width * 1 + (i >> 1) * 1] = logoCr;
            }
        }

        for (uint32_t i = 0; i < _countof(m_sProcessData); i++) {
            m_sProcessData[i].pDevLogo = m_cl->createFrameBuffer(m_sProcessData[i].width * sizeof(int16x2_t), m_sProcessData[i].height, RGY_CSP_Y8, CL_MEM_READ_ONLY);
            size_t dst_origin[3] = { 0, 0, 0 };
            size_t src_origin[3] = { 0, 0, 0 };
            size_t region[3] = { m_sProcessData[i].width * sizeof(int16x2_t), (size_t)m_sProcessData[i].height, 1 };
            auto err = err_cl_to_rgy(clEnqueueWriteBufferRect(m_cl->queue().get(), (cl_mem)m_sProcessData[i].pDevLogo->frame.ptr[0], true, dst_origin, src_origin,
                region, m_sProcessData[i].pDevLogo->frame.pitch[0], 0, m_sProcessData[i].width * sizeof(int16x2_t), 0, m_sProcessData[i].pLogoPtr.get(), 0, nullptr, nullptr));
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("error at sending logo data %d clEnqueueWriteBufferRect: %s.\n"), i, get_err_mes(err));
                return err;
            }
        }

        const auto options = strsprintf("-D Type=%s -D bit_depth=%d",
            RGY_CSP_BIT_DEPTH[pDelogoParam->frameOut.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[pDelogoParam->frameOut.csp]);
        m_delogo.set(m_cl->buildResourceAsync(_T("RGY_FILTER_DELOGO_CL"), _T("EXE_DATA"), options.c_str()));

        auto logo_name = char_to_string(CP_THREAD_ACP, logoData.header.name, CODE_PAGE_SJIS);;
        setFilterInfo(_T("delgo:") + char_to_tstring(logo_name) + pDelogoParam->print());
#if 0
        if (pDelogoParam->delogo.log) {
            m_logPath = pDelogoParam->inputFileName + tstring(_T(".delogo_log.csv"));
            std::unique_ptr<FILE, decltype(&fclose)> fp(_tfopen(m_logPath.c_str(), _T("w")), fclose);
            _ftprintf(fp.get(), _T("%s\n\n"), m_sFilterInfo.c_str());
            _ftprintf(fp.get(), _T(", NR, fade (adj), fade (raw)\n"));
            fp.reset();
        }
#endif
    }
    return sts;
}

tstring RGYFilterParamDelogo::print() const {
    return delogo.print();
}

RGY_ERR RGYFilterDelogo::run_filter(const RGYFrameInfo* pInputFrame, RGYFrameInfo** ppOutputFrames, int* pOutputFrameNum, RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent* event) {
    RGY_ERR sts = RGY_ERR_NONE;

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] && ppOutputFrames[0]->ptr[0] && ppOutputFrames[0]->mem_type == RGY_MEM_TYPE_CPU) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto pDelogoParam = std::dynamic_pointer_cast<RGYFilterParamDelogo>(m_param);
    if (!pDelogoParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    if (!m_delogo.get()) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load RGY_FILTER_DELOGO_CL(m_delogo)\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    if (pInputFrame->ptr[0] == nullptr) {
        //自動フェードや自動NRを使用しない場合、入力フレームがないということはない
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return sts;
    }
    if (ppOutputFrames[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("ppOutputFrames[0] must be set.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    *pOutputFrameNum = 1;

    const float fade = (float)m_sProcessData[LOGO__Y].fade;
    if (RGY_ERR_NONE != (sts = delogoFrame(ppOutputFrames[0], fade, queue, wait_events, event))) {
        return sts;
    }
    return sts;
}

void RGYFilterDelogo::close() {
    m_LogoFilePath.clear();
    m_sLogoDataList.clear();
    m_delogo.clear();
    m_cl.reset();
    m_nLogoIdx = -1;
}
