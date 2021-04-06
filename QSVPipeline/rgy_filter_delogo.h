// -----------------------------------------------------------------------------------------
// QSVEnc/VCEEnc by rigaya
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

#pragma once

#include "logo.h"
#include "rgy_filter.h"
#include "rgy_prm.h"

#define DELOGO_BLOCK_X  (32)
#define DELOGO_BLOCK_Y  (8)
#define DELOGO_BLOCK_LOOP_Y (4)

#define DELOGO_PARALLEL_FADE (33)
#define DELOGO_PRE_DIV_COUNT (4)
#define DELOGO_ADJMASK_DIV_COUNT (32)
#define DELOGO_ADJMASK_POW_BASE (1.1)

#define DELOGO_MASK_THRESHOLD_DEFAULT (1024)

#define LOGO_NR_MAX (4)
#define LOGO_FADE_AD_MAX	(10)
#define LOGO_FADE_AD_DEF	(7)

struct ProcessDataDelogo {
    unique_ptr<int16_t, aligned_malloc_deleter> pLogoPtr;
    unique_ptr<RGYCLFrame> pDevLogo;
    int    width;
    int    i_start;
    int    height;
    int    j_start;
    int    depth;
    short  offset[2];
    int    fade;

    ~ProcessDataDelogo() {
        pLogoPtr.reset();
        pDevLogo.reset();
    }
};

enum {
    LOGO_AUTO_SELECT_NOHIT   = -2,
    LOGO_AUTO_SELECT_INVALID = -1,
};

enum {
    LOGO__Y,
    LOGO_UV,
    LOGO__U,
    LOGO__V
};

typedef struct {
    int16_t x, y;
} int16x2_t;

struct LogoData {
    LOGO_HEADER header;
    vector<LOGO_PIXEL> logoPixel;

    LogoData() : header(), logoPixel() { memset(&header, 0, sizeof(header)); };
    ~LogoData() {};
};

typedef struct LOGO_SELECT_KEY {
    std::string key;
    char logoname[LOGO_MAX_NAME];
} LOGO_SELECT_KEY;

class RGYFilterParamDelogo : public RGYFilterParam {
public:
    const TCHAR *inputFileName; //入力ファイル名
    VppDelogo delogo;

    RGYFilterParamDelogo() : inputFileName(nullptr), delogo() {

    };
    virtual ~RGYFilterParamDelogo() {};
    virtual tstring print() const override;
};

class RGYFilterDelogo : public RGYFilter {
public:
    RGYFilterDelogo(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterDelogo();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const FrameInfo* pInputFrame, FrameInfo** ppOutputFrames, int* pOutputFrameNum, RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent* event) override;
    virtual void close() override;

    int readLogoFile(const std::shared_ptr<RGYFilterParamDelogo> pDelogoParam);
    int getLogoIdx(const std::string& logoName);
    int selectLogo(const tstring& selectStr, const tstring& inputFilename);
    std::string logoNameList();

    virtual RGY_ERR delogoPlane(FrameInfo* pOutputPlane, const ProcessDataDelogo *pDelego, const float fade, const bool target_y, RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent* event);
    virtual RGY_ERR delogoFrame(FrameInfo* pOutputPlane, const float fade, RGYOpenCLQueue& queue, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent* event);


    tstring m_LogoFilePath;
    int m_nLogoIdx;
    vector<LogoData> m_sLogoDataList;
    ProcessDataDelogo m_sProcessData[4];
    unique_ptr<RGYOpenCLProgram> m_delogo;
};
