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
// --------------------------------------------------------------------------------------------

#pragma once
#ifndef __PLUGIN_DELOGO_H__
#define __PLUGIN_DELOGO_H__

#include <stdlib.h>
#include <vector>
#include <mfxplugin++.h>

#include "qsv_prm.h"
#include "../base/plugin_base.h"
#include "logo.h"

struct ProcessDataDelogo {
    unique_ptr<mfxI16, aligned_malloc_deleter> pLogoPtr;
    mfxU32 pitch;
    mfxU32 i_start;
    mfxU32 height;
    mfxU32 j_start;
    int    depth;
    short  nv12_2_yc48_mul;
    short  nv12_2_yc48_sub;
    short  yc48_2_nv12_mul;
    short  yc48_2_nv12_add;
    short  offset[2];
    int    fade;

    ProcessDataDelogo() : pLogoPtr(),
            pitch(0),
            i_start(0),
            height(0),
            j_start(0),
            depth(0),
            nv12_2_yc48_mul(0),
            nv12_2_yc48_sub(0),
            yc48_2_nv12_mul(0),
            yc48_2_nv12_add(0),
            offset(),
            fade(0) {};
    ~ProcessDataDelogo(){};
};

class ProcessorDelogo : public Processor
{
public:
    virtual mfxStatus Init(mfxFrameSurface1 *frame_in, mfxFrameSurface1 *frame_out, const void *data) override;

protected:
    const ProcessDataDelogo *m_sData[2];
};

struct DelogoParam {
    mfxFrameAllocator *pAllocator;    //メインパイプラインのアロケータ
    MemType            memType;       //アロケータのメモリタイプ
    const TCHAR       *inputFileName; //入力ファイル名
    const TCHAR       *logoFilePath;  //ロゴファイル名
    const TCHAR       *logoSelect;    //ロゴの名前
    short posX, posY; //位置オフセット
    short depth;      //透明度深度
    short Y, Cb, Cr;  //(輝度・色差)オフセット
    int   add;        //付加モード

    DelogoParam() :
        pAllocator(nullptr),
        memType(SYSTEM_MEMORY),
        inputFileName(nullptr),
        logoFilePath(nullptr),
        logoSelect(nullptr),
        posX(0), posY(0), depth(128),
        Y(0), Cb(0), Cr(0), add(0) {
    };

    DelogoParam(mfxFrameAllocator *pAllocator, MemType memType, 
        const TCHAR *logoFilePath, const TCHAR *logoSelect, const TCHAR *inputFileName,
        short posX = 0, short posY = 0, short depth = 128,
        short Y = 0, short Cb = 0, short Cr = 0, int add = 0) {
        this->pAllocator    = pAllocator;
        this->memType       = memType;
        this->logoFilePath  = logoFilePath;
        this->logoSelect    = logoSelect;
        this->inputFileName = inputFileName;
        this->posX  = posX;
        this->posY  = posY;
        this->depth = depth;
        this->Y     = Y;
        this->Cb    = Cb;
        this->Cr    = Cr;
        this->add   = add;
    };
};

enum {
    LOGO_AUTO_SELECT_NOHIT   = -2,
    LOGO_AUTO_SELECT_INVALID = -1,
};


struct LogoData {
    LOGO_HEADER header;
    vector<LOGO_PIXEL> logoPixel;

    LogoData() : header(), logoPixel() {};
    ~LogoData() {};
};

struct LOGO_SELECT_KEY {
    std::string key;
    char logoname[LOGO_MAX_NAME];

    LOGO_SELECT_KEY() : key(), logoname() {
        memset(logoname, 0, sizeof(logoname));
    }
    ~LOGO_SELECT_KEY(){};
};

class Delogo : public QSVEncPlugin
{
public:
    Delogo();
    virtual ~Delogo();

    // methods to be called by Media SDK
    virtual mfxStatus Init(mfxVideoParam *mfxParam);
    virtual mfxStatus SetAuxParams(void* auxParam, int auxParamSize);
    virtual mfxStatus Submit(const mfxHDL *in, mfxU32 in_num, const mfxHDL *out, mfxU32 out_num, mfxThreadTask *task);
    // methods to be called by application
    static MFXGenericPlugin* CreateGenericPlugin() {
        return new Delogo();
    }

    virtual mfxStatus Close();

protected:
    mfxStatus readLogoFile();
    int getLogoIdx(const std::string& logoName);
    int selectLogo(const TCHAR *selectStr);
    std::string logoNameList();
    DelogoParam m_DelogoParam;

    mfxU32 m_nSimdAvail;
    int m_nLogoIdx;
    vector<LogoData> m_sLogoDataList;
    ProcessDataDelogo m_sProcessData[2];
};

#endif // __PLUGIN_DELOGO_H__
