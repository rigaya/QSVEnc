//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#pragma once
#ifndef __PLUGIN_DELOGO_H__
#define __PLUGIN_DELOGO_H__

#include <stdlib.h>
#include <vector>
#include <mfxplugin++.h>

#include "qsv_prm.h"
#include "../base/plugin_base.h"
#include "logo.h"

typedef struct {
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
} ProcessDataDelogo;

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

    DelogoParam() :
        pAllocator(nullptr),
        memType(SYSTEM_MEMORY),
        inputFileName(nullptr),
        logoFilePath(nullptr),
        logoSelect(nullptr),
        posX(0), posY(0), depth(128),
        Y(0), Cb(0), Cr(0) {
    };

    DelogoParam(mfxFrameAllocator *pAllocator, MemType memType, 
        const TCHAR *logoFilePath, const TCHAR *logoSelect, const TCHAR *inputFileName,
        short posX = 0, short posY = 0, short depth = 128,
        short Y = 0, short Cb = 0, short Cr = 0) {
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
    };
};

enum {
    LOGO_AUTO_SELECT_NOHIT   = -2,
    LOGO_AUTO_SELECT_INVALID = -1,
};


typedef struct LogoData {
    LOGO_HEADER header;
    vector<LOGO_PIXEL> logoPixel;
} LogoData;

typedef struct LOGO_SELECT_KEY {
    std::string key;
    char logoname[LOGO_MAX_NAME];
} LOGO_SELECT_KEY;

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
