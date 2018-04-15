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

#ifndef _AUO_YUVREADER_H_
#define _AUO_YUVREADER_H_
#if 0
#include "rgy_input.h"
#include "auo_version.h"
#include "auo.h"
#include "auo_conf.h"
#include "auo_system.h"
#include "output.h"

typedef struct InputInfoAuo {
    const OUTPUT_INFO *oip;
    const SYSTEM_DATA *sys_dat;
    CONF_GUIEX *conf;
    PRM_ENC *pe;
    int *jitter;
} InputInfoAuo;

class AUO_YUVReader : public RGYInput
{
private:
    const OUTPUT_INFO *oip;
    CONF_GUIEX *conf;
    PRM_ENC *pe;
    int *jitter;
    int m_iFrame;
    BOOL m_pause;
public :
    AUO_YUVReader();
    virtual ~AUO_YUVReader();

    virtual RGY_ERR Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const void *prm) override;
    virtual RGY_ERR LoadNextFrame(RGYFrame *pSurface) override;
    virtual void Close() override;
};

typedef struct AuoStatusData {
    const OUTPUT_INFO *oip;
};

class AUO_EncodeStatusInfo : public EncodeStatus
{
public :
    AUO_EncodeStatusInfo();
    virtual ~AUO_EncodeStatusInfo();
    virtual void SetPrivData(void *pPrivateData);
private:
    virtual void UpdateDisplay(const TCHAR *mes, double progressPercent = 0.0) override;
    virtual RGY_ERR UpdateDisplay(double progressPercent = 0.0) override;
    virtual void WriteLine(const TCHAR *mes) override;

    InputInfoAuo m_auoData;
    std::chrono::system_clock::time_point m_tmLastLogUpdate;
};
#endif
#endif //_AUO_YUVREADER_H_