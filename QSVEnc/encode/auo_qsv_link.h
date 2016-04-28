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

#include "qsv_input.h"
#include "auo_version.h"
#include "output.h"

class AUO_YUVReader : public CQSVInput
{
public :

    AUO_YUVReader();
    virtual ~AUO_YUVReader();

    virtual void Close() override;
    virtual mfxStatus Init(const TCHAR *strFileName, mfxU32 ColorFormat, const void *prm, CEncodingThread *pEncThread, shared_ptr<CEncodeStatusInfo> pEncSatusInfo, sInputCrop *pInputCrop) override;
    virtual mfxStatus LoadNextFrame(mfxFrameSurface1* pSurface) override;

private:
    mfxU32 current_frame;
};

typedef struct AuoStatusData {
    const OUTPUT_INFO *oip;
};

class AUO_EncodeStatusInfo : public CEncodeStatusInfo
{
public :
    AUO_EncodeStatusInfo();
    virtual ~AUO_EncodeStatusInfo();
    virtual void SetPrivData(void *pPrivateData);
private:
    virtual void UpdateDisplay(const char *mes, int drop_frames, double progressPercent) override;
    virtual mfxStatus UpdateDisplay(int drop_frames, double progressPercent = 0.0) override;
    virtual void WriteLine(const TCHAR *mes) override;

    AuoStatusData m_auoData;
    std::chrono::system_clock::time_point m_tmLastLogUpdate;
};

#endif //_AUO_YUVREADER_H_