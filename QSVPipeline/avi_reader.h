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
// ------------------------------------------------------------------------------------------
#ifndef _AVI_READER_H_
#define _AVI_READER_H_

#include "rgy_version.h"
#if ENABLE_AVI_READER
#include <Windows.h>
#include <vfw.h>
#pragma comment(lib, "vfw32.lib")
#include "qsv_input.h"

class CAVIReader : public CQSVInput
{
public:
    CAVIReader();
    virtual ~CAVIReader();

    virtual RGY_ERR Init(const TCHAR *strFileName, VideoInfo *pInputInfo, const void *prm, shared_ptr<EncodeStatus> pEncSatusInfo) override;
    virtual RGY_ERR LoadNextFrame(RGYFrame *pSurface) override;
    virtual void Close() override;

    PAVIFILE m_pAviFile;
    PAVISTREAM m_pAviStream;
    PGETFRAME m_pGetFrame;
    LPBITMAPINFOHEADER m_pBitmapInfoHeader;
    int m_nYPitchMultiplizer;

    uint32_t m_nBufSize;
    shared_ptr<uint8_t> m_pBuffer;
};

#endif //ENABLE_AVI_READER

#endif //_AVI_READER_H_
