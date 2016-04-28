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

#ifndef __SUB_BURN_PROCESS_H__
#define __SUB_BURN_PROCESS_H__

#include "plugin_subburn.h"

#if ENABLE_AVCODEC_QSV_READER && ENABLE_LIBASS_SUBBURN

class ProcessorSubBurnSSE41 : public ProcessorSubBurn
{
public:
    ProcessorSubBurnSSE41();
    virtual ~ProcessorSubBurnSSE41();

#if ENABLE_AVCODEC_QSV_READER && ENABLE_LIBASS_SUBBURN
    virtual void CopyFrameY() override;
    virtual void CopyFrameUV() override;
    virtual void BlendSubY(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcolory, uint8_t subTransparency, uint8_t *pBuf) override;
    virtual void BlendSubUV(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcoloru, uint8_t subcolorv, uint8_t subTransparency, uint8_t *pBuf) override;
#endif
};

class ProcessorSubBurnSSE41PshufbSlow : public ProcessorSubBurn
{
public:
    ProcessorSubBurnSSE41PshufbSlow();
    virtual ~ProcessorSubBurnSSE41PshufbSlow();

#if ENABLE_AVCODEC_QSV_READER && ENABLE_LIBASS_SUBBURN
    virtual void CopyFrameY() override;
    virtual void CopyFrameUV() override;
    virtual void BlendSubY(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcolory, uint8_t subTransparency, uint8_t *pBuf) override;
    virtual void BlendSubUV(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcoloru, uint8_t subcolorv, uint8_t subTransparency, uint8_t *pBuf) override;
#endif
};

class ProcessorSubBurnAVX : public ProcessorSubBurn
{
public:
    ProcessorSubBurnAVX();
    virtual ~ProcessorSubBurnAVX();

#if ENABLE_AVCODEC_QSV_READER && ENABLE_LIBASS_SUBBURN
    virtual void CopyFrameY() override;
    virtual void CopyFrameUV() override;
    virtual void BlendSubY(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcolory, uint8_t subTransparency, uint8_t *pBuf) override;
    virtual void BlendSubUV(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcoloru, uint8_t subcolorv, uint8_t subTransparency, uint8_t *pBuf) override;
#endif
};

class ProcessorSubBurnAVX2 : public ProcessorSubBurn
{
public:
    ProcessorSubBurnAVX2();
    virtual ~ProcessorSubBurnAVX2();

#if ENABLE_AVCODEC_QSV_READER && ENABLE_LIBASS_SUBBURN
    virtual void CopyFrameY() override;
    virtual void CopyFrameUV() override;
    virtual void BlendSubY(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcolory, uint8_t subTransparency, uint8_t *pBuf) override;
    virtual void BlendSubUV(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcoloru, uint8_t subcolorv, uint8_t subTransparency, uint8_t *pBuf) override;
#endif
};

class ProcessorSubBurnD3DSSE41 : public ProcessorSubBurn
{
public:
    ProcessorSubBurnD3DSSE41();
    virtual ~ProcessorSubBurnD3DSSE41();

#if ENABLE_AVCODEC_QSV_READER && ENABLE_LIBASS_SUBBURN
    virtual void CopyFrameY() override;
    virtual void CopyFrameUV() override;
    virtual void BlendSubY(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcolory, uint8_t subTransparency, uint8_t *pBuf) override;
    virtual void BlendSubUV(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcoloru, uint8_t subcolorv, uint8_t subTransparency, uint8_t *pBuf) override;
#endif
};

class ProcessorSubBurnD3DSSE41PshufbSlow : public ProcessorSubBurn
{
public:
    ProcessorSubBurnD3DSSE41PshufbSlow();
    virtual ~ProcessorSubBurnD3DSSE41PshufbSlow();

#if ENABLE_AVCODEC_QSV_READER && ENABLE_LIBASS_SUBBURN
    virtual void CopyFrameY() override;
    virtual void CopyFrameUV() override;
    virtual void BlendSubY(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcolory, uint8_t subTransparency, uint8_t *pBuf) override;
    virtual void BlendSubUV(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcoloru, uint8_t subcolorv, uint8_t subTransparency, uint8_t *pBuf) override;
#endif
};

class ProcessorSubBurnD3DAVX : public ProcessorSubBurn
{
public:
    ProcessorSubBurnD3DAVX();
    virtual ~ProcessorSubBurnD3DAVX();

#if ENABLE_AVCODEC_QSV_READER && ENABLE_LIBASS_SUBBURN
    virtual void CopyFrameY() override;
    virtual void CopyFrameUV() override;
    virtual void BlendSubY(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcolory, uint8_t subTransparency, uint8_t *pBuf) override;
    virtual void BlendSubUV(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcoloru, uint8_t subcolorv, uint8_t subTransparency, uint8_t *pBuf) override;
#endif
};

class ProcessorSubBurnD3DAVX2 : public ProcessorSubBurn
{
public:
    ProcessorSubBurnD3DAVX2();
    virtual ~ProcessorSubBurnD3DAVX2();

#if ENABLE_AVCODEC_QSV_READER && ENABLE_LIBASS_SUBBURN
    virtual void CopyFrameY() override;
    virtual void CopyFrameUV() override;
    virtual void BlendSubY(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcolory, uint8_t subTransparency, uint8_t *pBuf) override;
    virtual void BlendSubUV(const uint8_t *pAlpha, int bufX, int bufY, int bufW, int bufStride, int bufH, uint8_t subcoloru, uint8_t subcolorv, uint8_t subTransparency, uint8_t *pBuf) override;
#endif
};

#endif //#if ENABLE_AVCODEC_QSV_READER && ENABLE_LIBASS_SUBBURN

#endif // __SUB_BURN_PROCESS_H__
