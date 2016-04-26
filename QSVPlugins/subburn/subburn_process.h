//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

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
