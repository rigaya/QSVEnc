//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef __DELOGO_PROCESS_H__
#define __DELOGO_PROCESS_H__

#include "plugin_delogo.h"

class DelogoProcessSSE41 : public ProcessorDelogo
{
public:
    DelogoProcessSSE41();
    virtual ~DelogoProcessSSE41();

    virtual mfxStatus Process(DataChunk *chunk, mfxU8 *pBuffer) override;
};

class DelogoProcessAVX : public ProcessorDelogo
{
public:
    DelogoProcessAVX();
    virtual ~DelogoProcessAVX();

    virtual mfxStatus Process(DataChunk *chunk, mfxU8 *pBuffer) override;
};

class DelogoProcessAVX2 : public ProcessorDelogo
{
public:
    DelogoProcessAVX2();
    virtual ~DelogoProcessAVX2();

    virtual mfxStatus Process(DataChunk *chunk, mfxU8 *pBuffer) override;
};

class DelogoProcessD3DSSE41 : public ProcessorDelogo
{
public:
    DelogoProcessD3DSSE41();
    virtual ~DelogoProcessD3DSSE41();

    virtual mfxStatus Process(DataChunk *chunk, mfxU8 *pBuffer) override;
};

class DelogoProcessD3DAVX : public ProcessorDelogo
{
public:
    DelogoProcessD3DAVX();
    virtual ~DelogoProcessD3DAVX();

    virtual mfxStatus Process(DataChunk *chunk, mfxU8 *pBuffer) override;
};

class DelogoProcessD3DAVX2 : public ProcessorDelogo
{
public:
    DelogoProcessD3DAVX2();
    virtual ~DelogoProcessD3DAVX2();

    virtual mfxStatus Process(DataChunk *chunk, mfxU8 *pBuffer) override;
};

#endif // __DELOGO_PROCESS_H__
