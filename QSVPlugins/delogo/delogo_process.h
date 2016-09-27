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

class DelogoProcessAddSSE41 : public ProcessorDelogo {
public:
    DelogoProcessAddSSE41();
    virtual ~DelogoProcessAddSSE41();

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

class DelogoProcessAddD3DSSE41 : public ProcessorDelogo {
public:
    DelogoProcessAddD3DSSE41();
    virtual ~DelogoProcessAddD3DSSE41();

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
