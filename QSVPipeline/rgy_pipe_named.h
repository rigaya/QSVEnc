// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
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
#ifndef __RGY_PIPE_NAMED_H__
#define __RGY_PIPE_NAMED_H__

#include <cstdint>
#include <cstdio>
#include <vector>
#include "rgy_osdep.h"
#include "rgy_event.h"
#include "rgy_util.h"

class RGYNamedPipe {
public:
    RGYNamedPipe();
    virtual ~RGYNamedPipe();
    int init(const tstring& pipeName, const bool overlapped);
    int connect(DWORD timeout);
    int write(const void *data, const size_t size, size_t *writeSize);
    int read(void *data, const size_t size, size_t *readSize);
    int disconnect();
    int close();
    HANDLE handle() const { return m_handle; }
    bool connected() const { return m_connected; }
    const tstring& name() const { return m_name; }
    FILE *fp(const char *mode);
protected:
    HANDLE m_handle;
    FILE *m_fp;
    unique_event m_event;
    tstring m_name;
    bool m_connected;
    bool m_overlapped;
};

#endif //__RGY_PIPE_NAMED_H__
