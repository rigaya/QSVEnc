// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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

#ifndef __RGY_DUMMY_LOAD_H__
#define __RGY_DUMMY_LOAD_H__

#include "rgy_opencl.h"
#include "rgy_event.h"

class RGYDummyLoadCL {
public:
    RGYDummyLoadCL(std::shared_ptr<RGYOpenCLContext> cl);
    ~RGYDummyLoadCL();
    RGY_ERR run(const float targetLoadPercent, std::shared_ptr<RGYLog> log);
    void close();
protected:
    std::pair<RGY_ERR, double> runKernel(const int count, const int innerLoop, const float valA, const float valB);
    std::shared_ptr<RGYOpenCLContext> m_cl;
    RGYOpenCLQueue m_clQueue;
    RGYOpenCLProgramAsync m_prog;
    std::thread m_thread;
    unique_event m_event;
    std::shared_ptr<RGYLog> m_log;
    bool m_abort;
    int m_bufElemSize;
    std::unique_ptr<RGYCLBuf> m_clBuf;
};

#endif //__RGY_DUMMY_LOAD_H__
    