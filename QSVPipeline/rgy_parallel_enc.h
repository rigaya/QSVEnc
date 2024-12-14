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

#pragma once
#ifndef __RGY_PARALLEL_ENC_H__
#define __RGY_PARALLEL_ENC_H__

#include <thread>
#include "rgy_osdep.h"
#include "rgy_err.h"
#include "rgy_pipe.h"

class RGYInput;
#if ENCODER_QSV
struct sInputParams;
#elif ENCODER_NVENC
#elif ENCODER_VCEENC
#elif ENCODER_RKMPP
#endif

class RGYParallelEncProcess {
public:
    RGYParallelEncProcess();
    ~RGYParallelEncProcess();
    RGY_ERR run(const tstring& cmd);
    RGY_ERR recvStdErr();
    int64_t getVideofirstKeyPts() const;
    RGY_ERR sendEndPts();
    RGY_ERR close();
    RGY_ERR getSample();
protected:
    std::unique_ptr<RGYPipeProcess> m_process;
    std::thread m_thRecvStderr;
};

class RGYParallelEnc {
public:
    RGYParallelEnc();
    virtual ~RGYParallelEnc();
    bool isParallelEncPossible(const RGYInput *input) const;
    RGY_ERR parallelRun(const sInputParams *prm, const RGYInput *input);
    void close();
protected:
    tstring genCmd(const sInputParams *prm);

    std::vector<std::unique_ptr<RGYParallelEncProcess>> m_encProcess;
};


#endif //__RGY_PARALLEL_ENC_H__
