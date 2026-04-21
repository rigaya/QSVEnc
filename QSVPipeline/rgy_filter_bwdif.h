// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
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

#pragma once
#include "rgy_filter_cl.h"
#include "rgy_prm.h"

class RGYFilterParamBwdif : public RGYFilterParam {
public:
    VppBwdif bwdif;
    rgy_rational<int> timebase;

    RGYFilterParamBwdif() : bwdif(), timebase() {};
    virtual ~RGYFilterParamBwdif() {};
    virtual tstring print() const override { return bwdif.print(); };
};

class RGYFilterBwdif : public RGYFilter {
public:
    RGYFilterBwdif(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterBwdif();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
                               RGYOpenCLQueue &queue_main, const std::vector<RGYOpenCLEvent> &wait_events,
                               RGYOpenCLEvent *event) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamBwdif> pParam);

    RGY_ERR reconstructFrame(int idx_prev, int idx_cur, int idx_next,
                             bool inputTff, int preserveTopField, int outputSlot,
                             RGYOpenCLQueue &queue,
                             const std::vector<RGYOpenCLEvent> &wait_events);
    RGY_ERR generateOutput(int idx_prev, int idx_cur, int idx_next,
                           RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
                           RGYOpenCLQueue &queue,
                           const std::vector<RGYOpenCLEvent> &wait_events);
    void setBobTimestamp(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames);
    bool getInputTff(const RGYFrameInfo *frame) const;
    bool shouldPassthrough(const RGYFrameInfo *frame) const;

    RGYOpenCLProgramAsync m_bwdif;
    std::string m_bwdifBuildOptions;
    std::vector<std::unique_ptr<RGYCLFrame>> m_cacheFrames;
    int  m_inputCount;
    bool m_drained;
    bool m_defaultTff;
};
