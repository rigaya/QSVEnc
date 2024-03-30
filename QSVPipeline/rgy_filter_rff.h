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

#pragma once

#include <array>
#include "rgy_filter_cl.h"
#include "rgy_prm.h"
#include "rgy_input.h"

static const int FRAME_BUF_SIZE = 2;

class RGYFilterParamRff : public RGYFilterParam {
public:
    VppRff rff;
    rgy_rational<int> inFps;
    rgy_rational<int> timebase;
    tstring outFilename;

    RGYFilterParamRff() : rff(), inFps(), timebase(), outFilename() {

    };
    virtual ~RGYFilterParamRff() {};
    virtual tstring print() const override;
};

class RGYFilterRff : public RGYFilter {
public:
    RGYFilterRff(std::shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterRff();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue& queue_main, const std::vector<RGYOpenCLEvent>& wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    RGY_ERR checkParam(const RGYFilterParam *param);
    int64_t getInputDuration(const RGYFrameInfo *pInputFrame);
    RGY_FRAME_FLAGS getPrevBufFlags() const;

    std::tuple<RGY_ERR, int, RGYFrameCopyMode> copyFieldFromBuffer(RGYFrameInfo *dst, const int idx, RGYOpenCLQueue& queue, RGYOpenCLEvent *event);
    RGY_ERR copyFieldToBuffer(const RGYFrameInfo *src, const RGYFrameCopyMode copyMode, RGYOpenCLQueue& queue, RGYOpenCLEvent *event);

    int m_nFieldBufUsed;
    std::array<RGY_FRAME_FLAGS, FRAME_BUF_SIZE> m_nFieldBufPicStruct;
    int64_t m_ptsOffset;
    int64_t m_prevInputTimestamp;
    RGY_FRAME_FLAGS m_prevInputFlags;
    RGY_PICSTRUCT m_prevInputPicStruct;
    std::unique_ptr<FILE, fp_deleter> m_fpLog;
};
