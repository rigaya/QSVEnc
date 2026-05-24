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
//
// H.264 spatial deblocking filter -- ITU-T Rec. H.264 §8.7 non-strong
// path. Distinct from the encoder's `--no-deblock` bitstream flag, which
// controls only the encoder's in-loop deblock on the output.

#pragma once
#ifndef __RGY_FILTER_DEBLOCK_H__
#define __RGY_FILTER_DEBLOCK_H__

#include "rgy_filter_cl.h"
#include "rgy_prm.h"

class RGYFilterParamDeblock : public RGYFilterParam {
public:
    VppDeblock deblock;

    RGYFilterParamDeblock() : deblock() {};
    virtual ~RGYFilterParamDeblock() {};
    virtual tstring print() const override { return deblock.print(); };
};

class RGYFilterDeblock : public RGYFilter {
public:
    RGYFilterDeblock(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterDeblock();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
                               RGYOpenCLQueue &queue_main, const std::vector<RGYOpenCLEvent> &wait_events,
                               RGYOpenCLEvent *event) override;
    virtual void close() override;
    virtual RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamDeblock> pParam);

    // Run pass 1 (vertical edges) then pass 2 (horizontal edges) on one
    // plane of the destination buffer. The host pre-copies the source
    // plane into dst before this call; the kernels work in place.
    RGY_ERR runPassVertical  (RGYFrameInfo *pDstPlane,
                              int alpha, int beta, int tc0, int is_chroma,
                              RGYOpenCLQueue &queue,
                              const std::vector<RGYOpenCLEvent> &wait_events,
                              RGYOpenCLEvent *event);
    RGY_ERR runPassHorizontal(RGYFrameInfo *pDstPlane,
                              int alpha, int beta, int tc0, int is_chroma,
                              RGYOpenCLQueue &queue,
                              const std::vector<RGYOpenCLEvent> &wait_events,
                              RGYOpenCLEvent *event);

    RGYOpenCLProgramAsync m_deblock;
    std::string           m_buildOptions;
};

#endif // __RGY_FILTER_DEBLOCK_H__
