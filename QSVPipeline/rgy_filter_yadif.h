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

#include "rgy_filter.h"
#include "rgy_prm.h"
#include <array>

enum YadifTargetField {
    YADIF_GEN_FIELD_UNKNOWN = -1,
    YADIF_GEN_FIELD_TOP = 0,
    YADIF_GEN_FIELD_BOTTOM
};

class RGYFilterParamYadif : public RGYFilterParam {
public:
    VppYadif yadif;
    rgy_rational<int> timebase;
    RGYFilterParamYadif() : yadif(), timebase() {};
    virtual ~RGYFilterParamYadif() {};
    virtual tstring print() const override { return yadif.print(); };
};

class RGYFilterYadifSource {
public:
    RGYFilterYadifSource(std::shared_ptr<RGYOpenCLContext> cl);
    ~RGYFilterYadifSource();
    RGY_ERR add(const RGYFrameInfo *pInputFrame, RGYOpenCLQueue &queue);
    RGY_ERR alloc(const RGYFrameInfo& frameInfo);
    void clear();
    RGYCLFrame *get(int iframe) {
        iframe = clamp(iframe, 0, m_nFramesInput - 1);
        return m_buf[iframe % m_buf.size()].get();
    }
    int inframe() const { return m_nFramesInput; }
private:
    std::shared_ptr<RGYOpenCLContext> m_cl;
    int m_nFramesInput;
    int m_nFramesOutput;
    std::array<std::unique_ptr<RGYCLFrame>, 4> m_buf;
};

class RGYFilterYadif : public RGYFilter {
public:
    RGYFilterYadif(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterYadif();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    RGY_ERR checkParam(const std::shared_ptr<RGYFilterParamYadif> prm);

    virtual RGY_ERR procPlane(
        RGYFrameInfo *pOutputPlane,
        const RGYFrameInfo *pInputPlane0,
        const RGYFrameInfo *pInputPlane1,
        const RGYFrameInfo *pInputPlane2,
        const YadifTargetField targetField,
        const RGY_PICSTRUCT picstruct,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);
    virtual RGY_ERR procFrame(
        RGYFrameInfo *pOutputFrame,
        const RGYFrameInfo *pInputFrame0,
        const RGYFrameInfo *pInputFrame1,
        const RGYFrameInfo *pInputFrame2,
        const YadifTargetField targetField,
        const RGY_PICSTRUCT picstruct,
        RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event);

    RGYOpenCLProgramAsync m_yadif;
    int m_nFrame;
    int64_t m_pts;
    RGYFilterYadifSource m_source;
};
