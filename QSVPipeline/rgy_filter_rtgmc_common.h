// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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

#include <algorithm>

#include "rgy_filter_cl.h"

class RGYFrameDataRtgmcEdi : public RGYFrameData {
public:
    explicit RGYFrameDataRtgmcEdi(std::shared_ptr<RGYCLFrame> frame, const void *sourcePtr0 = nullptr) :
        m_frame(frame), m_sourcePtr0(sourcePtr0) {}
    virtual ~RGYFrameDataRtgmcEdi() {}

    const RGYFrameInfo *frame() const { return m_frame ? &m_frame->frame : nullptr; }
    RGYCLFrame *clFrame() const { return m_frame.get(); }
    std::shared_ptr<RGYCLFrame> frameRef() const { return m_frame; }
    const void *sourcePtr0() const { return m_sourcePtr0; }

protected:
    std::shared_ptr<RGYCLFrame> m_frame;
    // コピー元(EDI出力)フレームの先頭ptr。dataListの継承で無関係なフレームに
    // 添付が伝播しても、これとptr[0]の一致検証で「このフレームの内容同一コピー」
    // であることを確認できる (degrainゼロコピーキャッシュのアンカー判定用)。
    const void *m_sourcePtr0;
};

class RGYFrameDataRtgmcSourceTwin : public RGYFrameData {
public:
    RGYFrameDataRtgmcSourceTwin(std::shared_ptr<RGYCLFrame> frame, const void *sourcePtr0) :
        m_frame(frame), m_sourcePtr0(sourcePtr0) {}
    virtual ~RGYFrameDataRtgmcSourceTwin() {}

    const RGYFrameInfo *frame() const { return m_frame ? &m_frame->frame : nullptr; }
    std::shared_ptr<RGYCLFrame> frameRef() const { return m_frame; }
    const void *sourcePtr0() const { return m_sourcePtr0; }

protected:
    // search-prefilter出力と内容同一の入力キャッシュを保持する。
    // sourcePtr0との一致検証により、dataListが別フレームへ継承された場合は受理しない。
    std::shared_ptr<RGYCLFrame> m_frame;
    const void *m_sourcePtr0;
};

enum class RGYRtgmcCompDirection {
    Backward = 0,
    Forward = 1,
};

class RGYFrameDataRtgmcComp : public RGYFrameData {
public:
    RGYFrameDataRtgmcComp(std::shared_ptr<RGYCLFrame> frame, RGYRtgmcCompDirection direction, int delta) :
        m_frame(frame),
        m_direction(direction),
        m_delta(delta) {}
    virtual ~RGYFrameDataRtgmcComp() {}

    const RGYFrameInfo *frame() const { return m_frame ? &m_frame->frame : nullptr; }
    RGYCLFrame *clFrame() const { return m_frame.get(); }
    RGYRtgmcCompDirection direction() const { return m_direction; }
    int delta() const { return m_delta; }

protected:
    std::shared_ptr<RGYCLFrame> m_frame;
    RGYRtgmcCompDirection m_direction;
    int m_delta;
};

class RGYFrameDataRtgmcNoise : public RGYFrameData {
public:
    RGYFrameDataRtgmcNoise(std::shared_ptr<RGYCLFrame> frame, const RGYOpenCLEvent &event) :
        m_frame(frame),
        m_event(event) {}
    virtual ~RGYFrameDataRtgmcNoise() {}

    const RGYFrameInfo *frame() const { return m_frame ? &m_frame->frame : nullptr; }
    RGYCLFrame *clFrame() const { return m_frame.get(); }
    const RGYOpenCLEvent &event() const { return m_event; }

protected:
    std::shared_ptr<RGYCLFrame> m_frame;
    RGYOpenCLEvent m_event;
};

class RGYFrameDataRtgmcFrameRef : public RGYFrameData {
public:
    explicit RGYFrameDataRtgmcFrameRef(std::shared_ptr<RGYCLFrame> frame) :
        m_frame(frame) {}
    virtual ~RGYFrameDataRtgmcFrameRef() {}

    std::shared_ptr<RGYCLFrame> frameRef() const { return m_frame; }

protected:
    std::shared_ptr<RGYCLFrame> m_frame;
};

static std::shared_ptr<RGYCLFrame> rtgmcGetAttachedFrameRef(const RGYFrameInfo *frame) {
    if (!frame) {
        return nullptr;
    }
    for (const auto& data : frame->dataList) {
        auto ref = std::dynamic_pointer_cast<RGYFrameDataRtgmcFrameRef>(data);
        if (ref && ref->frameRef()) {
            return ref->frameRef();
        }
    }
    return nullptr;
}

static std::shared_ptr<RGYFrameDataRtgmcEdi> rtgmcGetAttachedEdi(const RGYFrameInfo *frame) {
    if (!frame) {
        return nullptr;
    }
    const auto frameData = std::find_if(frame->dataList.begin(), frame->dataList.end(), [](const std::shared_ptr<RGYFrameData> &data) {
        return std::dynamic_pointer_cast<RGYFrameDataRtgmcEdi>(data) != nullptr;
    });
    if (frameData == frame->dataList.end()) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<RGYFrameDataRtgmcEdi>(*frameData);
}

static std::shared_ptr<RGYFrameDataRtgmcSourceTwin> rtgmcGetAttachedSourceTwin(const RGYFrameInfo *frame) {
    if (!frame) {
        return nullptr;
    }
    const auto frameData = std::find_if(frame->dataList.begin(), frame->dataList.end(), [](const std::shared_ptr<RGYFrameData> &data) {
        return std::dynamic_pointer_cast<RGYFrameDataRtgmcSourceTwin>(data) != nullptr;
    });
    if (frameData == frame->dataList.end()) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<RGYFrameDataRtgmcSourceTwin>(*frameData);
}

static std::shared_ptr<RGYFrameDataRtgmcComp> rtgmcGetAttachedComp(const RGYFrameInfo *frame, RGYRtgmcCompDirection direction, int delta) {
    if (!frame) {
        return nullptr;
    }
    const auto frameData = std::find_if(frame->dataList.begin(), frame->dataList.end(), [direction, delta](const std::shared_ptr<RGYFrameData> &data) {
        auto comp = std::dynamic_pointer_cast<RGYFrameDataRtgmcComp>(data);
        return comp != nullptr && comp->direction() == direction && comp->delta() == delta;
    });
    if (frameData == frame->dataList.end()) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<RGYFrameDataRtgmcComp>(*frameData);
}

static std::shared_ptr<RGYFrameDataRtgmcNoise> rtgmcGetAttachedNoise(const RGYFrameInfo *frame) {
    if (!frame) {
        return nullptr;
    }
    const auto frameData = std::find_if(frame->dataList.begin(), frame->dataList.end(), [](const std::shared_ptr<RGYFrameData> &data) {
        return std::dynamic_pointer_cast<RGYFrameDataRtgmcNoise>(data) != nullptr;
    });
    if (frameData == frame->dataList.end()) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<RGYFrameDataRtgmcNoise>(*frameData);
}
