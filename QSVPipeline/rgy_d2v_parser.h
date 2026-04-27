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

#include <cstdint>
#include <vector>
#include "rgy_tchar.h"
#include "rgy_util.h"

// Per-frame flag-byte decomposition from a DGIndex D2V project file.
// Values are in DISPLAY ORDER, one entry per coded picture. See DGIndex
// spec Table 5 for the bit layout we decode from:
//   bit 7 : decodable without previous GOP
//   bit 6 : progressive_frame (1 = progressive, 0 = interlaced)
//   bits 5-4 : picture_coding_type (01=I, 10=P, 11=B)
//   bits 3-2 : reserved
//   bit 1 : TFF (top-field-first)
//   bit 0 : RFF (repeat-first-field)
struct D2VFrameInfo {
    uint8_t progressive;   // 0 or 1
    uint8_t pictureType;   // 1=I, 2=P, 3=B, 0=reserved
    uint8_t tff;           // 0 or 1
    uint8_t rff;           // 0 or 1
};

// Minimal DGIndex D2V parser. Reads only the data we need for IVTC
// (per-frame progressive/TFF/RFF). Ignores the settings section entirely.
class RGYD2VParser {
public:
    RGYD2VParser();
    ~RGYD2VParser();

    // Parse the file. Returns true on success. Self-contained — safe to
    // call multiple times (clears previous state first).
    bool load(const tstring &path);

    // Count of frames successfully parsed.
    size_t frameCount() const { return m_frames.size(); }

    // Pointer to the frame info for index i, or nullptr if out of range.
    const D2VFrameInfo *frame(size_t i) const {
        return (i < m_frames.size()) ? &m_frames[i] : nullptr;
    }

    int progressiveCount() const { return m_progressive; }
    int rffCount()         const { return m_rff; }
    int interlacedCount()  const { return m_interlaced; }

    // One-line stats summary (suitable for logging).
    tstring stats() const;

    const tstring &path() const { return m_path; }

private:
    std::vector<D2VFrameInfo> m_frames;
    tstring m_path;
    int m_progressive;
    int m_rff;
    int m_interlaced;
};
