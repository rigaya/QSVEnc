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
// ------------------------------------------------------------------------------------------

#include "qsv_hw_device.h"
#include "rgy_util.h"

void CQSVHWDevice::AddMessage(RGYLogLevel log_level, const tstring &str) {
    if (m_pQSVLog == nullptr || log_level < m_pQSVLog->getLogLevel(RGY_LOGT_DEV)) {
        return;
    }
    auto lines = split(str, _T("\n"));
    for (const auto &line : lines) {
        if (line[0] != _T('\0')) {
            m_pQSVLog->write(log_level, RGY_LOGT_DEV, (m_name + _T(": ") + line + _T("\n")).c_str());
        }
    }
}
void CQSVHWDevice::AddMessage(RGYLogLevel log_level, const TCHAR *format, ...) {
    if (m_pQSVLog == nullptr || log_level < m_pQSVLog->getLogLevel(RGY_LOGT_DEV)) {
        return;
    }

    va_list args;
    va_start(args, format);
    int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    tstring buffer;
    buffer.resize(len, _T('\0'));
    _vstprintf_s(&buffer[0], len, format, args);
    va_end(args);
    AddMessage(log_level, buffer);
}
