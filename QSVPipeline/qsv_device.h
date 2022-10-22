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

#ifndef _QSV_DEVICE_H_
#define _QSV_DEVICE_H_

#include "rgy_version.h"
#include "qsv_util.h"
#include "qsv_session.h"
#include "qsv_query.h"

class QSVDevice {
public:
    QSVDevice();
    virtual ~QSVDevice();

    RGY_ERR init(const QSVDeviceNum dev, const bool enableOpenCL, const bool suppressErrorMessage);
    RGY_ERR init(const QSVDeviceNum dev, const bool enableOpenCL, MemType memType, std::shared_ptr<RGYLog> m_log, const bool suppressErrorMessage);

    CodecCsp getDecodeCodecCsp(const bool skipHWDecodeCheck);
    uint64_t getEncodeFeature(const int ratecontrol, const RGY_CODEC codec, const bool lowpower);

    void close();

    tstring name() const;
    LUID luid();
    QSV_CPU_GEN CPUGen();

    QSVDeviceNum deviceNum() const { return m_devNum; };
    MemType memType() const { return m_memType; };
    CQSVHWDevice *hwdev() { return m_hwdev.get(); }
    QSVAllocator *allocator() { return m_allocator.get(); }
    bool externalAlloc() const { return m_externalAlloc; }
    const RGYOpenCLDeviceInfo *devInfo() const { return m_devInfo.get(); }
    MFXVideoSession2& mfxSession() { return m_session; };
protected:

    void PrintMes(RGYLogLevel log_level, const TCHAR *format, ...) {
        if (m_log.get() == nullptr) {
            if (log_level <= RGY_LOG_INFO) {
                return;
            }
        }
        else if (log_level < m_log->getLogLevel(RGY_LOGT_DEV)) {
            return;
        }

        va_list args;
        va_start(args, format);

        int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
        vector<TCHAR> buffer(len, 0);
        _vstprintf_s(buffer.data(), len, format, args);
        va_end(args);
        if (m_log.get() != nullptr) {
            m_log->write(log_level, RGY_LOGT_DEV, buffer.data());
        } else {
            _ftprintf(stderr, _T("%s"), buffer.data());
        }
    }
    QSVDeviceNum m_devNum;
    std::unique_ptr<CQSVHWDevice> m_hwdev;
    std::unique_ptr<RGYOpenCLDeviceInfo> m_devInfo;
    MFXVideoSession2 m_session;
    std::unique_ptr<QSVAllocator> m_allocator;
    bool m_externalAlloc;
    MemType m_memType;
    std::vector<QSVEncFeatureData> m_featureData;
    std::shared_ptr<RGYLog> m_log;
};

std::vector<std::unique_ptr<QSVDevice>> getDeviceList(const QSVDeviceNum dev, const bool enableOpenCL, const MemType memType, std::shared_ptr<RGYLog> log);

#endif //_QSV_DEVICE_H_
