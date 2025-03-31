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
#include "rgy_device_vulkan.h"
#include "rgy_device_info_cache.h"

class QSVDeviceInfoCache : public RGYDeviceInfoCache {
public:
    QSVDeviceInfoCache();
    virtual ~QSVDeviceInfoCache();
    RGY_ERR addEncFeature(const QSVEncFeatureData& encFeatures);
    RGY_ERR addEncFeature(const std::vector<QSVEncFeatureData>& encFeatures);
    std::pair<RGY_ERR, QSVEncFeatures> getEncodeFeature(const QSVDeviceNum dev, const int ratecontrol, const RGY_CODEC codec, const bool lowpower);
    std::vector<QSVEncFeatureData> getEncodeFeatures(const QSVDeviceNum dev);
protected:
    virtual RGY_ERR parseEncFeatures(std::ifstream& cacheFile) override;
    virtual void writeEncFeatures(std::ofstream& cacheFile) override;

    virtual void clearFeatureCache() override;

    std::vector<QSVEncFeatureData> m_featureData;
};

class QSVDevice {
public:
    QSVDevice();
    virtual ~QSVDevice();

    RGY_ERR init(const QSVDeviceNum dev, const bool enableOpenCL, const RGYParamInitVulkan enableVulkan, const bool suppressErrorMessage);
    RGY_ERR init(const QSVDeviceNum dev, const bool enableOpenCL, const RGYParamInitVulkan enableVulkan, MemType memType, const MFXVideoSession2Params& params, std::shared_ptr<QSVDeviceInfoCache> devInfoCache, std::shared_ptr<RGYLog> m_log, const bool suppressErrorMessage);

    CodecCsp getDecodeCodecCsp(const bool skipHWDecodeCheck);
    QSVEncFeatures getEncodeFeature(const int ratecontrol, const RGY_CODEC codec, const bool lowpower);

    void close();

    tstring name() const;
    LUID luid();
    QSV_CPU_GEN CPUGen();
    int adapterType();

    QSVDeviceNum deviceNum() const { return m_devNum; };
    MemType memType() const { return m_memType; };
    CQSVHWDevice *hwdev() { return m_hwdev.get(); }
    QSVAllocator *allocator() { return m_allocator.get(); }
#if ENABLE_VULKAN
    DeviceVulkan *vulkan() { return m_vulkan.get(); }
#else
    DeviceVulkan *vulkan() { return nullptr; }
#endif
    const IntelDeviceInfo *intelDeviceInfo() const { return (m_hwdev) ? m_hwdev->GetIntelDeviceInfo() : nullptr; }
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
#if ENABLE_VULKAN
    std::unique_ptr<DeviceVulkan> m_vulkan;
#endif
    MFXVideoSession2 m_session;
    MFXVideoSession2Params m_sessionParams;
    std::unique_ptr<QSVAllocator> m_allocator;
    bool m_externalAlloc;
    MemType m_memType;
    std::vector<QSVEncFeatureData> m_featureData;
    std::shared_ptr<QSVDeviceInfoCache> m_devInfoCache;
    std::shared_ptr<RGYLog> m_log;
};

std::vector<std::unique_ptr<QSVDevice>> getDeviceList(const QSVDeviceNum dev, const bool enableOpenCL, const RGYParamInitVulkan enableVulkan, const MemType memType, const MFXVideoSession2Params& params, std::shared_ptr<QSVDeviceInfoCache> devInfoCache, std::shared_ptr<RGYLog> log);

#endif //_QSV_DEVICE_H_
