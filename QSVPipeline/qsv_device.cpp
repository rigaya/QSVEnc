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

#include <iostream>
#include <fstream>
#include "qsv_util.h"
#include "qsv_session.h"
#include "qsv_device.h"
#include "gpu_info.h"
#include "rgy_avutil.h"

QSVDeviceInfoCache::QSVDeviceInfoCache() : RGYDeviceInfoCache(), m_featureData() { }
QSVDeviceInfoCache::~QSVDeviceInfoCache() { }

RGY_ERR QSVDeviceInfoCache::parseEncFeatures(std::ifstream& cacheFile) {
    m_featureData.clear();

    std::string line;
    while (std::getline(cacheFile, line)) {
        QSVEncFeatureData featureData;

        std::istringstream iss(line);
        int deviceId = 0;
        if (!(iss >> deviceId)) {
            return RGY_ERR_INVALID_FORMAT; // デバイスID読み取りエラー
        }
        featureData.dev = (QSVDeviceNum)deviceId;

        std::string codecNameStr;
        if (!(iss >> codecNameStr)) {
            return RGY_ERR_INVALID_FORMAT; // コーデック名読み取りエラー
        }
        featureData.codec = (RGY_CODEC)get_cx_value(list_rgy_codec, char_to_tstring(codecNameStr).c_str());
        if (featureData.codec == RGY_CODEC_UNKNOWN) {
            return RGY_ERR_INVALID_VIDEO_PARAM; // コーデック名変換エラー
        }
        if (!(iss >> featureData.lowPwer)) {
            return RGY_ERR_INVALID_FORMAT; // lowpower読み取りエラー
        }
        std::string ratecontrolStr;
        if (!(iss >> ratecontrolStr)) {
            return RGY_ERR_INVALID_FORMAT; // レート制御読み取りエラー
        }

        const int ratecontrol = get_value_from_chr(list_rc_mode, char_to_tstring(ratecontrolStr).c_str());
        if (ratecontrol == PARSE_ERROR_FLAG) {
            return RGY_ERR_INVALID_VIDEO_PARAM; // レート制御変換エラー
        }

        QSVEncFeatures encFeature;
        std::string featureStr;
        while (std::getline(iss, featureStr, ',')) {
            const auto featureName = trim(char_to_tstring(featureStr));
            if (const auto feature = qsv_feature_params_str_to_enm(featureName); feature != ENC_FEATURE_PARAMS_NONE) {
                encFeature |= feature;
            } else if (const auto rc_ext = qsv_feature_rc_ext_str_to_enm(featureName); rc_ext != ENC_FEATURE_RCEXT_NONE) {
                encFeature |= rc_ext;
            }
        }
        featureData.feature[ratecontrol] = encFeature;

        auto entry = std::find_if(m_featureData.begin(), m_featureData.end(), [featureData](const QSVEncFeatureData& data) {
            return data.dev == featureData.dev && data.codec == featureData.codec && data.lowPwer == featureData.lowPwer;
        });
        if (entry != m_featureData.end()) {
            entry->feature[ratecontrol] = encFeature;
        } else {
            m_featureData.push_back(featureData);
        }
    }
    return RGY_ERR_NONE;
}

void QSVDeviceInfoCache::writeEncFeatures(std::ofstream& cacheFile) {
    cacheFile << ENC_FEATURES_START_LINE << std::endl;

    for (const auto& featureData : m_featureData) {
        for (const auto& feature : featureData.feature) {
            cacheFile << static_cast<int>(featureData.dev) << " "
                << tchar_to_string(get_cx_desc(list_rgy_codec, featureData.codec)) << " "
                << featureData.lowPwer << " "
                << tchar_to_string(get_cx_desc(list_rc_mode, feature.first)) << " ";
            const auto encFeatureNameUnknown = qsv_feature_enm_to_str((QSVEncFeatureParams)-1ll);
            const auto encRCExtNameUnknown = qsv_feature_enm_to_str((QSVEncFeatureRCExt)-1ll);
            bool first = true;
            for (size_t i = 0; i < sizeof(QSVEncFeatureParams) * 8; i++) {
                const auto flag = (QSVEncFeatureParams)(1llu << i);
                if (qsv_feature_enm_to_str(flag) == encFeatureNameUnknown) {
                    continue;
                }
                if (feature.second & flag) {
                    if (!first) {
                        cacheFile << ",";
                    }
                    cacheFile << tchar_to_string(qsv_feature_enm_to_str(flag));
                    first = false;
                }
            }
            for (size_t i = 0; i < sizeof(QSVEncFeatureRCExt) * 8; i++) {
                const auto flag = (QSVEncFeatureRCExt)(1llu << i);
                if (qsv_feature_enm_to_str(flag) == encRCExtNameUnknown) {
                    continue;
                }
                if (feature.second & flag) {
                    if (!first) {
                        cacheFile << ",";
                    }
                    cacheFile << tchar_to_string(qsv_feature_enm_to_str(flag));
                    first = false;
                }
            }
            cacheFile << std::endl;
        }
    }
}

void QSVDeviceInfoCache::clearFeatureCache() {
    m_deviceDecCodecCsp.clear();
    m_featureData.clear();
    m_dataUpdated = true;
}

RGY_ERR QSVDeviceInfoCache::addEncFeature(const QSVEncFeatureData& encFeatures) {
    auto entry = std::find_if(m_featureData.begin(), m_featureData.end(), [encFeatures](const QSVEncFeatureData& data) {
        return data.dev == encFeatures.dev && data.codec == encFeatures.codec && data.lowPwer == encFeatures.lowPwer;
        });
    if (entry != m_featureData.end()) {
        for (const auto& feature : encFeatures.feature) {
            if (entry->feature.count(feature.first) == 0
                || entry->feature[feature.first] != feature.second) {
                entry->feature[feature.first] = feature.second;
                m_dataUpdated = true;
            }
        }
    } else {
        m_featureData.push_back(encFeatures);
        m_dataUpdated = true;
    }
    return RGY_ERR_NONE;
}

RGY_ERR QSVDeviceInfoCache::addEncFeature(const std::vector<QSVEncFeatureData>& encFeatures) {
    for (const auto& encFeature : encFeatures) {
        if (auto sts = addEncFeature(encFeature); sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

std::pair<RGY_ERR, QSVEncFeatures> QSVDeviceInfoCache::getEncodeFeature(const QSVDeviceNum dev, const int ratecontrol, const RGY_CODEC codec, const bool lowpower) {
    auto target = std::find_if(m_featureData.begin(), m_featureData.end(), [dev, codec, lowpower](const QSVEncFeatureData& data) {
        return data.dev == dev && data.codec == codec && data.lowPwer == lowpower;
        });
    if (target != m_featureData.end() && target->feature.count(ratecontrol) > 0) {
        return { RGY_ERR_NONE, target->feature[ratecontrol] };
    }
    return { RGY_ERR_NOT_FOUND, QSVEncFeatures() };
}

std::vector<QSVEncFeatureData> QSVDeviceInfoCache::getEncodeFeatures(const QSVDeviceNum dev) {
    std::vector<QSVEncFeatureData> featureData;
    for (const auto& data : m_featureData) {
        if (data.dev == dev) {
            featureData.push_back(data);
        }
    }
    return featureData;
}

QSVDevice::QSVDevice() :
    m_devNum(QSVDeviceNum::AUTO),
    m_hwdev(),
    m_devInfo(),
#if ENABLE_VULKAN
    m_vulkan(),
#endif
    m_session(),
    m_sessionParams(),
    m_allocator(),
    m_externalAlloc(false),
    m_memType(HW_MEMORY),
    m_featureData(),
    m_devInfoCache(),
    m_log() {
    m_log = std::make_shared<RGYLog>(nullptr, RGY_LOG_QUIET);
}

QSVDevice::~QSVDevice() {
    close();
}

void QSVDevice::close() {
    m_devInfoCache.reset();
    PrintMes(RGY_LOG_DEBUG, _T("Close device %d...\n"), (int)m_devNum);
    PrintMes(RGY_LOG_DEBUG, _T("Closing session...\n"));
    m_session.Close();
    PrintMes(RGY_LOG_DEBUG, _T("Closing device...\n"));
    m_hwdev.reset();
    // allocator if used as external for MediaSDK must be deleted after SDK components
    PrintMes(RGY_LOG_DEBUG, _T("Closing allocator...\n"));
    m_allocator.reset();
    m_featureData.clear();
#if ENABLE_VULKAN
    m_vulkan.reset();
#endif
    m_devInfo.reset();
    PrintMes(RGY_LOG_DEBUG, _T("Device %d closed.\n"), (int)m_devNum);
    m_log.reset();
}

RGY_ERR QSVDevice::init(const QSVDeviceNum dev, const bool enableOpenCL, const RGYParamInitVulkan enableVulkan, MemType memType, const MFXVideoSession2Params& params, std::shared_ptr<QSVDeviceInfoCache> devInfoCache, std::shared_ptr<RGYLog> log, const bool suppressErrorMessage) {
    m_log = log;
    m_memType = memType;
    m_sessionParams = params;
    m_devInfoCache = devInfoCache;
    return init(dev, enableOpenCL, enableVulkan, suppressErrorMessage);
}

RGY_ERR QSVDevice::init(const QSVDeviceNum dev, const bool enableOpenCL, [[maybe_unused]] const RGYParamInitVulkan enableVulkan, const bool suppressErrorMessage) {
    m_devNum = dev;
#if ENABLE_VULKAN
    if (enableVulkan == RGYParamInitVulkan::TargetVendor) {
        setenv("VK_LOADER_DRIVERS_SELECT", "*intel*", 1);
    }
#endif
    PrintMes(RGY_LOG_DEBUG, _T("QSVDevice::init: Start initializing device %d... memType: %s\n"), m_devNum, MemTypeToStr(m_memType));
    if (auto err = InitSessionAndDevice(m_hwdev, m_session, m_memType, m_devNum, m_sessionParams, m_log, suppressErrorMessage); err != RGY_ERR_NONE) {
        PrintMes((suppressErrorMessage) ? RGY_LOG_DEBUG : RGY_LOG_ERROR, _T("QSVDevice::init: failed to initialize session: %s.\n"), get_err_mes(err));
        return err;
    }
    PrintMes(RGY_LOG_DEBUG, _T("QSVDevice::init: initialized session with memType %s.\n"), MemTypeToStr(m_memType));

    auto err = CreateAllocator(m_allocator, m_externalAlloc, m_memType, m_hwdev.get(), m_session, m_log);
    if (err != RGY_ERR_NONE) {
        PrintMes(RGY_LOG_ERROR, _T("QSVDevice::init: failed to create allocator: %s.\n"), get_err_mes(err));
        return err;
    }
    PrintMes(RGY_LOG_DEBUG, _T("QSVDevice::init: initialized allocator.\n"));

    if (!enableOpenCL) {
        return RGY_ERR_NONE;
    }
    RGYOpenCL cl(m_log);
    if (!RGYOpenCL::openCLloaded()) {
        PrintMes(RGY_LOG_DEBUG, _T("Failed to load OpenCL.\n"));
        return RGY_ERR_NONE;
    }
    auto clPlatforms = cl.getPlatforms("Intel");
    const mfxHandleType hdl_t = mfxHandleTypeFromMemType(m_memType, true);
    mfxHDL hdl = nullptr;
    if (hdl_t
        && err_to_rgy(m_hwdev->GetHandle((hdl_t == MFX_HANDLE_DIRECT3D_DEVICE_MANAGER9) ? (mfxHandleType)0 : hdl_t, &hdl)) == RGY_ERR_NONE) {
        for (auto& platform : clPlatforms) {
            if (m_memType == D3D9_MEMORY && ENABLE_RGY_OPENCL_D3D9) {
                if (platform->createDeviceListD3D9(CL_DEVICE_TYPE_GPU, (void*)hdl, true) == CL_SUCCESS && platform->devs().size() > 0) {
                    m_devInfo = std::make_unique<RGYOpenCLDeviceInfo>(platform->dev(0).info());
                    break;
                }
            } else if (m_memType == D3D11_MEMORY && ENABLE_RGY_OPENCL_D3D11) {
                if (platform->createDeviceListD3D11(CL_DEVICE_TYPE_GPU, (void*)hdl, true) == CL_SUCCESS && platform->devs().size() > 0) {
                    m_devInfo = std::make_unique<RGYOpenCLDeviceInfo>(platform->dev(0).info());
                    break;
                }
            } else if (m_memType == VA_MEMORY && ENABLE_RGY_OPENCL_VA) {
                if (platform->createDeviceListVA(CL_DEVICE_TYPE_GPU, (void*)hdl, true) == CL_SUCCESS && platform->devs().size() > 0) {
                    m_devInfo = std::make_unique<RGYOpenCLDeviceInfo>(platform->dev(0).info());
                    break;
                }
            } else {
                if (platform->createDeviceList(CL_DEVICE_TYPE_GPU) == CL_SUCCESS && platform->devs().size() > 0) {
                    m_devInfo = std::make_unique<RGYOpenCLDeviceInfo>(platform->dev(0).info());
                    break;
                }
            }
        }
        if (!m_devInfo) {
            PrintMes((suppressErrorMessage) ? RGY_LOG_DEBUG : RGY_LOG_ERROR, _T("QSVDevice::init:   failed to find OpenCL device for dev #%d.\n"), dev);
        }
    }
#if ENABLE_VULKAN
    if (enableVulkan != RGYParamInitVulkan::Disable) {
        if (!m_devInfo) {
            PrintMes(RGY_LOG_WARN, _T("QSVDevice::init: OpenCL device not found, Vulkan device also not found.\n"));
        } else if (!m_devInfo->checkExtension(CL_KHR_DEVICE_UUID_EXTENSION_NAME)) {
            PrintMes(RGY_LOG_WARN, _T("QSVDevice::init: OpenCL device found, but does not support %s, Vulkan device could not be found.\n"), char_to_tstring(CL_KHR_DEVICE_UUID_EXTENSION_NAME).c_str());
        } else {
            auto uuidToString = [](const void *uuid) {
                tstring str;
                const uint8_t *buf = (const uint8_t *)uuid;
                for (size_t i = 0; i < VK_UUID_SIZE; ++i) {
                    str += strsprintf(_T("%02x"), buf[i]);
                }
                return str;
            };
            PrintMes(RGY_LOG_DEBUG, _T("QSVDevice::init: OpenCL device uuid %s.\n"), uuidToString(m_devInfo->uuid).c_str());

            auto vkdev = std::make_unique<DeviceVulkan>();
            int vkDevCount = vkdev->adapterCount();
            PrintMes(RGY_LOG_DEBUG, _T("QSVDevice::init: Vulkan device count: %d.\n"), vkDevCount);

            std::vector<const char *> extInstance;
            extInstance.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
            extInstance.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
            
            std::vector<const char *> extDevice;
            extDevice.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
            extDevice.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
            extDevice.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
#if defined(_WIN32) || defined(_WIN64)
            extDevice.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
            extDevice.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
            extDevice.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
            extDevice.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif //defined(_WIN32) || defined(_WIN64)

            for (int ivkdev = 0; ivkdev < vkDevCount; ivkdev++) {
                if (ivkdev > 0) {
                    vkdev = std::make_unique<DeviceVulkan>();
                }
                PrintMes(RGY_LOG_DEBUG, _T("Init Vulkan device %d...\n"), ivkdev);
                if ((err = vkdev->Init(ivkdev, extInstance, extDevice, m_log, true)) != RGY_ERR_NONE) {
                    PrintMes(RGY_LOG_DEBUG, _T("Failed to init Vulkan device %d, name %s, uuid %s.\n"), ivkdev);
                    continue;
                }
                PrintMes(RGY_LOG_DEBUG, _T("Init Vulkan device %d, name %s, uuid %s.\n"), ivkdev, char_to_tstring(vkdev->GetDisplayDeviceName()).c_str(), uuidToString(vkdev->GetUUID()).c_str());
                if (memcmp(vkdev->GetUUID(), m_devInfo->uuid, VK_UUID_SIZE) == 0) {
                    m_vulkan = std::move(vkdev);
                    break;
                }
            }
        }
    }
#endif
    if (m_devInfoCache) {
        m_featureData = m_devInfoCache->getEncodeFeatures(m_devNum);
    }
    return RGY_ERR_NONE;
}

tstring QSVDevice::name() const {
    if (m_devInfo) {
        auto gpu_name = m_devInfo->name;
        gpu_name = str_replace(gpu_name, "(R)", "");
        gpu_name = str_replace(gpu_name, "(TM)", "");
        return char_to_tstring(gpu_name);
    }
    return strsprintf(_T("device #%d"), m_devNum);
}

QSV_CPU_GEN QSVDevice::CPUGen() {
    return getCPUGen(&m_session);
}

int QSVDevice::adapterType() {
    mfxPlatform platform = { 0 };
    m_session.QueryPlatform(&platform);
    return platform.MediaAdapterType;
}

LUID QSVDevice::luid() {
    return (m_hwdev) ? m_hwdev->GetLUID() : LUID();
}

CodecCsp QSVDevice::getDecodeCodecCsp(const bool skipHWDecodeCheck) {
    if (m_devInfoCache) {
        auto& devDecCsp = m_devInfoCache->getDeviceDecCodecCsp();
        for (const auto& devCsp : devDecCsp) {
            if (devCsp.first == (int)m_devNum) {
                return devCsp.second;
            }
        }
    }
    vector<RGY_CODEC> codecLists;
    for (int i = 0; i < _countof(HW_DECODE_LIST); i++) {
        codecLists.push_back(HW_DECODE_LIST[i].rgy_codec);
    }
    auto codecCsp = MakeDecodeFeatureList(m_session, codecLists, m_log, skipHWDecodeCheck);
    if (m_devInfoCache) {
        m_devInfoCache->setDecCodecCsp(tchar_to_string(name()), { (int)m_devNum, codecCsp });
    }
    return codecCsp;
}

QSVEncFeatures QSVDevice::getEncodeFeature(const int ratecontrol, const RGY_CODEC codec, const bool lowpower) {
    auto target = std::find_if(m_featureData.begin(), m_featureData.end(), [codec, lowpower](const QSVEncFeatureData& data) {
        return data.codec == codec && data.lowPwer == lowpower;
        });
    if (target != m_featureData.end() && target->feature.count(ratecontrol) > 0) {
        return target->feature[ratecontrol];
    }
    //チェックする際は専用のsessionを作成するようにしないと異常終了することがある
    auto resultData = MakeFeatureList(m_devNum, { ratecontrol }, codec, lowpower, m_log);
    QSVEncFeatures& result = resultData.feature[ratecontrol];
    if (target != m_featureData.end()) {
        target->feature[ratecontrol] = result;
    } else {
        QSVEncFeatureData data;
        data.codec = codec;
        data.lowPwer = lowpower;
        data.dev = m_devNum;
        data.feature[ratecontrol] = result;
        m_featureData.push_back(data);
    }
    if (m_devInfoCache) {
        target = std::find_if(m_featureData.begin(), m_featureData.end(), [codec, lowpower](const QSVEncFeatureData& data) {
            return data.codec == codec && data.lowPwer == lowpower;
            });
        if (target != m_featureData.end()) {
            m_devInfoCache->addEncFeature(*target);
        }
    }
    return result;
}

std::optional<RGYOpenCLDeviceInfo> getDeviceCLInfoQSV(const QSVDeviceNum deviceNum) {
    auto dev = std::make_unique<QSVDevice>();
    if (dev->init(deviceNum, true, RGYParamInitVulkan::TargetVendor, true) == RGY_ERR_NONE && dev->devInfo()) {
        return std::optional<RGYOpenCLDeviceInfo>(*dev->devInfo());
    }
    return std::optional<RGYOpenCLDeviceInfo>();
}

std::vector<std::unique_ptr<QSVDevice>> getDeviceList(const QSVDeviceNum deviceNum, const bool enableOpenCL, const RGYParamInitVulkan enableVulkan, const MemType memType, const MFXVideoSession2Params& params, std::shared_ptr<QSVDeviceInfoCache> devInfoCache, std::shared_ptr<RGYLog> log) {
    auto openCLAvail = enableOpenCL;
    if (enableOpenCL) {
        RGYOpenCL cl(std::make_shared<RGYLog>(nullptr, RGY_LOG_QUIET));
        openCLAvail = RGYOpenCL::openCLloaded();
    }
    log->write(RGY_LOG_DEBUG, RGY_LOGT_DEV, _T("Start Create DeviceList, openCLAvail: %s.\n"), openCLAvail ? _T("yes") : _T("no"));

    std::vector<std::unique_ptr<QSVDevice>> devList;
    const int idevstart = (deviceNum != QSVDeviceNum::AUTO) ? (int)deviceNum : 1;
    const int idevfin   = (deviceNum != QSVDeviceNum::AUTO) ? (int)deviceNum : (int)QSVDeviceNum::MAX;
    for (int idev = idevstart; idev <= idevfin; idev++) {
        log->write(RGY_LOG_DEBUG, RGY_LOGT_DEV, _T("Check device %d...\n"), idev);
        auto dev = std::make_unique<QSVDevice>();
        if (dev->init((QSVDeviceNum)idev, enableOpenCL && openCLAvail, enableVulkan, memType, params, devInfoCache, log, idev != idevstart) != RGY_ERR_NONE) {
            break;
        }
        devList.push_back(std::move(dev));
    }
    return devList;
}
