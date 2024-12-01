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

#include "qsv_util.h"
#include "qsv_session.h"
#include "qsv_device.h"
#include "gpu_info.h"
#include "rgy_avutil.h"

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
    m_log() {
    m_log = std::make_shared<RGYLog>(nullptr, RGY_LOG_QUIET);
}

QSVDevice::~QSVDevice() {
    close();
}

void QSVDevice::close() {
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

RGY_ERR QSVDevice::init(const QSVDeviceNum dev, const bool enableOpenCL, const bool enableVulkan, MemType memType, const MFXVideoSession2Params& params, std::shared_ptr<RGYLog> log, const bool suppressErrorMessage) {
    m_log = log;
    m_memType = memType;
    m_sessionParams = params;
    return init(dev, enableOpenCL, enableVulkan, suppressErrorMessage);
}

RGY_ERR QSVDevice::init(const QSVDeviceNum dev, const bool enableOpenCL, [[maybe_unused]] const bool enableVulkan, const bool suppressErrorMessage) {
    m_devNum = dev;
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
    if (enableVulkan) {
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
    vector<RGY_CODEC> codecLists;
    for (int i = 0; i < _countof(HW_DECODE_LIST); i++) {
        codecLists.push_back(HW_DECODE_LIST[i].rgy_codec);
    }
    return MakeDecodeFeatureList(m_session, codecLists, m_log, skipHWDecodeCheck);
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
    return result;
}

std::optional<RGYOpenCLDeviceInfo> getDeviceCLInfoQSV(const QSVDeviceNum deviceNum) {
    auto dev = std::make_unique<QSVDevice>();
    if (dev->init(deviceNum, true, true, true) == RGY_ERR_NONE && dev->devInfo()) {
        return std::optional<RGYOpenCLDeviceInfo>(*dev->devInfo());
    }
    return std::optional<RGYOpenCLDeviceInfo>();
}

std::vector<std::unique_ptr<QSVDevice>> getDeviceList(const QSVDeviceNum deviceNum, const bool enableOpenCL, const bool enableVulkan, const MemType memType, const MFXVideoSession2Params& params, std::shared_ptr<RGYLog> log) {
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
        if (dev->init((QSVDeviceNum)idev, enableOpenCL && openCLAvail, enableVulkan, memType, params, log, idev != idevstart) != RGY_ERR_NONE) {
            break;
        }
        devList.push_back(std::move(dev));
    }
    return devList;
}
