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

QSVDevice::QSVDevice() :
    m_devNum(QSVDeviceNum::AUTO),
    m_hwdev(),
    m_devInfo(),
    m_session(),
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
    m_devInfo.reset();
    PrintMes(RGY_LOG_DEBUG, _T("Device %d closed.\n"), (int)m_devNum);
    m_log.reset();
}

RGY_ERR QSVDevice::init(const QSVDeviceNum dev, const bool enableOpenCL, MemType memType, std::shared_ptr<RGYLog> log, const bool suppressErrorMessage) {
    m_log = log;
    m_memType = memType;
    return init(dev, enableOpenCL, suppressErrorMessage);
}

RGY_ERR QSVDevice::init(const QSVDeviceNum dev, const bool enableOpenCL, const bool suppressErrorMessage) {
    m_devNum = dev;
    PrintMes(RGY_LOG_DEBUG, _T("QSVDevice::init: Start initializing device %d... memType: %s\n"), m_devNum, MemTypeToStr(m_memType));
    MFXVideoSession2Params params;
    if (auto err = InitSessionAndDevice(m_hwdev, m_session, m_memType, m_devNum, params, m_log, suppressErrorMessage); err != RGY_ERR_NONE) {
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
                    return RGY_ERR_NONE;
                }
            } else if (m_memType == D3D11_MEMORY && ENABLE_RGY_OPENCL_D3D11) {
                if (platform->createDeviceListD3D11(CL_DEVICE_TYPE_GPU, (void*)hdl, true) == CL_SUCCESS && platform->devs().size() > 0) {
                    m_devInfo = std::make_unique<RGYOpenCLDeviceInfo>(platform->dev(0).info());
                    return RGY_ERR_NONE;
                }
            } else if (m_memType == VA_MEMORY && ENABLE_RGY_OPENCL_VA) {
                if (platform->createDeviceListVA(CL_DEVICE_TYPE_GPU, (void*)hdl, true) == CL_SUCCESS && platform->devs().size() > 0) {
                    m_devInfo = std::make_unique<RGYOpenCLDeviceInfo>(platform->dev(0).info());
                    return RGY_ERR_NONE;
                }
            } else {
                if (platform->createDeviceList(CL_DEVICE_TYPE_GPU) == CL_SUCCESS && platform->devs().size() > 0) {
                    m_devInfo = std::make_unique<RGYOpenCLDeviceInfo>(platform->dev(0).info());
                    return RGY_ERR_NONE;
                }
            }
        }
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

uint64_t QSVDevice::getEncodeFeature(const int ratecontrol, const RGY_CODEC codec, const bool lowpower) {
    auto target = std::find_if(m_featureData.begin(), m_featureData.end(), [codec, lowpower](const QSVEncFeatureData& data) {
        return data.codec == codec && data.lowPwer == lowpower;
        });
    if (target != m_featureData.end() && target->feature.count(ratecontrol) > 0) {
        return target->feature[ratecontrol];
    }
    const auto result = CheckEncodeFeatureWithPluginLoad(m_session, ratecontrol, codec, lowpower);
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
    if (dev->init(deviceNum, true, true) == RGY_ERR_NONE && dev->devInfo()) {
        return std::optional<RGYOpenCLDeviceInfo>(*dev->devInfo());
    }
    return std::optional<RGYOpenCLDeviceInfo>();
}

std::vector<std::unique_ptr<QSVDevice>> getDeviceList(const QSVDeviceNum deviceNum, const bool enableOpenCL, const MemType memType, std::shared_ptr<RGYLog> log) {
    auto openCLAvail = enableOpenCL;
    if (enableOpenCL) {
        RGYOpenCL cl(std::make_shared<RGYLog>(nullptr, RGY_LOG_QUIET));
        openCLAvail = RGYOpenCL::openCLloaded();
    }
    std::vector<std::unique_ptr<QSVDevice>> devList;
    const int idevstart = (deviceNum != QSVDeviceNum::AUTO) ? (int)deviceNum : 1;
    const int idevfin   = (deviceNum != QSVDeviceNum::AUTO) ? (int)deviceNum : (int)QSVDeviceNum::MAX;
    for (int idev = idevstart; idev <= idevfin; idev++) {
        auto dev = std::make_unique<QSVDevice>();
        if (dev->init((QSVDeviceNum)idev, enableOpenCL && openCLAvail, memType, log, idev != idevstart) != RGY_ERR_NONE) {
            break;
        }
        devList.push_back(std::move(dev));
    }
    return devList;
}
