// -----------------------------------------------------------------------------------------
// RGY by rigaya
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

#ifndef __RGY_FILTER_LIBPLACEBO_H__
#define __RGY_FILTER_LIBPLACEBO_H__

#include "rgy_filter_cl.h"
#include "rgy_bitstream.h"
#include "rgy_libplacebo.h"
#include "rgy_device_vulkan.h"
#include "rgy_prm.h"
#include <array>

class RGYFilterParamLibplacebo : public RGYFilterParam {
public:
    DeviceVulkan *vk;
    VideoVUIInfo vui;
    RGYFilterParamLibplacebo() : vk(nullptr), vui() {};
    virtual ~RGYFilterParamLibplacebo() {};
};

class RGYFilterParamLibplaceboResample : public RGYFilterParamLibplacebo {
public:
    RGY_VPP_RESIZE_ALGO resize_algo;
    VppLibplaceboResample resample;
    RGYFilterParamLibplaceboResample() : RGYFilterParamLibplacebo(), resize_algo(RGY_VPP_RESIZE_AUTO), resample() {};
    virtual ~RGYFilterParamLibplaceboResample() {};
    virtual tstring print() const override;
};

class RGYFilterParamLibplaceboDeband : public RGYFilterParamLibplacebo {
public:
    VppLibplaceboDeband deband;
    RGYFilterParamLibplaceboDeband() : RGYFilterParamLibplacebo(), deband() {};
    virtual ~RGYFilterParamLibplaceboDeband() {};
    virtual tstring print() const override;
};

class RGYFilterParamLibplaceboToneMapping : public RGYFilterParamLibplacebo {
public:
    VppLibplaceboToneMapping toneMapping;
    VideoVUIInfo vui;
    const RGYHDRMetadata *hdrMetadataIn;
    const RGYHDRMetadata *hdrMetadataOut;
    RGYFilterParamLibplaceboToneMapping() : RGYFilterParamLibplacebo(), toneMapping(), vui(), hdrMetadataIn(nullptr), hdrMetadataOut(nullptr) {};
    virtual ~RGYFilterParamLibplaceboToneMapping() {};
    virtual tstring print() const override;
};

class RGYFilterParamLibplaceboShader : public RGYFilterParamLibplacebo {
public:
    VppLibplaceboShader shader;
    RGYFilterParamLibplaceboShader() : RGYFilterParamLibplacebo(), shader() {};
    virtual ~RGYFilterParamLibplaceboShader() {};
    virtual tstring print() const override;
};

#if ENABLE_LIBPLACEBO

#if ENABLE_D3D11
struct RGYFrameD3D11 : public RGYFrame {
public:
    RGYFrameD3D11();
    virtual ~RGYFrameD3D11();
    virtual RGY_ERR allocate(ID3D11Device *device, const int width, const int height, const RGY_CSP csp, const int bitdepth);
    virtual void deallocate();
    const RGYFrameInfo& frameInfo() { return frame; }
    virtual bool isempty() const { return !frame.ptr[0]; }
    virtual void setTimestamp(uint64_t timestamp) override { frame.timestamp = timestamp; }
    virtual void setDuration(uint64_t duration) override { frame.duration = duration; }
    virtual void setPicstruct(RGY_PICSTRUCT picstruct) override { frame.picstruct = picstruct; }
    virtual void setInputFrameId(int id) override { frame.inputFrameId = id; }
    virtual void setFlags(RGY_FRAME_FLAGS frameflags) override { frame.flags = frameflags; }
    virtual void clearDataList() override { frame.dataList.clear(); }
    virtual const std::vector<std::shared_ptr<RGYFrameData>>& dataList() const override { return frame.dataList; }
    virtual std::vector<std::shared_ptr<RGYFrameData>>& dataList() override { return frame.dataList; }
    virtual void setDataList(const std::vector<std::shared_ptr<RGYFrameData>>& dataList) override { frame.dataList = dataList; }
    
    virtual RGYCLFrameInterop *getCLFrame(RGYOpenCLContext *clctx, RGYOpenCLQueue& queue);
    virtual void resetCLFrame() { clframe.reset(); }
protected:
    RGYFrameD3D11(const RGYFrameD3D11 &) = delete;
    void operator =(const RGYFrameD3D11 &) = delete;
    virtual RGYFrameInfo getInfo() const override {
        return frame;
    }
    RGYFrameInfo frame;
    std::unique_ptr<RGYCLFrameInterop> clframe;
};
#elif ENABLE_VULKAN
struct RGYFrameVulkanImage {
protected:
    DeviceVulkan *m_vk;
    VkImage m_image;
    VkDeviceMemory m_bufferMemory;
    uint64_t m_bufferSize;
public:
    RGYFrameVulkanImage();
    virtual ~RGYFrameVulkanImage();

    virtual RGY_ERR alloc(DeviceVulkan *vk, const int width, const int height, const VkFormat format, const VkBufferUsageFlags usage);
    virtual void deallocate();
    VkImage image() { return m_image; }
    VkDeviceMemory bufferMemory() const { return m_bufferMemory; }
protected:
    VkExternalMemoryHandleTypeFlagBits getDefaultMemHandleType() const;
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void *getMemHandle(VkExternalMemoryHandleTypeFlagBits handleType);
};

struct RGYFrameVulkan : public RGYFrame {
public:
    RGYFrameVulkan();
    virtual ~RGYFrameVulkan();
    virtual RGY_ERR allocate(DeviceVulkan *vk, const int width, const int height, const RGY_CSP csp, const int bitdepth);
    virtual void deallocate();
    const RGYFrameInfo& frameInfo() { return frame; }
    virtual bool isempty() const { return !frame.ptr[0]; }
    virtual void setTimestamp(uint64_t timestamp) override { frame.timestamp = timestamp; }
    virtual void setDuration(uint64_t duration) override { frame.duration = duration; }
    virtual void setPicstruct(RGY_PICSTRUCT picstruct) override { frame.picstruct = picstruct; }
    virtual void setInputFrameId(int id) override { frame.inputFrameId = id; }
    virtual void setFlags(RGY_FRAME_FLAGS frameflags) override { frame.flags = frameflags; }
    virtual void clearDataList() override { frame.dataList.clear(); }
    virtual const std::vector<std::shared_ptr<RGYFrameData>>& dataList() const override { return frame.dataList; }
    virtual std::vector<std::shared_ptr<RGYFrameData>>& dataList() override { return frame.dataList; }
    virtual void setDataList(const std::vector<std::shared_ptr<RGYFrameData>>& dataList) override { frame.dataList = dataList; }
    
    virtual RGYCLFrame *getCLFrame(RGYOpenCLContext *clctx, cl_mem_flags flags = CL_MEM_READ_WRITE);
    virtual void resetCLFrame() { m_clframe.reset(); }
    VkFormat format() const { return m_format; }
    VkBufferUsageFlags usage() const { return m_usage; }
protected:
    RGYFrameVulkan(const RGYFrameVulkan &) = delete;
    void operator =(const RGYFrameVulkan &) = delete;
    virtual RGYFrameInfo getInfo() const override {
        return frame;
    }

    DeviceVulkan *m_vk;
    VkFormat m_format;
    VkBufferUsageFlags m_usage;
    std::vector<std::unique_ptr<RGYFrameVulkanImage>> m_imgs;

    RGYFrameInfo frame;
    std::unique_ptr<RGYCLFrame> m_clframe;
};

struct RGYSemaphoreVulkan {
protected:
    DeviceVulkan *m_vk;
    VkSemaphore m_semaphore;
    std::unique_ptr<RGYOpenCLSemaphore> m_semaphore_cl;
public:
    RGYSemaphoreVulkan();
    virtual ~RGYSemaphoreVulkan();
    virtual RGY_ERR create(DeviceVulkan *vk, RGYOpenCLContext *clctx);
    virtual void release();
    virtual VkSemaphore getVK() { return m_semaphore; }
    virtual RGY_ERR vkWait();
    virtual RGY_ERR vkSignal();
    virtual RGYOpenCLSemaphore *getCL() { return m_semaphore_cl.get(); }
protected:
    virtual RGY_ERR createCL(RGYOpenCLContext *clctx);
};
#endif

#if ENABLE_D3D11
using pl_device = pl_d3d11;
using PLDevice = ID3D11Device;
#define pl_tex_wrap p_d3d11_wrap
using pl_tex_wrap_params = pl_d3d11_wrap_params;
using RGYFrameInteropTexture = RGYFrameD3D11;
using RGYPLInteropDataFormat = DXGI_FORMAT;
static const TCHAR *RGY_LIBPLACEBO_DEV_API = _T("d3d11");
#elif ENABLE_VULKAN
using pl_device = pl_vulkan;
using PLDevice = DeviceVulkan;
#define pl_tex_wrap p_vulkan_wrap
using pl_tex_wrap_params = pl_vulkan_wrap_params;
using RGYFrameInteropTexture = RGYFrameVulkan;
using RGYPLInteropDataFormat = VkFormat;
static const TCHAR *RGY_LIBPLACEBO_DEV_API = _T("vulkan");
#endif

class RGYFilterLibplacebo : public RGYFilter {
public:
    RGYFilterLibplacebo(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterLibplacebo();
    virtual RGY_ERR init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) override;
    virtual void close() override;

    virtual RGY_ERR setFrameParam(const RGYFrameInfo * pInputFrame) { UNREFERENCED_PARAMETER(pInputFrame); return RGY_ERR_NONE; }
    virtual RGY_ERR initCommon(shared_ptr<RGYFilterParam> pParam);
    virtual RGY_ERR checkParam(const RGYFilterParam *param) = 0;
    virtual RGY_ERR setLibplaceboParam(const RGYFilterParam *param) = 0;
    virtual RGY_ERR procPlane(pl_tex texOut, const RGYFrameInfo *pDstPlane, pl_tex texIn, const RGYFrameInfo *pSrcPlane, [[maybe_unused]] const RGY_PLANE planeIdx) = 0;
    virtual RGY_ERR procFrame(pl_tex texOut[RGY_MAX_PLANES], const RGYFrameInfo *pDstFrame, pl_tex texIn[RGY_MAX_PLANES], const RGYFrameInfo *pSrcFrame) = 0;
    virtual RGY_ERR initLibplacebo(const RGYFilterParam *param);
    virtual RGY_CSP getTextureCsp(const RGY_CSP csp);
    virtual RGYPLInteropDataFormat getTextureDataFormat(const RGY_CSP csp);
    int getTextureBytePerPix(const RGYPLInteropDataFormat format) const;
    virtual tstring printParams(const RGYFilterParamLibplacebo * prm) const;
    virtual void setFrameProp(RGYFrameInfo * pFrame, const RGYFrameInfo * pSrcFrame) const { copyFramePropWithoutRes(pFrame, pSrcFrame); }

    bool m_procByFrame;

    RGY_CSP m_textCspIn;
    RGY_CSP m_textCspOut;
    RGYPLInteropDataFormat m_dataformatIn;
    RGYPLInteropDataFormat m_dataformatOut;

    std::unique_ptr<std::remove_pointer<pl_log>::type, RGYLibplaceboDeleter<pl_log>> m_log;
    std::unique_ptr<std::remove_pointer<pl_device>::type, RGYLibplaceboDeleter<pl_device>> m_pldevice;
    std::unique_ptr<std::remove_pointer<pl_dispatch>::type, RGYLibplaceboDeleter<pl_dispatch>> m_dispatch;
    std::unique_ptr<std::remove_pointer<pl_renderer>::type, RGYLibplaceboDeleter<pl_renderer>> m_renderer;
    std::unique_ptr<pl_shader_obj, decltype(&pl_shader_obj_destroy)> m_dither_state;

    std::unique_ptr<RGYCLFrame> m_textFrameBufOut;
    std::unique_ptr<RGYFrameInteropTexture> m_textIn;
    std::unique_ptr<RGYFrameInteropTexture> m_textOut;
#if ENABLE_VULKAN
    std::vector<std::unique_ptr<RGYSemaphoreVulkan>> m_semInVKWait;
    std::vector<std::unique_ptr<RGYSemaphoreVulkan>> m_semInVKStart;
    std::vector<std::unique_ptr<RGYSemaphoreVulkan>> m_semOutVKWait;
    std::vector<std::unique_ptr<RGYSemaphoreVulkan>> m_semOutVKStart;
#endif
    std::unique_ptr<RGYFilter> m_srcCrop;
    std::unique_ptr<RGYFilter> m_dstCrop;
    std::unique_ptr<RGYLibplaceboLoader> m_pl;
    PLDevice *m_device;
};

class RGYFilterLibplaceboResample : public RGYFilterLibplacebo {
public:
    RGYFilterLibplaceboResample(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterLibplaceboResample();
protected:
    virtual RGY_ERR checkParam(const RGYFilterParam *param) override;
    virtual RGY_ERR setLibplaceboParam(const RGYFilterParam *param) override;
    virtual RGY_ERR procPlane(pl_tex texOut, const RGYFrameInfo *pDstPlane, pl_tex texIn, const RGYFrameInfo *pSrcPlane, [[maybe_unused]] const RGY_PLANE planeIdx) override;
    virtual RGY_ERR procFrame(pl_tex texOut[RGY_MAX_PLANES], const RGYFrameInfo *pDstFrame, pl_tex texIn[RGY_MAX_PLANES], const RGYFrameInfo *pSrcFrame) override {
        UNREFERENCED_PARAMETER(texOut); UNREFERENCED_PARAMETER(pDstFrame); UNREFERENCED_PARAMETER(texIn); UNREFERENCED_PARAMETER(pSrcFrame);
        return RGY_ERR_UNSUPPORTED;
    };
    std::unique_ptr<pl_sample_filter_params> m_filter_params;
};

class RGYFilterLibplaceboDeband : public RGYFilterLibplacebo {
public:
    RGYFilterLibplaceboDeband(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterLibplaceboDeband();
protected:
    virtual RGY_ERR checkParam(const RGYFilterParam *param) override;
    virtual RGY_ERR setLibplaceboParam(const RGYFilterParam *param) override;
    virtual RGY_ERR procPlane(pl_tex texOut, const RGYFrameInfo *pDstPlane, pl_tex texIn, const RGYFrameInfo *pSrcPlane, [[maybe_unused]] const RGY_PLANE planeIdx) override;
    virtual RGY_ERR procFrame(pl_tex texOut[RGY_MAX_PLANES], const RGYFrameInfo *pDstFrame, pl_tex texIn[RGY_MAX_PLANES], const RGYFrameInfo *pSrcFrame) override {
        UNREFERENCED_PARAMETER(texOut); UNREFERENCED_PARAMETER(pDstFrame); UNREFERENCED_PARAMETER(texIn); UNREFERENCED_PARAMETER(pSrcFrame);
        return RGY_ERR_UNSUPPORTED;
    };

    std::unique_ptr<pl_deband_params> m_filter_params;
    std::unique_ptr<pl_deband_params> m_filter_params_c;
    std::unique_ptr<pl_dither_params> m_dither_params;
    int m_frame_index;
};

struct RGYLibplaceboToneMappingParams {
    std::unique_ptr<pl_render_params> renderParams;
    VppLibplaceboToneMappingCSP cspSrc;
    VppLibplaceboToneMappingCSP cspDst;
    pl_color_space plCspSrc;
    pl_color_space plCspDst;
    float src_max_org;
    float src_min_org;
    float dst_max_org;
    float dst_min_org;
    bool is_subsampled;
    pl_chroma_location chromaLocation;
    bool use_dovi;
    std::unique_ptr<pl_color_map_params> colorMapParams;
    std::unique_ptr<pl_peak_detect_params> peakDetectParams;
    std::unique_ptr<pl_sigmoid_params> sigmoidParams;
    std::unique_ptr<pl_dither_params> ditherParams;
    std::unique_ptr<pl_dovi_metadata> plDoviMeta;
    std::unique_ptr<pl_color_repr> reprSrc;
    std::unique_ptr<pl_color_repr> reprDst;
    VideoVUIInfo outVui;
};

class RGYFilterLibplaceboToneMapping : public RGYFilterLibplacebo {
public:
    RGYFilterLibplaceboToneMapping(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterLibplaceboToneMapping();
    VideoVUIInfo VuiOut() const;
protected:
    virtual RGY_ERR checkParam(const RGYFilterParam *param) override;
    virtual RGY_ERR setLibplaceboParam(const RGYFilterParam *param) override;
    virtual RGY_ERR setFrameParam(const RGYFrameInfo *pInputFrame) override;
    virtual RGY_ERR procPlane(pl_tex texOut, const RGYFrameInfo *pDstPlane, pl_tex texIn, const RGYFrameInfo *pSrcPlane, const RGY_PLANE planeIdx) override {
        UNREFERENCED_PARAMETER(texOut); UNREFERENCED_PARAMETER(pDstPlane); UNREFERENCED_PARAMETER(texIn); UNREFERENCED_PARAMETER(pSrcPlane); UNREFERENCED_PARAMETER(planeIdx);
        return RGY_ERR_UNSUPPORTED;
    };
    virtual RGY_ERR procFrame(pl_tex texOut[RGY_MAX_PLANES], const RGYFrameInfo *pDstFrame, pl_tex texIn[RGY_MAX_PLANES], const RGYFrameInfo *pSrcFrame) override;

    virtual RGY_CSP getTextureCsp(const RGY_CSP csp) override;
    virtual RGYPLInteropDataFormat getTextureDataFormat(const RGY_CSP csp) override;
    virtual tstring printParams(const RGYFilterParamLibplacebo *prm) const override;
    virtual void setFrameProp(RGYFrameInfo *pFrame, const RGYFrameInfo *pSrcFrame) const override;

    RGYLibplaceboToneMappingParams m_tonemap;
};

class RGYFilterLibplaceboShader : public RGYFilterLibplacebo {
public:
    RGYFilterLibplaceboShader(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterLibplaceboShader();
protected:
    virtual RGY_ERR checkParam(const RGYFilterParam *param) override;
    virtual RGY_ERR setLibplaceboParam(const RGYFilterParam *param) override;
    virtual RGY_ERR procPlane(pl_tex texOut, const RGYFrameInfo *pDstPlane, pl_tex texIn, const RGYFrameInfo *pSrcPlane, const RGY_PLANE planeIdx) override {
        UNREFERENCED_PARAMETER(texOut); UNREFERENCED_PARAMETER(pDstPlane); UNREFERENCED_PARAMETER(texIn); UNREFERENCED_PARAMETER(pSrcPlane); UNREFERENCED_PARAMETER(planeIdx);
        return RGY_ERR_UNSUPPORTED;
    };
    virtual RGY_ERR procFrame(pl_tex texOut[RGY_MAX_PLANES], const RGYFrameInfo *pDstFrame, pl_tex texIn[RGY_MAX_PLANES], const RGYFrameInfo *pSrcFrame) override;

    virtual RGY_CSP getTextureCsp(const RGY_CSP csp) override;
    virtual RGYPLInteropDataFormat getTextureDataFormat(const RGY_CSP csp) override;
    virtual tstring printParams(const RGYFilterParamLibplacebo *prm) const override;

    std::unique_ptr<pl_hook, RGYLibplaceboDeleter<const pl_hook*>> m_shader;
    pl_color_system m_colorsystem;
    pl_color_transfer m_transfer;
    pl_color_levels m_range;
    pl_chroma_location m_chromaloc;
    std::unique_ptr<pl_sample_filter_params> m_sample_params;
    std::unique_ptr<pl_sigmoid_params> m_sigmoid_params;
    int m_linear;
};

#else

class RGYFilterLibplaceboResample : public RGYFilterDisabled {
public:
    RGYFilterLibplaceboResample(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterLibplaceboResample();
};

class RGYFilterLibplaceboDeband : public RGYFilterDisabled {
public:
    RGYFilterLibplaceboDeband(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterLibplaceboDeband();
};

class RGYFilterLibplaceboToneMapping : public RGYFilterDisabled {
public:
    RGYFilterLibplaceboToneMapping(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterLibplaceboToneMapping();
};

class RGYFilterLibplaceboShader : public RGYFilterDisabled {
public:
    RGYFilterLibplaceboShader(shared_ptr<RGYOpenCLContext> context);
    virtual ~RGYFilterLibplaceboShader();
};

#endif // ENABLE_LIBPLACEBO

#endif // __RGY_FILTER_LIBPLACEBO_H__
