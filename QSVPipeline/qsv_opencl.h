// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2021 rigaya
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

#pragma once
#ifndef __QSV_OPENCL_H__
#define __QSV_OPENCL_H__

#include "rgy_opencl.h"
#include "qsv_allocator.h"
#include "qsv_allocator_d3d9.h"
#include "qsv_allocator_d3d11.h"

static std::unique_ptr<RGYCLFrameInterop> getOpenCLFrameInterop(mfxFrameSurface1 *mfxSurf, MemType memType, cl_mem_flags flags, QSVAllocator *allocator, RGYOpenCLContext *cl, RGYOpenCLQueue& queue, const RGYFrameInfo& frameInfo) {
    mfxMemId mid = mfxSurf->Data.MemId;
#if ENABLE_RGY_OPENCL_D3D11
    if (memType == D3D11_MEMORY) {
        mfxHDLPair mid_pair = { 0 };
        auto err = err_to_rgy(allocator->GetHDL(allocator->pthis, mid, reinterpret_cast<mfxHDL*>(&mid_pair)));
        if (err != RGY_ERR_NONE) {
            return std::unique_ptr<RGYCLFrameInterop>();
        }
        ID3D11Texture2D *surf = (ID3D11Texture2D*)mid_pair.first;
        return cl->createFrameFromD3D11Surface(surf, frameInfo, queue, flags);
    } else
#endif
#if ENABLE_RGY_OPENCL_D3D9
    if (memType == D3D9_MEMORY) {
        //mfxHDLPair mid_pair = { 0 };
        //auto err = err_to_rgy(allocator->GetHDL(allocator->pthis, mid, reinterpret_cast<mfxHDL*>(&mid_pair)));
        //if (err != RGY_ERR_NONE) {
        //    return std::unique_ptr<RGYCLFrameInterop>();
        //}
        IDirect3DSurface9 *surf = (IDirect3DSurface9*)((mfxHDLPair*)mid)->first;
        // このshared_handleも渡さないと、release/acquireで余計なオーバーヘッドが発生してしまう模様
        HANDLE shared_handle = (HANDLE)((mfxHDLPair*)mid)->second;
        return cl->createFrameFromD3D9Surface(surf, shared_handle, frameInfo, queue, flags);
    } else
#endif
#if ENABLE_RGY_OPENCL_VA
    if (memType == VA_MEMORY) {
        VASurfaceID* surf = NULL;
        auto err = err_to_rgy(allocator->GetHDL(allocator->pthis, mid, reinterpret_cast<mfxHDL*>(&surf)));
        if (err != RGY_ERR_NONE) {
            return std::unique_ptr<RGYCLFrameInterop>();
        }
        return cl->createFrameFromVASurface(surf, frameInfo, queue, flags);
    } else
#endif
    {
        return std::unique_ptr<RGYCLFrameInterop>();
    }
}

#if (defined(_WIN32) || defined(_WIN64)) && ENABLE_RGY_OPENCL_D3D11
class QSVOpenCLInputCopy {
    struct D3D11ObjectDeleter {
        void operator()(IUnknown *object) const {
            if (object != nullptr) {
                object->Release();
            }
        }
    };

public:
    QSVOpenCLInputCopy() = default;
    ~QSVOpenCLInputCopy() {
        waitForRelease();
        if (m_interop) {
            RGYOpenCLEvent releaseEvent;
            if (m_interop->release(&releaseEvent) == RGY_ERR_NONE && releaseEvent() != nullptr) {
                releaseEvent.wait();
            }
        }
    }
    QSVOpenCLInputCopy(const QSVOpenCLInputCopy&) = delete;
    QSVOpenCLInputCopy& operator=(const QSVOpenCLInputCopy&) = delete;

    RGY_ERR prepare(mfxFrameSurface1 *surface, QSVAllocator *allocator, RGYOpenCLContext *cl,
        RGYOpenCLQueue& queue, const RGYFrameInfo& frameInfo) {
        if (surface == nullptr || allocator == nullptr || cl == nullptr) {
            return RGY_ERR_NULL_PTR;
        }
        if (m_error != RGY_ERR_NONE) {
            return m_error;
        }
        if (m_inUse) {
            if (m_releaseEvent() == nullptr) {
                m_error = RGY_ERR_INVALID_CALL;
                return m_error;
            }
            if (const auto err = waitForRelease(); err != RGY_ERR_NONE) {
                m_error = err;
                return m_error;
            }
        }

        auto allocatorD3D11 = dynamic_cast<QSVAllocatorD3D11 *>(allocator);
        if (allocatorD3D11 == nullptr || allocatorD3D11->GetD3D11Device() == nullptr) {
            return RGY_ERR_INVALID_DEVICE;
        }

        mfxHDLPair sourceHandle = { nullptr, nullptr };
        auto err = err_to_rgy(allocator->GetHDL(allocator->pthis, surface->Data.MemId,
            reinterpret_cast<mfxHDL *>(&sourceHandle)));
        if (err != RGY_ERR_NONE || sourceHandle.first == nullptr) {
            return (err != RGY_ERR_NONE) ? err : RGY_ERR_INVALID_HANDLE;
        }
        auto sourceTexture = std::unique_ptr<ID3D11Texture2D, D3D11ObjectDeleter>(
            reinterpret_cast<ID3D11Texture2D *>(sourceHandle.first));
        sourceTexture->AddRef();

        D3D11_TEXTURE2D_DESC sourceDesc = {};
        sourceTexture->GetDesc(&sourceDesc);
        const auto sourceSubresource = static_cast<UINT>(reinterpret_cast<UINT_PTR>(sourceHandle.second));
        if (sourceDesc.MipLevels == 0 || sourceSubresource >= sourceDesc.MipLevels * sourceDesc.ArraySize) {
            return RGY_ERR_INVALID_HANDLE;
        }
        const auto sourceMip = sourceSubresource % sourceDesc.MipLevels;
        D3D11_TEXTURE2D_DESC copyDesc = sourceDesc;
        copyDesc.Width = (std::max)(1u, sourceDesc.Width >> sourceMip);
        copyDesc.Height = (std::max)(1u, sourceDesc.Height >> sourceMip);
        copyDesc.MipLevels = 1;
        copyDesc.ArraySize = 1;
        copyDesc.Usage = D3D11_USAGE_DEFAULT;
        copyDesc.BindFlags = 0;
        copyDesc.CPUAccessFlags = 0;
        copyDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

        const bool recreate = !m_texture || !m_copyQuery || !m_interop
            || !sameDesc(m_copyDesc, copyDesc)
            || m_frameWidth != frameInfo.width || m_frameHeight != frameInfo.height || m_frameCsp != frameInfo.csp
            || m_device != allocatorD3D11->GetD3D11Device() || m_cl != cl
            || m_queue != &queue || m_commandQueue != queue.get();
        if (recreate) {
            ID3D11Texture2D *copyTexture = nullptr;
            auto hr = allocatorD3D11->GetD3D11Device()->CreateTexture2D(&copyDesc, nullptr, &copyTexture);
            if (FAILED(hr)) {
                return RGY_ERR_MEMORY_ALLOC;
            }
            std::unique_ptr<ID3D11Texture2D, D3D11ObjectDeleter> texture(copyTexture);

            D3D11_QUERY_DESC queryDesc = {};
            queryDesc.Query = D3D11_QUERY_EVENT;
            ID3D11Query *copyQuery = nullptr;
            hr = allocatorD3D11->GetD3D11Device()->CreateQuery(&queryDesc, &copyQuery);
            if (FAILED(hr)) {
                return RGY_ERR_DEVICE_FAILED;
            }
            std::unique_ptr<ID3D11Query, D3D11ObjectDeleter> query(copyQuery);

            auto interop = cl->createFrameFromD3D11Surface(texture.get(), frameInfo, queue, CL_MEM_READ_ONLY);
            if (!interop) {
                return RGY_ERR_MEMORY_ALLOC;
            }

            // interopを先に破棄してから、その元になったD3D11テクスチャを解放する。
            m_interop = std::move(interop);
            m_texture = std::move(texture);
            m_copyQuery = std::move(query);
            m_copyDesc = copyDesc;
            m_frameWidth = frameInfo.width;
            m_frameHeight = frameInfo.height;
            m_frameCsp = frameInfo.csp;
            m_device = allocatorD3D11->GetD3D11Device();
            m_cl = cl;
            m_queue = &queue;
            m_commandQueue = queue.get();
        }

        // エラー時もコピー元を保持し、完了を確認できないslotを再利用しない。
        m_copySource = std::move(sourceTexture);
        err = err_to_rgy(allocatorD3D11->CopyFrameSurfaceToTexture(
            surface->Data.MemId, m_texture.get(), m_copyQuery.get()));
        if (err != RGY_ERR_NONE) {
            m_error = err;
            return m_error;
        }
        m_copySource.reset();
        m_inUse = true;
        return RGY_ERR_NONE;
    }

    RGYCLFrameInterop *interop() const {
        return m_interop.get();
    }

    void setReleaseEvent(const RGYOpenCLEvent& event) {
        m_releaseEvent = event;
    }

private:
    static bool sameDesc(const D3D11_TEXTURE2D_DESC& lhs, const D3D11_TEXTURE2D_DESC& rhs) {
        return lhs.Width == rhs.Width && lhs.Height == rhs.Height
            && lhs.MipLevels == rhs.MipLevels && lhs.ArraySize == rhs.ArraySize
            && lhs.Format == rhs.Format
            && lhs.SampleDesc.Count == rhs.SampleDesc.Count && lhs.SampleDesc.Quality == rhs.SampleDesc.Quality
            && lhs.Usage == rhs.Usage && lhs.BindFlags == rhs.BindFlags
            && lhs.CPUAccessFlags == rhs.CPUAccessFlags && lhs.MiscFlags == rhs.MiscFlags;
    }

    RGY_ERR waitForRelease() {
        if (m_releaseEvent() != nullptr) {
            const auto err = m_releaseEvent.wait();
            if (err != RGY_ERR_NONE) {
                return err;
            }
            m_releaseEvent.reset();
        }
        m_inUse = false;
        return RGY_ERR_NONE;
    }

    // 宣言の逆順で破棄されるため、interopはtextureより先に解放される。
    std::unique_ptr<ID3D11Texture2D, D3D11ObjectDeleter> m_texture;
    std::unique_ptr<ID3D11Query, D3D11ObjectDeleter> m_copyQuery;
    std::unique_ptr<RGYCLFrameInterop> m_interop;
    std::unique_ptr<ID3D11Texture2D, D3D11ObjectDeleter> m_copySource;
    RGYOpenCLEvent m_releaseEvent;
    D3D11_TEXTURE2D_DESC m_copyDesc = {};
    int m_frameWidth = 0;
    int m_frameHeight = 0;
    RGY_CSP m_frameCsp = RGY_CSP_NA;
    ID3D11Device *m_device = nullptr;
    RGYOpenCLContext *m_cl = nullptr;
    RGYOpenCLQueue *m_queue = nullptr;
    cl_command_queue m_commandQueue = nullptr;
    RGY_ERR m_error = RGY_ERR_NONE;
    bool m_inUse = false;
};
#endif

#endif //__QSV_OPENCL_H__
