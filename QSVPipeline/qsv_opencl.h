﻿// -----------------------------------------------------------------------------------------
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

#endif //__QSV_OPENCL_H__
