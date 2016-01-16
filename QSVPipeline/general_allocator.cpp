/*********************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2010-2014 Intel Corporation. All Rights Reserved.

**********************************************************************************/

#include "mfx_samples_config.h"

#include "general_allocator.h"

#if defined(_WIN32) || defined(_WIN64)
#include "d3d_allocator.h"
#include "d3d11_allocator.h"
#else
#include <stdarg.h>
#include "vaapi_allocator.h"
#endif

#include "sysmem_allocator.h"

#include "sample_defs.h"

// Wrapper on standard allocator for concurrent allocation of
// D3D and system surfaces
GeneralAllocator::GeneralAllocator()
{
};
GeneralAllocator::~GeneralAllocator()
{
};
mfxStatus GeneralAllocator::Init(mfxAllocatorParams *pParams)
{
    mfxStatus sts = MFX_ERR_NONE;

#if defined(_WIN32) || defined(_WIN64)
#if MFX_D3D11_SUPPORT
    D3D11AllocatorParams *d3d11AllocParams = dynamic_cast<D3D11AllocatorParams*>(pParams);
    if (d3d11AllocParams)
        m_D3DAllocator.reset(new D3D11FrameAllocator);
    else
#endif
        m_D3DAllocator.reset(new D3DFrameAllocator);
#endif
#ifdef LIBVA_SUPPORT
    m_D3DAllocator.reset(new vaapiFrameAllocator);
#endif

    m_SYSAllocator.reset(new SysMemFrameAllocator);

    if (m_D3DAllocator.get())
    {
        sts = m_D3DAllocator.get()->Init(pParams);
        MSDK_CHECK_RESULT(MFX_ERR_NONE, sts, sts);
    }

    sts = m_SYSAllocator.get()->Init(0);
    MSDK_CHECK_RESULT(MFX_ERR_NONE, sts, sts);

    return sts;
}
mfxStatus GeneralAllocator::Close()
{
    mfxStatus sts = MFX_ERR_NONE;
    if (m_D3DAllocator.get())
    {
        sts = m_D3DAllocator.get()->Close();
        MSDK_CHECK_RESULT(MFX_ERR_NONE, sts, sts);
    }

    sts = m_SYSAllocator.get()->Close();
    MSDK_CHECK_RESULT(MFX_ERR_NONE, sts, sts);

   return sts;
}

mfxStatus GeneralAllocator::LockFrame(mfxMemId mid, mfxFrameData *ptr)
{
    if (isD3DMid(mid) && m_D3DAllocator.get())
        return m_D3DAllocator.get()->Lock(m_D3DAllocator.get(), mid, ptr);
    else
        return m_SYSAllocator.get()->Lock(m_SYSAllocator.get(),mid, ptr);
}
mfxStatus GeneralAllocator::UnlockFrame(mfxMemId mid, mfxFrameData *ptr)
{
    if (isD3DMid(mid) && m_D3DAllocator.get())
        return m_D3DAllocator.get()->Unlock(m_D3DAllocator.get(), mid, ptr);
    else
        return m_SYSAllocator.get()->Unlock(m_SYSAllocator.get(),mid, ptr);
}

mfxStatus GeneralAllocator::GetFrameHDL(mfxMemId mid, mfxHDL *handle)
{
    if (isD3DMid(mid) && m_D3DAllocator.get())
        return m_D3DAllocator.get()->GetHDL(m_D3DAllocator.get(), mid, handle);
    else
        return m_SYSAllocator.get()->GetHDL(m_SYSAllocator.get(), mid, handle);
}

mfxStatus GeneralAllocator::ReleaseResponse(mfxFrameAllocResponse *response)
{
    // try to ReleaseResponse via D3D allocator
    if (isD3DMid(response->mids[0]) && m_D3DAllocator.get())
        return m_D3DAllocator.get()->Free(m_D3DAllocator.get(),response);
    else
        return m_SYSAllocator.get()->Free(m_SYSAllocator.get(), response);
}
mfxStatus GeneralAllocator::AllocImpl(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response)
{
    mfxStatus sts;
    if ((request->Type & MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET || request->Type & MFX_MEMTYPE_VIDEO_MEMORY_PROCESSOR_TARGET) && m_D3DAllocator.get())
    {
        sts = m_D3DAllocator.get()->Alloc(m_D3DAllocator.get(), request, response);
        StoreFrameMids(true, response);
    }
    else
    {
        sts = m_SYSAllocator.get()->Alloc(m_SYSAllocator.get(), request, response);
        StoreFrameMids(false, response);
    }
    return sts;
}
void    GeneralAllocator::StoreFrameMids(bool isD3DFrames, mfxFrameAllocResponse *response)
{
    for (mfxU32 i = 0; i < response->NumFrameActual; i++)
        m_Mids.insert(std::pair<mfxHDL, bool>(response->mids[i], isD3DFrames));
}
bool GeneralAllocator::isD3DMid(mfxHDL mid)
{
    std::map<mfxHDL, bool>::iterator it;
    it = m_Mids.find(mid);
    if (it == m_Mids.end())
        return false; // sys mem allocator will check validity of mid further
    else
        return it->second;
}
