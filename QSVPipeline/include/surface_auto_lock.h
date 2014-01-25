//* ////////////////////////////////////////////////////////////////////////////// */
//*
//
//              INTEL CORPORATION PROPRIETARY INFORMATION
//  This software is supplied under the terms of a license  agreement or
//  nondisclosure agreement with Intel Corporation and may not be copied
//  or disclosed except in  accordance  with the terms of that agreement.
//        Copyright (c) 2013 Intel Corporation. All Rights Reserved.
//
//
//*/

#pragma once

#include "mfxstructures.h"
#include "sample_utils.h"

/*
    Rationale: locks allocator if necessary to get RAW pointers, unlock it at the end
*/
class SurfaceAutoLock : private no_copy
{
public:
    SurfaceAutoLock(mfxFrameAllocator & alloc, mfxFrameSurface1 &srf)
        : m_alloc(alloc) , m_srf(srf), m_lockRes(MFX_ERR_NONE), m_bLocked() {
        LockFrame();
    }
    operator mfxStatus () {
        return m_lockRes;
    }
    ~SurfaceAutoLock() {
        UnlockFrame();
    }

protected:

    mfxFrameAllocator & m_alloc;
    mfxFrameSurface1 & m_srf;
    mfxStatus m_lockRes;
    bool m_bLocked;

    void LockFrame()
    {
        //no allocator used, no need to do lock
        if (m_srf.Data.Y != 0)
            return ;
        //lock required
        m_lockRes = m_alloc.Lock(m_alloc.pthis, m_srf.Data.MemId, &m_srf.Data);
        if (m_lockRes == MFX_ERR_NONE) {
            m_bLocked = true;
        }
    }

    void UnlockFrame()
    {
        if (m_lockRes != MFX_ERR_NONE || !m_bLocked) {
            return;
        }
        //unlock required
        m_alloc.Unlock(m_alloc.pthis, m_srf.Data.MemId, &m_srf.Data);
    }
};