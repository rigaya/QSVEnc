/*********************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2014 Intel Corporation. All Rights Reserved.

**********************************************************************************/

#include "mfx_samples_config.h"

#include <stdlib.h>

#include <mfx_buffering.h>

CBuffering::CBuffering():
    m_SurfacesNumber(0),
    m_OutputSurfacesNumber(0),
    m_pSurfaces(NULL),
    m_pVppSurfaces(NULL),
    m_FreeSurfacesPool(&m_Mutex),
    m_FreeVppSurfacesPool(&m_Mutex),
    m_UsedSurfacesPool(&m_Mutex),
    m_UsedVppSurfacesPool(&m_Mutex),
    m_pFreeOutputSurfaces(NULL),
    m_OutputSurfacesPool(&m_Mutex),
    m_DeliveredSurfacesPool(&m_Mutex)
{
}

CBuffering::~CBuffering()
{
}

mfxStatus
CBuffering::AllocBuffers(mfxU32 SurfaceNumber)
{
    if (!SurfaceNumber) return MFX_ERR_MEMORY_ALLOC;

    if (!m_OutputSurfacesNumber) { // true - if Vpp isn't enabled
        m_OutputSurfacesNumber = SurfaceNumber;
    }
    m_SurfacesNumber = SurfaceNumber;

    m_pSurfaces = (msdkFrameSurface*)calloc(m_SurfacesNumber, sizeof(msdkFrameSurface));
    if (!m_pSurfaces) return MFX_ERR_MEMORY_ALLOC;

    msdkOutputSurface* p = NULL;
    msdkOutputSurface* tail = NULL;

    m_pFreeOutputSurfaces = (msdkOutputSurface*)calloc(1, sizeof(msdkOutputSurface));
    if (!m_pFreeOutputSurfaces) return MFX_ERR_MEMORY_ALLOC;

    tail = m_pFreeOutputSurfaces;

    for (mfxU32 i = 1; i < m_OutputSurfacesNumber; ++i) {
        p = (msdkOutputSurface*)calloc(1, sizeof(msdkOutputSurface));
        if (!p) return MFX_ERR_MEMORY_ALLOC;
        tail->next = p;
        tail = p;
    }

    ResetBuffers();
    return MFX_ERR_NONE;
}

mfxStatus
CBuffering::AllocVppBuffers(mfxU32 VppSurfaceNumber)
{
    m_OutputSurfacesNumber = VppSurfaceNumber;
    m_pVppSurfaces = (msdkFrameSurface*)calloc(m_OutputSurfacesNumber, sizeof(msdkFrameSurface));
    if (!m_pVppSurfaces) return MFX_ERR_MEMORY_ALLOC;

    ResetVppBuffers();
    return MFX_ERR_NONE;
}

void
CBuffering::AllocOutputBuffer()
{
    AutomaticMutex lock(m_Mutex);

    m_pFreeOutputSurfaces = (msdkOutputSurface*)calloc(1, sizeof(msdkOutputSurface));
}

static void
FreeList(msdkOutputSurface*& head) {
    msdkOutputSurface* next;
    while (head) {
        next = head->next;
        free(head);
        head = next;
    }
}

void
CBuffering::FreeBuffers()
{
    if (m_pSurfaces) {
        free(m_pSurfaces);
        m_pSurfaces = NULL;
    }

    if (m_pVppSurfaces) {
        free(m_pVppSurfaces);
        m_pVppSurfaces = NULL;
    }

    FreeList(m_pFreeOutputSurfaces);
    FreeList(m_OutputSurfacesPool.m_pSurfacesHead);
    FreeList(m_DeliveredSurfacesPool.m_pSurfacesHead);

    m_UsedSurfacesPool.m_pSurfacesHead = NULL;
    m_UsedSurfacesPool.m_pSurfacesTail = NULL;
    m_UsedVppSurfacesPool.m_pSurfacesHead = NULL;
    m_UsedVppSurfacesPool.m_pSurfacesTail = NULL;

    m_FreeSurfacesPool.m_pSurfaces = NULL;
    m_FreeVppSurfacesPool.m_pSurfaces = NULL;
}

void
CBuffering::ResetBuffers()
{
    mfxU32 i;
    msdkFrameSurface* pFreeSurf = m_FreeSurfacesPool.m_pSurfaces = m_pSurfaces;

    for (i = 0; i < m_SurfacesNumber; ++i) {
        if (i < (m_SurfacesNumber-1)) {
            pFreeSurf[i].next = &(pFreeSurf[i+1]);
            pFreeSurf[i+1].prev = &(pFreeSurf[i]);
        }
    }
}

void
CBuffering::ResetVppBuffers()
{
    mfxU32 i;
    msdkFrameSurface* pFreeVppSurf = m_FreeVppSurfacesPool.m_pSurfaces = m_pVppSurfaces;

    for (i = 0; i < m_OutputSurfacesNumber; ++i) {
        if (i < (m_OutputSurfacesNumber-1)) {
            pFreeVppSurf[i].next = &(pFreeVppSurf[i+1]);
            pFreeVppSurf[i+1].prev = &(pFreeVppSurf[i]);
        }
    }
}

void
CBuffering::SyncFrameSurfaces()
{
    AutomaticMutex lock(m_Mutex);
    msdkFrameSurface *prev;
    msdkFrameSurface *next;
    prev = next = NULL;
    msdkFrameSurface *cur = m_UsedSurfacesPool.m_pSurfacesHead;

    while (cur) {
        if (cur->frame.Data.Locked || cur->render_lock) {
            // frame is still locked: just moving to the next one
            cur = cur->next;
        } else {
            // frame was unlocked: moving it to the free surfaces array
            m_UsedSurfacesPool.DetachSurfaceUnsafe(cur);
            m_FreeSurfacesPool.AddSurfaceUnsafe(cur);

            cur = next;
        }
    }
}

void
CBuffering::SyncVppFrameSurfaces()
{
    AutomaticMutex lock(m_Mutex);
    msdkFrameSurface *prev;
    msdkFrameSurface *next;
    prev = next = NULL;
    msdkFrameSurface *cur = m_UsedVppSurfacesPool.m_pSurfacesHead;

    while (cur) {
        if (cur->frame.Data.Locked || cur->render_lock) {
            // frame is still locked: just moving to the next one
            cur = cur->next;
        } else {
            // frame was unlocked: moving it to the free surfaces array
            m_UsedVppSurfacesPool.DetachSurfaceUnsafe(cur);
            m_FreeVppSurfacesPool.AddSurfaceUnsafe(cur);

            cur = next;
        }
    }
}
