/* ////////////////////////////////////////////////////////////////////////////// */
/*
//
//              INTEL CORPORATION PROPRIETARY INFORMATION
//  This software is supplied under the terms of a license  agreement or
//  nondisclosure agreement with Intel Corporation and may not be copied
//  or disclosed except in  accordance  with the terms of that agreement.
//        Copyright (c) 2010-2014 Intel Corporation. All Rights Reserved.
//
//
*/

#ifndef __GENERAL_ALLOCATOR_H__
#define __GENERAL_ALLOCATOR_H__

#include "sample_utils.h"
#include "base_allocator.h"

#include <memory>
#include <map>

class SysMemFrameAllocator;

// Wrapper on standard allocator for concurrent allocation of
// D3D and system surfaces
class GeneralAllocator : public BaseFrameAllocator
{
public:
    GeneralAllocator();
    virtual ~GeneralAllocator();

    virtual mfxStatus Init(mfxAllocatorParams *pParams);
    virtual mfxStatus Close();

protected:
    virtual mfxStatus LockFrame(mfxMemId mid, mfxFrameData *ptr);
    virtual mfxStatus UnlockFrame(mfxMemId mid, mfxFrameData *ptr);
    virtual mfxStatus GetFrameHDL(mfxMemId mid, mfxHDL *handle);

    virtual mfxStatus ReleaseResponse(mfxFrameAllocResponse *response);
    virtual mfxStatus AllocImpl(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response);

    void    StoreFrameMids(bool isD3DFrames, mfxFrameAllocResponse *response);
    bool    isD3DMid(mfxHDL mid);

    std::map<mfxHDL, bool>                  m_Mids;
    std::auto_ptr<BaseFrameAllocator>       m_D3DAllocator;
    std::auto_ptr<SysMemFrameAllocator>     m_SYSAllocator;
private:
    DISALLOW_COPY_AND_ASSIGN(GeneralAllocator);

};

#endif //__GENERAL_ALLOCATOR_H__
