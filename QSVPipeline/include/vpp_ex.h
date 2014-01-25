/* ****************************************************************************** *\

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2008-2012 Intel Corporation. All Rights Reserved.

\* ****************************************************************************** */

#ifndef __VPP_EX_H__
#define __VPP_EX_H__

#include "sample_utils.h"
#include "mfxvideo++.h"
#include <vector>

/* #define USE_VPP_EX */

class MFXVideoVPPEx : public MFXVideoVPP
{

public:
    MFXVideoVPPEx(mfxSession session);

#if defined USE_VPP_EX

    mfxStatus QueryIOSurf(mfxVideoParam *par, mfxFrameAllocRequest request[2]);
    mfxStatus Query(mfxVideoParam *in, mfxVideoParam *out);
    mfxStatus Init(mfxVideoParam *par);
    mfxStatus RunFrameVPPAsync(mfxFrameSurface1 *in, mfxFrameSurface1 *out, mfxExtVppAuxData *aux, mfxSyncPoint *syncp);
    mfxStatus GetVideoParam(mfxVideoParam *par);
    mfxStatus Close(void);

protected:

    std::vector<mfxFrameSurface1*>  m_LockedSurfacesList;
    mfxVideoParam                   m_VideoParams;

    mfxU64                          m_nCurrentPTS;

    mfxU64                          m_nIncreaseTime;
    mfxU64                          m_nArraySize;
    mfxU64                          m_nInputTimeStamp;

#endif

};

#endif //__VPP_EX_H__
