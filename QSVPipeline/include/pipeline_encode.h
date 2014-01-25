//* ////////////////////////////////////////////////////////////////////////////// */
//*
//
//              INTEL CORPORATION PROPRIETARY INFORMATION
//  This software is supplied under the terms of a license  agreement or
//  nondisclosure agreement with Intel Corporation and may not be copied
//  or disclosed except in  accordance  with the terms of that agreement.
//        Copyright (c) 2005-2011 Intel Corporation. All Rights Reserved.
//
//
//*/

#ifndef __PIPELINE_ENCODE_H__
#define __PIPELINE_ENCODE_H__

#include <process.h>

#include "qsv_version.h"
#include "qsv_util.h"
#include "qsv_prm.h"

#include "sample_defs.h"

#ifdef D3D_SURFACES_SUPPORT
#pragma warning(disable : 4201)
#include <d3d9.h>
#include <dxva2api.h>
#pragma comment(lib, "d3d9.lib")
#pragma comment(lib, "dxva2.lib")
#include "hw_device.h"
#endif

#include "sample_utils.h"
#include "base_allocator.h"

#include "mfxmvc.h"
#include "mfxvideo.h"
#include "mfxvideo++.h"
#include "mfxplugin++.h"

#include "scene_change_detection.h"

#include <vector>
#include <string>
#include <iostream>

#pragma comment(lib, "libmfx.lib")
#pragma comment(lib, "libmfxmd.lib")


typedef std::basic_string<TCHAR> tstring;
typedef std::basic_stringstream<TCHAR> TStringStream;

struct sTask
{
	mfxBitstream mfxBS;
	mfxSyncPoint EncSyncP;
	std::list<mfxSyncPoint> DependentVppTasks;
	CSmplBitstreamWriter *pWriter;

	sTask();
	mfxStatus WriteBitstream();
	mfxStatus Reset();
	mfxStatus Init(mfxU32 nBufferSize, CSmplBitstreamWriter *pWriter = NULL);
	mfxStatus Close();
};

class CEncTaskPool
{
public:
	CEncTaskPool();
	virtual ~CEncTaskPool();

	virtual mfxStatus Init(MFXVideoSession* pmfxSession, CSmplBitstreamWriter* pWriter, mfxU32 nPoolSize, mfxU32 nBufferSize, CSmplBitstreamWriter *pOtherWriter = NULL);

	mfxStatus GetFreeTask(sTask **ppTask);

	virtual mfxStatus SynchronizeFirstTask();
	virtual void Close();

protected:
	sTask* m_pTasks;
	mfxU32 m_nPoolSize;
	mfxU32 m_nTaskBufferStart;

	MFXVideoSession* m_pmfxSession;

	virtual mfxU32 GetFreeTaskIndex();
};

enum {
	MFX_PRM_EX_SCENE_CHANGE = 0x01,
	MFX_PRM_EX_VQP          = 0x02,
	MFX_PRM_EX_DEINT_NORMAL = 0x04,
	MFX_PRM_EX_DEINT_BOB    = 0x08
};
enum {
	SC_FIELDFLAG_INVALID_ALL  = 0xffffffff,
	SC_FIELDFLAG_INVALID_LOW  = 0x0000ffff,
	SC_FIELDFLAG_INVALID_HIGH = 0xffff0000,
};

/* This class implements a pipeline with 2 mfx components: vpp (video preprocessing) and encode */
class CEncodingPipeline
{
public:
	CEncodingPipeline();
	virtual ~CEncodingPipeline();

	virtual mfxStatus CheckParam(sInputParams *pParams);
	virtual mfxStatus Init(sInputParams *pParams);
	virtual mfxStatus Run();
	virtual mfxStatus Run(DWORD_PTR SubThreadAffinityMask);
	virtual void Close();
	virtual mfxStatus ResetMFXComponents(sInputParams* pParams);
	virtual mfxStatus ResetDevice();
#if ENABLE_MVC_ENCODING
	void SetMultiView();
	void SetNumView(mfxU32 numViews) { m_nNumView = numViews; }
#endif
	virtual mfxStatus CheckCurrentVideoParam();

	virtual void PrintMes(const TCHAR *format, ... );

	virtual void SetAbortFlagPointer(bool *abort);

protected:
	virtual mfxStatus RunEncode();
	mfxStatus CheckSceneChange();
	static unsigned int __stdcall RunEncThreadLauncher(void *pParam);
	static unsigned int __stdcall RunSubThreadLauncher(void *pParam);
	mfxVersion m_mfxVer;
	CEncodeStatusInfo *m_pEncSatusInfo;
	CEncodingThread m_EncThread;

	CSceneChangeDetect m_SceneChange;
	mfxU32 m_nExPrm;
	CQSVFrameTypeSimulation m_frameTypeSim;

	TCHAR *m_pStrLog;

	CSmplBitstreamWriter *m_pFileWriter;
	CSmplYUVReader *m_pFileReader;

	CEncTaskPool m_TaskPool;
	mfxU16 m_nAsyncDepth; // depth of asynchronous pipeline, this number can be tuned to achieve better performance

	mfxExtVideoSignalInfo m_mfxVSI;
	mfxExtCodingOption m_mfxCopt;
	mfxExtCodingOption2 m_mfxCopt2;
	MFXVideoSession m_mfxSession;
	MFXVideoENCODE* m_pmfxENC;
	MFXVideoVPP* m_pmfxVPP;

	mfxVideoParam m_mfxEncParams;
	mfxVideoParam m_mfxVppParams;


	std::vector<mfxExtBuffer*> m_EncExtParams;
	std::vector<mfxExtBuffer*> m_VppExtParams;
	tstring VppExtMes;

	mfxExtVPPDoNotUse m_ExtDoNotUse;
	mfxExtVPPDoNotUse m_ExtDoUse;
	mfxExtVPPDenoise m_ExtDenoise;
	mfxExtVPPDetail m_ExtDetail;
	mfxExtVPPFrameRateConversion m_ExtFrameRateConv;
	std::vector<mfxU32> m_VppDoNotUseList;
	std::vector<mfxU32> m_VppDoUseList;

	MFXFrameAllocator* m_pMFXAllocator;
	mfxAllocatorParams* m_pmfxAllocatorParams;
	MemType m_memType;
	bool m_bd3dAlloc; // use d3d surfaces
	bool m_bExternalAlloc; // use memory allocator as external for Media SDK

	bool m_bHaswellOrLater;

	bool *m_pAbortByUser;

	mfxFrameSurface1* m_pEncSurfaces; // frames array for encoder input (vpp output)
	mfxFrameSurface1* m_pVppSurfaces; // frames array for vpp input
	mfxFrameAllocResponse m_EncResponse;  // memory allocation response for encoder
	mfxFrameAllocResponse m_VppResponse;  // memory allocation response for vpp

#if ENABLE_MVC_ENCODING
	mfxU16 m_MVCflags; // MVC codec is in use
	mfxU32 m_nNumView;
	// for MVC encoder and VPP configuration
	mfxExtMVCSeqDesc m_MVCSeqDesc;
#endif
	// for disabling VPP algorithms
	//mfxExtVPPDoNotUse m_VppDoNotUse;

#if D3D_SURFACES_SUPPORT
	CHWDevice *m_hwdev;
#endif
	virtual mfxStatus DetermineMinimumRequiredVersion(const sInputParams &pParams, mfxVersion &version);

	virtual mfxStatus InitInOut(sInputParams *pParams);
	virtual mfxStatus InitMfxEncParams(sInputParams *pParams);
	virtual mfxStatus InitMfxVppParams(sInputParams *pParams);
	virtual mfxStatus InitSession(bool useHWLib, mfxU16 memType);
	//virtual void InitVppExtParam();
	virtual mfxStatus CreateVppExtBuffers(sInputParams *pParams);

	//virtual mfxStatus AllocAndInitVppDoNotUse();
	//virtual void FreeVppDoNotUse();
#if ENABLE_MVC_ENCODING
	virtual mfxStatus AllocAndInitMVCSeqDesc();
	virtual void FreeMVCSeqDesc();
#endif
	virtual mfxStatus CreateAllocator();
	virtual void DeleteAllocator();

	virtual mfxStatus CreateHWDevice();
	virtual void DeleteHWDevice();

	virtual mfxStatus AllocFrames();
	virtual void DeleteFrames();

	virtual mfxStatus AllocateSufficientBuffer(mfxBitstream* pBS);

	virtual mfxStatus GetFreeTask(sTask **ppTask);
	virtual mfxStatus SynchronizeFirstTask();
};

#endif // __PIPELINE_ENCODE_H__
