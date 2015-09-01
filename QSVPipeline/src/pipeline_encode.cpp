/*********************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2005-2014 Intel Corporation. All Rights Reserved.

**********************************************************************************/

#include <tchar.h>
#include <windows.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <process.h>
#include <sstream>
#include <algorithm>
#include <cassert>

#include "mfx_samples_config.h"
#include "pipeline_encode.h"
#include "vpy_reader.h"
#include "avs_reader.h"
#include "avi_reader.h"
#include "avcodec_reader.h"
#include "avcodec_writer.h"
#include "sysmem_allocator.h"

#include "plugin_loader.h"

#if D3D_SURFACES_SUPPORT
#include "d3d_allocator.h"
#include "d3d11_allocator.h"

#include "d3d_device.h"
#include "d3d11_device.h"
#endif

#ifdef LIBVA_SUPPORT
#include "vaapi_allocator.h"
#include "vaapi_device.h"
#endif

//#include "../../sample_user_modules/plugin_api/plugin_loader.h"

#define MSDK_CHECK_RESULT_MES(P, X, ERR, MES)    {if ((X) > (P)) {PrintMes(QSV_LOG_ERROR, _T("%s : %s\n"), MES, get_err_mes((int)P)); MSDK_PRINT_RET_MSG(ERR); return ERR;}}

#if 0
/* obtain the clock tick of an uninterrupted master clock */
msdk_tick time_get_tick(void)
{
    return msdk_time_get_tick();/*
    LARGE_INTEGER t1;

    QueryPerformanceCounter(&t1);
    return t1.QuadPart;*/

} /* vm_tick vm_time_get_tick(void) */

/* obtain the clock resolution */
msdk_tick time_get_frequency(void)
{
    return msdk_time_get_frequency();/*
    LARGE_INTEGER t1;

    QueryPerformanceFrequency(&t1);
    return t1.QuadPart;*/

} /* vm_tick vm_time_get_frequency(void) */
#endif

CEncTaskPool::CEncTaskPool()
{
    m_pTasks            = nullptr;
    m_pmfxSession       = nullptr;
    m_nTaskBufferStart  = 0;
    m_nPoolSize         = 0;
}

CEncTaskPool::~CEncTaskPool()
{
    Close();
}

mfxStatus CEncTaskPool::Init(MFXVideoSession* pmfxSession, MFXFrameAllocator *pmfxAllocator, CSmplBitstreamWriter* pBitstreamWriter, CSmplYUVWriter *pYUVWriter, mfxU32 nPoolSize, mfxU32 nBufferSize, CSmplBitstreamWriter *pOtherWriter)
{
    MSDK_CHECK_POINTER(pmfxSession, MFX_ERR_NULL_PTR);

    MSDK_CHECK_ERROR(nPoolSize, 0, MFX_ERR_UNDEFINED_BEHAVIOR);
    if (pBitstreamWriter) {
        MSDK_CHECK_ERROR(nBufferSize, 0, MFX_ERR_UNDEFINED_BEHAVIOR);
    } else {
        //フレーム出力時には、Allocatorも必要
        MSDK_CHECK_POINTER(pmfxAllocator, MFX_ERR_NULL_PTR);
    }

    // nPoolSize must be even in case of 2 output bitstreams
    if (pOtherWriter && (0 != nPoolSize % 2))
        return MFX_ERR_UNDEFINED_BEHAVIOR;

    m_pmfxSession = pmfxSession;
    m_nPoolSize = nPoolSize;

    m_pTasks = new sTask [m_nPoolSize];
    MSDK_CHECK_POINTER(m_pTasks, MFX_ERR_MEMORY_ALLOC);

    mfxStatus sts = MFX_ERR_NONE;

    if (pOtherWriter) // 2 bitstreams on output
    {
        for (mfxU32 i = 0; i < m_nPoolSize; i+=2)
        {
            sts = m_pTasks[i+0].Init(nBufferSize, pBitstreamWriter, pYUVWriter, pmfxAllocator);
            sts = m_pTasks[i+1].Init(nBufferSize, pOtherWriter);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
        }
    }
    else
    {
        for (mfxU32 i = 0; i < m_nPoolSize; i++)
        {
            sts = m_pTasks[i].Init(nBufferSize, pBitstreamWriter, pYUVWriter, pmfxAllocator);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
        }
    }

    return MFX_ERR_NONE;
}

mfxStatus CEncTaskPool::SynchronizeFirstTask()
{
    MSDK_CHECK_POINTER(m_pTasks, MFX_ERR_NOT_INITIALIZED);
    MSDK_CHECK_POINTER(m_pmfxSession, MFX_ERR_NOT_INITIALIZED);

    mfxStatus sts  = MFX_ERR_NONE;

    // non-null sync point indicates that task is in execution
    if (NULL != m_pTasks[m_nTaskBufferStart].EncSyncP)
    {
        sts = m_pmfxSession->SyncOperation(m_pTasks[m_nTaskBufferStart].EncSyncP, MSDK_WAIT_INTERVAL);

        if (MFX_ERR_NONE == sts)
        {
            sts = m_pTasks[m_nTaskBufferStart].WriteBitstream();
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            sts = m_pTasks[m_nTaskBufferStart].Reset();
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

            // move task buffer start to the next executing task
            // the first transform frame to the right with non zero sync point
            for (mfxU32 i = 0; i < m_nPoolSize; i++)
            {
                m_nTaskBufferStart = (m_nTaskBufferStart + 1) % m_nPoolSize;
                if (NULL != m_pTasks[m_nTaskBufferStart].EncSyncP)
                {
                    break;
                }
            }
        }
        else if (MFX_ERR_ABORTED == sts)
        {
            while (!m_pTasks[m_nTaskBufferStart].DependentVppTasks.empty())
            {
                // find out if the error occurred in a VPP task to perform recovery procedure if applicable
                sts = m_pmfxSession->SyncOperation(*m_pTasks[m_nTaskBufferStart].DependentVppTasks.begin(), 0);

                if (MFX_ERR_NONE == sts)
                {
                    m_pTasks[m_nTaskBufferStart].DependentVppTasks.pop_front();
                    sts = MFX_ERR_ABORTED; // save the status of the encode task
                    continue; // go to next vpp task
                }
                else
                {
                    break;
                }
            }
        }

        return sts;
    }
    else
    {
        return MFX_ERR_NOT_FOUND; // no tasks left in task buffer
    }
}

mfxU32 CEncTaskPool::GetFreeTaskIndex()
{
    mfxU32 off = 0;

    if (m_pTasks)
    {
        for (off = 0; off < m_nPoolSize; off++)
        {
            if (NULL == m_pTasks[(m_nTaskBufferStart + off) % m_nPoolSize].EncSyncP)
            {
                break;
            }
        }
    }

    if (off >= m_nPoolSize)
        return m_nPoolSize;

    return (m_nTaskBufferStart + off) % m_nPoolSize;
}

mfxStatus CEncTaskPool::GetFreeTask(sTask **ppTask)
{
    MSDK_CHECK_POINTER(ppTask, MFX_ERR_NULL_PTR);
    MSDK_CHECK_POINTER(m_pTasks, MFX_ERR_NOT_INITIALIZED);

    mfxU32 index = GetFreeTaskIndex();

    if (index >= m_nPoolSize)
    {
        return MFX_ERR_NOT_FOUND;
    }

    // return the address of the task
    *ppTask = &m_pTasks[index];

    return MFX_ERR_NONE;
}

void CEncTaskPool::Close()
{
    if (m_pTasks)
    {
        for (mfxU32 i = 0; i < m_nPoolSize; i++)
        {
            m_pTasks[i].Close();
        }
    }

    MSDK_SAFE_DELETE_ARRAY(m_pTasks);

    m_pmfxSession = NULL;
    m_nTaskBufferStart = 0;
    m_nPoolSize = 0;
}

sTask::sTask()
    : EncSyncP(0),
    pBsWriter(nullptr),
    pYUVWriter(nullptr),
    mfxSurf(nullptr)
{
    MSDK_ZERO_MEMORY(mfxBS);
}

mfxStatus sTask::Init(mfxU32 nBufferSize, CSmplBitstreamWriter *pBitstreamWriter, CSmplYUVWriter *pFrameWriter, MFXFrameAllocator *pAllocator)
{
    Close();

    pBsWriter = pBitstreamWriter;

    mfxStatus sts = Reset();
    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

    if (pBitstreamWriter) {
        MSDK_CHECK_ERROR(nBufferSize, 0, MFX_ERR_UNDEFINED_BEHAVIOR);
        sts = InitMfxBitstream(&mfxBS, nBufferSize);
        MSDK_CHECK_RESULT_SAFE(sts, MFX_ERR_NONE, sts, WipeMfxBitstream(&mfxBS));
    } else {
        //フレーム出力時には、Allocatorも必要
        MSDK_CHECK_POINTER(pFrameWriter, MFX_ERR_NULL_PTR);
        MSDK_CHECK_POINTER(pAllocator, MFX_ERR_NULL_PTR);
        pYUVWriter = pFrameWriter;
        pmfxAllocator = pAllocator;
    }

    return sts;
}

mfxStatus sTask::Close()
{
    WipeMfxBitstream(&mfxBS);
    EncSyncP = 0;
    pBsWriter = nullptr;
    pYUVWriter = nullptr;
    pmfxAllocator = nullptr;
    mfxSurf = nullptr;
    DependentVppTasks.clear();

    return MFX_ERR_NONE;
}

mfxStatus sTask::WriteBitstream()
{
    if (!pBsWriter && !pYUVWriter)
        return MFX_ERR_NOT_INITIALIZED;
    
    mfxStatus sts = MFX_ERR_NONE;
    if (pBsWriter) {
        sts = pBsWriter->WriteNextFrame(&mfxBS);
    } else {
        sts = pmfxAllocator->Lock(pmfxAllocator->pthis, mfxSurf->Data.MemId, &(mfxSurf->Data));
        MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

        sts = pYUVWriter->WriteNextFrame(mfxSurf);

        pmfxAllocator->Unlock(pmfxAllocator->pthis, mfxSurf->Data.MemId, &(mfxSurf->Data));

        //最終で加算したLockをここで減算する
        mfxSurf->Data.Locked--;
    }
    return sts;
}

mfxStatus sTask::Reset()
{
    // mark sync point as free
    EncSyncP = NULL;
    mfxSurf = nullptr;

    // prepare bit stream
    mfxBS.DataOffset = 0;
    mfxBS.DataLength = 0;

    DependentVppTasks.clear();

    return MFX_ERR_NONE;
}

#if ENABLE_MVC_ENCODING
mfxStatus CEncodingPipeline::AllocAndInitMVCSeqDesc()
{
    // a simple example of mfxExtMVCSeqDesc structure filling
    // actually equal to the "Default dependency mode" - when the structure fields are left 0,
    // but we show how to properly allocate and fill the fields

    mfxU32 i;

    // mfxMVCViewDependency array
    m_MVCSeqDesc.NumView = m_nNumView;
    m_MVCSeqDesc.NumViewAlloc = m_nNumView;
    m_MVCSeqDesc.View = new mfxMVCViewDependency[m_MVCSeqDesc.NumViewAlloc];
    MSDK_CHECK_POINTER(m_MVCSeqDesc.View, MFX_ERR_MEMORY_ALLOC);
    for (i = 0; i < m_MVCSeqDesc.NumViewAlloc; ++i)
    {
        MSDK_ZERO_MEMORY(m_MVCSeqDesc.View[i]);
        m_MVCSeqDesc.View[i].ViewId = (mfxU16) i; // set view number as view id
    }

    // set up dependency for second view
    m_MVCSeqDesc.View[1].NumAnchorRefsL0 = 1;
    m_MVCSeqDesc.View[1].AnchorRefL0[0] = 0;     // ViewId 0 - base view

    m_MVCSeqDesc.View[1].NumNonAnchorRefsL0 = 1;
    m_MVCSeqDesc.View[1].NonAnchorRefL0[0] = 0;  // ViewId 0 - base view

    // viewId array
    m_MVCSeqDesc.NumViewId = m_nNumView;
    m_MVCSeqDesc.NumViewIdAlloc = m_nNumView;
    m_MVCSeqDesc.ViewId = new mfxU16[m_MVCSeqDesc.NumViewIdAlloc];
    MSDK_CHECK_POINTER(m_MVCSeqDesc.ViewId, MFX_ERR_MEMORY_ALLOC);
    for (i = 0; i < m_MVCSeqDesc.NumViewIdAlloc; ++i)
    {
        m_MVCSeqDesc.ViewId[i] = (mfxU16) i;
    }

    // create a single operation point containing all views
    m_MVCSeqDesc.NumOP = 1;
    m_MVCSeqDesc.NumOPAlloc = 1;
    m_MVCSeqDesc.OP = new mfxMVCOperationPoint[m_MVCSeqDesc.NumOPAlloc];
    MSDK_CHECK_POINTER(m_MVCSeqDesc.OP, MFX_ERR_MEMORY_ALLOC);
    for (i = 0; i < m_MVCSeqDesc.NumOPAlloc; ++i)
    {
        MSDK_ZERO_MEMORY(m_MVCSeqDesc.OP[i]);
        m_MVCSeqDesc.OP[i].NumViews = (mfxU16) m_nNumView;
        m_MVCSeqDesc.OP[i].NumTargetViews = (mfxU16) m_nNumView;
        m_MVCSeqDesc.OP[i].TargetViewId = m_MVCSeqDesc.ViewId; // points to mfxExtMVCSeqDesc::ViewId
    }

    return MFX_ERR_NONE;
}
#endif

mfxStatus CEncodingPipeline::AllocAndInitVppDoNotUse()
{
    MSDK_ZERO_MEMORY(m_VppDoNotUse);
    m_VppDoNotUse.Header.BufferId = MFX_EXTBUFF_VPP_DONOTUSE;
    m_VppDoNotUse.Header.BufferSz = sizeof(mfxExtVPPDoNotUse);
    m_VppDoNotUse.NumAlg = (mfxU32)m_VppDoNotUseList.size();
    m_VppDoNotUse.AlgList = &m_VppDoNotUseList[0];
    return MFX_ERR_NONE;
} // CEncodingPipeline::AllocAndInitVppDoNotUse()

void CEncodingPipeline::FreeVppDoNotUse()
{
}

#if ENABLE_MVC_ENCODING
void CEncodingPipeline::FreeMVCSeqDesc()
{
    MSDK_SAFE_DELETE_ARRAY(m_MVCSeqDesc.View);
    MSDK_SAFE_DELETE_ARRAY(m_MVCSeqDesc.ViewId);
    MSDK_SAFE_DELETE_ARRAY(m_MVCSeqDesc.OP);
}
#endif

mfxStatus CEncodingPipeline::InitMfxDecParams()
{
#if ENABLE_AVCODEC_QSV_READER
    mfxStatus sts = MFX_ERR_NONE;
    if (m_pFileReader->getInputCodec()) {
        m_pFileReader->GetDecParam(&m_mfxDecParams);
        PrintMes(QSV_LOG_DEBUG, _T("")
            _T("InitMfxDecParams: QSVDec prm: %s, Level %d, Profile %d\n")
            _T("InitMfxDecParams: Frame: %s, %dx%d%s [%d,%d,%d,%d] %d:%d, bitdepth %d, shift %d\n"),
            CodecIdToStr(m_mfxDecParams.mfx.CodecId).c_str(), m_mfxDecParams.mfx.CodecLevel, m_mfxDecParams.mfx.CodecProfile,
            ColorFormatToStr(m_mfxDecParams.mfx.FrameInfo.ChromaFormat), m_mfxDecParams.mfx.FrameInfo.Width, m_mfxDecParams.mfx.FrameInfo.Height,
            (m_mfxDecParams.mfx.FrameInfo.PicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF)) ? _T("i") : _T("p"),
            m_mfxDecParams.mfx.FrameInfo.CropX, m_mfxDecParams.mfx.FrameInfo.CropY, m_mfxDecParams.mfx.FrameInfo.CropW, m_mfxDecParams.mfx.FrameInfo.CropH,
            m_mfxDecParams.mfx.FrameInfo.AspectRatioW, m_mfxDecParams.mfx.FrameInfo.AspectRatioH,
            m_mfxDecParams.mfx.FrameInfo.BitDepthLuma, m_mfxDecParams.mfx.FrameInfo.Shift);

        InitMfxBitstream(&m_DecInputBitstream, AVCODEC_READER_INPUT_BUF_SIZE);
        //TimeStampはQSVに自動的に計算させる
        m_DecInputBitstream.TimeStamp = (mfxU64)MFX_TIMESTAMP_UNKNOWN;
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_6)) {
            m_DecInputBitstream.DecodeTimeStamp = (mfxU64)MFX_TIMESTAMP_UNKNOWN;
        }

        sts = m_pFileReader->GetHeader(&m_DecInputBitstream);
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("InitMfxDecParams: Failed to get stream header from reader."));

        //デコーダの作成
        m_pmfxDEC = new MFXVideoDECODE(m_mfxSession);
        MSDK_CHECK_POINTER(m_pmfxDEC, MFX_ERR_MEMORY_ALLOC);

        if (m_pFileReader->getInputCodec() == MFX_CODEC_HEVC) {
            PrintMes(QSV_LOG_DEBUG, _T("InitMfxDecParams: Loading HEVC decoder plugin..."));
            m_pDecPlugin.reset(LoadPlugin(MFX_PLUGINTYPE_VIDEO_DECODE, m_mfxSession, MFX_PLUGINID_HEVCD_HW, 1));
            if (m_pDecPlugin.get() == NULL) {
                PrintMes(QSV_LOG_ERROR, _T("Failed to load hw hevc decoder.\n"));
                return MFX_ERR_UNSUPPORTED;
            }
            PrintMes(QSV_LOG_DEBUG, _T("InitMfxDecParams: Loaded HEVC decoder plugin.\n"));
        }

        sts = m_pmfxDEC->Init(&m_mfxDecParams);
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("InitMfxDecParams: Failed to initialize QSV decoder."));
        PrintMes(QSV_LOG_DEBUG, _T("InitMfxDecParams: Initialized QSVDec.\n"));
    }
#endif
    return MFX_ERR_NONE;
}

mfxStatus CEncodingPipeline::InitMfxEncParams(sInputParams *pInParams)
{
    if (pInParams->CodecId == MFX_CODEC_RAW) {
        PrintMes(QSV_LOG_DEBUG, _T("Raw codec is selected, disable encode.\n"));
        return MFX_ERR_NONE;
    }
    const mfxU32 blocksz = (pInParams->CodecId == MFX_CODEC_HEVC) ? 32 : 16;
    auto print_feature_warnings = [this](int log_level, const TCHAR *feature_name) {
        PrintMes(log_level, _T("%s is not supported on current platform, disabled.\n"), feature_name);
    };
    //エンコードモードのチェック
    mfxU64 availableFeaures = CheckEncodeFeature(pInParams->bUseHWLib, m_mfxVer, pInParams->nEncMode, pInParams->CodecId);
    PrintMes(QSV_LOG_DEBUG, _T("Detected avaliable features for %s API v%d.%d, %s, %s\n%s\n"),
        (pInParams->bUseHWLib) ? _T("hw") : _T("sw"), m_mfxVer.Major, m_mfxVer.Minor,
        CodecIdToStr(pInParams->CodecId).c_str(), EncmodeToStr(pInParams->nEncMode), MakeFeatureListStr(availableFeaures).c_str());
    if (!(availableFeaures & ENC_FEATURE_CURRENT_RC)) {
        PrintMes(QSV_LOG_ERROR, _T("%s mode is not supported on current platform.\n"), EncmodeToStr(pInParams->nEncMode));
        if (MFX_RATECONTROL_LA == pInParams->nEncMode) {
            if (!check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_7)) {
                PrintMes(QSV_LOG_ERROR, _T("Lookahead mode is only supported by API v1.7 or later.\n"));
            }
        }
        if (   MFX_RATECONTROL_ICQ    == pInParams->nEncMode
            || MFX_RATECONTROL_LA_ICQ == pInParams->nEncMode
            || MFX_RATECONTROL_VCM    == pInParams->nEncMode) {
            if (!check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)) {
                PrintMes(QSV_LOG_ERROR, _T("%s mode is only supported by API v1.8 or later.\n"), EncmodeToStr(pInParams->nEncMode));
            }
        }
        if (   MFX_RATECONTROL_LA_EXT == pInParams->nEncMode
            || MFX_RATECONTROL_LA_HRD == pInParams->nEncMode
            || MFX_RATECONTROL_QVBR   == pInParams->nEncMode) {
            if (!check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_11)) {
                PrintMes(QSV_LOG_ERROR, _T("%s mode is only supported by API v1.11 or later.\n"), EncmodeToStr(pInParams->nEncMode));
            }
        }
        return MFX_ERR_INVALID_VIDEO_PARAM;
    }
    if (MFX_RATECONTROL_VQP == pInParams->nEncMode && m_pFileReader->getInputCodec()) {
        PrintMes(QSV_LOG_ERROR, _T("%s mode cannot be used with avqsv reader.\n"), EncmodeToStr(pInParams->nEncMode));
        return MFX_ERR_INVALID_VIDEO_PARAM;
    }
    if (pInParams->nBframes == QSV_BFRAMES_AUTO) {
        pInParams->nBframes = (pInParams->CodecId == MFX_CODEC_HEVC) ? QSV_DEFAULT_HEVC_BFRAMES : QSV_DEFAULT_H264_BFRAMES;
    }
    //その他機能のチェック
    if (pInParams->bAdaptiveI && !(availableFeaures & ENC_FEATURE_ADAPTIVE_I)) {
        PrintMes(QSV_LOG_WARN, _T("Adaptve I-frame insert is not supported on current platform, disabled.\n"));
        pInParams->bAdaptiveI = false;
    }
    if (pInParams->bAdaptiveB && !(availableFeaures & ENC_FEATURE_ADAPTIVE_B)) {
        PrintMes(QSV_LOG_WARN, _T("Adaptve B-frame insert is not supported on current platform, disabled.\n"));
        pInParams->bAdaptiveB = false;
    }
    if (pInParams->bBPyramid && !(availableFeaures & ENC_FEATURE_B_PYRAMID)) {
        print_feature_warnings(QSV_LOG_WARN, _T("B pyramid"));
        pInParams->bBPyramid = false;
    }
    if (pInParams->bCAVLC && !(availableFeaures & ENC_FEATURE_CAVLC)) {
        print_feature_warnings(QSV_LOG_WARN, _T("CAVLC"));
        pInParams->bCAVLC = false;
    }
    if (pInParams->bExtBRC && !(availableFeaures & ENC_FEATURE_EXT_BRC)) {
        print_feature_warnings(QSV_LOG_WARN, _T("ExtBRC"));
        pInParams->bExtBRC = false;
    }
    if (pInParams->bMBBRC && !(availableFeaures & ENC_FEATURE_MBBRC)) {
        print_feature_warnings(QSV_LOG_WARN, _T("MBBRC"));
        pInParams->bMBBRC = false;
    }
    if (!pInParams->bforceGOPSettings && !(availableFeaures & ENC_FEATURE_SCENECHANGE)) {
        print_feature_warnings(QSV_LOG_WARN, _T("Scene change detection"));
        pInParams->bforceGOPSettings = true;
    }
    if (   (MFX_RATECONTROL_LA     == pInParams->nEncMode
         || MFX_RATECONTROL_LA_ICQ == pInParams->nEncMode)
        && pInParams->nLookaheadDS != MFX_LOOKAHEAD_DS_UNKNOWN
        && !(availableFeaures & ENC_FEATURE_LA_DS)) {
        print_feature_warnings(QSV_LOG_WARN, _T("Lookahead qaulity setting"));
        pInParams->nLookaheadDS = MFX_LOOKAHEAD_DS_UNKNOWN;
    }
    if (pInParams->nTrellis != MFX_TRELLIS_UNKNOWN && !(availableFeaures & ENC_FEATURE_TRELLIS)) {
        print_feature_warnings(QSV_LOG_WARN, _T("trellis"));
        pInParams->nTrellis = MFX_TRELLIS_UNKNOWN;
    }
    if (pInParams->bRDO && !(availableFeaures & ENC_FEATURE_RDO)) {
        print_feature_warnings(QSV_LOG_WARN, _T("RDO"));
        pInParams->bRDO = false;
    }
    if ((pInParams->nPicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF))
        && pInParams->vpp.nDeinterlace == MFX_DEINTERLACE_NONE
        && !(availableFeaures & ENC_FEATURE_INTERLACE)) {
        PrintMes(QSV_LOG_ERROR, _T("Interlaced encoding is not supported on current rate control mode.\n"));
        return MFX_ERR_INVALID_VIDEO_PARAM;
    }
    if (pInParams->nBframes > 2 && pInParams->CodecId == MFX_CODEC_HEVC) {
        PrintMes(QSV_LOG_WARN, _T("HEVC encoding + B-frames > 2 might cause artifacts, please check the output.\n"));
    }
    if (pInParams->bBPyramid && !pInParams->bforceGOPSettings && !(availableFeaures & ENC_FEATURE_B_PYRAMID_AND_SC)) {
        PrintMes(QSV_LOG_WARN, _T("B pyramid with scenechange is not supported on current platform, B pyramid disabled.\n"));
        pInParams->bBPyramid = false;
    }
    if (pInParams->bBPyramid && pInParams->nBframes >= 10 && !(availableFeaures & ENC_FEATURE_B_PYRAMID_MANY_BFRAMES)) {
        PrintMes(QSV_LOG_WARN, _T("B pyramid with too many bframes is not supported on current platform, B pyramid disabled.\n"));
        pInParams->bBPyramid = false;
    }
    if (pInParams->bBPyramid && pInParams->bUseHWLib && getCPUGen() < CPU_GEN_HASWELL) {
        PrintMes(QSV_LOG_WARN, _T("B pyramid on IvyBridge generation might cause artifacts, please check your encoded video.\n"));
    }
    if (pInParams->bNoDeblock && !(availableFeaures & ENC_FEATURE_NO_DEBLOCK)) {
        print_feature_warnings(QSV_LOG_WARN, _T("No deblock"));
        pInParams->bNoDeblock = false;
    }
    if (pInParams->bIntraRefresh && !(availableFeaures & ENC_FEATURE_INTRA_REFRESH)) {
        print_feature_warnings(QSV_LOG_WARN, _T("Intra Refresh"));
        pInParams->bIntraRefresh = false;
    }
    if (0 != (pInParams->nQPMin[0] | pInParams->nQPMin[1] | pInParams->nQPMin[2]
            | pInParams->nQPMax[0] | pInParams->nQPMax[1] | pInParams->nQPMax[2]) && !(availableFeaures & ENC_FEATURE_QP_MINMAX)) {
        print_feature_warnings(QSV_LOG_WARN, _T("Min/Max QP"));
        memset(pInParams->nQPMin, 0, sizeof(pInParams->nQPMin));
        memset(pInParams->nQPMax, 0, sizeof(pInParams->nQPMax));
    }
    if (0 != pInParams->nWinBRCSize) {
        if (!(availableFeaures & ENC_FEATURE_WINBRC)) {
            print_feature_warnings(QSV_LOG_WARN, _T("WinBRC"));
            pInParams->nWinBRCSize = 0;
        } else if (0 == pInParams->nMaxBitrate) {
            print_feature_warnings(QSV_LOG_WARN, _T("Min/Max QP"));
            PrintMes(QSV_LOG_WARN, _T("WinBRC requires Max bitrate to be set, disabled.\n"));
            pInParams->nWinBRCSize = 0;
        }
    }
    if (pInParams->bDirectBiasAdjust && !(availableFeaures & ENC_FEATURE_DIRECT_BIAS_ADJUST)) {
        print_feature_warnings(QSV_LOG_WARN, _T("Direct Bias Adjust"));
        pInParams->bDirectBiasAdjust = 0;
    }
    if (pInParams->bGlobalMotionAdjust && !(availableFeaures & ENC_FEATURE_GLOBAL_MOTION_ADJUST)) {
        print_feature_warnings(QSV_LOG_WARN, _T("MV Cost Scaling"));
        pInParams->bGlobalMotionAdjust = 0;
        pInParams->nMVCostScaling = 0;
    }
    if (pInParams->bUseFixedFunc && !(availableFeaures & ENC_FEATURE_FIXED_FUNC)) {
        print_feature_warnings(QSV_LOG_WARN, _T("Fixed Func"));
        pInParams->bUseFixedFunc = 0;
    }
    if (!(availableFeaures & ENC_FEATURE_VUI_INFO)) {
        if (pInParams->bFullrange) {
            print_feature_warnings(QSV_LOG_WARN, _T("fullrange"));
            pInParams->bFullrange = false;
        }
        if (pInParams->Transfer) {
            print_feature_warnings(QSV_LOG_WARN, _T("transfer"));
            pInParams->Transfer = (mfxU16)list_transfer[0].value;
        }
        if (pInParams->VideoFormat) {
            print_feature_warnings(QSV_LOG_WARN, _T("videoformat"));
            pInParams->VideoFormat = (mfxU16)list_videoformat[0].value;
        }
        if (pInParams->ColorMatrix) {
            print_feature_warnings(QSV_LOG_WARN, _T("colormatrix"));
            pInParams->ColorMatrix = (mfxU16)list_colormatrix[0].value;
        }
        if (pInParams->ColorPrim) {
            print_feature_warnings(QSV_LOG_WARN, _T("colorprim"));
            pInParams->ColorPrim = (mfxU16)list_colorprim[0].value;
        }
    }

    //Intra Refereshが指定された場合は、GOP関連の設定を自動的に上書き
    if (pInParams->bIntraRefresh) {
        pInParams->bforceGOPSettings = true; //シーンチェンジ検出オフ
    }

    //GOP長さが短いならVQPもシーンチェンジ検出も実行しない
    if (pInParams->nGOPLength != 0 && pInParams->nGOPLength < 4) {
        if (!pInParams->bforceGOPSettings) {
            PrintMes(QSV_LOG_WARN, _T("Scene change detection cannot be used with very short GOP length.\n"));
            pInParams->bforceGOPSettings = true;
        }
        if (pInParams->nEncMode == MFX_RATECONTROL_VQP)    {
            PrintMes(QSV_LOG_WARN, _T("VQP mode cannot be used with very short GOP length.\n"));
            PrintMes(QSV_LOG_WARN, _T("Switching to CQP mode.\n"));
            pInParams->nEncMode = MFX_RATECONTROL_CQP;
        }
    }
    //拡張設定
    if (!pInParams->bforceGOPSettings) {
        if (pInParams->nPicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF)) {
            switch (pInParams->vpp.nDeinterlace) {
            case MFX_DEINTERLACE_NORMAL:
            case MFX_DEINTERLACE_BOB:
            case MFX_DEINTERLACE_AUTO_SINGLE:
            case MFX_DEINTERLACE_AUTO_DOUBLE:
                break;
            default:
                PrintMes(QSV_LOG_WARN, _T("Scene change detection cannot be used with interlaced output, disabled.\n"));
                pInParams->bforceGOPSettings = true;
                break;
            }
        }
        if (m_pFileReader->getInputCodec()) {
            PrintMes(QSV_LOG_WARN, _T("Scene change detection cannot be used with transcoding, disabled.\n"));
            pInParams->bforceGOPSettings = true;
        }
        if (!pInParams->bforceGOPSettings) {
            m_nExPrm |= MFX_PRM_EX_SCENE_CHANGE;
        }
    }
    if (pInParams->nEncMode == MFX_RATECONTROL_VQP)    { 
        if (pInParams->nPicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF)) {
            switch (pInParams->vpp.nDeinterlace) {
            case MFX_DEINTERLACE_NORMAL:
            case MFX_DEINTERLACE_BOB:
            case MFX_DEINTERLACE_AUTO_SINGLE:
            case MFX_DEINTERLACE_AUTO_DOUBLE:
                break;
            default:
                PrintMes(QSV_LOG_ERROR, _T("VQP mode cannot be used with interlaced output.\n"));
                return MFX_ERR_INVALID_VIDEO_PARAM;
            }
        } else if (m_pFileReader->getInputCodec()) {
            PrintMes(QSV_LOG_ERROR, _T("VQP mode cannot be used with transcoding.\n"));
            return MFX_ERR_INVALID_VIDEO_PARAM;
        }
        m_nExPrm |= MFX_PRM_EX_VQP;
    }
    //profileを守るための調整
    if (pInParams->CodecProfile == MFX_PROFILE_AVC_BASELINE) {
        pInParams->nBframes = 0;
        pInParams->bCAVLC = true;
    }
    if (pInParams->bCAVLC)
        pInParams->bRDO = false;

    //設定開始
    m_mfxEncParams.mfx.CodecId                 = pInParams->CodecId;
    m_mfxEncParams.mfx.RateControlMethod       =(pInParams->nEncMode == MFX_RATECONTROL_VQP) ? MFX_RATECONTROL_CQP : pInParams->nEncMode;
    if (MFX_RATECONTROL_CQP == m_mfxEncParams.mfx.RateControlMethod) {
        //CQP
        m_mfxEncParams.mfx.QPI             = pInParams->nQPI;
        m_mfxEncParams.mfx.QPP             = pInParams->nQPP;
        m_mfxEncParams.mfx.QPB             = pInParams->nQPB;
    } else if (MFX_RATECONTROL_ICQ    == m_mfxEncParams.mfx.RateControlMethod
            || MFX_RATECONTROL_LA_ICQ == m_mfxEncParams.mfx.RateControlMethod) {
        m_mfxEncParams.mfx.ICQQuality      = pInParams->nICQQuality;
        m_mfxEncParams.mfx.MaxKbps         = 0;
    } else {
        if (pInParams->nBitRate > USHRT_MAX) {
            m_mfxEncParams.mfx.BRCParamMultiplier = (mfxU16)(max(pInParams->nBitRate, pInParams->nMaxBitrate) / USHRT_MAX) + 1;
            pInParams->nBitRate    /= m_mfxEncParams.mfx.BRCParamMultiplier;
            pInParams->nMaxBitrate /= m_mfxEncParams.mfx.BRCParamMultiplier;
        }
        m_mfxEncParams.mfx.TargetKbps      = (mfxU16)pInParams->nBitRate; // in kbps
        if (m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_AVBR) {
            //AVBR
            //m_mfxEncParams.mfx.Accuracy        = pInParams->nAVBRAccuarcy;
            m_mfxEncParams.mfx.Accuracy        = 500;
            m_mfxEncParams.mfx.Convergence     = pInParams->nAVBRConvergence;
        } else {
            //CBR, VBR
            m_mfxEncParams.mfx.MaxKbps         = (mfxU16)pInParams->nMaxBitrate;
        }
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_15)) {
        m_mfxEncParams.mfx.LowPower = (mfxU16)((pInParams->bUseFixedFunc) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF);
    }
    m_mfxEncParams.mfx.TargetUsage             = pInParams->nTargetUsage; // trade-off between quality and speed

    mfxU32 OutputFPSRate = pInParams->nFPSRate;
    mfxU32 OutputFPSScale = pInParams->nFPSScale;
    if ((pInParams->nPicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF))) {
        switch (pInParams->vpp.nDeinterlace) {
        case MFX_DEINTERLACE_IT:
        case MFX_DEINTERLACE_IT_MANUAL:
            OutputFPSRate = OutputFPSRate * 4;
            OutputFPSScale = OutputFPSScale * 5;
            break;
        case MFX_DEINTERLACE_BOB:
        case MFX_DEINTERLACE_AUTO_DOUBLE:
            OutputFPSRate = OutputFPSRate * 2;
            break;
        default:
            break;
        }
    } else {
        switch (pInParams->vpp.nFPSConversion) {
        case FPS_CONVERT_MUL2:
            OutputFPSRate = OutputFPSRate * 2;
            break;
        case FPS_CONVERT_MUL2_5:
            OutputFPSRate = OutputFPSRate * 5 / 2;
            break;
        default:
            break;
        }
    }
    mfxU32 gcd = qsv_gcd(OutputFPSRate, OutputFPSScale);
    OutputFPSRate /= gcd;
    OutputFPSScale /= gcd;
    PrintMes(QSV_LOG_DEBUG, _T("InitMfxEncParams: Output FPS %d/%d\n"), OutputFPSRate, OutputFPSScale);
    if (pInParams->nGOPLength == 0) {
        pInParams->nGOPLength = (mfxU16)((OutputFPSRate + OutputFPSScale - 1) / OutputFPSScale) * 10;
        PrintMes(QSV_LOG_DEBUG, _T("InitMfxEncParams: Auto GOP Length: %d\n"), pInParams->nGOPLength);
    }
    m_mfxEncParams.mfx.FrameInfo.FrameRateExtN = OutputFPSRate;
    m_mfxEncParams.mfx.FrameInfo.FrameRateExtD = OutputFPSScale;
    m_mfxEncParams.mfx.EncodedOrder            = 0; // binary flag, 0 signals encoder to take frames in display order
    m_mfxEncParams.mfx.NumSlice                = pInParams->nSlices;

    m_mfxEncParams.mfx.NumRefFrame             = pInParams->nRef;
    m_mfxEncParams.mfx.CodecLevel              = pInParams->CodecLevel;
    m_mfxEncParams.mfx.CodecProfile            = pInParams->CodecProfile;
    m_mfxEncParams.mfx.GopOptFlag              = 0;
    m_mfxEncParams.mfx.GopOptFlag             |= (!pInParams->bopenGOP) ? MFX_GOP_CLOSED : 0x00;
    m_mfxEncParams.mfx.IdrInterval             = (!pInParams->bopenGOP) ? 0 : (mfxU16)((OutputFPSRate + OutputFPSScale - 1) / OutputFPSScale) * 20 / pInParams->nGOPLength;
    //MFX_GOP_STRICTにより、インタレ保持時にフレームが壊れる場合があるため、無効とする
    //m_mfxEncParams.mfx.GopOptFlag             |= (pInParams->bforceGOPSettings) ? MFX_GOP_STRICT : NULL;

    m_mfxEncParams.mfx.GopPicSize              = (pInParams->bIntraRefresh) ? 0 : pInParams->nGOPLength;
    m_mfxEncParams.mfx.GopRefDist              = (mfxU16)(clamp(pInParams->nBframes, -1, 16) + 1);

    // specify memory type
    m_mfxEncParams.IOPattern = (mfxU16)((pInParams->memType != SYSTEM_MEMORY) ? MFX_IOPATTERN_IN_VIDEO_MEMORY : MFX_IOPATTERN_IN_SYSTEM_MEMORY);

    // frame info parameters
    m_mfxEncParams.mfx.FrameInfo.FourCC       = MFX_FOURCC_NV12;
    m_mfxEncParams.mfx.FrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
    m_mfxEncParams.mfx.FrameInfo.PicStruct    = (pInParams->vpp.nDeinterlace) ? MFX_PICSTRUCT_PROGRESSIVE : pInParams->nPicStruct;

    // set sar info
    mfxI32 m_iSAR[2] = { pInParams->nPAR[0], pInParams->nPAR[1] };
    adjust_sar(&m_iSAR[0], &m_iSAR[1], pInParams->nDstWidth, pInParams->nDstHeight);
    m_mfxEncParams.mfx.FrameInfo.AspectRatioW = (mfxU16)m_iSAR[0];
    m_mfxEncParams.mfx.FrameInfo.AspectRatioH = (mfxU16)m_iSAR[1];

    MSDK_ZERO_MEMORY(m_CodingOption);
    m_CodingOption.Header.BufferId = MFX_EXTBUFF_CODING_OPTION;
    m_CodingOption.Header.BufferSz = sizeof(mfxExtCodingOption);
    if (!pInParams->bUseHWLib) {
        //swライブラリ使用時のみ
        m_CodingOption.InterPredBlockSize = pInParams->nInterPred;
        m_CodingOption.IntraPredBlockSize = pInParams->nIntraPred;
        m_CodingOption.MVSearchWindow     = pInParams->MVSearchWindow;
        m_CodingOption.MVPrecision        = pInParams->nMVPrecision;
    }
    if (!pInParams->bUseHWLib || pInParams->CodecProfile == MFX_PROFILE_AVC_BASELINE) {
        //swライブラリ使用時かbaselineを指定した時
        m_CodingOption.RateDistortionOpt  = (mfxU16)((pInParams->bRDO) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_UNKNOWN);
        m_CodingOption.CAVLC              = (mfxU16)((pInParams->bCAVLC) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_UNKNOWN);
    }
    //m_CodingOption.FramePicture = MFX_CODINGOPTION_ON;
    //m_CodingOption.FieldOutput = MFX_CODINGOPTION_ON;
    //m_CodingOption.VuiVclHrdParameters = MFX_CODINGOPTION_ON;
    //m_CodingOption.VuiNalHrdParameters = MFX_CODINGOPTION_ON;
    m_CodingOption.AUDelimiter = MFX_CODINGOPTION_OFF;
    m_CodingOption.PicTimingSEI = MFX_CODINGOPTION_OFF;
    //m_CodingOption.SingleSeiNalUnit = MFX_CODINGOPTION_OFF;

    //API v1.6の機能
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_6)) {
        INIT_MFX_EXT_BUFFER(m_CodingOption2, MFX_EXTBUFF_CODING_OPTION2);
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)) {
            m_CodingOption2.AdaptiveI   = (mfxU16)((pInParams->bAdaptiveI) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_UNKNOWN);
            m_CodingOption2.AdaptiveB   = (mfxU16)((pInParams->bAdaptiveB) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_UNKNOWN);
            m_CodingOption2.BRefType    = (mfxU16)((pInParams->bBPyramid)  ? MFX_B_REF_PYRAMID   : MFX_B_REF_OFF);
            m_CodingOption2.LookAheadDS = pInParams->nLookaheadDS;
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_7)) {
            m_CodingOption2.LookAheadDepth = (pInParams->nLookaheadDepth == 0) ? pInParams->nLookaheadDepth : clamp(pInParams->nLookaheadDepth, QSV_LOOKAHEAD_DEPTH_MIN, QSV_LOOKAHEAD_DEPTH_MAX);
            m_CodingOption2.Trellis = pInParams->nTrellis;
        }
        if (pInParams->bMBBRC) {
            m_CodingOption2.MBBRC = MFX_CODINGOPTION_ON;
        }

        if (pInParams->bExtBRC) {
            m_CodingOption2.ExtBRC = MFX_CODINGOPTION_ON;
        }
        if (pInParams->bIntraRefresh) {
            m_CodingOption2.IntRefType = 1;
            m_CodingOption2.IntRefCycleSize = (pInParams->nGOPLength >= 2) ? pInParams->nGOPLength : (mfxU16)((OutputFPSRate + OutputFPSScale - 1) / OutputFPSScale) * 10;
        }
        if (pInParams->bNoDeblock) {
            m_CodingOption2.DisableDeblockingIdc = MFX_CODINGOPTION_ON;
        }
        for (int i = 0; i < 3; i++) {
            mfxU8 qpMin = min(pInParams->nQPMin[i], pInParams->nQPMax[i]);
            mfxU8 qpMax = max(pInParams->nQPMin[i], pInParams->nQPMax[i]);
            pInParams->nQPMin[i] = (0 == pInParams->nQPMin[i]) ? 0 : qpMin;
            pInParams->nQPMax[i] = (0 == pInParams->nQPMax[i]) ? 0 : qpMax;
        }
        m_CodingOption2.MaxQPI = pInParams->nQPMax[0];
        m_CodingOption2.MaxQPP = pInParams->nQPMax[1];
        m_CodingOption2.MaxQPB = pInParams->nQPMax[2];
        m_CodingOption2.MinQPI = pInParams->nQPMin[0];
        m_CodingOption2.MinQPP = pInParams->nQPMin[1];
        m_CodingOption2.MinQPB = pInParams->nQPMin[2];
        m_EncExtParams.push_back((mfxExtBuffer *)&m_CodingOption2);
    }

    //API v1.11の機能
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_11)) {
        INIT_MFX_EXT_BUFFER(m_CodingOption3, MFX_EXTBUFF_CODING_OPTION3);
        if (MFX_RATECONTROL_QVBR == m_mfxEncParams.mfx.RateControlMethod) {
            m_CodingOption3.QVBRQuality = pInParams->nQVBRQuality;
        }
        if (0 != pInParams->nMaxBitrate) {
            m_CodingOption3.WinBRCSize = (0 != pInParams->nWinBRCSize) ? pInParams->nWinBRCSize : (mfxU16)((OutputFPSRate + OutputFPSScale - 1) / OutputFPSScale);
            m_CodingOption3.WinBRCMaxAvgKbps = (mfxU16)pInParams->nMaxBitrate;
        }

        //API v1.13の機能
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_13)) {
            m_CodingOption3.GlobalMotionBiasAdjustment = (mfxU16)((pInParams->bGlobalMotionAdjust) ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF);
            m_CodingOption3.DirectBiasAdjustment       = (mfxU16)((pInParams->bDirectBiasAdjust)   ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF);
            if (pInParams->bDirectBiasAdjust)
                m_CodingOption3.MVCostScalingFactor    = pInParams->nMVCostScaling;
        }
        m_EncExtParams.push_back((mfxExtBuffer *)&m_CodingOption3);
    }

    //Bluray互換出力
    if (pInParams->nBluray) {
        if (   m_mfxEncParams.mfx.RateControlMethod != MFX_RATECONTROL_CBR
            && m_mfxEncParams.mfx.RateControlMethod != MFX_RATECONTROL_VBR
            && m_mfxEncParams.mfx.RateControlMethod != MFX_RATECONTROL_LA) {
                if (pInParams->nBluray == 1) {
                    PrintMes(QSV_LOG_ERROR, _T("")
                        _T("Current encode mode (%s) is not preferred for Bluray encoding,\n")
                        _T("since it cannot set Max Bitrate.\n")
                        _T("Please consider using Lookahead/VBR/CBR mode for Bluray encoding.\n"), EncmodeToStr(m_mfxEncParams.mfx.RateControlMethod));
                    return MFX_ERR_INCOMPATIBLE_VIDEO_PARAM;
                } else {
                    //pInParams->nBluray == 2 -> force Bluray
                    PrintMes(QSV_LOG_WARN, _T("")
                        _T("Current encode mode (%s) is not preferred for Bluray encoding,\n")
                        _T("since it cannot set Max Bitrate.\n")
                        _T("This output might not be able to be played on a Bluray Player.\n")
                        _T("Please consider using Lookahead/VBR/CBR mode for Bluray encoding.\n"), EncmodeToStr(m_mfxEncParams.mfx.RateControlMethod));
                }
        }
        if (   m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_CBR
            || m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_VBR
            || m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_LA) {
                m_mfxEncParams.mfx.MaxKbps    = min(m_mfxEncParams.mfx.MaxKbps, 40000);
                m_mfxEncParams.mfx.TargetKbps = min(m_mfxEncParams.mfx.TargetKbps, m_mfxEncParams.mfx.MaxKbps);
                m_mfxEncParams.mfx.BufferSizeInKB = m_mfxEncParams.mfx.MaxKbps / 8;
                m_mfxEncParams.mfx.InitialDelayInKB = m_mfxEncParams.mfx.BufferSizeInKB / 2;
        } else {
            m_mfxEncParams.mfx.BufferSizeInKB = 25000 / 8;
        }
        m_mfxEncParams.mfx.CodecLevel = (m_mfxEncParams.mfx.CodecLevel == 0) ? MFX_LEVEL_AVC_41 : (min(m_mfxEncParams.mfx.CodecLevel, MFX_LEVEL_AVC_41));
        m_mfxEncParams.mfx.NumSlice   = max(m_mfxEncParams.mfx.NumSlice, 4);
        m_mfxEncParams.mfx.GopOptFlag &= (~MFX_GOP_STRICT);
        m_mfxEncParams.mfx.GopRefDist = min(m_mfxEncParams.mfx.GopRefDist, 3+1);
        m_mfxEncParams.mfx.GopPicSize = (int)(min(m_mfxEncParams.mfx.GopPicSize, 30) / m_mfxEncParams.mfx.GopRefDist) * m_mfxEncParams.mfx.GopRefDist;
        m_mfxEncParams.mfx.NumRefFrame = min(m_mfxEncParams.mfx.NumRefFrame, 6);
        m_CodingOption.MaxDecFrameBuffering = m_mfxEncParams.mfx.NumRefFrame;
        m_CodingOption.VuiNalHrdParameters = MFX_CODINGOPTION_ON;
        m_CodingOption.VuiVclHrdParameters = MFX_CODINGOPTION_ON;
        m_CodingOption.AUDelimiter  = MFX_CODINGOPTION_ON;
        m_CodingOption.PicTimingSEI = MFX_CODINGOPTION_ON;
        m_CodingOption.ResetRefList = MFX_CODINGOPTION_ON;
        m_nExPrm &= (~MFX_PRM_EX_SCENE_CHANGE);
        //m_CodingOption.EndOfSequence = MFX_CODINGOPTION_ON; //hwモードでは効果なし 0x00, 0x00, 0x01, 0x0a
        //m_CodingOption.EndOfStream   = MFX_CODINGOPTION_ON; //hwモードでは効果なし 0x00, 0x00, 0x01, 0x0b
        PrintMes(QSV_LOG_DEBUG, _T("InitMfxEncParams: Adjusted param for Bluray encoding.\n"));
    }

    m_EncExtParams.push_back((mfxExtBuffer *)&m_CodingOption);

    //m_mfxEncParams.mfx.TimeStampCalc = (mfxU16)((pInParams->vpp.nDeinterlace == MFX_DEINTERLACE_IT) ? MFX_TIMESTAMPCALC_TELECINE : MFX_TIMESTAMPCALC_UNKNOWN);
    //m_mfxEncParams.mfx.ExtendedPicStruct = pInParams->nPicStruct;

    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_3) &&
        (pInParams->VideoFormat != list_videoformat[0].value ||
         pInParams->ColorPrim   != list_colorprim[0].value ||
         pInParams->Transfer    != list_transfer[0].value ||
         pInParams->ColorMatrix != list_colormatrix[0].value ||
         pInParams->bFullrange
        ) ) {
#define GET_COLOR_PRM(v, list) (mfxU16)((v == MFX_COLOR_VALUE_AUTO) ? ((pInParams->nDstHeight >= HD_HEIGHT_THRESHOLD) ? list[HD_INDEX].value : list[SD_INDEX].value) : v)
            //色設定 (for API v1.3)
            INIT_MFX_EXT_BUFFER(m_VideoSignalInfo, MFX_EXTBUFF_VIDEO_SIGNAL_INFO);
            m_VideoSignalInfo.ColourDescriptionPresent = 1; //"1"と設定しないと正しく反映されない
            m_VideoSignalInfo.VideoFormat              = pInParams->VideoFormat;
            m_VideoSignalInfo.VideoFullRange           = pInParams->bFullrange != 0;
            m_VideoSignalInfo.ColourPrimaries          = GET_COLOR_PRM(pInParams->ColorPrim,   list_colorprim);
            m_VideoSignalInfo.TransferCharacteristics  = GET_COLOR_PRM(pInParams->Transfer,    list_transfer);
            m_VideoSignalInfo.MatrixCoefficients       = GET_COLOR_PRM(pInParams->ColorMatrix, list_colormatrix);
#undef GET_COLOR_PRM
            m_EncExtParams.push_back((mfxExtBuffer *)&m_VideoSignalInfo);
    }

    //シーンチェンジ検出をこちらで行う場合は、GOP長を最大に設定する
    if (m_nExPrm & MFX_PRM_EX_SCENE_CHANGE)
        m_mfxEncParams.mfx.GopPicSize = USHRT_MAX;

    // set frame size and crops
    // width must be a multiple of 16
    // height must be a multiple of 16 in case of frame picture and a multiple of 32 in case of field picture
    m_mfxEncParams.mfx.FrameInfo.Width  = (mfxU16)MSDK_ALIGN(pInParams->nDstWidth, blocksz);
    m_mfxEncParams.mfx.FrameInfo.Height = (mfxU16)((MFX_PICSTRUCT_PROGRESSIVE == m_mfxEncParams.mfx.FrameInfo.PicStruct)?
        MSDK_ALIGN(pInParams->nDstHeight, blocksz) : MSDK_ALIGN(pInParams->nDstHeight, blocksz * 2));

    m_mfxEncParams.mfx.FrameInfo.CropX = 0;
    m_mfxEncParams.mfx.FrameInfo.CropY = 0;
    m_mfxEncParams.mfx.FrameInfo.CropW = pInParams->nDstWidth;
    m_mfxEncParams.mfx.FrameInfo.CropH = pInParams->nDstHeight;
#if ENABLE_MVC_ENCODING
    // we don't specify profile and level and let the encoder choose those basing on parameters
    // we must specify profile only for MVC codec
    if (MVC_ENABLED & m_MVCflags)
        m_mfxEncParams.mfx.CodecProfile = MFX_PROFILE_AVC_STEREO_HIGH;

    // configure and attach external parameters
    if (MVC_ENABLED & pInParams->MVC_flags)
        m_EncExtParams.push_back((mfxExtBuffer *)&m_MVCSeqDesc);

    if (MVC_VIEWOUTPUT & pInParams->MVC_flags)
    {
        // ViewOuput option requested
        m_CodingOption.ViewOutput = MFX_CODINGOPTION_ON;
        m_EncExtParams.push_back((mfxExtBuffer *)&m_CodingOption);
    }
#endif

    // In case of HEVC when height and/or width divided with 8 but not divided with 16
    // add extended parameter to increase performance
    if ( ( !((m_mfxEncParams.mfx.FrameInfo.CropW & 15 ) ^ 8 ) ||
           !((m_mfxEncParams.mfx.FrameInfo.CropH & 15 ) ^ 8 ) ) &&
             (m_mfxEncParams.mfx.CodecId == MFX_CODEC_HEVC) ) {
        INIT_MFX_EXT_BUFFER(m_ExtHEVCParam, MFX_EXTBUFF_HEVC_PARAM);
        m_ExtHEVCParam.PicWidthInLumaSamples = m_mfxEncParams.mfx.FrameInfo.CropW;
        m_ExtHEVCParam.PicHeightInLumaSamples = m_mfxEncParams.mfx.FrameInfo.CropH;
        m_EncExtParams.push_back((mfxExtBuffer*)&m_ExtHEVCParam);
    }

    if (m_mfxEncParams.mfx.CodecId == MFX_CODEC_VP8) {
        INIT_MFX_EXT_BUFFER(m_ExtVP8CodingOption, MFX_EXTBUFF_VP8_CODING_OPTION);
        m_ExtVP8CodingOption.SharpnessLevel = pInParams->nVP8Sharpness;
        m_EncExtParams.push_back((mfxExtBuffer*)&m_ExtVP8CodingOption);
    }

    if (m_mfxEncParams.mfx.CodecId == MFX_CODEC_HEVC) {
        m_pEncPlugin.reset(LoadPlugin(MFX_PLUGINTYPE_VIDEO_ENCODE, m_mfxSession, MFX_PLUGINID_HEVCE_HW, 1));
        if (m_pEncPlugin.get() == NULL) {
            PrintMes(QSV_LOG_ERROR, _T("Failed to load hw hevc encoder.\n"));
            return MFX_ERR_UNSUPPORTED;
        }
    } else if (m_mfxEncParams.mfx.CodecId == MFX_CODEC_VP8) {
        m_pEncPlugin.reset(LoadPlugin(MFX_PLUGINTYPE_VIDEO_ENCODE, m_mfxSession, MFX_PLUGINID_VP8E_HW, 1));
        if (m_pEncPlugin.get() == NULL) {
            PrintMes(QSV_LOG_ERROR, _T("Failed to load hw vp8 encoder.\n"));
            return MFX_ERR_UNSUPPORTED;
        }
    }

    // JPEG encoder settings overlap with other encoders settings in mfxInfoMFX structure
    if (MFX_CODEC_JPEG == pInParams->CodecId) {
        m_mfxEncParams.mfx.Interleaved = 1;
        m_mfxEncParams.mfx.Quality = pInParams->nQuality;
        m_mfxEncParams.mfx.RestartInterval = 0;
        MSDK_ZERO_MEMORY(m_mfxEncParams.mfx.reserved5);
    }

    if (!m_EncExtParams.empty()) {
        m_mfxEncParams.ExtParam = &m_EncExtParams[0]; // vector is stored linearly in memory
        m_mfxEncParams.NumExtParam = (mfxU16)m_EncExtParams.size();
        for (const auto& extParam : m_EncExtParams) {
            PrintMes(QSV_LOG_DEBUG, _T("InitMfxEncParams: set ext param %s.\n"), fourccToStr(extParam->BufferId).c_str());
        }
    }

    PrintMes(QSV_LOG_DEBUG, _T("InitMfxEncParams: enc input frame %dx%d (%d,%d,%d,%d)\n"),
        m_mfxEncParams.mfx.FrameInfo.Width, m_mfxEncParams.mfx.FrameInfo.Height,
        m_mfxEncParams.mfx.FrameInfo.CropX, m_mfxEncParams.mfx.FrameInfo.CropY, m_mfxEncParams.mfx.FrameInfo.CropW, m_mfxEncParams.mfx.FrameInfo.CropH);
    PrintMes(QSV_LOG_DEBUG, _T("InitMfxEncParams: enc input color format %s, chroma %d, bitdepth %d, shift %d\n"),
        ColorFormatToStr(m_mfxEncParams.mfx.FrameInfo.FourCC), m_mfxEncParams.mfx.FrameInfo.ChromaFormat, m_mfxEncParams.mfx.FrameInfo.BitDepthLuma, m_mfxEncParams.mfx.FrameInfo.Shift);
    PrintMes(QSV_LOG_DEBUG, _T("InitMfxEncParams: set all enc params.\n"));
    return MFX_ERR_NONE;
}

mfxStatus CEncodingPipeline::InitMfxVppParams(sInputParams *pInParams)
{
    const mfxU32 blocksz = (pInParams->CodecId == MFX_CODEC_HEVC) ? 32 : 16;
    mfxU64 availableFeaures = CheckVppFeatures(pInParams->bUseHWLib, m_mfxVer);
#if ENABLE_FPS_CONVERSION
    if (FPS_CONVERT_NONE != pInParams->vpp.nFPSConversion && !(availableFeaures & VPP_FEATURE_FPS_CONVERSION_ADV)) {
        PrintMes(QSV_LOG_WARN, _T("FPS Conversion not supported on this platform, disabled.\n"));
        pInParams->vpp.nFPSConversion = FPS_CONVERT_NONE;
    }
#else
    //現時点ではうまく動いてなさそうなので無効化
    if (FPS_CONVERT_NONE != pInParams->vpp.nFPSConversion) {
        PrintMes(QSV_LOG_WARN, _T("FPS Conversion not supported on this build, disabled.\n"));
        pInParams->vpp.nFPSConversion = FPS_CONVERT_NONE;
    }
#endif

    if (pInParams->vpp.nImageStabilizer && !(availableFeaures & VPP_FEATURE_IMAGE_STABILIZATION)) {
        PrintMes(QSV_LOG_WARN, _T("Image Stabilizer not supported on this platform, disabled.\n"));
        pInParams->vpp.nImageStabilizer = 0;
    }
    
    if ((pInParams->nPicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF))) {
        switch (pInParams->vpp.nDeinterlace) {
        case MFX_DEINTERLACE_IT_MANUAL:
            if (!(availableFeaures & VPP_FEATURE_DEINTERLACE_IT_MANUAL)) {
                PrintMes(QSV_LOG_ERROR, _T("Deinterlace \"it-manual\" is not supported on this platform.\n"));
                return MFX_ERR_INVALID_VIDEO_PARAM;
            }
        case MFX_DEINTERLACE_AUTO_SINGLE:
        case MFX_DEINTERLACE_AUTO_DOUBLE:
            if (!(availableFeaures & VPP_FEATURE_DEINTERLACE_AUTO)) {
                PrintMes(QSV_LOG_ERROR, _T("Deinterlace \"auto\" is not supported on this platform.\n"));
                return MFX_ERR_INVALID_VIDEO_PARAM;
            }
        default:
            break;
        }
    }

    MSDK_CHECK_POINTER(pInParams,  MFX_ERR_NULL_PTR);

    // specify memory type
    if (pInParams->memType != SYSTEM_MEMORY)
        m_mfxVppParams.IOPattern = MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    else
        m_mfxVppParams.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY | MFX_IOPATTERN_OUT_SYSTEM_MEMORY;


    m_mfxVppParams.vpp.In.PicStruct = pInParams->nPicStruct;
    m_mfxVppParams.vpp.In.FrameRateExtN = pInParams->nFPSRate;
    m_mfxVppParams.vpp.In.FrameRateExtD = pInParams->nFPSScale;
    m_mfxVppParams.vpp.In.AspectRatioW  = (mfxU16)pInParams->nPAR[0];
    m_mfxVppParams.vpp.In.AspectRatioH  = (mfxU16)pInParams->nPAR[1];

    mfxFrameInfo inputFrameInfo = { 0 };
    m_pFileReader->GetInputFrameInfo(&inputFrameInfo);
    if (inputFrameInfo.FourCC == 0 || inputFrameInfo.FourCC == MFX_FOURCC_NV12) {
        // input frame info
        m_mfxVppParams.vpp.In.FourCC       = MFX_FOURCC_NV12;
        m_mfxVppParams.vpp.In.ChromaFormat = MFX_CHROMAFORMAT_YUV420;

        // width must be a multiple of 16
        // height must be a multiple of 16 in case of frame picture and a multiple of 32 in case of field picture
        m_mfxVppParams.vpp.In.Width     = (mfxU16)MSDK_ALIGN(pInParams->nWidth, blocksz);
        m_mfxVppParams.vpp.In.Height    = (mfxU16)((MFX_PICSTRUCT_PROGRESSIVE == m_mfxVppParams.vpp.In.PicStruct)?
            MSDK_ALIGN(pInParams->nHeight, blocksz) : MSDK_ALIGN(pInParams->nHeight, blocksz));
    } else {
        m_mfxVppParams.vpp.In.FourCC         = inputFrameInfo.FourCC;
        m_mfxVppParams.vpp.In.ChromaFormat   = inputFrameInfo.ChromaFormat;
        m_mfxVppParams.vpp.In.BitDepthLuma   = pInParams->inputBitDepthLuma;
        m_mfxVppParams.vpp.In.BitDepthChroma = pInParams->inputBitDepthChroma;
        //QSVデコーダは特別にShiftパラメータを使う可能性がある
        if (m_pFileReader->getInputCodec()) {
            m_mfxVppParams.vpp.In.Shift      = inputFrameInfo.Shift;
        }

        // width must be a multiple of 16
        // height must be a multiple of 16 in case of frame picture and a multiple of 32 in case of field picture
        m_mfxVppParams.vpp.In.Width     = (mfxU16)MSDK_ALIGN(inputFrameInfo.CropW, blocksz);
        m_mfxVppParams.vpp.In.Height    = (mfxU16)((MFX_PICSTRUCT_PROGRESSIVE == m_mfxVppParams.vpp.In.PicStruct) ?
            MSDK_ALIGN(inputFrameInfo.CropH, blocksz) : MSDK_ALIGN(inputFrameInfo.CropH, blocksz * 2));
    }

    // set crops in input mfxFrameInfo for correct work of file reader
    // VPP itself ignores crops at initialization
    m_mfxVppParams.vpp.In.CropW = pInParams->nWidth;
    m_mfxVppParams.vpp.In.CropH = pInParams->nHeight;

    //QSVデコードを行う場合、CropはVppで行う
    if (m_pFileReader->getInputCodec()) {
        m_mfxVppParams.vpp.In.CropX = pInParams->sInCrop.left;
        m_mfxVppParams.vpp.In.CropY = pInParams->sInCrop.up;
        m_mfxVppParams.vpp.In.CropW -= (pInParams->sInCrop.left   + pInParams->sInCrop.right);
        m_mfxVppParams.vpp.In.CropH -= (pInParams->sInCrop.bottom + pInParams->sInCrop.up);
        PrintMes(QSV_LOG_DEBUG, _T("InitMfxVppParams: vpp crop enabled.\n"));
    }
    PrintMes(QSV_LOG_DEBUG, _T("InitMfxVppParams: vpp input frame %dx%d (%d,%d,%d,%d)\n"),
        m_mfxVppParams.vpp.In.Width, m_mfxVppParams.vpp.In.Height, m_mfxVppParams.vpp.In.CropX, m_mfxVppParams.vpp.In.CropY, m_mfxVppParams.vpp.In.CropW, m_mfxVppParams.vpp.In.CropH);
    PrintMes(QSV_LOG_DEBUG, _T("InitMfxVppParams: vpp input color format %s, chroma %d, bitdepth %d, shift %d\n"),
        ColorFormatToStr(m_mfxVppParams.vpp.In.FourCC), m_mfxVppParams.vpp.In.ChromaFormat, m_mfxVppParams.vpp.In.BitDepthLuma, m_mfxVppParams.vpp.In.Shift);

    // fill output frame info
    memcpy(&m_mfxVppParams.vpp.Out, &m_mfxVppParams.vpp.In, sizeof(mfxFrameInfo));

    // only resizing is supported
    m_mfxVppParams.vpp.Out.ChromaFormat   = MFX_CHROMAFORMAT_YUV420;
    m_mfxVppParams.vpp.Out.FourCC         = MFX_FOURCC_NV12;
    m_mfxVppParams.vpp.Out.BitDepthLuma   = 0;
    m_mfxVppParams.vpp.Out.BitDepthChroma = 0;
    m_mfxVppParams.vpp.Out.Shift          = 0;
    m_mfxVppParams.vpp.Out.PicStruct = (pInParams->vpp.nDeinterlace) ? MFX_PICSTRUCT_PROGRESSIVE : pInParams->nPicStruct;
    if ((pInParams->nPicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF))) {
        INIT_MFX_EXT_BUFFER(m_ExtDeinterlacing, MFX_EXTBUFF_VPP_DEINTERLACING);
        switch (pInParams->vpp.nDeinterlace) {
        case MFX_DEINTERLACE_NORMAL:
        case MFX_DEINTERLACE_AUTO_SINGLE:
            m_ExtDeinterlacing.Mode = (pInParams->vpp.nDeinterlace == MFX_DEINTERLACE_NORMAL) ? MFX_DEINTERLACING_30FPS_OUT : MFX_DEINTERLACE_AUTO_SINGLE;
            m_nExPrm |= MFX_PRM_EX_DEINT_NORMAL;
            break;
        case MFX_DEINTERLACE_IT:
        case MFX_DEINTERLACE_IT_MANUAL:
            if (pInParams->vpp.nDeinterlace == MFX_DEINTERLACE_IT_MANUAL) {
                m_ExtDeinterlacing.Mode = MFX_DEINTERLACING_FIXED_TELECINE_PATTERN;
                m_ExtDeinterlacing.TelecinePattern = pInParams->vpp.nTelecinePattern;
            } else {
                m_ExtDeinterlacing.Mode = MFX_DEINTERLACING_24FPS_OUT;
            }
            m_mfxVppParams.vpp.Out.FrameRateExtN = (m_mfxVppParams.vpp.Out.FrameRateExtN * 4) / 5;
            break;
        case MFX_DEINTERLACE_BOB:
        case MFX_DEINTERLACE_AUTO_DOUBLE:
            m_ExtDeinterlacing.Mode = (pInParams->vpp.nDeinterlace == MFX_DEINTERLACE_BOB) ? MFX_DEINTERLACING_BOB : MFX_DEINTERLACE_AUTO_DOUBLE;
            m_mfxVppParams.vpp.Out.FrameRateExtN = m_mfxVppParams.vpp.Out.FrameRateExtN * 2;
            m_nExPrm |= MFX_PRM_EX_DEINT_BOB;
            break;
        case MFX_DEINTERLACE_NONE:
        default:
            break;
        }
        if (pInParams->vpp.nDeinterlace != MFX_DEINTERLACE_NONE) {
#if ENABLE_ADVANCED_DEINTERLACE
            if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_13)) {
                m_VppExtParams.push_back((mfxExtBuffer *)&m_ExtDeinterlacing);
                m_VppDoUseList.push_back(MFX_EXTBUFF_VPP_DEINTERLACING);
            }
#endif
            VppExtMes += _T("Deinterlace (");
            VppExtMes += list_deinterlace[get_cx_index(list_deinterlace, pInParams->vpp.nDeinterlace)].desc;
            if (pInParams->vpp.nDeinterlace == MFX_DEINTERLACE_IT_MANUAL) {
                VppExtMes += _T(", ");
                VppExtMes += list_telecine_patterns[get_cx_index(list_telecine_patterns, pInParams->vpp.nTelecinePattern)].desc;
            }
            VppExtMes += _T(")\n");
            PrintMes(QSV_LOG_DEBUG, _T("InitMfxVppParams: vpp deinterlace enabled.\n"));
        }
        pInParams->vpp.nFPSConversion = FPS_CONVERT_NONE;
    } else {
        switch (pInParams->vpp.nFPSConversion) {
        case FPS_CONVERT_MUL2:
            m_mfxVppParams.vpp.Out.FrameRateExtN = m_mfxVppParams.vpp.Out.FrameRateExtN * 2;
            break;
        case FPS_CONVERT_MUL2_5:
            m_mfxVppParams.vpp.Out.FrameRateExtN = m_mfxVppParams.vpp.Out.FrameRateExtN * 5 / 2;
            break;
        default:
            break;
        }
    }
    m_mfxVppParams.vpp.Out.CropX = 0;
    m_mfxVppParams.vpp.Out.CropY = 0;
    m_mfxVppParams.vpp.Out.CropW = pInParams->nDstWidth;
    m_mfxVppParams.vpp.Out.CropH = pInParams->nDstHeight;
    m_mfxVppParams.vpp.Out.Width = (mfxU16)MSDK_ALIGN(pInParams->nDstWidth, blocksz);
    m_mfxVppParams.vpp.Out.Height = (mfxU16)((MFX_PICSTRUCT_PROGRESSIVE == m_mfxVppParams.vpp.Out.PicStruct)?
        MSDK_ALIGN(pInParams->nDstHeight, blocksz) : MSDK_ALIGN(pInParams->nDstHeight, blocksz));
    
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)
        && (   MFX_FOURCC_RGB3 == m_mfxVppParams.vpp.In.FourCC
            || MFX_FOURCC_RGB4 == m_mfxVppParams.vpp.In.FourCC)) {
        
        INIT_MFX_EXT_BUFFER(m_ExtVppVSI, MFX_EXTBUFF_VPP_VIDEO_SIGNAL_INFO);
        m_ExtVppVSI.In.NominalRange    = MFX_NOMINALRANGE_0_255;
        m_ExtVppVSI.In.TransferMatrix  = MFX_TRANSFERMATRIX_UNKNOWN;
        m_ExtVppVSI.Out.NominalRange   = (mfxU16)((pInParams->bFullrange) ? MFX_NOMINALRANGE_0_255 : MFX_NOMINALRANGE_16_235);
        m_ExtVppVSI.Out.TransferMatrix = MFX_TRANSFERMATRIX_UNKNOWN;
        if (pInParams->ColorMatrix == get_cx_index(list_colormatrix, _T("bt709"))) {
            m_ExtVppVSI.Out.TransferMatrix = MFX_TRANSFERMATRIX_BT709;
        } else if (pInParams->ColorMatrix == get_cx_index(list_colormatrix, _T("bt601"))) {
            m_ExtVppVSI.Out.TransferMatrix = MFX_TRANSFERMATRIX_BT601;
        }
        m_VppExtParams.push_back((mfxExtBuffer *)&m_ExtVppVSI);
        m_VppDoUseList.push_back(MFX_EXTBUFF_VPP_VIDEO_SIGNAL_INFO);
        PrintMes(QSV_LOG_DEBUG, _T("InitMfxVppParams: vpp colorspace conversion enabled.\n"));
    }
    PrintMes(QSV_LOG_DEBUG, _T("InitMfxVppParams: vpp output frame %dx%d (%d,%d,%d,%d)\n"),
        m_mfxVppParams.vpp.Out.Width, m_mfxVppParams.vpp.Out.Height, m_mfxVppParams.vpp.Out.CropX, m_mfxVppParams.vpp.Out.CropY, m_mfxVppParams.vpp.Out.CropW, m_mfxVppParams.vpp.Out.CropH);
    PrintMes(QSV_LOG_DEBUG, _T("InitMfxVppParams: vpp output color format %s, chroma %d, bitdepth %d, shift %d\n"),
        ColorFormatToStr(m_mfxVppParams.vpp.Out.FourCC), m_mfxVppParams.vpp.Out.ChromaFormat, m_mfxVppParams.vpp.Out.BitDepthLuma, m_mfxVppParams.vpp.Out.Shift);

    // configure and attach external parameters
    //AllocAndInitVppDoNotUse();
    //m_VppExtParams.push_back((mfxExtBuffer *)&m_VppDoNotUse);
#if ENABLE_MVC_ENCODING
    if (pInParams->bIsMVC)
        m_VppExtParams.push_back((mfxExtBuffer *)&m_MVCSeqDesc);
#endif
    PrintMes(QSV_LOG_DEBUG, _T("InitMfxVppParams: set all vpp params.\n"));
    return MFX_ERR_NONE;
}

mfxStatus CEncodingPipeline::CreateVppExtBuffers(sInputParams *pParams)
{
    m_VppDoNotUseList.push_back(MFX_EXTBUFF_VPP_PROCAMP);
    auto vppExtAddMes = [this](tstring str) {
        VppExtMes += str;
        PrintMes(QSV_LOG_DEBUG, _T("CreateVppExtBuffers: %s"), str.c_str());
    };

    if (FPS_CONVERT_NONE != pParams->vpp.nFPSConversion) {
        INIT_MFX_EXT_BUFFER(m_ExtFrameRateConv, MFX_EXTBUFF_VPP_FRAME_RATE_CONVERSION);
        m_ExtFrameRateConv.Algorithm = MFX_FRCALGM_FRAME_INTERPOLATION;
        m_VppExtParams.push_back((mfxExtBuffer*)&m_ExtFrameRateConv);

        vppExtAddMes(_T("fps conversion with interpolation\n"));
        m_VppDoUseList.push_back(MFX_EXTBUFF_VPP_FRAME_RATE_CONVERSION);
    }

    if (pParams->vpp.bUseDenoise) {
        INIT_MFX_EXT_BUFFER(m_ExtDenoise, MFX_EXTBUFF_VPP_DENOISE);
        m_ExtDenoise.DenoiseFactor  = pParams->vpp.nDenoise;
        m_VppExtParams.push_back((mfxExtBuffer*)&m_ExtDenoise);

        vppExtAddMes(strsprintf(_T("Denoise, strength %d\n"), m_ExtDenoise.DenoiseFactor));
        m_VppDoUseList.push_back(MFX_EXTBUFF_VPP_DENOISE);
    } else {
        m_VppDoNotUseList.push_back(MFX_EXTBUFF_VPP_DENOISE);
    }

    if (pParams->vpp.nImageStabilizer) {
        INIT_MFX_EXT_BUFFER(m_ExtImageStab, MFX_EXTBUFF_VPP_IMAGE_STABILIZATION);
        m_ExtImageStab.Mode = pParams->vpp.nImageStabilizer;
        m_VppExtParams.push_back((mfxExtBuffer*)&m_ExtImageStab);
        
        vppExtAddMes(strsprintf(_T("Stabilizer, mode %s\n"), get_vpp_image_stab_mode_str(m_ExtImageStab.Mode)));
        m_VppDoUseList.push_back(MFX_EXTBUFF_VPP_IMAGE_STABILIZATION);
    }

    if (pParams->vpp.bUseDetailEnhance) {
        INIT_MFX_EXT_BUFFER(m_ExtDetail, MFX_EXTBUFF_VPP_DETAIL);
        m_ExtDetail.DetailFactor = pParams->vpp.nDetailEnhance;
        m_VppExtParams.push_back((mfxExtBuffer*)&m_ExtDetail);
        
        vppExtAddMes(strsprintf(_T("Detail Enhancer, strength %d\n"), m_ExtDetail.DetailFactor));
    } else {
        m_VppDoNotUseList.push_back(MFX_EXTBUFF_VPP_DETAIL);
    }

    m_VppDoNotUseList.push_back(MFX_EXTBUFF_VPP_SCENE_ANALYSIS);

    if (   check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_3)
        && (pParams->nPicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF))) {
            switch (pParams->vpp.nDeinterlace) {
            case MFX_DEINTERLACE_IT:
            case MFX_DEINTERLACE_IT_MANUAL:
            case MFX_DEINTERLACE_BOB:
            case MFX_DEINTERLACE_AUTO_DOUBLE:
                INIT_MFX_EXT_BUFFER(m_ExtFrameRateConv, MFX_EXTBUFF_VPP_FRAME_RATE_CONVERSION);
                m_ExtFrameRateConv.Algorithm = MFX_FRCALGM_DISTRIBUTED_TIMESTAMP;

                m_VppDoUseList.push_back(MFX_EXTBUFF_VPP_FRAME_RATE_CONVERSION);
                break;
            default:
                break;
            }
    }

    if (m_VppDoUseList.size()) {
        INIT_MFX_EXT_BUFFER(m_VppDoUse, MFX_EXTBUFF_VPP_DOUSE);
        m_VppDoUse.NumAlg = (mfxU32)m_VppDoUseList.size();
        m_VppDoUse.AlgList = &m_VppDoUseList[0];

        m_VppExtParams.push_back((mfxExtBuffer *)&m_VppDoUse);
        for (const auto& extParam : m_VppDoUseList) {
            PrintMes(QSV_LOG_DEBUG, _T("CreateVppExtBuffers: set DoUse %s.\n"), fourccToStr(extParam).c_str());
        }
    }

    if (m_VppDoNotUseList.size()) {
        AllocAndInitVppDoNotUse();
        m_VppExtParams.push_back((mfxExtBuffer *)&m_VppDoNotUse);
        for (const auto& extParam : m_VppDoNotUseList) {
            PrintMes(QSV_LOG_DEBUG, _T("CreateVppExtBuffers: set DoNotUse %s.\n"), fourccToStr(extParam).c_str());
        }
    }

    m_mfxVppParams.ExtParam = &m_VppExtParams[0]; // vector is stored linearly in memory
    m_mfxVppParams.NumExtParam = (mfxU16)m_VppExtParams.size();

    return MFX_ERR_NONE;
}

#pragma warning (push)
#pragma warning (disable: 4100)
mfxStatus CEncodingPipeline::InitVppPrePlugins(sInputParams *pParams) {
    mfxStatus sts = MFX_ERR_NONE;
#if ENABLE_CUSTOM_VPP
    tstring vppPreMes = _T("");
    if (pParams->vpp.delogo.pFilePath) {
        unique_ptr<CVPPPlugin> filter(new CVPPPlugin());
        DelogoParam param(m_pMFXAllocator, m_memType, pParams->vpp.delogo.pFilePath, pParams->vpp.delogo.pSelect, pParams->strSrcFile,
            pParams->vpp.delogo.nPosOffset.x, pParams->vpp.delogo.nPosOffset.y, pParams->vpp.delogo.nDepth,
            pParams->vpp.delogo.nYOffset, pParams->vpp.delogo.nCbOffset, pParams->vpp.delogo.nCrOffset);
        sts = filter->Init(m_mfxVer, _T("delogo"), &param, sizeof(param), pParams->bUseHWLib, m_memType, m_hwdev, m_pMFXAllocator, 3, m_mfxVppParams.vpp.In, m_mfxVppParams.IOPattern, m_pQSVLog.get());
        if (sts == MFX_ERR_ABORTED) {
            PrintMes(QSV_LOG_WARN, _T("%s\n"), filter->getMessage().c_str());
            sts = MFX_ERR_NONE;
        } else if (sts != MFX_ERR_NONE) {
            PrintMes(QSV_LOG_ERROR, _T("%s\n"), filter->getMessage().c_str());
        } else {
            sts = MFXJoinSession(m_mfxSession, filter->getSession());
            MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to join vpp pre filter session."));
            tstring mes = filter->getMessage();
            PrintMes(QSV_LOG_DEBUG, _T("InitVppPrePlugins: add filter: %s\n"), mes.c_str());
            vppPreMes += mes;
            m_VppPrePlugins.push_back(std::move(filter));
        }
    }
    if (pParams->vpp.bHalfTurn) {
        unique_ptr<CVPPPlugin> filter(new CVPPPlugin());
        RotateParam param(180);
        sts = filter->Init(m_mfxVer, _T("rotate"), &param, sizeof(param), pParams->bUseHWLib, m_memType, m_hwdev, m_pMFXAllocator, 3, m_mfxVppParams.vpp.In, m_mfxVppParams.IOPattern, m_pQSVLog.get());
        if (sts != MFX_ERR_NONE) {
            PrintMes(QSV_LOG_ERROR, _T("%s\n"), filter->getMessage().c_str());
        } else {
            sts = MFXJoinSession(m_mfxSession, filter->getSession());
            MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to join vpp pre filter session."));
            tstring mes = filter->getMessage();
            PrintMes(QSV_LOG_DEBUG, _T("InitVppPrePlugins: add filter: %s\n"), mes.c_str());
            vppPreMes += mes;
            m_VppPrePlugins.push_back(std::move(filter));
        }
    }
    VppExtMes = vppPreMes + VppExtMes;
#endif
    return sts;
}

mfxStatus CEncodingPipeline::InitVppPostPlugins(sInputParams *pParams) {
    return MFX_ERR_NONE;
}
#pragma warning (pop)

//void CEncodingPipeline::DeleteVppExtBuffers()
//{
//    //free external buffers
//    if (m_ppVppExtBuffers)
//    {
//        for (mfxU8 i = 0; i < m_nNumVppExtBuffers; i++)
//        {
//            mfxExtVPPDoNotUse* pExtDoNotUse = (mfxExtVPPDoNotUse* )(m_ppVppExtBuffers[i]);
//            SAFE_DELETE_ARRAY(pExtDoNotUse->AlgList);
//            SAFE_DELETE(m_ppVppExtBuffers[i]);
//        }
//    }
//
//    SAFE_DELETE_ARRAY(m_ppVppExtBuffers);
//}

mfxStatus CEncodingPipeline::CreateHWDevice()
{
    mfxStatus sts = MFX_ERR_NONE;
#if D3D_SURFACES_SUPPORT
    POINT point = {0, 0};
    HWND window = WindowFromPoint(point);
    m_hwdev = NULL;

    if (m_memType) {
#if MFX_D3D11_SUPPORT
        if (m_memType == D3D11_MEMORY
            && NULL != (m_hwdev = new CD3D11Device())) {
            m_memType = D3D11_MEMORY;
            PrintMes(QSV_LOG_DEBUG, _T("HWDevice: d3d11 - initializing...\n"));

            sts = m_hwdev->Init(
                NULL,
                0,
                MSDKAdapter::GetNumber(m_mfxSession));
            if (sts != MFX_ERR_NONE) {
                m_hwdev->Close();
                delete m_hwdev;
                m_hwdev = NULL;
                PrintMes(QSV_LOG_DEBUG, _T("HWDevice: d3d11 - initializing failed.\n"));
            }
        }
#endif // #if MFX_D3D11_SUPPORT
        if (m_hwdev == NULL && NULL != (m_hwdev = new CD3D9Device())) {
            //もし、d3d11要求で失敗したら自動的にd3d9に切り替える
            //sessionごと切り替える必要がある
            if (m_memType != D3D9_MEMORY) {
                PrintMes(QSV_LOG_DEBUG, _T("Retry openning device, chaging to d3d9 mode, re-init session.\n"));
                InitSession(true, D3D9_MEMORY);
                m_memType = m_memType;
            }

            PrintMes(QSV_LOG_DEBUG, _T("HWDevice: d3d9 - initializing...\n"));
            sts = m_hwdev->Init(
                window,
                0,
                MSDKAdapter::GetNumber(m_mfxSession));
        }
    }
    MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to initialize HW Device."));
    PrintMes(QSV_LOG_DEBUG, _T("HWDevice: initializing success.\n"));
    
#elif LIBVA_SUPPORT
    m_hwdev = CreateVAAPIDevice();
    if (NULL == m_hwdev)
    {
        return MFX_ERR_MEMORY_ALLOC;
    }
    sts = m_hwdev->Init(NULL, 0, MSDKAdapter::GetNumber(m_mfxSession));
    MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to initialize HW Device."));
#endif
    return MFX_ERR_NONE;
}

mfxStatus CEncodingPipeline::ResetDevice()
{
    if (m_memType & (D3D9_MEMORY | D3D11_MEMORY)) {
        PrintMes(QSV_LOG_DEBUG, _T("HWDevice: reset.\n"));
        return m_hwdev->Reset();
    }
    return MFX_ERR_NONE;
}

mfxStatus CEncodingPipeline::AllocFrames() {
    mfxStatus sts = MFX_ERR_NONE;
    mfxFrameAllocRequest DecRequest;
    mfxFrameAllocRequest EncRequest;
    mfxFrameAllocRequest VppRequest[2];
    mfxFrameAllocRequest NextRequest; //出力されてくるフレーム情報とフレームタイプを記録する

    mfxU16 nEncSurfNum = 0; // number of surfaces for encoder
    mfxU16 nVppSurfNum = 0; // number of surfaces for vpp

    mfxU16 nInputSurfAdd = 0;
    mfxU16 nDecSurfAdd = 0; // number of surfaces for decoder
    mfxU16 nVppPreSurfAdd = 0; // number of surfaces for vpp pre
    mfxU16 nVppSurfAdd = 0;
    mfxU16 nVppPostSurfAdd = 0; // number of surfaces for vpp post

    MSDK_ZERO_MEMORY(DecRequest);
    MSDK_ZERO_MEMORY(EncRequest);
    MSDK_ZERO_MEMORY(VppRequest[0]);
    MSDK_ZERO_MEMORY(VppRequest[1]);
    for (const auto& filter : m_VppPrePlugins) {
        MSDK_ZERO_MEMORY(filter->m_PluginResponse);
    }
    for (const auto& filter : m_VppPostPlugins) {
        MSDK_ZERO_MEMORY(filter->m_PluginResponse);
    }
    MSDK_ZERO_MEMORY(NextRequest);
    
    PrintMes(QSV_LOG_DEBUG, _T("AllocFrames: m_nAsyncDepth - %d frames\n"), m_nAsyncDepth);

    // Calculate the number of surfaces for components.
    // QueryIOSurf functions tell how many surfaces are required to produce at least 1 output.
    // To achieve better performance we provide extra surfaces.
    // 1 extra surface at input allows to get 1 extra output.
    
    if (m_pmfxENC) {
        sts = m_pmfxENC->QueryIOSurf(&m_mfxEncParams, &EncRequest);
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to get required buffer size for encoder."));
        PrintMes(QSV_LOG_DEBUG, _T("AllocFrames: Enc query - %d frames\n"), EncRequest.NumFrameSuggested);
    }

    if (m_pmfxVPP) {
        // VppRequest[0] for input frames request, VppRequest[1] for output frames request
        sts = m_pmfxVPP->QueryIOSurf(&m_mfxVppParams, VppRequest);
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to get required buffer size for vpp."));
        PrintMes(QSV_LOG_DEBUG, _T("AllocFrames: Vpp query[0] - %d frames\n"), VppRequest[0].NumFrameSuggested);
        PrintMes(QSV_LOG_DEBUG, _T("AllocFrames: Vpp query[1] - %d frames\n"), VppRequest[1].NumFrameSuggested);
    }

    if (m_pmfxDEC) {
        sts = m_pmfxDEC->QueryIOSurf(&m_mfxDecParams, &DecRequest);
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to get required buffer size for decoder."));
        PrintMes(QSV_LOG_DEBUG, _T("AllocFrames: Dec query - %d frames\n"), DecRequest.NumFrameSuggested);
    }

    nInputSurfAdd = MSDK_MAX(m_EncThread.m_nFrameBuffer, 1);

    nDecSurfAdd = DecRequest.NumFrameSuggested;

    // The number of surfaces shared by vpp output and encode input.
    // When surfaces are shared 1 surface at first component output contains output frame that goes to next component input
    nEncSurfNum = EncRequest.NumFrameSuggested + (m_nAsyncDepth - 1);

    // The number of surfaces for vpp input - so that vpp can work at async depth = m_nAsyncDepth
    nVppSurfNum = VppRequest[0].NumFrameSuggested + (m_nAsyncDepth - 1);
    
    PrintMes(QSV_LOG_DEBUG, _T("AllocFrames: nInputSurfAdd %d frames\n"), nInputSurfAdd);
    PrintMes(QSV_LOG_DEBUG, _T("AllocFrames: nDecSurfAdd   %d frames\n"), nDecSurfAdd);

    if (m_pmfxDEC) {
        NextRequest = DecRequest;
    }

    //VppPrePlugins
    if (m_VppPrePlugins.size()) {
        for (mfxU32 i = 0; i < (mfxU32)m_VppPrePlugins.size(); i++) {
            mfxU16 mem_type = (mfxU16)((HW_MEMORY & m_memType) ? MFX_MEMTYPE_EXTERNAL_FRAME : MFX_MEMTYPE_SYSTEM_MEMORY);
            m_VppPrePlugins[i]->m_nSurfNum += m_nAsyncDepth;
            if (i == 0) {
                mem_type |= (nDecSurfAdd) ? (MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET | MFX_MEMTYPE_FROM_DECODE) : (MFX_MEMTYPE_VIDEO_MEMORY_PROCESSOR_TARGET | MFX_MEMTYPE_FROM_VPPOUT);
                m_VppPrePlugins[i]->m_nSurfNum += MSDK_MAX(1, nInputSurfAdd + nDecSurfAdd - m_nAsyncDepth + 1);
            } else {
                // If surfaces are shared by 2 components, c1 and c2. NumSurf = c1_out + c2_in - AsyncDepth + 1
                mem_type |= MFX_MEMTYPE_VIDEO_MEMORY_PROCESSOR_TARGET | MFX_MEMTYPE_FROM_VPPOUT;
                m_VppPrePlugins[i]->m_nSurfNum += m_VppPrePlugins[i-1]->m_nSurfNum - m_nAsyncDepth + 1;
            }
            m_VppPrePlugins[i]->m_PluginRequest.Type = mem_type;
            m_VppPrePlugins[i]->m_PluginRequest.NumFrameMin = m_VppPrePlugins[i]->m_nSurfNum;
            m_VppPrePlugins[i]->m_PluginRequest.NumFrameSuggested = m_VppPrePlugins[i]->m_nSurfNum;
            MSDK_MEMCPY_VAR(m_VppPrePlugins[i]->m_PluginRequest.Info, &(m_VppPrePlugins[i]->m_pluginVideoParams.mfx.FrameInfo), sizeof(mfxFrameInfo));
            if (m_pmfxDEC && nDecSurfAdd) {
                m_VppPrePlugins[i]->m_PluginRequest.Info.Width  = DecRequest.Info.Width;
                m_VppPrePlugins[i]->m_PluginRequest.Info.Height = DecRequest.Info.Height;
                m_VppPrePlugins[i]->m_pluginVideoParams.mfx.FrameInfo.Width  = DecRequest.Info.Width;
                m_VppPrePlugins[i]->m_pluginVideoParams.mfx.FrameInfo.Height = DecRequest.Info.Height;
            }
            NextRequest = m_VppPrePlugins[i]->m_PluginRequest;
            MSDK_MEMCPY_VAR(NextRequest.Info, &(m_VppPrePlugins[i]->m_pluginVideoParams.vpp.Out), sizeof(mfxFrameInfo));
            PrintMes(QSV_LOG_DEBUG, _T("AllocFrames: PrePlugins[%d] %s, type: %s, %dx%d [%d,%d,%d,%d], request %d frames\n"),
                i, m_VppPrePlugins[i]->getFilterName().c_str(), qsv_memtype_str(mem_type).c_str(),
                m_VppPrePlugins[i]->m_PluginRequest.Info.Width, m_VppPrePlugins[i]->m_PluginRequest.Info.Height,
                m_VppPrePlugins[i]->m_PluginRequest.Info.CropX, m_VppPrePlugins[i]->m_PluginRequest.Info.CropY,
                m_VppPrePlugins[i]->m_PluginRequest.Info.CropW, m_VppPrePlugins[i]->m_PluginRequest.Info.CropH,
                m_VppPrePlugins[i]->m_PluginRequest.NumFrameSuggested);
        }

        //後始末
        nDecSurfAdd = 0;
        nInputSurfAdd = 0;
        nVppPreSurfAdd = m_VppPrePlugins.back()->m_nSurfNum;
    }

    //Vpp
    if (m_pmfxVPP) {
        nVppSurfNum += MSDK_MAX(1, nInputSurfAdd + nDecSurfAdd + nVppPreSurfAdd - m_nAsyncDepth + 1);

        //VppRequest[0]の準備
        VppRequest[0].NumFrameMin = nVppSurfNum;
        VppRequest[0].NumFrameSuggested = nVppSurfNum;
        MSDK_MEMCPY_VAR(VppRequest[0].Info, &(m_mfxVppParams.mfx.FrameInfo), sizeof(mfxFrameInfo));
        if (m_pmfxDEC && nDecSurfAdd) {
            VppRequest[0].Type = DecRequest.Type;
            VppRequest[0].Info.Width  = DecRequest.Info.Width;
            VppRequest[0].Info.Height = DecRequest.Info.Height;
            m_mfxVppParams.mfx.FrameInfo.Width = DecRequest.Info.Width;
            m_mfxVppParams.mfx.FrameInfo.Height = DecRequest.Info.Height;
            //フレームのリクエストを出す時点でCropの値を入れておくと、
            //DecFrameAsyncでMFX_ERR_UNDEFINED_BEHAVIORを出してしまう
            //Cropの値はVppFrameAsyncの直前に渡すようにする
            VppRequest[0].Info.CropX = DecRequest.Info.CropX;
            VppRequest[0].Info.CropY = DecRequest.Info.CropY;
            VppRequest[0].Info.CropW = DecRequest.Info.CropW;
            VppRequest[0].Info.CropH = DecRequest.Info.CropH;
        }

        //後始末
        nInputSurfAdd = 0;
        nDecSurfAdd = 0;
        nVppPreSurfAdd = 0;
        nVppSurfAdd = MSDK_MAX(VppRequest[1].NumFrameSuggested, 1);
        NextRequest = VppRequest[1];
        MSDK_MEMCPY_VAR(NextRequest.Info, &(m_mfxVppParams.vpp.Out), sizeof(mfxFrameInfo));
        PrintMes(QSV_LOG_DEBUG, _T("AllocFrames: Vpp type: %s, %dx%d [%d,%d,%d,%d], request %d frames\n"),
            qsv_memtype_str(VppRequest[0].Type).c_str(),
            VppRequest[0].Info.Width, VppRequest[0].Info.Height,
            VppRequest[0].Info.CropX, VppRequest[0].Info.CropY, VppRequest[0].Info.CropW, VppRequest[0].Info.CropH, VppRequest[0].NumFrameSuggested);
    }

    //VppPostPlugins
    if (m_VppPostPlugins.size()) {
        for (mfxU32 i = 0; i < (mfxU32)m_VppPostPlugins.size(); i++) {
            mfxU16 mem_type = (mfxU16)((HW_MEMORY & m_memType) ? MFX_MEMTYPE_EXTERNAL_FRAME : MFX_MEMTYPE_SYSTEM_MEMORY);
            m_VppPostPlugins[i]->m_nSurfNum += m_nAsyncDepth;
            if (i == 0) {
                mem_type |= (nDecSurfAdd) ? (MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET | MFX_MEMTYPE_FROM_DECODE) : (MFX_MEMTYPE_VIDEO_MEMORY_PROCESSOR_TARGET | MFX_MEMTYPE_FROM_VPPOUT);
                m_VppPostPlugins[i]->m_nSurfNum += MSDK_MAX(1, nInputSurfAdd + nDecSurfAdd + nVppPreSurfAdd + nVppSurfAdd - m_nAsyncDepth + 1);
            } else {
                // If surfaces are shared by 2 components, c1 and c2. NumSurf = c1_out + c2_in - AsyncDepth + 1
                mem_type |= MFX_MEMTYPE_VIDEO_MEMORY_PROCESSOR_TARGET | MFX_MEMTYPE_FROM_VPPOUT;
                m_VppPostPlugins[i]->m_nSurfNum += m_VppPostPlugins[i-1]->m_nSurfNum - m_nAsyncDepth + 1;
            }
            m_VppPostPlugins[i]->m_PluginRequest.Type = mem_type;
            m_VppPostPlugins[i]->m_PluginRequest.NumFrameMin = m_VppPostPlugins[i]->m_nSurfNum;
            m_VppPostPlugins[i]->m_PluginRequest.NumFrameSuggested = m_VppPostPlugins[i]->m_nSurfNum;
            MSDK_MEMCPY_VAR(m_VppPostPlugins[i]->m_PluginRequest.Info, &(m_VppPostPlugins[i]->m_pluginVideoParams.mfx.FrameInfo), sizeof(mfxFrameInfo));
            if (m_pmfxDEC && nDecSurfAdd) {
                m_VppPostPlugins[i]->m_PluginRequest.Type = DecRequest.Type;
                m_VppPostPlugins[i]->m_PluginRequest.Info.Width  = DecRequest.Info.Width;
                m_VppPostPlugins[i]->m_PluginRequest.Info.Height = DecRequest.Info.Height;
                m_VppPostPlugins[i]->m_pluginVideoParams.mfx.FrameInfo.Width  = DecRequest.Info.Width;
                m_VppPostPlugins[i]->m_pluginVideoParams.mfx.FrameInfo.Height = DecRequest.Info.Height;
            }
            NextRequest = m_VppPostPlugins[i]->m_PluginRequest;
            MSDK_MEMCPY_VAR(NextRequest.Info, &(m_VppPostPlugins[i]->m_pluginVideoParams.vpp.Out), sizeof(mfxFrameInfo));
            PrintMes(QSV_LOG_DEBUG, _T("AllocFrames: PostPlugins[%d] %s, type: %s, %dx%d [%d,%d,%d,%d], request %d frames\n"),
                i, m_VppPostPlugins[i]->getFilterName().c_str(), qsv_memtype_str(mem_type).c_str(),
                m_VppPostPlugins[i]->m_PluginRequest.Info.Width, m_VppPostPlugins[i]->m_PluginRequest.Info.Height,
                m_VppPostPlugins[i]->m_PluginRequest.Info.CropX, m_VppPostPlugins[i]->m_PluginRequest.Info.CropY,
                m_VppPostPlugins[i]->m_PluginRequest.Info.CropW, m_VppPostPlugins[i]->m_PluginRequest.Info.CropH,
                m_VppPostPlugins[i]->m_PluginRequest.NumFrameSuggested);
        }

        //後始末
        nInputSurfAdd = 0;
        nDecSurfAdd = 0;
        nVppPreSurfAdd = 0;
        nVppSurfAdd = 0;
        nVppPostSurfAdd = m_VppPostPlugins.back()->m_nSurfNum;
    }

    //Enc、エンコーダが有効でない場合は出力フレーム
    {
        nEncSurfNum += MSDK_MAX(1, nInputSurfAdd + nDecSurfAdd + nVppPreSurfAdd + nVppSurfAdd + nVppPostSurfAdd - m_nAsyncDepth + 1);
        if (m_pmfxENC == nullptr) {
            EncRequest = NextRequest;
            nEncSurfNum += (m_nAsyncDepth - 1);
        } else {
            MSDK_MEMCPY_VAR(EncRequest.Info, &(m_mfxEncParams.mfx.FrameInfo), sizeof(mfxFrameInfo));
        }
        EncRequest.NumFrameMin = nEncSurfNum;
        EncRequest.NumFrameSuggested = nEncSurfNum;
        if (m_pmfxDEC && nDecSurfAdd) {
            EncRequest.Type |= MFX_MEMTYPE_FROM_DECODE;
            EncRequest.Info.Width = DecRequest.Info.Width;
            EncRequest.Info.Height = DecRequest.Info.Height;
        }
        if (nVppPreSurfAdd || nVppSurfAdd || nVppPostSurfAdd) {
            EncRequest.Type |= MFX_MEMTYPE_FROM_VPPOUT; // surfaces are shared between vpp output and encode input
        }

        //後始末
        nInputSurfAdd = 0;
        nDecSurfAdd = 0;
        nVppPreSurfAdd = 0;
        nVppSurfAdd = 0;
        nVppPostSurfAdd = 0;
        PrintMes(QSV_LOG_DEBUG, _T("AllocFrames: %s type: %s, %dx%d [%d,%d,%d,%d], request %d frames\n"),
            (m_pmfxENC) ? _T("Enc") : _T("Out"),
            qsv_memtype_str(EncRequest.Type).c_str(),
            EncRequest.Info.Width, EncRequest.Info.Height,
            EncRequest.Info.CropX, EncRequest.Info.CropY, EncRequest.Info.CropW, EncRequest.Info.CropH, EncRequest.NumFrameSuggested);
    }

    // alloc frames for encoder or output
    sts = m_pMFXAllocator->Alloc(m_pMFXAllocator->pthis, &EncRequest, &m_EncResponse);
    MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to allocate frames for encoder."));
    PrintMes(QSV_LOG_DEBUG, _T("AllocFrames: Allocated EncRequest\n"));

    // alloc frames for vpp if vpp is enabled
    if (m_pmfxVPP) {
        sts = m_pMFXAllocator->Alloc(m_pMFXAllocator->pthis, &(VppRequest[0]), &m_VppResponse);
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to allocate frames for vpp."));
        PrintMes(QSV_LOG_DEBUG, _T("AllocFrames: Allocated VppRequest\n"));
    }

    // prepare mfxFrameSurface1 array for encoder or output
    m_pEncSurfaces = new mfxFrameSurface1[m_EncResponse.NumFrameActual];
    MSDK_CHECK_POINTER(m_pEncSurfaces, MFX_ERR_MEMORY_ALLOC);

    for (int i = 0; i < m_EncResponse.NumFrameActual; i++) {
        memset(&(m_pEncSurfaces[i]), 0, sizeof(mfxFrameSurface1));
        MSDK_MEMCPY_VAR(m_pEncSurfaces[i].Info, &(EncRequest.Info), sizeof(mfxFrameInfo));

        if (m_bExternalAlloc) {
            m_pEncSurfaces[i].Data.MemId = m_EncResponse.mids[i];
        } else {
            // get YUV pointers
            sts = m_pMFXAllocator->Lock(m_pMFXAllocator->pthis, m_EncResponse.mids[i], &(m_pEncSurfaces[i].Data));
            MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to allocate surfaces for encoder."));
        }
    }

    // prepare mfxFrameSurface1 array for vpp if vpp is enabled
    if (m_pmfxVPP) {
        m_pVppSurfaces = new mfxFrameSurface1 [m_VppResponse.NumFrameActual];
        MSDK_CHECK_POINTER(m_pVppSurfaces, MFX_ERR_MEMORY_ALLOC);

        for (int i = 0; i < m_VppResponse.NumFrameActual; i++) {
            MSDK_ZERO_MEMORY(m_pVppSurfaces[i]);
            MSDK_MEMCPY_VAR(m_pVppSurfaces[i].Info, &(VppRequest[0].Info), sizeof(mfxFrameInfo));

            if (m_bExternalAlloc) {
                m_pVppSurfaces[i].Data.MemId = m_VppResponse.mids[i];
            } else {
                sts = m_pMFXAllocator->Lock(m_pMFXAllocator->pthis, m_VppResponse.mids[i], &(m_pVppSurfaces[i].Data));
                MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to allocate surfaces for vpp."));
            }
        }
    }

    for (const auto& filter : m_VppPrePlugins) {
        if (MFX_ERR_NONE != (sts = filter->AllocSurfaces(m_pMFXAllocator, m_bExternalAlloc))) {
            PrintMes(QSV_LOG_ERROR, _T("AllocFrames: Failed to alloc surface for %s\n"), filter->getFilterName().c_str());
            return sts;
        } else {
            PrintMes(QSV_LOG_DEBUG, _T("AllocFrames: Allocated surface for %s\n"), filter->getFilterName().c_str());
        }
    }

    for (const auto& filter : m_VppPostPlugins) {
        if (MFX_ERR_NONE != (sts = filter->AllocSurfaces(m_pMFXAllocator, m_bExternalAlloc))) {
            PrintMes(QSV_LOG_ERROR, _T("AllocFrames: Failed to alloc surface for %s\n"), filter->getFilterName().c_str());
            return sts;
        } else {
            PrintMes(QSV_LOG_DEBUG, _T("AllocFrames: Allocated surface for %s\n"), filter->getFilterName().c_str());
        }
    }

    return MFX_ERR_NONE;
}

mfxStatus CEncodingPipeline::CreateAllocator()
{
    mfxStatus sts = MFX_ERR_NONE;

    if (D3D9_MEMORY == m_memType || D3D11_MEMORY == m_memType) {
#if D3D_SURFACES_SUPPORT
        sts = CreateHWDevice();
        if (sts < MFX_ERR_NONE) return sts;

        mfxHDL hdl = NULL;
        mfxHandleType hdl_t =
        #if MFX_D3D11_SUPPORT
            D3D11_MEMORY == m_memType ? MFX_HANDLE_D3D11_DEVICE :
        #endif // #if MFX_D3D11_SUPPORT
            MFX_HANDLE_D3D9_DEVICE_MANAGER;

        sts = m_hwdev->GetHandle(hdl_t, &hdl);
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to get HW device handle."));
        
        // handle is needed for HW library only
        mfxIMPL impl = 0;
        m_mfxSession.QueryIMPL(&impl);
        if (impl != MFX_IMPL_SOFTWARE) {
            sts = m_mfxSession.SetHandle(hdl_t, hdl);
            MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to set HW device handle to encode session.")); 
        }

        // create D3D allocator
#if MFX_D3D11_SUPPORT
        if (D3D11_MEMORY == m_memType)
        {
            m_pMFXAllocator = new D3D11FrameAllocator;
            MSDK_CHECK_POINTER(m_pMFXAllocator, MFX_ERR_MEMORY_ALLOC);

            D3D11AllocatorParams *pd3dAllocParams = new D3D11AllocatorParams;
            MSDK_CHECK_POINTER(pd3dAllocParams, MFX_ERR_MEMORY_ALLOC);
            pd3dAllocParams->pDevice = reinterpret_cast<ID3D11Device *>(hdl);
            PrintMes(QSV_LOG_DEBUG, _T("CreateAllocator: d3d11...\n"));

            m_pmfxAllocatorParams = pd3dAllocParams;
        }
        else
#endif // #if MFX_D3D11_SUPPORT
        {
            m_pMFXAllocator = new D3DFrameAllocator;
            MSDK_CHECK_POINTER(m_pMFXAllocator, MFX_ERR_MEMORY_ALLOC);

            D3DAllocatorParams *pd3dAllocParams = new D3DAllocatorParams;
            MSDK_CHECK_POINTER(pd3dAllocParams, MFX_ERR_MEMORY_ALLOC);
            pd3dAllocParams->pManager = reinterpret_cast<IDirect3DDeviceManager9 *>(hdl);
            PrintMes(QSV_LOG_DEBUG, _T("CreateAllocator: d3d9...\n"));

            m_pmfxAllocatorParams = pd3dAllocParams;
        }

        /* In case of video memory we must provide MediaSDK with external allocator
        thus we demonstrate "external allocator" usage model.
        Call SetAllocator to pass allocator to Media SDK */
        sts = m_mfxSession.SetFrameAllocator(m_pMFXAllocator);
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to set frame allocator to encode session."));
        PrintMes(QSV_LOG_DEBUG, _T("CreateAllocator: frame allocator set to session.\n"));

        m_bExternalAlloc = true;
#endif
#ifdef LIBVA_SUPPORT
        sts = CreateHWDevice();
        if (sts < MFX_ERR_NONE) return sts;
        /* It's possible to skip failed result here and switch to SW implementation,
        but we don't process this way */

        mfxHDL hdl = NULL;
        sts = m_hwdev->GetHandle(MFX_HANDLE_VA_DISPLAY, &hdl);
        // provide device manager to MediaSDK
        sts = m_mfxSession.SetHandle(MFX_HANDLE_VA_DISPLAY, hdl);
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to set HW device handle to encode session.")); 

        // create VAAPI allocator
        m_pMFXAllocator = new vaapiFrameAllocator;
        MSDK_CHECK_POINTER(m_pMFXAllocator, MFX_ERR_MEMORY_ALLOC);

        vaapiAllocatorParams *p_vaapiAllocParams = new vaapiAllocatorParams;
        MSDK_CHECK_POINTER(p_vaapiAllocParams, MFX_ERR_MEMORY_ALLOC);

        p_vaapiAllocParams->m_dpy = (VADisplay)hdl;
        m_pmfxAllocatorParams = p_vaapiAllocParams;

        /* In case of video memory we must provide MediaSDK with external allocator 
        thus we demonstrate "external allocator" usage model.
        Call SetAllocator to pass allocator to mediasdk */
        sts = m_mfxSession.SetFrameAllocator(m_pMFXAllocator);
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to set frame allocator to encode session."));

        m_bExternalAlloc = true;
#endif
    }
    else
    {
#ifdef LIBVA_SUPPORT
        //in case of system memory allocator we also have to pass MFX_HANDLE_VA_DISPLAY to HW library
        mfxIMPL impl;
        m_mfxSession.QueryIMPL(&impl);

        if(MFX_IMPL_HARDWARE == MFX_IMPL_BASETYPE(impl))
        {
            sts = CreateHWDevice();
            if (sts < MFX_ERR_NONE) return sts;

            mfxHDL hdl = NULL;
            sts = m_hwdev->GetHandle(MFX_HANDLE_VA_DISPLAY, &hdl);
            // provide device manager to MediaSDK
            sts = m_mfxSession.SetHandle(MFX_HANDLE_VA_DISPLAY, hdl);
            MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to set HW device handle to encode session.")); 
        }
#endif

        // create system memory allocator
        m_pMFXAllocator = new SysMemFrameAllocator;
        MSDK_CHECK_POINTER(m_pMFXAllocator, MFX_ERR_MEMORY_ALLOC);
        PrintMes(QSV_LOG_DEBUG, _T("CreateAllocator: sys mem allocator...\n"));

        /* In case of system memory we demonstrate "no external allocator" usage model.
        We don't call SetAllocator, Media SDK uses internal allocator.
        We use system memory allocator simply as a memory manager for application*/
    }

    // initialize memory allocator
    sts = m_pMFXAllocator->Init(m_pmfxAllocatorParams);
    if (sts < MFX_ERR_NONE) {
        PrintMes(QSV_LOG_ERROR, _T("Failed to initialize %s memory allocator. : %s\n"), MemTypeToStr(m_memType), get_err_mes(sts));
        return sts;
    }
    PrintMes(QSV_LOG_DEBUG, _T("CreateAllocator: frame allocator initialized.\n"));

    return MFX_ERR_NONE;
}

void CEncodingPipeline::DeleteFrames()
{
    // delete surfaces array
    MSDK_SAFE_DELETE_ARRAY(m_pEncSurfaces);
    MSDK_SAFE_DELETE_ARRAY(m_pVppSurfaces);
    MSDK_SAFE_DELETE_ARRAY(m_pDecSurfaces);

    // delete frames
    if (m_pMFXAllocator)
    {
        m_pMFXAllocator->Free(m_pMFXAllocator->pthis, &m_EncResponse);
        m_pMFXAllocator->Free(m_pMFXAllocator->pthis, &m_VppResponse);
        m_pMFXAllocator->Free(m_pMFXAllocator->pthis, &m_DecResponse);
    }

    MSDK_ZERO_MEMORY(m_EncResponse);
    MSDK_ZERO_MEMORY(m_VppResponse);
    MSDK_ZERO_MEMORY(m_DecResponse);
}

void CEncodingPipeline::DeleteHWDevice()
{
#if D3D_SURFACES_SUPPORT
    MSDK_SAFE_DELETE(m_hwdev);
#endif
}

void CEncodingPipeline::DeleteAllocator()
{
    // delete allocator
    MSDK_SAFE_DELETE(m_pMFXAllocator);
    MSDK_SAFE_DELETE(m_pmfxAllocatorParams);

    DeleteHWDevice();
}

CEncodingPipeline::CEncodingPipeline()
{
    m_pmfxDEC = NULL;
    m_pmfxENC = NULL;
    m_pmfxVPP = NULL;
    m_pMFXAllocator = NULL;
    m_pmfxAllocatorParams = NULL;
    m_memType = SYSTEM_MEMORY;
    m_bExternalAlloc = false;
    m_pEncSurfaces = NULL;
    m_pVppSurfaces = NULL;
    m_pDecSurfaces = NULL;
    m_nAsyncDepth = 0;
    m_nExPrm = 0x00;
    m_bTimerPeriodTuning = false;

    m_pAbortByUser = NULL;

    m_pEncSatusInfo = NULL;
    m_pFileWriterListAudio.clear();
    m_pFileWriter = NULL;
    m_pFileReader = NULL;

    m_pTrimParam = NULL;

#if ENABLE_MVC_ENCODING
    m_bIsMVC = false;
    m_MVCflags = MVC_DISABLED;
    m_nNumView = 0;
    MSDK_ZERO_MEMORY(m_MVCSeqDesc);
    m_MVCSeqDesc.Header.BufferId = MFX_EXTBUFF_MVC_SEQ_DESC;
    m_MVCSeqDesc.Header.BufferSz = sizeof(m_MVCSeqDesc);
#endif
    INIT_MFX_EXT_BUFFER(m_VppDoNotUse,        MFX_EXTBUFF_VPP_DONOTUSE);
    INIT_MFX_EXT_BUFFER(m_VideoSignalInfo,    MFX_EXTBUFF_VIDEO_SIGNAL_INFO);
    INIT_MFX_EXT_BUFFER(m_CodingOption,       MFX_EXTBUFF_CODING_OPTION);
    INIT_MFX_EXT_BUFFER(m_CodingOption2,      MFX_EXTBUFF_CODING_OPTION2);
    INIT_MFX_EXT_BUFFER(m_CodingOption3,      MFX_EXTBUFF_CODING_OPTION3);
    INIT_MFX_EXT_BUFFER(m_ExtVP8CodingOption, MFX_EXTBUFF_VP8_CODING_OPTION);
    INIT_MFX_EXT_BUFFER(m_ExtHEVCParam,       MFX_EXTBUFF_HEVC_PARAM);
    INIT_MFX_EXT_BUFFER(m_ThreadsParam,       MFX_EXTBUFF_THREADS_PARAM);

#if D3D_SURFACES_SUPPORT
    m_hwdev = NULL;
#endif
    MSDK_ZERO_MEMORY(m_DecInputBitstream);
    
    MSDK_ZERO_MEMORY(m_InitParam);
    MSDK_ZERO_MEMORY(m_mfxDecParams);
    MSDK_ZERO_MEMORY(m_mfxEncParams);
    MSDK_ZERO_MEMORY(m_mfxVppParams);
    
    MSDK_ZERO_MEMORY(m_VppDoNotUse);
    MSDK_ZERO_MEMORY(m_VppDoUse);
    MSDK_ZERO_MEMORY(m_ExtDenoise);
    MSDK_ZERO_MEMORY(m_ExtDetail);

    MSDK_ZERO_MEMORY(m_EncResponse);
    MSDK_ZERO_MEMORY(m_VppResponse);
    MSDK_ZERO_MEMORY(m_DecResponse);
}

CEncodingPipeline::~CEncodingPipeline()
{
    Close();
}

void CEncodingPipeline::SetAbortFlagPointer(bool *abortFlag) {
    m_pAbortByUser = abortFlag;
}

#if ENABLE_MVC_ENCODING
void CEncodingPipeline::SetMultiView()
{
    m_pFileReader->SetMultiView();
    m_bIsMVC = true;
}
#endif

mfxStatus CEncodingPipeline::InitOutput(sInputParams *pParams) {
    mfxStatus sts = MFX_ERR_NONE;
    bool stdoutUsed = false;
    // prepare output file writer
#if ENABLE_AVCODEC_QSV_READER
    vector<int> audioTrackUsed; //使用した音声のトラックIDを保存する
    bool useH264ESOutput =
        ((pParams->pAVMuxOutputFormat && 0 == _tcscmp(pParams->pAVMuxOutputFormat, _T("raw")))) //--formatにrawが指定されている
        || (PathFindExtension(pParams->strDstFile) == nullptr || PathFindExtension(pParams->strDstFile)[0] != '.') //拡張子がしない
        || check_ext(pParams->strDstFile, { ".m2v", ".264", ".h264", ".avc", ".avc1", ".x264", ".265", ".h265", ".hevc" }); //特定の拡張子
    if (!useH264ESOutput) {
        pParams->nAVMux |= QSVENC_MUX_VIDEO;
    }
    if (pParams->CodecId == MFX_CODEC_RAW) {
        pParams->nAVMux &= ~QSVENC_MUX_VIDEO;
    }
    if (pParams->nAVMux & QSVENC_MUX_VIDEO) {
        if (pParams->CodecId == MFX_CODEC_HEVC || pParams->CodecId == MFX_CODEC_VP8) {
            PrintMes(QSV_LOG_ERROR, _T("Output: muxing not supported with %s.\n"), CodecIdToStr(pParams->CodecId).c_str());
            return MFX_ERR_UNSUPPORTED;
        }
        PrintMes(QSV_LOG_DEBUG, _T("Output: Using avformat writer.\n"));
        m_pFileWriter = new CAvcodecWriter();
        AvcodecWriterPrm writerPrm = { 0 };
        writerPrm.pOutputFormat = pParams->pAVMuxOutputFormat;
        writerPrm.pVideoInfo = &m_mfxEncParams.mfx;
        writerPrm.pVideoSignalInfo = &m_VideoSignalInfo;
        writerPrm.bVideoDtsUnavailable = !check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_6);
        if (pParams->nAVMux & QSVENC_MUX_AUDIO) {
            PrintMes(QSV_LOG_DEBUG, _T("Output: Audio muxing enabled.\n"));
            auto pAVCodecReader = reinterpret_cast<CAvcodecReader *>(m_pFileReader);
            bool copyAll = false;
            for (int i = 0; !copyAll && i < pParams->nAudioSelectCount; i++) {
                //トラック"0"が指定されていれば、すべてのトラックをコピーするということ
                copyAll = (pParams->ppAudioSelectList[i]->nAudioSelect == 0);
            }
            PrintMes(QSV_LOG_DEBUG, _T("Output: CopyAll=%s\n"), (copyAll) ? _T("true") : _T("false"));
            std::vector<AVDemuxAudio> audioList;
            if (pAVCodecReader->GetAudioTrackCount()) {
                audioList = pAVCodecReader->GetInputAudioInfo();
            }
            for (const auto& audioReader : m_AudioReaders) {
                if (audioReader->GetAudioTrackCount()) {
                    auto pAVCodecAudioReader = reinterpret_cast<CAvcodecReader *>(audioReader.get());
                    auto tempList = pAVCodecAudioReader->GetInputAudioInfo();
                    audioList.insert(audioList.end(), tempList.begin(), tempList.end());
                }
            }

            for (auto& audioTrack : audioList) {
                const sAudioSelect *pAudioSelect = nullptr;
                for (int i = 0; i < pParams->nAudioSelectCount; i++) {
                    if (audioTrack.nTrackId == pParams->ppAudioSelectList[i]->nAudioSelect
                        && pParams->ppAudioSelectList[i]->pAudioExtractFilename == nullptr) {
                        pAudioSelect = pParams->ppAudioSelectList[i];
                    }
                }
                if (pAudioSelect != nullptr || copyAll) {
                    audioTrackUsed.push_back(audioTrack.nTrackId);
                    AVOutputAudioPrm prm;
                    prm.src = audioTrack;
                    //pAudioSelect == nullptrは "copyAll" によるもの
                    prm.nBitrate = (pAudioSelect == nullptr) ? 0 : pAudioSelect->nAVAudioEncodeBitrate;
                    prm.pEncodeCodec = (pAudioSelect == nullptr) ? AVQSV_CODEC_COPY : pAudioSelect->pAVAudioEncodeCodec;
                    PrintMes(QSV_LOG_DEBUG, _T("Output: Added audio track#%d (stream idx %d) for mux, bitrate %d, codec: %s\n"),
                        audioTrack.nTrackId, audioTrack.nIndex, prm.nBitrate, prm.pEncodeCodec);
                    writerPrm.inputAudioList.push_back(std::move(prm));
                }
            }
        }
        m_pFileWriter->SetQSVLogPtr(m_pQSVLog.get());
        sts = m_pFileWriter->Init(pParams->strDstFile, &writerPrm, m_pEncSatusInfo);
        if (sts < MFX_ERR_NONE) {
            PrintMes(QSV_LOG_ERROR, m_pFileWriter->GetOutputMessage());
            return sts;
        } else if (pParams->nAVMux & QSVENC_MUX_AUDIO) {
            m_pFileWriterListAudio.push_back(m_pFileWriter);
        }
        stdoutUsed = m_pFileWriter->outputStdout();
        PrintMes(QSV_LOG_DEBUG, _T("Output: Initialized avformat writer%s.\n"), (stdoutUsed) ? _T("using stdout") : _T(""));
    } else if (pParams->nAVMux & QSVENC_MUX_AUDIO) {
        PrintMes(QSV_LOG_ERROR, _T("Audio mux cannot be used alone, should be use with video mux.\n"));
        return MFX_ERR_UNSUPPORTED;
    } else {
#endif
        if (pParams->CodecId == MFX_CODEC_RAW) {
            m_pFrameWriter.reset(new CSmplYUVWriter());
            m_pFrameWriter->SetQSVLogPtr(m_pQSVLog.get());
            YUVWriterParam param;
            param.bY4m = true;
            param.memType = m_memType;
            sts = m_pFrameWriter->Init(pParams->strDstFile, &param, m_pEncSatusInfo);
            if (sts < MFX_ERR_NONE) {
                PrintMes(QSV_LOG_ERROR, m_pFileWriter->GetOutputMessage());
                return sts;
            }
            stdoutUsed = m_pFrameWriter->outputStdout();
            PrintMes(QSV_LOG_DEBUG, _T("Output: Initialized yuv frame writer%s.\n"), (stdoutUsed) ? _T("using stdout") : _T(""));
        } else {
            m_pFileWriter = new CSmplBitstreamWriter();
            m_pFileWriter->SetQSVLogPtr(m_pQSVLog.get());
            bool bBenchmark = pParams->bBenchmark != 0;
            sts = m_pFileWriter->Init(pParams->strDstFile, &bBenchmark, m_pEncSatusInfo);
            if (sts < MFX_ERR_NONE) {
                PrintMes(QSV_LOG_ERROR, m_pFileWriter->GetOutputMessage());
                return sts;
            }
            stdoutUsed = m_pFileWriter->outputStdout();
            PrintMes(QSV_LOG_DEBUG, _T("Output: Initialized bitstream writer%s.\n"), (stdoutUsed) ? _T("using stdout") : _T(""));
        }
#if ENABLE_AVCODEC_QSV_READER
    } //ENABLE_AVCODEC_QSV_READER

    //音声の抽出
    if (pParams->nAudioSelectCount > audioTrackUsed.size()) {
        PrintMes(QSV_LOG_DEBUG, _T("Output: Audio file output enabled.\n"));
        auto pAVCodecReader = reinterpret_cast<CAvcodecReader *>(m_pFileReader);
        if (pParams->nInputFmt != INPUT_FMT_AVCODEC_QSV || pAVCodecReader == NULL) {
            PrintMes(QSV_LOG_ERROR, _T("Audio output is only supported with transcoding (avqsv reader).\n"));
            return MFX_ERR_UNSUPPORTED;
        } else {
            auto inutAudioInfoList = pAVCodecReader->GetInputAudioInfo();
            for (auto& audioTrack : inutAudioInfoList) {
                for (auto usedTrack : audioTrackUsed) {
                    if (usedTrack == audioTrack.nTrackId) {
                        PrintMes(QSV_LOG_ERROR, _T("Audio track #%d is already set to be muxed, so cannot be extracted to file.\n"), audioTrack.nTrackId);
                        return MFX_ERR_INVALID_AUDIO_PARAM;
                    }
                }
                const sAudioSelect *pAudioSelect = nullptr;
                for (int i = 0; i < pParams->nAudioSelectCount; i++) {
                    if (audioTrack.nTrackId == pParams->ppAudioSelectList[i]->nAudioSelect
                        && pParams->ppAudioSelectList[i]->pAudioExtractFilename != nullptr) {
                        pAudioSelect = pParams->ppAudioSelectList[i];
                    }
                }
                if (pAudioSelect == nullptr) {
                    PrintMes(QSV_LOG_ERROR, _T("Audio track #%d is not used anyware, this should not happen.\n"), audioTrack.nTrackId);
                    return MFX_ERR_INVALID_AUDIO_PARAM;
                }
                PrintMes(QSV_LOG_DEBUG, _T("Output: Output audio track #%d (stream index %d) to \"%s\", format: %s, codec %s, bitrate %d\n"),
                    audioTrack.nTrackId, audioTrack.nIndex, pAudioSelect->pAudioExtractFilename, pAudioSelect->pAudioExtractFormat, pAudioSelect->pAVAudioEncodeCodec, pAudioSelect->nAVAudioEncodeBitrate);

                AVOutputAudioPrm prm;
                prm.src = audioTrack;
                //pAudioSelect == nullptrは "copyAll" によるもの
                prm.nBitrate = pAudioSelect->nAVAudioEncodeBitrate;
                prm.pEncodeCodec = pAudioSelect->pAVAudioEncodeCodec;
                
                AvcodecWriterPrm writerAudioPrm = { 0 };
                writerAudioPrm.pOutputFormat = pAudioSelect->pAudioExtractFormat;
                writerAudioPrm.inputAudioList.push_back(prm);

                auto pWriter = new CAvcodecWriter();
                pWriter->SetQSVLogPtr(m_pQSVLog.get());
                sts = pWriter->Init(pAudioSelect->pAudioExtractFilename, &writerAudioPrm, m_pEncSatusInfo);
                if (sts < MFX_ERR_NONE) {
                    PrintMes(QSV_LOG_ERROR, pWriter->GetOutputMessage());
                    return sts;
                }
                PrintMes(QSV_LOG_DEBUG, _T("Output: Intialized audio output for track #%d.\n"), audioTrack.nTrackId);
                bool audioStdout = pWriter->outputStdout();
                if (stdoutUsed && audioStdout) {
                    PrintMes(QSV_LOG_ERROR, _T("Multiple stream outputs are set to stdout, please remove conflict.\n"));
                    return MFX_ERR_INVALID_AUDIO_PARAM;
                }
                stdoutUsed |= audioStdout;
                m_pFileWriterListAudio.push_back(std::move(pWriter));
            }
        }
    }
#endif //ENABLE_AVCODEC_QSV_READER
    return sts;
}

mfxStatus CEncodingPipeline::InitInput(sInputParams *pParams)
{
    mfxStatus sts = MFX_ERR_NONE;
    m_pQSVLog.reset(new CQSVLog(pParams->pStrLogFile, pParams->nLogLevel));

    //prepare for LogFile
    if (pParams->pStrLogFile) {
        m_pQSVLog->writeFileHeader(pParams->strDstFile);
    }

    int sourceAudioTrackIdStart = 1; //トラック番号は1スタート
    m_pEncSatusInfo = new CEncodeStatusInfo();

    //Auto detection by input file extension
    if (pParams->nInputFmt == INPUT_FMT_AUTO) {
#if ENABLE_AVISYNTH_READER
        if (check_ext(pParams->strSrcFile, { ".avs" }))
            pParams->nInputFmt = INPUT_FMT_AVS;
        else
#endif //ENABLE_AVISYNTH_READER
#if ENABLE_VAPOURSYNTH_READER
        if (check_ext(pParams->strSrcFile, { ".vpy" }))
            pParams->nInputFmt = INPUT_FMT_VPY;
        else
#endif //ENABLE_VAPOURSYNTH_READER
#if ENABLE_AVI_READER
        if (check_ext(pParams->strSrcFile, { ".avi", ".avs", ".vpy" }))
            pParams->nInputFmt = INPUT_FMT_AVI;
        else
#endif //ENABLE_AVI_READER
#if ENABLE_AVCODEC_QSV_READER
        if (check_ext(pParams->strSrcFile, { ".mp4", ".m4v", ".mkv", ".mov",
            ".mts", ".m2ts", ".ts", ".264", ".h264", ".x264", ".avc", ".avc1",
            ".265", ".h265", ".hevc",
            ".mpg", ".mpeg", "m2v", ".vob", ".vro", ".flv", ".ogm",
            ".wmv" }))
            pParams->nInputFmt = INPUT_FMT_AVCODEC_QSV;
        else
#endif //ENABLE_AVCODEC_QSV_READER
        if (check_ext(pParams->strSrcFile, { ".y4m" }))
            pParams->nInputFmt = INPUT_FMT_Y4M;
    }

    //Check if selected format is enabled
    if (pParams->nInputFmt == INPUT_FMT_AVS && !ENABLE_AVISYNTH_READER) {
        pParams->nInputFmt = INPUT_FMT_AVI;
        PrintMes(QSV_LOG_WARN, _T("avs reader not compiled in this binary.\n"));
        PrintMes(QSV_LOG_WARN, _T("switching to avi reader.\n"));
    }
    if (pParams->nInputFmt == INPUT_FMT_VPY && !ENABLE_VAPOURSYNTH_READER) {
        pParams->nInputFmt = INPUT_FMT_AVI;
        PrintMes(QSV_LOG_WARN, _T("vpy reader not compiled in this binary.\n"));
        PrintMes(QSV_LOG_WARN, _T("switching to avi reader.\n"));
    }
    if (pParams->nInputFmt == INPUT_FMT_VPY_MT && !ENABLE_VAPOURSYNTH_READER) {
        pParams->nInputFmt = INPUT_FMT_AVI;
        PrintMes(QSV_LOG_WARN, _T("vpy reader not compiled in this binary.\n"));
        PrintMes(QSV_LOG_WARN, _T("switching to avi reader.\n"));
    }
    if (pParams->nInputFmt == INPUT_FMT_AVI && !ENABLE_AVI_READER) {
        PrintMes(QSV_LOG_ERROR, _T("avi reader not compiled in this binary.\n"));
        return MFX_ERR_INCOMPATIBLE_VIDEO_PARAM;
    }
    if (pParams->nInputFmt == INPUT_FMT_AVCODEC_QSV && !ENABLE_AVCODEC_QSV_READER) {
        PrintMes(QSV_LOG_ERROR, _T("avcodec + QSV reader not compiled in this binary.\n"));
        return MFX_ERR_INCOMPATIBLE_VIDEO_PARAM;
    }

    //try to setup avs or vpy reader
    m_pFileReader = NULL;
    if (   pParams->nInputFmt == INPUT_FMT_VPY
        || pParams->nInputFmt == INPUT_FMT_VPY_MT
        || pParams->nInputFmt == INPUT_FMT_AVS) {
        void *input_options = nullptr;
#if ENABLE_VAPOURSYNTH_READER
        VSReaderPrm vsReaderPrm = { 0 };
#endif
        if (pParams->nInputFmt == INPUT_FMT_VPY || pParams->nInputFmt == INPUT_FMT_VPY_MT) {
#if ENABLE_VAPOURSYNTH_READER
            vsReaderPrm.use_mt = pParams->nInputFmt == INPUT_FMT_VPY_MT;
            input_options = &vsReaderPrm;
            m_pFileReader = new CVSReader();
            PrintMes(QSV_LOG_DEBUG, _T("Input: vpy reader selected.\n"));
#endif
        } else {
#if ENABLE_AVISYNTH_READER
            m_pFileReader = new CAVSReader();
            PrintMes(QSV_LOG_DEBUG, _T("Input: avs reader selected.\n"));
#endif
        }
        if (NULL == m_pFileReader) {
            //switch to avi reader and retry
            pParams->nInputFmt = INPUT_FMT_AVI;
        } else {
            m_pFileReader->SetQSVLogPtr(m_pQSVLog.get());
            sts = m_pFileReader->Init(pParams->strSrcFile, pParams->ColorFormat, input_options,
                &m_EncThread, m_pEncSatusInfo, &pParams->sInCrop);
            if (sts == MFX_ERR_INVALID_COLOR_FORMAT) {
                //if failed because of colorformat, switch to avi reader and retry.
                PrintMes(QSV_LOG_WARN, m_pFileReader->GetInputMessage());
                delete m_pFileReader;
                m_pFileReader = NULL;
                sts = MFX_ERR_NONE;
                PrintMes(QSV_LOG_WARN, _T("Input: switching to avi reader.\n"));
                pParams->nInputFmt = INPUT_FMT_AVI;
            }
            if (sts < MFX_ERR_NONE) {
                PrintMes(QSV_LOG_ERROR, m_pFileReader->GetInputMessage());
                return sts;
            }
        }
    }

    if (NULL == m_pFileReader) {
        const void *input_option = nullptr;
        bool bY4m = pParams->nInputFmt == INPUT_FMT_Y4M;
#if ENABLE_AVCODEC_QSV_READER
        AvcodecReaderPrm avcodecReaderPrm = { 0 };
#endif
        switch (pParams->nInputFmt) {
#if ENABLE_AVI_READER
            case INPUT_FMT_AVI:
                m_pFileReader = new CAVIReader();
                PrintMes(QSV_LOG_DEBUG, _T("Input: avi reader selected.\n"));
                break;
#endif
#if ENABLE_AVCODEC_QSV_READER
            case INPUT_FMT_AVCODEC_QSV:
                if (!pParams->bUseHWLib) {
                    PrintMes(QSV_LOG_ERROR, _T("Input: avqsv reader is only supported with HW libs.\n"));
                    return MFX_ERR_UNSUPPORTED;
                }
                m_pFileReader = new CAvcodecReader();
                avcodecReaderPrm.memType = pParams->memType;
                avcodecReaderPrm.bReadVideo = true;
                avcodecReaderPrm.pTrimList = pParams->pTrimList;
                avcodecReaderPrm.nTrimCount = pParams->nTrimCount;
                avcodecReaderPrm.nReadAudio |= pParams->nAudioSelectCount > 0; 
                avcodecReaderPrm.nAnalyzeSec = pParams->nAVDemuxAnalyzeSec;
                avcodecReaderPrm.nAudioTrackStart = (mfxU8)sourceAudioTrackIdStart;
                avcodecReaderPrm.ppAudioSelect = pParams->ppAudioSelectList;
                avcodecReaderPrm.nAudioSelectCount = pParams->nAudioSelectCount;
                input_option = &avcodecReaderPrm;
                PrintMes(QSV_LOG_DEBUG, _T("Input: avqsv reader selected.\n"));
                break;
#endif
            case INPUT_FMT_Y4M:
            case INPUT_FMT_RAW:
            default:
                input_option = &bY4m;
                m_pFileReader = new CSmplYUVReader();
                PrintMes(QSV_LOG_DEBUG, _T("Input: yuv reader selected (%s).\n"), (bY4m) ? _T("y4m") : _T("raw"));
                break;
        }
        m_pFileReader->SetQSVLogPtr(m_pQSVLog.get());
        sts = m_pFileReader->Init(pParams->strSrcFile, pParams->ColorFormat, input_option,
            &m_EncThread, m_pEncSatusInfo, &pParams->sInCrop);
    }
    if (sts < MFX_ERR_NONE) {
        PrintMes(QSV_LOG_ERROR, m_pFileReader->GetInputMessage());
        return sts;
    }
    PrintMes(QSV_LOG_DEBUG, _T("Input: reader initialization successful.\n"));
    sourceAudioTrackIdStart += m_pFileReader->GetAudioTrackCount();

#if ENABLE_AVCODEC_QSV_READER
    if (pParams->nAudioSourceCount && pParams->ppAudioSourceList) {
        if (pParams->nTrimCount > 0) {
            PrintMes(QSV_LOG_ERROR, _T("Input: audio source not supported with trim option.\n"));
            return MFX_ERR_UNSUPPORTED;
        }
        mfxFrameInfo videoInfo = { 0 };
        m_pFileReader->GetInputFrameInfo(&videoInfo);

        for (int i = 0; i < pParams->nAudioSourceCount; i++) {
            AvcodecReaderPrm avcodecReaderPrm = { 0 };
            avcodecReaderPrm.memType = pParams->memType;
            avcodecReaderPrm.bReadVideo = false;
            avcodecReaderPrm.nReadAudio |= pParams->nAudioSelectCount > 0;
            avcodecReaderPrm.nAnalyzeSec = pParams->nAVDemuxAnalyzeSec;
            avcodecReaderPrm.nVideoAvgFramerate = std::make_pair(videoInfo.FrameRateExtN, videoInfo.FrameRateExtD);
            avcodecReaderPrm.nAudioTrackStart = (mfxU8)sourceAudioTrackIdStart;
            avcodecReaderPrm.ppAudioSelect = pParams->ppAudioSelectList;
            avcodecReaderPrm.nAudioSelectCount = pParams->nAudioSelectCount;

            unique_ptr<CSmplYUVReader> audioReader(new CAvcodecReader());
            audioReader->SetQSVLogPtr(m_pQSVLog.get());
            sts = audioReader->Init(pParams->ppAudioSourceList[i], 0, &avcodecReaderPrm, nullptr, nullptr, nullptr);
            if (sts < MFX_ERR_NONE) {
                PrintMes(QSV_LOG_ERROR, audioReader->GetInputMessage());
                return sts;
            }
            sourceAudioTrackIdStart += audioReader->GetAudioTrackCount();
            m_AudioReaders.push_back(std::move(audioReader));
        }
    }
#endif

    if (m_pFileReader->getInputCodec()) {
        auto trimParam = m_pFileReader->GetTrimParam();
        m_pTrimParam = (trimParam->list.size()) ? trimParam : NULL;
        PrintMes(QSV_LOG_DEBUG, _T("Input: trim options set to reader\n"));
        for (int i = 0; i < (int)trimParam->list.size(); i++) {
            PrintMes(QSV_LOG_DEBUG, _T("%d-%d "), trimParam->list[i].start, trimParam->list[i].fin);
        }
        PrintMes(QSV_LOG_DEBUG, _T(" (offset: %d)\n"), trimParam->offset);
    } else if (pParams->pTrimList) {
        PrintMes(QSV_LOG_ERROR, _T("Input: Trim is only supported with transcoding (avqsv reader).\n"));
        return MFX_ERR_UNSUPPORTED;
    }

    return sts;
}

mfxStatus CEncodingPipeline::DetermineMinimumRequiredVersion(const sInputParams &pParams, mfxVersion &version)
{
    version.Major = 1;
    version.Minor = 0;

    if (MVC_DISABLED != pParams.MVC_flags)
        version.Minor = 3;
    return MFX_ERR_NONE;
}

mfxStatus CEncodingPipeline::CheckParam(sInputParams *pParams) {
    mfxFrameInfo inputFrameInfo = { 0 };
    m_pFileReader->GetInputFrameInfo(&inputFrameInfo);

    sInputCrop cropInfo = { 0 };
    m_pFileReader->GetInputCropInfo(&cropInfo);

    //Get Info From Input
    if (inputFrameInfo.Width)
        pParams->nWidth = inputFrameInfo.Width;

    if (inputFrameInfo.Height)
        pParams->nHeight = inputFrameInfo.Height;

    if (inputFrameInfo.PicStruct)
        pParams->nPicStruct = inputFrameInfo.PicStruct;

    if ((!pParams->nFPSRate || !pParams->nFPSScale) && inputFrameInfo.FrameRateExtN && inputFrameInfo.FrameRateExtD) {
        pParams->nFPSRate = inputFrameInfo.FrameRateExtN;
        pParams->nFPSScale = inputFrameInfo.FrameRateExtD;
    }

    if (0 == pParams->inputBitDepthLuma && inputFrameInfo.BitDepthLuma) {
        pParams->inputBitDepthLuma = inputFrameInfo.BitDepthLuma;
    }

    if (0 == pParams->inputBitDepthChroma && inputFrameInfo.BitDepthChroma) {
        pParams->inputBitDepthChroma = inputFrameInfo.BitDepthChroma;
    }

    //Checking Start...
    //if picstruct not set, progressive frame is expected
    if (!pParams->nPicStruct) {
        pParams->nPicStruct = MFX_PICSTRUCT_PROGRESSIVE;
    }

    //don't use d3d memory with software encoding
    if (!pParams->bUseHWLib) {
        pParams->memType = SYSTEM_MEMORY;
    }

    int h_mul = 2;
    bool output_interlaced = ((pParams->nPicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF)) != 0 && !pParams->vpp.nDeinterlace);
    if (output_interlaced)
        h_mul *= 2;
    //check for crop settings
    if (pParams->sInCrop.left % 2 != 0 || pParams->sInCrop.right % 2 != 0) {
        PrintMes(QSV_LOG_ERROR, _T("crop width should be a multiple of 2.\n"));
        return MFX_ERR_INVALID_VIDEO_PARAM;
    }
    if (pParams->sInCrop.bottom % h_mul != 0 || pParams->sInCrop.up % h_mul != 0) {
        PrintMes(QSV_LOG_ERROR, _T("crop height should be a multiple of %d.\n"));
        return MFX_ERR_INVALID_VIDEO_PARAM;
    }
    if (0 == pParams->nWidth || 0 == pParams->nHeight) {
        PrintMes(QSV_LOG_ERROR, _T("--input-res must be specified with raw input.\n"));
        return MFX_ERR_INVALID_VIDEO_PARAM;
    }
    if (pParams->nFPSRate == 0 || pParams->nFPSScale == 0) {
        PrintMes(QSV_LOG_ERROR, _T("--fps must be specified with raw input.\n"));
        return MFX_ERR_INVALID_VIDEO_PARAM;
    }
    if (   pParams->nWidth < (pParams->sInCrop.left + pParams->sInCrop.right)
        || pParams->nHeight < (pParams->sInCrop.bottom + pParams->sInCrop.up)) {
            PrintMes(QSV_LOG_ERROR, _T("crop size is too big.\n"));
            return MFX_ERR_INVALID_VIDEO_PARAM;
    }

    // if no destination picture width or height wasn't specified set it to the source picture size
    if (pParams->nDstWidth == 0) {
        pParams->nDstWidth = pParams->nWidth -  (pParams->sInCrop.left + pParams->sInCrop.right);
    }

    if (pParams->nDstHeight == 0) {
        pParams->nDstHeight = pParams->nHeight - (pParams->sInCrop.bottom + pParams->sInCrop.up);
    }

    if (0 == m_pFileReader->getInputCodec()) {
        //QSVデコードを使わない場合には、入力段階でCropが行われる
        pParams->nWidth -= (pParams->sInCrop.left + pParams->sInCrop.right);
        pParams->nHeight -= (pParams->sInCrop.bottom + pParams->sInCrop.up);
    }

    if (pParams->nDstHeight != pParams->nHeight || pParams->nDstWidth != pParams->nWidth) {
        pParams->vpp.bEnable = true;
        pParams->vpp.bUseResize = true;
    }

    if ((!pParams->nPAR[0] || !pParams->nPAR[1])
        && inputFrameInfo.AspectRatioW && inputFrameInfo.AspectRatioH
        && !pParams->vpp.bUseResize) {
        pParams->nPAR[0] = inputFrameInfo.AspectRatioW;
        pParams->nPAR[1] = inputFrameInfo.AspectRatioH;
    }

    if (pParams->nDstWidth % 2 != 0) {
        PrintMes(QSV_LOG_ERROR, _T("output width should be a multiple of 2."));
        return MFX_ERR_INVALID_VIDEO_PARAM;
    }

    if (pParams->nDstHeight % h_mul != 0) {
        PrintMes(QSV_LOG_ERROR, _T("output height should be a multiple of %d."), h_mul);
        return MFX_ERR_INVALID_VIDEO_PARAM;
    }

    //Cehck For Framerate
    if (pParams->nFPSRate == 0 || pParams->nFPSScale == 0) {
        PrintMes(QSV_LOG_ERROR, _T("unable to parse fps data.\n"));
        return MFX_ERR_INVALID_VIDEO_PARAM;
    }
    mfxU32 OutputFPSRate = pParams->nFPSRate;
    mfxU32 OutputFPSScale = pParams->nFPSScale;
    mfxU32 outputFrames = *(mfxU32 *)&inputFrameInfo.FrameId;
    if ((pParams->nPicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF))) {
        switch (pParams->vpp.nDeinterlace) {
        case MFX_DEINTERLACE_IT:
        case MFX_DEINTERLACE_IT_MANUAL:
            OutputFPSRate = OutputFPSRate * 4;
            OutputFPSScale = OutputFPSScale * 5;
            outputFrames = (outputFrames * 4) / 5;
            break;
        case MFX_DEINTERLACE_BOB:
        case MFX_DEINTERLACE_AUTO_DOUBLE:
            OutputFPSRate = OutputFPSRate * 2;
            outputFrames *= 2;
            break;
        default:
            break;
        }
    }
    switch (pParams->vpp.nFPSConversion) {
    case FPS_CONVERT_MUL2:
        OutputFPSRate = OutputFPSRate * 2;
        outputFrames *= 2;
        break;
    case FPS_CONVERT_MUL2_5:
        OutputFPSRate = OutputFPSRate * 5 / 2;
        outputFrames = outputFrames * 5 / 2;
        break;
    default:
        break;
    }
    mfxU32 gcd = qsv_gcd(OutputFPSRate, OutputFPSScale);
    OutputFPSRate /= gcd;
    OutputFPSScale /= gcd;
    m_pEncSatusInfo->Init(OutputFPSRate, OutputFPSScale, outputFrames, m_pQSVLog.get());
    PrintMes(QSV_LOG_DEBUG, _T("CheckParam: %dx%d%s, %d:%d, %d/%d, %d frames\n"),
        pParams->nDstWidth, pParams->nDstHeight, (output_interlaced) ? _T("i") : _T("p"),
        pParams->nPAR[0], pParams->nPAR[1], OutputFPSRate, OutputFPSScale, outputFrames);

    //デコードを行う場合は、入力バッファサイズを常に1に設定する (そうしないと正常に動かない)
    //また、バッファサイズを拡大しても特に高速化しない
    if (m_pFileReader->getInputCodec()) {
        pParams->nInputBufSize = 1;
        //HEVCデコーダを使用する場合はD3D11メモリを使用しないと正常に稼働しない (4080ドライバ)
        //if (m_pFileReader->getInputCodec() == MFX_CODEC_HEVC) {
        //    if (pParams->memType & D3D9_MEMORY) {
        //        pParams->memType &= ~D3D9_MEMORY;
        //        pParams->memType |= D3D11_MEMORY;
        //    }
        //    PrintMes(QSV_LOG_DEBUG, _T("Switched to d3d11 mode for HEVC encoding.\n"));
        //}
    }

    return MFX_ERR_NONE;
}

mfxStatus CEncodingPipeline::InitSessionInitParam(mfxU16 threads, mfxU16 priority) {
    INIT_MFX_EXT_BUFFER(m_ThreadsParam, MFX_EXTBUFF_THREADS_PARAM);
    m_ThreadsParam.NumThread = threads;
    m_ThreadsParam.Priority = priority;
    m_pInitParamExtBuf[0] = &m_ThreadsParam.Header;

    MSDK_ZERO_MEMORY(m_InitParam);
    m_InitParam.ExtParam = m_pInitParamExtBuf;
    m_InitParam.NumExtParam = 1;
    return MFX_ERR_NONE;
}

mfxStatus CEncodingPipeline::InitSession(bool useHWLib, mfxU16 memType) {
    mfxStatus sts = MFX_ERR_NONE;
    // init session, and set memory type
    mfxIMPL impl = 0;
    mfxVersion verRequired = MFX_LIB_VERSION_1_1;
    m_mfxSession.Close();
    PrintMes(QSV_LOG_DEBUG, _T("InitSession: Start initilaizing...\n"));

    auto InitSessionEx = [&]() {
#if ENABLE_SESSION_THREAD_CONFIG
        if (m_ThreadsParam.NumThread != 0 || m_ThreadsParam.Priority != get_value_from_chr(list_priority, _T("normal"))) {
            m_InitParam.Implementation = impl;
            m_InitParam.Version = MFX_LIB_VERSION_1_15;
            if (MFX_ERR_NONE == m_mfxSession.InitEx(m_InitParam)) {
                return MFX_ERR_NONE;
            } else {
                m_ThreadsParam.NumThread = 0;
                m_ThreadsParam.Priority = get_value_from_chr(list_priority, _T("normal"));
            }
        }
#endif
        return m_mfxSession.Init(impl, &verRequired);
    };

    if (useHWLib) {
        // try searching on all display adapters
        impl = MFX_IMPL_HARDWARE_ANY;
        m_memType = D3D9_MEMORY;

        //Win7でD3D11のチェックをやると、
        //デスクトップコンポジションが切られてしまう問題が発生すると報告を頂いたので、
        //D3D11をWin8以降に限定
        if (!check_OS_Win8orLater()) {
            memType &= (~D3D11_MEMORY);
            PrintMes(QSV_LOG_DEBUG, _T("InitSession: OS is Win7, do not check for d3d11 mode.\n"));
        }

        //D3D11モードは基本的には遅い模様なので、自動モードなら切る
        if (HW_MEMORY == (memType & HW_MEMORY) && false == check_if_d3d11_necessary()) {
            memType &= (~D3D11_MEMORY);
            PrintMes(QSV_LOG_DEBUG, _T("InitSession: d3d11 memory mode not required, switching to d3d9 memory mode.\n"));
        }

        for (int i_try_d3d11 = 0; i_try_d3d11 < 1 + (HW_MEMORY == (memType & HW_MEMORY)); i_try_d3d11++) {
            // if d3d11 surfaces are used ask the library to run acceleration through D3D11
            // feature may be unsupported due to OS or MSDK API version
#if MFX_D3D11_SUPPORT
            if (D3D11_MEMORY & memType) {
                if (0 == i_try_d3d11) {
                    impl |= MFX_IMPL_VIA_D3D11; //first try with d3d11 memory
                    m_memType = D3D11_MEMORY;
                    PrintMes(QSV_LOG_DEBUG, _T("InitSession: trying to init session for d3d11 mode.\n"));
                } else {
                    impl &= ~MFX_IMPL_VIA_D3D11; //turn of d3d11 flag and retry
                    m_memType = D3D9_MEMORY;
                    PrintMes(QSV_LOG_DEBUG, _T("InitSession: trying to init session for d3d9 mode.\n"));
                }
            }
#endif
            sts = InitSessionEx();

            // MSDK API version may not support multiple adapters - then try initialize on the default
            if (MFX_ERR_NONE != sts) {
                PrintMes(QSV_LOG_DEBUG, _T("InitSession: failed to init session for multi GPU mode, retry by single GPU mode.\n"));
                sts = m_mfxSession.Init((impl & (~MFX_IMPL_HARDWARE_ANY)) | MFX_IMPL_HARDWARE, &verRequired);
            }

            if (MFX_ERR_NONE == sts) {
                break;
            }
        }
        PrintMes(QSV_LOG_DEBUG, _T("InitSession: initialized using %s memory.\n"), (m_memType & D3D11_MEMORY) ? _T("d3d11") : _T("d3d9"));
    } else {
        impl = MFX_IMPL_SOFTWARE;
        sts = InitSessionEx();
        m_memType = SYSTEM_MEMORY;
        PrintMes(QSV_LOG_DEBUG, _T("InitSession: initialized with system memory.\n"));
    }
    //使用できる最大のversionをチェック
    m_mfxVer = get_mfx_lib_version(impl);
    PrintMes(QSV_LOG_DEBUG, _T("InitSession: mfx lib version: %d.%d\n"), m_mfxVer.Major, m_mfxVer.Minor);
    return sts;
}

mfxStatus CEncodingPipeline::Init(sInputParams *pParams)
{
    MSDK_CHECK_POINTER(pParams, MFX_ERR_NULL_PTR);

    mfxStatus sts = MFX_ERR_NONE;
    
    if (pParams->bBenchmark) {
        pParams->nAVMux = QSVENC_MUX_NONE;
        for (int i = 0; i < pParams->nAudioSelectCount; i++)
            MSDK_SAFE_FREE(pParams->ppAudioSelectList[i]);
        pParams->pAVMuxOutputFormat = _T("raw");
        PrintMes(QSV_LOG_DEBUG, _T("Param adjusted for benchmark mode.\n"));
    }

    sts = InitSessionInitParam(pParams->nSessionThreads, pParams->nSessionThreadPriority);
    if (sts < MFX_ERR_NONE) return sts;

    sts = InitInput(pParams);
    if (sts < MFX_ERR_NONE) return sts;

    sts = CheckParam(pParams);
    if (sts != MFX_ERR_NONE) return sts;

    sts = m_EncThread.Init(pParams->nInputBufSize);
    MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to allocate memory for thread control."));

    sts = InitSession(pParams->bUseHWLib, pParams->memType);
    MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to initialize encode session."));

    // create and init frame allocator
    sts = CreateAllocator();
    if (sts < MFX_ERR_NONE) return sts;

    sts = InitMfxEncParams(pParams);
    if (sts < MFX_ERR_NONE) return sts;

    sts = InitMfxVppParams(pParams);
    if (sts < MFX_ERR_NONE) return sts;

    sts = CreateVppExtBuffers(pParams);
    if (sts < MFX_ERR_NONE) return sts;

    sts = InitVppPrePlugins(pParams);
    if (sts < MFX_ERR_NONE) return sts;

    sts = InitVppPostPlugins(pParams);
    if (sts < MFX_ERR_NONE) return sts;

    sts = InitMfxDecParams();
    if (sts < MFX_ERR_NONE) return sts;

    sts = InitOutput(pParams);
    if (sts < MFX_ERR_NONE) return sts;

#if ENABLE_MVC_ENCODING
    sts = AllocAndInitMVCSeqDesc();
    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

    // MVC specific options
    if (MVC_ENABLED & m_MVCflags)
    {
        sts = AllocAndInitMVCSeqDesc();
        MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
    }
#endif

    // シーンチェンジ検出
    bool input_interlaced = 0 != (pParams->nPicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF));
    bool deinterlace_enabled = input_interlaced && (pParams->vpp.nDeinterlace != MFX_DEINTERLACE_NONE);
    bool deinterlace_normal = input_interlaced && (pParams->vpp.nDeinterlace == MFX_DEINTERLACE_NORMAL);
    if (m_nExPrm & (MFX_PRM_EX_VQP | MFX_PRM_EX_SCENE_CHANGE)) {
        if (m_SceneChange.Init(80, (deinterlace_enabled) ? m_mfxVppParams.mfx.FrameInfo.PicStruct : m_mfxEncParams.mfx.FrameInfo.PicStruct, pParams->nVQPStrength, pParams->nVQPSensitivity, 3, pParams->nGOPLength, deinterlace_normal))
            MSDK_CHECK_RESULT_MES(MFX_ERR_UNDEFINED_BEHAVIOR, MFX_ERR_NONE, MFX_ERR_UNDEFINED_BEHAVIOR, _T("Failed to start scenechange detection."));
        PrintMes(QSV_LOG_DEBUG, _T("Initialized Scene change detection.\n"));
    }

    // create encoder
    if (pParams->CodecId != MFX_CODEC_RAW) {
        m_pmfxENC = new MFXVideoENCODE(m_mfxSession);
        MSDK_CHECK_POINTER(m_pmfxENC, MFX_ERR_MEMORY_ALLOC);
    }

    // create preprocessor if resizing was requested from command line
    // or if different FourCC is set in InitMfxVppParams
    if (   pParams->nWidth  != pParams->nDstWidth
        || pParams->nHeight != pParams->nDstHeight
        || m_mfxVppParams.vpp.In.FourCC         != m_mfxVppParams.vpp.Out.FourCC
        || m_mfxVppParams.vpp.In.BitDepthLuma   != m_mfxVppParams.vpp.Out.BitDepthLuma
        || m_mfxVppParams.vpp.In.BitDepthChroma != m_mfxVppParams.vpp.Out.BitDepthChroma
        || m_mfxVppParams.NumExtParam > 1
        || pParams->vpp.nDeinterlace
        ) {
        PrintMes(QSV_LOG_DEBUG, _T("Vpp Enabled...\n"));
        m_pmfxVPP = new MFXVideoVPP(m_mfxSession);
        MSDK_CHECK_POINTER(m_pmfxVPP, MFX_ERR_MEMORY_ALLOC);
    }
    if (m_mfxVppParams.vpp.In.FourCC != m_mfxVppParams.vpp.Out.FourCC) {
        tstring mes = strsprintf(_T("ColorFmtConvertion: %s -> %s\n"), ColorFormatToStr(m_mfxVppParams.vpp.In.FourCC), ColorFormatToStr(m_mfxVppParams.vpp.Out.FourCC));
        PrintMes(QSV_LOG_DEBUG, _T("Vpp Enabled: %s\n"), mes.c_str());
        VppExtMes += mes;
    }
    if (pParams->nWidth  != pParams->nDstWidth ||
        pParams->nHeight != pParams->nDstHeight) {
        tstring mes = strsprintf(_T("Resizer, %dx%d -> %dx%d\n"), pParams->nWidth, pParams->nHeight, pParams->nDstWidth, pParams->nDstHeight);
        PrintMes(QSV_LOG_DEBUG, _T("Vpp Enabled: %s\n"), mes.c_str());
        VppExtMes += mes;
    }

    const int nPipelineElements = !!m_pmfxDEC + !!m_pmfxVPP + !!m_pmfxENC + (int)m_VppPrePlugins.size() + (int)m_VppPostPlugins.size();
    if (nPipelineElements == 0) {
        PrintMes(QSV_LOG_ERROR, _T("None of the pipeline element (DEC,VPP,ENC) are activated!\n"));
        return MFX_ERR_INVALID_VIDEO_PARAM;
    }
    PrintMes(QSV_LOG_DEBUG, _T("pipeline element count: %d\n"), nPipelineElements);

    // this number can be tuned for better performance
    m_nAsyncDepth = pParams->nAsyncDepth;
    if (m_nAsyncDepth == 0) {
        m_nAsyncDepth = (mfxU16)(std::min)(QSV_DEFAULT_ASYNC_DEPTH + (nPipelineElements - 1) * 2, (int)QSV_ASYNC_DEPTH_MAX);
        PrintMes(QSV_LOG_DEBUG, _T("async depth automatically set to %d\n"), m_nAsyncDepth);
    }

    if (!pParams->bDisableTimerPeriodTuning) {
        m_bTimerPeriodTuning = true;
        timeBeginPeriod(1);
        PrintMes(QSV_LOG_DEBUG, _T("timeBeginPeriod(1)\n"));
    }

    sts = ResetMFXComponents(pParams);
    if (sts < MFX_ERR_NONE) return sts;

    return MFX_ERR_NONE;
}

void CEncodingPipeline::Close()
{
    PrintMes(QSV_LOG_DEBUG, _T("Closing pipeline...\n"));
    //PrintMes(QSV_LOG_INFO, _T("Frame number: %hd\r"), m_pFileWriter.m_nProcessedFramesNum);

    MSDK_SAFE_DELETE(m_pEncSatusInfo);
    m_EncThread.Close();

    m_pDecPlugin.reset();
    m_pEncPlugin.reset();

    m_pTrimParam = NULL;

    MSDK_SAFE_DELETE(m_pmfxDEC);
    MSDK_SAFE_DELETE(m_pmfxENC);
    MSDK_SAFE_DELETE(m_pmfxVPP);
    m_VppPrePlugins.clear();
    m_VppPostPlugins.clear();

#if ENABLE_MVC_ENCODING
    FreeMVCSeqDesc();
#endif
    FreeVppDoNotUse();

    m_EncExtParams.clear();
    m_VppDoNotUseList.clear();
    m_VppDoUseList.clear();
    m_VppExtParams.clear();
    VppExtMes.clear();
    DeleteFrames();
    // allocator if used as external for MediaSDK must be deleted after SDK components
    DeleteAllocator();

    WipeMfxBitstream(&m_DecInputBitstream);

    m_TaskPool.Close();
    m_mfxSession.Close();

    m_SceneChange.Close();

    m_AudioReaders.clear();

    for (auto pWriter : m_pFileWriterListAudio) {
        if (pWriter) {
            if (pWriter != m_pFileWriter) {
                pWriter->Close();
                delete pWriter;
            }
        }
    }
    m_pFileWriterListAudio.clear();

    if (m_pFileWriter) {
        m_pFileWriter->Close();
        delete m_pFileWriter;
        m_pFileWriter = NULL;
    }

    if (m_pFileReader) {
        m_pFileReader->Close();
        delete m_pFileReader;
        m_pFileReader = NULL;
    }
    if (m_bTimerPeriodTuning) {
        timeEndPeriod(1);
        m_bTimerPeriodTuning = false;
        PrintMes(QSV_LOG_DEBUG, _T("timeEndPeriod(1)\n"));
    }
    
    m_pAbortByUser = NULL;
    m_nExPrm = 0x00;
    PrintMes(QSV_LOG_DEBUG, _T("Closed pipeline.\n"));
    if (m_pQSVLog.get() != nullptr) {
        m_pQSVLog->writeFileFooter();
        m_pQSVLog.reset();
    }
}

mfxStatus CEncodingPipeline::ResetMFXComponents(sInputParams* pParams)
{
    MSDK_CHECK_POINTER(pParams, MFX_ERR_NULL_PTR);

    mfxStatus sts = MFX_ERR_NONE;
    PrintMes(QSV_LOG_DEBUG, _T("ResetMFXComponents: Start...\n"));

    if (m_pmfxENC) {
        sts = m_pmfxENC->Close();
        MSDK_IGNORE_MFX_STS(sts, MFX_ERR_NOT_INITIALIZED);
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to reset encoder (fail on closing)."));
        PrintMes(QSV_LOG_DEBUG, _T("ResetMFXComponents: Enc closed.\n"));
    }

    if (m_pmfxVPP) {
        sts = m_pmfxVPP->Close();
        MSDK_IGNORE_MFX_STS(sts, MFX_ERR_NOT_INITIALIZED);
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to reset vpp (fail on closing)."));
        PrintMes(QSV_LOG_DEBUG, _T("ResetMFXComponents: Vpp closed.\n"));
    }

    if (m_pmfxDEC) {
        sts = m_pmfxDEC->Close();
        MSDK_IGNORE_MFX_STS(sts, MFX_ERR_NOT_INITIALIZED);
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to reset decoder (fail on closing)."));
        PrintMes(QSV_LOG_DEBUG, _T("ResetMFXComponents: Dec closed.\n"));
    }

    // free allocated frames
    DeleteFrames();
    PrintMes(QSV_LOG_DEBUG, _T("ResetMFXComponents: Frames deleted.\n"));

    m_TaskPool.Close();

    sts = AllocFrames();
    if (sts < MFX_ERR_NONE) return sts;
    PrintMes(QSV_LOG_DEBUG, _T("ResetMFXComponents: Frames allocated.\n"));

    if (m_pmfxENC) {
        sts = m_pmfxENC->Init(&m_mfxEncParams);
        if (MFX_WRN_PARTIAL_ACCELERATION == sts) {
            PrintMes(QSV_LOG_WARN, _T("ResetMFXComponents: partial acceleration on Encoding.\n"));
            MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
        }
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to initialize encoder."));
        PrintMes(QSV_LOG_DEBUG, _T("ResetMFXComponents: Enc initialized.\n"));
    }

    if (m_pmfxVPP) {
        sts = m_pmfxVPP->Init(&m_mfxVppParams);
        if (MFX_WRN_PARTIAL_ACCELERATION == sts) {
            PrintMes(QSV_LOG_WARN, _T("ResetMFXComponents: partial acceleration on vpp.\n"));
            MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
        }
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to initialize vpp."));
        PrintMes(QSV_LOG_DEBUG, _T("ResetMFXComponents: Vpp initialized.\n"));
    }

    if (m_pmfxDEC) {
        sts = m_pmfxDEC->Init(&m_mfxDecParams);
        if (MFX_WRN_PARTIAL_ACCELERATION == sts) {
            PrintMes(QSV_LOG_WARN, _T("ResetMFXComponents: partial acceleration on decoding.\n"));
            MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
        }
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to initialize decoder.\n"));
        PrintMes(QSV_LOG_DEBUG, _T("ResetMFXComponents: Dec initialized.\n"));
    }

    mfxU32 nEncodedDataBufferSize = m_mfxEncParams.mfx.FrameInfo.Width * m_mfxEncParams.mfx.FrameInfo.Height * 4;
    PrintMes(QSV_LOG_DEBUG, _T("ResetMFXComponents: Creating task pool, poolSize %d, bufsize %d KB.\n"), m_nAsyncDepth, nEncodedDataBufferSize >> 10);
    sts = m_TaskPool.Init(&m_mfxSession, m_pMFXAllocator, m_pFileWriter, m_pFrameWriter.get(), m_nAsyncDepth, nEncodedDataBufferSize, NULL);
    MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to initialize task pool for encoding."));
    PrintMes(QSV_LOG_DEBUG, _T("ResetMFXComponents: Created task pool.\n"));

    return MFX_ERR_NONE;
}

mfxStatus CEncodingPipeline::AllocateSufficientBuffer(mfxBitstream* pBS)
{
    MSDK_CHECK_POINTER(pBS, MFX_ERR_NULL_PTR);
    MSDK_CHECK_POINTER(m_pmfxENC, MFX_ERR_NOT_INITIALIZED);

    mfxVideoParam par;
    MSDK_ZERO_MEMORY(par);

    // find out the required buffer size
    mfxStatus sts = m_pmfxENC->GetVideoParam(&par);
    MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to get required output buffer size from encoder."));

    // reallocate bigger buffer for output
    sts = ExtendMfxBitstream(pBS, par.mfx.BufferSizeInKB * 1000 * max(1, par.mfx.BRCParamMultiplier));
    if (sts < MFX_ERR_NONE)
        PrintMes(QSV_LOG_ERROR, _T("Failed to allocate buffer for bitstream output.\n"));
    MSDK_CHECK_RESULT_SAFE(sts, MFX_ERR_NONE, sts, WipeMfxBitstream(pBS));

    return MFX_ERR_NONE;
}

mfxStatus CEncodingPipeline::GetFreeTask(sTask **ppTask)
{
    mfxStatus sts = MFX_ERR_NONE;

    sts = m_TaskPool.GetFreeTask(ppTask);
    if (MFX_ERR_NOT_FOUND == sts)
    {
        sts = SynchronizeFirstTask();
        MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

        // try again
        sts = m_TaskPool.GetFreeTask(ppTask);
    }

    return sts;
}

mfxStatus CEncodingPipeline::SynchronizeFirstTask()
{
    mfxStatus sts = m_TaskPool.SynchronizeFirstTask();

    return sts;
}

mfxStatus CEncodingPipeline::CheckSceneChange()
{
    PrintMes(QSV_LOG_DEBUG, _T("Starting Sub Thread...\n"));
    mfxStatus sts = MFX_ERR_NONE;

    const int bufferSize = m_EncThread.m_nFrameBuffer;
    sInputBufSys *pArrayInputBuf = m_EncThread.m_InputBuf;
    sInputBufSys *pInputBuf;

    mfxVideoParam videoPrm;
    MSDK_ZERO_MEMORY(videoPrm);
    m_pmfxENC->GetVideoParam(&videoPrm);

    m_frameTypeSim.Init(videoPrm.mfx.GopPicSize, videoPrm.mfx.GopRefDist-1, videoPrm.mfx.QPI, videoPrm.mfx.QPP, videoPrm.mfx.QPB,
        0 == (videoPrm.mfx.GopOptFlag & MFX_GOP_CLOSED), videoPrm.mfx.FrameInfo.FrameRateExtN / (double)videoPrm.mfx.FrameInfo.FrameRateExtD);
    //bool bInterlaced = (0 != (videoPrm.mfx.FrameInfo.PicStruct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF)));
    mfxU32 lastFrameFlag = 0;

    //入力ループ
    for (mfxU32 i_frames = 0; !m_EncThread.m_bthSubAbort; i_frames++) {
        pInputBuf = &pArrayInputBuf[i_frames % bufferSize];
        WaitForSingleObject(pInputBuf->heSubStart, INFINITE);

        m_EncThread.m_bthSubAbort |= ((m_EncThread.m_stsThread == MFX_ERR_MORE_DATA && i_frames == m_pEncSatusInfo->m_nInputFrames));

        if (!m_EncThread.m_bthSubAbort) {
            //フレームタイプとQP値の決定
            int qp_offset[2] = { 0, 0 };
            mfxU32 frameFlag = m_SceneChange.Check(pInputBuf->pFrameSurface, qp_offset);
            frameFlag = m_frameTypeSim.GetFrameType(!!((frameFlag | (lastFrameFlag>>8)) & MFX_FRAMETYPE_I));
            _InterlockedExchange((long *)&pInputBuf->frameFlag, (frameFlag & MFX_FRAMETYPE_I) ? frameFlag : 0x00); //frameFlagにはIDR,I,Ref以外は渡してはならない
            if (m_nExPrm & MFX_PRM_EX_VQP) {
                _InterlockedExchange((long *)&pInputBuf->AQP[0], m_frameTypeSim.CurrentQP(!!((frameFlag | (lastFrameFlag>>8)) & MFX_FRAMETYPE_I), qp_offset[0]));
            }
            m_frameTypeSim.ToNextFrame();
            if (m_nExPrm & MFX_PRM_EX_DEINT_BOB) {
                if (m_nExPrm & MFX_PRM_EX_VQP)
                    _InterlockedExchange((long *)&pInputBuf->AQP[1], m_frameTypeSim.CurrentQP(!!(frameFlag & MFX_FRAMETYPE_xI), qp_offset[1]));
                m_frameTypeSim.ToNextFrame();
            }
            if (m_nExPrm & MFX_PRM_EX_DEINT_NORMAL) {
                lastFrameFlag = frameFlag;
            }
        }

        SetEvent(pInputBuf->heInputDone);
    }
    
    PrintMes(QSV_LOG_DEBUG, _T("Sub Thread: finished.\n"));
    return sts;
}

unsigned int __stdcall CEncodingPipeline::RunEncThreadLauncher(void *pParam) {
    reinterpret_cast<CEncodingPipeline*>(pParam)->RunEncode();
    _endthreadex(0);
    return 0;
}

unsigned int __stdcall CEncodingPipeline::RunSubThreadLauncher(void *pParam) {
    reinterpret_cast<CEncodingPipeline*>(pParam)->CheckSceneChange();
    _endthreadex(0);
    return 0;
}

mfxStatus CEncodingPipeline::Run()
{
    return Run(NULL);
}

mfxStatus CEncodingPipeline::Run(DWORD_PTR SubThreadAffinityMask)
{
    mfxStatus sts = MFX_ERR_NONE;
    sts = m_EncThread.RunEncFuncbyThread(RunEncThreadLauncher, this, SubThreadAffinityMask);
    MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to start encode thread."));
    if (m_SceneChange.isInitialized()) {
        sts = m_EncThread.RunSubFuncbyThread(RunSubThreadLauncher, this, SubThreadAffinityMask);
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Failed to start encode sub thread."));
    }
    PrintMes(QSV_LOG_DEBUG, _T("Main Thread: Starting Encode...\n"));

    const int bufferSize = m_EncThread.m_nFrameBuffer;
    sInputBufSys *pArrayInputBuf = m_EncThread.m_InputBuf;
    sInputBufSys *pInputBuf;
    //入力ループ
    for (int i = 0; sts == MFX_ERR_NONE; i++) {
        pInputBuf = &pArrayInputBuf[i % bufferSize];
        //PrintMes(QSV_LOG_INFO, _T("run loop: wait for %d\n"), i);
        //PrintMes(QSV_LOG_INFO, _T("wait for heInputStart %d\n"), i);

        //空いているフレームがセットされるのを待機
        while (WAIT_TIMEOUT == WaitForSingleObject(pInputBuf->heInputStart, 10000)) {
            //エンコードスレッドが異常終了していたら、それを検知してこちらも終了
            DWORD exit_code = 0;
            if (0 == GetExitCodeThread(m_EncThread.GetHandleEncThread(), &exit_code) || exit_code != STILL_ACTIVE) {
                PrintMes(QSV_LOG_ERROR, _T("error at encode thread.\n"));
                sts = MFX_ERR_INVALID_HANDLE;
                break;
            }
            if (m_SceneChange.isInitialized()
                && (0 == GetExitCodeThread(m_EncThread.GetHandleSubThread(), &exit_code) || exit_code != STILL_ACTIVE)) {
                    PrintMes(QSV_LOG_ERROR, _T("error at sub thread.\n"));
                    sts = MFX_ERR_INVALID_HANDLE;
                    break;
            }
        }
        //PrintMes(QSV_LOG_INFO, _T("load next frame %d to %d\n"), i, pInputBuf->pFrameSurface);

        //フレームを読み込み
        if (!sts)
            sts = m_pFileReader->LoadNextFrame(pInputBuf->pFrameSurface);
        if (NULL != m_pAbortByUser && *m_pAbortByUser) {
            PrintMes(QSV_LOG_INFO, _T("                                                                         \r"));
            sts = MFX_ERR_ABORTED;
        }
        //PrintMes(QSV_LOG_INFO, _T("set for heInputDone %d\n"), i);

        //フレームの読み込み終了を通知
        SetEvent((m_SceneChange.isInitialized()) ? pInputBuf->heSubStart : pInputBuf->heInputDone);
    }
    m_EncThread.WaitToFinish(sts, m_pQSVLog.get());
    PrintMes(QSV_LOG_DEBUG, _T("Main Thread: Finished Main Loop...\n"));

    sFrameTypeInfo info = { 0 };
    if (m_nExPrm & MFX_PRM_EX_VQP)
        m_frameTypeSim.getFrameInfo(&info);
    m_pEncSatusInfo->WriteResults((m_nExPrm & MFX_PRM_EX_VQP) ? &info : NULL);

    sts = min(sts, m_EncThread.m_stsThread);
    MSDK_IGNORE_MFX_STS(sts, MFX_ERR_MORE_DATA);

    m_EncThread.Close();
    
    PrintMes(QSV_LOG_DEBUG, _T("Main Thread: finished.\n"));
    return sts;
}

mfxStatus CEncodingPipeline::RunEncode()
{
    PrintMes(QSV_LOG_DEBUG, _T("Encode Thread: Starting Encode...\n"));

    mfxStatus sts = MFX_ERR_NONE;

    mfxFrameSurface1 *pSurfInputBuf = NULL;
    mfxFrameSurface1 *pSurfEncIn = NULL;
    mfxFrameSurface1 *pSurfVppIn = NULL;
    vector<mfxFrameSurface1 *>pSurfVppPreFilter(m_VppPrePlugins.size() + 1, NULL);
    vector<mfxFrameSurface1 *>pSurfVppPostFilter(m_VppPostPlugins.size() + 1, NULL);
    mfxFrameSurface1 *pNextFrame;
    mfxSyncPoint lastSyncP = nullptr;
    bool bVppRequireMoreFrame = false;
    int nFramePutToEncoder = 0; //エンコーダに投入したフレーム数 (TimeStamp計算用)
    const double getTimeStampMul = m_mfxEncParams.mfx.FrameInfo.FrameRateExtD * (double)QSV_TIMEBASE / (double)m_mfxEncParams.mfx.FrameInfo.FrameRateExtN; //TimeStamp計算用

    sTask *pCurrentTask = NULL; // a pointer to the current task
    int nEncSurfIdx = -1; // index of free surface for encoder input (vpp output)
    int nVppSurfIdx = -1; // index of free surface for vpp input

    bool bVppMultipleOutput = false;  // this flag is true if VPP produces more frames at output
                                      // than consumes at input. E.g. framerate conversion 30 fps -> 60 fps

    int nInputFrameCount = -1; //入力されたフレームの数 (最初のフレームが0になるよう、-1で初期化する)

    mfxU16 nLastFrameFlag = 0;
    int nLastAQ = 0;
    bool bVppDeintBobFirstFeild = true;

    m_pEncSatusInfo->SetStart();

#if ENABLE_AVCODEC_QSV_READER
    //streamのindexから必要なwriteへのポインタを返すテーブルを作成
    vector<CAvcodecWriter *> pWriterForAudioStreams;
    for (auto pWriter : m_pFileWriterListAudio) {
        auto pAVCodecWriter = dynamic_cast<CAvcodecWriter *>(pWriter);
        if (pAVCodecWriter) {
            auto indexes = pAVCodecWriter->GetAudioStreamIndex();
            for (auto index : indexes) {
                if (index >= (int)pWriterForAudioStreams.size()) {
                    pWriterForAudioStreams.resize(index+1, NULL);
                }
                pWriterForAudioStreams[index] = pAVCodecWriter;
            }
        }
    }
#endif

#if ENABLE_MVC_ENCODING
    // Since in sample we support just 2 views
    // we will change this value between 0 and 1 in case of MVC
    mfxU16 currViewNum = 0;
#endif

    sts = MFX_ERR_NONE;

    auto get_all_free_surface =[&](mfxFrameSurface1 *pSurfEncInput) {
        //パイプラインの後ろからたどっていく
        pSurfInputBuf = pSurfEncInput; //pSurfEncInにはパイプラインを後ろからたどった順にフレームポインタを更新していく
        pSurfVppPostFilter[m_VppPostPlugins.size()] = pSurfInputBuf; //pSurfVppPreFilterの最後はその直前のステップのフレームに出力される
        for (int i_filter = (int)m_VppPostPlugins.size()-1; i_filter >= 0; i_filter--) {
            int freeSurfIdx = GetFreeSurface(m_VppPostPlugins[i_filter]->m_pPluginSurfaces.get(), m_VppPostPlugins[i_filter]->m_PluginResponse.NumFrameActual);
            pSurfVppPostFilter[i_filter] = &m_VppPostPlugins[i_filter]->m_pPluginSurfaces[freeSurfIdx];
            pSurfInputBuf = pSurfVppPostFilter[i_filter];
        }
        // if vpp is enabled find free surface for vpp input and point pSurf to vpp surface
        if (m_pmfxVPP) {
            //空いているフレームバッファを取得、空いていない場合は待機して、空くまで待ってから取得
            nVppSurfIdx = GetFreeSurface(m_pVppSurfaces, m_VppResponse.NumFrameActual);
            pSurfVppIn = &m_pVppSurfaces[nVppSurfIdx];
            MSDK_CHECK_ERROR(nVppSurfIdx, MSDK_INVALID_SURF_IDX, MFX_ERR_MEMORY_ALLOC);
            pSurfInputBuf = pSurfVppIn;
        }
        pSurfVppPreFilter[m_VppPrePlugins.size()] = pSurfInputBuf; //pSurfVppPreFilterの最後はその直前のステップのフレームに出力される
        for (int i_filter = (int)m_VppPrePlugins.size()-1; i_filter >= 0; i_filter--) {
            int freeSurfIdx = GetFreeSurface(m_VppPrePlugins[i_filter]->m_pPluginSurfaces.get(), m_VppPrePlugins[i_filter]->m_PluginResponse.NumFrameActual);
            pSurfVppPreFilter[i_filter] = &m_VppPrePlugins[i_filter]->m_pPluginSurfaces[freeSurfIdx];
            pSurfInputBuf = pSurfVppPreFilter[i_filter];
        }
        //最終的にpSurfInputBufには一番最初のステップのフレームポインタが入る
        return MFX_ERR_NONE;
    };

    auto set_surface_to_input_buffer = [&]() {
        mfxStatus sts_set_buffer = MFX_ERR_NONE;
        for (int i = 0; i < m_EncThread.m_nFrameBuffer; i++) {
            get_all_free_surface(&m_pEncSurfaces[GetFreeSurface(m_pEncSurfaces, m_EncResponse.NumFrameActual)]);

            if (m_bExternalAlloc) {
                sts_set_buffer = m_pMFXAllocator->Lock(m_pMFXAllocator->pthis, pSurfInputBuf->Data.MemId, &(pSurfInputBuf->Data));
                MSDK_BREAK_ON_ERROR(sts_set_buffer);
            }
            //空いているフレームを読み込み側に渡し、該当フレームの読み込み開始イベントをSetする(pInputBuf->heInputStart)
            m_pFileReader->SetNextSurface(pSurfInputBuf);
#if ENABLE_MVC_ENCODING
            m_pEncSurfaces[nEncSurfIdx].Info.FrameId.ViewId = currViewNum;
            if (m_bIsMVC) currViewNum ^= 1; // Flip between 0 and 1 for ViewId
#endif
        }
        return sts_set_buffer;
    };
    
    //先読みバッファ用フレームを読み込み側に提供する
    set_surface_to_input_buffer();
    PrintMes(QSV_LOG_DEBUG, _T("Encode Thread: Set surface to input buffer...\n"));

    auto copy_crop_info = [](mfxFrameSurface1 *dst, const mfxFrameInfo *src) {
        if (NULL != dst) {
            dst->Info.CropX = src->CropX;
            dst->Info.CropY = src->CropY;
            dst->Info.CropW = src->CropW;
            dst->Info.CropH = src->CropH;
        }
    };

    auto extract_audio =[&]() {
        mfxStatus sts = MFX_ERR_NONE;
#if ENABLE_AVCODEC_QSV_READER
        if (m_pFileWriterListAudio.size()) {
            auto pAVCodecReader = dynamic_cast<CAvcodecReader *>(m_pFileReader);
            vector<AVPacket> packetList;
            if (pAVCodecReader != nullptr) {
                packetList = pAVCodecReader->GetAudioDataPackets();
            }
            //音声ファイルリーダーからのトラックを結合する
            for (const auto& reader : m_AudioReaders) {
                auto pReader = dynamic_cast<CAvcodecReader *>(reader.get());
                if (pReader != nullptr) {
                    auto list = pReader->GetAudioDataPackets();
                    if (list.size()) {
                        packetList.insert(packetList.end(), list.begin(), list.end());
                    }
                }
            }
            //パケットを各Writerに分配する
            for (mfxU32 i = 0; i < packetList.size(); i++) {
                auto pWriter = pWriterForAudioStreams[packetList[i].stream_index];
                if (pWriter == nullptr) {
                    PrintMes(QSV_LOG_ERROR, _T("Failed to find writer for audio stream %d\n"), packetList[i].stream_index);
                    return MFX_ERR_NULL_PTR;
                }
                if (MFX_ERR_NONE != (sts = pWriter->WriteNextPacket(&packetList[i]))) {
                    return sts;
                }
            }
        }
#endif //ENABLE_AVCODEC_QSV_READER
        return sts;
    };

    auto decode_one_frame = [&](bool getNextBitstream) {
        mfxStatus dec_sts = MFX_ERR_NONE;
        if (m_pmfxDEC) {
            if (getNextBitstream) {
                //この関数がMFX_ERR_NONE以外を返せば、入力ビットストリームは終了
                dec_sts = m_pFileReader->GetNextBitstream(&m_DecInputBitstream);
                MSDK_IGNORE_MFX_STS(dec_sts, MFX_ERR_MORE_DATA);
                MSDK_CHECK_RESULT_MES(dec_sts, MFX_ERR_NONE, dec_sts, _T("Error on getting video bitstream."));
            }

            getNextBitstream |= 0 < m_DecInputBitstream.DataLength;

            //デコードも行う場合は、デコード用のフレームをpSurfVppInかpSurfEncInから受け取る
            mfxFrameSurface1 *pSurfDecWork = pNextFrame;
            mfxFrameSurface1 *pSurfDecOut = NULL;
            mfxBitstream *pInputBitstream = (getNextBitstream) ? &m_DecInputBitstream : nullptr;

            //デコード前には、デコード用のパラメータでFrameInfoを更新
            copy_crop_info(pSurfDecWork, &m_mfxDecParams.mfx.FrameInfo);

            for (int i = 0; ; i++) {
                mfxSyncPoint DecSyncPoint = NULL;
                dec_sts = m_pmfxDEC->DecodeFrameAsync(pInputBitstream, pSurfDecWork, &pSurfDecOut, &DecSyncPoint);
                lastSyncP = DecSyncPoint;

                if (MFX_ERR_NONE < dec_sts && !DecSyncPoint) {
                    if (MFX_WRN_DEVICE_BUSY == dec_sts)
                        sleep_hybrid(i); // wait if device is busy
                    if (i > 1024 * 1024 * 30) {
                        PrintMes(QSV_LOG_ERROR, _T("device kept on busy for 30s, unknown error occurred.\n"));
                        return MFX_ERR_UNKNOWN;
                    }
                } else if (MFX_ERR_NONE < dec_sts && DecSyncPoint) {
                    dec_sts = MFX_ERR_NONE; // ignore warnings if output is available
                    break;
                } else if (dec_sts < MFX_ERR_NONE && (dec_sts != MFX_ERR_MORE_DATA && dec_sts != MFX_ERR_MORE_SURFACE)) {
                    PrintMes(QSV_LOG_ERROR, _T("DecodeFrameAsync error: %s.\n"), get_err_mes(dec_sts));
                    break;
                } else {
                    break; // not a warning
                }
            }

            //次のステップのフレームをデコードの出力に設定
            pNextFrame = pSurfDecOut;
            nInputFrameCount += (pSurfDecOut != NULL);
        }
        return dec_sts;
    };

    auto filter_one_frame = [&](const unique_ptr<CVPPPlugin>& filter, mfxFrameSurface1** ppSurfIn, mfxFrameSurface1** ppSurfOut) {
        mfxStatus filter_sts = MFX_ERR_NONE;
        mfxSyncPoint filterSyncPoint = NULL;

        for (int i = 0; ; i++) {
            mfxHDL *h1 = (mfxHDL *)ppSurfIn;
            mfxHDL *h2 = (mfxHDL *)ppSurfOut;

            filter_sts = MFXVideoUSER_ProcessFrameAsync(filter->getSession(), h1, 1, h2, 1, &filterSyncPoint);

            if (MFX_WRN_DEVICE_BUSY == filter_sts) {
                sleep_hybrid(i);
                if (i > 1024 * 1024 * 30) {
                    PrintMes(QSV_LOG_ERROR, _T("device kept on busy for 30s, unknown error occurred.\n"));
                    return MFX_ERR_UNKNOWN;
                }
            } else {
                break;
            }
        }
        // save the id of preceding vpp task which will produce input data for the encode task
        if (filterSyncPoint) {
            lastSyncP = filterSyncPoint;
            //pCurrentTask->DependentVppTasks.push_back(filterSyncPoint);
            filterSyncPoint = NULL;
        }
        return filter_sts;
    };

    auto vpp_one_frame =[&](mfxFrameSurface1* pSurfVppIn, mfxFrameSurface1* pSurfVppOut) {
        mfxStatus vpp_sts = MFX_ERR_NONE;
        if (m_pmfxVPP) {
            if (bVppMultipleOutput && pSurfVppIn) {
                //bob化など、デコーダから出てきたフレームでないフレーム(=TimeStampが計算されていない)を途中ではさむ場合には、
                //下記のようにTimeStampをUNKNOWNに設定してやることで、自動計算される
                pSurfVppIn->Data.TimeStamp = (mfxU64)MFX_TIMESTAMP_UNKNOWN;
            }
            mfxSyncPoint VppSyncPoint = NULL; // a sync point associated with an asynchronous vpp call
            bVppMultipleOutput = false;   // reset the flag before a call to VPP
            bVppRequireMoreFrame = false; // reset the flag before a call to VPP

            //vpp前に、vpp用のパラメータでFrameInfoを更新
            copy_crop_info(pSurfVppIn, &m_mfxVppParams.mfx.FrameInfo);

            for (int i = 0; ; i++) {
                vpp_sts = m_pmfxVPP->RunFrameVPPAsync(pSurfVppIn, pSurfVppOut, NULL, &VppSyncPoint);
                lastSyncP = VppSyncPoint;

                if (MFX_ERR_NONE < vpp_sts && !VppSyncPoint) { // repeat the call if warning and no output
                    if (MFX_WRN_DEVICE_BUSY == vpp_sts)
                        sleep_hybrid(i); // wait if device is busy
                    if (i > 1024 * 1024 * 30) {
                        PrintMes(QSV_LOG_ERROR, _T("device kept on busy for 30s, unknown error occurred.\n"));
                        return MFX_ERR_UNKNOWN;
                    }
                } else if (MFX_ERR_NONE < vpp_sts && VppSyncPoint) {
                    vpp_sts = MFX_ERR_NONE; // ignore warnings if output is available
                    break;
                } else
                    break; // not a warning
            }

            // process errors
            if (MFX_ERR_MORE_DATA == vpp_sts) {
                bVppRequireMoreFrame = true;
            } else if (MFX_ERR_MORE_SURFACE == vpp_sts) {
                bVppMultipleOutput = true;
                vpp_sts = MFX_ERR_NONE;
            }

            // save the id of preceding vpp task which will produce input data for the encode task
            if (VppSyncPoint) {
                pCurrentTask->DependentVppTasks.push_back(VppSyncPoint);
                VppSyncPoint = NULL;
                pNextFrame = pSurfVppOut;
            }
        }
        return vpp_sts;
    };

    auto encode_one_frame =[&](mfxFrameSurface1* pSurfEncIn) {
        if (m_pmfxENC == nullptr) {
            //エンコードが有効でない場合、このフレームデータを出力する
            //パイプラインの最後のSyncPointをセットする
            pCurrentTask->EncSyncP = lastSyncP;
            //フレームデータが出力されるまで空きフレームとして使われないようLockを加算しておく
            //TaskのWriteBitstreamで減算され、解放される
            pSurfEncIn->Data.Locked++;
            //フレームのポインタを出力用にセット
            pCurrentTask->mfxSurf = pSurfEncIn;
            return MFX_ERR_NONE;
        }

        mfxStatus enc_sts = MFX_ERR_NONE;
        mfxEncodeCtrl *ptrCtrl = NULL;
        mfxEncodeCtrl encCtrl = { 0 };

        //以下の処理は
        if (pSurfEncIn) {
            if (m_nExPrm & (MFX_PRM_EX_SCENE_CHANGE | MFX_PRM_EX_VQP)) {
                if (m_nExPrm & MFX_PRM_EX_DEINT_NORMAL) {
                    mfxU32 currentFrameFlag = m_EncThread.m_InputBuf[pSurfEncIn->Data.TimeStamp].frameFlag;
                    if (nLastFrameFlag >> 8) {
                        encCtrl.FrameType = nLastFrameFlag >> 8;
                        encCtrl.QP = (mfxU16)nLastAQ;
                    } else {
                        encCtrl.FrameType = currentFrameFlag & 0xff;
                        encCtrl.QP = (mfxU16)m_EncThread.m_InputBuf[pSurfEncIn->Data.TimeStamp].AQP[0];
                    }
                    nLastFrameFlag = (mfxU16)currentFrameFlag;
                    nLastAQ = m_EncThread.m_InputBuf[pSurfEncIn->Data.TimeStamp].AQP[1];
                    pSurfEncIn->Data.TimeStamp = 0;
                } else if (m_nExPrm & MFX_PRM_EX_DEINT_BOB) {
                    if (bVppDeintBobFirstFeild) {
                        nLastFrameFlag = (mfxU16)m_EncThread.m_InputBuf[pSurfEncIn->Data.TimeStamp].frameFlag;
                        nLastAQ = m_EncThread.m_InputBuf[pSurfEncIn->Data.TimeStamp].AQP[1];
                        encCtrl.QP = (mfxU16)m_EncThread.m_InputBuf[pSurfEncIn->Data.TimeStamp].AQP[0];
                        encCtrl.FrameType = nLastFrameFlag & 0xff;
                        pSurfEncIn->Data.TimeStamp = 0;
                    } else {
                        encCtrl.FrameType = nLastFrameFlag >> 8;
                        encCtrl.QP = (mfxU16)nLastAQ;
                    }
                    bVppDeintBobFirstFeild ^= true;
                } else {
                    encCtrl.FrameType = (mfxU16)m_EncThread.m_InputBuf[pSurfEncIn->Data.TimeStamp].frameFlag;
                    encCtrl.QP = (mfxU16)m_EncThread.m_InputBuf[pSurfEncIn->Data.TimeStamp].AQP[0];
                    pSurfEncIn->Data.TimeStamp = 0;
                }
                ptrCtrl = &encCtrl;
            }
            //TimeStampを適切に設定してやると、BitstreamにTimeStamp、DecodeTimeStampが計算される
            pSurfEncIn->Data.TimeStamp = (int)(nFramePutToEncoder * getTimeStampMul + 0.5);
            nFramePutToEncoder++;
            //TimeStampをMFX_TIMESTAMP_UNKNOWNにしておくと、きちんと設定される
            pCurrentTask->mfxBS.TimeStamp = (mfxU64)MFX_TIMESTAMP_UNKNOWN;
            pCurrentTask->mfxBS.DecodeTimeStamp = (mfxU64)MFX_TIMESTAMP_UNKNOWN;
        }

        bool bDeviceBusy = false;
        for (int i = 0; ; i++) {
            enc_sts = m_pmfxENC->EncodeFrameAsync(ptrCtrl, pSurfEncIn, &pCurrentTask->mfxBS, &pCurrentTask->EncSyncP);
            bDeviceBusy = false;

            if (MFX_ERR_NONE < enc_sts && !pCurrentTask->EncSyncP) { // repeat the call if warning and no output
                bDeviceBusy = true;
                if (MFX_WRN_DEVICE_BUSY == enc_sts)
                sleep_hybrid(i);
                if (i > 1024 * 1024 * 30) {
                    PrintMes(QSV_LOG_ERROR, _T("device kept on busy for 30s, unknown error occurred.\n"));
                    return MFX_ERR_UNKNOWN;
                }
            } else if (MFX_ERR_NONE < enc_sts && pCurrentTask->EncSyncP) {
                enc_sts = MFX_ERR_NONE; // ignore warnings if output is available
                break;
            } else if (MFX_ERR_NOT_ENOUGH_BUFFER == enc_sts) {
                enc_sts = AllocateSufficientBuffer(&pCurrentTask->mfxBS);
                if (enc_sts < MFX_ERR_NONE) return enc_sts;
            } else if (enc_sts < MFX_ERR_NONE && (enc_sts != MFX_ERR_MORE_DATA && enc_sts != MFX_ERR_MORE_SURFACE)) {
                PrintMes(QSV_LOG_ERROR, _T("EncodeFrameAsync error: %s.\n"), get_err_mes(enc_sts));
                break;
            } else {
                // get next surface and new task for 2nd bitstream in ViewOutput mode
                MSDK_IGNORE_MFX_STS(enc_sts, MFX_ERR_MORE_BITSTREAM);
                break;
            }
        }
        return enc_sts;
    };

    // main loop, preprocessing and encoding
    while (MFX_ERR_NONE <= sts || MFX_ERR_MORE_DATA == sts || MFX_ERR_MORE_SURFACE == sts)
    {
        // get a pointer to a free task (bit stream and sync point for encoder)
        //空いているフレームバッファを取得、空いていない場合は待機して、出力ストリームの書き出しを待ってから取得
        sts = GetFreeTask(&pCurrentTask);
        MSDK_BREAK_ON_ERROR(sts);

        // find free surface for encoder input
        //空いているフレームバッファを取得、空いていない場合は待機して、空くまで待ってから取得
        nEncSurfIdx = GetFreeSurface(m_pEncSurfaces, m_EncResponse.NumFrameActual);
        MSDK_CHECK_ERROR(nEncSurfIdx, MSDK_INVALID_SURF_IDX, MFX_ERR_MEMORY_ALLOC);

        // point pSurf to encoder surface
        pSurfEncIn = &m_pEncSurfaces[nEncSurfIdx];

        if (!bVppMultipleOutput)
        {
            get_all_free_surface(pSurfEncIn);
            //if (m_VppPrePlugins.size()) {
            //    pSurfInputBuf = pSurfVppPreFilter[0];
            //    //ppNextFrame = &;
            //} else if (m_pmfxVPP) {
            //    pSurfInputBuf = &m_pVppSurfaces[nVppSurfIdx];
            //    //ppNextFrame = &pSurfVppIn;
            //} else if (m_VppPostPlugins.size()) {
            //    pSurfInputBuf = &pSurfVppPostFilter[0];
            //    //ppNextFrame = &;
            //} else {
            //    pSurfInputBuf = pSurfEncIn;
            //    //ppNextFrame = &pSurfEncIn;
            //}
            //読み込み側の該当フレームの読み込み終了を待機(pInputBuf->heInputDone)して、読み込んだフレームを取得
            //この関数がMFX_ERR_NONE以外を返すことでRunEncodeは終了処理に入る
            sts = m_pFileReader->GetNextFrame(&pNextFrame);
            MSDK_BREAK_ON_ERROR(sts);

            if (m_bExternalAlloc) {
                sts = m_pMFXAllocator->Unlock(m_pMFXAllocator->pthis, (pNextFrame)->Data.MemId, &((pNextFrame)->Data));
                MSDK_BREAK_ON_ERROR(sts);

                sts = m_pMFXAllocator->Lock(m_pMFXAllocator->pthis, pSurfInputBuf->Data.MemId, &(pSurfInputBuf->Data));
                MSDK_BREAK_ON_ERROR(sts);
            }

            //空いているフレームを読み込み側に渡す
            m_pFileReader->SetNextSurface(pSurfInputBuf);
#if ENABLE_MVC_ENCODING
            pSurfInputBuf->Info.FrameId.ViewId = currViewNum;
            if (m_bIsMVC) currViewNum ^= 1; // Flip between 0 and 1 for ViewId
#endif
            sts = extract_audio();
            MSDK_BREAK_ON_ERROR(sts);

            sts = decode_one_frame(true);
            if (sts == MFX_ERR_MORE_DATA || sts == MFX_ERR_MORE_SURFACE)
                continue;
            MSDK_BREAK_ON_ERROR(sts);

            for (int i_filter = 0; i_filter < (int)m_VppPrePlugins.size(); i_filter++) {
                mfxFrameSurface1 *pSurfFilterOut = pSurfVppPreFilter[i_filter + 1];
                sts = filter_one_frame(m_VppPrePlugins[i_filter], &pNextFrame, &pSurfFilterOut);
                MSDK_BREAK_ON_ERROR(sts);
                pNextFrame = pSurfFilterOut;
            }
            MSDK_BREAK_ON_ERROR(sts);

            pSurfVppIn = pNextFrame;
        }

        if (m_pTrimParam && !frame_inside_range(nInputFrameCount, m_pTrimParam->list))
            continue;

        sts = vpp_one_frame(pSurfVppIn, (m_VppPostPlugins.size()) ? pSurfVppPostFilter[0] : pSurfEncIn);
        if (bVppRequireMoreFrame)
            continue;
        MSDK_BREAK_ON_ERROR(sts);

        for (int i_filter = 0; i_filter < (int)m_VppPostPlugins.size(); i_filter++) {
            mfxFrameSurface1 *pSurfFilterOut = pSurfVppPostFilter[i_filter + 1];
            sts = filter_one_frame(m_VppPostPlugins[i_filter], &pNextFrame, &pSurfFilterOut);
            MSDK_BREAK_ON_ERROR(sts);
            pNextFrame = pSurfFilterOut;
        }
        MSDK_BREAK_ON_ERROR(sts);
        
        sts = encode_one_frame(pNextFrame);
    }
    
    // means that the input file has ended, need to go to buffering loops
    MSDK_IGNORE_MFX_STS(sts, MFX_ERR_MORE_DATA);
    // exit in case of other errors
    MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Error in encoding pipeline."));
    PrintMes(QSV_LOG_DEBUG, _T("Encode Thread: finished main loop.\n"));

    if (m_pmfxDEC)
    {
        sts = extract_audio();
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Error on extracting audio."));

        pNextFrame = NULL;

        while (MFX_ERR_NONE <= sts || sts == MFX_ERR_MORE_SURFACE) {
            // get a pointer to a free task (bit stream and sync point for encoder)
            //空いているフレームバッファを取得、空いていない場合は待機して、出力ストリームの書き出しを待ってから取得
            sts = GetFreeTask(&pCurrentTask);
            MSDK_BREAK_ON_ERROR(sts);

            // find free surface for encoder input
            //空いているフレームバッファを取得、空いていない場合は待機して、空くまで待ってから取得
            nEncSurfIdx = GetFreeSurface(m_pEncSurfaces, m_EncResponse.NumFrameActual);
            MSDK_CHECK_ERROR(nEncSurfIdx, MSDK_INVALID_SURF_IDX, MFX_ERR_MEMORY_ALLOC);

            // point pSurf to encoder surface
            pSurfEncIn = &m_pEncSurfaces[nEncSurfIdx];

            if (!bVppMultipleOutput)
            {
                get_all_free_surface(pSurfEncIn);
                pNextFrame = pSurfInputBuf;

                sts = decode_one_frame(false);
                if (sts == MFX_ERR_MORE_SURFACE)
                    continue;
                MSDK_BREAK_ON_ERROR(sts);

                for (int i_filter = 0; i_filter < (int)m_VppPrePlugins.size(); i_filter++) {
                    mfxFrameSurface1 *pSurfFilterOut = pSurfVppPreFilter[i_filter + 1];
                    sts = filter_one_frame(m_VppPrePlugins[i_filter], &pNextFrame, &pSurfFilterOut);
                    MSDK_BREAK_ON_ERROR(sts);
                    pNextFrame = pSurfFilterOut;
                }
                MSDK_BREAK_ON_ERROR(sts);

                pSurfVppIn = pNextFrame;
            }

            if (m_pTrimParam && !frame_inside_range(nInputFrameCount, m_pTrimParam->list))
                continue;

            sts = vpp_one_frame(pSurfVppIn, (m_VppPostPlugins.size()) ? pSurfVppPostFilter[0] : pSurfEncIn);
            if (bVppRequireMoreFrame)
                continue;
            MSDK_BREAK_ON_ERROR(sts);

            for (int i_filter = 0; i_filter < (int)m_VppPostPlugins.size(); i_filter++) {
                mfxFrameSurface1 *pSurfFilterOut = pSurfVppPostFilter[i_filter + 1];
                sts = filter_one_frame(m_VppPostPlugins[i_filter], &pNextFrame, &pSurfFilterOut);
                MSDK_BREAK_ON_ERROR(sts);
                pNextFrame = pSurfFilterOut;
            }
            MSDK_BREAK_ON_ERROR(sts);
        
            sts = encode_one_frame(pNextFrame);
        }

        // MFX_ERR_MORE_DATA is the correct status to exit buffering loop with
        // indicates that there are no more buffered frames
        MSDK_IGNORE_MFX_STS(sts, MFX_ERR_MORE_DATA);
        // exit in case of other errors
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Error in getting buffered frames from decoder."));
        PrintMes(QSV_LOG_DEBUG, _T("Encode Thread: finished getting buffered frames from decoder.\n"));
    }

#if ENABLE_AVCODEC_QSV_READER
    for (const auto& writer : m_pFileWriterListAudio) {
        auto pAVCodecWriter = dynamic_cast<CAvcodecWriter *>(writer);
        if (pAVCodecWriter != nullptr) {
            //エンコーダなどにキャッシュされたパケットを書き出す
            pAVCodecWriter->WriteNextPacket(nullptr);
        }
    }
#endif //ENABLE_AVCODEC_QSV_READER

    if (m_pmfxVPP)
    {
        // loop to get buffered frames from vpp
        while (MFX_ERR_NONE <= sts || MFX_ERR_MORE_DATA == sts || MFX_ERR_MORE_SURFACE == sts)
            // MFX_ERR_MORE_SURFACE can be returned only by RunFrameVPPAsync
            // MFX_ERR_MORE_DATA is accepted only from EncodeFrameAsync
        {
            pNextFrame = NULL;
            // find free surface for encoder input (vpp output)
            nEncSurfIdx = GetFreeSurface(m_pEncSurfaces, m_EncResponse.NumFrameActual);
            MSDK_CHECK_ERROR(nEncSurfIdx, MSDK_INVALID_SURF_IDX, MFX_ERR_MEMORY_ALLOC);

            pSurfEncIn = &m_pEncSurfaces[nEncSurfIdx];

            // get a free task (bit stream and sync point for encoder)
            sts = GetFreeTask(&pCurrentTask);
            MSDK_BREAK_ON_ERROR(sts);
            
            get_all_free_surface(pSurfEncIn);

            //for (int i_filter = 0; i_filter < (int)m_VppPrePlugins.size(); i_filter++) {
            //    bVppAllFiltersFlushed &= m_VppPrePlugins[i_filter]->m_bPluginFlushed;
            //    if (!m_VppPrePlugins[i_filter]->m_bPluginFlushed) {
            //        mfxFrameSurface1 *pSurfFilterOut = pSurfVppPreFilter[i_filter + 1];
            //        sts = filter_one_frame(m_VppPrePlugins[i_filter], &pNextFrame, &pSurfFilterOut);
            //        if (sts == MFX_ERR_MORE_DATA) {
            //            m_VppPrePlugins[i_filter]->m_bPluginFlushed = true;
            //            sts = MFX_ERR_NONE;
            //        }
            //        MSDK_BREAK_ON_ERROR(sts);
            //        pNextFrame = pSurfFilterOut;
            //    }
            //}
            //MSDK_BREAK_ON_ERROR(sts);

            sts = vpp_one_frame(pNextFrame, (m_VppPostPlugins.size()) ? pSurfVppPostFilter[0] : pSurfEncIn);
            if (bVppRequireMoreFrame) {
                break; // MFX_ERR_MORE_DATA is the correct status to exit vpp buffering loop
            }
            MSDK_BREAK_ON_ERROR(sts);

            for (int i_filter = 0; i_filter < (int)m_VppPostPlugins.size(); i_filter++) {
                mfxFrameSurface1 *pSurfFilterOut = pSurfVppPostFilter[i_filter + 1];
                sts = filter_one_frame(m_VppPostPlugins[i_filter], &pNextFrame, &pSurfFilterOut);
                MSDK_BREAK_ON_ERROR(sts);
                pNextFrame = pSurfFilterOut;
            }
            MSDK_BREAK_ON_ERROR(sts);

            sts = encode_one_frame(pNextFrame);
        }

        // MFX_ERR_MORE_DATA is the correct status to exit buffering loop with
        // indicates that there are no more buffered frames
        MSDK_IGNORE_MFX_STS(sts, MFX_ERR_MORE_DATA);
        // exit in case of other errors
        MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Error in getting buffered frames from vpp."));
        PrintMes(QSV_LOG_DEBUG, _T("Encode Thread: finished getting buffered frames from vpp.\n"));
    }

    // loop to get buffered frames from encoder
    while (MFX_ERR_NONE <= sts && m_pmfxENC)
    {
        // get a free task (bit stream and sync point for encoder)
        sts = GetFreeTask(&pCurrentTask);
        MSDK_BREAK_ON_ERROR(sts);

        sts = encode_one_frame(NULL);
    }
    PrintMes(QSV_LOG_DEBUG, _T("Encode Thread: finished getting buffered frames from encoder.\n"));

    // MFX_ERR_MORE_DATA is the correct status to exit buffering loop with
    // indicates that there are no more buffered frames
    MSDK_IGNORE_MFX_STS(sts, MFX_ERR_MORE_DATA);
    // exit in case of other errors
    MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Error in getting buffered frames from encoder."));

    // synchronize all tasks that are left in task pool
    while (MFX_ERR_NONE == sts)
    {
        sts = m_TaskPool.SynchronizeFirstTask();
    }

    // MFX_ERR_NOT_FOUND is the correct status to exit the loop with
    // EncodeFrameAsync and SyncOperation don't return this status
    MSDK_IGNORE_MFX_STS(sts, MFX_ERR_NOT_FOUND);
    // report any errors that occurred in asynchronous part
    MSDK_CHECK_RESULT_MES(sts, MFX_ERR_NONE, sts, _T("Error in encoding pipeline, synchronizing pipeline."));
    
    PrintMes(QSV_LOG_DEBUG, _T("Encode Thread: finished.\n"));
    return sts;
}

void CEncodingPipeline::PrintMes(int log_level, const TCHAR *format, ...) {
    if (m_pQSVLog.get() == nullptr || log_level < m_pQSVLog->getLogLevel()) {
        return;
    }

    va_list args;
    va_start(args, format);

    int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    vector<TCHAR> buffer(len, 0);
    _vstprintf_s(buffer.data(), len, format, args);
    va_end(args);

    (*m_pQSVLog)(log_level, buffer.data());
}

const char *CQSVLog::HTML_FOOTER = "</body>\n</html>\n";

void CQSVLog::init(const TCHAR *pLogFile, int log_level) {
    m_pStrLog = pLogFile;
    m_nLogLevel = log_level;
    if (pLogFile != nullptr) {
        FILE *fp = NULL;
        if (_tfopen_s(&fp, pLogFile, _T("a+")) || fp == NULL) {
            fprintf(stderr, "failed to open log file, log writing disabled.\n");
            pLogFile = nullptr;
        } else {
            if (check_ext(pLogFile, { ".html", ".htm" })) {
                _fseeki64(fp, 0, SEEK_SET);
                char buffer[1024] = { 0 };
                size_t file_read = fread(buffer, 1, sizeof(buffer)-1, fp);
                if (file_read == 0) {
                    m_bHtml = true;
                    writeHtmlHeader();
                } else {
                    std::transform(buffer, buffer + file_read, buffer, [](char in) -> char {return (char)tolower(in); });
                    if (strstr(buffer, "doctype") && strstr(buffer, "html")) {
                        m_bHtml = true;
                    }
                }
            }
            fclose(fp);
        }
    }
};

void CQSVLog::writeHtmlHeader() {
    FILE *fp = _tfopen(m_pStrLog, _T("wb"));
    if (fp) {
        std::wstring header =
            L"<!DOCTYPE html>\n"
            L"<html lang = \"ja\">\n"
            L"<head>\n"
            L"<meta charset = \"UTF-8\">\n"
            L"<title>QSVEncC Log</title>\n"
            L"<style type=text/css>\n"
            L"   body   { \n"
            L"       background-color: #303030;\n"
            L"       line-height:1.0; font-family: \"MeiryoKe_Gothic\",\"遊ゴシック\",\"ＭＳ ゴシック\",sans-serif;\n"
            L"       margin: 10px;\n"
            L"       padding: 0px;\n"
            L"   }\n"
            L"   div {\n"
            L"       white-space: pre;\n"
            L"   }\n"
            L"   .error { color: #FA5858 }\n"
            L"   .warn  { color: #F7D358 }\n"
            L"   .more  { color: #CEF6F5 }\n"
            L"   .info  { color: #CEF6F5 }\n"
            L"   .debug { color: #ACFA58 }\n"
            L"   .trace { color: #ACFA58 }\n"
            L"</style>\n"
            L"</head>\n"
            L"<body>\n";
        fprintf(fp, wstring_to_string(header, CP_UTF8).c_str());
        fprintf(fp, HTML_FOOTER);
        fclose(fp);
    }
}
void CQSVLog::writeFileHeader(const TCHAR *pDstFilename) {
    tstring fileHeader;
    int dstFilenameLen = (int)_tcslen(pDstFilename);
    static const TCHAR *const SEP5 = _T("-----");
    int sep_count = max(16, dstFilenameLen / 5 + 1);
    if (m_bHtml) {
        fileHeader += _T("<hr>");
    } else {
        for (int i = 0; i < sep_count; i++)
            fileHeader += SEP5;
    }
    fileHeader += _T("\n") + tstring(pDstFilename) + _T("\n");
    if (m_bHtml) {
        fileHeader += _T("<hr>");
    } else {
        for (int i = 0; i < sep_count; i++)
            fileHeader += SEP5;
    }
    fileHeader += _T("\n");
    (*this)(QSV_LOG_INFO, fileHeader.c_str());

    if (m_nLogLevel == QSV_LOG_DEBUG) {
        TCHAR cpuInfo[256] = { 0 };
        TCHAR gpu_info[1024] = { 0 };
        getCPUInfo(cpuInfo, _countof(cpuInfo));
        getGPUInfo("Intel", gpu_info, _countof(gpu_info));
        (*this)(QSV_LOG_DEBUG, _T("QSVEnc    %s (%s)\n"), VER_STR_FILEVERSION_TCHAR, BUILD_ARCH_STR);
        (*this)(QSV_LOG_DEBUG, _T("OS        %s (%s)\n"), getOSVersion(), is_64bit_os() ? _T("x64") : _T("x86"));
        (*this)(QSV_LOG_DEBUG, _T("CPU Info  %s\n"), cpuInfo);
        (*this)(QSV_LOG_DEBUG, _T("GPU Info  %s\n"), gpu_info);
    }
}
void CQSVLog::writeFileFooter() {
    (*this)(QSV_LOG_INFO, _T("\n\n"));
}

void CQSVLog::operator()(int log_level, const TCHAR *format, ...) {
    if (log_level < m_nLogLevel) {
        return;
    }

    auto convert_to_html = [log_level](std::string str) {
        //str = str_replace(str, "<", "&lt;");
        //str = str_replace(str, ">", "&gt;");
        //str = str_replace(str, "&", "&amp;");
        //str = str_replace(str, "\"", "&quot;");

        auto strLines = split(str, "\n");

        std::string strHtml;
        for (mfxU32 i = 0; i < strLines.size() - 1; i++) {
            strHtml += strsprintf("<div class=\"%s\">", tchar_to_string(list_log_level[log_level - QSV_LOG_TRACE].desc).c_str());
            strHtml += strLines[i];
            strHtml += "</div>\n";
        }
        return strHtml;
    };

    va_list args;
    va_start(args, format);

    int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    tstring buffer(len, 0);
    if (buffer.data() != nullptr) {

        _vstprintf_s(&buffer[0], len, format, args); // C4996
        
        HANDLE hStdErr = GetStdHandle(STD_ERROR_HANDLE);
        std::string buffer_char;
#ifdef UNICODE
        char *buffer_ptr = NULL;
        DWORD mode = 0;
        bool stderr_write_to_console = 0 != GetConsoleMode(hStdErr, &mode); //stderrの出力先がコンソールかどうか
        if (m_pStrLog || !stderr_write_to_console) {
            buffer_char = tchar_to_string(buffer, (m_bHtml) ? CP_UTF8 : CP_THREAD_ACP);
            if (m_bHtml) {
                buffer_char = convert_to_html(buffer_char);
            }
            buffer_ptr = &buffer_char[0];
        }
#else
        char *buffer_ptr = &buffer[0];
        if (m_bHtml) {
            buffer_char = wstring_to_string(char_to_wstring(buffer_ptr), CP_UTF8);
            if (m_bHtml) {
                buffer_char = convert_to_html(buffer_char);
            }
            buffer_ptr = &buffer_char[0];
        }
#endif
        EnterCriticalSection(&cs);
        if (m_pStrLog) {
            FILE *fp_log = NULL;
            //logはANSI(まあようはShift-JIS)で保存する
            if (0 == _tfopen_s(&fp_log, m_pStrLog, (m_bHtml) ? _T("rb+") : _T("a")) && fp_log) {
                if (m_bHtml) {
                    _fseeki64(fp_log, 0, SEEK_END);
                    __int64 pos = _ftelli64(fp_log);
                    _fseeki64(fp_log, 0, SEEK_SET);
                    _fseeki64(fp_log, pos -1 * strlen(HTML_FOOTER), SEEK_CUR);
                }
                fwrite(buffer_ptr, 1, strlen(buffer_ptr), fp_log);
                if (m_bHtml) {
                    fwrite(HTML_FOOTER, 1, strlen(HTML_FOOTER), fp_log);
                }
                fclose(fp_log);
            }
        }
#ifdef UNICODE
        if (!stderr_write_to_console) //出力先がリダイレクトされるならANSIで
            fprintf(stderr, buffer_ptr);
        if (stderr_write_to_console) //出力先がコンソールならWCHARで
#endif
            qsv_print_stderr(log_level, buffer.data(), hStdErr);
        LeaveCriticalSection(&cs);
    }
    va_end(args);
}

void CEncodingPipeline::GetEncodeLibInfo(mfxVersion *ver, bool *hardware) {
    if (NULL != ver && NULL != hardware) {
        mfxIMPL impl;
        m_mfxSession.QueryIMPL(&impl);
        *hardware = !!Check_HWUsed(impl);
        *ver = m_mfxVer;
    }

}

MemType CEncodingPipeline::GetMemType() {
    return m_memType;
}

mfxStatus CEncodingPipeline::GetEncodeStatusData(sEncodeStatusData *data) {
    if (NULL == data)
        return MFX_ERR_NULL_PTR;

    if (NULL == m_pEncSatusInfo)
        return MFX_ERR_NOT_INITIALIZED;

    m_pEncSatusInfo->GetEncodeData(data);
    return MFX_ERR_NONE;
}

const msdk_char *CEncodingPipeline::GetInputMessage() {
    return m_pFileReader->GetInputMessage();
}

mfxStatus CEncodingPipeline::CheckCurrentVideoParam(TCHAR *str, mfxU32 bufSize)
{
    mfxIMPL impl;
    m_mfxSession.QueryIMPL(&impl);

    mfxFrameInfo SrcPicInfo = m_mfxVppParams.vpp.In;
    mfxFrameInfo DstPicInfo = m_mfxEncParams.mfx.FrameInfo;

    mfxU8 spsbuf[256] = { 0 };
    mfxU8 ppsbuf[256] = { 0 };
    mfxExtCodingOptionSPSPPS spspps;
    INIT_MFX_EXT_BUFFER(spspps, MFX_EXTBUFF_CODING_OPTION_SPSPPS);
    spspps.SPSBuffer = spsbuf;
    spspps.SPSBufSize = sizeof(spsbuf);
    spspps.PPSBuffer = ppsbuf;
    spspps.PPSBufSize = sizeof(ppsbuf);

    mfxExtCodingOption cop;
    mfxExtCodingOption2 cop2;
    mfxExtCodingOption3 cop3;
    mfxExtVP8CodingOption copVp8;
    INIT_MFX_EXT_BUFFER(cop, MFX_EXTBUFF_CODING_OPTION);
    INIT_MFX_EXT_BUFFER(cop2, MFX_EXTBUFF_CODING_OPTION2);
    INIT_MFX_EXT_BUFFER(cop3, MFX_EXTBUFF_CODING_OPTION3);
    INIT_MFX_EXT_BUFFER(copVp8, MFX_EXTBUFF_VP8_CODING_OPTION);

    std::vector<mfxExtBuffer *> buf;
    buf.push_back((mfxExtBuffer *)&cop);
    buf.push_back((mfxExtBuffer *)&spspps);
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_6)) {
        buf.push_back((mfxExtBuffer *)&cop2);
    }
    if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_11)) {
        buf.push_back((mfxExtBuffer *)&cop3);
    }
    if (m_mfxEncParams.mfx.CodecId == MFX_CODEC_VP8) {
        buf.push_back((mfxExtBuffer *)&copVp8);
    }

    mfxVideoParam videoPrm;
    MSDK_ZERO_MEMORY(videoPrm);
    videoPrm.NumExtParam = (mfxU16)buf.size();
    videoPrm.ExtParam = &buf[0];

    mfxStatus sts = MFX_ERR_NONE;
    if (m_pmfxENC) {
        sts = m_pmfxENC->GetVideoParam(&videoPrm);
        MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
    } else if (m_pmfxVPP) {
        mfxVideoParam videoPrmVpp;
        MSDK_ZERO_MEMORY(videoPrmVpp);
        sts = m_pmfxVPP->GetVideoParam(&videoPrmVpp);
        MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
        videoPrm.mfx.FrameInfo = videoPrmVpp.vpp.Out;
        DstPicInfo = videoPrmVpp.vpp.Out;
    } else if (m_pmfxDEC) {
        sts = m_pmfxDEC->GetVideoParam(&videoPrm);
        MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
        DstPicInfo = videoPrm.mfx.FrameInfo;
    }

    if (m_pFileWriter) {
        if (MFX_ERR_NONE != (sts = m_pFileWriter->SetVideoParam(&videoPrm, &cop2))) {
            PrintMes(QSV_LOG_ERROR, _T("%s\n"), m_pFileWriter->GetOutputMessage());
            return sts;
        }
        PrintMes(QSV_LOG_DEBUG, _T("CheckCurrentVideoParam: SetVideoParam to video file writer.\n"));
    }

    TCHAR cpuInfo[256];
    getCPUInfo(cpuInfo, _countof(cpuInfo));

    TCHAR gpu_info[1024] = { 0 };
    if (Check_HWUsed(impl)) {
        getGPUInfo("Intel", gpu_info, _countof(gpu_info));
    }
    TCHAR info[4096];
    mfxU32 info_len = 0;

#define PRINT_INFO(fmt, ...) { info_len += _stprintf_s(info + info_len, _countof(info) - info_len, fmt, __VA_ARGS__); }
#define PRINT_INT_AUTO(fmt, i) { if (i) { info_len += _stprintf_s(info + info_len, _countof(info) - info_len, fmt, i); } else { info_len += _stprintf_s(info + info_len, _countof(info) - info_len, (fmt[_tcslen(fmt)-1]=='\n') ? _T("Auto\n") : _T("Auto")); } }
    PRINT_INFO(    _T("QSVEnc %s (%s), based on Intel(R) Media SDK Encoding Sample %s\n"), VER_STR_FILEVERSION_TCHAR, BUILD_ARCH_STR, MSDK_SAMPLE_VERSION);
    PRINT_INFO(    _T("OS             %s (%s)\n"), getOSVersion(), is_64bit_os() ? _T("x64") : _T("x86"));
    PRINT_INFO(    _T("CPU Info       %s\n"), cpuInfo);
    if (Check_HWUsed(impl)) {
        PRINT_INFO(_T("GPU Info       %s\n"), gpu_info);
    }
    if (Check_HWUsed(impl)) {
        static const TCHAR * const NUM_APPENDIX[] = { _T("st"), _T("nd"), _T("rd"), _T("th")};
        mfxU32 iGPUID = MSDKAdapter::GetNumber(m_mfxSession);
        PRINT_INFO(    _T("Media SDK      QuickSyncVideo (hardware encoder)%s, %d%s GPU, API v%d.%d\n"),
            get_low_power_str(videoPrm.mfx.LowPower), iGPUID + 1, NUM_APPENDIX[clamp(iGPUID, 0, _countof(NUM_APPENDIX) - 1)], m_mfxVer.Major, m_mfxVer.Minor);
    } else {
        PRINT_INFO(    _T("Media SDK      software encoder, API v%d.%d\n"), m_mfxVer.Major, m_mfxVer.Minor);
    }
    PRINT_INFO(    _T("Async Depth    %d frames\n"), m_nAsyncDepth);
    PRINT_INFO(    _T("Buffer Memory  %s, %d input buffer, %d work buffer\n"), MemTypeToStr(m_memType), m_EncThread.m_nFrameBuffer, m_EncResponse.NumFrameActual + m_VppResponse.NumFrameActual + m_DecResponse.NumFrameActual);
    //PRINT_INFO(    _T("Input Frame Format   %s\n"), ColorFormatToStr(m_pFileReader->m_ColorFormat));
    //PRINT_INFO(    _T("Input Frame Type     %s\n"), list_interlaced[get_cx_index(list_interlaced, SrcPicInfo.PicStruct)].desc);
    tstring inputMes = m_pFileReader->GetInputMessage();
    for (const auto& reader : m_AudioReaders) {
        inputMes += _T("\n") + tstring(reader->GetInputMessage());
    }
    auto inputMesSplitted = split(inputMes, _T("\n"));
    for (mfxU32 i = 0; i < inputMesSplitted.size(); i++) {
        PRINT_INFO(_T("%s%s\n"), (i == 0) ? _T("Input Info     ") : _T("               "), inputMesSplitted[i].c_str());
    }

    sInputCrop inputCrop;
    m_pFileReader->GetInputCropInfo(&inputCrop);
    if (0 != (inputCrop.bottom | inputCrop.left | inputCrop.right | inputCrop.up))
        PRINT_INFO(_T("Crop           %d,%d,%d,%d (%dx%d -> %dx%d)\n"),
            inputCrop.left, inputCrop.up, inputCrop.right, inputCrop.bottom,
            SrcPicInfo.CropW + inputCrop.left + inputCrop.right,
            SrcPicInfo.CropH + inputCrop.up + inputCrop.bottom,
            SrcPicInfo.CropW, SrcPicInfo.CropH);

    if (VppExtMes.size()) {
        const TCHAR *m = _T("VPP Enabled    ");
        size_t len = VppExtMes.length() + 1;
        TCHAR *vpp_mes = (TCHAR*)malloc(len * sizeof(vpp_mes[0]));
        memcpy(vpp_mes, VppExtMes.c_str(), len * sizeof(vpp_mes[0]));
        for (TCHAR *p = vpp_mes, *q; (p = _tcstok_s(p, _T("\n"), &q)) != NULL; ) {
            PRINT_INFO(_T("%s%s\n"), m, p);
            m    = _T("               ");
            p = NULL;
        }
        free(vpp_mes);
        VppExtMes.clear();
    }
    if (m_pTrimParam != NULL && m_pTrimParam->list.size()
        && !(m_pTrimParam->list[0].start == 0 && m_pTrimParam->list[0].fin == TRIM_MAX)) {
        PRINT_INFO(_T("Trim           "));
        for (auto trim : m_pTrimParam->list) {
            if (trim.fin == TRIM_MAX) {
                PRINT_INFO(_T("%d-fin "), trim.start + m_pTrimParam->offset);
            } else {
                PRINT_INFO(_T("%d-%d "), trim.start + m_pTrimParam->offset, trim.fin + m_pTrimParam->offset);
            }
        }
        PRINT_INFO(_T("[offset: %d]\n"), m_pTrimParam->offset);
    }
    if (m_pmfxENC) {
        PRINT_INFO(_T("Output         %s  %s @ Level %s\n"), CodecIdToStr(videoPrm.mfx.CodecId).c_str(),
            get_profile_list(videoPrm.mfx.CodecId)[get_cx_index(get_profile_list(videoPrm.mfx.CodecId), videoPrm.mfx.CodecProfile)].desc,
            get_level_list(videoPrm.mfx.CodecId)[get_cx_index(get_level_list(videoPrm.mfx.CodecId), videoPrm.mfx.CodecLevel)].desc);
    }
    PRINT_INFO(_T("%s         %dx%d%s %d:%d %0.3ffps (%d/%dfps)%s%s\n"),
        (m_pmfxENC) ? _T("      ") : _T("Output"),
        DstPicInfo.CropW, DstPicInfo.CropH, (DstPicInfo.PicStruct & MFX_PICSTRUCT_PROGRESSIVE) ? _T("p") : _T("i"),
        videoPrm.mfx.FrameInfo.AspectRatioW, videoPrm.mfx.FrameInfo.AspectRatioH,
        DstPicInfo.FrameRateExtN / (double)DstPicInfo.FrameRateExtD, DstPicInfo.FrameRateExtN, DstPicInfo.FrameRateExtD,
        (DstPicInfo.PicStruct & MFX_PICSTRUCT_PROGRESSIVE) ? _T("") : _T(", "),
        (DstPicInfo.PicStruct & MFX_PICSTRUCT_PROGRESSIVE) ? _T("") : list_interlaced[get_cx_index(list_interlaced, DstPicInfo.PicStruct)].desc);
    if (m_pFileWriter) {
        inputMesSplitted = split(m_pFileWriter->GetOutputMessage(), _T("\n"));
        for (auto mes : inputMesSplitted) {
            if (mes.length()) {
                PRINT_INFO(_T("%s%s\n"), _T("               "), mes.c_str());
            }
        }
    }
    for (auto pWriter : m_pFileWriterListAudio) {
        if (pWriter && pWriter != m_pFileWriter) {
            inputMesSplitted = split(pWriter->GetOutputMessage(), _T("\n"));
            for (auto mes : inputMesSplitted) {
                if (mes.length()) {
                    PRINT_INFO(_T("%s%s\n"), _T("               "), mes.c_str());
                }
            }
        }
    }
    
    if (m_pmfxENC) {
        PRINT_INFO(_T("Target usage   %s\n"), TargetUsageToStr(videoPrm.mfx.TargetUsage));
        PRINT_INFO(_T("Encode Mode    %s\n"), EncmodeToStr((videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_CQP && (m_nExPrm & MFX_PRM_EX_VQP)) ? MFX_RATECONTROL_VQP : videoPrm.mfx.RateControlMethod));
        if (m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_CQP) {
            if (m_nExPrm & MFX_PRM_EX_VQP) {
                //PRINT_INFO(_T("VQP params             I:%d  P:%d+  B:%d+  strength:%d  sensitivity:%d\n"), videoPrm.mfx.QPI, videoPrm.mfx.QPP, videoPrm.mfx.QPB, m_SceneChange.getVQPStrength(), m_SceneChange.getVQPSensitivity());
                PRINT_INFO(_T("VQP params     I:%d  P:%d+  B:%d+\n"), videoPrm.mfx.QPI, videoPrm.mfx.QPP, videoPrm.mfx.QPB);
            } else {
                PRINT_INFO(_T("CQP Value      I:%d  P:%d  B:%d\n"), videoPrm.mfx.QPI, videoPrm.mfx.QPP, videoPrm.mfx.QPB);
            }
        } else if (rc_is_type_lookahead(m_mfxEncParams.mfx.RateControlMethod)) {
            PRINT_INFO(_T("Lookahead      depth %d frames"), cop2.LookAheadDepth);
            if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)) {
                PRINT_INFO(_T(", quality %s"), list_lookahead_ds[get_cx_index(list_lookahead_ds, cop2.LookAheadDS)].desc);
            }
            PRINT_INFO(_T("\n"));
            if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_11)) {
                if (cop3.WinBRCSize) {
                    PRINT_INFO(_T("Windowed RC    %d frames, Max %d kbps\n"), cop3.WinBRCSize, cop3.WinBRCMaxAvgKbps);
                } else {
                    PRINT_INFO(_T("Windowed RC    off\n"));
                }
            }
            if (MFX_RATECONTROL_LA_ICQ == m_mfxEncParams.mfx.RateControlMethod) {
                PRINT_INFO(_T("ICQ Quality    %d\n"), videoPrm.mfx.ICQQuality);
            }
        } else if (MFX_RATECONTROL_ICQ == m_mfxEncParams.mfx.RateControlMethod) {
            PRINT_INFO(_T("ICQ Quality    %d\n"), videoPrm.mfx.ICQQuality);
        } else {
            PRINT_INFO(_T("Bitrate        %d kbps\n"), (mfxU32)videoPrm.mfx.TargetKbps * max(m_mfxEncParams.mfx.BRCParamMultiplier, 1));
            if (m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_AVBR) {
                //PRINT_INFO(_T("AVBR Accuracy range\t%.01lf%%"), m_mfxEncParams.mfx.Accuracy / 10.0);
                PRINT_INFO(_T("AVBR Converge  %d frames unit\n"), videoPrm.mfx.Convergence * 100);
            } else {
                PRINT_INFO(_T("Max Bitrate    "));
                PRINT_INT_AUTO(_T("%d kbps\n"), (mfxU32)videoPrm.mfx.MaxKbps * max(m_mfxEncParams.mfx.BRCParamMultiplier, 1));
                if (m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_QVBR) {
                    PRINT_INFO(_T("QVBR Quality   %d\n"), cop3.QVBRQuality);
                }
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_9)) {
            auto qp_limit_str = [](mfxU8 limitI, mfxU8 limitP, mfxU8 limitB) {
                mfxU8 limit[3] = { limitI, limitP, limitB };
                if (0 == (limit[0] | limit[1] | limit[2]))
                    return std::basic_string<msdk_char>(_T("none"));

                tstring buf;
                for (int i = 0; i < 3; i++) {
                    buf += ((i) ? _T(":") : _T(""));
                    if (limit[i]) {
                        buf += std::to_tstring(limit[i]);
                    } else {
                        buf += _T("-");
                    }
                }
                return buf;
            };
            PRINT_INFO(_T("QP Limit       min: %s, max: %s\n"),
                qp_limit_str(cop2.MinQPI, cop2.MinQPP, cop2.MinQPB).c_str(),
                qp_limit_str(cop2.MaxQPI, cop2.MaxQPP, cop2.MaxQPB).c_str());
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_7)) {
            PRINT_INFO(_T("Trellis        %s\n"), list_avc_trellis[get_cx_index(list_avc_trellis_for_options, cop2.Trellis)].desc);
        }

        if (videoPrm.mfx.CodecId == MFX_CODEC_AVC && !Check_HWUsed(impl)) {
            PRINT_INFO(_T("CABAC          %s\n"), (cop.CAVLC == MFX_CODINGOPTION_ON) ? _T("off") : _T("on"));
            PRINT_INFO(_T("RDO            %s\n"), (cop.RateDistortionOpt == MFX_CODINGOPTION_ON) ? _T("on") : _T("off"));
            if ((cop.MVSearchWindow.x | cop.MVSearchWindow.y) == 0) {
                PRINT_INFO(_T("mv search      precision: %s\n"), list_mv_presicion[get_cx_index(list_mv_presicion, cop.MVPrecision)].desc);
            } else {
                PRINT_INFO(_T("mv search      precision: %s, window size:%dx%d\n"), list_mv_presicion[get_cx_index(list_mv_presicion, cop.MVPrecision)].desc, cop.MVSearchWindow.x, cop.MVSearchWindow.y);
            }
            PRINT_INFO(_T("min pred size  inter: %s   intra: %s\n"), list_pred_block_size[get_cx_index(list_pred_block_size, cop.InterPredBlockSize)].desc, list_pred_block_size[get_cx_index(list_pred_block_size, cop.IntraPredBlockSize)].desc);
        }
        PRINT_INFO(_T("Ref frames     "));
        PRINT_INT_AUTO(_T("%d frames\n"), videoPrm.mfx.NumRefFrame);

        PRINT_INFO(_T("Bframes        "));
        switch (videoPrm.mfx.GopRefDist) {
        case 0:  PRINT_INFO(_T("Auto\n")); break;
        case 1:  PRINT_INFO(_T("none\n")); break;
        default: PRINT_INFO(_T("%d frame%s%s%s\n"),
            videoPrm.mfx.GopRefDist - 1, (videoPrm.mfx.GopRefDist > 2) ? _T("s") : _T(""),
            check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8) ? _T(", B-pyramid: ") : _T(""),
            (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8) ? ((MFX_B_REF_PYRAMID == cop2.BRefType) ? _T("on") : _T("off")) : _T(""))); break;
        }

        //PRINT_INFO(    _T("Idr Interval    %d\n"), videoPrm.mfx.IdrInterval);
        PRINT_INFO(_T("Max GOP Length "));
        PRINT_INT_AUTO(_T("%d frames\n"), min(videoPrm.mfx.GopPicSize, m_SceneChange.getMaxGOPLen()));
        PRINT_INFO(_T("Scene Change   %s\n"), m_SceneChange.isInitialized() ? _T("on") : _T("off"));
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_8)) {
            //PRINT_INFO(    _T("GOP Structure           "));
            //bool adaptiveIOn = (MFX_CODINGOPTION_ON == cop2.AdaptiveI);
            //bool adaptiveBOn = (MFX_CODINGOPTION_ON == cop2.AdaptiveB);
            //if (!adaptiveIOn && !adaptiveBOn) {
            //    PRINT_INFO(_T("fixed\n"))
            //} else {
            //    PRINT_INFO(_T("Adaptive %s%s%s insert\n"),
            //        (adaptiveIOn) ? _T("I") : _T(""),
            //        (adaptiveIOn && adaptiveBOn) ? _T(",") : _T(""),
            //        (adaptiveBOn) ? _T("B") : _T(""));
            //}
        }
        if (videoPrm.mfx.NumSlice >= 2) {
            PRINT_INFO(_T("Slices         %d\n"), videoPrm.mfx.NumSlice);
        }

        if (videoPrm.mfx.CodecId == MFX_CODEC_VP8) {
            PRINT_INFO(_T("Sharpness      %d\n"), copVp8.SharpnessLevel);
        }

        //last line
        tstring extFeatures;
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_6)) {
            if (cop2.MBBRC  == MFX_CODINGOPTION_ON) {
                extFeatures += _T("PerMBRC ");
            }
            if (cop2.ExtBRC == MFX_CODINGOPTION_ON) {
                extFeatures += _T("ExtBRC ");
            }
        }
        if (check_lib_version(m_mfxVer, MFX_LIB_VERSION_1_9)) {
            if (cop2.DisableDeblockingIdc) {
                extFeatures += _T("No-Deblock ");
            }
            if (cop2.IntRefType) {
                extFeatures += _T("Intra-Refresh ");
            }
        }
        //if (cop.AUDelimiter == MFX_CODINGOPTION_ON) {
        //    extFeatures += _T("aud ");
        //}
        //if (cop.PicTimingSEI == MFX_CODINGOPTION_ON) {
        //    extFeatures += _T("pic_struct ");
        //}
        //if (cop.SingleSeiNalUnit == MFX_CODINGOPTION_ON) {
        //    extFeatures += _T("SingleSEI ");
        //}
        if (extFeatures.length() > 0) {
            PRINT_INFO(_T("Ext. Features  %s\n"), extFeatures.c_str());
        }
    }

    PrintMes(QSV_LOG_INFO, info);
    if (str && bufSize > 0) {
        msdk_strcopy(str, bufSize, info);
    }

    return MFX_ERR_NONE;
#undef PRINT_INFO
#undef PRINT_INT_AUTO
}

