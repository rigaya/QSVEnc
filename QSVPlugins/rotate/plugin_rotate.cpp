//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <algorithm>
#include <stdio.h>
#include "sample_utils.h"
#include "plugin_rotate.h"

#pragma warning(disable : 4100)

#define SWAP_BYTES(a, b) {mfxU8 tmp; tmp = a; a = b; b = tmp;}

Rotate::Rotate() {
    memset(&m_Param, 0, sizeof(m_Param));
    m_pluginName = _T("rotate");
}

Rotate::~Rotate() {
    PluginClose();
    Close();
}

mfxStatus Rotate::Submit(const mfxHDL *in, mfxU32 in_num, const mfxHDL *out, mfxU32 out_num, mfxThreadTask *task) {
    if (in == nullptr || out == nullptr || *in == nullptr || *out == nullptr || task == nullptr) {
        return MFX_ERR_NULL_PTR;
    }
    if (in_num != 1 || out_num != 1) {
        return MFX_ERR_UNSUPPORTED;
    }
    if (!m_bInited) return MFX_ERR_NOT_INITIALIZED;

    mfxFrameSurface1 *surface_in = (mfxFrameSurface1 *)in[0];
    mfxFrameSurface1 *surface_out = (mfxFrameSurface1 *)out[0];
    mfxFrameSurface1 *real_surface_in = surface_in;
    mfxFrameSurface1 *real_surface_out = surface_out;

    mfxStatus sts = MFX_ERR_NONE;

    if (m_bIsInOpaque) {
        sts = m_mfxCore.GetRealSurface(surface_in, &real_surface_in);
        if (sts < MFX_ERR_NONE) return sts;
    }

    if (m_bIsOutOpaque) {
        sts = m_mfxCore.GetRealSurface(surface_out, &real_surface_out);
        if (sts < MFX_ERR_NONE) return sts;
    }

    sts = CheckInOutFrameInfo(&real_surface_in->Info, &real_surface_out->Info);
    if (sts < MFX_ERR_NONE) return sts;

    mfxU32 ind = FindFreeTaskIdx();

    if (ind >= m_sTasks.size()) {
        return MFX_WRN_DEVICE_BUSY;
    }

    m_mfxCore.IncreaseReference(&(real_surface_in->Data));
    m_mfxCore.IncreaseReference(&(real_surface_out->Data));

    m_sTasks[ind].In = real_surface_in;
    m_sTasks[ind].Out = real_surface_out;
    m_sTasks[ind].bBusy = true;

    switch (m_Param.Angle) {
    case 180:
        m_sTasks[ind].pProcessor.reset(new Rotator180);
        break;
    default:
        return MFX_ERR_UNSUPPORTED;
    }

    m_sTasks[ind].pProcessor->SetAllocator(&m_mfxCore.FrameAllocator());
    m_sTasks[ind].pProcessor->Init(real_surface_in, real_surface_out, nullptr);

    *task = (mfxThreadTask)&m_sTasks[ind];

    return MFX_ERR_NONE;
}

mfxStatus Rotate::Init(mfxVideoParam *mfxParam) {
    mfxStatus sts = MFX_ERR_NONE;
    m_VideoParam = *mfxParam;

    m_bIsInOpaque = (m_VideoParam.IOPattern & MFX_IOPATTERN_IN_OPAQUE_MEMORY) ? true : false;
    m_bIsOutOpaque = (m_VideoParam.IOPattern & MFX_IOPATTERN_OUT_OPAQUE_MEMORY) ? true : false;
    mfxExtOpaqueSurfaceAlloc* pluginOpaqueAlloc = NULL;

    if (m_bIsInOpaque || m_bIsOutOpaque) {
        pluginOpaqueAlloc = (mfxExtOpaqueSurfaceAlloc*)GetExtBuffer(m_VideoParam.ExtParam,
            m_VideoParam.NumExtParam, MFX_EXTBUFF_OPAQUE_SURFACE_ALLOCATION);
        if (sts != MFX_ERR_NONE) {
            m_message += _T("failed GetExtBuffer for OpaqueAlloc.\n");
            return sts;
        }
    }

    if ((m_bIsInOpaque && !pluginOpaqueAlloc->In.Surfaces) || (m_bIsOutOpaque && !pluginOpaqueAlloc->Out.Surfaces))
        return MFX_ERR_INVALID_VIDEO_PARAM;

    if (m_bIsInOpaque) {
        sts = m_mfxCore.MapOpaqueSurface(pluginOpaqueAlloc->In.NumSurface,
            pluginOpaqueAlloc->In.Type, pluginOpaqueAlloc->In.Surfaces);
        if (sts != MFX_ERR_NONE) {
            m_message += _T("failed MapOpaqueSurface[In].\n");
            return sts;
        }
    }

    if (m_bIsOutOpaque) {
        sts = m_mfxCore.MapOpaqueSurface(pluginOpaqueAlloc->Out.NumSurface,
            pluginOpaqueAlloc->Out.Type, pluginOpaqueAlloc->Out.Surfaces);
        if (sts != MFX_ERR_NONE) {
            m_message += _T("failed MapOpaqueSurface[Out].\n");
            return sts;
        }
    }

    m_sTasks.resize((std::max)(1, (int)m_VideoParam.AsyncDepth));
    m_sChunks.resize(m_PluginParam.MaxThreadNum);

    mfxU32 num_lines_in_chunk = mfxParam->vpp.In.CropH / (mfxU32)m_sChunks.size();
    mfxU32 remainder_lines = mfxParam->vpp.In.CropH % (mfxU32)m_sChunks.size();
    for (mfxU32 i = 0; i < m_sChunks.size(); i++) {
        m_sChunks[i].StartLine = (i == 0) ? 0 : m_sChunks[i-1].EndLine + 1;
        m_sChunks[i].EndLine = (i < remainder_lines) ? (i + 1) * num_lines_in_chunk : (i + 1) * num_lines_in_chunk - 1;
    }

    m_bInited = true;

    return MFX_ERR_NONE;
}

mfxStatus Rotate::SetAuxParams(void* auxParam, int auxParamSize) {
    RotateParam *pRotatePar = (RotateParam *)auxParam;

    mfxStatus sts = CheckParam(&m_VideoParam);
    if (sts < MFX_ERR_NONE) return sts;
    m_Param = *pRotatePar;

    m_message = _T("vpp-rotate (half-turn)\n");
    return MFX_ERR_NONE;
}

mfxStatus Rotate::Close() {

    if (!m_bInited)
        return MFX_ERR_NONE;

    memset(&m_Param, 0, sizeof(RotateParam));

    mfxStatus sts = MFX_ERR_NONE;

    mfxExtOpaqueSurfaceAlloc* pluginOpaqueAlloc = NULL;

    if (m_bIsInOpaque || m_bIsOutOpaque) {
        pluginOpaqueAlloc = (mfxExtOpaqueSurfaceAlloc*)
            GetExtBuffer(m_VideoParam.ExtParam, m_VideoParam.NumExtParam, MFX_EXTBUFF_OPAQUE_SURFACE_ALLOCATION);
    }

    if ((m_bIsInOpaque && !pluginOpaqueAlloc->In.Surfaces) || (m_bIsOutOpaque && !pluginOpaqueAlloc->Out.Surfaces))
        return MFX_ERR_INVALID_VIDEO_PARAM;

    if (m_bIsInOpaque) {
        sts = m_mfxCore.UnmapOpaqueSurface(pluginOpaqueAlloc->In.NumSurface,
            pluginOpaqueAlloc->In.Type, pluginOpaqueAlloc->In.Surfaces);
        if (sts < MFX_ERR_NONE) return sts;
    }

    if (m_bIsOutOpaque) {
        sts = m_mfxCore.UnmapOpaqueSurface(pluginOpaqueAlloc->Out.NumSurface,
            pluginOpaqueAlloc->Out.Type, pluginOpaqueAlloc->Out.Surfaces);
        if (sts < MFX_ERR_NONE) return sts;
    }

    m_message.clear();
    m_bInited = false;

    return MFX_ERR_NONE;
}

mfxStatus ProcessorRotate::Init(mfxFrameSurface1 *frame_in, mfxFrameSurface1 *frame_out, const void *data) {
    m_pIn = frame_in;
    m_pOut = frame_out;

    return MFX_ERR_NONE;
}


Rotator180::Rotator180() : ProcessorRotate() {
}

Rotator180::~Rotator180() {
}

mfxStatus Rotator180::Process(DataChunk *chunk, mfxU8 *pBuffer) {
    mfxStatus sts = MFX_ERR_NONE;
    if (MFX_ERR_NONE != (sts = LockFrame(m_pIn)))return sts;
    if (MFX_ERR_NONE != (sts = LockFrame(m_pOut))) {
        UnlockFrame(m_pIn);
        return sts;
    }

    mfxU32 i, j, in_pitch, out_pitch, h, w;

    in_pitch = m_pIn->Data.Pitch;
    out_pitch = m_pOut->Data.Pitch;
    h = m_pIn->Info.CropH;
    w = m_pIn->Info.CropW;

    m_YIn.assign(m_pIn->Data.Y, m_pIn->Data.Y + h * in_pitch);
    m_UVIn.assign(m_pIn->Data.UV, m_pIn->Data.UV + h * in_pitch / 2);

    m_YOut.resize(m_pOut->Info.Height * out_pitch);
    m_UVOut.resize(m_pOut->Info.Height * out_pitch / 2);

    //sts = UnlockFrame(m_pIn);
    //MSDK_CHECK_RESULT(MFX_ERR_NONE, sts, MFX_ERR_NONE);
    //sts = UnlockFrame(m_pOut);
    //MSDK_CHECK_RESULT(MFX_ERR_NONE, sts, MFX_ERR_NONE);

    mfxU8 *in_luma = &m_YIn.front() + m_pIn->Info.CropY * in_pitch + m_pIn->Info.CropX;
    mfxU8 *out_luma = &m_YOut.front() + m_pOut->Info.CropY * out_pitch + m_pOut->Info.CropX;

    mfxU8 *in_chroma = &m_UVIn.front() + m_pIn->Info.CropY / 2 * in_pitch + m_pIn->Info.CropX;
    mfxU8 *out_chroma = &m_UVOut.front() + m_pOut->Info.CropY / 2 * out_pitch + m_pOut->Info.CropX;

    mfxU8 *cur_line = 0;

    switch (m_pIn->Info.FourCC) {
    case MFX_FOURCC_NV12:
        for (i = chunk->StartLine; i <= chunk->EndLine; i++) {
            cur_line = out_luma + (h-1-i) * out_pitch;

            memcpy(cur_line, in_luma + i * in_pitch, w);

            for (j = 0; j < w / 2; j++) {
                SWAP_BYTES(cur_line[j], cur_line[w-1-j]);
            }

            cur_line = out_chroma + (h/2-1-i/2) * out_pitch;

            memcpy(cur_line, in_chroma + i/2 * in_pitch, w);

            for (j = 0; j < w/2 - 1; j = j + 2) {
                SWAP_BYTES(cur_line[j], cur_line[w-1-j-1]); // 0 -> -1
                SWAP_BYTES(cur_line[j+1], cur_line[w-1-j]); // 1 -> -0
            }
        }
        break;
    default:
        return MFX_ERR_UNSUPPORTED;
    }

    if (sts < MFX_ERR_NONE) return sts;
    memcpy(m_pOut->Data.Y + chunk->StartLine * out_pitch, &m_YOut.front(), m_YOut.size());
    memcpy(m_pOut->Data.UV + chunk->StartLine * out_pitch, &m_UVOut.front(), m_UVOut.size());
    sts = UnlockFrame(m_pIn);
    sts = UnlockFrame(m_pOut);

    return sts;
}
