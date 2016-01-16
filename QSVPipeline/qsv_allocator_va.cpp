//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include "qsv_allocator_va.h"

#if defined(LIBVA_SUPPORT)

#include <map>
#include <algorithm>
#include <stdio.h>
#include <assert.h>
#include "qsv_util.h"
#include "qsv_hw_va.h"

enum {
    MFX_FOURCC_VP8_NV12    = MFX_MAKEFOURCC('V','P','8','N'),
    MFX_FOURCC_VP8_MBDATA  = MFX_MAKEFOURCC('V','P','8','M'),
    MFX_FOURCC_VP8_SEGMAP  = MFX_MAKEFOURCC('V','P','8','S'),
};

static const std::map<mfxU32, uint32_t> fourccToVAFormat = {
    { MFX_FOURCC_NV12,    VA_FOURCC_NV12 },
    { MFX_FOURCC_YV12,    VA_FOURCC_YV12},
    { MFX_FOURCC_YUY2,    VA_FOURCC_YUY2},
    { MFX_FOURCC_RGB4,    VA_FOURCC_ARGB},
    { MFX_FOURCC_P8,      VA_FOURCC_P208},
};
static const std::map<mfxU32, mfxU32> VP8fourccToMFXfourcc = {
    { MFX_FOURCC_VP8_NV12,   MFX_FOURCC_NV12 },
    { MFX_FOURCC_VP8_MBDATA, MFX_FOURCC_NV12 },
    { MFX_FOURCC_VP8_SEGMAP, MFX_FOURCC_P8   },
};

QSVAllocatorVA::QSVAllocatorVA() : m_dpy(0) {
}

QSVAllocatorVA::~QSVAllocatorVA() {
    Close();
}

mfxStatus QSVAllocatorVA::Init(mfxAllocatorParams *pParams) {
    QSVAllocatorParamsVA *p_vaapiParams = dynamic_cast<QSVAllocatorParamsVA *>(pParams);
    if ((NULL == p_vaapiParams) || (NULL == p_vaapiParams->m_dpy)) {
        return MFX_ERR_NOT_INITIALIZED;
    }

    m_dpy = p_vaapiParams->m_dpy;
    return MFX_ERR_NONE;
}

mfxStatus QSVAllocatorVA::CheckRequestType(mfxFrameAllocRequest *request) {
    mfxStatus sts = QSVAllocator::CheckRequestType(request);
    if (MFX_ERR_NONE != sts) {
        return sts;
    }
    return ((request->Type & (MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET | MFX_MEMTYPE_VIDEO_MEMORY_PROCESSOR_TARGET)) != 0) ?
        MFX_ERR_NONE: MFX_ERR_UNSUPPORTED;
}

mfxStatus QSVAllocatorVA::Close() {
    return QSVAllocator::Close();
}

mfxStatus QSVAllocatorVA::AllocImpl(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response) {
    mfxStatus mfx_res = MFX_ERR_NONE;

    memset(response, 0, sizeof(mfxFrameAllocResponse));

    mfxU32 fourcc = request->Info.FourCC;
    if (VP8fourccToMFXfourcc.find(fourcc) == VP8fourccToMFXfourcc.end()) {
        return MFX_ERR_MEMORY_ALLOC;
    }
    const mfxU32 mfx_fourcc = VP8fourccToMFXfourcc.at(fourcc);
    if (fourccToVAFormat.find(mfx_fourcc) == fourccToVAFormat.end()) {
        return MFX_ERR_MEMORY_ALLOC;
    }
    mfxU32 va_fourcc = fourccToVAFormat.at(mfx_fourcc);
    static const auto SUPPORTED_FORMATS = make_array<uint32_t>(
        (uint32_t)VA_FOURCC_NV12,
        (uint32_t)VA_FOURCC_YV12,
        (uint32_t)VA_FOURCC_YUY2,
        (uint32_t)VA_FOURCC_ARGB,
        (uint32_t)VA_FOURCC_P208
    );
    if (std::find(SUPPORTED_FORMATS.begin(), SUPPORTED_FORMATS.end(), va_fourcc) == SUPPORTED_FORMATS.end()) {
        return MFX_ERR_MEMORY_ALLOC;
    }

    mfxU16 surfaces_num = request->NumFrameSuggested;
    if (!surfaces_num) {
        return MFX_ERR_MEMORY_ALLOC;
    }

    bool bCreateSrfSucceeded = false;
    mfxU32 numAllocated = 0;
    VASurfaceID *surfaces = (VASurfaceID *)calloc(surfaces_num, sizeof(VASurfaceID));
    vaapiMemId *vaapi_mids = (vaapiMemId *)calloc(surfaces_num, sizeof(vaapiMemId));
    mfxMemId *mids = (mfxMemId *)calloc(surfaces_num, sizeof(mfxMemId));
    if ((NULL == surfaces) || (NULL == vaapi_mids) || (NULL == mids)) {
        mfx_res = MFX_ERR_MEMORY_ALLOC;
    }
    if (MFX_ERR_NONE == mfx_res) {
        if (va_fourcc != VA_FOURCC_P208) {
            unsigned int format = va_fourcc;
            VASurfaceAttrib attrib;
            attrib.type          = VASurfaceAttribPixelFormat;
            attrib.flags         = VA_SURFACE_ATTRIB_SETTABLE;
            attrib.value.type    = VAGenericValueTypeInteger;
            attrib.value.value.i = va_fourcc;

            if (fourcc == MFX_FOURCC_VP8_NV12) {
                // special configuration for NV12 surf allocation for VP8 hybrid encoder is required
                attrib.type          = (VASurfaceAttribType)VASurfaceAttribUsageHint;
                attrib.value.value.i = VA_SURFACE_ATTRIB_USAGE_HINT_ENCODER;
            } else if (fourcc == MFX_FOURCC_VP8_MBDATA) {
                // special configuration for MB data surf allocation for VP8 hybrid encoder is required
                attrib.value.value.i = VA_FOURCC_P208;
                format               = VA_FOURCC_P208;
            } else if (va_fourcc == VA_FOURCC_NV12) {
                format = VA_RT_FORMAT_YUV420;
            }

            auto va_res = vaCreateSurfaces(m_dpy, format, request->Info.Width, request->Info.Height, surfaces, surfaces_num, &attrib, 1);
            mfx_res = va_to_mfx_status(va_res);
            bCreateSrfSucceeded = (MFX_ERR_NONE == mfx_res);
        } else {
            VAContextID context_id = request->reserved[0];
            int codedbuf_size = 0;
            VABufferType codedbuf_type;
            if (fourcc == MFX_FOURCC_VP8_SEGMAP) {
                codedbuf_size = request->Info.Width * request->Info.Height;
                codedbuf_type = (VABufferType)VAEncMacroblockMapBufferType;
            } else {
                int width32 = ALIGN32(request->Info.Width);
                int height32 = ALIGN32(request->Info.Height);
                codedbuf_size = static_cast<int>((width32 * height32) * 400LL / (16 * 16));
                codedbuf_type = VAEncCodedBufferType;
            }

            for (numAllocated = 0; numAllocated < surfaces_num; numAllocated++) {
                VABufferID coded_buf;
                auto va_res = vaCreateBuffer(m_dpy, context_id, codedbuf_type, codedbuf_size, 1, NULL, &coded_buf);
                mfx_res = va_to_mfx_status(va_res);
                if (MFX_ERR_NONE != mfx_res) break;
                surfaces[numAllocated] = coded_buf;
            }
        }
    }
    if (MFX_ERR_NONE == mfx_res) {
        for (mfxU32 i = 0; i < surfaces_num; ++i) {
            vaapiMemId *vaapi_mid = &(vaapi_mids[i]);
            vaapi_mid->m_fourcc = fourcc;
            vaapi_mid->m_surface = &(surfaces[i]);
            mids[i] = vaapi_mid;
        }
        response->mids = mids;
        response->NumFrameActual = surfaces_num;
    } else {
        response->mids = NULL;
        response->NumFrameActual = 0;
        if (VA_FOURCC_P208 != va_fourcc || fourcc == MFX_FOURCC_VP8_MBDATA ) {
            if (bCreateSrfSucceeded) {
                vaDestroySurfaces(m_dpy, surfaces, surfaces_num);
            }
        } else {
            for (mfxU32 i = 0; i < numAllocated; i++) {
                vaDestroyBuffer(m_dpy, surfaces[i]);
            }
        }
        if (mids)       { free(mids);       mids = NULL; }
        if (vaapi_mids) { free(vaapi_mids); vaapi_mids = NULL; }
        if (surfaces)   { free(surfaces);   surfaces = NULL; }
    }
    return mfx_res;
}

mfxStatus QSVAllocatorVA::ReleaseResponse(mfxFrameAllocResponse *response) {
    if (!response) return MFX_ERR_NULL_PTR;

    if (response->mids) {
        vaapiMemId *vaapi_mids = (vaapiMemId*)(response->mids[0]);
        mfxU32 mfx_fourcc = VP8fourccToMFXfourcc.at(vaapi_mids->m_fourcc);
        bool isBitstreamMemory = (MFX_FOURCC_P8 == mfx_fourcc);
        VASurfaceID *surfaces = vaapi_mids->m_surface;
        for (mfxU32 i = 0; i < response->NumFrameActual; ++i) {
            if (MFX_FOURCC_P8 == vaapi_mids[i].m_fourcc) {
                vaDestroyBuffer(m_dpy, surfaces[i]);
            } else if (vaapi_mids[i].m_sys_buffer) {
                free(vaapi_mids[i].m_sys_buffer);
            }
        }
        free(vaapi_mids);
        free(response->mids);
        response->mids = NULL;

        if (!isBitstreamMemory) {
            vaDestroySurfaces(m_dpy, surfaces, response->NumFrameActual);
        }
        free(surfaces);
    }
    response->NumFrameActual = 0;
    return MFX_ERR_NONE;
}

mfxStatus QSVAllocatorVA::FrameLock(mfxMemId mid, mfxFrameData *ptr) {
    mfxStatus mfx_res = MFX_ERR_NONE;
    VAStatus  va_res  = VA_STATUS_SUCCESS;
    vaapiMemId* vaapi_mid = (vaapiMemId*)mid;
    mfxU8* pBuffer = 0;
    VASurfaceAttrib attrib;

    if (!vaapi_mid || !(vaapi_mid->m_surface)) return MFX_ERR_INVALID_HANDLE;

    mfxU32 mfx_fourcc = VP8fourccToMFXfourcc.at(vaapi_mid->m_fourcc);

    if (MFX_FOURCC_P8 == mfx_fourcc) {  // bitstream processing
        VACodedBufferSegment *coded_buffer_segment;
        va_res = vaMapBuffer(m_dpy, *(vaapi_mid->m_surface), 
            (vaapi_mid->m_fourcc == MFX_FOURCC_VP8_SEGMAP) ? (void **)(&pBuffer) : (void **)(&coded_buffer_segment));
        mfx_res = va_to_mfx_status(va_res);
        if (MFX_ERR_NONE == mfx_res) {
            ptr->Y = (vaapi_mid->m_fourcc == MFX_FOURCC_VP8_SEGMAP) ? pBuffer : (mfxU8*)coded_buffer_segment->buf;
        }
    } else { // Image processing
        if (MFX_ERR_NONE != (mfx_res = va_to_mfx_status(vaSyncSurface(m_dpy, *(vaapi_mid->m_surface))))) {
            return mfx_res;
        }
        if (MFX_ERR_NONE != (mfx_res = va_to_mfx_status(vaDeriveImage(m_dpy, *(vaapi_mid->m_surface), &(vaapi_mid->m_image))))) {
            return mfx_res;
        }
        if (MFX_ERR_NONE != (mfx_res = va_to_mfx_status(vaMapBuffer(m_dpy, vaapi_mid->m_image.buf, (void **) &pBuffer)))) {
            return mfx_res;
        }
        switch (vaapi_mid->m_image.format.fourcc) {
        case VA_FOURCC_NV12:
            ptr->Pitch = (mfxU16)vaapi_mid->m_image.pitches[0];
            ptr->Y = pBuffer + vaapi_mid->m_image.offsets[0];
            ptr->U = pBuffer + vaapi_mid->m_image.offsets[1];
            ptr->V = ptr->U + 1;
            break;
        case VA_FOURCC_YV12:
            ptr->Pitch = (mfxU16)vaapi_mid->m_image.pitches[0];
            ptr->Y = pBuffer + vaapi_mid->m_image.offsets[0];
            ptr->V = pBuffer + vaapi_mid->m_image.offsets[1];
            ptr->U = pBuffer + vaapi_mid->m_image.offsets[2];
            break;
        case VA_FOURCC_YUY2:
            ptr->Pitch = (mfxU16)vaapi_mid->m_image.pitches[0];
            ptr->Y = pBuffer + vaapi_mid->m_image.offsets[0];
            ptr->U = ptr->Y + 1;
            ptr->V = ptr->Y + 3;
            break;
        case VA_FOURCC_ARGB:
            ptr->Pitch = (mfxU16)vaapi_mid->m_image.pitches[0];
            ptr->B = pBuffer + vaapi_mid->m_image.offsets[0];
            ptr->G = ptr->B + 1;
            ptr->R = ptr->B + 2;
            ptr->A = ptr->B + 3;
            break;
        case VA_FOURCC_P208:
            ptr->Pitch = (mfxU16)vaapi_mid->m_image.pitches[0];
            ptr->Y = pBuffer + vaapi_mid->m_image.offsets[0];
            break;
        default:
            mfx_res = MFX_ERR_LOCK_MEMORY;
            break;
        }
    }
    return mfx_res;
}

mfxStatus QSVAllocatorVA::FrameUnlock(mfxMemId mid, mfxFrameData *ptr) {
    vaapiMemId* vaapi_mid = (vaapiMemId*)mid;
    if (!vaapi_mid || !(vaapi_mid->m_surface)) {
        return MFX_ERR_INVALID_HANDLE;
    }
    if (VP8fourccToMFXfourcc.find(vaapi_mid->m_fourcc) == VP8fourccToMFXfourcc.end()) {
        return MFX_ERR_INVALID_HANDLE;
    }

    const auto mfx_fourcc = VP8fourccToMFXfourcc.at(vaapi_mid->m_fourcc);

    if (MFX_FOURCC_P8 == mfx_fourcc) {
        vaUnmapBuffer(m_dpy, *(vaapi_mid->m_surface));
    } else {
        vaUnmapBuffer(m_dpy, vaapi_mid->m_image.buf);
        vaDestroyImage(m_dpy, vaapi_mid->m_image.image_id);
        if (NULL != ptr) {
            ptr->Pitch = 0;
            ptr->Y     = nullptr;
            ptr->U     = nullptr;
            ptr->V     = nullptr;
            ptr->A     = nullptr;
        }
    }
    return MFX_ERR_NONE;
}

mfxStatus QSVAllocatorVA::GetFrameHDL(mfxMemId mid, mfxHDL *handle) {
    vaapiMemId* vaapi_mid = (vaapiMemId*)mid;
    if (!handle || !vaapi_mid || !(vaapi_mid->m_surface)) return MFX_ERR_INVALID_HANDLE;
    *handle = vaapi_mid->m_surface;
    return MFX_ERR_NONE;
}

#endif // #if defined(LIBVA_SUPPORT)
