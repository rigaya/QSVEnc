//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#define USE_SSE2   1
#define USE_SSSE3  1
#define USE_SSE41  1
#define USE_AVX    1
#define USE_AVX2   1
#define USE_FMA3   1
#define USE_POPCNT 1
#include "delogo_process_simd.h"
#include "delogo_process.h"

#if _MSC_VER >= 1800 && !defined(__AVX__) && !defined(_DEBUG)
static_assert(false, "do not forget to set /arch:AVX or /arch:AVX2 for this file.");
#endif

static __declspec(noinline) void process_delogo_frame_avx2(mfxU8 *dst, const mfxU32 dst_pitch, mfxU8 *buffer,
    mfxU8 *src, const mfxU32 src_pitch, const mfxU32 width, const mfxU32 height_start, const mfxU32 height_fin, const ProcessDataDelogo *data) {
    process_delogo_frame(dst, dst_pitch, buffer, src, src_pitch, width, height_start, height_fin, data);
}

template<mfxU32 step>
static __declspec(noinline) void process_delogo_avx2(mfxU8 *ptr, const mfxU32 pitch, mfxU8 *buffer, mfxU32 height_start, mfxU32 height_fin, const ProcessDataDelogo *data) {
    process_delogo<step>(ptr, pitch, buffer, height_start, height_fin, data);
}

DelogoProcessAVX2::DelogoProcessAVX2() : ProcessorDelogo() {
}

DelogoProcessAVX2::~DelogoProcessAVX2() {
}

mfxStatus DelogoProcessAVX2::Process(DataChunk *chunk, mfxU8 *pBuffer) {
    MSDK_CHECK_POINTER(chunk, MFX_ERR_NULL_PTR);

    if (m_pIn->Info.FourCC != MFX_FOURCC_NV12) {
        return MFX_ERR_UNSUPPORTED;
    }

    mfxStatus sts = MFX_ERR_NONE;
    
    if (MFX_ERR_NONE != (sts = LockFrame(m_pIn))) return sts;
    if (MFX_ERR_NONE != (sts = LockFrame(m_pOut))) {
        return UnlockFrame(m_pIn);
    }

    process_delogo_frame_avx2(m_pOut->Data.Y,  m_pOut->Data.Pitch, pBuffer, m_pIn->Data.Y,  m_pIn->Data.Pitch, m_pIn->Info.CropW, 0, m_pIn->Info.CropH,      m_sData[0]);
    process_delogo_frame_avx2(m_pOut->Data.UV, m_pOut->Data.Pitch, pBuffer, m_pIn->Data.UV, m_pIn->Data.Pitch, m_pIn->Info.CropW, 0, m_pIn->Info.CropH >> 1, m_sData[1]);

    if (MFX_ERR_NONE != (sts = UnlockFrame(m_pIn)))  return sts;
    if (MFX_ERR_NONE != (sts = UnlockFrame(m_pOut))) return sts;

    return sts;
}

DelogoProcessD3DAVX2::DelogoProcessD3DAVX2() : ProcessorDelogo() {
}

DelogoProcessD3DAVX2::~DelogoProcessD3DAVX2() {
}

mfxStatus DelogoProcessD3DAVX2::Process(DataChunk *chunk, mfxU8 *pBuffer) {
    MSDK_CHECK_POINTER(chunk, MFX_ERR_NULL_PTR);

    if (m_pIn->Info.FourCC != MFX_FOURCC_NV12) {
        return MFX_ERR_UNSUPPORTED;
    }

    mfxStatus sts = MFX_ERR_NONE;

    if (MFX_ERR_NONE != (sts = CopyD3DFrameGPU(m_pIn, m_pOut))) {
        return sts;
    }
    
    if (MFX_ERR_NONE != (sts = LockFrame(m_pOut))) {
        return sts;
    }

    process_delogo_avx2<64>(m_pOut->Data.Y,  m_pOut->Data.Pitch, pBuffer, 0, m_pIn->Info.CropH,      m_sData[0]);
    process_delogo_avx2<32>(m_pOut->Data.UV, m_pOut->Data.Pitch, pBuffer, 0, m_pIn->Info.CropH >> 1, m_sData[1]);

    return UnlockFrame(m_pOut);
}
