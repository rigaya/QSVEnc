//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include "qsv_prm.h"
#include "qsv_pipeline.h"

void init_qsvp_prm(sInputParams *prm) {
    memset(prm, 0, sizeof(sInputParams));
    prm->CodecId           = MFX_CODEC_AVC;
    prm->nTargetUsage      = QSV_DEFAULT_QUALITY;
    prm->nEncMode          = MFX_RATECONTROL_CQP;
    prm->bUseHWLib         = true;
    prm->memType           = HW_MEMORY;
    prm->ColorFormat       = MFX_FOURCC_NV12;
    prm->nPicStruct        = MFX_PICSTRUCT_PROGRESSIVE;
    prm->nBitRate          = 3000;
    prm->nMaxBitrate       = 15000;
    prm->nFPSRate          = 30000;
    prm->nFPSScale         = 1001;
    prm->nQPI              = QSV_DEFAULT_QPI;
    prm->nQPP              = QSV_DEFAULT_QPP;
    prm->nQPB              = QSV_DEFAULT_QPB;
    prm->nICQQuality       = QSV_DEFAULT_ICQ;
    prm->nQVBRQuality      = QSV_DEFAULT_QVBR;
    prm->nAVBRAccuarcy     = QSV_DEFAULT_ACCURACY;
    prm->nAVBRConvergence  = QSV_DEFAULT_CONVERGENCE;
    prm->nIdrInterval      = 0;
    prm->nBframes          = QSV_BFRAMES_AUTO;
    prm->nGOPLength        = QSV_DEFAULT_GOP_LEN;
    prm->nRef              = QSV_DEFAULT_REF;
    prm->bopenGOP          = false;
    prm->bBPyramid         = getCPUGen() >= CPU_GEN_HASWELL;
    prm->bforceGOPSettings = QSV_DEFAULT_FORCE_GOP_LEN;
    prm->ColorPrim         = (mfxU16)list_colorprim[0].value;
    prm->ColorMatrix       = (mfxU16)list_colormatrix[0].value;
    prm->Transfer          = (mfxU16)list_transfer[0].value;
    prm->VideoFormat       = (mfxU16)list_videoformat[0].value;
    prm->nInputBufSize     = QSV_DEFAULT_INPUT_BUF_HW;
    prm->bRDO              = false;

    prm->nVQPStrength      = QSV_DEFAULT_VQP_STRENGTH;
    prm->nVQPSensitivity   = QSV_DEFAULT_VQP_SENSITIVITY;
    prm->nPerfMonitorInterval = 200;
    prm->nOutputBufSizeMB  = QSV_DEFAULT_OUTPUT_BUF_MB;

    prm->nDstWidth          = 1280;
    prm->nDstHeight         = 720;
    prm->vpp.nDenoise       = 20;
    prm->vpp.nDetailEnhance = 20;
    prm->vpp.delogo.nDepth  = QSV_DEFAULT_VPP_DELOGO_DEPTH;
}
