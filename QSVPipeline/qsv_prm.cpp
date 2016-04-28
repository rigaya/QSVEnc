// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// --------------------------------------------------------------------------------------------

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
    prm->bRDO              = false;

    prm->nVQPStrength      = QSV_DEFAULT_VQP_STRENGTH;
    prm->nVQPSensitivity   = QSV_DEFAULT_VQP_SENSITIVITY;
    prm->nPerfMonitorInterval = QSV_DEFAULT_PERF_MONITOR_INTERVAL;
    prm->nOutputBufSizeMB  = QSV_DEFAULT_OUTPUT_BUF_MB;
    prm->nInputBufSize     = QSV_DEFAULT_INPUT_BUF_HW;
    prm->nOutputThread     = QSV_OUTPUT_THREAD_AUTO;
    prm->nAudioThread      = QSV_AUDIO_THREAD_AUTO;
    prm->nAudioIgnoreDecodeError = QSV_DEFAULT_AUDIO_IGNORE_DECODE_ERROR;

    prm->nDstWidth          = 1280;
    prm->nDstHeight         = 720;
    prm->vpp.nDenoise       = 20;
    prm->vpp.nDetailEnhance = 20;
    prm->vpp.delogo.nDepth  = QSV_DEFAULT_VPP_DELOGO_DEPTH;
}
