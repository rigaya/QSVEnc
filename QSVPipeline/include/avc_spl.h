//* ////////////////////////////////////////////////////////////////////////////// */
//*
//
//              INTEL CORPORATION PROPRIETARY INFORMATION
//  This software is supplied under the terms of a license  agreement or
//  nondisclosure agreement with Intel Corporation and may not be copied
//  or disclosed except in  accordance  with the terms of that agreement.
//        Copyright (c) 2010-2015 Intel Corporation. All Rights Reserved.
//
//
//*/

#ifndef _AVC_SPL_H__
#define _AVC_SPL_H__

#include <vector>
#include <list>
#pragma warning(disable : 4201)
#include <memory>
#pragma warning(default : 4201)

#include "abstract_splitter.h"

#include "avc_bitstream.h"
#include "avc_headers.h"
#include "avc_nal_spl.h"

namespace ProtectedLibrary
{

class AVCSlice : public SliceSplitterInfo
{
public:

    AVCSlice();

    AVCSliceHeader * GetSliceHeader();

    bool IsField() const {return m_sliceHeader.field_pic_flag != 0;}

    mfxI32 RetrievePicParamSetNumber(mfxU8 *pSource, mfxU32 nSourceSize);

    bool DecodeHeader(mfxU8 *pSource, mfxU32 nSourceSize);

    AVCHeadersBitstream *GetBitStream(void){return &m_bitStream;}

    AVCPicParamSet* m_picParamSet;
    AVCSeqParamSet* m_seqParamSet;
    AVCSeqParamSet* m_seqParamSetMvcEx;
    AVCSeqParamSetExtension* m_seqParamSetEx;

    mfxU64 m_dTime;

protected:
    AVCSliceHeader m_sliceHeader;
    AVCHeadersBitstream m_bitStream;

    void Reset();
};

class AVCFrameInfo
{
public:

    AVCFrameInfo();

    void Reset();

    AVCSlice  * m_slice;
    mfxU32      m_index;
};

class AVC_Spl : public AbstractSplitter
{
public:

    AVC_Spl();

    virtual ~AVC_Spl();

    virtual mfxStatus Reset();

    virtual mfxStatus GetFrame(mfxBitstream * bs_in, FrameSplitterInfo ** frame);

    virtual mfxStatus PostProcessing(FrameSplitterInfo *frame, mfxU32 sliceNum);

    void ResetCurrentState();

protected:
    std::auto_ptr<NALUnitSplitter> m_pNALSplitter;

    mfxStatus Init();

    void Close();

    mfxStatus ProcessNalUnit(mfxI32 nalType, mfxBitstream * destination);

    mfxStatus DecodeHeader(mfxBitstream * nalUnit);
    mfxStatus DecodeSEI(mfxBitstream * nalUnit);
    AVCSlice * DecodeSliceHeader(mfxBitstream * nalUnit);
    mfxStatus AddSlice(AVCSlice * pSlice);

    AVCFrameInfo * GetFreeFrame();

    mfxU8 * GetMemoryForSwapping(mfxU32 size);

    mfxStatus AddNalUnit(mfxBitstream * nalUnit);
    mfxStatus AddSliceNalUnit(mfxBitstream * nalUnit, AVCSlice * pSlice);
    bool IsFieldOfOneFrame(AVCFrameInfo * frame, const AVCSliceHeader * slice1, const AVCSliceHeader *slice2);

    bool                m_WaitForIDR;

    AVCHeaders     m_headers;
    std::auto_ptr<AVCFrameInfo> m_AUInfo;
    AVCFrameInfo * m_currentInfo;
    AVCSlice * m_pLastSlice;

    mfxBitstream * m_lastNalUnit;

    enum
    {
        BUFFER_SIZE = 1024 * 1024
    };

    std::vector<mfxU8>  m_currentFrame;
    std::vector<mfxU8>  m_swappingMemory;
    std::list<AVCSlice> m_slicesStorage;

    std::vector<SliceSplitterInfo>  m_slices;
    FrameSplitterInfo m_frame;
};

} // namespace ProtectedLibrary

#endif // _AVC_SPL_H__
