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

#ifndef _ABSTRACT_SPL_H__
#define _ABSTRACT_SPL_H__

#include "mfxstructures.h"
#include "vm/strings_defs.h"

enum SliceTypeCode {
    TYPE_I = 0,
    TYPE_P = 1,
    TYPE_B = 2,
    TYPE_SKIP=3,
    TYPE_UNKNOWN=4
};

struct SliceSplitterInfo
{
    mfxU32   DataOffset;
    mfxU32   DataLength;
    mfxU32   HeaderLength;
    SliceTypeCode SliceType;
};

struct FrameSplitterInfo
{
    SliceSplitterInfo * Slice;   // array
    mfxU32     SliceNum;
    mfxU32     FirstFieldSliceNum;

    mfxU8  *   Data;    // including data of slices
    mfxU32     DataLength;
    mfxU64     TimeStamp;
};

class AbstractSplitter
{
public:

    AbstractSplitter()
    {}

    virtual ~AbstractSplitter()
    {}

    virtual mfxStatus Reset() = 0;

    virtual mfxStatus GetFrame(mfxBitstream * bs_in, FrameSplitterInfo ** frame) = 0;

    virtual mfxStatus PostProcessing(FrameSplitterInfo *frame, mfxU32 sliceNum) = 0;

    virtual void ResetCurrentState() = 0;
};

#endif // _ABSTRACT_SPL_H__
