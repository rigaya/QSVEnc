/*********************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2005-2015 Intel Corporation. All Rights Reserved.

**********************************************************************************/

#ifndef __SAMPLE_UTILS_H__
#define __SAMPLE_UTILS_H__

#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>

#include "mfxstructures.h"
#include "mfxvideo.h"
#include "mfxjpeg.h"
#include "mfxplugin.h"

#include "vm/strings_defs.h"
#include "vm/file_defs.h"
#include "vm/time_defs.h"
#include "vm/atomic_defs.h"

#include "sample_types.h"
#include "sample_defs.h"
#include "qsv_prm.h"
#include "qsv_control.h"
#include "qsv_event.h"
#include "convert_csp.h"

using std::vector;

#include "abstract_splitter.h"
#include "avc_bitstream.h"
#include "avc_spl.h"
#include "avc_headers.h"
#include "avc_nal_spl.h"

// A macro to disallow the copy constructor and operator= functions
// This should be used in the private: declarations for a class
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName&);               \
    void operator=(const TypeName&)

//! Base class for types that should not be assigned.
class no_assign {
    // Deny assignment
    void operator=(const no_assign&);
public:
#if __GNUC__
    //! Explicitly define default construction, because otherwise gcc issues gratuitous warning.
    no_assign() {}
#endif /* __GNUC__ */
};

//! Base class for types that should not be copied or assigned.
class no_copy : no_assign {
    //! Deny copy construction
    no_copy(const no_copy&);
public:
    //! Allow default construction
    no_copy() {}
};

typedef std::basic_string<msdk_char> msdk_string;
typedef std::basic_stringstream<msdk_char> msdk_stringstream;
typedef std::basic_ostream<msdk_char, std::char_traits<msdk_char> > msdk_ostream;
typedef std::basic_istream<msdk_char, std::char_traits<msdk_char> > msdk_istream;

#ifdef UNICODE 
#define msdk_cout std::wcout
#define msdk_err std::wcerr
#else
#define msdk_cout std::cout
#define msdk_err std::cerr
#endif

struct DeletePtr {
    template <class T> T* operator () (T* p) const {
        delete p;
        return 0;
    }
};

enum {
    CODEC_VP8 = MFX_MAKEFOURCC('V','P','8',' '),
    CODEC_MVC = MFX_MAKEFOURCC('M','V','C',' '),
};

static inline int GetFreeSurface(mfxFrameSurface1* pSurfacesPool, int nPoolSize) {
    static const int SleepInterval = 1; // milliseconds
    //wait if there's no free surface
    for (mfxU32 j = 0; j < MSDK_WAIT_INTERVAL; j += SleepInterval) {
        for (mfxU16 i = 0; i < nPoolSize; i++) {
            if (0 == pSurfacesPool[i].Data.Locked)
                return i;
        }
        MSDK_SLEEP(SleepInterval);
    }
    return MSDK_INVALID_SURF_IDX;
}

static inline mfxU16 GetFreeSurfaceIndex(mfxFrameSurface1* pSurfacesPool, mfxU16 nPoolSize, mfxU16 step)
{
    if (pSurfacesPool)
    {
        for (mfxU16 i = 0; i < nPoolSize; i = (mfxU16)(i + step), pSurfacesPool += step)
        {
            if (0 == pSurfacesPool[0].Data.Locked)
            {
                return i;
            }
        }
    }

    return MSDK_INVALID_SURF_IDX;
}

// sets bitstream->PicStruct parsing first APP0 marker in bitstream
mfxStatus MJPEG_AVI_ParsePicStruct(mfxBitstream *bitstream);

// For MVC encoding/decoding purposes
std::basic_string<msdk_char> FormMVCFileName(const msdk_char *strFileName, const mfxU32 numView);

// function for getting a pointer to a specific external buffer from the array
mfxExtBuffer* GetExtBuffer(mfxExtBuffer** ebuffers, mfxU32 nbuffers, mfxU32 BufferId);

//declare used extended buffers
template<class T>
struct mfx_ext_buffer_id{
    enum { id = 0 };
};
template<>struct mfx_ext_buffer_id<mfxExtCodingOption>{
    enum { id = MFX_EXTBUFF_CODING_OPTION };
};
template<>struct mfx_ext_buffer_id<mfxExtCodingOption2>{
    enum { id = MFX_EXTBUFF_CODING_OPTION2 };
};
template<>struct mfx_ext_buffer_id<mfxExtAvcTemporalLayers>{
    enum { id = MFX_EXTBUFF_AVC_TEMPORAL_LAYERS };
};
template<>struct mfx_ext_buffer_id<mfxExtAVCRefListCtrl>{
    enum { id = MFX_EXTBUFF_AVC_REFLIST_CTRL };
};
template<>struct mfx_ext_buffer_id<mfxExtThreadsParam>{
    enum {id = MFX_EXTBUFF_THREADS_PARAM};
};


//helper function to initialize mfx ext buffer structure
template <class T>
void init_ext_buffer(T & ext_buffer)
{
    memset(&ext_buffer, 0, sizeof(ext_buffer));
    reinterpret_cast<mfxExtBuffer*>(&ext_buffer)->BufferId = mfx_ext_buffer_id<T>::id;
    reinterpret_cast<mfxExtBuffer*>(&ext_buffer)->BufferSz = sizeof(ext_buffer);
}

// returns false if buf length is insufficient, otherwise
// skips step bytes in buf with specified length and returns true
template <typename Buf_t, typename Length_t>
bool skip(const Buf_t *&buf, Length_t &length, Length_t step)
{
    if (length < step)
        return false;

    buf    += step;
    length -= step;

    return true;
}

#endif //__SAMPLE_UTILS_H__
