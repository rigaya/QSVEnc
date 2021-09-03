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

#ifndef __QSV_ALLOCATOR_H__
#define __QSV_ALLOCATOR_H__

#include <list>
#include <memory>
#include <cstring>
#include <functional>
#include "rgy_util.h"
#pragma warning (push)
#pragma warning (disable: 4201) //C4201: 非標準の拡張機能が使用されています: 無名の構造体または共用体です。
#pragma warning (disable: 4996) //C4996: 'MFXInit': が古い形式として宣言されました。
#pragma warning (disable: 4819) //C4819: ファイルは、現在のコード ページ (932) で表示できない文字を含んでいます。データの損失を防ぐために、ファイルを Unicode 形式で保存してください。
RGY_DISABLE_WARNING_PUSH
RGY_DISABLE_WARNING_STR("-Wdeprecated-declarations")
#include "mfxvideo.h"
RGY_DISABLE_WARNING_POP
#pragma warning (pop)
#include "rgy_log.h"

class mfxAllocatorParams {
public:
    virtual ~mfxAllocatorParams(){};
};

class QSVAllocator : public mfxFrameAllocator {
public:
    QSVAllocator();
    virtual ~QSVAllocator();
    virtual mfxStatus Init(mfxAllocatorParams *pParams, std::shared_ptr<RGYLog> pQSVLog) = 0;
    virtual mfxStatus Close();

    virtual mfxStatus FrameAlloc(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response);
    virtual mfxStatus FrameLock(mfxMemId mid, mfxFrameData *ptr) = 0;
    virtual mfxStatus FrameUnlock(mfxMemId mid, mfxFrameData *ptr) = 0;
    virtual mfxStatus GetFrameHDL(mfxMemId mid, mfxHDL *handle) = 0;
    virtual mfxStatus FrameFree(mfxFrameAllocResponse *response);
    uint32_t getExtAllocCounts() { return (uint32_t)m_ExtResponses.size(); }
private:
    static mfxStatus MFX_CDECL Alloc_(mfxHDL pthis, mfxFrameAllocRequest *request, mfxFrameAllocResponse *response);
    static mfxStatus MFX_CDECL Lock_(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr);
    static mfxStatus MFX_CDECL Unlock_(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr);
    static mfxStatus MFX_CDECL GetHDL_(mfxHDL pthis, mfxMemId mid, mfxHDL *handle);
    static mfxStatus MFX_CDECL Free_(mfxHDL pthis, mfxFrameAllocResponse *response);

protected:
    virtual mfxStatus CheckRequestType(mfxFrameAllocRequest *request);
    virtual mfxStatus ReleaseResponse(mfxFrameAllocResponse *response) = 0;
    virtual mfxStatus AllocImpl(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response) = 0;
    static const auto MEMTYPE_FROM_MASK = MFX_MEMTYPE_FROM_ENCODE | MFX_MEMTYPE_FROM_DECODE | \
        MFX_MEMTYPE_FROM_VPPIN | MFX_MEMTYPE_FROM_VPPOUT | \
        MFX_MEMTYPE_FROM_ENC | MFX_MEMTYPE_FROM_PAK;

    static const auto MEMTYPE_FROM_MASK_INT_EXT = MEMTYPE_FROM_MASK | MFX_MEMTYPE_INTERNAL_FRAME | MFX_MEMTYPE_EXTERNAL_FRAME;

    struct UniqueResponse : mfxFrameAllocResponse {
        mfxU16 m_width;
        mfxU16 m_height;
        mfxU32 m_refCount;
        mfxU16 m_type;

        UniqueResponse()
            : m_width(0)
            , m_height(0)
            , m_refCount(0)
            , m_type(0) {
            memset(static_cast<mfxFrameAllocResponse*>(this), 0, sizeof(mfxFrameAllocResponse));
        }

        // compare responses by actual frame size, alignment (w and h) is up to application
        UniqueResponse(const mfxFrameAllocResponse & response, mfxU16 width, mfxU16 height, mfxU16 type)
            : mfxFrameAllocResponse(response)
            , m_width(width)
            , m_height(height)
            , m_refCount(1)
            , m_type(type)
        {
        }

        //compare by resolution (and memory type for FEI ENC / PAK)
        bool operator () (const UniqueResponse &response)const
        {
            if (m_width <= response.m_width && m_height <= response.m_height)
            {
                // For FEI ENC and PAK we need to distinguish between INTERNAL and EXTERNAL frames

                if (m_type & response.m_type & (MFX_MEMTYPE_FROM_ENC | MFX_MEMTYPE_FROM_PAK))
                {
                    return !!((m_type & response.m_type) & 0x000f);
                } else
                {
                    return !!(m_type & response.m_type & MFX_MEMTYPE_FROM_DECODE);
                }
            } else
            {
                return false;
            }
        }

        static mfxU16 CropMemoryTypeToStore(mfxU16 type)
        {
            // Remain INTERNAL / EXTERNAL flag for FEI ENC / PAK
            switch (type & 0xf000)
            {
            case MFX_MEMTYPE_FROM_ENC:
            case MFX_MEMTYPE_FROM_PAK:
            case (MFX_MEMTYPE_FROM_ENC | MFX_MEMTYPE_FROM_PAK):
                return type & MEMTYPE_FROM_MASK_INT_EXT;
                break;
            default:
                return type & MEMTYPE_FROM_MASK;
                break;
            }
        }
    };

    void AddMessage(RGYLogLevel log_level, const tstring &str);
    void AddMessage(RGYLogLevel log_level, const TCHAR *format, ...);

    tstring m_name;
    std::list<mfxFrameAllocResponse> m_responses;
    std::list<UniqueResponse> m_ExtResponses;
    std::shared_ptr<RGYLog> m_pQSVLog;
};

void *getSurfaceHandle(mfxFrameSurface1 *surface);

#endif // __QSV_ALLOCATOR_H__