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

#ifndef __QSV_ALLOCATOR_D3D11_H__
#define __QSV_ALLOCATOR_D3D11_H__

#include "qsv_allocator.h"
#include <limits>

#ifdef __gnu_linux__
#include <stdint.h> // for uintptr_t on Linux
#endif

//application can provide either generic mid from surface or this wrapper
//wrapper distinguishes from generic mid by highest 1 bit
//if it set then remained pointer points to extended structure of memid
//64 bits system layout
/*----+-----------------------------------------------------------+
|b63=1|63 bits remained for pointer to extended structure of memid|
|b63=0|63 bits from original mfxMemId                             |
+-----+----------------------------------------------------------*/
//32 bits system layout
/*--+---+--------------------------------------------+
|b31=1|31 bits remained for pointer to extended memid|
|b31=0|31 bits remained for surface pointer          |
+---+---+-------------------------------------------*/
//#pragma warning (disable:4293)
class MFXReadWriteMid {
    static const uintptr_t bits_offset = std::numeric_limits<uintptr_t>::digits - 1;
    static const uintptr_t clear_mask = ~((uintptr_t)1 << bits_offset);
public:
    enum {
        not_set = 0,
        reuse   = 1,
        read    = 2,
        write   = 4,
    };
    //here mfxmemid might be as MFXReadWriteMid or mfxMemId memid
    MFXReadWriteMid(mfxMemId mid, mfxU8 flag = not_set) {
        m_mid_to_report = (mfxMemId)((uintptr_t)&m_mid | ((uintptr_t)1 << bits_offset));
        if (0 != ((uintptr_t)mid >> bits_offset)) {
            //it points to extended structure
            mfxMedIdEx *pMemIdExt = reinterpret_cast<mfxMedIdEx *>((uintptr_t)mid & clear_mask);
            m_mid.pId = pMemIdExt->pId;
            m_mid.read_write = (reuse == flag) ? pMemIdExt->read_write : flag;
        } else {
            m_mid.pId = mid;
            m_mid.read_write = (reuse == flag) ? not_set : flag;
        }
    }
    bool isRead() const {
        return 0 != (m_mid.read_write & read) || !m_mid.read_write;
    }
    bool isWrite() const {
        return 0 != (m_mid.read_write & write) || !m_mid.read_write;
    }
    mfxMemId raw() const {
        return m_mid.pId;
    }
    operator mfxMemId() const {
        return m_mid_to_report;
    }

private:
    struct mfxMedIdEx {
        mfxMemId pId;
        mfxU8 read_write;
    };
    mfxMedIdEx m_mid;
    mfxMemId   m_mid_to_report;
};

#if (defined(_WIN32) || defined(_WIN64))

#include <d3d11.h>
#include <vector>
#include <map>

struct ID3D11VideoDevice;
struct ID3D11VideoContext;

static void IUnknownSafeRelease(IUnknown* ptr) {
    if (ptr) {
        ptr->Release();
    }
}

class QSVAllocatorParamsD3D11 : public mfxAllocatorParams {
public:
    ID3D11Device *pDevice;
    bool bUseSingleTexture;
    DWORD uncompressedResourceMiscFlags;

    QSVAllocatorParamsD3D11() : pDevice(), bUseSingleTexture(), uncompressedResourceMiscFlags() { }
};

class QSVAllocatorD3D11 : public QSVAllocator {
public:
    QSVAllocatorD3D11();
    virtual ~QSVAllocatorD3D11();

    virtual mfxStatus Init(mfxAllocatorParams *pParams, std::shared_ptr<RGYLog> pQSVLog) override;
    virtual mfxStatus Close() override;
    virtual ID3D11Device * GetD3D11Device() {
        return m_initParams.pDevice;
    };
    virtual mfxStatus FrameLock(mfxMemId mid, mfxFrameData *ptr) override;
    virtual mfxStatus FrameUnlock(mfxMemId mid, mfxFrameData *ptr) override;
    virtual mfxStatus GetFrameHDL(mfxMemId mid, mfxHDL *handle) override;

protected:
    virtual mfxStatus CheckRequestType(mfxFrameAllocRequest *request) override;
    virtual mfxStatus ReleaseResponse(mfxFrameAllocResponse *response) override;
    virtual mfxStatus AllocImpl(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response) override;

    QSVAllocatorParamsD3D11 m_initParams;
    ID3D11DeviceContext *m_pDeviceContext;

    struct TextureResource {
        std::vector<mfxMemId> outerMids;
        std::vector<ID3D11Texture2D*> textures;
        std::vector<ID3D11Texture2D*> stagingTexture;
        bool bAlloc;

        TextureResource() : bAlloc(true) { }

        static bool isAllocated(TextureResource & that) {
            return that.bAlloc;
        }
        ID3D11Texture2D* GetTexture(mfxMemId id) {
            if (outerMids.empty()) {
                return nullptr;
            }
            return textures[((uintptr_t)id - (uintptr_t)outerMids.front()) % textures.size()];
        }
        UINT GetSubResource(mfxMemId id) {
            if (outerMids.empty())
                return NULL;

            return (UINT)(((uintptr_t)id - (uintptr_t)outerMids.front()) / textures.size());
        }
        void Release() {
            for (auto texture : textures) {
                texture->Release();
            }
            for (auto texture : stagingTexture) {
                texture->Release();
            }
            textures.clear();
            stagingTexture.clear();
            bAlloc = false;
        }
    };
    class TextureSubResource {
        TextureResource *m_pTarget;
        ID3D11Texture2D *m_pTexture;
        ID3D11Texture2D *m_pStaging;
        UINT m_subResource;
    public:
        TextureSubResource(TextureResource *pTarget = nullptr, mfxMemId id = 0) :
            m_pTarget(pTarget), m_pTexture(), m_subResource(), m_pStaging(NULL) {
            if (m_pTarget && !m_pTarget->outerMids.empty()) {
                ptrdiff_t idx = (uintptr_t)MFXReadWriteMid(id).raw() - (uintptr_t)m_pTarget->outerMids.front();
                m_pTexture = m_pTarget->textures[idx % m_pTarget->textures.size()];
                m_subResource = (UINT)(idx / m_pTarget->textures.size());
                m_pStaging = m_pTarget->stagingTexture.empty() ? nullptr : m_pTarget->stagingTexture[idx];
            }
        }
        ID3D11Texture2D* GetStaging() const {
            return m_pStaging;
        }
        ID3D11Texture2D* GetTexture() const {
            return m_pTexture;
        }
        UINT GetSubResource()const {
            return m_subResource;
        }
        void Release() {
            if (NULL != m_pTarget) {
                m_pTarget->Release();
            }
        }
    };

    TextureSubResource GetResourceFromMid(mfxMemId);

    std::list<TextureResource> m_resourcesByRequest;
    std::vector<std::list<TextureResource>::iterator> m_memIdMap;
};

#endif // #if defined(_WIN32) || defined(_WIN64)
#endif // __QSV_ALLOCATOR_D3D11_H__
