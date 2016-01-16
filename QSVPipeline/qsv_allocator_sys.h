//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef __QSV_ALLOCATOR_SYS_H__
#define __QSV_ALLOCATOR_SYS_H__

#include <memory>
#include "mfxvideo.h"
#include "qsv_allocator.h"

struct sBuffer {
    mfxU32 id;
    mfxU32 nbytes;
    mfxU16 type;
};

struct sFrame {
    mfxU32       id;
    mfxFrameInfo info;
};

class QSVBufferAllocatorSys : public QSVBufferAllocator {
public:
    QSVBufferAllocatorSys();
    virtual ~QSVBufferAllocatorSys();
    virtual mfxStatus BufAlloc(mfxU32 nbytes, mfxU16 type, mfxMemId *mid) override;
    virtual mfxStatus BufLock(mfxMemId mid, mfxU8 **ptr) override;
    virtual mfxStatus BufUnlock(mfxMemId mid) override;
    virtual mfxStatus BufFree(mfxMemId mid) override;
};

class QSVAllocatorSys : public QSVAllocator {
public:
    QSVAllocatorSys();
    virtual ~QSVAllocatorSys();

    virtual mfxStatus Init(mfxAllocatorParams *pParams) override;
    virtual mfxStatus Close() override;
    virtual mfxStatus FrameLock(mfxMemId mid, mfxFrameData *ptr) override;
    virtual mfxStatus FrameUnlock(mfxMemId mid, mfxFrameData *ptr) override;
    virtual mfxStatus GetFrameHDL(mfxMemId mid, mfxHDL *handle) override;

protected:
    virtual mfxStatus CheckRequestType(mfxFrameAllocRequest *request) override;
    virtual mfxStatus ReleaseResponse(mfxFrameAllocResponse *response) override;
    virtual mfxStatus AllocImpl(mfxFrameAllocRequest *request, mfxFrameAllocResponse *response) override;

    std::unique_ptr<QSVBufferAllocatorSys> m_pBufferAllocator;
};

#endif // __QSV_ALLOCATOR_SYS_H__