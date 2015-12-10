//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _AUO_YUVREADER_H_
#define _AUO_YUVREADER_H_

#include "sample_utils.h"
#include "auo_version.h"
#include "output.h"

class AUO_YUVReader : public CSmplYUVReader
{
public :

    AUO_YUVReader();
    virtual ~AUO_YUVReader();

    virtual void Close() override;
    virtual mfxStatus Init(const TCHAR *strFileName, mfxU32 ColorFormat, const void *prm, CEncodingThread *pEncThread, shared_ptr<CEncodeStatusInfo> pEncSatusInfo, sInputCrop *pInputCrop) override;
    virtual mfxStatus LoadNextFrame(mfxFrameSurface1* pSurface) override;

private:
    mfxU32 current_frame;
};

typedef struct AuoStatusData {
    const OUTPUT_INFO *oip;
};

class AUO_EncodeStatusInfo : public CEncodeStatusInfo
{
public :
    AUO_EncodeStatusInfo();
    virtual ~AUO_EncodeStatusInfo();
    virtual void SetPrivData(void *pPrivateData);
private:
    virtual void UpdateDisplay(const char *mes, int drop_frames, double progressPercent) override;
    virtual mfxStatus UpdateDisplay(int drop_frames, double progressPercent = 0.0) override;
    virtual void WriteLine(const TCHAR *mes) override;

    AuoStatusData m_auoData;
    std::chrono::system_clock::time_point m_tmLastLogUpdate;
};

#endif //_AUO_YUVREADER_H_