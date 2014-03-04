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

class AUO_YUVReader : public CSmplYUVReader
{
public :

    AUO_YUVReader();
    virtual ~AUO_YUVReader();

	virtual void Close();
	virtual mfxStatus Init(const TCHAR *strFileName, mfxU32 ColorFormat, int option, CEncodingThread *pEncThread, CEncodeStatusInfo *pEncSatusInfo, sInputCrop *pInputCrop);
    virtual mfxStatus LoadNextFrame(mfxFrameSurface1* pSurface);

private:
	mfxU32 current_frame;
	BOOL pause;
};

class AUO_EncodeStatusInfo : public CEncodeStatusInfo
{
public :
    AUO_EncodeStatusInfo();
    virtual ~AUO_EncodeStatusInfo();
private:
	virtual void UpdateDisplay(const char *mes, int drop_frames);
	virtual void WriteLine(const TCHAR *mes);
};

#endif //_AUO_YUVREADER_H_