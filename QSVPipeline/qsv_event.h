//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef __QSV_EVENT_H__
#define __QSV_EVENT_H__

#include <cstdint>
#include <climits>
#include "qsv_osdep.h"

#if defined(_WIN32) || defined(_WIN64)
#define CloseEvent CloseHandle
#else //#if defined(_WIN32) || defined(_WIN64)

enum : uint32_t {
    WAIT_OBJECT_0 = 0,
    WAIT_TIMEOUT = 258L,
    WAIT_ABANDONED_0 = 0x00000080L
};

static const uint32_t INFINITE = UINT_MAX;

void ResetEvent(HANDLE ev);

void SetEvent(HANDLE ev);

HANDLE CreateEvent(void *pDummy, int bManualReset, int bInitialState, void *pDummy2);

void CloseEvent(HANDLE ev);

uint32_t WaitForSingleObject(HANDLE ev, uint32_t millisec);

uint32_t WaitForMultipleObjects(uint32_t count, HANDLE *pev, int dummy, uint32_t millisec);

#endif //#if defined(_WIN32) || defined(_WIN64)

#endif //__QSV_EVENT_H__
