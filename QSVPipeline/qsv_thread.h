//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef __QSV_THREAD_H__
#define __QSV_THREAD_H__

#include <thread>
#include "qsv_osdep.h"

static void QSV_FORCEINLINE sleep_hybrid(int count) {
    _mm_pause();
    if ((count & 4095) == 4095) {
        std::this_thread::sleep_for(std::chrono::milliseconds((count & 65535) == 65535));
    }
}

#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>

static inline bool CheckThreadAlive(std::thread& thread) {
    DWORD exit_code = 0;
    return (0 != GetExitCodeThread(thread.native_handle(), &exit_code)) && exit_code == STILL_ACTIVE;
}

#else //#if defined(_WIN32) || defined(_WIN64)
#include <pthread.h>
#include <signal.h>

static inline bool CheckThreadAlive(std::thread& thread) {
    uint32_t exit_code = 0;
    return pthread_kill(thread.native_handle(), 0) != ESRCH;
}

#endif //#if defined(_WIN32) || defined(_WIN64)

#endif //__QSV_THREAD_H__
