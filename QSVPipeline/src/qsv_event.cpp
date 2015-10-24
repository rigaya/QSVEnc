//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#if !(defined(_WIN32) || defined(_WIN64))
#include "qsv_event.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <climits>
#include <chrono>
#include <algorithm>

class Event {
public:
    bool bManualReset;
    bool bReady;
    std::mutex mtx;
    std::condition_variable cv;

    Event() : bManualReset(false), bReady(false), mtx(), cv() {

    };
    Event(bool manualReset) : Event() {
        bManualReset = manualReset;
    };
};

void ResetEvent(HANDLE ev) {
    Event *event = (Event *)ev;
    event->bReady = false;
}

void SetEvent(HANDLE ev) {
    Event *event = (Event *)ev;

    {
        std::lock_guard<std::mutex> lock(event->mtx);
        event->bReady = true;
    }
    event->cv.notify_one();
}

HANDLE CreateEvent(void *pDummy, int bManualReset, int bInitialState, void *pDummy2) {
    Event *event = new Event(!!bManualReset);
    if (bInitialState) {
        SetEvent(event);
    }
    return event;
}

void CloseEvent(HANDLE ev) {
    if (ev != NULL) {
        Event *event = (Event *)ev;
       delete event;
    }
}

uint32_t WaitForSingleObject(HANDLE ev, uint32_t millisec) {
    Event *event = (Event *)ev;
    {
        std::unique_lock<std::mutex> uniq_lk(event->mtx);
        if (millisec == INFINITE) {
            event->cv.wait(uniq_lk, [&event]{ return event->bReady;});
        } else {
            event->cv.wait_for(uniq_lk, std::chrono::milliseconds(millisec), [&event]{ return event->bReady;});
            if (!event->bReady) {
                return WAIT_TIMEOUT;
            }
        }
    }
    if (!event->bManualReset) {
        ResetEvent(ev);
    }
    return WAIT_OBJECT_0;
}

uint32_t WaitForMultipleObjects(uint32_t count, HANDLE *pev, int dummy, uint32_t millisec) {
    Event **pevent = (Event **)pev;
    int success = 0;
    bool bTimeout = false;
    for (uint32_t i = 0; i < count; i++) {
        if (WAIT_TIMEOUT == WaitForSingleObject(pevent[i], (bTimeout) ? 0 : millisec)) {
            bTimeout = true;
        } else {
            success++;
        }
    }
    return (bTimeout) ? WAIT_TIMEOUT : (WAIT_OBJECT_0 + success);
}
#endif //#if !(defined(_WIN32) || defined(_WIN64))
