//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _QSV_QUEUE_H_
#define _QSV_QUEUE_H_

#include <cstdint>
#include <cstring>
#include <atomic>
#include <climits>
#include <memory>
#include "qsv_osdep.h"
#include "qsv_event.h"

template<typename Type, size_t align_byte = sizeof(Type)>
class CQueueSPSP {
    union queueData {
        Type data;
        char pad[((sizeof(Type) + (align_byte-1)) & (~(align_byte-1)))];
    };
public:
    //並列で1つの押し込みと1つの取り出しが可能なキューを作成する
    //スレッド並列対応のため、データにはパディングをつけてアライメントをとることが可能 (align_byte)
    //どこまで効果があるかは不明だが、align_byte=64としてfalse sharingを回避できる
    CQueueSPSP() : m_pBufStart(), m_heEvent(NULL), m_pBufFin(nullptr), m_pBufIn(nullptr), m_pBufOut(nullptr), m_bUsingData(false) {
        static_assert(std::is_pod<Type>::value == true, "CQueueSPSP is only for POD type.");
        //実際のメモリのアライメントに適切な2の倍数であるか確認する
        //そうでない場合は32をデフォルトとして使用
        m_nMallocAlign = 32;
        for (int i = 0; i < sizeof(i) * 8; i++) {
            int test = 1 << i;
            if (test == align_byte) {
                m_nMallocAlign = test;
                break;
            }
        }
        m_heEvent = CreateEvent(NULL, TRUE, TRUE, NULL);
    }
    ~CQueueSPSP() {
        clear();
    }
    //キューを初期化する
    //bufSizeはキューの内部データバッファサイズ maxCapacityを超えてもかまわない
    //maxCapacityはキューに格納できる最大のデータ数
    void init(size_t bufSize = 1024, size_t maxCapacity = SIZE_MAX) {
        clear();
        alloc(bufSize);
        m_nMaxCapacity = maxCapacity;
    }
    //キューのデータをクリアする
    void clear() {
        if (m_heEvent) {
            CloseEvent(m_heEvent);
            m_heEvent = NULL;
        }
        m_pBufStart.reset();
        m_pBufFin = nullptr;
        m_pBufIn = nullptr;
        m_pBufOut = nullptr;
        m_bUsingData = false;
    }
    //キューのデータをクリアする際に、指定した関数で内部データを開放してから、データ領域を解放する
    template<typename Func>
    void clear(Func deleter) {
        queueData *ptrFin = m_pBufIn;
        for (queueData *ptr = m_pBufOut; ptr < ptrFin; ptr++) {
            deleter(&ptr->data);
        }
        clear();
    }
    //データをキューにコピーし押し込む
    //キューのデータ量があらかじめ設定した上限に達した場合は、キューに空きができるまで待機する
    bool push(const Type& in) {
        //最初に決めた容量分までキューにデータがたまっていたら、キューに空きができるまで待機する
        while (size() >= m_nMaxCapacity) {
            ResetEvent(m_heEvent);
            WaitForSingleObject(m_heEvent, INFINITE);
        }
        if (m_pBufIn >= m_pBufFin) {
            //現時点でのm_pBufOut (この後別スレッドによって書き換わるかもしれない)
            queueData *pBufOutOld = m_pBufOut.load();
            //現在キューにあるデータサイズ
            const size_t dataSize = m_pBufFin - pBufOutOld;
            //新たに確保するバッファのデータサイズ
            const size_t bufSize = std::max((size_t)(m_pBufFin - m_pBufStart.get()), dataSize * 2);
            //新たなバッファ
            auto newBuf = std::unique_ptr<queueData, aligned_malloc_deleter>(
                (queueData *)_aligned_malloc(sizeof(queueData) * bufSize, m_nMallocAlign), aligned_malloc_deleter());
            if (!newBuf) {
                return false;
            }
            memcpy(newBuf.get(), pBufOutOld, sizeof(queueData) * dataSize);
            queueData *pBufOutNew = newBuf.get();
            queueData *pBufOutExpected = pBufOutOld;
            //更新前にnullptrをセット
            m_pBufIn = nullptr;
            //m_pBufOutが変更されていなければ、pBufOutNewをm_pBufOutに代入
            //変更されていれば、pBufOutNewを修正して再度代入
            while (!std::atomic_compare_exchange_weak(&m_pBufOut, &pBufOutExpected, pBufOutNew)) {
                pBufOutNew += (pBufOutExpected - pBufOutOld);
                pBufOutOld = pBufOutExpected;
            }
            //新しいバッファ用にデータを書き換え
            m_pBufIn  = newBuf.get() + dataSize;
            m_pBufFin = newBuf.get() + bufSize;
            //取り出し側のコピー終了を待機
            //一度falseになったことが確認できれば、
            //その次の取り出しは新しいバッファから行われていることになるので、
            //古いバッファは破棄してよい
            while (m_bUsingData.load()) {
                _mm_pause();
            }
            //古いバッファを破棄
            m_pBufStart = std::move(newBuf);
        }
        memcpy(m_pBufIn.load(), &in, sizeof(Type));
        m_pBufIn++;
        return true;
    }
    //キューのsizeを取得する
    size_t size() {
        if (!m_pBufStart)
            return true;
        //バッファはあるが、m_pBufInがnullptrの場合は、
        //押し込み処理で書き換え中なので待機する
        queueData *ptr = nullptr;
        while ((ptr = m_pBufIn.load()) == nullptr) {
            _mm_pause();
        }
        return ptr - m_pBufOut;
    }
    //キューが空ならtrueを返す
    bool empty() {
        return size() == 0;
    }
    //キューの最大サイズを取得する
    size_t capacity() {
        return m_nMaxCapacity;
    }
    //キューの先頭のデータを取り出す (outにコピーする)
    //キューが空ならなにもせずfalseを返す
    bool front_copy_no_lock(Type *out) {
        m_bUsingData++;
        bool bEmpty = empty();
        if (!bEmpty) {
            memcpy(out, m_pBufOut.load(), sizeof(Type));
        }
        m_bUsingData--;
        return !bEmpty;
    }
    //キューの先頭のデータを取り出しながら(outにコピーする)、キューから取り除く
    //キューが空ならなにもせずfalseを返す
    bool front_copy_and_pop_no_lock(Type *out) {
        m_bUsingData++;
        auto nSize = size();
        if (nSize) {
            memcpy(out, m_pBufOut++, sizeof(Type));
            if (nSize <= m_nMaxCapacity - m_nPushRestart) {
                SetEvent(m_heEvent);
            }
        }
        m_bUsingData--;
        return nSize != 0;
    }
    //キューの先頭のデータを取り除く
    //キューが空ならfalseを返す
    bool pop() {
        m_bUsingData++;
        auto nSize = size();
        if (nSize) {
            m_pBufOut++;
            if (nSize <= m_nMaxCapacity - m_nPushRestart) {
                SetEvent(m_heEvent);
            }
        }
        m_bUsingData--;
        return nSize != 0;
    }
protected:
    //bufSize分の内部領域を確保する
    //m_nMaxCapacity以上確保してもかまわない
    //基本的には大きいほうがパフォーマンスは向上する
    void alloc(size_t bufSize) {
        m_pBufStart = std::unique_ptr<queueData, aligned_malloc_deleter>(
            (queueData *)_aligned_malloc(sizeof(queueData) * bufSize, m_nMallocAlign), aligned_malloc_deleter());
        m_pBufFin = m_pBufStart.get() + bufSize;
        m_pBufIn  = m_pBufStart.get();
        m_pBufOut = m_pBufStart.get();
    }

    int m_nPushRestart;
    HANDLE m_heEvent; //キューからデータを取り出したときセットする
    int m_nMallocAlign; //メモリのアライメント
    size_t m_nMaxCapacity; //キューに詰められる有効なデータの最大数
    std::unique_ptr<queueData, aligned_malloc_deleter> m_pBufStart; //確保しているメモリ領域の先頭へのポインタ
    queueData *m_pBufFin; //確保しているメモリ領域の終端
    std::atomic<queueData*> m_pBufIn; //キューにデータを格納する位置へのポインタ
    std::atomic<queueData*> m_pBufOut; //キューから取り出すべき先頭のデータへのポインタ
    std::atomic<int> m_bUsingData; //キューから読み出し中のスレッドの数
};

#endif //_QSV_QUEUE_H_
