// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
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

#include "rgy_pipe_named.h"
#include "rgy_osdep.h"
#include <fcntl.h>

RGYNamedPipe::RGYNamedPipe() :
    m_handle(NULL),
    m_fp(nullptr),
    m_event(unique_event(nullptr, nullptr)),
    m_name(),
    m_connected(false),
    m_overlapped(false) {
}

RGYNamedPipe::~RGYNamedPipe() {
    close();
}

int RGYNamedPipe::init(const tstring& pipeName, const bool overlapped) {
    m_name = pipeName;
    m_overlapped = overlapped;
    auto flags = PIPE_ACCESS_OUTBOUND | PIPE_ACCESS_INBOUND;
    if (m_overlapped) {
        flags |= FILE_FLAG_OVERLAPPED;
        m_event = CreateEventUnique(nullptr, FALSE, FALSE);
    }
    m_handle = CreateNamedPipe(m_name.c_str(), flags, PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1, 4096, 4096, 0, NULL);
    if (!m_handle) {
        return 1;
    }
    return 0;
}

int RGYNamedPipe::connect(DWORD timeout) {
    if (m_connected) {
        return 0;
    }
    OVERLAPPED overlapped;
    memset(&overlapped, 0, sizeof(overlapped));
    overlapped.hEvent = m_event.get();
    if (ConnectNamedPipe(m_handle, (m_overlapped) ? &overlapped : nullptr) == 0) {
        const DWORD errorcode = GetLastError();
        if (errorcode == ERROR_PIPE_CONNECTED) {
            m_connected = true;
        } else if (m_overlapped && errorcode == ERROR_IO_PENDING) {
            // 非同期 I/O 操作が完了するのを待つ
            auto ret = WaitForSingleObject(overlapped.hEvent, timeout);
            if (ret == WAIT_OBJECT_0) {
                m_connected = true;
                return 0;
            } else {
                return 1;
            }
        } else {
            LPVOID lpMsgBuf;
            FormatMessage(
                FORMAT_MESSAGE_ALLOCATE_BUFFER  //      テキストのメモリ割り当てを要求する
                | FORMAT_MESSAGE_FROM_SYSTEM    //      エラーメッセージはWindowsが用意しているものを使用
                | FORMAT_MESSAGE_IGNORE_INSERTS,//      次の引数を無視してエラーコードに対するエラーメッセージを作成する
                NULL, errorcode, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),//   言語を指定
                (LPTSTR)&lpMsgBuf,                          //      メッセージテキストが保存されるバッファへのポインタ
                0,
                NULL);

            _ftprintf(stderr, _T("ConnectNamedPipe failed: %s\n"), (LPCTSTR)lpMsgBuf);
            LocalFree(lpMsgBuf);
            return 1;
        }
    }
    if (m_overlapped) {
        auto ret = WaitForSingleObject(overlapped.hEvent, timeout);
        return ret == WAIT_OBJECT_0 ? 0 : 1;
    }
    return 0;
}

FILE *RGYNamedPipe::fp(const char *mode) {
    if (!m_connected) {
        return nullptr;
    }
    if (m_fp) {
        return m_fp;
    }
    m_fp = _fdopen(_open_osfhandle((intptr_t)m_handle, _O_BINARY), mode);
    return m_fp;
}

int RGYNamedPipe::write(const void *data, const size_t size, size_t *writeSize) {
    if (!m_connected) {
        return 1;
    }
    OVERLAPPED overlapped;
    memset(&overlapped, 0, sizeof(overlapped));
    overlapped.hEvent = m_event.get();
    DWORD sizeWritten = 0;
    //非同期処理中は0を返すことがある
    auto ret = WriteFile(m_handle, data, (DWORD)size, &sizeWritten, (m_overlapped) ? &overlapped : nullptr) == 0 ? 1 : 0;
    if (m_overlapped) {
        ret = (WaitForSingleObject(overlapped.hEvent, INFINITE) != WAIT_OBJECT_0) ? 1 : 0;
    }
    *writeSize = sizeWritten;
    return ret;
}

int RGYNamedPipe::read(void *data, const size_t size, size_t *readSize) {
    if (!m_connected) {
        return 1;
    }
    OVERLAPPED overlapped;
    memset(&overlapped, 0, sizeof(overlapped));
    overlapped.hEvent = m_event.get();
    DWORD sizeRead = 0;
    //非同期処理中は0を返すことがある
    auto ret = ReadFile(m_handle, data, (DWORD)size, &sizeRead, (m_overlapped) ? &overlapped : nullptr) == 0 ? 1 : 0;
    if (m_overlapped) {
        ret = (WaitForSingleObject(overlapped.hEvent, INFINITE) != WAIT_OBJECT_0) ? 1 : 0;
    }
    *readSize = sizeRead;
    return ret;
}

int RGYNamedPipe::disconnect() {
    if (!m_connected) {
        return 0;
    }
    if (FlushFileBuffers(m_handle) == 0) {
        return 1;
    }
    if (DisconnectNamedPipe(m_handle) == 0) {
        return 1;
    }
    m_connected = false;
    return 0;
}

int RGYNamedPipe::close() {
    if (m_fp) {
        DisconnectNamedPipe(m_handle);
        fclose(m_fp);
        m_fp = nullptr;
        m_handle = NULL;
    } else if (m_handle) {
        DisconnectNamedPipe(m_handle);
        CloseHandle(m_handle);
        m_handle = NULL;
    }
    return 0;
}
