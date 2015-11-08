//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#include <fcntl.h>
#include <io.h>
#endif //#if defined(_WIN32) || defined(_WIN64)
#include <cstring>
#include "qsv_pipe.h"

CPipeProcess::CPipeProcess() {
#if defined(_WIN32) || defined(_WIN64)
    memset(&m_pi, 0, sizeof(m_pi));
#endif //#if defined(_WIN32) || defined(_WIN64)
}

CPipeProcess::~CPipeProcess() {

}

void CPipeProcess::init() {
    close();
}


int CPipeProcess::startPipes(ProcessPipe *pipes) {
#if defined(_WIN32) || defined(_WIN64)
    SECURITY_ATTRIBUTES sa ={ sizeof(SECURITY_ATTRIBUTES), NULL, TRUE };
    if (pipes->stdOut.mode) {
        if (!CreatePipe(&pipes->stdOut.h_read, &pipes->stdOut.h_write, &sa, pipes->stdOut.bufferSize) ||
            !SetHandleInformation(pipes->stdOut.h_read, HANDLE_FLAG_INHERIT, 0))
            return 1;
    }
    if (pipes->stdErr.mode) {
        if (!CreatePipe(&pipes->stdErr.h_read, &pipes->stdErr.h_write, &sa, pipes->stdErr.bufferSize) ||
            !SetHandleInformation(pipes->stdErr.h_read, HANDLE_FLAG_INHERIT, 0))
            return 1;
    }
    if (pipes->stdIn.mode) {
        if (!CreatePipe(&pipes->stdIn.h_read, &pipes->stdIn.h_write, &sa, pipes->stdIn.bufferSize) ||
            !SetHandleInformation(pipes->stdIn.h_write, HANDLE_FLAG_INHERIT, 0))
            return 1;
        if ((pipes->f_stdin = _fdopen(_open_osfhandle((intptr_t)pipes->stdIn.h_write, _O_BINARY), "wb")) == NULL) {
            return 1;
        }
    }
#endif //#if defined(_WIN32) || defined(_WIN64)
    return 0;
}

int CPipeProcess::run(const TCHAR *args, const TCHAR *exedir, ProcessPipe *pipes, uint32_t priority, bool hidden, bool minimized) {
#if defined(_WIN32) || defined(_WIN64)
    BOOL Inherit = FALSE;
    DWORD flag = priority;
    STARTUPINFO si;
    memset(&si, 0, sizeof(STARTUPINFO));
    memset(&m_pi, 0, sizeof(PROCESS_INFORMATION));
    si.cb = sizeof(STARTUPINFO);

    startPipes(pipes);

    if (pipes->stdOut.mode)
        si.hStdOutput = pipes->stdOut.h_write;
    if (pipes->stdErr.mode)
        si.hStdError = (pipes->stdErr.mode == PIPE_MODE_MUXED) ? pipes->stdOut.h_write : pipes->stdErr.h_write;
    if (pipes->stdIn.mode)
        si.hStdInput = pipes->stdIn.h_read;
    si.dwFlags |= STARTF_USESTDHANDLES;
    Inherit = TRUE;
    //flag |= DETACHED_PROCESS; //このフラグによるコンソール抑制よりCREATE_NO_WINDOWの抑制を使用する
    if (minimized) {
        si.dwFlags |= STARTF_USESHOWWINDOW;
        si.wShowWindow |= SW_SHOWMINNOACTIVE;
    }
    if (hidden)
        flag |= CREATE_NO_WINDOW;

    int ret = (CreateProcess(NULL, (TCHAR *)args, NULL, NULL, Inherit, flag, NULL, exedir, &si, &m_pi)) ? 0 : 1;

    if (pipes->stdOut.mode) {
        CloseHandle(pipes->stdOut.h_write);
        if (ret) {
            CloseHandle(pipes->stdOut.h_read);
            pipes->stdOut.mode = PIPE_MODE_DISABLE;
        }
    }
    if (pipes->stdErr.mode) {
        if (pipes->stdErr.mode)
            CloseHandle(pipes->stdErr.h_write);
        if (ret) {
            CloseHandle(pipes->stdErr.h_read);
            pipes->stdErr.mode = PIPE_MODE_DISABLE;
        }
    }
    if (pipes->stdIn.mode) {
        CloseHandle(pipes->stdIn.h_read);
        if (ret) {
            CloseHandle(pipes->stdIn.h_write);
            pipes->stdIn.mode = PIPE_MODE_DISABLE;
        }
    }
    return ret;
#else
    return 0;
#endif //#if defined(_WIN32) || defined(_WIN64)
}

#if defined(_WIN32) || defined(_WIN64)
const PROCESS_INFORMATION& CPipeProcess::getProcessInfo() {
    return m_pi;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

void CPipeProcess::close() {
#if defined(_WIN32) || defined(_WIN64)
    if (m_pi.hProcess) {
        CloseHandle(m_pi.hProcess);
    }
    if (m_pi.hThread) {
        CloseHandle(m_pi.hThread);
    }
    memset(&m_pi, 0, sizeof(m_pi));
#endif //#if defined(_WIN32) || defined(_WIN64)
}
