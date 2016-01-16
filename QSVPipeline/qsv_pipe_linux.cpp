//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#if !(defined(_WIN32) || defined(_WIN64))
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include "qsv_pipe.h"

CPipeProcessLinux::CPipeProcessLinux() {
}

CPipeProcessLinux::~CPipeProcessLinux() {

}

void CPipeProcessLinux::init() {
    close();
}


int CPipeProcessLinux::startPipes(ProcessPipe *pipes) {
    if (pipes->stdOut.mode) {
        if (-1 == (pipe((int *)&pipes->stdOut.h_read)))
            return 1;
    }
    if (pipes->stdErr.mode) {
        if (-1 == (pipe((int *)&pipes->stdErr.h_read)))
            return 1;
    }
    if (pipes->stdIn.mode) {
        if (-1 == (pipe((int *)&pipes->stdIn.h_read)))
            return 1;
        pipes->f_stdin = fdopen(pipes->stdIn.h_write, "w");
    }
    return 0;
}

int CPipeProcessLinux::run(const std::vector<const TCHAR *>& args, const TCHAR *exedir, ProcessPipe *pipes, uint32_t priority, bool hidden, bool minimized) {
    pid_t cpid = fork();
    if (cpid == -1) {
        return 1;
    }

    if (cpid == 0) {
        if (pipes->stdIn.mode) {
            ::close(pipes->stdIn.h_write);
            dup2(pipes->stdIn.h_read, STDIN_FILENO);
        }

        execvp(args[0], (char *const *)&args[0]);
        exit(0);
    }
    if (pipes->stdIn.mode) {
        ::close(pipes->stdIn.h_read);
    }
    if (pipes->stdOut.mode) {
        ::close(pipes->stdOut.h_write);
    }
    if (pipes->stdErr.mode) {
        ::close(pipes->stdErr.h_write);
    }
    return 0;
}

void CPipeProcessLinux::close() {
}
#endif //#if !(defined(_WIN32) || defined(_WIN64))
