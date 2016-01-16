//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _QSV_PIPE_H_
#define _QSV_PIPE_H_

#include <cstdint>
#include <cstdio>
#include <vector>
#include "qsv_osdep.h"
#include "qsv_tchar.h"

enum PipeMode {
    PIPE_MODE_DISABLE = 0,
    PIPE_MODE_ENABLE,
    PIPE_MODE_MUXED, //Stderrのモードに使用し、StderrをStdOutに混合する
};

static const int QSV_PIPE_READ_BUF = 2048;

#if defined(_WIN32) || defined(_WIN64)
typedef HANDLE PIPE_HANDLE;
#else
typedef int PIPE_HANDLE;
#endif

typedef struct {
    PIPE_HANDLE h_read;
    PIPE_HANDLE h_write;
    PipeMode mode;
    uint32_t bufferSize;
} PipeSet;

typedef struct {
    PipeSet stdIn;
    PipeSet stdOut;
    PipeSet stdErr;
    FILE *f_stdin;
    uint32_t buf_len;
    char read_buf[QSV_PIPE_READ_BUF];
} ProcessPipe;

class CPipeProcess {
public:
    CPipeProcess() { };
    virtual ~CPipeProcess() { };

    virtual void init() = 0;
    virtual int run(const std::vector<const TCHAR *>& args, const TCHAR *exedir, ProcessPipe *pipes, uint32_t priority, bool hidden, bool minimized) = 0;
    virtual void close() = 0;
private:
    virtual int startPipes(ProcessPipe *pipes) = 0;
};

#if defined(_WIN32) || defined(_WIN64)
class CPipeProcessWin : public CPipeProcess {
public:
    CPipeProcessWin();
    virtual ~CPipeProcessWin();

    virtual void init() override;
    virtual int run(const std::vector<const TCHAR *>& args, const TCHAR *exedir, ProcessPipe *pipes, uint32_t priority, bool hidden, bool minimized) override;
    virtual void close() override;
    const PROCESS_INFORMATION& getProcessInfo();
private:
    virtual int startPipes(ProcessPipe *pipes) override;
    PROCESS_INFORMATION m_pi;
};
#else
class CPipeProcessLinux : public CPipeProcess {
public:
    CPipeProcessLinux();
    virtual ~CPipeProcessLinux();

    virtual void init() override;
    virtual int run(const std::vector<const TCHAR *>& args, const TCHAR *exedir, ProcessPipe *pipes, uint32_t priority, bool hidden, bool minimized) override;
    virtual void close() override;
private:
    virtual int startPipes(ProcessPipe *pipes) override;
};
#endif //#if defined(_WIN32) || defined(_WIN64)

#endif //_QSV_PIPE_H_
