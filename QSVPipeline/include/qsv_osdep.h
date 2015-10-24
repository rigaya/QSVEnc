//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef __QSV_OSDEP_H__
#define __QSV_OSDEP_H__

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <process.h>
#include <io.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
#define QSV_GET_PROC_ADDRESS GetProcAddress

#else //#if defined(_WIN32) || defined(_WIN64)
#include <sys/stat.h>
#include <sys/times.h>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <dlfcn.h>

static inline void *_aligned_malloc(size_t size, size_t alignment) {
    void *p;
    int ret = posix_memalign(&p, alignment, size);
    return (ret == 0) ? p : 0;
}
#define _aligned_free free

typedef wchar_t WCHAR;
typedef int BOOL;
typedef void* HANDLE;
typedef void* HMODULE;
typedef int errno_t;

#define QSV_GET_PROC_ADDRESS dlsym

static uint32_t CP_THREAD_ACP = 0;
static uint32_t CP_UTF8 = 0;

#define __forceinline __attribute__((always_inline))
#define __declspec(noinline) __attribute__ ((noinline))
#define __stdcall
#define __fastcall

template <typename _CountofType, size_t _SizeOfArray>
char (*__countof_helper(_CountofType (&_Array)[_SizeOfArray]))[_SizeOfArray];
#define _countof(_Array) (int)sizeof(*__countof_helper(_Array))

#ifndef TRUE
#define TRUE (1)
#endif

#ifndef FALSE
#define FALSE (0)
#endif

static inline char *strtok_s(char *strToken, const char *strDelimit, char **context) {
    return strtok(strToken, strDelimit);
}
static inline char *strcpy_s(char *dst, size_t size, const char *src) {
    return strcpy(dst, src);
}
static inline char *strcpy_s(char *dst, const char *src) {
    return strcpy(dst, src);
}
static inline int _vsprintf_s(char *buffer, size_t size, const char *format, va_list argptr) {
    return vsprintf(buffer, format, argptr);
}
#define sscanf_s sscanf
#define _strnicmp strncasecmp

static inline void __cpuid(int cpuInfo[4], int param) {
    int eax = 0, ebx = 0, ecx = 0, edx = 0;
     __asm("xor %%ecx, %%ecx\n\t"
           "cpuid" : "=a"(eax), "=b" (ebx), "=c"(ecx), "=d"(edx)
                   : "0"(param));
    cpuInfo[0] = eax;
    cpuInfo[1] = ebx;
    cpuInfo[2] = ecx;
    cpuInfo[3] = edx;
}

static inline unsigned long long _xgetbv(unsigned int index){
  unsigned int eax, edx;
  __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
  return ((unsigned long long)edx << 32) | eax;
}

//uint64_t __rdtsc() {
//    unsigned int eax, edx;
//    __asm__ volatile("rdtsc" : "=a"(eax), "=d"(edx));
//    return ((uint64_t)edx << 32) | eax;
//}

static inline int _vscprintf(const char * format, va_list pargs) {
    int retval;
    va_list argcopy;
    va_copy(argcopy, pargs);
    retval = vsnprintf(NULL, 0, format, argcopy);
    va_end(argcopy);
    return retval;
 }

static inline int sprintf_s(char *dst, const char* format, ...) {
    va_list args;
    va_start(args, format);
    int ret = vsprintf(dst, format, args);
    va_end(args);
    return ret;
}
static inline int sprintf_s(char *dst, size_t size, const char* format, ...) {
    va_list args;
    va_start(args, format);
    int ret = vsprintf(dst, format, args);
    va_end(args);
    return ret;
}

static inline char *_fullpath(char *dst, const char *path, size_t size) {
    return realpath(path, dst);
}

static inline const char *PathFindExtension(const char *path) {
    return strrchr(basename(path), '.');
}

static inline const char *PathFindFileName(const char *path) {
    return basename(path);
}

static inline int PathFileExists(const char *path) {
    struct stat st;
    return 0 == stat(path, &st);
}

static inline int PathIsUNC(const char *path) {
    return 0;
}

static inline int fopen_s(FILE **pfp, const char *filename, const char *mode) {
    FILE *fp = fopen(filename, mode);
    *pfp = fp;
    return (fp == NULL) ? 1 : 0;
}

static pthread_t GetCurrentThread() {
    return pthread_self();
}

static void SetThreadAffinityMask(pthread_t thread, size_t mask) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (uint32_t j = 0; j < sizeof(mask) * 8; j++) {
        if (mask & (1 << j)) {
            CPU_SET(j, &cpuset);
        }
    }
    pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
}

enum {
    THREAD_PRIORITY_NORMAL,
    THREAD_PRIORITY_HIGHEST,
    THREAD_PRIORITY_ABOVE_NORMAL,
    THREAD_PRIORITY_BELOW_NORMAL,
    THREAD_PRIORITY_LOWEST,
    THREAD_PRIORITY_IDLE,
};

static void SetThreadPriority(pthread_t thread, int priority) {
    return; //何もしない
}

#define _fseeki64 fseek
#define _ftelli64 ftell

#endif //#if defined(_WIN32) || defined(_WIN64)

#endif //__QSV_OSDEP_H__
