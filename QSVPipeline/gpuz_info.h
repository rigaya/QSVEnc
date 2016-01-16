#ifndef __GPUZ_INFO_H__
#define __GPUZ_INFO_H__

#if (defined(_WIN32) || defined(_WIN64))
#include <Windows.h>
#include <tchar.h>

#define SHMEM_NAME _T("GPUZShMem")
#define MAX_RECORDS 128

#pragma pack(push, 1)
struct GPUZ_RECORD {
    WCHAR key[256];
    WCHAR value[256];
};

struct GPUZ_SENSOR_RECORD {
    WCHAR name[256];
    WCHAR unit[8];
    UINT32 digits;
    double value;
};

struct GPUZ_SH_MEM {
    UINT32 version;
    volatile LONG busy;
    UINT32 lastUpdate;
    GPUZ_RECORD data[MAX_RECORDS];
    GPUZ_SENSOR_RECORD sensors[MAX_RECORDS];
};
#pragma pack(pop)

int get_gpuz_info(GPUZ_SH_MEM *data);
double gpu_core_clock(GPUZ_SH_MEM *data);
double gpu_load(GPUZ_SH_MEM *data);

#endif //#if (defined(_WIN32) || defined(_WIN64))

#endif //__GPUZ_INFO_H__
