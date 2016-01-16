//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#if (defined(_WIN32) || defined(_WIN64))
#include <Windows.h>
#include "gpuz_info.h"

//大文字小文字を無視して、1文字検索
static inline const WCHAR *wcsichr(const WCHAR *str, int c) {
    c = tolower(c);
    for (; *str; str++)
        if (c == tolower(*str))
            return str;
    return NULL;
}

//大文字小文字を無視して、文字列を検索
static inline const WCHAR *wcsistr(const WCHAR *str, const WCHAR *substr) {
    size_t len = 0;
    if (substr && (len = wcslen(substr)) != NULL)
        for (; (str = wcsichr(str, substr[0])) != NULL; str++)
            if (_wcsnicmp(str, substr, len) == NULL)
                return str;
    return NULL;
}

int get_gpuz_info(GPUZ_SH_MEM *data) {
    HANDLE memmap = OpenFileMapping(FILE_MAP_READ, FALSE, SHMEM_NAME);
    if (NULL == memmap) {
        return 1;
    }

    GPUZ_SH_MEM *ptr = (GPUZ_SH_MEM *)MapViewOfFile(memmap, FILE_MAP_READ, 0, 0, 0);
    if (ptr == nullptr) {
        CloseHandle(memmap);
        return 1;
    }
    memcpy(data, ptr, sizeof(data[0]));
    UnmapViewOfFile(ptr);
    CloseHandle(memmap);
    return 0;
}

double gpu_core_clock(GPUZ_SH_MEM *data) {
    for (int i = 0; i < MAX_RECORDS; i++) {
        if (   wcsistr(data->sensors[i].name, L"Core")
            && wcsistr(data->sensors[i].name, L"Clock")) {
            return data->sensors[i].value;
        }
    }
    return 0.0;
}

double gpu_load(GPUZ_SH_MEM *data) {
    for (int i = 0; i < MAX_RECORDS; i++) {
        if (wcsistr(data->sensors[i].name, L"GPU Load")) {
            return data->sensors[i].value;
        }
    }
    return 0.0;
}

#endif //#if (defined(_WIN32) || defined(_WIN64))
