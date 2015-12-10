//  -----------------------------------------------------------------------------------------
//    AuoLink by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------

#pragma once
#ifndef __AUO_LINK_H__
#define __AUO_LINK_H__

#include <Windows.h>
#include <stdint.h>

static const int AUO_LINK_HEADER_SIZE = 256;
static const int AUO_LINK_MAX_TRIM = 512;

#define AUO_LINK_HEADER "AuoLink Data v1 #%d"

typedef struct {
    int start, fin;
} AUO_TRIM;

typedef struct {
    BOOL active;
    BOOL use_trim;
    int  trim_count;
    AUO_TRIM trim[AUO_LINK_MAX_TRIM];
} AUO_LINK_PARAM;

typedef struct {
    HANDLE h_mem_map;
    char header[AUO_LINK_HEADER_SIZE];
    AUO_LINK_PARAM prm;
    char input_file[1024];
    char input_file2[1024];
} AUO_LINK_DATA;

static void auo_link_header_name(char name[AUO_LINK_HEADER_SIZE]) {
    sprintf_s(name, AUO_LINK_HEADER_SIZE, AUO_LINK_HEADER, GetCurrentProcessId());
}

static void auo_link_init(AUO_LINK_DATA *link_data) {
    memset(link_data, 0, sizeof(link_data[0]));
    auo_link_header_name(link_data->header);
}

static bool auo_link_check_header(AUO_LINK_DATA *link_data) {
    char test[256] = { 0 };
    auo_link_header_name(link_data->header);
    return 0 == strcmp(link_data->header, test);
}

static AUO_LINK_DATA *auo_link_create() {
    char name[256] = { 0 };
    auo_link_header_name(name);
    HANDLE h_mem_map = CreateFileMapping((HANDLE)0xffffffff, NULL, PAGE_READWRITE, 0, sizeof(AUO_LINK_DATA), name);
    if (h_mem_map == NULL) {
        return nullptr;
    }
    AUO_LINK_DATA *link = (AUO_LINK_DATA *)MapViewOfFile(h_mem_map, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(AUO_LINK_DATA));
    if (link == nullptr) {
        CloseHandle(h_mem_map);
        return nullptr;
    }
    auo_link_header_name(link->header);
    link->h_mem_map = h_mem_map;
    return link;
}

static void auo_link_close(AUO_LINK_DATA **link) {
    HANDLE h_mem_map = NULL;
    if (link != NULL && *link != NULL) {
        h_mem_map = (*link)->h_mem_map;
        UnmapViewOfFile(*link);
    }
    if (h_mem_map != NULL) {
        CloseHandle(h_mem_map);
    }
    if (link) {
        *link = NULL;
    }
}

static HANDLE auo_link_open(AUO_LINK_DATA **link_data) {
    *link_data = NULL;
    char memname[256] = { 0 };
    auo_link_header_name(memname);
    HANDLE memmap = OpenFileMapping(FILE_MAP_READ, FALSE, memname);
    if (NULL == memmap) {
        return nullptr;
    }
    AUO_LINK_DATA *ptr = (AUO_LINK_DATA *)MapViewOfFile(memmap, FILE_MAP_READ, 0, 0, 0);
    if (ptr == nullptr) {
        CloseHandle(memmap);
        return nullptr;
    }
    *link_data = ptr;
    return memmap;
}

static int get_auo_link_data(AUO_LINK_DATA *data) {
    int ret = 1;
    AUO_LINK_DATA *link_data = nullptr;
    HANDLE h_mem_map = NULL;
    memset(data, 0, sizeof(data[0]));
    if (NULL != (h_mem_map = auo_link_open(&link_data))) {
        if (link_data->prm.active) {
            memcpy(data, link_data, sizeof(data[0]));
        }
        auo_link_close(&link_data);
        CloseHandle(h_mem_map);
        ret = 0;
    }
    return ret;
}

#endif //__AUO_LINK_H__
