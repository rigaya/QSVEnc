//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <Windows.h>
#include <tchar.h>
#include <process.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <atomic>
#include <intrin.h>
#include <emmintrin.h> //SSE2
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSE4.2
#include "scene_change_detection.h"

const int SC_SKIP = 2;
const int MAX_SUB_THREADS = 7;

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

unsigned __stdcall func_make_hist(void *p) {
    hist_thread_t *htt = (hist_thread_t *)p;
    const int thread_id = htt->id;
    CSceneChangeDetect *scd = (CSceneChangeDetect *)htt->ptr_csd;
    WaitForSingleObject(htt->he_start, INFINITE);
    while (!htt->abort) {
        scd->MakeHist(thread_id, scd->GetSubThreadNum() + 1, &htt->hist_thread);
        SetEvent(htt->he_fin);
        WaitForSingleObject(htt->he_start, INFINITE);
    }
    _endthreadex(0);
    return 0;
}

typedef BOOL (WINAPI *LPFN_GLPI)(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION, PDWORD);

static DWORD CountSetBits(ULONG_PTR bitMask) {
    DWORD LSHIFT = sizeof(ULONG_PTR)*8 - 1;
    DWORD bitSetCount = 0;
    for (ULONG_PTR bitTest = (ULONG_PTR)1 << LSHIFT; bitTest; bitTest >>= 1)
        bitSetCount += ((bitMask & bitTest) != 0);

    return bitSetCount;
}

static BOOL getProcessorCount(DWORD *physical_processor_core, DWORD *logical_processor_core) {
    *physical_processor_core = 0;
    *logical_processor_core = 0;

    LPFN_GLPI glpi = (LPFN_GLPI)GetProcAddress(GetModuleHandle(_T("kernel32")), "GetLogicalProcessorInformation");
    if (NULL == glpi)
        return FALSE;

    DWORD returnLength = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = NULL;
    while (FALSE == glpi(buffer, &returnLength)) {
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
            if (buffer) 
                free(buffer);
            if (NULL == (buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(returnLength)))
                return FALSE;
        }
    }

    DWORD logicalProcessorCount = 0;
    DWORD numaNodeCount = 0;
    DWORD processorCoreCount = 0;
    DWORD processorL1CacheCount = 0;
    DWORD processorL2CacheCount = 0;
    DWORD processorL3CacheCount = 0;
    DWORD processorPackageCount = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = buffer;
    for (DWORD byteOffset = 0; byteOffset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= returnLength;
        byteOffset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)) {
        switch (ptr->Relationship) {
        case RelationNumaNode:
            // Non-NUMA systems report a single record of this type.
            numaNodeCount++;
            break;
        case RelationProcessorCore:
            processorCoreCount++;
            // A hyperthreaded core supplies more than one logical processor.
            logicalProcessorCount += CountSetBits(ptr->ProcessorMask);
            break;

        case RelationCache:
            {
            // Cache data is in ptr->Cache, one CACHE_DESCRIPTOR structure for each cache. 
            PCACHE_DESCRIPTOR Cache = &ptr->Cache;
            processorL1CacheCount += (Cache->Level == 1);
            processorL2CacheCount += (Cache->Level == 2);
            processorL3CacheCount += (Cache->Level == 3);
            break;
            }
        case RelationProcessorPackage:
            // Logical processors share a physical package.
            processorPackageCount++;
            break;

        default:
            //Unsupported LOGICAL_PROCESSOR_RELATIONSHIP value.
            break;
        }
        ptr++;
    }

    *physical_processor_core = processorCoreCount;
    *logical_processor_core = logicalProcessorCount;

    return TRUE;
}

void CSceneChangeDetect::thread_close() {
    for (int i_th = 0; i_th < sub_thread_num; i_th++) {
        if (th_hist && th_hist[i_th].hnd) {
            th_hist[i_th].abort++; //th_hist[i_th].abort = TRUE
            SetEvent(th_hist[i_th].he_start);
            WaitForSingleObject(th_hist[i_th].hnd, INFINITE);
            if (th_hist[i_th].he_start)
                CloseHandle(th_hist[i_th].he_start);
            if (th_hist[i_th].he_fin)
                CloseHandle(th_hist[i_th].he_fin);
            if (th_hist[i_th].hnd)
                CloseHandle(th_hist[i_th].hnd);
        }
    }
    if (sub_thread_num) {
        if (th_hist)          ZeroMemory(th_hist, sizeof(hist_thread_t) * sub_thread_num);
        if (he_hist_fin_copy) ZeroMemory(he_hist_fin_copy, sizeof(HANDLE *) * sub_thread_num);
    }
}

int CSceneChangeDetect::thread_start() {
    if (!sub_thread_num)
        return 0;
    if (   NULL == th_hist
        || NULL == he_hist_fin_copy)
        return 1;
    ZeroMemory(th_hist, sizeof(hist_thread_t) * sub_thread_num);
    ZeroMemory(he_hist_fin_copy, sizeof(HANDLE *) * sub_thread_num);
    for (int i_th = 0; i_th < sub_thread_num; i_th++) {
        th_hist[i_th].id = i_th;
        th_hist[i_th].ptr_csd = this;
        if (   NULL == (th_hist[i_th].he_start = CreateEvent(NULL, FALSE, FALSE, NULL))
            || NULL == (th_hist[i_th].he_fin   = CreateEvent(NULL, FALSE, FALSE, NULL))
            || NULL == (th_hist[i_th].hnd = (HANDLE)_beginthreadex(NULL, 0, func_make_hist, &th_hist[i_th], FALSE, NULL))) {
            thread_close();
            return 1;
        }
        he_hist_fin_copy[i_th] = th_hist[i_th].he_fin;
    }
    return 0;
}

CSceneChangeDetect::CSceneChangeDetect() {
#if SC_DEBUG
    fp_sc_log = NULL;
#endif
    initialized = false;
    current_gop_len = 0;
    gop_len_min = 0;
    gop_len_max = USHRT_MAX;
    if (NULL != (hist = (hist_t*)_aligned_malloc(sizeof(hist_t) * HIST_COUNT, 64)))
        ZeroMemory(hist, sizeof(hist_t) * HIST_COUNT);
    //スレッド関連
    DWORD physical_cores = 0, logical_cores = 0;
    if (!getProcessorCount(&physical_cores, &logical_cores) || physical_cores == 0)
        physical_cores = 1;
#if _DEBUG
    sub_thread_num = 0;
#else
    sub_thread_num = max(0, min(MAX_SUB_THREADS, (physical_cores>>1)-1)); //自分も仕事をしてる分、1引いとく
#endif
    if (sub_thread_num) {
        if (NULL != (th_hist = (hist_thread_t *)_aligned_malloc(sizeof(hist_thread_t) * sub_thread_num, 64)))
            ZeroMemory(th_hist, sizeof(hist_thread_t) * sub_thread_num);
        he_hist_fin_copy = (HANDLE *)calloc(sub_thread_num, sizeof(HANDLE *));
    } else {
        he_hist_fin_copy = NULL;
        th_hist = NULL;
    }
}

CSceneChangeDetect::~CSceneChangeDetect() {
    thread_close();
    if (he_hist_fin_copy)
        free(he_hist_fin_copy);
    if (hist)
        _aligned_free(hist);
    if (th_hist)
        _aligned_free(th_hist);
    he_hist_fin_copy = NULL;
    hist = NULL;
    th_hist = NULL;
    sub_thread_num = 0;
    current_gop_len = 0;
}

int CSceneChangeDetect::Init(int _threshold, mfxU32 _pic_struct, mfxU16 _vqp_strength, mfxU16 _vqp_sensitivity, mfxU16 _gop_len_min, mfxU16 _gop_len_max, bool _deint_normal) {    
    int ret = 0;
    if (hist
#if SC_DEBUG
        && (0 == fopen_s(&fp_sc_log, "qsv_log.csv", "ab") && fp_sc_log)
#endif
        && 0 == thread_start()) {
        mask_histgram = get_make_hist_func();
        deint_normal = _deint_normal;
        initialized = true;
        index = 0;
        vqp_strength = clamp(_vqp_strength, 0, 51);
        vqp_sensitivity = clamp(_vqp_sensitivity, 0, 100);
        prev_fade = false;
        threshold = _threshold;
        current_gop_len = 0;
        gop_len_min = max(_gop_len_min, 1);
        gop_len_max = min(((deint_normal) ? _gop_len_max * 2 : _gop_len_max), (USHRT_MAX));
        pic_struct = (_pic_struct) ? _pic_struct : MFX_PICSTRUCT_PROGRESSIVE;
    } else {
        ret = 1;
        thread_close();
    }
    return ret;
}

void CSceneChangeDetect::MakeHist(int thread_id, int thread_max, hist_t *hist_buf) {
    const int height = target_frame->Info.CropH;
    const int width = target_frame->Info.CropW;
    const int pitch = target_frame->Data.Pitch;
    const mfxU8 *const frame_Y = target_frame->Data.Y;

    const int interlaced = (0 != (pic_struct & (MFX_PICSTRUCT_FIELD_TFF | MFX_PICSTRUCT_FIELD_BFF)));
    const int x_skip = (min(32, width / 8)) & ~31; //32の倍数であること
    const int y_skip = (min(32, height / 8)) & ~1; //2の倍数であること
    const int y_step = ((1 << SC_SKIP) + interlaced) & (~interlaced);
    const int y_offset = y_skip + (i_field + (pic_struct == MFX_PICSTRUCT_FIELD_BFF)) & 0x01;
    const int y_start = ((((height - y_skip * 2) * (thread_id + 0)) / thread_max) & (~((1<<(1+SC_SKIP))-1))) + y_offset;
    const int y_end   = ((((height - y_skip * 2) * (thread_id + 1)) / thread_max) & (~((1<<(1+SC_SKIP))-1))) + y_offset;

    mask_histgram(frame_Y, hist_buf, y_start, y_end, y_step, x_skip, width, pitch);
}

static inline void inverse(float result[3][3], float a[3][3]) {
    const float detinv = 1.0f
        / (  
           a[0][0] * a[1][1] * a[2][2]
         + a[0][1] * a[1][2] * a[2][0]
         + a[0][2] * a[1][0] * a[2][1]
         - a[0][0] * a[1][2] * a[2][1]
         - a[0][1] * a[1][0] * a[2][2]
         - a[0][2] * a[1][1] * a[2][0]);
    result[0][0] = detinv * (   a[1][1]*a[2][2] - a[1][2]*a[2][1]);
    result[0][1] = detinv * ( -(a[0][1]*a[2][2] - a[0][2]*a[2][1]));
    result[0][2] = detinv * (   a[0][1]*a[1][2] - a[0][2]*a[1][1]);
    result[1][0] = detinv * ( -(a[1][0]*a[2][2] - a[1][2]*a[2][0]));
    result[1][1] = detinv * (   a[0][0]*a[2][2] - a[0][2]*a[2][0]);
    result[1][2] = detinv * ( -(a[0][0]*a[1][2] - a[0][2]*a[1][0]));
    result[2][0] = detinv * (   a[1][0]*a[2][1] - a[1][1]*a[2][0]);
    result[2][1] = detinv * ( -(a[0][0]*a[2][1] - a[0][1]*a[2][0]));
    result[2][2] = detinv * (   a[0][0]*a[1][1] - a[0][1]*a[1][0]);
}

static inline void inverse(float result[2][2], float a[2][2]) {
    const float detinv = 1.0f / (a[0][0] * a[1][1] - a[0][1] * a[1][0]);
    result[0][0] = detinv * ( a[1][1]);
    result[0][1] = detinv * (-a[1][0]);
    result[1][0] = detinv * (-a[0][1]);
    result[1][1] = detinv * ( a[0][0]);
}

static inline float pow2(float a) {
    return a * a;
}

static float estimate_next_2(int *data, int num) {
    int sum[5] = { 0 };
    int Y[3] = { 0 };
    for (int i = 1; i <= num; i++) {
        sum[0] += 1;
        sum[1] += i;
        sum[2] += i * i;
        sum[3] += i * i * i;
        sum[4] += i * i * i * i;
        Y[0] += data[i-1] * i * i;
        Y[1] += data[i-1] * i;
        Y[2] += data[i-1];
    }

    float X[3][3] = {
        { (float)sum[4], (float)sum[3], (float)sum[2] },
        { (float)sum[3], (float)sum[2], (float)sum[1] },
        { (float)sum[2], (float)sum[1], (float)sum[0] }
    };

    float inv_X[3][3];
    inverse(inv_X, X);

    float A[3];
    A[0] = inv_X[0][0] * Y[0] + inv_X[0][1] * Y[1] + inv_X[0][2] * Y[2];
    A[1] = inv_X[1][0] * Y[0] + inv_X[1][1] * Y[1] + inv_X[1][2] * Y[2];
    A[2] = inv_X[2][0] * Y[0] + inv_X[2][1] * Y[1] + inv_X[2][2] * Y[2];
    
    return A[0] * (num+1) * (num+1) + A[1] * (num+1) + A[2];
}

static float estimate_next_1(float *data, int num) {
    int sum[3] = { 0 };
    float Y[2] = { 0 };
    for (int i = 1; i <= num; i++) {
        sum[0] += 1;
        sum[1] += i;
        sum[2] += i * i;
        Y[0] += data[i-1] * i;
        Y[1] += data[i-1];
    }

    float X[2][2] = {
        { (float)sum[2], (float)sum[1] },
        { (float)sum[1], (float)sum[0] }
    };

    float inv_X[2][2];
    inverse(inv_X, X);

    float A[2];
    A[0] = inv_X[0][0] * Y[0] + inv_X[0][1] * Y[1];
    A[1] = inv_X[1][0] * Y[0] + inv_X[1][1] * Y[1];
    
    return A[0] * (num+1) + A[1];
}

mfxU16 CSceneChangeDetect::Check(mfxFrameSurface1 *frame, int *qp_offset) {
    target_frame = frame;

    *qp_offset = 0;

    const mfxU32 KEY_FRAMETYPE[2] = {
        MFX_FRAMETYPE_I  | MFX_FRAMETYPE_REF, //プログレッシブ と 第1フィールド用
        MFX_FRAMETYPE_xI | MFX_FRAMETYPE_xREF //第2フィールド用
    };
    mfxU16 result = 0;

    
    for (i_field = 0; i_field < 2; i_field += (1 + (pic_struct == MFX_PICSTRUCT_PROGRESSIVE))) {

        prev_max_match_point[index]  = 100.0f;
        prev_fade_match_point[index] = 100.0f;

        for (int i_th = 0; i_th < sub_thread_num; i_th++)
            SetEvent(th_hist[i_th].he_start);
        MakeHist(sub_thread_num, sub_thread_num+1, &hist[index]);
        WaitForMultipleObjects(sub_thread_num, he_hist_fin_copy, TRUE, INFINITE);

        __m128i x0 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] +  0));
        __m128i x1 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] + 16));
        __m128i x2 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] + 32));
        __m128i x3 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] + 48));

        for (int i_th = 0; i_th < sub_thread_num; i_th++) {
            x0 = _mm_add_epi32(x0, _mm_load_si128((__m128i*)((BYTE *)&th_hist[i_th].hist_thread +  0)));
            x1 = _mm_add_epi32(x1, _mm_load_si128((__m128i*)((BYTE *)&th_hist[i_th].hist_thread + 16)));
            x2 = _mm_add_epi32(x2, _mm_load_si128((__m128i*)((BYTE *)&th_hist[i_th].hist_thread + 32)));
            x3 = _mm_add_epi32(x3, _mm_load_si128((__m128i*)((BYTE *)&th_hist[i_th].hist_thread + 48)));
        }

        _mm_store_si128((__m128i*)((BYTE *)&hist[index] +  0), x0);
        _mm_store_si128((__m128i*)((BYTE *)&hist[index] + 16), x1);
        _mm_store_si128((__m128i*)((BYTE *)&hist[index] + 32), x2);
        _mm_store_si128((__m128i*)((BYTE *)&hist[index] + 48), x3);

        __m128i x4 = _mm_add_epi32(x0, x2);
        __m128i x5 = _mm_add_epi32(x1, x3);
        x4= _mm_add_epi32(x4, x5);
        const int frame_size = x4.m128i_i32[0] + x4.m128i_i32[1] + x4.m128i_i32[2] + x4.m128i_i32[3];

        alignas(16) static const int MUL_ARRAY[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };

        x0 = _mm_mullo_epi32(x0, _mm_load_si128((__m128i*)((BYTE *)&MUL_ARRAY +  0)));
        x1 = _mm_mullo_epi32(x1, _mm_load_si128((__m128i*)((BYTE *)&MUL_ARRAY + 16)));
        x2 = _mm_mullo_epi32(x2, _mm_load_si128((__m128i*)((BYTE *)&MUL_ARRAY + 32)));
        x3 = _mm_mullo_epi32(x3, _mm_load_si128((__m128i*)((BYTE *)&MUL_ARRAY + 48)));

        x0 = _mm_add_epi32(x0, x2);
        x1 = _mm_add_epi32(x1, x3);
        x0 = _mm_add_epi32(x0, x1);

        avg_luma[index] = (x0.m128i_i32[0] + x0.m128i_i32[1] + x0.m128i_i32[2] + x0.m128i_i32[3]) / (float)frame_size;

#if SC_DEBUG
        fprintf(fp_sc_log, "%3d,", index);
        for (int i_hist = 0; i_hist < HIST_LEN; i_hist++)
            fprintf(fp_sc_log, "%4d,", hist[index].v[i_hist]);
        fprintf(fp_sc_log, ",");
#endif
        mfxU32 flag = 0;
        float simple_match_point = 100.0f;

        if (index >= 1) {

            x0 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] +  0));
            x1 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] + 16));
            x2 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] + 32));
            x3 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] + 48));

            x0 = _mm_min_epi32(x0, _mm_load_si128((__m128i*)((BYTE *)&hist[index-1] +  0)));
            x1 = _mm_min_epi32(x1, _mm_load_si128((__m128i*)((BYTE *)&hist[index-1] + 16)));
            x2 = _mm_min_epi32(x2, _mm_load_si128((__m128i*)((BYTE *)&hist[index-1] + 32)));
            x3 = _mm_min_epi32(x3, _mm_load_si128((__m128i*)((BYTE *)&hist[index-1] + 48)));

            x0 = _mm_add_epi32(x0, x2);
            x1 = _mm_add_epi32(x1, x3);

            x0 = _mm_add_epi32(x0, x1);

            const int count = x0.m128i_i32[0] + x0.m128i_i32[1] + x0.m128i_i32[2] + x0.m128i_i32[3];

            simple_match_point = count * 100 / (float)frame_size;

            flag = KEY_FRAMETYPE[i_field] * (simple_match_point < threshold);

        }

#if SC_DEBUG
        fprintf(fp_sc_log, "%f,,", simple_match_point);
#endif

        if (index >= HIST_COUNT - 1) {

            alignas(16) hist_t estimate = { 0 };
            alignas(16) hist_t fade_estimate = { 0 };
            struct alignas(16) histf_t {
                float f[HIST_LEN];
            } sigma = { 0 };

            //////////////////     estimate 1      //////////////////////////////////
            
            for (int i_hist = 0; i_hist < HIST_LEN; i_hist++) {
                int data_array[4];
                int sum = 0;
                for (int idx = 0; idx < index; idx++) {
                    data_array[idx] = hist[idx].v[i_hist];
                    sum += data_array[idx];
                }
                const float avg = sum / (float)index;
                float tmp = 0.0f;
                for (int idx = 0; idx < index; idx++) {
                    tmp += pow2(data_array[idx] - avg);
                }
                sigma.f[i_hist] = (sum) ? ((float)sqrt(tmp / (float)index) / (float)(frame_size >> HIST_LEN_2N)) : 0.0f;
                estimate.v[i_hist] = max(0, (int)(estimate_next_2(data_array, index) + 0.5f));
            }

            x0 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] +  0));
            x1 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] + 16));
            x2 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] + 32));
            x3 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] + 48));

            x0 = _mm_min_epi32(x0, _mm_load_si128((__m128i*)((BYTE *)&estimate +  0)));
            x1 = _mm_min_epi32(x1, _mm_load_si128((__m128i*)((BYTE *)&estimate + 16)));
            x2 = _mm_min_epi32(x2, _mm_load_si128((__m128i*)((BYTE *)&estimate + 32)));
            x3 = _mm_min_epi32(x3, _mm_load_si128((__m128i*)((BYTE *)&estimate + 48)));

            x0 = _mm_add_epi32(x0, x2);
            x1 = _mm_add_epi32(x1, x3);

            x0 = _mm_add_epi32(x0, x1);

            const int est_count = x0.m128i_i32[0] + x0.m128i_i32[1] + x0.m128i_i32[2] + x0.m128i_i32[3];

            const float esitimate_match_point = est_count * 100 / (float)frame_size;

            //////////////////     estimate 2     //////////////////////////////////

            float fade_esitimate_match_point = 0.0f;
            
            for (int i_mode = 0; i_mode < 2; i_mode++) {
                const float avg_luma_diff_per_step = avg_luma[index] - avg_luma[index-1];
                for (int i_hist = 0; i_hist < HIST_LEN; i_hist++) {
                    const int edge = 1 + 13 * (avg_luma_diff_per_step > 0);
                    float f_next;
                    if (i_mode == 0) {
                        f_next = i_hist + avg_luma_diff_per_step * (edge - i_hist) / ((float)edge - avg_luma[index-1]);
                    } else {
                        f_next = estimate_next_1(avg_luma, index);
                    }
                    const int i_next = (int)f_next;
                    fade_estimate.v[clamp(i_next + 0, 0, HIST_LEN-1)] += (int)(hist[index-1].v[i_hist] * ((float)(i_next + 1) - f_next) + 0.5f);
                    fade_estimate.v[clamp(i_next + 1, 0, HIST_LEN-1)] += (int)(hist[index-1].v[i_hist] * (f_next - (float)i_next)       + 0.5f);
                }

                x0 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] +  0));
                x1 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] + 16));
                x2 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] + 32));
                x3 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] + 48));

                x0 = _mm_min_epi32(x0, _mm_load_si128((__m128i*)((BYTE *)&fade_estimate +  0)));
                x1 = _mm_min_epi32(x1, _mm_load_si128((__m128i*)((BYTE *)&fade_estimate + 16)));
                x2 = _mm_min_epi32(x2, _mm_load_si128((__m128i*)((BYTE *)&fade_estimate + 32)));
                x3 = _mm_min_epi32(x3, _mm_load_si128((__m128i*)((BYTE *)&fade_estimate + 48)));

                x0 = _mm_add_epi32(x0, x2);
                x1 = _mm_add_epi32(x1, x3);

                x0 = _mm_add_epi32(x0, x1);

                const int fade_est_count = x0.m128i_i32[0] + x0.m128i_i32[1] + x0.m128i_i32[2] + x0.m128i_i32[3];

                fade_esitimate_match_point = max(fade_esitimate_match_point, fade_est_count * 100 / (float)frame_size);
            }


            //////////////////     しきい値計算      //////////////////////////////////

            __m128 xf0 = _mm_load_ps((float*)((BYTE *)&sigma +  0));
            __m128 xf1 = _mm_load_ps((float*)((BYTE *)&sigma + 16));
            __m128 xf2 = _mm_load_ps((float*)((BYTE *)&sigma + 32));
            __m128 xf3 = _mm_load_ps((float*)((BYTE *)&sigma + 48));

            xf0 = _mm_add_ps(xf0, xf2);
            xf1 = _mm_add_ps(xf1, xf3);

            xf0 = _mm_add_ps(xf0, xf1);

            const float sigma_avg = (xf0.m128_f32[0] + xf0.m128_f32[1] + xf0.m128_f32[2] + xf0.m128_f32[3]) * (1.0f / (float)HIST_LEN);

            const float threshold_internal = 100.0f - ((100.0f - threshold) * (1.0f + sqrt(sigma_avg)));

            //////////////////     判定      //////////////////////////////////

            float max_match_point = max(simple_match_point, esitimate_match_point);

            bool is_fade = false;
            if (fade_esitimate_match_point - esitimate_match_point < (100.0f - threshold) * (prev_fade ? 2 : 1)) {
                if (fade_esitimate_match_point - esitimate_match_point > 0.0f || fade_esitimate_match_point >= 100.0f - 1e-4f) {
                    is_fade = true;
                    const float diff = avg_luma[index] - avg_luma[index-1];
                    for (int idx = 2; idx < index && is_fade; idx++)
                        if (diff * (avg_luma[idx] - avg_luma[idx-1]) < 0)
                            is_fade = false;
                    for (int idx = 0, count = 0; idx <= index && is_fade; idx++)
                        if (2 <= (count += (prev_max_match_point[idx] > prev_fade_match_point[idx])))
                            is_fade = false;
                }
            }
            if (is_fade)
                max_match_point = max(max_match_point, fade_esitimate_match_point);

            flag &= KEY_FRAMETYPE[i_field] * (max_match_point < threshold_internal);

            result |= flag;

#if SC_DEBUG
            fprintf(fp_sc_log, "%4d,%f,%f,%f,%d,,", flag, threshold_internal, esitimate_match_point, fade_esitimate_match_point, is_fade);
            for (int i_hist = 0; i_hist < HIST_LEN; i_hist++)
                fprintf(fp_sc_log, "%4d,", estimate.v[i_hist]);
            fprintf(fp_sc_log, ",");
            for (int i_hist = 0; i_hist < HIST_LEN; i_hist++)
                fprintf(fp_sc_log, "%4d,", fade_estimate.v[i_hist]);
#endif
            //////////////////     各値を保存      //////////////////////////////////
            if (!flag) {
                prev_max_match_point[index] = max_match_point;
                prev_fade_match_point[index] = fade_esitimate_match_point;
            }
            prev_fade = is_fade;
        } else {
            //if !(index >= HIST_COUNT - 1)
            prev_max_match_point[index] = simple_match_point;
#if SC_DEBUG
            for (int i = 0; i < HIST_LEN * 2 + 1 + 6; i++)
                fprintf(fp_sc_log, ",");
#endif
        }

        //////////////////     QP値を計算      //////////////////////////////////

        if (qp_offset) {
            if (result & KEY_FRAMETYPE[i_field]) {
                qp_offset[i_field] = 0;
            } else {
                float prev_max_match_point_sum = 0.0f;
                for (int idx = 0; idx <= index; idx++)
                    prev_max_match_point_sum += prev_max_match_point[idx];
                const float prev_max_match_point_avg = prev_max_match_point_sum / (float)(index+1);
                float tmp = 0.0f;
                for (int idx = 0; idx <= index; idx++)
                    tmp += pow2(prev_max_match_point[idx] - prev_max_match_point_avg);
                const float prev_max_match_point_sigma = tmp / (float)(index+1);

                int point_count = 1;
                float match_point_sum = prev_max_match_point[index];
                for (int idx = 0; idx < index; idx++) {
                    if (abs(prev_max_match_point[idx] - prev_max_match_point[index]) < prev_max_match_point_sigma) {
                        match_point_sum += prev_max_match_point[idx];
                        point_count++;
                    }
                }
                const float avg_match_point = match_point_sum / point_count;
                const float strength = 8;//max(vqp_strength, 1e-4f);
                const int   sensitivity = 80;//clamp(vqp_sensitivity, 0, 99);
                const float sqrt_plus_coef = strength * 0.05f;
                const float lambda = -log10(1 - (float)sensitivity / 100.0f);
                const float value_at_0_5 = strength * (1.0f - exp(log10(1 - (float)sensitivity / 100.0f)*0.4f));
                const float special_bonus = (prev_fade) ? 0.2f : 1.0f;
                const float mv_value = (100.0f - avg_match_point) * special_bonus;
                const float offset = strength * (1.0f - exp(-lambda * mv_value)) - value_at_0_5 + sqrt_plus_coef * sqrt(mv_value);
                qp_offset[i_field] = (int)(offset + 0.5f - (offset < 0));
#if SC_DEBUG
                fprintf(fp_sc_log, ",%f,", offset);
#endif
            }
        }
#if SC_DEBUG
        fprintf(fp_sc_log, ",->%d\n", result);
#endif
        current_gop_len++;
        if (!result && current_gop_len >= gop_len_max)
            result |= KEY_FRAMETYPE[i_field];
        else if (result && current_gop_len < gop_len_min) {
            result &= (~KEY_FRAMETYPE[i_field]);
        }

        if (result & KEY_FRAMETYPE[i_field]) {
            if (index) {
                x0 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] +  0));
                x1 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] + 16));
                x2 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] + 32));
                x3 = _mm_load_si128((__m128i*)((BYTE *)&hist[index] + 48));
                _mm_store_si128((__m128i*)((BYTE *)&hist[0] +  0), x0);
                _mm_store_si128((__m128i*)((BYTE *)&hist[0] + 16), x1);
                _mm_store_si128((__m128i*)((BYTE *)&hist[0] + 32), x2);
                _mm_store_si128((__m128i*)((BYTE *)&hist[0] + 48), x3);
                prev_max_match_point[0] = prev_max_match_point[index];
                prev_fade_match_point[0] = prev_fade_match_point[index];
                avg_luma[0] = avg_luma[index];
            }
            index = 1;
            prev_fade = false;
            current_gop_len = 0;
        } else {
            if (index == HIST_COUNT - 1) {
                for (int idx = 1; idx <= index; idx++) {
                    x0 = _mm_load_si128((__m128i*)((BYTE *)&hist[idx] +  0));
                    x1 = _mm_load_si128((__m128i*)((BYTE *)&hist[idx] + 16));
                    x2 = _mm_load_si128((__m128i*)((BYTE *)&hist[idx] + 32));
                    x3 = _mm_load_si128((__m128i*)((BYTE *)&hist[idx] + 48));
                    _mm_store_si128((__m128i*)((BYTE *)&hist[idx-1] +  0), x0);
                    _mm_store_si128((__m128i*)((BYTE *)&hist[idx-1] + 16), x1);
                    _mm_store_si128((__m128i*)((BYTE *)&hist[idx-1] + 32), x2);
                    _mm_store_si128((__m128i*)((BYTE *)&hist[idx-1] + 48), x3);
                    prev_max_match_point[idx-1] = prev_max_match_point[idx];
                    prev_fade_match_point[idx-1] = prev_fade_match_point[idx];
                    avg_luma[idx-1] = avg_luma[idx];
                }
            } else {
                index++;
            }
        }
    }

    return result;
}
