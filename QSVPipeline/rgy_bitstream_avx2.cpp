// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2021 rigaya
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

#include <immintrin.h>
#include "rgy_bitstream.h"

#define CLEAR_LEFT_BIT(x) ((x) & ((x) - 1))

#if defined(_WIN32) || defined(_WIN64)
#define CTZ32(x) _tzcnt_u32(x)
#else
#define CTZ32(x) __builtin_ctz(x)
#endif

static inline int64_t memmem_avx2(const void *data_, const int64_t data_size, const void *target_, const int64_t target_size) {
    uint8_t *data = (uint8_t *)data_;
    const uint8_t *target = (const uint8_t *)target_;
    const __m256i target_first = _mm256_set1_epi8(target[0]);
    const __m256i target_last = _mm256_set1_epi8(target[target_size - 1]);

    for (int64_t i = 0; i < data_size; i += 32) {
        const __m256i r0 = _mm256_loadu_si256((const __m256i*)(data + i));
        const __m256i r1 = _mm256_loadu_si256((const __m256i*)(data + i + target_size - 1));
        uint32_t mask = _mm256_movemask_epi8(_mm256_and_si256(_mm256_cmpeq_epi8(r0, target_first), _mm256_cmpeq_epi8(r1, target_last)));
        while (mask != 0) {
            const auto j = CTZ32(mask);
            if (memcmp(data + i + j + 1, target + 1, target_size - 2) == 0) {
                const auto ret = i + j;
                return ret < data_size ? ret : -1;
            }
            mask = CLEAR_LEFT_BIT(mask);
        }
    }
    return -1;
}

std::vector<nal_info> parse_nal_unit_h264_avx2(const uint8_t * data, size_t size) {
    std::vector<nal_info> nal_list;
    if (size > 3) {
        static const uint8_t header[3] = { 0, 0, 1 };
        nal_info nal_start = { nullptr, 0, 0 };
        int64_t i = 0;
        for (;;) {
            int64_t next = memmem_avx2((const void *)(data + i), size - i, (const void *)header, sizeof(header));
            if (next < 0) break;

            i += next;
            if (nal_start.ptr) {
                nal_list.push_back(nal_start);
            }
            nal_start.ptr = data + i - (i > 0 && data[i - 1] == 0);
            nal_start.type = data[i + 3] & 0x1f;
            nal_start.size = data + size - nal_start.ptr;
            if (nal_list.size()) {
                auto prev = nal_list.end() - 1;
                prev->size = nal_start.ptr - prev->ptr;
            }
            i += 3;
        }
        if (nal_start.ptr) {
            nal_list.push_back(nal_start);
        }
    }
    _mm256_zeroupper();
    return nal_list;
}

std::vector<nal_info> parse_nal_unit_hevc_avx2(const uint8_t *data, size_t size) {
    std::vector<nal_info> nal_list;
    if (size > 3) {
        static const uint8_t header[3] = { 0, 0, 1 };
        nal_info nal_start = { nullptr, 0, 0 };
        int64_t i = 0;
        for (;;) {
            int64_t next = memmem_avx2((const void *)(data + i), size - i, (const void *)header, sizeof(header));
            if (next < 0) break;

            i += next;
            if (nal_start.ptr) {
                nal_list.push_back(nal_start);
            }
            nal_start.ptr = data + i - (i > 0 && data[i - 1] == 0);
            nal_start.type = (data[i + 3] & 0x7f) >> 1;
            nal_start.size = data + size - nal_start.ptr;
            if (nal_list.size()) {
                auto prev = nal_list.end() - 1;
                prev->size = nal_start.ptr - prev->ptr;
            }
            i += 3;
        }
        if (nal_start.ptr) {
            nal_list.push_back(nal_start);
        }
    }
    _mm256_zeroupper();
    return nal_list;
}
