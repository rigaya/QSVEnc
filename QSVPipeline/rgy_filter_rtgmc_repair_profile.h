// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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
// ------------------------------------------------------------------------------------------

#pragma once

#include <array>
#include <cstdint>

static constexpr uint8_t RGY_RTGMC_REPAIR_THIN_WIDE_CORE = 1 << 0;
static constexpr uint8_t RGY_RTGMC_REPAIR_THIN_CORE_BLEND = 1 << 1;
static constexpr uint8_t RGY_RTGMC_REPAIR_THIN_RANK_LIMIT = 1 << 2;
static constexpr uint8_t RGY_RTGMC_REPAIR_RESTORE_WIDE_ENVELOPE = 1 << 0;
static constexpr uint8_t RGY_RTGMC_REPAIR_RESTORE_LEVEL4_PATH = 1 << 1;
static constexpr uint8_t RGY_RTGMC_REPAIR_RESTORE_ENABLED = 1 << 2;
static constexpr int RGY_RTGMC_REPAIR_MIN_THIN_REJECT_LEVEL = 0;
static constexpr int RGY_RTGMC_REPAIR_MAX_THIN_REJECT_LEVEL = 7;
static constexpr int RGY_RTGMC_REPAIR_MIN_RESTORE_PADDING_LEVEL = 0;
static constexpr int RGY_RTGMC_REPAIR_MAX_RESTORE_PADDING_LEVEL = 3;
static constexpr int RGY_RTGMC_REPAIR_THIN_REJECT_LEVEL_COUNT =
    RGY_RTGMC_REPAIR_MAX_THIN_REJECT_LEVEL - RGY_RTGMC_REPAIR_MIN_THIN_REJECT_LEVEL + 1;
static constexpr int RGY_RTGMC_REPAIR_RESTORE_PADDING_LEVEL_COUNT =
    RGY_RTGMC_REPAIR_MAX_RESTORE_PADDING_LEVEL - RGY_RTGMC_REPAIR_MIN_RESTORE_PADDING_LEVEL + 1;

struct RGYRtgmcRepairProfile {
    uint8_t thinRejectLevel;
    uint8_t restorePaddingLevel;
    uint8_t thinRejectFlags;
    uint8_t restoreFlags;
};
static_assert(sizeof(RGYRtgmcRepairProfile) == sizeof(uint32_t), "RGYRtgmcRepairProfile must fit in one 32-bit value.");

inline int rgy_rtgmc_repair_clamp(const int value, const int minValue, const int maxValue) {
    return (value < minValue) ? minValue : (value > maxValue) ? maxValue : value;
}

inline bool rgy_rtgmc_repair_thin_level_is_valid(const int thinRejectLevel) {
    return thinRejectLevel >= RGY_RTGMC_REPAIR_MIN_THIN_REJECT_LEVEL
        && thinRejectLevel <= RGY_RTGMC_REPAIR_MAX_THIN_REJECT_LEVEL;
}

inline bool rgy_rtgmc_repair_pad_level_is_valid(const int restorePaddingLevel) {
    return restorePaddingLevel >= RGY_RTGMC_REPAIR_MIN_RESTORE_PADDING_LEVEL
        && restorePaddingLevel <= RGY_RTGMC_REPAIR_MAX_RESTORE_PADDING_LEVEL;
}

inline bool rgy_rtgmc_repair_levels_are_valid(const int thinRejectLevel, const int restorePaddingLevel) {
    return rgy_rtgmc_repair_thin_level_is_valid(thinRejectLevel)
        && rgy_rtgmc_repair_pad_level_is_valid(restorePaddingLevel);
}

inline RGYRtgmcRepairProfile rgy_rtgmc_repair_profile_from_levels(const int thinLevel, const int padLevel) {
    static constexpr std::array<uint8_t, RGY_RTGMC_REPAIR_THIN_REJECT_LEVEL_COUNT> thinFlagsByLevel = {
        0,
        RGY_RTGMC_REPAIR_THIN_CORE_BLEND,
        RGY_RTGMC_REPAIR_THIN_CORE_BLEND | RGY_RTGMC_REPAIR_THIN_RANK_LIMIT,
        0,
        RGY_RTGMC_REPAIR_THIN_CORE_BLEND,
        RGY_RTGMC_REPAIR_THIN_CORE_BLEND | RGY_RTGMC_REPAIR_THIN_RANK_LIMIT,
        RGY_RTGMC_REPAIR_THIN_WIDE_CORE,
        RGY_RTGMC_REPAIR_THIN_WIDE_CORE | RGY_RTGMC_REPAIR_THIN_CORE_BLEND
    };
    static constexpr std::array<uint8_t, RGY_RTGMC_REPAIR_THIN_REJECT_LEVEL_COUNT> restoreFlagsByLevel = {
        0,
        RGY_RTGMC_REPAIR_RESTORE_ENABLED,
        RGY_RTGMC_REPAIR_RESTORE_ENABLED,
        RGY_RTGMC_REPAIR_RESTORE_ENABLED,
        RGY_RTGMC_REPAIR_RESTORE_ENABLED,
        RGY_RTGMC_REPAIR_RESTORE_ENABLED | RGY_RTGMC_REPAIR_RESTORE_WIDE_ENVELOPE,
        RGY_RTGMC_REPAIR_RESTORE_ENABLED | RGY_RTGMC_REPAIR_RESTORE_WIDE_ENVELOPE,
        RGY_RTGMC_REPAIR_RESTORE_ENABLED | RGY_RTGMC_REPAIR_RESTORE_WIDE_ENVELOPE
    };
    static constexpr std::array<uint8_t, RGY_RTGMC_REPAIR_THIN_REJECT_LEVEL_COUNT> noPaddingRestoreFlagsByLevel = {
        0,
        0,
        0,
        0,
        RGY_RTGMC_REPAIR_RESTORE_LEVEL4_PATH,
        0,
        0,
        0
    };

    const int thinRejectLevel = rgy_rtgmc_repair_clamp(
        thinLevel,
        RGY_RTGMC_REPAIR_MIN_THIN_REJECT_LEVEL,
        RGY_RTGMC_REPAIR_MAX_THIN_REJECT_LEVEL);
    const int restorePaddingLevel = rgy_rtgmc_repair_clamp(
        padLevel,
        RGY_RTGMC_REPAIR_MIN_RESTORE_PADDING_LEVEL,
        RGY_RTGMC_REPAIR_MAX_RESTORE_PADDING_LEVEL);
    return RGYRtgmcRepairProfile {
        (uint8_t)thinRejectLevel,
        (uint8_t)restorePaddingLevel,
        thinFlagsByLevel[thinRejectLevel],
        (uint8_t)(restoreFlagsByLevel[thinRejectLevel]
            | ((restorePaddingLevel == 0) ? noPaddingRestoreFlagsByLevel[thinRejectLevel] : 0))
    };
}

inline uint32_t rgy_rtgmc_repair_profile_pack(const RGYRtgmcRepairProfile& profile) {
    return (uint32_t)profile.thinRejectLevel
        | ((uint32_t)profile.restorePaddingLevel << 8)
        | ((uint32_t)profile.thinRejectFlags << 16)
        | ((uint32_t)profile.restoreFlags << 24);
}
