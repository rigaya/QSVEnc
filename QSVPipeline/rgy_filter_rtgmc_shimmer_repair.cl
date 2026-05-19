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

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

static inline int rtgmc_read_pix(
    const __global uchar *src, int x, int y,
    const int pitch, const int width, const int height
) {
    x = clamp(x, 0, width  - 1);
    y = clamp(y, 0, height - 1);
    return (int)(*(const __global Type *)(src + y * pitch + x * sizeof(Type)));
}

static inline void rtgmc_write_pix(
    __global uchar *dst, int x, int y, const int pitch, const int value
) {
    __global Type *dstPix = (__global Type *)(dst + y * pitch + x * sizeof(Type));
    dstPix[0] = (Type)clamp(value, 0, max_val);
}

static inline int rtgmc_select_signed_correction(
    const int proposedSigned,
    const int positiveMaskSigned,
    const int negativeMaskSigned
) {
    switch ((proposedSigned > 0) - (proposedSigned < 0)) {
        case 1:
            return (positiveMaskSigned > 0) ? positiveMaskSigned : 0;
        case -1:
            return (negativeMaskSigned < 0) ? negativeMaskSigned : 0;
        default:
            return 0;
    }
}

static inline int rtgmc_apply_signed_correction(
    const int src,
    const int proposedSigned,
    const int positiveMaskSigned,
    const int negativeMaskSigned
) {
    const int appliedSigned = rtgmc_select_signed_correction(
        proposedSigned,
        positiveMaskSigned,
        negativeMaskSigned);
    return clamp(src + appliedSigned, 0, max_val);
}

static inline int rtgmc_signed_to_diff(const int signedValue) {
    return clamp(signedValue + range_half, 0, max_val);
}

static inline int rtgmc_shimmer_repair_candidate_signed(
    const __global uchar *input, const int inputPitch,
    const __global uchar *reference, const int referencePitch,
    const int x, const int y,
    const int width, const int height,
    const int positive) {
    int support = 0;
    int peak = 0;
    int sum = 0;
    for (int dy = -RTGMC_SHIMMER_REPAIR_SUPPORT_RADIUS; dy <= RTGMC_SHIMMER_REPAIR_SUPPORT_RADIUS; dy++) {
        for (int dx = -RTGMC_SHIMMER_REPAIR_SUPPORT_RADIUS; dx <= RTGMC_SHIMMER_REPAIR_SUPPORT_RADIUS; dx++) {
            const int delta = rtgmc_read_pix(reference, x + dx, y + dy, referencePitch, width, height)
                - rtgmc_read_pix(input, x + dx, y + dy, inputPitch, width, height);
            if (positive ? (delta > 0) : (delta < 0)) {
                const int magnitude = positive ? delta : -delta;
                support++;
                sum += magnitude;
                peak = max(peak, magnitude);
            }
        }
    }

    if (support < RTGMC_SHIMMER_REPAIR_MIN_SUPPORT_PIXELS) {
        return 0;
    }

    const int mean = (sum + (support / 2)) / support;
    const int candidate = clamp(min(peak, mean + RTGMC_SHIMMER_REPAIR_RESTORE_PADDING_LEVEL), 0, max_val);
    return positive ? candidate : -candidate;
}

__attribute__((reqd_work_group_size(rtgmc_shimmer_repair_block_x, rtgmc_shimmer_repair_block_y, 1)))
__kernel void kernel_rtgmc_shimmer_repair_copy(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *restrict pSrc, const int srcPitch,
    const int width,
    const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;
    const int value = rtgmc_read_pix((const __global uchar *)pSrc, ix, iy, srcPitch, width, height);
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch, value);
}

__attribute__((reqd_work_group_size(rtgmc_shimmer_repair_block_x, rtgmc_shimmer_repair_block_y, 1)))
__kernel void kernel_rtgmc_shimmer_repair_apply(
    __global Type *restrict pDst, const int dstPitch,
    const __global Type *pInput, const int inputPitch,
    const __global Type *pReference, const int referencePitch,
    const int width,
    const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const __global uchar *input = (const __global uchar *)pInput;
    const __global uchar *reference = (const __global uchar *)pReference;
    const int inputValue = rtgmc_read_pix(input, ix, iy, inputPitch, width, height);
    const int referenceValue = rtgmc_read_pix(reference, ix, iy, referencePitch, width, height);
    const int signedDelta = referenceValue - inputValue;
    int positiveGateSigned = 0;
    int negativeGateSigned = 0;
    if (signedDelta > 0) {
        positiveGateSigned = rtgmc_shimmer_repair_candidate_signed(
            input, inputPitch, reference, referencePitch, ix, iy, width, height, 1);
    } else if (signedDelta < 0) {
        negativeGateSigned = rtgmc_shimmer_repair_candidate_signed(
            input, inputPitch, reference, referencePitch, ix, iy, width, height, 0);
    }

    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch,
        rtgmc_apply_signed_correction(
            inputValue,
            signedDelta,
            positiveGateSigned,
            negativeGateSigned));
}

__attribute__((reqd_work_group_size(rtgmc_shimmer_repair_block_x, rtgmc_shimmer_repair_block_y, 1)))
__kernel void kernel_rtgmc_shimmer_repair_apply_fused(
    __global Type *restrict pDst, const int dstPitch,
    __global Type *restrict pCorrectionDelta, const int correctionDeltaPitch,
    __global Type *restrict pPositiveCorrectionGate, const int positiveCorrectionGatePitch,
    __global Type *restrict pNegativeCorrectionGate, const int negativeCorrectionGatePitch,
    const __global Type *pInput, const int inputPitch,
    const __global Type *pReference, const int referencePitch,
    const int width,
    const int height
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    if (ix >= width || iy >= height) return;

    const __global uchar *input = (const __global uchar *)pInput;
    const __global uchar *reference = (const __global uchar *)pReference;
    const int inputValue = rtgmc_read_pix(input, ix, iy, inputPitch, width, height);
    const int referenceValue = rtgmc_read_pix(reference, ix, iy, referencePitch, width, height);
    const int signedDelta = referenceValue - inputValue;
    int positiveGateSigned = 0;
    int negativeGateSigned = 0;
    if (signedDelta > 0) {
        positiveGateSigned = rtgmc_shimmer_repair_candidate_signed(
            input, inputPitch, reference, referencePitch, ix, iy, width, height, 1);
    } else if (signedDelta < 0) {
        negativeGateSigned = rtgmc_shimmer_repair_candidate_signed(
            input, inputPitch, reference, referencePitch, ix, iy, width, height, 0);
    }

    rtgmc_write_pix((__global uchar *)pCorrectionDelta, ix, iy, correctionDeltaPitch, rtgmc_signed_to_diff(signedDelta));
    rtgmc_write_pix((__global uchar *)pPositiveCorrectionGate, ix, iy, positiveCorrectionGatePitch, rtgmc_signed_to_diff(positiveGateSigned));
    rtgmc_write_pix((__global uchar *)pNegativeCorrectionGate, ix, iy, negativeCorrectionGatePitch, rtgmc_signed_to_diff(negativeGateSigned));
    rtgmc_write_pix((__global uchar *)pDst, ix, iy, dstPitch,
        rtgmc_apply_signed_correction(
            inputValue,
            signedDelta,
            positiveGateSigned,
            negativeGateSigned));
}
