// Type
// bit_depth
// YADIF_GEN_FIELD_TOP
// YADIF_GEN_FIELD_BOTTOM
// RGY_PICSTRUCT_TFF

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

#ifndef max3
#define max3(a, b, c) (max(max((a), (b)), (c)))
#endif

#ifndef min3
#define min3(a, b, c) (min(min((a), (b)), (c)))
#endif

#define SRC(ptr, x, y) (*(__global Type *)((__global uchar *)ptr + (y) * srcPitch + (x) * sizeof(Type)))

#define SRC_CLAMP(ptr, x, y) (*(__global Type *)((__global uchar *)ptr + clamp((y),0,srcHeight-1) * srcPitch + clamp((x),0,srcWidth-1) * sizeof(Type)))

int spatial(
    const __global Type *ptrSrc1,
    const int srcPitch,
    const int srcWidth,
    const int srcHeight,
    const int gIdX,
    const int gIdY
) {
    int ym1[7], yp1[7];
    #pragma unroll
    for (int ix = -3; ix <= 3; ix++) {
        ym1[ix+3] = (int)SRC_CLAMP(ptrSrc1, gIdX + ix, gIdY - 1);
        yp1[ix+3] = (int)SRC_CLAMP(ptrSrc1, gIdX + ix, gIdY + 1);
    }

    const int score[5] = {
        abs(ym1[2] - yp1[2]) + abs(ym1[3] - yp1[3]) + abs(ym1[4] - yp1[4]),
        abs(ym1[1] - yp1[3]) + abs(ym1[2] - yp1[4]) + abs(ym1[3] - yp1[5]),
        abs(ym1[0] - yp1[4]) + abs(ym1[1] - yp1[5]) + abs(ym1[2] - yp1[6]),
        abs(ym1[3] - yp1[1]) + abs(ym1[4] - yp1[2]) + abs(ym1[5] - yp1[3]),
        abs(ym1[4] - yp1[0]) + abs(ym1[5] - yp1[1]) + abs(ym1[6] - yp1[2])
    };
    int minscore = score[0];
    int minidx = 0;
    if (score[1] < minscore) {
        minscore = score[1];
        minidx = 1;
        if (score[2] < minscore) {
            minscore = score[2];
            minidx = 2;
        }
    }
    if (score[3] < minscore) {
        minscore = score[3];
        minidx = 3;
        if (score[4] < minscore) {
            minscore = score[4];
            minidx = 4;
        }
    }

    switch (minidx) {
    case 0: return (ym1[3] + yp1[3]) >> 1;
    case 1: return (ym1[2] + yp1[4]) >> 1;
    case 2: return (ym1[1] + yp1[5]) >> 1;
    case 3: return (ym1[4] + yp1[2]) >> 1;
    case 4:
    default:return (ym1[5] + yp1[1]) >> 1;
    }
}

int temporal(
    const __global Type *ptrSrc0,
    const __global Type *ptrSrc01,
    const __global Type *ptrSrc1,
    const __global Type *ptrSrc12,
    const __global Type *ptrSrc2,
    const int srcPitch,
    const int srcWidth,
    const int srcHeight,
    const int valSpatial,
    const int gIdX,
    const int gIdY
) {
    const int t00m1 = (int)SRC_CLAMP(ptrSrc0,  gIdX, gIdY - 1);
    const int t00p1 = (int)SRC_CLAMP(ptrSrc0,  gIdX, gIdY + 1);
    const int t01m2 = (int)SRC_CLAMP(ptrSrc01, gIdX, gIdY - 2);
    const int t01_0 = (int)SRC_CLAMP(ptrSrc01, gIdX, gIdY + 0);
    const int t01p2 = (int)SRC_CLAMP(ptrSrc01, gIdX, gIdY + 2);
    const int t10m1 = (int)SRC_CLAMP(ptrSrc1,  gIdX, gIdY - 1);
    const int t10p1 = (int)SRC_CLAMP(ptrSrc1,  gIdX, gIdY + 1);
    const int t12m2 = (int)SRC_CLAMP(ptrSrc12, gIdX, gIdY - 2);
    const int t12_0 = (int)SRC_CLAMP(ptrSrc12, gIdX, gIdY + 0);
    const int t12p2 = (int)SRC_CLAMP(ptrSrc12, gIdX, gIdY + 2);
    const int t20m1 = (int)SRC_CLAMP(ptrSrc2,  gIdX, gIdY - 1);
    const int t20p1 = (int)SRC_CLAMP(ptrSrc2,  gIdX, gIdY + 1);
    const int tm2 = (t01m2 + t12m2) >> 1;
    const int t_0 = (t01_0 + t12_0) >> 1;
    const int tp2 = (t01p2 + t12p2) >> 1;


    int diff = max3(
        abs(t01_0 - t12_0),
        (abs(t00m1 - t10m1) + abs(t00p1 - t10p1)) >> 1,
        (abs(t20m1 - t10m1) + abs(t10p1 - t20p1)) >> 1);
    diff = max3(diff,
                -max3(t_0 - t10p1, t_0 - t10m1, min(tm2 - t10m1, tp2 - t10p1)),
                 min3(t_0 - t10p1, t_0 - t10m1, max(tm2 - t10m1, tp2 - t10p1)));
    return max(min(valSpatial, t_0 + diff), t_0 - diff);
}

__kernel void kernel_yadif(
    __global Type *ptrDst,
    const int dstPitch,
    const int dstWidth,
    const int dstHeight,
    const __global Type *ptrSrc0,
    const __global Type *ptrSrc1,
    const __global Type *ptrSrc2,
    const int srcPitch,
    const int srcWidth,
    const int srcHeight,
    const int targetField,
    const int picstruct) {
    const int gIdX = get_global_id(0);
    const int gIdY = get_global_id(1);
    if (gIdX < dstWidth && gIdY < dstHeight) {
        Type ret;
        if ((gIdY & 1) != targetField) {
            ret = SRC(ptrSrc1, gIdX, gIdY);
        } else {
            const int valSpatial = spatial(ptrSrc1, srcPitch, srcWidth, srcHeight, gIdX, gIdY);
            const bool field2nd = ((targetField==YADIF_GEN_FIELD_TOP) == ((picstruct & RGY_PICSTRUCT_TFF) != 0));
            const __global Type *ptrSrc01 = field2nd ? ptrSrc1 : ptrSrc0;
            const __global Type *ptrSrc12 = field2nd ? ptrSrc2 : ptrSrc1;
            ret = (Type)clamp(
                temporal(ptrSrc0, ptrSrc01, ptrSrc1, ptrSrc12, ptrSrc2, srcPitch, srcWidth, srcHeight, valSpatial, gIdX, gIdY),
                0, ((1<<bit_depth)-1));
        }
        *(__global Type *)((__global uchar *)ptrDst + gIdY * dstPitch + gIdX * sizeof(Type)) = ret;
    }
}
