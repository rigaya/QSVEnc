//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <cstdint>
#include <intrin.h>
#include "qsv_simd.h"

unsigned int get_availableSIMD() {
	int CPUInfo[4];
	__cpuid(CPUInfo, 1);
	uint32_t simd = NONE;
	if (CPUInfo[3] & 0x04000000) simd |= SSE2;
	if (CPUInfo[2] & 0x00000001) simd |= SSE3;
	if (CPUInfo[2] & 0x00000200) simd |= SSSE3;
	if (CPUInfo[2] & 0x00080000) simd |= SSE41;
	if (CPUInfo[2] & 0x00100000) simd |= SSE42;
	if (CPUInfo[2] & 0x00800000) simd |= POPCNT;
#if (_MSC_VER >= 1600)
	uint64_t xgetbv = 0;
	if ((CPUInfo[2] & 0x18000000) == 0x18000000) {
		xgetbv = _xgetbv(0);
		if ((xgetbv & 0x06) == 0x06)
			simd |= AVX;
	}
#endif
#if (_MSC_VER >= 1700)
	__cpuid(CPUInfo, 7);
	if ((simd & AVX) && (CPUInfo[1] & 0x00000020))
		simd |= AVX2;
#endif
	return simd;
}
