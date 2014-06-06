//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ----------------------------------------------------------------------------------------

#include <intrin.h>
#include "qsv_simd.h"
#include "scene_change_detection.h"
#include "scene_change_detection_simd.h"


static const func_make_hist_simd FUNC_MAKE_HIST_LIST[] = {
	make_hist_sse2, make_hist_sse41_popcnt, make_hist_avx, make_hist_avx2
};
func_make_hist_simd get_make_hist_func() {
	const DWORD simd = get_availableSIMD();
	int index = ((simd & (SSE41|POPCNT)) == (SSE41|POPCNT)) + ((simd & (AVX|POPCNT)) == (AVX|POPCNT)) + ((simd & (AVX2|AVX|POPCNT)) == (AVX2|AVX|POPCNT));
	return FUNC_MAKE_HIST_LIST[index];
}

void make_hist_sse2(const BYTE *frame_Y, hist_t *hist_buf, int y_start, int y_end, int y_step, int x_skip, int width, int pitch) {
	make_hist_simd(frame_Y, hist_buf, y_start, y_end, y_step, x_skip, width, pitch, SSE2);
}
void make_hist_sse41_popcnt(const BYTE *frame_Y, hist_t *hist_buf, int y_start, int y_end, int y_step, int x_skip, int width, int pitch) {
	make_hist_simd(frame_Y, hist_buf, y_start, y_end, y_step, x_skip, width, pitch, POPCNT|SSE41|SSSE3|SSE3|SSE2);
}
