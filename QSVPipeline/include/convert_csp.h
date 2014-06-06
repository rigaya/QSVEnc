//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ---------------------------------------------------------------------------------------
#ifndef _CONVERT_CSP_H_
#define _CONVERT_CSP_H_

typedef void (*func_convert_csp) (void **dst, void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int *crop);

typedef struct ConvertCSP {
	unsigned int csp_from, csp_to;
	bool uv_only;
	func_convert_csp func[2];
	unsigned int simd;
} ConvertCSP;

const ConvertCSP *get_convert_csp_func(unsigned int csp_from, unsigned int csp_to, bool uv_only);
const TCHAR *get_simd_str(unsigned int simd);

#endif //_CONVERT_CSP_H_
