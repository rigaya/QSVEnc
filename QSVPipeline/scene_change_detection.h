//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _SCENE_CHANGE_DETECTION_H_
#define _SCENE_CHANGE_DETECTION_H_

#include <thread>
#include <atomic>
#include "qsv_osdep.h"
#include "mfxstructures.h"

#pragma warning (push)
#pragma warning (disable:4324)

const int HIST_LEN_2N = 4;
const int HIST_LEN = 1<<(8-HIST_LEN_2N);
const int HIST_COUNT = 5;

#define SC_DEBUG 0 // 0 or 1

typedef struct {
    int v[HIST_LEN];
} hist_t;

typedef struct {
    alignas(32) hist_t hist_thread;
    int id;
    std::atomic_int abort;
    void *ptr_csd;
    std::thread hnd;
    HANDLE he_start;
    HANDLE he_fin;
    mfxU8 reserved[128-(sizeof(hist_t)+sizeof(int)+sizeof(std::thread)+sizeof(std::atomic_int)+sizeof(void*)+sizeof(HANDLE)*2)];
} hist_thread_t;

typedef void (*func_make_hist_simd)(const uint8_t *frame_Y, hist_t *hist_buf, int y_start, int y_end, int y_step, int x_skip, int width, int pitch);

func_make_hist_simd get_make_hist_func();

//SSE2, SSSE4.1, SSE4.2 必須
class CSceneChangeDetect
{
public:
    CSceneChangeDetect();
    virtual ~CSceneChangeDetect();

    uint16_t Check(mfxFrameSurface1 *frame, int *qp_offset);
    //threshold 0-100 (デフォルト80, 小さいほうがシーンチェンジと判定しにくい)
    int Init(int _threshold, uint32_t _pic_struct, uint16_t _vqp_strength, uint16_t _vqp_sensitivity, uint16_t _gop_len_min, uint16_t _gop_len_max, bool _deint_normal);

    bool isInitialized() {
        return initialized;
    }
    void Close() {
        thread_close();
        initialized = false;
#if SC_DEBUG
        if (fp_sc_log) {
            fprintf(fp_sc_log, "\n\n\n");
            fclose(fp_sc_log);
        }
        fp_sc_log = NULL;
#endif
    }
    void MakeHist(int thread_id, int thread_max, hist_t *hist_buf);
    int GetSubThreadNum() {
        return sub_thread_num;
    }
    uint16_t getVQPStrength() {
        return vqp_strength;
    }
    uint16_t getVQPSensitivity() {
        return vqp_sensitivity;
    }
    uint16_t getMinGOPLen() {
        return gop_len_min;
    }
    uint16_t getMaxGOPLen() {
        return (deint_normal) ? gop_len_max>>1 : gop_len_max;
    }
private:
    bool initialized;
    uint32_t pic_struct;
    bool deint_normal;
    int threshold;
    int index;
    int current_gop_len;
    uint16_t gop_len_min;
    uint16_t gop_len_max;

    uint16_t vqp_strength;
    uint16_t vqp_sensitivity;

    func_make_hist_simd mask_histgram;

    mfxFrameSurface1 *target_frame;
    int i_field;

    //スレッド管理
    int sub_thread_num;
    int thread_start();
    void thread_close();

    //クラス内で_declspec(align(16))してもどうも効かないらしいので、
    //仕方ないのでaligned_mallocでとる
    hist_thread_t *th_hist;
    HANDLE *he_hist_fin_copy;

    hist_t *hist;
    float prev_max_match_point[HIST_COUNT];
    float prev_fade_match_point[HIST_COUNT];
    float avg_luma[HIST_COUNT];
    bool prev_fade;

#if SC_DEBUG
    FILE *fp_sc_log;
#endif
};

#endif //_SCENE_CHANGE_DETECTION_H_
