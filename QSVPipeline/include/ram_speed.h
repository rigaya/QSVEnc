//  -----------------------------------------------------------------------------------------
//    ram_speed by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#pragma once
#include <vector>

enum {
    RAM_SPEED_MODE_READ = 0,
    RAM_SPEED_MODE_WRITE = 1,
    RAM_SPEED_MODE_RW = 2,
};

double ram_speed_mt(int check_size_kilobytes, int mode, int thread_n);

std::vector<double> ram_speed_mt_list(int check_size_kilobytes, int mode, bool logical_core = false);
