#pragma once
#include <vector>

enum {
	RAM_SPEED_MODE_READ = 0,
	RAM_SPEED_MODE_WRITE = 1,
	RAM_SPEED_MODE_RW = 2,
};

double ram_speed_mt(int check_size_kilobytes, int mode, int thread_n);

std::vector<double> ram_speed_mt_list(int check_size_kilobytes, int mode);
