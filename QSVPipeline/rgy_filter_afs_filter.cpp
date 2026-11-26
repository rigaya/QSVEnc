// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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

#include <map>
#include <array>
#include <cmath>
#include "convert_csp.h"
#include "rgy_filter_afs.h"
#include "rgy_opencl.h"

#define MAPF_BLOCK_X  (64) //work groupサイズ(x) = スレッド数/work group
#define MAPF_ROWS     (16) //1スレッドが縦方向に処理する行数

RGY_ERR afsStripeCache::init(std::shared_ptr<RGYLog> log) {
    if (!m_analyzeMapFilter.get()) {
        const auto options = strsprintf("-D MAPF_BLOCK_X=%d -D MAPF_ROWS=%d",
            MAPF_BLOCK_X, MAPF_ROWS);
        m_analyzeMapFilter.set(m_cl->buildResourceAsync(_T("RGY_FILTER_AFS_FILTER_CL"), _T("EXE_DATA"), options.c_str()));
    }
    return RGY_ERR_NONE;
}

RGY_ERR afsStripeCache::map_filter(AFS_STRIPE_DATA *dst, AFS_STRIPE_DATA *sp, RGYOpenCLQueue &queue) {
    dst->count0 = sp->count0;
    dst->count1 = sp->count1;
    dst->frame  = sp->frame;
    dst->status = 1;
    if (sp->map->frame.pitch[0] % sizeof(uint32_t) != 0) {
        return RGY_ERR_INVALID_IMAGE_SIZE;
    }
    //横方向は1スレッドで1語(4画素)、縦方向は1スレッドでMAPF_ROWS行を処理する
    const int widthWords = divCeil<int>(sp->map->frame.width, sizeof(uint32_t));
    RGYWorkSize local(MAPF_BLOCK_X, 1);
    RGYWorkSize global(widthWords, divCeil<int>(sp->map->frame.height, MAPF_ROWS));
    RGY_ERR err = m_analyzeMapFilter.get()->kernel("kernel_afs_analyze_map_filter").config(queue, local, global).launch(
        (cl_mem)dst->map->frame.ptr[0], (cl_mem)sp->map->frame.ptr[0],
        widthWords, divCeil<int>(sp->map->frame.pitch[0], sizeof(uint32_t)), sp->map->frame.height);
    return err;
}
