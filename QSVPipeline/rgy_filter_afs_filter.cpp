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

#define FILTER_BLOCK_INT_X  (32) //work groupサイズ(x) = スレッド数/work group
#define FILTER_BLOCK_Y       (8) //work groupサイズ(y) = スレッド数/work group

RGY_ERR afsStripeCache::init(std::shared_ptr<RGYLog> log) {
    if (!m_analyzeMapFilter) {
        const auto options = strsprintf("-D FILTER_BLOCK_INT_X=%d -D FILTER_BLOCK_Y=%d",
            FILTER_BLOCK_INT_X, FILTER_BLOCK_Y);
        m_analyzeMapFilter = m_cl->buildResource(_T("RGY_FILTER_AFS_FILTER_CL"), _T("EXE_DATA"), options.c_str());
        if (!m_analyzeMapFilter) {
            log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to load RGY_FILTER_AFS_FILTER_CL\n"));
            return RGY_ERR_OPENCL_CRUSH;
        }
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
    RGYWorkSize local(FILTER_BLOCK_INT_X, FILTER_BLOCK_Y);
    RGYWorkSize global(sp->map->frame.width, sp->map->frame.height);
    RGY_ERR err = m_analyzeMapFilter->kernel("kernel_afs_analyze_map_filter").config(queue, local, global).launch(
        (cl_mem)dst->map->frame.ptr[0], (cl_mem)sp->map->frame.ptr[0],
        divCeil<int>(sp->map->frame.width, sizeof(uint32_t)), divCeil<int>(sp->map->frame.pitch[0], sizeof(uint32_t)), sp->map->frame.height);
    return err;
}
