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
#include "rgy_filter_afs.h"

#define BLOCK_INT_X  (32) //blockDim(x) = スレッド数/ブロック
#define BLOCK_Y       (8) //blockDim(y) = スレッド数/ブロック
#define BLOCK_LOOP_Y (16) //ブロックのy方向反復数

#define SHARED_INT_X (BLOCK_INT_X) //sharedメモリの幅
#define SHARED_Y     (16) //sharedメモリの縦

static const char* AFS_ANALYZE_KERNEL_NAME = "kernel_afs_analyze_12";

#pragma warning(push)
#pragma warning(disable: 4127) //C4127: 条件式が定数です。
RGY_ERR RGYFilterAfs::build_analyze(const RGY_CSP csp, const bool tb_order) {
    if (!m_analyze.get()) {
        auto options = strsprintf("-D BIT_DEPTH=%d -D YUV420=%d -D TB_ORDER=%d -D BLOCK_INT_X=%d -D BLOCK_Y=%d -D BLOCK_LOOP_Y=%d",
            RGY_CSP_BIT_DEPTH[csp],
            RGY_CSP_CHROMA_FORMAT[csp] == RGY_CHROMAFMT_YUV420 ? 1 : 0,
            (tb_order) ? 1 : 0,
            BLOCK_INT_X, BLOCK_Y, BLOCK_LOOP_Y);
        const auto sub_group_ext_avail = m_cl->platform()->checkSubGroupSupport(m_cl->queue().devid());
        if (ENCODER_QSV && sub_group_ext_avail != RGYOpenCLSubGroupSupport::NONE) { // VCEではこれを使用するとかえって遅くなる
            m_analyze.set(m_cl->threadPool()->enqueue([cl = m_cl, log = m_pLog, options, sub_group_ext_avail]() {
                auto buildoptions = options;
                if (   sub_group_ext_avail == RGYOpenCLSubGroupSupport::STD22
                    || sub_group_ext_avail == RGYOpenCLSubGroupSupport::STD20KHR) {
                    buildoptions += " -cl-std=CL2.0 ";
                }
                //subgroup情報を得るため一度コンパイル
                auto analyze = cl->buildResource(_T("RGY_FILTER_AFS_ANALYZE_CL"), _T("EXE_DATA"), buildoptions.c_str());
                if (!analyze) {
                    log->write(RGY_LOG_ERROR, RGY_LOGT_VPP, _T("failed to load RGY_FILTER_AFS_ANALYZE_CL\n"));
                    return std::unique_ptr<RGYOpenCLProgram>();
                }

                auto getKernelSubGroupInfo = clGetKernelSubGroupInfo != nullptr ? clGetKernelSubGroupInfo : clGetKernelSubGroupInfoKHR;
                RGYWorkSize local(BLOCK_INT_X, BLOCK_Y);
                size_t subgroup_size = 0;
                auto err = getKernelSubGroupInfo(analyze->kernel(AFS_ANALYZE_KERNEL_NAME).get()->get(), cl->platform()->dev(0).id(), CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR,
                    sizeof(local.w[0]) * 2, &local.w[0], sizeof(subgroup_size), &subgroup_size, nullptr);
                if (err == 0) {
                    buildoptions += strsprintf(" -D SUB_GROUP_SIZE=%u", subgroup_size);
                }
                return cl->buildResource(_T("RGY_FILTER_AFS_ANALYZE_CL"), _T("EXE_DATA"), buildoptions.c_str());
            }));
        } else {
            m_analyze.set(m_cl->buildResourceAsync(_T("RGY_FILTER_AFS_ANALYZE_CL"), _T("EXE_DATA"), options.c_str()));
        }
    }
    return RGY_ERR_NONE;
}
#pragma warning(pop)

typedef uint8_t Flag;

template<typename Type, bool yuv420>
RGY_ERR run_analyze_stripe(uint8_t *dst,
    afsSourceCacheFrame *p0, afsSourceCacheFrame *p1, const int bit_depth,
    const int dstPitch,
    unique_ptr<RGYCLBuf> &count_motion,
    const VppAfs *pAfsPrm, RGYOpenCLQueue &queue,
    std::vector<RGYOpenCLEvent> &wait_event, RGYOpenCLEvent &event,
    RGYOpenCLProgram *analyze, RGYOpenCLContext *cl) {
    auto err = RGY_ERR_NONE;

    const int srcWidth  = p0->y->frame.width;
    const int srcHeight = p0->y->frame.height;
    cl_mem texP0Y = 0;
    cl_mem texP1Y = 0;
    if ((err = cl->createImageFromPlane(texP0Y, p0->y->mem(0), bit_depth, CL_RGBA, false, p0->y->frame.pitch[0], (p0->y->frame.width + 3) / 4, p0->y->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
    if ((err = cl->createImageFromPlane(texP1Y, p1->y->mem(0), bit_depth, CL_RGBA, false, p1->y->frame.pitch[0], (p1->y->frame.width + 3) / 4, p1->y->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;

    cl_mem texP0U0 = 0;
    cl_mem texP0U1 = 0; //yuv444では使用されない
    cl_mem texP0V0 = 0;
    cl_mem texP0V1 = 0; //yuv444では使用されない
    cl_mem texP1U0 = 0;
    cl_mem texP1U1 = 0; //yuv444では使用されない
    cl_mem texP1V0 = 0;
    cl_mem texP1V1 = 0; //yuv444では使用されない
    if (yuv420) {
        if ((err = cl->createImageFromPlane(texP0U0, p0->cb[0]->mem(0), bit_depth, CL_R, true, p0->cb[0]->frame.pitch[0], p0->cb[0]->frame.width, p0->cb[0]->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
        if ((err = cl->createImageFromPlane(texP0U1, p0->cb[1]->mem(0), bit_depth, CL_R, true, p0->cb[1]->frame.pitch[0], p0->cb[1]->frame.width, p0->cb[1]->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
        if ((err = cl->createImageFromPlane(texP0V0, p0->cr[0]->mem(0), bit_depth, CL_R, true, p0->cr[0]->frame.pitch[0], p0->cr[0]->frame.width, p0->cr[0]->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
        if ((err = cl->createImageFromPlane(texP0V1, p0->cr[1]->mem(0), bit_depth, CL_R, true, p0->cr[1]->frame.pitch[0], p0->cr[1]->frame.width, p0->cr[1]->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
        if ((err = cl->createImageFromPlane(texP1U0, p1->cb[0]->mem(0), bit_depth, CL_R, true, p1->cb[0]->frame.pitch[0], p1->cb[0]->frame.width, p1->cb[0]->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
        if ((err = cl->createImageFromPlane(texP1U1, p1->cb[1]->mem(0), bit_depth, CL_R, true, p1->cb[1]->frame.pitch[0], p1->cb[1]->frame.width, p1->cb[1]->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
        if ((err = cl->createImageFromPlane(texP1V0, p1->cr[0]->mem(0), bit_depth, CL_R, true, p1->cr[0]->frame.pitch[0], p1->cr[0]->frame.width, p1->cr[0]->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
        if ((err = cl->createImageFromPlane(texP1V1, p1->cr[1]->mem(0), bit_depth, CL_R, true, p1->cr[1]->frame.pitch[0], p1->cr[1]->frame.width, p1->cr[1]->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
    } else {
        if ((err = cl->createImageFromPlane(texP0U0, p0->y->mem(1), bit_depth, CL_RGBA, false, p0->y->frame.pitch[1], (p0->y->frame.width + 3) / 4, p0->y->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
        if ((err = cl->createImageFromPlane(texP0V0, p0->y->mem(2), bit_depth, CL_RGBA, false, p0->y->frame.pitch[2], (p0->y->frame.width + 3) / 4, p0->y->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
        if ((err = cl->createImageFromPlane(texP1U0, p1->y->mem(1), bit_depth, CL_RGBA, false, p1->y->frame.pitch[1], (p1->y->frame.width + 3) / 4, p1->y->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
        if ((err = cl->createImageFromPlane(texP1V0, p1->y->mem(2), bit_depth, CL_RGBA, false, p1->y->frame.pitch[2], (p1->y->frame.width + 3) / 4, p1->y->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
        //以下はyuv444では本来使用しないが、ダミーで入れておかないとsetArg時にエラーになる
        texP0U1 = texP0U0; 
        texP0V1 = texP0V0;
        texP1U1 = texP1U0;
        texP1V1 = texP1V0;
    }

    const RGYWorkSize local(BLOCK_INT_X, BLOCK_Y);
    //横方向は1スレッドで4pixel処理する
    const RGYWorkSize global(divCeil(srcWidth, 4), divCeil(srcHeight, BLOCK_LOOP_Y));

    const auto grid_count = global.groups(local).total();
    if (!count_motion || count_motion->size() < grid_count * sizeof(int)) {
        count_motion = cl->createBuffer(grid_count * sizeof(int), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
        if (!count_motion) {
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    //opencl版を変更、横方向は1スレッドで4pixel処理するため、1/4にする必要がある
    const uint32_t scan_left   = pAfsPrm->clip.left >> 2;
    const uint32_t scan_width  = (srcWidth - pAfsPrm->clip.left - pAfsPrm->clip.right) >> 2;
    const uint32_t scan_top    = pAfsPrm->clip.top;
    const uint32_t scan_height = (srcHeight - pAfsPrm->clip.top - pAfsPrm->clip.bottom) & ~1;

    //YC48 -> yuv420/yuv444(bit_depth)へのスケーリングのシフト値
    const int thre_rsft = 12 - (bit_depth - 8);
    //8bitなら最大127まで、16bitなら最大32627まで (bitshift等を使って比較する都合)
    const int thre_max = (1 << (sizeof(Type) * 8 - 1)) - 1;
    //YC48 -> yuv420/yuv444(bit_depth)へのスケーリング
    const Type thre_shift_yuv   = (Type)clamp((pAfsPrm->thre_shift   * 219 +  383)>>thre_rsft, 0, thre_max);
    const Type thre_deint_yuv   = (Type)clamp((pAfsPrm->thre_deint   * 219 +  383)>>thre_rsft, 0, thre_max);
    const Type thre_Ymotion_yuv = (Type)clamp((pAfsPrm->thre_Ymotion * 219 +  383)>>thre_rsft, 0, thre_max);
    const Type thre_Cmotion_yuv = (Type)clamp((pAfsPrm->thre_Cmotion * 224 + 2112)>>thre_rsft, 0, thre_max);

    //YC48 -> yuv420/yuv444(bit_depth)へのスケーリング
    //色差はcudaReadModeNormalizedFloatをつ開くので、そのぶんのスケーリングも必要
    const float thre_mul = (224.0f / (float)(4096 >> (bit_depth - 8))) * (1.0f / (1 << (sizeof(Type) * 8)));
    const float thre_shift_yuvf   = std::max(0.0f, pAfsPrm->thre_shift * thre_mul);
    const float thre_deint_yuvf   = std::max(0.0f, pAfsPrm->thre_deint * thre_mul);
    const float thre_Cmotion_yuvf = std::max(0.0f, pAfsPrm->thre_Cmotion * thre_mul);

    err = analyze->kernel(AFS_ANALYZE_KERNEL_NAME).config(queue, local, global, wait_event, &event).launch(
        (cl_mem)dst, count_motion->mem(),
        texP0Y, texP0U0, texP0U1, texP0V0, texP0V1,
        texP1Y, texP1U0, texP1U1, texP1V0, texP1V1,
        divCeil(srcWidth, 4), (int)(dstPitch / sizeof(uint32_t)), srcHeight,
        thre_Ymotion_yuv, thre_deint_yuv, thre_shift_yuv,
        thre_Cmotion_yuv, thre_Cmotion_yuvf, thre_deint_yuvf, thre_shift_yuvf,
        scan_left, scan_top, scan_width, scan_height);

    clReleaseMemObject(texP0Y);
    clReleaseMemObject(texP1Y);
    clReleaseMemObject(texP0U0);
    clReleaseMemObject(texP0V0);
    clReleaseMemObject(texP1U0);
    clReleaseMemObject(texP1V0);
    if (yuv420) {
        clReleaseMemObject(texP0U1);
        clReleaseMemObject(texP0V1);
        clReleaseMemObject(texP1U1);
        clReleaseMemObject(texP1V1);
    }
    return err;
}

class analyze_func {
private:
    static decltype(run_analyze_stripe<uint8_t, true>) *func[2][2];
public:
    static decltype(run_analyze_stripe<uint8_t, true>) *get(RGY_CSP csp) {
        const int idx1 = RGY_CSP_BIT_DEPTH[csp] > 8 ? 1 : 0;
        const int idx2 = RGY_CSP_CHROMA_FORMAT[csp] == RGY_CHROMAFMT_YUV420 ? 1 : 0;
        return func[idx1][idx2];
    }
};
decltype(run_analyze_stripe<uint8_t, true>) *analyze_func::func[2][2] = {
    { run_analyze_stripe<uint8_t,  false>, run_analyze_stripe<uint8_t,  true> },
    { run_analyze_stripe<uint16_t, false>, run_analyze_stripe<uint16_t, true> }
};

RGY_ERR RGYFilterAfs::analyze_stripe(afsSourceCacheFrame *p0, afsSourceCacheFrame *p1, AFS_SCAN_DATA *sp, unique_ptr<RGYCLBuf> &count_motion, const RGYFilterParamAfs *pAfsPrm, RGYOpenCLQueue &queue, std::vector<RGYOpenCLEvent> wait_event, RGYOpenCLEvent &event) {
    if (   RGY_CSP_CHROMA_FORMAT[m_source.csp()] != RGY_CHROMAFMT_YUV420
        && RGY_CSP_CHROMA_FORMAT[m_source.csp()] != RGY_CHROMAFMT_YUV444) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid color format for afs synthesize: %s.\n"), RGY_CSP_NAMES[m_source.csp()]);
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }
    return analyze_func::get(m_source.csp())(
        sp->map->frame.ptr[0], p0, p1, RGY_CSP_BIT_DEPTH[m_source.csp()],
        sp->map->frame.pitch[0],
        count_motion, &pAfsPrm->afs, queue, wait_event, event, m_analyze.get(), m_cl.get());
}
