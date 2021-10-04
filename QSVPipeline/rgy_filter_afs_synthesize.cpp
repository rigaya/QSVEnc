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

#define SYN_BLOCK_INT_X  (32) //work groupサイズ(x) = スレッド数/work group
#define SYN_BLOCK_Y       (8) //work groupサイズ(y) = スレッド数/work group
#define SYN_BLOCK_LOOP_Y  (1) //work groupのy方向反復数

RGY_ERR RGYFilterAfs::build_synthesize(const RGY_CSP csp, const int mode) {
    if (!m_synthesize.get()) {
        const auto options = strsprintf("-D BIT_DEPTH=%d -D YUV420=%d -D mode=%d -D SYN_BLOCK_INT_X=%d -D SYN_BLOCK_Y=%d -D SYN_BLOCK_LOOP_Y=%d",
            RGY_CSP_BIT_DEPTH[csp],
            RGY_CSP_CHROMA_FORMAT[csp] == RGY_CHROMAFMT_YUV420 ? 1 : 0,
            mode,
            SYN_BLOCK_INT_X, SYN_BLOCK_Y, SYN_BLOCK_LOOP_Y);
        m_synthesize.set(std::move(m_cl->buildResourceAsync(_T("RGY_FILTER_AFS_SYNTHESIZE_CL"), _T("EXE_DATA"), options.c_str())));
    }
    return RGY_ERR_NONE;
}

static RGY_ERR run_synthesize(uint8_t **dst,
    afsSourceCacheFrame *p0, afsSourceCacheFrame *p1, uint8_t *sip,
    const int width, const int height,
    const int *dstPitch, const int sipPitch,
    const int tb_order, const uint8_t status, const RGY_CSP csp,
    int mode,
    RGYOpenCLQueue &queue, RGYOpenCLProgram *synthesize, RGYOpenCLContext *cl) {
    auto err = RGY_ERR_NONE;

    if (mode < 0) {
        const RGYWorkSize local(SYN_BLOCK_INT_X, SYN_BLOCK_Y);
        const RGYWorkSize global(divCeil(width, 2), divCeil(height, 2));

        err = synthesize->kernel("kernel_synthesize_mode_tune").config(queue, local, global).launch(
            (cl_mem)dst[0], (cl_mem)dst[1], (cl_mem)dst[2], (cl_mem)sip,
            width, height, dstPitch[0], dstPitch[1], sipPitch,
            tb_order, status);
    } else if (mode == 0) {
        const RGYWorkSize local(SYN_BLOCK_INT_X, SYN_BLOCK_Y);
        const RGYWorkSize global(divCeil(width, 8), divCeil(height, 2));

        if (RGY_CSP_CHROMA_FORMAT[csp] == RGY_CHROMAFMT_YUV420) {
            err = synthesize->kernel("kernel_synthesize_mode_0").config(queue, local, global).launch(
                (cl_mem)dst[0], (cl_mem)dst[1], (cl_mem)dst[2],
                p0->y->mem(0), p0->cb[0]->mem(0), p0->cr[0]->mem(0),
                p1->y->mem(0), p1->cb[0]->mem(0), p1->cr[0]->mem(0),
                width, height, p0->y->frame.pitch[0], p0->cb[0]->frame.pitch[0], dstPitch[0], dstPitch[1],
                tb_order, status);
        } else if (RGY_CSP_CHROMA_FORMAT[csp] == RGY_CHROMAFMT_YUV444) {
            err = synthesize->kernel("kernel_synthesize_mode_0").config(queue, local, global).launch(
                (cl_mem)dst[0], (cl_mem)dst[1], (cl_mem)dst[2],
                p0->y->mem(0), p0->y->mem(1), p0->y->mem(2),
                p1->y->mem(0), p1->y->mem(1), p1->y->mem(2),
                width, height, p0->y->frame.pitch[0], p0->y->frame.pitch[1], dstPitch[0], dstPitch[1],
                tb_order, status);
        } else {
            return RGY_ERR_UNSUPPORTED;
        }
    } else {
        const RGYWorkSize local(SYN_BLOCK_INT_X, SYN_BLOCK_Y);
        const RGYWorkSize global(divCeil(width, 8), divCeil(height, 2));
        const int bit_depth = RGY_CSP_BIT_DEPTH[csp];

        if (RGY_CSP_CHROMA_FORMAT[csp] == RGY_CHROMAFMT_YUV420) {
            cl_mem texP0U0, texP0U1, texP0V0, texP0V1, texP1U0, texP1U1, texP1V0,texP1V1;
            if ((err = cl->createImageFromPlane(texP0U0, p0->cb[0]->mem(0), bit_depth, CL_R, true, p0->cb[0]->frame.pitch[0], p0->cb[0]->frame.width, p0->cb[0]->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
            if ((err = cl->createImageFromPlane(texP0U1, p0->cb[1]->mem(0), bit_depth, CL_R, true, p0->cb[1]->frame.pitch[0], p0->cb[1]->frame.width, p0->cb[1]->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
            if ((err = cl->createImageFromPlane(texP0V0, p0->cr[0]->mem(0), bit_depth, CL_R, true, p0->cr[0]->frame.pitch[0], p0->cr[0]->frame.width, p0->cr[0]->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
            if ((err = cl->createImageFromPlane(texP0V1, p0->cr[1]->mem(0), bit_depth, CL_R, true, p0->cr[1]->frame.pitch[0], p0->cr[1]->frame.width, p0->cr[1]->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
            if ((err = cl->createImageFromPlane(texP1U0, p1->cb[0]->mem(0), bit_depth, CL_R, true, p1->cb[0]->frame.pitch[0], p1->cb[0]->frame.width, p1->cb[0]->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
            if ((err = cl->createImageFromPlane(texP1U1, p1->cb[1]->mem(0), bit_depth, CL_R, true, p1->cb[1]->frame.pitch[0], p1->cb[1]->frame.width, p1->cb[1]->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
            if ((err = cl->createImageFromPlane(texP1V0, p1->cr[0]->mem(0), bit_depth, CL_R, true, p1->cr[0]->frame.pitch[0], p1->cr[0]->frame.width, p1->cr[0]->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;
            if ((err = cl->createImageFromPlane(texP1V1, p1->cr[1]->mem(0), bit_depth, CL_R, true, p1->cr[1]->frame.pitch[0], p1->cr[1]->frame.width, p1->cr[1]->frame.height, CL_MEM_READ_ONLY)) != RGY_ERR_NONE) return err;

            err = synthesize->kernel("kernel_synthesize_mode_1234_yuv420").config(queue, local, global).launch(
                (cl_mem)dst[0], (cl_mem)dst[1], (cl_mem)dst[2],
                p0->y->mem(0), p1->y->mem(0), (cl_mem)sip,
                texP0U0, texP0U1, texP1U0, texP1U1,
                texP0V0, texP0V1, texP1V0, texP1V1,
                width, height, p0->y->frame.pitch[0], dstPitch[0], dstPitch[1], sipPitch,
                tb_order, status);

            clReleaseMemObject(texP0U0);
            clReleaseMemObject(texP0U1);
            clReleaseMemObject(texP0V0);
            clReleaseMemObject(texP0V1);
            clReleaseMemObject(texP1U0);
            clReleaseMemObject(texP1U1);
            clReleaseMemObject(texP1V0);
            clReleaseMemObject(texP1V1);
        } else {
            err = synthesize->kernel("kernel_synthesize_mode_1234_yuv444").config(queue, local, global).launch(
                (cl_mem)dst[0], (cl_mem)dst[1], (cl_mem)dst[2],
                p0->y->mem(0), p0->y->mem(1), p0->y->mem(2),
                p1->y->mem(0), p1->y->mem(1), p1->y->mem(2),
                (cl_mem)sip,
                width, height,
                p0->y->frame.pitch[0], dstPitch[0], //Y/U/Vのpitchはすべて共通であることを前提とする
                sipPitch,
                tb_order, status);
        }
    }
    return err;
}

RGY_ERR RGYFilterAfs::synthesize(int iframe, RGYCLFrame *pOut, afsSourceCacheFrame *p0, afsSourceCacheFrame *p1, AFS_STRIPE_DATA *sip, const RGYFilterParamAfs *pAfsPrm, RGYOpenCLQueue &queue) {
    int mode = pAfsPrm->afs.analyze;
    if (pAfsPrm->afs.tune) {
        mode = -1;
    }
    if (   RGY_CSP_CHROMA_FORMAT[pOut->frame.csp] != RGY_CHROMAFMT_YUV420
        && RGY_CSP_CHROMA_FORMAT[pOut->frame.csp] != RGY_CHROMAFMT_YUV444) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid color format for afs synthesize: %s.\n"), RGY_CSP_NAMES[pOut->frame.csp]);
        return RGY_ERR_INVALID_COLOR_FORMAT;
    }
    if (mode >= 1 && RGY_CSP_CHROMA_FORMAT[pOut->frame.csp] == RGY_CHROMAFMT_YUV444) {
        if (   pOut->frame.pitch[0] != pOut->frame.pitch[1]
            || pOut->frame.pitch[0] != pOut->frame.pitch[2]) {
            AddMessage(RGY_LOG_ERROR, _T("Invalid pitch for afs synthesize dst.\n"));
            return RGY_ERR_INVALID_COLOR_FORMAT;
        }
        if (   p1->frameinfo().pitch[0] != p1->frameinfo().pitch[1]
            || p1->frameinfo().pitch[0] != p1->frameinfo().pitch[2]) {
            AddMessage(RGY_LOG_ERROR, _T("Invalid pitch for afs synthesize source.\n"));
            return RGY_ERR_INVALID_COLOR_FORMAT;
        }
    }
    auto err = run_synthesize(
        pOut->frame.ptr, p0, p1, sip->map->frame.ptr[0],
        p1->frameinfo().width, p1->frameinfo().height,
        pOut->frame.pitch, sip->map->frame.pitch[0],
        pAfsPrm->afs.tb_order, m_status[iframe], pOut->frame.csp, mode, queue, m_synthesize.get(), m_cl.get());
    return err;
}
