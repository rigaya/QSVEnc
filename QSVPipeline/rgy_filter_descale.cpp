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

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include "rgy_avutil.h"
#include "rgy_filter_descale.h"
#include "rgy_filter_input_probe.h"

static const int DESCALE_BLOCK = 32;

// --- LDLT init helpers (host CPU only) -----------------------------------

static inline double dsq(double x) { return x * x; }
static inline double dcb(double x) { return x * x * x; }

static inline double sinc(double x) {
    const double PI = 3.14159265358979323846;
    return (x == 0.0) ? 1.0 : std::sin(x * PI) / (x * PI);
}

// Canonical kernel weighting at a given distance from the sampled
// pixel position. distance is in destination-pixel units; the
// caller passes |xpos - pos|. The math below is the published
// reduction of each scaling kernel and is the same expression
// used by every independent implementation of these resamplers.
static double calculate_weight(VppDescaleKernel kernel, int support, double distance,
                               double b, double c) {
    distance = std::fabs(distance);
    switch (kernel) {
    case VppDescaleKernel::Bilinear:
        return std::max(1.0 - distance, 0.0);
    case VppDescaleKernel::Bicubic:
        if (distance < 1.0) {
            return ((12.0 - 9.0 * b - 6.0 * c) * dcb(distance)
                  + (-18.0 + 12.0 * b + 6.0 * c) * dsq(distance)
                  + (6.0 - 2.0 * b)) / 6.0;
        } else if (distance < 2.0) {
            return ((-b - 6.0 * c) * dcb(distance)
                  + (6.0 * b + 30.0 * c) * dsq(distance)
                  + (-12.0 * b - 48.0 * c) * distance
                  + (8.0 * b + 24.0 * c)) / 6.0;
        }
        return 0.0;
    case VppDescaleKernel::Lanczos2:
    case VppDescaleKernel::Lanczos3:
    case VppDescaleKernel::Lanczos4:
        return distance < support ? sinc(distance) * sinc(distance / support) : 0.0;
    case VppDescaleKernel::Auto:
        // Unreachable in normal flow; runProbe rewrites kernel to a
        // concrete variant before prepareCore is ever called. Returning
        // a zero weight here is a defensive no-op.
        return 0.0;
    case VppDescaleKernel::Spline16:
        if (distance < 1.0) {
            return 1.0 - (1.0 / 5.0 * distance) - (9.0 / 5.0 * dsq(distance)) + dcb(distance);
        } else if (distance < 2.0) {
            distance -= 1.0;
            return (-7.0 / 15.0 * distance) + (4.0 / 5.0 * dsq(distance)) - (1.0 / 3.0 * dcb(distance));
        }
        return 0.0;
    case VppDescaleKernel::Spline36:
        if (distance < 1.0) {
            return 1.0 - (3.0 / 209.0 * distance) - (453.0 / 209.0 * dsq(distance)) + (13.0 / 11.0 * dcb(distance));
        } else if (distance < 2.0) {
            distance -= 1.0;
            return (-156.0 / 209.0 * distance) + (270.0 / 209.0 * dsq(distance)) - (6.0 / 11.0 * dcb(distance));
        } else if (distance < 3.0) {
            distance -= 2.0;
            return (26.0 / 209.0 * distance) - (45.0 / 209.0 * dsq(distance)) + (1.0 / 11.0 * dcb(distance));
        }
        return 0.0;
    case VppDescaleKernel::Spline64:
        if (distance < 1.0) {
            return 1.0 - (3.0 / 2911.0 * distance) - (6387.0 / 2911.0 * dsq(distance)) + (49.0 / 41.0 * dcb(distance));
        } else if (distance < 2.0) {
            distance -= 1.0;
            return (-2328.0 / 2911.0 * distance) + (4032.0 / 2911.0 * dsq(distance)) - (24.0 / 41.0 * dcb(distance));
        } else if (distance < 3.0) {
            distance -= 2.0;
            return (582.0 / 2911.0 * distance) - (1008.0 / 2911.0 * dsq(distance)) + (6.0 / 41.0 * dcb(distance));
        } else if (distance < 4.0) {
            distance -= 3.0;
            return (-97.0 / 2911.0 * distance) + (168.0 / 2911.0 * dsq(distance)) - (1.0 / 41.0 * dcb(distance));
        }
        return 0.0;
    }
    return 0.0;
}

static inline double round_halfup(double x) {
    // Match the reference: invariant round(x - 1) == round(x) - 1 must
    // hold across the pixel grid, so half-to-even / half-away-from-zero
    // are not usable here.
    return (x < 0) ? std::floor(x + 0.5) : std::floor(x + 0.49999999999999994);
}

// Build the dense source-to-destination weights matrix
// (dst_dim rows by src_dim columns). Each row sums to 1 after
// normalization. border governs how out-of-bounds source positions
// fold back into the valid range.
static void build_scaling_weights(VppDescaleKernel kernel, int support,
                                  int src_dim, int dst_dim,
                                  double b, double c, double shift, double active_dim,
                                  VppDescaleBorder border,
                                  std::vector<double> &weights) {
    weights.assign((size_t)src_dim * dst_dim, 0.0);
    const double ratio = (double)dst_dim / active_dim;

    for (int i = 0; i < dst_dim; ++i) {
        double total = 0.0;
        const double pos = (i + 0.5) / ratio + shift;
        const double begin_pos = round_halfup(pos - support) + 0.5;
        for (int j = 0; j < 2 * support; ++j) {
            const double xpos = begin_pos + j;
            total += calculate_weight(kernel, support, xpos - pos, b, c);
        }
        for (int j = 0; j < 2 * support; ++j) {
            const double xpos = begin_pos + j;
            double real_pos = xpos;
            if (xpos < 0.0 || xpos > src_dim) {
                if (border == VppDescaleBorder::Zero) {
                    continue;
                } else if (border == VppDescaleBorder::Repeat) {
                    if (xpos < 0.0)
                        real_pos = 0.0;
                    else if (xpos >= src_dim)
                        real_pos = src_dim - 0.5;
                } else { // Mirror
                    if (xpos < 0.0)
                        real_pos = -xpos;
                    else if (xpos >= src_dim)
                        real_pos = std::min(2.0 * src_dim - xpos, (double)src_dim - 0.5);
                }
            }
            const int idx = (int)std::floor(real_pos);
            const double w = calculate_weight(kernel, support, xpos - pos, b, c) / total;
            weights[(size_t)i * src_dim + idx] += w;
        }
    }
}

static void transpose_matrix(int rows, int cols, const std::vector<double> &src, std::vector<double> &dst) {
    dst.assign((size_t)cols * rows, 0.0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[(size_t)j * rows + i] = src[(size_t)i * cols + j];
        }
    }
}

// Sparse times dense: A_t (dst_dim x src_dim) * A (src_dim x dst_dim).
// Only the columns in [lidx[i], ridx[i]) of A_t contribute to row i.
static void multiply_sparse_matrices(int dst_dim, int src_dim,
                                     const std::vector<int> &lidx, const std::vector<int> &ridx,
                                     const std::vector<double> &lm, const std::vector<double> &rm,
                                     std::vector<double> &out) {
    out.assign((size_t)dst_dim * dst_dim, 0.0);
    for (int i = 0; i < dst_dim; ++i) {
        for (int j = 0; j < dst_dim; ++j) {
            double sum = 0.0;
            for (int k = lidx[i]; k < ridx[i]; ++k) {
                sum += lm[(size_t)i * src_dim + k] * rm[(size_t)k * dst_dim + j];
            }
            out[(size_t)i * dst_dim + j] = sum;
        }
    }
}

// In-place LDLT decomposition of a banded symmetric matrix. Only the
// upper triangle is read; result overwrites the upper triangle with
// L' and D. The unit diagonal of L' is not stored.
static void banded_ldlt(int n, int bandwidth, std::vector<double> &mat) {
    const int half = bandwidth / 2;
    const double eps = DBL_EPSILON;
    for (int i = 0; i < n; ++i) {
        const int end = std::min(half + 1, n - i);
        for (int j = 1; j < end; ++j) {
            const double d = mat[(size_t)i * n + i + j] / (mat[(size_t)i * n + i] + eps);
            for (int k = 0; k < end - j; ++k) {
                mat[(size_t)(i + j) * n + i + j + k] -= d * mat[(size_t)i * n + i + j + k];
            }
        }
        const double e = 1.0 / (mat[(size_t)i * n + i] + eps);
        for (int j = 1; j < end; ++j) {
            mat[(size_t)i * n + i + j] *= e;
        }
    }
}

static void multiply_banded_with_diagonal(int n, int bandwidth, std::vector<double> &mat) {
    const int half = bandwidth / 2;
    for (int i = 1; i < n; ++i) {
        const int start = std::max(i - half, 0);
        for (int j = start; j < i; ++j) {
            mat[(size_t)i * n + j] *= mat[(size_t)j * n + j];
        }
    }
}

// Pack the post-LDLT banded representation into c compact arrays
// each of length n for the lower and upper triangles, plus the
// reciprocal of the diagonal.
static void pack_lower_upper_diag(int n, int bandwidth,
                                  const std::vector<double> &lower_full,
                                  const std::vector<double> &upper_full,
                                  std::vector<float> &lower_packed,
                                  std::vector<float> &upper_packed,
                                  std::vector<float> &diagonal) {
    const int half = bandwidth / 2;
    const double eps = DBL_EPSILON;
    lower_packed.assign((size_t)half * n, 0.0f);
    upper_packed.assign((size_t)half * n, 0.0f);
    diagonal.assign(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        const int start = std::max(i - half, 0);
        for (int j = start; j < i; ++j) {
            lower_packed[(size_t)(j - i + half) * n + i] = (float)lower_full[(size_t)i * n + j];
        }
    }
    for (int i = 0; i < n; ++i) {
        const int start = std::min(i + half, n - 1);
        for (int j = start; j > i; --j) {
            upper_packed[(size_t)(j - i - 1) * n + i] = (float)upper_full[(size_t)i * n + j];
        }
    }
    for (int i = 0; i < n; ++i) {
        diagonal[i] = (float)(1.0 / (lower_full[(size_t)i * n + i] + eps));
    }
}

static int kernel_support(VppDescaleKernel kernel) {
    switch (kernel) {
    case VppDescaleKernel::Bilinear: return 1;
    case VppDescaleKernel::Bicubic:  return 2;
    case VppDescaleKernel::Spline16: return 2;
    case VppDescaleKernel::Spline36: return 3;
    case VppDescaleKernel::Spline64: return 4;
    case VppDescaleKernel::Lanczos2: return 2;
    case VppDescaleKernel::Lanczos3: return 3;
    case VppDescaleKernel::Lanczos4: return 4;
    case VppDescaleKernel::Auto:     return 2; // unreachable in normal path; runProbe rewrites kernel before prepareCore
    }
    return 2;
}

// --- Filter class --------------------------------------------------------

RGYFilterDescale::RGYFilterDescale(shared_ptr<RGYOpenCLContext> context)
    : RGYFilter(context), m_descale(), m_cores(), m_intermediateH(), m_intermediateV(),
      m_intermediatePitchFloats{}, m_frameIdx(0) {
    m_name = _T("descale");
}

RGYFilterDescale::~RGYFilterDescale() {
    close();
}

// Build forward-upscale weight tables in compressed-by-high-res-row
// form (the natural shape for y = A x where A is high x low).
RGY_ERR RGYFilterDescale::buildForwardWeights(ProbeForwardWeights &fw,
                                              int src_dim_low, int dst_dim_high,
                                              VppDescaleKernel kernel, double b, double c_param,
                                              double shift, VppDescaleBorder border) {
    const int support = kernel_support(kernel);
    if (support <= 0 || src_dim_low <= 0 || dst_dim_high <= src_dim_low) {
        return RGY_ERR_INVALID_PARAM;
    }
    // build_scaling_weights produces a dense (dst_dim_high x src_dim_low)
    // matrix where row i has the upscale weights for high-res pixel i.
    std::vector<double> dense;
    build_scaling_weights(kernel, support, src_dim_low, dst_dim_high,
                          b, c_param, shift, (double)src_dim_low, border, dense);

    std::vector<int> lidx(dst_dim_high, 0), ridx(dst_dim_high, 0);
    int maxw = 0;
    for (int i = 0; i < dst_dim_high; ++i) {
        int lj = 0;
        for (int j = 0; j < src_dim_low; ++j) {
            if (dense[(size_t)i * src_dim_low + j] != 0.0) { lj = j; break; }
        }
        int rj = 0;
        for (int j = src_dim_low - 1; j >= 0; --j) {
            if (dense[(size_t)i * src_dim_low + j] != 0.0) { rj = j + 1; break; }
        }
        lidx[i] = lj; ridx[i] = rj;
        maxw = std::max(maxw, rj - lj);
    }
    fw.weights_columns = maxw;
    std::vector<float> packed((size_t)dst_dim_high * maxw, 0.0f);
    for (int i = 0; i < dst_dim_high; ++i) {
        for (int j = 0; j < ridx[i] - lidx[i]; ++j) {
            packed[(size_t)i * maxw + j] = (float)dense[(size_t)i * src_dim_low + lidx[i] + j];
        }
    }
    fw.weights   = m_cl->createBuffer(packed.size() * sizeof(float), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, packed.data());
    fw.left_idx  = m_cl->createBuffer(lidx.size()   * sizeof(int),   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, lidx.data());
    fw.right_idx = m_cl->createBuffer(ridx.size()   * sizeof(int),   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ridx.data());
    if (!fw.weights || !fw.left_idx || !fw.right_idx) return RGY_ERR_MEMORY_ALLOC;
    return RGY_ERR_NONE;
}

namespace {
// Auto-detect kernel catalogue. Each row is { kernel, b, c } where
// b / c are only consulted when kernel == Bicubic. The 4 bicubic rows
// cover the common parameter pairs (Catmull-Rom, Mitchell-Netravali,
// sharp-bicubic, and the default). In kernel-only mode runProbe scores
// every row at the user-supplied dims; in resolution-search mode each
// candidate height generates one entry per row of this table.
struct DescaleAutoCandidate {
    VppDescaleKernel kernel;
    float b;
    float c;
    const TCHAR *label;
};
static const DescaleAutoCandidate kDescaleAutoCandidates[] = {
    { VppDescaleKernel::Bilinear, 0.0f,    0.0f,    _T("bilinear")             },
    { VppDescaleKernel::Bicubic,  0.0f,    0.5f,    _T("bicubic(b=0,c=0.5)")   },
    { VppDescaleKernel::Bicubic,  1.f/3.f, 1.f/3.f, _T("bicubic(b=1/3,c=1/3)") },
    { VppDescaleKernel::Bicubic,  0.0f,    0.75f,   _T("bicubic(b=0,c=0.75)")  },
    { VppDescaleKernel::Spline16, 0.0f,    0.0f,    _T("spline16")             },
    { VppDescaleKernel::Spline36, 0.0f,    0.0f,    _T("spline36")             },
    { VppDescaleKernel::Spline64, 0.0f,    0.0f,    _T("spline64")             },
    { VppDescaleKernel::Lanczos2, 0.0f,    0.0f,    _T("lanczos2")             },
    { VppDescaleKernel::Lanczos3, 0.0f,    0.0f,    _T("lanczos3")             },
    { VppDescaleKernel::Lanczos4, 0.0f,    0.0f,    _T("lanczos4")             },
};
static const int kAutoCandCount = (int)(sizeof(kDescaleAutoCandidates) / sizeof(kDescaleAutoCandidates[0]));

// Resolution-search Pass 0 fast-path: nine well-known native heights.
// These cover the overwhelming majority of TV-distribution upscales
// (SD broadcast, DVD, anime 480p/540p/576p/720p/810p/900p sources,
// HD masters). For a given source the in-range subset is what
// actually gets probed.
static const int kCommonNativeHeights[] = { 360, 480, 486, 540, 576, 720, 810, 900, 1080 };
static const int kCommonNativeHeightsCount =
    (int)(sizeof(kCommonNativeHeights) / sizeof(kCommonNativeHeights[0]));

// Three "representative" kernels used by Pass 1 coarse sweep. The
// idea is that Catmull-Rom / lanczos3 / spline36 between them span
// the response shapes the rest of the catalogue is close to: if
// the source was upscaled with bilinear, Catmull-Rom comes very
// close; if with spline64, spline36 comes close; if with lanczos4,
// lanczos3 comes close. So the coarse-pass winner reliably
// localises the height bucket even with only three kernels probed.
struct CoarseRepKernel { VppDescaleKernel kernel; float b; float c; };
static const CoarseRepKernel kCoarseRepKernels[] = {
    { VppDescaleKernel::Bicubic, 0.0f, 0.5f },   // Catmull-Rom
    { VppDescaleKernel::Lanczos3, 0.0f, 0.0f },
    { VppDescaleKernel::Spline36, 0.0f, 0.0f },
};
static const int kCoarseRepKernelsCount =
    (int)(sizeof(kCoarseRepKernels) / sizeof(kCoarseRepKernels[0]));

// Round x up to the nearest even integer. Both height and width
// candidates need to be even to remain compatible with the
// downstream YV12 / NV12 csp constraints.
static inline int round_up_even(int x) {
    return (x + 1) & ~1;
}

// Derive the width for a candidate height by maintaining the source
// Probe the source's SAR through the three places libavformat may
// expose it. Order matters: codecpar carries the VUI-encoded SAR
// (e.g. 134:135 for 16:9 anamorphic anime sources) and is the
// canonical reference used everywhere else in QSVEnc (see
// rgy_input_avcodec.cpp:2260). The stream-level AVStream::
// sample_aspect_ratio is sometimes set by demuxers to a rounded /
// less-precise pixel ratio (e.g. 1072:1073) that defeats DAR
// derivation, so it sits behind codecpar in the chain.
// av_guess_sample_aspect_ratio is the last-resort heuristic.
// Returns SAR = {0, 0} when nothing is known so the caller can
// fall back to the raw pixel ratio.
static inline AVRational resolve_source_sar(AVFormatContext *fmtCtx, AVStream *videoStream,
                                             const TCHAR **outSrcName) {
    AVRational sar = { 0, 0 };
    const TCHAR *src = _T("none");
    if (videoStream) {
        if (videoStream->codecpar) {
            sar = videoStream->codecpar->sample_aspect_ratio;
            if (sar.num > 0 && sar.den > 0) { src = _T("codecpar"); goto done; }
        }
        sar = videoStream->sample_aspect_ratio;
        if (sar.num > 0 && sar.den > 0) { src = _T("stream"); goto done; }
        sar = av_guess_sample_aspect_ratio(fmtCtx, videoStream, nullptr);
        if (sar.num > 0 && sar.den > 0) { src = _T("av_guess"); goto done; }
        sar = AVRational{ 0, 0 };
    }
done:
    if (outSrcName) *outSrcName = src;
    return sar;
}

// aspect ratio. Result is rounded to the nearest even pixel so the
// downstream chroma planes still align.
//
// DAR-aware: when the stream carries a non-1:1 sample aspect ratio
// (e.g. 1920x1072 SAR 134:135 -> DAR 16:9), the raw pixel ratio
// src_w / src_h derives a width that's a few pixels off from the
// user's actual native target (1290 instead of 1280 for 720p). SAR
// is read via the resolve_source_sar() chain above. When SAR is
// unknown, falls back to the raw pixel ratio.
static inline int width_from_height(int src_w, int src_h, int height,
                                    AVFormatContext *fmtCtx, AVStream *videoStream) {
    if (src_h <= 0) return 0;
    AVRational dar;
    AVRational sar = resolve_source_sar(fmtCtx, videoStream, nullptr);
    if (sar.num > 0 && sar.den > 0) {
        dar = av_mul_q(sar, av_make_q(src_w, src_h));
        av_reduce(&dar.num, &dar.den, dar.num, dar.den, 65536);
    } else {
        dar = av_make_q(src_w, src_h);
    }
    // Snap the derived DAR to the nearest standard display ratio when
    // the SAR-encoded DAR sits within 1% of a canonical value.
    // Source rationale: some encoders ship "essentially square" SARs
    // like 3427:3429 that, multiplied through to a DAR of ~1.79008,
    // miss the clean 16:9 (1.77778) target by ~0.7%. Multiplying that
    // DAR back out at height=720 gives width=1290 instead of the
    // mod-16 1280 the source actually targets, and the probe then
    // wastes candidates around a fractionally-off width. Snapping
    // recovers the clean target. Threshold 1% (relative) cleanly
    // separates 16:9 from 1.85:1 (4.0% apart) and from 21:9 / 2:1
    // (much further), so no false collisions on legitimate cinema
    // aspect ratios. Non-standard DARs (anamorphic specialty,
    // intentional crops) fall through to the raw computation.
    {
        static const AVRational kStandardDARs[] = {
            { 4,  3 },
            { 16, 9 },
            { 21, 9 },
            { 1,  1 },
            { 2,  1 },
        };
        const double darF = (double)dar.num / (double)dar.den;
        for (const auto &sd : kStandardDARs) {
            const double sdF = (double)sd.num / (double)sd.den;
            if (std::fabs(darF - sdF) / sdF < 0.01) {
                dar = sd;
                break;
            }
        }
    }
    const int w = (int)std::lround((double)height * (double)dar.num / (double)dar.den);
    return round_up_even(w);
}

// Precision approach (load-bearing for cross-resolution candidate
// ranking): the GPU MSE kernel uses Kahan compensated summation in
// float, since cl_khr_fp64 is not available on the Arc A770 OpenCL
// driver. Compensated summation is single-ULP-accurate for the row
// sum even when squared differences are near the float32
// epsilon-squared floor (~1e-14). Host then accumulates row sums in
// double to preserve the remaining precision across the full frame
// and across multiple probe frames.
//
// Labels are stored as tstring (== basic_string<TCHAR>) so the %s
// specifier in _T(...) format strings matches whatever character
// width the build was compiled for.

static tstring probe_label_for(VppDescaleKernel k, float b, float c) {
    switch (k) {
    case VppDescaleKernel::Bilinear: return _T("bilinear");
    case VppDescaleKernel::Bicubic:  return strsprintf(_T("bicubic(b=%.3f,c=%.3f)"), b, c);
    case VppDescaleKernel::Spline16: return _T("spline16");
    case VppDescaleKernel::Spline36: return _T("spline36");
    case VppDescaleKernel::Spline64: return _T("spline64");
    case VppDescaleKernel::Lanczos2: return _T("lanczos2");
    case VppDescaleKernel::Lanczos3: return _T("lanczos3");
    case VppDescaleKernel::Lanczos4: return _T("lanczos4");
    case VppDescaleKernel::Auto:     return _T("auto");
    }
    return _T("?");
}

} // anonymous namespace

RGY_ERR RGYFilterDescale::scoreCandidates(std::vector<ProbeCandidate> &candidates,
                                          const std::vector<std::unique_ptr<RGYCLBuf>> &lumaBufs,
                                          const std::vector<std::unique_ptr<RGYCLBuf>> &edgeWeightsBufs,
                                          int src_w, int src_h, int src_pixel_bytes,
                                          bool symmetricForward) {
    if (lumaBufs.empty()) return RGY_ERR_INVALID_PARAM;
    if (edgeWeightsBufs.size() != lumaBufs.size()) return RGY_ERR_INVALID_PARAM;
    RGYOpenCLQueue &queue = m_cl->queue();
    const int src_pitch_bytes = src_w * src_pixel_bytes;
    // Edge-weights buffers were written with no padding (pitch == src_w
    // in floats). kernel_descale_mse needs the float-element pitch.
    const int edge_pitch_floats = src_w;

    int candIdx = 0;
    for (auto &c : candidates) {
        AddMessage(RGY_LOG_DEBUG, _T("probe: candidate %d/%d %s %dx%d\n"),
            candIdx + 1, (int)candidates.size(), c.label.c_str(), c.width, c.height);

        // Defensive: a candidate with degenerate dims (>= source) can
        // never produce a valid LDLT system. Skip with a large MSE so
        // it sorts to the bottom.
        if (c.width <= 0 || c.height <= 0 || c.width >= src_w || c.height >= src_h) {
            c.mse = 1e30; candIdx++; continue;
        }

        RGYFilterDescaleCore coreH, coreV;
        if (prepareCore(coreH, src_w, c.width,  c.kernel, c.b, c.c, 0.0, VppDescaleBorder::Mirror) != RGY_ERR_NONE
         || prepareCore(coreV, src_h, c.height, c.kernel, c.b, c.c, 0.0, VppDescaleBorder::Mirror) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_WARN, _T("probe: prepareCore failed for %s %dx%d, skipping.\n"),
                c.label.c_str(), c.width, c.height);
            c.mse = 1e30; candIdx++; continue;
        }
        ProbeForwardWeights fwH, fwV;
        // Forward (re-upscale) kernel selection. Default (Stage 1,
        // resolution detection): fixed Catmull-Rom baseline so the
        // candidate kernel under test does not reinject its own
        // ringing into the reconstruction and bias the height pick.
        // Symmetric mode (Stage 2, kernel tie-break at locked
        // dimensions): forward kernel == candidate descale kernel,
        // restoring the natural inverse-problem pairing for the
        // kernel argmin alone.
        const VppDescaleKernel fwKernel = symmetricForward ? c.kernel              : VppDescaleKernel::Bicubic;
        const float            fwB      = symmetricForward ? c.b                   : 0.0f;
        const float            fwC      = symmetricForward ? c.c                   : 0.5f;
        if (buildForwardWeights(fwH, c.width,  src_w, fwKernel, fwB, fwC, 0.0, VppDescaleBorder::Mirror) != RGY_ERR_NONE
         || buildForwardWeights(fwV, c.height, src_h, fwKernel, fwB, fwC, 0.0, VppDescaleBorder::Mirror) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_WARN, _T("probe: buildForwardWeights failed for %s %dx%d, skipping.\n"),
                c.label.c_str(), c.width, c.height);
            c.mse = 1e30; candIdx++; continue;
        }
        auto bufDescaleH = m_cl->createBuffer((size_t)c.width * src_h    * sizeof(float),  CL_MEM_READ_WRITE);
        auto bufDescaleV = m_cl->createBuffer((size_t)c.width * c.height * sizeof(float),  CL_MEM_READ_WRITE);
        auto bufRescaleH = m_cl->createBuffer((size_t)src_w   * c.height * sizeof(float),  CL_MEM_READ_WRITE);
        auto bufReupV    = m_cl->createBuffer((size_t)src_w   * src_h    * sizeof(float),  CL_MEM_READ_WRITE);
        auto bufRowSums  = m_cl->createBuffer((size_t)src_h   * sizeof(float),             CL_MEM_READ_WRITE);
        if (!bufDescaleH || !bufDescaleV || !bufRescaleH || !bufReupV || !bufRowSums) {
            AddMessage(RGY_LOG_WARN, _T("probe: scratch allocation failed for %s %dx%d, skipping.\n"),
                c.label.c_str(), c.width, c.height);
            c.mse = 1e30; candIdx++; continue;
        }

        double accumSSD = 0.0;
        for (size_t fi = 0; fi < lumaBufs.size(); ++fi) {
            const auto &luma_buf = lumaBufs[fi];
            const auto &edge_buf = edgeWeightsBufs[fi];
            {
                RGYWorkSize local(32, 1);
                RGYWorkSize global(src_h, 1);
                m_descale.get()->kernel("kernel_descale_h").config(queue, local, global, {}, nullptr).launch(
                    bufDescaleH->mem(), c.width,
                    luma_buf->mem(), src_pitch_bytes,
                    src_h, c.width,
                    coreH.c, coreH.weights_columns,
                    coreH.weights->mem(), coreH.left_idx->mem(), coreH.right_idx->mem(),
                    coreH.lower->mem(), coreH.upper->mem(), coreH.diagonal->mem());
            }
            {
                RGYWorkSize local(32, 1);
                RGYWorkSize global(c.width, 1);
                // pDst stub: reuse bufDescaleV's cl_mem (the kernel never
                // touches pDst when writeIntegerOutput=0). pitch is also
                // unused in that path. The probe consumes bufDescaleV
                // directly for the rescale step, so the dead integer
                // write that used to land in a dedicated bufDstDummy
                // is now skipped entirely on the device.
                m_descale.get()->kernel("kernel_descale_v").config(queue, local, global, {}, nullptr).launch(
                    bufDescaleV->mem(), c.width * src_pixel_bytes,
                    bufDescaleV->mem(), c.width,
                    bufDescaleH->mem(), c.width,
                    src_h, c.width, c.height,
                    coreV.c, coreV.weights_columns,
                    coreV.weights->mem(), coreV.left_idx->mem(), coreV.right_idx->mem(),
                    coreV.lower->mem(), coreV.upper->mem(), coreV.diagonal->mem(),
                    0 /* writeIntegerOutput: probe path skips the integer quantise */);
            }
            {
                RGYWorkSize local(32, 8);
                RGYWorkSize global(src_w, c.height);
                m_descale.get()->kernel("kernel_rescale_h").config(queue, local, global, {}, nullptr).launch(
                    bufRescaleH->mem(), src_w,
                    bufDescaleV->mem(), c.width,
                    src_w, c.height, c.width,
                    fwH.weights_columns,
                    fwH.weights->mem(), fwH.left_idx->mem(), fwH.right_idx->mem());
            }
            {
                RGYWorkSize local(32, 8);
                RGYWorkSize global(src_w, src_h);
                m_descale.get()->kernel("kernel_rescale_v").config(queue, local, global, {}, nullptr).launch(
                    bufReupV->mem(), src_w,
                    bufRescaleH->mem(), src_w,
                    src_w, src_h, c.height,
                    fwV.weights_columns,
                    fwV.weights->mem(), fwV.left_idx->mem(), fwV.right_idx->mem());
            }
            {
                RGYWorkSize local(32, 1);
                RGYWorkSize global(src_h, 1);
                m_descale.get()->kernel("kernel_descale_mse").config(queue, local, global, {}, nullptr).launch(
                    bufRowSums->mem(),
                    luma_buf->mem(), src_pitch_bytes,
                    bufReupV->mem(), src_w,
                    edge_buf->mem(), edge_pitch_floats,
                    src_w, src_h);
            }
            std::vector<float> rowSums(src_h);
            queue.finish();
            cl_int rdErr = clEnqueueReadBuffer(queue.get(), bufRowSums->mem(), CL_TRUE,
                                               0, src_h * sizeof(float), rowSums.data(),
                                               0, nullptr, nullptr);
            if (rdErr != CL_SUCCESS) {
                AddMessage(RGY_LOG_WARN, _T("probe: clEnqueueReadBuffer failed (err=%d) for %s, skipping rest of frames.\n"),
                    (int)rdErr, c.label.c_str());
                accumSSD = 1e30; break;
            }
            double frameSSD = 0.0;
            for (float s : rowSums) frameSSD += (double)s;
            accumSSD += frameSSD;
        }
        c.mse = accumSSD / ((double)src_w * src_h * lumaBufs.size());
        AddMessage(RGY_LOG_DEBUG, _T("probe: candidate %d/%d %s %dx%d mse=%.6e\n"),
            candIdx + 1, (int)candidates.size(), c.label.c_str(), c.width, c.height, c.mse);
        candIdx++;
    }
    return RGY_ERR_NONE;
}

// Three-pass coarse-to-fine resolution search.
//
// Pass 0 (fast-path): every common native height in
//   {360, 480, 486, 540, 576, 720, 810, 900, 1080} that falls inside
//   [search_min, search_max], paired with all 10 catalogue kernels.
//   For sources whose native is one of these heights the winner here
//   is decisive.
//
// Pass 1 (coarse): a stride-of-(search_step * 8) sweep across the
//   full [search_min, search_max] range, paired with three
//   representative kernels (Catmull-Rom, lanczos3, spline36). Picks
//   the best height bucket when Pass 0 didn't already nail the
//   resolution (typical for off-spec sources at 850p, 944p, etc.).
//
// Pass 2 (fine): a stride-of-search_step sweep over
//   [bucket_h - 8*step, bucket_h + 8*step], paired with all 10
//   catalogue kernels. Final lock.
//
// All passes append into `candidates` so the caller can sort the
// combined set and pick the lowest-MSE entry. Cores are torn down
// inside scoreCandidates() between candidates so VRAM stays bounded
// regardless of how many entries the search produces.
RGY_ERR RGYFilterDescale::runResolutionSearch(RGYFilterParamDescale *prm,
                                              std::vector<ProbeCandidate> &candidates,
                                              const std::vector<std::unique_ptr<RGYCLBuf>> &lumaBufs,
                                              const std::vector<std::unique_ptr<RGYCLBuf>> &edgeWeightsBufs,
                                              int src_w, int src_h, int src_pixel_bytes,
                                              AVFormatContext *fmtCtx, AVStream *videoStream) {
    // Resolve search range. Defaults: [0.48 * src_h, 0.85 * src_h].
    // The upper bound deliberately excludes the near-identity regime
    // where the descale -> re-upscale residual collapses to noise.
    //
    // 0.48 * src_h keeps the floor above 480p for all sources with
    // src_h > 1000 (720p-native anime typically encoded at
    // 1080p / 1072p). For SD-upscaled sources (src_h ~ 960) floor
    // drops to ~461, exposing the native 480p cliff. Below 0.45 the
    // floor can drop under 480 for tall sources and trigger a false
    // 480p peak via DVD-master attractor artifacts.
    int searchMin = prm->descale.search_min > 0 ? prm->descale.search_min : (int)(src_h * 0.48);
    int searchMax = prm->descale.search_max > 0 ? prm->descale.search_max : (int)(src_h * 0.85);
    if (searchMin < 2) searchMin = 2;
    if (searchMax >= src_h) searchMax = src_h - 1;
    if (searchMin > searchMax) {
        AddMessage(RGY_LOG_ERROR,
            _T("auto-detect: search range invalid (search_min=%d > search_max=%d).\n"),
            searchMin, searchMax);
        return RGY_ERR_INVALID_PARAM;
    }
    const int searchStep = std::max(1, prm->descale.search_step);
    const int coarseStep = std::max(2, searchStep * 8);

    // Resolve SAR/DAR once for diagnostic logging - same chain as
    // width_from_height() uses internally. Cached as outer locals so
    // the per-candidate append_candidate lambda can echo them into
    // its first-seen-height debug log without re-invoking the chain.
    const TCHAR *sarSrc = nullptr;
    const AVRational diagSar = resolve_source_sar(fmtCtx, videoStream, &sarSrc);
    AVRational diagDar = av_make_q(src_w, src_h);
    if (diagSar.num > 0 && diagSar.den > 0) {
        diagDar = av_mul_q(diagSar, av_make_q(src_w, src_h));
        av_reduce(&diagDar.num, &diagDar.den, diagDar.num, diagDar.den, 65536);
        AddMessage(RGY_LOG_DEBUG,
            _T("probe: SAR %d:%d from %s, derived DAR %d:%d for width calc.\n"),
            diagSar.num, diagSar.den, sarSrc, diagDar.num, diagDar.den);
    } else {
        AddMessage(RGY_LOG_DEBUG,
            _T("probe: SAR not found in any libav source; falling back to pixel ratio %d:%d.\n"),
            src_w, src_h);
    }
    std::set<int> loggedHeights;  // dedupe per-height width-derivation log
    const bool sarValid = (diagSar.num > 0 && diagSar.den > 0);

    auto append_candidate = [&](VppDescaleKernel k, float b, float c, int height) {
        const int h = round_up_even(height);
        if (h < searchMin || h > searchMax) return;
        const int w = width_from_height(src_w, src_h, h, fmtCtx, videoStream);
        if (w <= 0 || w >= src_w) return;
        // Per-unique-height width-derivation diagnostic. Logs which
        // SAR source produced the DAR used (or the pixel-ratio
        // fallback), the rationals involved, and the resulting
        // width. Deduped via loggedHeights so we get one line per
        // distinct height across all passes, not one per candidate.
        if (loggedHeights.insert(h).second) {
            if (sarValid) {
                AddMessage(RGY_LOG_DEBUG,
                    _T("probe: width_from_height h=%d via SAR %d:%d (%s) -> DAR %d:%d -> width=%d.\n"),
                    h, diagSar.num, diagSar.den, sarSrc, diagDar.num, diagDar.den, w);
            } else {
                AddMessage(RGY_LOG_DEBUG,
                    _T("probe: width_from_height h=%d via pixel ratio %d:%d -> width=%d.\n"),
                    h, src_w, src_h, w);
            }
        }
        // Skip duplicates (same kernel + same (w,h)).
        for (const auto &existing : candidates) {
            if (existing.kernel == k && existing.b == b && existing.c == c
                && existing.width == w && existing.height == h) return;
        }
        candidates.push_back(ProbeCandidate{ k, b, c, w, h, 0.0,
            probe_label_for(k, b, c) });
    };

    // ---- Pass 0: common heights x all 10 kernels ----------------------
    const size_t pass0Begin = candidates.size();
    for (int i = 0; i < kCommonNativeHeightsCount; ++i) {
        const int h = kCommonNativeHeights[i];
        for (const auto &kv : kDescaleAutoCandidates) {
            append_candidate(kv.kernel, kv.b, kv.c, h);
        }
    }
    const size_t pass0End = candidates.size();
    if (pass0Begin == pass0End) {
        AddMessage(RGY_LOG_ERROR,
            _T("auto-detect: no common native heights fall within search range [%d, %d].\n"),
            searchMin, searchMax);
        return RGY_ERR_INVALID_PARAM;
    }
    AddMessage(RGY_LOG_DEBUG, _T("probe: Pass 0 - %d common-height candidates.\n"),
        (int)(pass0End - pass0Begin));
    {
        auto err = scoreCandidates(candidates, lumaBufs, edgeWeightsBufs, src_w, src_h, src_pixel_bytes);
        if (err != RGY_ERR_NONE) return err;
    }

    // ---- Pass 1: coarse stride x 3 representative kernels --------------
    const size_t pass1Begin = candidates.size();
    for (int h = searchMin; h <= searchMax; h += coarseStep) {
        for (int k = 0; k < kCoarseRepKernelsCount; ++k) {
            append_candidate(kCoarseRepKernels[k].kernel,
                             kCoarseRepKernels[k].b,
                             kCoarseRepKernels[k].c, h);
        }
    }
    const size_t pass1End = candidates.size();
    AddMessage(RGY_LOG_DEBUG, _T("probe: Pass 1 - %d coarse-stride candidates (step=%d).\n"),
        (int)(pass1End - pass1Begin), coarseStep);
    if (pass1End > pass1Begin) {
        // Score only the new Pass 1 candidates by copying them into a
        // scratch vector, scoring it, then writing the MSEs back.
        // scoreCandidates writes into its argument's .mse field, so we
        // need the temporary to avoid disturbing the already-scored
        // Pass 0 entries.
        std::vector<ProbeCandidate> tail(candidates.begin() + pass1Begin, candidates.end());
        auto err = scoreCandidates(tail, lumaBufs, edgeWeightsBufs, src_w, src_h, src_pixel_bytes);
        if (err != RGY_ERR_NONE) return err;
        // Copy scored MSEs back into candidates.
        for (size_t i = 0; i < tail.size(); ++i) {
            candidates[pass1Begin + i].mse = tail[i].mse;
        }
    }

    // ---- Pass 2: fine refinement around the best height so far --------
    // Determine the current best height across Pass 0 + Pass 1.
    int bestH = 0;
    double bestMse = 1e30;
    for (const auto &c : candidates) {
        if (std::isfinite(c.mse) && c.mse >= 0.0 && c.mse < bestMse) {
            bestMse = c.mse;
            bestH = c.height;
        }
    }
    if (bestH > 0) {
        const int loH = std::max(searchMin, bestH - coarseStep);
        const int hiH = std::min(searchMax, bestH + coarseStep);
        const size_t pass2Begin = candidates.size();
        for (int h = loH; h <= hiH; h += searchStep) {
            for (const auto &kv : kDescaleAutoCandidates) {
                append_candidate(kv.kernel, kv.b, kv.c, h);
            }
        }
        const size_t pass2End = candidates.size();
        AddMessage(RGY_LOG_DEBUG,
            _T("probe: Pass 2 - %d fine-refine candidates around best %dp.\n"),
            (int)(pass2End - pass2Begin), bestH);
        if (pass2End > pass2Begin) {
            std::vector<ProbeCandidate> tail(candidates.begin() + pass2Begin, candidates.end());
            auto err = scoreCandidates(tail, lumaBufs, edgeWeightsBufs, src_w, src_h, src_pixel_bytes);
            if (err != RGY_ERR_NONE) return err;
            for (size_t i = 0; i < tail.size(); ++i) {
                candidates[pass2Begin + i].mse = tail[i].mse;
            }
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDescale::runProbe(RGYFilterParamDescale *prm) {
    if (!prm) return RGY_ERR_NULL_PTR;
    // Mode selection:
    //   width>0 && height>0          -> kernel-only auto-detect (10 kernels scored at the supplied dims)
    //   width==0 && height==0        -> resolution + kernel auto-detect (3-pass pyramid)
    //   exactly one of width/height set -> ambiguous, error
    if ((prm->descale.width > 0) != (prm->descale.height > 0)) {
        AddMessage(RGY_LOG_ERROR,
            _T("auto-detect needs both width= and height= set, or neither.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->inputFilePath.empty()) {
        AddMessage(RGY_LOG_ERROR,
            _T("auto-detect requires a re-openable input file (got empty path).\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const auto probeStart = std::chrono::steady_clock::now();

    // The kernel program must already be built (init() does this
    // before calling runProbe).
    if (!m_descale.get()) {
        AddMessage(RGY_LOG_ERROR, _T("auto-detect: kernel program not loaded.\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }

    const int src_w = prm->frameIn.width;
    const int src_h = prm->frameIn.height;
    const int dst_w = prm->descale.width;
    const int dst_h = prm->descale.height;
    const int bit_depth = RGY_CSP_BIT_DEPTH[prm->frameIn.csp];
    const int src_pixel_bytes = (bit_depth > 8) ? 2 : 1;

    // ---- open private libavcodec context --------------------------------
    std::string fileUtf8;
    if (tchar_to_string(prm->inputFilePath.c_str(), fileUtf8, CP_UTF8) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("probe: failed to convert filename to utf-8.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (const auto protocol = unsupportedProbeProtocol(fileUtf8); protocol != nullptr) {
        AddMessage(RGY_LOG_ERROR,
            _T("auto-detect requires a re-openable input file, but input protocol is %s.\n")
            _T("    Please pass kernel=<concrete>,width=<int>,height=<int> explicitly.\n"),
            char_to_tstring(protocol).c_str());
        return RGY_ERR_UNSUPPORTED;
    }

    const int savedAvLogLevel = av_log_get_level();
    av_log_set_level(AV_LOG_FATAL);
    struct AvLogLevelRestorer { int prev; ~AvLogLevelRestorer() { av_log_set_level(prev); } } avGuard{savedAvLogLevel};

    AVFormatContext *fmtCtxRaw = nullptr;
    if (avformat_open_input(&fmtCtxRaw, fileUtf8.c_str(), nullptr, nullptr) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("probe: avformat_open_input failed.\n"));
        return RGY_ERR_FILE_OPEN;
    }
    std::unique_ptr<AVFormatContext, RGYAVDeleter<AVFormatContext>> fmtGuard(
        fmtCtxRaw, RGYAVDeleter<AVFormatContext>(avformat_close_input));
    AVFormatContext *fmtCtx = fmtGuard.get();

    if (fmtCtx->pb != nullptr && !(fmtCtx->pb->seekable & AVIO_SEEKABLE_NORMAL)) {
        AddMessage(RGY_LOG_WARN,
            _T("source not seekable; kernel=auto requires seekable input.\n")
            _T("    Please pass kernel=<concrete> explicitly.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("probe: avformat_find_stream_info failed.\n"));
        return RGY_ERR_UNKNOWN;
    }
    const int videoIdx = av_find_best_stream(fmtCtx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (videoIdx < 0) {
        AddMessage(RGY_LOG_ERROR, _T("probe: no video stream.\n"));
        return RGY_ERR_INVALID_DATA_TYPE;
    }
    AVStream *vst = fmtCtx->streams[videoIdx];
    const AVCodec *codec = avcodec_find_decoder(vst->codecpar->codec_id);
    if (!codec) {
        AddMessage(RGY_LOG_ERROR, _T("probe: decoder not available for stream.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    AVCodecContext *codecCtxRaw = avcodec_alloc_context3(codec);
    if (!codecCtxRaw) return RGY_ERR_NULL_PTR;
    std::unique_ptr<AVCodecContext, RGYAVDeleter<AVCodecContext>> codecGuard(
        codecCtxRaw, RGYAVDeleter<AVCodecContext>(avcodec_free_context));
    AVCodecContext *codecCtx = codecGuard.get();
    if (avcodec_parameters_to_context(codecCtx, vst->codecpar) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("probe: avcodec_parameters_to_context failed.\n"));
        return RGY_ERR_UNKNOWN;
    }
    codecCtx->time_base    = vst->time_base;
    codecCtx->pkt_timebase = vst->time_base;
    if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
        AddMessage(RGY_LOG_ERROR, _T("probe: avcodec_open2 failed.\n"));
        return RGY_ERR_UNKNOWN;
    }

    // ---- decode detect_frames luma planes -------------------------------
    //
    // Sampling strategy: when the file's duration is known (the common
    // case for re-openable local files), seek to N evenly-spread
    // timestamps in [10%, 90%] of the duration and pull one frame from
    // each. This avoids the "ten consecutive frames of a dark smoky
    // opening all score identically" failure mode that pure sequential
    // decoding suffered from. When duration is unknown - or any seek
    // fails - we fall back to consecutive decoding from frame 0.
    const int wantFrames = std::max(1, prm->descale.detect_frames);
    std::vector<std::vector<uint8_t>> lumaFrames;
    lumaFrames.reserve(wantFrames);

    AVPacket *pktRaw = av_packet_alloc();
    std::unique_ptr<AVPacket, RGYAVDeleter<AVPacket>> pktGuard(
        pktRaw, RGYAVDeleter<AVPacket>(av_packet_free));
    AVFrame *frameRaw = av_frame_alloc();
    std::unique_ptr<AVFrame, RGYAVDeleter<AVFrame>> frameGuard(
        frameRaw, RGYAVDeleter<AVFrame>(av_frame_free));

    // Per-probe key/non-key tallies so we can log how the frames were
    // sourced (all keyframes, or a mix after the non-key fallback
    // kicked in). totalDecoded counts every successful
    // avcodec_receive_frame across the seek loop; once it exceeds
    // wantFrames * 4 we drop the keyframe preference to avoid hanging
    // on streams where keyframes are sparse or unmarked.
    int keyframeCaptures = 0;
    int nonKeyCaptures = 0;
    int totalDecoded = 0;
    bool requireKey = true;
    const int decodeAttemptCap = std::max(8, wantFrames * 4);

    // Pull the next decoded frame from the codec; if requireKey is
    // true, frames with key_frame == 0 are skipped (the seek loop
    // will then send another packet and try again). Returns true if
    // a frame was captured.
    auto captureOneFrame = [&]() -> bool {
        int rv = avcodec_receive_frame(codecCtx, frameGuard.get());
        if (rv != 0) return false;
        AVFrame *f = frameGuard.get();
        totalDecoded++;
        // AV_FRAME_FLAG_KEY is the modern path; key_frame is the
        // legacy field that older libavcodec headers still expose.
#ifdef AV_FRAME_FLAG_KEY
        const bool isKey = (f->flags & AV_FRAME_FLAG_KEY) != 0;
#else
        const bool isKey = f->key_frame != 0;
#endif
        bool captured = false;
        if (f->width == src_w && f->height == src_h
            && (!requireKey || isKey)) {
            std::vector<uint8_t> luma((size_t)src_w * src_h * src_pixel_bytes);
            const int rowBytes = src_w * src_pixel_bytes;
            for (int y = 0; y < src_h; ++y) {
                memcpy(luma.data() + (size_t)y * rowBytes,
                       f->data[0] + (size_t)y * f->linesize[0],
                       rowBytes);
            }
            lumaFrames.push_back(std::move(luma));
            if (isKey) ++keyframeCaptures;
            else       ++nonKeyCaptures;
            captured = true;
        }
        av_frame_unref(f);
        return captured;
    };

    // Determine the seek window in stream time-base. Two modes:
    //   trim-window: prm->probeStartFrame/EndFrame are populated from
    //       --trim, so spread targets within [start_frame, end_frame].
    //   whole-file: spread targets within [10%, 90%] of duration.
    // vst->duration is the canonical source; fmtCtx->duration is in
    // AV_TIME_BASE and used as a fallback.
    int64_t streamDuration = vst->duration;
    if (streamDuration <= 0 && fmtCtx->duration > 0) {
        streamDuration = av_rescale_q(fmtCtx->duration,
            AVRational{1, AV_TIME_BASE}, vst->time_base);
    }

    const int64_t baseStartTs = (vst->start_time != AV_NOPTS_VALUE) ? vst->start_time : 0;

    // Frame-index to timestamp conversion via vst->avg_frame_rate.
    // av_inv_q gives seconds-per-frame as a rational; av_rescale_q then
    // expresses that in vst->time_base units.
    const AVRational fr = vst->avg_frame_rate;
    auto frameToTs = [&](int frame_idx) -> int64_t {
        if (fr.num <= 0 || fr.den <= 0) return -1;
        return baseStartTs + av_rescale_q((int64_t)frame_idx, av_inv_q(fr), vst->time_base);
    };

    int64_t windowStartTs = baseStartTs;
    int64_t windowSpanTs  = streamDuration > 0 ? (int64_t)(streamDuration * 0.80) : 0;
    int64_t windowOffsetTs = streamDuration > 0 ? (int64_t)(streamDuration * 0.10) : 0;
    bool useTrimWindow = false;
    if (prm->probeStartFrame > 0 && prm->probeEndFrame > prm->probeStartFrame) {
        const int64_t startTs = frameToTs(prm->probeStartFrame);
        const int64_t endTs   = frameToTs(prm->probeEndFrame);
        if (startTs > 0 && endTs > startTs) {
            windowStartTs = startTs;
            windowOffsetTs = 0;
            windowSpanTs   = endTs - startTs;
            useTrimWindow  = true;
            AddMessage(RGY_LOG_DEBUG,
                _T("probe: sampling within --trim window [frame %d, frame %d).\n"),
                prm->probeStartFrame, prm->probeEndFrame);
        }
    }

    bool seekMode = (windowSpanTs > 0 && wantFrames > 1);
    int seekFailures = 0;

    // User-facing probe-start summary (INFO, always visible). The
    // target-dims phrase varies by mode: in kernel-only mode the
    // dims are fixed; in resolution-search mode the probe scans a
    // range of heights and reports the bounds + step.
    const bool resSearch = (prm->descale.width <= 0 || prm->descale.height <= 0);
    tstring targetDesc;
    if (resSearch) {
        const int searchMin = prm->descale.search_min > 0 ? prm->descale.search_min : (src_h / 2);
        const int searchMax = prm->descale.search_max > 0 ? prm->descale.search_max : (int)(src_h * 0.85);
        const int searchStep = std::max(1, prm->descale.search_step);
        targetDesc = strsprintf(_T("heights %d-%d step %d"), searchMin, searchMax, searchStep);
    } else {
        targetDesc = strsprintf(_T("%dx%d"), dst_w, dst_h);
    }
    if (useTrimWindow) {
        AddMessage(RGY_LOG_INFO,
            _T("auto-detect: probing %s over %d frames (window=[frame %d, frame %d)).\n"),
            targetDesc.c_str(), wantFrames,
            prm->probeStartFrame, prm->probeEndFrame);
    } else {
        AddMessage(RGY_LOG_INFO,
            _T("auto-detect: probing %s over %d frames (window=full file).\n"),
            targetDesc.c_str(), wantFrames);
    }
    if (seekMode) {
        const int64_t startTs = windowStartTs + windowOffsetTs;
        for (int i = 0; i < wantFrames && (int)lumaFrames.size() < wantFrames; ++i) {
            // Spread targets evenly across the chosen window.
            const double frac = ((double)i + 0.5) / wantFrames;
            const int64_t target = startTs + (int64_t)((double)windowSpanTs * frac);
            if (av_seek_frame(fmtCtx, videoIdx, target, AVSEEK_FLAG_BACKWARD) < 0) {
                seekFailures++;
                continue;
            }
            avcodec_flush_buffers(codecCtx);
            // Decode forward from the keyframe at-or-before the target
            // until we capture exactly one frame. Bounded attempt count
            // protects against pathological streams where every receive
            // returns EAGAIN. captureOneFrame() honours requireKey: in
            // the initial pass we accept only keyframes (a fresh
            // keyframe just past the seek is what we want; subsequent
            // P/B frames are still in the keyframe's GOP and carry the
            // same scene content). The fallback below relaxes this.
            bool gotFrame = false;
            for (int attempts = 0; attempts < 64 && !gotFrame; ++attempts) {
                int rd = av_read_frame(fmtCtx, pktGuard.get());
                if (rd < 0) break;
                if (pktGuard.get()->stream_index != videoIdx) {
                    av_packet_unref(pktGuard.get()); continue;
                }
                avcodec_send_packet(codecCtx, pktGuard.get());
                av_packet_unref(pktGuard.get());
                gotFrame = captureOneFrame();
            }
            // Non-key fallback: if we've burned through enough decode
            // attempts without filling the lumaFrames vector, the
            // stream's keyframe density is too low for our seek-and-
            // wait strategy. Drop the keyframe preference so the rest
            // of the seek loop accepts any decoded frame.
            if (requireKey && totalDecoded >= decodeAttemptCap
                && (int)lumaFrames.size() < wantFrames) {
                AddMessage(RGY_LOG_DEBUG,
                    _T("probe: %d decoded frames without filling lumaFrames; relaxing keyframe requirement.\n"),
                    totalDecoded);
                requireKey = false;
            }
        }
        // If too many seeks failed, drop into sequential fallback to
        // top up. (Most failures would be at file boundaries; the
        // sequential path still terminates after wantFrames or EOF.)
        if (seekFailures > 0 && lumaFrames.empty()) {
            seekMode = false;
            // Reset to beginning before falling back.
            av_seek_frame(fmtCtx, videoIdx, 0, AVSEEK_FLAG_BACKWARD);
            avcodec_flush_buffers(codecCtx);
        }
    }

    // Sequential fallback (also runs to top up when seek-mode captured
    // fewer than wantFrames). Decodes consecutively from the current
    // position until wantFrames is hit or EOF.
    if (!seekMode || (int)lumaFrames.size() < wantFrames) {
        while ((int)lumaFrames.size() < wantFrames) {
            int rd = av_read_frame(fmtCtx, pktGuard.get());
            if (rd < 0) break;
            if (pktGuard.get()->stream_index != videoIdx) {
                av_packet_unref(pktGuard.get()); continue;
            }
            avcodec_send_packet(codecCtx, pktGuard.get());
            av_packet_unref(pktGuard.get());
            // Drain whatever the decoder has buffered.
            while ((int)lumaFrames.size() < wantFrames && captureOneFrame()) {}
        }
        if (lumaFrames.empty()) {
            // EOF drain.
            avcodec_send_packet(codecCtx, nullptr);
            while (captureOneFrame()) {}
        }
    }
    if (lumaFrames.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("probe: failed to decode any frames.\n"));
        return RGY_ERR_UNKNOWN;
    }
    AddMessage(RGY_LOG_INFO,
        _T("probe: captured %d frames (%d keyframes%s; %s; seek-failures=%d).\n"),
        (int)lumaFrames.size(),
        keyframeCaptures,
        nonKeyCaptures > 0
            ? strsprintf(_T(" + %d non-key fallback"), nonKeyCaptures).c_str()
            : _T(""),
        seekMode ? (useTrimWindow ? _T("spread within --trim window")
                                  : _T("spread across full file"))
                 : _T("sequential from frame 0"),
        seekFailures);

    // ---- upload luma frames to GPU --------------------------------------
    AddMessage(RGY_LOG_DEBUG, _T("probe: uploading %d luma frames to GPU...\n"), (int)lumaFrames.size());
    std::vector<std::unique_ptr<RGYCLBuf>> lumaBufs;
    lumaBufs.reserve(lumaFrames.size());
    for (auto &raw : lumaFrames) {
        auto buf = m_cl->createBuffer(raw.size(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, raw.data());
        if (!buf) {
            AddMessage(RGY_LOG_ERROR, _T("probe: failed to upload luma to GPU.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
        lumaBufs.push_back(std::move(buf));
    }
    AddMessage(RGY_LOG_DEBUG, _T("probe: upload complete (%zu MB total).\n"),
        (size_t)((lumaFrames.size() * (size_t)src_w * src_h * src_pixel_bytes) / (1024 * 1024)));

    // ---- compute per-frame edge weights ---------------------------------
    // The Sobel-magnitude map is identical across all candidates (it
    // only depends on the source frame), so we run
    // kernel_compute_edge_weight once per uploaded luma frame and
    // re-use the output across every candidate's MSE evaluation.
    AddMessage(RGY_LOG_DEBUG, _T("probe: computing edge weights for %d frames...\n"),
        (int)lumaBufs.size());
    std::vector<std::unique_ptr<RGYCLBuf>> edgeWeightsBufs;
    edgeWeightsBufs.reserve(lumaBufs.size());
    {
        RGYOpenCLQueue &queue = m_cl->queue();
        const int src_pitch_bytes_local = src_w * src_pixel_bytes;
        for (size_t i = 0; i < lumaBufs.size(); ++i) {
            auto wbuf = m_cl->createBuffer((size_t)src_w * src_h * sizeof(float), CL_MEM_READ_WRITE);
            if (!wbuf) {
                AddMessage(RGY_LOG_ERROR, _T("probe: failed to allocate edge weights buffer.\n"));
                return RGY_ERR_MEMORY_ALLOC;
            }
            RGYWorkSize local(16, 8);
            RGYWorkSize global(src_w, src_h);
            auto err = m_descale.get()->kernel("kernel_compute_edge_weight").config(queue, local, global, {}, nullptr).launch(
                lumaBufs[i]->mem(), src_pitch_bytes_local,
                wbuf->mem(), src_w,
                src_w, src_h);
            if (err != RGY_ERR_NONE) {
                AddMessage(RGY_LOG_ERROR, _T("probe: kernel_compute_edge_weight launch failed: %s.\n"),
                    get_err_mes(err));
                return err;
            }
            edgeWeightsBufs.push_back(std::move(wbuf));
        }
        queue.finish();
    }
    AddMessage(RGY_LOG_DEBUG, _T("probe: edge weights computed.\n"));

    // ---- build candidate list + score ----------------------------------
    // In kernel-only mode (user supplied width/height): one candidate
    // per row of kDescaleAutoCandidates, all sharing the same dims.
    // In resolution-search mode (user passed auto=true with width/
    // height=0): the three search passes build their own candidate
    // lists below and call scoreCandidates() once per pass.
    std::vector<ProbeCandidate> candidates;
    const bool resolutionSearchMode =
        (prm->descale.width <= 0 || prm->descale.height <= 0);

    if (!resolutionSearchMode) {
        candidates.reserve(kAutoCandCount);
        for (const auto &kv : kDescaleAutoCandidates) {
            ProbeCandidate pc{ kv.kernel, kv.b, kv.c, dst_w, dst_h, 0.0,
                               probe_label_for(kv.kernel, kv.b, kv.c) };
            candidates.push_back(std::move(pc));
        }
        AddMessage(RGY_LOG_DEBUG, _T("probe: kernel-only mode, scoring %d candidates.\n"),
            (int)candidates.size());
        auto err = scoreCandidates(candidates, lumaBufs, edgeWeightsBufs, src_w, src_h, src_pixel_bytes);
        if (err != RGY_ERR_NONE) return err;
    } else {
        // Resolution-search mode: 3-pass coarse-to-fine pyramid.
        AddMessage(RGY_LOG_DEBUG, _T("probe: resolution-search mode.\n"));
        auto err = runResolutionSearch(prm, candidates, lumaBufs, edgeWeightsBufs,
                                       src_w, src_h, src_pixel_bytes, fmtCtx, vst);
        if (err != RGY_ERR_NONE) return err;
    }
    AddMessage(RGY_LOG_DEBUG, _T("probe: scoring complete, locking winner.\n"));

    // ---- lock winner ---------------------------------------------------
    // The detailed step-by-step traces below are at DEBUG level only;
    // enable with --log-level debug if you need to diagnose a silent
    // abort in this section. Normal operation produces a single
    // user-facing INFO line summarising the result.

    // Sanitise MSE before sorting. A poorly-matched kernel can produce
    // overflow during descale -> re-upscale, which the Kahan inner loop
    // can propagate as Inf or NaN into the row sum. Letting NaN reach
    // std::sort violates strict-weak-ordering and triggers undefined
    // behaviour - on MSVC this silently aborts the process without a
    // stack trace.
    AddMessage(RGY_LOG_DEBUG, _T("lock: sanity-loop start (%d candidates).\n"),
        (int)candidates.size());
    int nonFiniteCount = 0;
    for (auto &c : candidates) {
        if (!std::isfinite(c.mse) || c.mse < 0.0) {
            c.mse = 1e30;
            nonFiniteCount++;
        }
    }
    AddMessage(RGY_LOG_DEBUG, _T("lock: sanity-loop done (%d clamped).\n"), nonFiniteCount);
    for (size_t i = 0; i < candidates.size(); ++i) {
        AddMessage(RGY_LOG_DEBUG, _T("lock:   candidates[%d] mse=%.6e finite=%d\n"),
            (int)i, candidates[i].mse, (int)std::isfinite(candidates[i].mse));
    }

    // Group candidates by height: per height, retain the lowest
    // (best) MSE and the index of the candidate that produced it.
    // This is the precondition for the argmax-ratio winner pick used
    // in resolution-search mode - argmin(MSE) alone biases toward the
    // largest tested height (under-determined regime where the
    // descale -> re-upscale residual collapses to noise).
    std::map<int, std::pair<size_t, double>> bestByHeight;
    // Parallel per-height all-kernels tracker used only by the
    // show_scores diagnostic below. bestByHeight already collapses each
    // height to its winning kernel for the argmax-ratio winner-pick
    // math; this auxiliary map preserves every competing kernel's
    // candidate index so the diagnostic table can print per-kernel
    // breakdowns under the primary per-height row. Empty in the
    // non-show_scores case the cost is negligible (one extra vector
    // push per finite-MSE candidate, ~350 entries worst case).
    std::map<int, std::vector<size_t>> heightToCandidates;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto &c = candidates[i];
        if (c.mse >= 1e29) continue;  // skip clamped / sentinel entries
        auto it = bestByHeight.find(c.height);
        if (it == bestByHeight.end() || c.mse < it->second.second) {
            bestByHeight[c.height] = { i, c.mse };
        }
        heightToCandidates[c.height].push_back(i);
    }
    const bool useArgmaxRatio = resolutionSearchMode && bestByHeight.size() >= 2;
    AddMessage(RGY_LOG_DEBUG,
        _T("lock: %d distinct heights with finite MSE; using %s winner-pick.\n"),
        (int)bestByHeight.size(),
        useArgmaxRatio ? _T("argmax-ratio") : _T("argmin"));

    size_t winnerIdx = 0;
    double winnerRatio = 0.0;
    double secondRatio = 0.0;
    const TCHAR *conf = _T("low");
    tstring gapDesc;
    double displayGap = 1.0;
    bool snappedToStandard = false;
    // ratiosByHeight is filled only in argmax-ratio mode and consumed
    // by show_scores below to print the per-height progression.
    // Tuple: (height, raw_mse, smoothed_mse, ratio_prev_to_this).
    std::vector<std::tuple<int, double, double, double>> ratiosByHeight;

    if (useArgmaxRatio) {
        // std::map iteration is sorted ascending by key (height).
        // ratio(h_i) = best_mse(h_{i-1}) / best_mse(h_i). Below native
        // the MSE drops fast (ratio >> 1); at native the single
        // largest drop occurs; above native the MSE flattens out as
        // the system becomes under-determined and ratios approach 1.
        std::vector<std::pair<int, double>> hMse;
        hMse.reserve(bestByHeight.size());
        for (const auto &kv : bestByHeight) {
            hMse.push_back({ kv.first, kv.second.second });
        }
        // 3-height centred moving average over the per-height best
        // MSE values, used as input to the ratio computation. The raw
        // curve is noisy under compression - single-height dips
        // produced by quantisation artefacts can win argmax-ratio
        // against the true native if scored directly. Smoothing
        // suppresses those spikes without erasing the broader
        // monotonic descent that's the real signal. The reported
        // winner MSE in the user-facing log still comes from the
        // unsmoothed value so the number matches what the descale
        // produced for that height.
        std::vector<double> smoothedMse(hMse.size(), 0.0);
        std::map<int, double> smoothedByHeight;
        for (size_t i = 0; i < hMse.size(); ++i) {
            const size_t lo = (i == 0) ? 0 : (i - 1);
            const size_t hi = std::min(hMse.size() - 1, i + 1);
            double sum = 0.0;
            int n = 0;
            for (size_t j = lo; j <= hi; ++j) {
                sum += hMse[j].second;
                ++n;
            }
            smoothedMse[i] = sum / (double)n;
            smoothedByHeight[hMse[i].first] = smoothedMse[i];
        }

        ratiosByHeight.reserve(hMse.size());
        ratiosByHeight.push_back(std::make_tuple(hMse[0].first, hMse[0].second, smoothedMse[0], 0.0));
        int bestRatioHeight = hMse[0].first;
        std::map<int, double> ratioByHeight;  // for snap-to-standard lookup
        ratioByHeight[hMse[0].first] = 0.0;
        for (size_t i = 1; i < hMse.size(); ++i) {
            const double prev = smoothedMse[i - 1];
            const double cur  = smoothedMse[i];
            const double r = (cur > 0.0) ? prev / cur : 0.0;
            ratiosByHeight.push_back(std::make_tuple(hMse[i].first, hMse[i].second, cur, r));
            ratioByHeight[hMse[i].first] = r;
            if (r > winnerRatio) {
                secondRatio = winnerRatio;
                winnerRatio = r;
                bestRatioHeight = hMse[i].first;
            } else if (r > secondRatio) {
                secondRatio = r;
            }
        }

        // Option A: common-height fallback when the argmax-ratio
        // peak isn't decisive. If winnerRatio is small in absolute
        // terms AND the gap to secondRatio is tight, the MSE-vs-
        // height curve is too smooth for adjacent-step ratios to
        // localise native. Restrict the winner pool to standard
        // native heights (kCommonNativeHeights) and pick the one
        // with the largest MSE drop relative to the previous probed
        // common height. This is a deliberate prior toward real-world
        // content distribution; argmin(MSE) over commons would also
        // bias upward (MSE decreases monotonically toward source
        // height), so we re-apply the argmax-ratio criterion in the
        // restricted pool.
        bool commonFallbackFired = false;
        const double decisiveness = (secondRatio > 0.0) ? (winnerRatio / secondRatio) : 1e30;
        const bool ambiguous = (decisiveness < 1.05) && (winnerRatio < 1.10);
        if (ambiguous) {
            // Sorted list of probed common heights (those actually
            // present in smoothedByHeight).
            std::vector<int> probedCommon;
            probedCommon.reserve(kCommonNativeHeightsCount);
            for (int k = 0; k < kCommonNativeHeightsCount; ++k) {
                const int ch = kCommonNativeHeights[k];
                if (smoothedByHeight.count(ch)) probedCommon.push_back(ch);
            }
            std::sort(probedCommon.begin(), probedCommon.end());

            int bestCommonH = -1;
            double bestCommonRatio = 0.0;
            for (size_t i = 1; i < probedCommon.size(); ++i) {
                const int h     = probedCommon[i];
                const int hPrev = probedCommon[i - 1];
                const double smHere = smoothedByHeight.at(h);
                const double smPrev = smoothedByHeight.at(hPrev);
                if (smHere <= 0.0) continue;
                const double r = smPrev / smHere;
                if (r > bestCommonRatio) {
                    bestCommonRatio = r;
                    bestCommonH = h;
                }
            }
            // If only one common height is probed (or none of the
            // adjacent-pair ratios are positive), there's nothing to
            // discriminate on; leave the original argmax-ratio winner.
            if (bestCommonH > 0) {
                AddMessage(RGY_LOG_DEBUG,
                    _T("lock: ambiguous peak (winner_ratio=%.4f, decisiveness=%.4f); "
                       "common-height argmax-ratio fallback -> %dp (ratio=%.4f).\n"),
                    winnerRatio, decisiveness, bestCommonH, bestCommonRatio);
                bestRatioHeight = bestCommonH;
                commonFallbackFired = true;
            }
        }

        // Snap-to-standard: when the argmax-ratio winner lands within
        // ±8 of a well-known native height AND that standard height
        // was actually probed with a comparable ratio (within 10% of
        // the winner's), prefer the standard. Compression noise can
        // pull the argmax peak a few pixels off the true native
        // (e.g. 718 vs 720, 808 vs 810). No-op when the common-height
        // fallback above already picked a standard.
        for (int k = 0; k < kCommonNativeHeightsCount; ++k) {
            const int common = kCommonNativeHeights[k];
            if (common == bestRatioHeight) break;  // already standard
            if (std::abs(common - bestRatioHeight) > 8) continue;
            auto it = ratioByHeight.find(common);
            if (it == ratioByHeight.end()) continue;
            const double commonRatio = it->second;
            if (winnerRatio <= 0.0) continue;
            if (commonRatio / winnerRatio >= 0.9) {
                AddMessage(RGY_LOG_DEBUG,
                    _T("lock: snap from %dp (ratio=%.4f) to standard %dp (ratio=%.4f).\n"),
                    bestRatioHeight, winnerRatio, common, commonRatio);
                bestRatioHeight = common;
                snappedToStandard = true;
                break;
            }
        }

        winnerIdx = bestByHeight[bestRatioHeight].first;
        // Confidence: the ratio's magnitude (how big the MSE drop at
        // native is) crossed with how decisively that drop dominates
        // its neighbours (winnerRatio / secondRatio). The common-
        // height fallback forces "low" since the original signal
        // was ambiguous - the fallback just biased toward a sensible
        // prior.
        const double ratioGap = (secondRatio > 0.0) ? winnerRatio / secondRatio : 1e30;
        if (commonFallbackFired)                   conf = _T("low (common-height fallback)");
        else if (winnerRatio > 1.10 && ratioGap > 1.02) conf = _T("high");
        else if (winnerRatio > 1.05)               conf = _T("medium");
        else                                       conf = _T("low");
        displayGap = winnerRatio;
        if (commonFallbackFired)         gapDesc = _T("common-height fallback");
        else if (snappedToStandard)      gapDesc = _T("argmax-ratio + snap to standard");
        else                             gapDesc = _T("argmax-ratio");
        AddMessage(RGY_LOG_DEBUG,
            _T("lock: winner_height=%d winner_ratio=%.4f second_ratio=%.4f ratio_gap=%.4f snap=%d.\n"),
            bestRatioHeight, winnerRatio, secondRatio, ratioGap, (int)snappedToStandard);
    } else {
        // argmin path: kernel-only mode, or resolution-search fallback
        // when only one height has finite MSE (fast-path lock).
        double bestMse = 1e30;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].mse < bestMse) {
                bestMse = candidates[i].mse;
                winnerIdx = i;
            }
        }
        double secondMse = 1e30;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (i == winnerIdx) continue;
            if (candidates[i].mse < secondMse) secondMse = candidates[i].mse;
        }
        const double mseRatio = (candidates[winnerIdx].mse > 0.0)
            ? secondMse / candidates[winnerIdx].mse : 1e30;
        if (mseRatio > 10.0)      conf = _T("high");
        else if (mseRatio > 1.02) conf = _T("medium");
        else                      conf = _T("low");
        displayGap = mseRatio;
        gapDesc = _T("argmin");
        AddMessage(RGY_LOG_DEBUG, _T("lock: winner_idx=%d winner_mse=%.6e mse_ratio=%.4f.\n"),
            (int)winnerIdx, candidates[winnerIdx].mse, mseRatio);
    }

    const ProbeCandidate &winner = candidates[winnerIdx];

    // Stage 1 height-snap to standard. Argmax-ratio / argmin gives the
    // best location on the MSE curve, but HEVC quantisation noise (and
    // other re-encode artefacts) can smear the cliff by a few pixels,
    // producing detections like 472p when the true native is 480p.
    // When confidence is low or medium AND the winner is within 10
    // pixels of a canonical native height, snap to that height and
    // recompute the matching width. High-confidence picks are left
    // alone (the peak was decisive; smearing isn't the problem there).
    int lockedHeight = winner.height;
    int lockedWidth  = winner.width;
    {
        static const int kStandardHeights[] = { 360, 480, 486, 540, 576, 720, 810, 900, 1080 };
        const bool highConfidence = (_tcsstr(conf, _T("high")) != nullptr);
        if (!highConfidence) {
            int nearestH = 0;
            int nearestDist = 11;  // strict: must be <= 10 to snap
            for (int sh : kStandardHeights) {
                const int d = std::abs(sh - winner.height);
                if (d < nearestDist) {
                    nearestDist = d;
                    nearestH = sh;
                }
            }
            if (nearestH != 0 && nearestH != winner.height) {
                const int snappedW = width_from_height(src_w, src_h, nearestH, fmtCtx, vst);
                if (snappedW > 0 && snappedW < src_w) {
                    AddMessage(RGY_LOG_INFO,
                        _T("Stage 1 height snapped %d->%d (within 10px of standard).\n"),
                        winner.height, nearestH);
                    lockedHeight = nearestH;
                    lockedWidth  = snappedW;
                }
            }
        }
    }

    // Single consolidated user-facing INFO line. The `pct` literal is
    // load-bearing: writing `%` here renders as a percent sign in the
    // output, which rgy_print_stderr re-parses as a printf format and
    // silently aborts on (see rgy_log.cpp:65).
    AddMessage(RGY_LOG_INFO,
        _T("auto-detected %s at %dx%d, mse=%.6e, gap=%.4f (%s), confidence=%s after %d frames.\n"),
        winner.label.c_str(), lockedWidth, lockedHeight,
        winner.mse, displayGap, gapDesc.c_str(), conf, (int)lumaBufs.size());

    if (prm->descale.show_scores) {
        if (useArgmaxRatio) {
            // Per-height table with adjacent-ratio progression. Raw
            // and smoothed MSE columns are both printed so the
            // smoothing's effect is visible; ratio is computed from
            // smoothed values (that's what the argmax sees).
            AddMessage(RGY_LOG_INFO,
                _T("  height   best_MSE       smoothed_MSE   ratio(prev/this)\n"));
            for (const auto &t : ratiosByHeight) {
                const int h = std::get<0>(t);
                const double raw = std::get<1>(t);
                const double sm  = std::get<2>(t);
                const double r   = std::get<3>(t);
                const TCHAR *marker = (h == winner.height) ? _T(" <- native") : _T("");
                if (r > 0.0) {
                    AddMessage(RGY_LOG_INFO,
                        _T("  %-7d  %-13.6e  %-13.6e  %-7.4f%s\n"),
                        h, raw, sm, r, marker);
                } else {
                    AddMessage(RGY_LOG_INFO,
                        _T("  %-7d  %-13.6e  %-13.6e  --     %s\n"),
                        h, raw, sm, marker);
                }
                // Per-kernel breakdown at this height. Sorted by MSE
                // ascending so the kernel that won bestByHeight[h]
                // appears first. Suppressed when only one kernel was
                // tested at this height (typical in Pass 1 strides),
                // because the primary row above is already definitive.
                auto htci = heightToCandidates.find(h);
                if (htci != heightToCandidates.end() && htci->second.size() > 1) {
                    std::vector<size_t> ordered = htci->second;
                    std::sort(ordered.begin(), ordered.end(),
                        [&candidates](size_t a, size_t b) { return candidates[a].mse < candidates[b].mse; });
                    for (size_t idx : ordered) {
                        const auto &kc = candidates[idx];
                        AddMessage(RGY_LOG_INFO,
                            _T("             %-26s mse=%.6e\n"),
                            kc.label.c_str(), kc.mse);
                    }
                }
            }
        } else {
            // argmin mode: sort by MSE ascending, print top-10.
            std::vector<ProbeCandidate> sorted = candidates;
            std::sort(sorted.begin(), sorted.end(),
                [](const ProbeCandidate &a, const ProbeCandidate &b) { return a.mse < b.mse; });
            const int dump_n = std::min((int)sorted.size(), 10);
            for (int i = 0; i < dump_n; ++i) {
                AddMessage(RGY_LOG_INFO, _T("  #%-2d %-26s %dx%d mse=%.6e\n"),
                    i + 1, sorted[i].label.c_str(),
                    sorted[i].width, sorted[i].height, sorted[i].mse);
            }
        }
    }

    // ---- Stage 2: symmetric kernel tie-break at locked (w, h) ------
    // Stage 1's per-height MSE argmin runs against a fixed Catmull-Rom
    // forward, which protects height detection from wide-tap candidate
    // self-ringing but biases the kernel argmin toward Bicubic (the
    // catalogue entry whose Catmull-Rom forward is a self-inverse-
    // shaped pair). Now that (lockedWidth, lockedHeight) are locked
    // by Stage 1 (post-snap), re-score all catalogue kernels at
    // exactly those dimensions with symmetricForward=true (forward
    // kernel == c.kernel). Overwrite only kernel/b/c -- height and
    // width stay from Stage 1.
    VppDescaleKernel stage2Kernel = winner.kernel;
    float            stage2B      = winner.b;
    float            stage2C      = winner.c;
    tstring          stage2Label  = winner.label;
    {
        std::vector<ProbeCandidate> stage2;
        stage2.reserve(kAutoCandCount);
        for (const auto &kv : kDescaleAutoCandidates) {
            stage2.push_back(ProbeCandidate{ kv.kernel, kv.b, kv.c,
                lockedWidth, lockedHeight, 0.0,
                probe_label_for(kv.kernel, kv.b, kv.c) });
        }
        auto s2err = scoreCandidates(stage2, lumaBufs, edgeWeightsBufs,
                                      src_w, src_h, src_pixel_bytes,
                                      /*symmetricForward=*/true);
        if (s2err == RGY_ERR_NONE) {
            size_t bestIdx = 0;
            double bestMse = 1e30;
            for (size_t i = 0; i < stage2.size(); ++i) {
                if (std::isfinite(stage2[i].mse) && stage2[i].mse >= 0.0 && stage2[i].mse < bestMse) {
                    bestMse = stage2[i].mse;
                    bestIdx = i;
                }
            }
            if (bestMse < 1e29) {
                stage2Kernel = stage2[bestIdx].kernel;
                stage2B      = stage2[bestIdx].b;
                stage2C      = stage2[bestIdx].c;
                stage2Label  = stage2[bestIdx].label;
                AddMessage(RGY_LOG_INFO,
                    _T("Stage 2 (symmetric kernel pick at %dx%d): %s mse=%.6e (was %s).\n"),
                    lockedWidth, lockedHeight, stage2Label.c_str(), bestMse,
                    winner.label.c_str());
                if (prm->descale.show_scores) {
                    std::vector<size_t> ordered(stage2.size());
                    for (size_t i = 0; i < stage2.size(); ++i) ordered[i] = i;
                    std::sort(ordered.begin(), ordered.end(),
                        [&stage2](size_t a, size_t b) { return stage2[a].mse < stage2[b].mse; });
                    AddMessage(RGY_LOG_INFO,
                        _T("Stage 2 symmetric ranking at %dx%d:\n"),
                        lockedWidth, lockedHeight);
                    for (size_t idx : ordered) {
                        AddMessage(RGY_LOG_INFO,
                            _T("             %-26s mse=%.6e\n"),
                            stage2[idx].label.c_str(), stage2[idx].mse);
                    }
                }
            } else {
                AddMessage(RGY_LOG_WARN,
                    _T("Stage 2: no finite MSE; keeping Stage 1 kernel %s.\n"),
                    winner.label.c_str());
            }
        } else {
            AddMessage(RGY_LOG_WARN,
                _T("Stage 2: scoring failed (%s); keeping Stage 1 kernel %s.\n"),
                get_err_mes(s2err), winner.label.c_str());
        }
    }

    // Rewrite the parameter so the rest of init() proceeds as the
    // manual path. autoDetect is cleared to prevent re-entry on
    // subsequent init() calls (e.g., dynamic param changes). Kernel /
    // b / c come from the Stage 2 symmetric pick; height / width from
    // Stage 1's resolution lock (after the height-snap post-filter).
    prm->descale.kernel     = stage2Kernel;
    prm->descale.b          = stage2B;
    prm->descale.c          = stage2C;
    prm->descale.width      = lockedWidth;
    prm->descale.height     = lockedHeight;
    prm->descale.autoDetect = false;

    const auto probeEnd = std::chrono::steady_clock::now();
    const double probeMs = std::chrono::duration<double, std::milli>(probeEnd - probeStart).count();
    AddMessage(RGY_LOG_INFO, _T("auto-detect: probe complete in %.1f ms.\n"), probeMs);
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDescale::prepareCore(RGYFilterDescaleCore &core, int src_dim, int dst_dim,
                                      VppDescaleKernel kernel, double b, double c_param,
                                      double shift, VppDescaleBorder border) {
    const int support = kernel_support(kernel);
    if (support <= 0 || src_dim <= 0 || dst_dim <= 0 || dst_dim >= src_dim) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid descale dimensions src=%d dst=%d support=%d (dst must be < src).\n"),
            src_dim, dst_dim, support);
        return RGY_ERR_INVALID_PARAM;
    }

    core.src_dim = src_dim;
    core.dst_dim = dst_dim;
    core.bandwidth = support * 4 - 1;
    core.c = core.bandwidth / 2;

    // Build dense forward-upscale weights matrix (dst_dim x src_dim).
    std::vector<double> dense;
    build_scaling_weights(kernel, support, dst_dim, src_dim, b, c_param, shift, (double)dst_dim, border, dense);

    // dense is (src_dim x dst_dim). Transpose to (dst_dim x src_dim)
    // for the A^T column-scan.
    std::vector<double> dense_t;
    transpose_matrix(src_dim, dst_dim, dense, dense_t);

    // Compute per-row [lidx, ridx) of the transposed matrix.
    std::vector<int> lidx(dst_dim, 0), ridx(dst_dim, 0);
    for (int i = 0; i < dst_dim; ++i) {
        for (int j = 0; j < src_dim; ++j) {
            if (dense_t[(size_t)i * src_dim + j] != 0.0) { lidx[i] = j; break; }
        }
        for (int j = src_dim - 1; j >= 0; --j) {
            if (dense_t[(size_t)i * src_dim + j] != 0.0) { ridx[i] = j + 1; break; }
        }
    }

    // A^T * A as a banded symmetric dst_dim x dst_dim matrix.
    std::vector<double> ata;
    multiply_sparse_matrices(dst_dim, src_dim, lidx, ridx, dense_t, dense, ata);

    // LDLT factorise A^T A in place; ata now holds [diag | L'] in its
    // upper triangle.
    banded_ldlt(dst_dim, core.bandwidth, ata);

    // Build the lower triangular L from the LDLT upper-triangle output.
    std::vector<double> lower_full;
    transpose_matrix(dst_dim, dst_dim, ata, lower_full);
    multiply_banded_with_diagonal(dst_dim, core.bandwidth, lower_full);

    // Pack into compact upper and lower per-layer arrays.
    std::vector<float> lower_packed, upper_packed, diagonal;
    pack_lower_upper_diag(dst_dim, core.bandwidth, lower_full, ata, lower_packed, upper_packed, diagonal);

    // Compact the dense_t weights into (dst_dim * weights_columns) where
    // weights_columns is the maximum non-zero run width.
    int maxw = 0;
    for (int i = 0; i < dst_dim; ++i) {
        maxw = std::max(maxw, ridx[i] - lidx[i]);
    }
    core.weights_columns = maxw;
    std::vector<float> weights_packed((size_t)dst_dim * maxw, 0.0f);
    for (int i = 0; i < dst_dim; ++i) {
        for (int j = 0; j < ridx[i] - lidx[i]; ++j) {
            weights_packed[(size_t)i * maxw + j] = (float)dense_t[(size_t)i * src_dim + lidx[i] + j];
        }
    }

    // Upload to device.
    core.weights = m_cl->createBuffer(weights_packed.size() * sizeof(float),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_packed.data());
    core.left_idx = m_cl->createBuffer(lidx.size() * sizeof(int),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, lidx.data());
    core.right_idx = m_cl->createBuffer(ridx.size() * sizeof(int),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ridx.data());
    core.lower = m_cl->createBuffer(lower_packed.size() * sizeof(float),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, lower_packed.data());
    core.upper = m_cl->createBuffer(upper_packed.size() * sizeof(float),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, upper_packed.data());
    core.diagonal = m_cl->createBuffer(diagonal.size() * sizeof(float),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, diagonal.data());

    if (!core.weights || !core.left_idx || !core.right_idx
        || !core.lower || !core.upper || !core.diagonal) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to allocate descale core buffers.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDescale::init(shared_ptr<RGYFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    RGY_ERR sts = RGY_ERR_NONE;
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<RGYFilterParamDescale>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameIn.width <= 0 || prm->frameIn.height <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid input dimensions.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    // Build the kernel program early - the auto-detect probe (below)
    // needs the descale + rescale + mse kernels available before any
    // candidate can be scored. The program's compile options depend
    // only on input csp / bit-depth, which is known here.
    auto prmPrev = std::dynamic_pointer_cast<RGYFilterParamDescale>(m_param);
    if (!m_descale.get() || !prmPrev
        || RGY_CSP_BIT_DEPTH[prmPrev->frameIn.csp] != RGY_CSP_BIT_DEPTH[prm->frameIn.csp]) {
        const auto options = strsprintf("-D Type=%s -D bit_depth=%d",
            RGY_CSP_BIT_DEPTH[prm->frameIn.csp] > 8 ? "ushort" : "uchar",
            RGY_CSP_BIT_DEPTH[prm->frameIn.csp]);
        m_descale.set(m_cl->buildResourceAsync(_T("RGY_FILTER_DESCALE_CL"), _T("EXE_DATA"), options.c_str()));
    }

    // Validate user-supplied target dims up front - before the
    // auto-detect probe even runs - so a 1:1 (or up-sampling) request
    // exits cleanly instead of churning through degenerate LDLT
    // candidates. width/height may legitimately be 0 at this point if
    // the user passed auto=true without explicit dims; runProbe enforces
    // its own "explicit width/height required" check in that case.
    if (prm->descale.width > 0 && prm->descale.height > 0) {
        if (prm->descale.width >= prm->frameIn.width || prm->descale.height >= prm->frameIn.height) {
            AddMessage(RGY_LOG_ERROR,
                _T("target dimensions (%dx%d) must be strictly smaller than the input (%dx%d).\n")
                _T("    A 1:1 scale produces a degenerate LDLT system; pass a smaller width/height,\n")
                _T("    or skip --vpp-descale entirely when no resolution recovery is needed.\n"),
                prm->descale.width, prm->descale.height, prm->frameIn.width, prm->frameIn.height);
            return RGY_ERR_INVALID_PARAM;
        }
    }

    // Auto-detect probe (pre-decode private libavcodec context). When
    // descale.kernel == Auto, runProbe scores every candidate against
    // the first decoded frames and rewrites prm->descale.kernel/.b/.c
    // to the winning configuration. After it returns, the rest of
    // init() proceeds exactly as the manual path.
    if (prm->descale.kernel == VppDescaleKernel::Auto || prm->descale.autoDetect) {
        auto probeErr = runProbe(prm.get());
        if (probeErr != RGY_ERR_NONE) return probeErr;
    }
    if (prm->descale.width <= 0 || prm->descale.height <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("--vpp-descale requires height= (and optionally kernel=, width=). For automatic kernel detection use auto=true.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    // Set the output frame info now that the (possibly auto-detected)
    // dimensions are locked.
    prm->frameOut = prm->frameIn;
    prm->frameOut.width = prm->descale.width;
    prm->frameOut.height = prm->descale.height;

    // Allocate output frame buffer in the new (smaller) dimensions.
    auto err = AllocFrameBuf(prm->frameOut, 2);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("Failed to allocate output frame buffer: %s.\n"), get_err_mes(err));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    // Build cores: indexed [direction (0=H, 1=V)][plane group (0=luma, 1=chroma)].
    // Chroma dimensions are inferred from the chroma plane's actual
    // subsampling; for monochrome there's no chroma core.
    const int plane_count = RGY_CSP_PLANES[prm->frameOut.csp];
    const auto luma_src_plane  = getPlane(&prm->frameIn,  RGY_PLANE_Y);
    const auto luma_dst_plane  = getPlane(&prm->frameOut, RGY_PLANE_Y);

    // H_luma
    err = prepareCore(m_cores[0][0], luma_src_plane.width, luma_dst_plane.width,
                      prm->descale.kernel, prm->descale.b, prm->descale.c,
                      (double)prm->descale.src_left, prm->descale.border);
    if (err != RGY_ERR_NONE) return err;
    // V_luma
    err = prepareCore(m_cores[1][0], luma_src_plane.height, luma_dst_plane.height,
                      prm->descale.kernel, prm->descale.b, prm->descale.c,
                      (double)prm->descale.src_top, prm->descale.border);
    if (err != RGY_ERR_NONE) return err;
    // Chroma cores
    if (plane_count > 1) {
        const auto chroma_src_plane = getPlane(&prm->frameIn,  RGY_PLANE_U);
        const auto chroma_dst_plane = getPlane(&prm->frameOut, RGY_PLANE_U);
        if (chroma_src_plane.width > 0 && chroma_dst_plane.width > 0) {
            err = prepareCore(m_cores[0][1], chroma_src_plane.width, chroma_dst_plane.width,
                              prm->descale.kernel, prm->descale.b, prm->descale.c,
                              (double)prm->descale.src_left, prm->descale.border);
            if (err != RGY_ERR_NONE) return err;
            err = prepareCore(m_cores[1][1], chroma_src_plane.height, chroma_dst_plane.height,
                              prm->descale.kernel, prm->descale.b, prm->descale.c,
                              (double)prm->descale.src_top, prm->descale.border);
            if (err != RGY_ERR_NONE) return err;
        }
    }

    // Per-plane float intermediates: H output is (dst_w x src_h), V
    // scratch is (dst_w x dst_h). Allocated separately to avoid
    // sub-buffer alignment requirements on the cl_mem region.
    for (int i = 0; i < plane_count && i < (int)m_intermediateH.size(); ++i) {
        const auto src_plane = getPlane(&prm->frameIn,  (RGY_PLANE)i);
        const auto dst_plane = getPlane(&prm->frameOut, (RGY_PLANE)i);
        m_intermediatePitchFloats[i] = dst_plane.width;
        const size_t bytes_h = (size_t)dst_plane.width * src_plane.height * sizeof(float);
        const size_t bytes_v = (size_t)dst_plane.width * dst_plane.height * sizeof(float);
        m_intermediateH[i] = m_cl->createBuffer(bytes_h, CL_MEM_READ_WRITE);
        m_intermediateV[i] = m_cl->createBuffer(bytes_v, CL_MEM_READ_WRITE);
        if (!m_intermediateH[i] || !m_intermediateV[i]) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to allocate descale intermediate buffers plane %d.\n"), i);
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return sts;
}

RGY_ERR RGYFilterDescale::runHPlane(RGYFrameInfo *pIntermediateFloat, const RGYFrameInfo *pInputPlane,
                                    const RGYFilterDescaleCore &core,
                                    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const char *kernel_name = "kernel_descale_h";
    RGYWorkSize local(DESCALE_BLOCK, 1);
    RGYWorkSize global(pInputPlane->height, 1);
    auto err = m_descale.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pIntermediateFloat->ptr[0], pIntermediateFloat->pitch[0] / (int)sizeof(float),
        (cl_mem)pInputPlane->ptr[0], pInputPlane->pitch[0],
        pInputPlane->height,
        core.dst_dim,
        core.c, core.weights_columns,
        core.weights->mem(), core.left_idx->mem(), core.right_idx->mem(),
        core.lower->mem(), core.upper->mem(), core.diagonal->mem());
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"), char_to_tstring(kernel_name).c_str(), get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDescale::runVPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pIntermediateFloat,
                                    RGYCLBuf *pVScratch,
                                    const RGYFilterDescaleCore &core,
                                    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    const int src_h = core.src_dim;
    const int dst_h = core.dst_dim;
    const int dst_w = pOutputPlane->width;
    const int src_pitch_floats = pIntermediateFloat->pitch[0] / (int)sizeof(float);
    const int scratch_pitch_floats = dst_w;

    const char *kernel_name = "kernel_descale_v";
    RGYWorkSize local(DESCALE_BLOCK, 1);
    RGYWorkSize global(dst_w, 1);
    auto err = m_descale.get()->kernel(kernel_name).config(queue, local, global, wait_events, event).launch(
        (cl_mem)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
        pVScratch->mem(), scratch_pitch_floats,
        (cl_mem)pIntermediateFloat->ptr[0], src_pitch_floats,
        src_h,
        dst_w, dst_h,
        core.c, core.weights_columns,
        core.weights->mem(), core.left_idx->mem(), core.right_idx->mem(),
        core.lower->mem(), core.upper->mem(), core.diagonal->mem(),
        1 /* writeIntegerOutput: live filter path feeds the next stage */);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at %s: %s.\n"), char_to_tstring(kernel_name).c_str(), get_err_mes(err));
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYFilterDescale::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    RGYOpenCLQueue &queue, const std::vector<RGYOpenCLEvent> &wait_events, RGYOpenCLEvent *event) {
    if (pInputFrame->ptr[0] == nullptr) {
        return RGY_ERR_NONE;
    }
    if (!m_descale.get()) {
        AddMessage(RGY_LOG_ERROR, _T("kernel program not loaded.\n"));
        return RGY_ERR_OPENCL_CRUSH;
    }
    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto &outFrame = m_frameBuf[(m_frameIdx++) % m_frameBuf.size()];
        ppOutputFrames[0] = &outFrame->frame;
    }
    const int plane_count = RGY_CSP_PLANES[ppOutputFrames[0]->csp];
    const std::vector<RGYOpenCLEvent> empty_events;
    for (int i = 0; i < plane_count; ++i) {
        const bool isChroma = (i > 0);
        const auto &coreH = m_cores[0][isChroma ? 1 : 0];
        const auto &coreV = m_cores[1][isChroma ? 1 : 0];

        auto srcPlane = getPlane(pInputFrame,         (RGY_PLANE)i);
        auto dstPlane = getPlane(ppOutputFrames[0],   (RGY_PLANE)i);

        // Wrap the H intermediate buffer as a single-plane RGYFrameInfo
        // so the run helpers can use the standard accessors.
        RGYFrameInfo intermediate{};
        intermediate.ptr[0]   = (uint8_t *)m_intermediateH[i]->mem();
        intermediate.pitch[0] = m_intermediatePitchFloats[i] * (int)sizeof(float);
        intermediate.width    = dstPlane.width;
        intermediate.height   = srcPlane.height;

        const auto &plane_wait_event = (i == 0) ? wait_events : empty_events;
        RGYOpenCLEvent *plane_event  = (i == plane_count - 1) ? event : nullptr;

        auto err = runHPlane(&intermediate, &srcPlane, coreH, queue, plane_wait_event, nullptr);
        if (err != RGY_ERR_NONE) return err;
        err = runVPlane(&dstPlane, &intermediate, m_intermediateV[i].get(), coreV, queue, {}, plane_event);
        if (err != RGY_ERR_NONE) return err;
    }
    return RGY_ERR_NONE;
}

void RGYFilterDescale::close() {
    for (auto &row : m_cores) {
        for (auto &core : row) {
            core.weights.reset();
            core.left_idx.reset();
            core.right_idx.reset();
            core.lower.reset();
            core.upper.reset();
            core.diagonal.reset();
        }
    }
    for (auto &buf : m_intermediateH) buf.reset();
    for (auto &buf : m_intermediateV) buf.reset();
    m_frameBuf.clear();
    m_descale.clear();
    m_cl.reset();
    m_frameIdx = 0;
}
