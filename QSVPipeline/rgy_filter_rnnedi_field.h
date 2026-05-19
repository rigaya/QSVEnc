// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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

#pragma once

#include <array>
#include <cstdint>

#include "rgy_err.h"
#include "rgy_prm.h"

enum class RGYRnnediField : int {
    Top = 0,
    Bottom = 1,
};

enum class RGYRnnediPlane : int {
    Y = 0,
    U = 1,
    V = 2,
    A = 3,
};

struct RGYRnnediFieldParam {
    VppRnnediField field;
    VppNnediNSize nsize;
    int nns;
    VppNnediQuality quality;
    int prescreen;
    VppNnediErrorType errortype;
    int clamp;
    std::array<bool, 4> processPlane;

    RGYRnnediFieldParam();
};

struct RGYRnnediNetworkShape {
    int xdia;
    int ydia;
    int neurons;
};

struct RGYRnnediTopology {
    int field;
    bool doubleRate;
    int frameMultiplier;
    int fpsMultiplier;
};

struct RGYRnnediFrameMap {
    int sourceFrame;
    RGYRnnediField generateField;
    RGYRnnediField copyField;
    int sourceFieldOffset;
    int evalRefOffsetY;
    bool doubleRate;
};

struct RGYRnnediPlanePadding {
    int width;
    int height;
    int hpad;
    int vpad;
    int refBaseOffsetX;
    int refBaseOffsetY;
};

struct RGYRnnediMirrorIndex {
    int index;
    bool padded;
};

struct RGYRnnediVec4MirrorIndex {
    int index4;
    bool reverseLanes;
    bool padded;
};

struct RGYRnnediMirrorPixelIndex {
    int index;
    int index4;
    int lane;
    bool reverseLanes;
    bool padded;
};

static constexpr int RGY_RNNEDI_HPAD = 32;
static constexpr int RGY_RNNEDI_VPAD = 3;
static constexpr VppRnnediField RGY_RNNEDI_DEFAULT_FIELD = VPP_RNNEDI_FIELD_BOB;
static constexpr VppNnediQuality RGY_RNNEDI_DEFAULT_QUALITY = VPP_NNEDI_QUALITY_FAST;
static constexpr int RGY_RNNEDI_DEFAULT_PRESCREEN = 2;
static constexpr VppNnediErrorType RGY_RNNEDI_DEFAULT_ERRORTYPE = VPP_NNEDI_ETYPE_ABS;
static constexpr int RGY_RNNEDI_DEFAULT_CLAMP = 1;

extern const std::array<int, 7> RGY_RNNEDI_XDIA;
extern const std::array<int, 7> RGY_RNNEDI_YDIA;
extern const std::array<int, 5> RGY_RNNEDI_NNS;

RGY_ERR rgy_rnnedi_validate_field_param(const RGYRnnediFieldParam& param);
RGYRnnediNetworkShape rgy_rnnedi_network_shape(int nsize, int nns);
RGY_ERR rgy_rnnedi_resolve_topology(RGYRnnediTopology *topology, int field, bool inputTff);
RGY_ERR rgy_rnnedi_map_output_frame(RGYRnnediFrameMap *frameMap, const RGYRnnediTopology& topology, int outputFrame);
RGYRnnediPlanePadding rgy_rnnedi_plane_padding(int srcWidth, int srcHeight, int xsub, int ysub);
RGYRnnediMirrorIndex rgy_rnnedi_mirror_index(int pos, int length);
RGYRnnediVec4MirrorIndex rgy_rnnedi_mirror_index4(int x4, int width4);
int rgy_rnnedi_mirror_lane(int lane, bool reverseLanes);
RGYRnnediMirrorPixelIndex rgy_rnnedi_mirror_pixel_index(int x, int width);
bool rgy_rnnedi_is_copied_field(int y, RGYRnnediField copyField);
bool rgy_rnnedi_is_generated_field(int y, RGYRnnediField generateField);
