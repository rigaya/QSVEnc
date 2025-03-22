// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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

#include <set>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <vector>
#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <shellapi.h>
#endif
#include <assert.h>
#include "rgy_osdep.h"
#include "qsv_query.h"
#include "rgy_version.h"
#include "rgy_avutil.h"
#include "rgy_filesystem.h"
#include "rgy_prm.h"
#include "rgy_cmd.h"
#include "qsv_cmd.h"

tstring GetQSVEncVersion() {
    static const TCHAR *const ENABLED_INFO[] = { _T("disabled"), _T("enabled") };
    tstring version;
    version += get_encoder_version();
    version += _T("\n");
    version += strsprintf(_T(" Intel Media SDK API v%d.%02d\n"), MFX_VERSION_MAJOR, MFX_VERSION_MINOR);
    version += _T(" reader: raw");
    if (ENABLE_AVI_READER)         version += _T(", avi");
    if (ENABLE_AVISYNTH_READER)    version += _T(", avs");
    if (ENABLE_VAPOURSYNTH_READER) version += _T(", vpy");
#if ENABLE_AVSW_READER && !FOR_AUO
    version += strsprintf(_T(", avsw, avhw"));
#endif
#if !(defined(_WIN32) || defined(_WIN64))
    version += _T("\n vpp:    resize, deinterlace, denoise, detail-enhance, image-stab");
    if (ENABLE_CUSTOM_VPP) version += _T(", delego");
    if (ENABLE_LIBASS_SUBBURN != 0 && ENABLE_AVSW_READER != 0) version += _T(", sub");
#endif
    version += _T("\n");
    version += _T(" others\n");
    version += strsprintf(_T("  libass     : %s\n"), ENABLED_INFO[ENABLE_LIBASS_SUBBURN]);
    version += strsprintf(_T("  libdovi    : %s\n"), ENABLED_INFO[ENABLE_LIBDOVI]);
#if (defined(_WIN32) || defined(_WIN64))
    version += strsprintf(_T("  d3d11      : %s\n"), ENABLED_INFO[ENABLE_D3D11]);
#else
    version += strsprintf(_T("  vulkan     : %s\n"), ENABLED_INFO[ENABLE_VULKAN]);
#endif
    version += strsprintf(_T("  libplacebo : %s\n"), ENABLED_INFO[ENABLE_LIBPLACEBO]);
    return version;
}

typedef struct ListData {
    const TCHAR *name;
    const CX_DESC *list;
    int default_index;
} ListData;

static tstring PrintMultipleListOptions(const TCHAR *option_name, const TCHAR *option_desc, const vector<ListData>& listDatas) {
    tstring str;
    const TCHAR *indent_space = _T("                                ");
    const int indent_len = (int)_tcslen(indent_space);
    const int max_len = 79;
    str += strsprintf(_T("   %s "), option_name);
    while ((int)str.length() < indent_len) {
        str += _T(" ");
    }
    str += strsprintf(_T("%s\n"), option_desc);
    const auto data_name_max_len = indent_len + 4 + std::accumulate(listDatas.begin(), listDatas.end(), 0,
        [](const int max_len, const ListData data) { return (std::max)(max_len, (int)_tcslen(data.name)); });

    for (const auto& data : listDatas) {
        tstring line = strsprintf(_T("%s- %s: "), indent_space, data.name);
        while ((int)line.length() < data_name_max_len) {
            line += strsprintf(_T(" "));
        }
        for (int i = 0; data.list[i].desc; i++) {
            if (i > 0 && data.list[i].value == data.list[i - 1].value) {
                continue; //連続で同じ値を示す文字列があるときは、先頭のみ表示する
            }
            const int desc_len = (int)(_tcslen(data.list[i].desc) + _tcslen(_T(", ")) + ((i == data.default_index) ? _tcslen(_T("(default)")) : 0));
            if (line.length() + desc_len >= max_len) {
                str += line + _T("\n");
                line = indent_space;
                while ((int)line.length() < data_name_max_len) {
                    line += strsprintf(_T(" "));
                }
            } else {
                if (i) {
                    line += strsprintf(_T(", "));
                }
            }
            line += strsprintf(_T("%s%s"), data.list[i].desc, (i == data.default_index) ? _T("(default)") : _T(""));
        }
        str += line + _T("\n");
    }
    return str;
}

tstring gen_cmd_help_vppmfx() {
    tstring str = strsprintf(_T("")
        _T("   --vpp-denoise <int> or\n")
        _T("                [<param1 >= <value>][, <param2 >= <value>][...]\n")
        _T("     use vpp denoise, short form will set strength(%d - %d)\n")
        _T("    params\n")
        _T("      mode=<string>         denoise mode\n")
        _T("        auto(default), auto_bdrate, auto_subjective, auto_adjust, pre, post\n")
        _T("      strength=<int>        set strength(%d - %d)\n"),
        QSV_VPP_DENOISE_MIN, QSV_VPP_DENOISE_MAX,
        QSV_VPP_DENOISE_MIN, QSV_VPP_DENOISE_MAX);
    str += strsprintf(_T("")
        _T("   --vpp-mctf [\"auto\" or <int>] use vpp motion compensated temporal filter(mctf)\n")
        _T("                                  set strength (%d-%d), default: %d (auto)\n"),
        QSV_VPP_MCTF_MIN, QSV_VPP_MCTF_MAX, QSV_VPP_MCTF_AUTO);
    str += strsprintf(_T("")
        _T("   --vpp-detail-enhance <int>   use vpp detail enahancer, set strength (%d-%d)\n")
        _T("   --vpp-deinterlace <string>   set vpp deinterlace mode\n"),
        QSV_VPP_DETAIL_ENHANCE_MIN, QSV_VPP_DETAIL_ENHANCE_MAX);
    str += strsprintf(_T("")
        _T("                                 - none     disable deinterlace\n")
        _T("                                 - normal   normal deinterlace\n")
        _T("                                 - it       inverse telecine\n")
#if ENABLE_ADVANCED_DEINTERLACE
        _T("                                 - it-manual <string>\n")
        _T("                                     \"32\", \"2332\", \"repeat\", \"41\"\n")
#endif
        _T("                                 - bob      double framerate\n")
#if ENABLE_ADVANCED_DEINTERLACE
        _T("                                 - auto     auto deinterlace\n")
        _T("                                 - auto-bob auto bob deinterlace\n")
#endif
#if ENABLE_FPS_CONVERSION
        _T("   --vpp-fps-conv <string>      set fps conversion mode\n")
        _T("                                enabled only when input is progressive\n")
        _T("                                 - none, x2, x2.5\n")
#endif
        _T("   --vpp-image-stab <string>    set image stabilizer mode\n")
        _T("                                 - none, upscale, box\n"));
    str += strsprintf(_T("")
        _T("   --vpp-perc-pre-enc           enable perceptual pre enc filter\n"));
    return str;
}

tstring encoder_help() {
    tstring str;
    str += strsprintf(_T("Usage: QSVEncC [Options] -i <filename> -o <filename>\n"));
    str += strsprintf(_T("\n")
#if ENABLE_AVSW_READER
        _T("When video codec could be decoded by QSV, any format or protocol supported\n")
        _T("by ffmpeg library could be used as a input.\n")
#endif
        _T("%s input can be %s%s%sraw YUV or YUV4MPEG2(y4m) format.\n")
        _T("when raw(default), fps, input-res are also necessary.\n")
        _T("\n")
        _T("output format will be automatically set by the output extension.\n")
        _T("when output filename is set to \"-\", H.264/AVC ES output is thrown to stdout.\n")
        _T("\n")
        _T("Example:\n")
        _T("  QSVEncC -i \"<avsfilename>\" -o \"<outfilename>\"\n")
        _T("  avs2pipemod -y4mp \"<avsfile>\" | QSVEncC --y4m -i - -o \"<outfilename>\"\n")
        _T("\n")
        _T("Example for Benchmark:\n")
        _T("  QSVEncC -i \"<avsfilename>\" --benchmark \"<benchmark_result.txt>\"\n")
        ,
        (ENABLE_AVSW_READER) ? _T("Also, ") : _T(""),
        (ENABLE_AVI_READER)         ? _T("avi, ") : _T(""),
        (ENABLE_AVISYNTH_READER)    ? _T("avs, ") : _T(""),
        (ENABLE_VAPOURSYNTH_READER) ? _T("vpy, ") : _T(""));
    str += strsprintf(_T("\n")
        _T("Information Options: \n")
        _T("-h,-? --help                    show help\n")
        _T("-v,--version                    show version info\n")
        _T("   --check-hw                   check if QuickSyncVideo is available\n")
        _T("   --check-lib                  check lib API version installed\n")
        _T("   --check-impl                 check VPL impl installed\n")
        _T("   --check-features [<string>]  check encode/vpp features\n")
        _T("                                 with no option value, result will on stdout,\n")
        _T("                                 otherwise, it is written to file path set\n")
        _T("                                 and opened by default application.\n")
        _T("                                 when writing to file, txt/html/csv format\n")
        _T("                                 is available, chosen by the extension\n")
        _T("                                 of the output file.\n")
        _T("   --check-features-html [<string>]\n")
        _T("                                check encode/vpp features and write html report to\n")
        _T("                                 specified path. With no value, \"qsv_check.html\"\n")
        _T("                                 will be created to current directory.\n")
        _T("   --check-environment          check environment info\n")
        _T("   --check-device               check device available\n")
        _T("   --check-clinfo               check OpenCL info\n")
#if ENABLE_AVSW_READER
        _T("   --check-avversion            show dll version\n")
        _T("   --check-codecs               show codecs available\n")
        _T("   --check-encoders             show audio encoders available\n")
        _T("   --check-decoders             show audio decoders available\n")
        _T("   --check-profiles <string>    show profile names available for specified audio codec\n")
        _T("   --check-formats              show in/out formats available\n")
        _T("   --check-protocols            show in/out protocols available\n")
        _T("   --check-avdevices            show in/out avdvices available\n")
        _T("   --check-filters              show filters available\n")
        _T("   --option-list                show option list\n")
#endif
        _T("\n"));

    str += strsprintf(_T("\n")
        _T("Basic Encoding Options: \n"));
    str += gen_cmd_help_input();
    str += strsprintf(_T("\n")
        _T("-d,--device <string> or <int>   set device number to encode\n")
        _T("                                 - auto(default), 1, 2, 3, 4\n")
        _T("-c,--codec <string>             set encode codec\n")
        _T("                                 - h264(default), hevc, mpeg2, vp9, av1, raw\n"));
    str += PrintMultipleListOptions(_T("--level <string>"), _T("set codec level"),
        { { _T("H.264"), list_avc_level,   0 },
          { _T("HEVC"),  list_hevc_level,  0 },
          { _T("MPEG2"), list_mpeg2_level, 0 },
          { _T("AV1"),   list_av1_level,   0 }
        });
    str += PrintMultipleListOptions(_T("--profile <string>"), _T("set codec profile"),
        { { _T("H.264"), list_avc_profile,   0 },
          { _T("HEVC"),  list_hevc_profile,  0 },
          { _T("MPEG2"), list_mpeg2_profile, 0 },
          { _T("VP9"),   list_vp9_profile,   0 },
          { _T("AV1"),   list_av1_profile,   0 }
        });
    str += PrintMultipleListOptions(_T("--tier <string>"), _T("set codec tier"),
        { { _T("HEVC"),  list_hevc_tier,  0 },
        });
    str += strsprintf(_T("\n")
        _T("   --output-depth <int>        output bit depth (default: 8)\n")
        _T("   --output-csp <string>       output colorspace (default: yuv420)\n")
        _T("                                 - yuv420, yuv422, yuv444, rgb\n")
        _T("\n"));
    str += strsprintf(_T("\n")
        _T("   --function-mode              select QSV function mode.\n")
        _T("                                 - auto, PG, FF\n")
        _T("   --fixed-func                 same as \"--function-mode FF\"\n")
        _T("   --hyper-mode <string>        set Deep Link Hyper Mode\n")
        _T("                                 off (=default), on, adaptive.\n")
        _T("\n"));
    str += strsprintf(_T("Frame buffer Options:\n")
        _T(" frame buffers are selected automatically by default.\n")
#if D3D_SURFACES_SUPPORT
        _T("   --disable-d3d                disable using d3d surfaces\n")
#if MFX_D3D11_SUPPORT
        _T("   --d3d                        use d3d9/d3d11 surfaces\n")
        _T("   --d3d9                       use d3d9 surfaces\n")
        _T("   --d3d11                      use d3d11 surfaces\n")
#else
        str += strsprintf(_T("")
            _T("   --d3d                        use d3d9 surfaces\n")
#endif //MFX_D3D11_SUPPORT
#endif //D3D_SURFACES_SUPPORT
#if LIBVA_SUPPORT
            _T("   --disable-va                 disable using va surfaces\n")
            _T("   --va                         use va surfaces\n")
#endif //#if LIBVA_SUPPORT
            _T("\n"));
    str += strsprintf(_T("Encode Mode Options:\n")
        _T(" EncMode default: --icq\n")
        _T("   --cqp <int> or               encode in Constant QP, default %d:%d:%d\n")
        _T("         <int>:<int>:<int>      set qp value for i:p:b frame\n")
        _T("   --la <int>                   set bitrate in Lookahead mode (kbps)\n")
        _T("   --la-hrd <int>               set bitrate in HRD-Lookahead mode (kbps)\n")
        _T("   --icq <int>                  encode in Intelligent Const. Quality mode\n")
        _T("                                  default value: %d\n")
        _T("   --la-icq <int>               encode in ICQ mode with Lookahead\n")
        _T("                                  default value: %d\n")
        _T("   --cbr <int>                  set bitrate in CBR mode (kbps)\n")
        _T("   --vbr <int>                  set bitrate in VBR mode (kbps)\n")
        _T("   --avbr <int>                 set bitrate in AVBR mode (kbps)\n")
        _T("                                 AVBR mode is only supported with API v1.3\n")
        _T("   --avbr-unitsize <int>        avbr calculation period in x100 frames\n")
        _T("                                 default %d (= unit size %d00 frames)\n")
        //_T("   --avbr-range <float>           avbr accuracy range from bitrate set\n)"
        //_T("                                   in percentage, defalut %.1f(%%)\n)"
        _T("   --qvbr <int>                 set bitrate in Quality VBR mode.\n")
        _T("                                 requires --qvbr-q option to be set as well\n")
        _T("   --qvbr-quality <int>         set quality used in qvbr mode. default: %d\n")
        _T("   --vcm <int>                  set bitrate in VCM mode (kbps)\n")
        _T("\n"),
        QSV_DEFAULT_QPI, QSV_DEFAULT_QPP, QSV_DEFAULT_QPB,
        QSV_DEFAULT_QPI, QSV_DEFAULT_QPP, QSV_DEFAULT_QPB,
        QSV_DEFAULT_ICQ, QSV_DEFAULT_ICQ,
        QSV_DEFAULT_CONVERGENCE, QSV_DEFAULT_CONVERGENCE,
        QSV_DEFAULT_QVBR);
    str += strsprintf(_T("Other Encode Options:\n")
        _T("   --fallback-rc                enable fallback of ratecontrol mode, when\n")
        _T("                                 platform does not support new ratecontrol modes.\n")
        _T("-a,--async-depth                set async depth for QSV pipeline.\n")
        _T("                                 default: 0 (=auto, %d)\n")
        _T("   --max-bitrate <int>          set max bitrate(kbps)\n")
        _T("   --qp-min <int> or            set min QP, default 0 (= unset)\n")
        _T("           <int>:<int>:<int>\n")
        _T("   --qp-max <int> or            set max QP, default 0 (= unset)\n")
        _T("           <int>:<int>:<int>\n")
        _T("   --qp-offset <int>[:<int>][:<int>]...\n")
        _T("                                set qp offset of each pyramid reference layers.\n")
        _T("                                 default 0 (= unset).\n")
        _T("-u,--quality <string>           encode quality\n")
        _T("                                  - best, higher, high, balanced(default)\n")
        _T("                                    fast, faster, fastest\n")
        _T("   --la-depth <int>             set Lookahead Depth, %d-%d\n")
        _T("   --la-window-size <int>       enables Lookahead Windowed Rate Control mode,\n")
        _T("                                  and set the window size in frames.\n")
        _T("   --la-quality <string>        set lookahead quality.\n")
        _T("                                 - auto(default), fast, medium, slow\n")
        _T("   --tune <string>[,...]        set tune encode quality mode.\n")
        _T("                                 - default, psnr, ssim, ms_ssim, vmaf, perceptual\n")
        _T("   --scenario-info <string>     set scenarios for the encoding.\n")
        _T("                                 unknown (default), display_remoting,\n")
        _T("                                 video_conference, archive, live_streaming,\n")
        _T("                                 camera_capture, video_surveillance,\n")
        _T("                                 game_streaming, remote_gaming\n")
        _T("   --(no-)extbrc                enables extbrc\n")
        _T("   --(no-)mbbrc                 enables per macro block rate control\n")
        _T("                                 default: auto\n")
        _T("   --ref <int>                  reference frames\n")
        _T("                                  default %d (auto)\n")
        _T("-b,--bframes <int>              number of sequential b frames\n")
        _T("                                  default auto\n")
        _T("   --gop-ref-dist <int>         <bframes>+1 for H.264/HEVC/MPEG2\n")
        _T("   --(no-)b-pyramid             enables B-frame pyramid reference (default:off)\n")
        _T("   --(no-)direct-bias-adjust    lower usage of B frame Direct/Skip type.\n")
        _T("   --gop-len <int>              (max) gop length, default %d (auto)\n")
        _T("                                  when auto, fps x 10 will be set.\n")
        _T("   --(no-)open-gop              enables open gop (default:off)\n")
        _T("   --strict-gop                 force gop structure\n")
        _T("   --(no-)i-adapt               enables adaptive I frame insert (default:off)\n")
        _T("   --(no-)b-adapt               enables adaptive B frame insert (default:off)\n")
        _T("   --(no-)weightp               enable weighted prediction for P frame\n")
        _T("   --(no-)weightb               enable weighted prediction for B frame\n")
        _T("   --(no-)adapt-ref             enable adaptive ref frames\n")
        _T("   --(no-)adapt-ltr             enable adaptive LTR frames\n")
        _T("   --(no-)adapt-cqm             enable adaptive CQM\n")
        _T("   --(no-)repartition-check     [H.264] enable prediction from small partitions\n")
#if ENABLE_FADE_DETECT
        _T("   --(no-)fade-detect           enable fade detection\n")
#endif //#if ENABLE_FADE_DETECT
        _T("   --trellis <string>           set trellis mode used in encoding\n")
        _T("                                 - auto(default), off, i, ip, all\n")
        _T("   --mv-scaling                 set mv cost scaling\n")
        _T("                                 - 0  set MV cost to be 0\n")
        _T("                                 - 1  set MV cost 1/2 of default\n")
        _T("                                 - 2  set MV cost 1/4 of default\n")
        _T("                                 - 3  set MV cost 1/8 of default\n")
        _T("   --slices <int>               number of slices, default 0 (auto)\n")
        _T("   --vbv-bufsize <int>          set vbv buffer size (kbit) / default: auto\n")
        _T("   --max-framesize <int         set max frame size (bytes) / default: auto>\n")
        _T("   --max-framesize-i <int>      set max I frame size (bytes) / default: auto\n")
        _T("   --max-framesize-p <int>      set max P/B frame size (bytes) / default: auto\n")
        _T("   --intra-refresh-cycle <int>  set intra refresh cycle (2 or larger).\n")
        _T("                                  default = 0 (disabled)\n")
        _T("   --no-deblock                 [h264] disables H.264 deblock feature\n")
        _T("   --tskip                      [hevc] enable transform skip\n")
        _T("   --sao <string>               [hevc]\n")
        _T("                                 - auto    default\n")
        _T("                                 - none    disable sao\n")
        _T("                                 - luma    enable sao for luma\n")
        _T("                                 - chroma  enable sao for chroma\n")
        _T("                                 - all     enable sao for luma & chroma\n")
        _T("   --ctu <int>                  [hevc] max ctu size\n")
        _T("                                 - auto(default), 16, 32, 64\n")
        _T("   --(no-)hevc-gpb              [hevc] enable(disable) GPB\n")
        _T("   --tile-row <int>             [av1] number of tile rows\n")
        _T("   --tile-col <int>             [av1] number of tile columns\n")
        //_T("   --sharpness <int>            [vp8] set sharpness level for vp8 enc\n")
        _T("\n"),
        QSV_DEFAULT_ASYNC_DEPTH,
        QSV_LOOKAHEAD_DEPTH_MIN, QSV_LOOKAHEAD_DEPTH_MAX,
        QSV_DEFAULT_REF,
        QSV_DEFAULT_GOP_LEN);

#if 0
    str += strsprintf(_T("\n")
        _T(" Settings below are available only for software ecoding.\n")
        _T("   --cavlc                      use cavlc instead of cabac\n")
        _T("   --rdo                        use rate distortion optmization\n")
        _T("   --inter-pred <int>           set minimum block size used for\n")
        _T("   --intra-pred <int>           inter/intra prediction\n")
        _T("                                  0: auto(default)   1: 16x16\n")
        _T("                                  2: 8x8             3: 4x4\n")
        _T("   --mv-search <int>            set window size for mv search\n")
        _T("                                  default: 0 (auto)\n")
        _T("   --mv-precision <int>         set precision of mv search\n")
        _T("                                  0: auto(default)   1: full-pell\n")
        _T("                                  2: half-pell       3: quater-pell\n")
    );
#endif
    str += strsprintf(_T("\n")
        _T("   --sar <int>:<int>            set Sample Aspect Ratio\n")
        _T("   --dar <int>:<int>            set Display Aspect Ratio\n")
        _T("   --bluray                     for H.264 bluray encoding\n")
        _T("\n"));
    str += strsprintf(_T("")
        _T("   --aud                        insert aud nal unit to ouput stream.\n")
        _T("   --pic-struct                 insert pic-timing SEI with pic_struct.\n")
        _T("   --buf-period                 insert buffering period SEI.\n")
        _T("   --(no-)repeat-headers        repeating insertion of headers\n"));

    str += _T("\n");
    str += gen_cmd_help_common();
    str += _T("\n");

    str += strsprintf(_T("\nVPP Options:\n"));
    str += gen_cmd_help_vppmfx();
    str += gen_cmd_help_vpp();

    str += strsprintf(_T("\n")
        _T("   --input-buf <int>            buffer size for input in frames (%d-%d)\n")
        _T("                                 default   hw: %d,  sw: %d\n")
        _T("                                 cannot be used with avqsv reader.\n"),
        QSV_INPUT_BUF_MIN, QSV_INPUT_BUF_MAX,
        QSV_DEFAULT_INPUT_BUF_HW, QSV_DEFAULT_INPUT_BUF_SW
        );
    str += strsprintf(_T("")
#if defined(_WIN32) || defined(_WIN64)
        _T("   --mfx-thread <int>           set mfx thread num (-1 (auto), 2, 3, ...)\n")
        _T("                                 note that mfx thread cannot be less than 2.\n")
#endif
        _T("   --gpu-copy                   Enables gpu accelerated copying between device and host.\n")
        _T("   --min-memory                 minimize memory usage of QSVEncC.\n")
        _T("                                 same as --output-thread 0 --audio-thread 0\n")
        _T("                                   --mfx-thread 2 -a 1 --input-buf 1 --output-buf 0\n")
        _T("                                 this will cause lower performance!\n")
#if defined(_WIN32) || defined(_WIN64)
        _T("   --(no-)timer-period-tuning   enable(disable) timer period tuning\n")
        _T("                                  default: enabled\n")
#endif //#if defined(_WIN32) || defined(_WIN64)
        );
#if ENABLE_SESSION_THREAD_CONFIG
    str += strsprintf(_T("")
        _T("   --session-threads            set num of threads for QSV session. (0-%d)\n")
        _T("                                 default: 0 (=auto)\n")
        _T("   --session-thread-priority    set thread priority for QSV session.\n")
        _T("                                  - low, normal(default), high\n"),
        QSV_SESSION_THREAD_MAX);
#endif

    str += _T("\n");
    str += gen_cmd_help_ctrl();
    str += strsprintf(_T("\n")
        _T("   --python <string>            set python path for --perf-monitor-plot\n")
        _T("                                 default: python\n")
        );
    str += strsprintf(_T("\n")
        _T("   --benchmark <string>         run in benchmark mode\n")
        _T("                                 and write result in txt file\n")
        _T("   --bench-quality \"all\" or <string>[,<string>][,<string>]...\n")
        _T("                                 default: \"best,balanced,fastest\"\n")
        _T("                                list of target quality to check on benchmark\n"));
    return str;
}

const TCHAR *cmd_short_opt_to_long(TCHAR short_opt) {
    const TCHAR *option_name = nullptr;
    switch (short_opt) {
    case _T('a'):
        option_name = _T("async-depth");
        break;
    case _T('b'):
        option_name = _T("bframes");
        break;
    case _T('c'):
        option_name = _T("codec");
        break;
    case _T('d'):
        option_name = _T("device");
        break;
    case _T('u'):
        option_name = _T("quality");
        break;
    case _T('f'):
        option_name = _T("output-format");
        break;
    case _T('i'):
        option_name = _T("input");
        break;
    case _T('o'):
        option_name = _T("output");
        break;
    case _T('m'):
        option_name = _T("mux-option");
        break;
    case _T('v'):
        option_name = _T("version");
        break;
    case _T('h'):
    case _T('?'):
        option_name = _T("help");
        break;
    default:
        break;
    }
    return option_name;
}

int parse_one_vppmfx_option(const TCHAR *option_name, const TCHAR *strInput[], int &i, [[maybe_unused]] int nArgNum, sVppParams *vppmfx, [[maybe_unused]] sArgsData *argData) {

    if (IS_OPTION("vpp-denoise")) {
        vppmfx->denoise.enable = true;
        if (i + 1 >= nArgNum || strInput[i + 1][0] == _T('-')) {
            return 0;
        }
        i++;

        const auto paramList = std::vector<std::string>{ "mode", "strength" };

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos + 1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("enable")) {
                    bool b = false;
                    if (!cmd_string_to_bool(&b, param_val)) {
                        vppmfx->denoise.enable = b;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("mode")) {
                    int value = 0;
                    if (get_list_value(list_vpp_mfx_denoise_mode, param_val.c_str(), &value)) {
                        vppmfx->denoise.mode = (mfxDenoiseMode)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val, list_vpp_mfx_denoise_mode);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("strength")) {
                    try {
                        vppmfx->denoise.strength = std::stoi(param_val);
                    }
                    catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                try {
                    vppmfx->denoise.strength = std::stoi(param);
                } catch (...) {
                    print_cmd_error_unknown_opt_param(option_name, param, paramList);
                    return 1;
                }
            }
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-mctf"))) {
        vppmfx->mctf.enable = true;
        vppmfx->mctf.strength = 0;
        if (strInput[i+1][0] != _T('-')) {
            i++;
            int value = 0;
            if (_tcsicmp(strInput[i], _T("auto")) == 0) {
                value = 0;
            } else if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
                print_cmd_error_invalid_value(option_name, strInput[i]);
                return 1;
            }
            vppmfx->mctf.strength = value;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-vpp-mctf"))) {
        vppmfx->mctf.enable = false;
        vppmfx->mctf.strength = 0;
        if (strInput[i+1][0] != _T('-')) {
            i++;
            int value = 0;
            if (_tcsicmp(strInput[i], _T("auto")) == 0) {
                value = 0;
            } if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
                print_cmd_error_invalid_value(option_name, strInput[i]);
                return 1;
            }
            vppmfx->mctf.strength = value;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-detail-enhance"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        vppmfx->detail.enable = true;
        vppmfx->detail.strength = value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-vpp-detail-enhance"))) {
        vppmfx->detail.enable = false;
        if (strInput[i+1][0] != _T('-')) {
            i++;
            int value = 0;
            if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
                print_cmd_error_invalid_value(option_name, strInput[i]);
                return 1;
            }
            vppmfx->detail.strength = value;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-deinterlace"))) {
        i++;
        int value = get_value_from_chr(list_deinterlace, strInput[i]);
        if (PARSE_ERROR_FLAG == value) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        vppmfx->bEnable = true;
        vppmfx->deinterlace = value;
        if (vppmfx->deinterlace == MFX_DEINTERLACE_IT_MANUAL) {
            i++;
            if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_telecine_patterns, strInput[i]))) {
                print_cmd_error_invalid_value(option_name, strInput[i]);
                return 1;
            } else {
                vppmfx->telecinePattern = value;
            }
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-image-stab"))) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            vppmfx->imageStabilizer = value;
        } else if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_vpp_image_stabilizer, strInput[i]))) {
            vppmfx->imageStabilizer = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-fps-conv"))) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            vppmfx->fpsConversion = value;
        } else if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_vpp_fps_conversion, strInput[i]))) {
            vppmfx->fpsConversion = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-perc-pre-enc"))) {
        vppmfx->percPreEnc = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-vpp-perc-pre-enc"))) {
        vppmfx->percPreEnc = false;
        return 0;
    }
    return -10;
}

int ParseOneOption(const TCHAR *option_name, const TCHAR* strInput[], int& i, int nArgNum, sInputParams* pParams, sArgsData *argData) {
    if (0 == _tcscmp(option_name, _T("device"))) {
        i++;
        if (0 == _tcsnccmp(strInput[i], _T("auto"), _tcslen(_T("auto")))) {
            pParams->device = QSVDeviceNum::AUTO;
        } else {
            try {
                int value = std::stoi(strInput[i]);
                if (value >= 0) {
                    pParams->device = (QSVDeviceNum)value;
                } else {
                    print_cmd_error_invalid_value(option_name, strInput[i]);
                }
            } catch (...) {
                print_cmd_error_invalid_value(option_name, strInput[i]);
                return 1;
            }
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("codec"))) {
        i++;
        int j = 0;
        for (; list_codec_rgy[j].desc; j++) {
            if (_tcsicmp(list_codec_rgy[j].desc, strInput[i]) == 0) {
                pParams->codec = (RGY_CODEC)list_codec_rgy[j].value;
                break;
            }
        }
        if (list_codec_rgy[j].desc == nullptr) {
            print_cmd_error_invalid_value(option_name, strInput[i], list_codec_rgy);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("quality"))) {
        i++;
        int value = MFX_TARGETUSAGE_BALANCED;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->nTargetUsage = clamp(value, MFX_TARGETUSAGE_BEST_QUALITY, MFX_TARGETUSAGE_BEST_SPEED);
        } else if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_quality_for_option, strInput[i]))) {
            pParams->nTargetUsage = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_quality_for_option);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("level"))) {
        if (i+1 < nArgNum) {
            i++;
            argData->cachedlevel = strInput[i];
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("profile"))) {
        if (i+1 < nArgNum) {
            i++;
            argData->cachedprofile = strInput[i];
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("tier"))) {
        if (i+1 < nArgNum) {
            i++;
            argData->cachedtier = strInput[i];
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("output-depth"))) {
        i++;
        int value = 0;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_output_depth, strInput[i]))) {
            pParams->outputDepth = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_output_depth);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("output-csp"))) {
        i++;
        int value = 0;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_output_csp, strInput[i]))) {
            pParams->outputCsp = (RGY_CHROMAFMT)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_output_csp);
            return 1;
        }
        return 0;
    }
    if (   0 == _tcscmp(option_name, _T("sar"))
        || 0 == _tcscmp(option_name, _T("dar"))) {
        i++;
        int value[2] = { 0 };
        if (   2 != _stscanf_s(strInput[i], _T("%dx%d"), &value[0], &value[1])
            && 2 != _stscanf_s(strInput[i], _T("%d,%d"), &value[0], &value[1])
            && 2 != _stscanf_s(strInput[i], _T("%d/%d"), &value[0], &value[1])
            && 2 != _stscanf_s(strInput[i], _T("%d:%d"), &value[0], &value[1])) {
            RGY_MEMSET_ZERO(pParams->nPAR);
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if (0 == _tcscmp(option_name, _T("dar"))) {
            value[0] = -value[0];
            value[1] = -value[1];
        }
        pParams->nPAR[0] = value[0];
        pParams->nPAR[1] = value[1];
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("slices"))) {
        i++;
        try {
            pParams->nSlices = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("gop-len"))) {
        i++;
        if (0 == _tcsnccmp(strInput[i], _T("auto"), _tcslen(_T("auto")))) {
            pParams->nGOPLength = 0;
        } else {
            try {
                pParams->nGOPLength = std::stoi(strInput[i]);
            } catch (...) {
                print_cmd_error_invalid_value(option_name, strInput[i]);
                return 1;
            }
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("open-gop"))) {
        pParams->bopenGOP = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-open-gop"))) {
        pParams->bopenGOP = false;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("strict-gop"))) {
        pParams->bforceGOPSettings = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("i-adapt"))) {
        pParams->bAdaptiveI = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-i-adapt"))) {
        pParams->bAdaptiveI = false;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("b-adapt"))) {
        pParams->bAdaptiveB = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-b-adapt"))) {
        pParams->bAdaptiveB = false;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("b-pyramid"))) {
        pParams->bBPyramid = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-b-pyramid"))) {
        pParams->bBPyramid = false;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("weightb"))) {
        pParams->nWeightB = MFX_WEIGHTED_PRED_DEFAULT;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-weightb"))) {
        pParams->nWeightB = MFX_WEIGHTED_PRED_UNKNOWN;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("weightp"))) {
        pParams->nWeightP = MFX_WEIGHTED_PRED_DEFAULT;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-weightp"))) {
        pParams->nWeightP = MFX_WEIGHTED_PRED_UNKNOWN;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("repartition-check"))) {
        pParams->nRepartitionCheck = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-repartition-check"))) {
        pParams->nRepartitionCheck = false;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("fade-detect"))) {
        pParams->nFadeDetect = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-fade-detect"))) {
        pParams->nFadeDetect = false;
        return 0;
    }
    if (   0 == _tcscmp(option_name, _T("lookahead-ds"))
        || 0 == _tcscmp(option_name, _T("la-quality"))) {
        i++;
        int value = MFX_LOOKAHEAD_DS_UNKNOWN;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_lookahead_ds, strInput[i]))) {
            pParams->nLookaheadDS = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_lookahead_ds);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("trellis"))) {
        i++;
        int value = MFX_TRELLIS_UNKNOWN;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_avc_trellis_for_options, strInput[i]))) {
            pParams->nTrellis = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_avc_trellis_for_options);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("bluray")) || 0 == _tcscmp(option_name, _T("force-bluray"))) {
        pParams->nBluray = (0 == _tcscmp(option_name, _T("force-bluray"))) ? 2 : 1;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("nv12"))) {
        pParams->ColorFormat = MFX_FOURCC_NV12;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("la"))) {
        i++;
        try {
            pParams->rcParam.bitrate = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        pParams->rcParam.encMode = MFX_RATECONTROL_LA;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("icq"))) {
        i++;
        try {
            pParams->rcParam.icqQuality = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        pParams->rcParam.encMode = MFX_RATECONTROL_ICQ;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("la-icq"))) {
        i++;
        try {
            pParams->rcParam.icqQuality = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        pParams->rcParam.encMode = MFX_RATECONTROL_LA_ICQ;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("la-hrd"))) {
        i++;
        try {
            pParams->rcParam.bitrate = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        pParams->rcParam.encMode = MFX_RATECONTROL_LA_HRD;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vcm"))) {
        i++;
        try {
            pParams->rcParam.bitrate = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        pParams->rcParam.encMode = MFX_RATECONTROL_VCM;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vbr"))) {
        i++;
        try {
            pParams->rcParam.bitrate = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        pParams->rcParam.encMode = MFX_RATECONTROL_VBR;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("cbr"))) {
        i++;
        try {
            pParams->rcParam.bitrate = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        pParams->rcParam.encMode = MFX_RATECONTROL_CBR;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("avbr"))) {
        i++;
        try {
            pParams->rcParam.bitrate = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        pParams->rcParam.encMode = MFX_RATECONTROL_AVBR;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("qvbr"))) {
        i++;
        try {
            pParams->rcParam.bitrate = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        pParams->rcParam.encMode = MFX_RATECONTROL_QVBR;
        return 0;
    }
    if (   0 == _tcscmp(option_name, _T("qvbr-q"))
        || 0 == _tcscmp(option_name, _T("qvbr-quality"))) {
        i++;
        try {
            pParams->rcParam.qvbrQuality = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        pParams->rcParam.encMode = MFX_RATECONTROL_QVBR;
        return 0;
    }

    if (IS_OPTION("dynamic-rc")) {
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        bool rc_mode_defined = false;
        auto paramList = std::vector<std::string>{ "start", "end", "max-bitrate", "avbr-accuracy", "avbr-convergence", "qvbr-quality" };
        for (int j = 0; list_rc_mode[j].desc; j++) {
            paramList.push_back(tolowercase(tchar_to_string(list_rc_mode[j].desc)));
        }
        QSVRCParam rcPrm;
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("start")) {
                    try {
                        rcPrm.start = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("end")) {
                    try {
                        rcPrm.end = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("cqp")) {
                    int ret = rcPrm.qp.parse(strInput[i]);
                    if (ret != 0) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    rcPrm.encMode = MFX_RATECONTROL_CQP;
                    rc_mode_defined = true;
                    continue;
                }
                if (param_arg == _T("icq")) {
                    try {
                        rcPrm.icqQuality = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    rcPrm.encMode = MFX_RATECONTROL_ICQ;
                    rc_mode_defined = true;
                    continue;
                }
                if (param_arg == _T("la-icq")) {
                    try {
                        rcPrm.icqQuality = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    rcPrm.encMode = MFX_RATECONTROL_LA_ICQ;
                    rc_mode_defined = true;
                    continue;
                }
                int temp = 0;
                if (get_list_value(list_rc_mode, tolowercase(param_arg).c_str(), &temp)) {
                    try {
                        rcPrm.bitrate = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    rcPrm.encMode = temp;
                    rc_mode_defined = true;
                    continue;
                }
                if (param_arg == _T("max-bitrate")) {
                    try {
                        rcPrm.maxBitrate = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("avbr-accuracy")) {
                    try {
                        rcPrm.avbrAccuarcy = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("avbr-convergence")) {
                    try {
                        rcPrm.avbrConvergence = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("qvbr-quality")) {
                    try {
                        rcPrm.qvbrQuality = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                pos = param.find_first_of(_T(":"));
                if (pos != std::string::npos) {
                    auto param_val0 = param.substr(0, pos);
                    auto param_val1 = param.substr(pos+1);
                    try {
                        rcPrm.start = std::stoi(param_val0);
                        rcPrm.end   = std::stoi(param_val1);
                    } catch (...) {
                        print_cmd_error_invalid_value(option_name, param);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        if (!rc_mode_defined) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("rate control mode unspecified!"));
            return 1;
        }
        if (rcPrm.start < 0) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("start frame ID unspecified!"));
            return 1;
        }
        if (rcPrm.end > 0 && rcPrm.start > rcPrm.end) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("start frame ID must be smaller than end frame ID!"));
            return 1;
        }
        pParams->dynamicRC.push_back(rcPrm);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("fallback-rc"))) {
        pParams->fallbackRC = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-fallback-rc"))) {
        pParams->fallbackRC = false;
        return 0;
    }
    if (   0 == _tcscmp(option_name, _T("max-bitrate"))
        || 0 == _tcscmp(option_name, _T("maxbitrate"))) //互換性のため
    {
        i++;
        try {
            pParams->rcParam.maxBitrate = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vbv-bufsize"))) {
        i++;
        try {
            pParams->rcParam.vbvBufSize = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("la-depth"))) {
        i++;
        try {
            pParams->nLookaheadDepth = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("la-window-size"))) {
        i++;
        try {
            pParams->nWinBRCSize = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("cqp"))) {
        i++;
        int ret = pParams->rcParam.qp.parse(strInput[i]);
        if (ret != 0) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        pParams->rcParam.encMode = MFX_RATECONTROL_CQP;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("avbr-unitsize"))) {
        i++;
        try {
            pParams->rcParam.avbrConvergence = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    //if (0 == _tcscmp(option_name, _T("avbr-range")))
    //{
    //    double accuracy;
    //    if (1 != _stscanf_s(strArgument, _T("%f"), &accuracy)) {
    //        print_cmd_error_invalid_value(option_name, strInput[i]);
    //        return 1;
    //    }
    //    pParams->rcParam.avbrAccuarcy = (mfxU16)(accuracy * 10 + 0.5);
    //    return 0;
    //}
    if (0 == _tcscmp(option_name, _T("fixed-func"))) {
        pParams->functionMode = QSVFunctionMode::FF;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("function-mode"))) {
        i++;
        int v = 0;
        if ((v = get_value_from_chr(list_qsv_function_mode, strInput[i])) == PARSE_ERROR_FLAG) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        pParams->functionMode = (QSVFunctionMode)v;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("hyper-mode"))) {
        i++;
        int v = 0;
        if ((v = get_value_from_chr(list_hyper_mode, strInput[i])) == PARSE_ERROR_FLAG) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        pParams->hyperMode = (mfxHyperMode)v;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("ref"))) {
        i++;
        try {
            pParams->nRef = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("bframes"))) {
        i++;
        try {
            pParams->GopRefDist = std::stoi(strInput[i])+1;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("gop-ref-dist"))) {
        i++;
        try {
            pParams->GopRefDist = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("cavlc"))) {
        pParams->bCAVLC = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("rdo"))) {
        pParams->bRDO = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("tune"))) {
        i++;
        auto values = split(strInput[i], _T(","), true);
        if (values.size() == 0) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        for (auto& str : values) {
            int v = 0;
            if ((v = get_value_from_chr(list_enc_tune_quality_mode, str.c_str())) == PARSE_ERROR_FLAG) {
                print_cmd_error_invalid_value(option_name, str, list_enc_tune_quality_mode);
                return 1;
            }
            pParams->tuneQuality |= (decltype(pParams->tuneQuality))v;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("scenario-info"))) {
        i++;
        int v = 0;
        if ((v = get_value_from_chr(list_scenario_info, strInput[i])) == PARSE_ERROR_FLAG) {
            print_cmd_error_invalid_value(option_name, strInput[i], list_scenario_info);
            return 1;
        }
        pParams->scenarioInfo = v;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("extbrc"))) {
        pParams->extBRC = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-extbrc"))) {
        pParams->extBRC = false;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("adapt-ref"))) {
        pParams->adaptiveRef = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-adapt-ref"))) {
        pParams->adaptiveRef = false;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("adapt-ltr"))) {
        pParams->adaptiveLTR = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-adapt-ltr"))) {
        pParams->adaptiveLTR = false;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("adapt-cqm"))) {
        pParams->adaptiveCQM = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-adapt-cqm"))) {
        pParams->adaptiveCQM = false;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("mbbrc"))) {
        pParams->bMBBRC = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-mbbrc"))) {
        pParams->bMBBRC = false;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("intra-refresh-cycle"))) {
        i++;
        try {
            pParams->intraRefreshCycle = std::stoi(strInput[i]) + 1;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("max-framesize"))) {
        i++;
        try {
            pParams->maxFrameSize = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("max-framesize-i"))) {
        i++;
        try {
            pParams->maxFrameSizeI = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("max-framesize-p"))) {
        i++;
        try {
            pParams->maxFrameSizeP = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-deblock"))) {
        pParams->bNoDeblock = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("ctu"))) {
        i++;
        int value = 0;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_hevc_ctu, strInput[i]))) {
            pParams->hevc_ctu = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_hevc_ctu);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("sao"))) {
        i++;
        int value = MFX_SAO_UNKNOWN;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_hevc_sao, strInput[i]))) {
            pParams->hevc_sao = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_hevc_sao);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-tskip"))) {
        pParams->hevc_tskip = MFX_CODINGOPTION_OFF;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("tskip"))) {
        pParams->hevc_tskip = MFX_CODINGOPTION_ON;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("hevc-gpb"))) {
        pParams->hevc_gpb = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-hevc-gpb"))) {
        pParams->hevc_gpb = false;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("tile-row"))) {
        i++;
        try {
            pParams->av1.tile_row = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("tile-col"))) {
        i++;
        try {
            pParams->av1.tile_col = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("qpmin")) || 0 == _tcscmp(option_name, _T("qp-min"))) {
        i++;
        int ret = pParams->qpMin.parse(strInput[i]);
        if (ret != 0) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("qpmax")) || 0 == _tcscmp(option_name, _T("qp-max"))) {
        i++;
        int ret = pParams->qpMax.parse(strInput[i]);
        if (ret != 0) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("qp-offset"))) {
        i++;
        auto values = split(strInput[i], _T(":"), true);
        if (values.size() == 0) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if (values.size() > 8) {
            print_cmd_error_invalid_value(option_name, strInput[i], strsprintf(_T("qp-offset value could be set up to 8 layers, but was set for %d layers.\n"), (int)values.size()));
            return 1;
        }
        int iv = 0;
        for (; iv < (int)values.size(); iv++) {
            TCHAR *eptr = nullptr;
            int v = _tcstol(values[iv].c_str(), &eptr, 0);
            if (v == 0 && (eptr != nullptr || *eptr == ' ')) {
                print_cmd_error_invalid_value(option_name, strInput[iv]);
                return 1;
            }
            if (v < -51 || v > 51) {
                print_cmd_error_invalid_value(option_name, strInput[i], _T("qp-offset value should be in range of -51 - 51.\n"));
                return 1;
            }
            pParams->pQPOffset[iv] = (int8_t)v;
        }
        for (; iv < _countof(pParams->pQPOffset); iv++) {
            pParams->pQPOffset[iv] = pParams->pQPOffset[iv-1];
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("mv-scaling"))) {
        i++;
        try {
            pParams->nMVCostScaling = std::stoi(strInput[i]);
            pParams->bGlobalMotionAdjust = true;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("direct-bias-adjust"))) {
        pParams->bDirectBiasAdjust = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-direct-bias-adjust"))) {
        pParams->bDirectBiasAdjust = false;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("inter-pred"))) {
        i++;
        int v = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_pred_block_size) - 1) {
            print_cmd_error_invalid_value(option_name, strInput[i], list_pred_block_size);
            return 1;
        }
        pParams->nInterPred = list_pred_block_size[v].value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("intra-pred"))) {
        i++;
        int v = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_pred_block_size) - 1) {
            print_cmd_error_invalid_value(option_name, strInput[i], list_pred_block_size);
            return 1;
        }
        pParams->nIntraPred = list_pred_block_size[v].value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("mv-precision"))) {
        i++;
        int v = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_mv_presicion) - 1) {
            print_cmd_error_invalid_value(option_name, strInput[i], list_mv_presicion);
            return 1;
        }
        pParams->nMVPrecision = list_mv_presicion[v].value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("mv-search"))) {
        i++;
        int v = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        pParams->MVSearchWindow.first = clamp(v, 0, 128);
        pParams->MVSearchWindow.second = clamp(v, 0, 128);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("sharpness"))) {
        i++;
        try {
            pParams->nVP8Sharpness = std::stoi(strInput[i]);
            if (pParams->nVP8Sharpness < 0 || 7 < pParams->nVP8Sharpness) {
                print_cmd_error_invalid_value(option_name, strInput[i], _T("Sharpness should be in range of 0 - 7."));
                return 1;
            }
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
#if D3D_SURFACES_SUPPORT
    if (0 == _tcscmp(option_name, _T("disable-d3d"))) {
        pParams->memType = SYSTEM_MEMORY;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("d3d9"))) {
        pParams->memType = D3D9_MEMORY;
        return 0;
    }
#if MFX_D3D11_SUPPORT
    if (0 == _tcscmp(option_name, _T("d3d11"))) {
        pParams->memType = D3D11_MEMORY;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("d3d"))) {
        pParams->memType = HW_MEMORY;
        return 0;
    }
#else
    if (0 == _tcscmp(option_name, _T("d3d"))) {
        pParams->memType = D3D9_MEMORY;
        return 0;
    }
#endif //MFX_D3D11_SUPPORT
#endif //D3D_SURFACES_SUPPORT
#if LIBVA_SUPPORT
    if (0 == _tcscmp(option_name, _T("va"))) {
        pParams->memType = D3D9_MEMORY;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("disable-va"))) {
        pParams->memType = SYSTEM_MEMORY;
        return 0;
    }
#endif //#if LIBVA_SUPPORT
    if (0 == _tcscmp(option_name, _T("aud"))) {
        pParams->bOutputAud = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("pic-struct"))) {
        pParams->bOutputPicStruct = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("buf-period"))) {
        pParams->bufPeriodSEI = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("repeat-headers"))) {
        pParams->repeatHeaders = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-repeat-pps"))
        || 0 == _tcscmp(option_name, _T("no-repeat-headers"))) {
        pParams->repeatHeaders = false;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("async-depth"))) {
        i++;
        int v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        } else if (v < 0 || QSV_ASYNC_DEPTH_MAX < v) {
            print_cmd_error_invalid_value(option_name, strInput[i], strsprintf(_T("async-depth should be in range of 0 - %d."), QSV_ASYNC_DEPTH_MAX));
            return 1;
        }
        pParams->nAsyncDepth = v;
        return 0;
    }
#if ENABLE_SESSION_THREAD_CONFIG
    if (0 == _tcscmp(option_name, _T("session-threads"))) {
        i++;
        int v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v) || v < 0 || QSV_SESSION_THREAD_MAX < v) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        pParams->nSessionThreads = v;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("session-thread-priority"))
        || 0 == _tcscmp(option_name, _T("session-threads-priority"))) {
        i++;
        mfxI32 v;
        if (PARSE_ERROR_FLAG == (v = get_value_from_chr(list_priority, strInput[i]))
            && 1 != _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_log_level) - 1) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        pParams->nSessionThreadPriority = v;
        return 0;
    }
#endif
    if (0 == _tcscmp(option_name, _T("input-buf"))) {
        i++;
        try {
            argData->nTmpInputBuf = std::stoi(strInput[i]);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if (argData->nTmpInputBuf < 0) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("input-buf should be positive value."));
            return 1;
        }
        return 0;
    }
#if defined(_WIN32) || defined(_WIN64)
    if (0 == _tcscmp(option_name, _T("mfx-thread"))) {
        i++;
        try {
            pParams->nSessionThreads = std::min(std::stoi(strInput[i]), RGY_OUTPUT_BUF_MB_MAX);
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if (pParams->nSessionThreads < -1) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("mfx-thread should be positive value."));
            return 1;
        }
        return 0;
    }
#endif
    if (0 == _tcscmp(option_name, _T("gpu-copy"))) {
        pParams->gpuCopy = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("min-memory"))) {
        pParams->ctrl.threadOutput = 0;
        pParams->ctrl.threadAudio = 0;
        pParams->nAsyncDepth = 1;
        argData->nTmpInputBuf = 1;
        pParams->ctrl.outputBufSizeMB = 0;
        pParams->nSessionThreads = 2;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("benchmark"))) {
        i++;
        pParams->bBenchmark = true;
        pParams->common.outputFilename = strInput[i];
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("bench-quality"))) {
        i++;
        pParams->bBenchmark = true;
        if (0 == _tcscmp(strInput[i], _T("all"))) {
            pParams->nBenchQuality = 0xffffffff;
        } else {
            pParams->nBenchQuality = 0;
            auto list = split(tstring(strInput[i]), _T(","));
            for (const auto& str : list) {
                int nQuality = 0;
                if (1 == _stscanf(str.c_str(), _T("%d"), &nQuality)) {
                    pParams->nBenchQuality |= 1 << nQuality;
                } else if ((nQuality = get_value_from_chr(list_quality_for_option, strInput[i])) > 0) {
                    pParams->nBenchQuality |= 1 << nQuality;
                } else {
                    print_cmd_error_invalid_value(option_name, strInput[i], list_quality_for_option);
                    return 1;
                }
            }
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("python"))) {
        i++;
        pParams->pythonPath = strInput[i];
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("timer-period-tuning"))) {
        pParams->bDisableTimerPeriodTuning = false;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-timer-period-tuning"))) {
        pParams->bDisableTimerPeriodTuning = true;
        return 0;
    }

    auto ret = parse_one_input_option(option_name, strInput, i, nArgNum, &pParams->input, &pParams->inprm, argData);
    if (ret >= 0) return ret;

    ret = parse_one_common_option(option_name, strInput, i, nArgNum, &pParams->common, argData);
    if (ret >= 0) return ret;

    ret = parse_one_ctrl_option(option_name, strInput, i, nArgNum, &pParams->ctrl, argData);
    if (ret >= 0) return ret;

    ret = parse_one_vppmfx_option(option_name, strInput, i, nArgNum, &pParams->vppmfx, argData);
    if (ret >= 0) return ret;

    ret = parse_one_vpp_option(option_name, strInput, i, nArgNum, &pParams->vpp, argData);
    if (ret >= 0) return ret;

    print_cmd_error_unknown_opt(strInput[i]);
    return 1;
}

int parse_cmd(sInputParams *pParams, const TCHAR *strInput[], int nArgNum, bool ignore_parse_err) {
    if (!pParams) {
        return 0;
    }

    bool debug_cmd_parser = false;
    for (int i = 1; i < nArgNum; i++) {
        if (tstring(strInput[i]) == _T("--debug-cmd-parser")) {
            debug_cmd_parser = true;
            break;
        }
    }

    if (debug_cmd_parser) {
        for (int i = 1; i < nArgNum; i++) {
            _ftprintf(stderr, _T("arg[%3d]: %s\n"), i, strInput[i]);
        }
    }

    sArgsData argsData;

    for (int i = 1; i < nArgNum; i++) {
        if (strInput[i] == nullptr) {
            return MFX_ERR_NULL_PTR;
        }

        const TCHAR *option_name = nullptr;

        if (strInput[i][0] == _T('|')) {
            break;
        } else if (strInput[i][0] == _T('-')) {
            if (strInput[i][1] == _T('-')) {
                option_name = &strInput[i][2];
            } else if (strInput[i][2] == _T('\0')) {
                if (nullptr == (option_name = cmd_short_opt_to_long(strInput[i][1]))) {
                    print_cmd_error_invalid_value(tstring(), tstring(), strsprintf(_T("Unknown option: \"%s\""), strInput[i]));
                    return 1;
                }
            } else {
                if (ignore_parse_err) continue;
                print_cmd_error_invalid_value(tstring(), tstring(), strsprintf(_T("Invalid option: \"%s\""), strInput[i]));
                return 1;
            }
        }

        if (option_name == nullptr) {
            if (ignore_parse_err) continue;
            print_cmd_error_unknown_opt(strInput[i]);
            return 1;
        }
        if (debug_cmd_parser) {
            _ftprintf(stderr, _T("parsing %3d: %s: "), i, strInput[i]);
        }
        auto sts = ParseOneOption(option_name, strInput, i, nArgNum, pParams, &argsData);
        if (debug_cmd_parser) {
            _ftprintf(stderr, _T("%s\n"), (sts == 0) ? _T("OK") : _T("ERR"));
        }
        if (!ignore_parse_err && sts != 0) {
            return sts;
        }
    }

    //parse cached profile and level
    if (argsData.cachedlevel.length() > 0) {
        const auto desc = get_level_list(pParams->codec);
        int value = 0;
        bool bParsed = false;
        if (desc != nullptr) {
            if (PARSE_ERROR_FLAG != (value = get_value_from_chr(desc, argsData.cachedlevel.c_str()))) {
                pParams->CodecLevel = value;
                bParsed = true;
            } else {
                double val_float = 0.0;
                if (1 == _stscanf_s(argsData.cachedlevel.c_str(), _T("%lf"), &val_float)) {
                    value = (int)(val_float * 10 + 0.5);
                    if (value == desc[get_cx_index(desc, value)].value) {
                        pParams->CodecLevel = value;
                        bParsed = true;
                    } else {
                        value = (int)(val_float + 0.5);
                        if (value == desc[get_cx_index(desc, value)].value) {
                            pParams->CodecLevel = value;
                            bParsed = true;
                        }
                    }
                }
            }
        }
        if (!bParsed) {
            print_cmd_error_invalid_value(_T("level"), argsData.cachedlevel, std::vector<std::pair<RGY_CODEC, const CX_DESC *>>{
                { RGY_CODEC_H264,  list_avc_level },
                { RGY_CODEC_HEVC,  list_hevc_level },
                { RGY_CODEC_MPEG2, list_mpeg2_level },
                { RGY_CODEC_VP9,   list_vp9_level },
                { RGY_CODEC_AV1,   list_av1_level }
            });
            return 1;
        }
    }
    if (argsData.cachedprofile.length() > 0) {
        const auto desc = get_profile_list(pParams->codec);
        int value = 0;
        if (desc != nullptr && PARSE_ERROR_FLAG != (value = get_value_from_chr(desc, argsData.cachedprofile.c_str()))) {
            pParams->CodecProfile = value;
        } else {
            print_cmd_error_invalid_value(_T("profile"), argsData.cachedprofile, std::vector<std::pair<RGY_CODEC, const CX_DESC *>>{
                { RGY_CODEC_H264,  list_avc_profile },
                { RGY_CODEC_HEVC,  list_hevc_profile },
                { RGY_CODEC_MPEG2, list_mpeg2_profile },
                { RGY_CODEC_VP9,   list_vp9_profile },
                { RGY_CODEC_AV1,   list_av1_profile }
            });
            return 1;
        }
    }
    if (pParams->codec == RGY_CODEC_HEVC) {
        if (pParams->outputDepth > 8
            && (pParams->CodecProfile == 0 || pParams->CodecProfile == MFX_PROFILE_HEVC_MAIN)) {
            pParams->CodecProfile = MFX_PROFILE_HEVC_MAIN10;
        }
        if (pParams->outputCsp != RGY_CHROMAFMT_YUV420) {
            pParams->CodecProfile = MFX_PROFILE_HEVC_REXT;
        }
        if (pParams->CodecProfile == MFX_PROFILE_HEVC_MAIN10) {
            pParams->outputDepth = 10;
        }
        if (argsData.cachedtier.length() > 0) {
            int value = 0;
            if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_hevc_tier, argsData.cachedtier.c_str()))) {
                pParams->hevc_tier = value;
            } else {
                print_cmd_error_invalid_value(_T("tier"), argsData.cachedtier, list_hevc_tier);
                return 1;
            }
        }
    }
    if (pParams->codec == RGY_CODEC_AV1) {
        if (pParams->outputCsp != RGY_CHROMAFMT_YUV420) {
            pParams->CodecProfile = MFX_PROFILE_AV1_PRO;
        }
    }

    if (!FOR_AUO) {
        // check if all mandatory parameters were set
        if (pParams->common.inputFilename.length() == 0) {
            _ftprintf(stderr, _T("Source file name not found.\n"));
            return 1;
        }

        if (pParams->common.outputFilename.length() == 0) {
            _ftprintf(stderr, _T("Destination file name not found.\n"));
            return 1;
        }

        if (pParams->common.chapterFile.length() > 0 && pParams->common.copyChapter) {
            _ftprintf(stderr, _T("--chapter and --chapter-copy are both set.\nThese could not be set at the same time.\n"));
            return 1;
        }
        if (pParams->common.inputFilename != _T("-")
            && pParams->common.outputFilename != _T("-")
            && rgy_path_is_same(pParams->common.inputFilename, pParams->common.outputFilename)) {
            _ftprintf(stderr, _T("destination file is equal to source file!\n"));
            return 1;
        }
    }

    pParams->nTargetUsage = clamp(pParams->nTargetUsage, MFX_TARGETUSAGE_BEST_QUALITY, MFX_TARGETUSAGE_BEST_SPEED);

    // if nv12 option isn't specified, input YUV file is expected to be in YUV420 color format
    if (!pParams->ColorFormat) {
        pParams->ColorFormat = MFX_FOURCC_YV12;
    }

    //set input buffer size
    if (argsData.nTmpInputBuf == 0) {
        argsData.nTmpInputBuf = QSV_DEFAULT_INPUT_BUF_HW;
    }
    pParams->nInputBufSize = (mfxU16)clamp(argsData.nTmpInputBuf, QSV_INPUT_BUF_MIN, QSV_INPUT_BUF_MAX);

    return 0;
}

#if defined(_WIN32) || defined(_WIN64)
int parse_cmd(sInputParams *pParams, const char *cmda, bool ignore_parse_err) {
    if (cmda == nullptr) {
        return 0;
    }
    std::wstring cmd = char_to_wstring(cmda);
    int argc = 0;
    auto argvw = CommandLineToArgvW(cmd.c_str(), &argc);
    if (argc <= 1) {
        return 0;
    }
    vector<tstring> argv_tstring;
    if (wcslen(argvw[0]) != 0) {
        argv_tstring.push_back(_T("")); // 最初は実行ファイルのパスが入っているのを模擬するため、空文字列を入れておく
    }
    for (int i = 0; i < argc; i++) {
        argv_tstring.push_back(wstring_to_tstring(argvw[i]));
    }
    LocalFree(argvw);

    vector<TCHAR *> argv_tchar;
    for (size_t i = 0; i < argv_tstring.size(); i++) {
        argv_tchar.push_back((TCHAR *)argv_tstring[i].data());
    }
    argv_tchar.push_back(_T("")); // 最後に空白を追加
    const TCHAR **strInput = (const TCHAR **)argv_tchar.data();
    return parse_cmd(pParams, strInput, (int)argv_tchar.size() - 1 /*最後の空白の分*/, ignore_parse_err);
}
#endif


tstring gen_cmd(const sVppParams *param, const sVppParams *defaultPrm, bool save_disabled_prm) {
#define OPT_FLOAT(str, opt, prec) if ((param->opt) != (defaultPrm->opt)) cmd << _T(" ") << (str) << _T(" ") << std::setprecision(prec) << (param->opt);
#define OPT_NUM(str, opt) if ((param->opt) != (defaultPrm->opt)) cmd << _T(" ") << (str) << _T(" ") << (int)(param->opt);
#define OPT_LST(str, opt, list) if ((param->opt) != (defaultPrm->opt)) cmd << _T(" ") << (str) << _T(" ") << get_chr_from_value(list, (param->opt));
#define OPT_BOOL(str_true, str_false, opt) if ((param->opt) != (defaultPrm->opt)) cmd << _T(" ") << ((param->opt) ? (str_true) : (str_false));
#define OPT_BOOL_VAL(str_true, str_false, opt, val) { \
    if ((param->opt) != (defaultPrm->opt) || (save_disabled_prm && (param->val) != (defaultPrm->val))) { \
        cmd << _T(" ") << ((param->opt) ? (str_true) : (str_false)) <<  _T(" ") << (param->val); \
    } \
}

#define OPT_TCHAR(str, opt) if ((param->opt) && _tcslen(param->opt)) cmd << _T(" ") << str << _T(" ") << (param->opt);
#define OPT_TSTR(str, opt) if (param->opt.length() > 0) cmd << _T(" ") << str << _T(" ") << param->opt.c_str();
#define OPT_CHAR(str, opt) if ((param->opt) && _tcslen(param->opt)) cmd << _T(" ") << str << _T(" ") << char_to_tstring(param->opt);
#define OPT_STR(str, opt) if (param->opt.length() > 0) cmd << _T(" ") << str << _T(" ") << char_to_tstring(param->opt).c_str();
#define OPT_CHAR_PATH(str, opt) if ((param->opt) && _tcslen(param->opt)) cmd << _T(" ") << str << _T(" \"") << (param->opt) << _T("\"");
#define OPT_STR_PATH(str, opt) if (param->opt.length() > 0) cmd << _T(" ") << str << _T(" \"") << (param->opt.c_str()) << _T("\"");

#define ADD_NUM(str, opt) if ((param->opt) != (defaultPrm->opt)) tmp << _T(",") << (str) << _T("=") << (param->opt);
#define ADD_LST(str, opt, list) if ((param->opt) != (defaultPrm->opt)) tmp << _T(",") << (str) << _T("=") << get_chr_from_value(list, (param->opt));

    std::basic_stringstream<TCHAR> tmp;
    std::basic_stringstream<TCHAR> cmd;

    OPT_LST(_T("--vpp-deinterlace"), deinterlace, list_deinterlace);
    OPT_BOOL_VAL(_T("--vpp-detail-enhance"), _T("--no-vpp-detail-enhance"), detail.enable, detail.strength);

    if (param->denoise != defaultPrm->denoise) {
        tmp.str(tstring());
        if (!param->denoise.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (param->denoise.enable || save_disabled_prm) {
            ADD_LST(_T("mode"), denoise.mode, list_vpp_mfx_denoise_mode);
            ADD_NUM(_T("strength"), denoise.strength);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-denoise ") << tmp.str().substr(1);
        }
    }
    if (param->mctf.enable && param->mctf.strength == 0) {
        cmd << _T(" --vpp-mctf auto");
    } else {
        OPT_BOOL_VAL(_T("--vpp-mctf"), _T("--no-vpp-mctf"), mctf.enable, mctf.strength);
    }
    OPT_LST(_T("--vpp-fps-conv"), fpsConversion, list_vpp_fps_conversion);
    OPT_LST(_T("--vpp-image-stab"), imageStabilizer, list_vpp_image_stabilizer);
    OPT_BOOL(_T("--vpp-perc-pre-enc"), _T("--no-vpp-perc-pre-enc"), percPreEnc);

#if 0
    if (param->colorspace != defaultPrm->colorspace) {
        tmp.str(tstring());
        if (!pParams->colorspace.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (pParams->colorspace.enable || save_disabled_prm) {
            for (size_t i = 0; i < pParams->colorspace.convs.size(); i++) {
                auto from = pParams->colorspace.convs[i].from;
                auto to = pParams->colorspace.convs[i].to;
                if (from.matrix != to.matrix) {
                    tmp << _T(",matrix=");
                    tmp << get_cx_desc(list_colormatrix, from.matrix);
                    tmp << _T(":");
                    tmp << get_cx_desc(list_colormatrix, to.matrix);
                }
                if (false && from.colorprim != to.colorprim) {
                    tmp << _T(",colorprim=");
                    tmp << get_cx_desc(list_colorprim, from.colorprim);
                    tmp << _T(":");
                    tmp << get_cx_desc(list_colorprim, to.colorprim);
                }
                if (false && from.transfer != to.transfer) {
                    tmp << _T(",transfer=");
                    tmp << get_cx_desc(list_transfer, from.transfer);
                    tmp << _T(":");
                    tmp << get_cx_desc(list_transfer, to.transfer);
                }
                if (from.colorrange != to.colorrange) {
                    tmp << _T(",range=");
                    tmp << get_cx_desc(list_colorrange, from.colorrange);
                    tmp << _T(":");
                    tmp << get_cx_desc(list_colorrange, to.colorrange);
                }
                ADD_BOOL(_T("approx_gamma"), colorspace.convs[i].approx_gamma);
                ADD_BOOL(_T("scene_ref"), colorspace.convs[i].scene_ref);
                ADD_LST(_T("hdr2sdr"), colorspace.hdr2sdr.tonemap, list_vpp_hdr2sdr);
                ADD_FLOAT(_T("ldr_nits"), colorspace.hdr2sdr.ldr_nits, 1);
                ADD_FLOAT(_T("source_peak"), colorspace.hdr2sdr.hdr_source_peak, 1);
                ADD_FLOAT(_T("a"), colorspace.hdr2sdr.hable.a, 3);
                ADD_FLOAT(_T("b"), colorspace.hdr2sdr.hable.b, 3);
                ADD_FLOAT(_T("c"), colorspace.hdr2sdr.hable.c, 3);
                ADD_FLOAT(_T("d"), colorspace.hdr2sdr.hable.d, 3);
                ADD_FLOAT(_T("e"), colorspace.hdr2sdr.hable.e, 3);
                ADD_FLOAT(_T("f"), colorspace.hdr2sdr.hable.f, 3);
                ADD_FLOAT(_T("transition"), colorspace.hdr2sdr.mobius.transition, 3);
                ADD_FLOAT(_T("peak"), colorspace.hdr2sdr.mobius.peak, 3);
                ADD_FLOAT(_T("contrast"), colorspace.hdr2sdr.reinhard.contrast, 3);
            }
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-colorspace ") << tmp.str().substr(1);
        } else if (pParams->colorspace.enable) {
            cmd << _T(" --vpp-colorspace");
        }
    }
#if ENABLE_CUSTOM_VPP
    OPT_CHAR_PATH(_T("--vpp-delogo"), delogo.pFilePath);
    OPT_CHAR(_T("--vpp-delogo-select"), delogo.pSelect);
    OPT_NUM(_T("--vpp-delogo-depth"), delogo.depth);
    if (pParams->delogo.posOffset.first > 0 || pParams->delogo.posOffset.second > 0) {
        cmd << _T(" --vpp-delogo-pos ") << pParams->delogo.posOffset.first << _T("x") << pParams->delogo.posOffset.second;
    }
    OPT_NUM(_T("--vpp-delogo-y"), delogo.YOffset);
    OPT_NUM(_T("--vpp-delogo-cb"), delogo.CbOffset);
    OPT_NUM(_T("--vpp-delogo-cr"), delogo.CrOffset);
#endif //#if ENABLE_CUSTOM_VPP
#endif
    return cmd.str();

#undef OPT_FLOAT
#undef OPT_NUM
#undef OPT_LST
#undef OPT_BOOL
#undef OPT_BOOL_VAL
#undef OPT_TCHAR
#undef OPT_TSTR
#undef OPT_CHAR
#undef OPT_STR
#undef OPT_CHAR_PATH
#undef OPT_STR_PATH
}

#pragma warning (push)
#pragma warning (disable: 4127)
tstring gen_cmd(const sInputParams *pParams, bool save_disabled_prm) {
    std::basic_stringstream<TCHAR> tmp;
    std::basic_stringstream<TCHAR> cmd;
    sInputParams encPrmDefault;

#define OPT_FLOAT(str, opt, prec) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << (str) << _T(" ") << std::setprecision(prec) << (pParams->opt);
#define OPT_NUM(str, opt) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << (str) << _T(" ") << (int)(pParams->opt);
#define OPT_TRI(str_true, str_false, opt, val_true, val_false) \
    if ((pParams->opt) != (encPrmDefault.opt) && pParams->opt != MFX_CODINGOPTION_UNKNOWN) { \
        if ((pParams->opt) == (val_true)) { \
            cmd << _T(" ") << (str_true); \
        } else if ((pParams->opt) == (val_false)) { \
            cmd << _T(" ") << (str_false); \
        } \
    }

#define OPT_LST(str, opt, list) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << (str) << _T(" ") << get_chr_from_value(list, decltype(list[0].value)(pParams->opt));
#define OPT_QP(str, qp, force) { \
    if ((force) \
    || (pParams->qp.qpI) != (encPrmDefault.qp.qpI) \
    || (pParams->qp.qpP) != (encPrmDefault.qp.qpP) \
    || (pParams->qp.qpB) != (encPrmDefault.qp.qpB)) { \
        if ((pParams->qp.qpI) == (pParams->qp.qpP) && (pParams->qp.qpI) == (pParams->qp.qpB)) { \
            cmd << _T(" ") << (str) << _T(" ") << (int)(pParams->qp.qpI); \
        } else if ((pParams->qp.qpP) == (pParams->qp.qpB)) { \
            cmd << _T(" ") << (str) << _T(" ") << (int)(pParams->qp.qpI) << _T(":") << (int)(pParams->qp.qpP); \
        } else { \
            cmd << _T(" ") << (str) << _T(" ") << (int)(pParams->qp.qpI) << _T(":") << (int)(pParams->qp.qpP) << _T(":") << (int)(pParams->qp.qpB); \
        } \
    } \
}
#define OPT_BOOL(str_true, str_false, opt) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << ((pParams->opt) ? (str_true) : (str_false));
#define OPT_BOOL_OPT(str_true, str_false, opt) if (pParams->opt.has_value()) cmd << _T(" ") << ((pParams->opt.value()) ? (str_true) : (str_false));
#define OPT_BOOL_VAL(str_true, str_false, opt, val) { \
    if ((pParams->opt) != (encPrmDefault.opt) || (save_disabled_prm && (pParams->val) != (encPrmDefault.val))) { \
        cmd << _T(" ") << ((pParams->opt) ? (str_true) : (str_false)) <<  _T(" ") << (pParams->val); \
    } \
}
#define OPT_CHAR(str, opt) if ((pParams->opt) && (pParams->opt[0] != 0)) cmd << _T(" ") << str << _T(" ") << (pParams->opt);
#define OPT_STR(str, opt) if (pParams->opt.length() > 0) cmd << _T(" ") << str << _T(" ") << (pParams->opt.c_str());
#define OPT_CHAR_PATH(str, opt) if ((pParams->opt) && (pParams->opt[0] != 0)) cmd << _T(" ") << str << _T(" \"") << (pParams->opt) << _T("\"");
#define OPT_STR_PATH(str, opt) if (pParams->opt.length() > 0) cmd << _T(" ") << str << _T(" \"") << (pParams->opt.c_str()) << _T("\"");

    OPT_NUM(_T("-d"), device);
    cmd << _T(" -c ") << get_chr_from_value(list_codec_rgy, pParams->codec);

    cmd << gen_cmd(&pParams->input, &encPrmDefault.input, &pParams->inprm, &encPrmDefault.inprm, save_disabled_prm);

    OPT_LST(_T("--output-depth"), outputDepth, list_output_depth);
    OPT_LST(_T("--output-csp"), outputCsp, list_output_csp);

    OPT_LST(_T("--quality"), nTargetUsage, list_quality_for_option);
    OPT_LST(_T("--function-mode"), functionMode, list_qsv_function_mode);
    OPT_LST(_T("--hyper-mode"), hyperMode, list_hyper_mode);
    OPT_NUM(_T("--async-depth"), nAsyncDepth);
    if (save_disabled_prm || ((pParams->memType) != (encPrmDefault.memType))) {
        switch (pParams->memType) {
#if D3D_SURFACES_SUPPORT
        case SYSTEM_MEMORY: cmd << _T(" --disable-d3d"); break;
        case HW_MEMORY:   cmd << _T(" --d3d"); break;
        case D3D9_MEMORY: cmd << _T(" --d3d9"); break;
#if MFX_D3D11_SUPPORT
        case D3D11_MEMORY: cmd << _T(" --d3d11"); break;
#endif
#endif
#if LIBVA_SUPPORT
        case SYSTEM_MEMORY: cmd << _T(" --disable-va"); break;
        case D3D11_MEMORY: cmd << _T(" --va"); break;
#endif
        default: break;
        }
    }
    if (save_disabled_prm || pParams->rcParam.encMode == MFX_RATECONTROL_QVBR) {
        OPT_NUM(_T("--qvbr-quality"), rcParam.qvbrQuality);
    }
    if (save_disabled_prm) {
        switch (pParams->rcParam.encMode) {
        case MFX_RATECONTROL_CBR:
        case MFX_RATECONTROL_VBR:
        case MFX_RATECONTROL_AVBR:
        case MFX_RATECONTROL_QVBR:
        case MFX_RATECONTROL_LA:
        case MFX_RATECONTROL_LA_HRD:
        case MFX_RATECONTROL_VCM: {
            OPT_QP(_T("--cqp"), rcParam.qp, true);
            cmd << _T(" --icq ") << pParams->rcParam.icqQuality;
        } break;
        case MFX_RATECONTROL_ICQ:
        case MFX_RATECONTROL_LA_ICQ: {
            OPT_QP(_T("--cqp"), rcParam.qp, true);
            cmd << _T(" --vbr ") << pParams->rcParam.bitrate;
        } break;
        case MFX_RATECONTROL_CQP:
        default: {
            cmd << _T(" --icq ") << pParams->rcParam.icqQuality;
            cmd << _T(" --vbr ") << pParams->rcParam.bitrate;
        } break;
        }
    }
    switch (pParams->rcParam.encMode) {
    case MFX_RATECONTROL_CBR: {
        cmd << _T(" --cbr ") << pParams->rcParam.bitrate;
    } break;
    case MFX_RATECONTROL_VBR: {
        cmd << _T(" --vbr ") << pParams->rcParam.bitrate;
    } break;
    case MFX_RATECONTROL_AVBR: {
        cmd << _T(" --avbr ") << pParams->rcParam.bitrate;
    } break;
    case MFX_RATECONTROL_QVBR: {
        cmd << _T(" --qvbr ") << pParams->rcParam.bitrate;
    } break;
    case MFX_RATECONTROL_LA: {
        cmd << _T(" --la ") << pParams->rcParam.bitrate;
    } break;
    case MFX_RATECONTROL_LA_HRD: {
        cmd << _T(" --la-hrd ") << pParams->rcParam.bitrate;
    } break;
    case MFX_RATECONTROL_VCM: {
        cmd << _T(" --vcm ") << pParams->rcParam.bitrate;
    } break;
    case MFX_RATECONTROL_ICQ: {
        cmd << _T(" --icq ") << pParams->rcParam.icqQuality;
    } break;
    case MFX_RATECONTROL_LA_ICQ: {
        cmd << _T(" --la-icq ") << pParams->rcParam.icqQuality;
    } break;
    case MFX_RATECONTROL_CQP:
    default: {
        OPT_QP(_T("--cqp"), rcParam.qp, true);
    } break;
    }
    if (save_disabled_prm || pParams->rcParam.encMode == MFX_RATECONTROL_AVBR) {
        OPT_NUM(_T("--avbr-unitsize"), rcParam.avbrConvergence);
    }
    if (save_disabled_prm
        || pParams->rcParam.encMode == MFX_RATECONTROL_LA
        || pParams->rcParam.encMode == MFX_RATECONTROL_LA_HRD
        || pParams->rcParam.encMode == MFX_RATECONTROL_LA_ICQ) {
        OPT_NUM(_T("--la-depth"), nLookaheadDepth);
        OPT_NUM(_T("--la-window-size"), nWinBRCSize);
        OPT_LST(_T("--la-quality"), nLookaheadDS, list_lookahead_ds);
    }
    if (save_disabled_prm || pParams->rcParam.encMode != MFX_RATECONTROL_CQP) {
        OPT_NUM(_T("--max-bitrate"), rcParam.maxBitrate);
    }
    OPT_NUM(_T("--vbv-bufsize"), rcParam.vbvBufSize);
    OPT_BOOL(_T("--fallback-rc"), _T("--no-fallback-rc"), fallbackRC);
    OPT_QP(_T("--qp-min"), qpMin, save_disabled_prm);
    OPT_QP(_T("--qp-max"), qpMax, save_disabled_prm);
    if (memcmp(pParams->pQPOffset, encPrmDefault.pQPOffset, sizeof(encPrmDefault.pQPOffset))) {
        tmp.str(tstring());
        bool exit_loop = false;
        for (int i = 0; i < _countof(pParams->pQPOffset) && !exit_loop; i++) {
            tmp << _T(":") << pParams->pQPOffset[i];
            exit_loop = true;
            for (int j = i+1; j < _countof(pParams->pQPOffset); j++) {
                if (pParams->pQPOffset[i] != pParams->pQPOffset[j]) {
                    exit_loop = false;
                    break;
                }
            }
        }
        cmd << _T(" --qp-offset ") << tmp.str().substr(1);
    }

    OPT_NUM(_T("--slices"), nSlices);
    OPT_NUM(_T("--ref"), nRef);
    if (gopRefDistAsBframe(pParams->codec)) {
        OPT_NUM(_T("-b"), GopRefDist - 1);
    } else {
        OPT_NUM(_T("--gop-ref-dist"), GopRefDist);
    }
    OPT_BOOL_OPT(_T("--b-pyramid"), _T("--no-b-pyramid"), bBPyramid);
    OPT_BOOL(_T("--open-gop"), _T("--no-open-gop"), bopenGOP);
    OPT_BOOL(_T("--strict-gop"), _T(""), bforceGOPSettings);
    OPT_BOOL_OPT(_T("--i-adapt"), _T("--no-i-adapt"), bAdaptiveI);
    OPT_BOOL_OPT(_T("--b-adapt"), _T("--no-b-adapt"), bAdaptiveB);
    OPT_TRI(_T("--weightb"), _T("--no-weightb"), nWeightB, MFX_WEIGHTED_PRED_DEFAULT, MFX_WEIGHTED_PRED_UNKNOWN);
    OPT_TRI(_T("--weightp"), _T("--no-weightp"), nWeightP, MFX_WEIGHTED_PRED_DEFAULT, MFX_WEIGHTED_PRED_UNKNOWN);
    OPT_BOOL_OPT(_T("--repartition-check"), _T("--no-repartition-check"), nRepartitionCheck);
    OPT_BOOL_OPT(_T("--fade-detect"), _T("--no-fade-detect"), nFadeDetect);
    if (pParams->nGOPLength == 0 && pParams->nGOPLength != encPrmDefault.nGOPLength) {
        cmd << _T(" --gop-len auto");
    } else {
        OPT_NUM(_T("--gop-len"), nGOPLength);
    }
    OPT_LST(_T("--mv-precision"), nMVPrecision, list_mv_presicion);
    OPT_NUM(_T("--mv-search"), MVSearchWindow.first);
    if (pParams->bGlobalMotionAdjust) {
        cmd << _T(" --mv-scaling ") << pParams->nMVCostScaling;
    }
    if (pParams->nPAR[0] > 0 && pParams->nPAR[1] > 0) {
        cmd << _T(" --sar ") << pParams->nPAR[0] << _T(":") << pParams->nPAR[1];
    } else if (pParams->nPAR[0] < 0 && pParams->nPAR[1] < 0) {
        cmd << _T(" --dar ") << -1 * pParams->nPAR[0] << _T(":") << -1 * pParams->nPAR[1];
    }
    if (pParams->tuneQuality != 0) {
        cmd << _T(" --tune ") << get_str_of_tune_bitmask(pParams->tuneQuality);
    }

    OPT_LST(_T("--scenario-info"), scenarioInfo, list_scenario_info);
    OPT_BOOL_OPT(_T("--extbrc"), _T("--no-extbrc"), extBRC);
    OPT_BOOL_OPT(_T("--mbbrc"), _T("--no-mbbrc"), bMBBRC);
    OPT_BOOL_OPT(_T("--adapt-ref"), _T("--no-adapt-ref"), adaptiveRef);
    OPT_BOOL_OPT(_T("--adapt-ltr"), _T("--no-adapt-ltr"), adaptiveLTR);
    OPT_BOOL_OPT(_T("--adapt-cqm"), _T("--no-adapt-cqm"), adaptiveCQM);
    OPT_NUM(_T("--intra-refresh-cycle"), intraRefreshCycle);
    OPT_NUM(_T("--max-framesize"), maxFrameSize);
    OPT_NUM(_T("--max-framesize-i"), maxFrameSizeI);
    OPT_NUM(_T("--max-framesize-p"), maxFrameSizeP);
    OPT_BOOL_OPT(_T("--direct-bias-adjust"), _T("--no-direct-bias-adjust"), bDirectBiasAdjust);
    OPT_LST(_T("--intra-pred"), nIntraPred, list_pred_block_size);
    OPT_LST(_T("--inter-pred"), nInterPred, list_pred_block_size);
    OPT_BOOL(_T("--aud"), _T(""), bOutputAud);
    OPT_BOOL(_T("--pic-struct"), _T(""), bOutputPicStruct);
    OPT_BOOL(_T("--buf-period"), _T(""), bufPeriodSEI);
    OPT_BOOL_OPT(_T("--repeat-headers"), _T("--no-repeat-headers"), repeatHeaders);
    OPT_LST(_T("--level"), CodecLevel, get_level_list(pParams->codec));
    OPT_LST(_T("--profile"), CodecProfile, get_profile_list(pParams->codec));
    if (save_disabled_prm || pParams->codec == RGY_CODEC_HEVC) {
        OPT_LST(_T("--ctu"), hevc_ctu, list_hevc_ctu);
        OPT_LST(_T("--sao"), hevc_sao, list_hevc_sao);
        OPT_BOOL(_T("--tskip"), _T("--no-tskip"), hevc_tskip);
        OPT_BOOL_OPT(_T("--hevc-gpb"), _T("--no-hevc-gpb"), hevc_gpb);
    }
    if (save_disabled_prm || pParams->codec == RGY_CODEC_AV1) {
        OPT_NUM(_T("--tile-row"), av1.tile_row);
        OPT_NUM(_T("--tile-col"), av1.tile_col);
    }
    if (save_disabled_prm || pParams->codec == RGY_CODEC_H264) {
        OPT_LST(_T("--trellis"), nTrellis, list_avc_trellis_for_options);
        switch (pParams->nBluray) {
        case 1: cmd << _T(" --bluray"); break;
        case 2: cmd << _T(" --force-bluray"); break;
        case 0:
        default: break;
        }
        OPT_BOOL(_T("--rdo"), _T(""), bRDO);
        OPT_BOOL(_T("--cavlc"), _T(""), bCAVLC);
        OPT_BOOL(_T("--no-deblock"), _T(""), bNoDeblock);
    }
    if (save_disabled_prm || pParams->codec == RGY_CODEC_VP8) {
        OPT_NUM(_T("--sharpness"), nVP8Sharpness);
    }
#if ENABLE_SESSION_THREAD_CONFIG
    OPT_NUM(_T("--session-threads"), nSessionThreads);
    OPT_LST(_T("--session-thread-priority"), nSessionThreadPriority, list_priority);
#endif //#if ENABLE_SESSION_THREAD_CONFIG

    cmd << gen_cmd(&pParams->common, &encPrmDefault.common, save_disabled_prm);
    cmd << gen_cmd(&pParams->vppmfx, &encPrmDefault.vppmfx, save_disabled_prm);
    cmd << gen_cmd(&pParams->vpp, &encPrmDefault.vpp, save_disabled_prm);

#if defined(_WIN32) || defined(_WIN64)
    OPT_NUM(_T("--mfx-thread"), nSessionThreads);
#endif //#if defined(_WIN32) || defined(_WIN64)
    OPT_BOOL(_T("--gpu-copy"), _T(""), gpuCopy);
    OPT_NUM(_T("--input-buf"), nInputBufSize);

    cmd << gen_cmd(&pParams->ctrl, &encPrmDefault.ctrl, save_disabled_prm);

    OPT_BOOL(_T("--timer-period-tuning"), _T("--no-timer-period-tuning"), bDisableTimerPeriodTuning);
    return cmd.str();
}
#pragma warning (pop)

