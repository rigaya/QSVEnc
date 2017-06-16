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
// --------------------------------------------------------------------------------------------

#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <set>
#include <vector>
#include <numeric>
#include <algorithm>
#include <ctime>
#include "rgy_osdep.h"
#if defined(_WIN32) || defined(_WIN64)
#include <shellapi.h>
#endif

#include "qsv_pipeline.h"
#include "qsv_prm.h"
#include "qsv_query.h"
#include "rgy_version.h"
#include "rgy_avutil.h"

#if ENABLE_CPP_REGEX
#include <regex>
#endif //#if ENABLE_CPP_REGEX
#if ENABLE_DTL
#include <dtl/dtl.hpp>
#endif //#if ENABLE_DTL

#if ENABLE_AVSW_READER
extern "C" {
#include <libavutil/channel_layout.h>
}
#endif

#if defined(_WIN32) || defined(_WIN64)
static bool check_locale_is_ja() {
    const WORD LangID_ja_JP = MAKELANGID(LANG_JAPANESE, SUBLANG_JAPANESE_JAPAN);
    return GetUserDefaultLangID() == LangID_ja_JP;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

static tstring GetQSVEncVersion() {
    static const TCHAR *const ENABLED_INFO[] = { _T("disabled"), _T("enabled") };
    tstring version;
    version += get_encoder_version();
    version += _T("\n");
    version += _T(" reader: raw");
    if (ENABLE_AVI_READER)         version += _T(", avi");
    if (ENABLE_AVISYNTH_READER)    version += _T(", avs");
    if (ENABLE_VAPOURSYNTH_READER) version += _T(", vpy");
    if (ENABLE_AVSW_READER) version += strsprintf(_T(", avqsv [%s]"), getHWDecSupportedCodecList().c_str());
#if !(defined(_WIN32) || defined(_WIN64))
    version += _T("\n vpp:    resize, deinterlace, denoise, detail-enhance, image-stab");
    if (ENABLE_CUSTOM_VPP) version += _T(", delego");
    if (ENABLE_LIBASS_SUBBURN != 0 && ENABLE_AVSW_READER != 0) version += _T(", sub");
#endif
    version += _T("\n");
    return version;
}

static void PrintVersion() {
    _ftprintf(stdout, _T("%s"), GetQSVEncVersion().c_str());
}

//適当に改行しながら表示する
static tstring PrintListOptions(const TCHAR *option_name, const CX_DESC *list, int default_index) {
    const TCHAR *indent_space = _T("                                ");
    const int indent_len = (int)_tcslen(indent_space);
    const int max_len = 77;
    tstring str = strsprintf(_T("   %s "), option_name);
    while ((int)str.length() < indent_len)
        str += _T(" ");
    int line_len = (int)str.length();
    for (int i = 0; list[i].desc; i++) {
        if (line_len + _tcslen(list[i].desc) + _tcslen(_T(", ")) >= max_len) {
            str += strsprintf(_T("\n%s"), indent_space);
            line_len = indent_len;
        } else {
            if (i) {
                str += strsprintf(_T(", "));
                line_len += 2;
            }
        }
        str += strsprintf(_T("%s"), list[i].desc);
        line_len += (int)_tcslen(list[i].desc);
    }
    str += strsprintf(_T("\n%s default: %s\n"), indent_space, list[default_index].desc);
    return str;
}

class CombinationGenerator {
public:
    CombinationGenerator(int i) : m_nCombination(i) {

    }
    void create(vector<int> used) {
        if ((int)used.size() == m_nCombination) {
            m_nCombinationList.push_back(used);
        }
        for (int i = 0; i < m_nCombination; i++) {
            if (std::find(used.begin(), used.end(), i) == used.end()) {
                vector<int> u = used;
                u.push_back(i);
                create(u);
            }
        }
    }
    vector<vector<int>> generate() {
        vector<int> used;
        create(used);
        return m_nCombinationList;
    };
    int m_nCombination;
    vector<vector<int>> m_nCombinationList;
};

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

static const TCHAR *short_opt_to_long(TCHAR short_opt) {
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
    case _T('u'):
        option_name = _T("quality");
        break;
    case _T('f'):
        option_name = _T("output-format");
        break;
    case _T('i'):
        option_name = _T("input-file");
        break;
    case _T('o'):
        option_name = _T("output-file");
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

static tstring help(const TCHAR *strAppName = nullptr) {
    tstring str;
    str += strsprintf(_T("Usage: %s [Options] -i <filename> -o <filename>\n"), (strAppName) ? PathFindFileName(strAppName) : _T("QSVEncC"));
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
#if ENABLE_AVSW_READER
        _T("   --check-avversion            show dll version\n")
        _T("   --check-codecs               show codecs available\n")
        _T("   --check-encoders             show audio encoders available\n")
        _T("   --check-decoders             show audio decoders available\n")
        _T("   --check-formats              show in/out formats available\n")
        _T("   --check-protocols            show in/out protocols available\n")
        _T("   --check-filters              show filters available\n")
#endif
        _T("\n"));

    str += strsprintf(_T("\n")
        _T("Basic Encoding Options: \n")
        _T("-c,--codec <string>             set encode codec\n")
        _T("                                 - h264(default), hevc, mpeg2\n")
        _T("-i,--input-file <filename>      set input file name\n")
        _T("-o,--output-file <filename>     set ouput file name\n")
#if ENABLE_AVSW_READER
        _T("                                 for extension mp4/mkv/mov,\n")
        _T("                                 avcodec muxer will be used.\n")
#endif
        _T("\n")
        _T(" Input formats (will be estimated from extension if not set.)\n")
        _T("   --raw                        set input as raw format\n")
        _T("   --y4m                        set input as y4m format\n")
#if ENABLE_AVI_READER
        _T("   --avi                        set input as avi format\n")
#endif
#if ENABLE_AVISYNTH_READER
        _T("   --avs                        set input as avs format\n")
#endif
#if ENABLE_VAPOURSYNTH_READER
        _T("   --vpy                        set input as vpy format\n")
        _T("   --vpy-mt                     set input as vpy format in multi-thread\n")
#endif
#if ENABLE_AVSW_READER
        _T("   --avqsv                      set input to use avcodec + qsv\n")
        _T("   --avsw                       set input to use avcodec + sw deocder\n")
        _T("   --input-analyze <int>        set time (sec) which reader analyze input file.\n")
        _T("                                 default: 5 (seconds).\n")
        _T("                                 could be only used with avqsv/avsw reader.\n")
        _T("                                 use if reader fails to detect audio stream.\n")
        _T("   --video-track <int>          set video track to encode in track id\n")
        _T("                                 1 (default)  highest resolution video track\n")
        _T("                                 2            next high resolution video track\n")
        _T("                                   ... \n")
        _T("                                 -1           lowest resolution video track\n")
        _T("                                 -2           next low resolution video track\n")
        _T("                                   ... \n")
        _T("   --video-streamid <int>       set video track to encode in stream id\n")
        _T("   --audio-source <string>      input extra audio file\n")
        _T("   --audio-file [<int>?][<string>:]<string>\n")
        _T("                                extract audio into file.\n")
        _T("                                 could be only used with avqsv reader.\n")
        _T("                                 below are optional,\n")
        _T("                                  in [<int>?], specify track number to extract.\n")
        _T("                                  in [<string>?], specify output format.\n")
        _T("   --trim <int>:<int>[,<int>:<int>]...\n")
        _T("                                trim video for the frame range specified.\n")
        _T("                                 frame range should not overwrap each other.\n")
        _T("   --seek [<int>:][<int>:]<int>[.<int>] (hh:mm:ss.ms)\n")
        _T("                                skip video for the time specified,\n")
        _T("                                 seek will be inaccurate but fast.\n")
        _T("   --input-format <string>      set input format of input file.\n")
        _T("                                 this requires use of avqsv/avsw reader.\n")
        _T("-f,--output-format <string>     set output format of output file.\n")
        _T("                                 if format is not specified, output format will\n")
        _T("                                 be guessed from output file extension.\n")
        _T("                                 set \"raw\" for H.264/ES output.\n")
        _T("   --audio-copy [<int>[,...]]   mux audio with video during output.\n")
        _T("                                 could be only used with\n")
        _T("                                 avqsv reader and avcodec muxer.\n")
        _T("                                 by default copies all audio tracks.\n")
        _T("                                 \"--audio-copy 1,2\" will extract\n")
        _T("                                 audio track #1 and #2.\n")
        _T("   --audio-codec [<int>?]<string>\n")
        _T("                                encode audio to specified format.\n")
        _T("                                  in [<int>?], specify track number to encode.\n")
        _T("   --audio-bitrate [<int>?]<int>\n")
        _T("                                set encode bitrate for audio (kbps).\n")
        _T("                                  in [<int>?], specify track number of audio.\n")
        _T("   --audio-ignore-decode-error <int>  (default: %d)\n")
        _T("                                set numbers of continuous packets of audio\n")
        _T("                                 decode error to ignore, replaced by silence.\n")
        _T("   --audio-ignore-notrack-error ignore error when audio track is unfound.\n")
        _T("   --audio-samplerate [<int>?]<int>\n")
        _T("                                set sampling rate for audio (Hz).\n")
        _T("                                  in [<int>?], specify track number of audio.\n")
        _T("   --audio-resampler <string>   set audio resampler.\n")
        _T("                                  swr (swresampler: default), soxr (libsoxr)\n")
        _T("   --audio-stream [<int>?][<string1>][:<string2>][,[<string1>][:<string2>]][..\n")
        _T("       set audio streams in channels.\n")
        _T("         in [<int>?], specify track number to split.\n")
        _T("         in <string1>, set input channels to use from source stream.\n")
        _T("           if unset, all input channels will be used.\n")
        _T("         in <string2>, set output channels to mix.\n")
        _T("           if unset, all input channels will be copied without mixing.\n")
        _T("       example1: --audio-stream FL,FR\n")
        _T("         splitting dual mono audio to each stream.\n")
        _T("       example2: --audio-stream :stereo\n")
        _T("         mixing input channels to stereo.\n")
        _T("       example3: --audio-stream 5.1,5.1:stereo\n")
        _T("         keeping 5.1ch audio and also adding downmixed stereo stream.\n")
        _T("       usable simbols\n")
        _T("         mono       = FC\n")
        _T("         stereo     = FL + FR\n")
        _T("         2.1        = FL + FR + LFE\n")
        _T("         3.0        = FL + FR + FC\n")
        _T("         3.0(back)  = FL + FR + BC\n")
        _T("         3.1        = FL + FR + FC + LFE\n")
        _T("         4.0        = FL + FR + FC + BC\n")
        _T("         quad       = FL + FR + BL + BR\n")
        _T("         quad(side) = FL + FR + SL + SR\n")
        _T("         5.0        = FL + FR + FC + SL + SR\n")
        _T("         5.1        = FL + FR + FC + LFE + SL + SR\n")
        _T("         6.0        = FL + FR + FC + BC + SL + SR\n")
        _T("         6.0(front) = FL + FR + FLC + FRC + SL + SR\n")
        _T("         hexagonal  = FL + FR + FC + BL + BR + BC\n")
        _T("         6.1        = FL + FR + FC + LFE + BC + SL + SR\n")
        _T("         6.1(front) = FL + FR + LFE + FLC + FRC + SL + SR\n")
        _T("         7.0        = FL + FR + FC + BL + BR + SL + SR\n")
        _T("         7.0(front) = FL + FR + FC + FLC + FRC + SL + SR\n")
        _T("         7.1        = FL + FR + FC + LFE + BL + BR + SL + SR\n")
        _T("         7.1(wide)  = FL + FR + FC + LFE + FLC + FRC + SL + SR\n")
        _T("   --audio-filter [<int>?]<string>\n")
        _T("                                set audio filter.\n")
        _T("                                  in [<int>?], specify track number of audio.\n")
        _T("   --chapter-copy               copy chapter to output file.\n")
        _T("   --chapter <string>           set chapter from file specified.\n")
        _T("   --chapter-no-trim            do not applyapply trim to chapter file.\n")
        _T("   --sub-copy [<int>[,...]]     copy subtitle to output file.\n")
        _T("                                 these could be only used with\n")
        _T("                                 avqsv reader and avcodec muxer.\n")
        _T("                                 below are optional,\n")
        _T("                                  in [<int>?], specify track number to copy.\n")
        _T("\n")
        _T("   --avsync <string>            method for AV sync (default: through)\n")
        _T("                                 through  ... assume cfr, no check but fast\n")
        _T("                                 forcecfr ... check timestamp and force cfr.\n")
        _T("-m,--mux-option <string1>:<string2>\n")
        _T("                                set muxer option name and value.\n")
        _T("                                 these could be only used with\n")
        _T("                                 avqsv reader and avcodec muxer.\n"),
        QSV_DEFAULT_AUDIO_IGNORE_DECODE_ERROR);
#endif
    str += strsprintf(_T("\n")
        _T("   --nv12                       set raw input as NV12 color format,\n")
        _T("                                if not specified YV12 is expected\n")
        _T("   --tff                        set as interlaced, top field first\n")
        _T("   --bff                        set as interlaced, bottom field first\n")
        _T("   --fps <int>/<int> or <float> video frame rate (frames per second)\n")
        _T("\n")
        _T("   --input-res <int>x<int>      input resolution\n")
        _T("   --output-res <int>x<int>     output resolution\n")
        _T("                                if different from input, uses vpp resizing\n")
        _T("                                if not set, output resolution will be same\n")
        _T("                                as input (no resize will be done).\n")
        _T("   --fixed-func                 use fixed func instead of GPU EU (default:off)\n")
        _T("\n"));
    str += strsprintf(_T("Frame buffer Options:\n")
        _T(" frame buffers are selected automatically by default.\n")
#ifdef D3D_SURFACES_SUPPORT
        _T(" d3d9 memory is faster than d3d11, so d3d9 frames are used whenever possible,\n")
        _T(" except decode/vpp only mode (= no encoding mode, system frames are used).\n")
        _T(" On particular cases, sush as runnning on a system with dGPU, or running\n")
        _T(" vpp-rotate, will require the uses of d3d11 surface.\n")
        _T(" Options below will change this default behavior.\n")
        _T("\n")
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
#ifdef LIBVA_SUPPORT
            _T("   --disable-va                 disable using va surfaces\n")
            _T("   --va                         use va surfaces\n")
#endif //#ifdef LIBVA_SUPPORT
            _T("\n"));
    str += strsprintf(_T("Encode Mode Options:\n")
        _T(" EncMode default: --cqp\n")
        _T("   --cqp <int> or               encode in Constant QP, default %d:%d:%d\n")
        _T("         <int>:<int>:<int>      set qp value for i:p:b frame\n")
        _T("   --vqp <int> or               encode in Variable QP, default %d:%d:%d\n")
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
        _T("   --qvbr-q <int>  or           set quality used in qvbr mode. default: %d\n")
        _T("   --qvbr-quality <int>          QVBR mode is only supported with API v1.11\n")
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
        _T("-a,--async-depth                set async depth for QSV pipeline. (0-%d)\n")
        _T("                                 default: 0 (=auto, 4+2*(extra pipeline step))\n")
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
        _T("   --(no-)mbbrc                 enables per macro block rate control\n")
        _T("                                 default: auto\n")
        _T("   --(no-)extbrc                enables extended rate control\n")
        _T("                                 default: auto\n")
        _T("   --ref <int>                  reference frames\n")
        _T("                                  default %d (auto)\n")
        _T("-b,--bframes <int>              number of sequential b frames\n")
        _T("                                  default %d(HEVC) / %d(others)\n")
        _T("   --(no-)b-pyramid             enables B-frame pyramid reference (default:off)\n")
        _T("   --(no-)direct-bias-adjust    lower usage of B frame Direct/Skip type\n")
        _T("   --gop-len <int>              (max) gop length, default %d (auto)\n")
        _T("                                  when auto, fps x 10 will be set.\n")
        _T("   --(no-)scenechange           enables scene change detection\n")
        _T("   --(no-)open-gop              enables open gop (default:off)\n")
        _T("   --strict-gop                 force gop structure\n")
        _T("   --(no-)i-adapt               enables adaptive I frame insert (default:off)\n")
        _T("   --(no-)b-adapt               enables adaptive B frame insert (default:off)\n")
        _T("   --(no-)weightp               enable weight prediction for P frame\n")
        _T("   --(no-)weightb               enable weight prediction for B frame\n")
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
        _T("   --no-deblock                 disables H.264 deblock feature\n")
        _T("   --sharpness <int>            [vp8] set sharpness level for vp8 enc\n")
        _T("\n"),
        QSV_ASYNC_DEPTH_MAX,
        QSV_LOOKAHEAD_DEPTH_MIN, QSV_LOOKAHEAD_DEPTH_MAX,
        QSV_DEFAULT_REF,
        QSV_DEFAULT_HEVC_BFRAMES, QSV_DEFAULT_H264_BFRAMES,
        QSV_DEFAULT_GOP_LEN);
    str += PrintMultipleListOptions(_T("--level <string>"), _T("set codec level"),
        { { _T("H.264"), list_avc_level,   0 },
          { _T("HEVC"),  list_hevc_level,  0 },
          { _T("MPEG2"), list_mpeg2_level, 0 }
        });
    str += PrintMultipleListOptions(_T("--profile <string>"), _T("set codec profile"),
        { { _T("H.264"), list_avc_profile,   0 },
          { _T("HEVC"),  list_hevc_profile,  0 },
          { _T("MPEG2"), list_mpeg2_profile, 0 }
        });

    str += strsprintf(_T("\n")
        _T("   --sar <int>:<int>            set Sample Aspect Ratio\n")
        _T("   --dar <int>:<int>            set Display Aspect Ratio\n")
        _T("   --bluray                     for H.264 bluray encoding\n")
        _T("\n")
        _T("   --crop <int>,<int>,<int>,<int>\n")
        _T("                                set crop pixels of left, up, right, bottom.\n")
        _T("\n")
        _T("   --fullrange                  set stream as fullrange yuv\n"));
    str += PrintListOptions(_T("--videoformat <string>"), list_videoformat, 0);
    str += PrintListOptions(_T("--colormatrix <string>"), list_colormatrix, 0);
    str += PrintListOptions(_T("--colorprim <string>"),   list_colorprim,   0);
    str += PrintListOptions(_T("--transfer <string>"),    list_transfer,    0);
    str += strsprintf(_T("")
        _T("   --aud                        insert aud nal unit to ouput stream.\n")
        _T("   --pic-struct                 insert pic-timing SEI with pic_struct.\n"));
    str += strsprintf(_T("\n")
        //_T("   --sw                         use software encoding, instead of QSV (hw)\n")
        _T("   --input-buf <int>            buffer size for input in frames (%d-%d)\n")
        _T("                                 default   hw: %d,  sw: %d\n")
        _T("                                 cannot be used with avqsv reader.\n"),
        QSV_INPUT_BUF_MIN, QSV_INPUT_BUF_MAX,
        QSV_DEFAULT_INPUT_BUF_HW, QSV_DEFAULT_INPUT_BUF_SW
        );
    str += strsprintf(_T("")
        _T("   --output-buf <int>           buffer size for output in MByte\n")
        _T("                                 default %d MB (0-%d)\n"),
        QSV_DEFAULT_OUTPUT_BUF_MB, RGY_OUTPUT_BUF_MB_MAX
        );
    str += strsprintf(_T("")
#if defined(_WIN32) || defined(_WIN64)
        _T("   --mfx-thread <int>          set input thread num (-1 (auto), 2, 3, ...)\n")
        _T("                                 note that mfx thread cannot be less than 2.\n")
#endif
        _T("   --input-thread <int>        set input thread num\n")
        _T("                                  0: disable (slow, but less cpu usage)\n")
        _T("                                  1: use one thread\n")
#if ENABLE_AVCODEC_OUT_THREAD
        _T("   --output-thread <int>        set output thread num\n")
        _T("                                 -1: auto (= default)\n")
        _T("                                  0: disable (slow, but less memory usage)\n")
        _T("                                  1: use one thread\n")
#if 0
        _T("   --audio-thread <int>         set audio thread num, available only with output thread\n")
        _T("                                 -1: auto (= default)\n")
        _T("                                  0: disable (slow, but less memory usage)\n")
        _T("                                  1: use one thread\n")
        _T("                                  2: use two thread\n")
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
#endif //#if ENABLE_AVCODEC_OUT_THREAD
        _T("   --min-memory                 minimize memory usage of QSVEncC.\n")
        _T("                                 same as --output-thread 0 --audio-thread 0\n")
        _T("                                   --mfx-thread -a 1 --input-buf 1 --output-buf 0\n")
        _T("                                 this will cause lower performance!\n")
        _T("   --max-procfps <int>         limit processing speed to lower resource usage.\n")
        _T("                                 default:0 (no limit)\n")
        );
    str += strsprintf(
        _T("   --log <string>               output log to file (txt or html).\n")
        _T("   --log-level <string>         set output log level\n")
        _T("                                 info(default), warn, error, debug\n")
        _T("   --log-framelist <string>     output frame info for avqsv reader (for debug)\n")
#if _DEBUG
        _T("   --log-mus-ts <string>         (for debug)\n")
        _T("   --log-copy-framedata <string> (for debug)\n")
#endif
        );
#if ENABLE_SESSION_THREAD_CONFIG
    str += strsprintf(_T("")
        _T("   --session-threads            set num of threads for QSV session. (0-%d)\n")
        _T("                                 default: 0 (=auto)\n")
        _T("   --session-thread-priority    set thread priority for QSV session.\n")
        _T("                                  - low, normal(default), high\n"),
        QSV_SESSION_THREAD_MAX);
#endif
    str += strsprintf(_T("\n")
        _T("   --benchmark <string>         run in benchmark mode\n")
        _T("                                 and write result in txt file\n")
        _T("   --bench-quality \"all\" or <int>[,<int>][,<int>]...\n")
        _T("                                 default: 1,4,7\n")
        _T("                                list of target quality to check on benchmark\n")
        _T("   --perf-monitor [<string>][,<string>]...\n")
        _T("       check performance info of QSVEncC and output to log file\n")
        _T("       select counter from below, default = all\n")
        _T("   --perf-monitor-plot [<string>][,<string>]...\n")
        _T("       plot perf monitor realtime (required python, pyqtgraph)\n")
        _T("       select counter from below, default = cpu,bitrate\n")
        _T("                                 \n")
        _T("     counters for perf-monitor, perf-monitor-plot\n")
        _T("                                 all          ... monitor all info\n")
        _T("                                 cpu_total    ... cpu total usage (%%)\n")
        _T("                                 cpu_kernel   ... cpu kernel usage (%%)\n")
#if defined(_WIN32) || defined(_WIN64)
        _T("                                 cpu_main     ... cpu main thread usage (%%)\n")
        _T("                                 cpu_enc      ... cpu encode thread usage (%%)\n")
        _T("                                 cpu_in       ... cpu input thread usage (%%)\n")
        _T("                                 cpu_out      ... cpu output thread usage (%%)\n")
        _T("                                 cpu_aud_proc ... cpu aud proc thread usage (%%)\n")
        _T("                                 cpu_aud_enc  ... cpu aud enc thread usage (%%)\n")
#endif //#if defined(_WIN32) || defined(_WIN64)
        _T("                                 cpu          ... monitor all cpu info\n")
#if defined(_WIN32) || defined(_WIN64)
        _T("                                 gpu_load    ... gpu usage (%%)\n")
        _T("                                 gpu_clock   ... gpu avg clock (%%)\n")
        _T("                                 gpu         ... monitor all gpu info\n")
#endif //#if defined(_WIN32) || defined(_WIN64)
        _T("                                 queue       ... queue usage\n")
        _T("                                 mem_private ... private memory (MB)\n")
        _T("                                 mem_virtual ... virtual memory (MB)\n")
        _T("                                 mem         ... monitor all memory info\n")
        _T("                                 io_read     ... io read  (MB/s)\n")
        _T("                                 io_write    ... io write (MB/s)\n")
        _T("                                 io          ... monitor all io info\n")
        _T("                                 fps         ... encode speed (fps)\n")
        _T("                                 fps_avg     ... encode avg. speed (fps)\n")
        _T("                                 bitrate     ... encode bitrate (kbps)\n")
        _T("                                 bitrate_avg ... encode avg. bitrate (kbps)\n")
        _T("                                 frame_out   ... written_frames\n")
        _T("                                 \n")
        _T("   --python <string>            set python path for --perf-monitor-plot\n")
        _T("                                 default: python\n")
        _T("   --perf-monitor-interval <int> set perf monitor check interval (millisec)\n")
        _T("                                 default 250, must be 50 or more\n")
#if defined(_WIN32) || defined(_WIN64)
        _T("   --(no-)timer-period-tuning   enable(disable) timer period tuning\n")
        _T("                                  default: enabled\n")
#endif //#if defined(_WIN32) || defined(_WIN64)
        );
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
    str += strsprintf(_T("\nVPP Options:\n")
        _T("   --vpp-denoise <int>          use vpp denoise, set strength (%d-%d)\n")
        _T("   --vpp-detail-enhance <int>   use vpp detail enahancer, set strength (%d-%d)\n")
        _T("   --vpp-deinterlace <string>   set vpp deinterlace mode\n")
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
        _T("                                 - none, upscale, box\n")
#if ENABLE_CUSTOM_VPP
#if ENABLE_AVSW_READER && ENABLE_LIBASS_SUBBURN
        _T("   --vpp-sub [<int>] or [<string>]\n")
        _T("                                burn in subtitle into frame\n")
        _T("                                set sub track number in input file by integer\n")
        _T("                                or set external sub file path by string.\n")
        _T("   --vpp-sub-charset [<string>] set subtitle char set\n")
        _T("   --vpp-sub-shaping <string>   simple(default), complex\n")
#endif //#if ENABLE_AVSW_READER && ENABLE_LIBASS_SUBBURN
        _T("   --vpp-delogo <string>        set delogo file path\n")
        _T("   --vpp-delogo-select <string> set target logo name or auto select file\n")
        _T("                                 or logo index starting from 1.\n")
        _T("   --vpp-delogo-pos <int>:<int> set delogo pos offset\n")
        _T("   --vpp-delogo-depth <int>     set delogo depth [default:%d]\n")
        _T("   --vpp-delogo-y  <int>        set delogo y  param\n")
        _T("   --vpp-delogo-cb <int>        set delogo cb param\n")
        _T("   --vpp-delogo-cr <int>        set delogo cr param\n")
        _T("   --vpp-delogo-add             add logo mode\n")
#endif //#if ENABLE_CUSTOM_VPP
        _T("   --vpp-rotate <int>           rotate image\n")
        _T("                                 90, 180, 270.\n")
        _T("   --vpp-mirror <string>        mirror image\n")
        _T("                                 - h   mirror in horizontal direction\n")
        _T("                                 - v   mirror in vertical   direction\n")
        _T("   --vpp-scaling <string>       set scaling quality\n")
        _T("                                 - auto(default)\n")
        _T("                                 - simple   use simple scaling\n")
        _T("                                 - fine     use high quality scaling\n")
        _T("   --vpp-half-turn              half turn video image\n")
        _T("                                 unoptimized and very slow.\n"),
        QSV_VPP_DENOISE_MIN, QSV_VPP_DENOISE_MAX,
        QSV_VPP_DETAIL_ENHANCE_MIN, QSV_VPP_DETAIL_ENHANCE_MAX,
        QSV_DEFAULT_VPP_DELOGO_DEPTH
        );
    return str;
}

#if ENABLE_CPP_REGEX
static vector<std::string> createOptionList() {
    vector<std::string> optionList;
    auto helpLines = split(tchar_to_string(help()), "\n");
    std::regex re1(R"(^\s{2,6}--([A-Za-z0-9][A-Za-z0-9-_]+)\s+.*)");
    std::regex re2(R"(^\s{0,3}-[A-Za-z0-9],--([A-Za-z0-9][A-Za-z0-9-_]+)\s+.*)");
    std::regex re3(R"(^\s{0,3}--\(no-\)([A-Za-z0-9][A-Za-z0-9-_]+)\s+.*)");
    for (const auto& line : helpLines) {
        std::smatch match;
        if (std::regex_match(line, match, re1) && match.size() == 2) {
            optionList.push_back(match[1]);
        } else if (std::regex_match(line, match, re2) && match.size() == 2) {
            optionList.push_back(match[1]);
        } else if (std::regex_match(line, match, re3) && match.size() == 2) {
            optionList.push_back(match[1]);
        }
    }
    return optionList;
}
#endif //#if ENABLE_CPP_REGEX

static void PrintHelp(const TCHAR *strAppName, const TCHAR *strErrorMessage, const TCHAR *strOptionName, const TCHAR *strErrorValue = nullptr) {
    if (strErrorMessage) {
        if (strOptionName) {
            if (strErrorValue) {
                _ftprintf(stderr, _T("Error: %s \"%s\" for \"--%s\"\n"), strErrorMessage, strErrorValue, strOptionName);
                if (0 == _tcsnccmp(strErrorValue, _T("--"), _tcslen(_T("--")))
                    || (strErrorValue[0] == _T('-') && strErrorValue[2] == _T('\0') && short_opt_to_long(strErrorValue[1]) != nullptr)) {
                    _ftprintf(stderr, _T("       \"--%s\" requires value.\n\n"), strOptionName);
                }
            } else {
                _ftprintf(stderr, _T("Error: %s for --%s\n\n"), strErrorMessage, strOptionName);
            }
        } else {
            _ftprintf(stderr, _T("Error: %s\n\n"), strErrorMessage);
#if (ENABLE_CPP_REGEX && ENABLE_DTL)
            if (strErrorValue) {
                //どのオプション名に近いか検証する
                auto optList = createOptionList();
                const auto invalid_opt = tchar_to_string(strErrorValue);
                //入力文字列を"-"で区切り、その組み合わせをすべて試す
                const auto invalid_opt_words = split(invalid_opt, "-", true);
                CombinationGenerator generator((int)invalid_opt_words.size());
                const auto combinationList = generator.generate();
                vector<std::pair<std::string, int>> editDistList;
                for (const auto& opt : optList) {
                    int nMinEditDist = INT_MAX;
                    for (const auto& combination : combinationList) {
                        std::string check_key;
                        for (auto i : combination) {
                            if (check_key.length() > 0) {
                                check_key += "-";
                            }
                            check_key += invalid_opt_words[i];
                        }
                        dtl::Diff<char, std::string> diff(check_key, opt);
                        diff.onOnlyEditDistance();
                        diff.compose();
                        nMinEditDist = (std::min)(nMinEditDist, (int)diff.getEditDistance());
                    }
                    editDistList.push_back(std::make_pair(opt, nMinEditDist));
                }
                std::sort(editDistList.begin(), editDistList.end(), [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
                    return b.second > a.second;
                });
                const int nMinEditDist = editDistList[0].second;
                _ftprintf(stderr, _T("Did you mean option(s) below?\n"));
                for (const auto& editDist : editDistList) {
                    if (editDist.second != nMinEditDist) {
                        break;
                    }
                    _ftprintf(stderr, _T("  --%s\n"), char_to_tstring(editDist.first).c_str());
                }
            }
#endif //#if ENABLE_DTL
        }
    } else {
        PrintVersion();
        _ftprintf(stdout, _T("%s\n"), help(strAppName).c_str());
    }
}

static int getAudioTrackIdx(const sInputParams* pParams, int iTrack) {
    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        if (iTrack == pParams->ppAudioSelectList[i]->nAudioSelect) {
            return i;
        }
    }
    return -1;
}

static int getFreeAudioTrack(const sInputParams* pParams) {
    for (int iTrack = 1;; iTrack++) {
        if (0 > getAudioTrackIdx(pParams, iTrack)) {
            return iTrack;
        }
    }
#ifndef _MSC_VER
    return -1;
#endif //_MSC_VER
}

static int writeFeatureList(tstring filename, FeatureListStrType type = FEATURE_LIST_STR_TYPE_UNKNOWN) {
    static const tstring header = _T(R"(
<!DOCTYPE html>
<html lang = "ja">
<head>
<meta charset = "UTF-8">
<title>QSVEncC Check Features</title>
<style type=text/css>
   body { 
        font-family: "Segoe UI","Meiryo UI", "ＭＳ ゴシック", sans-serif;
        background: #eeeeee;
        margin: 0px 20px 20px;
        padding: 0px;
        color: #223377;
   }
   div.page_top {
       background-color: #ffffff;
       border: 1px solid #aaaaaa;
       padding: 15px;
   }
   div.table_block {
       margin: 0px;
       padding: 0px;
   }
   p.table_block {
       margin: 0px;
       padding: 0px;
   }
   h1 {
       text-shadow: 0 0 6px #ccc;
       color: #CEF6F5;
       background: #0B173B;
       background: -moz-linear-gradient(top,  #0b173b 0%, #2167b2 50%, #155287 52%, #2d7d91 100%);
       background: -webkit-linear-gradient(top,  #0b173b 0%,#2167b2 50%,#155287 52%,#2d7d91 100%);
       background: -ms-linear-gradient(top,  #0b173b 0%,#2167b2 50%,#155287 52%,#2d7d91 100%);
       background: linear-gradient(to bottom,  #0b173b 0%,#2167b2 50%,#155287 52%,#2d7d91 100%);
       border-collapse: collapse;
       border: 0px;
       margin: 0px;
       padding: 10px;
       border-spacing: 0px;
       font-family: "Segoe UI", "Meiryo UI", "ＭＳ ゴシック", sans-serif;
   }
    a.table_block:link {
        color: #7991F1;
        font-size: small;
    }
    a.table_block:visited {
        color: #7991F1;
        font-size: small;
    }
    a.vpp_table_block:link {
        color: #223377;
    }
    a.cpp_table_block:visited {
        color: #223377;
    }
    hr {
        margin: 4px;
        padding: 0px;
    }
    li {
        padding: 5px 4px;
    }
    img {
        padding: 4px 0px;
    }
    table.simpleBlue {
        background-color: #ff9951;
        border-collapse: collapse;
        border: 0px;
        border-spacing: 0px;
        margin: 5px;
    }
    table.simpleBlue th,
    table.simpleBlue td {
        background-color: #ffffff;
        padding: 2px 10px;
        border: 1px solid #81BEF7;
        color: #223377;
        margin: 5px;
        text-align: center;
        font-size: small;
        font-family: "Segoe UI","Meiryo UI","ＭＳ ゴシック",sans-serif;
    }
    table.simpleOrange {
        background-color: #ff9951;
        border-collapse: collapse;
        border: 0px;
        border-spacing: 0px;
    }
    table.simpleOrange th,
    table.simpleOrange td {
        background-color: #ffffff;
        padding: 2px 10px;
        border: 1px solid #ff9951;
        color: #223377;
        margin: 10px;
        font-size: small;
        font-family: "Segoe UI","Meiryo UI","ＭＳ ゴシック",sans-serif;
    }
    table.simpleOrange td.ok {
        background-color: #A9F5A9;
        border: 1px solid #ff9951;
        tex-align: center;
        color: #223377;
        font-size: small;
    }
    table.simpleOrange td.fail {
        background-color: #F5A9A9;
        border: 1px solid #ff9951;
        tex-align: center;
        color: #223377;
        font-size: small;
    }
</style>
<script type="text/javascript">
<!--
function showTable(idno) {
    tc = ('TableClose' + (idno));
    to = ('TableOpen' + (idno));
    if( document.getElementById(tc).style.display == "none" ) {
        document.getElementById(tc).style.display = "block";
        document.getElementById(to).style.display = "none";
    } else {
        document.getElementById(tc).style.display = "none";
        document.getElementById(to).style.display = "block";
    }
}
//-->
</script>
</head>
<body>
)");

    uint32_t codepage = CP_THREAD_ACP;
    if (type == FEATURE_LIST_STR_TYPE_UNKNOWN) {
        if (check_ext(filename.c_str(), { ".html", ".htm" })) {
            type = FEATURE_LIST_STR_TYPE_HTML;
        } else if (check_ext(filename.c_str(), { ".csv" })) {
            type = FEATURE_LIST_STR_TYPE_CSV;
        } else {
            type = FEATURE_LIST_STR_TYPE_TXT;
        }
    }
    if (filename.length() == 0 && type == FEATURE_LIST_STR_TYPE_HTML) {
        filename = _T("qsv_check.html");
    }

    bool bUseJapanese = false;
    if (type == FEATURE_LIST_STR_TYPE_HTML) {
        codepage = CP_UTF8;
#if defined(_WIN32) || defined(_WIN64)
        bUseJapanese = check_locale_is_ja();
#endif
    }

    FILE *fp = stdout;
    if (filename.length()) {
        if (_tfopen_s(&fp, filename.c_str(), _T("w"))) {
            return 1;
        }
    }

    auto print_tstring = [&](tstring str, bool html_replace, tstring html_block_begin = _T(""), tstring html_block_end = _T("")) {
        if (type == FEATURE_LIST_STR_TYPE_TXT) {
            _ftprintf(fp, _T("%s"), str.c_str());
        } else if (type == FEATURE_LIST_STR_TYPE_CSV) {
            fprintf(fp, "%s", tchar_to_string(str, codepage).c_str());
        } else {
            if (html_replace) {
                str = str_replace(str, _T("<"),  _T("&lt;"));
                str = str_replace(str, _T(">"),  _T("&gt;"));
                str = str_replace(str, _T("\n"), _T("<br>\n"));
            }
            fprintf(fp, "%s%s%s", tchar_to_string(html_block_begin).c_str(), tchar_to_string(str, codepage).c_str(), tchar_to_string(html_block_end).c_str());
        }
    };

    fprintf(stderr, (bUseJapanese) ? "QSVの情報を取得しています...\n" : "Checking for QSV...\n");

    if (type == FEATURE_LIST_STR_TYPE_HTML) {
        print_tstring(header, false);
        print_tstring(_T("<h1>QSVEncC ") + tstring((bUseJapanese) ? _T("情報") : _T("Check Features")) + _T("</h1>\n<div class=page_top>\n"), false);
    }
    print_tstring(GetQSVEncVersion(), true, _T("<span style=font-size:small>"), _T("</span>"));

    if (type == FEATURE_LIST_STR_TYPE_HTML) {
        print_tstring(_T("<hr>\n"), false);
    }
    tstring environmentInfo = getEnviromentInfo(false);
    if (type == FEATURE_LIST_STR_TYPE_HTML) {
        environmentInfo = str_replace(environmentInfo, _T("Environment Info\n"), _T(""));
    }
    print_tstring(environmentInfo + ((type == FEATURE_LIST_STR_TYPE_HTML) ? _T("") : _T("\n")), true, _T("<span style=font-size:small>"), _T("</span>"));
    bool bOSSupportsQSV = true;
#if defined(_WIN32) || defined(_WIN64)
    OSVERSIONINFOEXW osver;
    tstring osversion = getOSVersion(&osver);
    bOSSupportsQSV &= osver.dwPlatformId == VER_PLATFORM_WIN32_NT;
    bOSSupportsQSV &= (osver.dwMajorVersion >= 7 || osver.dwMajorVersion == 6 && osver.dwMinorVersion >= 1);
    bOSSupportsQSV &= osver.wProductType == VER_NT_WORKSTATION;
#else
    tstring osversion = getOSVersion();
#endif

    mfxVersion test = { 0, 1 };
    for (int impl_type = 0; impl_type < 1; impl_type++) {
        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            print_tstring(_T("<hr>\n"), false);
        }
        mfxVersion lib = (impl_type) ? get_mfx_libsw_version() : get_mfx_libhw_version();
        const TCHAR *impl_str = (impl_type) ?  _T("Software") : _T("Hardware");
        if (!check_lib_version(lib, test)) {
            if (impl_type == 0) {
                if (type == FEATURE_LIST_STR_TYPE_HTML) {
                    print_tstring((bUseJapanese) ? _T("<b>QSVが使用できません。</b>") : _T("<b>QSV unavailable.</b>"), false);
                    char buffer[1024] = { 0 };
                    getCPUName(buffer, _countof(buffer));
                    tstring cpuname = char_to_tstring(buffer);
                    cpuname = cpuname.substr(cpuname.find(_T("Intel ") + _tcslen(_T("Intel "))));
                    cpuname = cpuname.substr(0, cpuname.find(_T(" @")));
                    cpuname = str_replace(cpuname, _T("Core2"), _T("Core 2"));
                    
                    if (bUseJapanese) {
                        print_tstring(_T("以下の項目を確認してみてください。"), true);
                        print_tstring(_T("<ol>\n"), false);
                        if (!bOSSupportsQSV) {
                            print_tstring(tstring(_T("<li>お使いのOS (") + osversion + _T(")はQSVに対応していません。<br>")
                                _T("QSVに対応しているOSはWindows7 / Windows8 / Windows8.1 / Windows10 のいずれかです。<br></li><br>\n")), false);
                        } else {
                            print_tstring(tstring(_T("<li>お使いのCPU (") + cpuname + _T(")がQSVに対応しているか、確認してみてください。<br>\n")
                                _T("<a target=\"_blank\" href=\"http://ark.intel.com/ja/search?q=") + cpuname + _T("\">こちらのリンク</a>から") +
                                _T("「グラフィックスの仕様」のところの「インテル クイック・シンク・ビデオ」が「Yes」になっているかで確認できます。<br>\n")
                                _T("<table class=simpleBlue><tr><td>QSVが使用できる例</td><td>QSVが使用できない例</td></tr>\n")
                                _T("<tr><td rowspan=2><img src=\"setup/intel_ark_qsv.png\" alt=\"intel_ark_qsv\" border=\"0\" width=\"480\"/><br>QSVが使用可能</td>\n")
                                _T("    <td><img src=\"setup/intel_ark_noqsv.png\" alt=\"intel_ark_noqsv\" border=\"0\" width=\"400\"/><br>GPUが搭載されていない場合</td></tr>\n")
                                _T("<tr><td><img src=\"setup/intel_ark_noqsv2.png\" alt=\"intel_ark_noqsv2\" border=\"0\" width=\"400\"/><br>QSVが使用できない場合</td></tr>\n")
                                _T("</tr></table></li><br>\n")), false);
                            print_tstring(tstring(_T("<li>QSV利用に必要なIntel GPUがPCで認識されているか確認してください。<br>\n")
                                _T("同梱の「デバイスマネージャを開く」をダブルクリックし、<br>\n")
                                _T("<img src=\"setup/intel_gpu_device_manager_open.png\" alt=\"intel_gpu_device_manager_open\" border=\"0\" width=\"240\"/><br>\n")
                                _T("デバイスマネージャの画面の「ディスプレイアダプタ」をクリックして")
                                _T("「Intel HD Graphics ～～」などとIntelのGPUが表示されていれば問題ありません。<br>\n")
                                _T("<img src=\"setup/intel_gpu_device_manager.png\" alt=\"intel_gpu_device_manager\" border=\"0\" width=\"280\"/><br>\n")
                                _T("Intel GPU以外にGPUが搭載されている場合、ここでIntel GPUが表示されない場合があります。\n")
                                _T("この場合、BIOS(UEFI)の「CPU Graphics Multi-Monitor」を有効(Enable)にする必要があります。<br>\n")
                                _T("<a target=\"_blank\" href=\"setup/intel_gpu_uefi_setting.jpg\"><img src=\"setup/intel_gpu_uefi_settings.jpg\" alt=\"intel_gpu_uefi_settings\" border=\"0\" width=\"400\"/></a><span style=font-size:x-small>(クリックして拡大)</span><br\n>")
                                _T("</li><br>\n")), false);
                            print_tstring(tstring(_T("<li>Intel GPUのドライバがWindows Update経由でインストールされた場合など、Intel ドライバのインストールが不完全な場合に正しく動作しないことがあります。<br>\n")
                                _T("<a target=\"_blank\" href=\"https://downloadcenter.intel.com/ja/search?keyword=") + cpuname + tstring(_T("\">こちらのリンク</a>から")) +
                                getOSVersion() + _T(" ") + tstring(rgy_is_64bit_os() ? _T("64bit") : _T("32bit")) + _T("用のドライバをダウンロードし、インストールしてみて下さい。<br>\n")
                                _T("<img src=\"setup/intel_driver_select.png\" alt=\"intel_driver_select\" border=\"0\" width=\"360\"/></li><br>\n")), false);
                            print_tstring(_T("</ol>\n"), false);
                            print_tstring(_T("<hr><br><br>\n"), false);
                            print_tstring(_T("導入方法等については、<a target=\"_blank\" href=\"http://rigaya34589.blog135.fc2.com/blog-entry-704.html\">こちら</a>もご覧ください。<br>\n"), false);
                            print_tstring(_T("<br><br><hr>\n"), false);
                        }
                    } else {
                        print_tstring(_T(" Please check the items below."), true);
                        print_tstring(_T("<ol>\n"), false);
                        if (!bOSSupportsQSV) {
                            print_tstring(tstring(_T("<li>Your OS (") + osversion + _T(")does not support QSV.<br>")
                                _T("QSV is supported on Windows7 / Windows8 / Windows8.1 / Windows10.<br></li><br>\n")), false);
                        } else {
                            print_tstring(tstring(_T("<li>Please check wether your CPU (") + cpuname + _T(") supports QSV.<br>\n")
                                _T("<a target=\"_blank\" href=\"http://ark.intel.com/search?q=") + cpuname + _T("\">Check from here</a>") +
                                _T(" whether \"Intel Quick Sync Video\" in \"Graphics Specifications\" says \"Yes\".<br>\n")
                                _T("<table class=simpleBlue><tr><td>QSV available</td><td>QSV unavailable</td></tr>\n")
                                _T("<tr><td rowspan=2><img src=\"setup/intel_ark_qsv_en.png\" alt=\"intel_ark_qsv_en\" border=\"0\" width=\"480\"/></td>\n")
                                _T("    <td><img src=\"setup/intel_ark_noqsv_en.png\" alt=\"intel_ark_noqsv_en\" border=\"0\" width=\"400\"/><br>example1</td></tr>\n")
                                _T("<tr><td><img src=\"setup/intel_ark_noqsv2_en.png\" alt=\"intel_ark_noqsv2_en\" border=\"0\" width=\"400\"/><br>example2</td></tr>\n")
                                _T("</tr></table></li><br>\n")), false);
                            print_tstring(tstring(_T("<li>Please check for device manager if Intel GPU is recognized under \"Display Adapter\".<br>\n")
                                _T("If you have discrete GPU on your PC, Intel GPU might not be shown.\n")
                                _T("For that case, you need yto enable \"CPU Graphics Multi-Monitor\" in your BIOS(UEFI).<br>\n")
                                _T("<a target=\"_blank\" href=\"setup/intel_gpu_uefi_setting.jpg\"><img src=\"setup/intel_gpu_uefi_settings.jpg\" alt=\"intel_gpu_uefi_settings\" border=\"0\" width=\"400\"/></a><span style=font-size:x-small>(Click ot enlarge)</span><br\n>")
                                _T("</li><br>\n")), false);
                            print_tstring(tstring(_T("<li>Sometimes Intel GPU driver is not installed properlly, especially when it is installed from Windows Update.<br>\n")
                                _T("Please install Intel GPU driver for") + getOSVersion() + _T(" ") + tstring(rgy_is_64bit_os() ? _T("64bit") : _T("32bit")) + _T(" ")
                                _T("<a target=\"_blank\" href=\"https://downloadcenter.intel.com/search?keyword=") + cpuname + tstring(_T("\">from here</a>")) +
                                _T(" and reinstall the driver.<br>\n")
                                _T("<img src=\"setup/intel_driver_select_en.png\" alt=\"intel_driver_select_en\" border=\"0\" width=\"320\"/></li><br>\n")), false);
                            print_tstring(_T("</ol>\n"), false);
                        }
                    }
                } else {
                    print_tstring(_T("QSV unavailable.\n"), false);
                }
            } else {
                print_tstring(strsprintf(_T("Media SDK %s unavailable.\n"), impl_str), true);
            }
        } else {
            const auto codec_feature_list = MakeFeatureListStr(0 == impl_type, type);
            if (codec_feature_list.size() == 0) {
                if (type == FEATURE_LIST_STR_TYPE_HTML) {
                    print_tstring((bUseJapanese) ? _T("<b>QSVが使用できません。</b><br>") : _T("<b>QSV unavailable.</b><br>"), false);

                    char buffer[1024] = { 0 };
                    getCPUName(buffer, _countof(buffer));
                    tstring cpuname = char_to_tstring(buffer);
                    cpuname = cpuname.substr(cpuname.find(_T("Intel ") + _tcslen(_T("Intel "))));
                    cpuname = cpuname.substr(0, cpuname.find(_T(" @")));

                    if (bUseJapanese) {
                        print_tstring(_T("以下の項目を確認してみてください。\n"), true);
                        print_tstring(_T("<ul>\n"), false);
                        print_tstring(tstring(_T("<li>Windows Update経由でインストールされた場合など、Intel ドライバのインストールが不完全な場合に正しく動作しないことがあります。<br>")
                            _T("<a target=\"_blank\" href=\"https://downloadcenter.intel.com/ja/search?keyword=") + cpuname + tstring(_T("\">こちら</a>から")) +
                            getOSVersion() + _T(" ") + tstring(rgy_is_64bit_os() ? _T("64bit") : _T("32bit")) + _T("用のドライバをダウンロードし、インストールしてみて下さい。</li>\n")), false);
                        print_tstring(_T("</ul>\n"), false);
                    }
                } else {
                    print_tstring(strsprintf(_T("Media SDK %s unavailable.\n"), impl_str), true);
                }
            } else {
                if (type == FEATURE_LIST_STR_TYPE_HTML) {
                    if (bUseJapanese) {
                        print_tstring(_T("<a target=\"_blank\" href=\"http://rigaya34589.blog135.fc2.com/blog-entry-337.html\">オプション一覧</a><br>\n"), false);
                        print_tstring(_T("<a target=\"_blank\" href=\"http://rigaya34589.blog135.fc2.com/blog-entry-704.html\">導入方法等について</a><br>\n"), false);
                        print_tstring(_T("<hr>\n"), false);
                    }
                    print_tstring((bUseJapanese) ? _T("<b>QSVが使用できます。</b><br>") : _T("<b>QSV available.</b><br>"), false);
                }
                print_tstring(strsprintf((bUseJapanese) ? _T("使用可能なMediaSDK: ") : _T("Media SDK Version: ")), false);
                print_tstring(strsprintf(_T("%s API v%d.%d\n\n"), impl_str, lib.Major, lib.Minor), true);
                auto codecHeader = (bUseJapanese) ? _T("エンコードに使用可能なコーデックとオプション:\n") : _T("Supported Enc features:\n");
                if (type == FEATURE_LIST_STR_TYPE_HTML) {
                    print_tstring(tstring(codecHeader) + _T("<br>"), false);
                } else {
                    print_tstring(codecHeader, false);
                }
                uint32_t i = 0;
                for (; i < codec_feature_list.size(); i++) {
                    auto codec_feature = codec_feature_list[i].second;
                    if (type == FEATURE_LIST_STR_TYPE_HTML) {
                        auto codec_name = codec_feature.substr(0, codec_feature.find_first_of('\n'));
                        codec_feature = codec_feature.substr(codec_feature.find_first_of('\n'));
                        if (bUseJapanese) {
                            codec_name = str_replace(codec_name, _T("Codec"), _T("コーデック"));
                        }
                        tstring optionHeader = (bUseJapanese) ? _T("利用可能なオプション") : _T("Available Options");
                        tstring str = codec_name;
                        str += strsprintf(_T("<div class=table_block id=\"TableOpen%d\">\n"), i);
                        str += _T("<p class=table_block>\n");
                        str += strsprintf(_T("<a class=table_block href=\"#\" title=\"%s▼\" onclick=\"showTable(%d);return false;\">%s▼</a>"), optionHeader.c_str(), i, optionHeader.c_str());
                        str += _T("</p>\n");
                        str += _T("</div>\n");
                        str += strsprintf(_T("<div class=table_block id=\"TableClose%d\" style=\"display: none\">\n"), i);
                        str += _T("<p class=table_block>\n");
                        str += strsprintf(_T("<a class=table_block href=\"#\" title=\"%s▲\" onclick=\"showTable(%d);return false;\">%s▲</a>\n"), optionHeader.c_str(), i, optionHeader.c_str());
                        str += _T("</p>\n");
                        str += codec_feature;
                        str += _T("</div><br>\n");
                        print_tstring(str, false);
                    } else {
                        print_tstring(strsprintf(_T("%s\n\n"), codec_feature.c_str()), false);
                    }
                }
                const auto vppHeader = tstring((bUseJapanese) ? _T("利用可能なVPP") : _T("Supported Vpp features:\n"));
                const auto vppFeatures = MakeVppFeatureStr(0 == impl_type, type);
                if (type == FEATURE_LIST_STR_TYPE_HTML) {
                    tstring str;
                    str += strsprintf(_T("<div class=table_block id=\"TableOpen%d\">\n"), i);
                    str += _T("<p class=table_block>\n");
                    str += strsprintf(_T("<a class=vpp_table_block href=\"#\" title=\"%s▼\" onclick=\"showTable(%d);return false;\">%s▼</a>"), vppHeader.c_str(), i, vppHeader.c_str());
                    str += _T("</p>\n");
                    str += _T("</div>\n");
                    str += strsprintf(_T("<div class=table_block id=\"TableClose%d\" style=\"display: none\">\n"), i);
                    str += _T("<p class=table_block>\n");
                    str += strsprintf(_T("<a class=vpp_table_block href=\"#\" title=\"%s▲\" onclick=\"showTable(%d);return false;\">%s▲</a>\n"), vppHeader.c_str(), i, vppHeader.c_str());
                    str += _T("</p>\n");
                    str += vppFeatures;
                    str += _T("</div><br>\n");
                    print_tstring(str, false);
                } else {
                    print_tstring(vppHeader + _T("\n"), true);
                    print_tstring(strsprintf(_T("%s\n\n"), vppFeatures.c_str()), false);
                }

                const auto decHeader = tstring((bUseJapanese) ? _T("利用可能なHWデコーダ") : _T("Supported Decode features:\n"));
                const auto decFeatures = MakeDecFeatureStr(0 == impl_type, type);
                if (type == FEATURE_LIST_STR_TYPE_HTML) {
                    tstring str;
                    str += strsprintf(_T("<div class=table_block id=\"TableOpen%d\">\n"), i);
                    str += _T("<p class=table_block>\n");
                    str += strsprintf(_T("<a class=vpp_table_block href=\"#\" title=\"%s▼\" onclick=\"showTable(%d);return false;\">%s▼</a>"), decHeader.c_str(), i, vppHeader.c_str());
                    str += _T("</p>\n");
                    str += _T("</div>\n");
                    str += strsprintf(_T("<div class=table_block id=\"TableClose%d\" style=\"display: none\">\n"), i);
                    str += _T("<p class=table_block>\n");
                    str += strsprintf(_T("<a class=vpp_table_block href=\"#\" title=\"%s▲\" onclick=\"showTable(%d);return false;\">%s▲</a>\n"), decHeader.c_str(), i, vppHeader.c_str());
                    str += _T("</p>\n");
                    str += vppFeatures;
                    str += _T("</div><br>\n");
                    print_tstring(str, false);
                } else {
                    print_tstring(decHeader + _T("\n"), true);
                    print_tstring(strsprintf(_T("%s\n\n"), decFeatures.c_str()), false);
                }
            }
        }
    }
    if (type == FEATURE_LIST_STR_TYPE_HTML) {
        print_tstring(_T("</div>\n</body>\n</html>"), false);
    }
    if (filename.length() && fp) {
        fclose(fp);
#if defined(_WIN32) || defined(_WIN64)
        TCHAR exePath[1024] = { 0 };
        if (32 <= (size_t)FindExecutable(filename.c_str(), nullptr, exePath) && _tcslen(exePath) && PathFileExists(exePath)) {
            ShellExecute(NULL, _T("open"), filename.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
        }
#endif //#if defined(_WIN32) || defined(_WIN64)
    }
    return 0;
}

struct sArgsData {
    tstring cachedlevel, cachedprofile;
    uint32_t nParsedAudioFile = 0;
    uint32_t nParsedAudioEncode = 0;
    uint32_t nParsedAudioCopy = 0;
    uint32_t nParsedAudioBitrate = 0;
    uint32_t nParsedAudioSamplerate = 0;
    uint32_t nParsedAudioSplit = 0;
    uint32_t nParsedAudioFilter = 0;
    uint32_t nTmpInputBuf = 0;
};

mfxStatus ParseOneOption(const TCHAR *option_name, const TCHAR* strInput[], int& i, int nArgNum, sInputParams* pParams, sArgsData *argData) {
    if (0 == _tcscmp(option_name, _T("output-res"))) {
        i++;
        if (   2 != _stscanf_s(strInput[i], _T("%hdx%hd"), &pParams->nDstWidth, &pParams->nDstHeight)
            && 2 != _stscanf_s(strInput[i], _T("%hd,%hd"), &pParams->nDstWidth, &pParams->nDstHeight)
            && 2 != _stscanf_s(strInput[i], _T("%hd/%hd"), &pParams->nDstWidth, &pParams->nDstHeight)
            && 2 != _stscanf_s(strInput[i], _T("%hd:%hd"), &pParams->nDstWidth, &pParams->nDstHeight)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("input-res"))) {
        i++;
        if (   2 != _stscanf_s(strInput[i], _T("%hdx%hd"), &pParams->nWidth, &pParams->nHeight)
            && 2 != _stscanf_s(strInput[i], _T("%hd,%hd"), &pParams->nWidth, &pParams->nHeight)
            && 2 != _stscanf_s(strInput[i], _T("%hd/%hd"), &pParams->nWidth, &pParams->nHeight)
            && 2 != _stscanf_s(strInput[i], _T("%hd:%hd"), &pParams->nWidth, &pParams->nHeight)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("crop"))) {
        i++;
        if (   4 != _stscanf_s(strInput[i], _T("%d,%d,%d,%d"), &pParams->sInCrop.e.left, &pParams->sInCrop.e.up, &pParams->sInCrop.e.right, &pParams->sInCrop.e.bottom)
            && 4 != _stscanf_s(strInput[i], _T("%d:%d:%d:%d"), &pParams->sInCrop.e.left, &pParams->sInCrop.e.up, &pParams->sInCrop.e.right, &pParams->sInCrop.e.bottom)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("codec"))) {
        i++;
        int j = 0;
        for (; list_codec[j].desc; j++) {
            if (_tcsicmp(list_codec[j].desc, strInput[i]) == 0) {
                pParams->CodecId = list_codec[j].value;
                break;
            }
        }
        if (list_codec[j].desc == nullptr) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("raw"))) {
        pParams->nInputFmt = RGY_INPUT_FMT_RAW;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("y4m"))) {
        pParams->nInputFmt = RGY_INPUT_FMT_Y4M;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("avi"))) {
        pParams->nInputFmt = RGY_INPUT_FMT_AVI;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("avs"))) {
        pParams->nInputFmt = RGY_INPUT_FMT_AVS;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpy"))) {
        pParams->nInputFmt = RGY_INPUT_FMT_VPY;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpy-mt"))) {
        pParams->nInputFmt = RGY_INPUT_FMT_VPY_MT;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("avqsv"))) {
        pParams->nInputFmt = RGY_INPUT_FMT_AVHW;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("avsw"))) {
        pParams->nInputFmt = RGY_INPUT_FMT_AVSW;
        return MFX_ERR_NONE;
    }
    if (   0 == _tcscmp(option_name, _T("input-analyze"))
        || 0 == _tcscmp(option_name, _T("avqsv-analyze"))) { //互換性のため
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        } else if (value < 0) {
            PrintHelp(strInput[0], _T("input-analyze requires non-negative value."), option_name);
            return MFX_PRINT_OPTION_ERR;
        } else {
            pParams->nAVDemuxAnalyzeSec = (mfxU16)((std::min)(value, USHRT_MAX));
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("input-file"))) {
        i++;
        _tcscpy_s(pParams->strSrcFile, strInput[i]);
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("output-file"))) {
        i++;
        if (!pParams->bBenchmark)
            _tcscpy_s(pParams->strDstFile, strInput[i]);
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("trim"))) {
        i++;
        auto trim_str_list = split(strInput[i], _T(","));
        std::vector<sTrim> trim_list;
        for (auto trim_str : trim_str_list) {
            sTrim trim;
            if (2 != _stscanf_s(trim_str.c_str(), _T("%d:%d"), &trim.start, &trim.fin) || (trim.fin > 0 && trim.fin < trim.start)) {
                PrintHelp(strInput[0], _T("Invalid Value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            if (trim.fin == 0) {
                trim.fin = TRIM_MAX;
            } else if (trim.fin < 0) {
                trim.fin = trim.start - trim.fin - 1;
            }
            trim_list.push_back(trim);
        }
        if (trim_list.size()) {
            std::sort(trim_list.begin(), trim_list.end(), [](const sTrim& trimA, const sTrim& trimB) { return trimA.start < trimB.start; });
            for (int j = (int)trim_list.size() - 2; j >= 0; j--) {
                if (trim_list[j].fin > trim_list[j+1].start) {
                    trim_list[j].fin = trim_list[j+1].fin;
                    trim_list.erase(trim_list.begin() + j+1);
                }
            }
            pParams->nTrimCount = (mfxU16)trim_list.size();
            pParams->pTrimList = (sTrim *)malloc(sizeof(pParams->pTrimList[0]) * trim_list.size());
            memcpy(pParams->pTrimList, &trim_list[0], sizeof(pParams->pTrimList[0]) * trim_list.size());
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("seek"))) {
        i++;
        int ret = 0;
        int hh = 0, mm = 0;
        float sec = 0.0f;
        if (   3 != (ret = _stscanf_s(strInput[i], _T("%d:%d:%f"),    &hh, &mm, &sec))
            && 2 != (ret = _stscanf_s(strInput[i],    _T("%d:%f"),         &mm, &sec))
            && 1 != (ret = _stscanf_s(strInput[i],       _T("%f"),              &sec))) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        if (ret <= 2) {
            hh = 0;
        }
        if (ret <= 1) {
            mm = 0;
        }
        if (hh < 0 || mm < 0 || sec < 0) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        if (hh > 0 && mm >= 60) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        mm += hh * 60;
        if (mm > 0 && sec >= 60.0f) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->fSeekSec = sec + mm * 60;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("video-track"))) {
        i++;
        int v = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        if (v == 0) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nVideoTrack = (int8_t)v;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("video-streamid"))) {
        i++;
        int v = 0;
        if (1 != _stscanf_s(strInput[i], _T("%i"), &v)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nVideoStreamId = v;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("audio-source"))) {
        i++;
        pParams->nAVMux |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        size_t audioSourceLen = _tcslen(strInput[i]) + 1;
        TCHAR *pAudioSource = (TCHAR *)malloc(sizeof(strInput[i][0]) * audioSourceLen);
        memcpy(pAudioSource, strInput[i], sizeof(strInput[i][0]) * audioSourceLen);
        pParams->ppAudioSourceList = (TCHAR **)realloc(pParams->ppAudioSourceList, sizeof(pParams->ppAudioSourceList[0]) * (pParams->nAudioSourceCount + 1));
        pParams->ppAudioSourceList[pParams->nAudioSourceCount] = pAudioSource;
        pParams->nAudioSourceCount++;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("audio-file"))) {
        i++;
        const TCHAR *ptr = strInput[i];
        sAudioSelect *pAudioSelect = nullptr;
        int audioIdx = -1;
        int trackId = 0;
        if (_tcschr(ptr, '?') == nullptr || 1 != _stscanf(ptr, _T("%d?"), &trackId)) {
            //トラック番号を適当に発番する (カウントは1から)
            trackId = argData->nParsedAudioFile+1;
            audioIdx = getAudioTrackIdx(pParams, trackId);
            if (audioIdx < 0 || pParams->ppAudioSelectList[audioIdx]->pAudioExtractFilename != nullptr) {
                trackId = getFreeAudioTrack(pParams);
                pAudioSelect = (sAudioSelect *)calloc(1, sizeof(pAudioSelect[0]));
                pAudioSelect->nAudioSelect = trackId;
            } else {
                pAudioSelect = pParams->ppAudioSelectList[audioIdx];
            }
        } else if (i <= 0) {
            //トラック番号は1から連番で指定
            PrintHelp(strInput[0], _T("Invalid track number"), option_name);
            return MFX_PRINT_OPTION_ERR;
        } else {
            audioIdx = getAudioTrackIdx(pParams, trackId);
            if (audioIdx < 0) {
                pAudioSelect = (sAudioSelect *)calloc(1, sizeof(pAudioSelect[0]));
                pAudioSelect->nAudioSelect = trackId;
            } else {
                pAudioSelect = pParams->ppAudioSelectList[audioIdx];
            }
            ptr = _tcschr(ptr, '?') + 1;
        }
        assert(pAudioSelect != nullptr);
        const TCHAR *qtr = _tcschr(ptr, ':');
        if (qtr != NULL && !(ptr + 1 == qtr && qtr[1] == _T('\\'))) {
            pAudioSelect->pAudioExtractFormat = _tcsdup(ptr);
            ptr = qtr + 1;
        }
        size_t filename_len = _tcslen(ptr);
        //ファイル名が""でくくられてたら取り除く
        if (ptr[0] == _T('\"') && ptr[filename_len-1] == _T('\"')) {
            filename_len -= 2;
            ptr++;
        }
        //ファイル名が重複していないかを確認する
        for (int j = 0; j < pParams->nAudioSelectCount; j++) {
            if (pParams->ppAudioSelectList[j]->pAudioExtractFilename != nullptr
                && 0 == _tcsicmp(pParams->ppAudioSelectList[j]->pAudioExtractFilename, ptr)) {
                PrintHelp(strInput[0], _T("Same output file name is used more than twice"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }

        if (audioIdx < 0) {
            audioIdx = pParams->nAudioSelectCount;
            //新たに要素を追加
            pParams->ppAudioSelectList = (sAudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
            pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
            pParams->nAudioSelectCount++;
        }
        pParams->ppAudioSelectList[audioIdx]->pAudioExtractFilename = _tcsdup(ptr);
        argData->nParsedAudioFile++;
        return MFX_ERR_NONE;
    }
    if (   0 == _tcscmp(option_name, _T("format"))
        || 0 == _tcscmp(option_name, _T("output-format"))) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            pParams->pAVMuxOutputFormat = _tcsdup(strInput[i]);
            if (0 != _tcsicmp(pParams->pAVMuxOutputFormat, _T("raw"))) {
                pParams->nAVMux |= RGY_MUX_VIDEO;
            }
        } else {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("input-format"))) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            pParams->pAVInputFormat = _tcsdup(strInput[i]);
        } else {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
#if ENABLE_AVSW_READER
    if (   0 == _tcscmp(option_name, _T("audio-copy"))
        || 0 == _tcscmp(option_name, _T("copy-audio"))) {
        pParams->nAVMux |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        std::set<int> trackSet; //重複しないよう、setを使う
        if (i+1 < nArgNum && (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0'))) {
            i++;
            auto trackListStr = split(strInput[i], _T(","));
            for (auto str : trackListStr) {
                int iTrack = 0;
                if (1 != _stscanf(str.c_str(), _T("%d"), &iTrack) || iTrack < 1) {
                    PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                    return MFX_PRINT_OPTION_ERR;
                } else {
                    trackSet.insert(iTrack);
                }
            }
        } else {
            trackSet.insert(0);
        }

        for (auto it = trackSet.begin(); it != trackSet.end(); it++) {
            int trackId = *it;
            sAudioSelect *pAudioSelect = nullptr;
            int audioIdx = getAudioTrackIdx(pParams, trackId);
            if (audioIdx < 0) {
                pAudioSelect = (sAudioSelect *)calloc(1, sizeof(pAudioSelect[0]));
                pAudioSelect->nAudioSelect = trackId;
            } else {
                pAudioSelect = pParams->ppAudioSelectList[audioIdx];
            }
            pAudioSelect->pAVAudioEncodeCodec = _tcsdup(RGY_AVCODEC_COPY);

            if (audioIdx < 0) {
                audioIdx = pParams->nAudioSelectCount;
                //新たに要素を追加
                pParams->ppAudioSelectList = (sAudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
                pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
                pParams->nAudioSelectCount++;
            }
            argData->nParsedAudioCopy++;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("audio-codec"))) {
        pParams->nAVMux |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        if (i+1 < nArgNum) {
            const TCHAR *ptr = nullptr;
            const TCHAR *ptrDelim = nullptr;
            if (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0')) {
                i++;
                ptrDelim = _tcschr(strInput[i], _T('?'));
                ptr = (ptrDelim == nullptr) ? strInput[i] : ptrDelim+1;
            }
            int trackId = 1;
            if (ptrDelim == nullptr) {
                trackId = argData->nParsedAudioEncode+1;
                int idx = getAudioTrackIdx(pParams, trackId);
                if (idx >= 0 && pParams->ppAudioSelectList[idx]->pAVAudioEncodeCodec != nullptr) {
                    trackId = getFreeAudioTrack(pParams);
                }
            } else {
                tstring temp = tstring(strInput[i]).substr(0, ptrDelim - strInput[i]);
                if (1 != _stscanf(temp.c_str(), _T("%d"), &trackId)) {
                    PrintHelp(strInput[0], _T("Invalid value"), option_name);
                    return MFX_PRINT_OPTION_ERR;
                }
            }
            sAudioSelect *pAudioSelect = nullptr;
            int audioIdx = getAudioTrackIdx(pParams, trackId);
            if (audioIdx < 0) {
                pAudioSelect = (sAudioSelect *)calloc(1, sizeof(pAudioSelect[0]));
                pAudioSelect->nAudioSelect = trackId;
            } else {
                pAudioSelect = pParams->ppAudioSelectList[audioIdx];
            }
            pAudioSelect->pAVAudioEncodeCodec = _tcsdup((ptr) ? ptr : RGY_AVCODEC_AUTO);

            if (audioIdx < 0) {
                audioIdx = pParams->nAudioSelectCount;
                //新たに要素を追加
                pParams->ppAudioSelectList = (sAudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
                pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
                pParams->nAudioSelectCount++;
            }
            argData->nParsedAudioEncode++;
        } else {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("audio-bitrate"))) {
        if (i+1 < nArgNum) {
            i++;
            const TCHAR *ptr = _tcschr(strInput[i], _T('?'));
            int trackId = 1;
            if (ptr == nullptr) {
                trackId = argData->nParsedAudioBitrate+1;
                int idx = getAudioTrackIdx(pParams, trackId);
                if (idx >= 0 && pParams->ppAudioSelectList[idx]->nAVAudioEncodeBitrate > 0) {
                    trackId = getFreeAudioTrack(pParams);
                }
                ptr = strInput[i];
            } else {
                tstring temp = tstring(strInput[i]).substr(0, ptr - strInput[i]);
                if (1 != _stscanf(temp.c_str(), _T("%d"), &trackId)) {
                    PrintHelp(strInput[0], _T("Invalid value"), option_name);
                    return MFX_PRINT_OPTION_ERR;
                }
                ptr++;
            }
            sAudioSelect *pAudioSelect = nullptr;
            int audioIdx = getAudioTrackIdx(pParams, trackId);
            if (audioIdx < 0) {
                pAudioSelect = (sAudioSelect *)calloc(1, sizeof(pAudioSelect[0]));
                pAudioSelect->nAudioSelect = trackId;
            } else {
                pAudioSelect = pParams->ppAudioSelectList[audioIdx];
            }
            int bitrate = 0;
            if (1 != _stscanf(ptr, _T("%d"), &bitrate)) {
                PrintHelp(strInput[0], _T("Invalid value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pAudioSelect->nAVAudioEncodeBitrate = bitrate;

            if (audioIdx < 0) {
                audioIdx = pParams->nAudioSelectCount;
                //新たに要素を追加
                pParams->ppAudioSelectList = (sAudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
                pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
                pParams->nAudioSelectCount++;
            }
            argData->nParsedAudioBitrate++;
        } else {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("audio-ignore-decode-error"))) {
        i++;
        uint32_t value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nAudioIgnoreDecodeError = value;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("audio-ignore-notrack-error"))) {
        pParams->bAudioIgnoreNoTrackError = 1;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("audio-samplerate"))) {
        if (i+1 < nArgNum) {
            i++;
            const TCHAR *ptr = _tcschr(strInput[i], _T('?'));
            int trackId = 1;
            if (ptr == nullptr) {
                trackId = argData->nParsedAudioSamplerate+1;
                int idx = getAudioTrackIdx(pParams, trackId);
                if (idx >= 0 && pParams->ppAudioSelectList[idx]->nAudioSamplingRate > 0) {
                    trackId = getFreeAudioTrack(pParams);
                }
                ptr = strInput[i];
            } else {
                tstring temp = tstring(strInput[i]).substr(0, ptr - strInput[i]);
                if (1 != _stscanf(temp.c_str(), _T("%d"), &trackId)) {
                    PrintHelp(strInput[0], _T("Invalid value"), option_name);
                    return MFX_PRINT_OPTION_ERR;
                }
                ptr++;
            }
            sAudioSelect *pAudioSelect = nullptr;
            int audioIdx = getAudioTrackIdx(pParams, trackId);
            if (audioIdx < 0) {
                pAudioSelect = (sAudioSelect *)calloc(1, sizeof(pAudioSelect[0]));
                pAudioSelect->nAudioSelect = trackId;
            } else {
                pAudioSelect = pParams->ppAudioSelectList[audioIdx];
            }
            int bitrate = 0;
            if (1 != _stscanf(ptr, _T("%d"), &bitrate)) {
                PrintHelp(strInput[0], _T("Invalid value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pAudioSelect->nAudioSamplingRate = bitrate;

            if (audioIdx < 0) {
                audioIdx = pParams->nAudioSelectCount;
                //新たに要素を追加
                pParams->ppAudioSelectList = (sAudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
                pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
                pParams->nAudioSelectCount++;
            }
            argData->nParsedAudioSamplerate++;
        } else {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("audio-resampler"))) {
        i++;
        mfxI32 v;
        if (PARSE_ERROR_FLAG != (v = get_value_from_chr(list_resampler, strInput[i]))) {
            pParams->nAudioResampler = (mfxU8)v;
        } else if (1 == _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_resampler) - 1) {
            pParams->nAudioResampler = (mfxU8)v;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("audio-stream"))) {
        if (!check_avcodec_dll()) {
            _ftprintf(stderr, _T("%s\n--audio-stream could not be used.\n"), error_mes_avcodec_dll_not_found().c_str());
            return MFX_PRINT_OPTION_ERR;
        }
        int trackId = -1;
        const TCHAR *ptr = nullptr;
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            ptr = _tcschr(strInput[i], _T('?'));
            if (ptr != nullptr) {
                tstring temp = tstring(strInput[i]).substr(0, ptr - strInput[i]);
                if (1 != _stscanf(temp.c_str(), _T("%d"), &trackId)) {
                    PrintHelp(strInput[0], _T("Invalid value"), option_name);
                    return MFX_PRINT_OPTION_ERR;
                }
                ptr++;
            } else {
                ptr = strInput[i];
            }
        }
        if (trackId < 0) {
            trackId = argData->nParsedAudioSplit+1;
            int idx = getAudioTrackIdx(pParams, trackId);
            if (idx >= 0 && bSplitChannelsEnabled(pParams->ppAudioSelectList[idx]->pnStreamChannelSelect)) {
                trackId = getFreeAudioTrack(pParams);
            }
        }
        sAudioSelect *pAudioSelect = nullptr;
        int audioIdx = getAudioTrackIdx(pParams, trackId);
        if (audioIdx < 0) {
            pAudioSelect = (sAudioSelect *)calloc(1, sizeof(pAudioSelect[0]));
            pAudioSelect->nAudioSelect = trackId;
        } else {
            pAudioSelect = pParams->ppAudioSelectList[audioIdx];
        }
        if (ptr == nullptr) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        } else {
            auto streamSelectList = split(tchar_to_string(ptr), ",");
            if (streamSelectList.size() > _countof(pAudioSelect->pnStreamChannelSelect)) {
                PrintHelp(strInput[0], _T("Too much streams splitted"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            static const char *DELIM = ":";
            for (uint32_t j = 0; j < streamSelectList.size(); j++) {
                auto selectPtr = streamSelectList[j].c_str();
                auto selectDelimPos = strstr(selectPtr, DELIM);
                if (selectDelimPos == nullptr) {
                    auto channelLayout = av_get_channel_layout(selectPtr);
                    pAudioSelect->pnStreamChannelSelect[j] = channelLayout;
                    pAudioSelect->pnStreamChannelOut[j]    = RGY_CHANNEL_AUTO; //自動
                } else if (selectPtr == selectDelimPos) {
                    pAudioSelect->pnStreamChannelSelect[j] = RGY_CHANNEL_AUTO;
                    pAudioSelect->pnStreamChannelOut[j]    = av_get_channel_layout(selectDelimPos + strlen(DELIM));
                } else {
                    pAudioSelect->pnStreamChannelSelect[j] = av_get_channel_layout(streamSelectList[j].substr(0, selectDelimPos - selectPtr).c_str());
                    pAudioSelect->pnStreamChannelOut[j]    = av_get_channel_layout(selectDelimPos + strlen(DELIM));
                }
            }
        }
        if (audioIdx < 0) {
            audioIdx = pParams->nAudioSelectCount;
            //新たに要素を追加
            pParams->ppAudioSelectList = (sAudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
            pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
            pParams->nAudioSelectCount++;
        }
        argData->nParsedAudioSplit++;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("audio-filter"))) {
        if (i+1 < nArgNum) {
            const TCHAR *ptr = nullptr;
            const TCHAR *ptrDelim = nullptr;
            if (strInput[i+1][0] != _T('-')) {
                i++;
                ptrDelim = _tcschr(strInput[i], _T('?'));
                ptr = (ptrDelim == nullptr) ? strInput[i] : ptrDelim+1;
            } else {
                PrintHelp(strInput[0], _T("Invalid value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            int trackId = 1;
            if (ptrDelim == nullptr) {
                trackId = argData->nParsedAudioFilter+1;
                int idx = getAudioTrackIdx(pParams, trackId);
                if (idx >= 0 && pParams->ppAudioSelectList[idx]->pAudioFilter != nullptr) {
                    trackId = getFreeAudioTrack(pParams);
                }
            } else {
                tstring temp = tstring(strInput[i]).substr(0, ptrDelim - strInput[i]);
                if (1 != _stscanf(temp.c_str(), _T("%d"), &trackId)) {
                    PrintHelp(strInput[0], _T("Invalid value"), option_name);
                    return MFX_PRINT_OPTION_ERR;
                }
            }
            sAudioSelect *pAudioSelect = nullptr;
            int audioIdx = getAudioTrackIdx(pParams, trackId);
            if (audioIdx < 0) {
                pAudioSelect = (sAudioSelect *)calloc(1, sizeof(pAudioSelect[0]));
                pAudioSelect->nAudioSelect = trackId;
            } else {
                pAudioSelect = pParams->ppAudioSelectList[audioIdx];
            }
            if (pAudioSelect->pAudioFilter) {
                free(pAudioSelect->pAudioFilter);
            }
            pAudioSelect->pAudioFilter = _tcsdup(ptr);

            if (audioIdx < 0) {
                audioIdx = pParams->nAudioSelectCount;
                //新たに要素を追加
                pParams->ppAudioSelectList = (sAudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
                pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
                pParams->nAudioSelectCount++;
            }
            argData->nParsedAudioFilter++;
        } else {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
#endif //#if ENABLE_AVSW_READER
    if (   0 == _tcscmp(option_name, _T("chapter-copy"))
        || 0 == _tcscmp(option_name, _T("copy-chapter"))) {
        pParams->bCopyChapter = TRUE;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("chapter"))) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            pParams->pChapterFile = _tcsdup(strInput[i]);
        } else {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("chapter-no-trim"))) {
        pParams->bChapterNoTrim = TRUE;
        return MFX_ERR_NONE;
    }
    if (   0 == _tcscmp(option_name, _T("sub-copy"))
        || 0 == _tcscmp(option_name, _T("copy-sub"))) {
        pParams->nAVMux |= (RGY_MUX_VIDEO | RGY_MUX_SUBTITLE);
        std::set<int> trackSet; //重複しないよう、setを使う
        if (i+1 < nArgNum && (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0'))) {
            i++;
            auto trackListStr = split(strInput[i], _T(","));
            for (auto str : trackListStr) {
                int iTrack = 0;
                if (1 != _stscanf(str.c_str(), _T("%d"), &iTrack) || iTrack < 1) {
                    PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                    return MFX_PRINT_OPTION_ERR;
                } else {
                    trackSet.insert(iTrack);
                }
            }
        } else {
            trackSet.insert(0);
        }
        for (int iTrack = 0; iTrack < pParams->nSubtitleSelectCount; iTrack++) {
            trackSet.insert(pParams->pSubtitleSelect[iTrack]);
        }
        if (pParams->pSubtitleSelect) {
            free(pParams->pSubtitleSelect);
        }

        pParams->pSubtitleSelect = (int *)malloc(sizeof(pParams->pSubtitleSelect[0]) * trackSet.size());
        pParams->nSubtitleSelectCount = (mfxU8)trackSet.size();
        int iTrack = 0;
        for (auto it = trackSet.begin(); it != trackSet.end(); it++, iTrack++) {
            pParams->pSubtitleSelect[iTrack] = *it;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("avsync"))) {
        int value = 0;
        i++;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_avsync, strInput[i]))) {
            pParams->nAVSyncMode = (RGYAVSync)value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("mux-option"))) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            auto ptr = _tcschr(strInput[i], ':');
            if (ptr == nullptr) {
                PrintHelp(strInput[0], _T("invalid value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            } else {
                if (pParams->pMuxOpt == nullptr) {
                    pParams->pMuxOpt = new muxOptList();
                }
                pParams->pMuxOpt->push_back(std::make_pair<tstring, tstring>(tstring(strInput[i]).substr(0, ptr - strInput[i]), tstring(ptr+1)));
            }
        } else {
            PrintHelp(strInput[0], _T("invalid option"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("quality"))) {
        i++;
        int value = MFX_TARGETUSAGE_BALANCED;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->nTargetUsage = (mfxU16)clamp(value, MFX_TARGETUSAGE_BEST_QUALITY, MFX_TARGETUSAGE_BEST_SPEED);
        } else if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_quality_for_option, strInput[i]))) {
            pParams->nTargetUsage = (mfxU16)value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("level"))) {
        if (i+1 < nArgNum) {
            i++;
            argData->cachedlevel = strInput[i];
        } else {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("profile"))) {
        if (i+1 < nArgNum) {
            i++;
            argData->cachedprofile = strInput[i];
        } else {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (   0 == _tcscmp(option_name, _T("sar"))
        || 0 == _tcscmp(option_name, _T("dar"))) {
        i++;
        int value[2] ={ 0 };
        if (   2 != _stscanf_s(strInput[i], _T("%dx%d"), &value[0], &value[1])
            && 2 != _stscanf_s(strInput[i], _T("%d,%d"), &value[0], &value[1])
            && 2 != _stscanf_s(strInput[i], _T("%d/%d"), &value[0], &value[1])
            && 2 != _stscanf_s(strInput[i], _T("%d:%d"), &value[0], &value[1])) {
            RGY_MEMSET_ZERO(pParams->nPAR);
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        if (0 == _tcscmp(option_name, _T("dar"))) {
            value[0] = -value[0];
            value[1] = -value[1];
        }
        pParams->nPAR[0] = value[0];
        pParams->nPAR[1] = value[1];
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("sw"))) {
        pParams->bUseHWLib = false;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("slices"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nSlices)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("gop-len"))) {
        i++;
        if (0 == _tcsnccmp(strInput[i], _T("auto"), _tcslen(_T("auto")))) {
            pParams->nGOPLength = 0;
        } else if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nGOPLength)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("open-gop"))) {
        pParams->bopenGOP = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-open-gop"))) {
        pParams->bopenGOP = false;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("strict-gop"))) {
        pParams->bforceGOPSettings = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-scenechange"))) {
        pParams->bforceGOPSettings = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("scenechange"))) {
        pParams->bforceGOPSettings = false;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("i-adapt"))) {
        pParams->bAdaptiveI = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-i-adapt"))) {
        pParams->bAdaptiveI = false;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("b-adapt"))) {
        pParams->bAdaptiveB = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-b-adapt"))) {
        pParams->bAdaptiveB = false;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("b-pyramid"))) {
        pParams->bBPyramid = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-b-pyramid"))) {
        pParams->bBPyramid = false;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("weightb"))) {
        pParams->nWeightB = MFX_WEIGHTED_PRED_IMPLICIT;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-weightb"))) {
        pParams->nWeightB = MFX_WEIGHTED_PRED_UNKNOWN;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("weightp"))) {
        pParams->nWeightP = MFX_WEIGHTED_PRED_DEFAULT;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-weightp"))) {
        pParams->nWeightP = MFX_WEIGHTED_PRED_UNKNOWN;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("repartition-check"))) {
        pParams->nRepartitionCheck = MFX_CODINGOPTION_ON;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-repartition-check"))) {
        pParams->nRepartitionCheck = MFX_CODINGOPTION_OFF;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("fade-detect"))) {
        pParams->nFadeDetect = MFX_CODINGOPTION_ON;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-fade-detect"))) {
        pParams->nFadeDetect = MFX_CODINGOPTION_OFF;
        return MFX_ERR_NONE;
    }
    if (   0 == _tcscmp(option_name, _T("lookahead-ds"))
        || 0 == _tcscmp(option_name, _T("la-quality"))) {
        i++;
        int value = MFX_LOOKAHEAD_DS_UNKNOWN;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_lookahead_ds, strInput[i]))) {
            pParams->nLookaheadDS = (mfxU16)value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("trellis"))) {
        i++;
        int value = MFX_TRELLIS_UNKNOWN;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_avc_trellis_for_options, strInput[i]))) {
            pParams->nTrellis = (mfxU16)value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("bluray")) || 0 == _tcscmp(option_name, _T("force-bluray"))) {
        pParams->nBluray = (0 == _tcscmp(option_name, _T("force-bluray"))) ? 2 : 1;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("nv12"))) {
        pParams->ColorFormat = MFX_FOURCC_NV12;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("tff"))) {
        pParams->nPicStruct = MFX_PICSTRUCT_FIELD_TFF;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("bff"))) {
        pParams->nPicStruct = MFX_PICSTRUCT_FIELD_BFF;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("la"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nEncMode = MFX_RATECONTROL_LA;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("icq"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nICQQuality)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nEncMode = MFX_RATECONTROL_ICQ;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("la-icq"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nICQQuality)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nEncMode = MFX_RATECONTROL_LA_ICQ;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("la-hrd"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nEncMode = MFX_RATECONTROL_LA_HRD;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vcm"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nEncMode = MFX_RATECONTROL_VCM;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vbr"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nEncMode = MFX_RATECONTROL_VBR;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("cbr"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nEncMode = MFX_RATECONTROL_CBR;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("avbr"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nEncMode = MFX_RATECONTROL_AVBR;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("qvbr"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nEncMode = MFX_RATECONTROL_QVBR;
        return MFX_ERR_NONE;
    }
    if (   0 == _tcscmp(option_name, _T("qvbr-q"))
        || 0 == _tcscmp(option_name, _T("qvbr-quality"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nQVBRQuality)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nEncMode = MFX_RATECONTROL_QVBR;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("fallback-rc"))) {
        pParams->nFallback = 1;
        return MFX_ERR_NONE;
    }
    if (   0 == _tcscmp(option_name, _T("max-bitrate"))
        || 0 == _tcscmp(option_name, _T("maxbitrate"))) //互換性のため
    {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nMaxBitrate)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("la-depth"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nLookaheadDepth)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("la-window-size"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nWinBRCSize)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("cqp")) || 0 == _tcscmp(option_name, _T("vqp"))) {
        i++;
        if (   3 != _stscanf_s(strInput[i], _T("%hd:%hd:%hd"), &pParams->nQPI, &pParams->nQPP, &pParams->nQPB)
            && 3 != _stscanf_s(strInput[i], _T("%hd,%hd,%hd"), &pParams->nQPI, &pParams->nQPP, &pParams->nQPB)
            && 3 != _stscanf_s(strInput[i], _T("%hd/%hd/%hd"), &pParams->nQPI, &pParams->nQPP, &pParams->nQPB)) {
            if (1 == _stscanf_s(strInput[i], _T("%hd"), &pParams->nQPI)) {
                pParams->nQPP = pParams->nQPI;
                pParams->nQPB = pParams->nQPI;
            } else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        pParams->nEncMode = (mfxU16)((0 == _tcscmp(option_name, _T("vqp"))) ? MFX_RATECONTROL_VQP : MFX_RATECONTROL_CQP);
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("avbr-unitsize"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nAVBRConvergence)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    //if (0 == _tcscmp(option_name, _T("avbr-range")))
    //{
    //    double accuracy;
    //    if (1 != _stscanf_s(strArgument, _T("%f"), &accuracy)) {
    //        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
    //        return MFX_PRINT_OPTION_ERR;
    //    }
    //    pParams->nAVBRAccuarcy = (mfxU16)(accuracy * 10 + 0.5);
    //    return MFX_ERR_NONE;
    //}
    else if (0 == _tcscmp(option_name, _T("fixed-func"))) {
        pParams->bUseFixedFunc = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-fixed-func"))) {
        pParams->bUseFixedFunc = false;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("ref"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nRef)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("bframes"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nBframes)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("cavlc"))) {
        pParams->bCAVLC = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("rdo"))) {
        pParams->bRDO = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("extbrc"))) {
        pParams->bExtBRC = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-extbrc"))) {
        pParams->bExtBRC = false;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("mbbrc"))) {
        pParams->bMBBRC = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-mbbrc"))) {
        pParams->bMBBRC = false;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-intra-refresh"))) {
        pParams->bIntraRefresh = false;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("intra-refresh"))) {
        pParams->bIntraRefresh = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-deblock"))) {
        pParams->bNoDeblock = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("qpmax")) || 0 == _tcscmp(option_name, _T("qpmin"))
        || 0 == _tcscmp(option_name, _T("qp-max")) || 0 == _tcscmp(option_name, _T("qp-min"))) {
        i++;
        int qpLimit[3] = { 0 };
        if (   3 != _stscanf_s(strInput[i], _T("%d:%d:%d"), &qpLimit[0], &qpLimit[1], &qpLimit[2])
            && 3 != _stscanf_s(strInput[i], _T("%d,%d,%d"), &qpLimit[0], &qpLimit[1], &qpLimit[2])
            && 3 != _stscanf_s(strInput[i], _T("%d/%d/%d"), &qpLimit[0], &qpLimit[1], &qpLimit[2])) {
            if (1 == _stscanf_s(strInput[i], _T("%d"), &qpLimit[0])) {
                qpLimit[1] = qpLimit[0];
                qpLimit[2] = qpLimit[0];
            } else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        uint8_t *limit = (0 == _tcscmp(option_name, _T("qpmin")) || 0 == _tcscmp(option_name, _T("qp-min"))) ? pParams->nQPMin : pParams->nQPMax;
        for (int j = 0; j < 3; j++) {
            limit[j] = (uint8_t)clamp(qpLimit[j], 0, 51);
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("qp-offset"))) {
        i++;
        auto values = split(strInput[i], _T(":"), true);
        if (values.size() == 0) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        if (values.size() > 8) {
            PrintHelp(strInput[0], strsprintf(_T("qp-offset value could be set up to 8 layers, but was set for %d layers.\n"), (int)values.size()).c_str(), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        uint32_t iv = 0;
        for (; iv < values.size(); iv++) {
            TCHAR *eptr = nullptr;
            int v = _tcstol(values[iv].c_str(), &eptr, 0);
            if (v == 0 && (eptr != nullptr || *eptr == ' ')) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[iv]);
                return MFX_PRINT_OPTION_ERR;
            }
            if (v < -51 || v > 51) {
                PrintHelp(strInput[0], _T("qp-offset value should be in range of -51 - 51.\n"), option_name, strInput[i]);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->pQPOffset[iv] = (int8_t)v;
        }
        for (; iv < _countof(pParams->pQPOffset); iv++) {
            pParams->pQPOffset[iv] = pParams->pQPOffset[iv-1];
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("mv-scaling"))) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->bGlobalMotionAdjust = true;
            pParams->nMVCostScaling = (mfxU8)value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("direct-bias-adjust"))) {
        pParams->bDirectBiasAdjust = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-direct-bias-adjust"))) {
        pParams->bDirectBiasAdjust = false;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("fullrange"))) {
        pParams->bFullrange = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("inter-pred"))) {
        i++;
        mfxI32 v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_pred_block_size) - 1) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nInterPred = (mfxU16)list_pred_block_size[v].value;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("intra-pred"))) {
        i++;
        mfxI32 v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_pred_block_size) - 1) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nIntraPred = (mfxU16)list_pred_block_size[v].value;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("mv-precision"))) {
        i++;
        mfxI32 v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_mv_presicion) - 1) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nMVPrecision = (mfxU16)list_mv_presicion[v].value;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("mv-search"))) {
        i++;
        mfxI32 v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->MVSearchWindow.x = (mfxU16)clamp(v, 0, 128);
        pParams->MVSearchWindow.y = (mfxU16)clamp(v, 0, 128);
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("sharpness"))) {
        i++;
        mfxI32 v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < 8) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nVP8Sharpness = (mfxU8)v;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("fps"))) {
        i++;
        if (   2 != _stscanf_s(strInput[i], _T("%d/%d"), &pParams->nFPSRate, &pParams->nFPSScale)
            && 2 != _stscanf_s(strInput[i], _T("%d:%d"), &pParams->nFPSRate, &pParams->nFPSScale)
            && 2 != _stscanf_s(strInput[i], _T("%d,%d"), &pParams->nFPSRate, &pParams->nFPSScale)) {
            double d;
            if (1 == _stscanf_s(strInput[i], _T("%lf"), &d)) {
                int rate = (int)(d * 1001.0 + 0.5);
                if (rate % 1000 == 0) {
                    pParams->nFPSRate = rate;
                    pParams->nFPSScale = 1001;
                } else {
                    pParams->nFPSScale = 100000;
                    pParams->nFPSRate = (int)(d * pParams->nFPSScale + 0.5);
                    rgy_reduce(pParams->nFPSRate, pParams->nFPSScale);
                }
            } else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("log-level"))) {
        i++;
        mfxI32 v;
        if (PARSE_ERROR_FLAG != (v = get_value_from_chr(list_log_level, strInput[i]))) {
            pParams->nLogLevel = (mfxI16)v;
        } else if (1 == _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_log_level) - 1) {
            pParams->nLogLevel = (mfxI16)v;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
#ifdef D3D_SURFACES_SUPPORT
    if (0 == _tcscmp(option_name, _T("disable-d3d"))) {
        pParams->memType = SYSTEM_MEMORY;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("d3d9"))) {
        pParams->memType = D3D9_MEMORY;
        return MFX_ERR_NONE;
    }
#if MFX_D3D11_SUPPORT
    if (0 == _tcscmp(option_name, _T("d3d11"))) {
        pParams->memType = D3D11_MEMORY;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("d3d"))) {
        pParams->memType = HW_MEMORY;
        return MFX_ERR_NONE;
    }
#else
    if (0 == _tcscmp(option_name, _T("d3d"))) {
        pParams->memType = D3D9_MEMORY;
        return MFX_ERR_NONE;
    }
#endif //MFX_D3D11_SUPPORT
#endif //D3D_SURFACES_SUPPORT
#ifdef LIBVA_SUPPORT
    if (0 == _tcscmp(option_name, _T("va"))) {
        pParams->memType = D3D9_MEMORY;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("disable-va"))) {
        pParams->memType = SYSTEM_MEMORY;
        return MFX_ERR_NONE;
    }
#endif //#ifdef LIBVA_SUPPORT
    if (0 == _tcscmp(option_name, _T("aud"))) {
        pParams->bOutputAud = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("pic-struct"))) {
        pParams->bOutputPicStruct = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("async-depth"))) {
        i++;
        int v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v) || v < 0 || QSV_ASYNC_DEPTH_MAX < v) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nAsyncDepth = (mfxU16)v;
        return MFX_ERR_NONE;
    }
#if ENABLE_SESSION_THREAD_CONFIG
    if (0 == _tcscmp(option_name, _T("session-threads"))) {
        i++;
        int v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v) || v < 0 || QSV_SESSION_THREAD_MAX < v) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nSessionThreads = (mfxU16)v;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("session-thread-priority"))
        || 0 == _tcscmp(option_name, _T("session-threads-priority"))) {
        i++;
        mfxI32 v;
        if (PARSE_ERROR_FLAG == (v = get_value_from_chr(list_priority, strInput[i]))
            && 1 != _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_log_level) - 1) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nSessionThreadPriority = (mfxU16)v;
        return MFX_ERR_NONE;
    }
#endif
    if (0 == _tcscmp(option_name, _T("vpp-denoise"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->vpp.nDenoise)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->vpp.bEnable = true;
        pParams->vpp.bUseDenoise = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-no-denoise"))) {
        i++;
        pParams->vpp.bUseDenoise = false;
        pParams->vpp.nDenoise = 0;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-detail-enhance"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->vpp.nDetailEnhance)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->vpp.bEnable = true;
        pParams->vpp.bUseDetailEnhance = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-no-detail-enhance"))) {
        i++;
        pParams->vpp.bUseDetailEnhance = false;
        pParams->vpp.nDetailEnhance = 0;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-deinterlace"))) {
        i++;
        int value = get_value_from_chr(list_deinterlace, strInput[i]);
        if (PARSE_ERROR_FLAG == value) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->vpp.bEnable = true;
        pParams->vpp.nDeinterlace = (mfxU16)value;
        if (pParams->vpp.nDeinterlace == MFX_DEINTERLACE_IT_MANUAL) {
            i++;
            if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_telecine_patterns, strInput[i]))) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return MFX_PRINT_OPTION_ERR;
            } else {
                pParams->vpp.nTelecinePattern = (mfxU16)value;
            }
        }
        if (pParams->vpp.nDeinterlace == MFX_DEINTERLACE_NONE
            && pParams->nPicStruct == MFX_PICSTRUCT_PROGRESSIVE) {
            pParams->nPicStruct = MFX_PICSTRUCT_FIELD_TFF;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-image-stab"))) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->vpp.nImageStabilizer = (mfxU16)value;
        } else if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_vpp_image_stabilizer, strInput[i]))) {
            pParams->vpp.nImageStabilizer = (mfxU16)value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-fps-conv"))) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->vpp.nFPSConversion = (mfxU16)value;
        } else if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_vpp_fps_conversion, strInput[i]))) {
            pParams->vpp.nFPSConversion = (mfxU16)value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-half-turn"))) {
        pParams->vpp.bHalfTurn = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-rotate"))) {
        i++;
        int value = 0;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_vpp_rotate_angle, strInput[i]))) {
            pParams->vpp.nRotate = (mfxU16)value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-mirror"))) {
        i++;
        int value = 0;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_vpp_mirroring, strInput[i]))) {
            pParams->vpp.nMirrorType = (mfxU16)value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-scaling"))) {
        i++;
        int value = 0;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_vpp_scaling_quality, strInput[i]))) {
            pParams->vpp.nScalingQuality = (mfxU16)value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
#if ENABLE_CUSTOM_VPP
#if ENABLE_AVSW_READER && ENABLE_LIBASS_SUBBURN
    if (0 == _tcscmp(option_name, _T("vpp-sub"))) {
        if (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0')) {
            i++;
            TCHAR *endPtr = nullptr;
            int nSubTrack = _tcstol(strInput[i], &endPtr, 10);
            if (pParams->vpp.subburn.pFilePath) {
                free(pParams->vpp.subburn.pFilePath);
            }
            if (0 < nSubTrack && (endPtr == nullptr || *endPtr == _T('\0'))) {
                pParams->vpp.subburn.nTrack = nSubTrack;
                pParams->vpp.subburn.pFilePath = nullptr;
            } else {
                pParams->vpp.subburn.nTrack = 0;
                pParams->vpp.subburn.pFilePath = _tcsdup(strInput[i]);
            }
        } else {
            pParams->vpp.subburn.nTrack = 1;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-sub-charset"))) {
        if (i+1 < nArgNum && (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0'))) {
            i++;
            if (pParams->vpp.subburn.pCharEnc) {
                free(pParams->vpp.subburn.pCharEnc);
            }
            pParams->vpp.subburn.pCharEnc = _tcsdup(strInput[i]);
        } else {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-sub-shaping"))) {
        i++;
        int v;
        if (PARSE_ERROR_FLAG != (v = get_value_from_chr(list_vpp_sub_shaping, strInput[i]))) {
            pParams->vpp.subburn.nShaping = v;
        } else if (1 == _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_vpp_sub_shaping) - 1) {
            pParams->vpp.subburn.nShaping = v;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
#endif //#if ENABLE_AVSW_READER && ENABLE_LIBASS_SUBBURN
    if (   0 == _tcscmp(option_name, _T("vpp-delogo"))
        || 0 == _tcscmp(option_name, _T("vpp-delogo-file"))) {
        i++;
        int filename_len = (int)_tcslen(strInput[i]);
        pParams->vpp.delogo.pFilePath = (TCHAR *)calloc(filename_len + 1, sizeof(pParams->vpp.delogo.pFilePath[0]));
        memcpy(pParams->vpp.delogo.pFilePath, strInput[i], sizeof(pParams->vpp.delogo.pFilePath[0]) * filename_len);
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-delogo-select"))) {
        i++;
        int filename_len = (int)_tcslen(strInput[i]);
        pParams->vpp.delogo.pSelect = (TCHAR *)calloc(filename_len + 1, sizeof(pParams->vpp.delogo.pSelect[0]));
        memcpy(pParams->vpp.delogo.pSelect, strInput[i], sizeof(pParams->vpp.delogo.pSelect[0]) * filename_len);
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-delogo-pos"))) {
        i++;
        mfxI16Pair posOffset;
        if (   2 != _stscanf_s(strInput[i], _T("%hdx%hd"), &posOffset.x, &posOffset.y)
            && 2 != _stscanf_s(strInput[i], _T("%hd,%hd"), &posOffset.x, &posOffset.y)
            && 2 != _stscanf_s(strInput[i], _T("%hd/%hd"), &posOffset.x, &posOffset.y)
            && 2 != _stscanf_s(strInput[i], _T("%hd:%hd"), &posOffset.x, &posOffset.y)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->vpp.delogo.nPosOffset = posOffset;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-delogo-depth"))) {
        i++;
        int depth;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &depth)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->vpp.delogo.nDepth = (uint8_t)clamp(depth, 0, 255);
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-delogo-y"))) {
        i++;
        mfxI16 value;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->vpp.delogo.nYOffset = value;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-delogo-cb"))) {
        i++;
        mfxI16 value;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->vpp.delogo.nCbOffset = value;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-delogo-cr"))) {
        i++;
        mfxI16 value;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->vpp.delogo.nCrOffset = value;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("vpp-delogo-add"))) {
        pParams->vpp.delogo.bAdd = 1;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-vpp-delogo-add"))) {
        pParams->vpp.delogo.bAdd = 0;
        return MFX_ERR_NONE;
    }
#endif //#if ENABLE_CUSTOM_VPP
    if (0 == _tcscmp(option_name, _T("input-buf"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &argData->nTmpInputBuf)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("output-buf"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        if (value < 0) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nOutputBufSizeMB = (int16_t)(std::min)(value, RGY_OUTPUT_BUF_MB_MAX);
        return MFX_ERR_NONE;
    }
#if defined(_WIN32) || defined(_WIN64)
    if (0 == _tcscmp(option_name, _T("mfx-thread"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        if (value < -1) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nSessionThreads = (int16_t)value;
        return MFX_ERR_NONE;
    }
#endif
    if (0 == _tcscmp(option_name, _T("input-thread"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        if (value < -1 || value >= 2) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nInputThread = (int8_t)value;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-output-thread"))) {
        pParams->nOutputThread = 0;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("output-thread"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        if (value < -1 || value >= 2) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nOutputThread = (int8_t)value;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("audio-thread"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        if (value < -1 || value >= 3) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nAudioThread = (int8_t)value;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("min-memory"))) {
        pParams->nOutputThread = 0;
        pParams->nAudioThread = 0;
        pParams->nAsyncDepth = 1;
        argData->nTmpInputBuf = 1;
        pParams->nOutputBufSizeMB = 0;
        pParams->nSessionThreads = 2;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("max-procfps"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        if (value < 0) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nProcSpeedLimit = (uint16_t)(std::min)(value, (int)UINT16_MAX);
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("log"))) {
        i++;
        int filename_len = (int)_tcslen(strInput[i]);
        pParams->pStrLogFile = (TCHAR *)calloc(filename_len + 1, sizeof(pParams->pStrLogFile[0]));
        memcpy(pParams->pStrLogFile, strInput[i], sizeof(pParams->pStrLogFile[0]) * filename_len);
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("log-framelist"))) {
        i++;
        int filename_len = (int)_tcslen(strInput[i]);
        pParams->pFramePosListLog = (TCHAR *)calloc(filename_len + 1, sizeof(pParams->pFramePosListLog[0]));
        memcpy(pParams->pFramePosListLog, strInput[i], sizeof(pParams->pFramePosListLog[0]) * filename_len);
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("log-mux-ts"))) {
        i++;
        pParams->pMuxVidTsLogFile = _tcsdup(strInput[i]);
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("log-copy-framedata"))) {
        i++;
        pParams->pLogCopyFrameData = _tcsdup(strInput[i]);
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("colormatrix"))) {
        i++;
        int value;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_colormatrix, strInput[i])))
            pParams->ColorMatrix = (mfxU16)value;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("colorprim"))) {
        i++;
        int value;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_colorprim, strInput[i])))
            pParams->ColorPrim = (mfxU16)value;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("transfer"))) {
        i++;
        int value;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_transfer, strInput[i])))
            pParams->Transfer = (mfxU16)value;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("videoformat"))) {
        i++;
        int value;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_videoformat, strInput[i])))
            pParams->ColorMatrix = (mfxU16)value;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("fullrange"))) {
        pParams->bFullrange = true;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("sar"))) {
        i++;
        if (   2 != _stscanf_s(strInput[i], _T("%dx%d"), &pParams->nPAR[0], &pParams->nPAR[1])
            && 2 != _stscanf_s(strInput[i], _T("%d,%d"), &pParams->nPAR[0], &pParams->nPAR[1])
            && 2 != _stscanf_s(strInput[i], _T("%d/%d"), &pParams->nPAR[0], &pParams->nPAR[1])
            && 2 != _stscanf_s(strInput[i], _T("%d:%d"), &pParams->nPAR[0], &pParams->nPAR[1])) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("benchmark"))) {
        i++;
        pParams->bBenchmark = TRUE;
        _tcscpy_s(pParams->strDstFile, strInput[i]);
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("bench-quality"))) {
        i++;
        pParams->bBenchmark = TRUE;
        if (0 == _tcscmp(strInput[i], _T("all"))) {
            pParams->nBenchQuality = 0xffffffff;
        } else {
            pParams->nBenchQuality = 0;
            auto list = split(tstring(strInput[i]), _T(","));
            for (const auto& str : list) {
                int nQuality = 0;
                if (1 == _stscanf(str.c_str(), _T("%d"), &nQuality)) {
                    pParams->nBenchQuality |= 1 << nQuality;
                } else {
                    PrintHelp(strInput[i], _T("Unknown value"), option_name);
                    return MFX_PRINT_OPTION_ERR;
                }
            }
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("perf-monitor"))) {
        if (strInput[i+1][0] == _T('-') || _tcslen(strInput[i+1]) == 0) {
            pParams->nPerfMonitorSelect = (int)PERF_MONITOR_ALL;
        } else {
            i++;
            auto items = split(strInput[i], _T(","));
            for (const auto& item : items) {
                int value = 0;
                if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_pref_monitor, item.c_str()))) {
                    PrintHelp(item.c_str(), _T("Unknown value"), option_name);
                    return MFX_PRINT_OPTION_ERR;
                }
                pParams->nPerfMonitorSelect |= value;
            }
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("perf-monitor-interval"))) {
        i++;
        mfxI32 v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nPerfMonitorInterval = std::max(50, v);
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("perf-monitor-plot"))) {
        if (strInput[i+1][0] == _T('-') || _tcslen(strInput[i+1]) == 0) {
            pParams->nPerfMonitorSelectMatplot =
                (int)(PERF_MONITOR_CPU | PERF_MONITOR_CPU_KERNEL
                    | PERF_MONITOR_THREAD_MAIN | PERF_MONITOR_THREAD_ENC | PERF_MONITOR_THREAD_OUT | PERF_MONITOR_THREAD_IN
                    | PERF_MONITOR_GPU_CLOCK | PERF_MONITOR_GPU_LOAD | PERF_MONITOR_MFX_LOAD
                    | PERF_MONITOR_FPS);
        } else {
            i++;
            auto items = split(strInput[i], _T(","));
            for (const auto& item : items) {
                int value = 0;
                if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_pref_monitor, item.c_str()))) {
                    PrintHelp(item.c_str(), _T("Unknown value"), option_name);
                    return MFX_PRINT_OPTION_ERR;
                }
                pParams->nPerfMonitorSelectMatplot |= value;
            }
        }
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("python"))) {
        i++;
        pParams->pPythonPath = _tcsdup(strInput[i]);
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("timer-period-tuning"))) {
        pParams->bDisableTimerPeriodTuning = false;
        return MFX_ERR_NONE;
    }
    if (0 == _tcscmp(option_name, _T("no-timer-period-tuning"))) {
        pParams->bDisableTimerPeriodTuning = true;
        return MFX_ERR_NONE;
    }
    tstring mes = _T("Unknown option: --");
    mes += option_name;
    PrintHelp(strInput[0], (TCHAR *)mes.c_str(), NULL, strInput[i]);
    return MFX_PRINT_OPTION_ERR;
}

mfxStatus ParseInputString(const TCHAR *strInput[], int nArgNum, sInputParams *pParams) {
    if (1 == nArgNum) {
        PrintHelp(strInput[0], NULL, NULL);
        PrintHelp(strInput[0], _T("options needed."), NULL);
        return MFX_PRINT_OPTION_ERR;
    }

    if (!pParams) {
        return MFX_ERR_NONE;
    }
    RGY_MEMSET_ZERO(*pParams);

    pParams->CodecId           = MFX_CODEC_AVC;
    pParams->nTargetUsage      = QSV_DEFAULT_QUALITY;
    pParams->nEncMode          = MFX_RATECONTROL_CQP;
    pParams->ColorFormat       = MFX_FOURCC_YV12;
    pParams->nPicStruct        = MFX_PICSTRUCT_PROGRESSIVE;
    pParams->nQPI              = QSV_DEFAULT_QPI;
    pParams->nQPP              = QSV_DEFAULT_QPP;
    pParams->nQPB              = QSV_DEFAULT_QPB;
    pParams->nRef              = QSV_DEFAULT_REF;
    pParams->bUseHWLib         = true;
#if defined(_WIN32) || defined(_WIN64)
    pParams->memType           = HW_MEMORY;
#else
    pParams->memType           = SYSTEM_MEMORY;
#endif
    pParams->nBframes          = QSV_BFRAMES_AUTO;
    pParams->bBPyramid         = getCPUGen() >= CPU_GEN_HASWELL;
    pParams->nGOPLength        = QSV_DEFAULT_GOP_LEN;
    pParams->ColorPrim         = (mfxU16)list_colorprim[0].value;
    pParams->ColorMatrix       = (mfxU16)list_colormatrix[0].value;
    pParams->Transfer          = (mfxU16)list_transfer[0].value;
    pParams->VideoFormat       = (mfxU16)list_videoformat[0].value;
    pParams->nInputBufSize     = QSV_DEFAULT_INPUT_BUF_HW;
    pParams->bforceGOPSettings = QSV_DEFAULT_FORCE_GOP_LEN;
    pParams->vpp.delogo.nDepth = QSV_DEFAULT_VPP_DELOGO_DEPTH;
    pParams->nSessionThreadPriority = (mfxU16)get_value_from_chr(list_priority, _T("normal"));
    pParams->nPerfMonitorInterval = QSV_DEFAULT_PERF_MONITOR_INTERVAL;
    pParams->nOutputBufSizeMB  = QSV_DEFAULT_OUTPUT_BUF_MB;
    pParams->nInputThread      = RGY_INPUT_THREAD_AUTO;
    pParams->nOutputThread     = RGY_OUTPUT_THREAD_AUTO;
    pParams->nAudioThread      = RGY_AUDIO_THREAD_AUTO;
    pParams->nBenchQuality     = QSV_DEFAULT_BENCH;
    pParams->nAudioIgnoreDecodeError = QSV_DEFAULT_AUDIO_IGNORE_DECODE_ERROR;

    sArgsData argsData;

    // parse command line parameters
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
                if (nullptr == (option_name = short_opt_to_long(strInput[i][1]))) {
                    PrintHelp(strInput[0], strsprintf(_T("Unknown options: \"%s\""), strInput[i]).c_str(), NULL, NULL);
                    return MFX_PRINT_OPTION_ERR;
                }
            } else {
                PrintHelp(strInput[0], strsprintf(_T("Invalid options: \"%s\""), strInput[i]).c_str(), NULL, NULL);
                return MFX_PRINT_OPTION_ERR;
            }
        }

        if (option_name == NULL) {
            PrintHelp(strInput[0], strsprintf(_T("Unknown option: \"%s\""), strInput[i]).c_str(), NULL, NULL);
            return MFX_PRINT_OPTION_ERR;
        }

        // process multi-character options
        if (0 == _tcscmp(option_name, _T("help")))
        {
            PrintHelp(strInput[0], NULL, NULL);
            return MFX_PRINT_OPTION_DONE;
        }
        if (0 == _tcscmp(option_name, _T("version")))
        {
            PrintVersion();
            return MFX_PRINT_OPTION_DONE;
        }

        if (0 == _tcscmp(option_name, _T("check-environment")))
        {
            PrintVersion();
            _ftprintf(stdout, _T("%s"), getEnviromentInfo(true).c_str());
            return MFX_PRINT_OPTION_DONE;
        }
        if (0 == _tcscmp(option_name, _T("check-features")))
        {
            tstring output = (strInput[i+1][0] != _T('-')) ? strInput[i+1] : _T("");
            writeFeatureList(output);
            return MFX_PRINT_OPTION_DONE;
        }
        if (0 == _tcscmp(option_name, _T("check-features-html")))
        {
            tstring output = (strInput[i+1][0] != _T('-')) ? strInput[i+1] : _T("");
            writeFeatureList(output, FEATURE_LIST_STR_TYPE_HTML);
            return MFX_PRINT_OPTION_DONE;
        }
        if (   0 == _tcscmp(option_name, _T("check-hw"))
            || 0 == _tcscmp(option_name, _T("hw-check"))) //互換性のため
        {
            mfxVersion ver = { 0, 1 };
            if (check_lib_version(get_mfx_libhw_version(), ver) != 0) {
                _ftprintf(stdout, _T("Success: QuickSyncVideo (hw encoding) available\n"));
                return MFX_PRINT_OPTION_DONE;
            } else {
                _ftprintf(stdout, _T("Error: QuickSyncVideo (hw encoding) unavailable\n"));
                return MFX_PRINT_OPTION_ERR;
            }
        }
        if (   0 == _tcscmp(option_name, _T("lib-check"))
            || 0 == _tcscmp(option_name, _T("check-lib")))
        {
            mfxVersion test = { 0, 1 };
            mfxVersion hwlib = get_mfx_libhw_version();
            mfxVersion swlib = get_mfx_libsw_version();
            PrintVersion();
#ifdef _M_IX86
            const TCHAR *dll_platform = _T("32");
#else
            const TCHAR *dll_platform = _T("64");
#endif
            if (check_lib_version(hwlib, test))
                _ftprintf(stdout, _T("libmfxhw%s.dll : v%d.%d\n"), dll_platform, hwlib.Major, hwlib.Minor);
            else
                _ftprintf(stdout, _T("libmfxhw%s.dll : ----\n"), dll_platform);
            if (check_lib_version(swlib, test))
                _ftprintf(stdout, _T("libmfxsw%s.dll : v%d.%d\n"), dll_platform, swlib.Major, swlib.Minor);
            else
                _ftprintf(stdout, _T("libmfxsw%s.dll : ----\n"), dll_platform);
            return MFX_PRINT_OPTION_DONE;
        }
#if ENABLE_AVSW_READER
        if (0 == _tcscmp(option_name, _T("check-avversion")))
        {
            _ftprintf(stdout, _T("%s\n"), getAVVersions().c_str());
            return MFX_PRINT_OPTION_DONE;
        }
        if (0 == _tcscmp(option_name, _T("check-codecs")))
        {
            _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC | RGY_AVCODEC_ENC)).c_str());
            return MFX_PRINT_OPTION_DONE;
        }
        if (0 == _tcscmp(option_name, _T("check-encoders")))
        {
            _ftprintf(stdout, _T("%s\n"), getAVCodecs(RGY_AVCODEC_ENC).c_str());
            return MFX_PRINT_OPTION_DONE;
        }
        if (0 == _tcscmp(option_name, _T("check-decoders")))
        {
            _ftprintf(stdout, _T("%s\n"), getAVCodecs(RGY_AVCODEC_DEC).c_str());
            return MFX_PRINT_OPTION_DONE;
        }
        if (0 == _tcscmp(option_name, _T("check-protocols")))
        {
            _ftprintf(stdout, _T("%s\n"), getAVProtocols().c_str());
            return MFX_PRINT_OPTION_DONE;
        }
        if (0 == _tcscmp(option_name, _T("check-filters")))
        {
            _ftprintf(stdout, _T("%s\n"), getAVFilters().c_str());
            return MFX_PRINT_OPTION_DONE;
        }
        if (0 == _tcscmp(option_name, _T("check-formats")))
        {
            _ftprintf(stdout, _T("%s\n"), getAVFormats((RGYAVFormatType)(RGY_AVFORMAT_DEMUX | RGY_AVFORMAT_MUX)).c_str());
            return MFX_PRINT_OPTION_DONE;
        }
#endif //ENABLE_AVSW_READER
        auto sts = ParseOneOption(option_name, strInput, i, nArgNum, pParams, &argsData);
        if (sts != MFX_ERR_NONE) {
            return sts;
        }
    }

    //parse cached profile and level
    if (argsData.cachedlevel.length() > 0) {
        const auto desc = get_level_list(pParams->CodecId);
        int value = 0;
        bool bParsed = false;
        if (desc != nullptr) {
            if (PARSE_ERROR_FLAG != (value = get_value_from_chr(desc, argsData.cachedlevel.c_str()))) {
                pParams->CodecLevel = (mfxU16)value;
                bParsed = true;
            } else {
                double val_float = 0.0;
                if (1 == _stscanf_s(argsData.cachedlevel.c_str(), _T("%lf"), &val_float)) {
                    value = (int)(val_float * 10 + 0.5);
                    if (value == desc[get_cx_index(desc, value)].value) {
                        pParams->CodecLevel = (mfxU16)value;
                        bParsed = true;
                    } else {
                        value = (int)(val_float + 0.5);
                        if (value == desc[get_cx_index(desc, value)].value) {
                            pParams->CodecLevel = (mfxU16)value;
                            bParsed = true;
                        }
                    }
                }
            }
        }
        if (!bParsed) {
            PrintHelp(strInput[0], _T("Unknown value"), _T("level"));
            return MFX_PRINT_OPTION_ERR;
        }
    }
    if (argsData.cachedprofile.length() > 0) {
        const auto desc = get_profile_list(pParams->CodecId);
        int value = 0;
        if (desc != nullptr && PARSE_ERROR_FLAG != (value = get_value_from_chr(desc, argsData.cachedprofile.c_str()))) {
            pParams->CodecProfile = (mfxU16)value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), _T("profile"));
            return MFX_PRINT_OPTION_ERR;
        }
    }

    // check if all mandatory parameters were set
    if (0 == _tcslen(pParams->strSrcFile)) {
        PrintHelp(strInput[0], _T("Source file name not found"), NULL);
        return MFX_PRINT_OPTION_ERR;
    }

    if (0 == _tcslen(pParams->strDstFile)) {
        PrintHelp(strInput[0], _T("Destination file name not found"), NULL);
        return MFX_PRINT_OPTION_ERR;
    }

    pParams->nTargetUsage = clamp(pParams->nTargetUsage, MFX_TARGETUSAGE_BEST_QUALITY, MFX_TARGETUSAGE_BEST_SPEED);

    // if nv12 option isn't specified, input YUV file is expected to be in YUV420 color format
    if (!pParams->ColorFormat) {
        pParams->ColorFormat = MFX_FOURCC_YV12;
    }

    //if picstruct not set, progressive frame is expected
    if (!pParams->nPicStruct) {
        pParams->nPicStruct = MFX_PICSTRUCT_PROGRESSIVE;
    }

    mfxVersion mfxlib_hw = get_mfx_libhw_version();
    mfxVersion mfxlib_sw = get_mfx_libsw_version();
    //check if dll exists
    if (pParams->bUseHWLib && (check_lib_version(mfxlib_hw, MFX_LIB_VERSION_1_1) == 0)) {
        PrintHelp(strInput[0], _T("QuickSyncVideo (hw encoding) unavailable"), NULL);
        return MFX_PRINT_OPTION_ERR;
    }

    if (!pParams->bUseHWLib && (check_lib_version(mfxlib_sw, MFX_LIB_VERSION_1_1) == 0)) {
#ifdef _M_IX86
        PrintHelp(strInput[0], _T("software encoding unavailable. Please Check for libmfxsw32.dll."), NULL);
#else
        PrintHelp(strInput[0], _T("software encoding unavailable. Please Check for libmfxsw64.dll."), NULL);
#endif
        return MFX_PRINT_OPTION_ERR;
    }

    //don't use d3d memory with software encoding
    if (!pParams->bUseHWLib) {
        pParams->memType = SYSTEM_MEMORY;
    }

    if (pParams->pChapterFile && pParams->bCopyChapter) {
        PrintHelp(strInput[0], _T("--chapter and --chapter-copy are both set.\nThese could not be set at the same time."), NULL);
        return MFX_PRINT_OPTION_ERR;
    }

    //set input buffer size
    if (argsData.nTmpInputBuf == 0) {
        argsData.nTmpInputBuf = (pParams->bUseHWLib) ? QSV_DEFAULT_INPUT_BUF_HW : QSV_DEFAULT_INPUT_BUF_SW;
    }
    pParams->nInputBufSize = (mfxU16)clamp(argsData.nTmpInputBuf, QSV_INPUT_BUF_MIN, QSV_INPUT_BUF_MAX);

    if (pParams->nRotationAngle != 0 && pParams->nRotationAngle != 180) {
        PrintHelp(strInput[0], _T("Angles other than 180 degrees are not supported."), NULL);
        return MFX_PRINT_OPTION_ERR; // other than 180 are not supported 
    }

    // not all options are supported if rotate plugin is enabled
    if (pParams->nRotationAngle == 180) {
        if (MFX_FOURCC_NV12 != pParams->ColorFormat) {
            PrintHelp(strInput[0], _T("Rotation plugin requires NV12 input. Please specify -nv12 option."), NULL);
            return MFX_PRINT_OPTION_ERR;
        }
        pParams->nPicStruct = MFX_PICSTRUCT_PROGRESSIVE;
        pParams->nDstWidth = pParams->nWidth;
        pParams->nDstHeight = pParams->nHeight;
        pParams->memType = SYSTEM_MEMORY;
    }

    return MFX_ERR_NONE;
}

//Ctrl + C ハンドラ
static bool g_signal_abort = false;
#pragma warning(push)
#pragma warning(disable:4100)
static void sigcatch(int sig) {
    g_signal_abort = true;
}
#pragma warning(pop)
static int set_signal_handler() {
    int ret = 0;
    if (SIG_ERR == signal(SIGINT, sigcatch)) {
        _ftprintf(stderr, _T("failed to set signal handler.\n"));
    }
    return ret;
}

int run_encode(sInputParams *params) {
    mfxStatus sts = MFX_ERR_NONE; // return value check

    unique_ptr<CQSVPipeline> pPipeline(new CQSVPipeline);
    if (!pPipeline) {
        return MFX_ERR_MEMORY_ALLOC;
    }

    sts = pPipeline->Init(params);
    if (sts < MFX_ERR_NONE) return sts;

    if (params->pStrLogFile) {
        free(params->pStrLogFile);
        params->pStrLogFile = NULL;
    }

    pPipeline->SetAbortFlagPointer(&g_signal_abort);
    set_signal_handler();

    if (MFX_ERR_NONE != (sts = pPipeline->CheckCurrentVideoParam())) {
        return sts;
    }

    for (;;) {
        sts = pPipeline->Run();

        if (MFX_ERR_DEVICE_LOST == sts || MFX_ERR_DEVICE_FAILED == sts) {
            _ftprintf(stderr, _T("\nERROR: Hardware device was lost or returned an unexpected error. Recovering...\n"));
            sts = pPipeline->ResetDevice();
            if (sts < MFX_ERR_NONE) return sts;

            sts = pPipeline->ResetMFXComponents(params);
            if (sts < MFX_ERR_NONE) return sts;
            continue;
        } else {
            if (sts < MFX_ERR_NONE) return sts;
            break;
        }
    }

    pPipeline->Close();

    return sts;
}

mfxStatus run_benchmark(sInputParams *params) {
    using namespace std;
    mfxStatus sts = MFX_ERR_NONE;
    basic_string<TCHAR> benchmarkLogFile = params->strDstFile;

    //テストする解像度
    const vector<pair<mfxU16, mfxU16>> test_resolution = { { 1920, 1080 }, { 1280, 720 } };

    //初回出力
    {
        params->nDstWidth = test_resolution[0].first;
        params->nDstHeight = test_resolution[0].second;
        params->nTargetUsage = MFX_TARGETUSAGE_BEST_SPEED;

        unique_ptr<CQSVPipeline> pPipeline(new CQSVPipeline);
        if (!pPipeline) {
            return MFX_ERR_MEMORY_ALLOC;
        }

        sts = pPipeline->Init(params);
        if (sts < MFX_ERR_NONE) return sts;

        pPipeline->SetAbortFlagPointer(&g_signal_abort);
        set_signal_handler();
        time_t current_time = time(NULL);
        struct tm *local_time = localtime(&current_time);

        TCHAR encode_info[4096] = { 0 };
        if (MFX_ERR_NONE != (sts = pPipeline->CheckCurrentVideoParam(encode_info, _countof(encode_info)))) {
            return sts;
        }

        bool hardware;
        mfxVersion ver;
        pPipeline->GetEncodeLibInfo(&ver, &hardware);

        auto enviroment_info = getEnviromentInfo();

        MemType memtype = pPipeline->GetMemType();

        basic_stringstream<TCHAR> ss;
        FILE *fp_bench = NULL;
        if (_tfopen_s(&fp_bench, benchmarkLogFile.c_str(), _T("a")) || NULL == fp_bench) {
            pPipeline->PrintMes(RGY_LOG_ERROR, _T("\nERROR: failed opening benchmark result file.\n"));
            return MFX_ERR_INVALID_HANDLE;
        } else {
            fprintf(fp_bench, "Started benchmark on %d.%02d.%02d %2d:%02d:%02d\n",
                1900 + local_time->tm_year, local_time->tm_mon + 1, local_time->tm_mday, local_time->tm_hour, local_time->tm_min, local_time->tm_sec);
            fprintf(fp_bench, "Input File: %s\n", tchar_to_string(params->strSrcFile).c_str());
            fprintf(fp_bench, "Basic parameters of the benchmark\n"
                              " (Target Usage and output resolution will be changed)\n");
            fprintf(fp_bench, "%s\n\n", tchar_to_string(encode_info).c_str());
            fprintf(fp_bench, "%s", tchar_to_string(enviroment_info).c_str());
            fprintf(fp_bench, "QSV: QSVEncC %s (%s) / API[%s]: v%d.%d / %s\n", 
                VER_STR_FILEVERSION, tchar_to_string(BUILD_ARCH_STR).c_str(), (hardware) ? "hw" : "sw", ver.Major, ver.Minor, tchar_to_string(MemTypeToStr(memtype)).c_str());
            fprintf(fp_bench, "\n");
            fclose(fp_bench);
        }
        basic_ofstream<TCHAR> benchmark_log_test_open(benchmarkLogFile, ios::out | ios::app);
        if (!benchmark_log_test_open.good()) {
            pPipeline->PrintMes(RGY_LOG_ERROR, _T("\nERROR: failed opening benchmark result file.\n"));
            return MFX_ERR_INVALID_HANDLE;
        }
        benchmark_log_test_open << ss.str();
        benchmark_log_test_open.close();

        for (;;) {
            sts = pPipeline->Run();

            if (MFX_ERR_DEVICE_LOST == sts || MFX_ERR_DEVICE_FAILED == sts) {
                pPipeline->PrintMes(RGY_LOG_ERROR, _T("\nERROR: Hardware device was lost or returned an unexpected error. Recovering...\n"));
                if (   MFX_ERR_NONE != (sts = pPipeline->ResetDevice())
                    || MFX_ERR_NONE != (sts = pPipeline->ResetMFXComponents(params)))
                    break;
            } else {
                break;
            }
        }

        EncodeStatusData data = { 0 };
        sts = pPipeline->GetEncodeStatusData(&data);

        pPipeline->Close();
    }

    //ベンチマークの集計データ
    typedef struct benchmark_t {
        pair<mfxU16, mfxU16> resolution;
        mfxU16 targetUsage;
        double fps;
        double bitrate;
        double cpuUsagePercent;
    } benchmark_t;

    //対象
    vector<CX_DESC> list_target_quality;
    for (uint32_t i = 0; i < _countof(list_quality); i++) {
        if (list_quality[i].desc) {
            int test = 1 << list_quality[i].value;
            if (params->nBenchQuality & test) {
                list_target_quality.push_back(list_quality[i]);
            }
        }
    }

    //解像度ごとに、target usageを変化させて測定
    vector<vector<benchmark_t>> benchmark_result;
    benchmark_result.reserve(test_resolution.size() * list_target_quality.size());

    for (uint32_t i = 0; MFX_ERR_NONE == sts && !g_signal_abort && i < list_target_quality.size(); i++) {
        params->nTargetUsage = (mfxU16)list_target_quality[i].value;
        vector<benchmark_t> benchmark_per_target_usage;
        for (const auto& resolution : test_resolution) {
            params->nDstWidth = resolution.first;
            params->nDstHeight = resolution.second;

            unique_ptr<CQSVPipeline> pPipeline(new CQSVPipeline);
            if (!pPipeline) {
                return MFX_ERR_MEMORY_ALLOC;
            }

            if (MFX_ERR_NONE != (sts = pPipeline->Init(params))) {
                break;
            }

            pPipeline->SetAbortFlagPointer(&g_signal_abort);
            set_signal_handler();
            if (MFX_ERR_NONE != (sts = pPipeline->CheckCurrentVideoParam())) {
                return sts;
            }

            for (;;) {
                sts = pPipeline->Run();

                if (MFX_ERR_DEVICE_LOST == sts || MFX_ERR_DEVICE_FAILED == sts) {
                    pPipeline->PrintMes(RGY_LOG_ERROR, _T("\nERROR: Hardware device was lost or returned an unexpected error. Recovering...\n"));
                    if (   MFX_ERR_NONE != (sts = pPipeline->ResetDevice())
                        || MFX_ERR_NONE != (sts = pPipeline->ResetMFXComponents(params)))
                        break;
                } else {
                    break;
                }
            }

            EncodeStatusData data = { 0 };
            sts = pPipeline->GetEncodeStatusData(&data);

            pPipeline->Close();

            benchmark_t result;
            result.resolution      = resolution;
            result.targetUsage     = (mfxU16)list_target_quality[i].value;
            result.fps             = data.encodeFps;
            result.bitrate         = data.bitrateKbps;
            result.cpuUsagePercent = data.CPUUsagePercent;
            benchmark_per_target_usage.push_back(result);

            _ftprintf(stderr, _T("\n"));

            if (MFX_ERR_NONE != sts || g_signal_abort)
                break;
        }

        benchmark_result.push_back(benchmark_per_target_usage);
    }

    //結果を出力
    if (MFX_ERR_NONE == sts && benchmark_result.size()) {
        basic_stringstream<TCHAR> ss;

        uint32_t maxLengthOfTargetUsageDesc = 0;
        for (uint32_t i = 0; i < list_target_quality.size(); i++) {
            maxLengthOfTargetUsageDesc = max(maxLengthOfTargetUsageDesc, (uint32_t)_tcslen(list_target_quality[i].desc));
        }

        FILE *fp_bench = NULL;
        if (_tfopen_s(&fp_bench, benchmarkLogFile.c_str(), _T("a")) || NULL == fp_bench) {
            _ftprintf(stderr, _T("\nERROR: failed opening benchmark result file.\n"));
            return MFX_ERR_INVALID_HANDLE;
        } else {
            fprintf(fp_bench, "TargetUsage ([TU-1]:Best Quality) ～ ([TU-7]:Fastest Speed)\n\n");

            fprintf(fp_bench, "Encode Speed (fps)\n");
            fprintf(fp_bench, "TargetUsage");
            for (const auto& resolution : test_resolution) {
                fprintf(fp_bench, ",   %dx%d", resolution.first, resolution.second);
            }
            fprintf(fp_bench, "\n");

            for (const auto &benchmark_per_target_usage : benchmark_result) {
                fprintf(fp_bench, " 　　TU-%d", benchmark_per_target_usage[0].targetUsage);
                for (const auto &result : benchmark_per_target_usage) {
                    fprintf(fp_bench, ",　　　%6.2f", result.fps);
                }
                fprintf(fp_bench, "\n");
            }
            fprintf(fp_bench, "\n");

            fprintf(fp_bench, "Bitrate (kbps)\n");
            fprintf(fp_bench, "TargetUsage");
            for (const auto& resolution : test_resolution) {
                fprintf(fp_bench, ",   %dx%d", resolution.first, resolution.second);
            }
            fprintf(fp_bench, "\n");
            for (const auto &benchmark_per_target_usage : benchmark_result) {
                fprintf(fp_bench, " 　　TU-%d", benchmark_per_target_usage[0].targetUsage);
                for (const auto &result : benchmark_per_target_usage) {
                    fprintf(fp_bench, ",　　　%6d", (int)(result.bitrate + 0.5));
                }
                fprintf(fp_bench, "\n");
            }
            fprintf(fp_bench, "\n");

            fprintf(fp_bench, "CPU Usage (%%)\n");
            fprintf(fp_bench, "TargetUsage");
            for (const auto& resolution : test_resolution) {
                fprintf(fp_bench, ",   %dx%d", resolution.first, resolution.second);
            }
            fprintf(fp_bench, "\n");
            for (const auto &benchmark_per_target_usage : benchmark_result) {
                fprintf(fp_bench, " 　　TU-%d", benchmark_per_target_usage[0].targetUsage);
                for (const auto &result : benchmark_per_target_usage) {
                    fprintf(fp_bench, ",　　　%6.2f", result.cpuUsagePercent);
                }
                fprintf(fp_bench, "\n");
            }
            fprintf(fp_bench, "\n");
            fclose(fp_bench);
            _ftprintf(stderr, _T("\nFinished benchmark.\n"));
        }
    } else {
        rgy_print_stderr(RGY_LOG_ERROR, _T("\nError occurred during benchmark.\n"));
    }

    return sts;
}

int run(int argc, TCHAR *argv[]) {
    mfxStatus sts = MFX_ERR_NONE;
    sInputParams Params = { 0 };

    vector<const TCHAR *> argvCopy(argv, argv + argc);
    argvCopy.push_back(_T(""));

    sts = ParseInputString(argvCopy.data(), (mfxU8)argc, &Params);
    if (sts >= MFX_PRINT_OPTION_DONE)
        return 0;

#if defined(_WIN32) || defined(_WIN64)
    //set stdin to binary mode when using pipe input
    if (_tcscmp(Params.strSrcFile, _T("-")) == NULL) {
        if (_setmode( _fileno( stdin ), _O_BINARY ) == 1) {
            PrintHelp(argv[0], _T("failed to switch stdin to binary mode."), NULL);
            return 1;
        }
    }

    //set stdout to binary mode when using pipe output
    if (_tcscmp(Params.strDstFile, _T("-")) == NULL) {
        if (_setmode( _fileno( stdout ), _O_BINARY ) == 1) {
            PrintHelp(argv[0], _T("failed to switch stdout to binary mode."), NULL);
            return 1;
        }
    }

    if (check_locale_is_ja()) {
        _tsetlocale(LC_ALL, _T("Japanese"));
    }
#endif //#if defined(_WIN32) || defined(_WIN64)

    if (Params.bBenchmark) {
        return run_benchmark(&Params);
    }

    unique_ptr<CQSVPipeline> pPipeline(new CQSVPipeline);
    if (!pPipeline) {
        return MFX_ERR_MEMORY_ALLOC;
    }

    sts = pPipeline->Init(&Params);
    if (sts < MFX_ERR_NONE) return 1;

    pPipeline->SetAbortFlagPointer(&g_signal_abort);
    set_signal_handler();
    
    if (MFX_ERR_NONE != (sts = pPipeline->CheckCurrentVideoParam())) {
        return sts;
    }

    for (;;) {
        sts = pPipeline->Run();

        if (MFX_ERR_DEVICE_LOST == sts || MFX_ERR_DEVICE_FAILED == sts) {
            pPipeline->PrintMes(RGY_LOG_ERROR, _T("\nERROR: Hardware device was lost or returned an unexpected error. Recovering...\n"));
            sts = pPipeline->ResetDevice();
            if (sts < MFX_ERR_NONE) return sts;

            sts = pPipeline->ResetMFXComponents(&Params);
            if (sts < MFX_ERR_NONE) return sts;
            continue;
        } else {
            if (sts < MFX_ERR_NONE) return 1;
            break;
        }
    }

    pPipeline->Close();
    pPipeline->PrintMes(RGY_LOG_INFO, _T("\nProcessing finished\n"));

    return sts;
}

int _tmain(int argc, TCHAR *argv[]) {
    int ret = 0;
    if (0 != (ret = run(argc, argv))) {
        rgy_print_stderr(RGY_LOG_ERROR, _T("QSVEncC.exe finished with error!\n"));
    }
#if ENABLE_AVSW_READER
    avformatNetworkDeinit();
#endif //#if ENABLE_AVSW_READER
    return ret;
}
