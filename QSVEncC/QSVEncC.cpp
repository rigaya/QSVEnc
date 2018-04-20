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
#include "qsv_cmd.h"
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

static void show_version() {
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

static tstring help() {
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
        _T("                                 - h264(default), hevc, mpeg2, raw\n")
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
        _T("   --avsw                       set input to use avcodec + sw decoder\n")
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
        _T("       usable symbols\n")
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
        _T("   --chapter-no-trim            do not apply trim to chapter file.\n")
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
        QSV_DEFAULT_AUDIO_IGNORE_DECODE_ERROR
#endif
    );
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
        _T(" On particular cases, such as runnning on a system with dGPU, or running\n")
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
        _T("   --(no-)weightp               enable weighted prediction for P frame\n")
        _T("   --(no-)weightb               enable weighted prediction for B frame\n")
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
        _T("\n"));
    str += PrintListOptions(_T("--videoformat <string>"), list_videoformat, 0);
    str += PrintListOptions(_T("--colormatrix <string>"), list_colormatrix, 0);
    str += PrintListOptions(_T("--colorprim <string>"),   list_colorprim,   0);
    str += PrintListOptions(_T("--transfer <string>"),    list_transfer,    0);
    str += strsprintf(_T("")
        _T("   --aud                        insert aud nal unit to ouput stream.\n")
        _T("   --pic-struct                 insert pic-timing SEI with pic_struct.\n")
        _T("   --fullrange                  set stream as fullrange yuv\n")
        _T("   --max-cll <int>,<int>        set MaxCLL and MaxFall in nits. e.g. \"1000,300\"\n")
        _T("   --master-display <string>    set Mastering display data.\n")
        _T("      e.g. \"G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)\"\n"));
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

static void show_help() {
    _ftprintf(stdout, _T("%s\n"), help().c_str());
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

static void PrintHelp(tstring strAppName, tstring strErrorMessage, tstring strOptionName, tstring strErrorValue) {
    if (strErrorMessage.length() > 0) {
        if (strOptionName.length() > 0) {
            if (strErrorValue.length() > 0) {
                _ftprintf(stderr, _T("Error: %s \"%s\" for \"--%s\"\n"), strErrorMessage.c_str(), strErrorValue.c_str(), strOptionName.c_str());
                if (0 == _tcsnccmp(strErrorValue.c_str(), _T("--"), _tcslen(_T("--")))
                    || (strErrorValue[0] == _T('-') && strErrorValue[2] == _T('\0') && cmd_short_opt_to_long(strErrorValue[1]) != nullptr)) {
                    _ftprintf(stderr, _T("       \"--%s\" requires value.\n\n"), strOptionName.c_str());
                }
            } else {
                _ftprintf(stderr, _T("Error: %s for --%s\n\n"), strErrorMessage.c_str(), strOptionName.c_str());
            }
        } else {
            _ftprintf(stderr, _T("Error: %s\n\n"), strErrorMessage.c_str());
#if (ENABLE_CPP_REGEX && ENABLE_DTL)
            if (strErrorValue.length() > 0) {
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
        show_version();
        show_help();
    }
}

static int writeFeatureList(tstring filename, bool for_auo, FeatureListStrType type = FEATURE_LIST_STR_TYPE_UNKNOWN) {
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
            const auto codec_feature_list = (for_auo) ? MakeFeatureListStr(0 == impl_type, type, make_vector(CODEC_LIST_AUO)) : MakeFeatureListStr(0 == impl_type, type);
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
                if (!for_auo) {
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
                    i++;

                    const auto decHeader = tstring((bUseJapanese) ? _T("利用可能なHWデコーダ") : _T("Supported Decode features:\n"));
                    const auto decFeatures = MakeDecFeatureStr(0 == impl_type, type);
                    if (type == FEATURE_LIST_STR_TYPE_HTML) {
                        tstring str;
                        str += strsprintf(_T("<div class=table_block id=\"TableOpen%d\">\n"), i);
                        str += _T("<p class=table_block>\n");
                        str += strsprintf(_T("<a class=dec_table_block href=\"#\" title=\"%s▼\" onclick=\"showTable(%d);return false;\">%s▼</a>"), decHeader.c_str(), i, decHeader.c_str());
                        str += _T("</p>\n");
                        str += _T("</div>\n");
                        str += strsprintf(_T("<div class=table_block id=\"TableClose%d\" style=\"display: none\">\n"), i);
                        str += _T("<p class=table_block>\n");
                        str += strsprintf(_T("<a class=dec_table_block href=\"#\" title=\"%s▲\" onclick=\"showTable(%d);return false;\">%s▲</a>\n"), decHeader.c_str(), i, decHeader.c_str());
                        str += _T("</p>\n");
                        str += decFeatures;
                        str += _T("</div><br>\n");
                        print_tstring(str, false);
                    } else {
                        print_tstring(decHeader + _T("\n"), true);
                        print_tstring(strsprintf(_T("%s\n\n"), decFeatures.c_str()), false);
                    }
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


int parse_print_options(const TCHAR *option_name, const TCHAR *arg1) {

    // process multi-character options
    if (0 == _tcscmp(option_name, _T("help"))) {
        show_version();
        show_help();
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("version"))) {
        show_version();
        return 1;
    }

    if (0 == _tcscmp(option_name, _T("check-environment"))) {
        show_version();
        _ftprintf(stdout, _T("%s"), getEnviromentInfo(true).c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-features"))) {
        tstring output = (arg1[0] != _T('-')) ? arg1 : _T("");
        writeFeatureList(output, false);
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-features-auo"))) {
        writeFeatureList(_T(""), true);
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-features-html"))) {
        tstring output = (arg1[0] != _T('-')) ? arg1 : _T("");
        writeFeatureList(output, false, FEATURE_LIST_STR_TYPE_HTML);
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-hw"))
        || 0 == _tcscmp(option_name, _T("hw-check"))) //互換性のため
    {
        mfxVersion ver = { 0, 1 };
        if (check_lib_version(get_mfx_libhw_version(), ver) != 0) {
            _ftprintf(stdout, _T("Success: QuickSyncVideo (hw encoding) available\n"));
            exit(0);
        } else {
            _ftprintf(stdout, _T("Error: QuickSyncVideo (hw encoding) unavailable\n"));
            exit(1);
        }
    }
    if (0 == _tcscmp(option_name, _T("lib-check"))
        || 0 == _tcscmp(option_name, _T("check-lib"))) {
        mfxVersion test = { 0, 1 };
        mfxVersion hwlib = get_mfx_libhw_version();
        mfxVersion swlib = get_mfx_libsw_version();
        show_version();
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
        return 1;
    }
#if ENABLE_AVSW_READER
    if (0 == _tcscmp(option_name, _T("check-avversion"))) {
        _ftprintf(stdout, _T("%s\n"), getAVVersions().c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-codecs"))) {
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC | RGY_AVCODEC_ENC)).c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-encoders"))) {
        _ftprintf(stdout, _T("%s\n"), getAVCodecs(RGY_AVCODEC_ENC).c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-decoders"))) {
        _ftprintf(stdout, _T("%s\n"), getAVCodecs(RGY_AVCODEC_DEC).c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-protocols"))) {
        _ftprintf(stdout, _T("%s\n"), getAVProtocols().c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-filters"))) {
        _ftprintf(stdout, _T("%s\n"), getAVFilters().c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-formats"))) {
        _ftprintf(stdout, _T("%s\n"), getAVFormats((RGYAVFormatType)(RGY_AVFORMAT_DEMUX | RGY_AVFORMAT_MUX)).c_str());
        return 1;
    }
#endif //ENABLE_AVSW_READER
    return 0;
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
#if defined(_WIN32) || defined(_WIN64)
    if (check_locale_is_ja()) {
        _tsetlocale(LC_ALL, _T("Japanese"));
    }
#endif //#if defined(_WIN32) || defined(_WIN64)
    
    if (argc == 1) {
        show_version();
        show_help();
        return 1;
    }

    for (int iarg = 1; iarg < argc; iarg++) {
        const TCHAR *option_name = nullptr;
        if (argv[iarg][0] == _T('-')) {
            if (argv[iarg][1] == _T('\0')) {
                continue;
            } else if (argv[iarg][1] == _T('-')) {
                option_name = &argv[iarg][2];
            } else if (argv[iarg][2] == _T('\0')) {
                if (nullptr == (option_name = cmd_short_opt_to_long(argv[iarg][1]))) {
                    PrintHelp(argv[0], strsprintf(_T("Unknown options: \"%s\""), argv[iarg]).c_str(), _T(""), _T(""));
                    return 1;
                }
            }
        }
        if (option_name != nullptr) {
            int ret = parse_print_options(option_name, (iarg+1 < argc) ? argv[iarg+1] : _T(""));
            if (ret != 0) {
                return ret == 1 ? 0 : 1;
            }
        }
    }

    sInputParams Params = { 0 };
    init_qsvp_prm(&Params);

    vector<const TCHAR *> argvCopy(argv, argv + argc);
    argvCopy.push_back(_T(""));

    ParseCmdError err;
    int ret = parse_cmd(&Params, argvCopy.data(), (mfxU8)argc, err);
    if (ret >= 1) {
        PrintHelp(err.strAppName, err.strErrorMessage, err.strOptionName, err.strErrorValue);
        return 0;
    }

#if defined(_WIN32) || defined(_WIN64)
    //set stdin to binary mode when using pipe input
    if (_tcscmp(Params.strSrcFile, _T("-")) == NULL) {
        if (_setmode( _fileno( stdin ), _O_BINARY ) == 1) {
            PrintHelp(argv[0], _T("failed to switch stdin to binary mode."), _T(""), _T(""));
            return 1;
        }
    }

    //set stdout to binary mode when using pipe output
    if (_tcscmp(Params.strDstFile, _T("-")) == NULL) {
        if (_setmode( _fileno( stdout ), _O_BINARY ) == 1) {
            PrintHelp(argv[0], _T("failed to switch stdout to binary mode."), _T(""), _T(""));
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

    auto sts = pPipeline->Init(&Params);
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
