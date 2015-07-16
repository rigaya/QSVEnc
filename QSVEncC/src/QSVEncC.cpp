//
//               INTEL CORPORATION PROPRIETARY INFORMATION
//  This software is supplied under the terms of a license agreement or
//  nondisclosure agreement with Intel Corporation and may not be copied
//  or disclosed except in accordance with the terms of that agreement.
//        Copyright (c) 2005-2010 Intel Corporation. All Rights Reserved.
//

//  -----------------------------------------------------------------------------------------
//    QSVEncC
//      modified from sample_encode.cpp by rigaya 
//  -----------------------------------------------------------------------------------------

#include <io.h>
#include <fcntl.h>
#include <Math.h>
#include <signal.h>
#include <fstream>
#include <iomanip>
#include <set>
#include <vector>
#include <algorithm>
#include "shlwapi.h"
#pragma comment(lib, "shlwapi.lib")

#include "pipeline_encode.h"
#include "qsv_prm.h"
#include "qsv_version.h"
#include "avcodec_qsv.h"

#if ENABLE_AVCODEC_QSV_READER
tstring getAVQSVSupportedCodecList();
#endif

static void PrintVersion() {
    static const TCHAR *const ENABLED_INFO[] = { _T("disabled"), _T("enabled") };
#ifdef _M_IX86
    _ftprintf(stdout, _T("QSVEncC (x86) %s by rigaya, build %s %s\n"), VER_STR_FILEVERSION_TCHAR, _T(__DATE__), _T(__TIME__));
#else
    _ftprintf(stdout, _T("QSVEncC (x64) %s by rigaya, build %s %s\n"), VER_STR_FILEVERSION_TCHAR, _T(__DATE__), _T(__TIME__));
#endif
    _ftprintf(stdout, _T("based on Intel(R) Media SDK Encoding Sample %s\n"), MSDK_SAMPLE_VERSION);
    _ftprintf(stdout, _T("  avi reader:   %s\n"), ENABLED_INFO[!!ENABLE_AVI_READER]);
    _ftprintf(stdout, _T("  avs reader:   %s\n"), ENABLED_INFO[!!ENABLE_AVISYNTH_READER]);
    _ftprintf(stdout, _T("  vpy reader:   %s\n"), ENABLED_INFO[!!ENABLE_VAPOURSYNTH_READER]);
    _ftprintf(stdout, _T("  avqsv reader: %s"),   ENABLED_INFO[!!ENABLE_AVCODEC_QSV_READER]);
#if ENABLE_AVCODEC_QSV_READER
    _ftprintf(stdout, _T(" [%s]"), getAVQSVSupportedCodecList().c_str());
#endif
    _ftprintf(stdout, _T("\n\n"));
}

//適当に改行しながら表示する
static void PrintListOptions(FILE *fp, const TCHAR *option_name, const CX_DESC *list, int default_index) {
    const TCHAR *indent_space = _T("                                  ");
    const int indent_len = (int)_tcslen(indent_space);
    const int max_len = 77;
    int print_len = _ftprintf(fp, _T("   %s "), option_name);
    while (print_len < indent_len)
         print_len += _ftprintf(stdout, _T(" "));
    for (int i = 0; list[i].desc; i++) {
        if (print_len + _tcslen(list[i].desc) + _tcslen(_T(", ")) >= max_len) {
            _ftprintf(fp, _T("\n%s"), indent_space);
            print_len = indent_len;
        } else {
            if (i)
                print_len += _ftprintf(fp, _T(", "));
        }
        print_len += _ftprintf(fp, _T("%s"), list[i].desc);
    }
    _ftprintf(fp, _T("\n%s default: %s\n"), indent_space, list[default_index].desc);
}

static void PrintHelp(const TCHAR *strAppName, const TCHAR *strErrorMessage, const TCHAR *strOptionName)
{
    if (strErrorMessage)
    {
        if (strOptionName)
            _ftprintf(stderr, _T("Error: %s for %s\n\n"), strErrorMessage, strOptionName);
        else
            _ftprintf(stderr, _T("Error: %s\n\n"), strErrorMessage);
    }
    else
    {
        PrintVersion();

        _ftprintf(stdout, _T("Usage: %s [Options] -i <filename> -o <filename>\n"), PathFindFileName(strAppName));
        _ftprintf(stdout, _T("\n")
            _T("input can be %s%s%sraw YUV or YUV4MPEG2(y4m) format.\n")
            _T("when raw(default), fps, input-res are also necessary.\n")
            _T("\n")
            _T("output format will be raw H.264/AVC ES.\n")
            _T("when output filename is set to \"-\", H.264/AVC ES output is thrown to stdout.\n")
            _T("\n")
            _T("Example:\n")
            _T("  QSVEncC -i \"<avsfilename>\" -o \"<outfilename>\"\n")
            _T("  avs2pipemod -y4mp \"<avsfile>\" | QSVEncC --y4m -i - -o \"<outfilename>\"\n")
            _T("\n")
            _T("Example for Benchmark:\n")
            _T("  QSVEncC -i \"<avsfilename>\" --benchmark \"<benchmark_result.txt>\"\n")
            _T("\n")
            _T("Options: \n")
            _T("-h,-? --help                    show help\n")
            _T("-v,--version                    show version info\n")
            _T("\n")
            _T("-c,--codec <string>             set encode codec\n")
            _T("                                 - h264(default), hevc, mpeg2\n")
            _T("-i,--input-file <filename>      set input file name\n")
            _T("-o,--output-file <filename>     set ouput file name\n")
#if ENABLE_AVCODEC_QSV_READER
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
#if ENABLE_AVCODEC_QSV_READER
            _T("   --avqsv                      set input to use avcodec + qsv\n")
            _T("   --avqsv-analyze <int>        set time in sec which reader analyze input file.\n")
            _T("                                 default: 5.\n")
            _T("                                 could be only used with avqsv reader.\n")
            _T("                                 use if reader fails to detect audio stream.\n")
            _T("   --audio-file [<int>?][<string>:]<string>\n")
            _T("                                extract audio into file.\n")
            _T("                                 could be only used with avqsv reader.\n")
            _T("                                 below are optional,\n")
            _T("                                  in [<int>?], specify track number to extract.\n")
            _T("                                  in [<string>?], specify output format.\n")
            _T("   --trim <int>:<int>[,<int>:<int>]...\n")
            _T("                                trim video for the frame range specified.\n")
            _T("                                 frame range should not overwrap each other.\n")
            _T("                                 could be only used with avqsv reader.\n")
            _T("-f,--format <string>            set output format of output file.\n")
            _T("                                 if format is not specified, output format will\n")
            _T("                                 be guessed from output file extension.\n")
            _T("                                 set \"raw\" for H.264/ES output.\n")
            _T("   --copy-audio [<int>[,...]]   mux audio with video during output.\n")
            _T("                                 could be only used with\n")
            _T("                                 avqsv reader and avcodec muxer.\n")
            _T("                                 by default copies all audio tracks.\n")
            _T("                                 \"--copy-audio 1,2\" will extract\n")
            _T("                                 audio track #1 and #2.\n")
#endif
            _T("\n")
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
            _T("   --crop <int>,<int>,<int>,<int>\n")
            _T("                                set crop pixels of left, up, right, bottom.\n")
            _T("\n")
            _T("   --slices <int>               number of slices, default 0 (auto)\n")
            _T("\n")
            _T("   --sw                         use software encoding, instead of QSV (hw)\n")
            _T("   --check-hw                   check if QuickSyncVideo is available\n")
            _T("   --check-lib                  check lib API version installed\n")
            _T("   --check-features             check encode features\n")
            _T("   --check-environment          check environment info\n")
#if ENABLE_AVCODEC_QSV_READER
            _T("   --check-codecs               show codecs available\n")
            _T("   --check-encoders             show audio encoders available\n")
            _T("   --check-decoders             show audio decoders available\n")
            _T("   --check-formats              show in/out formats available\n")
#endif
            ,
            (ENABLE_AVI_READER)         ? _T("avi, ") : _T(""),
            (ENABLE_AVISYNTH_READER)    ? _T("avs, ") : _T(""),
            (ENABLE_VAPOURSYNTH_READER) ? _T("vpy, ") : _T(""));
#ifdef D3D_SURFACES_SUPPORT
        _ftprintf(stdout, _T("")
            _T("   --disable-d3d                disable using d3d surfaces\n"));
#if MFX_D3D11_SUPPORT
        _ftprintf(stdout, _T("")
            _T("   --d3d                        use d3d9/d3d11 surfaces\n")
            _T("   --d3d9                       use d3d9 surfaces\n")
            _T("   --d3d11                      use d3d11 surfaces\n"));
#else
        _ftprintf(stdout, _T("")
            _T("   --d3d                        use d3d9 surfaces\n"));
#endif //MFX_D3D11_SUPPORT
#endif //D3D_SURFACES_SUPPORT
        _ftprintf(stdout,_T("\n")
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
            _T("   --qvbr <int>                 set bitrate in Quality VBR mode.\n")
            _T("   --qvbr-q <int>  or           set quality used in qvbr mode. default: %d\n")
            _T("   --qvbr-quality <int>          QVBR mode is only supported with API v1.11\n")
            _T("   --vcm <int>                  set bitrate in VCM mode (kbps)\n")
            //_T("   --avbr-range <float>           avbr accuracy range from bitrate set\n)"
            //_T("                                   in percentage, defalut %.1f(%%)\n)"
            _T("\n")
            _T("   --la-depth <int>             set Lookahead Depth, %d-%d\n")
            _T("   --la-window-size <int>       enables Lookahead Windowed Rate Control mode,\n")
            _T("                                  and set the window size in frames.\n")
            _T("   --max-bitrate <int>          set max bitrate(kbps)\n")
            _T("-u,--quality <string>           encode quality\n")
            _T("                                  - best, higher, high, balanced(default)\n")
            _T("                                    fast, faster, fastest\n")
            _T("\n")
            _T("   --ref <int>                  reference frames for sw encoding\n")
            _T("                                  default %d (auto)\n")
            _T("-b,--bframes <int>              number of sequential b frames\n")
            _T("                                  default %d (auto)\n")
            _T("\n")
            _T("   --gop-len <int>              (max) gop length, default %d (auto)\n")
            _T("                                  when auto, fps x 10 will be set.\n")
            _T("   --(no-)open-gop              enables open gop (default:off)\n")
            _T("   --strict-gop                 force gop structure\n")
            _T("   --(no-)scenechange           enables scene change detection\n")
            _T("\n")
            _T("   --level <string>             set codec level, default auto\n")
            _T("   --profile <string>           set codec profile, default auto\n")
            _T("                                 H.264: Baseline, Main, High\n")
            _T("                                 HEVC : Main\n")
            _T("                                 MPEG2: Simple, Main, High\n")
            _T("   --sar <int>:<int>            set Sample Aspect Ratio\n")
            _T("   --bluray                     for H.264 bluray encoding\n")
            _T("\n")
            _T("   --vpp-denoise <int>          use vpp denoise, set strength\n")
            _T("   --vpp-detail-enhance <int>   use vpp detail enahancer, set strength\n")
            _T("   --vpp-deinterlace <string>   set vpp deinterlace mode\n")
            _T("                                enabled only when set --tff or --bff\n")
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
            _T("   --vpp-delogo <string>        set delogo file path\n")
            _T("   --vpp-delogo-select <string> set target logo name or auto select file")
            _T("                                 or logo index starting from 1.\n")
            _T("   --vpp-delogo-pos <int>:<int> set delogo pos offset\n")
            _T("   --vpp-delogo-depth <int>     set delogo depth [default:%d]\n")
            _T("   --vpp-delogo-y  <int>        set delogo y  param\n")
            _T("   --vpp-delogo-cb <int>        set delogo cb param\n")
            _T("   --vpp-delogo-cr <int>        set delogo cr param\n")
            _T("   --vpp-half-turn              half turn video image\n")
            _T("                                 unoptimized and very slow.\n"),
            QSV_DEFAULT_QPI, QSV_DEFAULT_QPP, QSV_DEFAULT_QPB,
            QSV_DEFAULT_QPI, QSV_DEFAULT_QPP, QSV_DEFAULT_QPB,
            QSV_DEFAULT_ICQ, QSV_DEFAULT_ICQ,
            QSV_DEFAULT_CONVERGENCE, QSV_DEFAULT_CONVERGENCE,
            QSV_DEFAULT_QVBR,
            QSV_LOOKAHEAD_DEPTH_MIN, QSV_LOOKAHEAD_DEPTH_MAX,
            QSV_DEFAULT_REF,
            QSV_DEFAULT_BFRAMES,
            QSV_DEFAULT_GOP_LEN,
            QSV_DEFAULT_VPP_DELOGO_DEPTH
            );
        _ftprintf(stdout, _T("\n")
            _T("   --input-buf <int>            buffer size for input (%d-%d)\n")
            _T("                                 default   hw: %d,  sw: %d\n")
            _T("                                 cannot be used with avqsv reader.\n"),
            QSV_INPUT_BUF_MIN, QSV_INPUT_BUF_MAX,
            QSV_DEFAULT_INPUT_BUF_HW, QSV_DEFAULT_INPUT_BUF_SW
            );
        _ftprintf(stdout,
            _T("   --log <string>               output log to file.\n")
            _T("   --log-level <string>         set output log level\n")
            _T("                                 info(default), warn, error, debug\n"));
        _ftprintf(stdout, _T("\n")
            _T(" settings below are only supported with API v1.3\n")
            _T("   --fullrange                  set stream as fullrange yuv\n")
            );
        PrintListOptions(stdout, _T("--videoformat <string>"), list_videoformat, 0);
        PrintListOptions(stdout, _T("--colormatrix <string>"), list_colormatrix, 0);
        PrintListOptions(stdout, _T("--colorprim <string>"), list_colorprim, 0);
        PrintListOptions(stdout, _T("--transfer <string>"), list_transfer, 0);
        _ftprintf(stdout, _T("\n")
            _T(" settings below are only supported with API v1.6\n")
            _T("   --(no-)mbbrc                 enables per macro block rate control\n")
            _T("                                 default: off\n")
            _T("   --(no-)extbrc                enables extended rate control\n")
            _T("                                 default: off\n")
            );
        _ftprintf(stdout, _T("\n")
            _T(" settings below are only supported with API v1.7\n")
            _T("   --trellis <string>           set trellis mode used in encoding\n")
            _T("                                 - auto(default), off, i, ip, all\n")
            );
        _ftprintf(stdout, _T("\n")
            _T(" settings below are only supported with API v1.8\n")
            _T("   --(no-)i-adapt               enables adaptive I frame insert (default:off)\n")
            _T("   --(no-)b-adapt               enables adaptive B frame insert (default:off)\n")
            _T("   --(no-)b-pyramid             enables B-frame pyramid reference (default:off)\n")
            _T("   --lookahead-ds <string>      set lookahead quality.\n")
            _T("                                 - auto(default), fast, medium, slow\n")
            );
        _ftprintf(stdout, _T("\n")
            _T(" settings below are only supported with API v1.9\n")
            _T("   --(no-)intra-refresh         enables adaptive I frame insert\n")
            _T("   --no-deblock                 disables H.264 deblock feature\n")
            _T("   --qpmin <int> or             set min QP, default 0 (= unset)\n")
            _T("           <int>:<int>:<int>\n")
            _T("   --qpmax <int> or             set max QP, default 0 (= unset)\n")
            _T("           <int>:<int>:<int>\n")
            );
        _ftprintf(stdout, _T("\n")
            _T(" settings below are only supported with API v1.13\n")
            _T("   --mv-scaling                 set mv cost scaling\n")
            _T("                                 - 0  set MV cost to be 0\n")
            _T("                                 - 1  set MV cost 1/2 of default\n")
            _T("                                 - 2  set MV cost 1/4 of default\n")
            _T("                                 - 3  set MV cost 1/8 of default\n")
            _T("   --(no-)direct-bias-adjust    lower usage of B frame Direct/Skip type\n")
            );
        _ftprintf(stdout, _T("\n")
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
        _ftprintf(stdout, _T("\n")
            _T("   --benchmark <string>         run in benchmark mode\n")
            _T("                                 and write result in txt file\n")
            _T("   --(no-)timer-period-tuning   enable(disable) timer period tuning\n")
            _T("                                  default: enabled\n")
            );
    }
}

mfxStatus ParseInputString(TCHAR* strInput[], mfxU8 nArgNum, sInputParams* pParams)
{
    TCHAR* strArgument = _T("");

    if (1 == nArgNum)
    {
        PrintHelp(strInput[0], NULL, NULL);
        PrintHelp(strInput[0], _T("options needed."), NULL);
        return MFX_PRINT_OPTION_ERR;
    }


    MSDK_CHECK_POINTER(pParams, MFX_ERR_NULL_PTR);
    MSDK_ZERO_MEMORY(*pParams);
    mfxU16 tmp_input_buf  = 0;

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
    pParams->memType           = HW_MEMORY;
    pParams->nBframes          = QSV_DEFAULT_BFRAMES;
    pParams->nGOPLength        = QSV_DEFAULT_GOP_LEN;
    pParams->ColorPrim         = (mfxU16)list_colorprim[0].value;
    pParams->ColorMatrix       = (mfxU16)list_colormatrix[0].value;
    pParams->Transfer          = (mfxU16)list_transfer[0].value;
    pParams->VideoFormat       = (mfxU16)list_videoformat[0].value;
    pParams->nInputBufSize     = QSV_DEFAULT_INPUT_BUF_HW;
    pParams->bforceGOPSettings = QSV_DEFAULT_FORCE_GOP_LEN;
    pParams->vpp.delogo.nDepth = QSV_DEFAULT_VPP_DELOGO_DEPTH;

    // parse command line parameters
    for (mfxU8 i = 1; i < nArgNum; i++) {
        MSDK_CHECK_POINTER(strInput[i], MFX_ERR_NULL_PTR);

        const TCHAR *option_name = NULL;

        if (strInput[i][0] == _T('-')) {
            if (strInput[i][1] == _T('-')) {
                option_name = &strInput[i][2];
            } else if (strInput[i][2] == _T('\0')) {
                switch (strInput[i][1]) {
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
                    option_name = _T("format");
                    break;
                case _T('i'):
                    option_name = _T("input-file");
                    break;
                case _T('o'):
                    option_name = _T("output-file");
                    _tcscpy_s(pParams->strDstFile, strArgument);
                    break;
                case _T('v'):
                    option_name = _T("version");
                    break;
                case _T('h'):
                case _T('?'):
                    option_name = _T("help");
                    break;
                default:
                    PrintHelp(strInput[0], strsprintf(_T("Unknown options: \"%s\""), strInput[i]).c_str(), NULL);
                    return MFX_PRINT_OPTION_ERR;
                }
            } else {
                PrintHelp(strInput[0], strsprintf(_T("Invalid options: \"%s\""), strInput[i]).c_str(), NULL);
                return MFX_PRINT_OPTION_ERR;
            }
        }

        if (option_name == NULL) {
            PrintHelp(strInput[0], _T("Invalid options"), NULL);
            return MFX_PRINT_OPTION_ERR;
        }

        // process multi-character options
        if (0 == _tcscmp(option_name, _T("help")))
        {
            PrintHelp(strInput[0], NULL, NULL);
            return MFX_PRINT_OPTION_DONE;
        }
        else if (0 == _tcscmp(option_name, _T("version")))
        {
            PrintVersion();
            return MFX_PRINT_OPTION_DONE;
        }
        else if (0 == _tcscmp(option_name, _T("output-res")))
        {
            i++;
            if (2 == _stscanf_s(strInput[i], _T("%hdx%hd"), &pParams->nDstWidth, &pParams->nDstHeight))
                ;
            else if (2 == _stscanf_s(strInput[i], _T("%hd,%hd"), &pParams->nDstWidth, &pParams->nDstHeight))
                ;
            else if (2 == _stscanf_s(strInput[i], _T("%hd/%hd"), &pParams->nDstWidth, &pParams->nDstHeight))
                ;
            else if (2 == _stscanf_s(strInput[i], _T("%hd:%hd"), &pParams->nDstWidth, &pParams->nDstHeight))
                ;
            else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("input-res")))
        {
            i++;
            if (2 == _stscanf_s(strInput[i], _T("%hdx%hd"), &pParams->nWidth, &pParams->nHeight))
                ;
            else if (2 == _stscanf_s(strInput[i], _T("%hd,%hd"), &pParams->nWidth, &pParams->nHeight))
                ;
            else if (2 == _stscanf_s(strInput[i], _T("%hd/%hd"), &pParams->nWidth, &pParams->nHeight))
                ;
            else if (2 == _stscanf_s(strInput[i], _T("%hd:%hd"), &pParams->nWidth, &pParams->nHeight))
                ;
            else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("crop")))
        {
            i++;
            if (4 == _stscanf_s(strInput[i], _T("%hd,%hd,%hd,%hd"), &pParams->sInCrop.left, &pParams->sInCrop.up, &pParams->sInCrop.right, &pParams->sInCrop.bottom))
                ;
            else if (4 == _stscanf_s(strInput[i], _T("%hd:%hd:%hd:%hd"), &pParams->sInCrop.left, &pParams->sInCrop.up, &pParams->sInCrop.right, &pParams->sInCrop.bottom))
                ;
            else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("codec"))) {
            i++;
            int j = 0;
            for (; list_codec[j].desc; j++) {
                if (_tcsicmp(list_codec[j].desc, strInput[i]) == 0) {
                    pParams->CodecId = list_codec[j].value;
                    break;
                }
            }
            if (list_codec[j].desc == nullptr) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("raw")))
        {
            pParams->nInputFmt = INPUT_FMT_RAW;
        }
        else if (0 == _tcscmp(option_name, _T("y4m")))
        {
            pParams->nInputFmt = INPUT_FMT_Y4M;
        }
        else if (0 == _tcscmp(option_name, _T("avi")))
        {
            pParams->nInputFmt = INPUT_FMT_AVI;
        }
        else if (0 == _tcscmp(option_name, _T("avs")))
        {
            pParams->nInputFmt = INPUT_FMT_AVS;
        }
        else if (0 == _tcscmp(option_name, _T("vpy")))
        {
            pParams->nInputFmt = INPUT_FMT_VPY;
        }
        else if (0 == _tcscmp(option_name, _T("vpy-mt")))
        {
            pParams->nInputFmt = INPUT_FMT_VPY_MT;
        }
        else if (0 == _tcscmp(option_name, _T("avqsv")))
        {
            pParams->nInputFmt = INPUT_FMT_AVCODEC_QSV;
        }
        else if (0 == _tcscmp(option_name, _T("avqsv-analyze")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nAVDemuxAnalyzeSec)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("input-file")))
        {
            i++;
            _tcscpy_s(pParams->strSrcFile, strInput[i]);
        }
        else if (0 == _tcscmp(option_name, _T("output-file")))
        {
            i++;
            if (!pParams->bBenchmark)
                _tcscpy_s(pParams->strDstFile, strInput[i]);
        }
        else if (0 == _tcscmp(option_name, _T("trim")))
        {
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
                for (int i = (int)trim_list.size() - 2; i >= 0; i--) {
                    if (trim_list[i].fin > trim_list[i+1].start) {
                        trim_list[i].fin = trim_list[i+1].fin;
                        trim_list.erase(trim_list.begin() + i+1);
                    }
                }
                pParams->nTrimCount = (mfxU16)trim_list.size();
                pParams->pTrimList = (sTrim *)malloc(sizeof(pParams->pTrimList[0]) * trim_list.size());
                memcpy(pParams->pTrimList, &trim_list[0], sizeof(pParams->pTrimList[0]) * trim_list.size());
            }
        }
        else if (0 == _tcscmp(option_name, _T("audio-file")))
        {
            i++;
            TCHAR *ptr = strInput[i];
            int trackId = 0;
            if (_tcschr(ptr, '?') == NULL || 1 != _stscanf(ptr, _T("%d?"), &trackId)) {
                //トラック番号を適当に発番する (カウントは1から)
                bool trackFound = true;
                for (trackId = 0; trackFound; ) {
                    trackId++;
                    trackFound = false;
                    for (int i = 0; !trackFound && i < pParams->nAudioExtractFileCount; i++) {
                        trackFound = (pParams->pAudioExtractFileSelect[i] == trackId);
                    }
                }
            } else if (i <= 0) {
                //トラック番号は1から連番で指定
                PrintHelp(strInput[0], _T("Invalid track number"), option_name);
                return MFX_PRINT_OPTION_ERR;
            } else {
                //トラック番号が重複していないかを確認する
                for (int i = 0; i < pParams->nAudioExtractFileCount; i++) {
                    if (pParams->pAudioExtractFileSelect[0] == trackId) {
                        PrintHelp(strInput[0], _T("Same track number is used more than twice"), option_name);
                        return MFX_PRINT_OPTION_ERR;
                    }
                }
                ptr = _tcschr(ptr, '?') + 1;
            }
            TCHAR *format = NULL;
            TCHAR *qtr = _tcschr(ptr, ':');
            if (qtr != NULL && !(ptr + 1 == qtr && qtr[1] == _T('\\'))) {
                size_t len = (qtr - ptr);
                format = (TCHAR *)calloc((len + 1), sizeof(format[0]));
                memcpy(format, ptr, sizeof(format[0]) * len);
                ptr = qtr + 1;
            }
            //追加するもののidx
            const int idx = pParams->nAudioExtractFileCount;
            //領域再確保
            pParams->nAudioExtractFileCount++;
            pParams->pAudioExtractFileSelect   = (int *)realloc(pParams->pAudioExtractFileSelect, sizeof(pParams->pAudioExtractFileSelect[0]) * pParams->nAudioExtractFileCount);
            pParams->ppAudioExtractFilename = (TCHAR **)realloc(pParams->ppAudioExtractFilename,  sizeof(pParams->ppAudioExtractFilename[0])  * pParams->nAudioExtractFileCount);
            pParams->ppAudioExtractFormat    = (TCHAR **)realloc(pParams->ppAudioExtractFormat,   sizeof(pParams->ppAudioExtractFormat[0])    * pParams->nAudioExtractFileCount);
            pParams->pAudioExtractFileSelect[idx] = trackId;
            pParams->ppAudioExtractFormat[idx] = format;
            int filename_len = (int)_tcslen(ptr);
            //ファイル名が""でくくられてたら取り除く
            if (ptr[0] == _T('\"') && ptr[filename_len-1] == _T('\"')) {
                filename_len -= 2;
                ptr++;
            }
            //ファイル名が重複していないかを確認する
            for (int i = 0; i < pParams->nAudioExtractFileCount-1; i++) {
                if (0 == _tcsicmp(pParams->ppAudioExtractFilename[i], ptr)) {
                    PrintHelp(strInput[0], _T("Same output file name is used more than twice"), option_name);
                    return MFX_PRINT_OPTION_ERR;
                }
            }
            pParams->ppAudioExtractFilename[idx] = (TCHAR *)calloc((filename_len + 1), sizeof(pParams->ppAudioExtractFilename[idx][0]));
            memcpy(pParams->ppAudioExtractFilename[idx], ptr, sizeof(pParams->ppAudioExtractFilename[idx][0]) * filename_len);
        }
        else if (0 == _tcscmp(option_name, _T("format")))
        {
            pParams->nAVMux |= QSVENC_MUX_VIDEO;
            if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
                i++;
                const int formatLen = (int)_tcslen(strInput[i]);
                pParams->pAVMuxOutputFormat = (TCHAR *)realloc(pParams->pAVMuxOutputFormat, sizeof(pParams->pAVMuxOutputFormat[0]) * (formatLen + 1));
                _tcscpy_s(pParams->pAVMuxOutputFormat, formatLen + 1, strInput[i]);
            }
        }
        else if (0 == _tcscmp(option_name, _T("copy-audio")))
        {
            pParams->nAVMux |= (QSVENC_MUX_VIDEO | QSVENC_MUX_AUDIO);
            if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
                i++;
                auto trackListStr = split(strInput[i], _T(","));
                std::set<int> trackSet; //重複しないよう、setを使う
                for (auto str : trackListStr) {
                    int i = 0;
                    if (1 != _stscanf(str.c_str(), _T("%d"), &i) || i < 1) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name);
                        return MFX_PRINT_OPTION_ERR;
                    } else {
                        trackSet.insert(i);
                    }
                }
                pParams->nAudioSelectCount = (mfxU8)trackSet.size();
                if (NULL == (pParams->pAudioSelect = (int *)realloc(pParams->pAudioSelect, sizeof(pParams->pAudioSelect) * pParams->nAudioSelectCount))) {
                    return MFX_PRINT_OPTION_ERR;
                } else {
                    int i = 0;
                    for (auto it = trackSet.begin(); it != trackSet.end(); it++, i++) {
                        pParams->pAudioSelect[i] = *it;
                    }
                }
            }
        }
        else if (0 == _tcscmp(option_name, _T("quality")))
        {
            i++;
            int value = MFX_TARGETUSAGE_BALANCED;
            if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
                pParams->nTargetUsage = (mfxU16)clamp(value, MFX_TARGETUSAGE_BEST_QUALITY, MFX_TARGETUSAGE_BEST_SPEED);
            } else if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_quality_for_option, strInput[i]))) {
                pParams->nTargetUsage = (mfxU16)value;
            } else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("level")))
        {
            i++;
            double d;
            int value;
            if (_tcscmp(strInput[i], _T("1b")) == 0) {
                pParams->CodecLevel = MFX_LEVEL_AVC_1b;
                continue;
            }
            if (1 == _stscanf_s(strInput[i], _T("%lf"), &d)) {
                if (get_cx_index(list_avc_level, (int)(d * 10.0 + 0.5)) >= 0) {
                    pParams->CodecLevel = (mfxU16)(d * 10.0 + 0.5);
                    continue;
                }
            }
            if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_mpeg2_level, strInput[i]))) {
                pParams->CodecLevel = (mfxU16)value;
                continue;
            }                
            if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_vc1_level, strInput[i]))) {
                pParams->CodecLevel = (mfxU16)value;
                continue;
            }                
            if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_vc1_level_adv, strInput[i]))) {
                pParams->CodecLevel = (mfxU16)value;
                continue;
            }
            PrintHelp(strInput[0], _T("Unknown value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        else if (0 == _tcscmp(option_name, _T("profile")))
        {
            i++;
            int value;
            if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_avc_profile, strInput[i]))) {
                pParams->CodecProfile = (mfxU16)value;
                continue;
            }
            if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_mpeg2_profile, strInput[i]))) {
                pParams->CodecProfile = (mfxU16)value;
                continue;
            }
            if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_vc1_profile, strInput[i]))) {
                pParams->CodecProfile = (mfxU16)value;
                continue;
            }
            PrintHelp(strInput[0], _T("Unknown value"), option_name);
            return MFX_PRINT_OPTION_ERR;
        }
        else if (0 == _tcscmp(option_name, _T("sar")))
        {
            i++;
            if (2 == _stscanf_s(strInput[i], _T("%dx%d"), &pParams->nPAR[0], &pParams->nPAR[1]))
                ;
            else if (2 == _stscanf_s(strInput[i], _T("%d,%d"), &pParams->nPAR[0], &pParams->nPAR[1]))
                ;
            else if (2 == _stscanf_s(strInput[i], _T("%d/%d"), &pParams->nPAR[0], &pParams->nPAR[1]))
                ;
            else if (2 == _stscanf_s(strInput[i], _T("%d:%d"), &pParams->nPAR[0], &pParams->nPAR[1]))
                ;
            else {
                MSDK_ZERO_MEMORY(pParams->nPAR);
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("sw")))
        {
            pParams->bUseHWLib = false;
        }
        else if (0 == _tcscmp(option_name, _T("slices")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nSlices)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("gop-len")))
        {
            i++;
            if (0 == _tcsnccmp(strInput[i], _T("auto"), _tcslen(_T("auto")))) {
                pParams->nGOPLength = 0;
            } else if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nGOPLength)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("open-gop")))
        {
            pParams->bopenGOP = true;
        }
        else if (0 == _tcscmp(option_name, _T("no-open-gop")))
        {
            pParams->bopenGOP = false;
        }
        else if (0 == _tcscmp(option_name, _T("strict-gop")))
        {
            pParams->bforceGOPSettings = true;
        }
        else if (0 == _tcscmp(option_name, _T("no-scenechange")))
        {
            pParams->bforceGOPSettings = true;
        }
        else if (0 == _tcscmp(option_name, _T("scenechange")))
        {
            pParams->bforceGOPSettings = false;
        }
        else if (0 == _tcscmp(option_name, _T("i-adapt")))
        {
            pParams->bAdaptiveI = true;
        }
        else if (0 == _tcscmp(option_name, _T("no-i-adapt")))
        {
            pParams->bAdaptiveI = false;
        }
        else if (0 == _tcscmp(option_name, _T("b-adapt")))
        {
            pParams->bAdaptiveB = true;
        }
        else if (0 == _tcscmp(option_name, _T("no-b-adapt")))
        {
            pParams->bAdaptiveB = false;
        }
        else if (0 == _tcscmp(option_name, _T("b-pyramid")))
        {
            pParams->bBPyramid = true;
        }
        else if (0 == _tcscmp(option_name, _T("no-b-pyramid")))
        {
            pParams->bBPyramid = false;
        }
        else if (0 == _tcscmp(option_name, _T("lookahead-ds")))
        {
            i++;
            int value = MFX_LOOKAHEAD_DS_UNKNOWN;
            if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_lookahead_ds, strInput[i]))) {
                pParams->nLookaheadDS = (mfxU16)value;
            } else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("trellis")))
        {
            i++;
            int value = MFX_TRELLIS_UNKNOWN;
            if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_avc_trellis_for_options, strInput[i]))) {
                pParams->nTrellis = (mfxU16)value;
            } else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("bluray")))
        {
            pParams->nBluray = 1;
        }
        else if (0 == _tcscmp(option_name, _T("force-bluray")))
        {
            pParams->nBluray = 2;
        }
        else if (0 == _tcscmp(option_name, _T("nv12")))
        {
            pParams->ColorFormat = MFX_FOURCC_NV12;
        }
        else if (0 == _tcscmp(option_name, _T("tff")))
        {
            pParams->nPicStruct = MFX_PICSTRUCT_FIELD_TFF;
        }
        else if (0 == _tcscmp(option_name, _T("bff")))
        {
            pParams->nPicStruct = MFX_PICSTRUCT_FIELD_BFF;
        }
        else if (0 == _tcscmp(option_name, _T("la")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->nEncMode = MFX_RATECONTROL_LA;
        }
        else if (0 == _tcscmp(option_name, _T("icq")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nICQQuality)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->nEncMode = MFX_RATECONTROL_ICQ;
        }
        else if (0 == _tcscmp(option_name, _T("la-icq")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nICQQuality)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->nEncMode = MFX_RATECONTROL_LA_ICQ;
        }
        else if (0 == _tcscmp(option_name, _T("la-hrd")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->nEncMode = MFX_RATECONTROL_LA_HRD;
        }
        else if (0 == _tcscmp(option_name, _T("vcm")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->nEncMode = MFX_RATECONTROL_VCM;
        }
        else if (0 == _tcscmp(option_name, _T("vbr")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->nEncMode = MFX_RATECONTROL_VBR;
        }
        else if (0 == _tcscmp(option_name, _T("cbr")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->nEncMode = MFX_RATECONTROL_CBR;
        }
        else if (0 == _tcscmp(option_name, _T("avbr")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->nEncMode = MFX_RATECONTROL_AVBR;
        }
        else if (0 == _tcscmp(option_name, _T("qvbr")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->nEncMode = MFX_RATECONTROL_QVBR;
        }
        else if (0 == _tcscmp(option_name, _T("qvbr-q"))
              || 0 == _tcscmp(option_name, _T("qvbr-quality")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nQVBRQuality)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->nEncMode = MFX_RATECONTROL_QVBR;
        }
        else if (0 == _tcscmp(option_name, _T("max-bitrate"))
            ||   0 == _tcscmp(option_name, _T("maxbitrate"))) //互換性のため
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nMaxBitrate)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("la-depth")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nLookaheadDepth)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("la-window-size")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nWinBRCSize)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("cqp")) || 0 == _tcscmp(option_name, _T("vqp")))
        {
            i++;
            if (3 == _stscanf_s(strInput[i], _T("%hd:%hd:%hd"), &pParams->nQPI, &pParams->nQPP, &pParams->nQPB))
                ;
            else if (3 == _stscanf_s(strInput[i], _T("%hd,%hd,%hd"), &pParams->nQPI, &pParams->nQPP, &pParams->nQPB))
                ;
            else if (3 == _stscanf_s(strInput[i], _T("%hd/%hd/%hd"), &pParams->nQPI, &pParams->nQPP, &pParams->nQPB))
                ;
            else if (1 == _stscanf_s(strInput[i], _T("%hd"), &pParams->nQPI)) {
                pParams->nQPP = pParams->nQPI;
                pParams->nQPB = pParams->nQPI;
            } else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->nEncMode = (mfxU16)((0 == _tcscmp(option_name, _T("vqp"))) ? MFX_RATECONTROL_VQP : MFX_RATECONTROL_CQP);
        }
        else if (0 == _tcscmp(option_name, _T("avbr-unitsize")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nAVBRConvergence)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        //else if (0 == _tcscmp(option_name, _T("avbr-range")))
        //{
        //    double accuracy;
        //    if (1 != _stscanf_s(strArgument, _T("%f"), &accuracy)) {
        //        PrintHelp(strInput[0], _T("Unknown value"), option_name);
        //        return MFX_PRINT_OPTION_ERR;
        //    }
        //    pParams->nAVBRAccuarcy = (mfxU16)(accuracy * 10 + 0.5);
        //}
        else if (0 == _tcscmp(option_name, _T("ref")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nRef)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("bframes")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nBframes)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("cavlc")))
        {
            pParams->bCAVLC = true;
        }
        else if (0 == _tcscmp(option_name, _T("rdo")))
        {
            pParams->bRDO = true;
        }
        else if (0 == _tcscmp(option_name, _T("extbrc")))
        {
            pParams->bExtBRC = true;
        }
        else if (0 == _tcscmp(option_name, _T("no-extbrc")))
        {
            pParams->bExtBRC = false;
        }
        else if (0 == _tcscmp(option_name, _T("mbbrc")))
        {
            pParams->bMBBRC = true;
        }
        else if (0 == _tcscmp(option_name, _T("no-mbbrc")))
        {
            pParams->bMBBRC = false;
        }
        else if (0 == _tcscmp(option_name, _T("no-intra-refresh")))
        {
            pParams->bIntraRefresh = false;
        }
        else if (0 == _tcscmp(option_name, _T("intra-refresh")))
        {
            pParams->bIntraRefresh = true;
        }
        else if (0 == _tcscmp(option_name, _T("no-deblock")))
        {
            pParams->bNoDeblock = true;
        }
        else if (0 == _tcscmp(option_name, _T("qpmax")) || 0 == _tcscmp(option_name, _T("qpmin")))
        {
            i++;
            mfxU32 qpLimit[3] = { 0 };
            if (3 == _stscanf_s(strInput[i], _T("%d:%d:%d"), &qpLimit[0], &qpLimit[1], &qpLimit[2]))
                ;
            else if (3 == _stscanf_s(strInput[i], _T("%d,%d,%d"), &qpLimit[0], &qpLimit[1], &qpLimit[2]))
                ;
            else if (3 == _stscanf_s(strInput[i], _T("%d/%d/%d"), &qpLimit[0], &qpLimit[1], &qpLimit[2]))
                ;
            else if (1 == _stscanf_s(strInput[i], _T("%d"), &qpLimit[0])) {
                qpLimit[1] = qpLimit[0];
                qpLimit[2] = qpLimit[0];
            } else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            mfxU8 *limit = (0 == _tcscmp(option_name, _T("qpmin"))) ? pParams->nQPMin : pParams->nQPMax;
            for (int i = 0; i < 3; i++) {
                limit[i] = (mfxU8)clamp(qpLimit[i], 0, 51);
            }
        }
        else if (0 == _tcscmp(option_name, _T("mv-scaling")))
        {
            i++;
            int value = 0;
            if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
                pParams->bGlobalMotionAdjust = true;
                pParams->nMVCostScaling = (mfxU8)value;
            } else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("direct-bias-adjust")))
        {
            pParams->bDirectBiasAdjust = true;
        }
        else if (0 == _tcscmp(option_name, _T("no-direct-bias-adjust")))
        {
            pParams->bDirectBiasAdjust = false;
        }
        else if (0 == _tcscmp(option_name, _T("fullrange")))
        {
            pParams->bFullrange = true;
        }
        else if (0 == _tcscmp(option_name, _T("inter-pred")))
        {
            i++;
            mfxI32 v;
            if (1 != _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_pred_block_size) - 1) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->nInterPred = (mfxU16)list_pred_block_size[v].value;
        }
        else if (0 == _tcscmp(option_name, _T("intra-pred")))
        {
            i++;
            mfxI32 v;
            if (1 != _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_pred_block_size) - 1) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->nIntraPred = (mfxU16)list_pred_block_size[v].value;
        }
        else if (0 == _tcscmp(option_name, _T("mv-precision")))
        {
            i++;
            mfxI32 v;
            if (1 != _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_mv_presicion) - 1) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->nMVPrecision = (mfxU16)list_mv_presicion[v].value;
        }
        else if (0 == _tcscmp(option_name, _T("mv-search")))
        {
            i++;
            mfxI32 v;
            if (1 != _stscanf_s(strInput[i], _T("%d"), &v)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->MVSearchWindow.x = (mfxU16)clamp(v, 0, 128);
            pParams->MVSearchWindow.y = (mfxU16)clamp(v, 0, 128);
        }
        else if (0 == _tcscmp(option_name, _T("fps")))
        {
            i++;
            if (2 == _stscanf_s(strInput[i], _T("%d/%d"), &pParams->nFPSRate, &pParams->nFPSScale))
                ;
            else if (2 == _stscanf_s(strInput[i], _T("%d:%d"), &pParams->nFPSRate, &pParams->nFPSScale))
                ;
            else if (2 == _stscanf_s(strInput[i], _T("%d,%d"), &pParams->nFPSRate, &pParams->nFPSScale))
                ;
            else {
                double d;
                if (1 == _stscanf_s(strInput[i], _T("%lf"), &d)) {
                    int rate = (int)(d * 1001.0 + 0.5);
                    if (rate % 1000 == 0) {
                        pParams->nFPSRate = rate;
                        pParams->nFPSScale = 1001;
                    } else {
                        pParams->nFPSScale = 100000;
                        pParams->nFPSRate = (int)(d * pParams->nFPSScale + 0.5);
                        int gcd = GCD(pParams->nFPSRate, pParams->nFPSScale);
                        pParams->nFPSScale /= gcd;
                        pParams->nFPSRate  /= gcd;
                    }
                } else  {
                    PrintHelp(strInput[0], _T("Unknown value"), option_name);
                    return MFX_PRINT_OPTION_ERR;
                }
            }
        }
        else if (0 == _tcscmp(option_name, _T("log-level")))
        {
            i++;
            mfxI32 v;
            if (PARSE_ERROR_FLAG != (v = get_value_from_chr(list_log_level, strInput[i]))) {
                pParams->nLogLevel = (mfxI16)v;
            } else if (1 == _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_log_level) - 1) {
                pParams->nLogLevel = (mfxI16)v;
            } else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
#ifdef D3D_SURFACES_SUPPORT
        else if (0 == _tcscmp(option_name, _T("disable-d3d")))
        {
            pParams->memType = SYSTEM_MEMORY;
        }
        else if (0 == _tcscmp(option_name, _T("d3d9")))
        {
            pParams->memType = D3D9_MEMORY;
        }
#if MFX_D3D11_SUPPORT
        else if (0 == _tcscmp(option_name, _T("d3d11")))
        {
            pParams->memType = D3D11_MEMORY;
        }
        else if (0 == _tcscmp(option_name, _T("d3d")))
        {
            pParams->memType = HW_MEMORY;
        }
#else
        else if (0 == _tcscmp(option_name, _T("d3d")))
        {
            pParams->memType = D3D9_MEMORY;
        }
#endif //MFX_D3D11_SUPPORT
#endif //D3D_SURFACES_SUPPORT
        else if (0 == _tcscmp(option_name, _T("vpp-denoise")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->vpp.nDenoise)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->vpp.bEnable = true;
            pParams->vpp.bUseDenoise = true;
        }
        else if (0 == _tcscmp(option_name, _T("vpp-no-denoise")))
        {
            i++;
            pParams->vpp.bUseDenoise = false;
            pParams->vpp.nDenoise = 0;
        }
        else if (0 == _tcscmp(option_name, _T("vpp-detail-enhance")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->vpp.nDetailEnhance)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->vpp.bEnable = true;
            pParams->vpp.bUseDetailEnhance = true;
        }
        else if (0 == _tcscmp(option_name, _T("vpp-no-detail-enhance")))
        {
            i++;
            pParams->vpp.bUseDetailEnhance = false;
            pParams->vpp.nDetailEnhance = 0;
        }
        else if (0 == _tcscmp(option_name, _T("vpp-deinterlace")))
        {
            i++;
            int value = get_value_from_chr(list_deinterlace, strInput[i]);
            if (PARSE_ERROR_FLAG == value) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            } else {
                pParams->vpp.bEnable = true;
                pParams->vpp.nDeinterlace = (mfxU16)value;
            }
            if (pParams->vpp.nDeinterlace == MFX_DEINTERLACE_IT_MANUAL) {
                i++;
                if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_telecine_patterns, strInput[i]))) {
                    PrintHelp(strInput[0], _T("Unknown value"), option_name);
                    return MFX_PRINT_OPTION_ERR;
                } else {
                    pParams->vpp.nTelecinePattern = (mfxU16)value;
                }
            }
        }
        else if (0 == _tcscmp(option_name, _T("vpp-image-stab")))
        {
            i++;
            int value = 0;
            if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
                pParams->vpp.nImageStabilizer = (mfxU16)value;
            } else if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_vpp_image_stabilizer, strInput[i]))) {
                pParams->vpp.nImageStabilizer = (mfxU16)value;
            } else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("vpp-fps-conv")))
        {
            i++;
            int value = 0;
            if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
                pParams->vpp.nFPSConversion = (mfxU16)value;
            } else if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_vpp_fps_conversion, strInput[i]))) {
                pParams->vpp.nFPSConversion = (mfxU16)value;
            } else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("vpp-half-turn")))
        {
            pParams->vpp.bHalfTurn = true;
        }
        else if (0 == _tcscmp(option_name, _T("vpp-delogo"))
              || 0 == _tcscmp(option_name, _T("vpp-delogo-file")))
        {
            i++;
            int filename_len = (int)_tcslen(strInput[i]);
            pParams->vpp.delogo.pFilePath = (TCHAR *)calloc(filename_len + 1, sizeof(pParams->vpp.delogo.pFilePath[0]));
            memcpy(pParams->vpp.delogo.pFilePath, strInput[i], sizeof(pParams->vpp.delogo.pFilePath[0]) * filename_len);
        }
        else if (0 == _tcscmp(option_name, _T("vpp-delogo-select")))
        {
            i++;
            int filename_len = (int)_tcslen(strInput[i]);
            pParams->vpp.delogo.pSelect = (TCHAR *)calloc(filename_len + 1, sizeof(pParams->vpp.delogo.pSelect[0]));
            memcpy(pParams->vpp.delogo.pSelect, strInput[i], sizeof(pParams->vpp.delogo.pSelect[0]) * filename_len);
        }
        else if (0 == _tcscmp(option_name, _T("vpp-delogo-pos")))
        {
            i++;
            mfxI16Pair posOffset;
            if (2 == _stscanf_s(strInput[i], _T("%hdx%hd"), &posOffset.x, &posOffset.y))
                ;
            else if (2 == _stscanf_s(strInput[i], _T("%hd,%hd"), &posOffset.x, &posOffset.y))
                ;
            else if (2 == _stscanf_s(strInput[i], _T("%hd/%hd"), &posOffset.x, &posOffset.y))
                ;
            else if (2 == _stscanf_s(strInput[i], _T("%hd:%hd"), &posOffset.x, &posOffset.y))
                ;
            else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->vpp.delogo.nPosOffset = posOffset;
        }
        else if (0 == _tcscmp(option_name, _T("vpp-delogo-depth")))
        {
            i++;
            mfxI16 depth;
            if (1 != _stscanf_s(strInput[i], _T("%hd"), &depth)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->vpp.delogo.nDepth = depth;
        }
        else if (0 == _tcscmp(option_name, _T("vpp-delogo-y")))
        {
            i++;
            mfxI16 value;
            if (1 != _stscanf_s(strInput[i], _T("%hd"), &value)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->vpp.delogo.nYOffset = value;
        }
        else if (0 == _tcscmp(option_name, _T("vpp-delogo-cb")))
        {
            i++;
            mfxI16 value;
            if (1 != _stscanf_s(strInput[i], _T("%hd"), &value)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->vpp.delogo.nCbOffset = value;
        }
        else if (0 == _tcscmp(option_name, _T("vpp-delogo-cr")))
        {
            i++;
            mfxI16 value;
            if (1 != _stscanf_s(strInput[i], _T("%hd"), &value)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
            pParams->vpp.delogo.nCrOffset = value;
        }
        else if (0 == _tcscmp(option_name, _T("input-buf")))
        {
            i++;
            if (1 != _stscanf_s(strInput[i], _T("%hd"), &tmp_input_buf)) {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("log")))
        {
            i++;
            int filename_len = (int)_tcslen(strInput[i]);
            pParams->pStrLogFile = (TCHAR *)calloc(filename_len + 1, sizeof(pParams->pStrLogFile[0]));
            memcpy(pParams->pStrLogFile, strInput[i], sizeof(pParams->pStrLogFile[0]) * filename_len);
        }
        else if (0 == _tcscmp(option_name, _T("check-environment")))
        {
            PrintVersion();

            TCHAR buffer[4096];
            getEnviromentInfo(buffer, _countof(buffer));
            _ftprintf(stdout, buffer);
            return MFX_PRINT_OPTION_DONE;
        }
        else if (0 == _tcscmp(option_name, _T("check-features")))
        {
            PrintVersion();
            TCHAR buffer[4096];
            getEnviromentInfo(buffer, _countof(buffer), false);
            _ftprintf(stdout, _T("%s\n"), buffer);

            mfxVersion test = { 0, 1 };
            for (int impl_type = 0; impl_type < 2; impl_type++) {
                mfxVersion lib = (impl_type) ? get_mfx_libsw_version() : get_mfx_libhw_version();
                const TCHAR *impl_str = (impl_type) ?  _T("Software") : _T("Hardware");
                if (!check_lib_version(lib, test)) {
                    _ftprintf(stdout, _T("Media SDK %s unavailable.\n"), impl_str);
                } else {
                    _ftprintf(stdout, _T("Media SDK %s API v%d.%d\n"), impl_str, lib.Major, lib.Minor);
                    std::basic_string<msdk_char> strEnc;
                    MakeFeatureListStr(0 == impl_type, strEnc);
                    _ftprintf(stdout, _T("Supported Enc features:\n%s\n\n"), strEnc.c_str());
                    std::basic_string<msdk_char> strVpp;
                    MakeVppFeatureStr(0 == impl_type, strVpp);
                    _ftprintf(stdout, _T("Supported Vpp features:\n%s\n\n"), strVpp.c_str());
                }
            }
            return MFX_PRINT_OPTION_DONE;
        }
        else if (0 == _tcscmp(option_name, _T("check-hw"))
              || 0 == _tcscmp(option_name, _T("hw-check"))) //互換性のため
        {
            mfxVersion ver = { 0, 1 };
            if (check_lib_version(get_mfx_libhw_version(), ver) != 0) {
                _ftprintf(stdout, _T("Success: QuickSyncVideo (hw encoding) available"));
                return MFX_PRINT_OPTION_DONE;
            } else {
                _ftprintf(stdout, _T("Error: QuickSyncVideo (hw encoding) unavailable"));
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("lib-check"))
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
#if ENABLE_AVCODEC_QSV_READER
        else if (0 == _tcscmp(option_name, _T("check-codecs")))
        {
            _ftprintf(stdout, _T("%s\n"), getAVCodecs((AVQSVCodecType)(AVQSV_CODEC_DEC | AVQSV_CODEC_ENC)).c_str());
            return MFX_PRINT_OPTION_DONE;
        }
        else if (0 == _tcscmp(option_name, _T("check-encoders")))
        {
            _ftprintf(stdout, _T("%s\n"), getAVCodecs(AVQSV_CODEC_ENC).c_str());
            return MFX_PRINT_OPTION_DONE;
        }
        else if (0 == _tcscmp(option_name, _T("check-decoders")))
        {
            _ftprintf(stdout, _T("%s\n"), getAVCodecs(AVQSV_CODEC_DEC).c_str());
            return MFX_PRINT_OPTION_DONE;
        }
        else if (0 == _tcscmp(option_name, _T("check-formats")))
        {
            _ftprintf(stdout, _T("%s\n"), getAVFormats((AVQSVFormatType)(AVQSV_FORMAT_DEMUX | AVQSV_FORMAT_MUX)).c_str());
            return MFX_PRINT_OPTION_DONE;
        }
#endif //ENABLE_AVCODEC_QSV_READER
        else if (0 == _tcscmp(option_name, _T("colormatrix")))
        {
            i++;
            int value;
            if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_colormatrix, strInput[i])))
                pParams->ColorMatrix = (mfxU16)value;
        }
        else if (0 == _tcscmp(option_name, _T("colorprim")))
        {
            i++;
            int value;
            if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_colorprim, strInput[i])))
                pParams->ColorPrim = (mfxU16)value;
        }
        else if (0 == _tcscmp(option_name, _T("transfer")))
        {
            i++;
            int value;
            if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_transfer, strInput[i])))
                pParams->Transfer = (mfxU16)value;
        }
        else if (0 == _tcscmp(option_name, _T("videoformat")))
        {
            i++;
            int value;
            if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_videoformat, strInput[i])))
                pParams->ColorMatrix = (mfxU16)value;
        }
        else if (0 == _tcscmp(option_name, _T("fullrange")))
        {
            pParams->bFullrange = true;
        }
        else if (0 == _tcscmp(option_name, _T("sar")))
        {
            i++;
            if (2 == _stscanf_s(strInput[i], _T("%dx%d"), &pParams->nPAR[0], &pParams->nPAR[1]))
                ;
            else if (2 == _stscanf_s(strInput[i], _T("%d,%d"), &pParams->nPAR[0], &pParams->nPAR[1]))
                ;
            else if (2 == _stscanf_s(strInput[i], _T("%d/%d"), &pParams->nPAR[0], &pParams->nPAR[1]))
                ;
            else if (2 == _stscanf_s(strInput[i], _T("%d:%d"), &pParams->nPAR[0], &pParams->nPAR[1]))
                ;
            else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name);
                return MFX_PRINT_OPTION_ERR;
            }
        }
        else if (0 == _tcscmp(option_name, _T("benchmark")))
        {
            i++;
            pParams->bBenchmark = TRUE;
            _tcscpy_s(pParams->strDstFile, strInput[i]);
        }
        else if (0 == _tcscmp(option_name, _T("timer-period-tuning")))
        {
            pParams->bDisableTimerPeriodTuning = false;
        }
        else if (0 == _tcscmp(option_name, _T("no-timer-period-tuning")))
        {
            pParams->bDisableTimerPeriodTuning = true;
        }
        else
        {
            tstring mes = _T("Unknown option: --");
            mes += option_name;
            PrintHelp(strInput[0], (TCHAR *)mes.c_str(), NULL);
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

    // set default values for optional parameters that were not set or were set incorrectly
    pParams->nTargetUsage = clamp(pParams->nTargetUsage, MFX_TARGETUSAGE_BEST_QUALITY, MFX_TARGETUSAGE_BEST_SPEED);

    // calculate default bitrate based on the resolution (a parameter for encoder, so Dst resolution is used)
    if (pParams->nBitRate == 0) {
        pParams->nBitRate = CalculateDefaultBitrate(pParams->CodecId, pParams->nTargetUsage, pParams->nDstWidth,
            pParams->nDstHeight, pParams->nFPSRate / (double)pParams->nFPSScale);
    }

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

    //set input buffer size
    if (tmp_input_buf == 0)
        tmp_input_buf = (pParams->bUseHWLib) ? QSV_DEFAULT_INPUT_BUF_HW : QSV_DEFAULT_INPUT_BUF_SW;
    pParams->nInputBufSize = clamp(tmp_input_buf, QSV_INPUT_BUF_MIN, QSV_INPUT_BUF_MAX);

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

bool check_locale_is_ja() {
    const WORD LangID_ja_JP = MAKELANGID(LANG_JAPANESE, SUBLANG_JAPANESE_JAPAN);
    return GetUserDefaultLangID() == LangID_ja_JP;
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

    std::auto_ptr<CEncodingPipeline>  pPipeline; 
    //pPipeline.reset((Params.nRotationAngle) ? new CUserPipeline : new CEncodingPipeline);
    pPipeline.reset(new CEncodingPipeline);
    MSDK_CHECK_POINTER(pPipeline.get(), MFX_ERR_MEMORY_ALLOC);

    sts = pPipeline->Init(params);
    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, 1);

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
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, 1);

            sts = pPipeline->ResetMFXComponents(params);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, 1);
            continue;
        } else {
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, 1);
            break;
        }
    }

    pPipeline->Close();

    return sts;
}

mfxStatus run_benchmark(sInputParams *params) {
    using namespace std;
    mfxStatus sts = MFX_ERR_NONE;
    basic_string<msdk_char> benchmarkLogFile = params->strDstFile;

    //テストする解像度
    const vector<pair<mfxU16, mfxU16>> test_resolution = { { 1920, 1080 }, { 1280, 720 } };

    //初回出力
    {
        params->nDstWidth = test_resolution[0].first;
        params->nDstHeight = test_resolution[0].second;
        params->nTargetUsage = MFX_TARGETUSAGE_BEST_SPEED;

        auto_ptr<CEncodingPipeline> pPipeline;
        pPipeline.reset(new CEncodingPipeline);
        MSDK_CHECK_POINTER(pPipeline.get(), MFX_ERR_MEMORY_ALLOC);

        sts = pPipeline->Init(params);
        MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

        pPipeline->SetAbortFlagPointer(&g_signal_abort);
        set_signal_handler();

        SYSTEMTIME sysTime = { 0 };
        GetLocalTime(&sysTime);

        msdk_char encode_info[4096] = { 0 };
        if (MFX_ERR_NONE != (sts = pPipeline->CheckCurrentVideoParam(encode_info, _countof(encode_info)))) {
            return sts;
        }

        bool hardware;
        mfxVersion ver;
        pPipeline->GetEncodeLibInfo(&ver, &hardware);

        msdk_char enviroment_info[4096] = { 0 };
        getEnviromentInfo(enviroment_info, _countof(enviroment_info));

        MemType memtype = pPipeline->GetMemType();

        basic_stringstream<msdk_char> ss;
        FILE *fp_bench = NULL;
        if (_tfopen_s(&fp_bench, benchmarkLogFile.c_str(), _T("a")) || NULL == fp_bench) {
            pPipeline->PrintMes(QSV_LOG_ERROR, _T("\nERROR: failed opening benchmark result file.\n"));
            return MFX_ERR_INVALID_HANDLE;
        } else {
            fprintf(fp_bench, "Started benchmark on %d.%02d.%02d %2d:%02d:%02d\n",
                sysTime.wYear, sysTime.wMonth, sysTime.wDay, sysTime.wHour, sysTime.wMinute, sysTime.wSecond);
            fprintf(fp_bench, "Basic parameters of the benchmark\n"
                              " (Target Usage and output resolution will be changed)\n");
            fprintf(fp_bench, "%s\n\n", tchar_to_string(encode_info).c_str());
            fprintf(fp_bench, tchar_to_string(enviroment_info).c_str());
            fprintf(fp_bench, "QSV: QSVEncC %s (%s) / API[%s]: v%d.%d / %s\n", 
                VER_STR_FILEVERSION, tchar_to_string(BUILD_ARCH_STR).c_str(), (hardware) ? "hw" : "sw", ver.Major, ver.Minor, tchar_to_string(MemTypeToStr(memtype)).c_str());
            fprintf(fp_bench, "\n");
            fclose(fp_bench);
        }
        basic_ofstream<msdk_char> benchmark_log_test_open(benchmarkLogFile, ios::out | ios::app);
        if (!benchmark_log_test_open.good()) {
            pPipeline->PrintMes(QSV_LOG_ERROR, _T("\nERROR: failed opening benchmark result file.\n"));
            return MFX_ERR_INVALID_HANDLE;
        }
        benchmark_log_test_open << ss.str();
        benchmark_log_test_open.close();

        for (;;) {
            sts = pPipeline->Run();

            if (MFX_ERR_DEVICE_LOST == sts || MFX_ERR_DEVICE_FAILED == sts) {
                pPipeline->PrintMes(QSV_LOG_ERROR, _T("\nERROR: Hardware device was lost or returned an unexpected error. Recovering...\n"));
                if (   MFX_ERR_NONE != (sts = pPipeline->ResetDevice())
                    || MFX_ERR_NONE != (sts = pPipeline->ResetMFXComponents(params)))
                    break;
            } else {
                if (MFX_ERR_NONE != sts)
                    MSDK_PRINT_RET_MSG(sts);
                break;
            }
        }

        sEncodeStatusData data = { 0 };
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

    //解像度ごとに、target usageを変化させて測定
    vector<vector<benchmark_t>> benchmark_result;
    benchmark_result.reserve(test_resolution.size() * (_countof(list_quality) - 1));

    for (int i = 0; MFX_ERR_NONE == sts && !g_signal_abort && list_quality[i].desc; i++) {
        params->nTargetUsage = (mfxU16)list_quality[i].value;
        vector<benchmark_t> benchmark_per_target_usage;
        for (auto resolution : test_resolution) {
            params->nDstWidth = resolution.first;
            params->nDstHeight = resolution.second;

            auto_ptr<CEncodingPipeline> pPipeline;
            pPipeline.reset(new CEncodingPipeline);
            MSDK_CHECK_POINTER(pPipeline.get(), MFX_ERR_MEMORY_ALLOC);

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
                    pPipeline->PrintMes(QSV_LOG_ERROR, _T("\nERROR: Hardware device was lost or returned an unexpected error. Recovering...\n"));
                    if (   MFX_ERR_NONE != (sts = pPipeline->ResetDevice())
                        || MFX_ERR_NONE != (sts = pPipeline->ResetMFXComponents(params)))
                        break;
                } else {
                    if (MFX_ERR_NONE != sts)
                        MSDK_PRINT_RET_MSG(sts);
                    break;
                }
            }

            sEncodeStatusData data = { 0 };
            sts = pPipeline->GetEncodeStatusData(&data);

            pPipeline->Close();

            benchmark_t result;
            result.resolution      = resolution;
            result.targetUsage     = (mfxU16)list_quality[i].value;
            result.fps             = data.fEncodeFps;
            result.bitrate         = data.fBitrateKbps;
            result.cpuUsagePercent = data.fCPUUsagePercent;
            benchmark_per_target_usage.push_back(result);

            _ftprintf(stderr, _T("\n"));

            if (MFX_ERR_NONE != sts || g_signal_abort)
                break;
        }

        benchmark_result.push_back(benchmark_per_target_usage);
    }

    //結果を出力
    if (MFX_ERR_NONE == sts && benchmark_result.size()) {
        basic_stringstream<msdk_char> ss;

        mfxU32 maxLengthOfTargetUsageDesc = 0;
        for (int i = 0; list_quality[i].desc; i++) {
            maxLengthOfTargetUsageDesc = max(maxLengthOfTargetUsageDesc, (mfxU32)_tcslen(list_quality[i].desc));
        }

        FILE *fp_bench = NULL;
        if (_tfopen_s(&fp_bench, benchmarkLogFile.c_str(), _T("a")) || NULL == fp_bench) {
            _ftprintf(stderr, _T("\nERROR: failed opening benchmark result file.\n"));
            return MFX_ERR_INVALID_HANDLE;
        } else {
            fprintf(fp_bench, "TargetUsage ([TU-1]:Best Quality) ～ ([TU-7]:Fastest Speed)\n\n");

            fprintf(fp_bench, "Encode Speed (fps)\n");
            fprintf(fp_bench, "TargetUsage");
            for (auto resolution : test_resolution) {
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
            for (auto resolution : test_resolution) {
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
            for (auto resolution : test_resolution) {
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
        qsv_print_stderr(QSV_LOG_ERROR, _T("\nError occurred during benchmark.\n"));
    }

    return sts;
}

int run(int argc, TCHAR *argv[]) {
    mfxStatus sts = MFX_ERR_NONE;
    sInputParams Params = { 0 };

    sts = ParseInputString(argv, (mfxU8)argc, &Params);
    if (sts >= MFX_PRINT_OPTION_DONE)
        return 0;

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

    if (Params.bBenchmark) {
        return run_benchmark(&Params);
    }

    std::auto_ptr<CEncodingPipeline>  pPipeline; 
    //pPipeline.reset((Params.nRotationAngle) ? new CUserPipeline : new CEncodingPipeline);
    pPipeline.reset(new CEncodingPipeline);
    MSDK_CHECK_POINTER(pPipeline.get(), MFX_ERR_MEMORY_ALLOC);

    sts = pPipeline->Init(&Params);
    if (sts < MFX_ERR_NONE) return 1;

    if (Params.pStrLogFile) {
        free(Params.pStrLogFile);
        Params.pStrLogFile = NULL;
    }

    pPipeline->SetAbortFlagPointer(&g_signal_abort);
    set_signal_handler();
    
    if (MFX_ERR_NONE != (sts = pPipeline->CheckCurrentVideoParam())) {
        return sts;
    }

    for (;;) {
        sts = pPipeline->Run();

        if (MFX_ERR_DEVICE_LOST == sts || MFX_ERR_DEVICE_FAILED == sts) {
            pPipeline->PrintMes(QSV_LOG_ERROR, _T("\nERROR: Hardware device was lost or returned an unexpected error. Recovering...\n"));
            sts = pPipeline->ResetDevice();
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, 1);

            sts = pPipeline->ResetMFXComponents(&Params);
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, 1);
            continue;
        } else {
            if (sts < MFX_ERR_NONE) return 1;
            break;
        }
    }

    pPipeline->Close();  
    pPipeline->PrintMes(QSV_LOG_INFO, _T("\nProcessing finished\n"));

    return sts;
}

int _tmain(int argc, TCHAR *argv[]) {
    int ret = 0;
    if (0 != (ret = run(argc, argv))) {
        qsv_print_stderr(QSV_LOG_ERROR, _T("QSVEncC.exe finished with error!\n"));
    }
    return ret;
}
