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
#include <filesystem>
#include <iomanip>
#include <set>
#include <vector>
#include <numeric>
#include <algorithm>
#include <ctime>
#include "rgy_osdep.h"
#include "rgy_filesystem.h"
#if defined(_WIN32) || defined(_WIN64)
#include <shellapi.h>
#endif

#include "qsv_pipeline.h"
#include "qsv_cmd.h"
#include "qsv_prm.h"
#include "qsv_query.h"
#include "rgy_version.h"
#include "rgy_avutil.h"
#include "rgy_codepage.h"
#include "rgy_resource.h"
#include "rgy_env.h"
#include "rgy_opencl.h"

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

static void show_help() {
    _ftprintf(stdout, _T("%s\n"), encoder_help().c_str());
}

static void show_option_list() {
    show_version();

    std::vector<std::string> optList;
    for (const auto &optHelp : createOptionList()) {
        optList.push_back(optHelp.first);
    }
    std::sort(optList.begin(), optList.end());

    _ftprintf(stdout, _T("Option List:\n"));
    for (const auto &optHelp : optList) {
        _ftprintf(stdout, _T("--%s\n"), char_to_tstring(optHelp).c_str());
    }
}

static int writeFeatureList(tstring filename, const QSVDeviceNum deviceNum, bool for_auo, RGYParamLogLevel& loglevel, bool parallel, FeatureListStrType type = FEATURE_LIST_STR_TYPE_UNKNOWN) {
    auto log = std::make_shared<RGYLog>(nullptr, loglevel);
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

    _ftprintf(stderr, _T("%s...\n"), (bUseJapanese) ? _T("QSVの情報を取得しています") : _T("Checking for QSV"));

    if (type == FEATURE_LIST_STR_TYPE_HTML) {
        print_tstring(header, false);
        print_tstring(_T("<h1>QSVEncC ") + tstring((bUseJapanese) ? _T("情報") : _T("Check Features")) + _T("</h1>\n<div class=page_top>\n"), false);
    }
    print_tstring(GetQSVEncVersion(), true, _T("<span style=font-size:small>"), _T("</span>"));

    if (type == FEATURE_LIST_STR_TYPE_HTML) {
        print_tstring(_T("<hr>\n"), false);
    }
    tstring environmentInfo = getEnviromentInfo((int)deviceNum);
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
        mfxVersion lib = (impl_type) ? get_mfx_libsw_version() : get_mfx_libhw_version(deviceNum, loglevel);
        const TCHAR *impl_str = (impl_type) ?  _T("Software") : _T("Hardware");
        if (!check_lib_version(lib, test)) {
            if (impl_type == 0) {
                if (type == FEATURE_LIST_STR_TYPE_HTML) {
                    print_tstring((bUseJapanese) ? _T("<b>QSVが使用できません。</b>") : _T("<b>QSV unavailable.</b>"), false);
                    char buffer[1024] = { 0 };
                    getCPUName(buffer, _countof(buffer));
                    tstring cpuname = char_to_tstring(buffer);
                    cpuname = cpuname.substr(cpuname.find(_T("Intel ")) + _tcslen(_T("Intel ")));
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
                                _T("「グラフィックスの仕様」のところの「インテル クイック・シンク・ビデオ」が「Yes」になっているかで確認できます。<br>\n")), false);
                            print_tstring(tstring(_T("<li>QSV利用に必要なIntel GPUがPCで認識されているか確認してください。<br>\n")
                                _T("同梱の「デバイスマネージャを開く」をダブルクリックし、<br>\n")
                                _T("デバイスマネージャの画面の「ディスプレイアダプタ」をクリックして")
                                _T("「Intel HD Graphics ～～」などとIntelのGPUが表示されていれば問題ありません。<br>\n")
                                _T("Intel GPU以外にGPUが搭載されている場合、ここでIntel GPUが表示されない場合があります。\n")
                                _T("この場合、BIOS(UEFI)の「CPU Graphics Multi-Monitor」を有効(Enable)にする必要があります。<br>\n")), false);
                            print_tstring(tstring(_T("<li>Intel GPUのドライバがWindows Update経由でインストールされた場合など、Intel ドライバのインストールが不完全な場合に正しく動作しないことがあります。<br>\n")
                                _T("<a target=\"_blank\" href=\"https://downloadcenter.intel.com/ja/search?keyword=") + cpuname + tstring(_T("\">こちらのリンク</a>から")) +
                                getOSVersion() + _T(" ") + tstring(rgy_is_64bit_os() ? _T("64bit") : _T("32bit")) + _T("用のドライバをダウンロードし、インストールしてみて下さい。<br>\n")), false);
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
                                _T(" whether \"Intel Quick Sync Video\" in \"Graphics Specifications\" says \"Yes\".<br>\n")), false);
                            print_tstring(tstring(_T("<li>Please check for device manager if Intel GPU is recognized under \"Display Adapter\".<br>\n")
                                _T("If you have discrete GPU on your PC, Intel GPU might not be shown.\n")
                                _T("For that case, you need yto enable \"CPU Graphics Multi-Monitor\" in your BIOS(UEFI).<br>\n")), false);
                            print_tstring(tstring(_T("<li>Sometimes Intel GPU driver is not installed properlly, especially when it is installed from Windows Update.<br>\n")
                                _T("Please install Intel GPU driver for") + getOSVersion() + _T(" ") + tstring(rgy_is_64bit_os() ? _T("64bit") : _T("32bit")) + _T(" ")
                                _T("<a target=\"_blank\" href=\"https://downloadcenter.intel.com/search?keyword=") + cpuname + tstring(_T("\">from here</a>")) +
                                _T(" and reinstall the driver.<br>\n")), false);
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
            const auto codec_feature_list = (for_auo) ? MakeFeatureListStr(deviceNum, type, make_vector(CODEC_LIST_AUO), log, parallel) : MakeFeatureListStr(deviceNum, type, log, parallel);
            if (codec_feature_list.size() == 0) {
                if (type == FEATURE_LIST_STR_TYPE_HTML) {
                    print_tstring((bUseJapanese) ? _T("<b>QSVが使用できません。</b><br>") : _T("<b>QSV unavailable.</b><br>"), false);

                    char buffer[1024] = { 0 };
                    getCPUName(buffer, _countof(buffer));
                    tstring cpuname = char_to_tstring(buffer);
                    cpuname = cpuname.substr(cpuname.find(_T("Intel ")) + _tcslen(_T("Intel ")));
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
                print_tstring(strsprintf(_T("%s API v%d.%02d\n\n"), impl_str, lib.Major, lib.Minor), true);
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
                    const auto vppFeatures = MakeVppFeatureStr(deviceNum, type, log);
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
                    const auto decFeatures = MakeDecFeatureStr(deviceNum, type, log);
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
        if (32 <= (size_t)FindExecutable(filename.c_str(), nullptr, exePath) && _tcslen(exePath) && rgy_file_exists(exePath)) {
            ShellExecute(NULL, _T("open"), filename.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
        }
#endif //#if defined(_WIN32) || defined(_WIN64)
    }
    return 0;
}

int ParseDeviceOption(const TCHAR *option_name, const TCHAR *arg1, QSVDeviceNum& deviceNum) {
    if (0 == _tcscmp(option_name, _T("device"))) {
        if (arg1[0] != _T('-')) {
            int value = 0;
            if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_qsv_device, arg1))) {
                deviceNum = (QSVDeviceNum)value;
            } else {
                print_cmd_error_invalid_value(option_name, arg1, list_qsv_device);
                return 1;
            }
        }
        return 0;
    }
    return 0;
}

int parse_print_options(const TCHAR *option_name, const TCHAR *arg1, const QSVDeviceNum deviceNum, RGYParamLogLevel& loglevel) {

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
    if (IS_OPTION("option-list")) {
        show_option_list();
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-environment"))) {
        show_version();
        _ftprintf(stdout, _T("%s"), getEnviromentInfo((int)deviceNum).c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-environment-auo"))) {
        show_version();
        _ftprintf(stdout, _T("%s"), getEnviromentInfo((int)deviceNum).c_str());
        mfxVersion lib = get_mfx_libhw_version(QSVDeviceNum::AUTO, loglevel);
        mfxVersion test = { 0, 1 };
        if (check_lib_version(lib, test)) {
            _ftprintf(stdout, _T("Media SDK Version: Hardware API v%d.%02d\n\n"), lib.Major, lib.Minor);
        }
        for (auto& str : getDeviceNameList()) {
            _ftprintf(stdout, _T("%s\n"), str.c_str());
        }
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-features"))) {
        tstring output = (arg1[0] != _T('-')) ? arg1 : _T("");
        writeFeatureList(output, deviceNum, false, loglevel, loglevel.get(RGY_LOGT_DEV) > RGY_LOG_DEBUG);
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-features-auo"))) {
        writeFeatureList(_T(""), deviceNum, true, loglevel, loglevel.get(RGY_LOGT_DEV) > RGY_LOG_DEBUG);
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-features-html"))) {
        tstring output = (arg1[0] != _T('-')) ? arg1 : _T("");
        writeFeatureList(output, deviceNum, false, loglevel, loglevel.get(RGY_LOGT_DEV) > RGY_LOG_DEBUG, FEATURE_LIST_STR_TYPE_HTML);
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-hw"))
        || 0 == _tcscmp(option_name, _T("hw-check"))) //互換性のため
    {
        mfxVersion ver = { 0, 1 };
        if (check_lib_version(get_mfx_libhw_version(deviceNum, loglevel), ver) != 0) {
            tstring deviceName = (deviceNum == QSVDeviceNum::AUTO) ? _T("auto") : strsprintf(_T("%d"), (int)deviceNum);
            _ftprintf(stdout, _T("Success: QuickSyncVideo (hw encoding) available\n"));
            _ftprintf(stdout, _T("Supported Encode Codecs for device %s:\n"), deviceName.c_str());
            auto log = std::make_shared<RGYLog>(nullptr, loglevel);
            const auto encodeFeature = MakeFeatureListPerCodec(
                deviceNum, { MFX_RATECONTROL_CQP }, ENC_CODEC_LISTS, log, loglevel.get(RGY_LOGT_DEV) > RGY_LOG_DEBUG);
            for (auto& enc : encodeFeature) {
                if (enc.feature.count(MFX_RATECONTROL_CQP) > 0
                    && (enc.feature.at(MFX_RATECONTROL_CQP) & ENC_FEATURE_CURRENT_RC) != 0) {
                    _ftprintf(stdout, _T("%s %s\n"), CodecToStr(enc.codec).c_str(), enc.lowPwer ? _T("FF") : _T("PG"));
                }
            }
            return 1;
        } else {
            _ftprintf(stdout, _T("Error: QuickSyncVideo (hw encoding) unavailable\n"));
            return -1;
        }
    }
    if (0 == _tcscmp(option_name, _T("lib-check"))
        || 0 == _tcscmp(option_name, _T("check-lib"))) {
        mfxVersion test = { 0, 1 };
        mfxVersion hwlib = get_mfx_libhw_version(deviceNum, loglevel);
        mfxVersion swlib = get_mfx_libsw_version();
        show_version();
#ifdef _M_IX86
        const TCHAR *dll_platform = _T("32");
#else
        const TCHAR *dll_platform = _T("64");
#endif
        if (check_lib_version(hwlib, test)) {
            _ftprintf(stdout, _T("libmfxhw%s.dll : v%d.%02d\n"), dll_platform, hwlib.Major, hwlib.Minor);
            return 1;
        } else {
            _ftprintf(stdout, _T("libmfxhw%s.dll : ----\n"), dll_platform);
            return -1;
        }
    }
    if (0 == _tcscmp(option_name, _T("check-impl"))) {
        tstring str;
        const auto implCount = GetImplListStr(str);
        if (implCount > 0) {
            _ftprintf(stdout, _T("%s\n"), str.c_str());
            return 1;
        } else {
            _ftprintf(stdout, _T("Error: VPL impl unavailable!\n"));
            return -1;
        }
    }
    if (0 == _tcscmp(option_name, _T("check-clinfo"))) {
        tstring str = getOpenCLInfo(CL_DEVICE_TYPE_GPU);
        _ftprintf(stdout, _T("%s\n"), str.c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-device"))) {
        auto devs = getDeviceNameList();
        if (devs.size() > 0) {
            for (auto& str : devs) {
                _ftprintf(stdout, _T("%s\n"), str.c_str());
            }
            return 1;
        } else {
            return -1;
        }
    }
#if ENABLE_AVSW_READER
    if (0 == _tcscmp(option_name, _T("check-avcodec-dll"))) {
        const auto ret = check_avcodec_dll();
        _ftprintf(stdout, _T("%s\n"), ret ? _T("yes") : _T("no"));
        if (!ret) {
            _ftprintf(stdout, _T("%s\n"), error_mes_avcodec_dll_not_found().c_str());
        }
        return ret ? 1 : -1;
    }
    if (0 == _tcscmp(option_name, _T("check-avversion"))) {
        _ftprintf(stdout, _T("%s\n"), getAVVersions().c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-codecs"))) {
        _ftprintf(stdout, _T("Video\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC), { AVMEDIA_TYPE_VIDEO }).c_str());
        _ftprintf(stdout, _T("\nAudio\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC | RGY_AVCODEC_ENC), { AVMEDIA_TYPE_AUDIO }).c_str());
        _ftprintf(stdout, _T("\nSbutitles\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC | RGY_AVCODEC_ENC), { AVMEDIA_TYPE_SUBTITLE }).c_str());
        _ftprintf(stdout, _T("\nData / Attachment\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC | RGY_AVCODEC_ENC), { AVMEDIA_TYPE_DATA, AVMEDIA_TYPE_ATTACHMENT }).c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-encoders"))) {
        _ftprintf(stdout, _T("Audio\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_ENC), { AVMEDIA_TYPE_AUDIO }).c_str());
        _ftprintf(stdout, _T("\nSbutitles\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_ENC), { AVMEDIA_TYPE_SUBTITLE }).c_str());
        _ftprintf(stdout, _T("\nData / Attachment\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_ENC), { AVMEDIA_TYPE_DATA, AVMEDIA_TYPE_ATTACHMENT }).c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-decoders"))) {
        _ftprintf(stdout, _T("Video\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC), { AVMEDIA_TYPE_VIDEO }).c_str());
        _ftprintf(stdout, _T("\nAudio\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC), { AVMEDIA_TYPE_AUDIO }).c_str());
        _ftprintf(stdout, _T("\nSbutitles\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC), { AVMEDIA_TYPE_SUBTITLE }).c_str());
        _ftprintf(stdout, _T("\nData / Attachment\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC), { AVMEDIA_TYPE_DATA, AVMEDIA_TYPE_ATTACHMENT }).c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-profiles"))) {
        auto list = getAudioPofileList(arg1);
        if (list.size() == 0) {
            _ftprintf(stdout, _T("Failed to find codec name \"%s\"\n"), arg1);
        } else {
            _ftprintf(stdout, _T("profile name for \"%s\"\n"), arg1);
            for (const auto& name : list) {
                _ftprintf(stdout, _T("  %s\n"), name.c_str());
            }
        }
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-protocols"))) {
        _ftprintf(stdout, _T("%s\n"), getAVProtocols().c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-avdevices"))) {
        _ftprintf(stdout, _T("%s\n"), getAVDevices().c_str());
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

#if defined(_WIN32) || defined(_WIN64)
static tstring getErrorFmtStr(uint32_t err) {
    TCHAR errmes[4097];
    FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, NULL, err, NULL, errmes, _countof(errmes), NULL);
    return errmes;
}

static int run_on_os_codepage() {
    auto exepath = getExePath();
    auto tmpexe = std::filesystem::path(PathRemoveExtensionS(exepath));
    tmpexe += strsprintf(_T("A_%x"), GetCurrentProcessId());
    tmpexe += std::filesystem::path(exepath).extension();
    std::filesystem::copy_file(exepath, tmpexe, std::filesystem::copy_options::overwrite_existing);

    SetLastError(0);
    HANDLE handle = BeginUpdateResourceW(tmpexe.wstring().c_str(), FALSE);
    if (handle == NULL) {
        auto lasterr = GetLastError();
        _ftprintf(stderr, _T("Failed to create temporary exe file: [%d] %s.\n"), lasterr, getErrorFmtStr(lasterr).c_str());
        return 1;
    }
    void *manifest = nullptr;
    int size = getEmbeddedResource(&manifest,_T("APP_OSCODEPAGE_MANIFEST"), _T("EXE_DATA"), NULL);
    if (size == 0) {
        _ftprintf(stderr, _T("Failed to load manifest for OS codepage mode.\n"));
        return 1;
    }
    SetLastError(0);
    if (!UpdateResourceW(handle, RT_MANIFEST, CREATEPROCESS_MANIFEST_RESOURCE_ID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), manifest, size)) {
        auto lasterr = GetLastError();
        _ftprintf(stderr, _T("Failed to update manifest for ansi mode: [%d] %s.\n"), lasterr, getErrorFmtStr(lasterr).c_str());
        return 1;
    }
    SetLastError(0);
    if (!EndUpdateResourceW(handle, FALSE)) {
        auto lasterr = GetLastError();
        _ftprintf(stderr, _T("Failed to finish update manifest for OS codepage mode: [%d] %s.\n"), lasterr, getErrorFmtStr(lasterr).c_str());
        return 1;
    }

    const auto commandline = str_replace(str_replace(GetCommandLineW(),
        std::filesystem::path(exepath).filename(), std::filesystem::path(tmpexe).filename()),
        CODEPAGE_CMDARG, CODEPAGE_CMDARG_APPLIED);

    int ret = 0;
    try {
        DWORD flags = 0; // CREATE_NO_WINDOW;

        HANDLE hStdIn, hStdOut, hStdErr;
        DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_INPUT_HANDLE),  GetCurrentProcess(), &hStdIn,  0, TRUE, DUPLICATE_SAME_ACCESS);
        DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_OUTPUT_HANDLE), GetCurrentProcess(), &hStdOut, 0, TRUE, DUPLICATE_SAME_ACCESS);
        DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_ERROR_HANDLE),  GetCurrentProcess(), &hStdErr, 0, TRUE, DUPLICATE_SAME_ACCESS);

        SECURITY_ATTRIBUTES sa;
        memset(&sa, 0, sizeof(SECURITY_ATTRIBUTES));
        sa.nLength = sizeof(sa);
        sa.lpSecurityDescriptor = NULL;
        sa.bInheritHandle = TRUE; //TRUEでハンドルを引き継ぐ

        STARTUPINFO si;
        memset(&si, 0, sizeof(STARTUPINFO));
        si.cb = sizeof(STARTUPINFO);
        //si.dwFlags |= STARTF_USESHOWWINDOW;
        si.dwFlags |= STARTF_USESTDHANDLES;
        //si.wShowWindow |= SW_SHOWMINNOACTIVE;
        si.hStdInput = hStdIn;
        si.hStdOutput = hStdOut;
        si.hStdError = hStdErr;

        PROCESS_INFORMATION pi;
        memset(&pi, 0, sizeof(PROCESS_INFORMATION));

        SetLastError(0);
        if (CreateProcess(nullptr, (LPWSTR)commandline.c_str(), &sa, nullptr, TRUE, flags, nullptr, nullptr, &si, &pi) == 0) {
            auto lasterr = GetLastError();
            _ftprintf(stderr, _T("Failed to run process in OS codepage mode: [%d] %s.\n"), lasterr, getErrorFmtStr(lasterr).c_str());
            ret = 1;
        } else {
            WaitForSingleObject(pi.hProcess, INFINITE);
            DWORD proc_ret = 0;
            if (GetExitCodeProcess(pi.hProcess, &proc_ret)) {
                ret = (int)proc_ret;
            }
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
        }
    } catch (...) {
        ret = 1;
    }
    std::filesystem::remove(tmpexe);
    return ret;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

RGY_ERR run_encode(sInputParams *params) {
    RGY_ERR sts = RGY_ERR_NONE; // return value check

    unique_ptr<CQSVPipeline> pPipeline(new CQSVPipeline);
    if (!pPipeline) {
        return RGY_ERR_MEMORY_ALLOC;
    }

    sts = pPipeline->Init(params);
    if (sts < RGY_ERR_NONE) return sts;

    pPipeline->SetAbortFlagPointer(&g_signal_abort);
    set_signal_handler();

    if (RGY_ERR_NONE != (sts = pPipeline->CheckCurrentVideoParam())) {
        return sts;
    }

    if (RGY_ERR_NONE != (sts = pPipeline->Run())) {
        return sts;
    }

    pPipeline->Close();

    return sts;
}

RGY_ERR run_benchmark(sInputParams *params) {
    using namespace std;
    RGY_ERR sts = RGY_ERR_NONE;
    tstring benchmarkLogFile = params->common.outputFilename;

    //テストする解像度
    const vector<pair<int, int>> test_resolution = { { 1920, 1080 }, { 1280, 720 } };

    //初回出力
    {
        params->input.dstWidth = test_resolution[0].first;
        params->input.dstHeight = test_resolution[0].second;
        params->nTargetUsage = MFX_TARGETUSAGE_BEST_SPEED;

        unique_ptr<CQSVPipeline> pPipeline(new CQSVPipeline);
        if (!pPipeline) {
            return RGY_ERR_MEMORY_ALLOC;
        }

        sts = pPipeline->Init(params);
        if (sts < RGY_ERR_NONE) return sts;

        pPipeline->SetAbortFlagPointer(&g_signal_abort);
        set_signal_handler();
        time_t current_time = time(NULL);
        struct tm *local_time = localtime(&current_time);

        TCHAR encode_info[4096] = { 0 };
        if (RGY_ERR_NONE != (sts = pPipeline->CheckCurrentVideoParam(encode_info, _countof(encode_info)))) {
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
            return RGY_ERR_INVALID_HANDLE;
        } else {
            fprintf(fp_bench, "Started benchmark on %d.%02d.%02d %2d:%02d:%02d\n",
                1900 + local_time->tm_year, local_time->tm_mon + 1, local_time->tm_mday, local_time->tm_hour, local_time->tm_min, local_time->tm_sec);
            fprintf(fp_bench, "Input File: %s\n", tchar_to_string(params->common.inputFilename).c_str());
            fprintf(fp_bench, "Basic parameters of the benchmark\n"
                              " (Target Usage and output resolution will be changed)\n");
            fprintf(fp_bench, "%s\n\n", tchar_to_string(encode_info).c_str());
            fprintf(fp_bench, "%s", tchar_to_string(enviroment_info).c_str());
            fprintf(fp_bench, "QSV: QSVEncC %s (%s) / API[%s]: v%d.%02d / %s\n",
                VER_STR_FILEVERSION, tchar_to_string(BUILD_ARCH_STR).c_str(), (hardware) ? "hw" : "sw", ver.Major, ver.Minor, tchar_to_string(MemTypeToStr(memtype)).c_str());
            fprintf(fp_bench, "\n");
            fclose(fp_bench);
        }
        basic_ofstream<TCHAR> benchmark_log_test_open(benchmarkLogFile, ios::out | ios::app);
        if (!benchmark_log_test_open.good()) {
            pPipeline->PrintMes(RGY_LOG_ERROR, _T("\nERROR: failed opening benchmark result file.\n"));
            return RGY_ERR_INVALID_HANDLE;
        }
        benchmark_log_test_open << ss.str();
        benchmark_log_test_open.close();

        sts = pPipeline->Run();

        EncodeStatusData data = { 0 };
        sts = pPipeline->GetEncodeStatusData(&data);

        pPipeline->Close();
    }

    //ベンチマークの集計データ
    typedef struct benchmark_t {
        pair<int, int> resolution;
        int targetUsage;
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

    for (uint32_t i = 0; RGY_ERR_NONE == sts && !g_signal_abort && i < list_target_quality.size(); i++) {
        params->nTargetUsage = list_target_quality[i].value;
        vector<benchmark_t> benchmark_per_target_usage;
        for (const auto& resolution : test_resolution) {
            params->input.dstWidth = resolution.first;
            params->input.dstHeight = resolution.second;

            unique_ptr<CQSVPipeline> pPipeline(new CQSVPipeline);
            if (!pPipeline) {
                return RGY_ERR_MEMORY_ALLOC;
            }

            if (RGY_ERR_NONE != (sts = pPipeline->Init(params))) {
                break;
            }

            pPipeline->SetAbortFlagPointer(&g_signal_abort);
            set_signal_handler();
            if (RGY_ERR_NONE != (sts = pPipeline->CheckCurrentVideoParam())) {
                return sts;
            }
            if (RGY_ERR_NONE != (sts = pPipeline->Run())) {
                return sts;
            }

            EncodeStatusData data = { 0 };
            sts = pPipeline->GetEncodeStatusData(&data);

            pPipeline->Close();

            benchmark_t result;
            result.resolution      = resolution;
            result.targetUsage     = list_target_quality[i].value;
            result.fps             = data.encodeFps;
            result.bitrate         = data.bitrateKbps;
            result.cpuUsagePercent = data.CPUUsagePercent;
            benchmark_per_target_usage.push_back(result);

            _ftprintf(stderr, _T("\n"));

            if (RGY_ERR_NONE != sts || g_signal_abort)
                break;
        }

        benchmark_result.push_back(benchmark_per_target_usage);
    }

    //結果を出力
    if (RGY_ERR_NONE == sts && benchmark_result.size()) {
        basic_stringstream<TCHAR> ss;

        uint32_t maxLengthOfTargetUsageDesc = 0;
        for (uint32_t i = 0; i < list_target_quality.size(); i++) {
            maxLengthOfTargetUsageDesc = max(maxLengthOfTargetUsageDesc, (uint32_t)_tcslen(list_target_quality[i].desc));
        }

        FILE *fp_bench = NULL;
        if (_tfopen_s(&fp_bench, benchmarkLogFile.c_str(), _T("a")) || NULL == fp_bench) {
            _ftprintf(stderr, _T("\nERROR: failed opening benchmark result file.\n"));
            return RGY_ERR_INVALID_HANDLE;
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
    _tsetlocale(LC_CTYPE, _T(".UTF8"));
#endif //#if defined(_WIN32) || defined(_WIN64)

    if (argc == 1) {
        show_version();
        show_help();
        return 1;
    }

#if defined(_WIN32) || defined(_WIN64)
    if (GetACP() == CODE_PAGE_UTF8) {
        bool switch_to_os_cp = false;
        for (int iarg = 1; iarg < argc; iarg++) {
            if (iarg + 1 < argc
                && _tcscmp(argv[iarg + 0], CODEPAGE_CMDARG) == 0) {
                if (_tcscmp(argv[iarg + 1], _T("os")) == 0) {
                    switch_to_os_cp = true;
                } else if (_tcscmp(argv[iarg + 1], _T("utf8")) == 0) {
                    switch_to_os_cp = false;
                } else {
                    _ftprintf(stderr, _T("Unknown option for %s.\n"), CODEPAGE_CMDARG);
                    return 1;
                }
            }
        }
        if (switch_to_os_cp) {
            return run_on_os_codepage();
        }
    }
#endif //#if defined(_WIN32) || defined(_WIN64)

    //log-levelの取得
    RGYParamLogLevel loglevelPrint(RGY_LOG_ERROR);
    for (int iarg = 1; iarg < argc - 1; iarg++) {
        if (tstring(argv[iarg]) == _T("--log-level")) {
            parse_log_level_param(argv[iarg], argv[iarg + 1], loglevelPrint);
            break;
        }
    }

    // device IDの取得
    QSVDeviceNum deviceNum = QSVDeviceNum::AUTO;
    for (int iarg = 1; iarg < argc; iarg++) {
        const TCHAR *option_name = nullptr;
        if (argv[iarg][0] == _T('-')) {
            if (argv[iarg][1] == _T('\0')) {
                continue;
            } else if (argv[iarg][1] == _T('-')) {
                option_name = &argv[iarg][2];
            } else if (argv[iarg][2] == _T('\0')) {
                if (nullptr == (option_name = cmd_short_opt_to_long(argv[iarg][1]))) {
                    continue;
                }
            }
        }
        if (option_name != nullptr) {
            int ret = ParseDeviceOption(option_name, (iarg + 1 < argc) ? argv[iarg + 1] : _T(""), deviceNum);
            if (ret != 0) {
                return ret == 1 ? 0 : 1;
            }
        }
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
                    continue;
                }
            }
        }
        if (option_name != nullptr) {
            int ret = parse_print_options(option_name, (iarg+1 < argc) ? argv[iarg+1] : _T(""), deviceNum, loglevelPrint);
            if (ret != 0) {
                return ret == 1 ? 0 : 1;
            }
        }
    }

    //optionファイルの読み取り
    std::vector<tstring> argvCnfFile;
    for (int iarg = 1; iarg < argc; iarg++) {
        const TCHAR *option_name = nullptr;
        if (argv[iarg][0] == _T('-')) {
            if (argv[iarg][1] == _T('\0')) {
                continue;
            } else if (argv[iarg][1] == _T('-')) {
                option_name = &argv[iarg][2];
            }
        }
        if (option_name != nullptr
            && tstring(option_name) == _T("option-file")) {
            if (iarg + 1 >= argc) {
                _ftprintf(stderr, _T("option file name is not specified.\n"));
                return -1;
            }
            tstring cnffile = argv[iarg + 1];
            vector_cat(argvCnfFile, cmd_from_config_file(argv[iarg + 1]));
        }
    }

    vector<const TCHAR *> argvCopy(argv, argv + argc);
    //optionファイルのパラメータを追加
    for (size_t i = 0; i < argvCnfFile.size(); i++) {
        if (argvCnfFile[i].length() > 0) {
            argvCopy.push_back(argvCnfFile[i].c_str());
        }
    }
    argvCopy.push_back(_T(""));

    sInputParams Params;
    int ret = parse_cmd(&Params, argvCopy.data(), (int)argvCopy.size()-1);
    if (ret >= 1) {
        return 1;
    }

#if defined(_WIN32) || defined(_WIN64)
    //set stdin to binary mode when using pipe input
    if (Params.common.inputFilename == _T("-")) {
        if (_setmode( _fileno( stdin ), _O_BINARY ) == 1) {
            _ftprintf(stderr, _T("Error: failed to switch stdin to binary mode."));
            return 1;
        }
    }

    //set stdout to binary mode when using pipe output
    if (Params.common.outputFilename == _T("-")) {
        if (_setmode( _fileno( stdout ), _O_BINARY ) == 1) {
            _ftprintf(stderr, _T("Error: failed to switch stdout to binary mode."));
            return 1;
        }
    }

    if (check_locale_is_ja()) {
        _tsetlocale(LC_ALL, _T("Japanese"));
    }
#endif //#if defined(_WIN32) || defined(_WIN64)
    if (Params.ctrl.processMonitorDevUsageReset) {
        return processMonitorRGYDeviceResetEntry();
    }
    if (Params.ctrl.processMonitorDevUsage) {
        return processMonitorRGYDeviceUsage((int)Params.device);
    }
    if (Params.bBenchmark) {
        return run_benchmark(&Params);
    }
    unique_ptr<CQSVPipeline> pPipeline(new CQSVPipeline);
    if (!pPipeline) {
        return MFX_ERR_MEMORY_ALLOC;
    }

    auto sts = pPipeline->Init(&Params);
    if (sts < RGY_ERR_NONE) return sts;

    pPipeline->SetAbortFlagPointer(&g_signal_abort);
    set_signal_handler();

    if ((sts = pPipeline->CheckCurrentVideoParam()) != RGY_ERR_NONE) {
        return sts;
    }

    if ((sts = pPipeline->Run()) != RGY_ERR_NONE) {
        return sts;
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
    return ret;
}
