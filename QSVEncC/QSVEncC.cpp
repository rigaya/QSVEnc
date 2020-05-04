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
            const auto codec_feature_list = (for_auo) ? MakeFeatureListStr(type, make_vector(CODEC_LIST_AUO)) : MakeFeatureListStr(type);
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
                    const auto vppFeatures = MakeVppFeatureStr(type);
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
                    const auto decFeatures = MakeDecFeatureStr(type);
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
    if (0 == _tcscmp(option_name, _T("check-environment-auo"))) {
        show_version();
        _ftprintf(stdout, _T("%s"), getEnviromentInfo(false).c_str());
        mfxVersion lib = get_mfx_libhw_version();
        mfxVersion test = { 0, 1 };
        if (check_lib_version(lib, test)) {
            _ftprintf(stdout, _T("Media SDK Version: Hardware API v%d.%d\n\n"), lib.Major, lib.Minor);
        }
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
            fprintf(fp_bench, "Input File: %s\n", tchar_to_string(params->common.inputFilename).c_str());
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

    for (uint32_t i = 0; MFX_ERR_NONE == sts && !g_signal_abort && i < list_target_quality.size(); i++) {
        params->nTargetUsage = list_target_quality[i].value;
        vector<benchmark_t> benchmark_per_target_usage;
        for (const auto& resolution : test_resolution) {
            params->input.dstWidth = resolution.first;
            params->input.dstHeight = resolution.second;

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
            result.targetUsage     = list_target_quality[i].value;
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
                    continue;
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

    sInputParams Params;

    vector<const TCHAR *> argvCopy(argv, argv + argc);
    argvCopy.push_back(_T(""));

    int ret = parse_cmd(&Params, argvCopy.data(), (mfxU8)argc);
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
    return ret;
}
