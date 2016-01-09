//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <stdio.h>
#include <vector>
#include <numeric>
#include <memory>
#include <sstream>
#include <algorithm>
#include <type_traits>
#if (_MSC_VER >= 1800)
#include <Windows.h>
#include <VersionHelpers.h>
#endif
#ifndef _MSC_VER
#include <sys/sysinfo.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <iconv.h>
#endif
#include "mfxstructures.h"
#include "mfxvideo.h"
#include "mfxvideo++.h"
#include "mfxplugin.h"
#include "mfxplugin++.h"
#include "mfxjpeg.h"
#include "qsv_tchar.h"
#include "qsv_util.h"
#include "qsv_prm.h"
#include "qsv_plugin.h"
#include "ram_speed.h"

#ifdef LIBVA_SUPPORT
#include "hw_device.h"
#include "vaapi_device.h"
#include "vaapi_allocator.h"
#endif //#ifdef LIBVA_SUPPORT

#pragma warning (push)
#pragma warning (disable: 4100)
#if defined(_WIN32) || defined(_WIN64)
unsigned int wstring_to_string(const wchar_t *wstr, std::string& str, uint32_t codepage) {
    uint32_t flags = (codepage == CP_UTF8) ? 0 : WC_NO_BEST_FIT_CHARS;
    int multibyte_length = WideCharToMultiByte(codepage, flags, wstr, -1, nullptr, 0, nullptr, nullptr);
    str.resize(multibyte_length, 0);
    if (0 == WideCharToMultiByte(codepage, flags, wstr, -1, &str[0], multibyte_length, nullptr, nullptr)) {
        str.clear();
        return 0;
    }
    return multibyte_length;
}
#else
unsigned int wstring_to_string(const wchar_t *wstr, std::string& str, uint32_t codepage) {
    auto ic = iconv_open("UTF-8", "wchar_t"); //to, from
    auto input_len = wcslen(wstr);
    auto output_len = input_len * 4;
    str.resize(output_len, 0);
    char *outbuf = &str[0];
    iconv(ic, (char **)&wstr, &input_len, &outbuf, &output_len);
    return output_len;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

std::string wstring_to_string(const wchar_t *wstr, uint32_t codepage) {
    std::string str;
    wstring_to_string(wstr, str, codepage);
    return str;
}

std::string wstring_to_string(const std::wstring& wstr, uint32_t codepage) {
    std::string str;
    wstring_to_string(wstr.c_str(), str, codepage);
    return str;
}

unsigned int tchar_to_string(const TCHAR *tstr, std::string& str, uint32_t codepage) {
#if UNICODE
    return wstring_to_string(tstr, str, codepage);
#else
    str = std::string(tstr);
    return (unsigned int)str.length();
#endif
}

std::string tchar_to_string(const TCHAR *tstr, uint32_t codepage) {
    std::string str;
    tchar_to_string(tstr, str, codepage);
    return str;
}

std::string tchar_to_string(const tstring& tstr, uint32_t codepage) {
    std::string str;
    tchar_to_string(tstr.c_str(), str, codepage);
    return str;
}

#if defined(_WIN32) || defined(_WIN64)
unsigned int char_to_wstring(std::wstring& wstr, const char *str, uint32_t codepage) {
    int widechar_length = MultiByteToWideChar(codepage, 0, str, -1, nullptr, 0);
    wstr.resize(widechar_length, 0);
    if (0 == MultiByteToWideChar(codepage, 0, str, -1, &wstr[0], (int)wstr.size())) {
        wstr.clear();
        return 0;
    }
    return widechar_length;
}
#else
unsigned int char_to_wstring(std::wstring& wstr, const char *str, uint32_t codepage) {
    auto ic = iconv_open("wchar_t", "UTF-8"); //to, from
    auto input_len = strlen(str);
    std::vector<char> buf(input_len + 1);
    strcpy(buf.data(), str);
    auto output_len = input_len;
    wstr.resize(output_len, 0);
    char *inbuf = buf.data();
    char *outbuf = (char *)&wstr[0];
    iconv(ic, &inbuf, &input_len, &outbuf, &output_len);
    return output_len;
}
#endif //#if defined(_WIN32) || defined(_WIN64)
std::wstring char_to_wstring(const char *str, uint32_t codepage) {
    std::wstring wstr;
    char_to_wstring(wstr, str, codepage);
    return wstr;
}
std::wstring char_to_wstring(const std::string& str, uint32_t codepage) {
    std::wstring wstr;
    char_to_wstring(wstr, str.c_str(), codepage);
    return wstr;
}

unsigned int char_to_tstring(tstring& tstr, const char *str, uint32_t codepage) {
#if UNICODE
    return char_to_wstring(tstr, str, codepage);
#else
    tstr = std::string(str);
    return (unsigned int)tstr.length();
#endif
}

tstring char_to_tstring(const char *str, uint32_t codepage) {
    tstring tstr;
    char_to_tstring(tstr, str, codepage);
    return tstr;
}
tstring char_to_tstring(const std::string& str, uint32_t codepage) {
    tstring tstr;
    char_to_tstring(tstr, str.c_str(), codepage);
    return tstr;
}
std::string strsprintf(const char* format, ...) {
    va_list args;
    va_start(args, format);
    const size_t len = _vscprintf(format, args) + 1;

    std::vector<char> buffer(len, 0);
    vsprintf(buffer.data(), format, args);
    va_end(args);
    std::string retStr = std::string(buffer.data());
    return retStr;
}
#if defined(_WIN32) || defined(_WIN64)
std::wstring strsprintf(const WCHAR* format, ...) {
    va_list args;
    va_start(args, format);
    const size_t len = _vscwprintf(format, args) + 1;

    std::vector<WCHAR> buffer(len, 0);
    vswprintf(buffer.data(), format, args);
    va_end(args);
    std::wstring retStr = std::wstring(buffer.data());
    return retStr;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

std::string str_replace(std::string str, const std::string& from, const std::string& to) {
    std::string::size_type pos = 0;
    while(pos = str.find(from, pos), pos != std::string::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
    }
    return std::move(str);
}

#if defined(_WIN32) || defined(_WIN64)
std::wstring str_replace(std::wstring str, const std::wstring& from, const std::wstring& to) {
    std::wstring::size_type pos = 0;
    while (pos = str.find(from, pos), pos != std::wstring::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
    }
    return std::move(str);
}
#endif //#if defined(_WIN32) || defined(_WIN64)

#pragma warning (pop)

#if defined(_WIN32) || defined(_WIN64)
std::vector<std::wstring> split(const std::wstring &str, const std::wstring &delim) {
    std::vector<std::wstring> res;
    size_t current = 0, found, delimlen = delim.size();
    while (std::wstring::npos != (found = str.find(delim, current))) {
        res.push_back(std::wstring(str, current, found - current));
        current = found + delimlen;
    }
    res.push_back(std::wstring(str, current, str.size() - current));
    return res;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

std::vector<std::string> split(const std::string &str, const std::string &delim) {
    std::vector<std::string> res;
    size_t current = 0, found, delimlen = delim.size();
    while (std::string::npos != (found = str.find(delim, current))) {
        res.push_back(std::string(str, current, found - current));
        current = found + delimlen;
    }
    res.push_back(std::string(str, current, str.size() - current));
    return res;
}

tstring lstrip(const tstring& string, const TCHAR* trim) {
    auto result = string;
    auto left = string.find_first_not_of(trim);
    if (left != std::string::npos) {
        result = string.substr(left, 0);
    }
    return result;
}

tstring rstrip(const tstring& string, const TCHAR* trim) {
    auto result = string;
    auto right = string.find_last_not_of(trim);
    if (right != std::string::npos) {
        result = string.substr(0, right);
    }
    return result;
}

tstring trim(const tstring& string, const TCHAR* trim) {
    auto result = string;
    auto left = string.find_first_not_of(trim);
    if (left != std::string::npos) {
        auto right = string.find_last_not_of(trim);
        result = string.substr(left, right - left + 1);
    }
    return result;
}

std::string GetFullPath(const char *path) {
#if defined(_WIN32) || defined(_WIN64)
    if (PathIsRelativeA(path) == FALSE)
        return std::string(path);
#endif //#if defined(_WIN32) || defined(_WIN64)
    std::vector<char> buffer(strlen(path) + 1024, 0);
    _fullpath(buffer.data(), path, buffer.size());
    return std::string(buffer.data());
}
#if defined(_WIN32) || defined(_WIN64)
std::wstring GetFullPath(const WCHAR *path) {
    if (PathIsRelativeW(path) == FALSE)
        return std::wstring(path);
    
    std::vector<WCHAR> buffer(wcslen(path) + 1024, 0);
    _wfullpath(buffer.data(), path, buffer.size());
    return std::wstring(buffer.data());
}
#endif //#if defined(_WIN32) || defined(_WIN64)

bool check_ext(const TCHAR *filename, const std::vector<const char*>& ext_list) {
    const TCHAR *target = PathFindExtension(filename);
    if (target) {
        for (auto ext : ext_list) {
            if (0 == _tcsicmp(target, char_to_tstring(ext).c_str())) {
                return true;
            }
        }
    }
    return false;
}

bool qsv_get_filesize(const char *filepath, uint64_t *filesize) {
#if defined(_WIN32) || defined(_WIN64)
    WIN32_FILE_ATTRIBUTE_DATA fd = { 0 };
    bool ret = (GetFileAttributesExA(filepath, GetFileExInfoStandard, &fd)) ? true : false;
    *filesize = (ret) ? (((UINT64)fd.nFileSizeHigh) << 32) + (UINT64)fd.nFileSizeLow : NULL;
    return ret;
#else //#if defined(_WIN32) || defined(_WIN64)
    struct stat stat;
    FILE *fp = fopen(filepath, "rb");
    if (fp == NULL || fstat(fileno(fp), &stat)) {
        *filesize = 0;
        return 1;
    }
    if (fp) {
        fclose(fp);
    }
    *filesize = stat.st_size;
    return 0;
#endif //#if defined(_WIN32) || defined(_WIN64)
}

#if defined(_WIN32) || defined(_WIN64)
bool qsv_get_filesize(const WCHAR *filepath, UINT64 *filesize) {
    WIN32_FILE_ATTRIBUTE_DATA fd = { 0 };
    bool ret = (GetFileAttributesExW(filepath, GetFileExInfoStandard, &fd)) ? true : false;
    *filesize = (ret) ? (((UINT64)fd.nFileSizeHigh) << 32) + (UINT64)fd.nFileSizeLow : NULL;
    return ret;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

tstring qsv_memtype_str(mfxU16 memtype) {
    tstring str;
    if (memtype & MFX_MEMTYPE_INTERNAL_FRAME)         str += _T("internal,");
    if (memtype & MFX_MEMTYPE_EXTERNAL_FRAME)         str += _T("external,");
    if (memtype & MFX_MEMTYPE_OPAQUE_FRAME)           str += _T("opaque,");
    if (memtype & MFX_MEMTYPE_DXVA2_DECODER_TARGET)   str += _T("dxvadec,");
    if (memtype & MFX_MEMTYPE_DXVA2_PROCESSOR_TARGET) str += _T("dxvaproc,");
    if (memtype & MFX_MEMTYPE_SYSTEM_MEMORY)          str += _T("system,");
    if (memtype & MFX_MEMTYPE_FROM_ENCODE)            str += _T("enc,");
    if (memtype & MFX_MEMTYPE_FROM_DECODE)            str += _T("dec,");
    if (memtype & MFX_MEMTYPE_FROM_VPPIN)             str += _T("vppin,");
    if (memtype & MFX_MEMTYPE_FROM_VPPOUT)            str += _T("vppout,");
    if (memtype == 0)                                 str += _T("none,");
    return str.substr(0, str.length()-1);
}

int qsv_print_stderr(int log_level, const TCHAR *mes, HANDLE handle) {
#if defined(_WIN32) || defined(_WIN64)
    CONSOLE_SCREEN_BUFFER_INFO csbi = { 0 };
    static const WORD LOG_COLOR[] = {
        FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_BLUE, //水色
        FOREGROUND_INTENSITY | FOREGROUND_GREEN, //緑
        FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE,
        FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE,
        FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_RED, //黄色
        FOREGROUND_INTENSITY | FOREGROUND_RED //赤
    };
    if (handle == NULL) {
        handle = GetStdHandle(STD_ERROR_HANDLE);
    }
    if (handle && log_level != QSV_LOG_INFO) {
        GetConsoleScreenBufferInfo(handle, &csbi);
        SetConsoleTextAttribute(handle, LOG_COLOR[clamp(log_level, QSV_LOG_TRACE, QSV_LOG_ERROR) - QSV_LOG_TRACE] | (csbi.wAttributes & 0x00f0));
    }
    //このfprintfで"%"が消えてしまわないよう置換する
    int ret = _ftprintf(stderr, (nullptr == _tcschr(mes, _T('%'))) ? mes : str_replace(tstring(mes), _T("%"), _T("%%")).c_str());
    if (handle && log_level != QSV_LOG_INFO) {
        SetConsoleTextAttribute(handle, csbi.wAttributes); //元に戻す
    }
#else
    static const char *const LOG_COLOR[] = {
        "\x1b[36m", //水色
        "\x1b[32m", //緑
        "\x1b[39m", //デフォルト
        "\x1b[39m", //デフォルト
        "\x1b[33m", //黄色
        "\x1b[31m", //赤
    };
    int ret = _ftprintf(stderr, "%s%s%s", LOG_COLOR[clamp(log_level, QSV_LOG_TRACE, QSV_LOG_ERROR) - QSV_LOG_TRACE], mes, LOG_COLOR[QSV_LOG_INFO - QSV_LOG_TRACE]);
#endif //#if defined(_WIN32) || defined(_WIN64)
    fflush(stderr);
    return ret;
}

BOOL Check_HWUsed(mfxIMPL impl) {
    static const int HW_list[] = {
        MFX_IMPL_HARDWARE,
        MFX_IMPL_HARDWARE_ANY,
        MFX_IMPL_HARDWARE2,
        MFX_IMPL_HARDWARE3,
        MFX_IMPL_HARDWARE4,
        0
    };
    for (int i = 0; HW_list[i]; i++)
        if (HW_list[i] == (HW_list[i] & (int)impl))
            return TRUE;
    return FALSE;
}

int GetAdapterID(mfxIMPL impl) {
    return (std::max)(0, MFX_IMPL_BASETYPE(impl) - MFX_IMPL_HARDWARE_ANY);
}

int GetAdapterID(mfxSession session) {
    mfxIMPL impl;
    MFXQueryIMPL(session, &impl);
    return GetAdapterID(impl);
}

mfxVersion get_mfx_lib_version(mfxIMPL impl) {
    if (impl == MFX_IMPL_SOFTWARE) {
        return LIB_VER_LIST[0];
    }
    int i;
    for (i = 1; LIB_VER_LIST[i].Major; i++) {
        auto session_deleter = [](MFXVideoSession *session) { session->Close(); };
        std::unique_ptr<MFXVideoSession, decltype(session_deleter)> test(new MFXVideoSession(), session_deleter);
        mfxVersion ver;
        memcpy(&ver, &LIB_VER_LIST[i], sizeof(mfxVersion));
        mfxStatus sts = test->Init(impl, &ver);
        if (sts != MFX_ERR_NONE)
            break;
    }
    return LIB_VER_LIST[i-1];
}

mfxVersion get_mfx_libhw_version() {
    static const mfxU32 impl_list[] = {
        MFX_IMPL_HARDWARE_ANY | MFX_IMPL_VIA_D3D11,
        MFX_IMPL_HARDWARE_ANY,
        MFX_IMPL_HARDWARE,
    };
    mfxVersion test = { 0 };
    //Win7でD3D11のチェックをやると、
    //デスクトップコンポジションが切られてしまう問題が発生すると報告を頂いたので、
    //D3D11をWin8以降に限定
    for (int i = (check_OS_Win8orLater() ? 0 : 1); i < _countof(impl_list); i++) {
        test = get_mfx_lib_version(impl_list[i]);
        if (check_lib_version(test, MFX_LIB_VERSION_1_1))
            break;
    }
    return test;
}
bool check_if_d3d11_necessary() {
    bool check_d3d11 = (0 != check_lib_version(get_mfx_lib_version(MFX_IMPL_HARDWARE_ANY | MFX_IMPL_VIA_D3D11), MFX_LIB_VERSION_1_1));
    bool check_d3d9  = (0 != check_lib_version(get_mfx_lib_version(MFX_IMPL_HARDWARE_ANY | MFX_IMPL_VIA_D3D9), MFX_LIB_VERSION_1_1));

    return (check_d3d11 == true && check_d3d9 == false);
}
mfxVersion get_mfx_libsw_version() {
    return get_mfx_lib_version(MFX_IMPL_SOFTWARE);
}

BOOL check_lib_version(mfxVersion value, mfxVersion required) {
    if (value.Major < required.Major)
        return FALSE;
    if (value.Major > required.Major)
        return TRUE;
    if (value.Minor < required.Minor)
        return FALSE;
    return TRUE;
}

mfxU64 CheckVppFeaturesInternal(mfxSession session, mfxVersion mfxVer) {
    using namespace std;

    mfxU64 result = 0x00;
    result |= VPP_FEATURE_RESIZE;
    result |= VPP_FEATURE_DEINTERLACE;
    result |= VPP_FEATURE_DENOISE;
    result |= VPP_FEATURE_DETAIL_ENHANCEMENT;
    result |= VPP_FEATURE_PROC_AMP;
    if (!check_lib_version(mfxVer, MFX_LIB_VERSION_1_3))
        return result;

    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_13)) {
        result |= VPP_FEATURE_DEINTERLACE_AUTO;
        result |= VPP_FEATURE_DEINTERLACE_IT_MANUAL;
    }
    MFXVideoVPP vpp(session);

    mfxExtVPPDoUse vppDoUse;
    mfxExtVPPDoUse vppDoNotUse;
    mfxExtVPPFrameRateConversion vppFpsConv;
    mfxExtVPPImageStab vppImageStab;
    mfxExtVPPVideoSignalInfo vppVSI;
    mfxExtVPPRotation vppRotate;
    INIT_MFX_EXT_BUFFER(vppDoUse,     MFX_EXTBUFF_VPP_DOUSE);
    INIT_MFX_EXT_BUFFER(vppDoNotUse,  MFX_EXTBUFF_VPP_DONOTUSE);
    INIT_MFX_EXT_BUFFER(vppFpsConv,   MFX_EXTBUFF_VPP_FRAME_RATE_CONVERSION);
    INIT_MFX_EXT_BUFFER(vppImageStab, MFX_EXTBUFF_VPP_IMAGE_STABILIZATION);
    INIT_MFX_EXT_BUFFER(vppVSI,       MFX_EXTBUFF_VPP_VIDEO_SIGNAL_INFO);
    INIT_MFX_EXT_BUFFER(vppRotate,    MFX_EXTBUFF_VPP_ROTATION);

    vppFpsConv.Algorithm = MFX_FRCALGM_FRAME_INTERPOLATION;
    vppImageStab.Mode = MFX_IMAGESTAB_MODE_UPSCALE;
    vppVSI.In.TransferMatrix = MFX_TRANSFERMATRIX_BT601;
    vppVSI.Out.TransferMatrix = MFX_TRANSFERMATRIX_BT709;
    vppVSI.In.NominalRange = MFX_NOMINALRANGE_16_235;
    vppVSI.Out.NominalRange = MFX_NOMINALRANGE_0_255;
    vppRotate.Angle = MFX_ANGLE_180;

    vector<mfxExtBuffer*> buf;
    buf.push_back((mfxExtBuffer *)&vppDoUse);
    buf.push_back((mfxExtBuffer *)&vppDoNotUse);
    buf.push_back((mfxExtBuffer *)nullptr);

    mfxVideoParam videoPrm;
    QSV_MEMSET_ZERO(videoPrm);
    
    videoPrm.NumExtParam = (mfxU16)buf.size();
    videoPrm.ExtParam = (buf.size()) ? &buf[0] : NULL;
    videoPrm.AsyncDepth           = 3;
    videoPrm.IOPattern            = MFX_IOPATTERN_IN_SYSTEM_MEMORY | MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
    videoPrm.vpp.In.FrameRateExtN = 24000;
    videoPrm.vpp.In.FrameRateExtD = 1001;
    videoPrm.vpp.In.FourCC        = MFX_FOURCC_NV12;
    videoPrm.vpp.In.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    videoPrm.vpp.In.PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
    videoPrm.vpp.In.AspectRatioW  = 1;
    videoPrm.vpp.In.AspectRatioH  = 1;
    videoPrm.vpp.In.Width         = 1920;
    videoPrm.vpp.In.Height        = 1088;
    videoPrm.vpp.In.CropX         = 0;
    videoPrm.vpp.In.CropY         = 0;
    videoPrm.vpp.In.CropW         = 1920;
    videoPrm.vpp.In.CropH         = 1080;
    memcpy(&videoPrm.vpp.Out, &videoPrm.vpp.In, sizeof(videoPrm.vpp.In));
    videoPrm.vpp.Out.Width        = 1280;
    videoPrm.vpp.Out.Height       = 720;
    videoPrm.vpp.Out.CropW        = 1280;
    videoPrm.vpp.Out.CropH        = 720;

    mfxExtVPPDoUse vppDoUseOut;
    mfxExtVPPDoUse vppDoNotUseOut;
    mfxExtVPPFrameRateConversion vppFpsConvOut;
    mfxExtVPPImageStab vppImageStabOut;
    mfxExtVPPVideoSignalInfo vppVSIOut;
    mfxExtVPPRotation vppRotateOut;
    
    memcpy(&vppDoUseOut,     &vppDoUse,     sizeof(vppDoUse));
    memcpy(&vppDoNotUseOut,  &vppDoNotUse,  sizeof(vppDoNotUse));
    memcpy(&vppFpsConvOut,   &vppFpsConv,   sizeof(vppFpsConv));
    memcpy(&vppImageStabOut, &vppImageStab, sizeof(vppImageStab));
    memcpy(&vppVSIOut,       &vppVSI,       sizeof(vppVSI));
    memcpy(&vppRotateOut,    &vppRotate,    sizeof(vppRotate));
    
    vector<mfxExtBuffer *> bufOut;
    bufOut.push_back((mfxExtBuffer *)&vppDoUse);
    bufOut.push_back((mfxExtBuffer *)&vppDoNotUse);
    bufOut.push_back((mfxExtBuffer *)nullptr);

    mfxVideoParam videoPrmOut;
    memcpy(&videoPrmOut, &videoPrm, sizeof(videoPrm));
    videoPrmOut.NumExtParam = (mfxU16)bufOut.size();
    videoPrmOut.ExtParam = (bufOut.size()) ? &bufOut[0] : NULL;

    static const mfxU32 vppList[] = {
        MFX_EXTBUFF_VPP_PROCAMP,
        MFX_EXTBUFF_VPP_DENOISE,
        MFX_EXTBUFF_VPP_DETAIL,
        MFX_EXTBUFF_VPP_AUXDATA
    };
    auto check_feature = [&](mfxExtBuffer *structIn, mfxExtBuffer *structOut, mfxVersion requiredVer, mfxU64 featureNoErr, mfxU64 featureWarn) {
        if (check_lib_version(mfxVer, requiredVer)) {
            const mfxU32 target = structIn->BufferId;
            //vppDoUseListとvppDoNotUseListを構築する
            vector<mfxU32> vppDoUseList;
            vector<mfxU32> vppDoNotUseList;
            vppDoUseList.push_back(target);
            for (int i = 0; i < _countof(vppList); i++)
                vppDoNotUseList.push_back(vppList[i]);
            //出力側に同じものをコピー
            vector<mfxU32> vppDoUseListOut(vppDoUseList.size());
            vector<mfxU32> vppDoNotUseListOut(vppDoNotUseList.size());
            copy(vppDoUseList.begin(), vppDoUseList.end(), vppDoUseListOut.begin());
            copy(vppDoNotUseList.begin(), vppDoNotUseList.end(), vppDoNotUseListOut.begin());
            //入力側の設定
            vppDoUse.NumAlg     = (mfxU32)vppDoUseList.size();
            vppDoUse.AlgList    = &vppDoUseList[0];
            vppDoNotUse.NumAlg  = (mfxU32)vppDoNotUseList.size();
            vppDoNotUse.AlgList = &vppDoNotUseList[0];
            //出力側の設定
            vppDoUseOut.NumAlg     = (mfxU32)vppDoUseListOut.size();
            vppDoUseOut.AlgList    = &vppDoUseListOut[0];
            vppDoNotUseOut.NumAlg  = (mfxU32)vppDoNotUseListOut.size();
            vppDoNotUseOut.AlgList = &vppDoNotUseListOut[0];
            //bufの一番端はチェック用に開けてあるので、そこに構造体へのポインタを入れる
            *(buf.end()    - 1) = (mfxExtBuffer *)structIn;
            *(bufOut.end() - 1) = (mfxExtBuffer *)structOut;
            mfxStatus ret = vpp.Query(&videoPrm, &videoPrmOut);
            if (MFX_ERR_NONE <= ret)
                result |= (MFX_ERR_NONE == ret || MFX_WRN_PARTIAL_ACCELERATION == ret) ? featureNoErr : featureWarn;
        }
    };

    check_feature((mfxExtBuffer *)&vppImageStab, (mfxExtBuffer *)&vppImageStabOut, MFX_LIB_VERSION_1_6,  VPP_FEATURE_IMAGE_STABILIZATION, 0x00);
    check_feature((mfxExtBuffer *)&vppVSI,       (mfxExtBuffer *)&vppVSIOut,       MFX_LIB_VERSION_1_8,  VPP_FEATURE_VIDEO_SIGNAL_INFO,   0x00);
#if defined(_WIN32) || defined(_WIN64)
    check_feature((mfxExtBuffer *)&vppRotate,    (mfxExtBuffer *)&vppRotateOut,    MFX_LIB_VERSION_1_17, VPP_FEATURE_ROTATE,              0x00);
#endif //#if defined(_WIN32) || defined(_WIN64)
    
    videoPrm.vpp.Out.FrameRateExtN    = 60000;
    videoPrm.vpp.Out.FrameRateExtD    = 1001;
    videoPrmOut.vpp.Out.FrameRateExtN = 60000;
    videoPrmOut.vpp.Out.FrameRateExtD = 1001;
    check_feature((mfxExtBuffer *)&vppFpsConv,   (mfxExtBuffer *)&vppFpsConvOut,   MFX_LIB_VERSION_1_3,  VPP_FEATURE_FPS_CONVERSION_ADV,  VPP_FEATURE_FPS_CONVERSION);
    return result;
}

mfxU64 CheckVppFeatures(bool hardware, mfxVersion ver) {
    mfxU64 feature = 0x00;
    if (!check_lib_version(ver, MFX_LIB_VERSION_1_3)) {
        //API v1.3未満で実際にチェックする必要は殆ど無いので、
        //コードで決められた値を返すようにする
        feature |= VPP_FEATURE_RESIZE;
        feature |= VPP_FEATURE_DEINTERLACE;
        feature |= VPP_FEATURE_DENOISE;
        feature |= VPP_FEATURE_DETAIL_ENHANCEMENT;
        feature |= VPP_FEATURE_PROC_AMP;
    } else {
        mfxSession session = NULL;

        mfxStatus ret = MFXInit((hardware) ? MFX_IMPL_HARDWARE_ANY : MFX_IMPL_SOFTWARE, &ver, &session);

        feature = (MFX_ERR_NONE == ret) ? CheckVppFeaturesInternal(session, ver) : 0x00;

        MFXClose(session);
    }

    return feature;
}

mfxU64 CheckEncodeFeature(mfxSession session, mfxVersion mfxVer, mfxU16 ratecontrol, mfxU32 codecId) {
    if (codecId == MFX_CODEC_HEVC && !check_lib_version(mfxVer, MFX_LIB_VERSION_1_15)) {
        return 0x00;
    }
    const std::vector<std::pair<mfxU16, mfxVersion>> rc_list = {
        { MFX_RATECONTROL_VBR,    MFX_LIB_VERSION_1_1  },
        { MFX_RATECONTROL_CBR,    MFX_LIB_VERSION_1_1  },
        { MFX_RATECONTROL_CQP,    MFX_LIB_VERSION_1_1  },
        { MFX_RATECONTROL_VQP,    MFX_LIB_VERSION_1_1  },
        { MFX_RATECONTROL_AVBR,   MFX_LIB_VERSION_1_3  },
        { MFX_RATECONTROL_LA,     MFX_LIB_VERSION_1_7  },
        { MFX_RATECONTROL_LA_ICQ, MFX_LIB_VERSION_1_8  },
        { MFX_RATECONTROL_VCM,    MFX_LIB_VERSION_1_8  },
        //{ MFX_RATECONTROL_LA_EXT, MFX_LIB_VERSION_1_11 },
        { MFX_RATECONTROL_LA_HRD, MFX_LIB_VERSION_1_11 },
        { MFX_RATECONTROL_QVBR,   MFX_LIB_VERSION_1_11 },
    };
    for (auto rc : rc_list) {
        if (ratecontrol == rc.first) {
            if (!check_lib_version(mfxVer, rc.second)) {
                return 0x00;
            }
            break;
        }
    }

    MFXVideoENCODE encode(session);

    mfxExtCodingOption cop;
    mfxExtCodingOption2 cop2;
    mfxExtCodingOption3 cop3;
    mfxExtHEVCParam hevc;
    INIT_MFX_EXT_BUFFER(cop,  MFX_EXTBUFF_CODING_OPTION);
    INIT_MFX_EXT_BUFFER(cop2, MFX_EXTBUFF_CODING_OPTION2);
    INIT_MFX_EXT_BUFFER(cop3, MFX_EXTBUFF_CODING_OPTION3);
    INIT_MFX_EXT_BUFFER(hevc, MFX_EXTBUFF_HEVC_PARAM);

    std::vector<mfxExtBuffer *> buf;
    buf.push_back((mfxExtBuffer *)&cop);
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_6)) {
        buf.push_back((mfxExtBuffer *)&cop2);
    }
#if ENABLE_FEATURE_COP3_AND_ABOVE
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_11)) {
        buf.push_back((mfxExtBuffer *)&cop3);
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_15)
        && codecId == MFX_CODEC_HEVC) {
        buf.push_back((mfxExtBuffer *)&hevc);
    }
#endif //#if ENABLE_FEATURE_COP3_AND_ABOVE

    mfxVideoParam videoPrm;
    QSV_MEMSET_ZERO(videoPrm);

    auto set_default_quality_prm = [&videoPrm]() {
        if (   videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_VBR
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_AVBR
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_CBR
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_LA
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_LA_HRD
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_LA_EXT
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_VCM
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_QVBR) {
            videoPrm.mfx.TargetKbps = 3000;
            videoPrm.mfx.MaxKbps    = 3000; //videoPrm.mfx.MaxKbpsはvideoPrm.mfx.TargetKbpsと一致させないとCBRの時に失敗する
            if (videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_AVBR) {
                videoPrm.mfx.Accuracy     = 500;
                videoPrm.mfx.Convergence  = 90;
            }
        } else if (
               videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_CQP
            || videoPrm.mfx.RateControlMethod == MFX_RATECONTROL_VQP) {
            videoPrm.mfx.QPI = 23;
            videoPrm.mfx.QPP = 23;
            videoPrm.mfx.QPB = 23;
        } else {
            //MFX_RATECONTROL_ICQ
            //MFX_RATECONTROL_LA_ICQ
            videoPrm.mfx.ICQQuality = 23;
        }
    };

    videoPrm.NumExtParam = (mfxU16)buf.size();
    videoPrm.ExtParam = (buf.size()) ? &buf[0] : NULL;
    videoPrm.AsyncDepth                  = 3;
    videoPrm.IOPattern                   = MFX_IOPATTERN_IN_SYSTEM_MEMORY;
    videoPrm.mfx.CodecId                 = codecId;
    videoPrm.mfx.RateControlMethod       = (ratecontrol == MFX_RATECONTROL_VQP) ? MFX_RATECONTROL_CQP : ratecontrol;
    switch (codecId) {
    case MFX_CODEC_HEVC:
        videoPrm.mfx.CodecLevel          = MFX_LEVEL_HEVC_4;
        videoPrm.mfx.CodecProfile        = MFX_PROFILE_HEVC_MAIN;
        break;
    default:
    case MFX_CODEC_AVC:
        videoPrm.mfx.CodecLevel          = MFX_LEVEL_AVC_41;
        videoPrm.mfx.CodecProfile        = MFX_PROFILE_AVC_HIGH;
        break;
    }
    videoPrm.mfx.TargetUsage             = MFX_TARGETUSAGE_BEST_QUALITY;
    videoPrm.mfx.EncodedOrder            = 0;
    videoPrm.mfx.NumSlice                = 1;
    videoPrm.mfx.NumRefFrame             = 2;
    videoPrm.mfx.GopPicSize              = USHRT_MAX;
    videoPrm.mfx.GopOptFlag              = MFX_GOP_CLOSED;
    videoPrm.mfx.GopRefDist              = 3;
    videoPrm.mfx.FrameInfo.FrameRateExtN = 30000;
    videoPrm.mfx.FrameInfo.FrameRateExtD = 1001;
    videoPrm.mfx.FrameInfo.FourCC        = MFX_FOURCC_NV12;
    videoPrm.mfx.FrameInfo.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    videoPrm.mfx.FrameInfo.PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
    videoPrm.mfx.FrameInfo.AspectRatioW  = 1;
    videoPrm.mfx.FrameInfo.AspectRatioH  = 1;
    videoPrm.mfx.FrameInfo.Width         = 1920;
    videoPrm.mfx.FrameInfo.Height        = 1088;
    videoPrm.mfx.FrameInfo.CropX         = 0;
    videoPrm.mfx.FrameInfo.CropY         = 0;
    videoPrm.mfx.FrameInfo.CropW         = 1920;
    videoPrm.mfx.FrameInfo.CropH         = 1080;
    set_default_quality_prm();

    mfxExtCodingOption copOut;
    mfxExtCodingOption2 cop2Out;
    mfxExtCodingOption3 cop3Out;
    mfxExtHEVCParam hevcOut;
    std::vector<mfxExtBuffer *> bufOut;
    bufOut.push_back((mfxExtBuffer *)&copOut);
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_6)) {
        bufOut.push_back((mfxExtBuffer *)&cop2Out);
    }
#if ENABLE_FEATURE_COP3_AND_ABOVE
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_11)) {
        bufOut.push_back((mfxExtBuffer *)&cop3Out);
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_15)
        && codecId == MFX_CODEC_HEVC) {
        hevc.PicWidthInLumaSamples  = videoPrm.mfx.FrameInfo.CropW;
        hevc.PicHeightInLumaSamples = videoPrm.mfx.FrameInfo.CropH;
        bufOut.push_back((mfxExtBuffer*)&hevcOut);
    }
#endif //#if ENABLE_FEATURE_COP3_AND_ABOVE
    mfxVideoParam videoPrmOut;
    //In, Outのパラメータが同一となっているようにきちんとコピーする
    //そうしないとQueryが失敗する
    memcpy(&copOut,  &cop,  sizeof(cop));
    memcpy(&cop2Out, &cop2, sizeof(cop2));
    memcpy(&cop3Out, &cop3, sizeof(cop3));
    memcpy(&hevcOut, &hevc, sizeof(hevc));
    memcpy(&videoPrmOut, &videoPrm, sizeof(videoPrm));
    videoPrm.NumExtParam = (mfxU16)bufOut.size();
    videoPrm.ExtParam = &bufOut[0];

    mfxStatus ret = encode.Query(&videoPrm, &videoPrmOut);
    
    mfxU64 result = (MFX_ERR_NONE <= ret && videoPrm.mfx.RateControlMethod == videoPrmOut.mfx.RateControlMethod) ? ENC_FEATURE_CURRENT_RC : 0x00;
    if (result) {

        //まず、エンコードモードについてチェック
        auto check_enc_mode = [&](mfxU16 mode, mfxU64 flag, mfxVersion required_ver) {
            if (check_lib_version(mfxVer, required_ver)) {
                mfxU16 original_method = videoPrm.mfx.RateControlMethod;
                videoPrm.mfx.RateControlMethod = mode;
                set_default_quality_prm();
                memcpy(&copOut,  &cop,  sizeof(cop));
                memcpy(&cop2Out, &cop2, sizeof(cop2));
                memcpy(&cop3Out, &cop3, sizeof(cop3));
                memcpy(&hevcOut, &hevc, sizeof(hevc));
                memcpy(&videoPrmOut, &videoPrm, sizeof(videoPrm));
                videoPrm.NumExtParam = (mfxU16)bufOut.size();
                videoPrm.ExtParam = &bufOut[0];
                if (MFX_ERR_NONE <= encode.Query(&videoPrm, &videoPrmOut) && videoPrm.mfx.RateControlMethod == videoPrmOut.mfx.RateControlMethod)
                    result |= flag;
                videoPrm.mfx.RateControlMethod = original_method;
                set_default_quality_prm();
            }
        };
        check_enc_mode(MFX_RATECONTROL_AVBR,   ENC_FEATURE_AVBR,   MFX_LIB_VERSION_1_3);
        check_enc_mode(MFX_RATECONTROL_LA,     ENC_FEATURE_LA,     MFX_LIB_VERSION_1_7);
        check_enc_mode(MFX_RATECONTROL_ICQ,    ENC_FEATURE_ICQ,    MFX_LIB_VERSION_1_8);
        check_enc_mode(MFX_RATECONTROL_VCM,    ENC_FEATURE_VCM,    MFX_LIB_VERSION_1_8);
        check_enc_mode(MFX_RATECONTROL_LA_HRD, ENC_FEATURE_LA_HRD, MFX_LIB_VERSION_1_11);
        check_enc_mode(MFX_RATECONTROL_QVBR,   ENC_FEATURE_QVBR,   MFX_LIB_VERSION_1_11);

#define CHECK_FEATURE(membersIn, membersOut, flag, value, required_ver) { \
        if (check_lib_version(mfxVer, (required_ver))) { \
            mfxU16 temp = (membersIn); \
            (membersIn) = (value); \
            memcpy(&copOut,  &cop,  sizeof(cop)); \
            memcpy(&cop2Out, &cop2, sizeof(cop2)); \
            memcpy(&cop3Out, &cop3, sizeof(cop3)); \
            memcpy(&hevcOut, &hevc, sizeof(hevc)); \
            if (MFX_ERR_NONE <= encode.Query(&videoPrm, &videoPrmOut) \
                && (membersIn) == (membersOut) \
                && videoPrm.mfx.RateControlMethod == videoPrmOut.mfx.RateControlMethod) \
                result |= (flag); \
            (membersIn) = temp; \
        } \
    }\
        //これはもう単純にAPIチェックでOK
        if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_3)) {
            result |= ENC_FEATURE_VUI_INFO;
        }
        //ひとつひとつパラメータを入れ替えて試していく
#pragma warning(push)
#pragma warning(disable:4244) //'mfxU16' から 'mfxU8' への変換です。データが失われる可能性があります。
#define PICTYPE mfx.FrameInfo.PicStruct
        const mfxU32 MFX_TRELLIS_ALL = MFX_TRELLIS_I | MFX_TRELLIS_P | MFX_TRELLIS_B;
        CHECK_FEATURE(cop.AUDelimiter,           copOut.AUDelimiter,           ENC_FEATURE_AUD,           MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_1);
        CHECK_FEATURE(videoPrm.PICTYPE,          videoPrmOut.PICTYPE,          ENC_FEATURE_INTERLACE,     MFX_PICSTRUCT_FIELD_TFF, MFX_LIB_VERSION_1_1);
        CHECK_FEATURE(cop.PicTimingSEI,          copOut.PicTimingSEI,          ENC_FEATURE_PIC_STRUCT,    MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_1);
        CHECK_FEATURE(cop.RateDistortionOpt,     copOut.RateDistortionOpt,     ENC_FEATURE_RDO,           MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_1);
        CHECK_FEATURE(cop.CAVLC,                 copOut.CAVLC,                 ENC_FEATURE_CAVLC,         MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_1);
        CHECK_FEATURE(cop2.ExtBRC,               cop2Out.ExtBRC,               ENC_FEATURE_EXT_BRC,       MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_6);
        CHECK_FEATURE(cop2.MBBRC,                cop2Out.MBBRC,                ENC_FEATURE_MBBRC,         MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_6);
        CHECK_FEATURE(cop2.Trellis,              cop2Out.Trellis,              ENC_FEATURE_TRELLIS,       MFX_TRELLIS_ALL,         MFX_LIB_VERSION_1_7);
        CHECK_FEATURE(cop2.IntRefType,           cop2Out.IntRefType,           ENC_FEATURE_INTRA_REFRESH, 1,                       MFX_LIB_VERSION_1_7);
        CHECK_FEATURE(cop2.AdaptiveI,            cop2Out.AdaptiveI,            ENC_FEATURE_ADAPTIVE_I,    MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_8);
        CHECK_FEATURE(cop2.AdaptiveB,            cop2Out.AdaptiveB,            ENC_FEATURE_ADAPTIVE_B,    MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_8);
        CHECK_FEATURE(cop2.BRefType,             cop2Out.BRefType,             ENC_FEATURE_B_PYRAMID,     MFX_B_REF_PYRAMID,       MFX_LIB_VERSION_1_8);
        if (rc_is_type_lookahead(ratecontrol)) {
            CHECK_FEATURE(cop2.LookAheadDS,      cop2Out.LookAheadDS,          ENC_FEATURE_LA_DS,         MFX_LOOKAHEAD_DS_2x,     MFX_LIB_VERSION_1_8);
        }
        CHECK_FEATURE(cop2.DisableDeblockingIdc, cop2Out.DisableDeblockingIdc, ENC_FEATURE_NO_DEBLOCK,    MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_9);
        CHECK_FEATURE(cop2.MaxQPI,               cop2Out.MaxQPI,               ENC_FEATURE_QP_MINMAX,     48,                      MFX_LIB_VERSION_1_9);
        cop3.WinBRCMaxAvgKbps = 3000;
#if defined(_WIN32) || defined(_WIN64)
        CHECK_FEATURE(cop3.WinBRCSize,           cop3Out.WinBRCSize,           ENC_FEATURE_WINBRC,        10,                      MFX_LIB_VERSION_1_11);
        cop3.WinBRCMaxAvgKbps = 0;
        CHECK_FEATURE(cop3.EnableMBQP,                 cop3Out.EnableMBQP,                 ENC_FEATURE_PERMBQP,                    MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_13);
        CHECK_FEATURE(cop3.DirectBiasAdjustment,       cop3Out.DirectBiasAdjustment,       ENC_FEATURE_DIRECT_BIAS_ADJUST,         MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_13);
        CHECK_FEATURE(cop3.GlobalMotionBiasAdjustment, cop3Out.GlobalMotionBiasAdjustment, ENC_FEATURE_GLOBAL_MOTION_ADJUST,       MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_13);
        CHECK_FEATURE(videoPrm.mfx.LowPower,     videoPrmOut.mfx.LowPower,     ENC_FEATURE_FIXED_FUNC,    MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_15);
        CHECK_FEATURE(cop3.WeightedPred,         cop3Out.WeightedPred,         ENC_FEATURE_WEIGHT_P,      MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_16);
        CHECK_FEATURE(cop3.WeightedBiPred,       cop3Out.WeightedBiPred,       ENC_FEATURE_WEIGHT_B,      MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_16);
        CHECK_FEATURE(cop3.FadeDetection,        cop3Out.FadeDetection,        ENC_FEATURE_FADE_DETECT,   MFX_CODINGOPTION_ON,     MFX_LIB_VERSION_1_17);
#endif //#if defined(_WIN32) || defined(_WIN64)
#undef PICTYPE
#pragma warning(pop)
        //付随オプション
        result |= ENC_FEATURE_SCENECHANGE;
        //encCtrlを渡すことにより実現するVQPでは、B-pyramidは不安定(フレーム順が入れ替わるなど)
        if (MFX_RATECONTROL_VQP == ratecontrol && check_lib_version(mfxVer, MFX_LIB_VERSION_1_8)) {
            result &= ~ENC_FEATURE_B_PYRAMID;
        }
        if (result & ENC_FEATURE_B_PYRAMID) {
            result |= ENC_FEATURE_B_PYRAMID_MANY_BFRAMES;
        }
        if ((ENC_FEATURE_SCENECHANGE | ENC_FEATURE_B_PYRAMID) == (result & (ENC_FEATURE_SCENECHANGE | ENC_FEATURE_B_PYRAMID))) {
            result |= ENC_FEATURE_B_PYRAMID_AND_SC;
        }
        //以下特殊な場合
        if (rc_is_type_lookahead(ratecontrol)) {
            result &= ~ENC_FEATURE_RDO;
            result &= ~ENC_FEATURE_MBBRC;
            result &= ~ENC_FEATURE_EXT_BRC;
            if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_8)) {
                //API v1.8以降、LA + scenechangeは不安定(フリーズ)
                result &= ~ENC_FEATURE_SCENECHANGE;
                //API v1.8以降、LA + 多すぎるBフレームは不安定(フリーズ)
                result &= ~ENC_FEATURE_B_PYRAMID_MANY_BFRAMES;
            }
        } else if (MFX_RATECONTROL_CQP == ratecontrol
                || MFX_RATECONTROL_VQP == ratecontrol) {
            result &= ~ENC_FEATURE_MBBRC;
            result &= ~ENC_FEATURE_EXT_BRC;
        }
        //API v1.8以降では今のところ不安定(フレーム順が入れ替わるなど)
        if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_8)) {
            result &= ~ENC_FEATURE_B_PYRAMID_AND_SC;
        }
    }
#undef CHECK_FEATURE
    return result;
}

//サポートする機能のチェックをAPIバージョンのみで行う
//API v1.6以降はCheckEncodeFeatureを使うべき
//同一のAPIバージョンでも環境により異なることが多くなるため
static mfxU64 CheckEncodeFeatureStatic(mfxVersion mfxVer, mfxU16 ratecontrol) {
    mfxU64 feature = 0x00;
    //まずレート制御モードをチェック
    BOOL rate_control_supported = false;
    switch (ratecontrol) {
    case MFX_RATECONTROL_CBR:
    case MFX_RATECONTROL_VBR:
    case MFX_RATECONTROL_CQP:
    case MFX_RATECONTROL_VQP:
        rate_control_supported = true;
        break;
    case MFX_RATECONTROL_AVBR:
        rate_control_supported = check_lib_version(mfxVer, MFX_LIB_VERSION_1_3);
        break;
    case MFX_RATECONTROL_LA:
        rate_control_supported = check_lib_version(mfxVer, MFX_LIB_VERSION_1_7);
        break;
    case MFX_RATECONTROL_ICQ:
    case MFX_RATECONTROL_LA_ICQ:
    case MFX_RATECONTROL_VCM:
        rate_control_supported = check_lib_version(mfxVer, MFX_LIB_VERSION_1_8);
        break;
    case MFX_RATECONTROL_LA_EXT:
        rate_control_supported = check_lib_version(mfxVer, MFX_LIB_VERSION_1_10);
        break;
    case MFX_RATECONTROL_LA_HRD:
    case MFX_RATECONTROL_QVBR:
        rate_control_supported = check_lib_version(mfxVer, MFX_LIB_VERSION_1_11);
        break;
    default:
        break;
    }
    if (!rate_control_supported) {
        return feature;
    }

    //各モードをチェック
    feature |= ENC_FEATURE_CURRENT_RC;
    feature |= ENC_FEATURE_SCENECHANGE;

    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_1)) {
        feature |= ENC_FEATURE_AUD;
        feature |= ENC_FEATURE_PIC_STRUCT;
        feature |= ENC_FEATURE_RDO;
        feature |= ENC_FEATURE_CAVLC;
        feature |= ENC_FEATURE_INTERLACE;
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_3)) {
        feature |= ENC_FEATURE_VUI_INFO;
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_6)) {
        feature |= ENC_FEATURE_EXT_BRC;
        feature |= ENC_FEATURE_MBBRC;
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_7)) {
        feature |= ENC_FEATURE_TRELLIS;
    }
    if (check_lib_version(mfxVer, MFX_LIB_VERSION_1_8)) {
        feature |= ENC_FEATURE_ADAPTIVE_I;
        feature |= ENC_FEATURE_ADAPTIVE_B;
        feature |= ENC_FEATURE_B_PYRAMID;
        feature |= ENC_FEATURE_B_PYRAMID_MANY_BFRAMES;
        feature |= ENC_FEATURE_VUI_INFO;
        if (rc_is_type_lookahead(ratecontrol)) {
            feature |= ENC_FEATURE_LA_DS;
            feature &= ~ENC_FEATURE_B_PYRAMID_MANY_BFRAMES;
        }
    }

    //以下特殊な場合の制限
    if (rc_is_type_lookahead(ratecontrol)) {
        feature &= ~ENC_FEATURE_RDO;
        feature &= ~ENC_FEATURE_MBBRC;
        feature &= ~ENC_FEATURE_EXT_BRC;
    } else if (MFX_RATECONTROL_CQP == ratecontrol
            || MFX_RATECONTROL_VQP == ratecontrol) {
        feature &= ~ENC_FEATURE_MBBRC;
        feature &= ~ENC_FEATURE_EXT_BRC;
    }

    return feature;
}

mfxU64 CheckEncodeFeature(bool hardware, mfxVersion ver, mfxU16 ratecontrol, mfxU32 codecId) {
    mfxU64 feature = 0x00;
    //暫定的に、sw libのチェックを無効化する
    if (!hardware) {
        return feature;
    }
    if (!check_lib_version(ver, MFX_LIB_VERSION_1_0)) {
        ; //特にすることはない
    } else if (!check_lib_version(ver, MFX_LIB_VERSION_1_6)) {
        //API v1.6未満で実際にチェックする必要は殆ど無いので、
        //コードで決められた値を返すようにする
        feature = CheckEncodeFeatureStatic(ver, ratecontrol);
    } else {
        mfxSession session = NULL;

        mfxStatus ret = MFXInit((hardware) ? MFX_IMPL_HARDWARE_ANY : MFX_IMPL_SOFTWARE, &ver, &session);

#ifdef LIBVA_SUPPORT
        //in case of system memory allocator we also have to pass MFX_HANDLE_VA_DISPLAY to HW library
        std::unique_ptr<CHWDevice> phwDevice;
        if (ret == MFX_ERR_NONE) {
            mfxIMPL impl;
            MFXQueryIMPL(session, &impl);

            if (MFX_IMPL_HARDWARE == MFX_IMPL_BASETYPE(impl)) {
                phwDevice.reset(CreateVAAPIDevice());

                // provide device manager to MediaSDK
                mfxHDL hdl = NULL;
                if (phwDevice.get() != nullptr
                   && MFX_ERR_NONE != (ret = phwDevice->Init(NULL, 0, MSDKAdapter::GetNumber(session)))
                   && MFX_ERR_NONE != (ret = phwDevice->GetHandle(MFX_HANDLE_VA_DISPLAY, &hdl))) {
                    ret = MFXVideoCORE_SetHandle(session, MFX_HANDLE_VA_DISPLAY, hdl);
                }
            }
        }
#endif //#ifdef LIBVA_SUPPORT

        CSessionPlugins sessionPlugins(session);
        if (codecId == MFX_CODEC_HEVC) {
            sessionPlugins.LoadPlugin(MFX_PLUGINTYPE_VIDEO_ENCODE, MFX_PLUGINID_HEVCE_HW, 1);
        } else if (codecId == MFX_CODEC_VP8) {
            sessionPlugins.LoadPlugin(MFX_PLUGINTYPE_VIDEO_ENCODE, MFX_PLUGINID_VP8E_HW, 1);
        }
        feature = (MFX_ERR_NONE == ret) ? CheckEncodeFeature(session, ver, ratecontrol, codecId) : 0x00;
        
        sessionPlugins.UnloadPlugins();
        MFXClose(session);
#ifdef LIBVA_SUPPORT
        phwDevice.reset();
#endif //#ifdef LIBVA_SUPPORT
    }

    return feature;
}

mfxU64 CheckEncodeFeature(bool hardware, mfxU16 ratecontrol, mfxU32 codecId) {
    mfxVersion ver = (hardware) ? get_mfx_libhw_version() : get_mfx_libsw_version();
    return CheckEncodeFeature(hardware, ver, ratecontrol, codecId);
}

const TCHAR *EncFeatureStr(mfxU64 enc_feature) {
    for (const FEATURE_DESC *ptr = list_enc_feature; ptr->desc; ptr++)
        if (enc_feature == (mfxU64)ptr->value)
            return ptr->desc;
    return NULL;
}

vector<mfxU64> MakeFeatureList(bool hardware, mfxVersion ver, const vector<CX_DESC>& rateControlList, mfxU32 codecId) {
    vector<mfxU64> availableFeatureForEachRC;
    availableFeatureForEachRC.reserve(rateControlList.size());
    for (const auto& ratecontrol : rateControlList) {
        mfxU64 ret = CheckEncodeFeature(hardware, ver, (mfxU16)ratecontrol.value, codecId);
        if (ret == 0 && ratecontrol.value == MFX_RATECONTROL_CQP) {
            ver = MFX_LIB_VERSION_0_0;
        }
        availableFeatureForEachRC.push_back(ret);
    }
    return std::move(availableFeatureForEachRC);
}

vector<vector<mfxU64>> MakeFeatureListPerCodec(bool hardware, mfxVersion ver, const vector<CX_DESC>& rateControlList, const vector<mfxU32>& codecIdList) {
    vector<vector<mfxU64>> codecFeatures;
    for (auto codec : codecIdList) {
        codecFeatures.push_back(MakeFeatureList(hardware, ver, rateControlList, codec));
    }
    return std::move(codecFeatures);
}

vector<vector<mfxU64>> MakeFeatureListPerCodec(bool hardware, const vector<CX_DESC>& rateControlList, const vector<mfxU32>& codecIdList) {
    vector<vector<mfxU64>> codecFeatures;
    mfxVersion ver = (hardware) ? get_mfx_libhw_version() : get_mfx_libsw_version();
    for (auto codec : codecIdList) {
        codecFeatures.push_back(MakeFeatureList(hardware, ver, rateControlList, codec));
    }
    return std::move(codecFeatures);
}

static const TCHAR *const QSV_FEATURE_MARK_YES_NO[] ={ _T("×"), _T("○") };
static const TCHAR *const QSV_FEATURE_MARK_YES_NO_WITH_SPACE[] = { _T(" x    "), _T(" o    ") };

tstring MakeFeatureListStr(mfxU64 feature) {
    tstring str;
    for (const FEATURE_DESC *ptr = list_enc_feature; ptr->desc; ptr++) {
        str += ptr->desc;
        str += QSV_FEATURE_MARK_YES_NO_WITH_SPACE[!!(feature & ptr->value)];
        str += _T("\n");
    }
    str += _T("\n");
    return str;
}

tstring MakeFeatureListStr(bool hardware, FeatureListStrType type) {
    const vector<mfxU32> codecLists = { MFX_CODEC_AVC, MFX_CODEC_HEVC, MFX_CODEC_MPEG2, MFX_CODEC_VP8 };
    auto featurePerCodec = MakeFeatureListPerCodec(hardware, make_vector(list_rate_control_ry), codecLists);
    
    tstring str;
    
    for (mfxU32 i_codec = 0; i_codec < codecLists.size(); i_codec++) {
        auto& availableFeatureForEachRC = featurePerCodec[i_codec];
        //H.264以外で、ひとつもフラグが立っていなかったら、スキップする
        if (codecLists[i_codec] != MFX_CODEC_AVC
            && 0 == std::accumulate(availableFeatureForEachRC.begin(), availableFeatureForEachRC.end(), 0,
            [](mfxU32 sum, mfxU64 value) { return sum | (mfxU32)(value & 0xffffffff) | (mfxU32)(value >> 32); })) {
            continue;
        }
        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += _T("<b>");
        }
        str += _T("Codec: ") + tstring(CodecIdToStr(codecLists[i_codec])) + _T("\n");

        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += _T("</b><table class=simpleOrange>");
        }

        switch (type) {
        case FEATURE_LIST_STR_TYPE_HTML: str += _T("<tr><th></th>"); break;
        case FEATURE_LIST_STR_TYPE_TXT:
        default:
            //ヘッダ部分
            const mfxU32 row_header_length = (mfxU32)_tcslen(list_enc_feature[0].desc);
            for (mfxU32 i = 1; i < row_header_length; i++)
                str += _T(" ");
            break;
        }

        for (mfxU32 i = 0; i < _countof(list_rate_control_ry); i++) {
            switch (type) {
            case FEATURE_LIST_STR_TYPE_CSV: str += _T(","); break;
            case FEATURE_LIST_STR_TYPE_HTML: str += _T("<th>"); break;
            case FEATURE_LIST_STR_TYPE_TXT:
            default: str += _T(" "); break;
            }
            str += list_rate_control_ry[i].desc;
            if (type == FEATURE_LIST_STR_TYPE_HTML) {
                str += _T("</th>");
            }
        }
        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += _T("</tr>");
        }
        str += _T("\n");

        //モードがサポートされているか
        for (const FEATURE_DESC *ptr = list_enc_feature; ptr->desc; ptr++) {
            if (type == FEATURE_LIST_STR_TYPE_HTML) {
                str += _T("<tr><td>");
            }
            str += ptr->desc;
            switch (type) {
            case FEATURE_LIST_STR_TYPE_CSV: str += _T(","); break;
            case FEATURE_LIST_STR_TYPE_HTML: str += _T("</td>"); break;
            default: break;
            }
            for (mfxU32 i = 0; i < _countof(list_rate_control_ry); i++) {
                if (type == FEATURE_LIST_STR_TYPE_HTML) {
                    str += !!(availableFeatureForEachRC[i] & ptr->value) ? _T("<td class=ok>") : _T("<td class=fail>");
                }
                if (type == FEATURE_LIST_STR_TYPE_TXT) {
                    str += QSV_FEATURE_MARK_YES_NO_WITH_SPACE[!!(availableFeatureForEachRC[i] & ptr->value)];
                } else {
                    str += QSV_FEATURE_MARK_YES_NO[!!(availableFeatureForEachRC[i] & ptr->value)];
                }
                switch (type) {
                case FEATURE_LIST_STR_TYPE_CSV: str += _T(","); break;
                case FEATURE_LIST_STR_TYPE_HTML: str += _T("</td>"); break;
                default: break;
                }
            }
            if (type == FEATURE_LIST_STR_TYPE_HTML) {
                str += _T("</tr>");
            }
            str += _T("\n");
        }
        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += _T("</table><br>");
        }
        str += _T("\n");
    }
    return str;
}

tstring MakeVppFeatureStr(bool hardware, FeatureListStrType type) {
    mfxVersion ver = (hardware) ? get_mfx_libhw_version() : get_mfx_libsw_version();
    uint64_t features = CheckVppFeatures(hardware, ver);
    const TCHAR *MARK_YES_NO[] = { _T(" x"), _T(" o") };
    tstring str;
    if (type == FEATURE_LIST_STR_TYPE_HTML) {
        str += _T("<table class=simpleOrange>");
    }
    for (const FEATURE_DESC *ptr = list_vpp_feature; ptr->desc; ptr++) {
        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += _T("<tr><td>");
        }
        str += ptr->desc;
        switch (type) {
        case FEATURE_LIST_STR_TYPE_CSV: str += _T(","); break;
        case FEATURE_LIST_STR_TYPE_HTML: str += _T("</td>"); break;
        default: break;
        }
        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += (features & ptr->value) ? _T("<td class=ok>") : _T("<td class=fail>");
        }
        if (type == FEATURE_LIST_STR_TYPE_TXT) {
            str += MARK_YES_NO[ptr->value == (features & ptr->value)];
        } else {
            str += QSV_FEATURE_MARK_YES_NO[ptr->value == (features & ptr->value)];
        }
        if (type == FEATURE_LIST_STR_TYPE_HTML) {
            str += _T("</td></tr>");
        }
        str += _T("\n");
    }
    if (type == FEATURE_LIST_STR_TYPE_HTML) {
        str += _T("</table>\n");
    }
    return str;
}

BOOL check_lib_version(mfxU32 _value, mfxU32 _required) {
    mfxVersion value, required;
    value.Version = _value;
    required.Version = _required;
    if (value.Major < required.Major)
        return FALSE;
    if (value.Major > required.Major)
        return TRUE;
    if (value.Minor < required.Minor)
        return FALSE;
    return TRUE;
}

void adjust_sar(int *sar_w, int *sar_h, int width, int height) {
    int aspect_w = *sar_w;
    int aspect_h = *sar_h;
    //正負チェック
    if (aspect_w * aspect_h <= 0)
        aspect_w = aspect_h = 0;
    else if (aspect_w < 0) {
        //負で与えられている場合はDARでの指定
        //SAR比に変換する
        int dar_x = -1 * aspect_w;
        int dar_y = -1 * aspect_h;
        int x = dar_x * height;
        int y = dar_y * width;
        //多少のづれは容認する
        if (abs(y - x) > 16 * dar_y) {
            //gcd
            int a = x, b = y, c;
            while ((c = a % b) != 0)
                a = b, b = c;
            *sar_w = x / b;
            *sar_h = y / b;
        } else {
             *sar_w = *sar_h = 1;
        }
    } else {
        //sarも一応gcdをとっておく
        int a = aspect_w, b = aspect_h, c;
        while ((c = a % b) != 0)
            a = b, b = c;
        *sar_w = aspect_w / b;
        *sar_h = aspect_h / b;
    }
}

const TCHAR *get_vpp_image_stab_mode_str(int mode) {
    switch (mode) {
    case MFX_IMAGESTAB_MODE_UPSCALE: return _T("upscale");
    case MFX_IMAGESTAB_MODE_BOXING:  return _T("boxing");
    default: return _T("unknown");
    }
}

const TCHAR *get_err_mes(int sts) {
    switch (sts) {
        case MFX_ERR_NONE:                     return _T("no error.");
        case MFX_ERR_UNKNOWN:                  return _T("unknown error.");
        case MFX_ERR_NULL_PTR:                 return _T("null pointer.");
        case MFX_ERR_UNSUPPORTED:              return _T("undeveloped feature.");
        case MFX_ERR_MEMORY_ALLOC:             return _T("failed to allocate memory.");
        case MFX_ERR_NOT_ENOUGH_BUFFER:        return _T("insufficient buffer at input/output.");
        case MFX_ERR_INVALID_HANDLE:           return _T("invalid handle.");
        case MFX_ERR_LOCK_MEMORY:              return _T("failed to lock the memory block.");
        case MFX_ERR_NOT_INITIALIZED:          return _T("member function called before initialization.");
        case MFX_ERR_NOT_FOUND:                return _T("the specified object is not found.");
        case MFX_ERR_MORE_DATA:                return _T("expect more data at input.");
        case MFX_ERR_MORE_SURFACE:             return _T("expect more surface at output.");
        case MFX_ERR_ABORTED:                  return _T("operation aborted.");
        case MFX_ERR_DEVICE_LOST:              return _T("lose the HW acceleration device.");
        case MFX_ERR_INCOMPATIBLE_VIDEO_PARAM: return _T("incompatible video parameters.");
        case MFX_ERR_INVALID_VIDEO_PARAM:      return _T("invalid video parameters.");
        case MFX_ERR_UNDEFINED_BEHAVIOR:       return _T("undefined behavior.");
        case MFX_ERR_DEVICE_FAILED:            return _T("device operation failure.");
        
        case MFX_WRN_IN_EXECUTION:             return _T("the previous asynchrous operation is in execution.");
        case MFX_WRN_DEVICE_BUSY:              return _T("the HW acceleration device is busy.");
        case MFX_WRN_VIDEO_PARAM_CHANGED:      return _T("the video parameters are changed during decoding.");
        case MFX_WRN_PARTIAL_ACCELERATION:     return _T("SW is used.");
        case MFX_WRN_INCOMPATIBLE_VIDEO_PARAM: return _T("incompatible video parameters.");
        case MFX_WRN_VALUE_NOT_CHANGED:        return _T("the value is saturated based on its valid range.");
        case MFX_WRN_OUT_OF_RANGE:             return _T("the value is out of valid range.");
        default:                               return _T("unknown error."); 
    }
}

const TCHAR *get_low_power_str(mfxU16 LowPower) {
    switch (LowPower) {
    case MFX_CODINGOPTION_OFF: return _T(" PG");
    case MFX_CODINGOPTION_ON:  return _T(" FF");
    default: return _T("");
    }
}

mfxStatus WriteY4MHeader(FILE *fp, const mfxFrameInfo *info) {
    char buffer[256] = { 0 };
    char *ptr = buffer;
    mfxU32 len = 0;
    memcpy(ptr, "YUV4MPEG2 ", 10);
    len += 10;

    len += sprintf_s(ptr+len, sizeof(buffer)-len, "W%d H%d ", info->CropW, info->CropH);
    len += sprintf_s(ptr+len, sizeof(buffer)-len, "F%d:%d ", info->FrameRateExtN, info->FrameRateExtD);

    const char *picstruct = "Ip ";
    if (info->PicStruct & MFX_PICSTRUCT_FIELD_TFF) {
        picstruct = "It ";
    } else if (info->PicStruct & MFX_PICSTRUCT_FIELD_BFF) {
        picstruct = "Ib ";
    }
    strcpy_s(ptr+len, sizeof(buffer)-len, picstruct); len += 3;
    len += sprintf_s(ptr+len, sizeof(buffer)-len, "A%d:%d ", info->AspectRatioW, info->AspectRatioH);
    strcpy_s(ptr+len, sizeof(buffer)-len, "C420mpeg2\n"); len += (mfxU32)strlen("C420mpeg2\n");
    return (len == fwrite(buffer, 1, len, fp)) ? MFX_ERR_NONE : MFX_ERR_UNDEFINED_BEHAVIOR;
}

mfxStatus ParseY4MHeader(char *buf, mfxFrameInfo *info) {
    char *p, *q = NULL;
    memset(info, 0, sizeof(mfxFrameInfo));
    for (p = buf; (p = strtok_s(p, " ", &q)) != NULL; ) {
        switch (*p) {
            case 'W':
                {
                    char *eptr = NULL;
                    int w = strtol(p+1, &eptr, 10);
                    if (*eptr == '\0' && w)
                        info->Width = (mfxU16)w;
                }
                break;
            case 'H':
                {
                    char *eptr = NULL;
                    int h = strtol(p+1, &eptr, 10);
                    if (*eptr == '\0' && h)
                        info->Height = (mfxU16)h;
                }
                break;
            case 'F':
                {
                    int rate = 0, scale = 0;
                    if (   (info->FrameRateExtN == 0 || info->FrameRateExtD == 0)
                        && sscanf_s(p+1, "%d:%d", &rate, &scale) == 2) {
                            if (rate && scale) {
                                info->FrameRateExtN = rate;
                                info->FrameRateExtD = scale;
                            }
                    }
                }
                break;
            case 'A':
                {
                    int sar_x = 0, sar_y = 0;
                    if (   (info->AspectRatioW == 0 || info->AspectRatioH == 0)
                        && sscanf_s(p+1, "%d:%d", &sar_x, &sar_y) == 2) {
                            if (sar_x && sar_y) {
                                info->AspectRatioW = (mfxU16)sar_x;
                                info->AspectRatioH = (mfxU16)sar_y;
                            }
                    }
                }
                break;
            case 'I':
                switch (*(p+1)) {
            case 'b':
                info->PicStruct = MFX_PICSTRUCT_FIELD_BFF;
                break;
            case 't':
            case 'm':
                info->PicStruct = MFX_PICSTRUCT_FIELD_TFF;
                break;
            default:
                break;
                }
                break;
            case 'C':
                if (   0 != _strnicmp(p+1, "420",      strlen("420"))
                    && 0 != _strnicmp(p+1, "420mpeg2", strlen("420mpeg2"))
                    && 0 != _strnicmp(p+1, "420jpeg",  strlen("420jpeg"))
                    && 0 != _strnicmp(p+1, "420paldv", strlen("420paldv"))) {
                    return MFX_PRINT_OPTION_ERR;
                }
                break;
            default:
                break;
        }
        p = NULL;
    }
    return MFX_ERR_NONE;
}

#if defined(_WIN32) || defined(_WIN64)

#include <Windows.h>
#include <process.h>

typedef void (WINAPI *RtlGetVersion_FUNC)(OSVERSIONINFOEXW*);

static int getRealWindowsVersion(DWORD *major, DWORD *minor) {
    *major = 0;
    *minor = 0;
    OSVERSIONINFOEXW osver;
    HMODULE hModule = NULL;
    RtlGetVersion_FUNC func = NULL;
    int ret = 1;
    if (   NULL != (hModule = LoadLibrary(_T("ntdll.dll")))
        && NULL != (func = (RtlGetVersion_FUNC)GetProcAddress(hModule, "RtlGetVersion"))) {
        func(&osver);
        *major = osver.dwMajorVersion;
        *minor = osver.dwMinorVersion;
        ret = 0;
    }
    if (hModule) {
        FreeLibrary(hModule);
    }
    return ret;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

BOOL check_OS_Win8orLater() {
#if defined(_WIN32) || defined(_WIN64)
#if (_MSC_VER >= 1800)
    return IsWindows8OrGreater();
#else
    OSVERSIONINFO osvi = { 0 };
    osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
    GetVersionEx(&osvi);
    return ((osvi.dwPlatformId == VER_PLATFORM_WIN32_NT) && ((osvi.dwMajorVersion == 6 && osvi.dwMinorVersion >= 2) || osvi.dwMajorVersion > 6));
#endif //(_MSC_VER >= 1800)
#else //#if defined(_WIN32) || defined(_WIN64)
    return FALSE;
#endif //#if defined(_WIN32) || defined(_WIN64)
}

tstring getOSVersion() {
#if defined(_WIN32) || defined(_WIN64)
    const TCHAR *ptr = _T("Unknown");
    OSVERSIONINFO info = { 0 };
    info.dwOSVersionInfoSize = sizeof(info);
    GetVersionEx(&info);
    switch (info.dwPlatformId) {
    case VER_PLATFORM_WIN32_WINDOWS:
        if (4 <= info.dwMajorVersion) {
            switch (info.dwMinorVersion) {
            case 0:  ptr = _T("Windows 95"); break;
            case 10: ptr = _T("Windows 98"); break;
            case 90: ptr = _T("Windows Me"); break;
            default: break;
            }
        }
        break;
    case VER_PLATFORM_WIN32_NT:
        if (info.dwMajorVersion == 6) {
            getRealWindowsVersion(&info.dwMajorVersion, &info.dwMinorVersion);
        }
        switch (info.dwMajorVersion) {
        case 3:
            switch (info.dwMinorVersion) {
            case 0:  ptr = _T("Windows NT 3"); break;
            case 1:  ptr = _T("Windows NT 3.1"); break;
            case 5:  ptr = _T("Windows NT 3.5"); break;
            case 51: ptr = _T("Windows NT 3.51"); break;
            default: break;
            }
            break;
        case 4:
            if (0 == info.dwMinorVersion)
                ptr = _T("Windows NT 4.0");
            break;
        case 5:
            switch (info.dwMinorVersion) {
            case 0:  ptr = _T("Windows 2000"); break;
            case 1:  ptr = _T("Windows XP"); break;
            case 2:  ptr = _T("Windows Server 2003"); break;
            default: break;
            }
            break;
        case 6:
            switch (info.dwMinorVersion) {
            case 0:  ptr = _T("Windows Vista"); break;
            case 1:  ptr = _T("Windows 7"); break;
            case 2:  ptr = _T("Windows 8"); break;
            case 3:  ptr = _T("Windows 8.1"); break;
            case 4:  ptr = _T("Windows 10"); break;
            default:
                if (5 <= info.dwMinorVersion) {
                    ptr = _T("Later than Windows 10");
                }
                break;
            }
            break;
        case 10:
            ptr = _T("Windows 10");
            break;
        default:
            if (10 <= info.dwMajorVersion) {
                ptr = _T("Later than Windows 10");
            }
            break;
        }
        break;
    default:
        break;
    }
    return tstring(ptr);
#else //#if defined(_WIN32) || defined(_WIN64)
    std::string str = "";
    FILE *fp = popen("/usr/bin/lsb_release -a", "r");
    if (fp != NULL) {
        char buffer[2048];
        while (NULL != fgets(buffer, _countof(buffer), fp)) {
            str += buffer;
        }
        pclose(fp);
        if (str.length() > 0) {
            auto sep = split(str, "\n");
            for (auto line : sep) {
                if (line.find("Description") != std::string::npos) {
                    std::string::size_type pos = line.find(":");
                    if (pos == std::string::npos) {
                        pos = std::string("Description").length();
                    }
                    pos++;
                    str = line.substr(pos);
                    break;
                }
            }
        }
    }
    if (str.length() == 0) {
        struct utsname buf;
        uname(&buf);
        str += buf.sysname;
        str += " ";
        str += buf.release;
    }
    return char_to_tstring(trim(str));
#endif //#if defined(_WIN32) || defined(_WIN64)
}

BOOL is_64bit_os() {
#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO sinfo = { 0 };
    GetNativeSystemInfo(&sinfo);
    return sinfo.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64;
#else //#if defined(_WIN32) || defined(_WIN64)
    struct utsname buf;
    uname(&buf);
    return NULL != strstr(buf.machine, "x64")
        || NULL != strstr(buf.machine, "x86_64")
        || NULL != strstr(buf.machine, "amd64");
#endif //#if defined(_WIN32) || defined(_WIN64)
}

uint64_t getPhysicalRamSize(uint64_t *ramUsed) {
#if defined(_WIN32) || defined(_WIN64)
    MEMORYSTATUSEX msex ={ 0 };
    msex.dwLength = sizeof(msex);
    GlobalMemoryStatusEx(&msex);
    if (NULL != ramUsed) {
        *ramUsed = msex.ullTotalPhys - msex.ullAvailPhys;
    }
    return msex.ullTotalPhys;
#else //#if defined(_WIN32) || defined(_WIN64)
    struct sysinfo info;
    sysinfo(&info);
    if (NULL != ramUsed) {
        *ramUsed = info.totalram - info.freeram;
    }
    return info.totalram;
#endif //#if defined(_WIN32) || defined(_WIN64)
}

tstring getEnviromentInfo(bool add_ram_info) {
    tstring buf;

    TCHAR cpu_info[1024] = { 0 };
    getCPUInfo(cpu_info, _countof(cpu_info));

    TCHAR gpu_info[1024] = { 0 };
    getGPUInfo("Intel", gpu_info, _countof(gpu_info));

    uint64_t UsedRamSize = 0;
    uint64_t totalRamsize = getPhysicalRamSize(&UsedRamSize);

    buf += _T("Environment Info\n");
    buf += strsprintf(_T("OS : %s (%s)\n"), getOSVersion().c_str(), is_64bit_os() ? _T("x64") : _T("x86"));
    buf += strsprintf(_T("CPU: %s\n"), cpu_info);
    if (add_ram_info) {
        cpu_info_t cpuinfo;
        get_cpu_info(&cpuinfo);
        auto write_rw_speed = [&](const TCHAR *type, int test_size) {
            if (test_size) {
                auto ram_read_speed_list = ram_speed_mt_list(test_size, RAM_SPEED_MODE_READ);
                auto ram_write_speed_list = ram_speed_mt_list(test_size, RAM_SPEED_MODE_WRITE);
                double max_read  = *std::max_element(ram_read_speed_list.begin(), ram_read_speed_list.end())  * (1.0 / 1024.0);
                double max_write = *std::max_element(ram_write_speed_list.begin(), ram_write_speed_list.end()) * (1.0 / 1024.0);
                buf += strsprintf(_T("%s: Read:%7.2fGB/s, Write:%7.2fGB/s\n"), type, max_read, max_write);
            }
            return test_size > 0;
        };
        add_ram_info = false;
        add_ram_info |= write_rw_speed(_T("L1 "), cpuinfo.caches[0].size / 1024 / 8);
        add_ram_info |= write_rw_speed(_T("L2 "), cpuinfo.caches[1].size / 1024 / 2);
        add_ram_info |= write_rw_speed(_T("L3 "), cpuinfo.caches[2].size / 1024 / 2);
        add_ram_info |= write_rw_speed(_T("RAM"), (cpuinfo.max_cache_level) ? cpuinfo.caches[cpuinfo.max_cache_level-1].size / 1024 * 8 : 96 * 1024);
    }
    buf += strsprintf(_T("%s Used %d MB, Total %d MB\n"), (add_ram_info) ? _T("    ") : _T("RAM:"), (uint32_t)(UsedRamSize >> 20), (uint32_t)(totalRamsize >> 20));
    buf += strsprintf(_T("GPU: %s\n"), gpu_info);
    return buf;
}

mfxStatus mfxBitstreamInit(mfxBitstream *pBitstream, uint32_t nSize) {
    mfxBitstreamClear(pBitstream);

    if (nullptr == (pBitstream->Data = (uint8_t *)_aligned_malloc(nSize, 32))) {
        return MFX_ERR_NULL_PTR;
    }

    pBitstream->MaxLength = nSize;
    return MFX_ERR_NONE;
}

mfxStatus mfxBitstreamExtend(mfxBitstream *pBitstream, uint32_t nSize) {
    uint8_t *pData = (uint8_t *)_aligned_malloc(nSize, 32);
    if (nullptr == pData) {
        return MFX_ERR_NULL_PTR;
    }

    auto nDataLen = pBitstream->DataLength;
    if (nDataLen) {
        memmove(pData, pBitstream->Data + pBitstream->DataOffset, nDataLen);
    }

    mfxBitstreamClear(pBitstream);

    pBitstream->Data       = pData;
    pBitstream->DataOffset = 0;
    pBitstream->DataLength = nDataLen;
    pBitstream->MaxLength  = nSize;

    return MFX_ERR_NONE;
}

void mfxBitstreamClear(mfxBitstream *pBitstream) {
    if (pBitstream->Data) {
        _aligned_free(pBitstream->Data);
    }
    memset(pBitstream, 0, sizeof(pBitstream[0]));
}

mfxStatus mfxBitstreamAppend(mfxBitstream *pBitstream, const uint8_t *data, uint32_t size) {
    mfxStatus sts = MFX_ERR_NONE;
    if (data) {
        const uint32_t new_data_length = pBitstream->DataLength + size;
        if (pBitstream->MaxLength < new_data_length) {
            if (MFX_ERR_NONE != (sts = mfxBitstreamExtend(pBitstream, new_data_length))) {
                return sts;
            }
        }

        if (pBitstream->MaxLength < new_data_length + pBitstream->DataOffset) {
            memmove(pBitstream->Data, pBitstream->Data + pBitstream->DataOffset, pBitstream->DataLength);
            pBitstream->DataOffset = 0;
        }
        memcpy(pBitstream->Data + pBitstream->DataLength + pBitstream->DataOffset, data, size);
        pBitstream->DataLength = new_data_length;
    }
    return sts;
}

mfxExtBuffer *GetExtBuffer(mfxExtBuffer **ppExtBuf, int nCount, uint32_t targetBufferId) {
    if (ppExtBuf) {
        for (int i = 0; i < nCount; i++) {
            if (ppExtBuf[i] && ppExtBuf[i]->BufferId == targetBufferId) {
                return ppExtBuf[i];
            }
        }
    }
    return nullptr;
}

int getCPUGen() {
    int CPUInfo[4] = {-1};
    __cpuid(CPUInfo, 0x01);
    bool bMOVBE  = !!(CPUInfo[2] & (1<<22));
    bool bRDRand = !!(CPUInfo[2] & (1<<30));

    __cpuid(CPUInfo, 0x07);
    bool bClflushOpt = !!(CPUInfo[1] & (1<<23));
    bool bRDSeed     = !!(CPUInfo[1] & (1<<18));
    bool bFsgsbase   = !!(CPUInfo[1] & (1));

    if (bClflushOpt)         return CPU_GEN_SKYLAKE;
    if (bRDSeed)             return CPU_GEN_BROADWELL;
    if (bMOVBE && bFsgsbase) return CPU_GEN_HASWELL;

    bool bICQ = !!(CheckEncodeFeature(true, get_mfx_libhw_version(), MFX_RATECONTROL_ICQ, MFX_CODEC_AVC) & ENC_FEATURE_CURRENT_RC);

    if (bICQ)      return CPU_GEN_AIRMONT;
    if (bFsgsbase) return CPU_GEN_IVYBRIDGE;
    if (bRDRand)   return CPU_GEN_SILVERMONT;
    return CPU_GEN_SANDYBRIDGE;
}

const TCHAR *ColorFormatToStr(uint32_t format) {
    switch (format) {
    case MFX_FOURCC_NV12:
        return _T("nv12");
    case MFX_FOURCC_NV16:
        return _T("nv16");
    case MFX_FOURCC_YV12:
        return _T("yv12");
    case MFX_FOURCC_YUY2:
        return _T("yuy2");
    case MFX_FOURCC_RGB3:
        return _T("rgb24");
    case MFX_FOURCC_RGB4:
        return _T("rgb32");
    case MFX_FOURCC_BGR4:
        return _T("bgr32");
    case MFX_FOURCC_P010:
        return _T("nv12(10bit)");
    case MFX_FOURCC_P210:
        return _T("nv16(10bit)");
    default:
        return _T("unsupported");
    }
}

const TCHAR *CodecIdToStr(uint32_t nFourCC) {
    switch (nFourCC) {
    case MFX_CODEC_AVC:
        return _T("H.264/AVC");
    case MFX_CODEC_VC1:
        return _T("VC-1");
    case MFX_CODEC_HEVC:
        return _T("HEVC");
    case MFX_CODEC_MPEG2:
        return _T("MPEG2");
    case MFX_CODEC_VP8:
        return _T("VP8");
    case MFX_CODEC_JPEG:
        return _T("JPEG");
    default:
        return _T("NOT_SUPPORTED");
    }
}

const TCHAR *TargetUsageToStr(uint16_t tu) {
    switch (tu) {
    case MFX_TARGETUSAGE_BEST_QUALITY: return _T("1 - best");
    case 2:                            return _T("2 - higher");
    case 3:                            return _T("3 - high");
    case MFX_TARGETUSAGE_BALANCED:     return _T("4 - balanced");
    case 5:                            return _T("5 - fast");
    case 6:                            return _T("6 - faster");
    case MFX_TARGETUSAGE_BEST_SPEED:   return _T("7 - fastest");
    case MFX_TARGETUSAGE_UNKNOWN:      return _T("unknown");
    default:                           return _T("unsupported");
    }
}

const TCHAR *EncmodeToStr(uint32_t enc_mode) {
    switch (enc_mode) {
    case MFX_RATECONTROL_CBR:
        return _T("Bitrate Mode - CBR");
    case MFX_RATECONTROL_VBR:
        return _T("Bitrate Mode - VBR");
    case MFX_RATECONTROL_AVBR:
        return _T("Bitrate Mode - AVBR");
    case MFX_RATECONTROL_CQP:
        return _T("Constant QP (CQP)");
    case MFX_RATECONTROL_LA:
        return _T("Bitrate Mode - Lookahead");
    case MFX_RATECONTROL_ICQ:
        return _T("ICQ (Intelligent Const. Quality)");
    case MFX_RATECONTROL_VCM:
        return _T("VCM (Video Conference Mode)");
    case MFX_RATECONTROL_LA_ICQ:
        return _T("LA-ICQ (Intelligent Const. Quality with Lookahead)");
    case MFX_RATECONTROL_LA_EXT:
        return _T("LA-EXT (Extended Lookahead)");
    case MFX_RATECONTROL_LA_HRD:
        return _T("LA-HRD (HRD compliant Lookahead)");
    case MFX_RATECONTROL_QVBR:
        return _T("Quality VBR bitrate");
    case MFX_RATECONTROL_VQP:
        return _T("Variable QP (VQP)");
    default:
        return _T("unsupported");
    }
}

const TCHAR *MemTypeToStr(uint32_t memType) {
    switch (memType) {
    case SYSTEM_MEMORY:
        return _T("system");
#if D3D_SURFACES_SUPPORT
    case D3D9_MEMORY:
        return _T("d3d9");
#if MFX_D3D11_SUPPORT
    case D3D11_MEMORY:
        return _T("d3d11");
    case HW_MEMORY:
        return _T("d3d11+d3d9");
#endif //#if MFX_D3D11_SUPPORT
#endif //#if D3D_SURFACES_SUPPORT
#ifdef LIBVA_SUPPORT
    case VA_MEMORY:
        return _T("va");
#endif
    default:
        return _T("unsupported");
    }
}
