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
#include <vector>
#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <shellapi.h>
#endif
#include <assert.h>
#include "rgy_osdep.h"
#include "qsv_pipeline.h"
#include "qsv_query.h"
#include "rgy_version.h"
#include "rgy_avutil.h"
#include "qsv_cmd.h"

tstring GetQSVEncVersion() {
    static const TCHAR *const ENABLED_INFO[] = { _T("disabled"), _T("enabled") };
    tstring version;
    version += get_encoder_version();
    version += _T("\n");
    strsprintf(_T(" Intel Media SDK API v%d.%d\n"), MFX_VERSION_MAJOR, MFX_VERSION_MINOR);
    version += _T(" reader: raw");
    if (ENABLE_AVI_READER)         version += _T(", avi");
    if (ENABLE_AVISYNTH_READER)    version += _T(", avs");
    if (ENABLE_VAPOURSYNTH_READER) version += _T(", vpy");
#if ENABLE_AVSW_READER && !FOR_AUO
    version += strsprintf(_T(", avqsv [%s]"), getHWDecSupportedCodecList().c_str());
#endif
#if !(defined(_WIN32) || defined(_WIN64))
    version += _T("\n vpp:    resize, deinterlace, denoise, detail-enhance, image-stab");
    if (ENABLE_CUSTOM_VPP) version += _T(", delego");
    if (ENABLE_LIBASS_SUBBURN != 0 && ENABLE_AVSW_READER != 0) version += _T(", sub");
#endif
    version += _T("\n");
    return version;
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

static int getAudioTrackIdx(const sInputParams* pParams, int iTrack) {
    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        if (iTrack == pParams->ppAudioSelectList[i]->trackID) {
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

static int getSubTrackIdx(const sInputParams *pParams, int iTrack) {
    for (int i = 0; i < pParams->nSubtitleSelectCount; i++) {
        if (iTrack == pParams->ppSubtitleSelectList[i]->trackID) {
            return i;
        }
    }
    return -1;
}

static int getDataTrackIdx(const sInputParams *pParams, int iTrack) {
    for (int i = 0; i < pParams->nDataSelectCount; i++) {
        if (iTrack == pParams->ppDataSelectList[i]->trackID) {
            return i;
        }
    }
    return -1;
}


struct sArgsData {
    int outputDepth = 8;
    tstring cachedlevel, cachedprofile, cachedtier;
    uint32_t nParsedAudioFile = 0;
    uint32_t nParsedAudioEncode = 0;
    uint32_t nParsedAudioCopy = 0;
    uint32_t nParsedAudioBitrate = 0;
    uint32_t nParsedAudioSamplerate = 0;
    uint32_t nParsedAudioSplit = 0;
    uint32_t nParsedAudioFilter = 0;
    uint32_t nTmpInputBuf = 0;
};

int ParseOneOption(const TCHAR *option_name, const TCHAR* strInput[], int& i, int nArgNum, sInputParams* pParams, sArgsData *argData, ParseCmdError& err) {
#define SET_ERR(app_name, errmes, opt_name, err_val) \
    err.strAppName = (app_name) ? app_name : _T(""); \
    err.strErrorMessage = (errmes) ? errmes : _T(""); \
    err.strOptionName = (opt_name) ? opt_name : _T(""); \
    err.strErrorValue = (err_val) ? err_val : _T("");

    if (0 == _tcscmp(option_name, _T("output-res"))) {
        i++;
        if (   2 != _stscanf_s(strInput[i], _T("%hdx%hd"), &pParams->nDstWidth, &pParams->nDstHeight)
            && 2 != _stscanf_s(strInput[i], _T("%hd,%hd"), &pParams->nDstWidth, &pParams->nDstHeight)
            && 2 != _stscanf_s(strInput[i], _T("%hd/%hd"), &pParams->nDstWidth, &pParams->nDstHeight)
            && 2 != _stscanf_s(strInput[i], _T("%hd:%hd"), &pParams->nDstWidth, &pParams->nDstHeight)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->vpp.bUseResize = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("input-res"))) {
        i++;
        if (   2 != _stscanf_s(strInput[i], _T("%hdx%hd"), &pParams->nWidth, &pParams->nHeight)
            && 2 != _stscanf_s(strInput[i], _T("%hd,%hd"), &pParams->nWidth, &pParams->nHeight)
            && 2 != _stscanf_s(strInput[i], _T("%hd/%hd"), &pParams->nWidth, &pParams->nHeight)
            && 2 != _stscanf_s(strInput[i], _T("%hd:%hd"), &pParams->nWidth, &pParams->nHeight)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("crop"))) {
        i++;
        if (   4 != _stscanf_s(strInput[i], _T("%d,%d,%d,%d"), &pParams->sInCrop.e.left, &pParams->sInCrop.e.up, &pParams->sInCrop.e.right, &pParams->sInCrop.e.bottom)
            && 4 != _stscanf_s(strInput[i], _T("%d:%d:%d:%d"), &pParams->sInCrop.e.left, &pParams->sInCrop.e.up, &pParams->sInCrop.e.right, &pParams->sInCrop.e.bottom)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
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
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("raw"))) {
        pParams->nInputFmt = RGY_INPUT_FMT_RAW;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("y4m"))) {
        pParams->nInputFmt = RGY_INPUT_FMT_Y4M;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("avi"))) {
        pParams->nInputFmt = RGY_INPUT_FMT_AVI;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("avs"))) {
        pParams->nInputFmt = RGY_INPUT_FMT_AVS;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpy"))) {
        pParams->nInputFmt = RGY_INPUT_FMT_VPY;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpy-mt"))) {
        pParams->nInputFmt = RGY_INPUT_FMT_VPY_MT;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("avqsv")) || 0 == _tcscmp(option_name, _T("avhw"))) {
        pParams->nInputFmt = RGY_INPUT_FMT_AVHW;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("avsw"))) {
        pParams->nInputFmt = RGY_INPUT_FMT_AVSW;
        return 0;
    }
    if (   0 == _tcscmp(option_name, _T("input-analyze"))
        || 0 == _tcscmp(option_name, _T("avqsv-analyze"))) { //互換性のため
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        } else if (value < 0) {
            SET_ERR(strInput[0], _T("input-analyze requires non-negative value."), option_name, strInput[i]);
            return 1;
        } else {
            pParams->nAVDemuxAnalyzeSec = (mfxU16)((std::min)(value, USHRT_MAX));
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("input-file"))) {
        i++;
        _tcscpy_s(pParams->strSrcFile, strInput[i]);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("output-file"))) {
        i++;
        if (!pParams->bBenchmark)
            _tcscpy_s(pParams->strDstFile, strInput[i]);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("trim"))) {
        i++;
        auto trim_str_list = split(strInput[i], _T(","));
        std::vector<sTrim> trim_list;
        for (auto trim_str : trim_str_list) {
            sTrim trim;
            if (2 != _stscanf_s(trim_str.c_str(), _T("%d:%d"), &trim.start, &trim.fin) || (trim.fin > 0 && trim.fin < trim.start)) {
                SET_ERR(strInput[0], _T("Invalid Value"), option_name, strInput[i]);
                return 1;
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
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("seek"))) {
        i++;
        int ret = 0;
        int hh = 0, mm = 0;
        float sec = 0.0f;
        if (   3 != (ret = _stscanf_s(strInput[i], _T("%d:%d:%f"),    &hh, &mm, &sec))
            && 2 != (ret = _stscanf_s(strInput[i],    _T("%d:%f"),         &mm, &sec))
            && 1 != (ret = _stscanf_s(strInput[i],       _T("%f"),              &sec))) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (ret <= 2) {
            hh = 0;
        }
        if (ret <= 1) {
            mm = 0;
        }
        if (hh < 0 || mm < 0 || sec < 0) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        if (hh > 0 && mm >= 60) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        mm += hh * 60;
        if (mm > 0 && sec >= 60.0f) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->fSeekSec = sec + mm * 60;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("video-track"))) {
        i++;
        int v = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (v == 0) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nVideoTrack = (int8_t)v;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("video-streamid"))) {
        i++;
        int v = 0;
        if (1 != _stscanf_s(strInput[i], _T("%i"), &v)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nVideoStreamId = v;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("video-tag"))) {
        i++;
        pParams->videoCodecTag = strdup(tchar_to_string(strInput[i]).c_str());
        return 0;
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
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-file"))) {
        i++;
        const TCHAR *ptr = strInput[i];
        AudioSelect *pAudioSelect = nullptr;
        int audioIdx = -1;
        int trackId = 0;
        if (_tcschr(ptr, '?') == nullptr || 1 != _stscanf(ptr, _T("%d?"), &trackId)) {
            //トラック番号を適当に発番する (カウントは1から)
            trackId = argData->nParsedAudioFile+1;
            audioIdx = getAudioTrackIdx(pParams, trackId);
            if (audioIdx < 0 || pParams->ppAudioSelectList[audioIdx]->extractFilename.length() > 0) {
                trackId = getFreeAudioTrack(pParams);
                pAudioSelect = new AudioSelect();
                pAudioSelect->trackID = trackId;
            } else {
                pAudioSelect = pParams->ppAudioSelectList[audioIdx];
            }
        } else if (i <= 0) {
            //トラック番号は1から連番で指定
            SET_ERR(strInput[0], _T("Invalid track number"), option_name, strInput[i]);
            return 1;
        } else {
            audioIdx = getAudioTrackIdx(pParams, trackId);
            if (audioIdx < 0) {
                pAudioSelect = new AudioSelect();
                pAudioSelect->trackID = trackId;
            } else {
                pAudioSelect = pParams->ppAudioSelectList[audioIdx];
            }
            ptr = _tcschr(ptr, '?') + 1;
        }
        assert(pAudioSelect != nullptr);
        const TCHAR *qtr = _tcschr(ptr, ':');
        if (qtr != NULL && !(ptr + 1 == qtr && qtr[1] == _T('\\'))) {
            pAudioSelect->extractFormat = ptr;
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
            if (pParams->ppAudioSelectList[j]->extractFilename.length() > 0
                && 0 == _tcsicmp(pParams->ppAudioSelectList[j]->extractFilename.c_str(), ptr)) {
                SET_ERR(strInput[0], _T("Same output file name is used more than twice"), option_name, nullptr);
                return 1;
            }
        }

        if (audioIdx < 0) {
            audioIdx = pParams->nAudioSelectCount;
            //新たに要素を追加
            pParams->ppAudioSelectList = (AudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
            pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
            pParams->nAudioSelectCount++;
        }
        pParams->ppAudioSelectList[audioIdx]->extractFilename = ptr;
        argData->nParsedAudioFile++;
        return 0;
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
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("input-format"))) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            pParams->pAVInputFormat = _tcsdup(strInput[i]);
        } else {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
#if ENABLE_AVSW_READER
    auto set_audio_prm = [&](std::function<void(AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr)> func_set) {
        const TCHAR *ptr = nullptr;
        const TCHAR *ptrDelim = nullptr;
        int trackId = 0;
        if (i+1 < nArgNum) {
            if (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0')) {
                i++;
                ptrDelim = _tcschr(strInput[i], _T('?'));
                ptr = (ptrDelim == nullptr) ? strInput[i] : ptrDelim+1;
            }
            if (ptrDelim != nullptr) {
                tstring temp = tstring(strInput[i]).substr(0, ptrDelim - strInput[i]);
                trackId = std::stoi(temp);
            }
        }
        AudioSelect *pAudioSelect = nullptr;
        int audioIdx = getAudioTrackIdx(pParams, trackId);
        if (audioIdx < 0) {
            pAudioSelect = new AudioSelect();
            if (trackId != 0) {
                //もし、trackID=0以外の指定であれば、
                //これまでalltrackに指定されたパラメータを探して引き継ぐ
                AudioSelect *pAudioSelectAll = nullptr;
                for (int itrack = 0; itrack < pParams->nAudioSelectCount; itrack++) {
                    if (pParams->ppAudioSelectList[itrack]->trackID == 0) {
                        pAudioSelectAll = pParams->ppAudioSelectList[itrack];
                    }
                }
                if (pAudioSelectAll) {
                    *pAudioSelect = *pAudioSelectAll;
                }
            }
            pAudioSelect->trackID = trackId;
        } else {
            pAudioSelect = pParams->ppAudioSelectList[audioIdx];
        }
        func_set(pAudioSelect, trackId, ptr);
        if (trackId == 0) {
            for (int itrack = 0; itrack < pParams->nAudioSelectCount; itrack++) {
                func_set(pParams->ppAudioSelectList[itrack], trackId, ptr);
            }
        }

        if (audioIdx < 0) {
            audioIdx = pParams->nAudioSelectCount;
            //新たに要素を追加
            pParams->ppAudioSelectList = (AudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
            pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
            pParams->nAudioSelectCount++;
        }
        return 0;
    };
    auto set_sub_prm = [&](std::function<void(SubtitleSelect *pSubSelect, int trackId, const TCHAR *prmstr)> func_set) {
        const TCHAR *ptr = nullptr;
        const TCHAR *ptrDelim = nullptr;
        int trackId = 0;
        if (i+1 < nArgNum) {
            if (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0')) {
                i++;
                ptrDelim = _tcschr(strInput[i], _T('?'));
                ptr = (ptrDelim == nullptr) ? strInput[i] : ptrDelim+1;
            }
            if (ptrDelim != nullptr) {
                tstring temp = tstring(strInput[i]).substr(0, ptrDelim - strInput[i]);
                trackId = std::stoi(temp);
            }
        }
        SubtitleSelect *pSubSelect = nullptr;
        int subIdx = getSubTrackIdx(pParams, trackId);
        if (subIdx < 0) {
            pSubSelect = new SubtitleSelect();
            if (trackId != 0) {
                //もし、trackID=0以外の指定であれば、
                //これまでalltrackに指定されたパラメータを探して引き継ぐ
                SubtitleSelect *pSubSelectAll = nullptr;
                for (int itrack = 0; itrack < pParams->nSubtitleSelectCount; itrack++) {
                    if (pParams->ppSubtitleSelectList[itrack]->trackID == 0) {
                        pSubSelectAll = pParams->ppSubtitleSelectList[itrack];
                    }
                }
                if (pSubSelectAll) {
                    *pSubSelect = *pSubSelectAll;
                }
            }
            pSubSelect->trackID = trackId;
        } else {
            pSubSelect = pParams->ppSubtitleSelectList[subIdx];
        }
        func_set(pSubSelect, trackId, ptr);
        if (trackId == 0) {
            for (int itrack = 0; itrack < pParams->nSubtitleSelectCount; itrack++) {
                func_set(pParams->ppSubtitleSelectList[itrack], trackId, ptr);
            }
        }

        if (subIdx < 0) {
            subIdx = pParams->nSubtitleSelectCount;
            //新たに要素を追加
            pParams->ppSubtitleSelectList = (SubtitleSelect **)realloc(pParams->ppSubtitleSelectList, sizeof(pParams->ppSubtitleSelectList[0]) * (pParams->nSubtitleSelectCount + 1));
            pParams->ppSubtitleSelectList[pParams->nSubtitleSelectCount] = pSubSelect;
            pParams->nSubtitleSelectCount++;
        }
        return 0;
    };
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
                    SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                    return 1;
                } else {
                    trackSet.insert(iTrack);
                }
            }
        } else {
            trackSet.insert(0);
        }

        for (auto it = trackSet.begin(); it != trackSet.end(); it++) {
            int trackId = *it;
            AudioSelect *pAudioSelect = nullptr;
            int audioIdx = getAudioTrackIdx(pParams, trackId);
            if (audioIdx < 0) {
                pAudioSelect = new AudioSelect();
                pAudioSelect->trackID = trackId;
            } else {
                pAudioSelect = pParams->ppAudioSelectList[audioIdx];
            }
            pAudioSelect->encCodec = RGY_AVCODEC_COPY;

            if (audioIdx < 0) {
                audioIdx = pParams->nAudioSelectCount;
                //新たに要素を追加
                pParams->ppAudioSelectList = (AudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
                pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
                pParams->nAudioSelectCount++;
            }
            argData->nParsedAudioCopy++;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-codec"))) {
        pParams->nAVMux |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
            if (trackId != 0 || pAudioSelect->encCodec.length() == 0) {
                if (prmstr == nullptr) {
                    pAudioSelect->encCodec = RGY_AVCODEC_AUTO;
                } else {
                    tstring prm = prmstr;
                    auto delimEnc = prm.find(_T(":"));
                    auto delimDec = prm.find(_T("#"));
                    pAudioSelect->encCodec = prm.substr(0, std::min(delimEnc, delimDec));
                    if (delimEnc != tstring::npos) {
                        pAudioSelect->encCodecPrm = prm.substr(delimEnc + 1, (delimEnc < delimDec) ? delimDec - delimEnc - 1 : tstring::npos);
                    }
                    if (delimDec != tstring::npos) {
                        pAudioSelect->decCodecPrm = prm.substr(delimDec + 1, (delimDec < delimEnc) ? delimEnc - delimDec - 1 : tstring::npos);
                    }
                }
            }
        });
        if (ret) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return ret;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-profile"))) {
        pParams->nAVMux |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
            if (trackId != 0 || pAudioSelect->encCodecProfile.length() == 0) {
                pAudioSelect->encCodecProfile = prmstr;
            }
        });
        if (ret) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return ret;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-bitrate"))) {
        try {
            auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pAudioSelect->encBitrate == 0) {
                    pAudioSelect->encBitrate = std::stoi(prmstr);
                }
            });
            return ret;
        } catch (...) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
    }
    if (0 == _tcscmp(option_name, _T("audio-ignore-decode-error"))) {
        i++;
        uint32_t value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nAudioIgnoreDecodeError = value;
        return 0;
    }
    //互換性のため残す
    if (0 == _tcscmp(option_name, _T("audio-ignore-notrack-error"))) {
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-samplerate"))) {
        try {
            auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pAudioSelect->encSamplingRate == 0) {
                    pAudioSelect->encSamplingRate = std::stoi(prmstr);
                }
            });
            return ret;
        } catch (...) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
    }
    if (0 == _tcscmp(option_name, _T("audio-resampler"))) {
        i++;
        mfxI32 v;
        if (PARSE_ERROR_FLAG != (v = get_value_from_chr(list_resampler, strInput[i]))) {
            pParams->nAudioResampler = (mfxU8)v;
        } else if (1 == _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_resampler) - 1) {
            pParams->nAudioResampler = (mfxU8)v;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-stream"))) {
#if !FOR_AUO
        //ここで、av_get_channel_layout()を使うため、チェックする必要がある
        if (!check_avcodec_dll()) {
            _ftprintf(stderr, _T("%s\n--audio-stream could not be used.\n"), error_mes_avcodec_dll_not_found().c_str());
            return 1;
        }

        try {
            auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || (pAudioSelect->pnStreamChannelSelect[0] == 0 && pAudioSelect->pnStreamChannelOut[0] == 0)) {
                    auto streamSelectList = split(tchar_to_string(prmstr), ",");
                    if (streamSelectList.size() > _countof(pAudioSelect->pnStreamChannelSelect)) {
                        return 1;
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
                return 0;
            });
            if (ret) {
                SET_ERR(strInput[0], _T("Too much streams splitted"), option_name, strInput[i]);
                return ret;
            }
            return ret;
        } catch (...) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
#else
        //QSVEnc.auoでは、dllがあるかわからないので、libavutilの関数 av_get_channel_layout()を実行してはならない
        return 1;
#endif
    }
    if (0 == _tcscmp(option_name, _T("audio-filter"))) {
        try {
            auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pAudioSelect->filter.length() == 0) {
                    pAudioSelect->filter = prmstr;
                }
            });
            return ret;
        } catch (...) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
    }
#endif //#if ENABLE_AVSW_READER
    if (   0 == _tcscmp(option_name, _T("chapter-copy"))
        || 0 == _tcscmp(option_name, _T("copy-chapter"))) {
        pParams->bCopyChapter = TRUE;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("chapter"))) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            pParams->pChapterFile = _tcsdup(strInput[i]);
        } else {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("chapter-no-trim"))) {
        pParams->bChapterNoTrim = TRUE;
        return 0;
    }
    if (   0 == _tcscmp(option_name, _T("sub-copy"))
        || 0 == _tcscmp(option_name, _T("copy-sub"))) {
        pParams->nAVMux |= (RGY_MUX_VIDEO | RGY_MUX_SUBTITLE);
        std::map<int, SubtitleSelect> trackSet; //重複しないように
        if (i+1 < nArgNum && (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0'))) {
            i++;
            auto trackListStr = split(strInput[i], _T(","));
            for (auto str : trackListStr) {
                int iTrack = 0;
                if (1 != _stscanf(str.c_str(), _T("%d"), &iTrack) || iTrack < 1) {
                    if (str == _T("asdata")) {
                        trackSet[iTrack].trackID = iTrack;
                        trackSet[iTrack].encCodec = RGY_AVCODEC_COPY;
                        trackSet[iTrack].asdata = true;
                    } else {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return 1;
                    }
                } else {
                    trackSet[iTrack].trackID = iTrack;
                    trackSet[iTrack].encCodec = RGY_AVCODEC_COPY;
                    auto options = str.find(_T('?'));
                    if (str.substr(options+1) == _T("asdata")) {
                        trackSet[iTrack].asdata = true;
                    }
                }
            }
        } else {
            trackSet[0].trackID = 0;
            trackSet[0].encCodec = RGY_AVCODEC_COPY;
        }

        for (auto it = trackSet.begin(); it != trackSet.end(); it++) {
            int trackId = it->first;
            SubtitleSelect *pSubtitleSelect = nullptr;
            int subIdx = getSubTrackIdx(pParams, trackId);
            if (subIdx < 0) {
                pSubtitleSelect = new SubtitleSelect();
            } else {
                pSubtitleSelect = pParams->ppSubtitleSelectList[subIdx];
            }
            pSubtitleSelect[0] = it->second;

            if (subIdx < 0) {
                subIdx = pParams->nSubtitleSelectCount;
                //新たに要素を追加
                pParams->ppSubtitleSelectList = (SubtitleSelect **)realloc(pParams->ppSubtitleSelectList, sizeof(pParams->ppSubtitleSelectList[0]) * (pParams->nSubtitleSelectCount + 1));
                pParams->ppSubtitleSelectList[pParams->nSubtitleSelectCount] = pSubtitleSelect;
                pParams->nSubtitleSelectCount++;
            }
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("sub-codec"))) {
        pParams->nAVMux |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        auto ret = set_sub_prm([](SubtitleSelect *pSubSelect, int trackId, const TCHAR *prmstr) {
            if (trackId != 0 || pSubSelect->encCodec.length() == 0) {
                if (prmstr == nullptr) {
                    pSubSelect->encCodec = RGY_AVCODEC_AUTO;
                } else {
                    tstring prm = prmstr;
                    auto delimEnc = prm.find(_T(":"));
                    auto delimDec = prm.find(_T("#"));
                    pSubSelect->encCodec = prm.substr(0, std::min(delimEnc, delimDec));
                    if (delimEnc != tstring::npos) {
                        pSubSelect->encCodecPrm = prm.substr(delimEnc + 1, (delimEnc < delimDec) ? delimDec - delimEnc - 1 : tstring::npos);
                    }
                    if (delimDec != tstring::npos) {
                        pSubSelect->decCodecPrm = prm.substr(delimDec + 1, (delimDec < delimEnc) ? delimEnc - delimDec - 1 : tstring::npos);
                    }
                }
            }
        });
        if (ret) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return ret;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("caption2ass"))) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            C2AFormat format = FORMAT_INVALID;
            if (PARSE_ERROR_FLAG != (format = (C2AFormat)get_value_from_chr(list_caption2ass, strInput[i]))) {
                pParams->caption2ass = format;
            } else {
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return 1;
            }
        } else {
            pParams->caption2ass = FORMAT_SRT;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-caption2ass"))) {
        pParams->caption2ass = FORMAT_INVALID;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("data-copy"))) {
        pParams->nAVMux |= (RGY_MUX_VIDEO | RGY_MUX_SUBTITLE);
        std::map<int, DataSelect> trackSet; //重複しないように
        if (i+1 < nArgNum && (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0'))) {
            i++;
            auto trackListStr = split(strInput[i], _T(","));
            for (auto str : trackListStr) {
                int iTrack = 0;
                if (1 != _stscanf(str.c_str(), _T("%d"), &iTrack) || iTrack < 1) {
                    SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                    return 1;
                } else {
                    trackSet[iTrack].trackID = iTrack;
                }
            }
        } else {
            trackSet[0].trackID = 0;
        }

        for (auto it = trackSet.begin(); it != trackSet.end(); it++) {
            int trackId = it->first;
            DataSelect *pDataSelect = nullptr;
            int dataIdx = getDataTrackIdx(pParams, trackId);
            if (dataIdx < 0) {
                pDataSelect = new DataSelect();
            } else {
                pDataSelect = pParams->ppDataSelectList[dataIdx];
            }
            pDataSelect[0] = it->second;

            if (dataIdx < 0) {
                dataIdx = pParams->nDataSelectCount;
                //新たに要素を追加
                pParams->ppDataSelectList = (DataSelect **)realloc(pParams->ppDataSelectList, sizeof(pParams->ppDataSelectList[0]) * (pParams->nDataSelectCount + 1));
                pParams->ppDataSelectList[pParams->nDataSelectCount] = pDataSelect;
                pParams->nDataSelectCount++;
            }
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("avsync"))) {
        int value = 0;
        i++;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_avsync, strInput[i]))) {
            pParams->nAVSyncMode = (RGYAVSync)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("mux-option"))) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            auto ptr = _tcschr(strInput[i], ':');
            if (ptr == nullptr) {
                SET_ERR(strInput[0], _T("invalid value"), option_name, strInput[i]);
                return 1;
            } else {
                if (pParams->pMuxOpt == nullptr) {
                    pParams->pMuxOpt = new muxOptList();
                }
                pParams->pMuxOpt->push_back(std::make_pair<tstring, tstring>(tstring(strInput[i]).substr(0, ptr - strInput[i]), tstring(ptr+1)));
            }
        } else {
            SET_ERR(strInput[0], _T("invalid option"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("quality"))) {
        i++;
        int value = MFX_TARGETUSAGE_BALANCED;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->nTargetUsage = (mfxU16)clamp(value, MFX_TARGETUSAGE_BEST_QUALITY, MFX_TARGETUSAGE_BEST_SPEED);
        } else if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_quality_for_option, strInput[i]))) {
            pParams->nTargetUsage = (mfxU16)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("level"))) {
        if (i+1 < nArgNum) {
            i++;
            argData->cachedlevel = strInput[i];
        } else {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("profile"))) {
        if (i+1 < nArgNum) {
            i++;
            argData->cachedprofile = strInput[i];
        } else {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("tier"))) {
        if (i+1 < nArgNum) {
            i++;
            argData->cachedtier = strInput[i];
        } else {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("output-depth"))) {
        i++;
        int value = 0;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_hevc_output_depth, strInput[i]))) {
            argData->outputDepth = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
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
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nSlices)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("gop-len"))) {
        i++;
        if (0 == _tcsnccmp(strInput[i], _T("auto"), _tcslen(_T("auto")))) {
            pParams->nGOPLength = 0;
        } else if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nGOPLength)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
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
        pParams->nRepartitionCheck = MFX_CODINGOPTION_ON;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-repartition-check"))) {
        pParams->nRepartitionCheck = MFX_CODINGOPTION_OFF;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("fade-detect"))) {
        pParams->nFadeDetect = MFX_CODINGOPTION_ON;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-fade-detect"))) {
        pParams->nFadeDetect = MFX_CODINGOPTION_OFF;
        return 0;
    }
    if (   0 == _tcscmp(option_name, _T("lookahead-ds"))
        || 0 == _tcscmp(option_name, _T("la-quality"))) {
        i++;
        int value = MFX_LOOKAHEAD_DS_UNKNOWN;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_lookahead_ds, strInput[i]))) {
            pParams->nLookaheadDS = (mfxU16)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("trellis"))) {
        i++;
        int value = MFX_TRELLIS_UNKNOWN;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_avc_trellis_for_options, strInput[i]))) {
            pParams->nTrellis = (mfxU16)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
    if (0 == _tcscmp(option_name, _T("interlace"))) {
        i++;
        int value = MFX_PICSTRUCT_PROGRESSIVE;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_interlaced, strInput[i]))) {
            pParams->nPicStruct = (mfxU16)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("tff"))) {
        pParams->nPicStruct = MFX_PICSTRUCT_FIELD_TFF;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("bff"))) {
        pParams->nPicStruct = MFX_PICSTRUCT_FIELD_BFF;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("la"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nEncMode = MFX_RATECONTROL_LA;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("icq"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nICQQuality)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nEncMode = MFX_RATECONTROL_ICQ;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("la-icq"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nICQQuality)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nEncMode = MFX_RATECONTROL_LA_ICQ;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("la-hrd"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nEncMode = MFX_RATECONTROL_LA_HRD;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vcm"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nEncMode = MFX_RATECONTROL_VCM;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vbr"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nEncMode = MFX_RATECONTROL_VBR;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("cbr"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nEncMode = MFX_RATECONTROL_CBR;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("avbr"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nEncMode = MFX_RATECONTROL_AVBR;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("qvbr"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nBitRate)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nEncMode = MFX_RATECONTROL_QVBR;
        return 0;
    }
    if (   0 == _tcscmp(option_name, _T("qvbr-q"))
        || 0 == _tcscmp(option_name, _T("qvbr-quality"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nQVBRQuality)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nEncMode = MFX_RATECONTROL_QVBR;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("fallback-rc"))) {
        pParams->nFallback = 1;
        return 0;
    }
    if (   0 == _tcscmp(option_name, _T("max-bitrate"))
        || 0 == _tcscmp(option_name, _T("maxbitrate"))) //互換性のため
    {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->nMaxBitrate)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vbv-bufsize"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &pParams->VBVBufsize)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("la-depth"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nLookaheadDepth)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("la-window-size"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nWinBRCSize)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("cqp"))) {
        i++;
        if (   3 != _stscanf_s(strInput[i], _T("%hd:%hd:%hd"), &pParams->nQPI, &pParams->nQPP, &pParams->nQPB)
            && 3 != _stscanf_s(strInput[i], _T("%hd,%hd,%hd"), &pParams->nQPI, &pParams->nQPP, &pParams->nQPB)
            && 3 != _stscanf_s(strInput[i], _T("%hd/%hd/%hd"), &pParams->nQPI, &pParams->nQPP, &pParams->nQPB)) {
            if (1 == _stscanf_s(strInput[i], _T("%hd"), &pParams->nQPI)) {
                pParams->nQPP = pParams->nQPI;
                pParams->nQPB = pParams->nQPI;
            } else {
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return 1;
            }
        }
        pParams->nEncMode = (mfxU16)MFX_RATECONTROL_CQP;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("avbr-unitsize"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nAVBRConvergence)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    //if (0 == _tcscmp(option_name, _T("avbr-range")))
    //{
    //    double accuracy;
    //    if (1 != _stscanf_s(strArgument, _T("%f"), &accuracy)) {
    //        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
    //        return 1;
    //    }
    //    pParams->nAVBRAccuarcy = (mfxU16)(accuracy * 10 + 0.5);
    //    return 0;
    //}
    else if (0 == _tcscmp(option_name, _T("fixed-func"))) {
        pParams->bUseFixedFunc = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-fixed-func"))) {
        pParams->bUseFixedFunc = false;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("ref"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nRef)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("bframes"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%hd"), &pParams->nBframes)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
    //if (0 == _tcscmp(option_name, _T("extbrc"))) {
    //    pParams->bExtBRC = true;
    //    return 0;
    //}
    //if (0 == _tcscmp(option_name, _T("no-extbrc"))) {
    //    pParams->bExtBRC = false;
    //    return 0;
    //}
    if (0 == _tcscmp(option_name, _T("adapt-ltr"))) {
        pParams->extBrcAdaptiveLTR = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-adapt-ltr"))) {
        pParams->extBrcAdaptiveLTR = false;
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
    if (0 == _tcscmp(option_name, _T("no-intra-refresh"))) {
        pParams->bIntraRefresh = false;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("intra-refresh"))) {
        pParams->bIntraRefresh = true;
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
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return 1;
            }
        }
        uint8_t *limit = (0 == _tcscmp(option_name, _T("qpmin")) || 0 == _tcscmp(option_name, _T("qp-min"))) ? pParams->nQPMin : pParams->nQPMax;
        for (int j = 0; j < 3; j++) {
            limit[j] = (uint8_t)clamp(qpLimit[j], 0, 51);
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("qp-offset"))) {
        i++;
        auto values = split(strInput[i], _T(":"), true);
        if (values.size() == 0) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (values.size() > 8) {
            SET_ERR(strInput[0], strsprintf(_T("qp-offset value could be set up to 8 layers, but was set for %d layers.\n"), (int)values.size()).c_str(), option_name, strInput[i]);
            return 1;
        }
        uint32_t iv = 0;
        for (; iv < values.size(); iv++) {
            TCHAR *eptr = nullptr;
            int v = _tcstol(values[iv].c_str(), &eptr, 0);
            if (v == 0 && (eptr != nullptr || *eptr == ' ')) {
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[iv]);
                return 1;
            }
            if (v < -51 || v > 51) {
                SET_ERR(strInput[0], _T("qp-offset value should be in range of -51 - 51.\n"), option_name, strInput[i]);
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
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->bGlobalMotionAdjust = true;
            pParams->nMVCostScaling = (mfxU8)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
        mfxI32 v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_pred_block_size) - 1) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nInterPred = (mfxU16)list_pred_block_size[v].value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("intra-pred"))) {
        i++;
        mfxI32 v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_pred_block_size) - 1) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nIntraPred = (mfxU16)list_pred_block_size[v].value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("mv-precision"))) {
        i++;
        mfxI32 v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_mv_presicion) - 1) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nMVPrecision = (mfxU16)list_mv_presicion[v].value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("mv-search"))) {
        i++;
        mfxI32 v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->MVSearchWindow.x = (mfxU16)clamp(v, 0, 128);
        pParams->MVSearchWindow.y = (mfxU16)clamp(v, 0, 128);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("sharpness"))) {
        i++;
        mfxI32 v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < 8) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nVP8Sharpness = (mfxU8)v;
        return 0;
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
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return 1;
            }
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("log-level"))) {
        i++;
        mfxI32 v;
        if (PARSE_ERROR_FLAG != (v = get_value_from_chr(list_log_level, strInput[i]))) {
            pParams->nLogLevel = (mfxI16)v;
        } else if (1 == _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_log_level) - 1) {
            pParams->nLogLevel = (mfxI16)v;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
#ifdef D3D_SURFACES_SUPPORT
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
#ifdef LIBVA_SUPPORT
    if (0 == _tcscmp(option_name, _T("va"))) {
        pParams->memType = D3D9_MEMORY;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("disable-va"))) {
        pParams->memType = SYSTEM_MEMORY;
        return 0;
    }
#endif //#ifdef LIBVA_SUPPORT
    if (0 == _tcscmp(option_name, _T("aud"))) {
        pParams->bOutputAud = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("pic-struct"))) {
        pParams->bOutputPicStruct = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("async-depth"))) {
        i++;
        int v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v) || v < 0 || QSV_ASYNC_DEPTH_MAX < v) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nAsyncDepth = (mfxU16)v;
        return 0;
    }
#if ENABLE_SESSION_THREAD_CONFIG
    if (0 == _tcscmp(option_name, _T("session-threads"))) {
        i++;
        int v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v) || v < 0 || QSV_SESSION_THREAD_MAX < v) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nSessionThreads = (mfxU16)v;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("session-thread-priority"))
        || 0 == _tcscmp(option_name, _T("session-threads-priority"))) {
        i++;
        mfxI32 v;
        if (PARSE_ERROR_FLAG == (v = get_value_from_chr(list_priority, strInput[i]))
            && 1 != _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_log_level) - 1) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nSessionThreadPriority = (mfxU16)v;
        return 0;
    }
#endif
    if (0 == _tcscmp(option_name, _T("vpp-denoise"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->vpp.denoise.enable = true;
        pParams->vpp.denoise.strength = value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-vpp-denoise"))) {
        pParams->vpp.denoise.enable = false;
        if (strInput[i+1][0] != _T('-')) {
            i++;
            int value = 0;
            if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return 1;
            }
            pParams->vpp.denoise.strength = value;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-mctf"))) {
        pParams->vpp.mctf.enable = true;
        pParams->vpp.mctf.strength = 0;
        if (strInput[i+1][0] != _T('-')) {
            i++;
            int value = 0;
            if (_tcsicmp(strInput[i], _T("auto")) == 0) {
                value = 0;
            } else if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return 1;
            }
            pParams->vpp.mctf.strength = value;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-vpp-mctf"))) {
        pParams->vpp.mctf.enable = false;
        pParams->vpp.mctf.strength = 0;
        if (strInput[i+1][0] != _T('-')) {
            i++;
            int value = 0;
            if (_tcsicmp(strInput[i], _T("auto")) == 0) {
                value = 0;
            } if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return 1;
            }
            pParams->vpp.mctf.strength = value;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-detail-enhance"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->vpp.detail.enable = true;
        pParams->vpp.detail.strength = value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-vpp-detail-enhance"))) {
        pParams->vpp.detail.enable = false;
        if (strInput[i+1][0] != _T('-')) {
            i++;
            int value = 0;
            if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return 1;
            }
            pParams->vpp.detail.strength = value;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-deinterlace"))) {
        i++;
        int value = get_value_from_chr(list_deinterlace, strInput[i]);
        if (PARSE_ERROR_FLAG == value) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->vpp.bEnable = true;
        pParams->vpp.deinterlace = value;
        if (pParams->vpp.deinterlace == MFX_DEINTERLACE_IT_MANUAL) {
            i++;
            if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_telecine_patterns, strInput[i]))) {
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return 1;
            } else {
                pParams->vpp.telecinePattern = value;
            }
        }
        if (pParams->vpp.deinterlace != MFX_DEINTERLACE_NONE
            && pParams->nPicStruct == MFX_PICSTRUCT_PROGRESSIVE) {
            pParams->nPicStruct = MFX_PICSTRUCT_FIELD_TFF;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-image-stab"))) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->vpp.imageStabilizer = value;
        } else if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_vpp_image_stabilizer, strInput[i]))) {
            pParams->vpp.imageStabilizer = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-fps-conv"))) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->vpp.fpsConversion = value;
        } else if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_vpp_fps_conversion, strInput[i]))) {
            pParams->vpp.fpsConversion = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-half-turn"))) {
        pParams->vpp.halfTurn = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-rotate"))) {
        i++;
        int value = 0;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_vpp_rotate_angle, strInput[i]))) {
            pParams->vpp.rotate = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-mirror"))) {
        i++;
        int value = 0;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_vpp_mirroring, strInput[i]))) {
            pParams->vpp.mirrorType = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-resize"))
        || 0 == _tcscmp(option_name, _T("vpp-scaling"))) {
        i++;
        int value = 0;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_vpp_scaling_quality, strInput[i]))) {
            pParams->vpp.scalingQuality = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
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
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-sub-charset"))) {
        if (i+1 < nArgNum && (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0'))) {
            i++;
            if (pParams->vpp.subburn.pCharEnc) {
                free(pParams->vpp.subburn.pCharEnc);
            }
            pParams->vpp.subburn.pCharEnc = _tcsdup(strInput[i]);
        } else {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-sub-shaping"))) {
        i++;
        int v;
        if (PARSE_ERROR_FLAG != (v = get_value_from_chr(list_vpp_sub_shaping, strInput[i]))) {
            pParams->vpp.subburn.nShaping = v;
        } else if (1 == _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_vpp_sub_shaping) - 1) {
            pParams->vpp.subburn.nShaping = v;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
#endif //#if ENABLE_AVSW_READER && ENABLE_LIBASS_SUBBURN
    if (   0 == _tcscmp(option_name, _T("vpp-delogo"))
        || 0 == _tcscmp(option_name, _T("vpp-delogo-file"))) {
        i++;
        int filename_len = (int)_tcslen(strInput[i]);
        pParams->vpp.delogo.pFilePath = (TCHAR *)calloc(filename_len + 1, sizeof(pParams->vpp.delogo.pFilePath[0]));
        memcpy(pParams->vpp.delogo.pFilePath, strInput[i], sizeof(pParams->vpp.delogo.pFilePath[0]) * filename_len);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-delogo-select"))) {
        i++;
        int filename_len = (int)_tcslen(strInput[i]);
        pParams->vpp.delogo.pSelect = (TCHAR *)calloc(filename_len + 1, sizeof(pParams->vpp.delogo.pSelect[0]));
        memcpy(pParams->vpp.delogo.pSelect, strInput[i], sizeof(pParams->vpp.delogo.pSelect[0]) * filename_len);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-delogo-pos"))) {
        i++;
        int posOffsetx = 0, posOffsety = 0;
        if (   2 != _stscanf_s(strInput[i], _T("%dx%d"), &posOffsetx, &posOffsety)
            && 2 != _stscanf_s(strInput[i], _T("%d,%d"), &posOffsetx, &posOffsety)
            && 2 != _stscanf_s(strInput[i], _T("%d/%d"), &posOffsetx, &posOffsety)
            && 2 != _stscanf_s(strInput[i], _T("%d:%d"), &posOffsetx, &posOffsety)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->vpp.delogo.posOffset = std::make_pair(posOffsetx, posOffsety);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-delogo-depth"))) {
        i++;
        int depth;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &depth)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->vpp.delogo.depth = clamp(depth, 0, 255);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-delogo-y"))) {
        i++;
        int value;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->vpp.delogo.YOffset = value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-delogo-cb"))) {
        i++;
        int value;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->vpp.delogo.CbOffset = value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-delogo-cr"))) {
        i++;
        int value;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->vpp.delogo.CrOffset = value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("vpp-delogo-add"))) {
        pParams->vpp.delogo.add = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-vpp-delogo-add"))) {
        pParams->vpp.delogo.add = false;
        return 0;
    }
#endif //#if ENABLE_CUSTOM_VPP
    if (0 == _tcscmp(option_name, _T("input-buf"))) {
        i++;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &argData->nTmpInputBuf)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("output-buf"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < 0) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nOutputBufSizeMB = (int16_t)(std::min)(value, RGY_OUTPUT_BUF_MB_MAX);
        return 0;
    }
#if defined(_WIN32) || defined(_WIN64)
    if (0 == _tcscmp(option_name, _T("mfx-thread"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < -1) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nSessionThreads = (int16_t)value;
        return 0;
    }
#endif
    if (0 == _tcscmp(option_name, _T("input-thread"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < -1 || value >= 2) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nInputThread = (int8_t)value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-output-thread"))) {
        pParams->nOutputThread = 0;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("output-thread"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < -1 || value >= 2) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nOutputThread = (int8_t)value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-thread"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < -1 || value >= 3) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nAudioThread = (int8_t)value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("thread-csp"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < -1 || value >= 2) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->threadCsp = value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("simd-csp"))) {
        i++;
        int value = 0;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_simd, strInput[i]))) {
            pParams->vpp.rotate = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("min-memory"))) {
        pParams->nOutputThread = 0;
        pParams->nAudioThread = 0;
        pParams->nAsyncDepth = 1;
        argData->nTmpInputBuf = 1;
        pParams->nOutputBufSizeMB = 0;
        pParams->nSessionThreads = 2;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("max-procfps"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < 0) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nProcSpeedLimit = (uint16_t)(std::min)(value, (int)UINT16_MAX);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("log"))) {
        i++;
        int filename_len = (int)_tcslen(strInput[i]);
        pParams->pStrLogFile = (TCHAR *)calloc(filename_len + 1, sizeof(pParams->pStrLogFile[0]));
        memcpy(pParams->pStrLogFile, strInput[i], sizeof(pParams->pStrLogFile[0]) * filename_len);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("log-framelist"))) {
        i++;
        int filename_len = (int)_tcslen(strInput[i]);
        pParams->pFramePosListLog = (TCHAR *)calloc(filename_len + 1, sizeof(pParams->pFramePosListLog[0]));
        memcpy(pParams->pFramePosListLog, strInput[i], sizeof(pParams->pFramePosListLog[0]) * filename_len);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("log-mux-ts"))) {
        i++;
        pParams->pMuxVidTsLogFile = _tcsdup(strInput[i]);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("log-copy-framedata"))) {
        i++;
        pParams->pLogCopyFrameData = _tcsdup(strInput[i]);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("colormatrix"))) {
        i++;
        int value;
        if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_colormatrix, strInput[i]))) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->ColorMatrix = (mfxU16)value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("colorprim"))) {
        i++;
        int value;
        if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_colorprim, strInput[i]))) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->ColorPrim = (mfxU16)value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("transfer"))) {
        i++;
        int value;
        if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_transfer, strInput[i]))) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->Transfer = (mfxU16)value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("videoformat"))) {
        i++;
        int value;
        if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_videoformat, strInput[i]))) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->VideoFormat = (mfxU16)value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("fullrange"))) {
        pParams->bFullrange = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("chromaloc"))) {
        i++;
        int value;
        if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_chromaloc, strInput[i]))) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->chromaloc = value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("max-cll"))) {
        i++;
        pParams->sMaxCll = strdup(tchar_to_string(strInput[i]).c_str());
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("master-display"))) {
        i++;
        pParams->sMasterDisplay = strdup(tchar_to_string(strInput[i]).c_str());
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("sar"))) {
        i++;
        if (   2 != _stscanf_s(strInput[i], _T("%dx%d"), &pParams->nPAR[0], &pParams->nPAR[1])
            && 2 != _stscanf_s(strInput[i], _T("%d,%d"), &pParams->nPAR[0], &pParams->nPAR[1])
            && 2 != _stscanf_s(strInput[i], _T("%d/%d"), &pParams->nPAR[0], &pParams->nPAR[1])
            && 2 != _stscanf_s(strInput[i], _T("%d:%d"), &pParams->nPAR[0], &pParams->nPAR[1])) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("benchmark"))) {
        i++;
        pParams->bBenchmark = TRUE;
        _tcscpy_s(pParams->strDstFile, strInput[i]);
        return 0;
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
                } else if ((nQuality = get_value_from_chr(list_quality_for_option, strInput[i])) > 0) {
                    pParams->nBenchQuality |= 1 << nQuality;
                } else {
                    SET_ERR(strInput[i], _T("Unknown value"), option_name, strInput[i]);
                    return 1;
                }
            }
        }
        return 0;
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
                    SET_ERR(item.c_str(), _T("Unknown value"), option_name, strInput[i]);
                    return 1;
                }
                pParams->nPerfMonitorSelect |= value;
            }
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("perf-monitor-interval"))) {
        i++;
        mfxI32 v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nPerfMonitorInterval = std::max(50, v);
        return 0;
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
                    SET_ERR(item.c_str(), _T("Unknown value"), option_name, strInput[i]);
                    return 1;
                }
                pParams->nPerfMonitorSelectMatplot |= value;
            }
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("python"))) {
        i++;
        pParams->pPythonPath = _tcsdup(strInput[i]);
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
    tstring mes = _T("Unknown option: --");
    mes += option_name;
    SET_ERR(strInput[0], (TCHAR *)mes.c_str(), nullptr, strInput[i]);
    return 1;
}

int parse_cmd(sInputParams *pParams, const TCHAR *strInput[], int nArgNum, ParseCmdError& err, bool ignore_parse_err) {
    if (!pParams) {
        return 0;
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
                    SET_ERR(strInput[0], strsprintf(_T("Unknown options: \"%s\""), strInput[i]).c_str(), nullptr, nullptr);
                    return 1;
                }
            } else {
                if (ignore_parse_err) continue;
                SET_ERR(strInput[0], strsprintf(_T("Invalid options: \"%s\""), strInput[i]).c_str(), nullptr, nullptr);
                return 1;
            }
        }

        if (option_name == NULL) {
            if (ignore_parse_err) continue;
            SET_ERR(strInput[0], strsprintf(_T("Unknown option: \"%s\""), strInput[i]).c_str(), nullptr, nullptr);
            return 1;
        }
        auto sts = ParseOneOption(option_name, strInput, i, nArgNum, pParams, &argsData, err);
        if (!ignore_parse_err && sts != 0) {
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
            SET_ERR(strInput[0], _T("Unknown value"), _T("level"), nullptr);
            return 1;
        }
    }
    if (argsData.cachedprofile.length() > 0) {
        const auto desc = get_profile_list(pParams->CodecId);
        int value = 0;
        if (desc != nullptr && PARSE_ERROR_FLAG != (value = get_value_from_chr(desc, argsData.cachedprofile.c_str()))) {
            pParams->CodecProfile = (mfxU16)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), _T("profile"), nullptr);
            return 1;
        }
        if (pParams->CodecId == MFX_CODEC_HEVC
            && argsData.outputDepth == 10
            && (pParams->CodecProfile == 0 || pParams->CodecProfile == MFX_PROFILE_HEVC_MAIN)) {
            pParams->CodecProfile = MFX_PROFILE_HEVC_MAIN10;
        }
    }
    if (argsData.cachedtier.length() > 0 && pParams->CodecId == MFX_CODEC_HEVC) {
        int value = 0;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_hevc_tier, argsData.cachedtier.c_str()))) {
            pParams->hevc_tier = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), _T("level"), nullptr);
            return 1;
        }
    }

    // check if all mandatory parameters were set
    if (0 == _tcslen(pParams->strSrcFile)) {
        SET_ERR(strInput[0], _T("Source file name not found"), nullptr, nullptr);
        return 1;
    }

    if (0 == _tcslen(pParams->strDstFile)) {
        SET_ERR(strInput[0], _T("Destination file name not found"), nullptr, nullptr);
        return 1;
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

    if (pParams->pChapterFile && pParams->bCopyChapter) {
        SET_ERR(strInput[0], _T("--chapter and --chapter-copy are both set.\nThese could not be set at the same time."), nullptr, nullptr);
        return 1;
    }

    //set input buffer size
    if (argsData.nTmpInputBuf == 0) {
        argsData.nTmpInputBuf = QSV_DEFAULT_INPUT_BUF_HW;
    }
    pParams->nInputBufSize = (mfxU16)clamp(argsData.nTmpInputBuf, QSV_INPUT_BUF_MIN, QSV_INPUT_BUF_MAX);

    if (pParams->nRotationAngle != 0 && pParams->nRotationAngle != 180) {
        SET_ERR(strInput[0], _T("Angles other than 180 degrees are not supported."), nullptr, nullptr);
        return 1; // other than 180 are not supported
    }

    // not all options are supported if rotate plugin is enabled
    if (pParams->nRotationAngle == 180) {
        if (MFX_FOURCC_NV12 != pParams->ColorFormat) {
            SET_ERR(strInput[0], _T("Rotation plugin requires NV12 input. Please specify -nv12 option."), nullptr, nullptr);
            return 1;
        }
        pParams->nPicStruct = MFX_PICSTRUCT_PROGRESSIVE;
        pParams->nDstWidth = pParams->nWidth;
        pParams->nDstHeight = pParams->nHeight;
        pParams->memType = SYSTEM_MEMORY;
    }

    return 0;
}

#if defined(_WIN32) || defined(_WIN64)
int parse_cmd(sInputParams *pParams, const char *cmda, ParseCmdError& err, bool ignore_parse_err) {
    init_qsvp_prm(pParams);
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
    for (int i = 0; i < argc; i++) {
        argv_tstring.push_back(wstring_to_tstring(argvw[i]));
    }
    LocalFree(argvw);

    vector<TCHAR *> argv_tchar;
    for (int i = 0; i < argc; i++) {
        argv_tchar.push_back((TCHAR *)argv_tstring[i].data());
    }
    argv_tchar.push_back(_T(""));
    const TCHAR **strInput = (const TCHAR **)argv_tchar.data();
    return parse_cmd(pParams, strInput, argc, err, ignore_parse_err);
}
#endif


#pragma warning (push)
#pragma warning (disable: 4127)
tstring gen_cmd(const sInputParams *pParams, bool save_disabled_prm) {
    std::basic_stringstream<TCHAR> tmp;
    std::basic_stringstream<TCHAR> cmd;
    sInputParams encPrmDefault;
    init_qsvp_prm(&encPrmDefault);

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

#define OPT_LST(str, opt, list) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << (str) << _T(" ") << get_chr_from_value(list, (pParams->opt));
#define OPT_QP(str, force, qpi, qpp, qpb) { \
    if ((force) \
    || (pParams->qpi) != (encPrmDefault.qpi) \
    || (pParams->qpp) != (encPrmDefault.qpp) \
    || (pParams->qpb) != (encPrmDefault.qpb)) { \
        if ((pParams->qpi) == (pParams->qpp) && (pParams->qpi) == (pParams->qpb)) { \
            cmd << _T(" ") << (str) << _T(" ") << (int)(pParams->qpi); \
        } else { \
            cmd << _T(" ") << (str) << _T(" ") << (int)(pParams->qpi) << _T(":") << (int)(pParams->qpp) << _T(":") << (int)(pParams->qpb); \
        } \
    } \
}
#define OPT_BOOL(str_true, str_false, opt) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << ((pParams->opt) ? (str_true) : (str_false));
#define OPT_BOOL_VAL(str_true, str_false, opt, val) { \
    if ((pParams->opt) != (encPrmDefault.opt) || (save_disabled_prm && (pParams->val) != (encPrmDefault.val))) { \
        cmd << _T(" ") << ((pParams->opt) ? (str_true) : (str_false)) <<  _T(" ") << (pParams->val); \
    } \
}
#define OPT_CHAR(str, opt) if ((pParams->opt) && (pParams->opt[0] != 0)) cmd << _T(" ") << str << _T(" ") << (pParams->opt);
#define OPT_STR(str, opt) if (pParams->opt.length() > 0) cmd << _T(" ") << str << _T(" ") << (pParams->opt.c_str());
#define OPT_CHAR_PATH(str, opt) if ((pParams->opt) && (pParams->opt[0] != 0)) cmd << _T(" ") << str << _T(" \"") << (pParams->opt) << _T("\"");
#define OPT_STR_PATH(str, opt) if (pParams->opt.length() > 0) cmd << _T(" ") << str << _T(" \"") << (pParams->opt.c_str()) << _T("\"");

    cmd << _T(" -c ") << get_chr_from_value(list_codec, pParams->CodecId);
    OPT_CHAR_PATH(_T("-i"), strSrcFile);
    OPT_CHAR_PATH(_T("-o"), strDstFile);
    switch (pParams->nInputFmt) {
    case RGY_INPUT_FMT_RAW:    cmd << _T(" --raw"); break;
    case RGY_INPUT_FMT_Y4M:    cmd << _T(" --y4m"); break;
    case RGY_INPUT_FMT_AVI:    cmd << _T(" --avi"); break;
    case RGY_INPUT_FMT_AVS:    cmd << _T(" --avs"); break;
    case RGY_INPUT_FMT_VPY:    cmd << _T(" --vpy"); break;
    case RGY_INPUT_FMT_VPY_MT: cmd << _T(" --vpy-mt"); break;
    case RGY_INPUT_FMT_AVHW:   cmd << _T(" --avhw"); break;
    case RGY_INPUT_FMT_AVSW:   cmd << _T(" --avsw"); break;
    default: break;
    }
    if (save_disabled_prm || pParams->nPicStruct != RGY_PICSTRUCT_FRAME) {
        OPT_LST(_T("--interlace"), nPicStruct, list_interlaced);
    }
    if (cropEnabled(pParams->sInCrop)) {
        cmd << _T(" --crop ") << pParams->sInCrop.e.left << _T(",") << pParams->sInCrop.e.up
            << _T(",") << pParams->sInCrop.e.right << _T(",") << pParams->sInCrop.e.bottom;
    }
    if (pParams->nFPSRate * pParams->nFPSScale > 0) {
        cmd << _T(" --fps ") << pParams->nFPSRate << _T("/") << pParams->nFPSScale;
    }
    if (pParams->nWidth * pParams->nHeight > 0) {
        cmd << _T(" --input-res ") << pParams->nWidth << _T("x") << pParams->nHeight;
    }
    if (pParams->nDstWidth * pParams->nDstHeight > 0) {
        cmd << _T(" --output-res ") << pParams->nDstWidth << _T("x") << pParams->nDstHeight;
    }
    OPT_LST(_T("--quality"), nTargetUsage, list_quality_for_option);
    OPT_BOOL(_T("--fixed-func"), _T("--no-fixed-func"), bUseFixedFunc);
    OPT_NUM(_T("--async-depth"), nAsyncDepth);
    if (save_disabled_prm || ((pParams->memType) != (encPrmDefault.memType))) {
        switch (pParams->memType) {
#ifdef D3D_SURFACES_SUPPORT
        case SYSTEM_MEMORY: cmd << _T(" --disable-d3d"); break;
        case HW_MEMORY:   cmd << _T(" --d3d"); break;
        case D3D9_MEMORY: cmd << _T(" --d3d9"); break;
#if MFX_D3D11_SUPPORT
        case D3D11_MEMORY: cmd << _T(" --d3d11"); break;
#endif
#endif
#ifdef LIBVA_SUPPORT
        case SYSTEM_MEMORY: cmd << _T(" --disable-va"); break;
        case D3D11_MEMORY: cmd << _T(" --va"); break;
#endif
        default: break;
        }
    }
    if (save_disabled_prm || pParams->nEncMode == MFX_RATECONTROL_QVBR) {
        OPT_NUM(_T("--qvbr-q"), nQVBRQuality);
    }
    if (save_disabled_prm) {
        switch (pParams->nEncMode) {
        case MFX_RATECONTROL_CBR:
        case MFX_RATECONTROL_VBR:
        case MFX_RATECONTROL_AVBR:
        case MFX_RATECONTROL_QVBR:
        case MFX_RATECONTROL_LA:
        case MFX_RATECONTROL_LA_HRD:
        case MFX_RATECONTROL_VCM: {
            OPT_QP(_T("--cqp"), true, nQPI, nQPP, nQPB);
            cmd << _T(" --icq ") << pParams->nICQQuality;
        } break;
        case MFX_RATECONTROL_ICQ:
        case MFX_RATECONTROL_LA_ICQ: {
            OPT_QP(_T("--cqp"), true, nQPI, nQPP, nQPB);
            cmd << _T(" --vbr ") << pParams->nBitRate;
        } break;
        case MFX_RATECONTROL_CQP:
        default: {
            cmd << _T(" --icq ") << pParams->nICQQuality;
            cmd << _T(" --vbr ") << pParams->nBitRate;
        } break;
        }
    }
    switch (pParams->nEncMode) {
    case MFX_RATECONTROL_CBR: {
        cmd << _T(" --cbr ") << pParams->nBitRate;
    } break;
    case MFX_RATECONTROL_VBR: {
        cmd << _T(" --vbr ") << pParams->nBitRate;
    } break;
    case MFX_RATECONTROL_AVBR: {
        cmd << _T(" --avbr ") << pParams->nBitRate;
    } break;
    case MFX_RATECONTROL_QVBR: {
        cmd << _T(" --qvbr ") << pParams->nBitRate;
    } break;
    case MFX_RATECONTROL_LA: {
        cmd << _T(" --la ") << pParams->nBitRate;
    } break;
    case MFX_RATECONTROL_LA_HRD: {
        cmd << _T(" --la-hrd ") << pParams->nBitRate;
    } break;
    case MFX_RATECONTROL_VCM: {
        cmd << _T(" --vcm ") << pParams->nBitRate;
    } break;
    case MFX_RATECONTROL_ICQ: {
        cmd << _T(" --icq ") << pParams->nICQQuality;
    } break;
    case MFX_RATECONTROL_LA_ICQ: {
        cmd << _T(" --la-icq ") << pParams->nICQQuality;
    } break;
    case MFX_RATECONTROL_CQP:
    default: {
        OPT_QP(_T("--cqp"), true, nQPI, nQPP, nQPB);
    } break;
    }
    if (save_disabled_prm || pParams->nEncMode == MFX_RATECONTROL_AVBR) {
        OPT_NUM(_T("--avbr-unitsize"), nAVBRConvergence);
    }
    if (save_disabled_prm
        || pParams->nEncMode == MFX_RATECONTROL_LA
        || pParams->nEncMode == MFX_RATECONTROL_LA_HRD
        || pParams->nEncMode == MFX_RATECONTROL_LA_ICQ) {
        OPT_NUM(_T("--la-depth"), nLookaheadDepth);
        OPT_NUM(_T("--la-window-size"), nWinBRCSize);
        OPT_LST(_T("--la-quality"), nLookaheadDS, list_lookahead_ds);
    }
    if (save_disabled_prm || pParams->nEncMode != MFX_RATECONTROL_CQP) {
        OPT_NUM(_T("--max-bitrate"), nMaxBitrate);
    }
    OPT_NUM(_T("--vbv-bufsize"), VBVBufsize);
    OPT_BOOL(_T("--fallback-rc"), _T(""), nFallback);
    OPT_QP(_T("--qp-min"), save_disabled_prm, nQPMin[0], nQPMin[1], nQPMin[2]);
    OPT_QP(_T("--qp-max"), save_disabled_prm, nQPMax[0], nQPMax[1], nQPMax[2]);
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
    OPT_NUM(_T("-b"), nBframes);
    OPT_BOOL(_T("--b-pyramid"), _T("--no-b-pyramid"), bBPyramid);
    OPT_BOOL(_T("--open-gop"), _T("--no-open-gop"), bopenGOP);
    OPT_BOOL(_T("--strict-gop"), _T(""), bforceGOPSettings);
    OPT_BOOL(_T("--i-adapt"), _T("--no-i-adapt"), bAdaptiveI);
    OPT_BOOL(_T("--b-adapt"), _T("--no-b-adapt"), bAdaptiveB);
    OPT_TRI(_T("--weightb"), _T("--no-weightb"), nWeightB, MFX_WEIGHTED_PRED_DEFAULT, MFX_WEIGHTED_PRED_UNKNOWN);
    OPT_TRI(_T("--weightp"), _T("--no-weightp"), nWeightP, MFX_WEIGHTED_PRED_DEFAULT, MFX_WEIGHTED_PRED_UNKNOWN);
    OPT_TRI(_T("--repartition-check"), _T("--no-repartition-check"), nRepartitionCheck, MFX_CODINGOPTION_ON, MFX_CODINGOPTION_OFF);
    OPT_TRI(_T("--fade-detect"), _T("--no-fade-detect"), nFadeDetect, MFX_CODINGOPTION_ON, MFX_CODINGOPTION_OFF);
    if (pParams->nGOPLength == 0 && pParams->nGOPLength != encPrmDefault.nGOPLength) {
        cmd << _T(" --gop-len auto");
    } else {
        OPT_NUM(_T("--gop-len"), nGOPLength);
    }
    OPT_LST(_T("--mv-precision"), nMVPrecision, list_mv_presicion);
    OPT_NUM(_T("--mv-search"), MVSearchWindow.x);
    if (pParams->bGlobalMotionAdjust) {
        cmd << _T(" --mv-scaling ") << pParams->nMVCostScaling;
    }
    if (pParams->nPAR[0] > 0 && pParams->nPAR[1] > 0) {
        cmd << _T(" --sar ") << pParams->nPAR[0] << _T(":") << pParams->nPAR[1];
    } else if (pParams->nPAR[0] < 0 && pParams->nPAR[1] < 0) {
        cmd << _T(" --dar ") << -1 * pParams->nPAR[0] << _T(":") << -1 * pParams->nPAR[1];
    }

    //OPT_BOOL(_T("--extbrc"), _T("--no-extbrc"), bExtBRC);
    OPT_BOOL(_T("--mbbrc"), _T("--no-mbbrc"), bMBBRC);
    OPT_BOOL(_T("--adapt-ltr"), _T("--no-adapt-ltr"), extBrcAdaptiveLTR);
    OPT_BOOL(_T("--intra-refresh"), _T("--no-intra-refresh"), bIntraRefresh);
    OPT_BOOL(_T("--direct-bias-adjust"), _T("--no-direct-bias-adjust"), bDirectBiasAdjust);
    OPT_LST(_T("--intra-pred"), nIntraPred, list_pred_block_size);
    OPT_LST(_T("--inter-pred"), nInterPred, list_pred_block_size);
    OPT_BOOL(_T("--aud"), _T(""), bOutputAud);
    OPT_BOOL(_T("--pic-struct"), _T(""), bOutputPicStruct);
    OPT_BOOL(_T("--fullrange"), _T(""), bFullrange);
    OPT_LST(_T("--videoformat"), VideoFormat, list_videoformat);
    OPT_LST(_T("--colormatrix"), ColorMatrix, list_colormatrix);
    OPT_LST(_T("--colorprim"), ColorPrim, list_colorprim);
    OPT_LST(_T("--transfer"), Transfer, list_transfer);
    OPT_LST(_T("--chromaloc"), chromaloc, list_chromaloc);
    OPT_LST(_T("--level"), CodecLevel, get_level_list(pParams->CodecId));
    OPT_LST(_T("--profile"), CodecProfile, get_profile_list(pParams->CodecId));
    if (save_disabled_prm || pParams->CodecId == MFX_CODEC_HEVC) {
        OPT_CHAR(_T("--max-cll"), sMaxCll);
        OPT_CHAR(_T("--master-display"), sMasterDisplay);
        OPT_LST(_T("--ctu"), hevc_ctu, list_hevc_ctu);
        OPT_LST(_T("--sao"), hevc_sao, list_hevc_sao);
        OPT_BOOL(_T("--tskip"), _T("--no-tskip"), bFullrange);
    }
    if (save_disabled_prm || pParams->CodecId == MFX_CODEC_AVC) {
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
    if (save_disabled_prm || pParams->CodecId == MFX_CODEC_VP8) {
        OPT_NUM(_T("--sharpness"), nVP8Sharpness);
    }
#if ENABLE_SESSION_THREAD_CONFIG
    OPT_NUM(_T("--session-threads"), nSessionThreads);
    OPT_LST(_T("--session-thread-priority"), nSessionThreadPriority, list_priority);
#endif //#if ENABLE_SESSION_THREAD_CONFIG
#if ENABLE_AVSW_READER
    OPT_NUM(_T("--input-analyze"), nAVDemuxAnalyzeSec);
    if (pParams->nTrimCount > 0) {
        cmd << _T(" --trim ");
        for (int i = 0; i < pParams->nTrimCount; i++) {
            if (i > 0) cmd << _T(",");
            cmd << pParams->pTrimList[i].start << _T(":") << pParams->pTrimList[i].fin;
        }
    }
    OPT_FLOAT(_T("--seek"), fSeekSec, 2);
    OPT_CHAR(_T("--input-format"), pAVInputFormat);
    OPT_CHAR(_T("--output-format"), pAVMuxOutputFormat);
    OPT_NUM(_T("--video-track"), nVideoTrack);
    OPT_NUM(_T("--video-streamid"), nVideoStreamId);
    OPT_CHAR(_T("--video-tag"), videoCodecTag);
    if (pParams->pMuxOpt) {
        for (uint32_t i = 0; i < pParams->pMuxOpt->size(); i++) {
            cmd << _T(" -m ") << pParams->pMuxOpt->at(i).first << _T(":") << pParams->pMuxOpt->at(i).second;
        }
    }
    tmp.str(tstring());
    for (uint32_t i = 0; i < pParams->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = pParams->ppAudioSelectList[i];
        if (pAudioSelect->encCodec == RGY_AVCODEC_COPY) {
            if (pAudioSelect->trackID == 0) {
                tmp << _T(","); // --audio-copy のみの指定 (トラックIDを省略)
            } else {
                tmp << _T(",") << pAudioSelect->trackID;
            }
        }
    }
    if (!tmp.str().empty()) {
        cmd << _T(" --audio-copy ") << tmp.str().substr(1);
    }
    tmp.str(tstring());

    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = pParams->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY) {
            cmd << _T(" --audio-codec ") << pAudioSelect->trackID;
            if (pAudioSelect->encCodec != RGY_AVCODEC_AUTO) {
                cmd << _T("?") << pAudioSelect->encCodec;
            }
            if (pAudioSelect->encCodecPrm.length() > 0) {
                cmd << _T(":") << pAudioSelect->encCodecPrm;
            }
        }
    }

    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = pParams->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->encCodecProfile.length() > 0) {
            cmd << _T(" --audio-profile ") << pAudioSelect->trackID << _T("?") << pAudioSelect->encCodecProfile;
        }
    }

    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = pParams->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->encBitrate > 0) {
            cmd << _T(" --audio-bitrate ") << pAudioSelect->trackID << _T("?") << pAudioSelect->encBitrate;
        }
    }
#if !FOR_AUO //QSVEnc.auoでは、libavutilの関数 av_get_channel_layout_string()を実行してはならない
    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        tmp.str(tstring());
        const AudioSelect *pAudioSelect = pParams->ppAudioSelectList[i];
        for (int j = 0; j < MAX_SPLIT_CHANNELS; j++) {
            if (pAudioSelect->pnStreamChannelSelect[j] == 0) {
                break;
            }
            if (j > 0) cmd << _T(",");
            if (pAudioSelect->pnStreamChannelSelect[j] != RGY_CHANNEL_AUTO) {
                char buf[256];
                av_get_channel_layout_string(buf, _countof(buf), 0, pAudioSelect->pnStreamChannelOut[j]);
                cmd << char_to_tstring(buf);
            }
            if (pAudioSelect->pnStreamChannelOut[j] != RGY_CHANNEL_AUTO) {
                cmd << _T(":");
                char buf[256];
                av_get_channel_layout_string(buf, _countof(buf), 0, pAudioSelect->pnStreamChannelOut[j]);
                cmd << char_to_tstring(buf);
            }
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --audio-stream ") << pAudioSelect->trackID << _T("?") << tmp.str();
        }
    }
#endif //#if FOR_AUO
    tmp.str(tstring());

    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = pParams->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->encSamplingRate > 0) {
            cmd << _T(" --audio-samplerate ") << pAudioSelect->trackID << _T("?") << pAudioSelect->encSamplingRate;
        }
    }
    OPT_LST(_T("--audio-resampler"), nAudioResampler, list_resampler);

    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = pParams->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->filter.length() > 0) {
            cmd << _T(" --audio-filter ") << pAudioSelect->trackID << _T("?") << pAudioSelect->filter;
        }
    }
    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = pParams->ppAudioSelectList[i];
        if (pAudioSelect->extractFilename.length() > 0) {
            cmd << _T(" --audio-file ") << pAudioSelect->trackID << _T("?");
            if (pAudioSelect->extractFormat.length() > 0) {
                cmd << pAudioSelect->extractFormat << _T(":");
            }
            cmd << _T("\"") << pAudioSelect->extractFilename << _T("\"");
        }
    }
    for (int i = 0; i < pParams->nAudioSourceCount; i++) {
        cmd << _T(" --audio-source ") << _T("\"") << pParams->ppAudioSourceList[i] << _T("\"");
    }
    OPT_NUM(_T("--audio-ignore-decode-error"), nAudioIgnoreDecodeError);
    if (pParams->pMuxOpt) {
        for (uint32_t i = 0; i < pParams->pMuxOpt->size(); i++) {
            cmd << _T(" -m ") << (*pParams->pMuxOpt)[i].first << _T(":") << (*pParams->pMuxOpt)[i].second;
        }
    }

    tmp.str(tstring());
    for (int i = 0; i < pParams->nSubtitleSelectCount; i++) {
        tmp << _T(",") << pParams->ppSubtitleSelectList[i]->trackID;
        if (pParams->ppSubtitleSelectList[i]->asdata) {
            tmp << _T("?asdata");
        }
    }
    if (!tmp.str().empty()) {
        cmd << _T(" --sub-copy ") << tmp.str().substr(1);
    }
    tmp.str(tstring());
    OPT_LST(_T("--caption2ass"), caption2ass, list_caption2ass);

    tmp.str(tstring());
    for (int i = 0; i < pParams->nDataSelectCount; i++) {
        tmp << _T(",") << pParams->ppDataSelectList[i]->trackID;
    }
    if (!tmp.str().empty()) {
        cmd << _T(" --data-copy ") << tmp.str().substr(1);
    }
    tmp.str(tstring());

    OPT_CHAR_PATH(_T("--chapter"), pChapterFile);
    OPT_BOOL(_T("--chapter-copy"), _T(""), bCopyChapter);
    OPT_BOOL(_T("--chapter-no-trim"), _T(""), bChapterNoTrim);
    OPT_LST(_T("--avsync"), nAVSyncMode, list_avsync);
#endif //#if ENABLE_AVSW_READER

    OPT_LST(_T("--vpp-deinterlace"), vpp.deinterlace, list_deinterlace);
    OPT_BOOL_VAL(_T("--vpp-detail-enhance"), _T("--no-vpp-detail-enhance"), vpp.detail.enable, vpp.detail.strength);
    OPT_BOOL_VAL(_T("--vpp-denoise"), _T("--no-vpp-denoise"), vpp.denoise.enable, vpp.denoise.strength);
    if (pParams->vpp.mctf.enable && pParams->vpp.mctf.strength == 0) {
        cmd << _T(" --vpp-mctf auto");
    } else {
        OPT_BOOL_VAL(_T("--vpp-mctf"), _T("--no-vpp-mctf"), vpp.mctf.enable, vpp.mctf.strength);
    }
    OPT_BOOL(_T("--vpp-half-turn"), _T(""), vpp.halfTurn);
    OPT_LST(_T("--vpp-rotate"), vpp.rotate, list_vpp_rotate_angle);
    OPT_LST(_T("--vpp-mirror"), vpp.mirrorType, list_vpp_mirroring);
    OPT_LST(_T("--vpp-scaling"), vpp.scalingQuality, list_vpp_scaling_quality);
    OPT_LST(_T("--vpp-fps-conv"), vpp.fpsConversion, list_vpp_fps_conversion);
    OPT_LST(_T("--vpp-image-stab"), vpp.imageStabilizer, list_vpp_image_stabilizer);
#if ENABLE_CUSTOM_VPP
#if ENABLE_AVSW_READER && ENABLE_LIBASS_SUBBURN
    OPT_CHAR_PATH(_T("--vpp-sub"), vpp.subburn.pFilePath);
    OPT_CHAR_PATH(_T("--vpp-sub-charset"), vpp.subburn.pCharEnc);
    OPT_LST(_T("--vpp-sub-shaping"), vpp.subburn.nShaping, list_vpp_sub_shaping);
#endif //#if ENABLE_AVSW_READER && ENABLE_LIBASS_SUBBURN
    OPT_CHAR_PATH(_T("--vpp-delogo"), vpp.delogo.pFilePath);
    OPT_CHAR(_T("--vpp-delogo-select"), vpp.delogo.pSelect);
    OPT_NUM(_T("--vpp-delogo-depth"), vpp.delogo.depth);
    if (pParams->vpp.delogo.posOffset.first > 0 || pParams->vpp.delogo.posOffset.second > 0) {
        cmd << _T(" --vpp-delogo-pos ") << pParams->vpp.delogo.posOffset.first << _T("x") << pParams->vpp.delogo.posOffset.second;
    }
    OPT_NUM(_T("--vpp-delogo-y"), vpp.delogo.YOffset);
    OPT_NUM(_T("--vpp-delogo-cb"), vpp.delogo.CbOffset);
    OPT_NUM(_T("--vpp-delogo-cr"), vpp.delogo.CrOffset);
#endif //#if ENABLE_CUSTOM_VPP
#if defined(_WIN32) || defined(_WIN64)
    OPT_NUM(_T("--mfx-thread"), nSessionThreads);
#endif //#if defined(_WIN32) || defined(_WIN64)
    OPT_NUM(_T("--input-buf"), nInputBufSize);
    OPT_NUM(_T("--output-buf"), nOutputBufSizeMB);
    OPT_NUM(_T("--output-thread"), nOutputThread);
    OPT_NUM(_T("--input-thread"), nInputThread);
    OPT_NUM(_T("--audio-thread"), nAudioThread);
    OPT_NUM(_T("--thread-csp"), threadCsp);
    OPT_LST(_T("--simd-csp"), simdCsp, list_simd);
    OPT_NUM(_T("--max-procfps"), nProcSpeedLimit);
    OPT_CHAR_PATH(_T("--log"), pStrLogFile);
    OPT_LST(_T("--log-level"), nLogLevel, list_log_level);
    OPT_CHAR_PATH(_T("--log-framelist"), pFramePosListLog);
    OPT_CHAR_PATH(_T("--log-mux-ts"), pMuxVidTsLogFile);
    OPT_CHAR_PATH(_T("--log-copy-framedata"), pLogCopyFrameData);
    if (pParams->nPerfMonitorSelect != encPrmDefault.nPerfMonitorSelect) {
        auto select = (int)pParams->nPerfMonitorSelect;
        tmp.str(tstring());
        for (int i = 0; list_pref_monitor[i].desc; i++) {
            auto check = list_pref_monitor[i].value;
            if ((select & check) == check) {
                tmp << _T(",") << list_pref_monitor[i].desc;
                select &= (~check);
            }
        }
        if (tmp.str().empty()) {
            cmd << _T(" --perf-monitor");
        } else {
            cmd << _T(" --perf-monitor ") << tmp.str().substr(1);
        }
    }
    if (pParams->nPerfMonitorSelectMatplot != encPrmDefault.nPerfMonitorSelectMatplot) {
        auto select = (int)pParams->nPerfMonitorSelectMatplot;
        tmp.str(tstring());
        for (int i = 0; list_pref_monitor[i].desc; i++) {
            auto check = list_pref_monitor[i].value;
            if ((select & check) == check) {
                tmp << _T(",") << list_pref_monitor[i].desc;
                select &= (~check);
            }
        }
        if (tmp.str().empty()) {
            cmd << _T(" --perf-monitor-plot");
        } else {
            cmd << _T(" --perf-monitor-plot ") << tmp.str().substr(1);
        }
    }
    OPT_NUM(_T("--perf-monitor-interval"), nPerfMonitorInterval);
    OPT_CHAR_PATH(_T("--python"), pLogCopyFrameData);
    OPT_BOOL(_T("--timer-period-tuning"), _T("--no-timer-period-tuning"), bDisableTimerPeriodTuning);
    return cmd.str();
}
#pragma warning (pop)
