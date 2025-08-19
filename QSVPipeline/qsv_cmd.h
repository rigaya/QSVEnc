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

#ifndef __QSV_CMD_H__
#define __QSV_CMD_H__

#include "rgy_tchar.h"
#include "rgy_cmd.h"
#include "qsv_prm.h"

const TCHAR *cmd_short_opt_to_long(TCHAR short_opt);
tstring GetQSVEncVersion();

int parse_cmd(sInputParams *pParams, const TCHAR *strInput[], int nArgNum, bool ignore_parse_err = false);
#if defined(_WIN32) || defined(_WIN64)
int parse_cmd(sInputParams *pParams, const wchar_t *cmdw, bool ignore_parse_err = false);
int parse_cmd(sInputParams *pParams, const char *cmda, bool ignore_parse_err = false);
#endif

tstring gen_cmd(const sInputParams *pParams, bool save_disabled_prm);

tstring gen_cmd_help_vppmfx();

#endif //__QSV_CMD_H__
