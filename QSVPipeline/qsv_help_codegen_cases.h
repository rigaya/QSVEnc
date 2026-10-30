#ifndef __RGY_HELP_CODEGEN_CASES_H__
#define __RGY_HELP_CODEGEN_CASES_H__

#include "qsv_cmd.h"
#include "rgy_cmd_selftest.h"

// help整形とは独立に、今回見つかったパース精度と生成漏れの回帰を確認する。
static inline int rgy_help_codegen_cases() {
    RGYCmdSelfTest<sInputParams> checker(
        [](sInputParams *prm, const std::vector<tstring>& args) {
            auto argv = rgy_cmd_selftest_argv(args);
            return parse_cmd(prm, argv.data(), (int)argv.size() - 1);
        },
        [](const sInputParams *prm) { return gen_cmd(prm, false); });
    const auto roundTrips = checker.runRoundTripCases({
        {
            _T("colorspace default desat_base keeps double precision"),
            { _T("--vpp-colorspace"), _T("desat_base=0.18") },
            false,
            { _T("--vpp-colorspace") },
            { _T("desat_base=") }
        },
        {
            _T("deband preserves a non-default seed"),
            { _T("--vpp-deband"), _T("seed=9876") },
            false,
            { _T("--vpp-deband"), _T("seed=9876") },
            {}
        },
        {
            _T("deband saves a disabled filter seed"),
            { _T("--vpp-deband"), _T("enable=false,seed=9876") },
            true,
            { _T("--vpp-deband"), _T("enable=false"), _T("seed=9876") },
            {}
        },
        {
            _T("deband omits disabled seed without save flag"),
            { _T("--vpp-deband"), _T("enable=false,seed=9876") },
            false,
            {},
            { _T("--vpp-deband"), _T("seed=9876") }
        },
        {
            _T("AI frame interpolation preserves bare activation"),
            { _T("--vpp-ai-frameinterp") },
            false,
            { _T("--vpp-ai-frameinterp") },
            {}
        },
        {
            _T("AI frame interpolation preserves explicit default SCD"),
            { _T("--vpp-ai-frameinterp"), _T("scd=false") },
            false,
            { _T("--vpp-ai-frameinterp") },
            { _T("scd=true") }
        },
        {
            _T("AI frame interpolation preserves enabled SCD"),
            { _T("--vpp-ai-frameinterp"), _T("scd=true") },
            false,
            { _T("--vpp-ai-frameinterp"), _T("scd=true") },
            {}
        },
        {
            _T("AI frame interpolation saves disabled parameters"),
            { _T("--vpp-ai-frameinterp"), _T("enable=false,scd=true") },
            true,
            { _T("--vpp-ai-frameinterp"), _T("enable=false"), _T("scd=true") },
            {}
        },
        {
            _T("AI frame interpolation omits disabled parameters without save flag"),
            { _T("--vpp-ai-frameinterp"), _T("enable=false,scd=true") },
            false,
            {},
            { _T("--vpp-ai-frameinterp") }
        }
    }, [](const sInputParams *prm, bool saveDisabledPrm) { return gen_cmd(prm, saveDisabledPrm); });

    int stateTotal = 0, stateNg = 0;
    const auto check = [&](bool ok, const TCHAR *name) {
        ++stateTotal;
        if (!ok) ++stateNg;
        _ftprintf(stdout, _T("[CODEGEN-STATE %s] %s\n"), ok ? _T("OK") : _T("NG"), name);
    };
    {
        sInputParams prm;
        const std::vector<tstring> args = {
            _T("--vpp-colorspace"), _T("desat_base=0.18"),
            _T("-i"), _T("rgy_codegen_input.yuv"), _T("-o"), _T("rgy_codegen_output.raw")
        };
        auto argv = rgy_cmd_selftest_argv(args);
        auto& error = rgy_cmd_error_info();
        const bool quiet = error.quiet;
        error.quiet = true;
        error.reset();
        const bool parsed = parse_cmd(&prm, argv.data(), (int)argv.size() - 1) == 0 && !error.errorOccurred;
        error.quiet = quiet;
        check(parsed && prm.vpp.colorspace.hdr2sdr.desat_base == FILTER_DEFAULT_HDR2SDR_DESAT_BASE,
            _T("parsed desat_base equals the double default exactly"));
    }
    {
        sInputParams prm;
        prm.vpp.overlay.emplace_back();
        prm.vpp.overlay.back().enable = true;
        // 引数なしoverlayのパーサは状態を追加しないため、生成分岐の独立性は状態を直接作って確認する。
        // ここではエンコードを行わず、debandの状態がoverlayの出力を左右しないことだけを試す。
        prm.vpp.deband.enable = false;
        check(gen_cmd(&prm, false).find(_T("--vpp-overlay")) != tstring::npos,
            _T("enabled default overlay is emitted with deband disabled"));
        prm.vpp.deband.enable = true;
        check(gen_cmd(&prm, false).find(_T("--vpp-overlay")) != tstring::npos,
            _T("enabled default overlay is emitted with deband enabled"));
        prm.vpp.overlay.back().enable = false;
        prm.vpp.overlay.back().alpha = 0.5f;
        check(gen_cmd(&prm, false).find(_T("--vpp-overlay")) == tstring::npos,
            _T("disabled overlay is not emitted by enabled deband"));
        const auto saved = gen_cmd(&prm, true);
        check(saved.find(_T("--vpp-overlay")) != tstring::npos
            && saved.find(_T("enable=false")) != tstring::npos
            && saved.find(_T("alpha=")) != tstring::npos,
            _T("disabled overlay parameters are kept with save flag"));
    }
    _ftprintf(stdout, _T("codegen state total %d, NG %d\n"), stateTotal, stateNg);
    return roundTrips > 0 && stateNg == 0 ? 1 : -1;
}

#endif
