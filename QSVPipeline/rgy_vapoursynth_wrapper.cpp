// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// ------------------------------------------------------------------------------------------

#include "rgy_vapoursynth_wrapper.h"
#include "rgy_log.h"

#if ENABLE_VAPOURSYNTH_READER

// Implemented in separate translation units to avoid header name collisions.
std::unique_ptr<RGYVapourSynthWrapper> CreateVapourSynthWrapperV4(const tstring& vsdir, RGYLog *log);
std::unique_ptr<RGYVapourSynthWrapper> CreateVapourSynthWrapperV3(const tstring& vsdir, RGYLog *log);

std::unique_ptr<RGYVapourSynthWrapper> CreateVapourSynthWrapper(const tstring& vsdir, RGYLog *log) {
    if (auto v4 = CreateVapourSynthWrapperV4(vsdir, log)) {
        return v4;
    }
    if (auto v3 = CreateVapourSynthWrapperV3(vsdir, log)) {
        return v3;
    }
    if (log) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_IN, _T("vpy: failed to create VapourSynth wrapper.\n"));
    }
    return nullptr;
}

#else

std::unique_ptr<RGYVapourSynthWrapper> CreateVapourSynthWrapper(const tstring&, RGYLog*) {
    return nullptr;
}

#endif


