/* ****************************************************************************** *\

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011-2013 Intel Corporation. All Rights Reserved.

\* ****************************************************************************** */

#if defined(LIBVA_X11_SUPPORT)

#include "sample_defs.h"
#include "vaapi_utils_x11.h"


#define VAAPI_X_DEFAULT_DISPLAY ":0.0"

X11LibVA::X11LibVA(void)
{
    VAStatus va_res = VA_STATUS_SUCCESS;
    mfxStatus sts = MFX_ERR_NONE;
    int major_version = 0, minor_version = 0;
    char* currentDisplay = getenv("DISPLAY");

    if (currentDisplay) m_display = XOpenDisplay(currentDisplay);
    else m_display = XOpenDisplay(VAAPI_X_DEFAULT_DISPLAY);

    if (NULL == m_display) sts = MFX_ERR_NOT_INITIALIZED;
    if (MFX_ERR_NONE == sts)
    {
        m_va_dpy = vaGetDisplay(m_display);
        if (!m_va_dpy)
        {
            XCloseDisplay(m_display);
            sts = MFX_ERR_NULL_PTR;
        }
    }
    if (MFX_ERR_NONE == sts)
    {
        va_res = vaInitialize(m_va_dpy, &major_version, &minor_version);
        sts = va_to_mfx_status(va_res);
        if (MFX_ERR_NONE != sts)
        {
            XCloseDisplay(m_display);
        }
    }
    if (MFX_ERR_NONE != sts) throw std::bad_alloc();
}

X11LibVA::~X11LibVA(void)
{
    if (m_va_dpy)
    {
        vaTerminate(m_va_dpy);
    }
    if (m_display)
    {
        XCloseDisplay(m_display);
    }
}

#endif // #if defined(LIBVA_X11_SUPPORT)
