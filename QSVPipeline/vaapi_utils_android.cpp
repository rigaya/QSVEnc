/* ****************************************************************************** *\

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2013 Intel Corporation. All Rights Reserved.

\* ****************************************************************************** */

#ifdef LIBVA_ANDROID_SUPPORT
#ifdef ANDROID

#include "vaapi_utils_android.h"

CLibVA* CreateLibVA(void)
{
    return new AndroidLibVA;
}

/*------------------------------------------------------------------------------*/

typedef unsigned int vaapiAndroidDisplay;

#define VAAPI_ANDROID_DEFAULT_DISPLAY 0x18c34078

AndroidLibVA::AndroidLibVA(void):
    m_display(NULL)
{
    VAStatus va_res = VA_STATUS_SUCCESS;
    mfxStatus sts = MFX_ERR_NONE;
    int major_version = 0, minor_version = 0;
    vaapiAndroidDisplay* display = NULL;

    m_display = display = (vaapiAndroidDisplay*)malloc(sizeof(vaapiAndroidDisplay));
    if (NULL == m_display) sts = MFX_ERR_NOT_INITIALIZED;
    else *display = VAAPI_ANDROID_DEFAULT_DISPLAY;

    if (MFX_ERR_NONE == sts)
    {
        m_va_dpy = vaGetDisplay(m_display);
        if (!m_va_dpy)
        {
            free(m_display);
            sts = MFX_ERR_NULL_PTR;
        }
    }
    if (MFX_ERR_NONE == sts)
    {
        va_res = vaInitialize(m_va_dpy, &major_version, &minor_version);
        sts = va_to_mfx_status(va_res);
        if (MFX_ERR_NONE != sts)
        {
            free(display);
            m_display = NULL;
        }
    }
    if (MFX_ERR_NONE != sts) throw std::bad_alloc();
}

AndroidLibVA::~AndroidLibVA(void)
{
    if (m_va_dpy)
    {
        vaTerminate(m_va_dpy);
    }
    if (m_display)
    {
        free(m_display);
    }
}

#endif // #ifdef ANDROID
#endif // #ifdef LIBVA_ANDROID_SUPPORT
