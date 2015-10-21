/* ****************************************************************************** *\

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2012-2013 Intel Corporation. All Rights Reserved.

\* ****************************************************************************** */

#ifndef __VAAPI_UTILS_X11_H__
#define __VAAPI_UTILS_X11_H__

#if defined(LIBVA_X11_SUPPORT)

#include <va/va_x11.h>
#include "vaapi_utils.h"


class X11LibVA : public CLibVA
{
public:
    X11LibVA(void);
    virtual ~X11LibVA(void);

    void *GetXDisplay(void) { return m_display;}

protected:
    Display* m_display;

private:
    DISALLOW_COPY_AND_ASSIGN(X11LibVA);
};

#endif // #if defined(LIBVA_X11_SUPPORT)

#endif // #ifndef __VAAPI_UTILS_X11_H__
