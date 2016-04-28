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

#include "qsv_hw_va.h"

#if defined(LIBVA_DRM_SUPPORT) || defined(LIBVA_X11_SUPPORT)
#include <va/va.h>

mfxStatus va_to_mfx_status(VAStatus va_res) {
    switch (va_res) {
    case VA_STATUS_SUCCESS:
        return MFX_ERR_NONE;
    case VA_STATUS_ERROR_ALLOCATION_FAILED:
        return MFX_ERR_MEMORY_ALLOC;
    case VA_STATUS_ERROR_ATTR_NOT_SUPPORTED:
    case VA_STATUS_ERROR_UNSUPPORTED_PROFILE:
    case VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT:
    case VA_STATUS_ERROR_UNSUPPORTED_RT_FORMAT:
    case VA_STATUS_ERROR_UNSUPPORTED_BUFFERTYPE:
    case VA_STATUS_ERROR_FLAG_NOT_SUPPORTED:
    case VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED:
        return MFX_ERR_UNSUPPORTED;
    case VA_STATUS_ERROR_INVALID_DISPLAY:
    case VA_STATUS_ERROR_INVALID_CONFIG:
    case VA_STATUS_ERROR_INVALID_CONTEXT:
    case VA_STATUS_ERROR_INVALID_SURFACE:
    case VA_STATUS_ERROR_INVALID_BUFFER:
    case VA_STATUS_ERROR_INVALID_IMAGE:
    case VA_STATUS_ERROR_INVALID_SUBPICTURE:
        return MFX_ERR_NOT_INITIALIZED;
    case VA_STATUS_ERROR_INVALID_PARAMETER:
        return MFX_ERR_INVALID_VIDEO_PARAM;
    default:
        return MFX_ERR_UNKNOWN;
    }
}
CLibVA *CreateLibVA() {
#if defined(LIBVA_DRM_SUPPORT)
    return new DRMLibVA();
#elif defined(LIBVA_X11_SUPPORT)
    return new X11LibVA();
#endif
    return nullptr;
}

CQSVHWDevice *CreateVAAPIDevice() {
#if defined(LIBVA_DRM_SUPPORT)
    return new CQSVHWVADeviceDRM();
#elif defined(LIBVA_X11_SUPPORT)
    return new CQSVHWVADeviceX11();
#else
    return nullptr;
#endif
}
#endif // #if defined(LIBVA_DRM_SUPPORT) || defined(LIBVA_X11_SUPPORT)




#if defined(LIBVA_DRM_SUPPORT)

#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <dirent.h>
#include <stdexcept>

#define MFX_PCI_DIR "/sys/bus/pci/devices"
#define MFX_DRI_DIR "/dev/dri/"
#define MFX_PCI_DISPLAY_CONTROLLER_CLASS 0x03

struct mfx_disp_adapters {
    mfxU32 vendor_id;
    mfxU32 device_id;
};

static int mfx_dir_filter(const struct dirent* dir_ent) {
    if (!dir_ent) return 0;
    if (!strcmp(dir_ent->d_name, ".")) return 0;
    if (!strcmp(dir_ent->d_name, "..")) return 0;
    return 1;
}

typedef int (*fsort)(const struct dirent**, const struct dirent**);

static mfxU32 mfx_init_adapters(struct mfx_disp_adapters **p_adapters) {
    mfxU32 adapters_num = 0;
    struct mfx_disp_adapters* adapters = NULL;
    struct dirent** dir_entries = NULL;
    int entries_num = scandir(MFX_PCI_DIR, &dir_entries, mfx_dir_filter, (fsort)alphasort);

    for (int i = 0; i < entries_num; i++) {
        long int class_id = 0, vendor_id = 0, device_id = 0;

        if (dir_entries[i]) {
            char file_name[300] = {0};
            snprintf(file_name, _countof(file_name), "%s/%s/%s", MFX_PCI_DIR, dir_entries[i]->d_name, "class");
            FILE *file = fopen(file_name, "r");
            if (file) {
                char str[16] = {0};
                if (fgets(str, sizeof(str), file)) {
                    class_id = strtol(str, NULL, 16);
                }
                fclose(file);

                if (MFX_PCI_DISPLAY_CONTROLLER_CLASS == (class_id >> 16)) {
                    // obtaining device vendor id
                    snprintf(file_name, _countof(file_name), "%s/%s/%s", MFX_PCI_DIR, dir_entries[i]->d_name, "vendor");
                    file = fopen(file_name, "r");
                    if (file) {
                        if (fgets(str, sizeof(str), file)) {
                            vendor_id = strtol(str, NULL, 16);
                        }
                        fclose(file);
                    }
                    // obtaining device id
                    snprintf(file_name, _countof(file_name), "%s/%s/%s", MFX_PCI_DIR, dir_entries[i]->d_name, "device");
                    file = fopen(file_name, "r");
                    if (file) {
                        if (fgets(str, sizeof(str), file)) {
                            device_id = strtol(str, NULL, 16);
                        }
                        fclose(file);
                    }
                    // adding valid adapter to the list
                    if (vendor_id && device_id) {
                        struct mfx_disp_adapters* tmp_adapters =
                            (mfx_disp_adapters*)realloc(adapters, (adapters_num+1)*sizeof(struct mfx_disp_adapters));
                        if (tmp_adapters) {
                            adapters = tmp_adapters;
                            adapters[adapters_num].vendor_id = vendor_id;
                            adapters[adapters_num].device_id = device_id;
                            adapters_num++;
                        }
                    }
                }
            }
            free(dir_entries[i]);
        }
    }
    if (entries_num) free(dir_entries);
    if (p_adapters) *p_adapters = adapters;

    return adapters_num;
}

DRMLibVA::DRMLibVA(void) : m_fd(-1) {
    const mfxU32 IntelVendorID = 0x8086;
    //the first Intel adapter is only required now, the second - in the future
    const mfxU32 numberOfRequiredIntelAdapter = 1;
    const char nodesNames[][8] = {"renderD", "card"};

    VAStatus va_res = VA_STATUS_SUCCESS;
    mfxStatus sts = MFX_ERR_NONE;
    int major_version = 0, minor_version = 0;

    mfx_disp_adapters* adapters = NULL;
    int adapters_num = mfx_init_adapters(&adapters);

    // Search for the required display adapter
    int i = 0, nFoundAdapters = 0;
    int nodesNumbers[] = {0,0};
    while ((i < adapters_num) && (nFoundAdapters != numberOfRequiredIntelAdapter)) {
        if (adapters[i].vendor_id == IntelVendorID) {
            nFoundAdapters++;
            nodesNumbers[0] = i+128; //for render nodes
            nodesNumbers[1] = i;     //for card
        }
        i++;
    }
    if (adapters_num) free(adapters);
    // If Intel adapter with specified number wasn't found, throws exception
    if (nFoundAdapters != numberOfRequiredIntelAdapter)
        throw std::range_error("The Intel adapter with a specified number wasn't found");

    // Initialization of paths to the device nodes
    char** adapterPaths = new char* [2];
    for (int i=0; i<2; i++) {
        adapterPaths[i] = new char[sizeof(MFX_DRI_DIR) + sizeof(nodesNames[i]) + 3];
        sprintf(adapterPaths[i], "%s%s%d", MFX_DRI_DIR, nodesNames[i], nodesNumbers[i]);
    }

    // Loading display. At first trying to open render nodes, then card.
    for (int i=0; i<2; i++) {
        sts = MFX_ERR_NONE;
        m_fd = open(adapterPaths[i], O_RDWR);

        if (m_fd < 0) sts = MFX_ERR_NOT_INITIALIZED;
        if (MFX_ERR_NONE == sts) {
            m_va_dpy = vaGetDisplayDRM(m_fd);
            if (!m_va_dpy) {
                close(m_fd);
                sts = MFX_ERR_NULL_PTR;
            }
        }

        if (MFX_ERR_NONE == sts) {
            va_res = vaInitialize(m_va_dpy, &major_version, &minor_version);
            sts = va_to_mfx_status(va_res);
            if (MFX_ERR_NONE != sts) {
                close(m_fd);
                m_fd = -1;
            }
        }

        if (MFX_ERR_NONE == sts) break;
    }

    for (int i=0; i<2; i++) {
        delete [] adapterPaths[i];
    }
    delete [] adapterPaths;

    if (MFX_ERR_NONE != sts) {
        throw std::invalid_argument("Loading of VA display was failed");
    }
}

DRMLibVA::~DRMLibVA(void) {
    if (m_va_dpy) {
        vaTerminate(m_va_dpy);
    }
    if (m_fd >= 0) {
        close(m_fd);
    }
}

#endif // #if defined(LIBVA_DRM_SUPPORT)






#if defined(LIBVA_X11_SUPPORT)

#include <X11/Xlib.h>


#define VAAPI_X_DEFAULT_DISPLAY ":0.0"

X11LibVA::X11LibVA(void) {
    VAStatus va_res = VA_STATUS_SUCCESS;
    mfxStatus sts = MFX_ERR_NONE;
    int major_version = 0, minor_version = 0;
    char *currentDisplay = getenv("DISPLAY");

    m_display = (currentDisplay) ? XOpenDisplay(currentDisplay) : XOpenDisplay(VAAPI_X_DEFAULT_DISPLAY);

    if (m_display == NULL) {
        throw std::bad_alloc();
    }
    m_va_dpy = vaGetDisplay(m_display);
    if (!m_va_dpy) {
        XCloseDisplay(m_display);
        throw std::bad_alloc();
    }
    va_res = vaInitialize(m_va_dpy, &major_version, &minor_version);
    sts = va_to_mfx_status(va_res);
    if (MFX_ERR_NONE != sts) {
        XCloseDisplay(m_display);
        throw std::bad_alloc();
    }
}

X11LibVA::~X11LibVA(void) {
    if (m_va_dpy) {
        vaTerminate(m_va_dpy);
    }
    if (m_display) {
        XCloseDisplay(m_display);
    }
}


#define VAAPI_GET_X_DISPLAY(_display) (Display*)(_display)
#define VAAPI_GET_X_WINDOW(_window) (Window*)(_window)

CQSVHWVADeviceX11::~CQSVHWVADeviceX11(void) {
    Close();
}

mfxStatus CQSVHWVADeviceX11::Init(mfxHDL hWindow, mfxU32 nAdapterNum, shared_ptr<CQSVLog> pQSVLog) {
    mfxStatus mfx_res = MFX_ERR_NONE;
    m_pQSVLog = pQSVLog;
    Window* window = NULL;
    return mfx_res;
}

void CQSVHWVADeviceX11::Close() {
    if (m_window) {
        Display* display = VAAPI_GET_X_DISPLAY(m_X11LibVA.GetXDisplay());
        Window* window = VAAPI_GET_X_WINDOW(m_window);
        XDestroyWindow(display, *window);

        free(m_window);
        m_window = NULL;
    }
    m_pQSVLog.reset();
}

mfxStatus CQSVHWVADeviceX11::Reset() {
    return MFX_ERR_NONE;
}

mfxStatus CQSVHWVADeviceX11::GetHandle(mfxHandleType type, mfxHDL *pHdl) {
    if ((MFX_HANDLE_VA_DISPLAY == type) && (nullptr != pHdl)) {
        *pHdl = m_X11LibVA.GetVADisplay();

        return MFX_ERR_NONE;
    }
    return MFX_ERR_UNSUPPORTED;
}

mfxStatus CVAAPIDeviceX11::SetHandle(mfxHandleType type, mfxHDL hdl) {
    return MFX_ERR_UNSUPPORTED;
}

mfxStatus CVAAPIDeviceX11::RenderFrame(mfxFrameSurface1 *pSurface, mfxFrameAllocator *pmfxAlloc) {
    return MFX_ERR_UNSUPPORTED;
}

#endif //#if defined(LIBVA_X11_SUPPORT)
