/* ****************************************************************************** *\

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2012-2013 Intel Corporation. All Rights Reserved.

\* ****************************************************************************** */

#if defined(LIBVA_DRM_SUPPORT)

#include "vaapi_utils_drm.h"
#include <fcntl.h>

#include <dirent.h>
#include <stdexcept>

#define MFX_PCI_DIR "/sys/bus/pci/devices"
#define MFX_DRI_DIR "/dev/dri/"
#define MFX_PCI_DISPLAY_CONTROLLER_CLASS 0x03

struct mfx_disp_adapters
{
    mfxU32 vendor_id;
    mfxU32 device_id;
};

static int mfx_dir_filter(const struct dirent* dir_ent)
{
    if (!dir_ent) return 0;
    if (!strcmp(dir_ent->d_name, ".")) return 0;
    if (!strcmp(dir_ent->d_name, "..")) return 0;
    return 1;
}

typedef int (*fsort)(const struct dirent**, const struct dirent**);

static mfxU32 mfx_init_adapters(struct mfx_disp_adapters** p_adapters)
{
    mfxU32 adapters_num = 0;
    int i = 0;
    struct mfx_disp_adapters* adapters = NULL;
    struct dirent** dir_entries = NULL;
    int entries_num = scandir(MFX_PCI_DIR, &dir_entries, mfx_dir_filter, (fsort)alphasort);

    char file_name[300] = {};
    char str[16] = {0};
    FILE* file = NULL;

    for (i = 0; i < entries_num; ++i)
    {
        long int class_id = 0, vendor_id = 0, device_id = 0;

        if (!dir_entries[i])
            continue;

        // obtaining device class id
        snprintf(file_name, sizeof(file_name)/sizeof(file_name[0]), "%s/%s/%s", MFX_PCI_DIR, dir_entries[i]->d_name, "class");
        file = fopen(file_name, "r");
        if (file)
        {
            if (fgets(str, sizeof(str), file))
            {
                class_id = strtol(str, NULL, 16);
            }
            fclose(file);

            if (MFX_PCI_DISPLAY_CONTROLLER_CLASS == (class_id >> 16))
            {
                // obtaining device vendor id
                snprintf(file_name, sizeof(file_name)/sizeof(file_name[0]), "%s/%s/%s", MFX_PCI_DIR, dir_entries[i]->d_name, "vendor");
                file = fopen(file_name, "r");
                if (file)
                {
                    if (fgets(str, sizeof(str), file))
                    {
                        vendor_id = strtol(str, NULL, 16);
                    }
                    fclose(file);
                }
                // obtaining device id
                snprintf(file_name, sizeof(file_name)/sizeof(file_name[0]), "%s/%s/%s", MFX_PCI_DIR, dir_entries[i]->d_name, "device");
                file = fopen(file_name, "r");
                if (file)
                {
                    if (fgets(str, sizeof(str), file))
                    {
                        device_id = strtol(str, NULL, 16);
                    }
                    fclose(file);
                }
                // adding valid adapter to the list
                if (vendor_id && device_id)
                {
                    struct mfx_disp_adapters* tmp_adapters = NULL;

                    tmp_adapters = (mfx_disp_adapters*)realloc(adapters,
                                                               (adapters_num+1)*sizeof(struct mfx_disp_adapters));

                    if (tmp_adapters)
                    {
                        adapters = tmp_adapters;
                        adapters[adapters_num].vendor_id = vendor_id;
                        adapters[adapters_num].device_id = device_id;

                        ++adapters_num;
                    }
                }
            }
        }
        free(dir_entries[i]);
    }
    if (entries_num) free(dir_entries);
    if (p_adapters) *p_adapters = adapters;

    return adapters_num;
}

DRMLibVA::DRMLibVA(void):
    m_fd(-1)
{
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
    while ((i < adapters_num) && (nFoundAdapters != numberOfRequiredIntelAdapter))
    {
        if (adapters[i].vendor_id == IntelVendorID)
        {
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
    for (int i=0; i<2; i++)
    {
        adapterPaths[i] = new char[sizeof(MFX_DRI_DIR) + sizeof(nodesNames[i]) + 3];
        sprintf(adapterPaths[i], "%s%s%d", MFX_DRI_DIR, nodesNames[i], nodesNumbers[i]);
    }

    // Loading display. At first trying to open render nodes, then card.
    for (int i=0; i<2; i++)
    {
        sts = MFX_ERR_NONE;
        m_fd = open(adapterPaths[i], O_RDWR);

        if (m_fd < 0) sts = MFX_ERR_NOT_INITIALIZED;
        if (MFX_ERR_NONE == sts)
        {
            m_va_dpy = vaGetDisplayDRM(m_fd);
            if (!m_va_dpy)
            {
                close(m_fd);
                sts = MFX_ERR_NULL_PTR;
            }
        }

        if (MFX_ERR_NONE == sts)
        {
            va_res = vaInitialize(m_va_dpy, &major_version, &minor_version);
            sts = va_to_mfx_status(va_res);
            if (MFX_ERR_NONE != sts)
            {
                close(m_fd);
                m_fd = -1;
            }
        }

        if (MFX_ERR_NONE == sts) break;
    }

    for (int i=0; i<2; i++)
    {
        delete [] adapterPaths[i];
    }
    delete [] adapterPaths;

    if (MFX_ERR_NONE != sts)
        throw std::invalid_argument("Loading of VA display was failed");
}

DRMLibVA::~DRMLibVA(void)
{
    if (m_va_dpy)
    {
        vaTerminate(m_va_dpy);
    }
    if (m_fd >= 0)
    {
        close(m_fd);
    }
}

#endif // #if defined(LIBVA_DRM_SUPPORT)
