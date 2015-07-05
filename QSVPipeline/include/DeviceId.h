// Copyright 2010 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED ""AS IS.""
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.

//
// DeviceId.h
//
#pragma once

#define INITGUID
#include <windows.h>

// From DXUT.h
#ifndef SAFE_DELETE
#define SAFE_DELETE(p) { if (p) { delete (p); (p)=NULL; } }
#endif    
#ifndef SAFE_DELETE_ARRAY
#define SAFE_DELETE_ARRAY(p) { if (p) { delete[] (p); (p)=NULL; } }
#endif    
#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p) { if (p) { (p)->Release(); (p)=NULL; } }
#endif


// Define settings to reflect Fidelity abstraction levels you need
typedef enum
{
    NotCompatible,  // Found GPU is not compatible with the app
    Low,
    Medium,
    MediumPlus,
    High,
    Undefined  // No predefined setting found in cfg file. 
               // Use a default level for unknown video cards.
}
PresetLevel;

#define INTEL_VENDOR_ID 0x8086

// The new device dependent counter
#define INTEL_DEVICE_INFO_COUNTERS         "Intel Device Information"

typedef enum
{ 
    IGFX_UNKNOWN     = 0x0, 
    IGFX_SANDYBRIDGE = 0xC, 
    IGFX_IVYBRIDGE,    
    IGFX_HASWELL,
} PRODUCT_FAMILY;

// New device dependent structure
struct IntelDeviceInfoV1
{
    DWORD GPUMaxFreq;
    DWORD GPUMinFreq;
};

struct IntelDeviceInfoV2
{
    DWORD GPUMaxFreq;
    DWORD GPUMinFreq;
    DWORD GTGeneration;
    DWORD EUCount;
    DWORD PackageTDP;
    DWORD MaxFillRate;
};

struct IntelDeviceInfoHeader
{
    DWORD Size;
    DWORD Version;
};

/*****************************************************************************************
 * getGraphicsDeviceInfo
 *
 *     Function to get the primary graphics device's Vendor ID and Device ID, either 
 *     through the new DXGI interface or through the older D3D9 interfaces.
 *     The function also returns the amount of memory availble for graphics using 
 *     the value shared + dedicated video memory returned from DXGI, or, if the DXGI
 *       interface is not available, the amount of memory returned from WMI.
 *
 *****************************************************************************************/

bool getGraphicsDeviceInfo( unsigned int* VendorId,
                          unsigned int* DeviceId,
                          unsigned int* VideoMemory);


/*****************************************************************************************
 * setDefaultFidelityPresets
 *
 *     Function to find / set the default fidelity preset level, based on the type
 *     of graphics adapter present.
 *
 *     The guidelines for graphics preset levels for Intel devices is a generic one 
 *     based on our observations with various contemporary games. You would have to 
 *     change it if your game already plays well on the older hardware even at high
 *     settings.
 *
 *****************************************************************************************/

PresetLevel getDefaultFidelityPresets( unsigned int VendorId, unsigned int DeviceId );



/*****************************************************************************************
 * getIntelDeviceInfo
 *
 *     Returns the device info:
 *       GPU Max Frequency (Mhz)
 *       GPU Min Frequency (Mhz)
 *       GT Generation (enum)
 *       EU Count (unsigned int)
 *       Package TDP (Watts)
 *       Max Fill Rate (Pixel/Clk)
 * 
 * A return value of GGF_SUCCESS indicates 
 *       the frequency was returned correctly. 
 *     This function is only valid on Intel graphics devices SNB and later.
 *****************************************************************************************/

long getIntelDeviceInfo( unsigned int VendorId, IntelDeviceInfoHeader *pIntelDeviceInfoHeader, void *pIntelDeviceInfoBuffer );


/*****************************************************************************************
 * getVideoMemory
 *
 *     Returns the amount of dedicated video memory in the first device. The function 
 *     uses the Windows Management Instrumentation interfaces as is recommended in the
 *       Developer Guide.
 *
 *****************************************************************************************/

bool getVideoMemory( unsigned int* pVideoMemory );


/*****************************************************************************************
 * checkDxExtensionVersion
 *
 *      Returns the EXTENSION_INTERFACE_VERSION supported by the driver
 *      EXTENSION_INTERFACE_VERSION_1_0 supports extensions for pixel synchronization and
 *      and instant access of graphics memory
 *
 *****************************************************************************************/
unsigned int checkDxExtensionVersion();
#define GGF_SUCCESS 0
#define GGF_ERROR                    -1
#define GGF_E_UNSUPPORTED_HARDWARE    -2
#define GGF_E_UNSUPPORTED_DRIVER    -3
#define GGF_E_D3D_ERROR                -4