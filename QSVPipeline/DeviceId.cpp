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
// DeviceId.cpp : Implements the GPU Device detection and graphics settings
//                configuration functions.
//
#if _MSC_VER
#include "DeviceId.h"
#include <stdio.h>
#include "rgy_tchar.h"

#include <D3D11.h>
#include <D3DCommon.h>
#include <DXGI.h>
#include <dxgi1_2.h>

#include <oleauto.h>
#include <initguid.h>
#include <wbemidl.h>
#include <ObjBase.h>

#pragma warning(disable:4456)
#pragma comment(lib, "wbemuuid.lib")

static const int FIRST_GFX_ADAPTER = 0;

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

/*****************************************************************************************
 * getDXGIAdapterDesc
 *
 *     Function to get the a DXGI_ADAPTER_DESC structure for a given device.
 *
 *****************************************************************************************/

#pragma warning(push)
#pragma warning(disable: 4100)
bool getDXGIAdapterDesc(DXGI_ADAPTER_DESC* AdapterDesc, unsigned int adapterNum = FIRST_GFX_ADAPTER)
{
    bool retVal = false;
    bool bHasWDDMDriver = false;
    if(AdapterDesc == NULL)
        return false;
    HMODULE hD3D9 = LoadLibrary( _T("d3d9.dll") );
    if( hD3D9 == NULL )
        return false;

    /*
        * Try to create IDirect3D9Ex interface (also known as a DX9L interface). 
        * This interface can only be created if the driver is a WDDM driver.
        */

    // Define a function pointer to the Direct3DCreate9Ex function.
    typedef HRESULT ( WINAPI*LPDIRECT3DCREATE9EX )( UINT,
                                                    void** );

    // Obtain the address of the Direct3DCreate9Ex function.
    LPDIRECT3DCREATE9EX pD3D9Create9Ex = NULL;
    pD3D9Create9Ex = ( LPDIRECT3DCREATE9EX )GetProcAddress( hD3D9, "Direct3DCreate9Ex" );

    bHasWDDMDriver = ( pD3D9Create9Ex != NULL );

    if( bHasWDDMDriver )
    {
        // Has WDDM Driver (Vista, and later)
        HMODULE hDXGI = NULL;

        hDXGI = LoadLibrary( _T("dxgi.dll") );

        // DXGI libs should really be present when WDDM driver present.
        if( hDXGI )
        {
            // Define a function pointer to the CreateDXGIFactory1 function.
            typedef HRESULT ( WINAPI*LPCREATEDXGIFACTORY )( REFIID riid,
                                                            void** ppFactory );

            // Obtain the address of the CreateDXGIFactory1 function.
            LPCREATEDXGIFACTORY pCreateDXGIFactory = NULL;
            pCreateDXGIFactory = ( LPCREATEDXGIFACTORY )GetProcAddress( hDXGI, "CreateDXGIFactory" );

            if( pCreateDXGIFactory )
            {
                // Got the function hook from the DLL
                // Create an IDXGIFactory object.
                IDXGIFactory* pFactory;
                if( SUCCEEDED( ( *pCreateDXGIFactory )( __uuidof( IDXGIFactory ), ( void** )( &pFactory ) ) ) )
                {
                    // Enumerate adapters. Code here only gets the info for the first adapter.
                    // If secondary or multiple Gfx adapters will be used, the code needs to be 
                    // modified to accomodate that.
                    IDXGIAdapter* pAdapter;
                    if( SUCCEEDED( pFactory->EnumAdapters(adapterNum, &pAdapter ) ) )
                    {
                        pAdapter->GetDesc( AdapterDesc );
                        pAdapter->Release();

                        retVal = true;
                    }
                }
            }

            FreeLibrary( hDXGI );
        }
    }
    FreeLibrary( hD3D9 );
    return retVal;
}
#pragma warning(pop)

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
                            unsigned int* VideoMemory,
                            const int adapterID
) {
    bool retVal = false;
    if( ( VendorId == NULL ) || ( DeviceId == NULL ) )
        return retVal;
    
    DXGI_ADAPTER_DESC AdapterDesc;
    if(getDXGIAdapterDesc(&AdapterDesc, adapterID))
    {
        *VendorId = AdapterDesc.VendorId;
        *DeviceId = AdapterDesc.DeviceId;
        *VideoMemory = (unsigned int)(AdapterDesc.DedicatedVideoMemory + AdapterDesc.SharedSystemMemory);
        retVal = true;
    }
    else
    {
   //     if(getDeviceIdD3D9(VendorId, DeviceId) && getVideoMemory(VideoMemory))
            //retVal = true;
        retVal = false;
    }
        
    return retVal;
}


/*****************************************************************************************
 * getDefaultFidelityPresets
 *
 *     Function to find the default fidelity preset level, based on the type of
 *     graphics adapter present.
 *
 *     The guidelines for graphics preset levels for Intel devices is a generic one 
 *     based on our observations with various contemporary games. You would have to 
 *     change it if your game already plays well on the older hardware even at high
 *     settings.
 *
 *****************************************************************************************/

PresetLevel getDefaultFidelityPresets( unsigned int VendorId, unsigned int DeviceId )
{
    PresetLevel presets = Undefined;

    //
    // Look for a config file that qualifies devices from any vendor
    // The code here looks for a file with one line per recognized graphics
    // device in the following format:
    //
    // VendorIDHex, DeviceIDHex, CapabilityEnum      ;Commented name of card
    //

    FILE *fp = NULL;
    const char *cfgFileName = "IntelGfx.cfg";

    switch( VendorId )
    {
    case 0x8086:
        fopen_s( &fp, cfgFileName, "r" );
        break;

        // Add cases to handle other graphics vendors. 
        // The following commented out code is an example case.
        //case 0x1002:
        //    fopen_s ( &fp, "ATI.cfg", "r" );
        //    break;

        //case 0x10DE:
        //    fopen_s ( &fp, "Nvidia.cfg", "r" );
        //    break;

    default:
        break;
    }


    if( fp )
    {
        char line[100];
        char* context = NULL;

        char* szVendorId = NULL;
        char* szDeviceId = NULL;
        char* szPresetLevel = NULL;

        while( fgets( line, 100, fp ) )   // read one line at a time till EOF
        {
            // Parse and remove the comment part of any line
            int i; for( i = 0; line[i] && line[i] != ';'; i++ ); line[i] = '\0';

            // Try to extract VendorId, DeviceId and recommended Default Preset Level
            szVendorId    = strtok_s( line, ",\n", &context );
            szDeviceId    = strtok_s( NULL, ",\n", &context );
            szPresetLevel = strtok_s( NULL, ",\n", &context );

            if( ( szVendorId == NULL ) ||
                ( szDeviceId == NULL ) ||
                ( szPresetLevel == NULL ) )
            {
                continue;  // blank or improper line in cfg file - skip to next line
            }

            unsigned int vId, dId;
            sscanf_s( szVendorId, "%x", &vId );
            sscanf_s( szDeviceId, "%x", &dId );

            // If current graphics device is found in the cfg file, use the 
            // pre-configured default Graphics Presets setting.
            if( ( vId == VendorId ) && ( dId == DeviceId ) )
            {
                // Found the device
                char s[10];
                sscanf_s( szPresetLevel, "%s", s, (int)_countof( s ) );

                if( !_stricmp( s, "Low" ) )
                    presets = Low;
                else if( !_stricmp( s, "Medium" ) )
                    presets = Medium;
                else if( !_stricmp( s, "Medium+" ) )
                    presets = MediumPlus;
                else if( !_stricmp( s, "High" ) )
                    presets = High;
                else
                    presets = NotCompatible;

                break; // Done reading file.
            }
        }

        fclose( fp );  // Close open file handle
    }
    else
    {
        printf("%s not found! Presets undefined.\n", cfgFileName);
    }

    // If the current graphics device was not listed in any of the config
    // files, or if config file not found, use Low settings as default.
    // This should be changed to reflect the desired behavior for unknown
    // graphics devices.
    if( presets == Undefined )
        presets = Low;

    return presets;
}

/*****************************************************************************************
 * getVideoMemory
 *
 *     Function to find the amount of video memory using the Windows Management Interface
 *       (WMI), the recommended method for Intel Processor Graphics.
 *
 *****************************************************************************************/

bool getVideoMemory( unsigned int* pVideoMemory )
{
    ULONG mem = 0;
    bool success = false;
    IWbemLocator* pLocator = NULL;
    HRESULT hr = S_OK;
    CoInitialize( 0 );
    hr = CoCreateInstance( CLSID_WbemLocator, NULL, CLSCTX_INPROC_SERVER, IID_IWbemLocator, ( LPVOID* )&pLocator );
    if( S_OK == hr )
    {
        if( pLocator != NULL )
        {
            IWbemServices* pServices = NULL;
            BSTR nameSpace = SysAllocString( L"\\\\.\\root\\cimv2" );
            if( S_OK == pLocator->ConnectServer( nameSpace, NULL, NULL, 0, 0, NULL, NULL, &pServices )
                && pServices != NULL )
            {
                CoSetProxyBlanket( pServices, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, NULL, RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE, NULL, 0 );
                IEnumWbemClassObject* pEnumVideo = NULL;
                if( S_OK  == pServices->CreateInstanceEnum( L"Win32_VideoController", 0, NULL, &pEnumVideo )
                    && pEnumVideo != NULL )
                {
                    IWbemClassObject* pVideo = NULL;
                    DWORD numReturned = 0;
                    if( S_OK == pEnumVideo->Next( 1000, 1, &pVideo, &numReturned ) && numReturned == 1 )
                    {
                        VARIANT v;
                        if( S_OK == pVideo->Get( L"AdapterRAM", 0, &v, NULL, NULL ) )
                        {
                            mem = v.uintVal;
                        }
                    }
                }
            }
            SysFreeString( nameSpace );
        }
        CoUninitialize();
        success = true;
    }
    *pVideoMemory = mem;
    return success;
}

/******************************************************************************************************************************************
 * getIntelDeviceInfo
 *
 * Description:
 *       Gets device info if available
 *           Supported device info: GPU Max Frequency, GPU Min Frequency, GT Generation, EU Count, Package TDP, Max Fill Rate
 * 
 * Parameters:
 *         unsigned int VendorId                         - [in]     - Input:  system's vendor id
 *         IntelDeviceInfoHeader *pIntelDeviceInfoHeader - [in/out] - Input:  allocated IntelDeviceInfoHeader *
 *                                                                    Output: Intel device info header, if found
 *         void *pIntelDeviceInfoBuffer                  - [in/out] - Input:  allocated void *
 *                                                                    Output: IntelDeviceInfoV[#], cast based on IntelDeviceInfoHeader
 * Return:
 *         GGF_SUCCESS: Able to find Data is valid
 *         GGF_E_UNSUPPORTED_HARDWARE: Unsupported hardware, data is invalid
 *         GGF_E_UNSUPPORTED_DRIVER: Unsupported driver on Intel, data is invalid
 *****************************************************************************************************************************************/

long getIntelDeviceInfo( unsigned int VendorId, const int adapterID, IntelDeviceInfoHeader *pIntelDeviceInfoHeader, void *pIntelDeviceInfoBuffer )
{
    // The device information is stored in a D3D counter.
    // We must create a D3D device, find the Intel counter 
    // and query the counter info
    ID3D11Device *pDevice = NULL;
    ID3D11DeviceContext *pImmediateContext = NULL;
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = NULL;

    if ( pIntelDeviceInfoBuffer == NULL )
        return GGF_ERROR;

    if ( VendorId != INTEL_VENDOR_ID )
        return GGF_E_UNSUPPORTED_HARDWARE;

    ZeroMemory( &featureLevel, sizeof(D3D_FEATURE_LEVEL) );

    IDXGIFactory2 *pDXGIFactory = nullptr;
    if (FAILED(hr = CreateDXGIFactory(__uuidof(IDXGIFactory2), (void**)(&pDXGIFactory)))) {
        return FALSE;
    }
    IDXGIAdapter *pAdapter = nullptr;
    if (FAILED(hr = pDXGIFactory->EnumAdapters(adapterID, &pAdapter))) {
        SAFE_RELEASE(pDXGIFactory);
        return FALSE;
    }
    SAFE_RELEASE(pDXGIFactory);

    static const D3D_FEATURE_LEVEL FeatureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0
    };
    // First create the Device, must be SandyBridge or later to create D3D11 device
    hr = D3D11CreateDevice( pAdapter, D3D_DRIVER_TYPE_UNKNOWN, NULL, 0, FeatureLevels, _countof(FeatureLevels),
                            D3D11_SDK_VERSION, &pDevice, &featureLevel, &pImmediateContext);

    if ( FAILED(hr) )
    {
        SAFE_RELEASE( pImmediateContext );
        SAFE_RELEASE(pAdapter);
        SAFE_RELEASE( pDevice );

        printf("D3D11CreateDevice failed\n");

        return FALSE;
    }
    
    // The counter is in a device dependent counter
    D3D11_COUNTER_INFO counterInfo;
    D3D11_COUNTER_DESC pIntelCounterDesc;
    
    int numDependentCounters;
    UINT uiSlotsRequired, uiNameLength, uiUnitsLength, uiDescLength;
    LPSTR sName, sUnits, sDesc;

    ZeroMemory( &counterInfo, sizeof(D3D11_COUNTER_INFO) );
    ZeroMemory( &pIntelCounterDesc, sizeof(D3D11_COUNTER_DESC) );

    // Query the device to find the number of device dependent counters.
    pDevice->CheckCounterInfo( &counterInfo );

    if ( counterInfo.LastDeviceDependentCounter == 0 )
    {
        SAFE_RELEASE( pImmediateContext );
        SAFE_RELEASE(pAdapter);
        SAFE_RELEASE( pDevice );

        printf("No device dependent counters\n");

        // The driver does not support the Device Info Counter.
        return GGF_E_UNSUPPORTED_DRIVER;
    }

    numDependentCounters = counterInfo.LastDeviceDependentCounter - D3D11_COUNTER_DEVICE_DEPENDENT_0 + 1;

    // Search for the apporpriate counter - INTEL_DEVICE_INFO_COUNTERS
    for ( int i = 0; i < numDependentCounters; ++i )
    {
        D3D11_COUNTER_DESC counterDescription;
        D3D11_COUNTER_TYPE counterType;

        counterDescription.Counter = static_cast<D3D11_COUNTER>(i+D3D11_COUNTER_DEVICE_DEPENDENT_0);
        counterDescription.MiscFlags = 0;
        counterType = static_cast<D3D11_COUNTER_TYPE>(0);
        uiSlotsRequired = uiNameLength = uiUnitsLength = uiDescLength = 0;
        sName = sUnits = sDesc = NULL;

        if( SUCCEEDED( hr = pDevice->CheckCounter( &counterDescription, &counterType, &uiSlotsRequired, NULL, &uiNameLength, NULL, &uiUnitsLength, NULL, &uiDescLength ) ) )
        {
            LPSTR sName  = new char[uiNameLength];
            LPSTR sUnits = new char[uiUnitsLength];
            LPSTR sDesc  = new char[uiDescLength];
            
            if( SUCCEEDED( hr = pDevice->CheckCounter( &counterDescription, &counterType, &uiSlotsRequired, sName, &uiNameLength, sUnits, &uiUnitsLength, sDesc, &uiDescLength ) ) )
            {
                if ( strcmp( sName, INTEL_DEVICE_INFO_COUNTERS ) == 0 )
                {
                    int IntelCounterMajorVersion;
                    int IntelCounterSize;
                    int argsFilled = 0;

                    pIntelCounterDesc.Counter = counterDescription.Counter;

                    argsFilled = sscanf_s( sDesc, "Version %d", &IntelCounterMajorVersion);
                    
                    if ( argsFilled != 1 || 1 != sscanf_s( sUnits, "Size %d", &IntelCounterSize))
                    {
                        // Fall back to version 1.0
                        IntelCounterMajorVersion = 1;
                        IntelCounterSize = sizeof( IntelDeviceInfoV1 );
                    }

                    pIntelDeviceInfoHeader->Version = IntelCounterMajorVersion;
                    pIntelDeviceInfoHeader->Size = IntelCounterSize;
                }
            }
            
            SAFE_DELETE_ARRAY( sName );
            SAFE_DELETE_ARRAY( sUnits );
            SAFE_DELETE_ARRAY( sDesc );
        }
    }

    // Check if device info counter was found
    if ( pIntelCounterDesc.Counter == NULL )
    {
        SAFE_RELEASE( pImmediateContext );
        SAFE_RELEASE(pAdapter);
        SAFE_RELEASE( pDevice );

        printf("Could not find counter\n");

        // The driver does not support the Device Info Counter.
        return GGF_E_UNSUPPORTED_DRIVER;
    }
    
    // Intel Device Counter //
    ID3D11Counter *pIntelCounter = NULL;

    // Create the appropriate counter
    hr = pDevice->CreateCounter(&pIntelCounterDesc, &pIntelCounter);
    if ( FAILED(hr) )
    {
        SAFE_RELEASE( pIntelCounter );
        SAFE_RELEASE( pImmediateContext );
        SAFE_RELEASE(pAdapter);
        SAFE_RELEASE( pDevice );

        printf("CreateCounter failed\n");

        return GGF_E_D3D_ERROR;
    }

    // Begin and end counter capture
    pImmediateContext->Begin(pIntelCounter);
    pImmediateContext->End(pIntelCounter);

    // Check for available data
    hr = pImmediateContext->GetData( pIntelCounter, NULL, NULL, NULL );
    if ( FAILED(hr) || hr == S_FALSE )
    {
        SAFE_RELEASE( pIntelCounter );
        SAFE_RELEASE( pImmediateContext );
        SAFE_RELEASE(pAdapter);
        SAFE_RELEASE( pDevice );

        printf("Getdata failed \n");
        return GGF_E_D3D_ERROR;
    }
    
    DWORD pData[2] = {0};
    // Get pointer to structure
    hr = pImmediateContext->GetData(pIntelCounter, pData, 2*sizeof(DWORD), NULL);

    if ( FAILED(hr) || hr == S_FALSE )
    {
        SAFE_RELEASE( pIntelCounter );
        SAFE_RELEASE( pImmediateContext );
        SAFE_RELEASE(pAdapter);
        SAFE_RELEASE( pDevice );

        printf("Getdata failed \n");
        return GGF_E_D3D_ERROR;
    }

    //
    // Prepare data to be returned //
    //
    // Copy data to passed in parameter
    void *pDeviceInfoBuffer = *(void**)pData;

    memcpy( pIntelDeviceInfoBuffer, pDeviceInfoBuffer, pIntelDeviceInfoHeader->Size );

    //
    // Clean up //
    //
    SAFE_RELEASE( pIntelCounter );
    SAFE_RELEASE( pImmediateContext );
    SAFE_RELEASE(pAdapter);
    SAFE_RELEASE( pDevice );

    return GGF_SUCCESS;
}

#include "ID3D10Extensions.h"
UINT checkDxExtensionVersion( )
{
    UINT extensionVersion = 0;
    ID3D10::CAPS_EXTENSION intelExtCaps;
    ID3D11Device *pDevice = NULL;
    ID3D11DeviceContext *pImmediateContext = NULL;
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = NULL;

    ZeroMemory( &featureLevel, sizeof(D3D_FEATURE_LEVEL) );

    // First create the Device
    hr = D3D11CreateDevice( NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, NULL, NULL, NULL,
                            D3D11_SDK_VERSION, &pDevice, &featureLevel, &pImmediateContext);

    if ( FAILED(hr) )
    {
        SAFE_RELEASE( pImmediateContext );
        SAFE_RELEASE( pDevice );

        printf("D3D11CreateDevice failed\n");
    }
    ZeroMemory( &intelExtCaps, sizeof(ID3D10::CAPS_EXTENSION) );

    if ( pDevice )
    {
        if( S_OK == GetExtensionCaps( pDevice, &intelExtCaps ) )
        {
            extensionVersion = intelExtCaps.DriverVersion;
        }
    }

    return extensionVersion;
}
#endif
